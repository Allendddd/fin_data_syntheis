"""Evidence from captured native tools, independent of tool or provider names.

Provider-managed searches may expose only URLs. A separately downloaded
verification snapshot is labelled as such; it is never added to the agent's
observations or exported as a model prediction.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import httpx

from mint_synth.core.provenance import json_fingerprint
from mint_synth.core.run_state import atomic_json
from mint_synth.harness.documents import HTMLText, document_units
from mint_synth.harness.resources import canonical_url
from mint_synth.harness.snapshot import published_date


def _url(value):
    try:
        return canonical_url(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _urls(value):
    if isinstance(value, dict):
        return list(dict.fromkeys(url for item in value.values() for url in _urls(item)))
    if isinstance(value, list):
        return list(dict.fromkeys(url for item in value for url in _urls(item)))
    if isinstance(value, str):
        return list(dict.fromkeys(url for raw in re.findall(r'https?://[^\s<>"\]]+', value)
                                  if (url := _url(raw.rstrip(".,);")))))
    return []


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_text(item) for item in value)))
    if isinstance(value, dict):
        return "\n".join(_text(value[key]) for key in ("text", "content", "body", "snippet", "raw_content", "cited_text", "output", "stdout", "observation")
                         if key in value)
    return ""


def _sources(value):
    if isinstance(value, str):
        try:
            yield from _sources(json.loads(value))
        except ValueError:
            # Native tools also return plain text and Markdown search results.
            # Attribute a span only when its URL has a distinct result line.
            matches = list(re.finditer(r'https?://[^\s<>"\]]+', value))
            starts = [value.rfind("\n", 0, match.start()) + 1 for match in matches]
            for index, match in enumerate(matches):
                url = _url(match.group().rstrip(".,);"))
                if not url or starts.count(starts[index]) != 1:
                    continue
                end = starts[index + 1] if index + 1 < len(starts) else len(value)
                span = value[starts[index]:end]
                if span.strip() == match.group():
                    span = ""
                yield url, span, {}
    elif isinstance(value, list):
        for item in value:
            yield from _sources(item)
    elif isinstance(value, dict):
        url = next((url for key in ("url", "source_url", "link") if (url := _url(value.get(key)))), None)
        if url:
            yield url, _text(value), value
        for item in value.values():
            if isinstance(item, (dict, list, str)):
                yield from _sources(item)


def _date(case, url, metadata):
    for key in ("published_at", "published_date", "publication_date", "date"):
        date = published_date(metadata, key)
        if date:
            return date, "tool_metadata"
    for fact in case.task.get("evidence", []):
        original = fact.get("source_metadata") or {}
        if url in {_url(original.get("url")), _url(original.get("original_url"))}:
            date = published_date(original) or published_date(fact, "available_at")
            if date:
                return date, "reference_source_metadata"
    return None, "unknown"


def _archived_snapshot(case, use, url):
    if not case.archive_path or not case.archive_path.is_dir():
        return None
    for path in case.archive_path.glob("evidence_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("url") != url or record.get("locator", {}).get("tool_use_id") != use["id"]
                or record.get("metadata", {}).get("evidence_visibility") != "verifier_snapshot"):
            continue
        blob = case.archive_path / (record["sha256"] + ".bin")
        if (not blob.is_file() or hashlib.sha256(blob.read_bytes()).hexdigest() != record["sha256"]
                or hashlib.sha256(record["text"].encode()).hexdigest() != record.get("text_sha256")):
            raise ValueError("archived native web verification snapshot failed its content hash check")
        return record
    return None


def bind_web_observation(case, trace_dir, use, output):
    """Archive returned source spans, or explicit hosted verification snapshots."""
    observed = {}
    candidates = list(_sources(output))
    # Plain tool output (WebFetch, curl, custom MCP, etc.) can carry the URL
    # in its arguments rather than repeat it in the result.
    if not candidates and not use.get("hosted"):
        urls = _urls(use.get("input", {}))
        if not urls:
            urls = _urls(output)
        if len(urls) == 1 and (text := _text(output)).strip():
            candidates.append((urls[0], text, {}))
    if use.get("hosted"):
        candidates.extend((url, "", {}) for url in _urls(output)
                          if url not in {item[0] for item in candidates})
    visible_urls = {url for url, text, _ in candidates if text.strip()}
    candidates = [(url, text, metadata) for url, text, metadata in candidates if text.strip() or url not in visible_urls]
    seen = set()
    for url, text, metadata in candidates:
        if (url, text) in seen:
            continue
        seen.add((url, text))
        visibility, blob = "tool_observation", None
        if not text.strip():
            if not use.get("hosted"):
                continue
            visibility = "verifier_snapshot"
            cached = _archived_snapshot(case, use, url)
            if cached:
                observed[cached["chunk_id"]] = cached
                continue
            try:
                limit = int(case.retrieval.get("max_document_bytes", 32 * 1024 * 1024))
                timeout = case.retrieval.get("web", {}).get("timeout_seconds", 45)
                with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                    response.raise_for_status()
                    content = bytearray()
                    for block in response.iter_bytes():
                        content.extend(block)
                        if len(content) > limit:
                            raise ValueError("verification snapshot exceeds document size limit")
                    blob = bytes(content)
                    media = response.headers.get("content-type", "").lower()
                    encoding = response.encoding or "utf-8"
                if "pdf" in media or blob.startswith(b"%PDF"):
                    text = "\n".join(unit["text"] for unit in document_units(blob, ".pdf"))
                elif "html" in media or b"<html" in blob[:1024].lower():
                    parser = HTMLText()
                    parser.feed(blob.decode(encoding, errors="replace"))
                    text = "".join(parser.parts)
                    if parser.published_at:
                        metadata = {**metadata, "published_at": parser.published_at}
                else:
                    text = blob.decode(encoding, errors="replace")
            except (httpx.HTTPError, OSError, ValueError) as exc:
                with (trace_dir / "web-evidence-errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"tool_use_id": use["id"], "url": url, "error": str(exc)}, ensure_ascii=False) + "\n")
                continue
        if not text.strip():
            continue
        date, date_origin = _date(case, url, metadata)
        digest = hashlib.sha256(blob if blob is not None else text.encode()).hexdigest()
        key = "evidence_" + json_fingerprint([url, digest, use["id"], visibility])[:24]
        record = {"chunk_id": key, "source_id": "source_" + hashlib.sha256(url.encode()).hexdigest()[:24],
                  "url": url, "available_at": date, "text": text, "sha256": digest, "document_id": None,
                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "locator": {"kind": "native_web", "tool": use["name"], "tool_use_id": use["id"]},
                  "metadata": {"url": url, "published_at": date, "date_provenance": date_origin,
                               "evidence_visibility": visibility,
                               "evidence_attribution": metadata.get("evidence_attribution", "tool_call"),
                               "tool_use_ids": metadata.get("tool_use_ids", [use["id"]])},
                  "acquired_at": datetime.now(timezone.utc).isoformat()}
        observed[key] = record
        if case.archive_path:
            case.archive_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_json(case.archive_path / (key + ".json"), record)
            if blob is not None:
                (case.archive_path / (digest + ".bin")).write_bytes(blob)
    return observed
