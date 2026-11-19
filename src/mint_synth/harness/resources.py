"""Mode-exclusive original-file/web tools and evaluator-side acquisition records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sqlite3
from urllib.parse import urlsplit, urlunsplit

import httpx

from mint_synth.core.provenance import json_fingerprint, sha256_file
from mint_synth.core.run_state import atomic_json
from .documents import SUPPORTED_SUFFIXES, HTMLText, document_units
from .snapshot import SnapshotTools, published_date


def canonical_url(url):
    parsed = urlsplit(str(url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("source URL must be HTTP(S) without embedded credentials")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


def validate_retrieval(settings):
    mode = settings.get("mode")
    if mode == "static":
        path = settings.get("corpus_path")
        if not path or not Path(path).is_dir():
            raise ValueError("static retrieval requires an existing harness.retrieval.corpus_path directory")
        if settings.get("manifest_path") and not Path(settings["manifest_path"]).is_file():
            raise ValueError("original-document manifest is missing")
    elif mode == "web":
        web = settings.get("web", {})
        from .search_backend import validate_search_backend
        validate_search_backend(web)
    else:
        raise ValueError("retrieval.mode must be static or web")


def resource_fingerprint(settings):
    material = {"version": 1, "settings": dict(settings)}
    if settings["mode"] == "static":
        root = Path(settings["corpus_path"]).resolve()
        material["files"] = [(p.relative_to(root).as_posix(), sha256_file(p)) for p in sorted(root.rglob("*"))
                             if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED_SUFFIXES
                             and (not settings.get("manifest_path") or p.resolve() != Path(settings["manifest_path"]).resolve())]
        if settings.get("manifest_path"):
            material["manifest_sha256"] = sha256_file(settings["manifest_path"])
    return json_fingerprint(material)


class ResourceTools:
    """Same public API and acquisition semantics for every execution framework."""
    def __init__(self, settings, cutoff, *, entity_id, dated_only=False, archive=None):
        validate_retrieval(settings)
        self.settings, self.cutoff, self.entity_id, self.dated_only = dict(settings), cutoff, entity_id, dated_only
        self.mode = settings["mode"]
        self.archive = Path(archive) if archive else None
        self.seen_search, self.opened, self.sources, self.documents, self.urls = set(), set(), {}, {}, {}
        self.replay_history = []
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,source_id TEXT,available_at TEXT,text TEXT)")
        self.db.execute("CREATE TABLE source_entities(source_id TEXT,canonical_entity_id TEXT, UNIQUE(source_id,canonical_entity_id))")
        self.http = httpx.Client(timeout=float(settings.get("web", {}).get("timeout_seconds", 45)), follow_redirects=True)
        if self.mode == "static":
            root = Path(settings["corpus_path"]).resolve()
            metadata = {}
            if settings.get("manifest_path"):
                for line in Path(settings["manifest_path"]).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        metadata[str(row["file"])] = row
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if settings.get("manifest_path") and path.resolve() == Path(settings["manifest_path"]).resolve():
                    continue
                if not path.resolve().is_relative_to(root):
                    raise ValueError("corpus file escapes the configured directory")
                relative, digest = path.relative_to(root).as_posix(), sha256_file(path)
                meta = {**metadata.get(relative, {}), "original_path": str(path.resolve())}
                if meta.get("sha256") and meta["sha256"] != digest:
                    raise ValueError(f"original document failed manifest hash check: {relative}")
                key = "doc_" + json_fingerprint([relative, digest])[:24]
                self.documents[key] = {"path": path, "file": relative, "sha256": digest, "metadata": meta}
            if not self.documents:
                raise ValueError("original corpus contains no supported documents")
        identity = {"version": 1, "settings": self.settings}
        if self.mode == "static":
            identity["files"] = [(doc["file"], doc["sha256"]) for doc in self.documents.values()]
            if settings.get("manifest_path"):
                identity["manifest_sha256"] = sha256_file(settings["manifest_path"])
        self.fingerprint = json_fingerprint(identity)

    def close(self):
        self.http.close()
        self.db.close()

    def _date_ok(self, date):
        return (date is None or date <= self.cutoff) and (not self.dated_only or date is not None)

    def _units(self, document):
        if sha256_file(document["path"]) != document["sha256"]:
            raise ValueError("original document changed during collection")
        if "units" not in document:
            data = document["path"].read_bytes()
            if hashlib.sha256(data).hexdigest() != document["sha256"]:
                raise ValueError("original document changed while reading")
            document["units"] = document_units(data, document["path"].suffix.lower())
        return document["units"]

    def _publish(self, *, text, locator, sha256, metadata, document_id=None, url=None):
        if not text.strip():
            raise ValueError("requested source span is empty")
        if "char_start" in locator:
            locator = {**locator, "char_end": locator["char_start"] + len(text)}
        date = published_date(metadata, "published_at")
        if not self._date_ok(date):
            raise ValueError("source is undated for a time-sensitive task or after the task cutoff")
        source_id = "source_" + sha256[:24]
        key = "evidence_" + json_fingerprint([sha256, locator, text])[:24]
        result = {"chunk_id": key, "source_id": source_id, "available_at": date, "text": text,
                  "locator": locator, "sha256": sha256, "document_id": document_id, "url": url}
        self.sources[key] = {**result, "metadata": metadata, "acquired_at": datetime.now(timezone.utc).isoformat()}
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?)", (key, source_id, date, text))
        self.db.execute("INSERT OR IGNORE INTO source_entities VALUES(?,?)", (source_id, self.entity_id))
        self.opened.add(key)
        if self.archive:
            self.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
            if document_id is not None:
                blob = self.archive / (sha256 + ".bin")
                if not blob.exists():
                    data = self.documents[document_id]["path"].read_bytes()
                    if hashlib.sha256(data).hexdigest() != sha256:
                        raise ValueError("original file changed before evidence archival")
                    blob.write_bytes(data)
            atomic_json(self.archive / f"{key}.json", self.sources[key])
        return result

    def call(self, name, arguments):
        if self.replay_history:
            previous = self.replay_history.pop(0)
            if previous["action"].get("tool") != name or previous["action"].get("arguments", {}) != arguments:
                raise ValueError("checkpoint action differs from archived acquisition")
            observation = previous["observation"]
            path = self.archive / "calls.jsonl" if self.archive else None
            if path is None or not path.is_file():
                raise ValueError("resource checkpoint replay requires its original acquisition archive")
            expected = {"name": name, "arguments": arguments, "observation": observation}
            if not any(json.loads(line) == expected for line in path.read_text(encoding="utf-8").splitlines()):
                raise ValueError("checkpoint observation is not in the actual acquisition archive")
            for item in source_observations(observation):
                record_path = self.archive / (item["chunk_id"] + ".json")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if any(record.get(key) != value for key, value in item.items() if key not in {"unit", "units", "title"}):
                    raise ValueError("archived evidence disagrees with checkpoint observation")
                if not self._date_ok(record["available_at"]):
                    raise ValueError("archived source crosses the task cutoff")
                self.sources[item["chunk_id"]] = record
                self.db.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?)", (item["chunk_id"], item["source_id"], item["available_at"], item["text"]))
                self.db.execute("INSERT OR IGNORE INTO source_entities VALUES(?,?)", (item["source_id"], self.entity_id))
                self.opened.add(item["chunk_id"])
            return observation
        observation = self._call(name, arguments)
        if self.archive:
            self.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (self.archive / "calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"name": name, "arguments": arguments, "observation": observation}, ensure_ascii=False) + "\n")
        return observation

    def _call(self, name, arguments):
        if name == "calculate":
            return SnapshotTools.call(self, name, arguments)
        try:
            if self.mode == "static" and name == "search_documents":
                query = str(arguments["query"]).strip()
                terms = [x.casefold() for x in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", query)]
                terms = list(dict.fromkeys(term for token in terms for term in
                    ([token] if token.isascii() or len(token) < 3 else [token[i:i + 2] for i in range(len(token) - 1)])))[:32]
                hits, diagnostics = [], []
                for doc_id, doc in self.documents.items():
                    if not self._date_ok(published_date(doc["metadata"], "published_at")):
                        continue
                    try:
                        for number, unit in enumerate(self._units(doc), 1):
                            folded = unit["text"].casefold()
                            score = sum(term in folded for term in terms)
                            if score:
                                offset = min((folded.find(t) for t in terms if t in folded), default=0)
                                text = unit["text"][max(0, offset - 100):offset + 500]
                                hits.append((score, doc_id, number, text, {**unit["locator"], "char_start": max(0, offset - 100)}))
                    except (ValueError, OSError) as exc:
                        diagnostics.append({"file": doc["file"], "error": str(exc)})
                results = []
                for _, doc_id, number, text, locator in sorted(hits, key=lambda x: (-x[0], x[1], x[2]))[:int(arguments.get("limit", 5))]:
                    doc = self.documents[doc_id]
                    result = self._publish(text=text, locator={**locator, "file": doc["file"]}, sha256=doc["sha256"],
                                           metadata=doc["metadata"], document_id=doc_id)
                    results.append({**result, "unit": number, "units": len(self._units(doc))})
                return {"hits": results, "diagnostics": diagnostics}
            if self.mode == "static" and name == "read_document":
                doc_id = str(arguments["document_id"])
                doc = self.documents[doc_id]
                units = self._units(doc)
                number = int(arguments.get("unit", 1))
                if not 1 <= number <= len(units):
                    raise ValueError("document unit is outside the original document")
                start = int(arguments.get("char_start", 0))
                text = units[number - 1]["text"][start:start + int(arguments.get("max_chars", 12000))]
                if not text.strip():
                    raise ValueError("requested document span has no text")
                return {**self._publish(text=text, locator={**units[number - 1]["locator"], "file": doc["file"], "char_start": start},
                                       sha256=doc["sha256"], metadata=doc["metadata"], document_id=doc_id), "units": len(units)}
            if self.mode == "web" and name == "search_web":
                web = self.settings.get("web", {})
                endpoint = web.get("search_endpoint")
                if not endpoint:
                    return {"error": "optional search adapter is not configured; use the harness's native tools"}
                from .search_backend import search
                payload = search(self.http, web, arguments, archive=self.archive)
                if self.archive:
                    self.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
                    atomic_json(self.archive / ("search_" + json_fingerprint([arguments, payload]) + ".json"),
                                {"query": arguments, "response": payload, "acquired_at": datetime.now(timezone.utc).isoformat()})
                hits = []
                for hit in payload.get("results", [])[:int(arguments.get("limit", 5))]:
                    url = canonical_url(hit["url"])
                    meta = {"published_at": hit.get("published_at"), "url": url,
                            "backend_request_id": payload["backend_request_id"]}
                    self.urls[url] = meta
                    text = str(hit.get("content") or "")[:3000]
                    if text and self._date_ok(published_date(meta, "published_at")):
                        result = self._publish(text=text, locator={"kind": "search_excerpt", "url": url},
                            sha256=hashlib.sha256(text.encode()).hexdigest(), metadata=meta, url=url)
                        hits.append({**result, "title": hit.get("title", "")})
                    else:
                        hits.append({"url": url, "title": hit.get("title", ""), "evidence_status": "date_not_verified"})
                return {"hits": hits}
            if self.mode == "web" and name == "read_web":
                url = canonical_url(arguments["url"])
                limit = int(self.settings.get("max_document_bytes", 32 * 1024 * 1024))
                with self.http.stream("GET", url) as response:
                    response.raise_for_status()
                    data = bytearray()
                    for part in response.iter_bytes():
                        data.extend(part)
                        if len(data) > limit:
                            raise ValueError("web document exceeds configured byte limit")
                    final_url = canonical_url(str(response.url))
                    content_type = response.headers.get("content-type", "")
                data = bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                suffix = ".pdf" if data.startswith(b"%PDF") else ".html" if "html" in content_type else Path(urlsplit(final_url).path).suffix.lower()
                units = document_units(data, suffix)
                meta = dict(self.urls.get(url, {}))
                if suffix == ".html":
                    parser = HTMLText()
                    parser.feed(data.decode("utf-8", errors="replace"))
                    if parser.published_at:
                        meta["published_at"] = parser.published_at
                meta["url"] = final_url
                if self.archive:
                    self.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
                    (self.archive / (digest + ".bin")).write_bytes(data)
                number = int(arguments.get("unit", 1))
                if not 1 <= number <= len(units):
                    raise ValueError("web document unit is out of range")
                start = int(arguments.get("char_start", 0))
                text = units[number - 1]["text"][start:start + int(arguments.get("max_chars", 12000))]
                return {**self._publish(text=text, locator={**units[number - 1]["locator"], "url": final_url, "char_start": start},
                                       sha256=digest, metadata=meta, url=final_url), "units": len(units)}
            return {"error": f"tool {name!r} is unavailable in {self.mode} retrieval mode"}
        except (ValueError, KeyError, IndexError, OSError, httpx.HTTPError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def source_observations(observation):
    if not isinstance(observation, dict) or "error" in observation:
        return []
    return [item for item in [observation, *observation.get("hits", [])]
            if isinstance(item, dict) and item.get("chunk_id") and isinstance(item.get("text"), str)]
