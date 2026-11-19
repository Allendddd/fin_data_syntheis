"""Bind native search RPCs to one configured backend without replacing tools.

Codex uses its standalone web.run RPC. Claude Code's built-in WebSearch
uses a dedicated Messages request. Strands calls the same backend through
its selected SDK search entry point. Model requests pass through unchanged.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.resources import ResourceTools, canonical_url


class SearchRoutingError(ValueError):
    pass


@dataclass
class SearchReply:
    body: bytes
    status: int = 200
    content_type: str = "application/json"


class NativeSearchRouter:
    def __init__(self, case, runtime, trace_dir, *, tools=None, lock=None):
        self.case, self.runtime, self.trace_dir = case, runtime, trace_dir
        self.settings = dict(case.retrieval.get("web", {}))
        if case.retrieval.get("mode") != "web" or not self.settings.get("search_endpoint"):
            raise ValueError("native search routing requires a configured web search backend")
        self.tools = tools or ResourceTools(case.retrieval, case.cutoff, entity_id=case.entity_id,
                                            dated_only=case.dated_only, archive=case.archive_path)
        self.owns_tools = tools is None
        self.lock = lock or threading.RLock()
        self.references = {}
        self.links = {}
        self.search_tool = runtime.strands.search_tool if runtime.name == "strands" else None
        self.record = {"format_version": 1, "runtime": runtime.name,
                       "configured_endpoint": self.settings["search_endpoint"],
                       "protocol": self.settings.get("search_protocol", "tavily"),
                       "native_tool": {"codex": "web.run", "claude_code": "WebSearch", "strands": self.search_tool}[runtime.name],
                       "query_field": runtime.strands.search_query_field if runtime.name == "strands" else "query",
                       "binding": "pending", "rpc_requests": 0, "failed_rpc_requests": 0}
        self._save()

    def close(self):
        if self.owns_tools:
            self.tools.close()

    def _save(self):
        atomic_json(self.trace_dir / "search-backend.json", self.record)

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def handle(self, method, target, body, *, call_id):
        """Return an RPC reply, or None to forward an unchanged model request."""
        if method != "POST":
            return None
        path = urlsplit(target).path.rstrip("/")
        if not path.endswith(("/responses", "/messages", "/chat/completions", "/alpha/search", "/mint/search")):
            return None
        try:
            request = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise SearchRoutingError("configured search backend requires readable native JSON requests") from exc
        if not isinstance(request, dict):
            raise SearchRoutingError("native search routing requires a JSON object")
        with self.lock:
            if self.runtime.name == "codex" and path.endswith("/responses"):
                tools = request.get("tools", [])
                standalone = any(tool.get("type") == "namespace" and tool.get("name") == "web"
                                 and any(child.get("name") == "run" for child in tool.get("tools", [])) for tool in tools)
                hosted = any(str(tool.get("type", "")).startswith("web_search") for tool in tools)
                if not standalone or hosted:
                    self.record.update(binding="unsupported", error="Codex did not expose standalone web.run")
                    self._save()
                    raise SearchRoutingError("search_endpoint cannot bind this Codex/model combination: standalone web.run is required; provider-hosted search will not be used as a fallback")
                self.record["binding"] = "bound"
                self._save()
                return None
            if self.runtime.name == "claude_code" and path.endswith("/messages"):
                hosted = [tool for tool in request.get("tools", []) if str(tool.get("type", "")).startswith("web_search_")]
                if hosted:
                    return self._execute("claude_web_search", request, call_id)
                if any(tool.get("name") == "WebSearch" for tool in request.get("tools", [])):
                    self.record["binding"] = "bound"
                    self._save()
                return None
            if self.runtime.name == "strands" and path.endswith("/chat/completions"):
                names = {tool.get("function", {}).get("name") for tool in request.get("tools", [])}
                if self.search_tool not in names:
                    raise SearchRoutingError(f"configured Strands search entry point {self.search_tool!r} is missing from its tools")
                self.record["binding"] = "bound"
                self._save()
                return None
            if self.runtime.name == "codex" and path.endswith("/alpha/search"):
                return self._execute("codex_web_run", request, call_id)
            if self.runtime.name == "strands" and path.endswith("/mint/search"):
                return self._execute("strands_search", request, call_id)
        return None

    def _execute(self, protocol, request, call_id):
        self.record["binding"] = "bound"
        self.record["rpc_requests"] += 1
        log = {"call_id": call_id, "protocol": protocol, "endpoint": self.settings["search_endpoint"],
               "request": request, "started_at": datetime.now(timezone.utc).isoformat()}
        self._queries = []
        try:
            if protocol == "codex_web_run":
                observation = self._commands(request.get("commands"))
                observation, text = self._render(observation, int(request.get("max_output_tokens") or 12000))
                result = {"output": text, "encrypted_output": None,
                          "results": [{"type": "text_result", "ref_id": item.get("chunk_id", item["url"]),
                                       "url": item["url"], "title": item.get("title", "")}
                                      for item in observation["results"]]}
            elif protocol == "claude_web_search":
                observation = self._claude_search(request)
                # This is the native tool's search-service response, not an LLM
                # completion. No generated reasoning or encryption is invented.
                observation, text = self._render(observation, int(request.get("max_tokens") or 12000))
                result = {"id": "search_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
                          "model": request.get("model"), "content": [{"type": "text", "text": text}],
                          "stop_reason": "end_turn", "stop_sequence": None,
                          "usage": {"input_tokens": 0, "output_tokens": 0}}
            else:
                observation = self._query(request)
                result = observation
            log.update(status="completed", observation=observation)
            if protocol == "claude_web_search" and request.get("stream"):
                frames = [{"type": "message_start", "message": {**result, "content": [], "stop_reason": None}},
                          {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                          {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                          {"type": "content_block_stop", "index": 0},
                          {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 0}},
                          {"type": "message_stop"}]
                wire = b"".join(b"event: " + frame["type"].encode() + b"\ndata: " + self._json(frame) + b"\n\n" for frame in frames)
                return SearchReply(wire, content_type="text/event-stream")
            return SearchReply(self._json(result))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            self.record["failed_rpc_requests"] += 1
            self.record["last_error"] = str(exc)
            log.update(status="failed", error=str(exc))
            return SearchReply(self._json({"error": {"type": "search_backend_error", "message": str(exc)}}), status=502)
        finally:
            log["queries"] = self._queries
            with (self.trace_dir / "search-routing.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log, ensure_ascii=False) + "\n")
            self._save()

    def _call(self, name, arguments):
        result = self.tools.call(name, arguments)
        if result.get("error"):
            raise ValueError("configured search backend: " + result["error"])
        return result

    def _query(self, arguments):
        self._queries.append(dict(arguments))
        result = self._call("search_web", arguments)
        rows = result.get("hits", [])
        for item in rows:
            if item.get("chunk_id"):
                self.references[item["chunk_id"]] = item["url"]
        return {"results": rows}

    @staticmethod
    def _render(observation, budget):
        # Use real source spans and public reference IDs. The conservative
        # character budget reduces opaque CLI truncation of whole JSON blobs.
        remaining, visible, blocks = max(256, budget), [], []
        rows = observation["results"]
        for index, source in enumerate(rows):
            header = "\n".join(filter(None, (source.get("chunk_id"), source.get("title"), source["url"])))
            share = min(remaining, max(512, remaining // (len(rows) - index)))
            room = share - len(header) - 4
            if source.get("links"):
                room -= min(512, max(0, room // 3))
            if room <= 0:
                break
            span = source.get("text", "")[:room]
            block = header + "\n" + span
            visible.append({**source, "text": span})
            for link in source.get("links", []):
                line = f"\nLink {link['id']}: {link['url']}"
                if len(block) + len(line) > share:
                    break
                block += line
            blocks.append(block)
            remaining -= len(block) + 2
        return {"results": visible}, "\n\n".join(blocks) or "No search results."

    def _claude_search(self, request):
        texts = [block.get("text", "") for message in request.get("messages", []) if message.get("role") == "user"
                 for block in (message.get("content") if isinstance(message.get("content"), list)
                               else [{"text": message.get("content", "")}]) if isinstance(block, dict)]
        matches = [re.fullmatch(r"Perform a web search for the query: (.+)", text, re.S) for text in texts]
        queries = [match.group(1) for match in matches if match]
        if len(queries) != 1 or len(texts) != 1:
            raise ValueError("this Claude Code search RPC format is unsupported; refusing to bypass the configured backend")
        definitions = [tool for tool in request.get("tools", []) if str(tool.get("type", "")).startswith("web_search_")]
        if len(definitions) != 1 or len(request.get("tools", [])) != 1:
            raise ValueError("a mixed provider-hosted tool request cannot be redirected as a Claude Code search RPC")
        arguments = {"query": queries[0]}
        for source, target in (("allowed_domains", "domains"), ("blocked_domains", "exclude_domains")):
            if definitions[0].get(source):
                arguments[target] = definitions[0][source]
        return self._query(arguments)

    def _commands(self, commands):
        if not isinstance(commands, dict):
            raise ValueError("Codex search RPC has no commands object")
        supported = {"search_query", "open", "find", "click", "response_length"}
        unsupported = [name for name, value in commands.items() if value and name not in supported]
        if unsupported:
            if self.settings.get("search_protocol", "tavily") == "json":
                # A general service can implement the full native command set.
                # Forward the batch intact instead of rewriting its intent or
                # maintaining a project-wide allowlist of tool operations.
                return self._query({"commands": commands})
            raise ValueError("configured search adapter does not implement native operations: " + ", ".join(unsupported))
        rows = []
        for query in commands.get("search_query", []):
            arguments = {"query": query["q"]}
            for field in ("domains", "recency"):
                if query.get(field) is not None:
                    arguments[field] = query[field]
            rows.extend(self._query(arguments)["results"])
        operations = [*commands.get("open", []), *commands.get("find", [])]
        for operation in commands.get("click", []):
            target = self.links.get(operation["ref_id"], {}).get(int(operation["id"]))
            if target is None:
                raise ValueError("click refers to a link absent from previously returned pages")
            operations.append({"ref_id": target})
        for operation in operations:
            ref = operation["ref_id"]
            url = self.references.get(ref, ref)
            result = self._call("read_web", {"url": canonical_url(url), "max_chars": self.case.retrieval.get("max_document_bytes", 33554432)})
            original_text = result["text"]
            if operation.get("lineno") is not None or operation.get("pattern"):
                lines = original_text.splitlines(keepends=True)
                line_number = int(operation.get("lineno", 0))
                if not 0 <= line_number < max(1, len(lines)):
                    raise ValueError("requested line is outside the returned document unit")
                start = sum(map(len, lines[:line_number]))
                if operation.get("pattern"):
                    found = original_text.find(operation["pattern"])
                    if found < 0:
                        rows.append({"url": url, "text": "Requested pattern was not found in this document unit."})
                        continue
                    start = max(0, found - 300)
                record = self.tools.sources[result["chunk_id"]]
                result = self.tools._publish(text=original_text[start:start + 12000],
                    locator={**result["locator"], "char_start": start}, sha256=result["sha256"],
                    metadata=record["metadata"], url=result["url"])
            if self.case.archive_path:
                blob = self.case.archive_path / (result["sha256"] + ".bin")
                if blob.is_file():
                    class Links(HTMLParser):
                        def __init__(self):
                            super().__init__()
                            self.targets = []
                        def handle_starttag(self, tag, attrs):
                            if tag == "a" and (href := dict(attrs).get("href")):
                                try:
                                    self.targets.append(canonical_url(urljoin(result["url"], href)))
                                except ValueError:
                                    pass
                    parser = Links()
                    parser.feed(blob.read_bytes().decode("utf-8", errors="replace"))
                    result["links"] = [{"id": index, "url": target} for index, target in enumerate(dict.fromkeys(parser.targets))]
                    self.links[result["chunk_id"]] = {link["id"]: link["url"] for link in result["links"]}
                    self.links[result["url"]] = self.links[result["chunk_id"]]
            if result.get("chunk_id"):
                self.references[result["chunk_id"]] = result["url"]
            rows.append(result)
        return {"results": rows}
