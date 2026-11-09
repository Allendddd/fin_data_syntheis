"""Loopback HTTP proxy that records model API calls without changing their payloads.

Run one proxy per agent run. The agent's API base URL points at this proxy and
the proxy's upstream points at the vLLM/SGLang compatible endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx

from mint_synth.core.run_state import atomic_json, read_json


_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_SECRET_NAME = re.compile(r"(?:authorization|api[-_]?key|cookie|token|secret|credential|password)", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("upstream must be an HTTP(S) origin without a path, query, or credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("upstream has an invalid port") from exc
    return value.rstrip("/")


def _safe_headers(headers: list[tuple[str, str]]) -> list[list[str]]:
    return [[name, "[REDACTED]" if _SECRET_NAME.search(name) else value] for name, value in headers]


def _safe_target(target: str) -> str:
    parts = urlsplit(target)
    if not parts.query:
        return parts.path
    query = urlencode([
        (name, "[REDACTED]" if _SECRET_NAME.search(name) else value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ])
    return f"{parts.path}?{query}"


def _forward_headers(headers: list[tuple[str, str]], *, response: bool = False) -> list[tuple[str, str]]:
    connection_tokens = {
        item.strip().lower()
        for name, value in headers if name.lower() == "connection"
        for item in value.split(",")
    }
    excluded = _HOP_HEADERS | connection_tokens | {"content-length"}
    if not response:
        excluded = excluded | {"host", "expect"}
    return [(name, value) for name, value in headers if name.lower() not in excluded]


class _BadRequest(Exception):
    pass


class _BodyTooLarge(Exception):
    pass


@dataclass(frozen=True)
class TraceProxyConfig:
    upstream: str
    output_dir: Path
    run_id: str
    port: int = 8765
    max_request_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream", _origin(self.upstream))
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id must be nonempty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")


class TraceProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: TraceProxyConfig):
        self.config = config
        self.calls_dir = config.output_dir / "calls"
        config.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.calls_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            config.output_dir.chmod(0o700)
            self.calls_dir.chmod(0o700)
        manifest_path = config.output_dir / "run.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous.get("run_id") != config.run_id or previous.get("upstream") != config.upstream:
                raise ValueError("output_dir already belongs to a different trace run")
        else:
            atomic_json(manifest_path, {
                "format_version": 1,
                "run_id": config.run_id,
                "upstream": config.upstream,
                "started_at": _now(),
            })
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
            trust_env=False,
            follow_redirects=False,
        )
        try:
            super().__init__(("127.0.0.1", config.port), TraceProxyHandler)
        except BaseException:
            self.client.close()
            raise
        upstream = urlsplit(config.upstream)
        upstream_port = upstream.port or (443 if upstream.scheme == "https" else 80)
        if upstream.hostname in {"localhost", "127.0.0.1"} and upstream_port == self.server_port:
            self.server_close()
            raise ValueError("upstream points back to the proxy listener")

    def server_close(self) -> None:
        super().server_close()
        self.client.close()


class TraceProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: TraceProxyServer

    def log_message(self, _format: str, *_args: object) -> None:
        # Request URLs and headers can carry credentials; the trace files are private.
        pass

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy()

    def _send_error_json(self, status: int, message: str, call_id: str | None = None) -> None:
        body = json.dumps({"error": {"type": "trace_proxy_error", "message": message}}).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if call_id:
            self.send_header("X-Mint-Trace-Call-Id", call_id)
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def _read_body(self, output: Path) -> tuple[bytes, str]:
        digest = hashlib.sha256()
        body = bytearray()
        transfer = self.headers.get("Transfer-Encoding", "").lower()
        length = self.headers.get("Content-Length")
        if transfer and transfer != "chunked":
            raise _BadRequest("unsupported request transfer encoding")
        if transfer and length is not None:
            raise _BadRequest("request cannot have both transfer encoding and content length")

        def append(chunk: bytes, stream: object) -> None:
            if len(body) + len(chunk) > self.server.config.max_request_bytes:
                raise _BodyTooLarge()
            body.extend(chunk)
            digest.update(chunk)
            stream.write(chunk)  # type: ignore[attr-defined]

        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            if transfer == "chunked":
                while True:
                    header = self.rfile.readline(8193)
                    if not header.endswith(b"\r\n") or len(header) > 8192:
                        raise _BadRequest("malformed request chunk")
                    try:
                        size = int(header[:-2].split(b";", 1)[0], 16)
                    except ValueError as exc:
                        raise _BadRequest("malformed request chunk size") from exc
                    if size == 0:
                        trailer_bytes = 0
                        while True:
                            trailer = self.rfile.readline(8193)
                            trailer_bytes += len(trailer)
                            if not trailer.endswith(b"\r\n") or trailer_bytes > 65536:
                                raise _BadRequest("malformed request trailer")
                            if trailer == b"\r\n":
                                break
                        break
                    if size + len(body) > self.server.config.max_request_bytes:
                        raise _BodyTooLarge()
                    chunk = self.rfile.read(size)
                    if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                        raise _BadRequest("incomplete request chunk")
                    append(chunk, stream)
            elif length is not None:
                try:
                    remaining = int(length)
                except ValueError as exc:
                    raise _BadRequest("invalid content length") from exc
                if remaining < 0:
                    raise _BadRequest("invalid content length")
                if remaining > self.server.config.max_request_bytes:
                    raise _BodyTooLarge()
                while remaining:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        raise _BadRequest("incomplete request body")
                    append(chunk, stream)
                    remaining -= len(chunk)
        return bytes(body), digest.hexdigest()

    def _proxy(self) -> None:
        target = self.path
        parsed = urlsplit(target)
        if not target.startswith("/") or target.startswith("//") or parsed.scheme or parsed.netloc or "\\" in target:
            self._send_error_json(400, "request target must be a relative path")
            return

        call_id = uuid.uuid4().hex
        call_root = self.server.calls_dir / call_id
        request_path = call_root.with_suffix(".request.bin")
        response_path = call_root.with_suffix(".response.bin")
        chunks_path = call_root.with_suffix(".chunks.jsonl")
        metadata_path = call_root.with_suffix(".json")
        request_headers = list(self.headers.raw_items())
        started = time.monotonic_ns()
        metadata: dict[str, object] = {
            "format_version": 1,
            "run_id": self.server.config.run_id,
            "call_id": call_id,
            "started_at": _now(),
            "state": "receiving_request",
            "request": {
                "method": self.command,
                "target": _safe_target(target),
                "headers": _safe_headers(request_headers),
                "body_file": request_path.name,
            },
        }
        atomic_json(metadata_path, metadata)
        try:
            body, body_sha = self._read_body(request_path)
        except (_BadRequest, _BodyTooLarge) as exc:
            metadata["state"] = "request_rejected"
            metadata["finished_at"] = _now()
            atomic_json(metadata_path, metadata)
            status = 413 if isinstance(exc, _BodyTooLarge) else 400
            self._send_error_json(status, "request body too large" if status == 413 else str(exc), call_id)
            return
        request_data = metadata["request"]
        assert isinstance(request_data, dict)
        request_data["body_bytes"] = len(body)
        request_data["body_sha256"] = body_sha
        forwarded = _forward_headers(request_headers)
        request_id = self.headers.get("X-Request-Id") or call_id
        if not self.headers.get("X-Request-Id"):
            forwarded.append(("X-Request-Id", request_id))
        request_data["request_id"] = request_id
        metadata["state"] = "awaiting_upstream"
        atomic_json(metadata_path, metadata)

        response_started = False
        response_bytes = 0
        response_sha = hashlib.sha256()
        try:
            with self.server.client.stream(
                self.command,
                f"{self.server.config.upstream}{target}",
                headers=forwarded,
                content=body,
            ) as upstream:
                upstream_headers = list(upstream.headers.multi_items())
                has_body = self.command != "HEAD" and upstream.status_code not in {204, 304} and not 100 <= upstream.status_code < 200
                metadata["response"] = {
                    "status": upstream.status_code,
                    "headers": _safe_headers(upstream_headers),
                    "body_file": response_path.name,
                    "chunks_file": chunks_path.name,
                    "upstream_request_id": upstream.headers.get("X-Request-Id"),
                }
                metadata["state"] = "streaming_response"
                atomic_json(metadata_path, metadata)
                self.send_response_only(upstream.status_code)
                for name, value in _forward_headers(upstream_headers, response=True):
                    self.send_header(name, value)
                self.send_header("X-Mint-Trace-Call-Id", call_id)
                if has_body:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                response_started = True
                descriptor = os.open(response_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                chunk_descriptor = os.open(chunks_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as response_file, os.fdopen(chunk_descriptor, "w", encoding="utf-8") as chunks_file:
                    if has_body:
                        for chunk in upstream.iter_raw():
                            if not chunk:
                                continue
                            offset = response_bytes
                            response_file.write(chunk)
                            response_file.flush()
                            response_sha.update(chunk)
                            response_bytes += len(chunk)
                            chunks_file.write(json.dumps({
                                "offset_ms": round((time.monotonic_ns() - started) / 1_000_000, 3),
                                "byte_offset": offset,
                                "byte_length": len(chunk),
                            }) + "\n")
                            chunks_file.flush()
                            self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                response_data = metadata["response"]
                assert isinstance(response_data, dict)
                response_data["body_bytes"] = response_bytes
                response_data["body_sha256"] = response_sha.hexdigest()
                response_data["complete"] = True
                metadata["state"] = "complete"
        except (BrokenPipeError, ConnectionResetError):
            metadata["state"] = "client_disconnected"
            self.close_connection = True
        except (httpx.HTTPError, OSError) as exc:
            metadata["state"] = "response_interrupted" if response_started else "upstream_error"
            metadata["error_type"] = type(exc).__name__
            self.close_connection = True
            if not response_started:
                self._send_error_json(502, "upstream request failed", call_id)
        finally:
            response_data = metadata.get("response")
            if isinstance(response_data, dict) and "body_bytes" not in response_data:
                response_data["body_bytes"] = response_bytes
                response_data["body_sha256"] = response_sha.hexdigest()
                response_data["complete"] = False
            metadata["finished_at"] = _now()
            metadata["elapsed_ms"] = round((time.monotonic_ns() - started) / 1_000_000, 3)
            atomic_json(metadata_path, metadata)


def serve_trace_proxy(config: TraceProxyConfig) -> None:
    server = TraceProxyServer(config)
    print(f"trace proxy listening on http://127.0.0.1:{server.server_port}; run_id={config.run_id}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
