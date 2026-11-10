"""Private loopback MCP endpoint for one financial synthesis episode.

This server implements the stateless JSON response form of Streamable HTTP.
It exposes only the bridge's public tools; grading and the hidden task never
have an MCP method. The bridge owns episode state and serializes tool calls.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol


_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_REJECT_DRAIN_BYTES = 64 * 1024
_REJECT_DRAIN_TIMEOUT_SECONDS = 1.0


class ToolBridge(Protocol):
    @property
    def tool_definitions(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class FinanceMCPServer(ThreadingHTTPServer):
    """Serve one bridge using an unguessable bearer token on 127.0.0.1."""

    daemon_threads = True

    def __init__(self, bridge: ToolBridge):
        self.bridge = bridge
        self.token = secrets.token_urlsafe(32)
        self._thread: threading.Thread | None = None
        self._tools: dict[str, dict[str, Any]] = {}
        for definition in bridge.tool_definitions:
            function = definition.get("function", {})
            name = function.get("name")
            if not isinstance(name, str) or not name or name in self._tools:
                raise ValueError("bridge has an invalid or duplicate tool name")
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError("bridge tool needs an object input schema")
            self._tools[name] = json.loads(json.dumps({
                "name": name,
                "description": str(function.get("description", "")),
                "inputSchema": parameters,
            }, ensure_ascii=False, allow_nan=False))
        super().__init__(("127.0.0.1", 0), _MCPHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/mcp"

    def __enter__(self) -> FinanceMCPServer:
        if self._thread is not None:
            raise RuntimeError("MCP server is already running")
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self._thread is not None:
            self.shutdown()
            self._thread.join(timeout=5)
        self.server_close()


class _MCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FinanceMCPServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, _format: str, *_args: Any) -> None:
        # Tool arguments and the bearer token must not enter an access log.
        pass

    def _reply(self, status: int, value: Any = None, *, close: bool = False) -> None:
        if close:
            self._drain_rejected_body()
        body = b"" if value is None else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response_only(status)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        if status == 405:
            self.send_header("Allow", "POST")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _drain_rejected_body(self) -> None:
        """Consume a small declared body before closing a rejected POST.

        Closing with unread request bytes can reset the TCP connection before
        the client receives the HTTP error, especially on Windows. Large,
        ambiguous, or chunked bodies are never drained. A total deadline also
        bounds clients that declare a body and then stop sending it.
        """
        if self.command != "POST" or getattr(self, "_request_body_consumed", False):
            return
        self._request_body_consumed = True
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            return
        try:
            remaining = int(lengths[0])
        except ValueError:
            return
        if not 0 < remaining <= _MAX_REJECT_DRAIN_BYTES:
            return
        deadline = time.monotonic() + _REJECT_DRAIN_TIMEOUT_SECONDS
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(remaining, 8192))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass
        finally:
            self.connection.settimeout(previous_timeout)

    def _error(self, code: int, message: str, request_id: Any = None, *, status: int = 200, close: bool = False) -> None:
        self._reply(status, {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message},
        }, close=close)

    def _authorize(self) -> bool:
        if self.path != "/mcp":
            self._reply(404, close=True)
            return False
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host", "") not in allowed_hosts:
            self._reply(403, close=True)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in allowed_hosts}:
            self._reply(403, close=True)
            return False
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.token}"
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            self._reply(401, close=True)
            return False
        version = self.headers.get("MCP-Protocol-Version")
        if version is not None and version not in _PROTOCOL_VERSIONS:
            self._error(-32600, "Unsupported MCP protocol version", status=400, close=True)
            return False
        return True

    def do_GET(self) -> None:
        if self._authorize():
            self._reply(405)

    def do_HEAD(self) -> None:
        if self._authorize():
            self._reply(405)

    def do_DELETE(self) -> None:
        if self._authorize():
            self._reply(405, close=True)

    def do_POST(self) -> None:
        self._request_body_consumed = False
        if not self._authorize():
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._error(-32600, "A content length is required", status=400, close=True)
            return
        try:
            length = int(lengths[0])
        except ValueError:
            length = -1
        if length < 0:
            self._error(-32600, "Invalid content length", status=400, close=True)
            return
        if length > _MAX_REQUEST_BYTES:
            self._error(-32600, "Request body exceeds the limit", status=413, close=True)
            return
        if self.headers.get("Content-Type", "").partition(";")[0].strip().lower() != "application/json":
            self._error(-32600, "Content type must be application/json", status=415, close=True)
            return
        try:
            raw = self.rfile.read(length)
            self._request_body_consumed = True
            if len(raw) != length:
                self._error(-32700, "Incomplete JSON request", status=400, close=True)
                return
            request = json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant)
        except (UnicodeDecodeError, ValueError):
            self._error(-32700, "Invalid JSON request", status=400, close=True)
            return
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            self._error(-32600, "Invalid JSON-RPC request", status=400)
            return
        request_id = request.get("id")
        has_id = "id" in request
        if has_id and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
            self._error(-32600, "Invalid JSON-RPC request id", status=400)
            return
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            self._error(-32600, "Invalid JSON-RPC request", request_id, status=400)
            return
        if not has_id:
            if not method.startswith("notifications/"):
                self._reply(400)
            else:
                self._reply(202)
            return
        if method == "initialize":
            version = params.get("protocolVersion")
            if not isinstance(version, str) or not version:
                self._error(-32602, "protocolVersion is required", request_id)
                return
            self._result(request_id, {
                "protocolVersion": version if version in _PROTOCOL_VERSIONS else _PROTOCOL_VERSIONS[-1],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mint-finance", "version": "1.0"},
            })
        elif method == "ping":
            self._result(request_id, {})
        elif method == "tools/list":
            self._result(request_id, {"tools": list(self.server._tools.values())})
        elif method == "tools/call":
            self._call_tool(request_id, params)
        else:
            self._error(-32601, "Method not found", request_id)

    def _result(self, request_id: str | int, value: dict[str, Any]) -> None:
        self._reply(200, {"jsonrpc": "2.0", "id": request_id, "result": value})

    def _call_tool(self, request_id: str | int, params: dict[str, Any]) -> None:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or name not in self.server._tools:
            self._error(-32602, "Unknown tool", request_id)
            return
        if not isinstance(arguments, dict):
            self._error(-32602, "Tool arguments must be an object", request_id)
            return
        try:
            value = self.server.bridge.call_tool(name, arguments)
            if not isinstance(value, dict):
                raise ValueError("invalid bridge response")
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
            observation = value.get("observation")
            failed = "error" in value or (isinstance(observation, dict) and "error" in observation)
        except Exception:
            text = json.dumps({"error": "tool execution failed"})
            failed = True
        self._result(request_id, {"content": [{"type": "text", "text": text}], "isError": failed})


def _invalid_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON number")
