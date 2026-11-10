from __future__ import annotations

import gzip
import json
import os
import socket
import sys
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from mint_synth.cli import build_parser
from mint_synth.trace_proxy import TraceProxyConfig, TraceProxyServer
from mint_synth.trace_runner import run_agent_trace


@contextmanager
def running(server: ThreadingHTTPServer):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class TraceProxyTests(unittest.TestCase):
    def test_json_passthrough_and_private_trace(self) -> None:
        observed: dict[str, object] = {}

        class Upstream(QuietHandler):
            def do_POST(self) -> None:
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
                observed["request_id"] = self.headers.get("X-Request-Id")
                observed["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                answer = b'{"result":"ok"}'
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-Id", str(observed["request_id"]))
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        with TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            with running(upstream):
                proxy = TraceProxyServer(TraceProxyConfig(
                    upstream=f"http://127.0.0.1:{upstream.server_port}",
                    output_dir=Path(directory) / "trace",
                    run_id="query-42",
                    port=0,
                ))
                with running(proxy):
                    with httpx.Client(trust_env=False) as client:
                        result = client.post(
                            f"http://127.0.0.1:{proxy.server_port}/v1/messages?api_key=hidden&mode=test",
                            headers={"Authorization": "Bearer secret-value", "X-Api-Key": "other-secret"},
                            content=b'{"model":"local","prompt":"hello"}',
                        )
                    self.assertEqual(result.status_code, 201)
                    self.assertEqual(result.content, b'{"result":"ok"}')
                    self.assertEqual(observed["path"], "/v1/messages?api_key=hidden&mode=test")
                    self.assertEqual(observed["authorization"], "Bearer secret-value")
                    self.assertEqual(result.headers["x-request-id"], observed["request_id"])
                    self.assertEqual(result.headers["x-mint-trace-call-id"], observed["request_id"])

            root = Path(directory) / "trace"
            records = list((root / "calls").glob("*.json"))
            self.assertEqual(len(records), 1)
            record_text = records[0].read_text(encoding="utf-8")
            record = json.loads(record_text)
            self.assertEqual(record["run_id"], "query-42")
            self.assertEqual(record["state"], "complete")
            self.assertEqual(record["request"]["target"], "/v1/messages?api_key=%5BREDACTED%5D&mode=test")
            self.assertNotIn("secret-value", record_text)
            self.assertNotIn("other-secret", record_text)
            self.assertNotIn("hidden", record_text)
            self.assertEqual((root / "calls" / record["request"]["body_file"]).read_bytes(), observed["body"])
            self.assertEqual((root / "calls" / record["response"]["body_file"]).read_bytes(), result.content)
            self.assertEqual(record["response"]["body_bytes"], len(result.content))
            if os.name == "posix":
                self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                self.assertEqual(records[0].stat().st_mode & 0o777, 0o600)

    def test_sse_first_chunk_arrives_before_upstream_finishes(self) -> None:
        release = threading.Event()
        first_sent = threading.Event()
        first = b"event: response.output_text.delta\ndata: {\"delta\":\"a\"}\n\n"
        second = b"event: response.completed\ndata: {}\n\n"

        class Upstream(QuietHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(f"{len(first):X}\r\n".encode() + first + b"\r\n")
                self.wfile.flush()
                first_sent.set()
                if release.wait(5):
                    self.wfile.write(f"{len(second):X}\r\n".encode() + second + b"\r\n0\r\n\r\n")
                    self.wfile.flush()

        with TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            with running(upstream):
                proxy = TraceProxyServer(TraceProxyConfig(
                    upstream=f"http://127.0.0.1:{upstream.server_port}",
                    output_dir=Path(directory) / "trace",
                    run_id="stream-run",
                    port=0,
                ))
                with running(proxy):
                    try:
                        with httpx.Client(trust_env=False, timeout=4) as client:
                            with client.stream("POST", f"http://127.0.0.1:{proxy.server_port}/v1/responses", content=b"{}") as result:
                                chunks = result.iter_raw()
                                self.assertTrue(first_sent.wait(2))
                                self.assertEqual(next(chunks), first)
                                self.assertFalse(release.is_set())
                                release.set()
                                self.assertEqual(b"".join(chunks), second)
                    finally:
                        release.set()

            root = Path(directory) / "trace" / "calls"
            record = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(record["state"], "complete")
            self.assertEqual((root / record["response"]["body_file"]).read_bytes(), first + second)
            timing = [json.loads(line) for line in (root / record["response"]["chunks_file"]).read_text().splitlines()]
            self.assertEqual([item["byte_offset"] for item in timing], [0, len(first)])

    def test_chunked_request_and_compressed_response_keep_payload_bytes(self) -> None:
        seen: dict[str, bytes] = {}
        compressed = gzip.compress(b'{"answer":"ok"}')

        class Upstream(QuietHandler):
            def do_POST(self) -> None:
                seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)

        with TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            with running(upstream):
                proxy = TraceProxyServer(TraceProxyConfig(
                    upstream=f"http://127.0.0.1:{upstream.server_port}",
                    output_dir=Path(directory) / "trace",
                    run_id="binary-run",
                    port=0,
                ))
                with running(proxy):
                    with httpx.Client(trust_env=False) as client:
                        result = client.post(
                            f"http://127.0.0.1:{proxy.server_port}/v1/responses",
                            content=iter((b"first", b"second")),
                        )
                    self.assertEqual(result.content, b'{"answer":"ok"}')
            root = Path(directory) / "trace" / "calls"
            record = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(seen["body"], b"firstsecond")
            self.assertEqual((root / record["request"]["body_file"]).read_bytes(), b"firstsecond")
            self.assertEqual((root / record["response"]["body_file"]).read_bytes(), compressed)

    def test_upstream_connection_failure_and_request_limit_are_recorded(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            unused_port = sock.getsockname()[1]
            with TemporaryDirectory() as directory:
                proxy = TraceProxyServer(TraceProxyConfig(
                    upstream=f"http://127.0.0.1:{unused_port}",
                    output_dir=Path(directory) / "trace",
                    run_id="error-run",
                    port=0,
                    max_request_bytes=3,
                ))
                with running(proxy):
                    with httpx.Client(trust_env=False) as client:
                        base = f"http://127.0.0.1:{proxy.server_port}"
                        self.assertEqual(client.post(f"{base}/v1/messages", content=b"four").status_code, 413)
                        self.assertEqual(client.post(f"{base}/v1/messages", content=b"ok").status_code, 502)
                records = [json.loads(path.read_text(encoding="utf-8")) for path in (Path(directory) / "trace" / "calls").glob("*.json")]
                self.assertEqual({record["state"] for record in records}, {"request_rejected", "upstream_error"})

    def test_upstream_origin_rejects_embedded_credentials_or_path(self) -> None:
        for value in ("http://user:pass@localhost:8000", "http://localhost:8000/v1", "file:///tmp/socket"):
            with self.assertRaises(ValueError):
                TraceProxyConfig(upstream=value, output_dir=Path("trace"), run_id="test")

    def test_proxy_rejects_a_different_agent_run_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trace"
            root.mkdir()
            (root / "agent.json").write_text(json.dumps({"run_id": "another-run"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_id does not match"):
                TraceProxyServer(TraceProxyConfig(
                    upstream="http://127.0.0.1:8000",
                    output_dir=root,
                    run_id="expected-run",
                    port=0,
                ))

    def test_agent_event_timing_and_run_id_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trace"
            root.mkdir()
            (root / "run.json").write_text(json.dumps({"run_id": "query-42"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_agent_trace([sys.executable, "-c", "pass"], output_dir=root, run_id="wrong")
            command = [sys.executable, "-c", "import sys; print('{\"type\":\"turn.started\"}'); print('notice', file=sys.stderr)"]
            self.assertEqual(run_agent_trace(command, output_dir=root, run_id="query-42"), 0)
            self.assertEqual(json.loads((root / "agent.json").read_text())["state"], "complete")
            self.assertEqual((root / "agent.stdout.jsonl").read_text().strip(), '{"type":"turn.started"}')
            self.assertEqual((root / "agent.stderr.log").read_text().strip(), "notice")
            timing = [json.loads(line) for line in (root / "agent.timing.jsonl").read_text().splitlines()]
            self.assertEqual({item["stream"] for item in timing}, {"stdout", "stderr"})
            self.assertTrue(all(item["byte_length"] > 0 for item in timing))

    def test_cli_keeps_agent_flags_after_separator(self) -> None:
        args = build_parser().parse_args([
            "trace-run", "--output-dir", "/tmp/trace", "--run-id", "r1", "--",
            "codex", "exec", "--json", "-c", 'model_provider="trace"', "prompt",
        ])
        self.assertEqual(args.agent_command[1:], [
            "codex", "exec", "--json", "-c", 'model_provider="trace"', "prompt",
        ])


if __name__ == "__main__":
    unittest.main()
