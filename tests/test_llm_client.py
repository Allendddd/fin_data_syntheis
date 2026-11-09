from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mint_synth.llm import ClientConfig, OpenAICompatibleClient


class Handler(BaseHTTPRequestHandler):
    request_payload: dict = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).request_payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class LlmClientTests(unittest.TestCase):
    def test_openai_compatible_request_and_usage_metrics(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleClient(
                ClientConfig(
                    model="test-model",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    retries=0,
                    seed=7,
                )
            )
            self.assertEqual(client.complete_json(system="system", user="user"), {"ok": True})
            self.assertEqual(Handler.request_payload["seed"], 7)
            client.complete_json(system="system", user="user", sample_key="task:1")
            first_seed = Handler.request_payload["seed"]
            client.complete_json(system="system", user="user", sample_key="task:1")
            self.assertEqual(Handler.request_payload["seed"], first_seed)
            client.complete_json(system="system", user="user", sample_key="task:2")
            self.assertNotEqual(Handler.request_payload["seed"], first_seed)
            self.assertNotIn("sample_key", Handler.request_payload)
            self.assertEqual(Handler.request_payload["messages"][1]["content"], "user")
            self.assertEqual(client.stats()["total_tokens"], 48)
            client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
