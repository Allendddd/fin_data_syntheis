from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from mint_synth.agents.mcp_server import FinanceMCPServer


class FakeBridge:
    tool_definitions = [{
        "type": "function",
        "function": {
            "name": "search_corpus", "description": "Search a frozen corpus",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }]

    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if arguments.get("query") == "explode":
            raise RuntimeError("private verifier value must never leak")
        if arguments.get("query") == "bad":
            return {"done": False, "observation": {"error": "invalid query"}}
        return {"done": False, "observation": {"hits": ["chunk-1"]}}


class FinanceMCPTests(unittest.TestCase):
    def rpc(self, client, url, method, params=None, request_id=1):
        return client.post(url, json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})

    def test_initialize_discover_call_and_negotiate(self):
        bridge = FakeBridge()
        with FinanceMCPServer(bridge) as server:
            with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {server.token}"}) as client:
                result = self.rpc(client, server.url, "initialize", {"protocolVersion": "2025-03-26"}).json()["result"]
                self.assertEqual(result["protocolVersion"], "2025-03-26")
                self.assertEqual(result["capabilities"], {"tools": {"listChanged": False}})
                unknown = self.rpc(client, server.url, "initialize", {"protocolVersion": "2030-01-01"}).json()
                self.assertEqual(unknown["result"]["protocolVersion"], "2025-06-18")
                tools = self.rpc(client, server.url, "tools/list").json()["result"]["tools"]
                self.assertEqual([tool["name"] for tool in tools], ["search_corpus"])
                self.assertEqual(tools[0]["inputSchema"], bridge.tool_definitions[0]["function"]["parameters"])
                answer = self.rpc(client, server.url, "tools/call", {"name": "search_corpus", "arguments": {"query": "revenue"}}, "a").json()
                self.assertEqual(answer["id"], "a")
                self.assertFalse(answer["result"]["isError"])
                self.assertEqual(json.loads(answer["result"]["content"][0]["text"])["observation"], {"hits": ["chunk-1"]})
                self.assertEqual(bridge.calls, [("search_corpus", {"query": "revenue"})])

    def test_notifications_ping_and_unsupported_methods(self):
        with FinanceMCPServer(FakeBridge()) as server:
            with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {server.token}"}) as client:
                response = client.post(server.url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.content, b"")
                self.assertEqual(self.rpc(client, server.url, "ping").json()["result"], {})
                self.assertEqual(self.rpc(client, server.url, "grade").json()["error"]["code"], -32601)
                response = client.get(server.url)
                self.assertEqual(response.status_code, 405)
                self.assertEqual(response.headers["allow"], "POST")

    def test_auth_path_origin_and_protocol_header(self):
        bridge = FakeBridge()
        with FinanceMCPServer(bridge) as server:
            with httpx.Client(trust_env=False) as client:
                self.assertEqual(client.post(server.url, json={}).status_code, 401)
                headers = {"Authorization": f"Bearer {server.token}"}
                self.assertEqual(client.post(server.url + "/private", headers=headers, json={}).status_code, 404)
                self.assertEqual(client.post(server.url, headers={**headers, "Origin": "https://evil.example"}, json={}).status_code, 403)
                self.assertEqual(client.post(server.url, headers={**headers, "Host": "evil.example"}, json={}).status_code, 403)
                self.assertEqual(client.post(server.url, headers={**headers, "MCP-Protocol-Version": "bad"}, json={}).status_code, 400)
                self.assertEqual(bridge.calls, [])

    def test_malformed_requests_cannot_dispatch_tools(self):
        bridge = FakeBridge()
        with FinanceMCPServer(bridge) as server:
            with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {server.token}"}) as client:
                response = client.post(server.url, content=b"{", headers={"Content-Type": "application/json"})
                self.assertEqual(response.json()["error"]["code"], -32700)
                for payload in ([], {"id": 1, "method": "tools/list"}, {"jsonrpc": "2.0", "id": True, "method": "ping"}):
                    self.assertEqual(client.post(server.url, json=payload).status_code, 400)
                for params in ({"name": "grade"}, {"name": "search_corpus", "arguments": []}):
                    result = self.rpc(client, server.url, "tools/call", params).json()
                    self.assertEqual(result["error"]["code"], -32602)
                self.assertEqual(bridge.calls, [])

    def test_tool_errors_are_results_without_exception_details(self):
        with FinanceMCPServer(FakeBridge()) as server:
            with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {server.token}"}) as client:
                result = self.rpc(client, server.url, "tools/call", {"name": "search_corpus", "arguments": {"query": "bad"}}).json()["result"]
                self.assertTrue(result["isError"])
                self.assertIn("invalid query", result["content"][0]["text"])
                response = self.rpc(client, server.url, "tools/call", {"name": "search_corpus", "arguments": {"query": "explode"}})
                self.assertTrue(response.json()["result"]["isError"])
                self.assertNotIn("private verifier", response.text)

    def test_http_body_limit_and_content_type(self):
        with FinanceMCPServer(FakeBridge()) as server:
            with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {server.token}"}) as client:
                response = client.post(server.url, content=b"{}")
                self.assertEqual(response.status_code, 415)
                with patch("mint_synth.agents.mcp_server._MAX_REQUEST_BYTES", 8):
                    response = client.post(server.url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                self.assertEqual(response.status_code, 413)

    def test_rejection_drains_a_small_delayed_body_before_closing(self):
        with FinanceMCPServer(FakeBridge()) as server:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as connection:
                connection.sendall((
                    "POST /mcp HTTP/1.1\r\nHost: evil.example\r\n"
                    "Content-Length: 2\r\nContent-Type: application/json\r\n\r\n"
                ).encode("ascii"))
                body_started = threading.Event()

                def send_body():
                    time.sleep(0.05)
                    body_started.set()
                    connection.sendall(b"{}")

                sender = threading.Thread(target=send_body)
                sender.start()
                response = connection.recv(4096)
                body_arrived_before_response = body_started.is_set()
                sender.join(timeout=2)
                self.assertTrue(response.startswith(b"HTTP/1.1 403"))
                self.assertTrue(body_arrived_before_response, "response closed before the declared body arrived")

    def test_rejection_drain_has_a_total_deadline(self):
        with FinanceMCPServer(FakeBridge()) as server:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as connection:
                with patch("mint_synth.agents.mcp_server._REJECT_DRAIN_TIMEOUT_SECONDS", 0.1):
                    connection.sendall((
                        "POST /mcp HTTP/1.1\r\nHost: evil.example\r\n"
                        "Content-Length: 2\r\nContent-Type: application/json\r\n\r\n"
                    ).encode("ascii"))
                    response = connection.recv(4096)
                self.assertTrue(response.startswith(b"HTTP/1.1 403"))


if __name__ == "__main__":
    unittest.main()
