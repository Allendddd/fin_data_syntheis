from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mint_synth.agents.base import ClaudeCodeSettings, CodexSettings, ModelSettings, RuntimeSettings
from mint_synth.agents.cli_runtime import run_cli
from mint_synth.harness import case_from_task
from test_teacher_proof import snapshot, task


_FAKE_CLI = r'''
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

runtime, mode, report_path = sys.argv[1:4]
args = sys.argv[4:]
report = Path(report_path)
prompt = sys.stdin.read()
assert "public-query" in prompt
assert "hidden-answer" not in prompt
if mode == "timeout":
    if os.name == "posix":
        marker = str(report.with_suffix(".child"))
        subprocess.Popen([sys.executable, "-c", "import time; from pathlib import Path; time.sleep(0.7); Path(" + repr(marker) + ").write_text('alive')"])
    print(json.dumps({"type": "thread.started"}), flush=True)
    time.sleep(10)
    sys.exit(0)

if runtime == "codex":
    assert args[:2] == ["exec", "--json"]
    assert "--ephemeral" in args and "--strict-config" in args
    text = (Path(os.environ["CODEX_HOME"]) / "config.toml").read_text()
    def value(key):
        return json.loads(re.search(r"^" + re.escape(key) + r" = (.+)$", text, re.M).group(1))
    assert value("base_url") == "http://127.0.0.1:9876/v1"
    assert value("wire_api") == "responses"
    assert value("supports_websockets") is False
    if mode.startswith("native"):
        assert args[args.index("--sandbox") + 1] == "workspace-write"
        assert value("shell_tool") is True
        assert value("view_image") is True
        assert value("network_access") is False
        assert value("ignore_default_excludes") is False
        assert 'model_instructions_file' not in text
        assert '[mcp_servers.finance]' not in text
        assert 'MINT_MCP_TOKEN' not in os.environ
        assert not any('bypass' in part or part == '--yolo' for part in args)
        assert 'corpus-manifest.json' in prompt
        assert 'submit_final' not in prompt and 'search_corpus' not in prompt
        manifest = json.loads(Path('corpus-manifest.json').read_text())
        source = Path(manifest['sources'][0]['file'])
        assert 'Revenue was' in source.read_text()
        report.write_text(json.dumps({'home':os.environ['HOME'],'cwd':os.getcwd(),'prompt':prompt}))
        print(json.dumps({'type':'thread.started','thread_id':'native-thread'}))
        print(json.dumps({'type':'turn.started'}))
        if mode == 'native_mcp':
            print(json.dumps({'type':'item.completed','item':{'id':'mcp-1','type':'mcp_tool_call','server':'finance','tool':'submit_final'}}))
        else:
            print(json.dumps({'type':'item.completed','item':{'id':'exec-1','type':'command_execution','command':'cat -- '+str(source.resolve()),'aggregated_output':source.read_text(),'exit_code':0,'status':'completed'}}))
            print(json.dumps({'type':'item.completed','item':{'id':'patch-1','type':'file_change','changes':[],'status':'completed'}}))
        if mode != 'native_no_final':
            print(json.dumps({'type':'item.completed','item':{'id':'final-1','type':'agent_message','text':'finished'}}))
        if mode == 'native_failed':
            print(json.dumps({'type':'turn.failed','error':{'message':'failure'}}))
        else:
            print(json.dumps({'type':'turn.completed','usage':{}}))
        sys.exit(0)
    assert args[args.index("--sandbox") + 1] == "read-only"
    url = value("url")
    token = os.environ["MINT_MCP_TOKEN"]
    assert value("shell_tool") is False and value("apps") is False
    assert value("enable_request_compression") is False
    assert "submit_final" in value("enabled_tools")
else:
    assert "--print" in args
    if mode.startswith("native"):
        assert "--bare" not in args
        assert "--system-prompt" not in args and "--append-system-prompt" not in args
        assert "--disable-slash-commands" not in args
        assert args[args.index("--tools") + 1] == "Read,Bash"
        allowed = args[args.index("--allowedTools") + 1:args.index("--permission-mode")]
        assert "Read" in allowed and "Bash(echo *)" in allowed
        assert "corpus-manifest.json" in prompt
        assert "submit_final" not in prompt and "search_corpus" not in prompt
        assert args[args.index("--disallowedTools") + 1] == "mcp__*"
        assert "MINT_MCP_TOKEN" not in os.environ
    else:
        assert "--bare" in args and "--system-prompt" in args
        assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args
    assert args[args.index("--permission-mode") + 1] == "dontAsk"
    assert args[args.index("--output-format") + 1] == "stream-json"
    data = json.loads(Path(args[args.index("--mcp-config") + 1]).read_text())
    if mode.startswith("native"):
        assert data == {"mcpServers": {}}
        manifest = json.loads(Path("corpus-manifest.json").read_text())
        assert len(manifest["sources"]) == 2
        source = Path(manifest["sources"][0]["file"])
        assert "Revenue was" in source.read_text()
        report.write_text(json.dumps({"home": os.environ["HOME"], "cwd": os.getcwd(), "prompt": prompt}))
        name = {"native_foreign": "mcp__other__send", "native_finance": "mcp__finance__submit_final"}.get(mode, "Read")
        print(json.dumps({"type": "assistant", "message": {"id": "msg-read", "content": [
            {"type": "tool_use", "id": "read-1", "name": name, "input": {"file_path": str(source.resolve())}}]}}))
        print(json.dumps({"type": "assistant", "parent_tool_use_id": None, "message": {"id": "msg-final",
            "content": [{"type": "text", "text": "finished"}]}}))
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "finished"}))
        sys.exit(0)
    assert set(data["mcpServers"]) == {"finance"}
    url = data["mcpServers"]["finance"]["url"]
    token = data["mcpServers"]["finance"]["headers"]["Authorization"].removeprefix("Bearer ")
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9876"
    assert os.environ["ANTHROPIC_MODEL"] == "served-model"

report.write_text(json.dumps({"home": os.environ["HOME"], "cwd": os.getcwd(), "has_model_key": bool(os.environ["MINT_MODEL_API_KEY"])}))

def rpc(method, params, ident):
    body = json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}).encode()
    request = Request(url, body, {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": "Bearer " + token})
    with urlopen(request, timeout=3) as response:
        return json.loads(response.read())["result"]

rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "fake", "version": "1"}}, 1)
tools = rpc("tools/list", {}, 2)["tools"]
assert {item["name"] for item in tools} == {"echo", "submit_final"}
assert "grade" not in str(tools)
if mode not in {"incomplete", "exit"}:
    result = rpc("tools/call", {"name": "submit_final", "arguments": {"proof": {"answer": "42"}}}, 3)
    assert not result["isError"]

if mode == "malformed":
    print("not-json")
elif runtime == "codex":
    print(json.dumps({"type": "thread.started", "thread_id": "fake-thread"}))
    if mode == "forbidden":
        print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "echo bypassed"}}))
    else:
        print(json.dumps({"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "finance", "tool": "submit_final", "status": "completed"}}))
    print(json.dumps({"type": "turn.completed"}))
else:
    name = "Bash" if mode in {"forbidden", "native"} else "mcp__finance__submit_final"
    if mode == "native_foreign":
        name = "mcp__other__send"
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "call-1", "name": name, "input": {}}]}}))
    print(json.dumps({"type": "result", "is_error": False, "result": "finished"}))
print("fake diagnostic", file=sys.stderr)
sys.exit(7 if mode == "exit" else 0)
'''


class FakeBridge:
    system_prompt = "Use only finance tools; submit a financial proof."

    def __init__(self) -> None:
        self.done = False
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    @property
    def tool_definitions(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "description": "public tool",
             "parameters": {"type": "object", "properties": {}}}}
            for name in ("echo", "submit_final")
        ]

    def public_state(self) -> dict:
        return {"question": "public-query", "budgets": {"steps_remaining": 3}}

    def call_tool(self, name: str, arguments: dict) -> dict:
        with self._lock:
            self.calls.append((name, arguments))
            if name == "submit_final":
                self.done = True
            return {"done": self.done, "observation": {"ok": True}}


class CliRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.script = self.root / "fake_agent.py"
        self.script.write_text(_FAKE_CLI, encoding="utf-8")
        self.model = ModelSettings("served-model", "http://127.0.0.1:9876/v1", "test-model-key", timeout_seconds=5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_fake(self, runtime: str, mode: str = "complete", *, timeout: float = 5,
                 claude_code: ClaudeCodeSettings | None = None, codex: CodexSettings | None = None):
        bridge = FakeBridge()
        report = self.root / f"{runtime}-{mode}.report"
        settings = RuntimeSettings(runtime, command=(sys.executable, str(self.script), runtime, mode, str(report)),
                                   timeout_seconds=timeout, claude_code=claude_code or ClaudeCodeSettings(),
                                   codex=codex or CodexSettings())
        if ((runtime == "claude_code" and settings.claude_code.mode == "native")
                or (runtime == "codex" and settings.codex.mode == "native")):
            index = self.root / f"{mode}.sqlite3"
            snapshot(index)
            native_task = task()
            native_task["question"] = "public-query"
            native_task["reasoning_trace"] = ["hidden-answer"]
            bridge = case_from_task(native_task, index, cutoff="2025-12-31")
        trace = self.root / f"{runtime}-{mode}-trace"
        result = run_cli(bridge, self.model, settings, trace)
        return bridge, report, trace, result

    def test_codex_and_claude_call_real_mcp_and_clean_temporary_configuration(self) -> None:
        for runtime in ("codex", "claude_code"):
            with self.subTest(runtime=runtime):
                bridge, report, trace, result = self.run_fake(runtime)
                self.assertEqual(result.status, "completed", (result.error, (trace / "stderr.log").read_text()))
                self.assertTrue(bridge.done)
                self.assertEqual(bridge.calls, [("submit_final", {"proof": {"answer": "42"}})])
                child = json.loads(report.read_text())
                self.assertTrue(child["has_model_key"])
                self.assertFalse(Path(child["home"]).exists())
                self.assertFalse(Path(child["cwd"]).exists())
                self.assertEqual({path.name for path in trace.iterdir()}, {"stdout.jsonl", "stderr.log", "cli.json"})
                self.assertEqual(result.metadata["events_file"], "stdout.jsonl")
                self.assertGreater(result.metadata["event_count"], 0)
                self.assertNotIn("test-model-key", "".join(path.read_text() for path in trace.iterdir()))
                if os.name == "posix":
                    self.assertEqual(trace.stat().st_mode & 0o777, 0o700)
                    self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in trace.iterdir()))

    def test_successful_cli_exit_without_environment_final_is_incomplete(self) -> None:
        bridge, _, _, result = self.run_fake("codex", "incomplete")
        self.assertFalse(bridge.done)
        self.assertEqual(result.status, "incomplete")

    def test_non_finance_tool_events_fail_even_after_a_final_submission(self) -> None:
        for runtime in ("codex", "claude_code"):
            with self.subTest(runtime=runtime):
                bridge, _, _, result = self.run_fake(runtime, "forbidden")
                self.assertTrue(bridge.done)
                self.assertEqual(result.status, "failed")
                self.assertIn("non-finance", result.error)

    def test_claude_native_keeps_default_system_and_accepts_builtin_events(self) -> None:
        options = ClaudeCodeSettings(mode="native", tools=("Read", "Bash"), allowed_tools=("Read", "Bash(echo *)"))
        with patch("mint_synth.agents.cli_runtime.FinanceMCPServer", side_effect=AssertionError("no MCP in native mode")):
            case, report, trace, result = self.run_fake("claude_code", "native", claude_code=options)
        self.assertEqual(result.status, "completed", result.error)
        self.assertFalse(hasattr(case, "call_tool"))
        child = json.loads(report.read_text())
        self.assertFalse(Path(child["home"]).exists())
        manifest = json.loads((trace / "native-workspace.json").read_text())
        self.assertTrue(manifest["integrity"]["valid"])
        self.assertEqual(result.metadata["claude_code_mode"], "native")

    def test_claude_native_rejects_all_mcp_including_finance(self) -> None:
        options = ClaudeCodeSettings(mode="native", tools=("Read", "Bash"), allowed_tools=("Read", "Bash(echo *)"))
        for mode in ("native_foreign", "native_finance"):
            with self.subTest(mode=mode):
                _, _, _, result = self.run_fake("claude_code", mode, claude_code=options)
                self.assertEqual(result.status, "failed")
                self.assertIn("MCP", result.error)

    def test_codex_native_uses_builtin_tools_without_finance_server_or_system(self) -> None:
        with patch("mint_synth.agents.cli_runtime.FinanceMCPServer", side_effect=AssertionError("no MCP in native mode")):
            case, report, trace, result = self.run_fake("codex", "native", codex=CodexSettings(mode="native"))
        self.assertEqual(result.status, "completed", (result.error, (trace / "stderr.log").read_text()))
        self.assertFalse(hasattr(case, "call_tool"))
        child = json.loads(report.read_text())
        self.assertFalse(Path(child["home"]).exists())
        self.assertFalse(Path(child["cwd"]).exists())
        manifest = json.loads((trace / "native-workspace.json").read_text())
        self.assertTrue(manifest["integrity"]["valid"])
        self.assertEqual(result.metadata["codex_mode"], "native")
        self.assertEqual(result.metadata["sandbox"], "workspace-write")
        self.assertEqual(len(result.metadata["final_text_sha256"]), 64)

    def test_codex_native_rejects_mcp_failed_turn_and_missing_final(self) -> None:
        for mode in ("native_mcp", "native_failed", "native_no_final"):
            with self.subTest(mode=mode):
                _, _, _, result = self.run_fake("codex", mode, codex=CodexSettings(mode="native"))
                self.assertEqual(result.status, "failed", result)

    def test_malformed_events_and_nonzero_exit_are_failures(self) -> None:
        for mode in ("malformed", "exit"):
            with self.subTest(mode=mode):
                _, _, _, result = self.run_fake("codex", mode)
                self.assertEqual(result.status, "failed")

    def test_timeout_terminates_the_cli_and_its_child(self) -> None:
        _, report, _, result = self.run_fake("codex", "timeout", timeout=0.25)
        self.assertEqual(result.status, "timeout", result.error)
        time.sleep(0.9)
        self.assertFalse(report.with_suffix(".child").exists())

    def test_missing_cli_is_recorded_and_extra_flags_are_rejected(self) -> None:
        result = run_cli(FakeBridge(), self.model, RuntimeSettings("codex", command=(str(self.root / "missing-cli"),)),
                         self.root / "missing-trace")
        self.assertEqual(result.status, "failed")
        self.assertIn("FileNotFoundError", result.error)
        with self.assertRaisesRegex(ValueError, "extra CLI"):
            run_cli(FakeBridge(), self.model, RuntimeSettings("codex", extra_args=("--sandbox", "danger-full-access")),
                    self.root / "unsafe-trace")


if __name__ == "__main__":
    unittest.main()
