from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mint_synth.agents.base import ModelSettings
from mint_synth.agents.strands_native import run_strands_native
from mint_synth.agents.strands_native_worker import final_message, invoke
from mint_synth.agents.strands_runtime import run_strands
from mint_synth.agents.native_capture import _returned_text
from mint_synth.core.run_state import atomic_json


class NativeAgent:
    instances = []
    stop_reason = "end_turn"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = []
        self.cleaned = False
        self.instances.append(self)

    async def invoke_async(self, prompt):
        self.messages = [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "read-1", "name": "file_read",
                                                            "input": {"path": "/corpus/source.txt"}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "read-1", "status": "success",
                                                           "content": [{"text": "Revenue was 100."}]}}]},
            {"role": "assistant", "content": [{"text": "A native final answer."}]},
        ]
        self.kwargs["callback_handler"](message=self.messages[-1], client=self.kwargs["model"])
        return SimpleNamespace(message=self.messages[-1], stop_reason=self.stop_reason)

    async def shutdown_async(self):
        self.cleaned = True


class Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_sdk():
    modules = {name: types.ModuleType(name) for name in [
        "strands", "strands.models", "strands.models.openai", "strands.tools", "strands.tools.executors",
        "strands_tools", "strands_tools.file_read", "strands_tools.shell",
    ]}
    modules["strands"].Agent = NativeAgent
    modules["strands.models.openai"].OpenAIModel = Model
    modules["strands.tools.executors"].SequentialToolExecutor = type("SequentialToolExecutor", (), {})
    return modules


def payload(root):
    return {"trace_dir": str(root), "prompt": "Read public corpus files and answer.", "tools": ["file_read", "shell"],
            "model": {"model": "local-teacher", "base_url": "http://127.0.0.1:9000/v1",
                      "temperature": 0.3, "max_tokens": 2400, "timeout_seconds": 30}}


def settings(*, timeout=30, tools=("file_read", "shell")):
    return SimpleNamespace(name="strands", command=(), extra_args=(), timeout_seconds=timeout,
                           strands=SimpleNamespace(mode="native", tools=tools))


def test_worker_registers_official_modules_and_keeps_sdk_system_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MINT_MODEL_API_KEY", "private-provider-key")
    NativeAgent.stop_reason = "end_turn"
    modules = fake_sdk()
    with patch.dict(sys.modules, modules):
        outcome = asyncio.run(invoke(payload(tmp_path)))
    assert outcome["status"] == "completed"
    agent = NativeAgent.instances[-1]
    assert agent.kwargs["tools"] == [modules["strands_tools.file_read"], modules["strands_tools.shell"]]
    assert "system_prompt" not in agent.kwargs and "hooks" not in agent.kwargs
    assert not agent.kwargs["load_tools_from_directory"]
    assert agent.cleaned
    final = json.loads((tmp_path / "native-final.json").read_text())
    assert final == {"text": "A native final answer.", "message": agent.messages[-1]}
    assert json.loads((tmp_path / "agent.messages.json").read_text()) == agent.messages
    assert "private-provider-key" not in (tmp_path / "agent.events.jsonl").read_text()
    model = agent.kwargs["model"].kwargs
    assert model["client_args"]["base_url"] == payload(tmp_path)["model"]["base_url"]
    assert model["client_args"]["api_key"] == "private-provider-key"


@pytest.mark.parametrize("message,reason", [
    ({"role": "assistant", "content": [{"text": "truncated"}]}, "max_tokens"),
    ({"role": "assistant", "content": [{"toolUse": {"name": "shell"}}]}, "end_turn"),
    ({"role": "assistant", "content": []}, "end_turn"),
    ({"role": "user", "content": [{"text": "not an assistant"}]}, "end_turn"),
])
def test_worker_requires_an_untruncated_normal_final_reply(message, reason):
    with pytest.raises(ValueError):
        final_message(message, reason)


def test_worker_missing_optional_dependency_preserves_failure_artifacts(tmp_path):
    with patch.dict(sys.modules, {"strands": None}):
        outcome = asyncio.run(invoke(payload(tmp_path)))
    assert outcome["status"] == "failed"
    assert "mint-synth[strands]" in outcome["error"]
    assert (tmp_path / "agent.events.jsonl").is_file()
    assert json.loads((tmp_path / "agent.messages.json").read_text()) == []
    assert json.loads((tmp_path / "native-worker.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("status,code", [("error", "0"), ("success", "1"), ("error", "-15")])
def test_official_shell_error_output_cannot_supply_native_evidence(status, code):
    # Exact text layout emitted by strands-agents-tools 0.8.x shell(). The
    # OpenAI provider joins these adjacent native content blocks with newlines.
    text = ("Execution Summary:\nTotal commands: 1\nSuccessful: 0\nFailed: 1\n"
            "Command: cat -- '/workspace/corpus/source.txt'\n"
            f"Status: {status}\nExit Code: {code}\n"
            "Output: Revenue was 100.\nError: read failed")
    assert _returned_text(text) == ""


def test_official_shell_success_output_still_exposes_source_text():
    text = ("Execution Summary:\nTotal commands: 1\nSuccessful: 1\nFailed: 0\n"
            "Command: cat -- '/workspace/corpus/source.txt'\n"
            "Status: success\nExit Code: 0\nOutput: Revenue was 100.\nError: ")
    assert "Revenue was 100." in _returned_text(text)


def test_parent_uses_isolated_worker_cwd_and_environment_without_mcp(tmp_path, monkeypatch):
    model = ModelSettings("local", "http://127.0.0.1:9000/v1", "private-provider-key")
    original_cwd = Path.cwd()
    original_consent = os.environ.get("BYPASS_TOOL_CONSENT")
    observed = {}

    def prepare(case, workspace, root, *, runtime):
        assert runtime == "strands" and case == "public-case"
        observed["workspace"] = workspace
        return "ordinary public corpus prompt"

    def finish(workspace, root):
        assert workspace.is_dir()
        observed["finished"] = True

    class Worker:
        returncode = 0

        def __init__(self, command, **kwargs):
            observed.update(command=command, options=kwargs)

        def communicate(self, data, timeout):
            request = json.loads(data)
            observed["request"] = request
            root = Path(request["trace_dir"])
            atomic_json(root / "native-worker.json", {"status": "completed", "error": None})
            atomic_json(root / "native-final.json", {"text": "Final answer.", "message": {"role": "assistant", "content": [{"text": "Final answer."}]}})

    with (patch("mint_synth.agents.strands_native.prepare_workspace", side_effect=prepare),
          patch("mint_synth.agents.strands_native.finish_workspace", side_effect=finish),
          patch("mint_synth.agents.strands_native.subprocess.Popen", Worker),
          patch("mint_synth.agents.cli_runtime.FinanceMCPServer", side_effect=AssertionError("no MCP"))):
        result = run_strands("public-case", model, settings(), tmp_path)
    assert result.status == "completed"
    assert observed["finished"] and not observed["workspace"].exists()
    assert observed["options"]["cwd"] == observed["workspace"]
    environment = observed["options"]["env"]
    assert environment["STRANDS_NON_INTERACTIVE"] == environment["BYPASS_TOOL_CONSENT"] == "true"
    assert environment["MINT_MODEL_API_KEY"] == "private-provider-key"
    assert "MINT_MCP_TOKEN" not in environment
    assert "private-provider-key" not in json.dumps(observed["request"])
    assert observed["command"][-1] == "mint_synth.agents.strands_native_worker"
    assert Path.cwd() == original_cwd and os.environ.get("BYPASS_TOOL_CONSENT") == original_consent


def test_parent_timeout_terminates_worker_and_still_finishes_workspace(tmp_path):
    finished, terminated = [], []

    class Worker:
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, data, timeout):
            raise subprocess.TimeoutExpired("worker", timeout)

    def terminate(process):
        process.returncode = -9
        terminated.append(process)

    with (patch("mint_synth.agents.strands_native.prepare_workspace", return_value="public prompt"),
          patch("mint_synth.agents.strands_native.finish_workspace", side_effect=lambda *args: finished.append(True)),
          patch("mint_synth.agents.strands_native.subprocess.Popen", Worker),
          patch("mint_synth.agents.strands_native._terminate_worker", side_effect=terminate)):
        result = run_strands_native("case", ModelSettings("local", "http://localhost/v1", None), settings(timeout=0.01), tmp_path)
    assert result.status == "timeout" and terminated and finished


def test_real_worker_subprocess_loads_native_sdk_tools_in_its_workspace(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake-sdk"
    files = {
        "strands/__init__.py": '''
import os
from pathlib import Path
from types import SimpleNamespace
class Agent:
    def __init__(self, **kwargs):
        assert 'system_prompt' not in kwargs and 'hooks' not in kwargs
        assert [tool.__name__ for tool in kwargs['tools']] == ['strands_tools.file_read', 'strands_tools.shell']
        assert Path.cwd().name == 'workspace'
        assert os.environ['STRANDS_NON_INTERACTIVE'] == 'true'
        assert os.environ['BYPASS_TOOL_CONSENT'] == 'true'
        self.messages = []
    async def invoke_async(self, prompt):
        assert prompt == 'public source prompt'
        source = Path('corpus/source.txt').read_text()
        self.messages = [
            {'role':'user','content':[{'text':prompt}]},
            {'role':'assistant','content':[{'text':'Read: ' + source}]}]
        return SimpleNamespace(message=self.messages[-1], stop_reason='end_turn')
    async def shutdown_async(self):
        pass
''',
        "strands/models/__init__.py": "",
        "strands/models/openai.py": "class OpenAIModel:\n def __init__(self, **kwargs): self.kwargs=kwargs\n",
        "strands/tools/__init__.py": "",
        "strands/tools/executors.py": "class SequentialToolExecutor: pass\n",
        "strands_tools/__init__.py": "",
        "strands_tools/file_read.py": "TOOL_SPEC={'name':'file_read'}\n",
        "strands_tools/shell.py": "TOOL_SPEC={'name':'shell'}\n",
    }
    for name, source in files.items():
        path = fake_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake_root))
    finished = []

    def prepare(case, workspace, trace_dir, *, runtime):
        corpus = workspace / "corpus"
        corpus.mkdir()
        (corpus / "source.txt").write_text("Native source text", encoding="utf-8")
        return "public source prompt"

    def finish(workspace, trace_dir):
        assert (workspace / "corpus/source.txt").read_text() == "Native source text"
        finished.append(True)

    trace = tmp_path / "trace"
    with (patch("mint_synth.agents.strands_native.prepare_workspace", side_effect=prepare),
          patch("mint_synth.agents.strands_native.finish_workspace", side_effect=finish)):
        outcome = run_strands_native("case", ModelSettings("local", "http://localhost/v1", None), settings(), trace)
    assert outcome.status == "completed", (outcome.error, (trace / "worker.stderr.log").read_text())
    assert finished and outcome.metadata["exit_code"] == 0
    assert json.loads((trace / "native-final.json").read_text())["text"] == "Read: Native source text"
    assert json.loads((trace / "agent.messages.json").read_text())[-1]["role"] == "assistant"


@pytest.mark.parametrize("tools", [("mcp__finance__search_corpus",), ("file_read", "file_read"), (), ("load_tool",)])
def test_parent_rejects_custom_or_ambiguous_tools(tmp_path, tools):
    with pytest.raises(ValueError, match="official tool"):
        run_strands_native("case", ModelSettings("local", "http://localhost/v1", None), settings(tools=tools), tmp_path)
