from __future__ import annotations

import asyncio
import json
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mint_synth.agents.base import ModelSettings, RuntimeSettings
from mint_synth.agents.strands_runtime import run_strands


class FakeBridge:
    def __init__(self):
        self.done = False
        self.calls = []
        self.system_prompt = "Use only financial tools and submit_final."
        self.tool_definitions = [
            {"type": "function", "function": {"name": "financial_lookup", "description": "Look up a frozen value",
                "parameters": {"type": "object", "properties": {"period": {"type": "string", "enum": ["2024", "2025"]}},
                               "required": ["period"], "additionalProperties": False}}},
            {"type": "function", "function": {"name": "submit_final", "description": "Submit proof",
                "parameters": {"type": "object", "properties": {"proof": {"type": "object"}},
                               "required": ["proof"], "additionalProperties": False}}},
        ]

    def public_state(self):
        return {"question": "Find revenue", "budgets": {"steps_remaining": 3}}

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        self.done = name == "submit_final"
        if name not in {"financial_lookup", "submit_final"}:
            return {"done": False, "observation": {"error": "unknown tool"}}
        return {"done": self.done, "observation": None if self.done else {"value": "42"}}


class FakeModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)


class FakeTool:
    def __init__(self, *, tool_name, tool_spec, tool_func):
        self.tool_name, self.tool_spec, self.tool_func = tool_name, tool_spec, tool_func


class FakeAgent:
    instances = []
    mode = "complete"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = []
        self.cancelled = False
        self.cleaned = False
        self.before_callbacks = []
        registry = types.SimpleNamespace(add_callback=lambda _event_type, callback: self.before_callbacks.append(callback))
        for hook in kwargs["hooks"]:
            hook.register_hooks(registry)
        self.instances.append(self)

    def cancel(self, **_kwargs):
        self.cancelled = True

    async def shutdown_async(self):
        self.cleaned = True

    async def invoke_async(self, prompt):
        self.messages.append({"role": "user", "content": [{"text": prompt}]})
        callback = self.kwargs["callback_handler"]
        callback(data="model thinking", client=self.kwargs["model"])
        if self.mode == "timeout":
            await asyncio.Event().wait()
        if self.mode == "error":
            raise RuntimeError("provider included a secret API key")
        if self.mode != "natural":
            tools = {item.tool_name: item for item in self.kwargs["tools"]}
            if self.mode == "unknown":
                event = types.SimpleNamespace(selected_tool=None, tool_use={"name": "unregistered", "toolUseId": "call-0", "input": {}})
                for callback in self.before_callbacks:
                    callback(event)
                await event.selected_tool.tool_func(event.tool_use)
            lookup = await tools["financial_lookup"].tool_func({"toolUseId": "call-1", "input": {"period": "2025"}})
            self.messages.append({"role": "user", "content": [{"toolResult": lookup}]})
            if self.mode != "budget":
                final = await tools["submit_final"].tool_func({"toolUseId": "call-2", "input": {"proof": {"answer": "42"}}})
                self.messages.append({"role": "user", "content": [{"toolResult": final}]})
        self.messages.append({"role": "assistant", "content": [{"text": "finished"}]})
        return types.SimpleNamespace(message=self.messages[-1], stop_reason="cancelled" if self.cancelled else "end_turn")


def fake_sdk():
    modules = {name: types.ModuleType(name) for name in (
        "strands", "strands.hooks", "strands.hooks.events", "strands.models", "strands.models.openai",
        "strands.tools", "strands.tools.executors", "strands.tools.tools",
    )}
    modules["strands"].Agent = FakeAgent
    modules["strands.models.openai"].OpenAIModel = FakeModel
    modules["strands.tools.tools"].PythonAgentTool = FakeTool
    modules["strands.hooks"].HookProvider = type("HookProvider", (), {})
    modules["strands.hooks.events"].BeforeToolCallEvent = type("BeforeToolCallEvent", (), {})
    modules["strands.tools.executors"].SequentialToolExecutor = type("SequentialToolExecutor", (), {})
    return modules


class StrandsRuntimeTests(unittest.TestCase):
    def setUp(self):
        FakeAgent.mode = "complete"
        FakeAgent.instances = []
        FakeModel.instances = []
        self.model = ModelSettings("local-teacher", "http://127.0.0.1:8765/v1", "secret-api-key", temperature=0.3)

    def run_fake(self, bridge, root, *, timeout=1):
        with patch.dict(sys.modules, fake_sdk()):
            return run_strands(bridge, self.model, RuntimeSettings("strands", timeout_seconds=timeout), root)

    def test_routes_native_sdk_tools_to_bridge_and_preserves_schemas(self):
        bridge = FakeBridge()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_fake(bridge, root)
            self.assertEqual(result.status, "completed")
            self.assertEqual([name for name, _args in bridge.calls], ["financial_lookup", "submit_final"])
            agent = FakeAgent.instances[0]
            self.assertEqual([item.tool_name for item in agent.kwargs["tools"]], ["financial_lookup", "submit_final"])
            self.assertEqual(agent.kwargs["tools"][0].tool_spec["inputSchema"]["json"], bridge.tool_definitions[0]["function"]["parameters"])
            self.assertFalse(agent.kwargs["load_tools_from_directory"])
            self.assertEqual(type(agent.kwargs["tool_executor"]).__name__, "SequentialToolExecutor")
            self.assertTrue(agent.cleaned)
            self.assertEqual(FakeModel.instances[0].kwargs["client_args"]["base_url"], "http://127.0.0.1:8765/v1")
            messages = json.loads((root / "agent.messages.json").read_text(encoding="utf-8"))
            self.assertEqual(messages[-1]["role"], "assistant")
            events = (root / "agent.events.jsonl").read_text(encoding="utf-8")
            self.assertIn("model thinking", events)
            self.assertNotIn("secret-api-key", events)
            self.assertNotIn("secret-api-key", json.dumps(result.metadata))

    def test_natural_end_without_final_is_incomplete(self):
        FakeAgent.mode = "natural"
        with TemporaryDirectory() as directory:
            result = self.run_fake(FakeBridge(), Path(directory))
            self.assertEqual(result.status, "incomplete")
            self.assertIn("without submitting", result.error)

    def test_unknown_tool_attempt_is_still_recorded_by_environment(self):
        FakeAgent.mode = "unknown"
        bridge = FakeBridge()
        with TemporaryDirectory() as directory:
            result = self.run_fake(bridge, Path(directory))
            self.assertEqual(result.status, "completed")
            self.assertEqual(bridge.calls[0], ("unregistered", {}))

    def test_environment_budget_does_not_count_as_final(self):
        FakeAgent.mode = "budget"
        bridge = FakeBridge()
        original = bridge.call_tool

        def finish(name, arguments):
            result = original(name, arguments)
            bridge.done = result["done"] = True
            return result

        bridge.call_tool = finish
        with TemporaryDirectory() as directory:
            result = self.run_fake(bridge, Path(directory))
            self.assertEqual(result.status, "budget_exhausted")

    def test_timeout_is_bounded_and_keeps_trace(self):
        FakeAgent.mode = "timeout"
        with TemporaryDirectory() as directory:
            started = time.monotonic()
            result = self.run_fake(FakeBridge(), Path(directory), timeout=0.05)
            self.assertEqual(result.status, "timeout")
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue((Path(directory) / "agent.messages.json").is_file())
            self.assertIn('"status": "timeout"', (Path(directory) / "agent.events.jsonl").read_text())

    def test_exception_retains_messages_and_redacts_provider_error(self):
        FakeAgent.mode = "error"
        with TemporaryDirectory() as directory:
            result = self.run_fake(FakeBridge(), Path(directory))
            self.assertEqual(result.status, "failed")
            self.assertNotIn("secret", result.error)
            self.assertEqual(len(json.loads((Path(directory) / "agent.messages.json").read_text())), 1)
            self.assertTrue(FakeAgent.instances[0].cleaned)

    def test_optional_dependency_failure_is_explicit(self):
        with TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"strands": None}):
                result = run_strands(FakeBridge(), self.model, RuntimeSettings("strands"), Path(directory))
            self.assertEqual(result.status, "failed")
            self.assertIn("mint-synth[strands]", result.error)


if __name__ == "__main__":
    unittest.main()
