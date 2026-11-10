"""Run a Strands agent against the same private financial environment as other runtimes.

Strands is an optional dependency. Model HTTP requests are recorded by the
collector's outer proxy; this module saves SDK events and message history.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mint_synth.core.run_state import atomic_json

from .base import ModelSettings, RuntimeResult, RuntimeSettings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    """Keep SDK event data without serializing client objects or credentials."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "message") and hasattr(value, "stop_reason"):
        return {"message": _json_value(value.message), "stop_reason": _json_value(value.stop_reason)}
    return {"python_type": type(value).__name__}


class _Recorder:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            root.chmod(0o700)
        descriptor = os.open(root / "agent.events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._events = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._closed = False

    def event(self, kind: str, value: Any) -> None:
        with self._lock:
            if not self._closed:
                self._events.write(json.dumps({"observed_at": _now(), "kind": kind, "data": _json_value(value)}, ensure_ascii=False) + "\n")
                self._events.flush()

    def close(self, messages: Any) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                atomic_json(self.root / "agent.messages.json", _json_value(messages))
                self._events.flush()
                os.fsync(self._events.fileno())
                self._events.close()


async def _daemon_call(function: Callable, *args: Any) -> Any:
    """Avoid blocking the SDK event loop on a synchronous environment call.

    This intentionally does not use asyncio's default executor: an abandoned
    Python tool must not make event-loop shutdown wait indefinitely.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def deliver(result: Any, error: BaseException | None) -> None:
        if not future.done():
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

    def work() -> None:
        try:
            result, error = function(*args), None
        except BaseException as exc:
            result, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, result, error)
        except RuntimeError:
            # The outer runtime deadline may already have closed this loop.
            pass

    threading.Thread(target=work, name="strands-financial-tool", daemon=True).start()
    return await future


def run_strands(bridge: Any, model: ModelSettings, settings: RuntimeSettings, trace_dir: Path) -> RuntimeResult:
    recorder = _Recorder(Path(trace_dir))
    shared: dict[str, Any] = {"messages": [], "submitted_final": False, "environment_done": False}
    completed = threading.Event()
    metadata = {"runtime": "strands", "messages_file": "agent.messages.json", "events_file": "agent.events.jsonl"}

    if settings.timeout_seconds <= 0:
        recorder.close([])
        return RuntimeResult("failed", "Strands timeout must be positive", metadata)

    async def invoke() -> RuntimeResult:
        try:
            from strands import Agent
            from strands.hooks import HookProvider
            from strands.hooks.events import BeforeToolCallEvent
            from strands.models.openai import OpenAIModel
            from strands.tools.executors import SequentialToolExecutor
            from strands.tools.tools import PythonAgentTool
        except ImportError:
            return RuntimeResult("failed", "Strands runtime requires the optional dependency: pip install 'mint-synth[strands]'", metadata)

        if settings.command or settings.extra_args:
            return RuntimeResult("failed", "command and extra_args are only supported by CLI agent runtimes", metadata)

        sdk_model = OpenAIModel(
            model_id=model.model,
            client_args={
                "base_url": model.origin + "/v1",
                "api_key": model.api_key or "local-model",
                "timeout": model.timeout_seconds,
                "max_retries": 0,
            },
            params={"temperature": model.temperature, "max_tokens": model.max_tokens},
        )
        tools = []

        def tool_for(definition: dict[str, Any]) -> Any:
            function = definition["function"]
            name = function["name"]

            async def call(tool_use: dict[str, Any], **_invocation_state: Any) -> dict[str, Any]:
                result = await _daemon_call(bridge.call_tool, name, tool_use.get("input", {}))
                recorder.event("financial_tool_result", {"tool_use": tool_use, "result": result})
                observation = result.get("observation")
                failed = "error" in result or isinstance(observation, dict) and "error" in observation
                if result.get("done"):
                    shared["environment_done"] = True
                    shared["submitted_final"] = name == "submit_final" and not failed
                    agent = shared.get("agent")
                    cancel = getattr(agent, "cancel", None)
                    if callable(cancel):
                        cancel()
                    else:
                        # Older SDKs lack cooperative cancellation. The raw
                        # environment result has already been recorded above.
                        asyncio.get_running_loop().call_soon(shared["task"].cancel)
                return {
                    "toolUseId": tool_use["toolUseId"],
                    "status": "error" if failed else "success",
                    "content": [{"text": json.dumps(result, ensure_ascii=False, allow_nan=False)}],
                }

            return PythonAgentTool(
                tool_name=name,
                tool_spec={"name": name, "description": function.get("description", ""),
                           "inputSchema": {"json": copy.deepcopy(function["parameters"])}},
                tool_func=call,
            )

        for definition in bridge.tool_definitions:
            tools.append(tool_for(definition))

        class FinancialToolGuard(HookProvider):
            def register_hooks(self, registry: Any, **_kwargs: Any) -> None:
                registry.add_callback(BeforeToolCallEvent, self.before_tool)

            def before_tool(self, event: Any) -> None:
                if event.selected_tool is None:
                    # Preserve invalid tool attempts in the environment trace.
                    # The SDK would otherwise produce its own error without
                    # updating the financial action-validity gate.
                    name = event.tool_use.get("name", "")
                    event.selected_tool = tool_for({"function": {
                        "name": name, "description": "Unregistered tool attempt",
                        "parameters": {"type": "object"},
                    }})

        def callback(*args: Any, **kwargs: Any) -> None:
            recorder.event("sdk", kwargs if not args else {"args": args, "kwargs": kwargs})

        agent = Agent(
            model=sdk_model,
            tools=tools,
            system_prompt=bridge.system_prompt,
            callback_handler=callback,
            tool_executor=SequentialToolExecutor(),
            hooks=[FinancialToolGuard()],
            load_tools_from_directory=False,
            retry_strategy=None,
        )
        shared["agent"] = agent
        try:
            initial = await _daemon_call(bridge.public_state)
            task = asyncio.create_task(agent.invoke_async(json.dumps(initial, ensure_ascii=False)))
            shared["task"] = task
            result = await task
            stop_reason = getattr(result, "stop_reason", None)
            if stop_reason is not None:
                metadata["stop_reason"] = str(stop_reason)
            recorder.event("sdk_result", result)
        except asyncio.CancelledError:
            if not shared["environment_done"]:
                raise
        finally:
            shared["messages"] = _json_value(list(getattr(agent, "messages", [])))
            try:
                shutdown = getattr(agent, "shutdown_async", None)
                if callable(shutdown):
                    await shutdown()
                else:
                    cleanup = getattr(agent, "cleanup", None)
                    if callable(cleanup):
                        cleaned = await _daemon_call(cleanup)
                        if inspect.isawaitable(cleaned):
                            await cleaned
            except Exception as exc:
                recorder.event("cleanup_error", {"error_type": type(exc).__name__})
        if shared["environment_done"]:
            if shared["submitted_final"]:
                return RuntimeResult("completed", metadata=metadata)
            return RuntimeResult("budget_exhausted", "financial tool budget exhausted before final submission", metadata)
        return RuntimeResult("incomplete", "Strands ended without submitting a final proof", metadata)

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        shared["loop"] = loop
        task = loop.create_task(invoke())
        shared["outer_task"] = task
        try:
            shared["result"] = loop.run_until_complete(task)
        except asyncio.CancelledError:
            shared["result"] = RuntimeResult("timeout", "Strands exceeded the rollout deadline", metadata)
        except Exception as exc:
            # Provider exception strings can contain credentials and response
            # bodies. The proxy and SDK event files hold the request context.
            recorder.event("runtime_error", {"error_type": type(exc).__name__})
            shared["result"] = RuntimeResult("failed", f"Strands runtime failed ({type(exc).__name__})", metadata)
        finally:
            loop.close()
            completed.set()

    recorder.event("runtime_started", {"runtime": "strands", "model": model.model})
    thread = threading.Thread(target=worker, name="strands-runtime", daemon=True)
    thread.start()
    if not completed.wait(settings.timeout_seconds):
        loop, task = shared.get("loop"), shared.get("outer_task")
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        completed.wait(0.1)
        result = RuntimeResult("timeout", "Strands exceeded the rollout deadline", metadata)
    else:
        result = shared["result"]
    # A cancelled or stuck tool may outlive the deadline. Copy history once,
    # then close the recorder so late SDK callbacks cannot alter the trace.
    agent = shared.get("agent")
    if agent is not None:
        try:
            shared["messages"] = _json_value(list(agent.messages))
        except Exception:
            pass
    recorder.event("runtime_finished", {"status": result.status, "error": result.error})
    recorder.close(shared["messages"])
    return result
