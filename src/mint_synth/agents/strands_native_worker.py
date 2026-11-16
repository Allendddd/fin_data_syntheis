"""Private subprocess entry point for official Strands tools.

The worker runs in the ephemeral corpus cwd supplied by the parent. Source
integrity and final proof grading remain outside the SDK and outside this file.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from mint_synth.core.run_state import atomic_json

from .strands_native import NATIVE_TOOLS
from .strands_runtime import _Recorder, _json_value


def final_message(message: Any, stop_reason: Any) -> tuple[str, dict[str, Any]]:
    message = _json_value(message)
    if (stop_reason != "end_turn" or not isinstance(message, dict) or message.get("role") != "assistant"
            or not isinstance(message.get("content"), list)
            or any(not isinstance(block, dict) or "toolUse" in block for block in message["content"])):
        raise ValueError("Strands did not finish with a normal assistant reply")
    text = "\n".join(block["text"] for block in message["content"] if isinstance(block.get("text"), str)).strip()
    if not text:
        raise ValueError("Strands final assistant reply is empty")
    return text, message


async def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["trace_dir"])
    recorder = _Recorder(root)
    agent = None
    outcome: dict[str, Any] = {"status": "failed", "error": "Strands native worker did not start"}
    try:
        names = payload["tools"]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(name not in NATIVE_TOOLS for name in names)):
            raise ValueError("unsupported Strands native tool selection")
        from strands import Agent
        from strands.models.openai import OpenAIModel
        from strands.tools.executors import SequentialToolExecutor

        tools = [importlib.import_module("strands_tools." + name) for name in names]
        model = payload["model"]
        sdk_model = OpenAIModel(model_id=model["model"], client_args={
            "base_url": model["base_url"], "api_key": os.environ.get("MINT_MODEL_API_KEY", "local-model"),
            "timeout": model["timeout_seconds"], "max_retries": 0},
            params={"temperature": model["temperature"], "max_tokens": model["max_tokens"]})

        def callback(*args: Any, **kwargs: Any) -> None:
            recorder.event("sdk", kwargs if not args else {"args": args, "kwargs": kwargs})

        # Keep SDK defaults for system instructions. Supply only official tools;
        # no custom finance functions, hooks, or final-submission tool are added.
        agent = Agent(model=sdk_model, tools=tools, callback_handler=callback,
                      tool_executor=SequentialToolExecutor(), load_tools_from_directory=False, retry_strategy=None)
        recorder.event("runtime_started", {"runtime": "strands", "mode": "native", "tools": names})
        result = await agent.invoke_async(payload["prompt"])
        text, message = final_message(getattr(result, "message", None), getattr(result, "stop_reason", None))
        atomic_json(root / "native-final.json", {"text": text, "message": message})
        recorder.event("sdk_result", result)
        outcome = {"status": "completed", "error": None}
    except ImportError:
        outcome = {"status": "failed", "error": "Strands native mode requires pip install 'mint-synth[strands]' on Linux"}
    except Exception as exc:
        recorder.event("runtime_error", {"error_type": type(exc).__name__})
        outcome = {"status": "failed", "error": f"Strands native worker failed ({type(exc).__name__})"}
    finally:
        messages = _json_value(list(getattr(agent, "messages", []))) if agent is not None else []
        recorder.event("runtime_finished", outcome)
        recorder.close(messages)
        atomic_json(root / "native-worker.json", outcome)
        if agent is not None:
            try:
                shutdown = getattr(agent, "shutdown_async", None)
                if callable(shutdown):
                    await shutdown()
                else:
                    cleanup = getattr(agent, "cleanup", None)
                    if callable(cleanup):
                        result = cleanup()
                        if inspect.isawaitable(result):
                            await result
            except Exception:
                pass
    return outcome


def main() -> int:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    result = asyncio.run(invoke(payload))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
