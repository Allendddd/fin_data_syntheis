"""Expose public research resources without exposing a hidden task or evaluator."""

from __future__ import annotations

import json
import threading

from mint_synth.harness.resources import ResourceTools
from mint_synth.harness.registry import load_tool_registry


class ResearchBridge:
    def __init__(self, case, *, factory=None):
        self.case = case
        self.registry = load_tool_registry(factory, retrieval=case.retrieval, native_resources=True)
        self.tools = ResourceTools(case.retrieval, case.cutoff, entity_id=case.entity_id,
                                   dated_only=case.dated_only, archive=case.archive_path)
        self.lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        with self.lock:
            self.tools.close()

    @property
    def tool_definitions(self):
        return self.registry.definitions

    def call_tool(self, name, arguments):
        with self.lock:
            observation = self.registry.call(self.tools, name, arguments)
            if self.case.archive_path:
                path = self.case.archive_path.parent / "resource-tools.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"name": name, "arguments": arguments, "observation": observation}, ensure_ascii=False) + "\n")
            return {"observation": observation}
