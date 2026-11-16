"""Select the normal final reply from Codex's native JSONL event stream."""

from __future__ import annotations

import json
from pathlib import Path

from .trajectory import TrajectoryError


def native_final_reply(events_path: Path) -> tuple[str, str | None]:
    """Use the last completed root turn; never choose a correct earlier draft.

    Codex's ``item.id`` belongs to its CLI event stream and is not a model
    response ID. Only an explicit ``message_id`` can serve as that optional ID.
    The collector must independently bind this reply to the HTTP capture.
    """
    active = False
    final = None
    candidate = None
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TrajectoryError("Codex native event is not an object")
            kind = event.get("type")
            if kind in {"error", "turn.failed"}:
                raise TrajectoryError("Codex native did not finish successfully")
            if kind == "turn.started":
                if active:
                    raise TrajectoryError("Codex native has overlapping root turns")
                active, candidate, final = True, None, None
            elif kind == "turn.completed":
                if not active or candidate is None:
                    raise TrajectoryError("Codex native completed without a final root reply")
                final, active = candidate, False
            elif isinstance(kind, str) and kind.startswith("item."):
                item = event.get("item")
                if not isinstance(item, dict) or not active:
                    raise TrajectoryError("Codex native item is not bound to an active root turn")
                item_kind = item.get("type")
                if item_kind == "mcp_tool_call":
                    raise TrajectoryError("Codex native emitted an MCP tool call")
                if item_kind in {"command_execution", "file_change", "web_search"}:
                    candidate = None
                elif kind == "item.completed" and item_kind == "agent_message":
                    text = item.get("text")
                    if (not isinstance(text, str) or not text.strip()
                            or item.get("phase") in {"commentary", "partial_answer"}):
                        candidate = None
                    else:
                        message_id = item.get("message_id")
                        candidate = (text.strip(), message_id if isinstance(message_id, str) and message_id else None)
    if active or final is None:
        raise TrajectoryError("Codex native has no completed final root turn")
    return final
