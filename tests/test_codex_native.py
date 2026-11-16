from __future__ import annotations

import json
from pathlib import Path

import pytest

from mint_synth.agents.codex_events import native_final_reply
from mint_synth.agents.trajectory import TrajectoryError


def events(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def message(text: str, **extra) -> dict:
    return {"type": "item.completed", "item": {"type": "agent_message", "id": "local-item-1", "text": text, **extra}}


def test_final_reply_uses_last_completed_turn_and_never_cli_item_id(tmp_path):
    path = events(tmp_path / "events.jsonl", [
        {"type": "thread.started", "thread_id": "thread"},
        {"type": "turn.started"}, message("A correct earlier answer"), {"type": "turn.completed"},
        {"type": "turn.started"}, message("Final answer"), {"type": "turn.completed"},
    ])
    assert native_final_reply(path) == ("Final answer", None)


def test_explicit_message_id_can_be_returned_without_rewriting_it(tmp_path):
    path = events(tmp_path / "events.jsonl", [
        {"type": "turn.started"}, message("Final answer", message_id="msg-model"), {"type": "turn.completed"},
    ])
    assert native_final_reply(path) == ("Final answer", "msg-model")


@pytest.mark.parametrize("tail", [
    [{"type": "turn.started"}],
    [{"type": "error", "message": "failed"}],
    [{"type": "turn.started"}, {"type": "turn.failed"}],
])
def test_later_incomplete_or_failed_turn_does_not_fall_back(tmp_path, tail):
    path = events(tmp_path / "events.jsonl", [
        {"type": "turn.started"}, message("Earlier answer"), {"type": "turn.completed"}, *tail,
    ])
    with pytest.raises(TrajectoryError):
        native_final_reply(path)


@pytest.mark.parametrize("tool_type", ["command_execution", "file_change", "web_search"])
def test_tools_after_a_draft_require_a_new_final_reply(tmp_path, tool_type):
    path = events(tmp_path / "events.jsonl", [
        {"type": "turn.started"}, message("Earlier draft"),
        {"type": "item.completed", "item": {"type": tool_type}}, {"type": "turn.completed"},
    ])
    with pytest.raises(TrajectoryError, match="without a final"):
        native_final_reply(path)


def test_commentary_is_not_a_final_answer_and_mcp_is_forbidden(tmp_path):
    path = events(tmp_path / "events.jsonl", [
        {"type": "turn.started"}, message("Inspecting the sources", phase="commentary"), {"type": "turn.completed"},
    ])
    with pytest.raises(TrajectoryError, match="without a final"):
        native_final_reply(path)
    events(path, [{"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "mcp_tool_call"}},
        message("Final"), {"type": "turn.completed"}])
    with pytest.raises(TrajectoryError, match="MCP"):
        native_final_reply(path)
