"""Lossless model records and the separately parsed native environment action."""

from __future__ import annotations

import json
from typing import Any, Mapping


EXCHANGE_VERSION = 2
TEXT_EXCHANGE_VERSION = 1


def text_parts(message: Mapping[str, Any]) -> list[dict[str, str]]:
    """Preserve visible reasoning and answer text without parsing a JSON action."""
    if message.get("tool_calls"):
        raise ValueError("text completion received tool calls instead of an answer")
    parts = []
    for field in ("reasoning_content", "reasoning", "analysis", "content"):
        value = message.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"unsupported text completion {field} format")
        if value:
            parts.append({"field": field, "text": value})
    if message.get("reasoning_details"):
        raise ValueError("text reasoning_details needs an explicit content adapter")
    if not parts or not any(part["text"].strip() for part in parts):
        raise ValueError("model returned empty answer text")
    return parts


def native_content(content: Any) -> tuple[dict[str, Any], str, str]:
    """Find the action without treating surrounding model text as disposable."""
    if isinstance(content, dict):
        return content, "", ""
    if not isinstance(content, str):
        raise ValueError("native action response must contain text or an object")
    text = content
    # Code fences around the action are protocol formatting, not analysis.
    if text.strip().startswith("```") and text.strip().endswith("```"):
        fenced = text.strip()
        newline = fenced.find("\n")
        if newline >= 0:
            text = fenced[newline + 1:-3]
    decoder = json.JSONDecoder()
    position = 0
    candidates = []
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        end = start + length
        if isinstance(value, dict) and ("tool" in value) != ("final" in value):
            candidates.append((value, start, end))
        position = end
    if len(candidates) != 1:
        raise ValueError("native response must contain exactly one tool or final action")
    action, start, end = candidates[0]
    return action, text[:start], text[end:]


def native_parts(message: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if message.get("tool_calls"):
        raise ValueError("native JSON action mode received protocol tool calls instead of a JSON action")
    action, before, after = native_content(message.get("content"))
    parts: list[dict[str, Any]] = []
    for field in ("reasoning_content", "reasoning", "analysis"):
        reasoning = message.get(field)
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError(f"unsupported native {field} format")
            if reasoning:
                parts.append({"kind": "reasoning", "text": reasoning, "field": "response.choices[0].message." + field})
    if message.get("reasoning_details"):
        raise ValueError("native reasoning_details needs an explicit content adapter")
    if before.strip():
        parts.append({"kind": "text", "text": before, "field": "response.choices[0].message.content",
                      "segment": "before_action"})
    if "memory" in action:
        # Preserve the teacher's update, not the harness's accumulated memory.
        parts.append({"kind": "memory", "value": action["memory"], "field": "parsed_response.memory"})
    if "final" in action:
        key = "text" if isinstance(action["final"], str) else "proof"
        parts.append({"kind": "final", key: action["final"], "field": "parsed_response.final"})
    else:
        parts.append({"kind": "tool_call", "name": action.get("tool"),
                      "arguments": action.get("arguments", {}), "field": "parsed_response"})
    if after.strip():
        parts.append({"kind": "text", "text": after, "field": "response.choices[0].message.content",
                      "segment": "after_action"})
    return action, parts
