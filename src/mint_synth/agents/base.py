from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class ModelSettings:
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 0.2
    max_tokens: int = 2400
    timeout_seconds: float = 120.0

    @property
    def origin(self) -> str:
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("agent endpoint must be an HTTP(S) URL without embedded credentials")
        if parts.query or parts.fragment or parts.path.rstrip("/") not in {"", "/v1"}:
            raise ValueError("agent endpoint must be an origin or end in /v1")
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


@dataclass(frozen=True)
class ClaudeCodeSettings:
    mode: str = "finance"
    tools: tuple[str, ...] = ("default",)
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"finance", "native"}:
            raise ValueError("Claude Code mode must be finance or native")
        for values in (self.tools, self.allowed_tools):
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value.strip() or value.startswith("-") for value in values):
                raise ValueError("Claude Code tools must be tuples of nonempty strings")
        if "default" in self.tools and len(self.tools) != 1:
            raise ValueError("Claude Code default tool preset cannot be combined with explicit names")
        if self.mode == "native":
            if any(not re.fullmatch(r"default|[A-Z][A-Za-z0-9_]*", name) for name in self.tools):
                raise ValueError("Claude Code native mode accepts only built-in tool names")
            if any(not re.fullmatch(r"[A-Z][A-Za-z0-9_]*(?:\([^\r\n]*\))?", rule) for rule in self.allowed_tools):
                raise ValueError("Claude Code native mode accepts only built-in tool permission rules")


@dataclass(frozen=True)
class CodexSettings:
    mode: str = "finance"
    sandbox: str = "workspace-write"

    def __post_init__(self) -> None:
        if self.mode not in {"finance", "native"}:
            raise ValueError("Codex mode must be finance or native")
        if self.sandbox not in {"read-only", "workspace-write"}:
            raise ValueError("Codex native sandbox must be read-only or workspace-write")


@dataclass(frozen=True)
class StrandsSettings:
    mode: str = "finance"
    tools: tuple[str, ...] = ("file_read", "shell")

    def __post_init__(self) -> None:
        if self.mode not in {"finance", "native"}:
            raise ValueError("Strands mode must be finance or native")
        if (not isinstance(self.tools, tuple) or not self.tools
                or len(set(self.tools)) != len(self.tools)
                or any(name not in {"file_read", "file_write", "shell", "python_repl"} for name in self.tools)):
            raise ValueError("Strands native tools must be unique supported official tool names")


@dataclass(frozen=True)
class RuntimeSettings:
    name: str
    command: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    timeout_seconds: float = 600.0
    claude_code: ClaudeCodeSettings = field(default_factory=ClaudeCodeSettings)
    codex: CodexSettings = field(default_factory=CodexSettings)
    strands: StrandsSettings = field(default_factory=StrandsSettings)

    @property
    def native_tools(self) -> bool:
        return self.name in {"claude_code", "codex", "strands"} and getattr(self, self.name).mode == "native"

    @property
    def tool_mode(self) -> str:
        return getattr(self, self.name).mode if self.name in {"claude_code", "codex", "strands"} else "finance"


@dataclass
class RuntimeResult:
    status: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def settings_for_role(config: Mapping[str, Any], role: str) -> tuple[ModelSettings, RuntimeSettings]:
    training = config["training"]
    configured = training.get("agent_runtime", {})
    name = configured.get("by_role", {}).get(role, configured.get("default", "native"))
    if name not in {"native", "strands", "codex", "claude_code"}:
        raise ValueError(f"unsupported teacher runtime: {name}")
    llm = config["llm"]
    claude = configured.get("claude_code", {})
    role_config = llm["roles"][role]
    endpoint = llm["endpoints"][role_config["endpoint"]]
    api_key = os.getenv(endpoint["api_key_env"])
    if endpoint.get("require_api_key", True) and not api_key:
        raise ValueError(f"required API key environment variable is not set: {endpoint['api_key_env']}")
    return ModelSettings(
        model=role_config["model"], base_url=endpoint["base_url"], api_key=api_key,
        temperature=float(role_config.get("temperature", 0.0)),
        max_tokens=int(role_config.get("max_tokens") or 2400),
        timeout_seconds=float(llm.get("timeout_seconds", 120)),
    ), RuntimeSettings(
        name=name,
        command=tuple(configured.get("commands", {}).get(name, [])),
        extra_args=(),
        timeout_seconds=float(configured.get("timeout_seconds", 600)),
        claude_code=ClaudeCodeSettings(mode=claude.get("mode", "finance"),
                                       tools=tuple(claude.get("tools", ["default"])),
                                       allowed_tools=tuple(claude.get("allowed_tools", []))),
        codex=CodexSettings(**configured.get("codex", {})),
        strands=StrandsSettings(mode=configured.get("strands", {}).get("mode", "finance"),
                                tools=tuple(configured.get("strands", {}).get("tools", ["file_read", "shell"]))),
    )
