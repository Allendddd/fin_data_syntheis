from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
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
    search_tool: str = "web_search"
    search_query_field: str = "query"
    search_limit_field: str = "limit"

    def __post_init__(self) -> None:
        if self.mode not in {"finance", "native"}:
            raise ValueError("Strands mode must be finance or native")
        if (not isinstance(self.tools, tuple) or not self.tools
                or len(set(self.tools)) != len(self.tools)
                or any(not isinstance(name, str) or not name or not all(part.isidentifier() for part in name.split(".")) for name in self.tools)):
            raise ValueError("Strands native tools must be unique importable tool module names")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.search_tool, self.search_query_field, self.search_limit_field)):
            raise ValueError("Strands search binding must use nonempty tool and parameter names")


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


def runtime_for_retrieval(runtime: RuntimeSettings, retrieval: Mapping[str, Any] | None) -> RuntimeSettings:
    """Web research retains the selected framework's native action space."""
    if retrieval and retrieval.get("mode") == "web" and runtime.name in {"codex", "claude_code", "strands"}:
        if retrieval.get("web", {}).get("search_endpoint") and runtime.name == "claude_code":
            if "default" not in runtime.claude_code.tools and "WebSearch" not in runtime.claude_code.tools:
                raise ValueError("search_endpoint requires the native WebSearch entry point, which is absent from the explicit Claude Code tools configuration")
        return replace(runtime, **{runtime.name: replace(getattr(runtime, runtime.name), mode="native")})
    return runtime


def search_backend_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe planned bindings without loading credentials or claiming a probe."""
    retrieval = config.get("harness", {}).get("retrieval", {})
    web = retrieval.get("web", {})
    configured = retrieval.get("mode") == "web" and bool(web.get("search_endpoint"))
    result = {"configured": configured, "protocol": web.get("search_protocol", "tavily") if configured else None,
              "bindings": [], "runtime_verification": "pending" if configured else "not_required"}
    training = config.get("training", {})
    if not configured or not training.get("enabled", True) or not training.get("long_enabled", True):
        return result
    settings = training.get("agent_runtime", {})
    roles = training.get("long_teacher_roles") or [training.get("teacher_role", "constructor")]
    for role in roles:
        name = settings.get("by_role", {}).get(role, settings.get("default", "codex"))
        if name == "claude_code":
            tools = settings.get("claude_code", {}).get("tools", ["default"])
            if "default" not in tools and "WebSearch" not in tools:
                raise ValueError("search_endpoint cannot bind Claude Code: its explicit tools configuration omits WebSearch")
        entry = {"codex": "web.run", "claude_code": "WebSearch", "strands": settings.get("strands", {}).get("search_tool", "web_search"),
                 "native": "search_web"}[name]
        result["bindings"].append({"role": role, "runtime": name, "search_tool": entry})
    return result


def settings_for_role(config: Mapping[str, Any], role: str) -> tuple[ModelSettings, RuntimeSettings]:
    training = config["training"]
    configured = training.get("agent_runtime", {})
    name = configured.get("by_role", {}).get(role, configured.get("default", "codex"))
    if name not in {"native", "strands", "codex", "claude_code"}:
        raise ValueError(f"unsupported teacher runtime: {name}")
    llm = config["llm"]
    claude = configured.get("claude_code", {})
    role_config = llm["roles"][role]
    endpoint = llm["endpoints"][role_config["endpoint"]]
    api_key = os.getenv(endpoint["api_key_env"])
    if endpoint.get("require_api_key", True) and not api_key:
        raise ValueError(f"required API key environment variable is not set: {endpoint['api_key_env']}")
    model = ModelSettings(
        model=role_config["model"], base_url=endpoint["base_url"], api_key=api_key,
        temperature=float(role_config.get("temperature", 0.0)),
        max_tokens=int(role_config.get("max_tokens") or 2400),
        timeout_seconds=float(llm.get("timeout_seconds", 120)),
    )
    runtime = RuntimeSettings(
        name=name,
        command=tuple(configured.get("commands", {}).get(name, [])),
        extra_args=(),
        timeout_seconds=float(configured.get("timeout_seconds", 600)),
        claude_code=ClaudeCodeSettings(mode=claude.get("mode", "finance"),
                                       tools=tuple(claude.get("tools", ["default"])),
                                       allowed_tools=tuple(claude.get("allowed_tools", []))),
        codex=CodexSettings(**configured.get("codex", {})),
        strands=StrandsSettings(mode=configured.get("strands", {}).get("mode", "finance"),
                                tools=tuple(configured.get("strands", {}).get("tools", ["file_read", "shell"])),
                                search_tool=configured.get("strands", {}).get("search_tool", "web_search"),
                                search_query_field=configured.get("strands", {}).get("search_query_field", "query"),
                                search_limit_field=configured.get("strands", {}).get("search_limit_field", "limit")),
    )
    return model, runtime_for_retrieval(runtime, config.get("harness", {}).get("retrieval"))
