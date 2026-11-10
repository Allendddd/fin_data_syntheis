from __future__ import annotations

import os
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
class RuntimeSettings:
    name: str
    command: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    timeout_seconds: float = 600.0


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
    )
