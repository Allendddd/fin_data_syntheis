"""Run agent CLIs with either finance MCP or their built-in tools.

The CLIs own their agent loops. Their native JSONL streams and model HTTP calls
are retained independently. Native runs are graded offline from native reads
and the ordinary final reply. CLI switches are not OS-level isolation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mint_synth.core.run_state import atomic_json

from .base import ClaudeCodeSettings, CodexSettings, ModelSettings, RuntimeResult, RuntimeSettings
from .mcp_server import FinanceMCPServer
from .cc_workspace import finish_workspace, native_final_reply, prepare_workspace
from .codex_events import native_final_reply as codex_final_reply


_SECRET_ENV = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
_CODEX_ITEMS = {"agent_message", "reasoning", "mcp_tool_call", "plan", "todo_list"}
_CODEX_NATIVE_ITEMS = {"agent_message", "reasoning", "command_execution", "file_change", "plan", "todo_list"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _private_file(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _environment(home: Path, model: ModelSettings, token: str | None = None) -> dict[str, str]:
    # Avoid inheriting a developer's cloud credentials or another agent's login,
    # model endpoint, config directory, proxy, or runtime options.
    environment = {
        key: value for key, value in os.environ.items()
        if not _SECRET_ENV.search(key)
        and not key.upper().startswith(("ANTHROPIC_", "CLAUDE_", "CODEX_", "OPENAI_"))
        and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
    }
    environment.update({
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "CODEX_HOME": str(home / ".codex"),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "MINT_MODEL_API_KEY": model.api_key or "local-model",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    if token is not None:
        environment["MINT_MCP_TOKEN"] = token
    return environment


def _codex_command(
    prefix: tuple[str, ...], home: Path, workdir: Path, model: ModelSettings,
    bridge: Any, url: str | None, options: CodexSettings | None = None,
) -> list[str]:
    options = options or CodexSettings()
    native = options.mode == "native"
    sandbox = options.sandbox if native else "read-only"
    codex_home = home / ".codex"
    codex_home.mkdir(mode=0o700)
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    instructions_config = []
    if not native:
        instructions = home / "finance-instructions.md"
        _private_file(instructions, bridge.system_prompt)
        instructions_config.append(f"model_instructions_file = {quote(str(instructions))}")
    base_url = model.origin + "/v1"
    configuration_lines = [
        f"model = {quote(model.model)}",
        'model_provider = "mint_finance"',
        *instructions_config,
        'approval_policy = "never"',
        f"sandbox_mode = {quote(sandbox)}",
        'web_search = "disabled"',
        "project_doc_max_bytes = 0",
        "[history]",
        'persistence = "none"',
        "[analytics]",
        "enabled = false",
        "[features]",
        f"shell_tool = {'true' if native else 'false'}",
        "shell_snapshot = false",
        f"view_image = {'true' if native else 'false'}",
        "apps = false",
        "multi_agent = false",
        "memories = false",
        "goals = false",
        "hooks = false",
        "remote_plugin = false",
        "enable_request_compression = false",
        "[skills.bundled]",
        "enabled = false",
        "[model_providers.mint_finance]",
        'name = "Mint finance synthesis"',
        f"base_url = {quote(base_url)}",
        'env_key = "MINT_MODEL_API_KEY"',
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "supports_websockets = false",
        f"stream_idle_timeout_ms = {max(1000, int(model.timeout_seconds * 1000))}",
    ]
    if native:
        configuration_lines.extend([
            "[sandbox_workspace_write]", "network_access = false",
            "[shell_environment_policy]", "ignore_default_excludes = false",
            "[mcp_servers]",
        ])
    else:
        tool_names = [item["function"]["name"] for item in bridge.tool_definitions]
        configuration_lines.extend([
            "[mcp_servers.finance]", f"url = {quote(url)}",
            'bearer_token_env_var = "MINT_MCP_TOKEN"', "required = true",
            f"enabled_tools = {quote(tool_names)}", 'default_tools_approval_mode = "auto"',
            "supports_parallel_tool_calls = false", "startup_timeout_sec = 30",
            f"tool_timeout_sec = {max(1, int(model.timeout_seconds))}",
        ])
    _private_file(codex_home / "config.toml", "\n".join([*configuration_lines, ""]))
    return [*prefix, "exec", "--json", "--ephemeral", "--strict-config", "--ignore-rules",
            "--skip-git-repo-check", "--sandbox", sandbox, "--cd", str(workdir), "-"]


def _claude_command(
    prefix: tuple[str, ...], home: Path, model: ModelSettings, bridge: Any,
    url: str | None, token: str | None, environment: dict[str, str],
    options: ClaudeCodeSettings | None = None,
) -> list[str]:
    options = options or ClaudeCodeSettings()
    (home / ".claude").mkdir(mode=0o700)
    native = options.mode == "native"
    configuration = home / "mcp.json"
    _private_file(configuration, json.dumps({
        "mcpServers": {} if native else {"finance": {"type": "http", "url": url,
                                                        "headers": {"Authorization": f"Bearer {token}"}}},
    }, ensure_ascii=False))
    tool_names = [] if native else ["mcp__finance__" + item["function"]["name"] for item in bridge.tool_definitions]
    environment.update({
        "ANTHROPIC_BASE_URL": model.origin,
        "ANTHROPIC_API_KEY": model.api_key or "local-model",
        "ANTHROPIC_AUTH_TOKEN": model.api_key or "local-model",
        "ANTHROPIC_MODEL": model.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model.model,
        "API_TIMEOUT_MS": str(max(1000, int(model.timeout_seconds * 1000))),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    })
    permissions = (["--allowedTools", *options.allowed_tools] if options.allowed_tools else []) if native else [
        "--allowedTools", ",".join(tool_names)]
    return [
        *prefix, "--print", *([] if native else ["--bare"]),
        "--tools", ",".join(options.tools) if native else "",
        "--strict-mcp-config", "--mcp-config", str(configuration),
        *(["--disallowedTools", "mcp__*"] if native else []),
        *permissions,
        "--permission-mode", "dontAsk",
        *([] if native else ["--disable-slash-commands"]),
        "--no-session-persistence", "--output-format", "stream-json",
        "--verbose", "--include-partial-messages",
        *([] if native else ["--system-prompt", bridge.system_prompt]),
        "--model", model.model,
    ]


def _terminate_tree(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # The CLI can exit before a child does; kill the remaining process group
        # even when wait() has already reaped its original leader.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    process.wait(timeout=5)


def _audit_events(path: Path, runtime: str, tool_names: set[str], *, claude_native: bool = False,
                  codex_native: bool = False) -> tuple[int, str | None]:
    count = 0
    failure = None
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            count += 1
            try:
                event = json.loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise ValueError("event must be an object with a type")
            except (ValueError, UnicodeError):
                failure = failure or "CLI stdout contains a malformed JSONL event"
                continue
            event_type = event["type"]
            if runtime == "codex":
                if event_type in {"error", "turn.failed"}:
                    failure = failure or "Codex reported a failed turn"
                item = event.get("item")
                if event_type.startswith("item.") and isinstance(item, dict):
                    kind = item.get("type")
                    if codex_native and kind == "mcp_tool_call":
                        failure = failure or "Codex native mode emitted an MCP tool call"
                    elif codex_native and kind not in _CODEX_NATIVE_ITEMS:
                        failure = failure or "Codex native mode emitted an unsupported tool event"
                    elif not codex_native and kind not in _CODEX_ITEMS:
                        failure = failure or "Codex emitted a non-finance or unsupported tool event"
                    elif kind == "mcp_tool_call":
                        if item.get("server") != "finance" or item.get("tool") not in tool_names:
                            failure = failure or "Codex called an MCP tool outside the finance tool set"
            else:
                if event_type == "result" and event.get("is_error"):
                    failure = failure or "Claude Code reported an error result"
                message = event.get("message")
                content = message.get("content", []) if isinstance(message, dict) else []
                for block in content if isinstance(content, list) else []:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    allowed = {"mcp__finance__" + name for name in tool_names} | {"EndConversation"}
                    if claude_native:
                        if not isinstance(name, str) or name.startswith("mcp__"):
                            failure = failure or "Claude Code native mode emitted an MCP or invalid tool call"
                        continue
                    if name not in allowed:
                        failure = failure or "Claude Code emitted a non-finance tool call"
    if count == 0:
        failure = failure or "CLI did not emit any JSONL events"
    return count, failure


def run_cli(bridge: Any, model: ModelSettings, settings: RuntimeSettings, trace_dir: Path) -> RuntimeResult:
    # In CLI native mode the first argument is a data-only HarnessCase, not a
    # FinanceToolBridge. No session or custom tool server is constructed.
    if settings.name not in {"codex", "claude_code"}:
        raise ValueError("CLI runtime must be codex or claude_code")
    if settings.extra_args:
        raise ValueError("extra CLI arguments are not supported; configure a dedicated runtime command")
    if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
        raise ValueError("CLI timeout must be a finite positive number")
    if not math.isfinite(model.timeout_seconds) or model.timeout_seconds <= 0:
        raise ValueError("model timeout must be a finite positive number")
    prefix = settings.command or (("codex",) if settings.name == "codex" else ("claude",))
    if not prefix or any(not isinstance(part, str) or not part for part in prefix):
        raise ValueError("CLI command must be a nonempty argument vector")
    model.origin  # Validate the endpoint before launching a child.
    trace_dir = Path(trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        trace_dir.chmod(0o700)
    stdout_path, stderr_path = trace_dir / "stdout.jsonl", trace_dir / "stderr.log"
    metadata_path = trace_dir / "cli.json"
    for path in (stdout_path, stderr_path, metadata_path):
        if path.exists():
            raise ValueError("CLI trace files already exist in trace_dir")
    metadata: dict[str, Any] = {
        "runtime": settings.name,
        "started_at": _now(),
        "events_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "metadata_file": metadata_path.name,
        "sampling_parameters": "runtime-managed",
        "status": "running",
    }
    claude_native = settings.name == "claude_code" and settings.claude_code.mode == "native"
    codex_native = settings.name == "codex" and settings.codex.mode == "native"
    native = claude_native or codex_native
    if settings.name == "claude_code":
        metadata["claude_code_mode"] = settings.claude_code.mode
    elif settings.name == "codex":
        metadata["codex_mode"] = settings.codex.mode
        if codex_native:
            metadata["sandbox"] = settings.codex.sandbox
    atomic_json(metadata_path, metadata)
    started = time.monotonic()
    status, error, exit_code = "failed", None, None
    try:
        with tempfile.TemporaryDirectory(prefix="mint-agent-") as temporary:
            home = Path(temporary)
            workdir = home / "workspace"
            workdir.mkdir(mode=0o700)
            with (nullcontext() if native else FinanceMCPServer(bridge)) as mcp:
                environment = _environment(home, model, None if native else mcp.token)
                if settings.name == "codex":
                    command = _codex_command(prefix, home, workdir, model, bridge, None if native else mcp.url,
                                             settings.codex)
                else:
                    command = _claude_command(prefix, home, model, bridge, None if claude_native else mcp.url,
                                              None if claude_native else mcp.token, environment, settings.claude_code)
                prompt = (prepare_workspace(bridge, workdir, trace_dir, runtime="codex") if codex_native else
                          prepare_workspace(bridge, workdir, trace_dir)) if native else (
                    "Complete the financial query in this public state. Use only the finance MCP tools, "
                    "and finish by calling submit_final with the proof.\n" + json.dumps(bridge.public_state(), ensure_ascii=False))
                with os.fdopen(os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stdout:
                    with os.fdopen(os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stderr:
                        options: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {
                            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                        }
                        try:
                            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                                       cwd=workdir, env=environment, **options)
                            try:
                                process.communicate(input=prompt.encode("utf-8"), timeout=settings.timeout_seconds)
                                exit_code = process.returncode
                            except subprocess.TimeoutExpired:
                                status, error = "timeout", "CLI exceeded its execution deadline"
                                _terminate_tree(process)
                                exit_code = process.returncode
                            except BaseException:
                                _terminate_tree(process)
                                raise
                        finally:
                            if native:
                                finish_workspace(workdir, trace_dir)
                count, audit_error = _audit_events(stdout_path, settings.name, set() if native else {
                    item["function"]["name"] for item in bridge.tool_definitions
                }, claude_native=claude_native, codex_native=codex_native)
                metadata["event_count"] = count
                if status != "timeout":
                    if exit_code != 0:
                        error = "CLI exited with a nonzero status"
                    elif audit_error:
                        error = audit_error
                    elif claude_native:
                        native_final_reply(stdout_path)
                        status = "completed"
                    elif codex_native:
                        final_text, final_id = codex_final_reply(stdout_path)
                        metadata["final_text_sha256"] = hashlib.sha256(final_text.encode()).hexdigest()
                        if final_id is not None:
                            metadata["final_message_id"] = final_id
                        status = "completed"
                    elif bridge.done:
                        status = "completed"
                    else:
                        status, error = "incomplete", "CLI ended without a final environment submission"
    except (OSError, RuntimeError, ValueError) as exc:
        status = "failed"
        error = f"CLI runtime failed ({type(exc).__name__})"
    finally:
        metadata.update(status=status, error=error, exit_code=exit_code,
                        finished_at=_now(), elapsed_seconds=round(time.monotonic() - started, 3))
        atomic_json(metadata_path, metadata)
    return RuntimeResult(status=status, error=error, metadata=metadata)
