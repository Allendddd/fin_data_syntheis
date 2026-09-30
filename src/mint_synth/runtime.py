from __future__ import annotations

import os
import hashlib
import json
import sqlite3
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

from .llm import ClientConfig, JsonLLM, OpenAICompatibleClient


T = TypeVar("T")
R = TypeVar("R")


class SqliteResponseCache:
    """Durable, thread-safe request cache scoped to one run/stage/shard/model role."""

    def __init__(self, path: str | Path, *, policy: str, busy_timeout_seconds: float):
        if policy not in {"immutable", "refresh"}:
            raise ValueError(f"unsupported LLM cache policy: {policy}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_seconds,
            check_same_thread=False,
        )
        self._connection.execute(f"PRAGMA busy_timeout={max(1, int(busy_timeout_seconds * 1000))}")
        self._connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "request_key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._connection.commit()

    def get(self, request_key: str) -> dict[str, Any] | list[Any] | None:
        if self.policy == "refresh":
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM responses WHERE request_key = ?", (request_key,)
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"corrupt cached LLM response in {self.path}") from exc
        if not isinstance(value, (dict, list)):
            raise RuntimeError(f"cached LLM response is not an object or list in {self.path}")
        return value

    def put(self, request_key: str, response: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
        payload = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.policy == "immutable":
                self._connection.execute(
                    "INSERT OR IGNORE INTO responses(request_key, response_json) VALUES (?, ?)",
                    (request_key, payload),
                )
            else:
                self._connection.execute(
                    "INSERT OR REPLACE INTO responses(request_key, response_json) VALUES (?, ?)",
                    (request_key, payload),
                )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT response_json FROM responses WHERE request_key = ?", (request_key,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to persist cached LLM response in {self.path}")
        stored = json.loads(row[0])
        if not isinstance(stored, (dict, list)):
            raise RuntimeError(f"cached LLM response is not an object or list in {self.path}")
        return stored

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class ConfiguredRoleClient:
    def __init__(
        self,
        delegate: OpenAICompatibleClient,
        *,
        temperature: float,
        max_tokens: int | None,
        cache: SqliteResponseCache | None = None,
        cache_namespace: Mapping[str, Any] | None = None,
    ):
        self.delegate = delegate
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.cache_namespace = dict(cache_namespace or {})
        self._cache_stats_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any] | list[Any]:
        del temperature
        effective_max_tokens = self.max_tokens if self.max_tokens is not None else max_tokens
        material = {
            "namespace": self.cache_namespace,
            "system": system,
            "user": user,
            "temperature": self.temperature,
            "max_tokens": effective_max_tokens,
        }
        request_key = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if self.cache is not None:
            cached = self.cache.get(request_key)
            if cached is not None:
                with self._cache_stats_lock:
                    self._cache_hits += 1
                return cached
            with self._cache_stats_lock:
                self._cache_misses += 1
        response = self.delegate.complete_json(
            system=system,
            user=user,
            temperature=self.temperature,
            max_tokens=effective_max_tokens,
        )
        if self.cache is not None:
            response = self.cache.put(request_key, response)
        return response

    def stats(self) -> dict[str, int]:
        values = self.delegate.stats()
        with self._cache_stats_lock:
            values.update(cache_hits=self._cache_hits, cache_misses=self._cache_misses)
        return values

    def close(self) -> None:
        try:
            self.delegate.close()
        finally:
            if self.cache is not None:
                self.cache.close()


def client_for_role(
    config: Mapping[str, Any],
    role: str,
    *,
    cache_path: str | Path | None = None,
) -> JsonLLM:
    llm = config["llm"]
    role_config = llm["roles"][role]
    endpoint = llm["endpoints"][role_config["endpoint"]]
    api_key = os.getenv(str(endpoint["api_key_env"]))
    if endpoint.get("require_api_key", True) and not api_key:
        raise ValueError(f"required API key environment variable is not set: {endpoint['api_key_env']}")
    delegate = OpenAICompatibleClient(
        ClientConfig(
            model=str(role_config["model"]),
            base_url=str(endpoint["base_url"]),
            api_key=api_key,
            timeout_seconds=float(llm["timeout_seconds"]),
            retries=int(llm["retries"]),
            backoff_seconds=float(llm.get("backoff_seconds", 1.0)),
            strict_json=bool(endpoint.get("strict_json", True)),
            seed=int(config["run"]["random_seed"]) if endpoint.get("send_seed", False) else None,
            max_connections=int(llm["concurrency_per_worker"]),
        )
    )
    configured_max_tokens = role_config.get("max_tokens")
    cache_config = llm.get("cache", {})
    cache = None
    if cache_path is not None and cache_config.get("backend", "sqlite") == "sqlite":
        cache = SqliteResponseCache(
            cache_path,
            policy=str(cache_config.get("policy", "immutable")),
            busy_timeout_seconds=float(cache_config.get("busy_timeout_seconds", 60)),
        )
    return ConfiguredRoleClient(
        delegate,
        temperature=float(role_config.get("temperature", 0.0)),
        max_tokens=int(configured_max_tokens) if configured_max_tokens is not None else None,
        cache=cache,
        cache_namespace={
            "role": role,
            "endpoint": role_config["endpoint"],
            "base_url": endpoint["base_url"],
            "model": role_config["model"],
            "revision": role_config.get("revision"),
            "strict_json": endpoint.get("strict_json", True),
            "seed": config["run"]["random_seed"] if endpoint.get("send_seed", False) else None,
        },
    )


def bounded_map(
    function: Callable[[T], R],
    values: Iterable[T],
    *,
    workers: int,
    pending_factor: int = 2,
) -> Iterator[R]:
    """Ordered parallel map with bounded memory on Python 3.10+."""
    iterator = iter(values)
    maximum = max(workers, workers * pending_factor)
    queue: deque[Future[R]] = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(maximum):
            try:
                queue.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while queue:
            future = queue.popleft()
            yield future.result()
            try:
                queue.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
