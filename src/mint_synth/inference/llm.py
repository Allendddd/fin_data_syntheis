from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


class LLMHTTPError(LLMError):
    def __init__(self, status_code: int, detail: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status_code} from model API: {detail[:2000]}")
        self.status_code = status_code
        self.retry_after = retry_after


class JsonLLM(Protocol):
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any] | list[Any]: ...


def parse_json_payload(content: Any) -> dict[str, Any] | list[Any]:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        raise LLMError("model response content is not text or JSON")
    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            text = text[first_newline + 1 : last_fence].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = sorted(index for index in (text.find("{"), text.find("[")) if index >= 0)
        if not starts:
            raise LLMError("model response does not contain a JSON object or array")
        try:
            value, _ = decoder.raw_decode(text[starts[0] :])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise LLMError("model JSON response must be an object or array")
    return value


@dataclass(frozen=True)
class ClientConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    retries: int = 4
    backoff_seconds: float = 1.0
    strict_json: bool = True
    seed: int | None = None
    max_connections: int = 16


class OpenAICompatibleClient:
    """Thread-safe OpenAI-compatible client with pooled HTTP and bounded retries."""

    def __init__(self, config: ClientConfig):
        if not config.model:
            raise ValueError("model is required")
        self.config = config
        self._http_client: Any = None
        try:
            import httpx

            self._http_client = httpx.Client(
                timeout=config.timeout_seconds,
                limits=httpx.Limits(
                    max_connections=config.max_connections,
                    max_keepalive_connections=config.max_connections,
                ),
            )
        except ImportError:
            # Keeps pure unit tests and legacy use functional; v2 declares httpx as a dependency.
            self._http_client = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests": 0,
            "attempts": 0,
            "successful_requests": 0,
            "terminal_failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _increment(self, **values: int) -> None:
        with self._stats_lock:
            for key, value in values.items():
                self._stats[key] = self._stats.get(key, 0) + int(value)

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()

    def _request_json(self, encoded: bytes, headers: dict[str, str]) -> Any:
        if self._http_client is not None:
            try:
                response = self._http_client.post(self.endpoint, content=encoded, headers=headers)
            except Exception as exc:
                raise LLMError(f"model API transport error: {exc}") from exc
            if response.status_code >= 400:
                retry_after: float | None = None
                raw_retry_after = response.headers.get("Retry-After")
                if raw_retry_after:
                    try:
                        retry_after = float(raw_retry_after)
                    except ValueError:
                        retry_after = None
                raise LLMHTTPError(response.status_code, response.text, retry_after)
            try:
                return response.json()
            except Exception as exc:
                raise LLMError(f"model API returned invalid response JSON: {exc}") from exc

        request = urllib.request.Request(self.endpoint, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retry_after: float | None = None
            raw_retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if raw_retry_after:
                try:
                    retry_after = float(raw_retry_after)
                except ValueError:
                    retry_after = None
            raise LLMHTTPError(exc.code, detail, retry_after) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"model API transport error: {exc}") from exc

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any] | list[Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.strict_json:
            payload["response_format"] = {"type": "json_object"}
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error: BaseException | None = None
        self._increment(requests=1)
        for attempt in range(self.config.retries + 1):
            self._increment(attempts=1)
            retry_after: float | None = None
            try:
                body = self._request_json(encoded, headers)
                choices = body.get("choices") if isinstance(body, dict) else None
                if not isinstance(choices, list) or not choices:
                    raise LLMError("chat completion response has no choices")
                message = choices[0].get("message", {})
                content = message.get("content")
                if content in (None, ""):
                    content = message.get("reasoning_content")
                parsed = parse_json_payload(content)
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                self._increment(
                    successful_requests=1,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )
                return parsed
            except LLMHTTPError as exc:
                last_error = exc
                retry_after = exc.retry_after
                if exc.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (json.JSONDecodeError, LLMError) as exc:
                last_error = exc
            if attempt < self.config.retries:
                delay = retry_after if retry_after is not None else self.config.backoff_seconds * (2**attempt) + random.random()
                delay = min(60.0, max(0.0, delay))
                time.sleep(delay)
        self._increment(terminal_failures=1)
        raise LLMError(f"model request failed after retries: {last_error}") from last_error
