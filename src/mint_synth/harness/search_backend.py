"""HTTP search providers, separate from model-facing tool definitions."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

from mint_synth.core.run_state import atomic_json


SEARCH_PROTOCOLS = {"tavily", "json"}


def validate_search_backend(web):
    from .resources import canonical_url
    endpoint = web.get("search_endpoint")
    if not endpoint:
        return
    canonical_url(endpoint)
    if web.get("search_protocol", "tavily") not in SEARCH_PROTOCOLS:
        raise ValueError("search_protocol must be tavily or json; arbitrary endpoint protocols require an adapter")
    env = web.get("api_key_env")
    if env and not os.getenv(env):
        raise ValueError(f"web search API key environment variable is not set: {env}")


def search(http, web, arguments, *, archive=None):
    """Execute one search against the configured service; never fall back."""
    from .resources import canonical_url
    validate_search_backend(web)
    endpoint = web.get("search_endpoint")
    if not endpoint:
        raise ValueError("no search backend is configured")
    protocol = web.get("search_protocol", "tavily")
    limit = int(arguments.get("limit", 5))
    if limit < 1:
        raise ValueError("search result limit must be positive")
    native_commands = protocol == "json" and isinstance(arguments.get("commands"), dict)
    query = arguments.get("query")
    if not native_commands:
        if not isinstance(query, str):
            raise ValueError("search query must be text")
        query = query.strip()
        if not query:
            raise ValueError("search query is empty")
    if protocol == "tavily":
        extra = set(arguments) - {"query", "limit", "domains", "exclude_domains", "recency"}
        if extra:
            raise ValueError("unsupported parameters for the tavily adapter: " + ", ".join(sorted(extra)))
        body = {"query": query, "max_results": limit, "include_answer": False,
                "include_raw_content": False, "include_published_date": True}
        for source, target in (("domains", "include_domains"), ("exclude_domains", "exclude_domains")):
            if arguments.get(source):
                body[target] = arguments[source]
    else:
        body = {**arguments, "limit": limit}
        if query is not None:
            body["query"] = query
    token = os.getenv(web["api_key_env"]) if web.get("api_key_env") else None
    record = {"request_id": uuid.uuid4().hex, "configured_endpoint": endpoint, "protocol": protocol,
              "request": body, "started_at": datetime.now(timezone.utc).isoformat(), "status": "pending"}
    path = None
    if archive:
        archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = archive / ("backend_" + record["request_id"] + ".json")
        atomic_json(path, record)
    try:
        response = http.post(endpoint, headers={"Authorization": f"Bearer {token}"} if token else {}, json=body)
        record.update(effective_endpoint=str(response.url), http_status=response.status_code,
                      response_sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        payload = response.json()
        record["response"] = payload
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"{protocol} search backend response must contain a results array")
        results = []
        for row in rows[:limit]:
            if not isinstance(row, dict) or not isinstance(row.get("url"), str):
                raise ValueError("search backend result must contain a URL")
            content = row.get("content", row.get("text", row.get("snippet", "")))
            if not isinstance(content, str):
                raise ValueError("search backend result content must be text")
            results.append({"url": canonical_url(row["url"]), "title": str(row.get("title", "")),
                            "content": content, "published_at": row.get("published_date") or row.get("published_at")})
        if protocol == "tavily" and arguments.get("recency") is not None:
            days = int(arguments["recency"])
            if days < 0:
                raise ValueError("search recency must not be negative")
            cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
            filtered = []
            for item in results:
                try:
                    published = date.fromisoformat(str(item["published_at"])[:10])
                except ValueError:
                    continue  # Undated results cannot establish a recency bound.
                if published >= cutoff:
                    filtered.append(item)
            results = filtered
        def in_domains(url, domains):
            host = urlsplit(url).hostname or ""
            return any(host == domain.lower() or host.endswith("." + domain.lower()) for domain in domains)
        if arguments.get("domains"):
            results = [item for item in results if in_domains(item["url"], arguments["domains"])]
        if arguments.get("exclude_domains"):
            results = [item for item in results if not in_domains(item["url"], arguments["exclude_domains"])]
        record["status"] = "completed"
        return {"results": results, "backend_request_id": record["request_id"]}
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        if path:
            atomic_json(path, record)
