from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

from .util import normalize_text, stable_id


_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "plc", "lp", "group", "holdings", "holding", "sa", "ag", "nv", "spa",
    "股份有限公司", "有限责任公司", "有限公司", "集团", "控股",
}
_CJK_LEGAL_SUFFIXES = ("股份有限公司", "有限责任公司", "有限公司", "集团", "控股")


def normalized_entity_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_block_key(value: Any) -> str:
    normalized = normalized_entity_name(value)
    compact_cjk = normalized.replace(" ", "")
    for suffix in _CJK_LEGAL_SUFFIXES:
        if compact_cjk.endswith(suffix) and len(compact_cjk) > len(suffix):
            return compact_cjk[: -len(suffix)]
    tokens = normalized.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    compact = " ".join(tokens).strip()
    return compact or normalized


def entity_bucket(value: str, count: int) -> int:
    digest = stable_id("bucket", value, length=16).split("_", 1)[1]
    return int(digest, 16) % count


def load_aliases(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to load entity aliases") from exc
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"entity alias file does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("entity alias file must be a YAML mapping")
    raw = raw.get("aliases", raw)
    if not isinstance(raw, dict):
        raise ValueError("entity alias file aliases must be a mapping")
    result: dict[str, dict[str, str]] = {}
    for alias, target in raw.items():
        normalized = normalized_entity_name(alias)
        if not normalized:
            raise ValueError("entity alias keys must not be empty after normalization")
        if isinstance(target, str):
            canonical_name = target.strip()
            canonical_id = stable_id("entity", normalized_entity_name(target), length=32)
        elif isinstance(target, dict):
            canonical_name = str(target.get("canonical_name") or target.get("name") or alias).strip()
            canonical_id = str(target.get("canonical_id") or stable_id("entity", normalized_entity_name(canonical_name), length=32))
        else:
            raise ValueError(f"alias {alias!r} must map to a string or mapping")
        if not canonical_id.strip() or not canonical_name:
            raise ValueError(f"alias {alias!r} has an empty canonical id or name")
        resolution = {"canonical_entity_id": canonical_id, "canonical_entity_name": canonical_name}
        existing = result.get(normalized)
        if existing is not None and existing != resolution:
            raise ValueError(f"aliases that normalize to {normalized!r} have conflicting targets")
        result[normalized] = resolution
    return result


def identifier_for_fact(fact: Mapping[str, Any], identifier_fields: Iterable[str]) -> tuple[str, str] | None:
    metadata = fact.get("source_metadata")
    if not isinstance(metadata, Mapping):
        return None
    for field in identifier_fields:
        value = metadata.get(field)
        if value is not None and str(value).strip():
            return str(field), normalize_text(value)
    return None


def resolution_priority(method: str) -> int:
    """Order deterministic entity evidence: identifiers, aliases, then normalized names."""
    if method.startswith("identifier:"):
        return 2
    if method == "alias":
        return 1
    return 0


def rule_resolution(
    fact: Mapping[str, Any],
    *,
    aliases: Mapping[str, Mapping[str, str]],
    identifier_fields: Iterable[str],
) -> dict[str, str]:
    surface = str(fact.get("entity_surface") or fact.get("entity") or "").strip()
    if not surface:
        raise ValueError("fact has no entity surface")
    identifier = identifier_for_fact(fact, identifier_fields)
    if identifier:
        field, value = identifier
        return {
            "canonical_entity_id": stable_id("entity", field, value, length=32),
            "canonical_entity_name": surface,
            "resolution_method": f"identifier:{field}",
        }
    alias = aliases.get(normalized_entity_name(surface))
    if alias:
        return {
            "canonical_entity_id": str(alias["canonical_entity_id"]),
            "canonical_entity_name": str(alias["canonical_entity_name"]),
            "resolution_method": "alias",
        }
    block = entity_block_key(surface)
    return {
        "canonical_entity_id": stable_id("entity", block, length=32),
        "canonical_entity_name": surface,
        "resolution_method": "normalized_name",
    }


def validate_llm_clusters(
    response: Any,
    surfaces: Iterable[str],
) -> dict[str, tuple[str, str]]:
    """Accept only a total, non-overlapping partition of the supplied surfaces."""
    expected = {str(value) for value in surfaces}
    clusters = response.get("clusters") if isinstance(response, dict) else None
    if not isinstance(clusters, list):
        raise ValueError("resolver response.clusters must be a list")
    assigned: dict[str, tuple[str, str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            raise ValueError("resolver cluster must be an object")
        members = cluster.get("members")
        canonical_name = cluster.get("canonical_name")
        if not isinstance(members, list) or not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ValueError("resolver cluster requires members and canonical_name")
        canonical_id = stable_id("entity", entity_block_key(canonical_name), sorted(members), length=32)
        for member in members:
            if not isinstance(member, str) or member not in expected or member in assigned:
                raise ValueError("resolver returned unknown or duplicate surface")
            assigned[member] = (canonical_id, canonical_name.strip())
    if set(assigned) != expected:
        raise ValueError("resolver response does not cover every supplied surface")
    return assigned
