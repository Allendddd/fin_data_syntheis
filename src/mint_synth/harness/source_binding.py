"""Bind acquired originals to the hidden source pack without exposing that pack."""

from __future__ import annotations

from pathlib import Path
import re
from .resources import canonical_url
from mint_synth.core.util import normalize_text


def reference_sources(task, record):
    metadata = record.get("metadata", {})
    declared = set(metadata.get("source_ids", []))
    if metadata.get("source_id"):
        declared.add(str(metadata["source_id"]))
    matches = set()
    for fact in task.get("evidence", []):
        original = fact.get("source_metadata") or {}
        identity = str(fact.get("source_id", ""))
        if identity in declared:
            matches.add(identity)
        for key in ("url", "original_url"):
            expected = original.get(key)
            actual = record.get("url") or metadata.get("url")
            if expected and actual:
                try:
                    if canonical_url(expected) == canonical_url(actual):
                        matches.add(identity)
                except ValueError:
                    pass
        for key in ("document_sha256", "file_sha256"):
            if original.get(key) and original[key] == record.get("sha256"):
                matches.add(identity)
        expected_path = original.get("original_path") or original.get("document_path")
        actual_path = metadata.get("original_path")
        if expected_path and actual_path and Path(expected_path).is_absolute() and Path(expected_path).resolve() == Path(actual_path).resolve():
            matches.add(identity)
    return matches


def verify_source_bindings(task, review, records, acquired, *, tolerance="1e-6"):
    from .solution import _same_value, _same_dimensions
    bindings = {}
    for fact in review.get("facts", []):
        key = str(fact.get("chunk_id", ""))
        record = records.get(key)
        if record is None:
            return False, "candidate uses evidence absent from actual acquisition records", bindings
        allowed = reference_sources(task, record)
        if allowed:
            bindings[key] = {"method": "source_identity", "source_ids": sorted(allowed)}
            continue
        # Separate corpora need no global crosswalk. Verify task-level financial
        # fact correspondences when the actual source is different.
        references = {str(f.get("fact_id")): f for f in task.get("evidence", [])}
        correspondences = fact.get("reference_bindings", [])
        if not isinstance(correspondences, list) or not correspondences:
            return False, "acquired fact has neither a source identity nor a verified task-level correspondence", bindings
        observed = "\n".join(str(acquired.get(k, "")) for k, r in records.items() if r.get("source_id") == record.get("source_id"))
        for binding in correspondences:
            expected = references.get(str(binding.get("fact_id"))) if isinstance(binding, dict) else None
            if expected is None:
                return False, "fact correspondence names an unknown reference fact", bindings
            expected_value = expected.get("value_num") if expected.get("value_num") is not None else expected.get("value_text")
            if not _same_value(fact.get("value"), expected_value, tolerance) or not _same_dimensions(fact, expected):
                return False, "cross-source fact value or dimensions differ from the reference", bindings
            quotes = binding.get("scope_quotes", [])
            if not quotes or any(not isinstance(q, str) or not q.strip() or q not in observed for q in quotes):
                return False, "cross-source entity/period/metric scope was not actually acquired", bindings
            scope = normalize_text(" ".join(quotes))
            for field in ("entity", "period", "metric"):
                expected_scope = normalize_text(expected.get(field))
                if not expected_scope or normalize_text(binding.get(field)) != expected_scope:
                    return False, f"cross-source {field} correspondence is unverified", bindings
                if field == "period" and not set(re.findall(r"\d{4}", expected_scope)) <= set(re.findall(r"\d{4}", scope)):
                    return False, "cross-source reporting period is absent from observed scope", bindings
            # The semantic reviewer verifies alias/period entailment; exact
            # source spans and numerical/dimension checks remain deterministic.
        bindings[key] = {"method": "financial_fact_correspondence", "reference_bindings": correspondences}
    return True, "task-level evidence correspondence verified", bindings
