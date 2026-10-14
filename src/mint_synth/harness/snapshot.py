from __future__ import annotations

import re
import sqlite3
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping
from mint_synth.storage.parquetio import iter_payloads


TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "search_corpus", "description": "Search the frozen financial disclosure corpus. Returns chunk IDs and snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "minLength": 2}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "open_chunk", "description": "Read the complete text of a retrieved chunk.", "parameters": {"type": "object", "properties": {"chunk_id": {"type": "string", "minLength": 1}}, "required": ["chunk_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a basic decimal operation on supplied numbers.", "parameters": {"type": "object", "properties": {"operation": {"enum": ["add", "subtract", "multiply", "divide", "percent_change", "average"]}, "values": {"type": "array", "items": {"type": ["string", "number"]}, "minItems": 2, "maxItems": 12}}, "required": ["operation", "values"], "additionalProperties": False}}},
]

 
def published_date(metadata: Mapping[str, Any], field: str = "published_at") -> str | None:
    dates: list[str] = []
    for key in {field, "published_at"}:
        raw = metadata.get(key)
        if not raw:
            if key == field:
                return None
            continue
        try:
            dates.append(date.fromisoformat(str(raw)[:10]).isoformat())
        except ValueError:
            if key == field:
                return None
    return max(dates) if dates else None


def build_corpus_index(
    paths: list[Path], destination: Path, *, cutoff: str, batch_rows: int,
    available_at_field: str = "published_at",
) -> dict[str, int]:
    connection = sqlite3.connect(destination)
    counts = {"chunks": 0, "future_excluded": 0, "undated": 0}
    try:
        connection.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, available_at TEXT, text TEXT NOT NULL)")
        for path in paths:
            for chunk in iter_payloads(path, batch_rows=batch_rows):
                published = published_date(chunk.get("source_metadata") or {}, available_at_field)
                if published and published > cutoff:
                    counts["future_excluded"] += 1
                    continue
                if not published:
                    counts["undated"] += 1
                chunk_id = str(chunk["chunk_id"])
                text = str(chunk["text"])
                connection.execute("INSERT INTO chunks VALUES (?,?,?,?)", (chunk_id, str(chunk["source_id"]), published, text))
                counts["chunks"] += 1
        connection.execute("CREATE INDEX chunks_by_source ON chunks(source_id)")
        connection.commit()
    finally:
        connection.close()
    return counts


def build_entity_partition(
    corpus_path: Path, facts_path: Path, destination: Path, *, batch_rows: int,
) -> dict[str, int]:
    corpus = sqlite3.connect(corpus_path.resolve().as_uri() + "?mode=ro", uri=True)
    partition = sqlite3.connect(destination)
    counts = {"chunks": 0, "entity_source_links": 0, "sources": 0}
    try:
        partition.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, available_at TEXT, text TEXT NOT NULL)")
        partition.execute("CREATE VIRTUAL TABLE lookup USING fts5(text, content='chunks', content_rowid='rowid', tokenize='trigram')")
        partition.execute("CREATE TABLE source_entities(source_id TEXT NOT NULL, canonical_entity_id TEXT NOT NULL, PRIMARY KEY(source_id, canonical_entity_id)) WITHOUT ROWID")
        for fact in iter_payloads(facts_path, batch_rows=batch_rows):
            source_id = str(fact.get("source_id") or "")
            entity_id = str(fact.get("canonical_entity_id") or "")
            if source_id and entity_id:
                inserted = partition.execute(
                    "INSERT OR IGNORE INTO source_entities VALUES (?,?)", (source_id, entity_id)
                )
                counts["entity_source_links"] += max(0, inserted.rowcount)
        for (source_id,) in partition.execute("SELECT DISTINCT source_id FROM source_entities"):
            counts["sources"] += 1
            for chunk_id, available_at, text in corpus.execute(
                "SELECT chunk_id,available_at,text FROM chunks WHERE source_id=?", (source_id,)
            ):
                inserted = partition.execute(
                    "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
                    (chunk_id, source_id, available_at, text),
                )
                if inserted.rowcount:
                    partition.execute("INSERT INTO lookup(rowid,text) VALUES (?,?)", (inserted.lastrowid, text))
                    counts["chunks"] += 1
        partition.execute("CREATE INDEX source_entities_by_entity ON source_entities(canonical_entity_id, source_id)")
        partition.commit()
    finally:
        partition.close()
        corpus.close()
    return counts


class SnapshotTools:
    def __init__(self, path: Path, cutoff: str, *, entity_id: str, dated_only: bool = False):
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.cutoff = cutoff
        self.entity_id = entity_id
        self.dated_only = dated_only
        self.seen_search: set[str] = set()
        self.opened: set[str] = set()

    def close(self) -> None:
        self.db.close()

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "search_corpus":
            query = str(arguments.get("query", "")).strip()[:200]
            if len(query) < 2:
                return {"error": "query must contain at least 2 characters"}
            try:
                limit = max(1, min(10, int(arguments.get("limit", 5))))
            except (TypeError, ValueError):
                return {"error": "limit must be an integer"}
            # Trigram FTS accepts quoted Chinese and English terms without a language-specific dictionary.
            terms = [term for term in re.split(r"[\s,;:\uFF0C\uFF1B\u3001]+", query) if len(term) >= 3][:6]
            match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            rows = self.db.execute(
                "SELECT c.chunk_id,c.source_id,c.available_at,substr(c.text,1,350) "
                "FROM lookup JOIN chunks c ON c.rowid=lookup.rowid "
                "JOIN source_entities e ON e.source_id=c.source_id "
                "WHERE lookup MATCH ? AND e.canonical_entity_id=? "
                "AND (?=0 OR c.available_at IS NOT NULL) "
                "AND (c.available_at IS NULL OR c.available_at<=?) "
                "ORDER BY bm25(lookup) LIMIT ?",
                (match, self.entity_id, int(self.dated_only), self.cutoff, limit)
            ).fetchall() if match else []
            if not rows:
                rows = self.db.execute(
                    "SELECT c.chunk_id,c.source_id,c.available_at,substr(c.text,1,350) FROM chunks c "
                    "JOIN source_entities e ON e.source_id=c.source_id "
                    "WHERE c.text LIKE ? AND e.canonical_entity_id=? "
                    "AND (?=0 OR c.available_at IS NOT NULL) "
                    "AND (c.available_at IS NULL OR c.available_at<=?) LIMIT ?",
                    ("%" + query + "%", self.entity_id, int(self.dated_only), self.cutoff, limit)
                ).fetchall()
            self.seen_search.update(row[0] for row in rows)
            return {"hits": [{"chunk_id": row[0], "source_id": row[1], "available_at": row[2], "snippet": row[3]} for row in rows]}
        if name == "open_chunk":
            chunk_id = str(arguments.get("chunk_id", ""))
            if chunk_id not in self.seen_search:
                return {"error": "chunk_id must be returned by search_corpus first"}
            row = self.db.execute(
                "SELECT c.source_id,c.available_at,c.text FROM chunks c "
                "JOIN source_entities e ON e.source_id=c.source_id "
                "WHERE c.chunk_id=? AND e.canonical_entity_id=? "
                "AND (?=0 OR c.available_at IS NOT NULL) "
                "AND (c.available_at IS NULL OR c.available_at<=?)",
                (chunk_id, self.entity_id, int(self.dated_only), self.cutoff),
            ).fetchone()
            if row is None:
                return {"error": "unknown chunk_id"}
            self.opened.add(chunk_id)
            return {"chunk_id": chunk_id, "source_id": row[0], "available_at": row[1], "text": row[2]}
        if name == "calculate":
            try:
                values = [Decimal(str(value)) for value in arguments.get("values", [])]
                if not 2 <= len(values) <= 12 or any(not value.is_finite() for value in values):
                    raise ValueError("2 to 12 finite values required")
                op = str(arguments.get("operation", ""))
                if op == "add":
                    result = sum(values)
                elif op == "subtract" and len(values) == 2:
                    result = values[0] - values[1]
                elif op == "multiply" and len(values) == 2:
                    result = values[0] * values[1]
                elif op == "divide" and len(values) == 2:
                    result = values[0] / values[1]
                elif op == "percent_change" and len(values) == 2:
                    result = (values[0] - values[1]) / values[1] * 100
                elif op == "average":
                    result = sum(values) / len(values)
                else:
                    raise ValueError("unsupported operation or argument count")
                return {"value": str(result)}
            except (InvalidOperation, ValueError, ZeroDivisionError, TypeError) as exc:
                return {"error": str(exc)}
        return {"error": "unknown tool"}
