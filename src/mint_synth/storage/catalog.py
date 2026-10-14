from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from mint_synth.core.util import stable_id


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class InputSegment:
    segment_id: str
    file_id: str
    source_uri: str
    source_format: str
    segment_ordinal: int
    byte_start: int | None
    byte_end: int | None
    row_group: int | None
    estimated_bytes: int
    row_count: int | None
    file_size: int
    mtime_ns: int
    file_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matches(relative: str, patterns: Iterable[str]) -> bool:
    normalized = relative.replace("\\", "/")
    path = PurePosixPath(normalized)
    name = path.name
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/")
        if path.match(normalized_pattern) or fnmatch.fnmatchcase(normalized, normalized_pattern):
            return True
        if normalized_pattern.startswith("**/") and fnmatch.fnmatchcase(name, normalized_pattern[3:]):
            return True
    return False


def discover_files(input_config: Mapping[str, Any]) -> list[Path]:
    includes = list(input_config.get("include") or ["**/*.jsonl", "**/*.parquet"])
    excludes = list(input_config.get("exclude") or [])
    recursive = bool(input_config.get("recursive", True))
    follow_symlinks = bool(input_config.get("follow_symlinks", False))
    found: dict[str, Path] = {}

    def consider(path: Path, relative: str) -> None:
        if path.is_symlink() and not follow_symlinks:
            return
        suffix = path.suffix.casefold()
        if suffix not in {".jsonl", ".parquet"}:
            return
        if not _matches(relative, includes) or (excludes and _matches(relative, excludes)):
            return
        resolved = path.resolve()
        found[os.path.normcase(str(resolved))] = resolved

    for raw_root in input_config.get("paths", []):
        root = Path(str(raw_root))
        if not root.exists():
            raise CatalogError(f"input path does not exist: {root}")
        if root.is_file():
            consider(root, root.name)
            continue
        if not root.is_dir():
            raise CatalogError(f"input path is neither a regular file nor directory: {root}")
        if recursive:
            visited_directories: set[str] = set()
            for current, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
                identity = os.path.normcase(str(Path(current).resolve()))
                if identity in visited_directories:
                    dirnames.clear()
                    continue
                visited_directories.add(identity)
                if not follow_symlinks:
                    dirnames[:] = [name for name in dirnames if not (Path(current) / name).is_symlink()]
                for filename in filenames:
                    path = Path(current) / filename
                    consider(path, path.relative_to(root).as_posix())
        else:
            for path in root.iterdir():
                if path.is_file():
                    consider(path, path.name)
    files = sorted(found.values(), key=lambda path: os.path.normcase(str(path)))
    if not files:
        raise CatalogError("no .jsonl or .parquet files matched input.include/input.exclude")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path, mode: str) -> tuple[str, os.stat_result]:
    stat = path.stat()
    if mode == "sha256":
        material = {"sha256": _sha256(path), "size": stat.st_size}
    elif mode == "metadata":
        material = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    else:
        raise CatalogError(f"unsupported fingerprint mode: {mode}")
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), stat


def _parquet_segments(
    path: Path,
    *,
    fingerprint: str,
    stat: os.stat_result,
    text_field: str,
    id_field: str | None,
) -> Iterator[InputSegment]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise CatalogError("pyarrow is required to catalog Parquet inputs") from exc
    parquet = pq.ParquetFile(path)
    if text_field.split(".", 1)[0] not in parquet.schema_arrow.names:
        raise CatalogError(f"Parquet file {path} does not contain text field {text_field!r}")
    if id_field and id_field.split(".", 1)[0] not in parquet.schema_arrow.names:
        raise CatalogError(f"Parquet file {path} does not contain configured id field {id_field!r}")
    file_id = stable_id("file", str(path), fingerprint, length=32)
    total_compressed = sum(
        max(0, parquet.metadata.row_group(index).total_byte_size)
        for index in range(parquet.metadata.num_row_groups)
    )
    for index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(index)
        estimated = int(row_group.total_byte_size)
        if estimated <= 0 and total_compressed > 0:
            estimated = max(1, stat.st_size // parquet.metadata.num_row_groups)
        yield InputSegment(
            segment_id=stable_id("seg", file_id, "row_group", index, length=32),
            file_id=file_id,
            source_uri=str(path),
            source_format="parquet",
            segment_ordinal=index,
            byte_start=None,
            byte_end=None,
            row_group=index,
            estimated_bytes=estimated,
            row_count=row_group.num_rows,
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            file_fingerprint=fingerprint,
        )


def _jsonl_segments(
    path: Path,
    *,
    fingerprint: str,
    stat: os.stat_result,
    byte_shard_size: int,
) -> Iterator[InputSegment]:
    file_id = stable_id("file", str(path), fingerprint, length=32)
    if stat.st_size == 0:
        yield InputSegment(
            stable_id("seg", file_id, 0, 0, length=32), file_id, str(path), "jsonl", 0,
            0, 0, None, 0, 0, stat.st_size, stat.st_mtime_ns, fingerprint,
        )
        return
    for ordinal, start in enumerate(range(0, stat.st_size, byte_shard_size)):
        end = min(stat.st_size, start + byte_shard_size)
        yield InputSegment(
            segment_id=stable_id("seg", file_id, start, end, length=32),
            file_id=file_id,
            source_uri=str(path),
            source_format="jsonl",
            segment_ordinal=ordinal,
            byte_start=start,
            byte_end=end,
            row_group=None,
            estimated_bytes=end - start,
            row_count=None,
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            file_fingerprint=fingerprint,
        )


def build_catalog(config: Mapping[str, Any]) -> list[InputSegment]:
    files = discover_files(config["input"])
    excluded_roots = [
        Path(str(config[section][key])).resolve()
        for section, key in (("output", "root"), ("run", "work_dir"))
        if section in config and key in config[section]
    ]
    files = [
        path
        for path in files
        if not any(path == root or root in path.parents for root in excluded_roots)
    ]
    auxiliary_files = {
        Path(str(value)).resolve()
        for value in (
            config.get("query_realization", {}).get("seed_file"),
            config.get("quality", {}).get("evaluation_file"),
            config.get("entity_resolution", {}).get("alias_file"),
            config.get("dag", {}).get("formula_registry"),
        )
        if value and value != "builtin"
    }
    files = [path for path in files if path not in auxiliary_files]
    if not files:
        raise CatalogError("all matched inputs are output/work artifacts or configured auxiliary files")
    mode = str(config["run"].get("fingerprint_mode", "metadata"))
    byte_shard_size = int(config.get("reader", {}).get("jsonl_byte_shard_size", 1024**3))
    text_field = str(config["input"]["text_field"])
    id_field = str(config["input"]["id_field"]) if config["input"].get("id_field") else None
    segments: list[InputSegment] = []
    for path in files:
        fingerprint, stat = file_fingerprint(path, mode)
        if path.suffix.casefold() == ".jsonl":
            segments.extend(
                _jsonl_segments(
                    path,
                    fingerprint=fingerprint,
                    stat=stat,
                    byte_shard_size=byte_shard_size,
                )
            )
        else:
            segments.extend(
                _parquet_segments(
                    path,
                    fingerprint=fingerprint,
                    stat=stat,
                    text_field=text_field,
                    id_field=id_field,
                )
            )
    return segments


def pack_segments(segments: Iterable[InputSegment], target_bytes: int) -> list[list[InputSegment]]:
    """Deterministic first-fit packing that keeps every task below target when possible."""
    if target_bytes < 1:
        raise CatalogError("normalize_target_bytes must be positive")
    assignments: list[list[InputSegment]] = []
    current: list[InputSegment] = []
    current_size = 0
    for segment in segments:
        size = max(1, segment.estimated_bytes)
        if current and current_size + size > target_bytes:
            assignments.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += size
        if current_size >= target_bytes:
            assignments.append(current)
            current = []
            current_size = 0
    if current:
        assignments.append(current)
    if not assignments:
        raise CatalogError(
            "matched input files produced no readable segments; check for empty Parquet files"
        )
    return assignments


def catalog_fingerprint(segments: Iterable[InputSegment]) -> str:
    material = [
        (segment.segment_id, segment.file_fingerprint, segment.byte_start, segment.byte_end, segment.row_group)
        for segment in segments
    ]
    payload = json.dumps(material, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, segments: Iterable[InputSegment], *, compression: str = "zstd") -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise CatalogError("pyarrow is required to write the input manifest") from exc
    rows = [segment.to_dict() for segment in segments]
    schema = pa.schema(
        [
            ("segment_id", pa.string()), ("file_id", pa.string()), ("source_uri", pa.string()),
            ("source_format", pa.string()), ("segment_ordinal", pa.int32()),
            ("byte_start", pa.int64()), ("byte_end", pa.int64()), ("row_group", pa.int32()),
            ("estimated_bytes", pa.int64()), ("row_count", pa.int64()), ("file_size", pa.int64()),
            ("mtime_ns", pa.int64()), ("file_fingerprint", pa.string()),
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination, compression=None if compression == "none" else compression)


def read_assignment(path: str | Path) -> list[InputSegment]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise CatalogError(f"assignment must be a JSON list: {path}")
    return [InputSegment(**value) for value in raw]
