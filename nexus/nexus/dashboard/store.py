"""Directory-backed read-only catalog. No pickle, database, or runtime dependency."""

from __future__ import annotations

import gzip
import hashlib
import json

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nexus.gateway.trajectory import SCHEMA_VERSION


DEFAULT_MAX_BYTES = 64 * 1024 * 1024
Signature = tuple[int, int, int, int]


@dataclass(frozen=True)
class _Record:
    path: Path
    signature: Signature
    selector: tuple[str, ...] = ()
    line: int | None = None
    offset: int = 0
    length: int = 0
    run_id: str | None = None

    @property
    def key(self) -> str:
        identity = (
            self.path.name
            if self.line is None and not self.selector
            else json.dumps(
                [self.path.name, self.line, self.selector],
                ensure_ascii=True,
            )
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:24]

    @property
    def source(self) -> dict[str, Any]:
        source: dict[str, Any] = {'filename': self.path.name}
        if self.line is not None:
            source['line'] = self.line
        if self.run_id is not None:
            source['run_id'] = self.run_id
        if self.selector:
            source['record_path'] = '.'.join(self.selector)
            if len(self.selector) == 3:
                source['client'] = self.selector[-1]
        return source


@dataclass
class _Catalog:
    entries: list[tuple[_Record, dict[str, Any]]]
    errors: list[dict[str, Any]]


class ArtifactStore:
    def __init__(self, path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError('Artifact read limit must be positive')
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file() and not self.path.is_dir():
            raise ValueError('Dashboard path must be a directory or a JSON trajectory file')
        self.root = self.path if self.path.is_dir() else self.path.parent
        self.max_bytes = max_bytes
        self._files: dict[str, _Record] = {}
        self._catalogs: dict[Path, tuple[tuple[Signature, Signature | None], _Catalog]] = {}

    def _confined(self, path: Path) -> Path:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError('Artifact must be a regular file within the selected directory')
        return resolved

    def _read_json(self, path: Path, *, limit: int | None = None) -> dict[str, Any]:
        path = self._confined(path)
        limit = self.max_bytes if limit is None else limit
        if path.stat().st_size > limit:
            raise ValueError(f'Artifact exceeds the {limit}-byte read limit')
        opener = gzip.open if path.suffix == '.gz' else open
        try:
            with opener(path, 'rb') as stream:
                content = stream.read(limit + 1)
        except EOFError as exc:
            raise ValueError('Truncated compressed trajectory') from exc
        if len(content) > limit:
            raise ValueError(f'Decompressed artifact exceeds the {limit}-byte read limit')
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError('Artifact must contain a JSON object')
        return data

    @staticmethod
    def _unwrap(data: Any) -> dict[str, Any]:
        if isinstance(data, dict) and 'trajectory' in data:
            data = data['trajectory']
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported trajectory schema')
        return data

    def _signature(self, path: Path) -> Signature:
        stat = self._confined(path).stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns

    @staticmethod
    def _trajectory_shape(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return (
            value.get('schema_version') == SCHEMA_VERSION
            or (isinstance(value.get('nodes'), list) and isinstance(value.get('chains'), list))
            or (
                isinstance(value.get('trajectory'), dict)
                and value['trajectory'].get('schema_version') == SCHEMA_VERSION
            )
        )

    @classmethod
    def _candidates(cls, data: dict[str, Any]) -> Iterator[tuple[tuple[str, ...], Any]]:
        if 'schema_version' in data or 'trajectory' in data or 'processor_results' not in data:
            yield (), data
            return
        processors = data['processor_results']
        if not isinstance(processors, dict):
            raise ValueError('Run result contains no Gateway trajectory')
        found = False
        for name, value in processors.items():
            selector = ('processor_results', name)
            if cls._trajectory_shape(value) or (
                name in {'gateway', 'gateway_trajectory'}
                and isinstance(value, dict)
                and isinstance(value.get('schema_version'), str)
            ):
                found = True
                yield selector, value
            elif (
                isinstance(value, dict)
                and value
                and (
                    name in {'gateway', 'gateway_trajectory'}
                    or any(cls._trajectory_shape(item) for item in value.values())
                )
            ):
                found = True
                for client, trajectory in value.items():
                    yield (*selector, client), trajectory
        if not found:
            raise ValueError('Run result contains no Gateway trajectory')

    def _index_document(self, data: Any, record: _Record, catalog: _Catalog) -> None:
        try:
            if not isinstance(data, dict):
                raise ValueError('Artifact must contain a JSON object')
            if isinstance(data.get('run_id'), str):
                record = replace(record, run_id=data['run_id'])
            for selector, artifact in self._candidates(data):
                entry = replace(record, selector=selector)
                try:
                    trajectory = self._unwrap(artifact)
                    summary = {
                        'session_id': trajectory['session_id'],
                        'state': trajectory.get('state'),
                        'node_count': len(trajectory['nodes']),
                        'chain_count': len(trajectory['chains']),
                        'final_reward': trajectory.get('final_reward'),
                        'termination_reason': trajectory.get('termination_reason'),
                        'created_at': artifact.get('created_at'),
                    }
                    catalog.entries.append((entry, summary))
                except (ValueError, KeyError, TypeError) as exc:
                    catalog.errors.append({**entry.source, 'error': str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            catalog.errors.append({**record.source, 'error': str(exc)})

    def _scan_jsonl(self, record: _Record, catalog: _Catalog) -> None:
        with self._confined(record.path).open('rb') as stream:
            line = 0
            while True:
                offset = stream.tell()
                raw = stream.readline(self.max_bytes + 2)
                if not raw:
                    break
                line += 1
                entry = replace(record, line=line, offset=offset, length=len(raw))
                content = raw.removesuffix(b'\n').removesuffix(b'\r')
                if len(content) > self.max_bytes or (len(raw) > self.max_bytes and not raw.endswith(b'\n')):
                    # Drain an oversized record in bounded chunks, then resume at the next line.
                    while raw and not raw.endswith(b'\n'):
                        raw = stream.readline(64 * 1024)
                    catalog.errors.append({
                        **entry.source,
                        'error': f'JSONL record exceeds the {self.max_bytes}-byte read limit',
                    })
                    continue
                if not content.strip():
                    continue
                try:
                    self._index_document(json.loads(content), entry, catalog)
                except (ValueError, RecursionError) as exc:
                    catalog.errors.append({**entry.source, 'error': str(exc)})

    def _read_record(self, record: _Record) -> dict[str, Any]:
        if record.line is None:
            return self._read_json(record.path)
        if record.length > self.max_bytes + 2:
            raise ValueError(f'JSONL record exceeds the {self.max_bytes}-byte read limit')
        with self._confined(record.path).open('rb') as stream:
            stream.seek(record.offset)
            content = stream.read(record.length)
        if len(content) != record.length:
            raise ValueError('Source changed; refresh the catalog')
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError('Artifact must contain a JSON object')
        return data

    def _scan_file(self, path: Path) -> _Catalog:
        if path.name.endswith('.jsonl.gz'):
            raise ValueError('Compressed JSONL is not supported; decompress it to .jsonl first')
        signature = self._signature(path)
        summary_path = path.with_name(path.name.split('.trajectory.json')[0] + '.summary.json')
        has_summary = path.name.endswith(('.trajectory.json', '.trajectory.json.gz')) and summary_path.exists()
        summary_signature = self._signature(summary_path) if has_summary else None
        cache_key = (signature, summary_signature)
        cached = self._catalogs.get(path)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        catalog = _Catalog([], [])
        record = _Record(path, signature)
        if has_summary:
            summary = self._read_json(summary_path, limit=1024 * 1024)
            if summary.get('file') != path.name:
                raise ValueError('Invalid summary catalog entry')
            catalog.entries.append((record, summary))
        elif path.suffix == '.jsonl':
            self._scan_jsonl(record, catalog)
        else:
            self._index_document(self._read_json(path), record, catalog)
        if self._signature(path) != signature or (has_summary and self._signature(summary_path) != summary_signature):
            raise ValueError('Source changed while indexing; refresh the catalog')
        self._catalogs[path] = (cache_key, catalog)
        return catalog

    def scan(self) -> dict[str, Any]:
        summaries, errors, files = [], [], {}
        if self.path.is_file():
            candidates = [self.path]
        else:
            candidates = sorted({
                path
                for pattern in ('*.json', '*.json.gz', '*.jsonl')
                for path in self.path.glob(pattern)
                if not path.name.endswith('.summary.json') and not path.name.startswith('.')
            })
        active_cache = set()
        for path in candidates:
            try:
                catalog = self._scan_file(path)
                active_cache.add(path)
                errors.extend(catalog.errors)
                for record, summary in catalog.entries:
                    files[record.key] = record
                    summaries.append({**summary, **record.source, 'id': record.key})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'filename': path.name, 'error': str(exc)})
        self._files = files
        self._catalogs = {path: cache for path, cache in self._catalogs.items() if path in active_cache}
        summaries.sort(
            key=lambda item: (
                str(item.get('created_at') or ''),
                item['filename'],
                -(item.get('line') or 0),
                item.get('client', ''),
            ),
            reverse=True,
        )
        return {'items': summaries, 'errors': errors}

    def load(self, artifact_id: str) -> dict[str, Any]:
        record = self._files.get(artifact_id)
        if record is None:
            raise KeyError('Unknown artifact ID; refresh the catalog')
        if self._signature(record.path) != record.signature:
            raise ValueError('Source changed; refresh the catalog before loading')
        document = self._read_record(record)
        if self._signature(record.path) != record.signature:
            raise ValueError('Source changed while loading; refresh the catalog')
        data = document
        for key in record.selector:
            data = data[key]
        trajectory = self._unwrap(data)
        result = {
            'trajectory': trajectory,
            'export': {key: value for key, value in data.items() if key != 'trajectory'}
            if 'trajectory' in data
            else {},
        }
        if record.selector or record.line is not None:
            result['source'] = record.source
        if record.selector:
            result['run'] = {
                key: document[key]
                for key in ('run_id', 'status', 'success', 'error', 'start_time', 'end_time', 'retry_count')
                if key in document
            }
        return result
