"""Pluggable URI factories that create unified Dataset Readers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from inspect import signature
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

from nexus.core.registry import EntryNotFoundError, ProcessorResult, Registry
from nexus.dataset.reader import DatasetReader


@dataclass(frozen=True)
class DatasetSourceURI:
    """Parsed URI for a registered dataset source."""

    scheme: str
    path: str
    authority: str = ''
    query: dict[str, tuple[str, ...]] | None = None
    split: str | None = None
    raw: str = ''

    @property
    def display(self) -> str:
        return self.raw or f'{self.scheme}://{self.authority}{self.path}'


DatasetSourceFactory = Callable[[DatasetSourceURI], DatasetReader]


def validate_dataset_source(obj: Any, _name: str, _meta: dict[str, Any]) -> ProcessorResult:
    """Validate a dataset source factory."""
    if not callable(obj):
        raise TypeError(f'Dataset source must be callable, got {type(obj)}')
    return ProcessorResult(wrapped_obj=obj)


dataset_source_registry = Registry[DatasetSourceFactory]('dataset_source', validate_dataset_source)
dataset_source = dataset_source_registry.register


def parse_dataset_source_uri(value: str) -> DatasetSourceURI | None:
    """Parse a registered-source URI, returning ``None`` for normal dataset references."""
    raw = value.strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if not parsed.scheme or '://' not in raw:
        return None
    if parsed.fragment:
        raise ValueError(f'Dataset source URI must not contain a fragment: {raw}')
    query = {key: tuple(values) for key, values in parse_qs(parsed.query, keep_blank_values=True).items()}
    split_values = query.get('split')
    if split_values is not None and any(not value for value in split_values):
        raise ValueError(f'Dataset source URI split must be non-empty: {raw}')
    return DatasetSourceURI(
        scheme=parsed.scheme.lower(),
        authority=parsed.netloc,
        path=unquote(parsed.path),
        query=query,
        raw=raw,
    )


def open_dataset_reader(value: str, *, context: Any | None = None) -> DatasetReader:
    """Resolve a registered source URI into its unified Reader."""
    parsed = parse_dataset_source_uri(value)
    if parsed is None:
        raise ValueError(f'Not a dataset source URI: {value}')
    try:
        factory = dataset_source_registry.get(parsed.scheme)
    except EntryNotFoundError as exc:
        raise ValueError(
            f'Unknown dataset source scheme {parsed.scheme!r}. '
            'Install or enable the Nexus module that provides this source.'
        ) from exc
    split_values = (parsed.query or {}).get('split', ())
    if len(split_values) > 1:
        raise ValueError(f'Dataset source URI has multiple split values: {value}')
    provider_query = {key: values for key, values in (parsed.query or {}).items() if key != 'split'}
    source = replace(
        parsed,
        query=provider_query,
        split=split_values[0] if split_values else None,
    )
    if context is not None and 'context' in signature(factory).parameters:
        return cast(Callable[..., DatasetReader], factory)(source, context=context)
    return factory(source)


__all__ = [
    'DatasetSourceFactory',
    'DatasetSourceURI',
    'dataset_source',
    'dataset_source_registry',
    'open_dataset_reader',
    'parse_dataset_source_uri',
]
