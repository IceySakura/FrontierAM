"""Run compose models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RawPartKind = Literal['data', 'use', 'set', 'set_json', 'input']


@dataclass(frozen=True)
class RawRunPart:
    """A command-line run part before resolution."""

    kind: RawPartKind
    value: Any
    source: str
    merge: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunComposition:
    """Ordered run parts loaded from a compose YAML file."""

    parts: list[RawRunPart]
    item_id: str | None = None
    split: str | None = None
    limit: int | None = None
    source: str = '<run-composition>'


@dataclass(frozen=True)
class RunPart:
    """A resolved partial RunInput plus merge operations."""

    body: dict[str, Any]
    merge: dict[str, str] = field(default_factory=dict)
    source: str = '<run-part>'


@dataclass(frozen=True)
class UseSpec:
    """Parsed ``--use`` value."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
