"""Rule contract. Every concrete rule module exports one ``RULE`` object."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..context import DetectionContext
from ..models import Finding


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    title: str
    scan: Callable[[DetectionContext], list[Finding]]
