from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def task_name_from_input(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = value.get('task_name')
    return name if isinstance(name, str) and name else None


def task_name_from_spec(spec_type: str, spec: Any) -> str | None:
    if not isinstance(spec, Mapping):
        return None
    if spec_type in {'nexus_single', 'nexus_live'}:
        return task_name_from_input(spec.get('run_input', spec))
    return None
