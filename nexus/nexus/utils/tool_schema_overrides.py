from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_tool_schema_overrides(overrides: list[dict[str, Any]] | str | None) -> list[dict[str, Any]] | None:
    """Resolve tool schema overrides from either inline schemas or a config path."""
    if overrides is None:
        return None
    if isinstance(overrides, list):
        return overrides
    if not isinstance(overrides, str):
        raise TypeError(f'Unsupported tool_schema_overrides type: {type(overrides).__name__}')

    path = Path(overrides)
    if not path.exists():
        raise FileNotFoundError(f'tool_schema_overrides file not found: {path}')

    with path.open('r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)

    if isinstance(loaded, list):
        schemas = loaded
    elif isinstance(loaded, dict):
        if isinstance(loaded.get('tools'), list):
            schemas = [item.get('tool_schema') for item in loaded['tools'] if isinstance(item, dict)]
        elif 'tool_schema' in loaded and isinstance(loaded.get('tool_schema'), dict):
            schemas = [loaded['tool_schema']]
        elif loaded.get('type') == 'function' and isinstance(loaded.get('function'), dict):
            schemas = [loaded]
        else:
            raise TypeError(
                f'Unsupported tool_schema_overrides file format in {path}. '
                'Expected a list of schemas or a dict containing `tools`/`tool_schema`.'
            )
    else:
        raise TypeError(
            f'Unsupported tool_schema_overrides file format in {path}. '
            f'Expected YAML/JSON list or dict, got {type(loaded).__name__}.'
        )

    if not all(isinstance(schema, dict) for schema in schemas):
        raise TypeError(f'Invalid tool_schema_overrides in {path}: every schema must be a dict')

    return schemas
