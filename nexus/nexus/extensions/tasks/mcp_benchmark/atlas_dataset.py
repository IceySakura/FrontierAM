from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import tempfile

from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from .constants import ATLAS_DATASET_ROOT


ATLAS_HF_DATASET_ID = os.getenv('NEXUS_ATLAS_DATASET_ID', 'ScaleAI/MCP-Atlas')
ATLAS_HF_PARQUET_URI = os.getenv(
    'NEXUS_ATLAS_PARQUET_URI',
    f'https://huggingface.co/datasets/{ATLAS_HF_DATASET_ID}/resolve/main/MCP-Atlas.parquet?download=true',
)
ATLAS_HF_CACHE_PATH = Path(
    os.getenv('NEXUS_ATLAS_PARQUET_CACHE_PATH', str(ATLAS_DATASET_ROOT / 'hf_cache' / 'MCP-Atlas.parquet'))
)

_ATLAS_COLUMNS = ['TASK', 'ENABLED_TOOLS', 'PROMPT', 'GTFA_CLAIMS', 'TRAJECTORY']
_ATLAS_TOOL_NAME_MAPPINGS = {
    'brave_brave_web_search': 'brave-search_brave_web_search',
    'MongoDB_aggregate': 'mongodb_aggregate',
    'MongoDB_collection-schema': 'mongodb_collection-schema',
    'MongoDB_count': 'mongodb_count',
    'MongoDB_find': 'mongodb_find',
    'MongoDB_list-collections': 'mongodb_list-collections',
    'MongoDB_list-databases': 'mongodb_list-databases',
}


def atlas_source_path() -> str:
    return ATLAS_HF_PARQUET_URI


def _ensure_cached_parquet() -> Path:
    if ATLAS_HF_CACHE_PATH.exists() and ATLAS_HF_CACHE_PATH.stat().st_size > 0:
        return ATLAS_HF_CACHE_PATH

    ATLAS_HF_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix='atlas-', suffix='.parquet', dir=str(ATLAS_HF_CACHE_PATH.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    request = Request(ATLAS_HF_PARQUET_URI, headers={'User-Agent': 'nexus-mcp-benchmark'})

    try:
        with urlopen(request, timeout=300) as src, temp_path.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temp_path.replace(ATLAS_HF_CACHE_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return ATLAS_HF_CACHE_PATH


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _coerce_text(value: Any) -> str:
    if _is_missing(value):
        return ''
    return str(value).strip()


def _split_top_level_list_items(text: str) -> list[str]:
    inner = text[1:-1].strip()
    if not inner:
        return []

    items: list[str] = []
    start = 0
    quote: str | None = None
    escape = False
    depth = 0

    for index, char in enumerate(inner):
        if quote is not None:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            continue
        if char in '[{(':
            depth += 1
            continue
        if char in ']})':
            depth = max(0, depth - 1)
            continue
        if char == ',' and depth == 0:
            item = inner[start:index].strip()
            if item:
                items.append(item)
            start = index + 1

    tail = inner[start:].strip()
    if tail:
        items.append(tail)
    return items


def _decode_loose_quoted_string(item: str) -> str | None:
    if len(item) < 2 or item[0] not in {"'", '"'} or item[-1] != item[0]:
        return None

    body = item[1:-1]
    body = body.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')
    return body.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')


def _parse_loose_list_item(item: str) -> Any:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(item)
        except Exception:
            continue

    text = _decode_loose_quoted_string(item)
    if text is not None:
        return text

    claim_match = re.search(r"""['"]claim['"]\s*:\s*(["'])(.*)\1""", item, re.DOTALL)
    if claim_match:
        return {'claim': claim_match.group(2)}

    return item


def _parse_serialized_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return value

    text = str(value).strip()
    if not text:
        return []

    candidates = [text]
    if text.startswith('"[') and text.endswith(']"'):
        candidates.append(text[1:-1].replace('""', '"'))

    seen: set[str] = set()
    while candidates:
        candidate = candidates.pop(0).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                candidates.append(parsed)
        if candidate.startswith('[') and candidate.endswith(']'):
            parsed_items = [_parse_loose_list_item(item) for item in _split_top_level_list_items(candidate)]
            if parsed_items:
                return parsed_items

    raise ValueError(f'Unable to parse Atlas list field: {text[:160]!r}')


def parse_atlas_enabled_tools(value: Any) -> list[str]:
    raw_items = _parse_serialized_list(value)
    tools: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            name = item.get('name')
            if name:
                tools.append(str(name))
            continue
        text = _coerce_text(item)
        if text:
            tools.append(text)
    return tools


def parse_atlas_claims(value: Any) -> list[str]:
    try:
        raw_items = _parse_serialized_list(value)
    except ValueError:
        raw_items = [_coerce_text(value)]
    claims: list[str] = []
    for item in raw_items:
        text = _coerce_text(item.get('claim')) if isinstance(item, dict) else _coerce_text(item)
        if text:
            claims.append(text)
    return claims


def atlas_server_names_from_enabled_tools(enabled_tools: list[str]) -> list[str]:
    servers: list[str] = []
    seen: set[str] = set()
    for tool_name in enabled_tools:
        mapped = _ATLAS_TOOL_NAME_MAPPINGS.get(tool_name, tool_name)
        server = mapped.split('_', 1)[0]
        if server not in seen:
            seen.add(server)
            servers.append(server)
    return servers


def _normalize_record(
    *,
    task_id: str,
    prompt: str,
    enabled_tools: list[str],
    claims: list[str],
    trajectory: str = '',
    source_path: str,
) -> dict[str, Any]:
    return {
        'task_id': task_id,
        'prompt': prompt,
        'enabled_tools': enabled_tools,
        'claims': claims,
        'trajectory': trajectory,
        'source_path': source_path,
    }


def _read_atlas_frame() -> pd.DataFrame:
    parquet_path = _ensure_cached_parquet()
    try:
        return pd.read_parquet(parquet_path, columns=_ATLAS_COLUMNS)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        message = str(exc).lower()
        if 'pyarrow' in message or 'fastparquet' in message or 'parquet' in message:
            raise RuntimeError(
                'Reading MCP-Atlas requires a parquet engine. Install one locally, for example `pyarrow`, and rerun.'
            ) from exc
        raise


@lru_cache(maxsize=1)
def load_atlas_records() -> tuple[dict[str, Any], ...]:
    # Read the public parquet artifact directly to avoid the extra dataset-info
    # API calls that `load_dataset()` and `hf://` can issue before materializing rows.
    frame = _read_atlas_frame()
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient='records'):
        records.append(
            _normalize_record(
                task_id=_coerce_text(row.get('TASK')),
                prompt=_coerce_text(row.get('PROMPT')),
                enabled_tools=parse_atlas_enabled_tools(row.get('ENABLED_TOOLS')),
                claims=parse_atlas_claims(row.get('GTFA_CLAIMS')),
                trajectory=_coerce_text(row.get('TRAJECTORY')),
                source_path=ATLAS_HF_PARQUET_URI,
            )
        )
    return tuple(records)
