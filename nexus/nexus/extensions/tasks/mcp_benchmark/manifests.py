from __future__ import annotations

import base64
import json
import tempfile

from functools import lru_cache
from pathlib import Path
from typing import Any

from .atlas_dataset import load_atlas_records
from .constants import (
    ATLAS_MANIFEST_PATH,
    CONVERTED_DATASETS_ROOT,
    TOOLATHLON_DATASET_ROOT,
    TOOLATHLON_MANIFEST_PATH,
    UNIVERSE_DATASET_ROOT,
    UNIVERSE_MANIFEST_PATH,
)
from .credentials import load_toolathlon_task_overrides
from .server_specs import atlas_server_names_from_enabled_tools, universe_server_names


_DATASET_ORDER = ('toolathlon', 'atlas', 'universe')
_TOOLATHLON_BASE = TOOLATHLON_DATASET_ROOT / 'tasks' / 'finalpool'
_UNIVERSE_BASE = UNIVERSE_DATASET_ROOT / 'mcpuniverse' / 'benchmark' / 'configs' / 'mcpuniverse'
_TOOLATHLON_TEXT_SUFFIXES = {'.py', '.sh', '.md', '.json', '.yaml', '.yml', '.txt'}
_TOOLATHLON_GOOGLE_CREDENTIAL_MARKERS = (
    'configs/google_credentials.json',
    'configs/credentials.json',
)
_TOOLATHLON_SNOWFLAKE_PRIVATE_KEY_MARKER = 'snowflake_rsa_key.p8'
_TOOLATHLON_NOTION_DUPLICATION_MARKERS = (
    'notion_remove_and_duplicate',
    'source_notion_page_url',
    'eval_notion_page_url',
)


def manifest_path_for_dataset(dataset: str) -> Path:
    if dataset == 'toolathlon':
        return TOOLATHLON_MANIFEST_PATH
    if dataset == 'atlas':
        return ATLAS_MANIFEST_PATH
    if dataset == 'universe':
        return UNIVERSE_MANIFEST_PATH
    raise ValueError(f'Unsupported dataset: {dataset}')


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write('\n')


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f'Prepared manifest not found: {path}. '
            'Run `uv run python -m nexus.extensions.tasks.mcp_benchmark.prepare_data` first.'
        )

    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid JSONL at {path}:{line_number}') from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f'Invalid manifest row at {path}:{line_number}: expected object')
            rows.append(payload)
    return rows


def _relative_to(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _require_string(value: Any, *, field: str, dataset: str, case_id: str) -> str:
    text = str(value or '').strip()
    if not text:
        raise RuntimeError(f'Manifest row missing {field}: {dataset}/{case_id}')
    return text


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _base64_file_entry(content: bytes) -> dict[str, str]:
    return {
        'encoding': 'base64',
        'content': base64.b64encode(content).decode('ascii'),
    }


def _embedded_file_bytes(
    entry: dict[str, Any],
    *,
    field: str,
    dataset: str,
    case_id: str,
    path: str,
) -> bytes:
    encoding = str(entry.get('encoding') or 'base64').strip().lower()
    content = entry.get('content')
    if not isinstance(content, str):
        raise RuntimeError(f'Invalid embedded content for {dataset}/{case_id}: {field}[{path!r}]')
    if encoding == 'base64':
        try:
            return base64.b64decode(content)
        except Exception as exc:
            raise RuntimeError(f'Invalid base64 embedded content for {dataset}/{case_id}: {field}[{path!r}]') from exc
    if encoding == 'utf8':
        return content.encode('utf-8')
    raise RuntimeError(f'Unsupported embedded encoding for {dataset}/{case_id}: {field}[{path!r}] -> {encoding}')


def _embedded_file_text(
    file_map: dict[str, dict[str, str]],
    path: str,
    *,
    field: str,
    dataset: str,
    case_id: str,
) -> str | None:
    entry = file_map.get(path)
    if entry is None:
        return None
    return _embedded_file_bytes(entry, field=field, dataset=dataset, case_id=case_id, path=path).decode('utf-8')


def _normalize_embedded_file_map(
    raw: Any,
    *,
    field: str,
    dataset: str,
    case_id: str,
) -> dict[str, dict[str, str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Invalid {field} for {dataset}/{case_id}: expected object')

    normalized: dict[str, dict[str, str]] = {}
    for path, entry in sorted(raw.items()):
        rel_path = _require_string(path, field=f'{field}.path', dataset=dataset, case_id=case_id)
        if not isinstance(entry, dict):
            raise RuntimeError(f'Invalid {field}[{rel_path!r}] for {dataset}/{case_id}: expected object')
        content = entry.get('content')
        if not isinstance(content, str):
            raise RuntimeError(f'Invalid {field}[{rel_path!r}] for {dataset}/{case_id}: missing content')
        encoding = str(entry.get('encoding') or 'base64').strip().lower()
        if encoding not in {'base64', 'utf8'}:
            raise RuntimeError(
                f'Invalid {field}[{rel_path!r}] for {dataset}/{case_id}: unsupported encoding {encoding}'
            )
        normalized[rel_path] = {'encoding': encoding, 'content': content}
    return normalized


def _embed_directory_files(root: Path, *, prefix: str = '') -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        key = f'{prefix}/{relative}' if prefix else relative
        payload[key] = _base64_file_entry(path.read_bytes())
    return payload


def _materialize_embedded_files(root: Path, file_map: dict[str, dict[str, str]], *, case_id: str) -> None:
    for relative_path, entry in file_map.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = _embedded_file_bytes(
            entry,
            field='embedded_case_files',
            dataset='toolathlon',
            case_id=case_id,
            path=relative_path,
        )
        path.write_bytes(content)


def _normalize_task_override_value(value: Any, *, task_dir: Path) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_task_override_value(item, task_dir=task_dir) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_task_override_value(item, task_dir=task_dir) for item in value]

    text = str(value)
    normalized = text.replace('\\', '/')
    task_root = task_dir.as_posix().rstrip('/')
    if task_root:
        normalized = normalized.replace(f'{task_root}/', '')
        normalized = normalized.replace(task_root, '.')
    if normalized == task_root:
        return '.'
    if task_root and normalized.startswith(f'{task_root}/'):
        return normalized[len(task_root) + 1 :]

    candidate = (task_dir / text).resolve()
    if candidate.exists():
        try:
            return candidate.relative_to(task_dir).as_posix()
        except ValueError:
            return normalized
    return normalized


def _toolathlon_task_analysis(task_dir: Path) -> dict[str, Any]:
    raw_overrides = load_toolathlon_task_overrides(str(task_dir))
    task_override_error = raw_overrides.get('__load_error__')
    task_override_values = {
        str(key): _normalize_task_override_value(_jsonable(value), task_dir=task_dir)
        for key, value in raw_overrides.items()
        if not str(key).startswith('__')
    }

    task_references_google_creds = False
    task_references_snowflake_key = False
    task_references_notion_duplication = False
    for root in (task_dir / 'preprocess', task_dir / 'evaluation'):
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in _TOOLATHLON_TEXT_SUFFIXES:
                continue
            try:
                content = path.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            if not task_references_google_creds and any(
                marker in content for marker in _TOOLATHLON_GOOGLE_CREDENTIAL_MARKERS
            ):
                task_references_google_creds = True
            if not task_references_snowflake_key and _TOOLATHLON_SNOWFLAKE_PRIVATE_KEY_MARKER in content:
                task_references_snowflake_key = True
            if not task_references_notion_duplication and any(
                marker in content for marker in _TOOLATHLON_NOTION_DUPLICATION_MARKERS
            ):
                task_references_notion_duplication = True
            if task_references_google_creds and task_references_snowflake_key and task_references_notion_duplication:
                break
        if task_references_google_creds and task_references_snowflake_key and task_references_notion_duplication:
            break

    return {
        'task_override_values': task_override_values,
        'task_override_error': (
            _normalize_task_override_value(str(task_override_error), task_dir=task_dir) if task_override_error else None
        ),
        'has_preprocess_main': (task_dir / 'preprocess' / 'main.py').exists(),
        'has_preprocess_sh': (task_dir / 'preprocess' / 'run.sh').exists(),
        'has_initial_workspace': (task_dir / 'initial_workspace').exists(),
        'has_evaluation_main': (task_dir / 'evaluation' / 'main.py').exists(),
        'references_google_creds': task_references_google_creds,
        'references_snowflake_private_key': task_references_snowflake_key,
        'references_notion_duplication': task_references_notion_duplication,
    }


def _toolathlon_embedded_task_analysis(case_id: str, embedded_case_files: dict[str, dict[str, str]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f'nexus-toolathlon-{case_id}-') as temp_dir:
        task_dir = Path(temp_dir)
        _materialize_embedded_files(task_dir, embedded_case_files, case_id=case_id)
        return _toolathlon_task_analysis(task_dir)


def _resolve_toolathlon_task_dir(raw: dict[str, Any], case_id: str) -> Path:
    task_dir = _resolve_path(TOOLATHLON_DATASET_ROOT, raw.get('task_dir_rel'))
    if task_dir is not None:
        return task_dir

    evaluator_config = raw.get('evaluator_config') or {}
    task_dir = _resolve_path(TOOLATHLON_DATASET_ROOT, evaluator_config.get('cwd'))
    if task_dir is not None:
        return task_dir

    task_dir = _TOOLATHLON_BASE / case_id
    if task_dir.exists():
        return task_dir

    raise RuntimeError(f'Unable to resolve Toolathlon task dir for {case_id}')


def _resolve_universe_task_file(raw: dict[str, Any], case_id: str) -> Path:
    task_file = _resolve_path(UNIVERSE_DATASET_ROOT, raw.get('task_file_rel'))
    if task_file is not None:
        return task_file

    source_path = _resolve_path(UNIVERSE_DATASET_ROOT, raw.get('source_path'))
    if source_path is not None and source_path.suffix == '.json':
        return source_path

    metadata = raw.get('metadata') or {}
    source_path = _resolve_path(UNIVERSE_DATASET_ROOT, metadata.get('source_path'))
    if source_path is not None and source_path.suffix == '.json':
        return source_path

    matches = list(_UNIVERSE_BASE.rglob(f'{case_id}.json'))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f'Unable to resolve Universe task file for {case_id}')
    raise RuntimeError(f'Ambiguous Universe task file for {case_id}: {matches}')


def build_toolathlon_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    embedded_server_files = _embed_directory_files(
        TOOLATHLON_DATASET_ROOT / 'configs' / 'mcp_servers',
        prefix='configs/mcp_servers',
    )
    for config_path in sorted(_TOOLATHLON_BASE.glob('*/task_config.json')):
        task_dir = config_path.parent
        task_config = json.loads(config_path.read_text())
        prompt_path = task_dir / 'docs' / 'task.md'
        system_prompt_path = task_dir / 'docs' / 'agent_system_prompt.md'
        task_analysis = _toolathlon_task_analysis(task_dir)
        rows.append({
            'dataset': 'toolathlon',
            'case_id': task_dir.name,
            'prompt': prompt_path.read_text().strip() if prompt_path.exists() else task_dir.name,
            'system_prompt': system_prompt_path.read_text().strip() if system_prompt_path.exists() else None,
            'task_config': task_config,
            'server_names': list(task_config.get('needed_mcp_servers', [])),
            'embedded_case_files': _embed_directory_files(task_dir),
            'embedded_server_files': embedded_server_files,
            **task_analysis,
        })
    return tuple(rows)


def build_atlas_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for record in load_atlas_records():
        rows.append({
            'dataset': 'atlas',
            'case_id': record['task_id'],
            'prompt': record['prompt'],
            'system_prompt': None,
            'source_path': record['source_path'],
            'enabled_tools': list(record['enabled_tools']),
            'claims': list(record['claims']),
            'trajectory': record.get('trajectory', ''),
            'server_names': atlas_server_names_from_enabled_tools(record['enabled_tools']),
        })
    return tuple(rows)


def build_universe_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for task_file in sorted(_UNIVERSE_BASE.rglob('*.json')):
        task_data = json.loads(task_file.read_text())
        rows.append({
            'dataset': 'universe',
            'case_id': task_file.stem,
            'prompt': str(task_data.get('question', '')).strip() or task_file.stem,
            'system_prompt': None,
            'source_path': str(task_file),
            'task_file_rel': _relative_to(UNIVERSE_DATASET_ROOT, task_file),
            'task_data': task_data,
            'server_names': universe_server_names(task_data),
            'category': task_data.get('category'),
        })
    return tuple(rows)


def build_manifest_rows(dataset: str) -> tuple[dict[str, Any], ...]:
    if dataset == 'toolathlon':
        return build_toolathlon_manifest_rows()
    if dataset == 'atlas':
        return build_atlas_manifest_rows()
    if dataset == 'universe':
        return build_universe_manifest_rows()
    raise ValueError(f'Unsupported dataset: {dataset}')


def write_manifest(dataset: str, rows: tuple[dict[str, Any], ...] | None = None) -> Path:
    payload = list(rows or build_manifest_rows(dataset))
    path = manifest_path_for_dataset(dataset)
    _write_jsonl(path, payload)
    clear_manifest_caches()
    return path


def prepare_manifests(datasets: list[str] | None = None) -> dict[str, Path]:
    selected = [dataset for dataset in _DATASET_ORDER if datasets is None or dataset in set(datasets)]
    written: dict[str, Path] = {}
    for dataset in selected:
        written[dataset] = write_manifest(dataset)
    return written


def _normalize_toolathlon_row(raw: dict[str, Any]) -> dict[str, Any]:
    case_id = _require_string(
        raw.get('case_id') or raw.get('task_id'), field='case_id', dataset='toolathlon', case_id=''
    )
    embedded_case_files = _normalize_embedded_file_map(
        raw.get('embedded_case_files'),
        field='embedded_case_files',
        dataset='toolathlon',
        case_id=case_id,
    )
    embedded_server_files = _normalize_embedded_file_map(
        raw.get('embedded_server_files'),
        field='embedded_server_files',
        dataset='toolathlon',
        case_id=case_id,
    )
    task_dir: Path | None = None
    task_config = raw.get('task_config')
    if task_config is None:
        if embedded_case_files:
            embedded_task_config = _embedded_file_text(
                embedded_case_files,
                'task_config.json',
                field='embedded_case_files',
                dataset='toolathlon',
                case_id=case_id,
            )
            if embedded_task_config is not None:
                task_config = json.loads(embedded_task_config)
        if task_config is None:
            task_dir = _resolve_toolathlon_task_dir(raw, case_id)
            task_config = json.loads((task_dir / 'task_config.json').read_text())
    if not isinstance(task_config, dict):
        raise RuntimeError(f'Invalid Toolathlon task_config for {case_id}')
    server_names = raw.get('server_names')
    if not isinstance(server_names, list):
        server_names = list(task_config.get('needed_mcp_servers', []))
    prompt = str(raw.get('prompt') or '').strip()
    if not prompt and embedded_case_files:
        prompt = (
            _embedded_file_text(
                embedded_case_files,
                'docs/task.md',
                field='embedded_case_files',
                dataset='toolathlon',
                case_id=case_id,
            )
            or ''
        ).strip()
    if not prompt:
        prompt = case_id
    system_prompt = raw.get('system_prompt')
    if system_prompt is None and embedded_case_files:
        embedded_system_prompt = _embedded_file_text(
            embedded_case_files,
            'docs/agent_system_prompt.md',
            field='embedded_case_files',
            dataset='toolathlon',
            case_id=case_id,
        )
        if embedded_system_prompt is not None:
            system_prompt = embedded_system_prompt.strip()

    analysis_fields = (
        'task_override_values',
        'task_override_error',
        'has_preprocess_main',
        'has_preprocess_sh',
        'has_initial_workspace',
        'has_evaluation_main',
        'references_google_creds',
        'references_snowflake_private_key',
        'references_notion_duplication',
    )
    analysis = {field: raw.get(field) for field in analysis_fields if field in raw}
    missing_analysis = not isinstance(analysis.get('task_override_values'), dict) or any(
        field not in analysis for field in analysis_fields[1:]
    )
    if missing_analysis:
        if task_dir is None:
            if embedded_case_files:
                analysis = _toolathlon_embedded_task_analysis(case_id, embedded_case_files)
            else:
                task_dir = _resolve_toolathlon_task_dir(raw, case_id)
                analysis = _toolathlon_task_analysis(task_dir)
        else:
            analysis = _toolathlon_task_analysis(task_dir)

    if task_dir is None and not embedded_case_files:
        task_dir = _resolve_toolathlon_task_dir(raw, case_id)
    source_path = str(raw.get('source_path') or task_dir or f'manifest://toolathlon/{case_id}')
    task_dir_rel = raw.get('task_dir_rel')
    if task_dir_rel is None and task_dir is not None:
        task_dir_rel = _relative_to(TOOLATHLON_DATASET_ROOT, task_dir)
    task_override_error = str(analysis.get('task_override_error') or '') or None
    if task_override_error and task_dir is not None:
        task_override_error = str(_normalize_task_override_value(task_override_error, task_dir=task_dir))
    return {
        'dataset': 'toolathlon',
        'case_id': case_id,
        'prompt': prompt,
        'system_prompt': system_prompt,
        'source_path': source_path,
        'task_dir_rel': str(task_dir_rel or ''),
        'task_config': task_config,
        'server_names': [str(item) for item in server_names],
        'embedded_case_files': embedded_case_files,
        'embedded_server_files': embedded_server_files,
        'task_override_values': {
            str(key): _jsonable(value) for key, value in dict(analysis.get('task_override_values') or {}).items()
        },
        'task_override_error': task_override_error,
        'has_preprocess_main': bool(analysis.get('has_preprocess_main')),
        'has_preprocess_sh': bool(analysis.get('has_preprocess_sh')),
        'has_initial_workspace': bool(analysis.get('has_initial_workspace')),
        'has_evaluation_main': bool(analysis.get('has_evaluation_main')),
        'references_google_creds': bool(analysis.get('references_google_creds')),
        'references_snowflake_private_key': bool(analysis.get('references_snowflake_private_key')),
        'references_notion_duplication': bool(analysis.get('references_notion_duplication')),
    }


def _normalize_atlas_row(raw: dict[str, Any]) -> dict[str, Any]:
    case_id = _require_string(raw.get('case_id') or raw.get('task_id'), field='case_id', dataset='atlas', case_id='')
    enabled_tools = raw.get('enabled_tools') or []
    if not isinstance(enabled_tools, list):
        raise RuntimeError(f'Invalid Atlas enabled_tools for {case_id}')
    claims = raw.get('claims')
    if claims is None:
        evaluator_config = raw.get('evaluator_config') or {}
        claims = evaluator_config.get('claims', [])
    if not isinstance(claims, list):
        raise RuntimeError(f'Invalid Atlas claims for {case_id}')
    server_names = raw.get('server_names')
    if not isinstance(server_names, list):
        metadata = raw.get('metadata') or {}
        server_names = metadata.get('servers')
    if not isinstance(server_names, list):
        server_names = atlas_server_names_from_enabled_tools(enabled_tools)
    source_path = raw.get('source_path')
    if not source_path:
        metadata = raw.get('metadata') or {}
        source_path = metadata.get('source_path')
    return {
        'dataset': 'atlas',
        'case_id': case_id,
        'prompt': str(raw.get('prompt') or case_id).strip(),
        'system_prompt': raw.get('system_prompt'),
        'source_path': str(source_path or ''),
        'enabled_tools': [str(item) for item in enabled_tools],
        'claims': [str(item) for item in claims],
        'trajectory': str(raw.get('trajectory') or ''),
        'server_names': [str(item) for item in server_names],
    }


def _normalize_universe_row(raw: dict[str, Any]) -> dict[str, Any]:
    case_id = _require_string(raw.get('case_id') or raw.get('task_id'), field='case_id', dataset='universe', case_id='')
    task_file = _resolve_universe_task_file(raw, case_id)
    task_data = raw.get('task_data')
    if task_data is None:
        task_data = json.loads(task_file.read_text())
    if not isinstance(task_data, dict):
        raise RuntimeError(f'Invalid Universe task_data for {case_id}')
    server_names = raw.get('server_names')
    if not isinstance(server_names, list):
        server_names = universe_server_names(task_data)
    source_path = raw.get('source_path') or str(task_file)
    category = raw.get('category')
    if category is None:
        metadata = raw.get('metadata') or {}
        category = metadata.get('category')
    if category is None:
        category = task_data.get('category')
    return {
        'dataset': 'universe',
        'case_id': case_id,
        'prompt': str(raw.get('prompt') or task_data.get('question') or case_id).strip(),
        'system_prompt': raw.get('system_prompt'),
        'source_path': str(source_path),
        'task_file_rel': _relative_to(UNIVERSE_DATASET_ROOT, task_file),
        'task_data': task_data,
        'server_names': [str(item) for item in server_names],
        'category': category,
    }


@lru_cache(maxsize=1)
def load_toolathlon_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl(TOOLATHLON_MANIFEST_PATH)
    return tuple(_normalize_toolathlon_row(row) for row in rows)


@lru_cache(maxsize=1)
def load_atlas_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl(ATLAS_MANIFEST_PATH)
    return tuple(_normalize_atlas_row(row) for row in rows)


@lru_cache(maxsize=1)
def load_universe_manifest_rows() -> tuple[dict[str, Any], ...]:
    rows = _read_jsonl(UNIVERSE_MANIFEST_PATH)
    return tuple(_normalize_universe_row(row) for row in rows)


def clear_manifest_caches() -> None:
    load_toolathlon_manifest_rows.cache_clear()
    load_atlas_manifest_rows.cache_clear()
    load_universe_manifest_rows.cache_clear()


__all__ = [
    'ATLAS_MANIFEST_PATH',
    'CONVERTED_DATASETS_ROOT',
    'TOOLATHLON_MANIFEST_PATH',
    'UNIVERSE_MANIFEST_PATH',
    'build_atlas_manifest_rows',
    'build_manifest_rows',
    'build_toolathlon_manifest_rows',
    'build_universe_manifest_rows',
    'clear_manifest_caches',
    'load_atlas_manifest_rows',
    'load_toolathlon_manifest_rows',
    'load_universe_manifest_rows',
    'manifest_path_for_dataset',
    'prepare_manifests',
    'write_manifest',
]
