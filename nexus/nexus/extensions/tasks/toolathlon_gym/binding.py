"""Self-contained task/data binding for isolated Toolathlon-Gym rows.

The binding is embedded in ``nexus.task_config`` inside the parquet row.  It
serves two purposes:

* make the sandbox placement contract data-driven instead of relying on a
  process-wide environment variable;
* fail closed if the task spec or any embedded case/evaluation archive is
  changed independently after the row was produced.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import tarfile

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any


BINDING_VERSION = 1
BACKEND_MOCK_ISOLATION = 'backend_mock'
MAIN_WORKSPACE_ISOLATION = 'main_workspace'
SPLIT_WORKSPACE_ISOLATION = 'split_workspace'
MOCK_ONLY_OUTPUT_ARTIFACTS = ('result.json',)

MCP_PLACEMENT_MAIN = 'main'
MCP_PLACEMENT_BACKEND = 'backend'
MCP_PLACEMENT_HYBRID = 'hybrid'

# Traditional Toolathlon Gym uses a small, fixed server set.  Placement is
# intentionally explicit and embedded into every parquet row; unknown servers
# fail conversion instead of silently landing in an unsafe sandbox.
TOOLATHLON_GYM_SERVER_PLACEMENTS = {
    # The state of these tools is the agent workspace itself (or they are
    # stateless browser/fetch tools whose output/cache belongs with the agent).
    'filesystem': MCP_PLACEMENT_MAIN,
    'terminal': MCP_PLACEMENT_MAIN,
    'excel': MCP_PLACEMENT_MAIN,
    'word': MCP_PLACEMENT_MAIN,
    'pptx': MCP_PLACEMENT_MAIN,
    'pdf-tools': MCP_PLACEMENT_MAIN,
    'memory': MCP_PLACEMENT_MAIN,
    'playwright_with_chunk': MCP_PLACEMENT_MAIN,
    'fetch': MCP_PLACEMENT_MAIN,
    # Protected mock/service state stays in the backend.
    'snowflake': MCP_PLACEMENT_BACKEND,
    'notion': MCP_PLACEMENT_BACKEND,
    'google_calendar': MCP_PLACEMENT_BACKEND,
    'google_forms': MCP_PLACEMENT_BACKEND,
    'google_sheet': MCP_PLACEMENT_BACKEND,
    'woocommerce': MCP_PLACEMENT_BACKEND,
    'yahoo-finance': MCP_PLACEMENT_BACKEND,
    'arxiv-latex': MCP_PLACEMENT_BACKEND,
    'scholarly': MCP_PLACEMENT_BACKEND,
    'howtocook': MCP_PLACEMENT_BACKEND,
    'youtube': MCP_PLACEMENT_BACKEND,
    'rail_12306': MCP_PLACEMENT_BACKEND,
    'youtube-transcript': MCP_PLACEMENT_BACKEND,
    # These own protected state but also expose local file upload/download
    # operations.  The task installs a serialized mirror bridge per call.
    'emails': MCP_PLACEMENT_HYBRID,
    'canvas': MCP_PLACEMENT_HYBRID,
    'arxiv_local': MCP_PLACEMENT_HYBRID,
}

_TASK_SPEC_FIELDS = (
    'case_id',
    'server_mode',
    'prompt',
    'system_prompt',
    'mcp_servers',
    'enabled_tools',
    'initial_workspace_dir',
    'bootstraps',
    'preprocess',
    'evaluator',
    'metadata',
)
_WORKSPACE_MARKERS = ('agent_workspace', 'workspace_dir')
_SQL_MUTATION_WORDS = ('INSERT', 'UPDATE', 'DELETE', 'REPLACE', 'ALTER', 'DROP', 'CREATE')


def _canonical_value(value: Any) -> Any:
    # Pydantic normalizes numeric fields such as ``600`` to ``600.0``.  Treat
    # integral floats as the same semantic value so verification is stable
    # before and after TaskRunInput validation.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decoded_archive_sha256(value: str | None) -> str | None:
    if not value:
        return None
    try:
        payload = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError('embedded archive is not valid base64') from exc
    return _sha256_bytes(payload)


def task_spec_payload(task_config: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, security-relevant part of a Toolathlon task spec."""
    return {field: task_config[field] for field in _TASK_SPEC_FIELDS if field in task_config}


def task_spec_sha256(task_config: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(task_spec_payload(task_config)))


def _safe_artifact_path(path: str) -> str:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or '..' in pure.parts or path in {'.', '..'}:
        raise ValueError(f'output artifact must be a safe relative path: {path!r}')
    return str(pure)


def validate_output_artifacts(paths: Iterable[str]) -> tuple[str, ...]:
    values = tuple(_safe_artifact_path(path) for path in paths)
    if not values:
        raise ValueError('at least one output artifact is required')
    if len(values) != len(set(values)):
        raise ValueError('output artifacts must be unique')
    return values


def _iter_archive_text(archive_b64: str) -> Iterable[tuple[str, str]]:
    try:
        payload = base64.b64decode(archive_b64, validate=True)
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            entries = []
            for member in archive.getmembers():
                if not member.isfile() or member.size > 8_000_000:
                    continue
                if not member.name.endswith(('.py', '.json', '.yaml', '.yml', '.toml', '.sh')):
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    entries.append((member.name, handle.read().decode('utf-8', errors='replace')))
    except Exception as exc:
        raise ValueError('case_data_tar_b64 is not a valid tar.gz archive') from exc
    yield from entries


def _mcp_source_files(task_config: dict[str, Any]) -> Iterable[tuple[str, str]]:
    archive_b64 = task_config.get('case_data_tar_b64')
    if not archive_b64:
        return ()
    return (
        (name, text)
        for name, text in _iter_archive_text(archive_b64)
        if name == 'tools.py' or name.startswith('mcp_server/') or name.startswith('mcp/') or '/mcp/' in name
    )


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ''


def _qualified_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if not isinstance(node.func, ast.Attribute):
        return ''
    parts = [node.func.attr]
    value = node.func.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return '.'.join(reversed(parts))


def _has_real_file_io(name: str, text: str) -> bool:
    """Detect direct host-filesystem access in MCP tool implementations.

    Relative paths returned as mock values are intentionally allowed.  The
    forbidden cases here are actual Python file APIs in the tool implementation.
    Package-local reads of manifest/tool_schemas are allowed.
    """
    if not name.endswith('.py'):
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f'cannot audit MCP source {name}: {exc}') from exc
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = _call_name(node)
        qualified_call = _qualified_call_name(node)
        if qualified_call in {
            'os.remove',
            'os.unlink',
            'os.rename',
            'os.replace',
            'os.makedirs',
            'shutil.copy',
            'shutil.copy2',
            'shutil.copyfile',
            'shutil.copytree',
            'shutil.move',
            'shutil.rmtree',
        }:
            return True
        if call in {'write_text', 'write_bytes'}:
            return True
        if call in {'unlink', 'mkdir', 'rmdir', 'touch'}:
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            # Only flag an explicit pathlib-style expression.  Mock State models
            # may legitimately expose similarly named methods.
            if isinstance(receiver, ast.Call) and _qualified_call_name(receiver) in {'Path', 'pathlib.Path'}:
                return True
        if call != 'open':
            continue
        mode: str | None = None
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            mode = node.args[1].value
        for keyword in node.keywords:
            if (
                keyword.arg == 'mode'
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                mode = keyword.value.value
        if mode and any(flag in mode for flag in 'wax+'):
            return True
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ''
        if 'tool_schemas.json' not in line and 'manifest.json' not in line:
            return True
    return False


def audit_backend_mock_task(task_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and describe a task whose MCPs must remain in the backend."""
    if task_config.get('initial_workspace_dir'):
        raise ValueError('backend_mock tasks cannot seed an MCP-visible agent workspace')

    rendered_specs = json.dumps(task_config.get('mcp_servers') or [], ensure_ascii=False).lower()
    marker = next((value for value in _WORKSPACE_MARKERS if value in rendered_specs), None)
    if marker:
        raise ValueError(f'backend_mock MCP spec references forbidden workspace marker: {marker}')

    mutating_state = False
    audited_files = 0
    for name, text in _mcp_source_files(task_config):
        audited_files += 1
        # The generic server is expected to persist protected mock state (for
        # example ``--state-dump``).  Only tool implementations are forbidden
        # from reaching out to arbitrary host/workspace files.
        if (name == 'tools.py' or name.endswith('/tools.py')) and _has_real_file_io(name, text):
            raise ValueError(f'backend_mock MCP directly accesses host files: {name}')
        if name.endswith('/manifest.json'):
            try:
                manifest = json.loads(text)
            except json.JSONDecodeError:
                manifest = {}
            if any(tool.get('mutates') for tool in manifest.get('tools', []) if isinstance(tool, dict)):
                mutating_state = True
        if name == 'tools.py' or name.endswith('/tools.py'):
            upper = text.upper()
            if '.COMMIT(' in upper or any(f'{word} ' in upper for word in _SQL_MUTATION_WORDS):
                mutating_state = True

    return {
        'audited_mcp_files': audited_files,
        'mock_state_access': 'read_write' if mutating_state else 'read_only',
        'workspace_access': 'none',
    }


def build_backend_mock_binding(
    task_config: dict[str, Any],
    *,
    dataset: str,
    output_artifacts: Iterable[str] = MOCK_ONLY_OUTPUT_ARTIFACTS,
) -> dict[str, Any]:
    """Build the binding embedded into a self-contained parquet row."""
    artifacts = validate_output_artifacts(output_artifacts)
    audit = audit_backend_mock_task(task_config)
    return {
        'version': BINDING_VERSION,
        'dataset': dataset,
        'isolation': BACKEND_MOCK_ISOLATION,
        'task_spec_sha256': task_spec_sha256(task_config),
        'case_data_sha256': _decoded_archive_sha256(task_config.get('case_data_tar_b64')),
        'eval_data_sha256': _decoded_archive_sha256(task_config.get('eval_tar_b64')),
        'groundtruth_data_sha256': _decoded_archive_sha256(task_config.get('groundtruth_tar_b64')),
        'output_artifacts': list(artifacts),
        **audit,
    }


def build_toolathlon_gym_binding(task_config: dict[str, Any], *, dataset: str = 'toolathlon_gym') -> dict[str, Any]:
    """Build a split main/backend/hybrid binding for traditional Gym tasks."""
    placements: dict[str, str] = {}
    for server in task_config.get('mcp_servers') or []:
        name = server.get('name')
        if name not in TOOLATHLON_GYM_SERVER_PLACEMENTS:
            raise ValueError(f'unknown Toolathlon Gym MCP server placement: {name!r}')
        placements[name] = TOOLATHLON_GYM_SERVER_PLACEMENTS[name]
    if not placements:
        raise ValueError('Toolathlon Gym binding requires at least one MCP server')
    if MCP_PLACEMENT_MAIN not in placements.values():
        raise ValueError('Toolathlon Gym binding requires at least one main workspace MCP')

    return {
        'version': BINDING_VERSION,
        'dataset': dataset,
        'isolation': SPLIT_WORKSPACE_ISOLATION,
        'task_spec_sha256': task_spec_sha256(task_config),
        'case_data_sha256': _decoded_archive_sha256(task_config.get('case_data_tar_b64')),
        'eval_data_sha256': _decoded_archive_sha256(task_config.get('eval_tar_b64')),
        'groundtruth_data_sha256': _decoded_archive_sha256(task_config.get('groundtruth_tar_b64')),
        'server_placements': placements,
        'workspace_owner': MCP_PLACEMENT_MAIN,
        'workspace_sync': 'clean_mirror',
    }


def verify_task_binding(task_config: dict[str, Any]) -> dict[str, Any] | None:
    """Verify an embedded binding and return it.

    Legacy rows without a binding retain their existing environment-gated
    behavior.  Bound rows fail closed on any mismatch.
    """
    binding = task_config.get('data_binding')
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError('data_binding must be an object')
    if binding.get('version') != BINDING_VERSION:
        raise ValueError(f'unsupported data_binding version: {binding.get("version")!r}')
    if binding.get('isolation') not in {
        BACKEND_MOCK_ISOLATION,
        MAIN_WORKSPACE_ISOLATION,
        SPLIT_WORKSPACE_ISOLATION,
    }:
        raise ValueError(f'unsupported data_binding isolation: {binding.get("isolation")!r}')

    expected = {
        'task_spec_sha256': task_spec_sha256(task_config),
        'case_data_sha256': _decoded_archive_sha256(task_config.get('case_data_tar_b64')),
        'eval_data_sha256': _decoded_archive_sha256(task_config.get('eval_tar_b64')),
        'groundtruth_data_sha256': _decoded_archive_sha256(task_config.get('groundtruth_tar_b64')),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f'data_binding mismatch for {key}')
    if binding['isolation'] == BACKEND_MOCK_ISOLATION:
        binding['output_artifacts'] = list(validate_output_artifacts(binding.get('output_artifacts') or ()))
        audit = audit_backend_mock_task(task_config)
        for key in ('mock_state_access', 'workspace_access'):
            if binding.get(key) != audit[key]:
                raise ValueError(f'data_binding audit mismatch for {key}')
    elif binding['isolation'] == SPLIT_WORKSPACE_ISOLATION:
        expected_binding = build_toolathlon_gym_binding(
            {key: value for key, value in task_config.items() if key != 'data_binding'},
            dataset=binding.get('dataset') or 'toolathlon_gym',
        )
        for key in ('server_placements', 'workspace_owner', 'workspace_sync'):
            if binding.get(key) != expected_binding[key]:
                raise ValueError(f'data_binding placement mismatch for {key}')
    return binding
