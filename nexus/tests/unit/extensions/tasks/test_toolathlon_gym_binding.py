from __future__ import annotations

import base64
import io
import json
import tarfile

import pytest

from nexus.extensions.tasks.toolathlon_gym.bind_parquet import bind_nexus_payload
from nexus.extensions.tasks.toolathlon_gym.binding import (
    BACKEND_MOCK_ISOLATION,
    MCP_PLACEMENT_BACKEND,
    MCP_PLACEMENT_HYBRID,
    MCP_PLACEMENT_MAIN,
    SPLIT_WORKSPACE_ISOLATION,
    build_backend_mock_binding,
    build_toolathlon_gym_binding,
    verify_task_binding,
)


def _archive(files: dict[str, str]) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name, text in files.items():
            payload = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return base64.b64encode(buffer.getvalue()).decode()


def _task_config(*, tools_py: str = 'def lookup(db):\n    return db.value\n') -> dict:
    return {
        'case_id': 'mock-case',
        'server_mode': 'mock',
        'prompt': 'Write result.json.',
        'system_prompt': 'Use only MCP tools.',
        'mcp_servers': [
            {
                'name': 'mock',
                'runtime_name': 'main',
                'client': {
                    'transport': 'stdio',
                    'command': 'python',
                    'args': ['/workspace/case/mcp/mock/server.py', '--state-dump', '/workspace/case/dumps/mock.json'],
                },
            }
        ],
        'max_iterations': 10,
        'runtime_name': 'main',
        'evaluator': {'type': 'toolathlon_runtime', 'config': {'command': 'bash tests/test.sh'}},
        'case_data_tar_b64': _archive(
            {
                'mcp/mock/server.py': 'from pathlib import Path\nPath("/protected/dump").write_text("state")\n',
                'mcp/mock/tools.py': tools_py,
                'mcp/mock/manifest.json': json.dumps(
                    {
                        'tools': [
                            {'name': 'lookup', 'mutates': False},
                            {'name': 'update', 'mutates': True},
                        ]
                    }
                ),
            }
        ),
        'eval_tar_b64': _archive({'tests/test.sh': 'exit 0\n'}),
        'groundtruth_tar_b64': None,
    }


def test_backend_mock_binding_records_mutable_state_and_hashes() -> None:
    task = _task_config()
    binding = build_backend_mock_binding(task, dataset='mock-only')

    assert binding['isolation'] == BACKEND_MOCK_ISOLATION
    assert binding['mock_state_access'] == 'read_write'
    assert binding['workspace_access'] == 'none'
    assert binding['output_artifacts'] == ['result.json']
    task['data_binding'] = binding
    assert verify_task_binding(task) == binding


def test_binding_rejects_task_spec_or_archive_tampering() -> None:
    task = _task_config()
    task['data_binding'] = build_backend_mock_binding(task, dataset='mock-only')

    task['prompt'] = 'A modified prompt.'
    with pytest.raises(ValueError, match='task_spec_sha256'):
        verify_task_binding(task)

    task = _task_config()
    task['data_binding'] = build_backend_mock_binding(task, dataset='mock-only')
    task['case_data_tar_b64'] = _archive({'mcp/mock/tools.py': 'def different():\n    return 1\n'})
    with pytest.raises(ValueError, match='case_data_sha256'):
        verify_task_binding(task)


def test_binding_allows_operational_iteration_overrides() -> None:
    task = _task_config()
    task['data_binding'] = build_backend_mock_binding(task, dataset='mock-only')

    task['max_iterations'] = 200
    task['observation_time_timeout'] = 900

    assert verify_task_binding(task) is not None


def test_backend_mock_binding_rejects_workspace_specs_and_tool_file_io() -> None:
    task = _task_config()
    task['mcp_servers'][0]['client']['cwd'] = '${agent_workspace}'
    with pytest.raises(ValueError, match='workspace marker'):
        build_backend_mock_binding(task, dataset='mock-only')

    task = _task_config(tools_py='from pathlib import Path\ndef write():\n    Path("/tmp/out").write_text("x")\n')
    with pytest.raises(ValueError, match='directly accesses host files'):
        build_backend_mock_binding(task, dataset='mock-only')


def test_bind_nexus_payload_embeds_binding_without_changing_other_config() -> None:
    nexus = {'task_name': 'toolathlon_gym', 'task_config': _task_config(), 'agent_name': 'main_agent'}
    bound = bind_nexus_payload(json.dumps(nexus), dataset='toolathlon_synthesis')
    parsed = json.loads(bound)

    assert parsed['agent_name'] == 'main_agent'
    assert parsed['task_config']['data_binding']['dataset'] == 'toolathlon_synthesis'
    assert verify_task_binding(parsed['task_config']) is not None


def test_toolathlon_gym_binding_classifies_main_backend_and_hybrid_servers() -> None:
    task = _task_config()
    task['mcp_servers'] = [
        {'name': 'excel', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'excel'}},
        {'name': 'snowflake', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'snow'}},
        {'name': 'emails', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'email'}},
    ]
    binding = build_toolathlon_gym_binding(task)

    assert binding['isolation'] == SPLIT_WORKSPACE_ISOLATION
    assert binding['server_placements'] == {
        'excel': MCP_PLACEMENT_MAIN,
        'snowflake': MCP_PLACEMENT_BACKEND,
        'emails': MCP_PLACEMENT_HYBRID,
    }
    task['data_binding'] = binding
    assert verify_task_binding(task) == binding


def test_toolathlon_gym_binding_rejects_unknown_servers() -> None:
    task = _task_config()
    task['mcp_servers'] = [
        {'name': 'new-unclassified-server', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'x'}}
    ]
    with pytest.raises(ValueError, match='unknown Toolathlon Gym MCP server'):
        build_toolathlon_gym_binding(task)


def test_bind_nexus_payload_supports_toolathlon_gym() -> None:
    task = _task_config()
    task['mcp_servers'] = [
        {'name': 'filesystem', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'fs'}},
        {'name': 'notion', 'runtime_name': 'main', 'client': {'transport': 'stdio', 'command': 'notion'}},
    ]
    nexus = {'task_name': 'toolathlon_gym', 'task_config': task}
    parsed = json.loads(bind_nexus_payload(json.dumps(nexus), dataset='toolathlon_gym'))

    assert parsed['task_config']['data_binding']['isolation'] == SPLIT_WORKSPACE_ISOLATION
