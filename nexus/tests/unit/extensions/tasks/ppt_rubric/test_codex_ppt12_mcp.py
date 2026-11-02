from __future__ import annotations

import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pytest  # noqa: E402

from nexus.extensions.tasks.mm_agent.utils import TaskArtifact  # noqa: E402
from nexus.extensions.tasks.ppt_rubric.codex_ppt12_mcp import (  # noqa: E402
    LD_LIBRARY_PATH_PLACEHOLDER,
    MCP_SERVER_MODULE,
    MCP_SERVER_NAME,
    WORKER_PYTHON_PLACEHOLDER,
    build_ppt12_mcp_server,
)
from nexus.extensions.tasks.ppt_rubric.task import PPTTaskInput  # noqa: E402


def _task_input(**kwargs) -> PPTTaskInput:
    defaults: dict = {
        'question': 'Reproduce the slide.',
        'working_dir': '/workspace',
        'deck_filename': 'slide.pptx',
        'run_agent': AsyncMock(),
    }
    defaults.update(kwargs)
    return PPTTaskInput(**defaults)


def _runtime(venv_id: str | None = 'venv0ab12') -> SimpleNamespace:
    return SimpleNamespace(default_venv_id=venv_id)


def test_codex_row_builds_the_full_server_entry():
    server = build_ppt12_mcp_server(
        agent_name='codex',
        task_input=_task_input(),
        artifacts=[TaskArtifact(source='/host/slide.png', target='/workspace/inputs/slide.png')],
        runtime=_runtime(),
    )
    assert server == {
        'name': MCP_SERVER_NAME,
        'command': WORKER_PYTHON_PLACEHOLDER,
        'args': ['-m', MCP_SERVER_MODULE],
        'env': {
            'PPT12_WORKSPACE': '/workspace',
            'PPT12_DECK_FILENAME': 'slide.pptx',
            'PPT12_DUMP_DIR': '/workspace/ppt12_toolimg',
            'PPT12_MCP_LOG': '/workspace/ppt12_mcp.log',
            'LD_LIBRARY_PATH': LD_LIBRARY_PATH_PLACEHOLDER,
            'PPT12_REFERENCE_PNG': '/workspace/inputs/slide.png',
        },
        'startup_timeout_sec': 120,
        'tool_timeout_sec': 600,
        'required': True,
    }


def test_row_without_an_image_artifact_omits_the_reference():
    server = build_ppt12_mcp_server(
        agent_name='codex',
        task_input=_task_input(),
        artifacts=[TaskArtifact(source='/host/brief.pdf', target='/workspace/inputs/brief.pdf')],
        runtime=_runtime(),
    )
    assert server is not None
    assert 'PPT12_REFERENCE_PNG' not in server['env']
    assert server['env']['PPT12_WORKSPACE'] == '/workspace'


def test_deck_filename_stays_raw_for_subdirectory_rows():
    """The server joins workspace + name exactly like the loop's os.path.join, so a
    sub-directory deck_filename must not be flattened to its basename."""
    server = build_ppt12_mcp_server(
        agent_name='codex',
        task_input=_task_input(deck_filename='nested/slide.pptx'),
        artifacts=[],
        runtime=_runtime(),
    )
    assert server is not None
    assert server['env']['PPT12_DECK_FILENAME'] == 'nested/slide.pptx'


def test_non_codex_harness_raises_instead_of_running_toolless():
    """Review decision: warn-and-None let a non-codex harness run the arm with
    no tools and still land in the scores; it must fail like the missing-venv
    case below."""
    with pytest.raises(ValueError, match='ppt12'):
        build_ppt12_mcp_server(
            agent_name='claude_code',
            task_input=_task_input(agent_tools=['ppt12']),
            artifacts=[],
            runtime=_runtime(),
        )


def test_missing_worker_venv_fails_loud_with_the_fix_in_the_message():
    with pytest.raises(ValueError, match='worker_venv'):
        build_ppt12_mcp_server(
            agent_name='codex',
            task_input=_task_input(),
            artifacts=[],
            runtime=_runtime(venv_id=None),
        )


def test_runtime_without_the_attribute_at_all_fails_the_same_way():
    with pytest.raises(ValueError, match='ppt_rubric'):
        build_ppt12_mcp_server(
            agent_name='codex',
            task_input=_task_input(),
            artifacts=[],
            runtime=SimpleNamespace(),
        )
