from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.mac_os_world.evaluator import evaluate_mac_os_world
from nexus.extensions.tasks.mac_os_world.task import _command_result_from_response
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'response',
    [
        {'stdout': 'True', 'return_code': 1, 'stderr': 'script failed'},
        {'stdout': 'True', 'return_code': 0, 'status': 'TimeLimitExceeded'},
        {'stdout': 'True', 'status': 'Error'},
        {'stdout': 'True'},
    ],
)
async def test_grading_rejects_unsuccessful_commands_and_stops(response: dict) -> None:
    runtime = SimpleNamespace(run_command=AsyncMock(return_value=response))
    result = await evaluate_mac_os_world(runtime, [['broken', 100], ['echo True', 100]])
    assert result['score'] == 0
    assert result['error']
    assert result['results'][0]['return_code'] == response.get('return_code')
    assert result['results'][0]['stderr'] == response.get('stderr', '')
    runtime.run_command.assert_awaited_once_with('broken', timeout=60)
    assert not _command_result_from_response(response)['success']


@pytest.mark.asyncio
async def test_grading_preserves_official_substring_match_and_short_circuit() -> None:
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                {'output': 'False', 'returncode': 0},
                RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='Chess\nTrue\n'),
            ]
        )
    )
    result = await evaluate_mac_os_world(runtime, [['partial', 50], ['false', 100], ['true', 100], ['later', 100]])
    assert result['score'] == 1
    assert 'error' not in result
    assert runtime.run_command.await_count == 2
    assert result['results'][1]['status'] == 'Finished'


@pytest.mark.asyncio
async def test_grading_transport_failure_stops_before_later_true() -> None:
    runtime = SimpleNamespace(
        run_command=AsyncMock(side_effect=[TimeoutError('request deadline'), {'stdout': 'True', 'return_code': 0}])
    )
    result = await evaluate_mac_os_world(runtime, [['unreachable', 100], ['later', 100]])
    assert result['score'] == 0
    assert result['error'] == 'TimeoutError: request deadline'
    assert runtime.run_command.await_count == 1


@pytest.mark.asyncio
async def test_partial_grading_preserves_weighted_score() -> None:
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                {'stdout': 'True', 'return_code': 0},
                {'stdout': 'False', 'return_code': 0},
            ]
        )
    )
    result = await evaluate_mac_os_world(runtime, [['yes', 25], ['no', 75]], binary_grading=False)
    assert result['score'] == 0.25
    assert runtime.run_command.await_count == 2
