"""OSWorld evaluator integrity regressions."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.os_world.evaluation import evaluator
from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.execution import setup
from nexus.extensions.tasks.os_world.getters import chrome, general, vlc
from nexus.extensions.tasks.os_world.getters.runtime import get_file_bytes
from nexus.extensions.tasks.os_world.metrics.python_evaluator import check_python_file_by_test_suite


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 403, 500, 503, None])
async def test_file_transport_errors_are_not_missing_artifacts(status):
    exc = RuntimeError('sensitive remote detail')
    exc.status = status
    with pytest.raises(EvaluatorInfrastructureError, match='download failed') as caught:
        await get_file_bytes(SimpleNamespace(get_file=AsyncMock(side_effect=exc)), '/tmp/output')
    assert 'sensitive' not in str(caught.value)


@pytest.mark.asyncio
async def test_true_404_remains_missing_output():
    exc = RuntimeError('not found')
    exc.status = 404
    with pytest.raises(FileNotFoundError):
        await get_file_bytes(SimpleNamespace(get_file=AsyncMock(side_effect=exc)), '/tmp/output')


@pytest.mark.asyncio
@pytest.mark.parametrize('multi', [False, True])
async def test_expected_fixture_failure_cannot_score_zero(monkeypatch, tmp_path, multi):
    monkeypatch.setattr(evaluator, '_get_state', AsyncMock(side_effect=['actual', FileNotFoundError('gold')]))
    monkeypatch.setattr(evaluator, 'METRIC_REGISTRY', {'fixture_test': lambda *a: 1})
    config = {'func': ['fixture_test'] if multi else 'fixture_test'}
    with pytest.raises(EvaluatorInfrastructureError, match='expected getter'):
        await evaluator.evaluate_task(None, config, 'DONE', cache_dir=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('multi', [False, True])
async def test_missing_submitted_artifact_is_still_zero(monkeypatch, tmp_path, multi):
    monkeypatch.setattr(evaluator, '_get_state', AsyncMock(side_effect=FileNotFoundError('submitted')))
    monkeypatch.setattr(evaluator, 'METRIC_REGISTRY', {'fixture_test': lambda *a: 1})
    config = {'func': ['fixture_test'] if multi else 'fixture_test'}
    assert await evaluator.evaluate_task(None, config, 'DONE', cache_dir=tmp_path) == 0


def test_python_checker_import_error_is_infrastructure(tmp_path):
    fixture = tmp_path / 'test.py'
    fixture.write_text('import nonexistent_osworld_host_dependency\ndef test(): return True\n')
    with pytest.raises(EvaluatorInfrastructureError, match='dependency'):
        check_python_file_by_test_suite([], fixture)


def test_python_checker_negative_score_is_not_infrastructure(tmp_path):
    fixture = tmp_path / 'test.py'
    fixture.write_text('def test():\n    assert False\n')
    assert check_python_file_by_test_suite([], fixture) == 0


@pytest.mark.asyncio
async def test_python_checkers_are_process_isolated(tmp_path):
    cwd, path, modules = os.getcwd(), sys.path[:], set(sys.modules)
    fixtures = []
    for i in range(6):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / 'settings.py').write_text(f'VALUE={i}\n')
        fixture = directory / 'test.py'
        fixture.write_text(
            f'import settings\nimport time\ndef test():\n    time.sleep(.05)\n    return settings.VALUE == {i}\n'
        )
        fixtures.append(fixture)
    assert (
        await asyncio.gather(*(asyncio.to_thread(check_python_file_by_test_suite, [], f) for f in fixtures)) == [1] * 6
    )
    assert os.getcwd() == cwd
    assert sys.path == path
    assert 'settings' not in set(sys.modules) - modules


@pytest.mark.asyncio
async def test_vlc_prefers_real_gui_player_over_dummy_http(monkeypatch, tmp_path):
    run = AsyncMock(
        return_value={
            'returncode': 0,
            'output': json.dumps({
                'players': [
                    {'state': 'playing', 'url': 'vlc://pause:2147483647'},
                    {'state': 'playing', 'url': 'file:///home/user/My%20Video.mp4', 'title': 'Video'},
                ]
            }),
        }
    )
    monkeypatch.setattr(vlc, 'run_bash', run)
    path = await vlc.get_vlc_playing_info(SimpleNamespace(), {'__nexus_cache_dir': str(tmp_path)})
    content = Path(path).read_text()
    assert 'My Video.mp4' in content
    assert 'pause:' not in content
    assert run.await_count == 1
    from nexus.extensions.tasks.os_world.metrics.official.vlc import is_vlc_playing

    assert is_vlc_playing(path, {'type': 'file_name', 'file_name': 'My Video.mp4'}) == 1
    assert is_vlc_playing(path, {'type': 'file_name', 'file_name': 'Wrong.mp4'}) == 0


@pytest.mark.asyncio
async def test_vlc_does_not_pick_expected_answer_from_multiple_players(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vlc,
        'run_bash',
        AsyncMock(
            return_value={
                'output': json.dumps({
                    'players': [
                        {'state': 'playing', 'url': 'file:///a.mp4'},
                        {'state': 'playing', 'url': 'file:///b.mp4'},
                    ]
                })
            }
        ),
    )
    with pytest.raises(EvaluatorInfrastructureError, match='ambiguous'):
        await vlc.get_vlc_playing_info(SimpleNamespace(), {'__nexus_cache_dir': str(tmp_path)})


@pytest.mark.asyncio
async def test_vlc_absent_is_negative_but_failed_observation_is_not(monkeypatch, tmp_path):
    run = AsyncMock(side_effect=[{'output': '{"players":[]}'}, {'output': '', 'returncode': 7}])
    monkeypatch.setattr(vlc, 'run_bash', run)
    assert await vlc.get_vlc_playing_info(SimpleNamespace(), {'__nexus_cache_dir': str(tmp_path)}) is None
    from nexus.extensions.tasks.os_world.metrics.official.vlc import is_vlc_playing

    assert is_vlc_playing(None, {'type': 'url', 'url': 'https://example.test/video'}) == 0
    run.side_effect = [{'output': '{"probe_error":"DBusException"}'}, {'output': '', 'returncode': 7}]
    with pytest.raises(EvaluatorInfrastructureError, match='observation failed'):
        await vlc.get_vlc_playing_info(SimpleNamespace(), {'__nexus_cache_dir': str(tmp_path)})


@pytest.mark.parametrize('ref', ['A1:G36', '$A$1:$G$36', 'Sheet1!A1:G36', "'Sheet1'!$A$1:$G$36"])
def test_pivot_equivalent_coordinates(ref):
    from nexus.extensions.tasks.os_world.metrics.pivot_range import _pivot_source_coordinates

    assert _pivot_source_coordinates(ref, 'Sheet1') == ((1, 1), (36, 7))


@pytest.mark.parametrize('ref', ['Wrong!A1:G36', '[other]Sheet1!A1:G36', 'A1', 'G36:A1', 'A0:G36', 'A1:XFE1'])
def test_pivot_invalid_or_conflicting_reference_rejected(ref):
    from nexus.extensions.tasks.os_world.metrics.pivot_range import _pivot_source_coordinates

    with pytest.raises(ValueError, match='.+'):
        _pivot_source_coordinates(ref, 'Sheet1')


def test_pygame_is_declared_as_host_dependency():
    import tomllib

    root = Path(__file__).resolve().parents[5]
    config = tomllib.loads((root / 'nexus/extensions/tasks/os_world/pyproject.toml').read_text())
    assert any(dep.startswith('pygame') for dep in config['project']['optional-dependencies']['host'])


def test_vlc_already_desktop_user_does_not_invoke_root_only_runuser(tmp_path):
    import subprocess

    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_id = fake_bin / 'id'
    fake_id.write_text('#!/bin/sh\nprintf "1000\\n"\n')
    fake_id.chmod(0o755)
    result = subprocess.run(
        ['bash', '-c', vlc._vlc_probe_command('print("desktop-probe-ok")')],
        env={**os.environ, 'PATH': str(fake_bin) + ':' + os.environ['PATH']},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == 'desktop-probe-ok'


async def test_terminal_payload_and_errors_are_preserved():
    payload = {'output': 'hello', 'status': 'success'}
    runtime = SimpleNamespace(terminal_output=AsyncMock(return_value=payload))
    assert await general.get_vm_terminal_output(runtime, {}) == payload
    runtime.terminal_output.side_effect = ConnectionError('transport')
    with pytest.raises(ConnectionError, match='transport'):
        await general.get_vm_terminal_output(runtime, {})


async def test_postconfig_is_not_replaced_by_process_killing(monkeypatch, tmp_path):
    actions = [{'type': 'sleep', 'parameters': {'seconds': 0}}]
    run = AsyncMock(return_value=[])
    monkeypatch.setattr(evaluator, '_run_postconfig', run)
    await evaluator._evaluate_task_outcome(object(), {'postconfig': actions, 'func': 'infeasible'}, 'FAIL', tmp_path)
    assert run.call_args.args[1] is actions


async def test_postconfig_commands_remain_literal(monkeypatch, tmp_path):
    execute = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='', stderr=''))
    monkeypatch.setattr(evaluator, 'run_bash', execute)
    command = 'echo "{CLIENT_PASSWORD} {WIDTH} {HEIGHT}"'
    await evaluator._run_postconfig_action(object(), {'type': 'execute', 'parameters': {'command': command}}, tmp_path)
    assert execute.call_args.args[1] == command


def test_proxy_install_has_one_policy():
    command = setup._tinyproxy_install_command('test-only')
    assert 'lock_wait_seconds' not in command
    assert 'pkill -x tinyproxy' in command


async def test_missing_accessibility_tree_does_not_probe_cdp(monkeypatch):
    monkeypatch.setattr(chrome, 'get_accessibility_tree', AsyncMock(return_value=None))
    probe = AsyncMock(side_effect=AssertionError('must not probe CDP'))
    monkeypatch.setattr(chrome, '_cdp_connection', probe)
    assert await chrome.get_active_url_from_access_tree(object(), {}) is None
    probe.assert_not_awaited()


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        ({'success': True, 'task_output': {'status': 'success', 'score': 0}}, True),
        ({'task_output': {'status': 'success', 'score': 1}, 'agent_output': {'conclusion': 'infeasible'}}, True),
        ({'task_output': {'status': 'success', 'score': 0.5}, 'agent_output': {'exit_status': 'max_iterations'}}, True),
        ({'success': False, 'task_output': None, 'agent_output': {'exit_status': 'tool_error'}}, False),
        ({'success': True, 'task_output': {'score': 0}, 'error': {'message': 'evaluator failed'}}, False),
        ({'success': True, 'task_output': {}}, False),
        ({'success': True, 'task_output': {'status': 'judge_error', 'score': 0}}, False),
        ({'success': True, 'task_output': {'status': 'agent_error', 'score': 0}}, False),
        ({}, False),
    ],
)
def test_platform_validity_contract(result, expected):
    rule_engine = pytest.importorskip('rule_engine')
    root = Path(__file__).resolve().parents[5]
    runbook = (root / 'docs/how-to/osworld-runbook.md').read_text()
    config = next(
        json.loads(block.split('```', 1)[0])
        for block in runbook.split('```json\n')[1:]
        if '"nexus_run_valid_rule"' in block.split('```', 1)[0]
    )
    rule = rule_engine.Rule(config['nexus_run_valid_rule'], context=rule_engine.Context(default_value=None))
    assert rule.matches(result) is expected
