import asyncio
import json
import os
import sys
import tomllib

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import psutil
import pytest

from nexus.extensions.agents.kimi_code import runtime_setup
from nexus.extensions.agents.kimi_code.agent import KimiCodeOutput
from nexus.extensions.agents.kimi_code.output import parse_output
from nexus.extensions.agents.kimi_code.process import ProcessResult, run_process
from nexus.extensions.agents.kimi_code.runtime_setup import build_config, build_env
from nexus.extensions.harbor_bridge.agent_result import build_t3_agent_result


def transcript(*events):
    return '\n'.join(json.dumps(event) for event in events)


def test_zero_exit_without_completion_marker_is_not_success():
    output = parse_output(transcript({'role': 'assistant', 'content': 'working'}), 0)
    assert not output['success']
    assert output['failure_stage'] == 'runtime_protocol'


def test_final_error_overrides_a_previous_resume_hint():
    output = parse_output(
        transcript(
            {'type': 'session.resume_hint', 'session_id': 's'},
            {'type': 'error', 'message': 'backend unavailable'},
        ),
        0,
    )
    assert not output['success']
    assert output['detail'] == 'backend unavailable'


def test_tool_results_and_retries_do_not_become_assistant_text():
    output = parse_output(
        transcript(
            {'type': 'system.version', 'version': '2.3.2', 'role': 'meta'},
            {'role': 'assistant', 'tool_calls': [{'id': 't1', 'function': {'name': 'Bash', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 't1', 'content': 'tool only'},
            {'role': 'meta', 'type': 'turn.step.retrying', 'error_message': 'temporary'},
            {'role': 'assistant', 'content': 'done'},
            {'role': 'meta', 'type': 'session.resume_hint', 'session_id': 's'},
        ),
        0,
    )
    assert output['success']
    assert output['assistant_text'] == 'done'
    assert output['trace']['tool_calls'] == {'Bash': 1}
    assert output['trace']['tool_results'] == 1
    assert output['trace']['retries'] == 1


def test_timeout_survives_harbor_result_conversion():
    parsed = parse_output('', -1, timed_out=True)
    detail = parsed.pop('detail')
    output = KimiCodeOutput(output='', stderr=detail, exit_code=-1, execution_time=1, **parsed)
    result = build_t3_agent_result(mode='agent', agent_output=output)
    assert result.status == 'timeout'
    assert result.agent_exit_status == 'timeout'


def test_model_configuration_preserves_alias_and_thinking_scope():
    config = tomllib.loads(
        build_config(
            model_id='model/"quoted"',
            proxy_url='http://127.0.0.1:1234/v1',
            context_window=1000000,
            max_tokens=16384,
            thinking_effort='high',
            max_steps=200,
        )
    )
    assert config['providers']['nexus']['type'] == 'openai'
    assert config['models']['nexus']['model'] == 'model/"quoted"'
    assert config['thinking'] == {'enabled': True, 'effort': 'high', 'keep': 'all'}
    assert config['loop_control']['max_steps_per_turn'] == 200


def test_routing_environment_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('KIMI_MODEL_API_KEY', 'other-provider-secret')
    monkeypatch.setenv('KIMI_SECONDARY_MODEL', 'other-model')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'other-secret')
    env = build_env(tmp_path, {'KIMI_CODE_HOME': '/wrong', 'KIMI_MODEL_NAME': 'wrong', 'LANG': 'C.UTF-8'})
    assert env['KIMI_CODE_HOME'] == str(tmp_path)
    assert not any(
        k in env for k in ['KIMI_MODEL_API_KEY', 'KIMI_SECONDARY_MODEL', 'ANTHROPIC_API_KEY', 'KIMI_MODEL_NAME']
    )
    assert env['LANG'] == 'C.UTF-8'
    assert env['KIMI_CODE_NO_AUTO_UPDATE'] == '1'


@pytest.mark.asyncio
@pytest.mark.parametrize('unit', ['x', '中文🙂'])
async def test_setup_passes_long_prompt_as_a_private_file(tmp_path, monkeypatch, unit):
    import nexus.runtime.context as runtime_context
    import nexus.runtime.server.services.openai_proxy as proxy_module

    task = '  --model literal\r\n' + unit * 140000 + '\n"quoted" \\tail\n\n'
    run_root = tmp_path / 'run'
    run_root.mkdir()
    workspace = tmp_path / 'workspace'
    entry = tmp_path / 'main.mjs'
    entry.write_text('// CLI fixture\n')
    node = Path('/fixture/node')
    monkeypatch.setattr(runtime_setup.tempfile, 'mkdtemp', lambda **kwargs: str(run_root))
    monkeypatch.setattr(runtime_setup, 'install_kimi_code', AsyncMock(return_value=(node, entry)))
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(proxy_module, 'OpenAIProxy', lambda **kwargs: proxy)
    run = AsyncMock(
        side_effect=[
            ProcessResult('2.3.2\n', '', 0, False),
            ProcessResult(transcript({'type': 'session.resume_hint', 'session_id': 's'}), '', 0, False),
        ]
    )
    monkeypatch.setattr(runtime_setup, 'run_process', run)

    result = await runtime_setup.setup_and_run_kimi_code(
        task=task, working_dir=str(workspace), timeout=30, agent_profile='coder'
    )

    assert result['success']
    argv = run.await_args_list[1].args[0]
    assert argv == [
        str(node),
        str(Path(runtime_setup.__file__).with_name('_prompt_launcher.mjs')),
        str(entry),
        str(run_root / 'prompt.txt'),
        '--model',
        'nexus',
        '--output-format',
        'stream-json',
        '--agent',
        'coder',
    ]
    prompt = run_root / 'prompt.txt'
    assert prompt.read_bytes() == task.encode('utf-8')
    assert prompt.stat().st_mode & 0o777 == 0o600
    assert not list(workspace.iterdir())
    assert run.await_args_list[1].kwargs['transcript'] == run_root / 'transcript.jsonl'
    proxy.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_process_timeout_and_cancellation_stop_descendants(tmp_path, cancel):
    child_pid = tmp_path / 'child.pid'
    code = (
        'import subprocess,sys,time,pathlib; '
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); '
        f'pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); '
        'time.sleep(60)'
    )
    task = asyncio.create_task(
        run_process(
            [sys.executable, '-c', code],
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=30 if cancel else 0.5,
            transcript=tmp_path / 'transcript.jsonl',
        )
    )
    try:
        for _ in range(200):
            if child_pid.exists():
                break
            await asyncio.sleep(0.01)
        assert child_pid.exists(), 'The child process must start before testing cleanup'
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.timed_out
        pid = int(child_pid.read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if child_pid.exists():
            pid = int(child_pid.read_text())
            if psutil.pid_exists(pid):
                psutil.Process(pid).kill()
