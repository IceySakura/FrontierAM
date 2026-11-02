"""Exercise real SWE task error handling with in-process runtime and agent stubs."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import uuid

from types import SimpleNamespace

import pytest

from nexus.core import Registry, agent_registry
from nexus.core.context import ExecutionContext, _current_context
from nexus.extensions.agents.swe_agent.agent import SWEAgentInput, SWEAgentOutput
from nexus.extensions.agents.swe_agent.platform_status import PlatformExitStatus
from nexus.extensions.tasks.swe import task as swe_task
from nexus.extensions.tasks.swe.platform import STATE_KEY, SuperSWEPlatformHook
from nexus.runner import single
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.batch_config import BatchConfig
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.single import RunOutput
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.executors.command import run_command
from nexus.runtime.executors.models.command import RunCommandRequest
from nexus.runtime.providers.base import RuntimeProvider


PATCH = 'diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n'
OOM = 'Sandbox main DIED: Process.OOMKilled, exit code 137'


def _scenario(
    monkeypatch,
    *,
    fail_at=None,
    error=None,
    supplied_patch=False,
    resolved=False,
    agent_output=None,
    checkout_response=None,
    cleanup_executor=None,
    **task_options,
):
    events = []
    commands = []
    executor_calls = []
    failure = error if error is not None else RuntimeError(OOM)

    def visit(stage):
        events.append(stage)
        if stage == fail_at:
            raise failure

    class Runtime:
        async def call_executor(self, _executor, _params):
            commands.append(_params.get('command', ''))
            executor_calls.append((_executor, dict(_params)))
            if cleanup_executor is not None:
                result = await cleanup_executor(_params)
                if result is not None:
                    return result
            if 'stdin' in _params:
                digest = hashlib.sha256(_params['stdin'].encode()).hexdigest()
                return {'stdout': digest + '  /tmp/.b4_needles\n', 'status': 'Finished', 'return_code': 0}
            if _params.get('command', '').startswith('git checkout ') and checkout_response is not None:
                return checkout_response
            return {'stdout': '', 'stderr': '', 'status': 'Finished', 'return_code': 0}

    class Provider(RuntimeProvider):
        def configure(self, name, **params):
            visit(f'{name}_configure')
            super().configure(name, **params)

        async def acquire(self, name, **_params):
            visit(f'{name}_acquire')
            return Runtime()

        async def _release_impl(self, name):
            visit(f'{name}_release')

    class Handler:
        def get_repo_language(self, _instance):
            return 'py'

        def get_problem_statement(self, _instance):
            return 'Update the example.'

        async def infer_runtime_init(self, _instance, _runtime, **_kwargs):
            visit('infer_init')

        async def eval_runtime_init(self, _instance, _runtime):
            visit('eval_init')

    async def agent(**_kwargs):
        visit('agent')
        return agent_output if agent_output is not None else SimpleNamespace(exit_status='submitted')

    async def get_patch(*_args, **_kwargs):
        visit('patch')
        return PATCH

    async def verify_truncation(*_args, **_kwargs):
        return True

    async def grade(*_args, **_kwargs):
        visit('grade')
        return {'score': float(resolved), 'completed': True, 'resolved': resolved, 'report_json': '{}'}

    monkeypatch.setattr(swe_task, 'get_handler', lambda _dataset_type: Handler())
    monkeypatch.setattr(swe_task, '_build_leak_needles', lambda _instance: [])
    monkeypatch.setattr(swe_task, 'get_patch_from_runtime', get_patch)
    monkeypatch.setattr(swe_task, 'verify_git_truncation', verify_truncation)
    monkeypatch.setattr(swe_task, 'evaluate_and_grade_in_runtime', grade)
    ctx = ExecutionContext(runtime_provider=Provider(), task_name='swe_task')
    ctx.task_config['enable_anti_hack_prompt'] = False
    task_input = swe_task.SwebenchTaskInput(
        instance_id='example__example-1',
        instance={'instance_id': 'example__example-1', 'repo': 'example/example', 'base_commit': 'a' * 40},
        run_agent=agent,
        patch_text=PATCH if supplied_patch else None,
        **task_options,
    )

    async def execute(task_fn=swe_task.run_swe_task):
        token = _current_context.set(ctx)
        try:
            return await task_fn(task_input)
        finally:
            _current_context.reset(token)

    return SimpleNamespace(
        run=lambda: asyncio.run(execute()),
        execute=execute,
        task_input=task_input,
        ctx=ctx,
        events=events,
        failure=failure,
        commands=commands,
        executor_calls=executor_calls,
    )


def _sandbox_cleanup_executor(tmp_path, *, override=None, extra_env=None):
    sandbox = tmp_path / 'sandbox'
    for name in ('tmp', 'root', 'home'):
        (sandbox / name).mkdir(parents=True)

    async def execute(params):
        command = params.get('command', '')
        if '/tmp/.b4_needles' not in command:
            return None
        if override is not None:
            params = override(dict(params))
        command = re.sub(r'/(tmp|root|home)(?=/|[\s\'";|])', lambda m: str(sandbox / m[1]), params['command'])
        result = await run_command(
            RunCommandRequest(**{
                **params,
                'command': command,
                'cwd': str(sandbox),
                'extra_env': {'BASH_ENV': '/dev/null', **(extra_env or {})},
            })
        )
        return result.model_dump(mode='json')

    return sandbox, execute


@pytest.mark.parametrize('large', [False, True])
def test_fingerprints_use_stdin_and_preserve_full_content(monkeypatch, tmp_path, large):
    build_needles = swe_task._build_leak_needles
    added = 'value = "' + ('汉字$`' * 40000 if large else 'a complete reference value') + '"'
    patch = PATCH.replace('+new', '+' + added)
    expected = '\n'.join(build_needles({'patch': patch}))
    observed = []

    def observe(params):
        if 'PRUNE=' in params['command']:
            observed.append((sandbox / 'tmp/.b4_needles').read_bytes())
        return params

    sandbox, execute = _sandbox_cleanup_executor(tmp_path, override=observe)
    (sandbox / 'tmp/reference_copy').write_text(patch)
    scenario = _scenario(monkeypatch, cleanup_executor=execute, resolved=True)
    scenario.task_input.instance['patch'] = patch
    monkeypatch.setattr(swe_task, '_build_leak_needles', build_needles)

    output = scenario.run()

    assert output.completed is True, output.error
    assert scenario.events.count('agent') == 1
    assert observed == [expected.encode()]
    transfer = next(params for _, params in scenario.executor_calls if 'stdin' in params)
    assert transfer['stdin'] == expected
    assert expected not in transfer['command']
    assert len(transfer['command'].encode()) < 1000
    if large:
        assert len(expected.encode()) > os.sysconf('SC_PAGE_SIZE') * 32
    assert not (sandbox / 'tmp/reference_copy').exists()
    assert not (sandbox / 'tmp/.b4_needles').exists()


def test_empty_fingerprints_replace_stale_material(monkeypatch, tmp_path):
    sandbox, execute = _sandbox_cleanup_executor(tmp_path)
    (sandbox / 'tmp/.b4_needles').write_text('+new')
    (sandbox / 'tmp/ordinary_file').write_text(PATCH)
    scenario = _scenario(monkeypatch, cleanup_executor=execute, resolved=True)

    output = scenario.run()

    assert output.completed is True, output.error
    assert (sandbox / 'tmp/ordinary_file').read_text() == PATCH
    assert not (sandbox / 'tmp/.b4_needles').exists()


@pytest.mark.parametrize('operation', ['transfer', 'cleanup'])
@pytest.mark.parametrize(
    ('response', 'reason'),
    [
        ({'status': 'Error', 'stderr': 'executor unavailable'}, 'INITIALIZATION_ERROR'),
        ({'status': 'Finished', 'return_code': 1, 'stderr': 'permission denied'}, 'INITIALIZATION_ERROR'),
        ({'status': 'TimeLimitExceeded'}, 'TASK_TIMEOUT'),
        ({'stdout': ''}, 'INITIALIZATION_ERROR'),
    ],
)
def test_cleanup_failure_stops_before_agent(monkeypatch, operation, response, reason):
    async def execute(params):
        if ('stdin' in params and operation == 'transfer') or (
            'PRUNE=' in params['command'] and operation == 'cleanup'
        ):
            return response
        return None

    scenario = _scenario(monkeypatch, cleanup_executor=execute)
    output = scenario.run()

    assert output.termination_reason == reason
    assert output.completed is False
    assert output.extra_fields['error_stage'] == 'task.init'
    assert 'agent' not in scenario.events
    assert 'grade' not in scenario.events
    assert 'main_release' in scenario.events
    assert 'eval_release' in scenario.events
    assert scenario.ctx._runtimes == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['transfer', 'cleanup'])
@pytest.mark.parametrize('sandbox_died', [False, True])
async def test_platform_cleanup_failure_is_not_an_agent_verdict(monkeypatch, operation, sandbox_died):
    async def execute(params):
        if ('stdin' in params and operation == 'transfer') or (
            'PRUNE=' in params['command'] and operation == 'cleanup'
        ):
            if sandbox_died:
                raise RuntimeUnhealthyError('setup OOM', code=260422, reason='Process.OOMKilled')
            return {'status': 'Finished', 'return_code': 1, 'stderr': 'permission denied'}
        return None

    scenario = _scenario(monkeypatch, cleanup_executor=execute)
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(scenario.ctx)
    native = await scenario.execute()
    output = RunOutput(extra={'superswe': {'mode': 'model'}}, task_output=native)
    await hook.on_run_output(scenario.ctx, output)

    assert output.agent_output['exit_status'] == ('runtime_error_by_env' if sandbox_died else 'environment_error')
    assert output.agent_output['exit_code'] == -1
    assert output.agent_output['agent_started'] is False
    assert output.success is False
    assert output.judge_output is None
    assert 'agent' not in scenario.events
    assert not scenario.ctx.tool_state[STATE_KEY].get('terminal_agent_failure')


def test_corrupted_fingerprints_stop_before_agent(monkeypatch, tmp_path):
    def corrupt(params):
        if 'stdin' in params:
            params['stdin'] += 'unexpected bytes'
        return params

    _, execute = _sandbox_cleanup_executor(tmp_path, override=corrupt)
    scenario = _scenario(monkeypatch, cleanup_executor=execute)

    output = scenario.run()

    assert output.termination_reason == 'INITIALIZATION_ERROR'
    assert 'checksum mismatch' in output.error
    assert 'agent' not in scenario.events
    assert not any('PRUNE=' in command for command in scenario.commands)
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('tool', ['rm', 'find', 'grep', 'awk'])
def test_cleanup_shell_failure_is_not_hidden(monkeypatch, tmp_path, tool):
    tools_dir = tmp_path / 'bin'
    tools_dir.mkdir()
    failed_tool = tools_dir / tool
    failed_tool.write_text('#!/bin/sh\necho "injected cleanup failure" >&2\nexit 23\n')
    failed_tool.chmod(0o755)
    sandbox, execute = _sandbox_cleanup_executor(
        tmp_path,
        extra_env={'PATH': f'{tools_dir}:{os.environ.get("PATH", "")}'},
    )
    patch = PATCH.replace('+new', '+from functools import partial')
    (sandbox / 'tmp/reference_copy').write_text(patch)
    build_needles = swe_task._build_leak_needles
    scenario = _scenario(monkeypatch, cleanup_executor=execute)
    scenario.task_input.instance['patch'] = patch
    monkeypatch.setattr(swe_task, '_build_leak_needles', build_needles)

    output = scenario.run()

    assert output.termination_reason == 'INITIALIZATION_ERROR'
    assert 'artifact removal' in output.error
    assert 'injected cleanup failure' in output.error
    assert 'agent' not in scenario.events
    assert 'grade' not in scenario.events


def test_leak_cleanup_confirms_patch_copies_and_preserves_normal_dependencies(monkeypatch, tmp_path):
    build_needles = swe_task._build_leak_needles
    patch = (
        'diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n'
        '@@ -1 +1,3 @@\n-old\n+from functools import partial\n'
        '+@pytest.mark.parametrize("example", [1, 2])\n+def example(value): return value + 1\n'
    )
    scenario = _scenario(monkeypatch, resolved=True, artifact_cleanup_globs=['recon'])
    scenario.task_input.instance['patch'] = patch
    monkeypatch.setattr(swe_task, '_build_leak_needles', build_needles)
    scenario.run()
    fingerprints = '\n'.join(build_needles(scenario.task_input.instance))
    assert 'from functools import partial' in fingerprints.splitlines()

    # Execute the actual emitted cleanup script against isolated fixture roots.
    # Rewrite every sandbox root in one pass, including the protected paths and
    # fingerprint location; never scan the host's /tmp, /root or /home.
    sandbox = tmp_path / 'sandbox'
    for name in ('tmp', 'root', 'home'):
        (sandbox / name).mkdir(parents=True)
    original_command = next(command for command in scenario.commands if 'PRUNE=' in command)
    command = re.sub(r'/(tmp|root|home)(?=/|[\s\'";|])', lambda m: str(sandbox / m[1]), original_command)
    assert f'for ROOT in {sandbox / "tmp"} {sandbox / "root"} {sandbox / "home"}; do' in command
    (sandbox / 'tmp/.b4_needles').write_text(fingerprints)
    preserved = {
        'root/install.sh': 'from functools import partial\n',
        'tmp/helper.mjs': 'export default function run() { return 1; }\n',
        'home/project_tests/needed.py': 'from functools import partial\n',
        'root/.local/lib/python/site-packages/pkg/__init__.py': 'from functools import partial\n',
        'root/venv/lib/python/site-packages/_pytest/mark/__init__.py': '@pytest.mark.parametrize("example", [1, 2])\ndef normal_test(example): pass\n',
        'tmp/normal_dependency.py': 'from functools import partial\n',
        'root/stdlib/module.py': '"""A documentation example:\n' + patch + '"""\n',
        'tmp/nexus/runtime/helper': patch,
        'tmp/project_build/helper': patch,
        'root/compiled.so': patch,
    }
    removed = {
        'tmp/neutral_payload': patch,
        'root/another_payload': patch,
        'tmp/nexus_leak': patch,
        'tmp/recon/example.py': 'from functools import partial\ndef example(value): return value + 1\n',
        'root/reference.patch': patch,
    }
    for name, content in {**preserved, **removed}.items():
        path = sandbox / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(['/bin/bash', '-c', command], check=True, capture_output=True, text=True, timeout=10)
    assert {name: (sandbox / name).read_text() for name in preserved if (sandbox / name).exists()} == preserved
    assert all(not (sandbox / name).exists() for name in removed)
    assert not (sandbox / 'tmp/.b4_needles').exists()


@pytest.mark.parametrize('enabled', [None, False, True])
@pytest.mark.parametrize('backend', ['terminal', 'stateless'])
@pytest.mark.parametrize('reward_override', [None, -1.0])
def test_prompt_uses_validated_option_and_effective_backend(monkeypatch, enabled, backend, reward_override):
    options = {} if enabled is None else {'enable_anti_hack_prompt': enabled}
    scenario = _scenario(monkeypatch, enable_anti_hack=True, hack_reward_override=reward_override, **options)
    scenario.ctx.agent_config['bash_backend'] = backend
    scenario.ctx.task_config['enable_anti_hack_prompt'] = not bool(enabled)
    monkeypatch.setenv('SWE_AGENT_BASH_BACKEND', 'stateless' if backend == 'terminal' else 'terminal')
    scenario.run()
    prompt = scenario.ctx.task_config['problem_statement']
    assert prompt.count('<IMPORTANT>') == int(bool(enabled))
    assert ('Mining the harness' in prompt) is (bool(enabled) and backend == 'terminal')
    assert 'are penalized' not in prompt
    assert ('are invalid and are discarded' in prompt) is bool(enabled)


@pytest.mark.parametrize('extra_exclusions', [[], ['hints_text', 'private_note']])
def test_agent_input_keeps_permitted_hints_without_exposing_answers(monkeypatch, extra_exclusions):
    monkeypatch.delenv('SWE_TASK_CONFIG_KEEP_GOLD', raising=False)
    scenario = _scenario(monkeypatch, agent_instance_exclude_fields=extra_exclusions)
    original = scenario.task_input.instance
    original.update({
        'patch': PATCH,
        'test_patch': PATCH,
        'fix_patch': PATCH,
        'bug_patch': PATCH,
        'FAIL_TO_PASS': ['private_test'],
        'PASS_TO_PASS': ['regression_test'],
        'spec_dict': {'private': 'grading specification'},
        'hints_text': 'Public task hint.',
        'private_note': 'Optional local field.',
    })
    scenario.run()
    visible = scenario.ctx.task_config['instance']
    assert not set(visible) & swe_task.GOLD_FIELDS
    assert ('hints_text' in visible) is ('hints_text' not in extra_exclusions)
    assert ('private_note' in visible) is ('private_note' not in extra_exclusions)
    assert original['patch'] == PATCH
    assert original['hints_text'] == 'Public task hint.'


@pytest.mark.parametrize(
    ('fail_at', 'expected_reason', 'expected_stage', 'patch_preserved'),
    [
        ('main_acquire', 'INITIALIZATION_ERROR', 'task.init', False),
        ('infer_init', 'INITIALIZATION_ERROR', 'task.init', False),
        ('agent', 'ENV_ERROR', 'task.agent', False),
        ('patch', 'ENV_ERROR', 'task.agent.get_patch', False),
        ('main_release', 'ENV_ERROR', 'task.agent.release_runtime', True),
        ('eval_configure', 'ENV_ERROR', 'task.scoring', True),
        ('eval_acquire', 'ENV_ERROR', 'task.scoring.get_runtime', True),
        ('eval_init', 'ENV_ERROR', 'task.scoring.runtime_init', True),
        ('grade', 'ENV_ERROR', 'task.scoring.grade', True),
        ('eval_release', 'ENV_ERROR', 'task.scoring.release_runtime', True),
    ],
)
def test_runtime_failure_reports_actual_stage(monkeypatch, fail_at, expected_reason, expected_stage, patch_preserved):
    scenario = _scenario(monkeypatch, fail_at=fail_at)
    output = scenario.run()

    assert fail_at in scenario.events
    assert output.termination_reason == expected_reason
    assert output.extra_fields['error_stage'] == expected_stage
    assert output.score == 0.0
    assert output.completed is False
    assert output.resolved is False
    assert output.patch_text == (PATCH if patch_preserved else '')
    assert OOM in output.error
    assert scenario.events.count('agent') <= 1
    assert scenario.events.count('grade') <= 1
    assert 'main_release' in scenario.events
    assert 'eval_release' in scenario.events
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize(
    ('response', 'termination'),
    [
        ({'status': 'Finished', 'return_code': 128, 'stderr': 'unknown revision'}, 'ENV_ERROR'),
        ({'status': 'Error', 'return_code': None, 'stderr': 'executor unavailable'}, 'ENV_ERROR'),
        ({'status': 'TimeLimitExceeded', 'return_code': None}, 'TASK_TIMEOUT'),
    ],
)
def test_failed_checkout_keeps_patch_and_never_calls_grader(monkeypatch, response, termination):
    scenario = _scenario(monkeypatch, checkout_response=response)

    output = scenario.run()

    assert 'grade' not in scenario.events
    assert output.completed is False
    assert output.patch_text == PATCH
    assert output.termination_reason == termination
    assert output.extra_fields['error_stage'] == 'task.scoring.checkout'
    assert 'checkout' in output.error.lower()
    assert scenario.events.count('agent') == 1
    assert 'main_release' in scenario.events
    assert 'eval_release' in scenario.events
    assert scenario.ctx._runtimes == {}


def test_eval_failure_preserves_supplied_patch_without_running_agent(monkeypatch):
    scenario = _scenario(monkeypatch, fail_at='eval_init', supplied_patch=True)
    output = scenario.run()

    assert output.termination_reason == 'ENV_ERROR'
    assert output.extra_fields['error_stage'] == 'task.scoring.runtime_init'
    assert output.patch_text == PATCH
    assert output.completed is False
    assert 'agent' not in scenario.events
    assert 'main_acquire' not in scenario.events


@pytest.mark.parametrize('fail_at', ['agent', 'grade'])
def test_timeout_keeps_existing_termination_and_patch(monkeypatch, fail_at):
    scenario = _scenario(monkeypatch, fail_at=fail_at, error=TimeoutError('deadline exceeded'))
    output = scenario.run()

    assert output.termination_reason == 'TASK_TIMEOUT'
    assert output.extra_fields['error_stage'] == ('task.agent' if fail_at == 'agent' else 'task.scoring.grade')
    assert output.completed is False
    assert output.patch_text == (PATCH if fail_at == 'grade' else '')
    assert 'deadline exceeded' in output.error
    assert scenario.ctx._runtimes == {}


def test_raise_on_runtime_error_reraises_original_and_releases_runtimes(monkeypatch):
    scenario = _scenario(monkeypatch, fail_at='agent', raise_on_runtime_error=True)

    with pytest.raises(RuntimeError, match='Process.OOMKilled') as caught:
        scenario.run()

    assert caught.value is scenario.failure
    assert scenario.events.count('agent') == 1
    assert scenario.events[-2:] == ['main_release', 'eval_release']
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('resolved', [False, True])
def test_completed_verdict_is_preserved_without_regrading(monkeypatch, resolved):
    scenario = _scenario(monkeypatch, resolved=resolved, eval_max_attempts=3)
    output = scenario.run()

    assert output.completed is True
    assert output.resolved is resolved
    assert output.score == float(resolved)
    assert output.patch_text == PATCH
    assert output.termination_reason == 'ENV_DONE'
    assert output.extra_fields == {}
    assert scenario.events.count('agent') == 1
    assert scenario.events.count('grade') == 1
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('stage', ['install', 'proxy', 'pre_start', 'execution_transport'])
@pytest.mark.parametrize('stderr', ['setup failed', 'TimeoutError: setup deadline exceeded'])
def test_agent_setup_failure_skips_patch_and_verifier(monkeypatch, stage, stderr):
    agent_output = SimpleNamespace(
        exit_status='unhandled_error', failure_kind='infrastructure', failure_stage=stage, stderr=stderr
    )
    span_updates = []
    monkeypatch.setattr(swe_task, 'get_current_span_id', lambda: 'task-span')
    monkeypatch.setattr(swe_task, 'update_span', lambda _span, *, attributes: span_updates.append(attributes))
    scenario = _scenario(monkeypatch, agent_output=agent_output, resolved=True)

    output = scenario.run()

    assert output.termination_reason == 'INITIALIZATION_ERROR'
    assert output.completed is False
    assert output.resolved is False
    assert output.score == 0.0
    assert output.patch_text == ''
    assert output.extra_fields['error_stage'] == f'task.agent.{stage}'
    assert stderr in output.error
    assert any(update.get('hytuner.termination_reason') == 'INITIALIZATION_ERROR' for update in span_updates)
    assert not {'patch', 'eval_acquire', 'grade'}.intersection(scenario.events)
    assert scenario.events[-2:] == ['main_release', 'eval_release']
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('stage', ['install', 'proxy', 'pre_start', 'execution_transport', 'runtime_unhealthy'])
def test_agent_setup_failure_respects_runtime_error_rethrow(monkeypatch, stage):
    agent_output = SimpleNamespace(
        exit_status='unhandled_error',
        failure_kind='infrastructure',
        failure_stage=stage,
        stderr='TimeoutError: setup deadline exceeded',
    )
    scenario = _scenario(monkeypatch, agent_output=agent_output, raise_on_runtime_error=True)

    with pytest.raises(RuntimeError, match='setup deadline exceeded'):
        scenario.run()

    assert not {'patch', 'eval_acquire', 'grade'}.intersection(scenario.events)
    assert scenario.events[-2:] == ['main_release', 'eval_release']
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('failure_kind', [None, 'agent'])
def test_agent_failure_stage_alone_does_not_skip_scoring(monkeypatch, failure_kind):
    agent_output = SimpleNamespace(exit_status='submitted', failure_kind=failure_kind, failure_stage='install')
    scenario = _scenario(monkeypatch, agent_output=agent_output, resolved=True)

    output = scenario.run()

    assert output.completed is True
    assert output.score == 1.0
    assert scenario.events.count('grade') == 1
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('option', ['remove_git_dir', 'skip_infer_patch_eval'])
def test_agent_setup_failure_precedes_skipping_inference(monkeypatch, option):
    agent_output = SimpleNamespace(
        exit_status='unhandled_error', failure_kind='infrastructure', failure_stage='pre_start', stderr='setup failed'
    )
    scenario = _scenario(monkeypatch, agent_output=agent_output, **{option: True})

    output = scenario.run()

    assert output.termination_reason == 'INITIALIZATION_ERROR'
    assert output.completed is False
    assert not {'patch', 'eval_acquire', 'grade'}.intersection(scenario.events)
    assert scenario.ctx._runtimes == {}


@pytest.mark.parametrize('stage', ['runtime_unhealthy', 'process', 'unknown_stage', None])
@pytest.mark.parametrize(
    'stderr', ['sandbox lost', 'TimeoutError: transport lost', 'UnicodeDecodeError: transport lost']
)
def test_agent_infrastructure_failure_cannot_be_scored_as_a_completed_task(monkeypatch, stage, stderr):
    agent_output = SimpleNamespace(
        success=True,
        exit_status='env_done',
        failure_kind='infrastructure',
        failure_stage=stage,
        stderr=stderr,
    )
    scenario = _scenario(monkeypatch, agent_output=agent_output, resolved=True)

    output = scenario.run()

    assert output.termination_reason == 'ENV_ERROR'
    assert output.completed is False
    assert output.resolved is False
    assert output.score == 0.0
    assert output.patch_text == ''
    assert output.extra_fields['error_stage'] == f'task.agent.{stage or "infrastructure"}'
    assert stderr in output.error
    assert not {'patch', 'eval_acquire', 'grade'}.intersection(scenario.events)
    assert scenario.events[-2:] == ['main_release', 'eval_release']
    assert scenario.ctx._runtimes == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['Process.OOMKilled', 'Process.NonZeroExit', 'Process.CrashLoopBackOff'])
@pytest.mark.parametrize('returned', [False, True])
async def test_agent_caused_death_is_a_terminal_zero_without_fabricating_tests(monkeypatch, reason, returned):
    from nexus.extensions.tasks.swe import superswe

    evidence = {'code': 260422, 'reason': reason, 'exit_code': 137, 'trace_id': 'fixture'}
    failure = RuntimeUnhealthyError('sandbox died while executing tool', **evidence)
    agent_result = (
        SWEAgentOutput(
            exit_status=PlatformExitStatus.RUNTIME_ERROR_BY_AGENT,
            exit_code=0,
            error=str(failure),
            failure_stage='tool',
            failure_kind='agent',
            sandbox=evidence,
            iterations=4,
        )
        if returned
        else None
    )
    scenario = _scenario(
        monkeypatch,
        fail_at=None if returned else 'agent',
        error=failure,
        agent_output=agent_result,
        raise_on_runtime_error=True,
    )
    row = scenario.task_input.instance
    row['dataset_type'] = 'T3Build_SWEBuilder_Midtrain'
    row[superswe.COMPONENTS_KEY] = [
        {
            'name': 'cli',
            'instance': {**row, 'eval_script': 'exit 0'},
            'eval_timeout_seconds': 10,
        }
    ]
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(scenario.ctx)
    native = await scenario.execute(superswe.run_superswe_task)
    output = RunOutput(extra={'superswe': {'mode': 'model'}}, task_output=native, agent_output=agent_result)
    await hook.on_run_output(scenario.ctx, output)

    assert output.success is True
    assert output.model_dump(mode='json')['agent_output']['exit_status'] == 'runtime_error_by_agent'
    assert output.task_output['status'] == 'success'
    assert output.task_output['score'] == 0
    assert output.task_output['resolved'] is False
    assert output.task_output['error'] == output.model_dump(mode='json')['agent_output']['error']
    assert str(failure) in output.task_output['error']
    assert output.task_output['completed'] is False  # The tests were never executed.
    assert output.judge_output == {
        'exit_status': 'finished',
        'exit_code': 0,
        'error': None,
        'scoring_method': 'agent_runtime_failure',
        'tests_executed': False,
    }
    assert output.task_output['extra_fields']['superswe_components']['pending'] == ['cli']
    assert str(failure) in hook.result['native_task_error']
    assert hook.result['terminal_agent_failure']['sandbox'] == evidence
    assert not {'patch', 'eval_acquire', 'grade'}.intersection(scenario.events)
    assert scenario.events.count('agent') == 1
    assert scenario.ctx._runtimes == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('fail_at', 'status'),
    [
        ('main_acquire', 'runtime_error_by_env'),
        ('infer_init', 'runtime_error_by_env'),
        ('agent', 'runtime_error_by_env'),
        ('grade', 'finished'),
    ],
)
@pytest.mark.parametrize(('reason', 'code'), [('Process.Killed', 260500), ('Queue.Timeout', 260422), (None, None)])
async def test_environment_deaths_and_judge_errors_are_never_scored_as_agent_death(
    monkeypatch, fail_at, status, reason, code
):
    failure = RuntimeUnhealthyError('actual sandbox failure', code=code, reason=reason)
    agent_result = SimpleNamespace(exit_status='finished', exit_code=0, error=None) if fail_at == 'grade' else None
    scenario = _scenario(monkeypatch, fail_at=fail_at, error=failure, agent_output=agent_result)
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(scenario.ctx)
    native = await scenario.execute()
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        task_output=native,
        agent_output=vars(agent_result) if agent_result else None,
    )
    await hook.on_run_output(scenario.ctx, output)
    assert output.agent_output['exit_status'] == status
    assert output.task_output['status'] == ('judge_error' if fail_at == 'grade' else 'agent_error')
    assert 'actual sandbox failure' in output.task_output['error']
    assert not scenario.ctx.tool_state[STATE_KEY].get('terminal_agent_failure')
    if fail_at == 'grade':
        assert output.judge_output['exit_code'] == -1
        assert output.task_output['patch_text'] == PATCH


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_at', ['main_acquire', 'infer_init'])
async def test_process_death_during_startup_never_gets_terminal_agent_verdict(monkeypatch, fail_at):
    failure = RuntimeUnhealthyError('startup OOM', code=260422, reason='Process.OOMKilled')
    scenario = _scenario(monkeypatch, fail_at=fail_at, error=failure)
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(scenario.ctx)
    native = await scenario.execute()
    output = RunOutput(extra={'superswe': {'mode': 'model'}}, task_output=native)
    await hook.on_run_output(scenario.ctx, output)
    assert output.agent_output['exit_status'] == 'runtime_error_by_env'
    assert output.agent_output['agent_started'] is False
    assert output.task_output['status'] == 'agent_error'
    assert 'agent' not in scenario.events


@pytest.mark.asyncio
@pytest.mark.parametrize('returned', [False, True])
@pytest.mark.parametrize('by_agent', [False, True])
@pytest.mark.parametrize('raise_errors', [False, True])
async def test_real_runner_and_batch_retry_only_environment_death(
    register_test_component, monkeypatch, tmp_path, returned, by_agent, raise_errors
):
    # Exercise registry dispatch, the real SuperSWE task/hook, runner capture,
    # retry policy and SQLite persistence with only external services stubbed.
    Registry.import_module('nexus.extensions.tasks.swe.superswe')

    evidence = {
        'code': 260422 if by_agent else 260500,
        'reason': 'Process.NonZeroExit' if by_agent else 'Process.Killed',
        'exit_code': 137,
        'trace_id': 'fixture',
    }
    status = PlatformExitStatus.RUNTIME_ERROR_BY_AGENT if by_agent else PlatformExitStatus.RUNTIME_ERROR_BY_ENV
    agent_result = SWEAgentOutput(
        exit_status=status,
        exit_code=0 if by_agent else -1,
        error='actual sandbox error',
        failure_stage='tool',
        failure_kind='agent' if by_agent else 'infrastructure',
        sandbox=evidence,
    )
    scenario = _scenario(
        monkeypatch,
        fail_at=None if returned else 'agent',
        agent_output=agent_result,
        error=RuntimeUnhealthyError('actual sandbox error', **evidence),
        raise_on_runtime_error=raise_errors,
    )
    agent_name = 'test_superswe_status_' + uuid.uuid4().hex

    @register_test_component(agent_registry, agent_name)
    async def fixture_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
        return await scenario.task_input.run_agent()

    monkeypatch.setattr(single, '_create_runtime_provider', lambda config: scenario.ctx.runtime_provider)
    config = single.RunInput(
        task_name='superswe_task',
        agent_name=agent_name,
        agent_config={'platform_exit_status': True},
        task_config={**scenario.task_input.model_dump(exclude={'run_agent'}), 'enable_anti_hack_prompt': False},
        extra={'superswe': {'mode': 'model'}},
        runner={'hooks': {'superswe_platform': {'name': 'superswe_platform'}}},
    )
    generator_path = tmp_path / 'generator.py'
    generator_path.write_text(
        'from nexus.runner.single import RunInput\n'
        'from nexus.runner.batch.batch_config import BatchItem\n'
        'def generate_configs():\n'
        f'    yield BatchItem(config=RunInput.model_validate({config.model_dump(mode="json")!r}), run_id="case")\n'
    )
    db_path = tmp_path / 'batch.db'
    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(retry_failed=2, enable_trace_db=False, enable_log_capture=False),
    )
    assert summary.success == int(by_agent)
    assert summary.failed == int(not by_agent)
    assert scenario.events.count('agent') == (1 if by_agent else 3)
    async with BatchDatabase(db_path) as db:
        record = await db.get_run('case')
    assert record['retry_count'] == (0 if by_agent else 2)
    assert record['agent_output']['exit_status'] == status.value
    assert record['agent_output']['exit_code'] == (0 if by_agent else -1)
    assert record['task_output']['status'] == ('success' if by_agent else 'agent_error')
    assert record['task_output']['score'] == 0.0
    if by_agent:
        assert record['judge_output']['exit_code'] == 0
        assert record['judge_output']['tests_executed'] is False
    assert 'grade' not in scenario.events
