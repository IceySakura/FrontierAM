from __future__ import annotations

import asyncio
import json
import logging
import shutil

from pathlib import Path, PurePosixPath
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig, ArtifactConfig
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.trial.trial import Trial

from nexus.core import AgentRunOutput, ExecutionContext
from nexus.extensions.harbor_trial.agent import NexusCodexAgent
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding


class Output(AgentRunOutput):
    final_answer: str = 'done'


def _write_logs(root: Path, session_id: str = 'test-session') -> None:
    sessions = root / 'sessions/2026/09/24'
    sessions.mkdir(parents=True, exist_ok=True)
    (root / 'codex.txt').write_text('{"type":"turn.started"}\nwarning\n', encoding='utf-8')
    events = [
        {'type': 'session_meta', 'payload': {'id': session_id, 'cli_version': '0.114.0', 'cwd': '/workspace'}},
        {'type': 'turn_context', 'payload': {'model': 'test-model'}},
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': 'inspect file'}],
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'function_call',
                'call_id': 'call-1',
                'name': 'exec_command',
                'arguments': '{"cmd":"cat result.txt"}',
            },
        },
        {
            'type': 'response_item',
            'payload': {'type': 'function_call_output', 'call_id': 'call-1', 'output': 'file evidence'},
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'done'}],
            },
        },
        {
            'type': 'event_msg',
            'payload': {
                'type': 'token_count',
                'info': {
                    'total_token_usage': {'input_tokens': 120, 'output_tokens': 24, 'cached_input_tokens': 20},
                    'cost_usd': 0.0,
                },
            },
        },
    ]
    for event in events:
        event['timestamp'] = '2026-09-24T00:00:00Z'
    (sessions / f'rollout-{session_id}.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')


class LocalEnvironment:
    """File-only environment double; Harbor conversion and handoff stay real."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.default_user = None
        self.task_env_config = SimpleNamespace(workdir='/workspace')
        self.capabilities = SimpleNamespace(mounted=False)

    def path(self, path: str) -> Path:
        return self.root / path.lstrip('/')

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(self.path(source_dir), target_dir, dirs_exist_ok=True)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, self.path(target_dir), dirs_exist_ok=True)

    async def upload_file(self, source_path, target_path):
        self.path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, self.path(target_path))

    async def ensure_dirs(self, paths, **_kwargs):
        for path in paths:
            self.path(path).mkdir(parents=True, exist_ok=True)

    async def empty_dirs(self, paths, **kwargs):
        await self.ensure_dirs(paths, **kwargs)

    async def service_is_dir(self, source, **_kwargs):
        return self.path(source).is_dir()

    async def service_download_file(self, source_path, target_path, **_kwargs):
        shutil.copyfile(self.path(source_path), target_path)

    async def service_download_dir(self, source_dir, target_dir, **_kwargs):
        await self.download_dir(source_dir, target_dir)


def _trial(agent: NexusCodexAgent, environment: LocalEnvironment):
    trial = SimpleNamespace(
        agent=agent,
        user_agent=None,
        agent_environment=environment,
        agent_env_paths=SimpleNamespace(agent_dir=agent.environment_logs_dir),
        paths=SimpleNamespace(agent_dir=agent.logs_dir),
        config=SimpleNamespace(agent=AgentConfig()),
        logger=logging.getLogger(__name__),
        _are_agent_logs_downloaded=False,
    )
    for name in (
        '_download_agent_logs',
        '_download_role_logs',
        '_populate_agent_context',
        '_sync_agent_output',
        '_upload_agent_logs',
        '_upload_role_logs',
    ):
        setattr(trial, name, MethodType(getattr(Trial, name), trial))
    return trial


@pytest.mark.parametrize('outcome', ['success', 'error', 'timeout'])
@pytest.mark.parametrize('mounted', [False, True])
@pytest.mark.parametrize('remote_dir', ['/logs/agent', '/logs/custom-agent'])
async def test_codex_native_lifecycle_and_artifact_handoff(
    tmp_path: Path, outcome: str, mounted: bool, remote_dir: str
) -> None:
    source = LocalEnvironment(tmp_path / 'agent-env')
    source.capabilities.mounted = mounted
    host_logs = source.path(remote_dir) if mounted else tmp_path / 'host-logs'
    source.prepare_logs_for_host = AsyncMock()
    source.download_dir = AsyncMock(wraps=source.download_dir)
    source.upload_dir = AsyncMock(wraps=source.upload_dir)
    calls = []
    cancelled = False

    async def run_agent(**kwargs):
        nonlocal cancelled
        calls.append(kwargs)
        _write_logs(source.path(kwargs['logs_dir']))
        if outcome == 'error':
            raise RuntimeError('agent failed')
        if outcome == 'timeout':
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
        return Output()

    run_agent.nexus_meta = {'name': 'codex', 'meta': {'supports_execution_target': True}}
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    context = AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusCodexAgent(logs_dir=host_logs, environment_logs_dir=PurePosixPath(remote_dir))
        if outcome == 'error':
            with pytest.raises(RuntimeError, match='agent failed'):
                await agent.run('inspect file', source, context)
        elif outcome == 'timeout':
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(agent.run('inspect file', source, context), timeout=0.05)
            assert cancelled
        else:
            await agent.run('inspect file', source, context)

    assert calls[0]['logs_dir'] == remote_dir
    assert calls[0]['codex_home_name'] is None
    assert 'retain_json_events_stdout' not in calls[0]
    assert context.is_empty()
    assert not (host_logs / 'trajectory.json').exists()
    source.download_dir.assert_not_awaited()
    trial = _trial(agent, source)
    target = SimpleNamespace(agent_result=context)
    await trial._sync_agent_output(target)
    if mounted:
        source.prepare_logs_for_host.assert_awaited_once()
        source.download_dir.assert_not_awaited()
    else:
        source.prepare_logs_for_host.assert_not_awaited()
        source.download_dir.assert_awaited_once()
    if outcome == 'success':
        assert context.metadata == {'nexus_agent_output': {'final_answer': 'done'}}
    else:
        assert context.metadata is None

    payload = json.loads((host_logs / 'trajectory.json').read_text())
    assert payload['schema_version'] == 'ATIF-v1.7'
    assert payload['session_id'] == 'test-session'
    assert payload['agent']['name'] == 'codex'
    assert 'file evidence' in json.dumps(payload)
    assert 'exec_command' in json.dumps(payload)
    reference_logs = tmp_path / 'native-harbor'
    _write_logs(reference_logs)
    native_context = AgentContext()
    Codex(logs_dir=reference_logs).populate_context_post_run(native_context)
    assert payload == json.loads((reference_logs / 'trajectory.json').read_text())
    assert context.n_input_tokens == native_context.n_input_tokens == 120
    assert context.n_output_tokens == native_context.n_output_tokens == 24
    assert context.n_cache_tokens == native_context.n_cache_tokens == 20
    assert context.cost_usd == native_context.cost_usd == 0.0

    before = context.model_dump()
    await trial._sync_agent_output(target)
    assert context.model_dump() == before
    assert json.loads((host_logs / 'trajectory.json').read_text()) == payload
    await trial._upload_agent_logs()
    if mounted:
        source.upload_dir.assert_not_awaited()
    else:
        source.upload_dir.assert_awaited_once()
    verifier = LocalEnvironment(tmp_path / 'verifier-env')
    handler = ArtifactHandler(
        artifacts=[
            ArtifactConfig(source=f'{remote_dir}/{name}', service='main') for name in ('codex.txt', 'trajectory.json')
        ],
        logger=logging.getLogger(__name__),
    )
    artifacts_dir = tmp_path / 'artifacts'
    manifest = await handler.download_artifacts(
        source,
        artifacts_dir,
        source_artifacts_dir=PurePosixPath('/logs/artifacts'),
        services={'main'},
    )
    entries = [entry for entry in manifest.entries if entry.source.startswith(remote_dir)]
    assert len(entries) == 2
    assert all(entry.status == 'ok' for entry in entries)
    await handler.upload_artifacts(
        verifier,
        artifacts_dir,
        source_artifacts_dir=PurePosixPath('/logs/artifacts'),
        target_artifacts_dir=PurePosixPath('/logs/artifacts'),
    )
    assert verifier.path(f'{remote_dir}/codex.txt').read_bytes() == (host_logs / 'codex.txt').read_bytes()
    assert json.loads(verifier.path(f'{remote_dir}/trajectory.json').read_text()) == payload


async def test_codex_does_not_fabricate_trajectory_without_session(tmp_path: Path) -> None:
    source = LocalEnvironment(tmp_path / 'env')

    async def run_agent(**kwargs):
        log_dir = source.path(kwargs['logs_dir'])
        log_dir.mkdir(parents=True)
        (log_dir / 'codex.txt').write_text('startup failed\n')
        return Output()

    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    context = AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusCodexAgent(logs_dir=tmp_path / 'logs')
        await agent.run('test', source, context)
    trial = _trial(agent, source)
    await trial._sync_agent_output(SimpleNamespace(agent_result=context))
    assert (agent.logs_dir / 'codex.txt').read_text() == 'startup failed\n'
    assert not (agent.logs_dir / 'trajectory.json').exists()
    assert context.metadata == {'nexus_agent_output': {'final_answer': 'done'}}


async def test_codex_log_download_failure_preserves_agent_failure(tmp_path: Path, caplog) -> None:
    async def run_agent(**_kwargs):
        raise RuntimeError('original agent failure')

    source = LocalEnvironment(tmp_path / 'env')
    source.download_dir = AsyncMock(side_effect=OSError('download failed'))
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    context = AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusCodexAgent(logs_dir=tmp_path / 'logs')
        trial = _trial(agent, source)

        async def run_and_sync():
            try:
                await agent.run('test', source, context)
            finally:
                await trial._sync_agent_output(SimpleNamespace(agent_result=context))

        with pytest.raises(RuntimeError, match='original agent failure'):
            await run_and_sync()
    assert 'Failed to download logs' in caplog.text
    assert context.is_empty()


@pytest.mark.parametrize('second_fails', [False, True])
async def test_codex_steps_isolate_outputs_and_sessions(tmp_path: Path, second_fails: bool) -> None:
    source = LocalEnvironment(tmp_path / 'env')
    index = 0

    async def run_agent(**kwargs):
        nonlocal index
        index += 1
        _write_logs(source.path(kwargs['logs_dir']), f'step-{index}')
        if second_fails and index == 2:
            raise RuntimeError('second step failed')
        return Output(final_answer=f'answer-{index}')

    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    first_context, second_context = AgentContext(), AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusCodexAgent(logs_dir=tmp_path / 'logs')
        trial = _trial(agent, source)
        await agent.run('first', source, first_context)
        await trial._sync_agent_output(SimpleNamespace(agent_result=first_context))
        first_snapshot = first_context.model_dump()
        assert json.loads((agent.logs_dir / 'trajectory.json').read_text())['session_id'] == 'step-1'
        # Same reset as Harbor's non-resume step lifecycle.
        shutil.rmtree(source.path('/logs/agent'))
        shutil.rmtree(agent.logs_dir)
        trial._are_agent_logs_downloaded = False
        if second_fails:
            with pytest.raises(RuntimeError, match='second step failed'):
                await agent.run('second', source, second_context)
        else:
            await agent.run('second', source, second_context)
        assert second_context.is_empty()
        await trial._sync_agent_output(SimpleNamespace(agent_result=second_context))
        assert json.loads((agent.logs_dir / 'trajectory.json').read_text())['session_id'] == 'step-2'
        assert second_context.metadata == (
            None if second_fails else {'nexus_agent_output': {'final_answer': 'answer-2'}}
        )
        agent.populate_context_post_run(first_context)
        assert first_context.model_dump() == first_snapshot
