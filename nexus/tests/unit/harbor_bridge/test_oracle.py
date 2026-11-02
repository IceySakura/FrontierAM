from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from nexus.extensions.harbor_bridge.task import _run_oracle


class _FakeOracleEnvironment:
    def __init__(self) -> None:
        self.os = 'linux'
        self.capabilities = SimpleNamespace(mounted=True)
        self.default_user: str | int | None = None
        self._env_scopes: list[dict[str, str]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.exec_calls: list[dict[str, Any]] = []

    @contextmanager
    def with_default_user(self, user: str | int | None) -> Iterator[None]:
        previous = self.default_user
        self.default_user = user
        try:
            yield
        finally:
            self.default_user = previous

    @contextmanager
    def scoped_exec_env(self, env: dict[str, str]) -> Iterator[None]:
        self._env_scopes.append(dict(env))
        try:
            yield
        finally:
            self._env_scopes.pop()

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.uploads.append((Path(source_dir), target_dir))

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        **_: Any,
    ) -> Any:
        scoped_env = {
            key: value
            for scope in self._env_scopes
            for key, value in scope.items()
        }
        self.exec_calls.append({
            'command': command,
            'env': dict(env or {}),
            'scoped_env': scoped_env,
            'timeout_sec': timeout_sec,
            'user': user if user is not None else self.default_user,
        })
        return SimpleNamespace(return_code=0, stdout='', stderr='')


def _write_oracle_task(tmp_path: Path, *, agent_user: str | int = 'appuser') -> Task:
    task_dir = tmp_path / 'oracle-task'
    (task_dir / 'environment').mkdir(parents=True)
    (task_dir / 'tests').mkdir()
    (task_dir / 'solution').mkdir()
    user_literal = f'"{agent_user}"' if isinstance(agent_user, str) else str(agent_user)
    (task_dir / 'task.toml').write_text(
        f"""schema_version = "1.2"

[environment]
docker_image = "agent:v1"

[agent]
user = {user_literal}

[solution.env]
ORACLE_REQUIRED = "${{ORACLE_REQUIRED}}"
"""
    )
    (task_dir / 'instruction.md').write_text('Apply the reference solution.\n')
    (task_dir / 'tests' / 'test.sh').write_text('#!/bin/sh\n')
    (task_dir / 'solution' / 'solve.sh').write_text('#!/bin/sh\nexit 0\n')
    return Task(task_dir)


@pytest.mark.asyncio
async def test_native_oracle_uses_solution_env_user_logs_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv('ORACLE_REQUIRED', 'resolved-secret')
    harbor_task = _write_oracle_task(tmp_path)
    trial_paths = TrialPaths(trial_dir=tmp_path / 'trial')
    trial_paths.mkdir()
    environment = _FakeOracleEnvironment()

    await _run_oracle(
        environment,
        harbor_task,
        trial_paths,
        instruction=harbor_task.instruction,
        timeout_sec=17,
    )

    assert environment.uploads == [(harbor_task.paths.solution_dir, '/solution')]
    assert len(environment.exec_calls) == 2
    chmod_call, solve_call = environment.exec_calls
    assert chmod_call['user'] == 'root'
    assert '/solution/solve.sh' in chmod_call['command']
    assert solve_call['user'] == 'appuser'
    assert solve_call['timeout_sec'] == 17
    assert solve_call['env'] == {'DEBIAN_FRONTEND': 'noninteractive'}
    assert solve_call['scoped_env']['ORACLE_REQUIRED'] == 'resolved-secret'
    assert '/solution/solve.sh' in solve_call['command']
    assert '/logs/agent/oracle.txt' in solve_call['command']


@pytest.mark.asyncio
async def test_native_oracle_rejects_unsupported_numeric_agent_uid(monkeypatch, tmp_path):
    monkeypatch.setenv('ORACLE_REQUIRED', 'resolved-secret')
    harbor_task = _write_oracle_task(tmp_path, agent_user=1000)
    trial_paths = TrialPaths(trial_dir=tmp_path / 'trial')
    trial_paths.mkdir()

    with pytest.raises(ValueError, match='numeric agent UID is not supported'):
        await _run_oracle(
            _FakeOracleEnvironment(),
            harbor_task,
            trial_paths,
            instruction=harbor_task.instruction,
            timeout_sec=17,
        )
