"""OSWorld dataset auth regressions."""

from __future__ import annotations

import json
import os

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.dataset import auth, task_files
from nexus.dataset.auth import ENV_WEDATA_KEY
from nexus.dataset.exceptions import DatasetConfigurationError
from nexus.extensions.tasks.os_world import task as task_module
from nexus.extensions.tasks.os_world.data import prefetch
from nexus.extensions.tasks.os_world.data.dataset_files import WEDATA_KEY_INTERPOLATION
from nexus.extensions.tasks.os_world.data.paths import DEFAULT_CACHE_ROOT, resolve_evaluator_cache_dir
from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, _cache_dir
from nexus.runner.single import _build_input


@pytest.fixture
def isolate_auth(monkeypatch):
    monkeypatch.delenv('WEDATA_KEY', raising=False)
    monkeypatch.delenv('TAI_PAT', raising=False)
    auth.clear_wedata_key_cache()
    yield
    auth.clear_wedata_key_cache()


@pytest.fixture(autouse=True)
def _no_external_artifact_service(monkeypatch):
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts

    monkeypatch.setattr(
        RunArtifacts,
        'publish',
        AsyncMock(return_value={'status': 'published', 'uri': 'artifact://test/run/', 'artifact': []}),
    )


@pytest.mark.usefixtures('isolate_auth')
def test_shell_damaged_key_is_rejected_before_environment_publication(monkeypatch):
    valid = '{"id":"fixture","subject":"fixture","key":"secret-sentinel"}'
    monkeypatch.setenv('WEDATA_KEY', valid)
    with pytest.raises(DatasetConfigurationError, match='quote JSON') as exc:
        auth.apply_wedata_key('key:secret-sentinel')
    assert 'secret-sentinel' not in str(exc.value)
    assert os.environ['WEDATA_KEY'] == valid


@pytest.mark.usefixtures('isolate_auth')
def test_configured_invalid_key_cannot_downgrade_to_anonymous(monkeypatch):
    monkeypatch.setenv('WEDATA_KEY', 'shell-damaged-secret')
    with pytest.raises(DatasetConfigurationError):
        task_files._optional_task_files_auth_payload()


@pytest.mark.usefixtures('isolate_auth')
@pytest.mark.asyncio
async def test_dataset_task_json_failure_does_not_use_stale_cache_or_checkout(monkeypatch, tmp_path):
    (tmp_path / 'task.json').write_text('{"id":"stale"}')
    files = SimpleNamespace(download=AsyncMock(side_effect=RuntimeError('credential-secret signed-url')))
    monkeypatch.setattr(prefetch, 'require_context', lambda: SimpleNamespace(task_files=files))
    with pytest.raises(RuntimeError, match='no task_root fallback') as exc:
        await prefetch.prefetch_task_json(tmp_path)
    assert 'credential-secret' not in str(exc.value)
    assert 'signed-url' not in str(exc.value)


@pytest.mark.usefixtures('isolate_auth')
@pytest.mark.asyncio
async def test_dataset_missing_task_json_is_an_error(monkeypatch, tmp_path):
    files = SimpleNamespace(download=AsyncMock())
    monkeypatch.setattr(prefetch, 'require_context', lambda: SimpleNamespace(task_files=files))
    with pytest.raises(FileNotFoundError, match='task_files/task.json'):
        await prefetch.prefetch_task_json(tmp_path)


@pytest.mark.usefixtures('isolate_auth')
@pytest.mark.asyncio
async def test_plain_checkout_mode_still_has_no_prefetch(monkeypatch, tmp_path):
    monkeypatch.setattr(prefetch, 'require_context', lambda: SimpleNamespace(task_files=None))
    assert await prefetch.prefetch_task_json(tmp_path) is None


@pytest.mark.usefixtures('isolate_auth')
@pytest.mark.asyncio
async def test_key_from_task_config_reaches_first_signing_before_runtime(monkeypatch, tmp_path):
    key = {'id': 'fixture', 'subject': 'fixture', 'key': 'fixture-secret'}
    seen = []

    async def download(pattern, target):
        payload = task_files._optional_task_files_auth_payload()
        seen.append(payload)
        assert payload == {'kind': 'wedata_key', 'key': key}
        raise RuntimeError('stop-after-auth')

    ctx = SimpleNamespace(
        task_files=SimpleNamespace(download=download),
        get_runtime=AsyncMock(),
        tool_state={},
        run_id='',
        runtime_provider=None,
    )
    monkeypatch.setattr(prefetch, 'require_context', lambda: ctx)
    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    inp = OSWorldTaskInput(
        task_id='fixture',
        domain='chrome',
        wedata_key=key,
        cache_dir=str(tmp_path),
        run_agent=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match='no task_root fallback'):
        await task_module.run_os_world(inp)
    assert len(seen) == 1
    ctx.get_runtime.assert_not_called()


@pytest.mark.usefixtures('isolate_auth')
@pytest.mark.asyncio
async def test_mismatched_task_config_and_dataset_blob_fail_before_runtime(monkeypatch, tmp_path):
    source = tmp_path / 'task.json'
    source.write_text(json.dumps({'id': 'another-task', 'instruction': 'not the requested task'}))
    ctx = SimpleNamespace(get_runtime=AsyncMock(), tool_state={}, run_id='', runtime_provider=None)
    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(task_module, '_prefetch_task_json', AsyncMock(return_value=source))
    inp = OSWorldTaskInput(task_id='requested-task', domain='chrome', cache_dir=str(tmp_path), run_agent=AsyncMock())
    with pytest.raises(ValueError, match='does not match'):
        await task_module.run_os_world(inp)
    ctx.get_runtime.assert_not_called()


FIXTURE_WEDATA_KEY = '{"id": 1, "subject": "fixture-not-real", "key": "fixture-secret-not-real"}'


def _task_input(**task_config: Any) -> OSWorldTaskInput:
    """Build the task input the way ``execute_run`` does, filtering included.

    ``run_agent`` is required on the model and injected by the runner rather than by a
    config, so it is supplied here the same way ``single.py:870`` supplies it.
    """
    return _build_input(OSWorldTaskInput, {'run_agent': lambda **_: None, **task_config})


@pytest.mark.parametrize('field', ['cache_dir', 'wedata_key'])
def test_the_field_is_declared_so_build_input_keeps_it(field: str) -> None:
    """The filter's whole behaviour depends on the name being in ``model_fields``."""
    assert field in OSWorldTaskInput.model_fields


def test_cache_dir_arrives_from_task_config(tmp_path) -> None:
    task_input = _task_input(cache_dir=str(tmp_path / 'cache'))

    assert task_input.cache_dir == str(tmp_path / 'cache')
    assert _cache_dir(task_input) == str(tmp_path / 'cache')


def test_cache_dir_from_the_config_beats_the_environment(tmp_path, monkeypatch) -> None:
    """The platform route must win: ``run_single`` reads no environment there.

    Offline the env var is the convenient knob, but a platform run has only the config,
    so the config value cannot be the one that loses.
    """
    monkeypatch.setenv('NEXUS_OSWORLD_CACHE_DIR', str(tmp_path / 'from-env'))
    task_input = _task_input(cache_dir=str(tmp_path / 'from-config'))

    assert _cache_dir(task_input) == str(tmp_path / 'from-config')
    assert resolve_evaluator_cache_dir('task-a', _cache_dir(task_input)).is_relative_to(tmp_path / 'from-config')


def test_no_cache_dir_falls_through_to_the_legacy_field(monkeypatch) -> None:
    """Nothing passed must not invent a root: the older field name still speaks.

    ``setup_cache_dir`` is the same root under its previous name, and it only gets a say
    once ``cache_dir`` and the env var are both silent -- so an old config keeps working
    and a new one is never shadowed by it.
    """
    monkeypatch.delenv('NEXUS_OSWORLD_CACHE_DIR', raising=False)
    task_input = _task_input()

    assert task_input.cache_dir is None
    assert _cache_dir(task_input) == task_input.setup_cache_dir


def test_a_blank_cache_dir_counts_as_unset(monkeypatch) -> None:
    """Whitespace is not a cache root, and must not out-rank the env var either."""
    monkeypatch.setenv('NEXUS_OSWORLD_CACHE_DIR', '/tmp/from-env')
    task_input = _task_input(cache_dir='   ')

    assert _cache_dir(task_input) is None  # None = let paths read the env var


def test_with_nothing_configured_at_all_the_root_is_packaged(monkeypatch) -> None:
    """``Path('')`` is ``Path('.')`` and truthy, so an empty root must not become the cwd."""
    monkeypatch.delenv('NEXUS_OSWORLD_CACHE_DIR', raising=False)
    task_input = _task_input(setup_cache_dir='')

    assert resolve_evaluator_cache_dir('task-a', _cache_dir(task_input)).is_relative_to(DEFAULT_CACHE_ROOT)


def test_wedata_key_arrives_as_a_string_and_is_published(monkeypatch) -> None:
    """The local shape: a raw JSON string, as ``export WEDATA_KEY=...`` gives it."""
    monkeypatch.delenv(ENV_WEDATA_KEY, raising=False)
    task_input = _task_input(wedata_key=FIXTURE_WEDATA_KEY)

    prefetch.apply_wedata_key(task_input.wedata_key)

    assert os.environ[ENV_WEDATA_KEY] == FIXTURE_WEDATA_KEY


def test_wedata_key_arrives_as_a_mapping_and_is_published_as_json(monkeypatch) -> None:
    """The platform shape: ``${oc.env:WEDATA_KEY}`` resolves to a parsed dict before it
    reaches the task, and ``WedataKey.from_raw`` only parses the string form."""
    monkeypatch.delenv(ENV_WEDATA_KEY, raising=False)
    task_input = _task_input(wedata_key={'id': 1, 'subject': 'fixture-not-real', 'key': 'fixture-secret-not-real'})

    prefetch.apply_wedata_key(task_input.wedata_key)

    # Double quotes, not Python's repr: single quotes are not valid JSON and auth.py parses this.
    assert '"subject": "fixture-not-real"' in os.environ[ENV_WEDATA_KEY]


def test_the_generators_interpolation_survives_as_a_field_value(monkeypatch) -> None:
    """What the generator actually writes is the interpolation, never the resolved secret.

    It must still reach the field: locally nothing resolves it, so the task receives the
    literal ``${oc.env:WEDATA_KEY}`` and a pre-existing env var keeps working.
    """
    monkeypatch.setenv(ENV_WEDATA_KEY, FIXTURE_WEDATA_KEY)
    task_input = _task_input(wedata_key=WEDATA_KEY_INTERPOLATION)

    assert task_input.wedata_key == WEDATA_KEY_INTERPOLATION


def test_no_wedata_key_leaves_a_pre_existing_credential_alone(monkeypatch) -> None:
    """A plain local run passes nothing and must keep using its key file or env var."""
    monkeypatch.setenv(ENV_WEDATA_KEY, FIXTURE_WEDATA_KEY)
    task_input = _task_input()

    prefetch.apply_wedata_key(task_input.wedata_key)

    assert task_input.wedata_key is None
    assert os.environ[ENV_WEDATA_KEY] == FIXTURE_WEDATA_KEY
