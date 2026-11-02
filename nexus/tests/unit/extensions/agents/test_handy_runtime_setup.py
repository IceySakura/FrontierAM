from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.handy import installer, runtime_setup
from nexus.extensions.agents.handy.runtime_setup import _build_handy_env, _collect_artifact_manifest


def _write_artifact(root: Path, artifact_id: str, filename: str, payload: str) -> Path:
    """Lay out one finished artifact the way Handy's storeLocalArtifact does."""
    import json

    files_dir = root / artifact_id / 'files'
    files_dir.mkdir(parents=True)
    (files_dir / filename).write_text(payload)
    (root / artifact_id / 'artifact.json').write_text(
        json.dumps({
            'version': 1,
            'state': 'complete',
            'artifactId': artifact_id,
            'filename': filename,
            'contentType': 'text/html',
            'size': len(payload),
            'sha256': 'deadbeef',
            'relativePath': f'{artifact_id}/files/{filename}',
        })
    )
    return files_dir / filename


def test_build_handy_env_keeps_nexus_routing_authoritative(tmp_path: Path) -> None:
    env = _build_handy_env(
        proxy_port=4321,
        working_dir='/workspace',
        session_root=tmp_path,
        extra_env={
            'ANTHROPIC_BASE_URL': 'https://bypass.invalid',
            'ANTHROPIC_API_KEY': 'real-secret',
            'ANTHROPIC_MODEL': 'bypass-model',
            'ANTHROPIC_DEFAULT_OPUS_MODEL': 'bypass-opus',
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '1',
            'HOME': '/outside',
            'HYNESS_APP_CONFIG': '/custom/app.yaml',
            'SERPER_APP_ID': 'search-id',
        },
    )

    assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:4321'
    assert env['ANTHROPIC_API_KEY'] == 'dummy-key-for-proxy'
    assert env['ANTHROPIC_MODEL'] == 'claude-opus-5'
    assert env['ANTHROPIC_DEFAULT_OPUS_MODEL'] == 'claude-opus-5'
    assert env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '64000'
    assert env['HOME'] == str(tmp_path / 'home')
    assert env['CLAUDE_CONFIG_DIR'] == str(tmp_path / 'config')
    assert env['SERPER_APP_ID'] == 'search-id'
    assert env['HYNESS_PI_TUI_YOLO'] == '1'
    assert 'HYNESS_APP_CONFIG' not in env


@pytest.mark.asyncio
async def test_setup_and_run_handy_installs_routes_and_stops_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    binary = tmp_path / 'installed' / 'handy'
    binary.parent.mkdir()
    binary.write_text('#!/usr/bin/env node\n')

    async def fake_install_handy_via_npm(**kwargs):
        captured['install'] = kwargs
        return binary

    class FakeProxy:
        def __init__(self, **kwargs):
            captured['proxy_args'] = kwargs

        async def start(self):
            captured['proxy_started'] = True

        async def stop(self):
            captured['proxy_stopped'] = True

    async def fake_run_handy_streaming(cmd, **kwargs):
        captured['cmd'] = cmd
        captured['run'] = kwargs
        return 0, 'completed', ''

    session_root = tmp_path / 'session'
    monkeypatch.setattr(installer, 'install_handy_via_npm', fake_install_handy_via_npm)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio='socket'))
    monkeypatch.setattr('nexus.runtime.server.services.anthropic_proxy.AnthropicProxy', FakeProxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 4321)
    monkeypatch.setattr(runtime_setup.tempfile, 'mkdtemp', lambda **kwargs: str(session_root))
    monkeypatch.setattr(runtime_setup, '_run_handy_streaming', fake_run_handy_streaming)

    result = await runtime_setup.setup_and_run_handy(
        'do the task',
        working_dir=str(tmp_path / 'workspace'),
        extra_env={'SERPER_APP_ID': 'search-id'},
    )

    assert result['success'] is True
    assert result['output'] == 'completed'
    assert captured['proxy_started'] is True
    assert captured['proxy_stopped'] is True
    assert captured['cmd'] == [str(binary), '--dangerously-skip-permissions', '-p', 'do the task']
    run = captured['run']
    assert run['env']['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:4321'
    assert run['env']['ANTHROPIC_MODEL'] == 'claude-opus-5'
    assert run['env']['ANTHROPIC_DEFAULT_OPUS_MODEL'] == 'claude-opus-5'
    assert run['env']['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] == '64000'
    assert run['env']['SERPER_APP_ID'] == 'search-id'
    assert not session_root.exists()


def test_build_handy_env_sets_artifact_dir_only_when_requested(tmp_path: Path) -> None:
    artifact_dir = tmp_path / 'artifacts'
    env = _build_handy_env(
        proxy_port=4321,
        working_dir='/workspace',
        session_root=tmp_path / 'session',
        extra_env=None,
        artifact_dir=str(artifact_dir),
    )
    assert env[runtime_setup.ARTIFACT_OUTPUT_DIR_ENV] == str(artifact_dir)
    assert artifact_dir.is_dir()

    without = _build_handy_env(
        proxy_port=4321,
        working_dir='/workspace',
        session_root=tmp_path / 'session2',
        extra_env=None,
    )
    assert runtime_setup.ARTIFACT_OUTPUT_DIR_ENV not in without


def test_build_handy_env_artifact_dir_survives_session_cleanup(tmp_path: Path) -> None:
    """The artifact dir must not live under session_root, which is deleted per run."""
    session_root = tmp_path / 'session'
    env = _build_handy_env(
        proxy_port=4321,
        working_dir='/workspace',
        session_root=session_root,
        extra_env=None,
        artifact_dir=str(tmp_path / 'artifacts'),
    )
    artifact_dir = Path(env[runtime_setup.ARTIFACT_OUTPUT_DIR_ENV])
    assert session_root not in artifact_dir.parents


def test_collect_artifact_manifest_reads_completed_artifacts(tmp_path: Path) -> None:
    _write_artifact(tmp_path, 'art_one', 'report.html', '<h1>one</h1>')
    _write_artifact(tmp_path, 'art_two', 'data.html', '<h1>two</h1>')

    manifest = _collect_artifact_manifest(str(tmp_path))

    assert [entry['artifactId'] for entry in manifest] == ['art_one', 'art_two']
    assert manifest[0]['sandbox_path'] == str(tmp_path / 'art_one' / 'files' / 'report.html')
    # size is re-read from disk rather than trusted from the metadata
    assert manifest[0]['size'] == len('<h1>one</h1>')


def test_collect_artifact_manifest_skips_staging_and_missing(tmp_path: Path) -> None:
    _write_artifact(tmp_path, 'art_done', 'ok.html', 'done')
    # An in-flight delivery: Handy renames it to art_* only once complete.
    staging = tmp_path / '.staging-art_partial' / 'files'
    staging.mkdir(parents=True)
    (staging / 'partial.html').write_text('half')
    # Metadata pointing at a file that is not there.
    orphan = tmp_path / 'art_orphan'
    orphan.mkdir()
    (orphan / 'artifact.json').write_text('{"artifactId": "art_orphan", "filename": "gone.html"}')

    manifest = _collect_artifact_manifest(str(tmp_path))

    assert [entry['artifactId'] for entry in manifest] == ['art_done']


def test_collect_artifact_manifest_handles_absent_dir() -> None:
    assert _collect_artifact_manifest(None) == []
    assert _collect_artifact_manifest('/nonexistent/handy/artifacts') == []


@pytest.mark.asyncio
async def test_setup_and_run_handy_returns_artifact_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / 'installed' / 'handy'
    binary.parent.mkdir()
    binary.write_text('#!/usr/bin/env node\n')
    artifact_dir = tmp_path / 'artifacts'

    async def fake_install_handy_via_npm(**kwargs):
        return binary

    class FakeProxy:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    async def fake_run_handy_streaming(cmd, **kwargs):
        # Stand in for the agent calling submit_file_artifact mid-run.
        _write_artifact(artifact_dir, 'art_generated', 'report.html', '<h1>hi</h1>')
        return 0, 'completed', ''

    monkeypatch.setattr(installer, 'install_handy_via_npm', fake_install_handy_via_npm)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio='socket'))
    monkeypatch.setattr('nexus.runtime.server.services.anthropic_proxy.AnthropicProxy', FakeProxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 4321)
    monkeypatch.setattr(runtime_setup.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path / 'session'))
    monkeypatch.setattr(runtime_setup, '_run_handy_streaming', fake_run_handy_streaming)

    result = await runtime_setup.setup_and_run_handy(
        'build a report',
        working_dir=str(tmp_path / 'workspace'),
        artifact_dir=str(artifact_dir),
    )

    assert result['success'] is True
    assert [entry['artifactId'] for entry in result['artifacts']] == ['art_generated']
