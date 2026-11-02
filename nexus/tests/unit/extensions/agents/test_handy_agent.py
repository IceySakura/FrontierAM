from typing import Any

import pytest

from nexus.extensions.agents.handy import agent


@pytest.mark.asyncio
async def test_run_handy_invokes_runtime_with_expected_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured['target'] = target
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.25,
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    result = await agent.run_handy(
        agent.HandyInput(
            user_prompt='hello',
            working_dir='/workspace',
            extra_env={'SERPER_APP_ID': 'configured'},
        )
    )

    assert result.success is True
    assert result.output == 'done'
    assert captured['target'].__name__ == 'setup_and_run_handy'
    assert captured['version'] == '0.1.14'
    assert captured['model'] == 'claude-opus-5'
    assert captured['max_output_tokens'] == 64_000
    assert captured['extra_env'] == {'SERPER_APP_ID': 'configured'}
    # Artifact collection stays off unless asked for, so existing callers are unaffected.
    assert captured['artifact_dir'] is None
    assert result.artifacts == []
    # Likewise no session plugins, so the launch command matches the pre-feature one.
    assert captured['plugin_dirs'] == []


@pytest.mark.asyncio
async def test_run_handy_forwards_plugin_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0.1}

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    await agent.run_handy(
        agent.HandyInput(
            user_prompt='hello',
            working_dir='/workspace',
            plugin_dirs=['plugins/deep-research-worker', '/abs/ui-ux-pro-max'],
        )
    )

    # Resolution happens runtime-side against the install prefix; the host only forwards.
    assert captured['plugin_dirs'] == ['plugins/deep-research-worker', '/abs/ui-ux-pro-max']


def test_resolve_plugin_dirs_skips_unusable_entries(tmp_path) -> None:
    from nexus.extensions.agents.handy.runtime_setup import _resolve_plugin_dirs

    plugin = tmp_path / 'a-plugin'
    plugin.mkdir()
    a_file = tmp_path / 'not-a-dir.txt'
    a_file.write_text('x')

    assert _resolve_plugin_dirs(None) == []
    assert _resolve_plugin_dirs([]) == []
    # An empty or '.' entry would otherwise resolve to the package root and mount the
    # entire install as a single plugin.
    assert _resolve_plugin_dirs(['', '   ', '.']) == []
    # Missing paths and plain files are dropped: Handy only logs them at debug level,
    # so passing them through hides the mistake until the subagent turns up absent.
    assert _resolve_plugin_dirs(['plugins/does-not-exist']) == []
    assert _resolve_plugin_dirs([str(a_file)]) == []
    assert _resolve_plugin_dirs([f'  {plugin}  ']) == [str(plugin)]
    assert _resolve_plugin_dirs([str(plugin), str(plugin)]) == [str(plugin)]


def _manifest_entry(payload: bytes, **overrides: Any) -> dict[str, Any]:
    import hashlib

    entry = {
        'artifactId': 'art_1',
        'filename': 'report.html',
        'contentType': 'text/html',
        'size': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'title': 'Report',
        'createdAt': '2026-08-11T12:00:00.000Z',
        'sandbox_path': '/tmp/nexus/handy-artifacts/art_1/files/report.html',
    }
    entry.update(overrides)
    return entry


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, files: list[Any], errors: list[Any] | None = None) -> dict:
    """Stub the host-side runtime so download_file returns *files*."""
    from types import SimpleNamespace

    captured: dict[str, Any] = {}

    async def fake_download_file(files: list[Any]):
        captured['requested'] = [item.path for item in files]
        return SimpleNamespace(files=captured['response'], errors=errors)

    async def fake_get_runtime():
        return SimpleNamespace(download_file=fake_download_file)

    captured['response'] = files
    monkeypatch.setattr('nexus.core.require_context', lambda: SimpleNamespace(get_runtime=fake_get_runtime))
    return captured


@pytest.mark.asyncio
async def test_fetch_artifacts_writes_verified_files(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import base64

    from nexus.runtime.executors.models.file import FileItem

    payload = b'<h1>Nexus artifact</h1>'
    entry = _manifest_entry(payload)
    _patch_runtime(
        monkeypatch,
        [FileItem(path=entry['sandbox_path'], content=base64.b64encode(payload).decode(), encoding='base64')],
    )

    artifacts = await agent._fetch_artifacts([entry], str(tmp_path))

    assert len(artifacts) == 1
    stored = artifacts[0]
    assert stored.fetch_error is None
    assert stored.title == 'Report'
    # Namespaced by artifact_id, since Handy only guarantees a sanitised filename.
    assert stored.local_path == str(tmp_path / 'art_1' / 'report.html')
    assert (tmp_path / 'art_1' / 'report.html').read_bytes() == payload


@pytest.mark.asyncio
async def test_fetch_artifacts_rejects_checksum_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import base64

    from nexus.runtime.executors.models.file import FileItem

    entry = _manifest_entry(b'original', sha256='0' * 64)
    _patch_runtime(
        monkeypatch,
        [FileItem(path=entry['sandbox_path'], content=base64.b64encode(b'tampered').decode(), encoding='base64')],
    )

    artifacts = await agent._fetch_artifacts([entry], str(tmp_path))

    assert artifacts[0].local_path is None
    assert 'sha256 mismatch' in (artifacts[0].fetch_error or '')
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_fetch_artifacts_skips_files_over_transport_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    entry = _manifest_entry(b'x', size=agent.MAX_ARTIFACT_FETCH_BYTES + 1)

    def fail_get_runtime():  # pragma: no cover - must not be reached
        raise AssertionError('oversized artifacts must not be downloaded')

    monkeypatch.setattr('nexus.core.require_context', fail_get_runtime)
    artifacts = await agent._fetch_artifacts([entry], str(tmp_path))

    assert artifacts[0].local_path is None
    assert 'exceeds' in (artifacts[0].fetch_error or '')


@pytest.mark.asyncio
async def test_fetch_artifacts_reports_metadata_without_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    from nexus.runtime.executors.models.file import FileItem

    payload = b'body'
    entry = _manifest_entry(payload)
    _patch_runtime(
        monkeypatch,
        [FileItem(path=entry['sandbox_path'], content=base64.b64encode(payload).decode(), encoding='base64')],
    )

    artifacts = await agent._fetch_artifacts([entry], None)

    assert artifacts[0].local_path is None
    assert artifacts[0].fetch_error is None
    assert artifacts[0].sha256 == entry['sha256']


@pytest.mark.asyncio
async def test_run_handy_fetch_failure_preserves_run_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken artifact fetch must not discard the agent's completed work."""

    async def fake_call_runtime_agent(target, /, **kwargs):
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.25,
            'artifacts': [_manifest_entry(b'payload')],
        }

    async def exploding_fetch(manifest, output_dir):
        raise RuntimeError('sandbox already released')

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    monkeypatch.setattr(agent, '_fetch_artifacts', exploding_fetch)

    result = await agent.run_handy(
        agent.HandyInput(user_prompt='hello', working_dir='/workspace', collect_artifacts=True)
    )

    assert result.success is True
    assert result.output == 'done'
    assert result.artifacts == []
