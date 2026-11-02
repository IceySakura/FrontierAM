import base64
import hashlib
import zipfile

from pathlib import Path

import pytest

from nexus.core.module_meta import ModuleMeta
from nexus.runtime.server.venv import manager
from nexus.runtime.server.venv.models import ArtifactSpec


def test_collect_module_runtime_seeds_uses_compiled_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_modules = [
        ModuleMeta(
            name='code_interpreter',
            module_path='nexus.extensions.tools.code_interpreter',
            fs_path=Path(__file__),
            runtime_extras=('ipykernel>=6.29.0', 'jupyter-client>=8.6.0'),
        )
    ]

    monkeypatch.setattr(manager, '_discover_modules_for_runtime_deps', lambda whl_paths: manifest_modules)

    seeds = manager._collect_module_runtime_seeds([], modules=['code_interpreter'])

    assert ('ipykernel', frozenset(), 'ipykernel>=6.29.0') in seeds
    assert ('jupyter-client', frozenset(), 'jupyter-client>=8.6.0') in seeds


@pytest.mark.parametrize('source', ['base64', 'file', 'localhost'])
async def test_materialize_whl_preserves_an_installable_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    wheel = tmp_path / '本地 wheel #1%.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('nexus_agent-1.5.0.dist-info/METADATA', 'Name: nexus-agent\nVersion: 1.5.0\n')
        archive.writestr(
            'nexus_agent-1.5.0.dist-info/WHEEL',
            'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        )
    materialized = tmp_path / 'materialized'
    materialized.mkdir()
    monkeypatch.setattr(manager.tempfile, 'mkdtemp', lambda **_kwargs: str(tmp_path / 'materialized'))
    monkeypatch.setattr(manager.aiohttp, 'ClientSession', lambda **_kwargs: pytest.fail('Unexpected HTTP download'))

    content = wheel.read_bytes()
    artifact = ArtifactSpec(name='nexus-agent', whl_sha256=hashlib.sha256(content).hexdigest())
    if source == 'base64':
        artifact.whl_data = base64.b64encode(content).decode()
    else:
        artifact.whl_url = wheel.as_uri()
        if source == 'localhost':
            artifact.whl_url = artifact.whl_url.replace('file:///', 'file://localhost/', 1)

    path = await manager.VenvManager(root=tmp_path / 'venvs')._materialize_whl(artifact)

    assert path.name == 'nexus_agent-1.5-py3-none-any.whl'
    assert path.read_bytes() == content
    assert wheel.read_bytes() == content


async def test_materialize_local_whl_rejects_checksum_mismatch(tmp_path: Path) -> None:
    wheel = tmp_path / 'payload.whl'
    wheel.write_bytes(b'incorrect wheel contents')

    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        await manager.VenvManager(root=tmp_path / 'venvs')._materialize_whl(
            ArtifactSpec(name='nexus-agent', whl_url=wheel.as_uri(), whl_sha256='0' * 64)
        )


async def test_materialize_local_whl_rejects_remote_authority(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='local file'):
        await manager.VenvManager(root=tmp_path / 'venvs')._materialize_whl(
            ArtifactSpec(name='nexus-agent', whl_url='file://remote-host/payload.whl')
        )


async def test_install_artifact_copies_files_out_of_the_volatile_uv_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

    async def create_subprocess_exec(*args: str, **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess()

    monkeypatch.setattr(manager.asyncio, 'create_subprocess_exec', create_subprocess_exec)

    await manager.VenvManager(root=tmp_path / 'venvs')._install_artifact(
        tmp_path / 'venv',
        tmp_path / 'nexus_agent-1.5-py3-none-any.whl',
        ArtifactSpec(name='nexus-agent'),
        no_deps=True,
    )

    assert commands
    assert commands[0][commands[0].index('--link-mode') + 1] == 'copy'
    assert '--no-config' in commands[0]


async def test_install_artifact_uses_wheelhouse_without_network_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

    async def create_subprocess_exec(*args: str, **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess()

    monkeypatch.setattr(manager.asyncio, 'create_subprocess_exec', create_subprocess_exec)
    wheelhouse = tmp_path / 'wheelhouse'
    wheelhouse.mkdir()

    await manager.VenvManager(root=tmp_path / 'venvs')._install_artifact(
        tmp_path / 'venv',
        tmp_path / 'nexus_agent-1.5-py3-none-any.whl',
        ArtifactSpec(name='nexus-agent', extras=['runtime', 'distributed']),
        wheelhouse=wheelhouse,
    )

    assert len(commands) == 1
    assert '--no-index' in commands[0]
    assert commands[0][commands[0].index('--find-links') + 1] == str(wheelhouse)
    assert '--no-deps' not in commands[0]
    assert '--no-config' in commands[0]


async def test_create_bare_venv_ignores_caller_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

    async def create_subprocess_exec(*args: str, **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess()

    monkeypatch.setattr(manager.asyncio, 'create_subprocess_exec', create_subprocess_exec)

    await manager.VenvManager(root=tmp_path / 'venvs')._create_bare_venv('python3.13', tmp_path / 'venv')

    assert len(commands) == 1
    assert commands[0][1] == 'venv'
    assert '--no-config' in commands[0]


async def test_install_extra_ignores_caller_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'', b''

    async def create_subprocess_exec(*args: str, **_kwargs) -> CompletedProcess:
        commands.append(args)
        return CompletedProcess()

    monkeypatch.setattr(manager.asyncio, 'create_subprocess_exec', create_subprocess_exec)

    await manager.VenvManager(root=tmp_path / 'venvs')._install_extra(
        tmp_path / 'venv',
        ['litellm>=1.91.2,<1.92.0'],
        wheelhouse=tmp_path / 'wheelhouse',
    )

    assert len(commands) == 1
    assert commands[0][:3] == (commands[0][0], 'pip', 'install')
    assert '--no-config' in commands[0]


def test_swe_agent_runtime_includes_file_editor_dependencies() -> None:
    editor_dependencies = set(manager._collect_module_runtime_seeds([], modules=['file_editor']))
    agent_dependencies = set(manager._collect_module_runtime_seeds([], modules=['swe_agent']))

    assert editor_dependencies
    assert editor_dependencies <= agent_dependencies
