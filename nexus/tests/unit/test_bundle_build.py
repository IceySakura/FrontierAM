import hashlib
import json
import sys
import tarfile

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / 'deployment/bootstrap/windows/main.py'
    spec = spec_from_file_location('bundle_build', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_digest_matches_sha256(builder, tmp_path):
    payload = tmp_path / 'payload'
    payload.write_bytes(b'nexus')
    assert builder.digest(payload) == hashlib.sha256(b'nexus').hexdigest()


def test_flat_manifest_contains_only_runtime_payload(builder, tmp_path):
    for filename in ('entrypoint.py', 'python/python.exe', 'tools/uv.exe', 'python/__pycache__/test.pyc'):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'payload')
    for filename in ('nexus.whl', 'build.log', 'build.json', 'foundation-manifest.json', '.gitignore'):
        (tmp_path / filename).write_bytes(b'not-runtime')
    (tmp_path / 'wheels').mkdir()
    (tmp_path / 'wheels/nexus.whl').write_bytes(b'not-runtime')
    files = builder.payload_files(tmp_path)
    assert [item['path'] for item in files] == ['entrypoint.py', 'python/python.exe', 'tools/uv.exe']
    assert all(item['sha256'] == hashlib.sha256(b'payload').hexdigest() for item in files)


def test_build_log_is_outside_bundle(builder, monkeypatch, tmp_path):
    output = tmp_path / 'bundle'
    monkeypatch.setattr(builder, 'is_windows', lambda: True)
    monkeypatch.setattr(builder.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(builder.shutil, 'which', lambda name: 'uv')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--output', str(output)])

    def stop_before_build(*args, **kwargs):
        raise RuntimeError('stop before build')

    monkeypatch.setattr(builder, 'run', stop_before_build)
    with pytest.raises(RuntimeError, match='stop before build'):
        builder.main()
    assert output / 'windows-x86_64.build.log' == builder.BUILD_LOG
    assert builder.BUILD_LOG.is_file()
    assert not (output / 'build.log').exists()


def test_build_help_has_no_bootstrap_argument(builder, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['main.py', '--help'])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert '--output' in output
    assert '--arch' in output
    assert '--bootstrap' not in output


def test_base_python_copy_excludes_site_packages_and_bytecode(builder):
    assert builder.ignore_python_files('python/Lib/site-packages', ['unrelated', 'editable.pth']) == [
        'unrelated',
        'editable.pth',
    ]
    assert set(builder.ignore_python_files('python/Lib', ['os.py', '__pycache__', 'os.pyc'])) == {
        '__pycache__',
        'os.pyc',
    }


def test_existing_output_is_not_overwritten(builder, monkeypatch, tmp_path):
    output = tmp_path / 'output'
    (output / 'windows-x86_64').mkdir(parents=True)
    sentinel = output / 'windows-x86_64/keep'
    sentinel.write_bytes(b'keep')
    monkeypatch.setattr(builder, 'is_windows', lambda: True)
    monkeypatch.setattr(builder.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert sentinel.read_bytes() == b'keep'


def test_other_platform_output_is_preserved(builder, monkeypatch, tmp_path):
    output = tmp_path / 'output'
    other = output / 'linux-aarch64/manifest.json'
    other.parent.mkdir(parents=True)
    other.write_bytes(b'keep')
    monkeypatch.setattr(builder, 'is_windows', lambda: True)
    monkeypatch.setattr(builder.platform, 'machine', lambda: 'AMD64')
    monkeypatch.setattr(builder.shutil, 'which', lambda name: 'uv')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--output', str(output)])

    def stop_before_build(*args, **kwargs):
        raise RuntimeError('stop before build')

    monkeypatch.setattr(builder, 'run', stop_before_build)
    with pytest.raises(RuntimeError, match='stop before build'):
        builder.main()
    assert (output / 'windows-x86_64').is_dir()
    assert other.read_bytes() == b'keep'


def test_other_platform_is_rejected_without_creating_output(builder, monkeypatch, tmp_path):
    output = tmp_path / 'output'
    monkeypatch.setattr(builder, 'is_windows', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert not output.exists()


def test_unsupported_target_arch_is_rejected_without_creating_output(builder, monkeypatch, tmp_path):
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['main.py', '--arch', 'aarch64', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert not output.exists()


def test_unsupported_host_arch_is_rejected_without_creating_output(builder, monkeypatch, tmp_path):
    output = tmp_path / 'output'
    monkeypatch.setattr(builder, 'is_windows', lambda: True)
    monkeypatch.setattr(builder.platform, 'machine', lambda: 'ARM64')
    monkeypatch.setattr(sys, 'argv', ['main.py', '--arch', 'x86_64', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert not output.exists()


def test_tar_release_has_relative_archive_and_no_build_artifacts(builder, tmp_path):
    payload = tmp_path / 'payload'
    output = tmp_path / 'output'
    for name in ('entrypoint.py', 'python/python.exe', 'tools/uv.exe', 'wheels/unused.whl', 'build.json'):
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    result = builder.pack_release(payload, output, version='1.8.8', commit='a' * 40, arch='x86_64')
    assert (output / 'manifest-1.8.8.json').read_bytes() == (output / 'manifest.json').read_bytes()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['files'] == [
        {
            'path': 'bundles/1.8.8.tar.gz',
            'sha256': result['archive_sha256'],
            'extract': True,
        }
    ]
    assert manifest['entrypoint_sha256'] == hashlib.sha256(b'python/python.exe').hexdigest()
    with tarfile.open(result['archive']) as archive:
        assert archive.getnames() == ['entrypoint.py', 'python/python.exe', 'tools/uv.exe']
        assert archive.extractfile('entrypoint.py').read() == b'entrypoint.py'
    assert not (output / 'python').exists()
    assert not (output / 'entrypoint.py').exists()
    second = builder.pack_release(payload, tmp_path / 'second', version='1.8.8', commit='a' * 40, arch='x86_64')
    assert second['archive_sha256'] == result['archive_sha256']
    with pytest.raises(FileExistsError):
        builder.pack_release(payload, output, version='1.8.8', commit='a' * 40, arch='x86_64')


@pytest.mark.parametrize(('version', 'commit'), [('../bad', 'a' * 40), ('1.0', 'not-a-commit')])
def test_archive_rejects_unsafe_release_names(builder, tmp_path, version, commit):
    with pytest.raises(ValueError, match='safe version'):
        builder.pack_release(tmp_path, tmp_path / 'out', version=version, commit=commit, arch='x86_64')


def test_entrypoint_uses_loopback_for_worker_connections(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    path = Path(__file__).resolve().parents[2] / 'deployment/bootstrap/windows/entrypoint.py'
    spec = spec_from_file_location('bundle_entrypoint', path)
    entrypoint = module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    configure, run = Mock(), Mock()
    app = object()
    monkeypatch.setitem(sys.modules, 'nexus.runtime.server.app', SimpleNamespace(configure=configure, asgi_app=app))
    monkeypatch.setitem(sys.modules, 'uvicorn', SimpleNamespace(run=run))
    monkeypatch.setattr(entrypoint, 'bundle_info', lambda: {})
    monkeypatch.setattr(entrypoint.os, 'environ', dict(entrypoint.os.environ, NEXUS_RUNTIME_PORT='18888'))
    monkeypatch.setattr(
        sys, 'argv', ['entrypoint.py', '--state-root', str(tmp_path / 'state'), '--workspace', str(tmp_path / 'workspace')]
    )
    monkeypatch.chdir(tmp_path)
    entrypoint.main()
    configure.assert_called_once_with(host='127.0.0.1', port=18888)
    run.assert_called_once_with(app, host='0.0.0.0', port=18888, ws_max_size=100 * 1024 * 1024)


@pytest.mark.parametrize(
    ('override', 'accepted'),
    [
        ('litellm>=1.80,<1.82.7', True),
        ('litellm>=1.83', False),
        ('litellm>=1.80,<1.82.7; python_version < "3.0"', False),
    ],
)
def test_dependency_versions_follow_active_uv_overrides(builder, monkeypatch, tmp_path, override, accepted):
    from types import SimpleNamespace

    config = {
        'project': {'dependencies': ['litellm>=1.83.14,<2'], 'optional-dependencies': {'runtime': []}},
        'tool': {'uv': {'override-dependencies': [override]}},
    }
    (tmp_path / 'pyproject.toml').write_text('')
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder.tomllib, 'loads', lambda _: config)
    monkeypatch.setattr(builder, 'discover_all_modules', lambda: [])
    dist = SimpleNamespace(
        metadata={'Name': 'litellm'}, version='1.82.6', files=[], requires=[], read_text=lambda _: None
    )
    monkeypatch.setattr(builder.importlib.metadata, 'distribution', lambda _: dist)
    if accepted:
        assert builder.dependency_distributions() == {'litellm': dist}
    else:
        with pytest.raises(RuntimeError, match='Installed version does not satisfy'):
            builder.dependency_distributions()
