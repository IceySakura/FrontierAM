from __future__ import annotations

import asyncio
import subprocess
import sys
import tarfile
import threading

from pathlib import Path

import pytest

from nexus.runtime.bridge import artifacts
from nexus.runtime.bridge.artifacts import _bridge_runtime_requirements_file, build_app_bundle
from nexus.runtime.bridge.runtime_kind import (
    NativeBridgeBundleTarget,
    NativeBridgeRuntimeKind,
    classify_native_bridge_runtime,
    native_bridge_bundle_target_for_runtime,
)
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.ags.waa import AGSWindowsAgentArenaRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime


class FakeOSWorldRuntime(AGSOSWorldRuntime):
    def __init__(self) -> None:
        pass


class FakeRawOSWorldRuntime(AGSRawOSWorldRuntime):
    def __init__(self, nexus_runtime: FakeOSWorldRuntime) -> None:
        self.nexus_runtime = nexus_runtime

    def to_nexus(self) -> FakeOSWorldRuntime:
        return self.nexus_runtime


class FakeWAARuntime(AGSWindowsAgentArenaRuntime):
    def __init__(self) -> None:
        pass


def _bundle_text(output: Path, name: str) -> str:
    with tarfile.open(output, 'r:gz') as tf:
        member = tf.extractfile(name)
        assert member is not None
        return member.read().decode()


def test_classify_native_bridge_runtime_from_runtime_type() -> None:
    assert classify_native_bridge_runtime(FakeOSWorldRuntime()) is NativeBridgeRuntimeKind.AGS_OSWORLD
    assert classify_native_bridge_runtime(FakeRawOSWorldRuntime(FakeOSWorldRuntime())) is (
        NativeBridgeRuntimeKind.AGS_OSWORLD
    )
    devcloud_runtime = DevCloudRuntime(base_url='http://127.0.0.1:5000')
    assert classify_native_bridge_runtime(FakeWAARuntime()) is NativeBridgeRuntimeKind.AGS_WAA
    assert classify_native_bridge_runtime(devcloud_runtime) is NativeBridgeRuntimeKind.DEVCLOUD_MACOS
    assert native_bridge_bundle_target_for_runtime(FakeOSWorldRuntime()) is NativeBridgeBundleTarget.LINUX_X86_64_PY310
    assert native_bridge_bundle_target_for_runtime(FakeWAARuntime()) is NativeBridgeBundleTarget.WINDOWS_AMD64_PY310
    assert native_bridge_bundle_target_for_runtime(devcloud_runtime) is NativeBridgeBundleTarget.MACOS_X86_64_PY311


def test_build_app_bundle_contains_manifest_and_launchers(tmp_path: Path) -> None:
    output = tmp_path / 'nexus-llm-bridge-app-linux-x86_64-py310.tar.gz'

    build_app_bundle(output=output, target='linux-x86_64-py310', default_port=8080, include_vendor=False)

    with tarfile.open(output, 'r:gz') as tf:
        names = set(tf.getnames())
    assert 'nexus-llm-bridge/manifest.json' in names
    assert 'nexus-llm-bridge/bin/launch_bridge.sh' in names
    assert 'nexus-llm-bridge/src/nexus/runtime/bridge/__main__.py' in names


def test_arm64_macos_bundle_uses_isolated_python_and_detached_launcher(tmp_path: Path) -> None:
    output = tmp_path / 'bridge.tar.gz'
    build_app_bundle(output=output, target='macos-arm64-py311', default_port=8765, include_vendor=False)
    launcher = _bundle_text(output, 'nexus-llm-bridge/bin/launch_bridge.sh')
    assert '"$HOME/.local/share/nexus-native-bridge/venv/bin/python"' in launcher
    assert 'start_new_session=True' in launcher
    assert 'nohup' not in launcher
    subprocess.run(['bash', '-n'], input=launcher, text=True, check=True)
    assert 'macos-arm64-py311' in _bundle_text(output, 'nexus-llm-bridge/manifest.json')


def test_build_app_bundle_contains_anthropic_proxy_runtime_source_dependencies(tmp_path: Path) -> None:
    output = tmp_path / 'nexus-llm-bridge-app-linux-x86_64-py310.tar.gz'

    build_app_bundle(output=output, target='linux-x86_64-py310', default_port=8080, include_vendor=False)

    with tarfile.open(output, 'r:gz') as tf:
        names = set(tf.getnames())
    assert 'nexus-llm-bridge/src/nexus/utils/common.py' in names
    # anthropic_proxy.server imports create_detached_task at module load, so the
    # bundled bridge must ship event_loop.py and include it in cache invalidation.
    assert 'nexus-llm-bridge/src/nexus/utils/event_loop.py' in names
    assert 'nexus/utils/event_loop.py' in artifacts._HASHED_SOURCE_PATHS
    # anthropic_proxy.server imports RSIOLLMBridge at module load, so it must be vendored
    # or the bundled bridge crashes on startup (ModuleNotFoundError).
    assert 'nexus-llm-bridge/src/nexus/runtime/server/services/rsio_llm_bridge.py' in names
    assert 'nexus-llm-bridge/src/nexus/runtime/server/services/responses_proxy/server.py' in names
    assert 'nexus/runtime/server/services/responses_proxy' in artifacts._HASHED_SOURCE_PATHS


def test_build_app_bundle_uses_lightweight_namespace_inits(tmp_path: Path) -> None:
    output = tmp_path / 'nexus-llm-bridge-app-linux-x86_64-py310.tar.gz'

    build_app_bundle(output=output, target='linux-x86_64-py310', default_port=8080, include_vendor=False)

    assert 'module_registry' not in _bundle_text(output, 'nexus-llm-bridge/src/nexus/__init__.py')


@pytest.mark.parametrize('api_protocol', ['anthropic', 'responses'])
def test_bridge_sources_import_without_repository_modules(tmp_path: Path, api_protocol: str) -> None:
    artifacts._copy_bridge_sources(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            '-I',
            '-c',
            f'import sys; sys.path.insert(0, {str(tmp_path / "src")!r}); '
            'from nexus.runtime.bridge.app import create_bridge_asgi_app; '
            f'create_bridge_asgi_app(api_protocol={api_protocol!r})',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'nexus/runtime/server/services/codex_compaction.py' in artifacts._HASHED_SOURCE_PATHS


def test_bridge_runtime_requirements_file_is_locked_for_cross_platform_bundles() -> None:
    requirements = _bridge_runtime_requirements_file().read_text()

    assert 'fastapi[standard]' not in requirements
    assert 'fastapi==0.139.2' in requirements
    assert 'websockets==16.1.1' in requirements
    assert 'exceptiongroup==1.3.1' in requirements
    assert 'tenacity==9.1.4' in requirements
    assert 'async-timeout==5.0.1' in requirements
    assert 'colorama==0.4.6\n' in requirements
    assert '>=' not in requirements


def test_build_macos_app_bundle_uses_devcloud_python311(tmp_path: Path) -> None:
    output = tmp_path / 'nexus-llm-bridge-app-macos-x86_64-py311.tar.gz'

    build_app_bundle(output=output, target='macos-x86_64-py311', default_port=8765, include_vendor=False)

    with tarfile.open(output, 'r:gz') as tf:
        names = set(tf.getnames())
    assert 'nexus-llm-bridge/python/bin/python3' not in names
    assert 'nexus-llm-bridge/vendor' in names
    assert 'nexus-llm-bridge/src/nexus/runtime/bridge/__main__.py' in names

    launch = _bundle_text(output, 'nexus-llm-bridge/bin/launch_bridge.sh')
    assert '/usr/local/bin/python3 -m nexus.runtime.bridge' in launch
    assert 'nohup python3 -m nexus.runtime.bridge' not in launch
    assert '--conversion-log-level "$CONVERSION_LOG_LEVEL" --thinking-config "$THINKING_CONFIG"' in launch
    assert '--launch-id "$LAUNCH_ID"' in launch
    assert '--request-timeout "$REQUEST_TIMEOUT"' in launch
    assert '--api-protocol "$API_PROTOCOL"' in launch


def test_build_windows_app_bundle_contains_powershell_launcher(tmp_path: Path) -> None:
    output = tmp_path / 'nexus-llm-bridge-app-windows-amd64-py310.tar.gz'

    build_app_bundle(output=output, target='windows-amd64-py310', default_port=8765, include_vendor=False)

    launch = _bundle_text(output, 'nexus-llm-bridge/bin/launch_bridge.ps1')
    assert "Start-Process -FilePath 'python'" in launch
    assert "'-m', 'nexus.runtime.bridge'" in launch
    assert "'--request-timeout', $RequestTimeout" in launch
    assert "'--api-protocol', $ApiProtocol" in launch
    assert '$env:PYTHONPATH = "$BridgeHome\\src;$BridgeHome\\vendor"' in launch
    assert 'bridge.pid' in launch

    launch_cmd = _bundle_text(output, 'nexus-llm-bridge/bin/launch_bridge.cmd')
    assert 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass' in launch_cmd
    assert '-File "%BRIDGE_HOME%\\bin\\launch_bridge.ps1"' in launch_cmd
    assert '-Port %BRIDGE_PORT%' in launch_cmd
    assert '-LaunchId "%LAUNCH_ID%"' in launch_cmd
    assert '-RequestTimeout "%REQUEST_TIMEOUT%"' in launch_cmd
    assert '-ApiProtocol "%API_PROTOCOL%"' in launch_cmd


@pytest.mark.asyncio
async def test_build_bridge_artifact_cached_coalesces_in_process_builds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(artifacts, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(artifacts, '_bridge_content_hash', lambda target, default_port: 'hash')
    artifacts._result_cache.clear()
    artifacts._inflight.clear()

    build_calls = 0

    def fake_locked_build(target: NativeBridgeBundleTarget, default_port: int, cache_key: str) -> Path:
        nonlocal build_calls
        build_calls += 1
        output = tmp_path / cache_key / artifacts._ARTIFACT_NAMES[target]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('bundle')
        return output

    monkeypatch.setattr(artifacts, '_locked_build', fake_locked_build)

    runtime = FakeOSWorldRuntime()
    first, second = await asyncio.gather(
        artifacts.build_bridge_artifact_cached(runtime),
        artifacts.build_bridge_artifact_cached(runtime),
    )

    assert first == second
    assert first.read_text() == 'bundle'
    assert build_calls == 1

    third = await artifacts.build_bridge_artifact_cached(runtime)
    assert third == first
    assert build_calls == 1


@pytest.mark.asyncio
async def test_build_bridge_artifact_waiter_cancellation_does_not_cancel_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(artifacts, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(artifacts, '_bridge_content_hash', lambda target, default_port: 'cancel-safe')
    artifacts._result_cache.clear()
    artifacts._inflight.clear()
    build_started = threading.Event()
    release_build = threading.Event()

    def fake_locked_build(target: NativeBridgeBundleTarget, default_port: int, cache_key: str) -> Path:
        build_started.set()
        assert release_build.wait(timeout=5)
        output = tmp_path / cache_key / artifacts._ARTIFACT_NAMES[target]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('bundle')
        return output

    monkeypatch.setattr(artifacts, '_locked_build', fake_locked_build)
    runtime = FakeOSWorldRuntime()
    owner = asyncio.create_task(artifacts.build_bridge_artifact_cached(runtime))
    await asyncio.to_thread(build_started.wait, 5)
    waiter = asyncio.create_task(artifacts.build_bridge_artifact_cached(runtime))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_build.set()
    result = await owner
    assert result.read_text() == 'bundle'
