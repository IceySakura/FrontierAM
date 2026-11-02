from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from packaging.utils import canonicalize_name

from nexus.core.module_dependencies import ModuleDependencyError, resolve_host_module_dependencies
from nexus.core.module_meta import discover_modules_from_whl
from nexus.distributed.artifacts.gateway import (
    ApplicationGatewayArtifactTransport,
    DependencyBuildArtifactGateway,
)
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.dependency_build.logs import (
    BUILD_LOG_PORT,
    configure_build_log,
    create_build_log_app,
    output_logger,
)
from nexus.distributed.dependency_build.validation import (
    read_project_constraints,
    read_target_lock,
    validate_wheels_against_lock,
)
from nexus.distributed.dependency_build.wheelhouse import (
    NexusWheelMetadata,
    WheelhouseBuildSpec,
    WheelhouseMetadata,
    inspect_nexus_wheel,
    resolve_wheelhouse_build_key,
    write_wheelhouse_archive,
)
from nexus.distributed.observability.event_loop import monitor_event_loop
from nexus.distributed.submission.source_package import safe_extract_source_package
from nexus.distributed.utils.clock import monotonic_deadline_from_iso8601
from nexus.utils.provider_io import redact_provider_io


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


CommandRunner = Callable[..., Any]
logger = logging.getLogger(__name__)
_VERIFY_INSTALL_SCRIPT = """
import importlib.metadata
import json
import re
import sys

import nexus

actual = importlib.metadata.version('nexus-agent')
if actual != sys.argv[1]:
    raise SystemExit(f'nexus-agent version mismatch: expected={sys.argv[1]} actual={actual}')

expected = json.loads(sys.argv[2])
installed = {}
for distribution in importlib.metadata.distributions():
    name = re.sub(r'[-_.]+', '-', distribution.metadata.get('Name', '')).lower()
    if not name:
        raise SystemExit('installed distribution has no name')
    if name in installed:
        raise SystemExit(f'duplicate installed distribution: {name}')
    installed[name] = distribution.version
if installed != expected:
    raise SystemExit(f'installed distributions do not match target lock: expected={expected} actual={installed}')
""".strip()


class WheelhouseBuildError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _emit(text: str, *, console: bool = True) -> None:
    if output_logger.handlers:
        output_logger.info(text.rstrip('\n'), extra={'_build_console': console})
    elif console:
        print(text, end='', flush=True)


def _progress(message: str) -> None:
    _emit(f'[build] {message}\n')


def _run_streaming(command: list[str], *, cwd: Path, **kwargs: Any) -> None:
    """Drain command output continuously with bounded memory and console output."""
    tail = ''
    remaining = 2 * 1024 * 1024
    omitted = False
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PIP_PROGRESS_BAR': 'off', 'UV_NO_PROGRESS': '1'}
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    ) as process:
        assert process.stdout is not None
        try:
            while line := process.stdout.readline(8193):
                if len(line) > 8192:
                    # Discard the entire oversized line: splitting could expose a partial credential.
                    while not line.endswith('\n'):
                        line = process.stdout.readline(8193)
                        if not line:
                            break
                    line = '[build] oversized output line omitted\n'
                safe = str(redact_provider_io(line))
                tail = (tail + safe)[-65536:]
                size = len(safe.encode('utf-8'))
                to_console = size <= remaining and not omitted
                _emit(safe, console=to_console)
                if to_console:
                    remaining -= size
                elif not omitted:
                    _progress('command console output limit reached; further output available in build log file')
                    omitted = True
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output=tail, stderr=tail)


def build_wheelhouse(
    *,
    spec: WheelhouseBuildSpec,
    source_package: Path,
    nexus_wheel: Path,
    uv_lock: Path,
    workspace: Path,
    expected_build_key: str | None = None,
    runner: CommandRunner = _run_streaming,
) -> tuple[Path, WheelhouseMetadata]:
    _progress('Validating build inputs and extracting source')
    try:
        resolve_wheelhouse_build_key(spec, expected_build_key)
    except ValueError as exc:
        raise WheelhouseBuildError('dependency_build_key_mismatch', str(exc)) from exc
    try:
        wheel_metadata = inspect_nexus_wheel(nexus_wheel)
    except ValueError as exc:
        raise WheelhouseBuildError('dependency_nexus_wheel_invalid', str(exc)) from exc

    if workspace.is_symlink():
        raise WheelhouseBuildError('dependency_workspace_invalid', 'workspace must not be a symbolic link')
    workspace = workspace.resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    source_dir = workspace / 'source'
    wheels_dir = workspace / 'wheels'
    requirements = workspace / 'requirements.txt'
    module_requirements = workspace / 'module-host.in'
    module_lock = workspace / 'module-host.lock'
    nexus_root_requirements = workspace / 'nexus-root.in'
    nexus_root_dir = workspace / 'nexus-root'
    constraints = workspace / 'project-constraints.txt'
    wheels_dir.mkdir(parents=True)
    try:
        safe_extract_source_package(
            source_package,
            source_dir,
            limits=spec.source_package_limits.to_package_limits(),
        )
    except ValueError as exc:
        raise WheelhouseBuildError('dependency_source_package_invalid', str(exc)) from exc
    if not (source_dir / 'pyproject.toml').is_file():
        raise WheelhouseBuildError('dependency_source_project_invalid', 'source package pyproject.toml is missing')
    packaged_lock = source_dir / 'uv.lock'
    packaged_lock.unlink(missing_ok=True)
    shutil.copyfile(uv_lock, packaged_lock)
    packaged_lock.chmod(0o444)
    project_constraints = _validate_project_inputs(source_dir, wheel_metadata, extras=spec.extras)
    try:
        resolved_modules = resolve_host_module_dependencies(
            spec.module_selectors,
            discover_modules_from_whl(nexus_wheel),
        )
    except ModuleDependencyError as exc:
        raise WheelhouseBuildError(exc.code, exc.detail) from exc

    export_command = [
        'uv',
        'export',
        '--frozen',
        '--no-dev',
        '--no-editable',
        '--no-emit-project',
        '--format',
        'requirements-txt',
        '--output-file',
        str(requirements),
    ]
    for extra in spec.extras:
        export_command.extend(['--extra', extra])
    _run_checked(
        runner,
        export_command,
        cwd=source_dir,
        code='dependency_lock_export_failed',
        description='frozen dependency export failed',
    )
    if resolved_modules.requirements:
        nexus_root_dir.mkdir()
        canonical_nexus_wheel = nexus_root_dir / wheel_metadata.filename
        shutil.copyfile(nexus_wheel, canonical_nexus_wheel)
        extra_suffix = f'[{",".join(spec.extras)}]' if spec.extras else ''
        nexus_root_requirements.write_text(
            f'nexus-agent{extra_suffix} @ {canonical_nexus_wheel.as_uri()}\n',
            encoding='utf-8',
        )
        module_requirements.write_text('\n'.join(resolved_modules.requirements) + '\n', encoding='utf-8')
        shutil.copyfile(requirements, module_lock)
        compile_command = [
            'uv',
            'pip',
            'compile',
            '--no-config',
            '--no-sources',
            '--no-emit-package',
            'nexus-agent',
            '--generate-hashes',
            '--python-version',
            platform.python_version(),
            '--python-platform',
            'x86_64-unknown-linux-gnu',
            '--output-file',
            str(module_lock),
        ]
        if project_constraints:
            constraints.write_text('\n'.join(project_constraints) + '\n', encoding='utf-8')
            compile_command.extend(['--constraint', str(constraints)])
        compile_command.extend([str(nexus_root_requirements), str(module_requirements)])
        _run_checked(
            runner,
            compile_command,
            cwd=source_dir,
            code='dependency_module_lock_failed',
            description='joint project and module dependency resolution failed',
        )
        if not module_lock.is_file():
            raise WheelhouseBuildError(
                'dependency_module_lock_failed',
                'joint project and module dependency resolver did not produce a lock',
            )
        _log_dependency_changes(requirements, module_lock)
    target_lock = module_lock if resolved_modules.requirements else requirements
    wheel_command = [
        sys.executable,
        '-m',
        'pip',
        'wheel',
        '--disable-pip-version-check',
        '--no-cache-dir',
        '--no-build-isolation',
        '--require-hashes',
        '--requirement',
        str(target_lock),
    ]
    wheel_command.extend(['--wheel-dir', str(wheels_dir)])
    _run_checked(
        runner,
        wheel_command,
        cwd=source_dir,
        code='dependency_wheel_build_failed',
        description='dependency resolution or wheel build failed using the pinned builder environment',
    )
    try:
        locked_versions = validate_wheels_against_lock(target_lock, wheels_dir)
    except ValueError as exc:
        raise WheelhouseBuildError('dependency_wheelhouse_inconsistent', str(exc)) from exc
    packaged_nexus_wheel = wheels_dir / wheel_metadata.filename
    shutil.copyfile(nexus_wheel, packaged_nexus_wheel)
    _verify_offline_install(
        runner,
        workspace=workspace,
        source_dir=source_dir,
        wheels_dir=wheels_dir,
        nexus_wheel=packaged_nexus_wheel,
        extras=spec.extras,
        module_host_requirements=list(resolved_modules.requirements),
        locked_versions=locked_versions,
        expected_version=wheel_metadata.version,
    )
    _progress('Packaging wheelhouse archive')
    archive = workspace / 'wheelhouse.tar.zst'
    metadata = write_wheelhouse_archive(
        wheels_dir,
        spec,
        archive,
        nexus_wheel_path=f'wheels/{wheel_metadata.filename}',
        expected_build_key=expected_build_key,
        module_names=[module.name for module in resolved_modules.modules],
        module_paths=[module.module_path for module in resolved_modules.modules],
        module_host_requirements=list(resolved_modules.requirements),
        module_lock=module_lock if resolved_modules.requirements else None,
    )
    return archive, metadata


async def run_builder(
    *,
    spec: WheelhouseBuildSpec,
    base_url: str,
    token_file: Path,
    workspace: Path,
    expected_build_key: str | None = None,
) -> ArtifactRef:
    log_dir = Path(os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_LOG_DIR', str(workspace.parent / 'logs')))
    log_path = log_dir / 'build.jsonl'
    handler = configure_build_log(
        log_path,
        build_id=os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_ID', ''),
        attempt=os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_ATTEMPT', '0'),
    )
    server = None
    try:
        read_token_file = os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_LOG_TOKEN_FILE', '')
        if read_token_file:
            server = web.AppRunner(create_build_log_app(log_path, token=Path(read_token_file).read_text().strip()))
            await server.setup()
            await web.TCPSite(server, '0.0.0.0', BUILD_LOG_PORT).start()
        return await _run_builder(
            spec=spec,
            base_url=base_url,
            token_file=token_file,
            workspace=workspace,
            expected_build_key=expected_build_key,
        )
    finally:
        if server is not None:
            await server.cleanup()
        output_logger.removeHandler(handler)
        await asyncio.to_thread(handler.close)


async def _run_builder(
    *,
    spec: WheelhouseBuildSpec,
    base_url: str,
    token_file: Path,
    workspace: Path,
    expected_build_key: str | None = None,
) -> ArtifactRef:
    downloads = workspace.parent / f'{workspace.name}-downloads'
    if downloads.exists():
        shutil.rmtree(downloads)
    downloads.mkdir(parents=True)
    transport = ApplicationGatewayArtifactTransport(base_url=base_url, content_ticket_file=token_file)
    client = DependencyBuildArtifactGateway(transport)
    deadline_value = os.getenv('NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT', '').strip()
    download_deadline = monotonic_deadline_from_iso8601(deadline_value) if deadline_value else None
    try:
        source = downloads / 'source.tar.zst'
        nexus_wheel = downloads / 'nexus-agent.whl'
        uv_lock = downloads / 'uv.lock'
        _progress('Downloading source package')
        await client.download(spec.source_package, source, deadline=download_deadline)
        _progress('Downloading Nexus wheel')
        await client.download(spec.nexus_wheel, nexus_wheel, deadline=download_deadline)
        _progress('Downloading dependency lock')
        await client.download(spec.uv_lock, uv_lock, deadline=download_deadline)
        archive, metadata = await asyncio.to_thread(
            build_wheelhouse,
            spec=spec,
            source_package=source,
            nexus_wheel=nexus_wheel,
            uv_lock=uv_lock,
            workspace=workspace,
            expected_build_key=expected_build_key,
        )
        _progress('Uploading wheelhouse')
        result = await client.upload_wheelhouse(archive, metadata=metadata)
        _progress('Build complete')
        return result
    except Exception as exc:
        error_code, error_message = _failure_record(exc)
        _progress(f'Build failed: {error_code}: {error_message}')
        try:
            await client.report_failure(error_code=error_code, error_message=error_message)
        except Exception:
            logger.exception('Could not report dependency build failure: code=%s', error_code)
        raise
    finally:
        await transport.close()


def _validate_project_inputs(
    source_dir: Path,
    wheel: NexusWheelMetadata,
    *,
    extras: list[str],
) -> tuple[str, ...]:
    try:
        with (source_dir / 'pyproject.toml').open('rb') as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise WheelhouseBuildError(
            'dependency_source_project_invalid', f'source project metadata is invalid: {exc}'
        ) from exc
    project = document.get('project')
    if not isinstance(project, dict) or canonicalize_name(str(project.get('name') or '')) != 'nexus-agent':
        raise WheelhouseBuildError(
            'dependency_source_project_invalid',
            'source project distribution must be nexus-agent',
        )
    source_version = str(project.get('version') or '').strip()
    if not source_version:
        raise WheelhouseBuildError('dependency_source_project_invalid', 'source project version is missing')
    if source_version != wheel.version:
        raise WheelhouseBuildError(
            'dependency_input_mismatch',
            f'source project version does not match nexus-agent wheel: source={source_version} wheel={wheel.version}',
        )

    source_optional = project.get('optional-dependencies') or {}
    if not isinstance(source_optional, dict):
        raise WheelhouseBuildError(
            'dependency_source_project_invalid',
            'source project optional-dependencies must be a table',
        )
    source_extras = {canonicalize_name(str(value)) for value in source_optional}
    requested_extras = {canonicalize_name(value) for value in extras}
    missing_source = sorted(requested_extras - source_extras)
    if missing_source:
        raise WheelhouseBuildError(
            'dependency_extra_missing',
            f'source project does not define extras: {", ".join(missing_source)}',
        )

    missing_wheel = sorted(requested_extras - wheel.extras)
    if missing_wheel:
        raise WheelhouseBuildError(
            'dependency_extra_missing',
            f'nexus-agent wheel does not define extras: {", ".join(missing_wheel)}',
        )
    try:
        return read_project_constraints(document)
    except ValueError as exc:
        raise WheelhouseBuildError('dependency_source_project_invalid', str(exc)) from exc


def _verify_offline_install(
    runner: CommandRunner,
    *,
    workspace: Path,
    source_dir: Path,
    wheels_dir: Path,
    nexus_wheel: Path,
    extras: list[str],
    module_host_requirements: list[str],
    locked_versions: dict[str, str],
    expected_version: str,
) -> None:
    verify_venv = workspace / 'verify-venv'
    verify_python = verify_venv / 'bin' / 'python'
    _run_checked(
        runner,
        ['uv', 'venv', '--no-config', '--clear', '--python', sys.executable, str(verify_venv)],
        cwd=source_dir,
        code='dependency_offline_venv_failed',
        description='offline verification venv creation failed',
    )
    extra_suffix = f'[{",".join(extras)}]' if extras else ''
    nexus_target = f'{nexus_wheel}{extra_suffix}'
    install_prefix = [
        'uv',
        'pip',
        'install',
        '--no-config',
        '--link-mode',
        'copy',
        '--python',
        str(verify_python),
        '--no-index',
        '--find-links',
        str(wheels_dir),
    ]
    _run_checked(
        runner,
        [*install_prefix, nexus_target],
        cwd=source_dir,
        code='dependency_offline_install_failed',
        description='Nexus wheel and selected extras cannot install offline',
    )
    if module_host_requirements:
        _run_checked(
            runner,
            [*install_prefix, *module_host_requirements],
            cwd=source_dir,
            code='dependency_offline_install_failed',
            description='module Host requirements cannot install offline',
        )
    _run_checked(
        runner,
        ['uv', 'pip', 'check', '--python', str(verify_python)],
        cwd=source_dir,
        code='dependency_offline_validation_failed',
        description='offline installation has inconsistent package metadata',
    )
    expected_distributions = {**locked_versions, 'nexus-agent': expected_version}
    verify_command = [
        str(verify_python),
        # Verify the installed wheel without importing source or ambient PYTHONPATH packages.
        '-I',
        '-c',
        _VERIFY_INSTALL_SCRIPT,
        expected_version,
        json.dumps(expected_distributions, sort_keys=True, separators=(',', ':')),
    ]
    _run_checked(
        runner,
        verify_command,
        cwd=source_dir,
        code='dependency_offline_validation_failed',
        description='offline Nexus import or metadata validation failed',
    )


def _locked_versions_for_roots(lock_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, requirement in read_target_lock(lock_path).items():
        versions = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator in {'==', '==='} and '*' not in specifier.version
        ]
        if len(versions) != 1:
            raise ValueError(f'target lock requirement is not exactly pinned: {requirement}')
        result[name] = versions[0]
    return result


def _log_dependency_changes(base_lock: Path, target_lock: Path) -> None:
    try:
        before = _locked_versions_for_roots(base_lock)
        after = _locked_versions_for_roots(target_lock)
    except ValueError:
        return
    changes = [
        f'{name}={before.get(name, "-")}->{after.get(name, "-")}'
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    ]
    if not changes:
        return
    visible = changes[:50]
    suffix = f' (+{len(changes) - len(visible)} more)' if len(changes) > len(visible) else ''
    logger.info('Joint module resolution changed dependencies: %s%s', ', '.join(visible), suffix)


def _run_checked(
    runner: CommandRunner,
    command: list[str],
    *,
    cwd: Path,
    code: str,
    description: str,
) -> None:
    _progress(f'Starting {code.removesuffix("_failed")}')
    try:
        runner(command, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = _bounded_command_error(exc)
        raise WheelhouseBuildError(code, f'{description}: {detail}') from exc
    except subprocess.TimeoutExpired as exc:
        raise WheelhouseBuildError(code, f'{description}: command timed out') from exc


def _bounded_command_error(error: subprocess.CalledProcessError) -> str:
    detail = error.stderr or error.stdout or str(error)
    if isinstance(detail, bytes):
        detail = detail.decode(errors='replace')
    compact = ' '.join(str(redact_provider_io(detail)).split())
    return (compact or f'command exited with status {error.returncode}')[-1800:]


def _failure_record(error: Exception) -> tuple[str, str]:
    raw_code = error.code if isinstance(error, WheelhouseBuildError) else type(error).__name__
    error_code = re.sub(r'[^A-Za-z0-9_.:-]+', '_', str(raw_code)).strip('_')[:128] or 'DependencyBuildFailed'
    error_message = ' '.join(str(redact_provider_io(str(error))).split())[:2048] or error_code
    return error_code, error_message


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build an offline Nexus wheelhouse')
    parser.add_argument('--workspace', default=os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_WORKSPACE', '/work'))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        spec = WheelhouseBuildSpec.model_validate_json(os.environ['NEXUS_DIST_DEPENDENCY_BUILD_SPEC_JSON'])
        base_url = os.environ['NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL']
        token_file = Path(os.environ['NEXUS_DIST_ARTIFACT_CONTENT_TOKEN_FILE'])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f'invalid dependency builder configuration: {exc}') from exc
    ref = asyncio.run(
        monitor_event_loop(
            'dependency-builder',
            run_builder(
                spec=spec,
                base_url=base_url,
                token_file=token_file,
                workspace=Path(args.workspace),
                expected_build_key=os.getenv('NEXUS_DIST_DEPENDENCY_BUILD_KEY'),
            ),
        )
    )
    print(json.dumps({'wheelhouse': ref.model_dump(mode='json')}, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()


__all__ = ['WheelhouseBuildError', 'build_wheelhouse', 'main', 'run_builder']
