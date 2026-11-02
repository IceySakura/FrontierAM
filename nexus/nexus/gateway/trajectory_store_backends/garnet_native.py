"""Compile and cache the Garnet extension when a client first needs it."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
_TIMEOUT: float = 300.0
_NATIVE_PATH: Path = Path(__file__).resolve().parent / 'native'
_LOAD_LOCK = threading.Lock()
_client_type: type[Any] | None = None


def load_native_client_type() -> type[Any]:
    """Return the client class, building once per source/ABI in a local disk cache.

    The first call blocks during compilation. Call this before serving requests,
    or run this module to prewarm each node. Other backends never call this loader.
    Lock acquisition and compilation each have a 300-second timeout.
    """
    global _client_type
    if _client_type is not None:
        return _client_type
    if sys.platform != 'linux':
        raise RuntimeError('Garnet runtime compilation requires Linux')
    if not _LOAD_LOCK.acquire(timeout=_TIMEOUT):
        raise RuntimeError(f'Timed out after {_TIMEOUT:g}s waiting for Garnet native initialization')
    try:
        if _client_type is not None:
            return _client_type
        # NOTE(jerryhrzhou): The lock dependency and compiler metadata are needed only when Garnet is selected.
        from filelock import FileLock, Timeout

        cache_root = Path('/tmp') / f'nexus-garnet-native-{os.getuid()}'
        cache_root.mkdir(mode=0o700, exist_ok=True)
        cache_key = _compute_cache_key()
        directory = cache_root / cache_key
        extension_path = directory / f'_garnet_native{sysconfig.get_config_var("EXT_SUFFIX")}'
        try:
            with FileLock(cache_root / f'{cache_key}.lock', timeout=_TIMEOUT):
                if not extension_path.is_file():
                    directory.mkdir(exist_ok=True)
                    _compile_extension(extension_path)
        except Timeout as error:
            raise RuntimeError(f'Timed out after {_TIMEOUT:g}s waiting for Garnet build lock: {directory}') from error

        module_name = 'nexus.gateway.trajectory_store_backends._garnet_native'
        try:
            # NOTE(jerryhrzhou): Always use the keyed cache, even if an old editable install left a source-tree .so.
            spec = importlib.util.spec_from_file_location(module_name, extension_path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot load Garnet extension: {extension_path}')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except (ImportError, OSError) as error:
            raise RuntimeError(
                f'Cannot load Garnet extension {extension_path}: {error}. '
                f'Remove cache directory {directory} and retry with matching Python and system libraries.'
            ) from error
        sys.modules[module_name] = module
        _client_type = module.NativeGarnetClient
        return _client_type
    finally:
        _LOAD_LOCK.release()


def _compute_cache_key() -> str:
    try:
        dependency_versions = {name: version(name) for name in ('pybind11', 'setuptools')}
    except PackageNotFoundError as error:
        raise RuntimeError('Garnet requires pybind11 and setuptools; install Nexus with the garnet extra') from error
    fingerprint = {
        'python': sys.version,
        'abi': sysconfig.get_config_var('SOABI'),
        'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
        'architecture': platform.machine(),
        'libc': platform.libc_ver(),
        'dependencies': dependency_versions,
        'python_build': {
            name: sysconfig.get_config_var(name)
            for name in ('CC', 'CXX', 'CFLAGS', 'CCSHARED', 'LDSHARED', 'LDCXXSHARED', 'LDFLAGS', 'AR', 'ARFLAGS')
        },
        'environment': {
            name: os.environ.get(name)
            for name in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'LDSHARED', 'AR', 'ARFLAGS')
        },
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    sources = [_NATIVE_PATH / 'build.py', _NATIVE_PATH / 'garnet_client.cpp']
    sources.extend(sorted((_NATIVE_PATH / 'vendor/hiredis').glob('*.[ch]')))
    for source in sources:
        try:
            contents = source.read_bytes()
        except FileNotFoundError as error:
            raise RuntimeError(
                f'Missing Garnet build source: {source}; reinstall Nexus from a complete package'
            ) from error
        digest.update(str(source.relative_to(_NATIVE_PATH)).encode())
        digest.update(b'\0')
        digest.update(contents)
    return digest.hexdigest()


def _compile_extension(extension_path: Path) -> None:
    directory = Path(tempfile.mkdtemp(prefix='build-', dir=extension_path.parent))
    log_path = directory / 'build.log'
    logger.debug('Compiling Garnet native extension; log: %s', log_path)
    try:
        with log_path.open('wb') as log:
            process = subprocess.Popen(
                [sys.executable, str(_NATIVE_PATH / 'build.py'), str(directory)],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=_TIMEOUT)
            except subprocess.TimeoutExpired:
                # NOTE(jerryhrzhou): The compiler is a child of the Python builder; stop the entire build on timeout.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if returncode != 0:
            raise RuntimeError(f'compiler exited with status {returncode}')
        shutil.copy2(log_path, extension_path.parent / 'build.log')
        os.replace(directory / extension_path.name, extension_path)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 8192))
            tail = log.read().decode(errors='replace')
        raise RuntimeError(
            f'Garnet native compilation failed: {error}. '
            f'Install a C++17 compiler, matching Python development headers, and Nexus[garnet]. '
            f'Build log: {log_path}\n{tail}'
        ) from error
    shutil.rmtree(directory)


if __name__ == '__main__':
    client_type = load_native_client_type()
    print(f'Garnet native extension ready: {sys.modules[client_type.__module__].__file__}')
