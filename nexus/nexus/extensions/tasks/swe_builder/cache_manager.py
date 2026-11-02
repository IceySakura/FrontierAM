"""
Cache manager for nexus-builder package caches.

Only 4 safe cache types — pure download caches with no local path references:
  pip    → /root/.cache/pip        (wheel/http cache, no path refs)
  npm    → /root/.npm              (tarball cache, no path refs)
  maven  → /root/.m2/repository   (jar cache, coordinate-addressed)
  gradle → /root/.gradle/caches   (jar cache, coordinate-addressed)

Excluded (unsafe):
  cargo  — compiled artifacts embed registry source paths
  gem    — /usr/local/bundle is an install dir with bin symlinks
  go     — module cache embeds paths in debug info

Flow per container run:
  1. copy_in_to_container()  — after container init, copy host cache → container local dirs
  2. agent runs install.sh   — package managers use local cache, download missing packages
  3. copy_out_and_clean()    — before docker commit: copy container dirs → host cache, then
                               delete those dirs from container so committed image stays lean
"""

import asyncio
import logging
import os
import subprocess
import threading


logger = logging.getLogger(__name__)

CACHE_BASE = os.environ.get('NEXUS_CACHE_DIR', os.path.expanduser('~/dev/nexus_cache'))

# Only the 4 safe pure-download caches
SAFE_CACHE_VOLUMES: dict[str, dict[str, str]] = {
    'pip': {'host': f'{CACHE_BASE}/pip', 'container': '/root/.cache/pip'},
    'npm': {'host': f'{CACHE_BASE}/npm', 'container': '/root/.npm'},
    'maven': {'host': f'{CACHE_BASE}/maven', 'container': '/root/.m2/repository'},
    'gradle': {'host': f'{CACHE_BASE}/gradle', 'container': '/root/.gradle/caches'},
}

# Per-type locks: serialise concurrent copy-out writes from multiple containers
_lock_map: dict[str, threading.Lock] = {k: threading.Lock() for k in SAFE_CACHE_VOLUMES}


def setup_cache_dirs() -> str:
    """Create all cache directories. Safe to call multiple times."""
    dirs = [CACHE_BASE, f'{CACHE_BASE}/meta'] + [cfg['host'] for cfg in SAFE_CACHE_VOLUMES.values()]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return CACHE_BASE


def _cache_enabled() -> bool:
    """Cache is active only when the base directory exists (run setup_cache_dirs.sh first)."""
    return os.path.isdir(CACHE_BASE)


async def _run_docker(args: list[str], timeout: float) -> int:
    """Run a docker sub-command asynchronously. Returns exit code, never raises."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'docker',
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0
    except asyncio.TimeoutError:
        logger.warning(f'[CACHE] docker {args[0]} timed out ({timeout}s)')
        return 1
    except Exception as e:
        logger.warning(f'[CACHE] docker {args[0]} error: {e}')
        return 1


async def copy_in_to_container(container_name: str) -> None:
    """Copy host cache dirs into a running container.

    Best-effort: silently skips any type whose host dir is empty or whose copy fails.
    Called once after container is initialised, before the agent starts.
    """
    if not _cache_enabled():
        return

    setup_cache_dirs()
    copied: list[str] = []

    for cache_type, cfg in SAFE_CACHE_VOLUMES.items():
        host_path = cfg['host']
        container_path = cfg['container']

        # Skip empty host dirs — nothing to copy
        if not os.path.isdir(host_path) or not os.listdir(host_path):
            continue

        try:
            # Ensure target dir exists in container
            if await _run_docker(['exec', container_name, 'mkdir', '-p', container_path], timeout=15) != 0:
                continue

            # docker cp merges — existing container files are not deleted
            if (
                await _run_docker(
                    ['cp', f'{host_path}/.', f'{container_name}:{container_path}/'],
                    timeout=120,
                )
                == 0
            ):
                copied.append(cache_type)

        except Exception as e:
            logger.warning(f'[CACHE] copy-in {cache_type} error: {e}')

    if copied:
        logger.debug(f'[CACHE] copy-in done: {copied} → {container_name}')


async def copy_out_and_clean(container_name: str) -> None:
    """Copy cache from container back to host, then remove it from the container.

    Called just before docker commit so that:
    - Newly downloaded packages are persisted to host for future runs
    - The committed image contains no large cache dirs (lean image)

    Per-type threading lock serialises concurrent writes from parallel containers.
    """
    if not _cache_enabled():
        return

    setup_cache_dirs()
    saved: list[str] = []

    for cache_type, cfg in SAFE_CACHE_VOLUMES.items():
        host_path = cfg['host']
        container_path = cfg['container']

        try:
            # Skip if the dir doesn't exist in the container
            if await _run_docker(['exec', container_name, 'test', '-d', container_path], timeout=10) != 0:
                continue

            # Copy container → host under per-type lock (prevent concurrent corruption)
            def _do_copy_out(
                _cache_type: str = cache_type,
                _host_path: str = host_path,
                _container_path: str = container_path,
            ) -> int:
                with _lock_map[_cache_type]:
                    os.makedirs(_host_path, exist_ok=True)
                    result = subprocess.run(
                        ['docker', 'cp', f'{container_name}:{_container_path}/.', f'{_host_path}/'],
                        capture_output=True,
                        timeout=300,
                    )
                    return result.returncode

            loop = asyncio.get_event_loop()
            rc = await loop.run_in_executor(None, _do_copy_out)
            if rc == 0:
                saved.append(cache_type)

            # Remove from container regardless of copy result — keep committed image lean
            await _run_docker(['exec', container_name, 'rm', '-rf', container_path], timeout=60)

        except Exception as e:
            logger.warning(f'[CACHE] copy-out {cache_type} error: {e}')

    if saved:
        logger.debug(f'[CACHE] copy-out done: saved={saved} from {container_name}')


if __name__ == '__main__':
    setup_cache_dirs()
    print(f'Cache base: {CACHE_BASE}')
    for name, cfg in SAFE_CACHE_VOLUMES.items():
        host = cfg['host']
        size_gb = 0.0
        if os.path.isdir(host):
            for root, _, files in os.walk(host):
                size_gb += sum(os.path.getsize(os.path.join(root, f)) for f in files)
            size_gb /= 1024**3
        status = f'{size_gb:.2f} GB' if os.path.isdir(host) else 'missing'
        print(f'  {name:<8} {host}  {status}')
