"""File getters for WAA evaluation.

These getters download files from the VM or check file existence.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile

from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.file')


@getter_registry.register()
async def get_vm_file(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Download file(s) from the VM to local cache.

    Config:
        path (str|List[str]): absolute path(s) on the VM to fetch
        dest (str|List[str]): file name(s) for the downloaded file(s)
        multi (bool): optional, if path and dest are lists. defaults to False
        gives (List[int]): optional, which files are returned. defaults to [0]
        time_suffix (bool): optional, append current time to filename. defaults to False
        time_format (str): optional, format for time suffix. defaults to "%Y_%m_%d"
    """
    from nexus.extensions.tasks.waa.setup import cache_dir

    time_format = '%Y_%m_%d'
    if not config.get('multi', False):
        paths: list[str] = [config['path']]
        dests: list[str] = [config['dest']]
        if config.get('time_suffix'):
            if 'time_format' in config:
                time_format = config['time_format']
            paths = [
                p.split('.')[0] + datetime.now().strftime(time_format) + '.' + p.split('.')[1] if '.' in p else p
                for p in paths
            ]
            dests = [
                d.split('.')[0] + datetime.now().strftime(time_format) + '.' + d.split('.')[1] if '.' in d else d
                for d in dests
            ]
    else:
        paths = config['path']
        dests = config['dest']

    cache_paths: list[str | None] = []
    gives: set[int] = set(config.get('gives', [0]))

    for i, (p, d) in enumerate(zip(paths, dests, strict=False)):
        local_path = str(cache_dir / d)
        file_content = await runtime.get_file(p)
        if file_content is None:
            if i in gives:
                cache_paths.append(None)
            continue

        if i in gives:
            cache_paths.append(local_path)
        await asyncio.to_thread(_write_file, local_path, file_content)

    return cache_paths[0] if len(cache_paths) == 1 else cache_paths


def _write_file(path: str, content: bytes) -> None:
    with open(path, 'wb') as f:
        f.write(content)


def _cloud_file_cache_path(cache_dir: Path, url: str, dest: str) -> Path:
    suffix = Path(dest).suffix
    cache_key = hashlib.sha256(f'{url}\0{dest}'.encode()).hexdigest()
    return cache_dir / f'cloud_file_{cache_key}{suffix}'


def _is_retryable_cloud_file_response(url: str, status_code: int) -> bool:
    # expected/gold 文件下载来自多个外部源，实测 GitHub raw/CDN 会短时返回 400/5xx；
    # 这里按 HTTP 状态统一判断瞬时下载失败，不限定 host，避免同类 CDN 抖动再次打断全量评测。
    return status_code in {400, 408, 429, 500, 502, 503, 504}


async def _download_cloud_file_with_retry(url: str, dest: str, cache_path: Path) -> bytes:
    from nexus.utils.async_http import http_client

    retry_delays = (2.0,) * 30
    max_attempts = len(retry_delays) + 1
    last_error: Exception | None = None
    last_status: int | None = None
    last_body = ''

    for attempt in range(max_attempts):
        try:
            response = await http_client.get(url)
        except Exception as exc:
            last_error = exc
            if attempt >= len(retry_delays):
                break
            delay = retry_delays[attempt]
            logger.warning(
                'WAA evaluator cloud file download error; retrying: attempt=%s/%s url=%s error=%r',
                attempt + 1,
                max_attempts,
                url,
                exc,
            )

            if delay > 0:
                await asyncio.sleep(delay)
            continue

        last_status = response.status_code
        last_body = response.text[:500]
        if response.status_code < 400:
            data = response.content
            await asyncio.to_thread(cache_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(cache_path.write_bytes, data)
            return data

        retryable = _is_retryable_cloud_file_response(url, response.status_code) and attempt < len(retry_delays)
        if not retryable:
            raise RuntimeError(
                'expected_resource_download_failed: '
                f'HTTP {response.status_code} url={url!r} dest={dest!r} body={last_body!r}'
            )

        delay = retry_delays[attempt]
        logger.warning(
            'WAA evaluator cloud file download transient failure; retrying: '
            'attempt=%s/%s url=%s http_status=%s body=%r',
            attempt + 1,
            max_attempts,
            url,
            response.status_code,
            last_body[:200],
        )
        if delay > 0:
            await asyncio.sleep(delay)

    if last_status is not None:
        raise RuntimeError(
            'expected_resource_download_failed: exhausted retries '
            f'HTTP {last_status} url={url!r} dest={dest!r} body={last_body!r}'
        ) from last_error
    raise RuntimeError(
        f'expected_resource_download_failed: exhausted retries url={url!r} dest={dest!r}'
    ) from last_error


@getter_registry.register()
async def get_vm_file_exists_in_vm_folder(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a file exists in the specified VM folder.

    Returns 1.0 if the file exists, 0.0 otherwise.
    """
    folder_name = config['folder_name']
    file_name = config['file_name']
    if not folder_name or not file_name:
        return 0.0

    file_path = os.path.join(folder_name, file_name)
    file_content = await runtime.get_file(file_path)
    return 1.0 if file_content else 0.0


@getter_registry.register()
async def get_cloud_file(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Download file(s) from a URL to local cache.

    Config:
        path (str|List[str]): URL(s) to download from
        dest (str|List[str]): file name(s) for the downloaded file(s)
        multi (bool): optional, if path and dest are lists. defaults to False
        gives (List[int]): optional, which files are returned. defaults to [0]
    """
    from nexus.extensions.tasks.waa.setup import cache_dir

    cloud_cache_dir = Path(
        os.environ.get('WAA_DOWNLOAD_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'nexus_waa_download_cache'))
    )

    if not config.get('multi', False):
        paths: list[str] = [config['path']]
        dests: list[str] = [config['dest']]
    else:
        paths = config['path']
        dests = config['dest']

    cache_paths: list[str] = []
    gives: set[int] = set(config.get('gives', [0]))

    for i, (url, dest) in enumerate(zip(paths, dests, strict=False)):
        local_path = cache_dir / dest
        if i in gives:
            cache_paths.append(str(local_path))

        if local_path.exists():
            continue

        url_cache_path = _cloud_file_cache_path(cloud_cache_dir, url, dest)
        if url_cache_path.exists():
            data = await asyncio.to_thread(url_cache_path.read_bytes)
        else:
            # evaluator 拉 expected 文件失败不是 agent 行为问题；重试并持久缓存，减少外部网络对分数的影响。
            data = await _download_cloud_file_with_retry(url, dest, url_cache_path)

        await asyncio.to_thread(local_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(local_path.write_bytes, data)

    return cache_paths[0] if len(cache_paths) == 1 else cache_paths
