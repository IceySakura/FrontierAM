"""File-oriented OSWorld evaluator getters."""

from __future__ import annotations

import os
import shutil

from asyncio import to_thread
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.runtime import (
    cache_dir_from_config,
    download_url,
    get_file_bytes,
    write_bytes,
)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _osworld_repo_root(runtime: Any) -> Path | None:
    for attr in ('osworld_repo_root', 'osworld_root'):
        value = getattr(runtime, attr, None)
        if value:
            return Path(str(value))

    examples_root = getattr(runtime, 'osworld_examples_root', None)
    if examples_root:
        path = Path(str(examples_root))
        if path.name == 'examples' and path.parent.name == 'evaluation_examples':
            return path.parent.parent
        return path

    return None


def _resolve_local_file_source(runtime: Any, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path]

    if not path.is_absolute():
        repo_root = _osworld_repo_root(runtime)
        if repo_root is not None:
            candidates.append(repo_root / path)
            if path.parts[:2] == ('evaluation_examples', 'examples'):
                candidates.append(repo_root / Path(*path.parts[2:]))

        onboard_root = Path('workspace/project/osworld-onboard/OSWorld')
        candidates.append(onboard_root / path)
        if path.parts[:2] == ('evaluation_examples', 'examples'):
            candidates.append(onboard_root / Path(*path.parts[2:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'OSWorld local_file source not found: {raw_path}; tried: {tried}')


@getter_registry.register()
async def get_local_file(runtime: Any, config: dict[str, Any]) -> str:
    """Copy a host-local evaluator fixture into the evaluator cache.

    If the fixture is already in the cache at ``dest``, use it as-is. That is the
    dataset path: a packaged run pre-populates the evaluator cache from
    ``task_files``, and off the source machine there is no OSWorld checkout for
    the repo-relative ``path`` to resolve against, so the cache hit is the only
    way this getter can succeed there.
    """

    dest = cache_dir_from_config(config) / str(config.get('dest') or Path(str(config['path'])).name)
    if dest.is_file():
        return str(dest)

    source = _resolve_local_file_source(runtime, str(config['path']))

    def _copy() -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, dest)

    await to_thread(_copy)
    return str(dest)


@getter_registry.register()
async def get_vm_file(runtime: Any, config: dict[str, Any]) -> str | list[str | None] | None:
    """Download file(s) from the VM into the evaluator cache and return local paths."""

    if not config.get('multi', False):
        paths = [config['path']]
        dests = [config['dest']]
        if config.get('time_suffix', False):
            time_format = str(config.get('time_format', '%Y%m%d_%H%M%S'))
            dests = [
                f'{os.path.splitext(str(dest))[0]}_{datetime.now().strftime(time_format)}'
                f'{os.path.splitext(str(dest))[1]}'
                for dest in dests
            ]
    else:
        paths = _as_list(config['path'])
        dests = _as_list(config['dest'])

    cache_dir = cache_dir_from_config(config)
    gives = set(config.get('gives', [0]))
    cache_paths: list[str | None] = []

    for idx, (remote_path, dest) in enumerate(zip(paths, dests, strict=False)):
        local_path = cache_dir / str(dest)
        content = await get_file_bytes(runtime, str(remote_path))
        if content is None:
            if idx in gives:
                cache_paths.append(None)
            continue
        await write_bytes(local_path, content)
        if idx in gives:
            cache_paths.append(str(local_path))

    return cache_paths[0] if len(cache_paths) == 1 else cache_paths


@getter_registry.register()
async def get_cloud_file(runtime: Any, config: dict[str, Any]) -> str | list[str]:
    """Download expected file(s) from URLs into the evaluator cache."""

    if not config.get('multi', False):
        urls = [config['path']]
        dests = [config['dest']]
    else:
        urls = _as_list(config['path'])
        dests = _as_list(config['dest'])

    cache_dir = cache_dir_from_config(config)
    gives = set(config.get('gives', [0]))
    cache_paths: list[str] = []

    for idx, (url, dest) in enumerate(zip(urls, dests, strict=False)):
        local_path = cache_dir / str(dest)
        if idx in gives:
            cache_paths.append(str(local_path))
        if not local_path.exists():
            await download_url(str(url), local_path)

    return cache_paths[0] if len(cache_paths) == 1 else cache_paths


@getter_registry.register()
async def get_cache_file(runtime: Any, config: dict[str, Any]) -> str:
    path = cache_dir_from_config(config) / str(config['path'])
    if not path.exists():
        raise FileNotFoundError(str(path))
    return str(path)


@getter_registry.register()
async def get_content_from_vm_file(runtime: Any, config: dict[str, Any]) -> Any:
    remote_path = str(config['path'])
    local_path = await get_vm_file(
        runtime,
        {
            **config,
            'path': remote_path,
            'dest': Path(remote_path).name,
        },
    )
    if config.get('file_type') == 'xlsx' and config.get('file_content') == 'last_row':
        import pandas as pd

        df = pd.read_excel(str(local_path))
        return df.iloc[-1].astype(str).tolist()
    raise NotImplementedError(f'Unsupported content_from_vm_file config: {config!r}')
