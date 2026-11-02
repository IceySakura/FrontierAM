"""HTTP 请求结果的轻量级文件缓存工具。

提供基于文件系统的持久化缓存，用于对 Jina / Serper 等外部 HTTP 请求进行缓存。
缓存条目以 JSON 文件形式存储，命中时直接读取本地文件，避免重复发起网络请求。

Cache key = MD5(json.dumps(selected_kwargs, sort_keys=True))
文件路径  = {cache_dir}/{namespace}/{key}.json

环境变量：
    SEARCH_CACHE_DIR: 缓存根目录，默认 ~/.cache/nexus/http_cache

装饰器用法示例：
    @http_cache(namespace='jina', cache_keys=['url'])
    async def jina_readpage(url: str, ..., use_cache: bool = False, cache_dir: str | None = None) -> str:
        ...

    # 调用时显式开启缓存
    result = await jina_readpage(url, ..., use_cache=True)
    result = await jina_readpage(url, ..., use_cache=False)   # 跳过缓存，直接请求

    # 只读模式：从缓存读取，miss 时不写入（避免 CephFS MDS 写瓶颈）
    result = await jina_readpage(url, ..., use_cache=True, cache_read_only=True)

说明：
    - 被装饰的函数签名需包含 use_cache: bool = False 和 cache_dir: str | None = None 两个可选参数。
    - 可选 cache_read_only: bool = False 参数：开启后只从缓存读取，miss 时直接调下游但不写入。
    - 当 use_cache=False 时，装饰器走直通路径，不会绑定参数、不会做缓存逻辑，也不会改写传参。
    - 仅当 use_cache=True 且 cache_read_only=False 且原始函数返回值为 str 时才写入缓存；返回错误前缀时跳过写入。
    - cache_keys 指定用于计算缓存 key 的参数名列表；不传则使用全部非 use_cache/cache_dir/cache_read_only 的参数。

并发模型：
    - 文件 I/O 通过 aiofiles 异步执行，避免阻塞 event loop。
    - 单进程内对相同 cache key 的并发请求会被 single-flight 合并，只触发一次真正的下游调用。
    - 多进程共享同一 cache 目录时，会使用每个 key 的文件锁避免重复下游请求。
    - 写入采用 tmp + os.replace 原子提交，跨进程也不会读到半写文件。
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import logging
import os
import uuid

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from filelock import AsyncFileLock


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-trajectory cache statistics (ContextVar-based, async-safe)
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    """Accumulator for cache hit/miss counts, keyed by namespace."""

    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)


_cache_stats: ContextVar[CacheStats | None] = ContextVar('_cache_stats', default=None)


def reset_cache_stats() -> None:
    """Reset cache statistics for the current async context (call before each trajectory)."""
    _cache_stats.set(CacheStats())


def get_cache_stats() -> dict[str, dict[str, int]] | None:
    """Return accumulated cache stats for the current context, or None if not initialized."""
    stats = _cache_stats.get()
    if stats is None:
        return None
    return {'hits': dict(stats.hits), 'misses': dict(stats.misses)}


def _record_hit(namespace: str) -> None:
    stats = _cache_stats.get()
    if stats is not None:
        stats.hits[namespace] = stats.hits.get(namespace, 0) + 1


def _record_miss(namespace: str) -> None:
    stats = _cache_stats.get()
    if stats is not None:
        stats.misses[namespace] = stats.misses.get(namespace, 0) + 1


_DEFAULT_CACHE_DIR = os.environ.get('SEARCH_CACHE_DIR', str(Path.home() / '.cache' / 'nexus' / 'http_cache'))

# Result strings that should NOT be cached (treated as transient failures).
_FAILURE_PREFIXES = ('[visit] Failed', '[visit error]', '[visit llm error]', 'Search Timeout', 'No results found')

# Per-(loop, namespace, key) in-flight futures for single-flight deduplication.
# The running loop id is part of the key so futures are never shared across event loops.
_INFLIGHT: dict[tuple[int, str, str], asyncio.Future] = {}

_MISSING = object()
_LOCK_TIMEOUT_SECONDS = 300.0


def _compute_key(key_kwargs: dict[str, Any]) -> str:
    raw = json.dumps(key_kwargs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(namespace: str, key: str, cache_dir: str | None) -> Path:
    base = Path(cache_dir) if cache_dir else Path(_DEFAULT_CACHE_DIR)
    return base / namespace / f'{key}.json'


def _lock_path(namespace: str, key: str, cache_dir: str | None) -> Path:
    return _cache_path(namespace, key, cache_dir).with_suffix('.lock')


def _normalize_cache_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('1', 'true', 'yes', 'on')


async def _read_cache(namespace: str, cache_dir: str | None, key_kwargs: dict[str, Any]) -> str | None:
    key = _compute_key(key_kwargs)
    path = _cache_path(namespace, key, cache_dir)
    try:
        # Use aiofiles.os.path.exists to avoid blocking the event loop on stat.
        exists = await aiofiles.os.path.exists(path)
    except Exception as e:
        logger.warning(f'[http_cache] stat error: {e}')
        return None
    if not exists:
        logger.debug(f'[http_cache] miss: namespace={namespace}, key={key}')
        _record_miss(namespace)
        return None
    try:
        async with aiofiles.open(path, encoding='utf-8') as f:
            text = await f.read()
        data = json.loads(text)
        logger.debug(f'[http_cache] hit: namespace={namespace}, key={key}')
        _record_hit(namespace)
        return data['value']
    except Exception as e:
        logger.warning(f'[http_cache] read error: {e}')
        # Best-effort: drop the corrupt file so we don't keep tripping on it.
        with contextlib.suppress(Exception):
            await aiofiles.os.remove(path)
        return None


async def _write_cache(value: str, namespace: str, cache_dir: str | None, key_kwargs: dict[str, Any]) -> None:
    key = _compute_key(key_kwargs)
    path = _cache_path(namespace, key, cache_dir)
    # Use a unique tmp name (pid + uuid) so concurrent writers don't clobber each other's tmp file.
    tmp_path = path.with_suffix(f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    loop = asyncio.get_running_loop()
    try:
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        # Best-effort dir permission tightening; ignore failures (e.g. NFS without chmod).
        # chmod is offloaded to a thread to avoid blocking the loop on slow filesystems.
        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, os.chmod, path.parent, 0o700)
        payload = json.dumps({'value': value, 'key_kwargs': key_kwargs}, ensure_ascii=False, indent=2)
        async with aiofiles.open(tmp_path, mode='w', encoding='utf-8') as f:
            await f.write(payload)
        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, os.chmod, tmp_path, 0o600)
        # os.replace is atomic on POSIX and Windows; readers never see a half-written file.
        # Use aiofiles.os.replace so a slow rename (e.g. on FUSE/NFS) doesn't block the loop.
        await aiofiles.os.replace(tmp_path, path)
        logger.debug(f'[http_cache] write: namespace={namespace}, key={key}')
    except Exception as e:
        logger.warning(f'[http_cache] write error: {e}')
        # Clean up stray tmp file on failure.
        with contextlib.suppress(Exception):
            await aiofiles.os.remove(tmp_path)


def http_cache(namespace: str, cache_keys: list[str] | None = None) -> Callable:
    """Async-aware HTTP cache decorator.

    The wrapped function must accept (optional) keyword arguments:
        use_cache: bool = False   — toggle cache lookup/write
        cache_dir: str | None = None — override cache root

    When use_cache=False, the decorator becomes a pass-through and forwards the call as-is.
    When use_cache=True, the decorator consumes cache control arguments and handles keying,
    single-flight, and file I/O itself.

    Additionally, callers may pass ``cache_read_only=True`` as a keyword argument to enable
    read-only mode: cache hits are returned normally, but misses are NOT written back.

    Args:
        namespace: Cache namespace, e.g. 'jina' or 'serper'.
        cache_keys: Argument names that participate in cache key computation.
                    Defaults to all arguments (excluding use_cache / cache_dir).

    Concurrency guarantees:
        - All filesystem I/O is async (aiofiles); the event loop is never blocked.
        - Concurrent calls with the same cache key are coalesced via single-flight; only
          one downstream invocation runs and the rest await its result.
        - Concurrent processes sharing the same cache dir are deduped via per-key file locks.
        - Cross-process writers are safe because writes use atomic tmp + os.replace.

    Example:
        @http_cache(namespace='jina', cache_keys=['url'])
        async def jina_readpage(url: str, app_id: str, ...) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f'@http_cache only supports async functions, got {func!r}')
        sig = inspect.signature(func)
        parameters = list(sig.parameters.values())
        use_cache_position = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.name == 'use_cache'
                and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ),
            None,
        )
        default_use_cache = sig.parameters['use_cache'].default if 'use_cache' in sig.parameters else False

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            raw_use_cache = kwargs.get('use_cache', _MISSING)
            if raw_use_cache is _MISSING and use_cache_position is not None and len(args) > use_cache_position:
                raw_use_cache = args[use_cache_position]
            if raw_use_cache is _MISSING:
                raw_use_cache = default_use_cache

            if not _normalize_cache_flag(raw_use_cache):
                # Pass-through: strip cache_read_only before forwarding to avoid TypeError
                # in functions that don't declare it in their signature.
                kwargs.pop('cache_read_only', None)
                return await func(*args, **kwargs)

            # Extract cache_read_only before sig.bind (it may not be in the decorated function's signature).
            cache_read_only = _normalize_cache_flag(kwargs.pop('cache_read_only', False))

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            all_args: dict[str, Any] = dict(bound.arguments)

            all_args.pop('use_cache', None)
            cache_dir: str | None = all_args.pop('cache_dir', None)
            # Also pop cache_read_only from all_args in case the function declares it with a default.
            all_args.pop('cache_read_only', None)

            key_kwargs = {k: v for k, v in all_args.items() if cache_keys is None or k in cache_keys}
            key = _compute_key(key_kwargs)
            loop = asyncio.get_running_loop()
            inflight_id = (id(loop), namespace, key)

            # Atomic claim: either we win the slot in the current event loop and become
            # leader, or we attach to an existing same-loop leader's future.
            # setdefault is sync, so no await can race between the lookup and the registration.
            future: asyncio.Future = loop.create_future()
            existing = _INFLIGHT.setdefault(inflight_id, future)
            if existing is not future:
                # Follower path: wait for the leader's result. shield prevents follower
                # cancellation from cancelling the leader's underlying call.
                try:
                    return await asyncio.shield(existing)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Leader failed; fall through and run the call ourselves.
                    future = loop.create_future()
                    # Try to take over the slot if the failed leader has already cleared it.
                    existing = _INFLIGHT.setdefault(inflight_id, future)
                    if existing is not future:
                        # Another follower already became the new leader; await it.
                        try:
                            return await asyncio.shield(existing)
                        except Exception:
                            # Give up on dedup; just run the call directly.
                            return await func(**all_args)

            # Leader path. We also take a per-key async file lock so concurrent processes
            # sharing the same cache dir don't all stampede the downstream fetch.
            try:
                await aiofiles.os.makedirs(_cache_path(namespace, key, cache_dir).parent, exist_ok=True)
                async with AsyncFileLock(_lock_path(namespace, key, cache_dir), timeout=_LOCK_TIMEOUT_SECONDS):
                    cached = await _read_cache(namespace, cache_dir, key_kwargs)
                    if cached is not None:
                        future.set_result(cached)
                        return cached

                    result = await func(**all_args)
                    # Write to cache unless read-only mode is enabled.
                    if not cache_read_only and isinstance(result, str) and not result.startswith(_FAILURE_PREFIXES):
                        await _write_cache(result, namespace, cache_dir, key_kwargs)
                    future.set_result(result)
                    return result
            except BaseException as exc:
                wrapped = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                future.set_exception(wrapped)
                # Consume the exception so asyncio doesn't log "Future exception was never
                # retrieved" when no follower is attached.
                future.exception()
                raise
            finally:
                # Only evict our own entry; a later call with the same key starts fresh.
                if _INFLIGHT.get(inflight_id) is future:
                    _INFLIGHT.pop(inflight_id, None)

        return wrapper

    return decorator


__all__ = ['CacheStats', 'get_cache_stats', 'http_cache', 'reset_cache_stats']
