"""
全局速率限制器模块。

提供基于 asyncio.Semaphore 的并发控制，用于限制对外部服务的请求速率。

使用方式：
    from nexus.utils.rate_limiter import get_rate_limiter

    # 获取或创建限制器（默认并发数 120）
    limiter = get_rate_limiter('search', max_concurrent=120)

    async with limiter:
        await make_request()
"""

import asyncio
import logging
import math
import threading
import time


logger = logging.getLogger(__name__)


class RateLimiter:
    """基于 Semaphore 的异步速率限制器。"""

    def __init__(self, name: str, max_concurrent: int = 120):
        """
        Args:
            name: 限制器名称（用于日志）
            max_concurrent: 最大并发数
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_count = 0
        self._lock = asyncio.Lock()
        logger.debug(f'[RateLimiter:{name}] Initialized with max_concurrent={max_concurrent}')

    async def __aenter__(self):
        await self._semaphore.acquire()
        async with self._lock:
            self._current_count += 1
            if self._current_count % 10 == 0:  # 每 10 个请求打印一次
                logger.debug(f'[RateLimiter:{self.name}] Active: {self._current_count}/{self.max_concurrent}')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._current_count -= 1
        self._semaphore.release()
        return False

    @property
    def current_count(self) -> int:
        """当前活跃请求数。"""
        return self._current_count


# 全局限制器注册表
_rate_limiters: dict[str, RateLimiter] = {}
_registry_lock = asyncio.Lock()


def get_rate_limiter(name: str, max_concurrent: int = 120) -> RateLimiter:
    """
    获取或创建指定名称的速率限制器。

    注意：此函数是同步的，但返回的 RateLimiter 用于异步上下文。
    如果限制器已存在，max_concurrent 参数会被忽略。

    Args:
        name: 限制器名称
        max_concurrent: 最大并发数（仅在首次创建时生效）

    Returns:
        RateLimiter 实例
    """
    global _rate_limiters

    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(name, max_concurrent)

    return _rate_limiters[name]


def reset_rate_limiter(name: str) -> None:
    """重置指定名称的速率限制器。"""
    global _rate_limiters
    if name in _rate_limiters:
        del _rate_limiters[name]
        logger.debug(f'[RateLimiter:{name}] Reset')


def reset_all_rate_limiters() -> None:
    """重置所有速率限制器。"""
    global _rate_limiters
    _rate_limiters.clear()
    logger.debug('[RateLimiter] All limiters reset')


class QpsLimiter:
    """Pace requests at a fixed rate without bursts across event loops.

    Waiters compete for the next available issue timestamp under a threading
    lock, then wait in their own event loop. Slots are only consumed when an
    acquisition succeeds, so cancelled waiters do not create scheduling debt.
    """

    def __init__(self, name: str, qps: float):
        qps = float(qps)
        if not math.isfinite(qps) or qps <= 0:
            raise ValueError(f'QpsLimiter[{name}] qps must be a finite positive number, got {qps}')
        interval = 1.0 / qps
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(f'QpsLimiter[{name}] interval must be a finite positive number, got {interval}')
        self.name = name
        self.qps = qps
        self._interval = interval
        self._last_issued_at: float | None = None
        self._lock = threading.Lock()
        self._waiting_count = 0
        self._issued_count = 0
        self._last_delay_ms = 0.0

    def _register_waiter(self) -> None:
        with self._lock:
            self._waiting_count += 1

    def _try_acquire(self, started_at: float) -> tuple[float | None, float]:
        with self._lock:
            issued_at = time.monotonic()
            if self._last_issued_at is not None:
                retry_at = self._last_issued_at + self._interval
                if issued_at < retry_at:
                    return None, retry_at

            self._waiting_count -= 1
            self._last_issued_at = issued_at
            actual_delay_ms = (issued_at - started_at) * 1000.0
            self._issued_count += 1
            self._last_delay_ms = actual_delay_ms
            return actual_delay_ms, issued_at

    def _unregister_waiter(self) -> None:
        with self._lock:
            self._waiting_count -= 1

    async def acquire(self) -> float:
        """Wait for and issue the next slot, returning actual wait time in milliseconds."""
        started_at = time.monotonic()
        self._register_waiter()
        registered = True
        try:
            while True:
                actual_delay_ms, retry_at = self._try_acquire(started_at)
                if actual_delay_ms is not None:
                    registered = False
                    return actual_delay_ms

                await asyncio.sleep(max(0.0, retry_at - time.monotonic()))
        except BaseException:
            if registered:
                self._unregister_waiter()
            raise

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def snapshot(self) -> dict[str, float | int | str]:
        """Return pacing counters for diagnostics."""
        with self._lock:
            next_delay = (
                0.0
                if self._last_issued_at is None
                else max(0.0, self._last_issued_at + self._interval - time.monotonic())
            )
            return {
                'name': self.name,
                'qps': self.qps,
                'interval_ms': self._interval * 1000.0,
                'waiting': self._waiting_count,
                'issued': self._issued_count,
                'next_delay_ms': next_delay * 1000.0,
                'last_delay_ms': self._last_delay_ms,
            }


_qps_limiters: dict[str, QpsLimiter] = {}
_qps_registry_lock = threading.Lock()


def get_qps_limiter(name: str, qps: float) -> QpsLimiter:
    """Get a named QPS limiter, rejecting conflicting configurations."""
    with _qps_registry_lock:
        existing = _qps_limiters.get(name)
        if existing is None:
            limiter = QpsLimiter(name, qps)
            _qps_limiters[name] = limiter
            return limiter
        if existing.qps != qps:
            raise ValueError(f'QpsLimiter[{name}] already configured with qps={existing.qps}, got {qps}')
        return existing


def reset_qps_limiter(name: str) -> None:
    """Remove a limiter during setup or teardown when no requests are active."""
    with _qps_registry_lock:
        _qps_limiters.pop(name, None)


def reset_all_qps_limiters() -> None:
    """Remove all limiters during setup or teardown when no requests are active."""
    with _qps_registry_lock:
        _qps_limiters.clear()
