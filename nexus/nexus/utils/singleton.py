from __future__ import annotations

import asyncio

from logging import getLogger
from typing import Any, Generic, TypeVar, cast


T = TypeVar('T')
logger = getLogger(__name__)


class Singleton(Generic[T]):  # noqa
    _instance: T | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    async def get_instance_async(cls, *args, **kwargs):
        if not cls._instance:
            if not cls._lock:
                cls._lock = asyncio.Lock()
            async with cls._lock:
                if not cls._instance:
                    self = cls(*args, **kwargs)
                    assert hasattr(self, 'async_init'), 'async singletons must define async_init function'
                    await cast(Any, self).async_init()
                    cls._instance = cast(T, self)
                    logger.debug(f'singleton class {cls.__name__} initialized')
        return cls._instance

    @classmethod
    def get_instance_sync(cls, *args, **kwargs):
        if not cls._instance:
            self = cls(*args, **kwargs)
            assert not hasattr(self, 'async_init'), (
                f'class {cls.__name__} has async_init function, init it with get_instance_async.'
            )
            cls._instance = cast(T, self)
            logger.debug(f'singleton class {cls.__name__} initialized')
        return cls._instance
