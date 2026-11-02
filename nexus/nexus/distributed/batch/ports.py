from __future__ import annotations

from typing import Any, Protocol


class BatchControlError(RuntimeError):
    """Stable BatchControl failure exposed to execution drivers."""


class BatchControlTransientError(BatchControlError):
    """BatchControl could not complete an idempotent operation after retries."""


class BatchControlStorageError(BatchControlError):
    """BatchControl encountered a terminal storage failure."""


class BatchControl(Protocol):
    async def ensure_prepared(
        self,
        parent_execution_id: str,
        *,
        parent_attempt_id: str | None = None,
        bridge_timeout: float | None = None,
    ) -> Any: ...

    async def reconcile_once(
        self,
        parent_execution_id: str,
        *,
        max_inflight: int,
        submit_batch_size: int,
        parent_attempt_id: str | None = None,
    ) -> bool: ...

    async def stop_children(self, parent_execution_id: str, *, parent_attempt_id: str | None = None) -> None: ...

    async def release_parent(self, parent_execution_id: str) -> None: ...


class BatchControlServicePort(BatchControl, Protocol):
    async def prepare(self, parent_execution_id: str, *, parent_attempt_id: str | None = None) -> Any: ...

    async def close(self) -> None: ...


__all__ = [
    'BatchControl',
    'BatchControlError',
    'BatchControlStorageError',
    'BatchControlServicePort',
    'BatchControlTransientError',
]
