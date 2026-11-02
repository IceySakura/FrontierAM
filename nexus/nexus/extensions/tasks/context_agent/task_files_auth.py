"""Context Agent-specific authentication override for remote TaskFiles."""

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr

from nexus.dataset.auth import WedataKey
from nexus.dataset.task_files import (
    RemoteZipTaskFiles,
    TaskFiles,
    TaskFilesSession,
    _extract_signed_url,
    _sign_remote_task_files,
    _wedata_key_auth_payload,
)


class _TaskScopedRemoteZipTaskFiles(RemoteZipTaskFiles):
    """Remote TaskFiles bound to a credential supplied by this task."""

    _task_wedata_key: WedataKey = PrivateAttr()

    @classmethod
    def from_existing(cls, files: RemoteZipTaskFiles, key: WedataKey) -> _TaskScopedRemoteZipTaskFiles:
        scoped = cls(spec=files.spec, run_id=files.run_id)
        scoped._task_wedata_key = key
        return scoped

    async def create_session(self, *, run_id: str | None = None) -> TaskFilesSession:
        return TaskFilesSession(
            run_id=run_id or self.run_id,
            task_files=self.spec,
            auth=_wedata_key_auth_payload(self._task_wedata_key),
        )

    async def _get_signed_url(self) -> str:
        if self._signed_url is not None:
            return self._signed_url
        signed = await _sign_remote_task_files(
            self.spec,
            auth=_wedata_key_auth_payload(self._task_wedata_key),
        )
        self._signed_url = _extract_signed_url(signed, self.spec.path)
        return self._signed_url


def bind_task_wedata_key(
    task_files: TaskFiles | None,
    raw_key: str | dict[str, Any] | None,
) -> TaskFiles | None:
    """Bind ``task_config.WEDATA_KEY`` to remote files, otherwise preserve fallback behavior."""
    if raw_key is None or raw_key == '' or not isinstance(task_files, RemoteZipTaskFiles):
        return task_files
    if isinstance(raw_key, str):
        key = WedataKey.from_raw(raw_key, source='task_config.WEDATA_KEY')
    elif isinstance(raw_key, dict):
        key = WedataKey.from_mapping(raw_key, source='task_config.WEDATA_KEY')
    else:
        raise TypeError('task_config.WEDATA_KEY must be a Wedata key JSON string or object')
    return _TaskScopedRemoteZipTaskFiles.from_existing(task_files, key)


__all__ = ['bind_task_wedata_key']
