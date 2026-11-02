"""Stage files that only exist inside the runtime sandbox onto the host.

Multimodal search runs split the filesystem in two. ``code_interpreter``
(canonical ``jupyter_with_image``) is a RUNTIME tool, so its crops and zooms land
inside the sandbox, and ``mm_agent`` uploads the task's input images to the same
sandbox under ``/tmp``. The web_search tools are LOCAL: they execute in the agent
process, where none of those paths resolve. A model that passes the very path it
was handed therefore hits "file not found" for reasons it cannot see or fix.

The sandbox is the source of truth: it is consulted first, so a re-crop to a
path already seen yields the new bytes rather than a stale copy.

Two entry points, because writing every fetch to disk would leave a file behind
per trajectory and fill the agent node:

- ``read_sandbox_bytes`` returns the contents and touches no disk. Use it
  unless a path is genuinely required.
- ``stage_image_path`` materializes a copy for callers that must hand a path to
  something else (an upload helper that opens the file itself). The copy lands
  under a directory this module owns, never at the model-supplied path, and the
  caller deletes it with ``discard_staged_copy`` when done.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid


logger = logging.getLogger(__name__)

# NOTE(jaykaizhang): cap a single fetch so a stray large artifact in the sandbox
# cannot exhaust host memory during a tool call.
MAX_FETCH_BYTES = 32 * 1024 * 1024

STAGE_DIR_ENV = 'NEXUS_SANDBOX_STAGE_DIR'
_STAGE_SUBDIR = 'nexus_stage'
_ENV_FLAG = 'NEXUS_FETCH_SANDBOX_IMAGES'
_TRUE_VALUES = frozenset({'1', 'true', 'yes', 'y', 'on'})

# NOTE(jaykaizhang): a run without a run_id (offline batch mode, unit tests) still
# needs a private directory, so fall back to a per-process token rather than
# letting every such run share one name.
_PROCESS_TOKEN = f'pid{os.getpid()}-{uuid.uuid4().hex[:8]}'


def fetch_enabled() -> bool:
    """Whether sandbox-to-host staging is switched on.

    Returns:
        True when ``NEXUS_FETCH_SANDBOX_IMAGES`` is set to a truthy value.
    """
    return os.environ.get(_ENV_FLAG, '0').strip().lower() in _TRUE_VALUES


def is_remote_uri(path: str) -> bool:
    """Whether the string addresses a network or inline resource, not a file.

    Args:
        path (str): Value taken from a tool argument.

    Returns:
        True for http, https and data URIs.
    """
    return path.startswith(('http://', 'https://', 'data:'))


def _require_context():
    """Return the active run context.

    Wrapped so tests can substitute a context without reaching into
    ``nexus.core``, and imported lazily because the web_search tools must stay
    importable in runtime-free unit tests and offline batch mode.

    Raises:
        RuntimeError: When no run is active.
    """
    from nexus.core.context import require_context

    return require_context()


def _stage_dir() -> str:
    """Directory this module owns for staged copies, private to the run."""
    root = os.environ.get(STAGE_DIR_ENV) or os.path.join(tempfile.gettempdir(), _STAGE_SUBDIR)
    try:
        run_id = _require_context().run_id
    except RuntimeError:
        run_id = None
    return os.path.join(root, run_id or _PROCESS_TOKEN)


async def read_sandbox_bytes(path: str, max_bytes: int = MAX_FETCH_BYTES) -> bytes | None:
    """Return the contents of a sandbox file without touching host disk.

    For callers that only need the bytes. Staging to disk would leave a file
    behind for every trajectory, so it is reserved for the callers that
    genuinely need a path.

    Args:
        path (str): Absolute path inside the runtime sandbox.
        max_bytes (int): Reject anything larger.

    Returns:
        The file contents, or None when the sandbox has no such file, the
        transfer failed, or staging is switched off.
    """
    if not fetch_enabled() or not _is_absolute(path):
        return None
    return await _download_from_runtime(path, max_bytes)


async def stage_image_path(path: str) -> tuple[str, bool] | None:
    """Return a host-readable path for ``path``, preferring the run's sandbox.

    For callers that need a real path on disk, such as an upload helper that
    opens the file itself. A staged copy is the caller's to delete once it is
    done, which is what the second element of the result is for.

    Args:
        path (str): Absolute path as supplied by the model.

    Returns:
        A ``(path, is_staged_copy)`` pair, or None when the file is nowhere to
        be found. The path may differ from ``path``; callers must open the
        returned value and keep ``path`` only for cache keys and model-facing
        text. When ``is_staged_copy`` is True the caller owns that file and must
        call ``discard_staged_copy`` on it; when False the path is a host file
        that was already there and must be left alone.
    """
    if not _is_absolute(path):
        return None
    if fetch_enabled():
        raw = await _download_from_runtime(path, MAX_FETCH_BYTES)
        if raw is not None:
            return _write_staged_copy(path, raw), True
    return (path, False) if os.path.isfile(path) else None


def discard_staged_copy(path: str) -> None:
    """Delete a copy created by ``stage_image_path``.

    Args:
        path (str): The staged path, which must sit under this module's
            staging directory so a host file can never be removed by mistake.
    """
    stage_root = os.environ.get(STAGE_DIR_ENV) or os.path.join(tempfile.gettempdir(), _STAGE_SUBDIR)
    if os.path.commonpath((os.path.abspath(path), os.path.abspath(stage_root))) != os.path.abspath(stage_root):
        # NOTE(jaykaizhang): refuse anything outside the staging root. The only
        # way to get here is a caller passing a path it did not stage, and
        # deleting a model-supplied path would destroy the task's input image.
        logger.warning('[SandboxFetch] Refusing to delete a path outside the staging dir: %s', path)
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning('[SandboxFetch] Could not delete staged copy %s: %s', path, exc)


def _is_absolute(path: str) -> bool:
    """Whether the path is absolute, logging the reason when it is not."""
    if os.path.isabs(path):
        return True
    # NOTE(jaykaizhang): the sandbox resolves a relative path against
    # NEXUS_WORKDIR while the host would resolve it against the agent process
    # cwd, so the two sides would disagree about the file.
    logger.warning('[SandboxFetch] Refusing relative path, pass an absolute one: %s', path)
    return False


async def _download_from_runtime(path: str, max_bytes: int) -> bytes | None:
    """Pull one file out of the run's sandbox.

    Args:
        path (str): Absolute path inside the runtime sandbox.
        max_bytes (int): Reject anything larger.

    Returns:
        The file contents, or None when the sandbox has no such file and when
        the transfer failed.
    """
    try:
        from nexus.runtime.executors.models.file import FileItem

        try:
            ctx = _require_context()
        except RuntimeError:
            return None  # No active run, e.g. unit tests. Expected, stay quiet.
        if ctx.runtime_provider is None:
            return None  # Search-only run without a sandbox, nothing to fetch.

        runtime = await ctx.get_runtime()
        response = await runtime.download_file([FileItem(path=path, encoding='base64')])
        for item in response.files:
            if item.content is None:
                continue
            # NOTE(jaykaizhang): the cap is checked against the encoded payload
            # because decoding is what materializes the bytes in host memory.
            if len(item.content) * 3 // 4 > max_bytes:
                logger.warning('[SandboxFetch] File too large to fetch: %s', path)
                return None
            return base64.b64decode(item.content)

        # NOTE(jaykaizhang): "File does not exist" is the ordinary answer for a
        # host-only input such as a shared-storage image, so it must not look
        # like a transport failure to whoever reads these logs.
        for failure in response.errors or ():
            if failure.message == 'File does not exist':
                logger.debug('[SandboxFetch] Not in sandbox, will try the host: %s', path)
            else:
                logger.warning('[SandboxFetch] Sandbox refused %s: %s', path, failure.message)
    except Exception:
        # NOTE(jaykaizhang): a transport hiccup must not turn into a hard tool
        # failure. Callers fall back to their original "file not found" path.
        logger.warning('[SandboxFetch] Failed to stage %r from sandbox', path, exc_info=True)
    return None


def _write_staged_copy(path: str, raw: bytes) -> str:
    """Write sandbox bytes to a private path under the staging directory.

    Args:
        path (str): Original sandbox path, used only for its extension.
        raw (bytes): File contents pulled from the sandbox.

    Returns:
        The host path holding ``raw``.
    """
    stage_dir = _stage_dir()
    os.makedirs(stage_dir, exist_ok=True)
    # NOTE(jaykaizhang): a unique name per call, not a content hash. Each copy
    # belongs to one caller that deletes it when done, so two calls sharing a
    # name by having equal contents would let the first finisher pull the file
    # out from under the second.
    staged_path = os.path.join(stage_dir, uuid.uuid4().hex + os.path.splitext(path)[1])
    with open(staged_path, 'wb') as staged_file:
        staged_file.write(raw)
    logger.debug('[SandboxFetch] Staged sandbox file on host: %s -> %s (%d bytes)', path, staged_path, len(raw))
    return staged_path


__all__ = [
    'MAX_FETCH_BYTES',
    'STAGE_DIR_ENV',
    'discard_staged_copy',
    'fetch_enabled',
    'is_remote_uri',
    'read_sandbox_bytes',
    'stage_image_path',
]
