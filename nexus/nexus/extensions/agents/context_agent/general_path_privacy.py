"""Keep host filesystem paths out of what is sent to the model.

An absolute path on a shared cluster carries the org, the project and the user
in its directory names (``/apdcephfs_.../share_.../hunyuan/<user>/...``). Those
end up in a request body two ways that are easy to miss, because neither is the
file list:

* the system prompt names the rollout's working directory, so **every** request
  carries it, on every turn;
* tool results quote real paths — a Python traceback naming the REPL server, a
  ``pwd``, ``ls -la``, a grep header, curl's saved path.

Gateways that inspect request bodies flag this. So paths are mapped to a fake
root on the way out and mapped back on the way in.

The mapping is a **directory prefix swap**, and it is bijective and stable for
the life of the process::

    /apdcephfs_x/share_y/hunyuan/me/harness/outputs/run/t1/attempt_0/workspace/doc.md
    -> /home/agent/d1/workspace/doc.md

Basenames survive, so a filename the model is asked to cite is still the real
one, and the directory shape is preserved enough that relative reasoning about
it still works.

Why the inverse is mandatory
----------------------------
The model echoes paths back. It reads one in a tool result and passes it to the
next tool call, so a one-way scrub would hand it paths that no longer resolve —
the failure would look like the agent inventing filenames. :func:`restore_obj`
runs on every tool call's arguments before dispatch, which is the one choke
point they all pass through.

This is deliberately much smaller than a general redactor: it swaps directory
prefixes and nothing else. It does not pseudonymise arbitrary words, so it
cannot corrupt document text that merely happens to contain a project name.

``CTXAGENT_SCRUB_PATHS=0`` disables it. ``CTXAGENT_SCRUB_ROOTS`` adds
colon-separated prefixes to treat as sensitive when their names alone would not
say so (a dataset mounted somewhere neutral, say).
"""

from __future__ import annotations

import os
import re
import threading

from pathlib import Path
from typing import Any


FAKE_ROOT = '/home/agent'

# A directory whose path contains one of these is rewritten. They are the
# cluster/org/user segments that make a path identifying; matching on them
# rather than on a single configured root means a dataset elsewhere on the same
# filesystem is covered without anyone remembering to register it.
SENSITIVE_SEGMENTS: tuple[str, ...] = (
    'apdcephfs',
    'share_30',
    'hunyuan',
    'changzelv',
    'taiji',
    'polaris',
    'woa.com',
)

# Absolute POSIX paths. Trailing punctuation is excluded so a path at the end of
# a sentence, or inside quotes in a traceback, is matched without it.
_PATH_RE = re.compile(r'/[A-Za-z0-9._][A-Za-z0-9._+\-/]*')


def enabled() -> bool:
    return os.environ.get('CTXAGENT_SCRUB_PATHS', '1').strip().lower() not in (
        '0',
        'off',
        'false',
        'no',
    )


def _extra_roots() -> tuple[str, ...]:
    raw = os.environ.get('CTXAGENT_SCRUB_ROOTS', '')
    return tuple(p for p in (s.strip() for s in raw.split(':')) if p)


def is_sensitive(path: str) -> bool:
    low = path.lower()
    return any(seg in low for seg in SENSITIVE_SEGMENTS) or any(path.startswith(root) for root in _extra_roots())


def _walk(obj: Any, fn) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk(v, fn) for v in obj)
    return obj


class PathPrivacy:
    """Per-rollout path mapper.

    A mapper is intentionally scoped to one rollout. Runtime staging can map a
    host path to a different sandbox path; keeping that inverse in process-wide
    globals would let concurrent rollouts overwrite one another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._real_to_fake: dict[str, str] = {}
        self._fake_to_real: dict[str, str] = {}
        self._fake_to_backend: dict[str, str] = {}

    def _fake_dir_for(self, real_dir: str) -> str:
        """Stable fake stand-in for one real directory, assigned on first sight."""
        with self._lock:
            fake = self._real_to_fake.get(real_dir)
            if fake is None:
                fake = f'{FAKE_ROOT}/d{len(self._real_to_fake) + 1}'
                self._real_to_fake[real_dir] = fake
                self._fake_to_real[fake] = real_dir
            return fake

    def scrub(self, text: str) -> str:
        """Replace sensitive directory prefixes in *text* with fake ones."""
        if not text or not enabled():
            return text

        def swap(match: re.Match[str]) -> str:
            path = match.group(0)
            if not is_sensitive(path):
                return path
            head, sep, tail = path.rpartition('/')
            if not head:
                return path
            return f'{self._fake_dir_for(head)}{sep}{tail}'

        return _PATH_RE.sub(swap, text)

    def restore(self, text: str) -> str:
        """Map fake paths to paths usable by this rollout's active backend."""
        if not text or FAKE_ROOT not in text:
            return text
        with self._lock:
            backend_pairs = sorted(self._fake_to_backend.items(), key=lambda kv: -len(kv[0]))
            real_pairs = sorted(self._fake_to_real.items(), key=lambda kv: -len(kv[0]))
        for fake, backend in backend_pairs:
            if fake in text:
                text = text.replace(fake, backend)
        for fake, real in real_pairs:
            if fake in text:
                text = text.replace(fake, real)
        return text

    def scrub_obj(self, obj: Any) -> Any:
        """Deep copy of *obj* with every string scrubbed."""
        return _walk(obj, self.scrub) if enabled() else obj

    def restore_obj(self, obj: Any) -> Any:
        """Deep copy of *obj* with every string restored. Always on."""
        return _walk(obj, self.restore)

    def register(self, *paths: str | Path) -> None:
        """Pre-assign fake names for sensitive directories."""
        if not enabled():
            return
        for path in paths:
            text = str(path)
            if text and is_sensitive(text):
                self._fake_dir_for(text.rstrip('/'))

    def register_backend_path(self, host_path: str | Path, backend_path: str | Path) -> None:
        """Restore the scrubbed *host_path* to *backend_path* for this rollout."""
        host = str(host_path)
        if not host or not is_sensitive(host):
            return
        fake = self.scrub(host)
        if fake == host:
            return
        with self._lock:
            self._fake_to_backend[fake] = str(backend_path)


_DEFAULT = PathPrivacy()


def scrub(text: str) -> str:
    return _DEFAULT.scrub(text)


def restore(text: str) -> str:
    return _DEFAULT.restore(text)


def scrub_obj(obj: Any) -> Any:
    return _DEFAULT.scrub_obj(obj)


def restore_obj(obj: Any) -> Any:
    return _DEFAULT.restore_obj(obj)


def register(*paths: str | Path) -> None:
    _DEFAULT.register(*paths)


def register_backend_path(host_path: str | Path, backend_path: str | Path) -> None:
    _DEFAULT.register_backend_path(host_path, backend_path)
