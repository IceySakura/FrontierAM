"""Centralised path conventions for Nexus.

All Nexus components that need temp storage should derive paths from
the helpers here, keeping a consistent ``/tmp/nexus/<namespace>/...`` layout.
Persistent local state should live under ``~/.nexus``.

Directory structure::

    /tmp/nexus/
    +-- agents/          # CLI agent installs (openclaw, codex, claude-code ...)
    +-- cache/           # build-artifact caches, download caches
    +-- runtime/         # worker venvs and other runtime-managed state
    +-- tools/           # per-tool scratch dirs (terminal, editor-history ...)

    ~/.nexus/
    +-- auth/
        +-- taihu         # Taihu (太湖) personal access token (PAT) for the artifact service (file)
        +-- cos/
            +-- key       # Tencent COS credential JSON
        +-- gongfeng/
            +-- key       # Gongfeng sandbox token
        +-- wedata/
            +-- key       # Wedata security key downloaded from wedata.woa.com
    +-- datasets/         # fetched dataset cache
"""

from __future__ import annotations

import os
import tempfile

from pathlib import Path


ENV_NEXUS_WORKDIR = 'NEXUS_WORKDIR'

NEXUS_TMP_ROOT = Path(tempfile.gettempdir()) / 'nexus'
NEXUS_HOME = Path(os.getenv('NEXUS_HOME', '~/.nexus')).expanduser()

# ── Pre-defined namespaces ──────────────────────────────────────────

AGENTS_DIR = NEXUS_TMP_ROOT / 'agents'
CACHE_DIR = NEXUS_TMP_ROOT / 'cache'
RUNTIME_DIR = NEXUS_TMP_ROOT / 'runtime'
TOOLS_DIR = NEXUS_TMP_ROOT / 'tools'

AUTH_DIR = NEXUS_HOME / 'auth'
# Taihu (太湖) PAT key is stored directly as a file, not under a subdirectory.
ARTIFACT_AUTH_KEY_FILE = AUTH_DIR / 'taihu'
COS_AUTH_DIR = AUTH_DIR / 'cos'
COS_AUTH_KEY_FILE = COS_AUTH_DIR / 'key'
GONGFENG_AUTH_DIR = AUTH_DIR / 'gongfeng'
GONGFENG_AUTH_KEY_FILE = GONGFENG_AUTH_DIR / 'key'
WEDATA_AUTH_DIR = AUTH_DIR / 'wedata'
WEDATA_AUTH_KEY_FILE = WEDATA_AUTH_DIR / 'key'
# Taihu personal access token file; used as a fallback when no Wedata key exists.
TAIHU_AUTH_TOKEN_FILE = AUTH_DIR / 'taihu-pat'

DATASET_CACHE_DIR = NEXUS_HOME / 'datasets'


def resolve_workdir_path(path: str | Path) -> Path:
    """Resolve a relative path against ``$NEXUS_WORKDIR``, falling back to the working directory."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    workdir = os.getenv(ENV_NEXUS_WORKDIR)
    return Path(workdir) / candidate if workdir else Path.cwd() / candidate
