#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import runpy
import sys
import types

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar


_T = TypeVar('_T')


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lock_path(store_path: Path) -> Path:
    return store_path.with_suffix(store_path.suffix + '.lock')


@contextlib.contextmanager
def _file_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_store(store_path: str | Path, default_factory: Callable[[], _T]) -> _T:
    path = Path(store_path)
    with _file_lock(_lock_path(path)):
        if not path.exists():
            default = default_factory()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding='utf-8')
            return default
        return json.loads(path.read_text(encoding='utf-8'))


def mutate_store(store_path: str | Path, default_factory: Callable[[], _T], mutator: Callable[[_T], Any]) -> Any:
    path = Path(store_path)
    with _file_lock(_lock_path(path)):
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else default_factory()
        result = mutator(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


class _CompatDict(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _ensure_addict_compat() -> None:
    if 'addict' in sys.modules:
        return
    module = types.ModuleType('addict')
    module.Dict = _CompatDict
    sys.modules['addict'] = module


def _read_token_namespace(path: Path) -> dict[str, Any]:
    old_cwd = os.getcwd()
    try:
        os.chdir(path.parent)
        _ensure_addict_compat()
        namespace = runpy.run_path(str(path.name))
    finally:
        os.chdir(old_cwd)
    raw = namespace.get('all_token_key_session', {})
    try:
        payload = dict(raw)
    except Exception:
        payload = {}
    for key, value in namespace.items():
        if key.startswith('_') or key == 'all_token_key_session':
            continue
        payload.setdefault(key, value)
    return payload


def load_case_token_values() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    raw_env = os.getenv('NEXUS_TOOLATHLON_TOKEN_VALUES_JSON')
    if raw_env:
        try:
            payload = json.loads(raw_env)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            merged.update(payload)

    case_root_raw = os.getenv('NEXUS_TOOLATHLON_CASE_ROOT')
    if not case_root_raw:
        return merged
    case_root = Path(case_root_raw)
    candidates = [case_root / 'token_key_session.py', case_root / 'other_key.py']
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            merged.update(_read_token_namespace(candidate))
        except Exception:
            continue
    return merged


def slugify(text: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in text)
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    return cleaned.strip('-') or 'item'
