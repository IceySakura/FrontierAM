#!/usr/bin/env python3
from __future__ import annotations

import os

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar


try:
    from .toolathlon_mock_utils import mutate_store, read_store
except ImportError:
    from toolathlon_mock_utils import mutate_store, read_store


_T = TypeVar('_T')
STORE_PATH_ENV = 'NEXUS_TOOLATHLON_GOOGLE_STORE_PATH'


def case_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == 'agent_workspace':
        return cwd.parent
    return cwd


def google_store_path() -> str:
    configured = os.getenv(STORE_PATH_ENV)
    if configured:
        return configured
    return str(case_root_from_cwd() / '.nexus_google_store.json')


def default_google_state() -> dict[str, Any]:
    return {
        'version': 1,
        'next_ids': {
            'calendar_event': 1,
            'drive_file': 1,
            'form': 1,
            'spreadsheet': 1,
            'sheet': 1,
            'bigquery_dataset': 1,
            'bigquery_table': 1,
            'form_response': 1,
        },
        'calendar': {
            'events': {},
        },
        'drive': {
            'files': {},
        },
        'forms': {
            'forms': {},
            'responses': {},
        },
        'sheets': {
            'spreadsheets': {},
        },
        'bigquery': {
            'datasets': {},
        },
    }


def read_google_state() -> dict[str, Any]:
    return read_store(google_store_path(), default_google_state)


def mutate_google_state(mutator: Callable[[dict[str, Any]], _T]) -> _T:
    return mutate_store(google_store_path(), default_google_state, mutator)


def next_id(state: dict[str, Any], key: str) -> int:
    value = int(state['next_ids'].get(key, 1))
    state['next_ids'][key] = value + 1
    return value
