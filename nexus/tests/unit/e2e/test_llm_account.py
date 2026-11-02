from __future__ import annotations

import json
import sqlite3

from pathlib import Path

import pytest

from tests.e2e._llm_account import AccountLookupError, load_sqlite_model_account


def _write_account_db(path: Path, rows: list[tuple[str, dict]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute('create table runs (run_id text primary key, config text)')
        connection.executemany(
            'insert into runs (run_id, config) values (?, ?)',
            [(run_id, json.dumps(config)) for run_id, config in rows],
        )
        connection.commit()
    finally:
        connection.close()


def _config(*, api_key: str, base_url: str = 'https://gateway.example.test') -> dict:
    return {
        'openai_clients': {
            'main': {
                'name': 'vertex_anthropic',
                'config': {
                    'client_args': {'api_key': api_key, 'base_url': base_url},
                    'request_args': {'model': 'claude-fable-5'},
                },
            }
        }
    }


def test_load_sqlite_model_account_reads_exact_run_without_exposing_key_in_repr(tmp_path: Path) -> None:
    db_path = tmp_path / 'accounts.db'
    _write_account_db(
        db_path,
        [('run-one', _config(api_key='app-id:app-secret?provider=google&model=claude-fable-5'))],
    )

    account = load_sqlite_model_account(db_path, run_id='run-one')

    assert account.api_key == 'app-id:app-secret?provider=google&model=claude-fable-5'
    assert account.credential == 'app-id:app-secret'
    assert account.base_url == 'https://gateway.example.test'
    assert account.source_client == 'vertex_anthropic'
    assert account.source_model == 'claude-fable-5'
    assert 'app-secret' not in repr(account)


def test_load_sqlite_model_account_skips_env_placeholders_when_run_is_not_selected(tmp_path: Path) -> None:
    db_path = tmp_path / 'accounts.db'
    _write_account_db(
        db_path,
        [
            ('placeholder', _config(api_key='${oc.env:MODEL_API_KEY}')),
            ('resolved', _config(api_key='resolved-id:resolved-secret')),
        ],
    )

    account = load_sqlite_model_account(db_path)

    assert account.run_id == 'resolved'
    assert account.credential == 'resolved-id:resolved-secret'


def test_load_sqlite_model_account_error_does_not_include_secret_config(tmp_path: Path) -> None:
    db_path = tmp_path / 'accounts.db'
    secret = 'do-not-print-this-secret'
    _write_account_db(db_path, [('broken', _config(api_key=secret, base_url='not-a-url'))])

    with pytest.raises(AccountLookupError) as exc_info:
        load_sqlite_model_account(db_path, run_id='broken')

    assert secret not in str(exc_info.value)
    assert 'broken' in str(exc_info.value)


def test_load_sqlite_model_account_normalizes_malformed_url_error(tmp_path: Path) -> None:
    db_path = tmp_path / 'accounts.db'
    _write_account_db(db_path, [('broken-url', _config(api_key='secret', base_url='http://[broken'))])

    with pytest.raises(AccountLookupError, match='broken-url'):
        load_sqlite_model_account(db_path, run_id='broken-url')
