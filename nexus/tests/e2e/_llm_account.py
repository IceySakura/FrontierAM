from __future__ import annotations

import json
import sqlite3

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class AccountLookupError(RuntimeError):
    """Raised when a usable model account cannot be loaded from a batch database."""


@dataclass(frozen=True)
class SQLiteModelAccount:
    run_id: str
    api_key: str = field(repr=False)
    base_url: str
    source_client: str | None = None
    source_model: str | None = None

    @property
    def credential(self) -> str:
        return self.api_key.partition('?')[0]


def _account_from_config(run_id: str, config_json: str) -> SQLiteModelAccount | None:
    try:
        config = json.loads(config_json)
        client = config['openai_clients']['main']
        client_config = client['config']
        client_args = client_config['client_args']
        api_key = client_args['api_key']
        base_url = client_args['base_url']
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if not isinstance(api_key, str) or not api_key or api_key.startswith('${oc.env:'):
        return None
    if not isinstance(base_url, str) or not base_url:
        return None
    try:
        parsed_url = urlsplit(base_url)
    except ValueError:
        return None
    if parsed_url.scheme not in {'http', 'https'} or not parsed_url.netloc:
        return None

    request_args = client_config.get('request_args')
    source_model: Any = request_args.get('model') if isinstance(request_args, dict) else None
    source_client = client.get('name')
    return SQLiteModelAccount(
        run_id=run_id,
        api_key=api_key,
        base_url=base_url.rstrip('/'),
        source_client=source_client if isinstance(source_client, str) else None,
        source_model=source_model if isinstance(source_model, str) else None,
    )


def load_sqlite_model_account(db_path: str | Path, run_id: str | None = None) -> SQLiteModelAccount:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise AccountLookupError(f'model account SQLite database does not exist: {path}')

    try:
        connection = sqlite3.connect(f'{path.as_uri()}?mode=ro&immutable=1', uri=True)
        try:
            if run_id is not None:
                rows = connection.execute('select run_id, config from runs where run_id = ?', (run_id,))
            else:
                rows = connection.execute('select run_id, config from runs order by rowid')

            for candidate_run_id, config_json in rows:
                if not isinstance(candidate_run_id, str) or not isinstance(config_json, str):
                    continue
                account = _account_from_config(candidate_run_id, config_json)
                if account is not None:
                    return account
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise AccountLookupError(f'cannot read model account SQLite database: {path}') from error

    selector = f' for run {run_id}' if run_id is not None else ''
    raise AccountLookupError(f'no resolved model account found in SQLite database {path}{selector}')


__all__ = ['AccountLookupError', 'SQLiteModelAccount', 'load_sqlite_model_account']
