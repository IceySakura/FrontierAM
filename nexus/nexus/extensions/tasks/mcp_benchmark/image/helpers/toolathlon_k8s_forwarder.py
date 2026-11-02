#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sqlite3

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mysql_mimic import MysqlServer, Session
from mysql_mimic.auth import IdentityProvider, NativePasswordAuthPlugin, User
from mysql_mimic.errors import ErrorCode, MysqlError


try:
    from .toolathlon_k8s_sql import execute_query, schema_mapping
    from .toolathlon_k8s_state import ROOT_PATH_ENV, STORE_PATH_ENV, mysql_db_path, mysql_users, resolve_service_content
except ImportError:
    from toolathlon_k8s_sql import execute_query, schema_mapping
    from toolathlon_k8s_state import ROOT_PATH_ENV, STORE_PATH_ENV, mysql_db_path, mysql_users, resolve_service_content


class _HttpHandler(BaseHTTPRequestHandler):
    cluster_name: str = ''
    namespace: str = ''
    service_name: str = ''

    def do_GET(self) -> None:  # noqa: N802
        body, content_type = resolve_service_content(
            self.namespace,
            self.service_name,
            cluster_name=self.cluster_name,
        )
        payload = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self) -> None:  # noqa: N802
        body, content_type = resolve_service_content(
            self.namespace,
            self.service_name,
            cluster_name=self.cluster_name,
        )
        payload = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()

    def log_message(self, message_format: str, *args: Any) -> None:
        del message_format, args
        return


class _StateIdentityProvider(IdentityProvider):
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name

    async def get_user(self, username: str) -> User | None:
        users = mysql_users(cluster_name=self.cluster_name)
        payload = users.get(username)
        if payload is None:
            return None
        password = str(payload.get('password') or '')
        return User(
            name=username,
            auth_plugin=NativePasswordAuthPlugin.name,
            auth_string=NativePasswordAuthPlugin.create_auth_string(password),
        )


class _SQLiteMysqlSession(Session):
    def __init__(self, cluster_name: str):
        super().__init__()
        self.cluster_name = cluster_name

    @property
    def db_path(self) -> str:
        return mysql_db_path(cluster_name=self.cluster_name)

    async def query(self, expression, sql: str, attrs):  # type: ignore[override]
        try:
            column_names, rows = execute_query(self.db_path, sql)
            return rows, column_names
        except sqlite3.Error as exc:
            raise MysqlError(str(exc), code=ErrorCode.UNKNOWN_ERROR) from exc

    async def schema(self) -> dict[str, dict[str, str]]:  # type: ignore[override]
        return {'f1': schema_mapping(self.db_path)}


async def _serve_mysql(port: int, cluster_name: str) -> None:
    server = MysqlServer(
        session_factory=lambda: _SQLiteMysqlSession(cluster_name),
        identity_provider=_StateIdentityProvider(cluster_name),
        host='127.0.0.1',
        port=port,
    )
    await server.serve_forever()


def _serve_http(port: int, cluster_name: str, namespace: str, service_name: str) -> None:
    handler = type(
        'ToolathlonK8sHttpHandler',
        (_HttpHandler,),
        {
            'cluster_name': cluster_name,
            'namespace': namespace,
            'service_name': service_name,
        },
    )
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-path')
    parser.add_argument('--root-path')
    parser.add_argument('--cluster', required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--service', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--protocol', choices=('http', 'mysql'), required=True)
    args = parser.parse_args()

    if args.state_path:
        import os

        os.environ[STORE_PATH_ENV] = args.state_path
    if args.root_path:
        import os

        os.environ[ROOT_PATH_ENV] = args.root_path

    if args.protocol == 'http':
        _serve_http(args.port, args.cluster, args.namespace, args.service)
        return

    asyncio.run(_serve_mysql(args.port, args.cluster))


if __name__ == '__main__':
    main()
