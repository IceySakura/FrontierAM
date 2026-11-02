#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


try:
    from .toolathlon_k8s_state import (
        K8sStateError,
        apply_documents,
        delete_resource,
        get_helm_values,
        get_resource,
        helm_install_or_upgrade,
        k8s_root_path,
        list_helm_releases,
        list_resources,
        parse_yaml_documents,
        patch_resource_json,
        rollout_ready,
        scale_workload,
        wait_pods_ready,
    )
except ImportError:
    from toolathlon_k8s_state import (
        K8sStateError,
        apply_documents,
        delete_resource,
        get_helm_values,
        get_resource,
        helm_install_or_upgrade,
        k8s_root_path,
        list_helm_releases,
        list_resources,
        parse_yaml_documents,
        patch_resource_json,
        rollout_ready,
        scale_workload,
        wait_pods_ready,
    )


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _require_text(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _error(INVALID_PARAMS, f'Missing required text field: {key}')
    return value.strip()


def _optional_text(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise _error(INVALID_PARAMS, f'Invalid integer for {key}: {value}') from exc


def _spawn_port_forward(namespace: str, resource: str, local_port: int, remote_port: int) -> dict[str, Any]:
    logs_dir = k8s_root_path() / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f'port-forward-{namespace}-{resource.replace("/", "-")}-{local_port}.log'
    cli_path = Path(__file__).with_name('toolathlon_k8s_cli.py')
    process = subprocess.Popen(
        [
            sys.executable,
            str(cli_path),
            'kubectl',
            '-n',
            namespace,
            'port-forward',
            resource,
            f'{local_port}:{remote_port}',
        ],
        stdout=log_path.open('ab'),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        'pid': process.pid,
        'namespace': namespace,
        'resource': resource,
        'local_port': local_port,
        'remote_port': remote_port,
        'log_path': str(log_path),
    }


async def main() -> None:
    server = Server('toolathlon-k8s-server')

    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
        'list_namespaces': lambda arguments: list_resources('namespace'),
        'list_resources': lambda arguments: list_resources(
            _require_text(arguments, 'kind'),
            namespace=_optional_text(arguments, 'namespace'),
            label_selector=_optional_text(arguments, 'label_selector'),
        ),
        'get_resource': lambda arguments: get_resource(
            _require_text(arguments, 'kind'),
            _require_text(arguments, 'name'),
            namespace=_optional_text(arguments, 'namespace'),
        ),
        'apply_manifest': lambda arguments: {
            'applied': apply_documents(parse_yaml_documents(_require_text(arguments, 'manifest')))
        },
        'delete_resource': lambda arguments: (
            delete_resource(
                _require_text(arguments, 'kind'),
                _require_text(arguments, 'name'),
                namespace=_optional_text(arguments, 'namespace'),
            )
            or {'deleted': True}
        ),
        'scale_resource': lambda arguments: (
            scale_workload(
                _require_text(arguments, 'kind'),
                _require_text(arguments, 'name'),
                _optional_int(arguments, 'replicas') or 0,
                namespace=_optional_text(arguments, 'namespace'),
            )
            or {'scaled': True}
        ),
        'patch_resource': lambda arguments: patch_resource_json(
            _require_text(arguments, 'kind'),
            _require_text(arguments, 'name'),
            arguments.get('patch')
            if isinstance(arguments.get('patch'), list)
            else json.loads(_require_text(arguments, 'patch')),
            namespace=_optional_text(arguments, 'namespace'),
        ),
        'rollout_status': lambda arguments: {
            'ready': rollout_ready(
                _require_text(arguments, 'kind'),
                _require_text(arguments, 'name'),
                namespace=_optional_text(arguments, 'namespace'),
            )
        },
        'wait_ready': lambda arguments: {
            'ready': wait_pods_ready(
                namespace=_optional_text(arguments, 'namespace'),
                name=_optional_text(arguments, 'name'),
                label_selector=_optional_text(arguments, 'label_selector'),
            )
        },
        'port_forward': lambda arguments: _spawn_port_forward(
            _require_text(arguments, 'namespace'),
            _require_text(arguments, 'resource'),
            _optional_int(arguments, 'local_port') or 0,
            _optional_int(arguments, 'remote_port') or 0,
        ),
        'helm_list_releases': lambda arguments: list_helm_releases(namespace=_optional_text(arguments, 'namespace')),
        'helm_get_values': lambda arguments: get_helm_values(
            _require_text(arguments, 'release_name'),
            namespace=_optional_text(arguments, 'namespace'),
        ),
        'helm_upgrade_release': lambda arguments: helm_install_or_upgrade(
            _require_text(arguments, 'release_name'),
            _require_text(arguments, 'chart_ref'),
            namespace=_require_text(arguments, 'namespace'),
            version=_require_text(arguments, 'version'),
            values=arguments.get('values') if isinstance(arguments.get('values'), dict) else {},
            reuse_values=bool(arguments.get('reuse_values')),
        ),
    }

    descriptions = {
        'list_namespaces': 'List all namespaces in the current mock Kubernetes cluster.',
        'list_resources': 'List Kubernetes resources of a specific kind, optionally filtered by namespace or label selector.',
        'get_resource': 'Get one Kubernetes resource as structured JSON.',
        'apply_manifest': 'Apply one or more Kubernetes YAML documents to the mock cluster.',
        'delete_resource': 'Delete one Kubernetes resource from the mock cluster.',
        'scale_resource': 'Scale a deployment or statefulset to the requested replica count.',
        'patch_resource': 'Apply a JSON patch payload to one Kubernetes resource.',
        'rollout_status': 'Check whether a deployment or statefulset is fully ready.',
        'wait_ready': 'Check whether pods are ready in a namespace, by name, or by label selector.',
        'port_forward': 'Start a background kubectl-style port-forward process and keep it running.',
        'helm_list_releases': 'List Helm releases known to the mock cluster.',
        'helm_get_values': 'Get the stored values of one Helm release.',
        'helm_upgrade_release': 'Install or upgrade a Helm release in the mock cluster.',
    }

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=descriptions.get(name, name),
                inputSchema={'type': 'object', 'properties': {}, 'additionalProperties': True},
            )
            for name in tool_handlers
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = tool_handlers.get(name)
        if handler is None:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        try:
            return _json_content(handler(arguments or {}))
        except K8sStateError as exc:
            raise _error(INVALID_PARAMS, str(exc)) from exc
        except McpError:
            raise
        except Exception as exc:
            raise _error(INTERNAL_ERROR, str(exc)) from exc

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
