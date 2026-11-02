from __future__ import annotations

import asyncio
import json
import logging
import re

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests

from .types import BenchmarkCleanup


logger = logging.getLogger(__name__)


def _tool_call_return_value(tool_call: dict[str, Any]) -> Any:
    value = tool_call.get('response')
    if isinstance(value, list):
        return {
            'content': value,
            'is_error': bool(tool_call.get('is_error', False)),
        }
    return value


def _cleanup_arg_value(cleanup_args: Any, tool_call: dict[str, Any] | None) -> Any:
    if isinstance(cleanup_args, dict):
        return {key: _cleanup_arg_value(value, tool_call) for key, value in cleanup_args.items()}
    if isinstance(cleanup_args, list):
        return [_cleanup_arg_value(value, tool_call) for value in cleanup_args]
    if not isinstance(cleanup_args, str) or tool_call is None or '$' not in cleanup_args:
        return cleanup_args

    arg = cleanup_args.strip()
    if not arg.startswith('$'):
        raise ValueError(f'Invalid cleanup arg expression: {cleanup_args}')

    steps = [item.strip() for item in arg.split('->') if item.strip()]
    source = steps[0][1:]
    if source == 'return':
        value: Any = _tool_call_return_value(tool_call)
    else:
        arguments = tool_call.get('arguments', {})
        if source not in arguments:
            raise ValueError(f'Invalid cleanup arg expression source: {source}')
        value = arguments[source]

    for step in steps[1:]:
        name = step.split('(')[0].strip().lower()
        match = re.search(r'\((.*?)\)', step)
        step_arg = match.group(1).strip() if match else ''
        if name == 'get':
            value = value[step_arg]
        elif name == 'array':
            value = value[int(step_arg)]
        elif name == 'json':
            value = json.loads(value)
        else:
            raise ValueError(f'Unsupported cleanup op: {name}')
    return value


async def _github_delete_repository(*, env: dict[str, str], repo: str, owner: str = '', timeout: int = 30) -> str:
    owner = owner or env.get('GITHUB_PERSONAL_ACCOUNT_NAME', '')
    token = env.get('GITHUB_PERSONAL_ACCESS_TOKEN') or env.get('GITHUB_TOKENS', '')
    if not owner or not token:
        raise ValueError('missing GitHub owner/token for cleanup')
    response = requests.delete(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'nexus-mcp-benchmark',
        },
        timeout=timeout,
    )
    if response.status_code in {204, 404}:
        return f'github repository cleanup finished for {owner}/{repo}'
    raise RuntimeError(f'github cleanup failed: {response.status_code} {response.text[:1000]}')


async def _notion_delete_page(*, env: dict[str, str], page: str, timeout: int = 30) -> str:
    store_path = env.get('NEXUS_MCP_BENCHMARK_NOTION_STORE_PATH')
    if store_path:
        path = Path(store_path)
        if not path.exists():
            return f'notion page cleanup finished for {page}'
        state = json.loads(path.read_text(encoding='utf-8'))
        pages = state.get('pages', {})
        if page in pages:
            pages[page]['in_trash'] = True
            pages[page]['archived'] = True
            path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        return f'notion page cleanup finished for {page}'
    token = env.get('NOTION_API_KEY') or env.get('EVAL_NOTION_API_KEY', '')
    if not token:
        raise ValueError('missing Notion token for cleanup')
    response = requests.patch(
        f'https://api.notion.com/v1/pages/{page}',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Notion-Version': '2022-06-28',
        },
        data='{"in_trash": true}',
        timeout=timeout,
    )
    if response.status_code in {200, 404}:
        return f'notion page cleanup finished for {page}'
    raise RuntimeError(f'notion cleanup failed: {response.status_code} {response.text[:1000]}')


async def _blender_clear_scene_and_polyhaven_materials(*, env: dict[str, str], **_kwargs: Any) -> str:
    _ = env
    return 'blender cleanup finished'


async def run_cleanups(
    *,
    cleanups: Sequence[BenchmarkCleanup],
    tool_calls: Sequence[dict[str, Any]],
    env: dict[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    ordered_calls = list(reversed(tool_calls))

    for cleanup in cleanups:
        matching_calls = [
            call
            for call in ordered_calls
            if call.get('server') == cleanup.server and call.get('tool_name') == cleanup.tool
        ]
        if cleanup.tool == '':
            matching_calls = [None]
        elif not matching_calls:
            results.append({
                'server': cleanup.server,
                'tool': cleanup.tool,
                'cleanup_func': cleanup.cleanup_func,
                'status': 'skipped',
                'reason': 'no matching tool call',
            })
            continue

        for tool_call in matching_calls:
            try:
                kwargs = _cleanup_arg_value(cleanup.cleanup_args, tool_call)
                if cleanup.server == 'github' and cleanup.cleanup_func == 'delete_repository':
                    message = await _github_delete_repository(env=env, **kwargs)
                elif cleanup.server == 'notion' and cleanup.cleanup_func == 'delete_page':
                    message = await _notion_delete_page(env=env, **kwargs)
                elif cleanup.server == 'blender' and cleanup.cleanup_func == 'clear_scene_and_polyhaven_materials':
                    message = await _blender_clear_scene_and_polyhaven_materials(env=env, **kwargs)
                else:
                    raise ValueError(f'Unsupported cleanup handler {cleanup.server}/{cleanup.cleanup_func}')
                results.append({
                    'server': cleanup.server,
                    'tool': cleanup.tool,
                    'cleanup_func': cleanup.cleanup_func,
                    'status': 'ok',
                    'message': message,
                    'arguments': kwargs,
                })
            except Exception as exc:
                logger.warning('Cleanup failed for %s/%s: %s', cleanup.server, cleanup.cleanup_func, exc)
                results.append({
                    'server': cleanup.server,
                    'tool': cleanup.tool,
                    'cleanup_func': cleanup.cleanup_func,
                    'status': 'error',
                    'error': str(exc),
                })
            await asyncio.sleep(0)

    return results
