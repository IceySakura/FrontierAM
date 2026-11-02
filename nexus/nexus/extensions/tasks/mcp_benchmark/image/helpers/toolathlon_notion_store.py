#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import re
import uuid
import zipfile

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree as ET


try:
    from .toolathlon_mock_utils import mutate_store, read_store, utc_now_iso
except ImportError:
    from toolathlon_mock_utils import mutate_store, read_store, utc_now_iso


STORE_PATH_ENV = 'NEXUS_TOOLATHLON_NOTION_STORE_PATH'
DEFAULT_STORE_PATH = '/tmp/nexus-toolathlon-notion-store.json'
LOCAL_USER_ID = 'local-notion-user'
ROOT_PAGE_ID = '11111111-1111-1111-1111-111111111111'
SOURCE_PARENT_ID = '22222222-2222-2222-2222-222222222222'
EVAL_PARENT_ID = '33333333-3333-3333-3333-333333333333'
ROOT_PAGE_TITLE = 'Mock Toolathlon Workspace'
SOURCE_PARENT_TITLE = 'Notion Source Page'
EVAL_PARENT_TITLE = 'Notion Eval Page'
NOTION_VERSION = '2022-06-28'

_NOTION_ID_RE = re.compile(r'[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
_DUPLICATED_PAGE_IDS = {
    'mcp_experiments_recordings': '26ec4171-366e-8009-b539-dfda842c355c',
    'Job Finder': '26ec4171-366e-8027-a0e8-ec53d93c7786',
    'HR Record': '2a8fe807-c6ec-810c-b853-e0d5ae0d6c25',
    'Ultimate Movie Tracker': '28cd1b2a-54b2-815e-823f-cfea0980e9d4',
    'Colley Whisson': '271c4171-366e-8002-a076-ffe132689574',
    'Oil Price': '269c4171-366e-806b-9937-e6a4184f2d1c',
    'Quant Research': '289d1b2a-54b2-80cc-b875-e662a242b86c',
    'Task Tracker': '2c0fe807-c6ec-814e-a9db-c53e66438f87',
}
_TEXT_BLOCK_TYPES = {
    'paragraph',
    'heading_1',
    'heading_2',
    'heading_3',
    'bulleted_list_item',
    'numbered_list_item',
    'quote',
    'callout',
    'to_do',
}


class NotionStoreError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


def notion_store_path() -> str:
    configured = os.getenv(STORE_PATH_ENV)
    if configured:
        return configured
    return DEFAULT_STORE_PATH


def _toolathlon_dataset_root() -> Path:
    configured = os.getenv('NEXUS_TOOLATHLON_DATASET_ROOT')
    if configured:
        return Path(configured)
    candidates = [Path('/opt/mcp-benchmarks/datasets/toolathlon')]
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidates.append(parent / 'datasets' / 'toolathlon')
        candidates.append(parent / 'workspace' / 'mcp' / 'datasets' / 'toolathlon')
    for candidate in candidates:
        if (candidate / 'tasks' / 'finalpool').exists():
            return candidate
    return candidates[0]


def _tasks_root() -> Path:
    return _toolathlon_dataset_root() / 'tasks' / 'finalpool'


def _task_path(case_name: str) -> Path:
    return _tasks_root() / case_name


def _normalize_id(value: str | None) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    compact = ''.join(ch for ch in text if ch.isalnum()).lower()
    if len(compact) == 32 and all(ch in '0123456789abcdef' for ch in compact):
        return f'{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}'
    return text


def _compact_id(value: str) -> str:
    return _normalize_id(value).replace('-', '')


def _now() -> str:
    return utc_now_iso()


def _new_id() -> str:
    return str(uuid.uuid4())


def _stable_id(*parts: str) -> str:
    seed = 'toolathlon-notion:' + ':'.join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _notion_url(page_id: str) -> str:
    return f'https://www.notion.so/{_compact_id(page_id)}'


def _rich_text_item(text: Any) -> dict[str, Any]:
    value = str(text)
    return {'type': 'text', 'plain_text': value, 'text': {'content': value}}


def _coerce_rich_text(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                plain = str(
                    item.get('plain_text')
                    or item.get('text', {}).get('content')
                    or item.get('mention', {}).get('plain_text')
                    or ''
                )
                if not plain and item.get('type') == 'equation':
                    plain = str(item.get('equation', {}).get('expression') or '')
                items.append({'type': 'text', 'plain_text': plain, 'text': {'content': plain}})
            else:
                items.append(_rich_text_item(item))
        return items
    return [_rich_text_item(value)]


def _title_property(title: Any, *, property_id: str = 'title') -> dict[str, Any]:
    return {'id': property_id, 'type': 'title', 'title': _coerce_rich_text(title)}


def _database_property(
    name: str, prop_type: str, *, property_id: str | None = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        'id': property_id or _stable_id('property', name)[:8],
        'name': name,
        'type': prop_type,
        prop_type: copy.deepcopy(extra or {}),
    }


def _blank_page_property_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    prop_type = str(schema.get('type') or 'rich_text')
    property_id = str(schema.get('id') or _stable_id('property', schema.get('name') or prop_type)[:8])
    if prop_type == 'title':
        return {'id': property_id, 'type': 'title', 'title': []}
    if prop_type == 'rich_text':
        return {'id': property_id, 'type': 'rich_text', 'rich_text': []}
    if prop_type in {'select', 'status'}:
        return {'id': property_id, 'type': prop_type, prop_type: None}
    if prop_type == 'multi_select':
        return {'id': property_id, 'type': 'multi_select', 'multi_select': []}
    if prop_type == 'number':
        return {'id': property_id, 'type': 'number', 'number': None}
    if prop_type == 'date':
        return {'id': property_id, 'type': 'date', 'date': None}
    if prop_type == 'email':
        return {'id': property_id, 'type': 'email', 'email': None}
    if prop_type == 'url':
        return {'id': property_id, 'type': 'url', 'url': None}
    if prop_type == 'checkbox':
        return {'id': property_id, 'type': 'checkbox', 'checkbox': False}
    return {'id': property_id, 'type': 'rich_text', 'rich_text': []}


def _infer_property_type(value: Any, *, fallback: str = 'rich_text') -> str:
    if isinstance(value, dict):
        if isinstance(value.get('type'), str):
            return value['type']
        for candidate in (
            'title',
            'rich_text',
            'select',
            'status',
            'multi_select',
            'number',
            'date',
            'email',
            'url',
            'checkbox',
        ):
            if candidate in value:
                return candidate
    return fallback


def _normalize_database_properties(
    properties: dict[str, Any] | None, *, existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    existing = existing or {}
    for name, value in (properties or {}).items():
        current = copy.deepcopy(existing.get(name) or {})
        prop_type = _infer_property_type(value, fallback=str(current.get('type') or 'rich_text'))
        extra = {}
        if isinstance(value, dict):
            current_id = str(current.get('id') or value.get('id') or _stable_id('property', name)[:8])
            extra = copy.deepcopy(value.get(prop_type) if isinstance(value.get(prop_type), dict) else {})
        else:
            current_id = str(current.get('id') or _stable_id('property', name)[:8])
        normalized[name] = {
            'id': current_id,
            'name': name,
            'type': prop_type,
            prop_type: extra,
        }
    for name, value in existing.items():
        normalized.setdefault(name, copy.deepcopy(value))
    return normalized


def _normalize_page_property(
    name: str, value: Any, *, schema: dict[str, Any] | None = None, existing: dict[str, Any] | None = None
) -> dict[str, Any]:
    property_id = str((existing or {}).get('id') or (schema or {}).get('id') or _stable_id('property', name)[:8])
    prop_type = _infer_property_type(
        value, fallback=str((schema or {}).get('type') or (existing or {}).get('type') or 'rich_text')
    )
    if prop_type == 'title':
        items = value.get('title') if isinstance(value, dict) else value
        return {'id': property_id, 'type': 'title', 'title': _coerce_rich_text(items)}
    if prop_type == 'rich_text':
        items = value.get('rich_text') if isinstance(value, dict) else value
        return {'id': property_id, 'type': 'rich_text', 'rich_text': _coerce_rich_text(items)}
    if prop_type in {'select', 'status'}:
        raw = value.get(prop_type) if isinstance(value, dict) else value
        if isinstance(raw, dict):
            name_value = str(raw.get('name') or '').strip()
            selected = {'name': name_value} if name_value else None
        else:
            name_value = str(raw or '').strip()
            selected = {'name': name_value} if name_value else None
        return {'id': property_id, 'type': prop_type, prop_type: selected}
    if prop_type == 'multi_select':
        raw = value.get('multi_select') if isinstance(value, dict) else value
        items = []
        for item in raw or []:
            name_value = str(item.get('name') or '').strip() if isinstance(item, dict) else str(item).strip()
            if name_value:
                items.append({'name': name_value})
        return {'id': property_id, 'type': 'multi_select', 'multi_select': items}
    if prop_type == 'number':
        raw = value.get('number') if isinstance(value, dict) else value
        return {'id': property_id, 'type': 'number', 'number': None if raw in {None, ''} else float(raw)}
    if prop_type == 'date':
        raw = value.get('date') if isinstance(value, dict) else value
        if isinstance(raw, dict):
            date_value = copy.deepcopy(raw)
        else:
            text = str(raw or '').strip()
            date_value = {'start': text} if text else None
        return {'id': property_id, 'type': 'date', 'date': date_value}
    if prop_type == 'email':
        raw = value.get('email') if isinstance(value, dict) else value
        text = str(raw or '').strip()
        return {'id': property_id, 'type': 'email', 'email': text or None}
    if prop_type == 'url':
        raw = value.get('url') if isinstance(value, dict) else value
        text = str(raw or '').strip()
        return {'id': property_id, 'type': 'url', 'url': text or None}
    if prop_type == 'checkbox':
        raw = value.get('checkbox') if isinstance(value, dict) else value
        return {'id': property_id, 'type': 'checkbox', 'checkbox': bool(raw)}
    return _normalize_page_property(name, {'rich_text': _coerce_rich_text(value)}, schema=schema, existing=existing)


def _extract_title(properties: dict[str, Any] | None) -> str:
    for name, value in (properties or {}).items():
        if not isinstance(value, dict):
            continue
        if value.get('type') == 'title':
            title = ''.join(
                item.get('plain_text') or item.get('text', {}).get('content') or '' for item in value.get('title') or []
            )
            if title:
                return title
            if name:
                return str(name)
    return ''


def _property_text(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return ''
    prop_type = value.get('type')
    if prop_type in {'title', 'rich_text'}:
        key = 'title' if prop_type == 'title' else 'rich_text'
        return ''.join(
            item.get('plain_text') or item.get('text', {}).get('content') or '' for item in value.get(key) or []
        ).strip()
    if prop_type in {'select', 'status'}:
        selected = value.get(prop_type) or {}
        return str(selected.get('name') or '').strip()
    if prop_type == 'multi_select':
        return ', '.join(
            str(item.get('name') or '').strip()
            for item in value.get('multi_select') or []
            if str(item.get('name') or '').strip()
        )
    if prop_type == 'email':
        return str(value.get('email') or '').strip()
    if prop_type == 'url':
        return str(value.get('url') or '').strip()
    if prop_type == 'date':
        date_value = value.get('date') or {}
        return str(date_value.get('start') or '').strip()
    if prop_type == 'number':
        number = value.get('number')
        return '' if number is None else str(number)
    return ''


def _property_number(value: dict[str, Any] | None) -> float | None:
    if not isinstance(value, dict) or value.get('type') != 'number':
        return None
    number = value.get('number')
    if number is None:
        return None
    return float(number)


def _default_page(page_id: str, title: str, parent: dict[str, Any]) -> dict[str, Any]:
    created = _now()
    return {
        'object': 'page',
        'id': page_id,
        'archived': False,
        'in_trash': False,
        'created_time': created,
        'last_edited_time': created,
        'url': _notion_url(page_id),
        'parent': copy.deepcopy(parent),
        'properties': {'title': _title_property(title)},
    }


def _default_child_page_block(page_id: str, title: str) -> dict[str, Any]:
    return {
        'object': 'block',
        'id': page_id,
        'type': 'child_page',
        'archived': False,
        'in_trash': False,
        'has_children': False,
        'child_page': {'title': title},
    }


def _default_child_database_block(database_id: str, title: str) -> dict[str, Any]:
    return {
        'object': 'block',
        'id': database_id,
        'type': 'child_database',
        'archived': False,
        'in_trash': False,
        'has_children': False,
        'child_database': {'title': title},
    }


def _coerce_state(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault('version', 1)
    state.setdefault('pages', {})
    state.setdefault('databases', {})
    state.setdefault('blocks', {})
    state.setdefault('children', {})
    state.setdefault('comments', {})
    state.setdefault('users', {})
    state['users'].setdefault(
        LOCAL_USER_ID,
        {
            'object': 'user',
            'id': LOCAL_USER_ID,
            'name': 'local-notion',
            'type': 'person',
            'person': {'email': 'local-notion@mcp.local'},
        },
    )
    if ROOT_PAGE_ID not in state['pages']:
        state['pages'][ROOT_PAGE_ID] = _default_page(
            ROOT_PAGE_ID, ROOT_PAGE_TITLE, {'type': 'workspace', 'workspace': True}
        )
    if SOURCE_PARENT_ID not in state['pages']:
        state['pages'][SOURCE_PARENT_ID] = _default_page(
            SOURCE_PARENT_ID, SOURCE_PARENT_TITLE, {'type': 'page_id', 'page_id': ROOT_PAGE_ID}
        )
    if EVAL_PARENT_ID not in state['pages']:
        state['pages'][EVAL_PARENT_ID] = _default_page(
            EVAL_PARENT_ID, EVAL_PARENT_TITLE, {'type': 'page_id', 'page_id': ROOT_PAGE_ID}
        )
    state['blocks'].setdefault(SOURCE_PARENT_ID, _default_child_page_block(SOURCE_PARENT_ID, SOURCE_PARENT_TITLE))
    state['blocks'].setdefault(EVAL_PARENT_ID, _default_child_page_block(EVAL_PARENT_ID, EVAL_PARENT_TITLE))
    state['children'].setdefault(ROOT_PAGE_ID, [])
    state['children'].setdefault(SOURCE_PARENT_ID, [])
    state['children'].setdefault(EVAL_PARENT_ID, [])
    if SOURCE_PARENT_ID not in state['children'][ROOT_PAGE_ID]:
        state['children'][ROOT_PAGE_ID].append(SOURCE_PARENT_ID)
    if EVAL_PARENT_ID not in state['children'][ROOT_PAGE_ID]:
        state['children'][ROOT_PAGE_ID].append(EVAL_PARENT_ID)
    for container_id in list(state['children']):
        state['children'][container_id] = [item for item in state['children'][container_id] if item]
    for page_id, page in state['pages'].items():
        state['children'].setdefault(page_id, [])
        page['id'] = _normalize_id(page_id)
    for database_id, database in state['databases'].items():
        database['id'] = _normalize_id(database_id)
        database.setdefault('row_ids', [])
    for block_id, block in state['blocks'].items():
        block['id'] = _normalize_id(block_id)
    for comment_id, comment in state['comments'].items():
        comment['id'] = _normalize_id(comment_id)
    return state


def default_state() -> dict[str, Any]:
    return _coerce_state({
        'version': 1,
        'pages': {},
        'databases': {},
        'blocks': {},
        'children': {},
        'comments': {},
        'users': {},
    })


def read_notion_state() -> dict[str, Any]:
    return _coerce_state(read_store(notion_store_path(), default_state))


def mutate_notion_state(mutator):
    return mutate_store(notion_store_path(), default_state, lambda state: mutator(_coerce_state(state)))


def _append_child(state: dict[str, Any], container_id: str, child_id: str) -> None:
    state['children'].setdefault(container_id, [])
    if child_id not in state['children'][container_id]:
        state['children'][container_id].append(child_id)


def _remove_child(state: dict[str, Any], container_id: str, child_id: str) -> None:
    state['children'].setdefault(container_id, [])
    state['children'][container_id] = [item for item in state['children'][container_id] if item != child_id]


def _touch_timestamp(state: dict[str, Any], container_id: str) -> None:
    now = _now()
    if container_id in state['pages']:
        state['pages'][container_id]['last_edited_time'] = now
    if container_id in state['databases']:
        state['databases'][container_id]['last_edited_time'] = now
    if container_id in state['blocks']:
        state['blocks'][container_id]['last_edited_time'] = now


def _create_page(
    state: dict[str, Any],
    title: str,
    parent: dict[str, Any],
    *,
    page_id: str | None = None,
    properties: dict[str, Any] | None = None,
    children: list[Any] | None = None,
) -> str:
    page_id = _normalize_id(page_id or _new_id())
    created = _now()
    page = {
        'object': 'page',
        'id': page_id,
        'archived': False,
        'in_trash': False,
        'created_time': created,
        'last_edited_time': created,
        'url': _notion_url(page_id),
        'parent': copy.deepcopy(parent),
        'properties': {},
    }
    if parent.get('type') == 'database_id':
        database_id = _normalize_id(parent.get('database_id'))
        database = state['databases'].get(database_id)
        schema = database.get('properties', {}) if database else {}
        page['properties'] = {name: _blank_page_property_from_schema(prop) for name, prop in schema.items()}
        for name, value in (properties or {}).items():
            page['properties'][name] = _normalize_page_property(
                name, value, schema=schema.get(name), existing=page['properties'].get(name)
            )
        if title:
            title_name = next((name for name, prop in schema.items() if prop.get('type') == 'title'), 'title')
            page['properties'][title_name] = _normalize_page_property(
                title_name,
                {'title': _coerce_rich_text(title)},
                schema=schema.get(title_name),
                existing=page['properties'].get(title_name),
            )
        if database is not None and page_id not in database['row_ids']:
            database['row_ids'].append(page_id)
            database['last_edited_time'] = created
    else:
        merged = dict(properties or {})
        if not any(_infer_property_type(value, fallback='') == 'title' for value in merged.values()):
            merged['title'] = {'title': _coerce_rich_text(title)}
        page['properties'] = {
            name: _normalize_page_property(name, value, existing=page['properties'].get(name))
            for name, value in merged.items()
        }
    state['pages'][page_id] = page
    state['children'].setdefault(page_id, [])
    if parent.get('type') == 'page_id':
        parent_id = _normalize_id(parent.get('page_id'))
        state['blocks'][page_id] = _default_child_page_block(
            page_id, _extract_title(page['properties']) or title or 'Untitled'
        )
        _append_child(state, parent_id, page_id)
        _touch_timestamp(state, parent_id)
    if children:
        _append_blocks(state, page_id, children)
    return page_id


def _create_database(
    state: dict[str, Any],
    title: str,
    parent_page_id: str,
    properties: dict[str, Any],
    *,
    database_id: str | None = None,
) -> str:
    database_id = _normalize_id(database_id or _new_id())
    created = _now()
    database = {
        'object': 'database',
        'id': database_id,
        'archived': False,
        'in_trash': False,
        'created_time': created,
        'last_edited_time': created,
        'url': _notion_url(database_id),
        'parent': {'type': 'page_id', 'page_id': _normalize_id(parent_page_id)},
        'title': _coerce_rich_text(title),
        'properties': _normalize_database_properties(properties),
        'row_ids': [],
    }
    state['databases'][database_id] = database
    state['blocks'][database_id] = _default_child_database_block(database_id, title)
    _append_child(state, _normalize_id(parent_page_id), database_id)
    _touch_timestamp(state, _normalize_id(parent_page_id))
    return database_id


def _normalize_block(raw: Any) -> tuple[dict[str, Any], list[Any]]:
    if isinstance(raw, str):
        raw = {'type': 'paragraph', 'paragraph': {'rich_text': _coerce_rich_text(raw)}}
    elif not isinstance(raw, dict):
        raw = {'type': 'paragraph', 'paragraph': {'rich_text': _coerce_rich_text(str(raw))}}
    else:
        raw = copy.deepcopy(raw)
    block_type = str(raw.get('type') or 'paragraph')
    nested_children = []
    if isinstance(raw.get('children'), list):
        nested_children = raw.pop('children')
    payload = raw.get(block_type)
    if not isinstance(payload, dict):
        payload = {}
    if isinstance(payload.get('children'), list):
        nested_children = payload.pop('children')
    block_id = _normalize_id(raw.get('id') or _new_id())
    block = {
        'object': 'block',
        'id': block_id,
        'type': block_type,
        'archived': False,
        'in_trash': False,
        'has_children': bool(nested_children),
    }
    if block_type in _TEXT_BLOCK_TYPES:
        text_payload = dict(payload)
        text_payload['rich_text'] = _coerce_rich_text(
            text_payload.get('rich_text')
            or text_payload.get('text')
            or raw.get('plain_text')
            or raw.get('content')
            or ''
        )
        if block_type == 'to_do':
            text_payload.setdefault('checked', False)
        block[block_type] = text_payload
    elif block_type in {'bookmark', 'embed'}:
        block[block_type] = {
            'url': str(payload.get('url') or raw.get('url') or '').strip(),
            'caption': _coerce_rich_text(payload.get('caption') or raw.get('caption') or []),
        }
    elif block_type == 'video':
        video_payload = dict(payload)
        if 'external' not in video_payload and 'file' not in video_payload:
            url = str(video_payload.get('url') or raw.get('url') or '').strip()
            video_payload = {'external': {'url': url}} if url else {}
        video_payload['caption'] = _coerce_rich_text(video_payload.get('caption') or raw.get('caption') or [])
        block['video'] = video_payload
    elif block_type == 'divider':
        block['divider'] = {}
    elif block_type == 'child_page':
        block['child_page'] = {'title': str(payload.get('title') or raw.get('title') or '').strip()}
    elif block_type == 'child_database':
        block['child_database'] = {'title': str(payload.get('title') or raw.get('title') or '').strip()}
    else:
        generic = dict(payload)
        if 'rich_text' in generic or 'text' in generic or raw.get('content'):
            generic['rich_text'] = _coerce_rich_text(
                generic.get('rich_text') or generic.get('text') or raw.get('content') or ''
            )
        block[block_type] = generic
    return block, nested_children


def _append_blocks(state: dict[str, Any], container_id: str, children: Iterable[Any]) -> list[dict[str, Any]]:
    created_blocks: list[dict[str, Any]] = []
    for child in children:
        block, nested = _normalize_block(child)
        state['blocks'][block['id']] = block
        state['children'].setdefault(block['id'], [])
        _append_child(state, _normalize_id(container_id), block['id'])
        if nested:
            block['has_children'] = True
            _append_blocks(state, block['id'], nested)
        created_blocks.append(copy.deepcopy(block))
    _touch_timestamp(state, _normalize_id(container_id))
    return created_blocks


def _create_database_row(
    state: dict[str, Any],
    database_id: str,
    properties: dict[str, Any],
    *,
    title: str | None = None,
    children: list[Any] | None = None,
    page_id: str | None = None,
) -> str:
    database = state['databases'][_normalize_id(database_id)]
    resolved_title = title or ''
    if not resolved_title:
        for _name, value in properties.items():
            prop_type = _infer_property_type(value)
            if prop_type == 'title':
                resolved_title = ''.join(
                    item.get('plain_text') or item.get('text', {}).get('content') or ''
                    for item in _coerce_rich_text(value.get('title') if isinstance(value, dict) else value)
                )
                break
    return _create_page(
        state,
        resolved_title or 'Untitled',
        {'type': 'database_id', 'database_id': database['id']},
        page_id=page_id,
        properties=properties,
        children=children,
    )


def _archive_page_tree(state: dict[str, Any], page_id: str) -> None:
    page_id = _normalize_id(page_id)
    page = state['pages'].get(page_id)
    if page is None:
        return
    page['archived'] = True
    page['in_trash'] = True
    page['last_edited_time'] = _now()
    parent = page.get('parent') or {}
    if parent.get('type') == 'page_id':
        parent_id = _normalize_id(parent.get('page_id'))
        _remove_child(state, parent_id, page_id)
        if page_id in state['blocks']:
            state['blocks'][page_id]['archived'] = True
            state['blocks'][page_id]['in_trash'] = True
        _touch_timestamp(state, parent_id)
    elif parent.get('type') == 'database_id':
        database_id = _normalize_id(parent.get('database_id'))
        database = state['databases'].get(database_id)
        if database is not None:
            database['last_edited_time'] = _now()
    for block_id in list(state['children'].get(page_id, [])):
        if block_id in state['databases']:
            _archive_database(state, block_id)
        elif block_id in state['pages']:
            _archive_page_tree(state, block_id)
        elif block_id in state['blocks']:
            _archive_block(state, block_id)


def _archive_database(state: dict[str, Any], database_id: str) -> None:
    database_id = _normalize_id(database_id)
    database = state['databases'].get(database_id)
    if database is None:
        return
    database['archived'] = True
    database['in_trash'] = True
    database['last_edited_time'] = _now()
    parent = database.get('parent') or {}
    if parent.get('type') == 'page_id':
        parent_id = _normalize_id(parent.get('page_id'))
        _remove_child(state, parent_id, database_id)
        if database_id in state['blocks']:
            state['blocks'][database_id]['archived'] = True
            state['blocks'][database_id]['in_trash'] = True
        _touch_timestamp(state, parent_id)
    for row_id in list(database.get('row_ids', [])):
        _archive_page_tree(state, row_id)


def _archive_block(state: dict[str, Any], block_id: str) -> None:
    block_id = _normalize_id(block_id)
    if block_id in state['pages']:
        _archive_page_tree(state, block_id)
        return
    if block_id in state['databases']:
        _archive_database(state, block_id)
        return
    block = state['blocks'].get(block_id)
    if block is None:
        return
    block['archived'] = True
    block['in_trash'] = True
    block['last_edited_time'] = _now()
    for container_id, children in state['children'].items():
        if block_id in children:
            _remove_child(state, container_id, block_id)
            _touch_timestamp(state, container_id)
            break
    for child_id in list(state['children'].get(block_id, [])):
        _archive_block(state, child_id)


def _parse_markdown_table_headers(path: Path) -> list[str]:
    content = path.read_text(encoding='utf-8')
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and '---' not in stripped:
            return [cell.strip() for cell in stripped.strip('|').split('|')]
    return []


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    namespace = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in root.findall('a:si', namespace):
                texts = [node.text or '' for node in item.findall('.//a:t', namespace)]
                shared.append(''.join(texts))
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        rel_map = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels}
        first_sheet = workbook.find('a:sheets/a:sheet', namespace)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target = 'xl/' + rel_map[rel_id]
        sheet = ET.fromstring(archive.read(target))
        rows: list[list[str]] = []
        for row in sheet.findall('.//a:sheetData/a:row', namespace):
            values: list[str] = []
            for cell in row.findall('a:c', namespace):
                cell_type = cell.attrib.get('t')
                value_node = cell.find('a:v', namespace)
                if cell_type == 'inlineStr':
                    text = ''.join(node.text or '' for node in cell.findall('.//a:t', namespace))
                    values.append(text)
                    continue
                if value_node is None:
                    values.append('')
                    continue
                raw = value_node.text or ''
                if cell_type == 's':
                    values.append(shared[int(raw)])
                else:
                    values.append(raw)
            rows.append(values)
    if not rows:
        return []
    header = rows[0]
    output: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        output.append({header[index]: row[index] if index < len(row) else '' for index in range(len(header))})
    return output


@lru_cache(maxsize=1)
def _task_tracker_before_rows() -> list[dict[str, str]]:
    return _read_xlsx_rows(_task_path('task-tracker') / 'groundtruth_workspace' / 'notion_table_before.xlsx')


@lru_cache(maxsize=1)
def _experiments_headers() -> list[str]:
    return _parse_markdown_table_headers(
        _task_path('experiments-recordings') / 'initial_workspace' / 'table_template.txt'
    )


def _seed_experiments_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    headers = _experiments_headers()
    if not headers:
        headers = [
            'Run Name',
            'MultiHiertt',
            'HiTab',
            'SuperGPQA',
            'GPQA',
            'CodeIO',
            'ArcAgI1',
            'MATH',
            'AMC (4x)',
            'AIME (8x)',
            'Zebra Puzzle',
            'Ordering Puzzle',
            'Graph Logical',
            'MBPP',
            'LiveCodeBench',
            'HumanEval',
            'Best Step (Average)',
        ]
    properties = {}
    for header in headers:
        if header == 'Run Name':
            properties[header] = _database_property(header, 'title')
        elif header == 'Best Step (Average)':
            properties[header] = _database_property(header, 'rich_text')
        else:
            properties[header] = _database_property(header, 'number')
    _create_database(state, 'Experiments Recordings', page_id, properties)


def _seed_job_finder_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    properties = {
        'Company': _database_property('Company', 'title'),
        'Position': _database_property('Position', 'rich_text'),
        'Location': _database_property('Location', 'rich_text'),
        'Flexibility': _database_property('Flexibility', 'select'),
        'Status': _database_property('Status', 'select'),
        'Salary Range': _database_property('Salary Range', 'rich_text'),
        'Interview Date': _database_property('Interview Date', 'date'),
        'Connect Email': _database_property('Connect Email', 'email'),
    }
    database_id = _create_database(state, 'Job Tracker', page_id, properties)
    rows = [
        {
            'title': 'HCD',
            'Position': 'Software Engineer',
            'Location': 'Los Angeles, CA',
            'Flexibility': 'On-site',
            'Status': 'Checking',
            'Salary Range': '$4500/month',
            'Connect Email': 'laura.hall@mcp.com',
        },
        {
            'title': 'AHC',
            'Position': 'Software Engineer',
            'Location': 'Pasadena, CA',
            'Flexibility': 'On-site',
            'Status': 'Checking',
            'Salary Range': '$5200/month',
            'Connect Email': 'janetr@mcp.com',
        },
        {
            'title': 'BlueWave',
            'Position': 'Data Analyst',
            'Location': 'Santa Monica, CA',
            'Flexibility': 'On-site',
            'Status': 'Checking',
            'Salary Range': '$6100/month',
            'Connect Email': 'melissag@mcp.com',
        },
        {
            'title': 'NorthPeak',
            'Position': 'Software Engineer',
            'Location': 'San Francisco, CA',
            'Flexibility': 'On-site',
            'Status': 'Checking',
            'Salary Range': '$7800/month',
            'Connect Email': 'henry_taylor90@mcp.com',
        },
        {
            'title': 'RemoteOps',
            'Position': 'Software Manager',
            'Location': 'Irvine, CA',
            'Flexibility': 'Remote',
            'Status': 'Checking',
            'Salary Range': '$6800/month',
            'Connect Email': 'alexander.adams@mcp.com',
        },
    ]
    for row in rows:
        _create_database_row(
            state,
            database_id,
            {
                'Company': {'title': _coerce_rich_text(row['title'])},
                'Position': {'rich_text': _coerce_rich_text(row['Position'])},
                'Location': {'rich_text': _coerce_rich_text(row['Location'])},
                'Flexibility': {'select': {'name': row['Flexibility']}},
                'Status': {'select': {'name': row['Status']}},
                'Salary Range': {'rich_text': _coerce_rich_text(row['Salary Range'])},
                'Connect Email': {'email': row['Connect Email']},
            },
            title=row['title'],
        )


def _seed_hr_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    properties = {
        'Candidate Name': _database_property('Candidate Name', 'title'),
        'Email': _database_property('Email', 'email'),
        'School': _database_property('School', 'rich_text'),
        'Highest Degree': _database_property('Highest Degree', 'select'),
        'Applied Position': _database_property('Applied Position', 'rich_text'),
        'Status': _database_property('Status', 'select'),
    }
    database_id = _create_database(state, 'Candidates', page_id, properties)
    _create_database_row(
        state,
        database_id,
        {
            'Candidate Name': {'title': _coerce_rich_text('Sample Candidate')},
            'Email': {'email': 'sample.candidate@mcp.com'},
            'School': {'rich_text': _coerce_rich_text('Example Business School')},
            'Highest Degree': {'select': {'name': 'bachelor'}},
            'Applied Position': {'rich_text': _coerce_rich_text('Investment Analyst')},
            'Status': {'select': {'name': 'Sample'}},
        },
        title='Sample Candidate',
    )


def _seed_movie_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    properties = {
        'Name': _database_property('Name', 'title'),
        'Status': _database_property('Status', 'select'),
        'Genre': _database_property('Genre', 'multi_select'),
        'Released': _database_property('Released', 'number'),
        'Director': _database_property('Director', 'rich_text'),
        'Trailer': _database_property('Trailer', 'url'),
    }
    database_id = _create_database(state, 'Movies', page_id, properties)
    rows = [
        ('Once Upon a Time... in Hollywood', ['Comedy', 'Drama'], 'Watched'),
        ('Nobody', ['Action', 'Thriller'], 'Watched'),
        ('Kill Bill: Vol. 1', ['Action', 'Crime'], 'Watched'),
        ('John Wick 4', ['Action', 'Thriller'], 'Watched'),
        ('Barbie', ['Comedy', 'Fantasy'], 'Watched'),
        ('The Dark Knight', ['Action', 'Crime'], 'Watched'),
        ('Interstellar', ['Science Fiction', 'Drama'], 'Watched'),
        ('Oppenheimer', ['Drama', 'History'], 'Watched'),
    ]
    for movie_name, genres, status in rows:
        _create_database_row(
            state,
            database_id,
            {
                'Name': {'title': _coerce_rich_text(movie_name)},
                'Status': {'select': {'name': status}},
                'Genre': {'multi_select': [{'name': genre} for genre in genres]},
                'Released': {'number': None},
                'Director': {'rich_text': []},
                'Trailer': {'url': None},
            },
            title=movie_name,
        )


def _seed_personal_website_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(
        state,
        title,
        {'type': 'page_id', 'page_id': parent_page_id},
        page_id=page_id,
        children=[
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('About Me')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('Paintings')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('Workshop')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('Prizes')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('Exhibitions')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
        ],
    )


def _seed_oil_price_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    summary_properties = {
        'Month (UTC)': _database_property('Month (UTC)', 'title'),
        'WTI Close': _database_property('WTI Close', 'number'),
        'Brent Close': _database_property('Brent Close', 'number'),
        'WTI MoM %': _database_property('WTI MoM %', 'number'),
        'Brent MoM %': _database_property('Brent MoM %', 'number'),
        'Brent-WTI Spread': _database_property('Brent-WTI Spread', 'number'),
        'Spread MoM %': _database_property('Spread MoM %', 'number'),
        'Spread Z-Score (6m)': _database_property('Spread Z-Score (6m)', 'number'),
        'Regime': _database_property('Regime', 'select'),
        'Signal': _database_property('Signal', 'select'),
    }
    backtest_properties = {
        'Name': _database_property('Name', 'title'),
        'Type': _database_property('Type', 'select'),
        'Trades': _database_property('Trades', 'number'),
        'Total Return %': _database_property('Total Return %', 'number'),
        'Annualized Return %': _database_property('Annualized Return %', 'number'),
        'Sharpe (ann.)': _database_property('Sharpe (ann.)', 'number'),
        'Win Rate %': _database_property('Win Rate %', 'number'),
        'Max Drawdown %': _database_property('Max Drawdown %', 'number'),
        'Period Start': _database_property('Period Start', 'rich_text'),
        'Period End': _database_property('Period End', 'rich_text'),
        'Cost Assumption': _database_property('Cost Assumption', 'rich_text'),
        'Signal': _database_property('Signal', 'select'),
        'Entry Month': _database_property('Entry Month', 'rich_text'),
        'Exit Month': _database_property('Exit Month', 'rich_text'),
        'Entry Spread': _database_property('Entry Spread', 'number'),
        'Exit Spread': _database_property('Exit Spread', 'number'),
        'Net PnL %': _database_property('Net PnL %', 'number'),
        'Leg Returns %': _database_property('Leg Returns %', 'rich_text'),
        'Notes': _database_property('Notes', 'rich_text'),
    }
    _create_database(state, 'Oil Market Summary', page_id, summary_properties)
    _create_database(state, 'Spread Strategy Backtest', page_id, backtest_properties)


def _seed_quant_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(
        state,
        title,
        {'type': 'page_id', 'page_id': parent_page_id},
        page_id=page_id,
        children=[
            {'type': 'heading_1', 'heading_1': {'rich_text': _coerce_rich_text('Quant Research')}},
            {'type': 'paragraph', 'paragraph': {'rich_text': []}},
        ],
    )


def _seed_task_tracker_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)
    properties = {
        'Task Name': _database_property('Task Name', 'title'),
        'Task Status': _database_property('Task Status', 'select'),
        'Implementor': _database_property('Implementor', 'multi_select'),
    }
    database_id = _create_database(state, 'Task Tracker Items', page_id, properties)
    for row in _task_tracker_before_rows():
        task_name = str(row.get('Task Name') or '').strip()
        if not task_name:
            continue
        _create_database_row(
            state,
            database_id,
            {
                'Task Name': {'title': _coerce_rich_text(task_name)},
                'Task Status': {'select': {'name': str(row.get('Task Status') or '').strip()}},
                'Implementor': {
                    'multi_select': [{'name': str(row.get('Implementor') or '').strip()}]
                    if str(row.get('Implementor') or '').strip()
                    else []
                },
            },
            title=task_name,
        )


def _seed_generic_page(state: dict[str, Any], title: str, parent_page_id: str, page_id: str) -> None:
    _create_page(state, title, {'type': 'page_id', 'page_id': parent_page_id}, page_id=page_id)


_PAGE_SEEDERS = {
    'mcp_experiments_recordings': _seed_experiments_page,
    'Job Finder': _seed_job_finder_page,
    'HR Record': _seed_hr_page,
    'Ultimate Movie Tracker': _seed_movie_page,
    'Colley Whisson': _seed_personal_website_page,
    'Oil Price': _seed_oil_price_page,
    'Quant Research': _seed_quant_page,
    'Task Tracker': _seed_task_tracker_page,
}


def _seed_page_by_title(state: dict[str, Any], title: str, parent_page_id: str, *, page_id: str | None = None) -> str:
    resolved_page_id = _normalize_id(page_id or _stable_id('seeded-page', title, parent_page_id))
    seeder = _PAGE_SEEDERS.get(title, _seed_generic_page)
    seeder(state, title, _normalize_id(parent_page_id), resolved_page_id)
    return resolved_page_id


def _find_page_under_parent(state: dict[str, Any], parent_page_id: str, title: str) -> str | None:
    for child_id in state['children'].get(_normalize_id(parent_page_id), []):
        if child_id not in state['pages']:
            continue
        page = state['pages'][child_id]
        if page.get('in_trash'):
            continue
        if _extract_title(page.get('properties')) == title:
            return child_id
    return None


def _ensure_source_template_in_state(state: dict[str, Any], page_title: str) -> str:
    existing_id = _find_page_under_parent(state, SOURCE_PARENT_ID, page_title)
    if existing_id:
        return existing_id
    return _seed_page_by_title(state, page_title, SOURCE_PARENT_ID)


def ensure_source_template(page_title: str) -> str:
    def _ensure(state: dict[str, Any]) -> str:
        return _ensure_source_template_in_state(state, page_title)

    return mutate_notion_state(_ensure)


def duplicate_eval_page(page_title: str, duplicated_page_id: str | None = None) -> str:
    target_page_id = _normalize_id(
        duplicated_page_id or _DUPLICATED_PAGE_IDS.get(page_title) or _stable_id('duplicated-page', page_title)
    )

    def _duplicate(state: dict[str, Any]) -> str:
        _ensure_source_template_in_state(state, page_title)
        existing_id = _find_page_under_parent(state, EVAL_PARENT_ID, page_title)
        if existing_id:
            _archive_page_tree(state, existing_id)
        _seed_page_by_title(state, page_title, EVAL_PARENT_ID, page_id=target_page_id)
        return target_page_id

    return mutate_notion_state(_duplicate)


def _paginated_list(items: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    page_size = int(payload.get('page_size') or 100)
    start_cursor_raw = payload.get('start_cursor')
    start_index = 0
    if start_cursor_raw not in {None, ''}:
        try:
            start_index = max(0, int(start_cursor_raw))
        except (TypeError, ValueError):
            for index, item in enumerate(items):
                if item.get('id') == _normalize_id(str(start_cursor_raw)):
                    start_index = index + 1
                    break
    page = items[start_index : start_index + page_size]
    next_cursor = start_index + page_size if start_index + page_size < len(items) else None
    return {
        'object': 'list',
        'results': page,
        'has_more': next_cursor is not None,
        'next_cursor': None if next_cursor is None else str(next_cursor),
    }


def _search_matches_text(item: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack_parts = []
    if item.get('object') == 'page':
        haystack_parts.append(_extract_title(item.get('properties')))
    if item.get('object') == 'database':
        haystack_parts.append(
            ''.join(
                node.get('plain_text') or node.get('text', {}).get('content') or '' for node in item.get('title') or []
            )
        )
    if item.get('object') == 'page':
        for child_id in item.get('__child_ids__', []):
            block = item.get('__blocks__', {}).get(child_id) or {}
            payload = block.get(block.get('type') or '', {})
            if isinstance(payload, dict):
                if 'rich_text' in payload:
                    haystack_parts.append(
                        ''.join(
                            node.get('plain_text') or node.get('text', {}).get('content') or ''
                            for node in payload.get('rich_text') or []
                        )
                    )
                if 'url' in payload:
                    haystack_parts.append(str(payload.get('url') or ''))
    haystack = '\n'.join(haystack_parts).lower()
    return query.lower() in haystack


def search_objects(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(payload or {})
    filter_object = payload.get('filter') or {}
    query = str(payload.get('query') or '').strip()
    state = read_notion_state()
    results: list[dict[str, Any]] = []
    search_value = str(filter_object.get('value') or '').strip().lower()
    search_property = str(filter_object.get('property') or '').strip().lower()
    include_pages = not (search_property == 'object' and search_value == 'database')
    include_databases = not (search_property == 'object' and search_value == 'page')
    if include_pages:
        for page in state['pages'].values():
            if page.get('in_trash'):
                continue
            page_copy = copy.deepcopy(page)
            page_copy['__child_ids__'] = list(state['children'].get(page['id'], []))
            page_copy['__blocks__'] = state['blocks']
            if _search_matches_text(page_copy, query):
                page_copy.pop('__child_ids__', None)
                page_copy.pop('__blocks__', None)
                results.append(page_copy)
    if include_databases:
        for database in state['databases'].values():
            if database.get('in_trash'):
                continue
            title = ''.join(
                node.get('plain_text') or node.get('text', {}).get('content') or ''
                for node in database.get('title') or []
            )
            if not query or query.lower() in title.lower():
                results.append(copy.deepcopy(database))
    results.sort(key=lambda item: str(item.get('last_edited_time') or ''), reverse=True)
    return _paginated_list(results, payload)


def list_users(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    del payload
    state = read_notion_state()
    return {
        'object': 'list',
        'results': [copy.deepcopy(user) for user in state['users'].values()],
        'has_more': False,
        'next_cursor': None,
    }


def get_user(user_id: str) -> dict[str, Any]:
    state = read_notion_state()
    normalized = _normalize_id(user_id)
    if normalized not in state['users']:
        raise NotionStoreError(404, f'Unknown user: {user_id}')
    return copy.deepcopy(state['users'][normalized])


def get_self() -> dict[str, Any]:
    return get_user(LOCAL_USER_ID)


def get_page(page_id: str) -> dict[str, Any]:
    state = read_notion_state()
    normalized = _normalize_id(page_id)
    page = state['pages'].get(normalized)
    if page is None or page.get('in_trash'):
        raise NotionStoreError(404, f'Unknown page: {page_id}')
    return copy.deepcopy(page)


def create_page(payload: dict[str, Any]) -> dict[str, Any]:
    def _create(state: dict[str, Any]) -> dict[str, Any]:
        parent = copy.deepcopy(payload.get('parent') or {})
        if 'page_id' in parent and parent.get('type') != 'page_id':
            parent = {'type': 'page_id', 'page_id': _normalize_id(parent.get('page_id'))}
        elif 'database_id' in parent and parent.get('type') != 'database_id':
            parent = {'type': 'database_id', 'database_id': _normalize_id(parent.get('database_id'))}
        if parent.get('type') == 'page_id':
            parent['page_id'] = _normalize_id(parent.get('page_id'))
            if parent['page_id'] not in state['pages']:
                raise NotionStoreError(404, f'Unknown parent page: {parent["page_id"]}')
        elif parent.get('type') == 'database_id':
            parent['database_id'] = _normalize_id(parent.get('database_id'))
            if parent['database_id'] not in state['databases']:
                raise NotionStoreError(404, f'Unknown parent database: {parent["database_id"]}')
        else:
            raise NotionStoreError(400, 'page parent must include page_id or database_id')
        properties = copy.deepcopy(payload.get('properties') or {})
        title = _extract_title(properties) or str(payload.get('title') or payload.get('page_title') or 'Untitled')
        page_id = _create_page(state, title, parent, properties=properties, children=payload.get('children'))
        return copy.deepcopy(state['pages'][page_id])

    return mutate_notion_state(_create)


def update_page(page_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized_page_id = _normalize_id(page_id)

    def _update(state: dict[str, Any]) -> dict[str, Any]:
        page = state['pages'].get(normalized_page_id)
        if page is None:
            raise NotionStoreError(404, f'Unknown page: {page_id}')
        if 'archived' in payload or 'in_trash' in payload:
            archived = bool(payload.get('archived') if 'archived' in payload else payload.get('in_trash'))
            if archived:
                _archive_page_tree(state, normalized_page_id)
            else:
                page['archived'] = False
                page['in_trash'] = False
        properties = payload.get('properties')
        if isinstance(properties, dict):
            parent = page.get('parent') or {}
            database = (
                state['databases'].get(_normalize_id(parent.get('database_id')))
                if parent.get('type') == 'database_id'
                else None
            )
            schema = database.get('properties', {}) if database else {}
            for name, value in properties.items():
                page['properties'][name] = _normalize_page_property(
                    name, value, schema=schema.get(name), existing=page['properties'].get(name)
                )
        elif payload.get('title') or payload.get('page_title'):
            title = str(payload.get('title') or payload.get('page_title') or '').strip()
            if title:
                existing_title_key = next(
                    (
                        name
                        for name, value in page['properties'].items()
                        if isinstance(value, dict) and value.get('type') == 'title'
                    ),
                    'title',
                )
                page['properties'][existing_title_key] = _normalize_page_property(
                    existing_title_key,
                    {'title': _coerce_rich_text(title)},
                    existing=page['properties'].get(existing_title_key),
                )
        page['last_edited_time'] = _now()
        parent = page.get('parent') or {}
        if parent.get('type') == 'page_id' and normalized_page_id in state['blocks']:
            state['blocks'][normalized_page_id]['child_page']['title'] = _extract_title(page['properties']) or state[
                'blocks'
            ][normalized_page_id]['child_page'].get('title', '')
        return copy.deepcopy(page)

    return mutate_notion_state(_update)


def get_database(database_id: str) -> dict[str, Any]:
    state = read_notion_state()
    normalized = _normalize_id(database_id)
    database = state['databases'].get(normalized)
    if database is None or database.get('in_trash'):
        raise NotionStoreError(404, f'Unknown database: {database_id}')
    return copy.deepcopy(database)


def create_database(payload: dict[str, Any]) -> dict[str, Any]:
    def _create(state: dict[str, Any]) -> dict[str, Any]:
        parent = copy.deepcopy(payload.get('parent') or {})
        page_id = _normalize_id(parent.get('page_id') or (parent.get('type') == 'page_id' and parent.get('page_id')))
        if not page_id or page_id not in state['pages']:
            raise NotionStoreError(400, 'database parent.page_id is required')
        title_value = payload.get('title') or payload.get('database_title') or []
        if isinstance(title_value, list):
            title = ''.join(
                item.get('plain_text') or item.get('text', {}).get('content') or ''
                for item in _coerce_rich_text(title_value)
            )
        else:
            title = str(title_value or '').strip()
        database_id = _create_database(
            state, title or 'Untitled Database', page_id, copy.deepcopy(payload.get('properties') or {})
        )
        return copy.deepcopy(state['databases'][database_id])

    return mutate_notion_state(_create)


def update_database(database_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized_database_id = _normalize_id(database_id)

    def _update(state: dict[str, Any]) -> dict[str, Any]:
        database = state['databases'].get(normalized_database_id)
        if database is None:
            raise NotionStoreError(404, f'Unknown database: {database_id}')
        if isinstance(payload.get('properties'), dict):
            database['properties'] = _normalize_database_properties(
                payload['properties'], existing=database.get('properties')
            )
        if payload.get('title') is not None:
            title_value = payload.get('title')
            if isinstance(title_value, list):
                database['title'] = _coerce_rich_text(title_value)
            else:
                database['title'] = _coerce_rich_text(str(title_value))
        database['last_edited_time'] = _now()
        if normalized_database_id in state['blocks']:
            database_title = ''.join(
                item.get('plain_text') or item.get('text', {}).get('content') or ''
                for item in database.get('title') or []
            )
            state['blocks'][normalized_database_id]['child_database']['title'] = database_title
        return copy.deepcopy(database)

    return mutate_notion_state(_update)


def _matches_filter(page: dict[str, Any], filter_payload: dict[str, Any] | None) -> bool:
    if not filter_payload:
        return True
    if isinstance(filter_payload.get('and'), list):
        return all(_matches_filter(page, item) for item in filter_payload['and'])
    if isinstance(filter_payload.get('or'), list):
        return any(_matches_filter(page, item) for item in filter_payload['or'])
    property_name = str(filter_payload.get('property') or '').strip()
    if not property_name:
        return True
    prop = page.get('properties', {}).get(property_name)
    if prop is None:
        return False
    if 'title' in filter_payload or 'rich_text' in filter_payload:
        target = filter_payload.get('title') or filter_payload.get('rich_text') or {}
        text = _property_text(prop).lower()
        if 'equals' in target:
            return text == str(target.get('equals') or '').lower()
        if 'contains' in target:
            return str(target.get('contains') or '').lower() in text
        if target.get('is_empty'):
            return not text
    if 'select' in filter_payload or 'status' in filter_payload:
        target = filter_payload.get('select') or filter_payload.get('status') or {}
        selected = _property_text(prop).lower()
        if 'equals' in target:
            return selected == str(target.get('equals') or '').lower()
        if 'does_not_equal' in target:
            return selected != str(target.get('does_not_equal') or '').lower()
        if target.get('is_empty'):
            return not selected
    if 'multi_select' in filter_payload:
        target = filter_payload['multi_select'] or {}
        selected_names = [item.strip().lower() for item in _property_text(prop).split(',') if item.strip()]
        if 'contains' in target:
            return str(target.get('contains') or '').lower() in selected_names
        if target.get('is_empty'):
            return not selected_names
    if 'number' in filter_payload:
        target = filter_payload['number'] or {}
        value = _property_number(prop)
        if value is None:
            return False
        if 'equals' in target:
            return value == float(target.get('equals'))
        if 'greater_than' in target:
            return value > float(target.get('greater_than'))
        if 'greater_than_or_equal_to' in target:
            return value >= float(target.get('greater_than_or_equal_to'))
        if 'less_than' in target:
            return value < float(target.get('less_than'))
        if 'less_than_or_equal_to' in target:
            return value <= float(target.get('less_than_or_equal_to'))
    if 'date' in filter_payload:
        target = filter_payload['date'] or {}
        value = _property_text(prop)
        if 'equals' in target:
            return value == str(target.get('equals') or '')
        if 'on_or_after' in target:
            return value >= str(target.get('on_or_after') or '')
        if 'on_or_before' in target:
            return value <= str(target.get('on_or_before') or '')
    return True


def query_database(database_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(payload or {})
    state = read_notion_state()
    normalized = _normalize_id(database_id)
    database = state['databases'].get(normalized)
    if database is None or database.get('in_trash'):
        raise NotionStoreError(404, f'Unknown database: {database_id}')
    rows = []
    for row_id in database.get('row_ids', []):
        page = state['pages'].get(_normalize_id(row_id))
        if page is None or page.get('in_trash'):
            continue
        if not _matches_filter(page, payload.get('filter')):
            continue
        rows.append(copy.deepcopy(page))
    rows.sort(key=lambda item: str(item.get('last_edited_time') or ''), reverse=bool(payload.get('sorts')))
    return _paginated_list(rows, payload)


def get_block(block_id: str) -> dict[str, Any]:
    state = read_notion_state()
    normalized = _normalize_id(block_id)
    block = state['blocks'].get(normalized)
    if block is None or block.get('in_trash'):
        raise NotionStoreError(404, f'Unknown block: {block_id}')
    return copy.deepcopy(block)


def get_block_children(block_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    del payload
    state = read_notion_state()
    normalized = _normalize_id(block_id)
    if normalized not in state['children']:
        if normalized in state['blocks'] or normalized in state['pages'] or normalized in state['databases']:
            return {'object': 'list', 'results': [], 'has_more': False, 'next_cursor': None}
        raise NotionStoreError(404, f'Unknown block or page: {block_id}')
    results: list[dict[str, Any]] = []
    for child_id in state['children'].get(normalized, []):
        if child_id in state['blocks']:
            block = state['blocks'][child_id]
            if not block.get('in_trash'):
                results.append(copy.deepcopy(block))
    return {'object': 'list', 'results': results, 'has_more': False, 'next_cursor': None}


def append_block_children(block_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized_block_id = _normalize_id(block_id)

    def _append(state: dict[str, Any]) -> dict[str, Any]:
        if normalized_block_id not in state['children']:
            if normalized_block_id not in state['pages'] and normalized_block_id not in state['blocks']:
                raise NotionStoreError(404, f'Unknown block or page: {block_id}')
            state['children'].setdefault(normalized_block_id, [])
        created = _append_blocks(state, normalized_block_id, payload.get('children') or [])
        return {'object': 'list', 'results': created, 'has_more': False, 'next_cursor': None}

    return mutate_notion_state(_append)


def update_block(block_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized_block_id = _normalize_id(block_id)

    def _update(state: dict[str, Any]) -> dict[str, Any]:
        block = state['blocks'].get(normalized_block_id)
        if block is None:
            raise NotionStoreError(404, f'Unknown block: {block_id}')
        if 'archived' in payload and payload.get('archived'):
            _archive_block(state, normalized_block_id)
            return copy.deepcopy(state['blocks'].get(normalized_block_id, block))
        block_type = str(payload.get('type') or block.get('type') or '')
        if block_type and block_type in payload and isinstance(payload[block_type], dict):
            updated_block, nested = _normalize_block({
                'id': normalized_block_id,
                'type': block_type,
                block_type: payload[block_type],
            })
            block.update(updated_block)
            if nested:
                state['children'][normalized_block_id] = []
                _append_blocks(state, normalized_block_id, nested)
        elif block_type in _TEXT_BLOCK_TYPES and isinstance(block.get(block_type), dict):
            text_payload = block[block_type]
            if 'rich_text' in payload:
                text_payload['rich_text'] = _coerce_rich_text(payload.get('rich_text'))
            if 'text' in payload:
                text_payload['rich_text'] = _coerce_rich_text(payload.get('text'))
        block['last_edited_time'] = _now()
        return copy.deepcopy(block)

    return mutate_notion_state(_update)


def delete_block(block_id: str) -> dict[str, Any]:
    def _delete(state: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_id(block_id)
        if normalized in state['pages']:
            _archive_page_tree(state, normalized)
            return {'object': 'block', 'id': normalized, 'archived': True, 'in_trash': True}
        if normalized in state['databases']:
            _archive_database(state, normalized)
            return {'object': 'block', 'id': normalized, 'archived': True, 'in_trash': True}
        if normalized not in state['blocks']:
            raise NotionStoreError(404, f'Unknown block: {block_id}')
        _archive_block(state, normalized)
        return {'object': 'block', 'id': normalized, 'archived': True, 'in_trash': True}

    return mutate_notion_state(_delete)


def get_page_property(page_id: str, property_id: str) -> dict[str, Any]:
    page = get_page(page_id)
    property_id = str(property_id or '').strip()
    for name, value in page.get('properties', {}).items():
        if name == property_id or value.get('id') == property_id:
            return {'object': 'property_item', 'property': copy.deepcopy(value)}
    raise NotionStoreError(404, f'Unknown property: {property_id}')


def list_comments(block_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(payload or {})
    state = read_notion_state()
    normalized = _normalize_id(block_id)
    comments = [
        copy.deepcopy(comment)
        for comment in state['comments'].values()
        if _normalize_id(comment.get('parent', {}).get('page_id') or comment.get('parent', {}).get('block_id'))
        == normalized
    ]
    comments.sort(key=lambda item: str(item.get('created_time') or ''), reverse=False)
    return _paginated_list(comments, payload)


def get_comment(comment_id: str) -> dict[str, Any]:
    state = read_notion_state()
    normalized = _normalize_id(comment_id)
    comment = state['comments'].get(normalized)
    if comment is None:
        raise NotionStoreError(404, f'Unknown comment: {comment_id}')
    return copy.deepcopy(comment)


def create_comment(payload: dict[str, Any]) -> dict[str, Any]:
    def _create(state: dict[str, Any]) -> dict[str, Any]:
        parent = copy.deepcopy(payload.get('parent') or {})
        page_id = _normalize_id(parent.get('page_id') or parent.get('block_id'))
        if not page_id:
            raise NotionStoreError(400, 'comment parent.page_id or parent.block_id is required')
        if page_id not in state['pages'] and page_id not in state['blocks']:
            raise NotionStoreError(404, f'Unknown comment parent: {page_id}')
        comment_id = _normalize_id(payload.get('comment_id') or _new_id())
        created = _now()
        comment = {
            'object': 'comment',
            'id': comment_id,
            'created_time': created,
            'last_edited_time': created,
            'created_by': {'object': 'user', 'id': LOCAL_USER_ID},
            'parent': {'type': 'page_id', 'page_id': page_id},
            'rich_text': _coerce_rich_text(payload.get('rich_text') or payload.get('text') or ''),
        }
        state['comments'][comment_id] = comment
        if page_id in state['pages']:
            state['pages'][page_id]['last_edited_time'] = created
        return copy.deepcopy(comment)

    return mutate_notion_state(_create)


def _parse_request_payload(json_payload: Any, params: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(json_payload, dict):
        payload.update(copy.deepcopy(json_payload))
    if params:
        for key, value in params.items():
            if isinstance(value, list):
                payload[key] = value[0] if len(value) == 1 else value
            else:
                payload[key] = value
    return payload


def handle_notion_http_request(
    method: str, url: str, *, json_payload: Any = None, params: dict[str, Any] | None = None
) -> Any:
    parsed = urlparse(url)
    payload = _parse_request_payload(json_payload, parse_qs(parsed.query, keep_blank_values=True) | (params or {}))
    path_parts = [part for part in parsed.path.split('/') if part]
    if path_parts and path_parts[0] == 'v1':
        path_parts = path_parts[1:]
    method = method.upper()

    if method == 'POST' and path_parts == ['search']:
        return search_objects(payload)
    if method == 'GET' and path_parts == ['users']:
        return list_users(payload)
    if method == 'GET' and path_parts == ['users', 'me']:
        return get_self()
    if method == 'GET' and len(path_parts) == 2 and path_parts[0] == 'users':
        return get_user(path_parts[1])
    if method == 'GET' and len(path_parts) == 2 and path_parts[0] == 'pages':
        return get_page(path_parts[1])
    if method == 'PATCH' and len(path_parts) == 2 and path_parts[0] == 'pages':
        return update_page(path_parts[1], payload)
    if method == 'POST' and path_parts == ['pages']:
        return create_page(payload)
    if method == 'GET' and len(path_parts) == 4 and path_parts[0] == 'pages' and path_parts[2] == 'properties':
        return get_page_property(path_parts[1], path_parts[3])
    if method == 'GET' and len(path_parts) == 2 and path_parts[0] == 'databases':
        return get_database(path_parts[1])
    if method == 'PATCH' and len(path_parts) == 2 and path_parts[0] == 'databases':
        return update_database(path_parts[1], payload)
    if method == 'POST' and len(path_parts) == 3 and path_parts[0] == 'databases' and path_parts[2] == 'query':
        return query_database(path_parts[1], payload)
    if method == 'POST' and path_parts == ['databases']:
        return create_database(payload)
    if method == 'GET' and len(path_parts) == 3 and path_parts[0] == 'blocks' and path_parts[2] == 'children':
        return get_block_children(path_parts[1], payload)
    if method == 'PATCH' and len(path_parts) == 3 and path_parts[0] == 'blocks' and path_parts[2] == 'children':
        return append_block_children(path_parts[1], payload)
    if method == 'GET' and len(path_parts) == 2 and path_parts[0] == 'blocks':
        return get_block(path_parts[1])
    if method == 'PATCH' and len(path_parts) == 2 and path_parts[0] == 'blocks':
        return update_block(path_parts[1], payload)
    if method == 'DELETE' and len(path_parts) == 2 and path_parts[0] == 'blocks':
        return delete_block(path_parts[1])
    if method == 'GET' and path_parts == ['comments']:
        block_id = str(payload.get('block_id') or '').strip()
        if not block_id:
            raise NotionStoreError(400, 'comments GET requires block_id')
        return list_comments(block_id, payload)
    if method == 'GET' and len(path_parts) == 2 and path_parts[0] == 'comments':
        return get_comment(path_parts[1])
    if method == 'POST' and path_parts == ['comments']:
        return create_comment(payload)
    raise NotionStoreError(404, f'Unsupported Notion endpoint: {method} {parsed.path}')


__all__ = [
    'NOTION_VERSION',
    'NotionStoreError',
    'append_block_children',
    'create_comment',
    'create_database',
    'create_page',
    'delete_block',
    'duplicate_eval_page',
    'ensure_source_template',
    'get_block',
    'get_block_children',
    'get_comment',
    'get_database',
    'get_page',
    'get_page_property',
    'get_self',
    'get_user',
    'handle_notion_http_request',
    'list_comments',
    'list_users',
    'mutate_notion_state',
    'notion_store_path',
    'query_database',
    'read_notion_state',
    'search_objects',
    'update_block',
    'update_database',
    'update_page',
]
