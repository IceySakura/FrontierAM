from __future__ import annotations

import argparse
import json
import os
import re

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .credentials import WorkspaceKeys
from .manifests import load_atlas_manifest_rows
from .mode import BenchmarkServerMode
from .support import ATLAS_API_KEY_REQUIREMENTS, decide_atlas


ATLAS_MODE_ALL = 'all'
ATLAS_MODE_ORIGINAL = 'original'
ATLAS_CONTAINS_MATCH_ALL = 'all'
ATLAS_CONTAINS_MATCH_ANY = 'any'

ATLAS_MODE_ENV = 'MCP_BENCHMARK_ATLAS_MODE'
ATLAS_CONTAINS_SERVERS_ENV = 'MCP_BENCHMARK_ATLAS_CONTAINS_SERVERS'
ATLAS_CONTAINS_MATCH_ENV = 'MCP_BENCHMARK_ATLAS_CONTAINS_MATCH'
ATLAS_ALLOWED_SERVERS_ENV = 'MCP_BENCHMARK_ATLAS_ALLOWED_SERVERS'
ATLAS_ALLOWED_KEYED_SERVERS_ENV = 'MCP_BENCHMARK_ATLAS_ALLOWED_KEYED_SERVERS'
ATLAS_ALLOWED_API_SERVERS_ENV = 'MCP_BENCHMARK_ATLAS_ALLOWED_API_SERVERS'

ATLAS_KEY_REQUIREMENT_LABELS = {
    'airtable': ('airtable_api_key',),
    'alchemy': ('alchemy_api_key',),
    'brave-search': ('brave_api_key',),
    'e2b-server': ('e2b_api_key',),
    'exa': ('exa_api_key',),
    'github': ('github_token',),
    'google-maps': ('google_cloud_console_api_key',),
    'google-workspace': ('google_client_id', 'google_client_secret', 'google_refresh_token'),
    'lara-translate': ('lara_access_key_id', 'lara_access_key_secret'),
    'mongodb': ('mongodb_connection_string',),
    'national-parks': ('nps_api_key',),
    'notion': ('notion_integration_key_eval|notion_integration_key',),
    'oxylabs': ('oxylabs_username', 'oxylabs_password'),
    'slack': ('slack_mcp_xoxc_token', 'slack_mcp_xoxd_token'),
    'twelvedata': ('twelve_data_api_key',),
    'weather-data': ('weather_api_key',),
}


def _split_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(',') if item.strip())


def _dedupe_keep_order(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in values if item))


def _prompt_excerpt(prompt: str, *, limit: int = 80) -> str:
    normalized = ' '.join(prompt.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + '...'


def _flatten_key_requirements(servers: tuple[str, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for server in servers:
        labels.extend(ATLAS_KEY_REQUIREMENT_LABELS.get(server, ()))
    return _dedupe_keep_order(labels)


@dataclass(frozen=True, slots=True)
class AtlasSelectionFilters:
    mode: str = ATLAS_MODE_ALL
    contains_servers: tuple[str, ...] = ()
    contains_match: str = ATLAS_CONTAINS_MATCH_ALL
    allowed_servers: tuple[str, ...] = ()
    allowed_keyed_servers: tuple[str, ...] = ()
    limit: int | None = None

    def requested(self) -> bool:
        return any((
            self.mode != ATLAS_MODE_ALL,
            self.contains_servers,
            self.allowed_servers,
            self.allowed_keyed_servers,
            self.limit is not None,
        ))


@dataclass(frozen=True, slots=True)
class AtlasCaseInfo:
    case_id: str
    prompt: str
    prompt_excerpt: str
    source_path: str
    server_names: tuple[str, ...]
    key_required_servers: tuple[str, ...]
    key_requirement_labels: tuple[str, ...]
    original_supported_with_current_keys: bool
    original_reasons: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AtlasApiSourceSummary:
    server_name: str
    key_requirement_labels: tuple[str, ...]
    case_count: int
    case_ratio: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def atlas_mode_to_server_mode(mode: str) -> str:
    if mode == ATLAS_MODE_ORIGINAL:
        return BenchmarkServerMode.API.value
    raise ValueError(f'Unsupported Atlas mode: {mode}')


def atlas_selection_from_env() -> AtlasSelectionFilters | None:
    mode = (os.getenv(ATLAS_MODE_ENV) or ATLAS_MODE_ALL).strip() or ATLAS_MODE_ALL
    if mode not in {ATLAS_MODE_ALL, ATLAS_MODE_ORIGINAL}:
        raise ValueError(f'{ATLAS_MODE_ENV} must be one of: all, original')

    contains_match = (
        os.getenv(ATLAS_CONTAINS_MATCH_ENV) or ATLAS_CONTAINS_MATCH_ALL
    ).strip() or ATLAS_CONTAINS_MATCH_ALL
    if contains_match not in {ATLAS_CONTAINS_MATCH_ALL, ATLAS_CONTAINS_MATCH_ANY}:
        raise ValueError(f'{ATLAS_CONTAINS_MATCH_ENV} must be one of: all, any')

    filters = AtlasSelectionFilters(
        mode=mode,
        contains_servers=_split_csv(os.getenv(ATLAS_CONTAINS_SERVERS_ENV)),
        contains_match=contains_match,
        allowed_servers=_split_csv(os.getenv(ATLAS_ALLOWED_SERVERS_ENV)),
        allowed_keyed_servers=_split_csv(
            os.getenv(ATLAS_ALLOWED_KEYED_SERVERS_ENV) or os.getenv(ATLAS_ALLOWED_API_SERVERS_ENV)
        ),
    )
    return filters if filters.requested() else None


def atlas_case_infos(keys: WorkspaceKeys | None = None) -> tuple[AtlasCaseInfo, ...]:
    workspace_keys = WorkspaceKeys() if keys is None else keys
    rows: list[AtlasCaseInfo] = []

    for record in load_atlas_manifest_rows():
        server_names = tuple(record['server_names'])
        key_required_servers = tuple(server for server in server_names if server in ATLAS_API_KEY_REQUIREMENTS)
        original_decision = decide_atlas(list(server_names), workspace_keys, server_mode=BenchmarkServerMode.API)
        rows.append(
            AtlasCaseInfo(
                case_id=record['case_id'],
                prompt=record['prompt'],
                prompt_excerpt=_prompt_excerpt(record['prompt']),
                source_path=record['source_path'],
                server_names=server_names,
                key_required_servers=key_required_servers,
                key_requirement_labels=_flatten_key_requirements(key_required_servers),
                original_supported_with_current_keys=original_decision.supported,
                original_reasons=tuple(original_decision.reasons),
            )
        )

    rows.sort(key=lambda item: item.case_id)
    return tuple(rows)


def _contains_match(server_names: tuple[str, ...], values: tuple[str, ...], *, mode: str) -> bool:
    if not values:
        return True
    server_set = set(server_names)
    value_set = set(values)
    if mode == ATLAS_CONTAINS_MATCH_ANY:
        return bool(server_set & value_set)
    return value_set.issubset(server_set)


def _allowed_subset(server_names: tuple[str, ...], allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    return set(server_names).issubset(set(allowed))


def select_atlas_case_infos(
    filters: AtlasSelectionFilters | None = None,
    *,
    keys: WorkspaceKeys | None = None,
) -> tuple[AtlasCaseInfo, ...]:
    selected_filters = filters or AtlasSelectionFilters()
    rows = atlas_case_infos(keys)
    selected: list[AtlasCaseInfo] = []

    for row in rows:
        if selected_filters.mode == ATLAS_MODE_ORIGINAL and not row.original_supported_with_current_keys:
            continue
        if not _contains_match(
            row.server_names, selected_filters.contains_servers, mode=selected_filters.contains_match
        ):
            continue
        if not _allowed_subset(row.server_names, selected_filters.allowed_servers):
            continue
        if not _allowed_subset(row.key_required_servers, selected_filters.allowed_keyed_servers):
            continue
        selected.append(row)
        if selected_filters.limit is not None and len(selected) >= selected_filters.limit:
            break

    return tuple(selected)


def select_atlas_case_ids(
    filters: AtlasSelectionFilters | None = None,
    *,
    keys: WorkspaceKeys | None = None,
) -> tuple[str, ...]:
    return tuple(row.case_id for row in select_atlas_case_infos(filters, keys=keys))


def atlas_summary(keys: WorkspaceKeys | None = None) -> dict[str, Any]:
    rows = atlas_case_infos(keys)
    return {
        'total_cases': len(rows),
        'original_supported_cases': sum(1 for row in rows if row.original_supported_with_current_keys),
    }


def atlas_api_source_summaries(keys: WorkspaceKeys | None = None) -> tuple[AtlasApiSourceSummary, ...]:
    rows = atlas_case_infos(keys)
    total_cases = len(rows)
    counts: Counter[str] = Counter()

    for row in rows:
        for server_name in row.key_required_servers:
            counts[server_name] += 1

    summaries = [
        AtlasApiSourceSummary(
            server_name=server_name,
            key_requirement_labels=ATLAS_KEY_REQUIREMENT_LABELS.get(server_name, ()),
            case_count=case_count,
            case_ratio=(case_count / total_cases) if total_cases else 0.0,
        )
        for server_name, case_count in counts.items()
    ]
    summaries.sort(key=lambda item: (-item.case_count, item.server_name))
    return tuple(summaries)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _format_support_bool(value: bool) -> str:
    return '是' if value else '否'


def _localize_reason(reason: str) -> str:
    if match := re.fullmatch(r'([^:]+): missing key\(s\) (.+)', reason):
        server_name, missing_keys = match.groups()
        return f'{server_name}: 缺少凭据字段 {missing_keys}'

    if match := re.fullmatch(r'([^:]+): missing Notion API key', reason):
        server_name = match.group(1)
        return f'{server_name}: 缺少 Notion API key'

    if match := re.fullmatch(r'([^:]+): Notion API key is invalid', reason):
        server_name = match.group(1)
        return f'{server_name}: Notion API key 无效'

    if match := re.fullmatch(r'([^:]+): not supported in api mode', reason):
        server_name = match.group(1)
        return f'{server_name}: 当前不支持 original/API 链路'

    return reason


def _localize_reasons(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return '-'
    return '<br>'.join(_localize_reason(reason) for reason in reasons)


def render_atlas_matrix_markdown(rows: tuple[AtlasCaseInfo, ...]) -> str:
    table = _markdown_table(
        [
            'case_id',
            '题目摘要',
            '依赖 MCP 服务',
            '需要私有凭据的服务',
            '凭据字段名',
            'original 可跑',
            'original 说明',
        ],
        [
            [
                row.case_id,
                row.prompt_excerpt.replace('|', '\\|'),
                '<br>'.join(row.server_names) or '-',
                '<br>'.join(row.key_required_servers) or '-',
                '<br>'.join(row.key_requirement_labels) or '-',
                _format_support_bool(row.original_supported_with_current_keys),
                _localize_reasons(row.original_reasons),
            ]
            for row in rows
        ],
    )
    return '\n'.join([
        '# Atlas 题目 API 依赖矩阵',
        '',
        '由 `uv run python -m nexus.extensions.tasks.mcp_benchmark.atlas_selection matrix` 生成。',
        '',
        '- `依赖 MCP 服务`: 这道题会加载的 MCP 服务集合。',
        '- `需要私有凭据的服务`: 在 `original` 模式下需要用户自己准备凭据的服务。',
        '',
        table,
    ])


def _api_source_table(rows: tuple[AtlasApiSourceSummary, ...]) -> str:
    return _markdown_table(
        ['API 来源', 'Atlas 题量', '占比', '凭据字段名'],
        [
            [
                row.server_name,
                str(row.case_count),
                f'{row.case_ratio:.1%}',
                '<br>'.join(row.key_requirement_labels) or '-',
            ]
            for row in rows
        ],
    )


def _query_table(rows: tuple[AtlasCaseInfo, ...]) -> str:
    return _markdown_table(
        [
            'case_id',
            '题目摘要',
            '依赖 MCP 服务',
            '需要私有凭据的服务',
            'original 可跑',
        ],
        [
            [
                row.case_id,
                row.prompt_excerpt.replace('|', '\\|'),
                ','.join(row.server_names) or '-',
                ','.join(row.key_required_servers) or '-',
                _format_support_bool(row.original_supported_with_current_keys),
            ]
            for row in rows
        ],
    )


def _write_output(text: str, output: str | None) -> None:
    if output:
        path = os.path.abspath(output)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return
    print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspect Atlas original selection state.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    summary_parser = subparsers.add_parser('summary', help='Show Atlas summary.')
    summary_parser.add_argument('--output')

    query_parser = subparsers.add_parser('query', help='Query Atlas cases.')
    query_parser.add_argument('--mode', choices=(ATLAS_MODE_ALL, ATLAS_MODE_ORIGINAL), default=ATLAS_MODE_ALL)
    query_parser.add_argument('--contains-servers', default='')
    query_parser.add_argument(
        '--contains-match',
        choices=(ATLAS_CONTAINS_MATCH_ALL, ATLAS_CONTAINS_MATCH_ANY),
        default=ATLAS_CONTAINS_MATCH_ALL,
    )
    query_parser.add_argument('--allowed-servers', default='')
    query_parser.add_argument('--allowed-keyed-servers', default='')
    query_parser.add_argument('--limit', type=int)
    query_parser.add_argument('--format', choices=('case-ids', 'json', 'table'), default='table')
    query_parser.add_argument('--output')

    matrix_parser = subparsers.add_parser('matrix', help='Render the full Atlas matrix.')
    matrix_parser.add_argument('--output')

    api_counts_parser = subparsers.add_parser('api-counts', help='Show keyed Atlas API source counts.')
    api_counts_parser.add_argument('--format', choices=('json', 'table'), default='table')
    api_counts_parser.add_argument('--output')

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'summary':
        _write_output(json.dumps(atlas_summary(), ensure_ascii=False, indent=2), args.output)
        return 0

    if args.command == 'query':
        filters = AtlasSelectionFilters(
            mode=args.mode,
            contains_servers=_split_csv(args.contains_servers),
            contains_match=args.contains_match,
            allowed_servers=_split_csv(args.allowed_servers),
            allowed_keyed_servers=_split_csv(args.allowed_keyed_servers),
            limit=args.limit,
        )
        rows = select_atlas_case_infos(filters)
        if args.format == 'case-ids':
            _write_output(','.join(row.case_id for row in rows), args.output)
            return 0
        if args.format == 'json':
            _write_output(json.dumps([row.to_json() for row in rows], ensure_ascii=False, indent=2), args.output)
            return 0
        _write_output(_query_table(rows), args.output)
        return 0

    if args.command == 'matrix':
        _write_output(render_atlas_matrix_markdown(atlas_case_infos()), args.output)
        return 0

    if args.command == 'api-counts':
        rows = atlas_api_source_summaries()
        if args.format == 'json':
            _write_output(json.dumps([row.to_json() for row in rows], ensure_ascii=False, indent=2), args.output)
            return 0
        _write_output(_api_source_table(rows), args.output)
        return 0

    parser.error(f'Unsupported command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
