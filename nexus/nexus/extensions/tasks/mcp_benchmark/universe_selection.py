from __future__ import annotations

import argparse
import json
import os

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .credentials import WorkspaceKeys
from .manifests import load_universe_manifest_rows
from .mode import BenchmarkServerMode
from .support import decide_universe


UNIVERSE_MODE_ALL = 'all'
UNIVERSE_MODE_ORIGINAL = 'original'
UNIVERSE_CONTAINS_MATCH_ALL = 'all'
UNIVERSE_CONTAINS_MATCH_ANY = 'any'

UNIVERSE_MODE_ENV = 'MCP_BENCHMARK_UNIVERSE_MODE'
UNIVERSE_CONTAINS_SERVERS_ENV = 'MCP_BENCHMARK_UNIVERSE_CONTAINS_SERVERS'
UNIVERSE_CONTAINS_MATCH_ENV = 'MCP_BENCHMARK_UNIVERSE_CONTAINS_MATCH'
UNIVERSE_ALLOWED_SERVERS_ENV = 'MCP_BENCHMARK_UNIVERSE_ALLOWED_SERVERS'
UNIVERSE_ALLOWED_KEYED_SERVERS_ENV = 'MCP_BENCHMARK_UNIVERSE_ALLOWED_KEYED_SERVERS'
UNIVERSE_ALLOWED_API_SERVERS_ENV = 'MCP_BENCHMARK_UNIVERSE_ALLOWED_API_SERVERS'

UNIVERSE_KEY_REQUIREMENT_LABELS = {
    'github': ('github_token',),
    'google-maps': ('google_cloud_console_api_key',),
    'google-search': ('serper_api_key',),
    'notion': (
        'notion_integration_key',
        'notion_integration_key_eval',
        'source_notion_page_url',
        'eval_notion_page_url',
    ),
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
        labels.extend(UNIVERSE_KEY_REQUIREMENT_LABELS.get(server, ()))
    return _dedupe_keep_order(labels)


@dataclass(frozen=True, slots=True)
class UniverseSelectionFilters:
    mode: str = UNIVERSE_MODE_ALL
    contains_servers: tuple[str, ...] = ()
    contains_match: str = UNIVERSE_CONTAINS_MATCH_ALL
    allowed_servers: tuple[str, ...] = ()
    allowed_keyed_servers: tuple[str, ...] = ()
    limit: int | None = None

    def requested(self) -> bool:
        return any((
            self.mode != UNIVERSE_MODE_ALL,
            self.contains_servers,
            self.allowed_servers,
            self.allowed_keyed_servers,
            self.limit is not None,
        ))


@dataclass(frozen=True, slots=True)
class UniverseCaseInfo:
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
class UniverseApiSourceSummary:
    server_name: str
    key_requirement_labels: tuple[str, ...]
    case_count: int
    case_ratio: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def universe_mode_to_server_mode(mode: str) -> str:
    if mode == UNIVERSE_MODE_ORIGINAL:
        return BenchmarkServerMode.API.value
    raise ValueError(f'Unsupported Universe mode: {mode}')


def universe_selection_from_env() -> UniverseSelectionFilters | None:
    mode = (os.getenv(UNIVERSE_MODE_ENV) or UNIVERSE_MODE_ALL).strip() or UNIVERSE_MODE_ALL
    if mode not in {UNIVERSE_MODE_ALL, UNIVERSE_MODE_ORIGINAL}:
        raise ValueError(f'{UNIVERSE_MODE_ENV} must be one of: all, original')

    contains_match = (os.getenv(UNIVERSE_CONTAINS_MATCH_ENV) or UNIVERSE_CONTAINS_MATCH_ALL).strip()
    contains_match = contains_match or UNIVERSE_CONTAINS_MATCH_ALL
    if contains_match not in {UNIVERSE_CONTAINS_MATCH_ALL, UNIVERSE_CONTAINS_MATCH_ANY}:
        raise ValueError(f'{UNIVERSE_CONTAINS_MATCH_ENV} must be one of: all, any')

    filters = UniverseSelectionFilters(
        mode=mode,
        contains_servers=_split_csv(os.getenv(UNIVERSE_CONTAINS_SERVERS_ENV)),
        contains_match=contains_match,
        allowed_servers=_split_csv(os.getenv(UNIVERSE_ALLOWED_SERVERS_ENV)),
        allowed_keyed_servers=_split_csv(
            os.getenv(UNIVERSE_ALLOWED_KEYED_SERVERS_ENV) or os.getenv(UNIVERSE_ALLOWED_API_SERVERS_ENV)
        ),
    )
    return filters if filters.requested() else None


def universe_case_infos(keys: WorkspaceKeys | None = None) -> tuple[UniverseCaseInfo, ...]:
    workspace_keys = WorkspaceKeys() if keys is None else keys
    rows: list[UniverseCaseInfo] = []

    for row in load_universe_manifest_rows():
        task_data = row['task_data']
        prompt = row['prompt']
        server_names = tuple(row['server_names'])
        key_required_servers = tuple(server for server in server_names if server in UNIVERSE_KEY_REQUIREMENT_LABELS)
        original_decision = decide_universe(
            task_data, list(server_names), workspace_keys, server_mode=BenchmarkServerMode.API
        )
        rows.append(
            UniverseCaseInfo(
                case_id=row['case_id'],
                prompt=prompt,
                prompt_excerpt=_prompt_excerpt(prompt),
                source_path=row['source_path'],
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
    if mode == UNIVERSE_CONTAINS_MATCH_ANY:
        return bool(server_set & value_set)
    return value_set.issubset(server_set)


def _allowed_subset(server_names: tuple[str, ...], allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    return set(server_names).issubset(set(allowed))


def select_universe_case_infos(
    filters: UniverseSelectionFilters | None = None,
    *,
    keys: WorkspaceKeys | None = None,
) -> tuple[UniverseCaseInfo, ...]:
    selected_filters = filters or UniverseSelectionFilters()
    rows = universe_case_infos(keys)
    selected: list[UniverseCaseInfo] = []

    for row in rows:
        if selected_filters.mode == UNIVERSE_MODE_ORIGINAL and not row.original_supported_with_current_keys:
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


def select_universe_case_ids(
    filters: UniverseSelectionFilters | None = None,
    *,
    keys: WorkspaceKeys | None = None,
) -> tuple[str, ...]:
    return tuple(row.case_id for row in select_universe_case_infos(filters, keys=keys))


def universe_summary(keys: WorkspaceKeys | None = None) -> dict[str, Any]:
    rows = universe_case_infos(keys)
    return {
        'total_cases': len(rows),
        'original_supported_cases': sum(1 for row in rows if row.original_supported_with_current_keys),
    }


def universe_api_source_summaries(keys: WorkspaceKeys | None = None) -> tuple[UniverseApiSourceSummary, ...]:
    rows = universe_case_infos(keys)
    total_cases = len(rows)
    counts: Counter[str] = Counter()

    for row in rows:
        for server_name in row.key_required_servers:
            counts[server_name] += 1

    summaries = [
        UniverseApiSourceSummary(
            server_name=server_name,
            key_requirement_labels=UNIVERSE_KEY_REQUIREMENT_LABELS.get(server_name, ()),
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


def render_universe_matrix_markdown(rows: tuple[UniverseCaseInfo, ...]) -> str:
    table = _markdown_table(
        [
            'case_id',
            '题目摘要',
            '依赖 MCP 服务',
            '需要额外凭据的服务',
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
                '<br>'.join(row.original_reasons) or '-',
            ]
            for row in rows
        ],
    )
    return '\n'.join([
        '# Universe 题目 API / 凭据依赖矩阵',
        '',
        '由 `uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection matrix` 生成。',
        '',
        '- `依赖 MCP 服务`: 这道题会加载的 MCP 服务集合。',
        '- `需要额外凭据的服务`: 在 `original` 模式下需要额外准备凭据或第三方账号配置的服务。',
        '',
        table,
    ])


def _api_source_table(rows: tuple[UniverseApiSourceSummary, ...]) -> str:
    return _markdown_table(
        ['API / 凭据来源', 'Universe 题量', '占比', '凭据字段名'],
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


def _query_table(rows: tuple[UniverseCaseInfo, ...]) -> str:
    return _markdown_table(
        [
            'case_id',
            '题目摘要',
            '依赖 MCP 服务',
            '需要额外凭据的服务',
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
    parser = argparse.ArgumentParser(description='Inspect Universe original selection state.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    summary_parser = subparsers.add_parser('summary', help='Show Universe summary.')
    summary_parser.add_argument('--output')

    query_parser = subparsers.add_parser('query', help='Query Universe cases.')
    query_parser.add_argument(
        '--mode',
        choices=(UNIVERSE_MODE_ALL, UNIVERSE_MODE_ORIGINAL),
        default=UNIVERSE_MODE_ALL,
    )
    query_parser.add_argument('--contains-servers', default='')
    query_parser.add_argument(
        '--contains-match',
        choices=(UNIVERSE_CONTAINS_MATCH_ALL, UNIVERSE_CONTAINS_MATCH_ANY),
        default=UNIVERSE_CONTAINS_MATCH_ALL,
    )
    query_parser.add_argument('--allowed-servers', default='')
    query_parser.add_argument('--allowed-keyed-servers', default='')
    query_parser.add_argument('--limit', type=int)
    query_parser.add_argument('--format', choices=('case-ids', 'json', 'table'), default='table')
    query_parser.add_argument('--output')

    matrix_parser = subparsers.add_parser('matrix', help='Render the full Universe matrix.')
    matrix_parser.add_argument('--output')

    api_counts_parser = subparsers.add_parser('api-counts', help='Show Universe API / credential counts.')
    api_counts_parser.add_argument('--format', choices=('json', 'table'), default='table')
    api_counts_parser.add_argument('--output')

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == 'summary':
        _write_output(json.dumps(universe_summary(), ensure_ascii=False, indent=2), args.output)
        return 0

    if args.command == 'query':
        filters = UniverseSelectionFilters(
            mode=args.mode,
            contains_servers=_split_csv(args.contains_servers),
            contains_match=args.contains_match,
            allowed_servers=_split_csv(args.allowed_servers),
            allowed_keyed_servers=_split_csv(args.allowed_keyed_servers),
            limit=args.limit,
        )
        rows = select_universe_case_infos(filters)
        if args.format == 'case-ids':
            _write_output(','.join(row.case_id for row in rows), args.output)
            return 0
        if args.format == 'json':
            _write_output(json.dumps([row.to_json() for row in rows], ensure_ascii=False, indent=2), args.output)
            return 0
        _write_output(_query_table(rows), args.output)
        return 0

    if args.command == 'matrix':
        _write_output(render_universe_matrix_markdown(universe_case_infos()), args.output)
        return 0

    if args.command == 'api-counts':
        rows = universe_api_source_summaries()
        if args.format == 'json':
            _write_output(json.dumps([row.to_json() for row in rows], ensure_ascii=False, indent=2), args.output)
            return 0
        _write_output(_api_source_table(rows), args.output)
        return 0

    parser.error(f'Unsupported command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
