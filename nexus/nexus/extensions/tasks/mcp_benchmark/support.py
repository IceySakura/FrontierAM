from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mode import BenchmarkServerMode


TOOLATHLON_SUPPORTED_SERVERS = {
    'arxiv-latex',
    'arxiv_local',
    'canvas',
    'emails',
    'excel',
    'fetch',
    'filesystem',
    'git',
    'github',
    'google-cloud',
    'google_calendar',
    'google_forms',
    'google_map',
    'google_sheet',
    'howtocook',
    'huggingface',
    'k8s',
    'memory',
    'notion',
    'notion_official',
    'pdf-tools',
    'playwright_with_chunk',
    'pptx',
    'rail_12306',
    'scholarly',
    'snowflake',
    'terminal',
    'time',
    'wandb',
    'woocommerce',
    'word',
    'yahoo-finance',
    'youtube',
    'youtube-transcript',
}

TOOLATHLON_UNSUPPORTED_REASONS: dict[str, str] = {}

ATLAS_SUPPORTED_SERVERS = {
    'airtable',
    'alchemy',
    'anili',
    'arxiv',
    'balldontlie',
    'brave-search',
    'calculator',
    'cli-mcp-server',
    'clinicaltrialsgov-mcp-server',
    'context7',
    'ddg-search',
    'desktop-commander',
    'e2b-server',
    'exa',
    'f1-mcp-server',
    'fetch',
    'filesystem',
    'git',
    'github',
    'google-maps',
    'google-workspace',
    'lara-translate',
    'memory',
    'mcp-code-executor',
    'mcp-server-code-runner',
    'met-museum',
    'mongodb',
    'national-parks',
    'notion',
    'open-library',
    'osm-mcp-server',
    'oxylabs',
    'pubmed',
    'rijksmuseum-server',
    'slack',
    'twelvedata',
    'weather',
    'weather-data',
    'whois',
    'wikipedia',
}

ATLAS_API_KEY_REQUIREMENTS = {
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
    'notion': ('notion_integration_key_eval', 'notion_integration_key'),
    'oxylabs': ('oxylabs_username', 'oxylabs_password'),
    'slack': ('slack_mcp_xoxc_token', 'slack_mcp_xoxd_token'),
    'twelvedata': ('twelve_data_api_key',),
    'weather-data': ('weather_api_key',),
}

UNIVERSE_SUPPORTED_SERVERS = {
    'blender',
    'calculator',
    'date',
    'echo',
    'fetch',
    'filesystem',
    'github',
    'google-maps',
    'google-search',
    'notion',
    'playwright',
    'weather',
    'wikipedia',
    'yfinance',
}

UNIVERSE_UNSUPPORTED_REASONS = {
    'google-sheets': 'missing Google service-account credentials',
}


@dataclass(slots=True)
class SupportDecision:
    supported: bool
    reasons: list[str] = field(default_factory=list)


def _missing_all(keys: Any, *names: str) -> bool:
    return not any(keys.get(name) for name in names)


def _is_expected_toolathlon_override_error(error: str) -> bool:
    markers = ('folder_id.txt', 'sheet_id.txt', 'drive_url.txt')
    return any(marker in error for marker in markers)


def decide_toolathlon(
    row: dict[str, Any],
    keys: Any,
    *,
    server_mode: BenchmarkServerMode,
) -> SupportDecision:
    task_config = row['task_config']
    case_id = str(row.get('case_id') or 'toolathlon-case')
    reasons: list[str] = []
    servers = task_config.get('needed_mcp_servers', [])
    override_error = str(row.get('task_override_error') or '').strip()
    if override_error and not _is_expected_toolathlon_override_error(override_error):
        reasons.append(f'token_key_session.py failed to load: {override_error}')
    merged_task_keys = keys.merged_toolathlon_raw(
        case_root=f'/tmp/nexus-mcp-benchmark/{case_id}',
        task_override_values=dict(row.get('task_override_values') or {}),
        task_source_path=str(row.get('source_path') or ''),
        embedded_case_paths=set((row.get('embedded_case_files') or {}).keys()),
    )

    for server in servers:
        if server in TOOLATHLON_UNSUPPORTED_REASONS:
            reasons.append(f'{server}: {TOOLATHLON_UNSUPPORTED_REASONS[server]}')
            continue
        if server not in TOOLATHLON_SUPPORTED_SERVERS:
            reasons.append(f'{server}: not supported')
            continue
        if server == 'canvas' and server_mode == BenchmarkServerMode.API:
            missing = [name for name in ('canvas_api_token', 'canvas_domain') if not merged_task_keys.get(name)]
            if missing:
                reasons.append(f'canvas: missing value(s) {", ".join(missing)}')
        elif server == 'woocommerce' and server_mode == BenchmarkServerMode.API:
            missing = [
                name
                for name in ('woocommerce_site_url', 'woocommerce_api_key', 'woocommerce_api_secret')
                if not merged_task_keys.get(name)
            ]
            if missing:
                reasons.append(f'woocommerce: missing value(s) {", ".join(missing)}')
        if (
            server == 'google-cloud'
            and server_mode == BenchmarkServerMode.API
            and (not keys.get('gcp_project_id') or not keys.has_gcp_service_account_file())
        ):
            missing: list[str] = []
            if not keys.get('gcp_project_id'):
                missing.append('gcp_project_id')
            if not keys.has_gcp_service_account_file():
                missing.append('gcp_service_account_file')
            reasons.append(f'google-cloud: missing value(s) {", ".join(missing)}')
        if (
            server == 'google_map'
            and server_mode == BenchmarkServerMode.API
            and not keys.get('google_cloud_console_api_key')
        ):
            reasons.append('google_map: missing Google Maps API key')
        if (
            server == 'google_calendar'
            and server_mode == BenchmarkServerMode.API
            and not keys.has_google_oauth_credentials_file()
        ):
            reasons.append('google_calendar: missing Google OAuth credential file')
        if (
            server == 'google_forms'
            and server_mode == BenchmarkServerMode.API
            and not keys.has_google_oauth_credentials_file()
        ):
            reasons.append('google_forms: missing Google OAuth credential file')
        if (
            server == 'google_sheet'
            and server_mode == BenchmarkServerMode.API
            and not keys.has_google_oauth_credentials_file()
        ):
            reasons.append('google_sheet: missing Google OAuth credential file')
        if (
            server == 'snowflake'
            and server_mode == BenchmarkServerMode.API
            and not keys.has_snowflake_private_key_file()
        ):
            reasons.append('snowflake: missing Snowflake private key file')
        if (
            server == 'youtube'
            and server_mode == BenchmarkServerMode.API
            and not keys.get('google_cloud_console_api_key')
        ):
            reasons.append('youtube: missing Google YouTube API key')
        if server == 'github' and not keys.get('github_token'):
            reasons.append('github: missing key github_token')
        elif server == 'huggingface' and not keys.get('huggingface_token'):
            reasons.append('huggingface: missing key huggingface_token')
        elif server == 'wandb' and not keys.get('wandb_api_key'):
            reasons.append('wandb: missing key wandb_api_key')
        elif server == 'notion':
            if server_mode == BenchmarkServerMode.API:
                if not (keys.get('notion_integration_key_eval') or keys.get('notion_integration_key')):
                    reasons.append('notion: missing Notion API key')
                elif not keys.has_any_valid_notion_token():
                    reasons.append('notion: Notion API key is invalid')
            if server_mode == BenchmarkServerMode.API and not merged_task_keys.get('notion_allowed_page_ids'):
                reasons.append('notion: missing notion_allowed_page_ids task override')

    if (
        server_mode == BenchmarkServerMode.API
        and row.get('references_google_creds')
        and not keys.has_google_oauth_credentials_file()
    ):
        reasons.append('task script requires local Google OAuth credential file')
    if row.get('references_snowflake_private_key') and not keys.has_snowflake_private_key_file():
        reasons.append('task script requires local Snowflake private key file')
    if server_mode != BenchmarkServerMode.API and row.get('references_notion_duplication'):
        if not keys.has_valid_notion_token():
            reasons.append('task script requires valid Notion source token')
        if not keys.has_valid_notion_eval_token():
            reasons.append('task script requires valid Notion eval token')

    if not row.get('has_evaluation_main'):
        reasons.append('missing evaluation/main.py')

    return SupportDecision(not reasons, reasons)


def _atlas_api_reasons(server: str, keys: Any) -> list[str]:
    reasons: list[str] = []
    required = ATLAS_API_KEY_REQUIREMENTS.get(server)
    if required:
        if server == 'notion':
            if _missing_all(keys, *required):
                reasons.append('notion: missing Notion API key')
            elif not keys.has_any_valid_notion_token():
                reasons.append('notion: Notion API key is invalid')
            return reasons
        missing = [name for name in required if not keys.get(name)]
        if missing:
            reasons.append(f'{server}: missing key(s) {", ".join(missing)}')
    return reasons


def decide_atlas(server_names: list[str], keys: Any, *, server_mode: BenchmarkServerMode) -> SupportDecision:
    reasons: list[str] = []
    for server in server_names:
        if server not in ATLAS_SUPPORTED_SERVERS:
            reasons.append(f'{server}: not supported in {server_mode.value} mode')
            continue
        reasons.extend(_atlas_api_reasons(server, keys))
    return SupportDecision(not reasons, reasons)


def decide_universe(
    task_data: dict[str, Any],
    server_names: list[str],
    keys: Any,
    *,
    server_mode: BenchmarkServerMode,
) -> SupportDecision:
    reasons: list[str] = []

    if task_data.get('prepares'):
        reasons.append('task requires prepare hooks that are not wired into this benchmark task')

    for server in server_names:
        if server == 'google-search' and server_mode == BenchmarkServerMode.API and not keys.get('serper_api_key'):
            reasons.append('google-search: missing Serper API key')
            continue
        if (
            server == 'google-maps'
            and server_mode == BenchmarkServerMode.API
            and not keys.get('google_cloud_console_api_key')
        ):
            reasons.append('google-maps: missing Google Maps API key')
            continue
        if server in UNIVERSE_UNSUPPORTED_REASONS:
            reasons.append(f'{server}: {UNIVERSE_UNSUPPORTED_REASONS[server]}')
            continue
        if server not in UNIVERSE_SUPPORTED_SERVERS:
            reasons.append(f'{server}: not supported')
            continue
        if server == 'github':
            if not keys.get('github_token'):
                reasons.append('github: missing key github_token')
            if not keys.github_account_name():
                reasons.append('github: unable to derive GITHUB_PERSONAL_ACCOUNT_NAME')
        elif server == 'notion':
            if server_mode == BenchmarkServerMode.API:
                if not (keys.get('notion_integration_key_eval') or keys.get('notion_integration_key')):
                    reasons.append('notion: missing Notion API key')
                elif not keys.has_any_valid_notion_token():
                    reasons.append('notion: Notion API key is invalid')
                if not keys.notion_root_page():
                    reasons.append('notion: unable to derive NOTION_ROOT_PAGE')

    return SupportDecision(not reasons, reasons)
