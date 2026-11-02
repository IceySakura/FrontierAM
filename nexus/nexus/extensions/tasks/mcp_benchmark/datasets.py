from __future__ import annotations

import base64
import copy
import gzip
import json
import os

from functools import cache
from typing import Any

from .constants import (
    ATLAS_MONGODB_SERVICE_GONGFENG_IMAGE_NAME,
    ATLAS_MONGODB_SERVICE_IMAGE_NAME,
    ATLAS_MONGODB_SERVICE_RUNTIME_NAME,
    BENCHMARK_IMAGE_HELPERS,
    BENCHMARK_IMAGE_NAME,
    BENCHMARK_IMAGE_NPM_ROOT,
    BENCHMARK_IMAGE_VENV_BIN,
    BENCHMARK_WORK_ROOT,
    TOOLATHLON_IMAGE_ROOT,
    TOOLATHLON_IMAGE_TASKS_ROOT,
    TOOLATHLON_IMAGE_VENV,
    TOOLATHLON_IMAGE_VENV_BIN,
    TOOLATHLON_SHIMS_ROOT,
    UNIVERSE_DATASET_ROOT,
)
from .credentials import WorkspaceKeys
from .manifests import load_atlas_manifest_rows, load_toolathlon_manifest_rows, load_universe_manifest_rows
from .mode import BenchmarkServerMode
from .server_specs import (
    load_atlas_server_specs,
    load_toolathlon_server_config_texts,
    load_toolathlon_server_specs,
    load_universe_server_specs,
    render_value,
)
from .support import decide_atlas, decide_toolathlon, decide_universe
from .types import (
    BenchmarkCase,
    BenchmarkCleanup,
    BenchmarkCommand,
    BenchmarkDataset,
    BenchmarkEvaluator,
    BenchmarkInputFile,
    BenchmarkService,
    BenchmarkServiceConnection,
)


TOOLATHLON_CANVAS_HTTP_PLACEHOLDER = '__NEXUS_TOOLATHLON_CANVAS_HTTP_BASE_URL__'
TOOLATHLON_CANVAS_HTTP_DOMAIN_PLACEHOLDER = '__NEXUS_TOOLATHLON_CANVAS_HTTP_DOMAIN__'
TOOLATHLON_CANVAS_HTTPS_PLACEHOLDER = '__NEXUS_TOOLATHLON_CANVAS_HTTPS_BASE_URL__'
TOOLATHLON_CANVAS_DOMAIN_PLACEHOLDER = '__NEXUS_TOOLATHLON_CANVAS_DOMAIN__'
TOOLATHLON_WOOCOMMERCE_BASE_PLACEHOLDER = '__NEXUS_TOOLATHLON_WOOCOMMERCE_BASE_URL__'
TOOLATHLON_EMAIL_IMAP_PLACEHOLDER = '__NEXUS_TOOLATHLON_EMAIL_IMAP__'
TOOLATHLON_EMAIL_SMTP_PLACEHOLDER = '__NEXUS_TOOLATHLON_EMAIL_SMTP__'
TOOLATHLON_CANVAS_SERVICE_RUNTIME_NAME = 'service_toolathlon_canvas'
TOOLATHLON_WOOCOMMERCE_SERVICE_RUNTIME_NAME = 'service_toolathlon_woocommerce'
TOOLATHLON_EMAIL_SERVICE_RUNTIME_NAME = 'service_toolathlon_email'


@cache
def _toolathlon_email_users_b64gz() -> str:
    """Return base64-encoded gzip of the minimal users JSON array.

    Format: ``[{"e": "user@mcp.com", "p": "pass", "n": "Full Name"}, ...]``
    Source: ``$NEXUS_MCP_BENCHMARK_TOOLATHLON_ROOT/configs/users_data.json``
    """
    toolathlon_root = os.getenv('NEXUS_MCP_BENCHMARK_TOOLATHLON_ROOT', '')
    users_path = os.path.join(toolathlon_root, 'configs', 'users_data.json')
    with open(users_path, encoding='utf-8') as f:
        data = json.load(f)
    minimal = [{'e': u['email'], 'p': u['password'], 'n': u['full_name']} for u in data['users']]
    compact = json.dumps(minimal, separators=(',', ':'))
    compressed = gzip.compress(compact.encode())
    return base64.b64encode(compressed).decode()


def _case_root(dataset: BenchmarkDataset, case_id: str, server_mode: BenchmarkServerMode) -> str:
    return f'{BENCHMARK_WORK_ROOT}/{dataset.value}/{server_mode.value}/{case_id}'


def _agent_workspace(dataset: BenchmarkDataset, case_id: str, server_mode: BenchmarkServerMode) -> str:
    return f'{_case_root(dataset, case_id, server_mode)}/agent_workspace'


def _toolathlon_service_image() -> str:
    return os.getenv('MCP_BENCHMARK_MAIN_IMAGE', BENCHMARK_IMAGE_NAME)


def _toolathlon_original_services(
    *,
    case_id: str,
    case_source_dir: str,
    server_names: list[str],
    token_values_json: str,
    server_mode: BenchmarkServerMode,
) -> tuple[list[BenchmarkService], dict[str, str]]:
    services: list[BenchmarkService] = []
    placeholder_replacements: dict[str, str] = {}
    image = _toolathlon_service_image()

    if 'canvas' in server_names:
        canvas_root = f'/workspace/mcp-benchmark-services/toolathlon/canvas/{case_id}'
        services.append(
            BenchmarkService(
                name='toolathlon-canvas',
                runtime_name=TOOLATHLON_CANVAS_SERVICE_RUNTIME_NAME,
                image=image,
                exposed_ports=[10001, 20001],
                startup=BenchmarkCommand(
                    command=(
                        "bash -lc 'set -euo pipefail; "
                        f'mkdir -p {canvas_root}; '
                        f'nohup {TOOLATHLON_IMAGE_VENV_BIN}/python {BENCHMARK_IMAGE_HELPERS}/toolathlon_canvas_api_server.py '
                        '--host 0.0.0.0 --ports 10001,20001 '
                        f'>{canvas_root}/server.log 2>&1 & '
                        'for i in $(seq 1 60); do '
                        'curl -fsS http://127.0.0.1:10001/healthz >/dev/null && exit 0; '
                        'sleep 1; '
                        'done; '
                        'echo "toolathlon canvas service failed to start" >&2; exit 1\''
                    ),
                    env={
                        'NEXUS_TOOLATHLON_CANVAS_STORE_PATH': f'{canvas_root}/store.json',
                        'NEXUS_TOOLATHLON_CASE_ROOT': case_source_dir,
                        'NEXUS_TOOLATHLON_TOKEN_VALUES_JSON': token_values_json,
                    },
                    timeout_s=120.0,
                ),
                ready_check=BenchmarkCommand(
                    command="bash -lc 'curl -fsS http://127.0.0.1:10001/healthz >/dev/null'",
                    timeout_s=30.0,
                ),
                ready_timeout_s=180.0,
                ready_poll_interval_s=2.0,
                connections=[
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_CANVAS_HTTP_PLACEHOLDER,
                        internal_port=10001,
                        value_template='http://{host}:{port}',
                    ),
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_CANVAS_HTTP_DOMAIN_PLACEHOLDER,
                        internal_port=10001,
                        value_template='{host}:{port}',
                    ),
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_CANVAS_HTTPS_PLACEHOLDER,
                        internal_port=20001,
                        value_template='https://{host}:{port}',
                    ),
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_CANVAS_DOMAIN_PLACEHOLDER,
                        internal_port=20001,
                        value_template='{host}:{port}',
                    ),
                ],
            )
        )
        placeholder_replacements.update({
            'http://localhost:10001': TOOLATHLON_CANVAS_HTTP_PLACEHOLDER,
            'https://localhost:20001': TOOLATHLON_CANVAS_HTTPS_PLACEHOLDER,
            'http://localhost:20001': TOOLATHLON_CANVAS_HTTPS_PLACEHOLDER,
            'localhost:20001': TOOLATHLON_CANVAS_DOMAIN_PLACEHOLDER,
            'localhost:10001': TOOLATHLON_CANVAS_HTTP_DOMAIN_PLACEHOLDER,
        })

    if 'woocommerce' in server_names:
        woocommerce_root = f'/workspace/mcp-benchmark-services/toolathlon/woocommerce/{case_id}'
        services.append(
            BenchmarkService(
                name='toolathlon-woocommerce',
                runtime_name=TOOLATHLON_WOOCOMMERCE_SERVICE_RUNTIME_NAME,
                image=image,
                exposed_ports=[10003],
                startup=BenchmarkCommand(
                    command=(
                        "bash -lc 'set -euo pipefail; "
                        f'mkdir -p {woocommerce_root}; '
                        f'nohup {TOOLATHLON_IMAGE_VENV_BIN}/python {BENCHMARK_IMAGE_HELPERS}/toolathlon_woocommerce_api_server.py '
                        '--host 0.0.0.0 --port 10003 '
                        f'>{woocommerce_root}/server.log 2>&1 & '
                        'for i in $(seq 1 60); do '
                        'curl -fsS http://127.0.0.1:10003/healthz >/dev/null && exit 0; '
                        'sleep 1; '
                        'done; '
                        'echo "toolathlon woocommerce service failed to start" >&2; exit 1\''
                    ),
                    env={
                        'NEXUS_TOOLATHLON_WOOCOMMERCE_STORE_PATH': f'{woocommerce_root}/store.json',
                        'NEXUS_TOOLATHLON_WOOCOMMERCE_MEDIA_DIR': f'{woocommerce_root}/media',
                    },
                    timeout_s=120.0,
                ),
                ready_check=BenchmarkCommand(
                    command="bash -lc 'curl -fsS http://127.0.0.1:10003/healthz >/dev/null'",
                    timeout_s=30.0,
                ),
                ready_timeout_s=180.0,
                ready_poll_interval_s=2.0,
                connections=[
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_WOOCOMMERCE_BASE_PLACEHOLDER,
                        internal_port=10003,
                        value_template='http://{host}:{port}',
                    )
                ],
            )
        )
        placeholder_replacements.update({
            'https://localhost:10003': TOOLATHLON_WOOCOMMERCE_BASE_PLACEHOLDER,
            'http://localhost:10003': TOOLATHLON_WOOCOMMERCE_BASE_PLACEHOLDER,
        })

    if 'emails' in server_names:
        # Poste.io BenchmarkService: real Dovecot IMAP + Haraka SMTP.
        # gongfeng replaces the container entrypoint with nexus-runtime,
        # so Poste.io's supervisord never runs.  We start services via
        # /etc/init.d/ and create users with php console commands — the
        # same approach as upstream Toolathlon's setup.sh + create_users.sh.
        # NOTE: Poste.io image has NO python3, only php/bash/nc.
        users_b64gz = _toolathlon_email_users_b64gz()

        # Startup: start Dovecot, configure plaintext auth, create users.
        # All bash/php — no python3 available in Poste.io image.
        # Uses heredoc-style via env var for the user list to avoid
        # embedding 500+ php commands in the startup string.
        poste_startup_script = (
            'bash -c '
            "$'"
            # NOTE: do NOT use set -e here — some init scripts and stop/start
            # commands exit non-zero in non-fatal ways (e.g. "already running").
            #
            # Phase 1: Run s6 cont-init.d scripts to create /data, users.db etc.
            'echo "=== Phase 1: cont-init.d ==="; '
            'for f in /etc/cont-init.d/*.sh; do bash "$f" 2>/dev/null || true; done; '
            'test -f /data/users.db || { echo "/data/users.db missing after cont-init" >&2; exit 1; }; '
            #
            # Phase 2: Configure plaintext auth BEFORE starting Dovecot
            'echo "=== Phase 2: auth config ==="; '
            'sed -i "s/ssl = required/ssl = yes/" /etc/dovecot/conf.d/10-ssl.conf 2>/dev/null || true; '
            'sed -i "s/auth_allow_cleartext = no/auth_allow_cleartext = yes/" /etc/dovecot/conf.d/10-auth.conf 2>/dev/null || true; '
            'sed -i "/disable_plaintext_auth/d" /etc/dovecot/conf.d/10-auth.conf 2>/dev/null || true; '
            #
            # Phase 3: Start Dovecot (kill any existing process first via pidfile)
            'echo "=== Phase 3: start Dovecot ==="; '
            'test -f /run/master.pid && kill $(cat /run/master.pid) 2>/dev/null; rm -f /run/master.pid; sleep 1; '
            '/etc/init.d/dovecot start; '
            'sleep 2; '
            'echo QUIT | nc -w2 127.0.0.1 143 | grep -q OK || '
            '{ echo "Dovecot not listening on 143" >&2; exit 1; }; '
            #
            # Phase 4: Configure + start Haraka SMTP
            'sed -i "s/tls_required = true/tls_required = false/" /opt/haraka-smtp/config/auth.ini 2>/dev/null || true; '
            'sed -i "s/tls_required = true/tls_required = false/" /opt/haraka-submission/config/auth.ini 2>/dev/null || true; '
            'sed -i "s/^auth\\\\/poste/#auth\\\\/poste/" /opt/haraka-submission/config/plugins 2>/dev/null || true; '
            'echo 0.0.0.0/0 > /opt/haraka-submission/config/relay_acl_allow 2>/dev/null || true; '
            'echo 0.0.0.0/0 > /opt/haraka-smtp/config/relay_acl_allow 2>/dev/null || true; '
            'cd /opt/haraka-smtp && node node_modules/.bin/haraka -c . &>/dev/null & '
            'cd /opt/haraka-submission && node node_modules/.bin/haraka -c . &>/dev/null & '
            'sleep 3; '
            #
            # Phase 5: Create domain + users via php console.
            # Poste.io php console requires UID 8 (mail user); gongfeng runs as
            # root, so we use runuser to switch.
            'echo "=== Phase 5: create users ==="; '
            'RU="runuser -u mail --"; '
            '$RU php /opt/admin/bin/console domain:create mcp.com 2>&1 || true; '
            '$RU php /opt/admin/bin/console email:create mcpposte_admin@mcp.com mcpposte "System Administrator" 2>&1 || true; '
            '$RU php /opt/admin/bin/console email:admin mcpposte_admin@mcp.com 2>&1 || true; '
            'echo "$NEXUS_EMAIL_USERS_B64" | base64 -d | gzip -d > /tmp/users.json; '
            'chmod 644 /tmp/users.json; '
            'php -r \\x27'
            '$users = json_decode(file_get_contents("/tmp/users.json"), true);'
            'echo count($users)." users to create\\n";'
            '$ok = 0; $fail = 0;'
            'foreach ($users as $u) {'
            '  $cmd = "runuser -u mail -- php /opt/admin/bin/console email:create ".escapeshellarg($u["e"])." ".escapeshellarg($u["p"])." ".escapeshellarg($u["n"])." 2>&1";'
            '  exec($cmd, $out, $rc);'
            '  if ($rc === 0) $ok++; else $fail++;'
            '}'
            'echo "Created: $ok, Failed: $fail\\n";'
            '\\x27; '
            #
            # Phase 6: Restart Dovecot so it picks up newly created users
            'echo "=== Phase 6: restart Dovecot ==="; '
            'test -f /run/master.pid && kill $(cat /run/master.pid) 2>/dev/null; rm -f /run/master.pid; sleep 1; '
            '/etc/init.d/dovecot start; sleep 2; '
            #
            # Phase 7: Verify IMAP login works
            'echo "=== Phase 7: verify IMAP login ==="; '
            'doveadm auth test mcpposte_admin@mcp.com mcpposte 2>&1; '
            'RESP=$(echo "a LOGIN mcpposte_admin@mcp.com mcpposte" | nc -w5 127.0.0.1 143 2>&1); '
            'echo "IMAP: $RESP"; '
            'echo "$RESP" | grep -q "a OK" || '
            '{ echo "IMAP LOGIN failed" >&2; exit 1; }; '
            'echo "Poste.io ready"'
            "'"
        )
        services.append(
            BenchmarkService(
                name='toolathlon-email',
                runtime_name=TOOLATHLON_EMAIL_SERVICE_RUNTIME_NAME,
                image='analogic/poste.io:2.5.5',
                exposed_ports=[143, 587],
                startup=BenchmarkCommand(
                    command=poste_startup_script,
                    env={'NEXUS_EMAIL_USERS_B64': users_b64gz},
                    timeout_s=600.0,
                ),
                ready_check=BenchmarkCommand(
                    command=(
                        'RESP=$(echo "a LOGIN mcpposte_admin@mcp.com mcpposte" | nc -w5 127.0.0.1 143 2>&1); '
                        'echo "IMAP ready_check: $RESP"; '
                        'echo "$RESP" | grep -q "a OK"'
                    ),
                    timeout_s=30.0,
                ),
                ready_timeout_s=600.0,
                ready_poll_interval_s=5.0,
                connections=[
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_EMAIL_IMAP_PLACEHOLDER,
                        internal_port=143,
                        value_template='{host}:{port}',
                    ),
                    BenchmarkServiceConnection(
                        placeholder=TOOLATHLON_EMAIL_SMTP_PLACEHOLDER,
                        internal_port=587,
                        value_template='{host}:{port}',
                    ),
                ],
            )
        )

    return services, placeholder_replacements


def _replace_text_fragments(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for source, target in replacements.items():
            rendered = rendered.replace(source, target)
        return rendered
    if isinstance(value, list):
        return [_replace_text_fragments(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_text_fragments(item, replacements) for key, item in value.items()}
    return value


def _toolathlon_notion_client_shim_file() -> BenchmarkInputFile:
    """Read the local (patched) notion_client shim and return it as an input file.

    The image-bundled version always uses the local store which lacks source-page
    data.  The patched version delegates to the real ``notion-client`` SDK when
    running in API mode so that preprocess scripts can call the live Notion API.
    """
    from pathlib import Path

    shim_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_shims' / 'notion_client.py'
    return BenchmarkInputFile(
        path=f'{TOOLATHLON_SHIMS_ROOT}/notion_client.py',
        content=shim_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_sitecustomize_shim_file() -> BenchmarkInputFile:
    """Upload the patched sitecustomize that includes the Notion API duplicate patch."""
    from pathlib import Path

    shim_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_shims' / 'sitecustomize.py'
    return BenchmarkInputFile(
        path=f'{TOOLATHLON_SHIMS_ROOT}/sitecustomize.py',
        content=shim_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_gspread_shim_file() -> BenchmarkInputFile:
    """Upload the patched gspread shim that delegates to the real gspread in API mode.

    The image-bundled version unconditionally uses MockGspreadClient regardless of
    server mode, which breaks eval scripts (e.g. vlm-history-completer) that call
    gspread.authorize() to read real spreadsheets when running in API mode.
    The patched version checks NEXUS_TOOLATHLON_SERVER_MODE and falls through to the
    real gspread package when not in mock mode.
    """
    from pathlib import Path

    shim_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_shims' / 'gspread.py'
    return BenchmarkInputFile(
        path=f'{TOOLATHLON_SHIMS_ROOT}/gspread.py',
        content=shim_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_emails_server_file() -> BenchmarkInputFile:
    """Upload the patched emails MCP server wrapper.

    The image-bundled version lacks the ``_patch_email_header_bug`` fix for
    the ``emails-mcp`` package's Header-not-converted-to-str bug.  Uploading
    the local version ensures the patch reaches the container.
    """
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_emails_server.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_emails_server.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_email_config_patcher_file() -> BenchmarkInputFile:
    """Upload the email config patcher script.

    The patcher rewrites ``email_config.json`` (and related files) under the
    case root to point IMAP/SMTP fields at the real Poste.io BenchmarkService
    host:port instead of the upstream default ``localhost:1143`` / ``localhost:1587``.
    """
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_email_config_patcher.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_email_config_patcher.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_google_sheet_server_file() -> BenchmarkInputFile:
    """Upload the patched Google Sheet MCP server with httplib2 CA fix."""
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_google_sheet_server.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_sheet_server.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_google_calendar_server_file() -> BenchmarkInputFile:
    """Upload the patched Google Calendar MCP server with httplib2 CA fix."""
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_google_calendar_server.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_calendar_server.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_google_forms_server_file() -> BenchmarkInputFile:
    """Upload the patched Google Forms MCP server with httplib2 CA fix."""
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_google_forms_server.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_google_forms_server.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_google_maps_server_file() -> BenchmarkInputFile:
    """Upload the Toolathlon eval-side Google Maps server.

    The uploaded helper is a dedicated Python MCP server that:
    1. Exposes ``maps_*`` tool names so Toolathlon eval scripts match directly.
    2. Calls Google Maps REST APIs with the real API key.
    3. Applies explicit proxy + CA settings so it works inside stripped stdio
       subprocess environments.
    """
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'atlas_google_maps_server.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/atlas_google_maps_server.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_trip_adviser_format_file(*, case_root: str) -> BenchmarkInputFile:
    """Seed trip-adviser's template with the eval-required station exit labels."""
    format_payload = {
        'starbucks': {
            'recommended_store_name': '',
            'address': '',
        },
        'transportation_route': {
            'from': 'Tokyo Station Nihonbashi Entrance',
            'to': 'Kamakura Station',
            'line_name': '',
            'duration': '',
            'cost': '',
        },
        'kamakura_station': {
            'recommended_exit_name': 'West Exit',
        },
        'walking_route': {
            'from': 'Kamakura Station (West Exit)',
            'to': 'Kamakura Museum of History and Culture',
            'duration': '',
            'distance': '',
        },
    }
    return BenchmarkInputFile(
        path=f'{case_root}/agent_workspace/format.json',
        content=json.dumps(format_payload, ensure_ascii=False, indent=4),
        encoding='utf8',
    )


def _toolathlon_mcp_yaml_proxy_patcher_file() -> BenchmarkInputFile:
    """Upload the MCP YAML proxy patcher script.

    eval's MCPServerManager passes ``params.env`` from the YAML directly to
    ``StdioServerParameters(env=...)``, which *replaces* ``os.environ`` when
    spawning the subprocess.  Any Node.js MCP server with an ``env:`` section
    therefore loses HTTP_PROXY / HTTPS_PROXY / NODE_OPTIONS etc.  This patcher
    runs at bootstrap time (after YAML files are written by
    ``toolathlon_prepare_case.py``) and injects the missing proxy env vars
    into every Node.js server YAML so eval's MCPServerManager picks them up.
    """
    from pathlib import Path

    script_path = Path(__file__).resolve().parent / 'image' / 'helpers' / 'toolathlon_mcp_yaml_proxy_patcher.py'
    return BenchmarkInputFile(
        path=f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_mcp_yaml_proxy_patcher.py',
        content=script_path.read_text(encoding='utf-8'),
        encoding='utf8',
    )


def _toolathlon_bootstraps(
    *,
    case_id: str,
    server_names: list[str],
    case_root: str,
    toolathlon_env: dict[str, str],
    server_mode: BenchmarkServerMode,
) -> list[BenchmarkCommand]:
    # Write the undici CJS bootstrap file that toolathlon_mcp_yaml_proxy_patcher
    # injects as NODE_OPTIONS=--require into every Node.js MCP server YAML config.
    # This patches native fetch() to honour HTTP_PROXY/HTTPS_PROXY via undici.
    # The patcher runs after toolathlon_prepare_case.py so the YAML files exist.
    _undici_cjs_content = 'try{const u=require("undici");u.setGlobalDispatcher(new u.EnvHttpProxyAgent());}catch(e){}'
    bootstraps: list[BenchmarkCommand] = [
        BenchmarkCommand(
            command=f"printf '%s\n' '{_undici_cjs_content}' > /tmp/_undici_proxy_eval.cjs",
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=30.0,
        ),
        BenchmarkCommand(
            command=f'{TOOLATHLON_IMAGE_VENV_BIN}/python {BENCHMARK_IMAGE_HELPERS}/toolathlon_prepare_case.py --case-root {case_root}',
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=120.0,
        ),
        # Inject proxy env vars (HTTP_PROXY, NODE_OPTIONS, etc.) into every Node.js MCP
        # server YAML under configs/mcp_servers/.  eval's MCPServerManager passes params.env
        # directly to StdioServerParameters which replaces os.environ entirely, so proxy
        # vars must be present in the YAML itself.  Runs after prepare_case so files exist.
        BenchmarkCommand(
            command=(
                f'{TOOLATHLON_IMAGE_VENV_BIN}/python '
                f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_mcp_yaml_proxy_patcher.py '
                f'--config-dir {case_root}/configs/mcp_servers '
                f'--npm-node-modules {BENCHMARK_IMAGE_NPM_ROOT}/node_modules'
            ),
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=60.0,
        ),
    ]

    if case_id == 'git-milestone':
        # git-milestone ships its own utils/ (for preprocess: from utils.github_fetcher import ...).
        # But evaluation/main.py also needs from utils.general.helper import normalize_str,
        # which lives in the Toolathlon dataset root utils/general/.
        # Symlink the needed sub-packages into case utils/ so both imports resolve correctly.
        bootstraps.append(
            BenchmarkCommand(
                command=f'ln -sfn {TOOLATHLON_IMAGE_ROOT}/utils/general {case_root}/utils/general',
                cwd=case_root,
                env=toolathlon_env,
                timeout_s=30.0,
            )
        )

    # When email runs as a BenchmarkService (Poste.io container), the main
    # container's email_config.json files still reference localhost:1143/1587.
    # Patch them to use the real service host:port placeholders.
    # The bootstrap runs *after* _setup_services() resolves placeholders, so
    # task.py's _resolve_commands() will substitute the real addresses.
    if 'emails' in server_names:
        bootstraps.append(
            BenchmarkCommand(
                command=(
                    f'{TOOLATHLON_IMAGE_VENV_BIN}/python '
                    f'{BENCHMARK_IMAGE_HELPERS}/toolathlon_email_config_patcher.py '
                    f'--config-dir {case_root} '
                    f'--imap-endpoint {TOOLATHLON_EMAIL_IMAP_PLACEHOLDER} '
                    f'--smtp-endpoint {TOOLATHLON_EMAIL_SMTP_PLACEHOLDER}'
                ),
                cwd=case_root,
                env=toolathlon_env,
                timeout_s=30.0,
            )
        )

    return bootstraps


def _toolathlon_case(
    row: dict[str, Any],
    keys: WorkspaceKeys,
    *,
    server_mode: BenchmarkServerMode,
) -> tuple[BenchmarkCase | None, dict[str, Any] | None]:
    case_id = row['case_id']
    row['task_config']
    prompt = row['prompt']
    system_prompt = row.get('system_prompt')
    server_names = list(row['server_names'])
    embedded_case_paths = set((row.get('embedded_case_files') or {}).keys())
    embedded_server_files = row.get('embedded_server_files') or {}
    decision = decide_toolathlon(row, keys, server_mode=server_mode)
    if not decision.supported:
        return None, {
            'dataset': 'toolathlon',
            'case_id': case_id,
            'server_mode': server_mode.value,
            'reasons': decision.reasons,
        }
    case_root = _case_root(BenchmarkDataset.TOOLATHLON, case_id, server_mode)
    agent_workspace = _agent_workspace(BenchmarkDataset.TOOLATHLON, case_id, server_mode)
    case_source_dir = f'{TOOLATHLON_IMAGE_TASKS_ROOT}/{case_id}'
    template_values = keys.toolathlon_template_values(
        case_root=case_root,
        agent_workspace=agent_workspace,
        server_mode=server_mode,
        task_override_values=dict(row.get('task_override_values') or {}),
        task_source_path=str(row.get('source_path') or ''),
        embedded_case_paths=embedded_case_paths,
    )
    base_token_values_json = json.dumps(
        {key.removeprefix('token.'): value for key, value in template_values.items() if key.startswith('token.')},
        ensure_ascii=False,
    )
    services, placeholder_replacements = _toolathlon_original_services(
        case_id=case_id,
        case_source_dir=case_source_dir,
        server_names=server_names,
        token_values_json=base_token_values_json,
        server_mode=server_mode,
    )
    if placeholder_replacements:
        template_values = _replace_text_fragments(template_values, placeholder_replacements)
        token_values_json = json.dumps(
            {key.removeprefix('token.'): value for key, value in template_values.items() if key.startswith('token.')},
            ensure_ascii=False,
        )
        for service in services:
            if service.startup is None:
                continue
            if 'NEXUS_TOOLATHLON_TOKEN_VALUES_JSON' in service.startup.env:
                service.startup.env['NEXUS_TOOLATHLON_TOKEN_VALUES_JSON'] = token_values_json
    mcp_servers = load_toolathlon_server_specs(
        server_names,
        runtime_name='main',
        template_values=template_values,
        server_mode=server_mode,
        allow_runtime_token_placeholders=True,
        embedded_server_files=embedded_server_files,
    )

    # The Canvas MCP server (Node.js canvas-mcp-server) builds its base URL as
    # ``https://${CANVAS_DOMAIN}/api/v1`` — the scheme is hardcoded to *https*.
    # However the gongfeng sandbox exposes all service ports over plain HTTP
    # (TLS is not forwarded through the port mapping).  We replace the MCP
    # server command with a bash one-liner that: 1) lets npx install the
    # package, 2) patches client.js to use http:// instead of https://,
    # 3) runs the patched entry point.
    # CANVAS_DOMAIN is set to the bare host:port of the HTTP listener (port
    # 10001) so the final URL becomes ``http://host:port/api/v1``.
    if 'canvas' in server_names:
        for spec in mcp_servers:
            spec_client = spec.get('client') or spec.get('params') or {}
            spec_env = spec_client.get('env') or {}
            if spec.get('name') == 'canvas' and 'CANVAS_DOMAIN' in spec_env:
                spec_env['CANVAS_DOMAIN'] = '${token.canvas_domain}'
                # Replace "npx -y canvas-mcp-server" with a bash wrapper
                # that patches the installed source to use HTTP.
                # We use `npm install -g` (deterministic path under
                # /usr/local/lib/node_modules/) instead of npx (whose
                # cache location varies and may not persist).
                spec_client['command'] = 'bash'
                _canvas_index = '/usr/local/lib/node_modules/canvas-mcp-server/build/index.js'
                # canvas-mcp-server 2.2.3 bugs:
                # 1) TOOLS array registers conversation, notification, discussion,
                #    announcement, and quiz tools but the switch(toolName) handler
                #    is missing the corresponding case branches.
                # 2) canvas_update_course schema lacks the ``event`` field needed
                #    to publish a course (Canvas API: course[event]=offer).
                # 3) canvas_submit_quiz_attempt has a client method but no TOOLS
                #    entry or case handler.
                # We write a JS patch script that fixes all three issues.
                _patch_file = '/tmp/_canvas_patch.js'
                _case_lines = (
                    # conversation / notification (cb8a285)
                    'case "canvas_list_conversations":{const r=await this.client.listConversations();'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_get_conversation":{const{conversation_id:id}=args;'
                    'const r=await this.client.getConversation(id);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_create_conversation":{const{recipients:rr,body:b,subject:s}=args;'
                    'const r=await this.client.createConversation(rr,b,s);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_list_notifications":{const r=await this.client.listNotifications();'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    # discussion topics
                    'case "canvas_list_discussion_topics":{const{course_id:cid}=args;'
                    'const r=await this.client.listDiscussionTopics(cid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_get_discussion_topic":{const{course_id:cid,topic_id:tid}=args;'
                    'const r=await this.client.getDiscussionTopic(cid,tid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_post_to_discussion":{const{course_id:cid,topic_id:tid,message:msg}=args;'
                    'const r=await this.client.postToDiscussion(cid,tid,msg);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    # announcements
                    'case "canvas_list_announcements":{const{course_id:cid}=args;'
                    'const r=await this.client.listAnnouncements(cid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    # quizzes
                    'case "canvas_list_quizzes":{const{course_id:cid}=args;'
                    'const r=await this.client.listQuizzes(cid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_get_quiz":{const{course_id:cid,quiz_id:qid}=args;'
                    'const r=await this.client.getQuiz(cid,qid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_create_quiz":{const{course_id:cid,...qd}=args;'
                    'const r=await this.client.createQuiz(cid,qd);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    'case "canvas_start_quiz_attempt":{const{course_id:cid,quiz_id:qid}=args;'
                    'const r=await this.client.startQuizAttempt(cid,qid);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}\n'
                    # submit quiz attempt (client method exists, no TOOLS entry)
                    'case "canvas_submit_quiz_attempt":{'
                    'const{course_id:cid,quiz_id:qid,submission_id:sid,answers:ans}=args;'
                    'const r=await this.client.submitQuizAttempt(cid,qid,sid,ans);'
                    'return{content:[{type:"text",text:JSON.stringify(r,null,2)}]};}'
                )
                spec_client['args'] = [
                    '-c',
                    'npm install -g canvas-mcp-server 2>/dev/null; '
                    "sed -i 's|https://|http://|' "
                    '/usr/local/lib/node_modules/canvas-mcp-server/build/client.js; '
                    # Write case handler lines to patch file (heredoc, safe)
                    f"cat > {_patch_file} << 'JSPATCH'\n"
                    f'{_case_lines}\n'
                    'JSPATCH\n'
                    'node -e "'
                    "const fs=require('fs');"
                    f"const idx='{_canvas_index}';"
                    f"const pf='{_patch_file}';"
                    "let s=fs.readFileSync(idx,'utf8');"
                    "const patch=fs.readFileSync(pf,'utf8').trim();"
                    r"const marker='throw new Error(\x60Unknown tool:';"
                    'const i=s.lastIndexOf(marker);'
                    r"""if(i>0){s=s.slice(0,i)+patch+'\n                        '+s.slice(i);}"""
                    # add event field to canvas_update_course schema
                    # Target "Updated syllabus content" (unique to update, not create)
                    r""";var k=s.indexOf('Updated syllabus content');if(k>0){var j=s.indexOf('}',k);"""
                    r"""if(j>0){s=s.slice(0,j+1)+"""
                    "'"
                    r""",\n                event: { type: \x22string\x22, description: \x22Course event: offer to publish, claim to unpublish\x22 }"""
                    "'"
                    r"""+s.slice(j+1);}}"""
                    ';fs.writeFileSync(idx,s);'
                    '"; '
                    f'exec node {_canvas_index}',
                ]

    # Patch google-cloud-mcp BigQuery run_query to return all rows instead of
    # only the first 5 ("Sample results").  The upstream server.py hardcodes
    # ``rows[:5]`` which causes the agent to hallucinate missing data.
    # Strategy: pip install to a fixed path, sed-patch server.py, then exec.
    if 'google-cloud' in server_names:
        for spec in mcp_servers:
            spec_client = spec.get('client') or spec.get('params') or {}
            if spec.get('name') == 'google-cloud' and spec_client.get('command') == 'uvx':
                original_args = list(spec_client.get('args', []))
                # Drop the leading 'google-cloud-mcp' from args — the rest
                # are CLI flags (--project-id, --allowed-datasets, …).
                cli_args = [a for a in original_args if a != 'google-cloud-mcp']
                cli_args_str = ' '.join(f"'{a}'" if ' ' in a else a for a in cli_args)
                _gcloud_mcp_dir = '/tmp/_gcloud_mcp'
                spec_client['command'] = 'bash'
                spec_client['args'] = [
                    '-c',
                    # Install to a deterministic directory so we can patch.
                    f'pip install --target {_gcloud_mcp_dir} google-cloud-mcp 2>/dev/null; '
                    # Remove the 5-row truncation: show all rows, not just a sample.
                    f"sed -i 's/rows\\[:5\\]/rows/' {_gcloud_mcp_dir}/src/server.py; "
                    f"sed -i 's/Sample results (first 5 rows)/Query results/' {_gcloud_mcp_dir}/src/server.py; "
                    # Delete the "... and N more rows" line and its preceding if-guard.
                    f"sed -i '/len(rows) > 5/d' {_gcloud_mcp_dir}/src/server.py; "
                    f"sed -i '/more rows/d' {_gcloud_mcp_dir}/src/server.py; "
                    # Support glob wildcards (e.g. "promo-assets-for-b*") in allowed bucket names.
                    # Replace exact-match "x in ALLOWED_Y" with fnmatch-based matching.
                    f"sed -i '1s/^/import fnmatch\\n/' {_gcloud_mcp_dir}/src/server.py; "
                    f"sed -i 's/bucket_name in ALLOWED_BUCKETS/any(fnmatch.fnmatch(bucket_name, p) for p in ALLOWED_BUCKETS)/g' {_gcloud_mcp_dir}/src/server.py; "
                    f"sed -i 's/bucket_name in ALLOWED_LOG_BUCKETS/any(fnmatch.fnmatch(bucket_name, p) for p in ALLOWED_LOG_BUCKETS)/g' {_gcloud_mcp_dir}/src/server.py; "
                    # Also fix list_buckets filter to use fnmatch.
                    f"sed -i \"s/b.get('name') in ALLOWED_BUCKETS/any(fnmatch.fnmatch(b.get('name',''), p) for p in ALLOWED_BUCKETS)/g\" {_gcloud_mcp_dir}/src/server.py; "
                    # Run the patched server, passing through the original CLI args.
                    # Inject REST transport env vars so GCP API calls go through HTTP/1.1
                    # instead of gRPC (gongfeng Squid MITM proxy breaks HTTP/2 ALPN).
                    f'exec env PYTHONPATH={_gcloud_mcp_dir}:$PYTHONPATH '
                    'GOOGLE_CLOUD_ENABLE_REST_TRANSPORT=true '
                    'GOOGLE_CLOUD_DISABLE_GRPC=true '
                    'SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt '
                    f'python -c "from src.server import main; main()" {cli_args_str}',
                ]

    preprocess = None
    has_preprocess_main = bool(row.get('has_preprocess_main'))
    has_preprocess_sh = bool(row.get('has_preprocess_sh'))
    toolathlon_pythonpath_parts = [BENCHMARK_IMAGE_HELPERS, TOOLATHLON_SHIMS_ROOT]
    toolathlon_pythonpath_parts.extend([case_root, f'{TOOLATHLON_IMAGE_ROOT}/utils', TOOLATHLON_IMAGE_ROOT])
    toolathlon_pythonpath = ':'.join(toolathlon_pythonpath_parts)
    toolathlon_env = {
        'PYTHONPATH': toolathlon_pythonpath,
        'NEXUS_TOOLATHLON_CASE_ROOT': case_root,
        'NEXUS_TOOLATHLON_DATASET_ROOT': TOOLATHLON_IMAGE_ROOT,
        'NEXUS_TOOLATHLON_EMAIL_STORE_PATH': f'{case_root}/.nexus_email_store.json',
        'NEXUS_TOOLATHLON_GOOGLE_STORE_PATH': f'{case_root}/.nexus_google_store.json',
        'NEXUS_TOOLATHLON_NOTION_STORE_PATH': f'{case_root}/.nexus_notion_store.json',
        'NEXUS_TOOLATHLON_CANVAS_STORE_PATH': f'{case_root}/.nexus_canvas_store.json',
        'NEXUS_TOOLATHLON_K8S_STATE_PATH': f'{case_root}/.nexus_k8s_state.json',
        'NEXUS_TOOLATHLON_K8S_ROOT': f'{case_root}/.nexus_k8s',
        'NEXUS_TOOLATHLON_SNOWFLAKE_DB_PATH': f'{case_root}/.nexus_snowflake.sqlite',
        'NEXUS_TOOLATHLON_WOOCOMMERCE_STORE_PATH': f'{case_root}/.nexus_woocommerce_store.json',
        'NEXUS_TOOLATHLON_WOOCOMMERCE_MEDIA_DIR': f'{case_root}/.nexus_woocommerce_media',
        'NEXUS_TOOLATHLON_SERVER_MODE': server_mode.value,
        # Force Python/requests/curl to use system CA bundle which includes Gongfeng Squid Proxy CA.
        # Without this, conda/nix Python uses its own CA bundle that lacks the proxy CA cert,
        # causing SSL CERTIFICATE_VERIFY_FAILED when accessing Google APIs through the corporate proxy.
        'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
        'REQUESTS_CA_BUNDLE': '/etc/ssl/certs/ca-certificates.crt',
        'CURL_CA_BUNDLE': '/etc/ssl/certs/ca-certificates.crt',
        'HTTPLIB2_CA_CERTS': '/etc/ssl/certs/ca-certificates.crt',
        # Route HTTP/HTTPS requests through corporate proxy so that eval scripts that
        # spawn MCP server subprocesses (e.g. Google Maps node server started by
        # MCPServerManager in evaluation/main.py) can reach Google APIs.
        # Without these, the eval process env has no proxy settings and the spawned
        # node server gets empty HTTP responses (JSON decode error: char 0).
        'HTTP_PROXY': 'http://hyrl-proxy.woa.com:3128',
        'HTTPS_PROXY': 'http://hyrl-proxy.woa.com:3128',
        'NO_PROXY': '.woa.com,localhost,127.0.0.1',
        # NODE_EXTRA_CA_CERTS ensures node processes our framework spawns directly inherit
        # the Squid MITM CA.  eval's MCPServerManager subprocesses get it via the YAML
        # patcher (toolathlon_mcp_yaml_proxy_patcher.py) instead.
        'NODE_EXTRA_CA_CERTS': '/etc/ssl/certs/ca-certificates.crt',
        # When eval scripts spawn 'uv run' subprocesses, uv searches for pyproject.toml
        # upward from cwd. The Toolathlon dataset root has a pyproject.toml, so uv finds
        # it and tries to use the project's .venv — which may not have google packages.
        # UV_PROJECT_ENVIRONMENT forces uv to use the specified venv regardless of any
        # pyproject.toml it discovers. VIRTUAL_ENV alone is not enough because uv ignores
        # it when a project is found.
        'VIRTUAL_ENV': TOOLATHLON_IMAGE_VENV,
        'UV_PROJECT_ENVIRONMENT': TOOLATHLON_IMAGE_VENV,
        'GRPC_DEFAULT_SSL_ROOTS_FILE_PATH': '/etc/ssl/certs/ca-certificates.crt',
        # Force GCP Python SDK to use REST (HTTP/1.1) instead of gRPC (HTTP/2).
        # gongfeng Squid MITM proxy has intermittent ALPN negotiation failures with HTTP/2,
        # causing "Cannot check peer: missing selected ALPN property" errors.
        # Two env vars needed: GOOGLE_CLOUD_ENABLE_REST_TRANSPORT for gapic-generated
        # clients (ConfigServiceV2Client), and GOOGLE_CLOUD_DISABLE_GRPC for the
        # hand-written logging.Client which checks a different env var.
        'GOOGLE_CLOUD_ENABLE_REST_TRANSPORT': 'true',
        'GOOGLE_CLOUD_DISABLE_GRPC': 'true',
        'NEXUS_TOOLATHLON_TOKEN_VALUES_JSON': json.dumps(
            {key.removeprefix('token.'): value for key, value in template_values.items() if key.startswith('token.')},
            ensure_ascii=False,
        ),
    }
    input_files = [BenchmarkInputFile.model_validate(item) for item in keys.toolathlon_input_files(case_root=case_root)]
    input_files.extend(
        BenchmarkInputFile(path=path, content=content, encoding='utf8')
        for path, content in load_toolathlon_server_config_texts(
            server_mode=server_mode,
            embedded_server_files=embedded_server_files,
        ).items()
    )
    # Upload the patched notion_client shim that delegates to the real SDK in
    # API mode.  The image-bundled version always uses the local store, which
    # lacks source-page data and breaks preprocess (notion_page_duplicator).
    input_files.append(_toolathlon_notion_client_shim_file())
    # Upload the patched sitecustomize that monkey-patches duplicate_page_with_mcp
    # to use Notion REST API instead of mcp-remote (which can't reach mcp.notion.com
    # through the container proxy).
    input_files.append(_toolathlon_sitecustomize_shim_file())
    # Upload the patched gspread shim that delegates to the real gspread in API mode.
    # The image-bundled gspread.py shim unconditionally uses MockGspreadClient, which
    # breaks eval scripts (e.g. vlm-history-completer) that read real Google Sheets.
    input_files.append(_toolathlon_gspread_shim_file())
    if case_id == 'trip-adviser':
        input_files.append(_toolathlon_trip_adviser_format_file(case_root=case_root))
    # Upload the email config patcher script for cases that use the emails
    # MCP server.  The patcher rewrites email_config.json fields to point at
    # the real Poste.io BenchmarkService host:port instead of localhost.
    if 'emails' in server_names:
        input_files.append(_toolathlon_email_config_patcher_file())
    # Upload the patched emails MCP server wrapper (Header bug fix) for all
    # cases that use the emails server.  This overwrites the image-bundled version.
    if 'emails' in server_names:
        input_files.append(_toolathlon_emails_server_file())
    # Upload patched Google MCP server scripts with httplib2 CA fix for
    # gongfeng Squid proxy SSL MITM.  Overwrites the image-bundled versions.
    if 'google_sheet' in server_names:
        input_files.append(_toolathlon_google_sheet_server_file())
    if 'google_calendar' in server_names:
        input_files.append(_toolathlon_google_calendar_server_file())
    if 'google_forms' in server_names:
        input_files.append(_toolathlon_google_forms_server_file())
    # Upload a dedicated Google Maps eval server that exposes unprefixed
    # ``maps_*`` tool names and calls Google Maps REST APIs directly. This keeps
    # eval-side semantics aligned with Toolathlon's expectations without relying
    # on the agent-side Node.js MCP package.
    gmaps_eval_yaml_text: str | None = None
    if 'google_map' in server_names:
        input_files.append(_toolathlon_google_maps_server_file())
        # Replace the localized google_map.yaml in-place for eval only. Using a
        # single final entry avoids duplicate-path ambiguity during upload.
        # The Python server speaks the ``maps_*`` interface that Toolathlon eval
        # scripts call directly and uses the real Google Maps API key plus
        # proxy/CA env.
        import yaml

        google_maps_api_key = keys.get_any('google_cloud_console_api_key', 'google_maps_api_key') or ''
        _gmaps_eval_yaml = {
            'type': 'stdio',
            'name': 'google_map',
            'params': {
                'command': f'{BENCHMARK_IMAGE_VENV_BIN}/python',
                'args': [f'{BENCHMARK_IMAGE_HELPERS}/atlas_google_maps_server.py'],
                'env': {
                    'GOOGLE_MAPS_API_KEY': google_maps_api_key,
                    'HTTP_PROXY': 'http://hyrl-proxy.woa.com:3128',
                    'HTTPS_PROXY': 'http://hyrl-proxy.woa.com:3128',
                    'NO_PROXY': '.woa.com,localhost,127.0.0.1',
                    'SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
                    'REQUESTS_CA_BUNDLE': '/etc/ssl/certs/ca-certificates.crt',
                },
                'cwd': '${agent_workspace}',
            },
            'client_session_timeout_seconds': 30,
            'cache_tools_list': True,
        }
        gmaps_eval_yaml_text = yaml.safe_dump(_gmaps_eval_yaml, sort_keys=False, allow_unicode=True)
        replaced_google_map_yaml = False
        for index, input_file in enumerate(input_files):
            if input_file.path != 'configs/mcp_servers/google_map.yaml':
                continue
            input_files[index] = BenchmarkInputFile(
                path=input_file.path,
                content=gmaps_eval_yaml_text,
                encoding='utf8',
            )
            replaced_google_map_yaml = True
            break
        if not replaced_google_map_yaml:
            input_files.append(
                BenchmarkInputFile(
                    path='configs/mcp_servers/google_map.yaml',
                    content=gmaps_eval_yaml_text,
                    encoding='utf8',
                )
            )
    # Always upload the YAML proxy patcher — it runs at bootstrap time to inject
    # HTTP_PROXY / NODE_OPTIONS / etc. into Node.js MCP server YAML env sections
    # so that eval's MCPServerManager (which replaces os.environ when spawning
    # subprocesses) passes proxy settings to every Node.js MCP server.
    input_files.append(_toolathlon_mcp_yaml_proxy_patcher_file())
    # When a Canvas service is present, ensure token_key_session.py contains
    # canvas_domain so that preprocess scripts (which parse the file with simple
    # string matching for ``canvas_domain = ...``) resolve the Canvas HTTP URL
    # correctly instead of falling back to localhost:10001.
    if 'canvas' in server_names:
        token_dict = {
            key.removeprefix('token.'): value for key, value in template_values.items() if key.startswith('token.')
        }
        # Override canvas_domain to the HTTP (port 10001) domain placeholder.
        # The keys.py default points at port 20001 (HTTPS) but preprocess
        # scripts call the REST API over plain HTTP on port 10001.
        token_dict['canvas_domain'] = TOOLATHLON_CANVAS_HTTP_DOMAIN_PLACEHOLDER
        # canvas_base_url carries the full URL with http:// scheme, used by eval
        # scripts and any Python consumer that needs a complete HTTP URL.
        token_dict['canvas_base_url'] = TOOLATHLON_CANVAS_HTTP_PLACEHOLDER
        # Also set canvas_admin_domain (used by some preprocess scripts via other_key.py).
        token_dict.setdefault('canvas_admin_domain', TOOLATHLON_CANVAS_HTTP_DOMAIN_PLACEHOLDER)
        # Use double-quote assignment format that get_local_canvas_config() can
        # parse with its ``line.split('=')[1].strip().strip('"')`` logic.
        # Standalone assignments come first (for simple string-parsing consumers),
        # then the Dict for import-based consumers.
        lines = ['from addict import Dict', 'import os', '']
        for k, v in sorted(token_dict.items()):
            lines.append(f'{k} = "{v}"')
        lines.append('')
        lines.append('all_token_key_session = Dict(')
        for k, v in sorted(token_dict.items()):
            lines.append(f'    {k} = "{v}",')
        lines.append(')')
        lines.append('')
        input_files.append(
            BenchmarkInputFile(
                path=f'{case_root}/token_key_session.py',
                content='\n'.join(lines),
                encoding='utf8',
            )
        )
    bootstraps = _toolathlon_bootstraps(
        case_id=case_id,
        server_names=server_names,
        case_root=case_root,
        toolathlon_env=toolathlon_env,
        server_mode=server_mode,
    )
    if gmaps_eval_yaml_text is not None:
        gmaps_eval_yaml_b64 = base64.b64encode(gmaps_eval_yaml_text.encode('utf-8')).decode('ascii')
        bootstraps.insert(
            2,
            BenchmarkCommand(
                command=(
                    f"{TOOLATHLON_IMAGE_VENV_BIN}/python - <<'PY'\n"
                    'import base64\n'
                    'import os\n'
                    'from pathlib import Path\n'
                    '\n'
                    "yaml_path = Path(os.environ['NEXUS_GMAPS_YAML_PATH'])\n"
                    "payload = base64.b64decode(os.environ['NEXUS_GMAPS_YAML_B64']).decode('utf-8')\n"
                    "yaml_path.write_text(payload, encoding='utf-8')\n"
                    'PY'
                ),
                cwd=case_root,
                env={
                    **toolathlon_env,
                    'NEXUS_GMAPS_YAML_PATH': f'{case_root}/configs/mcp_servers/google_map.yaml',
                    'NEXUS_GMAPS_YAML_B64': gmaps_eval_yaml_b64,
                },
                timeout_s=30.0,
            ),
        )
    for spec in mcp_servers:
        if spec.get('name') != 'k8s':
            continue
        client = spec.setdefault('client', {})
        env = dict(client.get('env', {}) or {})
        env.update({
            'NEXUS_TOOLATHLON_K8S_STATE_PATH': toolathlon_env['NEXUS_TOOLATHLON_K8S_STATE_PATH'],
            'NEXUS_TOOLATHLON_K8S_ROOT': toolathlon_env['NEXUS_TOOLATHLON_K8S_ROOT'],
        })
        client['env'] = env
    if has_preprocess_main:
        preprocess = BenchmarkCommand(
            command=f'{TOOLATHLON_IMAGE_VENV_BIN}/python -m preprocess.main',
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=1200.0,
        )
    elif has_preprocess_sh:
        preprocess = BenchmarkCommand(
            command=f'bash {case_root}/preprocess/run.sh',
            cwd=f'{case_root}/preprocess',
            env=toolathlon_env,
            timeout_s=1200.0,
        )

    evaluator = BenchmarkEvaluator(
        type='toolathlon_script',
        config={
            'command': (
                f'{TOOLATHLON_IMAGE_VENV_BIN}/python -m evaluation.main '
                f'--agent_workspace {case_root}/agent_workspace '
                f'--groundtruth_workspace {case_root}/groundtruth_workspace'
            ),
            'cwd': case_root,
            'env': toolathlon_env,
            'timeout_s': 1200.0,
        },
    )

    return (
        BenchmarkCase(
            dataset=BenchmarkDataset.TOOLATHLON,
            case_id=case_id,
            server_mode=server_mode,
            prompt=prompt,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            services=services,
            max_iterations=50,
            runtime_name='main',
            case_source_dir=case_source_dir,
            initial_workspace_dir='initial_workspace' if row.get('has_initial_workspace') else None,
            input_files=input_files,
            case_text_replacements=placeholder_replacements,
            bootstraps=bootstraps,
            preprocess=preprocess,
            evaluator=evaluator,
            metadata={
                'server_names': server_names,
                'service_names': [service.name for service in services],
                'server_mode': server_mode.value,
            },
        ),
        None,
    )


def collect_toolathlon_cases(server_mode: BenchmarkServerMode) -> tuple[list[BenchmarkCase], list[dict[str, Any]]]:
    keys = WorkspaceKeys()
    cases: list[BenchmarkCase] = []
    skipped: list[dict[str, Any]] = []
    for row in load_toolathlon_manifest_rows():
        case, skip = _toolathlon_case(row, keys, server_mode=server_mode)
        if case is not None:
            cases.append(case)
        elif skip is not None:
            skipped.append(skip)
    return cases, skipped


def _atlas_mongodb_service_image() -> str:
    override = os.getenv('MCP_BENCHMARK_ATLAS_MONGODB_IMAGE')
    if override:
        return override

    provider = os.getenv('MCP_BENCHMARK_RUNTIME_PROVIDER_NAME', 'docker')
    if provider == 'gongfeng':
        return ATLAS_MONGODB_SERVICE_GONGFENG_IMAGE_NAME

    return ATLAS_MONGODB_SERVICE_IMAGE_NAME


def _atlas_case_services(server_names: list[str], server_mode: BenchmarkServerMode) -> list[BenchmarkService]:
    return []


def _atlas_mcp_servers(
    *,
    server_names: list[str],
    services: list[BenchmarkService],
    env_values: dict[str, str],
    server_mode: BenchmarkServerMode,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    has_atlas_mongodb = any(service.name == 'atlas-mongodb' for service in services)

    for server_name in server_names:
        server_env_values = dict(env_values)
        runtime_name = 'main'

        if server_name == 'mongodb' and has_atlas_mongodb:
            runtime_name = ATLAS_MONGODB_SERVICE_RUNTIME_NAME
            server_env_values['MONGODB_CONNECTION_STRING'] = 'mongodb://127.0.0.1:27017/?directConnection=true'

        specs.extend(
            load_atlas_server_specs(
                [server_name],
                runtime_name=runtime_name,
                env_values=server_env_values,
                server_mode=server_mode,
            )
        )

    return specs


def collect_atlas_cases(server_mode: BenchmarkServerMode) -> tuple[list[BenchmarkCase], list[dict[str, Any]]]:
    keys = WorkspaceKeys()
    cases: list[BenchmarkCase] = []
    skipped: list[dict[str, Any]] = []
    for row in load_atlas_manifest_rows():
        case_id = row['case_id']
        enabled_tools = row['enabled_tools']
        claims = row['claims']
        server_names = list(row['server_names'])
        decision = decide_atlas(server_names, keys, server_mode=server_mode)
        if not decision.supported:
            skipped.append({
                'dataset': 'atlas',
                'case_id': case_id,
                'server_mode': server_mode.value,
                'reasons': decision.reasons,
            })
            continue
        services = _atlas_case_services(server_names, server_mode)
        env_values = keys.atlas_env_values()
        mcp_servers = _atlas_mcp_servers(
            server_names=server_names,
            services=services,
            env_values=env_values,
            server_mode=server_mode,
        )
        cases.append(
            BenchmarkCase(
                dataset=BenchmarkDataset.ATLAS,
                case_id=case_id,
                server_mode=server_mode,
                prompt=row['prompt'],
                mcp_servers=mcp_servers,
                services=services,
                enabled_tools=enabled_tools,
                max_iterations=20,
                runtime_name='main',
                evaluator=BenchmarkEvaluator(
                    type='atlas_llm_judge',
                    config={'claims': claims, 'threshold': 0.5, 'judge_client': 'judge'},
                ),
                metadata={
                    'server_names': server_names,
                    'service_names': [service.name for service in services],
                    'source_path': row['source_path'],
                    'server_mode': server_mode.value,
                },
            )
        )
    return cases, skipped


def _universe_case(
    row: dict[str, Any],
    keys: WorkspaceKeys,
    *,
    server_mode: BenchmarkServerMode,
) -> tuple[BenchmarkCase | None, dict[str, Any] | None]:
    task_file = UNIVERSE_DATASET_ROOT / row['task_file_rel']
    task_data = row['task_data']
    case_id = row['case_id']
    server_names = list(row['server_names'])
    decision = decide_universe(task_data, server_names, keys, server_mode=server_mode)
    if not decision.supported:
        return None, {
            'dataset': 'universe',
            'case_id': case_id,
            'server_mode': server_mode.value,
            'reasons': decision.reasons,
        }

    case_root = _case_root(BenchmarkDataset.UNIVERSE, case_id, server_mode)
    env_values = keys.universe_env_values(case_root=case_root)
    input_files = [BenchmarkInputFile.model_validate(item) for item in keys.universe_input_files(case_root=case_root)]
    if 'blender' in server_names:
        env_values['BLENDER_APP_PATH'] = f'{BENCHMARK_IMAGE_HELPERS}/universe_blender_cli.py'
    rendered_prompt = str(render_value(task_data['question'], env_values)).strip()
    rendered_evaluators = render_value(task_data.get('evaluators', []), env_values)
    rendered_cleanups = render_value(task_data.get('cleanups', []), env_values)
    rendered_output_format = render_value(task_data.get('output_format'), env_values)
    mcp_servers = load_universe_server_specs(
        task_data,
        runtime_name='main',
        env_values=env_values,
        server_mode=server_mode,
    )
    cleanups = [BenchmarkCleanup.model_validate(item) for item in rendered_cleanups]
    preprocess = None
    if 'filesystem' in server_names:
        preprocess = BenchmarkCommand(command=f'mkdir -p {case_root}/filesystem', timeout_s=60.0)

    return (
        BenchmarkCase(
            dataset=BenchmarkDataset.UNIVERSE,
            case_id=case_id,
            server_mode=server_mode,
            prompt=rendered_prompt,
            mcp_servers=mcp_servers,
            max_iterations=20,
            runtime_name='main',
            input_files=input_files,
            preprocess=preprocess,
            evaluator=BenchmarkEvaluator(
                type='universe_official',
                config={
                    'evaluators': rendered_evaluators,
                    'context_env': env_values,
                    'timeout_s': 300.0,
                },
            ),
            cleanups=cleanups,
            context_env=env_values,
            output_format=rendered_output_format or None,
            metadata={
                'server_names': server_names,
                'category': task_data.get('category', 'general'),
                'source_path': str(task_file),
                'server_mode': server_mode.value,
            },
        ),
        None,
    )


def collect_universe_cases(server_mode: BenchmarkServerMode) -> tuple[list[BenchmarkCase], list[dict[str, Any]]]:
    keys = WorkspaceKeys()
    cases: list[BenchmarkCase] = []
    skipped: list[dict[str, Any]] = []
    for row in load_universe_manifest_rows():
        case, skip = _universe_case(row, keys, server_mode=server_mode)
        if case is not None:
            cases.append(case)
        elif skip is not None:
            skipped.append(skip)
    return cases, skipped


def _normalize_dataset_key(datasets: list[str] | None) -> tuple[str, ...]:
    requested = set(datasets or [d.value for d in BenchmarkDataset])
    return tuple(dataset.value for dataset in BenchmarkDataset if dataset.value in requested)


def _normalize_server_mode_key(server_modes: list[str] | None) -> tuple[str, ...]:
    requested = {BenchmarkServerMode(item).value for item in (server_modes or [BenchmarkServerMode.API.value])}
    return tuple(mode.value for mode in BenchmarkServerMode if mode.value in requested)


@cache
def _collect_cases_cached(
    dataset_key: tuple[str, ...],
    server_mode_key: tuple[str, ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    cases: list[BenchmarkCase] = []
    skipped: list[dict[str, Any]] = []

    for server_mode_name in server_mode_key:
        server_mode = BenchmarkServerMode(server_mode_name)
        if 'toolathlon' in dataset_key:
            c, s = collect_toolathlon_cases(server_mode)
            cases.extend(c)
            skipped.extend(s)
        if 'atlas' in dataset_key:
            c, s = collect_atlas_cases(server_mode)
            cases.extend(c)
            skipped.extend(s)
        if 'universe' in dataset_key:
            c, s = collect_universe_cases(server_mode)
            cases.extend(c)
            skipped.extend(s)

    cases.sort(key=lambda case: (case.dataset.value, case.server_mode.value, case.case_id))
    skipped.sort(key=lambda item: (item['dataset'], item.get('server_mode', ''), item['case_id']))
    return (
        tuple(case.model_dump(mode='json') for case in cases),
        tuple(copy.deepcopy(item) for item in skipped),
    )


def collect_cases(
    datasets: list[str] | None = None,
    server_modes: list[str] | None = None,
) -> tuple[list[BenchmarkCase], list[dict[str, Any]]]:
    dataset_key = _normalize_dataset_key(datasets)
    server_mode_key = _normalize_server_mode_key(server_modes)
    cached_cases, cached_skipped = _collect_cases_cached(dataset_key, server_mode_key)
    cases = [BenchmarkCase.model_validate(copy.deepcopy(case)) for case in cached_cases]
    skipped = [copy.deepcopy(item) for item in cached_skipped]
    return cases, skipped
