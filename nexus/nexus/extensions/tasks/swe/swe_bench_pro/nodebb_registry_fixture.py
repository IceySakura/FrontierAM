import ast
import copy
import hashlib
import json
import shlex
import textwrap

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe.swe_bench_pro.nodebb_registry_assets import (
    NODEBB_REGISTRY_ASSET_BUNDLE_BASE64,
    NODEBB_REGISTRY_MANIFEST_JSON,
    NODEBB_REGISTRY_SERVER_SOURCE,
)
from nexus.runtime.executors.models.file import FileItem


NODEBB_REGISTRY_FIXTURE_MARKER = '# SWE-bench-pro NodeBB exact-case local dual registry fixture'
NODEBB_REGISTRY_SERVER_FILENAME = 'nodebb_registry_server.py'
NODEBB_REGISTRY_MANIFEST_FILENAME = 'nodebb_registry_manifest.json'
NODEBB_REGISTRY_ASSET_BUNDLE_FILENAME = 'nodebb_registry_assets.tar.gz'

NODEBB_REGISTRY_ASSET_BUNDLE_SHA256 = '7ec806a529ca42681419bb3d486c31e4e910ffef11cecf3c0233ae0db699f6a7'
NODEBB_REGISTRY_SERVER_SHA256 = hashlib.sha256(NODEBB_REGISTRY_SERVER_SOURCE.encode()).hexdigest()
NODEBB_REGISTRY_MANIFEST_SHA256 = hashlib.sha256(NODEBB_REGISTRY_MANIFEST_JSON.encode()).hexdigest()


@dataclass(frozen=True)
class NodeBBRegistryCase:
    instance_id: str
    repo: str
    dataset_type: str
    base_commit: str
    workdir: str
    repo_language: str
    image: str
    selected_tests: tuple[str, ...]
    test_patch_sha256: str
    before_repo_set_cmd_sha256: str
    fail_to_pass_count: int
    fail_to_pass_sha256: str
    pass_to_pass_count: int
    pass_to_pass_sha256: str
    spec_sha256: tuple[tuple[str, str], ...]
    test_file_sha256: str
    node_version: str
    npm_version: str
    nodebb_version: str


NODEBB_REGISTRY_CASE = NodeBBRegistryCase(
    instance_id=(
        'instance_nodebb__nodebb-76c6e30282906ac664f2c9278fc90999b27b1f48-vd59a5728dfc977f44533186ace531248c2917516'
    ),
    repo='NodeBB/NodeBB',
    dataset_type='swe-bench_pro',
    base_commit='a3e1a666b876e0b3ccbb5284dd826c8c90c113b4',
    workdir='/app',
    repo_language='js',
    image=(
        'mirrors.tencent.com/hunyuan_yanguan/'
        'sweap-images-instance_nodebb__nodebb-76c6e30282906ac664f2c9278fc90999b27b1f48-'
        'vd59a5728dfc977f44533186ace531248c2917516:latest'
    ),
    selected_tests=('test/i18n.js', 'test/plugins.js'),
    test_patch_sha256='73590de413d2405f5b36db8ed967c39f3a44984af3ed0c24925bd8737ab626fb',
    before_repo_set_cmd_sha256='49a80d23fd759829b9de3c6c7ca88c4e2d6b68f364dc06f42d17b9fc18b46d6e',
    fail_to_pass_count=1,
    fail_to_pass_sha256='174a44296a7fe0ee5fb66aafc23e65f3470d4f1bcb5954174fda320531fa4a2d',
    pass_to_pass_count=2924,
    pass_to_pass_sha256='fa2984a51674b06fd80b4c140c3d629bc050b9f5798579deb789639439482363',
    spec_sha256=(
        ('baseDockerfile', 'e6f11cf3a5bc6b4a955eb4d18d7b9a69e48e742837ae9505fc71f0279b16bfcf'),
        ('instanceDockerfile', '8604fa0238d8426756cf5dd6bad8566bc3cd5187028f9c1756f1f82d2a3fe7b4'),
        ('parser.py', 'b9988675ff86bad71fc88729109ce3516f6df9189dc0052188d13aa24b877558'),
        ('run_script.sh', '3fdc38de6eb0de19236b3b5a186ebb9de5df08c13f0bccfcf1f82989cc4bcd0d'),
    ),
    test_file_sha256='b9344d41e89dfc82b2e6e8e80b0c767a33b553fedf05ee16de0b0c0918ac1d5f',
    node_version='v22.16.0',
    npm_version='10.9.2',
    nodebb_version='3.6.3',
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _parse_string_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f'invalid {field_name}') from e
    if not isinstance(value, list | tuple):
        raise ValueError(f'invalid {field_name}')
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f'invalid {field_name}')
    return list(value)


def _required_string(instance: Mapping[str, Any], field_name: str) -> str:
    value = instance.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f'invalid {field_name}')
    return value


def _canonical_test_fingerprint(value: Any, field_name: str) -> tuple[int, str]:
    tests = _parse_string_list(value, field_name)
    payload = json.dumps(tests, ensure_ascii=False, separators=(',', ':'))
    return len(tests), _sha256(payload)


def _instance_fingerprint(instance: Mapping[str, Any], expected: NodeBBRegistryCase) -> NodeBBRegistryCase:
    try:
        selected_tests = tuple(
            _parse_string_list(instance.get('selected_test_files_to_run'), 'selected_test_files_to_run')
        )
        fail_to_pass_count, fail_to_pass_sha256 = _canonical_test_fingerprint(
            instance.get('FAIL_TO_PASS'), 'FAIL_TO_PASS'
        )
        pass_to_pass_count, pass_to_pass_sha256 = _canonical_test_fingerprint(
            instance.get('PASS_TO_PASS'), 'PASS_TO_PASS'
        )
        specs = instance.get('spec_dict')
        if not isinstance(specs, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in specs.items()
        ):
            raise ValueError('invalid spec_dict')
        test_patch = instance.get('test_patch')
        before_repo_set_cmd = instance.get('before_repo_set_cmd')
        if not isinstance(test_patch, str):
            raise ValueError('invalid test_patch')
        if not isinstance(before_repo_set_cmd, str):
            raise ValueError('invalid before_repo_set_cmd')
        return NodeBBRegistryCase(
            instance_id=_required_string(instance, 'instance_id').lower(),
            repo=_required_string(instance, 'repo'),
            dataset_type=_required_string(instance, 'dataset_type'),
            base_commit=_required_string(instance, 'base_commit'),
            workdir=_required_string(instance, 'workdir'),
            repo_language=_required_string(instance, 'repo_language'),
            image=_required_string(instance, 'image'),
            selected_tests=selected_tests,
            test_patch_sha256=_sha256(test_patch),
            before_repo_set_cmd_sha256=_sha256(before_repo_set_cmd),
            fail_to_pass_count=fail_to_pass_count,
            fail_to_pass_sha256=fail_to_pass_sha256,
            pass_to_pass_count=pass_to_pass_count,
            pass_to_pass_sha256=pass_to_pass_sha256,
            spec_sha256=tuple(sorted((name, _sha256(value)) for name, value in specs.items())),
            test_file_sha256=expected.test_file_sha256,
            node_version=expected.node_version,
            npm_version=expected.npm_version,
            nodebb_version=expected.nodebb_version,
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f'NodeBB registry fixture dataset fingerprint is invalid: {e}') from e


def needs_nodebb_registry_fixture(instance: Mapping[str, Any]) -> bool:
    raw_instance_id = instance.get('instance_id')
    if not isinstance(raw_instance_id, str):
        return False
    instance_id = raw_instance_id.lower()
    expected = NODEBB_REGISTRY_CASE
    if instance_id != expected.instance_id:
        return False

    actual = _instance_fingerprint(instance, expected)
    if actual != expected:
        mismatches = [
            field.name
            for field in fields(NodeBBRegistryCase)
            if getattr(actual, field.name) != getattr(expected, field.name)
        ]
        raise ValueError(
            'NodeBB registry fixture dataset fingerprint mismatch for exact instance '
            f'{expected.instance_id}: {", ".join(mismatches)}'
        )
    return True


def get_nodebb_registry_fixture_files(instance: Mapping[str, Any], path: str) -> list[FileItem]:
    if not needs_nodebb_registry_fixture(instance):
        return []
    return [
        FileItem(path=f'{path}/{NODEBB_REGISTRY_SERVER_FILENAME}', content=NODEBB_REGISTRY_SERVER_SOURCE),
        FileItem(path=f'{path}/{NODEBB_REGISTRY_MANIFEST_FILENAME}', content=NODEBB_REGISTRY_MANIFEST_JSON),
        FileItem(
            path=f'{path}/{NODEBB_REGISTRY_ASSET_BUNDLE_FILENAME}',
            content=NODEBB_REGISTRY_ASSET_BUNDLE_BASE64,
            encoding='base64',
        ),
    ]


def enrich_nodebb_registry_test_output(
    instance: Mapping[str, Any],
    test_output: dict[str, Any],
    runtime_log_path: Path,
) -> dict[str, Any]:
    if not needs_nodebb_registry_fixture(instance):
        return test_output

    fixture_path = runtime_log_path / 'nodebb_registry_runtime'
    try:
        audit = json.loads((fixture_path / 'audit.json').read_text(encoding='utf-8'))
        if not isinstance(audit, dict) or set(audit) != {'requests', 'violations'}:
            raise ValueError('invalid audit object')

        requests = audit['requests']
        violations = audit['violations']
        if not isinstance(requests, list) or not isinstance(violations, list):
            raise ValueError('requests and violations must be lists')
        for request in requests:
            if (
                not isinstance(request, dict)
                or not {'method', 'path'} <= set(request) <= {'method', 'path', 'query'}
                or not isinstance(request['method'], str)
                or not isinstance(request['path'], str)
                or ('query' in request and not isinstance(request['query'], str))
            ):
                raise ValueError('invalid request record')
        for violation in violations:
            if (
                not isinstance(violation, dict)
                or set(violation) != {'reason', 'path'}
                or not isinstance(violation['reason'], str)
                or not isinstance(violation['path'], str)
            ):
                raise ValueError('invalid violation record')
        if violations:
            raise ValueError('audit contains violations')

        return_code_text = (fixture_path / 'test-return-code').read_text(encoding='utf-8').strip()
        try:
            test_return_code = int(return_code_text)
        except ValueError as e:
            raise ValueError('invalid test return code') from e

        cleanup = (fixture_path / 'cleanup-ok').read_text(encoding='utf-8').strip()
        if cleanup != 'ok':
            raise ValueError('cleanup confirmation is missing or invalid')
    except Exception as e:
        raise ValueError(f'NodeBB registry fixture audit validation failed: {e}') from e

    enriched = copy.deepcopy(test_output)
    enriched['pro_fixture_audit'] = {
        'name': 'nodebb_registry',
        'request_count': len(requests),
        'requests': requests,
        'violations': violations,
        'test_return_code': test_return_code,
        'cleanup': cleanup,
    }
    return enriched


def get_nodebb_registry_fixture_test_commands(
    instance: Mapping[str, Any],
    path: str,
) -> str:
    if not needs_nodebb_registry_fixture(instance):
        return ''

    case = NODEBB_REGISTRY_CASE
    selected_tests = ','.join(case.selected_tests)
    server_path = shlex.quote(f'{path}/{NODEBB_REGISTRY_SERVER_FILENAME}')
    manifest_path = shlex.quote(f'{path}/{NODEBB_REGISTRY_MANIFEST_FILENAME}')
    bundle_path = shlex.quote(f'{path}/{NODEBB_REGISTRY_ASSET_BUNDLE_FILENAME}')
    run_script_path = shlex.quote(f'{path}/run_script.sh')
    parser_path = shlex.quote(f'{path}/parser.py')
    base_dockerfile_path = shlex.quote(f'{path}/baseDockerfile')
    instance_dockerfile_path = shlex.quote(f'{path}/instanceDockerfile')
    stdout_path = shlex.quote(f'{path}/stdout.log')
    stderr_path = shlex.quote(f'{path}/stderr.log')
    test_output_path = shlex.quote(f'{path}/test_output.txt')
    spec_sha256 = dict(case.spec_sha256)

    return textwrap.dedent(
        f"""
        {NODEBB_REGISTRY_FIXTURE_MARKER}
        rm -f {test_output_path}
        set -eu

        nodebb_registry_runtime_dir={shlex.quote(f'{path}/nodebb_registry_runtime')}
        nodebb_registry_server={server_path}
        nodebb_registry_manifest={manifest_path}
        nodebb_registry_bundle={bundle_path}
        nodebb_registry_assets="$nodebb_registry_runtime_dir/assets"
        nodebb_registry_port_file="$nodebb_registry_runtime_dir/port"
        nodebb_registry_audit_file="$nodebb_registry_runtime_dir/audit.json"
        nodebb_registry_server_audit_file="$nodebb_registry_runtime_dir/audit-final.json"
        nodebb_registry_server_log="$nodebb_registry_runtime_dir/server.log"
        nodebb_registry_npmrc="$nodebb_registry_runtime_dir/npmrc"
        nodebb_registry_npm_cache="$nodebb_registry_runtime_dir/npm-cache"
        nodebb_registry_config_backup="$nodebb_registry_runtime_dir/config.json.before-nodebb-registry"
        nodebb_registry_server_pid=

        nodebb_registry_cleanup() {{
          nodebb_registry_cleanup_rc=0
          if [ -f "$nodebb_registry_config_backup" ]; then
            cp "$nodebb_registry_config_backup" config.json || nodebb_registry_cleanup_rc=1
          fi
          if [ -n "${{nodebb_registry_server_pid:-}}" ]; then
            kill "$nodebb_registry_server_pid" 2>/dev/null || true
            wait "$nodebb_registry_server_pid" 2>/dev/null || true
            nodebb_registry_server_pid=
          fi
          return "$nodebb_registry_cleanup_rc"
        }}

        nodebb_registry_on_exit() {{
          nodebb_registry_exit_rc=$?
          trap - EXIT INT TERM
          if ! nodebb_registry_cleanup; then
            nodebb_registry_exit_rc=97
          fi
          exit "$nodebb_registry_exit_rc"
        }}

        nodebb_registry_fail() {{
          printf 'NodeBB registry fixture infrastructure failure: %s\\n' "$*" >&2
          return 1
        }}

        nodebb_registry_check_sha256() {{
          nodebb_registry_expected_sha=$1
          nodebb_registry_file=$2
          nodebb_registry_label=$3
          nodebb_registry_actual_sha=$(sha256sum "$nodebb_registry_file" | awk '{{print $1}}')
          [ "$nodebb_registry_actual_sha" = "$nodebb_registry_expected_sha" ] ||
            nodebb_registry_fail "$nodebb_registry_label checksum mismatch"
        }}

        trap nodebb_registry_on_exit EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        rm -rf "$nodebb_registry_runtime_dir"
        mkdir -p "$nodebb_registry_assets" "$nodebb_registry_npm_cache"
        umask 077

        [ "$(git rev-parse HEAD)" = {shlex.quote(case.base_commit)} ] ||
          nodebb_registry_fail 'unexpected git HEAD'
        [ "$(node --version)" = {shlex.quote(case.node_version)} ] ||
          nodebb_registry_fail 'unexpected Node.js version'
        [ "$(npm --version)" = {shlex.quote(case.npm_version)} ] ||
          nodebb_registry_fail 'unexpected npm version'
        [ "$(node -p "require('./package.json').version")" = {shlex.quote(case.nodebb_version)} ] ||
          nodebb_registry_fail 'unexpected NodeBB version'
        nodebb_registry_check_sha256 {shlex.quote(case.test_file_sha256)} test/plugins.js test/plugins.js
        nodebb_registry_check_sha256 {shlex.quote(spec_sha256['baseDockerfile'])} \
          {base_dockerfile_path} baseDockerfile
        nodebb_registry_check_sha256 {shlex.quote(spec_sha256['instanceDockerfile'])} \
          {instance_dockerfile_path} instanceDockerfile
        nodebb_registry_check_sha256 {shlex.quote(spec_sha256['parser.py'])} {parser_path} parser.py
        nodebb_registry_check_sha256 {shlex.quote(spec_sha256['run_script.sh'])} {run_script_path} run_script.sh
        nodebb_registry_check_sha256 {shlex.quote(NODEBB_REGISTRY_SERVER_SHA256)} \
          "$nodebb_registry_server" registry-server
        nodebb_registry_check_sha256 {shlex.quote(NODEBB_REGISTRY_MANIFEST_SHA256)} \
          "$nodebb_registry_manifest" registry-manifest
        nodebb_registry_check_sha256 {shlex.quote(NODEBB_REGISTRY_ASSET_BUNDLE_SHA256)} \
          "$nodebb_registry_bundle" registry-assets

        tar -xzf "$nodebb_registry_bundle" --no-same-owner -C "$nodebb_registry_assets"
        [ -f config.json ] || nodebb_registry_fail 'config.json is missing'
        cp config.json "$nodebb_registry_config_backup"

        rm -f "$nodebb_registry_port_file" "$nodebb_registry_audit_file" \
          "$nodebb_registry_server_audit_file" "$nodebb_registry_server_log"
        python3 "$nodebb_registry_server" \
          --manifest "$nodebb_registry_manifest" \
          --asset-dir "$nodebb_registry_assets" \
          --port-file "$nodebb_registry_port_file" \
          --audit-file "$nodebb_registry_server_audit_file" \
          >"$nodebb_registry_server_log" 2>&1 &
        nodebb_registry_server_pid=$!

        nodebb_registry_ready=0
        for _ in $(seq 1 200); do
          if [ -s "$nodebb_registry_port_file" ]; then
            nodebb_registry_ready=1
            break
          fi
          if ! kill -0 "$nodebb_registry_server_pid" 2>/dev/null; then
            cat "$nodebb_registry_server_log" >&2
            nodebb_registry_fail 'local registry exited before readiness'
          fi
          sleep 0.05
        done
        if [ "$nodebb_registry_ready" -ne 1 ]; then
          cat "$nodebb_registry_server_log" >&2
          nodebb_registry_fail 'local registry readiness timeout'
        fi

        nodebb_registry_port=$(cat "$nodebb_registry_port_file")
        case "$nodebb_registry_port" in
          ''|*[!0-9]*) nodebb_registry_fail 'invalid local registry port' ;;
        esac
        nodebb_registry_url="http://127.0.0.1:$nodebb_registry_port"

        python3 - "$nodebb_registry_port" <<'PY'
        import json
        import sys
        from pathlib import Path

        config_path = Path('config.json')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['registry'] = f'http://127.0.0.1:{{sys.argv[1]}}'
        config_path.write_text(json.dumps(config, indent=4) + '\\n', encoding='utf-8')
        PY

        cat > "$nodebb_registry_npmrc" <<EOF
        registry=$nodebb_registry_url/
        cache=$nodebb_registry_npm_cache
        audit=false
        fund=false
        update-notifier=false
        fetch-retries=0
        fetch-retry-maxtimeout=1000
        fetch-retry-mintimeout=100
        EOF

        set +e
        NPM_CONFIG_USERCONFIG="$nodebb_registry_npmrc" \
        npm_config_userconfig="$nodebb_registry_npmrc" \
        NPM_CONFIG_CACHE="$nodebb_registry_npm_cache" \
        npm_config_cache="$nodebb_registry_npm_cache" \
        NPM_CONFIG_REGISTRY="$nodebb_registry_url/" \
        npm_config_registry="$nodebb_registry_url/" \
        bash {run_script_path} {shlex.quote(selected_tests)} > {stdout_path} 2> {stderr_path}
        nodebb_test_rc=$?
        set -e
        printf '%s\\n' "$nodebb_test_rc" > "$nodebb_registry_runtime_dir/test-return-code"

        python3 - "$nodebb_registry_port" "$nodebb_registry_audit_file" {stderr_path} "$nodebb_test_rc" <<'PY'
        import json
        import sys
        import urllib.request
        from pathlib import Path

        port, audit_path, stderr_path, test_rc = sys.argv[1:]
        with urllib.request.urlopen(f'http://127.0.0.1:{{port}}/__audit', timeout=5) as response:
            audit_bytes = response.read()
        Path(audit_path).write_bytes(audit_bytes)
        audit = json.loads(audit_bytes)
        requests = audit.get('requests')
        violations = audit.get('violations')
        if not isinstance(requests, list) or not isinstance(violations, list):
            raise RuntimeError('invalid NodeBB registry audit response')
        summary = (
            f'[nodebb-registry-audit] requests={{len(requests)}} '
            f'violations={{len(violations)}} test_rc={{test_rc}}\\n'
        )
        with Path(stderr_path).open('a', encoding='utf-8') as stderr:
            stderr.write(summary)
            if violations:
                stderr.write(
                    '[nodebb-registry-audit] violation_details='
                    + json.dumps(violations, ensure_ascii=True, separators=(',', ':'))
                    + '\\n'
                )
        if violations:
            raise RuntimeError(f'NodeBB registry audit violations: {{violations!r}}')
        PY

        if ! nodebb_registry_cleanup; then
          nodebb_registry_fail 'failed to restore config.json'
        fi
        printf 'ok\\n' > "$nodebb_registry_runtime_dir/cleanup-ok"

        set +e
        python {parser_path} {stdout_path} {stderr_path} {test_output_path}
        nodebb_parser_rc=$?
        set -e
        exit "$nodebb_parser_rc"
        """
    ).strip()
