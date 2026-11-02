"""Small, self-contained Harbor fixtures used by distributed E2E tests.

The runner-test entrypoint can import the fixture and argument helpers from
this module without importing the larger dynamic-cluster harness. The fixture
deliberately has no coordination service or barrier: a verifier only depends
on files that belong to its own Harbor task.
"""

from __future__ import annotations

import json
import shutil

from pathlib import Path


HARBOR_DATASET_ITEM_IDS = ('harbor-directory', 'harbor-zip')
HARBOR_DATASET_MARKERS = {
    'harbor-directory': 'fixture-harbor-directory',
    'harbor-zip': 'fixture-harbor-zip',
}
DEFAULT_GONGFENG_API_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'


def _toml_string(value: str) -> str:
    """Return a TOML basic string without another dependency."""

    return json.dumps(str(value))


def _verifier_script(*, item_id: str, marker: str) -> str:
    reward = json.dumps({'reward': 1.0, 'item_id': item_id, 'marker': marker}, sort_keys=True)
    sleep_line = f'sleep "{chr(36)}{{HARBOR_VERIFIER_SLEEP_SECONDS:-1}}"'
    return f"""#!/bin/sh
set -eu
test "$(cat /app/fixture/marker.txt)" = {_toml_string(marker)}
{sleep_line}
mkdir -p /logs/verifier
cat > /logs/verifier/reward.json <<'JSON'
{reward}
JSON
"""


def _task_toml(*, docker_image: str) -> str:
    return f"""schema_version = "1.2"

[environment]
docker_image = {_toml_string(docker_image)}
workdir = "/app"
cpus = 1
memory_mb = 1024
network_mode = "public"

[environment.upload]
source = "fixture/marker.txt"
target = "/app/fixture/marker.txt"

[agent]
timeout_sec = 120

[verifier]
environment_mode = "shared"
timeout_sec = 180
"""


def write_harbor_dataset_fixture(root: Path, *, workload_image: str) -> Path:
    """Write two portable NOP Harbor tasks and return their root directory."""

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for item_id in HARBOR_DATASET_ITEM_IDS:
        marker = HARBOR_DATASET_MARKERS[item_id]
        task_dir = root / item_id
        fixture_dir = task_dir / 'fixture'
        tests_dir = task_dir / 'tests'
        fixture_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'task.toml').write_text(_task_toml(docker_image=workload_image), encoding='utf-8')
        (task_dir / 'instruction.md').write_text(f'Verify {item_id}.\n', encoding='utf-8')
        (fixture_dir / 'marker.txt').write_text(f'{marker}\n', encoding='utf-8')
        verifier = tests_dir / 'test.sh'
        verifier.write_text(_verifier_script(item_id=item_id, marker=marker), encoding='utf-8')
        verifier.chmod(0o755)
    return root


def prepare_harbor_source_root(project_root: Path, source_root: Path) -> Path:
    """Prepare the minimal source package inputs required by the cloud builder.

    ``--project-root`` controls the wheel/lock generation on the submitter, while
    ``--source`` is the directory that gets uploaded as the source package.  The
    dependency builder validates that the uploaded package contains the project
    metadata and lock file, so keep the two roots explicit and copy only the
    three files needed by that contract.
    """

    project_root = Path(project_root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        source = project_root / name
        if not source.is_file():
            raise FileNotFoundError(f'Harbor source package input is missing: {source}')
        shutil.copyfile(source, source_root / name)
    return source_root


def build_harbor_batch_cli_args(
    *,
    data_value: str,
    project_root: Path,
    source_root: Path,
    control_url: str,
    environment_id: str,
    taihu_token_file: Path,
    idempotency_key: str,
    runtime_api_base_url: str = DEFAULT_GONGFENG_API_BASE_URL,
) -> list[str]:
    """Build the portable local-generation CLI invocation used by E2E tests."""

    runtime_config = json.dumps(
        {'api_base_url': runtime_api_base_url, 'token__env': 'GONGFENG_TOKEN'},
        sort_keys=True,
    )
    return [
        'batch',
        '--generation',
        'local',
        '--task-list-version',
        '2',
        '--data',
        data_value,
        '--use',
        'harbor_bridge:mode=nop',
        '--set',
        'runtime_provider_name=gongfeng',
        '--set-json',
        f'runtime_provider_config={runtime_config}',
        '--environment',
        environment_id,
        '--project-root',
        str(project_root),
        '--source',
        str(source_root),
        '--control-url',
        control_url,
        '--taihu-token-file',
        str(taihu_token_file),
        '--idempotency-key',
        idempotency_key,
    ]


__all__ = [
    'DEFAULT_GONGFENG_API_BASE_URL',
    'HARBOR_DATASET_ITEM_IDS',
    'HARBOR_DATASET_MARKERS',
    'build_harbor_batch_cli_args',
    'prepare_harbor_source_root',
    'write_harbor_dataset_fixture',
]
