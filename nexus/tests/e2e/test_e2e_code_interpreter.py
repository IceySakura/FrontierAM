import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


def _require_env(var_name: str) -> str | None:
    value = os.environ.get(var_name)
    return value if value and value.strip() else None


def _runtime_image_from_config(config_path: Path) -> str:
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith('runtime_image:'):
            return stripped.split(':', 1)[1].strip().strip('"').strip("'")
    return 'nexus-runtime:dev'


@pytest.mark.e2e
@pytest.mark.parametrize(
    'tool_name,log_name',
    [
        ('jupyter_code_interpreter', 'jupyter_tool_run.log'),
        ('python_code_interpreter', 'python_code_interpreter_run.log'),
    ],
)
def test_run_single_code_interpreter(
    tool_name: str,
    log_name: str,
    ensure_runtime_image,
    nexus_run_env,
    perf_logger_factory,
) -> None:
    if shutil.which('nexus') is None:
        pytest.skip('nexus CLI not found in PATH')

    required_envs = ['SEARCH_API_KEY', 'SEARCH_API_USER', 'OPENAI_API_KEY']
    missing = [name for name in required_envs if _require_env(name) is None]
    if missing:
        pytest.skip(f'missing env vars for real run: {", ".join(missing)}')

    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / 'tests' / 'e2e' / 'config' / 'infoseek_example.yaml'
    start_total = time.monotonic()
    start_image = time.monotonic()
    yaml_path = ensure_runtime_image(yaml_path)
    image_elapsed = time.monotonic() - start_image
    image_tag = _runtime_image_from_config(yaml_path)
    log_path = repo_root / 'workspace' / 'research' / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    perf_logger = perf_logger_factory(repo_root, tool_name, image_tag)

    start_env = time.monotonic()
    env = nexus_run_env(tool_name)
    env_elapsed = time.monotonic() - start_env
    start_run = time.monotonic()
    result = subprocess.run(
        ['nexus', 'run-single', str(yaml_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    run_elapsed = time.monotonic() - start_run

    combined = result.stdout + '\n' + result.stderr
    start_write = time.monotonic()
    log_path.write_text(combined)
    write_elapsed = time.monotonic() - start_write
    total_elapsed = time.monotonic() - start_total
    perf_logger.write(
        combined=combined,
        image_elapsed=image_elapsed,
        env_elapsed=env_elapsed,
        run_elapsed=run_elapsed,
        write_elapsed=write_elapsed,
        total_elapsed=total_elapsed,
        returncode=result.returncode,
    )

    assert result.returncode == 0, f'run-single failed: {result.stderr or result.stdout}'
    assert 'Successfully executed' in combined
    assert f'Tool call: {tool_name}' in combined
