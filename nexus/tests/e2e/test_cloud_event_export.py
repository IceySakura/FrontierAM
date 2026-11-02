"""Opt-in real runner-test export verification.

Set NEXUS_EXPORT_E2E_CONFIG to a private JSON containing trace_db_config,
db_password_file and taihu_token_file paths. Optional keys are output_dir,
environment, control_url and application_url. This test submits test-owned
Batch and Live executions; the trace database is only read.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).parents[2]


def run_owned_command(command, *, timeout=3600, cleanup_grace=180, evidence):
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        report = {
            'passed': False,
            'pid': process.pid,
            'forced_group_kill': False,
            'remote_cleanup': 'unverified; inspect ownership.json and cleanup.json in the run directory',
        }
        try:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=cleanup_grace)
            except subprocess.TimeoutExpired:
                report['forced_group_kill'] = True
        finally:
            # Even a successfully drained harness may have a surviving descendant.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                report['forced_group_kill'] = True
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(evidence, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(report, stream, indent=2)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def test_runner_test_cloud_event_export():
    config_path = os.environ.get('NEXUS_EXPORT_E2E_CONFIG')
    if not config_path:
        pytest.skip('set NEXUS_EXPORT_E2E_CONFIG to explicitly enable runner-test submissions')
    assert config_path is not None
    config = json.loads(Path(config_path).read_text())
    command = [sys.executable, '-m', 'tests.e2e.distributed.run_cloud_event_export']
    for key in ('trace_db_config', 'db_password_file', 'taihu_token_file'):
        command.extend([f'--{key.replace("_", "-")}', str(config[key])])
    for key in ('output_dir', 'environment', 'control_url', 'application_url'):
        if key in config:
            command.extend([f'--{key.replace("_", "-")}', str(config[key])])
    run_owned_command(
        command,
        evidence=Path(config.get('output_dir', ROOT / 'outputs/cloud-event-export-e2e')) / 'wrapper-timeout.json',
    )
