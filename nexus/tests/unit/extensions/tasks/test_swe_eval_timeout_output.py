from __future__ import annotations

import shutil
import subprocess
import uuid

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.extensions.tasks.swe import dataset_handler
from nexus.extensions.tasks.swe import eval as swe_eval
from nexus.extensions.tasks.swe.constants import SWEErrorCode


@pytest.fixture
def evaluate_timeout(monkeypatch, tmp_path):
    instance_id = f'timeout-output-{uuid.uuid4().hex}'
    log_root = Path('/tmp/swebench') / instance_id

    def evaluate(output, *, fail_write=False):
        def run(command, **kwargs):
            if isinstance(command, list) and command[-1].endswith('/eval.sh'):
                raise subprocess.TimeoutExpired(command, 1, output=output)
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

        monkeypatch.setattr(subprocess, 'run', run)
        if fail_write:
            original_write = Path.write_text

            def write_text(path, *args, **kwargs):
                if path == log_root / 'test_output.txt':
                    raise OSError('test output storage unavailable')
                return original_write(path, *args, **kwargs)

            monkeypatch.setattr(Path, 'write_text', write_text)
        return swe_eval._evaluate_patch(
            {'workdir': str(tmp_path)},
            'diff --git a/example.txt b/example.txt\n--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-old\n+new\n',
            instance_id,
            1,
            'printf "partial output\\n"\nsleep 10\n',
        )

    yield evaluate
    shutil.rmtree(log_root, ignore_errors=True)


@pytest.mark.parametrize('output', [b'partial output\n\xe4\xb8', 'partial output\n', None])
def test_timeout_keeps_captured_output_without_completing_evaluation(evaluate_timeout, output):
    result = evaluate_timeout(output)
    expected = output.decode('utf-8', errors='replace') if isinstance(output, bytes) else output or ''

    assert result['completed'] is False
    assert result['resolved'] is False
    assert result['error_code'] == SWEErrorCode.TEST_EXECUTION_TIMEOUT
    assert result['test_output'] == expected
    assert Path(result['test_output_path']).read_text(encoding='utf-8') == expected


def test_output_write_failure_preserves_original_timeout_and_captured_text(evaluate_timeout):
    result = evaluate_timeout(b'partial output\n', fail_write=True)

    assert result['completed'] is False
    assert result['error_code'] == SWEErrorCode.TEST_EXECUTION_TIMEOUT
    assert result['test_output'] == 'partial output\n'
    assert result['test_output_path'] == ''


def test_timeout_output_cannot_be_parsed_as_a_completed_verdict(monkeypatch):
    handler = SimpleNamespace(get_eval_script=lambda instance: '', get_eval_report_map=Mock())
    monkeypatch.setattr(dataset_handler, 'get_handler', lambda dataset_type: handler)
    monkeypatch.setattr(
        swe_eval,
        '_evaluate_patch',
        lambda **kwargs: {
            'completed': False,
            'error': 'Test execution timed out',
            'error_code': SWEErrorCode.TEST_EXECUTION_TIMEOUT,
            'test_output': 'ALL_TESTS_PASSED\n',
        },
    )

    result = swe_eval._evaluate_and_grade('patch', {'dataset_type': 'test'}, 'timeout-test', 1)

    assert result['completed'] is False
    assert result['resolved'] is False
    assert result['score'] == 0
    assert result['report_json'] == ''
    handler.get_eval_report_map.assert_not_called()
