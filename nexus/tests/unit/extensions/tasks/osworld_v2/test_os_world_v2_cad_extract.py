"""Only a reproduced, version-gated native STEP import crash permits fallback."""

import hashlib
import json
import signal
import subprocess

from pathlib import Path

import pytest

from nexus.extensions.tasks.os_world_v2.adapter.cad_extract import (
    MARKER,
    STREAM_LIMIT,
    ExtractSupervisor,
    _stream,
    confirmed_import_crash,
)


def result(code=0, stdout=b'', stderr=b''):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def probe(code=-signal.SIGSEGV, *, after=False, version=None):
    row = {
        'stage': 'before_import',
        'freecad': version or ['0', '19'],
        'occt': '7.5.1',
        'python': 'test',
    }
    output = MARKER + json.dumps(row) + '\n'
    if after:
        output += MARKER + json.dumps({**row, 'stage': 'after_import'}) + '\n'
    return result(code, output.encode(), b'Fatal Python error: Segmentation fault')


@pytest.fixture
def invocation(tmp_path):
    root = tmp_path / 'frozen'
    work = tmp_path / 'official-temp'
    root.mkdir()
    work.mkdir()
    for name in ('cad_freecad_extract.py', 'submission.step', 'reference.step'):
        (root / name).write_text(name)
        (work / name).write_text(name)
    cmd = [
        '/usr/bin/python3',
        *(str(work / name) for name in ('cad_freecad_extract.py', 'submission.step', 'reference.step')),
    ]
    opts = {'env': {'HOME': str(work)}, 'cwd': work, 'capture_output': True, 'timeout': 240, 'preexec_fn': lambda: None}
    return root, cmd, opts


def test_confirmation_requires_native_signal_at_import_and_known_versions():
    assert confirmed_import_crash(probe())
    assert not confirmed_import_crash(probe(1))
    assert not confirmed_import_crash(probe(-signal.SIGKILL))
    assert not confirmed_import_crash(probe(after=True))
    assert not confirmed_import_crash(probe(version=['1', '0']))
    assert not confirmed_import_crash(result(-signal.SIGSEGV, b'no import marker'))


def test_success_keeps_exact_original_command_and_result(invocation, monkeypatch):
    root, cmd, opts = invocation
    expected = result(stdout=b'METRICS::original')
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return expected
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(subprocess, 'run', run)
    supervisor = ExtractSupervisor(root)
    assert supervisor.run(cmd, **opts) is expected
    assert calls[0] == (cmd, opts)
    assert supervisor.diagnostics['import_path'] == 'official'
    assert supervisor.diagnostics['version_probe_error'] == 'TimeoutExpired'
    assert len(calls) == 2


@pytest.mark.parametrize('code', [2, -signal.SIGKILL, -signal.SIGXCPU, -signal.SIGABRT])
def test_other_exit_errors_never_retry(invocation, monkeypatch, code):
    root, cmd, opts = invocation
    calls = []
    expected = result(code)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: calls.append(a) or expected)
    assert ExtractSupervisor(root).run(cmd, **opts) is expected
    assert len(calls) == 1


@pytest.mark.parametrize('candidate', [probe(1), probe(-signal.SIGKILL), probe(after=True)])
def test_exit_one_is_not_enough_to_authorize_fallback(invocation, monkeypatch, candidate):
    root, cmd, opts = invocation
    original = result(1, stderr=b'original error')
    responses = iter([original, candidate])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: next(responses))
    supervisor = ExtractSupervisor(root)
    assert supervisor.run(cmd, **opts) is original
    assert supervisor.diagnostics['import_path'] == 'official'
    assert supervisor.diagnostics['fallback_declined'] == 'native_import_crash_not_confirmed'


def test_reference_failure_keeps_original_failure(invocation, monkeypatch):
    root, cmd, opts = invocation
    original = result(1)
    responses = iter([original, probe(), probe()])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: next(responses))
    supervisor = ExtractSupervisor(root)
    assert supervisor.run(cmd, **opts) is original
    assert supervisor.diagnostics['fallback_declined'] == 'official_reference_import_failed'


def test_single_fallback_preserves_limits_and_failure_diagnostics(invocation, monkeypatch):
    root, cmd, opts = invocation
    responses = iter([result(1, b'original stdout', b''), probe(), probe(0, after=True), result(0, b'METRICS::ok')])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(subprocess, 'run', run)
    supervisor = ExtractSupervisor(root)
    assert supervisor.run(cmd, **opts).stdout == b'METRICS::ok'
    assert len(calls) == 4
    assert calls[-1][0][2] == 'compat'
    assert calls[-1][0][3:] == cmd[1:]
    for _, kw in calls:
        assert kw['preexec_fn'] is opts['preexec_fn']
        assert kw['env'] is opts['env']
        assert kw['cwd'] == opts['cwd']
        assert kw['timeout'] <= opts['timeout']
    diagnostics = supervisor.diagnostics
    assert diagnostics['attempts'][0]['stdout']['text'] == 'original stdout'
    assert diagnostics['attempts'][1]['signal'] == signal.SIGSEGV
    assert diagnostics['import_path'] == 'part_shape_read_after_confirmed_native_crash'


def test_fallback_failure_is_not_a_zero_score(invocation, monkeypatch):
    root, cmd, opts = invocation
    failure = result(1, stderr=b'compat failed')
    responses = iter([result(1), probe(), probe(0, after=True), failure])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: next(responses))
    assert ExtractSupervisor(root).run(cmd, **opts) is failure


def test_timeout_propagates_without_compatibility_run(invocation, monkeypatch):
    root, cmd, opts = invocation

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 240, output=b'partial', stderr=b'detail')

    monkeypatch.setattr(subprocess, 'run', run)
    supervisor = ExtractSupervisor(root)
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor.run(cmd, **opts)
    assert len(supervisor.diagnostics['attempts']) == 1
    assert supervisor.diagnostics['attempts'][0]['stdout']['text'] == 'partial'


def test_tampered_extractor_rejected(invocation):
    root, cmd, opts = invocation
    Path(cmd[1]).write_text('changed')
    with pytest.raises(ValueError, match='differs from frozen'):
        ExtractSupervisor(root).run(cmd, **opts)


def test_stream_diagnostics_bounded_and_hashed():
    data = b'x' * (STREAM_LIMIT * 2)
    record = _stream(data)
    assert record['truncated']
    assert len(record['text']) == STREAM_LIMIT
    assert record['sha256'] == hashlib.sha256(data).hexdigest()
