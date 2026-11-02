"""Standalone, grader-only supervision of the pinned official CAD extractor.

The first invocation is untouched. Only a reproduced SIGSEGV in the known
FreeCAD 0.19/OCCT 7.5.1 STEP assembly importer permits one compatibility run.
This module is also uploaded beside runner.py and executes in child processes.
"""

import faulthandler
import hashlib
import json
import runpy
import signal
import subprocess
import sys
import time

from contextlib import suppress
from pathlib import Path
from typing import Any


VERSION = 'step-import-compat-v1'
MARKER = 'NEXUS_CAD_PROBE::'
# The minimal grader transport bounds the entire /execute output to 1MiB.
# Leave room for the score report and JSON escaping across all four attempts.
STREAM_LIMIT = 32 * 1024


def _stream(data):
    data = data or b''
    if isinstance(data, str):
        data = data.encode()
    return {
        'text': data[-STREAM_LIMIT:].decode(errors='replace'),
        'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
        'truncated': len(data) > STREAM_LIMIT,
    }


def _event(result):
    return {
        'returncode': result.returncode,
        'signal': -result.returncode if result.returncode < 0 else None,
        'stdout': _stream(result.stdout),
        'stderr': _stream(result.stderr),
    }


def _markers(result):
    records = []
    for line in (result.stdout or b'').decode(errors='replace').splitlines():
        if line.startswith(MARKER):
            with suppress(ValueError):
                records.append(json.loads(line[len(MARKER) :]))
    return records


def confirmed_import_crash(result):
    records = _markers(result)
    before = next((row for row in records if row.get('stage') == 'before_import'), {})
    return (
        result.returncode == -signal.SIGSEGV
        and before.get('freecad', [])[:2] == ['0', '19']
        and before.get('occt') == '7.5.1'
        and not any(row.get('stage') == 'after_import' for row in records)
    )


class ExtractSupervisor:
    def __init__(self, root, *, enabled=True):
        self.root = Path(root)
        self.enabled = enabled
        self.diagnostics: dict[str, Any] = {
            'version': VERSION,
            'loader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'import_path': 'not_executed',
            'attempts': [],
        }

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def _is_extractor(self, command):
        if not isinstance(command, (list, tuple)) or len(command) != 4:
            return False
        if Path(command[1]).name != 'cad_freecad_extract.py':
            return False
        # Do not change dependency probes or arbitrary subprocess commands.
        for actual, trusted in zip(
            command[1:], ('cad_freecad_extract.py', 'submission.step', 'reference.step'), strict=True
        ):
            if (
                hashlib.sha256(Path(actual).read_bytes()).digest()
                != hashlib.sha256((self.root / trusted).read_bytes()).digest()
            ):
                raise ValueError('CAD extractor invocation differs from frozen inputs')
        return True

    def _run(self, label, command, kwargs):
        started = time.monotonic()
        try:
            result = subprocess.run(command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            self.diagnostics['attempts'].append({
                'label': label,
                'timeout': True,
                'elapsed_s': time.monotonic() - started,
                'stdout': _stream(exc.stdout),
                'stderr': _stream(exc.stderr),
            })
            raise
        self.diagnostics['attempts'].append({'label': label, 'elapsed_s': time.monotonic() - started, **_event(result)})
        for row in _markers(result):
            if 'freecad' in row:
                self.diagnostics['environment'] = {key: row[key] for key in ('freecad', 'occt', 'python')}
        return result

    def run(self, command, **kwargs):
        if not self._is_extractor(command):
            return subprocess.run(command, **kwargs)
        self.diagnostics['import_path'] = 'official'
        deadline = time.monotonic() + kwargs['timeout']
        original = self._run('official', command, kwargs)

        def child(mode, arguments, *, limit):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            options = {**kwargs, 'timeout': min(limit, remaining)}
            return self._run(mode, [command[0], str(Path(__file__).resolve()), mode, *arguments], options)

        if original.returncode == 0:
            # Best-effort metadata only; a metadata probe cannot invalidate a score.
            try:
                child('versions', [], limit=10)
            except Exception as exc:
                self.diagnostics['version_probe_error'] = type(exc).__name__
            return original
        if not self.enabled or original.returncode not in {1, -signal.SIGSEGV}:
            return original
        try:
            candidate = child('probe', [command[2]], limit=30)
            if not confirmed_import_crash(candidate):
                self.diagnostics['fallback_declined'] = 'native_import_crash_not_confirmed'
                return original
            # Trusted reference must work in the original loader. Never hide
            # reference/dependency failures with an alternate parsing path.
            reference = child('probe', [command[3]], limit=30)
            if reference.returncode != 0 or not any(row.get('stage') == 'after_import' for row in _markers(reference)):
                self.diagnostics['fallback_declined'] = 'official_reference_import_failed'
                return original
        except Exception as exc:
            self.diagnostics['fallback_declined'] = f'confirmation_failed:{type(exc).__name__}'
            return original
        self.diagnostics['import_path'] = 'part_shape_read_after_confirmed_native_crash'
        # Same executable, environment, cwd and preexec resource limits. All
        # attempts share the original 240s wall deadline, not a fresh budget.
        return child('compat', list(command[1:]), limit=kwargs['timeout'])


def child_main():
    # Imports are intentionally confined to the resource-limited subprocess.
    import FreeCAD
    import Import
    import Part

    mode = sys.argv[1]
    versions = {'freecad': FreeCAD.Version(), 'occt': Part.OCC_VERSION, 'python': sys.version}

    def emit(stage):
        print(MARKER + json.dumps({'stage': stage, **versions}), flush=True)

    if mode == 'versions':
        emit('versions')
        return
    if mode == 'probe':
        # FreeCAD installs its own signal handlers during import. Reinstall
        # faulthandler afterwards so SIGSEGV stays a native signal, not exit(1).
        faulthandler.disable()
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        faulthandler.enable()
        doc = FreeCAD.newDocument('step_extract')
        emit('before_import')
        Import.insert(sys.argv[2], doc.Name)
        emit('after_import')
        FreeCAD.closeDocument(doc.Name)
        return
    if mode != 'compat':
        raise ValueError('Unknown CAD child mode')
    emit('compat')
    extractor, candidate, reference = sys.argv[2:]
    namespace = runpy.run_path(extractor, run_name='nexus_trusted_extractor')
    official_load = namespace['_load_document']

    def load_document(path):
        if Path(path).resolve() != Path(candidate).resolve():
            return official_load(path)
        if Path(path).suffix.lower() not in {'.step', '.stp'}:
            raise ValueError('Compatibility loader only accepts frozen STEP')
        shape = Part.Shape()
        shape.read(path)
        doc = FreeCAD.newDocument('step_extract')
        doc.addObject('Part::Feature', 'FrozenSTEP').Shape = shape
        doc.recompute()
        return doc

    # Preserve all solids/placements. Do not fuse, heal, filter, or change the
    # official largest-solid measurements, hard gates, tolerances or scoring.
    namespace['main'].__globals__['_load_document'] = load_document
    raise SystemExit(namespace['main']([extractor, candidate, reference]))


if __name__ == '__main__':
    child_main()
