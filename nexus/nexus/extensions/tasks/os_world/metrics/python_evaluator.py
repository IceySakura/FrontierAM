"""Run Python checker code in a disposable process, never in runner threads."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys

from pathlib import Path


def check_python_file_by_test_suite(actual_files, test_file, **options) -> float:
    from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError

    if not test_file or not Path(test_file).is_file():
        raise EvaluatorInfrastructureError('Python evaluator fixture is missing')
    request = {
        'test_file': str(Path(test_file).resolve()),
        'function': options.get('test_function_name', 'test'),
    }
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvaluatorInfrastructureError('Python evaluator exceeded its 120s execution budget') from exc
    if result.returncode:
        raise EvaluatorInfrastructureError(f'Python evaluator process exited with code {result.returncode}')
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise EvaluatorInfrastructureError('Python evaluator returned an invalid result') from exc
    if not isinstance(payload, dict):
        raise EvaluatorInfrastructureError('Python evaluator returned an invalid result')
    if payload.get('dependency_error'):
        # Only the module identifier is returned, never the checker output,
        # environment, traceback or evaluated program's arbitrary text.
        raise EvaluatorInfrastructureError('Python evaluator could not import a required dependency')
    if not isinstance(payload.get('score'), (int, float)):
        raise EvaluatorInfrastructureError('Python evaluator returned no score')
    return float(payload['score'])


def _evaluate(request: dict) -> dict:
    test_file = Path(request['test_file'])
    os.chdir(test_file.parent)
    sys.path.insert(0, str(test_file.parent))
    try:
        spec = importlib.util.spec_from_file_location('_osworld_checker', test_file)
        if spec is None or spec.loader is None:
            return {'dependency_error': True}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = getattr(module, request['function'])()
        score = max(0.0, min(1.0, float(result))) if isinstance(result, (int, float)) else float(bool(result))
        return {'score': score}
    except ImportError:
        return {'dependency_error': True}
    except Exception:
        # Invalid submitted code / test assertion failures remain negative
        # scores. Process isolation removes cwd/sys.modules races entirely.
        return {'score': 0.0}


if __name__ == '__main__':
    request = json.load(sys.stdin)
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        outcome = _evaluate(request)
    print(json.dumps(outcome))
