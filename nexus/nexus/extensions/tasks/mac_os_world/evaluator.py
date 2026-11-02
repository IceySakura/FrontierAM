"""macOSWorld evaluator: runs grading_command list via runtime.run_command and aggregates score."""

from __future__ import annotations

import logging

from typing import Any


logger = logging.getLogger(__name__)

# grading_command entry where score_weight == 100 is treated as the primary (binary) grading point.
_BINARY_WEIGHT = 100


async def evaluate_mac_os_world(
    runtime: Any,
    grading_command: list[list[Any]] | None,
    *,
    binary_grading: bool = True,
) -> dict[str, Any]:
    """Evaluate one macOSWorld task using its grading_command list.

    Each entry in grading_command is [shell_command, score_weight].
    With binary_grading=True (default, matches official macOSWorld behaviour):
      - only entries with score_weight == 100 are considered
      - the first successfully completed command whose stdout contains "true" wins and returns score 1.0
      - stop on a command error, as the official evaluator does
      - if none match, returns score 0.0
    With binary_grading=False all entries are run and scores are averaged.
    """
    if not grading_command:
        return {'score': 0.0, 'error': 'No grading_command configured', 'results': []}

    entries = [e for e in grading_command if e[1] == _BINARY_WEIGHT] if binary_grading else grading_command

    results: list[dict[str, Any]] = []
    error: str | None = None
    for entry in entries:
        command, weight = entry[0], entry[1]
        try:
            response = await runtime.run_command(command, timeout=60)
            stdout = str(_get_field(response, 'stdout', 'output') or '')
            stderr = str(_get_field(response, 'stderr', 'error') or '')
            return_code = _get_field(response, 'return_code', 'returncode')
            status = _get_field(response, 'status')
            status = getattr(status, 'value', status)
            succeeded = return_code == 0 and status in (None, 'Finished')
            passed = succeeded and 'true' in stdout.lower()
            results.append({
                'command': command,
                'weight': weight,
                'passed': passed,
                'stdout': stdout,
                'stderr': stderr,
                'return_code': return_code,
                'status': status,
            })
            if not succeeded:
                error = f'Grading command failed: status={status}, return_code={return_code}'
                results[-1]['error'] = error
                if binary_grading:
                    break
            if binary_grading and passed:
                break
        except Exception as exc:
            logger.warning('macOSWorld grading_command failed: %s — %s', command[:80], exc)
            error = f'{type(exc).__name__}: {exc}'
            results.append({'command': command, 'weight': weight, 'passed': False, 'error': error})
            if binary_grading:
                break

    if binary_grading:
        score = 1.0 if any(r['passed'] for r in results) else 0.0
    else:
        passed_weights = sum(r['weight'] for r in results if r['passed'])
        total_weights = sum(r['weight'] for r in results) or 1
        score = passed_weights / total_weights

    result = {'score': score, 'results': results, 'binary_grading': binary_grading}
    if error:
        result['error'] = error
    return result


def _get_field(response: Any, *names: str) -> Any:
    for name in names:
        value = response.get(name) if isinstance(response, dict) else getattr(response, name, None)
        if value is not None:
            return value
    return None
