"""Shared helpers for mini SWE agent benchmark evaluators."""

import base64
import binascii
import json
import logging
import os
import re
import shlex

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ENV_TEMPLATE_RE = re.compile(r'\$\{(?:oc\.env:)?([A-Za-z_][A-Za-z0-9_]*)\}')
VERIFIER_LOG_DIR = '/logs/verifier'
VERIFIER_LOG_PATH = f'{VERIFIER_LOG_DIR}/test.txt'
VERIFIER_RESULT_PATH = f'{VERIFIER_LOG_DIR}/verifier_result.json'
VERIFIER_LOG_MAX_CHARS = 16 * 1024
VERIFIER_OUTPUT_NAMES = ('reward.txt', 'reward.json', 'ctrf.json', 'verifier_result.json')

logger = logging.getLogger(__name__)


def verifier_registry_network_target(verifier_image: str) -> str:
    """Return the exact TLS registry endpoint required to pull a verifier image."""
    repository, separator, _digest = verifier_image.rpartition('@sha256:')
    if not separator or '/' not in repository:
        raise ValueError('verifier_image must use an explicit registry and sha256 digest')
    registry = repository.split('/', 1)[0]
    if not registry or registry.startswith('['):
        raise ValueError(f'unsupported verifier image registry: {registry!r}')
    return registry if ':' in registry else f'{registry}:443'


def resolve_verifier_env(raw_env: dict[str, str] | None, raw_refs: dict[str, str] | None) -> dict[str, str]:
    """Resolve verifier-only host environment references after the agent exits."""

    def expand(value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            if env_name not in os.environ:
                raise RuntimeError(f'Required verifier environment variable is not set: {env_name}')
            return os.environ[env_name]

        return ENV_TEMPLATE_RE.sub(replace, value)

    resolved = {str(key): expand(str(value)) for key, value in (raw_env or {}).items()}
    for key, env_name in (raw_refs or {}).items():
        if env_name not in os.environ:
            raise RuntimeError(f'Required verifier environment variable is not set: {env_name}')
        resolved[str(key)] = os.environ[env_name]
    return resolved


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in an LLM response."""
    json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not json_match:
        return None
    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_llm_judge_result(text: str) -> tuple[float, dict[str, Any]]:
    """Parse a strict boolean ``verdict`` and non-empty ``reasoning`` result."""
    payload = parse_first_json_object(text)
    if payload is None:
        raise ValueError('LLM judge response does not contain a valid JSON object')

    verdict = payload.get('verdict')
    if not isinstance(verdict, bool):
        raise ValueError('LLM judge verdict must be a JSON boolean')

    reasoning = payload.get('reasoning')
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError('LLM judge reasoning must be a non-empty string')

    return (1.0 if verdict else 0.0), {'verdict': verdict, 'reasoning': reasoning.strip()}


def read_text_if_exists(path: Path) -> str | None:
    """Read a UTF-8 file if it exists."""
    if not path.exists():
        return None
    return path.read_text(encoding='utf-8').strip()


@dataclass(frozen=True)
class VerifierOutputs:
    """Everything the verifier writes under ``/logs/verifier``, read in one round trip."""

    reward_text: str = ''
    reward_json_text: str = ''
    ctrf_text: str = ''
    log_tail: str = ''
    log_truncated: bool = False
    result: dict[str, Any] = field(default_factory=dict)


# base64 keeps every payload on a single space-free line, so one command can carry all of
# them back unambiguously. These are small control files; the log is capped by the tail.
_COLLECT_OUTPUTS_COMMAND = f"""
cd {shlex.quote(VERIFIER_LOG_DIR)} 2>/dev/null || exit 0
for name in {' '.join(VERIFIER_OUTPUT_NAMES)}; do
  printf '%s ' "$name"
  if [ -f "$name" ]; then base64 < "$name" | tr -d '\\n'; fi
  printf '\\n'
done
printf 'test.txt.size %s\\n' "$(if [ -f test.txt ]; then wc -c < test.txt; else echo 0; fi)"
printf 'test.txt '
if [ -f test.txt ]; then tail -c {VERIFIER_LOG_MAX_CHARS} test.txt | base64 | tr -d '\\n'; fi
printf '\\n'
"""


def _decode_collected(fields: dict[str, str], name: str) -> str:
    payload = fields.get(name, '')
    if not payload:
        return ''
    try:
        return base64.b64decode(payload, validate=True).decode('utf-8', 'replace').strip()
    except (binascii.Error, ValueError):
        logger.warning('Could not decode verifier output %s', name)
        return ''


async def collect_verifier_outputs(runtime: Any, *, timeout: int = 120) -> VerifierOutputs:
    """Read the reward, CTRF, wrapper result and bounded log tail in a single command.

    Observability must not turn a valid score into an evaluation failure, so a failed or
    unparseable collection is logged and surfaces as empty fields rather than an exception.
    """
    try:
        result = await runtime.run_command(command=_COLLECT_OUTPUTS_COMMAND, timeout=timeout)
    except Exception:
        logger.warning('Could not collect verifier outputs', exc_info=True)
        return VerifierOutputs()
    if getattr(result, 'return_code', 0) != 0:
        logger.warning('Could not collect verifier outputs: %s', result)
        return VerifierOutputs()

    fields: dict[str, str] = {}
    for line in (getattr(result, 'stdout', '') or '').splitlines():
        name, _, payload = line.partition(' ')
        if name:
            fields[name] = payload.strip()

    try:
        log_size = int(fields.get('test.txt.size', '0') or '0')
    except ValueError:
        log_size = 0

    wrapper: dict[str, Any] = {}
    result_text = _decode_collected(fields, 'verifier_result.json')
    if result_text:
        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning('Verifier result %s is not valid JSON', VERIFIER_RESULT_PATH)
        else:
            if isinstance(payload, dict):
                wrapper = payload
            else:
                logger.warning('Verifier result %s must contain a JSON object', VERIFIER_RESULT_PATH)

    return VerifierOutputs(
        reward_text=_decode_collected(fields, 'reward.txt'),
        reward_json_text=_decode_collected(fields, 'reward.json'),
        ctrf_text=_decode_collected(fields, 'ctrf.json'),
        log_tail=_decode_collected(fields, 'test.txt'),
        log_truncated=log_size > VERIFIER_LOG_MAX_CHARS,
        result=wrapper,
    )


def build_verifier_test_results(
    ctrf_data: Any,
    *,
    reward: float,
    return_code: Any,
    status: Any,
    duration_seconds: float,
    log_tail: str,
    log_truncated: bool,
    verifier_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep CTRF compatibility while attaching Nexus verifier diagnostics.

    Existing consumers expect ``test_results`` to contain the task-provided
    CTRF object at its top level.  Diagnostics therefore live under the
    namespaced ``nexus_verifier`` key instead of replacing that object.
    """
    if isinstance(ctrf_data, dict):
        results: dict[str, Any] = dict(ctrf_data)
    else:
        results = {'ctrf': ctrf_data}

    normalized_return_code: int | str | None
    if return_code is None:
        normalized_return_code = None
    else:
        try:
            normalized_return_code = int(return_code)
        except (TypeError, ValueError):
            normalized_return_code = str(return_code)

    status_value = getattr(status, 'value', status)
    status_text = str(status_value).strip().lower() if status_value is not None else ''
    if not status_text:
        status_text = 'finished' if normalized_return_code == 0 else 'failed'

    results['nexus_verifier'] = {
        'reward': reward,
        'return_code': normalized_return_code,
        'status': status_text,
        'duration_seconds': round(max(0.0, float(duration_seconds)), 3),
        'log_path': VERIFIER_LOG_PATH,
        'log_tail': log_tail,
        'log_truncated': bool(log_truncated),
    }
    if verifier_result:
        results['nexus_verifier']['wrapper_result'] = verifier_result
        test_return_code = verifier_result.get('test_return_code')
        if test_return_code is not None:
            try:
                test_return_code = int(test_return_code)
            except (TypeError, ValueError):
                test_return_code = str(test_return_code)
            results['nexus_verifier']['test_return_code'] = test_return_code
            results['nexus_verifier']['test_status'] = 'passed' if test_return_code == 0 else 'failed'
    return results


def normalize_ref_answer(reference: Any) -> str | None:
    """Normalize an evaluator reference value to prompt-ready text."""
    if reference is None:
        return None
    if isinstance(reference, str):
        text = reference.strip()
    else:
        try:
            text = json.dumps(reference, ensure_ascii=False, sort_keys=True).strip()
        except TypeError:
            text = str(reference).strip()
    return text or None


def load_ref_answer(*, reference: Any, task_dir: str = '') -> str:
    """Load host-side reference text from config or ``task_dir/ref_answer.txt``."""
    configured = normalize_ref_answer(reference)
    if configured:
        return configured

    if task_dir:
        path = Path(task_dir) / 'ref_answer.txt'
        task_reference = read_text_if_exists(path)
        if task_reference:
            return task_reference

    raise ValueError('Evaluation requires task_config.ref_answer or task_config.task_dir/ref_answer.txt')


__all__ = [
    'load_ref_answer',
    'normalize_ref_answer',
    'parse_first_json_object',
    'parse_llm_judge_result',
    'build_verifier_test_results',
    'collect_verifier_outputs',
    'read_text_if_exists',
    'resolve_verifier_env',
    'VerifierOutputs',
    'VERIFIER_LOG_DIR',
    'VERIFIER_LOG_MAX_CHARS',
    'VERIFIER_LOG_PATH',
    'VERIFIER_OUTPUT_NAMES',
    'VERIFIER_RESULT_PATH',
    'verifier_registry_network_target',
]
