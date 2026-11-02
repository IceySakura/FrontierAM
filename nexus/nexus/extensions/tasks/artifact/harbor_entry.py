#!/usr/bin/env python3
"""Run a registered artifact judge inside a Harbor verifier container.

``T3_JUDGE_NAME`` selects the strategy; strategies load their own parameters.
The entrypoint supplies common task facts and writes ``/logs/verifier/reward.json``.
Reward must be finite: ``-1`` denotes an evaluator failure, while ``[0, 1]``
contains valid scores, including zero. Evidence writing and upload are opt-in.

Usage: python -m nexus.extensions.tasks.artifact.harbor_entry
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from pathlib import Path
from typing import Any

# Use the shared evaluator-fault classification.
from nexus.extensions.tasks.artifact.judges.registry import EVALUATOR_FAULT_KINDS


logger = logging.getLogger(__name__)

DEFAULT_WORKDIR = '/workspace'
DEFAULT_LOG_DIR = '/logs/verifier'

# Only tests/ is uploaded; keep the prompt copy there.
TESTS_DIR = Path('/tests')

# The verifier environment is the sole dispatch source.
JUDGE_NAME_ENV = 'T3_JUDGE_NAME'

# Local evidence and upload are independently opt-in.
EVIDENCE_ENV = 'T3_JUDGE_EVIDENCE'
EVIDENCE_UPLOAD_ENV = 'T3_JUDGE_EVIDENCE_UPLOAD'
DEFAULT_EVIDENCE_DIR = '/tmp/judge_evidence'


def _flag(name: str) -> bool:
    return (os.environ.get(name) or '').strip().lower() in {'1', 'true', 'yes', 'on'}


# Normalize Harbor statuses for empty-artifact classification.
_AGENT_STATUS_TO_TERMINATION = {
    'success': 'ENV_DONE',
    'skipped': 'ENV_DONE',
    'timeout': 'TIMEOUT',
    'incomplete': 'MAX_STEPS',
}


def write_reward(log_dir: Path, payload: dict[str, Any]) -> None:
    """Write the one file Harbor reads, and echo it for the verifier log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'reward.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def resolve_judge_name_from_env() -> str:
    """Read the required T3_JUDGE_NAME from verifier configuration."""
    name = (os.environ.get(JUDGE_NAME_ENV) or '').strip()
    if not name:
        raise ValueError(
            f'no judge name: set {JUDGE_NAME_ENV} in the task\'s [verifier.env] (e.g. T3_JUDGE_NAME = "web_judger")'
        )
    return name


def load_instance(*, workdir: str, judge_name: str) -> dict[str, Any]:
    """Load common task facts; strategies load their own parameters.

    Prefer ``/tests/instruction.md`` over ``T3_PROBLEM_STATEMENT``.
    """
    instruction_path = TESTS_DIR / 'instruction.md'
    instruction_text = ''
    if instruction_path.is_file():
        instruction_text = instruction_path.read_text(encoding='utf-8').strip()

    problem_statement = instruction_text or os.environ.get('T3_PROBLEM_STATEMENT', '').strip()
    if not problem_statement:
        raise ValueError(
            f'no problem statement available: expected {instruction_path} (a copy of the task '
            'instruction, since Verifier uploads only tests/), or T3_PROBLEM_STATEMENT in '
            '[verifier.env]'
        )

    return {
        'instance_id': os.environ.get('T3_INSTANCE_ID', '').strip() or TESTS_DIR.parent.name,
        'problem_statement': problem_statement,
        'working_dir': workdir,
        'judge_name': judge_name,
    }


def load_agent_outcome() -> dict[str, Any]:
    """Normalize Harbor agent outcome for empty-artifact classification.

    Missing or unreadable result envelopes leave the outcome unknown.
    """
    path = (os.environ.get('T3_AGENT_RESULT_PATH') or '').strip()
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        logger.info('no readable agent result at %s (%s) — empty-artifact cause will be unknown', path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}

    # ``agent_termination_reason`` is the key ``artifact_empty_cause`` matches
    # on; Harbor's envelope spells the same facts as status / exit status.
    status = str(payload.get('status') or '').strip()
    return {
        'agent_termination_reason': _AGENT_STATUS_TO_TERMINATION.get(status, status.upper()),
        'agent_exit_status': str(payload.get('agent_exit_status') or ''),
        'agent_failure_kind': str(payload.get('failure_kind') or ''),
        'agent_failure_stage': str(payload.get('failure_stage') or ''),
    }


def resolve_judge_models():
    """Resolve each bound model in this process's explicitly injected environment."""
    from nexus.extensions.tasks.artifact.judges.profiles import ProfileError, select_models

    if any(name in os.environ for name in ('NEXUS_OPENAI_CLIENTS', 'T3_JUDGE_PROFILE')):
        raise ProfileError('Remove legacy clients JSON and T3_JUDGE_PROFILE; models are bound inside the Judge')
    name = resolve_judge_name_from_env()
    overrides = None
    if 'T3_JUDGE_MODELS' in os.environ:
        try:
            overrides = json.loads(os.environ['T3_JUDGE_MODELS'])
        except ValueError:
            raise ProfileError('T3_JUDGE_MODELS must be a role-to-YAML JSON map (agent/vlm)') from None
        if not isinstance(overrides, dict):
            raise ProfileError('T3_JUDGE_MODELS must be a role-to-YAML JSON map (agent/vlm)')
    return select_models(name, overrides)


async def collect_archive(runtime: Any, workdir: str) -> tuple[str, list[str]]:
    """Collect the deliverable using the shared artifact exclusion rules."""
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import collect_project_archive

    return await collect_project_archive(runtime, workdir)


def verdict_to_reward(result: Any, *, judge_name: str) -> dict[str, Any]:
    """Map evaluator failures to -1 and valid scores to [0, 1], with bounded diagnostics."""
    failure_kind = getattr(result, 'failure_kind', None)
    metadata = dict(getattr(result, 'metadata', None) or {})
    profile_info = metadata.get('judge_profile') or {}
    payload: dict[str, Any] = {
        'judge_name': getattr(result, 'judge_name', judge_name),
        'evidence': str(getattr(result, 'evidence', '') or '')[:4000],
        'judge_models': dict(profile_info.get('models') or {}),
    }
    if profile_info:
        payload['judge_profile'] = profile_info
    if metadata:
        payload['judge_metadata'] = metadata
    # Bound raw output in the reward payload.
    raw_output = str(getattr(result, 'raw_output', '') or '')
    if raw_output:
        payload['raw_output'] = raw_output[:8000]
    # Keep bounded rubric verdicts, not trajectories.
    rubric_results = getattr(result, 'rubric_results', None) or []
    if rubric_results:
        payload['rubric_results'] = [
            {
                'index': getattr(item, 'index', i),
                'passed': bool(getattr(item, 'passed', False)),
                'verdict': str(getattr(item, 'verdict', ''))[:200],
                'note': str(getattr(item, 'note', ''))[:500],
                'error': str(getattr(item, 'error', '') or '')[:300] or None,
            }
            for i, item in enumerate(rubric_results)
        ]

    # Expose child scores and bounded diagnostics.
    children = getattr(result, 'child_results', None) or []
    if children:
        payload['child_scores'] = {
            str(getattr(c, 'judge_name', f'leg{i}')): float(getattr(c, 'score', 0.0) or 0.0)
            for i, c in enumerate(children)
        }
        # Include child feedback alongside scores.
        payload['child_results'] = [
            {
                'judge_name': str(getattr(c, 'judge_name', f'leg{i}')),
                'score': float(getattr(c, 'score', 0.0) or 0.0),
                'evidence': str(getattr(c, 'evidence', '') or '')[:1000],
                'error': str(getattr(c, 'error', '') or '')[:500] or None,
                'failure_kind': str(getattr(c, 'failure_kind', '') or '') or None,
            }
            for i, c in enumerate(children)
        ]

    if failure_kind in EVALUATOR_FAULT_KINDS:
        # -1 makes harbor_bridge raise, so the rollout is dropped rather than
        # scored zero. Keep the reason: it is the only diagnostic that survives.
        payload['reward'] = -1.0
        payload['failure_kind'] = str(failure_kind)
        payload['error'] = str(getattr(result, 'error', '') or '')[:2000]
        return payload

    score = float(getattr(result, 'score', 0.0) or 0.0)
    payload['reward'] = min(max(score, 0.0), 1.0)
    if failure_kind:
        # An unrecognised kind is still reported, but does not silently mask out.
        payload['failure_kind'] = str(failure_kind)
    if metadata.get('deploy_failed'):
        # Artifact does not run — a real zero that belongs in training.
        payload['deploy_failed'] = True
    return payload


async def run(*, workdir: str, log_dir: Path) -> int:
    """Run the selected judge, persist optional evidence, and write Harbor reward.json."""
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput
    from nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway import reset_run_totals
    from nexus.extensions.tasks.artifact.judges.registry import run_judge
    from nexus.runtime.client.local import LocalRuntime

    # Reset usage for this judge run.
    reset_run_totals()

    # Require an explicit judge name before collecting the artifact.
    judge_name = resolve_judge_name_from_env()
    instance = load_instance(workdir=workdir, judge_name=judge_name)
    models = resolve_judge_models()
    openai_clients = models.build_clients(instance_id=instance['instance_id'])
    profile_info = models.provenance()
    # Agent execution selects the transport from runtime capabilities.
    runtime = LocalRuntime()
    archive_b64, file_list = await collect_archive(runtime, workdir)
    instance['file_list'] = file_list
    # Supply agent outcome for empty-artifact classification.
    instance['infer_metadata'] = load_agent_outcome()

    logger.debug(
        'judging %s via %s: %d files, instance keys=%d',
        instance['instance_id'],
        judge_name,
        len(file_list),
        # Log common input statistics only.
        len(instance),
    )

    # Dispatch one judge; aggregation belongs to the strategy.
    inp = JudgeInput(
        instance_id=instance['instance_id'],
        problem_statement=instance['problem_statement'],
        openai_clients=openai_clients,
        archive_b64=archive_b64,
        image=os.environ.get('T3_TASK_IMAGE', ''),
        runtime=runtime,
        instance=instance,
        save_dir=resolve_evidence_dir(instance['instance_id']),
        model_configs=models.bindings,
    )
    result = await run_judge(judge_name, inp)
    if profile_info:
        result = result.model_copy(update={'metadata': {**result.metadata, 'judge_profile': profile_info}})
    payload = verdict_to_reward(result, judge_name=judge_name)

    # Emit usage last so it remains visible in the bounded verifier log tail.
    from nexus.extensions.tasks.artifact.judges.capabilities import agent_direct_gateway as adg

    totals = adg.LLMHandlerBridge._run_totals
    if totals:
        print('LLM_USAGE_TOTALS', json.dumps(totals, ensure_ascii=False), flush=True)
        payload['llm_usage'] = dict(totals)

    # Ship the evidence tree before the container goes away. Best-effort: a
    # failed upload must not turn a scored rollout into reward=-1.
    if inp.save_dir:
        # Keep the evidence result schema consistent with artifact_task.
        try:
            Path(inp.save_dir).mkdir(parents=True, exist_ok=True)
            (Path(inp.save_dir) / 'judge_results.json').write_text(
                json.dumps(
                    {
                        'instance_id': instance['instance_id'],
                        'final_score': float(getattr(result, 'score', 0.0) or 0.0),
                        'judges': [result.model_dump()],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning('judge_results.json 落盘失败: %s: %s', type(exc).__name__, exc)

    upload = upload_evidence(inp.save_dir, instance['instance_id'])
    if upload:
        payload['evidence_archive'] = upload

    write_reward(log_dir, payload)
    return 0


def resolve_evidence_dir(instance_id: str) -> str | None:
    """Return the per-instance local evidence directory when T3_JUDGE_EVIDENCE is enabled."""
    if not _flag(EVIDENCE_ENV):
        return None
    root = os.environ.get('ARTIFACT_SAVE_SCREENSHOTS_DIR') or DEFAULT_EVIDENCE_DIR
    return str(Path(root) / (instance_id or 'unknown'))


def upload_evidence(save_dir: str | None, instance_id: str) -> dict[str, Any] | None:
    """Upload the evidence archive when enabled; upload failures never change the verdict."""
    if not save_dir or not _flag(EVIDENCE_UPLOAD_ENV):
        return None
    root = Path(save_dir)
    if not root.is_dir():
        return None

    import io
    import tarfile

    from nexus.extensions.tasks.artifact.agent_setup import cos_io

    try:
        buffer = io.BytesIO()
        files = 0
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(root)))
                    files += 1
        if not files:
            return None
        payload = buffer.getvalue()
        # Use the shared per-run archive key layout.
        key = cos_io.build_archive_key(
            instance_id=instance_id or 'unknown',
            archive_name='judge_evidence.tar.gz',
        )
        url = cos_io.upload_bytes(key, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning('judge evidence upload failed (verdict unaffected): %s: %s', type(exc).__name__, exc)
        return {'error': f'{type(exc).__name__}: {exc}'[:200]}

    logger.debug('judge evidence: %d files, %.1f KB → %s', files, len(payload) / 1024, url)
    return {'url': url, 'files': files, 'bytes': len(payload)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workdir', default=os.environ.get('T3_TASK_WORKDIR', DEFAULT_WORKDIR))
    parser.add_argument('--log-dir', default=os.environ.get('T3_VERIFIER_LOG_DIR', DEFAULT_LOG_DIR))
    parser.add_argument('--log-level', default=os.environ.get('T3_JUDGE_LOG_LEVEL', 'INFO'))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )
    log_dir = Path(args.log_dir)

    try:
        return asyncio.run(run(workdir=args.workdir, log_dir=log_dir))
    except Exception as exc:
        logger.exception('container-side judging failed before producing a verdict')
        # Never leave Harbor without a reward file: a missing one is reported
        # as a verifier crash, which hides the actual cause. -1 marks it
        # evaluator-side.
        write_reward(
            log_dir,
            {
                'reward': -1.0,
                'failure_kind': 'harbor_entry_error',
                'error': f'{type(exc).__name__}: {exc}'[:2000],
            },
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
