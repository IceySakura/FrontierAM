"""Check baseline artifact requirements with an agent; return a binary verdict."""

from __future__ import annotations

import logging
import re

from typing import Any

from nexus.extensions.tasks.artifact.agent_setup.claude_config import JUDGE_ROOT
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.agent import call_judge_agent
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge


logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r'<baseline_check>(.*?)</baseline_check>', re.DOTALL | re.IGNORECASE)
_GATE_RE = re.compile(r'<gate>\s*(\w+)\s*</gate>', re.IGNORECASE)
_RESULT_RE = re.compile(r'<result>\s*(\w+)\s*</result>', re.IGNORECASE)
_HIT_RE = re.compile(r'<hit\b([^>]*)>\s*(.*?)\s*</hit>', re.DOTALL | re.IGNORECASE)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# Keep the reward file under JUDGE_ROOT, outside the artifact workspace.
DEFAULT_REWARD_FILE = f'{JUDGE_ROOT}/reward_baseline_check.txt'


def parse_baseline_xml(text: str) -> dict[str, Any]:
    """解析 baseline-check 的 ``<baseline_check>`` XML → gate + result + hits。

    返回 ``{gate, result, hits:[{category,severity,note}], reason, parse_error}``。
    gate ∈ {pass, fatal, invalid, ''}；result ∈ {clean, violation, ''}。
    """
    block = _BLOCK_RE.search(text or '')
    if not block:
        return {'gate': '', 'result': '', 'hits': [], 'reason': '', 'parse_error': 'baseline_block_not_found'}
    body = block.group(1)

    gm = _GATE_RE.search(body)
    gate = gm.group(1).lower() if gm else ''

    rm = _RESULT_RE.search(body)
    result = rm.group(1).lower() if rm else ''

    hits: list[dict[str, str]] = []
    for attrs_raw, note in _HIT_RE.findall(body):
        attrs = dict(_ATTR_RE.findall(attrs_raw))
        hits.append({
            'category': attrs.get('category', ''),
            'severity': attrs.get('severity', ''),
            'note': note.strip(),
        })

    reason = ''
    reason_m = re.search(r'<reason>\s*(.*?)\s*</reason>', body, re.DOTALL | re.IGNORECASE)
    if reason_m:
        reason = reason_m.group(1).strip()

    return {'gate': gate, 'result': result, 'hits': hits, 'reason': reason, 'parse_error': None}


@register_judge(
    'baseline_check',
    requires=('agent',),
    capabilities=('deploy',),
    model_files={'agent': 'deepseek-flash-aihub-2.0.yaml'},
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Run baseline checks and classify unavailable evaluations separately from failures."""
    if inp.sandbox_ctx is None or not inp.sandbox_ctx.sandbox_url:
        return JudgeResult(
            judge_name='baseline_check',
            score=0.0,
            error='baseline_check requires a deployed sandbox_url; wrapper did not fill sandbox_ctx',
            failure_kind='no_sandbox',
        )
    sandbox_url = inp.sandbox_ctx.sandbox_url

    prompt = render_prompt(
        load_prompt('judge_baseline_check.txt'),
        query=inp.problem_statement,
        url=sandbox_url,
        reward_file=DEFAULT_REWARD_FILE,
    )

    # Use the shared client, runtime, and structured failure handling.
    try:
        run = await call_judge_agent(
            inp,
            prompt=prompt,
            reward_file=DEFAULT_REWARD_FILE,
            client_role='agent',
            with_playwright=True,
            with_skill=True,
        )
    except Exception:
        # Let the wrapper classify execution exceptions.
        raise

    cc = run.cc
    # Parse the reward file returned by the shared agent capability.
    raw = run.raw
    parsed = parse_baseline_xml(raw)
    parse_error = parsed['parse_error']

    gate = parsed['gate']
    result = parsed['result']
    hits = parsed['hits']

    # An invalid environment is not an artifact score.
    if gate == 'invalid':
        return JudgeResult(
            judge_name='baseline_check',
            score=0.0,
            error=f'gate=invalid: {parsed.get("reason", "")}'.strip() or 'gate=invalid',
            failure_kind='no_sandbox',
            raw_output=cc.output,
            metadata={
                'gate': 'invalid',
                'trajectories': cc.trajectories,
                'agent_execution_time': cc.execution_time,
                'sandbox_url': sandbox_url,
            },
        )

    # A fatal gate is a violation; otherwise use the parsed result.
    if gate == 'fatal':
        result = 'violation'

    # Unparseable output must not be treated as a clean result.
    if result not in ('clean', 'violation'):
        return JudgeResult(
            judge_name='baseline_check',
            score=0.0,
            error=f'parse_failed: {parse_error}' if parse_error else 'no valid <result> parsed',
            failure_kind='parse_failed',
            raw_output=cc.output,
            metadata={
                'gate': gate,
                'trajectories': cc.trajectories,
                'agent_execution_time': cc.execution_time,
                'sandbox_url': sandbox_url,
            },
        )

    is_violation = result == 'violation'
    score = 0.0 if is_violation else 1.0

    evidence = f'gate={gate or "?"} result={result} hits={len(hits)}' + (
        f' :: {"; ".join(h["category"] for h in hits if h["category"])}' if hits else ''
    )

    return JudgeResult(
        judge_name='baseline_check',
        score=score,
        passed=not is_violation,
        evidence=evidence,
        raw_output=cc.output,
        metadata={
            'gate': gate,
            'result': result,
            'hits': hits,  # [{category, severity, note}]
            'trajectories': cc.trajectories,
            'agent_execution_time': cc.execution_time,
            'sandbox_url': sandbox_url,
            'parse_error': parse_error,
        },
    )


__all__ = [
    'compute_score',
    'parse_baseline_xml',
]
