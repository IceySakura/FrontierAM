"""Compare candidate and reference screenshots with a vision model.

Candidate wins, ties, and losses map to 1, 0.5, and 0 respectively.
The wrapper supplies candidate screenshots; references come from instance data.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.llm import retry_with_backoff
from nexus.extensions.tasks.artifact.judges.capabilities.parse import loads_json_lenient
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge, resolve_client_role


logger = logging.getLogger(__name__)

NAME = 'visual_anchored'

# Candidate/reference/tie maps to win/lose/tie.
VERDICT_SCORES = {'win': 1.0, 'tie': 0.5, 'lose': 0.0}

# Use semantic winner labels rather than position-dependent A/B labels.
WINNER_VERDICTS = {'candidate': 'win', 'reference': 'lose', 'tie': 'tie'}


def _resolve_anchor_b64(value: str) -> str:
    """Return anchor screenshot base64 from a base64 string or a local file path."""
    if not value or not isinstance(value, str):
        return ''
    # Long strings are treated as already-base64; short ones as a file path.
    if len(value) > 500:
        return value
    if os.path.isfile(value):
        return base64.b64encode(Path(value).read_bytes()).decode()
    return ''


def _resolve_anchor_list(value: Any) -> list[str]:
    """Normalize an anchor field to a list of base64 shots.

    Accepts a single base64/path string or a list of them, so a dataset can hand over
    several reference states (matching the candidate's multiple flow shots) without the
    single-string rows breaking.
    """
    if isinstance(value, str):
        one = _resolve_anchor_b64(value)
        return [one] if one else []
    if isinstance(value, (list, tuple)):
        return [shot for shot in (_resolve_anchor_b64(item) for item in value) if shot]
    return []


# Where a per-case anchor screenshot may live on the instance.
ANCHOR_ALIASES = ('anchor_screenshot', 'anchor_screenshots')


def resolve_anchors_from_instance(instance: Any) -> list[str]:
    """Find the per-case anchor screenshot(s) on the instance / its nested metadata.

    RL datasets carry the reference shots per-case; when the judge config and kwargs
    don't supply any we look here. Returns ``[]`` when nothing found.
    """
    if not isinstance(instance, dict):
        return []
    sources = [instance, instance.get('metadata'), instance.get('shared')]
    md = instance.get('metadata')
    if isinstance(md, dict):
        sources.append(md.get('metadata'))
    for source in sources:
        if not isinstance(source, dict):
            continue
        for alias in ANCHOR_ALIASES:
            resolved = _resolve_anchor_list(source.get(alias, ''))
            if resolved:
                return resolved
    return []


def _shots_desc(paired: int, labels: list[str]) -> str:
    """Describe screenshot states for the pairwise prompt."""
    lines = []
    for i in range(paired):
        label = labels[i] if i < len(labels) else ''
        lines.append(f'  第 {i + 1} 组：{label}' if label else f'  第 {i + 1} 组')
    return '\n'.join(lines)


def parse_visual_verdict(raw: str) -> dict[str, Any]:
    """Parse the VLM pairwise output ``{"winner": "candidate|reference|tie", ...}``.

    Unknown/missing ``winner`` degrades to ``tie`` so this judge always returns a score.
    Callers that must not score a malformed reply (e.g. the web_judger aesthetic path,
    where a fabricated 0.5 would be fed to RL) validate ``winner`` themselves against
    :data:`WINNER_VERDICTS` and retry.
    """
    parsed, err = loads_json_lenient(raw)
    if err or not isinstance(parsed, dict):
        return {'verdict': 'tie', 'score': 0.5, 'reason': f'parse_error: {err}', 'error': err}
    winner = str(parsed.get('winner', 'tie')).strip().lower()
    verdict = WINNER_VERDICTS.get(winner, 'tie')
    return {
        'verdict': verdict,
        'score': VERDICT_SCORES[verdict],
        'winner_raw': winner,
        'reason': str(parsed.get('reason', '')),
        'per_dimension': parsed.get('per_dimension', {}),
    }


def _validate_input(inp: JudgeInput) -> str | None:
    """Require reference screenshots before deployment and capture."""
    row = dict(inp.instance or {})
    if not (_resolve_anchor_list(row.get('anchor_screenshot', '')) or resolve_anchors_from_instance(row)):
        return 'visual_anchored 需要参考图：instance 里没有 anchor_screenshot(s)'
    return None


@register_judge(
    NAME,
    requires=('vlm',),
    capabilities=('screenshot',),
    validate=_validate_input,
    model_files={'vlm': 'gemini-3.8-flash-aihub-2.0.yaml'},
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Compare the prepared candidate screenshots against reference screenshots."""
    candidates = list(inp.sandbox_ctx.screenshots.for_vlm()) if inp.sandbox_ctx else []
    if not candidates:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='no candidate screenshot available for comparison',
            failure_kind='no_sandbox',
        )

    row = dict(inp.instance or {})

    # Reference screenshots: explicit key → per-case instance metadata aliases.
    references = _resolve_anchor_list(row.get('anchor_screenshot', '')) or resolve_anchors_from_instance(row)
    if not references:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='no anchor_screenshot provided',
            failure_kind='config_error',
        )

    # Compare the same number of screenshots from each side.
    paired = min(len(references), len(candidates))
    references, candidates = references[:paired], candidates[:paired]

    prompt_text = render_prompt(
        str(row.get('visual_anchored_prompt') or '') or load_prompt('judge_visual_anchored.txt'),
        query=str(inp.problem_statement)[:2000],
        shots=_shots_desc(paired, row.get('shot_labels') or []),
    )

    client = resolve_client_role(inp.openai_clients, 'vlm')
    if client is None:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='no vlm client resolved',
            failure_kind='no_client',
        )

    max_attempts = int(row.get('vlm_max_attempts') or 3)
    try:
        raw = await _compare(client, prompt_text, references, candidates, max_attempts)
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error=f'Vision LLM call failed: {exc}',
            failure_kind='vlm_failed',
            metadata={'vlm_error': str(exc)},
        )

    comparison = parse_visual_verdict(raw)
    score = comparison['score']
    return JudgeResult(
        judge_name=NAME,
        score=round(float(score), 4),
        passed=comparison['verdict'] == 'win',
        evidence=f'[{comparison["verdict"]}] {comparison.get("reason", "")[:200]}',
        raw_output=raw,
        metadata={
            'verdict': comparison['verdict'],
            'reason': comparison.get('reason', ''),
            'per_dimension': comparison.get('per_dimension', {}),
            'n_pairs': paired,
            # Record reference hashes for per-instance provenance.
            'anchor_sha256': hashlib.sha256(''.join(references).encode()).hexdigest()[:12],
        },
    )


async def _compare(
    client: Any,
    prompt_text: str,
    references: list[str],
    candidates: list[str],
    max_attempts: int,
) -> str:
    """Send prompt + reference group then candidate group, return raw text.

    Hand-built message rather than :func:`call_judge_llm` because the grouping
    is load-bearing: all references, then all candidates, each with a caption —
    the judgement is "which implementation looks better", so each side must be
    seen as one coherent design rather than interleaved pairs.

    Label wording must match the prompt's ``reference`` / ``candidate`` vocabulary —
    the verdict is parsed by those words, so a mismatch silently inverts every score.
    """
    content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt_text}]
    for label, shots in (('参考', references), ('候选', candidates)):
        for i, shot in enumerate(shots, 1):
            content.append({'type': 'text', 'text': f'【{label} {i}/{len(shots)}】'})
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{shot}', 'detail': 'high'},
            })

    async def _call() -> str:
        response = await client.create_completion(messages=[{'role': 'user', 'content': content}])
        return response.choices[0].message.content or ''

    if max_attempts <= 1:
        return await _call()
    return await retry_with_backoff(_call, max_attempts=max_attempts, label=NAME)


__all__ = [
    'VERDICT_SCORES',
    'WINNER_VERDICTS',
    'compute_score',
    'parse_visual_verdict',
    'resolve_anchors_from_instance',
]
