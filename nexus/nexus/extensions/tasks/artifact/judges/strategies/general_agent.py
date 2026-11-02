"""Judge deployed pages with a browser-capable Claude Code agent.

Lite mode uses a strict [0, 4] score; full mode uses four [1, 5] dimensions.
"""

from __future__ import annotations

import logging
import re

from typing import Any, Literal

from nexus.extensions.tasks.artifact.agent_setup.claude_config import JUDGE_ROOT
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.capabilities.agent import call_judge_agent
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.registry import register_judge


logger = logging.getLogger(__name__)

NAME = 'general_agent'
REWARD_FILE = f'{JUDGE_ROOT}/reward_general_agent.txt'

# XML tags and metadata aliases follow prompt dimension order.
DIMENSIONS: tuple[tuple[str, str], ...] = (
    ('task_fulfillment_fidelity', 'fulfillment'),
    ('functional_experience', 'functional'),
    ('craft_robustness', 'craft'),
    ('content_semantics', 'content'),
)
SCORE_MAX = 5.0  # 每维 1–5 分

_JUDGE_BLOCK_RE = re.compile(r'<judge>(.*?)</judge>', re.DOTALL | re.IGNORECASE)
_GATE_RE = re.compile(r'<gate>.*?<content>\s*(\w+)\s*</content>.*?</gate>', re.DOTALL | re.IGNORECASE)


def _extract_dim(judge_body: str, tag: str) -> tuple[int | None, str]:
    """Extract a dimension score and reason from its XML tag."""
    m = re.search(rf'<{tag}>(.*?)</{tag}>', judge_body, re.DOTALL | re.IGNORECASE)
    if not m:
        return None, ''
    body = m.group(1)
    cm = re.search(r'<content>\s*(-?\d+)\s*</content>', body, re.IGNORECASE)
    rm = re.search(r'<reason>\s*(.*?)\s*</reason>', body, re.DOTALL | re.IGNORECASE)
    score = int(cm.group(1)) if cm else None
    reason = rm.group(1).strip() if rm else ''
    return score, reason


def parse_general_xml(text: str) -> dict[str, Any]:
    """Parse the general verdict gate, dimensions, and critical issues."""
    block = _JUDGE_BLOCK_RE.search(text or '')
    if not block:
        return {
            'gate': '',
            'gate_reason': '',
            'dims': {},
            'critical_issues': [],
            'parse_error': 'judge_block_not_found',
        }
    body = block.group(1)

    gm = _GATE_RE.search(body)
    gate = gm.group(1).lower() if gm else ''

    dims: dict[str, dict[str, Any]] = {}
    for tag, key in DIMENSIONS:
        score, reason = _extract_dim(body, tag)
        dims[key] = {'score': score, 'reason': reason}

    issues = re.findall(r'<item>\s*(.*?)\s*</item>', body, re.DOTALL | re.IGNORECASE)
    issues = [i.strip() for i in issues if i.strip()]

    return {
        'gate': gate,
        'gate_reason': (gm and '') or '',
        'dims': dims,
        'critical_issues': issues,
        'parse_error': None,
    }


# Strict scoring requires a category-specific prompt; no generic fallback.
STRICT_SCORE_MAX = 4.0  # strict 单分 0–4,归一 = score/4
_STRICT_SCORE_RE = re.compile(r'<score>\s*(\w+)\s*</score>', re.IGNORECASE)
StrictPromptType = Literal['auto', 'web', 'svg', '3d', 'game']
STRICT_PROMPT_FILES: dict[str, str] = {
    'web': 'judge_general_agent_strict_web.txt',
    'svg': 'judge_general_agent_strict_svg.txt',
    '3d': 'judge_general_agent_strict_3d.txt',
    'game': 'judge_general_agent_strict_game.txt',
}
"""strict 模式的品类 prompt。**没有 generic 兜底**——原先的
``judge_general_agent_strict.txt`` 已删除,因为一份清单打四类产物意味着每次调用里大部分
条目都不适用:给落地页喂 3D 穿模条目,模型要么照着找不存在的东西(误报),要么学会整体忽略
清单(漏报)。 所以 strict 必须落到具体品类;推断不出类型时判 ``config_error``,而不是
load 一个不存在的文件抛 FileNotFoundError。"""


def parse_general_score_xml(text: str) -> dict[str, Any]:
    """Parse a strict score, invalid flag, reasons, issues, and bonuses."""
    block = _JUDGE_BLOCK_RE.search(text or '')
    if not block:
        return {
            'score': None,
            'invalid': False,
            'reason': '',
            'issues': [],
            'bonuses': [],
            'parse_error': 'judge_block_not_found',
        }
    body = block.group(1)

    m = _STRICT_SCORE_RE.search(body)
    raw_score = m.group(1).lower() if m else ''
    if raw_score == 'invalid':
        return {'score': None, 'invalid': True, 'reason': '', 'issues': [], 'bonuses': [], 'parse_error': None}
    if not raw_score.isdigit():
        return {
            'score': None,
            'invalid': False,
            'reason': '',
            'issues': [],
            'bonuses': [],
            'parse_error': f'bad_score:{raw_score!r}',
        }
    score = int(raw_score)

    rm = re.search(r'<reason>\s*(.*?)\s*</reason>', body, re.DOTALL | re.IGNORECASE)
    reason = rm.group(1).strip() if rm else ''

    def _items(tag: str) -> list[str]:
        seg = re.search(rf'<{tag}>(.*?)</{tag}>', body, re.DOTALL | re.IGNORECASE)
        if not seg:
            return []
        return [
            i.strip()
            for i in re.findall(r'<item>\s*(.*?)\s*</item>', seg.group(1), re.DOTALL | re.IGNORECASE)
            if i.strip()
        ]

    return {
        'score': score,
        'invalid': False,
        'reason': reason,
        'issues': _items('issues'),
        'bonuses': _items('bonuses'),
        'parse_error': None,
    }


def resolve_strict_prompt_type(
    configured: StrictPromptType,
    instance: dict[str, Any] | None,
    query: str,
) -> str | None:
    """Resolve the strict prompt type from config, dataset metadata, or prompt text."""
    if configured != 'auto':
        return configured

    raw_type = str((instance or {}).get('_task_type') or (instance or {}).get('task_type') or '').lower().strip()
    aliases = {
        'web': 'web',
        'web_app': 'web',
        'website': 'web',
        'svg': 'svg',
        '3d': '3d',
        'webgl': '3d',
        'threejs': '3d',
        'three.js': '3d',
        'game': 'game',
    }
    if raw_type in aliases:
        return aliases[raw_type]

    lowered = query.lower()
    if re.search(r'\b(phaser|game|player|score|level|enemy|win|lose|restart)\b|游戏|玩家|关卡|得分|胜负', lowered):
        return 'game'
    if re.search(r'\b(svg|vector|viewbox)\b|矢量图', lowered):
        return 'svg'
    if re.search(r'\b(three\.?js|webgl|shader|raymarch|3d|orbitcontrols|mesh)\b|三维|立体场景', lowered):
        return '3d'
    if re.search(r'\b(web|website|dashboard|landing page|form|e-commerce|html)\b|网页|网站|表单|仪表盘', lowered):
        return 'web'
    return None


@register_judge(
    NAME, requires=('agent',), capabilities=('deploy',), model_files={'agent': 'deepseek-flash-aihub-2.0.yaml'}
)
async def compute_score(inp: JudgeInput) -> JudgeResult:
    """Evaluate the deployed page using strict or four-dimensional scoring."""
    row = dict(inp.instance or {})

    if inp.sandbox_ctx is None or not inp.sandbox_ctx.sandbox_url:
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error='wrapper 没有提供部署好的 sandbox_url',
            failure_kind='no_sandbox',
        )
    sandbox_url = inp.sandbox_ctx.sandbox_url

    lite = bool(row.get('general_agent_lite', True))
    override_prompt = str(row.get('general_agent_prompt') or '')

    strict_prompt_type = None
    if override_prompt:
        prompt_template = override_prompt
    elif lite:
        strict_prompt_type = resolve_strict_prompt_type(
            row.get('strict_prompt_type') or 'auto',
            row,
            inp.problem_statement,
        )
        # Reject inputs without a resolvable strict prompt category.
        if strict_prompt_type not in STRICT_PROMPT_FILES:
            return JudgeResult(
                judge_name=NAME,
                score=0.0,
                error=(
                    f'strict prompt type unresolved (configured={row.get("strict_prompt_type") or "auto"!r}); '
                    f'set strict_prompt_type or instance._task_type to one of {sorted(STRICT_PROMPT_FILES)}'
                ),
                failure_kind='config_error',
                metadata={'strict': True, 'strict_prompt_type': strict_prompt_type},
            )
        prompt_template = load_prompt(STRICT_PROMPT_FILES[strict_prompt_type])
    else:
        prompt_template = load_prompt('judge_general_agent.txt')

    task_prompt = render_prompt(
        prompt_template,
        query=inp.problem_statement,
        url=sandbox_url,
        reward_file=REWARD_FILE,
    )

    # Use the shared runtime and structured execution/read-error handling.
    run = await call_judge_agent(
        inp,
        prompt=task_prompt,
        reward_file=REWARD_FILE,
        client_role='agent',
        with_playwright=True,
        with_skill=True,
    )
    cc = run.cc

    # ══════════════════════════════════════════════════════════════════════
    # strict 单分格式(lite=True)。四维逻辑(下方)完全不碰。
    # ══════════════════════════════════════════════════════════════════════
    if lite:
        raw = run.raw
        parsed = parse_general_score_xml(raw)
        parse_error = parsed['parse_error']

        # Environment failures are not artifact scores.
        if parsed['invalid']:
            return JudgeResult(
                judge_name=NAME,
                score=0.0,
                error='score=invalid',
                failure_kind='no_sandbox',
                raw_output=cc.output,
                metadata={
                    'strict': True,
                    'score_raw': 'invalid',
                    'trajectories': cc.trajectories,
                    'agent_execution_time': cc.execution_time,
                },
            )

        raw_score = parsed['score']

        # Treat an unparseable strict score as an evaluator failure.
        if raw_score is None:
            return JudgeResult(
                judge_name=NAME,
                score=0.0,
                error=f'parse: {parse_error}' if parse_error else 'no valid <score> parsed',
                failure_kind='parse_failed',
                raw_output=cc.output,
                metadata={
                    'strict': True,
                    'trajectories': cc.trajectories,
                    'agent_execution_time': cc.execution_time,
                    'parse_error': parse_error,
                },
            )

        score = round(raw_score / STRICT_SCORE_MAX, 4)  # 0–4 → [0,1]
        return JudgeResult(
            judge_name=NAME,
            score=score,
            evidence=f'strict score={raw_score}/4 → {score} :: {parsed["reason"]}',
            raw_output=cc.output,
            metadata={
                'strict': True,
                'prompt_type': strict_prompt_type or 'generic',
                'score_raw': raw_score,  # 原始 0–4
                'reason': parsed['reason'],
                'issues': parsed['issues'],
                'bonuses': parsed['bonuses'],
                'trajectories': cc.trajectories,
                'agent_execution_time': cc.execution_time,
                'parse_error': parse_error,
            },
        )

    # ══════════════════════════════════════════════════════════════════════
    # 四维格式(lite=False)。原逻辑,不变。
    # ══════════════════════════════════════════════════════════════════════
    # reward 文件优先，回落 agent stdout。
    raw = run.raw
    parsed = parse_general_xml(raw)
    parse_error = parsed['parse_error']

    gate = parsed['gate']
    dims = parsed['dims']

    # Keep environment failures separate from artifact scores.
    if gate == 'invalid':
        return JudgeResult(
            judge_name=NAME,
            score=0.0,
            error=f'gate=invalid: {parsed.get("gate_reason", "")}'.strip() or 'gate=invalid',
            failure_kind='no_sandbox',
            raw_output=cc.output,
            metadata={
                'gate': 'invalid',
                'trajectories': cc.trajectories,
                'agent_execution_time': cc.execution_time,
            },
        )

    # A fatal gate zeros dimensions; otherwise preserve missing dimensions as None.
    dim_scores: dict[str, Any] = {}
    present: list[int] = []
    for _tag, key in DIMENSIONS:
        s = dims.get(key, {}).get('score')
        if gate == 'fatal':
            s = 0
        dim_scores[key] = s
        if isinstance(s, int):
            present.append(s)

    # Normalize the available dimension mean; missing all scores is a parse failure.
    if gate == 'fatal':
        score = 0.0
    elif present:
        score = round((sum(present) / len(present)) / SCORE_MAX, 4)
    else:
        score = 0.0

    # Attach failure_kind only when a zero score reflects an actual evaluation error.
    error = None
    failure_kind = None
    if score == 0.0 and gate != 'fatal' and parse_error:
        # A non-fatal verdict still requires at least one valid score.
        error = f'parse: {parse_error}'
        failure_kind = 'parse_failed'

    evidence = (
        f'gate={gate or "?"} '
        + ' '.join(f'{k}={dim_scores.get(k)}' for _t, k in DIMENSIONS)
        + (f' | mean_norm={score}' if gate != 'fatal' else ' | fatal')
    )

    return JudgeResult(
        judge_name=NAME,
        score=score,
        evidence=evidence,
        raw_output=cc.output,
        error=error,
        failure_kind=failure_kind,
        metadata={
            'gate': gate,
            # Preserve raw dimension scores for downstream analysis.
            'dim_scores': dim_scores,
            'dim_reasons': {k: dims.get(k, {}).get('reason', '') for _t, k in DIMENSIONS},
            'critical_issues': parsed['critical_issues'],
            'trajectories': cc.trajectories,
            'agent_execution_time': cc.execution_time,
            'parse_error': parse_error,
        },
    )


__all__ = [
    'compute_score',
    'parse_general_score_xml',
    'parse_general_xml',
    'resolve_strict_prompt_type',
]
