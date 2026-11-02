"""Verify flow evidence and aesthetics, then aggregate their scores."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.config import AESTHETIC_MODES
from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt, render_prompt


logger = logging.getLogger(__name__)

FUNC_WEIGHT = 0.5
AESTHETIC_WEIGHT = 0.5
# Functionality and aesthetics each contribute half the total score.

AESTHETIC_MAX_ATTEMPTS = 4
# Four outer attempts, including protocol validation; client transport retries are separate.

AESTHETIC_MAX = 5.0
# Normalize per-dimension aesthetic scores to [0, 1].

TIER_WEIGHT = {'core': 4.0, 'feature': 2.0, 'constraint': 1.0}
# Tier weights remain local to verification, independent of contract generation.

DEFAULT_TIER = 'feature'
# Use the middle-tier weight when tier is unspecified.

MAX_AESTHETIC_IMAGES = 3

VERDICTS = ('pass', 'fail', 'inconclusive')

INCONCLUSIVE_REASONS = ('none', 'script_error', 'no_evidence', 'evidence_unreadable', 'not_reached', 'other')

# Load templates once and render per-request fields at call time.
FUNC_PROMPT = load_prompt('webjudger_func_judge.md')
AESTHETIC_PROMPTS = {
    '': load_prompt('webjudger_aesthetic.md'),
    'web': load_prompt('webjudger_aesthetic_web.md'),
    'game': load_prompt('webjudger_aesthetic_game.md'),
    '3d': load_prompt('webjudger_aesthetic_3d.md'),
    'svg': load_prompt('webjudger_aesthetic_svg.md'),
}
"""按品类分流的美学 prompt，``''`` 是所有品类共用的那份。 选哪份由
``config.judge.aesthetic_prompt`` 决定，见 :func:`aesthetic_judge`。"""

_PROMPT_DIM_RE = re.compile(r'^\s{2}"([a-z_]+)"\s*:\s*\{\s*$', re.MULTILINE)

AESTHETIC_DIMS_BY_VARIANT = {name: frozenset(_PROMPT_DIM_RE.findall(text)) for name, text in AESTHETIC_PROMPTS.items()}
"""每份 prompt 要求的维度集，从它自己的输出模板里解析出来。

判分侧对「该有几维」保持无感知 —— 数目和名字都由 prompt 说了算，加一份 prompt、
改一份 prompt 的维度都不用动代码。 但**解出的维度必须与所选 prompt 要求的完全一致**：
少一维就说明这次 VLM 没照协议输出，那是判分侧的故障，要重试而不是降级打分
（见 :func:`aesthetic_judge`）。

模块加载时校验非空，所以改坏输出模板会在启动时炸，而不是每条 rollout 静默失败。
"""

for _variant, _dims in AESTHETIC_DIMS_BY_VARIANT.items():
    if not _dims:
        raise RuntimeError(
            f'美学 prompt {_variant or "(shared)"!r} 的输出模板里解析不出维度名；'
            f'``  "<dim>": {{`` 这一行的缩进/引号被改坏了，判分会无从校验维度完整性'
        )


# ── evidence → prompt ─────────────────────────────────────────────────────────


def render_evidence(item: dict[str, Any], *, max_images: int = 4) -> tuple[str, list[tuple[str, str]]]:
    """Render evidence as text and named screenshots for multimodal judging."""
    lines: list[str] = []
    status = str(item.get('status') or '')

    # Page unreachability is an artifact observation, not a collection-script error.
    if status == 'page_unreachable':
        lines.append(f'⚠ 页面打不开（产物或部署的问题，不是采集故障）：{item.get("error") or ""}')
        lines.append('')

    # Include execution status and traceback for failure attribution.
    if status and status not in {'completed', 'page_unreachable'}:
        lines.append(
            f'⚠【采集中途中断】脚本未跑完（{status}），以下证据是**残缺的**。'
            f' 请判断这是产物故障还是采集脚本自身的缺陷。'
        )
        if str(item.get('error') or '').strip():
            lines.append(f'中断原因：\n```\n{str(item["error"]).strip()}\n```')
        lines.append('')

    # Show the script before observations so empty output can be interpreted correctly.
    code = str(item.get('code') or '').strip()
    if code:
        lang = 'bash' if item.get('kind') == 'source_code' else 'python'
        lines.append(
            '【实际执行的脚本】下面的事实就是这段代码跑出来的。 **先读它再读事实** ——'
            '如果它压根没检查标准里要求的东西，或者选择器/命令明显写错了，'
            '那么事实为空只说明我们没测到，应判 inconclusive，不要算成产物的缺陷：'
        )
        if item.get('note'):
            lines.append(f'  （写这段代码的人自述：{item["note"]}）')
        lines.append(f'```{lang}')
        lines.append(code)
        lines.append('```')
        lines.append('')

    # Underscore-prefixed labels are evaluator diagnostics, not artifact facts.
    observations = [o for o in (item.get('observations') or []) if not str(o.get('label', '')).startswith('_')]
    if observations:
        lines.append(
            '【采集到的事实】上面脚本实测得到的原始值 —— 数量、文本、属性、等待耗时。'
            ' 这些是事实不是结论，要不要算达标由你对照标准判断：'
        )
        for row in observations:
            value = row.get('value')
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
            lines.append(f'  - {row.get("label", "?")} = {text}')
    else:
        lines.append(
            '【采集到的事实】无 —— 一条观察值都没记下。'
            + (
                ' 对照上面的脚本判断这是「产物确实没有」还是「脚本没查」。'
                if code
                else ' 而且没有采集脚本，属于我们没测到，应判 inconclusive。'
            )
        )

    for key, title, hint in (
        (
            'page_errors',
            '页面未捕获异常',
            '产物自己的 JS 崩了。 如果本条标准依赖的交互正是被它打断的，这就是 fail 的硬证据',
        ),
        ('console_errors', 'console 报错', '页面运行时报错，比未捕获异常弱一些'),
        (
            'failed_requests',
            '资源加载失败',
            '这些 URL 没加载成功（图片/脚本/字体 404）。 页面可能「打得开但内容是残的」',
        ),
    ):
        rows = item.get(key) or []
        if rows:
            lines.append('')
            lines.append(
                f'【{title}】{hint}。**只在与本条标准相关时才作为判据** —— '
                f'打点、favicon、第三方统计之类与本条无关的忽略（共 {len(rows)} 条）：'
            )
            lines += [f'  - {r}' for r in rows]

    images: list[tuple[str, str]] = []
    shots = item.get('shots') or []
    if shots:
        lines.append('')
        lines.append('【截图】按采集顺序，图片在本条消息之后单独给出，可用下面的名字引用：')
        for shot in shots[:max_images]:
            lines.append(f'  {shot.get("name")}: {shot.get("label") or "(无说明)"}')
            if shot.get('b64'):
                images.append((str(shot.get('name')), shot['b64']))
        if len(shots) > max_images:
            lines.append(f'  ...另有 {len(shots) - max_images} 张超出上限未给出')
    return '\n'.join(lines), images


def _parse_verdict(raw: str) -> dict[str, Any]:
    """Normalize model output; invalid verdicts become inconclusive."""
    from nexus.extensions.tasks.artifact.judges.capabilities.parse import coerce_score, loads_json_lenient

    data, error = loads_json_lenient(raw)
    if error or not isinstance(data, dict):
        return {
            'verdict': 'inconclusive',
            'reason': f'judge 输出无法解析: {error or "not a dict"}',
            'evidence_refs': [],
            'inconclusive_reason': 'other',
            'confidence': 0.0,
        }

    verdict = data.get('verdict') if data.get('verdict') in VERDICTS else 'inconclusive'
    reason = data.get('inconclusive_reason')
    confidence = data.get('confidence')
    # Require finite confidence values before normalization.
    confidence = coerce_score(confidence, low=0.0, high=1.0)
    if confidence is None:
        confidence = 0.0
    refs = data.get('evidence_refs')
    return {
        'verdict': verdict,
        'reason': str(data.get('reason') or '')[:1000],
        'evidence_refs': [str(r) for r in refs][:12] if isinstance(refs, list) else [],
        'inconclusive_reason': reason if reason in INCONCLUSIVE_REASONS else 'other',
        'confidence': confidence,
    }


_ERROR_KEYS = ('console_errors', 'page_errors', 'failed_requests')
_NO_EVIDENCE_STATUSES = frozenset({'missing_script', 'invoke_failed', 'no_result', 'harness_error'})


def _enforce(verdict: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic evidence constraints to the model verdict."""
    refs = {str(o.get('label')) for o in item.get('observations') or []}
    refs |= {str(s.get('name')) for s in item.get('shots') or []}
    refs |= {str(row) for key in _ERROR_KEYS for row in item.get(key) or []}

    if item.get('status') in _NO_EVIDENCE_STATUSES:
        return {
            **verdict,
            'verdict': 'inconclusive',
            'inconclusive_reason': 'script_error',
            'reason': f'采集未执行或未留下产物证据（{item.get("status")}）：{str(item.get("error") or "")[:300]}',
            'overridden': 'collection_unavailable',
        }

    if not refs and not str(item.get('code') or '').strip() and not str(item.get('error') or '').strip():
        return {
            **verdict,
            'verdict': 'inconclusive',
            'inconclusive_reason': 'script_error',
            'reason': f'采集没留下任何证据（{item.get("status", "unknown")}）：{str(item.get("error") or "")[:300]}',
            'overridden': 'no_material',
        }

    if verdict['verdict'] in {'pass', 'fail'} and item.get('status') == 'completed':
        requested_refs = [str(ref).strip() for ref in verdict['evidence_refs'] if str(ref).strip()]
        cited = [ref for ref in requested_refs if ref in refs] or [
            ref for ref in requested_refs if len(ref) >= 6 and any(ref in available for available in refs)
        ]
        if not cited:
            return {
                **verdict,
                'verdict': 'inconclusive',
                'inconclusive_reason': 'no_evidence',
                'reason': f'判定引用不出真实证据（原判 {verdict["verdict"]}：{verdict["reason"][:200]}）',
                'overridden': 'unciteable',
            }
        verdict = {**verdict, 'evidence_refs': cited}
    return {**verdict, 'overridden': ''}


# ── func judge ────────────────────────────────────────────────────────────────


async def func_judge(
    evidence: list[dict[str, Any]],
    judger: Any,
    *,
    flow: dict[str, Any],
    problem_statement: str,
    concurrency: int = 8,
    save_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Judge each flow independently using its evidence and the shared client."""
    flows = {f['fid']: f for f in (flow.get('flows') or [])}
    limit = asyncio.Semaphore(max(1, concurrency))

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        from nexus.extensions.tasks.artifact.judges.capabilities.llm import multimodal_completion

        fid = item.get('fid', '')
        spec = flows.get(fid, {})
        row: dict[str, Any] = {
            'fid': fid,
            'title': spec.get('title', ''),
            'criterion': spec.get('criterion', ''),
            'tier': spec.get('tier', DEFAULT_TIER),
            'source': spec.get('source', 'explicit'),
            'check_via': spec.get('check_via', 'browser'),
            'status': item.get('status', 'unknown'),
        }
        # Retain interrupted evidence; _enforce handles unusable records.
        text, images = render_evidence(item)
        prompt = render_prompt(
            FUNC_PROMPT,
            query=problem_statement,
            criterion=spec.get('criterion', '(契约里找不到这条 flow)'),
            check_via=row['check_via'],
            evidence=text,
        )
        content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt}]
        for name, b64 in images:
            content.append({'type': 'text', 'text': f'[截图 {name}]'})
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'},
            })
        try:
            async with limit:
                raw = await multimodal_completion(judger, content, max_attempts=2)
        except Exception as error:
            return {
                **row,
                'verdict': 'inconclusive',
                'reason': f'judge 调用失败: {error}',
                'evidence_refs': [],
                'inconclusive_reason': 'other',
                'confidence': 0.0,
                'overridden': 'judge_call_failed',
            }
        return {**row, **_enforce(_parse_verdict(raw), item), 'raw_output': raw[:2000]}

    verdicts = list(await asyncio.gather(*(one(e) for e in evidence)))
    if save_dir:
        root = Path(save_dir) / 'web_judger'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'func_verdicts.json').write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding='utf-8')
    return verdicts


# ── aesthetic judge ───────────────────────────────────────────────────────────


def choose_images(evidence: list[dict[str, Any]], maximum: int = MAX_AESTHETIC_IMAGES) -> list[dict[str, Any]]:
    """Select the first screenshot from up to maximum flows in order."""
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        if len(picked) >= maximum:
            break
        fid = item.get('fid', '')
        for shot in item.get('shots') or []:
            b64 = shot.get('b64')
            if not b64 or b64 in seen:
                continue
            seen.add(b64)
            picked.append({'fid': fid, **shot})
            break  # 每条 flow 只取首图
    return picked


def parse_aesthetic_output(raw: str) -> tuple[dict[str, float], dict[str, str], str]:
    """Extract dimension scores, reasons, and summary from VLM JSON.

    Normalize dimension names to lowercase and tolerate raw newlines in reasons.
    The caller validates the dimension set against the selected prompt.
    """
    dims: dict[str, float] = {}
    reasons: dict[str, str] = {}

    from nexus.extensions.tasks.artifact.judges.capabilities.parse import coerce_score, loads_json_lenient

    data, error = loads_json_lenient(raw, allow_raw_newlines=True)
    if error or not isinstance(data, dict):
        return dims, reasons, ''

    lowered = {str(key).strip().lower(): value for key, value in data.items()}
    for name, block in lowered.items():
        if not isinstance(block, dict):
            continue
        # Validate scores before normalization; invalid values must not become valid scores.
        score = coerce_score(block.get('score'), low=0.0, high=AESTHETIC_MAX)
        if score is not None:
            dims[name] = score
        reason = str(block.get('reason') or '').strip()
        if reason:
            reasons[name] = reason
    return dims, reasons, str(lowered.get('summary') or '').strip()


def _aesthetic_comments(summary: str, reasons: dict[str, str]) -> str:
    """Combine summary and dimension reasons for report comments."""
    parts = [f'{name}: {reason}' for name, reason in reasons.items()]
    if summary:
        parts.append(f'总结: {summary}')
    return '\n'.join(parts)


async def aesthetic_judge(
    evidence: list[dict[str, Any]],
    judger: Any,
    *,
    problem_statement: str,
    flow: dict[str, Any] | None = None,
    mode: str = '',
    instance: dict[str, Any] | None = None,
    archive_b64: str = '',
    save_dir: str | None = None,
) -> dict[str, Any]:
    """Dispatch screenshot scoring to absolute or anchor-based pairwise evaluation."""

    def _save(payload: dict[str, Any]) -> dict[str, Any]:
        if save_dir:
            root = Path(save_dir) / 'web_judger'
            root.mkdir(parents=True, exist_ok=True)
            (root / 'aesthetic.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return payload

    resolved = mode or config.get().judge.aesthetic_mode
    images = choose_images(evidence)
    result: dict[str, Any] = {
        'available': False,
        'score': 0.0,
        'raw_score': 0.0,
        'mode': resolved,
        'n_images': len(images),
        'images': [f'{i["fid"]}/{i["name"]}' for i in images],
        # Distinguish no_images from model failures for reward classification.
        'reason': None,
    }
    if resolved not in AESTHETIC_MODES:
        result['error'] = f'未知的美学模式 {resolved!r}（可选 {" / ".join(sorted(AESTHETIC_MODES))}）'
        result['reason'] = 'bad_mode'
        logger.warning('aesthetic 不可用: %s', result['error'])
        return _save(result)
    if not images:
        total = sum(len(e.get('shots') or []) for e in evidence)
        result['error'] = f'没有任何截图可用于美学判定（evidence 共 {total} 个 shot，均无可解码 b64）'
        result['reason'] = 'no_images'
        logger.warning('aesthetic 不可用（无截图）: %s', result['error'])
        return _save(result)

    if resolved == 'pairwise':
        return _save(
            await _aesthetic_pairwise(
                images,
                judger,
                result,
                problem_statement=problem_statement,
                instance=instance,
                archive_b64=archive_b64,
            )
        )
    return _save(await _aesthetic_absolute(images, judger, result, problem_statement=problem_statement, flow=flow))


async def _aesthetic_absolute(
    images: list[dict[str, Any]],
    judger: Any,
    result: dict[str, Any],
    *,
    problem_statement: str,
    flow: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score dimensions on a 0-5 scale, validating protocol output within retries."""
    from nexus.extensions.tasks.artifact.judges.capabilities.llm import retry_with_backoff, vlm_score

    routing = config.get().judge.aesthetic_prompt
    variant = str((flow or {}).get('task_type') or '') if routing == 'by_type' else routing
    if variant not in AESTHETIC_PROMPTS:
        variant = ''
    # Record the selected prompt variant for score provenance.
    result['prompt_variant'] = variant or 'shared'

    shots_desc = '\n'.join(
        f'  第 {i} 张：{img.get("label") or img.get("name") or "(无说明)"}' for i, img in enumerate(images, 1)
    )
    prompt = render_prompt(AESTHETIC_PROMPTS[variant], query=problem_statement, shots=shots_desc)
    expected = AESTHETIC_DIMS_BY_VARIANT[variant]
    parsed: dict[str, Any] = {}

    async def _score_once() -> str:
        """Call the VLM and require exactly the dimensions declared by the prompt."""
        raw = await vlm_score(judger, [i['b64'] for i in images], prompt, max_attempts=1)
        dims, reasons, summary = parse_aesthetic_output(raw)
        if set(dims) != expected:
            missing = sorted(expected - set(dims))
            extra = sorted(set(dims) - expected)
            raise ValueError(f'美学维度与 prompt 不一致（缺 {missing}，多 {extra}，要求 {sorted(expected)}）')
        parsed.update(dims=dims, reasons=reasons, summary=summary, raw=raw)
        return raw

    try:
        await retry_with_backoff(_score_once, max_attempts=AESTHETIC_MAX_ATTEMPTS, label='aesthetic_judge')
    except Exception as error:
        result['error'] = f'美学判定失败: {type(error).__name__}: {error}'
        result['reason'] = 'vlm_error'
        # Log failures even when evidence persistence is disabled.
        logger.warning(
            'aesthetic 不可用（已重试 %d 次，prompt=%s）: %s: %s',
            AESTHETIC_MAX_ATTEMPTS,
            variant or 'shared',
            type(error).__name__,
            error,
        )
        return result

    dims, reasons, summary, raw = parsed['dims'], parsed['reasons'], parsed['summary'], parsed['raw']
    mean = sum(dims.values()) / len(dims)
    result.update({
        'available': True,
        'raw_score': round(mean, 3),
        'score': round(mean / AESTHETIC_MAX, 4),
        'dims': dims,
        'dim_reasons': reasons,
        # Preserve fields consumed by reports.
        'comments': _aesthetic_comments(summary, reasons)[:1200],
        'raw_output': raw[:2000],
    })
    return result


async def _aesthetic_pairwise(
    images: list[dict[str, Any]],
    judger: Any,
    result: dict[str, Any],
    *,
    problem_statement: str,
    instance: dict[str, Any] | None,
    archive_b64: str,
) -> dict[str, Any]:
    """Compare against reference screenshots: win/tie/lose maps to 1/0.5/0."""
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import SandboxContext, ScreenshotSet
    from nexus.extensions.tasks.artifact.judges.capabilities.llm import retry_with_backoff
    from nexus.extensions.tasks.artifact.judges.capabilities.parse import loads_json_lenient
    from nexus.extensions.tasks.artifact.judges.registry import get_judge
    from nexus.extensions.tasks.artifact.judges.strategies.visual_anchored import WINNER_VERDICTS

    # Use the full sandbox screenshot list rather than one improved image.
    shot_ctx = SandboxContext(screenshots=ScreenshotSet(sandbox=[i['b64'] for i in images]))
    got: dict[str, Any] = {}

    # Reuse prepared screenshots; let the outer comparison own retries.
    sub_input = JudgeInput(
        instance_id='',
        problem_statement=problem_statement,
        openai_clients={'vlm': judger},
        archive_b64=archive_b64,
        sandbox_ctx=shot_ctx,
        instance={
            **(instance or {}),
            'vlm_max_attempts': 1,
            # Label corresponding interaction states in the prompt.
            'shot_labels': [str(i.get('label') or i.get('name') or '') for i in images],
        },
    )

    async def _compare_once() -> None:
        """Reject missing or invalid winners within the outer retry; a valid tie is accepted."""
        verdict = await get_judge('visual_anchored').fn(sub_input)
        got['verdict'] = verdict
        if verdict.failure_kind:
            # Propagate missing-anchor errors as unavailable comparisons.
            if verdict.failure_kind == 'config_error':
                got['fatal'] = True
                return
            raise RuntimeError(f'[{verdict.failure_kind}] {verdict.error}')
        payload, parse_error = loads_json_lenient(verdict.raw_output, allow_raw_newlines=True)
        winner = str((payload or {}).get('winner', '')).strip().lower()
        if parse_error or winner not in WINNER_VERDICTS:
            raise ValueError(f'pairwise 输出不合协议（winner={winner!r}）: {parse_error or "缺 winner 字段"}')

    try:
        await retry_with_backoff(_compare_once, max_attempts=AESTHETIC_MAX_ATTEMPTS, label='aesthetic_pairwise')
    except Exception as error:
        result['error'] = f'pairwise 判定失败（已重试 {AESTHETIC_MAX_ATTEMPTS} 次）: {type(error).__name__}: {error}'
        result['reason'] = 'vlm_error'
        result['raw_output'] = (getattr(got.get('verdict'), 'raw_output', '') or '')[:4000]
        logger.warning('aesthetic 不可用（pairwise）: %s', result['error'])
        return result

    verdict = got['verdict']
    meta = verdict.metadata or {}
    result['anchor_sha256'] = meta.get('anchor_sha256', '')
    if got.get('fatal'):
        result['error'] = f'pairwise 判定失败[{verdict.failure_kind}]: {verdict.error}'
        result['reason'] = 'vlm_error'
        logger.warning('aesthetic 不可用（pairwise）: %s', result['error'])
        return result

    reasons = {name: str(row.get('reason') or '') for name, row in (meta.get('per_dimension') or {}).items()}
    # Use the smaller reference/candidate group count.
    paired = int(meta.get('n_pairs') or 0)
    result.update({
        'available': True,
        'score': verdict.score,
        # Pairwise raw_score uses the normalized scale.
        'raw_score': verdict.score,
        'n_images': paired,
        'images': [f'{i["fid"]}/{i["name"]}' for i in images[:paired]],
        'verdict': meta.get('verdict', ''),
        'per_dimension': meta.get('per_dimension', {}),
        # Preserve fields consumed by reports.
        'comments': _aesthetic_comments(str(meta.get('reason') or ''), reasons)[:1200],
        'raw_output': (verdict.raw_output or '')[:2000],
    })
    return result


# ── 合成 ──────────────────────────────────────────────────────────────────────


def summary_score(
    verdicts: list[dict[str, Any]],
    aesthetic: dict[str, Any],
    *,
    func_weight: float = FUNC_WEIGHT,
    aesthetic_weight: float = AESTHETIC_WEIGHT,
) -> dict[str, Any]:
    """Aggregate tier-weighted functional verdicts and aesthetics.

    Exclude inconclusive items from the functional denominator, but include them
    in coverage. The total score is the equal-weight mean of both components.
    """
    tally = {'pass': 0, 'fail': 0, 'inconclusive': 0}
    passed_weight = 0.0
    determinate_weight = 0.0
    contract_weight = 0.0
    by_tier: dict[str, dict[str, int]] = {}
    for row in verdicts:
        verdict = row.get('verdict', 'inconclusive')
        tally[verdict] = tally.get(verdict, 0) + 1
        tier = str(row.get('tier') or DEFAULT_TIER)
        by_tier.setdefault(tier, {'pass': 0, 'fail': 0, 'inconclusive': 0})
        by_tier[tier][verdict] = by_tier[tier].get(verdict, 0) + 1
        weight = TIER_WEIGHT.get(tier, TIER_WEIGHT[DEFAULT_TIER])
        contract_weight += weight
        if verdict == 'inconclusive':
            continue
        determinate_weight += weight
        if verdict == 'pass':
            passed_weight += weight

    determinate = tally['pass'] + tally['fail']
    func_available = determinate_weight > 0
    func_score = round(passed_weight / determinate_weight, 4) if func_available else 0.0
    aesthetic_score = float(aesthetic.get('score') or 0.0) if aesthetic.get('available') else 0.0

    final = func_weight * (func_score if func_available else 0.0) + aesthetic_weight * aesthetic_score
    if func_available and aesthetic.get('available'):
        basis = 'func+aesthetic'
    elif func_available:
        basis = 'func_only'
    elif aesthetic.get('available'):
        basis = 'func_unavailable'
    else:
        basis = 'nothing_measured'

    total = len(verdicts) or 1
    return {
        'func_score': func_score,
        'aesthetic_score': round(aesthetic_score, 4),
        'final_score': round(max(0.0, min(1.0, final)), 4),
        'basis': basis,
        'func_available': func_available,
        'aesthetic_available': bool(aesthetic.get('available')),
        # Counts and coverage of conclusive functional verdicts.
        'determinate': determinate,
        # Weighted numerator and denominator for the functional score.
        'weighted': {'passed': round(passed_weight, 2), 'total': round(determinate_weight, 2)},
        'weighted_coverage': round(determinate_weight / contract_weight, 3) if contract_weight else 0.0,
        'contract_weight': round(contract_weight, 2),
        'by_tier': by_tier,
        'determinate_ratio': round(determinate / total, 3),
        'counts': tally,
        'n_flows': len(verdicts),
        'weights': {'func': func_weight, 'aesthetic': aesthetic_weight},
    }


async def verify(
    evidence: list[dict[str, Any]],
    *,
    flow: dict[str, Any],
    judger: Any,
    problem_statement: str,
    concurrency: int = 8,
    aesthetic_mode: str = '',
    instance: dict[str, Any] | None = None,
    archive_b64: str = '',
    save_dir: str | None = None,
) -> dict[str, Any]:
    """Judge functionality and aesthetics concurrently with the same client, then aggregate."""
    verdicts, aesthetic = await asyncio.gather(
        func_judge(
            evidence, judger, flow=flow, problem_statement=problem_statement, concurrency=concurrency, save_dir=save_dir
        ),
        aesthetic_judge(
            evidence,
            judger,
            problem_statement=problem_statement,
            flow=flow,
            mode=aesthetic_mode,
            instance=instance,
            archive_b64=archive_b64,
            save_dir=save_dir,
        ),
    )
    summary = summary_score(verdicts, aesthetic)
    logger.info(
        'web_judger: final=%.3f func=%.3f(%s) aes=%.3f(%s) pass/fail/inc=%d/%d/%d',
        summary['final_score'],
        summary['func_score'],
        summary['basis'],
        summary['aesthetic_score'],
        aesthetic.get('mode', ''),
        summary['counts']['pass'],
        summary['counts']['fail'],
        summary['counts']['inconclusive'],
    )
    result = {'summary': summary, 'verdicts': verdicts, 'aesthetic': aesthetic}
    if save_dir:
        root = Path(save_dir) / 'web_judger'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


__all__ = [
    'AESTHETIC_MAX',
    'AESTHETIC_WEIGHT',
    'FUNC_WEIGHT',
    'INCONCLUSIVE_REASONS',
    'MAX_AESTHETIC_IMAGES',
    'VERDICTS',
    'aesthetic_judge',
    'choose_images',
    'func_judge',
    'render_evidence',
    'summary_score',
    'verify',
]
