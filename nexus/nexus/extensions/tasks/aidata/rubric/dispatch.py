"""rubric 包的 DAG 顶层分发器。

拥有：
- ``_ASSERT_EVALUATORS`` —— v4.2 primitive 注册表。
- ``_dotted_get`` / ``_result_record`` / ``_accepts_limits`` —— 共享辅助函数。
- ``_run_v42_check`` 和 ``_v42_result_record`` —— v4.2 分发。
- ``_rubric_is_v42`` / ``_manifest_entry_*`` / ``_collect_hard_gate_ids`` /
  ``compute_scores`` / ``run_rubric_async`` / ``run_rubric``。

``selectors`` 、 ``legacy_v2`` 和 ``primitives/state_equals`` 通过懒加载方式
从本模块导入 ``_dotted_get`` （``legacy_v2`` 还额外导入 ``_result_record`` /
``_accepts_limits``），以打破循环依赖。
"""

import asyncio
import contextlib
import inspect
import json
import logging

from collections.abc import Awaitable, Callable
from typing import Any

from .config import (
    _DEFAULT_LIMITS,
    JunitLookupContext,
    RubricLimits,
    VerifierResult,
    _resolve_concurrency,
)
from .legacy_v2 import _run_v2_check
from .primitives.judge_claim import _eval_judge_claim
from .primitives.script_probe import _eval_script_probe
from .primitives.state_equals import _eval_state_equals
from .primitives.test_result import _eval_test_result
from .primitives.text_contains import _eval_text_contains, _eval_text_not_contains
from .primitives.tool_called import _eval_tool_called, _eval_tool_called_sequence


_logger = logging.getLogger(__name__)


_ASSERT_EVALUATORS: dict[str, Callable[..., Awaitable[VerifierResult]]] = {
    'state_equals': _eval_state_equals,
    'text_contains': _eval_text_contains,
    'text_not_contains': _eval_text_not_contains,
    'tool_called': _eval_tool_called,
    # tool_calls 的有序子序列（每一步复用 ``tool_called`` 的
    # step-spec 面）。 使用与 ``tool_called`` 兄弟条目一致的规范名
    # —— 有意省略 alias。
    'tool_called_sequence': _eval_tool_called_sequence,
    'judge_claim': _eval_judge_claim,
    'test_result': _eval_test_result,
    'script_probe': _eval_script_probe,
}


def _dotted_get(root: Any, path: str) -> tuple[bool, Any]:
    """沿点分路径 ``a.b.0.c`` 遍历 `root`。

    看起来是整数的段用作列表索引； 其他段作为 dict 键。 返回
    ``(found, value)`` —— 当某段缺失 、 值类型不支持该访问或路径
    为空时 `found=False`。
    """
    cur: Any = root
    for seg in path.split('.'):
        if seg == '':
            return False, None
        if isinstance(cur, dict):
            if seg not in cur:
                return False, None
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return False, None
            if idx < 0 or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


# ---------------------------------------------------------------------------
# 分发器
# ---------------------------------------------------------------------------


async def run_rubric_async(
    state: dict,
    rubric: dict,
    *,
    runtime: Any = None,
    junit_ctx: JunitLookupContext | None = None,
    limits: RubricLimits | None = None,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """异步评估器。 按 rubric 版本分发：

    - v4.2 （``rubric['version']`` 以 "4." 开头，或任一 check 带有
      ``assert`` 键）：评估器池为 :data:`_ASSERT_EVALUATORS`； 每行
      携带 ``dimension`` + ``weight`` + ``assert_type``。
    - v2/legacy （未发现 ``assert`` 时的默认路径）：按 check 通过
      历史 ``verify:`` 注册表分发； schema 保持不变。

    `junit_ctx` 仅由 v4.2 的 ``test_result`` 评估器使用； 对于 v2
    rubric 或不使用 ``test_result`` 的 rubric，请传入 None。

    `limits` 控制轨迹展平 + judge SOURCE 裁剪。 为 None 时，从
    ``rubric['config']`` 解析 limits （见 :class:`RubricLimits`）；
    显式传入的 ``limits`` 会覆盖 rubric 自身的 config，使外层 harness
    可以强制施加 run 级 override 而不修改 rubric dict。

    `concurrency` 限制在 ``asyncio.Semaphore`` 下并行评估的 check 数。
    默认为 ``NEXUS_RUBRIC_CONCURRENCY`` （环境变量）或 8 —— 足以吸收
    含双位数 ``judge_claim`` check 的 rubric 中 LLM-judge 的往返延迟
    （之前的串行循环会一个接一个累积重试感知的 ``_llm_judge`` 调用，
    在 judge 密集的 rubric 上主导墙钟时间）。 传入 ``concurrency=1``
    可回退到严格串行评估，与并发化前的行为一致。 无论完成顺序如何，
    输出列表始终按 rubric 声明顺序返回。

    在开始时打印一条每 rubric 的清单行，随后每个 check 完成时打印
    ``[rubric] check i/N id=... verdict=... pass=...``，以便日志尾随器
    （例如 viz Playground）可实时呈现每个 check 的进度，无需等待整个
    rubric 完成。 在 concurrency >1 时，每个 check 的日志按完成顺序而非
    rubric 声明顺序到达； ``i/N`` 索引仍能标识对应的 check，因此 viz
    正则表达式保持稳定。
    """
    effective_limits = limits or RubricLimits.from_rubric(rubric)
    if effective_limits != _DEFAULT_LIMITS:
        _logger.debug(
            '[rubric] limits: user_head=%d tool_call_arg=%d tool_result=%d judge_source_max=%d',
            effective_limits.user_head_chars,
            effective_limits.tool_call_arg_chars,
            effective_limits.tool_result_chars,
            effective_limits.judge_source_max_chars,
        )
    # 为本次评分运行实例化 Fetcher。 即便在 legacy 模式下也实例化 ——
    # primitive 可通过约定好的 key 从 state 中取用，无需修改其签名。
    from .fetch import Fetcher

    _fetcher = Fetcher(state, runtime=runtime, limits=effective_limits)
    state.setdefault('_fetcher', _fetcher)  # 供 primitive 使用
    checks = rubric.get('checks', []) or []
    total = len(checks)
    is_v42 = _rubric_is_v42(rubric, checks)
    if not is_v42:
        import warnings

        warnings.warn(
            f'v2 rubric (verify: blocks, version={rubric.get("version")!r}) is deprecated. '
            f'~59 production rubrics still use it. New cases must use v4.2 (assert: DSL). '
            f'Removal target: Q4 2026. See RUBRIC.md §7 for migration.',
            DeprecationWarning,
            stacklevel=2,
        )
    cap = _resolve_concurrency(concurrency)
    # 分配的许可数不超过工作数 —— 在小 rubric 上让 Semaphore 占用
    # 保持诚实，也让 2-check rubric 上的 ``concurrency=8`` 之类日志
    # 一目了然。
    cap = min(cap, total) if total > 0 else cap
    try:
        manifest = [_manifest_entry_v42(c) if is_v42 else _manifest_entry_legacy(c) for c in checks]
        _logger.debug(
            '[rubric] manifest (v=%s, concurrency=%d): %s',
            '4.2' if is_v42 else '2.x',
            cap,
            json.dumps(manifest, ensure_ascii=False),
        )
    except Exception:
        pass

    if total == 0:
        return []

    sem = asyncio.Semaphore(cap)

    async def _eval_one(idx: int, check: dict) -> tuple[int, dict[str, Any]]:
        cid = check.get('id', '') or f'check_{idx}'
        async with sem:
            if is_v42:
                row = await _run_v42_check(
                    check,
                    state,
                    runtime=runtime,
                    junit_ctx=junit_ctx,
                    limits=effective_limits,
                )
                _logger.info(
                    '[rubric] check %d/%d id=%s assert=%s verdict=%s pass=%s',
                    idx,
                    total,
                    cid,
                    row['assert_type'],
                    row['verdict'],
                    'Y' if row['passed'] else 'N',
                )
            else:
                row = await _run_v2_check(
                    check,
                    state,
                    runtime=runtime,
                    limits=effective_limits,
                )
                _logger.info(
                    '[rubric] check %d/%d id=%s verdict=%s pass=%s',
                    idx,
                    total,
                    cid,
                    row['verdict'],
                    'Y' if row['passed'] else 'N',
                )
        return idx, row

    pairs = await asyncio.gather(*(_eval_one(i, c) for i, c in enumerate(checks, 1)))
    pairs.sort(key=lambda p: p[0])
    return [row for _, row in pairs]


def _accepts_limits(fn: Callable[..., Any]) -> bool:
    """当且仅当 ``fn`` 声明了 ``limits`` 关键字参数时返回 True。

    缓存的检查可以让分发热路径避免对同一个 verifier 反复调用
    ``inspect.signature``。
    """
    cached = getattr(fn, '__rubric_accepts_limits__', None)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(fn)
        accepts = 'limits' in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts = False
    with contextlib.suppress(AttributeError, TypeError):
        fn.__rubric_accepts_limits__ = accepts  # type: ignore[attr-defined]
    return accepts


def run_rubric(state: dict, rubric: dict) -> list[dict[str, Any]]:
    """同步 shim。 对不使用 ``shell`` verifier （或任何未来仅异步的
    verifier）的 rubric 是安全的。 对包含 shell check 的 rubric，
    调用方应改用 :func:`run_rubric_async` 并传入 runtime —— 同步
    shim 会给出 NO_RUNTIME 失败。
    """
    return asyncio.run(run_rubric_async(state, rubric, runtime=None))


def _result_record(
    check: dict,
    kind: str,
    passed: bool,
    verdict: str,
    detail: str,
    raw_head: str,
) -> dict[str, Any]:
    return {
        'id': check.get('id', ''),
        'criterion': check.get('description', ''),
        'passed': passed,
        'verify': kind,
        'source': check.get('source'),
        'claim': check.get('claim'),
        'verdict': verdict,
        'judge_raw_head': raw_head,
        'detail': detail,
    }


# ---------------------------------------------------------------------------
# v4.2 辅助函数
# ---------------------------------------------------------------------------


def _rubric_is_v42(rubric: dict, checks: list) -> bool:
    """决定使用哪种评分模型。 必须保持廉价： 每次 rubric 运行仅调用一次。
    版本字符串优先； 否则通过嗅探是否有 check 携带 ``assert:`` —— 这是
    v4.2 独有的字段。
    """
    v = str(rubric.get('version') or '')
    if v.startswith('4.'):
        return True
    if v.startswith(('2.', '3.')):
        return False
    return any(isinstance(c, dict) and 'assert' in c for c in checks)


def _manifest_entry_v42(check: dict) -> dict[str, Any]:
    a = check.get('assert') or {}
    return {
        'id': check.get('id', ''),
        'description': check.get('description', ''),
        'dimension': check.get('dimension'),
        'weight': check.get('weight'),
        'assert_type': a.get('type'),
    }


def _manifest_entry_legacy(check: dict) -> dict[str, Any]:
    return {
        'id': check.get('id', ''),
        'description': check.get('description', ''),
        'verify': check.get('verify') or 'llm_judge',
        'claim': (check.get('claim', '') or '')[:280],
        'source': check.get('source'),
    }


async def _run_v42_check(
    check: dict,
    state: dict,
    *,
    runtime: Any,
    junit_ctx: JunitLookupContext | None,
    limits: RubricLimits | None = None,
) -> dict[str, Any]:
    cid = check.get('id', '')
    cfg = check.get('assert')
    if not isinstance(cfg, dict) or not cfg.get('type'):
        return _v42_result_record(
            check,
            assert_type='<missing>',
            passed=False,
            verdict='INVALID',
            detail="missing or malformed 'assert' block",
            raw_head='',
        )
    assert_type = cfg['type']
    evaluator = _ASSERT_EVALUATORS.get(assert_type)
    if evaluator is None:
        return _v42_result_record(
            check,
            assert_type=assert_type,
            passed=False,
            verdict='UNKNOWN_ASSERT_TYPE',
            detail=f'unknown assert.type: {assert_type!r}',
            raw_head='',
        )
    # 传入 check.id 让关心它的评估器（test_result）能默认使用它。
    cfg_with_id = dict(cfg)
    cfg_with_id.setdefault('_check_id', cid)
    try:
        ret = evaluator(cfg_with_id, state, runtime, junit_ctx, limits=limits)
        if inspect.isawaitable(ret):
            ret = await ret
        passed, verdict, detail, raw_head = ret
    except Exception as exc:
        _logger.exception('[rubric] v4.2 evaluator %r raised on check id=%s', assert_type, cid)
        passed, verdict, detail, raw_head = (
            False,
            'ERROR',
            f'evaluator_exception: {exc}',
            '',
        )
    return _v42_result_record(
        check,
        assert_type=assert_type,
        passed=passed,
        verdict=verdict,
        detail=detail,
        raw_head=raw_head,
    )


def _v42_result_record(
    check: dict,
    *,
    assert_type: str,
    passed: bool,
    verdict: str,
    detail: str,
    raw_head: str,
) -> dict[str, Any]:
    a = check.get('assert') or {}
    return {
        'id': check.get('id', ''),
        'criterion': check.get('description', ''),
        'passed': passed,
        'assert_type': assert_type,
        'dimension': check.get('dimension'),
        'weight': check.get('weight'),
        'verdict': verdict,
        'judge_raw_head': raw_head,
        'detail': detail,
        # 也把 assert.type 回显为 `verify`，这样为 v2 编写的 viz /
        # result.json 读取器仍然能得到它认识的键。
        'verify': assert_type,
        'claim': a.get('claim'),
        # 也回显机器可读的 hard-gate 标志，让下游评分 /
        # result.json 读取器能区分 gate 失败与普通未通过。
        'hard_gate': bool(check.get('hard_gate')),
    }


def _collect_hard_gate_ids(rubric: dict, checks: list) -> set[str]:
    """解析出充当 **hard gate** 的 check id 集合。

    某个 check 是 hard gate，当满足以下 **任一** 条件：

    * 该 check 上带有真值的 ``hard_gate`` 字段，或
    * 其 ``id`` 出现在 rubric 顶层的 ``hard_gates: [...]`` 数组中。

    历史上使用的 ``# [HARD GATE]`` **注释** 约定 **不** 被支持：
    YAML 注释在加载时就被丢弃，所以注释永远无法被强制执行。 作者必须
    使用机器可读的字段/数组。
    """
    ids: set[str] = set()
    top = rubric.get('hard_gates')
    if isinstance(top, (list, tuple)):
        ids.update(str(x) for x in top if x)
    for c in checks:
        if isinstance(c, dict) and c.get('hard_gate') and c.get('id'):
            ids.add(str(c['id']))
    return ids


def compute_scores(
    rubric: dict,
    rubric_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """产出 task 级评分摘要。

    对 v4.2 rubric： 加权总分 + 按维度分项，使用每个 check 上的
    ``dimension`` + ``weight``， **然后进行 hard-gate 钳制** 。
    对 legacy rubric： 使用历史上的 passed / total 比率，且
    ``dimension_scores`` 为空（hard gate 是 v4.2 独有特性）。

    Hard-gate 语义（v4.2）
    --------------------------
    任一被声明为 hard gate 的 check 未通过，会封顶整个 task 分数
    （即 reward 规范中的 ``gate_safety`` ∈ {0, 1} 乘子）。 封顶值
    默认为 ``0.0`` （"清零"）； rubric 可以通过 ``config.hard_gate_cap``
    将其放宽为"略低于成功阈值的封顶"。 钳制只会降低分数，永远不会
    抬高分数。

    形状：

        {
          "score":               float,   # 钳制后，权威值
          "raw_score":           float,   # 钳制前的加权总分
          "dimension_scores":    {dim: float, ...},  # 钳制前，供诊断
          "passed":              int,
          "total":               int,
          "hard_gate_failed":    [check_id, ...],
          "hard_gate_triggered": bool,
        }
    """
    checks = rubric.get('checks', []) or []
    if not rubric_results:
        return {
            'score': 0.0,
            'raw_score': 0.0,
            'dimension_scores': {},
            'passed': 0,
            'total': 0,
            'hard_gate_failed': [],
            'hard_gate_triggered': False,
        }

    if not _rubric_is_v42(rubric, checks):
        passed = sum(1 for r in rubric_results if r.get('passed'))
        total = len(rubric_results)
        score = passed / total if total else 0.0
        return {
            'score': score,
            'raw_score': score,
            'dimension_scores': {},
            'passed': passed,
            'total': total,
            'hard_gate_failed': [],
            'hard_gate_triggered': False,
        }

    gate_ids = _collect_hard_gate_ids(rubric, checks)
    cfg = rubric.get('config') or {}
    try:
        gate_cap = float(cfg.get('hard_gate_cap') or 0.0)
    except (TypeError, ValueError):
        gate_cap = 0.0

    weight_sum = 0.0
    weighted_hits = 0.0
    per_dim_weight: dict[str, float] = {}
    per_dim_hits: dict[str, float] = {}
    passed = 0
    failed_gates: list[str] = []
    for row in rubric_results:
        # 缺失 weight 默认取 1.0，让格式错误的 rubric 仍能产出一个
        # 有意义的数值，而不是抛 ZeroDivisionError。
        w = float(row.get('weight') or 1.0)
        dim = row.get('dimension') or 'unscored'
        ok = 1.0 if row.get('passed') else 0.0
        weight_sum += w
        weighted_hits += w * ok
        per_dim_weight[dim] = per_dim_weight.get(dim, 0.0) + w
        per_dim_hits[dim] = per_dim_hits.get(dim, 0.0) + w * ok
        if row.get('passed'):
            passed += 1
        else:
            rid = row.get('id')
            if rid and (bool(row.get('hard_gate')) or rid in gate_ids):
                failed_gates.append(str(rid))
    total = len(rubric_results)
    raw_score = weighted_hits / weight_sum if weight_sum > 0 else 0.0
    # 任何失败的 hard gate 都会封顶整个 task； 钳制只会降低分数。
    score = min(raw_score, gate_cap) if failed_gates else raw_score
    dim_scores = {dim: (per_dim_hits[dim] / w if w > 0 else 0.0) for dim, w in per_dim_weight.items()}
    return {
        'score': score,
        'raw_score': raw_score,
        'dimension_scores': dim_scores,
        'passed': passed,
        'total': total,
        'hard_gate_failed': failed_gates,
        'hard_gate_triggered': bool(failed_gates),
    }
