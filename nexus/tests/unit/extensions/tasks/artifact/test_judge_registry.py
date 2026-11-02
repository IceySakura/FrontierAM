"""Skeleton tests for the artifact-judge registry (v3 architecture).

Covers what the framework promises the contributor:

  * A judge registers by decoration; malformed decorations fail at import time.
  * ``get_judge`` returns a :class:`JudgeSpec` carrying the registration
    metadata; unknown names surface as :class:`ImportError`.
  * Logical roles require an exact binding; neither ``agent`` nor ``vlm``
    silently falls back to inference ``main``. Explicit custom roles remain supported.
  * ``run_judge`` — the ONE top-level entry — maps each layer's
    failure into a structured :attr:`JudgeResult.failure_kind`:
    unknown judge → ``config_error``, missing client → ``no_client``,
    :class:`JudgeExecutionError` → its own kind verbatim, any other
    ``Exception`` → ``internal_error`` (never guessed as ``agent_failed``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pydantic import ValidationError

from nexus.core.registry import RegistryError
from nexus.extensions.tasks.artifact.judges.base import (
    JudgeExecutionError,
    JudgeInput,
    JudgeResult,
)
from nexus.extensions.tasks.artifact.judges.registry import (
    EVALUATOR_FAULT_KINDS,
    JudgeSpec,
    _deploy_failure,
    get_judge,
    register_judge,
    resolve_client_role,
    run_judge,
)


# ────────────────────────── helpers ──────────────────────────


def _make_input(**overrides) -> JudgeInput:
    """Sensible :class:`JudgeInput` for wrapper tests. No sandbox, no runtime."""
    defaults: dict = {
        'instance_id': 'test-instance',
        'problem_statement': 'test problem',
        'openai_clients': {},
        'instance': {},
    }
    defaults.update(overrides)
    return JudgeInput(**defaults)


# ────────────────────────── @register_judge validation ──────────────────────────


def test_register_judge_rejects_sync_function():
    with pytest.raises(RegistryError) as exc_info:
        @register_judge('test_sync_rejected')
        def compute_score(inp):  # not async
            return None
    assert 'async' in str(exc_info.value)


def test_register_judge_rejects_wrong_arity():
    with pytest.raises(RegistryError) as exc_info:
        @register_judge('test_arity_rejected')
        async def compute_score(inp, extra):  # two params
            return None
    assert 'one positional JudgeInput parameter' in str(exc_info.value)


def test_register_judge_rejects_unknown_capability():
    """Unknown capability tokens are rejected. ``_CAPABILITY_TOKENS`` widens
    over time as judges migrate; unrecognised names still fail at import time
    so a typo (``'screenshotx'``) does not silently produce a judge whose
    resources the wrapper never prepares."""
    with pytest.raises(RegistryError) as exc_info:
        @register_judge('test_bad_cap_rejected', capabilities=('bogus',))
        async def compute_score(inp):
            return None
    assert 'unknown capabilities' in str(exc_info.value)


def test_register_judge_rejects_string_requires():
    """A bare string for ``requires`` must be rejected, not iterated.

    Regression: ``register_judge`` used to coerce with ``tuple(requires)``
    *before* the validator ran, so ``requires='judge'`` became
    ``('j','u','d','g','e')`` — five roles that then satisfied the
    ``tuple[str, ...]`` check. Every phantom role falls back to ``main``, so
    the judge registered fine and scored normally while its declared
    dependency on the judge client had silently vanished. The failure mode is
    a judge quietly evaluating with the wrong model, which no score can
    reveal.
    """
    with pytest.raises(RegistryError) as exc_info:
        @register_judge('test_string_requires', requires='judge')
        async def compute_score(inp):
            return None
    assert 'requires must be a tuple' in str(exc_info.value)


def test_register_judge_rejects_string_capabilities():
    """Same coercion trap on ``capabilities``.

    ``capabilities='deploy'`` would become seven single-character tokens; each
    is an unknown capability, so this happens to be caught by the token check
    — but with a misleading message. Reject on type first.
    """
    with pytest.raises(RegistryError) as exc_info:
        @register_judge('test_string_caps', capabilities='deploy')
        async def compute_score(inp):
            return None
    assert 'capabilities must be a tuple' in str(exc_info.value)


def test_register_judge_accepts_known_capabilities():
    """``deploy`` and ``screenshot`` are accepted after Step B widens the set.
    Registering a judge with both should not raise; the wrapper's preparation
    branches handle materialisation."""
    @register_judge('test_known_caps', capabilities=('deploy', 'screenshot'))
    async def compute_score(inp):
        return JudgeResult(judge_name='test_known_caps', score=1.0)
    spec = get_judge('test_known_caps')
    assert set(spec.capabilities) == {'deploy', 'screenshot'}


def test_register_judge_rejects_duplicate():
    @register_judge('test_duplicate_first')
    async def first(inp):
        return JudgeResult(judge_name='test_duplicate_first', score=1.0)

    with pytest.raises(RegistryError, match='test_duplicate_first'):
        @register_judge('test_duplicate_first')
        async def second(inp):
            return JudgeResult(judge_name='test_duplicate_first', score=0.0)


# ────────────────────────── get_judge ──────────────────────────


def test_get_judge_returns_spec_with_metadata():
    @register_judge('test_get_spec', requires=('judge', 'vlm'))
    async def compute(inp):
        return JudgeResult(judge_name='test_get_spec', score=1.0)

    spec = get_judge('test_get_spec')
    assert isinstance(spec, JudgeSpec)
    assert spec.name == 'test_get_spec'
    assert spec.requires == ('judge', 'vlm')
    assert spec.capabilities == ()


def test_get_judge_unknown_raises_import_error():
    """The registry never swallows a bad name into a bland "unknown judge" —
    :func:`run_judge` is the one that turns it into a structured
    ``config_error`` :class:`JudgeResult`. Callers that reach for
    :func:`get_judge` directly see the real traceback."""
    with pytest.raises(ImportError):
        get_judge('definitely_not_a_real_judge_xyz')


# ────────────────────────── resolve_client_role ──────────────────────────


@pytest.mark.parametrize('role', ['agent', 'vlm', 'custom'])
def test_resolve_client_role_uses_exact_binding(role):
    named, main = object(), object()
    assert resolve_client_role({role: named, 'main': main}, role) is named


@pytest.mark.parametrize('role', ['agent', 'vlm', 'custom', 'misspelled'])
def test_resolve_client_role_never_falls_back_to_main(role):
    assert resolve_client_role({'main': object()}, role) is None


# ────────────────────────── run_judge ──────────────────────────


async def test_wrapper_happy_path():
    @register_judge('test_happy')
    async def compute(inp):
        return JudgeResult(judge_name='test_happy', score=0.7)

    result = await run_judge(
        'test_happy',
        _make_input(openai_clients={'agent': MagicMock()}),
    )
    assert result.score == 0.7
    assert result.failure_kind is None


async def test_wrapper_unknown_judge_maps_to_config_error():
    """Broken/unknown judge → structured ``config_error``, never a raw crash."""
    result = await run_judge(
        'wrapper_unknown_judge_xyz',
        _make_input(openai_clients={'main': MagicMock()}),
    )
    assert result.failure_kind == 'config_error'
    assert result.score == 0.0


async def test_wrapper_missing_client_maps_to_no_client():
    @register_judge('test_missing_client', requires=('judge',))
    async def compute(inp):
        return JudgeResult(judge_name='test_missing_client', score=1.0)

    result = await run_judge(
        'test_missing_client',
        _make_input(openai_clients={}),  # neither ``judge`` nor ``main``
    )
    assert result.failure_kind == 'no_client'


async def test_wrapper_rejects_main_only_before_deploy(monkeypatch):
    from unittest.mock import AsyncMock

    deploy = AsyncMock(side_effect=AssertionError('must not deploy'))
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.deploy.prepare_sandbox_context', deploy,
    )
    called = []

    @register_judge('test_no_fallback_main', capabilities=('deploy',))
    async def compute(inp):
        called.append(inp)
        return JudgeResult(judge_name='test_no_fallback_main', score=0.5)

    assert get_judge('test_no_fallback_main').requires == ('agent',)
    result = await run_judge(
        'test_no_fallback_main', _make_input(openai_clients={'main': MagicMock()}, archive_b64='archive'),
    )
    assert result.failure_kind == 'no_client'
    assert result.score == 0.0
    assert not called
    deploy.assert_not_awaited()


async def test_wrapper_preserves_openai_clients_for_sub_judges():
    """Composite judges call ``get_judge('sub').fn(inp)`` directly. If the
    wrapper had rewritten ``inp.openai_clients`` down to the parent's
    ``requires``, a sub-judge needing another role would silently fail. This
    guards against that regression."""
    captured: dict = {}

    @register_judge('test_preserve_clients', requires=('judge',))
    async def compute(inp):
        captured['clients'] = dict(inp.openai_clients)
        return JudgeResult(judge_name='test_preserve_clients', score=1.0)

    await run_judge(
        'test_preserve_clients',
        _make_input(openai_clients={'judge': 'J', 'vlm': 'V', 'main': 'M'}),
    )
    assert captured['clients'] == {'judge': 'J', 'vlm': 'V', 'main': 'M'}


async def test_wrapper_judge_execution_error_carries_kind_verbatim():
    """Capability helpers raise ``JudgeExecutionError(kind=…)``; the wrapper
    must preserve ``kind`` unchanged. The alternative — remapping — would
    hide capability-level failure classifications."""
    @register_judge('test_exec_err')
    async def compute(inp):
        raise JudgeExecutionError('vlm_failed', 'simulated vlm timeout')

    result = await run_judge(
        'test_exec_err',
        _make_input(openai_clients={'agent': MagicMock()}),
    )
    assert result.failure_kind == 'vlm_failed'
    assert 'simulated vlm timeout' in (result.error or '')


async def test_wrapper_unexpected_exception_maps_to_internal_error():
    """A Python bug in the judge body must NOT be labelled ``agent_failed``.
    ``internal_error`` is a separate kind so evaluator-code bugs are
    distinguishable from real agent-run failures."""
    @register_judge('test_internal_err')
    async def compute(inp):
        raise KeyError('boom')

    result = await run_judge(
        'test_internal_err',
        _make_input(openai_clients={'agent': MagicMock()}),
    )
    assert result.failure_kind == 'internal_error'
    assert 'KeyError' in (result.error or '')


async def test_wrapper_enforces_judge_name_invariant():
    """If a judge returns the wrong ``judge_name`` (typo, or a composite
    accidentally forwarding a sub-judge's result), the wrapper rewrites it.
    Downstream reward mapping keys on this name and would otherwise attribute
    the score to the wrong judge."""
    @register_judge('test_name_invariant')
    async def compute(inp):
        return JudgeResult(judge_name='not_the_right_name', score=1.0)

    result = await run_judge(
        'test_name_invariant',
        _make_input(openai_clients={'agent': MagicMock()}),
    )
    assert result.judge_name == 'test_name_invariant'
    assert result.failure_kind is None
    assert result.score == 1.0


# ────────────────────────── EVALUATOR_FAULT_KINDS invariants ──────────────────────────


def test_evaluator_fault_kinds_excludes_no_artifact():
    """An empty deliverable is a legitimate zero, not a maskout — the policy
    should learn from it, not have the rollout dropped."""
    assert 'no_artifact' not in EVALUATOR_FAULT_KINDS


def test_evaluator_fault_kinds_includes_internal_error():
    """``internal_error`` (Python bug in the judge) is an evaluator fault —
    the rollout should be dropped rather than misinterpreted as a zero."""
    assert 'internal_error' in EVALUATOR_FAULT_KINDS


def test_evaluator_fault_kinds_is_shared_with_harbor_entry():
    """Harbor's reward mapping must read the SAME set, not a local copy. A
    drifted copy previously omitted ``internal_error``, which turned judge
    crashes into legitimate-looking zeros fed to training."""
    from nexus.extensions.tasks.artifact import harbor_entry

    assert harbor_entry.EVALUATOR_FAULT_KINDS is EVALUATOR_FAULT_KINDS


# ────────────────────────── return-value validation ──────────────────────────
#
# The type annotation on compute_score is not enforcement. Anything that slips
# through reaches the reward mapping and gets read as a real score, so the
# wrapper validates at runtime and attributes violations to the judge's
# implementation (internal_error) rather than the model.


async def test_wrapper_rejects_non_judgeresult_return():
    """A judge returning a dict (easy mistake when refactoring) must not have
    that dict flow into reward mapping."""
    @register_judge('test_returns_dict')
    async def compute_score(inp):
        return {'score': 1.0, 'judge_name': 'test_returns_dict'}

    result = await run_judge('test_returns_dict', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind == 'internal_error'
    assert 'expected JudgeResult' in (result.error or '')


async def test_wrapper_rejects_none_return():
    """A judge that forgets to return anything yields None — which must not be
    read as a zero score."""
    @register_judge('test_returns_none')
    async def compute_score(inp):
        return None

    result = await run_judge('test_returns_none', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind == 'internal_error'


async def test_wrapper_rejects_mutated_out_of_range_score():
    """Pydantic validates on construction, not on attribute assignment. A judge
    that builds a valid result then mutates ``score`` out of range would
    otherwise ship a >1 reward."""
    @register_judge('test_mutated_score')
    async def compute_score(inp):
        result = JudgeResult(judge_name='test_mutated_score', score=0.5)
        result.score = 42.0  # bypasses construction-time validation
        return result

    result = await run_judge('test_mutated_score', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind == 'internal_error'
    assert result.score == 0.0, 'must not leak the out-of-range score'


async def test_wrapper_accepts_child_results():
    """Composite judges attach sub-judge detail via ``child_results``; the
    wrapper must pass those through intact (it validates, it does not strip)."""
    @register_judge('test_child_results')
    async def compute_score(inp):
        return JudgeResult(
            judge_name='test_child_results',
            score=0.75,
            child_results=[
                JudgeResult(judge_name='leg_a', score=1.0),
                JudgeResult(judge_name='leg_b', score=0.5),
            ],
        )

    result = await run_judge('test_child_results', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind is None
    assert result.score == 0.75, 'composite score is authoritative, not re-derived from children'
    assert [c.judge_name for c in result.child_results] == ['leg_a', 'leg_b']
    assert [c.score for c in result.child_results] == [1.0, 0.5]


def test_child_results_does_not_override_top_level_score():
    """Guard the contract documented on the field: consumers read ``score``,
    they do not average ``child_results``. A composite may legitimately weight,
    gate or degrade — so the top-level value can differ from any mean."""
    result = JudgeResult(
        judge_name='composite',
        score=0.0,  # gated to zero despite a passing leg
        child_results=[
            JudgeResult(judge_name='required_leg', score=0.0, evidence='failed gate'),
            JudgeResult(judge_name='optional_leg', score=1.0),
        ],
    )
    assert result.score == 0.0
    mean = sum(c.score for c in result.child_results) / len(result.child_results)
    assert mean == 0.5, 'sanity: the naive average would have been 0.5'
#
# This split decides whether a rollout trains or gets dropped, so it gets
# explicit tests rather than relying on the smoke run to notice a regression.


def test_deploy_failure_artifact_fault_stays_in_training():
    """``artifact:`` means the model shipped code that does not build/run. That
    is a REAL zero: no ``failure_kind`` (so Harbor does not maskout) plus
    ``metadata['deploy_failed']`` so task.py can apply its penalty. Collapsing
    this into ``no_sandbox`` would withhold exactly the gradient that teaches
    the model not to ship broken code."""
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import classify_deploy_error

    result = _deploy_failure('some_judge', 'artifact: npm build failed', classify_deploy_error)
    assert result.score == 0.0
    assert result.failure_kind is None, 'artifact fault must NOT be masked out'
    assert result.metadata['deploy_failed'] is True
    assert result.passed is False


def test_deploy_failure_infra_fault_is_masked_out():
    """``infra:`` means our side broke (COS upload, render service, poll
    timeout). Emitting a zero would teach the policy that a working artifact
    scored nothing, so it must carry ``no_sandbox``."""
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import classify_deploy_error

    result = _deploy_failure('some_judge', 'infra: render POST failed (0): timeout', classify_deploy_error)
    assert result.score == 0.0
    assert result.failure_kind == 'no_sandbox'
    assert result.failure_kind in EVALUATOR_FAULT_KINDS


def test_deploy_failure_unprefixed_defaults_to_infra():
    """An unrecognised error shape defaults to maskout: withholding a gradient
    is safer than inventing a penalty the model may not deserve."""
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import classify_deploy_error

    result = _deploy_failure('some_judge', 'something unexpected', classify_deploy_error)
    assert result.failure_kind == 'no_sandbox'


def test_classify_deploy_error_matches_deploy_failure_results():
    """``pipelines._shared.deploy_failure_results`` (legacy consumers) and the
    registry wrapper must agree on attribution — they now share
    ``classify_deploy_error`` rather than each matching prefixes themselves."""
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import classify_deploy_error, deploy_failure_results

    for error in ('artifact: build failed', 'infra: timeout', 'no prefix here'):
        legacy = deploy_failure_results(('j',), error)[0]
        wrapped = _deploy_failure('j', error, classify_deploy_error)
        assert legacy.failure_kind == wrapped.failure_kind, f'divergent attribution for {error!r}'
        assert legacy.metadata.get('deploy_failed') == wrapped.metadata.get('deploy_failed')


# ────────────────────── JudgeResult 契约（严格性）──────────────────────
#
# 类型注解不是强制。任何漏过去的非法值都会直达 reward 映射，被当成真实分数，
# 所以这些边界值逐个钉死。


@pytest.mark.parametrize(
    ('label', 'kwargs'),
    [
        # bool 是 int 的子类，pydantic 会在构造时就把 True 转成 1.0 ——
        # wrapper 事后再校验已经看不到原始类型，所以必须在转换前拦。
        ('bool True', {'judge_name': 'x', 'score': True}),
        ('bool False', {'judge_name': 'x', 'score': False}),
        # NaN 让 ge/le 边界检查失效（与 NaN 的比较恒为 False），然后污染
        # 下游所有均值 —— 整个 GRPO group 的 advantage 变 NaN。
        ('NaN', {'judge_name': 'x', 'score': float('nan')}),
        ('inf', {'judge_name': 'x', 'score': float('inf')}),
        ('越界', {'judge_name': 'x', 'score': 1.5}),
        ('负分', {'judge_name': 'x', 'score': -0.1}),
        # 判分失败的 judge 不知道产物是否达标，声称 passed 会被读成「通过」。
        (
            '失败但 passed=True',
            {'judge_name': 'x', 'score': 0.0, 'passed': True, 'error': 'b', 'failure_kind': 'vlm_failed'},
        ),
        # error 必带 failure_kind：否则 task 层只能去 substring-match 错误文本。
        ('error 无 failure_kind', {'judge_name': 'x', 'score': 0.0, 'error': 'boom'}),
    ],
)
def test_judge_result_rejects_invalid_states(label: str, kwargs: dict):
    with pytest.raises((ValidationError, ValueError)):
        JudgeResult(**kwargs)


@pytest.mark.parametrize(
    ('label', 'kwargs'),
    [
        ('合法零分', {'judge_name': 'x', 'score': 0.0}),
        ('满分', {'judge_name': 'x', 'score': 1.0}),
        ('中间分', {'judge_name': 'x', 'score': 0.42}),
        ('int 0 可转 float', {'judge_name': 'x', 'score': 0}),
        (
            '失败且 passed=False',
            {'judge_name': 'x', 'score': 0.0, 'passed': False, 'error': 'b', 'failure_kind': 'vlm_failed'},
        ),
        # 失败结果不声明 passed（None）也是合法的 —— 判分没完成，不知道答案。
        ('失败且 passed=None', {'judge_name': 'x', 'score': 0.0, 'error': 'b', 'failure_kind': 'vlm_failed'}),
    ],
)
def test_judge_result_accepts_valid_states(label: str, kwargs: dict):
    result = JudgeResult(**kwargs)
    assert isinstance(result.score, float)


async def test_wrapper_maps_invalid_child_result_to_internal_error():
    """非法的 ``child_results`` 也要被拦 —— 复合 judge 的子结果同样进 reward 元数据。"""
    @register_judge('test_bad_child')
    async def compute_score(inp):
        parent = JudgeResult(judge_name='test_bad_child', score=0.5)
        # 绕过构造校验往 child_results 里塞一个非法对象
        object.__setattr__(parent, 'child_results', [{'judge_name': 'leg', 'score': 9.9}])
        return parent

    result = await run_judge('test_bad_child', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind == 'internal_error'


# ────────────────────── validate 钩子：贵操作之前 ──────────────────────


async def test_validate_hook_rejects_before_capabilities():
    """``validate`` 返回原因 → ``config_error``，且不碰 deploy / 截图。

    这是 J04 的核心：把配置检查放在 ``compute_score`` 里意味着它跑在 deploy
    之后，部署失败会先短路，数据错误就被记成产物错误 —— 一个假惩罚进训练。
    """
    prepared: list[str] = []

    def _reject(inp):
        return 'missing flow contract'

    @register_judge('test_validate_rejects', capabilities=('deploy',), validate=_reject)
    async def compute_score(inp):
        raise AssertionError('compute_score 不该被调用')

    import nexus.extensions.tasks.artifact.judges.capabilities.deploy as deploy_mod

    async def _spy(archive_b64, *, instance_id=''):
        prepared.append(instance_id)
        raise AssertionError('deploy 不该被调用')

    original = deploy_mod.prepare_sandbox_context
    deploy_mod.prepare_sandbox_context = _spy
    try:
        result = await run_judge(
            'test_validate_rejects',
            _make_input(openai_clients={'agent': MagicMock()}, archive_b64='x'),
        )
    finally:
        deploy_mod.prepare_sandbox_context = original

    assert result.failure_kind == 'config_error'
    assert 'missing flow contract' in (result.error or '')
    assert not prepared, '校验失败时不该部署'


async def test_validate_hook_passing_lets_the_judge_run():
    def _accept(inp):
        return None

    @register_judge('test_validate_accepts', validate=_accept)
    async def compute_score(inp):
        return JudgeResult(judge_name='test_validate_accepts', score=0.9)

    result = await run_judge('test_validate_accepts', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.score == 0.9


async def test_validate_hook_crash_is_internal_error():
    """校验器自己崩了是我们的 bug，不是坏数据 —— 不能报成 config_error。"""
    def _crash(inp):
        raise KeyError('bug in the validator')

    @register_judge('test_validate_crashes', validate=_crash)
    async def compute_score(inp):
        raise AssertionError('不该被调用')

    result = await run_judge('test_validate_crashes', _make_input(openai_clients={'agent': MagicMock()}))
    assert result.failure_kind == 'internal_error'


def test_register_judge_rejects_async_validate():
    """``validate`` 必须同步：它跑在昂贵操作之前，本身不该做 I/O。"""
    async def _async_validate(inp):
        return None

    with pytest.raises(RegistryError):
        @register_judge('test_async_validate', validate=_async_validate)
        async def compute_score(inp):
            return None


# ────────────────────── 注册边界 ──────────────────────


async def test_explicit_empty_requires_is_honoured():
    """``requires=()`` 表示「这个 judge 不调模型」，不能被默认值覆盖。

    ``meta.get('requires') or ('judge',)`` 会把空元组（falsy）替换成默认依赖，
    于是纯本地 judge 在没有 client 时被拒 ``no_client``。
    """
    @register_judge('test_no_client_needed', requires=())
    async def compute_score(inp):
        return JudgeResult(judge_name='test_no_client_needed', score=0.5)

    spec = get_judge('test_no_client_needed')
    assert spec.requires == ()

    result = await run_judge('test_no_client_needed', _make_input(openai_clients={}))
    assert result.failure_kind is None
    assert result.score == 0.5


def test_register_judge_rejects_keyword_only_signature():
    """wrapper 用 ``fn(inp)`` 位置传参，keyword-only 参数注册时就该拒绝。

    只查 ``len(parameters) == 1`` 会放过它，然后在 dispatch 时抛 TypeError ——
    比注册时报错晚得多。
    """
    with pytest.raises(RegistryError):
        @register_judge('test_kwonly')
        async def compute_score(*, inp):
            return None


def test_register_judge_rejects_varargs_signature():
    with pytest.raises(RegistryError):
        @register_judge('test_varargs')
        async def compute_score(*args):
            return None
