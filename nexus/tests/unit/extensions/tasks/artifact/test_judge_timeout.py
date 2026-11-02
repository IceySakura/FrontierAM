"""判分整体超时：主动放弃优于被动被杀。

单步超时（flow2code 1200s、VLM 调用、playwright 等待）各自都有，但**没人看总和**。
一个复合 judge 串 flow2code + 逐 flow 采证 + VLM 判定 + 美学，每一步都在自己的限内，
加起来照样能超过容器的墙钟。

而 Harbor 到点是**直接杀容器**：信号打下来什么都执行不了 ——

* 写不出 ``/logs/verifier/reward.json``
* ``harbor_bridge`` 只能报「verifier 崩了」
* ``harbor_entry.main()`` 的兜底（写 ``reward=-1``）也执行不到

于是真因「判分超预算」在日志里无处可寻，表现成一次程序崩溃。

判分侧自己先放弃，就能写出结构化判决、说清原因、保留已有诊断。这也是为什么
``eval_timeout`` 要和 ``internal_error`` 分开：前者要调预算或让判分变便宜，后者要
修代码。混成一类，RL 侧就看不出到底在发生哪一种。
"""

from __future__ import annotations

import asyncio
import os

import pytest

from nexus.extensions.tasks.artifact import task as artifact_task
from nexus.extensions.tasks.artifact.harbor_entry import verdict_to_reward
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.registry import (
    EVALUATOR_FAULT_KINDS,
    JUDGE_BUDGET_ENV,
    JUDGE_BUDGET_HEADROOM_RATIO,
    JUDGE_BUDGET_MIN_HEADROOM,
    VERIFIER_TIMEOUT_ENV,
    register_judge,
    resolve_judge_budget,
    run_judge,
)


def _inp() -> JudgeInput:
    return JudgeInput(instance_id='i', problem_statement='p', openai_clients={})


@pytest.fixture(autouse=True)
def clean_budget_env(monkeypatch: pytest.MonkeyPatch):
    """每个用例自己决定预算 —— 别让进程里残留的 env 影响判定。"""
    monkeypatch.delenv(JUDGE_BUDGET_ENV, raising=False)
    monkeypatch.delenv(VERIFIER_TIMEOUT_ENV, raising=False)


# ────────────────────── 预算解析 ──────────────────────


def test_no_budget_configured_means_unbounded():
    """都没配 → 不限时。

    宿主离线路径本来有外层 batch 超时兜着，强加一个默认值会让长判分莫名被砍。
    """
    assert resolve_judge_budget() is None


def test_explicit_budget_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '300')
    monkeypatch.setenv(VERIFIER_TIMEOUT_ENV, '3600')

    assert resolve_judge_budget() == 300.0


@pytest.mark.parametrize('value', ['0', '-1', '0.0'])
def test_non_positive_explicit_budget_means_unbounded(value: str, monkeypatch: pytest.MonkeyPatch):
    """显式 0 / 负数 = 关掉这个机制，是个有用的逃生口。"""
    monkeypatch.setenv(JUDGE_BUDGET_ENV, value)

    assert resolve_judge_budget() is None


def test_verifier_timeout_leaves_headroom(monkeypatch: pytest.MonkeyPatch):
    """从 Harbor 预算推导时必须留余量 —— 判完还要取证、写 reward.json、冲日志。

    不留余量就等于「刚好在被杀的那一刻才想起要写判决」。
    """
    monkeypatch.setenv(VERIFIER_TIMEOUT_ENV, '3600')

    budget = resolve_judge_budget()

    assert budget is not None
    assert budget < 3600.0, '必须留余量'
    assert budget == 3600.0 - 3600.0 * JUDGE_BUDGET_HEADROOM_RATIO


def test_small_verifier_timeout_uses_the_minimum_headroom(monkeypatch: pytest.MonkeyPatch):
    """小预算按比例算出的余量太少，要有下限。

    600s 的 5% 只有 30s，收尾（截图落盘 + 写文件 + 日志）不一定够。
    """
    monkeypatch.setenv(VERIFIER_TIMEOUT_ENV, '600')

    assert resolve_judge_budget() == 600.0 - JUDGE_BUDGET_MIN_HEADROOM


def test_verifier_timeout_too_small_to_fit_headroom_is_unbounded(monkeypatch: pytest.MonkeyPatch):
    """预算连余量都放不下 → 不限时而不是给个负数/零预算。

    零预算会让每一次判分都立刻「超时」，把一个配置错误变成全量 maskout。
    """
    monkeypatch.setenv(VERIFIER_TIMEOUT_ENV, '30')

    assert resolve_judge_budget() is None


@pytest.mark.parametrize(('env', 'value'), [(JUDGE_BUDGET_ENV, 'abc'), (VERIFIER_TIMEOUT_ENV, 'soon')])
def test_unparseable_budget_is_ignored_not_fatal(env: str, value: str, monkeypatch: pytest.MonkeyPatch):
    """配错格式 → 警告 + 不限时，不要让判分直接起不来。"""
    monkeypatch.setenv(env, value)

    assert resolve_judge_budget() is None


# ────────────────────── 超时行为 ──────────────────────


@pytest.mark.asyncio
async def test_slow_judge_is_abandoned_with_a_structured_verdict(monkeypatch: pytest.MonkeyPatch):
    """超预算 → ``eval_timeout``，而不是让容器被杀。"""
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '0.2')

    @register_judge('_timeout_slow', requires=())
    async def compute_score(inp):
        await asyncio.sleep(30)
        return JudgeResult(judge_name='_timeout_slow', score=1.0)

    result = await run_judge('_timeout_slow', _inp())

    assert result.failure_kind == 'eval_timeout'
    assert result.score == 0.0
    # 报错要能指导下一步动作：调哪个旋钮。
    assert JUDGE_BUDGET_ENV in (result.error or '')
    assert '0.2s' in (result.error or ''), f'预算要可读地出现在报错里: {result.error}'


@pytest.mark.asyncio
async def test_timeout_is_masked_out(monkeypatch: pytest.MonkeyPatch):
    """判分超时是评测侧故障 —— rollout 丢弃，不能当成模型拿了 0 分。"""
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '0.2')

    @register_judge('_timeout_masked', requires=())
    async def compute_score(inp):
        await asyncio.sleep(30)
        return JudgeResult(judge_name='_timeout_masked', score=1.0)

    result = await run_judge('_timeout_masked', _inp())

    assert 'eval_timeout' in EVALUATOR_FAULT_KINDS
    assert verdict_to_reward(result, judge_name='_timeout_masked')['reward'] == -1.0


@pytest.mark.asyncio
async def test_timeout_maps_to_an_observable_zero_reason(monkeypatch: pytest.MonkeyPatch):
    """zero_reason 里要带 TIMEOUT，才能接上已有的 ``judge/eval_timeout`` 指标。

    落到通用的 ``EVAL_ERROR:eval_timeout`` 的话那个指标永远是 0 —— 而「判分超时率」
    恰恰是决定要不要调预算的那个数。
    """
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '0.2')

    @register_judge('_timeout_reason', requires=())
    async def compute_score(inp):
        await asyncio.sleep(30)
        return JudgeResult(judge_name='_timeout_reason', score=1.0)

    result = await run_judge('_timeout_reason', _inp())
    _score, _resolved, zero_reason = artifact_task.resolve_eval_outcome(result)

    assert zero_reason == 'EVAL_TIMEOUT'
    assert 'TIMEOUT' in zero_reason, 'reward_protocol 的 judge/eval_timeout 按这个匹配'

    from nexus.extensions.tasks.artifact.reward_protocol import build_judge_metrics

    # 端到端：term_reason 里带 TIMEOUT → 指标位点亮
    from nexus.extensions.tasks.artifact.reward_protocol import Judge

    metrics = build_judge_metrics(Judge.EVAL_CONFIG_FAILED, term_reason=zero_reason)
    assert metrics['judge/eval_timeout'] == 1.0


@pytest.mark.asyncio
async def test_timeout_is_distinct_from_internal_error(monkeypatch: pytest.MonkeyPatch):
    """两者都 maskout，但必须能区分：一个调预算，一个修 bug。"""
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '0.2')

    @register_judge('_timeout_vs_bug_slow', requires=())
    async def slow(inp):
        await asyncio.sleep(30)
        return JudgeResult(judge_name='_timeout_vs_bug_slow', score=1.0)

    @register_judge('_timeout_vs_bug_broken', requires=())
    async def broken(inp):
        raise RuntimeError('a real bug')

    slow_result = await run_judge('_timeout_vs_bug_slow', _inp())
    broken_result = await run_judge('_timeout_vs_bug_broken', _inp())

    assert slow_result.failure_kind == 'eval_timeout'
    assert broken_result.failure_kind == 'internal_error'
    # 两者都屏蔽，所以 reward 语义不变 —— 区分只为可观测。
    assert verdict_to_reward(slow_result, judge_name='x')['reward'] == -1.0
    assert verdict_to_reward(broken_result, judge_name='x')['reward'] == -1.0


@pytest.mark.asyncio
async def test_judge_within_budget_is_untouched(monkeypatch: pytest.MonkeyPatch):
    """预算内的判分一切照常 —— 这个机制不能给正常路径改语义。"""
    monkeypatch.setenv(JUDGE_BUDGET_ENV, '30')

    @register_judge('_timeout_fast', requires=())
    async def compute_score(inp):
        return JudgeResult(judge_name='_timeout_fast', score=0.8, passed=True)

    result = await run_judge('_timeout_fast', _inp())

    assert result.failure_kind is None
    assert result.score == 0.8
    assert result.passed is True


@pytest.mark.asyncio
async def test_unbounded_budget_does_not_wrap_the_call(monkeypatch: pytest.MonkeyPatch):
    """不限时时不要套 ``wait_for`` —— 少一层无谓的 task 包装。"""
    monkeypatch.delenv(JUDGE_BUDGET_ENV, raising=False)

    @register_judge('_timeout_unbounded', requires=())
    async def compute_score(inp):
        await asyncio.sleep(0.05)
        return JudgeResult(judge_name='_timeout_unbounded', score=0.5)

    result = await run_judge('_timeout_unbounded', _inp())

    assert result.score == 0.5
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_structural_failures_still_map_correctly(monkeypatch: pytest.MonkeyPatch):
    """超时分支不能吃掉 ``JudgeExecutionError`` 的结构化 kind。"""
    from nexus.extensions.tasks.artifact.judges.base import JudgeExecutionError

    monkeypatch.setenv(JUDGE_BUDGET_ENV, '30')

    @register_judge('_timeout_structural', requires=())
    async def compute_score(inp):
        raise JudgeExecutionError('vlm_failed', 'vlm died')

    result = await run_judge('_timeout_structural', _inp())

    assert result.failure_kind == 'vlm_failed'


# ────────────────────── 两条路径共用 ──────────────────────


def test_budget_lives_in_the_wrapper_not_the_entrypoints():
    """超时加在 ``run_judge`` 里，所以宿主和容器自动都有。

    在两个入口各写一个 ``wait_for`` 会漂移 —— 这正是之前几类 bug 的成因。
    """
    import inspect

    from nexus.extensions.tasks.artifact import harbor_entry
    from nexus.extensions.tasks.artifact.judges import registry

    assert 'wait_for' in inspect.getsource(registry.run_judge)
    assert 'wait_for' not in inspect.getsource(harbor_entry)
    assert 'wait_for' not in inspect.getsource(artifact_task)


def test_container_reads_the_budget_from_verifier_env():
    """容器侧的预算来自 ``[verifier.env] T3_VERIFIER_TIMEOUT_SEC``。

    harbor_bridge 不会自动注入它（那是别人的代码，不改），所以数据侧显式写 ——
    样例 task.toml 里有。
    """
    assert VERIFIER_TIMEOUT_ENV == 'T3_VERIFIER_TIMEOUT_SEC'


def test_headroom_constants_are_sane():
    """余量要够写完 reward.json，又不能吃掉太多判分时间。"""
    assert 0 < JUDGE_BUDGET_HEADROOM_RATIO < 0.5
    assert JUDGE_BUDGET_MIN_HEADROOM >= 30.0
