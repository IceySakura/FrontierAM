"""针对 ``compute_scores`` 中 v4.2 硬门限截断的测试。

背景：aidata v4.2 的评分曾经是普通加权和，因此仅通过 YAML 注释
``# [HARD GATE]`` 声明为“hard gate”的检查在运行时毫无效果（注释在加载
时会被丢弃）。 本文件的这些测试固定了新的机器可读硬门限契约：

* 检查通过 per-check ``hard_gate: true`` 或 rubric 层级的
  ``hard_gates: [id, ...]`` 数组来标记为门限；
* 未通过任一门限会将整个任务分数封顶（默认封顶为 ``0.0``）；
* ``config.hard_gate_cap`` 会软化封顶；截断只会降低分数；
* 遗留（非 v4.2）rubric 不受影响。
"""

from __future__ import annotations

import pytest

from nexus.extensions.tasks.aidata.engine.rubric import compute_scores, run_rubric_async


def _rubric(checks: list[dict], **extra) -> dict:
    r = {'version': '4.2', 'checks': checks}
    r.update(extra)
    return r


def _check(cid: str, weight: float, dim: str = 'correctness', **extra) -> dict:
    c = {'id': cid, 'description': cid, 'dimension': dim, 'weight': weight, 'assert': {'type': 'state_equals'}}
    c.update(extra)
    return c


def _row(cid: str, passed: bool, weight: float, dim: str = 'correctness', **extra) -> dict:
    row = {'id': cid, 'passed': passed, 'weight': weight, 'dimension': dim}
    row.update(extra)
    return row


def test_weighted_sum_without_gates() -> None:
    rubric = _rubric([_check('a', 0.6), _check('b', 0.4, 'functional')])
    rows = [_row('a', True, 0.6), _row('b', False, 0.4, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['score'] == pytest.approx(0.6)
    assert out['raw_score'] == pytest.approx(0.6)
    assert out['hard_gate_triggered'] is False
    assert out['hard_gate_failed'] == []


def test_per_check_hard_gate_failure_zeroes_score() -> None:
    rubric = _rubric([_check('gate', 0.2, hard_gate=True), _check('b', 0.8, 'functional')])
    rows = [_row('gate', False, 0.2, hard_gate=True), _row('b', True, 0.8, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['raw_score'] == pytest.approx(0.8)
    assert out['score'] == 0.0
    assert out['hard_gate_triggered'] is True
    assert out['hard_gate_failed'] == ['gate']


def test_top_level_hard_gates_list_without_row_echo() -> None:
    # 门限仅在 rubric 层级声明；rows 中不带 ``hard_gate`` 回声。
    rubric = _rubric([_check('gate', 0.2), _check('b', 0.8, 'functional')], hard_gates=['gate'])
    rows = [_row('gate', False, 0.2), _row('b', True, 0.8, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['score'] == 0.0
    assert out['hard_gate_failed'] == ['gate']


def test_passing_hard_gate_does_not_clamp() -> None:
    rubric = _rubric([_check('gate', 0.2, hard_gate=True), _check('b', 0.8, 'functional')])
    rows = [_row('gate', True, 0.2, hard_gate=True), _row('b', False, 0.8, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['score'] == pytest.approx(0.2)
    assert out['hard_gate_triggered'] is False
    assert out['hard_gate_failed'] == []


def test_soft_cap_via_config() -> None:
    rubric = _rubric(
        [_check('gate', 0.1, hard_gate=True), _check('b', 0.9, 'functional')],
        config={'hard_gate_cap': 0.5},
    )
    rows = [_row('gate', False, 0.1, hard_gate=True), _row('b', True, 0.9, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['raw_score'] == pytest.approx(0.9)
    assert out['score'] == pytest.approx(0.5)
    assert out['hard_gate_triggered'] is True


def test_cap_only_lowers_never_raises() -> None:
    rubric = _rubric(
        [_check('gate', 0.1, hard_gate=True), _check('b', 0.9, 'functional')],
        config={'hard_gate_cap': 0.5},
    )
    rows = [_row('gate', False, 0.1, hard_gate=True), _row('b', False, 0.9, 'functional')]
    out = compute_scores(rubric, rows)
    assert out['raw_score'] == pytest.approx(0.0)
    assert out['score'] == pytest.approx(0.0)


def test_multiple_failed_gates_reported() -> None:
    rubric = _rubric([
        _check('g1', 0.1, hard_gate=True),
        _check('g2', 0.1, hard_gate=True),
        _check('b', 0.8, 'functional'),
    ])
    rows = [
        _row('g1', False, 0.1, hard_gate=True),
        _row('g2', False, 0.1, hard_gate=True),
        _row('b', True, 0.8, 'functional'),
    ]
    out = compute_scores(rubric, rows)
    assert out['score'] == 0.0
    assert sorted(out['hard_gate_failed']) == ['g1', 'g2']


def test_legacy_rubric_unaffected() -> None:
    rubric = {'version': '2.0', 'checks': [{'id': 'a'}, {'id': 'b'}]}
    rows = [{'id': 'a', 'passed': True}, {'id': 'b', 'passed': False}]
    out = compute_scores(rubric, rows)
    assert out['score'] == pytest.approx(0.5)
    assert out['hard_gate_triggered'] is False
    assert out['hard_gate_failed'] == []


def test_empty_results() -> None:
    out = compute_scores({'version': '4.2', 'checks': []}, [])
    assert out['score'] == 0.0
    assert out['hard_gate_triggered'] is False


@pytest.mark.asyncio
async def test_hard_gate_flows_end_to_end() -> None:
    """检查上设置的 ``hard_gate`` 会被评估器回显并被强制执行。"""
    state = {'agent_answer': 'nope'}
    rubric = {
        'version': '4.2',
        'checks': [
            {
                'id': 'gate',
                'description': 'must say done',
                'dimension': 'correctness',
                'weight': 0.3,
                'hard_gate': True,
                'assert': {'type': 'text_contains', 'from': 'agent_answer', 'contains_all': ['done']},
            },
            {
                'id': 'b',
                'description': 'mentions nope',
                'dimension': 'functional',
                'weight': 0.7,
                'assert': {'type': 'text_contains', 'from': 'agent_answer', 'contains_all': ['nope']},
            },
        ],
    }
    rows = await run_rubric_async(state, rubric)
    gate_row = next(r for r in rows if r['id'] == 'gate')
    assert gate_row['hard_gate'] is True  # 评估器回显该标志
    assert gate_row['passed'] is False

    out = compute_scores(rubric, rows)
    assert out['raw_score'] == pytest.approx(0.7)
    assert out['score'] == 0.0
    assert out['hard_gate_triggered'] is True
    assert out['hard_gate_failed'] == ['gate']
