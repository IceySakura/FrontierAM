"""针对 ``state_equals`` JMESPath 形状的 v4.2 静态验证器覆盖。

jmespath 分支通过 ``jmespath`` 键选择加入。 验证器必须：

* 接受具备九个谓词之一的良构 jmespath 形状；
* 拒绝误用（缺失查询、缺失谓词、多个谓词、
  ``count_op`` / ``count_value`` 半配对说明）；
* 让遗留的 ``equals`` / ``one_of`` / ``expected`` 形状继续通过，
  以便现有 rubric 不出现回退。
"""

from __future__ import annotations

import pytest

from nexus.extensions.tasks.aidata.aidata import _validate_check


def _check(assert_block: dict) -> dict:
    """将一个 ``assert`` block 包裹上最小必要的 v4.2 外围字段。"""
    return {
        'id': 'orders_paid_exists',
        'description': 'orders table has at least one paid record',
        'dimension': 'correctness',
        'weight': 0.1,
        'assert': assert_block,
    }


# ---------------------------------------------------------------------------
# 正常路径——每个谓词一个测试，让矩阵可读。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'extra',
    [
        {'exists': True},
        {'equals': 1},
        {'one_of': [1, 2, 3]},
        {'count_op': '>=', 'count_value': 1},
        {'contains': 'paid'},
        {'contains_all': ['paid', 'created']},
        {'contains_any': ['paid', 'refunded']},
        {'regex_match': r'^WS\d+$'},
        {'gt': 0},
        {'gte': 0},
        {'lt': 100},
        {'lte': 100},
    ],
)
def test_jmespath_shape_accepts_each_predicate(extra):
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': "life_payment_orders[?status=='paid']",
        **extra,
    })
    assert _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set()) == []


def test_jmespath_shape_without_from_is_allowed():
    """当省略 ``from`` 时， jmespath 分支默认作用于整体
    state ——验证器不得把缺失切片当作错误。
    """
    check = _check({
        'type': 'state_equals',
        'jmespath': 'agent_answer',
        'exists': True,
    })
    assert _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set()) == []


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


def test_empty_jmespath_query_is_flagged():
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': '   ',
        'exists': True,
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('jmespath query is empty' in p for p in problems), problems


def test_missing_predicate_is_flagged():
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('one predicate' in p for p in problems), problems


def test_multiple_predicates_is_flagged():
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].status',
        'equals': 'paid',
        'one_of': ['paid', 'refunded'],
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('exactly one predicate' in p for p in problems), problems


def test_count_op_without_count_value_is_flagged():
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
        'count_op': '>=',
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('count_op requires count_value' in p for p in problems), problems


def test_count_value_without_count_op_is_flagged():
    check = _check({
        'type': 'state_equals',
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
        'count_value': 1,
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('count_value requires count_op' in p for p in problems), problems


# ---------------------------------------------------------------------------
# 向后兼容——不带 `jmespath` 键的遗留形状继续通过。
# ---------------------------------------------------------------------------


def test_legacy_equals_shape_still_accepted():
    check = _check({
        'type': 'state_equals',
        'from': 'emails',
        'path': '0.subject',
        'equals': 'Welcome',
    })
    assert _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set()) == []


def test_legacy_shape_without_data_source_still_flagged():
    """回归护栏：非 jmespath 的 state_equals 若缺 ``from`` /
    ``source`` ，是最初的验证器失败模式，必须继续触发。
    """
    check = _check({
        'type': 'state_equals',
        'equals': 'paid',
    })
    problems = _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set())
    assert any('needs a data source' in p for p in problems), problems
