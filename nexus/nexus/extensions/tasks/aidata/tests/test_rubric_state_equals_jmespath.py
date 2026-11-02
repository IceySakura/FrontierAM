"""``state_equals`` JMESPath 分支的单元测试。

该分支通过 ``jmespath`` 键选择启用——出现该键时，evaluator 会针对选中的
state 切片（或整个 state 字典）运行查询，然后分派进入九个谓词之一。
这些测试固化每个谓词的行为，以及 rubric 作者在编写断言时可能遇到的错误
路径。
"""

from __future__ import annotations

import pytest

from nexus.extensions.tasks.aidata.engine.rubric import _eval_state_equals


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 下面大多数测试使用的、贴近现实的 life_payment 风格 state。
# 与 R3 dump_state 后 engine/state.py 组装出的形态保持一致。
# ---------------------------------------------------------------------------


def _life_payment_state() -> dict:
    return {
        'life_payment': {
            'life_payment_accounts': [
                {'account_id': 'ACC1', 'account_no': 'WS1234', 'type': 'water', 'auto_pay': False},
                {'account_id': 'ACC2', 'account_no': 'EL5678', 'type': 'electricity', 'auto_pay': True},
            ],
            'life_payment_orders': [
                {'order_id': 'O1', 'bill_ids': ['B-WS1234-01'], 'total_amount': 158.5, 'status': 'paid'},
                {'order_id': 'O2', 'bill_ids': ['B-EL5678-01'], 'total_amount': 240.0, 'status': 'created'},
            ],
            'life_payment_complaints': [
                {'account_id': 'WS1234', 'description': '住户白天不在家，请下午5点后上门', 'contact_phone': ''},
            ],
        },
        'agent_answer': '已为您查询用水量并报修，工单号 OUT-001。',
        'trajectory': [],
    }


async def _run(cfg: dict, state: dict | None = None):
    """便捷封装——_eval_state_equals 需要 runtime+junit_ctx，
    而在 jmespath 分支里我们并不关心它们。
    """
    return await _eval_state_equals(cfg, state or _life_payment_state(), runtime=None, junit_ctx=None)


# ---------------------------------------------------------------------------
# 谓词： exists
# ---------------------------------------------------------------------------


async def test_exists_true_matches_non_empty_list():
    cfg = {
        'from': 'life_payment',
        'jmespath': "life_payment_orders[?status=='paid']",
        'exists': True,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert passed
    assert verdict == 'Y'


async def test_exists_true_fails_on_empty_list():
    cfg = {
        'from': 'life_payment',
        'jmespath': "life_payment_orders[?status=='nonexistent']",
        'exists': True,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'N'


async def test_exists_false_passes_when_empty():
    """``exists: false`` 是断言 ``no record matches`` 的规范写法。"""
    cfg = {
        'from': 'life_payment',
        'jmespath': "life_payment_orders[?status=='refunded']",
        'exists': False,
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


# ---------------------------------------------------------------------------
# 谓词： equals + one_of （标量）
# ---------------------------------------------------------------------------


async def test_equals_on_scalar_jmespath_result():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].status',
        'equals': 'paid',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_equals_with_numeric_coercion():
    """``length(...)`` 返回 int； rubric 作者可能会写 equals: "1"。"""
    cfg = {
        'from': 'life_payment',
        'jmespath': "length(life_payment_orders[?status=='paid'])",
        'equals': '1',  # 故意用 str——应触发强制转换
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_one_of_hits():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].status',
        'one_of': ['paid', 'refunded'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_one_of_misses():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].status',
        'one_of': ['cancelled', 'expired'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert not passed


# ---------------------------------------------------------------------------
# 谓词： count_op + count_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('op', 'value', 'want'),
    [
        ('==', 1, True),
        ('==', 2, False),
        ('!=', 0, True),
        ('>=', 1, True),
        ('>', 0, True),
        ('<', 2, True),
        ('<=', 1, True),
    ],
)
async def test_count_op_all_operators(op, value, want):
    cfg = {
        'from': 'life_payment',
        'jmespath': "life_payment_orders[?status=='paid']",
        'count_op': op,
        'count_value': value,
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed is want


async def test_count_op_unknown_returns_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
        'count_op': '><',  # 不是真正的运算符
        'count_value': 1,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


async def test_count_op_missing_count_value_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
        'count_op': '>=',
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


async def test_count_op_non_sized_result_is_invalid():
    """``count_op`` 作用于标量（此处为 float）没有意义，
    应显式返回 INVALID 而不是悄悄做强制转换。
    """
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].total_amount',
        'count_op': '>=',
        'count_value': 1,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


# ---------------------------------------------------------------------------
# 谓词： contains / contains_all / contains_any
# ---------------------------------------------------------------------------


async def test_contains_substring_in_string_result():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_complaints[0].description',
        'contains': '下午5点',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_contains_element_in_list_result():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[*].status',
        'contains': 'paid',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_contains_all_passes_when_every_needle_present():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[*].status',
        'contains_all': ['paid', 'created'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_contains_all_fails_with_one_missing():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[*].status',
        'contains_all': ['paid', 'refunded'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert not passed


async def test_contains_any_passes_when_one_hits():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[*].status',
        'contains_any': ['refunded', 'paid'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_contains_any_fails_when_none_hit():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[*].status',
        'contains_any': ['refunded', 'expired'],
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert not passed


# ---------------------------------------------------------------------------
# 谓词： regex_match
# ---------------------------------------------------------------------------


async def test_regex_match_hits_string_result():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_complaints[0].description',
        'regex_match': r'(下午\s*5|17[:点])',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_regex_match_against_stringified_list():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_accounts[*].account_no',
        'regex_match': r'WS\d{4}',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_regex_match_invalid_pattern_surfaces_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_complaints[0].description',
        'regex_match': '[unterminated',
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


# ---------------------------------------------------------------------------
# 谓词： gt / gte / lt / lte （数值）
# ---------------------------------------------------------------------------


async def test_gte_passes():
    cfg = {
        'from': 'life_payment',
        'jmespath': "sum(life_payment_orders[?status=='paid'].total_amount)",
        'gte': 100,
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_lt_fails_when_value_too_big():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'sum(life_payment_orders[*].total_amount)',
        'lt': 100,  # 实际总和为 398.5
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert not passed


async def test_gt_against_non_numeric_result_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].order_id',
        'gt': 0,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


# ---------------------------------------------------------------------------
# 切片处理： `from` 可选、缺失切片、原始 state 访问
# ---------------------------------------------------------------------------


async def test_from_omitted_walks_entire_state():
    """省略 ``from`` → 查询作用于整个 state 字典，让
    rubric 可以跨 mock 命名空间（例如同时确认 life_payment
    与 wechat 都被触及）。
    """
    cfg = {
        'jmespath': 'agent_answer',
        'exists': True,
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_missing_slice_returns_missing_verdict():
    cfg = {
        'from': 'no_such_mock',
        'jmespath': 'anything[*]',
        'exists': True,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'MISSING'


# ---------------------------------------------------------------------------
# 错误路径：空查询、错误查询、缺失/多个谓词
# ---------------------------------------------------------------------------


async def test_empty_jmespath_query_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': '   ',
        'exists': True,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


async def test_malformed_query_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[?status==',  # 未终止的表达式
        'exists': True,
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


async def test_no_predicate_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders',
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


async def test_multiple_predicates_is_invalid():
    cfg = {
        'from': 'life_payment',
        'jmespath': 'life_payment_orders[0].status',
        'equals': 'paid',
        'one_of': ['paid', 'refunded'],
    }
    passed, verdict, _detail, _head = await _run(cfg)
    assert not passed
    assert verdict == 'INVALID'


# ---------------------------------------------------------------------------
# 向后兼容：老式 state_equals 形态必须继续原样工作
# ---------------------------------------------------------------------------


async def test_legacy_slice_path_branch_still_works():
    cfg = {
        'from': 'life_payment',
        'path': 'life_payment_orders.0.status',
        'equals': 'paid',
    }
    passed, _verdict, _detail, _head = await _run(cfg)
    assert passed


async def test_legacy_flat_dotted_path_branch_still_works():
    state = {'trajectory': [{'role': 'user'}, {'role': 'agent'}]}
    cfg = {'source': 'trajectory.0.role', 'equals': 'user'}
    passed, _verdict, _detail, _head = await _eval_state_equals(cfg, state, runtime=None, junit_ctx=None)
    assert passed
