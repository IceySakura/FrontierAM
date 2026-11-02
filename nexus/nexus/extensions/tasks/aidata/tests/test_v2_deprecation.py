"""v2 rubric 冻结测试（aidata engine 重构 Commit 2）。

设计：
- v2（version： "2" + verify: 块）已 **冻结** ： 现有 59 个生产 rubric
  仍按原样评分，但在加载时会发出 DeprecationWarning，并且 `aidata validate`
  会拒绝任何声明 version:"2" 的 **新** rubric。
- schema 常量（V42_ASSERT_TYPES 、 LEGACY_VERIFY_KINDS 、 METADATA_*_RE 等）
  统一到 case.schema； aidata.py 与 case.spec.py 从此处导入。
"""

import asyncio
import warnings

from nexus.extensions.tasks.aidata.rubric import run_rubric_async


V2_RUBRIC = {
    'version': '2',
    'checks': [
        {'id': 'x', 'verify': 'llm_judge', 'source': 'agent_answer', 'claim': 'test'},
    ],
}
V42_RUBRIC = {
    'version': '4.2',
    'checks': [
        {'id': 'x', 'assert': {'type': 'text_contains', 'from': 'agent_answer', 'contains': 'foo'}},
    ],
}
MINIMAL_STATE = {'agent_answer': 'foobar', 'trajectory': []}


def test_v2_rubric_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        asyncio.run(run_rubric_async(MINIMAL_STATE, V2_RUBRIC, runtime=None))
        matches = [w for w in caught if issubclass(w.category, DeprecationWarning) and 'v2 rubric' in str(w.message)]
        assert len(matches) == 1, (
            f'expected exactly 1 v2 DeprecationWarning, got {len(matches)}: {[str(w.message)[:120] for w in caught]}'
        )


def test_v42_rubric_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        asyncio.run(run_rubric_async(MINIMAL_STATE, V42_RUBRIC, runtime=None))
        matches = [w for w in caught if issubclass(w.category, DeprecationWarning) and 'v2 rubric' in str(w.message)]
        assert not matches, f'v4.2 rubric should not emit v2 warning; got: {[str(w.message)[:120] for w in matches]}'


def test_schema_constants_are_single_source():
    """aidata.py 与 case.schema 都暴露相同的常量（同一对象或集合相等）。"""
    from nexus.extensions.tasks.aidata import aidata
    from nexus.extensions.tasks.aidata.case import schema

    assert aidata._V42_ASSERT_TYPES is schema._V42_ASSERT_TYPES or set(aidata._V42_ASSERT_TYPES) == set(
        schema._V42_ASSERT_TYPES
    )
    assert aidata._LEGACY_VERIFY_KINDS is schema._LEGACY_VERIFY_KINDS or set(aidata._LEGACY_VERIFY_KINDS) == set(
        schema._LEGACY_VERIFY_KINDS
    )
    assert aidata._METADATA_DATE_RE.pattern == schema._METADATA_DATE_RE.pattern


def test_v2_rubric_still_scores_correctly():
    """v2 评估器（llm_judge/eq/regex/shell）仍会分发并产出行，
    即便触发了 DeprecationWarning。 现有的 59 个生产 rubric 依赖此行为。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        # eq 是确定性的 v2 primitive —— 测试成本最低
        v2_eq = {
            'version': '2',
            'checks': [
                {'id': 'sanity', 'verify': 'eq', 'source': 'agent_answer', 'expected': 'foobar'},
            ],
        }
        rows = asyncio.run(run_rubric_async(MINIMAL_STATE, v2_eq, runtime=None))
        assert len(rows) == 1, f'expected 1 row, got {len(rows)}'
        assert rows[0]['id'] == 'sanity'
        assert rows[0]['passed'], f'v2 eq should pass on matching value, got {rows[0]}'
