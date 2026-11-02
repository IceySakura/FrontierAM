"""针对 ``text_contains`` 谓词的 v4.2 静态校验器覆盖。"""

from __future__ import annotations

from nexus.extensions.tasks.aidata.aidata import _validate_check


def test_text_contains_accepts_regex_only():
    """``text_contains`` 将 regex 作为一等公民谓词记录在案。"""
    check = {
        'id': 'trajectory_regex',
        'description': 'ordered trajectory pattern',
        'dimension': 'functional',
        'weight': 0.1,
        'assert': {
            'type': 'text_contains',
            'from': 'trajectory',
            'regex': 'query_tickets.*maps_ride_hailing',
        },
    }
    problems = _validate_check(
        check,
        i=0,
        is_v42=True,
        rubric_rel='rubric.yaml',
        seen_ids=set(),
    )
    assert problems == []
