"""针对 ``tool_called`` 计数边界的 v4.2 静态验证器覆盖测试。"""

from __future__ import annotations

from nexus.extensions.tasks.aidata.aidata import _validate_check


def _check(**assert_overrides):
    assert_cfg = {
        'type': 'tool_called',
        'tool': 'email_send',
    }
    assert_cfg.update(assert_overrides)
    return {
        'id': 'tool_count',
        'description': 'tool count bounds',
        'dimension': 'correctness',
        'weight': 1,
        'assert': assert_cfg,
    }


def test_tool_called_count_bounds_accept_valid_range() -> None:
    assert (
        _validate_check(
            _check(min_count=0, max_count=1),
            i=0,
            is_v42=True,
            rubric_rel='rubric.yaml',
            seen_ids=set(),
        )
        == []
    )


def test_tool_called_count_bounds_reject_invalid_range() -> None:
    assert _validate_check(
        _check(min_count=3, max_count=1),
        i=0,
        is_v42=True,
        rubric_rel='rubric.yaml',
        seen_ids=set(),
    ) == ['rubric.yaml: checks[0].assert[tool_called] min_count=3 exceeds max_count=1']


def test_tool_called_accepts_acceptable_tools_as_identity() -> None:
    check = _check()
    check['assert'].pop('tool')
    check['assert']['acceptable_tools'] = ['wecom_im_history', 'wecom/im_search']

    assert (
        _validate_check(
            check,
            i=0,
            is_v42=True,
            rubric_rel='rubric.yaml',
            seen_ids=set(),
        )
        == []
    )


def test_tool_called_rejects_bad_acceptable_tools_shape() -> None:
    assert _validate_check(
        _check(acceptable_tools={'bad': 'shape'}),
        i=0,
        is_v42=True,
        rubric_rel='rubric.yaml',
        seen_ids=set(),
    ) == ['rubric.yaml: checks[0].assert[tool_called] acceptable_tools must be a list of strings']


def test_tool_called_sequence_accepts_acceptable_tools_step() -> None:
    check = {
        'id': 'seq',
        'description': 'sequence',
        'dimension': 'correctness',
        'weight': 1,
        'assert': {
            'type': 'tool_called_sequence',
            'steps': [
                {'acceptable_tools': ['wecom.im_history']},
            ],
        },
    }

    assert _validate_check(check, i=0, is_v42=True, rubric_rel='rubric.yaml', seen_ids=set()) == []
