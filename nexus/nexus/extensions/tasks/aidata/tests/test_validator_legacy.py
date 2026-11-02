from __future__ import annotations

from nexus.extensions.tasks.aidata.aidata import _validate_check


def _legacy_check(**overrides):
    check = {
        'id': 'check_1',
        'description': 'legacy check',
    }
    check.update(overrides)
    return check


def test_legacy_shell_does_not_require_source_or_claim() -> None:
    check = _legacy_check(verify='shell', command='exit 0')

    assert _validate_check(check, i=0, is_v42=False, rubric_rel='rubric.yaml', seen_ids=set()) == []


def test_legacy_regex_does_not_require_claim_and_defaults_source() -> None:
    check = _legacy_check(verify='regex', pattern='tool_name')

    assert _validate_check(check, i=0, is_v42=False, rubric_rel='rubric.yaml', seen_ids=set()) == []


def test_legacy_llm_judge_requires_claim_but_not_source() -> None:
    ok = _legacy_check(verify='llm_judge', claim='agent did the thing')
    missing_claim = _legacy_check(verify='llm_judge')

    assert _validate_check(ok, i=0, is_v42=False, rubric_rel='rubric.yaml', seen_ids=set()) == []
    assert _validate_check(missing_claim, i=0, is_v42=False, rubric_rel='rubric.yaml', seen_ids=set()) == [
        'rubric.yaml: checks[0].claim is empty'
    ]


def test_legacy_eq_requires_source_and_expected() -> None:
    check = _legacy_check(verify='eq')

    assert _validate_check(check, i=0, is_v42=False, rubric_rel='rubric.yaml', seen_ids=set()) == [
        'rubric.yaml: checks[0].source is empty',
        'rubric.yaml: checks[0].expected is missing',
    ]
