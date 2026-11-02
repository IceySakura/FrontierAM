
from nexus.extensions.tasks.mm_agent.reward_system.reward_platform_client import (
    build_expected_sub_service_classes,
    extract_sub_service_data,
)


PLATFORM_CLASS = 'MultimodalRewardPPTJudgeV2Scoring'


def _response(sub_class):
    return {
        'status': 0,
        'data': {'response_datas': [{'class': sub_class, 'error_code': 0, 'data': {'subj_accuracy': 0.75}}]},
    }


def test_versioned_route_accepts_the_platform_spelling():
    expected = build_expected_sub_service_classes('MultimodalReward', 'PPTJudgeScoringV2')
    assert PLATFORM_CLASS in expected
    data, error, hit = extract_sub_service_data(_response(PLATFORM_CLASS), expected)
    assert error == '', error
    assert hit == PLATFORM_CLASS
    assert data['subj_accuracy'] == 0.75


def test_unversioned_route_is_unchanged():
    assert build_expected_sub_service_classes('MultimodalReward', 'PPTJudgeScoring') == (
        'MultimodalRewardPPTJudgeScoring',
        'PPTJudgeScoring',
    )


def test_candidates_carry_no_duplicates():
    expected = build_expected_sub_service_classes('MultimodalReward', 'PPTJudgeScoringV2')
    assert len(expected) == len(set(expected))


def test_a_different_judge_is_still_refused():
    expected = build_expected_sub_service_classes('MultimodalReward', 'PPTJudgeScoringV2')
    data, error, _hit = extract_sub_service_data(_response('MultimodalRewardToolUseModelScoring'), expected)
    assert data is None
    assert 'no sub_service matching' in error
