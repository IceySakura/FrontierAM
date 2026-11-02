"""Sub-service selection for the mm_agent reward-platform client.

Regression guard for the bug where every ``consist_judge`` sample scored
-100: the client matched ``response_datas[].class`` against the bare
``class2`` from ``router_id``, but the platform names its sub-services
``MultimodalReward<class2>``, so no route ever matched and the real score
was discarded.

Two layers are covered:

- the pure functions that decide *which* sub-service supplies the reward
  (``parse_reward_platform_router_id``, ``build_expected_sub_service_classes``,
  ``extract_sub_service_data``);
- the public entry point ``call_reward_platform_consist_judge`` with a
  stubbed HTTP layer, so the router_id -> expected-class wiring between
  those functions is covered too.
"""

import pytest

from nexus.extensions.tasks.mm_agent.reward_system import reward_platform_client
from nexus.extensions.tasks.mm_agent.reward_system.reward_platform_client import (
    DEFAULT_ROUTER,
    FAILED_SCORE,
    extract_sub_service_data,
    build_expected_sub_service_classes,
    call_reward_platform_consist_judge,
    parse_reward_platform_router_id,
)


# router_id -> the sub-service class the platform answers with, as observed in
# the bus0920 training run.
PRODUCTION_ROUTES = [
    ('MultimodalReward-GeneralV2TextOnly-0', 'MultimodalRewardGeneralV2TextOnly'),
    ('MMOCR-OCRModelScoring-0', 'MultimodalRewardOCRModelScoring'),
    ('MMOCR-OCRRuleScoring-0', 'MultimodalRewardOCRRuleScoring'),
    ('MultimodalChartReward-ChartModelScoring-0', 'MultimodalRewardChartModelScoring'),
    ('MultimodalReward-GroundingRule-0', 'MultimodalRewardGroundingRule'),
]


def _reply(*classes: str, error_code: int = 0, reward_score: float | str = 0.5) -> dict:
    """A RewardService reply carrying one ``response_datas`` entry per class."""
    return {
        'status': 0,
        'data': {
            'response_datas': [
                {'class': cls, 'error_code': error_code, 'data': {'reward_score': reward_score}} for cls in classes
            ]
        },
    }


# ---------------------------------------------------------------------------
# parse_reward_platform_router_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('router_id', [None, ''])
def test_missing_router_id_falls_back_to_the_default_router(router_id):
    assert parse_reward_platform_router_id(router_id) == DEFAULT_ROUTER


def test_router_id_splits_into_class1_class2_and_version():
    assert parse_reward_platform_router_id('MMOCR-OCRRuleScoring-0') == ('MMOCR', 'OCRRuleScoring', '0')


def test_only_the_last_two_hyphens_separate_fields_so_class1_may_contain_hyphens():
    # rsplit('-', 2), not split('-'): a hyphenated class1 must survive intact
    # rather than being rejected or silently truncated.
    assert parse_reward_platform_router_id('Multimodal-Chart-Reward-ChartModelScoring-0') == (
        'Multimodal-Chart-Reward',
        'ChartModelScoring',
        '0',
    )


@pytest.mark.parametrize(
    'router_id',
    [
        'noseparator',
        'MMOCR-OCRRuleScoring',  # missing router_version
        'MMOCR--0',  # empty class2
        '-OCRRuleScoring-0',  # empty class1
        'MMOCR-OCRRuleScoring-',  # empty router_version
        '--',  # all fields empty
    ],
)
def test_malformed_router_id_raises_instead_of_scoring_against_the_default_router(router_id):
    """A malformed router_id is a data bug and must surface as one.

    Falling back to ``DEFAULT_ROUTER`` here would return a plausible score
    from an unrelated judge, which is far harder to notice than a crash.
    """
    with pytest.raises(ValueError, match='<class1>-<class2>-<router_version>'):
        parse_reward_platform_router_id(router_id)


# ---------------------------------------------------------------------------
# build_expected_sub_service_classes
# ---------------------------------------------------------------------------


def test_candidate_classes_are_ordered_most_specific_first():
    assert build_expected_sub_service_classes('MMOCR', 'OCRRuleScoring') == (
        'MMOCROCRRuleScoring',
        'MultimodalRewardOCRRuleScoring',
        'OCRRuleScoring',
    )


@pytest.mark.parametrize(('class1', 'class2'), [('MMOCR', 'OCRRuleScoring'), ('MultimodalChartReward', 'Chart')])
def test_every_candidate_class_derives_from_the_requested_class2(class1, class2):
    # This is what keeps a match from ever pointing at a different judge.
    assert all(class2 in candidate for candidate in build_expected_sub_service_classes(class1, class2))


def test_default_router_reaches_the_legacy_tool_use_judge_without_hardcoding_it():
    class1, class2, _ = DEFAULT_ROUTER

    assert 'MultimodalRewardToolUseModelScoring' in build_expected_sub_service_classes(class1, class2)


# ---------------------------------------------------------------------------
# extract_sub_service_data
# ---------------------------------------------------------------------------


def test_extract_picks_the_first_candidate_class_present_in_the_reply():
    sub_data, err, sub_class = extract_sub_service_data(
        _reply('OCRRuleScoring', 'MultimodalRewardOCRRuleScoring'),
        ('MultimodalRewardOCRRuleScoring', 'OCRRuleScoring'),
    )

    assert err == ''
    assert sub_class == 'MultimodalRewardOCRRuleScoring'
    assert sub_data == {'reward_score': 0.5}


def test_extract_refuses_a_reply_holding_only_sub_services_nobody_asked_for():
    sub_data, err, sub_class = extract_sub_service_data(
        _reply('MultimodalRewardGeneralV2TextOnly'),
        ('MultimodalRewardOCRRuleScoring',),
    )

    assert sub_data is None
    assert sub_class is None
    assert "available=['MultimodalRewardGeneralV2TextOnly']" in err


def test_extract_accepts_a_unique_suffix_match_for_an_unknown_vendor_prefix():
    sub_data, err, sub_class = extract_sub_service_data(
        _reply('VendorXOCRRuleScoring'),
        ('MultimodalRewardOCRRuleScoring',),
        fallback_class_suffix='OCRRuleScoring',
    )

    assert err == ''
    assert sub_class == 'VendorXOCRRuleScoring'
    assert sub_data == {'reward_score': 0.5}


def test_extract_rejects_an_ambiguous_suffix_match():
    # Two plausible judges and no way to tell which one the route meant:
    # refusing is the only answer that cannot silently mis-score.
    sub_data, err, _ = extract_sub_service_data(
        _reply('VendorAOCRRuleScoring', 'VendorBOCRRuleScoring'),
        ('MultimodalRewardOCRRuleScoring',),
        fallback_class_suffix='OCRRuleScoring',
    )

    assert sub_data is None
    assert 'no sub_service matching' in err


def test_extract_does_not_suffix_match_when_no_suffix_is_supplied():
    sub_data, err, _ = extract_sub_service_data(_reply('VendorXOCRRuleScoring'), ('MultimodalRewardOCRRuleScoring',))

    assert sub_data is None
    assert 'no sub_service matching' in err


def test_extract_surfaces_a_sub_service_level_error_code():
    sub_data, err, sub_class = extract_sub_service_data(
        _reply('MultimodalRewardOCRRuleScoring', error_code=9),
        ('MultimodalRewardOCRRuleScoring',),
    )

    assert sub_data is None
    assert sub_class == 'MultimodalRewardOCRRuleScoring'
    assert 'error_code=9' in err


@pytest.mark.parametrize(
    ('resp_json', 'expected_error'),
    [
        ([], 'malformed response (not a dict)'),
        ({'status': 7, 'msg': 'boom'}, 'platform status=7'),
        ({'status': 0, 'data': {'response_datas': []}}, 'response_datas is empty'),
    ],
)
def test_extract_reports_malformed_or_failed_replies(resp_json, expected_error):
    sub_data, err, _ = extract_sub_service_data(resp_json, ('MultimodalRewardOCRRuleScoring',))

    assert sub_data is None
    assert expected_error in err


def test_extract_falls_back_to_the_single_reward_body_when_there_are_no_sub_services():
    # Some routes answer with a flat ``data.reward`` instead of response_datas.
    sub_data, err, sub_class = extract_sub_service_data(
        {'status': 0, 'data': {'reward': {'error_code': 0, 'final_score': 0.25}}},
        ('MultimodalRewardOCRRuleScoring',),
    )

    assert err == ''
    assert sub_class == 'reward'
    assert sub_data == {'error_code': 0, 'final_score': 0.25}


# ---------------------------------------------------------------------------
# call_reward_platform_consist_judge (router_id -> expected-class wiring)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _platform_token(monkeypatch):
    monkeypatch.setenv('REWARD_PLATFORM_TOKEN', 'test-token')


def _stub_platform(monkeypatch, *classes: str, reward_score: float = 1.0):
    """Make the client see a RewardService reply carrying ``classes``.

    Returns the list that captures each outgoing payload.
    """
    sent: list[dict] = []

    async def fake_post(url, payload, headers):
        sent.append(payload)
        return {
            'status': 0,
            'data': {
                'request_id': payload['request_id'],
                'response_datas': [
                    {
                        'class': cls,
                        'error_code': 0,
                        'data': {'reward_score': reward_score, 'res_json': {'is_consistent': 1}},
                    }
                    for cls in classes
                ],
            },
        }

    monkeypatch.setattr(reward_platform_client, 'post_to_reward_platform', fake_post)
    return sent


async def _score(router_id: str | None) -> tuple[float, dict]:
    return await call_reward_platform_consist_judge(
        question='q',
        response='a',
        target_answer='a',
        router_id=router_id,
    )


@pytest.mark.parametrize(('router_id', 'returned_class'), PRODUCTION_ROUTES)
async def test_scores_multimodal_reward_prefixed_sub_service(monkeypatch, router_id, returned_class):
    _stub_platform(monkeypatch, returned_class)

    score, details = await _score(router_id)

    assert score == 1.0, details.get('error')
    assert details['sub_service_class'] == returned_class
    assert 'error' not in details


async def test_sends_router_id_split_into_class1_class2_and_version(monkeypatch):
    sent = _stub_platform(monkeypatch, 'MultimodalRewardOCRModelScoring')

    await _score('MMOCR-OCRModelScoring-0')

    assert sent[0]['class1'] == 'MMOCR'
    assert sent[0]['class2'] == 'OCRModelScoring'
    assert sent[0]['router_version'] == '0'


async def test_scores_default_tool_use_sub_service_when_router_id_is_absent(monkeypatch):
    _stub_platform(monkeypatch, 'MultimodalRewardToolUseModelScoring')

    score, details = await _score(None)

    assert score == 1.0
    assert details['sub_service_class'] == 'MultimodalRewardToolUseModelScoring'


@pytest.mark.parametrize(('router_id', '_returned_class'), PRODUCTION_ROUTES)
async def test_refuses_legacy_tool_use_judge_for_an_explicit_route(monkeypatch, router_id, _returned_class):
    """A named route must never be scored by the legacy default judge.

    ``MultimodalRewardToolUseModelScoring`` is only legitimate when the sample
    carries no ``router_id``. Accepting it for an explicit route would report
    another judge's score under that route's name — silently, and with a
    plausible value that survives the ``-99`` masking threshold.
    """
    _stub_platform(monkeypatch, 'MultimodalRewardToolUseModelScoring')

    score, details = await _score(router_id)

    assert score == FAILED_SCORE
    assert 'no sub_service matching' in details['error']
    assert 'sub_service_class' not in details


async def test_prefers_exact_class_over_vendor_prefixed_one(monkeypatch):
    _stub_platform(monkeypatch, 'VendorXOCRModelScoring', 'MultimodalRewardOCRModelScoring')

    _, details = await _score('MMOCR-OCRModelScoring-0')

    assert details['sub_service_class'] == 'MultimodalRewardOCRModelScoring'


async def test_fails_on_ambiguous_suffix_instead_of_scoring_against_the_wrong_judge(monkeypatch):
    _stub_platform(monkeypatch, 'VendorAOCRModelScoring', 'VendorBOCRModelScoring')

    score, details = await _score('MMOCR-OCRModelScoring-0')

    assert score == FAILED_SCORE
    assert 'no sub_service matching' in details['error']


async def test_fails_on_unrelated_sub_service_instead_of_taking_the_first_one(monkeypatch):
    _stub_platform(monkeypatch, 'MultimodalRewardGeneralV2TextOnly')

    score, details = await _score('MMOCR-OCRModelScoring-0')

    assert score == FAILED_SCORE
    assert 'no sub_service matching' in details['error']


@pytest.mark.parametrize('failure', ['http_error', 'no_matching_sub_service', 'non_float_score'])
async def test_failed_judge_call_still_reports_the_platform_request_id(monkeypatch, failure):
    """The request id is what the platform side needs to trace a failed call."""
    sent: list[dict] = []

    async def fake_post(url, payload, headers):
        sent.append(payload)
        if failure == 'http_error':
            raise ConnectionError('boom')
        reward_score = 'n/a' if failure == 'non_float_score' else 1.0
        return _reply('MultimodalRewardGeneralV2TextOnly', reward_score=reward_score)

    monkeypatch.setattr(reward_platform_client, 'post_to_reward_platform', fake_post)

    route = (
        'MMOCR-OCRModelScoring-0' if failure == 'no_matching_sub_service' else 'MultimodalReward-GeneralV2TextOnly-0'
    )
    score, details = await _score(route)

    assert score == FAILED_SCORE
    assert details['error']
    assert details['platform_request_id'] == sent[0]['request_id']


async def test_malformed_router_id_propagates_before_any_request_is_sent(monkeypatch):
    """A malformed router_id must not be softened into a ``-100`` score.

    ``-100`` is indistinguishable from a genuine platform failure and would
    bury a data bug in the reward-error metrics; raising keeps it visible.
    Spending a platform round-trip on a request we already know is wrong
    would also be pure waste.
    """
    sent = _stub_platform(monkeypatch, 'MultimodalRewardOCRModelScoring')

    with pytest.raises(ValueError, match='<class1>-<class2>-<router_version>'):
        await _score('MMOCR--0')

    assert sent == []
