import asyncio
import json

import pytest

from nexus.extensions.projects.yb_vlm_oneagent.reward_system import yb_rubric_base
from nexus.extensions.projects.yb_vlm_oneagent.reward_system.yb_rubric import yb_rubric_reward


@pytest.mark.asyncio
async def test_judge_request_concurrency_does_not_exceed_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('YB_RUBRIC_MAX_CONCURRENT', '1')
    monkeypatch.setenv('YB_RUBRIC_N_ROUNDS_VAL', '3')
    monkeypatch.setenv('YB_RUBRIC_JUDGE_MAX_RETRIES', '1')

    active_requests = 0
    peak_requests = 0
    request_count = 0

    async def fake_get_model_response(**kwargs: object) -> str:
        nonlocal active_requests, peak_requests, request_count
        del kwargs
        active_requests += 1
        request_count += 1
        peak_requests = max(peak_requests, active_requests)
        try:
            await asyncio.sleep(0.01)
            return json.dumps(
                {
                    'scored_checklist': [
                        {
                            'id': 'answer_1',
                            'kind': 'need',
                            'need_type': 'answer',
                            'score': 1,
                        }
                    ]
                }
            )
        finally:
            active_requests -= 1

    monkeypatch.setattr(yb_rubric_base, 'get_model_response', fake_get_model_response)
    monkeypatch.setattr(yb_rubric_base, 'parse_json_block', json.loads)

    target_answer = json.dumps(
        {
            'rubric_topic': 'shiwu',
            'rubrics': [
                {
                    'id': 'answer_1',
                    'kind': 'need',
                    'need_type': 'answer',
                    'conclusion': '回答问题',
                }
            ],
        }
    )
    results = await asyncio.gather(
        *(
            yb_rubric_reward(
                question=f'question-{index}',
                response='answer',
                target_answer=target_answer,
                validate=True,
            )
            for index in range(2)
        )
    )

    assert request_count == 6
    assert peak_requests == 1
    assert [score for score, _ in results] == [1.0, 1.0]
