import base64
import copy

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nexus.extensions.tasks.reward_system.gsm8k import compute_score as compute_gsm8k_score
from nexus.extensions.tasks.reward_system.schema import validate_prompt, validate_task_config
from nexus.extensions.tasks.reward_system.score import compute_score
from nexus.extensions.tasks.reward_system.task import RewardConfig, RewardSystemTaskInput, reward_system_task
from nexus.runner.single import RunInput
from nexus.utils.reward_system import RewardSystemClient


def test_gsm8k_scoring_keeps_legacy_extraction_rules():
    assert compute_gsm8k_score('work\n#### 1,234', '1234') == 1.0
    assert compute_gsm8k_score('work\n#### 1,234', '1234', method='flexible') == 0.0
    assert compute_gsm8k_score('no final answer', '1234') == 0.0


@pytest.mark.asyncio
async def test_unified_task_does_not_mutate_generation_prompt(monkeypatch):
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr('nexus.extensions.tasks.reward_system.task.compute_score', scorer)
    prompt = [{'role': 'user', 'content': '<image> solve'}]
    agent = AsyncMock(return_value=SimpleNamespace(response='#### 42'))
    result = await reward_system_task(
        RewardSystemTaskInput(
            prompt=prompt,
            run_agent=agent,
            reward=RewardConfig(scorer='gsm8k', reference_answer='42'),
        )
    )
    assert result.score == 1.0
    assert prompt == [{'role': 'user', 'content': '<image> solve'}]
    assert agent.call_args.kwargs['prompt'] == prompt
    assert scorer.call_args.kwargs['prompt'] == prompt


@pytest.mark.asyncio
async def test_user_prompt_mode_supports_agents_with_final_answer(monkeypatch):
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr('nexus.extensions.tasks.reward_system.task.compute_score', scorer)
    result = await reward_system_task(
        RewardSystemTaskInput(
            prompt=[{'role': 'user', 'content': 'q'}],
            run_agent=AsyncMock(return_value=SimpleNamespace(final_answer='#### 42')),
            reward=RewardConfig(
                scorer='gsm8k',
                reference_answer='42',
                question='q',
                agent_input_mode='user_prompt',
            ),
        )
    )
    assert result.response == '#### 42'
    assert scorer.call_args.kwargs['response'] == '#### 42'


@pytest.mark.asyncio
async def test_reward_system_message_shape_matches_legacy_builder(monkeypatch):
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'unit-test')
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr(RewardSystemClient, 'get_score', scorer)
    prompt = [{'role': 'user', 'content': 'question'}]
    await reward_system_task(
        RewardSystemTaskInput(
            prompt=prompt,
            run_agent=AsyncMock(return_value=SimpleNamespace(response='answer')),
            reward=RewardConfig(router_id='Science-HYA3B-5', reference_answer='a'),
        )
    )
    assert scorer.call_args.kwargs['messages'] == [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'answer'},
    ]


@pytest.mark.asyncio
async def test_reward_system_sends_local_images_as_data_urls(monkeypatch, tmp_path):
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'unit-test')
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr(RewardSystemClient, 'get_score', scorer)
    (tmp_path / 'a.png').write_bytes(b'png-a')
    (tmp_path / 'b.jpg').write_bytes(b'jpg-b')
    data_url = 'data:image/png;base64,AAAA'
    remote_url = 'https://x/c.png'
    prompt = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': str(tmp_path / 'a.png'), 'detail': 'high'}},
                {'type': 'image_url', 'image_url': {'url': f'file://{tmp_path / "b.jpg"}'}},
                {'type': 'image_url', 'image_url': {'url': data_url}},
                {'type': 'image_url', 'image_url': {'url': remote_url}},
                {'type': 'text', 'text': 'question'},
            ],
        }
    ]
    original_prompt = copy.deepcopy(prompt)
    await compute_score(
        scorer='reward_system', response='answer', reference_answer='a', prompt=prompt, router_id='Science-HYA3B-5'
    )
    assert scorer.call_args.kwargs['messages'][1]['content'] == [
        {
            'type': 'image_url',
            'image_url': {'url': f'data:image/png;base64,{base64.b64encode(b"png-a").decode()}', 'detail': 'high'},
        },
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64.b64encode(b"jpg-b").decode()}'}},
        {'type': 'image_url', 'image_url': {'url': data_url}},
        {'type': 'image_url', 'image_url': {'url': remote_url}},
        {'type': 'text', 'text': 'question'},
    ]
    assert prompt == original_prompt


@pytest.mark.asyncio
async def test_reward_system_strip_multimodal_does_not_read_images(monkeypatch):
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'unit-test')
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr(RewardSystemClient, 'get_score', scorer)
    prompt = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': '/missing/a.png'}},
                {'type': 'text', 'text': 'question'},
            ],
        }
    ]
    await compute_score(
        scorer='reward_system',
        response='answer',
        reference_answer='a',
        prompt=prompt,
        router_id='Science-HYA3B-5',
        strip_multimodal=True,
    )
    assert scorer.call_args.kwargs['messages'][1] == {'role': 'user', 'content': 'question'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('raw', 'reasoning', 'expected'),
    [
        ('full</think:token>answer', 'discarded', 'full</think:token>answer'),
        (None, 'full', 'full</think:token>answer'),
        (None, None, 'answer'),
    ],
)
async def test_reward_system_uses_raw_response_before_parsed_answer(monkeypatch, raw, reasoning, expected):
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr('nexus.extensions.tasks.reward_system.task.compute_score', scorer)
    result = await reward_system_task(
        RewardSystemTaskInput(
            prompt=[{'role': 'user', 'content': 'q'}],
            run_agent=AsyncMock(
                return_value=SimpleNamespace(
                    response='answer',
                    raw_response=raw,
                    reasoning_content=reasoning,
                )
            ),
            reward=RewardConfig(
                scorer='reward_system', reference_answer='a', router_id='router', reasoning_end_token='</think:token>'
            ),
        )
    )
    assert result.response == expected
    assert scorer.call_args.kwargs['response'] == expected


@pytest.mark.asyncio
async def test_verifier_uses_decoded_response_for_legacy_equivalence(monkeypatch):
    scorer = AsyncMock(return_value=1.0)
    monkeypatch.setattr('nexus.extensions.tasks.reward_system.task.compute_score', scorer)
    result = await reward_system_task(
        RewardSystemTaskInput(
            prompt=[{'role': 'user', 'content': 'q'}],
            run_agent=AsyncMock(
                return_value=SimpleNamespace(
                    response='#### 42',
                    raw_response='<think>different wire form</think>#### 999',
                    reasoning_content='different reasoning',
                )
            ),
            reward=RewardConfig(scorer='gsm8k', reference_answer='42'),
        )
    )
    assert result.response == '#### 42'
    assert scorer.call_args.kwargs['response'] == '#### 42'


@pytest.mark.asyncio
async def test_dispatches_gsm8k_without_reward_system(monkeypatch):
    gsm = Mock(return_value=1.0)
    monkeypatch.setattr('nexus.extensions.tasks.reward_system.score.compute_gsm8k_score', gsm)
    result = await compute_score(scorer='gsm8k', response='#### 42', reference_answer='42')
    assert result == 1.0
    gsm.assert_called_once()


@pytest.mark.parametrize(
    'prompt',
    [
        [{'role': 'user', 'content': 'plain text'}],
        [{'role': 'system', 'content': ''}, {'role': 'user', 'content': 'plain text'}],
        [
            {'role': 'system', 'content': ''},
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': '/ceph/images/a.png'}},
                    {'type': 'text', 'text': 'question'},
                ],
            },
        ],
        [
            {
                'role': 'user',
                'content': [{'type': 'image_url', 'image_url': {'url': 'https://x/a.png', 'detail': 'high'}}],
            }
        ],
        [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///ceph/a.png'}}]}],
        [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}],
        [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'follow-up'},
        ],
    ],
)
def test_canonical_prompt_accepts_the_openai_subset(prompt):
    validate_prompt(prompt)


@pytest.mark.parametrize(
    ('prompt', 'message'),
    [
        ([], 'non-empty'),
        ([{'role': 'developer', 'content': 'x'}, {'role': 'user', 'content': 'q'}], 'role must be'),
        ([{'role': 'user', 'name': 'alice', 'content': 'q'}], 'unsupported keys'),
        ([{'role': 'user', 'content': [{'type': 'text', 'text': 'q'}]}], 'text-only content must be a string'),
        (
            [{'role': 'system', 'content': [{'type': 'text', 'text': ''}]}, {'role': 'user', 'content': 'q'}],
            'system content',
        ),
        ([{'role': 'user', 'content': [{'type': 'image', 'image': '/a.png'}]}], "'text' or 'image_url'"),
        ([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'images/a.png'}}]}], 'absolute path'),
        (
            [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': '/ceph/imgs/756ab247'}}]}],
            'no recognised image file extension',
        ),
        (
            [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'file:///ceph/notes.txt'}}]}],
            'no recognised image file extension',
        ),
        (
            [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:text/plain,x'}}]}],
            'base64 images',
        ),
        (
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': '/a.png'}},
                        {'type': 'text', 'text': 'a'},
                        {'type': 'text', 'text': 'b'},
                    ],
                }
            ],
            'adjacent text parts',
        ),
        ([{'role': 'user', 'content': '<image>q'}, {'role': 'assistant', 'content': 'a'}], 'must end with'),
        (
            [{'role': 'assistant', 'content': 'a', 'reasoning_content': 'r'}, {'role': 'user', 'content': 'q'}],
            'unsupported keys',
        ),
    ],
)
def test_canonical_prompt_rejects_non_canonical_messages(prompt, message):
    with pytest.raises(ValueError, match=message):
        validate_prompt(prompt)


def test_task_input_rejects_non_canonical_prompt():
    with pytest.raises(ValueError, match='text-only content must be a string'):
        RewardSystemTaskInput(
            prompt=[{'role': 'user', 'content': [{'type': 'text', 'text': 'q'}]}],
            run_agent=AsyncMock(),
            reward=RewardConfig(scorer='gsm8k', reference_answer='1'),
        )


def test_stored_task_config_rejects_fields_the_runner_would_drop():
    reward = {'scorer': 'gsm8k', 'reference_answer': '1'}
    validate_task_config({'prompt': [{'role': 'user', 'content': 'q'}], 'reward': reward})
    with pytest.raises(ValueError, match=r"unsupported keys \['images'\]"):
        validate_task_config({
            'prompt': [{'role': 'user', 'content': '<image>q'}],
            'images': ['/a.png'],
            'reward': reward,
        })


@pytest.mark.parametrize('task_name', ['reward_system', 'any_other_task'])
def test_prompt_and_reward_text_is_never_interpolated(task_name):
    run_input = RunInput.from_dict({
        'task_name': task_name,
        'agent_name': 'single_turn',
        'task_config': {
            'prompt': [{'role': 'user', 'content': 'compute ${x} and ${oc.env:HOME}'}],
            'reward': {'scorer': 'gsm8k', 'reference_answer': '${oc.select:a.b,default}'},
        },
        'agent_config': {'prompt': [{'role': 'user', 'content': 'path ${oc.env:HOME}'}]},
    })
    assert run_input.task_config['prompt'][0]['content'] == 'compute ${x} and ${oc.env:HOME}'
    assert run_input.task_config['reward']['reference_answer'] == '${oc.select:a.b,default}'
    assert run_input.agent_config['prompt'][0]['content'] == 'path ${oc.env:HOME}'
