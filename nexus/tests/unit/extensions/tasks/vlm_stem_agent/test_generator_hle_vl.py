import json

import pytest

from nexus.extensions.tasks.vlm_stem_agent.generators import _common
from nexus.extensions.tasks.vlm_stem_agent.generators import hle_vl as hle_vl_generator
from nexus.runner.single import RunInput


SPEC = hle_vl_generator.SPEC
QUESTION = 'Which isotope is depicted in the decay chain?'


def _row(
    *,
    sample_id: str = '1',
    question: str = QUESTION,
    options: list[str] | None = None,
    answer: str = 'Pb-206',
) -> dict:
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'value': 'data:image/jpeg;base64,{cos_url}',
                        'cos_object_url': f'https://signed.example.invalid/hle_{sample_id}.jpg?signature=test',
                    },
                    {
                        'type': 'text',
                        'value': question,
                        'options': options or [],
                    },
                ],
            }
        ],
        'answer': [answer],
        'sample_id': sample_id,
        'extra_info': {'index': int(sample_id), 'ground_truth': answer},
    }


def test_exact_match_question_gets_the_hle_answer_format_block():
    config = _common._task_config(_row(), SPEC)

    assert config['question'] == (
        'Your response should be in the following format:\n'
        'Explanation: your explanation for your final answer\n'
        'Exact Answer: your succinct, final answer\n'
        'Confidence: your confidence score between 0% and 100% for your answer\n\n'
        f'{QUESTION}'
    )
    assert config['images'] == ['https://signed.example.invalid/hle_1.jpg?signature=test']
    assert config['answer'] == 'Pb-206'
    assert config['ground_truth'] == 'Pb-206'


def test_multiple_choice_question_switches_the_answer_label():
    row = _row(options=['A. A', 'B. B'], answer='A')

    config = _common._task_config(row, SPEC)

    assert 'Answer: your succinct, final answer' in config['question']
    assert 'Exact Answer:' not in config['question']
    assert _common.answer_type(row, SPEC) == 'multipleChoice'


def test_ground_truth_overrides_the_display_answer():
    row = _row()
    row['extra_info']['ground_truth'] = 'lead-206'

    config = _common._task_config(row, SPEC)

    assert config['answer'] == 'Pb-206'
    assert config['ground_truth'] == 'lead-206'


def test_placeholder_image_value_is_not_treated_as_base64():
    row = _row()
    row['messages'][0]['content'][0].pop('cos_object_url')

    with pytest.raises(ValueError, match='HTTPS URL, Base64 data URI, or absolute file path'):
        _common._task_config(row, SPEC)


def test_run_input_keeps_the_search_and_visit_tools(monkeypatch):
    monkeypatch.setenv('APP_ID', 'test-app')
    monkeypatch.setenv('APP_KEY', 'test-key')
    monkeypatch.setenv('HLE_VL_RUNTIME', 'local')
    monkeypatch.delenv('HLE_VL_MAX_ITERATIONS', raising=False)

    config = _common._build_base_run_input(SPEC)

    assert config.task_name == 'vlm_stem_agent'
    assert config.agent_config['max_iterations'] == 100
    assert config.agent_config['tools'] == [
        'search_stateful/search',
        'search_stateful/visit',
        'jupyter_with_image',
    ]
    assert config.agent_config['tool_overrides'] == [
        {'tool_name': 'search_stateful/search', 'name': 'search'},
        {'tool_name': 'search_stateful/visit', 'name': 'visit'},
        {'tool_name': 'jupyter_with_image', 'name': 'python'},
    ]
    assert set(config.tool_configs) == {
        'code_interpreter',
        'search_stateful',
    }
    assert config.tool_configs['search_stateful']['search_result_format'] == 'links'
    assert config.imports == [
        'nexus.runtime.providers.local',
        'nexus.extensions.openai_clients.litellm',
        'nexus.extensions.tasks.vlm_stem_agent.task',
        'nexus.extensions.agents.main.agent',
        'nexus.extensions.tools.search_stateful.search',
        'nexus.extensions.tools.search_stateful.visit',
        'nexus.extensions.tools.code_interpreter.jupyter_with_image',
    ]


def test_generate_configs_uses_the_hle_vl_env_prefix(tmp_path, monkeypatch):
    dataset = tmp_path / 'hle_vl.jsonl'
    dataset.write_text(
        ''.join(json.dumps(_row(sample_id=str(index))) + '\n' for index in (1, 2)),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        _common,
        '_build_base_run_input',
        lambda spec: RunInput(task_name='vlm_stem_agent'),
    )
    monkeypatch.setenv('HLE_VL_DATASET', str(dataset))
    monkeypatch.setenv('HLE_VL_LIMIT', '0')
    monkeypatch.delenv('HLE_VL_MIN_PIXELS', raising=False)
    monkeypatch.delenv('HLE_VL_MAX_PIXELS', raising=False)

    items = list(hle_vl_generator.generate_configs())

    assert [item.run_id for item in items] == ['hle_vl_1', 'hle_vl_2']
    assert items[0].metadata == {
        'index': 1,
        'sample_id': '1',
        'answer_type': 'exactMatch',
    }
