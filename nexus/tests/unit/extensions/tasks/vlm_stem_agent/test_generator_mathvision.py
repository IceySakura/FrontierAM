import json

from copy import deepcopy

import pytest

from nexus.extensions.tasks.vlm_stem_agent.generators import _common
from nexus.extensions.tasks.vlm_stem_agent.generators import mathvision as mathvision_generator
from nexus.runner.single import RunInput


SPEC = mathvision_generator.SPEC

OPEN_QUESTION = (
    'Hint: Please answer the question and provide the final answer at the end.\n'
    'Question: Which number replaces the question mark?'
)
CHOICE_QUESTION = (
    'Hint: Please answer the question and provide the correct option letter at '
    'the end.\nQuestion: Which bike is most expensive?\nChoices:\n'
    '(A) A\n(B) B\n(C) C\n(D) D\n(E) E'
)


def _row(
    *,
    sample_id: str = '1',
    question: str = OPEN_QUESTION,
    options: list[str] | None = None,
    answer: str = '60',
) -> dict:
    if options is None:
        options = []
    return {
        'id': None,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'value': 'data:image/jpeg;base64,{cos_url}',
                        'cos_url': (f'https://images.example.invalid/MathVision_{sample_id}_1.jpg'),
                        'cos_object_url': (
                            f'https://signed.example.invalid/MathVision_{sample_id}_1.jpg?signature=test'
                        ),
                    },
                    {
                        'type': 'text',
                        'value': question,
                        'question_type': 'opened',
                        'options': options,
                    },
                ],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'value': None}],
            },
        ],
        'answer': [answer],
        'sample_id': sample_id,
        'extra_info': {
            'index': int(sample_id),
            'question': question,
            'choices': repr([option[-1] for option in options]),
            'answer': answer.strip(),
            'level': 2,
            'category': 'arithmetic',
        },
    }


def test_open_question_is_passed_through_and_answer_is_stripped():
    row = _row(answer=' 60 ')

    config = _common._task_config(
        row,
        SPEC,
        min_pixels=2_257_920,
        max_pixels=8_000_000,
    )

    assert config == {
        'question': OPEN_QUESTION,
        'images': ['https://signed.example.invalid/MathVision_1_1.jpg?signature=test'],
        'answer': '60',
        'ground_truth': '60',
        'image_download_timeout': 60,
        'max_image_bytes': 20 * 1024 * 1024,
        'min_pixels': 2_257_920,
        'max_pixels': 8_000_000,
    }
    assert 'Explanation:' not in config['question']
    assert _common.answer_type(row, SPEC) == 'exactMatch'


def test_options_determine_answer_type_without_rewriting_question():
    row = _row(
        sample_id='2',
        question=CHOICE_QUESTION,
        options=['A. A', 'B. B', 'C. C', 'D. D', 'E. E'],
        answer='A',
    )

    config = _common._task_config(row, SPEC)

    assert config['question'] == CHOICE_QUESTION
    assert config['answer'] == 'A'
    assert _common.answer_type(row, SPEC) == 'multipleChoice'


def test_single_letter_open_answer_remains_exact_match():
    row = _row(answer='A', options=[])

    assert _common.answer_type(row, SPEC) == 'exactMatch'


def test_malformed_json_form_feed_in_latex_answer_is_restored():
    row = _row(answer='\x0crac{36}{13}')

    config = _common._task_config(row, SPEC)

    assert config['answer'] == r'\frac{36}{13}'
    assert config['ground_truth'] == r'\frac{36}{13}'


def test_placeholder_image_value_is_not_treated_as_base64():
    row = _row()
    image_item = row['messages'][0]['content'][0]
    image_item.pop('cos_object_url')
    image_item.pop('cos_url')

    with pytest.raises(ValueError, match='HTTPS URL, Base64 data URI, or absolute file path'):
        _common._task_config(row, SPEC)


def test_absolute_image_path_is_passed_through(tmp_path):
    row = _row()
    image_item = row['messages'][0]['content'][0]
    image_item.pop('cos_object_url')
    image_item.pop('cos_url')
    image_item['value'] = str(tmp_path / 'diagram.jpg')

    config = _common._task_config(row, SPEC)

    assert config['images'] == [str(tmp_path / 'diagram.jpg')]


def test_relative_image_path_is_rejected():
    row = _row()
    image_item = row['messages'][0]['content'][0]
    image_item.pop('cos_object_url')
    image_item.pop('cos_url')
    image_item['value'] = 'images/diagram.jpg'

    with pytest.raises(ValueError, match='HTTPS URL, Base64 data URI, or absolute file path'):
        _common._task_config(row, SPEC)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda row: row['messages'][0]['content'].__setitem__(
                slice(None),
                [row['messages'][0]['content'][0]],
            ),
            'exactly one text item',
        ),
        (
            lambda row: row.__setitem__('answer', []),
            'single-element',
        ),
    ],
)
def test_malformed_rows_fail_with_actionable_errors(mutation, message):
    row = _row()
    mutation(row)

    with pytest.raises(ValueError, match=message):
        _common._task_config(row, SPEC)


def test_pixel_bounds_are_optional_and_validated(monkeypatch):
    monkeypatch.setenv('MATHVISION_MIN_PIXELS', '2257920')
    monkeypatch.setenv('MATHVISION_MAX_PIXELS', '8000000')

    assert _common._pixel_bounds(SPEC) == (2_257_920, 8_000_000)

    monkeypatch.setenv('MATHVISION_MIN_PIXELS', '9000000')
    with pytest.raises(ValueError, match='must not exceed'):
        _common._pixel_bounds(SPEC)


def test_mathvision_run_input_keeps_only_python_tool(monkeypatch):
    monkeypatch.setenv('APP_ID', 'test-app')
    monkeypatch.setenv('APP_KEY', 'test-key')
    monkeypatch.setenv('MATHVISION_MAX_ITERATIONS', '17')
    monkeypatch.setenv('MATHVISION_INFERENCE_TIMEOUT', '101')
    monkeypatch.setenv('MATHVISION_OBSERVATION_TIMEOUT', '202')
    monkeypatch.setenv('MATHVISION_RUNTIME', 'local')

    config = _common._build_base_run_input(SPEC)

    assert config.task_name == 'vlm_stem_agent'
    assert config.agent_config['max_iterations'] == 17
    assert config.agent_config['inference_time_timeout'] == 101
    assert config.agent_config['observation_time_timeout'] == 202
    assert config.agent_config['tools'] == ['jupyter_with_image']
    assert config.agent_config['tool_overrides'] == [{'tool_name': 'jupyter_with_image', 'name': 'python'}]
    assert set(config.tool_configs) == {'code_interpreter'}
    assert config.imports == [
        'nexus.runtime.providers.local',
        'nexus.extensions.openai_clients.litellm',
        'nexus.extensions.tasks.vlm_stem_agent.task',
        'nexus.extensions.agents.main.agent',
        'nexus.extensions.tools.code_interpreter.jupyter_with_image',
    ]


@pytest.mark.parametrize(
    ('name', 'value', 'message'),
    [
        ('MATHVISION_CLIENT', 'legacy', 'must be one of'),
        ('MATHVISION_RUNTIME', 'docker', 'must be one of'),
        ('MATHVISION_TIMEOUT', '0', 'must be at least 1'),
        ('MATHVISION_MODEL_TIMEOUT', 'slow', 'must be an integer'),
    ],
)
def test_smoke_args_reject_invalid_environment_values(
    monkeypatch,
    name,
    value,
    message,
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        _common._smoke_args(SPEC)


def test_generate_configs_honors_limit_and_preserves_metadata(
    tmp_path,
    monkeypatch,
):
    rows = [
        _row(),
        _row(
            sample_id='2',
            question=CHOICE_QUESTION,
            options=['A. A', 'B. B', 'C. C', 'D. D', 'E. E'],
            answer='A',
        ),
    ]
    dataset = tmp_path / 'mathvision.jsonl'
    dataset.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )
    base = RunInput(task_name='vlm_stem_agent')
    monkeypatch.setattr(
        _common,
        '_build_base_run_input',
        lambda spec: base,
    )
    monkeypatch.setenv('MATHVISION_DATASET', str(dataset))
    monkeypatch.delenv('MATHVISION_MIN_PIXELS', raising=False)
    monkeypatch.delenv('MATHVISION_MAX_PIXELS', raising=False)

    monkeypatch.setenv('MATHVISION_LIMIT', '1')
    limited = list(mathvision_generator.generate_configs())
    assert [item.run_id for item in limited] == ['mathvision_1']

    monkeypatch.setenv('MATHVISION_LIMIT', '0')
    items = list(mathvision_generator.generate_configs())

    assert [item.run_id for item in items] == [
        'mathvision_1',
        'mathvision_2',
    ]
    assert items[0].metadata == {
        'index': 1,
        'sample_id': '1',
        'answer_type': 'exactMatch',
        'level': 2,
        'category': 'arithmetic',
    }
    assert items[1].metadata['answer_type'] == 'multipleChoice'
    assert items[0].config is not items[1].config
    assert items[0].config.task_config == _common._task_config(rows[0], SPEC)
    assert items[1].config.task_config == _common._task_config(rows[1], SPEC)


def test_duplicate_sample_ids_are_rejected(tmp_path, monkeypatch):
    first = _row()
    duplicate = deepcopy(first)
    dataset = tmp_path / 'duplicates.jsonl'
    dataset.write_text(
        json.dumps(first) + '\n' + json.dumps(duplicate) + '\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(
        _common,
        '_build_base_run_input',
        lambda spec: RunInput(task_name='vlm_stem_agent'),
    )
    monkeypatch.setenv('MATHVISION_DATASET', str(dataset))
    monkeypatch.setenv('MATHVISION_LIMIT', '0')

    with pytest.raises(ValueError, match='Duplicate MathVision sample_id'):
        list(mathvision_generator.generate_configs())


def test_generate_configs_rejects_negative_limit(monkeypatch):
    monkeypatch.setenv('MATHVISION_LIMIT', '-1')

    with pytest.raises(ValueError, match='must be at least 0'):
        list(mathvision_generator.generate_configs())
