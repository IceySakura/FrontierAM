from __future__ import annotations

from openai.types.chat import ChatCompletion

from nexus.core.types import ExtendedChatCompletion


def test_completion_preserves_top_level_provider_fields() -> None:
    completion = ExtendedChatCompletion(
        id='resp_1',
        created=1,
        choices=[],
        provider_specific_fields={'provider_call': {'protocol': 'responses'}},
    )

    assert completion.model_dump()['provider_specific_fields']['provider_call']['protocol'] == 'responses'


def test_extended_message_preserves_reasoning_items():
    payload = {
        'id': 'chatcmpl-test',
        'created': 1,
        'model': 'openai/responses/gpt-5.5',
        'object': 'chat.completion',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'done',
                    'reasoning_items': [
                        {
                            'type': 'reasoning',
                            'id': 'rs_1',
                            'summary': [{'type': 'summary_text', 'text': 'summary'}],
                            'encrypted_content': 'encrypted-payload',
                            'status': 'completed',
                        }
                    ],
                },
            }
        ],
    }

    completion = ExtendedChatCompletion.model_validate(payload)
    message = completion.choices[0].message

    assert message.reasoning_items == [
        {
            'type': 'reasoning',
            'id': 'rs_1',
            'summary': [{'type': 'summary_text', 'text': 'summary'}],
            'encrypted_content': 'encrypted-payload',
            'status': 'completed',
        }
    ]
    assert (
        completion.model_dump(exclude_none=True)['choices'][0]['message']['reasoning_items'][0]['encrypted_content']
        == 'encrypted-payload'
    )


def test_from_openai_preserves_refusal_without_content() -> None:
    completion = ChatCompletion.model_validate({
        'id': 'chatcmpl-refusal',
        'created': 1,
        'model': 'gpt-test',
        'object': 'chat.completion',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'refusal': 'I cannot help with that request.',
                },
            }
        ],
    })

    extended = ExtendedChatCompletion.from_openai(completion)

    assert extended.choices[0].message.content is None
    assert extended.choices[0].message.refusal == 'I cannot help with that request.'


def test_from_openai_preserves_thinking_blocks_and_extended_usage() -> None:
    completion = ChatCompletion.model_validate({
        'id': 'chatcmpl-extended',
        'created': 1,
        'model': 'claude-opus',
        'object': 'chat.completion',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'done',
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': 'private reasoning',
                            'signature': 'signed-thinking',
                        }
                    ],
                },
            }
        ],
        'usage': {
            'prompt_tokens': 100,
            'completion_tokens': 10,
            'total_tokens': 110,
            'cache_creation_input_tokens': 30,
            'cache_read_input_tokens': 70,
            'cost': 0.25,
            'prompt_tokens_details': {
                'cached_tokens': 70,
                'cache_creation_tokens': 30,
                'text_tokens': 100,
                'image_tokens': 2,
            },
            'completion_tokens_details': {
                'reasoning_tokens': 5,
                'text_tokens': 10,
                'image_tokens': 1,
            },
        },
    })

    extended = ExtendedChatCompletion.from_openai(completion)

    thinking = extended.choices[0].message.thinking_blocks
    assert thinking is not None
    assert thinking[0].model_dump() == {
        'type': 'thinking',
        'thinking': 'private reasoning',
        'signature': 'signed-thinking',
    }
    assert extended.usage is not None
    assert extended.usage.cache_creation_input_tokens == 30
    assert extended.usage.cache_read_input_tokens == 70
    assert extended.usage.cost == 0.25
    assert extended.usage.prompt_tokens_details is not None
    assert extended.usage.prompt_tokens_details.cache_creation_tokens == 30
    assert extended.usage.prompt_tokens_details.text_tokens == 100
    assert extended.usage.prompt_tokens_details.image_tokens == 2
    assert extended.usage.completion_tokens_details is not None
    assert extended.usage.completion_tokens_details.text_tokens == 10
    assert extended.usage.completion_tokens_details.image_tokens == 1
