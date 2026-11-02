from nexus.core.types import ExtendedUsage


def test_extended_usage_completion_detail_shape_is_stable():
    usage = ExtendedUsage.model_validate(
        {
            'prompt_tokens': 1,
            'completion_tokens': 2,
            'total_tokens': 3,
            'completion_tokens_details': {
                'reasoning_tokens': 1,
            },
            'prompt_tokens_details': {
                'cached_tokens': 0,
            },
        }
    )

    dumped = usage.model_dump()

    assert dumped['completion_tokens_details']['image_tokens'] is None
    assert dumped['completion_tokens_details']['video_tokens'] is None
    assert dumped['completion_tokens_details']['text_tokens'] is None
    assert dumped['prompt_tokens_details']['image_tokens'] is None
    assert dumped['prompt_tokens_details']['video_tokens'] is None
    assert dumped['prompt_tokens_details']['text_tokens'] is None
