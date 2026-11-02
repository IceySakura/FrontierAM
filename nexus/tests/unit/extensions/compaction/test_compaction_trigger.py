import pytest

from nexus.extensions.compaction.compaction_trigger import CompactionContext, get_compaction_condition


@pytest.mark.parametrize('condition_name', ['remaining_tokens_exhausted', 'current_segment_consumed_tokens_exhausted'])
def test_ratio_uses_max_model_len(condition_name: str) -> None:
    condition = get_compaction_condition(condition_name, ratio=0.5, max_model_len=100)

    decision = condition.should_compact(
        CompactionContext(current_prompt_tokens=50, segment_count=0, max_segments=None, curr_segment_consumed_tokens=50)
    )

    assert decision.should_compact
    assert 'max_model_len(100)' in decision.reason


def test_ratio_rejects_two_length_bases() -> None:
    with pytest.raises(ValueError, match='either max_model_len or max_position_embeddings'):
        get_compaction_condition(
            'remaining_tokens_exhausted', ratio=0.5, max_model_len=100, max_position_embeddings=200
        )
