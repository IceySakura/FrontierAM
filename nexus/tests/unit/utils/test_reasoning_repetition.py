"""Threshold and exclusion behaviour of the ``let me`` repetition detector."""

from __future__ import annotations

import pytest

from nexus.utils.reasoning_repetition import apply_let_me_loop_penalty, detect_let_me_loop


def _assistant(reasoning: str | None = None, content: str | None = None) -> dict:
    message: dict = {'role': 'assistant', 'content': content}
    if reasoning is not None:
        message['reasoning_content'] = reasoning
    return message


DENSE_EN = 'let me check. let me look. let me try. let me see. let me read. let me run.'
DENSE_ZH = '让我看看。让我想想。让我试试。让我读读。让我跑跑。让我查查。'


def test_dense_markers_within_window_is_a_hit() -> None:
    result = detect_let_me_loop([_assistant(reasoning=DENSE_EN)])

    assert result['is_let_me_loop'] is True
    assert result['reasoning_hit'] is True
    assert result['content_hit'] is False
    assert result['max_window_count'] == 6
    assert result['reasoning_marker_count'] == 6
    assert result['hits'][0]['turn_index'] == 0
    assert result['hits'][0]['field'] == 'reasoning_content'


def test_three_markers_is_exactly_the_threshold() -> None:
    text = 'let me check. let me look. let me try.'

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['is_let_me_loop'] is True
    assert result['max_window_count'] == 3


def test_two_markers_is_below_threshold() -> None:
    text = 'let me check. let me look.'

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['is_let_me_loop'] is False
    assert result['marker_count'] == 2
    assert result['hits'] == []


def test_markers_spread_beyond_the_window_are_not_a_hit() -> None:
    # Spaced so that only two markers ever share a window: the filler is what keeps
    # them apart, and a shorter one would make this a hit at the current threshold.
    filler = ' ' + '.' * 98 + ' '
    text = filler.join(['let me check.'] * 6)

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['is_let_me_loop'] is False
    assert result['marker_count'] == 6


def test_chinese_markers_are_detected() -> None:
    result = detect_let_me_loop([_assistant(reasoning=DENSE_ZH)])

    assert result['is_let_me_loop'] is True
    assert result['max_window_count'] == 6


def test_rang_women_counts_as_a_marker() -> None:
    text = '让我们看看。让我们想想。让我们试试。让我们读读。让我们跑跑。让我们查查。'

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['is_let_me_loop'] is True


def test_content_is_scanned_and_reported_separately() -> None:
    result = detect_let_me_loop([_assistant(content=DENSE_EN)])

    assert result['is_let_me_loop'] is True
    assert result['content_hit'] is True
    assert result['reasoning_hit'] is False
    assert result['content_marker_count'] == 6
    assert result['reasoning_marker_count'] == 0


def test_reasoning_and_content_are_counted_independently() -> None:
    result = detect_let_me_loop([_assistant(reasoning=DENSE_EN, content=DENSE_ZH)])

    assert result['reasoning_hit'] is True
    assert result['content_hit'] is True
    assert result['hit_fields'] == 2
    assert result['marker_count'] == 12


def test_fenced_code_is_excluded() -> None:
    text = '```\n' + DENSE_EN + '\n```'

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['is_let_me_loop'] is False
    assert result['marker_count'] == 0


def test_inline_code_is_excluded() -> None:
    text = ' '.join([f'`let me {i}`' for i in range(6)])

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['marker_count'] == 0


def test_quoted_lines_are_excluded() -> None:
    text = '\n'.join([f'> let me check {i}' for i in range(6)])

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['marker_count'] == 0


def test_negated_markers_are_excluded() -> None:
    text = "don't let me. do not let me. cannot let me. never let me. won't let me. not let me."

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['marker_count'] == 0


def test_chinese_negation_is_excluded() -> None:
    text = '不让我。别让我。没让我。不让我。别让我。没让我。'

    result = detect_let_me_loop([_assistant(reasoning=text)])

    assert result['marker_count'] == 0


def test_letme_without_word_boundary_is_not_a_marker() -> None:
    result = detect_let_me_loop([_assistant(reasoning='outlet mercury ' * 8)])

    assert result['marker_count'] == 0


def test_non_assistant_messages_are_ignored() -> None:
    messages = [
        {'role': 'user', 'content': DENSE_EN},
        {'role': 'tool', 'content': DENSE_EN},
    ]

    result = detect_let_me_loop(messages)

    assert result['is_let_me_loop'] is False
    assert result['marker_count'] == 0


def test_empty_input_is_clean() -> None:
    result = detect_let_me_loop(None)

    assert result['is_let_me_loop'] is False
    assert result['marker_count'] == 0
    assert result['window_chars'] == 200
    assert result['min_occurrences'] == 3


def _hit_fields() -> dict:
    return {'let_me_loop': detect_let_me_loop([_assistant(reasoning=DENSE_EN)])}


def test_metrics_are_written_on_a_hit() -> None:
    extra_fields = _hit_fields()

    apply_let_me_loop_penalty(1.0, extra_fields)

    metrics = extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['let_me_loop_hit'] == 1.0
    assert metrics['let_me_loop_reasoning_hit'] == 1.0
    assert metrics['let_me_loop_content_hit'] == 0.0
    assert metrics['let_me_loop_max_window_count'] == 6.0
    assert metrics['let_me_loop_marker_count'] == 6.0
    assert metrics['let_me_loop_reasoning_marker_count'] == 6.0
    assert metrics['let_me_loop_content_marker_count'] == 0.0
    assert extra_fields['reward_extra_info']['is_let_me_loop'] == 1.0


def test_metrics_are_zero_on_a_clean_trajectory() -> None:
    extra_fields = {'let_me_loop': detect_let_me_loop([_assistant(reasoning='all good')])}

    apply_let_me_loop_penalty(1.0, extra_fields)

    assert extra_fields['reward_extra_info']['extra_reward_metrics']['let_me_loop_hit'] == 0.0


def test_is_a_noop_when_the_monitor_did_not_run() -> None:
    extra_fields: dict = {}

    assert apply_let_me_loop_penalty(1.0, extra_fields) == 1.0
    assert extra_fields == {}


def test_existing_metrics_are_preserved() -> None:
    extra_fields = _hit_fields()
    extra_fields['reward_extra_info'] = {'extra_reward_metrics': {'codeswitch_hit': 1.0}}

    apply_let_me_loop_penalty(1.0, extra_fields)

    metrics = extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['codeswitch_hit'] == 1.0
    assert metrics['let_me_loop_hit'] == 1.0


def test_default_is_flag_only() -> None:
    """The metric-only mode: a hit is reported as eligible but nothing is deducted."""
    extra_fields = _hit_fields()

    assert apply_let_me_loop_penalty(1.0, extra_fields) == 1.0

    metrics = extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['let_me_loop_eligible'] == 1.0
    assert metrics['let_me_loop_penalized'] == 0.0
    assert metrics['let_me_loop_penalty'] == 0.0


def test_penalty_is_deducted_above_the_threshold() -> None:
    extra_fields = _hit_fields()

    assert apply_let_me_loop_penalty(1.0, extra_fields, penalty=0.2) == pytest.approx(0.8)

    metrics = extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['let_me_loop_penalized'] == 1.0
    assert metrics['let_me_loop_penalty'] == pytest.approx(0.2)
    assert extra_fields['let_me_loop']['original_score'] == 1.0


def test_penalty_is_skipped_below_the_threshold() -> None:
    extra_fields = _hit_fields()

    assert apply_let_me_loop_penalty(0.5, extra_fields, penalty=0.2) == 0.5

    metrics = extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['let_me_loop_eligible'] == 0.0
    assert metrics['let_me_loop_penalized'] == 0.0


def test_penalty_is_capped_at_the_score() -> None:
    extra_fields = _hit_fields()

    assert apply_let_me_loop_penalty(1.0, extra_fields, penalty=5.0) == 0.0

    assert extra_fields['reward_extra_info']['extra_reward_metrics']['let_me_loop_penalty'] == 1.0


def test_a_clean_trajectory_is_never_penalized() -> None:
    extra_fields = {'let_me_loop': detect_let_me_loop([_assistant(reasoning='all good')])}

    assert apply_let_me_loop_penalty(1.0, extra_fields, penalty=0.2) == 1.0


def test_a_none_score_is_left_alone() -> None:
    extra_fields = _hit_fields()

    assert apply_let_me_loop_penalty(None, extra_fields, penalty=0.2) is None

    assert extra_fields['reward_extra_info']['extra_reward_metrics']['let_me_loop_penalized'] == 0.0
