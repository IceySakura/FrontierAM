"""Regression cases for qualified thinking stutters and malformed formatting."""

import time

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator.rules._reasoning_stutter import (
    MIN_OCCURRENCES,
    WORD_MIN_REPEATS,
    occurrences,
)

from .test_reasoning_stutter import RULE, detect


@pytest.mark.parametrize('word', WORD_MIN_REPEATS)
def test_new_vocabulary_and_per_word_threshold(word):
    minimum = WORD_MIN_REPEATS[word]
    text = 'Compare ' + ' '.join([word] * minimum) + ' updates.'
    [event] = occurrences(text)
    assert event['word'] == word
    assert event['repeats'] == minimum
    assert occurrences(' '.join([word] * (minimum - 1))) == []


@pytest.mark.parametrize('word', ['good', 'yes', 'ok'])
def test_excluded_vocabulary(word):
    assert occurrences(' '.join([word] * 10)) == []


@pytest.mark.parametrize(
    'text',
    [
        'Throttling independent, maybe maybe.',
        'Maybe maybe no need to re-read.',
        'Perhaps perhaps not viable after all.',
        'Maybe maybe true.',
        'Perhaps perhaps cache? No.',
        'Maybe maybe tomorrow yes.',
        '2776 words maybe maybe too long.',
        'identify exact from phrase perhaps perhaps Abe Vigoda.',
        'Need an example maybe maybe check this.',
        'Need Need to retry the request.',
        'Maybe Maybe it works.',
        'Check root maybe maybe this is the cause.',
        '    Maybe maybe we should check.',
        '> quoted output\nMaybe maybe we should check.',
        'An orphan ` followed by maybe maybe a real check.',
        'An orphan `\nMaybe maybe we should check.\n` next line',
        'from estate," maybe maybe answer should be. Finish with "done".',
        'An orphan " followed by maybe maybe a real check.',
        'An orphan "\nMaybe maybe we should check.\n" next line',
        '<code>example\nMaybe maybe we should check.\nordinary prose</code>',
        '<code>\nexample\n\nMaybe maybe we should check.\n</code>',
        '<code>unfinished maybe maybe we should check.',
    ],
)
def test_prose_is_not_hidden_by_broad_guards_or_malformed_delimiters(text):
    [event] = occurrences(text)
    assert event['word'] in {'maybe', 'perhaps', 'need'}
    assert text[event['start'] : event['end']].lower() == f'{event["word"]} {event["word"]}'


@pytest.mark.parametrize(
    'text',
    [
        'I need need[0] to be zero.',
        'I need need+1 total sectors.',
        'Call current current() and use use[0].',
        'ns::maybe maybe. Maybe maybe::value.',
        'The nested type is Maybe Maybe Int.',
        'The type is Maybe Maybe a.',
        'The constructor returns maybe Maybe a.',
        'Could flatten Maybe Maybe to one present/get.',
        'For root Maybe maybe.',
        'List of Maybe: field_count.\nMaybe Maybe: both present same symbol impossible.',
        'We need Need as the parser status.',
        'People in need need help.',
        'The people we need need to know Python.',
        'The need need not be urgent.',
        'The visitor who was expected expected a welcome.',
        'We expected expected results.',
        'We normalized normalized vectors again.',
        'I met her last last week.',
        'Need CLDR likely likely subtags for en.',
        'The field actual actual values are defined.',
        'The generated keyword sequence is then then then.',
        'The phrase is maybe maybe.',
        'Example: perhaps perhaps.',
        'maybe maybe is a stutter.',
        '`maybe maybe`',
        '``maybe maybe``',
        '"maybe maybe"',
        '["ok","maybe maybe"]',
        '{"a":"x","maybe maybe":"need need"}',
        '[1,"need need"]',
        '[["ok"],"perhaps perhaps"]',
        '{"ok":{},"maybe maybe":true}',
        '[true,"need need",null,"maybe maybe"]',
        "'maybe maybe'",
        '> maybe maybe',
        '    value = "maybe maybe"',
        '```text\nmaybe maybe\n```',
        '~~~text\nmaybe maybe\n~~~',
        '<code>maybe maybe</code>',
        '<pre>\nmaybe maybe\n</pre>',
        '$maybe maybe$',
        '\\(maybe maybe\\)',
    ],
)
def test_explicit_code_variables_types_and_examples_are_excluded(text):
    assert occurrences(text) == []


@pytest.mark.parametrize(
    'prefix',
    [
        "Use `print('\"')`.",
        'The reference says "An orphan ` is literal".',
        '<code>An orphan ` is literal</code>',
        '$x = ` + y$',
        'Use \\` as literal punctuation.',
    ],
)
def test_delimiter_ownership_preserves_following_prose(prefix):
    text = prefix + ' Use `THEN then`; we need need a retry; finish with "done".'
    [event] = occurrences(text)
    assert event['start'] == text.index('need need')


@pytest.mark.parametrize('gap', ['\t', '\xa0', '\u2003'])
def test_horizontal_whitespace_offsets(gap):
    text = 'prefix need' + gap + 'need inspect.'
    [event] = occurrences(text)
    assert text[event['start'] : event['end']] == 'need' + gap + 'need'


@pytest.mark.parametrize('gap', ['\n', '\r', '\v', '\f', '\x85', '\u2028', '\u2029'])
def test_runs_do_not_cross_line_boundaries(gap):
    assert occurrences('need' + gap + 'need inspect.') == []


def test_source_content_offsets_and_trajectory_threshold_remain_three():
    assert MIN_OCCURRENCES == 3
    content = '<think:6124c78e>likely likely check. Maybe maybe no need. Perhaps perhaps?</think:6124c78e>answer'
    result = detect([{'role': 'assistant', 'content': content}])
    assert result['issue_counts'] == {RULE: 1}
    hit = result['hits'][0]
    evidence = hit['detail']['adjacent_word_stutter']
    assert evidence['min_trajectory_occurrences'] == 3
    assert evidence['detector_version'] == '2.1.1'
    for event in evidence['occurrences']:
        assert content[event['start'] : event['end']].lower() == f'{event["word"]} {event["word"]}'
    assert 'no need' in hit['quotes'][1]['snippet']
    assert 'suggested_penalty' not in hit['detail']
    assert not detect([{'role': 'assistant', 'reasoning_content': 'likely likely check.'}])['issue_counts']


@pytest.mark.parametrize('separator', [',', ', '])
def test_quoted_json_members_do_not_complete_trajectory_threshold(separator):
    members = ['"ok"', '"maybe maybe"', '"need need"', '"perhaps perhaps"']
    text = 'The fixture contains this JSON array: [' + separator.join(members) + ']'
    assert occurrences(text) == []
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


def test_long_line_of_quoted_fragments_scans_in_linear_time():
    # A quadratic scan needs several seconds for this line; a linear one needs milliseconds.
    text = 'Maybe inspect this JSON: [' + ', '.join(f'"field_{index}"' for index in range(8000)) + ']'
    started = time.perf_counter()
    assert occurrences(text) == []
    assert time.perf_counter() - started < 1.0


def test_malformed_formatting_cannot_prevent_three_location_trigger():
    text = 'An orphan `\nMaybe maybe no need.\nfrom estate," perhaps perhaps check. Finish "done".\nNeed need retry.'
    result = detect([{'role': 'assistant', 'reasoning_content': text}])
    assert result['issue_counts'] == {RULE: 1}
    assert result['hits'][0]['detail']['adjacent_word_stutter']['trajectory_occurrences'] == 3
