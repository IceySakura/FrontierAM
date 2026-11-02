"""Which natural language a run writes in, and whether a message is in it.

The two directions are not symmetric. A language with a marker is recognised by that
marker being there; a language without one is recognised by every other marker being
absent. The cases here cover both, and the code-stripping that keeps identifiers out
of the judgement.
"""

import pytest

from nexus.extensions.tasks.hitl.user_agent.language import (
    CHINESE,
    DEFAULT_LANGUAGE,
    ENGLISH,
    LANGUAGES,
    NaturalLanguage,
    detect,
    prose,
)


# -- detect --------------------------------------------------------------


@pytest.mark.parametrize(
    ('text', 'wanted'),
    [
        ('Add a retry around the upload and keep the timeout where it is.', ENGLISH),
        ('给上传加个重试，超时那块先不动。', CHINESE),
        ('', DEFAULT_LANGUAGE),
        ('   \n\n  ', DEFAULT_LANGUAGE),
        ('12345 !!! ---', DEFAULT_LANGUAGE),
    ],
)
def test_a_text_is_read_as_the_language_it_is_written_in(text: str, wanted: NaturalLanguage) -> None:
    assert detect(text) is wanted


def test_a_handful_of_borrowed_characters_does_not_carry_the_whole_text() -> None:
    """One dataset task runs to thousands of English letters with a dozen Han in it."""
    text = 'Fix the exporter. ' * 40 + 'The column is labelled 导出时间 in the fixture.'

    assert detect(text) is ENGLISH


def test_identifiers_in_a_message_do_not_make_it_the_language_they_are_written_in() -> None:
    zh = '把 `RetryPolicy.max_attempts` 调小，别动 `timeout_seconds`。'

    assert detect(zh) is CHINESE


def test_a_fenced_block_is_not_read_as_prose() -> None:
    text = '这里改一下\n\n```python\ndef handle(request, timeout, retries, backoff):\n    return None\n```'

    assert detect(text) is CHINESE


def test_prose_drops_fences_and_spans_rather_than_the_text_around_them() -> None:
    body = prose('keep this `drop_me` and this\n```\ndrop me too\n```\nkeep this as well')

    assert 'keep this' in body and 'keep this as well' in body
    assert 'drop_me' not in body and 'drop me too' not in body


# -- accepts -------------------------------------------------------------


@pytest.mark.parametrize(
    'message',
    [
        'The pagination is off on the second page.',
        'Roll back that deletion, it was not in the request.',
        'Use `paddedSize(includeRemoved=false)` instead.',
    ],
)
def test_an_english_message_may_go_out_on_an_english_run(message: str) -> None:
    assert ENGLISH.accepts(message) is True


@pytest.mark.parametrize(
    'message',
    [
        '分页不对',
        '继续',
        '合并接口还没按要求，`paddedSize` 也要接 includeRemoved',
    ],
)
def test_a_message_carrying_han_is_refused_on_an_english_run(message: str) -> None:
    assert ENGLISH.accepts(message) is False


def test_a_two_character_message_is_caught_like_any_other() -> None:
    """The shortest real violation measured was a bare 继续."""
    assert ENGLISH.accepts('继续') is False


def test_han_inside_code_does_not_refuse_an_english_message() -> None:
    message = 'The fixture label is wrong. `LABEL = "导出时间"` should be the English one.'

    assert ENGLISH.accepts(message) is True


def test_han_inside_a_fence_does_not_refuse_an_english_message() -> None:
    message = 'This line is the problem:\n\n```python\nLABEL = "导出时间"\n```\n\nUse the English label.'

    assert ENGLISH.accepts(message) is True


def test_a_chinese_message_may_go_out_on_a_chinese_run() -> None:
    assert CHINESE.accepts('分页不对，第二页的条数和第一页对不上') is True


def test_english_identifiers_do_not_refuse_a_chinese_message() -> None:
    message = '把 updateAncestorsSize 改成接收显式 delta 和 includeRemoved，paddedSize 默认 false'

    assert CHINESE.accepts(message) is True


def test_a_message_with_no_chinese_at_all_is_refused_on_a_chinese_run() -> None:
    assert CHINESE.accepts('The pagination is off on the second page.') is False


# -- the table -----------------------------------------------------------


def test_every_language_is_named_and_coded_exactly_once() -> None:
    codes = [language.code for language in LANGUAGES]

    assert len(set(codes)) == len(codes)
    for language in LANGUAGES:
        assert isinstance(language, NaturalLanguage)
        assert language.code.strip() and language.name.strip()


def test_the_default_is_a_language_the_table_declares() -> None:
    assert DEFAULT_LANGUAGE in LANGUAGES


def test_a_language_with_no_marker_claims_no_share_of_anything() -> None:
    """Otherwise the Latin alphabet would outvote every marker in a mixed text."""
    assert ENGLISH.share('anything at all 任何东西') == 0.0
