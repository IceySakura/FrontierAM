"""Rule: reasoning_repetition_loop.

Detect within-turn generation collapse in assistant reasoning/content. The
checks deliberately ignore tool-call arguments, where repeated strings are
often legitimate file contents or command data.
"""

from __future__ import annotations

import re

from collections import Counter
from collections.abc import Callable
from typing import TypedDict

from .base import Rule
from ..models import Finding


class _Params(TypedDict):
    min_run_chars: int
    single_char_min_run: int
    separator_min_run: int
    phrase_min_repeats: int
    window_tokens: int
    low_variety_stride: int
    low_variety_max_distinct: int
    low_variety_min_top_word: int
    dominant_stride: int
    dense_min_windows: int
    dominant_min_count: int
    dominant_max_words: int
    dominant_max_code_share: float
    filler_min_share: float
    filler_min_words: int
    filler_min_dashes: int
    line_min_chars: int
    line_min_repeats: int
    whitespace_min_run: int
    whitespace_block_chars: int
    excerpt_before: int
    excerpt_after: int


_PARAMS: _Params = {
    'min_run_chars': 200,
    'single_char_min_run': 1000,
    'separator_min_run': 300,
    'phrase_min_repeats': 6,
    'window_tokens': 60,
    'low_variety_stride': 30,
    'low_variety_max_distinct': 6,
    'low_variety_min_top_word': 12,
    'dominant_stride': 20,
    'dense_min_windows': 2,
    'dominant_min_count': 20,
    'dominant_max_words': 12,
    'dominant_max_code_share': 0.10,
    'filler_min_share': 0.6,
    'filler_min_words': 20,
    'filler_min_dashes': 30,
    'line_min_chars': 8,
    'line_min_repeats': 6,
    'whitespace_min_run': 200,
    'whitespace_block_chars': 300,
    'excerpt_before': 100,
    'excerpt_after': 220,
}

_UNIT_RE = re.compile(r'(.{1,8}?)\1{9,}', re.S)
_PHRASE_RE = re.compile(r'(.{9,40}?)\1{5,}', re.S)
_TOKEN_RE = re.compile(r'\w+|[^\w\s]+', re.U)
_WORDISH_RE = re.compile(r'[A-Za-z\u4e00-\u9fff]')
_LINE_RE = re.compile(r'[^\n]+')
_CODE_LINE_END_RE = re.compile(r'[,;{}()\[\]<>]$')
_SPACE_RUN_RE = re.compile(rf'[ \t]{{{_PARAMS["whitespace_min_run"]},}}')
_BLANK_BLOCK_RE = re.compile(rf'[ \t\n]{{{_PARAMS["whitespace_block_chars"]},}}')
_SEPARATOR = set('-=_*#~.·─━│┃═|+')
_WHITESPACE = set(' \t\n')
_VISUAL_WHITESPACE_MARKERS = set('⏎␣')
_CODE_PUNCT = set('{}()[]<>=/\\;_|@$&%^~#-')
_FILLER_SYMBOLS = set('—–…!?*"\':.-')
_DATA_WORDS = {
    'true',
    'false',
    'none',
    'null',
    'nil',
    'undefined',
    'nan',
    'na',
    'passed',
    'failed',
    'pass',
    'fail',
    'yes',
    'no',
}
_FILLER_WORDS = {
    'hmm',
    'hm',
    'hmmm',
    'wait',
    'ok',
    'okay',
    'maybe',
    'so',
    'well',
    'fine',
    'go',
    'done',
    'stop',
    'end',
    'no',
    'yes',
    'right',
    'grr',
    'grrr',
    'ugh',
    'sigh',
    'argh',
}


def _is_data(unit: str) -> bool:
    if re.search(r'\d', unit) and len(re.findall(r'[A-Za-z]', unit)) <= 2 and not re.search(r'[\u4e00-\u9fff]', unit):
        return True
    words = re.findall(r'[A-Za-z]+', unit)
    return (
        bool(words) and all(word.lower() in _DATA_WORDS for word in words) and not re.search(r'[\u4e00-\u9fff]', unit)
    )


def _looks_like_code(unit: str) -> bool:
    chars = [char for char in unit if not char.isspace()]
    return bool(chars) and sum(char in '(){}[]=;<>' for char in chars) / len(chars) >= 0.2


def _data_heavy(window: list[str]) -> bool:
    return sum(1 for token in window if re.search(r'\d', token)) / len(window) >= 0.2


def _raw_finding(check: str, unit: str, start: int, end: int) -> dict:
    return {'check': check, 'unit': unit, 'start': start, 'run_chars': end - start}


def _run_findings(text: str, regex: re.Pattern, check: str, min_repeats: int = 0) -> list[dict]:
    out = []
    for match in regex.finditer(text):
        unit = match.group(1)
        run = len(match.group(0))
        chars = set(unit)
        if (
            chars <= _WHITESPACE
            or set(unit.strip()) <= _VISUAL_WHITESPACE_MARKERS
            or _is_data(unit)
            or _looks_like_code(unit)
        ):
            continue
        if min_repeats and run // len(unit) < min_repeats:
            continue
        stripped = set(unit.strip())
        if chars <= _SEPARATOR | _WHITESPACE:
            need = _PARAMS['separator_min_run'] if '|' not in unit else None
        elif len(stripped) == 1 and next(iter(stripped)).isalnum():
            need = _PARAMS['single_char_min_run']
        else:
            need = _PARAMS['min_run_chars']
        if need and run >= need:
            out.append(_raw_finding(check, unit, match.start(), match.end()))
    return out


def _tokenize(text: str) -> list[tuple[str, int]]:
    return [(match.group(0), match.start()) for match in _TOKEN_RE.finditer(text)]


def _low_variety_findings(text: str) -> list[dict]:
    tokens = _tokenize(text)
    size = _PARAMS['window_tokens']
    out = []
    skip_to = -1
    for i in range(0, max(0, len(tokens) - size + 1), _PARAMS['low_variety_stride']):
        if i < skip_to:
            continue
        window = [token for token, _ in tokens[i : i + size]]
        if _data_heavy(window):
            continue
        normalized = [token.lower() for token in window]
        distinct = set(normalized)
        words = [token for token in normalized if len(token) >= 2 and _WORDISH_RE.search(token) and not _is_data(token)]
        top = max((words.count(word) for word in set(words)), default=0)
        if len(distinct) > _PARAMS['low_variety_max_distinct'] or top < _PARAMS['low_variety_min_top_word']:
            continue
        j = i + size
        while j < len(tokens) and tokens[j][0].lower() in distinct:
            j += 1
        end = tokens[j - 1][1] + len(tokens[j - 1][0])
        out.append(_raw_finding('low_variety', ' '.join(sorted(distinct)), tokens[i][1], end))
        skip_to = j
    return out


def _back_to_back_windows(
    text: str,
    check: str,
    qualify: Callable[[list[str]], str | None],
) -> list[dict]:
    tokens = _tokenize(text)
    size = _PARAMS['window_tokens']
    need = _PARAMS['dense_min_windows']
    cache: dict[int, str | None] = {}

    def unit(index: int) -> str | None:
        if index not in cache:
            cache[index] = qualify([token for token, _ in tokens[index : index + size]])
        return cache[index]

    out = []
    i = 0
    while i + size * need <= len(tokens):
        units = [unit(i + offset * size) for offset in range(need)]
        # Adjacent windows must collapse around the same dominant unit. Merely
        # transitioning from one repeated subject to another is not one loop.
        if all(units) and len(set(units)) == 1:
            j = i + size * need
            out.append(_raw_finding(check, str(units[0]), tokens[i][1], tokens[j - 1][1] + len(tokens[j - 1][0])))
            i = j
        else:
            i += _PARAMS['dominant_stride']
    return out


def _dominant_word(window: list[str]) -> str | None:
    code = sum(1 for token in window if not _WORDISH_RE.search(token) and set(token) & _CODE_PUNCT)
    if code / len(window) > _PARAMS['dominant_max_code_share'] or _data_heavy(window):
        return None
    words = Counter(
        token.lower() for token in window if len(token) >= 2 and _WORDISH_RE.search(token) and not _is_data(token)
    )
    if not words:
        return None
    word, count = words.most_common(1)[0]
    if (
        count >= _PARAMS['dominant_min_count']
        and len(words) <= _PARAMS['dominant_max_words']
        and len(word) >= 3
        and word not in _DATA_WORDS
    ):
        return word
    return None


def _is_filler_symbol(token: str) -> bool:
    return not _WORDISH_RE.search(token) and set(token) <= _FILLER_SYMBOLS


def _filler(window: list[str]) -> str | None:
    words = sum(1 for token in window if token.lower() in _FILLER_WORDS)
    symbols = sum(1 for token in window if _is_filler_symbol(token))
    dashes = sum(token.count('—') for token in window)
    if (words + symbols) / len(window) >= _PARAMS['filler_min_share'] and (
        words >= _PARAMS['filler_min_words'] or dashes >= _PARAMS['filler_min_dashes']
    ):
        return 'hmm / — / wait …'
    return None


def _line_loop_findings(text: str) -> list[dict]:
    lines = [(match.group(0).strip(), match.start()) for match in _LINE_RE.finditer(text)]
    out = []
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1][0] == lines[i][0]:
            j += 1
        line = lines[i][0]
        if (
            j - i + 1 >= _PARAMS['line_min_repeats']
            and len(line) >= _PARAMS['line_min_chars']
            and _WORDISH_RE.search(line)
            and not _CODE_LINE_END_RE.search(line)
        ):
            out.append(_raw_finding('line_loop', line[:80], lines[i][1], lines[j][1] + len(lines[j][0])))
        i = j + 1
    return out


def _whitespace_findings(text: str) -> list[dict]:
    out = [
        _raw_finding('whitespace_run', '<spaces/tabs>', match.start(), match.end())
        for match in _SPACE_RUN_RE.finditer(text)
    ]
    if out:
        return out
    for match in _BLANK_BLOCK_RE.finditer(text):
        if len(re.findall(r'[ \t]', match.group(0))) >= _PARAMS['whitespace_min_run']:
            out.append(_raw_finding('whitespace_run', '<blank lines>', match.start(), match.end()))
    return out


def _scan_text(text: str) -> list[dict]:
    if len(text) < _PARAMS['min_run_chars']:
        return []
    findings = _run_findings(text, _UNIT_RE, 'unit_run')
    findings += _run_findings(text, _PHRASE_RE, 'phrase_run', _PARAMS['phrase_min_repeats'])
    findings += _low_variety_findings(text)
    findings += _back_to_back_windows(text, 'dominant_word', _dominant_word)
    findings += _back_to_back_windows(text, 'filler_window', _filler)
    findings += _line_loop_findings(text)
    findings += _whitespace_findings(text)
    return findings


def _excerpt(text: str, start: int, end: int) -> str:
    left = max(0, start - _PARAMS['excerpt_before'])
    right = min(len(text), end, start + _PARAMS['excerpt_after'])
    return text[left:right]


def scan(ctx):
    out = []
    for message_index, _message in ctx.assistant_messages():
        for channel, text in (
            ('reasoning', ctx.reasoning(message_index)),
            ('content', ctx.content(message_index)),
        ):
            raw_findings = _scan_text(text)
            if not raw_findings:
                continue
            checks = list(dict.fromkeys(item['check'] for item in raw_findings))
            primary = max(raw_findings, key=lambda item: (item['run_chars'], -item['start']))
            start = int(primary['start'])
            end = start + int(primary['run_chars'])
            out.append(
                Finding(
                    message_index,
                    f'assistant {channel} 出现生成失控的重复循环: {", ".join(checks)}',
                    ['reasoning_content' if channel == 'reasoning' else 'content'],
                    _excerpt(text, start, end),
                    detail={
                        'channel': channel,
                        'checks': checks,
                        'raw_finding_count': len(raw_findings),
                        'primary_unit': primary['unit'],
                        'primary_start': start,
                        'primary_run_chars': primary['run_chars'],
                    },
                )
            )
    return out


RULE = Rule('reasoning_repetition_loop', '思维链或正文重复刷屏', scan)
