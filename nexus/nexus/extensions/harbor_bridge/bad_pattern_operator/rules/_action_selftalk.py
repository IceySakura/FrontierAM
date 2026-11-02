"""Thinking-only detection of repeated promises to act instead of acting.

Count complete prose beats, not keyword suffixes or newlines. Substantive
sentences and structured rows stay in the denominator so ordinary planning,
checklists and calculations cannot qualify by having many short lines.
"""

import re

from bisect import bisect_left, bisect_right
from collections import Counter


WINDOW = 800
MIN_BEATS = 10
MIN_ACTIONS = 5
MIN_REPEAT = 4
MIN_SHARE = 0.5
MIN_KINDS = 3
ACTION = re.compile(
    r'^(?:(?:好|好的|好了|OK|okay)[,， ]*)?'
    r'(?:(?:让(?:我|me)|let我|我|现在|直接|开始|先|就|马上|立即|真的|真正|最终|继续|再|立刻|已经|已|停止思考)[,， ]*)*'
    r'(执行|输出|写|完成|跑|调用|生成)(?:工具调用(?:块)?|调用块|代码|文件)?$',
    re.I,
)
FILLER = re.compile(r'^(好|好的|好了|够|够了|现在|停止|完成|OK|okay|now|done|go)$', re.I)
EN_ACTION = re.compile(
    r"^(?:(?:ok(?:ay)?|now)[, ]*)?(?:(?:let me|i will|i'll|i|just|now|really|finally)[ ]+)*"
    r'(execute|emit|write|output|call)(?: (?:it|the (?:tool calls?|code|file)))?$',
    re.I,
)
SEPARATOR = re.compile(r'[。！？!?\n]+|(?<!\d)\.(?!\d)(?=\s|$)')
CORRUPT = re.compile(r'(?<![A-Za-z])(?:[Il\\]*n[Il]+n?|\\+lnl|[Il]+(?:\\n|\n)[Il]+)(?![A-Za-z])')
DECORATION = re.compile(r'^[\s\-—*_]+$')
STRUCTURED = re.compile(r'^(?:\d+[.)、]\s*|[-*+]\s+|[>|])')


def _beats(text):
    """Offset, action/filler key or None, for each complete prose sentence."""
    # Normalization preserves offsets in the original text for evidence.
    normalized = CORRUPT.sub(lambda m: ' ' * len(m[0]), text)
    normalized = re.sub(r'\\+n', lambda m: '\n' + ' ' * (len(m[0]) - 1), normalized)
    rows, pos, fenced = [], 0, False
    for line in normalized.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('```', '~~~')):
            fenced = not fenced
            rows.append((pos, None))
        elif fenced:
            if stripped:
                rows.append((pos, None))
        else:
            start = 0
            for end in [m.end() for m in SEPARATOR.finditer(line)] + [len(line)]:
                value = line[start:end].strip().rstrip('。！？!?.,，;；:：').strip()
                offset = pos + start
                start = end
                if not value or DECORATION.fullmatch(value):
                    continue
                if STRUCTURED.match(stripped):
                    rows.append((offset, None))
                    continue
                if (value.startswith('（') and value.endswith('）')) or (value.startswith('(') and value.endswith(')')):
                    value = value[1:-1].strip()
                key = None
                match = ACTION.fullmatch(value) or EN_ACTION.fullmatch(value)
                if match:
                    key = ('filler' if match[1] == '完成' else 'action', match[1].lower())
                elif filler := FILLER.fullmatch(value):
                    key = ('filler', filler[1].lower())
                rows.append((offset, key))
        pos += len(line) + 1
    return rows


def detect_action_selftalk(text):
    """First dense window of repeated, complete action promises, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    rows = _beats(text)
    matches = [(pos, key) for pos, key in rows if key]
    if len(matches) < MIN_BEATS:
        return None
    positions = [pos for pos, _ in rows]
    hit_positions = [pos for pos, _ in matches]
    for hi, (end, _) in enumerate(matches):
        start = max(0, end - WINDOW)
        lo = bisect_left(hit_positions, start)
        count = hi - lo + 1
        if count < MIN_BEATS:
            continue
        keys = Counter(key for _, key in matches[lo : hi + 1])
        actions = sum(n for (kind, _), n in keys.items() if kind == 'action')
        total = bisect_right(positions, end) - bisect_left(positions, start)
        if (
            actions < MIN_ACTIONS
            or len(keys) < MIN_KINDS
            or max(keys.values()) < MIN_REPEAT
            or count < MIN_SHARE * total
        ):
            continue
        excerpt_end = min(len(text), text.find('\n', end) if '\n' in text[end:] else len(text))
        return {
            'window_chars': WINDOW,
            'start': start,
            'end': excerpt_end,
            'short_beats': count,
            'action_beats': actions,
            'total_beats': total,
            'short_share': count / total,
            'phrases': {f'{kind}:{key}': n for (kind, key), n in keys.items()},
            'excerpt': text[start:excerpt_end],
        }
    return None
