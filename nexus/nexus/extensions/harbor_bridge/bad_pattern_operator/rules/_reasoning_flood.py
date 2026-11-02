"""Detection logic from HyX rule/checkpoint_reasoning_flood@1.

Keep thresholds and judgments aligned with reasoning-flood-20261016.
This private module is not discovered as a standalone Nexus rule.
"""

import re

from bisect import bisect_left
from collections import Counter


WINDOW = 200
WINDOW_MIN = 28
WINDOW_MIN_LINES = 8
TEMPLATE_MIN = 50
TEMPLATE_MIN_SHARE = 0.08
FILLER = re.compile(
    r"^(?:h+m+|um+|uh+|ok(?:ay)?|wait|alright|well|go|actually|let me|let's|i'll|what about|consider|no(?!\.?\s*\d))\b.{0,55}$",
    re.I,
)
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^'\n]*'|`[^`\n]*`|\*\*[^*\n]*\*\*")
NUMBER = re.compile(r'\d+')
LEAD = re.compile(r'^(?:[#>*\-•]+|\d+[.)])\s*')
TRAIL = re.compile(r'[\s.,;:!?。，；：！？…]+$')
CODE_LINE = re.compile(r'[{}();=<>\[\]\\|]|^\s*[+\-@]|^\s*at\s|\.\w{1,4}:\d|[a-z][A-Z]|\w_\w|^\s*```')
CODE_WORDS = {'end', 'else', 'fi', 'done', 'return', 'break', 'continue', 'pass', 'then', 'do', 'esac', 'endif'}


def prose_lines(text):
    """Non-empty lines outside fenced code blocks, with their character offsets."""
    out, pos, fenced = [], 0, False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced and line.strip():
            out.append((pos, line))
        pos += len(line) + 1
    return out


def filler_window(text, lines):
    marks = [pos + len(line) - len(line.lstrip()) for pos, line in lines if FILLER.match(line.strip())]
    if len(marks) < WINDOW_MIN_LINES:
        return None
    breaks = [i for i, c in enumerate(text) if c == '\n']
    best = None
    for start in marks:
        end = start + WINDOW
        n_lines = bisect_left(marks, end) - bisect_left(marks, start)
        n_breaks = bisect_left(breaks, end) - bisect_left(breaks, start)
        if (
            n_lines >= WINDOW_MIN_LINES
            and n_lines + n_breaks >= WINDOW_MIN
            and (best is None or n_lines + n_breaks > best[0])
        ):
            best = (n_lines + n_breaks, n_lines, start)
    return best


def template_key(line):
    """Pattern of a natural-language line with quoted text and numbers masked; None for code, data rows and logs."""
    s = QUOTED.sub('\u00a4', LEAD.sub('', line.strip()))
    if CODE_LINE.search(s):
        return None
    masked = NUMBER.sub('\u00a7', s)
    if masked.count('\u00a7') >= 2 or re.search(r'[A-Za-z]\u00a7|\u00a7[A-Za-z]', masked):
        return None
    s = TRAIL.sub('', masked.lower()).strip().replace('\u00a4', 'Q').replace('\u00a7', 'N')
    if not re.search(r'[a-z\u4e00-\u9fff]{2}', s.replace('Q', '').replace('N', '')) or s in CODE_WORDS or len(s) > 120:
        return None
    if s.startswith('N') or re.fullmatch(r'[a-z]+(?: [a-z]+)?(?: N)?: (?:Q|N)|[a-z]+ N', s):
        return None
    if re.match(r'[a-z]+ N:', s) or len(re.findall(r'\b[a-z_]+: ', s)) >= 2:
        return None
    return s


def template_loop(lines):
    keys = [(template_key(line), line) for _, line in lines]
    counts = Counter(k for k, _ in keys if k)
    if not counts:
        return None
    key, n = counts.most_common(1)[0]
    if n < TEMPLATE_MIN or n / len(lines) < TEMPLATE_MIN_SHARE:
        return None
    first = next(line for k, line in keys if k == key)
    return key, first.strip(), n, len(lines)


def detect_thinking(text):
    """Result for one turn of thinking, or None when it does not hit. Filler flood takes precedence for the excerpt."""
    if not isinstance(text, str) or not text.strip():
        return None
    lines = prose_lines(text)
    if not lines:
        return None
    window = filler_window(text, lines)
    loop = template_loop(lines)
    if not window and not loop:
        return None
    if window:
        total, n_lines, start = window
        excerpt = text[start : start + WINDOW]
        detail = f'{WINDOW} 个字符内有 {n_lines} 行以 Hmm、OK、Let me 等开头的短句，短句数加换行数为 {total}。'
        kind = 'filler_flood'
    else:
        key, first, n, total_lines = loop
        excerpt = first[:300]
        detail = f'同一句式“{first[:120]}”（引号内容与数字不计）出现 {n} 次，占 {total_lines} 个非空行的 {n / total_lines:.0%}。'
        kind = 'template_loop'
    return {
        'kind': kind,
        'detail': detail,
        'excerpt': excerpt,
        'thinking_chars': len(text),
        'filler_window': {'total': window[0], 'lines': window[1]} if window else None,
        'template': {'pattern': loop[0][:200], 'count': loop[2], 'lines': loop[3]} if loop else None,
    }
