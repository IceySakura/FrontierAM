"""Count context-qualified adjacent word runs in assistant thinking prose."""

import re


DETECTOR_VERSION = '2.1.1'
MIN_OCCURRENCES = 3
WORD_MIN_REPEATS = {
    'maybe': 2,
    'need': 2,
    'perhaps': 2,
    'likely': 2,
    'actual': 2,
    'computed': 2,
    'concrete': 3,
    'current': 3,
    'enabled': 2,
    'expected': 2,
    'explicit': 2,
    'fixed': 2,
    'last': 2,
    'names': 3,
    'near': 3,
    'normalized': 2,
    'potential': 3,
    'recursively': 2,
    'then': 3,
    'use': 3,
}
TOKEN_END = r"(?![\w'’/\\-]|\.(?=\w)|:(?=[:\w])|[\[(])"
KEYWORD_RE = re.compile(r"(?<![\w'’./\\-])(?<!::)(?:" + '|'.join(WORD_MIN_REPEATS) + ')' + TOKEN_END, re.I)
HORIZONTAL_SPACE = r'[^\S\r\n\v\f\x85\u2028\u2029]'
GAP_RE = re.compile(HORIZONTAL_SPACE + '+')
FENCE_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})')
BACKTICK_RE = re.compile(r'(?<!`)(`+)(?!`)')
# In prose `word,"` closes a quote, so a comma blocks an opening quote; after a JSON value
# (string, number, literal, array or object) the comma separates compact members instead.
QUOTE_RE = re.compile(
    r'(?:(?<![\w,.;\\])|(?<=["\]}\d],)|(?<=true,)|(?<=false,)|(?<=null,))"(?:\\.|[^"\\\r\n])*"(?!\w)'
    r'|“[^”\r\n]*”|‘[^’\r\n]*’|(?<![\w\\])\x27[^\x27\r\n]*\x27(?!\w)'
)
MATH_RE = re.compile(r'\$(?:\\.|[^$\\\r\n])+\$|\\\([^\r\n]*?\\\)|\\\[[^\r\n]*?\\\]')
HTML_CODE_RE = re.compile(r'<(pre|code|script|style)\b[^>\r\n]*>[^\r\n]*?</\1\s*>', re.I)
HTML_OPEN_RE = re.compile(r'^ {0,3}<(pre|code|script|style)(?:\s+[^>\r\n]*)?>\s*$', re.I)
INDENTED_CODE_RE = re.compile(
    r'^(?: {4}|\t)\s*(?:def |class |import |from \w+ import |return |'
    r'(?:const|let|var) |[\w.]+\s*(?:=(?!=)|:=)|[{}])'
)
META_PREFIX_RE = re.compile(
    r'\b(?:literal|string|token|keyword|bigram|pattern|sequence|words?|phrase|example)'
    r'\s*(?:is|was|:|=)\s*$',
    re.I,
)
META_SUFFIX_RE = re.compile(
    r'^\s+(?:is|are)\s+(?:an?\s+)?(?:stutter|duplicate|repeated word|repetition example)\b', re.I
)
NAMED_ITEM_PREFIX = re.compile(
    r'\b(?:field|key|column|attribute|variable|property|parameter|argument|method|function|'
    r'type|class|module|package|label|constant|keyword|template)\s+'
    r'(?:(?:named|called|is)\s+)?$',
    re.I,
)
MAYBE_TYPE_RE = re.compile(r'\s+(?:Int|Integer|Bool|Char|String|Text|Float|Double)\b')
PARTICIPLES = {'computed', 'enabled', 'expected', 'fixed', 'normalized'}
MODIFIER_PREFIX = re.compile(
    r'\b(?:be|been|being|is|are|was|were|the|a|an|any|with|without|of|for|from|into|'
    r'compare|comparing|return|returns|returned|using|check|checking|consider|considering|'
    r'contain|contains|containing|exact|internal|external|fully)\s+$',
    re.I,
)
NORMAL_AFTER = {
    'last': re.compile(r'^\s+(?:week|month|year|time|night|summer|winter|rites|respects|calls?|stand)\b', re.I),
}


def _normal_two_word_usage(text, start, end, word):
    before, after = text[max(0, start - 100) : start], text[end : end + 100]
    tokens = text[start:end].split()
    if word == 'maybe':
        if 'Maybe' in tokens and re.search(r'\b(?:flatten|nested(?: type)?|root|list of)\s+$', before, re.I):
            return True
        if tokens == ['Maybe', 'Maybe'] and re.match(r':\s+both\s+present\b', after):
            return True
        return tokens[-1] == 'Maybe' and (
            NAMED_ITEM_PREFIX.search(before)
            or MAYBE_TYPE_RE.match(after)
            or (tokens[0].islower() and re.match(r'\s+(?:[a-z]|[0-9])\b', after))
        )
    if NAMED_ITEM_PREFIX.search(before):
        return True
    if word == 'need':
        if re.match(r'\s*[+*=<>-]\s*(?:\d|\w+\[)', after):
            return True
        if tokens[-1][0].isupper() and re.match(r'\s+(?:as|when|state|status|constructor|type)\b', after, re.I):
            return True
        if re.search(r'\bin\s+$', before, re.I):
            return True
        if re.search(r'\b(?:the|these|those|all|any)\s+\w+(?:\s+\w+)?\s+(?:that|which|who)\s+\w+\s+$', before, re.I):
            return True
        if re.search(r'\b(?:the|those|these|any|all)\s+(?:\w+\s+){0,3}(?:we|you|they|i)\s+$', before, re.I):
            return True
        return bool(
            re.search(r'\b(?:the|a|this|that|our|your|their|his|her|its|my|any)\s+$', before, re.I)
            and re.match(r'\s+(?:not|never|be|only|also)\b', after, re.I)
        )
    if word == 'likely':
        return bool(re.match(r'\s+subtags?' + TOKEN_END, after, re.I))
    if word in {'perhaps', 'explicit', 'recursively'}:
        return False
    if word in NORMAL_AFTER and NORMAL_AFTER[word].match(after):
        return True
    if word in PARTICIPLES:
        last_clause = re.split(r'[.!?;\n]', before)[-1]
        if re.search(r'\b(?:who|which|that)\s+(?:(?:was|were|is|are|had|been|being)\s+){1,3}$', last_clause, re.I):
            return True
        if not MODIFIER_PREFIX.search(before) and not re.match(r'\s+(?:from|by|using)\b', after, re.I):
            return True
    return bool(word == 'actual' and re.search(r'\b(?:expected/actual|actual/expected)\??\s*$', before, re.I))


def _line_ranges(line, offset):
    """Balanced same-line delimiters own their contents; orphan marks own nothing."""
    ranges = []
    cursor = 0
    # None of these patterns is anchored, so a match found from an earlier cursor is still the
    # leftmost one while it starts at or after the cursor. Searching again only once a cached
    # match falls behind keeps long lines of quoted fragments linear instead of quadratic.
    pending = {pattern: pattern.search(line) for pattern in (BACKTICK_RE, QUOTE_RE, MATH_RE, HTML_CODE_RE)}
    closings = {}
    while cursor < len(line):
        for pattern, cached in pending.items():
            if cached is not None and cached.start() < cursor:
                pending[pattern] = pattern.search(line, cursor)
        candidates = [match for match in pending.values() if match is not None]
        if not candidates:
            break
        match = min(candidates, key=lambda item: item.start())
        if match.re is BACKTICK_RE:
            slashes, previous = 0, match.start() - 1
            while previous >= 0 and line[previous] == '\\':
                slashes, previous = slashes + 1, previous - 1
            if slashes % 2:
                cursor = match.end()
                continue
            closing = closings.get(match[0])
            if match[0] not in closings or (closing is not None and closing.start() < match.end()):
                closing = re.compile(r'(?<!`)' + re.escape(match[0]) + r'(?!`)').search(line, match.end())
                closings[match[0]] = closing
            if closing is None:
                cursor = match.end()
                continue
            end = closing.end()
        else:
            end = match.end()
        ranges.append((offset + match.start(), offset + end))
        cursor = end
    return ranges


def _excluded_ranges(text):
    ranges = []
    position = 0
    fence = None
    html = None
    for line in text.splitlines(keepends=True):
        visible = re.sub(r'^(?:</?think:[0-9a-f]+>)+', '', line)
        marker = FENCE_RE.match(visible)
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= fence[1]
                and not re.sub(r'</?think:[0-9a-f]+>', '', visible[marker.end() :]).strip()
            ):
                ranges.append((fence[2], position + len(line)))
                fence = None
        elif html is not None:
            if re.fullmatch(r' {0,3}</' + html[0] + r'\s*>\s*', visible, re.I):
                ranges.append((html[1], position + len(line)))
                html = None
            elif not visible.strip():
                # A broken HTML wrapper must not swallow the next prose paragraph.
                html = None
        elif marker:
            fence = (marker[1][0], len(marker[1]), position)
        elif (opening := HTML_OPEN_RE.fullmatch(visible)) is not None:
            html = (opening[1], position)
        elif re.match(r'^ {0,3}>', visible) or INDENTED_CODE_RE.match(visible):
            ranges.append((position, position + len(line)))
        elif KEYWORD_RE.search(line):
            ranges.extend(_line_ranges(line, position))
        position += len(line)
    if fence is not None:
        ranges.append((fence[2], len(text)))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def occurrences(text: str) -> list[dict]:
    """Return maximal runs, with character offsets in the unmodified input."""
    if not isinstance(text, str):
        raise TypeError('occurrences expects a string')
    excluded = _excluded_ranges(text)
    words = []
    range_index = 0
    for match in KEYWORD_RE.finditer(text):
        while range_index < len(excluded) and excluded[range_index][1] <= match.start():
            range_index += 1
        blocked = range_index < len(excluded) and excluded[range_index][0] < match.end()
        words.append((match, blocked))
    events = []
    index = 0
    while index < len(words):
        first, blocked = words[index]
        word = first[0].lower()
        end_index = index + 1
        if not blocked:
            while end_index < len(words):
                previous = words[end_index - 1][0]
                following, following_blocked = words[end_index]
                if following_blocked or following[0].lower() != word:
                    break
                if GAP_RE.fullmatch(text[previous.end() : following.start()]) is None:
                    break
                end_index += 1
        count = end_index - index
        minimum = WORD_MIN_REPEATS.get(word)
        if minimum is not None and count >= minimum:
            start, end = first.start(), words[end_index - 1][0].end()
            meta = META_PREFIX_RE.search(text[max(0, start - 80) : start])
            meta = meta or META_SUFFIX_RE.search(text[end : end + 90])
            normal = count == 2 and _normal_two_word_usage(text, start, end, word)
            if not meta and not normal:
                events.append({'word': word, 'start': start, 'end': end, 'repeats': count})
        index = end_index
    return events
