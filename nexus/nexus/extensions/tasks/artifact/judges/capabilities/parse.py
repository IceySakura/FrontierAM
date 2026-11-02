"""Parse capability — extract scores/verdicts from LLM & agent output.

Pure functions, no I/O.
  * ``<judge>...</judge>`` block scoring (:func:`parse_judge_xml` /
    :func:`sum_tag_scores`).
  * :func:`parse_verdicts_xml` — pass/fail verdict tag extractor used by
    the rubric-agent judge. Preferred over JSON for LLM outputs (higher
    parse success rate empirically).
  * :func:`loads_json_lenient` — the shared "strip fence + slice ``{...}`` +
    ``json.loads``" primitive. Judges that still emit JSON verdicts
    (visual_anchored, svg) reuse it; only the field extraction (``winner`` /
    ``structure``+``aesthetics``) stays judge-specific.
  * :func:`coerce_score` — validate a model-supplied number BEFORE clamping.
"""

from __future__ import annotations

import json
import math
import re

from typing import Any


def coerce_score(value: Any, *, low: float, high: float) -> float | None:
    """Return a finite score within [low, high], or None for invalid input.

    Reject booleans and out-of-range values rather than clamping.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if not low <= parsed <= high:
        return None
    return parsed


_JUDGE_BLOCK_RE = re.compile(r'<judge>(.*?)</judge>', re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```', re.IGNORECASE)
# 逗号后紧跟收尾括号 —— LLM 常见的尾逗号，json 模块一律拒收。
_TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
_VERDICT_TAG_RE = re.compile(
    r'<verdict\s+index\s*=\s*"(?P<index>\d+)"\s+result\s*=\s*"(?P<value>yes|no)"\s*>'
    r'(?P<note>.*?)</verdict>',
    re.DOTALL | re.IGNORECASE,
)


_JSON_ESCAPES = '"\\/bfnrt'


def _repair_escapes(s: str) -> str:
    """Repair invalid JSON escapes while preserving already valid pairs.

    Scan escape pairs rather than matching individual backslashes with a regex.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i] != '\\':
            out.append(s[i])
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ''
        if nxt in _JSON_ESCAPES:
            out.append(s[i : i + 2])
            i += 2
        elif nxt == 'u' and re.fullmatch(r'[0-9a-fA-F]{4}', s[i + 2 : i + 6]):
            out.append(s[i : i + 6])
            i += 6
        else:
            out.append('\\\\')
            i += 1
    return ''.join(out)


def loads_json_lenient(text: str, *, allow_raw_newlines: bool = False) -> tuple[Any | None, str | None]:
    """Best-effort JSON extraction from an LLM/agent reply.

    Tolerates a ```json fenced block and surrounding prose: strips the fence,
    then falls back to the outermost ``{...}`` slice before ``json.loads``.
    A final retry repairs illegal backslash escapes (see :func:`_repair_escapes`)
    and removes trailing commas.
    Returns a discriminated result ``(obj, None)`` on success / ``(None, error)``
    on failure, so callers branch on it instead of wrapping in try/except.

    ``allow_raw_newlines`` opts into ``strict=False``, accepting literal newlines inside
    strings. It defaults off because ``code`` fields get exec'd in the sandbox, where
    rebuilding an ambiguous string is worse than dropping the row (see
    ``tests/unit/extensions/tasks/test_artifact_parse.py``). Pass it only when the payload
    is pure scoring data.
    """
    s = (text or '').strip()
    if not s:
        return None, 'empty output'
    fenced = _JSON_FENCE_RE.search(s)
    if fenced:
        s = fenced.group(1).strip()
    if not s.startswith('{'):
        start, end = s.find('{'), s.rfind('}')
        if start >= 0 and end > start:
            s = s[start : end + 1]
    try:
        return json.loads(s, strict=not allow_raw_newlines), None
    except json.JSONDecodeError as e:
        # 只在已经失败之后才修 —— 对本来能解析的输入零影响。
        try:
            repaired = _TRAILING_COMMA_RE.sub(r'\1', _repair_escapes(s))
            return json.loads(repaired, strict=not allow_raw_newlines), None
        except json.JSONDecodeError:
            return None, f'JSON parse failed: {e}'


def parse_verdicts_xml(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Extract ``<verdict index="N" result="yes|no">note</verdict>`` tags.

    The XML tag protocol is markedly more tolerant than JSON — surrounding
    prose, missing indices, unclosed fences and mixed quoting all pass
    silently. Only each tag itself needs to be well-formed. Returns
    ``(verdicts, error)``; ``verdicts`` is
    ``[{'index': int, 'verdict': str, 'note': str}, ...]`` in first-seen order.
    Duplicate indices keep the last occurrence. ``note`` is stripped of
    surrounding whitespace and may be empty.
    """
    s = text or ''
    if not s.strip():
        return [], 'empty output'
    matches = list(_VERDICT_TAG_RE.finditer(s))
    if not matches:
        return [], 'no <verdict> tags found'
    seen: dict[int, dict[str, Any]] = {}
    for m in matches:
        try:
            idx = int(m.group('index'))
        except (TypeError, ValueError):
            continue
        seen[idx] = {
            'index': idx,
            'verdict': m.group('value').lower(),
            'note': (m.group('note') or '').strip(),
        }
    return list(seen.values()), None


def _extract_tag_number(judge_body: str, tag: str) -> float | None:
    """Return the numeric content of ``<tag>X</tag>`` from *judge_body*, or None if missing/invalid."""
    m = re.search(rf'<{tag}>\s*(-?\d+(?:\.\d+)?)\s*</{tag}>', judge_body, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_judge_xml(text: str, tags: tuple[str, ...]) -> dict[str, float]:
    """Parse every requested tag inside the ``<judge>...</judge>`` block.

    Missing tags are reported as ``0.0``. The returned dict always contains every
    key in *tags* so downstream callers can sum/compose without guarding.
    """
    block = _JUDGE_BLOCK_RE.search(text)
    if not block:
        return dict.fromkeys(tags, 0.0)

    body = block.group(1)
    return {t: (_extract_tag_number(body, t) or 0.0) for t in tags}


def sum_tag_scores(
    text: str,
    tags: tuple[str, ...],
    score_max: float,
    weights: tuple[float, ...] | None = None,
) -> tuple[float, str]:
    """Sum multiple ``<tag>X</tag>`` values inside a ``<judge>...</judge>`` block.

    Each tag is parsed, optionally weighted, summed, and normalised to
    ``[0, 1]`` by dividing by *score_max*. Missing tags contribute 0.

    Args:
      text: raw LLM output that may contain a ``<judge>...</judge>`` block.
      tags: tags to sum, e.g. ``('craft', 'design')`` or
        ``('implementation', 'engineering')``.
      score_max: maximum total (weighted) used for normalisation.
      weights: per-tag multiplier. ``None`` = all 1.0. Must match ``len(tags)``.

    Returns ``(normalised_score, evidence_str)``. When the ``<judge>`` block
    is missing, returns ``(0.0, 'judge_block_not_found')``.
    """
    if weights is None:
        weights = tuple(1.0 for _ in tags)
    if len(weights) != len(tags):
        raise ValueError(f'weights length {len(weights)} != tags length {len(tags)}')

    block = _JUDGE_BLOCK_RE.search(text)
    if not block:
        return 0.0, 'judge_block_not_found'

    body = block.group(1)
    parts: list[str] = []
    total = 0.0
    for tag, w in zip(tags, weights, strict=True):
        raw = _extract_tag_number(body, tag)
        val = 0.0 if raw is None else raw
        total += val * w
        if w == 1.0:
            parts.append(f'{tag}={val}')
        else:
            parts.append(f'{tag}={val}*{w}')

    normalised = min(total / score_max, 1.0) if score_max > 0 else 0.0
    evidence = f'{"+".join(parts)}={total}/{score_max}'
    return normalised, evidence


__all__ = [
    'loads_json_lenient',
    'parse_judge_xml',
    'parse_verdicts_xml',
    'sum_tag_scores',
]
