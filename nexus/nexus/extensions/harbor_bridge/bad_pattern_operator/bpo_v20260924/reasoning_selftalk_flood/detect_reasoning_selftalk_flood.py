"""reasoning_selftalk_flood — self-addressed one-liners flood an output channel.

A density judgment, deliberately different from its sibling
``reasoning_filler_loop``: instead of asking "how many filler lines in a row",
it asks "how crowded is the densest 200-character stretch". The surface form is
the same defect a human reviewer reported::

    Let me write.
    Here.
    Now.
    Let me emit.
    OK.

Scans all three assistant output channels — ``reasoning_content``, ``content``
and tool-call argument text — with ONE shared bar. ``detail.channel`` records
which one tripped. The canonical name keeps its historical ``reasoning_`` prefix
because a released name is immutable (docs/archive/reference/detector_identity.md); it
is not a claim about scope.

Two independent sub-rules OR together; either one firing is a hit, and the
language is not classified up front:

    EN: some 200-char window holds (let me + newline) >= 18  and  let me >= 4
    ZH: some 200-char window holds (让我类 + newline) >= 18  and  让我类 >= 4
    both: keyword lines + filler lines >= 40% of the window's non-empty lines

让我类 is 让我 / 让me / let我. A filler line is a short all-Latin beat such as
``Here.`` / ``Now.`` / ``OK.``; short CJK lines are mostly headings and list
items, so Chinese counts keyword lines only.

Behavior 0.3.0 (2026-10-02) added the line share and moved the bar from 28/40
with a keyword floor of 2. Under 0.2.0 the bar was ~93% carried by newlines, so
two "let me" next to a byte dump, numbered list or ASCII glyph rows qualified;
on RL rollouts every 0.2.0 hit was such a window. The line share removes them,
which also lets the bar drop to catch self-talk with one newline per line
("Let me search." x14 never reached 28) and re-announcing the same next step
("让我先建任务。/ 实际上…… / 让我先建任务。"). The higher ZH total is no
longer needed because the share, not the count, now keeps ordinary Chinese
thinking out.

What the 0.2.0 bar meant per channel — measured over 1.09M assistant messages
of shared dev corpora, see docs/archive/notes/selftalk_flood_channel_census.md:

* The bar is ~93% (EN) / ~95% (ZH) carried by NEWLINES, since the keyword floor
  is only 2. So it selects "very short lines, crowded", and what clears it in
  ``content`` is layout collapse — ``</think>`` leakage followed by bare
  newlines, arrow-glyph floods, truncated JSON tails — rather than prose
  self-talk. The owner accepted those as bad cases under 0.2.0; since 0.3.0
  they rarely reach four keywords with a 40% line share, so this detector no
  longer reports them.
* Prose-length self-talk in ``content`` (lines ~20+ chars, e.g. a 552KB
  ``Let me persist.`` death spiral) does NOT clear the bar and is owned by
  ``repetition_content``. Lowering the bar to reach it admits ordinary code
  blocks first, so the bar stays where it is.
* ``tool_call`` argument text essentially never carries this shape (keyword
  present in 0.18% of arg strings; zero hits at every bar from 28 down to 12).
  It is scanned for completeness and as a regression tripwire, not because a
  phenomenon was found there.

The reported window is the FIRST one to qualify, which is the moment a
streaming implementation would trip. That matters because this judgment is
meant to be portable to the inference side: a streaming version needs only a
200-char tail plus in-window counters, no full text and no re-scan, and reaches
the same verdict as this offline implementation.

Kept self-contained (no ``from rules import``) on purpose — a bare ``rules``
import collides in ``sys.modules`` across detectors and silently yields zero
hits in batch runs.
"""
from __future__ import annotations

import bisect
import re
from typing import Any, Dict, List, Optional, Tuple

from .._core.detector import Finding
from .._core.messages import TrajectoryView
from .._core.rule_detector import RuleDetector

# Case-insensitive; matched against a lowercased copy of the text.
_EN_KEYWORDS = ("let me",)
# "让me" / "let我" are the code-switched forms that show up in mixed-language
# reasoning.
_ZH_KEYWORDS = ("让我", "让me", "let我")

# A filler line is a bare "Here." / "Now." / "OK." beat: short and nothing but
# Latin letters once punctuation and spaces are stripped. Byte dumps, numbered
# lists and ASCII glyph rows ("..", "##", "3.179", "1 Q", "A->A") are not
# filler, which is what keeps them out of the keyword-line share. Short CJK
# lines are mostly headings and list items, so Chinese counts keyword lines only.
_FILLER_MAX_CHARS = 12
_FILLER_PUNCT = re.compile(r"[\s.,!?;:…~\-—'\"`*()]+")
_FILLER_WORD = re.compile(r"^[A-Za-z]+$")

# Hit-point kinds, ordered so the sliding window can keep three counters.
_NEWLINE, _EN, _ZH = 0, 1, 2

# Channel ids are the repo's facet vocabulary (meta.yaml ``channel:``), so a
# Finding's detail.channel can be filtered with the same names the catalog uses.
_CH_REASONING = "reasoning"
_CH_CONTENT = "content"
_CH_TOOL_CALL = "tool_call"

# Colleague-facing channel labels (UI strings are Chinese; field names stay EN).
_CHANNEL_LABEL_ZH = {
    _CH_REASONING: "思维链(reasoning_content)",
    _CH_CONTENT: "正文(content)",
    _CH_TOOL_CALL: "工具调用参数(tool_call)",
}


def _hit_points(text: str) -> List[Tuple[int, int]]:
    """Character offsets of every keyword end and newline, ascending."""
    low = text.lower()
    pts: List[Tuple[int, int]] = []
    for kind, keywords in ((_EN, _EN_KEYWORDS), (_ZH, _ZH_KEYWORDS)):
        for kw in keywords:
            i = low.find(kw)
            while i >= 0:
                pts.append((i + len(kw), kind))
                i = low.find(kw, i + 1)
    i = text.find("\n")
    while i >= 0:
        pts.append((i, _NEWLINE))
        i = text.find("\n", i + 1)
    pts.sort()
    return pts


def _is_filler(line: str) -> bool:
    return len(line) <= _FILLER_MAX_CHARS and bool(_FILLER_WORD.match(_FILLER_PUNCT.sub("", line)))


def _line_table(text: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """Line start offsets plus prefix counts of non-empty, keyword and filler lines.

    Filler is counted only on lines that are not keyword lines of that sub-rule.
    """
    starts = [0]
    i = text.find("\n")
    while i >= 0:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    counts: Dict[str, List[int]] = {key: [0] for key in ("lines", "EN", "ZH", "EN_filler", "ZH_filler")}
    for j, start in enumerate(starts):
        line = text[start:starts[j + 1] - 1 if j + 1 < len(starts) else len(text)].strip()
        low = line.lower()
        en = bool(line) and any(kw in low for kw in _EN_KEYWORDS)
        zh = bool(line) and any(kw in low for kw in _ZH_KEYWORDS)
        filler = bool(line) and _is_filler(line)
        for key, flag in (("lines", bool(line)), ("EN", en), ("ZH", zh),
                          ("EN_filler", filler and not en), ("ZH_filler", filler and not zh)):
            counts[key].append(counts[key][-1] + flag)
    return starts, counts


def _densest_window(
    text: str,
    *,
    window_chars: int,
    en_total: int,
    en_keyword: int,
    zh_total: int,
    zh_keyword: int,
    keyword_line_share: float,
) -> Optional[Dict[str, Any]]:
    """First window satisfying either sub-rule, or None.

    Both sub-rules are evaluated in the same pass — whichever qualifies first
    wins — so one two-pointer sweep suffices. A window qualifies when its counts
    clear the bar AND keyword lines plus filler lines make up at least
    ``keyword_line_share`` of the non-empty lines it touches (whole lines).
    The line table is built only once some window's counts clear the bar, and
    each check is a constant-time prefix lookup.
    """
    pts = _hit_points(text)
    lo = 0
    n_nl = n_en = n_zh = 0
    table: Optional[Tuple[List[int], Dict[str, List[int]]]] = None

    for hi in range(len(pts)):
        kind = pts[hi][1]
        if kind == _NEWLINE:
            n_nl += 1
        elif kind == _EN:
            n_en += 1
        else:
            n_zh += 1
        # Shrink from the left until the span fits inside one window.
        while pts[hi][0] - pts[lo][0] > window_chars:
            kind0 = pts[lo][1]
            if kind0 == _NEWLINE:
                n_nl -= 1
            elif kind0 == _EN:
                n_en -= 1
            else:
                n_zh -= 1
            lo += 1

        end = pts[hi][0]
        for by, n_keyword, total, keyword in (("EN", n_en, en_total, en_keyword), ("ZH", n_zh, zh_total, zh_keyword)):
            if n_nl + n_keyword < total or n_keyword < keyword:
                continue
            if table is None:
                table = _line_table(text)
            starts, lines = table
            first_line = bisect.bisect_right(starts, pts[lo][0]) - 1
            last_line = bisect.bisect_right(starts, end)
            n_lines = lines["lines"][last_line] - lines["lines"][first_line]
            n_keyword_lines = lines[by][last_line] - lines[by][first_line]
            n_filler_lines = lines[f"{by}_filler"][last_line] - lines[f"{by}_filler"][first_line]
            if not n_lines or n_keyword_lines + n_filler_lines < keyword_line_share * n_lines:
                continue
            start = max(0, end - window_chars)
            return {
                "by": by,
                "start": start,
                "end": end,
                "n_total": n_nl + n_keyword,
                "n_keyword": n_keyword,
                "n_newline": n_nl,
                "n_keyword_lines": n_keyword_lines,
                "n_filler_lines": n_filler_lines,
                "n_lines": n_lines,
                "line": text.count("\n", 0, start) + 1,
                "excerpt": text[start:end],
            }
    return None


def _tool_call_text(msg: Any) -> str:
    """Tool-call argument text: ``args_raw`` plus every string-valued arg.

    Same union as ``response_protocol/_shared/_leak_shared.tool_call_arg_strings``
    (kept local, see the module docstring on ``rules`` import collisions). The
    decoded ``args`` values matter: a serialized JSON ``args_raw`` escapes
    newlines as the two characters ``\\n``, so a newline-density judgment can
    never fire on it alone.
    """
    parts: List[str] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        raw = getattr(tc, "args_raw", "") or ""
        if isinstance(raw, str) and raw:
            parts.append(raw)
        args = getattr(tc, "args", None)
        if isinstance(args, dict):
            parts.extend(v for v in args.values() if isinstance(v, str))
    return "\n".join(parts)


class ReasoningSelftalkFloodDetector(RuleDetector):
    detector_id = "reasoning_selftalk_flood"
    thresholds: Dict[str, Any] = {
        "window_chars": 200,
        # EN sub-rule. The keyword floor is what keeps byte-by-byte
        # enumerations and numbered lists out: newlines alone can clear the
        # total, so a window with no self-talk at all must not qualify.
        "en_total": 18,
        "en_keyword": 4,
        # ZH sub-rule. The line share below now does the work the higher
        # total used to do, so both languages share one bar.
        "zh_total": 18,
        "zh_keyword": 4,
        # Keyword lines plus filler lines must be at least this share of the
        # window's non-empty lines: the window has to read as self-talk, not
        # as a list or code block that happens to contain a few "let me".
        "keyword_line_share": 0.4,
    }

    def scan(self, view: TrajectoryView) -> List[Finding]:
        window_chars = int(self.thresholds["window_chars"])
        en_total = int(self.thresholds["en_total"])
        en_keyword = int(self.thresholds["en_keyword"])
        zh_total = int(self.thresholds["zh_total"])
        zh_keyword = int(self.thresholds["zh_keyword"])
        keyword_line_share = float(self.thresholds["keyword_line_share"])

        findings: List[Finding] = []
        for msg in view.messages:
            if msg.role != "assistant":
                continue
            # All three assistant output channels, one shared bar. At most one
            # Finding per (turn, channel): a turn whose reasoning AND content
            # both collapse is two distinct observations, not one merged row.
            for channel, text in (
                (_CH_REASONING, msg.reasoning_content or ""),
                (_CH_CONTENT, msg.content or ""),
                (_CH_TOOL_CALL, _tool_call_text(msg)),
            ):
                if not text.strip():
                    continue

                hit = _densest_window(
                    text,
                    window_chars=window_chars,
                    en_total=en_total,
                    en_keyword=en_keyword,
                    zh_total=zh_total,
                    zh_keyword=zh_keyword,
                    keyword_line_share=keyword_line_share,
                )
                if hit is None:
                    continue

                excerpt = hit["excerpt"]
                preview = " / ".join(
                    ln.strip() for ln in excerpt.split("\n") if ln.strip()
                )
                findings.append(Finding(
                    rule_id="reasoning_selftalk_flood",
                    severity="medium",
                    is_issue=True,
                    turn_index=msg.turn_index,
                    description=(
                        f"第 {msg.turn_index} 轮的"
                        f"{_CHANNEL_LABEL_ZH[channel]}里，"
                        f"某个 {window_chars} 字符窗口内挤进了 {hit['n_total']} 处"
                        f"自述句起头与换行"
                        f"(关键词 {hit['n_keyword']} + 换行 {hit['n_newline']}，"
                        f"自述句与填充行占 {hit['n_keyword_lines'] + hit['n_filler_lines']}/{hit['n_lines']} 行，"
                        f"{hit['by']} 判据)，该通道塌成一行行短句在空转:"
                        f"{preview[:120]!r}"
                    ),
                    detail={
                        "channel": channel,
                        "by": hit["by"],
                        "window_chars": window_chars,
                        "n_total": hit["n_total"],
                        "n_keyword": hit["n_keyword"],
                        "n_newline": hit["n_newline"],
                        "n_keyword_lines": hit["n_keyword_lines"],
                        "n_filler_lines": hit["n_filler_lines"],
                        "n_lines": hit["n_lines"],
                        "first_hit_line": hit["line"],
                        "first_hit_char_offset": hit["start"],
                        # Length of the channel text that tripped. Kept under
                        # the historical key so existing readers keep working.
                        "reasoning_chars": len(text),
                        "channel_chars": len(text),
                    },
                    quotes=[{
                        "kind": "reasoning_selftalk_flood",
                        "turn_index": msg.turn_index,
                        "role": "assistant",
                        "channel": channel,
                        "by": hit["by"],
                        "line": hit["line"],
                        "window_excerpt": excerpt[:500],
                    }],
                ))
        return findings


__all__ = ["ReasoningSelftalkFloodDetector"]
