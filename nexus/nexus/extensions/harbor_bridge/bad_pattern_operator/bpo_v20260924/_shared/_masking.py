"""F6_think_pattern_anomaly/_shared/_masking — assistant.content 标签计数预处理。

屏蔽代码块、行内代码、正则字面量、解释性上下文中出现的 <think>/</think>，
让 F6-1 到 F6-5 的 sub-detector 都在同一个"干净文本"上计数标签。

源头：搬自旧 ``detect_f6_think_pattern_anomaly.py``（masking + classification 逻辑）；
只是迁移位置，逻辑完全保持不变。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ─── 标签匹配正则 ──────────────────────────────────────────────────
# V1-verified: allow the optional ``:md5/token`` suffix on think tags (aligns
# with F6_6/F7 and the hashed serving-template form ``</think:6124c78e>``).
_SUF = r"(?::[A-Za-z0-9_-]+)?"
OPEN_TAG = re.compile(rf"<think{_SUF}>", re.IGNORECASE)
CLOSE_TAG = re.compile(rf"</think{_SUF}>", re.IGNORECASE)
PAIR_RE = re.compile(rf"<think{_SUF}>([\s\S]*?)</think{_SUF}>", re.IGNORECASE)
EMPTY_PAIR_RE = re.compile(rf"<think{_SUF}>\s*</think{_SUF}>", re.IGNORECASE)

# ─── 屏蔽用正则 ──────────────────────────────────────────────────
FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# 不屏蔽 tool_call/tool_calls/arg_value。
TOOL_BLOCK_PATTERNS: List[re.Pattern] = []

CODE_LIKE_LINE_RE = re.compile(
    r"""
    ^.*
    (?:
        re\.(?:search|match|compile|findall|finditer|sub)\s*\(
        |
        (?:PAIR_RE|OPEN_TAG|CLOSE_TAG|EMPTY_PAIR_RE)\s*=
        |
        r['"][^'"\n]*(?:<think>|</think>)[^'"\n]*['"]
        |
        ['"][^'"\n]*(?:<think>|</think>)[^'"\n]*['"]
    )
    .*$
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

EXPLANATION_CONTEXT_RE = re.compile(
    r"""
    (
        含义|意思|表示|代表|用于|用来|意味着|说明|
        标签|标记|tag|token|special\s*token|
        LLM|大模型|模型|推理标签|思考标签|reasoning\s*tag|
        示例|例子|example|代码|正则|字符串|literal|pattern|
        markdown|HTML|XML|转义|escape|escaped
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ─── 内部辅助函数 ────────────────────────────────────────────────

def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _empty_pair_spans(text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in EMPTY_PAIR_RE.finditer(text)]


def _add_span_unless_overlaps_empty_pair(
    spans: List[Tuple[int, int]],
    span: Tuple[int, int],
    empty_spans: List[Tuple[int, int]],
) -> None:
    if any(_overlaps(span, ep) for ep in empty_spans):
        return
    spans.append(span)


def _replace_spans_with_space(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            old_s, old_e = merged[-1]
            merged[-1] = (old_s, max(old_e, e))
    parts: List[str] = []
    last = 0
    for s, e in merged:
        parts.append(text[last:s])
        parts.append(" " * (e - s))
        last = e
    parts.append(text[last:])
    return "".join(parts)


def _collect_non_leakage_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    empty_spans = _empty_pair_spans(text)

    for m in FENCED_CODE_RE.finditer(text):
        _add_span_unless_overlaps_empty_pair(spans, m.span(), empty_spans)

    for m in INLINE_CODE_RE.finditer(text):
        _add_span_unless_overlaps_empty_pair(spans, m.span(), empty_spans)

    for pat in TOOL_BLOCK_PATTERNS:
        for m in pat.finditer(text):
            _add_span_unless_overlaps_empty_pair(spans, m.span(), empty_spans)

    for m in CODE_LIKE_LINE_RE.finditer(text):
        _add_span_unless_overlaps_empty_pair(spans, m.span(), empty_spans)

    for m in PAIR_RE.finditer(text):
        start, end = m.span()
        if EMPTY_PAIR_RE.fullmatch(m.group(0)):
            continue
        ctx = text[max(0, start - 120): min(len(text), end + 120)]
        if EXPLANATION_CONTEXT_RE.search(ctx):
            _add_span_unless_overlaps_empty_pair(spans, (start, end), empty_spans)

    for tag_re in (OPEN_TAG, CLOSE_TAG):
        for m in tag_re.finditer(text):
            start, end = m.span()
            if any(_overlaps((start, end), ep) for ep in empty_spans):
                continue
            ctx = text[max(0, start - 80): min(len(text), end + 80)]
            if EXPLANATION_CONTEXT_RE.search(ctx):
                spans.append((start, end))

    return spans


# ─── 公开接口 ────────────────────────────────────────────────────

def mask_non_leakage_context(text: str) -> str:
    """屏蔽代码块、正则字面量、解释性上下文中的 <think> 标签。"""
    spans = _collect_non_leakage_spans(text)
    return _replace_spans_with_space(text, spans)


def count_tags(text: str) -> Dict[str, Any]:
    """对 masked 文本计数 think 标签，返回所有标签统计信息。

    Returns:
        dict with keys:
            n_open          — <think> 标签数
            n_close         — </think> 标签数
            empty_pair_count — <think></think> 空对数
            tail_after_close — 最后一个 </think> 之后的文本（截短到120字符）
    """
    masked = mask_non_leakage_context(text)

    n_open = len(OPEN_TAG.findall(masked))
    n_close = len(CLOSE_TAG.findall(masked))
    empty_pair_count = len(EMPTY_PAIR_RE.findall(masked))

    tail_after_close = ""
    if n_close >= 1:
        idx = masked.lower().rfind("</think>")
        after = masked[idx + len("</think>"):]
        tail_after_close = after[:120]

    return {
        "n_open": n_open,
        "n_close": n_close,
        "empty_pair_count": empty_pair_count,
        "tail_after_close": tail_after_close,
    }


def get_combined_text(message: Any) -> str:
    """从 assistant message 中拼接 reasoning_content + content 用于标签计数。

    Accepts both dict-typed messages (legacy bpo path) and dqc
    ``Message`` dataclasses (post-rebrand). The helper intentionally
    stays loosely typed so the F6 sub-detectors can share it.
    """
    if isinstance(message, dict):
        text_main = ""
        c = message.get("content")
        if isinstance(c, str):
            text_main = c
        elif isinstance(c, list):
            parts = []
            for x in c:
                if isinstance(x, dict):
                    t = x.get("text") or x.get("content")
                    if isinstance(t, str):
                        parts.append(t)
                elif isinstance(x, str):
                    parts.append(x)
            text_main = "\n".join(parts)
        rc_raw = message.get("reasoning_content")
    else:
        # dqc Message dataclass
        text_main = getattr(message, "content", "") or ""
        rc_raw = getattr(message, "reasoning_content", "")
    rc = rc_raw if isinstance(rc_raw, str) else ""

    if rc and text_main:
        return rc + "\n" + text_main
    return rc or text_main


# ── V1-verified per-channel counting (ported from origin/v1-verified-deployed) ──
# F6_1..F6_5 count think-tag structure PER CHANNEL (not concatenated), so a
# reasoning segment and a content segment that each contain one <think></think>
# pair aren't merged into a false "2 closes" multi-close.
_MAX_THINK_TAGS = 40      # count cap: more = degenerate spam / truncation garbage / code
_MAX_EMPTY_PAIRS = 20


def get_channel_texts(message: Any) -> List[str]:
    """Return the reasoning_content and content channel texts separately (each
    only if non-empty). F6 counts per-channel."""
    out: List[str] = []
    if isinstance(message, dict):
        rc_raw = message.get("reasoning_content")
        c = message.get("content")
        text_main = c if isinstance(c, str) else ""
    else:
        rc_raw = getattr(message, "reasoning_content", "")
        text_main = getattr(message, "content", "") or ""
    if isinstance(rc_raw, str) and rc_raw.strip():
        out.append(rc_raw)
    if isinstance(text_main, str) and text_main.strip():
        out.append(text_main)
    return out


_TOOLCALL_FRAGMENT_RE = re.compile(
    r"<\s*/?\s*(?:arg_key|arg_value|tool_sep)(?::[A-Za-z0-9_-]+)?\s*>",
    re.IGNORECASE,
)
_ORPHAN_CLOSE_RE = re.compile(
    r"<\s*/\s*(?:arg_key|arg_value|tool_sep|tool_call|tool_calls)"
    r"(?::[A-Za-z0-9_-]+)?\s*>",
    re.IGNORECASE,
)
_HTML_GARBAGE_RE = re.compile(
    r"minced|grid_minced|table_minced|<\s*td\b[^>]*minced|<\s*a\s+name=",
    re.IGNORECASE,
)


def is_truncated_toolcall_garbage(text: str) -> bool:
    """Is the text a truncated/malformed tool_call tag block or synthetic
    garbage (where think-tag counting is unreliable)?"""
    if not text:
        return False
    if _HTML_GARBAGE_RE.search(text):
        return True
    n_open = len(re.findall(r"<\s*tool_calls?(?::[A-Za-z0-9_-]+)?\s*>", text, re.I))
    n_close = len(re.findall(r"<\s*/\s*tool_calls?(?::[A-Za-z0-9_-]+)?\s*>", text, re.I))
    if n_open != n_close:
        n_frag = len(_TOOLCALL_FRAGMENT_RE.findall(text))
        if n_frag >= 1 or n_open or n_close:
            return True
    n_open_arg = len(re.findall(
        r"<\s*(?:arg_key|arg_value|tool_sep|tool_call)(?::[A-Za-z0-9_-]+)?\s*>",
        text, re.I))
    n_close_arg = len(_ORPHAN_CLOSE_RE.findall(text))
    if n_close_arg > n_open_arg:
        return True
    return False


def channel_tag_stats(message: Any, skip_garbage: bool = True) -> List[Dict[str, Any]]:
    """count_tags each non-empty channel of the message separately; return a
    list of stats. F6_1..F6_5 unified entry (per-channel, no cross-channel
    concat). Each item carries ``_channel_text`` for evidence. skip_garbage
    skips truncated malformed tool_call blocks (F6_1/2/4/5); F6_3 (multi-close)
    treats the malformed block itself as signal, so passes skip_garbage=False."""
    out: List[Dict[str, Any]] = []
    for text in get_channel_texts(message):
        low = text.lower()
        if "<think" not in low and "</think" not in low:
            continue
        if skip_garbage and is_truncated_toolcall_garbage(text):
            continue
        stats = count_tags(text)
        # count cap: a clean think anomaly is small (1~4). Dozens/thousands =
        # degenerate output / truncation garbage / code literally containing
        # "<think>" — not a clean think anomaly, skip.
        if (
            stats["n_open"] >= _MAX_THINK_TAGS
            or stats["n_close"] >= _MAX_THINK_TAGS
            or stats["empty_pair_count"] > _MAX_EMPTY_PAIRS
        ):
            continue
        stats["_channel_text"] = text
        out.append(stats)
    return out


__all__ = [
    "OPEN_TAG",
    "CLOSE_TAG",
    "PAIR_RE",
    "EMPTY_PAIR_RE",
    "mask_non_leakage_context",
    "count_tags",
    "channel_tag_stats",
    "get_channel_texts",
    "is_truncated_toolcall_garbage",
    "get_combined_text",
]
