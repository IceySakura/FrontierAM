"""F7_tool_call_format_error/_shared/_token_patterns — tool-call 协议标签检测。

提供 tool_call 协议标签的所有正则模式、碎片检测逻辑、token 收集工具。
4 个 sub-detector（F7-1 ~ F7-4）都使用这些基础设施。

源头：搬自旧 ``detect_f7_tool_call_format_error.py``；逻辑完全保持不变。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Match, Optional


# ─── tool-call special-token DSL 词表 ─────────────────────────────
# 背景（owner 2026-07-08）：模型从不直接产 JSON。它产出的是这套 special-token
# DSL —— <tool_calls:HASH><tool_call:HASH>name<tool_sep:HASH><arg_key:HASH>k
# </arg_key:HASH><arg_value:HASH>v</arg_value:HASH></tool_call:HASH></tool_calls:HASH>
# —— tool parser 把它解析成结构化 tool_calls（arguments 是一段 parser 生成的
# JSON 字符串）。tool_choice=auto 时解析并发出；tool_choice=none 时原样留在文本
# （那是设计，不是泄漏，但 tool_choice 不在我们导出的数据里，见 memory）。
#
# 推论：解析后 parser 本应消费掉整套 DSL，所以这些 tag 里任何一个残留在解析后的
# content / reasoning_content 里都是非法泄漏 —— 不止 <tool_call>，还包括容器
# <tool_calls>、分隔 <tool_sep>、参数 <arg_key>/<arg_value>、以及 env 观察
# wrapper <tool_response>（实测漏进过 assistant.reasoning_content）。
#
# 覆盖两层（白名单 + hash 兜底，owner 选 "白名单+hash风格兼容"）：
#   1) 白名单（下方 TOOL_CALL_PROTOCOL_TAG）：命名精确，bare 与带后缀两种都收。
#   2) hash 兜底（HASHED_SPECIAL_TOKEN_RE）：<word:恰好8位hex> —— SFT chat
#      template 的 special-token marker 是固定 8 位 hex（全量实测 9606/9606 都是
#      6124c78e），所以任何 <任意词:8hex> 都是 DSL marker，未来新增 DSL tag 也能
#      兜住。护栏：必须"恰好 8 位 hex"—— WebDAV 的 <opaquelocktoken:UUID>（32/64
#      位）天然被排除，不会误伤文件内容。绝不能放宽成 "<任意:任意hash>"。
TOOL_CALL_PROTOCOL_TAG = (
    r"tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value|tool_response"
)
TOOL_CALL_PROTOCOL_SUFFIX = r"(?::[A-Za-z0-9_-]+)?"
# hash 兜底层：<word:恰好8位hex>（含可选 "/"）。8-hex 精确长度是排除
# opaquelocktoken(UUID) 的关键护栏；见上方注释。
HASHED_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*/?\s*[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{8}\s*>",
    re.IGNORECASE,
)
# HY hashed-only tool tags: the named HY DSL plus :8hex.
# Not any <word:8hex> — that is not the HY protocol.
# Bare <tool_call> stays on tool_call_protocol_leak_all_family;
# hashed <think:hex> stays on think_protocol_leak_hy.
HY_HASHED_TOOL_TAG = (
    r"tool_calls?|tool_sep|arg_key|arg_value|tool_response"
)
HY_HASHED_TOOL_TOKEN_RE = re.compile(
    rf"<\s*/?\s*(?:{HY_HASHED_TOOL_TAG}):[0-9a-f]{{8}}\s*>",
    re.IGNORECASE,
)
TOOL_CALL_LITERAL_TOKEN_RE = re.compile(
    rf"<\s*/?\s*(?:(?:{TOOL_CALL_PROTOCOL_TAG}){TOOL_CALL_PROTOCOL_SUFFIX}"
    rf"|[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{{8}})\s*>",
    re.IGNORECASE,
)

# PROTOCOL_TOOL_TOKEN_RE is the all-family lexicon used by
# tool_call_protocol_leak_all_family. TOOL_CALL_LITERAL_TOKEN_RE /
# TOOL_CALL_PROTOCOL_TAG stay the original GLM/hy3 XML whitelist so the
# parked channel-split tool_call_leak_* detectors and incomplete-fragment
# balance do not silently widen. tool_call_protocol_leak_hy uses
# HY_HASHED_TOOL_TOKEN_RE only. HY :8hex (named-tag suffix and the
# <word:8hex> catch-all) stays off this regex.
#
# Extra forms (open-weight chat templates): namespaced XML, attribute
# opens (<function=…>, <parameter=…>, <invoke name=…>), special tokens
# (DeepSeek / Kimi / InternLM / Step), and Kimi K3 XTML tool tags.
# Bare <function> / <parameter> / <tools> / <tool> / <response> are
# omitted — those collide with HTML and documentation.
_PROTOCOL_TOOL_XML_CI = (
    r"tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value|"
    r"tool_response|tool_responses|tool_output|tool_list|"
    r"longcat_tool_call|longcat_arg_key|longcat_arg_value|"
    r"longcat_tool_response|longcat_tool_declare|longcat_observation|"
    r"minimax:tool_call|seed:tool_call|"
    r"steptml:toolcall|steptml:invoke|steptml:parameter"
)
# Keep the real lowercase <parameters> protocol tag while excluding common C++
# template names such as <Parameters>.  The surrounding regex is IGNORECASE, so
# this one alternative explicitly disables it.
_PROTOCOL_TOOL_XML = rf"(?:{_PROTOCOL_TOOL_XML_CI}|(?-i:parameters))"
_PROTOCOL_TOOL_PIPE = (
    r"tool_calls_section_begin|tool_call_begin|"
    r"tool_call_argument_begin|tool_call_end|tool_calls_section_end|"
    r"action_start|action_end|plugin|observation"
)
_PROTOCOL_TOOL_FULLWIDTH = (
    r"tool▁calls▁begin|tool▁call▁begin|tool▁call▁end|"
    r"tool▁calls▁end|tool▁outputs▁begin|tool▁output▁begin|"
    r"tool▁output▁end|tool▁outputs▁end|tool▁sep|"
    r"tool_calls_begin|tool_call_begin|tool_call_end|"
    r"tool_calls_end|tool_outputs_begin|tool_output_begin|"
    r"tool_output_end|tool_outputs_end|tool_sep"
)
PROTOCOL_TOOL_TAG = _PROTOCOL_TOOL_XML
# Named-tag suffix for all-family: keep :request-v2 style, drop HY :8hex.
_PROTOCOL_TOOL_SUFFIX = r"(?::(?![0-9a-f]{8}(?![A-Za-z0-9_-]))[A-Za-z0-9_-]+)?"
_XML_ATTRIBUTE_NAME = r"[A-Za-z_:][A-Za-z0-9_.:-]*"
_XML_ATTRIBUTE_VALUE = r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s>"'=\r\n]+)'''
_XML_ATTRIBUTE = rf"[ \t]+{_XML_ATTRIBUTE_NAME}[ \t]*=[ \t]*{_XML_ATTRIBUTE_VALUE}"
_XML_ATTRIBUTES = rf"(?:{_XML_ATTRIBUTE})*"
PROTOCOL_TOOL_TOKEN_RE = re.compile(
    rf"(?:"
    rf"<[ \t]*/?[ \t]*(?:{_PROTOCOL_TOOL_XML}){_PROTOCOL_TOOL_SUFFIX}"
    rf"{_XML_ATTRIBUTES}[ \t]*>"
    rf"|<[ \t]*(?:function|parameter)[ \t]*=[ \t]*[^\s>/]+[ \t]*>"
    rf"|<[ \t]*parameter(?:{_XML_ATTRIBUTE})+[ \t]*>"
    rf"|<[ \t]*/[ \t]*parameter[ \t]*>"
    rf"|<[ \t]*/[ \t]*function[ \t]*>"
    rf"|<\|(?:{_PROTOCOL_TOOL_PIPE})\|>"
    rf"|<｜(?:{_PROTOCOL_TOOL_FULLWIDTH})｜>"
    rf"|<\|open\|>(?:call|tools|argument|json)\b"
    rf"|<\|close\|>(?:call|tools|argument|json)<\|sep\|>"
    rf")",
    re.IGNORECASE,
)

# ─── 容器标签 open/close 分离正则（d 规则专用）──────────────────────
TOOL_CALLS_OPEN_RE = re.compile(
    r"<\s*tool_calls(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
TOOL_CALLS_CLOSE_RE = re.compile(
    r"<\s*/\s*tool_calls(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
TOOL_CALL_OPEN_RE = re.compile(
    r"<\s*tool_call(?!s)(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
TOOL_CALL_CLOSE_RE = re.compile(
    r"<\s*/\s*tool_call(?!s)(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
# arg-level container tags (V1-verified FN-audit addition to check_incomplete_fragment):
# a balanced container can still hide an orphan arg_key/arg_value = incomplete fragment.
ARG_KEY_OPEN_RE = re.compile(
    r"<\s*arg_key(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
ARG_KEY_CLOSE_RE = re.compile(
    r"<\s*/\s*arg_key(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
ARG_VALUE_OPEN_RE = re.compile(
    r"<\s*arg_value(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)
ARG_VALUE_CLOSE_RE = re.compile(
    r"<\s*/\s*arg_value(?::[A-Za-z0-9_-]+)?\s*>", re.IGNORECASE)


# ─── 工具函数 ────────────────────────────────────────────────────

def find_all_tool_call_literal_tokens(text: str) -> List[str]:
    """返回文本中命中的 tool_call 字面 token，去重但保序。"""
    seen: set = set()
    out: List[str] = []
    for m in TOOL_CALL_LITERAL_TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        key = re.sub(r"\s+", "", tok.lower())
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out


def has_literal_tool_call_token(text: str) -> Optional[Match[str]]:
    """只要出现任意 tool-call protocol token 就返回 match。"""
    if not text:
        return None
    return TOOL_CALL_LITERAL_TOKEN_RE.search(text)


def check_incomplete_fragment(text: str) -> Optional[Dict[str, Any]]:
    """检查 text 中 tool_call 容器标签的 open/close 是否不匹配。

    只检测两级容器标签：
      - <tool_calls(:id)?> / </tool_calls(:id)?>
      - <tool_call(:id)?>  / </tool_call(:id)?>

    Returns:
        dict with fragment info if unbalanced, None if balanced or no tags.
    """
    if not text:
        return None

    n_open_calls = len(TOOL_CALLS_OPEN_RE.findall(text))
    n_close_calls = len(TOOL_CALLS_CLOSE_RE.findall(text))
    n_open_call = len(TOOL_CALL_OPEN_RE.findall(text))
    n_close_call = len(TOOL_CALL_CLOSE_RE.findall(text))

    unbalanced: List[Dict[str, Any]] = []
    if (n_open_calls or n_close_calls) and n_open_calls != n_close_calls:
        unbalanced.append({
            "tag": "tool_calls",
            "n_open": n_open_calls,
            "n_close": n_close_calls,
        })
    if (n_open_call or n_close_call) and n_open_call != n_close_call:
        unbalanced.append({
            "tag": "tool_call",
            "n_open": n_open_call,
            "n_close": n_close_call,
        })
    # arg-level tag balance (V1-verified FN-audit addition): a balanced
    # container can still hide an orphan arg_key/arg_value = incomplete fragment.
    n_open_ak = len(ARG_KEY_OPEN_RE.findall(text))
    n_close_ak = len(ARG_KEY_CLOSE_RE.findall(text))
    n_open_av = len(ARG_VALUE_OPEN_RE.findall(text))
    n_close_av = len(ARG_VALUE_CLOSE_RE.findall(text))
    if (n_open_ak or n_close_ak) and n_open_ak != n_close_ak:
        unbalanced.append({"tag": "arg_key", "n_open": n_open_ak, "n_close": n_close_ak})
    if (n_open_av or n_close_av) and n_open_av != n_close_av:
        unbalanced.append({"tag": "arg_value", "n_open": n_open_av, "n_close": n_close_av})

    if not unbalanced:
        return None

    return {
        "unbalanced_tags": unbalanced,
        "n_open_calls": n_open_calls,
        "n_close_calls": n_close_calls,
        "n_open_call": n_open_call,
        "n_close_call": n_close_call,
    }


def snippet(text: str, max_len: int = 2000) -> str:
    """把文本里的 \\n 转义成可读形式，截短。"""
    if not text:
        return ""
    s = text.replace("\n", "\\n")
    return s[:max_len] + (" …" if len(s) > max_len else "")


def around(text: str, start: int, end: Optional[int] = None, radius: int = 320) -> str:
    """截取命中位置附近的上下文。"""
    if not text:
        return ""
    if end is None:
        end = start
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return snippet(prefix + text[left:right] + suffix, max_len=radius * 2 + 20)


__all__ = [
    "TOOL_CALL_LITERAL_TOKEN_RE",
    "PROTOCOL_TOOL_TAG",
    "PROTOCOL_TOOL_TOKEN_RE",
    "HASHED_SPECIAL_TOKEN_RE",
    "HY_HASHED_TOOL_TAG",
    "HY_HASHED_TOOL_TOKEN_RE",
    "TOOL_CALLS_OPEN_RE",
    "TOOL_CALLS_CLOSE_RE",
    "TOOL_CALL_OPEN_RE",
    "TOOL_CALL_CLOSE_RE",
    "find_all_tool_call_literal_tokens",
    "has_literal_tool_call_token",
    "check_incomplete_fragment",
    "snippet",
    "around",
]
