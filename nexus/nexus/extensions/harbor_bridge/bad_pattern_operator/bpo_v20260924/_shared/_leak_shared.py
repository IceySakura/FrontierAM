"""Shared tool-call-leak gates for the FORMAT tool_call_leak_* detectors.

Extracted from the former merged tool_call_leak so the per-channel split detectors
(tool_call_leak_content / __reasoning / __tool_call / __reasoning_with_valid_call)
and incomplete_fragment share the exact same gates. See _token_patterns for the
low-level regexes/helpers this builds on.
"""
from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable, List


# Recall-first content contract: any known tool protocol tag (bare or suffixed),
# plus future DSL tags shaped as <word:exactly-8-hex>. Context and structured
# tool_calls state are evidence attributes, not skip gates. ``think`` is an
# attribution boundary rather than an exemption: think_protocol_leak_hy owns hashed
# HY think tags; tool_call_protocol_leak_hy owns hashed HY tool tags.
TOOL_TOKEN_IN_CONTENT_RE = re.compile(
    r"<\s*/?\s*(?:"
    r"(?:tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value|tool_response)"
    r"(?::[A-Za-z0-9_-]+)?"
    r"|(?!think:)[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{8}"
    r")\s*>",
    re.IGNORECASE,
)

_FENCED_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_HARD_FAIL_RE = re.compile(
    r"No valid JSON found|parsing errors|Missing required fields|Invalid JSON"
    r"|exceeded (?:the )?maximum output length|output token limit|truncat",
    re.IGNORECASE,
)
_HASH_FRAG_RE = re.compile(r"<\s*/?\s*tool_calls?:[A-Za-z0-9_-]{4,}\s*>", re.IGNORECASE)
_FILE_CONTENT_ARGUMENTS = frozenset({'code', 'content', 'file_content', 'new_string', 'newtext', 'text'})
_RAW_EDIT_TOOLS = frozenset({'filesystem_edit_file'})


@dataclass(frozen=True, slots=True)
class AssistantChannelEntry:
    field: str
    text: str
    tool_name: str = ''
    tool_family: str = ''
    argument_name: str = ''
    is_file_content: bool = False


def _mask_match(match: re.Match[str]) -> str:
    """Blank a non-runtime literal while keeping evidence offsets stable."""
    return " " * len(match.group(0))


def strip_fences(content: str) -> str:
    return _FENCED_RE.sub(_mask_match, content)


def code_literal_context(content: str, start: int) -> str:
    """Classify a token position without using the context as a skip gate."""
    for match in _FENCED_RE.finditer(content or ""):
        if match.start() <= start < match.end():
            return "fenced_code"
    for match in _INLINE_CODE_RE.finditer(content or ""):
        if match.start() <= start < match.end():
            return "inline_code"
    return "plain"


def describe_token_contexts(
    content: str,
    matches: Iterable[re.Match[str]],
) -> List[dict[str, str]]:
    """Return unique token/context pairs in source order for later judging."""
    seen: set[tuple[str, str]] = set()
    out: List[dict[str, str]] = []
    for match in matches:
        token = match.group(0)
        context = code_literal_context(content, match.start())
        key = (re.sub(r"\s+", "", token.lower()), context)
        if key not in seen:
            seen.add(key)
            out.append({"token": token, "context": context})
    return out


def find_all_content_leak_tokens(text: str) -> List[str]:
    """Return unique recall-first tool protocol tokens in source order."""
    seen: set[str] = set()
    out: List[str] = []
    for match in TOOL_TOKEN_IN_CONTENT_RE.finditer(text or ""):
        token = match.group(0)
        key = re.sub(r"\s+", "", token.lower())
        if key not in seen:
            seen.add(key)
            out.append(token)
    return out


def fragment_is_problematic(messages, i: int, content: str) -> bool:
    """Does this content fragment have evidence it actually broke / was truncated?
    (1) hash suffix (vLLM truncation); (2) next-turn feedback has a parse-fail /
    over-length; (3) it's the last assistant message (generation ended mid-call)."""
    if _HASH_FRAG_RE.search(content):
        return True
    for j in range(i + 1, len(messages)):
        if messages[j].role == "assistant":
            return False
        c = messages[j].content
        if isinstance(c, str) and _HARD_FAIL_RE.search(c[:400]):
            return True
    return True


def _string_argument_leaves(
    value: object,
    *,
    path: str = "",
    leaf_name: str = "",
) -> Iterable[tuple[str, str, str]]:
    """Yield ``(path, leaf_name, text)`` for every string in nested JSON arguments."""
    if isinstance(value, str):
        yield path, leaf_name, value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = str(key)
            nested_path = f"{path}.{key_name}" if path else key_name
            yield from _string_argument_leaves(nested, path=nested_path, leaf_name=key_name)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _string_argument_leaves(
                nested,
                path=f"{path}[{index}]",
                leaf_name=leaf_name,
            )


def tool_call_arg_strings(m) -> List[str]:
    """Every string-valued tool_call argument (+ args_raw) on an assistant turn —
    the text a leak in the tool_call CHANNEL would surface in. ToolCall is a
    frozen dataclass (``args`` dict + ``args_raw`` str), not a dict."""
    out: List[str] = []
    for tc in (getattr(m, "tool_calls", None) or []):
        raw = getattr(tc, "args_raw", "") or ""
        if isinstance(raw, str) and raw:
            out.append(raw)
        args = getattr(tc, "args", None)
        if isinstance(args, dict):
            out.extend(v for v in args.values() if isinstance(v, str))
    return out


def assistant_channel_texts(m) -> List[tuple[str, str]]:
    """content, reasoning_content, then each tool-call argument string."""
    out: List[tuple[str, str]] = [
        ("content", m.content or ""),
        ("reasoning_content", getattr(m, "reasoning_content", None) or ""),
    ]
    out.extend(("tool_call", arg) for arg in tool_call_arg_strings(m))
    return out


def assistant_channel_entries(m) -> List[AssistantChannelEntry]:
    """Channel text with tool/argument provenance for precision-sensitive rules.

    Parsed string arguments are preferred over ``args_raw`` so file-content
    policies can reason about the actual argument name without double-counting
    the same text inside its serialized JSON representation.
    """
    out = [
        AssistantChannelEntry("content", m.content or ""),
        AssistantChannelEntry("reasoning_content", getattr(m, "reasoning_content", None) or ""),
    ]
    for tc in (getattr(m, "tool_calls", None) or []):
        args = getattr(tc, "args", None)
        string_args = list(_string_argument_leaves(args)) if isinstance(args, dict) else []
        tool_name = str(getattr(tc, "raw_name", "") or getattr(tc, "name", "") or "")
        canonical_name = str(getattr(tc, "name", "") or "")
        is_edit_tool = canonical_name == "edit" or tool_name.casefold() in _RAW_EDIT_TOOLS
        for argument_path, leaf_name, value in string_args:
            out.append(
                AssistantChannelEntry(
                    "tool_call",
                    value,
                    tool_name=tool_name,
                    tool_family=canonical_name,
                    argument_name=argument_path,
                    is_file_content=is_edit_tool and leaf_name.casefold() in _FILE_CONTENT_ARGUMENTS,
                )
            )
        raw = getattr(tc, "args_raw", "") or ""
        if isinstance(raw, str) and raw and not string_args:
            out.append(
                AssistantChannelEntry(
                    "tool_call",
                    raw,
                    tool_name=tool_name,
                    tool_family=canonical_name,
                    argument_name="<raw>",
                )
            )
    return out


def unique_match_tokens(matches: Iterable[re.Match[str]]) -> List[str]:
    """Deduplicate match texts by collapsed lowercase spelling, keep source order."""
    seen: set[str] = set()
    tokens: List[str] = []
    for match in matches:
        token = match.group(0)
        key = re.sub(r"\s+", "", token.lower())
        if key not in seen:
            seen.add(key)
            tokens.append(token)
    return tokens


__all__ = [
    "TOOL_TOKEN_IN_CONTENT_RE",
    "code_literal_context",
    "describe_token_contexts",
    "find_all_content_leak_tokens",
    "strip_fences",
    "fragment_is_problematic",
    "tool_call_arg_strings",
    "assistant_channel_texts",
    "AssistantChannelEntry",
    "assistant_channel_entries",
    "unique_match_tokens",
]
