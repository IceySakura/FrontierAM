"""友商 Tool Call 泄漏（各家 chat template）。

用 PROTOCOL_TOOL_TOKEN_RE 扫描 assistant 的 content、reasoning_content 和
tool_call 参数。裸 <tool_call> 与各家 tool 协议标签由本算子报；HY hashed
<tool_call:8hex> 只由 tool_call_protocol_leak_hy 报。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List


from .._core.detector import Finding  # noqa: E402
from .._core.messages import Message, TrajectoryView  # noqa: E402
from .._core.rule_detector import RuleDetector  # noqa: E402
from .._shared._leak_shared import (  # noqa: E402
    assistant_channel_entries,
    code_literal_context,
)
from .._shared._token_patterns import PROTOCOL_TOOL_TOKEN_RE, around  # noqa: E402


SEVERITY = "high"
_FIELD_ORDER = ("content", "reasoning_content", "tool_call")
_TRAILING_PROTOCOL_CLOSES = re.compile(
    r"(?:"
    r"</[ \t]*(?:content|invoke|parameter|function|tool_call|tool_calls)"
    r"(?:\:[A-Za-z0-9_-]+)?[ \t]*>[ \t\r\n]*"
    r")+$",
    re.IGNORECASE,
)


def _append_unique(target: List[str], token: str) -> None:
    key = re.sub(r"\s+", "", token.casefold())
    if all(re.sub(r"\s+", "", item.casefold()) != key for item in target):
        target.append(token)


def _is_trailing_file_protocol_close(text: str, match: re.Match[str]) -> bool:
    suffix = _TRAILING_PROTOCOL_CLOSES.search(text)
    return suffix is not None and match.start() >= suffix.start()


def _is_inside_shell_quote(text: str, start: int) -> bool:
    quote = ""
    escaped = False
    for char in text[:start]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in {"'", '"'}:
            quote = char
    return bool(quote)


class ToolCallProtocolLeakDetector(RuleDetector):
    detector_id = "tool_call_protocol_leak_all_family"
    thresholds: Dict[str, Any] = {"severity": SEVERITY}

    def scan(self, view: TrajectoryView) -> List[Finding]:
        findings: List[Finding] = []
        for i, m in enumerate(view.messages):
            if m.role != "assistant":
                continue
            tokens_by_field: Dict[str, List[str]] = {}
            first_match = None
            first_field = ""
            first_text = ""
            all_tokens: List[str] = []
            ignored_matches: List[Dict[str, Any]] = []
            first_ignored = None
            for entry in assistant_channel_entries(m):
                for match in PROTOCOL_TOOL_TOKEN_RE.finditer(entry.text):
                    ignored_reason = ""
                    if entry.field in {"content", "reasoning_content"}:
                        literal_context = code_literal_context(entry.text, match.start())
                        if literal_context != "plain":
                            ignored_reason = literal_context
                    elif entry.is_file_content and not _is_trailing_file_protocol_close(entry.text, match):
                        ignored_reason = "file_content_non_terminal_protocol_text"
                    elif entry.tool_family == "shell" and _is_inside_shell_quote(entry.text, match.start()):
                        ignored_reason = "shell_quoted_literal"

                    if ignored_reason:
                        ignored_matches.append({
                            "field": entry.field,
                            "token": match.group(0),
                            "reason": ignored_reason,
                            "tool_name": entry.tool_name,
                            "tool_family": entry.tool_family,
                            "argument_name": entry.argument_name,
                        })
                        if first_ignored is None:
                            first_ignored = (match, entry.field, entry.text)
                        continue

                    existing = tokens_by_field.setdefault(entry.field, [])
                    _append_unique(existing, match.group(0))
                    _append_unique(all_tokens, match.group(0))
                    if first_match is None:
                        first_match, first_field, first_text = match, entry.field, entry.text
            if first_match is None:
                if first_ignored is not None:
                    ignored_match, ignored_field, ignored_text = first_ignored
                    findings.append(Finding(
                        rule_id=self.detector_id, severity=SEVERITY, is_issue=False,
                        turn_index=i,
                        description=(
                            f"assistant 第 {i} 条消息:协议样式文本仅出现在"
                            "代码引用或文件正文非末尾，按非问题证据记录。"
                        ),
                        detail={
                            "fields_hit": [],
                            "ignored_matches": ignored_matches,
                        },
                        quotes=[{
                            "kind": self.detector_id, "turn_index": i, "role": "assistant",
                            "field": ignored_field, "matched_tag": ignored_match.group(0),
                            "snippet": around(ignored_text, ignored_match.start(), ignored_match.end()),
                        }],
                    ))
                continue
            fields_hit = [field for field in _FIELD_ORDER if field in tokens_by_field]
            findings.append(Finding(
                rule_id=self.detector_id, severity=SEVERITY, is_issue=True,
                turn_index=i,
                description=(
                    f"assistant 第 {i} 条消息:各家 tool 协议标签出现在 "
                    f"{'+'.join(fields_hit)}。"
                ),
                detail={
                    "fields_hit": fields_hit,
                    "tokens_by_field": tokens_by_field,
                    "matched_tokens": list(all_tokens),
                    "first_matched_token": first_match.group(0),
                    "ignored_matches": ignored_matches,
                },
                quotes=[{
                    "kind": self.detector_id, "turn_index": i, "role": "assistant",
                    "field": first_field, "matched_tag": first_match.group(0),
                    "snippet": around(first_text, first_match.start(), first_match.end()),
                }],
            ))
        return findings


__all__ = ["ToolCallProtocolLeakDetector"]
