"""HY Tool Call 泄漏。

HY 协议标签是具名 DSL 加 :8hex（tool_call / tool_calls / tool_sep /
arg_key / arg_value / tool_response）。只扫 HY_HASHED_TOOL_TOKEN_RE，
三个 assistant 通道同一规则。任意 <word:8hex> 和裸 <tool_call> 不报。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


from .._core.detector import Finding  # noqa: E402
from .._core.messages import Message, TrajectoryView  # noqa: E402
from .._core.rule_detector import RuleDetector  # noqa: E402

from .._shared._token_patterns import HY_HASHED_TOOL_TOKEN_RE, around  # noqa: E402
from .._shared._leak_shared import (  # noqa: E402
    assistant_channel_texts, unique_match_tokens,
)

SEVERITY = "high"
_FIELD_ORDER = ("content", "reasoning_content", "tool_call")


class ToolCallProtocolLeakHyDetector(RuleDetector):
    detector_id = "tool_call_protocol_leak_hy"
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
            for field, text in assistant_channel_texts(m):
                matches = list(HY_HASHED_TOOL_TOKEN_RE.finditer(text))
                if not matches:
                    continue
                tokens = unique_match_tokens(matches)
                existing = tokens_by_field.setdefault(field, [])
                for token in tokens:
                    if token not in existing:
                        existing.append(token)
                    if token not in all_tokens:
                        all_tokens.append(token)
                if first_match is None:
                    first_match, first_field, first_text = matches[0], field, text
            if first_match is None:
                continue
            fields_hit = [field for field in _FIELD_ORDER if field in tokens_by_field]
            findings.append(Finding(
                rule_id=self.detector_id, severity=SEVERITY, is_issue=True,
                turn_index=i,
                description=(
                    f"assistant 第 {i} 条消息:HY hashed tool 协议标签出现在 "
                    f"{'+'.join(fields_hit)}。"
                ),
                detail={
                    "fields_hit": fields_hit,
                    "tokens_by_field": tokens_by_field,
                    "matched_tokens": list(all_tokens),
                    "first_matched_token": first_match.group(0),
                },
                quotes=[{
                    "kind": self.detector_id, "turn_index": i, "role": "assistant",
                    "field": first_field, "matched_tag": first_match.group(0),
                    "snippet": around(first_text, first_match.start(), first_match.end()),
                }],
            ))
        return findings


__all__ = ["ToolCallProtocolLeakHyDetector"]
