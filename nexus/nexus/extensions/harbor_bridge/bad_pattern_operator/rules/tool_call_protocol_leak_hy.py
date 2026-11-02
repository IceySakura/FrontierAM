"""Rule: tool_call_protocol_leak_hy (BPO behavior version 1.1.0)."""

from .base import Rule
from ..bpo_adapter import scan_bpo_detector


def scan(ctx):
    return scan_bpo_detector(ctx, 'tool_call_protocol_leak_hy')


RULE = Rule('tool_call_protocol_leak_hy', 'HY tool-call 协议泄漏', scan)
