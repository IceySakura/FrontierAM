"""Rule: tool_call_args_invalid (BPO behavior version 1.2.0)."""

from .base import Rule
from ..bpo_adapter import scan_bpo_detector


def scan(ctx):
    return scan_bpo_detector(ctx, 'tool_call_args_invalid')


RULE = Rule('tool_call_args_invalid', '工具参数不符合权威 Schema', scan)
