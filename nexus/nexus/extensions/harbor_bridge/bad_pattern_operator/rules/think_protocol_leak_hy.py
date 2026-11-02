"""Rule: think_protocol_leak_hy (BPO behavior version 1.0.0)."""

from .base import Rule
from ..bpo_adapter import scan_bpo_detector


def scan(ctx):
    return scan_bpo_detector(ctx, 'think_protocol_leak_hy')


RULE = Rule('think_protocol_leak_hy', 'HY think 协议泄漏', scan)
