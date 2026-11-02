"""Rule: final_turn_language backed by BPO zh_ask_foreign_reply v1.0.0."""

from .base import Rule
from ..bpo_adapter import scan_bpo_detector


def scan(ctx):
    return scan_bpo_detector(ctx, 'zh_ask_foreign_reply', exposed_rule_id='final_turn_language')


RULE = Rule('final_turn_language', '中文问外文答', scan)
