"""Combine BPO self-talk, checkpoint floods and repeated action promises.

Preserve BPO's three channels and apply checkpoint judgments to thinking only.
One finding per message/channel retains evidence from all detector families.
"""

from ._action_selftalk import detect_action_selftalk
from ._checkpoint_reasoning_flood import scan as scan_checkpoint
from .base import Rule
from ..bpo_adapter import scan_bpo_detector
from ..context import DetectionContext
from ..models import Finding


def scan(ctx: DetectionContext) -> list[Finding]:
    findings = scan_bpo_detector(ctx, 'reasoning_selftalk_flood')
    by_channel = {}
    for finding in findings:
        finding.detail['checks'] = ['selftalk_density']
        by_channel[(finding.message_index, finding.detail['channel'])] = finding

    for checkpoint in scan_checkpoint(ctx):
        checks = []
        if checkpoint.detail['filler_window']:
            checks.append('filler_flood')
        if checkpoint.detail['template']:
            checks.append('template_loop')
        key = (checkpoint.message_index, 'reasoning')
        existing = by_channel.get(key)
        if existing is not None:
            existing.detail['checks'].extend(checks)
            existing.detail['checkpoint_reasoning_flood'] = dict(checkpoint.detail)
            existing.message += ' ' + checkpoint.message
            existing.quotes.extend(checkpoint.quotes)
        else:
            checkpoint.detail['checks'] = checks
            checkpoint.detail['checkpoint_reasoning_flood'] = dict(checkpoint.detail)
            findings.append(checkpoint)
            by_channel[key] = checkpoint
    for index, message in ctx.assistant_messages():
        for field in ('thinking', 'reasoning_content', 'reasoning'):
            thinking = message.get(field)
            if isinstance(thinking, str) and thinking.strip():
                break
        else:
            continue
        result = detect_action_selftalk(thinking)
        if result is None:
            continue
        detail = {key: value for key, value in result.items() if key != 'excerpt'}
        quote = {
            'kind': 'action_selftalk_loop',
            'turn_index': index,
            'role': 'assistant',
            'channel': 'reasoning',
            'snippet': result['excerpt'],
        }
        key = (index, 'reasoning')
        existing = by_channel.get(key)
        if existing is not None:
            existing.detail['checks'].append('action_selftalk_loop')
            existing.detail['action_selftalk_loop'] = detail
            existing.quotes.append(quote)
        else:
            detail = {
                'channel': 'reasoning',
                'checks': ['action_selftalk_loop'],
                'action_selftalk_loop': detail,
            }
            finding = Finding(
                message_index=index,
                message=(
                    f'{result["window_chars"]} 字符内有 {result["short_beats"]} 个完整自述短句，'
                    f'其中 {result["action_beats"]} 个反复承诺执行、输出或写入，'
                    f'占 {result["total_beats"]} 个语句的 {result["short_share"]:.0%}。'
                ),
                path=[field],
                excerpt=result['excerpt'],
                severity='medium',
                detail=detail,
                quotes=[quote],
            )
            findings.append(finding)
            by_channel[key] = finding
    return findings


RULE = Rule('reasoning_selftalk_flood', '自述短句或同一句式刷屏', scan)
