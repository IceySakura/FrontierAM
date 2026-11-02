"""Rule: tool_call_args_parse_fail."""

from .base import Rule
from ..models import Finding
from ..patterns import ARGS_PARSE_RECEIPT
from ..utils import is_error_result, text


def scan(ctx):
    out = []
    for e in ctx.calls:
        if e.arguments_error:
            out.append(
                Finding(
                    e.message_index,
                    e.arguments_error,
                    ['tool_calls', e.tool_call_index, 'function', 'arguments'],
                    e.function.get('arguments') if e.function else None,
                    e.tool_call_index,
                    e.call_id,
                )
            )
        if e.result_index is not None and is_error_result(e.result) and ARGS_PARSE_RECEIPT.search(text(e.result)):
            out.append(
                Finding(
                    e.result_index,
                    '工具回执明确报告 arguments JSON 解析失败',
                    ['content'],
                    text(e.result),
                    e.tool_call_index,
                    e.call_id,
                    e.message_index,
                )
            )
    return out


RULE = Rule('tool_call_args_parse_fail', '工具调用参数 JSON 解析失败', scan)
