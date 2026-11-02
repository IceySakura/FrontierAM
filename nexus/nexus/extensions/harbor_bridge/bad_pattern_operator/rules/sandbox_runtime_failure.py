"""Rule: sandbox_runtime_failure."""

from .base import Rule
from ..models import Finding
from ..patterns import SANDBOX_FAILURE
from ..utils import text


def scan(ctx):
    out = []
    for e in ctx.calls:
        if (
            e.result_index is not None
            and e.name.casefold() in {'bash', 'shell', 'terminal_run_command', 'exec_command'}
            and SANDBOX_FAILURE.search(text(e.result))
        ):
            out.append(
                Finding(
                    e.result_index,
                    '工具回执显示 sandbox 无法启动或被 OOM 终止',
                    ['content'],
                    text(e.result),
                    e.tool_call_index,
                    e.call_id,
                    e.message_index,
                )
            )
    return out


RULE = Rule('sandbox_runtime_failure', 'Sandbox 运行异常', scan)
