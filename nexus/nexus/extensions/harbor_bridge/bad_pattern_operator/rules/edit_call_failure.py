"""Rule: edit_call_failure."""

from .base import Rule
from ..models import Finding
from ..patterns import EDIT_FAILURE
from ..utils import is_error_result, text


def scan(ctx):
    consecutive_failures = {}
    out = []
    for e in ctx.calls:
        if (
            e.name.casefold() not in {'edit', 'str_replace', 'filesystem_edit_file', 'apply_patch'}
            or e.arguments is None
            or e.result is None
        ):
            continue
        path = str(e.arguments.get('file_path') or e.arguments.get('path') or '')
        if not path:
            continue
        if is_error_result(e.result) and EDIT_FAILURE.search(text(e.result)):
            streak = consecutive_failures.get(path, 0) + 1
            consecutive_failures[path] = streak
            if streak == 2:
                out.append(
                    Finding(
                        e.result_index,
                        f'同一文件 {path!r} 的编辑连续失败',
                        ['content'],
                        text(e.result),
                        e.tool_call_index,
                        e.call_id,
                        e.message_index,
                    )
                )
        else:
            consecutive_failures[path] = 0
    return out


RULE = Rule('edit_call_failure', 'Edit 反复调用失败', scan)
