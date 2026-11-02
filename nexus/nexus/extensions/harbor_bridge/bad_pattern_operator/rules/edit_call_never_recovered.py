"""Rule: edit_call_never_recovered."""

from .base import Rule
from ..models import Finding
from ..patterns import EDIT_FAILURE
from ..utils import is_error_result, text


_EDIT_TOOLS = {'edit', 'str_replace', 'filesystem_edit_file', 'apply_patch'}
_OVERWRITE_TOOLS = {'write', 'filesystem_write_file', 'write_file'}


def scan(ctx):
    trailing_failures = {}
    out = []
    for e in ctx.calls:
        name = e.name.casefold()
        if name not in _EDIT_TOOLS | _OVERWRITE_TOOLS or e.arguments is None or e.result is None:
            continue
        path = str(e.arguments.get('file_path') or e.arguments.get('path') or '')
        if not path:
            continue
        if name in _EDIT_TOOLS and is_error_result(e.result) and EDIT_FAILURE.search(text(e.result)):
            trailing_failures.setdefault(path, []).append(e)
        elif not is_error_result(e.result):
            trailing_failures[path] = []
    for path, events in trailing_failures.items():
        if len(events) >= 2:
            e = events[-1]
            out.append(
                Finding(
                    e.result_index,
                    f'文件 {path!r} 连续编辑失败且之后未恢复',
                    ['content'],
                    text(e.result),
                    e.tool_call_index,
                    e.call_id,
                    e.message_index,
                )
            )
    return out


RULE = Rule('edit_call_never_recovered', 'Edit 连续失败且未恢复', scan)
