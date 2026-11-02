"""local_one_agent edit tool.

直接继承 harness v3 的 ``edit_tool_v2``，v3 已包含：
- 统一 unsupported-file-type 门禁（.ipynb 等）
- ``old_string == new_string`` → NO_CHANGES
- 单处/多处替换、歧义检测

local 层无额外行为，仅以 ``local_one_agent/edit`` 名注册并放宽 description 为可选。
"""

from __future__ import annotations

import logging

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.yb_one_agent.tools import (
    EditInputV2,
    _make_output,
    _unsupported_extension,
    _unsupported_response,
    edit_tool_v2,
)


logger = logging.getLogger(__name__)


async def _edit_via_waa(backend: WaaBackend, params: LocalEditInput) -> ToolResponse:
    """Windows 分支：读 WAA VM 内文件 → 内存做精确替换 → 写回，逻辑与 v3 edit 一致。"""
    unsupported_ext = _unsupported_extension(params.file_path)
    if unsupported_ext is not None:
        return _unsupported_response(unsupported_ext)

    if params.old_string == params.new_string:
        return ToolResponse(
            output=_make_output('No changes made: old_string and new_string are identical.'),
            meta={'ok': False, 'error': {'code': 'NO_CHANGES'}},
        )

    try:
        content = await backend.read_file(params.file_path)
    except Exception as e:  # noqa: BLE001  读失败转成 for_model 错误文本
        logger.warning('local edit(waa): failed to read %s: %s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))

    if params.old_string not in content:
        return ToolResponse(
            output=_make_output(
                f'Error: old_string not found in {params.file_path}. Make sure it matches exactly including whitespace.'
            )
        )

    if not params.replace_all:
        count = content.count(params.old_string)
        if count > 1:
            return ToolResponse(
                output=_make_output(
                    f'Error: old_string appears {count} times in the file. '
                    'Provide more context to make it unique, or use replace_all=true.'
                )
            )
        new_content = content.replace(params.old_string, params.new_string, 1)
        replacements = 1
    else:
        replacements = content.count(params.old_string)
        new_content = content.replace(params.old_string, params.new_string)

    try:
        await backend.write_file(params.file_path, new_content)
    except Exception as e:  # noqa: BLE001  写回失败转成 for_model 错误文本
        logger.warning('local edit(waa): failed to write %s: %s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))

    return ToolResponse(output=_make_output(f'Successfully edited {params.file_path} ({replacements} replacement(s)).'))


class LocalEditInput(EditInputV2):
    pass


@tool_registry.register(
    'local_one_agent/edit',
    meta={
        'name': 'edit',
        'description': (
            'Performs exact string replacement in a file.\n\n'
            'Usage:\n'
            '- old_string must match the file content exactly (including whitespace and indentation).\n'
            '- old_string must be unique in the file, or use replace_all=true.\n'
            '- The edit will fail if old_string is not found or is ambiguous.\n'
            '- Always read the file with read before editing to get the exact content.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def edit_tool(params: LocalEditInput, context: ToolContext) -> ToolResponse:
    """Local edit：Windows 走 WAA；其余直接委托 v3 edit_tool_v2。"""
    waa_backend = context.tool_state.get('waa_backend')
    if isinstance(waa_backend, WaaBackend):
        logger.debug('local edit: forwarding to WAA, path=%s', params.file_path)
        return await _edit_via_waa(waa_backend, params)

    v3_params = EditInputV2(
        description=params.description or 'edit file',
        file_path=params.file_path,
        old_string=params.old_string,
        new_string=params.new_string,
        replace_all=params.replace_all,
    )
    return await edit_tool_v2(v3_params, context)
