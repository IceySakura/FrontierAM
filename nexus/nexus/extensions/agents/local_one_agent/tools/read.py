"""local_one_agent read tool.

继承 harness v3 的 ``read_tool_v2``（已含统一 unsupported-file-type 门禁：
office/pdf/.ipynb/binary）。 local 层只做一处覆盖：

- **图片 → 链接文本**（Part Ⅰ §4）：入参是本地路径，但**输出不是本地路径**。
  线上行为是把图片**上传得到 http 短链**，再由 ``web_fetch`` 读取；for_model 放同形状文本。
  （PDF 仍按 v3 砍掉，后续走 skill。）

  TODO(O1)：本期用确定性 mock URL（``cos-mock://{sha1(path)}``）占位，保证训推形状一致。
  待接入**真实上传 + 短链服务**后，替换为真实 http 短链。
"""

from __future__ import annotations

import hashlib
import logging
import os

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.yb_one_agent.tools import (
    BINARY_CHECK_SIZE,
    IMAGE_EXTS,
    ReadInputV2,
    _is_binary_content,
    _make_output,
    _unsupported_extension,
    _unsupported_response,
    maybe_truncate,
    read_tool_v2,
)


logger = logging.getLogger(__name__)


async def _read_via_waa(backend: WaaBackend, params: LocalReadInput) -> ToolResponse:
    """Windows 分支：从 WAA VM 读取文件，渲染成与 v3 一致的 cat -n 形状。"""
    unsupported_ext = _unsupported_extension(params.file_path)
    if unsupported_ext is not None:
        return _unsupported_response(unsupported_ext)

    try:
        content = await backend.read_file(params.file_path)
    except Exception as e:  # noqa: BLE001  读失败（文件不存在等）转成 for_model 错误文本
        logger.warning('local read(waa): failed to read %s: %s', params.file_path, e)
        return ToolResponse(output=_make_output(f'Error: {e}'))

    if _is_binary_content(content.encode('utf-8', errors='replace')[:BINARY_CHECK_SIZE]):
        return _unsupported_response('binary')

    lines = content.splitlines()
    offset = max(params.offset - 1, 0)
    limit = params.limit
    selected = lines[offset : offset + limit]

    formatted = '\n'.join(f'{offset + i + 1:6}\t{line}' for i, line in enumerate(selected))
    if len(lines) > offset + limit:
        formatted += f'\n... ({len(lines) - offset - limit} more lines)'

    return ToolResponse(output=_make_output(maybe_truncate(formatted)))


class LocalReadInput(ReadInputV2):
    pass


def _mock_cos_url(file_path: str) -> str:
    """Deterministic mock COS URL for a local image path."""
    digest = hashlib.sha1(file_path.encode('utf-8')).hexdigest()
    return f'cos-mock://{digest}'


def _render_image_for_model(file_path: str) -> ToolResponse:
    """对齐线上 RenderReadImageForModel 的文本形状（Part Ⅰ §4）。"""
    url = _mock_cos_url(file_path)
    text = (
        f'The file {file_path} is a local image. Its COS URL is: {url}\n'
        'To use this image as input for image_generation, pass type=url with the above URL.'
    )
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={'ok': True, 'file_type': 'image', 'cos_url': url},
    )


@tool_registry.register(
    'local_one_agent/read',
    meta={
        'name': 'read',
        'description': (
            'Reads a file from the filesystem.\n\n'
            'Usage:\n'
            '- file_path must be an absolute path.\n'
            '- By default reads up to 2000 lines from the beginning.\n'
            '- Use offset and limit for large files.\n'
            '- Returns content in cat -n format with line numbers.\n'
            '- Image files return a COS URL; Office/PDF/binary files are not directly readable.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def read_tool(params: LocalReadInput, context: ToolContext) -> ToolResponse:
    """Local read：图片走 COS URL 文本覆盖，其余委托 v3 read_tool_v2。"""
    ext = os.path.splitext(params.file_path)[1].lower()
    if ext in IMAGE_EXTS:
        logger.debug('local read: image → COS URL, path=%s', params.file_path)
        return _render_image_for_model(params.file_path)

    waa_backend = context.tool_state.get('waa_backend')
    if isinstance(waa_backend, WaaBackend):
        logger.debug('local read: forwarding to WAA, path=%s', params.file_path)
        return await _read_via_waa(waa_backend, params)

    v3_params = ReadInputV2(
        description=params.description or 'read file',
        file_path=params.file_path,
        offset=params.offset,
        limit=params.limit,
    )
    return await read_tool_v2(v3_params, context)
