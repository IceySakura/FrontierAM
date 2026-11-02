"""宿主端上下文文件清单（面向 agent）。

``build_context_manifest`` / ``render_manifest`` 生成一段简短的 markdown 列表，
列出已上传到沙箱的文件（路径 + 大小），并被前置到 agent 的 user 消息中。 这是
纯观察性描述——它说明哪些文件存在、位于何处、多大；它不会告诉 agent 去读取它们。
文件 *内容* 从不内联；只有当任务需要该内容时，agent 才会自行读取文件。

如何读取每种类型在系统提示的工具链说明以及每种类型对应的
``/workspace/skills/<type>/SKILL.md`` 中一次性、权威地描述。 清单刻意不对文件进行分类，
也不推荐工具/MCP：文件扩展名在路径中已经可见，此处的每扩展名查找表只是重复——并偏离了
——那唯一的真实来源（历史上它曾固定过时的工具名，并对 .docx 等未列出的类型与系统提示相矛盾）。
未知/无扩展名的文件由 agent 通过 ``file`` 探测。

原始文件本身由 ``engine.task``（参见 ``AGENT_INPUTS_DIR``）上传到 agent 桌面。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 清单（面向 agent）
# ---------------------------------------------------------------------------


def build_context_manifest(paths: list[str], remote_root: str) -> list[dict[str, Any]]:
    """构建 agent 看到的文件列表。

    对每个上下文文件，我们仅报告其远程路径（沙箱内）和大小。 agent 自行读取内容
    ——它从不被内联到 prompt 中，且清单既不对文件分类也不指定工具（参见模块 docstring）。
    """
    manifest: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p).expanduser()
        if not (path.exists() and path.is_file()):
            manifest.append({
                'name': Path(p).name,
                'path': None,
                'size_bytes': 0,
                'note': 'FILE NOT FOUND on host — cannot be uploaded',
            })
            continue
        manifest.append({
            'name': path.name,
            'path': f'{remote_root}/{path.name}',
            'size_bytes': path.stat().st_size,
        })
    return manifest


def render_manifest(manifest: list[dict[str, Any]]) -> str:
    """将 ``manifest`` 格式化为 markdown，前置到 agent 的 user 消息中。"""
    if not manifest:
        return ''
    lines = [
        '## 上下文文件 (Context files)',
        '',
        '以下文件已上传到 sandbox，内容未包含在本 prompt 中。如任务需要，请自行读取相应文件。',
        '',
    ]
    for f in manifest:
        if f['path'] is None:
            lines.append(f'  - `{f["name"]}` — **MISSING** ({f.get("note", "")})')
            continue
        size_kb = f['size_bytes'] / 1024
        lines.append(f'  - `{f["path"]}`  ({size_kb:.1f} KB)')
    return '\n'.join(lines)


__all__ = [
    'build_context_manifest',
    'render_manifest',
]
