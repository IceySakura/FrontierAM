"""Localized model-facing descriptions for Code-MT tools."""

from __future__ import annotations

from typing import Any

from nexus.core import ToolExecutor


CHINESE_TOOL_OVERRIDES: dict[str, dict[str, Any]] = {
    'code_mt/bash': {
        'description': (
            '在运行环境中执行 Bash 命令。可用于运行 shell 命令、切换目录、搜索文件、运行测试以及执行 Python 脚本。'
        ),
        'params': {
            'command': '要执行的 Bash 命令。',
        },
    },
    'code_mt/str_replace_editor': {
        'description': (
            '用于查看、创建和编辑文件的工具，状态会在多次调用之间保留。'
            '当 `path` 指向文件时，`view` 显示 `cat -n` 的结果；当 `path` 指向目录时，'
            '`view` 列出最多两层的非隐藏文件和目录。若指定的 `path` 已经是文件，'
            '则不能使用 `create`。命令输出过长时会被截断，并以 `<response clipped>` 标记。'
            '`undo_edit` 会撤销对 `path` 文件的最近一次编辑。\n\n'
            '使用 `str_replace` 时请注意：\n'
            '* `old_str` 必须与原文件中一行或多行连续内容完全一致，请特别留意空白字符。\n'
            '* 如果 `old_str` 在文件中并不唯一，将不会执行替换；请提供足够上下文以确保唯一。\n'
            '* `new_str` 应包含用于替换 `old_str` 的完整新内容。'
        ),
        'params': {
            'command': ('要执行的操作。可选值：`view`、`create`、`str_replace`、`insert`、`undo_edit`。'),
            'path': '文件或目录的绝对路径，例如 `/testbed/file.py` 或 `/testbed`。',
            'file_text': '执行 `create` 时必填，表示要创建的文件内容。',
            'old_str': '执行 `str_replace` 时必填，表示要在 `path` 中替换的原字符串。',
            'new_str': (
                '执行 `str_replace` 时可选，表示替换后的新字符串；省略时不会添加新内容。'
                '执行 `insert` 时必填，表示要插入的字符串。'
            ),
            'insert_line': ('执行 `insert` 时必填；`new_str` 将插入到 `path` 的第 `insert_line` 行之后。'),
            'view_range': (
                '当 `path` 指向文件并执行 `view` 时可选。省略时显示完整文件；'
                '提供时仅显示指定行号范围，例如 `[11, 12]` 显示第 11 至 12 行。'
                '行号从 1 开始，`[start_line, -1]` 表示从 `start_line` 显示到文件末尾。'
            ),
        },
    },
    'code_mt/submit': {
        'description': (
            '提交当前工作以供评测。完成任务后调用此工具；系统会运行自动验证并返回结果。'
            '如果验证失败，你可能会收到反馈并获得再次尝试的机会。'
        ),
        'params': {
            'explanation': '简要说明已完成的工作。',
        },
    },
    'code_mt/submit_answer': {
        'description': (
            '提交最终答案以供评测。当任务要求书面回答而不是代码修改时使用；'
            '请通过 `answer` 字段提供完整答案。完成调查后调用此工具，提交内容将作为评测对象。'
        ),
        'params': {
            'answer': ('问题的完整最终答案。完成调查并准备提交后调用此工具；你在此字段中传入的答案将作为评测对象。'),
        },
    },
    'code_mt/submit_patch': {
        'description': (
            '以 Git patch 的形式提交修改。第一次调用会展示修改内容供你复核；再次调用 `submit` 以确认并完成提交。'
        ),
        'params': {},
    },
    'code_mt/switch_runtime_image': {
        'description': (
            '当当前镜像无法满足任务要求时，将运行环境切换到新的基础镜像。\n\n'
            '警告：\n'
            '- 当前运行容器会被销毁。\n'
            '- 对话历史会保留在宿主进程中。\n'
            '- 测试环境会通过 `setup_commands` 重新初始化。\n\n'
            '适用场景：\n'
            '- 当前基础镜像缺少必要的系统依赖。\n'
            '- 需要带有预装环境的特定编程语言镜像。\n\n'
            '最多可切换 3 次。'
        ),
        'params': {
            'new_image': '新的基础镜像名称，例如 `python-git:3.11-slim` 或 `node-git:18-bullseye`。',
            'reason': '切换镜像的原因。',
        },
    },
}


def apply_chinese_tool_descriptions(executor: ToolExecutor, tool_names: list[str]) -> None:
    """Replace descriptions for known Code-MT tools without renaming their API."""
    for tool_name in tool_names:
        override = CHINESE_TOOL_OVERRIDES.get(tool_name)
        if override is None:
            continue
        executor.set_tool_override(
            tool_name,
            description=override['description'],
            params=override['params'],
        )
