"""构建 harness 交给 ``main_agent`` 的系统提示词。

``main_agent`` 自身不做任何探测——它只是把我们传入的 ``system_prompt``
字符串塞进 ``history[0]``。 因此 harness 负责组装最终的前置文本：

1. **基础文本** ——要么是 :data:`SYSTEM_PROMPT` （精心策划的 aidata
   默认值），要么，如果案例引导脚本放置了一个，则使用
   :file:`/workspace/system_prompt.md` （沙箱内覆盖）。 覆盖版本获胜，
   因为覆盖被写作默认值的 * 完整替代 * ——通常是那些运行时的默认操作
   原则与实际契约冲突的案例（例如某个 ``wx_agent_hub`` 案例确实需要
   ``ask_user`` 澄清）。

2. **动态上下文** —— :class:`engine.prompt_builder.ProjectContext`
   渲染为纯 markdown。 由 harness 在 ``run_agent`` 之前发现，因此它反映
   引导 + MCP 加载后的当前沙箱。

3. **可选任务记忆** ——当 ``task_memory_enabled`` 时，
   :file:`/workspace/desktop/MEMORY.md` 的 XML 快照。

结果是一个纯 :class:`str`。 没有模式标志，没有边界标记，
没有类包装——只是 agent 将看到的文本。
"""

from __future__ import annotations

import logging

from html import escape as _xml_escape
from typing import Any

from .prompt_builder import ProjectContext


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an autonomous agent operating inside a Linux sandbox with access to an MCP tool catalog spanning shell, filesystem, document handling (xlsx / docx / pptx / pdf), messaging, scheduling, online docs, email, and any task-specific tools injected at runtime. Discover the tool catalog yourself before planning — every service is introspectable.

## Operating principles

- **Take real actions via tools.** Never fabricate a tool's output — if you need a fact, call a tool.
- **Plan before acting.** Emit a brief plan between tool calls.
- **Adapt to failures and delays.** If a tool returns `error` / `pending` / empty / `conflict` / `done`, interpret it and pick the next move.
- **Don't ask the user for clarification.** Everything you need is in the user prompt, the attached context, or reachable via tools.
- **Read the context files yourself.** If the user message lists "Context files", those files are ALREADY uploaded into the sandbox but their CONTENT IS NOT PASTED into the prompt. You must open each one yourself before you can reason about them — use the `pdf-tools` MCP (`read_pdf_pages` / `search_pdf_content` / `get_pdf_info`) for PDFs, shell command for text files, and the appropriate SKILL.md workflow for Office files (.xlsx/.docx/.pptx). Skipping this step means you're working blind.
- **Be concise in your reasoning.** Don't narrate every step.

## Sandbox toolchain hints

- **Office files (.xlsx, .docx, .pptx)** — read the corresponding SKILL.md under `/workspace/skills/{xlsx,docx,pptx}/` before starting. Each skill contains the authoritative workflow, helper scripts, and pitfall warnings for that file type. When a skill references `scripts/…`, run them from the skill directory: `cd /workspace/skills/<type> && python scripts/… <absolute-path-to-file>`.
- **PDFs (.pdf)** — `pdf-tools` MCP (`read_pdf_pages`, `search_pdf_content`, `get_pdf_info`, `extract_pdf_pages`, ...) is the fastest way to read PDFs. For editing, merging, form-filling, OCR, or creating PDFs, see `/workspace/skills/pdf/SKILL.md`.
- **Scheduled tasks / cron** — use the `shell` MCP to run `crontab` commands for creating, listing, or modifying scheduled tasks. For example: `echo '0 9 * * * /path/to/script.sh' | crontab -` to schedule a daily 9 AM job; `crontab -l` to list current entries. Do not attempt to call scheduling tools that are not in your tool catalog.
- **General** — the `shell` MCP wraps `bash -c` with cwd pinned to `/workspace/desktop`; treat that as your working directory.
- **Timeouts** — the default shell timeout is 180s. For heavy `libreoffice` operations (large spreadsheets, 100+ slide decks), pass `timeout: 600` to avoid premature kills.
- **Memory** — `libreoffice --headless` can consume 500 MB+ for large files. Avoid running multiple LibreOffice conversions in parallel or converting excessively large files (>100 MB). If a command is silently killed with no output, memory exhaustion (OOM) is a likely cause.

When the task is complete, call the `finish` tool with a **short** summary of what was produced and where it was saved. The user message below contains (a) a list of context files you must read yourself and (b) your actual task. Deliver concrete artifacts (scheduled sessions, persisted records, uploaded docs, sent emails) rather than prose describing what you would do.
"""


WORKSPACE_SYSTEM_PROMPT_PATH = '/workspace/system_prompt.md'
"""沙箱文件，如果存在且非空，则完全替换该次任务运行的
:data:`SYSTEM_PROMPT`。 由引导脚本写入。 """


WORKSPACE_SYSTEM_PROMPT_READ_TIMEOUT_S = 5
WORKSPACE_TASK_MEMORY_PATH = '/workspace/desktop/MEMORY.md'
WORKSPACE_TASK_MEMORY_READ_TIMEOUT_S = 5


async def _read_workspace_file(runtime: Any, path: str, *, timeout_s: int) -> tuple[bool, str]:
    """尽力对沙箱文件执行 ``cat`` 。 返回 ``(ok, stdout)`` 。"""
    try:
        result = await runtime.run_command(f'cat {path} 2>/dev/null', timeout=timeout_s)
    except Exception as exc:
        logger.debug('[aidata] workspace file read raised for %s: %s', path, exc)
        return False, ''

    if not hasattr(result, 'stdout') and not hasattr(result, 'exit_code'):
        return False, ''

    exit_code = getattr(result, 'exit_code', None)
    stdout = getattr(result, 'stdout', '') or ''
    if exit_code not in (0, None):
        return False, ''
    return True, stdout


async def load_workspace_system_prompt(runtime: Any) -> str | None:
    """返回 :file:`/workspace/system_prompt.md` ，若缺失 / 为空则返回 ``None`` 。"""
    ok, stdout = await _read_workspace_file(
        runtime,
        WORKSPACE_SYSTEM_PROMPT_PATH,
        timeout_s=WORKSPACE_SYSTEM_PROMPT_READ_TIMEOUT_S,
    )
    if not ok:
        return None
    stripped = stdout.strip()
    return stripped or None


async def load_workspace_task_memory(runtime: Any) -> str | None:
    """返回 :file:`/workspace/desktop/MEMORY.md` 。 缺失时返回 ``None`` ；
    存在但为空则返回 ``''`` ，以便调用方仍能发出空 XML。"""
    ok, stdout = await _read_workspace_file(
        runtime,
        WORKSPACE_TASK_MEMORY_PATH,
        timeout_s=WORKSPACE_TASK_MEMORY_READ_TIMEOUT_S,
    )
    if not ok:
        return None
    return stdout.rstrip()


def append_task_memory_xml(prompt_text: str, task_memory: str) -> str:
    """将任务记忆快照作为 XML 块附加到 ``prompt_text`` 后。

    ``task_memory`` 是 agent 放入 ``MEMORY.md`` 的任何内容，因此可能
    包含 ``<`` / ``>`` / ``&`` / 引号，会破坏周围的
    ``<memory_content>...</memory_content>`` 标签。 在插值前对其转义；
    重新将提示词解析为 XML 的下游消费者保持安全，而模型仍能看到可读文本，
    因为大多数消费者会把 XML 实体渲染回普通字符。
    """
    safe_memory = _xml_escape(task_memory, quote=False)
    block = (
        '## Task Memory Snapshot\n\n'
        f'<task_memory type="user_preference_memory" file="{WORKSPACE_TASK_MEMORY_PATH}">\n'
        '<description>\n'
        "This block is the current task's user preference memory.\n"
        'It stores stable user preferences, long-term constraints, and information the user explicitly asked you to remember.\n'
        'Do not treat it as temporary working state, step-by-step plans, or short-lived intermediate results.\n'
        'This XML block is only a startup snapshot; if you edit the memory file later, re-read the file to get the latest content.\n'
        '</description>\n'
        '<memory_content>\n'
        f'{safe_memory}\n'
        '</memory_content>\n'
        '</task_memory>'
    )
    return f'{prompt_text}\n\n{block}'


async def resolve_system_prompt(
    runtime: Any,
    *,
    project_context: ProjectContext,
    task_memory_enabled: bool = False,
) -> str:
    """为 ``main_agent`` 组装完整的系统提示词字符串。

    顺序：基础前置文本（若存在则为覆盖，否则为 :data:`SYSTEM_PROMPT` ）
    -> 动态 env / skills / workspace 段落 -> 可选的任务记忆
    XML 块。 结果直接输入 ``main_agent.system_prompt`` ，不再进一步包装。
    """
    override = await load_workspace_system_prompt(runtime)
    if override is not None:
        logger.debug(
            '[aidata] using workspace system prompt override (%d chars) from %s',
            len(override),
            WORKSPACE_SYSTEM_PROMPT_PATH,
        )
        base = override
    else:
        base = SYSTEM_PROMPT

    parts: list[str] = [base.rstrip()]
    dynamic = project_context.render().strip()
    if dynamic:
        parts.append(dynamic)
    text = '\n\n'.join(parts)

    if task_memory_enabled:
        task_memory = await load_workspace_task_memory(runtime)
        if task_memory is not None:
            text = append_task_memory_xml(text, task_memory)

    return text


__all__ = [
    'SYSTEM_PROMPT',
    'WORKSPACE_SYSTEM_PROMPT_PATH',
    'WORKSPACE_SYSTEM_PROMPT_READ_TIMEOUT_S',
    'WORKSPACE_TASK_MEMORY_PATH',
    'WORKSPACE_TASK_MEMORY_READ_TIMEOUT_S',
    'append_task_memory_xml',
    'load_workspace_system_prompt',
    'load_workspace_task_memory',
    'resolve_system_prompt',
]
