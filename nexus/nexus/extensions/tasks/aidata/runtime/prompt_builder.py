"""为 aidata 系统提示词进行沙箱侧上下文探测。

框架自身构建最终的 ``system_prompt`` 字符串并原样交给 ``main_agent`` ——
``main_agent`` 不做任何自身探测，只是把该字符串塞进 ``history[0]`` 作为
系统消息。 本模块是框架用于发现要放入该提示词动态部分的小型辅助层：

* 固定的 agent 沙箱环境；
* 一份工作区目录树 dump（``ls -la /workspace``），并过滤掉引擎私有的
  ``.aidata`` 条目 —— agent 仍能看到自己的真实表面（``desktop``、
  ``skills``、``tests``），但我们不再对外暴露那个曾经诱使 agent 去逆向的
  控制面 scratch 目录；
* 用例（或镜像）在 ``/workspace/skills/<skill>/SKILL.md`` 下安装的每个
  skill 的 SKILL.md frontmatter。

在布局上，框架只是简单地拼接静态基础 preamble、
:meth:`ProjectContext.render` 的输出和一个可选的 task-memory XML
块 —— 参见 :func:`engine.system_prompt.resolve_system_prompt`。
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from typing import Any

from .paths import AGENT_DESKTOP


logger = logging.getLogger(__name__)


# 为遗留 skills 提供的回退描述，这些 skills 未随附 YAML frontmatter 的
# ``description`` 字段。 skill 约定在 frontmatter 中携带 ``name`` +
# ``description``；我们在探测时解析它们，仅在文件缺少元数据时才回退到
# 该字典。
KNOWN_SKILLS: dict[str, str] = {
    'xlsx': 'Spreadsheet editing, formulas, charts, data cleaning',
    'docx': 'Word document creation, editing, formatting',
    'pptx': 'Presentation creation and editing',
    'pdf': 'PDF reading, merging, splitting, form filling, OCR',
    'task_schedule': 'Scheduled task management (RFC 5545 RRULE)',
}


# 限制从 frontmatter 解析出的描述长度。 冗长的 "Trigger on: …" 内容虽然
# 有信息量，但会让缓存的提示词膨胀；~600 字符 ≈ 150 tokens 已足以承载
# 触发关键词。
_SKILL_DESC_MAX_CHARS = 600

# SKILL.md 顶部的 YAML frontmatter 分隔符。
_FRONTMATTER_RE = re.compile(r'\A---\s*\n(.*?)\n---\s*(?:\n|$)', re.DOTALL)

# agent 在沙箱内工作，而不是在运行 ``nexus run`` 的宿主进程内。
# 保持该提示词上下文在所有用例间稳定。
_SANDBOX_PLATFORM = 'Linux'


def _parse_skill_frontmatter(text: str) -> dict[str, str]:
    """从 YAML frontmatter 中提取 ``{key: value}`` 映射。

    刻意 *不* 使用 ``yaml.safe_load``：SKILL.md frontmatter 常常包含
    ``#`` 注释和更严格解析器会拒绝的怪异引号，而我们只需要 ``name`` /
    ``description``。
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}

    result: dict[str, str] = {}
    lines = m.group(1).splitlines()

    current_key: str | None = None
    current_parts: list[str] = []
    block_mode = False
    block_fold = False

    def flush() -> None:
        nonlocal current_key, current_parts, block_mode, block_fold
        if current_key is not None:
            joined = ('\n' if (block_mode and not block_fold) else ' ').join(
                p.strip() for p in current_parts if p.strip()
            )
            if len(joined) >= 2 and joined[0] == joined[-1] and joined[0] in ('"', "'"):
                joined = joined[1:-1]
            result[current_key] = joined.strip()
        current_key = None
        current_parts = []
        block_mode = False
        block_fold = False

    for raw in lines:
        if raw.lstrip().startswith('#'):
            continue
        if current_key is not None and raw.startswith(('  ', '\t')):
            current_parts.append(raw.strip())
            continue
        if not raw.strip():
            if block_mode:
                current_parts.append('')
            continue

        flush()
        if ':' not in raw:
            continue
        key, _, val = raw.partition(':')
        key = key.strip()
        val = val.strip()
        current_key = key
        if val in ('|', '>'):
            block_mode = True
            block_fold = val == '>'
            current_parts = []
        else:
            current_parts = [val] if val else []

    flush()
    return result


@dataclass
class ProjectContext:
    """在静态基础 preamble 之后渲染的每次运行上下文。

    刻意保持精简，从而让 provider 的提示词缓存在同一次运行的多轮迭代间
    仍能命中。
    """

    cwd: str | None = AGENT_DESKTOP
    platform: str | None = _SANDBOX_PLATFORM
    workspace_tree: str | None = None
    skills: list[tuple[str, str]] = field(default_factory=list)  # （路径，描述）
    task_context_lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        """将动态部分渲染为纯 markdown —— 无边界标记，
        无模式标志。 若未发现任何内容则为空。"""
        sections: list[str] = []

        env_lines = ['# Environment']
        if self.cwd:
            env_lines.append(f'- Working directory: {self.cwd}')
        if self.platform:
            env_lines.append(f'- Platform: {self.platform}')
        if len(env_lines) > 1:
            sections.append('\n'.join(env_lines))

        if self.task_context_lines:
            sections.append('# Task context\n' + '\n'.join(f'- {line}' for line in self.task_context_lines))

        if self.workspace_tree:
            sections.append('# Workspace snapshot\n```\n' + self.workspace_tree.rstrip() + '\n```')

        if self.skills:
            skill_lines = [
                'Read the relevant SKILL.md before starting work on that file type:',
                '',
            ]
            for path, desc in self.skills:
                suffix = f' — {desc}' if desc else ''
                skill_lines.append(f'- `{path}/SKILL.md`{suffix}')
            sections.append('# Skills\n' + '\n'.join(skill_lines))

        return '\n\n'.join(sections)


async def discover_project_context(
    runtime: Any,
    *,
    probe_workspace: bool = True,
    task_context_lines: list[str] | None = None,
) -> ProjectContext:
    """构建沙箱提示词上下文，并探测 skills / 工作区目录树。

    所有探测均为尽力而为：任何失败都会静默省略对应部分，从而让
    提示词在运行时不可用时仍能干净地渲染。 ``runtime`` 必须暴露
    一个可 await 的 ``run_command(...)``，其返回对象至少带有
    ``stdout`` 属性。
    """
    pc = ProjectContext(task_context_lines=list(task_context_lines or []))

    if runtime is None:
        return pc

    # skill 发现：发出 **一条** shell 命令，(a) 枚举每个 SKILL.md，
    # (b) 打印其前 40 行并附带哨兵前缀，从而让一次往返即可获取每个
    # skill 的元数据。 frontmatter 的 ``description`` 承载了触发关键词
    # （"Use this skill when…"）；若丢弃它，skill 列表将只剩下不透明的
    # 路径字符串，真实模型往往会忽略它们。
    sentinel = '___AIDATA_SKILL___'
    try:
        probe_cmd = (
            'for d in $(find /workspace/skills -maxdepth 2 -name SKILL.md '
            '-printf "%h\\n" 2>/dev/null | sort); do '
            f'printf "{sentinel} %s\\n" "$d"; '
            'head -n 40 "$d/SKILL.md" 2>/dev/null; '
            'done'
        )
        result = await runtime.run_command(probe_cmd, timeout=15)
        stdout = (getattr(result, 'stdout', '') or '').strip()

        current_path: str | None = None
        buf: list[str] = []

        def _commit() -> None:
            if current_path is None:
                return
            name = current_path.rsplit('/', 1)[-1]
            fm = _parse_skill_frontmatter('\n'.join(buf))
            desc = (fm.get('description') or '').replace('\n', ' ').strip()
            if not desc:
                desc = KNOWN_SKILLS.get(name, '')
            if len(desc) > _SKILL_DESC_MAX_CHARS:
                desc = desc[: _SKILL_DESC_MAX_CHARS - 1].rstrip() + '…'
            pc.skills.append((current_path, desc))

        for line in stdout.splitlines():
            if line.startswith(sentinel):
                _commit()
                current_path = line[len(sentinel) :].strip()
                buf = []
            else:
                buf.append(line)
        _commit()
    except Exception:
        logger.debug('[aidata] failed to probe /workspace/skills', exc_info=True)

    if probe_workspace:
        try:
            # 列出 /workspace，让 agent 看到自己的真实表面（desktop、
            # skills、tests）。 然后丢弃引擎私有的 .aidata 条目：
            # 在系统提示词中提及它，正是曾经诱使 agent 去探测并
            # 逆向运行时控制面的原因。 （.aidata *内容* 已由 libconfine
            # 屏蔽；这里只是不再对外暴露它。）
            result = await runtime.run_command(
                'ls -la /workspace 2>/dev/null | head -50',
                timeout=5,
            )
            stdout = (getattr(result, 'stdout', '') or '').strip()
            if stdout:
                lines = [ln for ln in stdout.splitlines() if ln.split()[-1:] != ['.aidata']]
                pc.workspace_tree = '\n'.join(lines)
        except Exception:
            logger.debug('[aidata] failed to probe /workspace tree')

    return pc


__all__ = [
    'KNOWN_SKILLS',
    'ProjectContext',
    'discover_project_context',
]


# 为单元测试暴露解析器。
_parse_skill_frontmatter_for_tests = _parse_skill_frontmatter
