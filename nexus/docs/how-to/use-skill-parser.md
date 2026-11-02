# Use Skill Parser

本节介绍如何使用 `skill_parser` 模块解析 SKILL.md 文件，从目录中批量加载 Skill 元数据。

## 概述

`skill_parser` 模块（`nexus/utils/skill_parser.py`）用于为 Agent 构建 Skills 能力。 它提供了解析 SKILL.md 文件的功能，从目录中扫描并提取 Skill 的元数据（名称、描述、路径）。

**注意**：该模块只返回 Skill 的元信息，不包含 SKILL.md 文件的完整内容。 Agent 在需要使用某个 Skill 时，应根据返回的 `path` 自行读取文件详情。

### SKILL.md 文件格式

```markdown
---
name: my-skill
description: A brief description of what this skill does
---

# My Skill

Detailed documentation about the skill...
```

## 数据结构

### SkillMetadata

解析成功后返回的 Skill 元数据：

```python
@dataclass
class SkillMetadata:
    name: str         # Skill 名称（最大 64 字符）
    description: str  # Skill 描述（最大 1024 字符）
    path: str         # SKILL.md 文件的绝对路径
```

### SkillParseError

解析失败时的错误信息：

```python
@dataclass
class SkillParseError:
    path: str     # 出错文件的路径
    message: str  # 错误消息
```

### SkillLoadResult

批量加载的结果：

```python
@dataclass
class SkillLoadResult:
    skills: list[SkillMetadata]  # 成功解析的 Skill 列表
    errors: list[SkillParseError]  # 解析错误列表
```

## API 参考

### parse_skill_file

解析单个 SKILL.md 文件：

```python
async def parse_skill_file(skill_path: str) -> SkillMetadata:
    """Parse a single SKILL.md file.

    Args:
        skill_path: SKILL.md 文件的路径

    Returns:
        SkillMetadata: 包含 name、description、path 的元数据

    Raises:
        ValueError: 当文件格式不正确时（缺少 frontmatter、字段缺失或超长）
        FileNotFoundError: 当文件不存在时
    """
```

### load_skills_from_directories

从目录列表中批量扫描并加载所有 SKILL.md 文件：

```python
async def load_skills_from_directories(
    directories: list[str],
    max_depth: int = 6,
    max_dirs: int = 2000,
    follow_symlinks: bool = True,
) -> SkillLoadResult:
    """Scan directory list and return all skill metadata.

    Args:
        directories: 要扫描的目录列表
        max_depth: 最大递归深度（默认 6）
        max_dirs: 最大扫描目录数（默认 2000）
        follow_symlinks: 是否跟随符号链接（默认 True）

    Returns:
        SkillLoadResult: 包含成功解析的 skills 和解析错误的 errors
    """
```

## 在 Docker Runtime 中使用

由于 Skill 文件通常存放在 Docker 容器内，需要通过 `runtime.invoke()` 在容器中执行解析。

### 完整示例

以下示例展示如何在 Docker Runtime 中加载 Skills：

```python
"""Demo script for skill_parser module using Docker runtime.

This script demonstrates:
1. Using identity task with a custom agent defined in the script
2. Agent uses runtime.invoke() to execute skill_parser functions remotely
3. Loading and parsing SKILL.md files from a directory
"""

import asyncio
from nexus.core.agent import AgentRunInput, AgentRunOutput, WithUserPrompt, agent_registry
from nexus.core.context import require_context
from nexus.runner.single import RunInput, run
from nexus.utils.log import setup_logging


setup_logging()


# Configuration
DOCKER_IMAGE = 'mirrors.tencent.com/skill_and_mcp/skill'
SKILL_PATH = '/data/outputs_1_2500/repo/documents'


# ============== Agent Definition ==============


class SkillLoaderAgentInput(AgentRunInput, WithUserPrompt):
    """Skill loader agent input."""

    skill_directories: list[str]
    max_skills_to_show: int = 10


@agent_registry.register('skill_loader_agent')
async def run_skill_loader_agent(agent_input: SkillLoaderAgentInput) -> AgentRunOutput:
    """Agent that loads skills from directories using runtime.invoke."""
    from nexus.utils.skill_parser import load_skills_from_directories

    ctx = require_context()
    runtime = await ctx.get_runtime('main')

    result = await runtime.invoke(load_skills_from_directories, agent_input.skill_directories)

    print(f'\nFound {len(result.skills)} skills, {len(result.errors)} errors\n')

    print('Skills:')
    for i, skill in enumerate(result.skills[: agent_input.max_skills_to_show], 1):
        desc = skill.description[:80] + '...' if len(skill.description) > 80 else skill.description
        print(f'  {i}. {skill.name}')
        print(f'     {desc}')
        print(f'     {skill.path}')

    if len(result.skills) > agent_input.max_skills_to_show:
        print(f'  ... and {len(result.skills) - agent_input.max_skills_to_show} more skills')

    if result.errors:
        print('\nErrors:')
        for error in result.errors[:5]:
            print(f'  - {error.path}: {error.message}')

    return AgentRunOutput()


# ============== Main ==============


async def main():
    """Main entry point for the demo."""
    print('=' * 60)
    print('Skill Parser Demo with Docker Runtime')
    print('=' * 60)
    print(f'Docker Image: {DOCKER_IMAGE}')
    print(f'Skill Path: {SKILL_PATH}')
    print('=' * 60)

    run_input = RunInput(
        imports=['nexus.extensions.tasks.identity'],
        task_name='identity',
        task_config={},
        agent_name='skill_loader_agent',
        agent_config={
            'user_prompt': 'Load skills',
            'skill_directories': [SKILL_PATH],
            'max_skills_to_show': 10,
        },
        runtime_provider_name='docker',
        runtime_provider_config={
            'image': DOCKER_IMAGE,
        },
    )

    output = await run(run_input)

    print('\n' + '=' * 60)
    print(f'Demo completed: {"Success" if output.success else "Failed"}')
    if output.error:
        print(f'Error: {output.error.message}')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(main())
```

## 验证规则

Skill 解析时会进行以下验证：

- **name**：必填，最大 64 字符
- **description**：必填，最大 1024 字符
- **frontmatter**：必须以 `---` 开头和结尾
- **YAML 格式**：frontmatter 必须是有效的 YAML mapping

解析失败时会生成 `SkillParseError`，包含文件路径和错误消息。
