# 创建新模块

本文介绍如何创建一个 Nexus 扩展模块，包括元数据声明、依赖配置和对象注册。

## 目录模块

大多数模块是 **目录模块**——一个包含 `pyproject.toml` 的 Python package。

### 最小示例

假设要创建一个名为 `my_agent` 的 Agent 模块：

```
nexus/extensions/agents/my_agent/
├── __init__.py        # 可以为空
├── agent.py           # Agent 实现
└── pyproject.toml     # 模块元数据
```

`pyproject.toml`：

```toml
[project]
name = "my_agent"
version = "0.1.0"
description = "My custom agent"
dependencies = []

[tool.nexus]
agents = { my_agent = "nexus.extensions.agents.my_agent.agent" }
```

`agent.py`：

```python
from nexus.core.registry import agent_registry
from nexus.extensions.agents.base.schema import AgentRunInput, AgentRunOutput


@agent_registry.register('my_agent')
async def run_my_agent(agent_input: AgentRunInput) -> AgentRunOutput:
    # 你的 Agent 逻辑
    ...
```

### 带依赖的示例

如果模块有依赖，按三层模型声明：

```toml
[project]
name = "my_agent"
version = "0.1.0"
description = "Agent with dependencies"

# Base 依赖：Host 和 Runtime 都需要
dependencies = [
    "pyyaml>=6.0",
]

# 按环境分离的依赖
[project.optional-dependencies]
# Host 侧独有依赖（如数据集处理、评估脚本）
host = [
    "datasets>=2.0",
]
# Runtime 侧独有依赖（如沙盒内需要的工具库）
runtime = [
    "tree-sitter>=0.20",
]

[tool.nexus]
agents = { my_agent = "nexus.extensions.agents.my_agent.agent" }
```

### 注册多个对象

一个模块可以同时注册 agents、tools、tasks 等多种类型：

```toml
[tool.nexus]
agents = { my_agent = "nexus.extensions.agents.my_agent.agent" }

# 多个 tool 使用子表格式
[tool.nexus.tools]
"my_agent/bash" = "nexus.extensions.agents.my_agent.tools"
"my_agent/search" = "nexus.extensions.agents.my_agent.tools"
```

!!! note "TOML 格式注意"

    TOML 的 inline table（`{ k = "v" }`）**不能**跨行。 当注册多个对象时，使用 `[tool.nexus.tools]` 子表格式。

## 单文件模块

对于简单的模块，可以使用 PEP 723 inline script metadata，无需 `pyproject.toml`：

```python
# /// script
# dependencies = ["requests>=2.28"]
#
# [tool.nexus]
# tasks = { my_task = "nexus.extensions.tasks.my_task" }
# ///

from nexus.core.registry import task_registry


@task_registry.register('my_task')
async def my_task(task_input): ...
```

单文件模块适合轻量级扩展，如简单的 task 或 processor。

## 对象注册

### `[tool.nexus]` 字段映射

| TOML 键 | Registry 类型 | 说明 |
|---|---|---|
| `agents` | `agent_registry` | Agent 实现 |
| `tasks` | `task_registry` | Task 实现 |
| `tools` | `tool_registry` | Tool 实现 |
| `executors` | `executor_registry` | Executor 实现 |
| `openai_clients` | `openai_client_registry` | LLM 客户端 |
| `runtime_providers` | `runtime_provider_registry` | Runtime Provider |
| `span_processors` | `span_processor_registry` | Span Processor |
| `run_parts` | `run_part_registry` | `nexus run --use` 的 partial `RunInput` |

### 值格式

值是 `import_path` 字符串，指向包含 `@registry.register()` 装饰器的模块。 Discovery 时只记录 import path，**不导入代码**——只有在首次 `registry.get()` 时才真正导入。

```toml
# 值指向模块路径，不是具体函数
agents = { my_agent = "nexus.extensions.agents.my_agent.agent" }
# 当 registry.get('my_agent') 被调用时，才会 import nexus.extensions.agents.my_agent.agent
```

Run part 也是 registry 对象。 常见做法是在模块内放一个 `run_parts.py`，导出官方推荐配置：

```toml
[tool.nexus]
agents = { my_agent = "nexus.extensions.agents.my_agent.agent" }
run_parts = { my_agent = "nexus.extensions.agents.my_agent.run_parts" }
```

```python
from typing import Any

from nexus.runner.compose import run_part


@run_part('my_agent')
def my_agent_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part my_agent does not accept parameters: {sorted(params)}')
    return {
        'agent_name': 'my_agent',
        'agent_config': {},
    }
```

## 模块放置位置

| 类型 | 目录 |
|---|---|
| Agent | `nexus/extensions/agents/<name>/` |
| Task | `nexus/extensions/tasks/<name>/` |
| Tool | `nexus/extensions/tools/<name>/` |
| Agent + Task 组合 | `nexus/extensions/projects/<name>/` |
| OpenAI Client | `nexus/extensions/openai_clients/<name>.py` |
| Runtime Provider | `nexus/runtime/providers/<name>/` |
| Executor | `nexus/runtime/executors/<name>.py` |
| Span Processor | `nexus/trace/processors/<name>.py` |

## 验证模块

创建完成后，运行以下命令验证模块是否被正确发现：

```bash
nexus modules
```

你的模块应该出现在对应类别下。 如果依赖未安装，`Host` 或 `Runtime` 列会显示 `✗`——使用 `nexus modules install <name>` 安装。

## 完整示例

以下是 `swe_agent` 模块的 `pyproject.toml`，展示了一个典型的 Agent + Tools 组合：

```toml
[project]
name = "swe_agent"
version = "0.1.0"
description = "SWE-Agent: Software Engineering Agent"
dependencies = []

[project.optional-dependencies]
host = []
runtime = []

[tool.nexus]
agents = { swe_agent = "nexus.extensions.agents.swe_agent.agent" }

[tool.nexus.tools]
"swe_agent/bash" = "nexus.extensions.agents.swe_agent.tools"
"swe_agent/str_replace_editor" = "nexus.extensions.agents.swe_agent.tools"
"swe_agent/review_submit" = "nexus.extensions.agents.swe_agent.tools"
```
