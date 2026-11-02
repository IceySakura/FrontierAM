# Nexus Agent 开发手册

本手册指导开发者如何基于 Nexus 的 **Object-Oriented Agent (OOA)** 范式构建生产级智能体。

Nexus 提供了一套经过验证的 Agent 继承体系。通过继承基类，你可以专注于**业务状态管理**、**Prompt 工程**和**工具组合**，而无需重复编写底层的执行循环、记忆管理和 LLM 交互逻辑。

## 目录

1. [Agent 继承体系](#agent-继承体系)
2. [选择合适的基类](#选择合适的基类)
3. [教程：构建 ToolCallAgent](#教程构建-toolcallagent)
4. [教程：构建 DeepAgent](#教程构建-deepagent)
5. [API 速查表](#api-速查表)

## Agent 继承体系

Nexus 定义了层级清晰的 Agent 基类。开发者通常只需要继承 `ToolCallAgent`，通过定义不同的 Prompt 和 Tools 组合即可实现大部分业务逻辑。当前实现中，`ToolCallAgent` 直接继承 `BaseAgent`，并在内部采用 `think() -> act()` 的思考-执行循环。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent 继承层级                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   BaseAgent                                                             │
│   └── 定义 multi-step 执行流，所有 Agent 通过 run() 方法执行。               │
│         │                                                               │
│         ▼                                                               │
│   ToolCallAgent                                                         │
│   └── 核心生产基类。直接继承 BaseAgent，具备标准工具调用能力，                  │
│       并内置 think() -> act() 的思考-执行循环。                               │
│         │                                                               │
│         ▼                                                               │
│   DeepAgent                                                             │
│   └── 通用 Agent 抽象。具备任务规划、上下文管理及SubAgent 委派能力。            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## DeepAgent 能力对标

DeepAgent 是 Nexus 框架中功能最强大的抽象类，其设计目标是对齐业界最先进的通用 Agent 框架（如 Manus 和 Claude Code）。

| Feature | Manus | Claude Code | **DeepAgent (Nexus)** |
| :--- | :--- | :--- | :--- |
| **Tool Integration** | 集成 Search, Code, Browser, FileSystem 等 | 集成 Search, Code, Browser, FileSystem 等 | **全量集成** Search, Code, Browser, FileSystem, Delegate, TODOList 等 |
| **Task Delegation** | 支持任务拆分与子模块执行 | 支持任务委托给 SubAgent | **支持** 任务拆分与 SubAgent 委派 (Orchestration) |
| **Context Engineering** | 文件系统作为 Memory / 上下文存储 | 引入 TODOList 追踪任务步骤；文件系统作为 Memory | **多维上下文**：文件系统 Memory + Memory Tool + TODOList 状态追踪 |
| **Status** | 完整实现 | 完整实现 | **抽象层已就绪** (Reference Implementation 处于 Demo 阶段) |

## 选择合适的基类

根据你的业务需求选择合适的 Agent 基类：

| 基类 | 适用场景 | 关键能力 |
| :--- | :--- | :--- |
| **ToolCallAgent** | **推荐默认**。适用于 90% 的业务场景，包括数据查询、API 操作、代码执行等。 | 自动化 Tool Calling 循环 / 内置重试机制 / 兼容 OpenAI Tools 协议 |
| **DeepAgent** | **复杂任务**。适用于需要任务委派、长期记忆或多步骤状态追踪的场景。 | SubAgent 委派 / TODO 状态追踪 / 继承 ToolCallAgent 全部能力 |
| **BaseAgent** | **底层开发**。适用于构建全新架构。 | 基础状态机 / Memory 管理 / 执行循环控制 |

-----

## 教程：构建 ToolCallAgent

我们以 **SWEAgent** 为参考示例，展示如何继承 `ToolCallAgent` 构建一个专注于代码修复的智能体。

> **示例说明**：此处的代码旨在展示如何通过覆盖钩子函数来控制 Agent 行为，并非生产环境的完整代码。

### 1\. 定义输入输出

继承 `ToolCallAgentInput` 和 `ToolCallAgentOutput`，添加业务字段：

```python
class SWEAgentInput(ToolCallAgentInput):
    working_dir: str = Field(default='/testbed', description='工作目录')
    problem_statement: str | None = Field(default=None, description='Issue 描述')


class SWEAgentOutput(ToolCallAgentOutput):
    exit_status: ExitStatus = Field(default=ExitStatus.FINISHED, description='退出状态')
```

### 2\. 实现 Agent 逻辑

核心在于重写 `on_tool_executed` 方法。SWEAgent 的特点是将工具的执行结果包装为特定的 Observation 格式，并监听特定的退出信号。

```python
class SWEAgent(ToolCallAgent):
    name: str = 'swe'
    system_prompt: str = SWE_AGENT_SYSTEM_PROMPT

    # 定义业务状态
    exit_status: ExitStatus = Field(default=ExitStatus.FINISHED)

    def on_tool_executed(self, response: ToolResponse) -> MessageContent:
        """
        核心扩展点：
        1. 监听工具副作用 (Side Effects)
        2. 格式化工具输出 (Observation Formatting)
        """
        # 1. 检查 submit 工具是否设置了退出信号
        if response.context_updates.get('exit_status') == 'submitted':
            self.exit_status = ExitStatus.SUBMITTED

        # 2. 将工具输出包装为 LLM 更容易理解的格式
        raw_output = self._extract_text(response.output)
        return SWE_AGENT_NEXT_STEP_PROMPT.format(observation=raw_output) if raw_output else SWE_AGENT_NO_OUTPUT_PROMPT
```

### 3\. 注册 Agent

```python
@agent_registry.register('swe_agent')
async def run_swe_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
    # 1. 准备工具
    executor = ToolExecutor([
        tool_registry.get('swe_agent/bash'),
        tool_registry.get('swe_agent/str_replace_editor'),
        tool_registry.get('swe_agent/submit'),
    ])

    # 2. 创建实例 (finish_tool_names 使得调用 submit 后自动停止循环)
    agent = SWEAgent.create(
        agent_input,
        executor=executor,
        finish_tool_names=['submit'],
    )

    # 3. 运行
    await agent.run(agent_input.user_prompt)

    return agent.get_output()
```

-----

## 教程：构建 DeepAgent

`DeepAgent` 继承自 `ToolCallAgent`，在保留标准工具调用能力的基础上，额外提供了**内置 TODO 状态追踪**和**SubAgent 委派机制**。

当你的任务涉及以下场景时，推荐使用 `DeepAgent`：
- **多步骤任务规划**：需要维护 TODO 列表来追踪子任务完成状态
- **任务编排 (Orchestration)**：需要将复杂任务委派给专业的子智能体
- **持久化环境状态**：需要在多轮对话中追踪工作目录、打开的文件等上下文

我们以 **OmniAgent** 为参考示例，展示如何构建一个集成了多工具与状态追踪的通用智能体。

### 1\. 定义输入输出

继承 `DeepAgentInput` 和 `DeepAgentOutput`，添加需要持久化追踪的环境状态字段：

```python
from nexus.extensions.agents.base import DeepAgentInput, DeepAgentOutput


class OmniAgentInput(DeepAgentInput):
    """DeepAgentInput 已内置 user_prompt"""

    working_dir: str = Field(default='/workspace', description='当前工作目录')


class OmniAgentOutput(DeepAgentOutput):
    """DeepAgentOutput 已内置 todos 和 main_trajectory"""

    working_dir: str = Field(default='/workspace', description='最终工作目录')
    open_file: Optional[str] = Field(default=None, description='当前打开的文件')
```

### 2\. 实现 Agent 逻辑

核心在于重写 `on_tool_executed`。与 ToolCallAgent 不同，DeepAgent 需要先调用 `super()` 来处理内置的 TODO 列表更新，然后再处理自定义状态。

```python
class OmniAgent(DeepAgent):
    name: str = 'omni'
    
    # 定义环境状态 (DeepAgent 已内置 todos)
    working_dir: str = Field(default='/workspace')
    open_file: Optional[str] = Field(default=None)

    def on_tool_executed(self, response: ToolResponse) -> MessageContent:
        """
        核心扩展点：
        1. 调用父类处理 TODOs 更新
        2. 从 context_updates 同步环境状态
        3. 动态构建 Observation
        """
        # 1. 让 DeepAgent 父类处理 TODO 列表更新 (Critical)
        super().on_tool_executed(response)

        # 2. 更新自定义状态 (由工具通过 context_updates 返回)
        if 'working_dir' in response.context_updates:
            self.working_dir = response.context_updates['working_dir']
        if 'open_file' in response.context_updates:
            self.open_file = response.context_updates['open_file']

        # 3. 将当前状态注入到 Observation 中，让 LLM 始终感知环境
        raw_output = self._extract_text(response.output)
        return self._format_with_state(raw_output)

    def _format_with_state(self, observation: str) -> str:
        state_block = f"""CURRENT STATE:
- Working Directory: {self.working_dir}
- Open File: {self.open_file or 'None'}
- TODOs: {len(self.todos)} items
"""
        return f"{state_block}\nOBSERVATION:\n{observation}\n\nWhat's your next step?"
```

### 3\. 注册 Agent

DeepAgent 支持通过 `subagents` 参数启用任务委派机制。

```python
@agent_registry.register('omni_agent')
async def run_omni_agent(agent_input: OmniAgentInput) -> OmniAgentOutput:
    # 1. 准备工具 (支持多种执行环境)
    executor = ToolExecutor([
        tool_registry.get('bash'),
        tool_registry.get('python_repl'),
        tool_registry.get('str_replace_editor'),
    ])

    # 2. 创建实例
    agent = OmniAgent.create(agent_input, executor=executor)

    # 3. (可选) 配置子智能体委派
    # agent.subagents = [researcher, coder]
    # agent.setup_delegation()  # 启用 Orchestrator 模式

    # 4. 运行
    await agent.run(agent_input.user_prompt)

    return agent.get_output()
```

-----

## API 速查表

### 常用导入

```python
from nexus.extensions.agents.base import (
    ToolCallAgent,
    ToolCallAgentInput,
    ToolCallAgentOutput,
    DeepAgent,
    DeepAgentInput,
    DeepAgentOutput,
)
from nexus.core import ToolExecutor, ToolResponse, agent_registry
```

### 生命周期钩子 (Hooks)

| 方法 | 所属类 | 作用 | 常见用途 |
| :--- | :--- | :--- | :--- |
| `on_tool_executed` | ToolCallAgent | 工具执行后回调 | 格式化 Observation、更新 Agent 内部状态、处理副作用 |
| `get_final_answer` | BaseAgent | 结束前回调 | 处理超时逻辑、整理最终输出格式 |
| `think` | ToolCallAgent | 思考阶段 | 自定义 LLM 调用逻辑、插入特殊 Prompt、决定是否进入工具执行 |
| `act` | ToolCallAgent | 行动阶段 | 执行已选择的工具，或重写为自定义动作逻辑 |
| `step` | BaseAgent | 单步循环 | 完全接管单步执行逻辑 (通常不需要修改) |

### 工厂方法参数 (`Agent.create`)

| 参数 | 说明 |
| :--- | :--- |
| `executor` | `ToolExecutor` 实例，包含该 Agent 可用的所有工具。 |
| `finish_tool_names` | `list[str]`，调用列表中的工具将触发 Agent 立即终止 (如 `submit`, `finish`)。 |
| `max_iterations` | `int`，最大对话轮数，防止死循环。 |
| `system_prompt` | `str`，覆盖默认的系统提示词。 |
