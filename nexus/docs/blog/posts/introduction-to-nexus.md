---
date: 2026-01-22
readtime: 15
authors:
  - tarschen
---

# Introduction to Nexus

Nexus 是一套面向研究场景的 Agent Infra 框架，解决模型与外部世界交互的问题——沙盒环境、工具调用、状态管理、轨迹记录。 本文从 Agent Infra 的设计动机出发，介绍 Nexus 的整体架构和核心组件。

<!-- more -->

## Why Agent Infra

在 LLM Research 领域， Infra 建设主要围绕两大方向展开： **Training Infra** 和 **Inference Infra**。 2024 年的其中一个核心主题是 **Long Reasoning** （长推理）—— 从 OpenAI o1 的发布开始，业界掀起了一波通过延长推理链来提升模型能力的热潮。 无论是训练侧的 RL scaling（强化学习扩展），还是推理侧的 test-time compute（测试时计算），核心思路都是用更多计算换更好结果。

进入 2025 年，业界逐渐意识到单纯依赖 Reasoning 通往 AGI 的道路已遇瓶颈。虽然 Long reasoning 在数学与算法等结构化任务上表现优异，但本质上仍属于单轮、无状态的「输入→思考→输出」过程，无法改变外部世界或者从外部环境获取更多信息，也很难泛化提升各类实际使用场景的体验。 这催生了向 Agent 的根本性范式转变：模型不再局限于生成文本，而是进入了多轮、有状态的交互模式。在开放世界的复杂任务中，Agent 能够调用工具、执行命令、观察反馈并据此迭代行动——无论是修改文件、管理进程还是操作浏览器，每一次交互都在实质性地改变环境状态，从而补足了模型在实时感知与执行能力上的短板。

这种交互模式的转变，直接催生了对 **沙盒环境** 的需求。 简单任务或许可以用 **无状态沙盒** 应对 —— 每次工具调用启动一个全新的容器，执行完毕即销毁，天然隔离且易于扩展。 但真实的 Agent 任务往往需要 **有状态沙盒**：模型在第一轮创建了一个文件，第二轮需要读取它；第三轮启动了一个服务，第四轮需要向它发送请求。 状态必须在多轮交互之间延续。 这带来了一系列 Infra 挑战：沙盒的生命周期如何管理？ 长时间运行的会话如何处理超时和异常？ 大规模并发时如何高效调度资源？ 这些问题在 Reasoning 时代根本不存在，却是 Agent 时代必须解决的核心难题。

除此之外， **模型与环境之间的交互设计** 同样是 Agent 场景的关键组成部分。 模型通过什么接口与环境交互？ 工具的 Action Space 如何定义才能既表达力强又易于学习？ 执行结果以什么形式反馈给模型 —— 是原始的 stdout/stderr，还是结构化的 JSON，还是经过截断和摘要的文本？ 对于 GUI 操作，是返回截图、DOM 树、还是 Accessibility Tree？ 这些设计决策直接影响模型的任务完成能力。 好的交互设计能让模型更准确地理解环境状态、更高效地规划下一步行动；而糟糕的设计则会导致信息丢失、上下文膨胀、甚至模型陷入死循环。

这些问题带来了一个全新的 Infra 需求： **Agent Infra**。 不同于 Training/Inference Infra 主要关注模型本身的计算，Agent Infra 需要解决的是模型与外部世界交互的问题 —— 沙盒环境、工具调用、状态管理、轨迹记录等。 这是一个在传统 LLM Infra 之外需要单独构建的新模块。

Agent 时代的 LLM 对于在应用场景的优化有着迫切需求。 我们需要通过往模型里压缩更多实际使用场景的信息来增强体验，这也意味着我们进入了一种应用驱动的研究范式（Application-driven Research）。 Agent Infra 正是支撑这一范式的基础设施：它让研究者能够在真实环境中运行 Agent、收集轨迹数据、评估任务完成度，从而形成「应用 → 数据 → 训练 → 更好的应用」的闭环。

## Designing Agent Infra

如何设计一个完善的 Agent Infra？ 本节从 Infra 的视角出发，分析现有 Agent 产品的特点、研究场景的需求，以及如何用一个框架将两者结合起来。

### Agent Side

不同的 Agent 产品，操作的对象不同：有的调用 API，有的操作浏览器，有的执行终端命令，有的直接控制桌面。 操作层级越底层，需要的运行环境就越重。 我们可以把当前的 Agent 产品分为五层：

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/introduction-to-nexus/action-surface-layers.svg" style="width: 100%"></object><figcaption>Agent 产品按操作层级分类</figcaption></figure>

**1. Tool / API 层**

最轻量的一层。 Agent 通过结构化的工具调用（如 WebSearch、MCP 连接器、代码执行器等）完成任务，不直接操作 UI。 典型产品包括 **OpenAI Deep Research**，它调用 web search 工具来生成深度研究报告。 这类 Agent 不需要额外的虚拟化，应用级别即可运行。

**2. Browser DOM 层**

Agent 操作浏览器页面元素，包括导航、表单填写、点击、下载等。 工具链通常基于 Playwright/CDP/Selenium。 代表产品有 **browser-use** （开源浏览器自动化框架）和 **Manus Browser Operator** （将浏览器变成自主 Agent 的扩展）。 只需要一个可控的浏览器实例，不需要容器或虚拟机。

**3. Shell / FS 层**

Agent 操作文件系统和终端命令，是当前 **Coding Agent** 的主战场。 这一层的产品最为丰富：

- **CLI 形态**：Claude Code、OpenAI Codex CLI、OpenCode、Gemini CLI 等，直接在终端中运行，执行代码读写、git 操作、测试构建等任务
- **IDE 形态**：Cursor、VS Code Copilot Agent Mode、Cline 等，虽然包装在 GUI 中，但核心操作仍是文件和终端
- **云端 Devbox**：Devin 采用 VM 快照模式，每个 session 加载独立的虚拟环境，集成 shell/editor/browser 能力

这一层用容器就能跑，但如果要复现完整的 IDE 交互体验则需要虚拟机。

**4. OS GUI 层**

Agent 通过屏幕截图理解界面，再注入鼠标键盘操作，实现跨应用的桌面/手机自动化：

- **桌面端**：**GLM-PC** （智谱）支持 macOS/Windows， **UI-TARS/Agent TARS** （字节）覆盖三大桌面系统，均通过截图理解 + 坐标定位 + 键鼠操作实现跨应用自动化
- **移动端**：**豆包手机助手** 在 Android 上实现 GUI Agent + 系统工具调用，目前需搭载专用硬件（nubia M153）使用

这一层需要完整的桌面/窗口系统，通常要用虚拟机。

**5. Hardware I/O 层**

涉及电话、蓝牙、传感器、NFC、专有 ROM 等能力，无法用软件模拟，必须用真机或设备农场。 典型场景包括 iOS 系统级 Agent、以及手机助手的高保真评测。

从 Infra 视角，这五层对应四种运行环境，成本递增：应用 → 容器 → 虚拟机 → 真机/设备农场。 理解这个分层，有助于我们为不同类型的 Agent 设计合适的运行时环境和评测方案。

### Researcher Side

上述 Agent 产品都是在 **定义 Agent** —— 定义工具、定义工作流、定义交互方式。 但到了 Researcher 这边，为了提升模型水平，我们需要建立 **端到端的任务**，并且是可验证工作结果对错的任务，用来指导大模型和 Agent 设计的迭代优化。

这也验证了前面提到的，我们进入了一个应用驱动的研究范式。 结合对工具层级的分类，可以归纳出每一层产品对应的研究评估集：

<div class="center-table" markdown>

| 层级 | 代表评估集 |
|------|-----------|
| Tool / API | GAIA、BrowserComp |
| Browser DOM | WebArena、Mind2Web、VisualWebArena |
| Shell / FS | SWE-Bench、Terminal Bench |
| OS GUI | OSWorld、AndroidWorld |
| Hardware I/O | 目前缺乏标准化评估集 |

</div>

它们都是以解决真实场景问题为目标设立的。

从另一个正交的视角看，Infra 需要支持一系列的研究使用场景：

- **评估（Evaluation）**：在标准化环境中运行 Agent，测量成功率、效率等指标
- **数据生成（Pretrain/Midtrain/SFT/RFT）**：收集高质量轨迹数据用于训练
- **强化学习（RL）**：让 Agent 在环境中试错，获取奖励信号
- **体验测试**：人工试用 Agent 产品，观察实际表现
- **数据标注**：对轨迹进行人工标注，构建训练数据集

这些场景看似不同，但本质是一回事： 让 Agent 在某个环境中执行任务，记录完整轨迹，并评判执行效果。 区别只在于： 评估场景需要管理数据集和分数； 强化学习需要更详细的 Reward 设计以及训练和环境的稳定性； 体验测试需要框架往真实产品方向发展； 数据标注需要更灵活的对话控制。

这就引出了我们的第一层设计： **以 Agent 在各类环境下执行任务为核心，支持收集这些任务执行过程中产生的产物（对应轨迹、分数等），并支持以不同的方式启动这些任务（对应评估、RL 等不同使用场景）** 。

### Infra Side

结合前面的分析，我们可以自然地推导出 Infra 需要的四个核心抽象：

**Task**：对应研究侧的「端到端任务」。 定义任务描述、初始状态、成功条件，是可验证结果对错的最小单元。 不同的评估集（SWE-Bench、OSWorld 等）本质上是一组 Task 的集合。

**Agent**：对应产品侧的 Agent 实现。 封装 LLM 调用、工具选择、多轮交互等行为逻辑。 我们倾向于 **白盒复刻** 而非黑盒调用 —— 现有产品面向用户会做兜底处理，不利于暴露模型真实能力；黑盒调用也难以插入自定义指标和 reward。 复刻成本通常不高，社区已有先例：Manus 于 2025 年 3 月 6 日走红，OpenManus 同日发布，原型仅用 3 小时完成；Claude Cowork 于 2026 年 1 月 12 日上线，OpenWork 两天后发布；Devin 于 2024 年 3 月 12 日发布，OpenDevin（现 OpenHands）两周后发布首个可用版本。

**Tool**：Agent 可调用的原子操作，对应强化学习中的 Action Space。 不同 Agent 产品的工具设计各有特色，Tool 抽象强调保留 Agent 原有的工具定义，而非强制统一。 为了支撑不同 Tool 的执行需求，我们再抽象一层 Runtime 接口。

**Runtime**：提供工具执行的沙盒环境，是一个统一且全能的底层接口。 不同的工具层级需要不同的 Runtime 实现：

<div class="center-table" markdown>

| 工具层级 | Runtime 实现 | 说明 |
|---------|-------------|------|
| Tool / API | 应用进程 | 直接在 Agent 进程内执行，无需隔离 |
| Browser DOM | 浏览器实例 | Playwright/CDP 控制的浏览器 |
| Shell / FS | 容器 | Docker 容器提供文件系统和终端隔离 |
| OS GUI | 虚拟机 | 完整桌面环境，支持截图和键鼠注入 |
| Hardware I/O | 真机/设备农场 | 物理设备，无法软件模拟 |

</div>

框架通过 Runtime 接口屏蔽底层差异，研究者只需关心 Tool 的语义，无需关心执行环境是容器、虚拟机还是真机。

**设计原则**

- **极致的扩展性**：Task、Agent、Tool 都通过注册机制扩展，研究者可以快速接入新任务、新 Agent、新工具。 框架层面不阻碍任何形式的 Agent / Task 接入。
- **环境的稳定性**：Runtime 由 Infra 统一管理，对外提供一致的接口，对内负责沙盒的生命周期和运行稳定性，研究者无需关心底层环境差异。

## Nexus Architecture

基于上述设计原则，Nexus 框架的整体架构如下图所示：

<figure markdown="span" style="width: 100%"><object type="image/svg+xml" data="/assets/images/introduction-to-nexus/nexus-architecture.svg" style="width: 100%"></object><figcaption>Nexus 整体架构</figcaption></figure>

### Execution Model

框架的核心是一条清晰的执行链路：**Runner → Task → Agent → Tool → Runtime**。 `single.run()` 函数是整个流程的入口，它接收 `RunInput` 配置，初始化运行环境，执行任务，最后返回 `RunOutput`。

```python
run_input = RunInput(
    task_name='swe_bench',
    agent_name='coding_agent',
    openai_clients={'main': {'base_url': '...', 'api_key': '...'}},
    runtime_provider_name='docker',
    runtime_provider_config={'image': 'python:3.11'},
)
output = await run(run_input)
```

执行流程如下：

1. **加载模块**：通过 `Registry.import_module()` 加载 Task、Agent、Tool 等扩展模块
2. **创建 RuntimeProvider**：根据配置创建 Docker 或 Gongfeng 等沙盒提供者
3. **初始化 LLM 客户端**：支持配置多个 OpenAI 兼容的客户端（如 main、eval、grm）
4. **构建 ExecutionContext**：将上述依赖注入到上下文中，供 Task/Agent/Tool 使用
5. **执行 Task**：在 ExecutionContext 中运行任务函数，自动管理资源生命周期
6. **收集结果**：汇总 task、agent 和 SpanProcessor 结果，返回 `RunOutput`；managed runner 还会将 raw trace 和
   `run.output` 写入 run trace backend

**ExecutionContext** 是贯穿整个执行过程的依赖注入容器。 它基于 Python 的 `contextvars` 实现，在异步环境中提供线程安全的上下文传递。 通过 `require_context()` 函数，Task、Agent、Tool 可以随时获取当前上下文：

```python
ctx = require_context()

# 获取 Runtime 实例（懒加载，首次调用时创建沙盒）
runtime = await ctx.get_runtime(name='main')

# 获取 LLM 客户端
client = ctx.get_openai_client(name='main')

# 访问任务元信息
task_name = ctx.task_name
run_id = ctx.run_id
```

ExecutionContext 同时参与 **轨迹追踪**。框架通过 `@observe` 装饰器或 `span_context` 上下文管理器记录操作的开始时间、
结束时间、属性和事件。业务 **SpanProcessor** 消费普通执行 span，并形成不同格式的 `processor_results`：

- `ChatTrackerProcessor`：提取 LLM 对话历史，输出标准的 messages 格式
- `ChromeTraceProcessor`：Chrome Trace 格式，可在 chrome://tracing 可视化
- `HyTunerProcessor`：提取 RL 训练所需的 token 序列和 reward

live runner 和 distributed single 还会在 task 外层建立 `run.live` 或 `run.single` span。框架级
`RunTraceProcessor` 将整棵 run/task/runtime 树保存为 raw journal，供结果读取和 live SSE 投影使用；它与业务
SpanProcessor 相互独立。详见 [Trace 系统](../../explanation/trace-system.md)。

### Task Interface

**Task** 定义端到端的任务逻辑。 通过 `@task_registry.register()` 装饰器注册，框架会自动验证函数签名：

- 必须是 async 函数
- 必须恰好有一个参数，类型为 `TaskRunInput` 子类
- 返回值类型必须是 `TaskRunOutput` 子类
- 自动包装 `@observe` 装饰器用于链路追踪

```python
@task_registry.register('echo')
async def echo_task(task_input: EchoTaskInput) -> EchoTaskOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime()

    # 调用 Agent 完成任务
    await task_input.run_agent(user_prompt=f'Write "{task_input.message}" to /tmp/echo.txt')

    # 在沙盒中验证结果
    result = await runtime.call_function(lambda: open('/tmp/echo.txt').read().strip())
    return EchoTaskOutput(score=1.0 if result == task_input.message else 0.0)
```

Task 通过 **Mixin** 标记自己所需的输入和输出行为：

- `WithAgentRunner`：标记该 Task 需要传入 `run_agent()` 方法才能完成功能
- `WithScore`：标记该 Task 的输出包含评分字段，供评估和 RL 使用

目前框架已对接多个主流数据集：

<div id="dataset-table" class="center-table" markdown>

| 层级 | Task | 说明 |
|------|:-----|------|
| Tool/API | GAIA、HLE、BrowserComp | 多步推理，需要工具调用 |
| Browser | WebArena | 浏览器自动化任务 |
| Shell/FS | SWE-Bench、Terminal-Bench、SWE-Bench Pro、Multi-SWE-Bench、SWE-Bench Multilingual、SWE-Bench++、SWE-Gym、SWE-Smith、SWE-Rebench | 软件工程和终端操作 |

</div>

<style>
#dataset-table table td:nth-child(2) {
    max-width: 300px;
    word-wrap: break-word;
}
</style>

### Agent Interface

**Agent** 封装 LLM 推理循环。 通过 `@agent_registry.register()` 注册，核心是一个「LLM 调用 → 工具执行 → 更新历史」的迭代过程。 框架同样会验证函数签名：

- 必须是 async 函数
- 必须恰好有一个参数，类型为 `AgentRunInput` 子类
- 返回值类型必须是 `AgentRunOutput` 子类

```python
@agent_registry.register('demo_agent')
async def run_demo_agent(agent_input: DemoAgentInput) -> DemoAgentOutput:
    ctx = require_context()
    executor = ToolExecutor([tool_registry.get('demo/bash'), tool_registry.get('demo/think')])

    history = [{'role': 'user', 'content': agent_input.user_prompt}]
    for _ in range(agent_input.max_iterations):
        response = await ctx.openai_client.create_completion(
            messages=history,
            tools=executor.get_tool_schemas()  # 自动生成 OpenAI function calling 格式
        )
        # 处理响应、执行工具、更新历史 ...

    return DemoAgentOutput(main_trajectory=history)
```

Agent 同样通过 Mixin 标记行为：`WithUserPrompt` 标记需要用户提示词输入，`WithMainTrajectory` 标记输出包含对话历史，`WithInputProvider` 标记支持交互式输入。

目前框架已实现多个主流 Agent：

<div class="center-table" markdown>

| Agent | 接入模式 | 说明 |
|-------|----------|------|
| CodeAct | 白盒 | OpenHands 的 CodeAct 模式，用于软件工程任务 |
| SWE-Agent | 白盒 | 类似 CodeAct 的软件工程 agent，支持 bash、编辑器、提交工具 |
| Rondo | 白盒 | 通用 Agent，集成搜索、代码解释器、浏览器等工具 |
| BrowserGym | 白盒 | 浏览器自动化，支持 WebArena 等任务 |
| Terminus2 | 白盒 | 基于 Prompt 的终端交互，无传统工具调用 |
| Claude Code | 黑盒 | 在沙盒中运行任意版本的闭源 Claude Code |

</div>

### Tool Interface

**Tool** 是 Agent 的 Action Space，通过 `@tool_registry.register()` 注册。 框架验证签名并提取元数据：

- 必须是 async 函数
- 第一个参数类型必须是 Pydantic `BaseModel` 子类（自动转换为 OpenAI function schema）
- 第二个参数类型必须是 `ToolContext`
- 返回值类型必须是 `ToolResponse`
- 自动从 docstring 或 meta 参数提取工具描述

Tool 的输出是字符串列表，直接作为 tool response 返回给模型。 Tool 有两种执行模式：

- `ToolType.LOCAL`：在 Agent 进程内直接执行（如 think、记忆操作）
- `ToolType.RUNTIME`：通过 HTTP 在沙盒中执行（如 bash、文件操作、浏览器）

```python
@tool_registry.register('demo/bash', meta={'tool_type': ToolType.RUNTIME})
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    result = subprocess.run(params.command, shell=True, capture_output=True)
    return ToolResponse(
        output=[{'type': 'text', 'text': result.stdout}],
        context_updates={'cwd': params.working_dir}  # 状态更新会合并回上下文
    )
```

**ToolExecutor** 是统一的工具调用接口。 它根据 Tool 的类型自动路由到正确的执行位置，并从 Pydantic Input Model 自动生成 OpenAI function schema：

```python
executor = ToolExecutor([bash_tool, think_tool, browser_tool])

# 获取所有工具的 schema，传给 LLM
schemas = executor.get_tool_schemas()

# 调用工具，自动路由到 LOCAL 或 RUNTIME
response = await executor.call('demo/bash', {'command': 'ls -la'})
```

Tool 和 ToolExecutor 是为了在使用标准 OpenAI Function Calling 模式的情况下简化编程逻辑的抽象，并不是每个 Agent 都必须使用的。

### Runtime System

Runtime System 负责工具在沙盒中的执行。 它由三层组成：**Runtime Client**（接口层）、**Executor**（能力层）、**RuntimeProvider**（实现层）。

**Runtime Client** 定义了与沙盒通信的统一接口：

```python
class Runtime(ABC):
    async def call_tool(self, import_path: str, params: dict, context: ToolContext) -> ToolResponse
    async def call_function(self, func: Callable, *args, **kwargs) -> Any  # 通过 dill 序列化
    async def call_executor(self, executor_name: str, arg: Any) -> Any
```

**Executor** 是预注册的沙盒能力模块，通过 `@executor_registry.register()` 注册。 Runtime Client 通过 `ExecutorMixin` 提供类型安全的调用方式：

=== "Shell"

    <div style="height: 420px; overflow-y: auto; overflow-x: hidden; padding: 0 16px; border: 1px solid var(--md-default-fg-color--lightest); border-top: none;" markdown>

    在沙盒中执行命令和管理终端会话。

    | 方法 | 功能 |
    |------|------|
    | `run_command()` | 执行单条命令，返回 stdout/stderr/return_code |
    | `create_terminal_session()` | 创建持久化终端会话（基于 tmux） |
    | `destroy_terminal_session()` | 销毁终端会话 |
    | `run_command_in_terminal_session()` | 在终端会话中执行命令并等待完成 |
    | `start_command_in_terminal_session()` | 在终端会话中启动命令（不等待完成） |
    | `query_terminal_command_status()` | 查询命令执行状态 |
    | `send_keys_to_terminal_session()` | 向终端会话发送按键 |
    | `capture_terminal_pane()` | 捕获终端会话当前输出 |
    | `terminate_terminal_session_processes()` | 终止会话中的所有进程 |

    ```python
    # 单条命令执行
    result = await runtime.run_command('ls -la', timeout=30, cwd='/workspace')
    print(result.stdout, result.return_code)

    # 持久化终端会话
    await runtime.create_terminal_session(session_id='main', height=40, width=120)
    cmd = await runtime.run_command_in_terminal_session('main', 'python app.py', timeout=60)
    output = await runtime.capture_terminal_pane('main')
    ```

    </div>

=== "File"

    <div style="height: 420px; overflow-y: auto; overflow-x: hidden; padding: 0 16px; border: 1px solid var(--md-default-fg-color--lightest); border-top: none;" markdown>

    文件上传和下载。

    | 方法 | 功能 |
    |------|------|
    | `upload_file()` | 上传文件到沙盒 |
    | `download_file()` | 从沙盒下载文件 |

    ```python
    await runtime.upload_file(files=[FileItem(local='/local/script.py', remote='/workspace/script.py')])
    result = await runtime.download_file(files=[FileItem(remote='/workspace/output.json')])
    ```

    </div>

=== "Jupyter"

    <div style="height: 420px; overflow-y: auto; overflow-x: hidden; padding: 0 16px; border: 1px solid var(--md-default-fg-color--lightest); border-top: none;" markdown>

    管理和执行 IPython 会话。

    | 方法 | 功能 |
    |------|------|
    | `create_ipython_session()` | 创建 IPython 会话 |
    | `execute_ipython()` | 执行代码，返回输出和图表 |
    | `destroy_ipython_session()` | 销毁会话 |

    ```python
    session = await runtime.create_ipython_session()
    result = await runtime.execute_ipython(session.session_id, '''
    import pandas as pd
    df = pd.read_csv('data.csv')
    df.describe()
    ''', timeout=180)
    print(result.output, result.display_data)
    ```

    </div>

=== "Browser"

    <div style="height: 420px; overflow-y: auto; overflow-x: hidden; padding: 0 16px; border: 1px solid var(--md-default-fg-color--lightest); border-top: none;" markdown>

    控制 Chromium 浏览器进行自动化操作。

    | 方法 | 功能 |
    |------|------|
    | `create_browser_session()` | 创建浏览器会话 |
    | `destroy_browser_session()` | 销毁浏览器会话 |
    | `browser_goto()` | 导航到 URL |
    | `browser_click()` | 点击元素 |
    | `browser_fill()` | 填写表单 |
    | `browser_scroll()` | 滚动页面 |
    | `browser_press_key()` | 发送按键 |
    | `browser_hover()` | 悬停在元素上 |
    | `browser_select_option()` | 选择下拉选项 |
    | `browser_new_tab()` | 新建标签页 |
    | `browser_close_tab()` | 关闭标签页 |
    | `browser_focus_tab()` | 切换到指定标签页 |
    | `browser_go_back()` | 后退 |
    | `browser_go_forward()` | 前进 |
    | `browser_get_observation()` | 获取页面截图和 Accessibility Tree |

    ```python
    session = await runtime.create_browser_session(headless=True)
    await runtime.browser_goto(session.session_id, 'https://example.com')
    await runtime.browser_fill(session.session_id, bid='input#username', value='admin')
    await runtime.browser_click(session.session_id, bid='button#submit')
    obs = await runtime.browser_get_observation(session.session_id, include_screenshot=True)
    ```

    </div>

**RuntimeProvider** 负责沙盒的生命周期管理，通过 `acquire()` 创建实例，`release()` 释放资源：

```python
provider = DockerRuntimeProvider(image='python:3.11')
runtime = await provider.acquire('main')
# ... 使用 runtime
await provider.release('main')
```

**Runtime Server** 运行在沙盒内部，是一个 FastAPI 服务，提供 `/api/v1/call_tool`、`/api/v1/call_function`、`/api/v1/call_executor` 三个接口。 沙盒环境基于 Nix 构建，实现完全自包含 —— 将 `/nix` 目录挂载到任意 Linux 容器即可运行。 详见 [Runtime Injection](/explanation/runtime-injection)。

目前支持两种 RuntimeProvider：

- **Docker**：本地容器执行，适合开发调试和小规模评估
- **Gongfeng**：腾讯内部云沙盒平台，支持大规模并发和资源调度

### Runners

除了 `single.run()` 执行单个任务外，框架还提供两种高级 Runner 满足不同场景需求。

**Batch Runner** 用于大规模评估和数据生成。 它采用 SQLite 作为后端，支持断点续跑、失败重试、并发控制等能力：

```python
batch_config = BatchConfig(
    max_concurrent=32,        # 并发数
    retry_failed=3,           # 失败重试次数
    timeout_per_run=600,      # 单任务超时（秒）
    resume=True,              # 断点续跑
    enable_trace_db=True,     # 轨迹存入数据库
)
await batch_run(run_configs, batch_config)
```

Batch Runner 采用两阶段执行：第一阶段通过 generator 生成所有 RunInput 配置并写入数据库；第二阶段并发执行，实时更新状态（pending → running → success/failed）。 这种设计使得大规模评估具备可观测性和可恢复性。

下面的终端录屏展示了 `run-batch` 和 `inspect-batch` 两个命令行工具在 SWE-Bench Gold Patch 流程中的使用效果：

<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/asciinema-player@3.14.0/dist/bundle/asciinema-player.css" />
<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.14.0/dist/bundle/asciinema-player.min.js"></script>
<style>
  #batch-runner-demo .ap-overlay-loading { display: none !important; }
</style>
<div id="batch-runner-demo"></div>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    AsciinemaPlayer.create('/assets/images/introduction-to-nexus/run-batch.cast', document.getElementById('batch-runner-demo'), {
      preload: true,
      speed: 2,
    });
  });
</script>

**HyTuner Runner** 用于对接强化学习训练框架。 它在 `single.run()` 基础上增加了 RL 训练所需的输出格式：

```python
@dataclass
class HyTunerRunOutput:
    prompt_ids: list[int]      # Prompt token 序列
    response_ids: list[int]    # Response token 序列
    response_mask: list[int]   # Token 掩码
    reward_score: float        # 从 Task 输出提取的 reward
    num_turns: int             # 对话轮数
```

HyTuner Runner 通过专用的 `HyTunerOpenAIClient` 和 `HyTunerSpanProcessor` 实现 Token 级别的状态保持，确保多轮对话中历史 token 序列不变，解决训练时的一致性问题。 详见 [Training Consistency](/explanation/training-consistency)。

## Selected Topics

### [Runtime Injection](/explanation/runtime-injection)

传统方案需要在每个目标镜像中安装运行时依赖，面对 SWE-Gym 这样包含 2000+ 镜像的数据集时维护成本极高。 Nexus 基于 Nix 构建完全自包含的运行时环境，通过 Docker volume 挂载到任意 x86 Linux 容器即可运行，实现「一次构建，处处运行」。 同时支持运行时 Checkpoint 加速，将服务启动时间从秒级缩短到毫秒级。

### [Training Consistency](/explanation/training-consistency)

Agent 训练中存在一个容易被忽视的 Token 一致性问题：一方面，tool call 在推理框架中经过 JSON 解析和重新序列化后，空白符、键顺序等细节会发生变化；另一方面，模型输出的 token 序列转成字符串后，经 tokenizer 重新编码可能产生不同的 token 序列。 这些不一致会导致 logprobs 错位和 Prefix Cache 失效。 Nexus 的 HyTunerClient 通过 Token 级别的状态保持和增量 tokenization，确保多轮对话中历史 token 序列不变，从根本上解决这一问题。

### [Shell Tool Internals](/explanation/shell-tool)

Shell 工具是 Coding Agent 的核心能力。 业界主流方案分为两类：子进程模式（Codex CLI、Gemini CLI）和持久会话模式（SWE-ReX、OpenHands）。 Nexus 采用定制 GNU Bash + tmux 的方案，在 Bash 源码中植入 preexec/precmd 钩子，实现命令边界的精确检测和 stdout/stderr 的分离捕获，适合对数据可靠性要求高的训练场景。

## What's Next

回顾前文的 Agent 产品层级分类，Nexus 目前在 **Tool/API**、**Browser DOM**、**Shell/FS** 三层已经建立了较为完善的 Infra 支持。 接下来的工作将沿两个方向展开：

**横向扩充**：在已支持的层级上持续扩充数据，接入更多的 Task、Agent、Tool，覆盖更广泛的研究场景。

**纵向探索**：向下探索 **OS GUI 层** 和 **Hardware I/O 层** 的可能性。 OS GUI 层需要支持 Windows、macOS、Android、iOS 等多平台的桌面/移动环境；Hardware I/O 层则涉及真机设备农场的接入。 这两层的 Runtime 实现更为复杂，但也是通往更通用 Agent 的必经之路。
