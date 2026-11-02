# HITL V2 功能概览

本文介绍 HITL v2 提供的功能、当前的状态与两个入口。设计动机与目标见 [main.md](main.md)。

---

## 核心功能

HITL v2 为 user simulator 下的多轮交互、long horizon 场景提供了以下核心功能：

- **常用 agent harness 的统一多轮接口。** 把 Claude Code 等 harness 包装成命名 session，支持逐轮投递消息、逐轮取回它的运行结果与待答提问，并屏蔽后端差异。目前包括非交互式 Claude Code 与基于 PTY 的常驻 Claude Code。

- **Coding 场景下的通用 user simulator 设计。** 为通用 Coding 任务设计了一个带 user simulator 的多轮 Claude Code agent。它以 agent fn 的形式提供，可以直接代替现有单轮 Claude Code agent 接入 nexus 的各类 task，并提供了对 persona、工具集和 skill 的可配置性。

- **为 long-horizon 与多 agent 场景设计的轨迹追踪与用量统计。** 提供两个新 span processor（`hitl_usage`、`multiagent_langfuse`），配合上游的 `chat_tracker`，通过按 per-session label 分组的方法隔离多 agent 场景下的轨迹、langfuse trace 和 token 用量统计。

- **常用模型的请求参数模版。** 为 OpenAI、Bedrock Claude、GCP Claude、Hunyuan 等常用模型 API 提供请求参数模版，涵盖 thinking 与非 thinking 路径的参数、prompt cache 策略、usage 解析器、client 后端，使得切换模型只需要修改环境变量，减少模型调用链路上的配置错误。

- **第二代 HITL 任务设计。** 提供第二代 HITL 多轮 SWE 任务以及对应的 user simulator 设计，在第一代的数据、任务和评测设施的基础上，把 user 行为多样性与跨 topic 泛化纳入考虑。这套管线仍然在积极开发和迭代中。

---

## 实验性与待开发功能

HITL v2 依旧处于开发迭代中，部分基础设施的稳定性和可靠性依旧需要时间进行进一步的验证。我们非常欢迎各 topic 参与共同开发和测试这些功能，尽早暴露和解决相关问题。

- **PTY Claude Code 的稳定性。** 长时间 run 中断连的原因已定位并修复（终端快照写满管道、迟到的 AskUserQuestion 答案，见 [known-issues.md](known-issues.md) §1）。pty 比 `print_claude_code` 新，跑过的量也少，仍在继续验证。

- **group session 接口。** HITL v2 计划提供真正的 group session 接口，允许同时并行运行多个 Claude Code session，并异步处理它们的结果。挡住它的上游缺陷（一条 sio 连接上多个 proxy 抢注 `llm_response`）已修复，见 [known-issues.md](known-issues.md) §1，接口本身仍在开发。

- **完整的 MCP 支持。** 由于 Coding 任务对 MCP 几乎没有使用需求，HITL 只提供了一个实验性的搜索示例，用于替换 Claude Code 不可用的原生搜索工具。为了在通用任务上使用，该功能依旧需要单独开发和优化。

- **额外 harness 接入。** 目前计划接入 Codex，需要单独验证 Codex CLI 的接口，交互式实现暂时不考虑。其他 harness 也需要按照其接口特点和需求来进行适配。

---

## 在已有 task 上接入 user simulator

具体使用方式见 [usage.md](usage.md)。

为了在通用 coding 任务下使用我们的 user simulator，只需要在 generator 中将 agent 指向 `hitl/agent_with_user`，并按照参考文档补全多 agent 需要的配置。

该 user simulator 兼容单轮或多轮的任务。你可以选择继续使用原始任务的 user prompt 作为初始需求，由 user simulator 自行决定多轮拆分；也可以为其传入多轮需求，由 simulator 渐进释放和完成多轮需求。

---

## 二次开发

具体见 [develop.md](develop.md)。

HITL v2 除了提供开箱即用的 user simulator，也支持更加定制化的配置和自由开发。

- **用组件实现自己的 user simulator。** HITL v2 提供了较高的开发自由度，支持基于 session manager 的多轮接口与 nexus 提供的 agent 开发套件，开发任何定制化的设计需求。开发时可以参考内置 agent fn 作为基础骨架。

- **接入新的 harness。** 可以通过实现 `SessionManager` 接口接入新的 harness，用于适配多 scaffold 和通用／办公场景下的开发。有这方面的开发需求推荐与我们一同讨论并实现。

---

## 其他文档

- [design.md](design.md)：架构与分层
- [usage.md](usage.md)：运行一个 recipe、接入另一个 task、配置 user simulator
- [develop.md](develop.md)：用组件装配 agent fn、实现一个新的 harness 后端
- [known-issues.md](known-issues.md)：已解决与待解决的缺陷
