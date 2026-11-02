# 架构

HITL 提供三层面向多轮交互与多 agent 场景的基础设施：驱动 harness 的 `actor/`、构成 user 的 `user_agent/`、按来源分离可观测性的 `trace/`。在这三层之上，`agents/` 提供开箱可用的 agent fn，`recipe/` 提供具体的任务实现和适配。

---

## 1. 分层

| 目录　　　　　| 职责　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 | 设计文档　　　　　　　　　　　　　　　 |
| ---------------| ----------------------------------------------------------------| ----------------------------------------|
| `actor/`　　　| 把一个 agent harness 包装成可以逐轮驱动的命名 session　　　　　| [design_actor.md](design_actor.md)　　 |
| `user_agent/` | user 的组件：需求队列、prompt 装配、persona、skill、工具、提醒 | [design_usersim.md](design_usersim.md) |
| `trace/`　　　| label 机制与按 label 分组的三个 span processor　　　　　　　　 | [design_label.md](design_label.md)　　 |
| `agents/`　　 | 由上面三层拼成的 agent fn，可直接接入 nexus 的各类 task　　　　| [develop.md](develop.md)　　　　　　　 |
| `model/`　　　| 各族模型 API 的请求参数模版　　　　　　　　　　　　　　　　　　| 见 §4　　　　　　　　　　　　　　　　　|
| `eval/`　　　 | HITL 相关 task 的评估设施　　　　　　　　　　　　　　　　　　　| 见 §5　　　　　　　　　　　　　　　　　|
| `recipe/`　　 | 基于 HITL 的开发的任务集合　　　　　　　　　　　　　　　　　　 | 见 §6　　　　　　　　　　　　　　　　　|
| `utils/`　　　| generator 辅助、patch 与 git 工具、类型转换　　　　　　　　　　| 　　　　　　　　　　　　　　　　　　　 |
| `mcp/`　　　　| 内置 MCP server　　　　　　　　　　　　　　　　　　　　　　　　| 　　　　　　　　　　　　　　　　　　　 |

---

## 2. actor 与 user_agent 的隔离

默认的 agent fn 只使用 manager 提供的公开接口进行调用，允许对同一套 user simulator 接入不同的 worker backend。为了保持隔离性，需要遵守 manager 的公共接口约定。

---

## 3. 依赖方向

`actor/` 与 `user_agent/` 互不 import，两者各自依赖 `trace/`：manager 在 `start` 时给 session 生成 label，user 那一侧在发起调用时进入 `labelled()`，两条路径在 span 上汇合，由 run 级的 processor 分开读回。

`agents/` 与 `recipe/` 在上面组装，同时 import 这三层。一次 run 里 worker backend 与 user 的配置分别由 `worker_config` 与 `agent_config` 声明，由 recipe 的 generator 装配。

---

## 4. model：请求参数模版

为了标准化不同模型的调用约定，HITL 提供了 model 字模块用于封装不同 client 的参数构建方式。一个 profile 声明三样：用哪个 openai client、请求参数长什么形状、响应里的 usage 计数器放在哪。目前的 profile 包括：

* `bedrock_claude` 用于 bedrock claude 调用
* `gcp_claude` 用于 gcp 的 claude 模型，走 `vertex_anthropic` client
* `hy3` 用于部署在线推理的 hy 以及其他模型，默认开启 thinking
* `naci_glm` 用于 naci 部署的 GLM
* `openai_basic` 用于通用的 openai 模型
* `default` `openai_basic` 的别名，环境变量未指定时的取值

profile 推荐在 generator 中使用环境变量填写对应参数，`HITL_WORKER_CLIENT_PROFILE` 选 worker 那一侧，`HITL_HUMAN_CLIENT_PROFILE` 选 user 那一侧。同一个 profile 名也决定这一侧的用量解析方式，见 [design_label.md](design_label.md) §5.2。

新增一族模型时在 `model/profiles/` 下加一个模块并调 `register`，导入这个包即完成注册。

---

## 5. eval：评估基础设施

HITL 特有任务使用该模块存放评估基础设施，目前主要包含 SWE 的 agentic rubrics 和 单元测试相关评估。

`evaluators/` 一个模块一个评估器，各自只取自己要的东西 —— 官方单测要一个准备好的沙箱，rubrics judge 要一个模型 client —— 返回统一的 `EvaluationResult`。`aggregate` 把这些结果合成 run 的分数，与跑了哪些评估器无关。由 recipe 决定这次跑哪几个并装配它们的输入。

---

## 6. recipe

recipe 用于服务基于 HITL 开发与适配的任务，分为两类。

**接入类** 使用通用 agent fn 接一个原始 task，提供 generator 、run.sh 以及可能的 prompt 与 persona 配置目前提供了主分支上的 swe、terminal 与frontend 任务的接入示例。

**自带 task 类** 自行基于 HITL 组件开发的完整 任务，包括用于测试组件功能的 smoke 以及正在开发的 HITL v2 任务。

---

## 7. 其他文档

- [overview.md](overview.md)：功能与现状 —— 核心功能、实验性功能、两个接入入口
- [usage.md](usage.md)：运行 user simulator  —— generator 配置、persona 与 skill 的选用与新增、创建一个新的 recipe
- [develop.md](develop.md)：二次开发 —— 用组件装配 agent fn、实现一个新的 harness 后端
- [known-issues.md](known-issues.md)：已解决与待解决的缺陷
