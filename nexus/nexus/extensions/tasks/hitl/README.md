# HITL Task Infrastructure

`nexus.extensions.tasks.hitl` 提供构建 **Human-in-the-Loop 多智能体任务**的可复用基础设施。 一个 LLM 扮演只懂需求、不直接写代码的「人类协作者」（human orchestrator），通过工具把工作委派给沙箱里的 Claude Code「worker」，并验证其产出 —— 主要用于生成 SFT / RL 训练轨迹，也可用作评测。

它**刻意保持轻量**：共享层只提供 session 管理、tool dispatch、tracing、usage、评估容器和工具函数；每个 recipe 自己拥有任务逻辑、prompt、状态推进和评估策略。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/main.md](docs/main.md) | 从这里开始：动机、目标与收益、当前的欠缺 |
| [docs/overview.md](docs/overview.md) | 功能概览：核心功能、实验性功能、两个接入入口 |
| [docs/design.md](docs/design.md) | 架构：三层组件的分工，以及 model / eval / recipe 各自的职责 |
| [docs/design_actor.md](docs/design_actor.md) | Session manager：把一个 harness 包成可逐轮驱动的命名 session |
| [docs/design_usersim.md](docs/design_usersim.md) | User simulator：主循环、需求队列、system prompt、persona、skill |
| [docs/design_label.md](docs/design_label.md) | label 机制与按 label 分组的三个 span processor |
| [docs/usage.md](docs/usage.md) | 跑起来：运行已有 recipe、接入另一个 task、配置 user simulator |
| [docs/develop.md](docs/develop.md) | 二次开发：用组件装配自己的 agent fn、接入一个新的 harness |
| [docs/known-issues.md](docs/known-issues.md) | 已解决与待解决的缺陷 |
| [AGENTS.md](AGENTS.md) | 开发约定：改动范围、依赖方向、注释 |

## 包结构

```text
hitl/
  actor/           # worker session 抽象与后端（print_claude_code / pty_claude_code）
    labelling/     # 给每个请求带上 trace label 的 proxy 与 handler
  user_agent/      # user simulator：循环、工具派发、persona、需求队列
  agents/          # 可移植的 agent fn（agent_with_user）
  trace/           # 标签、按标签分桶的 usage 与 Langfuse、导出期 dedup
  model/           # worker 模型协议（profile）层
  eval/            # recipe-neutral EvaluationResult 与各 evaluator
  mcp/             # 内置 search MCP server
  utils/           # git / patch / json 辅助
  recipe/
    swe_chain_v2/  # long-horizon 多请求 SWE（自带 task 与 agent）
    swe/           # user simulator + 原生 swe_task
    terminal/      # user simulator + 原生 terminal_task
    frontend/      # user simulator + 原生 frontend_task
    smoke/         # 按脚本驱动 user 的冒烟检查（自带 task）
    parallel/      # 多 session 并发的检查（自带 task）
```

接入类的三个各含 `generator.py`、`run.sh`，以及该 topic 自己的 `persona.py`（如有）。

## 快速开始

```bash
export SWE_DATASET_PATH=/path/to/dataset.jsonl
export HITL_HUMAN_API_KEY=...  HITL_HUMAN_BASE_URL=...  HITL_HUMAN_MODEL=...
export HITL_WORKER_API_KEY=... HITL_WORKER_BASE_URL=... HITL_WORKER_MODEL=...
export SWE_RUNTIME_BASE_URL=... SWE_RUNTIME_TOKEN=...

nexus run-batch \
  -g nexus/extensions/tasks/hitl/recipe/swe_chain_v2/generator.py \
  -d results.db --max-concurrent 10
```

完整环境变量与 recipe 编写见 [docs/usage.md](docs/usage.md)。

## 核心约定

- **复用 Nexus 原生抽象**：task/agent registry、tool schema、runtime provider、openai client、chat tracker。
- **Handler 在原生工具之后运行**：schema 与校验全归 Nexus，HITL 只加「工具之后发生什么」。
- **多 tool-call 逐个处理**：每个 tool call 都必须写配对的 tool 响应，不可 all-or-nothing。
- **评估非通用语义**：基础设施只给 `EvaluationResult`，`score` / `resolved` 由 recipe 聚合。
- **只有 generator 读环境变量**：model 层和 recipe 逻辑保持 env-free。
