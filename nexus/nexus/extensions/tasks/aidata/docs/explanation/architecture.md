---
title: 架构与数据流
audience: 想理解 aidata 内部工作原理的开发者
prerequisites:
  - 完成 [tutorial/first-case.md](../tutorial/first-case.md)
related:
  - ../reference/module-layout.md
  - ./rubric-scoring-model.md
  - ./refactor-history.md
---

# 架构与数据流

本篇讲一次 `aidata.py run <case>` 的完整数据流 —— 从 CLI 触发到 result.json 落地 。 目标是让你在
排障时能明确 " 卡在哪一步 " 。

## 三层进程

```
终端                                                                        产物
┌──────┐                                                                    ─────
│ user │ ─┐
└──────┘  │
          ▼
┌────────────────────────────────────────────────┐
│ aidata.py                                       │  加载 .env 、 渲染模板 、 组织 argv
│ 或 aidata_orchestrator.py                       │
└────────────────────────────────────────────────┘
          │ 起子进程
          ▼
┌────────────────────────────────────────────────┐
│ nexus run-single                                │  框架入口 、 task_registry 分发
└────────────────────────────────────────────────┘
          │ 调 @task_registry.register('aidata')
          ▼
┌────────────────────────────────────────────────┐
│ engine/task.py::run_aidata                      │  ── 12 步 orchestration
│    │                                            │
│    ├── case/spec.py 加载 case bundle            │
│    ├── runtime/sandbox.py 沙箱布局              │
│    ├── runtime/services.py 起 sidecar service    │
│    ├── runtime/mcp_catalog.py 加载 MCP           │
│    ├── nexus.agent.run_agent() agent 循环        │
│    ├── runtime/state.py dump state              │
│    └── rubric/dispatch.py 打分                  │
└────────────────────────────────────────────────┘
          │
          ▼
                                                                       result.json
```

CLI 层只是 " 参数拼接器 + 环境组装 " 。 真正的编排在 `engine/task.py::run_aidata` 里 。

## run_aidata 的 12 步

代码入口 ：`nexus/extensions/tasks/aidata/engine/task.py` 顶部 docstring 与
`@task_registry.register('aidata')` 装饰的函数 。

| # | 步骤 | 关键模块 |
|---|---|---|
| 0 | 解析 case.yaml → 填充 `AidataTaskInput` | `case/spec.py::resolve_case` |
| 1 | 准备沙箱布局（`AGENT_DESKTOP` + `INTERNAL_ROOT` ） | `engine/paths.py` + `runtime/sandbox.py` |
| 2 | 上传随包发布的 mock MCP server 脚本 | `runtime/sandbox.py::upload_dir` |
| 3 | 上传 `context_files` 到 agent inputs | `runtime/context.py` |
| 4 | 上传 case assets 、 运行 `bootstrap_script` | `runtime/sandbox.py::run_bootstrap` |
| 5 | 加载 MCP 工具目录（默认 + case extras ） | `runtime/mcp_catalog.py::build_specs` |
| 6 | 组装 agent 用户消息（清单 + 任务 + skills ） | `runtime/prompt_builder.py` |
| 7 | 运行 agent | `nexus.agent.run_agent`（外部） |
| 8 | 关闭 MCP tools | `engine/task.py::_close_mcp_tools_with_agent_timeout` |
| 9 | 加载 rubric.yaml | `case/spec.py::load_case_rubric` |
| 10 | 下载 mock state 、 评估 rubric | `runtime/state.py::download_state` + `rubric/dispatch.py::run_rubric_async` |
| 11 | 返回 `AidataTaskOutput` | 传回 nexus run-single |

任一步出错都通过 `error_message` 提前返回 —— **不抛异常** 。 nexus 的 task_registry 据此把该次运行
标记为 " 失败但可上报 " ，UI / result.json 里仍拿到结构化结果 。

### error_message 分类

`aidata_orchestrator.py::INFRA_KINDS` + `CASE_BUG_KINDS` 定义了两类 error 分类 ：

**Infra kinds（可重试） ：**

- `runtime_acquire_failed` —— gongfeng 起容器失败 。
- `service_setup_failed` —— sidecar service 起不来 。
- `bootstrap_failed` —— `bootstrap_script` 非零 exit 。
- `judge_unavailable` —— judge LLM 网关 UNAVAILABLE 。
- `agent_timeout` —— agent 循环超过 `agent_timeout` 秒 。

**Case-bug kinds（不重试 、隔离） ：**

- `case_config_error` —— task.yaml schema 违规 。
- `missing_rubric` —— rubric.yaml 不存在 。
- `missing_user_prompt` —— 既没 inline 也没 `instruction.md` 。
- `rubric_load_failed` —— rubric.yaml 存在但解析失败 。

Orchestrator 会按分类决定重试与否 。 手工 `aidata.py run` 不做重试 ，直接把 `error_message` 打到
result.json 。

## 沙箱布局

```
/workspace/                                    容器根
├── desktop/                    (AGENT_DESKTOP —— agent 能看到的所有东西 )
│   ├── inputs/                     ← context_files 上传目标
│   ├── output/                     ← 预建的产物工作区
│   ├── skills/                     ← 软链到 /workspace/skills 、镜像自带
│   └── <case-public-assets>/       ← case bootstrap 可以往这写
│
└── .aidata/                    (INTERNAL_ROOT —— agent 的 shell MCP 看不到 )
    ├── servers/                    ← 自带 mock MCP server 脚本（step 2 上传）
    ├── case/                       ← 上传的 case assets（step 4 目标 ）
    ├── state/                      ← 每次运行的 mock 状态持久化（step 10 输入 ）
    └── tests/                      ← test_result 用的 run-test.sh + junit XML
```

关键约束 ：agent 的 shell MCP 的 `ALLOWED_DIR` **pin 在** `AGENT_DESKTOP` —— 因此 agent 无法通过
`cat /workspace/.aidata/state/...` 绕过 MCP 抽象 、直接读框架内部状态 。 这个隔离让 rubric 有独立的
可信真相源 ，不受 agent 干扰 。

细节见 `engine/paths.py` 。

## 数据流三条支线

### 支线 A ：case 加载 → 沙箱

```
cases/<id>/task.yaml          case/spec.py::resolve_case
cases/<id>/instruction.md   ────────►  AidataTaskInput
cases/<id>/data/*                       │
cases/<id>/assets/*                     │
                                        ▼
                              runtime/sandbox.py 上传
                                        │
                                        ▼
                              agent 沙箱内 /workspace/desktop/*
```

### 支线 B ：agent 循环 → mock state

```
agent 循环                    LLM-backed mock MCP servers
  │                                    │
  ├── tool_call                        ├── 每个 mock 有自己的
  │      │                             │  内部 state（在容器内 ）
  │      ▼                             │
  ├── mcp server 处理                   ├── 状态积累到 mock 内部
  │      │                             │
  │      ▼                             │
  └── tool_result                       │
                                        ▼
                              agent 结束
                                        │
                                        ▼
                              runtime/state.py::download_state
                                （对每个 mock 调 *_dump_state ）
                                        │
                                        ▼
                              合并成 state dict 、写盘 、rubric 消费
```

### 支线 C ：state → rubric → score

```
state dict                    rubric.yaml
    │                              │
    └────────┐        ┌────────────┘
             ▼        ▼
       rubric/dispatch.py::run_rubric_async
                  │
                  ├── 创建 Fetcher 门面（fetch.py ）
                  ├── 按 concurrency 并发评估每条 check
                  │       │
                  │       └── 分派给 primitives/*.py
                  ▼
       list[VerifierResult] （每条 check 的 passed + verdict + detail ）
                  │
                  ▼
       compute_scores() → hard-gate → ScorePolicy 或 config.reward.script → 最终 score
                  │
                  ▼
       task_output.score / rubric_results[]
```

Fetcher 是关键抽象 ：所有 primitive **不再** 直接触碰 `state[...]` ，而是通过 Fetcher 拿证据 。 详见
[`./fetcher-and-primitives.md`](./fetcher-and-primitives.md) 。

打分公式与 hard-gate 语义详见 [`./rubric-scoring-model.md`](./rubric-scoring-model.md) 。

## 主容器 vs sidecar service

case 可以声明 `services: [...]` 声明辅助服务容器 —— 每个 service 是一个 **独立** 的 nexus runtime
（第二个 gongfeng 沙箱） 。 engine 会 ：

1. 在 step 4（bootstrap） 之前把每个 service 拉起 ，等待 ready_check 通过 。
2. 把每个 connection 的 URL 通过 `placeholder` （出现在 `user_prompt` / `extra_mcp_servers` 里 ） +
   可选 `env_name` （注入到 bootstrap script 环境 ） 传给主容器 。
3. rubric 的 `script_probe` 或 `from: {file, service}` 也可以点到 service 里 。

镜像优先级 ：`runtime_provider_config.image` 故意不设 —— 每个 runtime 从 `runtime_configs.<name>.image`
（main） 或 `services[].image` （service） 拿 。 强行设 `runtime_provider_config.image` 会让 gongfeng
的合并逻辑压过 per-service image 。

## 相关文档

- 每个模块的详细职责 —— [`../reference/module-layout.md`](../reference/module-layout.md) 。
- 打分数学 —— [`./rubric-scoring-model.md`](./rubric-scoring-model.md) 。
- 从直接 state 访问到 Fetcher 门面的演化 —— [`./fetcher-and-primitives.md`](./fetcher-and-primitives.md) 。
- 结构重构时间线 —— [`./refactor-history.md`](./refactor-history.md) 。
