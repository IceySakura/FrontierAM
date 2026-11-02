---
title: 跑通第一个 case
audience: 从没跑过 aidata 的开发者
prerequisites:
  - Python 3.11+ 、 uv 已装好
  - Gongfeng sandbox 与 LLM 网关的凭据（团队 wiki 拿）
estimated_time: 15 min
related:
  - ../how-to/run-a-case.md
  - ../how-to/troubleshoot.md
  - ../reference/env-variables.md
---

# 跑通第一个 case

本教程把你从 " 只有 nexus 仓库的 clone " 带到 " 看见一个 rubric score 落地到 result.json" 。 全过程只碰
一个 case （ `work/poc/email_http_poc` ） ，最简通路 ，不涉及 batch 和 orchestrator 。

跑完你会理解 ：

- aidata 的执行入口在哪里 。
- CLI 、 `.env` 、cases 目录三者怎么联系起来 。
- 一次运行的产物是什么 、放在哪里 。

如果只想执行某个已知 case ，跳到 [how-to/run-a-case.md](../how-to/run-a-case.md) 。

## 前提检查

在项目根目录执行下面这一段 ，确认三件事都到位 ：

```bash
cd /path/to/nexus
uv --version                              # uv 已装
python -c "import sys; assert sys.version_info >= (3, 11)"
ls nexus/extensions/tasks/aidata/aidata.py  # 模块存在
```

## 第 1 步 ：准备 .env

aidata 需要两组凭据 ：Gongfeng sandbox provider（提供 docker 容器 ）和 LLM 网关（同时给 agent 和
mock LLM 用 ） 。 在仓库根目录写一份 `.env` ：

```bash
# nexus/.env
GONGFENG_API_BASE_URL=http://hyrl-sandbox.prod.woa.com:8052
GONGFENG_TOKEN=<从团队 wiki 拿>

SEARCH_OPENAI_BASE_URL=http://llm-api.model-eval.woa.com
SEARCH_OPENAI_API_KEY=<APP_ID>:<APP_KEY>
SEARCH_OPENAI_MODEL=api_naci_default_gemini-3.1-pro-preview
```

CLI 会按 `<cases-root>/../.env` → `<repo>/.env` 顺序自动加载（前者优先） ，无需手动 `source` 。 完整
优先级见 [reference/env-variables.md](../reference/env-variables.md) 。

## 第 2 步 ：找到 cases 目录

上游 nexus 仓库刻意 **不打包** case 数据 —— case 数据在下游 `Tasks` 仓 。 假设你已经拿到 cases 树 ：

```bash
export AIDATA_CASES_ROOT=/data/workspace/Tasks
ls "$AIDATA_CASES_ROOT/work/poc/email_http_poc/"     # 应看到 task.yaml / rubric.yaml / instruction.md
```

如果你没有下游 cases 数据 ，也可以在上游仓自建一个最小 case （见
[how-to/author-a-case.md](../how-to/author-a-case.md) ） ，或者请团队里已经有 cases 树的同事拷贝一份 。

## 第 3 步 ：先做静态校验

在真跑之前 ，用 `validate` 子命令做一次 schema 检查 —— 它不碰 gongfeng ，只读本地 YAML ，秒级返回 ：

```bash
./nexus/extensions/tasks/aidata/aidata.py validate work/poc/email_http_poc
```

期望输出 ：`ok` 或者一个空错误列表 。 若失败 ，输出会精确指到 `task.yaml` 或 `rubric.yaml` 的哪一行
不合规 ；参考 [reference/file-formats.md](../reference/file-formats.md) 修 。

## 第 4 步 ：跑一次

单 case 手工执行 ：

```bash
./nexus/extensions/tasks/aidata/aidata.py run work/poc/email_http_poc -v
```

CLI 底层做的事情按顺序是 ：

1. 加载 `.env` 到进程环境 。
2. 渲染 `runtime.template.yaml` （litellm 模板 ） 到临时文件 ，替换 LLM 网关字段 。
3. 起一个 `nexus run-single` 子进程 ，把 task 名 = `aidata` 、 `task_config.case_id = <case_id>` 传进去 。
4. task 内部按 [explanation/architecture.md](../explanation/architecture.md) 里那 12 步走 —— case
   加载 → 沙箱准备 → services 拉起 → bootstrap → MCP 目录加载 → agent 循环 → state dump → rubric
   评分 。
5. 结果 JSON 落到 `workspace/mcp/tmp/aidata_<case>_<ts>.result.json` 。

跑完最后一段日志会写出 `saved: workspace/mcp/tmp/aidata_...result.json` 。

## 第 5 步 ：读产物

```bash
ls -lh workspace/mcp/tmp/aidata_email_http_poc_*.result.json
```

关注这几个字段 ：

| 字段 | 含义 |
|---|---|
| `task_output.score` | 最终得分 ，`Σ weight × passed / Σ weight` （v4.2 ） 。 |
| `task_output.rubric_results[]` | 每条 check 的 `passed` / `verdict` / `detail` 。 |
| `task_output.state_summary` | 运行结束时 mock state 的精简快照 —— rubric 就是从这里读证据 。 |
| `agent_output.main_trajectory` | agent 每一轮消息与工具调用 。 |

一个健康的运行长这样 ：`score > 0` 、`main_trajectory` 长度 ≥ 3 、每条 rubric_results 有 `verdict` 字段 。
如果 `score == 0` 且 `error_message` 非空 ，说明 step 0-6 里某步失败了 ，直接跳到
[how-to/troubleshoot.md](../how-to/troubleshoot.md) 。

## 第 6 步 ：解读 rubric_results

`email_http_poc` 有两条 check ：一条 `tool_called` 、一条 `state_equals` 。 打开 result.json 会看到
类似这样一段 ：

```json
{
  "id": "email_sent",
  "assert_type": "tool_called",
  "passed": true,
  "verdict": "PASS",
  "detail": "matched 1 call to email_send"
}
```

含义 ：assert `tool_called` 找到了 agent 至少调用过一次 `email_send` 工具 ，PASS 计入总分 。
每种 assert 原语的判定语义见 [how-to/author-a-rubric.md](../how-to/author-a-rubric.md) 。

## 下一步

你已经跑通了单 case 通路 。 接下来常见的三种去向 ：

- 想跑一批 case 做回归 —— [how-to/batch-run.md](../how-to/batch-run.md) 。
- 想理解 " 为什么 rubric 是这样打分的 " —— [explanation/rubric-scoring-model.md](../explanation/rubric-scoring-model.md) 。
- 想给自己的项目造 case —— [how-to/author-a-case.md](../how-to/author-a-case.md) 。
