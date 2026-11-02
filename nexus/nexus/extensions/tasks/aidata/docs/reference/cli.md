---
title: CLI 参考
audience: 需要在脚本或 CI 中调 aidata 命令的开发者
prerequisites: 无
related:
  - ../how-to/run-a-case.md
  - ../how-to/batch-run.md
  - ./env-variables.md
---

# CLI 参考

aidata 提供两个可执行入口 ：

- `aidata.py` —— 单 case 手工执行 、静态校验 、打包 。
- `aidata_orchestrator.py` —— 批量 campaign 、监控 、dashboard 、优雅停机 。

两者都是 stdlib 实现（可选 rich 用于 orchestrator TUI ） ，都能直接用 python 或 uv run 起 。

## `aidata.py`

用法 ：`./nexus/extensions/tasks/aidata/aidata.py <subcommand> [args...]` 。

### 子命令 ：`run`

跑一个 case 。 底层拼起 `nexus run-single` 子进程 ，附带渲染好的 runtime 模板 。

```
aidata.py run [-t|--template <alias|path>] <case_id> [-- <extra passed to nexus run-single>]
```

参数 ：

| 参数 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `--template` / `-t` | run 之后 、`case_id` 之前 | `litellm` | 见下方 " template 别名 " 表 。 |
| `case_id` | 位置 | 必填 | 相对 `AIDATA_CASES_ROOT` 的 case_id ，支持嵌套如 `work/poc/xxx` 。 |
| `extra` | REMAINDER | 空 | 任何 `--` 之后的参数原样透传给 `nexus run-single` （例 `-v` ） 。 |

**template 别名表** ：`litellm` / `default` = `runtime.template.yaml` ；`anthropic` / `vertex` /
`passthrough` = `runtime.anthropic.template.yaml` ；也接受绝对路径或相对 aidata 目录的裸文件名 。

退出码 ：

- `0` —— 命令成功执行完 nexus run-single ，产物已落盘 。 分数是否为 0 不影响退出码 。
- `2` —— 参数错误 、 case 找不到 、 manifest 缺失 、模板找不到 。 打印 `error: ...` 到 stderr 。
- 其它非 0 —— 底层 `nexus run-single` 退出码原样传出 。

向后兼容 ：`aidata.py <case_id>` 与 `aidata.py run <case_id>` 等价 ，历史脚本不用改 。

### 子命令 ：`list`

列出所有 case id 及摘要。

```
aidata.py list
```

从 `AIDATA_CASES_ROOT` （若未设 → 包内 `cases/` ） 起递归找所有含 `task.yaml` 或 `case.yaml`
的目录 。 输出按 case_id 排序 。 退出码 0 。

### 子命令 ：`validate`

对一个 case 做纯 YAML 静态检查 ，不碰 gongfeng ，不起 LLM 。

```
aidata.py validate <case_id>
```

检查项 ：

- `task.yaml` 存在 、 schema 合法 、 metadata 里的 `created_at` 满足日期正则 。
- `context_files` 声明的 fixtures 都存在 。
- `rubric.yaml` 存在 、 `version` 是 `4.2` （或缺省但推断为 v4.2 ） ；每条 check 的
  `assert.type` 在 `V42_ASSERT_TYPES` 里 ；`dimension` 在 `ALLOWED_DIMENSIONS` 里 ；
  `weight` 是数值 。
- `agent_tools` 声明的类别与工具名合法 。

退出码 0 = 全部通过 ；非 0 = 打印详细违规列表 。

### 子命令 ：`package`

把一个 case 目录打成 tarball ，可分发给只有 aidata engine 的同事 。

```
aidata.py package <case_id> [out.tgz]
```

`out.tgz` 缺省为 `./<case_id>.tgz` （目录名会用 `_` 替代 `/` ） 。 tarball 里是完整 case 目录 ，
包括 solution 与 assets 。

### 全局参数

- `-h` / `--help` —— 打印用法 。 每个子命令也支持 `-h` 。

## `aidata_orchestrator.py`

用法 ：`./nexus/extensions/tasks/aidata/aidata_orchestrator.py <subcommand> [args...]` 。

所有子命令都需要 `--out <dir>` 指向 campaign 目录 。 计算侧（`run`） 是唯一 writer ；其它命令是
只读 reader ，可以随时启停 。

### 子命令 ：`run`

启动或恢复一次 campaign 。

```
aidata_orchestrator.py run --out <dir> [其它参数...]
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--out` | path | 必填 | campaign 目录 。 同一目录再跑等于 resume 。 |
| `--cases-root` | path | `/data/workspace/Tasks` | cases 树 ；等价于 `AIDATA_CASES_ROOT` 。 |
| `--models` | csv | `opus-4-8` | 逗号分隔 ，元素必须在 `MODELS` 表里（`opus-4-8` / `hy3` ） 。 |
| `--reps` | int | `1` | 每 (case, model) 组合跑几遍 。 |
| `--concurrency` | int | `32` | 并发上限 ；内部另有 AIMD 自适应逻辑 。 |
| `--max-attempts` | int | `5` | 每个 run_key 最大 attempt 次数（1 初次 + N-1 重试） 。 |
| `--judge-retries` | int | `5` | judge LLM UNAVAILABLE 时重试次数 ；写入子进程 `NEXUS_JUDGE_MAX_RETRIES` 。 |
| `--timeout-default` | int | `3600` | 每次 run 默认 wall timeout 秒 。 |
| `--timeout-cap` | int | `7200` | wall timeout 上限（case 声明覆盖时被 cap ） 。 |
| `--daemon` | flag | 否 | 后台跑 ，脱离终端 、SSH 关闭仍存活 。 |
| `--redo` | flag | 否 | 忽略已终态 run_key ，全部重跑 。 |
| `--include-pending` | flag | 否 | 强制带上 `pending_upstream:` 声明的 case 。 |
| `--only` | csv | 空 | case_id 白名单 。 |
| `--limit` | int | 无 | 只保留前 N 个 case（冒烟用） 。 |
| `--dry-run` | flag | 否 | 打印 plan 后退出 ，不启动子进程 。 |

### 子命令 ：`monitor`

终端 TUI reader 。 rich 装了就用 rich ，否则 refresh 文本 。

```
aidata_orchestrator.py monitor --out <dir> [--interval 2.0]
```

### 子命令 ：`dashboard`

本地 stdlib http 站点 。

```
aidata_orchestrator.py dashboard --out <dir> [--dashboard-port 8899] [--host 127.0.0.1]
```

`--host 0.0.0.0` 慎用 ，dashboard 无鉴权 。

### 子命令 ：`status`

一次性打印进度 ，适合 `watch` 或 CI 。

```
aidata_orchestrator.py status --out <dir>
```

### 子命令 ：`stop`

优雅 kill 一个 daemon 。 SIGTERM 、等 partition 封存 、退出 。

```
aidata_orchestrator.py stop --out <dir>
```

## 相关文档

- 环境变量矩阵 —— [env-variables.md](./env-variables.md) 。
- 单 case 场景 —— [`../how-to/run-a-case.md`](../how-to/run-a-case.md) 。
- 批量场景 —— [`../how-to/batch-run.md`](../how-to/batch-run.md) 。
