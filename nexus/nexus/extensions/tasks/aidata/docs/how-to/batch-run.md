---
title: 批量执行与监控
audience: 需要跑 case × model × rep 矩阵回归的开发者
prerequisites:
  - 完成 [tutorial/first-case.md](../tutorial/first-case.md)
  - 熟悉 [run-a-case.md](./run-a-case.md)
related:
  - ../reference/cli.md
  - ../reference/env-variables.md
---

# 批量执行与监控

`aidata_orchestrator.py` 是 aidata 自带的整轮编排器 ，负责在 `case × model × rep` 三维笛卡尔积上并发跑
`nexus run-single` 、把每次运行的产物落到磁盘 、失败自动重试 、并把结果流式写入按小时分区的
JSONL 。 计算侧只 **写** ；监控与 dashboard 是独立的 **读** 进程 —— 可以随时启停而不影响 daemon 。

不要跟 `nexus run-batch` （通用 batch runner ） 搞混 ：aidata_orchestrator 是 aidata 场景专属的 ，做了
三件通用 batch 做不到的事 —— AIMD 自适应并发 、per-run 独立进程组 kill 、以及按小时分区的封存产物 。

## 术语

- **run_key** = `<case>__<model>__rep<N>` 。 一次 campaign 里的最小单位 。
- **campaign** = 一个 `<out>` 目录 。 里面拿一切完成的 run_key 、buffer 、封存分区 。
- **model** = `MODELS` 表里的一个 key （目前有 `opus-4-8` 与 `hy3` ） 。 由 orchestrator 内部把凭据
  与 runtime 模板绑一起 —— 运行时无需手工改 `.env` 。

## run 子命令

启动或恢复一次 campaign ：

```bash
./nexus/extensions/tasks/aidata/aidata_orchestrator.py run \
  --out /tmp/aidata_camp \
  --cases-root /data/workspace/Tasks \
  --models opus-4-8,hy3 \
  --reps 3 \
  --concurrency 32 \
  --daemon
```

关键参数（完整表见 [reference/cli.md](../reference/cli.md) ） ：

- `--out` （必填） campaign 目录 ；同一 `--out` 再跑一次会自动 resume ，跳过已终态的 run_key 。
- `--cases-root` cases 树 ，等价于 `AIDATA_CASES_ROOT` 环境变量 。
- `--models` 逗号分隔 ；未列在 `MODELS` 表里的名字会拒绝 。
- `--reps` 每 (case, model) 组合跑几遍 —— 让 rubric score 分布可比 。
- `--concurrency` 上限 。 内部还有 AIMD 逻辑 ，会在遇到批量 infra failure 时自动降 。
- `--max-attempts` 每个 run_key 最多几次 attempt （默认 5 = 1 初次 + 4 重试 ） 。 重试
  策略见 [explanation/architecture.md](../explanation/architecture.md) 。
- `--judge-retries` judge LLM 遇到 UNAVAILABLE 时的重试次数 ，会被写到子进程的
  `NEXUS_JUDGE_MAX_RETRIES` 里（见 [reference/env-variables.md](../reference/env-variables.md) ） 。
- `--daemon` 后台跑 ，脱离终端 。 关 SSH 会话不会 kill 进程 。
- `--redo` 忽略已经终态的 run_key ，全部重跑 。 用于 rubric 或 case 有 breaking change 之后的强制重刷 。
- `--include-pending` 默认会跳过声明了 `pending_upstream:` 的 case ；这个开关强制带上 。
- `--only` 逗号分隔的 case_id 白名单 。 与 `--limit` 联合可只跑一小组冒烟 。
- `--dry-run` 打印 plan 就退出 ，不真启动子进程 。

## monitor 子命令

`monitor` 是终端 TUI reader ，rich 装了就用 rich ，没装就退化成普通 refresh 文本 ：

```bash
./nexus/extensions/tasks/aidata/aidata_orchestrator.py monitor --out /tmp/aidata_camp
```

只读 `--out` 目录 ，不影响 daemon 。 `--interval` 控制刷新周期（默认 2 秒 ） 。

## dashboard 子命令

本地起一个 stdlib http 站点 ，可视化查看进度 ：

```bash
./nexus/extensions/tasks/aidata/aidata_orchestrator.py dashboard \
  --out /tmp/aidata_camp \
  --dashboard-port 8899 \
  --host 127.0.0.1
```

同样只读 `--out` 目录 。 生产环境避免 `--host 0.0.0.0` 除非你确认这个 dashboard 可对外 。

## status 子命令

一次性打印当前 campaign 的进度 —— 等价于 monitor 的一个截屏 ，适合 `watch` 或 CI ：

```bash
./nexus/extensions/tasks/aidata/aidata_orchestrator.py status --out /tmp/aidata_camp
```

## stop 子命令

优雅 kill 一个 daemon 。 会 SIGTERM 守护进程 ，等它把当前 partition 封存到 `_SUCCESS` 后退出 ：

```bash
./nexus/extensions/tasks/aidata/aidata_orchestrator.py stop --out /tmp/aidata_camp
```

## 产物布局

```
<out>/
├── plan.json                              # 生成一次 ，之后不改
├── runs/<run_key>/
│   ├── result.json                        # 每次 attempt 都会覆盖这里
│   ├── attempts/N/                        # 每次 attempt 的完整 log 与 stderr
│   └── state/                             # AIDATA_STATE_DUMP_DIR ，per-run 隔离
├── stream/dt=YYYY-MM-DD/hour=HH/
│   ├── part.jsonl                         # 归一化后的每 run 一行
│   └── _SUCCESS                           # 封存标记（partition-close-out ）
└── daemon.pid                             # daemon 进程锁
```

`stream/` 下的 partition 一旦封存就不会再改 —— 下游 pipeline 可以按 dt/hour glob 直接读 。

## 用 `nexus run-batch`（可选替代路径 ）

aidata 也提供了一个 `generator.py` ，兼容 nexus 通用 `run-batch` runner 。 适用场景是 " 我不需要
orchestrator 那些 aidata 专属的 AIMD / 分区封存 ，我想直接接上 nexus 的 SQLite trace + `inspect-batch`
TUI" 。 完整命令见 [`../../README.md`](../../README.md) §Batch 段 —— 那里已经列全 ，本篇不复制 。

## 大规模跑的常见坑

- **AIMD 陷入低并发** —— 多半是 `judge_unavailable` 频繁触发把 concurrency 逼下来 。 检查 judge
  网关状态或者临时把 `--judge-retries` 抬高 。
- **stream/ 里 partition 数量爆炸** —— 每小时一个 partition ，跑几天正常有几十个 ；不是 bug 。
- **run_key 保留态阻塞了 resume** —— 用 `--redo` 强刷 ，或手动删 `runs/<run_key>/` 。

## 相关文档

- CLI 参数完整表 —— [reference/cli.md](../reference/cli.md) 。
- Retries 与 error 分类 —— [explanation/architecture.md](../explanation/architecture.md) §重试策略 。
- 环境变量矩阵 —— [reference/env-variables.md](../reference/env-variables.md) 。
