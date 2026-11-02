---
title: aidata 模块文档索引
audience: 所有第一次接触 aidata 的人
prerequisites: 无
related:
  - ../README.md
  - ../RUBRIC.md
---

# aidata 模块文档

aidata 是 nexus 的一个 task extension ，负责跑长链路办公 / 生活 agent 训练与评测 case 。 本目录按
[Diátaxis](https://diataxis.fr/) 的四象限组织：

- **tutorial/** —— 手把手教程 ，从零把第一个 case 跑通 。
- **how-to/** —— 具体任务指南 ，每篇解决一个明确问题 。
- **reference/** —— 事实性查表 ，模块布局 / CLI 参数 / 环境变量 / 文件契约 。
- **explanation/** —— 概念解释 ，为什么这么设计 、数据流是怎么走的 。

## 起步路径

第一次接触 aidata ，按下面顺序读效果最好 ：

1. 先读根目录 [README.md](../README.md) ，理解 aidata 存在的目的 —— "长链路办公 / 生活 agent 训练" 、
   case 是自包含的 bundle 、CLI 只做一件事 —— 通过 `case_id` 跑同一个 task 。
2. 跟 [tutorial/first-case.md](tutorial/first-case.md) 走一遍 ，从 clone 到把一个 case 跑出非零 score
   （约 15 分钟） 。
3. 之后按需查 how-to / reference —— 写新 case 就看 [how-to/author-a-case.md](how-to/author-a-case.md) ，
   排障就看 [how-to/troubleshoot.md](how-to/troubleshoot.md) 。

## 分区索引

### tutorial

| 文档 | 说明 |
|---|---|
| [first-case.md](tutorial/first-case.md) | 从 clone 到跑出 rubric score 的 15 分钟 walk-through 。 |

### how-to

| 文档 | 什么时候看 |
|---|---|
| [run-a-case.md](how-to/run-a-case.md) | 单 case 手工执行 ， `aidata.py run` 常用旋钮 。 |
| [batch-run.md](how-to/batch-run.md) | 用 `aidata_orchestrator.py` 大规模跑 + 监控 + dashboard 。 |
| [author-a-case.md](how-to/author-a-case.md) | 从零新建一个 case bundle 。 |
| [author-a-rubric.md](how-to/author-a-rubric.md) | v4.2 assert 原语速查 ，深入版链回 [`RUBRIC.md`](../RUBRIC.md) 。 |
| [use-fetcher-parse-as.md](how-to/use-fetcher-parse-as.md) | 用 `from: {file, service, parse_as}` 和 `script_probe` 。 |
| [troubleshoot.md](how-to/troubleshoot.md) | judge UNAVAILABLE / .env 加载 / 模板未找到 等常见错误 。 |

### reference

| 文档 | 说明 |
|---|---|
| [module-layout.md](reference/module-layout.md) | `engine/` `rubric/` `case/` `runtime/` `mock_servers/` 每层做什么 。 |
| [env-variables.md](reference/env-variables.md) | 所有 `AIDATA_*` / `NEXUS_*` / 上游凭据环境变量表 。 |
| [cli.md](reference/cli.md) | `aidata.py` + `aidata_orchestrator.py` 完整命令参考 。 |
| [file-formats.md](reference/file-formats.md) | `task.yaml` / `rubric.yaml` / `result.json` / junit XML 契约 。 |
| [rubric-primitives.md](reference/rubric-primitives.md) | v4.2 assert 原语的执行参考 —— 每个 primitive 从 YAML 到 `(passed, verdict, detail, raw_head)` 的分支 、 verdict 与真实例子 。 |

### explanation

| 文档 | 说明 |
|---|---|
| [architecture.md](explanation/architecture.md) | 组件边界 + 一次运行的完整数据流 。 |
| [rubric-scoring-model.md](explanation/rubric-scoring-model.md) | v4.2 加权 、v2 pass-count 、hard gate 、`ScorePolicy` 。 |
| [fetcher-and-primitives.md](explanation/fetcher-and-primitives.md) | 为什么把证据抓取抽成 Fetcher facade 。 |
| [v2-freeze.md](explanation/v2-freeze.md) | 为什么冻结 v2 rubric 而不删掉 。 |
| [refactor-history.md](explanation/refactor-history.md) | 最近 6 次结构重构的时间线与兼容承诺 。 |

## 与 aidata 根目录文档的分工

- [`../README.md`](../README.md) —— 面向 " 打开一个陌生模块 " 的人 ，包含目录结构 / 常用 CLI / demo 、
  几种典型 CLI 用法 。 本目录不再复制 ，只做导航 + 深度解释 。
- [`../RUBRIC.md`](../RUBRIC.md) —— rubric v4.2 的完整 DSL 参考 ，字段字典 + 示例 。
  本目录 [author-a-rubric.md](how-to/author-a-rubric.md) 只做速查 ，深挖直接跳过去 。
- [`../RUBRIC_AUTHORING.md`](../RUBRIC_AUTHORING.md) —— rubric 编写风格与推荐组合 ，同上 。

## 语言与风格

全部中文 ，遵循 [`CLAUDE.md`](../../../../../CLAUDE.md) 顶级样式约定 —— 中文标点 `，。：` 、
中文句号后一个英文空格 、 `**bold**` 前后加空格 、行长 ≤ 120 。 代码标识符
（函数名 、文件路径 、YAML 键 、环境变量 、命令 ） 保留英文原样 。
