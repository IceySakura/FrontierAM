---
title: 为什么 v2 rubric 是冻结而非删除
audience: 想理解 v2 / v4.2 兼容策略的开发者与 case 迁移者
prerequisites:
  - 已经知道 v4.2 是当前推荐（[author-a-rubric.md](../how-to/author-a-rubric.md)）
related:
  - ./rubric-scoring-model.md
  - ./refactor-history.md
  - ../../RUBRIC.md
---

# 为什么 v2 rubric 是冻结而非删除

## 背景

在 v4.2 之前 ，rubric 只有一种表达方式 —— v2 语法 ：

```yaml
# v2 rubric（legacy ）
version: "2"
checks:
  - id: email_sent
    verify:
      kind: llm_judge         # 或 eq / regex / shell
      claim: "agent sent an email to hr"
      source: agent_answer
```

verifier 种类是 `llm_judge` / `eq` / `regex` / `shell` （单一真相源 ：
`case/schema.py::LEGACY_VERIFY_KINDS` ） 。 打分是 pass count ：`raw_score = passed / total` 。

v4.2 换成了面向标注员的 DSL —— assert primitives 、 `dimension` + `weight` 、 `hard_gate` 、
`config` 段 。 更好写 、更好审 、更能表达业务 semantics 。

## 为什么不直接删掉 v2

**约 59 个生产 case 仍在用 v2** 。 全部迁移意味着 ：

- 每个 case 的 rubric 都要改 、都要重新审 —— 体量数十人天 。
- 已有的 baseline score 分布可能因为语义细微差别而漂移 —— 训练侧的历史比较全部作废 。
- 迁移 window 内 CI 全绿维护成本高 —— 新旧混跑期间任何 " 分数变化 " 都要判断是 " rubric 迁移引入 "
  还是 " 真的模型或 case 改动 " 。

删掉 v2 的成本远大于保留兼容路径的成本 。

## 冻结的具体内容

**允许 ：**

- 已有 v2 rubric 继续跑 —— 分派器 `dispatch.py::run_rubric_async` 里判断 rubric 版本 ，走
  `legacy_v2.py::_run_v2_check` 路径 。
- v2 pass-count 语义保持逐字节一致 —— 不改分数计算 、不引入新字段 。

**拒绝 ：**

- validate 阶段拒绝新写入 v2 rubric —— `aidata.py validate` 会把 `version: "2"` 或含
  `verify:` 块的 rubric 判为违规 ，退出码非 0 。 CLI 层的门槛让 " 新加的 case 只能是 v4.2" 变成
  硬约束 。

**警告 ：**

- 运行时加载 v2 rubric 会打 `DeprecationWarning: v2 rubric ... is deprecated` 。 message 里说清楚
  " 约 59 生产 rubric 仍用 、新 case 必须 v4.2 、Removal target: Q4 2026" 、并链回
  [`../../RUBRIC.md`](../../RUBRIC.md) §7 。

代码入口 ：`dispatch.py::run_rubric_async` 里那段 `_rubric_is_v42` 检查 + `warnings.warn(...)` 。

## 单一真相源

关键常量在 `case/schema.py` 里 ，被 CLI 校验器 + 运行时加载器 **两个** 消费者共用 —— 避免历史上的
双份漂移问题 ：

| 常量 | 作用 |
|---|---|
| `V42_ASSERT_TYPES` | v4.2 允许的 assert primitive 集合 。 加新 primitive 只改这里 。 |
| `LEGACY_VERIFY_KINDS` | v2 允许的 verify kind 集合 。 冻结中 —— **不再新增** 。 |
| `V2_VERSION_LITERALS` | 表示 " 这是 v2 rubric" 的取值集合 —— `{'2', '2.0', 2, 2.0}` 。 |

在合并前 ，`aidata.py` 与 `case/spec.py` 各自有一份 —— 加新 primitive 时会漏改一边 。 现在都从
`schema.py` 导入 。

## 迁移路径

想把一个 v2 rubric 迁到 v4.2 ？ 见 [`../../RUBRIC.md`](../../RUBRIC.md) §7 —— 每个 v2 verify kind
对应一个 v4.2 primitive ：

| v2 kind | v4.2 primitive |
|---|---|
| `eq` | `state_equals` |
| `regex` | `text_contains`（`regex:` 字段） |
| `llm_judge` | `judge_claim` |
| `shell` | `script_probe`（退出码 + stdout 谓词） |

迁移时注意 ：v2 pass-count 语义没有 `weight` / `dimension` 概念 —— 迁到 v4.2 时都要显式给 。 建议 ：

- 一开始所有 check `weight: 1` 、`dimension: correctness` ，先跑起来看 baseline 。
- 然后再按业务重要性 tune weight 、按 case 语义分 dimension 。

## Removal target

Q4 2026 计划的移除包括 ：

- 删掉 `legacy_v2.py` 分派路径 。
- CLI 校验器不再识别 legacy verify kind 名 。
- `LEGACY_VERIFY_KINDS` 常量本身保留（作为历史记录） ，但不再被消费 。

前提是届时已经把生产 case 全部迁到 v4.2 。 若届时还有 case 未迁 ，remove target 会顺延 —— 冻结
不代表 " 强制删除 " 。

## 相关文档

- v4.2 完整字段字典 —— [`../../RUBRIC.md`](../../RUBRIC.md) 。
- 打分模型（v4.2 与 v2 的 raw_score 差异） —— [`./rubric-scoring-model.md`](./rubric-scoring-model.md) 。
- Deprecation 相关的 warnings 处理 —— [`../how-to/troubleshoot.md`](../how-to/troubleshoot.md) §9 。
