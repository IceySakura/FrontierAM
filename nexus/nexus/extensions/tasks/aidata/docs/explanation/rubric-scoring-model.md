---
title: rubric 打分模型
audience: 想理解 " score 是怎么算出来的 " 的开发者
prerequisites:
  - 已经看过 [author-a-rubric.md](../how-to/author-a-rubric.md)
related:
  - ./architecture.md
  - ./v2-freeze.md
  - ../../RUBRIC.md
---

# rubric 打分模型

一次 `run_rubric_async` 里最终产出的 `task_output.score` 由四层组成 ：

1. **每条 check 的 pass/fail** —— 由 assert primitive 决定 。
2. **加权求和 → 连续分数** —— v4.2 用 `Σ weight × passed / Σ weight` ；v2 用 pass count 。
3. **hard-gate** —— 关键 check 失败时封顶。
4. **最终聚合** —— 默认走 ScorePolicy；配置 `config.reward.script` 时由脚本计算跨 check reward。

本篇讲每一层 、以及为什么这样设计 。 完整代码入口在
`nexus/extensions/tasks/aidata/rubric/dispatch.py::compute_scores` +
`rubric/policy.py::finalize_score` + `rubric/reward.py::resolve_reward` 。

## 层 1 ：每条 check 的 pass/fail

每条 v4.2 check 的 assert primitive 返回一个 `VerifierResult = (passed, verdict, detail, raw_head)` ：

- `passed` : bool —— 二值 pass/fail 。
- `verdict` : str —— `PASS` / `FAIL` / `ERROR` / 或 legacy `TRUE` / `FALSE` 。 主要给 UI / 日志用 。
- `detail` : str —— 一段人类可读描述 ，说清楚为什么 pass / fail 。
- `raw_head` : str —— 只对 LLM judge 有意义 —— 保留 model 字面回复的前 N 个字符便于 debug 。 确定性
  primitive 会留空 。

代码入口 ：`rubric/config.py::VerifierResult` typedef 。 每个 primitive 的实现在
`rubric/primitives/*.py` ，通过 `dispatch.py::_ASSERT_EVALUATORS` 注册表分派 。

## 层 2 ：加权求和 → 连续分数

### v4.2 语义（当前推荐 ）

```
raw_score = Σ (weight × passed) / Σ weight
```

- `weight` 是 rubric 里每条 check 声明的整数 。
- `passed` 是 0 或 1 。
- 分子分母都在 checks 里累加 —— dimension 不参与直接计算 ，只用于结果分组展示 。

举例 ：三条 check ，weight 分别是 1, 2, 3 ，pass 情况分别 True, False, True ：

```
raw_score = (1×1 + 2×0 + 3×1) / (1 + 2 + 3) = 4/6 ≈ 0.67
```

### v2 语义（冻结 ）

```
raw_score = passed_checks / total_checks
```

无 weight 概念 、无 dimension 分组 。 只保留兼容路径 。 详见 [`./v2-freeze.md`](./v2-freeze.md) 。

### dimension 分组

v4.2 支持三个 dimension —— `correctness` / `functional` / `robustness` （代码单一真相源 ：
`case/schema.py::ALLOWED_DIMENSIONS` ） 。 每条 check 属于一个 dimension 。 rubric_results 里
每条会带 `dimension` 字段 ，UI 与下游可以按 dimension 拆分展示 —— 比如 " 该 case correctness 100% 、
robustness 50%" 。

dimension 分组不影响 raw_score 计算 —— raw_score 是 checks 层面的加权求和 。

## 层 3 ：hard-gate

某些 rubric 有 " 关键 check —— 挂了整题得 0 " 的诉求 。 v4.2 通过 hard-gate 表达 ：

```yaml
checks:
  - id: user_authenticated_before_pay
    hard_gate: true       # 若这条 FAIL ，最终 raw_score 强制 = 0
    ...
```

`compute_scores` 会在加权求和之前先扫一遍 checks ，如果任一带 `hard_gate: true` 的 check FAIL ，
直接把 `raw_score` 打成 0 —— 无论其它 check 表现如何 。

这是 " 底线约束 " —— 用于 " 如果 agent 越权 / 遗失关键前置 ，无论后面做得多好都算不合格 " 。

## 层 4 ：最终聚合

### 默认：ScorePolicy 坍缩

`raw_score` 是连续分数（0.0-1.0） 。 但训练侧或评测侧有时想要二值 、或者带 threshold 截断的输出 。
`ScorePolicy` 三个正交开关控制这件事 ：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `threshold` | float | `1.0` | 截断点 。 |
| `continuous` | bool | `True` | 是否保留连续分数 。 |
| `use_threshold` | bool | `False` | 是否应用截断 。 |

组合语义（代码入口 ：`rubric/policy.py::finalize_score` ） ：

| continuous | use_threshold | 输出（`s` = 层 3 后的连续分数 ） |
|---|---|---|
| False | True | 二值 ：`1.0 if s>=thr else 0.0` |
| True | False | 纯连续 ：`s` （**默认**） |
| True | True | 带门控的连续 ：`s if s>=thr else 0.0` |
| False | False | 退化 ：透传 `s` （+ 警告） |

三个开关的优先级（`ScorePolicy.resolve` ） ：

1. 环境变量（最高） ：`AIDATA_SCORE_THRESHOLD` / `AIDATA_SCORE_CONTINUOUS` /
   `AIDATA_SCORE_USE_THRESHOLD` 。
2. rubric 的 `config.score` 块 。
3. dataclass 默认值 。

**为什么默认是 " 纯连续" ？** 训练侧对连续分数敏感（RM 、DPO 类都 benefits from 平滑信号） 。 评测
侧可以按需 export threshold 截断 。 走中间地带（`use_threshold=True + continuous=True`） 表达的是
" 达到某个阈值才计正分 ，否则 0" —— 比 hard-gate 粗一些 。

想复现历史的 0.8 二值截断 ？ 三个都设 ：`continuous: false` + `use_threshold: true` +
`threshold: 0.8` 。

### 可选：最终 reward 脚本

静态权重只表达线性加权。 需要“至少通过 N 条”“指定 A 通过且总数超过 N”或其它跨 check 关系时，
可以配置：

```yaml
config:
  reward:
    script: reward.py
    timeout_sec: 15
```

所有 checks 结束后，`reward.py` 会从 stdin 收到 `rubric_results`、静态 `score_summary` 和
`state_summary`，并在 stdout 输出 `[0, 1]` 内的 JSON number 或 `{"score": number}`。 脚本结果
取代 ScorePolicy，因此 `config.reward` 与 `config.score` 互斥；失败的 hard gate 仍会对
脚本结果封顶。 脚本执行/输出错误会让该样本的 `aidata.score_valid=false`，不回退到静态分。

## config 段 —— trajectory 拍平上限

只对 `judge_claim` / `text_contains` （只要涉及 " 把 trajectory 拍平成一大段字符串 " ） 起作用 。 见
[author-a-rubric.md](../how-to/author-a-rubric.md) config 段 。 代码入口 ：`rubric/config.py::RubricLimits` 。

优先级 ：`run_rubric_async(limits=...)` kwarg > rubric config > 默认值 。

## 并发

v4.2 分派器把每条 check 用 `asyncio.Semaphore` 保护并发评估 。 默认 concurrency = 8 ，可通过
`NEXUS_RUBRIC_CONCURRENCY` 环境变量或 `run_rubric_async(concurrency=N)` 覆盖 。

`concurrency=1` 就是严格串行 ，回到并发化前的行为 。 这对调 flaky judge 有用 —— 串行 log 更好读 。

无论完成顺序如何 ，输出 list 始终按 rubric 声明顺序返回 。 log 会按完成顺序打印
`[rubric] check i/N id=... verdict=...` —— viz Playground 的正则可以稳定匹配 。

## 相关文档

- assert primitive 语义 —— [`../how-to/author-a-rubric.md`](../how-to/author-a-rubric.md) 。
- v4.2 完整字段字典 —— [`../../RUBRIC.md`](../../RUBRIC.md) 。
- 为什么 v2 冻结 —— [`./v2-freeze.md`](./v2-freeze.md) 。
- Fetcher / primitive 演化 —— [`./fetcher-and-primitives.md`](./fetcher-and-primitives.md) 。
