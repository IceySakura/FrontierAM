# 协议 v6.3（2026-11-10 定稿）

相对 v6.2 的改动：三层数据、单轮 1 小时 / 迭代 4 小时、迭代评测异步、跨题统一的反馈格式、迭代尝试并行、评测时长封顶。用户决定均在 11-10 的讨论中做出（`results/OPEN_ITEMS.md`）。

## 数据三层

| 层 | 给模型 | 用途 | 仓库位置 |
| --- | --- | --- | --- |
| samples | 给（含参考答案和自检工具） | 理解题意和数据格式；从 public 中选取 | `tasks/<题目>/versions/<版本>/data/samples/` |
| public | 不给 | 迭代设置中由评测方评测，返回反馈 | `.../data/public/`（LFS，只在内部仓库） |
| private | 不给 | 最终评分 | `.../data/private/`（LFS，只在内部仓库） |

## 两种设置

| 设置 | 时限 | 开发中的反馈 | 评分对象 |
| --- | --- | --- | --- |
| 单轮 | 1 小时 | 无（只有 samples 自检） | 结束或到时的 `/workspace/submission` |
| 迭代 | 4 小时 | `submit --test` 返回 public 反馈 | `submit --final` 的版本；没有时为到时的 `/workspace/submission` |

题面写明时限和“只按最终私有集得分”，不提示用满时间；早停算模型能力的一部分。记录并展示提交（测试）次数和实际用时。

## 迭代评测（`arena/native.py` EvalServiceV63）

- `submit --test`：从运行中的容器快照 `/workspace/submission`（审查过的导出器，不暂停容器），立即返回；时钟照走。快照时若文件正在变化导致失败，提示重试，不算评测结论。
- 评测在后台进行，作为优先任务排队取一个空闲评测槽（`arena/slots.py`：有测试在等时新的 private 评分不开始；有迭代尝试运行时 private 评分不占最后一个空闲评测槽）。
- 同一时刻最多一个测试在评测中；每次评测从拿到评测槽起最多 15 分钟，到点停止，已评用例照常反馈。
- 报告写入 `/workspace/feedback/test-<n>/report.md`；`submit --status` 显示剩余时间和各次测试的状态与分数。
- 不能只测指定用例（v6.2 的 `--cases`/`--groups` 取消），避免借此得到逐例信息。

## 反馈格式（`arena/feedback.py`，各题适配在 `arena/public_v63.py`）

```
Public-set score: 47.0 / 100 (reference = 50, same mapping as the final score).

| category | cases | passed / wrong / timeout / crash | <accuracy>: you / reference | time: you / reference | score |
```

- 精度和速度只给“你 ÷ 参照”的比值；不给逐例结果，不给参照的绝对数值。
- 只有当用例明显分成本质不同的几类时才分类；否则只报一行总计。每题在 `meta.json` 的 `v63_data.feedback` 中声明类别字段、精度指标及方向、是否计时。
- **参照在公开集上的水平刻意可见（用户确定，11-11）。** 题面公开计分映射，反馈给分数和比值，模型能由此推出参照在公开集上的原始指标；这是“告诉模型哪里不如参照”的必然结果，与 Kaggle（公开榜 + 私有榜）、ALE-Bench（公开打分器与公开测试）、AlgoTune / KernelBench（直接返回相对参照的加速比）一致。不公开的是私有集上的锚点数值、私有输入与真值、参照代码。观察项：迭代尝试是否在公开分刚过 50 后就停止（“过线即停”）。

## 各题 v6.3 版本

| task | 版本（评测系统中的名字） | samples | public（反馈） | 分类 | 时限与规模的改动（`results/EVAL_TIME_LIMITS.md`） |
| --- | --- | --- | --- | --- | --- |
| conic | v3（`conic`，运行副本 `.runtime-conic-v3`，反馈 `.runtime-conic-v3-public`） | 每类最小 1 例，共 5 例 | 145 例 | 锥类型 | 参照解不出的用例 300 → 18 秒 |
| exponential | v3（`exponential-v3`，`.runtime-exp3`，反馈 `.runtime-exp3-public`） | 6 例 | 16 例 | 结构族 | 单请求 15 秒、复用 150 秒、启动 10 秒 |
| galaxy-gravity | v3（`galaxy-gravity-v3`） | smoke + case-01 | 10 张快照 | 场景 | 参照为出题方的两条 Treecode2（r1），对数刻度 |
| polarization-demosaicking | v2（`polarization-demosaicking-v2`） | 合成样例 + 2 景 | 105 景 | 不分类 | 无 |
| species-tree | v3（`species-tree-v3`） | 001、002 | 001–005 | 不分类 | 每例 135 秒；private 96 例 |
| trajectory-covariance | v2（`trajectory-covariance-v2`） | 01、02 | 150 个输入 | 情形 | 每例 5 秒；private 2,400 个输入，分片评测 |

封存脚本：`scripts/seal/prepare_v63.py`（四道 operator 题）、`prepare_exp_v3.py`、`prepare_conic_v3.py`；数据入库：`scripts/seal/export_v63_data.py`。

## 评测时长封顶

- public：每次测试 15 分钟（见上）。
- private：每个评分任务（或分片）最多 3,600 秒（systemd `RuntimeMaxSec`），超时记为“评测超时，未测”（`evaluation_cap`）。各题时限与规模已按理论最坏不超过目标设计，正常情况下不会触发。

## 并行与机器布局

- 迭代尝试与单轮一样并行运行；取消 v6.2 的全机迭代锁和“迭代期间不做其他评分”的规则。
- 机器改为 7 个开发槽 + 7 个评测槽（`paired` 布局：每个 L3 域最多 2 个评测核；`scripts/ops/relayout.py`）。切换后需实测评测核之间的干扰（`./bench interference`），结论写入 `docs/platform/CONCURRENCY.md`。

## 迁移

- 所有格子按 v6.3 重跑；v6.2 成绩保留在各题目录，不再上榜。
- 开始 v6.3 评测前，在评测槽上测量各题 public 参照（`bench public-baseline-v63`）和 private 参照（各版本的 baseline）。
