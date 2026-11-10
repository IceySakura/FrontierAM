# results/：结果与分析

## 先读这些

| 文件 | 内容 |
| --- | --- |
| [`LEADERBOARD.md`](LEADERBOARD.md) | 排行榜：各题、各协议版本的原始指标、换算和得分 |
| [`PROTOCOL_V62.md`](PROTOCOL_V62.md) | 现行评测协议 v6.2（单轮与迭代设置）及其全部结果 |
| [`BASELINE_POLICY.md`](BASELINE_POLICY.md) | 参照（baseline）的选取、冻结、复核原则 |
| [`BASELINE_REVIEW.md`](BASELINE_REVIEW.md) | 超过参照的格子的复核记录 |
| [`NEW_TASKS.md`](NEW_TASKS.md) | 出题方新题（galaxy、polarization、species-tree、trajectory）的接入、问题与结果 |
| [`OPEN_ITEMS.md`](OPEN_ITEMS.md) | 待讨论事项与默认决定 |

## 按题目

| 目录 | 内容 |
| --- | --- |
| [`conic/`](conic/) | 凸锥优化：650 例全集结果（`CONIC_FULL.md`）、逐例分析（`CONIC_PER_CASE.md`）、修正版题包分析（`CONIC_REVISION_SOL.md`）、评分提速（`CONIC_FAST_GRADING.md`）及逐例数据 |
| [`exponential/`](exponential/) | 指数传播 v2：设计、参照组、私有集与成绩（`EXPONENTIAL_V2.md`），sol 的 v6.1 评分记录 |
| [`galaxy-gravity/`](galaxy-gravity/) | 星系引力：规模调整（`GALAXY_SCALE.md`） |
| [`polarization/`](polarization/) | 偏振去马赛克：公开开发集说明（`POLARIZATION_PUBLIC_DEV_REQUEST.md`）、首景计时实验 |
| [`species-tree/`](species-tree/) | 物种树：模拟重建（`SPECIES_TREE_SIM.md`）、规模调整（`SPECIES_TREE_SCALE.md`）、v2 锚点与成绩 |

## 其他

| 目录 | 内容 |
| --- | --- |
| [`runs/`](runs/) | 协议 v6.2 各次私有集评分的汇总（`score.json`），供看板与报告引用 |
| [`platform/`](platform/) | 平台与基础设施：并发干扰实测（`CONCURRENCY.md`）、看板（`DASHBOARD.md`）、存储规划、迁移验证等 |
