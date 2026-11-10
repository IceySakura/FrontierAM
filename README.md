# FrontierAM

FrontierAM 收集**有明确对错、可复现评分**的数值算法题，让 AI 编程模型在断网环境里限时（2 小时）从零写出求解程序，再在隐藏测试集上用题目自带的评测器打分，并与在同一台机器上重测的人类参照实现（baseline）比较。

看板（办公网）：http://9.135.91.91:18765/

## 给出题人：从哪里读起

**每道题一个目录，都在 [`problems/`](problems/README.md)。** 打开 `problems/<题目>/README.md`，按顺序是：

1. **题目背景**：题目要解决什么问题。
2. **题面原文** `statement.md`：给模型的版本，逐字相同。
3. **数据与代码在哪里**：题包、公开开发材料（`public/`）、私有测试集、评测器、参照实现、详细文档的位置。
4. **评测依据**：输入、规模、时限、判定规则、原始指标和换算公式。
5. **Baseline**：参照方法和本机实测结果（`reference/` 为参照源码）。
6. **模型成绩与解法**：每份提交的原始指标、换算和得分，我们对其做法的说明，以及完整源码（`solutions/<模型>-<设置>/`）。

| 题目 | 目录 |
| --- | --- |
| 凸锥优化求解器（650 个隐藏实例） | [`problems/conic/`](problems/conic/README.md) |
| 稀疏系统的指数传播（对标 7 个已发表方法） | [`problems/exponential-v2/`](problems/exponential-v2/README.md) |
| 星系快照中的引力加速度（131,072 粒子） | [`problems/galaxy-gravity-v2/`](problems/galaxy-gravity-v2/README.md) |
| 偏振马赛克图像重建 | [`problems/polarization-demosaicking/`](problems/polarization-demosaicking/README.md) |
| 从一条 MCMC 轨迹估计长期协方差 | [`problems/trajectory-covariance/`](problems/trajectory-covariance/README.md) |
| 从短基因的估计树推断物种树（250 物种） | [`problems/species-tree-v2/`](problems/species-tree-v2/README.md) |

`problems/` 由 `scripts/build_problem_pages.py` 生成，内容与看板同源；题目或成绩更新后重新生成。

## 最终得分约定（所有题目）

所有题目的最终得分统一为**满分 100、参考实现 = 50**：

- **参考在本机重跑**：参考实现在本机上重新运行，作为计分锚点（baseline），不使用历史记录或其他机器上的数字。
- **映射的三个锚点**：题目原始指标经单调映射得到最终得分，参考的成绩映射为 50，题目允许的最好成绩映射为 100，无效提交为 0。
- **已有映射的题目**：题目自带以参考为锚点的分数时（例如 0–10 分、参考 5 分），直接线性换算到 100 分。
- **没有映射的新题**：出题时定好映射公式，并写进结果报告。
- **报告内容**：每份结果同时给出原始指标、映射公式和最终得分；基础设施失败记为未测，不记 0 分。

**超过参照的成绩要复核**：任何格子超过 50 分，都要解释超出的原因、寻找更强的人类方法并做消融，结论是“改进参照”或“承认模型的算法创新”，见 [`results/BASELINE_POLICY.md`](results/BASELINE_POLICY.md) 与 [`results/BASELINE_REVIEW.md`](results/BASELINE_REVIEW.md)。

## 评测协议（v6.2）

每个模型在每道题、每种设置下只尝试一次：一个物理核、4 GiB、断网，模型用各自原生的编程助手（Claude Code、Codex 等）。两种设置都是 2 小时：**单轮**（开发中没有反馈）和**迭代**（可在公开集上做测试评测，得到逐例结果和与 baseline 的对比）。只有时间结束时冻结的源码进入评测，评测在独立的评测核上进行。详见 [`results/PROTOCOL_V62.md`](results/PROTOCOL_V62.md)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `problems/` | **每道题的阅读入口**（生成） |
| `results/` | 结果与分析：排行榜、协议、复核记录、各题专题文档、评分运行记录（索引见 [`results/README.md`](results/README.md)） |
| `tasks/` | 题包、公开与私有数据、参照实现、我方重建题目的生成流程 |
| `arena/` | 评测与开发框架：任务定义、评分、开发 harness、机器隔离 |
| `scripts/` | 题目封存、评分流水线、分析与看板构建脚本（索引见 [`scripts/README.md`](scripts/README.md)） |
| `dashboard/` | 看板内容（`tasks/*.json`）与样式；构建脚本 `scripts/build_dashboard.py` |
| `docs/` | 平台与运维：服务器要求、安装、机器隔离、开发与评测流程（[`docs/PLATFORM.md`](docs/PLATFORM.md)） |
| `nexus/`、`docker/`、`fixtures/`、`provenance/` | 固定版本的 Nexus 框架、镜像构建、测试夹具、来源记录 |
| `var/` | 本机运行数据（不进 git）：`var/stores/` 评测与开发记录库，`var/raw/` 出题方原始题包，`var/outputs/` 导出包，`var/dist/` 构建产物 |
| `.runtime*/`、`.venv-*/` | 封存的运行副本与 Python 环境（隐藏目录，不进 git） |
