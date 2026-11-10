# FrontierAM

FrontierAM 收集**有明确对错、可复现评分**的数值算法题，让 AI 编程模型在断网环境里限时（2 小时）从零写出求解程序，再在隐藏测试集上用题目自带的评测器打分，并与在同一台机器上重测的人类参照实现（baseline）比较。

## 榜单（协议 v6.2）

<!-- board:start -->
| task | GPT-6.1 Sol 单轮 | GPT-6.1 Sol 迭代 | Claude Opus 5.5 单轮 | Claude Opus 5.5 迭代 |
| --- | --- | --- | --- | --- |
| [conic](tasks/conic/README.md) | 0.0 | 0.0 | 24.7 | 41.2 |
| [exponential](tasks/exponential/README.md) | 0.0 | 62.9* | 70.8* | 69.1* |
| [galaxy-gravity](tasks/galaxy-gravity/README.md) | 100.0* |  | 100.0* |  |
| [polarization-demosaicking](tasks/polarization-demosaicking/README.md) |  |  | 55.9* |  |
| [trajectory-covariance](tasks/trajectory-covariance/README.md) |  |  | 51.9* |  |
| [species-tree](tasks/species-tree/README.md) | 0.0 |  | 42.3 |  |

满分 100，参照 = 50；* 待复核（`results/BASELINE_POLICY.md`）。

**提交次数与用时**

| task | GPT-6.1 Sol 单轮 | GPT-6.1 Sol 迭代 | Claude Opus 5.5 单轮 | Claude Opus 5.5 迭代 |
| --- | --- | --- | --- | --- |
| [conic](tasks/conic/README.md) | 120 分钟 | 5 次提交，120 分钟 | 88 分钟 | 5 次提交，80 分钟 |
| [exponential](tasks/exponential/README.md) | 23 分钟 | 4 次提交，68 分钟 | 62 分钟 | 10 次提交，60 分钟 |
| [galaxy-gravity](tasks/galaxy-gravity/README.md) | 106 分钟 |  | 63 分钟 |  |
| [polarization-demosaicking](tasks/polarization-demosaicking/README.md) |  |  | 46 分钟 |  |
| [trajectory-covariance](tasks/trajectory-covariance/README.md) |  |  | 68 分钟 |  |
| [species-tree](tasks/species-tree/README.md) | 4 分钟 |  | 66 分钟 |  |
<!-- board:end -->

本仓库不含私有评测集（各题的私有测试数据另行保管，需要时单独提供）。

## 题目目录结构

每个题目目录：

```
tasks/<题目>/
├── README.md        阅读入口：背景、版本、题面/数据/评测器/参照的位置、评测依据、baseline、模型成绩与解法
├── meta.json        本题的数据源：说明文字、版本列表、各部分位置、已发布成绩（README 与看板都由它生成）
├── versions/<版本>/  每个题目版本一个目录：题面 statement.md、公开材料 public/、私有数据、评测器、数据生成流程、改版说明
├── references/      参照实现：r1/ 为现行参照，candidates/ 为复核中的候选（参照升级时新增 r2/，旧成绩仍对应 r1）
├── solutions/<协议>/<模型>-<设置>-<序号>/   模型冻结提交的源码 source/ 与成绩 result.json、score.json
└── analysis/        这道题的专题分析与复核记录
```

**增加题目、检查或修改现有题目**：流程见 [`docs/TASK_CHANGES.md`](docs/TASK_CHANGES.md)。

## 最终得分约定（所有题目）

所有题目的最终得分统一为**满分 100、参考实现 = 50**：

- **参考在本机重跑**：参考实现在本机上重新运行，作为计分锚点（baseline），不使用历史记录或其他机器上的数字。
- **映射的三个锚点**：题目原始指标经单调映射得到最终得分，参考的成绩映射为 50，题目允许的最好成绩映射为 100，无效提交为 0。
- **已有映射的题目**：题目自带以参考为锚点的分数时（例如 0–10 分、参考 5 分），直接线性换算到 100 分。
- **没有映射的新题**：出题时定好映射公式，并写进结果报告。
- **报告内容**：每份结果同时给出原始指标、映射公式和最终得分；基础设施失败记为未测，不记 0 分。

**超过参照的成绩要复核**：任何格子超过 50 分，都要解释超出的原因、寻找更强的人类方法并做消融，结论是“改进参照”或“承认模型的算法创新”，见 [`results/BASELINE_POLICY.md`](results/BASELINE_POLICY.md) 与 [`results/BASELINE_REVIEW.md`](results/BASELINE_REVIEW.md)。

## 评测协议（v6.2）

每个模型在每道题、每种设置下的每次尝试独立进行：一个物理核、4 GiB、断网，模型用各自原生的编程助手（Claude Code、Codex 等）。两种设置都是 2 小时：**单轮**（开发中没有反馈）和**迭代**（可在公开集上做测试评测，得到逐例结果和与 baseline 的对比）。只有时间结束时冻结的源码进入评测，评测在独立的评测核上进行。详见 [`results/PROTOCOL_V62.md`](results/PROTOCOL_V62.md)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `tasks/` | **每道题的全部内容**（见上） |
| `results/` | 跨题内容：排行榜、协议、参照原则与复核总览、新题接入记录、待讨论事项 |
| `arena/` | 评测与开发框架：任务定义、评分、开发 harness、机器隔离（题目按 `tasks/*/versions/*/task.json` 自动发现） |
| `scripts/` | 按用途分目录：`setup/`、`seal/`、`references/`、`rounds/`、`grade/`、`analyze/`、`site/`、`ops/`（索引见 [`scripts/README.md`](scripts/README.md)） |
| `dashboard/` | 看板的全站文字（`site.json`）与样式；各题内容来自 `tasks/*/meta.json` |
| `docs/` | 平台与运维（[`docs/PLATFORM.md`](docs/PLATFORM.md)、`docs/platform/`） |
| `docker/` | 评测与开发镜像 |
| `nexus/`、`fixtures/`、`provenance/` | 固定版本的 Nexus 框架、测试夹具、来源记录 |
| `var/` | 本机运行数据（不进 git）：`var/stores/` 评测与开发记录库，`var/raw/` 出题方原始题包，`var/outputs/` 导出包，`var/dist/` 构建产物 |
| `.runtime*/`、`.venv-*/` | 封存的运行副本与 Python 环境（隐藏目录，不进 git） |
