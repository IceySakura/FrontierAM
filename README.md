# FrontierAM

FrontierAM 收集**有明确对错、可复现评分**的数值算法题，让 AI 编程模型在断网环境里限时（2 小时）从零写出求解程序，再在隐藏测试集上用题目自带的评测器打分，并与在同一台机器上重测的人类参照实现（baseline）比较。

## 榜单（协议 v6.2）

<!-- board:start -->
协议 v6.3；更早协议与题目版本的成绩见各题 README。

| task | GPT-6.1 Sol 单轮 | GPT-6.1 Sol 迭代 |
| --- | --- | --- |
| [conic](tasks/conic/README.md) | 0.0 | 0.0 |
| [exponential](tasks/exponential/README.md) | 55.2 | 59.6 |
| [galaxy-gravity](tasks/galaxy-gravity/README.md) | 65.7 | 75.2 |
| [polarization-demosaicking](tasks/polarization-demosaicking/README.md) | 53.5 | 52.9 |
| [trajectory-covariance](tasks/trajectory-covariance/README.md) | 0.0 | 62.3 |
| [species-tree](tasks/species-tree/README.md) | 49.3 | 52.6 |
| [ihvp-sequence](tasks/ihvp-sequence/README.md) | 100.0 | 100.0 |

满分 100，参照 = 50。

**提交次数与用时**

| task | GPT-6.1 Sol 单轮 | GPT-6.1 Sol 迭代 |
| --- | --- | --- |
| [conic](tasks/conic/README.md) | 57 分钟 | 7 次提交，238 分钟 |
| [exponential](tasks/exponential/README.md) | 54 分钟 | 7 次提交，186 分钟 |
| [galaxy-gravity](tasks/galaxy-gravity/README.md) | 56 分钟 | 5 次提交，226 分钟 |
| [polarization-demosaicking](tasks/polarization-demosaicking/README.md) | 52 分钟 | 23 次提交，211 分钟 |
| [trajectory-covariance](tasks/trajectory-covariance/README.md) | 2 分钟 | 19 次提交，127 分钟 |
| [species-tree](tasks/species-tree/README.md) | 58 分钟 | 13 次提交，237 分钟 |
| [ihvp-sequence](tasks/ihvp-sequence/README.md) | 13 分钟 | 1 次提交，9 分钟 |
<!-- board:end -->

本仓库不含私有评测集（各题的私有测试数据另行保管，需要时单独提供）。

## 题目目录结构

每个题目目录：

```
tasks/<题目>/
├── README.md        阅读入口：背景、版本、题面/数据/评测器/参照的位置、评测依据、baseline、模型成绩与解法
├── meta.json        本题的数据源：说明文字、版本列表、各部分位置、已发布成绩（README 与看板都由它生成）
├── versions/<版本>/  每个题目版本一个目录：题面 statement.md、data/samples/（给模型的样例）、data/public/ 与 data/private/（反馈集与私有集，只在内部仓库）、封存说明
├── references/      参照实现：r1/ 为现行参照，candidates/ 为候选（更换参照时新增 r<k>/，旧成绩仍对应原参照）
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

## 评测协议（v6.3）

每个模型在每道题、每种设置下的每次尝试独立进行：一个物理核、4 GiB、断网，模型用各自原生的编程助手。

- **数据**：samples 给模型；public 不给，只用来在迭代设置中反馈；private 用于最终评分。
- **单轮**：1 小时，开发中没有反馈。
- **迭代**：4 小时，可随时提交测试：快照后在后台评测 public（时钟不停），报告给出 public 分数和按类别与参照的对比，不列逐例结果。
- **评测时长**：public 每次最多 15 分钟，private 最多 1 小时。

榜单当前展示的是 v6.2（单轮与迭代均为 2 小时）的成绩，v6.3 重跑后替换。详见 [`results/PROTOCOL_V63.md`](results/PROTOCOL_V63.md)。

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
