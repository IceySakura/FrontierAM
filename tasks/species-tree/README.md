# species-tree

50 棵互相矛盾的估计基因树、250 个物种、120 秒；数据由我们按标准模拟流程重建并校准到原题的统计量，200 个正式用例。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/` | 题目的各个版本；当前版本 `v3/`，题面 `versions/v3/statement.md` |
| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |
| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |
| `analysis/` | 这道题的专题分析与复核记录 |
| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |

## 版本

| 版本 | 评测系统中的名字 | 说明 |
| --- | --- | --- |
| `v1` | `species-tree` | 出题方原版：101 物种、50 例；出题人告知数据有误，暂停 |
| `v2` | `species-tree-v2` | FrontierAM 重建：250 物种、200 例私有 + 50 例公开、每例 180 秒；模拟流程 versions/v2/generation/（当前版本） |
| `v3` | `species-tree-v3` | 协议 v6.3 版本：每例 135 秒；private 96 例、public 反馈 20 例（003–022，不含给模型的 2 个样例；2026-11-26 前为 001–005）；工作区只含 2 个样例（scripts/seal/prepare_v63.py） |

## 题目背景

不同基因的进化历史并不完全相同：祖先种群里的谱系分选不完全（ILS）会让基因树与物种树不一致，短序列又会带来基因树估计误差。系统基因组学的标准做法是“先估计每个基因的树，再汇总成物种树”，代表方法有基于四元组的 ASTRAL、TREE-QMC，以及基于距离的方法。题目考察在大量冲突和噪声下恢复物种树的能力。

**为什么重建**：原版（101 个物种、50 个正式用例）两条参照每例只用 5–7 秒、时限 120 秒，时间预算几乎不起约束；而 50 个用例上一个候选平均误差的标准误折合约 15 分。重建版把物种数扩到 250（强参照每例约 40–70 秒，时间开始真正约束算法），正式用例扩到 200（标准误约减半）。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v3/statement.md |
| 公开开发材料 | versions/v3/data/samples/（给模型的样例）；versions/v3/data/public/（不给模型的反馈集，只在内部仓库） |
| 私有测试集 | versions/v3/data/private/（只在内部仓库） |
| 评测器 | 原包 1.0.1 的批量评测器与沙箱（物种数改为 250，时限 180 秒） |
| 参照实现 | references/r1/（TREE-QMC 3.0.4 不加权 / 加权；锚点 anchors-private.json、anchors-public.json） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 001, 002 |
| public（不给，反馈用） | 20 个反馈用例（003–022；不含样例 001–002） |
| 反馈分类 | 不分类，只报总计 |

## 评测依据

- **输入**：50 行 Newick，每行一棵覆盖 250 个叶子的估计基因树（外群 `0`，内群 `1`–`249`），带枝长和 aBayes 支持度。
- **数据生成**：SimPhy 1.0.2 在多物种溯祖模型下生成物种树和基因树，AliSim 模拟每个基因 200 bp 的序列，IQ-TREE 2.3.6（快速搜索、GTR+G4、aBayes）估计基因树。参数在 101 个物种上校准到原题封签记录的统计量（真基因树不一致度 0.463、基因树估计误差 0.468），模拟数据上两条参照的误差（0.131 / 0.109）与原题锚点（0.128 / 0.108）几乎一致；250 个物种时调整树高使不一致度仍约 0.46。
- **正式集（v3）**：96 个用例；工作区样例 2 个（带真值）；迭代设置的反馈集 20 个（同一组模拟参数、不同种子，不发给模型；2026-11-26 前为 5 个，其中 2 个就是样例）。
- **资源**：每例单核、4 GiB、墙钟 135 秒（v3）。
- **原始指标**：漏掉的真实分裂总数 / (用例数 × 247)，即平均归一化 RF 误差。
- **换算**：与原题相同。设不加权 TREE-QMC 的误差为 \(U\)、加权版为 \(W\)：\(E \ge U\) 记 0 分；\(W \le E < U\) 记 \(50(U-E)/(U-W)\)；\(E < W\) 记 \(100-50E/W\)。
- **评分器**：原题包的批量评分器和沙箱，只把固定的 101 个物种 / 98 个分裂改为 250 / 247。

## Baseline

两条参照与原题相同（TREE-QMC 3.0.4 不加权 / 加权），在本机 4 个评测核上分片实测 200 个正式用例：

| 参照 | 方法 | 漏掉的分裂（/ 49,400） | 批误差 | 记分 | 每例耗时（中位 / 最多） |
| --- | --- | --- | --- | --- | --- |
| 弱参照 | 不加权 TREE-QMC（只用基因树拓扑） | 7,423 | 0.150263 | 0 | 32 / 62.5 秒 |
| 强参照 | 加权 TREE-QMC（按分支支持度和长度加权四元组） | 6,569 | 0.132976 | 50 | 48 / 105.5 秒 |

每例时限 120 秒，两条参照 200/200 例都在时限内；强参照最慢一例离时限只有 12%。

公开开发集（50 例，给模型的公开数据）上两条参照的批误差为 0.158462 / 0.141700，公开集略难于私有集，两条参照的差距相近。

## 结果解读

| 类别 | 结论 | 置信度 | 关注 |
| --- | --- | --- | --- |
| 工程或调参小幅超过参照 | 迭代 52.6（95% 区间 51.3–53.9），确实高于参照，但幅度小；单轮 49.3（39.8–51.2）；迭代比单轮高 1.9–11.9 分。做法与 Weighted ASTRAL 同类（支持度加权四元组、NJ 起点、NNI）。（证据：`scripts/analyze/v63-bootstrap.py`） | 实测 |  |
| 参照不弱 | 更强候选 Weighted ASTRAL（ASTER，文档默认设置）实测更差：E = 0.1896，6/96 例超时；参照 TREE-QMC 加权版保留。（证据：`results/BASELINE_REVIEW.md`） | 实测 |  |
| 题目或数据缺陷（已修复） | v1 原始数据有误（出题人告知），v6.1 成绩基于 v1；v2 起为重建数据。（证据：`tasks/species-tree/README.md`） | 实测 |  |

## 模型成绩与解法（题目版本 v3 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.2 Sol | 单轮 | nRF \(E = 0.1259\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 52.3 | 46 分钟 | [`gpt-6.2-sol-single-1/`](solutions/v6.3/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | nRF \(E = 0.1253\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 52.5 | 6 次提交，50 分钟 | [`gpt-6.2-sol-iterative-1/`](solutions/v6.3/gpt-6.2-sol-iterative-1/) |
| GPT-6 Astra | 单轮 | nRF \(E = 0.1196\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 54.7 | 58 分钟 | [`gpt-6-astra-single-1/`](solutions/v6.3/gpt-6-astra-single-1/) |
| GPT-6 Astra | 迭代 | nRF \(E = 0.1227\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 53.5 | 22 次提交，227 分钟 | [`gpt-6-astra-iterative-1/`](solutions/v6.3/gpt-6-astra-iterative-1/) |
| GPT-6.1 Sol | 单轮 | nRF \(E = 0.1323\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 49.3 | 58 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | nRF \(E = 0.1253\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 52.6 | 13 次提交，128 分钟 | [`gpt-6.1-sol-iterative-2/`](solutions/v6.3/gpt-6.1-sol-iterative-2/) |
| Claude Opus 5.5 | 单轮 | nRF \(E = 0.1389\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 28.1 | 26 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | nRF \(E = 0.1288\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 51.2 | 4 次提交，110 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Hunyuan 4 | 单轮 | nRF \(E = 0.1574\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 58 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | nRF \(E = 0.1382\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 30.1 | 4 次提交，157 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Qwen4 Max | 单轮 | nRF \(E = 0.2038\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 44 分钟 | [`qwen4-max-single-1/`](solutions/v6.3/qwen4-max-single-1/) |
| Qwen4 Max | 迭代 | nRF \(E = 0.1393\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 26.6 | 12 次提交，96 分钟 | [`qwen4-max-iterative-1/`](solutions/v6.3/qwen4-max-iterative-1/) |
| Qwen3.8 Max | 单轮 | nRF \(E = 0.2013\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 30 分钟 | [`qwen3.8-max-0902-single-3/`](solutions/v6.3/qwen3.8-max-0902-single-3/) |
| Qwen3.8 Max | 迭代 | nRF \(E = 0.1664\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 2 次提交，230 分钟 | [`qwen3.8-max-0902-iterative-4/`](solutions/v6.3/qwen3.8-max-0902-iterative-4/) |
| MiniMax M3.1 Flash | 单轮 | nRF \(E = 0.2155\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 53 分钟 | [`minimax-m3.1-flash-single-1/`](solutions/v6.3/minimax-m3.1-flash-single-1/) |
| MiniMax M3.1 Flash | 迭代 | nRF \(E = 0.1368\)（\(U = 0.1476\)，\(W = 0.1320\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 34.7 | 4 次提交，66 分钟 | [`minimax-m3.1-flash-iterative-1/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.2-sol-single-1/source/`](solutions/v6.3/gpt-6.2-sol-single-1/source/)（提交 `species-tree-v3-20261120T073917Z-2ed294`）。

### GPT-6.2 Sol（迭代，v6.3）

public 分数随测试提交：0.0（2 分钟） → 38.1（6 分钟） → 52.6（20 分钟） → 55.3（36 分钟） → 55.0（44 分钟） → 55.3（48 分钟）

源码 [`solutions/v6.3/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/gpt-6.2-sol-iterative-1/source/)（提交 `species-tree-v3-20261120T084646Z-28c6ee`）。

### GPT-6 Astra（单轮，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

源码 [`solutions/v6.3/gpt-6-astra-single-1/source/`](solutions/v6.3/gpt-6-astra-single-1/source/)（提交 `species-tree-v3-20261121T170139Z-c2ab8b`）；未复制的二进制或大文件：solver（222,376 字节）。

### GPT-6 Astra（迭代，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

public 分数随测试提交：21.4（13 分钟） → 0.0（16 分钟） → 42.9（18 分钟） → 40.5（26 分钟） → 47.6（29 分钟） → 33.3（32 分钟） → 45.2（37 分钟） → 38.1（39 分钟） → 52.6（44 分钟） → 51.0（60 分钟） → 51.0（73 分钟） → 53.0（79 分钟） → 51.7（82 分钟） → 51.3（89 分钟） → 53.3（94 分钟） → 53.3（103 分钟） → 52.3（119 分钟） → 54.6（129 分钟） → 55.0（143 分钟） → 55.6（162 分钟） → 56.0（182 分钟） → 56.0（209 分钟）

源码 [`solutions/v6.3/gpt-6-astra-iterative-1/source/`](solutions/v6.3/gpt-6-astra-iterative-1/source/)（提交 `species-tree-v3-20261121T214329Z-62bb56`）；未复制的二进制或大文件：species-tree（210,856 字节）。

### GPT-6.1 Sol（单轮，v6.3）

支持度加权的内部节点距离 NJ 起点、加权四元组 NNI；最多 60 个基因座自助树，再用动态规划求兼容分裂频率最大的共识树。

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `species-tree-v3-20261110T201658Z-e087ed`）；未复制的二进制或大文件：solver（112,720 字节）。

### GPT-6.1 Sol（迭代，v6.3）

第 1 次尝试中有回合被 ATR 本地代理当时默认的 600 秒连接空闲超时切断（基础设施问题，记未测）；本格为 sol 改走 AI Hub azure 通道后的第 2 次尝试。

public 分数随测试提交：33.3（7 分钟） → 51.0（11 分钟） → 23.8（15 分钟） → 51.3（22 分钟） → 52.0（24 分钟） → 45.2（29 分钟） → 51.3（35 分钟） → 53.0（44 分钟） → 50.0（54 分钟） → 58.9（73 分钟） → 59.9（86 分钟） → 60.6（106 分钟） → 60.6（124 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-2/source/`](solutions/v6.3/gpt-6.1-sol-iterative-2/source/)（提交 `species-tree-v3-20261117T063610Z-197bc5`）；未复制的二进制或大文件：infer（108,856 字节）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `species-tree-v3-20261112T080924Z-453688`）；未复制的二进制或大文件：stree_avx2（140,240 字节）、stree_generic（136,064 字节）。

### Claude Opus 5.5（迭代，v6.3）

public 分数随测试提交：38.1（11 分钟） → 54.6（22 分钟） → 55.6（56 分钟） → 55.6（92 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `species-tree-v3-20261112T074642Z-0c4419`）；未复制的二进制或大文件：stree（131,296 字节）、stree_generic（131,352 字节）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `species-tree-v3-20261113T114818Z-2aedb2`）；未复制的二进制或大文件：species_tree（144,144 字节）。

### Hunyuan 4（迭代，v6.3）

public 分数随测试提交：50.0（77 分钟） → 51.3（105 分钟） → 50.7（133 分钟） → 51.3（150 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `species-tree-v3-20261113T151956Z-b398a9`）。

### Qwen4 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen4-max-single-1/source/`](solutions/v6.3/qwen4-max-single-1/source/)（提交 `species-tree-v3-20261125T110027Z-121f1d`）；未复制的二进制或大文件：species（136,080 字节）。

### Qwen4 Max（迭代，v6.3）

public 分数随测试提交：0.0（20 分钟） → 0.0（23 分钟） → 0.0（28 分钟） → 0.0（39 分钟） → 0.0（43 分钟） → 0.0（47 分钟） → 38.1（56 分钟） → 38.1（61 分钟） → 50.0（66 分钟） → 52.0（72 分钟） → 52.3（77 分钟） → 52.6（89 分钟）

源码 [`solutions/v6.3/qwen4-max-iterative-1/source/`](solutions/v6.3/qwen4-max-iterative-1/source/)（提交 `species-tree-v3-20261125T162613Z-165802`）；未复制的二进制或大文件：species_tree（168,440 字节）。

### Qwen3.8 Max（单轮，v6.3）

第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断；第 2 次尝试中有回合因客户端约 13 分钟收不到内容时静默重发、原请求作废（均为基础设施故障，记未测）；本格为修复后的第 3 次尝试。

源码 [`solutions/v6.3/qwen3.8-max-0902-single-3/source/`](solutions/v6.3/qwen3.8-max-0902-single-3/source/)（提交 `species-tree-v3-20261115T191211Z-05ca3a`）；未复制的二进制或大文件：species（255,056 字节）。

### Qwen3.8 Max（迭代，v6.3）

第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断；第 2、3 次尝试中有回合因上游流中断后改发非流式请求被切断，或因客户端约 13 分钟收不到内容时静默重发、原请求作废（均为基础设施故障，记未测）；本格为修复后的第 4 次尝试。

public 分数随测试提交：0.0（181 分钟） → 0.0（202 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-4/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-4/source/)（提交 `species-tree-v3-20261115T230408Z-072208`）；未复制的二进制或大文件：species_tree（138,424 字节）。

### MiniMax M3.1 Flash（单轮，v6.3）

源码 [`solutions/v6.3/minimax-m3.1-flash-single-1/source/`](solutions/v6.3/minimax-m3.1-flash-single-1/source/)（提交 `species-tree-v3-20261125T111247Z-d31cdf`）。

### MiniMax M3.1 Flash（迭代，v6.3）

public 分数随测试提交：—（17 分钟） → 31.0（19 分钟） → 47.6（41 分钟） → 50.0（60 分钟）（—：没有拿到报告）

源码 [`solutions/v6.3/minimax-m3.1-flash-iterative-1/source/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/source/)（提交 `species-tree-v3-20261125T174939Z-d6abf0`）。

<details><summary>旧版本的成绩与解法</summary>

## 模型成绩与解法（题目版本 v2 · 协议 v6.2）

协议 v6.2 单轮（2026-11-09 开发，11-10 评分）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | 漏掉 49,400 / 49,400 个分裂，\(E = 1.0\) | \(E \ge U\) 记 0 | 0.0 | 4 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.2/gpt-6.1-sol-single-1/) |
| Claude Opus 5.5 | 单轮 | 漏掉 6,700 个分裂，\(E = 0.135628\) | \(50\,(U-E)/(U-W)\) | 42.3 | 66 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |

### GPT-6.1 Sol（单轮，v6.2）

4.4 分钟就结束回合。程序每例 0.13 秒输出一棵 250 叶的二叉树，但与真树没有一个共同分裂：输出是一棵平衡树，最终说明写的是“选最大校正距离”合并，与邻接法（合并距离最近的一对）方向相反。它在公开用例 001 上用自带的 `check_rf.py` 测到 `normalized_rf = 1.0`（最差），没有处理，只检查了格式、可复现性和干净构建。

源码 [`solutions/v6.2/gpt-6.1-sol-single-1/source/`](solutions/v6.2/gpt-6.1-sol-single-1/source/)（提交 `species-tree-v2-20261109T124138Z-536e02`）；未复制的二进制或大文件：infer（61,432 字节）。

### Claude Opus 5.5（单轮，v6.2）

C++ 实现的支持度加权四元组方法（思路同加权 ASTRAL）：按 aBayes 支持度给基因树四元组加权，在候选类簇集合上做精确动态规划，候选集由基因树类簇、邻接法树、NNI 邻居和 24 次基因自助重抽样逐步扩充。每例中位 4.2 秒、最多 11.1 秒。逐例与加权 TREE-QMC 比：82 例更好、18 例相同、100 例更差，平均每例多漏 0.66 个分裂（95% 区间 0.07–1.20），显著差于强参照；与不加权参照比 150 例更好。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `species-tree-v2-20261109T134305Z-2caffd`）。

## 模型成绩与解法（题目版本 v1 · 协议 v6.1）

协议 v6.1（2026-11-06）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol |  | 暂停评测（提交已冻结） | — | — | 101 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.1/gpt-6.1-sol-single-1/) |

### GPT-6.1 Sol（，v6.1）

源码 [`solutions/v6.1/gpt-6.1-sol-single-1/source/`](solutions/v6.1/gpt-6.1-sol-single-1/source/)（提交 `species-tree-20261106T053955Z-34ce6b`）；未复制的二进制或大文件：infer（283,384 字节）。

</details>

## 观察与说明

- 数据、种子和生成流程见 `tasks/species-tree/analysis/SPECIES_TREE_SIM.md`（种子不公开）。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
