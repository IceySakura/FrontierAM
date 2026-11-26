# galaxy-gravity

粒子数加倍到 131,072、每次调用仍是 5 CPU 秒，使精确的直接求和无法在预算内完成，比较近似算法的精度。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/` | 题目的各个版本；当前版本 `v4/`，题面 `versions/v4/statement.md` |
| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |
| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |
| `analysis/` | 这道题的专题分析与复核记录 |
| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |

## 版本

| 版本 | 评测系统中的名字 | 说明 |
| --- | --- | --- |
| `v1` | `galaxy-gravity` | 出题方原版：65,536 粒子、5 CPU 秒；直接求和可在预算内求精确解 |
| `v2` | `galaxy-gravity-v2` | FrontierAM 重建：131,072 粒子、5 CPU 秒，快照用原包生成器以私有种子生成（当前版本） |
| `v3` | `galaxy-gravity-v3` | 协议 v6.3 版本：出题方的两条 Treecode2 参照（r1），对数刻度；工作区只含样例（scripts/seal/prepare_v63.py） |
| `v4` | `galaxy-gravity-v4` | 出题人 B.4 预标定公开包（0.4.0-precalibration，草稿）：四档规模 65,536–4,194,304 粒子、逐例 CPU 时限、theta=0.8/0.7 两条树作锚点；只含题面与开发数据，私有数据、时限、锚点程序和汇总规则待交付（analysis/B4_PRECALIBRATION.md） |

## 题目背景

N 体引力计算是天体物理模拟的核心：每一步都要算出每个粒子受到其余所有粒子的引力。直接两两求和是 \(O(N^2)\)，规模一大就算不动，因此有树算法（Barnes–Hut，按张角把远处的粒子团当作一个质点）和快速多极子方法（FMM），以精度换速度。题目的本意是比较在固定 CPU 预算下，谁的近似算法精度更高。

**为什么改规模**：原版 65,536 个粒子时，高度优化的直接两两求和能在 5 CPU 秒内算出精确解，近似算法的优劣无法区分。粒子数翻倍后直接求和的耗时乘 4（本机约 15 CPU 秒），树算法参照只乘约 2（约 3–4 CPU 秒）。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v3/statement.md |
| 公开开发材料 | versions/v3/data/samples/（给模型的样例）；versions/v3/data/public/（不给模型的反馈集，只在内部仓库） |
| 私有测试集 | versions/v3/data/private/（只在内部仓库） |
| 评测器 | 原包评测器 tasks/galaxy-gravity/harbor/tests/grade.py（只改输入大小检查与架构检查） |
| 参照实现 | references/r1/（出题方的两条 Treecode2 参照，各版本使用）；参考材料：references/r3/（agent 按 Dehnen 论文写的高阶 FMM）、references/r2/（opus 解法），均不作锚点（results/BASELINE_POLICY.md） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | smoke, case-01 |
| public（不给，反馈用） | 10 张公开快照 |
| 反馈分类 | Hernquist / Jaffe / Einasto / disc / group（各 2 张） |

## 评测依据

- **输入**：二进制快照，每个粒子一行（质量和三维位置）；正式快照每张 131,072 个等质量粒子。
- **数据**：新快照由公开生成器以私有种子生成，参考加速度用原题包的直接求和程序计算；另附 10 张正式规模的公开快照。
- **正式集**：5 类场景（Hernquist 球、Jaffe 球、Einasto 球、扁平盘、四星团群）各 4 张，每张以不同随机种子调用 3 次，共 60 次调用。
- **资源**：每次调用最多 5 个进程 CPU 秒（含启动和读写）、2 GiB 内存、单线程。
- **有效性**：任一次调用超限、异常或输出不合法，整份提交记 0 分。
- **原始指标**：加速度的整体相对 RMS 误差（越小越好）。
- **换算**：设弱参照误差为 \(U\)、强参照为 \(W\)。\(E \ge U\) 记 0 分；\(W \le E < U\) 记 \(50(U-E)/(U-W)\)；\(E < W\) 记 \(100 - 50E/W\)。

## Baseline

两条参照与原题相同（Treecode2 的两组参数），在本机评测核 grade-06 上实测 20 张快照、60 次调用：

| 参照 | 加速度相对 RMS 误差 | 记分 | 单次最多 CPU 秒（上限 5） |
| --- | --- | --- | --- |
| 弱参照 | 0.00125189 | 0 | 4.05 |
| 强参照 | 0.000590528 | 50 | 3.71 |

同样的硬件上，直接两两求和约需 15 CPU 秒，超出上限约 3 倍。

## 结果解读

| 类别 | 结论 | 置信度 | 关注 |
| --- | --- | --- | --- |
| 参照偏弱 | 参照是 1986/1990 年的树算法（出题方 Treecode2，W = 5.9e-4）；模型用快速多极方法把误差降低 3–5 个数量级（sol 迭代 6.9e-10）。按论文独立实现的高阶 FMM 也有 5.7e-5。v6.2 在线性映射下两个模型都挤在 99.99 分，v3 已改为对数刻度。（证据：`results/BASELINE_REVIEW.md`） | 实测 | **需出题人处理** |
| 方法创新（候选） | sol 的核拆分（牛顿部分用调和展开，软化差项用笛卡尔展开）未找到完全相同的文献先例；在人类实现上做消融，这一步贡献最大（1.9× → 9.7×）。需要领域专家确认先例。（证据：`results/BASELINE_REVIEW.md`） | 推断 | 建议人工复核 |
| 题目设置 | 弱参照在私有集上单次最多 4.05 CPU 秒（上限 5 秒），余量 19%，低于 25%。（证据：`results/BASELINE_REVIEW.md`） | 实测 | **需出题人处理** |
| 题面或接口问题 | 本轮题面写“50 分锚点为高阶快速多极方法参照”，与现行参照（Treecode2）不符；下一轮前重新封签。不影响本轮最终评分。（证据：`results/OPEN_ITEMS.md`） | 实测 |  |

## 模型成绩与解法（题目版本 v4 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.2 Sol | 单轮 | \(E = 2.75e-09\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 98%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 72.8 | 57 分钟 | [`v4/gpt-6.2-sol-single-1/`](solutions/v6.3/v4/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | \(E = 1e-07\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 94%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 66.2 | 8 次提交，146 分钟 | [`v4/gpt-6.2-sol-iterative-1/`](solutions/v6.3/v4/gpt-6.2-sol-iterative-1/) |
| GPT-6 Astra | 单轮 | \(E = 0.000242\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 98%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 51.9 | 60 分钟 | [`v4/gpt-6-astra-single-1/`](solutions/v6.3/v4/gpt-6-astra-single-1/) |
| GPT-6 Astra | 迭代 | \(E = 1.5e-08\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 100%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 69.6 | 4 次提交，236 分钟 | [`v4/gpt-6-astra-iterative-1/`](solutions/v6.3/v4/gpt-6-astra-iterative-1/) |
| GPT-6.1 Sol | 单轮 | \(E = 0.0265\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 99%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 0.0 | 60 分钟 | [`v4/gpt-6.1-sol-single-1/`](solutions/v6.3/v4/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | \(E = 1.03e-07\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 92%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 66.1 | 5 次提交，217 分钟 | [`v4/gpt-6.1-sol-iterative-1/`](solutions/v6.3/v4/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | \(E = 9.3e-06\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 63%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 57.8 | 36 分钟 | [`v4/claude-opus-5-5-single-2/`](solutions/v6.3/v4/claude-opus-5-5-single-2/) |
| Claude Opus 5.5 | 迭代 | \(E = 2.58e-07\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 73%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 64.4 | 7 次提交，164 分钟 | [`v4/claude-opus-5-5-iterative-1/`](solutions/v6.3/v4/claude-opus-5-5-iterative-1/) |
| Hunyuan 4 | 单轮 | 无效提交（提交失败） | 任一用例失败或提交检查未通过记 0 | 0.0 | 60 分钟 | [`v4/hy4-single-2/`](solutions/v6.3/v4/hy4-single-2/) |
| Hunyuan 4 | 迭代 | \(E = 0.000122\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 94%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 53.1 | 8 次提交，186 分钟 | [`v4/hy4-iterative-2/`](solutions/v6.3/v4/hy4-iterative-2/) |
| Qwen4 Max | 单轮 | \(E = 0.0462\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 82%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 0.0 | 31 分钟 | [`v4/qwen4-max-single-1/`](solutions/v6.3/v4/qwen4-max-single-1/) |
| Qwen4 Max | 迭代 | 无效提交（提交失败） | 任一用例失败或提交检查未通过记 0 | 0.0 | 10 次提交，189 分钟 | [`v4/qwen4-max-iterative-1/`](solutions/v6.3/v4/qwen4-max-iterative-1/) |
| Qwen3.8 Max | 单轮 | 无效提交（提交失败） | 任一用例失败或提交检查未通过记 0 | 0.0 | 58 分钟 | [`v4/qwen3.8-max-0902-single-2/`](solutions/v6.3/v4/qwen3.8-max-0902-single-2/) |
| Qwen3.8 Max | 迭代 | \(E = 0.000413\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 96%（经典树 theta 0.8：\(E_0 = 0.00175\)，改进树 theta 0.7：\(E_{50} = 0.000665\)） | 对数刻度：\(E < E_{50}\) 为 \(50 + 50\,\log(E_{50}/E)/\log(E_{50}/10^{-15})\)；\(E_{50} \le E < E_0\) 为 \(50\,\log(E_0/E)/\log(E_0/E_{50})\) | 50.9 | 3 次提交，198 分钟 | [`v4/qwen3.8-max-0902-iterative-4/`](solutions/v6.3/v4/qwen3.8-max-0902-iterative-4/) |
| MiniMax M3.1 Flash | 单轮 | 无效提交（build nonzero_exit） | 任一用例失败或提交检查未通过记 0 | 0.0 | 60 分钟 | [`v4/minimax-m3.1-flash-single-1/`](solutions/v6.3/v4/minimax-m3.1-flash-single-1/) |
| MiniMax M3.1 Flash | 迭代 | 无效提交（提交失败） | 任一用例失败或提交检查未通过记 0 | 0.0 | 6 次提交，195 分钟 | [`v4/minimax-m3.1-flash-iterative-1/`](solutions/v6.3/v4/minimax-m3.1-flash-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/v4/gpt-6.2-sol-single-1/source/`](solutions/v6.3/v4/gpt-6.2-sol-single-1/source/)（提交 `galaxy-gravity-v4-20261120T074536Z-7aa5cc`）；未复制的二进制或大文件：solve（171,584 字节）。

### GPT-6.2 Sol（迭代，v6.3）

public 分数随测试提交：59.0（15 分钟） → 63.4（32 分钟） → 64.6（56 分钟） → 0.0（70 分钟） → 63.8（88 分钟） → 65.5（105 分钟） → 65.3（120 分钟） → 65.4（134 分钟）

源码 [`solutions/v6.3/v4/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/v4/gpt-6.2-sol-iterative-1/source/)（提交 `galaxy-gravity-v4-20261120T101428Z-a1e515`）；未复制的二进制或大文件：solve（163,288 字节）。

### GPT-6 Astra（单轮，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

源码 [`solutions/v6.3/v4/gpt-6-astra-single-1/source/`](solutions/v6.3/v4/gpt-6-astra-single-1/source/)（提交 `galaxy-gravity-v4-20261121T170323Z-bc936e`）；未复制的二进制或大文件：solve（195,272 字节）。

### GPT-6 Astra（迭代，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

public 分数随测试提交：0.0（48 分钟） → 64.7（78 分钟） → 69.3（161 分钟） → 69.7（216 分钟）

源码 [`solutions/v6.3/v4/gpt-6-astra-iterative-1/source/`](solutions/v6.3/v4/gpt-6-astra-iterative-1/source/)（提交 `galaxy-gravity-v4-20261121T214846Z-42d260`）；未复制的二进制或大文件：harmonic.hpp（2,335,167 字节）、solve（3,023,312 字节）。

### GPT-6.1 Sol（单轮，v6.3）

源码 [`solutions/v6.3/v4/gpt-6.1-sol-single-1/source/`](solutions/v6.3/v4/gpt-6.1-sol-single-1/source/)（提交 `galaxy-gravity-v4-20261112T130631Z-857089`）；未复制的二进制或大文件：solve（292,088 字节）。

### GPT-6.1 Sol（迭代，v6.3）

public 分数随测试提交：61.3（40 分钟） → 0.0（97 分钟） → 0.0（131 分钟） → 63.6（153 分钟） → 65.8（197 分钟）

源码 [`solutions/v6.3/v4/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/v4/gpt-6.1-sol-iterative-1/source/)（提交 `galaxy-gravity-v4-20261112T154314Z-94db54`）；未复制的二进制或大文件：solve（183,888 字节）。

### Claude Opus 5.5（单轮，v6.3）

第 1 次尝试中有回合因客户端约 300 秒等不到响应头而重发、原请求作废（基础设施故障，记未测）；本格为修复后的第 2 次尝试。

源码 [`solutions/v6.3/v4/claude-opus-5-5-single-2/source/`](solutions/v6.3/v4/claude-opus-5-5-single-2/source/)（提交 `galaxy-gravity-v4-20261116T031513Z-a0e62d`）；未复制的二进制或大文件：solve（151,480 字节）。

### Claude Opus 5.5（迭代，v6.3）

public 分数随测试提交：53.8（49 分钟） → 60.7（69 分钟） → 63.4（83 分钟） → 63.2（105 分钟） → 63.5（125 分钟） → 63.9（139 分钟） → 63.9（145 分钟）

源码 [`solutions/v6.3/v4/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/v4/claude-opus-5-5-iterative-1/source/)（提交 `galaxy-gravity-v4-20261112T145010Z-8ba575`）；未复制的二进制或大文件：solve（323,216 字节）、solve.prebuilt（323,216 字节）。

### Hunyuan 4（单轮，v6.3）

第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断（基础设施故障，记未测）；本格为上游改用流式后的第 2 次尝试。

源码 [`solutions/v6.3/v4/hy4-single-2/source/`](solutions/v6.3/v4/hy4-single-2/source/)（提交 `galaxy-gravity-v4-20261114T090229Z-f3fb46`）；未复制的二进制或大文件：solve（345,264 字节）。

### Hunyuan 4（迭代，v6.3）

第 1 次尝试中有回合因客户端约 13 分钟收不到内容时静默重发、原请求作废（基础设施故障，记未测）；本格为修复后的第 2 次尝试。

public 分数随测试提交：0.0（69 分钟） → 48.8（95 分钟） → 52.2（116 分钟） → 52.9（125 分钟） → 52.6（138 分钟） → 53.5（148 分钟） → 53.3（162 分钟） → 53.3（175 分钟）

源码 [`solutions/v6.3/v4/hy4-iterative-2/source/`](solutions/v6.3/v4/hy4-iterative-2/source/)（提交 `galaxy-gravity-v4-20261115T214833Z-ba9b4d`）；未复制的二进制或大文件：solve（81,768 字节）。

### Qwen4 Max（单轮，v6.3）

源码 [`solutions/v6.3/v4/qwen4-max-single-1/source/`](solutions/v6.3/v4/qwen4-max-single-1/source/)（提交 `galaxy-gravity-v4-20261125T101622Z-a9fd2e`）；未复制的二进制或大文件：solve（79,728 字节）。

### Qwen4 Max（迭代，v6.3）

提交含 985 MB 预计算表，私有评分时 7 个分片同时解包，评测器 30 秒的 `docker create` 超时（评测设施问题，两次记未测）；分片错开 90 秒启动后评完：419 万粒子各例超出 2 GiB 内存上限，100 万粒子 jaffe 一例超 CPU 上限，判无效。

public 分数随测试提交：0.0（45 分钟） → 0.0（88 分钟） → 0.0（102 分钟） → 0.0（113 分钟） → 0.0（123 分钟） → 0.0（134 分钟） → 0.0（145 分钟） → 0.0（155 分钟） → —（166 分钟） → —（178 分钟）（—：没有拿到报告）

源码 [`solutions/v6.3/v4/qwen4-max-iterative-1/source/`](solutions/v6.3/v4/qwen4-max-iterative-1/source/)（提交 `galaxy-gravity-v4-20261125T160512Z-fe2477`）；未复制的二进制或大文件：fmm_p10_r220.bin（618,396,340 字节）、fmm_p8_r100.bin（127,861,332 字节）、fmm_p8_r140.bin（250,147,412 字节）、libgfortran-040039e1.so.5.0.0（2,686,065 字节）、libopenblasp-r0-23e5df77.3.21.dev.so（32,957,169 字节）、libquadmath-96973f99.so.0.0.0（247,609 字节）、solve（143,848 字节）。

### Qwen3.8 Max（单轮，v6.3）

第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断（基础设施故障，记未测）；本格为上游改用流式后的第 2 次尝试。

源码 [`solutions/v6.3/v4/qwen3.8-max-0902-single-2/source/`](solutions/v6.3/v4/qwen3.8-max-0902-single-2/source/)（提交 `galaxy-gravity-v4-20261114T090056Z-bf3072`）；未复制的二进制或大文件：solve（55,104 字节）。

### Qwen3.8 Max（迭代，v6.3）

第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断；第 2、3 次尝试中有回合因上游流中断后改发非流式请求被切断，或因客户端约 13 分钟收不到内容时静默重发、原请求作废（均为基础设施故障，记未测）；本格为修复后的第 4 次尝试。

public 分数随测试提交：21.5（82 分钟） → 50.3（108 分钟） → 50.1（167 分钟）

源码 [`solutions/v6.3/v4/qwen3.8-max-0902-iterative-4/source/`](solutions/v6.3/v4/qwen3.8-max-0902-iterative-4/source/)（提交 `galaxy-gravity-v4-20261115T223048Z-5f539f`）；未复制的二进制或大文件：solve（38,768 字节）。

### MiniMax M3.1 Flash（单轮，v6.3）

`build.sh` 把 CPU 检测小程序编译到 `/tmp` 再执行；评测构建容器的 `/tmp` 不可执行（Docker tmpfs 默认 noexec），开发容器可执行，题面未写明，构建失败，按题目规则记 0（已记入 results/OPEN_ITEMS.md 待反馈出题人）。60 分钟到时截停。

源码 [`solutions/v6.3/v4/minimax-m3.1-flash-single-1/source/`](solutions/v6.3/v4/minimax-m3.1-flash-single-1/source/)（提交 `galaxy-gravity-v4-20261125T103135Z-c4cfa1`）；未复制的二进制或大文件：solve（59,792 字节）。

### MiniMax M3.1 Flash（迭代，v6.3）

最大档 3 例（419 万粒子）三次重复都超 CPU 上限（100–110 秒；所在分片负载探针 1.06–1.10），判无效。

public 分数随测试提交：0.0（94 分钟） → 0.0（142 分钟） → 0.0（151 分钟） → 0.0（162 分钟） → 0.0（173 分钟） → 0.0（184 分钟）

源码 [`solutions/v6.3/v4/minimax-m3.1-flash-iterative-1/source/`](solutions/v6.3/v4/minimax-m3.1-flash-iterative-1/source/)（提交 `galaxy-gravity-v4-20261125T171414Z-efa660`）；未复制的二进制或大文件：solve（55,288 字节）。

<details><summary>旧版本的成绩与解法</summary>

## 模型成绩与解法（题目版本 v3 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | \(E = 1.22e-07\)，单次最多 4.56 CPU 秒（出题方 Treecode2：\(U = 0.00125\)，\(W = 0.000591\)） | 对数刻度：\(E < W\) 为 \(50 + 50\,\log(W/E)/\log(W/10^{-15})\)；\(W \le E < U\) 为 \(50\,\log(U/E)/\log(U/W)\) | 65.7 | 56 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | \(E = 6.91e-10\)，单次最多 4.89 CPU 秒（出题方 Treecode2：\(U = 0.00125\)，\(W = 0.000591\)） | 对数刻度：\(E < W\) 为 \(50 + 50\,\log(W/E)/\log(W/10^{-15})\)；\(W \le E < U\) 为 \(50\,\log(U/E)/\log(U/W)\) | 75.2 | 5 次提交，226 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | \(E = 0.000107\)，单次最多 3.09 CPU 秒（出题方 Treecode2：\(U = 0.00125\)，\(W = 0.000591\)） | 对数刻度：\(E < W\) 为 \(50 + 50\,\log(W/E)/\log(W/10^{-15})\)；\(W \le E < U\) 为 \(50\,\log(U/E)/\log(U/W)\) | 53.1 | 15 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | \(E = 1.47e-09\)，单次最多 3.95 CPU 秒（出题方 Treecode2：\(U = 0.00125\)，\(W = 0.000591\)） | 对数刻度：\(E < W\) 为 \(50 + 50\,\log(W/E)/\log(W/10^{-15})\)；\(W \le E < U\) 为 \(50\,\log(U/E)/\log(U/W)\) | 73.8 | 12 次提交，181 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |

### GPT-6.1 Sol（单轮，v6.3）

笛卡尔与调和多极展开结合的 FMM，远场也保留 Plummer 软化修正；按相互作用选展开阶数，按进程 CPU 时间控制工作量。

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `galaxy-gravity-v3-20261110T233945Z-494532`）；未复制的二进制或大文件：solve（103,872 字节）。

### GPT-6.1 Sol（迭代，v6.3）

核拆分：牛顿部分用调和多极展开，软化差项用笛卡尔修正；按调用规划精度，剩余 CPU 时间对部分粒子做完整直接求和。

public 分数随测试提交：45.2（29 分钟） → 56.8（67 分钟） → 60.1（181 分钟） → 62.1（205 分钟） → 63.5（212 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `galaxy-gravity-v3-20261110T145613Z-509fb2`）；未复制的二进制或大文件：solve（545,312 字节）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `galaxy-gravity-v3-20261112T085134Z-da2912`）；未复制的二进制或大文件：fmm_avx2.o（29,104 字节）、fmm_gen.o（23,880 字节）、main.o（4,264 字节）、solve（52,664 字节）。

### Claude Opus 5.5（迭代，v6.3）

public 分数随测试提交：45.5（74 分钟） → 53.9（100 分钟） → 56.7（106 分钟） → 59.8（120 分钟） → 0.0（124 分钟） → 58.7（137 分钟） → 58.8（144 分钟） → 59.8（150 分钟） → 60.4（158 分钟） → 61.3（165 分钟） → 61.4（169 分钟） → 61.6（174 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `galaxy-gravity-v3-20261112T113721Z-0c35f8`）；未复制的二进制或大文件：solve（370,112 字节）。

## 模型成绩与解法（题目版本 v2 · 协议 v6.2）

协议 v6.2 单轮（2026-11-09 开发，11-10 评分）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | \(L = 4.25\times10^{-9}\)，单次最多 4.45 CPU 秒 | \(100 - 50\,L/\text{强}\) | 100.0 | 106 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.2/gpt-6.1-sol-single-1/) |
| Claude Opus 5.5 | 单轮 | \(L = 1.06\times10^{-7}\)，单次最多 3.16 CPU 秒 | \(100 - 50\,L/\text{强}\) | 100.0 | 63 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |

### GPT-6.1 Sol（单轮，v6.2）

二叉空间树上的高阶调和多极展开（16 阶）处理牛顿核，软化核与牛顿核之差用 10 阶笛卡尔展开，近场直接求和；按当前输入标定的代价模型选择精度，并带计时保护。误差比强参照小五个数量级。

源码 [`solutions/v6.2/gpt-6.1-sol-single-1/source/`](solutions/v6.2/gpt-6.1-sol-single-1/source/)（提交 `galaxy-gravity-v2-20261109T132936Z-9e4112`）；未复制的二进制或大文件：solve（1,241,096 字节）。

### Claude Opus 5.5（单轮，v6.2）

笛卡尔坐标快速多极方法（FMM，双树遍历，11 阶展开，AVX2 向量化，Plummer 软化，双精度）。提交里没有 README。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `galaxy-gravity-v2-20261109T130056Z-aea0a3`）；未复制的二进制或大文件：solve（430,048 字节）。

## 模型成绩与解法（题目版本 v1 · 协议 v6.1）

协议 v6.1（2026-11-06）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol |  | 相对 RMS 误差 3.5e-15，60/60 次有效 | \(100-50E/W\) | 100.0 | 15 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.1/gpt-6.1-sol-single-1/) |

### GPT-6.1 Sol（，v6.1）

没有用近似算法，而是在 5 CPU 秒内用**直接两两求和**算出了精确解（本机每次 3.81–3.84 CPU 秒）：AVX2/FMA 向量化、单精度倒数平方根加修正回到双精度、利用牛顿第三定律对称减半、分块和等质量特化。这说明题目的规模与时限组合允许暴力求精确解，近似算法的优劣无法区分；已反馈出题人，建议增大粒子数或收紧时限。

源码 [`solutions/v6.1/gpt-6.1-sol-single-1/source/`](solutions/v6.1/gpt-6.1-sol-single-1/source/)（提交 `galaxy-gravity-20261106T040942Z-5de3b4`）；未复制的二进制或大文件：solve（22,408 字节）。

</details>

## 观察与说明

加倍粒子数（v2）让直接求和超时；模型都改用高阶快速多极方法，把误差降到 \(10^{-7}\) 以下。v2 的线性映射因此饱和（误差在 \(10^{-6}\) 以下都挤在 99.9 分以上），v3 改为对数刻度，参照仍是出题方的两条 Treecode2。

弱参照在私有集上单次最多 4.05 CPU 秒，离 5 秒上限只有 19% 余量，已列入待出题人处理的事项。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
