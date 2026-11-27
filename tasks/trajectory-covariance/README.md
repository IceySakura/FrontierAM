# trajectory-covariance

只给一条马尔可夫链轨迹，估计样本均值的长期协方差矩阵（MCMC 标准误的核心），30 种设置平均的白化误差越小越好。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/` | 题目的各个版本；当前版本 `v2/`，题面 `versions/v2/statement.md` |
| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |
| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |
| `analysis/` | 这道题的专题分析与复核记录 |
| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |

## 版本

| 版本 | 评测系统中的名字 | 说明 |
| --- | --- | --- |
| `v1` | `trajectory-covariance` | 出题方包 mcmc-long-run-covariance-maintainer-v1；本机评测固定的 3,000 例分层子集（当前版本） |
| `v2` | `trajectory-covariance-v2` | 协议 v6.3 版本：每例 5 秒；private 2,400 个输入（分片）、public 150 个；工作区只含 2 个样例（scripts/seal/prepare_v63.py） |

## 题目背景

MCMC 的输出是相关的样本，不能用独立样本的公式算均值的标准误；需要估计“长期协方差” \(\Sigma = \lim n\,\mathrm{Cov}(\bar X_n)\)，它决定了有效样本量和收敛诊断。经典方法有批均值（BM/OBM）、谱方差估计和基于 VAR 模型的拟合；难点在于不同采样器（随机游走 Metropolis、NUTS、Pólya–Gamma Gibbs、有限状态链）的自相关结构差别很大，而且轨迹长度从几千到几万步不等。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v2/statement.md |
| 公开开发材料 | versions/v2/data/samples/（给模型的样例）；versions/v2/data/public/（不给模型的反馈集，只在内部仓库） |
| 私有测试集 | versions/v2/data/private/（只在内部仓库） |
| 评测器 | 出题方包的验证器（arena/operator_drivers/trajectory_batch.py 启动其验证容器） |
| 参照实现 | references/r1/（OBM，R 包 mcmcse；VAR(1)）；人类候选参照 references/candidates/ |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 01, 02 |
| public（不给，反馈用） | 反馈池：每种情形最后 5 条母链 × 5 个前缀，共 150 个输入（与正式子集不相交） |
| 反馈分类 | 六种情形（三种模拟 / 三种真实采样） |

## 评测依据

- **输入**：一条 \(M \times d\) 的轨迹（\(d\) = 4、5 或 6），坐标可能经过本条轨迹固定的可逆线性变换和平移。
- **正式集**：6 种情形（3 种模拟、3 种真实数据采样器）× 5 种长度 = 30 档，每档 1,000 条母链；全量 3 万个输入。本机默认评测固定的 10% 分层子集（每档前 100 条母链，共 3,000 个输入），出题方的封存 agent 在同一子集上复评一致。
- **资源**：每例单核、2 GiB、墙钟 30 秒（含启动）。
- **原始指标**：每例白化误差 \(e = \|S^{-1/2}(H-S)S^{-1/2}\|_F\)；每档平均后 30 档等权平均得 \(L\)。失败的例按零矩阵记误差。
- **换算**：设 OBM 的 \(L\) 为 \(a\)、拟合 VAR(1) 为 \(b\)。\(L \ge a\) 记 0 分；\(b \le L < a\) 记 \(50(a-L)/(a-b)\)；\(L < b\) 记 \(100-50L/b\)。

## Baseline

| 参照 | 方法 | 子集上的 \(L\) | 记分 |
| --- | --- | --- | --- |
| 弱参照 | 重叠批均值（OBM，R 包 mcmcse） | 0.350317 | 0 |
| 强参照 | 拟合 VAR(1) 模型的长期协方差 | 0.230050 | 50 |

本机评测核实测，与出题方在同一子集上的记录相差不到 2e-15。

## 结果解读

| 类别 | 结论 | 置信度 | 关注 |
| --- | --- | --- | --- |
| 工程或调参超过参照 | 迭代 62.3（L = 0.1728，参照 VAR(1) 0.2290）；受故障影响的原尝试也是 62.3，两次独立的 4 小时尝试结果一致。做法：白化后用信息准则组合多阶向量自回归的零频谱估计，有限状态链用 Markov 泊松方程。实测 7 个人类方法都不如参照（最好的 VARHAC-BIC 为 0.2551），模型的组合超过了所有测过的人类方法。（证据：`results/BASELINE_REVIEW.md`） | 实测（成绩）/ 推断（归因） |  |
| 模型失误 | 单轮只用 2 分钟就结束，把估计除以了轨迹长度，误差约为参照的 10 倍；题面已加粗写明“交付的矩阵是 H”，属于模型失误，不是题面问题。（证据：`tasks/trajectory-covariance/versions/v2/statement.md`） | 实测 |  |

## 模型成绩与解法（题目版本 v2 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.2 Sol | 单轮 | 白化误差 \(E = 0.1654\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 63.9 | 42 分钟 | [`gpt-6.2-sol-single-1/`](solutions/v6.3/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | 白化误差 \(E = 0.1647\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 64.0 | 11 次提交，28 分钟 | [`gpt-6.2-sol-iterative-1/`](solutions/v6.3/gpt-6.2-sol-iterative-1/) |
| GPT-6 Astra | 单轮 | 白化误差 \(E = 0.2051\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 55.2 | 52 分钟 | [`gpt-6-astra-single-1/`](solutions/v6.3/gpt-6-astra-single-1/) |
| GPT-6 Astra | 迭代 | 白化误差 \(E = 0.1661\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 63.7 | 17 次提交，186 分钟 | [`gpt-6-astra-iterative-1/`](solutions/v6.3/gpt-6-astra-iterative-1/) |
| GPT-6.1 Sol | 单轮 | 白化误差 \(E = 2.2638\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 2 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 白化误差 \(E = 0.1728\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 62.3 | 28 次提交，116 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 白化误差 \(E = 0.2172\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 52.6 | 14 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | 白化误差 \(E = 0.1711\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 62.6 | 18 次提交，127 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Hunyuan 4 | 单轮 | 白化误差 \(E = 0.2342\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 47.9 | 53 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | 白化误差 \(E = 0.1862\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 59.3 | 21 次提交，177 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Qwen4 Max | 单轮 | 白化误差 \(E = 0.2225\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 51.4 | 26 分钟 | [`qwen4-max-single-1/`](solutions/v6.3/qwen4-max-single-1/) |
| Qwen4 Max | 迭代 | 白化误差 \(E = 0.2244\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 51.0 | 16 次提交，47 分钟 | [`qwen4-max-iterative-1/`](solutions/v6.3/qwen4-max-iterative-1/) |
| Qwen3.8 Max | 单轮 | 白化误差 \(E = 0.5920\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 0.0 | 35 分钟 | [`qwen3.8-max-0902-single-1/`](solutions/v6.3/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | 白化误差 \(E = 0.2240\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 51.1 | 5 次提交，169 分钟 | [`qwen3.8-max-0902-iterative-1/`](solutions/v6.3/qwen3.8-max-0902-iterative-1/) |
| MiniMax M3.1 Flash | 单轮 | 白化误差 \(E = 0.2255\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 50.8 | 47 分钟 | [`minimax-m3.1-flash-single-1/`](solutions/v6.3/minimax-m3.1-flash-single-1/) |
| MiniMax M3.1 Flash | 迭代 | 白化误差 \(E = 0.2294\)（\(U = 0.3493\)，\(W = 0.2290\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 49.8 | 6 次提交，117 分钟 | [`minimax-m3.1-flash-iterative-1/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.2-sol-single-1/source/`](solutions/v6.3/gpt-6.2-sol-single-1/source/)（提交 `trajectory-covariance-v2-20261120T073601Z-badec1`）。

### GPT-6.2 Sol（迭代，v6.3）

public 分数随测试提交：47.1（1 分钟） → 58.8（6 分钟） → 60.9（8 分钟） → 62.3（10 分钟） → 62.3（13 分钟） → 62.5（14 分钟） → 62.5（16 分钟） → 62.8（19 分钟） → 63.6（21 分钟） → 63.6（24 分钟） → 63.6（26 分钟）

源码 [`solutions/v6.3/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/gpt-6.2-sol-iterative-1/source/)（提交 `trajectory-covariance-v2-20261120T085218Z-2bc425`）。

### GPT-6 Astra（单轮，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

源码 [`solutions/v6.3/gpt-6-astra-single-1/source/`](solutions/v6.3/gpt-6-astra-single-1/source/)（提交 `trajectory-covariance-v2-20261121T165555Z-877e32`）。

### GPT-6 Astra（迭代，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

public 分数随测试提交：51.0（4 分钟） → 52.4（15 分钟） → 52.6（19 分钟） → 58.9（26 分钟） → 62.0（35 分钟） → 62.2（40 分钟） → 61.3（50 分钟） → 62.4（63 分钟） → 62.4（74 分钟） → 62.4（93 分钟） → 63.7（110 分钟） → 64.1（136 分钟） → 64.1（144 分钟） → 64.0（153 分钟） → 64.0（159 分钟） → 64.0（162 分钟） → 64.0（182 分钟）

源码 [`solutions/v6.3/gpt-6-astra-iterative-1/source/`](solutions/v6.3/gpt-6-astra-iterative-1/source/)（提交 `trajectory-covariance-v2-20261121T210425Z-0074d0`）。

### GPT-6.1 Sol（单轮，v6.3）

白化后自适应窗宽的重叠批均值；README 写明把估计除以了轨迹长度，与题目要求的尺度不符，误差约为强参照的 10 倍。会话只用了 2 分钟。

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `trajectory-covariance-v2-20261110T164002Z-c19fe6`）。

### GPT-6.1 Sol（迭代，v6.3）

白化后：有限状态链拟合正则化转移矩阵并解 Markov 泊松方程；其余拟合至多 32 阶的条件向量自回归，按信息准则加权平均零频谱估计；可逆性诊断通过时与共同动力学、分组动力学拟合比较。重跑（原尝试受测试排队故障影响）。

public 分数随测试提交：51.0（2 分钟） → 50.8（7 分钟） → 52.3（9 分钟） → 52.9（15 分钟） → 52.9（20 分钟） → 52.9（22 分钟） → 52.9（24 分钟） → 52.3（26 分钟） → 52.9（29 分钟） → 52.9（32 分钟） → 52.9（37 分钟） → 52.9（40 分钟） → 52.2（43 分钟） → 52.6（46 分钟） → 52.6（50 分钟） → 53.0（54 分钟） → 57.9（58 分钟） → 60.7（65 分钟） → 60.8（69 分钟） → 61.9（73 分钟） → 61.7（79 分钟） → 62.2（81 分钟） → 62.1（84 分钟） → 62.1（90 分钟） → 62.1（93 分钟） → 62.2（98 分钟） → 62.2（105 分钟） → 62.2（115 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `trajectory-covariance-v2-20261111T124958Z-f04810`）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `trajectory-covariance-v2-20261112T075745Z-f9b326`）。

### Claude Opus 5.5（迭代，v6.3）

public 分数随测试提交：50.8（4 分钟） → 52.2（18 分钟） → 59.1（25 分钟） → 59.0（41 分钟） → 60.0（48 分钟） → 60.6（64 分钟） → 62.1（67 分钟） → 62.0（71 分钟） → 62.8（73 分钟） → 62.8（77 分钟） → 63.0（80 分钟） → 63.0（82 分钟） → 62.5（94 分钟） → 63.0（98 分钟） → 63.1（104 分钟） → 59.8（110 分钟） → 62.8（115 分钟） → 63.1（118 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `trajectory-covariance-v2-20261112T095021Z-e3c24c`）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `trajectory-covariance-v2-20261113T104301Z-c020cd`）。

### Hunyuan 4（迭代，v6.3）

public 分数随测试提交：50.5（40 分钟） → 39.6（72 分钟） → 51.4（95 分钟） → 52.1（99 分钟） → 52.1（105 分钟） → 52.2（115 分钟） → 52.4（120 分钟） → 52.4（123 分钟） → 52.4（129 分钟） → 52.4（131 分钟） → 54.3（136 分钟） → 60.1（139 分钟） → 58.1（142 分钟） → 59.6（149 分钟） → 60.3（151 分钟） → 60.6（154 分钟） → 59.0（157 分钟） → 59.8（160 分钟） → 60.5（165 分钟） → 60.5（168 分钟） → 60.6（172 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `trajectory-covariance-v2-20261113T144608Z-18df17`）。

### Qwen4 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen4-max-single-1/source/`](solutions/v6.3/qwen4-max-single-1/source/)（提交 `trajectory-covariance-v2-20261125T104553Z-5a286b`）。

### Qwen4 Max（迭代，v6.3）

public 分数随测试提交：50.1（5 分钟） → 41.0（15 分钟） → 46.9（17 分钟） → 50.1（22 分钟） → 51.4（27 分钟） → 51.4（32 分钟） → 51.4（33 分钟） → 51.0（35 分钟） → 51.4（36 分钟） → 51.4（38 分钟） → 50.4（39 分钟） → 51.5（40 分钟） → 51.4（42 分钟） → 51.3（43 分钟） → 51.5（44 分钟） → 51.5（45 分钟）

源码 [`solutions/v6.3/qwen4-max-iterative-1/source/`](solutions/v6.3/qwen4-max-iterative-1/source/)（提交 `trajectory-covariance-v2-20261125T165212Z-9849d6`）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/qwen3.8-max-0902-single-1/source/)（提交 `trajectory-covariance-v2-20261113T111630Z-5a37c7`）。

### Qwen3.8 Max（迭代，v6.3）

public 分数随测试提交：0.0（16 分钟） → 50.8（111 分钟） → 50.7（119 分钟） → 50.9（144 分钟） → 50.8（151 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-1/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-1/source/)（提交 `trajectory-covariance-v2-20261113T093846Z-0ba0ec`）。

### MiniMax M3.1 Flash（单轮，v6.3）

源码 [`solutions/v6.3/minimax-m3.1-flash-single-1/source/`](solutions/v6.3/minimax-m3.1-flash-single-1/source/)（提交 `trajectory-covariance-v2-20261125T111901Z-d540df`）。

### MiniMax M3.1 Flash（迭代，v6.3）

public 分数随测试提交：0.0（30 分钟） → 0.0（58 分钟） → 33.1（72 分钟） → 33.1（86 分钟） → 50.0（99 分钟） → 50.0（110 分钟）

源码 [`solutions/v6.3/minimax-m3.1-flash-iterative-1/source/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/source/)（提交 `trajectory-covariance-v2-20261125T191131Z-fe391d`）。

<details><summary>旧版本的成绩与解法</summary>

## 模型成绩与解法（题目版本 v1 · 协议 v6.2）

协议 v6.2 单轮（2026-11-09 开发，11-10 评分，同一 3,000 例子集）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | \(L = 0.221264\)，3000/3000 合格（单例最多 1.66 秒） | \(100 - 50\,L/\text{强}\) | 51.9 | 68 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |

### Claude Opus 5.5（单轮，v6.2）

比强参照（拟合 VAR(1)，0.230050）略好，比 sol 的 v6.1 提交（0.176933，61.5 分）差。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `trajectory-covariance-20261109T130605Z-481cfb`）。

## 模型成绩与解法（题目版本 v1 · 协议 v6.1）

协议 v6.1（2026-11-06，3,000 例固定子集）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol |  | L = 0.176933，3000/3000 合格（单例最多 1.06 秒） | \(100-50L/b\) | 61.5 | 55 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.1/gpt-6.1-sol-single-1/) |

### GPT-6.1 Sol（，v6.1）

按母链配对，VAR(1) 与 sol 的误差差为 0.0531，95% 区间 [0.0441, 0.0622]，显著优于强参照。但只在 30 档中的 15 档更好：优势主要来自随机游走 Metropolis（0.215 对 0.504）和有限状态 MH 链（0.147 对 0.196），在 NUTS、Pólya–Gamma 和全正线性链上略差。出题方封存的 agent 在同一子集上为 0.217481（52.7 分）。

源码 [`solutions/v6.1/gpt-6.1-sol-single-1/source/`](solutions/v6.1/gpt-6.1-sol-single-1/source/)（提交 `trajectory-covariance-20261106T045544Z-58f573`）。

</details>

## 观察与说明

- v6.3 私有集为 2,400 个输入（分片到多个评测核），反馈集 150 个；每个输入限时 5 秒。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
