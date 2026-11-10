# 从一条 MCMC 轨迹估计长期协方差

只给一条马尔可夫链轨迹，估计样本均值的长期协方差矩阵（MCMC 标准误的核心），30 种设置平均的白化误差越小越好。

看板页面：http://9.135.91.91:18765/tasks/trajectory-covariance.html（办公网）

## 怎么读这道题

1. **题目背景**：下面第一节。
2. **题面原文**：[`statement.md`](statement.md)，与给模型的版本逐字相同。
3. **数据**：见“数据与代码在哪里”；公开开发材料的文本文件在 [`public/`](public/)。
4. **评测与计分**：“评测依据”一节；评测器位置见下表。
5. **Baseline**：“Baseline”一节；参照源码在 [`reference/`](reference/)。
6. **模型解法**：“模型成绩与解法”一节；每份提交的完整源码在 [`solutions/`](solutions/)。

## 题目背景

MCMC 的输出是相关的样本，不能用独立样本的公式算均值的标准误；需要估计“长期协方差” \(\Sigma = \lim n\,\mathrm{Cov}(\bar X_n)\)，它决定了有效样本量和收敛诊断。经典方法有批均值（BM/OBM）、谱方差估计和基于 VAR 模型的拟合；难点在于不同采样器（随机游走 Metropolis、NUTS、Pólya–Gamma Gibbs、有限状态链）的自相关结构差别很大，而且轨迹长度从几千到几万步不等。

## 数据与代码在哪里

| 内容 | 位置 |
| --- | --- |
| 题包 | 出题方包 mcmc-long-run-covariance-maintainer-v1（/srv/frontieram-raw/，git 忽略）；封存副本 .runtime-ops/trajectory-covariance/ |
| 公开开发材料 | public/（本目录） |
| 私有测试集 | 3 万个私有输入（本机评测固定的 3,000 个分层子集），在出题方包 results/mcmc-long-run-covariance/formal-v1/private/ |
| 评测器 | 出题方包的验证器（arena/operator_drivers/trajectory_batch.py 直接启动其验证容器） |
| 参照实现 | reference/（OBM：R 包 mcmcse；VAR(1)）；人类候选参照见 tasks/operator/trajectory-covariance/reference_candidates/ |
| 详细文档 | `results/NEW_TASKS.md`、`results/BASELINE_REVIEW.md` |

公开包中的数据文件（未复制到本目录）：public-dev/01/input.npy（160,128 字节）；public-dev/01/reference.npy（256 字节）；public-dev/02/input.npy（120,128 字节）；public-dev/02/reference.npy（416 字节）；public-dev/trajectory-dev.zip（273,744 字节）。

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

## 模型成绩与解法（协议 v6.2，最新）

| 模型 | 设置 | 原始指标 | 换算 | 得分（满分 100，参照 50） | 复核 | 源码 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | \(L = 0.221264\)，3000/3000 合格（单例最多 1.66 秒） | \(100 - 50\,L/\text{强}\) | 51.9 | 待复核 | [`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/) |

超过参照（50 分）的成绩在复核结论出来之前标“待复核”：复核会解释超出的原因、寻找更强的人类方法并做消融，见 `results/BASELINE_REVIEW.md` 与 `results/BASELINE_POLICY.md`。

### Claude Opus 5.5（单轮）

比强参照（拟合 VAR(1)，0.230050）略好，比 sol 的 v6.1 提交（0.176933，61.5 分）差。

源码：[`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/)（提交 `trajectory-covariance-20261109T130605Z-481cfb`）。

## 观察与说明

- 全量 3 万例的评测在单核上约需 5 小时，需要分片到多个评测核；是否以全量作为正式规模仍待确定。

## 历史版本

v6.1 sol 成绩见 `results/NEW_TASKS.md`。
