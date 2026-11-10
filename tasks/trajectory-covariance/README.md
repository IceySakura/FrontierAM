# 从一条 MCMC 轨迹估计长期协方差

只给一条马尔可夫链轨迹，估计样本均值的长期协方差矩阵（MCMC 标准误的核心），30 种设置平均的白化误差越小越好。

看板页面：[`dashboard/site/tasks/trajectory-covariance.html`](../../dashboard/site/tasks/trajectory-covariance.html)（克隆后用浏览器直接打开）

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/` | 题目的各个版本；当前版本 `v1/`，题面 `versions/v1/statement.md` |
| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |
| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |
| `analysis/` | 这道题的专题分析与复核记录 |
| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |

## 版本

| 版本 | 评测系统中的名字 | 说明 |
| --- | --- | --- |
| `v1` | `trajectory-covariance` | 出题方包 mcmc-long-run-covariance-maintainer-v1；本机评测固定的 3,000 例分层子集（当前版本） |

## 题目背景

MCMC 的输出是相关的样本，不能用独立样本的公式算均值的标准误；需要估计“长期协方差” \(\Sigma = \lim n\,\mathrm{Cov}(\bar X_n)\)，它决定了有效样本量和收敛诊断。经典方法有批均值（BM/OBM）、谱方差估计和基于 VAR 模型的拟合；难点在于不同采样器（随机游走 Metropolis、NUTS、Pólya–Gamma Gibbs、有限状态链）的自相关结构差别很大，而且轨迹长度从几千到几万步不等。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v1/statement.md |
| 公开开发材料 | versions/v1/public/ |
| 私有测试集 | 3 万个私有输入与参考矩阵（出题方包，/srv/frontieram-raw/） |
| 评测器 | 出题方包的验证器（arena/operator_drivers/trajectory_batch.py 启动其验证容器） |
| 参照实现 | references/r1/（OBM，R 包 mcmcse；VAR(1)）；人类候选参照 references/candidates/ |

公开包中的数据文件未复制到仓库，清单见 `versions/v1/public/DATA_FILES.md`。

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

## 模型成绩与解法（协议 v6.2）

| 模型 | 设置 | 原始指标 | 换算 | 得分（满分 100，参照 50） | 复核 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | \(L = 0.221264\)，3000/3000 合格（单例最多 1.66 秒） | \(100 - 50\,L/\text{强}\) | 51.9 | 待复核 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |

超过参照（50 分）的成绩在复核结论出来之前标“待复核”，见 `analysis/` 与 `results/BASELINE_POLICY.md`。

### Claude Opus 5.5（单轮）

比强参照（拟合 VAR(1)，0.230050）略好，比 sol 的 v6.1 提交（0.176933，61.5 分）差。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `trajectory-covariance-20261109T130605Z-481cfb`）。

## 观察与说明

- 全量 3 万例的评测在单核上约需 5 小时，需要分片到多个评测核；是否以全量作为正式规模仍待确定。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
