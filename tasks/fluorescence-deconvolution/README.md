# fluorescence-deconvolution

由 32³ 的荧光观测体和给定的点扩散函数（PSF）重建非负强度体；每例 1.25 秒（含启动），96 个正式用例，nMSE 越小越好。

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
| `v1` | `fluorescence-deconvolution` | 出题人 1.1.0，重建 amd64 镜像，时限 1.25 秒，本机重测锚点（当前版本） |

## 题目背景

荧光显微镜的成像可近似为真实强度与点扩散函数的卷积再加噪声；反卷积恢复更清晰的结构。经典方法是 Richardson–Lucy 迭代（泊松噪声下的乘性更新），加速与正则化变体（如 Deconwolf 的加速迭代）在相同时间预算内效果更好。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v1/statement.md |
| 公开开发材料 | 公开包 public-dev（12 例含真值，给模型，也是 public 反馈集） |
| 私有测试集 | 维护包 private-data（96 例，只在内部机器） |
| 评测器 | 维护包 runtime/grade.py、runner.py、worker.py（移植改动见 versions/v1/task.json）；arena/operator_drivers/fluorescence.py |
| 参照实现 | 维护包 runtime/references/rl20（0 分）、shb10（50 分） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 12 个公开例子（public-dev，含真值） |
| public（不给，反馈用） | 与样例相同的 12 个例子 |
| 反馈分类 | 点状、管状、混合、边界各向异性 |

## 评测依据

- **输入**：`observed`（32³）与 `psf`（9³，归一化）float32；输出同形状、有限、非负的 `.npy`。
- **正式集**：96 例（4 类结构 × 2 类 PSF × 2 类噪声，每层 6 例）。
- **资源**：每例全新进程，单核、8 GiB、无 swap、断网，墙钟 1.25 秒（含启动和导入）；可选构建 300 秒。
- **有效性**：任一例失败或超时，整份提交记 0 分。
- **原始指标**：全体素 nMSE，96 例等权平均得 \(E\)。
- **换算**：RL20 的误差为 \(U\)、SHB10 为 \(W\)：\(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\)。

## Baseline

| 参照 | 方法 | 平均 nMSE | 记分 | 每例最大墙钟 |
| --- | --- | --- | --- | --- |
| 弱参照 | RL20（20 次 Richardson–Lucy 更新） | 0.219945 | 0 | 0.66 秒 |
| 强参照 | SHB10（灵敏度修正加加速，10 次更新） | 0.161141 | 50 | 0.67 秒 |

本机评测核实测（96 例，时限 1.25 秒）；出题人记录 0.219945 / 0.161172。

## 模型成绩与解法（题目版本 v1 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | nMSE \(E = 0.07422\)，96 例，单例最长 0.23 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 77.0 | 25 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `fluorescence-deconvolution-20261117T070120Z-a62bc0`）。

## 观察与说明

- 原包只提供 ARM64 数值镜像（Dockerfile 只有 `FROM <摘要>`）；本机按 ARM64 镜像内的包清单在 x86-64 上重建同版本环境。
- 出题人在 Apple M 上标定 0.500 秒（参照最长 0.23 / 0.20 秒）；本机评测核（AMD EPYC 7K62）参照最长 0.58 / 0.57 秒，时限按 RL20 的速度比放大到 1.25 秒，参照余量与出题人相同（约 2.2 倍）。
- 出题人附带的三份 agent 提交在原环境为 74.6–93.8 分（换算到 100 分），高分段可能区分度不足。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
