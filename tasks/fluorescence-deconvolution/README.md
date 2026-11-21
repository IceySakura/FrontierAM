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
| GPT-6.2 Sol | 单轮 | nMSE \(E = 0.01264\)，96 例，单例最长 0.72 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 96.1 | 20 分钟 | [`gpt-6.2-sol-single-1/`](solutions/v6.3/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | nMSE \(E = 0.01558\)，96 例，单例最长 0.85 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 95.2 | 5 次提交，24 分钟 | [`gpt-6.2-sol-iterative-1/`](solutions/v6.3/gpt-6.2-sol-iterative-1/) |
| GPT-6.1 Sol | 单轮 | nMSE \(E = 0.01821\)，96 例，单例最长 1.10 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 94.3 | 59 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | nMSE \(E = 0.02092\)，96 例，单例最长 0.74 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 93.5 | 3 次提交，182 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | nMSE \(E = 0.07422\)，96 例，单例最长 0.23 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 77.0 | 25 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | nMSE \(E = 0.02284\)，96 例，单例最长 0.91 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 92.9 | 4 次提交，199 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Qwen3.8 Max | 单轮 | nMSE \(E = 0.10694\)，96 例，单例最长 0.20 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 66.8 | 49 分钟 | [`qwen3.8-max-0902-single-1/`](solutions/v6.3/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | nMSE \(E = 0.07322\)，96 例，单例最长 0.19 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 77.3 | 3 次提交，220 分钟 | [`qwen3.8-max-0902-iterative-2/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/) |
| Hunyuan 4 | 单轮 | nMSE \(E = 0.06654\)，96 例，单例最长 0.41 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 79.4 | 53 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | nMSE \(E = 0.04844\)，96 例，单例最长 0.48 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 85.0 | 8 次提交，178 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Intern S2 | 单轮 | 无效提交（开发结束时没有冻结出有效提交） | 无效提交记 0 | 0.0 | 60 分钟 | [`intern-s2-single-1/`](solutions/v6.3/intern-s2-single-1/) |
| Intern S2 | 迭代 | nMSE \(E = 0.18191\)，96 例，单例最长 0.29 秒（RL20：\(U = 0.21995\)，SHB10：\(W = 0.16114\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 32.3 | 1 次提交，164 分钟 | [`intern-s2-iterative-1/`](solutions/v6.3/intern-s2-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.2-sol-single-1/source/`](solutions/v6.3/gpt-6.2-sol-single-1/source/)（提交 `fluorescence-deconvolution-20261120T075629Z-6cd045`）。

### GPT-6.2 Sol（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：40.4（2 分钟） → 95.0（9 分钟） → 96.5（18 分钟） → 96.6（23 分钟） → 96.6（24 分钟）

源码 [`solutions/v6.3/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/gpt-6.2-sol-iterative-1/source/)（提交 `fluorescence-deconvolution-20261120T091644Z-d81f50`）。

### GPT-6.1 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `fluorescence-deconvolution-20261117T121636Z-2df5ef`）；未复制的二进制或大文件：core.so（103,344 字节）。

### GPT-6.1 Sol（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：66.1（9 分钟） → 94.5（37 分钟） → 95.5（171 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `fluorescence-deconvolution-20261117T184503Z-5b3f6f`）；未复制的二进制或大文件：solver.so（176,144 字节）、solver_fast.so（184,336 字节）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `fluorescence-deconvolution-20261117T070120Z-a62bc0`）。

### Claude Opus 5.5（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：78.8（21 分钟） → 89.0（46 分钟） → 89.1（52 分钟） → 93.5（197 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `fluorescence-deconvolution-20261117T132401Z-a0ecca`）；未复制的二进制或大文件：model.npz（1,735,675 字节）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/qwen3.8-max-0902-single-1/source/)（提交 `fluorescence-deconvolution-20261117T074310Z-68427e`）。

### Qwen3.8 Max（迭代，v6.3）

两次尝试都有回合遇到上游（阿里百炼）流约 600 秒没有任何输出后断开（AIHub 记为成功、0 token，无其他供应商可换），第 1 次损失约 50 分钟、第 2 次约 40 分钟；按用户决定（11-18）取损失较少的第 2 次为正式格。

public 分数随测试提交：75.1（100 分钟） → 78.3（119 分钟） → 78.8（218 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-2/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/source/)（提交 `fluorescence-deconvolution-20261118T064428Z-b197e6`）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `fluorescence-deconvolution-20261117T080810Z-ec50e3`）；未复制的二进制或大文件：model.npz（1,512 字节）。

### Hunyuan 4（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：69.1（18 分钟） → 85.3（44 分钟） → 86.3（51 分钟） → 86.9（115 分钟） → 87.0（146 分钟） → 87.0（149 分钟） → 87.1（170 分钟） → 87.0（174 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `fluorescence-deconvolution-20261117T160645Z-009254`）；未复制的二进制或大文件：ensemble_weights.npz（960,466 字节）。

### Intern S2（单轮，v6.3）

11-18 补跑：同一把 key 两个并发（裸测单请求 32.7 token/秒、两并发每路约 31），会话内生成约 17–23 token/秒，无上游错误、断流或重试（results/OPEN_ITEMS.md）。

源码 [`solutions/v6.3/intern-s2-single-1/source/`](solutions/v6.3/intern-s2-single-1/source/)（提交 `fluorescence-deconvolution-20261118T101203Z-243425`）；未复制的二进制或大文件：（提交存档不在本服务器）。

### Intern S2（迭代，v6.3）

11-18 补跑：同一把 key 两个并发（裸测单请求 32.7 token/秒、两并发每路约 31），会话内生成约 17–23 token/秒，无上游错误、断流或重试（results/OPEN_ITEMS.md）。 出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：4.5（116 分钟）

源码 [`solutions/v6.3/intern-s2-iterative-1/source/`](solutions/v6.3/intern-s2-iterative-1/source/)（提交 `fluorescence-deconvolution-20261119T030540Z-88870e`）。

## 观察与说明

- 原包只提供 ARM64 数值镜像（Dockerfile 只有 `FROM <摘要>`）；本机按 ARM64 镜像内的包清单在 x86-64 上重建同版本环境。
- 出题人在 Apple M 上标定 0.500 秒（参照最长 0.23 / 0.20 秒）；本机评测核（AMD EPYC 7K62）参照最长 0.58 / 0.57 秒，时限按 RL20 的速度比放大到 1.25 秒，参照余量与出题人相同（约 2.2 倍）。
- 出题人附带的三份 agent 提交在原环境为 74.6–93.8 分（换算到 100 分），高分段可能区分度不足。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
