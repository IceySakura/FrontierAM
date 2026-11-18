# continuum-estimation

给出 4–5 个正离散步长上的数值结果，估计步长趋于 0 时的极限值；每例 60 秒，按相对误差的批量平均计分。

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
| `v1` | `continuum-estimation` | 出题人 1.0.1，移植到 x86-64，本机重测锚点（当前版本） |

## 题目背景

数值计算常在若干离散步长上给出近似值，需要外推到连续极限（Richardson 外推）。收敛阶未知、网格不均匀时，经典的一阶外推误差较大；概率外推（如基于高斯过程的 GRE）对收敛阶和误差结构建模，能更稳健地估计极限。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v1/statement.md |
| 公开开发材料 | 公开包 task/environment/public（8 个例子含真值，给模型，也是 public 反馈集） |
| 私有测试集 | 维护包 task/tests/private（primary 60 例、audit 15 例，只在内部机器） |
| 评测器 | 维护包 task/tests/grade.py（移植改动见 versions/v1/task.json）；arena/operator_drivers/continuum.py |
| 参照实现 | 维护包 operator/references/zero（0 分）、five（50 分） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 8 个公开例子（public/dev，含真值） |
| public（不给，反馈用） | 与样例相同的 8 个例子 |
| 反馈分类 | 不分类，只报总计 |

## 评测依据

- **输入**：x、y 两个等长数组（4 或 5 个值，x 为不同的正数，未排序）。
- **正式集**：60 例；另有 15 例不利情形只作诊断，不计分。
- **资源**：每例全新受限进程（Landlock + seccomp），1 CPU、2 GiB 内存 + 2 GiB swap、60 秒墙钟（含启动）。
- **有效性**：任一例输出无效、缺失、超时或失败，整份提交记 0 分。
- **原始指标**：每例 \(|z-t|/\max(1,|t|)\)，60 例等权平均得 \(E\)。
- **换算**：一阶 Richardson 的误差为 \(U\)、GRE 为 \(W\)：\(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\)。

## Baseline

| 参照 | 方法 | 整批损失 | 记分 |
| --- | --- | --- | --- |
| 弱参照 | 一阶 Richardson（最小两个步长） | 0.017163 | 0 |
| 强参照 | GRE（收敛阶、核与长度尺度的固定网格） | 0.0016482 | 50 |

本机评测核实测（60 例）；出题人记录 0.017163 / 0.0016401。

## 模型成绩与解法（题目版本 v1 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | 损失 \(E = 0.000942335\)，60 例，单例最长 9.0 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 71.4 | 49 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 损失 \(E = 0.00102139\)，60 例，单例最长 3.0 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 69.0 | 3 次提交，38 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 损失 \(E = 0.00474595\)，60 例，单例最长 1.2 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 40.0 | 24 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | 损失 \(E = 0.0101005\)，60 例，单例最长 0.4 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 22.8 | 5 次提交，80 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Qwen3.8 Max | 单轮 | 无效提交（60 of 60 cases failed; first case_000: README.md and run.sh are required） | 任一用例无效、缺失、超时或失败记 0 | 0.0 | 60 分钟 | [`qwen3.8-max-0902-single-1/`](solutions/v6.3/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | 损失 \(E = 0.00168707\)，60 例，单例最长 0.5 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 49.9 | 4 次提交，216 分钟 | [`qwen3.8-max-0902-iterative-2/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/) |
| Hunyuan 4 | 单轮 | 损失 \(E = 0.000840049\)，60 例，单例最长 1.6 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 74.5 | 51 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | 损失 \(E = 0.00232668\)，60 例，单例最长 0.3 秒（一阶 Richardson：\(U = 0.017163\)，GRE：\(W = 0.00164816\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 47.8 | 7 次提交，135 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Intern S2 | 单轮 | 无效提交（开发结束时没有冻结出有效提交） | 无效提交记 0 | 0.0 | 60 分钟 | [`intern-s2-single-1/`](solutions/v6.3/intern-s2-single-1/) |

### GPT-6.1 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `continuum-estimation-20261117T100054Z-6bbc6f`）。

### GPT-6.1 Sol（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：97.3（9 分钟） → 97.4（16 分钟） → 97.4（24 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `continuum-estimation-20261117T154315Z-cf43e8`）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `continuum-estimation-20261117T055656Z-ee456a`）。

### Claude Opus 5.5（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：89.4（8 分钟） → 89.9（30 分钟） → 90.0（36 分钟） → 90.0（47 分钟） → 90.5（71 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `continuum-estimation-20261117T090022Z-90b6d7`）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/qwen3.8-max-0902-single-1/source/)（提交 `continuum-estimation-20261117T074044Z-58fc7f`）。

### Qwen3.8 Max（迭代，v6.3）

两次尝试都有回合遇到上游（阿里百炼）流约 600 秒没有任何输出后断开（AIHub 记为成功、0 token，无其他供应商可换），第 1 次损失约 70 分钟、第 2 次约 30 分钟；按用户决定（11-18）取损失较少的第 2 次为正式格。

public 分数随测试提交：30.7（30 分钟） → 0.0（56 分钟） → 85.9（136 分钟） → 86.1（212 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-2/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/source/)（提交 `continuum-estimation-20261118T064024Z-317d25`）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `continuum-estimation-20261117T075924Z-1e478f`）。

### Hunyuan 4（迭代，v6.3）

出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。

public 分数随测试提交：28.0（29 分钟） → 0.0（34 分钟） → 49.8（43 分钟） → 79.5（54 分钟） → 95.8（69 分钟） → 95.8（93 分钟） → 95.8（127 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `continuum-estimation-20261117T111508Z-f546c9`）。

### Intern S2（单轮，v6.3）

11-18 补跑：同一把 key 两个并发（裸测单请求 32.7 token/秒、两并发每路约 31），会话内生成约 17–23 token/秒，无上游错误、断流或重试（results/OPEN_ITEMS.md）。

源码 [`solutions/v6.3/intern-s2-single-1/source/`](solutions/v6.3/intern-s2-single-1/source/)（提交 `continuum-estimation-20261118T101203Z-8c0524`）；未复制的二进制或大文件：（提交存档不在本服务器）。

## 观察与说明

- 原包要求原生 ARM64；本机用同版本（Python 3.11.15 / NumPy 1.26.4 / SciPy 1.12.0）在 x86-64 上重建运行时，评分器只改架构检查，并为 v6.3 反馈加一个由 8 个公开例子组成的 public 分区。
- 出题人说明 1.0.1 未重跑参照；按本仓库约定在本机重跑，强参照与记录相差 0.5%。
- 题面中“开发限 300 秒”一句已去掉，开发时限由协议规定。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
