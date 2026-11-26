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
| public（不给，反馈用） | 15 个不发给模型的反馈例（60 个 primary 例按真值排序每 4 个取 1，正式计分只用其余 45 例；2026-11-26 前为与样例相同的 8 例） |
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
| GPT-6.2 Sol | 单轮 | 损失 \(E = 0.000961162\)，45 例，单例最长 1.0 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 70.0 | 12 分钟 | [`gpt-6.2-sol-single-1/`](solutions/v6.3/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | 损失 \(E = 0.000826802\)，45 例，单例最长 1.1 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 74.2 | 5 次提交，20 分钟 | [`gpt-6.2-sol-iterative-1/`](solutions/v6.3/gpt-6.2-sol-iterative-1/) |
| GPT-6 Astra | 单轮 | 损失 \(E = 0.000841913\)，45 例，单例最长 2.3 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 73.7 | 44 分钟 | [`gpt-6-astra-single-1/`](solutions/v6.3/gpt-6-astra-single-1/) |
| GPT-6 Astra | 迭代 | 损失 \(E = 0.000919333\)，45 例，单例最长 4.6 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 71.3 | 4 次提交，56 分钟 | [`gpt-6-astra-iterative-1/`](solutions/v6.3/gpt-6-astra-iterative-1/) |
| GPT-6.1 Sol | 单轮 | 损失 \(E = 0.000876169\)，45 例，单例最长 9.1 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 72.6 | 49 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 损失 \(E = 0.00103382\)，45 例，单例最长 2.3 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 67.7 | 3 次提交，38 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 损失 \(E = 0.00600676\)，45 例，单例最长 1.3 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 36.3 | 24 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | 损失 \(E = 0.0131157\)，45 例，单例最长 0.5 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 14.1 | 5 次提交，80 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Hunyuan 4 | 单轮 | 损失 \(E = 0.00085626\)，45 例，单例最长 1.5 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 73.3 | 51 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | 损失 \(E = 0.00290018\)，45 例，单例最长 0.3 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 45.9 | 7 次提交，135 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Qwen4 Max | 单轮 | 损失 \(E = 0.000985423\)，45 例，单例最长 0.8 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 69.2 | 21 分钟 | [`qwen4-max-single-1/`](solutions/v6.3/qwen4-max-single-1/) |
| Qwen4 Max | 迭代 | 损失 \(E = 0.001318\)，45 例，单例最长 0.8 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 58.8 | 5 次提交，40 分钟 | [`qwen4-max-iterative-1/`](solutions/v6.3/qwen4-max-iterative-1/) |
| Qwen3.8 Max | 单轮 | 无效提交（45 of 45 cases failed; first case_002: README.md and run.sh are required） | 任一用例无效、缺失、超时或失败记 0 | 0.0 | 60 分钟 | [`qwen3.8-max-0902-single-1/`](solutions/v6.3/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | 损失 \(E = 0.00198245\)，45 例，单例最长 0.5 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 48.8 | 4 次提交，216 分钟 | [`qwen3.8-max-0902-iterative-2/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/) |
| MiniMax M3.1 Flash | 单轮 | 损失 \(E = 0.00329319\)，45 例，单例最长 1.3 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 44.7 | 60 分钟 | [`minimax-m3.1-flash-single-1/`](solutions/v6.3/minimax-m3.1-flash-single-1/) |
| MiniMax M3.1 Flash | 迭代 | 损失 \(E = 0.0105931\)，45 例，单例最长 1.1 秒（一阶 Richardson：\(U = 0.0176266\)，GRE：\(W = 0.00160119\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 21.9 | 5 次提交，101 分钟 | [`minimax-m3.1-flash-iterative-1/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.2-sol-single-1/source/`](solutions/v6.3/gpt-6.2-sol-single-1/source/)（提交 `continuum-estimation-20261120T074711Z-5ceaf6`）。

### GPT-6.2 Sol（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：35.8（3 分钟） → 80.0（4 分钟） → 82.0（8 分钟） → 82.0（11 分钟） → 82.4（15 分钟）

源码 [`solutions/v6.3/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/gpt-6.2-sol-iterative-1/source/)（提交 `continuum-estimation-20261120T090954Z-ca982a`）。

### GPT-6 Astra（单轮，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

源码 [`solutions/v6.3/gpt-6-astra-single-1/source/`](solutions/v6.3/gpt-6-astra-single-1/source/)（提交 `continuum-estimation-20261121T174536Z-f8cfee`）。

### GPT-6 Astra（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：92.6（8 分钟） → 95.9（18 分钟） → 96.2（37 分钟） → 96.2（50 分钟）

源码 [`solutions/v6.3/gpt-6-astra-iterative-1/source/`](solutions/v6.3/gpt-6-astra-iterative-1/source/)（提交 `continuum-estimation-20261121T215822Z-89843b`）。

### GPT-6.1 Sol（单轮，v6.3）

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `continuum-estimation-20261117T100054Z-6bbc6f`）。

### GPT-6.1 Sol（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：97.3（9 分钟） → 97.4（16 分钟） → 97.4（24 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `continuum-estimation-20261117T154315Z-cf43e8`）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `continuum-estimation-20261117T055656Z-ee456a`）。

### Claude Opus 5.5（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：89.4（8 分钟） → 89.9（30 分钟） → 90.0（36 分钟） → 90.0（47 分钟） → 90.5（71 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `continuum-estimation-20261117T090022Z-90b6d7`）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `continuum-estimation-20261117T075924Z-1e478f`）。

### Hunyuan 4（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：28.0（29 分钟） → 0.0（34 分钟） → 49.8（43 分钟） → 79.5（54 分钟） → 95.8（69 分钟） → 95.8（93 分钟） → 95.8（127 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `continuum-estimation-20261117T111508Z-f546c9`）。

### Qwen4 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen4-max-single-1/source/`](solutions/v6.3/qwen4-max-single-1/source/)（提交 `continuum-estimation-20261125T114010Z-b10c67`）。

### Qwen4 Max（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：38.7（10 分钟） → 89.5（18 分钟） → 88.9（33 分钟） → 93.0（35 分钟） → 93.7（39 分钟）

源码 [`solutions/v6.3/qwen4-max-iterative-1/source/`](solutions/v6.3/qwen4-max-iterative-1/source/)（提交 `continuum-estimation-20261125T180305Z-bdc8e2`）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/qwen3.8-max-0902-single-1/source/)（提交 `continuum-estimation-20261117T074044Z-58fc7f`）。

### Qwen3.8 Max（迭代，v6.3）

反馈集改为 15 个留出例后的第 3 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：30.7（30 分钟） → 0.0（56 分钟） → 85.9（136 分钟） → 86.1（212 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-2/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-2/source/)（提交 `continuum-estimation-20261118T064024Z-317d25`）。

### MiniMax M3.1 Flash（单轮，v6.3）

源码 [`solutions/v6.3/minimax-m3.1-flash-single-1/source/`](solutions/v6.3/minimax-m3.1-flash-single-1/source/)（提交 `continuum-estimation-20261125T124448Z-5009de`）。

### MiniMax M3.1 Flash（迭代，v6.3）

反馈集改为 15 个留出例后的第 2 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。

public 分数随测试提交：34.5（13 分钟） → 41.8（28 分钟） → 41.9（60 分钟） → 91.6（73 分钟） → 91.6（94 分钟）

源码 [`solutions/v6.3/minimax-m3.1-flash-iterative-1/source/`](solutions/v6.3/minimax-m3.1-flash-iterative-1/source/)（提交 `continuum-estimation-20261125T221313Z-78dc64`）。

## 观察与说明

- 原包要求原生 ARM64；本机用同版本（Python 3.11.15 / NumPy 1.26.4 / SciPy 1.12.0）在 x86-64 上重建运行时，评分器只改架构检查，并为 v6.3 反馈加一个由 8 个公开例子组成的 public 分区。
- 出题人说明 1.0.1 未重跑参照；按本仓库约定在本机重跑，强参照与记录相差 0.5%。
- 题面中“开发限 300 秒”一句已去掉，开发时限由协议规定。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
