# blackbox-cubature

在 [0,1]^d（d = 2、3、5、8）上估计未知函数的积分；程序只能按点向评测器查询函数值，每例最多 2048 个点、30 秒，按与参照同例配对的误差增益计分。

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
| `v1` | `blackbox-cubature` | 出题人 v4 包原样三族，多项式为守门族（只扣分） |
| `v2` | `blackbox-cubature-v2` | 多项式族换成折点 ridge 叠加，三族都计分（当前版本） |
| `v3` | `blackbox-cubature-v3` | 理论 + 实验双考候选版：高斯测度，困难合成赛道（4 族）与文献期权赛道（6 族）等权，GPCA-RQMC = 50、普通蒙特卡洛 = 0，按真值精度截断；只做试跑，不上榜（analysis/DUAL_PLAN.md） |
| `v3.1` | `blackbox-cubature-v3.1` | v3 的修订候选版：困难赛道改为谱各异的各向异性二次型（广义卡方真值），期权赛道改为单资产局部波动率 Euler 模型（反向 Nyström 真值），样例查询上限 2^18；评分、参照与镜像同 v3；只做试跑，不上榜（analysis/DUAL_PLAN.md 9.9） |
| `v3.2` | `blackbox-cubature-v3.2` | v3.1 加理论评分原则：理论结论须 Lean 4 形式化并通过 lake comparator，每条与人类强结果对比（更强、更弱、相当、不可比），理论分取通过结论中的最高分；开发容器提供离线 Lean v4.35 与 Mathlib、8 GiB 内存；实验部分同 v3.1；只做试跑，不上榜（analysis/DUAL_PLAN.md 9.11，analysis/THEORY_REVIEW.md） |
| `v3.3` | `blackbox-cubature-v3.3` | v3.2 的题面修订：写明提前结束不加分、理论分只看最强的一条通过验证的结论（一条有分量的结论胜过多条初等结论），建议先在 THEORY.md 定目标与证明路线再分步形式化；环境、数据与评分同 v3.2；只做试跑，不上榜（analysis/DUAL_PLAN.md 9.12） |

## 题目背景

多维数值积分（cubature）。函数只能通过查询得到，考察在固定查询预算下怎样选点和估计：准蒙特卡洛（Sobol、数字网）、自适应求积（Genz–Malik、pcubature）、周期函数的格点与梯形规则，以及先识别函数结构再换专门方法。隐藏函数有三族：周期 ridge（fourier）、Genz 乘积峰，以及 2–3 条方向各异、带折点的 ridge 叠加。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | .runtime-ops/blackbox-cubature-v2/workspace/problem.md（出题人题面加 FrontierAM 改动，见 versions/v2/task.json） |
| 公开开发材料 | versions/v2/data/public/（不给模型的反馈集，只在内部仓库） |
| 私有测试集 | versions/v2/data/private/（只在内部仓库） |
| 评测器 | arena/operator_drivers/blackbox_cubature.py（沙箱交互评测）；预言机 versions/v2/grader/oracle.py |
| 参照实现 | references/r1/mc（0 分）、references/r1/portfolio（50 分） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 出题人试跑包 12 例（environment/public-dev）, 示例提交（Sobol） |
| public（不给，反馈用） | A 组 24 例（三族 × 四个维度 × 2） |
| 反馈分类 | 折点 ridge 叠加、fourier ridge、乘积峰 |

## 评测依据

- **协议**：程序经标准输入输出按 JSON Lines 查询函数值，每次最多 512 个点，每例最多 2048 个点。
- **正式集**：144 例（出题人 B、C 两组切分），三族 × 四个维度。
- **资源**：每例全新进程，1 CPU、2 GiB、30 秒墙钟；任一例失败、超时或违反协议，整份提交记 0 分。
- **原始指标**：每例 log10 相对误差截到 \([-15, 0]\)，与本机重跑的 HumanPortfolio-v1 同例配对，增益截到 \([-2, 2]\)，按族×维度平均、维度平均、三族平均得 \(G\)。
- **换算**：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\)，\(G_0\) 为普通蒙特卡洛的增益。

## Baseline

| 参照 | 方法 | 增益 \(G\) | 记分 | 单例最长 |
| --- | --- | --- | --- | --- |
| 弱参照 | 普通蒙特卡洛（2048 个独立均匀点取平均） | −0.713 | 0 | 0.20 秒 |
| 强参照 | HumanPortfolio-v1（2、5 维 pcubature，3、8 维 QMCPy 数字网） | 0 | 50 | 0.92 秒 |

本机评测核实测（private 144 例）。

## 模型成绩与解法（题目版本 v2 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.2 Sol | 单轮 | 增益 \(G = 0.941\)（fourier +1.17，折点 ridge -0.34，乘积峰 +2.00），144 例，单例最长 4.6 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 73.5 | 50 分钟 | [`v2/gpt-6.2-sol-single-1/`](solutions/v6.3/v2/gpt-6.2-sol-single-1/) |
| GPT-6.2 Sol | 迭代 | 增益 \(G = 1.734\)（fourier +1.34，折点 ridge +1.86，乘积峰 +2.00），144 例，单例最长 11.5 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 93.4 | 9 次提交，24 分钟 | [`v2/gpt-6.2-sol-iterative-1/`](solutions/v6.3/v2/gpt-6.2-sol-iterative-1/) |
| GPT-6 Astra | 单轮 | 增益 \(G = 0.349\)（fourier -0.68，折点 ridge -0.27，乘积峰 +2.00），144 例，单例最长 4.0 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 58.7 | 55 分钟 | [`v2/gpt-6-astra-single-1/`](solutions/v6.3/v2/gpt-6-astra-single-1/) |
| GPT-6 Astra | 迭代 | 增益 \(G = 1.357\)（fourier +1.23，折点 ridge +0.84，乘积峰 +2.00），144 例，单例最长 3.4 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 83.9 | 18 次提交，232 分钟 | [`v2/gpt-6-astra-iterative-1/`](solutions/v6.3/v2/gpt-6-astra-iterative-1/) |
| GPT-6.1 Sol | 单轮 | 增益 \(G = 0.371\)（fourier -0.12，折点 ridge -0.76，乘积峰 +2.00），144 例，单例最长 17.3 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 59.3 | 60 分钟 | [`v2/gpt-6.1-sol-single-1/`](solutions/v6.3/v2/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 增益 \(G = 1.496\)（fourier +0.88，折点 ridge +1.60，乘积峰 +2.00），144 例，单例最长 2.7 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 87.4 | 15 次提交，221 分钟 | [`v2/gpt-6.1-sol-iterative-1/`](solutions/v6.3/v2/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 增益 \(G = -0.061\)（fourier -0.42，折点 ridge -0.04，乘积峰 +0.28），144 例，单例最长 1.2 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 45.7 | 11 分钟 | [`v2/claude-opus-5-5-single-1/`](solutions/v6.3/v2/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | 增益 \(G = 1.356\)（fourier +1.56，折点 ridge +0.51，乘积峰 +2.00），144 例，单例最长 1.4 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 83.9 | 16 次提交，97 分钟 | [`v2/claude-opus-5-5-iterative-1/`](solutions/v6.3/v2/claude-opus-5-5-iterative-1/) |
| Hunyuan 4 | 单轮 | 增益 \(G = 0.265\)（fourier -0.44，折点 ridge -0.49，乘积峰 +1.73），144 例，单例最长 3.8 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 56.6 | 51 分钟 | [`v2/hy4-single-1/`](solutions/v6.3/v2/hy4-single-1/) |
| Hunyuan 4 | 迭代 | 增益 \(G = 0.640\)（fourier +0.81，折点 ridge -0.90，乘积峰 +2.00），144 例，单例最长 11.4 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 66.0 | 12 次提交，179 分钟 | [`v2/hy4-iterative-1/`](solutions/v6.3/v2/hy4-iterative-1/) |
| Qwen4 Max | 单轮 | 增益 \(G = 0.629\)（fourier -0.10，折点 ridge -0.02，乘积峰 +2.00），144 例，单例最长 2.5 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 65.7 | 31 分钟 | [`v2/qwen4-max-single-1/`](solutions/v6.3/v2/qwen4-max-single-1/) |
| Qwen4 Max | 迭代 | 增益 \(G = 0.353\)（fourier -0.44，折点 ridge -0.50，乘积峰 +2.00），144 例，单例最长 26.8 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 58.8 | 13 次提交，121 分钟 | [`v2/qwen4-max-iterative-1/`](solutions/v6.3/v2/qwen4-max-iterative-1/) |
| Qwen3.8 Max | 单轮 | 增益 \(G = -0.544\)（fourier -0.77，折点 ridge -0.86，乘积峰 -0.00），144 例，单例最长 5.8 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 11.9 | 56 分钟 | [`v2/qwen3.8-max-0902-single-1/`](solutions/v6.3/v2/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | 增益 \(G = 0.207\)（fourier -0.28，折点 ridge -1.06，乘积峰 +1.96），144 例，单例最长 1.1 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 55.2 | 1 次提交，240 分钟 | [`v2/qwen3.8-max-0902-iterative-1/`](solutions/v6.3/v2/qwen3.8-max-0902-iterative-1/) |
| MiniMax M3.1 Flash | 单轮 | 增益 \(G = -0.094\)（fourier -0.51，折点 ridge -0.07，乘积峰 +0.29），144 例，单例最长 1.2 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 43.4 | 26 分钟 | [`v2/minimax-m3.1-flash-single-1/`](solutions/v6.3/v2/minimax-m3.1-flash-single-1/) |
| MiniMax M3.1 Flash | 迭代 | 增益 \(G = 0.146\)（fourier -0.98，折点 ridge -0.59，乘积峰 +2.00），144 例，单例最长 1.7 秒（普通蒙特卡洛 \(G_0 = -0.713\)，HumanPortfolio-v1 为 0） | 与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\(G \ge 0\) 为 \(50 + 25\,G\)；\(G < 0\) 为 \(\max(0,\,50\,(G - G_0)/(-G_0))\) | 53.6 | 9 次提交，90 分钟 | [`v2/minimax-m3.1-flash-iterative-1/`](solutions/v6.3/v2/minimax-m3.1-flash-iterative-1/) |

### GPT-6.2 Sol（单轮，v6.3）

源码 [`solutions/v6.3/v2/gpt-6.2-sol-single-1/source/`](solutions/v6.3/v2/gpt-6.2-sol-single-1/source/)（提交 `blackbox-cubature-v2-20261120T082441Z-019de5`）。

### GPT-6.2 Sol（迭代，v6.3）

public 分数随测试提交：74.2（6 分钟） → 85.9（9 分钟） → 85.9（12 分钟） → 85.9（13 分钟） → 89.9（16 分钟） → 89.9（17 分钟） → 90.1（19 分钟） → 90.1（21 分钟） → 90.1（23 分钟）

源码 [`solutions/v6.3/v2/gpt-6.2-sol-iterative-1/source/`](solutions/v6.3/v2/gpt-6.2-sol-iterative-1/source/)（提交 `blackbox-cubature-v2-20261120T091101Z-c8aa3c`）。

### GPT-6 Astra（单轮，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

源码 [`solutions/v6.3/v2/gpt-6-astra-single-1/source/`](solutions/v6.3/v2/gpt-6-astra-single-1/source/)（提交 `blackbox-cubature-v2-20261121T175608Z-ef3eb0`）。

### GPT-6 Astra（迭代，v6.3）

经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。

public 分数随测试提交：61.0（8 分钟） → 56.3（16 分钟） → 56.3（22 分钟） → 64.9（26 分钟） → 61.8（32 分钟） → 62.0（36 分钟） → 59.2（39 分钟） → 69.8（41 分钟） → 70.3（46 分钟） → 69.8（52 分钟） → 70.5（56 分钟） → 70.8（63 分钟） → 73.1（90 分钟） → 73.3（106 分钟） → 83.9（138 分钟） → 81.3（168 分钟） → 79.6（193 分钟） → 83.9（216 分钟）

源码 [`solutions/v6.3/v2/gpt-6-astra-iterative-1/source/`](solutions/v6.3/v2/gpt-6-astra-iterative-1/source/)（提交 `blackbox-cubature-v2-20261122T004740Z-4b9782`）。

### GPT-6.1 Sol（单轮，v6.3）

源码 [`solutions/v6.3/v2/gpt-6.1-sol-single-1/source/`](solutions/v6.3/v2/gpt-6.1-sol-single-1/source/)（提交 `blackbox-cubature-v2-20261117T084315Z-e1ad13`）。

### GPT-6.1 Sol（迭代，v6.3）

public 分数随测试提交：54.5（4 分钟） → 53.5（12 分钟） → 52.1（20 分钟） → 61.4（25 分钟） → 61.4（31 分钟） → 54.0（35 分钟） → 54.0（40 分钟） → 64.7（51 分钟） → 66.0（62 分钟） → 65.0（92 分钟） → 76.9（189 分钟） → 75.1（194 分钟） → 75.9（205 分钟） → 93.0（210 分钟） → 93.0（214 分钟）

源码 [`solutions/v6.3/v2/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/v2/gpt-6.1-sol-iterative-1/source/)（提交 `blackbox-cubature-v2-20261117T155806Z-e78d6e`）；未复制的二进制或大文件：libpcubature.so（118,352 字节）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/v2/claude-opus-5-5-single-1/source/`](solutions/v6.3/v2/claude-opus-5-5-single-1/source/)（提交 `blackbox-cubature-v2-20261117T062703Z-dbf6f1`）。

### Claude Opus 5.5（迭代，v6.3）

先用查询检验结构再精确积分：可分离函数做一维自适应 Gauss–Kronrod，脊函数 g(w·x) 估方向后做一维 Chebyshev 积分，分段线性函数定位折面后用有理数精确积分，否则退回通用求积。fourier_ridge 是单方向脊函数、product_peak 可分离，两族因此接近精确（+1.56、+2.00）；16 次测试，public 39.7 → 83.0。超过 80，题目设计弱点已记录待出题人决定（results/OPEN_ITEMS.md）。

public 分数随测试提交：39.7（4 分钟） → 63.8（15 分钟） → 64.8（24 分钟） → 62.6（36 分钟） → 66.1（43 分钟） → 69.0（49 分钟） → 73.7（53 分钟） → 78.3（56 分钟） → 82.9（62 分钟） → 82.5（65 分钟） → 83.0（67 分钟） → 83.0（71 分钟） → 83.0（74 分钟） → 79.6（77 分钟） → 83.0（81 分钟） → 83.0（94 分钟）

源码 [`solutions/v6.3/v2/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/v2/claude-opus-5-5-iterative-1/source/)（提交 `blackbox-cubature-v2-20261117T100518Z-61d35b`）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/v2/hy4-single-1/source/`](solutions/v6.3/v2/hy4-single-1/source/)（提交 `blackbox-cubature-v2-20261117T064039Z-84edbd`）。

### Hunyuan 4（迭代，v6.3）

public 分数随测试提交：29.5（13 分钟） → 56.6（42 分钟） → 57.5（65 分钟） → 63.6（74 分钟） → 61.5（102 分钟） → 30.3（125 分钟） → 61.4（129 分钟） → 63.1（138 分钟） → 63.1（146 分钟） → 62.9（154 分钟） → 62.6（166 分钟） → 63.1（173 分钟）

源码 [`solutions/v6.3/v2/hy4-iterative-1/source/`](solutions/v6.3/v2/hy4-iterative-1/source/)（提交 `blackbox-cubature-v2-20261117T141441Z-0cefd9`）。

### Qwen4 Max（单轮，v6.3）

源码 [`solutions/v6.3/v2/qwen4-max-single-1/source/`](solutions/v6.3/v2/qwen4-max-single-1/source/)（提交 `blackbox-cubature-v2-20261125T113126Z-3c47b7`）。

### Qwen4 Max（迭代，v6.3）

public 分数随测试提交：52.5（19 分钟） → 52.5（39 分钟） → 23.6（50 分钟） → 41.9（62 分钟） → 50.6（70 分钟） → 50.6（76 分钟） → 51.1（82 分钟） → 38.4（88 分钟） → 46.5（96 分钟） → 50.5（102 分钟） → 52.5（107 分钟） → 50.5（112 分钟） → 52.5（117 分钟）

源码 [`solutions/v6.3/v2/qwen4-max-iterative-1/source/`](solutions/v6.3/v2/qwen4-max-iterative-1/source/)（提交 `blackbox-cubature-v2-20261125T185325Z-295817`）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/v2/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/v2/qwen3.8-max-0902-single-1/source/)（提交 `blackbox-cubature-v2-20261117T054921Z-23c0f5`）。

### Qwen3.8 Max（迭代，v6.3）

两次尝试都有回合遇到上游（阿里百炼）流约 600 秒没有任何输出后断开（AIHub 记为成功、0 token，无其他供应商可换），第 1 次损失约 23 分钟、第 2 次约 49 分钟；按用户决定（11-18）取损失较少的第 1 次为正式格。

public 分数随测试提交：44.1（114 分钟）

源码 [`solutions/v6.3/v2/qwen3.8-max-0902-iterative-1/source/`](solutions/v6.3/v2/qwen3.8-max-0902-iterative-1/source/)（提交 `blackbox-cubature-v2-20261117T140100Z-184b43`）。

### MiniMax M3.1 Flash（单轮，v6.3）

源码 [`solutions/v6.3/v2/minimax-m3.1-flash-single-1/source/`](solutions/v6.3/v2/minimax-m3.1-flash-single-1/source/)（提交 `blackbox-cubature-v2-20261125T114442Z-a1cb07`）。

### MiniMax M3.1 Flash（迭代，v6.3）

public 分数随测试提交：52.2（19 分钟） → 52.2（26 分钟） → 52.3（36 分钟） → 39.0（41 分钟） → 53.3（54 分钟） → 53.3（55 分钟） → 53.3（63 分钟） → 51.6（67 分钟） → 51.6（84 分钟）

源码 [`solutions/v6.3/v2/minimax-m3.1-flash-iterative-1/source/`](solutions/v6.3/v2/minimax-m3.1-flash-iterative-1/source/)（提交 `blackbox-cubature-v2-20261125T204134Z-599310`）。

## 观察与说明

- v2 用同种子的折点 ridge 叠加替换出题人的 4 次多项式 ridge（认出方向即可精确积分）；fourier 与乘积峰原样保留。
- 出题人原指标（三族中位数的中位数）只由 fourier 一族决定、参照顺序在切分间反转，改用配对增益；两处改动待出题人确认。
- 给模型的样例是出题人的 12 例试跑包（Genz 风格族），public 反馈为 A 组每个（族，维度）2 例共 24 例。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
