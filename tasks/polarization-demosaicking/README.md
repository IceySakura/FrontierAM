# polarization-demosaicking

从一张 1024×1024 的偏振马赛克中重建 0°/45°/90°/135° 四幅完整图像，每景 1 秒，132 个正式场景，RMSE 越小越好。

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
| `v1.2` | `polarization-demosaicking` | 出题方维护包 v1.2（32 景测试集） |
| `v2` | `polarization-demosaicking-v2` | 协议 v6.3 版本：工作区增加 2 景公开场景作样例，其余不变（scripts/seal/prepare_v63.py） |
| `v3` | `polarization-demosaicking-v3` | 出题人 vNext 1.3.0-dev：132 个正式场景（Qiu32 + Wen100），Wen5 为样例与 public 反馈（当前版本） |

## 题目背景

偏振相机在每个像素前放一个微型偏振片，相邻 2×2 像素分别测 4 个偏振方向，所以每个方向只直接测到 1/4 的像素，其余要插值重建（去马赛克）。偏振信息用于识别材质、去除反光、工业检测和三维形状恢复，重建误差会直接放大到偏振度和偏振角的计算里。经典做法是逐方向双线性插值；更先进的方法（如 LEIC、EARI）利用不同方向之间的相关性和边缘信息。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v3/statement.md |
| 公开开发材料 | versions/v3/data/samples/（Wen5，给模型的样例，也是 public 反馈集） |
| 私有测试集 | versions/v3/data/private/（qiu32、wen100，只在内部仓库） |
| 评测器 | 维护包 harbor/polarization-demosaicking/grade_vnext.py（只改 CPU 检查；评分前预热镜像文件缓存） |
| 参照实现 | references/r1/（逐方向双线性；镜像 LEIC） |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | Wen5（出题人开发场景） |
| public（不给，反馈用） | Wen5（与样例相同；Wen100 在正式集里，不能用作反馈） |
| 反馈分类 | 不分类，只报总计 |

## 评测依据

- **输入**：1024×1024 的 `uint8` 马赛克，滤片相位固定。
- **正式集**：132 个场景（出题人 vNext：Qiu 32 景 + Wen 100 景）。
- **资源**：每景独立进程，单核、8 GiB、墙钟 1.000 秒（含 Python 启动和导入）。
- **有效性**：任一景超时、超内存或输出非法，整份提交记 0 分。
- **原始指标**：每景对 4 个方向分别算全图 RMSE 再平均，132 景等权平均得 \(E\)。
- **换算**：双线性的误差为 \(B\)、强参照为 \(H_5\)：\(E \ge B\) 记 0；\(H_5 \le E < B\) 为 \(50\,(B-E)/(B-H_5)\)；\(E < H_5\) 为 \(100 - 50\,E/H_5\)。

## Baseline

| 参照 | 方法 | 整批 RMSE | 记分 | 每景最大墙钟 |
| --- | --- | --- | --- | --- |
| 弱参照 | 逐方向双线性插值 | 2.10798 | 0 | 0.204 秒 |
| 强参照 | 出题人强参照（镜像延拓） | 1.50069 | 50 | 0.890 秒 |

本机评测核实测（132 景）。

## 结果解读

| 类别 | 结论 | 置信度 | 关注 |
| --- | --- | --- | --- |
| 工程或组合小幅超过参照 | v6.3 单轮 53.5（95% 区间 51.8–54.9）、迭代 52.9（51.4–54.4），都确实高于参照 LEIC；v6.2 opus 55.9。sol 用约 7.7k 参数的小卷积网络，只用 2 张公开场景训练；opus 用频域解调加学习滤波。提升来自组合还是参照偏弱，需要组件消融。（证据：`scripts/analyze/v63-bootstrap.py`） | 实测（区间）/ 待查（归因） |  |
| 数据 | 私有集 32 景，分数区间约 ±1.5 分，能区分 2 分以上的差距。（证据：`results/INSIGHTS.md`） | 实测 |  |

## 模型成绩与解法（题目版本 v3 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | 无效提交（formal scene failure） | 任一用例失败或提交检查未通过记 0 | 0.0 | 13 分钟 | [`v3/claude-opus-5-5-single-1/`](solutions/v6.3/v3/claude-opus-5-5-single-1/) |

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/v3/claude-opus-5-5-single-1/source/`](solutions/v6.3/v3/claude-opus-5-5-single-1/source/)（提交 `polarization-demosaicking-v3-20261117T045252Z-3abed5`）；未复制的二进制或大文件：weights.npy（38,528 字节）。

<details><summary>旧版本的成绩与解法</summary>

## 模型成绩与解法（题目版本 v2 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | RMSE \(E = 1.0320\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 53.5 | 52 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | RMSE \(E = 1.0456\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 52.9 | 23 次提交，211 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | RMSE \(E = 1.4320\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 26.9 | 31 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.3/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | RMSE \(E = 1.2173\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 42.3 | 22 次提交，169 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.3/claude-opus-5-5-iterative-1/) |
| Qwen3.8 Max | 单轮 | RMSE \(E = 1.1935\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 44.0 | 30 分钟 | [`qwen3.8-max-0902-single-1/`](solutions/v6.3/qwen3.8-max-0902-single-1/) |
| Qwen3.8 Max | 迭代 | RMSE \(E = 1.1512\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 47.0 | 14 次提交，202 分钟 | [`qwen3.8-max-0902-iterative-1/`](solutions/v6.3/qwen3.8-max-0902-iterative-1/) |
| Hunyuan 4 | 单轮 | 无效提交（提交失败） | 任一用例失败或提交检查未通过记 0 | 0.0 | 43 分钟 | [`hy4-single-1/`](solutions/v6.3/hy4-single-1/) |
| Hunyuan 4 | 迭代 | RMSE \(E = 1.2144\)（\(U = 1.8082\)，\(W = 1.1099\)） | \(E \ge U\) 记 0；\(W \le E < U\) 为 \(50\,(U-E)/(U-W)\)；\(E < W\) 为 \(100 - 50\,E/W\) | 42.5 | 31 次提交，196 分钟 | [`hy4-iterative-1/`](solutions/v6.3/hy4-iterative-1/) |
| Intern S2 | 单轮 | 无效提交（开发结束时没有冻结出有效提交） | 无效提交记 0 | 0.0 | 60 分钟 | [`intern-s2-single-1/`](solutions/v6.3/intern-s2-single-1/) |
| Intern S2 | 迭代 | 无效提交（solution.py and README.md are required） | 任一用例失败或提交检查未通过记 0 | 0.0 | 0 次提交，240 分钟 | [`intern-s2-iterative-1/`](solutions/v6.3/intern-s2-iterative-1/) |

### GPT-6.1 Sol（单轮，v6.3）

低通估计通道差、把马赛克细节转移到四个通道，再用约 7.7k 参数的残差卷积网络修正（只用 2 张公开样例加增强训练），最后做以强度为引导的局部仿射回归。

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `polarization-demosaicking-v2-20261110T224315Z-da6c17`）；未复制的二进制或大文件：weights.npz（33,421 字节）。

### GPT-6.1 Sol（迭代，v6.3）

逐格双线性插值加通道差的偏振估计，用全部测量恢复共同细节引导，再用小型残差卷积网络修正，带结构门控与残差校准。

public 分数随测试提交：35.3（4 分钟） → 0.0（21 分钟） → 0.0（26 分钟） → 0.0（33 分钟） → 0.0（37 分钟） → 52.0（42 分钟） → 52.0（46 分钟） → 52.2（51 分钟） → 0.0（56 分钟） → 0.0（62 分钟） → 52.2（89 分钟） → 52.4（97 分钟） → 52.4（109 分钟） → 52.4（118 分钟） → 52.4（133 分钟） → 52.4（166 分钟） → 52.5（172 分钟） → 52.4（175 分钟） → 51.5（182 分钟） → 52.0（188 分钟） → 52.5（193 分钟） → 51.9（200 分钟） → 52.5（207 分钟）

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `polarization-demosaicking-v2-20261110T231338Z-f36b08`）；未复制的二进制或大文件：weights.npz（104,886 字节）。

### Claude Opus 5.5（单轮，v6.3）

源码 [`solutions/v6.3/claude-opus-5-5-single-1/source/`](solutions/v6.3/claude-opus-5-5-single-1/source/)（提交 `polarization-demosaicking-v2-20261112T081428Z-924e95`）；未复制的二进制或大文件：gated_r4.npz（25,542 字节）。

### Claude Opus 5.5（迭代，v6.3）

public 分数随测试提交：47.3（20 分钟） → 51.0（31 分钟） → 0.0（35 分钟） → 0.0（38 分钟） → 0.0（59 分钟） → 0.0（66 分钟） → 0.0（72 分钟） → 52.1（78 分钟） → 0.0（86 分钟） → 0.0（93 分钟） → 52.0（98 分钟） → 0.0（101 分钟） → 0.0（105 分钟） → 0.0（112 分钟） → 47.0（117 分钟） → 50.0（127 分钟） → 0.0（131 分钟） → 51.6（138 分钟） → 51.3（141 分钟） → 51.9（145 分钟） → 52.2（157 分钟） → 52.2（166 分钟）

源码 [`solutions/v6.3/claude-opus-5-5-iterative-1/source/`](solutions/v6.3/claude-opus-5-5-iterative-1/source/)（提交 `polarization-demosaicking-v2-20261112T064029Z-017b5e`）；未复制的二进制或大文件：model.npz（19,330 字节）。

### Qwen3.8 Max（单轮，v6.3）

源码 [`solutions/v6.3/qwen3.8-max-0902-single-1/source/`](solutions/v6.3/qwen3.8-max-0902-single-1/source/)（提交 `polarization-demosaicking-v2-20261113T120225Z-89850e`）。

### Qwen3.8 Max（迭代，v6.3）

public 分数随测试提交：34.7（78 分钟） → 26.0（100 分钟） → 37.0（128 分钟） → 39.8（133 分钟） → 37.8（139 分钟） → 39.7（144 分钟） → 35.8（148 分钟） → 34.6（160 分钟） → 36.9（165 分钟） → 40.1（179 分钟） → 39.5（183 分钟） → 39.7（190 分钟） → 39.4（195 分钟） → 40.1（200 分钟）

源码 [`solutions/v6.3/qwen3.8-max-0902-iterative-1/source/`](solutions/v6.3/qwen3.8-max-0902-iterative-1/source/)（提交 `polarization-demosaicking-v2-20261113T101122Z-cd0499`）；未复制的二进制或大文件：filter_map_30_38.npy（153,728 字节）、filter_map_528_528.npy（35,819,648 字节）。

### Hunyuan 4（单轮，v6.3）

源码 [`solutions/v6.3/hy4-single-1/source/`](solutions/v6.3/hy4-single-1/source/)（提交 `polarization-demosaicking-v2-20261113T113200Z-32fbde`）；未复制的二进制或大文件：params.npz（25,312 字节）。

### Hunyuan 4（迭代，v6.3）

public 分数随测试提交：38.0（13 分钟） → 46.8（21 分钟） → 41.2（27 分钟） → 41.4（37 分钟） → 42.2（50 分钟） → 46.8（65 分钟） → 44.0（76 分钟） → 50.5（83 分钟） → 49.2（94 分钟） → 50.6（100 分钟） → 50.6（104 分钟） → 0.0（112 分钟） → 48.0（120 分钟） → 49.4（124 分钟） → 50.2（126 分钟） → 50.7（129 分钟） → 50.7（131 分钟） → 50.8（140 分钟） → 50.8（142 分钟） → 0.0（145 分钟） → 51.0（148 分钟） → 51.1（151 分钟） → 51.1（164 分钟） → 51.1（166 分钟） → 51.1（175 分钟） → 51.4（177 分钟） → 51.3（180 分钟） → 0.0（182 分钟） → 51.4（187 分钟） → 0.0（190 分钟） → 51.4（193 分钟）

源码 [`solutions/v6.3/hy4-iterative-1/source/`](solutions/v6.3/hy4-iterative-1/source/)（提交 `polarization-demosaicking-v2-20261113T142233Z-a57fcb`）。

### Intern S2（单轮，v6.3）

源码 [`solutions/v6.3/intern-s2-single-1/source/`](solutions/v6.3/intern-s2-single-1/source/)（提交 `polarization-demosaicking-v2-20261116T053043Z-b06743`）；未复制的二进制或大文件：（提交存档不在本服务器）。

### Intern S2（迭代，v6.3）

源码 [`solutions/v6.3/intern-s2-iterative-1/source/`](solutions/v6.3/intern-s2-iterative-1/source/)（提交 `polarization-demosaicking-v2-20261116T093358Z-fd94ba`）。

## 模型成绩与解法（题目版本 v1.2 · 协议 v6.2）

协议 v6.2 单轮（2026-11-09 开发，11-10 评分）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 单轮 | 平均 RMSE 0.97877，32/32 景通过（每景最多 0.710 秒） | \(100 - 50\,L/\text{强}\) | 55.9 | 46 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |

### Claude Opus 5.5（单轮，v6.2）

频率复用模型：把马赛克写成强度加两个偏振载波，载波用离线学到的 11×11 滤波器按局部结构方向加权解调，再在比值域做边缘感知平滑，只用 NumPy（FFT 卷积）。整批 RMSE 0.97877，比强参照（LEIC，1.10992）低 12%，也比 sol 的 v6.1 提交（1.09499）好。

第一次评分（11-10 03:53）31/32 景通过，第一个运行的景 1.003 秒超过 1 秒上限，其余 31 景 0.56–0.58 秒。它的耗时只取决于图像尺寸、与内容无关，判为评测环境的首景冷启动（与 sol 11-06 第一次评分相同），第一次记为未测；07:14 重评 32/32 通过，第一景 0.710 秒，仍是最慢的一景，说明首景偏慢是系统性的，已建议评分器加不计时的预热运行。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `polarization-demosaicking-20261109T124343Z-553533`）；未复制的二进制或大文件：kernels.npy（9,808 字节）。

## 模型成绩与解法（题目版本 v1.2 · 协议 v6.1）

协议 v6.1（2026-11-06）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol |  | 整批 RMSE 1.09499，32/32 景通过（每景最多 0.591 秒） | \(100-50E/H_5\) | 50.7 | 66 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.1/gpt-6.1-sol-single-1/) |

### GPT-6.1 Sol（，v6.1）

比强参照略好（19/32 景更好）。第一次评分时第一景因评测机调度核上的其他编译负载超过 1 秒被终止，按“平台故障不计”的约定记为未测，正式成绩取重评（默认决定，待讨论）。

源码 [`solutions/v6.1/gpt-6.1-sol-single-1/source/`](solutions/v6.1/gpt-6.1-sol-single-1/source/)（提交 `polarization-demosaicking-20261106T050331Z-a6d746`）。

</details>

## 观察与说明

- 题目原包的开发材料只有两组合成样例；反馈集用 Wen 等 2019 的公开数据集（105 景，裁成 1024×1024，LEIC 在 100/105 景上优于双线性），待出题人确认。
- 测试场景经过按参照表现的筛选（去掉了双线性比 LEIC 更准的场景），会放大两条参照之间的差距，已请出题人确认。
- 评分时第一景偶尔慢 0.14–0.5 秒，原因是镜像文件不在页缓存里；评分前先预热镜像文件，已消除。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
