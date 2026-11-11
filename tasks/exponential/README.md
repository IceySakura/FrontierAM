# exponential

对大型稀疏对称半正定矩阵计算 \(e^{-tL}b_0 + t\varphi_1(-tL)b_1\)，在给定精度下越快越好；参照是 7 个已发表方法逐例取最快者。

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
| `v1` | `exponential` | 旧版：解释器中单个 KIOPS 作参考、私有用例几千维；已由 v2 取代 |
| `v2` | `exponential-v2` | 对标 7 个已发表方法的参照组；数据集与设计见 versions/v2/（当前版本） |
| `v3` | `exponential-v3` | 协议 v6.3 版本：单请求 15 秒、复用 150 秒、启动 10 秒；public 反馈 16 例；工作区只含 6 个样例（scripts/seal/prepare_exp_v3.py） |

## 题目背景

求矩阵指数作用在向量上的结果 \(e^{-tL}b\)（以及带源项的 \(\varphi_1\) 函数）是求解刚性线性常微分方程、热传导和扩散过程、图上信息传播和指数积分器的核心操作。矩阵太大不能直接求矩阵指数，主流方法是 Krylov 子空间（Lanczos、自适应 Krylov）、缩放加截断 Taylor、在谱区间上做 Chebyshev 展开，以及有理/移位求逆 Krylov。好的实现要根据矩阵的谱范围、精度要求和右端数量选方法，并在多个右端之间复用计算。

这一版的设计原则是**学术上可信**：参照组只用已发表的方法和作者代码或公认的库，参数只在公开数据上标定并在任何模型提交之前冻结；私有测试集在参照组冻结之后才生成。因此“超过参照组”意味着在这些用例上快于已发表的最好方法。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | versions/v3/statement.md |
| 公开开发材料 | versions/v3/data/samples/（给模型的样例）；versions/v3/data/public/（不给模型的反馈集，只在内部仓库） |
| 私有测试集 | versions/v3/data/private/（只在内部仓库） |
| 评测器 | arena/exponential_v2.py（流式接口、精度检查、计时、virtual best） |
| 参照实现 | references/r1/（7 个成员：Expokit、KIOPS、expmv、SciPy、Chebyshev、SLEPc、ExponentialUtilities.jl；标定 references/r1/calibration/）；复核中的候选 references/candidates/ |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | syn_diffusion2d_p4_k1, syn_geometric_p4_k1, syn_heavytail_p4_k1, syn_logspectrum_p4_k1, syn_signed_p4_k1, control_diagonal |
| public（不给，反馈用） | 16 例全尺寸公开用例（每个结构族单请求、复用各 1 例） |
| 反馈分类 | 结构族（manifest 的 family 字段）：mesh_diffusion / standard_fem / geometric_graph / standard_graph / network_operator / signed_incidence / diagonal_local / analytic |

## 评测依据

- **输入**：会话给定稀疏矩阵 \(L\)（\(n\) 为 3.5 万–12 万，非零元最多约 90 万）、时间 \(t\) 和精度档；依次收到 1 个（one_shot）或 16 个（reuse）右端向量对，每次返回完整向量。
- **正式集**：11 个母矩阵（SuiteSparse 标准矩阵 5 个，包括有限元和真实网络图；合成算子 6 族，与公开集的结构族相同；母矩阵与公开集不重合）× 3 个精度档（rtol 1e-4 / 1e-7 / 1e-10）× 2 种场景，共 66 个计分会话，另有 8 个解析控制例。
- **参考解**：扩展精度 Chebyshev 为主，双精度 Taylor 独立审计，差异超过容差 2% 时用稠密特征分解仲裁；66 例全部合格，最大审计差异为门槛的 0.0156。
- **有效性**：所有用例（含控制例）三次重复都必须在资源上限内达到精度，任何一例失败整份提交记 0 分。
- **计时**：从交付矩阵第一个字节到全部结果输出、进程退出；READY 之前的导入不计。每例三次取中位数。
- **原始指标**：\(R\) = 每例耗时与参照组该例最快者之比，按母矩阵、结构族、场景、精度逐层等权平均。
- **换算**：分数 \(=100/(1+R)\)。与参照组逐例最快者持平为 50 分，平均快一倍为 66.7 分。补充报告几何速度比 \(S\) 及其 95% 分层 bootstrap 区间。

## Baseline

参照组（panel）7 个成员，全部在公开数据上按事先声明的规则标定后冻结（2026-11-06 17:04）：

| 成员 | 来源 |
| --- | --- |
| Expokit | Sidje 1998，作者 Fortran 代码 |
| KIOPS | Gaudreault 等 2018，作者代码 |
| expmv | Al-Mohy & Higham 2011，作者代码 |
| SciPy `expm_multiply` | SciPy 库 |
| Chebyshev | Tal-Ezer & Kosloff 等，由未接触任何模型提交的独立 agent 按文献实现 |
| SLEPc MFN | SLEPc 3.26.1 库 |
| ExponentialUtilities.jl | SciML Julia 库 |

每个用例取三次都合格的成员中最快者作为参考时间（virtual best），在本机评测核上实测，覆盖全部 74 例。逐例最快的成员：Chebyshev 60 例、Expokit 9 例、SLEPc 4 例、ExponentialUtilities.jl 1 例。有理（移位求逆）Krylov 成员因在一个公开用例上精度略超余量未能入选。

## 模型成绩与解法（题目版本 v3 · 协议 v6.3）

协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 至 11-11）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | 74/74 合格，\(R = 0.813\)（对 7 方法参照组逐例最快成员） | \(100/(1+R)\) | 55.2 | 54 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.3/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 74/74 合格，\(R = 0.679\)（对 7 方法参照组逐例最快成员） | \(100/(1+R)\) | 59.6 | 7 次提交，186 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.3/gpt-6.1-sol-iterative-1/) |

### GPT-6.1 Sol（单轮，v6.3）

C++ 实现：Chebyshev 展开把指数与 φ₁ 合成一条稳定 Clenshaw 递推，正向量加权 Gershgorin 收紧谱区间；小连通块扩展精度对角化。

源码 [`solutions/v6.3/gpt-6.1-sol-single-1/source/`](solutions/v6.3/gpt-6.1-sol-single-1/source/)（提交 `exponential-v3-20261111T003322Z-4a23d6`）；未复制的二进制或大文件：solver（151,704 字节）。

### GPT-6.1 Sol（迭代，v6.3）

对角与 2×2 块解析求解，小连通块扩展精度特征分解；其余用 Chebyshev 展开，两种矩阵函数共用一条 Clenshaw 递推，按输出大小自适应截断，另有自适应 Lanczos 试探。

源码 [`solutions/v6.3/gpt-6.1-sol-iterative-1/source/`](solutions/v6.3/gpt-6.1-sol-iterative-1/source/)（提交 `exponential-v3-20261110T141618Z-f78745`）；未复制的二进制或大文件：solver（175,184 字节）。

<details><summary>旧版本的成绩与解法</summary>

## 模型成绩与解法（题目版本 v2 · 协议 v6.2）

协议 v6.2（2 小时，单轮 / 迭代，2026-11-09 开发，11-10 评测）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | 58 例精度不合格 | 任一例失败记 0 | 0.0 | 23 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.2/gpt-6.1-sol-single-1/) |
| GPT-6.1 Sol | 迭代 | 74/74 合格，R = 0.589 | \(100/(1+R)\) | 62.9 | 4 次提交，68 分钟 | [`gpt-6.1-sol-iterative-1/`](solutions/v6.2/gpt-6.1-sol-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 74/74 合格，R = 0.412 | \(100/(1+R)\) | 70.8 | 62 分钟 | [`claude-opus-5-5-single-1/`](solutions/v6.2/claude-opus-5-5-single-1/) |
| Claude Opus 5.5 | 迭代 | 74/74 合格，R = 0.448 | \(100/(1+R)\) | 69.1 | 10 次提交，60 分钟 | [`claude-opus-5-5-iterative-1/`](solutions/v6.2/claude-opus-5-5-iterative-1/) |
| Claude Opus 5.5 | 单轮 | 74/74 合格，R = 0.659 | \(100/(1+R)\) | 60.3 | 53 分钟 | [`claude-opus-5-5-single-2/`](solutions/v6.2/claude-opus-5-5-single-2/) |
| Claude Opus 5.5 | 单轮 | 74/74 合格，R = 0.934 | \(100/(1+R)\) | 51.7 | 76 分钟 | [`claude-opus-5-5-single-3/`](solutions/v6.2/claude-opus-5-5-single-3/) |
| GPT-6.1 Sol | 单轮 | 53 例精度不合格（无效提交） | \(100/(1+R)\) | 0.0 | 4 分钟 | [`gpt-6.1-sol-single-2/`](solutions/v6.2/gpt-6.1-sol-single-2/) |
| GPT-6.1 Sol | 单轮 | 74/74 合格，R = 0.758 | \(100/(1+R)\) | 56.9 | 100 分钟 | [`gpt-6.1-sol-single-3/`](solutions/v6.2/gpt-6.1-sol-single-3/) |

### GPT-6.1 Sol（单轮，v6.2）

23 分钟就结束回合。最后几分钟把源项改成“并入 \(b_0 + t\,b_1\) 再乘 \(e^{-tL}\)”，这在数学上不等价；改完只做了协议检查，没有重跑数值检验，私有集 58 例精度不合格（含解析控制例）。

源码 [`solutions/v6.2/gpt-6.1-sol-single-1/source/`](solutions/v6.2/gpt-6.1-sol-single-1/source/)（提交 `exponential-v2-20261109T090735Z-9ccd95`）；未复制的二进制或大文件：solver（58,896 字节）。

### GPT-6.1 Sol（迭代，v6.2）

C++ 单个 Chebyshev 展开同时计算两项（\(e^{-tL}b_0\) 与 \(t\varphi_1(-tL)b_1\)），谱上界用行和与 Collatz 界，抵消严重的输出用扩展精度重算。68 分钟正式提交；4 次全量公开集测试 57.5、59.3、58.7、59.5。私有集 \(S = 1.84\)，95% 区间 [1.48, 2.30]，与 v6.1 的单轮成绩（62.7）相当。

源码 [`solutions/v6.2/gpt-6.1-sol-iterative-1/source/`](solutions/v6.2/gpt-6.1-sol-iterative-1/source/)（提交 `exponential-v2-20261109T153537Z-142f3b`）；未复制的二进制或大文件：solver（57,208 字节）。

### Claude Opus 5.5（单轮，v6.2）

C 实现的 Chebyshev 展开，长展开时用短 Lanczos 收紧谱上界，SELL-4 稀疏格式加 AVX2 内核，按实际输出大小自适应截断。62 分钟结束回合。私有集 \(S = 2.99\)，95% 区间 [2.05, 4.47]，是本题目前最快的提交。

源码 [`solutions/v6.2/claude-opus-5-5-single-1/source/`](solutions/v6.2/claude-opus-5-5-single-1/source/)（提交 `exponential-v2-20261109T125907Z-5a6038`）；未复制的二进制或大文件：expo_v1（51,360 字节）、expo_v3（51,432 字节）。

### Claude Opus 5.5（迭代，v6.2）

把 \(\varphi_1\) 项写成增广矩阵的指数，一次 Chebyshev 递推同时得到两项；小连通分量直接对角化，几何图先做 RCM 重排。60 分钟内 10 次公开集测试（全量最高 63.7）后正式提交。私有集 \(S = 2.60\)，95% 区间 [1.88, 3.69]，与单轮尝试相当（区间重叠）。

源码 [`solutions/v6.2/claude-opus-5-5-iterative-1/source/`](solutions/v6.2/claude-opus-5-5-iterative-1/source/)（提交 `exponential-v2-20261109T175508Z-fdfad8`）；未复制的二进制或大文件：expo（42,600 字节）。

### Claude Opus 5.5（单轮，v6.2）

源码 [`solutions/v6.2/claude-opus-5-5-single-2/source/`](solutions/v6.2/claude-opus-5-5-single-2/source/)（提交 `exponential-v2-20261110T041026Z-22d0e1`）；未复制的二进制或大文件：bin/expprop_avx2（39,200 字节）、bin/expprop_generic（39,200 字节）。

### Claude Opus 5.5（单轮，v6.2）

源码 [`solutions/v6.2/claude-opus-5-5-single-3/source/`](solutions/v6.2/claude-opus-5-5-single-3/source/)（提交 `exponential-v2-20261110T043349Z-380434`）；未复制的二进制或大文件：build/libk_avx2.so（19,792 字节）、build/libk_base.so（19,792 字节）。

### GPT-6.1 Sol（单轮，v6.2）

源码 [`solutions/v6.2/gpt-6.1-sol-single-2/source/`](solutions/v6.2/gpt-6.1-sol-single-2/source/)（提交 `exponential-v2-20261110T031548Z-80934e`）。

### GPT-6.1 Sol（单轮，v6.2）

源码 [`solutions/v6.2/gpt-6.1-sol-single-3/source/`](solutions/v6.2/gpt-6.1-sol-single-3/source/)（提交 `exponential-v2-20261110T053617Z-d8bb77`）；未复制的二进制或大文件：solver（288,792 字节）。

## 模型成绩与解法（题目版本 v2 · 协议 v6.1）

协议 v6.1（2026-11-06 开发，11-09 评测）

| 模型 | 设置 | 原始指标 | 换算 | 得分 | 提交次数与用时 | 目录 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol |  | 74/74 合格，R = 0.596 | \(100/(1+R)\) | 62.7 | 34 分钟 | [`gpt-6.1-sol-single-1/`](solutions/v6.1/gpt-6.1-sol-single-1/) |

### GPT-6.1 Sol（，v6.1）

几何速度比 \(S = 2.01\)，95% 区间 [1.45, 2.91]，下限大于 1：在这个私有集上显著快于参照组的逐例最快者。优势集中在几何图（耗时为参照的 0.14 倍）、有限元标准矩阵（0.29）和网格扩散（0.34）；66 个计分会话中有 7 个比参照慢（最多 1.34 倍）。复用 16 个右端的场景优势更大。

源码 [`solutions/v6.1/gpt-6.1-sol-single-1/source/`](solutions/v6.1/gpt-6.1-sol-single-1/source/)（提交 `exponential-v2-20261106T043748Z-9f6d55`）；未复制的二进制或大文件：solver（179,048 字节）。

</details>

## 观察与说明

- 逐例最快的参照成员有 60/74 是按文献实现的 Chebyshev 方法，不是作者代码；如果这一实现本身不够强，参照组会被低估。是否替换为作者代码仍在讨论。
- sol 的单轮尝试 23 分钟就结束回合，最后改动源项处理后没有重新检验，私有集 58 例精度不合格；迭代设置下它做了 4 次公开集测试，成绩正常。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
