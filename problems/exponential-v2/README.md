# 稀疏系统的指数传播（对标人类 SOTA）

对大型稀疏对称半正定矩阵计算 \(e^{-tL}b_0 + t\varphi_1(-tL)b_1\)，在给定精度下越快越好；参照是 7 个已发表方法逐例取最快者。

看板页面：http://9.135.91.91:18765/tasks/exponential-v2.html（办公网）

## 怎么读这道题

1. **题目背景**：下面第一节。
2. **题面原文**：[`statement.md`](statement.md)，与给模型的版本逐字相同。
3. **数据**：见“数据与代码在哪里”；公开开发材料的文本文件在 [`public/`](public/)。
4. **评测与计分**：“评测依据”一节；评测器位置见下表。
5. **Baseline**：“Baseline”一节。
6. **模型解法**：“模型成绩与解法”一节；每份提交的完整源码在 [`solutions/`](solutions/)。

## 题目背景

求矩阵指数作用在向量上的结果 \(e^{-tL}b\)（以及带源项的 \(\varphi_1\) 函数）是求解刚性线性常微分方程、热传导和扩散过程、图上信息传播和指数积分器的核心操作。矩阵太大不能直接求矩阵指数，主流方法是 Krylov 子空间（Lanczos、自适应 Krylov）、缩放加截断 Taylor、在谱区间上做 Chebyshev 展开，以及有理/移位求逆 Krylov。好的实现要根据矩阵的谱范围、精度要求和右端数量选方法，并在多个右端之间复用计算。

这一版的设计原则是**学术上可信**：参照组只用已发表的方法和作者代码或公认的库，参数只在公开数据上标定并在任何模型提交之前冻结；私有测试集在参照组冻结之后才生成。因此“超过参照组”意味着在这些用例上快于已发表的最好方法。

## 数据与代码在哪里

| 内容 | 位置 |
| --- | --- |
| 题包 | tasks/exponential/sota_v1/（设计 DESIGN.md、数据生成 dataset_v2/、参照组 panel/、标定 calibration_v2/） |
| 公开开发材料 | .runtime-exp2/public.tar（本服务器，git 忽略）：task/ 下的接口说明与公开用例；用例清单见 tasks/exponential/sota_v1/dataset_v2/cases/public*/ |
| 私有测试集 | tasks/exponential/sota_v1/dataset_v2/cases/private/（清单）；矩阵与参考解封存在 .runtime-exp2/（本服务器，git 忽略） |
| 评测器 | arena/exponential_v2.py（流式接口评测、精度检查、计时与 virtual best 计分） |
| 参照实现 | tasks/exponential/sota_v1/panel/（7 个已发表方法：Expokit、KIOPS、expmv、SciPy、Chebyshev、SLEPc、ExponentialUtilities.jl） |
| 详细文档 | `results/exponential/EXPONENTIAL_V2.md` |

公开包中的数据文件（未复制到本目录）：task/dev/syn_diffusion2d_p4_k1.npz（69,183 字节）；task/dev/syn_diffusion2d_p4_k16.npz（247,777 字节）；task/dev/syn_diffusion2d_p7_k1.npz（53,860 字节）；task/dev/syn_diffusion2d_p7_k16.npz（247,998 字节）；task/dev/syn_diffusion2d_p10_k1.npz（77,071 字节）；task/dev/syn_diffusion2d_p10_k16.npz（248,127 字节）；task/dev/syn_diffusion2d_b_p4_k1.npz（50,076 字节）；task/dev/syn_diffusion2d_b_p4_k16.npz（230,566 字节）；task/dev/syn_diffusion2d_b_p7_k1.npz（36,181 字节）；task/dev/syn_diffusion2d_b_p7_k16.npz（230,860 字节）；task/dev/syn_diffusion2d_b_p10_k1.npz（57,865 字节）；task/dev/syn_diffusion2d_b_p10_k16.npz（231,160 字节）；……。

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

## 模型成绩与解法（协议 v6.2，最新）

| 模型 | 设置 | 原始指标 | 换算 | 得分（满分 100，参照 50） | 复核 | 源码 |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6.1 Sol | 单轮 | 58 例精度不合格 | 任一例失败记 0 | 0.0 |  | [`solutions/gpt-6.1-sol-single/`](solutions/gpt-6.1-sol-single/) |
| GPT-6.1 Sol | 迭代 | 74/74 合格，R = 0.589 | \(100/(1+R)\) | 62.9 | 待复核 | [`solutions/gpt-6.1-sol-iterative/`](solutions/gpt-6.1-sol-iterative/) |
| Claude Opus 5.5 | 单轮 | 74/74 合格，R = 0.412 | \(100/(1+R)\) | 70.8 | 待复核 | [`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/) |
| Claude Opus 5.5 | 迭代 | 74/74 合格，R = 0.448 | \(100/(1+R)\) | 69.1 | 待复核 | [`solutions/claude-opus-5-5-iterative/`](solutions/claude-opus-5-5-iterative/) |

超过参照（50 分）的成绩在复核结论出来之前标“待复核”：复核会解释超出的原因、寻找更强的人类方法并做消融，见 `results/BASELINE_REVIEW.md` 与 `results/BASELINE_POLICY.md`。

### GPT-6.1 Sol（单轮）

23 分钟就结束回合。最后几分钟把源项改成“并入 \(b_0 + t\,b_1\) 再乘 \(e^{-tL}\)”，这在数学上不等价；改完只做了协议检查，没有重跑数值检验，私有集 58 例精度不合格（含解析控制例）。

源码：[`solutions/gpt-6.1-sol-single/`](solutions/gpt-6.1-sol-single/)（提交 `exponential-v2-20261109T090735Z-9ccd95`）；未复制的二进制或大文件：solver（58,896 字节）。

### GPT-6.1 Sol（迭代）

C++ 单个 Chebyshev 展开同时计算两项（\(e^{-tL}b_0\) 与 \(t\varphi_1(-tL)b_1\)），谱上界用行和与 Collatz 界，抵消严重的输出用扩展精度重算。68 分钟正式提交；4 次全量公开集测试 57.5、59.3、58.7、59.5。私有集 \(S = 1.84\)，95% 区间 [1.48, 2.30]，与 v6.1 的单轮成绩（62.7）相当。

源码：[`solutions/gpt-6.1-sol-iterative/`](solutions/gpt-6.1-sol-iterative/)（提交 `exponential-v2-20261109T153537Z-142f3b`）；未复制的二进制或大文件：solver（57,208 字节）。

### Claude Opus 5.5（单轮）

C 实现的 Chebyshev 展开，长展开时用短 Lanczos 收紧谱上界，SELL-4 稀疏格式加 AVX2 内核，按实际输出大小自适应截断。62 分钟结束回合。私有集 \(S = 2.99\)，95% 区间 [2.05, 4.47]，是本题目前最快的提交。

源码：[`solutions/claude-opus-5-5-single/`](solutions/claude-opus-5-5-single/)（提交 `exponential-v2-20261109T125907Z-5a6038`）；未复制的二进制或大文件：expo_v1（51,360 字节）、expo_v3（51,432 字节）。

### Claude Opus 5.5（迭代）

把 \(\varphi_1\) 项写成增广矩阵的指数，一次 Chebyshev 递推同时得到两项；小连通分量直接对角化，几何图先做 RCM 重排。60 分钟内 10 次公开集测试（全量最高 63.7）后正式提交。私有集 \(S = 2.60\)，95% 区间 [1.88, 3.69]，与单轮尝试相当（区间重叠）。

源码：[`solutions/claude-opus-5-5-iterative/`](solutions/claude-opus-5-5-iterative/)（提交 `exponential-v2-20261109T175508Z-fdfad8`）；未复制的二进制或大文件：expo（42,600 字节）。

## 观察与说明

- 逐例最快的参照成员有 60/74 是按文献实现的 Chebyshev 方法，不是作者代码；如果这一实现本身不够强，参照组会被低估。是否替换为作者代码仍在讨论。
- sol 的单轮尝试 23 分钟就结束回合，最后改动源项处理后没有重新检验，私有集 58 例精度不合格；迭代设置下它做了 4 次公开集测试，成绩正常。

## 历史版本

v1（解释器中的单个 KIOPS 参考，已取代）与 v6.1 成绩见 `results/LEADERBOARD.md`。
