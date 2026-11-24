# 时序正定线性系统：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

在线求解 50 个对称正定系统 \(H_tx_t=b_t\)（\(d=16384\)），矩阵只能通过 HVP（Hessian-vector product，每次最多 64 列批量）访问，给定谱界 \(\mu I\preceq H_t\preceq LI\)，阶段变化有标记。单步损失为相对能量范数误差的平方 \(\ell_t=\|\hat x_t-x_t^\star\|_{H_t}^2/\|x_t^\star\|_{H_t}^2\)。资源是单核、每条序列 30 秒、峰值 RSS 256 MiB。

自然的代价模型是 HVP 次数加向量运算。内存约束是实质性的：256 MiB 约等于 2000 个长度 16384 的 float64 向量，可保留的历史子空间维数有上限（几百）。这是一个“在线、预算受限、带记忆约束”的序列求解问题。

## 2. 已知最好的结果（上界）

- **单个系统**：CG 满足 \(\|e_k\|_A\le2\big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^k\|e_0\|_A\)，且每步在 Krylov 子空间内使 \(A\) 范数误差最小。谱聚簇时出现超线性收敛（Beckermann, Kuijlaars 2001）。
- **利用历史的手段**：
  - 投影热启动：用历史解张成的子空间做 Galerkin 投影给初值（Fischer 1998）。
  - Krylov 子空间回收与紧缩：deflated CG（Saad, Yeung, Erhel, Guyomarc'h 2000）、GCRO-DR（Parks 等 2006），综述见 Soodhalter, de Sturler, Kilmer（2020）。回收子空间若近似 \(k\) 个最小特征向量，有效条件数从 \(\lambda_n/\lambda_1\) 降到约 \(\lambda_n/\lambda_{k+1}\)。
  - 有限记忆拟牛顿预条件器：Morales, Nocedal（2000）；Gratton, Sartenaer, Tshimanga（2011）给出这类 LMP 预条件器的谱性质。本题 50 分参照 L-BFGS-PCG 属于这一类。
  - 随机 Nyström 预条件：秩 \(\ell\) 的 Nyström 近似使 PCG 迭代数由有效维数决定，而不是由 \(\kappa\) 决定（Frangella, Tropp, Udell 2023），适合“低有效秩 + 正则项”的结构，与本题“低有效秩”场景对应。
  - 在实例序列上学习求解器参数：Khodak, Chow, Balcan, Talwalkar（ICLR 2024 ✓）对 SOR 松弛参数做 bandit 选择，只用迭代数作反馈，证明总迭代数趋于最优固定参数；有结构信息时，上下文 bandit 趋于实例最优。这是“跨实例序列的求解器”目前少见的端到端理论保证。

## 3. 下界与不可能性

- **一阶 / Krylov 下界**：对强凸二次函数，任何 span-respecting 方法（输出落在梯度张成的空间里，包括 GD、CG、加速梯度）\(k\) 步后误差不小于 \(\big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^k\) 量级（Nemirovski, Yudin 1983；Nesterov 2004 §2.1.4）。冷启动下达到精度 \(\epsilon\) 需要 \(\Omega(\sqrt\kappa\log(1/\epsilon))\) 次 HVP（维数足够大时），CG 达到这一阶，单个系统意义下紧。
- **序列**：没有一般的下界理论。可利用的信息完全取决于漂移模型：对相邻系统不作任何假设时，历史无用，下界退化为单系统下界。假设相对扰动 \(\|H_t^{-1/2}(H_{t+1}-H_t)H_t^{-1/2}\|\le\delta\)、\(\|b_{t+1}-b_t\|\) 有界时，“热启动 + 回收”有上界可证，但匹配下界在文献中未见。

**是否紧**：单系统紧；序列设定开放。

## 4. 开放问题：什么样的结果算贡献

- 在明确的漂移或低秩更新模型下，序列求解的 HVP 复杂度上下界：总 HVP 数如何依赖序列长度、\(\delta\)、有效秩和记忆预算。
- 回收子空间在线选择的遗憾分析，类似 Khodak 等对 SOR 参数所做的工作，但面向 Krylov 回收和预条件器。
- 阶段突变（已有标记）下的自适应重启与保证。
- 序列内时间预算的跨步分配（本题允许自由分配）：可表述为在线资源分配问题，并给出遗憾界。

## 5. 与本题参照、已有成绩的关系

- 0 分锚点是冷启动 CG，50 分锚点是 L-BFGS-PCG，出题人明确不声称它是全局 SOTA。`results/INSIGHTS.md` 记录过：50 分锚点没有利用历史解（子空间回收、Galerkin 热启动），v2 中模型用历史子空间投影加 CG 做到 99.7 / 99.9，因此改版到 v3 / v4 增加难度。
- v4 已有成绩（私有集）：GPT-6.1 Sol 62.6 / 63.0、GPT-6 Astra 60.2 / 63.2、Hunyuan 4 52.6 / 56.1（单轮 / 迭代）。

## 6. 理论赛道建议

**适配度：高。** 单系统理论完备且紧，序列理论正处于前沿；HVP 计数和逐步误差都能在评测端精确记录，便于核对理论声明。

**题面需补的数学前提**：漂移模型的数学描述（例如 \(H_t=\bar H+U_t\Lambda_tU_t^\top+\) 漂移项，或相对扰动界 \(\delta\)、阶段内平滑性），否则序列定理无从谈起；同时要求以 HVP 计数为代价单位。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 给定预条件器下 PCG 的迭代界（由谱或有效条件数决定） |
| 增量 | \(\delta\)-漂移模型下回收或投影热启动的每步 HVP 上界（依赖 \(\delta\)、回收子空间与不变子空间的夹角，用 Davis–Kahan 型扰动界）；记忆预算下的子空间截断误差 |
| 前沿 | 漂移模型下任何算法所需 HVP 数的匹配下界；在线学习式的遗憾界 |

**可自动检验的部分**：评测端可以记录每步 HVP 数，并计算 \(\ell_t\)（参考解已知），直接对照声明的界。

**形式化可行性**：CG 的 \(A\) 范数最优性和 Chebyshev 界可以在 Lean 4 中证明，但要先建 Krylov 空间与 Chebyshev 极值性质，工作量不小；Davis–Kahan 型扰动界更难。

## 参考文献

1. B. Beckermann, A. B. J. Kuijlaars, Superlinear convergence of conjugate gradients, SIAM J. Numer. Anal. 39 (2001) 300–329。
2. A. S. Nemirovski, D. B. Yudin, Problem Complexity and Method Efficiency in Optimization, Wiley, 1983。
3. Yu. Nesterov, Introductory Lectures on Convex Optimization, Kluwer, 2004。§2.1.4 二次函数下界。
4. P. F. Fischer, Projection techniques for iterative solution of \(Ax=b\) with successive right-hand sides, Comput. Methods Appl. Mech. Engrg. 163 (1998) 193–204。
5. Y. Saad, M. Yeung, J. Erhel, F. Guyomarc'h, A deflated version of the conjugate gradient algorithm, SIAM J. Sci. Comput. 21 (2000) 1909–1926。
6. M. L. Parks, E. de Sturler, G. Mackey, D. D. Johnson, S. Maiti, Recycling Krylov subspaces for sequences of linear systems, SIAM J. Sci. Comput. 28 (2006) 1651–1674。
7. K. M. Soodhalter, E. de Sturler, M. E. Kilmer, A survey of subspace recycling iterative methods, GAMM-Mitt. 43 (2020) e202000016。
8. J. L. Morales, J. Nocedal, Automatic preconditioning by limited memory quasi-Newton updating, SIAM J. Optim. 10 (2000) 1079–1096。
9. S. Gratton, A. Sartenaer, J. Tshimanga, On a class of limited memory preconditioners for large scale linear systems with multiple right-hand sides, SIAM J. Optim. 21 (2011) 912–935。
10. Z. Frangella, J. A. Tropp, M. Udell, Randomized Nyström preconditioning, SIAM J. Matrix Anal. Appl. 44 (2023) 718–752。
11. ✓ M. Khodak, E. Chow, M.-F. Balcan, A. Talwalkar, Learning to relax: setting solver parameters across a sequence of linear system instances, ICLR 2024. <https://arxiv.org/abs/2310.02246>。
