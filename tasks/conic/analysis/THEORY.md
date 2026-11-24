# 凸锥优化：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

求解带二次目标的锥规划 \(\min\tfrac12x^\top Px+q^\top x\ \text{s.t.}\ Ax+s=b,\ s\in\mathcal K\)，\(\mathcal K\) 是零锥、非负锥、二阶锥、指数锥、半正定锥的乘积。必须返回最优解，或原始/对偶不可行证书；精度 \(\varepsilon=10^{-8}\)（归一化残差和对偶间隙）。650 个隐藏实例，逐例时限为参考（Clarabel 0.11.1）最大耗时的 1.2 倍，最低 1 秒。分数由成功率 \(p\) 决定，在 \(0.9p_B\) 处有断崖。

计分指标与理论的最坏情形迭代复杂度相距很远。决定成败的是稀疏 KKT 线性代数、启动开销（1 秒下限的实例上，Python 启动加导入就要 0.2–0.7 秒）和数值稳健性（病态、弱不可行、无严格可行点）。

## 2. 已知最好的结果（上界）

- **自和谐障碍内点法**：用 \(\nu\)-自和谐障碍，\(O(\sqrt\nu\log(\nu/\epsilon))\) 次牛顿步（Nesterov, Nemirovski 1994）。对称锥上的原始-对偶法用 Nesterov–Todd 缩放（1997、1998）。齐次自对偶嵌入不需要初始可行点，同时给出最优解或不可行证书（Ye, Todd, Mizuno 1994）。
- **非对称锥（指数锥）**：Nesterov（2012）、Skajaa, Ye（2015）的齐次内点法，Dahl, Andersen（2022）在 MOSEK 中的指数锥原始-对偶算法。Clarabel（Goulart, Chen，arXiv 2024、Math. Program. Comput. 2026 ✓）把齐次嵌入专门化到二次目标，不需要 epigraph 改写。
- **障碍参数**：任意 \(n\) 维闭凸锥都有 \(\nu\le n\) 的障碍（Lee, Yue 2021：universal barrier 是 \(n\)-自和谐的；Bubeck, Eldan：熵障碍 \(\nu\le(1+o(1))n\)）。非负象限上 \(\nu=n\) 不可能再小，所以 \(\sqrt n\) 是障碍框架内在的量。
- **LP 总复杂度的理论突破**（与实践脱节）：\(\tilde O(n^\omega\log(1/\delta))\)（Cohen, Lee, Song 2019 / JACM 2021；van den Brand 2020 确定性版本；Jiang, Song, Weinstein, Zhang 2021 进一步改进）；\(\tilde O(\sqrt{\mathrm{rank}})\) 次迭代（Lee, Sidford 2014）；SDP 见 Jiang, Kathuria, Lee, Padmanabhan, Song（2020）。
- **一阶方法**：SCS（O'Donoghue 等 2016，ADMM 加齐次嵌入）、PDLP（重启 PDHG）。适合大规模中等精度；\(10^{-8}\) 精度下通常不如内点法。

## 3. 下界与不可能性

- **迭代下界**：长步原始-对偶路径跟踪法存在需要 \(\Omega(n^{1/3})\) 次迭代才能把对偶间隙降一个常数倍的 LP（Todd 1994；Todd, Ye 1996 ✓）。这些实例的条件数是指数级的；对条件良好的 LP 能否证明类似下界仍未知。
- **Klee–Minty 型**：中心路径可以被迫绕行，使内点法的迭代数与 \(\sqrt n\) 上界同量级（Deza, Nematollahi, Terlaky 2008）。
- **强多项式不可能**：对数障碍内点法不是强多项式的（Allamigeon, Benchimol, Gaubert, Joswig 2018）；更进一步，任何自和谐障碍内点法都不是强多项式的：存在 \(n\) 维、\(2n\) 个约束的 LP，需要 \(\Omega(2^n)\) 次迭代（Allamigeon, Gaubert, Vandame，STOC 2022 ✓）。LP 是否存在强多项式算法仍是 Smale 第 9 问题。
- **一阶方法**：重启 PDHG 的线性速率在 span-respecting 原始-对偶方法类中最优（Applegate 等 2023）；双线性鞍点问题的一阶下界见 Ouyang, Xu（2021）。

**是否紧**：障碍框架内，\(\sqrt\nu\) 迭代上界与 \(\nu\) 的下界一致；“所有算法”意义下，LP 的最优复杂度（与矩阵乘法指数 \(\omega\) 的关系）与强多项式性都开放。

## 4. 开放问题：什么样的结果算贡献

- 强多项式 LP（Smale 第 9 问题）——不应期待在一次做题中解决。
- 实践中的 Mehrotra 型预测-校正法普遍没有多项式保证；给实践算法加上可证的保护机制并保留速度，是有意义的增量方向。
- 有限精度下的不可行检测：弱不可行、无严格可行点时需要 facial reduction（Permenter, Friberg, Andersen 2017）。
- 混合锥上依赖条件数（Renegar 条件数）的复杂度刻画。

## 5. 与本题参照、已有成绩的关系

- 参考 Clarabel 0.11.1 在本机解出 513/650（\(p_B=0.789\)），及格线 \(0.9p_B=0.710\)。v6.3 只有 Opus 迭代过线（\(p=0.889\)，73.7 分），其余模型 \(p\) 在 0.30–0.67 之间。
- v6.2 的分析（`results/LEADERBOARD.md`）：Opus 写原生 C++ 内点法，绕开了 1 秒下限的启动开销；Sol 用 Python + SciPy，被启动开销卡住。本题的胜负主要在工程，不在理论。

## 6. 理论赛道建议

**适配度：中。** 理论成熟，但这里“超越 SOTA 理论”等于解决长期开放问题（强多项式性、\(\omega\) 指数），一次做题不现实；能有效考核的是“正确性理论”，即证书和收敛保证。

**题面需补的数学前提**：无需改动问题本身；如果要考复杂度声明，需要求诊断输出中报告迭代数与 KKT 求解次数。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所实现内点法在齐次嵌入下的 \(O(\sqrt\nu\log(1/\epsilon))\) 复杂度；证书有效性（弱对偶、Farkas 型择一定理） |
| 增量 | 实践启发式（预测-校正、步长规则、正则化 KKT 求解）在保护机制下的全局收敛；有限精度下证书必定通过评测器归一化判据的充分条件 |
| 前沿 | 混合锥上依赖条件数的复杂度；强多项式性（不现实） |

**可自动检验的部分**：证书已由评测器独立检查；迭代数日志可与声明的迭代界对照。

**形式化可行性**：LP 的 Farkas 引理和强对偶已有 Lean 4 形式化（Dvorak, Kolmogorov，Annals of Formalized Mathematics 2026 ✓）。锥版本的弱对偶与证书有效性是可行的小目标；内点法复杂度分析的形式化工作量很大。

## 参考文献

1. Yu. Nesterov, A. Nemirovskii, Interior-Point Polynomial Algorithms in Convex Programming, SIAM, 1994。
2. Yu. Nesterov, M. J. Todd, Self-scaled barriers and interior-point methods for convex programming, Math. Oper. Res. 22 (1997) 1–42；Primal-dual interior-point methods for self-scaled cones, SIAM J. Optim. 8 (1998) 324–364。
3. Y. Ye, M. J. Todd, S. Mizuno, An \(O(\sqrt nL)\)-iteration homogeneous and self-dual linear programming algorithm, Math. Oper. Res. 19 (1994) 53–67。
4. Yu. Nesterov, Towards non-symmetric conic optimization, Optim. Methods Softw. 27 (2012) 893–917。
5. A. Skajaa, Y. Ye, A homogeneous interior-point algorithm for nonsymmetric convex conic optimization, Math. Program. 150 (2015) 391–422。
6. J. Dahl, E. D. Andersen, A primal-dual interior-point algorithm for nonsymmetric exponential-cone optimization, Math. Program. 194 (2022) 341–370。
7. ✓ P. J. Goulart, Y. Chen, Clarabel: an interior-point solver for conic programs with quadratic objectives, arXiv:2405.12762（2024）；Math. Program. Comput.（2026）. <https://arxiv.org/abs/2405.12762>。本题参考实现。
8. Y. T. Lee, M.-C. Yue, Universal barrier is \(n\)-self-concordant, Math. Oper. Res. 46 (2021) 1129–1148。
9. S. Bubeck, R. Eldan, The entropic barrier: exponential families, log-concave geometry, and self-concordance, Math. Oper. Res. 44 (2019) 264–276。
10. M. B. Cohen, Y. T. Lee, Z. Song, Solving linear programs in the current matrix multiplication time, J. ACM 68 (2021) 3。
11. J. van den Brand, A deterministic linear program solver in current matrix multiplication time, SODA 2020。
12. S. Jiang, Z. Song, O. Weinstein, H. Zhang, Faster dynamic matrix inverse for faster LPs, STOC 2021。
13. H. Jiang, T. Kathuria, Y. T. Lee, S. Padmanabhan, Z. Song, A faster interior point method for semidefinite programming, FOCS 2020。
14. Y. T. Lee, A. Sidford, Path finding methods for linear programming: solving linear programs in \(\tilde O(\sqrt{\mathrm{rank}})\) iterations and faster algorithms for maximum flow, FOCS 2014。
15. ✓ M. J. Todd, Y. Ye, A lower bound on the number of iterations of long-step primal-dual linear programming algorithms, Ann. Oper. Res. 62 (1996) 233–252. <https://doi.org/10.1007/BF02206818>。
16. A. Deza, E. Nematollahi, T. Terlaky, How good are interior point methods? Klee–Minty cubes tighten iteration-complexity bounds, Math. Program. 113 (2008) 1–14。
17. X. Allamigeon, P. Benchimol, S. Gaubert, M. Joswig, Log-barrier interior point methods are not strongly polynomial, SIAM J. Appl. Algebra Geom. 2 (2018) 140–178。
18. ✓ X. Allamigeon, S. Gaubert, N. Vandame, No self-concordant barrier interior point method is strongly polynomial, STOC 2022, 515–528. <https://doi.org/10.1145/3519935.3519997>。
19. S. Smale, Mathematical problems for the next century, Math. Intelligencer 20 (1998) 7–15。第 9 问题。
20. B. O'Donoghue, E. Chu, N. Parikh, S. Boyd, Conic optimization via operator splitting and homogeneous self-dual embedding, J. Optim. Theory Appl. 169 (2016) 1042–1068。
21. F. Permenter, H. A. Friberg, E. D. Andersen, Solving conic optimization problems via self-dual embedding and facial reduction: a unified approach, SIAM J. Optim. 27 (2017) 1257–1282。
22. J. Renegar, Linear programming, complexity theory and elementary functional analysis, Math. Program. 70 (1995) 279–351。
23. ✓ M. Dvorak, V. Kolmogorov, Duality theory in linear optimization and its extensions — formally verified, Annals of Formalized Mathematics 2 (2026). <https://arxiv.org/abs/2409.08119>。Lean 4 中的 Farkas 引理与 LP 强对偶。
