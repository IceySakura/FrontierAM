# 稀疏系统的指数传播：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

计算 \(y=e^{-tL}b_0+t\,\varphi_1(-tL)b_1\)，\(L\) 为稀疏对称半正定（可奇异），等价于 \(u'=-Lu+b_1\)、\(u(0)=b_0\) 在 \(s=t\) 处的解。记谱区间为 \([0,\beta]\)，\(\beta=t\|L\|_2\le t\|L\|_\infty\le10^5\)。精度是全向量混合绝对/相对 \(\infty\) 范数，rtol 分 \(10^{-4}/10^{-7}/10^{-10}\) 三档。one_shot 会话 1 个右端，reuse 会话 16 个右端依次揭示。

计分是单核墙钟相对参照组逐例最快者的时间比，任何一例精度不合格整份 0 分。理论上最自然的代价模型是“矩阵向量乘次数 + 稀疏线性求解次数 + 向量运算”；墙钟另外取决于内存带宽、预处理和格式转换，理论无法直接覆盖。

## 2. 已知最好的结果（上界）

- **多项式方法（只用 matvec）**：在 \([0,b]\) 上把 \(e^{-x}\) 逼近到 \(\delta\) 所需的最低多项式次数为
  \[
  d_{b;\delta}(e^{-x})=\Theta\Big(\max\Big\{\sqrt{b\log(1/\delta)},\ \frac{\log(1/\delta)}{\log\big(b^{-1}\log(1/\delta)\big)}\Big\}\Big)
  \]
  （Aggarwal, Alman 2022 ✓ 给出精确渐近并确定大多数参数区间的首项系数；此前 Sachdeva, Vishnoi 2014 ✓ 给出上界 \(O(\sqrt{\max\{b,\log(1/\delta)\}\log(1/\delta)})\) 和与 \(\delta\) 无关的下界 \(\Omega(\sqrt b)\)）。Chebyshev 展开（系数为修正 Bessel 函数）达到这一量级。
- **Lanczos / Krylov**：谱在 \([-4\rho,0]\) 时误差不超过 \(10e^{-m^2/(5\rho)}\)（\(\sqrt{4\rho}\le m\le2\rho\)），\(m\ge2\rho\) 后超线性下降（Hochbruck, Lubich 1997；早期结果见 Druskin, Knizhnerman 1989）。对一类有理函数，Lanczos-FA 的误差不超过“同步数最佳 Krylov 多项式逼近误差”的常数倍，常数与迭代数无关（Amsel, Chen, Greenbaum, Musco, Musco，NeurIPS 2024 ✓）。
- **有理方法（用线性求解）**：\(e^{-x}\) 在 \([0,\infty)\) 上最佳 \((n,n)\) 有理逼近的误差按 Halphen 常数几何衰减（约 \(9.28903^{-n}\)；Cody, Meinardus, Varga 1969 提出，Gonchar, Rakhmanov 1987 证明精确速率），所需极点数约为 \(\log(1/\delta)\)，与 \(\beta\) 无关。移位求逆 Krylov 的收敛与网格无关（Moret, Novati 2004；van den Eshof, Hochbruck 2006；综述 Güttel 2013）。对称对角占优矩阵（含图拉普拉斯）可借近线性时间求解器在约 \(\tilde O(m\log(1/\delta))\) 时间算出 \(e^{-A}v\)，与 \(\|A\|\) 无关（Orecchia, Sachdeva, Vishnoi，STOC 2012 ✓）。
- **φ 函数**：增广矩阵技巧把 \(\varphi_1\) 化为更大矩阵的指数。Al-Mohy, Higham（2011）的 expmv 用截断 Taylor 加缩放，按后向误差分析选参数；另有 Niesen, Wright（2012）的 phipm、Gaudreault 等（2018）的 KIOPS。
- **多个右端**：16 个右端依次揭示，不能用 block Krylov；能在请求之间复用的只有与 \(L\) 有关的量（谱区间估计、Chebyshev 系数、分解或预条件器、各移位的因子分解）。多项式方法每个请求都要付约 \(\sqrt\beta\) 次 matvec，有理方法可以把稀疏分解的代价摊到 16 个请求上。这是“会话复用”在理论上的核心权衡。

## 3. 下界与不可能性

- 只用 matvec 的算法输出都落在 Krylov 空间里，即 \(p(L)v\) 的形式。所以在谱填满 \([0,\beta]\) 的最坏情形下，达到 \(\delta\) 精度所需的 matvec 次数不少于上述最低多项式次数；\(\beta\gg\log(1/\delta)\) 时为 \(\Omega(\sqrt{\beta\log(1/\delta)})\)。这由逼近论下界直接推出，且被 Chebyshev / Lanczos 达到，在阶意义下紧，Aggarwal–Alman 甚至给出了首项常数。
- 有理方法所需的极点数受 Halphen 速率约束：要更少的线性求解达到同样精度是不可能的。
- 但总时间还取决于线性求解的代价。一般稀疏对称正定矩阵没有近线性时间求解器（图拉普拉斯有）。题面明确要求不假设 \(L\) 是拉普拉斯或对角占优，所以 Orecchia–Sachdeva–Vishnoi 的结果不能直接套用；有限元矩阵可用不完全 Cholesky 或多重网格，代价依赖结构。

**是否紧**：matvec 模型下紧（含首项常数）；求解模型下极点数紧；墙钟没有理论最优性。

## 4. 开放问题：什么样的结果算贡献

- 可证明的后验误差界与停止准则，且针对本题的混合 \(\infty\) 范数准则；现有 Krylov 后验估计多为启发式，或只在 2 范数下成立。
- 有限精度下 Lanczos 的 Chebyshev 型误差界在 \(\infty\) 范数下的完整版本。
- 16 个依次揭示的右端：在线摊销模型下“多项式还是有理”的切换阈值，以及 matvec 与求解组合的最优策略。
- \(\varphi_1\) 项与奇异 \(L\)（零特征值）的统一误差界。

## 5. 与本题参照、已有成绩的关系

- 参照组是 7 个已发表方法的逐例最快者，其中 Chebyshev 在 74 例中赢 60 例，与理论一致：对称谱、谱区间已知时，Chebyshev 达到最优次数且常数小。有理（移位求逆）Krylov 成员因在一个公开用例上精度余量不足未入选。
- 已有成绩：GPT-6.2 Sol 迭代 \(R=0.487\)，按 \(100/(1+R)\) 得 67.2；GPT-6.1 Sol 迭代 \(R=0.567\)，63.8；做法是 C++ 实现的 Chebyshev / Taylor / Krylov 组合。Opus 两格都因 1 例精度不合格记 0。

## 6. 理论赛道建议

**适配度：高。** matvec 复杂度的上下界清晰且紧，“复杂度声明”和“误差声明”都可以在评测时自动核对。

**题面需补的数学前提**：要求以 matvec 和线性求解次数为代价单位，并在诊断输出中报告；误差准则已有精确的数学形式（混合 \(\infty\) 范数）。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | Chebyshev / Lanczos 截断阶 \(m\) 的先验误差界（Bessel 系数尾界），matvec 次数 \(O(\sqrt{\beta\log(1/\delta)})\) |
| 增量 | 混合 \(\infty\) 范数准则、\(\varphi_1\) 项和奇异 \(L\) 下的严格误差界与可计算停止准则；会话内复用的摊销复杂度定理（多项式与有理的切换阈值） |
| 前沿 | 在有限精度下可证有效的后验误差估计；对特定结构族（图拉普拉斯加对角势）带常数的近线性时间算法 |

**可自动检验的部分**：在 stderr 诊断中记录 matvec 与求解次数，与复杂度定理对照；声称的先验误差界与参考解比对，一旦违反即判定理论有误。

**形式化可行性**：Mathlib 已有 Chebyshev 多项式、矩阵指数和 Hermitian 矩阵的谱定理，“\(\|e^{-L}v-p(L)v\|_2\le\max_{\lambda\in[0,\beta]}|e^{-\lambda}-p(\lambda)|\,\|v\|_2\)”这类引理可在 Lean 4 中证明。Bessel 系数尾界和 Aggarwal–Alman 下界的形式化工作量大。

## 参考文献

1. ✓ A. Aggarwal, J. Alman, Optimal-degree polynomial approximations for exponentials and Gaussian kernel density estimation, CCC 2022, LIPIcs 234, 22. <https://doi.org/10.4230/LIPIcs.CCC.2022.22>。\(d_{B;\delta}(e^{-x})\) 的精确渐近。
2. ✓ S. Sachdeva, N. K. Vishnoi, Faster algorithms via approximation theory, Found. Trends Theor. Comput. Sci. 9 (2014) 125–210. <https://doi.org/10.1561/0400000065>。定理 4.1（上界）、定理 5.3（\(\Omega(\sqrt b)\) 下界）。
3. M. Hochbruck, C. Lubich, On Krylov subspace approximations to the matrix exponential operator, SIAM J. Numer. Anal. 34 (1997) 1911–1925。
4. V. Druskin, L. Knizhnerman, Two polynomial methods of calculating functions of symmetric matrices, USSR Comput. Math. Math. Phys. 29 (1989) 112–121。
5. ✓ N. Amsel, T. Chen, A. Greenbaum, C. Musco, C. Musco, Nearly optimal approximation of matrix functions by the Lanczos method, NeurIPS 2024. <https://arxiv.org/abs/2303.03358>。
6. W. J. Cody, G. Meinardus, R. S. Varga, Chebyshev rational approximations to \(e^{-x}\) in \([0,+\infty)\) and applications to heat-conduction problems, J. Approx. Theory 2 (1969) 50–65。
7. A. A. Gonchar, E. A. Rakhmanov, Equilibrium distributions and degree of rational approximation of analytic functions, Mat. Sb. 134 (1987) 306–352。Halphen 常数速率。
8. I. Moret, P. Novati, RD-rational approximations of the matrix exponential, BIT 44 (2004) 595–615。
9. J. van den Eshof, M. Hochbruck, Preconditioning Lanczos approximations to the matrix exponential, SIAM J. Sci. Comput. 27 (2006) 1438–1457。
10. S. Güttel, Rational Krylov approximation of matrix functions: numerical methods and optimal pole selection, GAMM-Mitt. 36 (2013) 8–31。
11. ✓ L. Orecchia, S. Sachdeva, N. K. Vishnoi, Approximating the exponential, the Lanczos method and an \(\tilde O(m)\)-time spectral algorithm for balanced separator, STOC 2012, 1141–1160. <https://www.cs.yale.edu/homes/vishnoi/OSV.pdf>。
12. A. H. Al-Mohy, N. J. Higham, Computing the action of the matrix exponential, with an application to exponential integrators, SIAM J. Sci. Comput. 33 (2011) 488–511。
13. J. Niesen, W. M. Wright, Algorithm 919: a Krylov subspace algorithm for evaluating the φ-functions appearing in exponential integrators, ACM TOMS 38 (2012) 22。
14. S. Gaudreault, G. Rainwater, M. Tokman, KIOPS: a fast adaptive Krylov subspace solver for exponential integrators, J. Comput. Phys. 372 (2018) 236–255。
15. R. B. Sidje, Expokit: a software package for computing matrix exponentials, ACM TOMS 24 (1998) 130–156。
