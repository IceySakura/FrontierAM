# 三维荧光反卷积：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

观测 \(y=\mathrm{clip}_{\ge0}\big(\mathrm{Poisson}(h*x)+\text{Gaussian}\big)\)，卷积为零延拓后裁剪到原网格（same 卷积）；体积 \(32^3\)，PSF 为 \(9^3\) 居中核（各向同性或各向异性，出题说明为可分离高斯），噪声参数未知。要求恢复非负的 \(x\)，指标是 nMSE 的 96 例平均，每例墙钟 1.25 秒。

理论上有两个视角：

1. **非参数逆问题**：\(x\) 属于 Sobolev / Besov 类。高斯 PSF 的傅里叶变换指数衰减（“超光滑”），问题严重不适定。
2. **稀疏或参数模型**：点、管、片三类结构由少量参数描述，对应超分辨理论与参数估计（Cramér–Rao 下界）。

## 2. 已知最好的结果（上界）

- **Richardson–Lucy = Poisson 似然的 EM**（Richardson 1972；Lucy 1974；Shepp, Vardi 1982）：似然单调上升并收敛（Vardi, Shepp, Kaufman 1985；Csiszár, Tusnády 1984），但有半收敛现象，需要早停或正则化（正则化理论见 Engl, Hanke, Neubauer 1996）。加速 RL 见 Biggs, Andrews（1997），全变差正则化的 RL-TV 见 Dey 等（2006）。
- **Poisson–高斯噪声**：广义 Anscombe 变换稳定方差，Mäkitalo, Foi（2013）给出其最优逆变换。
- **非参数极小极大速率**：小波-小波子分解（WVD，Donoho 1995）；周期设定下的小波反卷积（Johnstone, Kerkyacharian, Picard, Raimondo 2004）；Poisson 逆问题（Antoniadis, Bigot 2006）。
- **稀疏超分辨**：满足最小间隔条件时，测度上的全变差范数最小化可精确恢复点源（Candès, Fernandez-Granda 2014）；正源不需要间隔条件也能稳定恢复（Morgenshtern, Candès 2016）；单分子定位精度有 Cramér–Rao 下界（Ober, Ram, Ward 2004）。
- **经验 SOTA**：学习型复原，如 CARE（Weigert 等 2018）。

## 3. 下界与不可能性

- **超光滑卷积只有对数速率**：误差分布为超光滑（如高斯）时，\(s\) 阶光滑类上反卷积的最优收敛速率只有 \((\log n)^{-s/2}\) 量级（Fan 1991 ✓，密度反卷积设定；同类结论适用于高斯模糊的图像反卷积）。非参数类上不可能有快的速率。
- **超分辨极限**：稀疏约束下的超分辨分析始于 Donoho（1992）；Demanet, Nguyen（2015）给出可恢复极限，误差随超分辨因子按 \(\mathrm{SRF}^{2s-1}\) 增长（\(s\) 为簇内尖峰数），在其模型中上下界匹配。
- **Cramér–Rao 下界**：参数模型下任何无偏估计的方差下界。

**是否紧**：非参数类与稀疏点源类的速率在各自模型下已知（阶）；本题“点 + 管 + 片、Poisson–高斯、噪声参数未知、1.25 秒”的复合设定没有统一的最优性结论。

## 4. 开放问题：什么样的结果算贡献

- 有连续结构先验（管、片）的三维超分辨理论；现有理论主要针对点源。
- 噪声参数未知时的自适应估计及其风险界。
- 有限计算预算下的统计-计算权衡：1.25 秒内的最优迭代数与停止准则。

## 5. 与本题参照、已有成绩的关系

- 弱参照 RL20（20 次 Richardson–Lucy），\(U=0.2199\)；强参照 SHB10（灵敏度修正加加速，10 次更新），\(W=0.1611\)。
- 已有成绩很高：GPT-6.2 Sol 单轮 nMSE 0.0126，按 \(100-50E/W\) 得 96.1 分；GPT 系列都在 93–96 分。`results/INSIGHTS.md` 指出顶端接近饱和：Sol 识别高斯点、有限长高斯管和平滑高斯片三类基元，在前向模型下做 Poisson 加高斯读出噪声的似然拟合，题面公开了结构类型，属于正当先验。已建议出题人改用更丰富的体模。

## 6. 理论赛道建议

**适配度：中。** 理论存在，但要连接到本题的复合设定需要大量假设，且顶端分数已接近饱和。若进入理论赛道，题面需明确信号类（如 \(K\) 个点加管段的参数族、最小间隔）与噪声模型。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | RL / EM 的似然单调性与不动点收敛；PSF 可分离时 FFT 实现的复杂度 |
| 增量 | 在声明的参数族（点 / 管 / 片）上估计器的一致性，以及接近 Cramér–Rao 下界的效率；早停准则的风险界 |
| 前沿 | 三维连续结构的超分辨恢复保证 |

**可自动检验的部分**：评测端有干净体积，声明的误差界只要可计算就可以逐例核对。

**形式化可行性**：EM 单调性（Jensen 不等式）可以在 Lean 4 中形式化；超分辨和极小极大结论超出现状。

## 参考文献

1. W. H. Richardson, Bayesian-based iterative method of image restoration, J. Opt. Soc. Am. 62 (1972) 55–59。
2. L. B. Lucy, An iterative technique for the rectification of observed distributions, Astron. J. 79 (1974) 745–754。
3. L. A. Shepp, Y. Vardi, Maximum likelihood reconstruction for emission tomography, IEEE Trans. Med. Imaging 1 (1982) 113–122。
4. Y. Vardi, L. A. Shepp, L. Kaufman, A statistical model for positron emission tomography, J. Amer. Statist. Assoc. 80 (1985) 8–37。
5. I. Csiszár, G. Tusnády, Information geometry and alternating minimization procedures, Statistics & Decisions, Suppl. 1 (1984) 205–237。
6. H. W. Engl, M. Hanke, A. Neubauer, Regularization of Inverse Problems, Kluwer, 1996。
7. D. S. C. Biggs, M. Andrews, Acceleration of iterative image restoration algorithms, Appl. Opt. 36 (1997) 1766–1775。
8. N. Dey 等, Richardson–Lucy algorithm with total variation regularization for 3D confocal microscope deconvolution, Microsc. Res. Tech. 69 (2006) 260–266。
9. M. Mäkitalo, A. Foi, Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise, IEEE Trans. Image Process. 22 (2013) 91–103。
10. D. L. Donoho, Nonlinear solution of linear inverse problems by wavelet–vaguelette decomposition, Appl. Comput. Harmon. Anal. 2 (1995) 101–126。
11. I. M. Johnstone, G. Kerkyacharian, D. Picard, M. Raimondo, Wavelet deconvolution in a periodic setting, J. R. Stat. Soc. B 66 (2004) 547–573。
12. A. Antoniadis, J. Bigot, Poisson inverse problems, Ann. Statist. 34 (2006) 2132–2158。
13. ✓ J. Fan, On the optimal rates of convergence for nonparametric deconvolution problems, Ann. Statist. 19 (1991) 1257–1272. <https://doi.org/10.1214/aos/1176348248>。超光滑误差下的对数速率。
14. D. L. Donoho, Superresolution via sparsity constraints, SIAM J. Math. Anal. 23 (1992) 1309–1331。
15. E. J. Candès, C. Fernandez-Granda, Towards a mathematical theory of super-resolution, Comm. Pure Appl. Math. 67 (2014) 906–956。
16. V. I. Morgenshtern, E. J. Candès, Super-resolution of positive sources: the discrete setup, SIAM J. Imaging Sci. 9 (2016) 412–444。
17. L. Demanet, N. Nguyen, The recoverability limit for superresolution via sparsity, arXiv:1502.01385（2015）。
18. R. J. Ober, S. Ram, E. S. Ward, Localization accuracy in single-molecule microscopy, Biophys. J. 86 (2004) 1185–1200。
19. M. Weigert 等, Content-aware image restoration: pushing the limits of fluorescence microscopy, Nat. Methods 15 (2018) 1090–1097。
