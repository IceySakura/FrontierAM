# 从一条采样轨迹估计长期协方差：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

由一条（近似）平稳链 \(x_1,\dots,x_M\)（\(d=4\)–6，\(M\in[2500,80000]\)）估计长期协方差 \(\Sigma=\lim_n n\,\mathrm{Cov}(\bar X_n)=\sum_{k\in\mathbb Z}\Gamma_k=2\pi f(0)\)，\(f\) 为谱密度矩阵。误差是仿射不变的 \(e=\|S^{-1/2}(H-S)S^{-1/2}\|_F\)，输入坐标还可能经过可逆线性变换和平移。六种情形：线性动态（VAR 型）、有限状态链、贝叶斯回归在三种采样器下的轨迹。

评测对每档 1000 条轨迹取平均，近似的是期望误差（风险），这与统计理论中的风险概念一致。模拟情形的 \(S\) 可由生成机制精确算出，真实数据情形的 \(S\) 来自独立长链。

## 2. 已知最好的结果（上界）

- **批均值与谱方差估计**：强一致性及其条件（Jones, Haran, Caffo, Neath 2006；Flegal, Jones 2010），多元版本（Vats, Flegal, Jones 2018、2019）。批均值或 Bartlett 窗的最优均方误差阶为 \(n^{-2/3}\)（批长约 \(n^{1/3}\)）；二阶核在谱密度二阶光滑时可达 \(n^{-4/5}\)，Andrews（1991）证明二次谱（QS）核在一类正定核中均方误差最优。
- **减偏与预白化**：平顶窗（Politis, Romano 1995）在自协方差快速衰减时接近参数速率；lugsail 窗（Vats, Flegal 2022）；VAR 预白化（Andrews, Monahan 1992）。本题 50 分参照就是拟合 VAR(1) 后的长期协方差。
- **可逆链的形状约束**：可逆链的自协方差是 \([-1,1]\) 上某个正测度的矩序列。据此有初始正序列、单调序列、凸序列估计（Geyer 1992），以及矩最小二乘估计（Berg, Song 2023 ✓）：几何遍历的可逆链上，它对自协方差序列（\(\ell_2\) 意义）和渐近方差都强一致。

## 3. 下界与不可能性

- **光滑谱类上的极小极大速率**：Sobolev 型谱密度类上为 \(n^{-2s/(2s+1)}\)，平滑周期图可以达到。
- **不可能性**：Pötscher（2002 ✓）证明，在常用的数据生成过程集合上，估计零频谱密度的极小极大风险对任何样本量都是无穷大，置信集近乎没有信息量。也就是说，不对谱的衰减或遍历常数作定量约束，“最优估计”无从谈起。
- **是否紧**：Sobolev 类上上下界阶匹配；MCMC 特有的类（几何遍历加可逆）上，速率与常数的最优性没有完整结论（Berg–Song 给出一致性，没有给出匹配下界）。

## 4. 开放问题：什么样的结果算贡献

- 可逆或几何遍历链类上（例如谱隙下界 \(\gamma\)），多元长期协方差在仿射不变 Frobenius 损失下的非渐近风险界与匹配下界。
- 跨采样器的自适应：同一个估计器在 VAR、有限状态链、HMC/NUTS、Gibbs 上都接近最优，即 oracle 不等式。
- 有限样本（\(M\) 从 2500 起）下 VAR 阶选择加预白化的风险保证。

## 5. 与本题参照、已有成绩的关系

- 弱参照：重叠批均值（OBM，R 包 mcmcse），\(L=0.3503\)；强参照：VAR(1) 长期协方差，\(L=0.2290\)。
- v6.1 GPT-6.1 Sol 61.5 分（\(L=0.1769\)），优势集中在随机游走 Metropolis 与有限状态链，在 NUTS、Pólya–Gamma 上略差。v6.3：GPT-6.2 Sol 63.9 / 64.0，Hunyuan 4 47.9 / 59.3（单轮 / 迭代）。`results/INSIGHTS.md` 记录过一处题面误读：交付的是 \(H\) 而不是 \(H/M\)，单轮把估计除以轨迹长度属于模型失误。

## 6. 理论赛道建议

**适配度：高。** 统计理论完整（一致性、速率、极小极大、不可能性都有），评测本身就是蒙特卡洛风险估计，可以直接检验风险界声明。

**题面需补的数学前提**：各情形的数学假设，包括几何遍历常数或谱隙的范围、哪些情形满足可逆性（需出题人声明），以及仿射变换的取法。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所用批均值、谱方差或 VAR 估计在几何遍历下的一致性与均方误差阶 |
| 增量 | 声明类（谱隙 \(\ge\gamma\)、可逆）上的非渐近风险界，并通过白化把仿射不变损失化归到单位协方差情形；VAR 预白化加非参数修正的 oracle 不等式 |
| 前沿 | 多元可逆链类上匹配的极小极大下界，以及达到它的估计器 |

**可自动检验的部分**：模拟情形的 \(S\) 精确已知，每档 1000 条轨迹给出风险的蒙特卡洛估计，可直接核对“风险 \(\le\) 声明界”。

**形式化可行性**：Mathlib 有测度论概率与大数定律，但马尔可夫链 CLT、谱表示等缺失。可形式化的局限于代数部分，例如损失的仿射不变性、批均值估计的期望表达式。

## 参考文献

1. G. L. Jones, M. Haran, B. S. Caffo, R. Neath, Fixed-width output analysis for Markov chain Monte Carlo, J. Amer. Statist. Assoc. 101 (2006) 1537–1547。
2. J. M. Flegal, G. L. Jones, Batch means and spectral variance estimators in Markov chain Monte Carlo, Ann. Statist. 38 (2010) 1034–1070。
3. D. Vats, J. M. Flegal, G. L. Jones, Strong consistency of multivariate spectral variance estimators in Markov chain Monte Carlo, Bernoulli 24 (2018) 1860–1909；Multivariate output analysis for Markov chain Monte Carlo, Biometrika 106 (2019) 321–337。
4. D. W. K. Andrews, Heteroskedasticity and autocorrelation consistent covariance matrix estimation, Econometrica 59 (1991) 817–858。
5. D. W. K. Andrews, J. C. Monahan, An improved heteroskedasticity and autocorrelation consistent covariance matrix estimator, Econometrica 60 (1992) 953–966。VAR 预白化。
6. W. K. Newey, K. D. West, A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix, Econometrica 55 (1987) 703–708。
7. D. N. Politis, J. P. Romano, Bias-corrected nonparametric spectral estimation, J. Time Ser. Anal. 16 (1995) 67–103。平顶窗。
8. D. Vats, J. M. Flegal, Lugsail lag windows for estimating time-average covariance matrices, Biometrika 109 (2022) 735–750。
9. C. J. Geyer, Practical Markov chain Monte Carlo, Statist. Sci. 7 (1992) 473–483。
10. ✓ S. Berg, H. Song, Efficient shape-constrained inference for the autocovariance sequence from a reversible Markov chain, Ann. Statist. 51 (2023) 2440–2470. <https://doi.org/10.1214/23-AOS2335>。
11. ✓ B. M. Pötscher, Lower risk bounds and properties of confidence sets for ill-posed estimation problems with applications to spectral density and persistence estimation, unit roots, and estimation of long memory parameters, Econometrica 70 (2002) 1035–1065。零频谱密度的极小极大风险无穷。
