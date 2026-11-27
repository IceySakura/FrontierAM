# 理论附录（v3.1，2026-11-25）

本附录随题面公开，规定理论结果所针对的对象：问题类、信息与代价模型、误差准则，并列出已知结果和开放问题。你的理论交付应当针对这里声明的类（或你明确写出的子类），并说明隐藏实例中哪些满足你的假设。

## 1. 问题

计算 \(I(f)=\mathbb E[f(Z)]=\int_{\mathbb R^d}f(z)\,\gamma_d(\mathrm dz)\)，其中 \(\gamma_d\) 是 \(\mathbb R^d\) 上的标准高斯测度，\(Z\sim N(0,I_d)\)。

## 2. 函数类

给定参数 \(K\in\mathbb N\)（非光滑面数上限）、\(r\in\mathbb N\cup\{\infty\}\)（光滑阶）、\(M\ge1\)、\(c\ge0\)（导数与增长界），类 \(\mathcal F_d=\mathcal F_d(K,r,M,c)\) 由满足以下条件的 \(f:\mathbb R^d\to\mathbb R\) 组成：

1. **分片结构**：存在 \(k\le K\) 个函数 \(\varphi_1,\dots,\varphi_k\in C^r(\mathbb R^d)\)，对每个符号模式 \(\sigma\in\{-1,+1\}^k\) 有 \(h_\sigma\in C^r(\mathbb R^d)\)，使得
   \[
   f(z)=h_{\sigma(z)}(z),\qquad \sigma(z)=\big(\operatorname{sign}\varphi_1(z),\dots,\operatorname{sign}\varphi_k(z)\big),
   \]
   在零测集 \(\bigcup_i\{\varphi_i=0\}\) 上 \(f\) 可取任意有限值。
2. **导数与增长**：对所有 \(|\alpha|\le r\)（\(r=\infty\) 时对所有 \(\alpha\)），\(|\partial^\alpha\varphi_i(z)|\le Me^{c|z|}\)，\(|\partial^\alpha h_\sigma(z)|\le Me^{c|z|}\)。
3. **非退化**：在 \(\{\varphi_i=0\}\) 上 \(|\nabla\varphi_i(z)|\ge M^{-1}e^{-c|z|}\)，即每个非光滑面都是正则超曲面；不同的面可以相交。
4. **连续性类型**：若相邻片在公共边界上取值相同，\(f\) 是连续的（折点型）；否则跨面的跳跃有界，\(|h_\sigma-h_{\sigma'}|\le Me^{c|z|}\)（跳跃型）。

说明：

- \(\mathcal F_d\) 不是凸的，也不关于加法对称（两个函数的非光滑面不同，相加后面数会增加），所以 Bakhvalov–Smolyak 的“自适应无用”定理不适用。
- 非光滑面的方向未知；它们不必沿任何坐标或任何固定方向单调；输入坐标可能经过未知的正交变换。
- \(k=0\) 时 \(\mathcal F_d\) 退化为带指数增长界的光滑函数类，所以光滑函数的下界对 \(\mathcal F_d\) 也成立。

**代表性实例族**（来自本题的理论 + 实验版本，说明本题关心的困难情形；纯理论版不评程序）：\(d\in\{8,16,32,64\}\)，所有 \(\varphi_i\)、\(h_\sigma\) 都是解析函数（\(r=\infty\)），输入先经过隐藏的随机正交变换，每例的参数随机抽取。两条赛道计分时等权：

- **困难赛道**（4 个合成族）：取预积分理论最难的情形——沿任何方向都不单调的闭合跳跃面（含尾部稀有事件）、只在零锥方向单调的鞍形折点、带函数值不为零处内部折点的相交折点面；非光滑面是维数为 2 到 \(d\) 的未知子空间里谱各异（各向异性）的二次超曲面，并叠加真正全维的光滑项。这些实例 \(K\le3\)。
- **期权赛道**（4 个族）：单资产离散监测收益（欧式看涨、数字、向下敲出看涨、双障碍敲出看涨），价格路径由波动率随时间和价格水平光滑变化的单因子扩散按 \(m\) 步离散生成，\(d=m\in\{16,32,64\}\)。折点与跳跃面在查询空间中是弯曲的；向下敲出每个监测日贡献一个跳跃面，双敲出为 \(2m\) 个，所以这些实例 \(K\le2m+1\le129\)。

具体形式、参数和正交变换不公开。

## 3. 信息与代价模型

- **信息**：只能逐点查询函数值 \(f(z)\)，\(z\in\mathbb R^d\) 任意有限。查询点可以依赖此前得到的值（自适应），可以随机化（用题面给定的种子）。
- **代价**：每例最多 \(N=2^{14}\) 次查询，重复查询同一点也计次；试探、梯度的有限差分、交叉验证都计入。计算本身不计次，但每例有 30 秒墙钟上限。查询点的每个坐标须满足 \(|z_i|\le40\)（之外高斯密度在双精度下为 0，对积分没有贡献）。
- **输出**：估计值 \(\hat I\)；可选输出声称的误差界 \(B\) 及其类型（确定性，或置信水平 \(1-\alpha\)）。

## 4. 误差准则

- 相对误差 \(|\hat I-I|/|I|\)（代表性实例都满足 \(|I|\) 远离 0）或绝对误差。
- 理论结论可以采用任一标准设定，但必须写明：类上的最坏情形误差；随机化算法的均方根误差或高概率误差；在明确子类上的渐近误差。

## 5. 已知结果（摘要）

| 主题 | 结果 | 出处 |
| --- | --- | --- |
| 光滑类 | \(C^r\) 类确定性 \(n^{-r/d}\)、随机化 \(n^{-r/d-1/2}\)；\(C^r_d\) 类的维数灾难 | Bakhvalov 1959；Hinrichs, Novak, Ullrich, Woźniakowski 2014 |
| 高斯测度、解析函数 | 高斯核 RKHS 中张量 Gauss–Hermite 需 \(O((\ln\varepsilon^{-1})^d)\) 点，指数 \(d\) 紧 | Kuo, Sloan, Woźniakowski 2017 |
| 扰乱网 | 光滑函数上方差 \(O(n^{-3}(\log n)^{d-1})\) | Owen 1997 |
| 一维奇异点 | 一个未知奇异点：自适应 \(n^{-r}\)，非自适应不快于 \(n^{-1}\)；多个奇异点只有渐近结果 | Plaskota, Wasilkowski 2005 |
| 等权 QMC 与跳跃 | 半空间偏差 \(\Theta(n^{1/2-1/(2d)})\)，故等权规则对沿超平面跳跃的函数最坏误差不快于 \(n^{-1/2-1/(2d)}\) | Alexander 1990；Matoušek 1995 |
| RQMC 与不连续函数 | RMSE \(O(n^{-1/2-1/(4d-2)+\epsilon})\)；某些指示函数 \(O(n^{-1/2-1/(2d)})\) | He, Wang 2015 |
| 预积分（单调） | 沿使非光滑面严格单调的变量先积分，得到光滑的 \(d-1\) 维被积函数 | Griebel, Kuo, Sloan 2013；Griewank, Kuo, Leövey, Sloan 2018 |
| 条件 RQMC | 典型期权问题 \(O(n^{-1+\epsilon})\)；最高阶 ANOVA 项仍为 \(O(n^{-1/2-1/(4d-2)+\epsilon})\) | He 2019 |
| 单调性必要 | 单调性失效时预积分后一般出现平方根等奇性，此时 QMC 没有理论支持 | Gilbert, Kuo, Sloan 2022 |
| 方向选择 | 沿主动子空间首向量预积分；预积分不增加扰乱网方差；没有速率保证 | Liu, Owen 2023 |
| 数值光滑化 | 求根加一维积分，配合自适应稀疏网格或 QMC | Bayer, Ben Hammouda, Tempone 2023 |
| 误差估计 | Walsh 系数“锥”内的带保证自适应 QMC | Hickernell, Jiménez Rugama 2016；Clancy 等 2014 |
| 随机化自适应 | 某些问题上随机化自适应比非自适应快 \(n\) 的幂次 | Heinrich 2024 |
| QMC 友好的不连续面 | 不连续面都平行于坐标轴时 RQMC 误差为 \(O(n^{-1+\epsilon})\)，一般情形为 \(O(n^{-1/2-1/(4d-2)+\epsilon})\) | He 2017 |
| 跳跃与平均维数 | 跳跃型岭函数的平均维数可随 \(\sqrt d\) 增长，折点有界；沿合适方向预积分可降到 \(O(1)\) | Hoyt, Owen 2020 |
| 预积分加重要性抽样 | 临界指数增长下 RQMC 加最优漂移重要性抽样仍有接近 \(n^{-1}\) 的速率，预积分后的函数满足所需增长条件 | Chen, Xu, Du, Wang 2025 |

完整书目见文末“参考文献”。

## 6. 开放问题

以下问题在文献中没有完整答案，在本题的规模下都直接影响误差：

1. **不单调与多折点面的正则性**：\(\varphi\) 为 max / min 结构或有多个折点面、且不存在使之严格单调的方向时（例如闭合曲面沿任何方向都进出各一次；不定二次型只沿零锥方向单调），预积分或其他光滑化（包括非线性坐标，如沿射线）之后函数的正则性，以及相应的 RQMC 或稀疏网格速率。
2. **方向选择的保证**：只用点查询、只花预算中 \(o(N)\) 的部分，选出预积分方向或 \(k\) 维主动子空间，并对所得误差给出可证明的界。
3. **黑盒定位**：只有函数值时，沿直线定位折点或跳跃的查询复杂度，包括函数值不为零处的内部折点（零点检测无效），以及定位与积分之间的最优预算分配。
4. **自适应与非自适应**：\(\mathcal F_d\)（或其自然子类）上自适应与非自适应算法的最优误差阶，以及匹配的下界。
5. **误差估计的可靠性**：在什么条件下，误差界或置信区间在 \(\mathcal F_d\) 的子类上有保证的覆盖率。

## 7. 期望的理论结论

- 针对你设计的算法（不需要实现）、在明确写出的子类上的误差界（速率，最好带显式常数或常数的依赖关系）。
- 查询复杂度结论：定位、方向估计、预算分配。
- 下界：任意算法（或某类算法）在类上的最坏情形或随机化下界。
- 误差估计的有效性：所输出误差界的覆盖率保证及其条件。

把结论与第 5 节的已知结果逐条对比，写明属于复现、推广到新类，还是改进已知界。初等的引理（例如抽屉原理型的非自适应下界、一维单调情形预积分的光滑性、二分定位的查询数）鼓励在 Lean 中形式化。

## 参考文献

- R. Alexander, Geometric methods in the study of irregularities of distribution, Combinatorica 10 (1990) 115–136.
- N. S. Bakhvalov, On approximate computation of integrals, Vestnik MGU 4 (1959) 3–18.
- C. Bayer, C. Ben Hammouda, R. Tempone, Numerical smoothing with hierarchical adaptive sparse grids and quasi-Monte Carlo methods for efficient option pricing, Quant. Finance 23 (2023) 209–227.
- J. Chen, Y. Xu, J. Du, X. Wang, Randomized quasi-Monte Carlo and importance sampling for super-fast growing functions with applications to finance, arXiv:2510.06705 (2025).
- N. Clancy, Y. Ding, C. Hamilton, F. J. Hickernell, Y. Zhang, The cost of deterministic, adaptive, automatic algorithms: cones, not balls, J. Complexity 30 (2014) 21–45.
- A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Preintegration is not smoothing when monotonicity fails, in Advances in Modeling and Simulation, Springer (2022); arXiv:2112.11621.
- M. Griebel, F. Y. Kuo, I. H. Sloan, The smoothing effect of integration in \(\mathbb R^d\) and the ANOVA decomposition, Math. Comp. 82 (2013) 383–400.
- A. Griewank, F. Y. Kuo, H. Leövey, I. H. Sloan, High dimensional integration of kinks and jumps — smoothing by preintegration, J. Comput. Appl. Math. 344 (2018) 259–274.
- Z. He, X. Wang, On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 53 (2015).
- Z. He, Quasi-Monte Carlo for discontinuous integrands with singularities along the boundary of the unit cube, arXiv:1702.03361 (2017).
- Z. He, On the error rate of conditional quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 57 (2019) 854–874.
- S. Heinrich, Randomized complexity of mean computation and the adaption problem, J. Complexity (2024) 101872.
- F. J. Hickernell, Ll. A. Jiménez Rugama, Reliable adaptive cubature using digital sequences, in MCQMC 2014, Springer (2016).
- A. Hinrichs, E. Novak, M. Ullrich, H. Woźniakowski, The curse of dimensionality for numerical integration of smooth functions, Math. Comp. 83 (2014) 2853–2863.
- C. R. Hoyt, A. B. Owen, Mean dimension of ridge functions, SIAM J. Numer. Anal. 58 (2020).
- F. Y. Kuo, I. H. Sloan, H. Woźniakowski, Multivariate integration for analytic functions with Gaussian kernels, Math. Comp. 86 (2017) 829–853.
- S. Liu, A. B. Owen, Preintegration via active subspace, SIAM J. Numer. Anal. 61 (2023) 495–514.
- J. Matoušek, Tight upper bounds for the discrepancy of half-spaces, Discrete Comput. Geom. 13 (1995) 593–601.
- E. Novak, On the power of adaption, J. Complexity 12 (1996) 199–237.
- A. B. Owen, Scrambled net variance for integrals of smooth functions, Ann. Statist. 25 (1997) 1541–1562.
- L. Plaskota, G. W. Wasilkowski, Adaption allows efficient integration of functions with unknown singularities, Numer. Math. 102 (2005) 123–144.
- Y. Xiao, X. Wang, Enhancing quasi-Monte Carlo simulation by minimizing effective dimension for derivative pricing, Comput. Econ. 54 (2019) 343–366.
