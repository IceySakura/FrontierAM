# 黑盒多维积分：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

程序对 \(f:[0,1]^d\to\mathbb R\)（\(d\in\{2,3,5,8\}\)）最多做 \(n=2048\) 次函数值查询，查询点可以依赖已得到的函数值（自适应），输出 \(\hat I\approx I(f)=\int_{[0,1]^d}f\)。这正是信息基复杂度（IBC）的标准积分问题：只用标准信息（函数值），误差相对一个函数类 \(F\) 定义，可以取最坏情形、随机化或平均情形。

本题没有公开函数类，只说明隐藏实例是“可解析积分”的函数，族包括周期（Fourier）ridge、多方向折点 ridge 叠加、乘积峰。计分用 \(\log_{10}\) 相对误差相对冻结的 HumanPortfolio-v1 的配对增益，截到 \([-2,2]\)，再按族 × 维度平均。因此分数是有限隐藏集上的经验量，不对应任何函数类上的最坏误差；截断使“比参照好 100 倍以上”的边际收益为零。

## 2. 已知最好的结果（上界）

- **各向同性光滑类** \(C^r([0,1]^d)\)：确定性算法的最优误差阶 \(n^{-r/d}\)，随机化算法 \(n^{-r/d-1/2}\)（Bakhvalov 1959）。随机化比确定性多出一个 \(n^{-1/2}\)，代价是只在期望意义下成立。
- **混合光滑类**（主导混合导数 \(W^r_{\rm mix}\)）：Frolov 格点规则达到最优阶 \(n^{-r}(\log n)^{(d-1)/2}\)；Smolyak 稀疏网格达到 \(n^{-r}\) 乘对数因子；高阶 QMC（数字网、格点规则）在加权 Korobov / Sobolev 空间上接近 \(n^{-\alpha}\)，在权重衰减足够快时误差界与维数无关（Dick, Kuo, Sloan 2013；可解性理论见 Novak, Woźniakowski）。
- **解析类**：张量 Gauss / Clenshaw–Curtis 指数收敛；Trefethen（2017）说明在超立方上按欧氏次数截断比张量次数更省点。
- **自适应何时有用**：对凸且对称的函数类，自适应不改善最坏误差阶（Bakhvalov；Novak 1996 综述）。对含未知位置奇异点的分段光滑类则不同：一维、一个未知奇异点的 \(r\) 阶光滑函数，自适应求积的最坏误差为 \(n^{-r}\)，而任何非自适应方法不会快于 \(n^{-1}\)（Plaskota, Wasilkowski 2005 ✓）。折点 ridge \(g(a\cdot x)\)（\(g\) 有折点）在投影到 \(a\cdot x\) 后正是这种情形。
- **ridge 函数**：\(f(x)=g(a\cdot x)\) 的方向 \(a\) 可以用与维数无关（或多项式依赖）的查询数恢复（Cohen, Daubechies, DeVore, Kerkyacharian, Picard 2012；Fornasier, Schnass, Vybíral 2012）。之后积分化为一维：\(\int g(a\cdot x)\,dx=\int g(s)\rho_a(s)\,ds\)，其中 \(\rho_a\) 是 \(a\cdot U\)（\(U\) 在 \([0,1]^d\) 均匀）的分段多项式密度，由 \(a\) 精确可算。“先识别结构再积分”的路线以此为理论依据。
- **Genz 测试族**（Genz 1984）：oscillatory、product peak、corner peak、Gaussian、continuous、discontinuous。本题的乘积峰就是 Genz product peak；可分离函数用张量求积按一维速率收敛，这也是该族增益普遍顶到截断上限 \(+2\) 的原因。

## 3. 下界与不可能性

- Bakhvalov 下界与上述上界在阶意义下匹配，\(C^r\) 类在确定性和随机化两种设定下都是紧的。
- **维数灾难**：对 \(C^r_d\)（所有导数有界）类，确定性积分达到误差 \(\varepsilon<1\) 所需点数至少为 \(c_r(1-\varepsilon)\,d^{\,d/(2r+3)}\)（Hinrichs, Novak, Ullrich, Woźniakowski 2014 ✓），是超指数的。可见“光滑”本身在高维给不出好的保证，必须借助结构（混合光滑加权重、低有效维、ridge 结构）。
- 有未知奇异点时，非自适应方法的最坏误差不快于 \(n^{-1}\)（Plaskota, Wasilkowski 2005 ✓）。
- 有限 \(n=2048\) 下的非渐近最优性一般没有精确结论：已知结果多是阶，常数未定。

**是否紧**：经典函数类上上下界在阶意义下匹配。本题“多族混合、族未知、有限预算”的设定没有现成的最优性定理。

## 4. 开放问题：什么样的结果算贡献

- ridge、折点、周期、峰的混合类上，自适应算法的最坏误差阶及匹配下界。Plaskota–Wasilkowski 的结果主要是一维；奇异面位置未知的高维分段光滑积分，结果很少。
- 带模型选择的“识别-积分”算法的 oracle 不等式：误差不超过族内最优方法的误差加上识别代价。这类结论在统计的自适应估计中常见，在积分中少见。
- 有限预算下的实例最优积分。
- 本题隐藏集尚未包含旋转或相关的生成器（题面已声明）；对这类非轴对齐结构的稳健性也是开放问题。

## 5. 与本题参照、已有成绩的关系

- 弱参照普通蒙特卡洛：随机化最坏误差 \(n^{-1/2}\)，增益 \(G_0=-0.713\)。
- 强参照 HumanPortfolio-v1：2、5 维用 pcubature（自适应 Clenshaw–Curtis 型），3、8 维用 QMCPy 数字网，都是光滑类或混合光滑类的理论方法，没有专门处理折点。
- 已有成绩：GPT-6.1 Sol 迭代 \(G=1.496\)（fourier +0.88、折点 ridge +1.60、乘积峰 +2.00），按 \(50+25G\) 得 87.4；GPT-6.2 Sol 迭代 93.4。它们的做法是识别可分离、脊函数、折面（定位梯度跳变）和截断幂 \(\max(0,v\cdot(x-u))^p\) 的参数，理论上对应“ridge 恢复 + 一维精确积分”，但提交没有给出任何保证。Hunyuan 4 迭代（\(G=0.640\)，66.0）没有识别出折点 ridge（该族 −0.90）。`results/INSIGHTS.md` 已指出：题面点名了族，顶端接近饱和，建议出题人加入非参数族。

## 6. 理论赛道建议

**适配度：高。** IBC 对这个问题有完整的上下界框架，结构识别路线又有明确的理论缺口（高维自适应、混合类、oracle 不等式）。

**题面需补的数学前提**：公开至少一部分函数类的定义，例如：折点 ridge 类 \(\{g(a\cdot x): g\in PC^r,\ \text{至多 }k\text{ 个折点},\ \|a\|=1\}\)、可分离峰类、周期 Korobov 类 \(E_\alpha\)，以及“族的有限并”。否则“最坏误差”和“比 SOTA 更好”都无从定义。隐藏实例仍可以只是这些类中的样本。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所用规则在声明类上的误差阶，例如格点规则在 Korobov 空间的 \(O(n^{-\alpha}(\log n)^{c})\)、张量 Gauss 在解析类的指数收敛 |
| 增量 | 带识别步骤的算法的 oracle 不等式（识别失败概率 × 截断误差 + 积分误差）；折点 ridge 类上自适应误差阶的 \(d\) 维推广 |
| 前沿 | 混合类上的匹配上下界；有限预算的实例最优性；自适应在高维分段光滑类上的定量优势 |

**可自动检验的部分**：如果定理给出可计算的误差上界（误差估计器加置信度），可以在满足类假设的隐藏实例上统计违反率；声称的界一旦被违反，就说明定理或实现有错。

**形式化可行性**：一维复合 Newton–Cotes / 梯形规则的误差界、有限点集上的 Koksma–Hlawka 型不等式，有希望在 Lean 4 + Mathlib 中完成。高维 IBC 下界的 fooling-function 论证需要测度论和卷积光滑化，几小时内难以完成。

## 参考文献

1. N. S. Bakhvalov, On the approximate calculation of multiple integrals, Vestnik MGU 4 (1959) 3–18；英译 J. Complexity 31 (2015) 502–516。确定性 \(n^{-r/d}\) 与随机化 \(n^{-r/d-1/2}\) 的最优阶。
2. E. Novak, On the power of adaption, J. Complexity 12 (1996) 199–237。自适应何时有用的综述。
3. ✓ L. Plaskota, G. W. Wasilkowski, Adaption allows efficient integration of functions with unknown singularities, Numer. Math. 102 (2005) 123–144. <https://doi.org/10.1007/s00211-005-0640-3>。自适应 \(n^{-r}\) 对比非自适应 \(n^{-1}\)。
4. ✓ A. Hinrichs, E. Novak, M. Ullrich, H. Woźniakowski, The curse of dimensionality for numerical integration of smooth functions, Math. Comp. 83 (2014) 2853–2863. <https://arxiv.org/abs/1211.0871>。\(C^r_d\) 积分的超指数下界。
5. J. Dick, F. Y. Kuo, I. H. Sloan, High-dimensional integration: the quasi-Monte Carlo way, Acta Numerica 22 (2013) 133–288。QMC、加权空间与可解性。
6. E. Novak, H. Woźniakowski, Tractability of Multivariate Problems, Vols. I–III, EMS, 2008–2012。
7. K. K. Frolov, Upper error bounds for quadrature formulas on function classes, Dokl. Akad. Nauk SSSR 231 (1976) 818–821。混合光滑类的最优格点规则。
8. A. Cohen, I. Daubechies, R. DeVore, G. Kerkyacharian, D. Picard, Capturing ridge functions in high dimensions from point queries, Constr. Approx. 35 (2012) 225–243。
9. M. Fornasier, K. Schnass, J. Vybíral, Learning functions of few arbitrary linear parameters in high dimensions, Found. Comput. Math. 12 (2012) 229–262。
10. A. Genz, Testing multidimensional integration routines, in Tools, Methods and Languages for Scientific and Engineering Computation, North-Holland, 1984, 81–94。Genz 测试族。
11. L. N. Trefethen, Cubature, approximation, and isotropy in the hypercube, SIAM Review 59 (2017) 469–491。
