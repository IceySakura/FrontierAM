# blackbox-cubature：人类 SOTA 锚点与价值偏好（11-27 预先登记）

用于 `THEORY_REVIEW.md` 第 2 节的分档评分，适用于 v3.2–v3.5 的理论部分与 `blackbox-cubature-theory-v1`。出处都读过原文（arXiv 源码或 PDF），未能读到原文的单独标出。问题见 `versions/v3.5/workspace/THEORY_APPENDIX.md`：在 \(Z\sim N(0,I_d)\) 下积分分段光滑函数，片间可以有折点或跳跃，只能逐点查询。

**比较的约定**：只比较同一类型的结论（算法保证与算法保证，下界与下界，正则性与正则性）。界按阶比较，忽略常数因子。只在部分参数范围内更好、在其他范围更差，算作不可比。

## 坐标

| 类型 | 假设坐标（越往右，定理越强） | 强度坐标 |
| --- | --- | --- |
| 算法保证（上界） | **几何**：有限参数族（如已知内点或已知框架的二次曲面）→ 二次曲面 → 光滑凸曲面 → 一般 \(C^2\) 曲面（有限 Minkowski 容度）。**单调性**：要求在已知坐标单调 → 要求在某个未知方向单调 → 不要求。**已知结构**：需要已知位置、框架或参数 → 黑盒。**片的正则性**：解析 → \(C^r\) → 有界变差 / \(L^2\) | **保证类型**：渐近或单个实例 → 随机化均方根误差 → 确定性最坏误差。**速率**：\(n\) 的指数与对数因子、维数依赖 |
| 下界 | **困难实例所在的类**：越小越强。**覆盖的算法**：确定性非自适应 → 确定性自适应 → 随机化非自适应 → 随机化自适应 | 下界的阶（\(n\)、\(\varepsilon\)、\(d\)） |
| 正则性 | 同算法保证的假设坐标 | 得到的光滑性：有奇性 → \(C^{1,1/2}\) 等低阶 → Lipschitz、\(C^1\) → 加权 Sobolev（各向同性或混合光滑）\(r\) 阶 |

## 锚点（人类前沿）

| 编号 | 类型 | 出处 | 内容 |
| --- | --- | --- | --- |
| A-MC | 算法保证 | 普通蒙特卡洛 | 任意 \(f\in L^2(\gamma_d)\)，随机化均方根误差 \(\sigma(f)\,n^{-1/2}\)，与维数无关；对 \(L^2\) 类是最优阶。只在更强假设下得到 \(n^{-1/2}\) 的结论被它支配 |
| A-GKLS | 正则性 | Griewank–Kuo–Leövey–Sloan（J. Comput. Appl. Math. 344 (2018)；arXiv:1712.00920）定理 1、2 | \(f=\theta\cdot\mathrm{ind}(\phi)\)，在已知坐标 \(j\) 上 \(D_j\phi>0\) 处处成立且 \(x_j\to\infty\) 时 \(\phi\to\infty\)，\(\theta,\phi\) 属于加权 Sobolev 空间：预积分 \(P_jf\) 保持 \(r\) 阶光滑（定理 1），在 \(r_j\ge\sum_{k\ne j}r_k\) 时保持混合光滑（定理 2），因而 QMC 有理论速率 |
| A-GKS | 正则性（反面） | Gilbert–Kuo–Sloan（arXiv:2112.11621）定理 1 | \(\phi\in C^2\)，某点 \(\partial\phi/\partial y_1=0\)、\(\partial^2\phi/\partial y_1^2\ne0\)、\(\nabla\phi\ne0\) 时，\(P_1f\) 沿不与 \(\nabla_{-1}\phi\) 正交的直线出现平方根奇性 |
| A-HW | 算法保证 | He–Wang（SIAM J. Numer. Anal. 53 (2015)）定理 3.5、3.6（经 He，arXiv:1702.03361 命题 2.3 核对）；He 2017（同一 arXiv）对边界奇性的推广 | \(f=g\,\mathbf 1_\Omega\)，\(g\) 的 Hardy–Krause 变差有界，\(\partial\Omega\) 有 \((d-1)\) 维 Minkowski 容度：扰乱 \((t,d)\) 序列的方差 \(O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})\)（无单调性、黑盒）；部分轴平行（不平行维数 \(d_u\)）时 \(d\) 换成 \(d_u\)。He 2017：沿单位立方体边界有奇性、增长率为 \(A_i\) 时期望误差 \(O(n^{-\gamma(1/2+1/(4d-2))+\epsilon})\)，\(\gamma=1-\max_iA_i\) |
| A-LO | 算法保证 | Liu–Owen（SIAM J. Numer. Anal. 61 (2023)；arXiv:2202.02682） | 沿任一方向预积分不增加扰乱网的方差；主动子空间第一方向与“最大化线性组合的 Sobol' 指数”这一不可行选择的关系。方向在数值例中由样本梯度估计，没有给出估计误差对积分误差的保证 |
| A-RV | 下界 | Rademacher–Vempala（Adv. Math. 219 (2008)；arXiv:cs/0608054）定理 2、定理 23 | 随机化算法把凸体体积近似到 \((1+c)\) 倍需要 \(\Omega(n^2/\log n)\) 次成员查询，对平行多面体已成立；平行多面体上 \(O(n^2\log n)\) 足够；定理 23 为非自适应下界 |
| A-BF | 下界 | Bárány–Füredi（1987）定理 1 | 确定性多项式次成员查询的算法，体积上下估计之比至少 \((cd/\log d)^d\) |
| A-HNUW | 下界 | Hinrichs–Novak–Ullrich–Woźniakowski（Math. Comp. 83 (2014)） | 光滑函数类上确定性积分的维数灾难 |
| A-PW | 算法保证与下界 | Plaskota–Wasilkowski（Numer. Math. 102 (2005)）；Plaskota–Wasilkowski–Zhao（Math. Comp. 77 (2008)） | 一维、未知奇点：自适应恢复光滑类的最优速率；多个奇点时自适应在最坏情形下无益（后者针对逼近）。**定理细节未读到原文**，只按标题与此前评审的引用使用 |

**判定复述用的经典结果**（不是 SOTA 锚点，用于判断“偏序不优”）：Sikorski 1982（二分对求零最优）、Kiefer 1957（单调函数积分的最优误差）、Novak 1996 综述中的阶跃例（自适应二分约 \(2^{-n}\)，非自适应约 \(1/(2n+2)\)）、Bakhvalov 平均法（Novak 1988）。模型在它们的特例上复现，记为偏序不优。

## 价值偏好（不可比一档的审查依据，取自人类论文原文）

- **P1 单调性不成立时的理论支持**：Gilbert–Kuo–Sloan 结论原文："because of the proven lack of smoothness, the resulting preintegrated function no longer belongs to the space of \((d-1)\)-variate functions of dominating mixed smoothness of order one, and as a consequence there is at present no theoretical support for our use of QMC integration for this \((d-1)\)-dimensional integral"。对单调性不成立的自然类（例如二次曲面上的非单调跳跃或折点），可证明地给出快于蒙特卡洛的保证（QMC/RQMC、分层等，不限于预积分路线），有价值。
- **P2 预积分方向的选择**：Gilbert–Kuo–Sloan 结论原文："The paper does not offer guidance on the choice of variable if there is more than one such variable"；Liu–Owen 用主动子空间自动选方向，但没有估计误差的保证。只用点查询选出方向、并对积分误差给出可证明界的结果，有价值。
- **P3 对 QMC 友好的间断**：He 2017 结论原文："We also show theoretically the benefits of making discontinuities QMC-friendly"，以及 "It would be interesting to know how generally the problems from financial engineering fit into this setting"；He–Wang 定理 3.6 表明轴平行结构能改进速率。对不轴平行、但有更多结构的边界给出可证明的更好速率，有价值。
- **P4 体积与查询复杂度**：Rademacher–Vempala 讨论一节原文："It is an open problem as to whether this factor [\(1/\log n\)] can be removed in the variance lower bound"；"It is conceivable that our lower bound for membership oracle queries can be improved to \(n^3\), although one would have to use bodies other than parallelopipeds"；"it is an open problem to give a faster algorithm using a separation oracle"。回应其中之一的可证明结果有价值。
- **P5 非凸类上的自适应**：Novak 2024 综述（arXiv:2307.06787）指出有重要的函数类不是凸对称的（引 Plaskota–Wasilkowski 等），并写明随机化设定下自适应是否有用长期未决、直到 Heinrich 2024 才对某些类解决。对多元分段光滑类、未知间断面，刻画自适应（尤其随机化自适应）的作用，有价值；归结为一维阶跃例的有限参数族（如同心球、单参数椭球族）不算。

**价值低的情形**（预先登记）：只因限于有限参数族并精确识别参数而不可比（已知内点的二次曲面、精确信息门槛、矩阵恢复），自定义类上的“大海捞针”下界，以及经典一维分离在 \(d\) 维特殊族上的推广，且不满足 P1–P5。
