# 星系快照引力加速度：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

计算 Plummer 软化引力加速度

\[
a_i=G\sum_{j\ne i}m_j\frac{x_j-x_i}{(\|x_j-x_i\|^2+\epsilon^2)^{3/2}},\qquad G=1,\ \epsilon=0.01 .
\]

v4 正式集为 4 档规模（6.5 万到约 419 万个等质量粒子）× 5 类分布（Hernquist、Jaffe、Einasto、扁平盘、四星团群），逐例 CPU 时限按参照标定（6.5 万粒子约 1.6 秒，更大规模按比例放宽）。误差是逐粒子相对误差的 RMS，再按快照、类别平均；计分按对数刻度，\(E_0\) 为经典树（\(\theta=0.8\)），\(E_{50}\) 为改进树（\(\theta=0.7\)，两者都是 Treecode2 参数），\(E\le10^{-15}\) 记 100 分。

本质上这是“固定 CPU 预算下的精度”，即精度-工作量权衡曲线上的一点。

## 2. 已知最好的结果（上界）

- **直接求和**：\(O(N^2)\)，精度到浮点舍入（约 \(10^{-15}\)–\(10^{-16}\)）。本题规模下远超时限。
- **Barnes–Hut 树**（1986）：\(O(N\log N)\)，误差由张角 \(\theta\) 和展开阶控制。Salmon, Warren（1994）给出严格误差界，以及比几何张角更安全的多极接受准则（MAC）。
- **快速多极方法（FMM）**（Greengard, Rokhlin 1987）：多极 / 局部展开加平移，固定精度下 \(O(N)\)。三维截断误差的典型形式为 \(\lesssim\frac{A}{r-a}(a/r)^{p+1}\)，所以 \(p=O(\log(1/\epsilon))\)。平移算子的成本从 \(O(p^4)\) 降到 \(O(p^3)\)（旋转），再到 \(O(p^2)\)（平面波 / 指数展开，Greengard, Rokhlin 1997；自适应版本见 Cheng, Greengard, Rokhlin 1999）。
- **天体物理专用**：Dehnen 的笛卡尔 FMM（falcON，2000、2002；综述 2014）支持一般软化核，并保持动量守恒，是星系动力学中的主流快速方法。
- **核无关 FMM**（Ying, Biros, Zorin 2004）与 \(\mathcal H^2\) 矩阵（Hackbusch；Börm）。
- **任意分布**：well-separated pair decomposition（Callahan, Kosaraju 1995）用 \(O(N\log N)\) 预处理构造分解，之后对固定精度只需 \(O(N)\) 个相互作用，不依赖粒子分布的均匀性。

## 3. 下界与不可能性

- \(\Omega(N)\)：读入和写出就需要线性时间。
- 自适应 FMM 的 \(O(N)\) 依赖分布：Aluru（1996）指出 Greengard 的自适应算法对某些分布不是 \(O(N)\)（树深度不受控）。WSPD 类方法用 \(O(N\log N)\) 预处理规避这个问题。本题的 Jaffe（中心密度发散）和四星团（多尺度团块）正是树深度敏感的分布。
- 展开项数：三维 \(p\) 次调和多项式空间维数为 \((p+1)^2\)，标准论证给出固定分离比下 \(\epsilon\)-秩为 \(O(\log^2(1/\epsilon))\) 的上界；匹配下界（Kolmogorov 宽度意义）的严格文献本次未核实，可作为模型自行证明的命题。
- 浮点极限：双精度下 \(E\approx10^{-15}\) 是实际下限，题面已把它设为 100 分点。

**是否紧**：固定精度下 \(O(N)\) 与 \(\Omega(N)\) 匹配（分布良好或使用 WSPD 时）；精度-工作量曲线的常数级最优性没有理论结论。

## 4. 开放问题：什么样的结果算贡献

- 软化核不再调和：Plummer 核的多极展开截断误差，以及近场软化区域的处理，缺少与 Laplace 核同样严格的误差界。
- CPU 预算约束下 \((\theta,p,\text{叶子大小})\) 最优选择的理论刻画，以及对重尾、团块分布（Jaffe、group）的复杂度保证。
- SIMD / 缓存感知的代价模型下的最优算法设计（理论常数与实测常数之间的差距）。

## 5. 与本题参照、已有成绩的关系

- 参照 Treecode2 是 1986 / 1990 年代的树算法。`results/INSIGHTS.md` 已记录参照偏弱：模型用 FMM 把误差降低 3–5 个数量级（v3 中 GPT-6.1 Sol 迭代 \(6.9\times10^{-10}\)），因此 v3 改为对数刻度、v4 增大规模。
- v4 成绩：GPT-6.2 Sol 单轮 \(E=2.75\times10^{-9}\)（72.8 分），GPT-6 Astra 迭代 \(1.5\times10^{-8}\)（69.6 分）。Hunyuan 4 单轮设计了高阶双树 FMM，到截止仍在修 bug，80 个用例 × 3 次全部 CPU 超时，记 0。

## 6. 理论赛道建议

**适配度：高。** 误差界和复杂度都有成熟理论，软化核又留有真实缺口；评测端有逐粒子参考加速度，可以直接逐粒子核对误差声明，这是各题中最好的自动检验条件之一。

**题面需补的数学前提**：分布类的数学刻画（例如密度比有界、最小间距、层级聚集度），以及代价单位（相互作用计数或浮点运算计数）。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所用 MAC 与展开阶 \(p\) 在 Laplace 核下的逐粒子先验误差界；\(O(Np^2)\) 或 \(O(Np^3)\) 的运算计数 |
| 增量 | Plummer 软化核下的严格误差界（含近场软化的处理）；对声明分布类的相互作用计数上界；CPU 预算下 \((\theta,p)\) 的最优选择定理 |
| 前沿 | 精度-工作量的匹配下界（\(\epsilon\)-秩下界加树结构下界） |

**可自动检验的部分**：评测端有逐粒子参考值，可检验“逐粒子误差 \(\le\) 声明界”是否在所有粒子上成立；可要求输出相互作用计数，对照复杂度声明。

**形式化可行性**：标量或一维版本的截断误差界（几何级数）和树相互作用计数论证可以在 Lean 4 中完成；球谐展开与加法定理不在 Mathlib 中，三维完整误差界的形式化困难。

## 参考文献

1. J. Barnes, P. Hut, A hierarchical \(O(N\log N)\) force-calculation algorithm, Nature 324 (1986) 446–449。
2. ✓ J. K. Salmon, M. S. Warren, Skeletons from the treecode closet, J. Comput. Phys. 111 (1994) 136–155。Barnes–Hut 的严格误差分析与 MAC。
3. L. Greengard, V. Rokhlin, A fast algorithm for particle simulations, J. Comput. Phys. 73 (1987) 325–348。
4. L. Greengard, V. Rokhlin, A new version of the fast multipole method for the Laplace equation in three dimensions, Acta Numerica 6 (1997) 229–269。
5. H. Cheng, L. Greengard, V. Rokhlin, A fast adaptive multipole algorithm in three dimensions, J. Comput. Phys. 155 (1999) 468–498。
6. W. Dehnen, A very fast and momentum-conserving tree code, Astrophys. J. 536 (2000) L39–L42；A hierarchical \(O(N)\) force calculation algorithm, J. Comput. Phys. 179 (2002) 27–42。
7. W. Dehnen, A fast multipole method for stellar dynamics, Comput. Astrophys. Cosmol. 1 (2014) 1。
8. L. Ying, G. Biros, D. Zorin, A kernel-independent adaptive fast multipole algorithm in two and three dimensions, J. Comput. Phys. 196 (2004) 591–626。
9. P. B. Callahan, S. R. Kosaraju, A decomposition of multidimensional point sets with applications to k-nearest-neighbors and n-body potential fields, J. ACM 42 (1995) 67–90。
10. ✓ S. Aluru, Greengard's N-body algorithm is not order N, SIAM J. Sci. Comput. 17 (1996) 773–776。
11. W. Hackbusch, Hierarchical Matrices: Algorithms and Analysis, Springer, 2015；S. Börm, Efficient Numerical Methods for Non-local Operators, EMS, 2010。
