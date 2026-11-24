# 连续极限外推：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

每例给 4 或 5 个点 \((x_i,y_i)\)，\(x_i>0\) 互异、未排序，要求估计 \(t=\lim_{x\to0^+}y(x)\)，损失 \(|z-t|/\max(1,|t|)\)，60 例平均。数据是确定性的，没有噪声，困难完全来自模型不确定：收敛阶未知、系数未知。

各模型对公开样例的验证一致表明，计分族满足

\[
y(x)=t+a\,x^{p}+b\,x^{q}+c\,x^{q+3/2},\qquad 0<p<q,
\]

有 6 个未知数 \((t,a,b,c,p,q)\)，观测只有 4–5 个，因此**不可辨识**：总存在多组参数在全部观测层上给出相同的 \(y_i\)，极限却不同。题目实质上是从有限信息恢复一个非线性泛函，属于最优恢复（optimal recovery）和 IBC 的范畴，而不是“谁的拟合更准”。

## 2. 已知最好的结果（上界）

- **Richardson 外推**（Richardson 1911；Richardson & Gaunt 1927）：已知首项阶 \(p\) 时，两层即可消去首项，误差降到下一项的阶。Romberg 积分是它的递归形式。
- **广义 Richardson / E-算法**（Håvie 1979；Brezinski 1980）：误差展开 \(y(x)=t+\sum_j a_j g_j(x)\) 中基函数 \(g_j\) 已知时，\(k+1\) 层可消去 \(k\) 项。Sidi 的专著给出收敛与稳定性理论，稳定性由与层级几何有关的 Lebesgue 型常数控制。
- **未知阶**：非线性变换（Aitken \(\Delta^2\)、Wynn \(\varepsilon\) 算法、Levin 变换）在特定序列类上有加速定理。工程上常用三层“观测收敛阶”估计（Roache 1994 的 GCI）。
- **概率外推（GRE）**：Oates 等（JRSS-B 2025 ✓）用高斯过程给误差项建模，称为 Gauss–Richardson Extrapolation。他们证明：误差函数在相应 RKHS 中足够正则时，条件均值相对原数值方法获得多项式（甚至指数）级加速，未知收敛阶可以统计估计。本题强参照就是 GRE（收敛阶、核与长度尺度取固定网格）。前身是 Teymur 等的 BBPN（NeurIPS 2021 ✓）。
- **贝叶斯判决**：给定先验时，绝对损失下的贝叶斯最优估计是后验中位数。Hunyuan 4 单轮采用“在未知阶上做贝叶斯模型平均、报告后验中位数”，得到全榜该格最高分（\(E=0.00084<W=0.00165\)，按 \(100-50E/W\) 得 74.5）。

## 3. 下界与不可能性

- **加速的负结果**（Delahaye, Germain-Bonne 1980 ✓、1982 ✓）：不存在能加速所有对数收敛序列的算法。他们引入“剩余性”（remanence）：具有这一性质的序列集合不可被任何变换一致加速。所以“无条件更好的外推”不存在，最优性只能在受限的类上讨论。
- **不可辨识导致的极小极大下界（二点法）**：若存在 \(\theta,\theta'\) 在所有观测层上给出相同的 \(y_i\)、极限却是 \(t\ne t'\)，那么任何估计在两者之一上的误差至少为 \(|t-t'|/2\)。本题 6 个未知数多于 5 个观测，类内总存在这样的对，所以**不限制参数范围时最坏误差无界**。只有限制参数（例如 \(|a|,|b|,|c|\le M\)、\(p\in[p_{\min},p_{\max}]\)、\(q-p\ge\Delta\)）或给出先验，问题才有有限的极小极大风险或贝叶斯风险。在最优恢复理论里，这个下界就是信息半径：与数据相容的参数集上 \(t\) 取值范围的一半（Micchelli, Rivlin 1977；Traub, Wasilkowski, Woźniakowski 1988）。
- **是否紧**：一般设定下没有匹配的上下界结论；固定参数类后，极小极大风险就等于信息半径，只是没有现成算法能对本题的非线性类精确算出它。

## 4. 开放问题：什么样的结果算贡献

- 在明确的参数类（含层级几何）上计算或界住信息半径，并给出达到或接近它的可计算外推器（极小极大最优或近最优）。
- 模型错设下的稳健性：对不服从光滑展开的情形（振荡、噪声，即本题的 15 例诊断集），给出可证明的检测与回退保证。
- 层级选择（实验设计）：GRE 论文已把选点表述为连续优化，最优设计的理论还不完整。

## 5. 与本题参照、已有成绩的关系

- 弱参照：一阶 Richardson（用最小两个步长），\(U=0.017163\)。强参照：GRE，\(W=0.0016482\)。
- 已有成绩：GPT 系列与 Hunyuan 4 单轮都在 71–75 分（\(E\approx0.00084\)–0.00095），方法多为“在公开样例上校准过的贝叶斯或最小二乘模型平均”。
- 迭代格的问题已有记录（`results/OPEN_ITEMS.md`）：public 反馈就是 8 个带真值的公开样例，没有留出集。Opus 和 Hunyuan 4 的迭代都对样例过拟合（public 90.5 / 95.8，private 22.8 / 47.8），Hunyuan 4 的迭代 README 甚至把“开发样例中两个次主导振幅为负”写进了先验。这正好说明：没有声明的类或先验，“更好”无法定义；而按类或先验声明，过拟合会直接表现为定理假设与隐藏数据不符。

## 6. 理论赛道建议

**适配度：高，但必须先补类定义。** 问题小而清晰；下界论证初等（二点法），模型有能力严格写出；而“达到信息半径”的上界确实开放。

**题面需补的数学前提**：参数类或先验（\(p,q\) 的范围与间隔、系数界、层级生成规则），以及诊断集的性质。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 已知阶时广义 Richardson 的误差阶，以及稳定性常数随层级几何的增长 |
| 增量 | 给定先验下估计器的贝叶斯最优性（后验中位数）及数值积分误差控制；声明类上的二点下界（显式构造不可区分对，给出信息半径下界） |
| 前沿 | 声明类上匹配的上下界（精确刻画信息半径）及达到它的可计算算法；模型错设下的稳健性保证 |

**可自动检验的部分**：若提交声明“在类 \(\mathcal C\) 上误差不超过 \(B(x,y)\)”（可计算的后验界或最坏界），可以在隐藏实例上检测违反。二点下界本身也可以用数值反例核对：两组参数在观测层上的差应当在浮点舍入以内。

**形式化可行性**：二点下界只用实数不等式和显式构造，Richardson 消项是代数恒等式，两者都适合在 Lean 4 中机器检查，是各题里形式化门槛最低的。GP/RKHS 收敛结果超出现有库。

## 参考文献

1. L. F. Richardson, J. A. Gaunt, The deferred approach to the limit, Phil. Trans. R. Soc. A 226 (1927) 299–361。
2. C. Brezinski, A general extrapolation algorithm, Numer. Math. 35 (1980) 175–187。E-算法。
3. T. Håvie, Generalized Neville type extrapolation schemes, BIT 19 (1979) 204–213。
4. A. Sidi, Practical Extrapolation Methods: Theory and Applications, Cambridge University Press, 2003。
5. P. Wynn, On a device for computing the \(e_m(S_n)\) transformation, MTAC 10 (1956) 91–96。
6. D. Levin, Development of non-linear transformations for improving convergence of sequences, Int. J. Comput. Math. B 3 (1973) 371–388。
7. ✓ J. P. Delahaye, B. Germain-Bonne, Résultats négatifs en accélération de la convergence, Numer. Math. 35 (1980) 443–457. <https://eudml.org/doc/186301>。
8. ✓ J. P. Delahaye, B. Germain-Bonne, The set of logarithmically convergent sequences cannot be accelerated, SIAM J. Numer. Anal. 19 (1982) 840–844. <https://doi.org/10.1137/0719059>。
9. ✓ C. J. Oates, T. Karvonen, A. L. Teckentrup, M. Strocchi 等, Probabilistic Richardson extrapolation, J. R. Stat. Soc. B（2024 年在线，2025）. <https://doi.org/10.1093/jrsssb/qkae098>。GRE 及其加速定理（本题强参照）。
10. ✓ O. Teymur, C. Foley, P. Breen, T. Karvonen, C. J. Oates, Black box probabilistic numerics, NeurIPS 2021. <https://proceedings.neurips.cc/paper/2021/hash/c4de8ced6214345614d33fb0b16a8acd-Abstract.html>。
11. C. A. Micchelli, T. J. Rivlin, A survey of optimal recovery, in Optimal Estimation in Approximation Theory, Plenum, 1977, 1–54。信息半径与最优恢复。
12. J. F. Traub, G. W. Wasilkowski, H. Woźniakowski, Information-Based Complexity, Academic Press, 1988。
13. P. J. Roache, Perspective: a method for uniform reporting of grid refinement studies, J. Fluids Eng. 116 (1994) 405–413。观测收敛阶与 GCI。
