# 稀疏凸二次规划：理论背景与理论赛道评估（2026-11-24 调研）

理论赛道改造的逐题底稿，总览见 [`results/THEORY_TRACK.md`](../../../results/THEORY_TRACK.md)。参考文献中带 ✓ 的条目已联网核对作者、出处和所引结论；其余是领域经典文献，按通行引用给出。

## 1. 题目在数学上问什么

求解 \(\min_x \tfrac12x^\top Qx+c^\top x\ \text{s.t.}\ Ax\le b\)（\(Q\succeq0\)，\(n=256\)–1024），输出 \((x,y)\)，四项归一化 KKT 残差（原始可行、对偶符号、平稳性、互补）都不超过 \(10^{-7}\)。逐例时限由 CPU 参照决定（1–60 秒），分数 \(100\,p/(1+g/g_{\rm ref})\)，\(p\) 是成功比例，\(g\) 是几何平均时间。

理论上有两种代价模型：一阶（只用 \(Q,A,A^\top\) 的矩阵向量乘）与允许分解（IPM、活动集）。本题规模小，分解完全可行；参照是一阶方法加活动集精化。

## 2. 已知最好的结果（上界）

- **PDHG**（Chambolle, Pock 2011）：遍历平均 \(O(1/k)\)。
- **LP 上的重启 PDHG**：对锐（sharp）的原始-对偶问题线性收敛，迭代数 \(O(\kappa\log(1/\epsilon))\)；\(\kappa\) 与 KKT 系统的锐度（Hoffman 常数的倒数）相关。不重启时存在 \(\Omega(\kappa^2\log(1/\epsilon))\) 的下界（Applegate, Hinder, Lu, Lubin 2023 ✓）。工程实现为 PDLP（Applegate 等 NeurIPS 2021）。
- **QP 上的重启加速 PDHG（rAPDHG）**：Lu, Yang（Math. Program. 2026，2025 年在线 ✓）证明线性收敛，速率在一大类原始-对偶方法中最优，并把速率与 QP KKT 系统的锐度常数联系起来；实现为 PDQP.jl。
- **PDHCG**（Huang, Zhang, Li, Ge, Liu, Ye，INFORMS J. Computing 2025 ✓）：在 rAPDHG 框架内用 CG 不精确求解原始子问题，保持线性收敛且收敛常数改进。本题参照就是出题人的重启 PDHCG 加内层 CG 和活动集精化。
- **其他路线**：ADMM / OSQP（Stellato 等 2020）及其不可行检测（Banjac, Goulart, Stellato, Boyd 2019）；IPM \(O(\sqrt n\log(1/\epsilon))\) 次迭代；凸 QP 多项式时间可解（Kozlov, Tarasov, Khachiyan 1979）；活动集方法有限终止。

## 3. 下界与不可能性

- 仿射约束的光滑凸问题上，一阶方法一般不能从 \(O(1/t)\) 加速到 \(O(1/t^2)\)，\(O(1/t)\) 对凸问题最优；强凸时 \(O(1/t^2)\) 最优（Ouyang, Xu 2021 ✓）。
- 锐度意义下：span-respecting 的原始-对偶方法存在 \(\mathrm{dist}(z^t,Z^\star)\ge(1-\alpha/C)^t\,\mathrm{dist}(z^0,Z^\star)\) 的下界（Applegate 等 2023 ✓，推论 1），重启 PDHG 和 rAPDHG 在常数意义下达到。
- 这些下界只针对一阶（matvec-only）模型。允许分解后不再适用；本题 \(n\le1024\)，直接法完全可行，所以“一阶最优”不等于“本题最快”。

**是否紧**：一阶模型下紧（常数意义）；“所有算法”的意义下，本题规模的实际最优由线性代数常数决定，没有理论结论。

## 4. 开放问题：什么样的结果算贡献

- 一阶法的复杂度依赖锐度或 Hoffman 常数，而这些常数难以计算，复杂度界难以事前评估；可计算的实例相关界是开放方向。
- PDHCG 内层 CG 精度与外层线性速率的最优匹配（不精确子问题的复杂度权衡）。
- 有限精度下可证达到 \(10^{-7}\) KKT 精度。
- 小规模、高精度下 IPM、一阶法、活动集的混合策略，以及有理论保证的切换准则。

## 5. 与本题参照、已有成绩的关系

参照来自出题人（`references/r1/`）。本题尚未完成目标机验收和正式 baseline，还没有发布模型成绩（见 `results/OPEN_ITEMS.md`）。

## 6. 理论赛道建议

**适配度：高。** 上下界在一阶模型下都清楚，参照方法本身就是近年带最优性结论的研究成果；KKT 证书由评测器独立检查，矩阵向量乘次数可以在评测时记录。

**题面需补的数学前提**：代价模型（以矩阵向量乘计，或允许分解时以分解与回代计）；说明隐藏实例的结构族（对角病态、稀疏低秩、网络、箱约束）对应的锐度或条件数范围。

**可要求的定理**

| 级别 | 示例 |
| --- | --- |
| 教科书 | 所用算法（重启 PDHG / PDHCG / IPM）的收敛速率与 KKT 残差保证；KKT 条件对凸 QP 最优性的充分性 |
| 增量 | 带不精确内层 CG 的总 matvec 复杂度（内外层精度匹配）；活动集精化的有限终止与正确性 |
| 前沿 | 突破 rAPDHG / PDHCG 的锐度依赖（必须走出下界模型，例如利用 \(Q\) 的结构或分解） |

**可自动检验的部分**：KKT 残差已由评测器检查；可在诊断输出中报告矩阵向量乘次数，与复杂度声明对照。

**形式化可行性**：凸 QP 的 KKT 充分性可在 Lean 4 中证明；LP 的 Farkas 引理与强对偶已有 Lean 4 形式化（Dvorak, Kolmogorov 2026，见 conic 题的 THEORY.md），可作为起点。线性收敛速率的形式化工作量大。

## 参考文献

1. A. Chambolle, T. Pock, A first-order primal-dual algorithm for convex problems with applications to imaging, J. Math. Imaging Vis. 40 (2011) 120–145。
2. ✓ D. Applegate, O. Hinder, H. Lu, M. Lubin, Faster first-order primal-dual methods for linear programming using restarts and sharpness, Math. Program. 201 (2023) 133–184. <https://arxiv.org/abs/2105.12715>。重启 PDHG 的 \(O(\kappa\log(1/\epsilon))\) 及其在 span-respecting 方法类中的最优性。
3. D. Applegate 等, Practical large-scale linear programming using primal-dual hybrid gradient, NeurIPS 2021。PDLP。
4. ✓ H. Lu, J. Yang, A practical and optimal first-order method for large-scale convex quadratic programming, Math. Program. 215 (2026) 771–808（2025 年在线）. <https://doi.org/10.1007/s10107-025-02241-0>。rAPDHG / PDQP。
5. ✓ Y. Huang, W. Zhang, H. Li, D. Ge, H. Liu, Y. Ye, A restarted primal-dual hybrid conjugate gradient method for large-scale quadratic programming, INFORMS J. Comput.（2025）. <https://doi.org/10.1287/ijoc.2024.0983>；arXiv:2405.16160。本题参照方法。
6. ✓ Y. Ouyang, Y. Xu, Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems, Math. Program. 185 (2021) 1–35. <https://doi.org/10.1007/s10107-019-01420-0>。
7. B. Stellato, G. Banjac, P. Goulart, A. Bemporad, S. Boyd, OSQP: an operator splitting solver for quadratic programs, Math. Program. Comput. 12 (2020) 637–672。
8. G. Banjac, P. Goulart, B. Stellato, S. Boyd, Infeasibility detection in the alternating direction method of multipliers for convex optimization, J. Optim. Theory Appl. 183 (2019) 490–519。
9. M. K. Kozlov, S. P. Tarasov, L. G. Khachiyan, Polynomial solvability of convex quadratic programming, Soviet Math. Dokl. 20 (1979) 1108–1111。
