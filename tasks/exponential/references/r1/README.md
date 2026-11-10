# 指数传播：人类参照组（panel）

每个子目录都是一份普通提交（`run.sh` 入口，FAMEXP2 协议），全部来自已发表的方法。上游作者代码一律未修改，只加薄适配器；参数只在 29 个公开开发用例上按事先声明的规则选定，不使用私有数据，也不参考任何 AI 提交。评分时逐例取参照组中最快且合格的成员作为参考时间（virtual best），见 `../DESIGN.md` 第 4 节。

| 成员 | 方法与出处 | 实现 | 参数（公开数据标定） | 公开用例 |
| --- | --- | --- | --- | --- |
| `expokit` | Sidje, Expokit, ACM TOMS 24(1), 1998：对称 Lanczos，DSPHIV | 作者 Fortran，编译执行 | tol = 1000·rtol，Krylov m = 30（作者默认） | 29/29 |
| `kiops` | Gaudreault, Rainwater, Tokman, KIOPS, J. Comput. Phys. 372, 2018 | 作者 MATLAB 代码，Octave | tol = 0.01·rtol（原 pilot 冻结策略） | 29/29 |
| `expmv` | Al-Mohy & Higham, SIAM J. Sci. Comput. 33(2), 2011：截断 Taylor + 缩放 | 作者 MATLAB 代码，Octave | 精度 p4 single，p7/p10 double | 29/29 |
| `scipy` | SciPy `expm_multiply`（Al-Mohy–Higham 算法的 SciPy 实现） | SciPy 1.16 / Python | 无可调参数 | 29/29 |
| `chebyshev` | Tal-Ezer & Kosloff 1984；Bergamaschi & Vianello 2000；Higham 2008：谱区间上的 Chebyshev 展开 | 独立的文献实现（C），由未接触任何 AI 提交的 agent 编写 | 解析参数（安全系数 8） | 29/29 |

尚未接入：SLEPc MFN（需在镜像中构建 PETSc/SLEPc）、ExponentialUtilities.jl（镜像中没有 Julia）、phipm（许可证未明确）、rational Krylov（没有许可明确的作者实现）。见 `../DESIGN.md` 第 3 节。
