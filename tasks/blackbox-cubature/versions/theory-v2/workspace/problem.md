# 任务：高斯测度下分段光滑函数的黑盒积分（纯理论，theory-v2）

本题只考理论：推进黑盒积分问题的理论，目标是超越人类已有的最好结果。不需要写程序，也不评测程序；会话里的全部时间都用于研究、证明和 Lean 形式化。只有通过形式化检查、且形式化陈述忠实表达文档结论的结果才计分；每条结论都要与强的人类已知结果对比。

会话时长见启动说明（本题为 6 小时）。提前结束不加分：剩余的时间应当用来把一条结论做得更强，或把它完整形式化。开发过程中没有评测反馈，但可以在本地运行与评测相同的形式化检查。

## 交付

所有交付物写入 `/workspace/submission/`：

| 文件 | 内容 |
| --- | --- |
| `theory/THEORY.md` | 理论文档（Markdown，数学公式用 LaTeX），要求见“理论文档” |
| `theory/formal/` | Lean 4 形式化工程（从 `/workspace/examples/lean_project/` 开始），要求见“形式化” |

评测只收集 `/workspace/submission/`；不要把交付物写在别处。

## 问题

估计 \(I(f)=\mathbb E[f(Z)]\)，\(Z\sim N(0,I_d)\)，只能逐点查询函数值（可以自适应，可以随机化），每个查询计一次代价。被积函数 \(f:\mathbb R^d\to\mathbb R\) 是分段光滑的：在至多 \(K\) 个光滑超曲面 \(\{\varphi_i=0\}\) 上可以有折点（连续、导数跳跃）或有界跳跃，各片内部光滑，增长不超过 \(Me^{c|z|}\)。正式定义（类 \(\mathcal F_d(K,r,M,c)\)）、信息与代价模型、误差准则、已知结果、开放问题和本题关心的困难情形见 `/workspace/THEORY_APPENDIX.md`，它是本题的一部分。理论结论应针对其中声明的类，或你明确写出的子类，或这个问题的自然推广（例如去掉黑盒限制、已知非光滑面时的预积分理论）。

## 理论文档

`theory/THEORY.md` 应包含：

1. **结论与证明**：在明确写出的假设下的定理，例如误差界或收敛速率（最好带显式常数或常数的依赖关系）、查询复杂度、任意算法的下界、正则性。每条结论编号（R1、R2、……），给出完整的数学证明和对应的 Lean 定理名；未证明的标为“猜想”。涉及算法时，把算法写到可以据此分析的程度（不需要实现）。
2. **与人类已知结果对比**：对每条结论，找出一个或多个最强的可比人类结果（`THEORY_APPENDIX.md` 第 5 节的文献或你检索到的其他文献；写明可核对的出处，例如 arXiv 编号或 DOI，以及定理编号，并对照原文准确复述其假设与结论），并归入一类：
   - **严格更强**：在相同或更弱的假设下得到严格更好的结论（更快的速率，更好的常数或维数依赖，更广的函数类），或用更弱的假设得到相同结论，即突破了该人类结果；
   - **严格更弱**：结论被某个已知结果蕴含，且后者严格更强；
   - **相当**：与已知结果等价（复现）；
   - **不可比**：各有所长。说明得失，并依据人类论文自身表达的观点（它们指出的局限、提出的开放问题、关心的参数范围）论证这条结论的价值。

   如果回答了附录第 6 节某个开放问题的一部分，也在这里写明。

## 人类 SOTA 目标与登记的锚点

下面四个目标都取自人类论文中的定理，或作者自己写明的空白（出处已核对，可以联网阅读原文）。至少选择一个，尝试给出**严格强于**所列人类结果的结论：在相同或更弱的假设下得到更强的结论（更快的速率、更好的常数、更紧的上下界），或用更弱的假设得到相同的结论；或者在作者写明的空白上取得可证明的进展。

**目标 A：单调性不成立时 QMC 的理论支持（附录开放问题 1）。** Gilbert, Kuo, Sloan, *Preintegration is not smoothing when monotonicity fails*（arXiv:2112.11621），定理 1：\(\varphi\in C^2(\mathbb R^d)\)，若在 \(y^*\) 处 \(\partial\varphi/\partial y_1=0\)、\(\partial^2\varphi/\partial y_1^2\ne0\) 且 \(\nabla\varphi\ne0\)，则沿 \(y_1\) 预积分后的函数在 \(y^*_{-1}\) 处、沿任何不与 \(\nabla_{-1}\varphi(y^*)\) 正交的直线出现平方根型奇性。结论一节原文："because of the proven lack of smoothness, the resulting preintegrated function no longer belongs to the space of \((d-1)\)-variate functions of dominating mixed smoothness of order one, and as a consequence there is at present no theoretical support for our use of QMC integration for this \((d-1)\)-dimensional integral." 进展的例子：对一类自然的非单调跳跃或折点（例如二次曲面），证明预积分（或其他降维、光滑化方式）之后的积分在 QMC 或 RQMC 下有快于蒙特卡洛的收敛保证；或给出与平方根奇性匹配的正则性上界。

**目标 B：不连续函数的 RQMC 收敛阶。** He, Wang, *On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions*, SIAM J. Numer. Anal. 53 (2015) 2488–2503（doi:10.1137/15M1007963；He, arXiv:1702.03361 命题 2.3 有复述），定理 3.5：\(f=g\,\mathbf 1_\Omega\)，\(g\in L^2\) 且 Hardy–Krause 变差有界，\(\partial\Omega\) 有 \((d-1)\) 维 Minkowski 容度时，扰乱 \((t,d)\) 序列估计的方差为 \(O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})\)；定理 3.6：\(\Omega\) 为部分轴平行集合（不平行于坐标轴的维数为 \(d_u\)）时为 \(O(n^{-1-1/(2d_u-1)}(\log n)^{2d/(2d_u-1)})\)。严格改进的例子：对不平行于坐标轴、但有更多结构的边界（例如光滑凸曲面或二次曲面）证明严格更好的方差阶。

**目标 C：随机化算法的查询下界（附录开放问题 4）。** Rademacher, Vempala, *Dispersion of mass and the complexity of randomized geometric algorithms*（Advances in Mathematics 219 (2008)；arXiv:cs/0608054）：任何把凸体体积近似到 \((1+c)\) 倍以内的随机化算法都需要 \(\Omega(n^2/\log n)\) 次成员查询；这个下界对平行多面体 \(\{x:\|Ax\|_\infty\le1\}\) 已经成立，而在该类上 \(O(n^2\log n)\) 次查询就够了；一般凸体当时最好的算法约为 \(n^4\)，猜想为 \(n^3\)；目前一般凸体的上界为 \(\tilde O(d^{7/2}+d^3/\varepsilon^2)\)（arXiv:2610.04987 表 1），下界仍为 \(\tilde\Omega(d^2)\)。严格改进的例子：对本题的封闭跳跃面（例如椭球的高斯质量）证明随机化查询下界并去掉 \(\log\) 因子、或加入对精度的依赖，或给出与下界匹配的上界。

**目标 D：只用点查询选方向的保证（附录开放问题 2）。** Gilbert–Kuo–Sloan（arXiv:2112.11621）结论原文："The paper does not offer guidance on the choice of variable if there is more than one such variable"。Liu, Owen, *Pre-integration via active subspaces*（arXiv:2202.02682）提出用主动子空间（梯度主成分）的第一个特征向量作预积分方向，证明了沿一个坐标预积分不会增加扰乱网估计的方差，并说明了该方向与“最大化线性组合的 Sobol' 指数”这一不可行选择的关系；方向在数值例子中用有限差分梯度估计，没有给出估计误差对积分误差影响的保证。严格改进的例子：只用点查询、只花预算中 \(o(N)\) 的部分选出方向，并对所得积分误差给出可证明的界。

**目标 E：预积分保持混合光滑性的条件（可达参照）。** Griewank, Kuo, Leövey, Sloan, *High dimensional integration of kinks and jumps — smoothing by preintegration*（J. Comput. Appl. Math. 344 (2018)；arXiv:1712.00920），定理 3：在定理 2 的条件下（\(f=\theta\cdot\mathrm{ind}(\phi)\)，\(D_j\phi>0\) 处处成立且 \(x_j\to\infty\) 时 \(\phi\to\infty\)，以及定理 2 中对边界与可积性的条件），若 \(\theta,\phi\) 属于混合光滑空间 \(\mathcal W^{\boldsymbol r}_{d,p,\zeta,\mathrm{mix}}\cap\mathcal C^{\boldsymbol r}_{\mathrm{mix}}\)，且多重指标满足 \(r_j\ge\sum_{k\ne j}r_k\ge1\)，则 \(P_jf\in\mathcal W^{\boldsymbol r_{-j}}_{d-1,p,\zeta}\cap\mathcal C^{\boldsymbol r_{-j}}_{\mathrm{mix}}\)。原文解释这个条件来自证明：\(D^{\boldsymbol\alpha}P_jf\) 中含有依赖 \(D_j^{|\boldsymbol\alpha|}\phi\) 与 \(D_j^{|\boldsymbol\alpha|-1}\theta\) 的项；但没有证明它是必要的，检索到 2026-11 也没有看到放宽它的工作。严格改进的例子：在更弱的条件下证明同样的结论（例如对某个自然的子类，把 \(\sum_{k\ne j}r_k\) 换成更小的量）；或者构造反例，证明这个条件不能再弱，这按“最优性”计为最高档（见“计分”）。

**登记的锚点**（评审时据此比较；编号用于 `manifest.json`；详细陈述以原文为准）：

| 编号 | 结果 |
| --- | --- |
| A-MC | 普通蒙特卡洛：任意 \(f\in L^2\)，随机化均方根误差 \(\sigma(f)n^{-1/2}\)，与维数无关，对 \(L^2\) 类是最优阶 |
| A-GKLS | Griewank–Kuo–Leövey–Sloan 2018 定理 2、3：在已知坐标单调等条件下，预积分保持光滑性（各向同性 \(r\) 阶；混合光滑需 \(r_j\ge\sum_{k\ne j}r_k\)） |
| A-GKS | Gilbert–Kuo–Sloan 定理 1（目标 A） |
| A-HW | He–Wang 2015 定理 3.5、3.6（目标 B）；He 2017（arXiv:1702.03361）对边界奇性的推广 |
| A-LO | Liu–Owen 2023（目标 D）；Liu 2024（arXiv:2212.13232）加入可预积分的约束，方向仍由有限差分梯度估计 |
| A-RV | Rademacher–Vempala 2008 定理 2、23（目标 C） |
| A-BF | Bárány–Füredi 1987 定理 1：确定性多项式次成员查询，体积上下估计之比至少 \((cd/\log d)^d\) |
| A-HNUW | Hinrichs–Novak–Ullrich–Woźniakowski 2014：光滑函数类上确定性积分的维数灾难 |
| A-PW | Plaskota–Wasilkowski 2005、Plaskota–Wasilkowski–Zhao 2008：一维未知奇点的自适应积分与逼近 |

判定复述时还会用到经典的一维结果：Sikorski 1982（二分对求零最优）、Kiefer 1957（单调函数积分的最优误差）、Novak 1996 综述中的阶跃例、Bakhvalov 平均法。

**已经没有空间或已被堵死的方向**：单调性不成立时，“沿单个坐标预积分 + 混合光滑”这条路线被 GKS 定理 1 排除（其他路线不受影响）；对一般超平面跳跃，等权 QMC 的最坏误差已紧（半空间偏差 \(\Theta(n^{-1/2-1/(2d)})\)，Alexander 1990、Matoušek 1995），He–Wang 对指示函数的速率达到这一阶；Rademacher–Shao 2008 证明“乘积集划分”这条去掉 \(\log\) 因子的路线走不通（只排除这一种证明方法）。

要求：

- 在 `THEORY.md` 开头写明所选目标、所对照锚点的精确陈述（与原文核对，写明定理编号），以及你的结论与它的关系；
- **引用的适用性**：每个用到或对照的外部结果，写明出处与版本、定理编号、与你的结论相关的假设和结论、换算到本题的方法，并区分引用的内容与自己的证明。误读或误用引用时，依赖它的对比无效；
- 检索到本节没有、但比所列结果更强的已有人类结果时，在 `THEORY.md` 中写明，并以它为对照；
- 超越目标的结论必须完整通过形式化验证才计分（见“形式化”）。分析类结论在 Mathlib 中可能缺少基础设施，需要自己建立所需的定义与引理，或把结论限制在能完整形式化的范围；
- 没有做到严格改进时如实分类（不可比、相当或更弱），不要夸大。

## 形式化

环境里有离线的 Lean 4（v4.35.0）和预编译的 Mathlib v4.35.0、CSLib v4.35.0：`lean` 和 `lake` 在 PATH 上，工具链在 `/opt/lean`，依赖包在只读的 `/opt/lean-packages`（含源码，可以用 grep 查找已有定义和定理，例如 `/opt/lean-packages/mathlib/Mathlib/Probability/Distributions/Gaussian/`）。也可以经代理使用 Mathlib 在线文档（`https://leanprover-community.github.io/mathlib4_docs/`）以及 Loogle、LeanSearch 等在线检索。

- 把 `/workspace/examples/lean_project/` 复制到 `/workspace/submission/theory/formal/`（`cp -r /workspace/examples/lean_project /workspace/submission/theory/formal && chmod -R u+w /workspace/submission/theory/formal`），在其中工作。`lake build` 编译 `Theory/` 下的所有模块（产物写到 `/tmp/lean-build`，不进提交）；`lake env lean Theory/X.lean` 只编译一个文件。
- `Theory/Statements.lean`：每条要计分的结论的形式化陈述，证明写 `sorry`。审阅者读这个文件（以及它导入的你自己的模块，例如公共定义）来判断形式化陈述是否忠实表达 `THEORY.md` 中的结论，所以它应当可读：不要用自定义记号或宏掩盖含义，尽量使用 Mathlib 已有的概念。
- `Theory/Proofs.lean`：逐字重复这些陈述（同名、同类型）并给出完整证明。它可以导入你自己的其他模块，但不能导入 `Theory.Statements`。
- `manifest.json`：每条结论一项，包括编号、在 `THEORY.md` 中的位置、Lean 定理名（可以多个）、状态（`proved` 已形式化证明，`stated` 只有形式化陈述，`informal` 未形式化）、与人类结果的对比（类别 `stronger`、`weaker`、`equal` 或 `incomparable`，对比的文献，简短理由），以及结果卡片 `coordinates`：类型（`upper`、`lower`、`regularity` 或 `other`）、适用的类、算法用到的已声明输入、结论（界或正则性）、有效范围、保证类型、对照的锚点编号。格式见模板。
- `python3 /workspace/tools/lean_check.py` 运行与评测相同的检查：先 `lake build`，再用 `lake comparator` 判定 manifest 中标为 `proved` 的定理。通过的条件是：`Theory.Proofs` 中的同名定理与 `Theory.Statements` 中的陈述完全一致，通过 Lean 内核检查，并且只用到 `propext`、`Quot.sound`、`Classical.choice` 三个公理（`sorry`、自定义公理和 `native_decide` 都不能通过）。一次检查约需 2–3 分钟。评测时用固定的 lakefile 和依赖版本重新构建，只取 `theory/formal/Theory/` 下的 `.lean` 文件。
- 资源：开发容器共 8 GiB 内存、1 个 CPU。全量 `import Mathlib` 约需 1 分钟，只导入用到的模块会快一些。

## 网络

开发容器可以经代理访问公网，用于查阅文献和资料。环境变量 `HTTPS_PROXY` 和 `HTTP_PROXY` 已设置，`curl`、Python 的 `urllib` 和 `requests` 会自动使用。

- 不可访问：GitHub 及其镜像、包管理源（PyPI、conda 等）、在线大模型服务、内网。被拒绝的请求返回 403。所有访问都会被记录。
- 文献检索可以用 arXiv API（`https://export.arxiv.org/api/query?search_query=...`）、OpenAlex（`https://api.openalex.org/works?search=...`）、Crossref（`https://api.crossref.org/works?query=...`）、zbMATH Open 和 Semantic Scholar（无 key 时限流）。arXiv 论文的 LaTeX 源码可以从 `https://arxiv.org/e-print/<编号>` 下载（gzip 或 tar），比 PDF 更方便阅读。
- 网络只用于查阅资料：不要查找或使用与本评测（FrontierAM）相关的任何材料（题目的生成代码、他人的解答、评审文档），也不要使用任何在线 AI 服务。

## 计分

**正确性**：只有通过检查的结论计分：在 manifest 中标为 `proved`，通过 `lake comparator`，并且 Lean 陈述与 `THEORY.md` 中的结论匹配。偷换了假设、前提不可满足、结论被弱化成平凡命题的，都不算通过；把一个困难结论形式化成较弱的陈述，只按较弱的陈述计分。未形式化的结论可以写进文档作为讨论，但不计分。

**比较协议**：在“假设 × 结论强度”上比较，只比较同一类型的结论（算法保证、下界、正则性）。在同一组精度目标下比较；在双方共同的有效范围上一致比较，到通用常数为止；暂态项、有效性阈值与保证类型（最坏情形、随机化均方根、期望、高概率）都计入；改进不能以同一坐标上的变差作补偿；只改常数算相当；比不清的记为待定，不计分。

**分档**（每条通过检查的结论分档，本题得分取最高一档；没有通过检查的结论时为 0 分）：

| 档 | 条件 | 分 |
| --- | --- | --- |
| 偏序优 | 严格强于某个登记的锚点（假设不更强、结论不更弱，且至少一项严格更好），且不被任何已知人类结果支配；或者严格证明某个登记的结果不能再改进、某个假设不能再弱（例如目标 E 的反例） | 100 |
| 不可比·有价值 | 与已知人类结果都不可比，但回应了人类论文写明的问题（下面 P1–P5 之一） | 50 |
| 不可比·价值低 | 不可比，但不回应这些问题（例如只因限于有限参数族或依赖额外的已知结构而不可比） | 25 |
| 偏序不优 | 被已知人类结果支配或与之等同（复述、特例） | 10 |

人类论文写明的问题（价值偏好）：

- **P1**：单调性不成立时，对自然的类给出快于蒙特卡洛的可证明保证（Gilbert–Kuo–Sloan 结论，不限于预积分路线）；
- **P2**：只用点查询选出预积分方向，并对积分误差给出可证明的界（Gilbert–Kuo–Sloan 结论；Liu–Owen 2023）；
- **P3**：对不轴平行、但有更多结构的边界给出更好的速率，或让间断“对 QMC 友好”的理论（He 2017 结论）；
- **P4**：Rademacher–Vempala 讨论一节的三个开放问题：随机化下界中的 \(\log\) 因子、\(n^3\) 型成员查询下界、分离预言机下更快的算法；
- **P5**：多元分段光滑类、未知间断面上自适应（尤其随机化自适应）的作用（Novak 2024 综述）；归结为一维阶跃例的有限参数族不算。

建议先在 `THEORY.md` 中写清目标陈述、对照的锚点和证明路线，再分步形式化（先证关键引理）。目标应当是本次会话内能够完整证明的最强结论。

## 禁止

- 查找或使用与本评测（FrontierAM）相关的材料；使用在线 AI 服务；读取评测器或环境中与题目无关的信息。

## 工作区

| 路径 | 内容 |
| --- | --- |
| `/workspace/problem.md` | 本题面 |
| `/workspace/THEORY_APPENDIX.md` | 函数类、信息与代价、误差准则、已知结果、开放问题 |
| `/workspace/examples/lean_project/` | Lean 形式化工程模板（`Theory/Statements.lean`、`Theory/Proofs.lean`、`manifest.json` 各有一个示例条目，替换掉） |
| `/workspace/tools/lean_check.py` | 本地运行与评测相同的形式化检查（`--each` 逐条判定） |
