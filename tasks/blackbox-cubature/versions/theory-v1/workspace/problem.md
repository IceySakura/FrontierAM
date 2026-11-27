# 任务：高斯测度下分段光滑函数的黑盒积分（纯理论，theory-v1）

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

## 人类 SOTA 目标

下面四个目标都取自人类论文中的定理，或作者自己写明的空白（出处已核对，可以联网阅读原文）。至少选择一个，尝试给出**严格强于**所列人类结果的结论：在相同或更弱的假设下得到更强的结论（更快的速率、更好的常数、更紧的上下界），或用更弱的假设得到相同的结论；或者在作者写明的空白上取得可证明的进展。

**目标 A：单调性不成立时 QMC 的理论支持（附录开放问题 1）。** Gilbert, Kuo, Sloan, *Preintegration is not smoothing when monotonicity fails*（arXiv:2112.11621），定理 1：\(\varphi\in C^2(\mathbb R^d)\)，若在 \(y^*\) 处 \(\partial\varphi/\partial y_1=0\)、\(\partial^2\varphi/\partial y_1^2\ne0\) 且 \(\nabla\varphi\ne0\)，则沿 \(y_1\) 预积分后的函数在 \(y^*_{-1}\) 处、沿任何不与 \(\nabla_{-1}\varphi(y^*)\) 正交的直线出现平方根型奇性。结论一节原文："because of the proven lack of smoothness, the resulting preintegrated function no longer belongs to the space of \((d-1)\)-variate functions of dominating mixed smoothness of order one, and as a consequence there is at present no theoretical support for our use of QMC integration for this \((d-1)\)-dimensional integral." 进展的例子：对一类自然的非单调跳跃或折点（例如二次曲面），证明预积分（或其他降维、光滑化方式）之后的积分在 QMC 或 RQMC 下有快于蒙特卡洛的收敛保证；或给出与平方根奇性匹配的正则性上界。

**目标 B：不连续函数的 RQMC 收敛阶。** He, Wang, *On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions*, SIAM J. Numer. Anal. 53 (2015) 2488–2503（doi:10.1137/15M1007963；He, arXiv:1702.03361 命题 2.3 有复述），定理 3.5：\(f=g\,\mathbf 1_\Omega\)，\(g\in L^2\) 且 Hardy–Krause 变差有界，\(\partial\Omega\) 有 \((d-1)\) 维 Minkowski 容度时，扰乱 \((t,d)\) 序列估计的方差为 \(O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})\)；定理 3.6：\(\Omega\) 为部分轴平行集合（不平行于坐标轴的维数为 \(d_u\)）时为 \(O(n^{-1-1/(2d_u-1)}(\log n)^{2d/(2d_u-1)})\)。严格改进的例子：对不平行于坐标轴、但有更多结构的边界（例如光滑凸曲面或二次曲面）证明严格更好的方差阶。

**目标 C：随机化算法的查询下界（附录开放问题 4）。** Rademacher, Vempala, *Dispersion of mass and the complexity of randomized geometric algorithms*（Advances in Mathematics 219 (2008)；arXiv:cs/0608054）：任何把凸体体积近似到 \((1+c)\) 倍以内的随机化算法都需要 \(\Omega(n^2/\log n)\) 次成员查询；这个下界对平行多面体 \(\{x:\|Ax\|_\infty\le1\}\) 已经成立，而在该类上 \(O(n^2\log n)\) 次查询就够了；一般凸体当时最好的算法约为 \(n^4\)，猜想为 \(n^3\)。严格改进的例子：对本题的封闭跳跃面（例如椭球的高斯质量）证明随机化查询下界并去掉 \(\log\) 因子、或加入对精度的依赖，或给出与下界匹配的上界。

**目标 D：只用点查询选方向的保证（附录开放问题 2）。** Gilbert–Kuo–Sloan（arXiv:2112.11621）结论原文："The paper does not offer guidance on the choice of variable if there is more than one such variable"。Liu, Owen, *Pre-integration via active subspaces*（arXiv:2202.02682）提出用主动子空间（梯度主成分）的第一个特征向量作预积分方向，证明了沿一个坐标预积分不会增加扰乱网估计的方差，并说明了该方向与“最大化线性组合的 Sobol' 指数”这一不可行选择的关系；方向在数值例子中用有限差分梯度估计，没有给出估计误差对积分误差影响的保证。严格改进的例子：只用点查询、只花预算中 \(o(N)\) 的部分选出方向，并对所得积分误差给出可证明的界。

要求：

- 在 `THEORY.md` 开头写明所选目标、所对照人类结果的精确陈述（与原文核对，写明定理编号），以及你的结论与它的关系；
- 超越目标的结论必须完整通过形式化验证才计分（见“形式化”）。分析类结论在 Mathlib 中可能缺少基础设施，需要自己建立所需的定义与引理，或把结论限制在能完整形式化的范围；
- 没有做到严格改进时如实分类（不可比、相当或更弱），不要夸大。

## 形式化

环境里有离线的 Lean 4（v4.35.0）和预编译的 Mathlib v4.35.0、CSLib v4.35.0：`lean` 和 `lake` 在 PATH 上，工具链在 `/opt/lean`，依赖包在只读的 `/opt/lean-packages`（含源码，可以用 grep 查找已有定义和定理，例如 `/opt/lean-packages/mathlib/Mathlib/Probability/Distributions/Gaussian/`）。也可以经代理使用 Mathlib 在线文档（`https://leanprover-community.github.io/mathlib4_docs/`）以及 Loogle、LeanSearch 等在线检索。

- 把 `/workspace/examples/lean_project/` 复制到 `/workspace/submission/theory/formal/`（`cp -r /workspace/examples/lean_project /workspace/submission/theory/formal && chmod -R u+w /workspace/submission/theory/formal`），在其中工作。`lake build` 编译 `Theory/` 下的所有模块（产物写到 `/tmp/lean-build`，不进提交）；`lake env lean Theory/X.lean` 只编译一个文件。
- `Theory/Statements.lean`：每条要计分的结论的形式化陈述，证明写 `sorry`。审阅者读这个文件（以及它导入的你自己的模块，例如公共定义）来判断形式化陈述是否忠实表达 `THEORY.md` 中的结论，所以它应当可读：不要用自定义记号或宏掩盖含义，尽量使用 Mathlib 已有的概念。
- `Theory/Proofs.lean`：逐字重复这些陈述（同名、同类型）并给出完整证明。它可以导入你自己的其他模块，但不能导入 `Theory.Statements`。
- `manifest.json`：每条结论一项，包括编号、在 `THEORY.md` 中的位置、Lean 定理名（可以多个）、状态（`proved` 已形式化证明，`stated` 只有形式化陈述，`informal` 未形式化）、与人类结果的对比（类别 `stronger`、`weaker`、`equal` 或 `incomparable`，对比的文献，简短理由）。格式见模板。
- `python3 /workspace/tools/lean_check.py` 运行与评测相同的检查：先 `lake build`，再用 `lake comparator` 判定 manifest 中标为 `proved` 的定理。通过的条件是：`Theory.Proofs` 中的同名定理与 `Theory.Statements` 中的陈述完全一致，通过 Lean 内核检查，并且只用到 `propext`、`Quot.sound`、`Classical.choice` 三个公理（`sorry`、自定义公理和 `native_decide` 都不能通过）。一次检查约需 2–3 分钟。评测时用固定的 lakefile 和依赖版本重新构建，只取 `theory/formal/Theory/` 下的 `.lean` 文件。
- 资源：开发容器共 8 GiB 内存、1 个 CPU。全量 `import Mathlib` 约需 1 分钟，只导入用到的模块会快一些。

## 网络

开发容器可以经代理访问公网，用于查阅文献和资料。环境变量 `HTTPS_PROXY` 和 `HTTP_PROXY` 已设置，`curl`、Python 的 `urllib` 和 `requests` 会自动使用。

- 不可访问：GitHub 及其镜像、包管理源（PyPI、conda 等）、在线大模型服务、内网。被拒绝的请求返回 403。所有访问都会被记录。
- 文献检索可以用 arXiv API（`https://export.arxiv.org/api/query?search_query=...`）、OpenAlex（`https://api.openalex.org/works?search=...`）、Crossref（`https://api.crossref.org/works?query=...`）、zbMATH Open 和 Semantic Scholar（无 key 时限流）。arXiv 论文的 LaTeX 源码可以从 `https://arxiv.org/e-print/<编号>` 下载（gzip 或 tar），比 PDF 更方便阅读。
- 网络只用于查阅资料：不要查找或使用与本评测（FrontierAM）相关的任何材料（题目的生成代码、他人的解答、评审文档），也不要使用任何在线 AI 服务。

## 计分

- 只有形式化验证通过的结论计分：在 manifest 中标为 `proved`，通过 `lake comparator`，并且审阅者认定它的 Lean 陈述忠实表达了 `THEORY.md` 中的结论。偷换了假设、前提不可满足、结论被弱化成平凡命题的，都不算通过；把一个困难结论形式化成较弱的陈述，只按较弱的陈述计分。
- 每条通过的结论按它与最强可比人类结果的关系定分（0–100）：与题目所属问题上的人类强结果相当为 50 分；严格更强的在 50 分以上，按突破的幅度和意义定分，对重要已知结果的显著突破为 100 分；严格更弱的在 50 分以下；不可比的由审阅者依据人类论文自身表达的观点定分。复现经典结果或只在很窄的参数特例上成立的结论低于 50 分。你给出的分类和理由是审阅的出发点，审阅者会独立核对。
- 对“人类 SOTA 目标”中所列人类结果的严格改进按“严格更强”定分（50 分以上）；在目标中作者写明的空白上取得可证明的进展，即使与原定理不可比，也可以高于 50 分。
- 本题得分取所有通过结论中的最高分；没有通过的结论时为 0 分。未形式化的结论可以写进文档作为讨论，但不计分。
- 因为只看最强的一条，一条严格强于人类已知结果的结论，比多条初等结论更有价值。建议先在 `THEORY.md` 中写清目标陈述、要比较的人类结果和证明路线，再分步形式化（先证关键引理；Mathlib 缺少的基础可以自己定义并证明）。目标应当是本次会话内能够完整证明的最强结论，而不是无法完成的宏大猜想。

## 禁止

- 查找或使用与本评测（FrontierAM）相关的材料；使用在线 AI 服务；读取评测器或环境中与题目无关的信息。

## 工作区

| 路径 | 内容 |
| --- | --- |
| `/workspace/problem.md` | 本题面 |
| `/workspace/THEORY_APPENDIX.md` | 函数类、信息与代价、误差准则、已知结果、开放问题 |
| `/workspace/examples/lean_project/` | Lean 形式化工程模板（`Theory/Statements.lean`、`Theory/Proofs.lean`、`manifest.json` 各有一个示例条目，替换掉） |
| `/workspace/tools/lean_check.py` | 本地运行与评测相同的形式化检查（`--each` 逐条判定） |
