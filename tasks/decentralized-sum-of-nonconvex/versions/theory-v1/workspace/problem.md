# 任务：去中心化有限和 sum-of-nonconvex 强凸复合优化的迭代复杂度（纯理论，theory-v1）

本题只考理论：推进一个优化复杂度问题的理论，目标是超越人类已有的最好结果。不需要写程序，也不评测程序；会话里的全部时间都用于研究、证明和 Lean 形式化。只有通过形式化检查、且形式化陈述忠实表达文档结论的结果才计分；每条结论都要与强的人类已知结果对比。

会话时长见启动说明（本题为 6 小时）。提前结束不加分：剩余的时间应当用来把一条结论做得更强，或把它完整形式化。开发过程中没有评测反馈，但可以在本地运行与评测相同的形式化检查。

## 交付

所有交付物写入 `/workspace/submission/`：

| 文件 | 内容 |
| --- | --- |
| `theory/THEORY.md` | 理论文档（Markdown，数学公式用 LaTeX），要求见“理论文档” |
| `theory/formal/` | Lean 4 形式化工程（从 `/workspace/examples/lean_project/` 开始），要求见“形式化” |

评测只收集 `/workspace/submission/`；不要把交付物写在别处。

## 问题

问题是 OptBound 基准的 ULB-profile-63，原文见 `/workspace/PROFILE.md`，它是本题的一部分。简述：\(n\) 个节点组成静态网络，节点 \(i\) 私有 \(m\) 个分量 \(F_i(\cdot;\xi)\)，每个分量 \(L\)-光滑但可以非凸；\(f=\frac1n\sum_i\frac1m\sum_\xi F_i(\cdot;\xi)\) 凸，共享正则项 \(h\) 可精确求 prox，\(\phi=f+h\) 为 \(\mu\)-强凸，\(\kappa=L/\mu\)，初始间隙至多 \(\Delta\)。每个节点每次迭代至多做一次分量梯度查询、一次 prox 查询和一轮通信（通信矩阵 \(W\in\mathcal W_{n,\rho}\)：非负、对称、行和为 1、\(\|W-J\|_2\le\rho\)；节点只拿到加权和）。算法是随机化、zero-respecting 的，复杂度是平均输出达到 \(\mathbb E[\phi(\widehat x_t)-\phi^\star]\le\varepsilon\) 所需的迭代数 \(T_\varepsilon\)。要确定

\[
\inf_{A\in\mathcal A}\ \sup_{\text{实例}}\ \sup_{W\in\mathcal W_{n,\rho}}\ T_\varepsilon
\]

对 \(n,m,L,\mu,\Delta,\rho,\varepsilon\) 的阶：给出上界（构造算法）和下界（对每个算法构造困难实例），最好互相匹配（至多差对数因子）。

**形式化模型是固定的**：`/workspace/examples/lean_project/Theory/Model.lean` 把题面逐条写成 Lean 定义（问题类 `Instance.InClass`、通信类 `CommClass`、算法类 `Algorithm.Admissible`、首达时间 `Algorithm.hittingTime`、目标量 `minimax`），`Theory/Sanity.lean` 证明了这些定义的基本性质。条款与定义的对应、以及题面没写明之处的解释，见 `/workspace/MODEL_NOTES.md`。关于本问题的结论必须用这些定义陈述；这两个文件在评测时用原件替换，修改它们没有作用。

## 理论文档

`theory/THEORY.md` 应包含：

1. **结论与证明**：在明确写出的假设下的定理：上界（算法及其迭代复杂度）、下界（困难实例的构造）、匹配的阶，或者它们在某个参数范围或某个 `PROFILE.md` 第 3 节所列替代设定下的版本。每条结论编号（R1、R2、……），给出完整的数学证明和对应的 Lean 定理名；未证明的标为“猜想”。常数可以不显式，但要写清对每个参数的依赖（`PROFILE.md` 第 2 节要求保留所有参数的依赖）。
2. **与人类已知结果对比**：对每条结论，找出一个或多个最强的可比人类结果（下节所列文献或你检索到的其他文献；写明可核对的出处，例如 arXiv 编号或 DOI，以及定理编号，并对照原文准确复述其假设与结论，包括记号和计数口径的换算），并归入一类：
   - **严格更强**：在相同或更弱的假设下得到严格更好的结论（更紧的上界或下界、更好的参数依赖、更大的类），或用更弱的假设得到相同结论，即突破了该人类结果；
   - **严格更弱**：结论被某个已知结果蕴含，且后者严格更强；
   - **相当**：与已知结果等价（复现）；
   - **不可比**：各有所长。说明得失，并依据人类论文自身表达的观点（它们指出的局限、提出的开放问题、关心的参数范围）论证这条结论的价值。

## 人类 SOTA 目标

以下是我们核对过原文的人类结果（可以联网阅读原文；你仍需自己对照原文复述）。按这些文献，本题在逐分量光滑下的上下界之间还有差距。

**上界**：Liu, Low, *Decentralized Sum-of-Nonconvex Optimization*（AAAI 2024；arXiv:2402.02356v1），定理 5.1 与推论 5.3（算法 PMGT-KatyushaX）。原文记号与本题不同：原文有 \(m\) 个节点、每个节点 \(n\) 个分量 \(f_{i,j}\)，每个分量 \((\ell_1,\ell_2)\)-光滑，\(f\) 凸且 \(L\)-光滑，\(F=f+\psi\) 为 \(\sigma\)-强凸，\(W\) 对称双随机、\(0\preceq W\preceq I\)、\(\mathrm{null}(I-W)=\mathrm{span}(\mathbf 1)\)。推论 5.3：得到 \(\epsilon\)-近似解需要
\(O\big((n+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}n^{3/4})\log\frac{F(\bar y^0)-F(x^*)}{\epsilon}\big)\) 次 SFO 调用和
\(\tilde O\big(\frac{\sqrt n+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}n^{1/4}}{\sqrt{1-\lambda_2(W)}}\log\frac{F(\bar y^0)-F(x^*)}{\epsilon}\big)\) 轮通信。取 \(\ell_1=\ell_2=L\) 并换成本题记号（\(n\) 个节点、每个节点 \(m\) 个分量），分量梯度约为 \(O((m+m^{3/4}\sqrt\kappa)\log(\Delta/\varepsilon))\)，通信约为 \(\tilde O((\sqrt m+m^{1/4}\sqrt\kappa)(1-\lambda_2(W))^{-1/2}\log(\Delta/\varepsilon))\)；计数口径（每个节点还是合计、与本题“每次迭代至多一次梯度、一次 prox、一轮通信”的换算、对 \(W\) 的不同假设）请对照原文核实。OptBound 把本题的上界记为这篇文献，把下界记为未知。

**下界**：按我们核对的文献，没有针对本题类的下界结果。可以直接搬用的经典下界：

- Woodworth, Srebro, *Tight Complexity Bounds for Optimizing Composite Objectives*（NeurIPS 2016；arXiv:1605.08003），定理 8：对任意随机化算法（每次查询得到一个分量的值、梯度和 prox），存在 \(m'\) 个 \(\gamma\)-光滑、\(\lambda\)-强凸的分量（要求 \(\gamma/\lambda>161m'\)），使达到 \(\mathbb E[F(\hat x)-F^*]<\epsilon\) 至少需要 \(\Omega\big(m'+\sqrt{m'\gamma/\lambda}\,\log(\frac{\epsilon_0}{\epsilon}\sqrt{m'\lambda/\gamma})\big)\) 次查询。凸分量属于本题的类，所有节点合计 \(N=nm\) 个分量。
- Scaman, Bach, Bubeck, Lee, Massoulié, *Optimal Algorithms for Smooth and Strongly Convex Distributed Optimization in Networks*（ICML 2017；arXiv:1702.08704），定理 2：存在谱隙为 \(\gamma\) 的 gossip 矩阵和 \(\alpha\)-强凸、\(\beta\)-光滑的局部函数，使任何使用该矩阵的黑盒算法满足 \(\bar f(\theta_{i,t})-\bar f(\theta^*)\ge\frac{3\alpha}{2}(1-\frac{16}{\sqrt{\kappa_l}})^{1+t/(1+\tau/(5\sqrt\gamma))}\|\theta_{i,0}-\theta^*\|^2\)（\(\tau\) 为一次通信的时间）。证明中节点数依 \(\gamma\) 选取，而本题 \(n\) 是固定参数（`MODEL_NOTES.md` 第 2 节第 7 条）。
- Zhou, Gu, *Lower Bounds for Smooth Nonconvex Finite-Sum Optimization*（ICML 2019；arXiv:1901.11224），定理 4.1：sum-of-nonconvex 且 \(F\) 为 \(\sigma\)-强凸时，linear-span 随机化算法需要 \(\Omega\big(n+n^{3/4}\sqrt{L/\sigma}\log\frac{n\sigma\Delta}{L\epsilon}\big)\) 次分量查询。但该定理只假设分量平均光滑（\(\mathbb E_i\|\nabla f_i(x)-\nabla f_i(y)\|^2\le L^2\|x-y\|^2\)），弱于本题的逐分量 \(L\)-光滑，其困难实例不在本题的类中。该文第 4.3 节指出 KatyushaX 等上界是在逐分量光滑下证明的，结论一节把“每个 \(f_i\) 为 \((l,L)\)-光滑”的情形列为未来工作。
- 相近问题的现状：Zhou, *Near-Optimal Complexity of Finding Approximate Local Minima of Nonconvex Finite Sums*（arXiv:2611.05807，2026 年 11 月）的摘要写道 "Under individual smoothness, a factor of \(n^{1/4}\) remains"。

**目标**（至少选一个，尝试给出严格强于所列人类结果的结论）：

- **A（下界）**：在本题的逐分量 \(L\)-光滑类中，证明严格强于上述经典结果直接推出的下界，例如数据项超过 \(\sqrt{N\kappa}\) 型，或网络项与数据项同时出现的下界。
- **B（上界）**：在本题的迭代计数方式下，给出在某个参数范围内严格优于 Liu–Low 的上界。
- **C（匹配）**：在某个参数范围内给出匹配（至多差对数因子）的上下界，即在该范围内解决 OptBound 的问题。

要求：

- 在 `THEORY.md` 开头写明所选目标、所对照人类结果的精确陈述（与原文核对，写明定理编号），以及你的结论与它的关系；
- 如果你检索到本节没有列出、但比所列结果更强的已有人类结果，在 `THEORY.md` 中写明，并以它为对照分类；
- 超越目标的结论必须完整通过形式化验证才计分（见“形式化”）。可以先在替代设定（`PROFILE.md` 第 3 节，例如加上凸分量的假设）或较窄的参数范围内完成，这样的结论按它实际陈述的设定对比；
- 没有做到严格改进时如实分类（不可比、相当或更弱），不要夸大。

## 形式化

环境里有离线的 Lean 4（v4.35.0）和预编译的 Mathlib v4.35.0、CSLib v4.35.0：`lean` 和 `lake` 在 PATH 上，工具链在 `/opt/lean`，依赖包在只读的 `/opt/lean-packages`（含源码，可以用 grep 查找已有定义和定理，例如 `/opt/lean-packages/mathlib/Mathlib/Analysis/Convex/`）。也可以经代理使用 Mathlib 在线文档（`https://leanprover-community.github.io/mathlib4_docs/`）以及 Loogle、LeanSearch 等在线检索。

- 把 `/workspace/examples/lean_project/` 复制到 `/workspace/submission/theory/formal/`（`cp -r /workspace/examples/lean_project /workspace/submission/theory/formal && chmod -R u+w /workspace/submission/theory/formal`），在其中工作。`lake build` 编译 `Theory/` 下的所有模块（产物写到 `/tmp/lean-build`，不进提交）；`lake env lean Theory/X.lean` 只编译一个文件。
- `Theory/Model.lean`、`Theory/Sanity.lean`：固定的问题模型及其性质，不要修改（评测时用原件替换）。
- `Theory/Statements.lean`：每条要计分的结论的形式化陈述，证明写 `sorry`。关于本问题的结论用 `Theory/Model.lean` 的定义陈述：下界对每个 `Algorithm.Admissible` 的算法给出实例使 `hittingTime` 足够大（或直接下界 `minimax`），上界给出一个 `Admissible` 的算法并上界它在类中所有实例上的 `hittingTime`。替代设定可以作为额外假设写进陈述。审阅者读这个文件（以及它导入的你自己的模块）来判断形式化陈述是否忠实表达 `THEORY.md` 中的结论，所以它应当可读：不要用自定义记号或宏掩盖含义。
- `Theory/Proofs.lean`：逐字重复这些陈述（同名、同类型）并给出完整证明。它可以导入你自己的其他模块和 `Theory.Sanity`，但不能导入 `Theory.Statements`。
- `manifest.json`：每条结论一项，包括编号、在 `THEORY.md` 中的位置、Lean 定理名（可以多个）、状态（`proved` 已形式化证明，`stated` 只有形式化陈述，`informal` 未形式化）、与人类结果的对比（类别 `stronger`、`weaker`、`equal` 或 `incomparable`，对比的文献，简短理由）。格式见模板。
- `python3 /workspace/tools/lean_check.py` 运行与评测相同的检查：先核对固定文件未被修改，再 `lake build`，再用 `lake comparator` 判定 manifest 中标为 `proved` 的定理。通过的条件是：`Theory.Proofs` 中的同名定理与 `Theory.Statements` 中的陈述完全一致，通过 Lean 内核检查，并且只用到 `propext`、`Quot.sound`、`Classical.choice` 三个公理（`sorry`、自定义公理和 `native_decide` 都不能通过）。一次检查约需 3–5 分钟（模型导入完整的 Mathlib）。评测时用固定的 lakefile 和依赖版本重新构建，只取 `theory/formal/Theory/` 下的 `.lean` 文件。
- 资源：开发容器共 8 GiB 内存、1 个 CPU。

## 网络

开发容器可以经代理访问公网，用于查阅文献和资料。环境变量 `HTTPS_PROXY` 和 `HTTP_PROXY` 已设置，`curl`、Python 的 `urllib` 和 `requests` 会自动使用。

- 不可访问：GitHub 及其镜像、包管理源（PyPI、conda 等）、在线大模型服务、内网，以及 OptBound 网站。被拒绝的请求返回 403。所有访问都会被记录。
- 文献检索可以用 arXiv API（`https://export.arxiv.org/api/query?search_query=...`，有频率限制）、OpenAlex（`https://api.openalex.org/works?search=...`）、Crossref（`https://api.crossref.org/works?query=...`）、zbMATH Open 和 Semantic Scholar（无 key 时限流）。arXiv 论文的 LaTeX 源码可以从 `https://arxiv.org/e-print/<编号>` 下载（gzip 或 tar），摘要页是 `https://arxiv.org/abs/<编号>`。
- 网络只用于查阅资料：不要查找或使用与本评测（FrontierAM）或 OptBound 基准相关的任何材料（题目已在工作区；不要找他人对本题的解答、评审或讨论），也不要使用任何在线 AI 服务。

## 计分

- 只有形式化验证通过的结论计分：在 manifest 中标为 `proved`，通过 `lake comparator`，并且审阅者认定它的 Lean 陈述忠实表达了 `THEORY.md` 中的结论。偷换了假设、前提不可满足、结论被弱化成平凡命题的，都不算通过；把一个困难结论形式化成较弱的陈述，只按较弱的陈述计分。
- 每条通过的结论按它与最强可比人类结果的关系定分（0–100）：与题目所属问题上的人类强结果相当为 50 分；严格更强的在 50 分以上，按突破的幅度和意义定分，对重要已知结果的显著突破为 100 分；严格更弱的在 50 分以下；不可比的由审阅者依据人类论文自身表达的观点定分。复现经典结果或只在很窄的参数特例上成立的结论低于 50 分。你给出的分类和理由是审阅的出发点，审阅者会独立核对。
- 对“人类 SOTA 目标”中所列人类结果的严格改进按“严格更强”定分（50 分以上）；在作者写明的空白（例如逐分量光滑下的下界）上取得可证明的进展，即使与原定理不可比，也可以高于 50 分。
- 本题得分取所有通过结论中的最高分；没有通过的结论时为 0 分。未形式化的结论可以写进文档作为讨论，但不计分。
- 因为只看最强的一条，一条严格强于人类已知结果的结论，比多条初等结论更有价值。建议先在 `THEORY.md` 中写清目标陈述、要比较的人类结果和证明路线，再分步形式化（先证关键引理）。目标应当是本次会话内能够完整证明的最强结论，而不是无法完成的宏大猜想。

## 禁止

- 查找或使用与本评测（FrontierAM）或 OptBound 基准相关的材料；使用在线 AI 服务；读取评测器或环境中与题目无关的信息。

## 工作区

| 路径 | 内容 |
| --- | --- |
| `/workspace/problem.md` | 本题面 |
| `/workspace/PROFILE.md` | OptBound ULB-profile-63 原文：问题、任务要求、替代设定、相关文献 |
| `/workspace/MODEL_NOTES.md` | 形式化模型说明：题面条款与 Lean 定义的对应、题面没写明之处的解释 |
| `/workspace/examples/lean_project/` | Lean 形式化工程模板：固定的 `Theory/Model.lean`、`Theory/Sanity.lean`，以及 `Theory/Statements.lean`、`Theory/Proofs.lean`、`manifest.json` 各一个示例条目（替换掉） |
| `/workspace/tools/lean_check.py` | 本地运行与评测相同的形式化检查（`--each` 逐条判定） |
