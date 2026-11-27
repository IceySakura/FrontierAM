# 任务：去中心化有限和 sum-of-nonconvex 强凸复合优化的迭代复杂度（纯理论，theory-v2）

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

## 人类 SOTA 目标与登记的锚点

下表是我们核对过原文、在评审前登记的人类结果（锚点；编号用于 `manifest.json`）。可以联网阅读原文，但你仍需自己对照原文复述。按这些文献，本题在逐分量光滑下仍有差距；有两个方向已经没有空间（见表后）。

| 编号 | 类型 | 出处 | 换算到本题模型（迭代数 \(T\)，\(N=nm\)，\(R=\Delta/\varepsilon\)） |
| --- | --- | --- | --- |
| H-UB | 上界 | Liu–Low（AAAI 2024；arXiv:2402.02356v1）定理 5.1、推论 5.3（PMGT-KatyushaX）。原文有 \(m\) 个节点、每节点 \(n\) 个 \((\ell_1,\ell_2)\)-光滑分量，\(W\) 对称双随机且 \(0\preceq W\preceq I\)；SFO 调用 \(O((n+n^{3/4}(L^{1/2}+(\ell_1\ell_2)^{1/4})/\sqrt\sigma)\log)\)，通信 \(\tilde O((\sqrt n+n^{1/4}(\cdot)/\sqrt\sigma)/\sqrt{1-\lambda_2(W)}\log)\) | \(T=\tilde O\big((m+m^{3/4}\sqrt\kappa+(\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2})\log R\big)\)。换算：\(\ell_1=\ell_2=L\)，非半正定的 \(W\) 用 \((I+W)/2\)，梯度调用与通信在同一迭代内进行；计数口径请对照原文核实 |
| H-LB-data | 下界 | Woodworth–Srebro（NeurIPS 2016；arXiv:1605.08003）定理 8：任意随机算法（查询得到分量的值、梯度与 prox），\(m'\) 个 \(\gamma\)-光滑、\(\lambda\)-强凸分量，\(\gamma/\lambda>161m'\)，需要 \(\Omega(m'+\sqrt{m'\gamma/\lambda}\log(\frac{\epsilon_0}{\epsilon}\sqrt{m'\lambda/\gamma}))\) 次查询 | 集中式模拟（全部节点在 \(T\) 次迭代中至多 \(nT\) 次分量查询）：\(\kappa>161N\) 时 \(T\ge c\,(m+\sqrt{m\kappa/n}\log(R\sqrt{N/\kappa}))\) |
| H-LB-pl | 下界 | Peng–Tang–Jia（arXiv:2609.00045，2026）：逐分量光滑、随机化 IFO 算法；\(3\le\kappa\le\sqrt n\) 时 \(\Omega(n+n\log(\Delta/\varepsilon)/[1+\log(\sqrt n/\kappa)])\)，困难实例平均强凸、分量非凸。该文 \(\kappa\ge\sqrt n\) 区间的下界只针对 PL 类，不在本题的类中 | \(3\le\kappa\le\sqrt N\) 时 \(T\ge c\,(m+m\log R/[1+\log(\sqrt N/\kappa)])\) |
| H-LB-net | 下界 | Scaman–Bach–Bubeck–Lee–Massoulié（ICML 2017；arXiv:1702.08704）定理 1（任意直径为 \(D\) 的图、黑盒方法）与定理 2（gossip、谱隙 \(\gamma\)，证明中节点数依 \(\gamma\) 选取） | 固定 \(n\) 时按两条定理合并的标准换算：\(T\ge c\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log R\) |
| H-LB-avg | 下界 | Zhou–Gu（ICML 2019；arXiv:1901.11224）定理 4.1：sum-of-nonconvex、\(\sigma\)-强凸，linear-span 随机算法，\(\Omega(n'+n'^{3/4}\sqrt{L/\sigma}\log\frac{n'\sigma\Delta}{L\epsilon})\)。只假设分量平均光滑，困难实例不在本题的类中；该文第 4.3 节与结论一节把逐分量光滑列为未来工作 | 只对平均光滑的类：\(T\ge c\,(m+m^{3/4}n^{-1/4}\sqrt\kappa\log(\cdot))\) |

相近问题的现状：Zhou（arXiv:2611.05807，2026-11）的摘要写道 "Under individual smoothness, a factor of \(n^{1/4}\) remains"。

**已经没有空间的方向**（证明它们只算复述）：\(\kappa\le\sqrt N\) 区间的数据项下界（由 H-LB-pl 经集中式模拟得到）；网络项下界 H-LB-net。

**目标**（至少选一个）：

- **A（下界）**：在本题的逐分量 \(L\)-光滑类、\(\kappa\ge\sqrt N\) 区间，证明严格强于 H-LB-data 的数据项下界（例如超过 \(\sqrt{m\kappa/n}\) 型）。
- **B（上界）**：在本题的迭代计数方式下，给出处处不差于 H-UB、并在某个参数范围严格更好的上界（数据项或通信项）。
- **C（匹配）**：在某个参数范围内给出匹配（至多差对数因子）的上下界。
- **D（最优性）**：严格证明某个登记的界不能再改进，或某个假设不能再弱（给出反例），证明覆盖相应的度量与算法类。

要求：

- 在 `THEORY.md` 开头写明所选目标、所对照锚点的精确陈述（与原文核对，写明定理编号），以及你的结论与它的关系；
- **引用的适用性**：每个用到或对照的外部结果，写明出处与版本、定理编号、与你的结论相关的假设和结论、换算到本题的方法，并区分引用的内容与自己的证明。误读或误用引用时，依赖它的对比无效；
- 检索到本表没有、但比表中结果更强的已有人类结果时，在 `THEORY.md` 中写明，并以它为对照；
- 超越目标的结论必须完整通过形式化验证才计分（见“形式化”）。可以先在替代设定（`PROFILE.md` 第 3 节）或较窄的参数范围内完成，这样的结论按它实际陈述的设定对比；
- 没有做到严格改进时如实分类，不要夸大。

## 形式化

环境里有离线的 Lean 4（v4.35.0）和预编译的 Mathlib v4.35.0、CSLib v4.35.0：`lean` 和 `lake` 在 PATH 上，工具链在 `/opt/lean`，依赖包在只读的 `/opt/lean-packages`（含源码，可以用 grep 查找已有定义和定理，例如 `/opt/lean-packages/mathlib/Mathlib/Analysis/Convex/`）。也可以经代理使用 Mathlib 在线文档（`https://leanprover-community.github.io/mathlib4_docs/`）以及 Loogle、LeanSearch 等在线检索。

- 把 `/workspace/examples/lean_project/` 复制到 `/workspace/submission/theory/formal/`（`cp -r /workspace/examples/lean_project /workspace/submission/theory/formal && chmod -R u+w /workspace/submission/theory/formal`），在其中工作。`lake build` 编译 `Theory/` 下的所有模块（产物写到 `/tmp/lean-build`，不进提交）；`lake env lean Theory/X.lean` 只编译一个文件。
- `Theory/Model.lean`、`Theory/Sanity.lean`：固定的问题模型及其性质，不要修改（评测时用原件替换）。
- `Theory/Statements.lean`：每条要计分的结论的形式化陈述，证明写 `sorry`。关于本问题的结论用 `Theory/Model.lean` 的定义陈述：下界对每个 `Algorithm.Admissible` 的算法给出实例使 `hittingTime` 足够大（或直接下界 `minimax`），上界给出一个 `Admissible` 的算法并上界它在类中所有实例上的 `hittingTime`。替代设定可以作为额外假设写进陈述。审阅者读这个文件（以及它导入的你自己的模块）来判断形式化陈述是否忠实表达 `THEORY.md` 中的结论，所以它应当可读：不要用自定义记号或宏掩盖含义。
- `Theory/Proofs.lean`：逐字重复这些陈述（同名、同类型）并给出完整证明。它可以导入你自己的其他模块和 `Theory.Sanity`，但不能导入 `Theory.Statements`。
- `manifest.json`：每条结论一项，包括编号、在 `THEORY.md` 中的位置、Lean 定理名（可以多个）、状态（`proved` 已形式化证明，`stated` 只有形式化陈述，`informal` 未形式化）、与人类结果的对比（类别 `stronger`、`weaker`、`equal` 或 `incomparable`，对比的文献，简短理由），以及结果卡片 `coordinates`：类型（`upper`、`lower` 或 `other`）、适用的类、算法用到的已声明输入、界、有效范围、保证类型、对照的锚点编号，和 `rows`（“计分”一节各行的比较：`+`、`=`、`-` 或 `none`）。格式见模板。
- `python3 /workspace/tools/lean_check.py` 运行与评测相同的检查：先核对固定文件未被修改，再 `lake build`，再用 `lake comparator` 判定 manifest 中标为 `proved` 的定理。通过的条件是：`Theory.Proofs` 中的同名定理与 `Theory.Statements` 中的陈述完全一致，通过 Lean 内核检查，并且只用到 `propext`、`Quot.sound`、`Classical.choice` 三个公理（`sorry`、自定义公理和 `native_decide` 都不能通过）。一次检查约需 3–5 分钟（模型导入完整的 Mathlib）。评测时用固定的 lakefile 和依赖版本重新构建，只取 `theory/formal/Theory/` 下的 `.lean` 文件。
- 资源：开发容器共 8 GiB 内存、1 个 CPU。

## 网络

开发容器可以经代理访问公网，用于查阅文献和资料。环境变量 `HTTPS_PROXY` 和 `HTTP_PROXY` 已设置，`curl`、Python 的 `urllib` 和 `requests` 会自动使用。

- 不可访问：GitHub 及其镜像、包管理源（PyPI、conda 等）、在线大模型服务、内网，以及 OptBound 网站。被拒绝的请求返回 403。所有访问都会被记录。
- 文献检索可以用 arXiv API（`https://export.arxiv.org/api/query?search_query=...`，有频率限制）、OpenAlex（`https://api.openalex.org/works?search=...`）、Crossref（`https://api.crossref.org/works?query=...`）、zbMATH Open 和 Semantic Scholar（无 key 时限流）。arXiv 论文的 LaTeX 源码可以从 `https://arxiv.org/e-print/<编号>` 下载（gzip 或 tar），摘要页是 `https://arxiv.org/abs/<编号>`。
- 网络只用于查阅资料：不要查找或使用与本评测（FrontierAM）或 OptBound 基准相关的任何材料（题目已在工作区；不要找他人对本题的解答、评审或讨论），也不要使用任何在线 AI 服务。

## 计分

**正确性**：只有通过检查的结论计分：在 manifest 中标为 `proved`，通过 `lake comparator`，并且 Lean 陈述与 `THEORY.md` 中的结论匹配。偷换了假设、前提不可满足、结论被弱化成平凡命题的，都不算通过；把一个困难结论形式化成较弱的陈述，只按较弱的陈述计分。未形式化的结论可以写进文档作为讨论，但不计分。

**比较协议**：在“假设 × 结论强度”上比较。在同一组精度目标下比较；在双方共同的有效范围上一致比较，到通用常数为止（锚点中写作 \(\tilde O\) 的项忽略多对数因子）；暂态项、有效性阈值与保证类型都计入；在同一行内，改进不能以变差作补偿；只改常数算相当；比不清的记为待定，不计分。

本题计算两个得分，取高者。

**1. 前沿分档 \(T\)**：每条结论分档，取最高一档。

| 档 | 条件 | 分 |
| --- | --- | --- |
| 偏序优 | 严格强于某个登记的锚点（假设不更强、界不更弱，且至少一项严格更好），且不被任何已知人类结果支配；或者严格证明某个登记的界不能再改进、某个假设不能再弱 | 100 |
| 不可比·有价值 | 与已知人类结果都不可比，但回应了人类论文写明的问题：逐分量光滑下的数据项下界（Zhou–Gu 2019、Zhou 2026）；通信项（Liu–Low 2024 结论）；标准问题上的匹配上下界（`PROFILE.md` 第 2、3 节） | 50 |
| 不可比·价值低 | 不可比，但不回应这些问题 | 25 |
| 偏序不优 | 被已知人类结果支配或与之等同（复述、特例） | 10 |

**2. 分行得分 \(R\)**：对整份提交，逐行取通过检查的结论在这一行的最好比较。改进（同一行内没有补偿性的变差）或证明最优，得满分；相当或公认的权衡，得一半；更差、未涉及或待定，得 0。上界行与资源行的结论必须来自同一个算法和同一套调参。

| 行 | 对照 | 满分 |
| --- | --- | --- |
| 假设与调参信息（适用的类；算法需要声明的输入） | 各行所对照锚点的假设 | 15 |
| 上界：数据项 | H-UB | 20 |
| 上界：网络与通信项 | H-UB | 15 |
| 下界：数据项 | H-LB-data、H-LB-pl、H-LB-avg 中对应参数范围内最强者 | 20 |
| 下界：网络项 | H-LB-net | 15 |
| 资源：每节点的计算与内存、每轮每条边传送的向量数 | H-UB 的算法 | 15 |

通过检查的结论全部是已知人类结果的复述时，\(T=R=10\)。本题得分为 \(\max(T,R)\)：每行都相当的完整解答为 50 分，每行都改进为 100 分；一条严格强于锚点的结论即为 100 分。没有通过检查的结论时为 0 分。

建议先在 `THEORY.md` 中写清目标陈述、对照的锚点和证明路线，再分步形式化（先证关键引理）。目标应当是本次会话内能够完整证明的最强结论。

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
