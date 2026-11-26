# 任务：高斯测度下分段光滑函数的黑盒积分（理论 + 实验，v3.5）

本题同时考两件事，两部分都会被评价：

1. **实验**：写一个程序，只通过逐点查询函数值，估计 \(I(f)=\mathbb E[f(Z)]\)，\(Z\sim N(0,I_d)\)。程序在隐藏实例上自动计分（满分 100，参照 = 50）。
2. **理论**：写一份理论文档，给出关于你的算法（或本题函数类上的积分问题本身）的结论，并在 Lean 4 中形式化证明。只有通过形式化检查、且形式化陈述忠实表达文档结论的结果才计分；每条结论都要与强的人类已知结果对比。理论单独评分、单独报告，不计入实验分数。开发过程中没有理论反馈，但可以在本地运行与评测相同的形式化检查。

两部分在同一个会话里完成，不分阶段；时间怎么分配由你决定。会话时长和测试方式见启动说明（本题为 6 小时）。提前结束不加分：剩余的时间可以继续改进程序，或者把一条理论结论做得更强。

## 交付

所有交付物写入 `/workspace/submission/`：

| 文件 | 内容 |
| --- | --- |
| `run.sh`（可执行）及全部源码 | 积分程序，按下面的协议工作 |
| `theory/THEORY.md` | 理论文档（Markdown，数学公式用 LaTeX），要求见“理论部分” |
| `theory/formal/` | Lean 4 形式化工程（从 `/workspace/examples/lean_project/` 开始），要求见“形式化” |

评测只收集 `/workspace/submission/`；不要把交付物写在别处。

## 问题与函数类

被积函数 \(f:\mathbb R^d\to\mathbb R\) 是分段光滑的：在至多 \(K\) 个光滑超曲面 \(\{\varphi_i=0\}\) 上可以有折点（连续、导数跳跃）或有界跳跃，各片内部光滑，增长不超过 \(Me^{c|z|}\)。正式定义（类 \(\mathcal F_d(K,r,M,c)\)）、信息与代价模型、误差准则、已知结果和开放问题见 `/workspace/THEORY_APPENDIX.md`，它是本题的一部分，理论结论应针对其中声明的类或你明确写出的子类。

隐藏实例分两条赛道，计分时等权：

- **困难赛道（合成）**：取预积分与 QMC 理论中最难的情形——沿任何方向都不单调的闭合跳跃面（含落在尾部的稀有事件）、只沿零锥方向单调的鞍形折点、在函数值不为零处相交的折点面（零值检测无效）。非光滑面是未知子空间（维数 2 到 \(d\)）里谱各异的二次超曲面，并叠加一个真正全维的光滑项。\(d\in\{8,16,32,64\}\)，共 4 个族。
- **期权赛道**：单资产、离散监测的路径收益——欧式看涨、数字（现金或无）、向下敲出看涨、双障碍敲出看涨，共 4 个族。价格路径由一个单因子扩散模型按 \(m\) 个等距时间步离散生成，波动率随时间和价格水平光滑变化，形状每例不同、不公开。\(d=m\in\{16,32,64\}\)。

所有实例的输入都先经过一个隐藏的随机正交变换 \(z\mapsto Qz\)（不改变积分），所以结构的方向必须从查询中发现。每个实例的参数（二次型的谱、中心与子空间，波动率函数，行权价与障碍，背景项）都是随机抽取的：在样例上拟合出的具体形式不会出现在隐藏实例中。具体公式、参数和变换不公开；你的程序在运行时也拿不到族的名称。

## 交互协议

`run.sh` 从 stdin 读 JSON Lines，向 stdout 写 JSON Lines。启动后先收到：

```json
{"type": "init", "dimension": 32, "budget": 16384, "max_batch": 4096, "seed": 1234567, "time_limit_s": 30}
```

程序可以多次查询（每次 1 到 `max_batch` 个点）：

```json
{"type": "query", "points": [[0.1, -1.3, ...], [2.0, 0.4, ...]]}
```

评测器回复同样个数的函数值：

```json
{"type": "values", "values": [0.91, 0.37]}
```

最后写出一次结果：

```json
{"type": "result", "estimate": 12.3456, "error_bound": 0.002, "confidence": 0.95}
```

- 查询点是 \(\mathbb R^d\) 中的点（不是单位立方体），每个坐标必须有限且 \(|z_i|\le 40\)（之外高斯密度在双精度下为 0）。
- 每个点计一次预算，重复点、试探点、有限差分、交叉验证都计入；累计不超过 `budget` \(=2^{14}\)。可以提前给出结果。
- 查询可以依赖此前的函数值（自适应），可以随机化；`seed` 用于让随机化可复现。
- `error_bound` 与 `confidence` 可选：`confidence` 为 `"deterministic"` 或 \((0,1)\) 中的数（声称 \(|\hat I-I|\le\) `error_bound` 的概率）。给了 `error_bound` 就必须给 `confidence`。
- stdout 只能写协议消息；日志写 stderr。估计值必须是有限数。

`/workspace/examples/mc_submission/` 是一个完整的示例（普通蒙特卡洛，附 CLT 误差界），`cubature_protocol.py` 是可复用的协议客户端。

## 资源与运行方式

- 每个实例在一个全新进程中运行，1 个 CPU、2 GiB 内存、30 秒墙钟（含启动和导入），断网。
- 提交目录在评测时是只读的；只有 `/tmp` 与家目录可写，且在实例之间清空。每个实例必须独立求解，不能在实例之间传递信息。
- 环境：Python 3.11、NumPy 2.1.3、SciPy 1.17.1、QMCPy 2.4、C 编译器。评测时断网，只能使用这些已安装的库（不能安装或附带第三方包）。

## 网络（开发阶段）

开发容器可以经代理访问公网，用于查阅文献和资料。环境变量 `HTTPS_PROXY` 和 `HTTP_PROXY` 已设置，`curl`、Python 的 `urllib` 和 `requests` 会自动使用。

- 不可访问：GitHub 及其镜像、包管理源（PyPI、conda 等）、在线大模型服务、内网。被拒绝的请求返回 403。所有访问都会被记录。
- 文献检索可以用 arXiv API（`https://export.arxiv.org/api/query?search_query=...`）、OpenAlex（`https://api.openalex.org/works?search=...`）、Crossref（`https://api.crossref.org/works?query=...`）、zbMATH Open 和 Semantic Scholar（无 key 时限流）。arXiv 论文的 LaTeX 源码可以从 `https://arxiv.org/e-print/<编号>` 下载（gzip 或 tar），比 PDF 更方便阅读。
- 网络只用于查阅资料：不要查找或使用与本评测（FrontierAM）相关的任何材料（题目的生成代码、他人的解答、评审文档），也不要使用任何在线 AI 服务。
- 提交在评测时仍然离线运行，只能使用上面列出的已安装环境。

## 计分（实验部分）

对每个隐藏实例：

- 误差 \(e=\log_{10}(|\hat I-I|/|I|)\)，截到 \([e_{\text{floor}},0]\)。所有实例的真值都由确定性方法算到 \(10^{-13}\) 量级并经独立方法核对，\(e_{\text{floor}}=-12\)。
- 与在评测机上重跑的冻结参照 **GPCA-RQMC**（Xiao–Wang 2019：用有限差分梯度估计 \(E[\nabla f\nabla f^\top]\)，查询计入预算，再在其特征基上取一个扰乱 Sobol' 网）在同一实例上配对：增益 \(g=e_{\text{参照}}-e\)，截到 \([-2,2]\)。

汇总：同一格内的实例平均（困难赛道的格是“族 × 维数”，期权赛道是“族 × 配置”），格平均得族，族平均得赛道，两条赛道平均得 \(G\)。

\[
\text{得分}=\begin{cases}50+25\,G, & G\ge0\quad(\text{每例都比参照准 100 倍时为 100 分}),\\[2pt] \max\!\big(0,\;50\,(G-G_0)/(-G_0)\big), & G<0,\end{cases}
\]

其中 \(G_0\) 是普通蒙特卡洛（\(2^{14}\) 个独立点取平均）在同一组实例上的 \(G\)，对应 0 分。任一实例失败、超时或违反协议，整份提交记 0 分。只要在时限内完成，用时不加分。

迭代测试用的反馈集只有 20 例、每例运行一次，分数噪声较大：普通蒙特卡洛在反馈集上约得 59 分（在正式集上为 0 分）。只有私有正式集决定最终分数。

误差界只作诊断：评测端统计覆盖率（\(|\hat I-I|\le\) `error_bound` 的比例，按声称的置信水平分组）和松紧（`error_bound` 与实际误差之比的中位数），交给理论评审参考，不计入实验分数。

## 理论部分

`theory/THEORY.md` 应包含：

1. **算法**：你提交的程序实际执行的算法，精确到可以据此分析（预算如何分配、查询点如何选取、估计量是什么）。
2. **结论与证明**：在明确写出的假设下（\(\mathcal F_d\) 或其子类，写清参数）的定理，例如误差界或收敛速率（最好带显式常数或常数的依赖关系）、定位或方向估计的查询复杂度、任意算法的下界、所输出误差界的覆盖率保证。每条结论编号（R1、R2、……），给出完整的数学证明和对应的 Lean 定理名；未证明的标为“猜想”。
3. **与人类已知结果对比**：对每条结论，找出一个或多个最强的可比人类结果（`THEORY_APPENDIX.md` 第 5 节的文献或你检索到的其他文献；写明可核对的出处，例如 arXiv 编号或 DOI，以及定理编号，并对照原文准确复述其假设与结论），并归入一类：
   - **严格更强**：在相同或更弱的假设下得到严格更好的结论（更快的速率，更好的常数或维数依赖，更广的函数类），或用更弱的假设得到相同结论，即突破了该人类结果；
   - **严格更弱**：结论被某个已知结果蕴含，且后者严格更强；
   - **相当**：与已知结果等价（复现）；
   - **不可比**：各有所长。说明得失，并依据人类论文自身表达的观点（它们指出的局限、提出的开放问题、关心的参数范围）论证这条结论的价值。

   如果回答了第 6 节某个开放问题的一部分，也在这里写明。
4. **与实验的联系**：隐藏实例中哪些族满足你的假设、理论预测的误差量级，以及在样例上的实测是否一致。

## 人类 SOTA 目标

理论部分的目标是超越人类已有的最好结果。下面四个目标都取自人类论文中的定理，或作者自己写明的空白（出处已核对，可以联网阅读原文）。至少选择一个，尝试给出**严格强于**所列人类结果的结论：在相同或更弱的假设下得到更强的结论（更快的速率、更好的常数、更紧的上下界），或用更弱的假设得到相同的结论；或者在作者写明的空白上取得可证明的进展。

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
- 资源：开发容器共 8 GiB 内存、1 个 CPU，Lean 与你的程序共用。全量 `import Mathlib` 约需 1 分钟，只导入用到的模块会快一些。

## 计分（理论部分）

- 只有形式化验证通过的结论计分：在 manifest 中标为 `proved`，通过 `lake comparator`，并且审阅者认定它的 Lean 陈述忠实表达了 `THEORY.md` 中的结论。偷换了假设、前提不可满足、结论被弱化成平凡命题的，都不算通过；把一个困难结论形式化成较弱的陈述，只按较弱的陈述计分。
- 每条通过的结论按它与最强可比人类结果的关系定分（0–100）：与人类结果相当为 50 分；严格更强的在 50 分以上，按突破的幅度和意义定分，对重要已知结果的显著突破为 100 分；严格更弱的在 50 分以下；不可比的由审阅者依据人类论文自身表达的观点定分。你给出的分类和理由是审阅的出发点，审阅者会独立核对。
- 理论分取所有通过结论中的最高分；没有通过的结论时为 0 分。未形式化的结论可以写进文档作为讨论，但不计分。
- 对“人类 SOTA 目标”中所列人类结果的严格改进按“严格更强”定分（50 分以上）；在目标中作者写明的空白上取得可证明的进展，即使与原定理不可比，也可以高于 50 分。复现经典结果或只在很窄的参数特例上成立的结论低于 50 分。
- 因为只看最强的一条，一条严格强于人类已知结果的结论（或在不可比时推进了人类论文明确提出的问题的结论），比多条初等结论更有价值。建议把大部分理论时间投入到一条有分量的结论上：先在 `THEORY.md` 中写清目标陈述、要比较的人类结果和证明路线，再分步形式化（先证关键引理；Mathlib 缺少的基础可以自己定义并证明）。只有完整通过形式化验证的结论才计分，所以目标应当是本次会话内能够完整证明的最强结论，而不是无法完成的宏大猜想。
- 理论分单独报告，不计入实验分数。

## 允许与禁止

- 允许：任何数值方法、随机化、QMC、自适应查询、环境中已有的库；为本题函数类专门设计算法。
- 禁止：读取评测器、隐藏数据或环境中的其他信息；程序在评测时联网；查找或使用与本评测（FrontierAM）相关的材料；使用在线 AI 服务；在实例之间保存或传递信息；用查询以外的方式识别实例；返回与函数值无关的固定答案。

## 工作区

| 路径 | 内容 |
| --- | --- |
| `/workspace/problem.md` | 本题面 |
| `/workspace/THEORY_APPENDIX.md` | 函数类、信息与代价、误差准则、已知结果、开放问题 |
| `/workspace/samples/samples.json` | 12 个样例实例（每个族 1–2 个，与隐藏实例同分布、种子不同）：维数、族、种子、真值、\(e_{\text{floor}}\)，以及参照（GPCA-RQMC）与普通蒙特卡洛在该例上的误差 |
| `/workspace/tools/local_eval.py` | 按正式协议与限制在样例上运行你的提交，并按正式映射给出样例分（仅供参考） |
| `/workspace/tools/oracle_client.py` | 直接查询样例函数（每个样例在本次会话内累计最多 \(2^{18}\) 个点，本地评测工具的查询也计入） |
| `/workspace/examples/mc_submission/` | 示例提交与协议客户端 |
| `/workspace/examples/lean_project/` | Lean 形式化工程模板（`Theory/Statements.lean`、`Theory/Proofs.lean`、`manifest.json` 各有一个示例条目，替换掉） |
| `/workspace/tools/lean_check.py` | 本地运行与评测相同的形式化检查（`--each` 逐条判定） |

样例函数在容器外运行，通过 `/run/cubature-samples/oracle.sock` 应答；工作区里没有它们的源码。只有隐藏的正式集决定最终分数。
