# 形式化模型说明（`Theory/Model.lean`）

`/workspace/examples/lean_project/Theory/Model.lean` 把 `/workspace/PROFILE.md` 第 1 节的问题逐条写成 Lean 定义；`Theory/Sanity.lean` 证明了这些定义的基本性质（扩展实值凸性与 Mathlib 一致、prox 的例子、各类非空、\(\varepsilon\ge\Delta\) 时 minimax 为 0）。两个文件都是题目的一部分，评测时用原件替换提交中的副本。本说明列出题面条款与定义的对应，以及题面没有写明、模型做出的解释。

## 1. 题面条款与 Lean 定义

| 题面（`PROFILE.md` §1） | Lean |
| --- | --- |
| \(x\in\mathbb R^d\)，欧氏范数 | `E d := EuclideanSpace ℝ (Fin d)` |
| \(F_i(\cdot;\xi)\) 是 \(L\)-smooth | `LSmooth L`：可微，且 `‖∇F x − ∇F y‖ ≤ L‖x − y‖` |
| \(f_i=\frac1m\sum_{\xi\in D_i}F_i\)，\(f=\frac1n\sum_i f_i\)，\(f\) 凸 | `Instance.f`；`InClass.f_convex : ConvexOn ℝ univ f`。\(D_i\) 取为 `Fin m` |
| \(h:\mathbb R^d\to\mathbb R\cup\{+\infty\}\)，proper、closed、convex | `h : E d → EReal`；`h_ne_bot`、`h_proper`、`h_closed : LowerSemicontinuous h`、`h_convex : StronglyConvexE 0 h` |
| \(h\) 可精确求 prox；\(O_{\mathrm{prox}}(v;\eta)=\operatorname{prox}_{\eta h}(v)\) | 实例自带 `prox`；`prox_exact`：对每个 \(\eta>0\)，`prox v η` 是 \(h(u)+\|u-v\|^2/(2\eta)\) 的极小点（`IsProx`） |
| \(O_i(x,\xi)=\nabla F_i(x;\xi)\) | `Algorithm.answer`：返回 `gradient (I.F i ξ) x` |
| \(\phi=f+h\) 为 \(\mu\)-强凸，\(x^\star=\arg\min\phi\) | `φ_strongly_convex : StronglyConvexE μ φ`；`minimizer`；`φstar := ⨅ x, φ x` |
| \(\phi(x_0)-\phi(x^\star)\le\Delta\) | `initial_gap : φ x0 − φstar ≤ Δ` |
| \(\mathcal W_{n,\rho}\)：\(W\ge0\)、\(W=W^\top\)、\(W\mathbf1=\mathbf1\)、\(\|W-J\|_2\le\rho\) | `CommClass`；谱范数写成 \(\forall v,\ \|(W-J)v\|_2\le\rho\|v\|_2\) |
| 节点 \(i\) 只拿到 \(\sum_j w_{ij}z_{j}\)；每轮至多 \(c\) 个向量 | `Algorithm.hist` 的通信步：每个通道 \(k\) 收到 `∑ j, W i j • send_j k`；消息类型 `Fin c → E d` |
| 每节点每迭代至多一次分量梯度、一次 prox、一次通信 | 每迭代 5 个微步：oracle 槽、oracle 槽、通信、oracle 槽、oracle 槽；`one_grad`、`one_prox` 限制每迭代各至多一次 |
| zero-respecting（初始点、此前全部 oracle 输出、全部收到的混合消息） | `allowed`；`call_/send_/out_zero_respecting` |
| 共享随机性，与实例无关 | `Algorithm.Ω`、`P`（任意概率空间），所有节点都看到种子 |
| \(\widehat x_t=\frac1n\sum_i\widehat x_{t,i}\)，由评价器形成 | `avgOutput`（算法看不到） |
| \(T_\varepsilon=\inf\{t:\mathbb E[\phi(\widehat x_t)-\phi(x^\star)]\le\varepsilon\}\) | `hittingTime`：取值在 `ℕ∞` 中，`inf ∅ = ∞`；期望为 `∫⁻`（下 Lebesgue 积分），被积函数 `gap ∈ [0, ∞]` |
| \(\inf_A\sup_{f}\sup_{O}\sup_{W}T_\varepsilon\)，下界可取足够大的 \(d\) | `minimax`：上确界也遍历维度 \(d\)；oracle 类对给定 \(h\) 只有一个元素，已并入 `Instance` |

## 2. 题面没有写明之处的解释

依据是 OptBound 中同一模板的其他题面（它们把 63 号没写的几处写明了），以及定义这类算法类的经典文献：Carmon–Duchi–Hinder–Sidford（arXiv:1710.11606）、Scaman 等（arXiv:1702.08704）、Lu–De Sa（arXiv:2006.08085）、Hendrikx 等（arXiv:2005.10675）。

1. **一轮内的顺序：不限定，可以自适应。** 文献没有统一约定（Scaman 等、Hendrikx 等允许计算与通信并行；Lu–De Sa 规定先查询后通信），题面又允许把一轮内常数个原语拆成常数个单位迭代，所以顺序只影响常数。算法类允许任意顺序（两个 oracle 槽、通信、两个 oracle 槽），因此对它成立的下界对所有顺序约定都成立。`Algorithm.Sequential` 是更强的可选性质：每轮至多一个原语，调用都在通信之前，有调用的那一轮发零消息；满足它的算法在任何顺序约定下都合法，适合用作上界的见证算法。
2. **节点知道自己的编号：是。** 同模板的较新题面逐字写明算法可以使用 "its node identity, and its local row of \(W\)"；知道第 \(i\) 行本身也包含知道 \(i\)。
3. **随机算法必须可测：是。** Carmon 等的标准定义是读取均匀种子的确定性算法，每一步映射可测；同模板题面也要求 oracle 与查询点可测。模型的要求更弱：对每个合法实例，运行中用到的每次调用、消息和输出都是种子的可测函数（种子空间任意）。不要求可测性时，依赖“某个事件的概率”的下界论证在形式上可能不成立（可以用不可测集合构造反例）。
4. **期望与首达时间。** 期望用下积分 `∫⁻`，不用 Bochner 积分 `∫`（后者对不可积函数取 0，会让上界平凡成立）；\(\phi(\widehat x_t)=+\infty\) 时 gap 为 \(\infty\)。集合为空时 \(T_\varepsilon=+\infty\)，与 `ℕ∞` 的约定一致。
5. **zero-respecting 和每轮预算对所有历史都要求**（Carmon 等的定义对任意 \(f\) 成立）。每一段历史都能由某个实例实现，所以这与“对所有实例”等价，不改变题面的算法类。
6. **消息宽度 \(c\)。** 题面要求 \(c\) 是与所有问题参数无关的绝对常数。下界应对每个固定的 \(c\ge1\) 成立（常数可以与 \(c\) 无关或写明依赖）；上界可以只用某个固定的 \(c\)，并写明。
7. **算法是对所有维度一致定义的族；\(n\) 是固定参数。** 同模板题面写明 "\(d\) may be chosen sufficiently large, while \(n\) remains a fixed parameter"；量词顺序是先给定算法，再在足够高的维度里取困难实例。`minimax n m c L μ Δ ρ ε` 中 \(n,m,c\) 和各参数都是固定的。
8. **本地计算任意、精确实数。** 同模板题面写明 "Arbitrary local computation on previously acquired information is permitted"；模型中的算法可以是任意实函数，不考虑位复杂度。节点只能通过 \(W\)-混合接收邻居信息，不能单独接收某个邻居的向量。
