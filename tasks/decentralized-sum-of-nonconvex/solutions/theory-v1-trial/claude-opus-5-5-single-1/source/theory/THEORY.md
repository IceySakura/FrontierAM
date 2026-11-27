# Lower bounds for decentralized finite-sum, sum-of-nonconvex, strongly convex composite optimization (OptBound ULB-profile-63)

All results below are lower bounds in the **fixed formal model** `Theory/Model.lean`, with nothing changed: the function class `Instance.InClass`, the communication class `CommClass`, the algorithm class `Algorithm.Admissible`, the hitting time `Algorithm.hittingTime`, and the quantity `minimax`. Every result marked *proved* is machine-checked: `python3 /workspace/tools/lean_check.py` ends with `LEAN-CHECK: 8/8 proved results accepted` (kernel-checked, statements identical, axioms limited to `propext`, `Quot.sound`, `Classical.choice`).

**摘要（中文）。** 本文选择目标 A（下界），并给出一个 C 型推论（在参数子区域与人类上界匹配到对数因子）。在固定的 Lean 模型中（随机化、zero-respecting 算法，每轮至多一次分量梯度、一次 prox、一轮通信，$W\in\mathcal W_{n,\rho}$，$n$ 固定），我们完整形式化证明了：
$$\mathrm{minimax}=\Omega\Big(\mathcal D+\min\big\{n,(1-\rho)^{-1/2}\big\}\sqrt\kappa\log\tfrac{\Delta}{\kappa\varepsilon}\Big),$$
其中数据项 $\mathcal D$ 在 $\kappa\ge nm$ 时为 $m+\sqrt{m\kappa/n}\log\frac{nm\Delta}{\kappa\varepsilon}$，在 $\kappa\le nm$ 时为 $m\max\{1,\log\frac\Delta\varepsilon/\log\frac{6nm}\kappa\}$（R1–R8，`LEAN-CHECK: 8/8`；R8 表明在 $W\succeq0$ 且分量凸的受限类中同样成立）。
- 这是本问题类（逐分量 $L$-光滑、固定 $n$、一般 zero-respecting 随机算法）上第一个同时含网络项与数据项的下界。
- 相对 Woodworth–Srebro 定理 8 直接推出的下界 $\Omega(m+\sqrt{m\kappa/n}\log)$，当 $n>m$ 时严格更强，强出 $\sqrt{n/m}$ 倍；它还覆盖 $\kappa\le161nm$。
- 与 Liu–Low 推论 5.3 的上界结合：$m=O(1)$ 且 $1-\rho\gtrsim n^{-2}$ 时，复杂度确定为 $\widetilde\Theta(\sqrt{\kappa/(1-\rho)}\log\frac\Delta\varepsilon)$；$\kappa\le\sqrt m$ 且 $1-\rho\ge1/m$ 时，确定为 $\widetilde\Theta(m\log\frac\Delta\varepsilon)$。
- 未解决：逐分量光滑下 $\sqrt{m\kappa/n}$ 与 $m^{3/4}\sqrt\kappa$ 之间的差距（集中式情形同样未解决），以及 $1-\rho\ll n^{-2}$ 时的网络项上界。

## 0. Summary: goal, human baseline, and how the results relate to it

**Chosen goal: A (lower bounds).** There is also a C-type corollary: a parameter range where our lower bound matches the human upper bound up to a logarithmic factor (§6).

**Human baseline.** The two main comparators are stated here as in the arXiv sources; all comparators are in §7.
- *Liu–Low, arXiv:2402.02356v1, Cor. 5.3* (their notation: $m$ agents, $n$ components each, components $(\ell_1,\ell_2)$-smooth, $f$ convex and $L$-smooth, $F=f+\psi$ $\sigma$-strongly convex, $W$ symmetric doubly stochastic with $0\preceq W\preceq I$ and $\mathrm{null}(I-W)=\mathrm{span}(\mathbf 1)$): "to obtain an $\epsilon$-approximate solution $\bar y^K$, i.e., $F(\bar y^K)-F(x^*)\le\epsilon$, PMGT-KatyushaX requires at most $\mathcal O\big((n+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}n^{3/4})\log\frac{F(\bar y^0)-F(x^*)}\epsilon\big)$ SFO calls and $\tilde{\mathcal O}\big(\frac{\sqrt n+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}n^{1/4}}{\sqrt{1-\lambda_2(W)}}\log\frac{F(\bar y^0)-F(x^*)}\epsilon\big)$ rounds of communication."
- *Woodworth–Srebro, arXiv:1605.08003v3, Thm. 8*: "For any $m\ge2$, any $\gamma,\lambda>0$ such that $\frac\gamma\lambda>161m$, any $\epsilon>0$, any $\epsilon_0>60\epsilon\sqrt{\frac{\gamma}{\lambda m}}$, and any randomized algorithm $A$, there exists a dimension $d$, domain $\mathcal X\subseteq\mathbb R^d$, $x_0\in\mathcal X$, and $m$ functions $f_i$ … which are $\gamma$-smooth and $\lambda$-strongly convex, and such that $F(x_0)-F(x^*)=\epsilon_0$ and such that in order to find a point $\hat x$ such that $\mathbb E[F(\hat x)-F(x^*)]<\epsilon$, $A$ must make $\Omega\big(m+\sqrt{\frac{m\gamma}\lambda}\log\big(\frac{\epsilon_0}\epsilon\sqrt{\frac{m\lambda}\gamma}\big)\big)$ queries to $h_F$." Here one query returns $[f_i(x),\nabla f_i(x),\mathrm{prox}_{f_i}(x,\beta)]$.

Translated to our notation and iteration count, these give the following.
- *Upper bound*: Liu–Low, arXiv:2402.02356v1, Cor. 5.3. In our notation it gives $\widetilde O\big((m+m^{3/4}\sqrt\kappa)\log\frac\Delta\varepsilon+(\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2}\log\frac\Delta\varepsilon\big)$ iterations.
- *Lower bounds*: no human lower bound is stated for this class. The only listed classical bound that applies **directly** to our algorithm class (randomized, zero-respecting, with arbitrary nonlinear local computation) is Woodworth–Srebro, arXiv:1605.08003, Thm. 8. Applied to all $N=nm$ components and divided by the $n$ parallel queries per iteration, it gives
$$T_\varepsilon=\Omega\Big(m+\sqrt{m\kappa/n}\,\log\big(\tfrac{\Delta}{\varepsilon}\sqrt{nm/\kappa}\big)\Big)\qquad(\kappa>161\,nm).$$
- The network lower bounds of Scaman et al. (arXiv:1702.08704, Thm. 2) and Hendrikx–Bach–Massoulié (arXiv:2005.10675, Thm. 1, Cor. 3) are proved only for *span-based black-box procedures*, a strict subclass of our algorithm class. Their corollaries also choose the number of nodes $n\approx\gamma^{-1/2}$ from the spectral gap. So they do not apply directly to this problem, where $n$ is a fixed parameter.

**Our results.** Throughout, $\kappa=L/\mu$, $N=nm$, and $T$ ranges over natural numbers. We write
$$\ell(x):=\log\frac{\sqrt{1+x}+1}{\sqrt{1+x}-1}\qquad(x>0).$$
This is $\log(1/q)$ for the decay rate $q$ of the chain minimizer, and it satisfies $\ell(x)\le\frac{2}{\sqrt{1+x}-1}\le\frac{4.9}{\sqrt x}$ for $x\ge1$ and $\ell(x)\le\log(2+\frac4x)$ for all $x$.

| # | Statement (informal) | Lean theorem | Status | Comparison |
|---|---|---|---|---|
| R1 | Every matrix $W$, node sets $S_A,S_B$ at $W$-graph distance $\ge D$, $\lvert S_A\rvert=\lvert S_B\rvert=a$: some instance forces $T_\varepsilon\ge T$ for every integer $T\le D\big[\log\tfrac{4n\Delta}{a\kappa\varepsilon}/(2\ell(\tfrac{a\kappa}n))-2\big]$ (components convex) | `ULB63.lower_bound_network` | proved | incomparable (Scaman Thm 2; Hendrikx Thm 1) |
| R2 | $\mathrm{minimax}\ge T$ for every integer $T\le \log\tfrac{8\Delta}{\kappa\varepsilon}/(2\ell(\tfrac\kappa3))-2$, every $n\ge2,\rho\ge0$ | `ULB63.minimax_ge_sqrt_kappa_log` | proved | equal (classical $\sqrt\kappa\log$ term) |
| R3 | Every $W$: some instance forces $T_\varepsilon\ge T$ for every integer $T\le\max\big\{\tfrac m5(1-\lambda\tfrac\varepsilon\Delta),\ \tfrac m{10}\log\tfrac{\Delta}{\lambda\varepsilon}\big/\ell(\tfrac\kappa{nm})\big\}$, $\lambda=1+\tfrac{\kappa}{2nm}$ (components convex) | `ULB63.lower_bound_data` | proved | equal (WS Thm 8 transferred), with minor improvements |
| R4 | $\mathrm{minimax}\ge\max\{\text{R3 terms},\ \text{R2 term}\}$ for every $n\ge2,\rho\ge0$ | `ULB63.minimax_ge_combined` | proved | **stronger** than the bound WS Thm 8 directly implies (larger by $\sqrt{n/m}$ when $n>m$) |
| R5 | $n\ge4$, $\rho\ge\sqrt{1-1/(2n^2)}$: $\mathrm{minimax}\ge\max\{\text{R3 terms},\ \tfrac n2[\log\tfrac{16\Delta}{\kappa\varepsilon}/(2\ell(\tfrac\kappa7))-2]\}$ (lazy path) | `ULB63.minimax_ge_path_combined` | proved | **stronger** (special case of R6 up to constants) |
| R6 | **every $\rho$**: for any $8\le K\le n$ with $\rho\ge\sqrt{1-1/(4K^2)}$, $\mathrm{minimax}\ge\max\{\text{R3 terms},\ \text{R2 term},\ (\tfrac K2-2)[\log\tfrac{16\Delta}{\kappa\varepsilon}/(2\ell(\tfrac\kappa7))-2]\}$ (path of $K$ cliques) | `ULB63.minimax_ge_all` | proved | **stronger**: data + $\sqrt\kappa$ + network $\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa$ terms at fixed $n$ for the general zero-respecting class |
| R7 | explicit $\rho$: with $K_\rho=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}\ge8$, $\mathrm{minimax}\ge(\tfrac{K_\rho}2-2)[\log\tfrac{16\Delta}{\kappa\varepsilon}/(2\ell(\tfrac\kappa7))-2]$ | `ULB63.minimax_ge_explicit_rho` | proved | **stronger** (R6 with $\rho$ explicit) |
| R8 | the R6 data and network bounds are attained with a **positive semidefinite** $W\in\mathcal W_{n,\rho}$ and convex components (restricted class $W\succeq0$, convex components) | `ULB63.lower_bound_psd` | proved | **stronger** (same bound on a smaller class; within Liu–Low's assumption $0\preceq W\preceq I$) |

In $\Omega$-form, for $\varepsilon$ small (§2.7), and writing $\mathcal D:=m+\sqrt{\tfrac{m\kappa}{n}}\log\tfrac{nm\Delta}{\kappa\varepsilon}$ for $\kappa\ge nm$ and $\mathcal D:=m\max\{1,\log\tfrac\Delta\varepsilon/\log\tfrac{6nm}\kappa\}$ for $\kappa\le nm$ (data term):
$$\mathrm{minimax}=\Omega\Big(\mathcal D+\sqrt{\kappa}\log\tfrac{\Delta}{\kappa\varepsilon}\Big)\quad(\rho\ge0),$$
$$\mathrm{minimax}=\Omega\Big(\mathcal D+\min\Big\{n,\tfrac1{\sqrt{1-\rho}}\Big\}\sqrt{\kappa}\log\tfrac{\Delta}{\kappa\varepsilon}\Big)\quad(\text{every }\rho\in[0,1),\ \text{R4 and R6}).$$
The last term is the network term. It comes from R6 with $K=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}$ when that $K$ is at least $8$, and from R2 otherwise. This uses $1-\rho\le1-\rho^2\le2(1-\rho)$.

**What is new relative to the human results.**
1. R4 strictly improves the lower bound that the classical results imply for this problem, by the factor $\sqrt{n/m}$ on the $\log(1/\varepsilon)$ term when $n>m$. It covers every $\kappa$, including $\kappa\le161nm$ where WS Thm 8 is silent, and it is fully machine-checked in the official model.
2. R6 (and its special case R5) is, as far as we found (§7.6), the first lower bound for this problem class with a **network term and data terms together** at **fixed** $n$, valid for **all randomized zero-respecting algorithms**, not only span-based black-box procedures. Its network term $\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log$ covers every $\rho$.
   - It is the fixed-$n$ analog of Scaman et al. Cor. 2 / Hendrikx et al. Cor. 3, which instead pick $n\approx\gamma^{-1/2}$.
   - With Liu–Low it settles the order $\widetilde\Theta(\sqrt{\kappa/(1-\rho)}\log\frac\Delta\varepsilon)$ for $m=O(1)$ in the whole range $1-\rho\ge c/n^2$ (§6, C2).
3. R1 gives the network term for **every** gossip matrix through a graph-distance potential, with $n$ fixed.
4. R7 states the network term with $\rho$ explicit. R8 shows that all of this already holds on the smaller class with $W\succeq0$ and convex components, so it applies inside the exact assumptions of the human upper bound (Liu–Low: $0\preceq W\preceq I$).
5. R3/R4 also give the well-conditioned data term $\Omega(m\log\frac\Delta\varepsilon/\log\frac{nm}\kappa)$ for $\kappa\le nm$. With Liu–Low this gives $\widetilde\Theta(m\log\frac\Delta\varepsilon)$ for $\kappa\le\sqrt m$ and $1-\rho\ge1/m$ (§6, C1).

**What is not achieved.** We do not close the $m^{1/4}$-type gap between $\sqrt{m\kappa/n}$ and Liu–Low's $m^{3/4}\sqrt\kappa$. Under individual smoothness this gap is open even in the centralized case (Zhou, arXiv:2611.05807). We prove no upper bound. The network term is $\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa$, not $(1-\rho)^{-1/2}\sqrt\kappa$, when $1-\rho\ll1/n^2$. We conjecture this is the truth in the exact-arithmetic model (§6). Our network term does not multiply the data factor $m^{1/4}$ that appears in Liu–Low's communication bound. All hard instances have convex components, so we do not exploit nonconvexity of the components (§6).

---

## 1. Model and notation

We use the fixed model verbatim; see `MODEL_NOTES.md`.
- An instance in dimension $d$ consists of components $F_i(\cdot;\xi):\mathbb R^d\to\mathbb R$ for $i\in[n]$ and $\xi\in[m]$, a regularizer $h:\mathbb R^d\to\overline{\mathbb R}$, an initial point $x_0$, and a proximal oracle.
- $f=\frac1n\sum_i\frac1m\sum_\xi F_i(\cdot;\xi)$ and $\phi=f+h$.
- Membership `InClass L μ Δ` requires:
  - every component is $L$-smooth;
  - $f$ is convex;
  - $h$ is proper, closed and convex with an exact prox;
  - $\phi$ is $\mu$-strongly convex and has a minimizer;
  - $\phi(x_0)-\inf\phi\le\Delta$.
- An admissible algorithm (`Admissible`):
  - is randomized with a shared seed;
  - runs per iteration 5 micro-steps: two oracle slots, one communication round with $c$ channels where node $i$ receives $\sum_j W_{ij}z_j$, then two more oracle slots;
  - makes at most one component-gradient query and at most one prox query per iteration;
  - is zero-respecting: queries, messages and outputs vanish outside the coordinates that are nonzero in $x_0$, in previous oracle answers, or in received messages;
  - is measurable.
- $T_\varepsilon=\inf\{t:\ \mathbb E[\phi(\hat x_t)-\phi^\star]\le\varepsilon\}$, where $\hat x_t=\frac1n\sum_i\hat x_{t,i}$ and the expectation is a lower Lebesgue integral.
- $\mathrm{minimax}=\inf_A\sup_{d,I,W}T_\varepsilon$, with $W$ ranging over $\mathcal W_{n,\rho}$.

Notation: $\kappa=L/\mu$, $N=nm$. For a matrix $W$, the support graph has edges $\{i,j\}$ with $W_{ij}\ne0$. $\mathrm{Pre}_j=\{p:p<j\}$ is the set of the first $j$ coordinates.

All hard instances below have $x_0=0$, $h=\frac\mu2\|\cdot\|^2$ with prox $v\mapsto v/(1+\eta\mu)$, and quadratic, **convex** components.

---

## 2. Statements

### 2.1 R1 — network-aware chain lower bound, every gossip matrix (`ULB63.lower_bound_network`)

**Theorem R1.** Assume:
- $m\ge1$ and $L,\mu,\Delta,\varepsilon>0$, with $\rho$ arbitrary;
- $A$ is an admissible algorithm and $W\in\mathbb R^{n\times n}$ is **any** matrix;
- $D\ge1$ and $\delta:[n]\to\{0,\dots,D\}$ satisfies $|\delta(i)-\delta(j)|\le1$ whenever $W_{ij}\ne0$;
- $S_A,S_B\subseteq[n]$ with $\delta|_{S_A}=0$, $\delta|_{S_B}=D$, $|S_A|=|S_B|=a\ge1$.

Let $T\in\mathbb N$ with
$$T\le D\Big(\frac{\log\frac{4n\mu\Delta}{aL\varepsilon}}{2\,\ell(aL/(n\mu))}-2\Big).$$
Then there are a dimension $d$ and an instance $I\in\mathcal F_{L,\mu,\Delta,m}$, all of whose components are convex, such that $T\le T_\varepsilon(A,I,W)$.

Since $\frac1{2\ell(x)}\ge\frac{\sqrt{1+x}-1}4$, every $T\le D\big(\frac{\sqrt{1+a\kappa/n}-1}{4}\log\frac{4n\Delta}{a\kappa\varepsilon}-2\big)$ satisfies the hypothesis.

The hypotheses on $\delta$ say exactly that $S_A$ and $S_B$ are at distance $\ge D$ in the support graph of $W$; take $\delta(i)=\min\{\mathrm{dist}_W(i,S_A),D\}$. In $\Omega$-form, for $\kappa\ge n/a$ and $\varepsilon$ small:
$$T_\varepsilon=\Omega\Big(D\sqrt{\tfrac{a\kappa}{n}}\log\tfrac{n\Delta}{a\kappa\varepsilon}\Big).$$
The hard instance does not depend on $A$.

### 2.2 R2 — the $\sqrt\kappa\log$ term (`ULB63.minimax_ge_sqrt_kappa_log`)

**Theorem R2.** For $n\ge2$, $m\ge1$, $L,\mu,\Delta,\varepsilon>0$, $\rho\ge0$, and every $T$ with
$$T\le\frac{\log\frac{8\Delta}{\kappa\varepsilon}}{2\,\ell(\kappa/3)}-2,$$
we have $T\le\mathrm{minimax}(n,m,c,L,\mu,\Delta,\rho,\varepsilon)$ (in $\mathbb N_\infty$).

### 2.3 R3 — data lower bound, every gossip matrix (`ULB63.lower_bound_data`)

**Theorem R3.** Let $n,m\ge1$, $L,\mu,\Delta,\varepsilon>0$, let $A$ be admissible and $W$ any matrix. Put $\lambda:=1+\frac{L}{2nm\mu}$. For every $T$ with
$$T\le\max\Big\{\frac m5\Big(1-\lambda\frac{\varepsilon}{\Delta}\Big),\ \ \frac m{10}\,\frac{\log\frac{\Delta}{\lambda\varepsilon}}{\ell(\kappa/(nm))}\Big\},$$
there are $d$ and an instance $I\in\mathcal F_{L,\mu,\Delta,m}$, all of whose components are convex, with $T\le T_\varepsilon(A,I,W)$. The instance does not depend on $A$ or $W$.

### 2.4 R4 — combined minimax bound for every $\rho\ge0$ (`ULB63.minimax_ge_combined`)

**Theorem R4.** For $n\ge2$, $m\ge1$, $L,\mu,\Delta,\varepsilon>0$, $\rho\ge0$, and every $T$ with
$$T\le\max\Big\{\frac m5\Big(1-\lambda\frac\varepsilon\Delta\Big),\ \frac m{10}\frac{\log\frac\Delta{\lambda\varepsilon}}{\ell(\kappa/(nm))},\ \frac{\log\frac{8\Delta}{\kappa\varepsilon}}{2\ell(\kappa/3)}-2\Big\}$$
we have $T\le\mathrm{minimax}$.

### 2.5 R5 — combined minimax bound with the network term (`ULB63.minimax_ge_path_combined`)

**Theorem R5.** For $n\ge4$, $m\ge1$, $L,\mu,\Delta,\varepsilon>0$, $\rho\ge\sqrt{1-1/(2n^2)}$, and every $T$ with
$$T\le\max\Big\{\frac m5\Big(1-\lambda\frac\varepsilon\Delta\Big),\ \frac m{10}\frac{\log\frac\Delta{\lambda\varepsilon}}{\ell(\kappa/(nm))},\ \frac n2\Big(\frac{\log\frac{16\Delta}{\kappa\varepsilon}}{2\ell(\kappa/7)}-2\Big)\Big\}$$
we have $T\le\mathrm{minimax}$.

### 2.6 R6 — every $\rho$: path of cliques (`ULB63.minimax_ge_all`)

**Theorem R6.** Let $m\ge1$, $L,\mu,\Delta,\varepsilon>0$, and let $K\in\mathbb N$ satisfy $8\le K\le n$ and $\rho\ge\sqrt{1-1/(4K^2)}$. Then $T\le\mathrm{minimax}$ for every $T$ with
$$T\le\max\Big\{\frac m5\Big(1-\lambda\frac\varepsilon\Delta\Big),\ \frac m{10}\frac{\log\frac\Delta{\lambda\varepsilon}}{\ell(\kappa/(nm))},\ \frac{\log\frac{8\Delta}{\kappa\varepsilon}}{2\ell(\kappa/3)}-2,\ \Big(\frac K2-2\Big)\Big(\frac{\log\frac{16\Delta}{\kappa\varepsilon}}{2\ell(\kappa/7)}-2\Big)\Big\}.$$

The best choice is the largest admissible $K$, namely $K=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}$. The last term is then of order $\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log\frac\Delta{\kappa\varepsilon}$.

**Corollary R7 (`ULB63.minimax_ge_explicit_rho`; explicit $\rho$).** Let $0\le\rho<1$ and $K_\rho:=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}$. If $K_\rho\ge8$, then $T\le\mathrm{minimax}$ for every $T\le(\frac{K_\rho}2-2)\big(\log\frac{16\Delta}{\kappa\varepsilon}/(2\ell(\kappa/7))-2\big)$.

*Proof.* $K_\rho\le1/(2\sqrt{1-\rho^2})$ gives $4K_\rho^2(1-\rho^2)\le1$, i.e. $\rho\ge\sqrt{1-1/(4K_\rho^2)}$ (`rho_ok`). Then apply R6, through `minimax_ge_cliques`. ∎

Since $1-\rho\le1-\rho^2\le2(1-\rho)$, we have $K_\rho\asymp\min\{n,(1-\rho)^{-1/2}\}$.

**Theorem R8 (`ULB63.lower_bound_psd`; restricted class).** Under the hypotheses of R6, and for every $T$ bounded by the largest of the two data terms and the network term of R6, the following holds. For every admissible algorithm there are a matrix $W\in\mathcal W_{n,\rho}$ with $W\succeq0$ (`Matrix.PosSemidef`) and an instance of $\mathcal F_{L,\mu,\Delta,m}$ whose components are all convex, with $T_\varepsilon(A,I,W)\ge T$.

So the lower bound also holds for the "(+)" alternatives "$W\succeq0$" and "convex components" of `PROFILE.md` §3. It also holds within Liu–Low's network assumption $0\preceq W\preceq I$: a symmetric doubly stochastic $W$ satisfies $W\preceq I$.

### 2.7 Reading the bounds ($\Omega$-forms)

Use $\ell(x)\le2/(\sqrt{1+x}-1)\le\frac{2}{(\sqrt2-1)\sqrt x}<\frac{4.9}{\sqrt x}$ for $x\ge1$, and $\ell(x)\le\log(2+4/x)$ for every $x>0$. The second bound follows from $\frac{s+1}{s-1}=\frac{(s+1)^2}{x}\le\frac{2(2+x)}x$ with $s=\sqrt{1+x}$.

- **Data term $m$.** If $\varepsilon\le\Delta/(2\lambda)$, the first term is $\ge m/10$. For $\kappa\le nm$ this holds as soon as $\varepsilon\le\Delta/3$.
- **Data term $\frac m{10}\log\frac\Delta{\lambda\varepsilon}/\ell(\frac\kappa{nm})$.** With $s=\sqrt{1+\kappa/(nm)}$ we have $\ell=\log\frac{s+1}{s-1}$, $\frac{s+1}{s-1}=\frac{(s+1)^2nm}{\kappa}\le\frac{4\lambda nm}{\kappa}$, and $\ell\le\frac2{s-1}$.
  - For $\kappa\ge nm$, the term is $\ge\frac m{20}(s-1)\log\frac\Delta{\lambda\varepsilon}\ge0.02\sqrt{m\kappa/n}\,\log\frac{2nm\Delta}{3\kappa\varepsilon}$.
  - For $\kappa\le nm$, the term is $\ge\frac m{10}\log\frac{2\Delta}{3\varepsilon}\big/\log\frac{6nm}{\kappa}$, i.e. $\Omega(m\log\frac\Delta\varepsilon/\log\frac{nm}\kappa)$. This is the well-conditioned finite-sum rate.
- **$\sqrt\kappa\log$ term.** For $\kappa\ge3$, the R2 term is $\ge\frac{\sqrt{\kappa/3}}{9.8}\log\frac{8\Delta}{\kappa\varepsilon}-2\ge0.058\sqrt\kappa\log\frac{8\Delta}{\kappa\varepsilon}-2$. For $\kappa\le3$ it is $\ge\log\frac{8\Delta}{\kappa\varepsilon}/(2\log\frac{18}\kappa)-2$. Since $\kappa\ge1$ in the class, this gives $\Omega(\log\frac\Delta\varepsilon)$ iterations even for $\kappa=\Theta(1)$.
- **Network term.** For $\kappa\ge7$, the R5 term is $\ge\frac n2\big(0.038\sqrt\kappa\log\frac{16\Delta}{\kappa\varepsilon}-2\big)$, because $\ell(\kappa/7)\le4.9\sqrt{7/\kappa}$. For $1\le\kappa\le7$ it is $\ge\frac n2\big(\log\frac{16\Delta}{\kappa\varepsilon}/(2\log42)-2\big)$, i.e. $\Omega(n\log\frac\Delta\varepsilon)$.
  - R5 applies as soon as $1-\rho\le1/(4n^2)$, because $1-\tfrac y2\ge\sqrt{1-y}$.
  - In that range $n\le\frac12(1-\rho)^{-1/2}$. So the term equals $\Theta(\sqrt\kappa(1-\rho)^{-1/2}\log\frac\Delta{\kappa\varepsilon})$ when $1-\rho=\Theta(1/n^2)$, and stays $\Theta(n\sqrt\kappa\log\frac\Delta{\kappa\varepsilon})$ for $\rho$ closer to $1$; see §6, conjecture 2.

- **Network term of R6.** For $\kappa\ge7$ and $K\ge8$, the last term is $\ge(\frac K2-2)\big(0.038\sqrt\kappa\log\frac{16\Delta}{\kappa\varepsilon}-2\big)$. When the bracket is nonnegative this is $\ge\frac K4\big(0.038\sqrt\kappa\log\frac{16\Delta}{\kappa\varepsilon}-2\big)$. With $K=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}$ this is $\Omega(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log\frac\Delta{\kappa\varepsilon})$ once $\varepsilon$ is small.

This gives the two displays of §0.

---

## 3. Proof of R1 and R2 (zero-chain with alternating holders)

### 3.1 Quadratic building blocks (`Theory/ChainBasic.lean`)

For $x\in\mathbb R^d$ write $x_k$ for $k<d$ and $x_k:=0$ for $k\ge d$ (`cx`). Let $e_k$ be the $k$-th coordinate vector, with $e_k:=0$ for $k\ge d$ (`ev`). The **link vectors** are
$$u_k:=e_{k-1}-e_k\ (k\ge1),\qquad u_0:=-e_0 .$$
Note $u_d=e_{d-1}$, because $e_d=0$. For $U:\mathbb N\to\mathbb R^d$, $S\subseteq\mathbb N$ finite, weights $w$ and offsets $c$, set
$$\mathrm{quad}(x)=\frac\beta2\sum_{k\in S}w_k\big(\langle U_k,x\rangle+c_k\big)^2 .$$

- **Gradient** (`hasGradientAt_quad`): $\nabla\mathrm{quad}(x)=\beta\sum_{k\in S}w_k(\langle U_k,x\rangle+c_k)U_k$.
- **Smoothness** (`norm_sum_le`, `lsmooth_quad`). Assume $0\le w_k\le1$ and $\sum_{k\in S}\langle U_k,z\rangle^2\le K\|z\|^2$ for all $z$. Put $y=\sum_k w_k\langle U_k,v\rangle U_k$. Then
$$\|y\|^2=\sum_kw_k\langle U_k,v\rangle\langle U_k,y\rangle\le\sqrt{K}\|v\|\sqrt K\|y\|$$
by Cauchy–Schwarz, so $\|y\|\le K\|v\|$. Hence $\nabla\mathrm{quad}$ is $\beta K$-Lipschitz.
- **Bound for the chain vectors** (`sum_inner_u_sq_le`). $\langle u_k,z\rangle=z_{k-1}-z_k$, so $\sum_{k\le d}\langle u_k,z\rangle^2\le2\sum z_{k-1}^2+2\sum z_k^2\le4\|z\|^2$. With $\beta=L/4$, every $\mathrm{quad}(u,L/4,S,w,c)$ with $S\subseteq\{0,\dots,d\}$ and $0\le w\le 1$ is $L$-smooth.
- **Support propagation** (`quadGrad_u_vanish`). Suppose $c_k=0$ for $k\ne0$ and $x_k=0$ for all $k\ge j$. Then $\nabla\mathrm{quad}(x)$ vanishes on all coordinates $\ge j+\mathbf 1[j\in S]$.
  - If $k>j$, the coefficient $\langle u_k,x\rangle+c_k=x_{k-1}-x_k=0$.
  - If $k<j$, then $\mathrm{supp}\,u_k\subseteq\mathrm{Pre}_j$.
  - If $k=j$, then $\mathrm{supp}\,u_j\subseteq\mathrm{Pre}_{j+1}$.

  So **only the link $j$ can move a vector from $\mathrm{Pre}_j$ to $\mathrm{Pre}_{j+1}$.**

### 3.2 The chain objective and its exact minimizer (`Theory/ChainValue.lean`)

Fix $d\ge1$, $\alpha,\mu>0$, $R\in\mathbb R$ and $q\in(0,1)$ with
$$\mu q=\alpha(1-q)^2 .\tag{3.1}$$
Take weights $w_k=1$ for $k<d$, $w_d=1-q$, and offsets $c_0=R$, $c_k=0$ otherwise. Define
$$\phi_c(x)=\frac\alpha2\Big[(x_0-R)^2+\sum_{k=1}^{d-1}(x_{k-1}-x_k)^2+(1-q)x_{d-1}^2\Big]+\frac\mu2\|x\|^2=\mathrm{quad}(u,\alpha,\{0..d\},w,c)(x)+\tfrac\mu2\|x\|^2,$$
and $x^\star_k:=Rq^{k+1}$.

**Lemma 3.1 (exact expansion, `φc_eq`).** For all $x$,
$$\phi_c(x)=\phi_c(x^\star)+\frac\alpha2\sum_{k\le d}w_k\langle u_k,x-x^\star\rangle^2+\frac\mu2\|x-x^\star\|^2 .$$

*Proof.*
1. Put $v=x-x^\star$ and expand both squares. It suffices that the linear term
$$\alpha\sum_k w_kg_k\langle u_k,v\rangle+\mu\langle x^\star,v\rangle$$
vanishes, where $g_k=\langle u_k,x^\star\rangle+c_k$.
2. One checks $w_kg_k=R(1-q)q^k$ for every $k\le d$ (`wg_xstar`). The boundary terms work out: $k=0$ uses the anchor, and $k=d$ uses the weight $1-q$; this is why the terminal weight is $1-q$.
3. Summation by parts gives
$$\sum_{k\le d}q^k(v_{k-1}-v_k)=(q-1)\sum_{k<d}q^kv_k .$$
4. Also $\langle x^\star,v\rangle=Rq\sum_{k<d}q^kv_k$.
5. Hence the linear term equals $R\,(\mu q-\alpha(1-q)^2)\sum_kq^kv_k=0$ by (3.1) (`lin_part`). ∎

**Consequences.**
- $x^\star$ is the minimizer (`φc_xstar_le`).
- If $x_K=0$ for some $K<d$, then $\phi_c(x)-\phi_c(x^\star)\ge\frac\mu2(x^\star_K)^2=\frac\mu2R^2q^{2K+2}$ (`φc_gap_ge`).
- $\phi_c(0)=\frac\alpha2R^2$ (`φc_zero`).
- The optimal value is $\phi_c(x^\star)=\frac\alpha2R^2(1-q)$ (`φc_xstar_val`, used for R3). To see this, apply the linear identity with $v=x^\star$: it gives $\mu\|x^\star\|^2=\alpha\sum w_kg_kc_k-\alpha\sum w_kg_k^2=\alpha R^2(1-q)-\alpha\sum w_kg_k^2$.

**Choice of $q$.** With $\kappa':=1+4\alpha/\mu$ and $s:=\sqrt{\kappa'}$, the value $q=\frac{s-1}{s+1}$ satisfies (3.1). Moreover $\log(1/q)=\ell(4\alpha/\mu)=\log(1+\frac2{s-1})\le\frac2{s-1}$.

### 3.3 The hard instance (`Theory/ChainInstance.lean`)

Let $S^{\rm ev}=\{k\le d: k\text{ even}\}$ and $S^{\rm od}=\{k\le d:k\text{ odd}\}$. Every component of a node $i$ equals

| node | component |
|---|---|
| $i\in S_A$ | $\mathrm{quad}(u,L/4,S^{\rm ev},w,c)$ |
| $i\in S_B$ | $\mathrm{quad}(u,L/4,S^{\rm od},w,c)$ |
| otherwise | $0$ |

The other data are $h=\frac\mu2\|\cdot\|^2$ and $x_0=0$ (`chainInst`).

Since $S_A\cap S_B=\emptyset$ and $|S_A|=|S_B|=a$, we get $f=\mathrm{quad}(u,\alpha,\{0..d\},w,c)$ with $\alpha=aL/(4n)$, and so $\phi=\phi_c$ (`chainInst_f`, `chainInst_φ`). Membership in the class (`chainInst_inClass`):
- the components are $L$-smooth (§3.1) and convex (sums of squares of affine maps, `quad_convex`);
- $f$ is convex and $\phi$ is $\mu$-strongly convex: $\phi-\frac\mu2\|\cdot\|^2=f$, combined with `strongConvexOn_iff_convex`;
- $h$ satisfies the prox requirement by `isProx_halfSq` (`Sanity.lean`);
- the minimizer is $x^\star$;
- $\phi(0)-\phi^\star\le\phi(0)=\frac\alpha2R^2$, because $\phi\ge0$. We choose $R^2=2\Delta/\alpha$.

### 3.4 Support propagation through the run (`Theory/Support.lean`)

**Lemma 3.2 (`allowed_subset_lev`).** Fix a raise rule $r(i,j)\in\{0,1\}$. Assume:
- $x_0=0$;
- every gradient answer of node $i$ maps $\mathrm{Pre}_j$ into $\mathrm{Pre}_{j+r(i,j)}$;
- prox answers preserve $\mathrm{Pre}_j$.

Define levels $\ell_s(i)$ by $\ell_0\equiv0$ and
- communication micro-steps: $\ell_{s+1}(i)=\max\{\ell_s(i),\max_{j:W_{ij}\ne0}\ell_s(j)\}$;
- oracle micro-steps: $\ell_{s+1}(i)=\ell_s(i)+r(i,\ell_s(i))$.

Then, for every seed $\omega$, every micro-step $s$ and every node $i$, the zero-respecting set `allowed` of node $i$ is contained in $\mathrm{Pre}_{\ell_s(i)}$.

*Proof.* Induction on $s$, following the recursion of `Algorithm.hist`.
- A received message is $\sum_jW_{ij}z_j$. Its support lies in $\bigcup_{j:W_{ij}\ne0}\mathrm{supp}\,z_j$, and each $z_j$ is supported in the allowed set of node $j$ (zero-respecting sends).
- An oracle query is supported in the allowed set (zero-respecting calls), so its answer is supported in $\mathrm{Pre}_{\ell+r}$. ∎

The averaged output lies in $\mathrm{Pre}_K$ if every node's allowed set does (`supp_avgOutput`). **No probability is involved.** All our proofs show $\phi(\hat x_t(\omega))-\phi^\star>\varepsilon$ for **every** seed $\omega$ and every $t<T$. So the lower bounds hold even for success with positive probability, not only in expectation. The bound holds for every seed, which is why randomization does not help zero-respecting algorithms here, and why no measurability is needed. The lower integral of a function $\ge g_0$ is $\ge g_0$ (`lintegral_mono`).

**Lemma 3.3 (potential, `arr_lev_le`, `lev_le`).** Let $\delta$, $D$, $S_A$, $S_B$ be as in R1. Let $r(i,j)=1$ iff ($i\in S_A$ and $j$ even) or ($i\in S_B$ and $j$ odd); this is the instance's rule, by §3.1. Define the arrival function
$$\mathrm{arr}(k,i)=\begin{cases}0,&k=0,\\(k-1)D+\delta(i),&k\text{ odd},\\(k-1)D+D-\delta(i),&k\ge2\text{ even}.\end{cases}$$
Then $\mathrm{arr}(\ell_s(i),i)\le\lfloor(s+2)/5\rfloor$, the number of communication rounds before micro-step $s$. Consequently, after $t$ iterations every node satisfies $\ell_{5t}(i)\le\lfloor t/D\rfloor+1$.

*Proof.* Induction on $s$.
- **Communication step.** The maximum is attained at some $j$ with $W_{ij}\ne0$, or at $i$ itself. Since $|\delta(i)-\delta(j)|\le1$, we have $\mathrm{arr}(k,i)\le\mathrm{arr}(k,j)+1$, and the round counter increases by one.
- **Oracle step.** A raise from an even $k$ happens only at $\delta(i)=0$. There $\mathrm{arr}(k+1,i)=kD=\mathrm{arr}(k,i)$. A raise from an odd $k$ happens only at $\delta(i)=D$. There $\mathrm{arr}(k+1,i)=kD+D-D=\mathrm{arr}(k,i)$.

Finally $(\ell-1)D\le\mathrm{arr}(\ell,i)\le t$. ∎

Intuition: each new level needs the information to cross from the holders of the previous link to the holders of the next link, which takes $\ge D$ communication rounds.

### 3.5 Conclusion of R1 (`Theory/LowerChain.lean`, `lower_bound`)

Take $d=T+2$. Let $t<T$ and $K=\lfloor t/D\rfloor+1<d$.
1. For every seed, the averaged output vanishes at coordinate $K$, so its gap is at least
$$\frac\mu2R^2q^{2K+2}=\frac{4n\mu\Delta}{aL}q^{2K+2}.$$
2. Hence $\mathbb E[\mathrm{gap}]$ is at least the same quantity.
3. `key_ineq` shows that this exceeds $\varepsilon$:
   - $2K+2\le 2t/D+4$;
   - $(2t/D+4)\log(1/q)<\log\frac{4n\mu\Delta}{aL\varepsilon}$ by the hypothesis on $T$ and $t<T$;
   - here $\log(1/q)=\ell(aL/(n\mu))$, since $q=\frac{s-1}{s+1}$ with $s=\sqrt{1+aL/(n\mu)}$.
4. So no $t<T$ meets the target, i.e. $T\le T_\varepsilon$. ∎

### 3.6 Proof of R2 (`Theory/Corollaries.lean`)

Take $W=J$. It lies in $\mathcal W_{n,\rho}$ for $\rho\ge0$ (`J_commClass`).
- $a=\lfloor n/2\rfloor$, so $a/n\in[1/3,1/2]$.
- $S_A=\{i<a\}$ and $S_B=\{a\le i<2a\}$.
- $\delta=\mathbf 1[i\ge a]$ and $D=1$.

Apply R1. The transfer uses that $\ell$ is decreasing and $\log\frac{4n\Delta}{a\kappa\varepsilon}\ge\log\frac{8\Delta}{\kappa\varepsilon}$ (`Transfer.transfer`, `Transfer.ell_anti`). ∎

---

## 4. Proof of R3 (one node holds $m$ private chains) — `Theory/Blocks.lean`, `LowerData*.lean`

**Instance.** Take $d=mb$ with $b=5T+1$, and identify $\mathbb R^{mb}$ with $m$ blocks of $\mathbb R^b$ (`finProdFinEquiv`). Let $P_\xi$ be the block projections and $E_\xi=P_\xi^*$ the block embeddings (`inner_Em`, `norm_sq_eq_sum_P`).
- Node $i_0=0$ holds the $m$ components $F_{i_0}(x;\xi)=\mathrm{quad}_b(u,L/4,\{0..b\},w,c)(P_\xi x)$, a chain on block $\xi$.
- All other nodes hold $0$, and $h=\frac\mu2\|\cdot\|^2$.

Then $\nabla F_{i_0}(x;\xi)=E_\xi\nabla\mathrm{quad}(P_\xi x)$ (`hasGradientAt_comp_P`). This map is $L$-smooth because $\|P_\xi\|\le1$ and $E_\xi$ is an isometry (`lsmooth_bq`), and it is convex (`bq_convex`). The objective splits as
$$\phi(x)=\sum_\xi\phi_c(P_\xi x)$$
with $\alpha=L/(4nm)$ (`blockInst_φ`), the minimizer has $x^\star$ in every block, and $\phi^\star=m\frac\alpha2R^2(1-q)$. With the exact optimal value (§3.2) the initial gap is **exactly** $m\frac\alpha2R^2q$. We choose $R^2=2\Delta/(m\alpha q)$ (`blockInst_inClass`).

**Budget invariant (`exists_budget`).** For every seed and micro-step $s$ there is $k:[m]\to\mathbb N$ with $\sum_\xi k_\xi\le s$ such that all nodes' allowed sets lie in $\{\text{first }k_\xi\text{ coordinates of block }\xi\ \forall\xi\}$.
- Only node $i_0$'s gradient query to component $\xi$ can deepen block $\xi$, and only by one (§3.1 applied blockwise).
- Communication and prox steps create no new coordinates.

**Gap.** Fix a seed with budget $k$, so $\sum_\xi k_\xi\le5t$. By §3.2 blockwise,
$$\phi(\hat x_t)-\phi^\star\ \ge\ \frac\mu2R^2\sum_\xi q^{2k_\xi+2}=\frac{\Delta(1-q)^2}{m}\sum_\xi q^{2k_\xi}.$$
- **Counting.** $q^{2k+2}\ge q^2(1-k)$ for $k\in\mathbb N$ (`pow_ge_lin`), so the gap is $\ge\Delta(1-q)^2(1-5t/m)$.
- **AM–GM.** $\sum_\xi q^{2k_\xi}\ge m\,q^{2\sum_\xi k_\xi/m}\ge m\,q^{10t/m}$ (`sum_pow_ge`), so the gap is $\ge\Delta(1-q)^2q^{10t/m}$.
- **Constant.** Here $s=\sqrt{1+\kappa/(nm)}=\sqrt{1+4\alpha/\mu}$ and $q=\frac{s-1}{s+1}$, so $(1-q)^2=4/(s+1)^2\ge1/\lambda$, since $(s+1)^2\le2s^2+2=4\lambda$.

The hypothesis on $T$ then gives gap $>\varepsilon$ for all $t<T$.
- *Counting branch*: directly.
- *AM–GM branch*: since $\frac{10t}m\log\frac1q<\log\frac\Delta{\lambda\varepsilon}$ and $\frac1q=\frac{s+1}{s-1}$, we get $\frac\Delta\lambda q^{10t/m}>\varepsilon$. ∎

---

## 5. Proofs of R4, R5 and R6; the lazy path and the path of cliques

**R4** (`Theory/Combined.lean`). Apply R3 with $W=J$, or R2. ∎

**The lazy path** (`Theory/PathSeq.lean`, `Theory/PathW.lean`). Let $W^{\rm path}=I-\frac14L_{\rm path}$, where $L_{\rm path}$ is the Laplacian of the path $0-1-\dots-(n-1)$.

**Lemma 5.1 (`pathW_commClass`).** $W^{\rm path}\in\mathcal W_{n,\rho}$ for every $\rho\ge\sqrt{1-1/(2n^2)}$.

*Proof.* Nonnegativity, symmetry and unit row sums are direct, using `pathW_mulVec` for the row sums. For the contraction, write $\bar v$ for the mean, $u=v-\bar v\mathbf 1$ and $S=\sum_k(u_k-u_{k+1})^2$. Then $(W-J)v=u-\frac14L u$ (`lap_shift`), and
$$\|u-\tfrac14Lu\|^2=\|u\|^2-\tfrac12\langle u,Lu\rangle+\tfrac1{16}\|Lu\|^2 ,$$
where:
- $\langle u,Lu\rangle=S$ (`lap_inner`);
- $\|Lu\|^2\le4S$ (`lap_sq_le`);
- the discrete Poincaré inequality $\|u\|^2\le\frac{n^2}2S$ for mean-zero $u$ (`poincare`). It follows from telescoping plus Cauchy–Schwarz, $(u_i-u_j)^2\le nS$, together with $\sum_{i,j}(u_i-u_j)^2=2n\|u\|^2$.

Hence $\|(W-J)v\|^2\le(1-\frac1{2n^2})\|u\|^2\le(1-\frac1{2n^2})\|v\|^2$. ∎

**R5** (`Theory/PathCorollary.lean`). In the support graph of $W^{\rm path}$, take:
- $a=\lfloor n/4\rfloor$, $S_A=\{i<a\}$ and $S_B=\{i\ge n-a\}$;
- $D=n-2a+1\ge n/2$;
- $\delta(i)=\min\{(i-a+1)^+,D\}$, which is 1-Lipschitz along path edges.

R1 then gives the network term, using $a/n\ge1/7$ and $n/a\ge4$. R3 with $W=W^{\rm path}$ gives the data terms. ∎

**The path of $K$ cliques** (`Theory/Cliques.lean`).

*Construction.*
- Cut the nodes into $K$ consecutive groups: $g(i)=\min\{\lfloor i/s\rfloor,K-1\}$ with $s=\lfloor n/K\rfloor$.
- Every group has size $s_g\ge s$ (`sz_ge`), and $\sum_gs_g=n$.
- With $\theta=s/4$, set
$$W_{ij}=\begin{cases}\big(1-\tfrac{c^-_g+c^+_g}{s_g}\big)/s_g,&g(i)=g(j)=g,\\ \theta/(s_gs_h),&|g(i)-g(j)|=1,\\0,&\text{otherwise},\end{cases}$$
  where $c^-_g=\theta\,\mathbf 1[g\ge1]$ and $c^+_g=\theta\,\mathbf 1[g+1<K]$ (`cqW`).

*Action on vectors.* $(Wv)_i=\bar y_g-\frac\theta{s_g}(L_K\bar y)_g$ for $g=g(i)$ (`cqW_mulVec`). Here $\bar y_g$ are the group means and $L_K$ is the path Laplacian on the $K$ groups. In words, $W$ averages inside groups and is a lazy path on the group means.

**Lemma 5.2 (`cqW_commClass`).** For $1\le K\le n$ and $\rho\ge\sqrt{1-1/(4K^2)}$, $W\in\mathcal W_{n,\rho}$.

*Proof.* Nonnegativity uses $c^\pm_g\le\theta=s/4\le s_g/4$. Symmetry is `coef_symm`. Row sums equal 1 because the Laplacian of the constant sequence vanishes. For the contraction, let $z_g=\bar y_g-\bar v$, so $\sum_gs_gz_g=0$, and let $E=\sum_{g<K-1}(z_g-z_{g+1})^2$. Then
$$\|(W-J)v\|^2=\sum_gs_g\Big(z_g-\tfrac\theta{s_g}(L_Kz)_g\Big)^2=\sum_gs_gz_g^2-2\theta E+\theta^2\sum_g\tfrac{(L_Kz)_g^2}{s_g}\le\sum_gs_gz_g^2-\theta E,$$
using $\sum(L_Kz)^2\le4E$ and $s_g\ge s=4\theta$.
- *Weighted Poincaré inequality.* From $\sum_{g,h}s_gs_h(z_g-z_h)^2=2n\sum_gs_gz_g^2$ and $(z_g-z_h)^2\le KE$ we get $\sum_gs_gz_g^2\le\frac{nK}2E$.
- *Group means versus the vector.* $\sum_gs_gz_g^2\le\sum_i(v_i-\bar v)^2\le\|v\|^2$, by Cauchy–Schwarz inside each group.
- *Constant.* $2\theta/(nK)=s/(2nK)\ge1/(4K^2)$, because $n<K(s+1)\le2Ks$.

Together, $\|(W-J)v\|^2\le(1-\frac1{4K^2})\|v\|^2$. ∎

**R6** (`Theory/CliquesCorollary.lean`). The support graph of $W$ only joins equal or consecutive groups (`cqW_ne_zero`). Take:
- $a=\lfloor n/4\rfloor$, $S_A=\{i<a\}$, $S_B=\{i\ge n-a\}$;
- $g_A=g(a-1)$, $g_B=g(n-a)$, $D=g_B-g_A$;
- $\delta(i)=\min\{(g(i)-g_A)^+,D\}$, which is 1-Lipschitz on the support graph.

The group arithmetic ($sK\le n<K(s+1)$, $s\,g_A\le a-1$, and either $n-a<s(g_B+1)$ or $g_B=K-1$) gives $K<2D+4$. Then R1 gives the network term. R3 with this $W$ gives the data terms, and R2 gives the $\sqrt\kappa\log$ term. ∎

**Lemma 5.3 (`cqW_posSemidef`).** The path of cliques is positive semidefinite.

*Proof.* Grouping by groups and writing $\bar y$ for the group means,
$$v^\top Wv=\sum_gs_g\bar y_g^2-\theta\sum_{g<K-1}(\bar y_g-\bar y_{g+1})^2 .$$
The edge sum is $\le4\sum_g\bar y_g^2\le\frac4s\sum_gs_g\bar y_g^2$ (`edgeSum_le_four`), and $\theta=s/4$. Hence $v^\top Wv\ge0$. ∎

**R8** (`Theory/CliquesPSD.lean`). Same proof as R6, with the witness $W$ exhibited together with Lemmas 5.2 and 5.3. ∎

---

## 6. Matching regimes (C) and open gaps

Upper bound (Liu–Low Cor. 5.3), applied to $W'=(I+W)/2$:
- $W'$ is computable in one round, since each node knows its own message;
- $0\preceq W'\preceq I$ and $1-\lambda_2(W')\ge(1-\rho)/2$;
- the iteration count is at most the SFO count plus the number of communication rounds.

This gives
$$T_{\rm up}=\widetilde O\Big((m+m^{3/4}\sqrt\kappa)\log\tfrac\Delta\varepsilon+\frac{\sqrt m+m^{1/4}\sqrt\kappa}{\sqrt{1-\rho}}\log\tfrac\Delta\varepsilon\Big).$$

**Corollary C1 (formal lower bound R4 + Liu–Low).** In the range $\kappa\le\sqrt m$, $1-\rho\ge1/m$, $\varepsilon\le\Delta/3$:
$$\mathrm{minimax}=\widetilde\Theta\big(m\log\tfrac\Delta\varepsilon\big).$$
- *Lower bound* (R4, data term): $\mathrm{minimax}\ge\frac m{10}\log\frac{2\Delta}{3\varepsilon}\big/\log\frac{6nm}\kappa$.
- *Upper bound* (Liu–Low): $\widetilde O(m\log\frac\Delta\varepsilon)$, because $m^{3/4}\sqrt\kappa\le m$ and $(\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2}\le2m$ in this range.

They match up to the factor $\log\frac{6nm}\kappa$ and Liu–Low's polylogarithmic communication factors. None of these factors depends on $\varepsilon$. Only the lower half is formalized (R4); the upper half is the cited human result.

**Corollary C2 (R6/R2 + Liu–Low).** Let $m=O(1)$, $n\ge8$, $\kappa\ge1$ (always true in the class, since $L\ge\mu$), $1-\rho\ge c_1/n^2$ for a constant $c_1$, and $\varepsilon\le c_2\Delta/\kappa$ for a small absolute constant $c_2$. Then
$$\mathrm{minimax}=\widetilde\Theta\Big(\sqrt{\tfrac{\kappa}{1-\rho}}\log\tfrac\Delta\varepsilon\Big),$$
up to logarithmic factors.
- *Upper bound* (Liu–Low): $\widetilde O((1+\sqrt\kappa)(1-\rho)^{-1/2}\log\frac\Delta\varepsilon)$.
- *Lower bound* (R6): take $K=\min\{n,\lfloor1/(2\sqrt{1-\rho^2})\rfloor\}=\Theta((1-\rho)^{-1/2})$ in this range. The network term is $\ge(\frac K2-2)\big(\log\frac{16\Delta}{\kappa\varepsilon}/(2\ell(\kappa/7))-2\big)$, and $\ell(\kappa/7)\le\min\{4.9\sqrt{7/\kappa},\log30\}$ for $\kappa\ge1$. This gives $\Omega(K\max\{1,\sqrt\kappa\}\log\frac\Delta{\kappa\varepsilon})$. When $K<8$, i.e. for $\rho$ bounded away from 1, R2 gives the bound instead.

So for $O(1)$ components per node, the network-dependent complexity is determined up to logarithmic factors on the whole range where $(1-\rho)^{-1/2}\lesssim n$. Only the lower half is ours and formalized.

**Open gaps and conjectures.**

1. *Conjecture* (data term). Under individual smoothness, the true dependence lies between $\sqrt{m\kappa/n}$ (R3, R4) and $m^{3/4}\sqrt\kappa$ (Liu–Low). Our constructions have convex components and cannot exceed $\sqrt{m\kappa/n}$.
   - With node-private components, a link held by $r$ components contributes curvature $\le rL/N$ to $f$.
   - Pathwise support arguments are blind to which component the algorithm queries.

   Closing this gap is the open problem named by Zhou (arXiv:2611.05807) for the centralized case.
2. *Conjecture* (network term when $(1-\rho)^{-1/2}\gg n$). With exact real arithmetic and only zero-respecting constraints, communication is limited by **graph distance**, not by the spectral gap; see R1. Since any $W\in\mathcal W_{n,\rho}$ has diameter $\le1+\operatorname{arccosh}(n)/\operatorname{arccosh}(1/\rho)$ (Chebyshev argument), we conjecture that the network term of the minimax complexity is $\widetilde\Theta(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log\frac1\varepsilon)$. Note that $\min\{n,(1-\rho)^{-1/2}\}$ is not $(1-\rho)^{-1/2}$ when $\rho$ is extremely close to 1. For example, for $n=2$ every $W\in\mathcal W_{2,\rho}$ lets each node recover its neighbour's message exactly. R6 proves the lower half $\Omega(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log)$ for every $\rho$. What remains open is the upper half when $1-\rho\ll1/n^2$, where Liu–Low's bound grows like $(1-\rho)^{-1/2}$ while ours stays at $n$.

---

## 7. Comparison with human results (precise statements, translation, classification)

Notation translation: Liu–Low and Hendrikx et al. use $m$ for the number of nodes and $n$ for the number of components per node (Liu–Low), or the reverse (Hendrikx et al.); below everything is in **our** notation ($n$ nodes, $m$ components per node, $N=nm$). Our iteration allows one component gradient, one prox and one communication round per node, so a "time" or "SFO + rounds" count of a human result converts to iterations up to a factor 2.

### 7.1 Liu–Low, *Decentralized Sum-of-Nonconvex Optimization*, AAAI 2024, arXiv:2402.02356v1, Thm. 5.1 / Cor. 5.3 (upper bound)

- *Assumptions*:
  - $F=f+\psi$ is $\sigma$-strongly convex;
  - $f$ is convex and $L$-smooth;
  - each component is $(\ell_1,\ell_2)$-smooth, i.e. $-\frac{\ell_2}2\|x-y\|^2\le f(x)-f(y)-\langle\nabla f(y),x-y\rangle\le\frac{\ell_1}2\|x-y\|^2$;
  - $W$ is symmetric doubly stochastic, $0\preceq W\preceq I$, $\mathrm{null}(I-W)=\mathrm{span}(\mathbf 1)$.
- *Conclusion*: $\epsilon$-accuracy with $O\big((m+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}m^{3/4})\log\frac{F(\bar y^0)-F^\star}\epsilon\big)$ SFO calls per node and $\widetilde O\big(\frac{\sqrt m+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}m^{1/4}}{\sqrt{1-\lambda_2(W)}}\log\frac{F(\bar y^0)-F^\star}\epsilon\big)$ communication rounds.
  - Communication uses Chebyshev-accelerated FastMix, with $M$ rounds per call.
  - With $\ell_1=\ell_2=L$ this is the $T_{\rm up}$ of §6.
- *Relation*: no lower bound in that paper; we prove no upper bound. Our results bound the same quantity from below and match it in the ranges of §6.

### 7.2 Woodworth–Srebro, arXiv:1605.08003v3, Thm. 8 (centralized randomized lower bound)

- *Statement*: for any $m'\ge2$, $\gamma/\lambda>161m'$, $\epsilon_0>60\epsilon\sqrt{\gamma/(\lambda m')}$ and any randomized algorithm with oracle $h_F(x,i,\beta)=[f_i(x),\nabla f_i(x),\mathrm{prox}_{f_i}(x,\beta)]$, there are $m'$ functions, $\gamma$-smooth and $\lambda$-strongly convex, with $F(x_0)-F^\star=\epsilon_0$, such that $\mathbb E[F(\hat x)-F^\star]<\epsilon$ needs $\Omega\big(m'+\sqrt{m'\gamma/\lambda}\log(\frac{\epsilon_0}\epsilon\sqrt{m'\lambda/\gamma})\big)$ queries.
- *Transfer to our problem*:
  - take $m'=N$, $\gamma\le L$, $\lambda=\mu$, $h=0$;
  - a $T$-iteration decentralized algorithm is simulated by $nT$ queries;
  - this gives $T=\Omega(m+\sqrt{m\kappa/n}\log(\frac\Delta\varepsilon\sqrt{nm/\kappa}))$ for $\kappa>161nm$.
- *R3 is classified **equal***. In WS's range it has the same orders $m$ and $\sqrt{m\kappa/n}\log$, with the log argument $\frac{nm\Delta}{\kappa\varepsilon}$ instead of $\frac\Delta\varepsilon\sqrt{nm/\kappa}$, which is equal up to a factor 2 for small $\varepsilon$. For $\kappa\le nm$, where WS Thm 8 is silent, it gives $\Omega(m\log\frac\Delta\varepsilon/\log\frac{nm}\kappa)$.
  - Minor gains: no condition $\kappa>161N$; every $W$; the instance is independent of $A$; fully formalized.
  - Loss: it holds only for zero-respecting algorithms, whereas WS covers all randomized algorithms. The problem's class is zero-respecting, so this loss is irrelevant for the problem.
- *R4 is classified **stronger***. For this problem it dominates the WS-implied bound up to constants and the log argument, and it is larger by $\sqrt{n/m}$ when $n>m$: its term $\sqrt\kappa\log\frac\Delta{\kappa\varepsilon}$ is not implied by WS. It also applies for $\kappa\le161nm$, where Thm. 8 does not apply.
  - Context: a $\sqrt\kappa\log\frac1\varepsilon$ term is classical for a single machine (Nesterov's lower bound). For parallel or graph-oracle models it is known in related forms, e.g. Woodworth et al., arXiv:1805.10222, and Scaman Thm 2 with a complete graph, for span-based procedures.
  - It does not follow from the single-machine bound by simulation, because a decentralized iteration makes $n$ parallel queries. Its novelty here is the combination, in the problem's own model, with the data terms.

### 7.3 Scaman–Bach–Bubeck–Lee–Massoulié, ICML 2017, arXiv:1702.08704v2, Thm. 2 (decentralized lower bound)

- *Statement*: for $\alpha,\beta>0$ and $\gamma\in(0,1]$ there exist a gossip matrix of eigengap $\gamma$ and $\alpha$-strongly convex, $\beta$-smooth $f_i$ such that every black-box procedure using $W$ has
$$\bar f(\theta_{i,t})-\bar f(\theta^*)\ge\frac{3\alpha}2(1-\frac{16}{\sqrt{\kappa_l}})^{1+t/(1+\tau/(5\sqrt\gamma))}\|\theta_{i,0}-\theta^*\|^2 .$$
- *Model and construction*:
  - black-box procedures are span-based: iterates lie in spans of past iterates, local gradients and conjugate gradients, and of neighbours' memories;
  - in the proof $n$ is chosen so that $\gamma_n\ge\gamma>\gamma_{n+1}$, i.e. $n=\Theta(\gamma^{-1/2})$;
  - the hard nodes are the first $\lceil n/32\rceil$ nodes of a path, and each node holds a single function.
- *R1 is classified **incomparable***.
  - *R1 gains*:
    - the algorithm class is all randomized zero-respecting algorithms with nonlinear local computation and adaptive component selection, which is strictly larger than span-based procedures;
    - $n$ is fixed and $W$ is arbitrary (per-$W$ bound);
    - the metric is the problem's averaged-output gap;
    - finite sums and the exact per-iteration budget of the problem are handled.
  - *Scaman gains*: an explicit spectral-gap parametrization for every $\gamma$, and a stronger oracle (conjugate gradients).
  - Scaman et al. themselves note that the number of nodes is tied to $\gamma$ in their construction. Yuan et al. (arXiv:2210.07863) stress that "this result is not valid in stochastic scenarios where $n$ is required to be fixed", and the task statement points out the same issue for this problem.

### 7.4 Hendrikx–Bach–Massoulié, arXiv:2005.10675v1, Thm. 1 and Cor. 3 (decentralized finite-sum lower bound)

- *Thm. 1*: for a graph of size $n$ and diameter $\Delta_G$, there exist $n\times m$ convex $L_{i,j}$-smooth functions with local $\sigma_i$-strong convexity such that every black-box procedure satisfies
$$2\frac{1-q^2}{1-q}\mathbb E\frac{\|\theta^t-\theta^*\|^2}{\|\theta^0-\theta^*\|^2}\ge\Big(1-\frac{2m}{m+\sqrt{m\kappa_s/3}}\Big)^{4\lceil t\rceil/m}+\Big(1-\frac2{1+\sqrt{\kappa_\ell/3}}\Big)^{2+2\lceil t\rceil/(\Delta_G\tau)}.$$
  The black-box class is span-based, one component per unit of time, and includes component prox and conjugate gradients.
- *Cor. 3*: there is a gossip matrix with spectral gap $\gamma$ such that the time is $\Omega([m+\sqrt{m\kappa_s}+\tau\sqrt{\kappa_\ell/\gamma}]\log\frac1\varepsilon)$. Its proof is a line graph with $n$ of order $\gamma^{-1/2}$.
- *R1, R3 and R5–R8, compared with Thm. 1 itself, are **incomparable***. Thm. 1 is the closest human analog: it has both data and network terms per graph at fixed $n$, but only for span-based procedures and with locally normalized condition numbers $\kappa_s,\kappa_\ell$. It does not apply to the problem's algorithm class, so it implies nothing for this problem. With respect to the bound directly implied for this problem (WS Thm. 8), R5–R8 are **stronger** (§7.7).
  - Ours hold for the problem's larger, zero-respecting, randomized class, with the global normalization of the problem ($L$ per component, $\mu$ of $\phi$) and the problem's metric.
  - Theirs has the $\log(1/\varepsilon)$ factor on the data term, under span assumptions.
  - Hendrikx et al. use local strong convexity $\sigma_i$ and local condition numbers. In our normalization, putting all hard links on few nodes dilutes the coupling by $a/n$; this is why R1 is stated with $|S_A|=a$.
- Cor. 3 (and Scaman Cor. 2) choose $n$ from $\gamma$, so for fixed $n$ they do not imply R5 or R6. R6 is the fixed-$n$ statement in the problem's own model: the same $\sqrt{\kappa/\gamma}$-type network term, valid for every $\rho$ as $\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa$, for all randomized zero-respecting algorithms.

### 7.5 Zhou–Gu, ICML 2019, arXiv:1901.11224v1, Thm. 4.1 (sum-of-nonconvex, average smoothness)

- *Statement*: for linear-span randomized algorithms there are $\{f_i\}\in\mathcal V^{(L)}$ (*average* smoothness) with $F\in\mathcal S^{(\sigma,L)}$ and $F(x^0)-\inf F\le\Delta$, needing $\Omega(n'+n'^{3/4}\sqrt{L/\sigma}\log\frac{n'\sigma\Delta}{L\epsilon})$ IFO calls, for $\epsilon\le8\Delta n'^{7/4}\sigma^{3/2}L^{-3/2}$.
- Their hard components are $\sqrt{n'}\,f_{sc}$ with $f_{sc}\in\mathcal S^{(\alpha,1)}$, hence individually only $\sqrt{n'}L$-smooth. So they are **not** in our class. Sec. 4.3 and the conclusion name the individually smooth case as future work.
- *Relation*: **incomparable / weaker in the data term**. Our class is individually smooth, and we only reach $\sqrt{m\kappa/n}$. We do not claim progress on the $n^{1/4}$ gap.

### 7.6 Other related lower bounds found

- Lu–De Sa, arXiv:2006.08085v4, Thm. 1 / Cor. 1: zero-respecting, synchronous, $B$ queries per worker per iteration, nonconvex stochastic, $\Omega(\frac{\Delta L\sigma^2}{nB\epsilon^4}+\frac{\Delta L D}{\epsilon^2})$ on graphs of diameter $D$; Cor. 1 uses a path with $\lambda=\cos(\pi/n)$. This is a different (nonconvex stochastic) setting, but the same model style; our Lemma 3.3 is the strongly convex chain analog of their distance argument.
- Yuan et al., arXiv:2210.07863: ring lattices give, for fixed $n$, every $\beta\in[0,\cos(\pi/n)]$ with diameter $\Theta(1/\sqrt{1-\beta})$, in a stochastic setting. Our path of cliques (Lemma 5.2) plays the same role, with two $\Theta(n)$-size node sets at distance $\Theta(\min\{n,(1-\rho)^{-1/2}\})$, and gives R6 for all $\rho$.
- Luo et al., arXiv:2210.13931; Metelev et al., arXiv:2402.02490: decentralized finite-sum lower bounds under span/LIFO assumptions, with separate hard instances for computation and communication.
- No paper found gives a decentralized lower bound for sum-of-nonconvex problems, or one that combines individual component smoothness with a network term, valid for general zero-respecting algorithms.

### 7.7 Classification summary (also in `manifest.json`)

| Result | Class | Main comparator | Reason |
|---|---|---|---|
| R1 | incomparable | Scaman Thm 2; Hendrikx Thm 1 | larger algorithm class, fixed $n$, every $W$, problem metric; theirs: $\gamma$-parametrization, stronger oracle, data term |
| R2 | equal | Nesterov's $\sqrt\kappa\log$ bound (graph-oracle form) | classical term, reproved in the exact model |
| R3 | equal | WS Thm 8 (transferred) | same orders where WS applies; also $m\log\frac1\varepsilon/\log\frac{nm}\kappa$ for $\kappa\le nm$; no $\kappa>161N$ condition; zero-respecting class |
| R4 | stronger | WS Thm 8 (transferred) | dominates it, plus $\sqrt\kappa\log$, i.e. $\times\sqrt{n/m}$ when $n>m$ |
| R5 | stronger | WS Thm 8; Scaman Thm 2 / Hendrikx Cor 3 (do not apply at fixed $n$) | network + data terms together at fixed $n$ for the general zero-respecting class (regime $1-\rho\le1/(4n^2)$) |
| R6 | stronger | WS Thm 8; Scaman Cor 2 / Hendrikx Cor 3 (span-based, $n$ tied to $\gamma$); Hendrikx Thm 1 (span-based, per graph) | all four terms for every $\rho$ at fixed $n$, general zero-respecting randomized class; matches Liu–Low for $m=O(1)$, $1-\rho\gtrsim n^{-2}$ |
| R7 | stronger | as R6 | R6 with $\rho$ explicit: network term $\Omega(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log)$ |
| R8 | stronger | as R6 | R6's bound on the smaller class $W\succeq0$, convex components, i.e. inside Liu–Low's assumptions |

---

## 8. Lean formalization map

| File | Content |
|---|---|
| `Theory/ChainBasic.lean` | coordinates, link vectors, `quad`, gradient, smoothness (`lsmooth_quad`), support (`quadGrad_u_vanish`) |
| `Theory/ChainValue.lean` | `φc`, exact expansion `φc_eq`, minimizer, gap bound `φc_gap_ge`, value `φc_xstar_val` |
| `Theory/ChainInstance.lean` | the chain instance and `chainInst_inClass` |
| `Theory/Support.lean` | run invariant `allowed_subset_lev`, potential `arr_lev_le`, `lev_le` |
| `Theory/LowerChain.lean` | `Chain.lower_bound` (R1) |
| `Theory/Transfer.lean` | monotonicity of $\ell$ and the transfer lemma |
| `Theory/Corollaries.lean` | `le_minimax_of`, R2 |
| `Theory/Blocks.lean`, `LowerData.lean`, `LowerData2.lean`, `LowerData3.lean` | block instance, budget invariant, AM–GM, R3 |
| `Theory/Combined.lean` | R4 |
| `Theory/PathSeq.lean`, `Theory/PathW.lean`, `Theory/PathCorollary.lean` | Poincaré inequality, `pathW_commClass`, R5 |
| `Theory/Cliques.lean`, `Theory/CliquesCorollary.lean` | path of cliques `cqW`, `cqW_commClass` (weighted Poincaré), R6, R7 |
| `Theory/CliquesPSD.lean` | `cqW_posSemidef`, R8 |
| `Theory/Statements.lean` / `Theory/Proofs.lean` | the eight statements / their proofs |

All statements use only `Theory/Model.lean` definitions. No custom notation appears in `Statements.lean`.

**How to read the Lean statements.**

*Lower bounds and the minimax quantity.*
- `(T : ℕ∞) ≤ A.hittingTime I W ε` means $T_\varepsilon(A,I,W)\ge T$. Equivalently, $\mathbb E[\phi(\hat x_t)-\phi^\star]>\varepsilon$ for all $t<T$.
- `(T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε` means: for every admissible algorithm there are a dimension, an instance of the class and a matrix of $\mathcal W_{n,\rho}$ on which it needs $\ge T$ iterations.
- Stating the bounds for every natural $T$ below an explicit real expression avoids rounding: the minimax is at least the integer part of that expression.

*Hypotheses of `lower_bound_network` (R1).*
- `hδ : ∀ i j, W i j ≠ 0 → δ i ≤ δ j + 1 ∧ δ j ≤ δ i + 1` says that $\delta$ is 1-Lipschitz on the support graph of $W$.
- `hSA`, `hSB` put $S_A$ at level $0$ and $S_B$ at level $D$.
- `hcard` requires equal sizes.
- The conclusion `∃ d, ∃ I, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧ …` provides a hard instance whose components are convex.

*Hypotheses of `minimax_ge_all` (R6) and `minimax_ge_explicit_rho` (R7).*
- `hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ` is the condition under which the path of $K$ cliques lies in $\mathcal W_{n,\rho}$ (Lemma 5.2).
- In R7, $K$ is `min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊`.
- The expression `Real.log ((Real.sqrt (1 + x) + 1) / (Real.sqrt (1 + x) - 1))` is $\ell(x)$.

*Parameters.*
- $c$ (messages per round) is arbitrary in every statement.
- $\rho$ is arbitrary in R1 and R3, which hold for every matrix $W$.
- The algorithm is arbitrary among admissible ones.
- The instance dimension is chosen after the algorithm, as allowed by the model.

**Remarks on robustness.** The path-of-cliques part below is formalized as R8; the rest is not formalized.
- All hard gossip matrices used ($J$, the lazy path, the path of cliques) are positive semidefinite.
  - $J$ is an orthogonal projection.
  - The lazy path is $I-\frac14L_{\rm path}$ and $L_{\rm path}\preceq4I$.
  - The path of cliques acts as $0$ on within-group deviations, and as a lazy birth–death chain with holding probability $\ge\frac12$ on group means.

  So the minimax lower bounds R2, R4, R5 and R6 also hold for the restricted communication class with $W\succeq0$, which is a "(+)" alternative of `PROFILE.md` §3, and for Liu–Low's network assumptions $0\preceq W\preceq I$.
- All hard instances have convex components; this is formalized in R1 and R3. So the bounds also hold for the "(+)" restriction to convex components.
