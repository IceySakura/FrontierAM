# Separate-chains lower bounds for ULB-profile-63 (decentralized finite-sum, strongly convex, composite)

All results below are **lower bounds**. They are stated with the fixed model `Theory/Model.lean`
and proved completely in Lean (`theory/formal/`, checked with `lake comparator`; the axioms are only
`propext`, `Quot.sound`, `Classical.choice`). There are no upper bounds in this submission.

---

## 0. Summary: target, anchors, and how the results relate to them

**Target.** We prove a lower bound on the **data term**, for the problem's whole algorithm class
$\mathcal A$. That means randomized, zero-respecting algorithms with *adaptive* choice of component
indices, run on the decentralized model with any mixing matrix. The bound holds in the regime
$\kappa-1\le 8N$ (R3), and in H-LB-hly's full form for all $\kappa\ge2$ (R9). The hard instances have
$\mu$-strongly convex, $L$-smooth **quadratic** components. Our claim is the "strictly stronger
than a registered anchor" tier against **H-LB-hly** and **H-LB-zsh**: we prove the same bound for a
strictly larger algorithm class, on the same class of hard instances, and the result is not
dominated by a known human result. We do **not** reach goal A as the task phrases it (beating
H-LB-data for $\kappa\ge\sqrt N$), nor goals B or C (§6 says what is missing).

**The registered anchors we compare against.** Each was checked against the arXiv source.

* **H-LB-hly**: Hannah, Liu, O'Connor, Yin, *Breaking the span assumption yields fast finite-sum
  minimization*, NeurIPS 2018, arXiv:1805.07786v1 (only version).
  * Statement: Theorem 2 (§2.1), in full in the appendix theorem "Lower complexity bound oblivious
    p-CLI algorithms". Verbatim: *"For any oblivious p-CLI algorithm $A$, for all $\mu,L,k$, there
    exist $L$-smooth, and $\mu$-strongly convex functions $f_i$ such that at least
    $K(\epsilon)=\tilde\Omega\big(\big(\frac{n}{1+(\ln(\frac n\kappa))_+}+\sqrt{n\kappa}\big)\ln\frac1\epsilon+n\big)$
    iterations are needed for $A$ to obtain expected suboptimality
    $\mathbb E[f(K(\epsilon))-f(X^*)]<\epsilon$."*
  * Footnote: "We absorb some smaller low-accuracy terms (high $\epsilon$)".
  * Algorithm class: Arjevani–Shamir p-CLI. The oracle is $A\nabla f_j(w)+Bw+C$, and the parameters
    (including the index $j$) do not depend on earlier oracle answers ("oblivious").
  * Hard instance: strongly convex quadratics.
* **H-LB-zsh**: Zhang, Shu, He, arXiv:2010.08766v2 (2022), Theorem 3.1.
  * Verbatim: *"For any $L,\mu>0$, $n\ge2$, $0<\epsilon<\frac{B^2\mu}4$, and any randomized
    incremental gradient method $\mathcal M$, there exist a dimension … and $n$ $(\mu,L)$-smooth
    functions $\{f_i\}$ … such that $\|x^0-x^*\|=B$ … In order to find
    $F(\hat x)-F(x^*)\le\epsilon$, $\mathcal M$ needs at least
    $\Omega((n+\sqrt{nL/\mu})\log\frac{B^2\mu}{4\epsilon})$ IFO queries when $\frac L\mu>\frac72 n$, and
    needs at least $\Omega(n+\frac{n}{\log\frac{n\mu}{L}}\log\frac1{n\epsilon})$ IFO queries when
    $n\gg\frac L\mu$."*
  * Here $(\mu,L)$-smooth means $\mu$-strongly convex and $L$-smooth.
  * Algorithm class: "randomized incremental gradient methods" using the IFO. The paper never defines
    the class (it includes SAG/SAGA/SVRG/Katyusha and excludes SDCA).
  * Implicit span assumption: the proof uses *"$K_i$ queries to $\nabla f_i$ result in at most $K_i$
    coordinates of nonzero elements in $x^K$"*, i.e. a linear-span structure.
  * The regime "$n\gg L/\mu$" and the constant ("$\gamma$ is some constant") are informal.

**Relation.** Theorem **R3** below (with R1 for the additive $m$ term) gives the following, for
$2\le\kappa\le 8N+1$ and $\Delta/\varepsilon\ge 16$:

$$
T_\varepsilon\ \ge\ \frac1{28}\Big(m+\frac{m\,\ln(\Delta/\varepsilon)}{1+(\ln(N/\kappa))_+}\Big),
$$

where $N=nm$, $T_\varepsilon$ is the worst case over instances with $\mu$-strongly convex
components, and the bound holds for every $A\in\mathcal A$ and every $W$ with explicit constants.
It combines R1 and R3, which use two different instances (Lean: `lower_bound_transient`,
`lower_bound_small_kappa`). R9 below gives the same on a single instance, and R8 gives it down to
$\kappa=1$ on convex-component instances.

Theorem **R9** (`lower_bound_hly_form`) is a single statement in exactly H-LB-hly's form, on one
instance with $\mu$-strongly convex components, for all $\kappa\ge2$ and every $A\in\mathcal A$:

$$
T_\varepsilon\ \ge\ \Big\lceil\tfrac1{100}\Big(m+\Big(\frac{m}{1+(\ln(N/\kappa))_+}+\sqrt{\frac{m(\kappa-1)}{n}}\Big)\ln\frac\Delta\varepsilon\Big)\Big\rceil .
$$

It requires the small-$\varepsilon$ conditions $\Delta/\varepsilon\ge16$ and
$\Delta/\varepsilon\ge((\kappa-1)/(4N))^2$.

* After the standard conversion (centralized simulation, $N$ components, $n$ queries per iteration),
  this is exactly the order of H-LB-hly's $\big[\frac{m}{1+(\ln(N/\kappa))_+}+\sqrt{m\kappa/n}\big]\ln R+m$.
  For $\kappa\le N$ the $\sqrt{m\kappa/n}$ term is dominated by the first one. The same order is
  H-LB-zsh's bound for $n\gg\kappa$.
* The difference is the **algorithm class**:
  * ours: all of $\mathcal A$, proved directly in the decentralized iteration model;
  * ZSH's IFO "randomized incremental gradient methods": the proof needs each query to reveal at
    most one coordinate, i.e. a zero-respecting or linear-span structure, so their class is contained
    in $\mathcal A$;
  * HLY's oblivious p-CLI: no adaptive choice of indices or parameters.
  * The registration table classes oblivious p-CLI as narrower than $\mathcal A$. Strictly speaking,
    the p-CLI oracle $A\nabla f_j(w)+Bw+C$ also allows dense oblivious matrices, so the gain over HLY
    is "adaptive vs. oblivious". The gain over ZSH is a clean class inclusion together with the
    explicit validity range below.
  * The anchors reach our model only through a centralized simulation, which (as the registration
    table notes) is valid only for algorithms whose simulation stays in their class.
* The hard instances are of the same type: $\mu$-strongly convex, $L$-smooth quadratic components.
* **Not dominated** by any human result we found:
  * Peng–Tang–Jia (H-LB-pl) needs nonconvex components and $3\le\kappa\le\sqrt N$.
  * Woodworth–Srebro (H-LB-data) needs $\kappa>161N$ and otherwise gives only $\Omega(m)$.
  * Han–Xie–Zhang 2024 (not registered) proves the same shape only for i.i.d. index sampling with
    linear span.
  * Hendrikx–Bach–Massoulié 2021 is adaptive but its instance forces a condition number $\ge 3m$.
  * Details in §5.

---

## 1. Model and notation

We use the fixed formal model exactly (`Theory/Model.lean`, explained in `/workspace/MODEL_NOTES.md`):

* $n$ nodes, each holding $m$ components $F_i(\cdot;\xi)$, $\xi\in[m]$;
  $f=\frac1n\sum_i\frac1m\sum_\xi F_i(\cdot;\xi)$; $\phi=f+h$.
* **Class** `Instance.InClass L μ Δ`:
  * each component is $L$-smooth;
  * $f$ is convex;
  * $h$ is proper, closed and convex, with an exact prox;
  * $\phi$ is $\mu$-strongly convex and has a minimizer;
  * $\phi(x_0)-\phi^\star\le\Delta$.
* **Algorithms** `Algorithm.Admissible L μ Δ ρ`: randomized (shared seed), zero-respecting
  (queries, messages and outputs lie in the coordinates seen so far), adaptive. Each iteration has
  five micro-steps, with at most one component-gradient query and one prox query per node, and one
  communication round of $c$ mixed vectors.
* **Metric** `hittingTime`: $T_\varepsilon=\inf\{t:\mathbb E[\phi(\widehat x_t)-\phi^\star]\le\varepsilon\}$,
  with $\widehat x_t=\frac1n\sum_i\widehat x_{t,i}$ formed by the evaluator.
* `minimax` is $\inf_A\sup_{d,I,W}T_\varepsilon$.

Notation: $N:=nm$, $\kappa:=L/\mu$, $R:=\Delta/\varepsilon$, and $\lceil\cdot\rceil$ is the natural
ceiling `⌈·⌉₊` (so a negative argument gives $0$).

All our statements hold:

* for every $n\ge1$ (the problem has $n\ge2$) and every $m\ge1$;
* for every message width $c$ and every $\rho$;
* for **every** $n\times n$ matrix $W$, not only $W\in\mathcal W_{n,\rho}$, so the network plays no
  role;
* with a single hard instance that does not depend on the algorithm (we prove
  $\exists I\ \forall A\ \forall W$, which is stronger than $\forall A\ \exists I$).

---

## 2. Results

| # | Statement (all: $\mu,\Delta,\varepsilon>0$, $n,m\ge1$; for every admissible $A$, every $W$) | hard components | Lean (`ULB63.`) |
|---|---|---|---|
| R1 | $0<\mu\le L$: $\ T_\varepsilon\ge\lceil m(1-\varepsilon/\Delta)\rceil$ | $\mu$-strongly convex | `lower_bound_transient` |
| R2 | $q\in(0,1)$, $4N\mu q\le(L-\mu)(1-q)^2$: $\ T_\varepsilon\ge\big\lceil\frac m2\,\ln\frac{(1-q)^2\Delta}{\varepsilon}\big/\ln\frac1q\big\rceil$ | $\mu$-strongly convex | `lower_bound_chain` |
| R3 | $0<\kappa-1\le 8N$: $\ T_\varepsilon\ge\big\lceil\frac m2\,\ln\frac{\Delta}{4\varepsilon}\big/\ln\frac{16N}{\kappa-1}\big\rceil$ | $\mu$-strongly convex | `lower_bound_small_kappa` |
| R4 | $\kappa-1\ge 16N$: $\ T_\varepsilon\ge\big\lceil\frac m8\sqrt{\frac{\kappa-1}{N}}\,\ln\frac{4N\Delta}{(\kappa-1)\varepsilon}\big\rceil$ | $\mu$-strongly convex | `lower_bound_large_kappa` |
| R5 | $\mu\le L$, $q\in(0,1)$, $4N\mu q\le L(1-q)^2$: same bound as R2 | convex ($h=\frac\mu2\lVert\cdot\rVert^2$) | `lower_bound_chain_convex` |
| R6 | $\rho\ge0$, $q$ as in R5: $\ \mathrm{minimax}\ge\max\{\lceil m(1-\varepsilon/\Delta)\rceil,\ \lceil\frac m2\ln\frac{(1-q)^2\Delta}\varepsilon/\ln\frac1q\rceil\}$ | — | `minimax_lower_bound` |
| R7 | hypotheses of R2: for **every seed** and every $t<\frac m2\ln\frac{(1-q)^2\Delta}{\varepsilon}/\ln\frac1q$, $\ \phi(\widehat x_t)-\phi^\star>\varepsilon$ | $\mu$-strongly convex | `lower_bound_chain_pathwise` |
| R8 | $\mu\le L\le 8N\mu$: $\ T_\varepsilon\ge\big\lceil\frac m2\,\ln\frac{\Delta}{4\varepsilon}\big/\ln\frac{16N}{\kappa}\big\rceil$ | convex ($h=\frac\mu2\lVert\cdot\rVert^2$) | `lower_bound_small_kappa_convex` |
| R9 | $L\ge2\mu$, $R\ge16$, $R\ge(\frac{\kappa-1}{4N})^2$: $\ T_\varepsilon\ge\big\lceil\frac1{100}\big(m+\big(\frac{m}{1+(\ln\frac N\kappa)_+}+\sqrt{\frac{m(\kappa-1)}n}\big)\ln R\big)\big\rceil$ | $\mu$-strongly convex | `lower_bound_hly_form` |
| R10 | hypotheses of R9 and $\rho\ge0$: $\ \mathrm{minimax}\ge$ the bound of R9 | — | `minimax_lower_bound_hly_form` |
| R11 | hypotheses of R9: for **every seed** and every $t<\frac1{100}\big(m+(\frac{m}{1+(\ln\frac N\kappa)_+}+\sqrt{\frac{m(\kappa-1)}n})\ln R\big)$, $\ \phi(\widehat x_t)-\phi^\star>\varepsilon$ | $\mu$-strongly convex | `lower_bound_hly_form_pathwise` |

In R2, R3, R4, R7, R9 and R11, "hard components" means that every $F_i(\cdot;\xi)$ of the instance satisfies
`StrongConvexOn Set.univ μ`. In R5 and R8 every component satisfies `ConvexOn ℝ Set.univ`.

**Asymptotic reading.**

* *Regime $2\le\kappa\le 8N+1$, $R\ge16$.* Here $\ln\frac{16N}{\kappa-1}\le\ln\frac{32N}{\kappa}\le3.5\,(1+(\ln\frac N\kappa)_+)$
  and $\ln\frac R4\ge\frac12\ln R$, so R3 gives $T_\varepsilon\ge\frac{m\ln R}{14(1+(\ln(N/\kappa))_+)}$.
  Since $R\ge16$ gives $\varepsilon\le\Delta/2$, R1 gives $T_\varepsilon\ge m/2$ on its own instance.
  Together, for the worst case over the two instances ($\max\{a,b\}\ge\frac{a+b}2$; R9 does this on
  one instance):
  $$T_\varepsilon\ \ge\ \tfrac1{28}\Big(m+\frac{m\ln R}{1+(\ln(N/\kappa))_+}\Big).$$
* *Regime $\kappa\ge16N+1$, $R\ge((\kappa-1)/(4N))^2$.* R4 gives
  $T_\varepsilon\ge\frac1{16}\sqrt{m(\kappa-1)/n}\,\ln R$.
* *Near $\kappa=1$.* R8 (convex components) gives
  $T_\varepsilon\ge\frac m2\ln\frac{\Delta}{4\varepsilon}/\ln\frac{16N}{\kappa}$ for $1\le\kappa\le8N$, where
  R3 degenerates as $\kappa\to1$. The strongly convex chains have condition number
  $1+(\kappa-1)/N$; HLY's own proof has the same $\ln(1+6n/(\kappa-1))$.

---

## 3. The hard instance (separate chains)

Fix $n,m,K\ge1$, $N=nm$, $\mu,\Delta>0$ and $q\in[0,1)$.

* **Coordinates.** The dimension is $d=mK$, with coordinates $(\xi,j)\in[m]\times\{0,\dots,K-1\}$
  (Lean: `finProdFinEquiv`). We write $x_{\xi,j}$ and use the convention $x_{\xi,K}:=0$.
* **Weights.** $w_j=1$ for $j<K-1$, and $w_{K-1}=1-q$.
* **The chain form of block $\xi$**:
  $$B_\xi(x,v)=x_{\xi,0}v_{\xi,0}+\sum_{j=0}^{K-1}w_j\,(x_{\xi,j}-x_{\xi,j+1})(v_{\xi,j}-v_{\xi,j+1}),\qquad Q_\xi(x)=B_\xi(x,x).$$
* **Constants.**
  $$\alpha=\frac{\mu q}{(1-q)^2},\qquad c=(1-q)\sqrt{\frac{2\Delta}{m\mu}},\qquad \beta=\frac{\mu c}{(1-q)^2},\qquad \psi_\xi(x)=\frac\alpha2 Q_\xi(x)-\beta x_{\xi,0}.$$
* **Instance $\mathbf I_{\rm SC}$** (Lean `hardSC`; used in R1–R4, R7, R9 and R10). Node $1$ holds
  $F_1(x;\xi)=N\psi_\xi(x)+\frac\mu2\|x\|^2$. Nodes $i\ge2$ hold $F_i(x;\xi)=\frac\mu2\|x\|^2$.
  Further $h\equiv0$, $\operatorname{prox}_{\eta h}=\mathrm{id}$, and $x_0=0$.
* **Instance $\mathbf I_{\rm C}$** (Lean `hard`; used in R5, R6 and R8). $F_1(x;\xi)=N\psi_\xi(x)$ and
  $F_i\equiv0$ for $i\ge2$. Further $h=\frac\mu2\|\cdot\|^2$,
  $\operatorname{prox}_{\eta h}(v)=v/(1+\eta\mu)$, and $x_0=0$.
* **The objective.** In both instances the factor $N$ cancels the averaging $\frac1n\frac1m$, so
  $$\phi=f+h=\Phi,\qquad \Phi(x):=\sum_{\xi}\psi_\xi(x)+\frac\mu2\|x\|^2 .$$
  Block $\xi$ is a Nesterov chain with condition number $1+4\alpha/\mu$, so $q=(\sqrt{1+4\alpha/\mu}-1)/(\sqrt{1+4\alpha/\mu}+1)$.
* **Why the instance is hard.** Each component $\xi$ of node 1 owns its own chain. A gradient query
  to it can reveal at most one new coordinate of block $\xi$, and no other component can reveal any
  coordinate of that block.
* **R1 case.** For $q=0$ we get $\alpha=0$: the components are linear functions plus
  $\frac\mu2\|x\|^2$.

---

## 4. Proofs

**Lemma 4.1 (the chain form).** For all $x,v,y$ and $a\in\mathbb R$:

1. $0\le Q_\xi(v)\le4\|v\|^2$;
2. $B_\xi(x,v)^2\le Q_\xi(x)Q_\xi(v)$;
3. $Q_\xi(ax+(1-a)y)=aQ_\xi(x)+(1-a)Q_\xi(y)-a(1-a)Q_\xi(x-y)$.

*Proof.*

1. Use $0\le w_j\le1$ and $(s-t)^2\le2s^2+2t^2$. Since $v_{\xi,K}=0$, the shifted sum is
   $\sum_{j<K}v_{\xi,j+1}^2=\sum_{j<K}v_{\xi,j}^2-v_{\xi,0}^2$. Hence
   $Q_\xi(v)\le v_{\xi,0}^2+2\sum_{j<K}v_{\xi,j}^2+2\sum_{j<K}v_{\xi,j+1}^2\le4\sum_{j<K}v_{\xi,j}^2\le4\|v\|^2$.
2. $B_\xi$ is a symmetric bilinear form and $Q_\xi\ge0$. So $0\le Q_\xi(x-tv)=Q_\xi(x)-2tB_\xi(x,v)+t^2Q_\xi(v)$
   for every $t$. If $Q_\xi(v)>0$, take $t=B_\xi(x,v)/Q_\xi(v)$. If $Q_\xi(v)=0$, letting $t$ range
   over $\mathbb R$ forces $B_\xi(x,v)=0$.
3. Bilinearity. $\square$

(Lean: `Q_nonneg`, `Q_le`, `bil_sq_le`, `Q_convex_comb`.)

**Lemma 4.2 (gradient, smoothness).** Let $G_\xi(x)$ be the Riesz vector of $v\mapsto B_\xi(x,v)$.

* $N\psi_\xi$ has gradient $N(\alpha G_\xi(x)-\beta e_{\xi,0})$.
* $G_\xi$ is linear and $\|G_\xi(x)\|\le4\|x\|$, so $N\psi_\xi$ is $4N\alpha$-smooth.
* Hence the components of $\mathbf I_{\rm C}$ are $L$-smooth when $4N\alpha\le L$, i.e. when $4N\mu q\le L(1-q)^2$.
* The components of $\mathbf I_{\rm SC}$ are $L$-smooth when $4N\alpha+\mu\le L$, i.e. when $4N\mu q\le(L-\mu)(1-q)^2$.

*Proof.*

* *Gradient.* The remainder is exact:
  $N\psi_\xi(x+v)-N\psi_\xi(x)-\langle N(\alpha G_\xi(x)-\beta e_{\xi,0}),v\rangle=\frac{N\alpha}2Q_\xi(v)\in[0,2N\alpha\|v\|^2]$,
  which is $o(\|v\|)$.
* *Norm bound.* $\|G_\xi(x)\|^2=B_\xi(x,G_\xi(x))\le\sqrt{Q_\xi(x)Q_\xi(G_\xi(x))}\le2\|x\|\cdot2\|G_\xi(x)\|$,
  by Lemma 4.1.
* *The extra term in $\mathbf I_{\rm SC}$.* $\nabla\frac\mu2\|x\|^2=\mu x$, which is $\mu$-Lipschitz. $\square$

(Lean: `hasGradientAt_comp`, `norm_G_le`, `comp_smooth`, `hardSC_hasGradientAt₀`.)

**Lemma 4.3 (convexity and the rest of the class).**

* Each $N\psi_\xi$ is convex, by Lemma 4.1(3) and $\alpha\ge0$. So the components of $\mathbf I_{\rm C}$
  are convex, those of $\mathbf I_{\rm SC}$ are $\mu$-strongly convex, $f$ is convex, and $\Phi$ is
  $\mu$-strongly convex.
* $h$ is proper, closed and convex.
* The stated prox maps are exact: `isProx_halfSq` of `Sanity.lean` for $\mathbf I_{\rm C}$, and the
  identity for $h=0$. $\square$

**Lemma 4.4 (minimizer, initial gap).** Let $x^\star_{\xi,j}:=c\,q^j$. For every block $\xi$ and
every $v$,
$$\alpha B_\xi(x^\star,v)+\mu\sum_{j<K}x^\star_{\xi,j}v_{\xi,j}=\beta v_{\xi,0}.\tag{$\star$}$$
Consequently:

* $\Phi(x^\star+v)=\Phi(x^\star)+\sum_\xi\frac\alpha2Q_\xi(v)+\frac\mu2\|v\|^2\ \ge\ \Phi(x^\star)+\frac\mu2\|v\|^2$;
* $x^\star$ is the minimizer;
* $\Phi(x^\star)=-m\beta c/2=-\Delta$ and $\Phi(0)=0$;

so $\phi(x_0)-\phi^\star=\Delta$ exactly.

*Proof.*

* *The boundary weight.* For every $j<K$, $w_j(x^\star_{\xi,j}-x^\star_{\xi,j+1})=c(1-q)q^j$. This is
  immediate for $j<K-1$. For $j=K-1$ it holds because $w_{K-1}=1-q$ and $x^\star_{\xi,K}=0$; this is
  the purpose of the weight.
* *A shifted sum.* Let $S=\sum_{j<K}q^jv_{\xi,j}$ and $S'=\sum_{j<K}q^jv_{\xi,j+1}$. Since
  $v_{\xi,K}=0$, $qS'=S-v_{\xi,0}$.
* *Proof of $(\star)$.* We have $B_\xi(x^\star,v)=cv_{\xi,0}+c(1-q)(S-S')$ and
  $\sum_jx^\star_{\xi,j}v_{\xi,j}=cS$. Use $\alpha(1-q)^2=\mu q$ and $\beta(1-q)^2=\mu c$:
  $$(1-q)^2\big[\alpha B_\xi+\mu cS-\beta v_{\xi,0}\big]=\mu qcv_{\xi,0}+\mu qc(1-q)(S-S')+\mu c(1-q)^2S-\mu cv_{\xi,0}=-\mu c(1-q)\,(qS'-S+v_{\xi,0})=0.$$
* *The expansion.* Combine $Q_\xi(x^\star+v)=Q_\xi(x^\star)+2B_\xi(x^\star,v)+Q_\xi(v)$,
  $\|x^\star+v\|^2=\|x^\star\|^2+2\langle x^\star,v\rangle+\|v\|^2$, and $(\star)$ summed over $\xi$.
* *The value at the minimizer.* $(\star)$ with $v=x^\star$ gives
  $\frac\alpha2Q_\xi(x^\star)+\frac\mu2\sum_j(x^\star_{\xi,j})^2=\frac{\beta c}2$. Hence
  $\Phi(x^\star)=m(\frac{\beta c}2-\beta c)=-\frac{m\mu c^2}{2(1-q)^2}=-\Delta$. $\square$

(Lean: `block_identity`, `Phi_expand`, `Phi_ge`, `Phi_xstar_eq`, `hard_φstar`, `hard_inClass`,
`hardSC_inClass`.)

**Lemma 4.5 (supports of oracle answers).** Let $\mathrm{Box}(k)=\{(\xi,j):j<k_\xi\}$ for
$k\in\mathbb N^m$, and suppose $\operatorname{supp}x\subseteq\mathrm{Box}(k)$. Then:

* $\operatorname{supp}\nabla F_1(x;\xi)\subseteq\mathrm{Box}(k+e_\xi)$;
* $\operatorname{supp}\nabla F_i(x;\xi)\subseteq\mathrm{Box}(k)$ for $i\ge2$;
* $\operatorname{supp}\operatorname{prox}_{\eta h}(x)\subseteq\mathrm{Box}(k)$.

*Proof.* Coordinate $(\xi',j')$ of $G_\xi(x)$ equals $B_\xi(x,e_{\xi',j'})$.

* If $\xi'\ne\xi$, it is $0$.
* If $\xi'=\xi$ and $j'\ge k_\xi+1$, every term vanishes. The first term vanishes because $j'\ne0$.
  For the $j$-th summand: if $j<k_\xi$, then $j,j+1<j'$ and the $e$-coordinates vanish; if
  $j\ge k_\xi$, then $x_{\xi,j}=x_{\xi,j+1}=0$.
* Finally, $e_{\xi,0}\in\mathrm{Box}(k+e_\xi)$, and $\mu x$ and the prox maps (scaling, identity)
  do not enlarge supports. $\square$

(Lean: `G_supp`, `gcomp_supp`, `hard_props`, `hardSC_props`.)

**Lemma 4.6 (support propagation; this is the only place where $\mathcal A$ enters).** Let $A$ be
admissible, $W$ any matrix, and $\omega$ any seed. For every $t$ there is $k\in\mathbb N^m$ with
$\sum_\xi k_\xi\le t$ such that $\operatorname{supp}\widehat x_t\subseteq\mathrm{Box}(k)$.

*Proof.* Induction over micro-steps $s$, on this statement: there is $k$ with $\sum k\le$ (number of
gradient observations in node 1's history) and every node's allowed set contained in $\mathrm{Box}(k)$.

* *Base.* $x_0=0$.
* *Communication step.* Node $i$ receives $\sum_jW_{ij}z_j$, where
  $\operatorname{supp}z_j\subseteq$ allowed$_j\subseteq\mathrm{Box}(k)$ (`send_zero_respecting`).
  Sums and scalings stay in $\mathrm{Box}(k)$, and no gradient observation is added.
* *Oracle step.* Query points lie in the allowed sets (`call_zero_respecting`). By Lemma 4.5 every
  answer lies in $\mathrm{Box}(k)$, except a gradient query of node 1 to component $\xi$, which lies
  in $\mathrm{Box}(k+e_\xi)$. We then replace $k$ by $k+e_\xi$. So $\sum k$ grows by one exactly when
  node 1's gradient count does.
* *Counting.* By `one_grad`, each iteration (5 micro-steps) adds at most one gradient observation
  per node. So after $t$ iterations node 1 has at most $t$ of them.
* *Outputs.* `out_zero_respecting` puts every $\widehat x_{t,i}$ in $\mathrm{Box}(k)$, hence also
  their average.

$W$, the prox steps and the order of primitives within an iteration play no role. $\square$

(Lean: `Support.support_invariant`, `Support.gradCount_le`, `Support.avgOutput_supp`.)

**Lemma 4.7 (gap on boxes).** If $\operatorname{supp}x\subseteq\mathrm{Box}(k)$ and $k_\xi<K$ for
all $\xi$, then
$$\phi(x)-\phi^\star\ \ge\ \frac\mu2\sum_\xi(c\,q^{k_\xi})^2\ =\ \frac{\Delta(1-q)^2}{m}\sum_\xi q^{2k_\xi}.$$
*Proof.* By Lemma 4.4, $\phi(x)-\phi^\star\ge\frac\mu2\|x-x^\star\|^2\ge\frac\mu2\sum_\xi(x_{\xi,k_\xi}-x^\star_{\xi,k_\xi})^2$,
and $x_{\xi,k_\xi}=0$. Finally use $\mu c^2=2\Delta(1-q)^2/m$. $\square$

(Lean: `Phi_gap_box`, `gap_along_run`.)

**Proof of R2, R5 and R7.**

1. *Choice of chain length.* Let $T=\big\lceil\frac m2\ln\frac{(1-q)^2\Delta}{\varepsilon}/\ln\frac1q\big\rceil$
   and $K=T+1$.
2. *Gap along the run.* Fix $t<T$. Lemma 4.6 gives $k$ with $\sum k\le t<K$, so Lemma 4.7 applies.
   By AM–GM (Jensen for $x\mapsto q^{2x}$),
   $\frac1m\sum_\xi q^{2k_\xi}\ge q^{2\sum k/m}\ge q^{2t/m}$. Hence, for every seed,
   $$\phi(\widehat x_t)-\phi^\star\ge\Delta(1-q)^2q^{2t/m}.$$
3. *Threshold.* $t<\frac m2\ln\frac{(1-q)^2\Delta}\varepsilon/\ln\frac1q$ means
   $\frac{2t}m\ln\frac1q<\ln\frac{(1-q)^2\Delta}\varepsilon$, i.e. $\Delta(1-q)^2q^{2t/m}>\varepsilon$.
   This is R7.
4. *Expectation.* The lower Lebesgue integral is monotone and $P$ is a probability measure, so
   $\mathbb E[\phi(\widehat x_t)-\phi^\star]>\varepsilon$ for all $t<T$, i.e. $T_\varepsilon\ge T$
   (lemma `le_hittingTime_of`). This is R2 with $\mathbf I_{\rm SC}$, and R5 with $\mathbf I_{\rm C}$.
   $\square$

**Proof of R1.** Take $q=0$. Then $(q^k)^2=[k=0]\ge1-k$, so $\frac1m\sum_\xi q^{2k_\xi}\ge1-\frac tm$
and $\phi(\widehat x_t)-\phi^\star\ge\Delta(1-t/m)>\varepsilon$ for $t<m(1-\varepsilon/\Delta)$.
(Lean: `expectedGap_ge_count`.) $\square$

**Proof of R3.** Take $q=\frac{\kappa-1}{16N}\in(0,\frac12]$.

* Feasibility: $4N\mu q=\frac{L-\mu}4\le(L-\mu)(1-q)^2$.
* $(1-q)^2\ge\frac14$, so $\ln\frac{(1-q)^2\Delta}{\varepsilon}\ge\ln\frac\Delta{4\varepsilon}$.
* $\ln\frac1q=\ln\frac{16N}{\kappa-1}$. $\square$

**Proof of R8.** As for R3, with $\mathbf I_{\rm C}$, the constraint of R5, and $q=\frac{\kappa}{16N}\le\frac12$:
$4N\mu q=\frac L4\le L(1-q)^2$ and $\ln\frac1q=\ln\frac{16N}{\kappa}$. $\square$

**Proof of R4.** Take $s=\sqrt{N/(\kappa-1)}\le\frac14$ and $q=1-2s\in[\frac12,1)$.

* Feasibility: $(L-\mu)(1-q)^2=4N\mu\ge4N\mu q$.
* $(1-q)^2=\frac{4N}{\kappa-1}$.
* $\ln\frac1q\le\frac1q-1\le4s$.
* If $A:=\ln\frac{4N\Delta}{(\kappa-1)\varepsilon}\le0$, the bound is $0$. Otherwise
  $\frac m2A/\ln\frac1q\ge\frac m2A/(4s)=\frac m8\sqrt{\frac{\kappa-1}N}A$. $\square$

**Proof of R6 and R10.** Both instances are in the standard class (for R10: the instance of R9), and $J\in\mathcal W_{n,\rho}$ for $\rho\ge0$
(`J_commClass` in `Sanity.lean`). So for every admissible $A$ the supremum in `minimax` is at least
the value of $T_\varepsilon$ at $(\mathbf I,J)$. $\square$


**Proof of R9 and R11.** Write $X=\frac{m}{1+\ell}+\sqrt{m(\kappa-1)/n}$ with $\ell=(\ln(N/\kappa))_+$, put
$\ln R=\ln(\Delta/\varepsilon)$ and $T=\lceil(m+X\ln R)/100\rceil$.

*Two bounds on one instance.* For the instance $\mathbf I_{\rm SC}$ with chain length $T+1$ and any
admissible $q$, Lemma 4.7 applied along a run gives two bounds at every $t<T$ for every seed (Lean:
`expectedGap_gt_of`, `count_bound`):

* the chain bound $\Delta(1-q)^2q^{2t/m}$ (AM–GM);
* the counting bound $\Delta(1-q)^2(1-t/m)$, from $q^{2k}\ge1-k$.

So it suffices to show, for each $t<T$, that one of the two exceeds $\varepsilon$ (`finish_SC`;
`finish_SC_pathwise` gives the per-seed statement R11). The choice of $q$ is the lemma
`hly_choice`.

*Splitting $t<T$.* Either $t<m/2$, or $98t<X\ln R$ (because $m\le2t$).

*The three regimes.*

1. $\kappa-1\le8N$, with $q=\frac{\kappa-1}{16N}$ (`regime1`).
   * If $t<m/2$: $(1-q)^2\ge\frac14$ gives $\Delta(1-q)^2(1-\frac tm)>\frac\Delta8\ge2\varepsilon$.
   * Otherwise use:
     * $\ln\frac1q\le\ln\frac{32N}\kappa\le3.5(1+\ell)$, from $\kappa-1\ge\kappa/2$ and $5\ln2<3.5$;
     * $\ln\frac{(1-q)^2\Delta}\varepsilon\ge\frac12\ln R$;
     * $X\le\frac{4m}{1+\ell}$. For $\kappa\le N$ this uses $1+\ln y\le2\sqrt y$ with $y=N/\kappa$;
       for $\kappa>N$ it uses $\sqrt{(\kappa-1)/N}\le3$.
   * Then $\frac m2\ln\frac{(1-q)^2\Delta}\varepsilon/\ln\frac1q\ge\frac{m\ln R}{14(1+\ell)}\ge\frac{X\ln R}{56}>t$.
2. $8N<\kappa-1<16N$, with $q=\frac13$ (`regime2`).
   * Feasibility holds because $\kappa-1\ge3N$.
   * Here $\ell=0$, $X\le5m$, $\ln3\le2$ and $\ln\frac{4R}9\ge\frac12\ln R$. So
     $\frac m2\ln\frac{4R}9/\ln3\ge\frac{m\ln R}8\ge\frac{5m\ln R}{98}>t$.
   * The case $t<m/2$ is as in regime 1.
3. $\kappa-1\ge16N$, with $s=\sqrt{N/(\kappa-1)}\le\frac14$ and $q=1-2s$ (`regime3`).
   * Here $\ell=0$, $X=m+m/s\le\frac{5m}{4s}$ and $m\le X\ln R$, so for $t<T$,
     $t<\frac{m+X\ln R}{100}\le\frac{X\ln R}{50}\le\frac{m\ln R}{40s}$.
   * On the other side, $\ln\frac1q\le4s$, and $\ln(4s^2R)\ge\frac12\ln R$ since
     $16s^4R\ge1\iff R\ge(\frac{\kappa-1}{4N})^2$.
   * So $\frac m2\ln\frac{(1-q)^2\Delta}\varepsilon/\ln\frac1q\ge\frac{m\ln R}{16s}>t$. $\square$

---

## 5. Comparison with human results

### 5.1 The anchors and other relevant results (assumptions as stated in the sources)

* **H-LB-hly** and **H-LB-zsh**: stated verbatim in §0.

* **H-LB-data**: Woodworth–Srebro, NeurIPS 2016, arXiv:1605.08003v3, Theorem 8.
  * Verbatim: *"For any $m\ge2$, any $\gamma,\lambda>0$ such that $\frac\gamma\lambda>161m$, any
    $\epsilon>0$, any $\epsilon_0>60\epsilon\sqrt{\frac{\gamma}{\lambda m}}$, and any randomized
    algorithm $A$, there exists a dimension … and $m$ functions … $\gamma$-smooth and
    $\lambda$-strongly convex … $F(x_0)-F(x^*)=\epsilon_0$ … must make
    $\Omega(m+\sqrt{\frac{m\gamma}\lambda}\log(\frac{\epsilon_0}\epsilon\sqrt{\frac{m\lambda}\gamma}))$
    queries to $h_F$."*
  * Algorithm class: arbitrary randomized algorithms with value, gradient and prox access to
    components; no span assumption.
  * For $\kappa=O(m)$ the paper gives only $\Omega(m)$.
* **H-LB-pl**: Peng–Tang–Jia, arXiv:2609.00045v3, Theorem `thm:pl-main` (§3).
  * Bound: for $n\ge n_0$, $\kappa_s\ge3$, $0<\varepsilon\le c_\varepsilon\Delta$, the complexity is
    $\Theta(n+\frac{n\log(\Delta/\varepsilon)}{1+\log(\sqrt n/\kappa_s)})$ for $3\le\kappa_s\le\sqrt n$.
  * Algorithm class: randomized IFO algorithms with unrestricted query and index selection.
  * Guarantee: fixed-budget, constant-probability complexity.
  * Hard instance: in the small-$\kappa$ branch it is average strongly convex, and "its components
    need not be convex".
* **H-LB-net**: Scaman et al., arXiv:1702.08704v2, Theorems 1 and 2 (black-box procedures; graph
  diameter, respectively gossip eigengap).
* **H-LB-avg**: Zhou–Gu, arXiv:1901.11224v1, Theorem 4.1. Linear-span randomized algorithms,
  average smoothness, $\Omega(n+n^{3/4}\sqrt{L/\sigma}\log\frac{n\sigma\Delta}{L\epsilon})$.
* **H-UB**: Liu–Low, arXiv:2402.02356v1, Theorem 5.1 and Corollary 5.3 (PMGT-KatyushaX; composite
  $F=f+\psi$, $(\ell_1,\ell_2)$-smooth components, $0\preceq W\preceq I$). It gives
  $O((n+\frac{L^{1/2}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}n^{3/4})\log\frac{F(\bar y^0)-F^*}{\epsilon})$
  SFO calls and $\tilde O\big(\frac{\sqrt n+\cdots n^{1/4}}{\sqrt{1-\lambda_2(W)}}\log\big)$
  communication rounds per agent. Here $n$ is the number of local components, our $m$.
* **Not registered, found by our search.**
  * **Han–Xie–Zhang**, *Lower complexity bounds of finite-sum optimization problems: the results and
    construction*, JMLR 2024, arXiv:2103.08280v5. The theorem labelled `thm:strongly` in the appendix
    "Main Results" gives $\Omega\big(n+\frac{n}{1+(\log(n/\kappa))_+}\log\frac1\epsilon\big)$ for
    $\kappa=O(n)$, $\kappa\ge2$, with $L$-smooth, $\mu$-strongly convex components. Its class is
    PIFO algorithms with indices drawn i.i.d. from a fixed distribution, Bernoulli full-gradient
    calls, and linear span. The authors list "methods that break the linear-span protocol" as future
    work.
  * **Hendrikx–Bach–Massoulié**, arXiv:2005.10675 (SIAM J. Optim. 2021). This is a decentralized
    finite-sum lower bound whose pathwise argument allows any index choice, but the instance forces
    the condition number $\kappa_s=3mL/\sigma\ge3m$, so $\kappa\le N$ is not covered.
  * **Lan–Zhou**, arXiv:1507.02000v3, Theorem 3: i.i.d. index sampling.
  * **Arjevani–Shamir**, arXiv:1606.09333, Theorem 2: oblivious algorithms.

**Converting units.** A centralized bound of $Q$ IFO queries on $N$ components becomes $T\ge Q/n$ in
our model, but only for algorithms whose centralized simulation stays in the cited class. Our results
need no simulation: they are proved in the decentralized model with the per-node budget. For $n=1$
our statements *are* centralized statements, since one component gradient per iteration means
$T={}$number of IFO queries: R3 reads $T\ge\frac m2\ln\frac\Delta{4\varepsilon}/\ln\frac{16m}{\kappa-1}$.

**Accuracy normalization.** We measure accuracy by $R=\Delta/\varepsilon$.

* ZSH normalize by $\|x^0-x^*\|=B$. Their proof gives
  $K\ge n\log\frac{B^2\mu}{4\epsilon}/\log\frac1{q_1}$, and for their (block-chain) instance
  $B^2\mu/4\asymp F(x^0)-F^*$. So in $\Delta$-normalization their bound has the same order as ours.
  The $\log\frac1{n\epsilon}$ in their statement comes from their rewriting of $\log\frac{B^2\mu}{4\epsilon}$,
  and is absorbed by the $+n$ term. We therefore claim **no** improvement of the bound over ZSH, only
  of the algorithm class, explicit regime, constants and rigour.
* HLY's $\epsilon$ is absolute, with an instance-dependent constant (their proof uses
  $\ln\frac{(\mu/4)(nR/(\kappa-1))^2}{\epsilon}$).

### 5.2 Classification of each result

* **R3** (with R1), and R2 which contains it: **strictly stronger than H-LB-zsh**, and than H-LB-hly
  in the sense of the registration table (adaptive vs. oblivious; see §0 for the p-CLI caveat).
  * *Bound.* The same order as theirs on their validity ranges (§0). Our constants are explicit,
    validity is explicit ($0<\kappa-1\le8N$, $\varepsilon,\Delta>0$), and there are no informal
    "$n\gg\kappa$" or $\tilde\Omega$ qualifiers.
  * *Hard instances.* The same class: $\mu$-strongly convex, $L$-smooth quadratics. Our hard instance
    is block-separable, like ZSH's.
  * *Algorithm class.*
    * $\mathcal A$ contains every linear-span IFO method with adaptive, i.i.d. or without-replacement
      indices, and hence ZSH's class (whose proof needs exactly this structure).
    * It also contains every zero-respecting method that uses nonlinear local computation, the prox
      oracle of $h$, and decentralized communication with any $W$.
    * Against HLY's oblivious p-CLI the gain is adaptivity. The registration table records oblivious
      p-CLI as narrower than $\mathcal A$; literally, p-CLI also admits dense oblivious matrices
      (§0).
  * *Guarantee type.* R7 gives a pathwise version: failure on every seed, not only in expectation.
    It therefore also covers success-probability criteria.
  * *Not dominated.* Peng–Tang–Jia's stronger denominator $1+\log(\sqrt N/\kappa)$ is proved only on
    instances with nonconvex components and $3\le\kappa\le\sqrt N$, so it says nothing about the
    strongly-convex-component class. Woodworth–Srebro give only $\Omega(m)$ unless $\kappa>161N$.
    Han–Xie–Zhang restrict the index distribution. Hendrikx–Bach–Massoulié need $\kappa\ge3m$.
  * *Prior partial arguments.* The pathwise counting argument (one new coordinate per query, then
    Jensen) is classical in spirit. It appears inside ZSH's proof (informally, for an undefined
    class) and in Hendrikx–Bach–Massoulié (only for $\kappa\ge3m$). What we add is the rigorous
    theorem for the full class $\mathcal A$ in the regime $\kappa\le N$, in the decentralized model,
    machine-checked.
  * *Answers a stated question.* Han–Xie–Zhang (arXiv:2103.08280v5, introduction of the minimization
    part) write: *"the distribution $\mathcal D$ over $[n]$ … need not be stationary over time.
    Sampling without replacement and methods that break the linear-span protocol are also worth
    considering."* $\mathcal A$ contains all of these (any adaptive index rule, any zero-respecting
    nonlinear computation). R3 and R9 settle the strongly convex, $\kappa=O(N)$ case of that question
    for zero-respecting methods.
  * *Value.* This is the regime singled out by HLY ("big data regime", where "the span assumption
    prevents algorithms from converging … in less than $n\ln(1/\epsilon)$ iterations"). For convex
    components, HLY's modified SVRG/SARAH upper bound
    $O(n+\frac{n}{1+(\ln(n/\kappa))_+}\ln\frac1\epsilon)$ (their Corollary 2, centralized) shows that
    the shape is the right one. Our result establishes that no zero-respecting adaptive method can do
    better, which HLY and HXZ prove only for oblivious or i.i.d. index selection.
* **Validity range compared with ZSH.** Suppose ZSH's class is read generously as "all zero-respecting
  IFO methods". Even then their theorem covers only "$n\gg L/\mu$" (informal) and
  $L/\mu>\frac72n$. R2 covers every $\kappa>1$ with explicit constants: R3 for
  $\kappa-1\le8N$, and in particular the intermediate range $\kappa\asymp N$ that ZSH's two cases
  leave out. The protocol counts validity thresholds, so R3 is not dominated by ZSH even under that
  reading.
* **R5** and its explicit form **R8**: **strictly stronger than H-LB-hly**, in the same sense. The registration table lists HLY's
  hard instances as "凸分量" (convex components), and R5 uses convex components. R5 also covers
  $1\le\kappa<2$, where the strongly-convex-component chains degenerate.
* **R9**: **strictly stronger than H-LB-hly**. It is HLY's exact form
  $m+[\frac m{1+(\ln(N/\kappa))_+}+\sqrt{m\kappa/n}]\ln R$ for all $\kappa\ge2$, with constant
  $\frac1{100}$, on HLY's hard class ($\mu$-strongly convex quadratics), for all of $\mathcal A$
  instead of oblivious p-CLI.
  * Validity: small $\varepsilon$, $R\ge\max\{16,((\kappa-1)/(4N))^2\}$. HLY's $\tilde\Omega$
    likewise absorbs low-accuracy terms.
  * Not dominated: for $\kappa\le161N$ no known result covers adaptive algorithms on
    strongly-convex-component instances. For $\kappa>161N$ it has the order of Woodworth–Srebro
    Theorem 8, which covers all randomized algorithms; there R9 adds nothing.
* **R7** and **R11**: **strictly stronger**. They are the pathwise forms of R2 and R9, so the
  guarantee type is stronger.
* **R1**: **equal**. The $\Omega(m)$ term is known (Woodworth–Srebro's $m$ term; the $n$ term of
  Peng–Tang–Jia and of HXZ).
* **R4**: **incomparable**.
  * For $\kappa>161N$ it is dominated by Woodworth–Srebro Theorem 8. Theirs has a larger algorithm
    class, the same $\sqrt{m\kappa/n}$, and a better logarithm $\ln(R\sqrt{N/\kappa})$ against our
    $\ln(4NR/\kappa)$.
  * On $16N+1\le\kappa\le161N$, where WS does not apply, it extends the HLY/ZSH large-$\kappa$ shape
    to $\mathcal A$.
* **R6** and **R10**: **incomparable**. They bound the minimax quantity of the standard class; R10
  does so in HLY form.
  * For $3\le\kappa\le\sqrt N$ and $n\ge n_0$ it is weaker than Peng–Tang–Jia, by up to a
    $\log N$ factor near $\kappa\approx\sqrt N$.
  * It also covers $1\le\kappa<3$, small $N$, and $\sqrt N<\kappa\le161N$. There neither PL nor WS
    applies directly. Combining PL at $\kappa=\sqrt N$ with the monotonicity of the class in $L$
    would give $m\ln R$, which is stronger.

---

## 6. What is not achieved (honest scope)

* **No upper bound.** H-UB is neither improved nor matched. The upper-bound and resource rows are
  untouched.
* **Network term.** Not addressed; our bounds hold for every $W$ and are pure data terms. H-LB-net
  remains the network lower bound. Liu–Low's open question about the $\sqrt n$ factor in their
  communication bound (our $\sqrt m$) remains open.
* **Goal A** ($\kappa\ge\sqrt N$, individually smooth, beating $\sqrt{m\kappa/n}$) remains open.
  Our chains give $\sqrt{m\kappa/n}\log$ for $\kappa\ge16N$ (R4), which does not beat H-LB-data.
* **Remark (conjecture level, not proved here).** For $\kappa=O(1)$ one expects an
  $\tilde O\big((m+(1-\rho)^{-1/2})\log R\big)$ method: a proximal gradient method with exact local
  full gradients ($m$ iterations per step) and Chebyshev-accelerated gossip for the averages
  ($\tilde O((1-\rho)^{-1/2})$ rounds per step). In that regime this would remove the $\sqrt m$
  factor that Liu–Low's conclusion calls "unsatisfying". We have neither written nor formalized this
  analysis, and claim nothing about it.
* **Conjecture (not proved).** For $\sqrt N\le\kappa\le N$, the data-term complexity of the standard
  class is $\tilde\Theta(m\log R)$ up to the $N^{1/4}$ gap of the upper bound. The lower bound
  $m\ln R$ there follows from H-LB-pl and monotonicity; we have not formalized it.

## 7. Row-by-row self-assessment

| row | assessment |
|---|---|
| assumptions and tuning | none (no algorithm) |
| upper: data | none |
| upper: network | none |
| lower: data | "=" overall. In $\sqrt N<\kappa\le161N$ no registered data anchor applies directly, and R3/R9 give $m\ln R/(1+(\ln(N/\kappa))_+)$. In $3\le\kappa\le\sqrt N$ it is weaker than H-LB-pl, but holds on strongly-convex-component instances. For $\kappa>161N$, R4 matches H-LB-data up to the logarithm's argument. |
| lower: network | none |
| resources | none |

## 8. Formalization map

Files in `theory/formal/Theory/`:

* `Chain.lean`: block coordinates, the form $B_\xi$, $Q_\xi$ (Lemma 4.1), the gradient $G_\xi$ and
  smoothness (Lemma 4.2), boxes and gradient supports (Lemma 4.5), the minimizer and the expansion
  (Lemma 4.4), and the instance $\mathbf I_{\rm C}$ (`inst`).
* `Support.lean`: the support invariant along `Algorithm.hist` and the one-gradient-per-iteration
  count (Lemma 4.6), for any instance whose oracles respect a monotone box family.
* `LowerBound.lean`: class membership of $\mathbf I_{\rm C}$ (`hard_inClass`), the gap bound along
  runs (`gap_along_run`, Lemma 4.7), AM–GM (`amgm`), the expected-gap bounds, and R5.
* `StronglyConvex.lean`: the instance $\mathbf I_{\rm SC}$ (`hardSC`) and its class membership with
  $\mu$-strongly convex components.
* `Main.lean`: R1–R4, R7, R8, and the bridge to `minimax`.
* `Unified.lean`: R9 and R11 (counting and chain bounds on the same instance; the three regime
  lemmas; `hly_choice`).
* `Statements.lean` / `Proofs.lean`: the eleven statements R1–R11, verbatim, with sorry and with
  proofs.

How the Lean statements say what this document says:

* *The instance.* `∃ d (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧ …`
  is a member of $\mathcal F_{L,\mu,\Delta,m}$ (the oracle class is built into `Instance`), in a
  dimension we choose, with the stated property of every component.
* *Algorithms and networks.* `∀ A, A.Admissible L μ Δ ρ → ∀ W, bound ≤ A.hittingTime I W ε` ranges
  over every admissible algorithm and every mixing matrix.
* *The bound.* It is a natural-number ceiling, cast to `ℕ∞`.
* *R6* and *R10* bound `minimax` itself.
* *R7* and *R11* state `ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t)` for every seed `ω`.
