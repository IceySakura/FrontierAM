# The mixed-smoothness condition of preintegration cannot be weakened

**Selected target: Target E** (registered anchor **A-GKLS**: Griewank, Kuo, Leövey, Sloan 2018, Theorem 3).

**Main result (R1, Lean name `GKLS.gkls_mixed_condition_sharp`; all properties (H0)–(H6), (C), (E) of
§3 are kernel-checked, including that the classical derivatives are the weak derivatives and GKLS's
integrability condition (h2) for all parameters).**
For every dimension $d\ge 2$, every preintegration coordinate $j$ and **every** smoothness
multi-index $\boldsymbol r\in\mathbb N_0^d$ with
$$1\le r_j<\sum_{k\ne j} r_k ,$$
there is an explicit pair $(\theta,\phi)$ — namely $\theta\equiv 1$ and a function $\phi$ that is
linear in $\boldsymbol x_{-j}$ — which satisfies **all** hypotheses of GKLS Theorem 3 except the
condition $r_j\ge\sum_{k\ne j}r_k$ (for the standard normal density $\rho$, every $p\in[1,\infty)$
and every admissible family of weights $\zeta$), but for which the preintegrated function
$P_jf$, $f=\theta\cdot\mathrm{ind}(\phi)$, is **not** in $\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}(\mathbb R^{d-1})$.
More precisely, $P_jf$ has all mixed partial derivatives of total order $\le r_j$ (in every
direction and order, all continuous), and *no* iterated partial derivative of total order
$> r_j$ exists everywhere.

Consequently the condition $r_j\ge\sum_{k\ne j}r_k$ of GKLS Theorem 3 is not only sufficient but
also **necessary**: it cannot be weakened (for any multi-index with $r_j\ge1$ violating it, the
conclusion of the theorem, as stated, fails for some admissible data; what fails is the classical
part $\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ of the conclusion, see Remark 4.1 for the Sobolev part). Together with GKLS Theorem 3 this gives an exact
description of the mixed smoothness that preintegration guarantees (Remark 4.2): total order $r_j$.

**Complementary result (R2, formalized, Lean name `GKLS.gkls_mixed_condition_kink`).** For kinks
($\theta=\phi$, $f=\max(\phi,0)$, also covered by GKLS Theorem 3) the same $\phi$ gives
$P_jf$ of exact total mixed order $r_j+1$; hence for kinks the condition can be relaxed by at most
one: the conclusion fails whenever $r_j\le\sum_{k\ne j}r_k-2$ (the borderline $r_j=\sum_{k\ne j}r_k-1$
stays open, so R2 is only a partial optimality statement).

**Minor side result (R3, formalized, Lean name `GKLS.quadric_preintegration_sqrt`; Target A).** For
jumps across hypersurfaces that are quadratic — hence not monotone — in the preintegration variable
(every quadric surface $\boldsymbol x^\top A\boldsymbol x+\boldsymbol b^\top\boldsymbol x+c=0$ with $A_{jj}\ne0$:
ellipsoids, saddles), an explicit global Hölder-$\frac12$ **upper bound**
$|P_jf(\boldsymbol y)-P_jf(\boldsymbol y')|\le(|\beta(\boldsymbol y)-\beta(\boldsymbol y')|+\sqrt{|D(\boldsymbol y)-D(\boldsymbol y')|})/(|a|\sqrt{2\pi})$
and an explicit $\sqrt D$ lower bound. This is an elementary, explicit-constant generalization of the
closed forms computed by Gilbert–Kuo–Sloan (anchor A-GKS, Examples 1–3); it gives no QMC rate (§7).

**Idea.** The obstruction is one-dimensional. Take $\phi(\boldsymbol x)=g(x_j)+\sum_{i\ne j}x_i$ with an
increasing bijection $g\in\mathcal C^{r_j}\setminus\mathcal C^{r_j+1}$ ($g'\ge\frac12$). Then $\{\phi>0\}$ meets each
$x_j$-line in the half-line $x_j>g^{-1}(-s)$, $s=\sum_{i\ne j}x_i$, so
$P_jf(\boldsymbol y)=\Psi(g^{-1}(-\sum_iy_i))$ with $\Psi$ the Gaussian tail: a *ridge function* of
$\sum_iy_i$. Every mixed derivative $D^{\boldsymbol\alpha}$ of a ridge function $H(\sum_iy_i)$ equals
$H^{(|\boldsymbol\alpha|)}(\sum_iy_i)$, however the $|\boldsymbol\alpha|$ differentiations are distributed over the
coordinates; and $H=\Psi\circ g^{-1}\circ(-\cdot)$ is exactly $\mathcal C^{r_j}$. Since
$\sum_{k\ne j}r_k\ge r_j+1$, some $D^{\boldsymbol\alpha}$ with $\boldsymbol\alpha\le\boldsymbol r_{-j}$ has order $r_j+1$ and fails.
This is precisely the mechanism GKLS point to ("$D^{\boldsymbol\alpha}P_jf$ includes terms that depend on
$D_j^{|\boldsymbol\alpha|}\phi$"), turned into a proof that the terms cannot be avoided.

*中文摘要*：选择目标 E。我们证明 GKLS 2018 定理 3 中的条件 $r_j\ge\sum_{k\ne j}r_k$ 不能再弱：
对**每一个**满足 $1\le r_j<\sum_{k\ne j}r_k$ 的多重指标，构造了满足定理 3 其余全部假设（高斯
$\rho$、任意 $p$、任意可容许权 $\zeta$、$\theta\equiv1$、$D_j\phi\ge 1/2$、$\phi\to\infty$、
$U_j=\mathbb R^{d-1}$）的显式 $\phi$，使预积分后的函数不属于 $\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$；
并且精确地确定其混合光滑性恰为总阶 $r_j$。反例的全部性质（H0–H6、C、E；包括经典导数即弱导数、以及对全部参数
逐字形式化的 GKLS 可积条件 (h2)）已在 Lean 4 + Mathlib 中形式化并通过内核检查。失效的是结论中的经典部分 $\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$（Sobolev 部分见注 4.1）。
按题面计分规则，这属于“严格证明某个登记结果的假设不能再弱”（目标 E 的反例）。

---

## 1. The anchor (A-GKLS), quoted from the source

A. Griewank, F. Y. Kuo, H. Leövey, I. H. Sloan, *High dimensional integration of kinks and jumps —
smoothing by preintegration*, J. Comput. Appl. Math. 344 (2018) 259–274,
doi:10.1016/j.cam.2018.04.009, arXiv:1712.00920 (v1, the only version; the source file is
`GKLS-20171023-rev.tex`). Theorem numbers below are those of the arXiv version (Lemma 1,
Theorem 2 = isotropic result `thm:main1`, Theorem 3 = mixed result `thm:main2`; one shared counter),
which are the numbers used in the task statement.

**Setting (GKLS §2 and §4.1).** $I_df=\int_{\mathbb R^d}f(\boldsymbol x)\rho_d(\boldsymbol x)\,d\boldsymbol x$ with
$\rho_d(\boldsymbol x)=\prod_k\rho(x_k)$, $\rho$ a continuous, strictly positive probability density;
$f(\boldsymbol x)=\theta(\boldsymbol x)\,\mathrm{ind}(\phi(\boldsymbol x))$ where "$\mathrm{ind}(\cdot)$ is
the indicator function which gives the value 1 if the input is positive and 0 otherwise";
the preintegration operator is (GKLS eq. (firstPj))
$$(P_jf)(\boldsymbol x_{-j}) := \int_{-\infty}^{\infty} f(x_j,\boldsymbol x_{-j})\,\rho(x_j)\,dx_j .$$
$D^{\boldsymbol\alpha}=\prod_j D_j^{\alpha_j}$; "$\mathcal C^{\boldsymbol r}_{\rm mix}(\mathbb R^d)$ [is] the
space of functions $g$ whose classical derivatives $D^{\boldsymbol\alpha}g$ with
$\boldsymbol\alpha\le\boldsymbol r$ are all continuous at every point in $\mathbb R^d$"; with weights
$\zeta_{d,\boldsymbol\alpha}(\boldsymbol x)=\prod_k\zeta_{\alpha_k}(x_k)$, where $\{\zeta_i\}$ are continuous
integrable functions with $\rho=\zeta_0\le\zeta_1\le\cdots$,
$\mathcal W^{\boldsymbol r}_{d,p,\boldsymbol\zeta,\rm mix}=\{g: D^{\boldsymbol\alpha}g\in\mathcal L_{p,\zeta_{d,\boldsymbol\alpha}}\ \forall\boldsymbol\alpha\le\boldsymbol r\}$ (weak derivatives).

**Theorem 2 (isotropic; the setting that Theorem 3 modifies).** $d\ge2$, $r\ge1$, $p\in[1,\infty)$,
$\rho\in\mathcal C^{r-1}$ strictly positive density, $\theta,\phi\in\mathcal W^r_{d,p,\boldsymbol\zeta}\cap\mathcal C^r(\mathbb R^d)$,
$$(D_j\phi)(\boldsymbol x)>0\ \ \forall\boldsymbol x,\qquad \phi(\boldsymbol x)\to\infty \text{ as } x_j\to\infty\ \ \forall\boldsymbol x_{-j};$$
$U_j=\{\boldsymbol y:\phi(x_j,\boldsymbol y)=0\text{ for some }x_j\}$, $U_j^+=\mathbb R^{d-1}\setminus U_j$;
every function
$$h(\boldsymbol y)=\frac{(D^{\boldsymbol\eta}\theta)(\psi(\boldsymbol y),\boldsymbol y)\prod_{i=1}^a[(D^{\boldsymbol\gamma^{(i)}}\phi)(\psi(\boldsymbol y),\boldsymbol y)]}{[(D_j\phi)(\psi(\boldsymbol y),\boldsymbol y)]^b}\,\rho^{(c)}(\psi(\boldsymbol y))$$
(with constraints on $a,b,c,\boldsymbol\gamma^{(i)},\boldsymbol\eta$) must satisfy (h1) $h(\boldsymbol y)\to0$ as
$\boldsymbol y$ approaches a boundary point of $U_j$ lying in $U_j^+$, and (h2)
$\int_{U_j}|h|^p\zeta_{d-1,\boldsymbol\alpha_{-j}}<\infty$. Conclusion: $P_jf\in\mathcal W^r_{d-1,p,\boldsymbol\zeta}\cap\mathcal C^r(\mathbb R^{d-1})$.

**Theorem 3 (verbatim).** "Let $d\ge2$, $p\in[1,\infty)$, $j\in\mathcal D$, and let
$\rho\in\mathcal C^{r-1}(\mathbb R)$ be a strictly positive probability density function. Let
$\boldsymbol r=(r_1,\dots,r_d)$ be a multi-index satisfying
$$r_j\ \ge\ \textstyle\sum_{1\le k\le d,\,k\ne j} r_k\ \ge\ 1.$$
If we replace the conditions on $\theta,\phi$ in Theorem 2 by
$\theta,\phi\in\mathcal W^{\boldsymbol r}_{d,p,\boldsymbol\zeta,\rm mix}\cap\mathcal C^{\boldsymbol r}_{\rm mix}(\mathbb R^d)$, and further
restrict (h)–(h2) to functions $h$ with multi-indices $\boldsymbol\gamma^{(i)}\le\boldsymbol r$,
$\boldsymbol\eta<\boldsymbol r$, and $\boldsymbol\alpha\le\boldsymbol r$, then the conclusion becomes:
$P_jf\in\mathcal W^{\boldsymbol r_{-j}}_{d-1,p,\boldsymbol\zeta}\cap\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}(\mathbb R^{d-1})$."
(Two slips of the source: the scalar $r$ in $\mathcal C^{r-1}$ is not defined in the mixed setting,
and the $\mathcal W$ of the conclusion lacks the subscript "mix". Neither matters below: our
$\rho$ is $\mathcal C^\infty$, and our counterexample violates the $\mathcal C_{\rm mix}$ part of the conclusion.)

GKLS justify the condition only through their proof: "the requirement that $r_j$ is greater than or
equal to the sum of the remaining $r_k$ for $k\ne j$ is needed because, for any multi-index
$\boldsymbol\alpha\le\boldsymbol r$ with $\alpha_j=0$, … the expression for $D^{\boldsymbol\alpha}P_jf$ includes some
terms that depend on $D_j^{|\boldsymbol\alpha|}\phi$ and some terms that depend on $D_j^{|\boldsymbol\alpha|-1}\theta$."
Whether the condition is *necessary* was left open (task statement, Target E: "没有证明它是必要的").

**A later refinement with the same condition.** A. D. Gilbert, F. Y. Kuo, I. H. Sloan, *Analysis of
preintegration followed by quasi-Monte Carlo integration for distribution functions and densities*,
SIAM J. Numer. Anal. 61 (2023) 135–166, doi:10.1137/21M146658X, arXiv:2112.10308. Assumption 1
requires $D^0\phi>0$, $\phi(y_0,\boldsymbol y)\to\infty$ as $y_0\to\infty$, and
"$\phi\in\mathcal H^{(\nu_0,\boldsymbol\nu)}_{d+1}\cap C^{(\nu_0,\boldsymbol\nu)}(\mathbb R^{d+1})$, where
$\nu_0:=|\boldsymbol\nu|+1$"; their Theorem 3 (arXiv numbering; "a refined version of [GKLS18,
Theorem 3]") gives $P_0\,\mathrm{ind}(t-\phi)\in\mathcal H^{\boldsymbol\nu}_d\cap C^{\boldsymbol\nu}(\mathbb R^d)$, and the
authors remark: "it would suffice to have $\nu_0:=|\boldsymbol\nu|$ … for Theorem 3 to hold". This is
exactly the GKLS condition $r_j\ge\sum_{k\ne j}r_k$ for indicator integrands ($\theta\equiv1$).
(We read arXiv:2112.10308v5.) Note, however, that their Assumption 2 (the analogue of GKLS's (h))
involves the functions $h_{q,\boldsymbol\eta}$ with factors $D^{\boldsymbol\alpha_\ell}\phi$, $\alpha_{\ell,0}\le|\boldsymbol\eta|+q$, i.e.
derivatives of order up to $|\boldsymbol\nu|$ in $y_0$; so in their formulation the order condition is
built into the hypotheses, and R1's $\phi$ (with only $|\boldsymbol\nu|-1$ derivatives in $y_0$) does not
satisfy them as written. R1 is therefore a statement about the GKLS formulation, whose (h) involves
only $D^{\boldsymbol\gamma}\phi$ with $\boldsymbol\gamma\le\boldsymbol r$.

---

## 2. Definitions used (identical to the Lean definitions in `formal/Theory/Defs.lean`)

Points of $\mathbb R^{d}$, $d=n+1$, are $\boldsymbol x=(x_1,\dots,x_d)$; for the preintegration
coordinate $j$ we write $\boldsymbol x=(x_j,\boldsymbol y)$ with $\boldsymbol y=\boldsymbol x_{-j}\in\mathbb R^{n}$
(Lean: `Fin.insertNth j t y`; the $k$-th entry of $\boldsymbol y$ is $x_{j.\mathrm{succAbove}(k)}$, i.e.
the coordinates other than $j$ in their natural order).

* **Classical partial derivative** (`partialDeriv`): $(D_kF)(\boldsymbol x)=\frac{d}{dt}F(\boldsymbol x\text{ with }x_k:=t)\big|_{t=x_k}$.
* **Iterated partial derivatives** (`iterPartial`, `IterPartialExists`): for a finite list
  $l=[k_1,\dots,k_m]$ of directions, $D_lF=D_{k_1}D_{k_2}\cdots D_{k_m}F$ (the last direction is
  differentiated first). "$D_lF$ exists" means that every one of these $m$ classical partial
  derivatives exists **at every point** of the space. A list $l$ encodes the mixed derivative
  $D^{\boldsymbol\alpha}$ with $\alpha_i=\#\{s:k_s=i\}$ (`l.count i`), taken in the order prescribed by $l$.
* **$\mathcal C^{\boldsymbol r}_{\rm mix}$** (`ContDiffMixed r F`): for every list $l$ with
  $\#_i(l)\le r_i$ for all $i$, $D_lF$ exists everywhere and is continuous. This is GKLS's
  definition, read with "classical derivatives $D^{\boldsymbol\alpha}g$" taken in any order (for functions in
  the class all orders give the same function). Requiring all orders makes the positive statements
  (H1), (E) stronger and the negative statement (C) weaker; but the negative half of (E) excludes
  *every* list of length $>r_j$, in particular the list realizing $D^{\boldsymbol\alpha}$ in GKLS's fixed order
  $\prod_kD_k^{\alpha_k}$ for any $\boldsymbol\alpha\le\boldsymbol r_{-j}$ with $|\boldsymbol\alpha|=r_j+1$, so non-membership
  also holds under the single-order reading.
* $\mathrm{ind}(u)=1$ if $u>0$, $0$ otherwise (`ind`).
* $(P_jf)(\boldsymbol y)=\int_{\mathbb R}f(t,\boldsymbol y)\,\rho(t)\,dt$ with $\rho(t)=e^{-t^2/2}/\sqrt{2\pi}$
  (`preintegrate j f y = ∫ t, f (Fin.insertNth j t y) ∂(gaussianReal 0 1)`).

---

## 3. Theorem R1

**Theorem R1** (Lean: `GKLS.gkls_mixed_condition_sharp` in `formal/Theory/Statements.lean`, proved
in `formal/Theory/Proofs.lean`). Let $d=n+1$, $j\in\{1,\dots,d\}$, and $\boldsymbol r\in\mathbb N_0^d$ with
$1\le r_j<\sum_{k\ne j}r_k$. There exists $\phi:\mathbb R^d\to\mathbb R$ such that, with $\theta\equiv1$ and
$f=\theta\cdot\mathrm{ind}(\phi)=\mathrm{ind}\circ\phi$:

* **(H0)** $\theta\equiv1$ lies in $\mathcal C^{\boldsymbol r}_{\rm mix}$, all its iterated partial derivatives of positive
  order vanish identically, and $\theta\in L^p(\gamma_d)$ for every $p$ (so $\theta\in\mathcal W^{\boldsymbol r}_{d,p,\boldsymbol\zeta,\rm mix}$);
* **(H1)** $\phi\in\mathcal C^{\boldsymbol r}_{\rm mix}(\mathbb R^d)$; in fact for **every** list $l$ with at most
  $r_j$ entries equal to $j$ (and arbitrarily many other entries) $D_l\phi$ exists everywhere and is
  continuous ($\phi$ is $\mathcal C^\infty$ in $\boldsymbol x_{-j}$ jointly with up to $r_j$ derivatives in $x_j$;
  in particular $\phi\in\mathcal C^{r_j}(\mathbb R^d)$);
* **(H2)** $D_j\phi(\boldsymbol x)\ge\frac12$ for all $\boldsymbol x$, and $\phi(t,\boldsymbol y)\to\infty$ as $t\to\infty$ for every $\boldsymbol y$;
* **(H3)** for every $\boldsymbol y$ there is $t$ with $\phi(t,\boldsymbol y)=0$ (so $U_j=\mathbb R^{d-1}$, $U_j^+=\emptyset$);
* **(H4)** $|\phi(\boldsymbol x)|\le1+\sum_i|x_i|$, and $|D_l\phi(\boldsymbol x)|\le\frac32$ for every nonempty $l$ with at most $r_j$ entries $j$;
* **(H5)** for every $p\in[1,\infty)$ (in Lean even every $p\in[0,\infty)$: `p : ℝ≥0∞`, `p ≠ ∞`,
  resp. real `p ≥ 0`) and every such $l$: $D_l\phi\in L^p(\gamma_d)$, where
  $\gamma_d=\rho_d\,d\boldsymbol x$ is the standard Gaussian measure; and for every nonempty such $l$ and
  **every** sequence of integrable functions $\zeta_0,\zeta_1,\dots$ on $\mathbb R$:
  $\int_{\mathbb R^d}|D_l\phi(\boldsymbol x)|^p\prod_i\zeta_{\#_i(l)}(x_i)\,d\boldsymbol x<\infty$;
* **(H5w)** these classical derivatives are the weak derivatives: for every such $l$ and every
  $v\in\mathcal C_c^\infty(\mathbb R^d)$, $\int\phi\,D_{\mathrm{rev}(l)}v=(-1)^{|l|}\int(D_l\phi)\,v$
  (`Theory/Weak.lean`: induction on $l$ with Mathlib's integration by parts along a coordinate direction);
* **(H6)** GKLS's condition (h2), for all parameters: for any root function $\psi_j$ with
  $\phi(\psi_j(\boldsymbol y),\boldsymbol y)=0$ for all $\boldsymbol y$ (GKLS's $\psi$; not to be confused with $\psi=g^{-1}$ of §4),
  any list $\eta$, any lists $\gamma^{(1)},\dots,\gamma^{(a)}$ that are nonempty and have
  at most $r_j$ entries $j$, any $b,c\in\mathbb N$, the function
  $$h(\boldsymbol y)=\frac{D_\eta\theta(\psi_j(\boldsymbol y),\boldsymbol y)\prod_{i=1}^aD_{\gamma^{(i)}}\phi(\psi_j(\boldsymbol y),\boldsymbol y)}{(D_j\phi(\psi_j(\boldsymbol y),\boldsymbol y))^b}\,\rho^{(c)}(\psi_j(\boldsymbol y))\qquad(\theta\equiv1)$$
  satisfies $\int_{\mathbb R^{d-1}}|h(\boldsymbol y)|^p\prod_k\zeta_{\alpha_k}(y_k)\,d\boldsymbol y<\infty$ for every $p\ge0$, every
  sequence of integrable $\zeta_i$ and every multi-index $\boldsymbol\alpha$ (Lean: the long `Integrable` conjunct);

and

* **(C)** $P_jf\notin\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}(\mathbb R^{d-1})$;
* **(E)** for every list $l$ of directions in $\mathbb R^{d-1}$ with $|l|\le r_j$, $D_l(P_jf)$ exists
  everywhere and is continuous; for every list with $|l|>r_j$, $D_l(P_jf)$ does **not** exist
  everywhere (one of the classical partial derivatives fails to exist at the origin).

**Explicit counterexample.** Put $m=r_j\ge1$. Let $\chi\in\mathcal C^\infty(\mathbb R)$ with
$0\le\chi\le1$, $\chi=1$ on $[-1,1]$, $\operatorname{supp}\chi\subset(-2,2)$ (Lean: Mathlib's
`ContDiffBump` with radii 1 and 2). Let $P(t)=t_+^{m+1}\chi(t)$ ($t_+=\max(t,0)$),
$B=\max_{0\le a\le m}\sup_t|P^{(a)}(t)|<\infty$, $\varepsilon=1/(2(B+1))\in(0,\frac12]$, $G=\varepsilon P$, and
$$\boxed{\ \phi(\boldsymbol x)=G(x_j)+\sum_{i=1}^d x_i=\varepsilon\,\chi(x_j)\,(x_j)_+^{\,r_j+1}+\sum_{i=1}^d x_i .\ }$$
(Lean: `phiCE k j` with $k=m-1$, `Gk`, `Pk`, `epsk`, `bump` in `Theory/OneDim.lean`, `Theory/Main.lean`.)

### Why (H0)–(H6) are exactly the hypotheses of GKLS Theorem 3 (other than the condition on $\boldsymbol r$)

* $\rho$ (standard normal) is $\mathcal C^\infty$, strictly positive, a probability density. $d\ge2$ is
  automatic: $\sum_{k\ne j}r_k>r_j\ge1$ forces $n\ge1$.
* $\theta\equiv1$: (H0) — $D^{\boldsymbol 0}\theta=1\in\mathcal L_{p,\rho_d}$ and $D^{\boldsymbol\alpha}\theta=0$ for
  $\boldsymbol\alpha\neq\boldsymbol 0$, so $\theta\in\mathcal W^{\boldsymbol r}_{d,p,\boldsymbol\zeta,\rm mix}\cap\mathcal C^{\boldsymbol r}_{\rm mix}$.
* $\phi\in\mathcal C^{\boldsymbol r}_{\rm mix}$ is (H1). The classical mixed derivatives $D^{\boldsymbol\alpha}\phi$,
  $\boldsymbol\alpha\le\boldsymbol r$, are also the weak derivatives: this is (H5w).
  By (H5) with $\zeta_0=\rho$ ($\boldsymbol\alpha=\boldsymbol0$) and with arbitrary integrable $\zeta_i$
  ($\boldsymbol\alpha\ne\boldsymbol0$), $\phi\in\mathcal W^{\boldsymbol r}_{d,p,\boldsymbol\zeta,\rm mix}$ for **every** $p\in[1,\infty)$ and **every**
  admissible weight family $\boldsymbol\zeta$ ((H5) does not even use continuity or monotonicity of the $\zeta_i$).
* Monotonicity and growth (GKLS (phi)): (H2).
* (h1)–(h2): by (H3), $U_j=\mathbb R^{d-1}$, so $U_j^+=\emptyset$ and (h1) is vacuous. (h2) is (H6), which
  quantifies over *all* $a,b,c$, all $\boldsymbol\eta$ and all $\boldsymbol 0\ne\boldsymbol\gamma^{(i)}\le\boldsymbol r$ (a superset
  of GKLS's ranges), all $p$ and all integrable weights. (Why it holds, `Theory/Hcond.lean`: $\psi_j$ is
  the unique root $\psi_j(\boldsymbol y)=g^{-1}(-\sum_iy_i)$, hence continuous; $D_\eta\theta\in\{0,1\}$; each $|D_{\gamma^{(i)}}\phi|\le\frac32$
  (H4); $1/(D_j\phi)^b\le2^b$ (H2); and $\rho^{(c)}=(-1)^c\,\mathrm{He}_c\,\rho$ (Mathlib's Hermite formula) is
  bounded; so $h$ is continuous and bounded.)

So the data satisfy every hypothesis of GKLS Theorem 3 except $r_j\ge\sum_{k\ne j}r_k$, and (C) says that
the conclusion $P_jf\in\mathcal W^{\boldsymbol r_{-j}}\cap\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ fails. As this holds for every
$\boldsymbol r$ with $1\le r_j<\sum_{k\ne j}r_k$, no weakening of the condition that only admits additional
multi-indices with $r_j\ge1$ preserves the theorem (GKLS's theorem itself only concerns
$r_j\ge\sum_{k\ne j}r_k\ge1$; the case $r_j=0$ is discussed, informally, in Remark 4.3).

The function $f=\mathrm{ind}\circ\phi$ also lies in the task's class $\mathcal F_d(K,r,M,c)$ of the appendix with
$K=1$, $r=r_j$, $c>0$ arbitrary and suitable $M$ (one jump surface $\{\phi=0\}$, $|\nabla\phi|\ge D_j\phi\ge\frac12$,
$h_+=1$, $h_-=0$, $|\partial^{\boldsymbol\alpha}\phi|\le\max(\frac32,1+\sqrt d|\boldsymbol z|)$).

### 3a. Theorem R2 (kinks)

**Theorem R2** (Lean: `GKLS.gkls_mixed_condition_kink`). Let $d=n+1$, $j$, and $\boldsymbol r$ with
$1\le r_j$ and $r_j+1<\sum_{k\ne j}r_k$. For the same $\phi$ as in R1 (so (H1)–(H5w) hold) and
$\theta=\phi$, i.e. $f=\phi\cdot\mathrm{ind}(\phi)=\max(\phi,0)$:
$P_jf\notin\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}(\mathbb R^{d-1})$; every $D_l(P_jf)$ with $|l|\le r_j+1$
exists everywhere and is continuous, and no $D_l(P_jf)$ with $|l|>r_j+1$ exists everywhere.

All hypotheses of GKLS Theorem 3 other than the condition on $\boldsymbol r$ hold for $(\theta,\phi)=(\phi,\phi)$:
$\theta=\phi$ satisfies (H1)–(H5w), and (H6) — GKLS's (h2) with $\theta=\phi$ and $\boldsymbol\eta$ having at most $r_j$
entries $j$ — is part of the formal statement (the factor $D^{\boldsymbol\eta}\theta$ is
$\theta(\psi_j(\boldsymbol y),\boldsymbol y)=\phi(\psi_j(\boldsymbol y),\boldsymbol y)=0$ for $\boldsymbol\eta=\boldsymbol0$ and is
bounded by $3/2$ for $\boldsymbol\eta\ne\boldsymbol 0$).
So for kinks, GKLS's condition $r_j\ge\sum_{k\ne j}r_k$ cannot be replaced by any condition that admits
a multi-index with $r_j\le\sum_{k\ne j}r_k-2$; only the borderline $r_j=\sum_{k\ne j}r_k-1$ remains open
(Remark 4.4).

---

## 4. Proof of R1

Throughout, $m=r_j\ge1$, $s=s(\boldsymbol y)=\sum_{i}y_i$ for $\boldsymbol y\in\mathbb R^{d-1}$,
$\Psi(a)=\gamma_1((a,\infty))=\int_a^\infty\rho$ (Lean `gtail`), $g(t)=G(t)+t$ (Lean `gfun`).
Note $\phi(t,\boldsymbol y)=G(t)+t+s(\boldsymbol y)=g(t)+s(\boldsymbol y)$ (Lean `phiCE_insertNth`).

**Step 1 (one-dimensional building blocks; `Theory/OneDim.lean`).**
(a) For $q\ge0$, $\frac{d}{dt}t_+^{q+2}=(q+2)t_+^{q+1}$ for **all** $t$ (for $t\ne0$ locally; at $0$
the one-sided derivatives within $(-\infty,0]$ and $[0,\infty)$ are both $0$). By induction,
$t\mapsto t_+^{q+1}$ is $\mathcal C^{q}$ (`contDiff_pp`). Hence $P=t_+^{m+1}\chi\in\mathcal C^m$, $P\ge0$,
$P=0$ on $(-\infty,0]$, $P(t)=t^{m+1}$ on $[0,1]$, and $P$ has compact support, so each
$P^{(a)}$, $a\le m$, is continuous with compact support, hence bounded; this gives $B$ and
$|G^{(a)}|=\varepsilon|P^{(a)}|\le\varepsilon B\le\frac12$ for $0\le a\le m$ (`epsk_iteratedDeriv_le`).
(b) $g\in\mathcal C^m$, $g'=1+G'\ge\frac12$, so $g$ is strictly increasing; $g(t)=t$ for $t\le0$ and
$g(t)\ge t$, so $g$ is onto (intermediate value theorem). Let $\psi=g^{-1}$. By the inverse function
theorem (Mathlib `Homeomorph.contDiff_symm_deriv`), $\psi\in\mathcal C^m$. Moreover $\psi(v)=v$ for
$v\le0$; $0\le\psi(v)\le v$ for $v\ge0$; and for $0\le v\le1$, $w=\psi(v)\in[0,1]$ satisfies
$w+\varepsilon w^{m+1}=v$ (`psi_eq`).

**Step 2 (the preintegrated function is a ridge function; `preintegrate_phiCE`).** For fixed
$\boldsymbol y$, since $g$ is an increasing bijection,
$\mathrm{ind}(\phi(t,\boldsymbol y))=\mathbf 1\{g(t)>-s\}=\mathbf 1\{t>\psi(-s)\}$. Hence
$$P_jf(\boldsymbol y)=\gamma_1\big((\psi(-s),\infty)\big)=\Psi(\psi(-s))=:H(s),\qquad s=\textstyle\sum_iy_i .$$

**Step 3 ($H\in\mathcal C^m$; `Theory/Tail.lean`).** $\Psi(a)=\Psi(0)-\int_0^a\rho$, so $\Psi'=-\rho$
(fundamental theorem of calculus) and $\Psi\in\mathcal C^\infty$; thus $H=\Psi\circ\psi\circ(-\cdot)\in\mathcal C^m$
(`contDiff_Hfun`). For $s\ge0$: $\psi(-s)=-s$, so $H(s)=\Psi(-s)$ (`Hfun_of_nonneg`).

**Step 4 (quantitative jump of the $(m+1)$-st derivative; `Hfun_lower`).** For $-1\le s\le0$,
$$H(s)-\Psi(-s)\ \ge\ \rho(1)\,\varepsilon\,(|s|/2)^{m+1}.$$
Proof: let $v=-s\in[0,1]$, $w=\psi(v)$. From $w+\varepsilon w^{m+1}=v$, $w\le1$ and $\varepsilon\le\frac12$ we get
$\varepsilon w^{m+1}\le w/2$, hence $w\ge v/2$ and $v-w=\varepsilon w^{m+1}\ge\varepsilon(v/2)^{m+1}$. By the mean value
theorem and since $\rho$ decreases on $[0,\infty)$, $\Psi(w)-\Psi(v)=\int_w^v\rho\ge(v-w)\rho(v)\ge(v-w)\rho(1)$. ∎

**Step 5 (Taylor-type lemma; `taylor_vanish`).** *Let $q\ge0$ and $D\in\mathcal C^q(\mathbb R)$ with $D=0$ on
$[0,\infty)$ and $D^{(q)}$ differentiable at $0$. Then $D(s)=o(|s|^{q+1})$ as $s\to0$.*
Induction on $q$. If $D$ vanishes on $[0,\infty)$ and is differentiable at a point $s\ge0$ then
$D'(s)=0$ (its derivative within $[0,\infty)$ is $0$, and $[0,\infty)$ has unique derivatives;
`deriv_eq_zero_of_vanish`). For $q=0$ this gives $D(s)=D(0)+D'(0)s+o(s)=o(s)$. For $q\to q+1$:
$D'\in\mathcal C^q$, $D'=0$ on $[0,\infty)$ and $(D')^{(q)}=D^{(q+1)}$ is differentiable at $0$, so
$|D'(\sigma)|\le\epsilon|\sigma|^{q+1}$ for $|\sigma|<\delta$; the mean value inequality on $[-|s|,|s|]$ gives
$|D(s)|=|D(s)-D(0)|\le\epsilon|s|^{q+1}\cdot|s|$. ∎

**Step 6 ($H^{(m)}$ is not differentiable at $0$; `not_differentiableAt_iteratedDeriv_Hfun`).**
Let $D(s)=H(s)-\Psi(-s)\in\mathcal C^m$; $D=0$ on $[0,\infty)$ (Step 3). If $H^{(m)}$ were differentiable at
$0$, so would be $D^{(m)}=H^{(m)}-(\Psi(-\cdot))^{(m)}$, and Step 5 would give $D(s)=o(|s|^{m+1})$,
contradicting Step 4 at $s=-\min(\delta/2,1)$. ∎ (Thus $H\in\mathcal C^m\setminus\mathcal C^{m+1}$.)

**Step 7 (mixed derivatives of ridge functions; `Theory/MultiDim.lean`).** Let $K\in\mathcal C^N(\mathbb R)$ and
$F(\boldsymbol y)=K(\sum_iy_i)$. Since $\sum_i(\boldsymbol y\text{ with }y_k:=t)_i=t+(\sum_iy_i-y_k)$, we have
$D_k[K^{(a)}(\sum_iy_i)]=K^{(a+1)}(\sum_iy_i)$ whenever $K^{(a)}$ is differentiable, i.e. $a<N$. By induction on
the list: for every $l$ with $|l|\le N$, $D_lF$ exists everywhere and $D_lF(\boldsymbol y)=K^{(|l|)}(\sum_iy_i)$
(continuous) (`ridge_iterPartial`). If moreover $K^{(N)}$ is not differentiable at $0$ and $|l|\ge N+1$, write
$l=l_1\,{+\!\!+}\,(k::l_2)$ with $|l_2|=N$; existence along $l$ would imply existence along $k::l_2$
(`IterPartialExists.of_append`), i.e. differentiability of $t\mapsto K^{(N)}(t)$ at $t=0$ (take $\boldsymbol y=\boldsymbol0$),
which is false (`ridge_not_exists`).

**Step 8 (conclusions (E) and (C)).** Apply Step 7 to $K=H$, $N=m$ (Steps 2, 3, 6): this is (E). Since
$\sum_{k\ne j}r_k\ge m+1$, the list $L$ containing each direction $k$ of $\mathbb R^{d-1}$ exactly
$r_{j.\mathrm{succAbove}(k)}$ times (`exists_list_count`) satisfies the count constraints of
$\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ and has length $>m$, so $D_L(P_jf)$ does not exist everywhere: (C).
(In particular, for any $\boldsymbol\alpha\le\boldsymbol r_{-j}$ with $|\boldsymbol\alpha|=m+1$, the derivative
$D^{\boldsymbol\alpha}P_jf$ fails to exist on the hyperplane $\sum_iy_i=0$ in every order of differentiation.)

**Step 9 (hypotheses (H1)–(H6); `phi_iterPartial`, `phiCE_*`, `phiCE_hcond_*`).** For constants $c_1,c_2,c_3$ and a function $G_a$,
$D_k\big[c_1\sum_ix_i+c_2G_a(x_j)+c_3\big]=c_1+[k=j]\,c_2\,G_a'(x_j)$ (provided $G_a$ is differentiable when
$k=j$ and $c_2\ne0$). By induction over the list, for every $l$ with $\#_j(l)\le m$,
$$D_l\phi(\boldsymbol x)=[l=\emptyset]\textstyle\sum_ix_i+[\text{all entries of }l\text{ equal }j]\,G^{(|l|)}(x_j)+[|l|=1],$$
and all intermediate derivatives exist (a differentiation in direction $j$ is applied to $G^{(a)}$ only
with $a<\#_j(l)\le m$). This gives (H1) (continuity: only $G^{(a)}$ with $a\le m$ occur with nonzero
coefficient), (H4) ($|G^{(a)}|\le\frac12$), $D_j\phi=1+G'(x_j)\ge\frac12$ and $\phi(t,\boldsymbol y)=g(t)+s\ge t+s\to\infty$
(H2), and $\phi(\psi(-s),\boldsymbol y)=0$ (H3). For (H5): if $l\ne\emptyset$, $D_l\phi$ is continuous and bounded by
$\frac32$, so it lies in every $L^p(\gamma_d)$, and $|D_l\phi|^p\prod_i\zeta_{\#_i(l)}(x_i)$ is a bounded measurable
function times the integrable product $\prod_i\zeta_{\#_i(l)}(x_i)$ (Mathlib `Integrable.fintype_prod`); if
$l=\emptyset$, $\phi=G(x_j)+\sum_ix_i$ with $G$ bounded and each coordinate in $L^p(\gamma_d)$ (Gaussian moments,
Mathlib `memLp_id_gaussianReal'` and the coordinate projections are measure preserving). ∎

### Remarks

**4.1 (what exactly fails; the Sobolev part).** The formal statement (C) concerns the classical part
$\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ of GKLS's conclusion $P_jf\in\mathcal W^{\boldsymbol r_{-j}}\cap\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$.
The weak (Sobolev) part, which is what QMC error bounds use, behaves as follows for the same example
(paper arguments, **not formalized**). Near $s=0$, $g$ is $t$ on $(-\infty,0]$ and $t+\varepsilon t^{m+1}$ on $[0,1]$, so
$\psi$ and $H$ are $\mathcal C^\infty$ up to $s=0$ from each side; by Step 6, $H^{(m)}$ has different one-sided
derivatives at $0$, i.e. $H^{(m+1)}$ has a jump there and $H^{(m)}$ is Lipschitz.
(i) If $\sum_{k\ne j}r_k\ge r_j+2$, a mixed derivative $D^{\boldsymbol\alpha}$, $\boldsymbol\alpha\le\boldsymbol r_{-j}$, $|\boldsymbol\alpha|=m+2$, of
$P_jf=H(\sum_iy_i)$ is, in the sense of distributions, a bounded function plus a nonzero multiple of the
surface measure of the hyperplane $\{\sum_iy_i=0\}$; hence $P_jf\notin\mathcal W^{\boldsymbol r_{-j}}_{d-1,p,\boldsymbol\zeta}$ for every
$p$: the Sobolev part fails as well. (ii) At the borderline $\sum_{k\ne j}r_k=r_j+1$, the example's
$P_jf$ has bounded weak mixed derivatives of order $m+1$, so only the classical part fails; replacing
$t_+^{m+1}$ by $t_+^{m+1/2}$ in $P$ gives $|H^{(m+1)}(s)|\asymp|s|^{-1/2}$ as $s\to0^-$, so that the weak
derivative is not in $L^p$ for $p\ge2$ and the Sobolev part fails for $p\ge2$ too. (iii) Under the
alternative reading of $\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ as "weak derivatives with continuous
representatives", the example also fails: at the borderline the weak derivative $H^{(m+1)}(\sum_iy_i)$
jumps across the hyperplane, and beyond the borderline it is not a function at all.

**4.2 (exact smoothness guaranteed by preintegration).** Under the hypotheses of GKLS Theorem 3 for a
multi-index $\boldsymbol r$ with $r_j\ge1$ but *without* the condition on $\boldsymbol r$, apply GKLS Theorem 3 to
the smaller multi-index $\boldsymbol r'=(r_j,\boldsymbol\beta)$ for any $\boldsymbol 0\ne\boldsymbol\beta\le\boldsymbol r_{-j}$ with
$|\boldsymbol\beta|\le r_j$ (all hypotheses for $\boldsymbol r'$ are implied by those for $\boldsymbol r$, and
$r'_j=r_j\ge|\boldsymbol\beta|$): $P_jf\in\mathcal C^{\boldsymbol\beta}_{\rm mix}$. R1(E) shows that this is optimal: in general
$P_jf\notin\mathcal C^{\boldsymbol\beta}_{\rm mix}$ as soon as $|\boldsymbol\beta|=r_j+1$. So the mixed smoothness that
preintegration guarantees is exactly "total order $r_j$" (the smoothness of $\phi$ in the preintegration
variable), no matter how smooth $\theta,\phi$ are in the other variables. (This remark combines R1 with
the anchor; only R1 is formalized.)

**4.3 (the case $r_j=0$; not formalized).** GKLS's theorem never has $r_j=0$, and R1 assumes $r_j\ge1$.
For $r_j=0$ (where $\mathcal C^{\boldsymbol r}_{\rm mix}$ does not make $D_j\phi$ continuous, although $D_j\phi>0$ may
still exist): if $\sum_{k\ne j}r_k\ge2$, the counterexample for $m=1$ works (it lies in
$\mathcal C^{(1,\boldsymbol r_{-j})}_{\rm mix}\subset\mathcal C^{(0,\boldsymbol r_{-j})}_{\rm mix}$); if $\sum_{k\ne j}r_k=1$, a $g$ with $g'>0$
everywhere but $g'$ discontinuous (e.g. $g(t)=2t+t^2\sin(1/t)$ near $0$) makes $H'=\rho(\psi(-s))/g'(\psi(-s))$
discontinuous.

**4.4 (kinks, borderline case).** R2 leaves open whether $r_j=\sum_{k\ne j}r_k-1$ suffices for kinks.
*Conjecture* (not proved here): yes — formally $D_{y_k}P_j\phi_+=P_j(D_k\phi\cdot\mathrm{ind}(\phi))$
(the boundary term vanishes because $\phi_+=0$ on $\{\phi=0\}$), and GKLS Theorem 3 for the jump
integrand $D_k\phi\cdot\mathrm{ind}(\phi)$, whose data lie in $\mathcal C^{\boldsymbol r-\boldsymbol e_k}_{\rm mix}$, needs only
$r_j\ge\sum_{k\ne j}r_k-1$.

## 4a. Proof of R2 (`Theory/Kink.lean`)

Let $K(s)=\int\max(g(t)+s,0)\,d\gamma_1(t)$ (Lean `Kfun`). Since $u\cdot\mathrm{ind}(u)=\max(u,0)$ and
$\phi(t,\boldsymbol y)=g(t)+s(\boldsymbol y)$, $P_jf(\boldsymbol y)=K(\sum_iy_i)$ (`preintegrate_kink`).
**$K'=H$** (`hasDerivAt_Kfun`): differentiate under the integral sign (Mathlib
`hasDerivAt_integral_of_dominated_loc_of_lip`): $s\mapsto\max(g(t)+s,0)$ is 1-Lipschitz; it is
integrable in $t$ because $|g(t)|\le\frac12+|t|$ and the Gaussian has a first moment; and for every
$t\ne\psi(-s_0)$ — a $\gamma_1$-null exceptional set, a single point — it is differentiable at $s_0$ with
derivative $\mathrm{ind}(g(t)+s_0)$ (locally it equals $g(t)+s$ or $0$). Hence
$K'(s_0)=\int\mathrm{ind}(g(t)+s_0)\,d\gamma_1=H(s_0)$ (Step 2). So $K\in\mathcal C^{m+1}$ and
$K^{(m+1)}=H^{(m)}$ is not differentiable at $0$ (Step 6). Step 7 with $N=m+1$ gives the exact order
$m+1$, and the list $L$ of Step 8 (length $\sum_{k\ne j}r_k\ge m+2$) gives $P_jf\notin\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$. ∎

---

## 5. Comparison with human results

| Human result (exact source) | Their statement (relevant part) | Relation to R1 | Class |
| --- | --- | --- | --- |
| **A-GKLS**: GKLS 2018, Theorem 3 (arXiv:1712.00920; JCAM 344, 259–274) | $r_j\ge\sum_{k\ne j}r_k\ge1$ (+ the conditions of Theorem 2) $\Rightarrow P_jf\in\mathcal W^{\boldsymbol r_{-j}}\cap\mathcal C^{\boldsymbol r_{-j}}_{\rm mix}$ | R1: for **every** $\boldsymbol r$ with $1\le r_j<\sum_{k\ne j}r_k$ the conclusion fails for admissible data. The assumption cannot be weakened (for $r_j\ge1$; it is the classical $\mathcal C_{\rm mix}$ part of the conclusion that fails); with GKLS, the condition is necessary and sufficient in this sense. | **optimality of a registered result's assumption** (top tier of the protocol; recorded as `stronger` in the manifest) |
| GKS 2023, Assumptions 1–2 + Theorem 3 and the remark after it (arXiv:2112.10308v5; SIAM J. Numer. Anal. 61, 135–166) | $\phi\in C^{(\nu_0,\boldsymbol\nu)}$, $\nu_0=|\boldsymbol\nu|+1$ ("it would suffice to have $\nu_0:=|\boldsymbol\nu|$") $\Rightarrow P_0\,\mathrm{ind}(t-\phi)\in\mathcal H^{\boldsymbol\nu}\cap C^{\boldsymbol\nu}$ | Same mechanism, but no formal comparison: their Assumption 2 involves $D_0^{\alpha}\phi$ with $\alpha\le|\boldsymbol\nu|$, so the order condition is built into their hypotheses and R1's $\phi$ (only $|\boldsymbol\nu|-1$ derivatives in $y_0$) does not satisfy them as written. R1 shows that, in the GKLS formulation (where (h) uses only $D^{\boldsymbol\gamma}\phi$, $\boldsymbol\gamma\le\boldsymbol r$), these derivatives cannot be dispensed with. | not registered; related, not compared |
| **A-GKLS**, kink case $\theta=\phi$ of Theorem 3 | same, with $f=\phi\cdot\mathrm{ind}(\phi)=\phi_+$ | **R2**: for kinks no condition admitting some $r_j\le\sum_{k\ne j}r_k-2$ works; the condition is optimal up to one unit, the borderline is open (Remark 4.4), so R2 does not show that the kink condition cannot be weakened. | partial optimality; recorded as `incomparable` |
| **A-GKS**: Gilbert–Kuo–Sloan 2022, Theorem 1 (arXiv:2112.11621) | $\phi\in C^2$; at $\boldsymbol y^*$: $\partial\phi/\partial y_1=0$, $\partial^2\phi/\partial y_1^2\ne0$, $\nabla\phi\ne\boldsymbol0$; for $t=\phi(\boldsymbol y^*)$, $P_1\,\mathrm{ind}(\phi-t)$ has a square-root singularity at $\boldsymbol y^*_{-1}$ along lines not orthogonal to $\nabla_{-1}\phi(\boldsymbol y^*)$ | Different hypothesis: GKS show monotonicity is necessary; R1 shows that *with* uniform monotonicity ($D_j\phi\ge\frac12$) the order of smoothness in $x_j$ is the second necessary ingredient, and pins it down exactly. | incomparable / complementary |

**Is R1 dominated by a known human result?** No result proving the necessity of
$r_j\ge\sum_{k\ne j}r_k$ (or of $\nu_0\ge|\boldsymbol\nu|$) is known to us. GKLS justify it only through their
proof; the task statement records that the necessity was not proved and that no relaxation was found
up to 2026-11. Our search (arXiv API, OpenAlex and Semantic Scholar, including the works citing GKLS,
GKS 2022 and GKS 2023; sources of the closest ones read) found: Griebel–Kuo–Sloan, Math. Comp. 86
(2017) 1847–1854 (kinks, isotropic Sobolev spaces); GKS 2022 (monotonicity necessary, A-GKS);
GKS 2023 and its sequels Gilbert–Kuo–Sloan–Srikumar (arXiv:2212.11493), Gilbert–Kuo–Srikumar
(arXiv:2402.11807), Gilbert (arXiv:2510.22493), which reuse the sufficient condition
$\nu_0=|\boldsymbol\nu|+1$ without a necessity statement; He, SIAM J. Numer. Anal. 57 (2019) (isotropic);
Liu–Owen 2023 and Liu 2024 (direction choice); Bayer–Ben Hammouda–Tempone ($C^\infty$ setting). None
contains a counterexample under monotonicity or a necessity result for the order of smoothness in
GKLS's setting on $\mathbb R^d$. A limitation result of a different kind is Griebel–Kuo–Sloan, *The
smoothing effect of the ANOVA decomposition*, J. Complexity 26 (2010) 523–551 [GKS10], which we know
only through GKS 2023, Appendix A ("for the case of the cube … preintegration needs to be performed
repeatedly if the mixed derivative smoothness is to be improved, with each successive preintegration
yielding at most one additional order of smoothness"); on $\mathbb R^d$ with a monotone direction a
single step can restore full smoothness (GKLS), and R1 identifies exactly how much smoothness in that
direction this requires.

**Value for the problem.** QMC theory for preintegrated integrands uses first-order dominating mixed
smoothness in the remaining $d-1$ variables, $\boldsymbol\nu=(1,\dots,1)$, for which GKLS Theorem 3 requires
$r_j\ge d-1$ orders of smoothness in the preintegration variable. R1 shows (formally) that the classical
statement $P_jf\in\mathcal C^{(1,\dots,1)}_{\rm mix}$ fails for $r_j=d-2$ (any $d\ge3$), even with $\phi$ linear in $\boldsymbol x_{-j}$
and uniformly monotone in $x_j$. For the weak (Sobolev) space $\mathcal H^{(1,\dots,1)}$ that QMC error bounds
actually use, the same example fails for $r_j\le d-3$, and the $t_+^{m+1/2}$ variant fails for $r_j=d-2$
and $p\ge2$ (Remark 4.1; paper arguments, not formalized); at $r_j=d-2$ R1's own example still has
bounded weak mixed derivatives, so the formal result says nothing about QMC rates there. For the
representative instances of the task (all $\phi_i$, $h_\sigma$ analytic, $r=\infty$) the condition is
automatically satisfied, so R1 does not affect them; it delimits how far the smoothing theory can be
pushed for piecewise-$\mathcal C^r$ integrands with finite $r$ (the appendix class
$\mathcal F_d(K,r,M,c)$ with $r<\infty$).

None of the open questions P1–P5 is answered by R1 or R2; they address Target E directly ("构造反例，证明这个
条件不能再弱").

---

## 6. Formalization

Lean 4 (v4.35.0) + Mathlib, project in `formal/`; checked with `python3 /workspace/tools/lean_check.py`
(`lake build` + `lake comparator`; only the axioms `propext`, `Quot.sound`, `Classical.choice`).

| File | Content |
| --- | --- |
| `Theory/Defs.lean` | `partialDeriv`, `iterPartial`, `IterPartialExists`, `ContDiffMixed` ($\mathcal C^{\boldsymbol r}_{\rm mix}$), `ind`, `preintegrate` ($P_j$ with Gaussian $\rho$), `quadricJump` (R3) |
| `Theory/Statements.lean` | statements of R1 `GKLS.gkls_mixed_condition_sharp`, R2 `GKLS.gkls_mixed_condition_kink`, R3 `GKLS.quadric_preintegration_sqrt` (proofs `sorry`) |
| `Theory/Proofs.lean` | the same statements with their proofs |
| `Theory/OneDim.lean` | $t_+^{q}$, the cutoff, $P$, $B$, $\varepsilon$, $g$, $\psi=g^{-1}$ (Step 1) |
| `Theory/Tail.lean` | $\Psi$, $H$, Steps 3–6 (`not_differentiableAt_iteratedDeriv_Hfun`) |
| `Theory/MultiDim.lean` | Step 7 (ridge functions) and the derivative formula of Step 9 |
| `Theory/Main.lean` | Steps 2, 8, 9 and `main_counterexample` (R1 without (H0), (H5w), (H6)) |
| `Theory/Kink.lean` | §4a: the kink profile `Kfun`, `K' = H`, and `kink_counterexample` (R2 without (H5w), (H6)) |
| `Theory/Weak.lean` | (H5w): classical iterated partial derivatives (existing and continuous along all suffixes) are weak derivatives (`integral_iterPartial_eq`) |
| `Theory/Hcond.lean` | (H6): boundedness of $\rho^{(c)}$ (Hermite formula), derivatives of constants, uniqueness and continuity of the root, GKLS's (h2); final assemblies `main_counterexample_h` (R1) and `kink_counterexample_h` (R2) |
| `Theory/Quadric.lean` | §7: `quadMass`, roots, Hölder bound, lower bound, `quadric_assembly` (R3) |

Correspondence between the Lean statement and §3: the conjuncts appear in the order (H0) [three
conjuncts about `fun _ => (1 : ℝ)`], (H1) [two
conjuncts: `ContDiffMixed r φ`, and the stronger "every list with `l.count j ≤ r j`"], (H2) [two],
(H3), (H4) [two], (H5) [`MemLp … (Measure.pi fun _ => gaussianReal 0 1)` for all `p ≠ ∞`; and
`Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))` for all real `p ≥ 0` and all
integrable `ζ`], (H5w) [`∫ x, φ x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l φ x * v x`
for all smooth compactly supported `v`], (H6) [the `Integrable (fun y => |iterPartial η (fun _ => 1) (…) * (γs.map …).prod /
partialDeriv j φ (…) ^ b * iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i))`
conjunct], (C) [`¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => ind (φ x)))`],
(E) [two]. The hypothesis `hsum : r j < ∑ k : Fin n, r (j.succAbove k)` is $r_j<\sum_{k\ne j}r_k$.
R2 has the same (H1)–(H5w) conjuncts and the analogous (H6) with `iterPartial η φ` in place of
`iterPartial η (fun _ => 1)` and the extra hypothesis `η.count j ≤ r j` (GKLS: $\boldsymbol\eta<\boldsymbol r$); its conclusions concern `preintegrate j (fun x => φ x * ind (φ x))`
($\theta=\phi$), with `r j + 1` in place of `r j`, under `hsum : r j + 1 < ∑ k : Fin n, r (j.succAbove k)`.

## 7. R3 (Target A, minor): explicit Hölder-½ bounds for quadric jumps

**Anchor A-GKS** (A. D. Gilbert, F. Y. Kuo, I. H. Sloan, *Preintegration is not smoothing when
monotonicity fails*, in Advances in Modeling and Simulation, Springer 2022, 169–191, arXiv:2112.11621,
Theorem 1 = `thm:sqrt`), verbatim: "Let $\phi\in C^2(\mathbb R^d)$, and assume that
$\boldsymbol y^*=(y_1^*,\boldsymbol y^*_{-1})$ is such that $\frac{\partial\phi}{\partial y_1}(\boldsymbol y^*)=0$,
$\frac{\partial^2\phi}{\partial y_1^2}(\boldsymbol y^*)\ne0$, and $\nabla\phi(\boldsymbol y^*)\ne\boldsymbol0$. Define
$t:=\phi(\boldsymbol y^*)$. Then $(P_1f_t)(\boldsymbol y_{-1})$ has a square-root singularity at $\boldsymbol y^*_{-1}$ …
along any line in $\mathbb R^{d-1}$ through $\boldsymbol y^*_{-1}$ that is not orthogonal to
$\nabla_{-1}\phi(\boldsymbol y^*)$", where $f_t=\mathrm{ind}(\phi-t)$ (their eq. (general)). Their Examples 1–3
compute $P_1f_t$ in closed form for specific quadratic $\phi$ (e.g. $\phi=y_2-y_1^2$ gives
$P_1f_0=\int_{-\sqrt{y_2}}^{\sqrt{y_2}}\rho$ for $y_2>0$, $0$ otherwise), and the proof of their Theorem 1 gives a
local two-sided asymptotic of the form $c\sqrt{\cdot}\,(1+o(1))$ near the fold. They conclude that there is
"no theoretical support" for QMC on the preintegrated function. The task lists, as progress on
Target A, "给出与平方根奇性匹配的正则性上界" (a regularity upper bound matching the square-root singularity).

**Theorem R3** (Lean `GKLS.quadric_preintegration_sqrt`). Let $d=n+1$, $j$, $a\ne0$ and arbitrary
$\beta,\kappa:\mathbb R^{n}\to\mathbb R$; let
$f(\boldsymbol x)=\mathrm{ind}\big(a x_j^2+\beta(\boldsymbol x_{-j})x_j+\kappa(\boldsymbol x_{-j})\big)$ (Lean `quadricJump`), and
$D=\beta^2-4a\kappa$. Then for all $\boldsymbol y,\boldsymbol y'$:

* **(U)** $|P_jf(\boldsymbol y)-P_jf(\boldsymbol y')|\le\dfrac{|\beta(\boldsymbol y)-\beta(\boldsymbol y')|+\sqrt{|D(\boldsymbol y)-D(\boldsymbol y')|}}{|a|\sqrt{2\pi}}$;
* **(Z)** if $a<0$ and $D(\boldsymbol y)\le0$, then $P_jf(\boldsymbol y)=0$;
* **(L)** if $a<0$ and $|\beta(\boldsymbol y)|+\sqrt{D(\boldsymbol y)}\le2|a|R$ (both roots in $[-R,R]$), then
  $P_jf(\boldsymbol y)\ge\rho(R)\sqrt{D(\boldsymbol y)}/|a|$.

*Proof* (`Theory/Quadric.lean`). Fibrewise $P_jf(\boldsymbol y)=Q(a,\beta(\boldsymbol y),\kappa(\boldsymbol y))$ with
$Q(a,b,c)=\gamma_1(\{t:at^2+bt+c>0\})$ (`preintegrate_quadricJump`). Let $s=\sqrt{\max(D,0)}$ and
$r_\pm=(-b\pm s)/(2a)$. Since $4a(at^2+bt+c)=(2at+b)^2-D$: for $a<0$ the positivity set is the open
interval between $r_+\le r_-$ (if $D\ge0$, $4aq=(2at+b-s)(2at+b+s)<0$ iff $|2at+b|<s$; if $D<0$ it is
empty and $r_+=r_-$), so $Q=\Psi(r_+)-\Psi(r_-)$ (`quad_pos_iff_neg`, `quadMass_of_neg`). For $a>0$,
$Q(a,b,c)=1-Q(-a,-b,-c)$ because the zero set of a quadratic with $a\ne0$ lies in $\{r_+,r_-\}$ and is
$\gamma_1$-null (`quadMass_add_neg`). The tail $\Psi$ is $1/\sqrt{2\pi}$-Lipschitz ($|\Psi'|=\rho\le1/\sqrt{2\pi}$),
$|\sqrt u-\sqrt v|\le\sqrt{|u-v|}$ (`abs_sqrt_sub_sqrt_le`), and
$|r_\pm-r'_\pm|\le(|b-b'|+|s-s'|)/(2|a|)$; adding the two roots gives (U) (`quadMass_holder`). If $a<0$ and
$D\le0$ then $s=0$, $r_+=r_-$ and $Q=0$: (Z). If $a<0$ and the roots lie in $[-R,R]$, the mean value
theorem gives $Q=\rho(\xi)(r_--r_+)$ with $|\xi|\le R$, so $Q\ge\rho(R)\,s/|a|$ (`quadMass_lower`, using
that $\rho$ is even and decreasing on $[0,\infty)$): (L). ∎

**Consequences (paper argument).** For a quadric $q(\boldsymbol x)=\boldsymbol x^\top A\boldsymbol x+\boldsymbol b^\top\boldsymbol x+c$ and $A_{jj}\ne0$ one has
$a=A_{jj}$, $\beta$ affine and $\kappa$ quadratic in $\boldsymbol y=\boldsymbol x_{-j}$, so $D$ is a quadratic polynomial and
(U) gives $|P_jf(\boldsymbol y)-P_jf(\boldsymbol y')|\le C_B|\boldsymbol y-\boldsymbol y'|^{1/2}$ on every bounded set $B$
(for $|\boldsymbol y-\boldsymbol y'|\le1$): $P_jf$ is locally Hölder-$\frac12$, for every such quadric and every
direction $j$ with $A_{jj}\ne0$ (for a closed ellipsoid, $A$ definite, every direction qualifies; this is
the "closed jump surface not monotone in any direction" of the task's hard track; saddles are covered
too). For an ellipsoid ($f=\mathrm{ind}(-q)$, so $a=-A_{jj}<0$), (Z) and (L) show that $P_jf$ vanishes
outside the "shadow" $\{D>0\}$ and grows like $\sqrt D$ inside it; at a boundary point of the shadow
with $\nabla D\ne0$ (the generic fold points of GKS) the exponent $\frac12$ is attained, so (U) cannot be
improved there (this attainment is a paper consequence of (Z)+(L), not a formal statement). Without such a
fold point the exponent can be better: for $f=\mathrm{ind}(1+|\boldsymbol y|^2-x_j^2)$, $D=4(1+|\boldsymbol y|^2)>0$ everywhere
and $P_jf$ is $\mathcal C^\infty$. So "Hölder-$\frac12$ and not better" holds at fold points with $\nabla D\ne0$.

**Comparison.** A-GKS (Theorem 1): general $\phi\in C^2$, existence of the square-root singularity at
folds, with a local asymptotic in the proof and closed forms for specific quadrics (Examples 1–3). R3:
the narrower class of jumps quadratic in $x_j$ (all quadric surfaces with $A_{jj}\ne0$), with a *global*
explicit upper bound (U) and explicit-constant versions (Z), (L) of the lower bound; (Z)+(L) are special
cases of what GKS show, and (U) is an elementary generalization of their closed forms. Class:
**incomparable, of low novelty**. It gives a regularity upper bound matching the square-root singularity
for this class, but **no** QMC/RQMC rate (Hölder-$\frac12$ alone does not give a rate faster than Monte
Carlo in dimension $\ge2$), so it does not answer P1.

## References

* [GKLS18] A. Griewank, F. Y. Kuo, H. Leövey, I. H. Sloan, High dimensional integration of kinks and jumps — smoothing by preintegration, J. Comput. Appl. Math. 344 (2018) 259–274, doi:10.1016/j.cam.2018.04.009, arXiv:1712.00920.
* [GKS13] M. Griebel, F. Y. Kuo, I. H. Sloan, The smoothing effect of integration in $\mathbb R^d$ and the ANOVA decomposition, Math. Comp. 82 (2013) 383–400.
* [GKS17] M. Griebel, F. Y. Kuo, I. H. Sloan, Note on "The smoothing effect of integration in $\mathbb R^d$ and the ANOVA decomposition", Math. Comp. 86 (2017) 1847–1854, doi:10.1090/mcom/3172.
* [GKS22] A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Preintegration is not smoothing when monotonicity fails, in Advances in Modeling and Simulation, Springer (2022) 169–191, arXiv:2112.11621.
* [GKS23] A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Analysis of preintegration followed by quasi-Monte Carlo integration for distribution functions and densities, SIAM J. Numer. Anal. 61 (2023) 135–166, doi:10.1137/21M146658X, arXiv:2112.10308.
* [GKS10] M. Griebel, F. Y. Kuo, I. H. Sloan, The smoothing effect of the ANOVA decomposition, J. Complexity 26 (2010) 523–551 (cited via [GKS23], Appendix A).
* [GKSS] A. D. Gilbert, F. Y. Kuo, I. H. Sloan, A. Srikumar, arXiv:2212.11493. [GKSr] A. D. Gilbert, F. Y. Kuo, A. Srikumar, arXiv:2402.11807. [G25] A. D. Gilbert, arXiv:2510.22493.
* Z. He, On the error rate of conditional quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 57 (2019) 854–874.
* S. Liu, A. B. Owen, Preintegration via active subspace, SIAM J. Numer. Anal. 61 (2023) 495–514, arXiv:2202.02682; S. Liu, arXiv:2212.13232.
