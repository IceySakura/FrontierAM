# Preintegration without monotonicity: sharp ½-Hölder regularity, and stratified RQMC at the Lipschitz rate up to a logarithm

**摘要（中文）。** 本文针对目标 A（Gilbert–Kuo–Sloan 指出的空白：单调性不成立时预积分后的 QMC 没有理论支持）。在“沿预积分变量强凸/强凹（非单调但非退化）”这一自然类上，我们证明预积分后的跳跃函数整体是 ½-Hölder 的（常数显式、指数与常数均最优），折点函数是 $C^{1,1/2}$ 的，三阶切触给出 ⅓-Hölder（与 GKS 例 4 的立方根奇性匹配），二次曲面沿任意非零锥方向都有显式模；进而证明“预积分 + 分层（jittered）RQMC”在无单调性时无偏、均方误差 $O(n^{-1-1/(d-1)})$，在阴影边界横截条件下达到 $O(n^{-1-2/(d-1)}\log n)$，即与 Lipschitz（单调光滑）情形只差一个对数因子；对任意方向的二次曲面跳跃面（本题困难赛道，含光滑振幅 θ）同样成立；反之，任何确定性求积（QMC）在该类上的最坏误差不优于 $n^{-1/(d-1)}$（R11），说明 GKS 所说的“缺乏 QMC 理论支持”是真实的最坏情形障碍，随机化正是消除它的手段。全部结论（R1–R11）都在 Lean 4/Mathlib 中形式化并通过 `lake comparator`。

**Chosen target: A** (open problem 1 of the appendix; the gap that Gilbert–Kuo–Sloan state for QMC
after preintegration when monotonicity fails). Some results also bear on **target B** (an RQMC rate for
discontinuous integrands); see §6.4, §8, §9, §11 and §15.

All results R1–R11 below are formalized in Lean 4 / Mathlib and checked with `lake comparator`
(statements in `formal/Theory/Statements.lean`, proofs in `formal/Theory/Proofs.lean` plus auxiliary
modules `Sublevel`, `Preint`, `Piecewise`, `Sharp`, `Stratified`, `Cube`, `Quadric`, `Tube`, `Refined`, `Cubic`, `QuadStrat`, `QuadTheta`, `Lower`;
namespace `Cubature`). §16 gives the dictionary between the mathematics and the Lean statements. Statements labelled *Conjecture* or
*Remark (informal)* are not formalized and are not claimed as results.

---

## 0. Summary

### 0.1 The human results we compare with (checked against the original sources)

* **[GKS] A. D. Gilbert, F. Y. Kuo, I. H. Sloan, *Preintegration is not smoothing when monotonicity
  fails*, arXiv:2112.11621 (in *Advances in Modeling and Simulation*, Springer 2022).**
  Setting: $I_d f=\int_{\mathbb R^d}f(\boldsymbol y)\rho_d(\boldsymbol y)\,d\boldsymbol y$, $\rho_d=\prod\rho(y_k)$,
  $f_t(\boldsymbol y)=\operatorname{ind}(\phi(\boldsymbol y)-t)$ with $\operatorname{ind}(s)=1$ for $s>0$ and $0$ otherwise,
  $(P_1f)(\boldsymbol y_{-1})=\int_{\mathbb R} f(y_1,\boldsymbol y_{-1})\rho(y_1)\,dy_1$.
  - *Theorem 1 (verbatim content).* Let $\phi\in C^2(\mathbb R^d)$ and let $\boldsymbol y^*$ satisfy
    $\partial\phi/\partial y_1(\boldsymbol y^*)=0$, $\partial^2\phi/\partial y_1^2(\boldsymbol y^*)\neq0$ and
    $\nabla\phi(\boldsymbol y^*)\ne\boldsymbol 0$. Put $t:=\phi(\boldsymbol y^*)$. Then $(P_1f_t)(\boldsymbol y_{-1})$ has a
    square-root singularity at $\boldsymbol y^*_{-1}$, similar to those in Examples 1 and 2, along any line through
    $\boldsymbol y^*_{-1}$ that is not orthogonal to $\nabla_{-1}\phi(\boldsymbol y^*)$. (The proof shows that the
    contribution of a neighbourhood of $y_1^*$ is $0$ on one side and $2c\sqrt{y_2-y_2^*}\,\rho(y_1^*)(1+o(1))$ on the other.)
  - *Theorem 2*: such singular points are generically not isolated (they occur for $t$ in an interval).
  - *Section 1.2, Examples 1–4*: $\phi=y_2-y_1^2$ gives $P_1f(y_2)=2\sqrt{y_2}\rho(0)+O(y_2^{3/2})$ (square root);
    $\phi=y_2^2-y_1^2-1$ (two square roots); $\phi=y_2^2-y_1^2$ (absolute value); $\phi=y_1^3-y_2$ (cube root). For the
    kink $\max(\phi,0)$ with $\phi$ of Example 1: $P_1f=\tfrac43y_2^{3/2}\rho(0)+O(y_2^{5/2})$.
  - *Conclusion (quoted in the task):* "because of the proven lack of smoothness, the resulting preintegrated function no
    longer belongs to the space of $(d-1)$-variate functions of dominating mixed smoothness of order one, and as a
    consequence there is at present no theoretical support for our use of QMC integration for this $(d-1)$-dimensional
    integral."
* **[He19] Z. He, *On the error rate of conditional quasi-Monte Carlo for discontinuous functions*, SIAM J. Numer. Anal.
  57 (2019) 854–874, arXiv:1708.09512.** All rate results for conditional (= preintegrated) RQMC assume
  *Assumption 3* (numbering of arXiv v2): $(D_j\phi)(\boldsymbol x)\neq0$ for all $\boldsymbol x\in\mathbb R^d$, i.e. strict
  monotonicity in the conditioning variable. Under it (plus growth conditions) *Theorem 14* (the main theorem of §4) gives
  $\mathbb E|\hat I(P_jf)-I(f)|=O(n^{-1+\gamma_j+\epsilon})$, and $O(n^{-1+\epsilon})$ when the growth exponents are arbitrarily small.
* **[HW15] Z. He, X. Wang, *On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions*, SIAM J.
  Numer. Anal. 53 (2015) 2488–2503, doi:10.1137/15M1007963**, Theorems 3.5/3.6 (restated as Proposition 2.3 of He,
  arXiv:1702.03361): for $f=g\,\mathbf 1_\Omega$ on $[0,1]^d$, $g\in L^2$ with $V_{HK}(g)<\infty$, $\partial\Omega$ with a
  $(d-1)$-dimensional Minkowski content, a scrambled $(t,d)$-sequence gives
  $\operatorname{Var}\hat I\le c\,M_g^2\,n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)}$; for partially axis-parallel $\Omega$ with irregular
  dimension $d_u$: $n^{-1-1/(2d_u-1)}(\log n)^{2d/(2d_u-1)}$.
* Classical tools: the **van der Corput sublevel-set lemma** (E. M. Stein, *Harmonic Analysis*, Princeton 1993, Ch. VIII,
  §1.2; A. Carbery, M. Christ, J. Wright, J. Amer. Math. Soc. 12 (1999) 981–1015) — if $|g^{(k)}|\ge\alpha$ then
  $|\{|g|\le\eta\}|\le C_k(\eta/\alpha)^{1/k}$ — and its sharp polynomial form due to **Pólya** ($|\{|p|\le\eta\}|\le4(\eta/2)^{1/k}$
  for monic $p$ of degree $k$); **stratified (jittered) sampling** (S. Haber, Math. Comp. 20 (1966) 361–368, variance $O(n^{-1-2/s})$ for $C^1$
  integrands; the analogous $O(n^{-1-2\beta/s})$ for $\beta$-Hölder integrands is standard) and **Bakhvalov's** lower bound $n^{-1/2-\beta/s}$ for randomized
  integration of $\beta$-Hölder functions (1959).

### 0.2 Our results and how they relate

| # | Result (all formalized) | Closest human result | Class |
|---|---|---|---|
| R1 | Sharp sublevel estimate: $\phi$ $\alpha$-strongly convex $\Rightarrow$ $\lvert\{\lvert\phi\rvert\le\eta\}\rvert\le4\sqrt{\eta/\alpha}$, constant 4 optimal | van der Corput ($k=2$) / Pólya | equal (classical tool) |
| R2 | **Non-monotone preintegration is ½-Hölder**, globally, with explicit constants: $\lvert P(\boldsymbol y)-P(\boldsymbol y')\rvert\le L_\theta\delta+4\rho B\sqrt{L\delta/\alpha}$ | GKS Thm 1 (singularity exists) | incomparable; matching upper bound = progress on GKS gap |
| R2′ | Same with **finitely many non-degenerate tangencies per line** (pieces where $\phi(\cdot,\boldsymbol y)$ is strongly convex, concave or monotone): bound $L_\theta\delta+\rho Br(4\sqrt{L\delta/\alpha}+2L\delta/\alpha)$ | GKS Thms 1–2 (global form of their hypothesis) | incomparable |
| R3 | Non-monotone preintegrated **kinks** are $C^{1,1/2}$ (Zygmund bound $O(\lVert h\rVert^{3/2})$ for second differences) | GKS §1.2 kink remark | incomparable |
| R4 | Sharpness: constant 4 in R1 attained; exponent ½ in R2 attained (two-sided bound on GKS Example 1) | GKS Example 1 | equal (quantitative restatement) |
| R5 | **Preintegration + stratified (jittered) RQMC without monotonicity: unbiased, explicit MSE bound; on $[0,1)^s$ MSE $\le(L_\theta/m+4\rho B\sqrt{L/(\alpha m)})^2/m^s=O(n^{-1-1/s})$**, $n=m^s$, $s=d-1$ | GKS conclusion; He19 Thm 14 (needs monotonicity); HW15 Thm 3.5 | incomparable; first rate guarantee for preintegrated RQMC when monotonicity fails |
| R6 | **Quadric jump surfaces, any direction $u$ with $\langle u,Au\rangle\neq0$**: exact modulus $\lvert P_u(x)-P_u(x')\rvert\le(\lvert\Delta\beta\rvert+\sqrt{\lvert\Delta D\rvert})/(\sqrt{2\pi}\lvert\langle u,Au\rangle\rvert)$, asymptotically sharp on GKS Examples 1 and 3; explicit local ½-Hölder constant | GKS Thm 1, Examples 1–3 | incomparable |
| R7 | **Faster rate under a tube (Minkowski-content type) condition** $(\nu\otimes\mu)\{\lvert\phi\rvert\le\eta\}\le K\eta$: jittered preintegration MSE $\le\frac2n\big(L_\theta^2h^2+4\rho KB^2Lh\sqrt{Lh/\alpha}\big)=O(n^{-1-3/(2s)})$; hypothesis verified on GKS Example 1 (explicit bound $16/(\sqrt{2\pi}m^{s+2})\cdot\sqrt{1/(2m)}$) | HW15 Thm 3.5 (same type of boundary assumption, exponent $1+\frac1{2d-1}$); GKS conclusion | incomparable; exponent $1+\frac{3}{2(d-1)}$ |
| R8 | **Lipschitz rate up to a log without monotonicity**: refined band estimate $\lambda\{\lvert\psi\rvert\le\eta\}\le4\sqrt6\,\eta/\sqrt{\alpha H}$ away from tangency; under transversality of the shadow boundary ($\mu\circ H^{-1}\le K\,\mathrm{Leb}$, $H=\max_t\phi$) jittered preintegration has MSE $O(n^{-1-2/s}\log n)$; GKS Example 1: MSE $\le(48\log\frac m2+24)/(\pi m^{s+3})$ | GKS conclusion; HW15 Thm 3.5; He19 Thm 14; Haber (Lipschitz rate) | incomparable; exponent $1+\frac2{d-1}$ (log) |
| R9 | **Third-order tangencies** ($\partial_t\phi$ strongly convex/concave, GKS Example 4): $\lambda\{\lvert\phi\rvert\le\eta\}\le10(\eta/\alpha)^{1/3}$, preintegrated jump is ⅓-Hölder, two-sided cube-root bound on GKS Example 4 | GKS Example 4, Remark 1 | incomparable (matching upper bound) |
| R10, R10′ | **Quadrics in any non-null direction** (hard-track instances; amplitude $1$ or Lipschitz $\theta$): preintegration + stratified sampling has MSE $\le\frac1n\frac{(2\lVert A\rVert h)^2+K(\eta^2\log\frac{D_{\max}}{2\eta}+3\eta^2)}{\pi\langle u,Au\rangle^2}=O(n^{-1-2/s}\log n)$, no monotonicity or uniform Lipschitz condition needed | GKS Thm 1 / Examples 1–3, He19, HW15 | incomparable |
| R11 | **Deterministic lower bound**: for any $n$ nodes there is a non-monotone jump (class of R2) whose preintegrated function vanishes at all nodes with $\int P\ge c_s/m\asymp n^{-1/s}$; no deterministic rule (QMC) beats $n^{-1/s}$ — slower than MC for $s\ge3$ | GKS conclusion; Bakhvalov (technique) | incomparable |

**What is new relative to GKS.** GKS prove (locally, asymptotically) that the singularity *exists* and conclude that there
is no theory supporting QMC after non-monotone preintegration. We prove the *matching upper bound*: whenever the
non-monotonicity is non-degenerate (strong convexity/concavity along the preintegration variable — exactly the situation of
their Theorem 1, made uniform), the preintegrated function is globally ½-Hölder with explicit constants (R2), kinks become
$C^{1,1/2}$ (R3), both exponents are optimal (R4, GKS Examples), third-order tangencies give exactly the cube-root
regularity of GKS Example 4 (R9), and for quadric surfaces — the hard instances of this task — the modulus is explicit for
every direction (R6). Using only this regularity we prove the first convergence-rate guarantee for
a randomized-QMC method (stratified/jittered sampling) after preintegration *without* monotonicity (R5): MSE
$O(n^{-1-1/(d-1)})$, strictly faster than Monte Carlo, unbiased, with explicit constants, and fully formal. The task statement
lists both "a regularity upper bound matching the square-root singularity" and "an RQMC guarantee faster than MC after
preintegration for a natural class of non-monotone jumps" as progress on target A; R2/R4/R6 give the former, R5 the latter.
The rate in R5 is the optimal randomized rate for the ½-Hölder class it uses (Bakhvalov), so it cannot be improved
using regularity alone. R7 goes beyond it by exploiting that the singular set is thin: under a tube condition of the same
type as the Minkowski-content assumption of He–Wang (formally verified on GKS Example 1), the mean-square error of the
same estimator is $O(n^{-1-3/(2(d-1))})$, compared with $O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})$ for scrambled nets on the
original integrand (HW15 Thm 3.5) and $O(n^{-1-1/d})$ for plain jittered sampling. Finally R8 shows that, under
transversality of the *shadow boundary* (the set of lines tangent to the surface; again verified formally on GKS Example 1),
the same estimator attains MSE $O(n^{-1-2/(d-1)}\log n)$: **the rate of jittered sampling for Lipschitz (e.g. monotone,
smooth) preintegrated integrands, up to a single logarithm.** For stratified RQMC, therefore, the loss caused by the failure
of monotonicity — the phenomenon GKS identified — is at most logarithmic. R10 proves the same for every quadric jump surface
preintegrated along any non-null direction (the hard instances of this task), without any uniformity assumption. Conversely,
R11 shows that **deterministic** rules cannot do better than $n^{-1/s}$ on the non-monotone class (fooling functions that
vanish at all nodes), which is slower than Monte Carlo for $s\ge3$: the GKS observation "no theoretical support for QMC"
is a genuine worst-case barrier, and randomization is what removes it.

**Limitations (stated honestly).** (i) All rates come from stratification and therefore deteriorate with the dimension
($n^{-1-2/s}$ with $s=d-1$); they are not dimension-robust rates of the $n^{-1+\epsilon}$ type that He19 obtains for scrambled
nets under monotonicity, and for $d=64$ the gain over Monte Carlo is modest. When the jump surface lives in a low-dimensional
subspace (as in the hard-track instances) the relevant $s$ is the dimension of that subspace, but exploiting this requires
stratifying in the right coordinates. (ii) As in GKS, GKLS18 and He19, the one-dimensional preintegral is assumed to be
computed exactly (white-box access along lines); §13 sketches the black-box cost. (iii) For deterministic QMC we only
prove the negative result R11; nothing is proved for scrambled nets (Conjecture C3).


---

## 1. Setting and notation

* $t\in\mathbb R$ is the preintegration variable, $\boldsymbol y\in Y$ the remaining variables; for the problem of the task
  $Y=\mathbb R^{d-1}$ (or $[0,1)^{d-1}$ after the usual transformation), but most results hold for an arbitrary
  (pseudo)metric space $Y$, which also allows restricting to bounded sets $Y_0\subset\mathbb R^{d-1}$ (local versions).
* $\nu$ is a probability measure on $\mathbb R$ (the law of the preintegration variable) with density at most $\rho$:
  $\nu(S)\le\rho\,|S|$ for all $S$. For $\nu=N(0,1)$, $\rho=1/\sqrt{2\pi}$; for $\nu=U[0,1]$, $\rho=1$.
* Jump integrand $f(t,\boldsymbol y)=\theta(t,\boldsymbol y)\operatorname{ind}(\phi(t,\boldsymbol y))$ ($\operatorname{ind}(s)=\mathbf 1\{s>0\}$ as in GKS);
  kink integrand $f(t,\boldsymbol y)=\max(\phi(t,\boldsymbol y),0)$.
* Preintegrated functions (Lean: `preintJump ν φ θ`, `preintKink ν φ`):
  $$P(\boldsymbol y)=\int_{\mathbb R}\theta(t,\boldsymbol y)\operatorname{ind}(\phi(t,\boldsymbol y))\,\nu(dt),\qquad
    P^{\rm kink}(\boldsymbol y)=\int_{\mathbb R}\max(\phi(t,\boldsymbol y),0)\,\nu(dt).$$
* **Non-monotone non-degeneracy (NM$_\alpha$):** for every $\boldsymbol y$, $t\mapsto\phi(t,\boldsymbol y)$ is $\alpha$-strongly convex
  or $\alpha$-strongly concave ($\alpha>0$), i.e. $\phi(\cdot,\boldsymbol y)\mp\frac\alpha2t^2$ is convex/concave; for $C^2$
  functions this is $\partial_t^2\phi\ge\alpha$ or $\le-\alpha$. Then $\phi(\cdot,\boldsymbol y)$ is never monotone on $\mathbb R$:
  each line meets the surface at most twice, and tangencies are non-degenerate — the situation of GKS Theorem 1
  ($\partial_t\phi=0\Rightarrow\partial_t^2\phi\ne0$), made uniform.
* **Uniform Lipschitz bounds:** $|\phi(t,\boldsymbol y)-\phi(t,\boldsymbol y')|\le L\,d(\boldsymbol y,\boldsymbol y')$,
  $|\theta(t,\boldsymbol y)-\theta(t,\boldsymbol y')|\le L_\theta\,d(\boldsymbol y,\boldsymbol y')$, $|\theta|\le B$.

Relation to the class $\mathcal F_d(K,r,M,c)$ of the appendix: an integrand with one jump surface $\{\phi=0\}$ ($K=1$),
$h_+=\theta$, $h_-=0$; NM$_\alpha$ is the non-monotone (hard) case. The uniform Lipschitz bound in $\boldsymbol y$ requires
$\partial_{\boldsymbol y}\phi$ to be bounded uniformly in $t$ on the region considered (R2 holds on any subset
$Y_0\subset\mathbb R^{d-1}$); this is the case for the GKS examples and for quadrics preintegrated along a principal axis,
but not for quadrics in a general direction (there $\partial_{\boldsymbol y}\phi(t,\boldsymbol y)$ grows linearly in $t$). For quadric
surfaces with an arbitrary (hidden, rotated) orientation — the hard-track instances — R6 gives the regularity directly,
without any uniformity assumption.

---

## 2. R1 — sharp sublevel-set estimate (Lean: `Cubature.volume_sublevel_le_of_strongConvexOn`)

**Theorem R1.** Let $\phi:\mathbb R\to\mathbb R$ be $\alpha$-strongly convex, $\alpha>0$, and $\eta\ge0$. Then
$$\lambda\{t:\ |\phi(t)|\le\eta\}\le4\sqrt{\eta/\alpha}.$$

*Proof.* Strong convexity means: for $x,y$ and $a,b\ge0$, $a+b=1$,
$\phi(ax+by)\le a\phi(x)+b\phi(y)-\frac\alpha2ab(x-y)^2$. Let $S=\{|\phi|\le\eta\}$.

*Case 1: there is $u$ with $\phi(u)<-\eta$.* Then $u\notin S$ and $S\subset(S\cap(-\infty,u))\cup(S\cap(u,\infty))$.
Let $x<y<u$ with $x,y\in S$. Write $y=ax+bu$, $a=\frac{u-y}{u-x}$, $b=\frac{y-x}{u-x}$. Using $\phi(y)\ge-\eta$,
$\phi(x)\le\eta$, $\phi(u)\le-\eta$:
$-\eta\le a\eta-b\eta-\frac\alpha2ab(u-x)^2$, i.e. $\frac\alpha2ab(u-x)^2\le2a\eta$, so $\frac\alpha2(y-x)(u-x)\le2\eta$
(as $b(u-x)=y-x$), and since $u-x\ge y-x$: $\alpha(y-x)^2\le4\eta$. Hence $S\cap(-\infty,u)$ has diameter
$\le2\sqrt{\eta/\alpha}$; symmetrically (write $x=au+by$) so does $S\cap(u,\infty)$. A subset of $\mathbb R$ has Lebesgue
measure at most its diameter, so $\lambda(S)\le4\sqrt{\eta/\alpha}$.

*Case 2: $\phi\ge-\eta$ everywhere.* For $x<y$ in $S$ the midpoint inequality gives
$-\eta\le\phi(\frac{x+y}2)\le\eta-\frac\alpha8(y-x)^2$, so $y-x\le4\sqrt{\eta/\alpha}$ and $\lambda(S)\le\operatorname{diam}S\le4\sqrt{\eta/\alpha}$. $\square$

For strongly concave $\phi$ apply R1 to $-\phi$ (Lean: `Aux.volume_sublevel_le_of_strong`). **Sharpness** is R4a.

*Comparison.* This is the order-2 van der Corput sublevel lemma with the optimal constant (Pólya's constant for quadratics),
under the weaker hypothesis of strong convexity (no $C^2$ needed). **Class: equal** (classical; used as the engine of R2–R5).

---

## 3. R2 — ½-Hölder regularity of non-monotone preintegration
(Lean: `Cubature.preintJump_holder_half_of_density_le`, Gaussian case `Cubature.preintJump_holder_half`)

**Theorem R2.** Assume NM$_\alpha$, the uniform Lipschitz bounds $L,L_\theta$, $|\theta|\le B$, $\theta(\cdot,\boldsymbol y)$
measurable, and $\nu(S)\le\rho\lambda(S)$. Then for all $\boldsymbol y,\boldsymbol y'\in Y$, $\delta=d(\boldsymbol y,\boldsymbol y')$,
$$|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\,\delta+4\rho B\sqrt{L\delta/\alpha}.$$
For $\nu=N(0,1)$: $|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+\frac{4B}{\sqrt{2\pi}}\sqrt{L\delta/\alpha}$.

*Proof.* $\phi(\cdot,\boldsymbol z)$ is convex or concave on $\mathbb R$, hence continuous, so all sets below are measurable
and the integrands are bounded measurable, hence $\nu$-integrable. Pointwise,
$$\theta(t,\boldsymbol y)\operatorname{ind}\phi(t,\boldsymbol y)-\theta(t,\boldsymbol y')\operatorname{ind}\phi(t,\boldsymbol y')
=\operatorname{ind}\phi(t,\boldsymbol y)\,[\theta(t,\boldsymbol y)-\theta(t,\boldsymbol y')]+[\operatorname{ind}\phi(t,\boldsymbol y)-\operatorname{ind}\phi(t,\boldsymbol y')]\,\theta(t,\boldsymbol y').$$
If the two indicators differ, $\phi(t,\boldsymbol y)$ and $\phi(t,\boldsymbol y')$ lie on different sides of $0$, so
$|\phi(t,\boldsymbol y)|\le|\phi(t,\boldsymbol y)-\phi(t,\boldsymbol y')|\le L\delta$. Hence the modulus of the difference is at most
$L_\theta\delta+B\,\mathbf 1_S(t)$, $S=\{t:|\phi(t,\boldsymbol y)|\le L\delta\}$. Integrating against $\nu$:
$|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+B\nu(S)\le L_\theta\delta+B\rho\lambda(S)\le L_\theta\delta+4\rho B\sqrt{L\delta/\alpha}$
by R1. (For $N(0,1)$, $\nu(S)=\int_S\frac{e^{-t^2/2}}{\sqrt{2\pi}}dt\le\lambda(S)/\sqrt{2\pi}$.) $\square$

*Remarks.* (i) Nothing is assumed about monotonicity, smoothness of $\phi$ in $t$ beyond strong convexity/concavity, or
the number/location of tangencies (GKS Theorem 2 shows they are typically not isolated; R2 is uniform over all of them).
(ii) $Y$ is any pseudometric space, so R2 applies on any subset $Y_0\subset\mathbb R^{d-1}$ with the Lipschitz constants
on $Y_0$ (local version, e.g. for closed surfaces such as $\phi=r^2-t^2-|\boldsymbol y|^2$ on balls).
(iii) With R4 and GKS Theorem 1 the exponent ½ is exact: the preintegrated function is ½-Hölder and, at every
non-degenerate tangency, not better.

**Theorem R2′ (piecewise non-degenerate sections; Lean: `Cubature.preintJump_holder_half_piecewise`).** Suppose that
for every $\boldsymbol y$ the real line is covered by $r$ convex pieces $J_1(\boldsymbol y),\dots,J_r(\boldsymbol y)$ such that on each
piece $t\mapsto\phi(t,\boldsymbol y)$ is $\alpha$-strongly convex, or $\alpha$-strongly concave, or $\alpha$-strongly monotone
($\alpha|t-t'|\le|\phi(t,\boldsymbol y)-\phi(t',\boldsymbol y)|$), that $\phi(\cdot,\boldsymbol y)$ is continuous, and that the other
hypotheses of R2 hold. Then
$$|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+\rho Br\Big(4\sqrt{L\delta/\alpha}+\frac{2L\delta}{\alpha}\Big).$$
*Proof.* As in R2, $|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+B\nu(S)$ with $S=\{|\phi(\cdot,\boldsymbol y)|\le\eta\}$, $\eta=L\delta$
(`Aux.jump_diff_le`). $S\subset\bigcup_i(S\cap J_i)$; on a strongly convex/concave piece the argument of R1 works verbatim
inside the convex set $J_i$ (all convex combinations stay in $J_i$), giving $\lambda(S\cap J_i)\le4\sqrt{\eta/\alpha}$
(`Aux.volume_band_le_of_strongConvexOn`); on a strongly monotone piece two points of $S\cap J_i$ satisfy
$\alpha|t-t'|\le|\phi(t)-\phi(t')|\le2\eta$, so $\lambda(S\cap J_i)\le2\eta/\alpha$ (`Aux.volume_band_le_of_monotone`). Sum over
the $r$ pieces and use $\nu\le\rho\lambda$. $\square$

This is the global, quantitative form of the hypothesis of GKS Theorem 1: a $C^2$ function whose sections have finitely
many critical points, all non-degenerate ($\partial_t\phi=0\Rightarrow|\partial_t^2\phi|\ge\alpha'$) with uniform constants,
admits such a cover (near each critical point $|\partial_t^2\phi|$ is bounded below, away from them $|\partial_t\phi|$ is), so
R2′ applies to all of GKS's square-root situations at once, including the non-isolated singular sets of their Theorem 2
and lines with several tangencies (e.g. $\phi(t,\boldsymbol y)=y_1-t^3+3t$, two critical points per line).

*Comparison with GKS Theorem 1.* GKS: under $\phi\in C^2$, $\partial_1\phi(\boldsymbol y^*)=0\ne\partial_1^2\phi(\boldsymbol y^*)$,
$\nabla\phi(\boldsymbol y^*)\ne0$, the preintegrated function has a square-root singularity (a *lower* bound on the
irregularity, local and asymptotic). R2: under the uniform version of the same non-degeneracy, a global, quantitative
*upper* bound of exactly the same order, with explicit constants. Neither implies the other. **Class: incomparable**;
it is precisely the "regularity upper bound matching the square-root singularity" named in the task as progress on
target A, and it is what R5 uses to give QMC-type (RQMC) theoretical support.

---

## 4. R3 — non-monotone preintegrated kinks are $C^{1,1/2}$
(Lean: `Cubature.preintKink_second_difference_of_density_le`, Gaussian case `Cubature.preintKink_second_difference`)

**Theorem R3.** Let $E$ be a seminormed group, $\phi:\mathbb R\times E\to\mathbb R$ satisfy NM$_\alpha$,
$\phi(\cdot,\boldsymbol y)\in L^1(\nu)$, $|\phi(t,\boldsymbol y)-\phi(t,\boldsymbol y')|\le L\|\boldsymbol y-\boldsymbol y'\|$ and the
second-difference bound $|\phi(t,\boldsymbol y+h)+\phi(t,\boldsymbol y-h)-2\phi(t,\boldsymbol y)|\le M\|h\|^2$. Then
$$\big|P^{\rm kink}(\boldsymbol y+h)+P^{\rm kink}(\boldsymbol y-h)-2P^{\rm kink}(\boldsymbol y)\big|\le M\|h\|^2+4\rho L\|h\|\sqrt{L\|h\|/\alpha}.$$
So the second differences are $O(\|h\|^{3/2})$: $P^{\rm kink}$ lies in the Zygmund/Hölder class $\Lambda^{3/2}=C^{1,1/2}$.

*Proof.* Fix $t$; write $a=\phi(t,\boldsymbol y-h)$, $b=\phi(t,\boldsymbol y)$, $c=\phi(t,\boldsymbol y+h)$, $l=L\|h\|$, $m=M\|h\|^2$;
so $|a-b|,|c-b|\le l$ and $|a+c-2b|\le m$. *If $|b|>l$*, $a,b,c$ have the same strict sign; then
$a_++c_+-2b_+$ equals $a+c-2b$ or $0$, of modulus $\le m$. *If $|b|\le l$*, a case check on the signs of $a,b,c$
(Lean: `Aux.kink_pt_near`) gives $|a_++c_+-2b_+|\le m+l$. Hence the integrand is bounded by $m+l\,\mathbf 1_S(t)$,
$S=\{|\phi(t,\boldsymbol y)|\le l\}$, and integrating, $|\Delta_h^2P^{\rm kink}(\boldsymbol y)|\le m+l\,\nu(S)\le m+4\rho l\sqrt{l/\alpha}$ by R1. $\square$

*Comparison.* GKS (§1.2) observe for Example 1 that kinks give a $y^{3/2}$ singularity ($\frac43y_2^{3/2}\rho(0)$). R3 is the
matching global upper bound (order $3/2$ is attained there). **Class: incomparable** (new upper bound; complements GKS).
In dimension $d-1=2$ this already gives an integrable mixed derivative (bounded Hardy–Krause variation on bounded boxes),
but we do not formalize that consequence.

---

## 5. R4 — sharpness (Lean: `Cubature.sublevel_bound_sharp`, `Cubature.gks_parabola_two_sided`)

**R4a.** For $\alpha>0$, $\eta\ge0$, $\phi(t)=\frac\alpha2t^2-\eta$ is $\alpha$-strongly convex and
$\lambda\{|\phi|\le\eta\}=\lambda[-2\sqrt{\eta/\alpha},2\sqrt{\eta/\alpha}]=4\sqrt{\eta/\alpha}$: the constant 4 in R1 is optimal.

**R4b.** For GKS Example 1, $\phi(t,y)=y-t^2$ ($\alpha=2$, $L=1$), $\theta=1$, $\nu=N(0,1)$, and $0\le y\le1$:
$$\frac{2e^{-1/2}}{\sqrt{2\pi}}\sqrt y\ \le\ P(y)-P(0)\ \le\ \frac{4}{\sqrt{2\pi}}\sqrt{y/2}.$$
*Proof.* $\{t:y-t^2>0\}=(-\sqrt y,\sqrt y)$ and $P(0)=0$; on this interval the density is $\ge e^{-1/2}/\sqrt{2\pi}$, giving
the lower bound; the upper bound is R2 with $\alpha=2,L=1,B=1,L_\theta=0$. $\square$
So the exponent ½ in R2 cannot be improved. **Class: equal** (a non-asymptotic restatement of GKS Example 1 plus the
classical optimality of the quadratic case).

---

## 6. R5 — stratified (jittered) RQMC after non-monotone preintegration

### 6.1 Algorithm (jittered preintegration estimator, JPI)

Input: $m\ge1$, $s=d-1$, $n=m^s$; cells $C_{\boldsymbol k}=\prod_{j=1}^s[k_j/m,(k_j+1)/m)$, $\boldsymbol k\in\{0,\dots,m-1\}^s$
(Lean: `cubeCell s m k`).
1. Draw independent $\boldsymbol X_{\boldsymbol k}\sim U(C_{\boldsymbol k})$ (one uniform point per cell; existence of such a design:
   `Cubature.jittered_design_exists`).
2. For each $\boldsymbol k$ compute the preintegral $P(\boldsymbol X_{\boldsymbol k})=\int\theta(t,\boldsymbol X_{\boldsymbol k})\operatorname{ind}\phi(t,\boldsymbol X_{\boldsymbol k})\nu(dt)$.
   Under NM$_\alpha$ the set $\{t:\phi(t,\boldsymbol X_{\boldsymbol k})>0\}$ is an interval or the complement of one (at most two
   roots), so this is a one-dimensional integral over at most two intervals (for $\theta=1$ and $\nu=N(0,1)$ it is a
   difference of normal CDFs at the roots), as in the preintegration framework of GKS/GKLS18/He19.
3. Output $\hat Q_n=\frac1n\sum_{\boldsymbol k}P(\boldsymbol X_{\boldsymbol k})$.

More generally (R5b) any equal-mass partition $\{A_i\}$ of a probability space $(Y,\mu)$ may be used, with
$\boldsymbol X_i\sim\mu(\cdot\mid A_i)$ independent.

### 6.2 Results

**R5a (classical, abstract stratified sampling; `Cubature.stratified_sampling_mse`).** Let $A_i$ ($i\in\iota$, $n=|\iota|$)
be measurable, pairwise disjoint, covering $Y$ up to a $\mu$-null set; $\boldsymbol X_i$ independent with law
$n\,\mu|_{A_i}$; $F$ measurable, $|F|\le B$, and $|F(x)-F(x')|\le c_i$ for $x,x'\in A_i$. Then
$\mathbb E\hat Q=\int F\,d\mu$ and $\mathbb E(\hat Q-\int Fd\mu)^2\le n^{-2}\sum_ic_i^2$.

*Proof.* $\mathbb E F(\boldsymbol X_i)=n\int_{A_i}F\,d\mu$; summing over the partition gives unbiasedness. By independence
$\operatorname{Var}\hat Q=n^{-2}\sum_i\operatorname{Var}F(\boldsymbol X_i)$; $\boldsymbol X_i\in A_i$ a.s., so $F(\boldsymbol X_i)$ lies in an
interval of length $2c_i$ and Popoviciu's inequality gives $\operatorname{Var}F(\boldsymbol X_i)\le c_i^2$. $\square$

**R5b (`Cubature.preintJump_stratified_mse`).** Under the hypotheses of R2 (with $L,L_\theta\ge0$), $Y$ a pseudometric
space with Borel sets, for any partition as in R5a, any set $G$ of cells with $\operatorname{diam}A_i\le D_i$ ($i\in G$) and
arbitrary other cells (e.g. unbounded Gaussian tail cells):
$$\mathbb E\hat Q=\int P\,d\mu,\qquad
\mathbb E\Big(\hat Q-\int Pd\mu\Big)^2\le\frac1{n^2}\Big(\sum_{i\in G}\big(L_\theta D_i+4\rho B\sqrt{LD_i/\alpha}\big)^2+\sum_{i\notin G}(2B)^2\Big).$$
*Proof.* $P$ is continuous by R2 (hence measurable) and $|P|\le B$. Apply R5a with $c_i=L_\theta D_i+4\rho B\sqrt{LD_i/\alpha}$
(R2) for $i\in G$ and $c_i=2B$ otherwise. $\square$

**R5c (`Cubature.preintJump_jittered_cube_mse`) — explicit rate.** Take $Y=\mathbb R^s$ with the sup-distance,
$\mu$ = Lebesgue measure on $[0,1)^s$, and the JPI estimator of §6.1. Under the hypotheses of R2 on $\mathbb R^s$
(Lipschitz constants w.r.t. the sup-distance, $L,L_\theta\ge0$),
$$\mathbb E\hat Q_n=\int_{[0,1)^s}P,\qquad
\mathbb E\Big(\hat Q_n-\int_{[0,1)^s}P\Big)^2\le\frac{\big(L_\theta/m+4\rho B\sqrt{L/(\alpha m)}\big)^2}{m^s}
\;\le\;\frac{2L_\theta^2}{n^{1+2/s}}+\frac{32\rho^2B^2L}{\alpha\,n^{1+1/s}}.$$
*Proof.* The cells are measurable, pairwise disjoint (two different multi-indices differ in some coordinate $j$, and the
half-open intervals $[k_j/m,(k_j+1)/m)$ are disjoint), cover $[0,1)^s$ ($k_j=\lfloor mx_j\rfloor$), and have sup-diameter
$\le1/m$. Apply R5b with $G$ = all cells, $D_i=1/m$, and $\sum_{\boldsymbol k}c^2=m^sc^2$. $\square$

**R5e (`Cubature.preintJump_integral_eq`) — Fubini.** If $\phi,\theta$ are jointly measurable and $|\theta|\le B$, then
$\int_YP\,d\mu=\iint\theta\operatorname{ind}\phi\,d(\nu\otimes\mu)$. Hence $\hat Q_n$ is an unbiased estimator of the original
$d$-dimensional integral $I(f)=\mathbb E f(T,\boldsymbol Y)$, $(T,\boldsymbol Y)\sim\nu\otimes\mu$, and R5c is a mean-square
error bound for $I(f)$ itself.

### 6.3 Gaussian parameter space (Remark, informal — the formal statement R5b covers it, the computation below is not formalized)

For $Y=\mathbb R^s$ with $\mu=N(0,I_s)$, use the product of the $m$ equal-mass slabs $[q_i,q_{i+1})$, $\Phi(q_i)=i/m$.
The $\le2sm^{s-1}$ cells touching an outermost slab are put outside $G$; for the others $D_{\boldsymbol k}\le\max_jw_{k_j}$ with
slab widths $w_i=q_{i+1}-q_i$, $\sum_{i=1}^{m-2}w_i=q_{m-1}-q_1\le2\sqrt{2\ln m}$ and $\max w_i=O(1)$. With
$(a+b)^2\le2a^2+2b^2$ and $\sum_{\boldsymbol k}D_{\boldsymbol k}\le s\,m^{s-1}\sum_iw_i$, R5b gives
$\mathrm{MSE}=O\big(s\sqrt{\log n}\;n^{-1-1/s}\big)$ (constants depending on $L,L_\theta,B,\alpha$) for the Gaussian problem
of the task, provided R2's hypotheses hold on $\mathbb R^s$ (or on the region where the cells lie).

### 6.4 Comparison

* **GKS conclusion (target A gap).** GKS state that, because $P_1f\notin$ the mixed-smoothness space, there is "no
  theoretical support" for QMC after non-monotone preintegration. R5 gives, for the natural non-degenerate non-monotone
  class NM$_\alpha$ a *proved* convergence rate for a randomized QMC-type rule (jittered
  sampling = a randomly jittered grid, the simplest RQMC rule): unbiased, RMSE $O(n^{-1/2-1/(2(d-1))})$, strictly
  faster than Monte Carlo, with explicit constants, and formalized. It does not need any mixed smoothness — it needs only
  the ½-Hölder regularity of R2, which is exactly what survives without monotonicity.
  We do **not** claim a positive result for deterministic QMC (by R11 no deterministic rule beats $n^{-1/s}$ in the worst
  case on this class)
  or for scrambled nets (see C3). **Class: incomparable** (progress on the
  stated gap).
* **He19, Theorem 14.** CQMC rates $O(n^{-1+\epsilon})$ under Assumption 3 ($\partial_j\phi\ne0$ everywhere). R5 drops
  monotonicity entirely (NM$_\alpha$ is incompatible with it) but obtains the weaker exponent $1/2+1/(2(d-1))$ for the RMSE
  and uses stratified sampling instead of scrambled nets. **Incomparable.**
* **HW15, Theorem 3.5 (target B).** For $f=g\mathbf 1_\Omega$ with Minkowski-content boundary, scrambled-net variance
  $O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})$ on the original $d$-dimensional integrand. For integrands in our class
  (the jump surface is non-degenerately non-monotone along one coordinate, $\theta$ Lipschitz, preintegration available),
  JPI has MSE $O(n^{-1-1/(d-1)})$, a strictly larger exponent ($\frac1{d-1}>\frac1{2d-1}$ for all $d\ge2$) and no logarithm.
  The estimators differ (and ours uses exact one-dimensional preintegration), so this is **incomparable**, not a strict
  improvement of HW15. Plain jittered sampling on the $d$-dimensional indicator gives $O(n^{-1-1/d})$ (Haber-type
  bound); preintegration improves the exponent to $1/(d-1)$ even without monotonicity.
* **Optimality relative to the information used.** For the class of $\frac12$-Hölder functions on $[0,1]^s$, the optimal
  randomized error is $\asymp n^{-1/2-1/(2s)}$ (Bakhvalov 1959). R5c attains it, so no better rate can be derived from R2
  alone. Better rates need the finer structure of $P$ (smooth away from a codimension-one "shadow boundary"); this is
  what R7 and R8 exploit.

---

## 7. R6 — quadric jump surfaces in an arbitrary direction
(Lean: `Cubature.quadric_preint_holder`, `Cubature.quadric_preint_holder_local`)

Let $E$ be a real inner-product space (e.g. $\mathbb R^d$), $A:E\to E$ linear and continuous (not necessarily symmetric),
$b\in E$, $c\in\mathbb R$, $\phi(z)=\langle z,Az\rangle+\langle b,z\rangle+c$, $f=\operatorname{ind}\phi$, and a direction $u$
with $a:=\langle u,Au\rangle\ne0$. Preintegration along $u$: $P_u(x)=\Pr(\phi(x+Zu)>0)$, $Z\sim N(0,1)$
(Lean: `preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => 1)`). Put
$\beta(x)=\langle x,Au\rangle+\langle u,Ax\rangle+\langle b,u\rangle$, $D(x)=\beta(x)^2-4a\phi(x)$.

**Theorem R6a.** For all $x,x'\in E$:
$$|P_u(x)-P_u(x')|\le\frac{|\beta(x)-\beta(x')|+\sqrt{|D(x)-D(x')|}}{\sqrt{2\pi}\,|a|}.$$
**Theorem R6b.** If moreover $\|u\|=1$, $\|x\|,\|x'\|\le R$ and $\delta=\|x-x'\|$:
$$|P_u(x)-P_u(x')|\le\frac{2\|A\|\delta+\sqrt{8\|A\|(2\|A\|R+\|b\|)\,\delta}}{\sqrt{2\pi}\,|a|}.$$
So $P_u$ is locally ½-Hölder for every quadric and every direction off the null cone of $A$.

*Proof.* (1) Expansion: $\phi(x+tu)=at^2+\beta(x)t+\phi(x)$ (bilinearity). (2) Completing the square,
$4a(at^2+\beta t+\gamma)=4a^2(t-m)^2-D$ with $m=-\beta/(2a)$, $D=\beta^2-4a\gamma$. For $a<0$ this gives
$\{t:at^2+\beta t+\gamma>0\}=(m-w,m+w)$ with $w=\sqrt{\max(D,0)}/(2|a|)$ (empty if $D\le0$). (3) For intervals,
$|\gamma_1(l,r)-\gamma_1(l',r')|\le(|l-l'|+|r-r'|)/\sqrt{2\pi}$, since $(l,r)\setminus(l',r')\subset[\min(l,l'),\max(l,l')]\cup[\min(r,r'),\max(r,r')]$.
(4) $|m-m'|=|\beta-\beta'|/(2|a|)$ and $|w-w'|\le\sqrt{|D-D'|}/(2|a|)$, using $|\sqrt p-\sqrt q|\le\sqrt{|p-q|}$ and
$|\max(D,0)-\max(D',0)|\le|D-D'|$; this gives R6a for $a<0$. (5) For $a>0$: the zero set of a non-degenerate quadratic has at
most two points, hence is Gaussian-null, so $\gamma_1\{\phi>0\}=1-\gamma_1\{-\phi>0\}$, and $-\phi$ has leading coefficient
$-a<0$ and the same discriminant. (6) R6b: $\beta(x)-\beta(x')=\langle x-x',Au\rangle+\langle u,A(x-x')\rangle$,
$|\beta|\le2\|A\|R+\|b\|$, $|\phi(x)-\phi(x')|\le(2\|A\|R+\|b\|)\delta$, $|a|\le\|A\|$, and
$D(x)-D(x')=(\beta(x)-\beta(x'))(\beta(x)+\beta(x'))-4a(\phi(x)-\phi(x'))$ give $|D(x)-D(x')|\le8\|A\|(2\|A\|R+\|b\|)\delta$. $\square$

*Sharpness.* GKS Example 1 ($\phi=y-t^2$): $a=-1$, $\beta=0$, $D=4y$, so R6a gives $P(y)-P(0)\le2\sqrt y/\sqrt{2\pi}$,
while $P(y)=2\int_0^{\sqrt y}\rho=2\rho(0)\sqrt y(1+o(1))$: R6a is **asymptotically exact**. GKS Example 3
($\phi=y^2-t^2$): $D=4y^2$, R6a gives $|P(y)-P(0)|\le2|y|/\sqrt{2\pi}$, again asymptotically exact
($P=2|y|\rho(0)+O(|y|^3)$). GKS Example 2 is the case $D=4(y^2-1)$.

*Relevance to the task.* The hard-track jump/kink surfaces are anisotropic quadrics in unknown rotated subspaces. Since
$u\mapsto\langle u,Au\rangle$ vanishes only on the null cone, preintegration along *any* direction off the null cone
(in particular a random one, almost surely) yields an explicitly ½-Hölder function, with constant $\propto|\langle u,Au\rangle|^{-1}$
(R6a) — this suggests choosing $u$ to maximise $|\langle u,Au\rangle|$, i.e. the dominant eigenvector of the symmetric part
of $A$ (cf. target D; not formalized). Unlike R2, R6 needs no uniform Lipschitz bound in $x$ (for quadrics
$\partial_x\phi(x+tu)$ grows linearly in $t$ unless $u$ is a principal axis).

*Comparison.* GKS Theorem 1 and Examples 1–3 exhibit the singularities for (special) quadrics; R6 gives the exact global
modulus for all quadrics and all non-null directions. **Class: incomparable.**

---

## 8. R7 — a faster stratified rate under a tube condition
(Lean: `Cubature.preintJump_stratified_mse_tube`, `Cubature.preintJump_jittered_cube_mse_tube`,
`Cubature.gks_example_tube_condition`, `Cubature.gks_example_jittered_mse`)

R5 uses only the worst-case modulus $\sqrt h$ on every cell. But the square-root behaviour occurs only for parameters
$\boldsymbol y$ whose line is (nearly) tangent to the surface; R7 quantifies this with an average (tube) condition.

**Tube condition (T$_K$).** $(\nu\otimes\mu)\{(t,\boldsymbol y):|\phi(t,\boldsymbol y)|\le\eta\}\le K\eta$ (needed only at
$\eta=Lh$). This is the probabilistic form of a finite Minkowski content of the jump surface (the assumption of HW15 Thm
3.5). It holds, e.g., with $K=2/c$ whenever $\phi$ is $c$-strictly monotone in some *other* coordinate $y_j$ (each
$y_j$-section of the tube is an interval of length $\le2\eta/c$) — which is the typical situation of GKS Theorem 1, where
$\partial_t\phi=0$ but $\nabla\phi\ne0$ at the tangency; and, by the co-area formula, whenever $|\nabla_{(t,\boldsymbol y)}\phi|\ge c$
near a surface of finite area (informal; only the example below is formalized).

**Theorem R7a.** Under the hypotheses of R2 with $L,L_\theta\ge0$, $\phi$ jointly measurable, $\mu$ a probability measure,
an equal-mass partition into cells of diameter $\le h$, independent $\boldsymbol X_i\sim\mu(\cdot\mid A_i)$, and (T$_K$):
$$\mathbb E\Big(\hat Q-\int P\,d\mu\Big)^2\le\frac2n\Big(L_\theta^2h^2+4\rho KB^2\,Lh\sqrt{Lh/\alpha}\Big).$$

**Theorem R7b (unit cube).** For the JPI estimator of §6.1 ($h=1/m$, $n=m^s$), if (T$_K$) holds at $\eta=L/m$:
$$\mathbb E\Big(\hat Q_n-\int_{[0,1)^s}P\Big)^2\le\frac2{m^s}\Big(\frac{L_\theta^2}{m^2}+4\rho KB^2\frac Lm\sqrt{\frac L{\alpha m}}\Big)
=O\big(n^{-1-\frac3{2s}}\big).$$

**R7c/R7d (GKS Example 1, all hypotheses verified formally).** $\phi(t,\boldsymbol y)=y_0-t^2$ on $\mathbb R\times[0,1)^{s+1}$:
(T$_2$) holds for every probability $\nu$ and every $\eta\ge0$, and for $t\sim N(0,1)$, $\theta=1$, jittered $\boldsymbol y$ with $n=m^{s+1}$:
$$\mathbb E\big(\hat Q_n-I\big)^2\le\frac{16}{\sqrt{2\pi}\,m^{s+2}}\sqrt{\frac1{2m}}=\frac{8}{\sqrt\pi}\,n^{-1-\frac3{2(s+1)}}.$$

*Proof of R7a.* (i) *Weighted stratification lemma* (`Aux.stratified_mse_weighted`): if $|F(x)-F(x')|\le g(x)$ for
$x,x'$ in the same cell, then $\operatorname{Var}F(\boldsymbol X_i)=\operatorname{Var}(F(\boldsymbol X_i)-F(x_i))\le\mathbb E(F(\boldsymbol X_i)-F(x_i))^2
\le\mathbb E\,g(\boldsymbol X_i)^2=n\int_{A_i}g^2d\mu$ for any $x_i\in A_i$; summing over the independent cells,
$\mathrm{MSE}=n^{-2}\sum_i\operatorname{Var}F(\boldsymbol X_i)\le n^{-1}\int g^2d\mu$.
(ii) The first step of the proof of R2 (`Aux.jump_diff_le`) gives $|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+B\,\nu\{t:|\phi(t,\boldsymbol y)|\le L\delta\}$;
for $\delta\le h$ this is $\le g(\boldsymbol y):=L_\theta h+B\,m(\boldsymbol y)$, $m(\boldsymbol y)=\nu\{t:|\phi(t,\boldsymbol y)|\le Lh\}$.
(iii) $g^2\le2L_\theta^2h^2+2B^2m^2$ and, by R1, $0\le m\le M_0:=4\rho\sqrt{Lh/\alpha}$, so $m^2\le M_0m$.
(iv) Tonelli: $\int m\,d\mu=(\nu\otimes\mu)\{|\phi|\le Lh\}\le KLh$. Hence
$\int g^2d\mu\le2L_\theta^2h^2+2B^2M_0KLh$, which is the claim. R7b: cells of §6.1, $h=1/m$. R7c: for fixed $t$ the section
$\{\boldsymbol y\in[0,1)^{s+1}:|y_0-t^2|\le\eta\}$ lies in a box with side $2\eta$ in coordinate $0$ and $1$ elsewhere; integrate
over $t$. R7d: R7b with $\alpha=2$ ($y_0-t^2$ is 2-strongly concave), $L=1$ (sup-distance), $\theta=1$ ($B=1$, $L_\theta=0$),
$\rho=1/\sqrt{2\pi}$, $K=2$. $\square$

*Comparison.* HW15 Theorem 3.5 assumes the same kind of boundary regularity (finite Minkowski content) and obtains, for
scrambled nets applied to the original $d$-dimensional integrand, variance $O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})$.
R7 obtains MSE $O(n^{-1-3/(2(d-1))})$ (no logarithm) for preintegration + jittered sampling on the non-monotone class
NM$_\alpha$; the exponent gain is $\frac3{2(d-1)}$ versus $\frac1{2d-1}$ (e.g. $d=3$: $1.75$ vs $1.2$; $d=64$: $1.024$ vs
$1.008$). The estimators differ (ours needs exact one-dimensional preintegration), so the class is **incomparable**; for
the GKS gap it is a second, sharper RQMC guarantee without monotonicity. R7 also shows that the Bakhvalov barrier of R5
(worst-case ½-Hölder) is not the truth for these integrands.

---

## 9. R8 — the Lipschitz rate up to a logarithm
(Lean: `Cubature.volume_band_le_of_strongConcaveOn_max`, `Cubature.preintJump_stratified_mse_height`,
`Cubature.preintJump_stratified_mse_height_convex`, `Cubature.preintJump_jittered_cube_mse_height`,
`Cubature.gks_example_jittered_mse_log`)

R7 still uses the crude bound $m_\eta(\boldsymbol y)\le4\rho\sqrt{\eta/\alpha}$ on every line. Lines that stay far from
tangency cross the surface transversally and their band is much thinner. R8 makes this quantitative.

**Theorem R8a (refined band estimate).** Let $\psi$ be $\alpha$-strongly concave with maximum value $H$ (attained at $u$),
$\eta>0$, $2\eta\le H$. Then $\lambda\{|\psi|\le\eta\}\le2\sqrt{24\eta^2/(\alpha H)}=4\sqrt6\,\eta/\sqrt{\alpha H}$.

*Proof.* $\psi(u)=H>\eta$, so $u$ is not in the band. Let $x<y<u$ be band points. (1) Strong concavity at the midpoint of
$x,u$ and $\psi\le H$ give $\frac\alpha8(u-x)^2\le H-\frac{\psi(x)+H}2\le\frac{H+\eta}2$, i.e. $\alpha(u-x)^2\le4(H+\eta)$.
(2) Writing $y=ax+bu$, concavity gives $\eta\ge\psi(y)\ge-a\eta+bH$, so $bH\le2\eta$. (3) $y-x=b(u-x)$, hence
$\alpha H(y-x)^2=(bH)\,b\,\alpha(u-x)^2\le2\eta\cdot\frac{2\eta}H\cdot4(H+\eta)\le24\eta^2$ because $H+\eta\le\frac32H$.
The same holds to the right of $u$; the band is contained in two sets of diameter $\le\sqrt{24\eta^2/(\alpha H)}$. $\square$

**Hypotheses (H).** For every $\boldsymbol y$, $t\mapsto\phi(t,\boldsymbol y)$ is $\alpha$-strongly concave (closed surfaces
with the inside positive, parabolic caps, GKS Examples 1–2, …), with maximum $H(\boldsymbol y)=\max_t\phi(t,\boldsymbol y)$
attained; $H$ is measurable, $H\le H_{\max}$ $\mu$-a.e., and the **shadow boundary is transversal**: the law of
$H(\boldsymbol Y)$, $\boldsymbol Y\sim\mu$, has density at most $K$ ($\mu\circ H^{-1}\le K\,\mathrm{Leb}$). The shadow boundary
$\{H=0\}$ is exactly the set where the preintegrated function has its square-root singularity (GKS Theorem 1); the
condition says it is a non-degenerate level set (e.g. $|\nabla H|\ge c$ on a bounded domain).
The strongly **convex** case (indicator of the *outside* of a closed surface, e.g. a tail event) is identical with
$H(\boldsymbol y)=-\min_t\phi(t,\boldsymbol y)$, since the band $\{|\phi|\le\eta\}$ is unchanged under $\phi\mapsto-\phi$
(Theorem R8b′, Lean: `Cubature.preintJump_stratified_mse_height_convex`).

**Theorem R8b.** Under the hypotheses of R2 (with $L>0$), joint measurability of $\phi$, (H), an equal-mass partition into
cells of diameter $\le h$ with $2Lh\le H_{\max}$, and independent $\boldsymbol X_i\sim\mu(\cdot\mid A_i)$:
$$\mathbb E\Big(\hat Q-\int P\,d\mu\Big)^2\le\frac2n\Big(L_\theta^2h^2+B^2K\Big(\frac{96\rho^2(Lh)^2}{\alpha}\log\frac{H_{\max}}{2Lh}
+\frac{48\rho^2(Lh)^2}{\alpha}\Big)\Big)=O\Big(\frac{h^2\log(1/h)}{n}\Big).$$
**Theorem R8c (unit cube).** With the JPI design ($h=1/m$, $n=m^s$): MSE $=O(m^{-s-2}\log m)=O(n^{-1-2/s}\log n)$ with the
explicit constant obtained from R8b.
**Theorem R8d (GKS Example 1, $\phi=y_0-t^2$ on $\mathbb R\times[0,1)^{s+1}$, $t\sim N(0,1)$, $m\ge2$).** Here $H(\boldsymbol y)=y_0$
has the uniform density on $[0,1)$ ($K=1$, $H_{\max}=1$), $\alpha=2$, $L=1$, $B=1$, $L_\theta=0$, $\rho=1/\sqrt{2\pi}$, and
$$\mathbb E\big(\hat Q_n-I\big)^2\le\frac{48\log(m/2)+24}{\pi\,m^{s+3}}=O\big(n^{-1-\frac2{s+1}}\log n\big),\qquad n=m^{s+1}.$$

*Proof of R8b.* As in R7a, $\mathrm{MSE}\le n^{-1}\int g^2d\mu$ with $g=L_\theta h+B\,m_\eta$, $\eta=Lh$,
$m_\eta(\boldsymbol y)=\nu\{t:|\phi(t,\boldsymbol y)|\le\eta\}$, and $g^2\le2L_\theta^2h^2+2B^2m_\eta^2$. Pointwise
$m_\eta(\boldsymbol y)^2\le\Psi(H(\boldsymbol y))$ for the profile
$\Psi(r)=\frac{96\rho^2\eta^2}{\alpha r}\mathbf 1_{[2\eta,H_{\max}]}(r)+\frac{16\rho^2\eta}{\alpha}\mathbf 1_{[-\eta,2\eta)}(r)$
(`Aux.band_sq_le_profile`): if $H\ge2\eta$ use R8a and $\nu\le\rho\lambda$; if $-\eta\le H<2\eta$ use R1; if $H<-\eta$ the band is
empty. Then $\int\Psi(H)\,d\mu=\int\Psi\,d(\mu\circ H^{-1})\le K\int\Psi\,dr
=K\big(\frac{96\rho^2\eta^2}\alpha\log\frac{H_{\max}}{2\eta}+\frac{48\rho^2\eta^2}\alpha\big)$ (`Aux.bandProfile_integral`). $\square$
R8c is R8b on the cells of §6.1; R8d checks (H) for $H=y_0$ (the push-forward of the uniform measure on the cube by
$\boldsymbol y\mapsto y_0$ is the uniform measure on $[0,1)$, `Aux.cube_eval_zero_le`).

*Comparison and significance.* For Lipschitz integrands on $[0,1)^s$, jittered sampling has MSE $O(n^{-1-2/s})$ (Haber),
and this is the optimal randomized order for the Lipschitz class (Bakhvalov). In the **monotone** case the preintegrated
function is smooth (GKLS18, He19), so jittered sampling gives $O(n^{-1-2/s})$; R8 shows that **without monotonicity** the
same estimator gives $O(n^{-1-2/s}\log n)$. Thus, for stratified RQMC, failure of monotonicity costs at most a logarithmic
factor, although the preintegrated function is not even $C^1$ (GKS) — a provable answer to the GKS statement that "there is
at present no theoretical support for our use of QMC integration" in the non-monotone case, for this class of RQMC rules.
The logarithm is genuine in this analysis (in one dimension, $\sum_{i\le m}h^2/(ih)=h\log m$). Compared with HW15
Theorem 3.5 (scrambled nets on the original discontinuous integrand, exponent $1+\frac1{2d-1}$, same type of boundary
regularity assumption), the exponent here is $1+\frac2{d-1}$; the estimators differ, so the class is **incomparable**.
He19's CQMC rate $O(n^{-1+\epsilon})$ for scrambled nets requires monotonicity (Assumption 3) — R8 does not.

*Optimality (remark, not formalized).* Up to the logarithm, the rate of R8 cannot be improved by **any** randomized
algorithm that uses $n$ values of $P$: take $\phi(t,\boldsymbol y)=1+\tfrac12y_0+\sum_k\varepsilon_k\psi_k(\boldsymbol y)-t^2$ with
$\varepsilon_k\in\{\pm1\}$ and $\psi_k$ disjointly supported $\tfrac14$-Lipschitz tents of height $\asymp n^{-1/s}$ on $\asymp2n$
cells. These satisfy all hypotheses of R8 (strongly concave in $t$; $H=1+\tfrac12y_0+\sum\varepsilon_k\psi_k$ has
$\partial_{y_0}H\ge\tfrac14$, hence a bounded density), and $P=p\circ H$ with $p(r)=\gamma_1(|t|<\sqrt r)$ bi-Lipschitz on
$[\tfrac12,2]$; Bakhvalov's averaging argument over the random signs then gives MSE $\gtrsim n\cdot(n^{-1/s}\cdot n^{-1})^2
=n^{-1-2/s}$ for every randomized rule. Together with R11 this gives the picture: on the non-monotone class, deterministic
rules are limited to $n^{-1/s}$, while the optimal randomized order $n^{-1-2/s}$ (MSE) is attained, up to $\log n$, by
preintegration + jittered sampling.

---

## 10. R9 — third-order tangencies: cube-root regularity
(Lean: `Cubature.volume_band_le_of_deriv_strong`, `Cubature.preintJump_holder_third`, `Cubature.gks_example4_two_sided`)

**Theorem R9a.** Let $\phi$ be differentiable with derivative $\phi'$, and let $\phi'$ be $\alpha$-strongly convex or
$\alpha$-strongly concave ($|\phi^{(3)}|\ge\alpha$ for $C^3$ functions). Then for every $\eta\ge0$ and $\beta>0$,
$\lambda\{|\phi|\le\eta\}\le4\sqrt{\beta/\alpha}+6\eta/\beta$, and hence $\lambda\{|\phi|\le\eta\}\le10(\eta/\alpha)^{1/3}$.

*Proof.* Convex case (the concave case follows by $\phi\mapsto-\phi$). Let $S=\{|\phi|\le\eta\}$ and $I=\{\phi'\le\beta\}$, a
convex set (sublevel set of a convex function). Every $t\in S$ lies in $A_1=\{|\phi'|\le\beta\}$, or in $A_4=S\cap\{\phi'<-\beta\}$,
or has $\phi'(t)>\beta$, in which case $t\notin I$ and, $I$ being an interval, $t$ lies to the left of $I$ (set $A_2$) or to
its right (set $A_3$). By R1 applied to $\phi'$, $\lambda(A_1)\le4\sqrt{\beta/\alpha}$. If $x<y$ are both in $A_2$ (or both in
$A_3$), every $z\in[x,y]$ has $\phi'(z)\ge\beta$ (otherwise $z\in I$ would lie between two points left (right) of $I$); if
both are in $A_4$, $[x,y]\subset\{\phi'<-\beta\}$ by convexity of that set. The mean value theorem gives
$\beta(y-x)\le|\phi(y)-\phi(x)|\le2\eta$, so $A_2,A_3,A_4$ have diameter $\le2\eta/\beta$. Summing,
$\lambda(S)\le4\sqrt{\beta/\alpha}+6\eta/\beta$. For $\eta>0$ take $\beta=\alpha c^2$, $c=(\eta/\alpha)^{1/3}$: the bound is
$4c+6c=10c$; for $\eta=0$ let $\beta\downarrow0$. $\square$

**Theorem R9b.** Under the hypotheses of R2 with NM$_\alpha$ replaced by "$t\mapsto\phi(t,\boldsymbol y)$ has derivative
$\phi'(t,\boldsymbol y)$ which is $\alpha$-strongly convex or concave in $t$":
$|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+10\rho B(L\delta/\alpha)^{1/3}$ (proof: `Aux.jump_diff_le` and R9a).

**Theorem R9c (GKS Example 4).** For $\phi(t,y)=t^3-y$, $\theta=1$, $\nu=N(0,1)$, $0\le y\le1$:
$$\frac{e^{-1/2}}{\sqrt{2\pi}}\,y^{1/3}\le P(0)-P(y)\le\frac{10}{\sqrt{2\pi}}\Big(\frac y6\Big)^{1/3}.$$
*Proof.* $\{t:t^3>y\}=(y^{1/3},\infty)$, so $P(0)-P(y)=\gamma_1((0,y^{1/3}])\ge y^{1/3}e^{-1/2}/\sqrt{2\pi}$; the upper bound is R9b
with $\phi'=3t^2$ ($6$-strongly convex), $L=1$, $B=1$, $L_\theta=0$. $\square$

*Comparison.* GKS Example 4 (and their Remark 1) exhibits a cube-root singularity when $\partial_t\phi$ and $\partial_t^2\phi$
vanish simultaneously, a case outside their Theorem 1. R9 gives the matching global upper bound for the whole class and
shows (R9c) that the exponent $1/3$ is exact. R9a is an explicit-constant instance of the order-3 van der Corput sublevel
lemma. **Class: incomparable** (with GKS); the sublevel estimate itself is classical.

---

## 11. R10 — quadric surfaces in any direction: the Lipschitz rate up to a logarithm
(Lean: `Cubature.quadric_stratified_mse`, `Cubature.quadric_theta_stratified_mse`)

R8 needs a uniform Lipschitz bound in $\boldsymbol y$, which fails for quadrics preintegrated along a general direction
($\partial_{\boldsymbol y}\phi$ grows linearly in $t$). For quadrics we can instead use the exact root formula of R6.

**Theorem R10.** Let $\phi(z)=\langle z,Az\rangle+\langle b,z\rangle+c$ on a real inner-product space $E$ (any $A$: ellipsoids,
hyperboloids, saddles, inside or outside), $f=\operatorname{ind}\phi$, $u$ a unit vector with $a=\langle u,Au\rangle\ne0$,
$P_u(x)=\Pr(\phi(x+Zu)>0)$, $Z\sim N(0,1)$, and $\beta,D$ as in R6 ($\{D=0\}$ is the shadow boundary). Let $\mu$ be a
probability measure on $E$ partitioned (up to a null set) into $n$ equal-mass cells of diameter $\le h$ contained in the
ball of radius $R$, with $D\le D_{\max}$ on the cells and **$D(\boldsymbol X)$, $\boldsymbol X\sim\mu$, having density at most
$K$**. Put $\eta=8\|A\|(2\|A\|R+\|b\|)h$ and assume $2\eta\le D_{\max}$. Then the stratified estimator
$\frac1n\sum_iP_u(\boldsymbol X_i)$ with independent $\boldsymbol X_i\sim\mu(\cdot\mid C_i)$ satisfies
$$\mathbb E\Big(\hat Q-\int P_u\,d\mu\Big)^2\le\frac1n\cdot\frac{(2\|A\|h)^2+K\big(\eta^2\log\frac{D_{\max}}{2\eta}+3\eta^2\big)}{\pi a^2}
=O\Big(\frac{h^2\log(1/h)}{n}\Big).$$
For a jittered grid in $s$ coordinates of $u^\perp$ ($h\asymp n^{-1/s}$) this is $O(n^{-1-2/s}\log n)$.

*Proof.* (i) As in R6, $P_u(x)=\gamma_1(m-w,m+w)$ (or $1-\gamma_1[m-w,m+w]$ for $a>0$) with $m=-\beta/(2a)$,
$w=\sqrt{D_+}/(2|a|)$, so (`Aux.quad_mass_diff_sqrt`)
$|P_u(x)-P_u(x')|\le\big(|\beta(x)-\beta(x')|+|\sqrt{D(x)_+}-\sqrt{D(x')_+}|\big)/(\sqrt{2\pi}|a|)$.
(ii) For $x,x'$ in one cell, $|\beta(x)-\beta(x')|\le2\|A\|h$ and $|D(x)-D(x')|\le\eta$ (`Aux.quadric_incr`, as in R6b).
(iii) If $|D-D'|\le\eta$ then $|\sqrt{D_+}-\sqrt{D'_+}|^2\le\Psi(D)$ with $\Psi(r)=\frac{\eta^2}r\mathbf 1_{[2\eta,D_{\max}]}(r)+\eta\,\mathbf 1_{[-\eta,2\eta)}(r)$
(`Aux.sqrt_incr_sq_le`: for $D\ge2\eta$, $|\sqrt D-\sqrt{D'}|=|D-D'|/(\sqrt D+\sqrt{D'})\le\eta/\sqrt D$; for $D<-\eta$ both are $0$;
otherwise $\le\sqrt\eta$). (iv) The weighted stratification lemma (R7a(i)) with
$g(x)=(2\|A\|h+\sqrt{\Psi(D(x))})/(\sqrt{2\pi}|a|)$ gives $\mathrm{MSE}\le n^{-1}\int g^2d\mu\le n^{-1}\frac{(2\|A\|h)^2+\int\Psi(D)\,d\mu}{\pi a^2}$, and
$\int\Psi(D)\,d\mu=\int\Psi\,d(\mu\circ D^{-1})\le K\int\Psi=K(\eta^2\log\frac{D_{\max}}{2\eta}+3\eta^2)$. $\square$

**Theorem R10′ (Lipschitz amplitude; Lean: `Cubature.quadric_theta_stratified_mse`).** Same setting for
$f=\theta\cdot\operatorname{ind}\phi$ with $|\theta|\le B$ and $|\theta(z)-\theta(z')|\le L_\theta\|z-z'\|$:
$$\mathbb E\Big(\hat Q-\int P_u\,d\mu\Big)^2\le\frac1n\Big(2(L_\theta h)^2+\frac{2B^2\big((2\|A\|h)^2+K(\eta^2\log\frac{D_{\max}}{2\eta}+3\eta^2)\big)}{\pi a^2}\Big).$$
*Proof.* $|P_u(x)-P_u(x')|\le\int|\theta(x+tu)-\theta(x'+tu)|\,d\gamma_1(t)+B\,\gamma_1(S_x\,\triangle\,S_{x'})$ with
$S_x=\{t:\phi(x+tu)>0\}$ (`Aux.indicator_integral_diff`); the first term is $\le L_\theta\|x-x'\|$, and
$\gamma_1(S_x\triangle S_{x'})\le(|\Delta\beta|+|\Delta\sqrt{D_+}|)/(\sqrt{2\pi}|a|)$ (`Aux.quad_symmDiff`: for $a<0$ both sets are
intervals with endpoints $m\pm w$; for $a>0$ pass to complements, the zero sets being null). Then proceed as in R10 with
$g=L_\theta h+B(2\|A\|h+\sqrt{\Psi(D)})/(\sqrt{2\pi}|a|)$. $\square$
A one-surface jump function of $\mathcal F_d$ is $h_-+(h_+-h_-)\operatorname{ind}\phi$; with $\theta=h_+-h_-$ and the smooth part
$h_-$ (Lipschitz, contributing only an $L$-term), R10′ covers every such integrand whose jump surface is a quadric, on
bounded domains.

*Example (the hypotheses of R10 are satisfiable).* $E=\mathbb R^2$, $\phi(z)=z_1-z_0^2$ (GKS Example 1: $A=\operatorname{diag}(-1,0)$,
$b=e_1$, $c=0$), $u=e_0$ ($a=-1$), $\mu$ the uniform law on the segment $\{0\}\times[0,1)$ of $u^\perp$, cells
$\{0\}\times[k/m,(k+1)/m)$. Then $\beta\equiv0$, $D(x)=4x_1$ has the uniform density $1/4$ on $[0,4)$ ($K=1$), $R=1$, $h=1/m$,
$D_{\max}=4$, $\|A\|=\|b\|=1$, $\eta=24/m$, and $2\eta\le D_{\max}$ for $m\ge12$; R10 gives MSE $\le\frac1m\cdot
\frac{4/m^2+(24/m)^2(\log\frac m{12}+3)}{\pi}=O(m^{-3}\log m)$, the same order as R8d.

*Low intrinsic dimension.* R10 holds for an arbitrary probability measure $\mu$. If the quadric only involves the
projection of $z$ onto a $k$-dimensional subspace $V$ ($A=\Pi_VA\Pi_V$, $b\in V$) and $u\in V$, then $P_u(x)$ depends only on
$\Pi_{V\cap u^\perp}x$, so the $(d-1)$-dimensional integral equals an integral against the $(k-1)$-dimensional law of that
projection; applying R10 to this law, with cells of diameter $h\asymp n^{-1/(k-1)}$, gives MSE $O(n^{-1-2/(k-1)}\log n)$ —
governed by the intrinsic dimension of the jump surface (between $2$ and $d$ in the hard track), not by $d$. Exploiting it in
practice requires stratifying in coordinates adapted to $V$ (not formalized).

*Relevance and comparison.* This covers the hard-track jump instances of the task (anisotropic quadrics in unknown rotated
subspaces): preintegrating along **any** direction off the null cone — in particular along a random direction, or the
leading eigenvector of $A$'s symmetric part — and stratifying the remaining coordinates on a bounded domain gives the
Lipschitz rate up to a log (see §14 for the Gaussian case), with an explicit constant $\propto1/(\pi\langle u,Au\rangle^2)$ (so directions with large $|\langle u,Au\rangle|$ are
preferable, cf. target D). The density condition on $D$ is the transversality of the shadow boundary; it holds whenever
$D$ has no critical point on its zero level inside the region (informal). GKS treat quadrics only through their
singularities; He19's rates need monotonicity, which quadrics never have along non-null directions. **Class: incomparable.**

---

## 12. R11 — deterministic rules cannot beat $n^{-1/s}$ on the non-monotone class
(Lean: `Cubature.deterministic_lower_bound`)

**Theorem R11.** Let $s\ge1$, $y_1,\dots,y_n\in\mathbb R^s$ and $m\ge1$ with $2n\le m^s$. There is $\phi$ with
$t\mapsto\phi(t,\boldsymbol y)$ $2$-strongly concave and $|\phi(t,\boldsymbol y)-\phi(t,\boldsymbol y')|\le\|\boldsymbol y-\boldsymbol y'\|_\infty$
(the class of R2 with $\alpha=2$, $L=1$, $\theta=1$) such that $\phi(t,y_i)\le0$ for all $t$ and $i$ (so
$f=\operatorname{ind}\phi$ vanishes on every line through a node), $P(y_i)=0$ for all $i$, and
$$\int_{[0,1)^s}P\ \ge\ \frac{\sqrt2\,e^{-1/2}}{\sqrt{2\pi}\;2^{s+2}\,m}.$$

*Proof.* Let $\delta=1/(2m)$ and let $E$ be the set of grid cells of side $1/m$ containing no node; since each node lies
in at most one cell, $|E|\ge m^s-n\ge m^s/2$. Let $U$ be the union of the open sup-norm balls $B(c_k,\delta)$ (the open
cells) for $k\in E$, and $\phi(t,\boldsymbol y)=\delta\operatorname{dist}(\boldsymbol y,U^c)-t^2$: $2$-strongly concave in $t$,
$\delta$-Lipschitz in $\boldsymbol y$ ($\delta\le\frac12$). Nodes lie in $U^c$, so $\phi(t,y_i)=-t^2\le0$ and $P(y_i)=0$. On the
inner box $\bar B(c_k,\delta/2)$, $k\in E$, $\operatorname{dist}(\boldsymbol y,U^c)\ge\delta/2$, hence $\{t:\phi>0\}\supset(-\delta/\sqrt2,\delta/\sqrt2)$
and $P\ge\sqrt2\,\delta\,e^{-1/2}/\sqrt{2\pi}$; the inner boxes are disjoint, lie in $[0,1)^s$ and have volume $\delta^s$.
Summing, $\int P\ge|E|\sqrt2\,\delta^{s+1}e^{-1/2}/\sqrt{2\pi}$. $\square$

*Consequence.* Any deterministic algorithm — any quadrature weights, nodes chosen adaptively or not — that uses $n$
values of the preintegrated function, or $n$ point values of the $(s+1)$-dimensional integrand (their projections are $n$
points $y_i$), sees only zeros on this $\phi$, while the integral is $\ge c_s/m\asymp c_s'n^{-1/s}$; since the function with
$P\equiv0$ (e.g. $\phi=-t^2-1$) is also in the class and produces the same observations, the worst-case error is
$\ge c_s'n^{-1/s}/2$. For $s\ge3$ this is **slower than Monte Carlo**, whereas the randomized
estimators of R5 and R8 achieve RMSE $n^{-1/2-1/(2s)}$ and $n^{-1/2-1/s}\sqrt{\log n}$ (under their hypotheses).

*Comparison.* The fooling-function technique is classical (Bakhvalov); what is new is that the obstruction lives inside
the non-monotone preintegration class itself. It turns the GKS remark "there is at present no theoretical support for our
use of QMC" into a theorem: **no deterministic rule can have a worst-case guarantee better than $n^{-1/s}$ there**, so
randomization (R5–R10) is necessary for rates beating Monte Carlo in dimension $\ge4$. **Class: incomparable.**

---

## 13. Further remarks, conjectures and black-box implementation (informal, not formalized)

**Remark C1 (order-$k$ tangency, $k\ge4$; informal).** If $|\partial_t^k\phi|\ge\alpha$ for all $(t,\boldsymbol y)$ (and the
uniform Lipschitz bounds hold), then $|P(\boldsymbol y)-P(\boldsymbol y')|\le L_\theta\delta+C_k\rho B(L\delta/\alpha)^{1/k}$, by induction
on $k$ exactly as in R9a (or by the general van der Corput sublevel lemma of Carbery–Christ–Wright). Formalized for
$k=2$ (R2, R2′) and $k=3$ (R9).

**Conjecture C2 (piecewise and convex-concave mixtures).** R8 extends to the piecewise setting of R2′ (several
non-degenerate tangencies per line, strongly convex pieces with $H=\min_t\phi$), with the transversality condition imposed
on each local extremum value; we formalized only the strongly concave case.

**Conjecture C3.** Scrambled $(t,s)$-net variance for $P$ under NM$_\alpha$ is $O(n^{-1-1/s+\epsilon})$ as well.

**Black-box preintegration (open problem 3).** With point queries only and $\theta=1$, $\nu=N(0,1)$, under NM$_\alpha$ the
positive set on each line is an interval (concave case) or the complement of one. A grid of step $h$ on $[-T,T]$ plus
bisection at the two ends to tolerance $\varepsilon$ costs $2T/h+2\log_2(h/\varepsilon)$ queries per line and has error
$\le\rho(h+2\varepsilon)+2(1-\Phi(T))$ (an interval shorter than $h$ may be missed, but its mass is $\le\rho h$). Combined with
R5c this gives a fully black-box estimator; we did not formalize this accounting.

---

## 14. Which instances of the task satisfy the hypotheses (informal)

* **Hard track, closed jump surfaces (anisotropic quadrics in a hidden rotated subspace, possibly in the tail).** For the
  indicator part, R6 (½-Hölder modulus) and R10 (stratified rate $O(n^{-1-2/s}\log n)$) hold for **every** preintegration
  direction $u$ with $\langle u,Au\rangle\ne0$ — this excludes only the null cone, so a randomly chosen direction works almost
  surely, and no knowledge of the rotation is needed. R10 (like R7a, R8b) is formal for partitions into cells of diameter
  $\le h$ inside a ball (bounded support: the unit cube, a truncated domain). For the Gaussian measure on $\mathbb R^s$ itself,
  equal-mass cells near radius $r$ necessarily have diameter $\asymp n^{-1/s}e^{r^2/(2s)}$; the weighted stratification lemma
  (R7a(i)) still applies cell by cell, and since $\mathbb E\,e^{|\boldsymbol Y|^2/s}<\infty$ for $s>2$ we expect the same rates
  with larger constants, but this variable-diameter version is **not** formalized (with product quantile grids the outermost
  slabs dominate and only the $O(n^{-1-1/s})$ rate of §6.3 follows). The density condition on the discriminant $D$ holds
  when $D$ has no critical point on $\{D=0\}$, the generic case.
* **Hard track, saddle kinks / intersecting kink surfaces.** For kinks the preintegrated function is $C^{1,1/2}$ under the
  hypotheses of R3, and in any case locally Lipschitz, so jittered sampling already attains $O(n^{-1-2/s})$ (Haber). Several
  intersecting surfaces ($K\le3$): the sublevel argument applies surface by surface (the indicator changes only where one of
  the $\phi_i$ changes sign), giving the sum of the individual bounds; this extension is not formalized.
* **Smooth full-dimensional terms** ($f=g+\theta\operatorname{ind}\phi$): the preintegrated smooth part is smooth and only adds
  a Lipschitz contribution ($L_\theta$-terms in R5–R8).
* **Option track.** Discretely monitored payoffs are monotone in suitable variables (e.g. the first Brownian increment), so
  the classical monotone theory (GKLS18, He19) applies; for (double) barriers each monitoring constraint is monotone in that
  variable, the admissible set on a line is an interval, and R2′ (two monotone pieces) gives an explicit Hölder bound.
* **Black-box aspect.** All our estimators assume that the one-dimensional preintegral is computed exactly (as in
  GKS/GKLS18/He19); with point queries only it is approximated by root finding along the line (§13), at a cost of
  $O(T/h+\log(1/\varepsilon))$ queries per line.

---

## 15. Answers to the appendix's open problems (partial)

* **Open problem 1 (regularity and RQMC rates without monotonicity):** answered for the non-degenerate non-monotone class
  NM$_\alpha$ (R2: sharp ½-Hölder; R3: $C^{1,1/2}$ for kinks; R4: exponents optimal) and for all quadric surfaces in all
  non-null directions (R6); and RQMC (stratified) rates $O(n^{-1-1/(d-1)})$ with explicit constants (R5), improved to
  $O(n^{-1-3/(2(d-1))})$ under a Minkowski-content-type tube condition (R7) and to the Lipschitz rate up to a log,
  $O(n^{-1-2/(d-1)}\log n)$, under transversality of the shadow boundary (R8); both verified on GKS Example 1. For
  quadric jump surfaces (the hard-track instances) in any orientation and any non-null preintegration direction, the same
  near-Lipschitz rate holds without any uniformity assumption (R10); third-order tangencies give exactly ⅓-Hölder
  regularity (R9).
* **Open problem 4 (partial; deterministic vs randomized):** on the non-monotone class every deterministic algorithm,
  adaptive or not, has worst-case error $\gtrsim n^{-1/s}$ (R11), while preintegration + jittered sampling attains RMSE
  $n^{-1/2-1/(2s)}$ (R5) and $n^{-1/2-1/s}\sqrt{\log n}$ under shadow-boundary transversality (R8): randomization gains a
  polynomial factor (for the preintegrated problem with exact one-dimensional preintegration).
* **Target B:** for discontinuous integrands of our class, a randomized rule with MSE exponent $1+\frac1{d-1}$ (R5c, R5e),
  $1+\frac3{2(d-1)}$ under the tube condition (R7) and $1+\frac2{d-1}$ (up to a log) under shadow-boundary
  transversality (R8), larger than the $1+\frac1{2d-1}$ of HW15 for scrambled nets;
  different estimator, so incomparable.

---

## 16. Formalization map (Lean 4, Mathlib; `formal/Theory/`)

| Result | Lean theorem(s) (namespace `Cubature`) | Notes on the formal statement |
|---|---|---|
| R1 | `volume_sublevel_le_of_strongConvexOn` | `StrongConvexOn univ α φ` (Mathlib), Lebesgue `volume`, bound `ENNReal.ofReal (4 * √(η / α))` |
| R2 | `preintJump_holder_half_of_density_le`, `preintJump_holder_half` | `ν S ≤ ofReal ρ * volume S` for all `S`; `gaussianReal 0 1` is $N(0,1)$; `Y` any `PseudoMetricSpace` |
| R2′ | `preintJump_holder_half_piecewise` | pieces `J y i` convex with `⋃ i, J y i = univ`; on each piece strong convexity, strong concavity, or strong monotonicity `α * \|t - t'\| ≤ \|φ t y - φ t' y\|` |
| R3 | `preintKink_second_difference_of_density_le`, `preintKink_second_difference` | `E` any `SeminormedAddCommGroup`; integrability of `φ(·,y)` assumed |
| R4 | `sublevel_bound_sharp`, `gks_parabola_two_sided` | exact equality of the band measure; two-sided bound for $0\le y\le1$ |
| R5 | `stratified_sampling_mse`, `preintJump_stratified_mse`, `preintJump_jittered_cube_mse`, `jittered_design_exists`, `preintJump_integral_eq` | independence = Mathlib `iIndepFun`; law of `X i` = `n • μ.restrict (A i)`; MSE written as `∫ ω, (Q ω - I)^2 ∂P`; `cubeCell` in `Defs.lean`; existence via `Measure.pi` |
| R6 | `quadric_preint_holder`, `quadric_preint_holder_local` | `A : E →L[ℝ] E`, inner product space `E`; `φ`, `β`, `D` given with defining equations |
| R7 | `preintJump_stratified_mse_tube`, `preintJump_jittered_cube_mse_tube`, `gks_example_tube_condition`, `gks_example_jittered_mse` | tube condition as `(ν.prod μ) {p \| \|φ p.1 p.2\| ≤ L * h} ≤ ofReal (K * (L * h))`; joint measurability `Measurable (Function.uncurry φ)` |
| R8 | `volume_band_le_of_strongConcaveOn_max`, `preintJump_stratified_mse_height`, `preintJump_stratified_mse_height_convex`, `preintJump_jittered_cube_mse_height`, `gks_example_jittered_mse_log` | height `H` with `φ t y ≤ H y` and `∃ u, φ u y = H y`; transversality as `μ.map H ≤ ENNReal.ofReal K • volume` |
| R9 | `volume_band_le_of_deriv_strong`, `preintJump_holder_third`, `gks_example4_two_sided` | derivative via `HasDerivAt (φ · y) (φ' t y) t`; exponent `(1 : ℝ) / 3` as `Real.rpow` |
| R10, R10′ | `quadric_stratified_mse`, `quadric_theta_stratified_mse` | cells `C i` with `‖x‖ ≤ R`, `dist ≤ h`, `D ≤ Dmax`; density condition `μ.map D ≤ ENNReal.ofReal K • volume` |
| R11 | `deterministic_lower_bound` | existential statement: `∃ φ`, strongly concave in `t`, 1-Lipschitz in `y`, `φ t (y i) ≤ 0`, `P (y i) = 0`, integral lower bound over `[0,1)^s` |

Definitions used in the statements (`Theory/Defs.lean`): `preintJump ν φ θ y = ∫ t, {s | 0 < φ s y}.indicator (θ · y) t ∂ν`,
`preintKink ν φ y = ∫ t, max (φ t y) 0 ∂ν`, `cubeCell s m k = ∏ⱼ [k j / m, (k j + 1) / m)`. All proofs use only the axioms
`propext`, `Quot.sound`, `Classical.choice` (checked by `lake comparator`).

---

## References

* N. S. Bakhvalov, On approximate computation of integrals, Vestnik MGU 4 (1959) 3–18.
* A. Carbery, M. Christ, J. Wright, Multidimensional van der Corput and sublevel set estimates, J. Amer. Math. Soc. 12 (1999) 981–1015.
* A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Preintegration is not smoothing when monotonicity fails, arXiv:2112.11621 (2021/2022).
* A. Griewank, F. Y. Kuo, H. Leövey, I. H. Sloan, High dimensional integration of kinks and jumps — smoothing by preintegration, J. Comput. Appl. Math. 344 (2018) 259–274.
* S. Haber, A modified Monte-Carlo quadrature, Math. Comp. 20 (1966) 361–368.
* Z. He, X. Wang, On the convergence rate of randomized quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 53 (2015) 2488–2503, doi:10.1137/15M1007963.
* Z. He, Quasi-Monte Carlo for discontinuous integrands with singularities along the boundary of the unit cube, arXiv:1702.03361 (Proposition 2.3).
* Z. He, On the error rate of conditional quasi-Monte Carlo for discontinuous functions, SIAM J. Numer. Anal. 57 (2019) 854–874, arXiv:1708.09512.
* E. M. Stein, Harmonic Analysis, Princeton University Press, 1993 (Ch. VIII, van der Corput lemma).
