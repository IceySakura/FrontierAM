# Theory: structure-exploiting adaptive cubature for piecewise smooth Gaussian integrals

This document describes the algorithm that `run.sh` executes (§2), states and proves the results R1–R5 about the
per-line localisation/integration problem that the algorithm solves repeatedly (§3; all five are formalised in Lean and
accepted by `lake comparator`), records further statements about the algorithm that are **not** formalised (§4), compares
everything with the human literature (§5) and connects the theory with the experiments (§6).

## 1. Problem and notation

$I(f)=\mathbb E f(Z)$, $Z\sim N(0,I_d)$, $\gamma_d$ the standard Gaussian measure on $\mathbb R^d$, $\gamma=\gamma_1$,
$\Phi$ and $\phi$ the standard normal distribution function and density. $f$ belongs to the class
$\mathcal F_d(K,r,M,c)$ of `THEORY_APPENDIX.md` §2: $f=h_{\sigma(z)}(z)$ with $\sigma(z)=(\operatorname{sign}\varphi_i(z))_{i\le k}$.
Information: point values only, $N=2^{14}$ of them per instance, adaptively chosen; output an estimate $\hat I$.
For a line $\ell=\{y+s v: s\in\mathbb R\}$ ($|v|=1$, $y\perp v$) the restriction $s\mapsto f(y+sv)$ is piecewise smooth
with breakpoints at the zeros of $s\mapsto\varphi_i(y+sv)$; since $\langle v,Z\rangle\sim N(0,1)$ is independent of the
projection of $Z$ on $v^\perp$, $I(f)=\mathbb E_Y\int_{\mathbb R}f(Y+sv)\,\gamma(\mathrm ds)$ (preintegration along $v$).
The one-dimensional problem "integrate a piecewise function on a line against $\gamma$ from point values when the
breakpoints are unknown" is the subject of §3.

## 2. The algorithm (as implemented in `solution.py`, `linequad.py`, `quadric.py`)

All random choices are derived from the `seed` of the init message (numpy `default_rng`); every scrambled Sobol' set
below is an independent Owen-type scrambling (`scipy.stats.qmc.Sobol(scramble=True)`), mapped to $\mathbb R^d$ by
$\Phi^{-1}$ coordinatewise (clipped to $[10^{-15},1-10^{-15}]$) and then by an orthogonal matrix. Every query point is
clipped to $|z_i|\le 39.9$ (never active in practice). The total number of queries never exceeds $N$ (hard check before
every query batch); all work is batched (one query message per stage of all lines), so the wall time is a few seconds.

**Step 0 (pilot).** $n_0=\min(1024,N/16)$ scrambled Sobol' points $z_i$; values $f_i$; $\bar f$ their mean. Stein moments
$$b=\tfrac1{n_0}\sum_i (f_i-\bar f)\,z_i\ \ (\approx\mathbb E[\nabla f]),\qquad
H=\tfrac1{n_0}\sum_i (f_i-\bar f)\,(z_iz_i^\top-I)\ \ (\approx\mathbb E[\nabla^2 f]),$$
(Gaussian integration by parts; the distributional derivatives include the surface terms of kinks and jumps).
Let $\zeta$ be the fraction of pilot values that are exactly $0$.

**Step 1 (mode selection).**
* *Mode W (window preintegration)* if $\zeta\ge0.2$, at least 8 pilot values are nonzero and all nonzero pilot values are
  equal (a payoff that is constant where it does not vanish: cash-or-nothing digitals).
* *Mode H (cross-fitted hinge control variates)* if $\zeta\ge0.05$ otherwise (option payoffs that vanish on a large set).
* *Mode Q (learned jump surface)* if $\zeta<0.05$ (hard-track type), with the routing described there (mode H for
  functions without jumps, mode S when a jump surface cannot be learned).

**Mode G.** Forward-difference gradients (step $10^{-5}$) at $k_g=\min(16,\lfloor N_{\rm rem}/(8(d+1))\rfloor)$ pilot
points (spread over the ranking of $|f_i-\operatorname{median}|$ among nonzero values), cost $k_g(d+1)$;
$C=\frac1{k_g}\sum g_jg_j^\top$; rotation $R$ = eigenvectors of $C$ by decreasing eigenvalue, the null space of $C$
ordered by $|$eigenvalues$|$ of $H$ restricted to it. All remaining queries form one scrambled Sobol' set
$z=R\,\Phi^{-1}(u)$; $\hat I$ is its plain average. (This is the reference method of Xiao and Wang, with a smaller
gradient budget.)

**Mode S (Stein-rotated RQMC, fallback).** Rotation = eigenvectors of $bb^\top+HH^\top$ (decreasing); all remaining queries
as one scrambled Sobol' set; $\hat I$ = average of pilot and main points weighted by their numbers.

**Mode Q (jump across a single closed quadric).** The hard track uses quadric surfaces; for a jump family
$f=g+J(z)\mathbf 1\{q(z)<0\}$ with $q(z)=z^\top Mz+m^\top z+m_0$ and $g,J$ smooth:
1. *Jump detection and an interior point.* Lines $y+sv$ ($v$ uniform on the sphere, $y\perp v$; half of the $y$ are the
   pilot points with the largest $|f-\operatorname{median}|$, half fresh $N(0,I)$ points) are evaluated on the grid
   $s\in\{-5,-4.5,\dots,5\}$; every grid interval whose increment exceeds $\max(8\,\mathrm{median}|\Delta f|,10^{-6}\,
   \overline{|f|})$ is bisected (branch test "$f(m)$ is closer to the left or to the right end value"); a crossing is a
   *jump* if the final increment exceeds $0.02\,\mathrm{sd}(f_i)$ (a genuine jump survives bisection, a steep smooth part
   does not). Ten lines without $\ge3$ jumps classify $f$ as a kink function (mode H), unless the pilot shows a rare region
   ($\max|f_i-\operatorname{median}|>20\times$ the 98% quantile), in which case up to 40 lines are scanned. The midpoint of
   a line's two crossings is an interior point $c$ of the (convex) region.
2. *Boundary points by rays.* $p=\binom{d+1}2+d+1$ coefficients, $n_b=\lceil1.25p\rceil+12$ ($d\le16$) or
   $\lceil1.08p\rceil+12$ ($d\le40$) boundary points. Rays $c+r\theta$, $\theta$ uniform, $r\in[0,7]$: one query at $r=7$,
   rays without a jump between $r=0$ and $r=7$ are dropped, then 22 ($d\le16$) or 16 bisection steps (precision
   $1.7\cdot10^{-6}$ or $1.1\cdot10^{-4}$) — every ray from an interior point crosses a convex surface exactly once, so a
   boundary point costs 17–23 queries. The jump $J_j$ (inside minus outside) at the boundary point is read off the final
   bracket. If no interior point is available, lines are used as in step 1 until $n_b$ crossings are found (at most
   $0.6N$ queries). For $d>40$ ($p\ge 2145$) the surface is not learned: mode S if jumps were detected, otherwise
   mode H.
3. *Exact fit.* $(M,m,m_0)$ is the right singular vector of the smallest singular value of the row-normalised feature
   matrix $[z_iz_j\ (i\le j),\,z_i,\,1]$ of the boundary points; points with geometric residual $|q(z)|/|\nabla q(z)|$
   above $\max(\tau,3\,\mathrm{median})$ are removed and the fit repeated (at most 4 rounds); accepted only if all geometric
   residuals are $<\tau$ ($\tau=2.5\times$ the bisection precision); otherwise mode S.
   Orientation: $q(c)<0$. Eigenvalues of
   $M$ below $10^{-6}$ of the largest are set to $0$ (fit noise in the null directions of a cylinder).
4. *Exact moments.* $p_0=\gamma_d(q<0)$ and $g_0=\mathbb E[Z\mathbf 1\{q(Z)<0\}]$. Diagonalising $M$,
   $q(Z)\overset d=\sum_i\lambda_i(X_i+\delta_i)^2+\sigma\xi+c$ (generalised chi-square) and
   $p_0=\frac12-\frac1\pi\int_0^\infty\frac{\operatorname{Im}\varphi_q(u)}u\,du$ (Gil-Pelaez), evaluated by the trapezoidal
   rule in $t=\log u$ (step $0.004$, exponentially convergent; for $\le3$ active eigenvalues a conditional Sobol' rule with
   $2^{20}$ points, analytic in the most curved direction, is used instead). $g_0=\nabla_\mu\gamma_d(q(\cdot+\mu)<0)|_{\mu=0}$
   (Gaussian integration by parts), by central differences with step $10^{-4}$. Both are cross-checked against $2^{17}$
   free Monte Carlo evaluations of the *fitted* surface (no queries); on a mismatch ($>6$ standard errors) the control
   variate is not used.
5. *Estimator.* Jump model $\hat J(z)=a+c^\top z$ by least squares on the boundary values $J_j$; $h=\mathbf 1\{\hat q<0\}$
   ("inside", the side of $c$). Candidates:
   * plain: $\bar f$;
   * Jin, Jout: $\overline{f-h\hat J}+a\,p_0+c^\top g_0$ and $\overline{f+(1-h)\hat J}-\big(a(1-p_0)-c^\top g_0\big)$ (the jump
     model subtracted inside, respectively added outside; exact means of the subtracted terms);
   * reg: regression control variates $X=(h-p_0,\ z\,h-g_0)$, $\overline{f-X^\top\hat\beta}$;
   * Jsmall+reg: the jump model on the side of smaller Gaussian mass followed by the regression.

   *Large remaining budget* ($n\ge8000$, i.e. $d\le16$): the $n$ queries form $R=4$ independently scrambled Sobol' sets;
   every candidate is computed on each set (regression coefficients pooled over the sets) and the candidate with the
   smallest spread across the sets is returned as the mean over the sets (spreads multiplied by $1,1.05,1.05,1.1,1.15$
   to prefer simpler candidates). *Small remaining budget* ($d=32$, $n\approx3300$ after the ray search): one scrambled
   Sobol' set and the Jsmall+reg candidate (regression only if at least 200 points fall on the smaller side).
   For a fixed fitted $(\hat q,\hat J)$ the plain, Jin and Jout candidates are exactly unbiased (the subtracted term has
   a known mean) and the regression versions have $O(1/n)$ bias. If $\hat q=q$ and $J$ is affine, $f-h\hat J=g$ is smooth
   and the jump disappears from the RQMC integrand; if $J$ is far from affine (sample H3-d32: the jump model *increases*
   the spread from $0.016$ to $0.037$), the regression fits the jump over the whole region instead (spread $0.0012$).
   Safeguard: a value farther from $\bar f$ than $8\times$ the standard error of $\bar f$ over the sets (resp. $6$
   Monte Carlo standard errors for one set) plus $6\,\mathrm{sd}(f)/\sqrt n$ is rejected in favour of $\bar f$.

**Mode H (cross-fitted hinge control variates; kink functions and non-digital option payoffs).** Used for hard-track
functions without jumps (any $d$) and for option payoffs that vanish on a large set but are not constant ($\zeta\ge0.05$).
GPCA rotation as in mode G (16 gradients); $U$ = the top $\min(d,16)$ ($d\le32$) or $12$ ($d=64$) GPCA directions,
$x=U^\top z$. The remaining queries form two independently scrambled Sobol' sets $A,B$. On $A$ the hinge model
$f\approx P_2(x)+\max(q(x),0)$ ($P_2,q$ quadratic in $x$) is fitted by alternating least squares on the active set
$\{q>0\}$ (initial active set: points above the quadratic regression; at most 30 sweeps; the best sweep is kept), and on
$B$ the regression control variates $\max(q,0)-e_0$, $z\max(q,0)-\nabla e_0$ and the centred quadratic monomials of $x$
are used with coefficients estimated on $A$; then the roles are exchanged and the two estimates averaged. Each half-
estimate is unbiased because its control variates and coefficients do not depend on the half it is evaluated on. The
exact means: $e_0=\mathbb E\max(q(Z),0)=\frac12(\mathbb E q+\mathbb E|q|)$ with
$\mathbb E|q|=\frac2\pi\int_0^\infty(1-\operatorname{Re}\varphi_q(u))u^{-2}du$ (same quadrature as in mode Q),
$\nabla e_0$ by central differences in the shift. If the hinge model does not fit (rms on a half $>0.3\,$sd), or the
correction is implausible (as in mode Q), the plain average of both halves is returned, which is mode G with two
scrambles.

**Mode W (window preintegration along a learned direction).** Payoffs that vanish outside one interval of every line in a
monotone direction (calls, digitals, knock-outs along the "level" direction of the path).
1. Direction $v_0=b/|b|$. Lines $y+sv$ with $y=B\,\Phi^{-1}(u)$, $u$ a scrambled Sobol' point in $[0,1]^{d-1}$, $B$ an
   orthonormal basis of $v^\perp$ ordered by $|$eigenvalues$|$ of $H$ restricted to $v^\perp$.
2. Per line (`WindowIntegrator`): grid of 25 points on $[-6,6]$; the window $[s_{lo},s_{hi}]$ between the first and the last
   nonzero grid value; both ends bisected on the predicate $f\ne0$ to width $2\cdot10^{-5}$ (open ends: $\pm7.5$). If all
   nonzero observed values are equal to $c$ the line integral is $c\,(\Phi(s_{hi})-\Phi(s_{lo}))$ (exact up to the
   bisection error), otherwise a Gauss rule for $\gamma$ restricted to the window (Stieltjes procedure on a 96-point
   Gauss–Legendre discretisation; $4+2.5\cdot$length nodes, at most 20). Lines with several windows go to the general
   integrator `LineIntegrator` (ENO-type detection of kinks/jumps by 4th differences, bisection/branch-intersection
   refinement, Gauss rules on the pieces).
3. Stages of $\max(32,d/2+16)$ and $d+16$ lines, then the rest of the budget. After each of the first two stages the
   direction is replaced by the normal of the crossing surface: total least squares (smallest principal axis) of all
   crossing points $y+s_{lo}v$ collected so far if there are at least $d+8$, otherwise the ridge regression
   $s_{lo}\approx\alpha+\beta^\top B^\top y$ and $v\leftarrow (v-B\beta)/|v-B\beta|$ (exact for a hyperplane).
4. $\hat I=\sum_s w_s\hat I_s/\sum_s w_s$ over the stages, $w_s=L_s/\widehat{\operatorname{Var}}_s$(line integrals).

**Error bound.** No `error_bound` is reported (it is optional and does not enter the score).

## 3. Results R1–R5: locating and integrating jumps on a line

All five results are formalised in Lean 4 (Mathlib v4.35.0) and accepted by `python3 /workspace/tools/lean_check.py`
(`LEAN-CHECK: 5/5 proved results accepted`): definitions `theory/formal/Theory/Defs.lean`, statements
`Theory/Statements.lean`, proofs `Theory/Lemmas.lean` (for an arbitrary atomless probability measure on $\mathbb R$) and
`Theory/Proofs.lean` (instantiation to the Gaussian measure).

### 3.0 Setting

**Why this subproblem.** The cubature algorithm integrates $f$ along lines (preintegration along a unit direction
$\theta$). For fixed orthogonal component $z_\perp$, the component of $Z\sim N(0,I_d)$ along $\theta$ is standard normal, so
the inner integral is $\int_{\mathbb R} f(z_\perp+t\theta)\,\gamma(\mathrm dt)$ with $\gamma=N(0,1)$. Along the line the
non-smooth surfaces $\{\varphi_i=0\}$ are crossed at unknown points, which must be located from function values only
(open problems 3 and 4 of `THEORY_APPENDIX.md`). The results below give the exact complexity of the simplest jump
instances of this per-line problem: one crossing (a surface that is monotone along the line) and two crossings (a closed
surface crossed twice, e.g. the closed jump surfaces of the hard track).

**Measure and integral.** $\gamma$ is the standard Gaussian measure on $\mathbb R$ (Lean: `gaussianReal 0 1`),
$\Phi(t)=\gamma((-\infty,t])$ its distribution function, and $I(f)=\int_{\mathbb R} f\,\mathrm d\gamma$
(Lean: `∫ t, f t ∂gaussianReal 0 1`, the Bochner integral). We use only that $\gamma$ is a probability measure without
atoms: $\Phi$ is continuous, nondecreasing, $\Phi(-\infty)=0$, $\Phi(+\infty)=1$. (The Lean proofs in
`Theory/Lemmas.lean` are written for an arbitrary probability measure $\mu$ on $\mathbb R$ with $\mu(\{x\})=0$ for all $x$,
and `Theory/Proofs.lean` instantiates $\mu=\gamma$; so every result holds verbatim for any atomless probability measure.)

**Function classes** (Lean: `Cubature.StepClass`, `Cubature.IntervalClass` in `Theory/Defs.lean`).

* One crossing:
  $J=\{0,\ 1\}\ \cup\ \{f:\mathbb R\to\mathbb R \;:\; \exists s\in\mathbb R,\ f(t)=0\ (t<s),\ f(t)=1\ (t>s)\}$.
  The value $f(s)$ on the surface is an arbitrary real number (as in $\mathcal F_d$, where $f$ may take any finite
  value on $\{\varphi=0\}$). The constants $0,1$ are the lines that do not cross the surface ($s=\pm\infty$).
  For $f\in J$ with jump $s$: $I(f)=\gamma((s,\infty))=1-\Phi(s)$; $I(0)=0$, $I(1)=1$.
* Two crossings:
  $P=\{f \;:\; \exists a\le b \text{ real},\ f(t)=1\ (a<t<b),\ f(t)=0\ (t<a \text{ or } t>b)\}$,
  values $f(a),f(b)$ arbitrary. With $a=b$ this contains $f\equiv0$ (the line misses the closed surface).
  $I(f)=\gamma((a,b))=\Phi(b)-\Phi(a)$.

**Algorithms** (Lean: `Cubature.AdaptiveAlg`, `AdaptiveAlg.info`, `Cubature.NonadaptiveAlg`, `NonadaptiveAlg.info`).

* A deterministic **adaptive** algorithm with $n$ values is $A=(\psi_0,\psi_1,\dots,\psi_{n-1},\phi)$ with
  $\psi_k:\mathbb R^k\to\mathbb R$ (arbitrary maps) and $\phi:\mathbb R^n\to\mathbb R$ (arbitrary). On $f$ it observes
  $y_{k+1}=f(\psi_k(y_1,\dots,y_k))$, $k=0,\dots,n-1$, and returns $A(f)=\phi(y_1,\dots,y_n)$.
  In Lean, `node k : (Fin k → ℝ) → ℝ`, `output : (Fin n → ℝ) → ℝ`, the observed vector is defined by recursion
  (`observe f (k+1) = Fin.snoc (observe f k) (f (node k (observe f k)))`), and the sanity lemma
  `AdaptiveAlg.info_apply` (proved in `Defs.lean`) states $y_{i}=f(\psi_{i}(y_{<i}))$ for every index.
* A deterministic **nonadaptive** algorithm: fixed nodes $x_1,\dots,x_n$ and an arbitrary (nonlinear) $\phi$,
  $A(f)=\phi(f(x_1),\dots,f(x_n))$.
* A **randomized** algorithm: a probability space $(\Omega,\Sigma,P)$ and a family $(A_\omega)_{\omega\in\Omega}$ of
  deterministic adaptive algorithms with $n$ values such that $\omega\mapsto A_\omega(f)$ is measurable for every $f\in J$
  (the usual requirement that the output be a random variable).
* Error of $A$ on $f$: $e(A,f)=|A(f)-I(f)|$; worst case $e(A,F)=\sup_{f\in F}e(A,f)$; randomized error
  $\sup_{f}\mathbb E_\omega|A_\omega(f)-I(f)|$ (Lean: lower Lebesgue integral `∫⁻ ω, ENNReal.ofReal |…| ∂P` with values in
  $[0,\infty]$; it is the ordinary expectation since the integrand is measurable).

**Summary.**

| class | adaptive (deterministic) | randomized | nonadaptive |
| --- | --- | --- | --- |
| $J$ (one crossing) | exactly $2^{-(n+1)}$ (R1, R2) | $\ge \frac{1}{2(2^{n+1}-1)}>2^{-(n+2)}$ (R5) | exactly $\frac1{2(n+1)}$ (R3) |
| $P$ (two crossings) | $\ge\frac1{2(n+1)}$ (R4) | — | $\le\frac1{n+1}$ (R4) |

Consequences: for one crossing, adaption is exponentially better than nonadaption, and randomization cannot improve on
bisection by a factor $2$ or more; locating the crossing to accuracy $\varepsilon$ in quantile units (equivalently,
integrating to accuracy $\varepsilon$) needs exactly $\max(0,\lceil\log_2(1/\varepsilon)\rceil-1)$ adaptive values and
$\lceil 1/(2\varepsilon)\rceil-1$ nonadaptive values (for $\varepsilon\le\frac12$; an immediate corollary of the two `IsLeast`
statements, not separately formalized). For two crossings, adaption gains at most a factor 2 in the worst case:
a pair of crossings can hide between two queries.

### 3.1 Common tools

**Lemma A (one value reveals one bit about the level).** For $f\in J$ put $\lambda(f)=1-I(f)\in[0,1]$
($\lambda=\Phi(s)$ for a jump at $s$, $\lambda(1)=0$, $\lambda(0)=1$). For every $x\in\mathbb R$:
if $f(x)=0$ then $\Phi(x)\le\lambda(f)$; if $f(x)\ne0$ then $\lambda(f)\le\Phi(x)$.

*Proof.* $f=0$: $\lambda=1\ge\Phi(x)$ and the second case does not occur. $f=1$: the first case does not occur and
$\lambda=0\le\Phi(x)$. Jump at $s$: if $f(x)=0$ then $x\le s$ (for $x>s$, $f(x)=1$), so $\Phi(x)\le\Phi(s)=\lambda$; if
$f(x)\ne0$ then $x\ge s$ (for $x<s$, $f(x)=0$), so $\lambda=\Phi(s)\le\Phi(x)$. Here $I(f)=1-\Phi(s)$ because $f$ equals
$\mathbf 1_{(s,\infty)}$ except at $s$ and $\gamma(\{s\})=0$. $\square$ (Lean: `stepClass_query`, `integral_step_eq`.)

**Monotone 0/1 steps.** Let $\mathcal M=\{0,1\}\cup\{\mathbf 1_{(x,\infty)},\mathbf 1_{[x,\infty)}:x\in\mathbb R\}\subset J$
(Lean: `StepFam`). Its members are nondecreasing and $\{0,1\}$-valued, and
$I(\mathbf 1_{(x,\infty)})=I(\mathbf 1_{[x,\infty)})=\gamma((x,\infty))$ (no atoms). For $h\le g$ in $\mathcal M$ (pointwise)
the *uncertainty region* is $U(g,h)=\{g\ne h\}=\{g=1,h=0\}$, an interval, and $I(g)-I(h)=\gamma(U(g,h))$.
If $g(x)\ne h(x)$ then $g(x)=1$, $h(x)=0$ and, by monotonicity,
$$h\le \mathbf 1_{(x,\infty)}\le \mathbf 1_{[x,\infty)}\le g. \tag{S}$$

**Two-point argument.** If $g,h$ have the same information under $A$, then $A(g)=A(h)=:o$ and
$|o-I(g)|+|o-I(h)|\ge I(g)-I(h)$, so one of them has error $\ge (I(g)-I(h))/2$. (Lean: `two_point`.)

### 3.2 R1 — adaptive lower bound on $J$

**Theorem R1.** For every $n\ge0$ and every deterministic adaptive algorithm $A$ using $n$ function values there is
$f\in J$ with $|A(f)-I(f)|\ge 2^{-(n+1)}$.

Lean: `Cubature.adaptive_lower_bound_step` —
`∀ (n : ℕ) (A : AdaptiveAlg n), ∃ f ∈ StepClass, 1 / 2 ^ (n + 1) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1|`.

*Proof (adversary).* Claim: for every $k\ge0$ there are $h\le g$ in $\mathcal M$ with $I(g)-I(h)\ge2^{-k}$ such that every
$f$ with $h\le f\le g$ produces the same first $k$ observed values as $g$. For $k=0$ take $g=1$, $h=0$. Given the pair
for $k$, let $x$ be the $(k+1)$-st node of $A$ on $g$; by the claim it is also the $(k+1)$-st node on every $f\in[h,g]$.
If $g(x)=h(x)$, keep the pair: each $f\in[h,g]$ has $f(x)=g(x)$. Otherwise $g(x)=1$, $h(x)=0$, and by (S) the pairs
$L=(g,\mathbf 1_{[x,\infty)})$ and $R=(\mathbf 1_{(x,\infty)},h)$ are ordered and lie inside $[h,g]$. With
$m=\gamma((x,\infty))$ their gaps are $I(g)-m$ and $m-I(h)$, which add up to $I(g)-I(h)\ge2^{-k}$; take the one with gap
$\ge2^{-(k+1)}$. In $L$, every $f$ with $\mathbf 1_{[x,\infty)}\le f\le g$ has $f(x)=1=g(x)$. In $R$, every $f$ with
$h\le f\le\mathbf 1_{(x,\infty)}$ has $f(x)=0=\mathbf 1_{(x,\infty)}(x)$, and $\mathbf 1_{(x,\infty)}\in[h,g]$ has the
same first $k$ values as $g$, so its $(k+1)$-st node is also $x$. Thus the first $k+1$ values agree on the new pair.
For $k=n$, $g$ and $h$ have the same information and $I(g)-I(h)\ge2^{-n}$; by the two-point argument one of
$g,h\in\mathcal M\subset J$ has error $\ge2^{-(n+1)}$. $\square$ (Lean: `adversary`, `adaptive_lower_bound_general`.)

**Comparison.** Sikorski (1982, *Bisection is optimal*) proves that for zero finding of continuous functions with a sign
change on $[a,b]$ the minimal worst-case error of adaptive methods using $n$ function values is $(b-a)/2^{n+1}$, attained
by bisection. R1 + R2 are the same statement for the integration of one-jump step functions: in quantile coordinates
$u=\Phi(t)$ the problem becomes locating the jump level $\lambda(f)\in[0,1]$, and $I(f)=1-\lambda(f)$. Plaskota and
Wasilkowski (2005) study integration of piecewise $C^r$ functions with one unknown singular point on an interval and
show that adaption recovers the rate $n^{-r}$ of smooth functions while nonadaptive methods are not better than $n^{-1}$;
on piecewise-constant integrands (our $J$, any $r$) the adaptive rate is exponential, and R1/R2 give the exact constant.
Class: **equal** (a reproduction of classical bisection optimality, transferred to integration under $\gamma$ with exact
constants; no stronger claim).

### 3.3 R2 — bisection in Gaussian quantiles; exact minimal adaptive error

**Theorem R2.** For every $n\ge0$ there is a deterministic adaptive algorithm $A$ with $n$ values such that
$|A(f)-I(f)|\le2^{-(n+1)}$ for every $f\in J$. Consequently (with R1)
$$\min_{A\ \text{adaptive},\,n\text{ values}}\ \sup_{f\in J}|A(f)-I(f)| \;=\; 2^{-(n+1)},$$
and the minimum is attained.

Lean: `Cubature.bisection_upper_bound_step` —
`∀ n, ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass, |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / 2 ^ (n + 1)`;
`Cubature.minimal_adaptive_error_step` —
`IsLeast {e | ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass, |…| ≤ e} (1 / 2 ^ (n + 1))`.

*Algorithm.* Keep an interval of levels $[\ell,h]\subseteq[0,1]$, initially $[0,1]$. At each step let $m=(\ell+h)/2$,
query $x=\Phi^{-1}(m)$; if $f(x)=0$ set $\ell=m$, otherwise set $h=m$. After $n$ steps output $1-(\ell+h)/2$.

*Proof.* $\Phi$ is continuous: it is right-continuous, and its left limit at $x$ is $\gamma((-\infty,x))=\Phi(x)$ because
$\gamma(\{x\})=0$. As $\Phi\to0$ at $-\infty$ and $\to1$ at $+\infty$, the intermediate value theorem gives for each
$m\in(0,1)$ a point $x$ with $\Phi(x)=m$ (Lean: `continuous_cdf_of_nullSingleton`, `cdf_quantile`). By induction on $k$:
$0\le\ell$, $h\le1$, $h-\ell=2^{-k}$ and $\ell\le\lambda(f)\le h$. Indeed $m\in(\ell,h)\subseteq(0,1)$ so $\Phi(x)=m$, and
by Lemma A, $f(x)=0$ gives $m\le\lambda(f)$ while $f(x)\ne0$ gives $\lambda(f)\le m$. Finally
$|1-\tfrac{\ell+h}2-I(f)|=|\lambda(f)-\tfrac{\ell+h}2|\le\tfrac{h-\ell}2=2^{-(n+1)}$. The minimal-error statement combines
this with R1: any $e$ achieved by some $A$ satisfies $e\ge2^{-(n+1)}$. $\square$ (Lean: `bisect`, `bisectionAlg`,
`bisect_invariant`, `bisection_upper_bound_general`.) The algorithm is robust: arbitrary values on the surface do not
break it.

**Comparison.** As for R1 (Sikorski 1982; Plaskota–Wasilkowski 2005). Class: **equal**.

### 3.4 R3 — nonadaptive algorithms on $J$: exact minimal error $\frac1{2(n+1)}$

**Theorem R3.** (a) For every nonadaptive algorithm with $n$ nodes and any output map there is $f\in J$ with
$|A(f)-I(f)|\ge\frac1{2(n+1)}$. (b) The nonadaptive algorithm with nodes $x_i=\Phi^{-1}\!\big(\tfrac{i}{n+1}\big)$,
$i=1,\dots,n$, and output $1-\frac{k+1/2}{n+1}$, $k=\#\{i: f(x_i)=0\}$, has error $\le\frac1{2(n+1)}$ for every $f\in J$.
Hence $\min_{A\ \text{nonadaptive}}\sup_{f\in J}|A(f)-I(f)|=\frac1{2(n+1)}$, versus $2^{-(n+1)}$ for adaptive algorithms.

Lean: `Cubature.nonadaptive_lower_bound_step`, `Cubature.nonadaptive_upper_bound_step` (stated existentially: there
is a nonadaptive algorithm with error $\le\frac1{2(n+1)}$; the proof uses the algorithm above, `quantileRule`),
`Cubature.minimal_nonadaptive_error_step` (`IsLeast … (1 / (2 * (n + 1)))`).

*Gap lemma (pigeonhole).* Let $S\subset\mathbb R$ be finite, $h\le g$ in $\mathcal M$, and suppose at most $m$ points of
$S$ lie in $U(g,h)$. Then there are $h'\le g'$ in $\mathcal M$ with $g'=h'$ on $S$ and
$I(g')-I(h')\ge\frac{I(g)-I(h)}{m+1}$.

*Proof* by induction on $m$. If no point of $S$ lies in $U(g,h)$, take $(g,h)$. Otherwise let $x$ be the smallest point
of $S$ in $U(g,h)$; then $g(x)=1$, $h(x)=0$. The pair $(g,\mathbf 1_{[x,\infty)})$ agrees on $S$: for $t\ge x$,
$g(t)\ge g(x)=1$; for $t<x$, $t\notin U$ (minimality), so $g(t)=h(t)\le h(x)=0$. If its gap $I(g)-\gamma((x,\infty))$ is
$\ge D/(m+1)$, with $D=I(g)-I(h)$, we are done. Otherwise the pair $(\mathbf 1_{(x,\infty)},h)$ has gap $>Dm/(m+1)$, and
its uncertainty region $U\cap(x,\infty)$ contains at most $m-1$ points of $S$ ($x$ itself is excluded, and no point of
$S\cap U$ lies below $x$); the induction hypothesis gives a pair agreeing on $S$ with gap $\ge\frac{Dm/(m+1)}{m}=\frac
D{m+1}$. $\square$ (Lean: `gap_pairs`.)

*Proof of (a).* Apply the gap lemma to $S=\{x_1,\dots,x_n\}$, $g=1$, $h=0$ ($D=1$, $m\le n$): $g',h'\in\mathcal M\subset J$
agree at all nodes, so they have the same information and output, and $I(g')-I(h')\ge\frac1{n+1}$; by the two-point
argument one of them has error $\ge\frac1{2(n+1)}$. (Classically: the $n$ nodes cut $\mathbb R$ into at most $n+1$ gaps of
total mass $1$; a jump anywhere inside the heaviest gap is invisible.) $\square$

*Proof of (b).* Let $\lambda=\lambda(f)$ and $Z=\{i:f(x_i)=0\}$, $k=|Z|$. By Lemma A and $\Phi(x_i)=\frac i{n+1}$:
$i\in Z\Rightarrow\frac i{n+1}\le\lambda$ and $i\notin Z\Rightarrow\lambda\le\frac i{n+1}$. If $k\ge1$, $Z$ has $k$ distinct
elements of $\{1,\dots,n\}$, so $\max Z\ge k$ and $\lambda\ge\frac k{n+1}$ (trivial if $k=0$). If $k<n$, the $n-k$ indices
outside $Z$ cannot all exceed $k+1$, so some $i\notin Z$ has $i\le k+1$ and $\lambda\le\frac{k+1}{n+1}$ (trivial if $k=n$).
Hence $|1-\frac{k+1/2}{n+1}-I(f)|=|\lambda-\frac{k+1/2}{n+1}|\le\frac1{2(n+1)}$. $\square$
(Lean: `nonadaptive_upper_bound_general`; the Lean nodes are indexed by `i : Fin n` with level `(i+1)/(n+1)`.)

**Comparison.** Plaskota–Wasilkowski (2005): nonadaptive algorithms cannot integrate functions with an unknown
discontinuity faster than order $n^{-1}$, whereas adaptive ones can; R3 gives the exact constant on the step class and,
with R1/R2, an exponential gap ($2^{-(n+1)}$ vs $\frac1{2(n+1)}$). Novak (1996) surveys when adaption helps: not for linear
problems on convex symmetric classes (Bakhvalov; Gal–Micchelli), possibly a lot on non-convex classes such as $J$. The
value $\frac1{2(n+1)}$ is also Kiefer's (1957) optimal error for integrating monotone functions with values in $[0,1]$
(after the quantile transform, the $\{0,1\}$-valued members of $J$ are monotone); R3 shows that the step subclass is
already extremal for nonadaptive methods, while R1/R2 show that on this subclass adaption is exponentially better. Class:
**equal**.

### 3.5 R4 — two crossings: adaption helps at most by a factor 2

**Theorem R4.** (a) For every deterministic adaptive algorithm $A$ with $n$ values there is $f\in P$ with
$|A(f)-I(f)|\ge\frac1{2(n+1)}$. (b) The nonadaptive algorithm with nodes $x_i=\Phi^{-1}(\frac i{n+1})$ and output
$\frac k{n+1}$, $k=\#\{i:f(x_i)=1\}$, has error $\le\frac1{n+1}$ for every $f\in P$.

Lean: `Cubature.adaptive_lower_bound_interval`, `Cubature.nonadaptive_upper_bound_interval` (stated existentially:
there is a nonadaptive algorithm with error $\le\frac1{n+1}$; the proof uses `countingRule`).

*Proof of (a).* Run $A$ on $f\equiv0\in P$; it queries fixed points $x_1,\dots,x_n$ and outputs $o$. If
$|o|\ge\frac1{2(n+1)}$, $f=0$ works. Otherwise $o<\frac1{2(n+1)}$. The gap lemma ($S=\{x_i\}$, $g=1$, $h=0$) gives an interval
$U=U(g',h')$ with no $x_i$ and $\gamma(U)=I(g')-I(h')\ge\frac1{n+1}>o+\frac1{2(n+1)}$. Since $P$ contains only bounded
intervals, truncate: $\gamma(U\cap(-R,R))\uparrow\gamma(U)$ as $R\to\infty$ (continuity from below), so for some $R$ the
bounded interval $B=U\cap(-R,R)$ has $\gamma(B)>o+\frac1{2(n+1)}$. Then $f=\mathbf 1_B\in P$ ($a=\inf B$, $b=\sup B$; or $f=0$
if $B=\emptyset$), $f$ vanishes at all $x_i$, so by induction over the queries $A$ observes the same zeros on $f$ as on
$0$ and outputs $o$; its error is $\ge\gamma(B)-o>\frac1{2(n+1)}$. $\square$ (Lean: `adaptive_lower_bound_interval_general`,
`gap_ordConnected`, `integral_gap_eq`, `exists_bounded_part`, `indicator_mem_intervalClass`.)

*Proof of (b).* For $f\in P$ with endpoints $a\le b$ put $\alpha=\Phi(a)$, $\beta=\Phi(b)$, so $I(f)=\beta-\alpha$. A node
with value $1$ lies in $[a,b]$ (outside, $f=0$), so its level is in $[\alpha,\beta]$; a node with another value lies
outside $(a,b)$, so its level is $\le\alpha$ or $\ge\beta$. Upper bound: if $\beta-\alpha>\frac{k+1}{n+1}$, the open interval
$((n+1)\alpha,(n+1)\beta)\subseteq(0,n+1)$ has length $>k+1$ and contains the $k+1$ integers
$\lfloor(n+1)\alpha\rfloor+1,\dots,\lfloor(n+1)\alpha\rfloor+k+1\in\{1,\dots,n\}$, whose nodes have levels in $(\alpha,\beta)$ and
hence value $1$: more than $k$ ones, a contradiction. Lower bound: if $k\ge2$, the smallest and largest indices with value
$1$ differ by at least $k-1$ and both levels lie in $[\alpha,\beta]$, so $\beta-\alpha\ge\frac{k-1}{n+1}$ (trivial for $k\le1$).
Hence $|\frac k{n+1}-I(f)|\le\frac1{n+1}$. $\square$ (Lean: `nonadaptive_upper_bound_interval_general`,
`intervalClass_integral`.)

**Remark (sharpness of R4; not formalised).** (i) For $n\ge1$ the minimal nonadaptive error on $P$ is *exactly*
$\frac1{n+1}$: if the levels of the nodes cut $[0,1]$ into gaps $g_0,\dots,g_n$ ($\sum g_i=1$), the two largest gaps
$g_i,g_l$ ($i<l$) satisfy $g_i+g_l\ge\frac2{n+1}$, and the pulses whose left end ranges over gap $i$ and whose right end
ranges over gap $l$ all produce the same information (value 1 exactly at the nodes between the two gaps) while their
masses fill an interval of length $g_i+g_l$; so every output errs by at least $\frac{g_i+g_l}2\ge\frac1{n+1}$ on one of
them, and the quantile rule of R4(b) attains it. (ii) The minimal *adaptive* error lies in $[\frac1{2(n+1)},\frac1{n+1}]$ by
R4; a dynamic program over the adversary game (all-zero answers keep a pulse hidden in a gap; a hit reduces the rest to
two independent bisections) gives $\frac12,\frac13$ for $n=1,2$ (equal to the nonadaptive values) and at most $0.2375$
and $0.1646$ for $n=3,4$ (below $\frac14,\frac15$): adaption helps on two crossings, but by less than a factor 2 — in
contrast with the exponential gain on one crossing (R1–R3).

**Comparison.** Plaskota, Wasilkowski and Zhao (2008) show for $L_p$ approximation of piecewise smooth functions that
with one unknown singularity adaption helps greatly, but with two or more singularities adaption does not help in the
worst-case setting (two nearby singularities can hide between nodes); Plaskota–Wasilkowski (2005) treat integration with
one singularity. R4 is the integration version of this worst-case phenomenon for two crossings, under $\gamma$, with
explicit constants: adaptive $\ge\frac1{2(n+1)}$, nonadaptive $\le\frac1{n+1}$. Class: **equal**.

### 3.6 R5 — randomized lower bound on $J$

**Theorem R5.** Let $(\Omega,\Sigma,P)$ be a probability space and $(A_\omega)_{\omega\in\Omega}$ deterministic adaptive
algorithms with $n$ values such that $\omega\mapsto A_\omega(f)$ is measurable for each $f\in J$. Then there is $f\in J$ with
$$\mathbb E_\omega\,|A_\omega(f)-I(f)|\ \ge\ \frac1{2(2^{n+1}-1)}\ >\ 2^{-(n+2)}.$$
So randomization improves the optimal deterministic error $2^{-(n+1)}$ by less than a factor 2.

Lean: `Cubature.randomized_lower_bound_step` —
`∀ n {Ω} [MeasurableSpace Ω] (P) [IsProbabilityMeasure P] (A : Ω → AdaptiveAlg n), (∀ f ∈ StepClass, Measurable fun ω => (A ω).output ((A ω).info f)) → ∃ f ∈ StepClass, ENNReal.ofReal (1 / (2 * (2 ^ (n + 1) - 1))) ≤ ∫⁻ ω, ENNReal.ofReal |(A ω).output ((A ω).info f) - ∫ t, f t ∂gaussianReal 0 1| ∂P`.

*Proof (Bakhvalov/Yao averaging).* Let $N=2^{n+1}$, $M=N-1$ and $f_j$ ($j=0,\dots,M$): $f_0=1$, $f_M=0$,
$f_j=\mathbf 1_{(\Phi^{-1}(j/M),\infty)}$ for $0<j<M$. They lie in $J$, are $\{0,1\}$-valued, and $I(f_j)=1-j/M$.
(i) For a deterministic adaptive $A$, the information of each $f_j$ is a vector in $\{0,1\}^n$, so the $N$ indices split into
at most $2^n$ classes with equal information, hence equal output $o$. In a class $C$ with $c\ge2$ elements, the smallest and
largest indices $j_-<j_+$ satisfy $j_+-j_-\ge c-1$, so
$\sum_{j\in C}|o-I(f_j)|\ge|o-I(f_{j_-})|+|o-I(f_{j_+})|\ge\frac{j_+-j_-}M\ge\frac{c-1}M$ (also true for $c\le1$). Summing over
the at most $2^n$ classes: $\sum_{j=0}^{M}|A(f_j)-I(f_j)|\ge\frac{N-2^n}{M}=\frac{2^n}{2^{n+1}-1}$.
(ii) For the randomized algorithm, measurability allows exchanging sum and expectation:
$\sum_j\mathbb E|A_\omega(f_j)-I(f_j)|=\mathbb E\sum_j|A_\omega(f_j)-I(f_j)|\ge\frac{2^n}{2^{n+1}-1}$, so some $j$ has
$\mathbb E|A_\omega(f_j)-I(f_j)|\ge\frac1N\cdot\frac{2^n}{2^{n+1}-1}=\frac1{2(2^{n+1}-1)}$. $\square$
(Lean: `testFun`, `fiber_bound`, `sum_error_lower_bound`, `randomized_lower_bound_general`.) For $n=0$ the bound $\frac12$ is
sharp. Part (i) is also an average-case lower bound for deterministic algorithms under the uniform prior on
$\{f_0,\dots,f_M\}$.

**Comparison.** Bakhvalov's technique for randomized lower bounds (averaging over finitely many functions, presented in
Novak 1988, LNM 1349) and Yao's principle; for bisection-type search problems the randomized order $2^{-n}$ is classical
(each $\{0,1\}$-valued observation carries one bit). Our contribution is the explicit constant for $J$, which places the
randomized optimum within a factor 2 of Sikorski-type bisection. Class: **equal**. Caveat: the model has a fixed number $n$
of values for every $\omega$ (no random cardinality).

### 3.7 Scope and caveats

* **What is formalized exactly.** The Lean statements quantify over all deterministic adaptive algorithms (arbitrary node
  maps and output maps, no measurability or linearity assumptions), all nonadaptive algorithms, and all randomized
  algorithms in the sense above; the integral is the Bochner integral against `gaussianReal 0 1`; the classes are exactly
  $J$ and $P$ above (arbitrary values on the surface). Upper bounds are existential statements; the algorithms used in the
  proofs (bisection in quantiles, quantile rules) are defined in `Theory/Lemmas.lean`.
* **Relation to $\mathcal F_d$.** $J$ and $P$ model the restriction of a jump function with constant pieces $0$ and $1$ to a
  line during preintegration. E.g. $f(z)=\mathbf 1\{\langle\theta,z\rangle>s\}$ belongs to $\mathcal F_d(1,\infty,M,c)$ when
  $Mc\ge1$ and $|s|\le M$; the line restrictions in direction $\theta$ are then in $J$. Our lower bounds allow arbitrary jump
  positions $s\in\mathbb R$; restricting $s$ to $|s|\le M$ changes the constants only through the Gaussian tail mass outside
  $[-M,M]$ (not analyzed formally). Smooth non-constant pieces $h_\pm$ (kinks, jumps with varying height, interior kinks with
  nonzero values) are not covered: the lower bounds transfer (subclass), the upper bounds do not.
* **Orientation.** $J$ contains upward steps only (the surface is crossed from $\{\varphi<0\}$, value $0$, to
  $\{\varphi>0\}$, value $1$, as $t$ increases). Downward steps $1-f$ are symmetric (replace $f$ by $1-f$). If the
  orientation is unknown, the class $J\cup(1-J)$ is larger: the lower bounds R1, R3, R5 still hold (subclass), but R2 and
  the upper bound in R3 are only proved for $J$.
* **Citations.** The human results are quoted from memory (the session had no library access); we state them at the
  level of precision we are sure of (main theorems of the cited papers, no theorem numbers where we are not certain).
* **Nothing here is claimed to be stronger than the literature**: R1–R5 reproduce classical phenomena (bisection optimality;
  adaption vs nonadaption for one vs several singularities; Bakhvalov-type randomized bounds) with exact constants for the
  Gaussian measure and machine-checked proofs. They answer the per-line, piecewise-constant special case of open problems 3
  (black-box localization along a line) and 4 (adaptive vs nonadaptive orders with matching lower bounds) of the appendix.


## 4. Statements about the algorithm that are not formalised

These are recorded for the analysis of §2; they are **not** part of the formal results (status `informal` would apply;
they are not listed in the manifest).

**P1 (identifiability of a quadric jump surface; elementary).** Let $S=\{q=0\}$ with $q(z)=z^\top Mz+m^\top z+m_0\not\equiv0$
and let $z_1,\dots,z_n\in S$ be such that the $n\times p$ feature matrix $\Psi=[z_iz_j\ (i\le j),z_i,1]$ has rank $p-1$
($p=\binom{d+1}2+d+1$). Then $(M,m,m_0)$ is determined up to a scalar factor by the points: it spans $\ker\Psi$.
*Proof.* $\Psi c=0$ for the coefficient vector $c$ of $q$, and $\dim\ker\Psi=1$. $\square$ Generic points on a
nondegenerate quadric give rank $p-1$ as soon as $n\ge p-1$; with bisection to width $w$ the points are
$O(w)$-perturbed and the fitted coefficients are $O(w/\sigma_{p-1}(\Psi))$-accurate (standard SVD perturbation).
In the implementation, $w=1.7\cdot10^{-6}$ ($d\le16$) or $1.1\cdot10^{-4}$ ($d=32$) and acceptance requires all
geometric residuals $<2.5\,w$.

**P2 (control variates).** For any fixed measurable $h$ with known mean, $\bar f-\bar h+\mathbb Eh$ is unbiased under
Owen scrambling (each scrambled point is uniform), and its variance is that of the scrambled-net average of $f-h$. In
mode Q, $h=\hat J\,\mathbf 1\{\hat q<0\}$ is fixed before the final sample is drawn (it depends only on the boundary
search), so the Jin/Jout estimators are exactly unbiased (the selection among candidates and the regression variants
add a small selection/estimation bias). If $\hat q=q$ and $J(z)=a+c^\top z$ then $f-h=g$ is smooth, so the jump
does not enter the RQMC error at all; the residual jump is the non-affine part of $J$ (about 3% of $J$ on the sample
H3-d8; exactly 0 on H3t, where $J$ is constant). The decisive property — the expectation of the control variate is known exactly although the
surface is a non-monotone closed quadric — is what makes this work where preintegration is not smoothing
(Gilbert–Kuo–Sloan 2022).

**P3 (cost).** With rays from an interior point a boundary point costs $1+s$ queries ($s$ bisection steps): for $d=8$
($p=45$) about 1300, for $d=16$ ($p=153$) about 4700, for $d=32$ ($p=561$) about 11000 of the 16384 queries. For
$d=64$ ($p=2145$) the full quadric is not affordable and the algorithm falls back to RQMC; learning the subspace of the
quadric first would remove this limitation (conjecture: with the subspace dimension $k$, $O(k^2\log(1/w))$ queries
suffice).

**Conjecture C1.** On the subclass of $\mathcal F_d$ of functions $g+J\mathbf 1\{q<0\}$ with $g,J$ analytic and $q$ a
nondegenerate quadric, the algorithm of mode Q has error $O(e_{\rm RQMC}(g)+e_{\rm RQMC}(J-\Pi_1J)\,+w)$, where $\Pi_1$
is the $L^2(\gamma_d)$ projection on affine functions restricted to $\{q<0\}$, i.e. the jump enters only through its
non-affine part. Not proved.

## 5. Comparison with human results (summary)

| result | strongest comparable human result | class | reason |
| --- | --- | --- | --- |
| R1 | Sikorski 1982 (bisection is optimal for zero finding: minimal worst-case error $(b-a)2^{-(n+1)}$) | equal | same adversary argument and the same constant, transported to integration of one-jump functions under $\gamma$ (any atomless measure) |
| R2 | Sikorski 1982; Plaskota–Wasilkowski 2005 (one singularity: adaption recovers the smooth rate) | equal | bisection in Gaussian quantiles attains $2^{-(n+1)}$; exact minimal adaptive error (IsLeast) |
| R3 | Plaskota–Wasilkowski 2005 (nonadaptive algorithms are not better than order $n^{-1}$ with an unknown discontinuity); Novak 1996; Kiefer 1957 (optimal integration of monotone functions) | equal | exact minimal nonadaptive error $\frac1{2(n+1)}$ on $J$; exponential gap to R1/R2 |
| R4 | Plaskota–Wasilkowski–Zhao 2008 (with several singularities adaption does not help in the worst case) | equal | two crossings: adaptive $\ge\frac1{2(n+1)}$, nonadaptive $\le\frac1{n+1}$ |
| R5 | Bakhvalov's averaging technique (Novak 1988); Yao's principle | equal | randomized lower bound $\frac1{2(2^{n+1}-1)}>2^{-(n+2)}$ on $J$ |

None of R1–R5 is claimed to be stronger than the literature. Their value for this problem: they settle, with exact
constants and machine-checked proofs, the per-line jump case of open problems 3 (black-box localisation along a line)
and 4 (optimal adaptive versus nonadaptive errors with matching lower bounds) of the appendix, and R4 explains why
line-wise adaptive preintegration cannot be uniformly efficient for closed (non-monotone) jump surfaces — the case of
open problem 1 — which is why mode Q learns the surface globally instead of locating it line by line.

## 6. Connection with the experiments

**Which hidden families satisfy which assumptions.**
* H3 and H3t (closed jump surfaces, quadrics, $K=1$): the restriction to a line through the region is in the two-crossing
  situation of R4 (with smooth instead of constant pieces); R4's lower bound applies to line-wise methods, which is why the
  algorithm locates boundary points only to *learn the surface* (P1) — along rays from an interior point, where each ray
  is in the one-crossing situation of R1/R2 and bisection is optimal — and integrates with exact control variates (P2).
  Mode Q applies for $d\le40$; for $d=64$ the fallback is RQMC.
* H1 (saddle/tent kinks on one quadric) and H2 (two intersecting kink surfaces, two regimes inside the kink region):
  mode H. On H1-d8 the hinge model fits (residual sd $0.029$ versus $1.25$); on H2 a single hinge does not fit and the
  safeguard returns the plain GPCA-rotated average. On synthetic tents and saddles ($d=8$–$64$) mode H averages
  $-4.6$ versus $-4.1$ for GPCA-RQMC.
* Calls and down-and-out calls (mode H): the hinge in the top GPCA directions captures most of the kink
  ($+0.3$ on the sample call, $+0.3$ on average on a local-volatility simulator).
* Digital options along the learned level direction: one crossing per line (the class $J$ after the quantile transform,
  with a constant payoff), exactly the setting of R1/R2; mode W integrates each line exactly up to the bisection error
  $c\,\phi\,w/2\le 10^{-5}c$ (R2 with $w=2\cdot10^{-5}$ in place of the quantile width).
* For modes G, H and S the theory above gives no rate; they are RQMC methods with control variates whose means are
  exact, so they are unbiased and the safeguards keep them within a few standard errors of the plain RQMC average.

**Measured (sample instances, single runs, log10 relative error; reference GPCA-RQMC in parentheses).**
H3-d8: $-4.55$ to $-5.51$ ($-3.41$); H3t-d8: $-4.33$ to $-5.33$ ($-2.32$); H3t-d32: $-3.22$ to $-3.62$ ($-1.39$);
H3-d32: $-2.9$ to $-3.8$ ($-3.53$, here the RQMC sample left after the 11000-query boundary search is small and $J$ is
far from affine); H1-d8 (mode H): $-4.16$ to $-4.64$ ($-3.28$). The quadric control variates gain 1–2 orders of magnitude
where the surface is learned, as predicted by P2 (residual standard deviation $0.04$ versus $20$ on H3t-d32). Synthetic
jump families with a varying (affine) jump height at $d=8,16$: $-3.9$ to $-5.4$ versus $-2.1$ to $-3.1$ for RQMC.
Public feedback set (20 cases, one run each): 55.7 for plain Stein-rotated RQMC everywhere; 65.0 for the final
program (category scores: H1 55.7, H2 63.0, H3 46.5, H3t 89.2, call 64.7, digital 93.0, double knock-out 54.9,
down-and-out 50.9). H3 remains the weakest family: at $d=64$ the quadric (2145 coefficients) cannot be learned and the
fallback is RQMC, and the hidden H3 instances have strongly non-affine jump heights, for which the regression variant
of mode Q is needed.

## 7. References

* K. Sikorski, Bisection is optimal, Numer. Math. 40 (1982) 111–117.
* L. Plaskota, G. W. Wasilkowski, Adaption allows efficient integration of functions with unknown singularities,
  Numer. Math. 102 (2005) 123–144.
* L. Plaskota, G. W. Wasilkowski, Y. Zhao, The power of adaption for approximating functions with singularities,
  Math. Comp. 77 (2008) 2309–2338.
* E. Novak, On the power of adaption, J. Complexity 12 (1996) 199–237.
* E. Novak, Deterministic and Stochastic Error Bounds in Numerical Analysis, Lecture Notes in Math. 1349, Springer, 1988.
* J. Kiefer, Optimum sequential search and approximation methods under minimum regularity assumptions, J. Soc. Indust.
  Appl. Math. 5 (1957) 105–136.
* A. C.-C. Yao, Probabilistic computations: toward a unified measure of complexity, FOCS 1977, 222–227.
* A. D. Gilbert, F. Y. Kuo, I. H. Sloan, Preintegration is not smoothing when monotonicity fails, in Advances in
  Modeling and Simulation, Springer (2022).
* Y. Xiao, X. Wang, Enhancing quasi-Monte Carlo simulation by minimizing effective dimension for derivative pricing,
  Comput. Econ. 54 (2019) 343–366.
* A. B. Owen, Scrambled net variance for integrals of smooth functions, Ann. Statist. 25 (1997) 1541–1562.
* J. P. Imhof, Computing the distribution of quadratic forms in normal variables, Biometrika 48 (1961) 419–426.
* J. Gil-Pelaez, Note on the inversion theorem, Biometrika 38 (1951) 481–482.
