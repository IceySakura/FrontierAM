# Black-box Gaussian cubature

## Targets and comparison

**Targets A, B, and C.** R4 gives a global Lipschitz bound after integrating two Gaussian coordinates of a nonmonotone quadratic section and a stratified variance bound of order $n^{-1-2/(d-2)}$. This addresses the lack of a rate guarantee after nonmonotone preintegration on a structured subclass with known sections. R1 gives a deterministic, multiplicative enclosure for an unknown, translated, arbitrarily oriented ellipsoid with a supplied interior anchor. It uses bounded-coordinate membership queries and takes logarithmically many queries in inverse relative precision at fixed dimension. R2 shows a precision cost for nonadaptive algorithms; R3 shows an inverse-rarity cost even for adaptive randomized algorithms when no anchor is supplied. The production solver also uses statistical models not covered by these guarantees.

- **Gilbert, Kuo and Sloan**, [arXiv:2112.11621](https://arxiv.org/abs/2112.11621), Theorem 1. For $\phi\in C^2(\mathbb R^d)$, let $t=\phi(y^*)$. If $\partial_1\phi(y^*)=0$, $\partial^2_1\phi(y^*)\ne 0$, and $\nabla\phi(y^*)\ne0$, then preintegrating $\mathbf 1_{\{\phi>t\}}$ in $y_1$ exhibits a square-root singularity at $y^*_{-1}$ along each line there not orthogonal to $\nabla_{-1}\phi(y^*)$. In their conclusion the authors explain that an ordinary coordinate preintegration can leave a function outside the dominating mixed smoothness space of order one; they also single out the need to find the endpoints of potentially multiple intervals. **Incomparable.** R1 neither contradicts their obstruction nor bounds the convergence of a scrambled net on the Cartesian preintegral. It resolves a restricted case of the appendix's open problem 1 by changing the sections: rays from an interior point have one exit, and the quadratic structure couples all sections, so only finitely many directions need to be queried.
- **Rademacher and Vempala**, [arXiv:cs/0608054](https://arxiv.org/abs/cs/0608054), Theorem 2 (“volume,” numbers in the arXiv version). For a convex body given by a membership oracle with $B_d\subseteq K\subseteq O(d^8)B_d$, for some absolute $c>0$ a randomized algorithm returning $V\in[(1-c)\operatorname{vol}K,(1+c)\operatorname{vol}K]$ with probability at least $1-1/d$ needs $\Omega(d^2/\log d)$ queries. Their Theorem 3 (“determinant”) establishes the related restriction to origin-centered parallelepipeds; the introduction also describes an $O(d^2\log d)$ query upper bound via approximating their matrix. **Incomparable.** R1 supplies a precision-dependent upper bound for a different body class and a different measure. It neither supplies a quadratic lower bound for Gaussian ellipsoids nor removes the logarithm in their lower bound. Their introduction asks which geometric parameters require recovering a quadratic number of entries. The analogous sufficiency statement here is that chord data of order $d^2$ determine a relative Gaussian mass, even for a translated body in the tail; absolute volume accuracy would not give that conclusion uniformly in the Gaussian mass.
- **Rademacher and Vempala**, same preprint, Theorem 23 (“nonadaptive lower bound,” §5.1). For $B_d\subseteq K\subseteq 2dB_d$, a nonadaptive randomized algorithm achieving volume ratio in $[0.9,1.1]$ with probability at least $3/4$ for every such body needs at least $d^{d/2}/(2e(d+2))$ queries. Their argument separates two rounded bodies that are hard to distinguish after rotation. **Incomparable.** R2 obtains a linear lower bound in inverse relative precision, with a dimension-independent leading constant, even for a fixed nested family of Gaussian balls. It does not improve their dimensional lower bound. The contrast is relevant to the paper's stated interest in the power of adaptive randomization, and gives a precision separation when paired with R1.
- **The same Theorems 2 and 23 versus R3. Incomparable.** The former concerns adaptive algorithms with a known common interior $B_d$, and the latter concerns nonadaptive algorithms, both for Lebesgue volume. R3 permits full adaptivity but drops the provision of an interior point, and gives an explicit inverse-mass obstruction for Gaussian measure, on set families whose members are perfectly round. The distinction matters because the introduction identifies roundness and adaptive randomization as central to the volume problem. Here roundness of each individual body is not enough, even with randomization, when an unknown location makes positive responses rare. This is a restricted answer to appendix open problem 4 in the unanchored setting; it does not improve the $\Omega(d^2/\log d)$ lower bound for anchored general convex bodies.
- **He and Wang**, [doi:10.1137/15M1007963](https://doi.org/10.1137/15M1007963), Theorems 3.5 and 3.6 as restated in He, [arXiv:1702.03361](https://arxiv.org/abs/1702.03361), Proposition 2.3. On $[0,1]^d$, for $g\mathbf1_\Omega$ with $g\in L^2$ of bounded Hardy–Krause variation and $\partial\Omega$ of finite $(d-1)$-dimensional Minkowski content, the variance of a scrambled $(t,d)$-sequence rule is $O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})$. Under the partially axis-parallel assumption and the corresponding Minkowski content in irregular dimension $d_u$, Theorem 3.6 replaces $2d-1$ in the denominators by $2d_u-1$. **Incomparable as whole theorems.** On the common structured subclass with all axes irregular and a known elliptical section, R4 gives a better exponent at the same point budget. It uses a different design (stratification *after* conditioning) and more structural information; it does not improve the bound for the same untreated scrambled estimator on general boundaries. Its gain is also relative to direct isotropic stratification, for which the generic boundary cells alone give an $n^{-1-1/d}$ variance term. In the language of Gilbert, Kuo and Sloan, R4 replaces the square-root obstruction for one coordinate with a volume-of-annuli argument for two coordinates. It needs neither a monotone direction nor dominating mixed smoothness of order one in the remaining variables.

## R1. Certified enclosure after reciprocal-chord learning

**Lean:** `Cubature.ellipsoid_mass_enclosure`.

Write $\gamma_d=\bigotimes_{i=1}^d N(0,1)$. Let $d\ge 1$, $a\in\mathbb R^d$,
\[
  E=\{z:q(z-a)\le 1\},\qquad q(x)=x^\top A x+b^\top x,
\]
where $A=A^\top\succeq 0$. An oracle answers **inside** when $q(z-a)<1$ and **outside** when $q(z-a)>1$; either answer is allowed at equality. The input to the algorithm consists of the oracle, the anchor $a$, a number $R\ge0$ with $E\subseteq a+[-R,R]^d$, two numbers $0<l\le h$ specified below, and an integer $n\ge0$. Neither $A$ nor $b$ is given. The bounding box is a promise, not the output of an additional query. For this bounded class $A$ is necessarily positive definite: otherwise, along a nonzero null direction, at least one of the two half-rays has $q\le0$ indefinitely.

Use directions $w=e_i$ and $w=e_i+e_j$ for $j<i$. Let $u_w,v_w$ be the reciprocal positive distances at which the rays $a+tw$ and $a-tw$, respectively, exit. They are
\[
 u_w=\frac{\sqrt{(b^\top w)^2+4 w^\top A w}+b^\top w}{2},
 \qquad
 v_w=\frac{\sqrt{(b^\top w)^2+4 w^\top A w}-b^\top w}{2}.
\]
The promise is $u_w,v_w\in[l,h]$ for these directions and $|a_i|+1/l\le40$ for every $i$. A reciprocal bracket can alternatively be obtained from an a priori ball sandwich about $a$; unnormalized two-coordinate directions have norm $\sqrt{2}$, so the same bounds must cover both kinds of direction.

Set
\[
  \delta=(h-l)2^{-n},\qquad
  \eta=(3h d^2 R^2+2dR)\delta,\qquad
  s=\frac{1+2\eta}{1-2\eta},\qquad
  C(a,s)=s^d\exp\!\left(\frac{s-1}{2(s+1)}\|a\|_2^2\right).
\]
If $\eta<1/2$, the algorithm defined below uses at most $2d^2 n$ queries, each with all coordinates in $[-40,40]$, and returns $L,U$ with
\[
             L\le\gamma_d(E)\le U\le C(a,s)L.
\]
The bound is deterministic and multiplicative even for very small $\gamma_d(E)$. Lean states the bracket, explicit query bound, and coordinate bound together. It uses finite tuples of reals and Mathlib's product of univariate Gaussian measures.

### Algorithm and proof of R1

1. **Sections.** For each of the chosen directions run two independent bisections on $[l,h]$: the positive search queries $z=a+w/t$, the negative search queries $z=a-w/t$, with $t$ the arithmetic midpoint. Keep the lower half for an inside answer and the upper half otherwise. Indeed, $u_wv_w=w^\top A w$ and $u_w-v_w=b^\top w$, and
   \[
   q(w/t)-1=\frac{(u_w-t)(v_w+t)}{t^2},\qquad
   q(-w/t)-1=\frac{(v_w-t)(u_w+t)}{t^2}.
   \]
   Thus the maintained interval contains the respective reciprocal root, independent of how equality is answered. After $n$ readings take the lower endpoints $\hat u_w,\hat v_w$. Both estimates are in $[l,h]$ and their absolute errors are at most $\delta$. All points queried are allowed since $t\ge l$ and $|w_i|\le1$. The algorithm stores the two endpoints per direction and uses no further oracle calls. There are $d(d+1)n$ readings in the triangular frame, at most the formal bound $2d^2n$.
2. **Coefficients.** Write $\hat Q_w=\hat u_w\hat v_w$, $\hat A_{ii}=\hat Q_{e_i}$ and $\hat b_i=\hat u_{e_i}-\hat v_{e_i}$. Set $\hat A_{ij}=\hat A_{ji}=(\hat Q_{e_i+e_j}-\hat A_{ii}-\hat A_{jj})/2$ for $j<i$. Because $|\hat u_w|,|v_w|\le h$,
   \[
     |\hat Q_w-w^\top Aw|\le
       |\hat u_w|\,|\hat v_w-v_w|+|v_w|\,|\hat u_w-u_w|\le 2h\delta.
   \]
   Consequently $|\hat b_i-b_i|\le2\delta$ and $|\hat A_{ij}-A_{ij}|\le 3h\delta$ for all entries. For $\hat q(x)=x^\top\hat A x+\hat b^\top x$ this gives $|\hat q(x)-q(x)|\le\eta$ whenever $\|x\|_\infty\le R$. Positivity of $\hat A$ is not required.
3. **Sandwich.** In the promised box let
   \[
     E_-=\{z:\hat q(z-a)\le1-\eta\},\qquad
     E_+=\{z:\hat q(z-a)\le1+\eta\};
   \]
   both sets are intersected with $a+[-R,R]^d$. Then $E_-\subseteq E\subseteq E_+$. If $z=a+x\in E_+$, then $q(x)\le1+2\eta$. Since $s\ge1$ and $x^\top A x\ge0$, $q(x/s)\le q(x)/s\le1-2\eta$. The contracted point $a+x/s$ is still in the box, so $\hat q(x/s)\le1-\eta$. This proves $E_+\subseteq a+s(E_--a)$.
4. **Relative measure bound.** For every measurable set $D$, $\gamma_d(a+s(D-a))$ equals the probability that the contracted Gaussian $a+(Z-a)/s$ belongs to $D$. In coordinate $i$ the pushforward density divided by the standard normal density is
   \[
   s\exp\!\left(-\tfrac12\big[(s y-(s-1)a_i)^2-y^2\big]\right)
   \ \le\ s\exp\!\left(\frac{s-1}{2(s+1)}a_i^2\right).
   \]
   To see the inequality, multiply the difference of the squares by $s+1$ and add $(s-1)a_i^2$; the result is $(s-1)((s+1)y-sa_i)^2\ge0$. Independence gives the product bound $C(a,s)$ for the joint density ratio. Therefore
   \[
     L:=\gamma_d(E_-)\ \le\ \gamma_d(E)\ \le\
     U:=\gamma_d(E_+)\ \le\ \gamma_d(a+s(E_--a))\ \le\ C(a,s)L .
   \]
   The proxy sets are measurable (the defining inequalities are continuous), so all probabilities exist.

These are exact auxiliary integrals of *known* coefficients, not additional black-box calls. In the information model one can approximate them by interval subdivision inside the known box: enclose the polynomial's range on each rectangle, and sum products of univariate normal masses over included rectangles. With arbitrarily fine subdivision the undecided mass tends to zero, since the levels $1\pm\eta$ are nonzero and a nonconstant polynomial, or a nonzero constant after subtracting the level, has Lebesgue-null zero set. Known coefficients can be evaluated to sufficient precision; finite arithmetic and the complexity of this auxiliary integration are not certified by R1.

For example, $U$ is a relative-error estimate with error at most $C(a,s)-1$. For $\eta\le1/4$, $\log s\le8\eta$ and $(s-1)/(2(s+1))=\eta$. The condition $\eta\le\log(1+\varepsilon)/(8d+\|a\|^2)$ then suffices for relative error at most $\varepsilon$. In particular the precision dependence is $O(d^2\log(1/\varepsilon))$ queries at fixed dimension and geometric promises, as $\varepsilon\downarrow0$. The bound for exact auxiliary integrals is the formal claim, not a certificate for the floating-point solver.

This is a subclass of $\mathcal F_d(1,\infty,M,c)$ for any fixed $c>0$ and sufficiently large $M$ depending on the body and $c$: $1-q(z-a)$ is a quadratic, the two extensions are constants $0$ and $1$, and the gradient is bounded away from zero on the compact boundary. The gradient cannot vanish on $q=1$, because a stationary point of this convex quadratic minimizes $q$ and $q(0)=0$. The boundary is null under $\gamma_d$; the choice of oracle answer on the boundary does not affect the integral or the proof. The ellipsoid may be far from the origin as long as the query-coordinate promise holds.

## R2. Inverse-precision lower bound for nonadaptive randomized queries

**Lean:** `Cubature.gaussian_balls_nonadaptive_precision`.

For $d\ge1$ let $p(r)=\gamma_d(\{z:\|z\|_2\le r\})$, $p_*=p(1)$ and $r_*=p_*/8$. The fixed radius range is $[r_*,1]$; it does not depend on the target precision. The input is the indicator of any ball with radius in this range, with a closed-ball convention. Let $m$ be any positive integer and set $\varepsilon=1/(8m)$. A nonadaptive randomized point-query algorithm that, on *every* such input, achieves relative error at most $\varepsilon$ with probability at least $3/4$ must have worst-case query budget $N$ satisfying
\[
                         N+1\ \ge\ 3m/4.
\]
The random seed can have any probability law. The algorithm is a seed-dependent list of points of length at most $N$, followed by a seed- and transcript-dependent real-valued estimate. Only measurability of the success event for each radius is assumed. The lower bound even allows query points outside the protocol box. The class has $K=1$ and analytic extensions as in R1 and allows a common $M$ at fixed $d$ since $r_*>0$.

**Proof.** First $p_*>0$ since the unit ball contains a box of positive Gaussian measure. Also $p(r)\le\Pr(|Z_1|\le r)\le2r$ since the normal density is at most one. Thus $0<r_*<1$ and $p(r_*)\le p_*/4$. The centered special case of the dilation estimate in R1 gives $p(sr)\le s^d p(r)$ for $s\ge1$. For positive $t,r$, this and monotonicity give
\[
     p(r)\min\{1,(t/r)^d\}\le p(t)\le p(r)\max\{1,(t/r)^d\}.
\]
Taking $t\to r$ proves continuity on the positive axis. By the intermediate value theorem choose $r_i\in[r_*,1]$ with $p(r_i)=p_*(1/2+i/(2m))$ for $i=0,\ldots,m-1$. These radii are increasing since their masses are strictly increasing. The accuracy intervals for distinct $i$ are disjoint: neighboring masses differ by $p_*/(2m)$, whereas the sum of any two accuracy half-widths is at most $2\varepsilon p_*=p_*/(4m)$.

Fix the random seed. Along the ordered radii each fixed query changes its membership bit at most once, from zero to one. More explicitly, the sets of positions with bit one form a nested chain; two of these sets of the same cardinality are equal. There are at most $N+1$ distinct transcripts. A deterministic output for a given transcript can succeed for at most one $i$, by disjointness of the accuracy intervals. Thus the sum of the $m$ success indicators at this seed is at most $N+1$. Integrating the finite sum over the seed law bounds the sum of the success probabilities by $N+1$. It is at least $3m/4$ by assumption, proving the claim.

These balls also satisfy R1 with the same radius promise $R=1$, anchor $a=0$, $l=1$ and $h=\sqrt{2}/r_*$. This gives an adaptive logarithmic upper bound on this family (one could simply bisect the radius on a single ray if the spherical shape is known). So R2 is a separation in **precision**, not a claim about the optimal dimensional exponent on all ellipsoids. It answers a restricted version of appendix open problem 4.

## R3. Locating a rare body without an interior anchor

**Lean:** `Cubature.gaussian_rare_balls_adaptive`.

Fix $d\ge1$ and an integer $m\ge1$. Denote the standard normal density by $\phi$. The input is an indicator of a closed ball $B(a,r)$, with unknown $a\in[0,1]^d$ and radius in the known interval
\[
  r_-\le r\le r_+,\qquad r_+=\frac{1}{3m},\qquad
  r_-=\frac{\phi(2)}{4(d+1)}r_+ .
\]
Every such ball has mass in the interval
\[
  0< p_{\min}:=\left(\frac{2\phi(2)r_-}{d+1}\right)^d
  \ \le\ \gamma_d(B(a,r))\ \le\ (2r_+)^d.
\]
For a randomized algorithm of worst-case depth $N$ that estimates each mass with relative error at most $1/3$ and probability at least $3/4$, necessarily $2N\ge m^d$. The result holds for any seed law for which the success events are measurable. It allows arbitrary adaptive queries, even outside the protocol box. At fixed $d$, all input masses are $\Theta_d(m^{-d})$, so this is an inverse-mass lower bound on a uniformly rare input family. The input subclass is in $\mathcal F_d(1,\infty,M,c)$ for fixed $c>0$ and a common sufficiently large $M$ at fixed $d,m,c$; the gradient lower bound for the radius-squared defining polynomial is $2r_-$ on these boundaries. The claim does not keep the smoothness parameter $M$ fixed as $m\to\infty$.

**Proof.** A ball with radius $r\le1$ is contained in the coordinate box of half-width $r$ and contains the coordinate box of half-width $r/(d+1)$. In the smaller box each coordinate lies in $[-1,2]$, on which $\phi\ge\phi(2)$; globally $\phi\le1$. Independence gives
\[
 \left(\frac{2\phi(2)r}{d+1}\right)^d\le \gamma_d(B(a,r))\le(2r)^d .
\]
The mass ratio between radii $r_-$ and $r_+$ at the same center is at most $4^{-d}\le1/4$. The two $1/3$-relative-error intervals are therefore disjoint.

For $j\in\{0,\ldots,m-1\}^d$ consider the two radii at center $a_j=(j+\mathbf1/2)/m$. The $m^d$ outer balls are pairwise disjoint: distinct centers differ by at least $1/m$ in some coordinate, exceeding $2r_+$. At any fixed seed, follow the tree along the path that always receives zero. Its length is at most $N$, so its points hit at most $N$ outer balls. For any center not touched, both radii produce exactly this zero transcript and its single output; the algorithm can succeed for at most one of the two balls. For touched centers it can succeed on at most two. Thus at most $m^d+N$ of the $2m^d$ inputs succeed at that seed. On averaging over the seed the sum of their success probabilities is at least $(3/4)2m^d$. Therefore $(3/2)m^d\le m^d+N$, as claimed. This counts only success events and makes no assumption of independence between different runs of the same seed.

## R4. Two-coordinate smoothing and a variance rate

**Lean:** `Cubature.two_axis_preintegration_stratified`.

Let $m=d-2>0$, and take two known independent standard Gaussian coordinates $u\in\mathbb R^2$. Their section is elliptical: for some given $a\in\mathbb R^2$ and $\lambda_1,\lambda_2>0$,
\[
 P(t)=\Pr\!\left(\lambda_1(U_1-a_1)^2+\lambda_2(U_2-a_2)^2\le t\right).
\]
Let $g,q$ on $\mathbb R^m$ be Lipschitz in the maximum norm with constants $L_g,L_q$, and $|g|\le M$. The section integrand is $g(x)$ times the indicator of the displayed quadratic at level $q(x)$. Its integral in the two Gaussian variables is $H(x)=g(x)P(q(x))$. Then $H$ is Lipschitz with constant
\[
 L=L_g+\frac{M L_q}{2\sqrt{\lambda_1}\sqrt{\lambda_2}}.
\]
For $k\ge1$ take $n=k^m$ independent samples $X_j$ with laws $\mu_j$, indexed by the $k^m$ cells. Assume that each $\mu_j$ is a probability law supported within maximum-norm distance $1/(2k)$ of a center $c_j$ (an almost-sure condition suffices). Then the estimator $\widehat I=n^{-1}\sum_j H(X_j)$ has expectation equal to $n^{-1}\sum_j\int H\,d\mu_j$ and variance at most
\[
                   \frac{L^2}{4k^2n}=\frac{L^2}{4}n^{-1-2/m}.
\]
Lean verifies the Gaussian section identity, the Lipschitz bound, the expectation, and this variance bound. Uniform laws inside the regular cells of $[0,1]^m$ satisfy the stated support hypothesis and have uniform mixture on the cube.

**Proof.** Set $V_i=\sqrt{\lambda_i}(U_i-a_i)$. The joint density of $V$ is at most $(2\pi\sqrt{\lambda_1}\sqrt{\lambda_2})^{-1}$, irrespective of the shift. For $s\le t$ the Lebesgue area of the annulus with squared radius between $s$ and $t$ is $\pi(\max(t,0)-\max(s,0))$. Hence
\[
 0\le P(t)-P(s)\le \frac{t-s}{2\sqrt{\lambda_1}\sqrt{\lambda_2}} .
\]
There is no singular factor when $s$ or $t$ tends to zero. The identity for $H$ follows from integrating an indicator times a constant. In the product difference $H(x)-H(y)$, the part from changing $g$ is at most $L_g\|x-y\|_\infty$ since $P\le1$, and the part from changing $P$ is at most $M|q(x)-q(y)|/(2\sqrt{\lambda_1}\sqrt{\lambda_2})$. This proves the claimed Lipschitz constant. In cell $j$, $H(X_j)$ lies in $[H(c_j)-L/(2k),H(c_j)+L/(2k)]$ almost surely. A real random variable with range width $w$ has variance at most $w^2/4$: its mean squared distance to the midpoint is at most $w^2/4$, and its mean minimizes mean squared distance. Thus the variance in each cell is at most $L^2/(4k^2)$. Independence adds the variances, division by $n$ squares the normalization, and linearity gives the expectation.

**Point queries and Gaussian weights.** In the Gaussian problem an example is
\[
 f(u,y)=G(y)\mathbf1_{\{\lambda_1(u_1-a_1)^2+\lambda_2(u_2-a_2)^2\le q_0(y)\}},
\]
with a supplied quadratic $q_0$ and frame, and unknown smooth $G$ supported on a compact box in the remaining variables. For positive $q_0(y)$ one query $f(a,y)$ obtains $G(y)$; for nonpositive $q_0(y)$ the section has mass zero. Computing $P$ needs no unknown function evaluations. Affinely map the remaining box from the unit cube and include the product Gaussian density and box Jacobian in $g$. Both $g$ and $q=q_0$ composed with the affine map are bounded Lipschitz on the cube for smooth bounded $G$; coordinatewise clipping extends them with the same Lipschitz constants for the formal hypothesis. This gives unbiased cubature of the original integral with at most $n$ readings assuming exact auxiliary two-dimensional integrals, provided the query points in the supplied frame respect the coordinate bound. A smooth compactly supported $G$ with bounded derivatives through a prescribed finite $r$, and a regular defining quadratic, give a subclass of $\mathcal F_d(1,r,M',c)$ for sufficiently large $M'$. For a noncompact $G$ the same result concerns the truncated integral; a separate tail bound is necessary. An ordinary shifted positive-definite ellipsoid, viewed in an eigencoordinate frame, has the displayed sections, and every line through its center sees both an entry and an exit. The result does not require that a direction be monotone, but does require a known section and an amplitude independent of its two coordinates. The unknown rotations and general amplitudes of the hidden cases are not assumed to satisfy that extra information model.

## Numerical solver

`run.sh` starts a single-threaded Python process. The process reads only the provided dimension, budget, batch limit, time limit, seed, and queried values. Its pilot is one evaluation at the origin and 512 evaluations from a scrambled Sobol' net transformed by $\Phi^{-1}$. All probes, including finite-difference stencils and discarded fits, are charged to the budget. Rotations are orthogonal, and the final residual designs use fresh scrambles conditional on the pilots. No error bound is emitted.

The standard estimator for a control $C$ of known or separately computed mean $I(C)$ is $I(C)+\sum_B w_B\,\overline{(f-C)}_B$, over independent blocks $B$ of complete Sobol' nets. The block sizes are the descending powers of two in the remaining budget, and $w_B$ is proportional to $|B|^{1.35}$ in the ordinary Gaussian rule. Importance rules multiply residuals by the standard-Gaussian-to-proposal density ratio. Fits and time guards in the source determine the exact realized allocation. The primary implementations are `ellipsoid.py`, `kinks.py`, `options.py`, `callfit.py`, and `barriers.py`.

### Main branches

- **Closed jumps.** A large gap in observed values identifies a candidate membership label. Otherwise the detector takes 24 rays with 12 points each and seven refinements of the largest visible drop on each ray. Additional line segments between high and low pilot values can supply an off-center anchor. The main geometric fit uses at most 86 initial rays and then bisects signed jumps along more unit rays from the anchor. It regresses
  \[
  (r u)^\top\hat A(r u)+\hat b^\top (r u)\approx1
  \]
  in anchor-relative coordinates. The second ray sample is capped at 1,180 for constant rare jumps and 940 otherwise, with both the number of rays and subsequent refinements chosen to reserve at least roughly 5,000 readings when possible. The program tries both a regularized full fit and positive low-rank factorizations, with rank selected on held-out rays. It reuses the recorded brackets to refine instead of paying again to locate the same section. A concentrated high-dimensional pilot can trigger a frame fit: both signs along axes recover the diagonal and linear terms; subsequent single sections in pair directions estimate the off-diagonals. The positive-semidefinite projection preserves the fitted diagonal.
- **Jump residuals.** Let $D(z)=J(z)\mathbf 1_{\{\widehat q(z)<0\}}$ be the fitted jump. For a nearly constant rare jump, $J$ is a robust average of endpoint differences. For an ordinary jump, $\log J$ is fitted as affine. The smooth-residual and jump-residual integrals are estimated separately using the identity
  \[
  f=\big(f-J\,\ell\big)+D+J\big(\ell-\mathbf1_{\{\widehat q<0\}}\big),
  \]
  where $\ell$ is a pilot-defined value classification (threshold for rare jumps; for ordinary jumps a comparison to the estimated smooth background near the surface). The smooth part uses ordinary rotated nets, and the discrepancy uses a defensive tilted Gaussian mixture for rare events, or a mixture of ordinary points and radial-band points for ordinary jumps. High-dimensional rare bodies with a reliable origin anchor can instead use paired radial endpoints, three refinements and a last randomized radial draw per direction; the calculation subtracts the model radius's Gaussian mass. The proxy Gaussian mass is evaluated by characteristic-function inversion or rank-two conditioning. For an exponential-affine $J(z)=e^{c+b^\top z}$, its proxy integral is $e^{c+\|b\|^2/2}$ times the fitted region's mass under a Gaussian shifted by $b$. Poorly predictive surfaces fall back to a scaled control or standard rotated sampling.
- **Continuous kinks.** Between 46 and 128 short lines (depending on dimension) at Gaussian and radial pilot locations are scanned at nine nodes for changes in slope. Candidate crossings are refined by intersecting the separate local quadratics through three samples on either side; the code checks that the slope jump persists at a much smaller scale. At a retained crossing it differences two nearby same-side stencils to estimate the gradient and Hessian of the *difference* of the smooth pieces, first attempting a small column sketch in large dimensions. It learns up to three quadratic surfaces and selects a sum of hinges or a min/max-type expression using out-of-stencil data. A hinge has a one-dimensional characteristic-function integral. For a min/max expression the program integrates the piecewise quadratics analytically along one Gaussian coordinate and uses an offline net on the remaining coordinates of the proxy, or integrates the product of marginal survival functions when two fitted concave components can be made orthogonal. The final residual net is based on gradients of the smooth remainder, not of the original hinge-dominated payoff. Hermite polynomials have known zero means; a Gaussian kernel control is centered by its explicit Gaussian mean and used only if a pilot split confirms a better residual.
- **Digital payoffs.** The detector requires exactly two observed values, zero and the positive amount. There is a logistic pilot direction, a 64-section endpoint check, and 128 sections to refine the normal and estimate transverse curvature and a quadratic control for the threshold. The final conditional estimate uses up to 1,024 transverse Sobol' points. If the final section bracket has Gaussian CDF endpoints $L,H$, the row value divided by the positive payout is $1-H+(H-L)B_{\rm mid}+(1-H+L)(B_{\rm off}-B_{\rm right})$. Here $B_{\rm mid}$ is the indicator at a Gaussian draw restricted to the bracket, $B_{\rm off}$ is the indicator at a draw restricted to its complement, and $B_{\rm right}$ tests if the latter lies to the right. This is a randomized residual correction even when a line is not monotone. The mean of the fitted quadratic-threshold control is a Gaussian quadratic-form probability.
- **Other option payoffs.** Up to 32 positive points are differentiated using one-sided stencils to estimate a gradient basis. The code fits a discounted call shape $(e^{c+b^\top z}-K)_+$ using positive pilot values. Where exercise appears continuous and agrees with that pilot, it uses up to 128 short sections to fit an exercise threshold $\widehat r(w)$ in transverse coordinates, and up to 1,024 additional tilted training values. It then fits a control $(t-\widehat r(w))_+P(t,w)$, polynomial in the leading coordinate $t$; truncated Gaussian moments integrate this coordinate and offline nets give the rest of the proxy mean. Where many apparently in-the-money samples pay zero, a tilted 640-point probe can fit a soft survival factor for the call proxy; the factor is a probit of an affine combination and its Gaussian call-weighted mean reduces to a one-dimensional integral. A defensive one-coordinate normal mixture can be inverted for importance sampling of the remaining option residual. Very sparse or poorly explained payoffs use the rotated Gaussian fallback. This fallback also covers rejected structural fits on the synthetic track.

### Relation to experiments

R1 applies to the exact membership component of a *bounded, full-rank* closed jump if a valid interior anchor and radius promises are available. Neither the smooth backgrounds nor the arbitrary jump amplitudes in the hard track are assumed known. Low-rank cylinders are not bounded ambient ellipsoids, and a tail ellipsoid need not provide an interior point a priori. R3 concerns why obtaining that point can be expensive; it does not predict a rarity cost on a case whose Gaussian mean is already inside. R4 describes a known-section algorithm and does not certify a rate for the learned ordinary-jump or option controls. None of these bounds uses the score floor or the floating-point resolution.

The experiment uses the same idea of coupling many sections through a quadratic, but trades certified reconstruction for a fitted proxy and an independent correction. In sample tests the ordinary-jump absolute relative errors were on the order of $10^{-5}$ to $10^{-4}$, versus reference errors of about $4\cdot10^{-4}$. On the two rare-jump samples the errors were about $2.4\cdot10^{-5}$ and $7.2\cdot10^{-5}$, versus reference errors about $4.8\cdot10^{-3}$ and $4.1\cdot10^{-2}$. These observations are consistent with geometry learning removing most of the indicator variance when the fit is accurate. They do not validate any asymptotic claim on the full task class. Amplitude fitting, auxiliary numerical integration and the remaining smooth background all contribute to the sample errors.
