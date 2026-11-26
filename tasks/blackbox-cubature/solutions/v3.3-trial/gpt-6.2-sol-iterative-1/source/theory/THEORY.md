# Gaussian cubature from value queries

## 1. Submitted algorithm

The driver reads the dimension, budget, batch limit and seed. It uses one CPU,
single-threaded numerical libraries, and no files from earlier instances.
All model fitting uses oracle values. A Gaussian pilot with 384 scrambled
Sobol' nodes estimates scales and checks for zero values or a discrete payoff
level. The branch labels mentioned below refer to these numerical tests, not
to a family identifier. There is no output error bound.

**Continuous branches.** The solver probes 16 lines, mostly through the
origin, on a 41-node grid. Large fourth differences are refined on successive
15-node grids. The final pair of values distinguishes a jump from a kink, and
small central differences on either side estimate the two gradients. Subtracting
the gradients and Hessians on opposite sides of a kink removes the common
smooth component to leading order. The Hessian calculation starts with a sketch
of at most 12 columns, checks a low-rank completion against additional entries,
and expands the stencil when needed. Quadratic branch differences, including
the affine terms, are assembled into candidate positive parts and maxima.
Selection uses a held-out portion of the pilot and allows a zero control.

For a quadratic \(Q=c+b^T Z+Z^TAZ\), with eigenvalues \(\lambda_i\) and
coordinates \(b_i\) in its eigenbasis, the implementation integrates a single
positive part using
\[
 \mathbb E Q_+
 =\tfrac12\mathbb E Q+\tfrac1\pi\int_0^\infty
      \frac{1-\Re\varphi_Q(t)}{t^2}\,dt,\qquad
 \varphi_Q(t)=\exp\left(itc-\tfrac12\sum_i\log(1-2it\lambda_i)
         -\tfrac12 t^2\sum_i\frac{b_i^2}{1-2it\lambda_i}\right).
\]
The formula follows by integrating \(1-\cos(t Q)\) first at fixed \(Q\).
Multiple nearly independent concave quadratics are integrated from their
one-dimensional distribution functions; otherwise the fitted maximum is
integrated by additional scrambled nets without oracle calls.

**Closed jumps.** From the localized values and gradients the solver estimates
a putative amplitude \(a(z)=\exp(c+\beta^Tz)\). It tests consistency between
several edges before using the fit. It finds an interior anchor by probing the
origin and, if necessary, using high-branch pilot points. In the random-ray
design it brackets radii, uses local endpoint values with the jump amplitude
removed on the inside, and regresses the resulting crossings on quadratic
features. In the pair design it first probes both signs of orthogonal axes.
Axis sums identify the cross terms using binary tests in coefficient space.
The negative-tilt rays have an extra local scan to avoid extrapolating an
uncertain background. If the axes are poorly bracketed, small transverse
shifts around detected crossings estimate normals by root differences.
An SVD of several normals proposes a lower-dimensional fit, which has to
predict other observed crossings. Larger pair fits can be refined by
signed rays along the estimated eigenvectors. For sufficiently high-rank
nonconstant jumps the solver can instead use the eigenbasis of the
Gaussian mass derivative with respect to the curvature matrix, computed
from the quadratic characteristic function. If a paired crossing reveals
that the interior is the lower branch, the solver fits the negated payoff.
It can also search at two extra Gaussian scales when the first probe set
finds no structure.

The surrogate contribution is
\(\exp(c+\|\beta\|^2/2)\Pr_{N(\beta,I)}(\widehat E)\). For a positive
quadratic form written as \(\sum_i \lambda_i(X_i+\mu_i)^2\), the solver computes
its CDF by a gamma mixture (or characteristic-function inversion when the
series converges slowly). With \(s=(1-2\lambda_{\min}t)^{-1}\), the MGF divided
by \(s^{k/2}\) has a power series in \(s\) with nonnegative coefficients;
the coefficients are recovered by exponentiating the log series recursively.
For an accurate geometric fit, the residual is integrated directly against
the original normal. When the jump is very large and the geometry less precise,
the solver also estimates a membership correction under a mixture of a
tilted Gaussian and a radial shell about the fitted boundary. The proposal
density, including both mixture terms, appears in the weight. For a moderate
jump it can instead bracket the two sides on each outer ray, use the
\(\chi_d\) CDF for the part of the ray with known sign, and sample the remaining
interval conditionally. The rare-jump correction also uses this bracketing
where its validation shows that the shell contains reliable crossings. For
cylinders with a fitted kernel, the active coordinates supply the \(\chi_k\)
radius and the inactive coordinates are independent Gaussian draws.
These choices are checked by independent probes;
the theoretical result below does not assume that these empirical phase
decisions are always correct.

**Residuals and options.** On the synthetic branches, the remaining budget is
used for rotated scrambled Sobol' rules and polynomial controls. Univariate
controls are fitted on 9-node Hermite rules in a numerical curvature basis;
linear, diagonal quadratic and a few cross terms can be fitted on normal
pilot points. The polynomial control integrals are their constant Hermite
coefficients. The usual output is the surrogate integral plus the average
oracle-minus-surrogate residual, often with antipodal pairs. Low-dimensional
continuous fits may also use a nested Gaussian sparse rule with an independent
scrambled-net check. In larger continuous cases the final antipodal nodes
are split into independent scramble groups. Centered quadratic and quartic
features fitted on one group correct the integral estimate in the other.

For a two-valued nonnegative payoff, a logistic fit and local membership
searches estimate the crossing direction. Root observations in the orthogonal
complement train a height predictor (176 or 224 pilot lines), with an
optional curvature sketch from nearby root measurements. The production
design uses up to four
observations per ray to bracket the predicted crossing, then draws one point
uniformly in conditional Gaussian probability between the remaining endpoints.
It integrates the predicted conditional probability separately with larger
surrogate-only nets. For a continuous positive payoff, finite differences
inside the live branch estimate an active direction. A monotone axis payoff
curve, if validated, is inverted by interpolation. The solver fits deviations
from this inverse coordinate using localized basis functions of the active
coordinate times linear and low-order transverse features. These fits use
1,536 or 2,048 normal training nodes and 640 or 1,024 additional nodes
concentrated in the active coordinate. Derivatives already queried for the
heading can also constrain the fit. A held-out test guards the warp and a
separate optional kernel correction, whose Gaussian expectation is known
by completing the square. If necessary a logistic gate approximates
knockouts; it is a control, not a restriction on the final residual.
The residual points are stratified by normal probability in the leading
direction unless held-out second moments favor a defensive importance mixture.
For a sufficiently large stratified design the residual correction is
cross-fitted on separate scrambles. It uses Gaussian functions of the
leading coordinate times Gaussian polynomial features of the other
coordinates. The exact stratum means are computed by completing the square
and taking a normal CDF difference.
If the pilot reports few live points, the solver learns successively refined
Gaussian proposals with regularized conditional moments and integrates by a
defensive mixture with the original Gaussian. An optional richer mixture
in the leading coordinates has to pass a separate second-moment check.
For densities \(q_i\) chosen with deterministic design weights \(\pi_i\),
the residual contribution uses \((f-g)\gamma_d/\sum_i\pi_i q_i\) at all
nodes, with the node averages from the components combined with weights
\(\pi_i\). Late checks and model-fit rejection can trigger a simpler
rotated or unrotated net.

In all branches fitted quantities use a finite portion of the oracle budget.
Before expensive stencils, bisections, or shell corrections the code reserves
points for final residual integration. Replicates that evaluate only the model
consume CPU time but no oracle points. `solve.py`, `hard.py`, `jumps.py`,
`options.py`, `warp.py`, `kernels.py`, `mixfit.py`, `sparse.py`, `strata.py` and
`numerics.py` specify the numerical thresholds, stage sizes and fallback
rules.

## 2. R1: Adaptive separation for closed Gaussian jumps

Let \(d\ge1\), let \(0<\rho\le20\) be known, and write \(u=z/\rho\). The oracle
is a membership payoff
\[
 f(z)=\mathbf1_{E}(z),\qquad
 E=\{z:q(z/\rho)<1\},\qquad q(u)=u^TAu+b^Tu .
\]
The unknown real symmetric matrix satisfies \(I\preceq A\preceq2I\).
The unknown vector satisfies \(|b^Tu|\le(1+\|u\|_2^2)/2\) for every \(u\)
(in particular, all \(b\) of Euclidean norm at most 1 are allowed).
The value at equality is 0. No axes, eigenvalues or center are supplied.
The class is invariant under rotations. The normalization fixes a known
scale, but includes genuinely noncentral ellipsoids, and the entire class
contains the origin in its interior.

This is a subclass with \(K=1\), \(r=\infty\) and \(c=1\) in the appendix.
For example \(M=100d(1+\rho+\rho^{-1}+\rho^{-2})\) suffices: all derivatives of
\(q(z/\rho)-1\) of order above 2 vanish, and the lower orders are bounded by
quadratic, affine and constant functions of \(|z|\). On the boundary,
\(z\cdot\nabla_z(q(z/\rho)-1)=1+(z/\rho)^TA(z/\rho)\ge1\)
and \(|z|\le \rho\sqrt3\). The pieces are the constants 0 and 1.
The boundary is compact. On the line through the origin in every nonzero
direction there are two crossings, so no linear direction can make the
boundary function globally strictly monotone.

Put \(I(E)=\gamma_d(E)\), \(F_d(r)=\Pr(\|Z\|_2<r)\) and
\(\Delta=F_d(\rho)-F_d(3\rho/4)>0\).

1. For \(m\ge0\) set \(\delta=80d\,2^{-m}\). If \(\delta<1\), there is a
   deterministic adaptive rule using at most \(d(d+1)m\) legal queries with
   \[
     (1-\delta)^d I(E)\ \le\ \widehat I\ \le\ (1+\delta)^d I(E)
     \quad\text{for every admissible } E .
   \]
2. Every deterministic nonadaptive rule on \(n\) points, even if its points
   are unrestricted and its output is an arbitrary function of their values,
   has absolute error at least \(\Delta/[2(n+1)]\) on some input in the class.
3. For every randomized nonadaptive rule with \(n\) points and finite mean
   square errors, there is one input, chosen independently of the random seed,
   with mean square error at least \(\Delta^2/[128(n+1)^2]\). This lower bound
   is nontrivial even for output clipped to \([0,1]\), which can only decrease
   error. The integrability premise in the Lean statement avoids using the
   Bochner integral of a nonintegrable function.

Lean: `Cubature.ellipsoid_query_separation` (1 and 2) and
`Cubature.ellipsoid_nonadaptive_mean_square` (3). The Gaussian measure
used in these statements is proved equal to the product of the standard
normal measures, and its mass of \(E\) is the ordinary integral of the payoff.
The tree depth is the maximum count over all binary transcripts.

### Proof of the adaptive bound

Use the \(D=d(d+1)/2\) directions \(v=e_i\) and \(v=e_i+e_j\) for \(i<j\).
Write \(\alpha(v)=v^T A v\), \(\beta(v)=b^T v\). If \(s>0\) is the reciprocal
of the physical radius in normalized coordinates, querying \(z=(\rho/s)v\)
returns 1 precisely when
\[
     s^2>\alpha(v)+\beta(v)s .
\]
There is one positive threshold. In fact \(1\le\|v\|^2\le2\), and
\(G(s)=s^2-\alpha(v)-\beta(v)s\) has opposite signs at \(1/2\) and 4:
\(G(1/2)\le1/4-\|v\|^2+(1+\|v\|^2)/4<0\), and
\(G(4)\ge16-2\|v\|^2-2(1+\|v\|^2)>0\).
If \(r\) is a positive root then \(r-\beta(v)=\alpha(v)/r>0\), and
\(G(s)=(s-r)(s+r-\beta(v))\). The same proof applies to \(-v\).
Bisection with \(m\) midpoint queries returns \(s_+(v),s_-(v)\) whose
errors from the positive thresholds \(r_+(v),r_-(v)\) are each bounded by
\(\epsilon=2\,2^{-m}\). All queries have \(\|z\|_\infty\le2\rho\le40\).

Subtracting the two quadratic root identities gives
\[
\beta(v)=r_+(v)-r_-(v),\qquad \alpha(v)=r_+(v)r_-(v).
\]
Use the product on each direction to recover \(\hat\alpha\). Use the
difference on the axes to recover \(\hat b\). Set \(\hat A_{ii}=\hat\alpha(e_i)\)
and \(\hat A_{ij}=(\hat\alpha(e_i+e_j)-\hat\alpha(e_i)-\hat\alpha(e_j))/2\)
for \(i<j\), with symmetry below the diagonal. Every product error is at most
\(8\epsilon\) because all the factors lie in \([1/2,4]\). Thus the entry
errors are bounded by \(12\epsilon\) for \(A\) and \(2\epsilon\) for \(b\).
If \(\|u\|^2\le3\) then \(|u_i|\le2\) and
\((\sum_i|u_i|)^2\le d\|u\|^2\); consequently
\[
|\hat q(u)-q(u)|\le12\epsilon\,d\|u\|^2+4\epsilon d
\le 40d\epsilon=\delta .
\]

Since \(q(u)\ge(\|u\|^2-1)/2\), all interior points satisfy \(\|u\|^2<3\).
Define \(\widehat E\) by \(\hat q(z/\rho)<1\) and \(\|z/\rho\|^2\le3\).
This clipping avoids any requirement that \(\hat A\) be positive definite.
For \(0\le t\le1\),
\(q(tu)\le t q(u)\) because the quadratic part is nonnegative. This proves
\((1-\delta)E\subseteq\widehat E\): after scaling a true interior point, the
clipped norm condition still holds and
\(\hat q((1-\delta)u)\le(1-\delta)q(u)+\delta<1\).
Conversely, membership in \(\widehat E\) implies \(q(u)<1+\delta\);
applying the contraction \(1/(1+\delta)\) gives
\(\widehat E\subseteq(1+\delta)E\).

For any measurable set \(S\) and \(t\ge1\), changing variables \(z=tx\)
in the Gaussian density gives
\(\gamma_d(tS)\le t^d\gamma_d(S)\), since the density decreases along rays.
Using the inverse dilation yields \(\gamma_d(tS)\ge t^d\gamma_d(S)\) for
\(0<t\le1\). The two inclusions give precisely the claimed interval for
\(\widehat I=\gamma_d(\widehat E)\). Evaluation of the learned region's
mass uses no more oracle calls. As is usual in information complexity, the
statement is in exact arithmetic and does not charge that computation.
An independent deterministic evaluation of the surrogate to a chosen
additional tolerance can be added; for example, bounded-box Riemann
integration applies since the polynomial level set has volume zero
(it is not identically 1), as does the clipping sphere.
For a requested relative tolerance \(0<\eta\le1\), taking
\(m\ge \lceil\log_2(160d^2/\eta)\rceil\) suffices: the upper relative error
is at most \(e^{d\delta}-1\le2d\delta\), and the lower side is at most \(d\delta\).

### Proof of the nonadaptive bounds

For \(r\in[3\rho/4,\rho]\), the ball of radius \(r\) is in the class by taking
\(b=0\), \(A=(\rho/r)^2I\). Its Gaussian mass varies continuously and strictly
increases on this interval. Indeed, any nonempty open annulus has positive
Gaussian mass. For \(0<a\le b\), the dilation argument gives
\[
0\le F_d(b)-F_d(a)\le ((b/a)^d-1) F_d(a)\le (b/a)^d-1 ,
\]
which gives the continuity at every positive radius. Thus all intermediate
mass levels are attained.

Choose \(n+2\) radii with equally spaced masses over the whole interval.
A fixed node distinguishes at most one consecutive pair: its squared norm
must belong to that pair's half-open squared-radius interval \([r_{\rm L}^2,
r_{\rm R}^2)\). There are \(n+1\) pairs, so some pair gives identical
transcripts. The errors from the common output on the two inputs have sum
at least their mass gap \(\Delta/(n+1)\). This proves (2).

For (3), choose \(4(n+1)+1\) equally spaced masses including the endpoints,
write \(w=\Delta/[4(n+1)]\), and form \(2(n+1)\) disjoint pairs by pairing
indices \(2i\) and \(2i+1\). At least \(n+1\) pairs are indistinguishable
for every deterministic realization of a nonadaptive design. If such a
pair has true values \(a,b\) and common output \(c\), completing the square
gives \((c-a)^2+(c-b)^2\ge (a-b)^2/2=w^2/2\).
Averaging over all paired inputs gives at least \(w^2/8\).
This finite prior is fixed before the seed is drawn. Averaging the
inequality over seeds and interchanging the finite sum with expectation
gives an input with mean square error at least \(w^2/8\), as claimed.

### Comparison with human results

**Incomparable.** Gilbert, Kuo and Sloan, *Preintegration is not smoothing
when monotonicity fails* (2022), Theorem 1 (arXiv:2112.11621), concerns
ordinary coordinate preintegration of \(f=\theta\,\mathbf1_{\{\phi>0\}}\).
At a nondegenerate tangency \(\phi=0,\partial_j\phi=0,\partial_j^2\phi\ne0\),
with a transverse derivative nonzero, a nonvanishing smooth amplitude and
positive density give a square-root type singularity in the preintegrated
integrand. Their negative result explains why regularity arguments for
strictly monotone preintegration do not apply to curved boundaries. R1 does
not assert smooth Cartesian preintegration at a tangency. It uses one-sided
rays and algebraic identification instead, and gets a deterministic relative
error guarantee without knowing the rotation. It handles a much narrower
family of surfaces and does not give their general singularity conclusion.

Plaskota and Wasilkowski, *Adaption allows efficient integration of functions
with unknown singularities*, Numer. Math. 102 (2005), 123–144, one-unknown-
singularity result summarized in Appendix §5, compare value-based integration
of piecewise \(r\)-smooth univariate functions: adaptive \(n^{-r}\) is possible
for one unknown singularity, while nonadaptive worst-case error on the jump
class is bounded below at order \(n^{-1}\). (The appendix does not specify
this result's theorem number.) Here the adaptive exponent is in the query
count rather than the smoothness order and the nonadaptive obstruction
persists for random point sets and arbitrary estimators. The quadratic
geometry and constant branch values are essential restrictions; we do not
improve their theorem on its own class. The distinction between a fixed grid
and oracle-driven localization motivates both results.

R1 addresses part of Appendix §6(1), §6(3), and §6(4): for this class a
closed boundary requires no globally monotone direction; an unknown
noncentral jump is located along known design rays using values only; and
adaptive and nonadaptive query orders separate exponentially at fixed
dimension. The statement does not give an \(O(d^2)\) *time* algorithm or a
minimax upper rate over the full \(\mathcal F_d\).

## 3. R2: An unknown odd background

Fix the same ellipsoid class and scale as in R1, but observe real values of
\(f(z)=h(z)+\mathbf1_E(z)\), with \(h(-z)=-h(z)\) and \(h\in L^1(\gamma_d)\).
The rule is not told \(h\), and zeros of \(f\) need not reveal the boundary.
For a subclass of \(\mathcal F_d\) take \(h\) analytic with all derivatives
bounded by \(B e^{|z|}\) for some fixed \(B\). The two pieces are \(h,h+1\);
the parameters are as in R1 with \(M\) increased by \(B+1\).

With \(\delta'=128d\,2^{-m}<1\), at most \(2d(d+1)m\) real value queries
suffice for the two-sided interval in R1(1), with \(\delta'\) in place
of \(\delta\), for the integral of the full \(f\). Every deterministic
nonadaptive \(n\)-node real-value rule still has worst-case absolute
error at least \(\Delta/[2(n+1)]\). Lean:
`Cubature.odd_background_query_separation`.

### Proof

Central reflection preserves the product Gaussian measure, so \(I(h)=0\)
and the target is \(I(E)\). For each \(v\) in the R1 design, query both
\(z=(\rho/s)v\) and \(-z\). Their values add to an integer count in
\(\{0,1,2\}\). Its being greater than \(1/2\) is equivalent to
\(s>\min(r_+,r_-)\); its being greater than \(3/2\) is equivalent to
\(s>\max(r_+,r_-)\). Both tests use the initial bracket of R1.
Run \(m\) bisection iterations for each test, for a total of \(2d(d+1)m\)
point evaluations. Each sorted root is obtained to \(\epsilon=2\,2^{-m}\).

The product is unaffected by ordering, so the same reconstruction gives
entry errors at most \(12\epsilon\) for \(A\). The true difference is
\(|\beta(v)|\). Let \(\widehat w(v)\) be the difference of the approximate
upper and lower roots. Then \(|\widehat w(v)-|\beta(v)||\le2\epsilon\) for
all the design directions. Choose \(k\) maximizing \(\widehat w(e_k)\);
for the analysis, reflect the true ellipsoid if needed so that \(b_k\ge0\).
Neither its mass nor its absolute linear measurements changes. The pivot
choice implies \(|b_i|\le b_k+4\epsilon\) for every \(i\).

Set \(\hat b_k=\widehat w(e_k)\) and
\(\hat b_i=\widehat w(e_i+e_k)-\widehat w(e_k)\) for \(i\ne k\).
Since \(b_i+b_k\ge -4\epsilon\), replacing \(b_i+b_k\) by its absolute
value incurs error at most \(8\epsilon\). Together with the two errors in
\(\widehat w\) this gives \(|\hat b_i-b_i|\le12\epsilon\), including the
pivot. The construction never divides by a small coordinate.

On \(\|u\|^2\le3\) the quadratic error is at most \(40d\epsilon\)
by the R1 estimate with linear coefficients set to zero. The linear
error is at most \(24d\epsilon\) since each \(|u_i|\le2\). Apply the
clipping and dilation proof with their sum, \(64d\epsilon=\delta'\),
to the oriented ellipsoid. The bound targets its reflection-invariant
probability. For the lower bound restrict to \(h=0\); composing the
real reconstruction map with the binary embedding gives R1(2).

### Comparison with human results

**Incomparable.** The Gilbert–Kuo–Sloan tangency result compared for R1
applies to unknown \(C^2\) defining functions satisfying its derivative
conditions and requires no parity. R2 gives an explicit deterministic
relative bound for a closed shifted surface despite unknown full-dimensional
nuisance, but relies on quadratic geometry, a unit jump and antipodal
cancellation. It identifies a geometric observable rather than smoothing
the Cartesian preintegrand. The Plaskota–Wasilkowski results compared for
R1 allow a more general smooth background but are univariate. Neither
parity cancellation nor stable recovery of a multivariate offset is a
conclusion of those results. R2 advances Appendix §6(1) and §6(3) on
its class without an assumption about zeros of the payoff.

## 4. R3: Exact oracle reduction for a Lipschitz background

Use the R1 ellipsoid class, with \(0<\rho\le20\) and unit jump, and let
\(f=h+\mathbf1_E\) where \(h\) is arbitrary unknown and \(L\)-Lipschitz in
the maximum-coordinate norm. Neither its mean nor any parity is assumed.
In \(\mathcal F_d\) this covers, for example, \(K=1,r=\infty\) with the
analytic piece \(h\) obeying the appendix derivative bounds and the stated
global Lipschitz condition; use the same defining polynomial and modify
\(M\) to bound \(h\) and \(h+1\). The formal reduction requires only the
Lipschitz condition and makes no smoothness assumption about \(h\).

Let \(k\ge1\) be an integer with \(k\ge64L\rho\), and let
\(\delta=80d\,2^{-m}<1\) also satisfy \(16L\rho\delta\le1\).
For any deterministic rule \(\mathcal A_h\) with at most \(n\) legal point
queries to \(h\), there is a rule using only \(f\) values, with at most
\[
                d(d+1)(k+1+m)+2n
\]
legal queries, whose output is a pair \((\widehat I_E,\widehat I_h)\).
The second component is **exactly** the output of \(\mathcal A_h\) on \(h\),
including if the queries of \(\mathcal A_h\) are adaptive. The first component
has the relative interval in R1(1). Thus adding the two estimates incurs
no error beyond that of the original background rule and the explicit
geometric error. At fixed \(d,\rho,L\), reducing the geometric error to
order \(n^{-p}\) for any fixed \(p>0\) costs only \(O(\log n)\) extra training
queries. A smooth-background lower bound also remains an obstruction for
this enlarged class: on a fixed known ellipsoid, queries to \(h+\mathbf1_E\)
can be simulated by one query to \(h\) each.

There is a more precise per-query cost bound besides the worst-case factor of
two. Under \(\gamma_d\), the set of locations where the decoder takes its
second observation has mass at most
\[
        \big((1+2\delta)^d-(1-2\delta)^d\big)\,I(E),\qquad 2\delta<1 .
\]
For instance, for a nonadaptive random background design with each node
distributed as \(\gamma_d\) conditioned on legal coordinates, the expected
extra query count after training is at most \(n\) times this bound divided
by the Gaussian probability of legal coordinates. This observation applies
to dependent designs as well, by linearity of expectation. It does not
substitute an expected bound for the deterministic budget.

Lean: `Cubature.closed_jump_cubature_reduction` and
`Cubature.decoding_shell_probability`. `ValueTree.within` bounds the count
on *all* real-valued answer paths, and `ValueTree.admissible` requires that
every queried coordinate on every path be legal.

### Proof

For each signed design direction \(v\) of R1, use the same inverse-radius
interval \([1/2,4]\). Sample its \(k+1\) equally spaced endpoints.
The jump is absent at the first endpoint and present at the last, by the
strict endpoint checks in R1. For \(s,t\ge1/2\) we have
\[
 \|(\rho/s)v-(\rho/t)v\|_\infty\le4\rho |s-t| .
\]
All successive endpoints are at most \(4/k\) apart in inverse radius, hence
the background changes by at most \(16L\rho/k\le1/4\) between them.
Exactly one ordered difference of observed function values is greater than
\(1/2\): the one that crosses the positive jump. The left endpoint, whose
payoff is outside, is kept together with its observed value. During each
bisection step the difference of the midpoint value from this saved value is
greater than \(1/2\) exactly when the midpoint is inside. If outside, replace
the left endpoint and its saved value; if inside, replace the right endpoint.
Subintervals stay below the mesh-width bound, so the comparisons remain valid.
After \(m\) steps the roots meet the R1 precision \(\epsilon=2\,2^{-m}\);
the same coefficient reconstruction, clipping and dilation give the
promised \(\widehat I_E\). This has used \(k+1+m\) raw values per signed ray.

For a requested background ordinate at \(z\), first query \(f(z)\).
With \(u=z/\rho\), if \(\|u\|^2>3\) the point is outside. Otherwise the
reconstructed value \(p=\hat q(u)\) differs from \(q(u)\) by at most \(\delta\).
If \(p<1-\delta\) the point is inside, and if \(p\ge1+\delta\) it is outside.
Subtract 1 in the former case and 0 in the latter. In the remaining case
\(q(u)<1+2\delta\), so \(y=z/(1+2\delta)\) is a certified interior point
by the contraction inequality. It is legal whenever \(z\) is legal.
Also \(\|z\|_\infty\le2\rho\) in the clipped ball, so
\(\|y-z\|_\infty\le4\rho\delta\) and \(|h(y)-h(z)|\le1/4\).
The difference \(f(z)-f(y)\) is therefore greater than \(-1/2\) exactly
when \(z\) is inside. This decodes \(h(z)\) exactly with the one additional
query. Inserting this routine at each query of \(\mathcal A_h\) preserves
its full transcript and hence every adaptive choice it makes.

For the extra-query bound, ambiguity implies
\(1-2\delta\le q(u)<1+2\delta\). By contraction, such a point belongs to
\((1+2\delta)E\) but cannot belong to \((1-2\delta)E\). The latter sets are
nested because \(E\) is star-shaped about the origin. The Gaussian dilation
inequalities bound the measure of their difference by the displayed
expression. This gives the claimed frequency bound.

### Comparison with human results

**Incomparable.** The tangency theorem of Gilbert, Kuo and Sloan discussed
for R1 rules out the usual regularity argument along a fixed linear
preintegration direction. The strictly monotone preintegration theorem of
Griewank, Kuo, Leövey and Sloan, *High dimensional integration of kinks
and jumps* (2018), takes smooth pieces and a smooth defining function,
strict monotonicity in the integrated coordinate, and weighted derivative
and integrability conditions; it yields smoothness of the preintegrated
function in a weighted mixed Sobolev setting. It allows defining functions
that are not quadratic and does not have the information model of R3.
R3 neither computes that preintegrand nor requires the defining function
to be callable. It handles a closed nonmonotone family by identifying
geometry, then restores the smooth oracle **exactly** instead of bounding
a derivative of a preintegral.

The one-unknown-singularity result of Plaskota and Wasilkowski compared
for R1, with smooth one-dimensional pieces, gives an adaptive rate
matching the smooth rate, and a nonadaptive obstruction. R3 similarly
separates the work of localization from smooth integration, but the
conclusion here is a quantitative simulation of an arbitrary
multivariate algorithm, with explicit counts and boundary-shell mass.
It assumes known unit height and the conditioned ellipsoid class and is
not a theorem for their general singularity model. This gives a partial
solution of Appendix §6(1), §6(3), and §6(4) for a full-dimensional
nonparametric background, including internal jumps at nonzero payoff values.

## 5. R4: Sharp finite query threshold for continuous quadratic kinks

Let \(d\ge1\), \(0<\rho\le20\), and
\[
             f_A(z)=\big(1-(z/\rho)^T A(z/\rho)\big)_+,
             \qquad I_d\preceq A=A^T\preceq 2I_d .
\]
This bounded continuous class has one regular nonmonotone kink surface.
It is in \(\mathcal F_d(1,\infty,M,1)\) with the same defining polynomial
and choice of \(M\) as in R1 (here \(b=0\)). The branches are
\(1-(z/\rho)^TA(z/\rho)\) and 0.

Exactly \(D=d(d+1)/2\) value queries suffice for exact Gaussian integration
in the real-arithmetic query model. They can be fixed in advance and all
are legal. Conversely there exists an \(\eta_{d,\rho}>0\), independent of
the algorithm, such that any deterministic algorithm making fewer than
\(D\) queries (even adaptive, and without coordinate restrictions) has
worst-case absolute error at least \(\eta_{d,\rho}\). Lean:
`Cubature.centered_kink_optimal_queries`.

### Proof

For the upper bound, query at \(z=(\rho/4)v\) for each \(v=e_i,e_i+e_j\),
\(i<j\) on sums. Since \(v^TAv\le4\), each value is \(1-v^TAv/16\).
The diagonal and polarization formulas in R1 now give the entire \(A\)
exactly. Output its Gaussian kink integral. This statement is about the
number of oracle values; it treats evaluating the integral of the known
function as an exact real operation. Deterministic numerical integration
of this bounded piecewise polynomial requires no further oracle calls.

For the lower bound, run the proposed decision tree on \(A_0=\tfrac32 I_d\).
Its realized list \(x_1,\ldots,x_n\) has fewer than \(D\) entries.
The homogeneous linear system on symmetric matrices
\[
             (x_i/\rho)^T H(x_i/\rho)=0,\qquad i=1,\ldots,n,
\]
has a nonzero solution since the domain has dimension \(D\).
Normalize it to \(\max_{ij}|H_{ij}|=1\).
Then \(|u^T H u|\le d\|u\|^2\) by Cauchy–Schwarz. With
\(t=1/(4d)\), both \(A_\pm=A_0\pm tH\) belong to the class and return
exactly the same values as \(A_0\) along its transcript. Inducting through
the tree proves that their queries and output are identical to those on
the baseline input. This argument does not assume nonadaptivity.

For any real \(a,b\) the convexity gap
\[
       \tfrac12\big((a+b)_+ +(a-b)_+\big)-a_+
\]
is nonnegative and is at least \((|b|-|a|)/2\). In our case take
\(a=1-\tfrac32\|u\|^2\) and \(b=t u^T H u\).
Among the design directions \(v\) there is one with
\(|v^THv|\ge\|v\|^2/2\): choose an axis if an appropriate diagonal
entry has modulus at least \(1/2\), and otherwise use \(e_i+e_j\) for
an off-diagonal entry of modulus 1. In this latter case the two
diagonals can change \(2H_{ij}\) by at most 1.

Put \(c_v=\sqrt{(2/3)/\|v\|^2}\,v\) and \(r_d=1/(1000d^2)\).
At \(c_v\), \(a=0\), and \(|c_v^THc_v|\ge1/3\).
For \(\|u-c_v\|_\infty<r_d\), each coordinate of \(c_v\) has modulus at
most 1 and each coordinate of \(u\) has modulus at most 2. Hence
\(|u_i u_j-(c_v)_i(c_v)_j|\le3r_d\). Summing over pairs gives
\(|u^THu-c_v^THc_v|\le3/1000\), and summing the diagonal products gives
\(\big|\|u\|^2-2/3\big|\le3/(1000d)\).
In particular \(|b|\ge1/(16d)\), \(|a|\le9/(2000d)\), and the
positive-part gap is at least \(1/(50d)\) there.

Let \(\beta\) be the minimum of the Gaussian masses of the finitely many
open boxes \(\{z:\|z/\rho-c_v\|_\infty<r_d\}\) over the design directions.
It is positive because every box is nonempty and the Gaussian has positive
density. Integrating the gap gives
\(\tfrac12(I(A_+)+I(A_-))-I(A_0)\ge\beta/(50d)\).
If all three errors from one common output were less than
\(\eta_{d,\rho}=\beta/(100d)\), this difference would be less than
\(2\eta_{d,\rho}\), a contradiction.

### Comparison with human results

**Incomparable.** Bakhvalov's 1959 smooth-class result summarized in Appendix
§5 gives the asymptotic deterministic and randomized orders
\(n^{-r/d}\) and \(n^{-r/d-1/2}\) on smoothness balls, including lower bounds
for broad nonparametric classes. It is not an exact finite-data
identifiability result. Plaskota–Wasilkowski's one-singularity result
compared for R1 concerns more general branches in one dimension and
gives asymptotic orders rather than the dimension-dependent threshold.
The obstruction here is quadratic parameter ambiguity: even though the
Gaussian integral ignores the orientation, unseen perturbations change
it. The class is much narrower, and the exact regime depends on that
restriction. This resolves the information question for this particular
nonmonotone kink model in Appendix §6(4), not for all continuous kinks.

## 6. R5: Learning the unknown rank-two subspace in linear dimension

Keep the cap \(f_A(z)=(1-(z/\rho)^T A(z/\rho))_+\) and known
\(0<\rho\le20\), but allow positive semidefinite \(A\) of rank at
most two, \(0\preceq A\preceq 2I\). No factor or orientation is
known. Equivalently \(A=U U^T+V V^T\) for two real vectors \(U,V\),
possibly zero or dependent. These caps are in \(\mathcal F_d(1,\infty,M,1)\)
with \(M\) as in R1: they have polynomial pieces and the same growth
bounds. On any nonempty boundary, \(z\cdot\nabla\varphi(z)=2\)
gives \(\|\nabla\varphi(z)\|\ge2/|z|\ge e^{-|z|}\), even when the
least positive eigenvalue is small. Rank deficiency need not produce
a compact boundary in the original coordinates.

There is a deterministic rule with at most \(3d\) legal value queries
that returns both the **entire** matrix \(A\) and the exact Gaussian
integral in the real query model. For each fixed scale \(\rho\) there
is a positive error threshold independent of \(d\): any deterministic
rule with fewer than \(d\) queries, including unrestricted adaptive
queries, has worst-case absolute error at least
\(\Pr(N(0,1)>\rho)/2\). Thus the dependence on ambient dimension is
tight in order at sufficiently high accuracy, unlike the quadratic
dependence for R4. Lean: `Cubature.rank_two_kink_linear_queries`.

### Proof

The observations at \((\rho/4)e_i\) give every diagonal entry as in
R4. Given the diagonal, the points \((\rho/4)(e_i+e_j)\), \(j\ne i\),
give any prescribed row \(i\) by polarization. The points remain in
the positive branch since \(v^TAv\le2\|v\|^2\le4\).
In particular none of these measurements requires an eigenbasis.

If all diagonal entries are zero, their formula
\(A_{ii}=U_i^2+V_i^2\) shows that \(A=0\). Otherwise pick a positive
pivot \(p=A_{ii}\), learn the row \(a=A_{i,:}\), and consider
\[
                   S=A-aa^T/p .
\]
Its diagonal is known immediately. In fact
\[
 S_{jl}=(U_i V_j-V_i U_j)(U_i V_l-V_i U_l)/p ,
\]
by expanding the numerator. Thus if \(S\) has zero diagonal it is
zero, and otherwise for any \(k\) with \(S_{kk}>0\),
\(S=S_{k,:} S_{k,:}^T/S_{kk}\). One extra original row \(A_{k,:}\)
supplies the required row of \(S\) by subtraction. Reconstruct \(A\)
and output its cap integral as an exact real. At most the diagonal
and two rows have been read, each with at most \(d\) queries.
The returned matrix also identifies the active subspace as its
column span.

For the lower bound, run the algorithm on \(A=0\), so every observed
value is 1. Fewer than \(d\) observed points admit a unit vector
\(w\) orthogonal to all of them. The matrix \(A=ww^T\) meets the
rank and spectral assumptions and gives the identical transcript.
But \(w^TZ\sim N(0,1)\); the cap is at most 1 everywhere and is zero
when \(w^TZ>\rho\). The two integrals thus differ by at least the
indicated one-sided tail probability, and one error is at least half
that difference. The Gaussian projection law is established in Lean
by mapping the product normal measure to the Euclidean standard normal
and applying its covariance and linear-image identities.

### Comparison with human results

**Incomparable.** Liu and Owen, *Preintegration via active subspace*
(2023), use gradient covariance to select a direction and prove a
scrambled-net variance comparison after preintegration, rather than a
point-query guarantee for learning the direction. Xiao and Wang (2019)
estimate gradient covariance, with finite difference queries, to
rotate RQMC rules for derivative payoffs. These methods allow broader
integrands and target effective dimension, not exact finite recovery.
On the algebraic rank-two class here, no gradient oracle, known
orientation or preintegration formula is needed: the proof identifies
the entire active matrix with a linear query bound and gives a
dimension-uniform obstruction to sublinear query count. It partially
answers Appendix §6(2) and §6(4), with the explicit limitation of
exact real returns from pure caps and a rank bound.

## 7. R6: Exact cubature adaptive to an unspecified rank

Consider any \(0\preceq A\preceq 2I_d\) and the same known-scale cap
as in R5. **Neither the rank nor an active subspace is given.** The
regularity and growth checks for R5 continue to hold, regardless of
rank. One deterministic, legal rule for each \(d,\rho\) recovers both
\(A\) and its exact Gaussian cap integral. On an input of rank \(r\)
the number of value queries actually issued is at most \(d(r+1)\).
All possible transcripts, including real-value transcripts not
consistent with a cap, have at most \(d(d+1)\) queries. Lean:
`Cubature.rank_adaptive_exact_cap_cubature`.

In the formalization a Gram factorization with rows \(v_i\in\mathbb R^k\)
is a witness to model membership, not an input to the algorithm.
The rank in the bound is \(\dim\operatorname{span}\{v_i\}\), not
the supplied size \(k\) of this witness. Indeed the kernel of \(A\)
equals the kernel of \(x\mapsto\sum_i x_i v_i\), because
\(x^TAx=\|\sum_i x_i v_i\|^2\). Rank-nullity gives the claimed
interpretation.

### Proof

Read the diagonal in \(d\) queries. As in R5, any row can then be
read whenever needed in at most \(d\) queries. Maintain the accumulated
matrix \(B\), initially zero. The invariant is \(A=B+S\), where \(S\)
is a Gram matrix of residual vectors \(w_i\), initially \(v_i\).
Its diagonal is always known as \(A_{ii}-B_{ii}\).

If the residual diagonal vanishes, each residual vector has zero
norm, and hence \(B=A\). Otherwise choose \(p\) with \(S_{pp}>0\), read
the original row \(A_{p,:}\), and subtract \(B_{p,:}\) to get \(s=S_{p,:}\).
Update \(B\gets B+ss^T/S_{pp}\), which is computed entirely from those
values and the prior state. The new residual is the Gram matrix of
\[
                w'_i=w_i-\frac{\langle w_p,w_i\rangle}
                                   {\langle w_p,w_p\rangle}\,w_p .
\]
This follows by expansion of the inner products. Every \(w'_i\) is
in the old span and is orthogonal to \(w_p\); the old span includes
the nonzero vector \(w_p\). Thus the residual span strictly decreases
in dimension. There can be no more than \(r\) pivots. A loop limited
to \(d\) pivots is therefore enough for all true inputs, and its
fixed cap bounds the cost even for inconsistent real values. With
the diagonal scan, the actual count is bounded by \(d+rd\). The
recovered matrix determines the integral, with no more value calls.

### Comparison with human results

**Incomparable.** The algebraic projection is the classical pivoted
Cholesky update; the result is a reduction from **payoff values** to
those updates, not a new matrix factorization identity. The
active-subspace and GPCA results compared for R5 instead concern
variance after choosing a direction or learning a gradient covariance
and apply to much broader classes. On pure caps R6 provides a
deterministic query bound for discovering the entire subspace with
no rank parameter given to the algorithm. In combination with R4
and R5 it shows how the affine information cost changes between
full and bounded rank. This is an exact-arithmetic partial answer
to Appendix §6(2), not a stability guarantee for recovering rank or
directions from noisy values on the full class.

## 8. R7: Stable linear-cost recovery from bits for rank-two jumps

Let \(d\ge1\), \(0<\rho\le16\), and let
\[
  E=\{z:q(z/\rho)<1\},\qquad q(x)=x^TAx,\qquad
  0\preceq A\preceq 2I_d,\quad \operatorname{rank}(A)\le2 .
\]
Only the values of \(\mathbf1_E\) are observed, with zero assigned on the
boundary. The rule is given one vector \(w\) satisfying
\(\|w\|_\infty\le1\) and \(1\le q(w)\le2\), but not \(q(w)\) itself, any
eigenvalue, or either active direction. The vector is a witness of curvature,
not an active basis. The assumptions are nonempty for rotated rank-one and
rank-two matrices. There is no positive lower bound on the smaller
eigenvalue. The zero level set of \(q(z/\rho)-1\) is closed and can be
cylindrical. As in R5, on this set \(z\cdot\nabla\varphi(z)=2\) gives
\(\|\nabla\varphi(z)\|\ge 2/|z|\ge e^{-|z|}\); together with the polynomial
growth bounds this gives a subclass of \(\mathcal F_d(1,\infty,M,1)\).

Write \(\theta=16\,2^{-m}\), \(\Gamma=8000\theta\),
\(\delta=\Gamma d R^2\), and
\(\tau_R=\Pr(\|Z/\rho\|>R)\) for a chosen truncation radius \(R\ge0\).
When \(\theta\le1/8\) and \(\delta<1\), there is a deterministic rule using
at most \((3d+1)m\) legal point queries. It outputs a matrix \(\widehat A\)
and an estimate \(\widehat I\) satisfying
\[
\begin{split}
  |\widehat A_{ij}-A_{ij}|&\le\Gamma\quad\text{for every }i,j,\\
  (1-\delta)^d(\gamma_d(E)-\tau_R)&\le \widehat I
                           \le (1+\delta)^d\gamma_d(E).
\end{split}
\]
Lean: `Cubature.rank_two_jump_query_bound`. The matrix estimate does not
require resolving the smaller eigenvalue when it is at the level of the
measurement error. The stated mass computation is in the exact-real query
model as in R1.

### Proof

**Measuring curvature from bits.** Write \(A=UU^T+VV^T\),
and put \(\xi=1/16\). We will use \(w\) and directions of the form
\(w+\xi v\) with \(v=\pm e_i\) or \(v=e_i+e_j\). Even if \(i=j\),
\(\|v\|^2\le4\), so \(q(\xi v)\le1/32\). The cross term
\(|w^TA(\xi v)|\le\sqrt{q(w)q(\xi v)}\le1/4\) by Cauchy–Schwarz
on the two Gram factors. Hence \(1/2\le q(w+\xi v)\le3\), and its
maximum coordinate is at most \(9/8\). For any such direction \(u\),
the query at \((\rho/s)u\) returns 1 exactly when \(s>\sqrt{q(u)}\).
This square root lies in \([1/2,4]\). An \(m\)-step bisection gives an
error at most \(2\,2^{-m}\) in the root, so squaring gives error at most
\(\theta\) in \(q(u)\), since the sum of the two roots is at most 8.
All query coordinates are bounded by \(2\rho(9/8)\le36\).

**The two rank-one parts.** Put \(a=q(w)\) and \(b=Aw\). Let \(p_i\)
and \(n_i\) denote the true curvatures at \(w+\xi e_i\) and \(w-\xi e_i\).
Then
\[
 b_i=4(p_i-n_i),\qquad A_{ii}=128(p_i+n_i-2a).
\]
The measured versions have errors at most \(8\theta\) and \(512\theta\),
respectively. For a mixed direction \(w+\xi(e_j+e_i)\), the measured
curvature \(c_{ji}\) yields the row entry
\(128(c_{ji}-p_j-p_i+a)\), also to error \(512\theta\).
The anchor \(a\) and the two coordinate readings are cached.

By Cauchy–Schwarz, \(|b_i|\le\sqrt{a A_{ii}}\le2\). Thus
\(|\widehat b_i|\le3\), \(\widehat a\ge1/2\), and \(a\ge1\).
Separating numerator errors and the reciprocal error gives
\[
\left|\frac{\hat b_i\hat b_j}{\hat a}-\frac{b_i b_j}{a}\right|
\le (8\theta)(6)+(8\theta)(4)+\theta(2)(4)=88\theta.
\]
For example the three terms arise by successively replacing each numerator
factor and then the denominator. The residual \(S=A-bb^T/a\) has rank at
most one and is PSD. In fact it is \(vv^T\), where
\[
  v_i=\frac{(U\cdot w)V_i-(V\cdot w)U_i}{\sqrt a};
\]
the identity follows by expanding the products. The errors in the
estimated diagonal of \(S\) and in any subsequently measured row of \(S\)
are at most \(600\theta\). Below set \(t=600\theta\).

If the largest measured residual diagonal is at most \(2t\), every
true \(v_i^2\) is at most \(3t\). Dropping all of \(S\) therefore incurs
entry errors at most \(3t\). Otherwise let \(j\) maximize the measured
residual diagonal, and let \(p=v_j^2\), \(\hat p=\widehat S_{jj}>2t\).
We have \(\hat p/2<p\le(3/2)\hat p\), and for all \(i\),
\(v_i^2\le\hat p+t\le3p\). The true pivot row \(s_i=v_jv_i\)
satisfies \(|s_i|\le2p\), so \(|s_i|/\hat p\le3\) and
\(|\hat s_i|/\hat p\le7/2\). The same three-term estimate, now for
the estimated residual row, yields
\[
\left|\frac{\hat s_i\hat s_l}{\hat p}-\frac{s_i s_l}{p}\right|
\le t(7/2)+t(3)+t(2)(3)\le13t .
\]
Adding this rank-one approximation to the first part gives the entry
error at most \(88\theta+13(600\theta)\le\Gamma\) in either case.
There were one anchor direction, \(2d\) shifted axes, and at most \(d\)
mixed directions, each with \(m\) queries.

**Mass bound.** On \(\|x\|\le R\) the matrix entry error implies
\(|\hat q(x)-q(x)|\le\Gamma(\sum_i|x_i|)^2\le\delta\).
Use the region defined by \(\hat q(z/\rho)<1\) intersected with
\(\|z/\rho\|\le R\). It contains the dilation by \(1-\delta\) of the
similarly truncated true region, and is contained in the dilation by
\(1+\delta\) of the unrestricted true region. These inclusions use
\(q(sx)=s^2 q(x)\le s q(x)\) for \(0\le s\le1\), as in R1. The mass
of the true truncated region is at least \(\gamma_d(E)-\tau_R\).
The Gaussian dilation inequalities of R1 give the two promised bounds.

For example, at fixed \(\rho\) and desired absolute error \(\eta\le1\),
one may choose \(\rho^2 R^2=2d\log2+4\log(2/\eta)\).
Markov's inequality applied to \(\exp(\|Z\|^2/4)\), whose mean is
\(2^{d/2}\), gives \(\tau_R\le\eta/2\). Choosing \(m\) so that additionally
\(d\delta\le\eta/4\) gives absolute error at most \(\eta\), using
\((1+\delta)^d-1\le2d\delta\) and \(1-(1-\delta)^d\le d\delta\).
This is \(O_\rho(d\log(d/\eta))\) queries; the formal statement gives the
explicit inequality before this choice of \(R,m\).

### Comparison with human results

**Incomparable.** The tangency theorem of Gilbert, Kuo and Sloan discussed
for R1 forbids assuming smoothness after one linear preintegration on
general nonmonotone surfaces. R7 provides an alternative on a low-rank
subclass, approximating the geometry itself with pointwise binary data.
It is not a regularity result for arbitrary surfaces. Liu and Owen's
active-subspace method compared for R5 uses derivative information and
does not give a value-query rate for this binary surface-learning
problem. Here the only geometric information supplied is a curved
ray, and the explicit stability bound persists at a rank change.
The tradeoff is the known center, rank restriction, bounded spectrum,
and curved-ray witness. This addresses part of Appendix §6(1)–(3),
with a genuine approximation guarantee for binary jumps rather than
assuming exact real-valued quadratic observations as in R6.

## 9. Relation to the experiment

R1 isolates the geometry of the closed jumps. The full-rank compact jumps
are the closest instances; even there the spectrum ratio, known scale and
origin condition need not hold. The hard-track data add an unknown smooth
term and sometimes a nonconstant jump amplitude. R2 gives a mechanism by
which an odd background need not mask geometry, but a general even part
does not cancel, and no sample is asserted to satisfy the odd assumption.
R1–R3 concern full-rank ellipsoids; R7 covers rank-deficient centered
binary cylinders given a curved witness. R4–R6 concern inward-facing
PSD caps, not the indefinite or interacting kinks with background in
the experiment. No option family meets these geometric hypotheses,
and none of the supplied synthetic families is assumed to satisfy
all the purity, scale and witness conditions of a certified rule. In
the solver an approximate classification, a fitted surrogate and a
numerical residual correction replace exact membership. In particular
the envelope \(80d\,2^{-m}\) is a uniform worst-case bound, not a prediction
of the errors of these estimates at the fixed budget. The program does not
claim a coverage rate for its diagnostics.

`theory/demo.py` reproduces a separate R3 experiment on an independently
generated 8-dimensional rotated ellipsoid with \(\rho=2.85\),
a full-dimensional trigonometric background, \(L=0.081737\), and 15
scan intervals per signed ray. The only observations used by the
reconstruction and decoder are the real oracle values. The reference
mass uses the generated coefficients and the quadratic-form CDF, not the
observed responses. The 1,036 requested background nodes include 12
deliberately placed on the boundary.

| Bisection steps | Training queries | Relative mass error | Upper relative envelope | Extra decoder queries |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 2,016 | \(1.46\cdot10^{-5}\) | 2.195 | 210 |
| 16 | 2,304 | \(8.37\cdot10^{-7}\) | 0.08085 | 26 |
| 20 | 2,592 | \(5.24\cdot10^{-8}\) | 0.004893 | 13 |
| 24 | 2,880 | \(2.45\cdot10^{-9}\) | 0.0003052 | 12 |

The mass errors obey the envelope and the decoded values agree with
the known background to floating-point precision. The envelope is
conservative; the extra observations at fine precision are mostly the
intentionally ambiguous points. The supplied samples are broader than
this experiment and are not used as evidence of a uniform theorem on
arbitrary unknown amplitudes or surfaces.

The same demo illustrates R6 on rotated pure caps in 64 dimensions.
Ranks 2, 7, and 19 are inferred in 190, 505, and 1,261 point evaluations,
respectively, with relative Frobenius reconstruction errors below
\(1.4\cdot10^{-14}\). The floating-point demonstration chooses a largest
residual pivot and uses a small stopping tolerance; the theorem uses
exact comparisons and requires no finite-precision rank heuristic.

For R7, the demo uses binary queries to centered rank-one and rank-two
cylinders in 64 dimensions. At 34 bisection steps the rank-one case
skips the extra row (4,386 calls), and the rank-two case uses the full
6,562 calls. The maximum entry errors are \(6.80\cdot10^{-10}\) and
\(1.12\cdot10^{-7}\), respectively, below the guaranteed entry envelope.
For comparison, the untruncated quadratic-form CDFs of the fitted
and true regions differ by \(2.41\cdot10^{-11}\) and \(2.32\cdot10^{-8}\)
in relative terms. These CDF numbers are a numerical illustration,
not a floating-point version of the formal mass certificate.

A local run of the numerical solver from a read-only copy, with a
temporary empty home directory, passed all 12 supplied samples. The longest
process took 7.1 seconds in that run. The log-relative-error ranges were
\([-6.37,-5.86]\) for H1, \([-6.24,-4.66]\) for H2,
\([-5.19,-4.51]\) for H3, and \([-5.05,-3.72]\) for H3t.
The call, digital, lower-barrier, and double-barrier sample errors were
\(-3.88,-3.84,-3.16,-2.30\), respectively. This sample score was 83.2;
the retained public snapshot scored 77.9 with no case failures. Both are
development results, not held-out scores. The more accurate synthetic
cases are consistent with precise recovery and a small smooth residual,
but the variable amplitudes and fitted phase tests are not covered by
the certified pure-cap or unit-jump algorithms.
