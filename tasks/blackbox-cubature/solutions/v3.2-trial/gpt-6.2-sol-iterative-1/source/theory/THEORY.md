# Gaussian cubature from value queries

## 1. Submitted algorithm

The executable reads dimension, seed, budget and batch size, then uses only the values returned by
the oracle. It does not receive a family name. All budgets below are **point** budgets; there is no
discount for simultaneous evaluation. The default total budget is \(16\,384\). All coordinates
sent to the oracle are finite doubles in \([-40,40]\). Gaussian designs use inverse-normal
transforms of scrambled Sobol' designs; separate designs receive separate scrambles from the
instance seed. The program emits a point estimate, not a confidence interval.

**Pilot and dispatch.** The first \(768\) queries are in the unrotated Gaussian design.
Two exact levels, each seen at least four times, select the digital method. At least \(8\%\)
of the values at one common level, with additional distinct values, select the option method.
Otherwise, the program probes for a jump or kink on 24 lines. These include random directions
from the origin, directions to representative pilot points, and lines through pairs of pilot
points. It takes 37 points on each line. The largest scaled second differences generate
candidate intervals, which are refined on successive nine-point grids. Comparing the one-sided
linear extrapolants distinguishes a jump (different limiting values) from a kink (different
slopes). The latter need not occur at a zero of the function. A completely flat pilot
instead probes 1024 contracted Gaussian points; it uses discovered positive payoffs for
an importance tilt, or a new full Gaussian design if it cannot find enough.
The initial line search uses \(24\cdot37\) queries, then examines nine
candidate intervals with five nine-point refinements and six side probes each.

**Continuous hard functions.** Near a kink, two forward triangular finite-difference stencils
give quadratic jets on the two sides. A stencil of step \(h=0.00068\) uses the center, \(d\)
steps of length \(h\), \(d\) of length \(2h\), and \(\binom d2\) mixed steps. The centers
are displaced by \(2.8h\) on either side using the local gradient difference. The subtraction
of jets, shifted back to the crossing, estimates the difference between branches, including
the linear and constant coefficients. The smooth background cancels to leading order.
The program fits up to three such differences. It tests signed sums of hinges and min/max
selectors of three branches by how well they cancel third differences on the line scans.
It stops on a good fit or when there are too few queries left for another pair of stencils
and a remainder estimate. With \(p=(d+1)(d+2)/2\) points per stencil, the code reserves
\(2p+2(d+1)+2300\) points before accepting a new jet pair, counting the two
preliminary gradients. It then uses at most 512 new Gaussian points to begin
fitting the remainder, at most 32 forward-gradient stencils of \(d+1\) points,
and up to 1400 scalar training anchors (including the 512 just mentioned).
After fitting, the remaining budget goes to the independent integration design.

Write \(C\) for the chosen control. For a form \(q(z)=z^TAz+b^Tz+c\), diagonalize \(A\)
and rotate \(b\) into its eigenbasis. The control integrals for individual hinges are computed
by inverting the characteristic function
\[
 \phi_q(s)=e^{isc}\prod_j (1-2is\lambda_j)^{-1/2}
       \exp\!\left(-\frac{s^2 b_j^2}{2(1-2is\lambda_j)}\right)
\]
using \(E[q_+]=(E[q]+E|q|)/2\) and
\(E|q|=(2/\pi)\int_0^\infty(1-\operatorname{Re}\phi_q(s))\,ds/s^2\).
For a min/max control the program slices along the leading eigenvector of a sum of branch
gradient-moment proxies. It solves the quadratic equality of each pair of branches on that
line and integrates the active polynomial between consecutive roots with truncated Gaussian
moments. The remaining \(d-1\) dimensions use up to \(2^{19}\) offline Sobol' points
(fewer at \(d=64\)); none query \(f\).

The final estimate adds a rotated Sobol' mean of \(f-C\), with an additional smooth
control when it passes validation. The rotation and Gaussian kernel scales come from
finite-difference gradients of the remainder. The ridge-regularized smooth control has
Gaussian kernel terms, a constant, and homogeneous linear and quadratic kernels.
In particular, for a center \(u\) and diagonal \(D\ge0\),
\[
 E[\exp(- (Z-u)^TD(Z-u)/2)]
  = \det(I+D)^{-1/2}\exp(-u^TD(I+D)^{-1}u/2).
\]
The estimator adds the control integral and subtracts its value on each fresh integration
point. Held-out residual error among pilot points chooses a kernel scale, regularization,
and whether to reuse finite-difference gradients as derivative observations.
The integral of a derivative kernel is the derivative, in its center, of the integrated
kernel above. The final design is separate from the fitted control.

**Closed jump functions.** Near a crossing, the difference of the two value extrapolants
and the gradient difference fit a local jumping term \(D(z)=D_0\exp(\beta^T(z-z_*))\).
A markedly separated value band uses a constant height instead. A fitted background
gives an approximate threshold classifier when the bands overlap. An interior point is
chosen by testing the origin and, if needed, averaging detected interior points.
For a large jump the code uses reciprocal-radius bisection on signed coordinate rays
and comparisons in each coordinate-pair direction. It obtains diagonal and linear terms
from opposing radii, then fits the off-diagonal terms with pivoted low-rank reconstruction
where useful. For moderate jumps it either tracks crossings on shorter random chords and
solves a regularized quadratic regression (\(d<24\)), or tracks the signed coordinate
crossings from value differences, then fits the off-diagonals with comparisons.
The tracking estimates the smooth remainder by interpolation across the bracket.
Eigenvalue truncation removes implausible negative or tiny components. Extra comparison
queries check the disagreement band on independent directions.
The pair-direction method typically retains 3000 to 4300 points for validation
and integration. At most four pivot rows are treated first; remaining batches
are allocated by the number of unresolved pairs. The random-chord method uses
up to 19 refinements on each surviving bracket subject to keeping 3250 points.

The control is \(C(z)=D(z)\mathbf1_{\{\hat q(z)\le0\}}\). Completing the square replaces the
exponential weight by a translation \(Z\mapsto Z+\beta\) and a known multiplier. Its
Gaussian probability is obtained by \(\Pr(q(Z)>0)=1/2+
\pi^{-1}\int_0^\infty\operatorname{Im}(\phi_q(s))\,ds/s\) with the sign chosen for the
interior. Remainders use an independently fitted smooth control and either ordinary
rotated Sobol' sampling or radial strata. In the active eigenspace, the normalized
Gaussian direction is independent of the radius, whose square is \(\chi_k^2\).
For each direction and orthogonal Gaussian complement the roots of
\(\hat q=\pm\text{band}\) supply radial CDF cuts. The weighted remainder uses each
nonempty stratum's conditional Gaussian probability, not its fraction of the total
number of queries. This protects the estimate against an imperfect fitted classifier.
In higher dimension, with a central anchor and a wide band for a moderate jump,
two local bisections per angular point can refine the radial cutoff before allocating
the three remainder strata. The change in control integral along that ray is added
with a short quadrature over the interval between the two cutoffs.

**Digital functions.** For levels \(L<U\), the centered first moment of the values gives
an initial direction, refined by opposite perpendicular perturbations of a bisection
threshold. A pilot of 192 outer slices learns a low-order threshold predictor and
orders the tangential axes by the absolute eigenvalues of a residual second moment.
For each of \(1024\) or \(2048\) fresh outer points, the program queries the predicted
bracket endpoints, expands flagged brackets subject to a reserve, and refines by
comparison. On a valid bracket \([a,b]\), the returned conditional high-level
probability is
\[
  1-\Phi(b)+(\Phi(b)-\Phi(a))\mathbf1_{\{f(y+Tv)> (L+U)/2\}},
  \qquad T\sim N(0,1)\mid a<T<b.
\]
The inner random coordinate is part of the scrambled net. For a genuinely monotone
threshold this removes root-rounding bias; for invalid brackets an approximate threshold
is used. If \(\bar p\) denotes the mean of the slice estimates, the result is
\(L+(U-L)\bar p\).

**Other options.** A pilot with repeated floor values supplies non-floor points for
forward gradients. The average normalized gradient is the leading slice direction; the
gradient second moment orders the complement. Probes on 48 short slices look for
terminations or a positive jump at onset. If consistent with a continuous single onset,
linear extrapolations of the positive payoff at four trial points estimate a slice
threshold. Tabulated 6-to-8 node Gaussian quadrature for \([t_{\mathrm{on}},\infty)\)
then preintegrates the payoff. The tabulation uses a finite discretization of the
standard-normal weight and the eigendecomposition of its small Jacobi matrix. The
tangential mean uses two independent designs and cross-fitted mean-zero Hermite
controls. The main slicing uses up to 1024 outer points, each costing four
threshold probes and the truncated Gaussian quadrature nodes. If the probe
suggests a barrier, the fallback is defensive importance
sampling: it fits a regularized location and covariance to positive payoffs, with
weights proportional to a fractional power of payoff. A rare-positive pilot is
reweighted for the next fit. When validation predicts a smaller weighted second
moment, a histogram in the leading Gaussian probability coordinate replaces its
normal marginal, and tangential means vary affinely with that coordinate. For
original and adapted proposal sizes \(n_0,n_1\), the final estimate subtracts
the observed floor, divides the resulting payoff by
\([n_0+n_1 q(z)/\gamma_d(z)]/(n_0+n_1)\) at every sample, averages, then adds
the floor. Numerical failures that leave a useful budget fall back to rotated Sobol'
sampling. Otherwise the program keeps an already available finite estimate.

The disk/ellipsoid model, one-onset model, and control quadratures are numerical tools,
not assertions that all members of \(\mathcal F_d\) have these structures. In particular,
the program makes no uniform smoothness, unbiasedness, or coverage claim at its
finite floating-point budget.

## R1: Query complexity for offset ellipsoids

**Lean:** `Cubature.translated_ellipsoid_cubature`.

Let \(\gamma_d\) be the product of \(d\) standard real Gaussians. Consider a single
unknown ellipsoid \(E=\{z:q(z-a)\le1\}\), where
\[
    q(x)=x^TAx+b^Tx.
\]
Here \(A\) is symmetric and positive semidefinite, \(b\) and \(A\) are unknown,
and an interior anchor \(a\) is known. Assume \(\|z-a\|_1\le R\) on \(E\) for
a known \(R\ge0\). In particular this is not a theorem for unbounded cylinders.
The integration subclass is \(f=\mathbf1_E\); more generally the theorem recovers
\(\gamma_d(E)\) if \(f(z)>1/2\) on \(q(z-a)<1\) and \(f(z)\le1/2\) on
\(q(z-a)>1\). No restriction on comparison answers is needed on the boundary.
When \(A\) is positive definite this indicator belongs to
\(\mathcal F_d(1,\infty,M,c)\) for some \(M\ge1,c>0\) depending on the ellipsoid.
Indeed the surface \(q=1\) is compact and its gradient is nonzero there:
a zero gradient of a convex quadratic is a global minimum, which cannot exceed
\(q(0)=0\). The derivatives of the defining polynomial are bounded by a fixed
exponential for these parameters, and the two branches are constant.

Consider the \(d^2\) directions \(u_{ij}=e_i\) for \(i=j\) and
\(u_{ij}=e_i+e_j\) otherwise, plus their negatives. Assume known \(0<l\le H\)
bound all the reciprocal positive crossing radii for these directions, and
that \(|a_i|+1/l\le40\) for every coordinate. These conditions are nonempty,
for example \(A=I,\ b=0,\ a=0,\ l=1,\ H=2,\ R=d\).

With \(m\) bisections per signed ray, there is an algorithm using exactly
\(2d^2m\) admissible point queries (even counting redundant ordered pairs).
Set
\[
   \delta=\frac{H-l}{2^{m+1}},\quad
   \eta=3H\delta R^2+2\delta R,\quad
   C=(1+\eta)^d\exp\!\left(\frac{\eta}{2+\eta}\frac{\|a\|_2^2}{2}\right).
\]
For its estimate \(\widehat p\) of \(p=\gamma_d(E)\), it satisfies
\[
                    |\widehat p-p|\le (C-1)p .
\]
The explicit bound implies exponential decay in \(m\), i.e. \(O(d^2\log(1/\epsilon))\)
queries for fixed geometric parameters and dimension as relative tolerance
\(\epsilon\downarrow0\). More explicitly
\(\log C\le(d+\|a\|_2^2/4)\eta\); it suffices to reduce \(\delta\) until this
upper bound is at most \(\log(1+\epsilon)\). There is no \(1/p\) multiplier.
The algorithm's output is the Gaussian mass of an explicit set with known coefficients;
the theorem counts point information, and does not bound the computational cost or
precision of that offline integration. An additional offline absolute error \(\tau\)
adds at most \(\tau\) to the displayed error.

**Proof.** For direction \(u\), let \(\alpha=u^TAu\), \(\beta=b^Tu\), and
\(\rho_\pm=(\pm\beta+\sqrt{\beta^2+4\alpha})/2\). Both are nonnegative, and
\(\rho_+\rho_-=\alpha,\ \rho_+-\rho_-=\beta\).
For \(v>0\), at the point \(a+u/v\) we have
\[
  1-q(u/v)=(v-\rho_+)(v+\rho_-)/v^2 .
\]
The comparison therefore decides whether \(\rho_+\) is above or below \(v\),
with either half-bracket allowed on equality. Starting with \([l,H]\) and using
midpoints halves the bracket \(m\) times. Its midpoint estimates the reciprocal
with error at most \(\delta\) and is in \([l,H]\). Every queried midpoint radius
has reciprocal at least \(l\) and \(|u_i|\le1\), proving admissibility.

For every \(u_{ij}\), estimate \(u_{ij}^TAu_{ij}\) by the product of the two
estimated reciprocals. The error is at most \(2H\delta\), since
\(P Q-pq=(P-p)Q+p(Q-q)\) and all four factors are between \(l\) and \(H\).
On the diagonal this estimates \(A_{ii}\). Off the diagonal subtract the
two diagonal estimates from the estimated directional quadratic and divide by 2;
the error in each \(A_{ij}\) is at most \(3H\delta\). The difference of the
opposing estimates for \(e_i\) estimates \(b_i\) to error \(2\delta\).
Thus for \(\hat q(x)=x^T\hat A x+\hat b^Tx\),
\[
 |\hat q(x)-q(x)|\le 3H\delta\|x\|_1^2+2\delta\|x\|_1 .
\]
The algorithm takes \(\widehat p=\gamma_d(F)\), where
\(F=\{z:\|z-a\|_1\le R,\ \hat q(z-a)\le1\}\).
No definiteness constraint on \(\hat A\) is needed.

Put \(t=1+\eta\) and \(s=t^{-1}\). For \(0<s\le1\), the nonnegative quadratic
part gives \(q(sx)\le s q(x)\). If \(a+x\in E\) then \(\|x\|_1\le R\),
and \(|\hat q(sx)-q(sx)|\le s\eta\), so \(a+sx\in F\). If \(a+x\in F\),
then \(q(x)\le1+\eta\) and \(a+sx\in E\). If \(D_t(z)=a+t(z-a)\),
these two implications give both \(E\subseteq D_t F\) and \(F\subseteq D_t E\).

For any Borel \(S\), a Gaussian density change of variables gives
\[
 \gamma_d(D_t S)=t^d\int_S
       \exp\!\left(\frac{\|z\|^2-\|D_tz\|^2}{2}\right)\gamma_d(dz)
 \le t^d \exp\!\left(\frac{t-1}{t+1}\frac{\|a\|^2}{2}\right)\gamma_d(S).
\]
In each coordinate the exponent bound follows from the polynomial identity
\[
 (t+1)[z_i^2-(a_i+t(z_i-a_i))^2]
   +(t-1)((t+1)z_i-ta_i)^2=(t-1)a_i^2,
\]
and summing. This density-ratio constant is uniform in \(S\), even very far into
the tail. The inclusions now give \(\widehat p\le Cp\) and \(p\le C\widehat p\).
If \(\widehat p\ge p\) use the first inequality; otherwise use
\(p-\widehat p\le(C-1)\widehat p\le(C-1)p\).

**Comparison (incomparable).** Gilbert, Kuo and Sloan, *Preintegration is not
smoothing when monotonicity fails* (2022), Theorem 1, treat the indicator of a
smooth level set at a nondegenerate tangent to the integration line: the defining
function has zero first and nonzero second derivative along the line and a nonzero
transverse derivative. Directional preintegration can then produce a square-root
singularity in the transverse variables. For example, the length of a vertical chord
of the unit disk is \(2\sqrt{1-x^2}\); weighting by a nonvanishing smooth normal
density does not remove the leading square root. Their result concerns the
regularity of straight-line conditional integration, and applies without knowing a
quadric parametrization. R1 gives no such regularity on general surfaces. Instead it
shows that the closed, nonmonotone ellipsoid subclass has a logarithmic
value-query algorithm using rays from the interior and finite geometry.
It addresses parts of Appendix §6.1 (a different treatment of closed surfaces)
and §6.3 (localization from values), not the general nonmonotone smoothing problem.

Matoušek, *Tight upper bounds for the discrepancy of half-spaces* (1995), §1's
half-space discrepancy bound, treats worst half-space counting error for equal-weight
point sets, of order \(n^{1/2-1/(2d)}\) in unnormalized counts for fixed dimension
in the corresponding worst-case setting (that is, \(n^{-1/2-1/(2d)}\) after
normalizing counts). This is not a lower bound on adaptive reconstruction with a
known ellipsoid model and oracle comparisons. Conversely R1 gives no discrepancy
bound, does not act on arbitrary half-spaces, and relies on a supplied anchor and
ray bracket. It does not claim to improve a general QMC theorem.

**Link to experiment.** The large separated jump code uses the same opposing-ray
identities, but learns the anchor, adapts directional brackets, uses single-ray
comparisons for off-diagonals, and removes any remaining error with an independent
remainder estimate. In the unrestricted hard track a smooth background is present
and the subspace may have rank below \(d\); a uniform threshold or the boundedness
hypothesis need not hold. Thus the formal theorem applies to the pure-jump
ellipsoid control problem, not to a promised whole benchmark family. A \(14\)-bit
reconstruction at \(d=8\) is well within the budget, whereas the
same resolution on all ordered rays at \(d=64\) is not. This is why the code
exploits symmetry, low rank, and residual quadrature.

## R2: Robust value-only localization of an internal kink

**Lean:** `Cubature.noisy_internal_kink`.

Along a line, suppose the observed function has the representation
\[
  f(x)=g(x)+J(x-\xi)_++e(x), \qquad J\ne0,\qquad |e(x)|\le\epsilon.
\]
Assume \(|g(x+s)-2g(x)+g(x-s)|\le B s^2\) for all \(x\) and \(s\ge0\).
For instance the last inequality follows for \(g\) with \(B\)-Lipschitz derivative
by writing the second difference as
\(\int_0^s(g'(x+t)-g'(x-t))\,dt\). The assumptions need only be used on
the starting interval; the formal version uses uniform hypotheses on \(\mathbb R\).
There is no bound on the value \(g(\xi)\) in the result, and an arbitrary affine
function can be added without affecting the rule. In the noise-free experiment
\(\epsilon=0\); positive \(\epsilon\) allows deterministic error at every observation.

Let \(c\) be a given center and \(h>0\), with \(|\xi-c|\le3h/2\).
Fix \(m\) such that for \(s_j=h 2^{-j}\), \(0\le j<m\),
\[
                    8(Bs_j^2+4\epsilon)\le |J|s_j .
\]
There is a deterministic value-query rule returning \(\widehat \xi\) with
\(|\widehat\xi-\xi|\le 3h\,2^{-m}/2\) after exactly \(5m\) evaluations,
all in \([c-2h,c+2h]\). This counts repeated points; they could be cached.
When \(\epsilon=0\), a starting bracket with \(8Bh\le|J|\) permits any number of
rounds. If \(\epsilon>0\), refinement stops being certified once the
signal-to-noise condition fails. If the initial line segment is in the admissible
query box, so are all queries.

**Proof.** With current center \(c_j\) and spacing \(s=s_j\), query
\(c_j-2s,c_j-s,c_j,c_j+s,c_j+2s\). At the three interior centers \(x\),
take \(S(x)=f(x+s)-2f(x)+f(x-s)\). Choose \(c_{j+1}\) at the maximum
of \(|S|\), with ties settled in line order, and halve the spacing. The exact
second difference of the unit hinge is \((s-|x-\xi|)_+\), as is seen by
splitting the interval at the kink. Thus
\[
 |S(x)-J(s-|x-\xi|)_+|\le B s^2+4\epsilon=D.
\]
The invariant \(|\xi-c_j|\le3s/2\) implies some interior center is within
\(s/2\) of \(\xi\), giving a score at least \(|J|s/2-D\).
At the selected center that score is at most
\(|J|(s-|c_{j+1}-\xi|)_++D\). Consequently
\((s-|c_{j+1}-\xi|)_+\ge s/2-2D/|J|\ge s/4\).
It follows that \(|c_{j+1}-\xi|\le3s/4=3s_{j+1}/2\), establishing the
invariant at the next scale. Since \(|c_{j+1}-c_j|\le s\), the next
query interval is contained in this one. Induction gives the location bound,
the domain bound, and the stated count.

**Comparison (incomparable).** Plaskota and Wasilkowski, *Adaption allows
efficient integration of functions with unknown singularities* (2005),
the single-unknown-singularity adaptive quadrature result (Appendix §5),
studies piecewise \(C^r\) functions on a bounded interval, with uniform
smoothness control away from the unknown singular point. The result is an
integration rate \(O(n^{-r})\) by adaption, against a nonadaptive lower
rate no faster than \(n^{-1}\) in that setting. It addresses a more global
problem and broader branch regularity. R2 is about localization once a
single kink is bracketed, assumes a quantitatively visible derivative jump,
and covers bounded perturbations of the sampled values. Its logarithmic
location cost is not a faster integration rate. Bayer, Ben Hammouda and
Tempone, *Numerical smoothing with hierarchical adaptive sparse grids and
quasi-Monte Carlo methods for efficient option pricing* (2023), the
one-dimensional numerical smoothing construction, use root finding on a
specified switching function and then smooth integration on the two sides.
They target the cost and error of pricing after smoothing. R2 needs neither
the switch nor its zero level, but supplies neither a global bracket nor an
outer-integrand smoothness theorem.

**Link to experiment.** This addresses part of Appendix §6.3. The production
scanner uses the same largest-second-difference mechanism on nine nodes,
with a subsequent interpolation and classification step, rather than
the five-node robust variant analyzed here. On a transversal line through
one isolated continuous kink with smooth branches, subtraction of the derivative
jump times a hinge makes the derivative continuous. Bounded branch second
derivatives in a neighborhood give the required second-difference bound there.
The required starting scale and distance from other crossings are not
guaranteed when several surfaces nearly intersect. This explains both why
the internal kink can be located away from a zero and why a coarse scan
can miss a sufficiently weak or clustered kink.

## R3: An adaptive lower bound without an interior witness

**Lean:** `Cubature.hidden_ball_relative_lower`,
`Cubature.hidden_ball_randomized_relative_lower`.

For each \(d\ge 1\), let the possible centers be \(s\in\{-1,1\}^d\), and let
\[
 f_{s,A}(z)=A\,\mathbf1_{\{\|z-s\|_2<1/2\}},\qquad A\in\{1,2\}.
\]
For every deterministic adaptive algorithm using \(N<2^d\) point evaluations,
some member of this family has a positive integral and relative error at least
\(1/3\). The query locations can even be unrestricted. The center and height
are unknown, but the radius and all candidate centers may be given to the
algorithm. Allowing variable stopping with an upper bound \(N\) does not help:
pad the strategy and have the decoder disregard observations after its stop.
There is also a finite-randomization statement: for a uniformly chosen seed
among \(m>0\) possibilities, there is one fixed input on which the probability
of relative error at least \(1/3\) is at least
\((2^d-N)/(2\cdot2^d)\) when \(N\le 2^d\). Thus to guarantee a failure
probability at most \(\alpha<1/2\) for all inputs at any tolerance below
\(1/3\), such a strategy needs \(N\ge(1-2\alpha)2^d\) queries.

**Proof.** The \(2^d\) open balls are disjoint. Run the query strategy on the
transcript with every answer zero. At most \(N\) balls contain one of its query
points. Choose a ball that contains none. For either of its two heights, the
actual transcript is all zero by induction on the query number, so the
algorithm makes one common estimate \(u\). The integrals are \(p\) and \(2p\),
where \(p=\gamma_d(B(s,1/2))>0\) by the strictly positive Gaussian density.
If both relative errors were less than \(1/3\), the first would imply
\(u<4p/3\) and the second would imply \(u>4p/3\), a contradiction.
For the finite-seed version, fix a seed. At least \(2^d-N\) balls are
untouched on its all-zero transcript, and at least one height is bad for
each. Across \(m\) seeds the number of bad (seed, input) pairs is therefore
at least \(m(2^d-N)\). Average over the \(2\cdot2^d\) inputs; some input has
at least that average number of bad seeds. Lean states this as a count
inequality before division, with natural-number subtraction to cover
the trivial \(N>2^d\) case.

These are members of the Appendix class with \(K=1,r=\infty,M=8,c=1\)
for all \(d\ge1\). For \(\varphi(z)=(\|z-s\|^2-1/4)/(d+1)\), the zeroth
derivative is bounded by \(2\|z\|^2+2+1/4\le 8e^{\|z\|}\),
the first partials by \(2(1+\|z\|)\), and the second partials by 2.
All higher derivatives vanish. The branches are constant. At a boundary point
\(\|\nabla\varphi\|=1/(d+1)\), and \(\|z\|\ge\sqrt d-1/2\) gives
\((d+1)e^{-\|z\|}\le 2e^{1/2}<8\), using \(e^{\sqrt d}\ge1+d/2\).
The Lean statement specifies the actual payoffs, oracle transcript and
Gaussian integral. It does not rely on axiomatizing access to these payoffs
or assuming their integrals positive.

**Comparison (incomparable).** Hinrichs, Novak, Ullrich and Woźniakowski,
*The curse of dimensionality for numerical integration of smooth functions*
(2014), Theorem 1, study deterministic worst-case **absolute** error on
smooth function classes on volume-one domains with derivative bounds.
Their curse-of-dimensionality results show that bounded smoothness alone
does not remove exponential dependence on \(d\) in that problem. R3 gives
an explicit obstruction on Gaussian space for an otherwise known compact
jump shape, even with constant analytic branches. Its normalization is
relative and the ball masses can be very small, so it is not an improvement
to the absolute-error curse. The finite-seed extension is a worst-case
relative result for this particular family, not a randomized complexity
bound for general smooth functions.
The lower bound isolates the cost of discovering a witness that R1 takes
as input. This is a restricted answer to Appendix §6.4 about adaptive
worst-case information. Since the benchmark controls its data distribution
and integral scale, \(2^d\) is not a lower bound for its typical cases.

## Sample results

In the final local check, all 12 samples passed. The table uses the problem's
\(\log_{10}\) relative error and reports the observed range within each family
(a single number for one sample).

| Family | Samples | Log-relative error |
| --- | ---: | ---: |
| H1 | 2 | \(-4.57\) to \(-4.01\) |
| H2 | 2 | \(-4.48\) to \(-3.81\) |
| H3 | 2 | \(-5.08\) to \(-3.81\) |
| H3t | 2 | \(-4.27\) to \(-3.63\) |
| Local-volatility call | 1 | \(-2.85\) |
| Local-volatility digital | 1 | \(-3.70\) |
| Down-and-out call | 1 | \(-2.19\) |
| Double-knockout call | 1 | \(-1.82\) |

The rare-jump gains are consistent with the usefulness of geometric reconstruction
without dividing a coefficient error bound by a tiny event probability. They
are not a test of R1's deterministic bound, since the production method also
estimates the smooth background, the jump height, and the anchor. The two
H2 errors show that nonzero kink levels did not prevent a useful control on
these samples, but are not guarantees of localization for lines near a surface
intersection. The one-barrier option is an example where a terminal direction
does not eliminate the pathwise discontinuities and the submitted estimate
was worse than GPCA-RQMC on the sample.

## Evaluation protocol

`python3 /workspace/tools/local_eval.py` runs the submission against the supplied
black-box samples. `python3 /workspace/tools/lean_check.py` checks theorem statements
against proofs, with only `propext`, `Quot.sound`, and `Classical.choice` allowed.
The public score is diagnostic only; the final held-out score is not observable
during development.
