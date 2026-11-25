# Learning Gaussian cubature controls from values

The two sources of information in this problem are different. A continuous
piecewise-quadratic function reveals its branch difference through matched
differences on the two sides of a kink. A constant jump reveals only membership,
but membership at *chosen* radii is enough to reconstruct a quadratic surface.
In both cases, fitting the control and integrating the control are separate
tasks. The fitted geometry need not be exact for a residual estimator to be
valid.

The arguments below use exact values and continuous uniform random variables
unless stated otherwise. All query counts include probes. Parameter constants
in a theorem are assumptions of that theorem, not constants known to the
program on an arbitrary instance.

## 0. Algorithm in the submitted program

The protocol, query limits, and seeded randomization live in
`protocol.py`, `solver.py`, and `cubature.py`. A normal block is
\(\Phi^{-1}\) applied componentwise to a 40-bit linearly scrambled
and digitally shifted Sobol' block, optionally followed by an
orthogonal matrix. Non-power-of-two sizes use a sum of independent
power-of-two blocks. The program never uses instance names.

**Routing and fallback.** Start with 256 normal points. If any
payoff is zero, treat it as a zero-masked payoff. Equal nonzero
pilot values trigger the digital algorithm; other zero-masked
values trigger the gradient and payoff algorithms below.
Otherwise query 32 lines of 25 points, of which half have
transverse origins closer to zero. Try a jump model, then kink
models, and, if needed and enough budget is left, a second search
for a large rare jump. Failed checks can return to rotated
Gaussian quadrature. The generic rotation uses up to 24 forward
finite-difference gradients at fresh Gaussian points, costing
\(d+1\) each and leaving at least 8,192 points when possible.
The eigenvectors of their second-moment matrix order the net.
The protocol runner clips coordinates to the legal cube,
tracks the total, and retains the pilot mean for a time or
numerical emergency. It returns only the estimate.

**Kinks (`kinks.py`, `smooth.py`).** Rank the atlas by fourth
differences. Around a candidate use two one-sided polynomial
extrapolants to estimate a crossing, and check that their
difference has a stable root under two finer extrapolations.
Forward gradients just to either side propose a normal.
Matched stencils at offsets \(\pm8\cdot10^{-5}\) give the
branch-difference quadratic. Its Hessian in the normal frame
uses a \(2.5\cdot10^{-5}\) normal step and \(1.7\cdot10^{-3}\)
tangential steps. Compute columns in batches, using a
spectral low-rank completion only if the extra measured
columns and diagonal agree; otherwise continue to the dense
stencil if the budget permits. Spend at most three stages
with at most thirteen candidate tests per stage, reserving
approximately 4,900 queries. Select a sum of signed hinges
or a signed three-piece max by the residual fourth differences
on the atlas. Choose the equivalent polynomial origin of that
representation with smallest pilot remainder variance.
Integrate a hinge by the one-dimensional Fourier formula
in Section 6. For a max of two positive parts use the sum
minus the overlap. A line integral of the overlap is
piecewise quadratic (Section 5); average it on \(2^{19}\)
surrogate points, or \(2^{18}\) for \(d\ge40\), costing no
oracle queries.

For the smooth remainder, take a curvature stencil at the
origin when affordable (or use a patch Hessian or a small
range sketch). Query eleven Gauss-Hermite sites per proposed
eigenvector and subtract the resulting additive Hermite
polynomial if a separate 128-point check supports it.
The mean of this control is its Hermite constant. The final
normal residual is taken in antithetic pairs, and, when at
least 2,200 queries remain at the start of correction, uses
two independently scrambled groups with cross-fitted
zero-mean degree-two and degree-four features. The mean
of the fitted nonsmooth control is added exactly once.

**Closed jumps (`ellipsoid.py`, `response.py`).** A persistent
jump in a refined value bracket supplies an interior candidate
and two near-boundary extensions. Fit their gap by \(L
e^{\ell^\top z}\), estimating \(\ell\) by the gradient difference
divided by the gap. Very large jumps use \(\ell=0\).
A bounded linear baseline predictor gives binary labels.
Extra small-norm, normal, and expanded probes help find
interior points; a second search is available for a rare
jump missed on the atlas. In a random frame with the tilt
spread evenly across its axes, use \(2d\) axial rays, with
20 narrowing comparisons per ray for \(d<40\), 17 otherwise.
If the observed jump cannot be verified at a measured exit,
scan that ray for the gap and refine it. Paired directions
measure the normalized off-diagonal entries using the
reciprocal-radius formulas in Section 4.

Start with a leading block of up to ten coordinates at
14 bits for \(d<40\) or 11 for \(d\ge40\); in sufficiently
anisotropic larger cases extend it to eighteen at 12 bits.
When a low-rank fit is supported, complete the Gram matrix
from selected columns at 8 to 18 comparisons per pair.
Otherwise distribute the remaining shape budget over all
pairs (at least 3 and at most 16 comparisons each) and
project the model to positive semidefinite shape.
The shape reserve for the final stages is 5,300, 5,000,
or 3,200 points as the dimension increases. Project the
linear part onto the fitted active space and check labels
against the pilot. For some large jumps with only an
approximate active span, remeasure important spectral rays
before integrating the model.

The ordinary estimator is (3): analytic model probability,
randomized label correction inside a band in fitted quadratic
value, and antithetic smooth correction including the
*entire* outside-band residual. The band depends on
measured coefficient widths and nearby pilot disagreements.
Use up to 2,048 angular points within the remaining band
allowance, with either one uniform radial query, or one
or three comparison probes followed by the defensive
draw (4). The defense mass in this draw is 0.18. Large,
nearly constant jumps can use a separate full radial
correction, with Gaussian angular reweighting if zero is
verified inside; accepted dense, high-dimensional large
jumps may instead use the first-variation correction of
Section 7. These two special routes subtract the label
model everywhere, and require that the geometry and labels
really describe the jump. They are not the general
full-support band identity. The smooth control and final
residual use the remaining points.

**Digital (`options.py`).** Augment the pilot with 512
Gaussian points and two panels of 256 near the current
logistic boundary. Condition in the resulting separating
direction. Train a quadratic threshold model on 256
transverse points, with two endpoint checks and seven
cuts per point. Regress the approximate roots on affine,
radial, and low-rank quadratic features. Its conditional
probability is a control: the mean is the probability of a
quadratic inequality with one independent linear coordinate.
In the final transverse nets each row costs two endpoint
checks, three or four cuts in Gaussian probability
coordinates, and one defensive correction, with whole-line
probability 0.032. Use the final average minus the model
probability at those same transverse points, then add its
numerically integrated mean.

**Other options (`payoff_axes.py`, `regression.py`, `rare.py`).**
Use at most 16 forward gradients on positive-payoff points
spread across payoff quantiles; add a 512-point search if
the first pilot has fewer than 13 positives. Normalize
gradients, use their mean as the leading direction, and
diagonalize their second-moment matrix in its complement.
For a positive fraction at least 0.13, optionally refine
the axes with 20 further gradients and query a 512-point
scale pilot (half shifted along the main direction).
Their squared payoffs and a pulse near the estimated
exercise level choose a positive piecewise-linear density
in the first uniform coordinate; all other coordinates
stay standard normal. The weight is the reciprocal
density. The two final groups of nets each fit the control
for the other group. The control is a ridge fit of cubic
B-splines in the leading Gaussian coordinate multiplied
by centered Legendre products in a few transverse
probability coordinates, plus lower-order normal features
in the remaining coordinates. Its mean uses the spline
mean and only the constant transverse feature; spline
means are computed by Gauss-Legendre quadrature with the
Gaussian weight and the constant continuations in the
two tails.

For a positive fraction below 0.13 and enough observed
survivors, draw adaptation panels of \(2,048+256\) and
\(1,024+256\) points from a fitted Gaussian plus a standard
Gaussian defense. Reweighted positive values estimate
the mean and covariance of a payoff-weighted Gaussian;
regularize the variance in the secondary directions
by the effective number of positive paths. Final fixed
counts, normally 512 from the standard Gaussian and
the rest from the adjusted Gaussian, are combined with
the actual mixture likelihood in the denominator.
If validated on the second panel, subtract a logistic
survival times a ridge payout control and add its
mean estimated on eight cheap \(2^{14}\)-point nets.
None of these fits assumes a known diffusion or a
particular time discretization.

## 1. Scope and notation

Write \(\phi,\Phi\) for the standard univariate normal density and CDF,
and use \(q(z)=z^\top A z+b^\top z+c_0\), with \(A=A^\top\). The norms
\(\|\cdot\|,\|\cdot\|_F\) on matrices are operator and Frobenius norms.

Three subclasses and local hypotheses are used:

* **Quadratic differences.** Near a regular face, the two continuations are
  \(g+p_-\) and \(g+p_+\), where \(p_\pm\) are quadratics, \(g\) has a
  Lipschitz Hessian on a neighborhood of the probes, and \(q=p_+-p_-\)
  changes sign at the face. For global statements the background and pieces
  obey the growth assumptions of \(\mathcal F_d(K,r,M,c)\), with \(r\ge3\).
  At the continuous faces the function takes the common trace, not an
  arbitrary exceptional value.
* **Ellipsoid membership.** \(f(z)=B+J1_{q(z)<0}\), \(J>0\);
  \(q(a+x)=x^\top A x+b^\top x-1\), where an interior point \(a\) is given,
  \(0<\lambda_-\le\lambda_{\min}(A)\le\lambda_{\max}(A)\le\lambda_+\),
  and \(|b|\le B_b\). The portions of the rays used below must lie in the
  legal query cube. A cylindrical version specifies the rank and valid
  measured rays instead of a global lower eigenvalue. For the *integration*
  identities the jump can be \(D(z)=L e^{\ell^\top z}\), and the smooth
  background need not be known. For *binary reconstruction* the labels
  must be correct.
* **Envelope.** \(g+\max_{1\le j\le m}p_j\) with distinct quadratics
  \(p_j\), or a finite continuous expression made from their max and min.
  Active faces are among \(p_i-p_j=0\). The regularity proof works even at a
  double root on an integration fibre. When placing the model in
  \(\mathcal F_d\), require regularity and the specified derivative bounds
  for the defining functions and use \(K\le m(m-1)/2\).

For any fixed finite coefficients, polynomials of degree at most two and
exponential-affine functions with \(\max_i|\ell_i|\le1\) admit the
all-orders upper growth bounds for suitable \(M,c>0\). Nondegeneracy must
still be checked on each face. Nothing below assumes that arbitrary analytic
functions have the appendix's *uniform in derivative order* bounds.

## 2. An internal kink can be located without zero detection

**Proposition 1 (value-only location).** On an interval let \(F\) be
continuous, with just one interior kink \(\tau\), two \(C^2\)
continuations, \(|F_\pm''|\le L\), and
\(|J_\tau|=|F'_+(\tau)-F'_-(\tau)|\ge J>0\).
Choose a mesh of width \(h\) with \(Lh<J/8\), with \(\tau\) at
least \(2h\) from the endpoints. Tests of
\[
 |F(x+h)-2F(x)+F(x-h)|>Jh/4
\]
on that mesh find a bracket of width \(2h\) containing \(\tau\). Given
a bracket of width \(w_0\le J/(48L)\), at distance at least \(2w_0\)
from the original endpoints, four new values per stage suffice for width
\[
 w_j\ \le\ (24L/J)^{-1}\big((24L/J)w_0\big)^{2^j}.
\]
For \(L=0\) the first refinement is exact. In particular the refinement
cost is \(O(\log\log (1/\varepsilon))\) for fixed positive \(J,L,w_0\).

**Proof.** Subtract \(J_\tau(x-\tau)_+\) from \(F\). The result has
continuous first derivative, which is \(L\)-Lipschitz: on a segment
crossing \(\tau\), add the two one-sided Lipschitz estimates at \(\tau\).
The second difference of this result is bounded in absolute value by
\(Lh^2\), by integrating its first derivative on the two half intervals.
The subtracted hinge contributes exactly \(J_\tau(h-|x-\tau|)_+\).
Hence a positive test implies \(|x-\tau|<h\), and the closest grid
center, at distance at most \(h/2\), tests positive.

For a bracket \([l,u]\) of width \(w\), let \(P_-\) be the secant
extrapolant from \(l-2w,l-w\), and \(P_+\) the one from \(u+w,u+2w\).
The interpolation remainder at \(\tau\) is bounded by
\((L/2)(2w)(3w)=3Lw^2\) for each side (the formula holds also for
extrapolation, by Rolle's theorem). The difference of the secant slopes
differs from \(J_\tau\) by at most \(6Lw\), by the mean-value theorem.
Thus the intersection \(t_*\) of the secants satisfies
\[
 |t_*-\tau| \le \frac{6Lw^2}{J-6Lw}\le 12Lw^2/J .
\]
Intersect the old bracket with the interval of that radius about \(t_*\).
Its width is at most \((24L/J)w^2\). The endpoint and width conditions
are inherited because the brackets are nested. Iteration gives the stated
bound. Neither test uses the level \(F(\tau)\). \(\square\)

The mesh stage is for a prescribed line and a separated kink. If several
kinks are separated by more than \(4h\), each second-difference window
contains at most one, and the corresponding positive tests can be grouped
by their overlapping brackets. This does not solve the problem of finding
an informative line, nor does it cover a tangency with vanishing derivative
jump. In the solver, fourth differences and two one-sided polynomial
extrapolants are empirical versions of the detector and the refiner.

## 3. Two-sided quadratic and subspace recovery

**Proposition 2 (matched Hessian probes).** In a patch as in Section 1,
let \(p\) be its crossing point and take an orthonormal basis \(u_i\)
with \(u_1=n\), pointing into the plus side. Set \(x_\pm=p\pm a n\),
and let \(s_i=h_i u_i\), \(h_i>0\). Suppose all of
\(x_\pm,x_\pm+s_i,x_\pm+s_i+s_j\) (including \(i=j\))
are in their respective open patches and let \(H_g\) be
\(L_3\)-Lipschitz in operator norm on a convex neighborhood containing
the stencils. Then
\[
 \widehat H_{ij}
 = \frac{\Delta_{s_i}\Delta_{s_j}f(x_+)
             -\Delta_{s_i}\Delta_{s_j}f(x_-)}{h_i h_j}
\]
satisfies \(|\widehat H_{ij}-u_i^\top (2A)u_j|\le 2aL_3\),
where \(q=p_+-p_-\). This uses \((d+1)(d+2)\) values; in
particular it holds for indefinite \(A\). If the smallest nonzero
absolute eigenvalue of \(2A\) is \(\lambda>2adL_3\) and
\(\widehat U\) spans the corresponding number of leading absolute
eigenvectors of \(\widehat H\) in the ambient frame, then, with \(P\)
the true range projector,
\[
 \|(I-P)\widehat U\|\le \frac{2adL_3}{\lambda-2adL_3}.
\]

**Proof.** A double difference of a quadratic is exactly its mixed
second derivative multiplied by \(h_i h_j\). A double difference of
\(g\), divided by \(h_i h_j\), is the integral over \([0,1]^2\)
of \(u_i^\top H_g(x+t s_i+w s_j)u_j\). In the two integrals the
arguments differ by exactly \(2a n\). This proves the entry bound
without a term proportional to the transverse step. The operator norm
error is at most \(2adL_3\). With \(E=\widehat H-2A\) in the
ambient frame, \((I-P)\widehat U\widehat\Lambda=(I-P)E\widehat U\);
the smallest singular value of \(\widehat\Lambda\) is at least
\(\lambda-\|E\|\), by the singular-value minmax characterization
and the triangle inequality. Multiply on the right by
\(\widehat\Lambda^{-1}\). Finally, each branch uses
\(1+d+d(d+1)/2\) distinct stencil sites. \(\square\)

For completeness, the gradient at the crossing can use the same
probes. Denote the individual difference Hessians by \(H_\pm^\mathrm{fd}\).
Estimate component \(i\) of \(\nabla q(p)\) by
\[
 \frac{\Delta_{s_i}f(x_+)-\Delta_{s_i}f(x_-)}{h_i}
 -\frac{h_i}{2}\widehat H_{ii}
 -a(H_+^\mathrm{fd}+H_-^\mathrm{fd})_{i1}.
\]
The polynomial terms are exact. For the first term's common background
contribution, integrate \(2a\,u_i^\top H_g n\) along the segment
between the parallel first-difference stencils and compare with its
value at \(p\). The error is at most \(2aL_3(a+h_i)\).
The middle term contributes at most \(a h_i L_3\). Each
\(H_\pm^\mathrm{fd}\) entry in the last term differs from the
corresponding Hessian at \(p\) by at most \(L_3(a+h_i+h_1)\).
The background Hessian terms at \(p\) cancel. This gives an error
at most \(aL_3(4a+7\max_i h_i)\) per component.

One sufficient stencil construction is as follows. For a true unit normal
\(n_*\) and \(|\nabla q(p)|=\gamma>0\), use \(n\) with
\(\|n-n_*\|\le\eta\), \(h_1\le a/8\), and transverse steps at most
\(h\). The sign of the linear part on each stencil point is the desired
one with magnitude at least
\(\gamma(a(1-\eta)-a/4-2h\eta)\).
If this exceeds \(\|A\|(a+2\max(h,h_1))^2\), the sign is unchanged
by curvature. If the estimated \(p\) has error \(\delta\), replace
the bound on the linear part by that bound minus \(\gamma\delta\),
and add \(\delta\) inside the square. With an isolated transversal
line, Proposition 1 gives \(\delta=o(a)\). At distance \(a_0\)
along that line on either side, forward coordinate differences
of step \(h_0=o(a_0)\) give the gradient difference with error
\(O(\sqrt d(a_0+h_0+\delta))\) under a local Hessian bound:
each component is an average of the true derivative at arguments
within \(a_0+h_0+\delta\) of the crossing. The normal is obtained by
normalizing the gradient difference, with an angular error bounded
by twice the gradient error divided by \(\gamma\) when that error is
less than \(\gamma/2\). Choosing \(h=O(\sqrt a)\) with a sufficiently
small constant and \(a_0=O(a)\) satisfies the stencil conditions
for sufficiently small \(a\).

For fixed \(d\) and patch constants, \(a=N^{-\beta}\) for any
fixed \(\beta>0\) therefore costs \(O(d^2+\log\log N)\) queries
after the initial line search and gives \(O(N^{-\beta})\) Hessian
and subspace error. This is an exact-value result, not a recommendation
to send a floating-point difference step tending to zero. The program
uses fixed steps and rejects inconsistent crossings. The inferred
quadratic's subspace may also include a component of \(b\) outside
\({\rm range}(A)\); this is not estimated by the Hessian alone,
but is available from \(\nabla q(p)-2Ap\).

An immediate control-error bound useful below is
\[
 \|q_+-\widetilde q_+\|_{L^2(\gamma_d)}^2
 \le 2\|A-\widetilde A\|_F^2+
 |b-\widetilde b|^2+
 (c_0-\widetilde c_0+\operatorname{tr}(A-\widetilde A))^2 .
\]
Indeed \(s\mapsto s_+\) is 1-Lipschitz, the variance of a
centered Gaussian quadratic is twice the sum of squared eigenvalues
(diagonalize and use the fourth normal moment 3), and its covariance
with a linear form vanishes by oddness. Similar bounds for a finite
sum use Minkowski's inequality; for a max or min use that the
pointwise error is bounded by the maximum of the piece errors.

## 4. Learning and integrating a closed jump surface

**Proposition 3 (binary tomography).** Under the positive-definite
ellipsoid hypotheses, let
\[
 r_-=\frac{2}{\sqrt{B_b^2+4\lambda_+}+B_b},\qquad
 r_+=\frac{\sqrt{B_b^2+4\lambda_-}+B_b}{2\lambda_-}.
\]
There is a deterministic membership algorithm with
\(O(d^2\log(Cd/\varepsilon))\) queries returning
\(\|\widetilde A-A\|_F+|\widetilde b-b|\le\varepsilon\);
here and below \(C\) in this proposition depends on the given
spectral and linear bounds. For the ellipsoid jump integral with
known \(L,\ell\), the induced absolute error is at most
\[
 L e^{|\ell|^2/2}
 \frac{|S^{d-1}| r_+^{d-1}}{(2\pi)^{d/2}}\,
 \frac{r_+^2\|\widetilde A-A\|+r_+|\widetilde b-b|}
 {2\sqrt{\lambda_-}},
 \tag{1}
\]
if the fitted parameters also obey the stated bounds.

**Proof.** On a unit ray \(u\) from \(a\), put
\(a_u=u^\top A u,\ b_u=b^\top u\).
The positive root is
\(r(u)=2/(\sqrt{b_u^2+4a_u}+b_u)\), in the stated bracket.
Along that ray membership is a threshold; bisection of the radial
interval gives error \(\delta\) with at most
\(\lceil\log_2^+((r_+-r_-)/\delta)\rceil+1\) tests. At an exact
boundary value either branch of the bisection preserves a closed
bracket. Project reported radii onto \([r_-,r_+]\).
Writing \(s(u)=1/r(u)\), the reciprocal equations are
\(s(\pm u)^2\mp b_u s(\pm u)-a_u=0\). Hence
\[
 b_i=s(e_i)-s(-e_i),\qquad A_{ii}=s(e_i)s(-e_i),\qquad
 A_{ij}=s(u_{ij})^2-s(u_{ij})b_{u_{ij}}-(A_{ii}+A_{jj})/2,
 \quad u_{ij}=(e_i+e_j)/\sqrt2 .
\]
Each reciprocal error is at most \(\delta/r_-^2\). The above
formulas, using \(|s|\le1/r_-\) and \(|b|\le B_b\), give a
max-entry error bounded by \(C\delta\) for both parameters.
Taking \(\delta\le\varepsilon/(2Cd)\) gives the norm result.
Projection onto the spectral interval and \(b\)-ball, in the joint
Euclidean \((A,b)\) norm, cannot increase distance to the true
parameter; this follows by expanding squared distances and using
the variational inequality for the nearest point in a convex set.
This changes at most the constant in the sum-norm bound.

Interpolate linearly between the two parameter pairs. On each
intermediate boundary the radial derivative is
\(\sqrt{b_u^2+4a_u}\ge2\sqrt{\lambda_-}\). Implicit
differentiation along the interpolation gives the radial difference
bound in (1). Complete the square in \(e^{\ell^\top z}\,d\gamma(z)\)
to obtain \(e^{|\ell|^2/2} d\gamma(z-\ell)\). Around \(a\), the
shifted Gaussian volume element per unit solid angle is at most
\((2\pi)^{-d/2}r_+^{d-1}\) on a differing radial interval.
Integrate over the sphere. \(\square\)

For a rank \(k\) PSD cylinder the same entry formulas apply on
rays with two finite measured axial radii and finite paired radii;
assume all these radii have positive lower and finite upper bounds.
If a leading \(k\) by \(k\) principal block \(B\) is positive
definite, it is enough to measure the first \(k\) columns \(C\).
The identity \(A=CB^{-1}C^\top\) follows by a block congruence
eliminating the off-diagonal, after which the lower Schur complement
has rank zero. With \(\lambda_{\min}(B)\ge\lambda_0\),
\(\|\Delta B\|\le\lambda_0/2\),
\(\|\widetilde B^{-1}\|\le2/\lambda_0\) and
\(\|\widetilde B^{-1}-B^{-1}\|\le2\|\Delta B\|/\lambda_0^2\).
Expanding \(\widetilde C\widetilde B^{-1}\widetilde C^\top-CB^{-1}C^\top\)
thus bounds the error by
\[
 \frac{2}{\lambda_0}(2\|C\|+\|\Delta C\|)\|\Delta C\|
 +\frac{2\|C\|^2}{\lambda_0^2}\|\Delta B\|.
\]
This gives a certified \(O(d(k+1)\log(C'd/\varepsilon))\)
reconstruction under the pivot and ray hypotheses. It explains the
compression used in the code, which checks a numerical nullspace
and diagonal consistency, rather than claiming to certify all
unmeasured entries. The theorem (1) applies to cylinders in their
active subspace if the subspace is known and shared by both
quadratics; it is not asserted with \(\lambda_-=0\) in the full
space.

**Proposition 4 (geometry needs adaptive queries less often than
nonadaptive ones).** In the above membership problem let the
admissible \((A,b)\) contain a \(D=d(d+3)/2\)-dimensional Euclidean
ball of radius \(R>0\), using Frobenius coordinates for \(A\).
To return the parameters to joint Euclidean error \(\varepsilon<R/4\)
with probability at least \(3/4\) on every parameter, a randomized
adaptive algorithm needs
\[
 n\ge D\log_2(R/\varepsilon)+\log_2(3/4).
\]
A randomized nonadaptive algorithm needs
\(n\ge (D/e)(R/\varepsilon)(3/4)^{1/D}\).
These are lower bounds for reconstruction, not for the scalar
integral on the whole parameter ball.

**Proof.** Put the uniform prior on the ball and fix the seed.
An adaptive binary decision tree has at most \(2^n\) leaves.
Each leaf's output can be accurate on at most an
\(\varepsilon\)-ball of parameters, of volume ratio at most
\((\varepsilon/R)^D\). In the nonadaptive case each response is
the sign of one affine functional of the parameters, so at most
\(\sum_{j=0}^D {n\choose j}\) full-dimensional sign cells
can occur. The cell count follows by inserting hyperplanes one at
a time: the cells split at insertion are bounded by the number of
cells on the inserted hyperplane, in dimension \(D-1\). This
gives the binomial recurrence, including degenerate arrangements
by the same upper bound. Hyperplanes themselves have prior
measure zero. If \(n<D\), the adaptive bound already rules out
the requested probability. For \(n\ge D\), multiplying each
summand by \((D/n)^{j-D}\ge1\) bounds the sum by
\((n/D)^D(1+D/n)^n\le(en/D)^D\).
The resulting average success bounds hold for every seed, hence
after averaging over seeds. \(\square\)

For the scalar integral there is also an elementary strict gap on
the one-parameter *closed* subfamily
\(A=R_0^{-2}I,\ b=0,\ a=0,\ R_0\in[u,v]\subset(0,40)\).
Bisection gives error bounded by a constant times \(2^{-n}\).
For nonadaptive querying and a uniform prior on \(R_0\), fixed
query radii cut \([u,v]\) into at most \(n+1\) intervals. The
Gaussian mass \(I(R_0)\) has derivative at least \(m_0>0\) on
the interval. On a cell of length \(w\), the best constant
under absolute loss is \(I\) of the midpoint (a median), and
the integrated loss on that cell is at least \(m_0w^2/4\).
Summing and using \(\sum w^2\ge(v-u)^2/(n+1)\) gives Bayes
and hence worst-case randomized risk at least
\(m_0(v-u)/(4(n+1))\). No analogous reconstruction-to-scalar
implication was used in Proposition 4.

**Proposition 5 (an unrestricted-spacing obstruction).** For fixed
\(d\), in \(\mathcal F_d(2,\infty,2d+10,1)\) there are
positive baselined jumps with regular concentric spherical faces
for which the worst-case expected absolute error of *any*
randomized adaptive \(n\)-query algorithm is at least \(c_d/n\).
The same holds for relative error with a changed constant.

**Proof.** Partition \([d,d+1]\) into \(m=2n\) equal intervals.
Use \(f_0=1\), and \(f_j=1+1_{\{|z|^2\text{ in the interior of interval }j\}}\).
The defining functions are \(|z|^2\) minus the two endpoints;
they and their derivatives satisfy the claimed upper bounds
(\(r^2e^{-r}\) and \(2re^{-r}\) are bounded for \(r\ge0\)),
and the boundary gradient is at least \(2\sqrt d\).
Pieces are constant and the jump is one. Since the
\(\chi^2_d\) density has a positive minimum \(v_d\) on the
chosen interval, \(I(f_j)-I(f_0)\ge v_d/m\).
Fix a seed and follow the transcript on \(f_0\). At most \(n\)
open shells are hit. For any other shell the entire transcript
is the same on \(f_j\), by induction on the adaptive steps.
For these \(j\), the sum of the absolute errors on \(f_0\) and
\(f_j\) is at least \(v_d/m\). Average over the shells and seeds:
the sum of the risk on \(f_0\) and the average risk on the shells
is at least \(v_d/(2m)\). The maximum risk is at least
\(v_d/(8n)\). All integrals lie in \([1,2]\).
\(\square\)

## 5. Regularity after conditioning without monotonicity

**Proposition 6 (quadratic envelopes).** Let \(f=g+F\), where \(F\)
is a finite min/max expression in distinct quadratics on
\(\mathbb R^d\). Let \(v\) be a unit direction such that for each
pair \(q=p_i-p_j\):

* if its quadratic part \(A_q\ne0\), then \(v^\top A_qv\ne0\);
* if it is nonconstant affine, then \(b_q^\top v\ne0\).

For \(g\in C^2\) with the stated exponential bounds, the conditional
integrand \(G(x)=\int f(x+tv)\phi(t)\,dt\), \(x\in v^\perp\), is
\(C^{1,1/2}_{\rm loc}\). The local constants depend on the compact
set, the piece coefficients, the growth bounds on \(g\), the
number of pairs, and lower bounds on the nonzero leading
coefficients above. Almost every direction satisfies the
conditions. The exponent \(1/2\) cannot in general be increased.

**Proof.** A nonzero real polynomial has zero set of Lebesgue
measure zero: induct on dimension, expand as a polynomial in
the last variable, and use Fubini apart from the exceptional
simultaneous-zero set of the coefficient polynomials. Apply
this to a Gaussian direction and then normalize. Thus the
exceptional directions for our homogeneous polynomials and
linear forms have sphere measure zero.

At fixed \(x\) all ties involve nonzero univariate polynomials,
so outside a finite set of \(t\) there is a unique active
polynomial (different leaves with the same polynomial may
be identified). On compact \(x\) sets the difference
quotients of the envelope in any coordinate are bounded by
\(C(1+|t|)\), since max and min are 1-Lipschitz in the largest
change of an input. Dominated convergence thus gives the first
derivative as the integral of the first derivative of the active
polynomial. The smooth term has locally Lipschitz gradient by the
same argument with its integrable second derivative bound.

To estimate the difference of the integrated gradients at
\(x,x'\) with \(|x-x'|=\delta\le1\), there is an \(O(\delta)\)
term for changes within a piece. A change of active polynomial
requires a reversal of order of at least two leaves. On
\(t\in[m,m+1]\), this places \(t\) in at least one sublevel
set
\[
 |q(x+t v)|\le C_1\delta(2+|m|).
\]
For a quadratic \(a t^2+b t+c_0\) with \(a\ne0\), the length
of the sublevel set \(|q|\le\eta\) on the *whole line* is at
most \(4\sqrt{\eta/|a|}\). To see this, complete the square;
on each half-line from its vertex the allowed squared radii
lie in an interval of length at most \(2\eta/|a|\), and the
length of its range under square root is at most the square
root of that length. For a linear polynomial with nonzero
slope \(b\) the length is \(2\eta/|b|\). A nonzero constant
cannot reverse order. Multiply these bounds by
\(2C(2+|m|)\sup_{t\in[m,m+1]}\phi(t)\), a bound on the
weighted gradients. The sum over integer \(m\) is bounded
by a constant times \(\sqrt{\delta}\), since the Gaussian
decay absorbs all the powers of \(m\) and \(\delta\le\sqrt\delta\).
The gradient is therefore locally \(1/2\)-Hölder.

For sharpness take \(F(x,t)=(x-t^2)_+\). For \(x>0\)
the conditional derivative is \(2\Phi(\sqrt{x})-1
=2\phi(0)\sqrt{x}+O(x^{3/2})\), and for \(x<0\) it is
zero. The switching surface in the input space is regular.
This proves the claimed limitation. \(\square\)

For fitted quadratic envelopes the conditional integral needs no
oracle calls. Sort all real roots of the piece differences on the
line; choose the active piece by a midpoint test in each open
interval. The moments
\[
 \int_a^b (1,t,t^2)\phi(t)\,dt
 =\big(\Phi(b)-\Phi(a),\ \phi(a)-\phi(b),\
          \Phi(b)-\Phi(a)+a\phi(a)-b\phi(b)\big)
\]
follow from \(\phi'(t)=-t\phi(t)\). The implementation uses this
for the overlap of two positive parts. Proposition 6 does not
give a high-dimensional Sobol' rate through the inverse normal
map. On a bounded outer rectangle it does give the elementary
stratification rate proved below (after a smooth change of
coordinates, if used).

**Proposition 7 (radial removal of a closed jump).** With a
positive-definite ellipsoid and an interior polar origin \(a\),
radial preintegration of
\(\Delta(z)1_{q(z)<0}\), \(\Delta\in C^j\) on a
neighborhood of the ellipsoid, yields a \(C^j\) function of
direction on \(S^{d-1}\). This holds with any Gaussian mean.
It is true even though a line can enter and leave the
ellipsoid. The same statement holds in a known active
subspace, if the orthogonal coordinates have already been
integrated out (for an exponential-affine jump this is exact).

**Proof.** The exit radius \(r(u)\) of Proposition 3 is smooth
on the sphere because its discriminant is bounded below by
\(4\lambda_-\), and its denominator is strictly positive.
For Gaussian density \(\rho\) the directional integral is
\[
 r(u)^d\int_0^1 \Delta(a+s r(u)u)\rho(a+s r(u)u)s^{d-1}\,ds .
\]
On every compact direction chart, derivatives through order
\(j\) under the integral are bounded by a constant times
\(s^{d-1}\). Differentiate by domination. In the cylindrical
case apply the identical proof in the reduced space. For
\(\Delta=L e^{\ell^\top z}\) use independence and the
normal moment-generating function on the orthogonal
complement. \(\square\)

A bounded chart cover is important here. Partition \(S^{k-1}\)
according to the largest absolute coordinate (ties have zero
area) and its sign. On each of the \(2k\) cells the map is
\(u=\pm(e_i+\sum_{j\ne i}x_j e_j)/\sqrt{1+|x|^2}\),
\(x\in[-1,1]^{k-1}\). The metric matrix on this chart is
\((1+|x|^2)^{-1}(I-xx^\top/(1+|x|^2))\).
Its square-root determinant is
\((1+|x|^2)^{-k/2}\) (diagonalize in the radial and transverse
directions of \(x\)). Thus all charts preserve \(C^j\) bounds
up to constants. This avoids the singular chart at the center
of the normalized inverse-normal map.

Here is a self-contained cubature consequence, not used as a
rate claim for the program's scrambled nets. On a fixed
\(p\)-cube, partition into \(m^p\) equal cubes of diameter
\(h=O(m^{-1})\). For a \(C^{1,\beta}\) integrand \(H\),
\(0<\beta\le1\), sample uniformly in each cell and also use
its reflection about the center of the cell. Different
cells are independent. The pair average is unbiased for
the cell mean; its difference from \(H\) at the center is
at most \(L h^{1+\beta}/(1+\beta)\), by integrating the
Hölder gradient along each radius and cancelling the linear
terms. Its variance is at most the square of this bound.
Weighting by the equal cell volumes gives RMSE
\(O(n^{-1/2-(1+\beta)/p})\) with \(n=2m^p\) integrand
evaluations. In particular the exponent on the radial charts
is \(1/2+2/(k-1)\) for \(j\ge2,k\ge2\), and a bounded
domain in Proposition 6 gives \(1/2+3/(2p)\).
These are rates for the *conditioned* integrands, not
black-box query rates without paying for any unknown roots.

## 6. Correcting a learned control

**Proposition 8 (independent residual identity).** Suppose a
training transcript specifies a control \(h\) and its mean \(H\).
For final points of marginal law \(Q\), where
\(w=d\gamma_d/dQ\) exists, the estimator
\[
 H+\frac1n\sum_{i=1}^n w(Z_i)(f(Z_i)-h(Z_i))
 \tag{2}
\]
has conditional mean \(I(f)\). Independence between points is
not needed for this assertion. For IID points its variance is
\(n^{-1}(\mathbb E_Q w^2(f-h)^2-(I(f)-H)^2)\).
If \(Q\ge\alpha\gamma_d\), it is at most
\(\|f-h\|_{L^2(\gamma_d)}^2/(\alpha n)\).
With a mean \(H+\epsilon_H\) in place of \(H\), the conditional
bias is \(\epsilon_H\). These statements follow by changing
measure and, for the variance formula, summing independent
centered variables. For a mixture with prescribed counts
\(n_j\), use \(Q=\sum_j(n_j/n)Q_j\) in the denominator;
the weighted sum of the marginal identities proves unbiasedness
for this deterministic allocation too.

In combination with Proposition 2, suppose the background has
a control of \(L^2\) error \(\zeta\), the finitely many patches
have the stated separation, and the correct expression in the
learned quadratics is selected. After \(O(Kd^2+K\log\log N)\)
probes on informative lines already found, (2) with IID
correction and exact control means has RMSE at most
\(O((\zeta+N^{-\beta})/\sqrt{N-O(Kd^2+K\log\log N)})\).
This uses the displayed quadratic control bound and fixed
coefficient/growth bounds. It is conditional on the patch and
representation assumptions, not a universal cost for discovering
all informative lines or controls. The main benefit of geometry
learning is to put the discontinuity or kink in \(h\), rather
than attempting to fit it with a global polynomial.

**Band version for jumps.** Given a fitted \(q\) and amplitude
\(D(z)=L e^{\ell^\top z}\), define a band
\(\mathcal B=\{|q|<e\}\) and an arbitrary frozen labeling
procedure \(J(z,f(z))\in\{0,1\}\) on this band. Put
\[
 C(z)=1_{q(z)<0},\quad
 D_*(z)=D(z)\big[C(z)+1_{\mathcal B}(J(z,f(z))-C(z))\big].
\]
Use (2) for \(f-D_*\) with an optional smooth control for that
remainder, and add
\[
 L e^{|\ell|^2/2}\Big(
 \Pr_{N(\ell,I)}(q<0)+
 \mathbb E_{N(\ell,I)}[1_{\mathcal B}(J-C)]\Big).
 \tag{3}
\]
This is an exact identity, even when the label rule makes errors,
when \(D\) is only an approximation to the jump, or when the
surface is not an ellipsoid. Crucially the ordinary residual
still contains errors *outside* the band.

Suppose the fitted \(q\) is PSD with \(b\) in its active span,
of dimension \(k\ge1\). Under \(N(\ell,I)\) decompose the
active Gaussian into independent direction \(U\) and
\(\chi_k\) radius \(R\), with independent Gaussian nuisance
coordinates \(W\). The distribution statement follows by the
Gaussian polar change of variables; in the probability coordinate
\(s=F_{\chi_k}(R)\) the conditional density is one on \([0,1]\).
The band is at most two intervals on each fibre, found from
the quadratic inequalities. Denote their union by \(S(U)\)
and its length by \(m(U)\); the integral of \(C\) on \(S\) is
available analytically.

Drawing one \(s\) uniformly on \(S\) and returning
\(m(U)(J-C)\) estimates the band difference conditionally,
with conditional variance at most
\(m(U)^2\Pr(J\ne C\mid U,W,s\in S)\).
If \(S=[a,b]\), a few bisection probes can propose an auxiliary
threshold \(T\) and a narrow working interval \([l,h]\subseteq S\).
Set \(P(s)=1_{s<T}\) and sample with density
\(\alpha/m+(1-\alpha)1_{[l,h]}/(h-l)\) on \(S\).
Return
\[
 \int_S(P-C)\,ds+
 \frac{J(s)-P(s)}
      {\alpha/m+(1-\alpha)1_{[l,h]}(s)/(h-l)} .
 \tag{4}
\]
Given \(U,W\) and the probes, the new uniform variable has
the stated transformed density. Integrating the last term
cancels the auxiliary threshold exactly. This proves
unbiasedness even if the chosen working interval misses
the threshold or the label is nonmonotone. If \(J=P\)
outside \([l,h]\), then the conditional variance is at most
\((h-l)^2/(1-\alpha)\), by integrating the square of
the residual divided by the sampling density. The case
\(m=0\) contributes zero. This proves the band estimator
in the code without a monotonicity requirement.

The reduction in variance can be made quantitative for the
pure ellipsoid case. Suppose inside the active span the shifted
origin is interior to both ellipsoids with uniform margin, their
spectra and \(b\) satisfy common finite positive bounds, and
the shape errors are at most \(\eta\) in operator and
linear norms. Apply Proposition 3 about the shifted origin
to bound \(|r_{\rm true}(U)-r_{\rm fit}(U)|\le C\eta\).
With a band \(e=C'\eta\), \(C'\) sufficiently large,
all disagreeing radii lie in the band: at either boundary
the fitted \(q\) differs from the true one by at most
\(\eta(r_+^2+r_++1)\), and \(q\) stays between the
end values on the interval when \(e\) is sufficiently
small (the radial derivative near its exit is
uniformly positive). The probability length of this band
is at most \(C''e\) since both fitted level radii differ from
the exit by \(O(e)\) and the \(\chi_k\) density is bounded
there. With exact labels the outside remainder has no
surface error; for IID angular samples the total band
RMSE is \(O(\eta/\sqrt{n_{\rm band}})\) by the variance
bounds and the law of total variance. A separate smooth
residual contributes its own \(L^2/\sqrt{n_{\rm smooth}}\)
term. This is a budget allocation rule: enough tomography
to reduce \(\eta\), then probe the small residual rather
than impose a global geometry guarantee. The constants
are not uniform near tangencies, inaccurate labeling,
or a badly conditioned learned span.

**Digital fibres.** For an increasing binary threshold on
a fibre, checking the proposed endpoints and dropping an
invalid end to 0 or 1 gives a bracket in Gaussian
probability coordinates. Each successful bisection
halves its width. The code uses even a potentially
incorrect bracket \([l,h]\) only to propose
\(P(s)=1_{s>(l+h)/2}\). It implements the full-support
version of (4) on \(S=[0,1]\), returning
\(1-(l+h)/2+(Y-P)/q(s)\) with
\(q(s)=\alpha+(1-\alpha)1_{[l,h]}(s)/(h-l)\).
Taking the conditional mean cancels \(P\), so the
identity does not require monotonicity. When the
true threshold is bracketed, the same conditional
variance bound \((h-l)^2/(1-\alpha)\) applies. The
conditional probability of the fitted quadratic
threshold is subtracted at every transverse point
as in (2).

**Cross-fitting.** Conditional on the pilot, let two
independent groups have the correct marginalized
importance law, of fixed sizes \(n_1,n_2\), and fit
controls \(h_1,h_2\) from the two groups. Apply (2)
in group 1 using \(h_2\), and in group 2 using \(h_1\),
then average with weights \(n_i/(n_1+n_2)\).
Each summand satisfies Proposition 8 conditional on
the other group; linearity proves the mean identity
even though the two corrected results are not
independent. Hermite moments vanish by Gaussian
orthogonality. For the tensor control the transformed
transverse coordinate is uniform, so nonconstant
Legendre products are also centered. Thus the only
control coefficients contributing to the tensor mean
are those multiplying a constant transverse feature.

**Known means of quadratics.** In eigen-coordinates,
\(Q=c_0+\sum_i(\lambda_i X_i^2+b_i X_i)\) with
IID normal \(X_i\). Completing the square yields
\[
 M(t)=e^{tc_0}\prod_i(1-2t\lambda_i)^{-1/2}
        \exp\!\left(\frac{t^2 b_i^2}{2(1-2t\lambda_i)}\right).
\]
The formula first holds for real \(t\) with positive
denominators, then for complex \(t\) in the same vertical
strip by dominated analytic continuation. The
continuous branch at \(t=0\) is used. An expression for
\(\mathbb E Q_+\) is
\[
 \frac{\mathbb E Q}{2}
    +\frac1\pi\int_0^\infty\frac{1-\Re M(it)}{t^2}\,dt .
\]
Indeed the inner integral for a fixed \(Q\) is
\(\pi|Q|/2\), by scaling and
\(1/t^2=\int_0^\infty v e^{-vt}\,dv\): integrating
\((1-\cos t)e^{-vt}\) first gives
\(\int_0^\infty (1+v^2)^{-1}dv=\pi/2\). Tonelli applies.
For the left probability choose \(s<0\) inside the strip,
and tilt the law by \(e^{sQ}/M(s)\).
The Fourier transform (convention \(e^{-itx}\)) of
\(e^{-sx}1_{x<0}\) is \((-s-it)^{-1}\).
Using its Gaussian-smoothed Fourier inverse and then
letting the smoothing tend to zero gives
\[
 \Pr(Q<0)=\frac1\pi\int_0^\infty
               \Re\frac{M(s+it)}{-s-it}\,dt .
\]
The jump of the test function at zero is harmless because
a nonconstant quadratic of a Gaussian has no atom there
(by the polynomial zero-set argument). Dominated
convergence on the Fourier side follows because the
tilted law is another Gaussian after completing the
square. Its characteristic function decays at least
as \(O(t^{-k/2})\) for quadratic rank \(k>0\), by the
product formula, or Gaussian-fast for a nonconstant affine
form. The solver numerically integrates these formulas;
it does not equate a requested quadrature tolerance with
a certified error.

**Randomized net and coverage.** The code randomizes Sobol'
by a linear matrix scramble and digital shift, not a claim
of Owen's nested scramble. In the finite \(B\)-bit
model, conditional on the linear matrices, let \(P\)
be the unshifted set of digital vectors in
\(\mathbb F_2^{Bd}\), and \(S\) the uniform shift.
For any transformed integrand \(V\) on this grid,
\[
 {\rm Var}_S(\overline V)=
 \sum_{k\ne0}|\widehat V(k)|^2
 \left|\frac1{|P|}\sum_{p\in P}(-1)^{k\cdot p}\right|^2 .
\]
To prove this, expand \(V\) in the orthonormal characters
\((-1)^{k\cdot u}\), average \(u=p+S\), and
use orthogonality over \(S\). For a subspace \(P\)
the inner factor is one on its annihilator and zero
elsewhere (translate the sum by a vector where the
character equals \(-1\)). This identity applies
conditional on training and on any fixed deterministic
transport, including weighted residuals. It does not
imply a uniform asymptotic rate without spectral decay.
A digital shift gives uniform marginals on the grid,
not on the continuous cube. Finite bits, clipped extreme
coordinates, and numerical control means are therefore
distinct discretization errors; no exact continuous-law
unbiasedness assertion is made about floating-point runs.
For the ideal estimator with an *a priori* valid variance
bound \(V\) and absolute bias bound \(b\), Chebyshev
gives coverage at least \(1-\alpha\) for
\(b+\sqrt{V/\alpha}\), also conditionally on training.
In a general black-box run such bounds are not known.
The solver intentionally does not claim a confidence
interval.

There is also a limitation to an honest *finite* confidence
bound without any supplied bound on amplitude. Consider the
union of the classes allowing arbitrarily large \(M\), use
the shells of Proposition 5 with \(m\) arbitrarily large
relative to \(n\), and replace their height by \(H\).
For a fixed seed, at most \(n\) shells are hit on the
constant transcript. For every missed shell the output
interval is exactly the one on the constant transcript.
Its estimate and radius are finite almost surely, so the
probability that it reaches \(1+H v_d/m\) tends to zero as
\(H\to\infty\). A uniform prior over shells therefore
gives average coverage at most \(n/m+o(1)\); some shell
has no larger coverage. This is not an impossibility
result with known fixed \(M,c\), but explains why a few
apparently stable replicates are insufficient for a
distribution-free guarantee when no such bounds are given.

## 7. A second-order radial correction for a dense ellipsoid

**Proposition 9 (match the first variation with \(d\) rays).**
In coordinates about a known interior point let the fitted
quadratic be \(Q(x)=x^\top A x+b^\top x+c_0\), \(c_0<0\).
The true quadratic has matrix \(A+E\) and the same \(b,c_0\).
Assume both have spectra in the fixed interval
\([\lambda_-,\lambda_+]\subset(0,\infty)\), and its membership
can be queried. Take \(X\sim N(m,I)\) and assume
\(\|E\|\le\eta\) for a known bound \(\eta\).
There is a correction of the fitted probability with
error, in root mean square for IID angles,
\[
 C\,\frac{\eta^2}{\sqrt{n_\theta}}+
 \epsilon_{\rm ray}\sum_{i=1}^d |g_i|
 \tag{5}
\]
in exact arithmetic and with exact model integrals. Here
\(g_i\) are the eigenvalues of the matrix first variation
at the fit, and \(\epsilon_{\rm ray}\) is the accuracy
of each additional directional coefficient measurement.
The coefficient measurements need
\(O(d\log(C/\epsilon_{\rm ray}))\) value queries.
With this error bound for initial bracketing, a bounded
number of radial probes per angle suffices for the
\(\eta^2\) scale in (5). Constants depend on \(d\),
the spectral, linear, and interior-margin bounds,
and \(|m|\).

**Proof.** Let \(u\) be uniform on the sphere, and let
\(\rho,\tau\) be the fitted and true positive radii
in direction \(u\). Write
\[
 w_u(r)= \frac{2^{1-d/2}}{\Gamma(d/2)}
 r^{d-1}\exp(-r^2/2+r u^\top m-|m|^2/2),\qquad
 \kappa_u=\frac{w_u(\rho)}{\partial_r Q(\rho u)} .
\]
This is the radial density relative to uniform direction,
including the change of Gaussian mean. Implicit
differentiation of the exit radius along a change in
\(A\) gives the matrix derivative of the probability,
\(G=-\mathbb E_u[\kappa_u\rho^2 uu^\top]\), in the
sense that its first variation against \(E\) is
\(\langle G,E\rangle=\operatorname{tr}(GE)\).
The interchange with angular integration is valid by
the uniform radius bounds and the positive lower bound
on the radial derivative, as in Proposition 3.

Measure \(u_i^\top(A+E)u_i\) on the \(d\) eigenvectors
of \(G\) using the one-sided ray equation with known
\(b,c_0\). Recovering the scalar from a bisection radius
is Lipschitz on the uniform radius interval. The resulting
weighted sum differs from \(\langle G,E\rangle\) by at
most the second term of (5).

For the remaining term define
\[
 B_u(r)=w_u(r)-\kappa_u
             \frac{d}{dr}\left(\frac{\rho^2 Q(ru)}{r^2}\right).
\]
Since \(Q(\tau u)=-\tau^2 u^\top E u\) and
\(Q(\rho u)=0\),
\[
 \Pr(Q(X)+X^\top E X<0)-\Pr(Q(X)<0)
 = \langle G,E\rangle + \mathbb E_u\int_\rho^\tau B_u(r)\,dr.
\]
At \(r=\rho\) the integrand \(B_u\) vanishes by the
definition of \(\kappa_u\). Its derivative is bounded
uniformly on the compact radius interval, enlarged
slightly if necessary. By Proposition 3,
\(|\tau-\rho|\le C\eta\). Therefore the displayed
radial remainder is bounded in absolute value by
\(C'\eta^2\) for every direction.

Now bracket \(\tau\) by \([l,h]\) with deviations from
\(\rho\) at most \(C\eta\), using the radius bound and,
optionally, midpoint membership tests. For a fresh
uniform \(R\) on \([l,h]\), return
\(\int_\rho^l B_u(r)\,dr+(h-l)B_u(R)1_{R<\tau}\).
Its conditional expectation is the desired remainder.
Its conditional variance is at most
\((h-l)^2\sup_{[l,h]}|B_u|^2=O(\eta^4)\).
Adding the between-direction variance and averaging IID
directions proves (5). If the fitted \(b,c_0\) are only
approximate, the same calculation has an additional
absolute perturbation term of order
\(|\Delta b|+|\Delta c_0|\): the identity
for \(Q(\tau u)/\tau^2\) now has extra terms
\(-u^\top\Delta b/\tau-\Delta c_0/\tau^2\).
The directional measurements have analogous errors
bounded with the same uniform radius bounds. \(\square\)

The code computes the gradient using the one-dimensional
left-probability transform of Section 6. To see the formula,
let \(M(w)=\mathbb E e^{wQ(X)}\), \(C_w=(I-2wA)^{-1}\), and
\(\mu_w=m+wC_w(b+2Am)\). Differentiating under the expectation
and completing the square gives
\[
 \partial_A M(w)[D]
   = w M(w)\operatorname{tr}\!\big((C_w+\mu_w\mu_w^\top)D\big).
\]
This identity first holds for real tilts where the Gaussian
completion is valid and extends to the allowed vertical strip
by analyticity. In the positive definite case the characteristic
factor \(M(s+it)\) is uniformly \(O(t^{-d/2})\) on a small
parameter neighborhood, and \(C_w,\mu_w\) are bounded there
on the contour. For \(d>2\) this permits differentiation of
the CDF integral by domination. The matrix integrand is
\(-\Re[M(w)(C_w+\mu_w\mu_w^\top)]/\pi\), integrated in \(t\).
In the solver the measured linear terms, the tilt, the contour
integral and the radius bracket are all approximate. This
branch is reserved for dense large jumps with strong label
contrast and successful value brackets; there is no claim
of guaranteed brackets on an arbitrary test.

The other special large-jump route uses an angular central
Gaussian proposal. If \(V\sim N(0,C)\) for positive definite
\(C\), then the density of \(V/|V|\) relative to uniform
direction is
\(\det(C)^{-1/2}(u^\top C^{-1}u)^{-d/2}\).
Indeed integrate its Gaussian polar density over the radius,
rescaling radius by \(\sqrt{u^\top C^{-1}u}\), and divide
by the same integral at \(C=I\). With exact threshold
membership and an interior origin, its inverse supplies
the weight for the residual of conditional \(\chi_d\)
probabilities. This route uses the simple bracket estimator
\(l+(h-l)1_{s<\tau}\) in probability coordinates (where
the bracket is known or falls back to the entire relevant
tail). Its exactness relies on the single exit hypothesis,
unlike the defensive band rule.

## 8. Comparison with the appendix results

The comparisons below are to the statements in Section 5
of `THEORY_APPENDIX.md`. They do not assert a uniform
improvement on the whole \(\mathcal F_d\), or priority over
the wider literature.

| Known result or subject | What is reproduced or extended here |
| --- | --- |
| Bakhvalov; Hinrichs et al., smooth-class rates and curse of dimension | We do not alter the smooth-class rates. The shape results use finite-dimensional parameters that can be interrogated directly. Proposition 5 gives a separate analytic piecewise lower bound, even with a smooth constant background. |
| Kuo, Sloan, Woźniakowski, Gaussian RKHS integration | The Fourier means concern learned quadratic controls, not the Gaussian-kernel RKHS. Reducing their means to one dimension does not remove the RKHS curse. |
| Owen, scrambled-net smooth variance | We use scrambled nets, but prove only the finite digital-shift identity for the implemented scramble. The chart and antithetic results are explicit alternative cubatures for conditioned functions, not a new Owen rate. |
| Plaskota and Wasilkowski, unknown one-dimensional singularities | Proposition 1 concerns *locating* an internal kink with a nonzero trace, not a general integration rate. It uses a jump lower bound and two smooth extrapolants for a quadratically shrinking bracket. This addresses part of open problem 3. |
| Alexander; Matoušek, halfspace discrepancy | Those bounds concern equal-weight rules. Our adaptive directional measurements and unequal-weight residuals are not subject to that premise. The nonadaptive lower bound in Proposition 4 uses affine sign cells in parameter space, not halfspace discrepancy in input space. |
| He and Wang, general discontinuous RQMC | The band bound is a conditional \(L^2\)/IID statement for a *learned* jump residual. It reduces the scale of the residual, rather than claiming a new worst-case RQMC exponent for general jumps. |
| Griebel, Kuo, Sloan; Griewank et al., monotone preintegration | Proposition 7 extends smoothing to closed ellipsoids by nonlinear polar coordinates. Proposition 6 obtains \(C^{1,1/2}\) without monotonicity for quadratic envelopes, including intersecting switches. These are scoped answers to open problem 1. |
| He, conditional option RQMC | Digital conditioning reproduces the reduction to a threshold probability on suitable fibres. The defensive last draw extends the value-only estimator to nonmonotone or inaccurate fibre proposals; we do not claim the known option rates for unknown rotated discretizations. |
| Gilbert, Kuo, Sloan, failure of monotone smoothing | Our sharp example exhibits the fractional singularity, with a \(t_+^{3/2}\) leading term for a *continuous* hinge. The interior-origin radial chart removes the degeneracy only on the specified ellipsoid class. |
| Liu and Owen, direction selection by active subspace | The generic fallback uses gradient information. Proposition 2 instead recovers the range of a local Hessian *difference*, with cancellation of the common background and an explicit subspace bound from point queries. This addresses part of open problem 2 under an isolated-patch and gap condition. |
| Bayer, Ben Hammouda, Tempone, numerical smoothing | Root localization plus conditioning is the same broad strategy. The additional value-only kink refinement, band defense, and first-variation radial residual treat particular inaccessible or nonmonotone surfaces. No claim is made for multilevel complexity in a diffusion model. |
| Hickernell and Jiménez Rugama; Clancy et al., reliable cones | No cone assumption or certified stopping rule is used. Section 6 gives a conditional coverage statement with actual variance and bias bounds as assumptions, and an unknown-envelope obstruction. This is a limited answer to open problem 5. |
| Heinrich, randomized adaptation | Proposition 4 supplies a parameter-space separation for membership tomography and a scalar integration gap on the simpler unknown-radius subfamily. Proposition 5 shows why this cannot give an exponential rate on the unrestricted-spacing class. This addresses part of open problem 4. |
| He, axis-friendly discontinuity surfaces | Fitted bands are made cheap by radial geometry, not assumed parallel to Cartesian axes. Proposition 8 localizes the correction to those bands unless the geometry misses the jump; it is not an extension of the axis-aligned RQMC exponent. |
| Hoyt and Owen, mean dimension of ridges | The active rank here is a matrix rank for a quadratic, not an ANOVA mean dimension. The computation benefits from true or learned null directions, without a bound on ANOVA order. |
| Chen, Xu, Du, Wang, preintegration and importance sampling | Completing the square for an exponential-affine jump and using defensive Gaussian proposals are related tools. Our analysis gives an exact change of measure for the fitted controls and second-order radial correction; it does not assume their critical-growth regime or prove their near-\(n^{-1}\) rate. |

## 9. Connection to the experiment

The equations behind the synthetic pilots can be checked
without knowing their parameters. On the two H1 samples
the accepted single difference removed the sharp fourth
differences. On H2 two differences were needed; their
remaining atlas fourth-difference energy was below
\(4\cdot10^{-5}\) of the original energy. The kink values
were nonzero. This supports the local quadratic-difference
assumption, not an assertion that the method always
finds all surfaces in a hidden H1 or H2.

The H3 samples showed closed jumps with approximately
exponential-affine gap, and H3t had a much larger,
nearly constant gap. On H3-d32 a suspect axial radius
was repaired by searching for the actual value gap, and
the rank-16 completed shape then agreed with the Gaussian
pilot labels. On H3t-d8 the fitted active rank was two,
so the band calculation was essentially two-dimensional
with a Gaussian nuisance space. These are the cases where
Propositions 3, 7 and 8 are most relevant. High-rank
ellipsoids can leave only a few bits per cross coefficient;
the program has no uniform conditioning or label-margin
guarantee. The stricter hypotheses of Proposition 9 are
a model for some dense large jumps, not a deduction from
the problem's analytic-growth assumptions alone.

The digital sample had a one-direction threshold close
enough to use short brackets (model-training root RMS
about \(0.035\) in normal coordinates). The conditional
contrast rule applies more generally by its defense,
but on heavily curved barrier payoffs the regressions
and importance proposals are variance heuristics; none
of the pure ellipsoid or patch rate results asserts a
rate for those hidden paths. With an exact fitted model
and a smooth residual of scale \(\zeta\), the IID benchmark
of Section 6 predicts absolute error of order
\(\zeta/\sqrt{n}\); geometry error reduces this scale
from the original jump or hinge to the band or patch
error. The observed net errors can be smaller or larger
than this benchmark depending on the residual's digital
frequencies. In particular a favorable reference
scramble can beat a smoother fit on a single case.

Measured with the supplied sample evaluator
(\(\log_{10}\) of relative error; gain clipped as in
the problem). The table is deliberately per sample,
not a claim about hidden-held-out accuracy:

| Sample | Program log error | Gain over reference |
| --- | ---: | ---: |
| H1 d8 | −5.11 | +1.84 |
| H1 d32 | −5.67 | +2.00 |
| H2 d8 | −4.71 | +1.17 |
| H2 d32 | −4.33 | +0.62 |
| H3 d8 | −4.67 | +1.26 |
| H3 d32 | −4.71 | +1.17 |
| H3t d8 | −4.32 | +2.00 |
| H3t d32 | −4.13 | +2.00 |
| European call m64 | −2.81 | −0.05 |
| Digital m32 | −3.37 | +0.87 |
| Down-and-out m64 | −2.82 | −0.10 |
| Double knock-out m16 | −1.81 | +0.71 |

All twelve used at most 16,384 queries. The mean log gain
was about \(1.51\) on these hard samples and \(0.36\)
on the option samples. The improvement is accordingly
more consistent for the recovered quadratic geometry
than for paths. The sample score is 73.3; the samples
are few, and neither this score nor any public score is
a confidence statement or a bound on final performance.
