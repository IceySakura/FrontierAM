# Gaussian cubature from local changes of analytic pieces

## Scope and notation

Let \(\gamma_d\) be standard Gaussian measure, \(I(f)=\int f\,d\gamma_d\),
and \(\phi,\Phi\) the standard normal density and CDF. A query means one
function value, including each point in a simultaneous batch. Write
\(a_+=\max(a,0)\). Numerical identification of a model is not a proof that
an arbitrary oracle belongs to the model. The positive results below
specify the model, what is known before the queries in the theorem, and
whether they are exact-arithmetic or error statements.

The submodels can be intersected with \(\mathcal F_d(K,r,M,c)\) of the
appendix. In results using arbitrary exponential loadings, one may take
finite \(r\) and sufficiently large \(M,c\); real analyticity alone does not
imply an \(r=\infty\) bound with the *same* \(M,c\) for all derivative
orders. Bounds on a query region below are in addition to any global
growth bound. Where the test model uses a sign at an exact surface point,
take its specified one-sided convention. Otherwise probing and bracketing
can avoid the surface or count equality as a valid bracket endpoint. The
results concern absolute error unless otherwise noted. With \(|I(f)|\)
bounded below they immediately imply relative error bounds.

## 1. Implemented algorithm

All information about an instance comes from its protocol and its function
values. The launch script uses one numerical thread. The initial design
queries 512 Gaussian points from a scrambled Sobol net and the origin.
The initial estimate is the mean of the 512 points. The numerical
discussion here refers to `solution.py`, `hard*.py`, `jump.py`, `smooth.py`,
`sparse_gauss.py`, `options.py`, `perpetuity.py`, `barrier.py`, and
`digital.py`.

**Synthetic models.** The program scans straight lines through the origin
and selected pilot sites. Large adjacent differences compared to their
neighbors suggest jumps; local maxima of the absolute fourth difference
suggest continuous switches. It refines a jump bracket and interpolates
both sides; a continuous switch is localized by finding the intersection
of the two one-sided quintic interpolants, with successively shorter
stencils. It checks that the derivative difference is stable. This
procedure does not look for zeros of the payoff.

* At a jump, the program tests a constant or log-linear difference
  between the two branches. It chooses a likely interior pivot, locates
  exits in opposite directions, and computes the normalized normal and
  trace of the quadratic from the paired roots. It uses two extra chord
  directions for the repeated nonzero eigenvalue and the centered or
  tilted Gaussian projection. The validity checks include integer rank,
  additional chord slopes, and raw value jumps on the two sides of the
  proposed exit. The modeled jump integral is a noncentral chi-square CDF
  times the exponential-tilt factor. In intermediate ranks it usually
  removes the jump using its fitted height and a smooth baseline level;
  additional probes at a geometrically equivalent site can help in
  ambiguous cases. It does not assume it knows the full projector.
* At a continuous switch it estimates a jump normal and forms finite
  second differences on both sides, transporting the linear differences
  to the crossing. The difference of these local second-order models
  proposes a quadratic. For larger dimensions it grows a column sketch
  and accepts a low-rank completion only when it predicts the diagonals
  and extra directional curvatures. Rayleigh diagonals from paired
  extrapolants refine the spectrum. The two signs of an internal-switch
  matrix can propose two cap subspaces; an outer kink can set the missing
  common constant. After one outer cap is known, a cheap directional
  test can also determine a second cap in its entire orthogonal
  complement. The code tests a positive part of one quadratic, a sum
  of two positive parts, or a maximum of two concave caps and zero, using
  high differences of the residual on the exploratory lines. For the
  latter model it checks near orthogonality and orthogonalizes the second
  projector before integrating the model. A hinge mean is calculated by
  one-dimensional characteristic-function quadrature; the maximum of
  independent caps is integrated by their one-dimensional CDFs.
* A successful synthetic model supplies a number \(J\) and a pointwise
  residual \(r(z)\). The estimate is \(J+\widehat I(r)\). In particular,
  the quadratic does not have to equal the oracle's nonsmooth component
  exactly. The residual estimate uses gradient sketches to order an
  orthogonal frame, then Gaussian Sobol nets with antithetic pairs.
  Polynomial controls use centered Gaussian Hermite products. Their
  coefficients for a given net are fit on the other nets; the correction
  subtracts the sample mean of the fitted control and adds its Gaussian
  mean. A fitted quadratic can supply an extra centered feature.
  In dimension eight, if the geometry checks support it and enough
  queries remain, a level-four nested Gaussian sparse rule is tried on
  the residual, with the level-three estimate as a check. A spare
  budget permits terms with five or six level-one factors. If there is
  little budget left, the simpler uncorrected residual mean is used.

**Smooth positive option pieces.** Zero values trigger a separate group
of tests. On positive regions the fitter first tests a single exponential
minus a strike, then uses two finite-step multivariate stencils to
identify an exponential sum. On a min payoff it looks for a second
positive branch where the first extrapolated piece is larger than the
oracle, including probes that hold the first piece fixed along its null
space. Stencils and their connecting segments are screened for positivity
and convexity; when the first stencil detects lower rank, the shifted
stencil can be restricted to fewer coordinates.
The fitted terms are ordered using the Gram matrix of their
Gaussian loadings and checked against equal-date Brownian covariance and
geometric coefficients. The recovered option is also evaluated against
previous oracle values at multiple scales. For an accepted Asian model,
the program uses one- or two-dimensional CDF recursion, with Gaussian
convolution by FFT on padded log-sum grids, local degree-five
interpolation, and final Gaussian quadrature. For a terminal barrier
model it searches for flat faces by membership, reads each face's date
from its angle to the terminal direction, and fits the common log
barriers and total drift. A first exposed face can steer the search
toward the opposite barrier in the terminal-orthogonal space.
A one-dimensional killed transition kernel
then gives the price. Neither calculation queries the oracle at its
internal grid points.

**Digital options.** A two-level payoff uses logistic separation to
propose a direction. Bracketed slice roots give a finite-difference
normal and a few slope secants in the transverse plane. With the resulting
coordinates, a pilot of 192 or 256 root slices fits two polynomials: one
predicts the root, the other predicts its Gaussian conditional
probability. On fresh Sobol slices the algorithm tests and expands the
predicted brackets, bisects, and uses one random threshold in Gaussian
probability coordinates in the remaining bracket. The polynomial with
known mean acts as a control. Unresolved slices or early budget stops
make this an approximate estimator rather than a coverage certificate.

**Budget and guards.** Discovery, finite differences, validation and
bisection are all charged by the same client. The sizes below explain
why the methods are usable at \(N=16384\); the branch guards and clipping of
the last batches determine the exact count on an instance.

| Operation | Main query cost |
|---|---|
| Discovery | 513 initially; typically 830 or 1105 more for line scans on the synthetic track; extra probes if needed |
| One quadratic jet pair | \((d+1)(d+2)\) for dense matrices, potentially fewer with the column sketch, plus \(O(d)\) extrapolation probes |
| One exponential pencil | At most \((d+1)(d+2)\) for a single accepted shift, less for a restricted shift; patch screens and unsuccessful shifts also cost queries |
| One spherical chord direction | Two scalar searches, up to 29 bisections each after finding brackets |
| One exposed flat face | \(d+O(1)\) nearby searches, up to 15 bisections each, plus an axial refinement |
| Residual | The remaining power-of-two designs, charging two queries per antithetic site |

Failed recognition can fall back to a fresh rotated scrambled net; an
unresolved barrier can also use a net restricted to its inferred terminal
interval. The fallback orders dimensions by a finite-difference gradient
sketch when the budget permits. A result contains only an estimate; no
formal error or confidence field is emitted.

An unresolved min payoff with one identified piece can use that
piece as a control. For such a payoff \(0\le f\le A_+\), with
\(A_+\) the recovered individual call. The residual \(f-A_+\)
vanishes when that call does not exercise. On a line where the
recovered exponential sum is increasing, the program computes its
root using the model only, draws from the conditional Gaussian
tail, and weights the oracle residual by the tail probability.
The identity follows by conditioning on the orthogonal Gaussian
coordinates and restricting the one-dimensional integral to where
\(A_+>0\). It is exact for the correctly recovered piece and exact
line integration, before discretization and sampling error.

The sparse option uses nested one-dimensional Gaussian rules
\(U_0,U_1,U_2,U_3\) of sizes \(1,3,9,19\).
The weights match the Gaussian Hermite moments through degree one less
than the number of abscissae. For \(\Delta_j=U_j-U_{j-1}\),
\(U_{-1}=0\), the level-\(\ell\) rule is
\(\sum_{\sum_i j_i\le\ell,\;0\le j_i\le3}
\Delta_{j_1}\otimes\cdots\otimes\Delta_{j_8}\) in the fitted
orthogonal frame. Shared nodes are queried only once. Signed weights
and a small observed level difference are not an error certificate.

## 2. Recovering a projected closed surface from chords

### Theorem 1 (logarithmic-precision identification and tilted integral)

Suppose
\[
 f(z)=h(z)+J(z)\mathbf 1_{\|Pz-c_0\|^2<R^2},\qquad
 J(z)=A e^{b^Tz},\quad A>0 ,
\]
where \(P\) is an unknown orthogonal projector of unknown rank
\(2\le k\le d\), \(c_0\) is in its range and \(R>0\). This model has one
regular quadratic surface (\(K=1\)) for \(R>0\); \(h\) need not depend
only on the range of \(P\). Assume for this theorem:

1. A pivot \(p\) is known to be strictly inside, with \(Pp\ne c_0\).
2. The two sides are labelable from each query on the search region.
   For example, given \(J\) and \(|h(z)-H(z)|\le B(z)<J(z)/2\),
   the test \(f(z)>H(z)+J(z)/2\) suffices.
3. The rays used below have known exit brackets with upper bound \(L\)
   and the true positive and negative exit distances are at least
   \(l>0\). All their queries lie in the coordinate box. In particular,
   no search direction required by the theorem lies in the null space.
   If \(p=b\), the search direction \(p-b\) is omitted.

In exact arithmetic \(d+2\) paired exit distances suffice to determine
the integral of the jump, without fully determining \(P\). Bisection with
absolute root error at most \(\delta\) uses at most
\[
   2(d+2)\big(2+\lceil\log_2(L/\delta)\rceil\big)
\]
queries after fixing the directions and brackets, including verification
of the two endpoints. For fixed \(l,L,d\), a positive lower bound on
\(\|Pp-c_0\|/(R^2-\|Pp-c_0\|^2)\), and bounded \(p,b,A\), the inferred rank
is eventually exact and the error of the inferred jump integral is
\(O(\delta)\). The constants and a sufficient rank condition are given
in the proof.

**Proof.** Divide the surface equation about \(p\) by its interior
slack \(\eta=R^2-\|Pp-c_0\|^2>0\). The inside condition for \(z=p+y\) is
\[
 a\, y^TPy+n^Ty-1<0,\qquad
 a=1/\eta,\quad n=2a(Pp-c_0).
\]
On the line \(p+tu\), \(\|u\|=1\), the roots are \(-r_-,r_+\).
Their product and sum give
\[
 \alpha(u)=a u^TPu=\frac1{r_+r_-},\qquad
 u^Tn=\frac1{r_+}-\frac1{r_-}.
\]
The \(d\) orthogonal directions give \(n\) and
\(S=\sum_i\alpha(e_i)=ka\). Since \(n\) is in the range of \(P\),
the chord along \(n/\|n\|\) gives \(a\). Let \(s=p-b\).
Use the last chord to get \(s^TaPs\), multiplying its curvature by
\(\|s\|^2\); set this value to zero if \(s=0\).

The identity \(e^{b^Tz}\,d\gamma_d(z)=e^{\|b\|^2/2}\,dN(b,I_d)(z)\)
follows by completing the square in the density. On this shifted
Gaussian the event in the theorem has probability \(F_{k,\lambda}(R^2)\),
where \(F_{k,\lambda}\) is the noncentral chi-square CDF and
\[
 R^2=\frac{1}{a}+\frac{\|n\|^2}{4a^2},\qquad
 \lambda=\frac{s^TaPs-s^Tn}{a}+\frac{\|n\|^2}{4a^2}.
\]
Indeed the mean displacement from \(c_0=Pp-n/(2a)\) is the projection of
\(-s\) plus \(n/(2a)\). Thus the desired number is
\(A e^{\|b\|^2/2} F_{k,\lambda}(R^2)\).

For stability take \(\delta\le l/2\).
Each reciprocal has error at most \(2\delta/l^2\) and is at most \(2/l\)
in magnitude, so the errors in \(\alpha(u)\) and \(u^Tn\) are at most
\(6\delta/l^3\) and \(4\delta/l^2\), respectively. Thus
\(\|\widehat n-n\|\le E_n=4\sqrt d\,\delta/l^2\) and
\(|\widehat S-ka|\le E_S=6d\,\delta/l^3\). If \(E_n<\|n\|/2\),
normalizing changes the unit vector by at most \(2E_n/\|n\|\). The
Rayleigh quotient at this perturbed vector differs from \(a\) by at most
\(4a E_n^2/\|n\|^2\), because the difference is \(a\) times the squared
projection onto the null space. Therefore
\[
 E_a=6\delta/l^3+4a E_n^2/\|n\|^2
\]
bounds \(|\widehat a-a|\). If \(E_a<a/2\) and
\((E_S+kE_a)/(a-E_a)<1/2\), rounding \(\widehat S/\widehat a\) gives
\(k\). The rational expressions above for \(R^2,\lambda\) are
Lipschitz on the corresponding compact parameter sets with
\(\widehat a\ge a/2\); \(a\ge 1/L^2\) from its chord. The error in
\(s^TaPs\) is at most \(6\|s\|^2\delta/l^3\).
Clamp a negative numerical \(\widehat\lambda\) to zero, which does not
increase its error.

For \(k\ge2\) the chi-square density at any nonnegative noncentrality
is bounded by \(1/2\). To see this, the noncentral law is the mixture of
central laws of degrees \(k+2j\) with Poisson weights
\(e^{-\lambda/2}(\lambda/2)^j/j!\). Completing a Gaussian square gives
its Laplace transform; expanding
\(\exp[\lambda((1+2t)^{-1}-1)/2]\) gives exactly the mixture transform.
Each component is the sum of an independent \(\chi^2_2\) and a
nonnegative variable, and convolution preserves the density bound.
Differentiating the Poisson weights (one-sided at zero) gives
\(\partial_\lambda F_{k,\lambda}
=(F_{k+2,\lambda}-F_{k,\lambda})/2\), of magnitude at most \(1/2\).
Thus the integral error is at most
\(A e^{\|b\|^2/2}(|\widehat R^2-R^2|+
|\widehat\lambda-\lambda|)/2=O(\delta)\). The query bound follows by
halving each scalar bracket; the assumption includes the adaptively
chosen direction \(\widehat n/\|\widehat n\|\). \(\square\)

**Equivalent membership without the rotation.** Put
\(D=1-n^Ty\). If \(D>0\), define \(y'=-y/D\). Direct expansion gives
\[
  a{y'}^TPy'+n^Ty'-1=
  \big(a y^TPy+n^Ty-1\big)/D^2.
\]
Thus querying \(p+y'\) can test the same side as \(p+y\), with no
knowledge of \(P\), when \(p+y'\) is admissible and its jump is easier to
distinguish. For an approximate \(n\), the identity is not exact; the
program also checks distance and amplitude before trying a related
correction. In particular decompose \(y=\alpha u+w\), with
\(u=n/\|n\|\) and \(w\perp u\), and set
\(B(t)=1-a t^2-\|n\| t\).
The normalized quadratic is \(a w^TPw-B(\alpha)\) because \(Pu=u\).
For \(0<s\le1\) and \(B(\alpha)>0\), solve
\(B(\alpha')=s^2 B(\alpha)\) and put \(y'=\alpha' u+s w\).
The discriminant is nonnegative because the level
\(s^2B(\alpha)\) is at most the maximum of the concave quadratic.
The resulting surface value is \(s^2\) times the original value.
As \(s\) decreases, either root \(\alpha'\) approaches an axial
endpoint. The implementation checks ambiguous sites in the vicinity
of the endpoint with greater modeled height, comparing to a
baseline level measured just outside it. That comparison is still
heuristic without a separation bound on this neighborhood.

**Budget consequence.** On the separated subclass the residual can be
evaluated exactly as \(h\) from a payoff query (given \(J\)). For fixed
parameters of Theorem 1, setting \(\delta=N^{-p_0}\) costs
\(O(p_0d\log N)\) queries. For example, with \(p_0>1/2\), the structural
bias is \(o(N^{-1/2})\) and independent Gaussian residual sampling has
RMSE at most
\[
 C N^{-p_0}+\frac{\sqrt{\operatorname{Var}(h(Z))}}
 {\sqrt{N-O(d\log N)}} .
\]
The variance is finite under the stated exponential growth condition.
The displayed expression is for ideal unrestricted residual sampling.
With the coordinate box restriction, discard residual draws outside
the box, using zero without querying. Replace the variance by that of
the truncated residual and add a tail bias bound
\[
 M\big(\mathbb E e^{2c\|Z\|}\big)^{1/2}
       \sqrt{2d\,\bar\Phi(40)} .
\]
Cauchy--Schwarz and the union bound prove this inequality.
For instance \(\|Z\|\le\sum_i |Z_i|\) bounds the exponential
moment by \((2e^{2c^2})^d\).
This does not bound the search for the *first* interior point. In
particular it is not an \(O(d\log N)\) theorem for arbitrary rare events.

### Proposition 1a (a radial alternative for a nonmonotone surface)

This is an auxiliary result about the function class, not a second
pricing branch in the implementation. Let \(2\le k\le d\), and
suppose the active \(k\)-space and a pivot \(p\) in it are given.
In that space let the event be
\(\{p+t\omega:0\le t<R(\omega),\ \omega\in S^{k-1}\}\), with
\(0<a_0<R(\omega)<a_1<\infty\).
Assume the event is defined by a \(C^r\) regular function
(\(r\ge1\)) with nonzero radial derivative at every root, a single
radial crossing, and queryable membership. Assume all radial
probes below are admissible. For a known jump height
\(J(z)=A e^{b^Tz}\) on the active space, radial preintegration yields
a \(C^r\) function on the sphere. Stratified sampling of this
radial integral with approximate radii gives, for fixed parameters,
RMSE \(O((N/\log N)^{-1/2-1/(k-1)})\) for the **jump contribution**
with \(N\) membership queries. No single Cartesian direction is
required to have strictly monotone defining function.

**Proof.** The implicit function theorem in each local angle chart
gives \(C^r\) radii because the radial derivative does not vanish.
The radius functions agree on overlaps by uniqueness. By
compactness their derivatives up to order \(r\) are bounded on
closed subcharts. In polar coordinates about \(p\) the angle
integrand is
\[
 G(\omega)=\int_0^{R(\omega)}
  J(p+t\omega)(2\pi)^{-k/2} e^{-\|p+t\omega\|^2/2}
    t^{k-1}\,dt .
\]
The change of variables \(t=sR(\omega)\), \(0\le s\le1\),
puts this on a compact fixed interval. All derivatives in the
chart up to order \(r\) are continuous and bounded there, hence
can be taken under the integral. The Gaussian integral on the
orthogonal complement is one, since the height in this proposition
depends only on the active space.

Partition the sphere into \(2k\) patches according to the signed
largest absolute coordinate, ignoring ties on a surface-area
null set. Each patch has parameterization
\(\omega(u)=(\pm e_j+\sum_{i\ne j}u_i e_i)/\sqrt{1+\|u\|^2}\),
\(u\in[-1,1]^{k-1}\). Differentiating gives the metric
\((1+\|u\|^2)^{-1} I-(1+\|u\|^2)^{-2} uu^T\);
its eigenvalues are \((1+\|u\|^2)^{-1}\) transverse to \(u\),
and \((1+\|u\|^2)^{-2}\) along \(u\), with the same determinant
formula at zero. Thus the area density is
\((1+\|u\|^2)^{-k/2}\). Let \(L_*\) bound the Lipschitz
constants of the area-weighted \(G\) on these cubes. Take one
independent uniform point per equal cube cell in a subdivision
into \(m^{k-1}\) cells per patch. The cell diameter is
\(2\sqrt{k-1}/m\), and the cell estimator variance is bounded by
its volume squared times the squared range of the integrand
divided by four. With \(n=2k m^{k-1}\) this gives RMSE at most
\[
 2^{k-1}\sqrt{2k(k-1)}\,L_*\,m^{-(k+1)/2}.
\]
Locating each radius to \(\delta\) by bisection costs
\(O(1+\log(1/\delta))\) queries and changes the integrand
uniformly by at most a constant times \(\delta\); the radial
integral of the known height is an ordinary scalar integral
and needs no additional oracle values. Taking
\(\delta=n^{-1/2-1/(k-1)}\), one can choose the largest
integer \(m\) whose \(n\) searches fit in \(N\), with
\(n\) of order \(N/\log N\) for fixed \(k\). Add the uniform
error to the stratified RMSE bound. \(\square\)

The regularity hypothesis can hold for translated ellipsoids viewed
from an interior pivot, despite the lack of a globally monotone
straight-line direction. The assumptions are not implied by
\(\mathcal F_d\) alone, and finding the pivot and active space is a
separate problem. This addresses a star-shaped case of open problem 1.

## 3. Localization by differences of branches

### Lemma 2 (nonzero continuous knot, local query complexity)

On a known interval \([a,b]\) of length \(\ell\), suppose \(f\) has
one knot \(\theta\) and piece extensions \(h_-,h_+\) on
\([a-2\ell,b+2\ell]\), with \(h_+(\theta)=h_-(\theta)\),
\(\|h_\pm''\|_\infty\le H\), and
\(|(h_+-h_-)'|\ge j>0\) on \([a,b]\). The interval extensions must be
queryable in the correct pieces outside the bracket. The common
value at the knot is unrestricted. Four queries produce a certified
new interval of length at most \(24H\ell^2/j\).
For \(24H\ell/j<1\), iterating while the same conditions hold localizes
within \(\epsilon\) in \(O(\log\log(1/\epsilon))\) queries at fixed
\(H,j,\ell\), in exact values.

**Proof.** Linearly extrapolate on each side from \(a-2\ell,a-\ell\)
and \(b+\ell,b+2\ell\). For a \(C^2\) function the error at \(x\) is
bounded by \(H|(x-x_0)(x-x_1)|/2\): subtract the interpolant plus a
multiple of the quadratic vanishing at the two nodes and apply Rolle's
theorem twice at those nodes and \(x\). On \([a,b]\) the bound is
\(3H\ell^2\) per side. Let \(D=h_+-h_-\) and \(L\) be the difference of
the two interpolants. For \(\widehat\theta\) minimizing \(|L|\) on the
bracket, \(|L(\widehat\theta)|\le |L(\theta)|\le6H\ell^2\).
Then \(|D(\widehat\theta)|\le12H\ell^2\) and the mean value theorem
gives \(|\widehat\theta-\theta|\le12H\ell^2/j\). Intersect this interval
with the old bracket. If \(C=24H/j>0\), the recurrence satisfies
\(\ell_i\le C^{-1}(C\ell_0)^{2^i}\). A zero-length bracket is already
solved, and \(H=0\) gives the exact root in one step. \(\square\)

This is a certified local variant, not a theorem that the implemented
quintic initial scan finds all knots. For example two nearby switches
can corrupt a one-sided stencil. The linear proof protects against a
nonzero payoff at the crossing because it applies to \(D\), not to \(f\).

### Lemma 3 (paired extrapolation of a branch difference)

Suppose a site \(x\) and direction \(v\), \(\|v\|=1\), admit probes
\(x\pm j h v\), \(j=1,\ldots,m\), in their respective pieces, with
directional \(m\)-th derivatives of the two extensions bounded by \(B\)
on the stencils and up to \(x\). Define
\[
 D_h(x)=\sum_{j=1}^m (-1)^{j+1}\binom mj
         [f(x+j h v)-f(x-j h v)].
\]
Then \(D_h(x)\) differs from \(D(x)=h_+(x)-h_-(x)\) by at most \(2Bh^m\).
If \(D(x)=\pm A e^{b^Tx}\), \(|D|\ge J_*>0\) on these sites, replacing
\(D\) by \(D_h\) in
\[
 b_i=\frac{\log |D(x+\epsilon e_i)|-\log |D(x-\epsilon e_i)|}{2\epsilon}
\]
has error at most \(4Bh^m/(J_*\epsilon)\), provided
\(2Bh^m\le J_*/2\) and the same branch probes are valid at the shifted
sites. This uses \(2m(2d+1)\) queries including \(D_h(x)\) to recover the
height parameter.

**Proof.** In one variable the stated weights are the degree-\(m-1\)
Lagrange extrapolant at zero from the nodes \(h,2h,\ldots,mh\);
substituting the nodes in the Lagrange formula gives the binomial
weights. The interpolation remainder is bounded by the supremum of
the \(m\)-th derivative times \(\prod_j jh/m!=h^m\), by the
same Rolle argument with \(m\) derivatives. Apply it to both sides.
The logarithm on magnitudes at least \(J_*/2\) has derivative bounded
by \(2/J_*\), and the displayed log difference for the exact exponential
is exact at finite \(\epsilon\). \(\square\)

### Proposition 3a (quadratic columns and the active subspace)

Suppose the two extensions of an exposed interface are
\(h+Q_+\) and \(h+Q_-\), where here \(Q_+,Q_-\) **name two
quadratic polynomials**, not positive parts. Let \(A\) be the
symmetric coefficient matrix of \(Q_+-Q_-\).
Assume a pair of stencils centered at \(p\pm\delta v\) stays on
the respective sides, with \(v\) a unit vector. Suppose
\(|\partial_v\partial_i\partial_j h|\le B\) on the tube between
the stencils, and oracle values have absolute error at most \(\eta\).
With \(q\) chosen coordinate columns and forward step \(\ell>0\),
at most \(2(1+d+dq-q(q-1)/2)\) queries give a column matrix
\(\widehat C\) for \(C=A E\), where \(E\) selects the \(q\) coordinates.
Its error satisfies
\[
 \|\widehat C-C\|_2\le
 E_C=\sqrt{dq}\,(\delta B+4\eta/\ell^2).
\]
If \(\operatorname{rank}(C)=\operatorname{rank}(A)=r\), then \(C\)
determines \(A\) exactly when \(E_C=0\). More generally the span
of the top \(r\) left singular vectors of \(\widehat C\) estimates
the range of \(A\) with one-sided projection error at most
\(2E_C/\sigma_r(C)\).

**Proof.** At each center query the base, the \(d\) single
coordinate shifts, and the pair sums where at least one coordinate
is chosen, counting an unordered pair only once. This includes
the double shifts needed for the diagonal. The mixed second
forward difference of a quadratic divided by \(\ell^2\) equals
twice its coefficient matrix entry, even for a diagonal. For
\(h\), the same divided difference is the average of
\(\partial_i\partial_j h\) over the two forward offsets, by
the fundamental theorem of calculus twice. The averages for
the two centers differ by at most \(2\delta B\); halving the
whole difference gives the stated matrix entry error, since
each divided difference uses coefficients of total absolute
weight \(4/\ell^2\). The matrix norm is at most the Frobenius
norm, which yields the bound.

With exact columns, take orthonormal \(U\) spanning their range,
which equals the range of \(A\) by the rank hypothesis. Then
\(A=U R U^T\) and \(U^TC=R U^TE\). Here \(R\) is invertible
on the range; \(U^TE\) has full row rank, so its right inverse
recovers \(R\) and \(A\). For noisy columns, let \(\widehat P\)
project onto the top \(r\) left singular vectors of \(\widehat C\).
For the rank-\((r+1)\) top right singular space of \(\widehat C\),
some unit vector is in the kernel of \(C\); therefore
\(\sigma_{r+1}(\widehat C)\le E_C\) (trivially also when \(r\)
is maximal). It follows that
\(\|(I-\widehat P)C\|_2\le 2 E_C\).
Any unit vector in the range of \(C\) can be written \(Cw\)
with \(\|w\|\le1/\sigma_r(C)\), by the SVD, giving the claimed
projection error. \(\square\)

This explains the \(O(dq)\) reconstruction when a low-rank
difference is visible in the probed columns. The stated
hypotheses do not guarantee that a particular random choice of
coordinate columns is well conditioned, or that a second interface
does not cross a stencil. The code uses diagonal and independent
directional checks and can grow the sketch to all columns.

### Integrating a quadratic control

For \(Q(z)=z^TAz+b^Tz+c\) with \(A\) symmetric, diagonalize \(A\) and
write \(\lambda_j,b_j\) for the eigenvalues and rotated loadings.
The Gaussian integral in one dimension, then independence, gives
\[
 \varphi_Q(t)=\mathbb E e^{itQ(Z)}
 =\exp\left(itc-\frac12\sum_j\log(1-2it\lambda_j)
             -\frac{t^2}{2}\sum_j\frac{b_j^2}{1-2it\lambda_j}\right).
\]
The logarithms are continuous from \(t=0\). The identity
\[
 I(Q_+)=\tfrac12(\operatorname{tr} A+c)+
           \frac1\pi\int_0^\infty\frac{1-\Re\varphi_Q(t)}{t^2}\,dt
\]
uses \(\int_0^\infty (1-\cos(tx))/t^2\,dt=\pi|x|/2\).
For the latter constant, integrate by parts and integrate
\(e^{-\epsilon t}\sin(t)/t\) first: differentiating in the sine's
frequency gives the exponentially damped cosine integral and hence
\(\arctan(1/\epsilon)\). The undamped limit follows by integration by
parts on the tails of \(\sin(t)/t\); the undamped
\((1-\cos(t))/t^2\) is integrable. Taking expectations in the
nonnegative integral is justified by Tonelli, and
\(Q_+=(Q+|Q|)/2\). In the code the integral beyond a cutoff \(T\) in the
normalized variable \(Q/s\) is approximated by \(1/T\); the
resulting price error due to truncation is at most
\((s/\pi)\int_T^\infty |\varphi_{Q/s}(t)|/t^2\,dt\), also bounded
by \((s/\pi)\int_T^\infty
\prod_j(1+4(\lambda_j/s)^2 t^2)^{-1/4}t^{-2}\,dt\).
This bound and the integral identity apply to the *proposed* quadratic,
irrespective of the error of its local fit. If two caps lie in
orthogonal subspaces, their Gaussian projections are independent;
the survival of the maximum is \(p_1+p_2-p_1p_2\) for cap
survivals \(p_1,p_2\). Integrating this survival over positive levels
gives the stated maximum control.

If \(\widetilde Q\) is approximate, the positive-part map is
1-Lipschitz, and setting \(D=A-\widetilde A,\ e=b-\widetilde b,\
\kappa=c-\widetilde c\) gives
\[
 \|(Q)_+-(\widetilde Q)_+\|_{L^2(\gamma_d)}
 \le
 \sqrt{2\|D\|_F^2+\|e\|^2+(\operatorname{tr}D+\kappa)^2}.
\]
To check the right-hand side, diagonalize \(D\). Centered
Gaussian squares have variance 2, are independent across
coordinates, and their covariance with linear coordinates is
zero by parity. The mean of the difference is
\(\operatorname{tr}D+\kappa\). The same reasoning bounds the
error of the capped maximum by the sum of the two corresponding
bounds, since the maximum changes by at most the largest
pointwise component change. These bounds quantify the residual
size, not an additional bias if the residual is integrated exactly.

For a switch between two caps on orthogonal subspaces, there is
another exact-arithmetic reduction. Write
\(Q_i=x^TA_ix+b_i^Tx+c_i\) with \(A_i\) negative definite on
its own active subspace and zero on its orthogonal complement,
and \(b_i\) in that subspace. The negative spectral part of
\(A_1-A_2\) is \(A_1\), the positive part is \(-A_2\), and
projecting \(b_1-b_2\) onto those two subspaces gives \(b_1,-b_2\).
This follows by block diagonalization on the orthogonal direct sum;
on the remaining subspace both matrices vanish. The difference
determines \(c_1-c_2\), and either exposed zero boundary, when
correctly attributed, determines its cap constant by \(Q_i(x)=0\).
This explains why a nonzero internal kink can provide useful
information even when one further exposed switch is needed.
If the second cap is hypothesized to fill the entire complement of
the first projector \(P\), then \(A_2=-a(I-P)\) for a scalar \(a>0\).
For a correctly attributed switch, subtracting the known piece from
the branch comparison gives curvature \(-a\|(I-P)v\|^2\) in direction
\(v\). The gradient and value of the second cap at the switch determine
its remaining linear and constant terms once \(a\) is known.
The program checks several independent directional curvatures and the
gradient's projection before using this shortcut.

## 4. Finite-stencil identification of option pieces

### Theorem 4 (an exponential pencil)

On the particular stencils to be queried, suppose
\[
 f(z)=\sum_{j=1}^r a_j e^{b_j^Tz}-K,\qquad a_j>0,\quad r\le d .
\]
Let \(h\ne0\) be a step and \(v_{ji}=e^{h b_{ji}}-1\). Assume the
columns \(v_j=(v_{j1},\ldots,v_{jd})^T\) are independent and that
\(e^{b_j^T\Delta}\) are distinct. Query two stencils, at \(p\) and
\(p+\Delta\), each containing the center, its \(d\) shifts \(he_i\)
and all \(h(e_i+e_j)\), \(i\le j\), entirely within the same
smooth branch. In exact arithmetic the \((d+1)(d+2)\) values
determine \(r,(a_j,b_j)_{j=1}^r,K\), up to permutation. No limit
\(h\to0\) is needed.

**Proof.** Let \(w_j=a_je^{b_j^Tp}\). With \(\Delta_i\) the forward
difference, form \(g_i=\Delta_i f(p)\), \(H_{ij}=\Delta_i\Delta_j f(p)\)
and \(H'\) at the second center. Direct multiplication, including
when \(i=j\), gives
\[
 g=\sum_jw_jv_j,\quad H=\sum_j w_j v_j v_j^T,\quad
 H'=\sum_j w_j e^{b_j^T\Delta} v_j v_j^T .
\]
Diagonalize the rank-\(r\) PSD matrix \(H=U\Lambda U^T\) on its range
and let \(Y=U\Lambda^{-1/2}\). The \(r\) vectors
\(o_j=\sqrt{w_j}Y^Tv_j\) satisfy \(\sum_j o_j o_j^T=I_r\), so they
form an orthonormal basis. The symmetric matrix \(Y^TH'Y\) has these
eigenvectors with eigenvalues \(e^{b_j^T\Delta}\). If an eigenvector
\(\widetilde o_j\) is signed arbitrarily, let
\(s_j=\widetilde o_j^TY^Tg=\pm\sqrt{w_j}\). Then
\(v_j=U\Lambda^{1/2}\widetilde o_j/s_j\).
Recover \(b_{ji}=h^{-1}\log(1+v_{ji})\),
\(a_j=s_j^2 e^{-b_j^Tp}\), and \(K=\sum_j s_j^2-f(p)\).

If the first stencil has already determined \(H\), any coordinate
set \(J\) for which \(H_{J,J}\) retains rank \(r\) suffices for the
second stencil. Indeed the same proof whitens in those coordinates
to obtain \(o_j=\sqrt{w_j}Y^Tv_{j,J}\), with \(g\) restricted
to \(J\). Now \(H_{:,J}Y o_j=\sqrt{w_j}v_j\); division by the
signed square root from the restricted \(g\) recovers the complete
loading. Thus the second stencil can use only
\((|J|+1)(|J|+2)/2\) queries. \(\square\)

The finite-difference vectors are independent for sufficiently small
nonzero \(h\) if the \(b_j\) are independent, since \(v_j/h\to b_j\)
and a nonzero minor remains nonzero. For distinct loadings the bad
shifts lie in finitely many hyperplanes, but the same-branch condition
still matters. Conditioning deteriorates near a small eigenvalue of
\(H\), a small separation of the pencil eigenvalues, or small
\(w_j\); the theorem does not assert binary64 accuracy in those cases.
As a direction-identification result it is \(O(d^2)=o(N)\) for fixed
\(d\) or \(d=o(\sqrt N)\), *after* suitable stencil centers are found.
For a minimum of two sums it applies separately on each exposed
positive branch. It does not identify a digital payoff from local
differences away from a surface.

### Reduction for equal-time Brownian loadings

Assume the identified exponent rows have Gram blocks
\(\sigma_l\sigma_{l'}\rho_{ll'}\min(i,j)\), ordered within each asset
by dates \(i,j\), and coefficients \(a_{li}=s_l e^{i\mu_l}\).
The Gaussian law of the exponents depends on this Gram matrix, not
on their input directions. For one asset, or two with correlated
one-step log increments \(X_j\), define \(S_0=0\) and
\[
 S_j=e^{X_j}\odot(1+S_{j-1}).
\]
Here independence is across dates; the product and sum are componentwise.
Expanding the recurrence expresses \(S_j\) as sums of exponentials
of suffix increments. Reversing the iid **vectors** of increments
gives the required joint law of the two prefix path sums.
For \(F_j\) the CDF of \(\log S_j\), condition on the last increment
and use \(u\mapsto\log(e^u-1)\) for \(u>0\), interpreted as \(-\infty\)
otherwise, to obtain the convolution recursion implemented in
`perpetuity.py`. The CDF starts with the one-step Gaussian law.
The final payoff identities are
\[
 \mathbb E(X-K)_+=\int_K^\infty \mathbb P(X>t)\,dt,\qquad
 \mathbb E(K-X-Y)_+=\int_0^K \mathbb P(X\le t,Y\le K-t)\,dt
\]
for nonnegative \(X,Y\), together with
\((X+Y-K)_+-(K-X-Y)_+=X+Y-K\).
They follow by integrating the interval indicators pointwise and
using Tonelli. For a minimum use the first identity with the joint
survival. These are distributional reductions, not error bounds for
the FFT interpolation.

### Proposition 5 (a barrier facet encodes its date)

Assume one GBM asset, \(m\) equal dates, terminal stride \(\sigma>0\),
total log drift \(\mu\) and \(X_i=\mu i/m+b_i^Tz\)
with \(b_i^Tb_m=\sigma^2 i/m\) and \(\|b_i\|^2=\sigma^2 i/m\).
Let \(v=b_m/\sigma\). Suppose a ray \(z=x_0+t w\) from a positive
point crosses the relative interior of a single exposed face
\(b_i^Tz+\mu i/m=B\), with \(b_i^T w\ne0\) and strictly positive
vanilla payoff there. Nearby rays see the same face. If \(\lambda_j\)
is minus the change in crossing parameter per unit anchor displacement
in coordinate direction \(j\), then
\[
 i=m\frac{(\lambda^Tv)^2}{\|\lambda\|^2},\qquad
 \sigma\frac{\lambda^T z_{\rm face}}{\lambda^Tv}
    = B\,m/i-\mu.
\]
Thus \(d+O(1)\) scalar searches identify a face to arbitrary precision
under a local same-face margin. Two distinct dates for one barrier
and one identified date for the other (if present) determine \(\mu\)
and the log barriers, if the terminal payoff is itself identified.

**Proof.** Perturbing the anchor by \(y\) changes the ray crossing by
\(-b_i^Ty/(b_i^Tw)\) from the affine equation. Thus
\(\lambda=b_i/(b_i^Tw)\). The stated Brownian norms give the squared
cosine \(i/m\), and \(b_i^Tv=\sigma i/m\) gives the second formula.
The sign of \(\lambda^Tv\) distinguishes exit through the upper and
lower faces if \(w\) points from inside to outside. Given absolute
bracket error \(\delta\) and fixed nonzero displacement \(h\)
inside the same-face neighborhood, each component has error at most
\(2\delta/h\) (one crossing for the base). The cosine calculation
is continuous in a neighborhood of the nonzero \(\lambda\);
eventually rounding to the nearest date is exact. The formula for the
intercept is continuous there as \(\lambda^Tv\ne0\).
The equations for \(B,\mu\) are linear with columns \(m/i,-1\);
distinct dates give a nonsingular pair. With a common upper and
lower barrier, one then needs only one equation for the other
barrier. Each search uses \(O(\log(1/\delta))\) queries on a bounded
bracket. \(\square\)

**Directed search from one face.** With the full equal-date Brownian
Gram structure \(b_i^Tb_j=\sigma^2\min(i/m,j/m)\), write
\(t_j=j/m\). For an intermediate face \(i<m\), its normal oriented
towards increasing log-price gives the unit vector
\[
 u_i=\frac{b_i-\sigma t_i v}{\sigma\sqrt{t_i(1-t_i)}} .
\]
Direct use of the Gram identity shows that \(u_i^Tb_m=0\), whereas
\(u_i^Tb_j=\sigma(\min(t_i,t_j)-t_it_j)/\sqrt{t_i(1-t_i)}>0\)
for all intermediate \(j\). From a positive point the \(+u_i\)
ray cannot hit a lower face before an upper one, and the
\(-u_i\) ray cannot hit an upper face before a lower one. The
terminal payoff stays fixed. This is a search rule, not an assumption
about which date the ray exposes. Also, when the initial spot is
strictly between the enabled barriers, projecting a positive point
onto the terminal axis retains positivity: its successive log prices
are the linear interpolation between zero and the same terminal
log-price, which are both strictly admissible. In floating point,
the code still tests the axial anchor and the proposed face labels.

For the recovered model, condition on each preceding log-price.
The elementary kernel for the killed walk is a normal transition
density restricted to the barrier interval. The last step integrates
the positive discounted call against that density. For a normal
\(Y\sim N(m_0,s^2)\), \(\mathbb E[e^Y\mathbf 1_{l<Y<u}]
=e^{m_0+s^2/2}[\Phi((u-m_0-s^2)/s)-\Phi((l-m_0-s^2)/s)]\)
by completing the square; subtracting the strike times the interval
probability proves the terminal formula. Repeated conditional
expectation proves the backward recursion. If a down-only model
truncates intermediate log-prices above \(H\), the omitted price is
at most
\[
 A e^{\sigma^2/2}
 \sum_{i<m}\bar\Phi\!\left(
 \frac{H-(\mu+\sigma^2)i/m}{\sigma\sqrt{i/m}}\right),
\]
for terminal discounted underlying \(A e^{X_m-\mu}\). Indeed bound
the payoff by that underlying, use the tilt by \(b_m\), and apply
the union bound under the tilted walk with means
\((\mu+\sigma^2)i/m\). This is a truncation bound, not a certified error
bound for the finite transition quadrature or imperfect face discovery.

## 5. Randomized conditioning and residual error

If conditional on a transverse point \(x\) the digital payoff divided
by its height is \(\mathbf 1_{t>\theta(x)}\), and an exact bracket
\([l,u]\) is found, let \(U\) be uniform on
\([\Phi(l),\Phi(u)]\) independently after constructing the bracket.
Then
\[
 Y(x)=1-\Phi(u)+(\Phi(u)-\Phi(l))
                  \mathbf 1_{f(x+\Phi^{-1}(U)v)>0}
\]
has conditional mean \(1-\Phi(\theta(x))\) and variance at most
\((u-l)^2/(8\pi)\). In fact the binary term has success probability
\((\Phi(u)-\Phi(\theta(x)))/(\Phi(u)-\Phi(l))\); its variance is
at most \(1/4\), and \(\Phi\) is \(1/\sqrt{2\pi}\)-Lipschitz.
The degenerate zero-width bracket gives the conditional mean directly.
For a fixed \(n\)-slice design, the squared contribution from the
independent fills is at most
\(\sum_i (u_i-l_i)^2/(8\pi n^2)\), conditional on the slices and
their brackets. Bisection decreases the widths geometrically; a
midpoint estimate without the random fill instead has deterministic
per-slice bias bounded by the width divided by \(2\sqrt{2\pi}\).

For a digital exponential sum and direction \(v\) with
\(\alpha_i=b_i^Tv\ge \alpha_*>0\), every transverse slice has a
unique root of \(\sum_i a_i e^{(b_i^\perp)^Tx+\alpha_i t}=K>0\).
The sum is increasing from zero to infinity. At the root, use weights
\(w_i=a_i e^{(b_i^\perp)^Tx+\alpha_i\theta(x)}/K\). Implicitly
differentiating once and twice gives
\[
 \nabla\theta=-\frac{\sum_i w_i b_i^\perp}{\sum_i w_i\alpha_i},
 \qquad
 \nabla^2\theta=
 -\frac{\sum_i w_i (b_i^\perp+\alpha_i\nabla\theta)
                   (b_i^\perp+\alpha_i\nabla\theta)^T}
        {\sum_i w_i\alpha_i}.
\]
The denominator is at least \(\alpha_*\), so the root is smooth
with \(\|\nabla\theta\|\le\max_i\|b_i^\perp\|/\alpha_*\) and a
bounded, negative semidefinite Hessian. The direction guessed by the
program is not certified to satisfy this positivity condition.

More generally, with a fixed pointwise residual \(r=f-q\) and
an integrated control \(J=I(q)+\varepsilon_J\), for sites of Gaussian
marginal law and a control \(C\) learned independently of these sites,
\[
 J+I(C)+\frac1n\sum_i (r(Z_i)-C(Z_i))
\]
has mean \(I(f)+\varepsilon_J\), by linearity and conditioning on
the training data. Conditional on that data, with independent samples
its variance is \(\operatorname{Var}(r-C)/n\). A deterministic
antithetic map replaces \(r-C\) by its even part, which has no
larger variance by Jensen. Linearity also proves unbiasedness of
fixed-weight cross-fitting with mutually independent scrambles,
even though the different cross-fitted corrections need not be
independent. For such corrections the square root of the variance
of the weighted sum is at most the sum of their individual standard
deviations times their absolute weights, by the triangle inequality
in \(L^2\). This is an analysis of the ideal Gaussian-marginal model
with fixed independent designs: finite-bit nets, validation-dependent
fallbacks/damping, imperfect labels and numerical \(J\) are separate
effects. No general rate or coverage is inferred from observed
agreement on a few stencils.

## 6. An adaptive randomized lower bound and a coverage obstruction

### Theorem 6 (thin shells, fixed analytic bounds)

For any \(d\ge2\) and \(N\ge1\), worst-case randomized RMSE on
\(\mathcal F_d(2,\infty,4,1)\) is at least
\(e^{-1}/(8\sqrt2 N)\) for algorithms making at most \(N\) adaptive
point queries. This holds on a subclass with integrals between 1
and 2. For deterministic algorithms there is a worst-case absolute
error at least \(e^{-1}/(4(N+1))\). Moreover, an interval with
nominal coverage \(1-\alpha\) on the whole subclass cannot, on the
constant oracle, have half-width below \(e^{-1}/(8N)\) with
probability greater than \(3\alpha\).

**Proof.** For \(m\) equal disjoint open subintervals of \([1,2]\),
take \(f_0=1\) and \(f_j=1+\mathbf 1_{\{s(z)\in J_j\}}\), where
\(s(z)=z_1^2+z_2^2\); use value 1 at endpoints. The two defining
polynomials \(s-a_j,s-b_j\) and constant pieces satisfy the
derivative bounds with \(M=4,c=1\). In fact the polynomials are
bounded by \(|z|^2+2\le4 e^{|z|}\), the first derivatives by
\(2|z|\), the second by 2 and the rest are zero. The surface
gradient norm is at least 2. These are regular closed shells in a
two-dimensional projection. Since \(s(Z)\sim\chi^2_2\),
\(\Delta_j=I(f_j)-I(f_0)\ge e^{-1}/(2m)=:\Delta\).

For a fixed seed, the query sequence on \(f_0\) intersects at most
\(N\) distinct shells. For every untouched shell, the transcript
and the output on \(f_j\) equal those on \(f_0\), by induction on
the adaptive query index. With \(m=N+1\), one of the two deterministic
errors is at least \(\Delta/2\).
For randomized algorithms use \(m=2N\) and input distribution giving
probability \(1/2\) to \(f_0\) and \(1/(2m)\) to each shell.
On any untouched pair, the sum of the squared errors on the two
oracles with the same seed is at least \(\Delta^2/2\), as the best
common output is the midpoint. Averaging over seeds and the uniform
shell first, then giving each of the two alternatives weight \(1/2\),
gives MSE at least \((1-N/m)\Delta^2/4\). Worst-case RMSE is no
smaller than its square root.

For coverage let \(E\) be the seeds on \(f_0\) where the interval
contains \(I(f_0)\) and the half-width is less than \(\Delta/2\).
On every seed in \(E\) at least \(1-N/m\) of the shells have the
same interval, which cannot include their integral. The average
noncoverage across shells is at least
\(\mathbb P_{f_0}(E)(1-N/m)\), and by the hypothesis it is at most
\(\alpha\). Add the at-most-\(\alpha\) noncoverage on \(f_0\) itself;
with \(m=2N\) this gives \(3\alpha\) as claimed. \(\square\)

The result uses two surfaces with no required separation, not
shrinking balls whose defining functions might violate the uniform
nondegeneracy or derivative bounds. It neither proves the minimax
rate on the full class nor contradicts rapid identification given an
interior point and a one-surface model.

## 7. Comparisons and experimental scope

The names in this table refer to the bibliography in the provided
appendix. "Extension" here refers to the particular subclass and
stated oracle assumptions, not a dominance claim over all inputs.

| Appendix section 5 result | Comparison |
|---|---|
| Bakhvalov; Hinrichs et al., smooth worst-case rates | Not improved. Theorem 6 gives a different obstruction with fixed analytic bounds on each piece and nonconvex support. |
| Kuo, Sloan, Wozniakowski, Gaussian analytic RKHS | No claim on the RKHS or its tractability. The finite-stencil results exploit identifiable parameter structure beyond analyticity. |
| Owen, smooth scrambled nets | Used as motivation for the residual nets only. We do not invoke the stated inverse-Gaussian mixed-smoothness rate. |
| Plaskota and Wasilkowski, unknown singularities in one dimension | Lemma 2 gives an explicit **local** forward-extrapolation bound for an internal nonzero kink. It neither replaces their global search nor changes their worst-case rate. |
| Alexander; Matousek, half-space discrepancy | The structured estimators are not equal-weight rules on the discontinuous integrand. No improved discrepancy bound is claimed. |
| He and Wang, general discontinuous RQMC | The chord result removes an identified jump instead of changing the generic RQMC exponent. |
| Griebel, Kuo and Sloan; Griewank et al., monotone preintegration | The digital root argument recovers their type of smoothing for positive exponential slopes. The closed projected-sphere computation is an alternative not requiring a globally monotone direction. |
| He, conditional RQMC | The digital conditional estimator uses an extra randomized threshold with an explicit bracket-level variance bound. No stronger outer-net convergence rate is claimed. |
| Gilbert, Kuo and Sloan, nonmonotonicity | No claim that conditioning on a fixed axis smooths closed spheres or indefinite quadrics. Geometry and characteristic-function integration avoid that obstruction on their subclasses. |
| Liu and Owen, active directions | Gradient sketches are heuristic. Theorems 1 and 4 and Propositions 3a and 5 give point-query identification for specific structures with exposed probes, partially addressing open problem 2. |
| Bayer, Ben Hammouda and Tempone, numerical smoothing | Bracketed one-dimensional smoothing is shared. Lemmas 2 and 3 apply to a difference of extensions, including a kink away from zero; these and Proposition 5 partly address open problem 3. |
| Hickernell and Jimenez Rugama; Clancy et al., certified Walsh cones | We assume no cone and claim no certificate. Theorem 6 bounds what uniform intervals can achieve without further assumptions (open problem 5). |
| Heinrich, randomized adaption | The shell lower bound covers adaptive randomized algorithms on this specific analytic subclass. No adaptive/nonadaptive separation is claimed (part of open problem 4). |
| He, axis-parallel discontinuities | Chords and fitted facets use rotation-invariant quantities; no appeal to axis alignment in the original input. |
| Hoyt and Owen, mean dimension of ridges | The exponential digital root can remove the largest discontinuous direction when one is known, but we do not estimate or bound mean dimension. |
| Chen, Xu, Du and Wang, preintegration and importance sampling | The exact exponential tilt in Theorem 1 and the barrier truncation estimate are Gaussian changes of measure, not an improvement of their RQMC rate. |

The samples suggest one indefinite hinge in H1, two projected caps
with an internal switch in H2, and closed projected spheres with
log-linear or constant jump heights in H3/H3t. These are empirical
model fits, not assumptions about every held-out instance. When the
fits are accurate, the error on H1/H2 should be governed chiefly by
integration of the smooth residual, and the large amplification of
binomial error in H3t should disappear once the chord probability
has been computed. When an interior point is missed, the latter
conclusion need not apply. Theorem 4 is relevant to arithmetic Asian
and basket calls and to each positive exposed branch of the rainbow
call; Proposition 5 is relevant to the two barrier families. The
digital theorem predicts a small bracket-filling variance, not
exact recovery of the entire option model.

### Measured sample results

These are the provided sample oracles under the 30-second JSON Lines
harness, launched from a read-only copy of the submission. The error
column is the harness's clipped \(\log_{10}\) relative error; an
entry at the floor does not establish the actual error below it.
The seed is the one supplied for the sample; this is not an average
over independently chosen seeds.

| Sample | Clipped log error | Queries | Gain over reference |
|---|---:|---:|---:|
| H1, \(d=8,k=4\) | -5.31 | 14,798 | +0.83 |
| H1, \(d=32,k=32\) | -4.94 | 16,364 | +1.91 |
| H2, \(d=8,k=8\) | -5.08 | 15,616 | +1.21 |
| H2, \(d=32,k=4\) | -3.92 | 16,268 | -0.18 |
| H3, \(d=8,k=8\) | -6.03 | 15,592 | +2.00 |
| H3, \(d=32,k=16\) | -4.32 | 16,300 | +1.09 |
| H3t, \(d=8,k=2\) | -6.74 | 14,989 | +2.00 |
| H3t, \(d=32,k=32\) | -4.15 | 16,216 | +2.00 |
| Arithmetic Asian, \(m=64\) | -6.16 (floor) | 6,238 | +2.00 |
| Basket Asian, \(m=16\) | -5.38 (floor) | 2,830 | +2.00 |
| Digital Asian, \(m=64\) | -4.47 | 16,381 | +2.00 |
| Rainbow Asian, \(m=16\) | -5.76 (floor) | 4,251 | +2.00 |
| Down-and-out, \(m=64\) | -8.48 | 7,101 | +2.00 |
| Double-knock-out, \(m=16\) | -6.70 | 6,326 | +2.00 |

There were no protocol failures. The illustrative sample score is
92.0/100; the longest local elapsed time in this run was 4.2 seconds.
In the rare jump samples the fitted inside probabilities were
about 0.00548 and 0.00378, with corresponding jump means of about
1.447 and 1.263. For comparison, an independent Bernoulli estimate
of a probability \(p\) with \(N\) observations has relative
standard deviation \(\sqrt{(1-p)/(Np)}\), about 11% and 13% for
these probabilities at \(N=16384\). Estimating the projected
probability structurally avoids that sampling term; the remaining
smooth residual error was much smaller.

The residual on the 32-dimensional cap sample is a counterexample
to treating a good geometry fit as proof of an accurate total.
Its fitted cap model reduced a fourth-difference roughness score
on the exploratory lines by more than three orders of magnitude,
but the error of the remaining integral still exceeded the
reference on this particular sample. The bounded derivative and
conditioning constants in the propositions are not estimated
uniformly from the oracle, so neither the residual estimates nor
agreement with the sample references certifies a rate on the
complete hidden distribution.
