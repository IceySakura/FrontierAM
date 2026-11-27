# Black-box Gaussian integration across closed jumps

## Target and comparison

**Primary target: A (nonmonotone jumps), with a comparison to B and to
high-precision membership-oracle volume estimation.**

At fixed dimension, the results give sharp different adaptive and
nonadaptive error exponents on one class of regular closed jumps
(R1–R4). Under a uniform analytic chart bound, adaptivity gives
stretched-exponential decay while every nonadaptive rule is subject to
an inverse-budget obstruction (R5–R7, R9). Without a supplied interior
point, adaptation itself can require exponentially many queries in
dimension for relative accuracy (R8). With a known center and a
spectral band, centered ellipsoids instead admit a dimension-explicit
relative guarantee with only quadratically many rays (R10). All of
these count queries of the discontinuous input, not queries of a
preintegrated function. The Lean statements give the finite bounds and,
in R9, the all-sufficiently-large-budget separation. The rates below
are consequences of those bounds.

- Gilbert, Kuo and Sloan, [arXiv:2112.11621](https://arxiv.org/abs/2112.11621),
  **Theorem 1**: let \(\varphi\in C^2(\mathbb R^d)\) and suppose that at
  \(y^*\) one has \(D_1\varphi=0\), \(D_1^2\varphi\ne0\), and
  \(\nabla\varphi\ne0\). For \(t=\varphi(y^*)\), the Cartesian
  preintegral \(P_1\mathbf1_{\{\varphi>t\}}\) has a square-root
  singularity at \(y^*_{-1}\) along any line through that point not
  orthogonal to \(\nabla_{-1}\varphi(y^*)\). The authors' conclusion
  identifies the failure of the order-one dominating mixed smoothness
  condition as an obstacle to a QMC guarantee. **R1 makes progress on
  that obstacle by changing the fibers, not by disputing the
  square-root singularity.** It uses nonlinear radial fibers and has a
  faster-than-Monte-Carlo rate charged to the original point oracle.
- He, [arXiv:1708.09512v2](https://arxiv.org/abs/1708.09512v2),
  [DOI:10.1137/18M118270X](https://doi.org/10.1137/18M118270X),
  **Theorem 4.8** (conditional QMC): for \(g\mathbf1_{\{\varphi\ge0\}}\)
  with \(g,\varphi\in C^r\), it assumes \(D_j\varphi\ne0\) everywhere
  and the derivative, inverse-derivative, and density-tail bounds
  of Assumptions 3.3, 4.5, and 4.6. The first conclusion is
  \(P_j f\in C^r\) if \(B_j<1/(2r+1)\). For \(r\ge d-1\),
  \(B_j<1/(2d-1)\), and
  \(\gamma_j=(2d-1)\max_{i\ne j} B_i+B<1\), the conditional RQMC
  expected absolute error is \(O(n^{-1+\gamma_j+\epsilon})\);
  it is \(O(n^{-1+\epsilon})\) when the growth exponents \(B_i,B\)
  are arbitrarily small. R1, R5 and R6 are incomparable with this
  theorem: the latter covers nonconstant amplitudes and exact
  Cartesian preintegration under monotonicity, whereas the radial
  method counts point queries even for a jump with no such
  Cartesian direction.
- He and Wang, [DOI:10.1137/15M1007963](https://doi.org/10.1137/15M1007963),
  **Theorems 3.5–3.6**, as restated in He,
  [arXiv:1702.03361](https://arxiv.org/abs/1702.03361), **Proposition 2.3**:
  for \(g\in L^2([0,1]^d)\) of bounded Hardy–Krause variation and
  \(\partial\Omega\) of finite \((d-1)\)-dimensional Minkowski content,
  the first \(n\) points of a scrambled \((t,d)\)-sequence give variance
  \(O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})\) for \(g\mathbf1_\Omega\).
  If \(\Omega=\Omega_u\times\prod_{i\notin u}[a_i,b_i)\) and
  \(\partial\Omega_u\) has the corresponding Minkowski content, the
  rate has \(d_u=|u|\) in place of \(d\) in both exponents. The bound's
  prefactor includes
  \(\max(V_{\mathrm{HK}}(g),\sup|g|)^2\).
  **R1 has a strictly faster exponent on the common class of smooth
  bounded radial indicators, but is incomparable as a theorem:** it
  needs radial structure and uses adaptivity, not their scrambled
  sequence. In particular, it does not improve a theorem about the
  variance of that same scramble. To identify the overlap, apply the
  standard normal CDF coordinatewise. A smooth compact jump surface
  in a finite Gaussian annulus maps to a smooth compact boundary in
  the interior of the cube, with finite Minkowski content; here \(g=1\).
- Cornelissen, Apers and Gribling,
  [arXiv:2503.02207v1](https://arxiv.org/abs/2503.02207v1),
  **Theorems 3.4 and 4.6**: for fixed \(d\ge2\), a convex body
  \(B_d\subseteq K\subseteq R B_d\) with a membership oracle admits a
  relative-\(\varepsilon\) Lebesgue volume estimate with success
  probability \(2/3\) using
  \(\widetilde O(\varepsilon^{-2(d-1)/(d+3)})\) queries. The
  randomized volume lower bound is
  \(\Omega_d(\varepsilon^{-2(d-1)/(d+3)})\) in the fixed-roundness,
  high-precision regime; the paper also studies deterministic and
  quantum complexity. Their upper bound uses an inner/outer
  approximation and sampling in the volume gap. **R1 is
  incomparable:** it allows nonconvex shapes and works for Gaussian
  mass, with the same precision exponent and no polylogarithmic query
  factor when the radial charts have a uniform quadratic Taylor
  remainder. It does not apply at that rate to arbitrary nonsmooth
  convex bodies. **R3** matches their adaptive precision exponent
  for Gaussian RMSE while restricting the hard inputs to regular
  radial surfaces. It is likewise incomparable to the Lebesgue
  volume lower bound. No novelty is claimed for the general ideas
  of using inner/outer approximations or embedding a bit string
  in local boundary perturbations. **R5–R6** use stronger
  regularity and have faster precision rates than the generic
  upper bound, up to a polylogarithmic query complexity in R6.
  They are incomparable as theorems, since the generic result
  needs no radial Taylor condition and studies Lebesgue volume.
- **C and appendix problem 4, specifically the effect of adaptivity.**
  Rademacher and Vempala, [arXiv:cs/0608054v2](https://arxiv.org/abs/cs/0608054v2),
  **Theorem 2 (volume)** gives an \(\Omega(d^2/\log d)\) randomized
  membership-query lower bound for Lebesgue volume of
  \(B_d\subseteq K\subseteq O(d^8)B_d\), for a fixed constant relative
  error and success probability at least \(1-1/d\). Their
  **Theorem 8 (nonadaptive lower bound), §5** gives at least
  \(d^{d/2}/(2e(d+2))\) nonadaptive queries for a \(10\%\) relative
  volume estimate with success probability \(3/4\) on convex
  \(B_d\subseteq K\subseteq2dB_d\). **R2 is incomparable** to these
  bounds: it is a precision-dependent lower bound on absolute
  Gaussian RMSE for smooth radial jumps at every fixed dimension.
  With R1 it exhibits a strict separation of error exponents on
  the *same* radial class. Cornelissen–Apers–Gribling's Theorem 4.6
  measures adaptive query complexity and does not supply this
  nonadaptive precision lower bound. The distinction is in the
  quantifier on query locations: an unknown common radial level
  prevents fixed query sites from focusing their calls on the
  perturbations. **R4** gives the matching nonadaptive upper
  bound on the quadratic chart class. Like R1, it is
  incomparable with the general scrambled-sequence theorems,
  but R4 keeps all query locations independent of the answers.
  **R7** uses only constant radii and is not meant as a new
  theorem on scalar threshold identification by itself. Its role
  is to show that even on the *same uniform analytic class* as
  R6, the stretched-exponential upper bound requires adaptivity.
  It is incomparable with the dimension-dependent bounds of
  Rademacher–Vempala and the adaptive bound of
  Cornelissen–Apers–Gribling.
  **R8** concerns the relative-accuracy, large-dimension regime,
  but with no known interior point. It shows that this difference
  in information can be decisive even for regular analytic
  spheres lying in a fixed coordinate box. It is incomparable
  to Theorem 2 of Rademacher–Vempala, not a removal of their
  logarithmic factor on their input class. **R9** gives the exact
  same-budget comparison of R6 and R7 on one analytic class.
  It is likewise incomparable with the cited arbitrary-convex-body
  bounds, which do not distinguish those two precision regimes
  for uniformly analytic radial jumps.
- **C, the centered ellipsoid upper bound (R10).** Rademacher–
  Vempala's **Theorem 3 (determinant)** concerns an unknown
  \(d\times d\) matrix \(B\) with entries in \([-1,1]\) and
  smallest singular value at least \(2^{-12}d^{-7}\). From
  membership in \(\{x:\|Bx\|_\infty\le1\}\), estimating
  \(|\det B|\) to fixed constant relative accuracy with success
  probability \(1-1/d\) requires \(\Omega(d^2/\log d)\) queries.
  R10 instead gives a deterministic precision-dependent upper
  bound for **Gaussian mass** of centered ellipsoids with a
  known spectral band. This comparison is **incomparable**:
  the bodies, measure, and quantity are different. Compared to
  Cornelissen–Apers–Gribling Theorem 3.4, R10 has logarithmic
  rather than polynomial dependence on inverse precision and
  explicit polynomial dependence on dimension, but only because
  the quadratic oracle has finitely many unknown coefficients.
  No general convex-body improvement is asserted.

## Model and hypotheses

Let \(d=p+1\ge2\), \(S=S^{d-1}\), and let \(\sigma\) be normalized
surface measure on \(S\). Let \(0<a\le b\le40\) be known. A measurable
radius \(R:S\to[a,b]\) determines the strict radial threshold set

$$
K_R=\{0\}\ \cup\ \{tu:u\in S,\ 0<t<R(u)\},\qquad
f_R=\mathbf1_{K_R}.
$$

The input oracle is **only** \(z\mapsto f_R(z)\). In particular, neither
\(R\), its derivatives nor a defining function of the boundary is
available. The target is \(I(R)=\gamma_d(K_R)\), with standard
\(N(0,I_d)\) law. Every query has each coordinate in \([-40,40]\).
We use the usual exact-real, query-only information model. Deterministic
computations from the transcript and independent random samples from
known distributions are not counted as oracle calls. We do not claim
a bound under the numerical 30-second limit of the experimental task.

For each of the \(2d\) pairs \(f=(i,\pm)\), put the indicated sign
in coordinate \(i\) and fill the other coordinates with \(2x_j-1\),
\(x\in[0,1]^p\). Write \(v_f(x)\) for the resulting vector and
\(u_f(x)=v_f(x)/\|v_f(x)\|_2\) for its unit direction. Every direction
is covered by the faces for which \(|u_i|\) achieves the maximum
absolute coordinate, with sign equal to that of \(u_i\).

Fix \(s\in\mathbb N\) and \(A\ge(b-a)/2\). The required regularity is
the following precisely stated **chart Taylor condition**: for every
face \(f\) and \(x\in[0,1]^p\) there is a vector \(\beta_f(x)\in\mathbb R^p\)
such that for every \(y\in[0,1]^p\) and \(\delta\ge0\) with
\(\max_j|y_j-x_j|\le\delta\),

$$
\left|R(u_f(y))-R(u_f(x))-\sum_{j=1}^{p}\beta_{f,j}(x)(y_j-x_j)\right|
\le A\delta^{s}.                                                     \tag{1}
$$

This is `ChartRemainder` in Lean. The main use is \(s=2\);
if the chart derivative is Lipschitz with constant \(H\) from the
sup norm to its dual norm, integrating the derivative along the
segment gives (1) with \(A\ge H/2\). For \(s=1\), chartwise Lipschitz
radii satisfy (1) with \(\beta=0\). The algorithm is told \(A,s,a,b\)
but does not need the \(\beta\)'s.

For example all centered ellipsoids
\(K=\{z:z^\top Qz<1\}\) with
\(b^{-2}I\preceq Q\preceq a^{-2}I\) have analytic radii
\((u^\top Q u)^{-1/2}\). There is a common finite \(A(d,a,b)\)
for (1) with \(s=2\), by joint continuity of the chart Hessians on
the compact product of the chart cube and the specified matrix set.
The orientation, eigenvalues, and defining quadratic are unknown.
The class is not limited to ellipsoids or convex shapes. A
\(C^{1,1}\) radius on the whole sphere (in smooth charts) with positive
lower radius yields a \(C^1\) regular closed jump hypersurface.
For the appendix's \(r=1\) class, one may use a radial cutoff
\(\eta\) that is zero below \(a/4\) and one above \(a/2\):
\(\eta(\|z\|)(\|z\|-R(z/\|z\|))-(1-\eta(\|z\|))a/2\) is a global
\(C^1\) defining function. Its radial derivative is 1 on the
boundary. It and its first derivatives obey the appendix's bounds
for suitable \(c>0,M\), and the piece extensions are constants.
For ellipsoids the quadratic defining function itself meets the
bounds, including for \(r=\infty\) with suitable \(M,c\).

These hypotheses are relevant to the closed, anisotropic jump
geometry of the difficult track, including a rotated full-rank
ellipsoid whose interior contains the origin and whose boundary
lies in the given annulus. They do not imply that the full hidden
integrands with an unknown added smooth term are membership
oracles. The upper bounds in this model do not apply to a translated
rare-event component missing the origin, cylindrical jumps with
unbounded radius, internal kinks, or the option payoffs as given.
R8 addresses the unknown-center case separately as a lower bound.

## R1: adaptive cubature with no per-node logarithm

**Lean:** `Cubature.gaussian_radial_adaptive_cubature`.

For every integer level \(k\ge0\) and sample count \(n\ge1\) there
is a single randomized adaptive algorithm, independent of the
unknown \(R\), with the following guarantees on the class above.
Define

$$
B_p=2(p+1)2^p,\qquad
W_b=\operatorname{area}(S^{d-1})(2\pi)^{-d/2} b^{d-1}.
$$

It uses at most \(2 B_p(2^k)^p(s+1)+n\) memberships, including all
pilot queries. Its real output \(\widehat I\) is measurable in the
seed, is unbiased, and satisfies

$$
\mathbb E(\widehat I-I(R))^2
\le \frac{(2A W_b\,2^{-sk})^2}{n}.                                  \tag{2}
$$

In particular for fixed \(d,a,b,A,s\), choose \(n=\lfloor N/2\rfloor\)
and take the largest \(k\) with \(2B_p(s+1)2^{pk}\le N-n\), when
the budget is large enough. Then \(2^{pk}=\Theta_{p,s}(N)\) and
the worst-case absolute RMSE is \(O(N^{-1/2-s/(d-1)})\).
For \(s=2\) the variance exponent is \(1+4/(d-1)\). Relative RMSE
has the same rate with the additional factor
\(\gamma_d(B(0,a))^{-1}\). Squared-error Markov gives a confidence
radius \(2A W_b 2^{-sk}/\sqrt{n\alpha}\) at level \(1-\alpha\).

### Proof

**Gaussian radial identity.** For \(Z\sim\gamma_d\), write \(Z=TU\).
Lebesgue polar coordinates and the Gaussian density give independent
\(U\sim\sigma\) and \(T\) with density

$$
w(t)=\mathbf1_{\{t>0\}}\operatorname{area}(S^{d-1})(2\pi)^{-d/2}
t^{d-1}e^{-t^2/2}.
$$

Consequently \(I(R)=\int_S F(R(u))\,d\sigma(u)\), where
\(F(r)=\int_{-\infty}^r w(t)\,dt\). Notice \(0\le w(t)\le W_b\) on
\([a,b]\). The Lean proof uses `stdGaussian`, derives the product
normal density and uses the Haar polar change of variables; it
does not postulate a factorized Gaussian model.

**Interpolation.** At level \(j\), let \(m=2^j\) and use all tensor
nodes of mesh length \(1/m\) on every face, at most \(2d(m+1)^p\)
nodes including duplicates. In one coordinate \(x_i\), put
\(h=\min(m-1,\lfloor m x_i\rfloor)\) and
\(\theta=mx_i-h\). The weights \(1-\theta,\theta\) are placed on
nodes \(h/m,(h+1)/m\). They are nonnegative, sum to one, and have
weighted first moment \(x_i\). Multiplying the coordinate weights
therefore gives nonnegative weights summing to one, whose weighted
coordinate moments are \(x_i\), and whose nonzero support is within
sup-norm distance \(1/m\) of \(x\). The affine terms of (1) cancel
in the weighted sum. With exact nodal radii, the interpolant at
every direction has error at most \(\rho_j=A2^{-sj}\).
If several faces represent the same direction, average over the
active faces; positivity and the same error bound are preserved.
All these operations, including the tie rule, are Borel. If the
nodal radii have errors at most \(\rho_j\), positivity gives
additional interpolation error at most \(\rho_j\); write \(P_j\)
for this interpolant, so \(|P_j-R|\le2\rho_j\).

**Pilot.** Set the level-zero values to \((a+b)/2\) without a
query. To compute the value at every level-\((j+1)\) node \(u\),
clip the two endpoints of
\([P_j(u)-2\rho_j,P_j(u)+2\rho_j]\) to \([a,b]\).
Clipping preserves order and containment of \(R(u)\) and does not
increase width. Bisect the resulting interval \(s+1\) times,
querying \(f_R(tu)\) at each midpoint \(t\). A positive answer
keeps the upper half, a negative answer keeps the lower half;
either half contains the boundary at equality. Use the midpoint of
the last interval as the nodal estimate. Its error is at most
\(4\rho_j/(2\cdot 2^{s+1})=\rho_{j+1}\). This proves the pilot
invariant by induction. The number of nodes at level \(j\) is at
most \(B_p2^{pj}\); summing over the queried levels costs at most
\(2 B_p2^{pk}(s+1)\), since \(p\ge1\). Notice that the initial
bracket for each new ray is already of the previous level's
precision, so the number of bisections per new node is independent
of the level. Taking every ray from \([a,b]\) keeps the Cartesian
query coordinates within the cap.

**One-query randomized correction.** From \(P_k\) form the
clipped functions \(L(u),H(u)\) as in the pilot. Let \(D(u)=H(u)-L(u)\).
One has \(L\le R\le H\) and \(0\le D(u)\le4\rho_k\). Compute the
known baseline \(C=\int_S F(L(u))\,d\sigma(u)\) from the transcript.
It depends on no further membership information. For \(n\) independent
pairs \(U_i\sim\sigma,\ V_i\sim{\rm Unif}[0,1]\), independently within
each pair, compute \(T_i=L(U_i)+D(U_i)V_i\), make one query at
\(T_i U_i\), and return

$$
C+\frac1n\sum_{i=1}^n X_i,\qquad
X_i=D(U_i)w(T_i)f_R(T_i U_i).
$$

When \(D=0\) this sample term is zero. Otherwise conditional change
of variables gives
\(\mathbb E[X_i\mid U_i=u]=\int_{L(u)}^{R(u)}w(t)\,dt\).
The endpoint has Lebesgue measure zero in the nonzero-width case.
This proves unbiasedness. Each \(X_i\in[0,4\rho_k W_b]\), so
\(\operatorname{Var}X_i\le(2\rho_k W_b)^2\), for example by
bounding its squared distance from the midpoint of that interval.
Independence yields (2). The baseline is exact in this theorem;
to use numerical integration of the known baseline instead one
can approximate it to an absolute accuracy below the RMSE target
without further querying \(f_R\), though this would no longer be
the exact unbiased estimator.

### What is resolved

For this subclass there need not be any direction of strict
Cartesian monotonicity. For instance, even on a sphere a Cartesian
tangent fiber has a quadratic contact with the boundary. The two
intersections on a full line can coalesce, producing the singularity
of Gilbert–Kuo–Sloan. Each ray from the given interior point,
however, has a single threshold; the proof gives a quantitative
integration result without invoking a mixed-derivative space or
pretending that an exact conditional integral is one oracle call.
This answers part of appendix problem 1 and gives a budget allocation
for radial localization in problem 3. It is a variance-reduced
randomized cubature result, not a statement about the behavior
of standard QMC on the Cartesian fold.

## R2: a precision lower bound for nonadaptive rules

**Lean:** `Cubature.gaussian_radial_nonadaptive_lower`.

Use the R1 class with \(a=1\), \(b=2\), \(s=2\), and

$$
A_*(p)=1+72(6p+64p^2),\qquad
\lambda=(2\pi)^{-1/2}e^{-2},\qquad
w_*=\operatorname{area}(S^{d-1})(2\pi)^{-d/2}e^{-2},\qquad
c_p=w_*(2/3)^p\,\lambda(\lambda/3)^p>0 .
$$

For every integer \(m\ge1\) and \(N\) satisfying \(2N\le m^{p+2}\),
every randomized nonadaptive \(N\)-query algorithm has some radius
in that class with

$$
\mathbb E(\widehat I-I(R))^2
\ \ge\ \frac{m^p}{8}\left(\frac{c_p}{m^{p+2}}\right)^2.                 \tag{3}
$$

The lower bound allows all Cartesian query points, hence applies in
the capped model. The seed is drawn from an arbitrary probability
space; the queries are functions only of that seed, while the output
can also depend arbitrarily on all the answers. The Lean statement
uses integrability of the squared loss on the class. This entails
no loss in measuring the best achievable RMSE: clipping outputs
to \([0,1]\) preserves measurability and cannot increase error.
Choosing \(m=\lceil(2N)^{1/(p+2)}\rceil\) for \(N\ge1\) gives a
worst-case RMSE lower bound \(\Omega_d(N^{-1/2-1/(d+1)})\).
In the same class R1 gives \(O_d(N^{-1/2-2/(d-1)})\).

### Proof

**Bumps.** Put \(z_+=\max(0,z)\) and

$$
B(z)=\tfrac23\big[z_+^2-3(z-1)_+^2+3(z-2)_+^2-(z-3)_+^2\big].
$$

This is zero outside \((0,3)\). On the three subintervals of
\([0,3]\), its respective expressions are
\(\frac23 z^2,\ \frac23[1+2(z-1)(2-z)],\ \frac23(z-3)^2\).
Thus \(0\le B\le1\), \(B\ge2/3\) on \([1,2]\), and \(B'=0\) at
every zero. The derivative is bounded in absolute value by 2.
The positive-part square has the global remainder bound
\(|y_+^2-x_+^2-2x_+(y-x)|\le |y-x|^2\): if both \(x,y\ge0\) this
is equality; if both are nonpositive the left side vanishes.
For \(x\le0\le y\) it follows from \(y^2\le(y-x)^2\); for
\(y\le0\le x\) the remainder is \(x^2-2xy\in[0,(y-x)^2]\).
The four terms therefore give a bound \(6|y-x|^2\) on the affine
remainder of \(B\). Combining the derivative bound with that
remainder when \(|y-x|\le1\), and using \(0\le B\le1\) otherwise,
gives the Lipschitz bound \(8|y-x|\).

Use the direction coordinates \(t_j=u_{j+1}/u_1\) on \(u_1>0\).
For \(\iota\in\{0,\ldots,m-1\}^p\) let

$$
P_\iota(u)=\prod_{j=1}^p B(3m(t_j+1/2)-3\iota_j)
$$

there, and zero for \(u_1\le0\). The nonzero angular supports
are disjoint: each is inside the open slope box with intervals
\((-\frac12+\iota_j/m,-\frac12+(\iota_j+1)/m)\).
It is contained in the interior of the positive first-coordinate
face, where \(t_j=2x_j-1\); all other face restrictions vanish.
On that first face, the arguments of the scalar splines change
by at most \(6m\delta\) when \(\|x-y\|_\infty\le\delta\).

For completeness, the Taylor remainder of a product of \(p\)
factors in \([0,1]\), with respective changes bounded by \(Lh\)
and affine remainders by \(Hh^2\), is at most
\((pH+p^2L^2)h^2\). To see it, add one factor to a product of
\(k\) factors. If the old product is \(P\) at \(x\), \(Q\) at
\(y\), with linearization \(T\), and the new factor is \(a\)
at \(x\), \(b\) at \(y\), with linearization \(c\), the new
remainder is
\(a(Q-P-T)+(b-a-c)P+(b-a)(Q-P)\).
Telescoping also gives \(|Q-P|\le kLh\). The induction follows
from \(0\le a,P\le1\). The linearization of each patch is zero
when the patch value is zero, by the product rule and the
zero-derivative property of \(B\).

**The finite family.** Let \(J=m^2\), \(M=m^p\), and
\(r_j=1+j/m^2\) for \(j=0,\ldots,J-1\). For \(M\) bits
\(\eta_\iota\in\{0,1\}\) take

$$
R_{j,\eta}(u)=r_j+m^{-2}\sum_\iota\eta_\iota P_\iota(u).
$$

These radii lie in \([1,2]\). At either of the two points in an
affine remainder estimate at most one patch is nonzero; all
other patch remainders vanish there, including their
linearization. At most two bounds
\((6p+64p^2)(6m\delta)^2\) contribute, multiplied by \(m^{-2}\).
Hence (1) holds with \(A_*(p)\). The splines are \(C^{1,1}\)
and vanish with their derivatives at the support endpoints.
There is no singularity in extending by zero away from
\(u_1>0\), since on the supports \(u_1\ge(1+p/4)^{-1/2}\).
These are regular closed jump surfaces, not a construction
using discontinuous radii.

**Mass per bit.** By the radial identity and disjoint supports,

$$
I(R_{j,\eta})=F(r_j)+\sum_\iota\eta_\iota\Delta_{j,\iota},\qquad
\Delta_{j,\iota}=\int_S
[F(r_j+m^{-2}P_\iota(u))-F(r_j)]\,d\sigma(u).
$$

Restrict to the core slope box where all spline arguments are
in \([1,2]\). Each slope interval has length \(1/(3m)\), and
\(P_\iota\ge(2/3)^p\) on the core. The angular law of a standard
Gaussian is the law of \(Z/\|Z\|\). Restricting to \(Z_1\in[1,2]\)
costs a probability of at least \(\lambda\). Given \(Z_1=z\) there,
each slope interval corresponds to an interval of \(Z_{j+1}\) of
length at least \(1/(3m)\), inside \([-2,2]\), where its density
is at least \(\lambda\). Independence of the coordinates gives
core mass at least \(\lambda(\lambda/(3m))^p\). The Gaussian
radial density on \([1,2]\) is at least \(w_*\). It follows that
\(\Delta_{j,\iota}\ge c_p/m^{p+2}\).

**Information bound.** Put the uniform prior on the \(J\) levels
and independent fair bits. A query \(q\) can distinguish a bit
\((j,\iota)\) only when \(\|q\|\in[r_j,r_j+m^{-2})\) and
\(P_\iota(q/\|q\|)\ne0\). The windows and the angular supports are
disjoint. At \(q=0\) there is no such bit. Each fixed query
therefore marks at most one pair. Fix the seed, let \(S_j\) be
the unmarked bits for level \(j\), and denote the loss variable
in the bits by \(g_j(\eta)=\widehat I_j(\eta)-I(R_{j,\eta})\).
The output is invariant under flipping any bit in \(S_j\).
For \(\xi_\iota=2\eta_\iota-1\), the \(\xi_\iota\) have pairwise
zero cross moments and second moments one. For \(\iota\in S_j\),
flip invariance of the output and the additive mass formula
give \(\mathbb E_\eta[g_j\xi_\iota]=-\Delta_{j,\iota}/2\).
Expanding the nonnegative expectation of
\((g_j+\sum_{\iota\in S_j}\Delta_{j,\iota}\xi_\iota/2)^2\)
gives
\(\mathbb E_\eta g_j^2\ge\sum_{\iota\in S_j}\Delta_{j,\iota}^2/4\).
Since \(\sum_j|S_j|\ge JM-N\ge JM/2\), averaging over \(j\)
gives (3) for the prior average, uniformly in the seed.
Integrate over the seed and select one member of the finite
prior with risk at least the average. This establishes (3)
for arbitrary randomized nonadaptive rules.

## R3: matching adaptive lower bound

**Lean:** `Cubature.gaussian_radial_adaptive_lower`.

In the same class as R2, for all integers \(m\ge1\) and \(N\)
with \(2N\le m^p\), the lower bound (3) holds for **every**
randomized adaptive algorithm using at most \(N\) queries.
This implies worst-case RMSE
\(\Omega_d(N^{-1/2-2/(d-1)})\) by taking
\(m=\lceil(2N)^{1/p}\rceil\). In conjunction with R1 this gives
the exact adaptive minimax exponent on this class. As in R2,
the formal squared loss is integrable and the lower bound does
not impose the box cap. An algorithm with budget \(N\) can be
capped at depth \(N\) even on answer sequences not produced by
the class, without changing its output on valid inputs.

**Proof.** Fix the radial level \(j=0\) in the R2 construction.
Fix the seed of an adaptive rule and let \(S(\eta)\) be the
set of bits not marked along its realized path on \(\eta\).
There are at least \(M-N\ge M/2\) of these for every path.
If \(\iota\in S(\eta)\), flipping \(\iota\) leaves each answer
on that path unchanged, because none of its queries has the
corresponding site; induction on the path gives the same
output and the same set \(S\). Conversely a bit marked before
the flip cannot become unmarked after it, by involutivity.

Let \(g=\widehat I-I\) for the fixed seed, and set
\(V_\iota=(\Delta_{0,\iota}/2)\xi_\iota
\mathbf1_{\{\iota\in S(\eta)\}}\), with \(\xi_\iota=2\eta_\iota-1\).
Under the flip at \(\iota\), on the unmarked event
\(V_\iota\) changes sign and \(g\) increases by
\(\Delta_{0,\iota}\xi_\iota\). Off that event \(V_\iota=0\)
before and after the flip. Therefore
\(\mathbb E[gV_\iota]=-\mathbb E[V_\iota^2]\).
For \(\kappa\ne\iota\), \(V_\iota V_\kappa\) changes sign under
the flip at \(\iota\) (on the unmarked event the entire set
\(S\) is invariant); so its mean is zero.
Expanding \(\mathbb E[(g+\sum_\iota V_\iota)^2]\ge0\) gives

$$
\mathbb E[g^2]\ge \mathbb E\sum_\iota V_\iota^2
\ge (M-N)\,(c_p/m^{p+2})^2/4.
$$

Average over the seed and take a worst-case input as before.
This reasoning does not treat the adaptive set \(S(\eta)\) as
independent of the bits; its flip invariance is what is used.

## R4: matching nonadaptive cubature

**Lean:** `Cubature.gaussian_radial_nonadaptive_cubature`.

For the R1 class with \([a,b]=[1,2]\), mesh parameter \(m\ge1\),
sample count \(n\ge1\), and \(B=m^s\), there is an unbiased
nonadaptive algorithm using

$$
N_0 = [\,2d(m+1)^p+n\,]\,B
$$

point queries in the permitted box, with

$$
\mathbb E(\widehat I-I(R))^2
\le \frac{((A+2)W_2/m^s)^2}{n}.                                     \tag{4}
$$

Thus with \(n=m^p\) and \(m\) chosen to use a constant fraction of
the large budget \(N\), the RMSE is \(O(N^{-1/2-s/(2(p+s))})\).
In particular, R1–R4 together give these sharp exponents on the
same fixed class \(s=2,\ A=A_*(p),\ [a,b]=[1,2]\), at every fixed
\(d\ge2\):

| Query rule | Worst-case absolute RMSE | Results |
|---|---:|---|
| Adaptive randomized | \(\Theta_d(N^{-1/2-2/(d-1)})\) | R1, R3 |
| Nonadaptive randomized | \(\Theta_d(N^{-1/2-1/(d+1)})\) | R2, R4 |

There are no logarithmic factors in either assertion. R4's
variance exponent \(1+2/(d+1)\) is greater than
\(1+1/(2d-1)\) from He–Wang Theorem 3.5 on the overlapping
full-dimensional bounded smooth radial indicators. The different
algorithm and stricter geometry are essential; the conclusion
does not assert that scrambling alone attains this rate. For
comparison, ordinary independent Cartesian jitter on \(m^d\)
equal cubes also gives variance \(O(N^{-1-1/d})\) for indicators
with a finite Minkowski tube bound: at most \(O(m^{d-1})\)
cubes within the \(O(1/m)\) boundary tube can contribute to
variance, each with a squared weight \(m^{-2d}\).
Since \(2/(d+1)>1/d\) for \(d>1\), R4 also improves on that
indicator-specific baseline on its smooth radial subclass.

### Proof

For a threshold \(q\in[0,1]\) and phase \(v\in[0,1]\), let

$$
Q_B(q,v)=\frac1B\sum_{i=0}^{B-1}\mathbf1_{\{(i+v)/B<q\}}.
$$

Write \(g_q(t)=\mathbf1_{\{t<q\}}\). On the cell
\([i/B,(i+1)/B]\), both the test value and the cell average
lie between \(g_q((i+1)/B)\) and \(g_q(i/B)\). Their
difference times \(1/B\) in absolute value is at most the
difference of these endpoints times \(1/B\). Summation
telescopes, showing \(|Q_B(q,v)-q|\le1/B\) for every phase,
including the endpoints. If \(V\) is uniform, a change of
variables on each cell gives \(\mathbb E Q_B(q,V)=q\);
independence of the tests within a ray is not required.

Use all \(2d(m+1)^p\) tensor nodes of R1 with mesh \(1/m\).
At every node \(u\), test the radii \(1+i/B\), independently
of all answers. The readout \(1+Q_B(R(u)-1,0)\) is within
\(1/B\) of \(R(u)\) and lies in \([1,2]\). Use R1's
tensor-linear interpolation on these values to obtain \(P\).
By (1) and positivity of the weights, \(P\in[1,2]\) and
\(\|P-R\|_\infty\le (A+1)/m^s\).

Before looking at any answers, also choose \(n\) independent
pairs \((U_i,V_i)\) as in R1. Write
\(S=F(2)-F(1)>0\) and \(q(u)=(F(R(u))-F(1))/S\).
The \(B\) tests on a random ray are at the preplanned radii

$$
F^{-1}\!\left(F(1)+S(j+V_i)/B\right),\qquad 0\le j<B .
$$

This inverse exists and is unique on the band because \(F\) is
continuous and strictly increasing there. The tested levels do
not use \(q\) or \(P\). Their answers give
\(J_i=F(1)+S Q_B(q(U_i),V_i)\). We have
\(\mathbb E[J_i\mid U_i]=F(R(U_i))\) and
\(|J_i-F(R(U_i))|\le S/B\le W_2/B\).

Return \(\int F(P)\,d\sigma+n^{-1}\sum_i[J_i-F(P(U_i))]\).
Conditional on the fixed input, the pilot based on fixed points
is deterministic. The correction terms are iid, have the
correct mean, and lie in the interval
\([-(A+2)W_2/B,(A+2)W_2/B]\), since \(F\) is \(W_2\)-Lipschitz
on the band. Bounding their variance by the square of the
half-width proves (4). The inverse and the baseline integral
are computations with known functions and use no additional
oracle calls.

## R5: arbitrary-order adaptive radial cubature

**Lean:** `Cubature.gaussian_radial_polynomial_cubature`.

Retain the model and known band \([a,b]\) of R1. Fix an
integer \(r\ge1\). Replace (1) by the following polynomial
remainder condition with constant \(A\ge0\): for every face
\(f\) and point \(x\in[0,1]^p\), there exists a polynomial
\(P_{f,x}(y)\) in \(y-x\), of degree less than \(r\) in each
coordinate, such that for all \(y\in[0,1]^p\) and
\(\delta\ge\|y-x\|_\infty\),

$$
|R(u_f(y))-P_{f,x}(y)|\le A\delta^r.                                \tag{5}
$$

This is `ChartPolynomialRemainder`. The polynomials are not
given to the algorithm. In particular (5) includes \(C^r\)
chart functions with a uniform derivative bound of order \(r\):
expand along the straight segment from \(x\) to \(y\), using the
ordinary multivariate Taylor polynomial of total degree \(r-1\).

Set \(\Lambda=8^{rp}\), take \(E\ge\max((b-a)/2,\Lambda A)\),
and take an integer \(\ell\) with \(2^\ell\ge\Lambda+1\).
For every \(k\ge0,\ n\ge1\), there is an unbiased adaptive
algorithm making at most

$$
4d(2^k r)^p(r+\ell)+n
$$

queries within the cap and satisfying

$$
\mathbb E(\widehat I-I(R))^2\le
\frac{(2^\ell E\,2^{-rk} W_b)^2}{n}.                                 \tag{6}
$$

At fixed \(r\) its worst-case RMSE is
\(O_{d,r,a,b,A}(N^{-1/2-r/(d-1)})\) with no logarithmic factor
in the budget. The comparison to the Cartesian obstruction of
Gilbert–Kuo–Sloan is as in R1; R5 gives rates at arbitrary
integer order on the radial alternative, not increased
smoothness of their Cartesian preintegrand.

### Proof

**Interpolation and stability.** With \(m=2^j\) at level \(j\),
partition each chart axis into \(m\) intervals. In interval
\([c/m,(c+1)/m]\) use the nodes
\((c+i/r)/m,\ i=0,\ldots,r-1\). Use the interval selected by
\(\min(m-1,\lfloor mx\rfloor)\), including at an endpoint.
The one-coordinate Lagrange weights in local coordinates are
\(\lambda_i(t)=\prod_{h\ne i}(t-h)/(i-h)\), with
\(t=r(mx-c)\in[0,r]\).
They exactly reproduce every univariate polynomial of degree
less than \(r\): the interpolant agrees with the polynomial
at \(r\) distinct points. The same holds for any polynomial
of coordinate degrees less than \(r\) under the tensor product,
by factoring each monomial's interpolation over the axes.

For the stability estimate,
\(\prod_{h\ne i}|i-h|=i!(r-1-i)!\) and
\(\prod_{h\ne i}|t-h|\le r^{r-1}\). Summing the resulting
bound and using the binomial identity and the exponential
series gives

$$
\sum_i|\lambda_i(t)|
\le \frac{2^{r-1}r^{r-1}}{(r-1)!}
\le 2^{r-1}e^r
\le 8^r .
$$

Thus the tensor weights have absolute sum at most \(\Lambda\).
Every used node is within sup-norm \(1/m\) of \(x\). Exact
reproduction, (5) at \(x\) (which gives \(P_{f,x}(x)=R(u_f(x))\)),
and (5) at the nodes bound the exact-data interpolation error
by \(\Lambda A/m^r\). Averaging the estimates on the active
faces preserves this bound and the bound on the sum of the
absolute weights. The weights are measurable, since interval
selection, polynomial evaluation, and the finite face rule are
measurable.

**Pilot and correction.** Use precision \(\rho_j=E2^{-rj}\).
Initialize with the band midpoint. When the nodal radius error
is at most \(\rho_j\), the interpolant's total error is at most
\((1+\Lambda)\rho_j\le 2^\ell\rho_j\). Use this last radius
on either side of the predictor and clip to the known band.
Bisection \(r+\ell\) times at each new node brings the midpoint
error to at most \(\rho_{j+1}\), since the initial width is at
most \(2\cdot2^\ell\rho_j\). These subdivisions depend only on
the observed bits. There are \(2d(2^j r)^p\) nodes counting
all faces; the sum of the per-level counts is at most twice
the last count. The final bracket width is at most
\(2\cdot2^\ell\rho_k\), so R1's one-query-per-ray correction
has variance at most \((2^\ell\rho_k W_b)^2/n\).
Both clipping and the ray query map work as in R1. This proves
the stated finite budget and (6).

## R6: analytic radial jumps in polylogarithmic query complexity

**Lean:** `Cubature.gaussian_radial_analytic_cubature`.

Suppose the same known constants \(A\ge(b-a)/2\), \(L\ge1\)
give (5) with \(A L^j\) in place of \(A\) at *every* order
\(j\ge1\). Fix \(k\) with \(m=2^k\ge2\cdot64^p L\).
For every \(r\ge1\) there is an unbiased adaptive algorithm with
at most \(4d m^p(3p+2)r^d+1\) queries within the cap and

$$
\mathbb E(\widehat I-I(R))^2\le(2 A W_b\,2^{-r})^2 .                 \tag{7}
$$

At fixed \(d,a,b,A,L\), this gives
\(O(\exp(-c N^{1/d}))\) worst-case RMSE for some \(c>0\),
equivalently \(O((\log(1/\varepsilon))^d)\) queries for RMSE
\(\varepsilon\). This is a precision result, not tractability
uniform in \(d\): the displayed \(m^p\) can be very large.

A sufficient analytic hypothesis is a common holomorphic
extension of the chart functions to polydiscs of radius
\(1/L\) centered at the real chart points, bounded in modulus
by \(A\) there. For a unit sup-norm direction, evaluate on the
complex line in that direction and apply the one-variable
Cauchy estimate at each point of the real segment. The order
\(j\) directional derivative is at most \(j! A L^j\).
Taylor's formula on the segment then gives (5) with \(A L^j\).
Uniformly well-rounded centered ellipsoids furnish one example:
the chart radius is
\(\sqrt{v_f^\top v_f}/\sqrt{v_f^\top Qv_f}\). For the compact
set of permitted \(Q\) and chart points, both real quadratic
values are uniformly positive. In a sufficiently small common
complex neighborhood their real parts stay positive, and the
holomorphic square roots and the quotient have a common bound.
The theorem covers all radii satisfying the geometric remainder
condition, not only quadrics.

**Proof.** In R5 use \(n=1\), the order-\(r\) bound \(A L^r\),
\(\ell=3rp+1\), and \(E=\Lambda A L^r\). Since \(\Lambda\ge1\)
and \(L\ge1\), this choice satisfies the initial span assumption.
Also \(2^\ell=2\Lambda\ge\Lambda+1\), and

$$
2^\ell E/m^r=2 A(64^p L/m)^r\le 2 A\,2^{-r}.
$$

Finally \(r+\ell\le(3p+2)r\) for \(r\ge1\), yielding the stated
query bound and (7). At budget \(N\) take the largest order
allowed by this query bound. The square-root fold of the
Cartesian preintegrand is compatible with (7) because none of
the recovery steps use a full straight-line Cartesian fiber.

## R7: analytic adaptivity separation

**Lean:** `Cubature.gaussian_radial_analytic_nonadaptive_lower`.

On the R6 class with \(a=1,b=2\) (with any fixed \(A\ge0\) and
\(L\ge1\) in its remainder conditions), every nonadaptive
randomized rule with \(N\ge1\) has a worst-case MSE at least
\((F(2)-F(1))^2/(32N^2)\), with the same seed and integrability
conventions as R2. In particular, for \(A\ge1/2\) this is the
class on which R6 gives stretched-exponential convergence.
Thus the analytic-class precision separation is not merely a
separation between two upper bounds.

**Proof.** Put \(J=2N\) and \(S=F(2)-F(1)>0\).
For each level \(j=0,\ldots,J-1\), the two input radii are constant,
with Gaussian masses \(F(1)+jS/J\) and \(F(1)+(j+1)S/J\).
They exist by the strict increase and continuity of \(F\) on
\([1,2]\); constant radii satisfy (5) at every order with zero
remainder. Take a uniform level and one fair bit. The windows
between the two radii of each level, including the lower endpoint
and excluding the upper, are disjoint. Thus for a fixed seed
at least \(J-N\ge J/2\) levels give identical transcripts for
their two inputs. At such a level the average of the two squared
losses for a common output is at least \((S/J)^2/4\), as can be
seen by completing the square about the midpoint mass. Average
over levels and seeds to find a worst-case input with squared
risk at least \((S/J)^2/8\), as claimed.

## R8: unknown centers cause a search lower bound

**Lean:** `Cubature.gaussian_unlocated_ball_lower`.

Here the input is an indicator \(\mathbf1_{B(c,t)}\) of an open
Euclidean ball, with \(c\in[-1,1]^d\) unknown and
\(t\in\{1/16,1/2\}\) unknown. No interior point is supplied.
Write \(V_d=\operatorname{vol}(B(0,1))\). For every \(d\ge1\) and
\(N\ge0\) with \(2N V_d<2^d\), **any randomized adaptive
\(N\)-query rule** has an input on which its probability of
relative error strictly greater than \(1/4\) is at least \(1/4\).
The mass on this input is strictly positive. The seed law may
be arbitrary, with measurable outputs for the input balls.
Queries can even range over all of \(\mathbb R^d\).

For a guarantee of success probability \(>3/4\) uniformly over
the class, the bound is \(N\ge 2^{d-1}/V_d\). Since
\(V_d=\pi^{d/2}/\Gamma(1+d/2)\), the logarithm of this
threshold is \((d/2)\log d+O(d)\) by Stirling's formula.
This is a statement about rare masses and relative accuracy,
not an absolute-error obstruction when the mass is tiny.

This family uses a single regular analytic surface and constant
piece extensions, with a defining polynomial
\(\varphi(z)=t^2-\|z-c\|^2\). In the appendix it is contained in
\(\mathcal F_d(1,\infty,8+4d,1)\):
\(|\varphi|\le 2\|z\|^2+2d+1/4\), first partials are bounded by
\(2(\|z\|+\sqrt d)\), second partials by 2, and higher ones vanish.
The exponential envelope follows from \(s\le e^s\) and
\(s^2\le2e^s\) for \(s\ge0\). The boundary gradient has
norm \(2t\ge1/8\). These balls fit inside the query box.
Their masses need not be bounded away from zero with dimension,
so this result does not assert such a lower bound for all the
representative inputs described in the appendix.

### Proof

**Packing.** Greedily choose \(M=2N+1\) centers in the cube
at pairwise distances at least one. After \(j\le2N\) centers,
the union of the open unit balls at these centers has volume
at most \(j V_d<2^d\); hence it cannot cover the cube.
Pick the next center outside it. The large-radius balls at
the resulting centers are pairwise disjoint.

**Mass ratios.** Fix one center \(c\), put \(u=\|c\|\),
\(t_s=1/16,\ t_\ell=1/2\), and write \(g_d=(2\pi)^{-d/2}\).
On the small ball, \(\|z\|^2\ge u^2-2t_s u\), by squaring
\(\|z\|-u\) and using \(\|z\|\ge u-t_s\).
On the large ball, \(\|z\|\le u+t_\ell\).
Thus the small mass is at most
\(g_d e^{-\alpha}t_s^d V_d\) with
\(\alpha=(u^2-2t_su)/2\), and the large mass is at least
\(g_d e^{-\beta}t_\ell^d V_d\) with
\(\beta=(u+t_\ell)^2/2\).
We have \(u\le\sqrt d\le d\) and
\(\beta-\alpha=(t_\ell+t_s)u+t_\ell^2/2\le d\).
The ratio of large mass to small mass is consequently at least
\(8^d e^{-d}\ge2\), using \(e<3\) and \(d\ge1\).
Both masses are positive, as the Gaussian density is positive
on the balls. Their relative-\(1/4\) accuracy intervals
are therefore disjoint.

**Queries.** Fix the seed and follow the path of all zero
answers. Each query on that path lies in at most one large ball.
At least \(M-N\) centers are not touched, and for either radius
at such a center the actual run follows the all-zero path.
Its common output fails on at least one of the two radii by
the disjoint accuracy intervals. The uniform prior on the
\(2M\) inputs therefore has failure probability at least
\((M-N)/(2M)\ge1/4\) for every seed. Average over seeds and
select a single worst-case input from the finite prior.

## R9: unbounded worst-case gap at identical analytic budgets

**Lean:** `Cubature.gaussian_analytic_same_budget_gap`.

Fix \(p\ge1\) and the analytic class of R6 on \([1,2]\),
with \(A\ge1/2\), \(L\ge1\). Fix \(m=2^k\ge2\cdot64^p L\)
and let \(N_r=4d m^p(3p+2)r^d+1\), \(S=F(2)-F(1)>0\).
For every prescribed factor \(q>0\), there is \(N_0\ge1\)
such that for **every** \(N\ge N_0\) both statements hold at the
**same budget** \(N\) and on the **same class**:

- There is a measurable unbiased adaptive rule within the
  box cap with MSE strictly below \(S^2/(32qN^2)\) on every
  radius in the class.
- Every nonadaptive rule under the seed and integrability
  conventions of R2 has a radius in that class with MSE at
  least \(S^2/(32N^2)\).

**Proof.** The nonadaptive assertion is R7. R6 gives adaptive
MSE at most \(U^2\,4^{-r}\) with \(U=2AW_2\).
Write \(N_r=C r^d+1\), with the fixed \(C=4d m^p(3p+2)\).
The elementary limit \((C r^d+1)/2^r\to0\) follows, for
example, from the ratio test for \(r^d/2^r\).
For large \(N\), choose the largest integer \(r\) with \(N_r\le N\).
This exists since \(N_r\) is increasing and unbounded, and
\(r\to\infty\) as \(N\to\infty\). For \(r\ge1\),
\(N<N_{r+1}\le2^d N_r\). Thus \(32qU^2 N^2/4^r<S^2\) for
all sufficiently large \(N\). Divide by \(32qN^2\) and apply R6.
The Lean statement packages both quantifiers in a single
theorem; neither algorithm is given an exact value of \(R\).

## R10: relative Gaussian cubature for centered ellipsoids

**Lean:** `Cubature.gaussian_ellipsoid_relative_cubature`.

Let \(d\ge1\), and let \(Q\) be an unknown real symmetric
matrix whose quadratic form \(q_Q(z)=z^\top Qz\) satisfies

$$
L\|z\|^2\le q_Q(z)\le U\|z\|^2\quad\text{for all }z,
\qquad 0<L\le U,\qquad 1600 L\ge1.
$$

The spectral bounds \(L,U\) are known. The oracle is only
\(\mathbf1_{\{q_Q(z)<1\}}\), and \(I\) is the positive
Gaussian mass of this ellipsoid. Set

$$
\delta=\frac{2U-L}{2\cdot2^k},\qquad \eta=\frac{2d\delta}{L}.
$$

For any \(k\ge0\) for which \(d\eta\le1/2\), a deterministic
rule uses at most \(k\binom{d+1}{2}\) queries within the
coordinate cap and satisfies \(|\widehat I-I|\le2d\eta I\).
In particular, for \(0<\varepsilon\le1\) it suffices to take
\(2^k\ge 2d^2(2U-L)/(L\varepsilon)\). This is
\(O(d^2\log(d U/(L\varepsilon)))\) queries, with no factor
depending on \(I^{-1}\). It allows any hidden rotation. A
centered ellipsoid satisfying these bounds is an analytic
one-surface example in the appendix; unknown additive
smooth terms or missing centers are not covered by R10.

### Proof

**Queries for the form.** For each unordered coordinate pair
with repetition take \(v_{ii}=e_i\) and \(v_{ij}=e_i+e_j\)
for \(i\ne j\). The probe value \(q_Q(v_{ij})\) lies in
\([L,2U]\), because \(\|v_{ij}\|^2\) is one or two.
For a bisection level \(s\) in this interval, query
\(s^{-1/2}v_{ij}\); the complemented membership bit says
\(s\le q_Q(v_{ij})\). Every coordinate has magnitude at most
\(L^{-1/2}\le40\). After \(k\) subdivisions the bracket midpoint
\(\widehat s_{ij}=\widehat s_{ji}\) differs from \(q_Q(v_{ij})\)
by at most \(\delta\). All calls, including diagonal probes,
are counted in the stated budget.

Reconstruct \(H_{ii}=\widehat s_{ii}\) and for \(i\ne j\)

$$
H_{ij}=(\widehat s_{ij}-\widehat s_{ii}-\widehat s_{jj})/2 .
$$

The exact polarization formula has the same form without hats.
Hence every coefficient error is at most \(2\delta\). For every
\(z\), Cauchy–Schwarz gives

$$
|q_H(z)-q_Q(z)|\le2\delta\left(\sum_i|z_i|\right)^2
\le2d\delta\|z\|^2\le\eta q_Q(z).
$$

**Relative mass stability.** Write \(t=1-\eta>0\) and
\(K_G=\{z:q_G(z)<1\}\). Homogeneity and
\((1+\eta)t^2\le(1+\eta)t=1-\eta^2\le1\) give
\(tK_Q\subseteq K_H\). Conversely if \(q_H(z)<1\) then
\(t^2 q_Q(z)\le t q_Q(z)\le q_H(z)<1\), so
\(K_H\subseteq t^{-1}K_Q\).
For a Borel set \(K\) and a dilation \(u>0\), the Gaussian
change of variables gives

$$
\gamma_d(uK)=u^d\int_K (2\pi)^{-d/2}\exp(-u^2\|z\|^2/2)\,dz .
$$

Thus for \(0<u\le1\) its mass is at least \(u^d\gamma_d(K)\),
and for \(u\ge1\) it is at most \(u^d\gamma_d(K)\).
Return the mass of \(K_H\), a known integral from the
transcript requiring no additional memberships. Then
\(t^d I\le\widehat I\le t^{-d}I\). Bernoulli's inequality
gives \(t^d\ge1-d\eta\). For \(0\le d\eta\le1/2\),
\((1-d\eta)^{-1}\le1+2d\eta\), which gives the asserted
relative bound. The mass \(I\) is positive because \(K_Q\)
contains the open origin-centered ball of radius \(U^{-1/2}\).

## Offline verification

The numbered claims are listed in `formal/manifest.json`.
`formal/Theory/Statements.lean` is the statement layer, and
`formal/Theory/Proofs.lean` proves the same statements without
importing the statement layer. The other modules contain the
definitions and supporting lemmas, including the Gaussian
polar identity and the finite decision-tree cost model.
From the task container the check is:

```sh
python3 /workspace/tools/lean_check.py /workspace/submission/theory/formal
```
