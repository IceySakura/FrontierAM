# Gaussian cubature from point values

## Executable algorithm

The executable does not receive a track or a family label. It starts with 512 standard-normal points, obtained from a scrambled Sobol' net using coordinatewise \(\Phi^{-1}\). It dispatches from the observed values: a nonnegative payoff with exact zeros goes to the option branch, and if its nonzero values agree to numerical precision to the digital sub-branch. Otherwise it searches for persistent jumps and then for continuous kinks. If structure cannot be validated it estimates a finite-difference gradient covariance at up to 20 pilot sites, caps large gradient norms, and spends the remaining budget on a Gaussian net rotated by its eigenvectors. The RNG is local to the process and derived from the instance seed, without using the seed to select a case.

`numerics.py` implements the nets and linear algebra. A request for \(n\) net points uses the first \(n\) points of a new SciPy linearly scrambled and digitally shifted Sobol' net of size \(2^{\lceil\log_2 n\rceil}\). Some branches partition a budget into independently scrambled power-of-two blocks plus a remainder. Every proposed point is checked for finiteness and constrained to the protocol box. The client charges every point, including every finite-difference stencil and repeated point. Model-only integration never calls the oracle. The program returns an estimate, not a confidence interval.

### Synthetic jumps

The initial line survey has 32 Gaussian directions normalized to length \(\sqrt d\), and a grid of 19 radii from 0 to 2.7. On each ray the two largest local differences are each refined seven times, keeping the subinterval with the larger contrast. Large sustained contrasts indicate a jump; unusually large responses trigger a search for a rare island, with contracted or expanded pilot probes when needed. For a rare island the anchor is zero if it is inside, and otherwise an average of detected interior points that is checked by a query. Ordinary jumps use polarity and local contrasts, so the two global ranges of values need not be disjoint.

The fitted surface is \((z-a)^\mathsf T A(z-a)+b^\mathsf T(z-a)=1\). `learn_shape` starts with a random orthogonal frame. Opposite axial roots \(r_+,r_-\) give \(A_{ii}=1/(r_+r_-)\) and \(b_i=1/r_+-1/r_-\). An off-diagonal coefficient is found by bisection directly on that coefficient: for each trial coefficient, the model predicts a chord endpoint, and the value at that endpoint chooses the next half-interval. Ordinary jumps shift the chord directions towards a measured crossing and use its nearby phase values. When a pivoted low-rank approximation accounts for the estimated diagonal it stops early. Otherwise the remaining cross terms get up to 19, 10, or 5 bits per term for \(d\le16\), \(d=32\), or \(d=64\), respectively, further limited by the remaining budget with 4,450 points reserved when allocating those terms. Coarse Gaussian-quantile coefficient grids have independent random offsets to avoid correlated rounding at zero. Near-zero or negative fitted eigenvalues are discarded. Validation rays can reject the fit.

For ordinary jumps the additive control is \(C(z)=B(z)+J(z)\mathbf1_{\widehat E}(z)\). The background \(B\) and height \(J\) are quadratic regressions in different learned rotations. The regression for \(J\) additionally uses one-sided differences saved from the chord searches. Up to 52 validated new crossings and up to 14 pairs of near-boundary gradients inform the rotations; up to 1,536 extra Gaussian points train the regressions. A fit that also uses the one-sided gradients is accepted if it lowers a held-out regression loss. For a large rare jump the height is a constant refined from up to 32 pairs of close one-sided values, with up to 1,024 extra Gaussian points for the background. In either case, the integral of the gated quadratic control is computed from the fitted coefficients: for fitted rank at most four the implementation uses a finite tensor quadrature with the last normal coordinate integrated by truncated moments; otherwise it uses exponentially tilted scalar Fourier inversion with Gaussian quadratic moments.

For a fitted surface of non-negligible uncertainty, there are two corrections. Let \(T\) be a radial band around the fitted boundary, \(H(z)\) a classification from the observed value, and \(D(z)=J(z)(H(z)-\mathbf1_{\widehat E}(z))\). One net estimates \(\int_T D\,d\gamma_d\) by drawing directions from a normalized elliptical Gaussian and radii uniformly in the band. The weights include the reciprocal angular density, the translation from the anchor, the chi radial density and the band length. The other net estimates \(\int(f-C-\mathbf1_T D)\,d\gamma_d\); disagreements outside the band remain in this latter correction. Thus, in the exact-sampling idealization, splitting the residual does not require correct classification. The ordinary-jump band uses 35% of the then remaining queries when active; the rare-jump band uses at most 4,096 points, up to one quarter of the remainder for a precise fit and otherwise one half. Finite Fourier truncations, clipping and approximate probability calculations are numerical approximations, not certified enclosures.

### Synthetic continuous kinks and smooth residuals

`kinks.py` augments the first line survey by 48 or 64 lines, partly offset from zero, and looks for second differences that persist when converted to slope jumps at finer scales. Two extrapolated one-sided secants update the candidate location. Resolved roots get finite differences on both sides; the count is capped at 76, 63, or 46 roots in the respective dimension ranges, and by the budget remaining after reserving 5,248 points. The solver clusters the normals by consistency of signed chord products. For two exact points \(p,q\) on the same quadratic surface with gradient field \(G\), \(G(p)\cdot(q-p)=G(q)\cdot(p-q)\). The normal scalings are fitted using these equations; a spectral gap can justify projecting the fit onto a smaller subspace. A single high-confidence quadric can get up to 2,304 further queries in unresolved directions. These queries look for new roots near model-predicted crossings to estimate the missing quadratic block. No condition \(f(p)=0\) is used.

The surrogate is a quadratic background plus fitted positive parts of the learned quadratics. The gradient jumps provide amplitude priors where their estimated magnitudes agree, and up to 1,536 new Gaussian points fit and calibrate the control. If \(Q\) is a quadratic statistic of a standard normal, its positive-part mean uses \(\mathbb E Q_+=(\mathbb E Q+\mathbb E|Q|)/2\), with \(\mathbb E|Q|=(2/\pi)\int_0^\infty (1-\operatorname{Re}\varphi_Q(t))\,dt/t^2\). When the held-out residual benefits from an overlap control, the solver adds the minimum of two fitted nonnegative hinges. To integrate it, it subtracts the expected maximum from the sum of the hinge means; the maximum uses an oracle-free outer net and exact truncated moments on the intervals between pairwise quadratic crossings in the inner coordinate. The final net is rotated using finite differences of the residual.

Several branches fit a validated residual control from Gaussian kernels. For a kernel \(K(z)=\exp[-\sum_i s_i(z_i-c_i)^2/2]\) in rotated coordinates its exact normal mean is
\(\prod_i(1+s_i)^{-1/2}\exp[-\sum_i s_ic_i^2/(2(1+s_i))]\).
When enough budget remains, two independently scrambled folds can fit and subtract kernel controls on each other's evaluation points, adding the known means. Symmetric pairs \(z,-z\) are used only if a 256-point pilot shows that the variance of their average is less than 0.29 of the unpaired variance. The pilot contributes a fixed small fraction to the residual average; the subsequent net is fresh.

### Options

For a digital payoff, a logistic separator fitted to the 512 pilot labels supplies the first axis. A training set of 192 perpendicular coordinates uses two bracketing values and seven refinements per line to fit the threshold as a quadratic of the outer coordinates. A control for the conditional exercise probability is integrated by scalar characteristic-function inversion: it is the probability that an independent standard normal exceeds this quadratic. The numerical integration decays rapidly because of that independent Gaussian. Bad training brackets can trigger the simpler rotated binary estimator in `options.py`.

The main digital rule uses six queries per perpendicular coordinate: five comparisons choose a bracket in the Gaussian probability coordinate \(u=\Phi(t)\), and one draw corrects it. If \([l,h]\) is that bracket, \(t_0=(l+h)/2\), and \(J\) is the known nonzero payoff, the per-line estimate divided by \(J\) is
\[
 (1-t_0) + \frac{\mathbf1_{\{f>0\}}-\mathbf1_{\{u>t_0\}}}
 {0.09+0.91\,\mathbf1_{[l,h]}(u)/(h-l)}.
\]
Here the draw is from the density in the denominator, so nonmonotone lines still receive a correction. A quadratic calibration control is also subtracted and its mean added.

For continuous option payoffs the initial gradient covariance uses up to 18 positive sites for \(d\le32\) and 14 for \(d=64\). With a low observed positive fraction, batches of 384 contracted normals search for surviving paths and fit a defensive Gaussian proposal. Validation selects among the following procedures using observed values only:

1. An exercise-branch model trained on up to 2,048 extra values, plus the available gradient jets. It fits a capped positive part of an exponential with an unknown scale. The log payoff is either quadratic in the leading coordinate with polynomial lateral coefficients, or spline in that coordinate and linear in the others. A one-dimensional multiplicative calibration can account for omitted barrier survival. The mean is computed by truncated Gaussian moments in the leading coordinate or by analytic averaging over the lateral Gaussian followed by a one-dimensional quadrature. The residual net can tilt the leading normal coordinate by an amount selected on pilot points, with the exact normal density ratio.
2. A direct line control for an intermediate range of observed survival rates. Five initial values and a sixth chosen in the largest estimated residual interval yield a piecewise quadratic interpolant with truncated Gaussian moments; the seventh is an importance draw for its residual. Acceptance is based on the dispersion of preliminary conditional estimates. The sampling density includes a positive Gaussian component on the entire numerical line interval.
3. A defensive importance mixture for barrier-shaped payoffs. A first Gaussian is fitted using survival-weighted pilot moments; in the intermediate-rate branch another pilot of up to 1,536 points refines its covariance, with eigenvalue floors and shrinkage. The final proposal contains a standard Gaussian component. A logistic survival control times a smooth payoff control is used only when its weighted held-out residual improves. The simplest fallback uses the rotated net, optionally with a quadratic/spline control whose hinge means use exact truncated-normal moments.

The files `solution.py`, `hard.py`, `kinks.py`, `options.py`, `exercise.py`, `payoff_spline.py`, `line_option.py`, and `barriers.py` specify the branch thresholds and the exact allocations when a pilot or validation is rejected. All model expectations here are numerical devices. In particular, the executable neither assumes that its classifier is exact nor invokes the certified R1 algorithm to claim a bound on its result.

## R1. Deterministic relative Gaussian mass certificate for a shifted ellipsoid

Let \(d\geq 1\), \(0<\rho\leq R\), and let \(a\in\mathbb R^d\) be a supplied interior point, not necessarily the center of the ellipsoid. Assume \(|a_i|+R\leq 40\) for all \(i\). The unknown integrand is \(f(z)=J\mathbf 1_E(z)\), where \(J>0\) is unknown and
\[
 E=\{a+x: Q(x)+b^\mathsf{T}x\leq 1\},
 \qquad Q(x)=x^\mathsf{T}A x,\quad A=A^\mathsf{T}\succ0.
\]
Assume \([-\rho,\rho]^d\subseteq E-a\subseteq [-R,R]^d\). The oracle is allowed either sign of the membership bit at a point of the boundary; for the integral it does not matter how \(f\) is defined there. There is no prior lower bound on \(\gamma_d(E)\), and neither the spectrum, the orientation, nor the center is supplied.

For \(m\) bisections per ray put
\[
 e_m=(R-\rho)2^{-m},\qquad
 \delta_m=e_m\left(3d^2R^2/\rho^3+2dR/\rho^2\right).
\]
For any supplied \(\delta\geq\delta_m\) with \(0\leq\delta<1/2\), use \(t=1-2\delta\) and
\[
 L_d(a,t)=t^d\exp\!\left(-\frac{1-t}{2(1+t)}\|a\|_2^2\right).
\]
With \(1+2d^2m\) point values in the permitted box, the algorithm described below constructs a set \(S\) from its transcript and returns the lower estimate \(J\gamma_d(S)\). It certifies
\[
 L_d(a,t) I(f)\ \leq\ J\gamma_d(S)\ \leq\ I(f).
\]
All postprocessing is in the information-based model; evaluation of the *known* set's Gaussian mass (to prescribed numerical precision if needed) is not an additional query to \(f\). The result does not assert a computational complexity bound for that postprocessing.

### Proof

Write \(u_{ii}=e_i\), \(u_{ij}=e_i+e_j\) if \(i\neq j\). The auxiliary directions are in the coordinates of the query interface; they do not assume axes of the ellipsoid. Only \(\pm u_{ij}\) are sampled. For every such line, write \(c=Q(u)>0\) and \(\ell=b^\mathsf{T}u\). The two boundary points are \(r_+ u\) and \(-r_-u\), where \(r_\pm>0\) and
\[
 p_\pm=1/r_\pm=\frac{\sqrt{\ell^2+4c}\pm\ell}{2}.
\]
Indeed \(1-ct^2-\ell t=(1-p_+t)(1+p_-t)\), with the two reciprocal parameters exchanged on the negative ray. The inner and outer boxes imply \(\rho\leq r_\pm\leq R\). For the outer bound, any positive parameter greater than \(R\) but still below a purported root would be strictly inside while violating the outer box in a coordinate of \(u\) of absolute value one. Starting from \([\rho,R]\), query the midpoint, keeping the upper half for a positive value of \(f\) and the lower half otherwise. The bracket retains the root regardless of the treatment of equality, has length \(e_m\), and its midpoint \(\widehat r\) is in \([\rho,R]\). Every sampled point \(a\pm s u\) has each coordinate bounded by \(|a_i|+R\).

Record both midpoint radii for each ordered pair once. The true coefficient \(Q(u_{ij})\) is \(p_+p_-\), and \(b_i=p_+-p_-\) on the \(i\)-th coordinate line. Use these expressions with \(\widehat p_\pm=1/\widehat r_\pm\), and use polarization to set \(\widehat A_{ij}=(\widehat Q(u_{ij})-\widehat Q(u_{ii})-\widehat Q(u_{jj}))/2\) for \(i\ne j\). Since \(|\widehat p-p|\leq e_m/\rho^2\) and both reciprocals are at most \(1/\rho\), the error in each ray product is at most \(2e_m/\rho^3\). Thus the errors in each \(A_{ij}\) and \(b_i\) are at most \(3e_m/\rho^3\) and \(2e_m/\rho^2\), respectively. On the outer box this gives \(|\widehat Q(x)+\widehat b^\mathsf{T}x-Q(x)-b^\mathsf{T}x|\leq\delta_m\).

Take \(S=\{a+x:\ |x_i|\leq R\ (\forall i),\ \widehat Q(x)+\widehat b^\mathsf{T}x\leq1-\delta\}\). It is measurable and contained in \(E\). In the other direction, if \(a+x\in E\), then \(Q(tx)+b^\mathsf{T}(tx)\leq t[Q(x)+b^\mathsf{T}x]\leq t\), since \(Q(x)\geq0\) and \(t^2\leq t\). Therefore \(a+t(E-a)\subseteq S\). The extra query at \(a\) gives \(J\).

For any \(y\in\mathbb R^d\), completing the square gives
\[
 \|t y+(1-t)a\|_2^2-\|y\|_2^2
 = \frac{1-t}{1+t}\|a\|_2^2-
       (1-t)(1+t)\left\|y-\frac{t}{1+t}a\right\|_2^2 .
\]
Thus dilation by \(t\) about \(a\) decreases the Gaussian density by at most a factor \(\exp[-(1-t)\|a\|_2^2/(2(1+t))]\). Its Lebesgue Jacobian is \(t^d\). Integrating this pointwise comparison over \(E\) gives \(\gamma_d(S)\geq \gamma_d(a+t(E-a))\geq L_d(a,t)\gamma_d(E)\). The pointwise density constant is sharp: equality holds at \(y=t a/(1+t)\) and the ratio approaches it on shrinking neighborhoods of that point.

For fixed geometry, taking \(\delta=\delta_m\) yields relative error at most \((2d+\|a\|_2^2)\delta_m\). Indeed \(1-t^d\leq d(1-t)\) and \(1-e^{-s}\leq s\) for \(s\geq0\); for two factors in \([0,1]\) the defect of their product is no greater than the sum of their defects. In particular the query count for relative error \(\varepsilon\) is \(O(d^2[1+\log_+( (R-\rho)(3d^2R^2/\rho^3+2dR/\rho^2)(2d+\|a\|_2^2)/\varepsilon)])\). The dependence on rarity is through geometry, not an added inverse probability. For example, for scaled ellipsoids about the same anchor with fixed aspect bounds, \(R/\rho\) is constant, the prefactor of \(2^{-m}\) in \(\delta_m\) is unchanged, and their probabilities tend to zero with \(R\).

This is a subclass of \(\mathcal F_d(1,\infty,M,c)\) for sufficiently large \(M\) and any \(c>0\). The phase functions are constants and the defining polynomial and its derivatives are bounded by \(M e^{c|z|}\) (derivatives above order two vanish). Positive definiteness places the unique stationary point strictly inside the sublevel set, so the gradient has positive minimum on the compact boundary. The boundary is null, for example by slicing in any coordinate and noting that each line hits it at at most two points.

### Comparison and scope

**Incomparable with the general RQMC and preintegration results.** He and Wang (2015), *SIAM Journal on Numerical Analysis* 53(5), 2488–2503, [doi:10.1137/15M1007963](https://doi.org/10.1137/15M1007963), Theorem 3.5, bound the variance of scrambled integration for \(g\mathbf1_\Omega\) on the unit cube when \(g\) is square-integrable with bounded Hardy–Krause variation and \(\partial\Omega\) admits finite \((d-1)\)-dimensional Minkowski content. Their bound is \(O(n^{-1-1/(2d-1)}(\log n)^{2d/(2d-1)})\); it covers much more general boundaries and amplitudes. For the intersection with the model here, the deterministic relative certificate has an exponential query rate rather than a polynomial guarantee at fixed geometry. It does not extend the scrambled-net variance theorem to larger classes. The formula for that variance bound is also reproduced, including its hypotheses, in He, [arXiv:1702.03361](https://arxiv.org/abs/1702.03361), Proposition 2.3.

Gilbert, Kuo and Sloan, *Preintegration is not smoothing when monotonicity fails*, [arXiv:2112.11621](https://arxiv.org/abs/2112.11621), Theorems 1 and 2, concern Cartesian one-variable preintegration. Their Theorem 1 assumes \(C^2\) defining data, a tangency with nonzero second derivative in the integrated variable and nonzero full gradient; the preintegrated indicator has square-root behavior along transverse lines at the tangency value. Theorem 2, under nonparallel-gradient hypotheses, shows that bad levels are not isolated. Their conclusion specifically identifies the need to locate both endpoints of finite intervals and the lack of mixed-smoothness justification for QMC there. R1 uses both endpoints, but reconstructs the polynomial once, without solving every Cartesian fiber or asserting it is smooth. This addresses the closed-quadratic case of appendix questions 1 and 3, under narrower geometric and value assumptions. In particular, R1 does not provide a certificate for the synthetic families with an unknown full-dimensional background or for curved option paths.

Lean: `Cubature.ellipsoid_mass_certificate`.

## R2. Exact dimension threshold for deterministic quadratic-value information

For \(d\ge1\), let \(D=d(d+1)/2\). Consider \(f_A=\mathbf1_{E_A}\) with
\[
  E_A=\{z:z^\mathsf{T}A z\leq 1/100\},\qquad
  A=A^\mathsf T,\quad \tfrac12 I_d\preceq A\preceq\tfrac32 I_d .
\]
The center and the spectral bounds are known. Every deterministic adaptive algorithm using fewer than \(D\) queries has relative error at least \(1/50\) on some \(A\) in this class. This holds even if the response to the query \(z\) is the exact real \(z^\mathsf T A z\). More generally, the response can be any fixed function of \(z\) and that real; membership is obtained by comparison with \(1/100\). The lower bound even allows queries outside the protocol box. With the exact quadratic-value oracle, \(D\) **nonadaptive** queries in the protocol box suffice for exact reconstruction and hence arbitrary accuracy without further queries. Thus \(D\) is the exact deterministic query complexity of obtaining relative error strictly below \(2\%\) in the quadratic-value model. The membership upper bound is R1 instead; one must not treat one bit as an exact quadratic value.

### Proof

Run the algorithm on \(A_0=I_d\), producing the sequence \(z_1,\ldots,z_N\) of queried points. The equations \(z_i^\mathsf T Bz_i=0\) are \(N\) homogeneous linear equations on the \(D\) entries of a symmetric matrix \(B\). Since \(N<D\), there exists a nonzero solution. Rescale \(B\) so that its maximum eigenvalue in absolute value is \(1/2\). Both \(A_\pm=I_d\pm B\) then have spectrum in \([1/2,3/2]\), and both give exactly the same quadratic values as \(A_0\) at all recorded points. By induction down the adaptive query path the response histories and output are identical for both candidates and \(A_0\). An algorithm stopping early can pad the path with ignored queries to the origin.

Writing \(\lambda_j\) for the eigenvalues of the rescaled \(B\), we have
\[
 \det(A_+)\det(A_-)=\prod_j (1-\lambda_j^2)\leq 3/4.
\]
All factors are positive and one is \(3/4\). Since \((7/8)^2>3/4\), one choice \(A_*\) satisfies \(\det(A_*)\leq 7/8\), and hence \(\sqrt{\det(A_*)}\leq15/16\). The map with matrix \(\operatorname{diag}(\sqrt{\lambda_j(A_*)}) U^\mathsf T\), where \(U\) is an orthogonal eigenbasis of \(A_*\), sends \(E_{A_*}\) onto the radius-\(1/10\) ball \(E_{A_0}\). Therefore
\[
\operatorname{vol}(E_{A_0})=\sqrt{\det(A_*)}\,\operatorname{vol}(E_{A_*}).
\]
On \(E_{A_*}\) the squared norm is at most \(2/100\), so its Gaussian density is at least \(C_d e^{-1/100}\geq (99/100)C_d\), where \(C_d=(2\pi)^{-d/2}\). The Gaussian density on \(E_{A_0}\) is at most \(C_d\). Since \((21/20)(15/16)<99/100\), the Gaussian masses satisfy \(I(f_{A_*})\geq (21/20)I(f_{A_0})\). Both masses are positive. No common estimate can have relative error smaller than \(1/50\) at both, since
\[
 (1-1/50)(21/20)>1+1/50.
\]

For the matching upper bound, query at \(e_i\) and at \(e_i+e_j\) for \(i<j\). The former give \(A_{ii}\) and the latter give \(A_{ii}+2A_{ij}+A_{jj}\), determining \(A\). Its Gaussian mass is now a known integral; computing it to arbitrary accuracy uses no further oracle calls.

These are regular analytic quadrics and constant branches, in \(\mathcal F_d(1,\infty,M,c)\) as in R1. R1 applies to this subclass with \(a=0\), \(R=1\), \(\rho=1/(10(d+1))\); those simple bounds suffice, without exploiting the exact extremal radii. Thus at a fixed relative tolerance the membership-query bounds agree in their quadratic dimension power, up to a logarithm. Neither bound is a lower bound for randomized algorithms.

### Comparison and scope

**Incomparable.** Bárány and Füredi, *Computing the Volume is Difficult*, *Discrete & Computational Geometry* 2 (1987), 319–326, [doi:10.1007/BF02187886](https://doi.org/10.1007/BF02187886), Theorem 1 and its proof in §2, consider deterministic polynomial-time Lebesgue-volume enclosures for centrally symmetric convex bodies with a well-guaranteed separation oracle. For a fixed polynomial budget in their basis-and-ball promise model, the enclosure ratio can be at least \((c d/\log d)^d\) for a budget-dependent \(c>0\). They run the algorithm on the unit ball, then replace it by a matching polytope and its polar. They explicitly comment on the apparent paradox of a ball transcript failing to determine its volume, and relate it to approximating ellipsoids by polytopes. Their construction does not keep the candidates ellipsoidal. R2 achieves that restriction and quantifies the precise dimension threshold for exact quadratic values, but only gives a fixed small relative gap and does not cover general convex bodies or a separation oracle. In particular, a separation oracle that returns normal vectors does not reduce to a function of the single queried quadratic value. This is a restricted answer to the deterministic side of appendix question 4.

Lean: `Cubature.centered_ellipsoid_query_lower` and `Cubature.exact_quadratic_reconstruction`.

## R3. Joint dimension and precision lower bound for membership queries

Let \(d\ge1\), \(m\ge2\), \(D=d(d+1)/2\), and set
\[
 \eta=\frac{1}{4dm},\qquad \theta=\frac{\eta}{8d^3},\qquad
 \varepsilon=\frac{\theta^2}{20}.
\]
Fix any supplied squared radius \(0<r^2\leq\theta^2/100\), and use the class of indicators of
\(E_A(r)=\{z:z^\mathsf{T}A z\leq r^2\}\) with the same uniform spectral bounds as R2. If \(2^N<m^D\), every deterministic adaptive \(N\)-query algorithm for this membership oracle has relative error at least \(\varepsilon\) on a member of the class. In particular the necessary bit-query count is at least \(D\log_2 m\). Unlike R2, this statement is only about binary information. Here \(m\) is the number of levels in a coefficient grid, not the number of bisections in R1.

### Proof

Construct \(m^D\) test matrices \(I+G\), where each independent entry of the symmetric \(G\) ranges over \(0,\eta,\ldots,(m-1)\eta\). For all \(x\),
\[
 |x^\mathsf{T}Gx|
 \leq \frac{1}{4d}\sum_{ij}|x_i x_j|
 \leq \frac14\|x\|^2.
\]
Thus these matrices are in the class. Any two distinct ones differ by at least \(\eta\) in one entry. There are at most \(2^N\) possible response strings, so two distinct matrices \(A,B\) have the same string. Each comparison at the queried point \(z\) is a linear inequality in the entries of \(A\). The midpoint \(C=(A+B)/2\) satisfies the same side of each inequality, including in the case of equality. Induction along the adaptive path therefore gives the same output on \(A,B,C\).

Write \(T=(A-B)/2\), choose \(P=\operatorname{diag}(\sqrt{\lambda_i(C)})U^\mathsf{T}\) from an orthogonal eigenbasis of \(C\), and set \(H=P^{-\mathsf{T}}TP^{-1}\). Then \(P^\mathsf{T}P=C\), each entry of \(P\) is bounded by \(2\), and \(I+H,I-H\) are congruent to \(A,B\), respectively. The eigenvalues \(\nu_i\) of \(H\) lie in \([-1,1]\). If \(L=\max_i|\nu_i|\), the spectral decomposition bounds each entry of \(H\) by \(dL\) (each orthogonal matrix entry is bounded by one). Hence every entry of \(T=P^\mathsf{T}HP\) is at most \(4d^3L\) in absolute value. The separated entry supplies \(L\geq\theta\).

Taking determinants gives
\[
 \det A\,\det B=(\det C)^2\prod_i(1-\nu_i^2)
     \leq (\det C)^2(1-\theta^2).
\]
At least one endpoint \(A_*\) has \(\det A_*\leq(1-\theta^2/2)\det C\), since \((1-\theta^2/2)^2\geq1-\theta^2\). Its volume comparison with the midpoint is
\[
 \operatorname{vol}E_C(r)
    \leq (1-\theta^2/4)\operatorname{vol}E_{A_*}(r),
\]
by the same linear change of variables as in R2 and \(\sqrt{1-\theta^2/2}\leq1-\theta^2/4\). The Gaussian density on either ellipsoid is between \(C_d(1-r^2)\) and \(C_d\) because \(\|z\|^2\leq 2r^2\) there. Since \(r^2\leq\theta^2/100\) and \(0<\theta\leq1\),
\[
 \gamma_d(E_C(r))\leq (1-\theta^2/5)\gamma_d(E_{A_*}(r)).
\]
Both masses are positive. The open relative error intervals of radius \(\varepsilon\) around the two masses are disjoint, since \((1+\theta^2/20)(1-\theta^2/5)<1-\theta^2/20\). Their identical output consequently fails on at least one.

For this class the R1 oracle upper bound applies with \(a=0\), \(\rho=r/[2(d+1)]\), \(R=2r\). To reach accuracy of order \(\varepsilon\) it needs \(O(d^2(\log(d+1)+\log m))\) queries, with all radii obtained from point values. Thus for \(m\geq d+1\) the two bounds match in order \(d^2\log m\). The lower bound is for relative error and the class radius is allowed to shrink as \(m\) increases; it is not an asymptotic theorem at fixed nonzero radius. The \(\mathcal F_d(1,\infty,M,c)\) embedding uses \(M\) chosen for the supplied \(r\), as in R1.

### Comparison and scope

**Incomparable.** The broad-class oracle lower bound of Bárány and Füredi (1987), Theorem 1, uses arbitrary well-guaranteed centrally symmetric convex bodies and even separation information, not just bits; its volume enclosure can be much worse than a fixed relative factor. It does not give a joint dimension-and-precision lower bound for Gaussian integration within a fixed finite-parametric shape class. Their proof and ensuing remark stress the significance of indistinguishable ball transcripts. Here a transcript cell contains a midpoint along a segment of *ellipsoids*, avoiding the polytope replacement and exposing a precision dependence. The comparison is not an improvement to their convex-body theorem. R3 gives a near-matching upper/lower answer to the deterministic, closed-quadratic part of appendix question 4 under the stated small-radius condition.

Lean: `Cubature.ellipsoid_membership_precision_lower`.

## R4. Joint dimension and precision lower bound at a fixed radius

Use the same \(d,m,D,\eta,\theta\) as in R3, but now fix **any** \(0<r\leq 1/10\), independently of \(m\). Use the indicators of \(E_A(r)\) with the narrower constraint \(\tfrac34 I_d\preceq A\preceq\tfrac54 I_d\). For every deterministic adaptive membership algorithm with \(2^N<m^D\), some such indicator has relative error at least \(\theta^2/40\). The radius and the spectral band may both be given to the algorithm. In particular, the \(d^2\log m\) lower bound holds for arbitrarily high precision on one fixed class, not only on shrinking ellipsoids.

### Proof

Use R3's coefficient grid, transcript collision, and midpoint \(C\). The grid estimate \(|x^\mathsf TGx|\leq\|x\|^2/4\) and convexity put all three matrices in the new spectral band. Use \(P,T,H,L\) as in R3 with \(V=P^{-1}\). The entry argument still gives \(L\geq\theta>0\). Moreover, \(|x^\mathsf TTx|\leq\|x\|^2/4\) and \(\|V y\|^2\leq(4/3)\|y\|^2\), hence \(L\leq1/3\), and the proof below only uses \(L\leq1/2\).

Let \(s(u)=1-u/2+u^2/4\). For \(|u|\leq1/2\), \(s(u)>0\) and
\[
 (1+u)s(u)^2
 =1-u^2(1/4-u/2)-u^4(3/16-u/16)\leq1.
\]
Set \(M_\pm=V s(\pm H)P\), where the polynomial \(s\) acts on the symmetric matrix \(H\). Diagonalizing \(H\) and applying this inequality shows \(M_+ E_C(r)\subseteq E_A(r)\) and \(M_- E_C(r)\subseteq E_B(r)\). The determinants \(k_\pm=\det M_\pm\) are positive, and
\[
 k_+k_-=\prod_i s(\nu_i)s(-\nu_i)
   =\prod_i(1+\nu_i^2/4+\nu_i^4/16)\ \geq\ 1+L^2/4.
\]
In particular \(\sqrt{k_+k_-}\geq 1+L^2/10\): square the right side and use \(L^2\leq1\).

The linear terms in the two transforms have opposite signs. Write \(M_\pm x=x+b/4\mp a/2\), where \(a=VHPx\) and \(b=VH^2Px\). The spectral bounds on \(C=P^\mathsf TP\) give
\(\|a\|\leq 2L\|x\|\) and \(\|b\|\leq 2L^2\|x\|\).
By the parallelogram identity and triangle inequality,
\[
 \frac{\|M_+x\|^2+\|M_-x\|^2}{2}
 \leq \left[(1+L^2/2)^2+L^2\right]\|x\|^2
 \leq (1+3L^2)\|x\|^2.
\]
On \(E_C(r)\), \(\|x\|^2\leq 2/100\). If \(\phi_d\) is the Gaussian density, the arithmetic-geometric mean inequality now implies
\[
 \begin{aligned}
 k_+\phi_d(M_+x)+k_-\phi_d(M_-x)
 &\geq 2\sqrt{k_+k_-}\sqrt{\phi_d(M_+x)\phi_d(M_-x)}\\
 &\geq 2(1+L^2/10)e^{-3L^2/100}\phi_d(x)\\
 &\geq 2(1+L^2/16)\phi_d(x).
 \end{aligned}
\]
In the last step use \(e^{-t}\geq1-t\), and \((1+v/10)(1-3v/100)\geq1+v/16\) for \(0\leq v\leq1\).
Integrate over \(E_C(r)\) and use the two determinant changes of variables and the inclusions. At least one of \(\gamma_d(E_A(r)),\gamma_d(E_B(r))\) is at least \((1+\theta^2/16)\gamma_d(E_C(r))\). All three share an output, and
\((1-\theta^2/40)(1+\theta^2/16)>1+\theta^2/40\).
Therefore the common output has error at least \(\theta^2/40\) relative to one of the positive masses.

The R1 upper bound again gives \(O(d^2(\log(d+1)+\log m))\) queries at this tolerance, so the dimension and precision powers match when \(m\geq d+1\). Equivalently, for each fixed positive \(r\leq1/10\) the bounded-spectrum class has deterministic membership complexity of order \(d^2\log(1/\varepsilon)\) in the high-precision regime where \(\log(1/\varepsilon)\) dominates \(\log(d+1)\). Here the argument may keep \(M,c\) in \(\mathcal F_d(1,\infty,M,c)\) fixed as the accuracy varies: \(d,r\) are fixed, the coefficients are uniformly bounded, and the lower bound on the boundary gradient follows from \(A\succeq(3/4)I\) and the fixed positive \(r\).

### Comparison and scope

**Incomparable with Bárány and Füredi (1987), Theorem 1.** As in R2 and R3, the shared transcript is the obstruction, but R4 derives the mass gap on a fixed ellipsoid class. It uses two explicit paired transports and a Gaussian density comparison, not replacement by polytopes, and the comparison is uniform in precision at fixed radius. The original theorem uses arbitrary centrally symmetric convex bodies, a separation oracle with geometric guarantees, Lebesgue volume, and worst-case polynomial-time enclosures. Neither conclusion subsumes the other. R4 gives matching deterministic bounds in its precision regime for the nonmonotone ellipsoid subclass of appendix question 4, and does not decide that question for general \(\mathcal F_d\) or for randomized algorithms.

Lean: `Cubature.fixed_radius_ellipsoid_precision_lower`.

## Connection to the experiments

R1, R3 and R4 have binary or two-level ellipsoid oracles; R2 also analyzes an enhanced oracle returning the defining quadratic. None of these assumptions is asserted for an entire held-out family. In particular, the synthetic families have an additional full-dimensional smooth term, the closed jump may live on a proper subspace, and the options do not expose quadratic membership. A full-dimensional isolated quadratic jump with a known anchor and a separable constant height is the closest geometry to R1. The experiment uses the geometry for a control and then estimates the remaining integral, rather than applying the theorem to an unknown background. The saddle and intersecting kinks use a different slope-based locator; the option routes use conditional sampling or importance sampling. Neither R4 nor the other lower bounds predict the numerical score on those distributions.

For a separate check of R1's scale, I tested the stated two-sided bisection construction on translated, randomly oriented ellipsoids with eigenvalues evenly spaced from 6 to 13 and \(\rho=0.078,\ R=0.62,\ J=1\). The orientation is the \(Q\) factor of a standard Gaussian matrix from NumPy's `default_rng(220+d)`; the next two Gaussian vectors from that stream give \(a\) and \(b\) after scaling by 0.16 and 0.12 respectively. In this check the fitted inner ellipsoid lies strictly in the outer box, so its mass can be integrated without the additional box truncation. The following values use numerical integration of the known ellipsoids; the enclosing inequalities are the analytic result proved above.

| Dimension | Approximate true mass | Bisections per ray | Point values | Observed relative error | R1 relative bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | \(9.370\cdot10^{-3}\) | 21 | 379 | \(8.482\cdot10^{-3}\) | \(3.453\cdot10^{-2}\) |
| 3 | \(9.370\cdot10^{-3}\) | 25 | 451 | \(5.308\cdot10^{-4}\) | \(2.182\cdot10^{-3}\) |
| 3 | \(9.370\cdot10^{-3}\) | 29 | 523 | \(3.317\cdot10^{-5}\) | \(1.365\cdot10^{-4}\) |
| 8 | \(3.019\cdot10^{-7}\) | 21 | 2,689 | \(1.512\cdot10^{-1}\) | \(4.952\cdot10^{-1}\) |
| 8 | \(3.019\cdot10^{-7}\) | 25 | 3,201 | \(9.997\cdot10^{-3}\) | \(4.024\cdot10^{-2}\) |
| 8 | \(3.019\cdot10^{-7}\) | 29 | 3,713 | \(6.270\cdot10^{-4}\) | \(2.558\cdot10^{-3}\) |

The errors in the smaller-width regime fall by roughly a factor of sixteen for each four added bisections, consistent with the \(2^{-m}\) rate even for the much smaller probability. The constants in the certified bound are deliberately conservative.

In the protocol run on all 12 supplied samples, with a read-only copy of the executable, the measured log-relative errors were:

| Family | Log-relative errors on the supplied cases |
| --- | --- |
| H1 | \(-5.05,-4.36\) |
| H2 | \(-4.16,-3.84\) |
| H3 | \(-5.13,-3.71\) |
| H3t | \(-4.59,-3.12\) |
| Vanilla call | \(-2.68\) |
| Digital | \(-4.06\) |
| Down-and-out call | \(-2.45\) |
| Double-knockout call | \(-2.04\) |

This read-only run, before the final coarse-grid adjustment, scored 70.4 on the sample scale. The final public snapshot scored 67.2 and passed all 20 cases. Neither number is a held-out prediction. The hard-track pattern is consistent with geometric information being particularly useful for the rare jumps, but does not test the deterministic error bound on the mixed smooth/jump functions. All launches returned a finite protocol result within the time and query limits; no empirical confidence interval was promoted to a certified bound.
