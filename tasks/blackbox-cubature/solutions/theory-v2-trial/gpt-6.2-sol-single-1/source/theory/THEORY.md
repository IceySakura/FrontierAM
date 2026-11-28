# A sharp derivative threshold for Gaussian preintegration

**Selected target: E. Registered anchor: A-GKLS.** In Griewank, Kuo,
Leövey and Sloan, *High dimensional integration of kinks and jumps:
Smoothing by preintegration*, [arXiv:1712.00920v1](https://arxiv.org/abs/1712.00920v1),
[doi:10.1016/j.cam.2018.04.009](https://doi.org/10.1016/j.cam.2018.04.009),
Theorem 3 gives a **sufficient** order for preintegration to preserve
dominating mixed smoothness. R1 proves that the phase order in the
integrated coordinate is **necessary in general**, even with Gaussian
weights, constant jump height, and a root on every fiber with
\(\partial_j\phi\ge 1\). The sharpness is for the theorem's joint Sobolev
and classical conclusion; R1 disproves even its classical part below
the threshold for every retained order with sum at least two. R2
shows that at the first genuinely mixed pair \((1,1)\), even the
weak \(L^2\)-Sobolev part alone can fail. Neither is a competing
quadrature rate.

## Anchor and comparison

Use \(t=x_j\), \(y=x_{-j}\), and denote the amplitude by \(\theta\).
Theorem 2 of the cited version takes \(d\ge2\), \(R\ge1\),
\(1\le p<\infty\), a strictly positive probability density
\(\rho\in C^{R-1}(\mathbb R)\), and \(\theta,\phi\) in the weighted
isotropic \(W^R_{d,p,\zeta}\cap C^R\). The weights are
\(\zeta_{d,\alpha}(x)=\prod_k\zeta_{\alpha_k}(x_k)\) with continuous
integrable \(\rho=\zeta_0\le\zeta_1\le\cdots\). It assumes
\(\partial_t\phi>0\) everywhere and \(\phi(t,y)\to+\infty\) as
\(t\to+\infty\) for every \(y\), as in (18). Write \(U_j\) for the set of
retained points whose fiber has a root, and \(\psi(y)\) for that root.
If \(U_j\ne\varnothing\), it requires all factors of the form

$$
h(y)=
\frac{D^\eta\theta(\psi(y),y)\prod_{\ell=1}^a D^{\gamma^{(\ell)}}\phi(\psi(y),y)}
     {[\partial_t\phi(\psi(y),y)]^b}\,\rho^{(c)}(\psi(y))
\tag{B}
$$

in (19) to tend to zero at boundary points in \(U_j^c\) and to belong
to \(L^p(U_j,\zeta_{d-1,\alpha_{-j}})\) for the index range in (21).
The scalar ranges are \(1\le a,b\le2R-1\),
\(0\le|\eta|,c\le R-1\), and
\(1\le|\gamma^{(\ell)}|,\ |\gamma^{(\ell)}|+|\eta|+c\le R\);
the output weight indices have \(\alpha_j=0\), \(|\alpha|\le R\).
Theorem 2 concludes \(P_j(\theta\mathbf 1_{\{\phi>0\}})\in
W^R_{d-1,p,\zeta}\cap C^R\).

Theorem 3 replaces the isotropic input by
\(\theta,\phi\in W^{\boldsymbol r}_{d,p,\zeta,\mathrm{mix}}\cap
C^{\boldsymbol r}_{\mathrm{mix}}\), restricts (19)–(21) to the
multi-indices \(\gamma^{(\ell)}\le\boldsymbol r\),
\(\eta<\boldsymbol r\) and \(\alpha\le\boldsymbol r\) specified in the
theorem, and assumes \(r_j\ge \sum_{k\ne j}r_k\ge1\).
Its conclusion is \(P_j(\theta\mathbf 1_{\{\phi>0\}})\in
W^{\boldsymbol r_{-j}}_{d-1,p,\zeta,\mathrm{mix}}\cap
C^{\boldsymbol r_{-j}}_{\mathrm{mix}}\), with the Sobolev derivatives
understood weakly. The proof of Theorem 3 identifies
\(D_j^{|\alpha|}\phi\) and \(D_j^{|\alpha|-1}\theta\) in the
differentiation formula; the occurrence of these terms is not itself
a counterexample to a lower threshold.

R1 considers precisely the proposed lowering of the input order to
\((q,\boldsymbol r_{-j})\), using the **same** boundary-factor condition
with its index restriction at that proposed order. It verifies even
boundedness and continuity of every factor (B) involving positive-order phase
derivatives with at most \(q\) differentiations in \(t\), without upper
bounds on the number of factors, density derivative order, or retained
orders. If a replacement hypothesis separately demands a derivative
with \(q+1\) or more differentiations in \(t\), that is not the order-only
relaxation being tested. The standard normal (all \(\zeta_i=\rho\)) is
one of the weight choices allowed by the anchor.

**Classification: sharpness of A-GKLS, an opposite-direction result
(incomparable as standalone implications).** In the range
\(\sum_{k\ne j}r_k\ge2\), even replacing the phase order by
\(\sum_{k\ne j}r_k-1\), while allowing *arbitrarily* high retained
orders and an infinitely smooth amplitude, invalidates the stated
conclusion. Thus the two results locate the exact universal order
threshold in this range, under the corresponding trace conditions.
This is the optimality case of the comparison protocol; it is not a
claim that a necessary condition by itself implies the published
sufficiency theorem.

Three nearby results are relevant:

- Gilbert, Kuo and Sloan, *Analysis of preintegration followed by
  quasi-Monte Carlo integration for distribution functions and
  densities*, [arXiv:2112.10308v5](https://arxiv.org/abs/2112.10308v5),
  Theorem 3 with Assumptions 1 and 2: for indicators
  \(g_\tau=\mathbf 1_{\{\tau-\phi>0\}}\), \(\tau\in[a,b]\), it assumes
  \(D^0\phi>0\), the positive-infinity growth condition,
  \(\phi\in\mathcal H^{(\nu_0,\boldsymbol\nu)}\cap C^{(\nu_0,\boldsymbol\nu)}\)
  with \(\nu_0=|\boldsymbol\nu|+1\), and
  \(\rho_0\in C^{|\boldsymbol\nu|}\). The relevant instance of
  Assumption 2 requires the trace functions indexed by \(q=0\) and
  \(0\ne\eta\le\boldsymbol\nu\) to vanish at the edge of the root set
  and have uniformly bounded weighted \(L^2\) norms in \(\tau\).
  It concludes \(P_0g_\tau\in\mathcal H_d^{\boldsymbol\nu}\cap
  C^{\boldsymbol\nu}\) and gives a uniform \(\mathcal H_d^{\boldsymbol\nu}\)
  norm bound in terms of those trace bounds. For the comparison with
  R1 take \(\tau=0\), identify \(y_0=t\), \(\boldsymbol\nu=\boldsymbol r_{-j}\),
  and complement the indicator; the unique Gaussian-null root in each
  fiber makes the two preintegrals sum to one. The norm bound is
  additional information on its own class, but the assumed phase order
  is *higher* than the threshold tested here, and the theorem does not
  imply its necessity.
- Gilbert, Kuo and Sloan, *Preintegration is not smoothing when
  monotonicity fails*, [arXiv:2112.11621v1](https://arxiv.org/abs/2112.11621v1),
  Theorem 1 (A-GKS): for the indicator at level \(\tau=\phi(x^*)\),
  \(\phi\in C^2\), \(\partial_j\phi(x^*)=0\),
  \(\partial_j^2\phi(x^*)\ne0\), and \(\nabla\phi(x^*)\ne0\), it yields
  a square-root singularity along every retained line through \(x^*_{-j}\)
  not orthogonal to \(\nabla_{-j}\phi(x^*)\). Its critical-point
  hypothesis is excluded everywhere in R1. Neither obstruction is an
  instance of the other.
- He, *On the error rate of conditional quasi-Monte Carlo for
  discontinuous functions*, [arXiv:1708.09512v2](https://arxiv.org/abs/1708.09512v2)
  (SIAM J. Numer. Anal. 57 (2019), 854–874), Theorem 3.4:
  with isotropic \(C^R\) input amplitude and phase, \(\mathbb E|f|<\infty\),
  \(\rho\in C^{R-1}\), and a nonvanishing derivative in the integrated
  coordinate, it assumes locally uniform convergence of the fiber
  integrals of retained amplitude derivatives up to total order \(R\)
  near each point of \(U_j\), and vanishing of its trace factors at the
  edge of \(U_j\). It gives isotropic \(C^R\) output and a derivative
  bound by the integral of the absolute amplitude derivative plus a
  finite sum of absolute trace factors. With \(\rho\) normal and
  \(\theta=1\), the uniform convergence and edge requirements hold
  in our examples. But to give the *target* order \(q+1\), this
  theorem assumes isotropic \(C^{q+1}\) input, including the missing
  pure derivative. It is not a necessity result, and its classical
  conclusion is not a substitution for the Gaussian \(L^2\) result
  in R2.

The cited results are used for comparison only, not in the proofs
of R1 or R2.

## R1. The pure integrated-coordinate order cannot be lowered

**Lean:** `Cubature.gaussianMixedOrder_sharp`.

Let \(m\ge1\) be the number of retained coordinates,
\(r=(r_1,\ldots,r_m)\in\mathbb N_0^m\), and let \(q\) satisfy
\(1\le q<\sum_i r_i\). Put

$$
S(y)=\sum_{i=1}^m y_i,\qquad
U_n(t)=\frac{\max(0,t)^{n+1}}{(n+1)!},\qquad
A(t)=t+U_q(t),\qquad
\phi(t,y)=A(t)-S(y),\qquad \theta(t,y)=1 .
$$

Use \(f=\theta\mathbf 1_{\{\phi>0\}}\),
\(\rho(t)=(2\pi)^{-1/2}e^{-t^2/2}\) and let \(P(y)\) be the actual
preintegral \(\int_{\mathbb R} f(t,y)\rho(t)\,dt\). Then:

1. \(\phi\in C^q(\mathbb R^{m+1})\). For every retained order vector
   \(r'\), \(\phi\in C_{\rm mix}^{(q,r')}\cap
   W^{(q,r')}_{m+1,p,\rho,\rm mix}\) for every finite \(p\ge1\);
   all the certified derivatives have polynomial growth. The same
   input properties hold for \(\theta\) at every order.
2. \(\partial_t\phi\ge1\), and \(\phi(t,y)\to+\infty\) for each \(y\)
   as \(t\to+\infty\). Every fiber has a unique root, given by a
   continuous function \(\psi\) on all of \(\mathbb R^m\). Every factor
   (B) with \(b,c\ge0\), any finite list of positive-order phase derivatives
   having at most \(q\) differentiations in \(t\), and any derivative
   of \(\theta\) (including order zero), is continuous and bounded.
3. \(P\in C^q(\mathbb R^m)\), but **no** ordered sequence of \(q+1\)
   retained-coordinate differentiations yields an everywhere-defined,
   continuous classical derivative. In particular
   \(P\notin C_{\rm mix}^{r}\).

### Proof

**Input derivatives and weights.** For \(n\ge0\) the classical
derivative of \(U_{n+1}\) is \(U_n\). Off zero this follows by
polynomial differentiation, and at zero the slopes from both sides
are zero since the degree is at least two. Thus \(U_q\in C^q\) and
\(U_q^{(q)}=\max(0,\cdot)\). The two one-sided slopes of \(U_q^{(q)}\)
at zero are 0 and 1, so \(U_q\), and therefore \(A\), is not
\(C^{q+1}\).

The pure \(t\) derivatives of \(\phi\) of orders \(1,\ldots,q\)
are the corresponding derivatives of \(A\). The first \(y_i\)
derivatives are \(-1\); all derivatives of order at least two involving
a retained coordinate are zero, in every differentiation order.
This proves the classical claims and bounds every available derivative
by \(C(1+\|x\|_\infty)^n\) for some \(C,n\). The weak derivatives
are the same functions: for a smooth, compactly supported test
function, integration by parts in one coordinate transfers a single
derivative with a minus sign. Applying this along the given word,
and using the compact supports of the test derivatives, gives the
weak-derivative identity for the word. Fubini is applicable at each
step because the continuous differentiated factors are multiplied by
compactly supported functions. The Gaussian \(L^p\) claims follow
from \(\|x\|_\infty\le\sum_k |x_k|\) and finiteness of all finite
absolute moments of a one-dimensional normal. The latter follows by
integrating \(|t|^\beta e^{-t^2/2}\), which is locally integrable for
\(\beta\ge0\) and bounded by a constant times \(e^{-t^2/4}\) outside
a compact interval. The constant-amplitude claims follow in the
same way, with all its positive-order derivatives zero.

**Root and boundary factors.** \(A'=1+U_{q-1}\ge1\).
The function \(A\) is strictly increasing, equals \(t\) for \(t\le0\),
and satisfies \(A(t)\ge t\), so it is a homeomorphism of the real line
onto itself. Thus \(\psi(y)=A^{-1}(S(y))\) and \(U_j=\mathbb R^m\).
There is no root-set edge at which a limit must be taken.

For (B), the denominator is continuous and at least 1. Every
nonzero-order phase derivative that occurs in a numerator depends
only on \(t\) and has polynomial growth there. The amplitude
derivative is either 0 or 1. By repeated differentiation
\(\rho^{(c)}(t)=p_c(t)e^{-t^2/2}\) with a polynomial \(p_c\)
(one can start with \(p_0=(2\pi)^{-1/2}\) and use
\(p_{c+1}=p'_c-tp_c\)). Hence the numerator times
\(\rho^{(c)}(t)\) is continuous and bounded in absolute value by
\(C(1+|t|)^n e^{-t^2/2}\). An explicit uniform bound is
\(C n!e^{3/2}\): use
\((1+|t|)^n\le n!e^{1+|t|}\) and
\(1+|t|-t^2/2\le3/2\).
Evaluation at the continuous \(\psi(y)\) and division by the
denominator preserve continuity and the bound. This entails (21)
for the allowed Gaussian weights, since their total mass is finite.

**The output profile.** Let
\(T(a)=\int_a^\infty\rho(t)\,dt\) and \(H=T\circ A^{-1}\).
The positivity set in the \(t\)-fiber is
\((A^{-1}(S(y)),\infty)\); therefore \(P(y)=H(S(y))\).
The fundamental theorem of calculus gives \(T'=-\rho\) and
\(T\in C^\infty\). Since \(A\in C^q\), \(q\ge1\), and
\(A'\ge1\), its inverse is \(C^q\) by the one-dimensional inverse
function theorem. Thus \(P\in C^q\).

If \(H\) were \(C^{q+1}\), the identity \(H(A(t))=T(t)\) would give

$$
H'(A(t))\, A'(t)=-\rho(t),\qquad
A'(t)=\frac{-\rho(t)}{H'(A(t))}.
$$

The denominator never vanishes, and \(H'\circ A\) would be \(C^q\).
This would make \(A'\in C^q\) and \(A\in C^{q+1}\), a contradiction.

Finally, restriction to the \(i\)-th coordinate line through \(y\)
turns the sum \(S\) into \(S(y)+u\), where \(u\) is the displacement.
Consequently the first \(q\) derivatives in any coordinate order
are \(H^{(k)}\circ S\), \(1\le k\le q\). If the next derivative
existed everywhere and were continuous, its restrictions to the
coordinate lines would give the derivative of \(H^{(q)}\) at every
argument because \(S\) is onto. Evaluating the continuous derivative
function at \(y=u e_i\) would then make \(H^{(q+1)}\) continuous. This
contradicts \(H\notin C^{q+1}\). Since \(q+1\le\sum_i r_i\), a word
of length \(q+1\) with at most \(r_i\) copies of each \(i\) exists
(take a prefix of a word using exactly those multiplicities).
Its derivative is one of the required mixed derivatives. This
argument does not depend on the convention for ordering the
coordinates in a multi-index. \(\square\)

### Scope and formal correspondence

This is a regularity assertion for exact preintegration with a known
direction, as allowed in the question. No point-query algorithm,
budget, error bound, or withdrawal of the nondegeneracy condition is
claimed. The witness is in the finite-order class
\(\mathcal F_{m+1}(1,q,M,c)\) for a sufficiently large \(M\) and any
\(c>0\): the two pieces are 0 and 1, the phase is \(C^q\) with
polynomially bounded derivatives of all total orders at most \(q\),
and \(\|\nabla\phi\|_2\ge1\). The polynomial bounds imply the
required exponential bounds by taking the maximum over the finitely
many derivatives; the jump height is 1. The surface is a regular
graph, so its zero set has Lebesgue and Gaussian measure zero.
For example \(m=2,\ r=(1,1),\ q=1\) already fails at the genuine
mixed second derivative; for \(m\ge2\) and \(r=(1,\ldots,1)\), the
required phase order is \(m\), not one. R1 makes no claim about the
case \(\sum_i r_i=1\) or about lowering the amplitude order alone.
The finite-order witness does not assert a defect on the appendix's
hidden analytic instances, whether the rotated quadratic families
or the option families. The result concerns Target E, not the
appendix's open problem on nonmonotone preintegration. The failure
of the classical part does not assert failure of the weak Sobolev
part alone.

`Theory.Setup` uses `none` for \(t\) and `some i` for each retained
coordinate and defines the Gaussian integral with Mathlib's normal
measure. `HasPartialsAlong` requires the actual existence of the
successive classical derivatives. `GaussianMixedInput` checks
existence, continuity, weak test identities against smooth
compactly supported functions, polynomial growth, and `MemLp` for
all finite exponents, with the standard product Gaussian measure.
The statement checks the unique roots and bounded factors, then the
failure for every word of the first missing order and for the mixed
class itself. No human smoothing theorem is a premise of the Lean
result.

## R2. The mixed weak \(L^2\) condition itself can fail

**Lean:** `Cubature.gaussianMixedSecond_weakL2_sharp`.

Take \(d=3\), with \(t\) integrated and \(y=(y_1,y_2)\) retained.
Use \(S(y)=y_1+y_2\), and define

$$
B(u)=u+\frac23\,u_+^{3/2}-\frac23\,(u-1)_+^{3/2},
\qquad A=B^{-1},\qquad
\phi(t,y)=A(t)-S(y),\quad \theta(t,y)=1.
$$

The inverse exists globally. Use the same
\(f=\theta\mathbf 1_{\{\phi>0\}}\), Gaussian density and
preintegration \(P\) as in R1.
The input is \(C^1\) and has the same weighted mixed input and bounded
trace properties as in R1 with \(q=1\), for all retained orders. Now
\(1/2\le\partial_t\phi\le1\), the phase tends to \(+\infty\) in each
fiber, and the unique root is \(\psi(y)=B(S(y))\).
The output is \(C^1\), but for every locally integrable weak mixed
derivative representative \(G=D_1D_2P\), \(G^2\) is not integrable
on \([-1,1]^2\) for Lebesgue measure. In particular, no such \(G\)
is in Gaussian \(L^2\), and \(P\) does not belong to the Gaussian
weak mixed Sobolev space of retained order \((1,1)\).

### Proof

The derivative of \((2/3)u_+^{3/2}\) is \(\sqrt{u_+}\), including
at zero by the two one-sided difference quotients. Thus

$$
B'(u)=1+\sqrt{u_+}-\sqrt{(u-1)_+}\in[1,2].
$$

The lower bound follows from monotonicity of the square root.
For \(u\le1\) the upper bound follows from \(\sqrt{u_+}\le1\);
for \(u>1\) use \(\sqrt u\le1+\sqrt{u-1}\), by squaring.
Since \(B(u)=u\) for \(u\le0\) and \(B(u)\ge u\), \(B\) is onto
and strictly increasing. Its \(C^1\) inverse has
\(A'=1/(B'\circ A)\in[1/2,1]\) and \(|A(t)|\le|t|\):
for \(A(t)\le0\), \(t=B(A(t))=A(t)\); otherwise
\(t=B(A(t))\ge A(t)>0\).
The only nonzero available positive-order derivatives of \(\phi\)
are \(A'(t)\) and \(D_{y_i}\phi=-1\). Thus the phase and derivatives
have polynomial growth, are in all finite Gaussian \(L^p\) spaces,
and give the weak input derivatives by coordinate integration by
parts as in R1. The root formula is immediate from invertibility
and is valid for all \(y\). The available factors (B) are continuous
and bounded: \(|A'|\le1\), the reciprocal denominator is at most
\(2^b\), and every Gaussian density derivative is bounded.

The fiber formula gives \(P(y)=H(S(y))\) with \(H=T\circ B\) and
the same Gaussian tail \(T\) as in R1, so \(P\in C^1\). On
\(0<u<1\), \(B''(u)=1/(2\sqrt u)\). Differentiating
\(T'=-\rho\) and using \(\rho'(t)=-t\rho(t)\) gives

$$
H''(u)=\rho(B(u))
\left[B(u)\,B'(u)^2-\frac{1}{2\sqrt u}\right].
$$

If \(0<u<1/16\), then \(0\le B(u)\le2u<1\) and
\(B(u)B'(u)^2\le8u\le 1/(4\sqrt u)\). The density decreases
on \([0,1]\). Hence

$$
-H''(u)\ge \frac{\rho(1)}{4\sqrt u},
\qquad
|H''(u)|^2\ge \frac{(\rho(1)/4)^2}{u}.
$$

Let \(\Omega=\{0<S(y)<1/16,\ |y_2|<1/4\}\), an open subset of
\([-1,1]^2\). There \(G_0=H''\circ S\) is the classical mixed
derivative and is continuous. For every \(v\in C_c^\infty(\Omega)\)
two integrations by parts give
\(\int G_0 v=\int P D_2D_1v\); there is no contribution at the
edge of \(\Omega\) because the test support is compactly contained
in \(\Omega\). A weak representative \(G\) gives the same identity.
Uniqueness of locally integrable distributions gives \(G=G_0\)
almost everywhere on \(\Omega\): their difference annihilates
these tests, so its mollifications vanish on every compact
interior subset for small enough mollifier support, and the
mollifications converge locally in \(L^1\).

The shear \((u,v)\mapsto(u-v,v)\) has Jacobian determinant one and
takes \((0,1/16)\times(-1/4,1/4)\) onto \(\Omega\).
Consequently \(\int_{\Omega}1/S(y)\,dy=\infty\). The above lower
bound makes \(\int_\Omega G^2=\infty\), and thus the box integral is
infinite as well. On that box the product Gaussian density is at
least \(\rho(1)^2>0\); the Gaussian \(L^2\) condition would imply
finite Lebesgue square integral there. \(\square\)

### Comparison, scope and formal correspondence

R2 is the weak-\(L^2\) optimality obstruction for A-GKLS at
\(\boldsymbol r_{-j}=(1,1)\): the required order is 2, and the
example has integrated phase order 1 and satisfies every remaining
trace condition at that proposed order. It disproves a universal
weak-Sobolev-only relaxation at this point, even if the classical
conclusion is discarded. It is not a sufficient condition, and
does not follow from R1's lack of classical continuity. The
comparison is again opposite-direction sharpness, not domination
as a standalone regularity implication. The later distribution
function norm bound requires integrated order 3 at this retained
pair, and He’s isotropic regularity theorem needs order 2 to
conclude \(C^2\). A-GKS requires a nontransverse fiber,
which this example excludes.

The witness is in \(\mathcal F_3(1,1,M,c)\) for sufficiently large
\(M\) and any \(c>0\); the pieces are constant, the first phase
derivatives are bounded, and the phase has at most linear growth.
It is not a hidden analytic instance and no query guarantee is
claimed. `WeakMixedSecond` is exactly the test identity on
Lebesgue measure; the theorem rules out both square integrability
on the stated box and `MemLp` for the product Gaussian. In Lean,
transfer of derivatives uses line-direction integration by parts,
uniqueness uses the local smooth-test characterization of equality
almost everywhere, and the shear is measure-preserving for the
product Lebesgue measure.
