# Finite-network and parallel upper bounds for decentralized sum-of-nonconvex optimization

**Targets B and C (improved upper bounds and a matching regime).**
The component class is unchanged: the components are individually
\(L\)-smooth, their global mean is convex, and only the **composite**
objective is required to be \(\mu\)-strongly convex. Write
\(r=\min\{n,(1-\rho)^{-1/2}\}\), \(\kappa=L/\mu\).
The leading polynomial terms of the uniform upper bound R7 are
\(\widetilde O(m+[r+m^{3/4}n^{-1/4}]\sqrt\kappa)\).
We obtain the following order up to logarithms across all spectral
gaps at fixed \(n\):
\[
\widetilde\Theta\bigl(m+r\sqrt\kappa\bigr),
\]
when \(\kappa\ge128,\ 0<\varepsilon<\Delta/4\), and
\(\min\{\kappa^2/n,\ m^{3/4}n^{-1/4}\sqrt\kappa\}
\lesssim m+r\sqrt\kappa\), combining R5 and R7.
The exact logarithmic and batch factors are stated below. The
upper bounds use one communicated vector and a finite
reachability radius \(D=O(\min\{n,\log(2n)/\sqrt{1-\rho}\})\).
Their communication term is \(D\sqrt\kappa\) up to logs, with no
polynomial \(m\) factor. The method uses the problem's exact real
arithmetic; it does not address bit complexity.

## Comparison target (checked against the original)

Liu and Low, *Decentralized Sum-of-Nonconvex Optimization*, AAAI 2024
([arXiv:2402.02356v1](https://arxiv.org/abs/2402.02356v1),
[DOI:10.1609/aaai.v38i13.29318](https://doi.org/10.1609/aaai.v38i13.29318)),
Theorem 5.1 and Corollary 5.3, is the directly comparable accelerated composite result.
Their \(m_{\rm LL}\) counts agents and \(n_{\rm LL}\) counts components **per agent**:
\((m_{\rm LL},n_{\rm LL})=(n,m)\) here. Their \(F=f+\psi\) is our \(\phi\),
\(\sigma\) is the strong-convexity parameter, and their setting in Section 3
specifies nonnegative curvature parameters \(\sigma_f,\sigma_\psi\) with
\(\sigma=\sigma_f+\sigma_\psi>0\). Theorem 5.1 states the condition that \(F\)
is \(\sigma\)-strongly convex; \(f\) is convex and \(L_{\rm LL}\)-smooth,
and the components have first-order remainders bounded below by
\(-\ell_2\|x-y\|^2/2\) and above by \(\ell_1\|x-y\|^2/2\).
Their Definition 3.2 requires symmetric doubly stochastic \(W_{\rm LL}\),
\(0\preceq W_{\rm LL}\preceq I\), and
\(\ker(I-W_{\rm LL})=\operatorname{span}\{\mathbf1\}\).
With the prescribed parameters, Theorem 5.1 gives, at outer epoch \(K\),
\[
\mathbb E[F(\bar y^K)-F(x^\star)]
\le 3(1+\tau/2)^{-K}[F(\bar y^0)-F(x^\star)].
\]
It also gives mean-square bounds on \(\bar x^K-x^\star\) and
\(\bar q^K-x^\star\) with the same geometric factor, prefactors
\(128L_{\rm LL}/\sigma\) and \(1750L_{\rm LL}/(\sigma\tau^4)\), respectively,
and initial squared distance \(\|\bar x^0-x^\star\|^2\).
Corollary 5.3 selects batch size \(\sqrt{n_{\rm LL}}\) (integer rounding
understood) and reports the complexities
\[
\begin{split}
O\!\left(\left[n_{\rm LL}
 + \frac{\sqrt{L_{\rm LL}}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}\,
 n_{\rm LL}^{3/4}\right]\log\frac{\Delta_{\rm LL}}{\varepsilon}\right),
\\
\widetilde O\!\left(
\frac{\sqrt{n_{\rm LL}}
 +\frac{\sqrt{L_{\rm LL}}+(\ell_1\ell_2)^{1/4}}{\sqrt\sigma}\,
n_{\rm LL}^{1/4}}{\sqrt{1-\lambda_2(W_{\rm LL})}}\,
\log\frac{\Delta_{\rm LL}}{\varepsilon}\right),
\end{split}
\]
where \(\Delta_{\rm LL}=F(\bar y^0)-F(x^\star)\). As in the theorem,
the gap guarantee is in expectation. The remark preceding Corollary 5.3
counts one full **local** gradient and in expectation \(n_{\rm LL}\) sampled
component positions per epoch: the first rate is per node, and the total SFO
count is \(m_{\rm LL}\) times as large. Each gradient difference costs at
most two component calls (one if the snapshot gradient is cached).
Each batch must be serialized at a node here; a FastMix of length \(M\)
costs \(M\) rounds, and the inner prox costs one query. Serializing the
three primitive kinds costs at most the sum of their counts; conversely a
given schedule takes at least the maximum count. Substituting
\(\ell_1=\ell_2=L_{\rm LL}=L\), \(\sigma=\mu\) gives the per-node cost
\(O((m+m^{3/4}\sqrt\kappa)\log(\Delta/\varepsilon))\); the communication
term is
\(\widetilde O((\sqrt m+m^{1/4}\sqrt\kappa)
 (1-\lambda_2)^{-1/2}\log(\Delta/\varepsilon))\).
Using \((I+W)/2\) for a PSD mix at the cost of one primitive mix plus
local arithmetic costs at most a factor \(\sqrt 2\) in the worst-case
inverse square-root gap. Its second eigenvalue is at most \((1+\rho)/2\).
Already on \(W=J\) there is no PSD or topology conversion needed.

The paper's SFO calculation uses geometric epoch lengths. Its expected
work to finish the selected number of epochs is not automatically a
fixed-budget `hittingTime` bound. For counting a serialized implementation
of those epochs, the geometric lengths can be truncated with an explicit
failure budget: use the theorem at target \(\varepsilon/2\), cap each of the
\(K\) lengths at \(O((1+m/b)\log(2K\Delta/\varepsilon))\), and on any
truncation output \(x_0\). The union bound gives failure probability
at most \(\varepsilon/(2\Delta)\), and the successful paths contribute
at most \(\varepsilon/2\) by nonnegativity and the uncapped theorem.
This conversion costs logarithmic factors; it presupposes the original
algorithm's initialization, including its common initial tracker.
It does not improve the polynomial factors. Our iteration bounds below
are deterministic budgets from the start and do not require any
initially available gradients.

**Relation.** R1 and R2 are incomparable as global replacements for Liu and Low's
theorem. R1 pays \(m\) per accelerated step and can be worse on a
well-connected, data-rich problem. In R2 the improved rate uses
parallelism; at small \(n\) the per-node sample cost for the prescribed
variance can exceed theirs.
On overlapping parameter ranges there are strict improvements in the
iteration upper bound. For example, take an increasing integer \(s\),
\(m=s^2,\ \kappa=s^4,\ n\) a sufficiently large constant times \(s^6\)
(rounded up), and \(0\le\rho\le 1/(2n)\). The R2 bound has coefficient
of \(\log(\Delta/\varepsilon)\) of order \(s^2\), versus \(s^{7/2}\)
in the stated Corollary 5.3 SFO bound alone. Its communication coefficient
also improves on the paper's \(\sqrt m+m^{1/4}\sqrt\kappa\).
The batched version also improves the bound when \(m=\kappa=s^2\) and
\(n\) is only a sufficiently large constant times \(s^2\): with
\(B=\Theta(s)\), R2/R3 give coefficient \(O(s^2)\) instead of
the \(s^{5/2}\) term in Corollary 5.3. R3 gives a strictly stronger
guarantee **on its overlapping dense, parallel regime**, not a
replacement for their theorem outside it.
Section 8 of Liu and Low explicitly asks for improved communication
efficiency and singles out the \(\sqrt{n_{\rm LL}}\) dependence. The
improvement here comes from averaging independent, **locally centered**
gradient differences, not from treating a batch or local full gradient
as a primitive call. A separate improvement on bad-gap networks follows
from the finite \(n-1\) cap; it is a statement about this exact-real
model, not a numerical stability guarantee.

## Results and parameters

Use \(n\ge2,\ m\ge1,\ L\ge\mu>0,\ 0\le\rho<1\).
For the small-accuracy setting use \(0<\varepsilon<\Delta\).
Let \(\alpha=\sqrt{\mu/(8L)}\). Choose integers \(R,H\) satisfying
\[
\alpha R\ge 6,\qquad 2^{-H}\Delta\le\varepsilon.
\tag{1}
\]
For example \(R=\lceil 6\sqrt{8\kappa}\rceil\) and
\(H=\lceil\log_2(\Delta/\varepsilon)\rceil\).
The public integer \(D\) may be chosen by **any one** of the criteria
\[
\begin{gathered}
D\ge n-1;\qquad
2n((1+\rho)/2)^D<1;\qquad D=1\ \text{and}\ n\rho<1;\\
0<\rho\quad\text{and}\quad 4n<(1+\sqrt{1-\rho})^D.
\end{gathered}
\tag{2}
\]
There is no prescribed topology or requirement of nonnegative spectrum.
In particular one may use the smallest certified \(D\). Except in the
dense case, where one can use \(D=1\) whenever \(\rho<1/n\), one can use
\[
D=\min\!\left\{n-1,\ 1+\left\lceil
\frac{\log(4n)}{\log(1+\sqrt{1-\rho})}
\right\rceil\right\}.
\]
Using \(\log(1+s)\ge s/2\) for \(0\le s\le1\), this gives the
square-root spectral dependence stated above.

### R1. Deterministic bound

There exists a one-channel admissible algorithm which, uniformly over
all dimensions, all instances in the declared class, and all
\(W\in\mathcal W_{n,\rho}\), has
\[
T_\varepsilon\ \le\ H R (m+4D+1).
\tag{R1}
\]
The exhibited algorithm is deterministic (its finite seed space is a
singleton). Its corresponding minimax upper bound is
\(O((m+D+1)\sqrt\kappa\,\lceil\log_2(\Delta/\varepsilon)\rceil)\).
Lean: `ULB63.finite_network_exact`.

### R2. Parallel bound

Use an integer local batch size \(B\ge1\) satisfying \(nB\ge8\kappa R\).
There exists a one-channel admissible algorithm with
\[
T_\varepsilon\ \le\ H\,[m+4D+R(B+1+4D)]
\tag{R2}
\]
over the same full instance and communication classes. With
\(B=\max\{1,\lceil8\kappa R/n\rceil\}\) this is
\[
O\!\left(\left[m+(1+D)\sqrt{\kappa}+\frac{\kappa^2}{n}\right]
\left\lceil\log_2(\Delta/\varepsilon)\right\rceil\right).
\]
All \(B\) queries are charged separately; using \(B>m\) is allowed
with replacement, though R1 may then be preferable. If \(n\ge8\kappa R\)
one can take \(B=1\). R2 uses \(4DH(1+R)\) mixing rounds, without a
factor of \(B\) in that count.
Lean: `ULB63.finite_network_parallel`.

### R3. Fast-network minimax bracket

In addition to (1) and (2), suppose \(B\ge1\), \(nB\ge8\kappa R\)
and \(RB\le m+R\). For any \(q\in[0,1)\) and integer \(T\ge0\)
with \(4\mu\le L(1-q)^2\) and
\[
0\le\varepsilon<\min\{\Delta/4,\ \Delta q^{10T}\},
\]
there is a lower bound for every message width \(c\), and an upper
bound for width one:
\[
\begin{split}
 \max\{\lfloor m/10\rfloor+1,T+1\}
     &\le \operatorname{minimax}_{c},\\
 \operatorname{minimax}_{1}
     &\le H[2m+4D+R(2+4D)].
\end{split}\tag{R3}
\]
Here \(\operatorname{minimax}_c\) denotes the quantity in the problem
with width \(c\) and the other parameters fixed. In particular when
\(0<\varepsilon<\Delta/4,\ \kappa\ge16,\ n\rho<1\), and
\(nm\ge 8\kappa\lceil6\sqrt{8\kappa}\rceil^2\), the order for
constant width with at least one channel is between
\(\Omega(m+\sqrt{\kappa}\log(\Delta/\varepsilon))\) and
\(O((m+\sqrt{\kappa})\log(\Delta/\varepsilon))\).
Indeed choose \(R,H\) as in (1), and take \(B=\lceil8\kappa R/n\rceil\)
so \(RB\le8\kappa R^2/n+R\le m+R\). Take \(q=1-2/\sqrt\kappa\),
so \(-\log q\) is between \(2/\sqrt\kappa\) and \(4/\sqrt\kappa\)
by integrating \(1/(1-s)\) between \(0\) and \(2/\sqrt\kappa\).
Choose \(T=\lceil\log(\Delta/\varepsilon)/(10(-\log q))\rceil-1\)
to satisfy the strict inequality. The maximum of the two lower terms
is at least half their sum. More than one channel cannot invalidate
the lower bound, and additional channels may be left unused for the
upper bound.
If \(1-\rho\) is bounded below by a positive constant, the same
parallel-work condition yields matching bounds up to logarithms for
the entire communication class, not just for dense matrices: choose
\(D=O(\log(2n))\) in (2) and use the same lower bound.
Lean: `ULB63.fast_network_minimax_bracket`.

R3 does not claim matching bounds in the general sparse-network or
small-\(n\) regime. Its lower bound uses classical hiding and chain
ideas, with the composite prox and averaged-output first-hit metric
checked explicitly. The advance over the comparison target is the
parallel upper bound and its match to these lower terms in the same
parameter region.

### R4. Fixed-size slow-network minimax bracket

Suppose \(n\ge4\), \(1-1/(16n^2)\le\rho<1\), and use \(R,H\)
as in (1). Suppose \(B\ge1\), \(nB\ge8\kappa R\) and \(RB\le m+nR\).
For \(q\in[0,1)\) and integers \(k,T\ge0\) assume
\[
32\mu\le L(1-q)^2,\qquad 5T\le k\lfloor n/4\rfloor,\qquad
0\le\varepsilon<\min\{\Delta/4,\ \Delta q^{2k}\}.
\]
Then
\[
\begin{split}
 \max\{\lfloor m/10\rfloor+1,T+1\}
     &\le\operatorname{minimax}_c\qquad\text{for every }c,\\
 \operatorname{minimax}_1
     &\le H[2m+4n+R(5n+1)] .
\end{split}\tag{R4}
\]
Lean: `ULB63.slow_network_minimax_bracket`.

For \(R=\lceil6\sqrt{8\kappa}\rceil\), a sufficient parallel-work
condition is \(8\kappa R^2\le nm+n(n-1)R\), by taking
\(B=\lceil8\kappa R/n\rceil\). For \(\kappa\ge128\) and
\(0<\varepsilon<\Delta/4\), take \(q=1-\sqrt{32/\kappa}\),
\(k=\lceil\log(\Delta/\varepsilon)/(2(-\log q))\rceil-1\), and
\(T=\lfloor k\lfloor n/4\rfloor/5\rfloor\). Since \(q\ge1/2\),
\(-\log q\) is between \(1-q\) and \(2(1-q)\). The argument of
the ceiling defining \(k\) is greater than one, so \(k\) is at
least half of that argument. We obtain a lower bound
\(\Omega(m+n\sqrt\kappa\log(\Delta/\varepsilon))\) and upper bound
\(O((m+n\sqrt\kappa)\log(\Delta/\varepsilon))\). In this regime the
worst-case dependence on \(\rho\) has saturated at the network size;
the bound is not asserting rapid communication on every graph at
smaller \(\rho\).

R4 is **incomparable** to the Scaman et al. theorems discussed below:
the matching finite-size bracket is for first hitting of the gap of
the **average**, an incremental private-component and shared-prox
oracle, and an adversary with \(n\) fixed explicitly. Their work
establishes an optimal eigengap dependence in a different
quantification and includes a centralized per-edge messaging regime.
The same-model R4 upper bound is better than the inverse-gap term
reported by Liu and Low when \(1/\sqrt{1-\rho}\) is arbitrarily
larger than \(n\) in the overlap. This is a saturation result, not
a claim to improve the \((1-\rho)^{-1/2}\) exponent before saturation.

### R5. Matching parallel regime across the spectral range

Let \(p\le n\) be a nonnegative integer such that
\[
p<8\quad\text{or}\quad 1-\frac{1}{32p^2}\le\rho .
\]
Use \(D,R,H,B\) as in (1), (2), and R2. Let \(q\in[0,1)\)
and \(k,T\) be nonnegative integers, with
\[
32\mu\le L(1-q)^2,\qquad
5T\le k\max\{1,\lfloor p/16\rfloor\},\qquad
0\le\varepsilon<\min\{\Delta/4,\Delta q^{2k}\}.
\]
Then
\[
\begin{split}
\max\{\lfloor m/10\rfloor+1,T+1\}
  &\le\operatorname{minimax}_{c}\qquad\text{for every }c,\\
\operatorname{minimax}_1
  &\le H[m+4D+R(B+1+4D)].
\end{split}\tag{R5}
\]
Lean: `ULB63.parallel_minimax_bracket`.

For \(\kappa\ge128,\ 0<\varepsilon<\Delta/4\), take \(q,k\) as
in R4 and choose
\[
 p=\min\!\left\{n,\left\lfloor\frac1{\sqrt{32(1-\rho)}}\right\rfloor\right\},
 \qquad
 T=\left\lfloor k\max\{1,\lfloor p/16\rfloor\}/5\right\rfloor .
\]
The delay factor \(\max\{1,\lfloor p/16\rfloor\}\) is between
absolute constant multiples of \(r\), including when \(p<8\)
(in which case \(r\) is bounded). The lower bound is
\(\Omega(m+r\sqrt\kappa\log(\Delta/\varepsilon))\) in the
standard model. Choose \(D\) by (2) and \(R,H,B\) as in R2
for the upper bound
\[
O\!\left(\left[m+\frac{\kappa^2}{n}
       +r\sqrt\kappa\,\log(2n)\right]
       \left\lceil\log_2(\Delta/\varepsilon)\right\rceil\right).
\]
Thus the two bounds match up to logarithms under the parallelism
condition in the introduction, without the dense or saturation
assumptions. Without that condition, the upper bound still holds
but need not match. The classification relative to Liu and Low
is **strictly stronger in the reported matching regime** at the
level of polynomial factors: the bounds retain \(m\) for full
initialization and remove the extra accelerated \(m\)-power and the
unsaturated inverse-gap penalty where the improvements apply.
This is not a globally better bound in the unresolved small-\(n\),
large-\(\kappa\) regime. Relative to classical network lower bounds
the fixed \(n\), averaged-output, shared-prox construction supplies
the required quantifiers and metric. No lower bound separating
individual smoothness from convex components is claimed.

### R6. Accelerated regularized upper bound

Take \(\mu\le\nu\) with \(\mu+\nu\le L\), and \(D\) as
in (2). Let \(R,H,B,S,K\) be nonnegative integers. Put
\(a_{\rm in}=\sqrt{(\mu+\nu)/(8L)}\) and
\(a_{\rm out}=\sqrt{\mu/(2\nu)}\). Require
\[
\begin{gathered}
B\ge1,\qquad a_{\rm in}R\ge6,\qquad nB\ge 8\frac L{\mu+\nu}R,\\
a_{\rm out}S\ge6,\qquad 3(1+\nu/\mu)2^{-H}S\le1/4,\qquad
2^{-K}\Delta\le\varepsilon .
\end{gathered}
\]
Then a one-channel admissible algorithm uniformly has
\[
T_\varepsilon\le K S H\,[m+4D+R(B+1+4D)] .
\tag{R6}
\]
Lean: `ULB63.accelerated_parallel_bound`.

For \(\kappa\ge2\sqrt{nm}\), choose \(\mu+\nu=L/\sqrt{nm}\)
and the smallest budgets (rounding up). Then \(R=O((nm)^{1/4})\),
\(B=O(1+m^{3/4}n^{-1/4})\),
\(S=O(\sqrt{\kappa}/(nm)^{1/4})\), and \(H=O(1+\log\kappa)\).
In particular \(S R=O(\sqrt\kappa)\);
\(S m=O(m^{3/4}n^{-1/4}\sqrt\kappa)\); and
\(S R B=O(\sqrt\kappa[1+m^{3/4}n^{-1/4}])\).
There are \(KSH(m+RB)\) gradient calls **per node** and
\(4DKSH(1+R)\) mixing rounds. The prox count per node is \(KSHR\).
If \(\kappa<2\sqrt{nm}\), use R2, since \(\kappa^2/n<4m\).
The resulting envelope is
\[
O\!\left(
  \bigl[m+\sqrt\kappa(1+D+m^{3/4}n^{-1/4})\bigr]
  (1+\log\kappa)\left\lceil\log_2(\Delta/\varepsilon)\right\rceil
\right)
\]
for \(0<\varepsilon<\Delta\), with the \(\log\kappa\) overhead
unnecessary on the R2 branch. In particular the lower bound of R5
matches up to logarithms also when
\(m^{3/4}n^{-1/4}\sqrt\kappa\lesssim m+r\sqrt\kappa\), without
the original \(\kappa^2/n\) condition.

R6 is **strictly stronger in the indicated parameter ranges**
than the published computation/communication bounds of Liu and
Low. E.g., with \(m=n=s^4,\ \kappa=s^{12}\), and \(W=J\),
the leading round coefficient is \(O(s^8(1+\log s))\) compared to
the published \(s^9\) per-node SFO coefficient; in the overlap
with small spectral gap the communication factor \(m^{1/4}\)
is also removed up to logarithms, and the network term saturates
at fixed size. This compares upper guarantees, not lower bounds
on the running time of every implementation of PMGT-KatyushaX.

The proximal regularization device itself has a precedent in
Lin, Mairal and Harchaoui, *A Universal Catalyst for First-Order
Optimization* ([arXiv:1506.02186](https://arxiv.org/abs/1506.02186)),
Theorem 3.1 and Proposition 3.2. In the notation of that paper
\(\kappa_{\rm C}\) is a **regularization strength** and
\(q_{\rm C}=\mu/(\mu+\kappa_{\rm C})\).
For their centralized Catalyst schedule with
\(\alpha_0=\sqrt{q_{\rm C}}\) and subproblem tolerances
\((2/9)(F(x_0)-F^\star)(1-\rho_{\rm C})^k\),
\(0<\rho_{\rm C}<\sqrt{q_{\rm C}}\), Theorem 3.1 bounds the
outer gap by
\(8(1-\rho_{\rm C})^{k+1}(F(x_0)-F^\star)/
(\sqrt{q_{\rm C}}-\rho_{\rm C})^2\).
Proposition 3.2 shows that with their warm start and an inner
linear rate \(A(1-\tau_{\rm in})^t\) on the subproblem gap
relative to its initial value, each inner target takes
\(\widetilde O(1/\tau_{\rm in})\) work. The abstract acceleration
does not give the decentralized solver, per-node batch tradeoff,
or fixed weighted-sum implementation here. We prove a relative
expected-gap version for our restarted outer schedule directly;
we do not claim an improvement to Catalyst as an abstract
acceleration method.

### R7. Tuned uniform upper bound and fixed-size lower bound

Use the class parameters \(n\ge2,\ m\ge1,\ L\ge\mu>0,\ \Delta>0\),
\(0\le\rho<1\). Take \(D\) from (2), and integers \(H\ge1,\ K\ge0\)
such that
\[
120(1+\kappa)\sqrt{\kappa}\,2^{-H}\le 1,\qquad
2^{-K}\Delta\le\varepsilon .
\]
There is an admissible one-channel algorithm with
\[
T_\varepsilon
  \le KH\left\lceil 10^5\left[
       m+\sqrt\kappa\left(1+D+\frac{m}{(nm)^{1/4}}\right)
                         \right]\right\rceil
\tag{R7}
\]
for every instance and \(W\) in the fixed model. The same theorem
includes the R5 lower bound
\(\max\{\lfloor m/10\rfloor+1,T+1\}\le\operatorname{minimax}_c\)
for every \(c\) whenever the **lower-bound** conditions on \(p,k,T,q\)
and \(\varepsilon\) of R5 hold; it does not assume the R5/R2 inner
budgets for that lower bound.
Lean: `ULB63.optimized_parallel_minimax`.

For instance, take \(H=\lceil\log_2(120(1+\kappa)\sqrt\kappa)\rceil\)
and \(K=\lceil\log_2(\Delta/\varepsilon)\rceil\) for
\(0<\varepsilon<\Delta\). The upper theorem has no regime
restriction on \(\kappa,n,m,\rho\). The lower side gives
\(\Omega(m+r\sqrt\kappa\log(\Delta/\varepsilon))\) for
\(\kappa\ge128,\ 0<\varepsilon<\Delta/4\), as in R5.
When \(m^{3/4}n^{-1/4}\sqrt\kappa\lesssim m+r\sqrt\kappa\)
the bounds match up to logarithms. This includes, for example,
arbitrarily large \(\kappa\) at \(n\ge m^3\), regardless of \(\rho\).
One can also still use R2 where its \(\kappa^2/n\) term is smaller.

Compared to Liu and Low, R7 gives a stronger bound at the level
of polynomial factors, with strict improvements in the examples
above and without requiring PSD \(W\) or adding oracles.
The tradeoff is additional logarithmic factors and exact arithmetic.
In terms of their bounds, the accelerated per-node gradient term
gains \(n^{-1/4}\), and the communication term replaces
\((\sqrt m+m^{1/4}\sqrt\kappa)/\sqrt{1-\lambda_2}\) by
\(\sqrt\kappa\min\{n,\log(2n)/\sqrt{1-\rho}\}\), up to logarithms.
This does not assert a pointwise improvement of every constant or
logarithm. The dimension-free rate still need not match the
lower bound when the accelerated finite-sum term dominates.

For the unrestricted finite-sum strongly convex lower-bound question,
Woodworth and Srebro ([arXiv:1605.08003](https://arxiv.org/abs/1605.08003)),
Theorem 8, uses an oracle returning the value, gradient and prox of a
chosen component. Their strongly convex smooth components are on a
domain \(\mathcal X\subseteq\mathbb R^d\); their parameters satisfy
\(\gamma/\lambda>161N\), \(\varepsilon_0>60\varepsilon
\sqrt{\gamma/(\lambda N)}\), and their threshold is
\(\mathbb E[F(\hat x)-F^\star]<\varepsilon\). Their
\(\Omega(N+\sqrt{N\gamma/\lambda}
\log((\varepsilon_0/\varepsilon)\sqrt{N\lambda/\gamma}))\) bound is on
**total** sequential queries (up to \(n\) such queries can be executed
in a problem iteration at \(N=nm\)), not a lower bound on the
parallel \(m^{3/4}\sqrt\kappa\) factor. Zhou and Gu
([arXiv:1901.11224](https://arxiv.org/abs/1901.11224)), Theorem 4.1,
give the centralized linear-span IFO lower bound
\(\Omega(N+N^{3/4}\sqrt{L/\sigma}
\log(N\sigma\Delta/(L\varepsilon)))\) under averaged squared-gradient
smoothness with \(F\) in \(\mathcal S^{(\sigma,L)}\) and
\(\varepsilon\le8\Delta N^{7/4}\sigma^{3/2}L^{-3/2}\). The construction
does not enforce the same individual \(L\); Sections 4.3 and 6 discuss
this distinction. Neither result rules out R2. The second-order
stationarity results of Q. Zhou
([arXiv:2611.05807](https://arxiv.org/abs/2611.05807)) concern a different
oracle (including a component Hessian-vector product) and objective class.

For communication, Scaman, Bach, Bubeck, Lee and Massoulié
([arXiv:1702.08704](https://arxiv.org/abs/1702.08704)), Theorems 1, 2,
and 4, study smooth noncomposite optimization. Their decentralized gossip
matrix \(G\) is **Laplacian** type (PSD with kernel
\(\operatorname{span}\{\mathbf1\}\)); its normalized gap is
\(\gamma=\lambda_{n-1}(G)/\lambda_1(G)\) in descending order.
Theorem 2 constructs \(\alpha_{\rm S}\)-strongly convex and
\(\beta_{\rm S}\)-smooth local functions on \(\ell_2\), with a lower
bound at each node of
\(\tfrac{3\alpha_{\rm S}}2(1-16/\sqrt{\kappa_l})^{1+t/(1+\tau/(5\sqrt\gamma))}
\|\theta_{i,0}-\theta^\star\|^2\) in their span black-box model,
where \(\kappa_l=\beta_{\rm S}/\alpha_{\rm S}\) and \(\tau\) prices one
communication. The nontrivial rate is in the large-\(\kappa_l\)
regime. Their graph size in the construction depends on \(\gamma\),
and the metric is per-node rather than the gap of the averaged output.
Taking \(I-G/\lambda_1(G)\) maps the gap to the mixing-matrix convention
when \(G\) is a graph Laplacian; it does not remove the size condition.
Theorem 4 attains \(O(\sqrt{\kappa_l}(1+\tau/\sqrt{\gamma})
\log(1/\varepsilon))\) with local conjugate-gradient access. The
diameter-based centrally scheduled method in that paper has a different
communication interface. Neither is an upper bound for the
private-component/exact-shared-prox setting. The discussion after
Corollary 2 explicitly notes that \(1/\sqrt\gamma\) can overestimate
diameter. Exact finite-time consensus itself has precedents, e.g.,
Sundaram and Hadjicostis,
*Finite-Time Distributed Consensus in Graphs with Time-Invariant
Topologies* ([DOI:10.1109/ACC.2007.4282726](https://doi.org/10.1109/ACC.2007.4282726)).
The contribution here is a support-compatible weighted-sum realization
and the resulting optimization bound, not the general idea of finite-time
consensus.

A nearer noncomposite comparison is Luo, Bai, Chen, Liu and Ye,
*On the Complexity of Decentralized Smooth Nonconvex Finite-Sum
Optimization* ([arXiv:2210.13931v4](https://arxiv.org/abs/2210.13931v4)),
Theorem 5 and Corollary 2. Their variable \(f\) is a smooth
lower-bounded finite sum, their smoothness constants are for the
mean (\(L_{\rm D}\)) and for global mean-squared component gradient
differences (\(\bar L_{\rm D}\)), and the mean satisfies
\(\|\nabla f(x)\|^2\ge2\mu_{\rm D}(f(x)-f^\star)\) with
\(\mu_{\rm D}<L_{\rm D}\le\bar L_{\rm D}\). The communication
assumption is on a symmetric PSD mix, with gap \(\gamma\).
With their \(m_{\rm D}\) agents and \(n_{\rm D}\) terms per agent,
Theorem 5 obtains expected gap at most \(\varepsilon\) at **each**
agent after \(\lceil16\kappa_{\rm D}\ln(2\Delta/\varepsilon+1)\rceil\)
outer steps with the stated mixing and sampling parameters. Their
Corollary 2 gives communication
\(\widetilde O(\kappa_{\rm D}\gamma^{-1/2}\ln(\Delta/\varepsilon))\),
total expected LIFO calls
\(O((m_{\rm D}n_{\rm D}+
\min\{m_{\rm D}n_{\rm D}\kappa_{\rm D},
\sqrt{m_{\rm D}n_{\rm D}}\bar\kappa_{\rm D}\})
\ln(\Delta/\varepsilon))\),
and expected computation rounds
\(\widetilde O((n_{\rm D}+\kappa_{\rm D}+
\min\{n_{\rm D}\kappa_{\rm D},\sqrt{n_{\rm D}/m_{\rm D}}\bar\kappa_{\rm D}\})
\ln(\Delta/\varepsilon))\).
Here \(\kappa_{\rm D}=L_{\rm D}/\mu_{\rm D}\) and
\(\bar\kappa_{\rm D}=\bar L_{\rm D}/\mu_{\rm D}\);
\((m_{\rm D},n_{\rm D})=(n,m)\) in our notation.
This is incomparable in general: PL allows nonconvex global
objectives, whereas the result does not give the shared-prox
composite guarantee, and it uses per-agent convergence. On the
overlapping smooth strongly convex case at \(W=J\), the
\(m=s^2,\ \kappa=s^4,\ n\asymp s^6\) example also improves the
reported \(\kappa_{\rm D}\) term when the smoothness parameters
are both bounded by \(L\) and the rate is expressed in that bound.
When \(\kappa\) is only of order \(m\), their noncomposite computation
bound can already be of order \(m\) up to logarithms. Their
Theorem 6 has a communication lower bound of order
\(\kappa_{\rm D}/\sqrt\gamma\) times a logarithm for a **PL**
construction and \(\gamma\ge1-\cos(\pi/m_{\rm D})\);
it is not a lower bound on strongly convex composite instances.

## Proof of the network and oracle implementation

Put \(Q=(I+W)/2\). One application of \(Q\) to a stack of vectors is
one primitive mix by \(W\) followed by local averaging with the node's
own vector. \(Q\) is symmetric and nonnegative and \(Q_{ii}\ge1/2\).
Write \(P=Q^D\). We first verify \(P_{ij}>0\) under (2).

On the mean-zero subspace \(Q\) contracts the Euclidean norm by at most
\(\eta=(1+\rho)/2\): use the triangle inequality and
\(Wv=(W-J)v\). Since \(QJ=JQ=J\), for a standard unit vector \(e_j\)
\[
\|(Q^D-J)e_j\|\le\eta^D\|(I-J)e_j\|\le2\eta^D.
\]
The last inequality uses \(\|Je_j\|=1/\sqrt n\le1\).
This proves the second certificate and eventual strict positivity. To
get the first certificate, for fixed \(j\) let
\(S_k=\{i:(Q^k)_{ij}>0\}\). The power update gives
\(S_{k+1}=\{i:Q_{iu}>0\text{ for some }u\in S_k\}\).
The diagonal makes this update monotone and extensive. If the set ever
stops growing before reaching all nodes it must remain unchanged forever,
contrary to eventual positivity. It starts with one node, so \(S_{n-1}\)
contains every node. Finally \(|W_{ij}-1/n|\le\rho\) by the contraction
on the input \(e_j\). If \(n\rho<1\), \(W\) and hence \(Q\) are
entrywise positive. These arguments use the upper bound \(\rho\), not
a spectrum revealed to the algorithm.

For the fourth certificate, let \(T_k\) denote the first-kind Chebyshev
polynomial and \(S=(W-J)/\rho\). The spectral theorem and
\(|T_D(s)|\le1\) on \([-1,1]\) give \(\|T_D(S)\|_2\le1\).
Since \(SJ=JS=0\) and \(J^2=J\), by the polynomial recurrence (or by
splitting the constant and mean-zero subspaces),
\[
T_D(W/\rho)=T_D(S)+(T_D(1/\rho)-T_D(0))J .
\]
Let \(b=(1+\sqrt{1-\rho^2})/\rho\), so
\((b+b^{-1})/2=1/\rho\) and \(b\ge1+\sqrt{1-\rho}\).
Then \(T_D(1/\rho)=(b^D+b^{-D})/2>2n\) under (2).
Each entry of \(T_D(S)\) has absolute value at most \(1\), and
\(|T_D(0)|\le1\), so \(T_D(W/\rho)\) is entrywise positive.
If \((Q^D)_{ij}=0\), there is no walk from \(j\) to \(i\) of length
at most \(D\) (pad with self-loops in \(Q\)). Every power of \(W\)
of degree at most \(D\), including \(I\), is then zero in that entry.
This contradicts positivity of the degree-\(D\) polynomial. Thus it
is the ordinary \(Q^D\), not the Chebyshev polynomial, that is applied
in the algorithm.

To average arbitrary local vectors \(a_i\), nominate node \(0\) as the
root \(r\). All nodes run the same four \(D\)-round blocks below; they
only need their identity and the weighted sums. The calculation is
coordinatewise. Let \(M(s)=\mathbf 1_{\{s\ne0\}}\), with \(0/0=0\).

1. Apply \(P\) to the initial mask \(M(a_i)\). At the root the resulting
   coordinate is nonzero exactly when some \(a_i\) has that coordinate
   nonzero.
2. Apply \(P\) to the mask of this result at the root, with zero initial
   input at all other nodes. Store the result \(b_i\); on an active
   coordinate it is \(P_{ir}>0\), and otherwise it is zero.
3. Apply \(P\) to \(a_i/b_i\); let \(u_i\) be the result. On an active
   coordinate \(u_r=\sum_i P_{ri}a_i/P_{ir}=\sum_i a_i\), by symmetry.
4. Apply \(P\) to \(u_r/n\) at the root, with zero initial input at the
   other nodes, and divide the received result at \(i\) by \(b_i\).
   This returns \(n^{-1}\sum_i a_i\) to every node. If a coordinate was
   inactive, every intermediate value and the returned coordinate are zero.

Masks and division by a local vector never introduce a coordinate that
was zero in their input (for division, its numerator). The root only
starts block 2 with coordinates it learned in block 1. In all blocks,
arithmetic on already held vectors and mixed messages stays within
their coordinate union. This is a zero-respecting protocol even when
there was no initially available nonzero coordinate. The root is a
node of the given network, not an extra communication primitive; all
four blocks use precisely \(W\), and no node is given a neighbor's
individual vector or a nonlocal matrix entry.

## Proof of convergence

Let \(u\) be a minimizer and write
\(g(x)=\phi(x)-\phi(u)\) at finite-value points.
Such points include \(u,x_0\); this follows from properness, the
minimum, and the initial-gap assumption. They form a convex set.
All prox answers are finite-value points (compare the prox objective
with any finite-value point). For feasible \(x\), strong convexity
at the midpoint and minimality imply
\[
\tfrac{\mu}{4}\|x-u\|^2\le g(x).
\tag{3}
\]
The gradient of \(f\) is the mean of the component gradients and is
\(L\)-Lipschitz. Convexity supplies the lower tangent bound for \(f\).
The mean-value theorem along the segment from \(y\) to \(p\) gives
the upper tangent bound
\[
f(p)\le f(y)+\langle\nabla f(y),p-y\rangle+L\|p-y\|^2.
\]
Here one can use the usual \(L/2\) coefficient; the loose constant
above follows directly by bounding the derivative of the tangent
remainder along \([0,1]\) by \(L\|p-y\|^2\).

At the start of each epoch the common snapshot is \(b\), and
\(x=z=b\). At each of the \(R\) inner steps set
\[
y=(1-\alpha)x+\alpha z,\quad
p=\operatorname{prox}_{h/(4L)}\!\left(y-\widehat G/(4L)\right),\quad
z'=z+(p-y)/\alpha,\quad x'=p.
\tag{4}
\]
The prox subscript means prox step size \(1/(4L)\). The next snapshot
is the equal-weight average of the \(R\) values of \(x\) **before**
these steps. It is feasible by convexity; \(z\) and \(y\) need not be
feasible and are never used to evaluate \(h\).

Write \(e=\widehat G-\nabla f(y)\), \(w=(1-\alpha)x+\alpha u\).
For a prox with input \(v\), step size \(\eta_p\), and output \(p\)
we have
\[
h(p)-h(w)\le\langle p-v,w-p\rangle/\eta_p.
\]
Indeed compare the prox objective at \(p\) and \(p+t(w-p)\), bound
the \(h\)-term by convexity, expand the square, divide by \(t>0\),
and let \(t\downarrow0\). Combining this with the tangent bounds
for \(f\) and expanding \(\|w-y\|^2\) yields
\[
\phi(p)-\phi(w)\le
2L(\|w-y\|^2-\|w-p\|^2)-L\|p-y\|^2+\langle e,w-p\rangle.
\]
Now \(w-y=\alpha(u-z)\), \(w-p=\alpha(u-z')\), and
\(2L\alpha^2=\mu/4\). With
\(\mathcal P(x,z)=g(x)+\frac{\mu}{4}\|z-u\|^2\),
convexity at \(w\) and
\(-\langle e,p-y\rangle-L\|p-y\|^2\le\|e\|^2/(4L)\) give
\[
\mathcal P(x',z')+\alpha g(x)
\le \mathcal P(x,z)-\alpha\langle e,z-u\rangle+\|e\|^2/(4L).
\tag{5}
\]

For R1 use \(\widehat G=\nabla f(y)\) exactly. To do this each node
serially queries its \(m\) components at \(y\), and then averages its
local mean with the four-block protocol. For R2, at epoch start cache
all \(m\) local snapshot gradients and form the common global gradient
\(\nabla f(b)\) with one exact average. At the inner step each node
uses \(B\) independent uniform samples \(\xi_{ij}\) with replacement,
each with one new component gradient and its cached snapshot gradient.
It averages the local differences before communication, giving the
exact aggregate
\[
\widehat G=\nabla f(b)+\frac1{nB}\sum_{i,j}
   [\nabla F_i(y;\xi_{ij})-\nabla F_i(b;\xi_{ij})].
\tag{6}
\]
Use fresh independent draws across inner steps and epochs. Conditional
on the past, (6) is unbiased for \(\nabla f(y)\). Subtract the local
mean from each summand in brackets. The centered summands are independent,
so their cross inner products have mean zero. The mean squared norm of
a centered summand is the raw second moment minus the squared mean,
at most \(L^2\|y-b\|^2\). Consequently
\(\mathbb E[\|e\|^2\mid\text{past}]\le C\|y-b\|^2\)
with \(C=L^2/(nB)\), or \(C=0\) in R1. No bounded variance or agreement
of the raw gradients is used.

For an epoch conditioned on its snapshot, put \(G=g(b)\) and
\(B_t=\mathcal P(x_t,z_t)+G\). By (3) and convexity of the squared
norm,
\(\|y-u\|^2\le4\mathcal P(x,z)/\mu\) and
\(\|y-b\|^2\le 8(\mathcal P(x,z)+G)/\mu\).
The conditional mean of the linear error term in (5) vanishes.
With \(\chi=2C/(L\mu)\) we obtain
\[
\mathbb E[B_{t+1}\mid\text{past}]+\alpha g(x_t)
\le(1+\chi)B_t.
\]
Iterating and using nonnegativity of the gaps (so the multipliers
on accumulated terms can be lowered to one) gives
\[
\mathbb E B_R+\alpha\sum_{t=0}^{R-1}\mathbb E g(x_t)
\le(1+\chi)^R B_0 .
\]
We have \(B_R\ge G\), \(B_0\le3G\), and \(\chi R\le1/4\) by
the hypotheses (or \(\chi=0\)). In fact
\((1+\chi)^R(1-\chi R)\le1\): in the induction multiply
\(1-\chi(k+1)\) by \(1+\chi\), obtaining at most \(1-\chi k\).
Thus \((1+\chi)^R\le4/3\) and the expected gap of the new snapshot
is at most \(3G/(\alpha R)\le G/2\), by Jensen's inequality for
\(\phi\). Independent epochs and the tower rule give
\(\mathbb E g(b_H)\le2^{-H}\Delta\).

Every inner step uses one prox query and operates on equal common
states at all nodes: the gradient is made identical by the averaging
protocol before evaluating (4). Gradients, communication rounds,
and proximal queries use disjoint unit iterations. For R1 one step
costs \(m+4D+1\) iterations; for R2 an epoch costs \(m+4D\) for the
snapshot and \(R(B+4D+1)\) for the inner steps. Local arithmetic,
memory and output require no oracle work. Each query input, send and
output uses only available coordinates. Shared seeds are finite
arrays with independent uniform entries; taking the discrete
measurable space makes the run's seed functions measurable. At the
claimed fixed budget each node outputs the identical \(b_H\), so
the evaluator's average is \(b_H\) and its expected gap is the
one just bounded. This proves the required first-hitting bounds.

## Proof of the fast-network minimax bracket

For the upper bound apply R2 with the certified \(D\) and use \(RB\le m+R\):
\[
m+4D+R(B+1+4D)\le 2m+4D+R(2+4D).
\]
The one-channel algorithm constructed above can be used on every
legal instance; taking the indicated supremum and infimum gives
the upper half of R3. For the lower half we use two separate legal
instances, each on \(J\in\mathcal W_{n,\rho}\).

### Data term

Use dimension one, \(x_0=0\), \(u=\sqrt{2\Delta/\mu}\),
and \(h(x)=\mu x^2/2\), so the prox is \(v/(1+\eta\mu)\).
All components are zero except component \(j\) at node \(0\),
which is \(F_0(x;j)=-nm\mu u x\). Its gradient is constant
and the mean is affine. The instance is therefore individually
\(L\)-smooth (in fact \(0\)-smooth); the composite objective is
\(\mu(x-u)^2/2-\mu u^2/2\), with gap \(\Delta\) at zero.

Compare any admissible randomized algorithm to its run on the
all-zero components with the same \(h,x_0,W\), which is also legal.
In the reference run all histories and outputs have empty support:
gradients vanish, prox inputs and hence answers vanish, and mixing
zero messages gives zero. For a seed, record at most one index in
each microstep at node \(0\), assigning the dummy index \(0\) if
there is no gradient query. Up to iteration \(t_0=\lfloor m/10\rfloor\)
there are at most \(5t_0\le m/2\) recorded indices, counting distinct
indices only. The event that a given index is not recorded is
measurable: it is a finite intersection of events determined by
the measurable call encodings (communication slots can be ignored
or assigned the same deterministic dummy). The sum of the indicators
of these events is at least \(m/2\) for each seed. Integrating the
finite sum over the probability space shows that some fixed \(j\)
is missed with probability at least \(1/2\); \(1/4\) suffices here.

Plant this fixed \(j\). On its missed event the two histories coincide
through \(5t_0\): by induction, all calls and sends are the same
given identical histories, and the only differing oracle answer
would require querying \(j\) at node \(0\). Each averaged output
through \(t_0\) is still zero on the event. Thus at each of those times
the expected gap is at least \(\Delta/4\), so the hitting time is
at least \(t_0+1\). No restriction on component selection or message
width was used.

### Condition and accuracy term

Let the dimension be \(d=5T+1\), indexed by \(0,\dots,d-1\).
Put \(A=L/4\), \(w=1-q^2\), \(s=\sqrt{2\Delta/(Aw)}\) and
\(u_k=sq^k\). Define the linear map \(K\) by
\((Kx)_0=\sqrt w\,x_0\) and \((Kx)_k=x_k-qx_{k-1}\) for \(k>0\).
Use every component equal to
\[
F(x)=\tfrac A2\|K(x-u)\|^2,\qquad x_0^{\rm initial}=0,\qquad h=0.
\]
The notation \(x_0^{\rm initial}\) denotes the initial vector, not
the coordinate with index zero. The prox is identity and the optimum
value is exactly zero at \(u\). The initial gap is \(Aws^2/2=\Delta\)
since \(Ku\) is zero outside the first coordinate.

To see the class constraints, write \(K\) as a diagonal with entries
in \([0,1]\) minus \(q\) times the shift by one index with zero at
the first coordinate. Each operator has norm at most one, so
\(\|K\|\le1+q\) and the gradient \(AK^\top K(x-u)\) is
\(A(1+q)^2\le L\)-Lipschitz. For \(k>0\),
\[
(x_k-qx_{k-1})^2\ge(1-q)x_k^2-q(1-q)x_{k-1}^2
\]
by adding the nonnegative term \(q(x_k-x_{k-1})^2\). At \(k=0\)
we have \(w x_0^2\ge(1-q)x_0^2\). Sum and use that the shifted
vector has norm at most \(\|x\|\) to obtain
\(\|Kx\|^2\ge(1-q)^2\|x\|^2\). The quadratic is
\(A(1-q)^2\)-strongly convex and hence meets \(L(1-q)^2\ge4\mu\).

At a gradient query supported in indices \(<a\), the residual
\(K(x-u)\) is zero at indices \(\ge a+1\). Since \(K\) has entries
only at the diagonal and the preceding index, \(K^\top\) cannot
send a residual at a smaller index to a larger one. Thus the gradient
is supported in indices \(<a+1\). The prox and all zero-respecting
arithmetic do not enlarge the prefix, and a round of mixed sums
does not enlarge a prefix common to every node. Induction from the
zero start shows that at microstep \(v\) all available coordinates
have index \(<v\). Consequently for every seed and every \(t\le T\),
the averaged output is zero in coordinate \(5T\).

For every \(z\) and \(k>0\) set \(r_k=z_k-qz_{k-1}\). The identity
\[
w z_{k-1}^2+r_k^2-w z_k^2=(w z_{k-1}-q r_k)^2
\]
implies \(w z_k^2\le w z_0^2+\sum_{\ell=1}^k r_\ell^2\le\|Kz\|^2\).
Taking \(z=x-u\) at an output with \(x_{5T}=0\) gives
\(F(x)\ge Aw s^2 q^{10T}/2=\Delta q^{10T}>\varepsilon\).
This pointwise bound survives expectation and applies at **every**
earlier iteration, so \(T_\varepsilon\ge T+1\). The worst-case
supremum of the hitting times is at least the maximum of the two
lower bounds. Both constructions are uniform across every
admissible algorithm, with only the index \(j\) allowed to depend
on the algorithm's seed law; it does not depend on its realized seed.

## Proof of the slow-network bracket

The R4 upper bound uses \(D=n-1\) in R2. Then \(RB\le m+nR\)
implies that the per-epoch cost is at most \(2m+4n+R(5n+1)\).
The data lower bound above applies without change, using \(J\).
We prove the other lower term with one fixed instance on exactly
\(n\) nodes.

**Path in the communication class.** Let \(W\) be the reflecting
walk on \(0,\ldots,n-1\): at \(i\) it puts weight \(1/2\) on self,
\(1/4\) on \(\max\{i-1,0\}\), and \(1/4\) on \(\min\{i+1,n-1\}\).
Weights to the same node are added. It is symmetric, nonnegative
and row stochastic. For mean-zero \(v\) let
\(E(v)=\sum_{i=1}^{n-1}(v_i-v_{i-1})^2\).
For three numbers \(a,b,c\), the weighted square average with
weights \(1/2,1/4,1/4\) minus the square of the weighted average
equals
\[
\tfrac18(a-b)^2+\tfrac18(a-c)^2+\tfrac1{16}(b-c)^2 .
\]
Sum this rowwise using symmetry and drop all but the predecessor
term to obtain \(\|Wv\|^2+E(v)/8\le\|v\|^2\).
Telescoping from zero and Cauchy--Schwarz give
\((v_i-v_0)^2\le nE(v)\).
Since \(\sum_i(v_i-v_0)^2=\|v\|^2+n v_0^2\) at mean zero,
\(\|v\|^2\le n^2 E(v)\). Hence
\[
\|Wv\|^2\le (1-1/(8n^2))\|v\|^2
\le (1-1/(16n^2))^2\|v\|^2.
\]
Subtracting the mean costs no norm and
\((W-J)v=W(v-Jv)\) for general \(v\), proving membership in
\(\mathcal W_{n,\rho}\).

**Instance and gap.** Let \(a=\lfloor n/4\rfloor\). The first and
last \(a\) nodes are the left and right groups, respectively.
Use coordinate indices \(0,\ldots,k\). Use \(A=L/32\), and
otherwise use \(w,s,u,K\) from the condition lower bound above
with this \(A\) and dimension \(k+1\).
Define the unscaled left component to contain
\(Aq(x_j-x_{j-1})^2/2\) for every even \(j\ge1\), plus the affine
driver \(-A w s\,x_0\) in coordinate zero. The unscaled right
component contains the same quadratic for odd \(j\ge1\) and no
affine term. Multiply these local functions by \(n/a\) at the
corresponding nodes, use zero at the middle nodes, and make all
\(m\) components at a node identical. Thus \(f\) is the sum of the
two unscaled components. A matrix of differences restricted to
one parity has norm at most two: it is a restriction of identity
minus the shift. The components have Lipschitz gradient at most
\(4(n/a)Aq\le 28A\le L\), and they are convex. The shared penalty is
\[
h(x)=\tfrac A2\big[(1-q)^2\|x\|^2+
                  q(1-q)(x_0^2+x_k^2)\big].
\]
The endpoint terms are both counted even when \(k=0\).
It is a continuous convex diagonal quadratic; if \(h(x)=
\sum_j c_j x_j^2/2\), its exact prox has coordinate
\(v_j/(1+\eta c_j)\) by completing the one-dimensional square.
Thus it never reveals a new coordinate.

Expanding the squares in \(Kx\) gives
\[
\|Kx\|^2=q\sum_{j=1}^k(x_j-x_{j-1})^2
 +(1-q)^2\|x\|^2+q(1-q)(x_0^2+x_k^2).
\]
The cross term satisfies \(\langle Kx,Ku\rangle=w s x_0\).
Consequently, for \(F(x)=A\|K(x-u)\|^2/2\) as in the previous proof,
\(\phi(x)=F(x)-\Delta\), and the gap is exactly \(F(x)\).
The optimum is attained at \(u\); the composite is
\(\mu\)-strongly convex by \(A(1-q)^2\ge\mu\); at the zero
initial vector the gap is \(\Delta\).

**Propagation.** Set
\(r(i)=\min\{a,\max(0,i+1-a)\}\), which is 0 at the left
group, \(a\) at the right group, and changes by at most 1
along an edge. Give coordinate \(j\) at node \(i\) the timestamp
\[
\theta(i,j)=ja+
\begin{cases}
r(i),&j\text{ even},\\
a-r(i),&j\text{ odd}.
\end{cases}
\]
After \(v\) microsteps the node can use only coordinates with
\(\theta(i,j)<v\). This holds initially. Mixing and the
zero-respecting sends preserve the invariant at \(v+1\) because
\(\theta(i,j)\le\theta(i',j)+1\) on a nonzero edge \(W_{ii'}\).
The prox contributes no new coordinate. At an oracle input at
microstep \(v\), fix \(j\) with timestamp at least \(v+1\).
Its input coordinate is zero. The affine term is only at
coordinate zero in the left group, which has timestamp zero.
A quadratic derivative in coordinate \(j\) can come only from
the edge ending at \(j\) or at \(j+1\).
At a node owning the edge ending at \(j>0\), the timestamps
of \(j-1\) and \(j\) coincide: their two offsets are \(a\)
and \(0\), respectively. For an edge ending at \(j+1\),
the timestamp of \(j+1\) is at least that of \(j\).
In either case the other endpoint is also zero in the input,
so the derivative contribution vanishes. This completes
the induction for every seed and for every width.

For \(t\le T\), \(5t\le ka\le\theta(i,k)\) for every node, so
coordinate \(k\) of the averaged output is zero. The earlier
weighted-residual gap estimate gives gap at least \(\Delta q^{2k}\)
pointwise. This lower bounds the expected gap strictly above
\(\varepsilon\) at all \(t\le T\), yielding \(T_\varepsilon\ge T+1\).
Combining with the data instance proves (R4).

## Proof of R5

The upper bound is R2, and the data term uses the R3 construction.
For \(p<8\) use the R3 condition lower bound directly:
\(5T\le k\) and \(q\le1\) imply
\(\varepsilon<\Delta q^{2k}\le\Delta q^{10T}\).

For \(p\ge8\) let \(s_0=\lfloor n/p\rfloor+1\) and
\(b=\lfloor n/s_0\rfloor\). Then \(\lfloor p/2\rfloor\le b\le p\).
In fact \(n<p s_0\), and \(s_0\le2\lfloor n/p\rfloor\), so
\(\lfloor p/2\rfloor s_0\le n\).
Partition the \(n\) nodes into consecutive blocks, the first \(b-1\)
of size \(s_0\) and the last of size \(n-(b-1)s_0\).
Each size \(s_a\) is in \([s_0,2s_0]\).
Denote the reflecting path on the \(b\) blocks by \(P\) (as in R4).
At nodes \(i,j\) with block indices \(a,a'\), respectively, use
the matrix
\[
W_{ij}=\frac{s_0 P_{aa'}+(s_a-s_0)\mathbf 1_{\{a=a'\}}}{s_a s_{a'}} .
\]
This is symmetric and nonnegative, and summing over all nodes in
each block gives row sum one. A nonzero edge joins either the same
or adjacent blocks.

For a real node vector \(v\), let \(u_a\) be its average on block \(a\).
The mixed vector is constant on blocks, with values
\[
(Qu)_a=\frac{s_0}{s_a}(Pu)_a+
                 \frac{s_a-s_0}{s_a}u_a.
\]
Write \(\|u\|_s^2=\sum_a s_a u_a^2\). Jensen's inequality gives
\(\|u\|_s^2\le\|v\|^2\) and
\[
\|Qu\|_s^2\le s_0\|Pu\|^2 + \sum_a(s_a-s_0)u_a^2
\le\|u\|_s^2-s_0 E(u)/8,
\]
using the path dissipation from R4 on \(b\) vertices.
If \(v\) has mean zero, then \(\sum_a s_a u_a=0\).
As before, \((u_a-u_0)^2\le b E(u)\) and
\(\|u\|_s^2\le\sum_a s_a(u_a-u_0)^2\le nbE(u)\le2s_0 b^2 E(u)\).
We get
\(\|Wv\|^2=\|Qu\|_s^2\le(1-1/(16b^2))\|v\|^2\).
Centering and taking the square root as in R4 show
\(\|W-J\|_2\le 1-1/(32b^2)\le \rho\), using \(b\le p\).

Use exactly the split objective of R4 on the \(n\) nodes.
For node \(i\) put
\(a(i)=\min\{\lfloor i/s_0\rfloor,b-1\}\) and
\(\widetilde r(i)=\min\{\lfloor b/4\rfloor,
\max(0,a(i)-\lfloor b/4\rfloor)\}\).
On the left group \(i<\lfloor n/4\rfloor\), hence
\(4i<(b+1)s_0\) and \(a(i)\le\lfloor b/4\rfloor\).
On the right group \(4i\ge3n\ge3bs_0\), so
\(a(i)\ge2\lfloor b/4\rfloor\).
Thus the height function goes from zero on the left to
\(\lfloor b/4\rfloor\) on the right and is 1-Lipschitz across
every nonzero edge. The propagation proof of R4 uses only those
properties and applies with delay \(\lfloor b/4\rfloor\),
which is at least \(\max\{1,\lfloor p/16\rfloor\}\).
The objective parameters and component counts did not change.
This proves the lower bound at the stated \(T\), without requiring
a multiple-of-\(b\) node count. Taking the maximum with the data
bound proves R5.

## Proof of R6

At the start of an outer epoch set \(x=z=b\).
At an outer step let \(a=a_{\rm out}\),
\(y=(1-a)x+a z\), and define
\[
Q_y(v)=\phi(v)+\tfrac{\nu}{2}\|v-y\|^2 .
\]
Run the R2 algorithm for \(H\) inner epochs, initialized at \(x\),
on \(Q_y\), with strong convexity \(\mu+\nu\) and unchanged smoothness
\(L\). Its result is \(p\); the outer update is \(x'=p\),
\(z'=z+(p-y)/a\). At the end of the \(S\) steps average the
\(S\) pre-update values of \(x\) and restart at that point.

The subproblem class has the same \(f\) and components and the
regularizer \(h_y(v)=h(v)+\nu\|v-y\|^2/2\). It is proper, closed,
convex, and has the same domain as \(h\). The sum \(Q_y\) is
\((\mu+\nu)\)-strongly convex. These convexity assertions follow
from the squared-norm barycentric identity at finite endpoints;
with a \(+\infty\) endpoint and positive weight the convexity
inequality is automatic. The subproblem has a minimizer: a
nonempty sublevel at a feasible point is closed, and bounded
by \(\phi(v)\ge\phi^\star\) and the squared penalty. Lower
semicontinuity on that compact sublevel gives a minimizer,
which is also global. Its prox queries are implemented by the
identity
\[
\operatorname{prox}_{\eta h_y}(v)=
\operatorname{prox}_{\frac{\eta}{1+\eta\nu}h}
\left(\frac{v+\eta\nu y}{1+\eta\nu}\right)
\]
obtained by completing the square. This asks the original
oracle at a positive step on a vector supported on the union
of the available supports of \(v\) and \(y\). Nothing about the
unknown optimum or values of \(h\) is queried. Thus the R2 inner
run costs exactly the primitive budget of an ordinary R2 run,
with the same fixed matrix.

Fix the outer state and let \(v_\star\) be the subproblem minimum,
\(\delta(p)=Q_y(p)-Q_y(v_\star)\). The proof of R2 gives
\(\mathbb E[\delta(p)\mid x,z]\le2^{-H}(Q_y(x)-Q_y(v_\star))\);
this is a relative contraction, so the starting gap need not be
known to run the inner schedule. To control its size, let \(u\)
be an original minimizer and write \(g(x)=\phi(x)-\phi(u)\)
and \(\mathcal P(x,z)=g(x)+(\mu/4)\|z-u\|^2\).
With \(\nu a^2/2=\mu/4\),
\[
Q_y(x)-\inf Q_y\le g(x)+\tfrac\mu4\|x-z\|^2
\le 3\mathcal P(x,z).
\]
The last inequality uses \(\|x-z\|^2\le
2(\|x-u\|^2+\|z-u\|^2)\) and (3).

For any feasible \(w\) and \(t=\mu/(\mu+\nu)>0\), compare
\(Q_y(p)\) to \(Q_y((1-t)p+t w)+\delta(p)\) and use
strong convexity. After rearranging, using
\((1-t)(\mu+\nu)=\nu\), this gives
\[
Q_y(p)-Q_y(w)+\tfrac\nu2\|w-p\|^2
\le (1+\nu/\mu)\delta(p).
\]
Remove the two \(Q_y\) penalties and set \(w=(1-a)x+a u\).
As before \(w-y=a(u-z)\), \(w-p=a(u-z')\), and
\(\phi(w)\le(1-a)\phi(x)+a\phi(u)\). Dropping the nonpositive
term \(-\nu\|p-y\|^2/2\) gives
\[
\mathcal P(p,z')+a g(x)\le
\mathcal P(x,z)+(1+\nu/\mu)\delta(p).
\]
In conditional expectation the multiplier on the previous
potential is at most \(1+\chi\) with
\(\chi=3(1+\nu/\mu)2^{-H}\). Each independent outer subproblem
seed is fresh. As in the R2 telescoping argument, the expected
sum of the \(S\) gaps, multiplied by \(a\), is at most
\((1+\chi)^S\mathcal P(b,b)\).
By \(\chi S\le1/4\) and \(\mathcal P(b,b)\le2g(b)\),
this is at most \(8g(b)/3\). Jensen's inequality and \(aS\ge6\)
give a contraction of at least a factor two for the snapshot.
After \(K\) outer epochs the expected gap is at most
\(2^{-K}\Delta\).

All state vectors are formed by support-preserving arithmetic
or the explicit existing oracles. The subproblem centers are
identical across nodes, because the batched inner protocol
outputs common states. Initial and final states are feasible,
as are the snapshot averages. All randomness is finite and
independent of the instance. The nested schedule uses
\(K S H\) inner epochs of exactly the accounted cost, so its
final common state gives the hitting-time guarantee.

## Integer tuning for R7

For a strong-convexity parameter \(\sigma\le L\) set
\(t=L/\sigma\), \(v=\sqrt t\), \(R=\lceil18v\rceil\), and
\(B=\lceil8tR/n\rceil\). Here \(t\ge1\), and
\(\sqrt{\sigma/(8L)}\,v=1/\sqrt8\ge1/3\), so (1)'s inner
step requirement holds. Also \(B\ge1,\ nB\ge8tR\),
and \(R\le20v\). The actual cost of one batched epoch satisfies
\[
\begin{split}
 RB&\le R+8tR^2/n\le R+3200t^2/n,\\
 C:=m+4D+R(B+1+4D)
     &\le m+3200t^2/n+120v(D+1).
\end{split}
\]
For the second line use \(4D\le4vD\) and
\(R(2+4D)\le20v(2+4D)\).

If \(\kappa<2\sqrt{nm}\), take \(\sigma=\mu\) and use R2
with \(K\) epochs. Its per-epoch cost is at most
\(12801m+120\sqrt\kappa(D+1)\), which is bounded by
the expression inside the ceiling in (R7), and \(H\ge1\).
Otherwise take \(\sigma=L/\sqrt{nm}\), \(\nu=\sigma-\mu\),
and \(S=\lceil9\sqrt\kappa/v\rceil\).
These choices give \(t=\sqrt{nm}\), \(\mu\le\nu\), and
\(\mu+\nu=\sigma\le L\).
With \(o=\sqrt\kappa/v\ge1\) we have
\((a_{\rm out}o)^2=\sigma/(2\nu)\ge1/2\), so
\(a_{\rm out}S\ge6\). Also \(S\le10o\le10\sqrt\kappa\);
the prescribed \(H\) then gives
\(3(1+\nu/\mu)2^{-H}S\le1/4\).
Since \(t^2=nm\), \(C\le3201m+120v(D+1)\) and
\[
SC\le32010\sqrt\kappa\,m/v +1200\sqrt\kappa(D+1).
\]
Use R6 and round up to (R7). This choice depends only on the
public parameters, and preserves the inf-sup quantifiers.
The two lower-bound certificates in R7 are exactly the data and
radius proofs of R5, which did not use any upper solver parameters.

## Lean correspondence

`Theory/Composite.lean` proves the real-valued gap facts on the
finite domain within `Instance.InClass`, the gradient identities,
and (5). `Theory/FiniteMean.lean` implements the uniform finite
expectation and independently centered variance identity;
`Theory/Epoch.lean` proves the conditional epoch and restart bounds.
`Theory/Network.lean` and `Theory/Spectral.lean` certify the mixing powers, and
`Theory/Aggregation.lean` proves the four-block protocol.
`Theory/Protocol.lean` proves that the local continuation rules
compile into the fixed five-microstep history model, including
admissibility on arbitrary histories. `Theory/Schedule.lean` and
`Theory/Batch.lean` give the schedules. `Theory/Upper.lean` ties them to the
lower Lebesgue integral and `Algorithm.hittingTime`.
`Theory/Hidden.lean` and `Theory/Chain.lean` instantiate the lower
bounds via `Theory/LowerTools.lean`; the upper and lower witnesses
are combined at the fixed minimax quantifiers in R3.
`Theory/Path.lean` proves the communication-class membership of
the fixed-size path, `Theory/Split.lean` and `Theory/SplitInstance.lean`
check the split quadratic and prox, and `Theory/SlowLower.lean`
proves the propagation bound for R4.
`Theory/Blocks.lean` and `Theory/Lifted.lean` prove the
arbitrary-size clustered path construction for R5, and
`Theory/ClusterLower.lean` instantiates the same propagation
lemma and selects the integer scale.
`Theory/Regularized.lean` defines the subproblems and proves
existence and the prox translation; `Theory/Inexact.lean` proves
the outer potential and restart inequality; `Theory/Nested.lean`
implements the fixed-budget composition in R6.
`Theory/Rate.lean` chooses the inner and outer budgets and proves the
explicit rounded bound of R7.
