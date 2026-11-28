# Data lower bounds and two-node complexity for decentralized composite finite sums

**Targets A, C and D.** Write \(N=nm\), \(\kappa=L/\mu\) and \(R=\Delta/\varepsilon\). All iteration counts below use the averaged-output, expected-gap, first-hitting-time model of `Theory/Model.lean`. In particular, one full local gradient costs \(m\) queries, and none of the lower bounds restricts the algorithm to linear combinations or to an oblivious index schedule.

- **R1 versus H-LB-data (strictly stronger in the registered model).** Woodworth and Srebro, [arXiv:1605.08003v1](https://arxiv.org/abs/1605.08003v1), Theorem 8, prove a lower bound of \(\Omega(M+\sqrt{M\gamma/\lambda}\log((\epsilon_0/\epsilon)\sqrt{M\lambda/\gamma}))\) component calls for randomized algorithms with a component value/gradient/prox oracle, with \(M\) \(\gamma\)-smooth, \(\lambda\)-strongly convex components on their domain, \(\gamma/\lambda>161 M\), and \(\epsilon_0>60\epsilon\sqrt{\gamma/(\lambda M)}\). The registered conversion uses \(M=N,\gamma=L,\lambda=\mu,\epsilon_0=\Delta\) and at most \(n\) component calls per iteration. R1 replaces \(\log(R\sqrt{N/\kappa})\) by \(\log R\) and needs only \(R\ge 2\) for its high-\(\kappa\) corollary. Its exact form also holds for every \(1<R<\infty\) and every \(\kappa\ge 1\). This is a joint accuracy/condition-number improvement, not a claim to improve the polynomial exponent in \(m\) at arbitrarily high accuracy with \(\kappa,N\) held fixed.
- **R2 versus H-UB (necessity of smoothness control beyond smooth local averages).** Liu and Low, [arXiv:2402.02356v1](https://arxiv.org/abs/2402.02356v1), Theorem 5.1 and Corollary 5.3, analyze PMGT-KatyushaX with individually \((\ell_1,\ell_2)\)-smooth components, a smooth convex average and a strongly convex composite, using static symmetric positive semidefinite doubly stochastic mixing with a simple eigenvalue at one. The registered round bound is
  \[
  \widetilde O\!\left(\left[m+m^{3/4}\sqrt\kappa+
      (\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2}\right]\log R\right).
  \]
  R2 proves that this order cannot hold if the individual smoothness assumption is replaced by only \(L\)-smoothness of each local average, using that local \(L\) in \(\kappa\). This impossibility holds for **every** admissible algorithm even on \(W=J\), with exact, support-preserving shared prox and convex quadratic components. It is not a lower bound with individual smoothness constant \(L\).
- **R3 versus H-UB (tradeoff and matching range).** On two nodes, an exact weighted-sum decoder and restarted proximal acceleration yield matching lower and upper bounds \(\Theta(m\sqrt{\kappa}\log R)\) under the local-average-only relaxation, uniformly in the spectral gap. They also give \(\Theta(\sqrt{\kappa}\log R)\) in the standard class with \(n=2,m=1\). The upper uses less smoothness information and fewer mixing rounds than the registered H-UB rate at \(n=2\), but full local gradients at each accelerated step can cost more data queries when \(m\) is large. This is a two-node result, not an upper bound for unrestricted \(n\).
- **R4 versus H-UB and H-LB-net (connectivity at the endpoint).** Allowing the identity matrix makes the minimax time infinite even though the evaluator averages the nodes' outputs. This proves a necessity of connectivity for any uniform finite guarantee over the resulting class. It does not improve H-LB-net for \(\rho<1\).

## R1. Standard-model data bound with no condition-dependent accuracy loss

**Statement.** Let \(n\ge2\), \(m\ge1\), \(L\ge\mu>0\), \(0\le\rho<1\), and \(0<\varepsilon<\Delta\). Let \(c\) be any fixed message width (the lower bound also allows \(c=0\)). For the exact expression below set
\[
\ell=\frac{L}{5N},\qquad r=\frac{\sqrt{1+4\ell/\mu}-1}{2},\qquad q=\frac{r}{1+r},
\]
and write
\[
B(q)=\left\lceil\max\left\{m(1-\varepsilon/\Delta),
              \frac{m\log R}{2\log(1/q)}\right\}\right\rceil .
\]
Then \(\operatorname{minimax}(n,m,c,L,\mu,\Delta,\rho,\varepsilon)\ge B(q)\).
In particular, if \(\kappa\ge5N\) and \(0<\varepsilon\le\Delta/2\), then
\[
\operatorname{minimax}(n,m,c,L,\mu,\Delta,\rho,\varepsilon)
\ \ge\
\left\lceil\frac{m}{4} + \frac{1}{20}\sqrt{\frac{m\kappa}{n}}\,\log R\right\rceil .
\]
The lower bound is independent of \(\rho\); the witness can be run on \(W=J\). It does not purport to lower-bound the additional network term on poorly connected graphs.

**Lean:** `ULB63.minimax_data_no_accuracy_prefactor`, `ULB63.minimax_data_high_condition`.

### Proof

**Instance and smoothness.** We first keep the positive coupling \(\ell\) symbolic. For a nonnegative integer \(p\) take one block of \(p+1\) Euclidean coordinates for each \(b=(i,\xi)\). Write \(x_{b,j}\) for coordinate \(j\in\{0,\ldots,p\}\). Take \(x_0=0\), \(h(x)=\mu\|x\|^2/2\), and \(\operatorname{prox}_{\eta h}(v)=v/(1+\eta\mu)\) for \(\eta>0\). The regularizer is proper, closed and convex, and the proximal formula follows by completing the square. Choose \(q\) as above so that \(0<q<1\) and
\[
\mu=\ell(1-q)^2/q,\qquad \alpha=\ell(1-q)/q,\qquad
a=\sqrt{\frac{2\Delta}{N\alpha}} .
\]
Indeed \(r(1+r)=\ell/\mu\), and \(q/(1-q)^2=r(1+r)\). Define the unnormalized block function
\[
G_b(x)=\frac{\ell}{2}\sum_{j=0}^{p-1}(x_{b,j+1}-x_{b,j})^2
 +\frac{\ell(1-q)}2 x_{b,p}^2-\alpha a x_{b,0}.
\tag{1}
\]
Node \(i\)'s component \(\xi\) is \(F_i(x;\xi)=N G_{(i,\xi)}(x)\). Thus \(f=\sum_b G_b\). Each \(G_b\) is convex: it is a sum of positively weighted squares of linear maps plus a linear function. If \(D_bx\) is the vector of edge differences and \(P_bx=x_{b,p}\), then
\[
\nabla G_b(x)=H_b x-\alpha a e_{b,0},\qquad
H_b=\ell D_b^\top D_b+\ell(1-q)P_b^\top P_b.
\tag{2}
\]
Here \(\|D_b\|\le2\) by \((u-v)^2\le2u^2+2v^2\), since each vertex touches at most two edges, and \(\|P_b\|\le1\). Consequently \(\|H_b\|\le5\ell\) and the individual smoothness bound is \(5N\ell=L\) for the chosen \(\ell\). Convex \(f\) plus \(h\) is \(\mu\)-strongly convex on \(\mathbb R^{N(p+1)}\).

**Exact initial gap.** For one block the identity
\[
\begin{split}
G_b(x)+\frac{\mu}{2}\sum_{j=0}^{p}x_{b,j}^2+\frac{\alpha a^2}{2}
   &=\mathcal E_b(x),\\
\mathcal E_b(x)&=\frac{\alpha}{2}(x_{b,0}-a)^2
  +\frac{\ell}{2q}\sum_{j=0}^{p-1}(x_{b,j+1}-q x_{b,j})^2
\end{split}
\tag{3}
\]
is exact, including the two endpoints. For example, writing \(u=x_{b,j}\), \(v=x_{b,j+1}\),
\[
\frac{\ell}{2q}(v-qu)^2
=\frac{\ell}{2}(v-u)^2+\frac{\alpha}{2}v^2-\frac{\ell(1-q)}2u^2 .
\]
Summing these identities and substituting \(\mu=\ell(1-q)^2/q\) gives (3). This also covers \(p=0\). All energies are nonnegative and they vanish simultaneously at \(x^\star_{b,j}=a q^j\). It follows that \(\phi^\star=-N\alpha a^2/2=-\Delta\) and that the initial gap is exactly \(\Delta\), not an upper bound in terms of distance times the global smoothness.

**One omitted coordinate gives an objective-gap certificate.** Set \(e_j=x_{b,j}-a q^j\) and \(u_j=x_{b,j+1}-q x_{b,j}\). Then \(e_{j+1}=q e_j+u_j\). For \(0<q<1\),
\[
(q e+u)^2
\le q e^2+\frac{u^2}{1-q}
\le e^2+\frac{u^2}{1-q}.
\]
The first inequality is the convexity of the square applied to \(e\) and \(u/(1-q)\). Induction gives
\(e_k^2\le e_0^2+(1-q)^{-1}\sum_{j<k}u_j^2\).
Since \(\alpha/(1-q)=\ell/q\), (3) gives \(\mathcal E_b(x)\ge\alpha e_k^2/2\) for every \(k\le p\). In particular,
\[
 x_{b,k}=0\quad\Longrightarrow\quad
 \mathcal E_b(x)\ge(\Delta/N)\,q^{2k}.
\tag{4}
\]
This is where the left endpoint differs from a chain with an extra \(x_{b,0}^2\) term. It avoids a small ratio of a distance-based lower estimate to the actual initial gap.

**Budget and zero-respecting invariant.** Fix a seed, an algorithm and a matrix \(W\). At any micro-step let \(k_b\) count how often the owner of block \(b\) has queried its component so far. For every node, the initial point and every vector received or observed so far vanish in coordinates \(j\ge k_b\) of block \(b\). Initially all coordinates are zero. A query input, send or output outside the allowed support vanishes by admissibility. The proximal response is a scalar multiple of its input. A mixing step is a weighted sum of sends from the immediately preceding histories, so it cannot create a coordinate that is zero in all those sends. By (2), a gradient outside its own block vanishes there identically. Within the queried block it is tridiagonal, except for the forcing term at coordinate zero. If the input is zero at \(j\ge k_b\) it is zero at \(j>k_b\) in the response, exactly as needed after incrementing \(k_b\) by one. These arguments prove the invariant by induction even when the order and content of calls are adaptive. No statistical independence of query counts is needed.

At the end of \(t\) iterations, each owner has made at most \(t\) component queries, so \(k_b\le t\) and \(\sum_b k_b\le nt\). The invariant also applies to the evaluator's average: averaging local outputs still gives zero at \(j=k_b\) of every block when \(t\le p\). Summing (4) and using Jensen's inequality for the exponential gives, seed by seed,
\[
\phi(\widehat x_t)-\phi^\star
\ \ge\ \frac{\Delta}{N}\sum_b q^{2k_b}
\ \ge\ \Delta\exp\!\left(2(t/m)\log q\right).
\tag{5}
\]
The inequality uses \(\log q<0\) and \(\frac1N\sum_b k_b\le t/m\). Also \(q^{2k_b}\ge 1-k_b\) for integer \(k_b\ge0\), which gives the separate bound \(\Delta(1-t/m)\) from (4). These lower bounds survive the lower Lebesgue integral because they are uniform in the seed. The formalization counts gradient observations in each actual iteration (not in an algorithmically chosen batching of iterations).

**First hit.** Take \(p=B(q)\), giving a single dimension \(d=N(p+1)\) before the run. If any \(t<B(q)\) had expected gap at most \(\varepsilon\), (5) and the startup bound would imply both \(t\ge m(1-\varepsilon/\Delta)\) and \(t\ge m\log R/(2\log(1/q))\). This is impossible for integer \(t<B(q)\). The same instance works for all early times and all seeds and is independent of the algorithm. In particular \(W=J\) is a legal witness for every \(\rho\ge0\). Taking the infimum over algorithms proves the first statement. This argument does not substitute a fixed-budget complexity definition for the first hit.

**High condition number.** If \(\ell/\mu\ge1\), then \(r>1/2\) and
\(\sqrt{5\ell/\mu}\le5r\), from \(r(1+r)=\ell/\mu\). Also \(\log(1/q)=\log(1+1/r)\le1/r\). Thus the exponential term in \(B(q)\) is at least \(m\sqrt{5\ell/\mu}\log R/10\). For \(\varepsilon\le\Delta/2\) the startup term is at least \(m/2\). The maximum is at least half the sum. Substituting \(\ell=L/(5N)\) proves the corollary.

## R2. Smooth local averages alone do not preserve the H-UB order

**Statement (PROFILE §3, function-class relaxation).** Keep the standard parameters, geometry, communications, and gradient/shared-prox access. Let \(L\) bound the smoothness of each *local average* \(f_i\). For \(1\le\theta\le m\), allow components to be \(\theta L\)-smooth. There is an instance in this relaxed class for which every admissible algorithm requires at least \(B(q)\) rounds, with the same expression as in R1 but with \(\ell=\theta L/(5N)\). The instance forces the lower bound even on \(J\), and in fact the bound holds for every matrix the algorithm is run with. When \(\theta\kappa\ge5N\) and \(\varepsilon\le\Delta/2\), the expression is at least
\[
 \left\lceil \frac{m}{4}+
  \frac{m}{20}\sqrt{\frac{\theta L}{\mu nm}}\log R\right\rceil .
\tag{6}
\]
In particular the class with *only* local-average smoothness contains the \(\theta=m\) witnesses. No \(C>0\) and fixed nonnegative integer \(s\) give a uniform upper bound of \(\lceil C m^{5/4}(1+\log m)^s\rceil\) on the two-node sequence \(L=m,\ \mu=\Delta=1,\ \varepsilon=1/m,\ W=J\). This rules out the registered H-UB rate with the weakened smoothness assumption, even with arbitrary fixed polylogarithmic factors.

**Lean:** `ULB63.local_average_data_lower`, `ULB63.local_average_only_breaks_hub`. The proof uses `ULB63.local_smooth_high` for (6). `Instance.localAverage` is the local finite-sum average, not a mean-square gradient condition. The `InClass (θ * L)` conjunct builds hard instances with a finite individual bound inside the relaxed class. `Admissible L` is sufficient for the lower bound: any algorithm admissible throughout the local-average-only class is admissible on the smaller standard class (whose \(L\)-smooth components imply \(L\)-smooth local averages), and the proof uses only the all-history budget and support restrictions, not measurability on any larger class.

### Proof

Use (1) with the larger \(\ell\) and redo the same exact gap normalization. All class and trajectory claims of R1 are unchanged except the bound \(5N\ell\) is now \(\theta L\).
Let \(\Pi_b\) be the orthogonal projection onto block \(b\). From (2), \(H_b v=H_b\Pi_b v\) and this vector is supported on \(b\). The local average is \(f_i=n\sum_\xi G_{(i,\xi)}\). Thus, with \(v=x-y\),
\[
\begin{split}
\|\nabla f_i(x)-\nabla f_i(y)\|^2
 &= n^2\sum_\xi\|H_{(i,\xi)}\Pi_{(i,\xi)}v\|^2\\
 &\le (5n\ell)^2\sum_\xi\|\Pi_{(i,\xi)}v\|^2
 \le (5n\ell)^2\|v\|^2.
\end{split}
\]
The first equality uses mutually orthogonal supports. The local Lipschitz modulus is therefore at most \(5n\ell=\theta L/m\le L\). The proof of R1 up through \(B(q)\) applies with this \(\ell\), and its high-condition analysis gives (6).

For the claimed separation take \(n=2,\theta=m,L=m,\mu=\Delta=1,\varepsilon=1/m\), with \(m\ge10\). The hypothesis \(5N\mu\le\theta L\) becomes \(10m\le m^2\), and (6) is at least \(\left\lceil m/4 + m\sqrt{m/2}\log m/20\right\rceil\). With \(\rho=0\) the four powers of \(m\) in the registered upper bound are \(m,\ m^{5/4},\ m^{1/2},\ m^{3/4}\); the precision factor is \(\log m\). These, and any fixed polylogarithmic overhead in the declared parameters, are bounded above by \(C m^{5/4}(1+\log m)^s\) for suitable fixed \(C,s\) on this sequence. Since \(m^{1/4}\) grows faster than every fixed power of \(\log m\), the displayed lower bound exceeds the proposed upper bound for all large \(m\). The margin can be taken greater than one before the ceilings, as in the Lean proof. This is an impossibility for the class, not merely a failure of a given tuning of PMGT-KatyushaX.

## R3. Matching two-node bounds by exact averaging and proximal acceleration

**Upper bound.** Let \(n=2\), \(m\ge1\), \(L\ge\mu>0\), and \(\Delta,\varepsilon>0\). It suffices that the components are differentiable and the local averages \(f_0,f_1\) have \(L\)-Lipschitz gradients. All the other function and regularizer assumptions of the standard model are unchanged. For every static two-node matrix of the allowed type with a nonzero off-diagonal entry, a deterministic, one-channel admissible algorithm attains the target by round
\[
T=(m+2)\left\lceil\sqrt{16 L/\mu}\right\rceil
           \left\lceil\frac{\log R}{\log 2}\right\rceil_+ .
\tag{7}
\]
Here \(\lceil\cdot\rceil_+\) truncates at zero. The algorithm does not take \(\rho\), an individual smoothness bound, or an initial gradient discrepancy as an input. In particular (7) is an upper bound on the standard-model minimax for \(n=2,c=1\) and all \(0\le\rho<1\).

**Matching lower bound.** For \(0<\varepsilon<\Delta\) in the same local-average-only class, every admissible algorithm takes at least \(B(q)\) on a common hard instance with \(W=J\), where \(q\) is the ratio of R1 for \(\ell=L/10\). For \(R\ge2\) these upper and lower bounds give order \(\Theta(m\sqrt{\kappa}\log R)\), independent of \(\rho\). For the standard class with \(n=2,m=1\), R1 supplies the same lower bound (with \(m=1\)), so (7) also matches it.

**Lean:** `ULB63.two_node_local_average_upper`, `ULB63.two_node_local_average_lower`, `ULB63.two_node_standard_upper`. `Instance.InLocalClass` expresses the relaxed class by requiring differentiability of the real components and applying the fixed geometry tests to a verification instance with each component replaced by its local average. This replacement leaves \(f,h,x_0,\operatorname{prox}\) unchanged when \(m\ge1\); it is not used to query an average for the price of one component.

### Algorithm and proof

**Exact communication.** If \(i\) is one node and \(j\) the other, a round with inputs \(g_i,g_j\) returns only \(r_i=w_{ii}g_i+w_{ij}g_j\). As \(w_{ij}\ne0\), node \(i\) computes
\[
\bar g_i=\tfrac12\{g_i+(r_i-w_{ii}g_i)/w_{ij}\}=\tfrac12(g_i+g_j).
\tag{8}
\]
This needs one channel, the local row and exact arithmetic. In the standard class a zero off-diagonal entry would imply \(W=I\) by symmetry and stochasticity, contradicting \(\|(W-J)(1,-1)\|\le\rho\|(1,-1)\|\) for \(\rho<1\). No quantitative spectral knowledge is needed to run (8).

**One accelerated step and its schedule.** At the start of an epoch set \(x=z\) to the last restart point (initially \(x_0\)). In epoch step \(j=0,1,\ldots,p-1\), let \(a_j=(j+2)/2\) and set
\[
y=(1-a_j^{-1})x+a_j^{-1}z,\qquad
x^+=\operatorname{prox}_{h/L}(y-L^{-1}\nabla f(y)),\qquad
z^+=a_j x^+-(a_j-1)x .
\]
The notation \(\operatorname{prox}_{h/L}\) denotes the model's step \(\eta=1/L\). For \(m\) iterations each node queries its components at the common \(y\), one at a time. In the next iteration it sends the average of its \(m\) gradients and decodes using (8). In the final iteration it makes one prox query at the displayed input. Both nodes therefore have the same new \(x,z\). They output the most recently completed \(x\); after \(p\) steps they set \(z=x\) and restart. The rules only use linear combinations of the initial vector or previously observed vectors for all query inputs, sends and outputs, including (8). Each query takes place in the first oracle slot of its assigned iteration; there are no other calls in that iteration. The Lean algorithm reads the requisite gradients and prox answers from their indexed positions in its local history. It has a singleton seed, making all seed maps measurable. Thus the evaluator sees the same iterate as both nodes at the times used in (7).

**Three-point inequality.** Let \(x^\star\) be a minimizer and \(G(x)=\phi(x)-\phi^\star\) on \(\operatorname{dom}h\). Properness gives a feasible point; the minimizer and the finite-gap start are feasible as well. For any \(y\) and \(u\in\operatorname{dom}h\), set \(v=y-L^{-1}\nabla f(y)\) and \(x^+=\operatorname{prox}_{h/L}(v)\). The proximal objective \(Q(w)=h(w)+(L/2)\|w-v\|^2\) is \(L\)-strongly convex and \(x^+\) is a feasible minimizer. For \(0<t<1\), compare \(Q(x^+)\) to \(Q((1-t)x^++tu)\), use strong convexity, and let \(t\) decrease to zero to obtain \(Q(x^+)+(L/2)\|u-x^+\|^2\le Q(u)\). Expanding the squares about \(y\), using the \(L\)-smooth upper tangent quadratic for \(f(x^+)\) and the convex lower tangent for \(f(u)\), gives
\[
\phi(x^+)\le\phi(u)+\frac L2(\|u-y\|^2-\|u-x^+\|^2).
\tag{9}
\]
The finite average of the \(L\)-smooth \(f_i\)'s is \(L\)-smooth. The smooth upper tangent inequality follows by applying the gradient Lipschitz bound along the line segment from \(y\) to \(x^+\); the lower tangent inequality is first-order convexity. No derivative of \(h\) is used.

**Epoch contraction.** In (9) take \(u=(1-a_j^{-1})x+a_j^{-1}x^\star\), which is feasible by convexity. Then \(u-y=a_j^{-1}(x^\star-z)\) and \(u-x^+=a_j^{-1}(x^\star-z^+)\). By convexity of \(\phi\) and multiplication by \(a_j^2\),
\[
a_j^2 G(x^+)+(L/2)\|z^+-x^\star\|^2
\le (a_j^2-a_j)G(x)+(L/2)\|z-x^\star\|^2 .
\]
The initial weight on the right is zero because \(a_0=1\). For \(j\ge1\), \(a_j^2-a_j\le a_{j-1}^2\); nonnegativity of \(G\) allows telescoping. Consequently at the end \(G(x_p)\le2L\|x_{\rm start}-x^\star\|^2/(p+1)^2\). Strong convexity at the midpoint of \(x_{\rm start},x^\star\) and minimality of \(x^\star\) give \(\mu\|x_{\rm start}-x^\star\|^2/4\le G(x_{\rm start})\). Taking \(p=\lceil\sqrt{16 L/\mu}\rceil\) thus halves the gap. After \(K=\lceil \log R/\log 2\rceil_+\) epochs it is at most \(2^{-K}\Delta\le\varepsilon\). The \(K=0\) case uses the initial gap. The horizon is deterministic and costs \(pK(m+2)\) original iterations, with \(pK\) actual mixing operations.

**Matching.** Take \(n=2,\theta=m\) in R2, keeping the local smoothness \(L\), so \(\ell=L/10\). For \(\kappa\ge1\), \(r(1+r)=\kappa/10\) implies \(r\ge1/20\) and \(\sqrt{\kappa}\le15r\). Hence \(\log(1/q)\le 15/\sqrt{\kappa}\) and \(B(q)\ge m\sqrt{\kappa}\log R/30\); the startup term \(m(1-\varepsilon/\Delta)\) is also retained in \(B(q)\). Bound (7) has order \((m+2)\sqrt{\kappa}(1+\log R)\) for \(R>1\). For \(R\ge2\) and \(m\ge1\) these bounds match up to numerical constants. Under \(m=1\), R1 already uses this very value of \(\ell\), yielding the standard-class corollary.

**Comparison (incomparable as a general upper bound).** The acceleration estimate itself has the usual proximal accelerated gradient scaling; it is not an acceleration claim beyond the classical full-gradient method. Scaman et al., [arXiv:1702.08704v1](https://arxiv.org/abs/1702.08704v1), §3.2 after Corollary 1, already discuss evaluating the global gradient by routing local gradients to a master, taking a Nesterov step, and broadcasting the next iterate on smooth problems. R3 uses the actual weighted-sum oracle with neither master nor individually transmitted messages, handles extended-real \(h\) and strong convexity of the composite only, and counts the component calls and proxes in the same model as the matching lower bound. Compared to H-UB's \((\sqrt m+m^{1/4}\sqrt\kappa)(1-\rho)^{-1/2}\log R\) network term, the count \(pK=O(\sqrt\kappa\log R)\) at \(n=2\) requires no factor depending on \(m\) or the gap. In particular there is strict round improvement for fixed \(m\) and \(\rho\uparrow1\). The larger data budget \(mpK\) for general \(m,\kappa\), and the restriction to two nodes, prevent a dominance claim over H-UB. This is a matching result in a stated range and class, relevant to Liu–Low's explicit communication question in their conclusion. For an implementation the previous two prox outputs, a running gradient sum, and a mixed vector suffice; the Lean result certifies the one-channel width and the schedule, not a separate arithmetic/bit-complexity bound.

## R4. Removing connectivity can make the averaged-output problem unsolvable

**Statement (communication relaxation).** Retain all standard function and algorithm assumptions. Allow \(\rho=1\), so in particular \(W=I_n\) is legal. For \(n\ge2\), \(m\ge1\), set \(\ell=(n-1)L/(2n)\) and \(q_0=\ell/(\mu+\ell)\). If \(0<\varepsilon<\Delta q_0^2\), then
\[
 \operatorname{minimax}(n,m,c,L,\mu,\Delta,1,\varepsilon)=+\infty .
\tag{10}
\]
The threshold is at least \(\Delta/25\) for all standard \(\kappa\ge1\) and tends to \(\Delta\) as \(\kappa\to\infty\) with \(n\) fixed. The bound is independent of \(m,c\) because even unlimited local attempts on this instance cannot transfer support between nodes.

**Lean:** `ULB63.minimax_infinite_at_disconnected_endpoint`.

### Proof

Use dimension two and let \(u,v\) be the two coordinates. Label one node the source and repeat the same component at each of its \(m\) indices. Set \(s=\mu+\ell\), \(\beta=s-\ell^2/s>0\), \(a=\sqrt{2\Delta/\beta}\). At the source use the affine component \(-n\beta a u\); at each other node and index use \(L(v-u)^2/4\). The latter has Hessian norm \(L\), since the squared norm of \((-1,1)\) is \(2\); the affine component has constant gradient. Both are convex. Let \(h(x)=\mu\|x\|^2/2\) and start at the origin. The proximal map is the scaling of R1. The average is \(f=\ell(v-u)^2/2-\beta a u\); \(\phi=f+h\) is \(\mu\)-strongly convex. Completing squares gives
\[
\phi(u,v)+\Delta=\frac{s}{2}(v-q_0u)^2+\frac{\beta}{2}(u-a)^2 .
\]
This proves that the optimum \((a,q_0 a)\) has value \(-\Delta\) and the initial gap is \(\Delta\). If \(v=0\), the gap equals \(s(u-\beta a/s)^2/2+\Delta q_0^2\).

The identity has the required nonnegativity, symmetry and row sums. For the contraction bound at one, \(I-J\) subtracts the average: \(\sum_i (z_i-\bar z)^2=\sum_i z_i^2-n\bar z^2\le\sum_i z_i^2\). With this matrix all non-source nodes remain confined to empty support. Their components have zero gradient at the origin, their prox preserves zero, and the only message each receives is its own. At the source the affine gradient and the prox can never create a nonzero second coordinate. Induction on micro-steps proves both facts for every seed, so the averaged output always has \(v=0\). Its gap is at least \(\Delta q_0^2>\varepsilon\) before taking any expectation, and there is no first hitting time.

**Comparison (connectivity necessity, outside the finite-\(\rho\) bound).** Liu–Low's Definition 3.2 (doubly stochastic matrix properties in §3.2) assumes \(\ker(I-W)=\operatorname{span}\{\mathbf1\}\). Scaman et al.'s network lower bound in Theorem 2 uses a positive normalized eigengap. Their local-output formulation does not by itself rule out optimizing through a disconnected network when an evaluator averages the outputs: some objectives, such as a shared isotropic quadratic plus local linear terms, can in fact be solved that way. Here the missing coordinate is a property of the evaluator's average itself. Together with R3, the endpoint example shows that at two nodes a quantitative contraction bound is unnecessary for finiteness, but dropping connectivity altogether is not possible.

## Comparison details and scope

**H-LB-data, Woodworth–Srebro.** Theorem 8 of the version cited above concerns any randomized algorithm under the sequential component oracle returning value, gradient and a component prox, on a domain \(\mathcal X\). It requires initial gap exactly \(\epsilon_0\), and its stated objective-gap target is strictly below \(\epsilon\). Taking a constant-factor shift of tolerance if needed accommodates a weak inequality without eliminating the accuracy prefactor. We compare the registered centralized-simulation bound for the present oracle/model, not the unrestricted-oracle theorem as a statement about its larger algorithm class. R1 does **not** improve a lower bound for arbitrary non-zero-respecting algorithms or one with a proximal oracle for every component. In the registered setting, R1 implies a lower bound of the order of the converted H-LB-data throughout the common validity range, since \(\log R\ge \log(R\sqrt{N/\kappa})\) there. The improvement is unbounded even *in* that common range: for integers \(k\to\infty\), use \(n=2,\ m=k^2,\ \kappa=Nk^2,\ R=120k,\ \rho=0\). The registered lower expression is \(O(k^3)\), whereas R1 is \(\Omega(k^3\log k)\), with the same \(\varepsilon\) in both. The threshold \(60\sqrt{\kappa/N}\) in the paper matters; this example is within it, not obtained by evaluating an out-of-range formula.

**H-LB-zsh, Zhang–Shu–He.** [arXiv:2010.08766v2](https://arxiv.org/abs/2010.08766v2), Theorem 3.1 and §4.1, prove an IFO lower bound on smooth strongly convex components, with \(\|x^0-x^\star\|=B_0\) and \(0<\epsilon<\mu B_0^2/4\). The high-condition branch (\(L/\mu>7M/2\), \(M\) components) is \(\Omega((M+\sqrt{M L/\mu})\log(\mu B_0^2/(4\epsilon)))\); the large-\(M\) branch has order \(\Omega(M+(M/\log(M\mu/L))\log(1/(M\epsilon)))\) in the paper's notation. The proof of the former counts at most one new coordinate per component gradient in each block; this argument uses a support/first-order-iterate restriction, irrespective of how broadly "randomized incremental gradient method" is stated. It is a direct precedent for the disjoint-block design and Jensen step in R1. The important difference is normalization. In the high-condition construction in §4.1, put \(\ell_{\rm zsh}=(L-\mu)/(4M)\), with minimizer entries \((q_{\rm zsh},\ldots,q_{\rm zsh}^{p_0})\) in each block and \(\mu=\ell_{\rm zsh}(1-q_{\rm zsh})^2/q_{\rm zsh}\). Then \(D_0=F(x^0)-F(x^\star)=M\ell_{\rm zsh}q_{\rm zsh}/2\), since the linear forcing in each block is \(\ell_{\rm zsh}\) at its head. The geometric sum gives
\[
\frac{\mu B_0^2}{4D_0}
=\frac{(1-q_{\rm zsh})(1-q_{\rm zsh}^{2p_0})}{2(1+q_{\rm zsh})}
=O\!\left(\sqrt{M/\kappa}\right)
\quad (\kappa/M\to\infty).
\]
So the displayed logarithm is not \(\log(\Delta/\epsilon)\) uniformly when that construction is scaled to fit \(D_0\le\Delta\). Dividing the call counts by \(n\) on a sequential simulation does not change this fact. Our accounting also explicitly covers different adaptive schedules at the nodes, support-preserving shared prox, and the evaluator's averaged output.

**H-LB-hly and the related PIFO result.** Hannah–Liu–O'Connor–Yin, [arXiv:1805.07786v1](https://arxiv.org/abs/1805.07786v1), §2.1, Theorem 2 (“Lower complexity bound of Prox-SVRG and SARAH”) and Appendix Theorem 4 (“Lower complexity bound oblivious p-CLI algorithms”), use smooth strongly convex components and oblivious p-CLI rules. The printed order is \(\widetilde\Omega(M+(M/[1+(\log(M/\kappa))_+]+\sqrt{M\kappa})\log(1/\epsilon))\); lower-accuracy terms suppressed there are not assumed uniform in \(\kappa,M\). Han–Xie–Zhang, [arXiv:2103.08280v5](https://arxiv.org/abs/2103.08280v5), also [JMLR 25(2), 2024](https://jmlr.org/papers/v25/21-0264.html), §6 and Theorem E.6 (“Smooth Cases”, source label `thm:strongly`), gives \(\Omega((M+\sqrt{M\kappa})\log(1/\epsilon))\) for \(\kappa=\Omega(M)\) and a log-denominator bound when \(\kappa=O(M)\). This theorem assumes \(\kappa\ge2\), uses a domain radius \(R_0\) and \(0<\epsilon\le LR_0^2/4\), and is for their PIFO class (linear span, draws from a fixed distribution, occasional full gradients); the component prox available there is stronger than the shared prox here. Theorem E.9 (“Average Smooth Case”, `thm:average:strongly`) has \(M^{3/4}\sqrt\kappa\) in the high-condition range under mean-squared smoothness. A simulation dividing the call bound by \(n\) must preserve the respective algorithm restrictions. Neither a span requirement nor an oblivious/fixed-distribution schedule follows from the fixed decentralized model.

**H-LB-pl, Peng–Tang–Jia.** [arXiv:2609.00045v2](https://arxiv.org/abs/2609.00045v2), Theorem 3.2 (“Tight PL complexity under both smoothness assumptions”) and the small-\(\kappa\) construction in §§6 and Appendix D, use an adaptive randomized IFO returning both value and gradient. For \(M\) above a universal threshold, \(0<\epsilon\le c_\epsilon\Delta\) and fixed failure tolerance, the bound for \(3\le\kappa\le\sqrt M\) is \(\Omega(M+M\log R/[1+\log(\sqrt M/\kappa)])\), with a globally strongly convex hard average. The fixed-budget probability criterion yields a corresponding expected-gap obstruction at constant-factor smaller accuracy (a constant-probability gap above the test threshold already lower-bounds the expectation). We do not claim an improvement in that low-\(\kappa\) region. Also, by nesting of the smoothness classes, the hard examples at \(\kappa\) near \(\sqrt N\) can be used at larger allowed \(L\), yielding the term of order \(m+m\log R\) after the registered \(n\)-query-per-iteration conversion; this does not dominate R1 in the strict-comparison sequence above. The paper's large-\(\kappa\) branch for PL functions does not supply hard instances with a convex average and strongly convex composite.

**H-LB-net, Scaman–Bach–Bubeck–Lee–Massoulié.** [arXiv:1702.08704v1](https://arxiv.org/abs/1702.08704v1), Theorems 1 and 2 and their immediate corollaries, concern smooth strongly convex distributed objectives, a linear-span black-box model, a local-output guarantee and a communication delay \(\tau\). Theorem 1 uses graph diameter \(D\) and a global condition number \(\kappa_g\), with a distance-normalized suboptimality factor \((1-4/\sqrt{\kappa_g})^{1+t/(1+D\tau)}\). Theorem 2 chooses a network of normalized gossip eigengap \(\gamma\) and uses a local condition number \(\kappa_l\), with factor \((1-16/\sqrt{\kappa_l})^{1+t/(1+\tau/(5\sqrt{\gamma}))}\). The benchmark registers the corresponding fixed-\(n\) network dependence \(\min\{n,(1-\rho)^{-1/2}\}\sqrt\kappa\log R\). We grant this comparison as registered, without identifying the original local-output theorem with an averaged-output theorem in general. At \(\rho=0\) the registered scale is \(\sqrt\kappa\log R\), smaller than R1 when \(m/n\) grows. R1 is not just a network lower bound under another name.

**H-LB-avg and individual versus average smoothness.** Zhou–Gu, [arXiv:1901.11224v1](https://arxiv.org/abs/1901.11224v1), Theorem 4.1, assume a \(\sigma\)-strongly convex smooth average and a mean-square gradient-difference bound \(L_{\rm ms}\); they prove a lower bound for linear-span randomized IFO of order
\(\Omega(M+M^{3/4}\sqrt{L_{\rm ms}/\sigma}\log(M\sigma\Delta/(L_{\rm ms}\epsilon)))\)
in their stated range \(\epsilon\le8\Delta M^{7/4}(\sigma/L_{\rm ms})^{3/2}\) (the log term is informative when positive). Their §4.3 and conclusion distinguish that assumption from individual smoothness. Dividing by \(n\) produces the registered \(m^{3/4}n^{-1/4}\sqrt{\kappa}\) coefficient only if \(\kappa\) refers to the mean-square parameter, not an individual bound on their instances. Zhou, [arXiv:2611.05807v1](https://arxiv.org/abs/2611.05807v1), §6 “Near-Optimal Complexity” (discussion following its high-accuracy characterization), likewise identifies the gap for strongly convex finite-sum linear systems with individually bounded indefinite terms. Its own main high-accuracy theorem concerns **nonconvex stationarity with Hessian–vector access**, not this objective-gap problem. R1 improves the standard normalized data bound without resolving the open polynomial gap.

**H-UB counting and the scope of R2.** In Liu–Low's notation \(m_{\rm paper}=n,\ n_{\rm paper}=m\); their assumptions preceding Theorem 5.1 set \(\sigma=\sigma_f+\sigma_\psi>0\) with both curvature contributions nonnegative. They use a two-sided quadratic Taylor bound, so an individually \(L\)-Lipschitz gradient provides \(\ell_1=\ell_2=L\). Theorem 5.1 bounds \(\mathbb E[F(\bar y^K)-F(x^\star)]\) by \(3(F(\bar y^0)-F(x^\star))(1+\tau/2)^{-K}\) in **outer** epochs. With \(b\asymp\sqrt m\), the remark and Corollary 5.3 count \(O(m)\) SFO calls per node in expectation per epoch, including full local gradients and an inner geometric loop of mean \(m/b\) with batches of size \(b\). A repeated gradient at the same component at two points takes two calls, a constant factor. The count also includes a constant number of FastMix invocations per inner step, each of accelerated communication length; constant-per-step prox calls are absorbed in the query count. Serializing batches and communication gives the sum of the registered data and network orders, not their product. Lazifying \(W\) with \((I+W)/2\) costs one actual \(W\)-round per lazy round and gives \(\lambda_2\le(1+\rho)/2\). Algorithm 1 writes a common full-gradient tracker at initialization, and appendix tuning uses consensus and curvature parameters. The paper's work counts in expectation over the geometric epoch lengths are not by themselves a deterministic wall-clock horizon. We compare the **registered order**, taking it as a target rather than invoking the paper as a theorem of the fixed model with free initialization or additional advice. Our counterexample needs no issue about arbitrary initial disagreement or the spectrum: \(W=J\) throughout the separation.

Smoothness of \(f_i\) alone is strictly weaker than a mean-square bound on the \(F_i(\cdot;\xi)\). In R2 at \(\theta=m\), the individual Hessian scale can be order \(mL\) for vectors supported on one block, while summing gradients to a local average scales that block by \(1/m\). A mean-square bound takes its norm **before** averaging the squares and does not permit this scaling at the same \(L\). Thus H-LB-avg or the PIFO mean-squared result is not an impossibility proof at the scale (6). Conversely, R2 does not rule out retaining H-UB under a bound on the second moment or another weaker assumption that still controls individual gradient differences sufficiently; it proves specifically that **only** local-average smoothness is insufficient.

**Summary.** Relative to the registered translations, R1 is strictly stronger than H-LB-data on the data row. R2 proves the necessity of retaining more than local-average smoothness for the H-UB data order. R3 gives matching complexity in the two-node relaxation and in the \(m=1\) two-node standard problem, and improves the communication count for two nodes at the stated data cost. R4 establishes the obstruction at the disconnected endpoint. The standard problem with unrestricted \(n,m\) is not closed by these results.
