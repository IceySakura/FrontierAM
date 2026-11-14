# Offline conic QP solver

## Running

```sh
./build.sh          # optional; copies the bundled HiGHS extension locally
./run.sh --input /abs/path/input.npz --output /abs/path/result.npz
```

`run.sh` is an executable POSIX shell script that `exec`s `solver.py` with the
same arguments.  It writes an NPZ file containing a scalar Unicode
`metadata_json` (`{"schema_version":1,"status":...}`) and the vectors required
by the status (`x,s,z` for `solved`, `z` for `primal_infeasible`, `x,s` for
`dual_infeasible`).  Results are written atomically (temporary file +
`os.replace`) and a watchdog thread writes a legal `unresolved` result early,
so a process killed by a short per-case deadline still leaves a well-formed
file.  Only one thread and one process are used; BLAS is forced single
threaded.

## Dependencies (all offline, no network, no downloads)

* Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3 (preinstalled; `resources.yaml`).
  SciPy is imported lazily, only for sparse large-QP KKT solves.
* HiGHS 1.8.0, through SciPy's bundled pybind11 extension
  `scipy/optimize/_highspy/_core*.so`.  `cuts.py` / `lpqp.py` load that `.so`
  directly by path (module name `_core`) so that importing all of SciPy
  (≈0.3 s) is avoided on ordinary cases; `build.sh` also copies it next to
  the sources.  A copy is included in the submission.
* gcc/g++ 12.2 are not needed: no native code is compiled.

Optional files `test_lpqp.py`, `test_cones.py`, `tests/` are development
self-checks and are not used at run time.

## Method

The problem is `min .5 x'Px + q'x + c0` s.t. `A x + s = b`, `s ∈ K`, with
`K` a product of zero, nonnegative, second-order, PSD and exponential cones.
Several complementary engines are used; every candidate is verified in the
ORIGINAL coordinates with a local reimplementation of the independent
checker's conditions (relative primal/dual residuals, normalised cone
defects, duality gap) before being returned.

### 1. Cutting-plane engine (`cuts.py`) — primary

For any `g ∈ K*` the inequality `g's ≥ 0` is valid on `K`, hence

```
        (g' A_block) x <= g' b_block
```

is a valid linear outer approximation of the block constraint.  HiGHS solves
the resulting LP/QP exactly; its row duals are nonnegative multiples of the
cut normals, so the reconstructed `z = Σ (-row_dual_j) g_j` lies in `K*`,
stationarity holds exactly and complementarity is exact (LP complementarity
gives `λ_j (g_j's) = 0`).  Cuts tangent at the most violated point are added
until `s = b - A x` lies in `K` to the required `1e-8` accuracy:

* SOC: `g = (1, -v/||v||)`;
* PSD: `g = svec(u u')`, `u` eigenvectors of the negative eigenvalues;
* exponential: `g(τ) = (-1, log τ - 1, 1/τ)` (Fenchel cuts for `exp(u/v)`).

A fresh HiGHS model is built at every outer iteration (reusing a simplex
basis across cuts loses about one decimal near the cone boundary, which is
decisive at `1e-8`).  Large models use a sparse CSC/CSR assembly and never
form a dense LP matrix.  Infeasible/unbounded relaxations produce Farkas /
recession certificates from small auxiliary LPs, validated against the
checker's scaled certificate rules.

### 2. QP accuracy and speed

HiGHS' general QP solver is accurate to only ≈1e-8 and can be slow, so two
active-set KKT refinements are used instead:

* small/medium convex QPs: an iterative primal active-set method starting
  from the fast LP-relaxation basis (and, when the LP is unbounded, from the
  equality-constrained QP optimum).  Each step solves
  `[P A_act'; A_act 0][x;λ] = [-q;b_act]`; violated inequalities are added,
  rows with negative multipliers are dropped, and the exact KKT point is
  verified.
* large sparse QPs: the same method with `scipy.sparse.linalg.spsolve` on a
  quasi-definitely regularised sparse KKT matrix.

A dense active-set polish is also applied to conic QP models from the
cutting-plane LP/QP duals.

### 3. Interior-point fallback (`conic.py`, `cones.py`)

A primal-dual IPM with a damped Newton step on

```
A x + s = b ,     P x + q + A' z = 0 ,     z + μ ∇F(s) = 0 ,
```

where `F` is a logarithmically homogeneous self-concordant barrier
(`-Σlog s_i`, `-log(t²-‖v‖²)`, `-log det S`, and
`-log(v log(w/v)-u)-log v-log w`).  Only the barrier gradient and
Hessian-vector product are needed; `cones.py` supplies closed forms for all
five families, exact fraction-to-boundary steps and checker-compatible
cone-defect functions.  Equality rows carry free multipliers.  Data are
equilibrated (Ruiz-style, block-uniform row factors) and the objective is
scaled; every candidate is validated in the original coordinates.

### 4. LP/QP HiGHS path (`lpqp.py`) — final fallback

Direct `HighsLp`/`HighsModel` models for zero/nonnegative cones, tight
tolerances, active-set KKT polishing, Farkas/recession certificate LPs and
time-budget guards.  This is used when the cutting-plane and IPM paths both
fail.

## References / prior art

* S. Wright, *Primal-Dual Interior-Point Methods*, SIAM 1997;
  Y. Ye, M. J. Todd, S. Mizuno, homogeneous self-dual embedding (1994);
  S. Mehrotra, predictor-corrector methods (1992).
* A. Domahidi et al., *ECOS: An SOCP solver for embedded systems*, ECC 2013;
  P. Goulart, *Clarabel*; ECOS/Clarabel barrier treatment of exponential
  cones (`F = -log(v log(w/v)-u) - log v - log w`) and PSD `svec` encoding.
* J. E. Kelley, *The cutting-plane method for solving convex programs*,
  J. SIAM 1960; bundle/proximal stabilization ideas.
* CVXPY cone canonicalisation conventions (`P,q,A,b,cones`; upper-triangle
  column-wise PSD packing with √2 off-diagonals), per `INSTRUCTION.md`.
* Q. Huangfu and J. A. J. Hall, HiGHS (Math. Prog. Comp. 2018); HiGHS 1.8.0
  C++ library shipped inside SciPy 1.16.3.

## Experimental basis

Local checks with `samples/io/check_result.py`:

* all six interface examples (mixed QP, SOC, exponential, PSD, primal- and
  dual-infeasible certificates) verify at `1e-8`;
* all five public dev instances verify; conic instances typically take
  0.3–1.0 s including interpreter startup on the development core;
* `test_cones.py` checks barrier gradients/Hessians against finite
  differences (`<1e-9`), dual interiority of `-∇F(s)`, and exact agreement
  of cone defects with the supplied checker over random vectors;
* `tests/run_all.py` runs the full entry point over 105 synthetic instances
  of all five cone families (feasible, primal/dual infeasible, ill-scaled,
  degenerate, up to n=m=3000); the final code passes 75+ of them;
* a 100 000-variable sparse LP verifies in ≈0.6 s and a 100 000-variable
  sparse QP in ≈1.1 s.

All results are computed from the given input only; no instance names,
precomputed answers, network access or cross-run state are used.
