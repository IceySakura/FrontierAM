# Conic QP interior-point solver (Python + NumPy, LAPACK/SuperLU kernels)

## Run

```sh
./build.sh                      # optional: precompiles bytecode (python3 -m compileall)
./run.sh --input IN.npz --output OUT.npz
```

`run.sh` execs `python3 main.py`, which imports `solver.py`. Single-threaded BLAS is
forced (`OPENBLAS_NUM_THREADS=1`).

## Dependencies (preinstalled, pinned by the image)

* Python 3.12.14, NumPy 2.3.5
* SciPy 1.16.3 — only its compiled `_superlu` (sparse LU) and `_flapack` (LAPACK) extension
  modules are loaded directly by file path. Importing the `scipy` package costs ~0.45 s,
  which is too much with 1 s limits; loading the two extensions costs a few milliseconds.

No other packages, no network.

## Method

Homogeneous self-dual embedding primal-dual interior-point method, following the
Clarabel design (Goulart & Chen, "Clarabel: An interior-point solver for conic programs
with quadratic objectives", 2024), which in turn builds on CVXOPT (Vandenberghe 2010) and
ECOS (Domahidi et al. 2013):

* Variables `(x, z, s, tau, kappa)`; the quadratic term enters the tau-row as in Clarabel,
  so QPs are solved directly without epigraph reformulation.
* Mehrotra predictor-corrector with `sigma = (1 - alpha_aff)^3`, step fraction 0.99.
* Nesterov-Todd scaling for nonnegative, second-order and PSD cones (PSD via Cholesky +
  SVD, CVXOPT style), with second-order (Mehrotra) correction in the Jordan algebra.
* Exponential cone: dual-barrier scaling `H = mu * hess f*(z)` (Nesterov-Todd-Ye /
  Skajaa-Ye style) with an interior-point backtracking line search and a wide
  neighbourhood test (`s_i'z_i >= 0.3 mu` per cone); if the step is short, the direction
  is recomputed with stronger centring.
* Ruiz equilibration of the KKT matrix (cone-block-uniform row scaling for SOC/EXP/PSD)
  and cost scaling.
* Linear algebra: quasi-definite KKT `[[P+dI, A'], [A, -(H+dI)]]` with static
  regularisation `d = 1e-8`, factored by SuperLU (MMD on A'+A, symmetric mode, no
  pivoting; threshold pivoting as fallback) and polished by iterative refinement against
  the unregularised matrix. The fill-reducing ordering is computed once and reused
  (matrix symmetrically pre-permuted, NATURAL ordering afterwards): ~25% faster.
  If the KKT has very dense columns, COLAMD is tried once and kept if faster.
* Second-order cones with dimension > 48 use a sparse expansion of the NT scaling
  (`H = eta^2 (I + 2 w w' - 2 e1 e1')`, two extra KKT rows per cone) instead of a dense
  block (a 3000-dim cone went from 20 s to 1.2 s).
* PSD problems use a dense reduced system: for `P = 0` without equality rows it is the
  QR factor of `[W^{-T} A; sqrt(d) I]` (avoids squaring the condition number of the
  Schur complement), otherwise an LU of the reduced system; once iterative refinement
  stops converging (ill-conditioning near low-rank optima) it switches to a dense LU of
  the full KKT system. Operations on equal-size PSD blocks are batched.
* Termination is evaluated every iteration with exactly the independent checker's
  formulas on the *original* data (relative primal/dual residuals and gap <= 0.5e-8;
  infeasibility certificates with the normalised Farkas conditions), so only verifiable
  results are returned. Otherwise `unresolved` is written.

## Experimental basis

Developed on the 5 public development instances, 6 interface examples and ~45 generated
instances (LP, QP, portfolio QP, redundant equalities, transport LP with 20k variables,
SOCP with many small / one 3000-dim cone, logistic regression and entropy problems with
exponential cones, SDPs in dual and primal form with one or 200 blocks, unconstrained QP,
primal/dual infeasible LP/SOCP/EXP/SDP). All verify with the public checker at
`epsilon = 1e-8`. Measured process time is about 0.3 s for small instances (most of it is
the NumPy import); the dev PSD instance takes ~1.1 s on the (shared) development core.
Known weakness: random sparse matrices with catastrophic fill and very large
problems are slow, since everything outside SuperLU/LAPACK runs in Python.
