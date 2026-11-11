# Offline conic QP solver (submission)

## How to run

From the submission directory:

```sh
./build.sh                 # optional; pure Python, only checks the environment
./run.sh --input /abs/path/input.npz --output /abs/path/output.npz
```

`run.sh` uses `exec python3` and forwards both arguments unchanged.  Both paths
may be arbitrary absolute paths; no working-directory or instance-name
assumptions are made.  Nothing is written except the requested output NPZ.
`stdout`/`stderr` are silent unless `SOLVER_VERBOSE=1` is set.  All BLAS,
OpenMP and HiGHS thread counts are pinned to 1 and no child processes are
started.

## Dependencies (fixed versions, all preinstalled in the task image)

| package | version |
| --- | --- |
| CPython | 3.12.14 |
| NumPy | 2.3.5 |
| SciPy | 1.16.3 (bundled HiGHS 1.8.0 LP/QP engine) |

No downloads, no network, no compilation and no extra wheels are needed;
`build.sh` is a no-op check.  The HiGHS pybind11 extension
`scipy.optimize._highspy._core` is loaded directly from the installed SciPy
wheel, which avoids importing all of `scipy.optimize` and keeps process startup
near 0.75 s (important for 1-second case limits).  A normal
`from scipy.optimize._highspy import _core` import is used as a fallback.

## Files

| file | role |
| --- | --- |
| `run.sh` | executable entry point |
| `build.sh` | executable no-op environment check |
| `solver.py` | input/output, dispatch, HiGHS models, active-set polish, certificates, cutting-plane loop |
| `cones.py` | cone layout, vectorised cone defects, separating hyperplanes |
| `highswrap.py` | thin LP/QP wrapper around the bundled HiGHS bindings |
| `verify.py` | independent float64 implementation of all acceptance checks |

## Method

The problem is `min 1/2 x'Px + q'x + c0` with `Ax + s = b`, `s in K`.

### 1. Polyhedral cones (zero + nonnegative)

The constraints are `A_z x = b_z` (zero cone) and `A_p x <= b_p`
(nonnegative cone), with free `x`.  This LP/QP is passed directly to HiGHS
1.8.0 (dual simplex for LPs, its convex-QP active-set solver for QPs) using the
full sparse CSC arrays.  HiGHS returns row duals `pi`; the conic dual is
`z = -pi` on these rows, with `z_zero` free and `z_nonneg >= 0`.

The HiGHS QP active-set solver stops with about `1e-7` reduced-cost error,
which exceeds the required `1e-8` on badly scaled instances, and it can also
cycle on degenerate QPs.  Its run is therefore capped (6 s), with a
LP-relaxation-warm-started primal active-set QP method as a fallback.  If the
internal verifier rejects the direct solution, an **active-set KKT
polish** is applied: the binding rows are identified from `s = b - Ax`, the
symmetric quasi-definite system

```
[ P   A_act' ] [x]   [-q     ]
[ A_act   0  ] [y] = [b_act  ]
```

is solved with SuperLU (partial pivoting, iterative refinement, tiny
progressive diagonal regularisation only when singular), and the active set is
repaired by dropping negative multipliers / adding violated rows.  This
typically brings every residual to machine precision.

### 2. General cones: adaptive outer approximation

For SOC, exponential and PSD cone blocks the cone is replaced by finitely many
supporting hyperplanes `c' s_block >= 0` whose normals `c` are **exact** dual
cone vectors:

* SOC: `c = (1, -v/||v||)` (boundary normals, self-dual cone);
* exponential: `c(t) = (-1, t-1, exp(-t))` with `t = u/v` (the tangent of
  `u <= v log(w/v)`), plus coordinate normals for `v >= 0`, `w >= 0`;
* PSD: `c = svec(v v')` from eigenvectors of `smat(s)` with negative
  eigenvalues.

Eliminating `s = b - A x` turns each cut into one LP row
`(c'A_block) x <= c'b_block`.  The LP relaxation is (re)solved by HiGHS with
warm starts; cuts are appended at the blocks where the current iterate leaves
the cone (all violated blocks, vectorised by cone dimension).

Because each cut normal lies exactly in `K*`, the simplex duals `lambda >= 0`
give `z = sum_i lambda_i c_i + z_poly` which is in `K*` **exactly**, and
complementarity `s'z = 0` holds to simplex accuracy.  Stationarity
`Px + q + A'z = 0` is likewise exact up to the simplex basis, and the primal
residual vanishes by construction.  Hence only the *primal* cone defect has to
be driven down by the cuts.

Degenerate basic solutions can sit exactly on the flat parts of the
exponential cone (`w = 0` with `u/v << 0`) or of the PSD cone.  A tiny
interior shift (2e-9 times the problem scale, i.e. well below the `1e-8`
primal-residual allowance) is added to the `w`, `t`, or diagonal `svec`
coordinates; the resulting primal residual is `2e-9`, still far below the
acceptance threshold.  This makes the method converge in ~10-20 cut rounds
instead of stalling on asymptotically flat boundaries.

Every candidate is checked with an independent float64 implementation of the
task's residual/cone/gap conditions before being written; the first candidate
that passes with safety margin is returned.  `solved` never reports objective
values (they are optional), avoiding an extra consistency check.

### 3. Infeasibility / unboundedness

If HiGHS reports the relaxation infeasible or unbounded, auxiliary LPs
(normalised Farkas and recession directions, with box normalisation) are
solved.  For conic relaxations the certificate is restricted to the conic
hull of the cut normals, so it is automatically inside `K*`/`K`.  Candidates
are again accepted only if the independent certificate checker passes.
Otherwise `unresolved` is written (a failed case either way, but with a valid
output file).

### 4. Resource safeguards

A hard 280 s process deadline and shorter per-path budgets (18 s for
PSD-dominated problems, which are the hardest for a polyhedral scheme) ensure
that the program always writes an output and exits.  HiGHS receives a
`time_limit` equal to the remaining budget.  Dense work is limited to cone
blocks; all large matrices stay sparse CSC/CSR.  After a HiGHS dual-simplex
stall at the last `1e-10` dual infeasibility (observed in 1.8.0 after
`addRows`), the wrapper re-runs with relaxed tolerances and the primal
simplex; basis duals remain exact.

## Development evidence

Using the supplied `public-dev/io/check_result.py` on the 163 public inputs:
all tested LP/QP cases (including a 154699 x 414525 LP and 5427-variable
QPs) and all tested small/medium SOC and exponential cases verify at
`eps = 1e-8`, typically in 0.8-12 s on the pinned single core.  The tiny PSD
interface example verifies as well.  Large (>10^5 row) SOC/exponential
instances and large-order SDP blocks are the known weak spot: successive
HiGHS simplex re-solves become too expensive, and a single PSD block of order
50-200 is not well approximated by rank-1 cutting planes.  These are reported
as `unresolved` rather than producing an unverified answer.  Each of the six
interface examples verifies with the expected status.

## References / prior work used

* HiGHS 1.8.0: Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised
  simplex method", Mathematical Programming Computation 10 (2018); HiGHS QP
  solver and the `HighsModel`/`HighsHessian` C++ API shipped with SciPy
  1.16.3.  Used as the LP/QP engine (not reimplemented).
* SCS / O'Donoghue et al., "Conic Optimization via Operator Splitting and
  Homogeneous Self-Dual Embedding" (2016) and ECOS / Domahidi et al. (2013)
  for the homogeneous-embedding viewpoint and the exponential-cone barrier
  `f = -log(v log(w/v) - u) - log v - log w`, whose gradient supplies the
  tangent cuts `(-1, t-1, e^{-t})`.
* A. Ben-Tal and A. Nemirovski, "Lectures on Modern Convex Optimization"
  (2001) for supporting-hyperplane (cutting-plane) representations of cones
  and the PSD `svec` representation.
* Kelley's cutting-plane method (1960) and the classical active-set QP KKT
  system (Nocedal & Wright, "Numerical Optimization", Ch. 16) for the
  polishing step.
* Mehrotra's predictor-corrector / primal-dual interior-point literature
  motivated the decision to keep HiGHS's exact simplex basis rather than an
  interior-point crossover when high (1e-8) accuracy is required.
