# Native conic interior-point solver

Run: `./build.sh` once (compiles `src/*.c` with gcc into `bin/conic_solver`; falls back to the
bundled `bin/conic_solver.prebuilt` if compilation fails), then
`./run.sh --input IN.npz --output OUT.npz`.

Dependencies (all preinstalled in the image, nothing downloaded): gcc 12, system `libz.so.1`
(NPZ inflate), reference `liblapack.so.3`/`libblas.so.3` (small dense PSD kernels), libm.
No Python at run time: Python start-up plus numpy/scipy imports alone cost ~0.6 s here, which
is most of the typical 1 s limit, so the solver is a single C binary.

## Method
* Homogeneous self-dual embedding with quadratic objective, as in Clarabel
  (Goulart & Chen, "Clarabel: An interior-point solver for conic programs with quadratic
  objectives", 2024); Mehrotra predictor-corrector, step fraction 0.99.
* Ruiz equilibration of [P A'; A 0] (cone blocks get a common row scale) plus cost scaling.
* KKT system [P+dI, A'; A, -(H+dI)] factored by a sparse LDL' (QDLDL-style, AMD ordering
  ported from CSparse cs_amd), static (1e-8) and dynamic (Clarabel-style) regularization,
  iterative refinement against the unregularized matrix.
* Small/dense KKT systems (N <= 6000 whose AMD factor is mostly one dense block, e.g. SDPs)
  use a blocked dense LDL' whose trailing updates call OpenBLAS `dgemm` from the scipy wheel
  already in the image (`dlopen`, single thread); otherwise / if unavailable the sparse LDL'.
  If a factorization breaks down, the static regularization is raised (x100, up to 3 times).
* Cones: zero, nonnegative, second-order (NT scaling; cones of dim <= 24 enter the KKT in the
  CVXOPT-style scaled form [W^-1 A_b, -I] so that W^2 never appears; larger cones use a
  quasi-definite sparse rank-2 expansion), PSD (NT scaling via Cholesky+SVD, dense
  symmetric-Kronecker block),
  exponential (dual-barrier scaling H = mu * hess f*(z), backtracking + neighborhood check;
  central initial point from Clarabel).
* For SOC rows the slack step is taken from the linearized primal equation
  (ds = -eta*r_z + b*dtau - A*dx) instead of ds = -d_s - H dz, which avoids amplifying KKT
  solve errors by ||W||^2 near the boundary (this fixed most high-accuracy SOC failures).
* Termination re-evaluates exactly the independent checker's criteria (residuals, cone
  defects, gap, infeasibility certificates with the checker's normalisation) in the original
  coordinates every iteration, so a returned status is one the checker accepts.

## References
Clarabel (Goulart & Chen 2024); ECOS (Domahidi, Chu, Boyd 2013); CVXOPT cone programming
notes (Vandenberghe 2010) for NT scaling formulas; Dahl & Andersen (2022) and Serrano (2015)
for exponential-cone IPMs; Davis, "Direct Methods for Sparse Linear Systems" (cs_amd, LDL).

## Experimental basis
Developed against the 5 public samples, the 6 interface examples and ~140 generated instances
(LP/QP/SOCP/exp/SDP, feasible, primal- and dual-infeasible; tests/ in the workspace), checked
with the official `check_result.py`. Key measured effects: Python start-up+imports 0.6 s vs
~5 ms for the C binary; residual-form SOC slack step 12/22 -> 17/22 generated SOCPs verified;
exp-cone neighbourhood fallback 1/17 -> 12/17; dense OpenBLAS KKT path ~10x faster on a
theta-type SDP (50x50 PSD block, 1.2 s total).
