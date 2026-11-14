# Conic solver (submission 0.3.0)

Solves `min 1/2 x'Px + q'x + c0` s.t. `Ax + s = b`, `s in K` for
K = product of zero / nonnegative / second-order / exponential / PSD cones,
and returns `solved`, `primal_infeasible`, `dual_infeasible` or `unresolved`.

## Run

```sh
./build.sh                                    # once, compiles libcore.so
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` execs `python3 ./solve.py` from the submission directory; paths are
taken from the arguments only. `solve.py` reads the NPZ, calls the compiled C
core through `ctypes`, and writes the result NPZ.

## Dependencies (preinstalled, no downloads)

* Python 3.12.14, NumPy 2.3.5 (NPZ I/O and marshalling only)
* SciPy 1.16.3 — used *only* for the exact LP path (`scipy.optimize.linprog`,
  HiGHS) when `P` is empty and all cone blocks are zero/nonnegative
* gcc 12.2.0 (build), libm (runtime)

## Method

Main engine (`core.c`, single threaded, no external numeric library):

* ADMM on the conic form with a proximal-regularised x-step. The x-step is the
  SPD system `M x = sigma x^k - q + A'(rho (b - s^k) - z^k)` with
  `M = P + sigma I + rho A'A`, solved by warm-started preconditioned CG with a
  Jacobi preconditioner (`diag|P| + sigma + rho diag(A'A)`). `M` never changes
  structure, so `rho` can be adapted (residual balancing) with no
  refactorisation. Over-relaxation on the x-step.
* Because `s` is obtained by an exact projection onto `K` and the multiplier
  update is equivalent to a projection onto `K*` (Moreau decomposition),
  `s in K`, `z in K*` and `s'z = 0` hold exactly at every iterate; only the
  primal residual `Ax+s-b` and dual residual `Px+q+A'z` have to be driven down.
* Cone projections: zero, nonnegative, second-order (closed form), PSD
  (cyclic Jacobi eigendecomposition, eigenvalue clamping, reconstruction in the
  prescribed upper-triangular-by-column / sqrt(2) packing), exponential
  (candidate set = the point itself, the `(u<=0, 0, w>=0)` face and roots of the
  smooth boundary condition, each root found by a bracketing scan plus
  bisection on the scalar stationarity equation; the nearest candidate wins).
* Statuses are only reported after re-verifying, in C, the exact acceptance
  conditions of the task (relative primal/dual residuals, primal and dual cone
  defects, duality gap; Farkas-type conditions for the two infeasibility
  certificates, using the iterate `z` and the iterate differences as candidate
  certificates). Anything that does not verify is reported as `unresolved`,
  so a wrong status is never emitted.

LP fast path (`solve.py`): when `P` has no nonzeros and every block is zero or
nonnegative, the instance is an LP and is handed to HiGHS through
`scipy.optimize.linprog`; the primal solution and the (sign-corrected)
constraint marginals give `x`, `s = b - Ax` and `z`, and the same acceptance
conditions are re-checked in NumPy before `solved` is reported. This avoids
iterative convergence entirely for the linear family.

## References

* Stellato et al., *OSQP: An operator splitting solver for quadratic programs*
  (ADMM with a fixed KKT matrix, over-relaxation, residual balancing).
* O'Donoghue et al., *SCS: Splitting conic solver* (homogeneous ADMM for cones,
  adaptive/indirect linear solves, exponential-cone projection).
* Boyd et al., *Distributed Optimization and Statistical Learning via ADMM*.
* Lu/Freund-style projected Newton treatment of the 3-D exponential cone;
  cyclic Jacobi for the symmetric eigenproblem used by the PSD projection.
* SciPy `linprog` (HiGHS) for the exact LP path.

## Notes / limits

* Single process, single thread; no temporary files; memory is O(n + m + nnz).
* The internal wall-clock budget defaults to 250 s (`SOLVER_BUDGET` overrides).
* Non-finite iterates are scrubbed before writing so the output NPZ always
  contains finite arrays and a valid `metadata_json`.
