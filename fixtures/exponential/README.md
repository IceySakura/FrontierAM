# Sparse exponential propagator (FAMEXP2)

Computes `y = exp(-tL) b0 + t*phi1(-tL) b1` for symmetric PSD sparse `L`.

## Algorithm
- Reads the binary protocol directly (`solve.py`) and prints `READY` only after imports.
- Uses a separate Lanczos process for `b0` with `exp(-t x)` and for `b1` with `phi1(t x) = (1-exp(-t x))/(t x)`.
- Applies full reorthogonalization twice per step. The projected tridiagonal function is computed with `scipy.linalg.eigh_tridiagonal`.
- Stops when the change between evaluations six steps apart and the a posteriori residual estimate `beta_m*|e_m^T f(T)e_1|*||b||` are both below `0.01*(atol+rtol*||y||_inf)/2`. This check is applied only after the Chebyshev-type minimum dimension `sqrt(1.25*ln(10)*t*||L||_inf)` is reached, which prevents false early convergence for stiff cases.
- If the required dimension exceeds the memory-based basis cap, the time interval is split recursively: `y(t) = y_{t/2}(y_{t/2}(b0), b1)`.
- Handles happy breakdown exactly. `t=0` and an empty matrix are handled directly.

## Dependencies
Python 3, NumPy and SciPy. These are preinstalled. No build step is required.
`run.sh` runs `python3 -u solve.py` with single-threaded BLAS.

## Public checks
- `tools/run_local.py --repeats 3` passes all 87 runs. The largest scaled error is around 2e-3 of the tolerance.
- Synthetic stress tests compare against `expm_multiply` on an augmented matrix. They cover a 2D Laplacian with n=3249 and t||L||=2200, a diagonal potential, signed congruence, and n=14400 with t||L||=1e5. All three precisions pass.
