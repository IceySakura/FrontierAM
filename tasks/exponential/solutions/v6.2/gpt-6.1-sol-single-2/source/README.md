# Sparse Exponential Propagation

Run `./run.sh` from this directory. No build step, network access, package
installation, agent tools, or input files are required.

## Dependencies

Python 3, NumPy, and SciPy, using the versions already installed in the
evaluation environment. The launcher constrains numerical libraries to one
thread and retains the inherited CPU affinity.

## Algorithm

The program implements the FAMEXP2 binary streaming interface. All generic
imports happen before `READY`; all input-dependent preparation happens after
it. Matrix preparation and polynomial coefficients are reused across requests.

Zero-time problems return the initial vector. Diagonal operators use scalar
exponentials and cancellation-resistant `expm1` integral factors. Operators
with at most 64 rows use a symmetric dense eigendecomposition.

For larger operators, constant forcing is folded into the initial condition,
giving `exp(-t L) (b0 + t b1)`. The exponential is evaluated by a Chebyshev
expansion on the Gershgorin spectral interval. Exponentially scaled modified
Bessel functions supply the coefficients without overflow. Their tail is
truncated below a precision-dependent threshold, and the three-term recurrence
uses sparse CSR matrix-vector products.

Only the current and previous recurrence vectors are retained, so memory is
linear in the matrix and vector sizes. Output consists solely of `READY` and
the requested raw little-endian float64 vectors.

## Public Checks

All eight public analytic controls passed using `tools.run_local.run_session`,
including the 16-request control. Their largest measured scaled error was
`4.51e-6`.

Constructed 20,000-row tridiagonal diffusion tests with zero forcing passed at
the strictest precision for `t * ||L||_inf` equal to `0.01`, `10`, and `100000`.
The stiffest test used degree 1926 and had maximum absolute error `1.57e-13`.
These are development measurements, not calibrated benchmark scores.
