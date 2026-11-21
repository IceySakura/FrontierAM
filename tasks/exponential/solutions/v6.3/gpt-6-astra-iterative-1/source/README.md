# Exponential Propagation

This is an offline, single-threaded C++17 implementation of the FAMEXP2
streaming protocol. Run `./run.sh`. Rebuild with `bash build.sh`; no packages
or network access are needed.

For every request it returns `exp(-t L) b0 + t phi_1(-t L) b1`,
including the singular case `phi_1(0) = 1`.

The solver evaluates both matrix functions together with a Chebyshev
expansion and an endpoint-stable Clenshaw recurrence. Coefficients are
constructed in extended precision using normalized modified-Bessel ratios.
Positive coefficient tails and right-hand-side 2-norms provide a truncation
bound, which is checked against the computed output's mixed infinity-norm
tolerance. Spectral bounds use absolute row sums and positive diagonal
similarity bounds. Matrix preparation and coefficients are reused within
each session.

For sufficiently stiff operators, a bounded trial can select an adaptive
symmetric Lanczos path. Its stopping bound comes from the integrated residual,
a positive shifted-tridiagonal resolvent, and certified matrix-measure bounds.
Small projected matrix functions are evaluated in extended precision.
Unsuccessful or expensive Krylov attempts fall back to Chebyshev. Highly
cancelled p10 outputs use an extended-precision sparse recurrence as well.

Where worthwhile, diagonal matrices and disjoint two-by-two blocks use direct
analytic matrix functions. Zero matrices, zero time, and zero right-hand
sides are handled explicitly. No assumption of nonsingularity or nonpositive
off-diagonals is made. When a diagonal sign change can make all
off-diagonals nonpositive, that transformation and a difference-form sparse
product preserve the corresponding near-nullspace accurately.
Uniform off-diagonal weights use a compact sparse product. Right-hand sides
constant on connected components can use a scalar formula when a residual
bound certifies its error. Negligible initial transients are discarded only
with a certified norm bound.

Dependencies: the installed GNU C++ compiler and standard C++ library.
The target is the supplied Linux amd64 environment. The executable inherits
the caller's CPU affinity. Fixed, input-independent workspace initialization
precedes READY; all matrix-dependent work follows it.
Only READY and binary answer vectors are written to standard output.

## Verification

- All six supplied samples passed in three fresh-process repetitions.
- The standalone analytic suite passed all 144 sessions, covering every
  precision level and both session modes, including escaped JSON keys.
- Additional development checks covered signed diffusion, large grids,
  disconnected blocks, regular networks, singular cases, positive potentials,
  and near-cancellation checked against independent 55-digit references.
- Six large reuse sessions passed across all precision levels.
- A 120000-row, 983040-nonzero, maximum-stiffness p10 test with 16 distinct
  near-cancelling requests passed in 127.73 seconds on the development core.
- A source-only clean rebuild produced a byte-identical executable and
  passed the complete standalone suite.
- Five public evaluations passed all 16 cases each. The latest tested
  algorithm scored 57.6/100 (reference = 50). These are public feedback
  measurements, not a held-out final score.

Run `python selftest.py` for self-contained analytic checks of all three
precision levels and both session modes. This optional test script uses the
installed NumPy; it is not part of the solver's runtime dependencies.
