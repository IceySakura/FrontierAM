# Sparse exponential propagation

The program implements the FAMEXP2 binary streaming protocol and returns
`exp(-t L) b0 + t phi1(-t L) b1` for every sequential request.

## Build and run

Run `bash build.sh` to reproduce the executable, then `./run.sh`. Standard
input/output use the binary interface, not a text command-line interface.
The run script also builds if the executable is absent. No network access,
package installation, task files, or agent tools are needed by the program.

Dependencies: the installed GNU C++ compiler, Eigen headers, LAPACKE, OpenBLAS,
and FFTW's double-precision runtime on Linux amd64. BLAS is restricted to one
thread; inherited CPU affinity is unchanged. The build targets x86-64-v3,
available on the portable-v1 host. No runtime JIT compilation is used.

## Algorithm

Zero time and zero matrices are handled directly. Diagonal operators and
disjoint two-by-two blocks use analytic matrix functions. Small general
operators use an extended-precision symmetric eigendecomposition, with
quadruple-precision Rayleigh refinement near zero eigenvalues.
Low-stiffness inputs use a terminated Taylor expansion of the forced
evolution equation without the general matrix-format preparation.

Isolated rows are evaluated analytically and removed from the iterative
right-hand sides. Spectral bounds then apply to the remaining invariant
block; their complete-vector output magnitudes also set valid error budgets.
This does not assume that the full operator is invertible.

The general method is a joint Chebyshev expansion of the exponential and
phi1 functions. A backward Clenshaw recurrence evaluates both actions with
one sparse matrix-vector product per degree. Positive scaled Bessel
coefficients are computed by a normalized backward ratio recurrence;
phi1 coefficients follow by summing the exponential coefficient tails.
Gershgorin and nonnegative-matrix upper bounds enclose the spectrum.
Truncation uses coefficient-tail bounds and the actual right-hand-side
norms, with reevaluation if the output scale requires more terms.
Coefficients and matrix preparation are reused within the session only.

Heterogeneous high-stiffness operators may use adaptive symmetric Lanczos
propagation. Its stopping rule bounds the positive residual integral by a
shifted tridiagonal resolvent, optimized over the shift. A bounded probe
falls back to Chebyshev when Krylov convergence is not advantageous.
Pure exponential requests avoid the affine cancellation in the forced form.
For diagonally dominant operators, the bound uses the infinity norm of
the actual Lanczos residual. Similar requests can reuse output-scale
estimates; final acceptance always checks the current request's bound.

When a positive lower bound certifies that the transient is negligible,
Jacobi-preconditioned CG computes the forced steady state. Both the true
residual error and the omitted transient are bounded before acceptance.
This path can operate on a nonsingular block of a singular full matrix.

Verified separable path/grid operators use FFTW sine/cosine transforms.
Every edge, weight, sign, and diagonal is checked; diagonal representation
discrepancies are covered by a perturbation bound or trigger the general
fallback. General disconnected blocks of at most 32 rows use extended-
precision symmetric eigendecompositions.
Verified conservative operators have an exact constant-vector path.
Stiff high-accuracy inputs use a difference-form sparse multiplication,
with residual row potentials computed from the received data in quadruple
precision. Selected kernels use a four-row AVX2 sparse layout.

## Public Checks

All six supplied samples passed with three independent process repetitions.
Independent checks cover all precision levels, singular and signed
operators, large 3D grids, zero time and matrices, subnormal entries,
strict cancellation, Hadamard spectral blocks, permuted isolated rows,
and varied 16-request lifecycles. A 20,000-row regular-graph check has
980,000 stored entries; larger interleaved benchmarks stay within the
120,000-row and 1,000,000-entry limits.

The final clean-copy build is byte-identical to the supplied executable.
All 182 independent checks passed with a stripped runtime environment.
AddressSanitizer, UndefinedBehaviorSanitizer, and floating-division checks
passed the supplied samples without diagnostics.

All seven public evaluations passed all 43 cases. The selected candidate's
latest feedback score is 59.4/100; the best development feedback was
60.3/100. These are public-set results, not a held-out final score.
