# Exponential propagation for sparse symmetric operators (FAMEXP2)

Given a real symmetric positive semidefinite sparse matrix `L` (CSR, both
triangles stored), a time `t` and right hand sides `b0`, `b1`, a session answers

    y = exp(-t L) b0 + t * phi1(-t L) b1,      phi1(z) = sum_{k>=0} z^k/(k+1)!,

for one request (`one_shot`) or sixteen sequential requests (`reuse`) whose
right hand sides are revealed one at a time.

## Algorithm

Both matrix functions are expanded in **Chebyshev polynomials of the normalised
operator**

    A = (L - shift I) / scale ,      spectrum(A) subset of [-1, 1],

    exp(-t L) b0    = sum_j f_j T_j(A) b0
    t phi1(-t L) b1 = sum_j g_j T_j(A) b1 .

Because `A` is symmetric with spectrum in `[-1,1]`, every degree costs exactly
one sparse matrix-vector product and the recurrence

    T_0 v = v ,  T_1 v = A v ,  T_{j+1} v = 2 A T_j v - T_{j-1} v

is stable and needs no orthogonalisation. The two chains (`b0` and `b1`) are
advanced together, so the matrix is streamed once per degree and both right
hand sides are handled in a single pass.

**Spectral window.** `shift` and `scale` come from a window `[a,b]` that
contains the spectrum of `L`:

* the extreme eigenvalues are estimated by a fully reorthogonalised Lanczos run
  (24 steps, exact diagonalisation of the tridiagonal Ritz matrix);
* the window is padded by `2%` of the estimated width, by ten residual norms
  and by the difference between the Ritz values of the leading half and of the
  full tridiagonal matrix (a convergence indicator);
* the upper end is clipped by the rigorous bound `||L||_inf`;
* the lower end is at zero (the operator is positive semidefinite) unless the
  Gershgorin bound `min_i (L_ii - sum_{j!=i} |L_ij|)` is positive, in which
  case that certified lower bound is used. The window is never extended far
  below zero, because `exp(-t lambda)` grows there.

When the number of Chebyshev terms that the accuracy requires is small, the
Lanczos run is skipped altogether and the rigorous window `[-delta,
||L||_inf]` is used; the Lanczos run costs about a hundred matrix-vector
products and only pays off when it saves clearly more terms than that.

**Coefficients.** `f_j` and `g_j` are the Chebyshev coefficients of
`exp(-t lambda)` and `t phi1(-t lambda)` on the window. They are obtained from
the values of the two functions at the Chebyshev-Gauss nodes by a discrete
cosine transform (direct summation with a shared cosine recurrence). This is
stable for the very different dynamic ranges of the two functions and, unlike a
closed-form formula, it also delivers the coefficient tails that are needed for
the error bound.

**Degree selection.** The coefficients of an entire function on an interval of
width `W` in the exponential argument decay like `exp(-j^2/W)`, so the neglected
tail behaves like

    tail(m) ~ scale * sqrt(W)/(m sqrt(pi)) * exp(-m^2/W).

The degree is obtained from this model with the tolerance
`atol + rtol*||y||_inf`, where `||y||_inf` is measured by a short provisional
pass (12 terms) and re-evaluated while the chains grow. The bound uses the
Euclidean norms of the right hand sides: the Chebyshev polynomials are
contractions in the Euclidean norm, and that norm also bounds the error in the
infinity norm that the judge uses. The infinity norm of the right hand sides
is *not* a contraction bound and would badly under-estimate the error for
large `n`.

The model is used rather than the measured coefficient tails because the
transform itself has a round-off floor: once the true coefficients have
decayed, the measured tails are dominated by that noise and can be smaller
than the true tail, which would select too few terms.

**Accumulation.** The terms are summed in a single fused pass with the
recurrence.

The special cases `t = 0` (answer `b0`) and `L = 0` (answer `b0 + t b1`) are
handled directly.

## Files

| path | role |
|---|---|
| `run.sh` | entry point; runs the native solver, falls back to `solve.py` |
| `build.sh` | offline build of the native solver (g++, no network) |
| `src/fastsolve.cpp` | the native solver (single file, C++11) |
| `solve.py` | pure Python fallback with the same algorithm (NumPy/SciPy) |
| `tools/selftest.py` | analytic self test used by `build.sh` |

## Build

`build.sh` compiles `src/fastsolve.cpp` with `g++ -O2` and then runs the
analytic self test (`tools/selftest.py`). The binary is installed as
`bin/fastsolve` only if the self test passes; otherwise the submission falls
back to the Python implementation. The build needs no network access and no
test inputs; it takes a few seconds. The compiler flags are deliberately
generic (no `-march=native`) so that the binary runs on any x86-64 host.

## Dependencies

* native solver: a C++ compiler (`g++`) and libm only;
* fallback: Python 3 with NumPy and SciPy (both present in the frozen image).

No other third-party code is used.

## Protocol

The process writes exactly `READY\n` and flushes it, then reads the
`FAMEXP2\n` magic, the JSON header, the CSR arrays (`indptr` uint64,
`indices` uint32, `data` float64, little endian) and then, for every request,
two `n`-vector float64 arrays. For each request it writes exactly `n`
float64 values and flushes. It exits with status 0 after the last answer and
writes nothing else to stdout. Diagnostics (none in normal operation) would go
to stderr.

## Public checks

The six public sample cases of `samples/manifest.json` pass with the reference
answers, and the analytic self test in `tools/selftest.py` passes:

```sh
cd /workspace/task
python tools/run_local.py --manifest samples/manifest.json \
    --command "/workspace/submission/run.sh" --repeats 3 --output dev_report.json
```

All six cases are reported `valid` (accuracy and format), the largest observed
error is about 3% of the allowed threshold, and the memory use is ~40 MB.

Additional checks used during development (not part of the submission):

* an independent dense `eigendecomposition` reference on synthetic operators of
  the documented families (diffusion, geometric graph, heavy-tailed network,
  signed congruence, log-spectrum), for which the solver agrees to 1e-13
  relative or better;
* a self-consistency check between the `p4` and the `p10` answers on operators
  with up to 120000 rows, where the two agree to the `p4` tolerance;
* a sixteen-request `reuse` session, which reproduces the dense reference.
