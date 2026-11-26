# Exponential propagation solver (FAMEXP2)

Solves the streaming session

```
y_j = exp(-t L) b0_j + t * phi1(-t L) b1_j ,      phi1(z) = sum_k z^k/(k+1)!
```

for a fixed real symmetric positive semidefinite sparse matrix `L` (given in CSR
form, `n <= 120000`, `nnz <= 10^6`), a fixed time `t` and a fixed accuracy, for
one (`one_shot`) or sixteen (`reuse`) right hand side pairs.

## Entry point

```
./run.sh            # reads the session from stdin, writes the answers to stdout
./build.sh          # offline build of the C kernel (already run for the shipped .so)
```

`run.sh` starts `main.py`.  `main.py` links `libfamexp.so` (see `kernel.c`),
which contains the numerical inner loop.  The shared library is shipped
pre-built; `build.sh` reproduces it with the system compiler and is safe to
re-run.  No network access, no downloads, no optional dependencies.

Dependencies: Python 3.12 and NumPy only (the Chebyshev transform is
implemented with `numpy.fft`; importing SciPy would add several milliseconds
of process teardown to every session), plus a C compiler and the C math
library for the kernel.

## Algorithm

### Structure exploited

`L` is symmetric and positive semidefinite.  Both scalar functions are
expanded in Chebyshev polynomials of the *scaled* operator

```
Ltilde = (L - c I)/h,    c = (lo + hi)/2,    h = (hi - lo)/2,
```

where `[lo, hi]` is a rigorous enclosure of the spectrum of `L` (see below).
The spectrum of `Ltilde` is contained in `[-1, 1]`.  With
`x = (lambda - c)/h`,

```
exp(-t lambda)      = sum_k a_k T_k(x),
t phi1(-t lambda)   = sum_k d_k T_k(x),
```

with the coefficients `a_k` and `d_k` taken from the Chebyshev-Lobatto samples
of the two functions (Chebyshev interpolation).  Both functions are entire, so
their Chebyshev series converge on the whole interval; the number of
interpolation nodes is a few times the degree that is finally used, so that
the interpolation resolves the series far beyond the truncation point.  The
second function is evaluated as `t*(1-exp(-t lambda))/lambda`, with the limit
`t` at `lambda = 0` (computed with `expm1`).

Both series are evaluated with the three term recurrence

```
T_0 = I,  T_1 = Ltilde,  T_{k+1} = 2 Ltilde T_k - T_{k-1},
```

so one pass over the matrix produces the next degree for *both* right hand
sides at once.  The two right hand sides are kept in one interleaved array, so
a single sweep of the sparse matrix serves both series and both cache lines
are shared by the two gathers.

### Rigorous error control and stopping rule

`|T_k(x)| <= 1` on `[-1,1]`, and `||T_k(Ltilde)||_2 <= 1` because the spectrum
of `Ltilde` is contained in `[-1,1]`.  Hence for the truncation after degree m

```
|| sum_{k>m} c_k T_k(Ltilde) b ||_inf
        <= || sum_{k>m} c_k T_k(Ltilde) b ||_2
        <= ( sum_{k>m} |c_k| ) || b ||_2 .
```

The recurrence is stopped at the first degree for which this bound satisfies

```
R * (1 + rtol) <= atol + rtol * || y_partial ||_inf ,
```

using the running value of `||y_partial||_inf`; since
`||y||_inf >= ||y_partial||_inf - R`, the inequality implies
`||y - y_ref||_inf <= atol + rtol * ||y_ref||_inf`, which is exactly the
acceptance rule of the task.  The bound is valid for every right hand side,
so no per-request tuning is needed.

### Spectral enclosure

The interval has to contain the spectrum, so an upper bound for
`lambda_max(L)` is needed.  Two *rigorous* bounds are combined:

* Gershgorin: `max_i ( L_ii + sum_{j!=i} |L_ij| )`.
* Collatz-Wielandt on the elementwise absolute matrix `|L|`: for a
  non-negative matrix `B` and a strictly positive vector `x`,
  `max_i (Bx)_i / x_i >= rho(B)`, and `rho(|L|) >= lambda_max(L)` because
  `x^T |L| x >= |x^T L x|`.  The iteration `x <- |L| x` (normalised, with a
  tiny floor that keeps `x` strictly positive) converges to that bound; the
  smallest value seen over the iterations is kept.

The number of Collatz-Wielandt iterations is scaled with the estimated degree
of the Chebyshev series, so cheap (weakly stiff) cases do not pay for it.

### Special cases

* `t = 0`: the answer is `b0`.
* `L = 0`: the answer is `b0 + t b1`.
* diagonal `L`: the answer is evaluated element wise with
  `exp(-t lambda)` and `(1-exp(-t lambda))/lambda` (the limit `t` at
  `lambda = 0`), using `expm1` for accuracy.
* `L` splits into small dense blocks (for instance pairs): the operator is
  block diagonal, so every block is diagonalised exactly (an analytic formula
  for 2x2 blocks) and the matrix functions are applied block wise in one
  vectorised pass.  This replaces the whole Chebyshev series.

### Reuse sessions

The matrix dependent work (enclosure, coefficients) is done once per session
and reused for all sixteen requests; each request only runs the recurrence.

## Interface

`main.py` follows `sdk/wire.py` of the task package byte for byte: it writes
`READY\n`, then reads the `FAMEXP2\n` magic, the JSON header, the CSR arrays
(`uint64` row pointers, `uint32` column indices, `float64` values) and, for
each request, two `float64` vectors; it answers with exactly `8*n` bytes of
little endian `float64` and flushes.  Diagnostics (only when `FAMEXP_DEBUG` is
set) go to stderr.  The process exits with status 0 through `os._exit` after
the last answer, so no interpreter shutdown time is charged to the session.

Before `READY` the process enlarges the session pipe (its default capacity on
this image is small, which would turn the transfer of a megabyte sized matrix
into hundreds of round trips) and touches the numpy routines that are used
later, so that none of that initialisation is charged to the session.

The incoming CSR arrays are used in place; the matrix is never converted,
sorted or copied.

## Public checks

```
cd /workspace/task
python3 tools/run_local.py --manifest samples/manifest.json \
       --command "/workspace/submission/run.sh" --repeats 3 --output dev_report.json
```

All six public cases (five synthetic structure families and one analytic
control) pass the accuracy check in every repetition, and the analytic
control reproduces the closed form.

Additional checks that were run during development (not part of the public
package):

* the public families at all three accuracy levels (`p4`, `p7`, `p10`),
  compared against a dense symmetric eigen decomposition of the operator: every
  answer is inside the required tolerance;
* scaled variants of the public families at `n` up to 1.2e5 and `nnz` up to
  1.3e6 with `t ||L||_inf` between 1e-2 and 1e5, including scale free graph
  Laplacians and block diagonal operators, in the `one_shot` and in the
  `reuse` scenario (sixteen requests);
* degenerate inputs: `t = 0`, zero matrix, single element matrix, empty rows,
  explicit zeros, `t ||L||_inf = 1e4`, and a matrix that is not positive
  semidefinite.
