# FAMEXP2 sparse exponential propagation

## Interface

`run.sh` starts the self-contained C++ program `solve`. It prints `READY\n`,
reads the FAMEXP2 binary stream from stdin, and writes exactly `8*n` bytes for
each request before a zero exit. Diagnostics are only emitted when
`FAMEXP2_DEBUG` is set by a developer; they go to stderr. No network access,
data files, or runtime downloads are used.

`build.sh` compiles `solve.cpp` with the offline `g++` and libm in the frozen
image. `run.sh` checks a native-build stamp and rebuilds once if the submitted
binary is missing or was built for a different host. Compilation happens before
`READY`, never on session input.

## Algorithm

For a symmetric PSD matrix `L`, the program computes

```
y = exp(-t L) b0 + t phi1(-t L) b1 .
```

A positive diagonal Collatz-Wielandt scaling of `|L|` gives a safe upper bound
`beta >= lambda_max(L)`; power iterations on `|L|` tighten it. With
`B = 2 L/beta - I` and `a = t beta/2`, both functions have Chebyshev series on
the spectrum of `B`:

```
exp(-a(1+u)) = q_0 + 2 sum_{k>=1} (-1)^k q_k T_k(u),
t phi1(-a(1+u)) = inv_factor * (Q_0 + 2 sum_{k>=1} (-1)^k Q_k T_k(u)),
```

where `q_k = exp(-a) I_k(a)`, `Q_k = integral_0^a exp(-z) I_k(z) dz`, and
`inv_factor = 2/beta`. Scaled Bessel ratios are computed by Miller's downward
continued fraction and normalized by `exp(a)=I_0(a)+2 sum I_k(a)`. The `Q_k`
are obtained from the stable identity
`[z(q_k+q_{k+1})]' = (k+1)q_k - k q_{k+1}`.

The two Chebyshev series are evaluated together by a forward three-term
recurrence, so each degree needs one sparse matrix pass for both right-hand
sides. Coefficient-tail bounds, source 2-norms, the current output maximum,
and the requested `atol + rtol*max|y_ref|` determine an adaptive stopping
degree. Long doubles and compensated accumulation are used for small systems
and tight small-output requests; large systems use a four-accumulator double
sparse matvec with a strict tail criterion. Diagonal, zero-matrix, and `t=0`
cases use exact scalar formulas.

The input-dependent Perron bound, coefficient generation, preparation, all
requests, outputs, and process exit are inside the measured interval as
required. No state is reused across sessions.

## Dependencies

- C++17 compiler (`g++`), libm, and POSIX `read`/`write`.
- No Python, BLAS, LAPACK, external packages, or network at runtime.

## Public checks

The complete public sample manifest passes with `run_local.py`, including the
analytic control. Additional local development checks covered 16-request
sessions, `p4`/`p7`/`p10`, signed sparse PSD Laplacians with zero eigenvalues,
wide-spectrum diagonal operators, stationary and high-frequency right-hand
sides, low and `t*||L||` near `1e5` stiffness, and `n=120000` timing cases.
Examples:

```sh
python tools/run_local.py --manifest samples/manifest.json \
  --command "./run.sh" --repeats 3 --output dev_report.json
```

Build and run from this directory with:

```sh
sh ./build.sh
./run.sh
```
