# FAMEXP2 exponential propagation

    y = exp(-t L) b0 + t * phi1(-t L) b1,     phi1(z) = (exp(z) - 1) / z

equivalently the state at `s = t` of `u' = -L u + b1, u(0) = b0`, for a real
symmetric positive semidefinite sparse `L` delivered as full CSR.

## Layout

| file | role |
|---|---|
| `run.sh` | entry point executed by the controller (`exec python3 -B solver.py`) |
| `solver.py` | protocol driver, Chebyshev series machinery, block application |
| `kernel.c` | C kernels: `||L||_inf`, block CSR products, fused Chebyshev recurrence |
| `build.sh` | optional offline precompilation of `kernel.c` into `famexp_kernel.so` |

Nothing outside this directory is needed.  `run.sh` works with or without
`build.sh`: if no shared object is present the kernel is compiled during the
untimed pre-`READY` startup (well under the 10 s startup deadline), first into
this directory and otherwise into the temp directory, and if no compiler is
available at all the solver falls back to a pure NumPy path that implements the
same algorithm.  No network access, no package installation, no scipy needed.

## Algorithm

1. **Trivial cases.** `t == 0` returns `b0`; `||L||_inf == 0` means `L` is the
   zero matrix and returns `b0 + t*b1`.

2. **Rigorous spectral enclosure.** `g = max_i sum_j |L_ij|` is computed in one
   pass over the CSR data.  `L` is symmetric, so every eigenvalue lies in
   `[0, g]`; `eta = t*g <= 1e5` by the task statement.  The approximation
   interval is `[a, b] = [-1e-8 g, g]` (the tiny negative margin protects
   against rounding-level indefiniteness while keeping `|T_k|` bounded).

3. **Chebyshev series of both functions.** `f(l) = exp(-t l)` and
   `h(l) = phi1(-t l)` are sampled at Chebyshev-Lobatto nodes and their series
   obtained by a DCT-I (`numpy.fft.rfft` on the even extension).  `phi1` is
   evaluated with `expm1`, which stays accurate for `t*l << 1` and `t*l >> 1`.
   The table length `M` doubles until it resolves the degree demanded by the
   tightest tolerance that can occur (`2*deg + 16 <= M`, capped at 8192).

4. **Provable truncation.** For the symmetrically scaled operator
   `A~ = (2L/(b-a) - (a+b)/(b-a) I)` we have `|T_k(A~)| <= 1`, hence

       || sum_{k>m} c_k T_k(A~) v ||_2  <=  ( sum_{k>m} |c_k| ) ||v||_2 .

   The degree is the smallest `m` for which that bound drops below
   `SAFETY * (atol + rtol*||v||_2)` with `SAFETY = 0.01`, i.e. 100 times
   tighter than the usual published relative stopping rule.  The coefficient
   noise floor of the transform is subtracted before the tail is accumulated,
   otherwise the tail stagnates at transform round-off and the degree would be
   pushed to the table length for the `p10` level.  The truncation error of the
   two output parts is bounded separately and each keeps its own degree.

5. **One fused recurrence for both right hand sides.** `b0` and `t*b1` form a
   two column block.  The C kernel performs

       W_{k+1} = 2(c1 L + c0 I) W_k - W_{k-1},   Y += c_k * W_{k+1}

   in a single pass over the matrix data per degree, writing the new iterate
   into the buffer of the obsolete one, so both columns cost one matrix sweep.
   The cheaper of the two columns stops at its own degree and the more
   expensive one continues with a single column kernel.

6. **Wire level.** The matrix, both right hand sides and the result are moved
   with `readinto` into preallocated aligned NumPy arrays, so nothing is copied
   or converted after the pipe read.  Only `READY` and the raw result bytes go
   to stdout; diagnostics never do.

## Resource behaviour

* Startup (imports, kernel build) happens before `READY` and is not part of the
  measured session; it is ~0.3 s wall and CPU.
* Working set: CSR (12 B per nonzero) plus six `n`-sized `float64` work arrays,
  about 30 MB for `n = 120000`, `nnz = 1e6`.
* Cost is `O(deg * nnz)` with `deg ~ sqrt(eta * log(1/eps))`; measured on one
  core with `n = 90000`, `nnz = 449000`: 0.35 ms per degree for both columns,
  i.e. 0.02 s at `eta = 100`, 0.17 s at `eta = 1e4`, 0.54 s at `eta = 1e5`.
  A 16 request reuse session stays far below the 150 s cap.

## Public checks

    python tools/run_local.py --manifest samples/manifest.json \
        --command "./run.sh" --repeats 3 --output dev_report.json

run from `/workspace/task`.  All six public cases (five synthetic families plus
the analytic control, which is mandatory) pass.  Accuracy was additionally
cross checked against dense eigendecompositions of the public operators at all
three precision levels, and against a 100x tighter truncation at full scale.
