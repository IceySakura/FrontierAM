# Exponential propagation

Implementation of the FAMEXP2 session interface for
`y = exp(-tL) b0 + t phi1(-tL) b1`, with real symmetric PSD sparse `L`.

## Algorithm

The Python process handles the binary protocol and input-dependent dispatch.
Heavy vector recurrences run in a small C++ shared library.

* Zero time, zero operator, and diagonal operators use closed forms.
* Very small operators (`n <= 128`) use a dense symmetric eigendecomposition.
* Small scaled time uses a combined Taylor/Horner recurrence.
* If a positive Gershgorin lower bound proves that `exp(-tL)` is negligible,
  the solver approximates `L^{-1} b1` by a Chebyshev expansion on the
  resulting spectral interval.
* Otherwise it uses a combined Chebyshev/Clenshaw recurrence for both terms.
  One sparse matvec per degree advances both right-hand sides simultaneously.
  Exponential coefficients come from scaled modified Bessel functions; phi
  coefficients are derived by a suffix-sum identity evaluated in extended
  precision before casting to double.

The spectral interval uses rigorous bounds: the infinity norm and, for large
scaled times, a Collatz-Wielandt bound on `|L|`. Short-row operators may be
converted to ELLPACK storage; irregular rows may use a SELL-C-32 layout.
The C++ kernels have an AVX2/FMA path selected at runtime and a portable
fallback.

## Dependencies

Python 3, NumPy, SciPy (`scipy.special.ive`), and the system C++ compiler.
No network access or downloaded packages are used.

`build.sh` builds `libkernel.so`. `run.sh` builds it if missing and then starts
the session process.

## Checks

Run from the task directory:

```sh
python tools/run_local.py --manifest samples/manifest.json \
  --command "/workspace/submission/run.sh" --repeats 3 --output dev_report.json
```
