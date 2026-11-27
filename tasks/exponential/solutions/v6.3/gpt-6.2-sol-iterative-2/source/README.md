# Sparse exponential propagation

## Build and run

From this directory, run `./build.sh` to rebuild, and `./run.sh` to start
the solver. The build uses the installed GNU C++20 compiler, system
`libmvec`, and the double-precision FFTW 3 runtime (`libfftw3.so.3`).
It targets Linux amd64 with AVX2 and FMA, as provided by the task host.
Neither building nor running requires a network connection or Python.

The executable uses one thread and inherits the process affinity.
It writes `READY\n` after input-independent initialization, consumes
the FAMEXP2 wire header and full CSR matrix, and then handles the requests
in order. Standard output contains only the handshake and result bytes.

## Numerical methods

- Zero time, diagonal entries, isolated pairs, and small disconnected
  blocks are handled directly. Isolated rows can be separated from the
  rest of a larger operator so that they do not inflate its spectral
  interval.
- A centered Taylor expansion with an infinity-norm remainder bound
  handles short steps. The general fallback is a Chebyshev expansion on
  a bounded symmetric spectrum, using row, neighbor, and scaled
  Gershgorin bounds. Cosine coefficients are generated in extended
  precision. Injecting both right-hand sides into one Clenshaw
  recurrence needs one sparse multiply per degree.
- Sparse multiplication uses the CSR, four-row packed AVX2, or
  fixed-offset stencil kernel as appropriate. The packed formats use
  smaller indices and values when the conversions are lossless.
- Verified separable stencils use FFTW transforms. Constant-mode and
  negligible-transient shortcuts are checked against the operator.
  No row-sum or sign property is assumed without checking it.
- Selected stiff problems can use a short Lanczos approximation or
  rational partial fractions with shared poles for the exponential and
  `phi1`. The rational rules use AAA interpolation and the CRAM16
  coefficients of M. Pusa (2011). Complex shifted solves are checked by
  recomputing compensated residuals against the CSR matrix. The p10
  rational path also compensates the solution updates. If the budget
  check fails, the solver uses the polynomial path.

Spectral bounds, coefficients, transforms, and sparse layouts can be
reused inside a session. Nothing is retained across processes.

## Checks

Run the provided local sample harness from this directory:

```sh
python /workspace/task/tools/run_local.py \
    --manifest /workspace/task/samples/manifest.json \
    --command ./run.sh --repeats 3 --output /tmp/famexp-samples.json
```

Development also checked the solver against extended-precision dense
diagonalization, Fourier and cosine grid references, an independent
forward Chebyshev recurrence, and exact sparse nullspace examples.
Those checks included positive off-diagonals, cancellation, and
16-request sessions. Public-set timing feedback is not a held-out score.
