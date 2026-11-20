# FAMEXP2 exponential propagation

From this directory, build with `./build.sh` and run with `./run.sh`.
The executable targets Linux amd64 (x86-64-v3, including AVX2 and FMA).
It uses GNU C++17 and the installed glibc vector math library (`libmvec`).
The build is offline. The solver is single-threaded and inherits the
container's affinity.

## Algorithm

The solver reads one shared CSR matrix after `READY`, then handles each
request as it arrives. It reuses coefficients, sparse layouts, and work
buffers within the session. No subsequent right-hand side is read before
the previous result is sent.

The general path expands the exponential and its time integral on a
spectral interval enclosing `tL`. It computes their Chebyshev coefficients
in extended precision by backward Bessel ratios and positive tail
summation. A fused Clenshaw recurrence applies both functions with one
sparse product per degree. Coefficient tails times the input 2-norms
bound the truncation error; the degree is checked against the actual
answer scale. Close to the soft spectral endpoint, the recurrence
uses its difference form. Graph products retain the row sums of the
stored matrix, including nonzero diagonal potentials.

The spectral upper bound uses absolute row sums and optional positive
diagonal scaling. The sparse kernels use CSR, packed stencils, or
four-row SIMD slices. For local stencils, multiple recurrence steps use
an in-cache wavefront with checked dependencies.

Zero time, zero operators, diagonal and isolated two-by-two blocks have
direct paths. A symmetric eigensolver handles other small connected
blocks when its defect bound is small enough. A short Taylor polynomial
can be used at small scaled times. At large scaled times, the solver can use a separated hub mode,
a checked stationary solution, or a change of origin around an approximate
equilibrium. These paths check their residual or approximation error and
fall back to the general recurrence when necessary. Tight cases have a
second-interval roundoff check and an extended-precision fallback.

## Local Checks

```sh
./build.sh
python selfcheck.py
python /workspace/task/tools/run_local.py \
    --manifest /workspace/task/samples/manifest.json \
    --command ./run.sh --repeats 3 --output /tmp/famexp-check.json
```

`selfcheck.py` uses only the Python standard library. It checks the binary
protocol, sequential reuse, zero time, zero operators, and signed and
singular matrices against closed-form answers.
