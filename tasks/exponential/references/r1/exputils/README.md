# ExponentialUtilities.jl panel member

This is the ExponentialUtilities.jl human-panel baseline for FAMEXP2. It uses
SciML's unmodified `phiv_timestep!` implementation to evaluate

`exp(-tL)b0 + t*phi1(-tL)b1`.

The submission vendors Julia 1.11.7, the pinned Julia environment in
`Manifest.toml`, and a prebuilt incremental sysimage. It needs no executable or
package from the evaluator image and performs no network access.

## Run

```sh
./run.sh
./run.sh 0.000001
```

The optional first argument is the tolerance multiplier `m`; the adapter passes
`tol = m * rtol` to ExponentialUtilities. The public-calibrated default is
`m = 1e-6`; see `CALIBRATION.md`.

`run.sh` sets `JULIA_DEPOT_PATH` to the vendored depot,
`JULIA_PKG_OFFLINE=true`, and all Julia/BLAS/OpenMP thread counts to one. The
sysimage contains the package load and a tiny generic `phiv_timestep!`
precompile workload, so the process prints `READY` without runtime package
compilation.

## Thin adapter

`stream_exputils.jl`:

1. reads the FAMEXP2 header and arrays;
2. views the full symmetric CSR arrays as CSC for the transpose (equal to `L`)
   and negates the values to obtain `A = -L`;
3. allocates `B = [b0 b1]`, the output, and the upstream timestep caches once;
4. calls upstream `phiv_timestep!` for each request with `adaptive=true`,
   the fixed public-calibrated basis rule below, `ishermitian=true`, and the
   precomputed infinity norm/nnz;
5. writes each binary64 result and flushes before reading the next request.

`ishermitian=true` selects ExponentialUtilities' Lanczos path. The same upstream
`_phiv_timestep_caches` tuple is reused for all 16 requests; no state derived
from a right-hand side is carried to a future request.

The basis rule uses `min(256,n)` when `n >= 20000` and
`t*opnorm(A,Inf) >= 10000`, otherwise `min(60,n)`. It depends only on public
protocol values and the matrix norm, never on case identity or right-hand-side
values.

The adapter handles only exact protocol degeneracies outside the Krylov call:
`t=0`, the zero operator, and the one-dimensional scalar case. The scalar
formula uses `expm1` to avoid cancellation in `phi1`; it is needed by the
public stiff scalar control. No ExponentialUtilities source file is changed.

## Vendored material

- Julia 1.11.7 official Linux x86-64 binary distribution;
- ExponentialUtilities 1.37.1 and its resolved dependencies;
- upstream ExponentialUtilities source and MIT `LICENSE.md`;
- `Project.toml`, `Manifest.toml`, and the prebuilt sysimage.

Non-runtime registries, downloaded artifacts, tests, documentation, and package
compile caches were removed after sysimage creation to satisfy the 1 GiB
submission limit. The retained runtime was tested offline from a read-only
mount with only `/tmp` writable.
