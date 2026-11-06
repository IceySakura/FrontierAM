# Public calibration

Only the public dataset-v2 files under `/tmp/exp2-public/task/dev` and
`dev_full` were used. No private case, result, model submission, or panel timing
was inspected.

The adapter exposes one calibration argument:

`internal tolerance = multiplier * protocol rtol`.

The frozen default is `multiplier = 1e-6`. The upstream adaptive timestep
algorithm is enabled and the Hermitian flag is always true. The basis is
`min(256,n)` for large stiff sessions (`n >= 20000` and
`t*opnorm(A,Inf) >= 10000`) and `min(60,n)` otherwise.

## Selection record

Public development showed:

- the upstream default Krylov dimension 10 was impractical on the stiff
  heavy-tail and log-spectrum smoke cases;
- dimension 30 with multiplier `1e-2` failed the cancellation-sensitive
  `syn_logspectrum_p10_k16` request (maximum scaled error about 692);
- dimension 120 with multiplier `1e-4` still failed that request (maximum
  scaled error about 19.3);
- multiplier `1e-6` passed it with dimensions 30, 60, and 120;
- dimension 60 removed three reuse timeouts seen with dimension 30 while
  retaining the upstream Lanczos/timestep policy;
- on the public full-size `syn_geometric_b` cases at target stiffness `1e5`,
  dimensions 60 and 120 exceeded the 60 s limit, while 256 passed p7 in
  `11.119 s` and p10 in `13.152 s`. The `n >= 20000` guard keeps the wider
  dense projected problem out of reduced smoke sessions.

The smallest publicly robust tested multiplier and the stated two-level basis
rule were therefore frozen. The optional argument exists so the benchmark
maintainers can reproduce public-only tolerance calibration; it is not selected
from hidden labels or right-hand-side properties at runtime.

## Final public validation

Evaluator image and isolation:

`frontieram-exponential:port-v1`, network disabled, one pinned CPU, 4 GiB,
read-only submission mount, and 256 MiB tmpfs.

- `dev`: 56/56 sessions passed; maximum scaled error
  `0.05138723145404088`; median measured session time
  `0.24055710798711516 s`; maximum `15.308276697993279 s`.
- `dev_full`: 40/40 full-size one-shot sessions passed; maximum scaled error
  `0.020730860289868078`; median measured session time
  `0.31400263048999477 s`; maximum `13.341982222016668 s`.

These are one-repeat public development measurements, not official calibrated
panel scores. The benchmark score mapping is
`final score = 100 / (1 + R)` with the rerun local panel reference anchored at
50; no `R` or final score is available from `run_local.py`.
