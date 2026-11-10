# Public calibration

The optional first argument to `run.sh` is the dimensionless multiplier `m` in
the internal stopping threshold

`m * (atol + rtol * ||y||_inf)`.

The analytically selected default is **0.1**, providing a tenfold safety margin
between the successive-approximation estimate and the protocol tolerance.

For the frozen panel, the multiplier is to be selected only by the benchmark's
declared public calibration rule.  No timing-based tuning or private-data
tuning is performed by this implementation's author.

## Correction 1 (2026-11-06, benchmark orchestrator; public data only)

The first public calibration showed an error floor of 0.40 of the protocol threshold on the public
`syn_geometric_b_p4_k1` / `p7_k1` cases (eta 1e5) at every stopping multiplier: for stiff sessions
(t*||L||_inf > 5e4) the inner-solve tolerance was fixed at `max(2e-13, 0.1*rtol)` and did not follow
the multiplier. It now uses `max(2e-13, m*rtol)`, identical at the default `m = 0.1`, so the single
calibration argument controls every tolerance of the method. No other change.
