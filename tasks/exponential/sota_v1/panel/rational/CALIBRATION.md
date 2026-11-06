# Public calibration

The optional first argument to `run.sh` is the dimensionless multiplier `m` in
the internal stopping threshold

`m * (atol + rtol * ||y||_inf)`.

The analytically selected default is **0.1**, providing a tenfold safety margin
between the successive-approximation estimate and the protocol tolerance.

For the frozen panel, the multiplier is to be selected only by the benchmark's
declared public calibration rule.  No timing-based tuning or private-data
tuning is performed by this implementation's author.
