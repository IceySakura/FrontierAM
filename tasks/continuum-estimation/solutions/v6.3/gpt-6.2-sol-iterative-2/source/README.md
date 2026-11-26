# Continuum estimator

Run `/bin/sh /workspace/submission/run.sh`. The program reads
`/workspace/inputs/input.json` and writes `/workspace/result/output.json`.
For local use, pass an input and output path as the first and second
arguments. The bundled Python and NumPy are the only runtime dependencies.

The primary model is `t + a*x**p + b*x**q + c*x**(q+1.5)`, with
`q > p > 0`. The spacing of the final two orders matches the common smooth
trend observed in the public development examples. The solver sorts the
levels, then takes first divided differences in `log(x)` to eliminate
`t`. It integrates over positive orders using smoothly tapered order and
coefficient priors informed by the development examples. Four
observations leave a two-dimensional integral; with five, a scalar
determinant root enforces the additional observation. Quadrature weights
account for the Jacobian of conditioning on the observed values.

The output is the weighted median of compatible limits, using the
evaluation loss's factor `1/max(1,abs(t))`. Constant and numerically
resolved single-power trends are handled directly. When the primary fit
is unusable, the solver uses a bounded local Richardson correction, or
the finest observation if the local trend reverses. The program uses
one thread and reads no other case data at runtime.
