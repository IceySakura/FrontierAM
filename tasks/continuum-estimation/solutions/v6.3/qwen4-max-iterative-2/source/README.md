# Continuum estimation

`run.sh` reads `/workspace/inputs/input.json` and writes
`/workspace/result/output.json`.  For local checks it also accepts
`run.sh INPUT OUTPUT`.

The estimator fits the convergent two-power trend

```text
y(x) = t + a x^p + b x^q,   p,q > 0
```

and returns the continuum limit `t`.  For each pair of positive orders on a
deterministic grid, the linear coefficients `(t,a,b)` are marginalized with a
small observation-noise model and weak coefficient priors.  The final estimate is
a posterior-weighted average over orders, with a mixture prior on the sign of
the secondary coefficient.  A guarded exact two-power fit sharpens
well-conditioned grids, with more weight given to wider level spans.  Levels are
sorted internally and rescaled for numerical stability.  The program uses one
thread, NumPy, SciPy, and the Python standard library; it performs no network
access and keeps no state between invocations.

Runtime is well below the 60-second per-case limit (about one second including
interpreter startup on the bundled runtime in local checks).
