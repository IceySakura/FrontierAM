# Continuum estimation

The submission uses the bundled Python 3.11 and NumPy. It requires no build
step, network access, or persistent state, and runs with one numerical
thread.

```sh
/bin/sh /workspace/submission/run.sh
```

The default input and output are `/workspace/inputs/input.json` and
`/workspace/result/output.json`. The input may list levels in any order.
The output contains just one finite number under the key `estimate`.
To test with a public example:

```sh
PATH="/workspace/runtime/bin:$PATH" /bin/sh /workspace/submission/run.sh \
    /workspace/public/dev/case_000.input.json /tmp/continuum/output.json
/workspace/runtime/bin/python3 /workspace/public/check.py \
    /workspace/public/dev/case_000.input.json /tmp/continuum/output.json \
    /workspace/public/dev/case_000.truth.json
```

## Method

The main model is `T + a*x**p + b*x**q + c*x**(q+1.5)`, with `0 < p < q`.
This form fits the public smooth development examples to rounding
precision. There are more parameters than observations, so the estimate
integrates over compatible fits rather than choosing a single set of
exponents.

Adjacent secants on the log-level grid eliminate the intercept. For five
levels the integration variable is `q`; for four levels it is `(q, c)`.
At each quadrature point the algorithm eliminates the linear coefficients,
brackets the remaining scalar root for `p`, and recovers `T` and the
coefficients. It uses scaled powers and `expm1` to compute small differences
accurately on nonuniform grids.

The integration weights include the inverse Jacobian for conditioning on
the observations. Soft priors favor orders and amplitudes observed in the
public smooth examples, with a broader component to allow deviations.
The reported limit is the weighted median after reweighting by
`1 / max(1, abs(T))`, the denominator of the scoring loss. Constant and exact
single-power data have a direct path. When no compatible three-power fit is
plausible, the program uses a scale-normalized regularized fit or a local
estimate for oscillatory or numerically indistinguishable observations.

At runtime the program reads only the supplied input. It does not look up
examples or use their case identifiers or truth values.
