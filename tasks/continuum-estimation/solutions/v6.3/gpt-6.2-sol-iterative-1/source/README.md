# Continuum estimation

Run the submission with the bundled numerical runtime on `PATH`:

```sh
/bin/sh /workspace/submission/run.sh
```

It reads `/workspace/inputs/input.json` and writes
`/workspace/result/output.json`. For local runs, pass input and output paths
as the first and second arguments. It uses Python and NumPy only and sets
numerical libraries to one thread.

The estimator uses the model

```
y(x) = t + a*x^p + b*x^(p+g) + c*x^(p+g+1.5),  p,g > 0.
```

Three terms with this spacing describe the smooth public development examples.
The solution does not fix the leading order or choose a single interpolant.
It eliminates linear parameters using divided differences in `log(x)`. For
four levels it integrates over the third coefficient as well as the compatible
orders; for five levels the extra observation constrains that coefficient.
Two complementary root coordinates let the quadrature account for multiple
compatible fits without depending on one root parametrization. The shape
prior mixes a component calibrated on the development examples with a broader
component so orders and coefficient signs are not treated as hard constraints.
The final estimate minimizes posterior absolute error weighted by
`1 / max(1, abs(t))`.

A constant or a single positive power is used directly when it reproduces
every observation to numerical precision. For extremely close levels, an
estimated rounding-error bound on the highest divided difference can trigger
a fit with fewer levels.

When the grid is numerically degenerate or the model cannot fit without
implausibly large coefficients, the fallback uses a bounded single-power
extrapolation for monotone data. For repeated reversals it averages the
observations with weights concentrated at the finest levels; for a single
reversal it uses the finest observation. No public data, development truth,
or previous invocation is consulted at run time.
