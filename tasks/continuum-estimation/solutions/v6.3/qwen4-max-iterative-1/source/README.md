# Continuum estimator

`run.sh` reads `/workspace/inputs/input.json` and writes
`/workspace/result/output.json`.  Optional positional arguments override the
input and output paths for local checks.

The estimator sorts the positive levels and checks for a monotone smooth trend.
The smooth error model has three power terms, with the third order fixed 1.5
above the second, a structure identified from the public development materials.

For five levels it traces the scalar compatibility condition that the two
unknown orders must satisfy, then integrates along that curve with a Jacobian
weight and weak Gaussian coefficient priors.  For four levels it integrates the
exact-fit order surface under bounded coefficient priors.  A regularized grid
quadrature is available as a fallback.  The reported limit blends a posterior
mean with a loss-weighted median where appropriate.

Smooth trends with a positive leading coefficient may have one turning point;
the guard only rejects trends with multiple turning points, which are treated
as oscillatory and extrapolated with a robust Cauchy linear fit.  If the smooth
quadrature itself is nonfinite, the program falls back to ordinary linear
extrapolation.

The program uses the bundled Python/NumPy runtime, one thread, and no network
or cross-invocation state.
