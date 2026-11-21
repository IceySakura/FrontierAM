# Long-Run Covariance Estimator

## Running

```sh
./run.sh /absolute/path/input.npy /absolute/path/output.npy
```

The input is one C-order, little-endian float64 NPY array of shape `(M, d)`.
Rows are observations in time order. The output is one C-order, little-endian
float64 NPY array of shape `(d, d)`, in the same coordinate order.

The output is the long-run covariance **H**, not `H/M`. It is finite, symmetric,
and positive semidefinite; rank-deficient and constant inputs are supported.

## Dependencies

Python 3, NumPy, and SciPy from the installed environment. Development used
Python 3.11, NumPy 1.26.4, and SciPy 1.12.0. There are no downloads, external
services, runtime compilation, or additional packages. No build step is needed.
`run.sh` sets numerical-library thread counts to one and sets `SEED=1`.

## Method

- Center, scale, and SVD-whiten the input to make the estimation rules independent
  of the supplied coordinate scales and linear mixing.
- For genuinely recurrent finite-valued trajectories, fit a transition model
  and solve its Poisson equation. A BIC check permits two-step contexts when
  the observations show strong evidence of additional memory. Reversibility
  is used only when a transition likelihood check supports it.
- Otherwise, select a vector autoregression by a corrected Akaike criterion
  and evaluate its zero-frequency spectral covariance. Similar dynamics are
  pooled using data-driven shrinkage and guarded structural model selection.
- Estimate covariance uncertainty using influence functions and overlapping
  batch sums, then regularize noisy covariance directions.
- For eligible slow rejection trajectories, fit a Gaussian random-walk model
  from the observed accepted and rejected transitions. Radial normality and
  efficient score checks control its contribution. The fitted kernel's Poisson
  equation is integrated by deterministic quadrature, not by generating a
  new trajectory. This optional fit has an internal time limit and a fallback.
- Transform back to the original coordinates and enforce symmetry and positive
  semidefiniteness. A batch covariance is available as a numerical fallback.

## Source And State

All submitted Python source is original implementation code. NumPy and SciPy
provide array operations, linear algebra, optimization, and special functions.
The estimator uses only the current input file. It does not load reference
matrices, use trajectory identifiers, simulate additional chains, cache input
data, or transfer information between invocations.
