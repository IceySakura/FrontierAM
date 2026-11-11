# Long-Run Covariance Estimator

## Dependencies

Python 3 and NumPy, already installed in the supplied environment. There are no
downloads, compiled extensions, external services, or build steps. All estimator
source is included in `estimator.py`; the implementation is original and uses
only NumPy's standard numerical operations.

## Running

```sh
./run.sh /absolute/input/input.npy /absolute/output/pred.npy
```

The input is a two-dimensional time-ordered trajectory. The output is one
C-contiguous, little-endian float64 NPY array with shape `(d, d)`. The launcher
limits numerical libraries to one thread and works from any current directory.
The estimator is deterministic and does not retain state between invocations.

## Method

The trajectory is centered, and its lag-one covariance is whitened using the
empirical zero-lag covariance. The largest absolute eigenvalue of the symmetric
lag matrix determines an adaptive batch width. This accounts for both positive
and negative serial correlation while making bandwidth selection insensitive to
invertible changes of coordinates.

Overlapping window sums are computed using cumulative sums. A finite-sample
corrected covariance is calculated at three nearby batch widths and averaged.
Each term is a positive multiple of a Gram matrix, so their average is positive
semidefinite. Dividing the averaged batch estimate by the trajectory length
normalizes it to the requested long-run covariance scale.

The implementation uses linear time in trajectory length for fixed dimension,
with memory proportional to the input size. It handles constant and
rank-deficient trajectories without a matrix inverse. No sample reference
matrices or model-specific answers are used.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 python3 -B test_estimator.py
```

Tests cover the executable interface, finite outputs, symmetry, positive
semidefiniteness, affine transformations, constant trajectories, and rank
deficiency. The supplied sample checker can also be run against the output.
