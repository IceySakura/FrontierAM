# Budgeted Cubature

Run `./run.sh`. Input and output use the JSON Lines oracle protocol in the
task specification. Standard output contains only protocol messages.

The implementation uses the installed Python, NumPy, and SciPy. It does not
download packages, access the network, read test cases, or require generated
data files.

## Method

- Scrambled Sobol samples initialize the fit. Structural probes are kept
  separate from the digital-net averages.
- Separable functions are checked using independent coordinate combinations
  and integrated with adaptive one-dimensional rules.
- Polynomial, exponential-quadratic, periodic, tensor-cross, and ridge
  approximations are accepted only after checks against observed values.
- Finite differences locate candidate ridge directions and breakpoints.
- Smooth ridge integration uses stable uniform-convolution densities.
  Cancellation-prone truncated moments are recomputed with decimal arithmetic.
- Polynomial and kernel cubature, with a digital-net sampling fallback,
  handle functions not captured by the validated approximations.

All calls to the unknown function pass through the budgeted oracle interface.
Repeated points, probes, and validation queries are charged. Batches contain
at most 512 points. Evaluation of a learned interpolant requires no further
unknown-function queries.

## Resource Controls

Numerical libraries are restricted to one thread. The process maintains a
shared point counter and a computation deadline, including import time.
If fitting cannot finish in time, it returns an estimate from the uniform
samples already received. The deadline handler defers interruption during
a protocol exchange so that a JSON message is never cut short.

## Files

- `solver.py`: protocol, selection, quadrature, and fallbacks.
- `sampling.py`: uniform-sample bookkeeping and stable averaging.
- `ridges.py`: periodic and piecewise-affine models.
- `spectral.py`: two-dimensional exponential interpolation.
- `ridgepoly.py`: fitted polynomial ridge interpolation.
- `projection.py`: projection models and integrated basis functions.
- `tensor.py`: cross-interpolatory tensor quadrature.
