# Fixed development examples

This directory contains twelve fixed examples for local interface checks and
algorithm development. Each input is a compressed NumPy `.npz` file with only
`observed` and `psf`; the matching `*_truth.npy` is provided for local scoring.
All arrays are `float32`. `observed` is finite and nonnegative with shape
`(Z,Y,X)`. `psf` is finite and nonnegative with odd side lengths and sum 1.

The directory intentionally contains no generator, generator source, random
seeds, noise parameters, or reconstruction procedure. `manifest.json` records
file hashes and broad example categories only. The hidden formal evaluator uses
independently generated cases and does not reuse these clean volumes.

A submission may use `check.py` to validate an output against one public truth.
The checker is a development aid; its score is not a formal benchmark score.
The standard-v1 checker validates the output dtype and uses float64 error arithmetic.

Every formal case has a 0.500-second hard wall-clock limit, including process
startup, imports, input, computation, output and normal exit. The evaluator
supplies `TIME_BUDGET_SECONDS=0.5` and `DECONV_DEADLINE_MONOTONIC`.
The standalone `check.py` validates a saved prediction and reports its error.
Measure the full solver invocation separately when checking its runtime.
