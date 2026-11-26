# Polarization demosaicking

`reconstruct(raw)` returns four float32 planes in the order 0, 45, 90, 135 degrees.

The method combines:

- a small block-wise linear predictor trained on the public development scenes;
- smooth cross-channel detail transfer from the observed mosaic sample;
- a cubic per-channel fallback when the estimated inter-channel polarization
  difference is large.

Observed mosaic samples are preserved exactly. Runtime uses only NumPy and the
standard library, and limits OpenBLAS to one thread when possible.

`weights.npy` contains the learned predictor parameters. No build step is
required; `build.sh` is a no-op.
