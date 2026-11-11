# Polarization Demosaicking

## Interface

```python
from solution import reconstruct

pred = reconstruct(raw)
```

`raw` is a read-only, two-dimensional NumPy `uint8` array. The sensor tile is
`[90, 45] / [135, 0]`. The returned array is contiguous `float32` with shape
`(4, H, W)`, ordered **0, 45, 90, 135 degrees**, in the original intensity units.
Measured samples are preserved exactly. Predictions are finite and clipped to
`[0, 255]`. Full-size predictions are rounded to 1/16 intensity units.

## Method

Each polarization lattice is bilinearly interpolated. Smooth inter-channel
differences provide a polarization estimate, while all mosaic measurements
recover a sharp common-detail guide. A compact residual convolutional network
then refines missing samples. Its phase-packed features separate raw/guide
differences, shared detail, and local intensity context.

Reflection handles boundaries. An analytic path handles constant lattices and
small images. A structure gate and per-phase bias centering limit the learned
correction. A fixed residual calibration balances it against the analytic
estimate; measured pixels are explicitly copied back afterward.

Inference uses small NumPy matrix-multiplication blocks and limits BLAS to one
thread. It does not change CPU affinity or save scene-dependent state.

## Dependencies And Sources

- Runtime: Python 3.11, NumPy, and `threadpoolctl`, already installed.
- `polar_net.py` is the original NumPy inference implementation.
- `weights.npz` contains the supplied, offline-trained model parameters.
- Training used PyTorch on the two supplied Wen et al. 2019 sample scenes,
  with geometric, photometric, polarization, and noise augmentation.
- No pretrained third-party model, evaluation references, scene lookup table,
  downloads, or cross-scene data are used.

PyTorch and SciPy are not imported or required at inference. No compilation or
build step is necessary.

## Local Check

```sh
python3 /workspace/samples/check.py /workspace/submission/solution.py
```
