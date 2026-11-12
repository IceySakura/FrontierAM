# Polarization mosaic demosaicking — gated learned filters

偏振马赛克去马赛克：方向/强度门控的学习型线性滤波器（纯 NumPy，无需编译）。

## Method

Sensor layout (2x2 tile, rows from 0): `[90°, 45°] / [135°, 0°]`; output order `pred[0..3] = 0°, 45°, 90°, 135°`.

For every pixel, each of the three unobserved channels is predicted from the 9x9 raw neighbourhood
(reflect padding at the borders) by a *gated* linear model:

    y_c(p) = sum_g  gate_g(p) * ( w_{phase,g,c} . patch(p) + b_{phase,g,c} )

* There is a separate weight set for each of the four 2x2 phases (each phase observes a different
  angle). The observed channel is copied through unchanged.
* Gates (6): `[1, c1, c2, st, st*c1, st*c2]`, computed from a local structure tensor:
  1. gradients `gx, gy` = 2-pixel central differences of the raw mosaic (both taps lie behind the
     same polarizer angle, so the polarization modulation cancels without any blurring);
  2. `Jxx, Jyy, Jxy` = Gaussian-smoothed (sigma 0.6) gradient products;
  3. `c1 = (Jxx-Jyy)/(Jxx+Jyy+1)`, `c2 = 2Jxy/(Jxx+Jyy+1)` (edge orientation x coherence),
     `st = s/(s+2)` with `s = sqrt(Jxx+Jyy)` (edge strength).
  The filter thus steers with local edge orientation and adapts its smoothing to edge strength.
* Output is clipped to [0, 255] and returned as float32 of shape (4, H, W).

The weights (`gated_r4.npz`, 8 small arrays) were fitted offline by ridge-regularized least squares
(relative ridge 3e-5) on the two public development scenes `samples/wen-001` and `samples/wen-002`
only; hyper-parameters (patch size, gate smoothing, strength constant, ridge) were chosen by
cross-scene validation (fit on one scene, test on the other). No held-out/formal data was used.
`train.py` reproduces `gated_r4.npz` exactly: `python3 train.py /path/to/samples`.
The same weights are embedded in `solution.py` (`_WEIGHTS_B64` = base64 of `gated_r4.npz`), so
`solution.py` is self-contained at run time; after retraining, re-embed the new file there.

Development results (mean RMSE over the 4 angles):

| method | wen-001 | wen-002 | cross-scene avg |
|---|---:|---:|---:|
| bilinear | 3.42 | 3.22 | 3.32 |
| plain linear learned filter (no gates) | — | — | ~1.86 |
| this method | 1.42 (in-sample) | 1.66 (in-sample) | 1.699 |

## Dependencies

Python 3.11 + NumPy (1.26.4). No SciPy, no compiled code, no build step (no `build.sh`).
No third-party code was copied; the approach follows the general ideas of learned linear
demosaicking and structure-tensor-gated (RAISR-style) filtering.

## Running

```python
import numpy as np
from solution import reconstruct      # solution.py and gated_r4.npz must be in the same directory
pred = reconstruct(raw)               # raw: (H, W) uint8 mosaic -> (4, H, W) float32
```

Deterministic (no randomness). Measured on the development machine: ~0.50 s per 1024x1024
scene for the whole process (Python start-up, import, load, reconstruct, save), peak RSS ~280 MB,
single thread (BLAS thread count forced to 1).

Files: `solution.py` (runtime, self-contained), `gated_r4.npz` (same weights as a file),
`train.py` (offline training, optional),
`README.md`.
