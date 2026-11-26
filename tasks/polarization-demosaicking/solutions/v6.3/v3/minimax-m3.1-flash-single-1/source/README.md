# Polarization demosaicking — 0° / 45° / 90° / 135° reconstruction

`solution.py` defines `reconstruct(raw)` which takes the single-channel mosaic
frame `raw` of shape `(H, W)` (`uint8`) and returns a `float32` array of shape
`(4, H, W)` holding the four polarization planes in the order **0°, 45°, 90°,
135°**, in the same intensity units as the input.

```python
import numpy as np
from solution import reconstruct

raw = np.load("raw.npy")            # (1024, 1024) uint8 mosaic
planes = reconstruct(raw)           # (4, 1024, 1024) float32
```

No other files, no build step (`build.sh` is not required — the solution is
pure Python + NumPy + SciPy) and no state is kept between calls, so every
scene is processed independently in a fresh process.

## The mosaic

The polarizer wheel repeats every two rows and two columns:

|            | even column | odd column |
|------------|------------:|-----------:|
| **even row** |  90°       |  45°      |
| **odd row**  | 135°       |   0°      |

so each pixel carries exactly one of the four polarization measurements, and
each plane is observed on one quarter of the frame.

## Method

Write the plane belonging to parity `j` as `mu + d_j`, where `mu` is the
common grayscale intensity content of the scene and `d_j` the smooth
per-angle polarization deviation (`sum_j d_j = 0`).  The mosaic pixel at a
position of parity `j` reads `mu + d_j`, therefore

```
plane_k  =  mosaic  +  d_k - d_j                                    (1)
```

Two estimates of the left-hand side are combined.

**`S_k` — per-plane interpolation.** A local quadratic (order 2) spline
interpolation of the samples that belong to plane `k`.  It carries the
plane's own polarization deviation, but only the lower half of the spatial
spectrum: at the three quarters of the positions where plane `k` is *not*
observed, the top octave of the scene texture is missing.  Interpolating each
plane independently is a classical baseline and already quite strong, because
the four planes are strongly correlated (0.92–0.99) and locally almost
identical.

**`M_k` — wide low-pass of a lattice.** A Gaussian low-pass (σ = 0.7
half-resolution samples) of the same lattice tracks `mu + d_k`, so
`M_k - M_j` is a smooth estimate of the polarization deviation difference
`d_k - d_j` appearing in (1).  Adding it to the mosaic value transfers the
mosaic's *full-resolution* texture to every plane, which is where the
remaining accuracy comes from.  The two lattices are always compared at the
same pixel, so the smooth part of the scene gradient cancels in the
difference.

The two estimates are blended with weight `w`:

```
out_k = S_k + w * ( mosaic + (M_k - M_j) - S_k )
```

`w = 1` is the pure texture transfer, `w = 0` the pure per-plane
interpolation.

### Adaptive blend weight

The polarization estimate is only as good as the separation between the smooth
`d_k - d_j` and the scene texture, and that separation degrades with texture
density.  The contamination is measured directly on the mosaic: the
combination `(a + c - b - d) / 2` cancels the smooth polarization part but
not the texture, so the high-frequency residue of that combination measures
how noisy the polarization estimate will be.  `w` is then

```
w = clip(1.30 - 0.50 * contamination, 0.05, 0.62)
```

so strongly polarized but heavily textured scenes fall back towards plain
interpolation (where the correction is not worth its error), while clean
scenes make full use of the mosaic texture.  The rule was checked against
several alternative parameterisations; all of them land within 2 % of each
other on the public development scenes, i.e. the gain comes from the
principle, not from the exact constants.

### Implementation notes

* The interpolation is evaluated only on the half-resolution grid that each
  plane actually needs (three of the four parities), which keeps the whole
  reconstruction at ≈ 0.28 s for a 1024×1024 frame.
* The lattice low-passes are compared through a half-resolution bilinear
  shift, which is exact for the half-pixel offsets between the four lattices.
* Everything is vectorised NumPy/SciPy; no per-pixel Python loops, no
  randomness, no I/O.

## Results on the public development scenes

`python public-dev/check.py submission/solution.py`

| scene | RMSE 0° | RMSE 45° | RMSE 90° | RMSE 135° | mean |
|------:|--------:|---------:|---------:|----------:|-----:|
| 43 | 1.753 | 1.715 | 1.668 | 1.677 | 1.703 |
| 55 | 1.541 | 1.501 | 1.461 | 1.471 | 1.493 |
| 72 | 1.468 | 1.401 | 1.350 | 1.379 | 1.399 |
| 91 | 1.698 | 1.621 | 1.568 | 1.608 | 1.624 |
| 99 | 1.480 | 1.444 | 1.372 | 1.392 | 1.422 |

**public mean RMSE = 1.5283**

For reference, on the same scenes the plain per-plane spline interpolation
(without the cross-plane texture transfer) scores 1.744, and bilinear
interpolation of each plane scores 1.877.

## Runtime

End-to-end wall clock for one 1024×1024 scene, measured from process launch
(including interpreter start-up, NumPy/SciPy imports, reading the mosaic and
writing the result) on the development host:

```
0.50 - 0.52 s   (limit: 1.000 s)
```

Peak memory is well under 100 MiB (limit: 8 GiB); the algorithm is
single-threaded and uses no swap.
