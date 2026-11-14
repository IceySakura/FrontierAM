# Polarization mosaic reconstruction

## Files and runtime

- `solution.py` defines `reconstruct(raw)` and returns a `(4, H, W)` `float32`
  array in channel order `0°, 45°, 90°, 135°`.
- `params.npz` contains only scene-independent numerical parameters fitted
  offline before submission. Keep it beside `solution.py`.
- Runtime dependency: Python 3.11 and NumPy (tested with NumPy 1.26.4).
  SciPy was used only during offline parameter fitting, not at evaluation time.
  No compiler, build step, network access, or `build.sh` is required.

Run the public checker from the task directory with:

```sh
python3 samples/check.py /workspace/submission/solution.py
```

The evaluator can also use `from solution import reconstruct` directly. The
function is deterministic, uses only its input and `params.npz`, and keeps no
state between scenes.

## Method

The four observed channels are highly correlated. At a pixel whose observed
polarization channel is `k`, the missing channel `c` is written as

```
prediction[c] = raw + mean_offset[k, c] + correction[k, c](local raw patch)
```

where `mean_offset` is estimated from the four 2×2 phase means of the current
scene.
The learned correction uses a 9×9 patch of values relative to the center pixel,
`raw[y+dy,x+dx] - raw[y,x]`. Center-referencing removes most of the common
intensity signal and makes the filters insensitive to exposure.

Training (offline, on the two public real sample scenes only):

1. Compute a 3×3 box-filtered mosaic and split pixels into six equal-frequency
   bins by local gradient magnitude.
2. For each of the four 2×2 phases and six bins, fit a ridge-regularized linear
   map from the 81 center-referenced patch values to the three missing-channel
   corrections. A global filter is fitted over all bins and averaged with each
   bin-specific filter (weight 0.5).
3. Fit a small affine calibration (slope and intercept per phase, missing
   channel, and texture bin) to the filter outputs.
4. Save the filters and calibration in `params.npz`.

At runtime the phase means and texture-bin quantiles are recomputed for each
scene. Learned filters are applied with single-precision matrix products to
only the pixels in their phase/bin. Observed pixels are left exactly equal to
`raw`; final values are clipped to `[0, 255]`.

A guard detects unusually strong, smooth polarization modulation (unlike the
real camera data used for fitting). In that case a classical per-channel
bilinear interpolation estimate is mixed in with a strength derived from the
scene's modulation statistics. This also makes the method robust on the public
synthetic checker examples.

## Sources and data

- The only image data used for fitting are the two public Wen et al. 2019
  sample scenes supplied in `samples/wen-001` and `samples/wen-002`. No hidden
  scene, hidden reference, or derivative of a hidden scene was used.
- Method components are standard ridge regression, 2×2-lattice bilinear
  interpolation, and box/gradient filtering. No third-party source code is
  included or required.
- NumPy is used at runtime; SciPy 1.12.0 was used for offline training
  convenience only.

## Development measurements

On the supplied real samples the full four-channel mean RMSE is about 1.57 and
1.80. The public two-example synthetic checker reports about 2.79 mean RMSE.
A complete development process (Python startup, NumPy import, loading a
1024×1024 mosaic, reconstruction, and writing a 16 MiB result) took roughly
0.69–0.87 s on the shared development core, with peak RSS below 350 MiB.
Verification runs on an isolated core; no speed-dependent credit is sought.
