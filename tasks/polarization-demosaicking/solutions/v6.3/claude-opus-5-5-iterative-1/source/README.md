# Polarization mosaic demosaicking (0°/45°/90°/135°)

## Method
A small learned, intensity-normalized ("ratio") demosaicker that works on the 2×2 super-pixel grid
and needs only a handful of NumPy/BLAS operations (≈0.15 s of compute for 1024×1024 on one core;
≈0.27 s per scene including interpreter start-up, NumPy import and I/O on the development machine).

1. **Normalization.** The raw mosaic is reflect-padded (reflection preserves the 2×2 filter phase).
   A local intensity `L` is computed with a 3×3 binomial filter — it weights the four polarizer
   angles equally, so `L ≈ S0/2` — and the mosaic is normalized, `n = raw / (L + 1)`. Polarization
   then appears as small, slowly varying ratios, largely independent of the scene texture/exposure.
2. **Features per super-pixel** (each 2×2 cell, 512×512 cells): the 3×3 super-pixel neighbourhood of
   `n` (36 values), 5×5 and 9×9 box means of each of the four phases of `n` (8 values of wider
   context for denoising), and `log(mean L + 1)` (a noise-level cue).
3. **Prediction** of the 12 unobserved values of the cell (4 positions × 3 missing angles):
   * a ratio term `Σ_q L_q · (W_q · features)` (linear in the normalized features, re-scaled by the
     local intensity of each of the four positions), plus
   * a non-linear residual `mean(L) · MLP(features)` with two hidden ReLU layers (24 and 16 units).
4. Observed pixels are copied from the input; predictions are clipped to [0, 255]; output is float32.

## Training (offline, not part of the submission run)
Weights (`model.npz`, ~19 KB; the same weights are also embedded in `solution.py`, so the file is
self-contained — `model.npz` is used when present) were fitted with PyTorch on the two public sample scenes only
(`samples/wen-001`, `samples/wen-002`); no test data were used. Augmentations: 1-pixel shifts and the
8 dihedral transforms (with the matching relabelling of polarizer angles, e.g. a horizontal flip swaps
45° and 135°), Gaussian blur (σ ∈ {0, 0.7, 1.2, 2.0}) followed by re-synthesized signal-dependent noise
(σ² ≈ 0.024·I + 1e-4·I², 25 % of it shared by the four angles of a pixel, as measured on the samples),
and uint8 re-quantization. The linear part was initialised by least squares, then everything was
trained with Adam on the MSE of the unobserved values.

## Dependencies
Python 3.11 and NumPy only at run time (no SciPy import, no compiled code, no build step).

## Third-party sources
None. The ideas (colour-ratio / residual interpolation, learned demosaicking filters) come from the
general demosaicking literature; all code is original.

## Running
```python
from solution import reconstruct
pred = reconstruct(raw)   # raw: (H, W) uint8 -> (4, H, W) float32, order 0°, 45°, 90°, 135°
```
`python3 samples/check.py submission/solution.py` runs the public interface check.
Any H, W ≥ 2 are accepted (odd sizes are padded internally and cropped back).
