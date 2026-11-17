# Polarization demosaicking — gated LMMSE filters

`solution.py` defines `reconstruct(raw)` for a `(H, W)` uint8 DoFP mosaic
(phases: (even,even)=90°, (even,odd)=45°, (odd,even)=135°, (odd,odd)=0°) and returns
a `(4, H, W)` float32 array in the order 0°, 45°, 90°, 135°.

Dependencies: NumPy only (no SciPy import, no compilation, no `build.sh` needed).
Runtime ≈0.5 s per 1024×1024 scene on the (shared) dev core including interpreter start-up,
imports, `np.load` and `np.save`; peak RSS ≈210 MB.

## Method

1. **Local regime gates** (computed from the mosaic only, on the 512×512 per-angle sub-images):
   * `g` – mean local correlation (box window radius 8) between band-passed sub-images of
     the four angles: high where the angles share texture, low where they are noise-dominated.
   * `h` – local RMS (window radius 4) of the Stokes-model violation `I0+I90−I45−I135`,
     squashed as `d/(d+5)`: high where the angles disagree (motion, flicker, non-ideal optics).
2. **Gated linear prediction**: for every mosaic phase and every output angle, the value is
   `w0·x + Σ_j a_j·(w_j·x + b_j) + b0` with gate terms `a = (g, h, g², h², g·h)` and `x` the
   7×7 raw neighbourhood.
   This is a linear filter whose taps change smoothly with the local regime, so cross-angle
   information is used where it is reliable and suppressed where it is not.
3. Observed samples are copied unchanged; outputs are clipped to [0, 255].

The weights (`weights.npy`, shape 4 phases × 299 features × 4 angles) were fitted offline by
ridge-regularised least squares (normal equations) on the five public development scenes.
Reflect padding by an even width keeps the mosaic phase at the image border.
Inference is a single small GEMM per phase on the strided 7×7 patches (float32).

## Development results (public Wen5 scenes, mean of four per-angle RMSEs)

| method | 43 | 55 | 72 | 91 | 99 | mean |
|---|---|---|---|---|---|---|
| bilinear | 1.779 | 1.613 | 2.128 | 2.125 | 1.738 | 1.877 |
| 5×5, gates (g,h), fit on all 5 | 1.607 | 1.432 | 1.233 | 1.487 | 1.407 | 1.433 |
| **final** 7×7, gates (g,h,g²,h²,gh), fit on all 5 | 1.492 | 1.401 | 1.163 | 1.442 | 1.371 | **1.374** |
| final, leave-one-scene-out (43 kept in training) | — | 1.441 | 1.207 | 1.462 | 1.419 | — |

Held-out check inside scene 43 (train on everything except the bottom half of 43, test there):
1.352 for the final model (1.378 for the 5×5 two-gate model).

Scene 43 has weakly correlated angles (strong Stokes-model violation); filters fitted
without such a scene over-trust cross-angle information there, which is why it is included
in the final fit and why the regime gates exist.

Training code is not needed at run time; the procedure is: build the 299 features per pixel
for each phase (`X`, `X·a_j`, `1`, `a_j`), accumulate `FᵀF` and `FᵀY` over all scenes, add ridge `1e-4·tr/n·I`, solve.

Offline training scripts (not used at run time) are in `training/`; `python3 train_final2.py 7 B` from that directory
reproduces `weights.npy` (it writes to `/workspace/dev/weights_K7B.npy`).
