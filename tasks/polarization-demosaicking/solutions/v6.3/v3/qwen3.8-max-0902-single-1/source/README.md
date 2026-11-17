# Polarization CFA demosaicking — reconstruct four angle planes from one mosaic

## Interface

```python
from solution import reconstruct
out = reconstruct(raw)          # raw: (H, W) uint8
# out: (4, H, W) float32, channels 0 deg, 45 deg, 90 deg, 135 deg
```

Mosaic phase (rows/cols 0-based), matching `problem.md`:

| | even col | odd col |
|---|---:|---:|
| **even row** | 90° | 45° |
| **odd row**  | 135° | 0° |

## Method

Each pixel measures exactly one of the four angles, so `raw` is already a
*full-rate* measurement of the scene — it is angle-multiplexed, not
resolution-starved. Independent bilinear interpolation of each angle plane from
its own stride-2 lattice throws away the detail carried by the other three
lattices and is the dominant error source (dev RMSE 2.34). Writing
`I_θ = a + b·cos2θ + c·sin2θ`, the modulation amplitudes `b, c` are small
(RMS ≈ 1 grey level) compared with the image `a` (RMS ≈ 58) and are almost as
rough as `a` itself, so the reliable strategy is to keep `raw` as the carrier of
image detail and only predict the small *correction* to each missing angle:

```
I_θ(p) ≈ raw(p) + Σ_k w[θ, parity(p), k] · (raw(p+k) − raw(p)) + bias[θ, parity(p)]
```

* **Per-phase regression.** The mosaic has period 2×2, so pixels are split into
  four phases `p = 2·(row&1) + (col&1)`. Each phase gets its own optimal filter
  bank, since the geometry of which angles are nearby differs per phase.
* **LMMSE weights, fitted offline.** The weights are ridge-regularised ordinary
  least squares regressions fitted on the five public development scenes
  (window radius 2 → 24 taps + bias, 4 phases × 3 missing angles = 12
  predictors). They are shipped as a base64 float32 blob inside `solution.py`
  and are constant across scenes, as the rules require. Nothing is read from or
  written to disk at run time; no per-scene state.
* **Difference features.** Regressing on `raw(p+k) − raw(p)` rather than raw
  neighbourhood values makes the model invariant to a global intensity offset,
  which helps it transfer between scenes of different brightness.
* **Exact data consistency.** Where angle θ is itself observed, the correction
  is defined to be zero, so the reconstruction reproduces `raw` bit-exactly at
  every sampled pixel (1/4 of each output plane).
* **Range clipping.** The reference planes are `uint8`, so predictions are
  clipped to `[0, 255]`; this can only reduce absolute error and also guarantees
  finite output.

## Validation

Leave-one-scene-out cross-validation over the five public scenes (fit on four,
score the held-out one) — the honest generalisation estimate, since the
submitted weights are fitted on all five:

| method | LOO mean RMSE | dev mean RMSE (fit on all 5) | runtime |
|---|---:|---:|---:|
| per-channel bilinear (classical baseline) | 2.343 | 2.343 | ~23 ms |
| Stokes/`b`,`c` diagonal-difference + quincunx fill | 2.156 | 2.156 | ~34 ms |
| directional (edge-directed) variant of the above | 2.101 | — | ~82 ms |
| **submitted: per-phase LMMSE, radius 2** | **1.664** | **1.495** | ~80 ms |

Tried and rejected: radius 1 (LOO 1.897) and radius 3 (LOO 1.743 — overfits the
five available scenes); a locally-adaptive gated version that adds
`g·taps` features with `g = A/(A+8)` and `A` the mean absolute neighbourhood
difference (LOO 1.647, i.e. ~1 % better at more than 2× the runtime, so not
worth the timing risk).

## Timing and memory

Measured on one core of the development host, whole-process from launch to exit
including interpreter start, `numpy` import, load, compute and writing the
result: ≈ 0.25 s against the 1.000 s limit. Compute alone is ≈ 80 ms. `numpy` is
the only import (`scipy` is not used — it would add ≈ 0.2 s of import time).
Peak allocation ≈ 50 MB against the 8 GiB limit. All array work is float32 and
single-threaded; NumPy's element-wise kernels do not spawn threads, and no BLAS
call is made at run time (the only matrix solves happen offline, during fitting).

## Files

* `solution.py` — self-contained; defines `reconstruct(raw)` and embeds the
  fitted weights.
* `build.sh` — no-op apart from pre-byte-compiling the module; nothing needs
  compiling.

## Reproducing the fit

The fitting script is not needed at run time. It loads
`public-dev/wen5/{43,55,72,91,99}/{raw,reference}.npy`, builds per-phase
difference features, solves the ridge system for each (phase, missing angle)
pair, packs the 4×4×25 weight tensor and base64-encodes it into `solution.py`.
Random subsampling of training pixels uses `SEED = 1`.
