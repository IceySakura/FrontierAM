# Polarization DoFP demosaicking — 1 mosaic → 4 intensity images

## Interface

```python
from solution import reconstruct
pred = reconstruct(raw)          # raw: (H, W) uint8, read-only
# pred: (4, H, W) float32, order 0°, 45°, 90°, 135°, values in [0, 255]
```

Run the public self-check with:

```sh
python3 samples/check.py /workspace/submission/solution.py
```

## Dependencies

* **NumPy only** (developed against NumPy 1.26.4 / Python 3.11). SciPy is
  deliberately *not* imported — `import scipy.ndimage` costs ~125 ms of the
  1.000 s budget for no benefit here.
* No third-party code, no downloaded data, no compiled artefacts, no random
  numbers (so `SEED` is irrelevant — the algorithm is fully deterministic).
* `build.sh` is a no-op that exits 0; nothing needs compiling.

## Sensor phase

Rows/cols are 0-indexed with period 2:

|            | even col | odd col |
|------------|---------:|--------:|
| **even row** | 90°    | 45°     |
| **odd row**  | 135°   | 0°      |

Verified against both provided real scenes: `raw[r::2, c::2]` equals
`reference[ch, r::2, c::2]` **exactly** for all four tiles, so the sampled pixel
of the matching output plane can be reproduced with zero error.

## Method

### 1. Stokes reparameterisation

Instead of interpolating four almost-identical images separately, write them in
the Stokes domain with `g = S0/2`, `s1 = S1/2`, `s2 = S2/2`:

```
I0 = g + s1     I90 = g - s1     I45 = g + s2     I135 = g - s2
```

Every raw reading is then a single linear measurement

```
v(p) = g(p) + sgn(p) · s(p),    sgn(p) = (-1)^(col+1)
s(p) = s1(p) if (row+col) even else s2(p)
```

This matters because the scenes are only weakly polarised (measured DoLP ≈ 0.04
mean, ≈ 0.11–0.15 at the 95th percentile; `std(s1) ≈ 2.0`, `std(s2) ≈ 1.7`
versus `std(g) ≈ 31–46`). All the image structure lives in `g`; `s1`/`s2` are
small, smooth fields. So only *those* need interpolating, and `g` is recovered
algebraically from the raw reading itself:

```
g = v − sgn · s     ⇒     the observed plane at each pixel equals v exactly
```

Propagating the two component errors `e1 = ŝ1 − s1`, `e2 = ŝ2 − s2` through the
four phases of the tile gives `MSE = E[e1²] + E[e2²]`, i.e. the score depends
*only* on how well `s1` and `s2` are estimated. That reduced the problem to
denoising two smooth low-amplitude fields.

### 2. Curvature-cancelling sample estimator

`s1` is sampled on the `(row+col)`-even checkerboard, `s2` on the odd one. A
local estimator for `s` must annihilate `g`, which is far larger than the signal.
The diagonal neighbours of a pixel share its `(row+col)` parity but flip `sgn`,
so a diagonal Laplacian isolates `s` and cancels `g` up to its curvature. The
implementation uses the wider kernel

```
ŝ = sgn · ( 3/8·v − 1/8·Σdiag±1 v + 1/32·Σdiag±2 v )
```

whose coefficients solve `Σh = 0`, `Σh·r² = 0`, `Σh·c² = 0`, `Σh·r = Σh·c = 0`
with unit gain on `s`. It therefore cancels constant, **linear and quadratic**
profiles of `g` exactly, leaving only third-order terms. Measured sample error
drops from `std = 2.21` (plain 3×3 diagonal Laplacian) to `std = 1.67`.

### 3. Hole filling + smoothing

Holes of each sub-lattice are filled from their 4 axial neighbours (always of the
correct parity), then each field gets 3 normalised 3×3 box passes (effective
support ≈ 9×9). This is the bias/variance optimum found by sweeping the number
of passes: the residual interference is high-frequency (a Laplacian of `g`
modulated to horizontal Nyquist by `sgn`) whereas `s1`/`s2` are smooth, so
low-pass filtering is close to the Wiener-optimal separation.

### 4. Recombine and clamp

Outputs are clipped to `[0, 255]` — the references are `uint8`, so clamping to
the known-valid range can only reduce squared error — and passed through
`nan_to_num` to guarantee the finiteness contract.

## Tuning notes / what was rejected

Parameters were chosen on the two supplied public scenes only (`wen-001`,
`wen-002`), with each candidate checked on both to avoid fitting one scene.

| method | mean RMSE (wen-001 / wen-002) |
|---|---|
| bilinear per channel (classic) | 3.42 / 3.22 |
| nearest raw value replicated to all 4 planes | 2.83 / 3.56 |
| **this submission** | **1.71 / 2.00** |

* A global scalar gain on `(s1, s2)` was swept; `α = 1.0` is optimal (0.9 → 1.87,
  1.1 → 1.88), confirming the estimator is calibrated rather than under-fitting.
* Multi-scale least-squares fitting of the smoothing weights (6 features fitted
  on the sample scenes) was tried and **rejected**: the box-pyramid features are
  nearly collinear, the fitted weights were wildly non-monotonic (±12), and
  leave-one-scene-out generalisation was worse than the fixed 3-pass box.
* Fixed-point iteration (re-estimating `s` from an updated `g`) was **rejected**:
  it helped `wen-001` slightly but degraded `wen-002`, and it is partly
  degenerate — because `g` is itself derived from `v − sgn·s`, correcting the
  curvature term simply returns the already-smoothed estimate.
* Heavier smoothing (4–6 passes) keeps improving `wen-001` but monotonically
  hurts `wen-002`; 3 passes is the balanced choice.

## Cost (measured on the development core, which is 10–30 % slower than grading)

Whole process, per scene, `1024×1024`: **≈ 180 ms** wall clock from launch to
exit including interpreter start, `import numpy`, `np.load`, `reconstruct`
(≈ 80 ms of that) and `np.save`. Peak RSS **≈ 107 MiB**. Single-threaded, one
CPU, no scratch files, no cross-scene state. That leaves >5× headroom against
the 1.000 s / 8 GiB limits.

Boundary handling uses whole-sample-symmetric reflection, which preserves the
2-periodic sub-lattices for even dimensions; for axes shorter than the pad width
it falls back to `edge` padding, so degenerate shapes (`1×7`, `2×2`, odd sizes)
still return finite, in-range output.
