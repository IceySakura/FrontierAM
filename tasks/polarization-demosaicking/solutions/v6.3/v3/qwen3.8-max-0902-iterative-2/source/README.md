# Polarization mosaic demosaicking

`reconstruct(raw)` estimates the four full-resolution polarization planes
(0/45/90/135 deg) from a single DoFP mosaic frame whose filter pattern is

|            | even col | odd col |
|------------|---------:|--------:|
| even row   | 90       | 45      |
| odd  row   | 135      | 0       |

## Method

Each plane is observed on exactly one of the four 2x2 sublattices, i.e. every
plane must be upsampled 2x per axis from its own samples.  The estimator uses
the decomposition of the four planes into a shared intensity field
`S = mean_k I_k` and per-plane polarization details `D_k = I_k - S`:

1. `S` is initialised as the mean of four independent separable cubic
   (Keys family, weights `-t, .5+t, .5+t, -t`) upsamplings of the observed
   sublattices; averaging the four planes makes `S` about twice as clean as
   any single plane.
2. On its own sublattice the detail is observed directly, `D_k = I_k - S`,
   and is upsamplinged with the same separable cubic (row-missing pixels:
   cubic along the sample column; col-missing pixels: cubic along the sample
   row; diagonal-missing pixels: the tensor-product cubic, implemented as the
   average of the two cubic passes over the half-filled fields).
3. The planes are re-formed as `I_k = S + cubic(D_k)` (observed samples are
   reproduced exactly by construction), `S` is re-estimated and step 2 is
   repeated once.
4. The result is clipped to [0, 255].

Per-scene adaptations, all from robust statistics of the mosaic itself:

* the cubic tension `t` depends on a robust noise scale (MAD of the lattice
  second differences): cleaner scenes get a slightly sharper kernel
  (`t` in [0.05, 0.095]); the detail cubic uses `t - 0.01`;
* for noisy scenes the cubic taps of the detail interpolation are blended
  towards the noise-averaging 2-tap / 4-tap box taps;
This shared-intensity + interpolated-detail scheme is markedly more accurate
than interpolating each plane independently (bilinear, bicubic, edge-adaptive
directional blends) on the development scenes, and - unlike a small residual
CNN that was also tried - it does not hallucinate texture on smooth or
synthetic-looking scenes.  Runtime ~0.4 s per 1024x1024 frame, pure NumPy,
single threaded, no compiled code and no external weights.

## Files

* `solution.py` - implementation of `reconstruct(raw)`.
