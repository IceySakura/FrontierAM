# Polarisation mosaic demosaicing

`solution.py` exposes `reconstruct(raw)`:

```python
import numpy as np
from solution import reconstruct
planes = reconstruct(raw)      # raw: (1024, 1024) uint8  ->  (4, 1024, 1024) float32
```

`planes[0]`, `planes[1]`, `planes[2]`, `planes[3]` are the intensity images at
0 deg, 45 deg, 90 deg and 135 deg, in the same intensity units as `raw`.

## Files

| file | contents |
|---|---|
| `solution.py` | the estimator (pure NumPy; SciPy is *not* required) |
| `filters.npy` | the linear reconstruction filters, `(4, 4, 9, 9)` float64 |
| `README.md` | this file |

No build step is needed (there is no `build.sh`); NumPy is the only dependency.
The filters are also embedded in `solution.py` (compressed), so the file works
on its own even if it is copied without `filters.npy`.

## Method

The mosaic repeats every two rows and two columns

```
        | even col | odd col |
        |---------|---------|
        | even row|  90 |  45 |
        | odd row | 135 |  0  |
```

so **every 2x2 block contains one sample of each of the four polarisation
planes**.  The four planes are four views of the same scene, each sampled on
one coset of the lattice `2Z x 2Z`, i.e. the task is a super-resolution
problem with a fixed 2x2 sub-pixel phase pattern.

Two observations drive the design:

1. The scenes are extremely smooth: more than 90 % of the power spectrum of a
   plane lies below 0.05 cycles/pixel.  A cubic (Catmull-Rom) interpolation of
   every plane on its own lattice therefore already removes the low frequency
   part of the error (mean RMSE 1.75 on the development scenes).
2. What is left is the content *above* the lattice Nyquist limit
   (0.25 cycles/pixel).  That content is small (std ~1.5 grey levels) but it is
   exactly the residual error, and it is partly *shared* between the four
   planes (pairwise correlation 0.4-0.7 of their above-Nyquist content,
   depending on the scene).  It can therefore be recovered from the samples of
   the *other* three planes - information that a per-plane interpolation throws
   away.

The estimator exploits this with a small linear filter per (plane, sub-pixel
phase) pair:

* the local mean (9x9 box) is removed, so the filters are insensitive to the
  absolute brightness/contrast of a scene, and added back afterwards;
* for every missing phase of every plane a `9x9` filter over the mosaic window
  is applied to predict that plane at those positions; the filter therefore
  sees the samples of all four planes (the three planes that miss the same
  phase share the window, so it is copied and applied only once per phase);
* the observed positions are replaced by the exact mosaic samples.

The `4 x 4 x 9 x 9` filters were obtained by ridge-regularised least squares on
the public development scenes of `public-dev/wen5`; nothing scene specific is
hard-coded and the same filters are applied to every scene.  Leave-one-scene-out
validation gives a mean RMSE of 1.42 over the development scenes (1.75 for a
plain cubic interpolation), and the estimator reaches an RMSE of 0.874 on the
public feedback set, i.e. about 13 % better than the reference method.

### Scene dependent fallback

The learned filters rely on the four planes sharing their high frequency
content.  A few scenes do not share it - development scene 43, a night scene
with fine building texture, is such a case - and there the filters are *worse*
than a plain interpolation (leave-one-scene-out RMSE 3.04 versus 1.68 for the
cubic).  Two measures deal with this:

* the filters are fitted on the development scenes whose four planes do share
  their high frequency content (55, 72, 91, 99).  Including scene 43 in the
  fit distorts the filters for all other scenes, because its residuals are much
  larger (leave-one-scene-out RMSE over the four remaining scenes: 1.35 with
  scene 43 excluded versus 1.45 with it included).
* at run time the estimator measures, from the mosaic alone, how much the four
  planes share their high frequency content: the mean pairwise correlation of
  the high frequency parts of the four cubic interpolations of the mosaic.  On
  the development scenes this statistic is 0.004 for scene 43 and 0.47-0.90 for
  the others, so the two cases are separated cleanly.  When the statistic is
  low the result is mixed with the plain cubic interpolation of the plane,
  which keeps such scenes safe (leave-one-scene-out RMSE 1.42 instead of 1.77).

The output is finally clipped to the valid intensity range `[0, 255]`, and every
step is guarded against non-finite values (a scene that would produce an
invalid output falls back to the plain cubic interpolation).

## Cost

`reconstruct` takes about 0.37 s for a 1024x1024 mosaic on one core (~0.49 s for
the whole process including the NumPy import), comfortably below the 1 s
per-scene limit.  Peak memory is ~170 MB.
