# Polarization demosaicking (vNext draft)

`solution.py` exposes

```python
reconstruct(raw) -> np.ndarray   # (4, H, W) float32, angles 0, 45, 90, 135
```

It turns a single 1024x1024 uint8 polarization mosaic into the four full
resolution angle images.  Only NumPy is used, so nothing has to be compiled
(`build.sh` is a no-op).

## The problem

Every pixel of the mosaic measures the light intensity for exactly one
polarizer orientation.  The filter repeats every two rows and two columns

|            | even col | odd col |
|------------|---------:|---------:|
| even row   |   90 deg |   45 deg |
| odd row    |  135 deg |    0 deg |

so the image splits into four interleaved half resolution lattices, one per
angle, each of which has to be expanded back to full resolution.

## Method

Each angle image is written as

    I_k(p) = B(p) + D_k(p)

* `B` - the intensity field that the four angles share (the scene),
* `D_k` - a small, angle specific offset (the polarization signal; for
  opposite angles `D_0 = -D_90` and `D_45 = -D_135` up to sensor effects).

The reconstruction is built from those two parts.

**1. Shared intensity field `B`.**
`B` is estimated from *all four* lattices at once: each lattice is
interpolated to the full grid (Catmull-Rom cubic, the lattice samples
themselves are copied exactly) and the four interpolations are averaged.
Averaging the four lattices cancels the angle offsets to first order
(the four offsets of a pixel sum to about zero), while the image structure
that the angles share survives at full resolution - a single lattice could
only ever give a half resolution version of it.

**2. Angle offsets `D_k`.**
The residual of the k-th lattice, `I_k - B`, is exactly the offset of that
angle plus whatever error the shared field still has.  The residual is
interpolated the same way and added back to `B` for the corresponding
channel.

**3. Measured samples.**
Every pixel that was actually measured is copied back verbatim, so the four
reconstructed planes agree with the sensor at all 1/4 of their pixels and
only the unmeasured 3/4 are estimated.

## Accuracy

On the public development scenes (5 scenes, 4 angles each) the mean RMSE
over the four angles is 1.58 in 0-255 intensity units; the per scene values
are 1.86, 1.52, 1.36, 1.63 and 1.55.  For reference, an interpolation that
simply uses the raw mosaic for every angle gives 2.65, and interpolating
each angle from its own lattice gives 1.88.

## Cost

Purely O(H*W) array operations, single threaded, about 0.09 s of compute
for a 1024x1024 scene (roughly 0.1-0.2 s including interpreter start-up and
the NumPy import) and a few tens of MB of memory, well inside the one
second / 8 GiB budget.

## Files

* `solution.py` - the estimator (NumPy only).
* `build.sh`   - optional no-op build script.
* `README.md`  - this file.
