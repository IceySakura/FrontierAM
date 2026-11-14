# Polarisation demosaicking

`reconstruct(raw)` takes the read-only `(H, W)` `uint8` mosaic and returns a
`(4, H, W)` C-contiguous `float32` array in the order **0°, 45°, 90°, 135°**.
No state is kept between calls; the function is deterministic.

## Sensor layout

The 2×2 tile starts at the top-left pixel:

| | even column | odd column |
|---|---|---|
| **even row** | 90° | 45° |
| **odd row** | 135° | 0° |

## Method

The reconstruction is a **polyphase FIR interpolation**.  For each of the four
output planes and each of the four pixel parities, a 9×9 kernel maps the
neighbourhood of the mosaic directly to the wanted plane value.  The 16 kernels
were fitted once, offline, by regularised least squares (ridge regression):

* targets are the four full reference planes of the two **sample scenes**
  `samples/wen-001` and `samples/wen-002` (public Wen et al. 2019 data supplied
  with the task, unrelated to the evaluated scenes);
* training examples are augmented with the eight dihedral transforms, small
  sub-pixel shifts, channel permutations, random polarisation fields /
  amplitudes and additive sensor noise, so the coefficients do not depend on
  the particular polarisation statistics of those two scenes;
* a few fits (different augmentation draws / ridge strengths) are averaged into
  one kernel bank, which lowers sensitivity to a single fit;
* every plane has its measured pixels restored exactly from `raw` afterwards,
  and the result is sanitised and clipped to the valid 0…255 range.

The fitted coefficients are embedded directly in `solution.py` as a hex string,
so the solution is a single self-contained file.  At run time the four phase
sub-lattices are gathered with `numpy.lib.stride_tricks.sliding_window_view`
and evaluated with four small matrix multiplications (one BLAS `sgemm` per
phase).  Cost for 1024×1024 is about 0.17 s of compute plus Python/NumPy
start-up; peak memory is well below 1 GiB.  Odd image sizes are handled by
padding one row/column with edge values and cropping the result.

## Dependencies / sources

Only **NumPy** (tested with 1.26.4, Python 3.11).  No compilation, no network,
no third-party model or code, no build script needed.  The only external data
used for fitting are the two sample scenes distributed with the task; all
coefficients are constants shared by every scene.

## Running

```python
from solution import reconstruct
pred = reconstruct(raw)   # (4, H, W) float32
```

The self-check can be run with `python3 samples/check.py submission/solution.py`
from `/workspace`.
