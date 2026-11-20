# Polarization demosaicking

`solution.py` exports `reconstruct(raw)`. It uses the installed NumPy only,
with no build step, external data, or saved state. Its result is `(4, H, W)`
in `float32`, with planes ordered 0, 45, 90 and 135 degrees.

The four staggered sample grids are aligned and smoothed to estimate
angle-dependent offsets. Removing those offsets gives a full-resolution
intensity guide. A local Wiener estimator combines directional predictors
of this guide, with noise calibrated in quiet areas of the input mosaic.
A windowed Fourier Wiener pass uses the directional estimate as a pilot
to refine noise suppression while keeping the low-frequency intensity.
Small gain and alignment differences are estimated on overlapping scales
using a locally regularized first-order model. Where the angles have
distinct detail, the shared-guide estimate is blended with independently
denoised bicubic interpolation.

If the sampled planes agree substantially better after translation, a
registration-aware branch builds noise-regularized interpolation filters
from the inferred sampling geometry. The output is weighted by local
registration agreement and faded out at the image boundary. It uses a
bounded stencil and processes row blocks to keep latency predictable.

The original samples are always preserved at the measured angles.

To run the supplied local contract and RMSE check:

```sh
python /workspace/public-dev/check.py /workspace/submission/solution.py
```
