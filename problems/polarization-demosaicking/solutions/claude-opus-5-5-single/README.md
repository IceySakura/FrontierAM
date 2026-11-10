# Four-angle polarization demosaicking

`solution.py` defines `reconstruct(raw)`: given the 2-D `uint8` mosaic with the 2×2 tile
`[90°, 45°] / [135°, 0°]` (rows/cols counted from 0), it returns a `float32` array of shape
`(4, H, W)` with the 0°, 45°, 90° and 135° images, clipped to [0, 255].

## Method

1. **Frequency-multiplexing model.** With `X = (-1)^col`, `Y = (-1)^row`, every mosaic
   value is exactly `r = L − X·C1 − Y·C2`, where `L = S0/2`, `C1 = (S1+S2)/4`,
   `C2 = (S1−S2)/4`. The polarization signals sit on the (π,0) and (0,π) carriers, and
   for ideal polarizers the (π,π) band holds only intensity (see step 4). Once C1 and C2
   are estimated,
   `L = r + X·C1 + Y·C2` and
   `I0 = L+C1+C2, I45 = L+C1−C2, I90 = L−C1−C2, I135 = L−C1+C2`.
   This reproduces every observed pixel exactly and keeps the full-resolution intensity
   detail that is in `r`.
2. **Steered learned filters.** C1 and C2 come from the demodulated mosaics `−X·r` and
   `−Y·r`. Each is filtered with five learned 11×11 kernels. Their responses are combined
   per pixel with weights `1, cosφ, sinφ, cos2φ, sin2φ`, where φ = 2θ comes from a
   smoothed structure tensor of same-phase gradients. Low-energy regions get small
   weights through a threshold. The kernels were fitted offline by least squares on
   synthetic polarization scenes built from SciPy's bundled `face`/`ascent` test images
   (with random piecewise-smooth DoLP/AoP fields and sensor noise) and on generated
   scenes like the public examples. Flips and transposes were used for augmentation.
   No evaluation data was used. The coefficients are embedded in `solution.py`
   (`kernels.npy` is the same array).
3. **Edge-aware ratio-domain blend.** Within a surface the polarization ratio C/L varies
   more slowly than C itself. Each chroma becomes `0.4·C + 0.6·L·smooth(w·C·L) /
   (smooth(w·L²) + 1)`, using a separable [1,4,6,4,1] window. The weight is
   `w = 1/(1 + g²/36)`, where g is the same-phase gradient across the carrier direction
   (horizontal for C1, vertical for C2). This down-weights the pixels next to edges,
   which is where intensity leaks into the chroma estimate.
4. **(π,π) component.** `D = (I0+I90−I45−I135)/4` is zero for ideal polarizers. It can
   still absorb smooth gain differences between the angle channels. It is estimated as
   a strongly smoothed ratio `D/L` (à-trous [1,4,6,4,1] passes with dilations 1, 2, 4, 8)
   of the (π,π)-demodulated mosaic, and is included in the output. The observed pixels
   are still reproduced exactly.
5. Mirror padding keeps the 2×2 phase at the borders. The filters are applied with
   `numpy.fft` on a fast transform size (1040 for 1024² inputs). Inputs smaller than
   12×12 use a nearest-sample fallback.

## Dependencies and running

- Python 3.11 and NumPy only. There is no SciPy import at run time, which keeps
  start-up fast. Nothing needs compiling, so there is no `build.sh`.
- Files: `solution.py` (self-contained: the learned coefficients, shape `(2, 5, 11, 11)`, are
  embedded in it as base64). `kernels.npy` holds the same coefficients for reference only.
- Usage: `from solution import reconstruct; pred = reconstruct(raw)`, or
  `python3 public-dev/check.py submission/solution.py`.
- The method is deterministic, single-threaded and stateless. A full 1024×1024 scene
  runs end to end (interpreter start, imports, compute) in about 0.45 s on the
  development machine.

## Third-party sources

The code is original. The modulation analysis follows the standard frequency-domain
view of CFA/DoFP demosaicking (Alleysson et al.; Dubois). Steering learned filters by
local orientation is in the spirit of RAISR (Romano et al.). The ratio smoothing is
related to residual and guided interpolation. The training images are the public SciPy
sample images `scipy.misc.face` and `scipy.misc.ascent`.
