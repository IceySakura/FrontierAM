# Polarisation demosaicking

`reconstruct(raw)` returns four full-resolution intensity planes (0°, 45°,
90°, 135°) from one 2×2 division-of-focal-plane polarisation mosaic.

## Method

The four planes share one broadband intensity image `I` and differ only by
three low-amplitude fields `A`, `B`, `C` modulated by the sampling pattern,

```
raw(y,x) = I(y,x) + (-1)^x A + (-1)^y B + (-1)^(x+y) C .
```

Hence at *any* pixel a plane can be written as

```
plane_k = raw - q + p_k ,    q = (-1)^x A + (-1)^y B + (-1)^(x+y) C ,
```

with `p_k` a fixed ± combination of `A,B,C`.  This keeps all measured
high-frequency detail, reproduces the measured samples exactly, and leaves
only the three fields to be estimated.  Each field is estimated by a 9×9
linear FIR filter of the mosaic plus a constant; the coefficients of two such
filters are blended by a local flatness weight, forming a gated /
piecewise-linear Wiener estimator.  Coefficients were fitted offline by ridge
regression on the public development scenes and on flipped and synthetically
shifted versions of them.  No scene-specific fitting, external data or
randomness is used at run time.

Some captures have the four planes mutually displaced by a few pixels (e.g. a
rotating-polariser sequence with camera motion).  A cheap self-contained test
detects this from the mosaic alone: it looks for a nonzero shift that improves
sub-lattice cross-correlation, and for demodulated fields that become
gradient-like, which is the signature of shifted planes.  A second expert,
trained on a wide variety of synthetic shifts (including sub-pixel ones), is
selected for such captures.  Predictions are clipped to the valid intensity
range, and a bilinear-interpolation fallback guarantees a finite valid output
if anything unexpected happens.

## Runtime

Pure NumPy, single-threaded, no compilation and no SciPy.  About 0.50 s of
computation and ~0.62 s end-to-end (interpreter start, import, input read,
compute) on the development machine, below the 1 s per-scene limit.  Peak
memory is well under 1 GiB.  Output: `float32`, shape `(4, H, W)`, all finite.
