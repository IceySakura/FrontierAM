# Polarization (DoFP) mosaic demosaicking

`reconstruct(raw)` takes the (H, W) uint8 division-of-focal-plane mosaic and
returns (4, H, W) float64 planes in the order 0, 45, 90, 135 degrees.

Mosaic layout: (even row, even col)=90, (even row, odd col)=45,
(odd row, even col)=135, (odd row, odd col)=0.

## Method

Each plane is decomposed as `I_k = C + D_k`:

* `C` is a dense guide: a separable 5-tap low-pass filter
  `[b, 1/4, 1-2b-1/2, 1/4, b]` (b = -0.05) applied to the mosaic.  Its taps
  null the Nyquist frequencies, which is where the mosaic's per-pixel
  polarization offset concentrates, so the guide retains the full-rate
  intensity detail while rejecting most of the offset.
* `D_k = I_k - C` is a small residual, known exactly on channel k's quarter
  lattice (there `I_k` equals the raw sample).  It is extended to the full
  grid two ways: separable bilinear interpolation on the lattice, and a
  guided local-linear model `D_k ~ a*C + b` fitted on the lattice samples
  inside the 5x5-pixel window centred on each pixel (window contents depend
  on the pixel's position class; the fits are obtained from boxed lattice
  sums, so the whole model is O(1) per pixel).  The two are blended 0.6/0.4.

The guide-based estimate is then blended against the plain per-channel
bilinear interpolant with a spatially varying weight chosen inside the scene
by leave-one-out cross-validation on the sampled lattice: every sample is
replaced by the average of its four diagonal lattice neighbours, the guide's
analytic response to that replacement is computed, and the weight minimising
the cross-validated squared error (moments smoothed over a 9-sample lattice
window) is used.  This adapts the noise/detail trade-off per scene, per
channel and per region.  Sampled pixels are reproduced exactly.

The whole construction runs twice: pass 2 replaces the guide `C` by the
channel average of the pass-1 reconstruction, which estimates the common
intensity component with roughly half the error of the filtered-mosaic guide.

Inputs with odd dimensions are reflect-padded to even sizes and the result is
cropped back; tiny inputs fall back to copying the mosaic to all planes.

## Constraints

Pure NumPy/SciPy, single threaded, no state kept between calls, no runtime
data other than the input mosaic, scene-independent constants only.
Runtime ~0.45 s of computation per 1024x1024 scene (~0.65 s per process
including interpreter start-up and imports), well inside the 1.0 s limit.
