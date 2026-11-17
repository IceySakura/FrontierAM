# Polarization demosaicking solution

`reconstruct(raw)` expands one `uint8` `(1024, 1024)` polarization mosaic into a
`(4, 1024, 1024)` `float32` array in channel order 0°, 45°, 90°, 135°.

## Method

The four angle planes are strongly correlated in low frequencies, but the
amount of *high-frequency* correlation varies a lot between scenes (some scenes
have small registration differences between planes).  A single fixed
cross-channel interpolator is therefore fragile.  The submitted estimator:

1. **Phase-aware FIR features.** The mosaic is split into its four 2×2 phase
   sub-images (512×512 each).  For every half-resolution position, a 5×5 window
   of each sub-image is collected: `4·25+1 = 101` features, equivalent to a
   full-resolution 9×9 support.  They are stored as a contiguous
   `(101, 262144)` array so the whole reconstruction is a single BLAS gemm.
   Per-phase DC offsets are subtracted first and restored afterwards.

2. **Registration cue `rho`.** The mean cross-phase correlation of the
   high-pass sub-images is computed from the mosaic itself (six dot products
   after separable 3-tap smoothing).  It is ~0.3–0.65 for well-aligned planes
   and near zero when planes are mutually displaced.  It requires no reference
   data and is computed from the current scene only.

3. **Two registration experts.** Sixteen filters (four channels × four target
   phases) are fitted twice, each with the basis `(1, rho, rho²)`, using ridge
   regression on the public development scenes plus augmentations: dihedral
   transforms with the matching polarization-angle permutations, per-channel
   gain/offset changes, smooth random polarization fields, and per-channel
   integer shifts of 0–3 pixels.  One expert is fitted mostly on high-`rho`
   (aligned) examples, the other mostly on low-`rho` (displaced) examples.  At
   runtime they are combined with a smooth logistic weight in `rho`.  The
   augmentations make the fit insensitive to the polarization statistics of
   any particular scene and teach it to fall back on within-channel
   interpolation when inter-plane registration is poor.

4. **Exact data + physical range.** At every observed pixel the corresponding
   channel is restored to the measured value (the reference equals the raw
   there).  The result is clipped to `[0, 255]`, which can only decrease squared
   error because all targets lie in that interval.

The coefficients of both experts are embedded as base64 in `solution.py`;
nothing is read from disk at runtime.  Only NumPy is imported.  No compiled extension, no `build.sh`, no
cross-scene state, no randomness.

## Development results

`python3 /workspace/public-dev/check.py solution.py` gives
`public_mean_rmse = 1.4563` (per-scene 1.660 / 1.459 / 1.187 / 1.481 / 1.495).
For reference, on the same five scenes plain bilinear interpolation gives
1.877, guided residual interpolation gives 1.586, and a single (non-expert)
learned filter gives 1.487.  Leave-one-scene-out cross-validation of the whole
pipeline gives 1.501 mean RMSE (1.525 for the single-filter variant).

## Runtime

Measured in a fresh Python process per scene (including interpreter startup,
NumPy import, reading `raw.npy`, reconstruction and writing a 16 MB result):
**~0.21 s wall**, ~0.09 s inside `reconstruct`, peak RSS ~175 MB.  That leaves
a wide margin below the 1.000 s / 8 GiB limits.  Output is always finite
`float32` of the required shape; constant, saturated, odd-sized and
rectangular inputs are handled defensively as well.
