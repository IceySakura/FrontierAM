# Polarization mosaic demosaicking

`solution.py` defines `reconstruct(raw)` for a `(H, W)` uint8 polarization mosaic
(2x2 pattern: (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°) and returns a
`(4, H, W)` float32 array with channels 0°, 45°, 90°, 135°. NumPy only, no build
step. Observed samples are passed through unchanged.

Method (all linear filtering is done in the FFT domain of the four 2x-subsampled
sub-lattices, with even reflect padding so the mosaic phase is preserved):

0. **Variance stabilisation** – the sensor noise variance grows ~linearly with
   intensity (about 0.027 * I), so the mosaic is mapped through
   `T(I) = 2*sqrt(I/0.027)` (unit noise variance); all steps below run in that
   domain and the result is mapped back at the end.

1. **Polarization fields** `D_c` – the smooth deviation of each channel from the
   common intensity image – predicted from the mosaic by learned 15x15 linear
   filters (`filters_dt.npy`, ridge least squares on the public development scenes).
2. **Common image** `J` – the polarization-corrected mosaic `raw - D_{channel(x)}`
   is assembled as a full-resolution spectrum (four alias blocks) and Wiener
   filtered; the white-noise floor is estimated per scene as the 25th percentile
   of the smoothed periodogram. Estimate of channel `c` is `J + D_c`.
3. **Guided refinement** – on its own samples each channel is fitted locally as
   `a*J + b` (guided filter, 5x5 windows); the estimate moves 60% of the way to that fit.
4. **Misalignment gate** – sequential captures can be spatially misaligned. The
   inter-channel shift is estimated from the phase slope of the demodulated
   sub-lattice cross-spectra (plus a phase-coherence check); with growing
   misalignment the result is blended towards a learned cross-channel LMMSE
   estimate (aligned-trained filters, switching to filters trained on
   synthetically mildly-misaligned captures, `filters_rob.npy`, as the shift
   grows) and then towards a learned own-channel-only LMMSE estimate
   (`filters_v1t.npy`, 11x11). Blend weights were calibrated on real and
   synthetically misaligned development scenes.

A bilinear fallback is used if anything unexpected happens (unsupported shape,
non-finite values). No state is kept between calls. Typical runtime is about
0.3-0.4 s per 1024x1024 scene (plus interpreter start-up); peak memory < 0.3 GB.
