# Polarization demosaicking: scene-adaptive, shift-aware Wiener (LMMSE) filtering

`reconstruct(raw)` takes the `(H, W)` mosaic (pattern `[[90, 45], [135, 0]]`) and returns a
`(4, H, W)` float32 array with channels 0°, 45°, 90°, 135°, clipped to [0, 255]. Measured samples
are copied unchanged. Only NumPy is used at run time, and there is no build step. Computation takes
about 0.35 s per 1024×1024 scene (about 0.5 s including interpreter start-up and I/O). There is no
randomness.

## Files
- `solution.py`: the algorithm.
- `wtab.npz`: parameters shared by all scenes. It holds the 4×4 cross-covariance functions
  (integer lags up to ±24) of the reference images of the five public development scenes, with
  the white sensor-noise spike removed. It also holds the inter-channel shifts of those scenes,
  measured with the same estimator as at run time, and their noise variances.
- `wtab_bayer.npz`: the same statistics after a simulated Bayer colour capture with bilinear
  demosaicking of every angle image (R = G = B = scene, gray = (R + 2G + B) / 4). This models
  sources that are smooth and Stokes-consistent, as produced by colour polarization cameras.

## Method
1. **Inter-channel registration.** The four angle images of a scene are often slightly
   misaligned: about 0.1 px typically, half a pixel for sensor-interleaved data, and several
   pixels when the camera drifted between exposures. For each channel, the 512×512 sub-image is
   registered against the 90° sub-image. Phase correlation gives the integer part. A weighted
   least-squares fit of the cross-spectrum phase slope at low frequencies refines it, and the known
   lattice offset is then subtracted. On the development data this matches the shifts measured on
   the full-resolution references to about 0.02 px.
2. **Covariance model of the scene.** Every cross-channel covariance of the training statistics is
   re-aligned (bicubic interpolation in the lag domain) from the training scenes' shifts to the
   shifts estimated for the scene. Three further parameters are chosen per scene:
   - `k`, a scaling of the lag axis (smoother or sharper images than the training scenes);
   - the white-noise variance;
   - `beta`, how much of the Stokes-inconsistency component `I0 + I90 - I45 - I135` the model keeps.
3. **Self-supervised model selection.** The candidate models are compared by how well their
   implied LMMSE predictors predict every observed mosaic sample from its neighbours, with the
   centre excluded (a blind-spot error on a sub-sample of the image). The search is coarse to
   fine: first `k` × noise level, then `beta` for the two best `k`. All noise levels are evaluated
   at once from one eigendecomposition per candidate. Negative eigenvalues of the interpolated
   covariance are clipped so that every solve stays well conditioned. The final filter is the
   average of the LMMSE filters of the two best candidates (when they share the same support),
   which hedges against noise in the selection.
4. **Demosaicking filters.** For each of the 4 mosaic phases and each of the 3 missing channels,
   the LMMSE predictor is solved from the selected covariance over a 9×9 neighbourhood. An 11×11
   neighbourhood is used when the scene is noisy. For each input channel, the window is centred on
   the position its content moved to (the rounded shift). All 12 filters are applied at once as an
   im2col + matrix product on the four mean-removed channel sub-images.
5. **Clean, Stokes-consistent scenes.** These are scenes where the search picks low noise and
   `beta < 1`. For them, the filters are averaged with the LMMSE filters of the Bayer-capture model
   (`wtab_bayer.npz`, Stokes-consistent). Both are linear, so this is a single filter with averaged
   weights, and it hedges between two plausible image models.
6. **Safety.** If the channels are misaligned by more than 3 px, the search is skipped and the
   default model (training statistics) is used. If no plausible registration exists (shift above
   12 px), per-channel bilinear interpolation is used. Every result is checked to be finite and
   close to a nearest-neighbour reference. If a check fails, the next fallback is used. The output
   always passes through `nan_to_num` and clipping, and odd or tiny inputs are padded and cropped.
