# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)` for a `(1024, 1024)` uint8 mosaic.
It returns a contiguous `(4, 1024, 1024)` float32 array in the order
0, 45, 90, 135 degrees, preserving all observed samples exactly.

The mosaic is packed into four half-resolution sample planes. Independent
bilinear interpolation supplies a baseline. A local statistical estimator
accounts for subpixel translations between exposures, estimated from the
current mosaic. Its covariance model separates shared spatial structure,
smooth angle-specific variation, and noise, while preserving independent
constant levels in each angle.

A contrast normalization is enabled for large inter-angle contrast differences.
Predictions are converted back to the original intensity units before output.

A compact convolutional residual model refines the interpolation. Its inputs
are local high-pass measurements, statistical interpolation corrections, and
leave-one-out prediction errors at measured positions. Only unobserved pixels
are corrected; the measured samples remain exact. Predictions are clipped to
the valid intensity range.

The model weights are general parameters trained offline on the supplied
public development images, using seed 1, crop and symmetry augmentation,
exposure changes, and synthetic noise/motion variations. Inference is
deterministic. There is no scene identification, reference-image access, or
cross-scene state at inference time.

Inference uses only Python 3.11 and NumPy. Convolutions use tiled matrix
multiplication with one BLAS thread. No build step, compiler, network access,
PyTorch import, or external service is needed. `model.npz` must remain beside
`solution.py`, together with `polar_prior.py`.

## Validation

The supplied development checker reports mean RMSE approximately 1.287.
All five development scenes were used in the final fit, so this is not a
held-out accuracy estimate. Separate checks cover finite output, exact
observations, immutable and noncontiguous inputs, repeatability, constant
planes, saturation, periodic texture, and unequal angle contrasts.
