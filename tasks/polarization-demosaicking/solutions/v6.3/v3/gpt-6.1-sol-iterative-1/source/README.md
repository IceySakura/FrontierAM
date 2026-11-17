# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)` and returns a finite, contiguous
`float32` array in the order 0, 45, 90, 135 degrees.

The method packs the four mosaic phases onto a half-resolution grid, constructs
an aligned bilinear estimate, and predicts local corrections for the twelve
unmeasured phase/channel combinations. A compact four-layer convolutional
model transfers common image detail while retaining separate polarization
information. Its first layer is conditioned on blind between-view translation
estimates from the mosaic's low-frequency spectra. Uninformative spectra use
zero offsets. The measured samples are preserved exactly. Predictions are
bounded to 0-255. Spatially constant sampled planes are extended as constant
planes.

The fixed filters in `weights.npz` were learned offline from the supplied public
development images with geometric, channel, and exposure augmentation. There is
no scene identification, reference access, adaptation across scenes, or runtime
training. Offline training and augmentation used seed 1.

Runtime requirements: Python 3.11 and NumPy 1.26.4. No build, compiler, network,
PyTorch, or additional packages are required. Convolutions use single-threaded
NumPy matrix products in bounded strips. CPU affinity is inherited unchanged.

Verification: the last snapshot passed all five public-set cases and scored
61.7/100, with RMSE 0.765 times the reference. Final clean-process measurements
on the development scenes were about 0.58 seconds and 166 MiB peak RSS, including
startup, imports, input loading, output contract checks, and result writing.

Run the supplied interface check with:

```sh
python /workspace/public-dev/check.py /workspace/submission/solution.py
```
