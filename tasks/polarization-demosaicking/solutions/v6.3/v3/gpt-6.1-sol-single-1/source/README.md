# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)` and returns a contiguous float32 array
of shape `(4, H, W)`, ordered as 0, 45, 90, and 135 degrees.

The reconstruction starts from independent bilinear interpolation of the four
sampling lattices. Two complementary local estimators correct only missing
samples: a residual multilayer network and a learned mixture of linear
interpolation filters. Their corrections are averaged. Both use 7-by-7
contrast-normalized neighborhoods of all four half-resolution lattices.
Observed samples are preserved exactly. Boundaries use nearest-sample
extension, and predictions are clipped to the valid intensity range.

The fixed coefficients in `weights.npz` and `auxiliary.npz` were trained using
the public development references with spatial symmetry, intensity, and
independent channel-translation augmentation. Training used seed 1.
These coefficients are common to every input; no scene identification,
reference access, external data, runtime training, or cross-scene state is
used. Runtime evaluation is deterministic.

Only the installed NumPy, SciPy, and Python standard library are needed at
runtime. SciPy's compiled uniform filters compute local statistics; the
network is evaluated with tiled NumPy matrix operations.
No build step or package download is required. BLAS is restricted to one
thread without changing the inherited CPU affinity.
