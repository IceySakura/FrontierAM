# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)`. It returns a contiguous NumPy
float32 array of shape `(4, H, W)`, ordered 0, 45, 90, 135 degrees. Observed
samples are retained exactly, and predictions are clipped to `[0, 255]`.

The method predicts residual corrections to bilinear interpolation with a
compact convolutional network. Measurements are locally centered separately
for each mosaic phase. All parameters in `weights.npz` are fixed and reusable
across scenes. Training used augmented development patches and synthetic
polarization variations; runtime uses only the supplied mosaic and these
parameters. There are no scene lookups or cross-scene state.

Run `./build.sh` once to generate native convolution kernels and Python
bytecode. The build is entirely offline, uses Python and NumPy, and needs no
C compiler or package downloads. Missing kernel files are generated in memory.

Inference uses NumPy and single-threaded x86-64 AVX2/FMA kernels from
`native.py`, implementing ordinary float32 convolution and ReLU. It inherits
CPU affinity and does not start worker threads. On CPUs without AVX2/FMA, or
when executable memory is unavailable, it uses fast NumPy cubic interpolation.
SciPy and PyTorch are not runtime dependencies.
