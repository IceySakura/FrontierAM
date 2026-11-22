# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)` for a `(1024, 1024)` NumPy uint8 array.
The only runtime dependency is NumPy.
No build step, network access, external data, or persistent state is needed.
Keep `weights.npz` next to `solution.py`.

The input is split into its four half-resolution sampling phases. Four
convolutional layers with 32 hidden features refine independent-channel
bilinear interpolation. The kernels are 5x5, 3x3, 3x3, and 3x3, with ReLU
after the first three layers. The first-layer filters have zero spatial sum
for each input plane, making the residual insensitive to constant differences
between polarization channels. Predictions are interleaved into full-resolution
planes in the order 0, 45, 90, 135 degrees. All observed samples are preserved exactly.
The output is a contiguous finite float32 array with shape `(4, H, W)`.
The mosaic is reflected with even-width padding to preserve its sampling phase
at image boundaries.

The fixed convolution parameters were trained offline using only the five public
development scenes, with spatial transformations, polarization-channel
permutations, relative motion, and intensity augmentation. Training used seed 1;
the exported parameters are an exponential moving average. Inference is
deterministic. Parameters are shared by all scenes; there is no scene
identification or reference-data access at inference time.

Valid convolutions consume the reflected border without padding intermediate
feature maps. Tiled NumPy matrix multiplication bounds temporary memory.
BLAS is restricted to one thread, and inherited CPU affinity is unchanged.

## Development Checks

The public checker reports mean RMSE **1.340569293**. This is a development
measurement, not a held-out score.

Fresh-process checks covered all five public inputs and six synthetic cases,
twice each, with a separate clean submission copy and working directory for each
run. These included startup, imports, input loading, output writing, and extra
contract checks. Median wall time was approximately 0.60 seconds, with a
0.71-second maximum and approximately 113 MiB peak memory. All outputs
were finite float32 arrays, observed samples were exact, inputs were unchanged,
and the processes used one thread. Local timing is not an evaluator guarantee.
