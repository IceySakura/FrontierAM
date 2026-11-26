# Polarization demosaicking

`reconstruct(raw)` accepts the `(1024, 1024)` uint8 mosaic and returns a
`(4, 1024, 1024)` float32 array ordered as 0, 45, 90, 135 degrees.

The method preserves every observed mosaic sample. It first builds a fast
bilinear tentative reconstruction and a cubic upsampling fallback. A
high-frequency correlation statistic across the four tentative channels
controls how much the cross-channel estimate is trusted.

The cross-channel estimate interpolates residuals relative to the tentative
channel mean. Nearby same-channel samples are weighted by spatial distance and
guide similarity. A second pass uses a relaxed guide from the first pass, and a
small Laplacian correction restores high-frequency detail. When the channels
share little high-frequency structure, the cubic fallback is used instead.

Only NumPy and SciPy are required. There is no build step and no per-scene
state.
