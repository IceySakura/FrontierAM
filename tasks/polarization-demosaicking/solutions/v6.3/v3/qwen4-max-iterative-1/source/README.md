# Polarization demosaicking

`solution.py` defines `reconstruct(raw)` for the supplied 1024x1024 uint8 mosaic.
It returns a finite `(4, H, W)` float32 array in the order 0, 45, 90, 135 degrees.

The method is a compact learned patch regressor. For each pixel it evaluates a
5x5 raw mosaic patch plus the mosaic phase through a one-hidden-layer MLP in a
square-root intensity domain, squares the result, clips to the sensor range,
and restores the exact 45-degree observations. Weights are stored in
`weights.npz` and were fit on the public development scenes.

Runtime dependencies are NumPy and SciPy only. No build step is required.
