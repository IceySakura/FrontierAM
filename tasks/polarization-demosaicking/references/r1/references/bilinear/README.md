# Bilinear reference

Call `solution.reconstruct(raw)` with a read-only uint8 mosaic. The image uses the 2×2 sensor tile `[[90,45],[135,0]]`. The output consists of four full-size float64 planes in the order 0°, 45°, 90°, 135°. The only dependency is NumPy. No build step is required.

The 1-second reference implements the same separable bilinear interpolation and nearest-sample edge extension as the original full-frame reference. It writes samples directly into the final planes in output order, fills missing columns on the sampled rows, and then fills missing rows. uint8 samples, half-integers, and quarter-integers are all exact in float64. The entry point requires neither SciPy nor the LEIC implementation.
