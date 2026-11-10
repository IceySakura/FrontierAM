# bilinear-edge

Call `solution.reconstruct(raw)` with a read-only uint8 mosaic. The image uses the 2×2 sensor tile `[[90,45],[135,0]]`. Returns four float64 planes in the order 0°, 45°, 90°, 135°. Requires Python 3.11, NumPy 1.26.4 and SciPy 1.12.0. No build step or external data.

The adapter selects the `bilinear-edge` branch from the frozen full-frame implementation. Its dependencies are vendored unchanged under their original relative layout. The LEIC core is the project's NumPy/SciPy port of Liu et al. (2025), author revision `5590310ca08b59a93d37f8f9a1da7125378159d4`.
