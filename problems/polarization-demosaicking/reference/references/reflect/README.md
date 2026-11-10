# reflect

Call `solution.reconstruct(raw)` with a read-only uint8 mosaic in the sensor phase [[90,45],[135,0]]. Returns four float64 planes in 0°, 45°, 90°, 135° order.

This adapter selects the existing `reflect` branch: extend the mosaic by 6 pixels on every side with NumPy `reflect` (edge pixels not repeated), apply LEIC, then remove only the added border. The original image is scored in full. Even padding preserves the filter phase. The per-scene scale is 255/(max(raw)-min(raw)), or 1 for constant input.

Requires Python 3.11, NumPy 1.26.4 and SciPy 1.12.0. No build step or external data. The unchanged vendored source hashes are in ../source-manifest.json. The LEIC core is the project's port of Liu et al., author revision 5590310ca08b59a93d37f8f9a1da7125378159d4.

B v1.2 assigns its historical 32-scene error 1.1099197825170286 to 5 points. These historical errors do not certify the current 1-second end-to-end deadline.
