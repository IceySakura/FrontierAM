#include <stddef.h>

void rotate_layers(double *a, int n, int k, int depth,
                   const double *cosines, const double *sines,
                   const int *bits, int transpose)
{
    for (int layer = 0; layer < depth; ++layer) {
        int j = transpose ? layer : depth - 1 - layer;
        int bit = bits[j];
        const double *cs = cosines + (size_t)j * (n / 2);
        const double *sn = sines + (size_t)j * (n / 2);
        int row = 0;
        for (int block = 0; block < n; block += 2 * bit) {
            for (int offset = 0; offset < bit; ++offset, ++row) {
                double c = cs[row];
                double s = transpose ? -sn[row] : sn[row];
                double *lo = a + (size_t)(block + offset) * k;
                double *hi = a + (size_t)(block + offset + bit) * k;
                #pragma omp simd
                for (int col = 0; col < k; ++col) {
                    double x = lo[col], y = hi[col];
                    lo[col] = c * x - s * y;
                    hi[col] = s * x + c * y;
                }
            }
        }
    }
}
