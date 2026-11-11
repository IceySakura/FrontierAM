#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <sys/stat.h>

struct Particle {
    double m, x, y, z;
};
struct Acceleration {
    double x, y, z;
};
static_assert(sizeof(Particle) == 32, "Unexpected input layout");
static_assert(sizeof(Acceleration) == 24, "Unexpected output layout");

constexpr int BLOCK = 512;

static inline double horizontal_sum(__m256d v) {
    __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v),
                          _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

static inline __m256d inverse_cube(__m256d r2) {
    __m256d r = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
    __m256d rr = _mm256_mul_pd(r, r);
    __m256d e = _mm256_fnmadd_pd(r2, rr, _mm256_set1_pd(1.0));
    __m256d w = _mm256_mul_pd(rr, r);

    // (1-e)^(-3/2), through e^4. All correction arithmetic is binary64.
    __m256d p = _mm256_fmadd_pd(e, _mm256_set1_pd(315.0 / 128.0),
                               _mm256_set1_pd(35.0 / 16.0));
    p = _mm256_fmadd_pd(e, p, _mm256_set1_pd(15.0 / 8.0));
    p = _mm256_fmadd_pd(e, p, _mm256_set1_pd(1.5));
    return _mm256_fmadd_pd(_mm256_mul_pd(w, e), p, w);
}

static void uniform_forces(int n, const double *x, const double *y,
                           const double *z, double *ax, double *ay,
                           double *az) {
    const __m256d eps2 = _mm256_set1_pd(0.0001);
    for (int ib = 0; ib < n; ib += BLOCK) {
        int ie = std::min(ib + BLOCK, n);
        for (int jb = ib; jb < n; jb += BLOCK) {
            int je = std::min(jb + BLOCK, n);
            if (ib == jb) {
                for (int i = ib; i < ie; ++i) {
                    __m256d sx = _mm256_setzero_pd(), sy = sx, sz = sx;
                    __m256d xi = _mm256_set1_pd(x[i]);
                    __m256d yi = _mm256_set1_pd(y[i]);
                    __m256d zi = _mm256_set1_pd(z[i]);
                    for (int j = jb; j < je; j += 4) {
                        __m256d dx = _mm256_sub_pd(_mm256_load_pd(x + j), xi);
                        __m256d dy = _mm256_sub_pd(_mm256_load_pd(y + j), yi);
                        __m256d dz = _mm256_sub_pd(_mm256_load_pd(z + j), zi);
                        __m256d r2 = _mm256_fmadd_pd(
                            dx, dx, _mm256_fmadd_pd(
                                dy, dy, _mm256_fmadd_pd(dz, dz, eps2)));
                        __m256d w = inverse_cube(r2);
                        sx = _mm256_fmadd_pd(dx, w, sx);
                        sy = _mm256_fmadd_pd(dy, w, sy);
                        sz = _mm256_fmadd_pd(dz, w, sz);
                    }
                    ax[i] += horizontal_sum(sx);
                    ay[i] += horizontal_sum(sy);
                    az[i] += horizontal_sum(sz);
                }
                continue;
            }

            // Sum each source block separately to avoid long accumulation chains.
            alignas(64) double local[3 * BLOCK] = {};
            double *tx = local, *ty = local + BLOCK, *tz = local + 2 * BLOCK;
            for (int i = ib; i < ie; i += 2) {
                __m256d sx[2], sy[2], sz[2];
                for (int k = 0; k < 2; ++k)
                    sx[k] = sy[k] = sz[k] = _mm256_setzero_pd();
                for (int j = jb; j < je; j += 4) {
                    __m256d jx = _mm256_load_pd(x + j);
                    __m256d jy = _mm256_load_pd(y + j);
                    __m256d jz = _mm256_load_pd(z + j);
                    __m256d jax = _mm256_load_pd(tx + j - jb);
                    __m256d jay = _mm256_load_pd(ty + j - jb);
                    __m256d jaz = _mm256_load_pd(tz + j - jb);
                    for (int k = 0; k < 2; ++k) {
                        __m256d dx = _mm256_sub_pd(jx, _mm256_set1_pd(x[i + k]));
                        __m256d dy = _mm256_sub_pd(jy, _mm256_set1_pd(y[i + k]));
                        __m256d dz = _mm256_sub_pd(jz, _mm256_set1_pd(z[i + k]));
                        __m256d r2 = _mm256_fmadd_pd(
                            dx, dx, _mm256_fmadd_pd(
                                dy, dy, _mm256_fmadd_pd(dz, dz, eps2)));
                        __m256d w = inverse_cube(r2);
                        sx[k] = _mm256_fmadd_pd(dx, w, sx[k]);
                        sy[k] = _mm256_fmadd_pd(dy, w, sy[k]);
                        sz[k] = _mm256_fmadd_pd(dz, w, sz[k]);
                        jax = _mm256_fnmadd_pd(dx, w, jax);
                        jay = _mm256_fnmadd_pd(dy, w, jay);
                        jaz = _mm256_fnmadd_pd(dz, w, jaz);
                    }
                    _mm256_store_pd(tx + j - jb, jax);
                    _mm256_store_pd(ty + j - jb, jay);
                    _mm256_store_pd(tz + j - jb, jaz);
                }
                for (int k = 0; k < 2; ++k) {
                    ax[i + k] += horizontal_sum(sx[k]);
                    ay[i + k] += horizontal_sum(sy[k]);
                    az[i + k] += horizontal_sum(sz[k]);
                }
            }
            for (int j = jb; j < je; j += 4) {
                _mm256_store_pd(ax + j, _mm256_add_pd(
                    _mm256_load_pd(ax + j), _mm256_load_pd(tx + j - jb)));
                _mm256_store_pd(ay + j, _mm256_add_pd(
                    _mm256_load_pd(ay + j), _mm256_load_pd(ty + j - jb)));
                _mm256_store_pd(az + j, _mm256_add_pd(
                    _mm256_load_pd(az + j), _mm256_load_pd(tz + j - jb)));
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    FILE *input = std::fopen(argv[1], "rb");
    if (!input) return 2;
    struct stat st;
    if (fstat(fileno(input), &st) || st.st_size <= 0 ||
        st.st_size % sizeof(Particle) != 0 ||
        st.st_size / sizeof(Particle) > std::numeric_limits<int>::max() - 8) {
        std::fclose(input);
        return 3;
    }
    int n = int(st.st_size / sizeof(Particle));
    std::vector<Particle> particles(n);
    bool read_ok = std::fread(particles.data(), sizeof(Particle), n, input) == size_t(n);
    int close_status = std::fclose(input);
    if (!read_ok || close_status) return 4;

    const int padded = (n + 7) & ~7;
    double *memory = static_cast<double *>(
        std::aligned_alloc(64, size_t(padded) * 6 * sizeof(double)));
    if (!memory) return 5;
    double *x = memory, *y = x + padded, *z = y + padded;
    double *ax = z + padded, *ay = ax + padded, *az = ay + padded;
    std::fill(memory, memory + size_t(padded) * 6, 0.0);
    bool uniform = true;
    for (int i = 0; i < n; ++i) {
        x[i] = particles[i].x;
        y[i] = particles[i].y;
        z[i] = particles[i].z;
        if (particles[i].m != particles[0].m) uniform = false;
    }
    bool fast_path = uniform && n % 4 == 0;
    if (fast_path) {
        uniform_forces(n, x, y, z, ax, ay, az);
    } else {
        for (int i = 0; i < n; ++i) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int j = 0; j < n; ++j) {
                double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
                double r2 = dx * dx + dy * dy + dz * dz + 0.0001;
                double w = particles[j].m / (r2 * std::sqrt(r2));
                sx += dx * w;
                sy += dy * w;
                sz += dz * w;
            }
            ax[i] = sx;
            ay[i] = sy;
            az[i] = sz;
        }
    }

    std::vector<Acceleration> result(n);
    double mass = fast_path ? particles[0].m : 1.0;
    for (int i = 0; i < n; ++i)
        result[i] = {ax[i] * mass, ay[i] * mass, az[i] * mass};
    std::free(memory);
    FILE *output = std::fopen(argv[2], "wb");
    if (!output) return 6;
    bool write_ok = std::fwrite(result.data(), sizeof(Acceleration), n, output) == size_t(n);
    close_status = std::fclose(output);
    return write_ok && close_status == 0 ? 0 : 7;
}
