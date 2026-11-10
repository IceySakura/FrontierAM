// Direct summation reference and Plummer sphere generator for FMM paper validation
// Reproduces the Plummer sphere tests of Dehnen (2000, 2002, 2014).

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

// Generate N equal-mass particles (total mass 1, G = 1, scale radius r_s = 1)
// drawn from a Plummer (1911) sphere: M(r) = M_tot * r^3 / (r^2 + r_s^2)^(3/2)
// Inverse CDF: r = r_s / sqrt(u^(-2/3) - 1) for u in (0, 1).
// Following Dehnen (2000) Section 3, optionally truncate at r < r_max (e.g. 1000).
void generate_plummer(int n, uint64_t seed, double r_max, const char* out_path) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> rows(4 * static_cast<size_t>(n));
    const double m = 1.0 / static_cast<double>(n);

    for (int i = 0; i < n; ++i) {
        double r = 0.0;
        for (;;) {
            const double u = uni(rng);
            if (u <= 0.0) continue;
            const double um23 = std::pow(u, -2.0 / 3.0);
            if (um23 <= 1.0) continue;
            r = 1.0 / std::sqrt(um23 - 1.0);
            if (r_max <= 0.0 || r <= r_max) break;
        }
        double dx, dy, dz, d2;
        do {
            dx = norm(rng);
            dy = norm(rng);
            dz = norm(rng);
            d2 = dx * dx + dy * dy + dz * dz;
        } while (d2 == 0.0);
        const double s = r / std::sqrt(d2);
        rows[4 * i + 0] = m;
        rows[4 * i + 1] = dx * s;
        rows[4 * i + 2] = dy * s;
        rows[4 * i + 3] = dz * s;
    }

    FILE* f = std::fopen(out_path, "wb");
    std::fwrite(rows.data(), 32, static_cast<size_t>(n), f);
    std::fclose(f);
}

// Compute exact direct-summation accelerations in double precision (AVX2 accelerated)
void compute_direct(const char* in_path, double eps, const char* ref_path) {
    FILE* in = std::fopen(in_path, "rb");
    std::fseek(in, 0, SEEK_END);
    const long bytes = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    const int n = static_cast<int>(bytes / 32);

    std::vector<double> rows(4 * static_cast<size_t>(n));
    if (std::fread(rows.data(), 32, static_cast<size_t>(n), in) != static_cast<size_t>(n)) {
        std::fclose(in);
        std::exit(1);
    }
    std::fclose(in);

    std::vector<double> x(n), y(n), z(n), m(n);
    for (int i = 0; i < n; ++i) {
        m[i] = rows[4 * i + 0];
        x[i] = rows[4 * i + 1];
        y[i] = rows[4 * i + 2];
        z[i] = rows[4 * i + 3];
    }

    std::vector<double> ax(n, 0.0), ay(n, 0.0), az(n, 0.0);
    const double eps2 = eps * eps;
    const __m256d veps2 = _mm256_set1_pd(eps2);
    const __m256d vone  = _mm256_set1_pd(1.0);

    auto hsum = [](__m256d v) -> double {
        __m128d lo = _mm256_castpd256_pd128(v);
        __m128d hi = _mm256_extractf128_pd(v, 1);
        lo = _mm_add_pd(lo, hi);
        __m128d shuf = _mm_unpackhi_pd(lo, lo);
        return _mm_cvtsd_f64(_mm_add_sd(lo, shuf));
    };

    clock_t t0 = std::clock();
    for (int i = 0; i + 1 < n; ++i) {
        const double xi = x[i], yi = y[i], zi = z[i], mi = m[i];
        const __m256d vxi = _mm256_set1_pd(xi);
        const __m256d vyi = _mm256_set1_pd(yi);
        const __m256d vzi = _mm256_set1_pd(zi);
        const __m256d vmi = _mm256_set1_pd(mi);

        __m256d vaxi = _mm256_setzero_pd();
        __m256d vayi = _mm256_setzero_pd();
        __m256d vazi = _mm256_setzero_pd();

        int j = i + 1;
        for (; j + 4 <= n; j += 4) {
            __m256d vxj = _mm256_loadu_pd(x.data() + j);
            __m256d vyj = _mm256_loadu_pd(y.data() + j);
            __m256d vzj = _mm256_loadu_pd(z.data() + j);
            __m256d vmj = _mm256_loadu_pd(m.data() + j);

            __m256d dx = _mm256_sub_pd(vxi, vxj);
            __m256d dy = _mm256_sub_pd(vyi, vyj);
            __m256d dz = _mm256_sub_pd(vzi, vzj);

            __m256d r2 = _mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dx, dx, veps2)));
            __m256d inv_r = _mm256_div_pd(vone, _mm256_sqrt_pd(r2));
            __m256d inv_r3 = _mm256_mul_pd(inv_r, _mm256_mul_pd(inv_r, inv_r));

            __m256d fi = _mm256_mul_pd(vmj, inv_r3);
            __m256d fj = _mm256_mul_pd(vmi, inv_r3);

            vaxi = _mm256_fnmadd_pd(dx, fi, vaxi);
            vayi = _mm256_fnmadd_pd(dy, fi, vayi);
            vazi = _mm256_fnmadd_pd(dz, fi, vazi);

            __m256d axj = _mm256_loadu_pd(ax.data() + j);
            __m256d ayj = _mm256_loadu_pd(ay.data() + j);
            __m256d azj = _mm256_loadu_pd(az.data() + j);

            axj = _mm256_fmadd_pd(dx, fj, axj);
            ayj = _mm256_fmadd_pd(dy, fj, ayj);
            azj = _mm256_fmadd_pd(dz, fj, azj);

            _mm256_storeu_pd(ax.data() + j, axj);
            _mm256_storeu_pd(ay.data() + j, ayj);
            _mm256_storeu_pd(az.data() + j, azj);
        }

        double axi = hsum(vaxi), ayi = hsum(vayi), azi = hsum(vazi);
        for (; j < n; ++j) {
            const double dx = xi - x[j], dy = yi - y[j], dz = zi - z[j];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            axi -= dx * (m[j] * inv_r3);
            ayi -= dy * (m[j] * inv_r3);
            azi -= dz * (m[j] * inv_r3);
            ax[j] += dx * (mi * inv_r3);
            ay[j] += dy * (mi * inv_r3);
            az[j] += dz * (mi * inv_r3);
        }
        ax[i] += axi;
        ay[i] += ayi;
        az[i] += azi;
    }
    clock_t t1 = std::clock();
    std::fprintf(stderr, "direct N=%d eps=%g cpu=%.4f s\n", n, eps, double(t1 - t0) / CLOCKS_PER_SEC);

    std::vector<double> out(3 * static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[3 * i + 0] = ax[i];
        out[3 * i + 1] = ay[i];
        out[3 * i + 2] = az[i];
    }
    FILE* f = std::fopen(ref_path, "wb");
    std::fwrite(out.data(), 24, static_cast<size_t>(n), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (std::string(argv[1]) == "gen" && argc == 6) {
        generate_plummer(std::atoi(argv[2]), std::strtoull(argv[3], nullptr, 10), std::atof(argv[4]), argv[5]);
        return 0;
    }
    if (std::string(argv[1]) == "direct" && argc == 5) {
        compute_direct(argv[2], std::atof(argv[3]), argv[4]);
        return 0;
    }
    return 2;
}
