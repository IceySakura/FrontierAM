/* Full O(N^2) direct sum, AVX2 vectorised over j. Verify against *.ref */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

static const double EPS = 0.01, G = 1.0;

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage direct64 IN REF\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz % 32) { fprintf(stderr, "bad input size %ld\n", sz); return 2; }
    long N = sz / 32;
    double *m = malloc(N * sizeof(double));
    double *X = malloc(N * sizeof(double)), *Y = malloc(N * sizeof(double)), *Z = malloc(N * sizeof(double));
    double *ax = calloc(N, sizeof(double)), *ay = calloc(N, sizeof(double)), *az = calloc(N, sizeof(double));
    if (!m || !X || !Y || !Z || !ax || !ay || !az) { fprintf(stderr, "oom\n"); return 2; }
    for (long i = 0; i < N; ++i) {
        double v[4]; if (fread(v, 32, 1, f) != 1) { fprintf(stderr, "read fail\n"); return 2; }
        m[i] = v[0]; X[i] = v[1]; Y[i] = v[2]; Z[i] = v[3];
    }
    fclose(f);
    clock_t t0 = clock();
    long N4 = N & ~3L;
    for (long i = 0; i < N; ++i) {
        double xi = X[i], yi = Y[i], zi = Z[i];
        __m256 axv = _mm256_setzero_pd(), ayv = _mm256_setzero_pd(), azv = _mm256_setzero_pd();
        __m256 bxi = _mm256_set1_pd(xi), byi = _mm256_set1_pd(yi), bzi = _mm256_set1_pd(zi);
        __m256 skip = _mm256_set1_pd(i * 1.0);
        for (long j = 0; j < N4; j += 4) {
            __m256 xj = _mm256_load_pd(X + j), yj = _mm256_load_pd(Y + j), zj = _mm256_load_pd(Z + j);
            __m256 dx = _mm256_sub_pd(xj, bxi), dy = _mm256_sub_pd(yj, byi), dz = _mm256_sub_pd(zj, bzi);
            __m256 r2 = _mm256_fma_pd(dx, dx, _mm256_mul_pd(dy, dy));
            r2 = _mm256_fma_pd(dz, dz, r2);
            r2 = _mm256_add_pd(r2, _mm256_set1_pd(EPS*EPS));
            __m256 inv = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(r2));
            inv = _mm256_mul_pd(inv, inv); inv = _mm256_mul_pd(inv, inv); inv = _mm256_mul_pd(inv, _mm256_set1_pd(G*m[i]));
            /* mask out self if it fell in this block */
            __m256 jv = _mm256_set_pd(j+3.0, j+2.0, j+1.0, j+0.0);
            __m256d cmp = _mm256_cmp_pd(jv, skip, _CMP_EQ_OQ);
            inv = _mm256_andnot_pd(cmp, inv);
            axv = _mm256_fma_pd(dx, inv, axv);
            ayv = _mm256_fma_pd(dy, inv, ayv);
            azv = _mm256_fma_pd(dz, inv, azv);
        }
        double ta[3] = {0,0,0};
        _mm256_store_pd(ta, axv); _mm256_store_pd(ta+1, ayv); _mm256_store_pd(ta+2, azv);
        /* tail + self correction: recompute tail scalar */
        for (long j = N4; j < N; ++j) {
            if (j == i) continue;
            double dx = X[j]-xi, dy = Y[j]-yi, dz = Z[j]-zi;
            double r2 = dx*dx+dy*dy+dz*dz+EPS*EPS, f = G*m[i]/(r2*sqrt(r2));
            ta[0] += dx*f; ta[1] += dy*f; ta[2] += dz*f;
        }
        /* also fix the masked self element in the vector part */
        long ib = i & ~3L;
        if (ib < N4) {
            long k = i - ib;
            double dx = X[ib+k]-xi;  /* =0 */
            /* the vector accumulator already zeroed it via mask; nothing to add */
        }
        ax[i] = ta[0]; ay[i] = ta[1]; az[i] = ta[2];
    }
    double dt = (double)(clock()-t0)/CLOCKS_PER_SEC;
    fprintf(stderr, "N=%ld cpu=%.3fs\n", N, dt);
    /* compare with reference */
    FILE *g = fopen(argv[2], "rb"); if (!g) { perror(argv[2]); return 2; }
    double maxrel = 0, rms = 0; long bad = 0;
    for (long i = 0; i < N; ++i) {
        double v[3]; if (fread(v, 24, 1, g) != 1) { fprintf(stderr, "ref read fail\n"); return 2; }
        double dx = ax[i]-v[0], dy = ay[i]-v[1], dz = az[i]-v[2];
        double num = sqrt(dx*dx+dy*dy+dz*dz), den = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        double rel = num/den;
        if (!isfinite(rel)) { bad++; continue; }
        rms += rel*rel;
        if (rel > maxrel) maxrel = rel;
    }
    fclose(g);
    fprintf(stderr, "max rel err = %.6e   rms rel err = %.6e   nonfinite %ld\n", maxrel, sqrt(rms/N), bad);
    free(m); free(X); free(Y); free(Z); free(ax); free(ay); free(az);
    return 0;
}
