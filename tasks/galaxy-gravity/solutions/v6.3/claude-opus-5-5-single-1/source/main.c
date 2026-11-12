#include <stdio.h>
#include <stdlib.h>

void compute_avx2(int n, const double *m, const double *x, const double *y, const double *z, double *out);
void compute_gen(int n, const double *m, const double *x, const double *y, const double *z, double *out);

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("input"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / 32);
    double *raw = malloc((size_t)n * 32 + 32);
    if (n > 0 && fread(raw, 32, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    double *m = malloc(sizeof(double) * (n + 1)), *x = malloc(sizeof(double) * (n + 1));
    double *y = malloc(sizeof(double) * (n + 1)), *z = malloc(sizeof(double) * (n + 1));
    for (int i = 0; i < n; i++) { m[i] = raw[4 * i]; x[i] = raw[4 * i + 1]; y[i] = raw[4 * i + 2]; z[i] = raw[4 * i + 3]; }
    double *out = calloc((size_t)3 * n + 3, sizeof(double));
    if (n > 1) {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) compute_avx2(n, m, x, y, z, out);
        else compute_gen(n, m, x, y, z, out);
    }
    for (long i = 0; i < 3L * n; i++) if (!(out[i] == out[i]) || out[i] > 1e308 || out[i] < -1e308) out[i] = 0.0;
    FILE *g = fopen(argv[2], "wb");
    if (!g) { perror("output"); return 1; }
    if (n > 0 && fwrite(out, 24, n, g) != (size_t)n) { fprintf(stderr, "short write\n"); return 1; }
    if (fclose(g) != 0) { perror("close"); return 1; }
    return 0;
}
