/* test_npzio.c - test driver for npzio.
 *
 *   test_npzio INPUT.npz [OUT.npz] [--dump PREFIX]
 *       read INPUT, print sizes/checksums; if OUT given write a "solved" result
 *       with x = 0.5*ones(n), s = b, z = -ones(m), objectives 1.0 / 2.0.
 *       --dump writes raw arrays PREFIX.{Pp,Pi,Px,Ap,Ai,Ax,q,b,ct,cd}.bin
 *   test_npzio --write OUT STATUS N M MASK VAL [OBJP OBJD]
 *       write a result with the vectors named in MASK (letters x,s,z, or "-")
 *       filled with VAL; objectives written if OBJP/OBJD given.
 * Exit code 0 on success, 1 on read/write error (never crashes on bad input).
 */
#include "npzio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int dump(const char *prefix, const char *nm, const void *p, size_t bytes)
{
    char path[4096];
    FILE *f;
    snprintf(path, sizeof path, "%s.%s.bin", prefix, nm);
    f = fopen(path, "wb");
    if (!f) return -1;
    if (bytes && fwrite(p, 1, bytes, f) != bytes) { fclose(f); return -1; }
    return fclose(f);
}

static double *filled(int len, double v)
{
    double *a = (double *)malloc((size_t)(len > 0 ? len : 1) * sizeof(double));
    int i;
    if (!a) exit(3);
    for (i = 0; i < len; i++) a[i] = v;
    return a;
}

static int write_mode(int argc, char **argv)
{
    const char *out, *status, *mask;
    int n, m, rc, has_obj = 0;
    double val, op = 0, od = 0;
    double *x = NULL, *s = NULL, *z = NULL;
    if (argc < 8) { fprintf(stderr, "usage: --write OUT STATUS N M MASK VAL [OBJP OBJD]\n"); return 2; }
    out = argv[2]; status = argv[3]; n = atoi(argv[4]); m = atoi(argv[5]); mask = argv[6];
    val = strtod(argv[7], NULL);
    if (argc >= 10) { has_obj = 1; op = strtod(argv[8], NULL); od = strtod(argv[9], NULL); }
    if (strchr(mask, 'x')) x = filled(n, val);
    if (strchr(mask, 's')) s = filled(m, val);
    if (strchr(mask, 'z')) z = filled(m, val);
    rc = npz_write_result(out, status, x, n, s, z, m, has_obj, op, od);
    free(x); free(s); free(z);
    if (rc) { fprintf(stderr, "write failed\n"); return 1; }
    printf("written %s\n", out);
    return 0;
}

int main(int argc, char **argv)
{
    problem_t p;
    char err[512];
    const char *in = NULL, *out = NULL, *prefix = NULL;
    double t0, t1, sPx = 0, sAx = 0, sq = 0, sb = 0, wPx = 0, wAx = 0;
    unsigned long long iP = 0, iA = 0, cs = 0;
    int i, j, k;

    if (argc >= 2 && strcmp(argv[1], "--write") == 0) return write_mode(argc, argv);
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) prefix = argv[++i];
        else if (!in) in = argv[i];
        else if (!out) out = argv[i];
    }
    if (!in) { fprintf(stderr, "usage: test_npzio INPUT.npz [OUT.npz] [--dump PREFIX]\n"); return 2; }

    t0 = now();
    if (npz_read_problem(in, &p, err, sizeof err) != 0) {
        t1 = now();
        printf("error=%s\n", err);
        printf("read_seconds=%.6f\n", t1 - t0);
        return 1;
    }
    t1 = now();
    for (j = 0; j < p.n; j++) {
        for (k = p.Pp[j]; k < p.Pp[j + 1]; k++) {
            sPx += p.Px[k];
            wPx += p.Px[k] * (double)((k % 7) + 1);
            iP += (unsigned long long)(p.Pi[k] + 1) * (unsigned long long)(j + 1);
        }
        for (k = p.Ap[j]; k < p.Ap[j + 1]; k++) {
            sAx += p.Ax[k];
            wAx += p.Ax[k] * (double)((k % 7) + 1);
            iA += (unsigned long long)(p.Ai[k] + 1) * (unsigned long long)(j + 1);
        }
    }
    for (i = 0; i < p.n; i++) sq += p.q[i];
    for (i = 0; i < p.m; i++) sb += p.b[i];
    for (i = 0; i < p.ncones; i++) cs += (unsigned long long)p.cone_type[i] * 1000ULL + (unsigned long long)p.cone_dim[i];
    printf("n=%d\nm=%d\nnnzP=%d\nnnzA=%d\nncones=%d\noffset=%.17g\n", p.n, p.m, p.Pp[p.n], p.Ap[p.n], p.ncones,
           p.objective_offset);
    printf("sumPx=%.17g\nsumAx=%.17g\nsumq=%.17g\nsumb=%.17g\nwPx=%.17g\nwAx=%.17g\n", sPx, sAx, sq, sb, wPx, wAx);
    printf("idxP=%llu\nidxA=%llu\nconesum=%llu\n", iP, iA, cs);
    printf("read_seconds=%.6f\n", t1 - t0);

    if (prefix) {
        int bad = 0;
        bad |= dump(prefix, "Pp", p.Pp, (size_t)(p.n + 1) * sizeof(int));
        bad |= dump(prefix, "Pi", p.Pi, (size_t)p.Pp[p.n] * sizeof(int));
        bad |= dump(prefix, "Px", p.Px, (size_t)p.Pp[p.n] * sizeof(double));
        bad |= dump(prefix, "Ap", p.Ap, (size_t)(p.n + 1) * sizeof(int));
        bad |= dump(prefix, "Ai", p.Ai, (size_t)p.Ap[p.n] * sizeof(int));
        bad |= dump(prefix, "Ax", p.Ax, (size_t)p.Ap[p.n] * sizeof(double));
        bad |= dump(prefix, "q", p.q, (size_t)p.n * sizeof(double));
        bad |= dump(prefix, "b", p.b, (size_t)p.m * sizeof(double));
        bad |= dump(prefix, "ct", p.cone_type, (size_t)p.ncones * sizeof(int));
        bad |= dump(prefix, "cd", p.cone_dim, (size_t)p.ncones * sizeof(int));
        if (bad) { fprintf(stderr, "dump failed\n"); problem_free(&p); return 1; }
    }

    if (out) {
        double *x = filled(p.n, 0.5), *z = filled(p.m, -1.0);
        double tw0 = now(), tw1;
        int rc = npz_write_result(out, "solved", x, p.n, p.b, z, p.m, 1, 1.0, 2.0);
        tw1 = now();
        free(x); free(z);
        if (rc) { printf("error=write failed\n"); problem_free(&p); return 1; }
        printf("write_seconds=%.6f\n", tw1 - tw0);
    }
    problem_free(&p);
    return 0;
}
