/* FAMEXP2 stream adapter around Expokit (R. B. Sidje, ACM TOMS 24(1):130-156, 1998).
 *
 * y = exp(-tL) b0 + t*phi1(-tL) b1 is the solution of w' = A w + u, w(0) = v with A = -L, u = b1, v = b0,
 * which is exactly what Expokit's DSPHIV computes for a symmetric A. The vendored Fortran is unmodified;
 * this file only speaks the wire protocol, supplies the CSR product and applies the frozen policy:
 *   - t = 0, an all-zero operator or b0 = b1 = 0: y = b0 + t*b1 exactly;
 *   - n <= 64: Expokit DGPADM (Pade, degree 6) on the augmented matrix [[A, u], [0, 0]];
 *   - otherwise DSPHIV with the author's default Krylov size m = 30 (m < n) and tol = TOL_MULTIPLIER * rtol.
 * Expokit prints some diagnostics with Fortran PRINT, so protocol bytes go to a private descriptor and fd 1 is
 * pointed at stderr before any Fortran runs.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define KRYLOV_DEFAULT 30
#define TOL_MULTIPLIER_DEFAULT 1000.0 /* selected by calibrate.py on public dev cases; see calibration.json */
#define DENSE_MAX 64
#define PADE_DEGREE 6

extern void dsphiv_(int *n, int *m, double *t, double *u, double *v, double *w, double *tol, double *anorm,
                    double *wsp, int *lwsp, int *iwsp, int *liwsp, void (*matvec)(double *, double *),
                    int *itrace, int *iflag);
extern void dgpadm_(int *ideg, int *m, double *t, double *h, int *ldh, double *wsp, int *lwsp, int *ipiv,
                    int *iexph, int *ns, int *iflag);

static int n_rows;
static int64_t *row_ptr;
static uint32_t *col_idx;
static double *values;

static void fail(const char *message) {
    fprintf(stderr, "stream_expokit: %s\n", message);
    exit(2);
}

static void read_exact(void *buffer, size_t length) {
    char *cursor = buffer;
    while (length > 0) {
        ssize_t got = read(0, cursor, length);
        if (got <= 0) fail("short input");
        cursor += got;
        length -= (size_t)got;
    }
}

static void write_all(int fd, const void *buffer, size_t length) {
    const char *cursor = buffer;
    while (length > 0) {
        ssize_t put = write(fd, cursor, length);
        if (put <= 0) fail("short output");
        cursor += put;
        length -= (size_t)put;
    }
}

static double header_number(const char *json, const char *key) {
    char quoted[32];
    snprintf(quoted, sizeof quoted, "\"%s\"", key);
    const char *at = strstr(json, quoted);
    if (!at) fail("missing header field");
    at = strchr(at + strlen(quoted), ':');
    if (!at) fail("malformed header");
    return strtod(at + 1, NULL);
}

/* Expokit's matvec callback: y <- A x with A = -L. */
static void matvec(double *x, double *y) {
    for (int i = 0; i < n_rows; i++) {
        double sum = 0.0;
        for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) sum += values[k] * x[col_idx[k]];
        y[i] = -sum;
    }
}

static int all_zero(const double *x, int64_t count) {
    for (int64_t i = 0; i < count; i++)
        if (x[i] != 0.0) return 0;
    return 1;
}

static void dense(double t, const double *b0, const double *b1, double *y) {
    int size = n_rows + 1, ldh = size, ideg = PADE_DEGREE, iexph, ns, iflag;
    int lwsp = 4 * size * size + ideg + 1;
    double *h = calloc((size_t)size * size, sizeof *h), *wsp = malloc((size_t)lwsp * sizeof *wsp);
    int *ipiv = malloc((size_t)size * sizeof *ipiv);
    if (!h || !wsp || !ipiv) fail("out of memory");
    for (int i = 0; i < n_rows; i++) {
        for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) h[(size_t)col_idx[k] * size + i] -= values[k];
        h[(size_t)n_rows * size + i] = b1[i];
    }
    dgpadm_(&ideg, &size, &t, h, &ldh, wsp, &lwsp, ipiv, &iexph, &ns, &iflag);
    if (iflag != 0) fail("DGPADM failed");
    const double *e = wsp + iexph - 1;
    for (int i = 0; i < n_rows; i++) {
        double sum = e[(size_t)n_rows * size + i];
        for (int j = 0; j < n_rows; j++) sum += e[(size_t)j * size + i] * b0[j];
        y[i] = sum;
    }
    free(h), free(wsp), free(ipiv);
}

int main(int argc, char **argv) {
    double multiplier = TOL_MULTIPLIER_DEFAULT;
    int krylov = KRYLOV_DEFAULT;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--tol-multiplier")) multiplier = strtod(argv[i + 1], NULL);
        else if (!strcmp(argv[i], "--krylov")) krylov = atoi(argv[i + 1]);
        else fail("unknown argument");
    }
    int out = dup(1);
    if (out < 0 || dup2(2, 1) < 0) fail("cannot reserve protocol descriptor");
    write_all(out, "READY\n", 6);

    char magic[8];
    read_exact(magic, 8);
    if (memcmp(magic, "FAMEXP2\n", 8)) fail("wrong magic");
    uint32_t length;
    read_exact(&length, 4);
    if (length < 1 || length > 4096) fail("bad header length");
    char json[4097];
    read_exact(json, length);
    json[length] = '\0';
    n_rows = (int)header_number(json, "n");
    int64_t nnz = (int64_t)header_number(json, "nnz");
    int requests = (int)header_number(json, "requests");
    double t = header_number(json, "t"), rtol = header_number(json, "rtol");

    uint64_t *raw_ptr = malloc((size_t)(n_rows + 1) * 8);
    row_ptr = malloc((size_t)(n_rows + 1) * sizeof *row_ptr);
    col_idx = malloc((size_t)(nnz ? nnz : 1) * sizeof *col_idx);
    values = malloc((size_t)(nnz ? nnz : 1) * sizeof *values);
    double *b0 = malloc((size_t)n_rows * sizeof *b0), *b1 = malloc((size_t)n_rows * sizeof *b1);
    double *y = malloc((size_t)n_rows * sizeof *y);
    if (!raw_ptr || !row_ptr || !col_idx || !values || !b0 || !b1 || !y) fail("out of memory");
    read_exact(raw_ptr, (size_t)(n_rows + 1) * 8);
    for (int i = 0; i <= n_rows; i++) row_ptr[i] = (int64_t)raw_ptr[i];
    read_exact(col_idx, (size_t)nnz * 4);
    read_exact(values, (size_t)nnz * 8);
    int zero_operator = all_zero(values, nnz);

    double anorm = 0.0;
    for (int i = 0; i < n_rows; i++) {
        double row = 0.0;
        for (int64_t k = row_ptr[i]; k < row_ptr[i + 1]; k++) row += fabs(values[k]);
        if (row > anorm) anorm = row;
    }
    int m = krylov < n_rows - 1 ? krylov : n_rows - 1;
    int lwsp = n_rows * (m + 3) + 5 * (m + 3) * (m + 3) + PADE_DEGREE + 1, liwsp = m + 3;
    double *wsp = n_rows > DENSE_MAX ? malloc((size_t)lwsp * sizeof *wsp) : NULL;
    int *iwsp = n_rows > DENSE_MAX ? malloc((size_t)liwsp * sizeof *iwsp) : NULL;
    if (n_rows > DENSE_MAX && (!wsp || !iwsp)) fail("out of memory");

    for (int r = 0; r < requests; r++) {
        read_exact(b0, (size_t)n_rows * 8);
        read_exact(b1, (size_t)n_rows * 8);
        if (t == 0.0 || zero_operator || (all_zero(b0, n_rows) && all_zero(b1, n_rows))) {
            for (int i = 0; i < n_rows; i++) y[i] = b0[i] + t * b1[i];
        } else if (n_rows <= DENSE_MAX) {
            dense(t, b0, b1, y);
        } else {
            double tol = multiplier * rtol;
            int itrace = 0, iflag = 0;
            dsphiv_(&n_rows, &m, &t, b1, b0, y, &tol, &anorm, wsp, &lwsp, iwsp, &liwsp, matvec, &itrace, &iflag);
            if (iflag != 0) {
                fprintf(stderr, "stream_expokit: DSPHIV iflag=%d\n", iflag);
                exit(3);
            }
        }
        write_all(out, y, (size_t)n_rows * 8);
    }
    return 0;
}
