/*
 * Independent C99 Chebyshev action for
 *   y = exp(-t L)b0 + t phi_1(-t L)b1
 *     = b0 + g(L)(b1-Lb0),  g(x)=(1-exp(-t x))/x, g(0)=t.
 *
 * The spectrum of the symmetric PSD L is enclosed by a rounded-up
 * Gershgorin bound.  Scaled Bessel coefficients for exp(-t x) are formed
 * by a stable backward ratio recurrence, and coefficients of g are obtained
 * from x g(x)=1-exp(-t x).  A Chebyshev recurrence then needs one CSR SpMV
 * per retained degree.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SAFETY_FACTOR 8.0L
#define MAX_HEADER 4096
#define MAX_DEGREE 4096

static int n_rows;
static uint64_t *row_ptr;
static uint32_t *col_idx;
static double *a_val;
static double lambda_upper;

static void die(const char *s) {
    fprintf(stderr, "stream_chebyshev: %s\n", s);
    exit(2);
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die("out of memory");
    return p;
}

static void read_exact(void *p, size_t bytes) {
    unsigned char *q = (unsigned char *)p;
    while (bytes) {
        ssize_t got = read(STDIN_FILENO, q, bytes);
        if (got <= 0) die("short input");
        q += (size_t)got;
        bytes -= (size_t)got;
    }
}

static void write_all(const void *p, size_t bytes) {
    const unsigned char *q = (const unsigned char *)p;
    while (bytes) {
        ssize_t put = write(STDOUT_FILENO, q, bytes);
        if (put <= 0) die("short output");
        q += (size_t)put;
        bytes -= (size_t)put;
    }
}

static double json_number(const char *json, const char *key) {
    char needle[32];
    snprintf(needle, sizeof needle, "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p || !(p = strchr(p + strlen(needle), ':'))) die("missing JSON field");
    errno = 0;
    char *end;
    double value = strtod(p + 1, &end);
    if (end == p + 1 || errno) die("invalid JSON number");
    return value;
}

static int vector_is_zero(const double *x) {
    for (int i = 0; i < n_rows; ++i)
        if (x[i] != 0.0) return 0;
    return 1;
}

/* y=Lx.  Accumulation order deliberately follows the supplied CSR. */
static void spmv_l(const double *restrict x, double *restrict y) {
    for (int i = 0; i < n_rows; ++i) {
        double s = 0.0;
        for (uint64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            s += a_val[k] * x[col_idx[k]];
        y[i] = s;
    }
}

/* y=(2L/lambda-I)x. */
static void spmv_scaled(const double *restrict x, double *restrict y) {
    const double alpha = 2.0 / lambda_upper;
    for (int i = 0; i < n_rows; ++i) {
        double s = 0.0;
        for (uint64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            s += a_val[k] * x[col_idx[k]];
        y[i] = alpha * s - x[i];
    }
}

/*
 * Return coefficients q[k] in q(z)=sum q[k]T_k(z), where
 * z=2x/lambda-1 and q(z)=g(x).
 *
 * E_k=exp(-a)I_k(a), a=t lambda/2.  Ratios I_{k+1}/I_k are generated
 * backwards and E_0 is normalized with E_0+2 sum E_k=1.  This avoids
 * overflow for a as large as 50000.  The division recurrence follows
 * (1+z)q=(1-f)/a.
 */
static int make_coefficients(double t, double **q_out, long double **tail_out) {
    long double aa = (long double)t * (long double)lambda_upper * 0.5L;
    int top = (int)ceill(12.0L * sqrtl(aa + 1.0L) + 128.0L);
    if (top < 160) top = 160;
    if (top > MAX_DEGREE - 2) die("spectral interval exceeds coefficient capacity");

    long double *ratio = xmalloc((size_t)(top + 1) * sizeof *ratio);
    long double *e = xmalloc((size_t)(top + 1) * sizeof *e);
    long double *f = xmalloc((size_t)(top + 1) * sizeof *f);
    long double *q = xmalloc((size_t)(top + 1) * sizeof *q);
    long double *tail = xmalloc((size_t)(top + 2) * sizeof *tail);

    ratio[top] = 0.0L;
    for (int k = top; k >= 1; --k)
        ratio[k - 1] = 1.0L / (2.0L * (long double)k / aa + ratio[k]);
    e[0] = 1.0L;
    long double norm = 1.0L;
    for (int k = 1; k <= top; ++k) {
        e[k] = e[k - 1] * ratio[k - 1];
        norm += 2.0L * e[k];
    }
    for (int k = 0; k <= top; ++k) e[k] /= norm;
    f[0] = e[0];
    for (int k = 1; k <= top; ++k)
        f[k] = (k & 1 ? -2.0L : 2.0L) * e[k];

    memset(q, 0, (size_t)(top + 1) * sizeof *q);
    /* q[top]=0 is the asymptotic boundary; omitted E_top is far below
       binary64 roundoff by the 12*sqrt(a)+128 choice. */
    long double q_next = 0.0L, q_here = 0.0L;
    for (int j = top; j >= 2; --j) {
        long double rj = -f[j] / aa;
        long double q_prev = 2.0L * (rj - q_here) - q_next;
        q[j - 1] = q_prev;
        q_next = q_here;
        q_here = q_prev;
    }
    q[0] = -f[1] / aa - q[1] - 0.5L * q[2];

    /* The recurrence above solves (1+z)q=(1-f)/a, i.e. phi_1(-tx).
       Multiply by t to obtain g(x)=t*phi_1(-tx). */
    double *qd = xmalloc((size_t)(top + 1) * sizeof *qd);
    for (int k = 0; k <= top; ++k) qd[k] = (double)((long double)t * q[k]);
    tail[top + 1] = 0.0L;
    for (int k = top; k >= 0; --k)
        tail[k] = tail[k + 1] + (long double)t * fabsl(q[k]);

    free(ratio);
    free(e);
    free(f);
    free(q);
    *q_out = qd;
    *tail_out = tail;
    return top;
}

static int choose_degree(int top, const long double *tail, long double c_norm,
                         double atol, double rtol, long double scale) {
    long double budget = ((long double)atol + (long double)rtol * scale) /
                         SAFETY_FACTOR;
    if (c_norm == 0.0L) return 0;
    long double coefficient_limit = budget / c_norm;
    for (int d = 0; d < top; ++d)
        if (tail[d + 1] <= coefficient_limit) return d;
    return top;
}

static void solve_one(double t, double atol, double rtol, int coeff_top,
                      const double *q, const long double *tail,
                      const double *b0, const double *b1, double *y,
                      double *c, double *v0, double *v1, double *v2) {
    if (t == 0.0 || lambda_upper == 0.0) {
        for (int i = 0; i < n_rows; ++i) y[i] = b0[i] + t * b1[i];
        return;
    }
    spmv_l(b0, c);
    long double c2 = 0.0L, scale = 0.0L;
    for (int i = 0; i < n_rows; ++i) {
        c[i] = b1[i] - c[i];
        c2 += (long double)c[i] * (long double)c[i];
        long double s0 = fabsl((long double)b0[i]);
        long double s1 = fabsl((long double)t * (long double)b1[i]);
        if (s0 > scale) scale = s0;
        if (s1 > scale) scale = s1;
    }
    if (vector_is_zero(c)) {
        memcpy(y, b0, (size_t)n_rows * sizeof *y);
        return;
    }
    int degree = choose_degree(coeff_top, tail, sqrtl(c2), atol, rtol, scale);

    memcpy(v0, c, (size_t)n_rows * sizeof *v0);
    for (int i = 0; i < n_rows; ++i) y[i] = b0[i] + q[0] * v0[i];
    if (degree == 0) return;
    spmv_scaled(v0, v1);
    for (int i = 0; i < n_rows; ++i) y[i] += q[1] * v1[i];
    for (int k = 2; k <= degree; ++k) {
        spmv_scaled(v1, v2);
        for (int i = 0; i < n_rows; ++i) {
            v2[i] = 2.0 * v2[i] - v0[i];
            y[i] += q[k] * v2[i];
        }
        double *tmp = v0; v0 = v1; v1 = v2; v2 = tmp;
    }
}

int main(void) {
    write_all("READY\n", 6);

    char magic[8];
    read_exact(magic, sizeof magic);
    if (memcmp(magic, "FAMEXP2\n", 8) != 0) die("bad magic");
    uint32_t json_len;
    read_exact(&json_len, sizeof json_len);
    if (json_len == 0 || json_len > MAX_HEADER) die("bad header length");
    char json[MAX_HEADER + 1];
    read_exact(json, json_len);
    json[json_len] = '\0';

    n_rows = (int)json_number(json, "n");
    uint64_t nnz = (uint64_t)json_number(json, "nnz");
    int requests = (int)json_number(json, "requests");
    double t = json_number(json, "t");
    double rtol = json_number(json, "rtol");
    double atol = json_number(json, "atol");
    if (n_rows < 1 || nnz > 1000000 || (requests != 1 && requests != 16))
        die("invalid dimensions");

    row_ptr = xmalloc((size_t)(n_rows + 1) * sizeof *row_ptr);
    col_idx = xmalloc((size_t)(nnz ? nnz : 1) * sizeof *col_idx);
    a_val = xmalloc((size_t)(nnz ? nnz : 1) * sizeof *a_val);
    read_exact(row_ptr, (size_t)(n_rows + 1) * sizeof *row_ptr);
    read_exact(col_idx, (size_t)nnz * sizeof *col_idx);
    read_exact(a_val, (size_t)nnz * sizeof *a_val);

    /* PSD Gershgorin: lambda_max <= max_i (L_ii + sum_{j!=i}|L_ij|).
       Sum in long double, then round upward and add a row-length-aware
       binary64 guard. */
    long double bound = 0.0L;
    uint64_t max_row_nnz = 0;
    for (int i = 0; i < n_rows; ++i) {
        long double radius = 0.0L, diagonal = 0.0L;
        uint64_t row_nnz = row_ptr[i + 1] - row_ptr[i];
        if (row_nnz > max_row_nnz) max_row_nnz = row_nnz;
        for (uint64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (col_idx[k] == (uint32_t)i) diagonal = (long double)a_val[k];
            else radius += fabsl((long double)a_val[k]);
        }
        long double edge = diagonal + radius;
        if (edge > bound) bound = edge;
    }
    if (bound < 0.0L) bound = 0.0L;
    lambda_upper = nextafter((double)bound, INFINITY);
    lambda_upper *= 1.0 + (double)(8 + max_row_nnz) * DBL_EPSILON;

    double *coeff = NULL;
    long double *tail = NULL;
    int coeff_top = 0;
    if (t != 0.0 && lambda_upper != 0.0)
        coeff_top = make_coefficients(t, &coeff, &tail);

    size_t bytes = (size_t)n_rows * sizeof(double);
    double *b0 = xmalloc(bytes), *b1 = xmalloc(bytes), *y = xmalloc(bytes);
    double *c = xmalloc(bytes), *v0 = xmalloc(bytes), *v1 = xmalloc(bytes), *v2 = xmalloc(bytes);
    for (int r = 0; r < requests; ++r) {
        read_exact(b0, bytes);
        read_exact(b1, bytes);
        solve_one(t, atol, rtol, coeff_top, coeff, tail, b0, b1, y, c, v0, v1, v2);
        write_all(y, bytes);
    }
    return 0;
}
