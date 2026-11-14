/* core.c - conic solver core: ADMM (OSQP/SCS style) with preconditioned CG
 * on the reduced KKT matrix M = P + sigma I + rho A'A, warm started.
 * Cones: zero(0), nonnegative(1), soc(2), exponential(3), psd(4).
 * Single threaded, no external deps beyond libm.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define SQRT2 1.4142135623730950488
#define ISQRT2 0.7071067811865475244
#define EXP_E 2.7182818284590452354

typedef struct {
    int n, m;
    int *Pp, *Pi; double *Pv; long Pnnz;
    int *Ap, *Ai; double *Av; long Annz;
    const double *q, *b;
    int nblk; int *btype; long *bstart; int *bdim;
    double c0, budget;
    double *Wpsd, *Vpsd, *Epsd; int maxd;
} Prob;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------------- sparse ops (CSC) ---------------- */
static void spmv(const int *Ap, const int *Ai, const double *Av, int rows, int cols,
                 const double *x, double *out) {
    memset(out, 0, sizeof(double) * (size_t)(rows > 0 ? rows : 1));
    for (int j = 0; j < cols; j++) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (int k = Ap[j]; k < Ap[j + 1]; k++) out[Ai[k]] += Av[k] * xj;
    }
}
static void spmvT(const int *Ap, const int *Ai, const double *Av, int cols,
                  const double *y, double *out) {
    for (int j = 0; j < cols; j++) {
        double s = 0.0;
        for (int k = Ap[j]; k < Ap[j + 1]; k++) s += Av[k] * y[Ai[k]];
        out[j] = s;
    }
}

/* ---------------- Jacobi symmetric eigensolver ---------------- */
static void jacobi_eig(double *A, int d, double *V, double *ev) {
    if (V) {
        memset(V, 0, sizeof(double) * (size_t)d * d);
        for (int i = 0; i < d; i++) V[(size_t)i * d + i] = 1.0;
    }
    double scale = 0.0;
    for (int i = 0; i < d; i++) scale = fmax(scale, fabs(A[(size_t)i * d + i]));
    for (int i = 0; i + 1 < d; i++) scale = fmax(scale, fabs(A[(size_t)i * d + i + 1]));
    double thr = 1e-16 * (scale > 0 ? scale : 1.0);
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < d - 1; p++)
            for (int q = p + 1; q < d; q++) {
                double v = A[(size_t)p * d + q];
                off += v * v;
            }
        if (sqrt(off) <= thr * (double)d) break;
        int rotated = 0;
        for (int p = 0; p < d - 1; p++) {
            for (int q = p + 1; q < d; q++) {
                double apq = A[(size_t)p * d + q];
                if (fabs(apq) <= thr) continue;
                rotated = 1;
                double app = A[(size_t)p * d + p], aqq = A[(size_t)q * d + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e30) t = 0.5 / theta;
                else t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                A[(size_t)p * d + p] = app - t * apq;
                A[(size_t)q * d + q] = aqq + t * apq;
                A[(size_t)p * d + q] = 0.0;
                A[(size_t)q * d + p] = 0.0;
                for (int i = 0; i < p; i++) {
                    double aip = A[(size_t)i * d + p], aiq = A[(size_t)i * d + q];
                    A[(size_t)i * d + p] = c * aip - s * aiq;
                    A[(size_t)i * d + q] = s * aip + c * aiq;
                    A[(size_t)p * d + i] = A[(size_t)i * d + p];
                    A[(size_t)q * d + i] = A[(size_t)i * d + q];
                }
                for (int i = p + 1; i < q; i++) {
                    double api = A[(size_t)p * d + i], aiq = A[(size_t)i * d + q];
                    A[(size_t)p * d + i] = c * api - s * aiq;
                    A[(size_t)i * d + q] = s * api + c * aiq;
                    A[(size_t)i * d + p] = A[(size_t)p * d + i];
                    A[(size_t)q * d + i] = A[(size_t)i * d + q];
                }
                for (int i = q + 1; i < d; i++) {
                    double api = A[(size_t)p * d + i], aqi = A[(size_t)q * d + i];
                    A[(size_t)p * d + i] = c * api - s * aqi;
                    A[(size_t)q * d + i] = s * api + c * aqi;
                    A[(size_t)i * d + p] = A[(size_t)p * d + i];
                    A[(size_t)i * d + q] = A[(size_t)q * d + i];
                }
                if (V) {
                    for (int i = 0; i < d; i++) {
                        double vip = V[(size_t)i * d + p], viq = V[(size_t)i * d + q];
                        V[(size_t)i * d + p] = c * vip - s * viq;
                        V[(size_t)i * d + q] = s * vip + c * viq;
                    }
                }
            }
        }
        if (!rotated) break;
    }
    if (ev) for (int i = 0; i < d; i++) ev[i] = A[(size_t)i * d + i];
}

/* ---------------- exponential cone ---------------- */
static double exp_defect(double a, double b, double c) {
    double ap = a > 0 ? a : 0.0;
    double cm = c < 0 ? -c : 0.0;
    double ab = fabs(b);
    double B = ap > cm ? ap : cm;
    if (ab > B) B = ab;
    if (b > 0.0 && c > 0.0) {
        double v = a - b * (log(c) - log(b));
        if (v < 0) v = 0;
        return v < B ? v : B;
    }
    return B;
}

/* Projection onto K_exp = {(x,y,z): y>0, y e^{x/y} <= z} u {(x,0,z): x<=0,z>=0}.
 * On the smooth boundary, with t = x/y and z = y e^t, stationarity of
 * min ||(x,y,z)-(a,b,c)||^2 s.t. y e^{x/y} - z = 0 gives
 *     x - a = -(z - c) e^t ,   y - b + (z - c) e^t (1 - t) = 0
 * which reduce to  y = (a + c e^t)/(t + e^{2t})  and the scalar equation
 *     g(t) = (a + c e^t)(1 + e^{2t}(1 - t)) - (t + e^{2t})(b + c e^t (1 - t)) = 0.
 * Roots are located by a coarse scan plus bisection; every admissible candidate
 * (point itself, the y=0 face, each root) is evaluated and the closest kept. */
static double expg(double t, double a, double b, double c, double *yout) {
    double e = exp(t), e2 = e * e;
    double den = t + e2;
    *yout = (a + c * e) / den;
    return (a + c * e) * (1.0 + e2 * (1.0 - t)) - den * (b + c * e * (1.0 - t));
}

static void proj_exp(double *v) {
    double a = v[0], b = v[1], c = v[2];
    if (b > 0.0) {
        if (c > 0.0 && a - b * log(c / b) <= 0.0) return;
    } else if (b == 0.0) {
        if (a <= 0.0 && c >= 0.0) return;
    }
    double bx = a < 0 ? a : 0.0, by = 0.0, bz = c > 0 ? c : 0.0;
    double best = (bx - a) * (bx - a) + (by - b) * (by - b) + (bz - c) * (bz - c);
    double ytmp;
    double roots[6]; int nr = 0;
    double prev_t = -40.0, prev_g = expg(prev_t, a, b, c, &ytmp);
    for (int k = 1; k <= 350 && nr < 6; k++) {
        double t = -40.0 + 0.2 * (double)k;
        double g = expg(t, a, b, c, &ytmp);
        if (!isfinite(g)) { prev_t = t; prev_g = g; continue; }
        if (isfinite(prev_g) && ((prev_g > 0) != (g > 0))) {
            double lo = prev_t, hi = t, glo = prev_g;
            for (int it = 0; it < 80; it++) {
                double mid = 0.5 * (lo + hi);
                if (mid == lo || mid == hi) break;
                double gm = expg(mid, a, b, c, &ytmp);
                if ((glo > 0) == (gm > 0)) { lo = mid; glo = gm; } else hi = mid;
            }
            roots[nr++] = 0.5 * (lo + hi);
        }
        prev_t = t; prev_g = g;
    }
    for (int i = 0; i < nr; i++) {
        double t = roots[i];
        double g = expg(t, a, b, c, &ytmp);
        double y = ytmp;
        if (!isfinite(g) || !(y > 0.0) || !isfinite(y)) continue;
        double x = y * t, z2 = y * exp(t);
        if (!isfinite(x) || !isfinite(z2)) continue;
        double d = (x - a) * (x - a) + (y - b) * (y - b) + (z2 - c) * (z2 - c);
        if (d < best) { best = d; bx = x; by = y; bz = z2; }
    }
    v[0] = bx; v[1] = by; v[2] = bz;
}

/* ---------------- cone projection / defects ---------------- */
static void proj_block(double *s, long off, int type, int dim, Prob *P) {
    switch (type) {
    case 0:
        memset(s + off, 0, sizeof(double) * (size_t)dim);
        break;
    case 1:
        for (int i = 0; i < dim; i++) if (s[off + i] < 0) s[off + i] = 0;
        break;
    case 2: {
        double t = s[off];
        double nn = 0;
        for (int i = 1; i < dim; i++) nn += s[off + i] * s[off + i];
        nn = sqrt(nn);
        if (t >= nn) { /* keep */ }
        else if (t <= -nn) memset(s + off, 0, sizeof(double) * (size_t)dim);
        else {
            double f = 0.5 * (1.0 + t / nn);
            s[off] = f * nn;
            for (int i = 1; i < dim; i++) s[off + i] *= f;
        }
        break;
    }
    case 3:
        proj_exp(s + off);
        break;
    case 4: {
        int d = dim;
        double *W = P->Wpsd, *V = P->Vpsd, *ev = P->Epsd;
        long pos = off;
        for (int j = 0; j < d; j++)
            for (int i = 0; i <= j; i++) {
                double val = s[pos++];
                if (i != j) val *= ISQRT2;
                W[(size_t)i * d + j] = val;
                W[(size_t)j * d + i] = val;
            }
        jacobi_eig(W, d, V, ev);
        for (int i = 0; i < d; i++) if (ev[i] < 0) ev[i] = 0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) V[(size_t)i * d + j] *= ev[j];
        for (int i = 0; i < d; i++)
            for (int j = i; j < d; j++) {
                double sum = 0;
                for (int k = 0; k < d; k++) sum += V[(size_t)i * d + k] * V[(size_t)j * d + k];
                W[(size_t)i * d + j] = sum;
            }
        pos = off;
        for (int j = 0; j < d; j++)
            for (int i = 0; i <= j; i++) {
                double val = W[(size_t)i * d + j];
                if (i != j) val *= SQRT2;
                s[pos++] = val;
            }
        break;
    }
    }
}

static void project_cone(double *s, Prob *P) {
    for (int k = 0; k < P->nblk; k++)
        proj_block(s, P->bstart[k], P->btype[k], P->bdim[k], P);
}

static double block_defect(const double *v, long off, int type, int dim, int dual, Prob *P) {
    double nrm = 0;
    for (long i = 0; i < dim; i++) { double t = fabs(v[off + i]); if (t > nrm) nrm = t; }
    double sc = 1.0 / (nrm > 1.0 ? nrm : 1.0);
    switch (type) {
    case 0: return dual ? 0.0 : nrm * sc;
    case 1: {
        double mn = 1e300;
        for (long i = 0; i < dim; i++) { double t = v[off + i] * sc; if (t < mn) mn = t; }
        return mn < 0 ? -mn : 0.0;
    }
    case 2: {
        double t = v[off] * sc, nn = 0;
        for (long i = 1; i < dim; i++) { double w = v[off + i] * sc; nn += w * w; }
        nn = sqrt(nn);
        return nn > t ? nn - t : 0.0;
    }
    case 3: {
        double a = v[off] * sc, b = v[off + 1] * sc, c = v[off + 2] * sc;
        if (dual) return exp_defect(-b, -a, EXP_E * c);
        return exp_defect(a, b, c);
    }
    case 4: {
        int d = dim;
        double *W = P->Wpsd;
        long pos = off;
        for (int j = 0; j < d; j++)
            for (int i = 0; i <= j; i++) {
                double val = v[pos++] * sc;
                if (i != j) val *= ISQRT2;
                W[(size_t)i * d + j] = val;
                W[(size_t)j * d + i] = val;
            }
        jacobi_eig(W, d, NULL, P->Epsd);
        double mn = P->Epsd[0];
        for (int i = 1; i < d; i++) if (P->Epsd[i] < mn) mn = P->Epsd[i];
        return mn < 0 ? -mn : 0.0;
    }
    }
    return 0.0;
}

static double cone_defect(const double *v, int dual, Prob *P) {
    double w = 0;
    for (int k = 0; k < P->nblk; k++) {
        double d = block_defect(v, P->bstart[k], P->btype[k], P->bdim[k], dual, P);
        if (!(d >= 0)) return 1e300;
        if (d > w) w = d;
    }
    return w;
}

static int finite_vec(const double *v, int len) {
    for (int i = 0; i < len; i++) if (!isfinite(v[i])) return 0;
    return 1;
}

static int check_solved(Prob *P, const double *x, const double *s, const double *z,
                        double *Ax, double *Px, double *Atz) {
    if (!finite_vec(x, P->n) || !finite_vec(s, P->m) || !finite_vec(z, P->m)) return 0;
    spmv(P->Ap, P->Ai, P->Av, P->m, P->n, x, Ax);
    spmv(P->Pp, P->Pi, P->Pv, P->n, P->n, x, Px);
    spmvT(P->Ap, P->Ai, P->Av, P->n, z, Atz);
    double nAx = 0, ns = 0, nb = 0, rp = 0;
    for (int i = 0; i < P->m; i++) {
        double t = Ax[i] + s[i] - P->b[i];
        if (fabs(t) > rp) rp = fabs(t);
        if (fabs(Ax[i]) > nAx) nAx = fabs(Ax[i]);
        if (fabs(s[i]) > ns) ns = fabs(s[i]);
        if (fabs(P->b[i]) > nb) nb = fabs(P->b[i]);
    }
    double nPx = 0, nq = 0, nAtz = 0, rd = 0;
    for (int i = 0; i < P->n; i++) {
        double t = Px[i] + P->q[i] + Atz[i];
        if (fabs(t) > rd) rd = fabs(t);
        if (fabs(Px[i]) > nPx) nPx = fabs(Px[i]);
        if (fabs(P->q[i]) > nq) nq = fabs(P->q[i]);
        if (fabs(Atz[i]) > nAtz) nAtz = fabs(Atz[i]);
    }
    double ep = 1.0 + fmax(nAx, fmax(ns, nb));
    double ed = 1.0 + fmax(nPx, fmax(nq, nAtz));
    double tol = 1e-8;
    if (getenv("DBG")) fprintf(stderr, "chk rp=%.3e rd=%.3e ck=%.3e ckd=%.3e", rp/ep, rd/ed, cone_defect(s,0,P), cone_defect(z,1,P));
    if (!(rp / ep <= 0.5 * tol)) return 0;
    if (!(rd / ed <= 0.5 * tol)) return 0;
    if (!(cone_defect(s, 0, P) <= 0.5 * tol)) return 0;
    if (!(cone_defect(z, 1, P) <= 0.5 * tol)) return 0;
    double xPx = 0, qx = 0, bz = 0;
    for (int i = 0; i < P->n; i++) { xPx += x[i] * Px[i]; qx += P->q[i] * x[i]; }
    for (int i = 0; i < P->m; i++) bz += P->b[i] * z[i];
    double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
    double gp = fabs(p0 - d0);
    double mg = fabs(p0) > fabs(d0) ? fabs(p0) : fabs(d0);
    if (mg == 0.0) gp = 0.0;
    if (getenv("DBG")) fprintf(stderr, " gap=%.3e p0=%.3e d0=%.3e\n", gp, p0, d0);
    if (!(gp <= tol || gp / mg <= tol)) return 0;
    return 1;
}

static int check_pinf(Prob *P, const double *z, double *Atz) {
    double nz = 0;
    for (int i = 0; i < P->m; i++) { double t = fabs(z[i]); if (t > nz) nz = t; }
    if (!(nz > 0) || !finite_vec(z, P->m)) return 0;
    double nb = 0;
    for (int i = 0; i < P->m; i++) { double t = fabs(P->b[i]); if (t > nb) nb = t; }
    if (!(nb > 0)) return 0;
    spmvT(P->Ap, P->Ai, P->Av, P->n, z, Atz);
    double nAt = 0;
    for (int i = 0; i < P->n; i++) { double t = fabs(Atz[i]); if (t > nAt) nAt = t; }
    double beta = 0;
    for (int i = 0; i < P->m; i++) beta += (P->b[i] / nb) * (z[i] / nz);
    beta = -beta;
    double tol = 1e-8;
    if (!(beta > tol)) return 0;
    if (!(nAt / nz / beta <= 0.5 * tol)) return 0;
    double *zh = (double *)malloc(sizeof(double) * (size_t)(P->m > 0 ? P->m : 1));
    if (!zh) return 0;
    for (int i = 0; i < P->m; i++) zh[i] = z[i] / nz;
    double d = cone_defect(zh, 1, P);
    free(zh);
    if (!(d / beta <= 0.5 * tol)) return 0;
    return 1;
}

static int check_dinf(Prob *P, const double *x, const double *s, double *Px, double *Ax, double a) {
    double nx = 0;
    for (int i = 0; i < P->n; i++) { double t = fabs(x[i]); if (t > nx) nx = t; }
    if (!(nx > 0) || !finite_vec(x, P->n) || !finite_vec(s, P->m)) return 0;
    double nq = 0;
    for (int i = 0; i < P->n; i++) { double t = fabs(P->q[i]); if (t > nq) nq = t; }
    if (!(nq > 0)) return 0;
    spmv(P->Pp, P->Pi, P->Pv, P->n, P->n, x, Px);
    spmv(P->Ap, P->Ai, P->Av, P->m, P->n, x, Ax);
    double nPx = 0;
    for (int i = 0; i < P->n; i++) { double t = fabs(Px[i]); if (t > nPx) nPx = t; }
    double rr = 0;
    for (int i = 0; i < P->m; i++) { double t = fabs(Ax[i] + s[i] / a); if (t > rr) rr = t; }
    double alpha = 0;
    for (int i = 0; i < P->n; i++) alpha += (P->q[i] / nq) * (x[i] / nx);
    alpha = -alpha;
    double tol = 1e-8;
    if (!(alpha > tol)) return 0;
    if (!(nPx / nx / alpha <= 0.5 * tol)) return 0;
    if (!(rr / nx / alpha <= 0.5 * tol)) return 0;
    double *sc = (double *)malloc(sizeof(double) * (size_t)(P->m > 0 ? P->m : 1));
    if (!sc) return 0;
    for (int i = 0; i < P->m; i++) sc[i] = s[i] / (a * nx);
    double d = cone_defect(sc, 0, P);
    free(sc);
    if (!(d / alpha <= 0.5 * tol)) return 0;
    return 1;
}

/* ---------------- exact dense factorisation path (small n) ---------------- */
#define DENSE_MAX 1200
static int dense_factor(double *L, const Prob *P, double sigma, double rho, double *wm) {
    int n = P->n, m = P->m;
    memset(L, 0, sizeof(double) * (size_t)n * n);
    for (int j = 0; j < n; j++)
        for (int k = P->Pp[j]; k < P->Pp[j + 1]; k++)
            L[(size_t)P->Pi[k] * n + j] += P->Pv[k];
    for (int j = 0; j < n; j++) {
        memset(wm, 0, sizeof(double) * (size_t)(m > 0 ? m : 1));
        for (int k = P->Ap[j]; k < P->Ap[j + 1]; k++) wm[P->Ai[k]] = P->Av[k];
        for (int i = 0; i < n; i++) {
            double t = 0;
            for (int k = P->Ap[i]; k < P->Ap[i + 1]; k++) t += P->Av[k] * wm[P->Ai[k]];
            if (t != 0.0) L[(size_t)i * n + j] += rho * t;
        }
    }
    for (int i = 0; i < n; i++) L[(size_t)i * n + i] += sigma;
    for (int j = 0; j < n; j++) {
        double d = L[(size_t)j * n + j];
        for (int k = 0; k < j; k++) d -= L[(size_t)j * n + k] * L[(size_t)j * n + k];
        if (!(d > 1e-280)) return 0;
        d = sqrt(d);
        L[(size_t)j * n + j] = d;
        double inv = 1.0 / d;
        for (int i = j + 1; i < n; i++) {
            double t = L[(size_t)i * n + j];
            for (int k = 0; k < j; k++) t -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
            L[(size_t)i * n + j] = t * inv;
        }
    }
    return 1;
}
static void dense_solve(const double *L, int n, const double *rhs, double *x) {
    for (int i = 0; i < n; i++) {
        double t = rhs[i];
        for (int k = 0; k < i; k++) t -= L[(size_t)i * n + k] * x[k];
        x[i] = t / L[(size_t)i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double t = x[i];
        for (int k = i + 1; k < n; k++) t -= L[(size_t)k * n + i] * x[k];
        x[i] = t / L[(size_t)i * n + i];
    }
}

/* ---------------- CG ---------------- */
typedef struct {
    double sigma, rho;
    const double *diag;
    double *r, *p, *z, *Ap2, *tmp, *mv;
    Prob *P;
} CGCtx;

static void M_matvec(CGCtx *c, const double *v, double *out) {
    Prob *P = c->P;
    double *w = c->mv;
    spmv(P->Pp, P->Pi, P->Pv, P->n, P->n, v, w);
    for (int i = 0; i < P->n; i++) w[i] += c->sigma * v[i];
    if (c->rho > 0) {
        spmv(P->Ap, P->Ai, P->Av, P->m, P->n, v, c->tmp);
        spmvT(P->Ap, P->Ai, P->Av, P->n, c->tmp, out);
        for (int i = 0; i < P->n; i++) out[i] = w[i] + c->rho * out[i];
    } else {
        for (int i = 0; i < P->n; i++) out[i] = w[i];
    }
}

static void cg_solve(CGCtx *c, const double *rhs, double *x, int maxit, double reltol) {
    int n = c->P->n;
    double nr = 0;
    for (int i = 0; i < n; i++) { double t = fabs(rhs[i]); if (t > nr) nr = t; }
    if (nr == 0) { memset(x, 0, sizeof(double) * (size_t)n); return; }
    M_matvec(c, x, c->r);
    double rz = 0;
    for (int i = 0; i < n; i++) {
        c->r[i] = rhs[i] - c->r[i];
        c->z[i] = c->r[i] / c->diag[i];
        c->p[i] = c->z[i];
        rz += c->r[i] * c->z[i];
    }
    for (int it = 0; it < maxit; it++) {
        double rn = 0;
        for (int i = 0; i < n; i++) { double t = fabs(c->r[i]); if (t > rn) rn = t; }
        if (rn <= reltol * nr) return;
        M_matvec(c, c->p, c->Ap2);
        double pAp = 0;
        for (int i = 0; i < n; i++) pAp += c->p[i] * c->Ap2[i];
        if (!(pAp > 0)) return;
        double al = rz / pAp;
        for (int i = 0; i < n; i++) { x[i] += al * c->p[i]; c->r[i] -= al * c->Ap2[i]; }
        double rz2 = 0;
        for (int i = 0; i < n; i++) {
            c->z[i] = c->r[i] / c->diag[i];
            rz2 += c->r[i] * c->z[i];
        }
        if (!(rz2 > 0)) return;
        double be = rz2 / rz;
        if (!isfinite(be)) return;
        for (int i = 0; i < n; i++) c->p[i] = c->z[i] + be * c->p[i];
        rz = rz2;
    }
}

/* ---------------- main solver ---------------- */
int solve_core(Prob *P, double *x, double *s, double *z, int *status, double budget) {
    int n = P->n, m = P->m;
    *status = 0;
    if (n == 0 && m == 0) { *status = 1; return 0; }
    size_t un = (size_t)(n > 0 ? n : 1), um = (size_t)(m > 0 ? m : 1);

    double aA = 1.0;
    for (long k = 0; k < P->Annz; k++) { double t = fabs(P->Av[k]); if (t > aA) aA = t; }
    if (aA <= 0) aA = 1.0;

    double *Ax = (double *)malloc(sizeof(double) * um);
    double *Pxv = (double *)malloc(sizeof(double) * un);
    double *Atz = (double *)malloc(sizeof(double) * un);
    double *rhs = (double *)malloc(sizeof(double) * un);
    double *xnew = (double *)malloc(sizeof(double) * un);
    double *xh = (double *)malloc(sizeof(double) * un);
    double *tmp = (double *)malloc(sizeof(double) * um);
    double *tmp2 = (double *)malloc(sizeof(double) * un);
    double *diag = (double *)malloc(sizeof(double) * un);
    double *r = (double *)malloc(sizeof(double) * un);
    double *pp = (double *)malloc(sizeof(double) * un);
    double *zz = (double *)malloc(sizeof(double) * un);
    double *Ap2 = (double *)malloc(sizeof(double) * un);
    double *mv = (double *)malloc(sizeof(double) * un);
    double *xprev = (double *)malloc(sizeof(double) * un);
    double *zprev = (double *)malloc(sizeof(double) * um);
    double *scand = (double *)malloc(sizeof(double) * um);
    double *pdiag = (double *)calloc(un, sizeof(double));
    double *col2 = (double *)calloc(un, sizeof(double));
    int use_dense = (n > 0 && n <= DENSE_MAX && (double)n * (double)P->Annz <= 4e7);
    double *Ld = use_dense ? (double *)malloc(sizeof(double) * (size_t)n * n) : NULL;
    double dense_rho = -1.0; int dense_ok = 0;
    if (use_dense && !Ld) use_dense = 0;
    if (!Ax || !Pxv || !Atz || !rhs || !xnew || !xh || !tmp || !tmp2 || !diag || !r ||
        !pp || !zz || !Ap2 || !mv || !xprev || !zprev || !scand || !pdiag || !col2) {
        *status = 0;
        return 1;
    }
    for (int j = 0; j < n; j++)
        for (int k = P->Pp[j]; k < P->Pp[j + 1]; k++)
            if (P->Pi[k] == j) pdiag[j] += fabs(P->Pv[k]);
    for (int j = 0; j < n; j++)
        for (int k = P->Ap[j]; k < P->Ap[j + 1]; k++) col2[j] += P->Av[k] * P->Av[k];
    double qn = 0;
    for (int i = 0; i < n; i++) { double t = fabs(P->q[i]); if (t > qn) qn = t; }
    double bn = 0;
    for (int i = 0; i < m; i++) { double t = fabs(P->b[i]); if (t > bn) bn = t; }

    double sigma = 1e-6;
    double rho = 1.0;

    CGCtx C;
    C.sigma = sigma; C.rho = rho; C.diag = diag; C.r = r; C.p = pp; C.z = zz; C.Ap2 = Ap2;
    C.tmp = tmp; C.mv = mv; C.P = P;

    memset(x, 0, sizeof(double) * un);
    memset(z, 0, sizeof(double) * um);
    if (m > 0) memcpy(s, P->b, sizeof(double) * um);
    project_cone(s, P);

    double t0 = now_sec();
    double alpha = 1.0;
    int dbg = getenv("DBG") != NULL;
    long iter = 0;
    long maxiter = 3000000;
    int restarts = 0;
    double best_res = 1e300; long stall = 0;

    for (; iter < maxiter; iter++) {
        for (int i = 0; i < n; i++) {
            double d = pdiag[i] + sigma + rho * col2[i];
            diag[i] = d > 1e-12 ? d : 1e-12;
        }
        for (int i = 0; i < m; i++) tmp[i] = rho * (P->b[i] - s[i]) - z[i];
        spmvT(P->Ap, P->Ai, P->Av, n, tmp, tmp2);
        for (int i = 0; i < n; i++) rhs[i] = sigma * x[i] - P->q[i] + tmp2[i];

        memcpy(xnew, x, sizeof(double) * un);
        if (use_dense) {
            if (dense_rho != rho || !dense_ok) {
                dense_ok = dense_factor(Ld, P, sigma, rho, tmp);
                dense_rho = rho;
            }
            if (dense_ok) dense_solve(Ld, n, rhs, xnew);
            else { int cgmax = 20 + (int)(iter / 5); if (cgmax > 1500) cgmax = 1500; cg_solve(&C, rhs, xnew, cgmax, 1e-11); }
        } else {
            int cgmax = 20 + (int)(iter / 5); if (cgmax > 1500) cgmax = 1500;
            cg_solve(&C, rhs, xnew, cgmax, 1e-11);
        }

        for (int i = 0; i < n; i++) xh[i] = alpha * xnew[i] + (1.0 - alpha) * x[i];
        spmv(P->Ap, P->Ai, P->Av, m, n, xh, Ax);
        for (int i = 0; i < m; i++) tmp[i] = P->b[i] - Ax[i] - z[i] / rho;
        memcpy(s, tmp, sizeof(double) * um);
        project_cone(s, P);
        for (int i = 0; i < m; i++) z[i] += rho * (Ax[i] + s[i] - P->b[i]);
        memcpy(x, xh, sizeof(double) * un);

        double rp = 0, nAx = 0, ns = 0;
        for (int i = 0; i < m; i++) {
            double t = Ax[i] + s[i] - P->b[i];
            if (fabs(t) > rp) rp = fabs(t);
            if (fabs(Ax[i]) > nAx) nAx = fabs(Ax[i]);
            if (fabs(s[i]) > ns) ns = fabs(s[i]);
        }
        spmv(P->Pp, P->Pi, P->Pv, n, n, x, Pxv);
        spmvT(P->Ap, P->Ai, P->Av, n, z, Atz);
        double rd = 0, nPx = 0, nAtz = 0;
        for (int i = 0; i < n; i++) {
            double t = Pxv[i] + P->q[i] + Atz[i];
            if (fabs(t) > rd) rd = fabs(t);
            if (fabs(Pxv[i]) > nPx) nPx = fabs(Pxv[i]);
            if (fabs(Atz[i]) > nAtz) nAtz = fabs(Atz[i]);
        }
        double ep = 1.0 + fmax(nAx, fmax(ns, bn));
        double ed = 1.0 + fmax(nPx, fmax(qn, nAtz));
        double rp_n = rp / ep, rd_n = rd / ed;

        if (dbg && (iter % 200) == 0) fprintf(stderr, "it=%ld rp=%.3e rd=%.3e rho=%.3e\n", iter, rp_n, rd_n, rho);
        if (rp_n <= 5e-9 && rd_n <= 5e-9) {
            if (check_solved(P, x, s, z, tmp, Pxv, Atz)) { *status = 1; break; }
        }
        if ((iter % 50) == 49) {
            if (rp_n > 10.0 * rd_n && rho < 1e2) rho *= 2.0;
            else if (rd_n > 10.0 * rp_n && rho > 1e-3) rho *= 0.5;
            C.rho = rho;
        }
        if (iter > 40 && (iter % 25) == 0 && rp_n > 1e-7) {
            if (check_pinf(P, z, tmp2)) { *status = 2; break; }
            for (int i = 0; i < m; i++) scand[i] = z[i] - zprev[i];
            if (check_pinf(P, scand, tmp2)) { *status = 2; break; }
            for (int i = 0; i < n; i++) tmp2[i] = x[i] - xprev[i];
            spmv(P->Ap, P->Ai, P->Av, m, n, tmp2, scand);
            for (int i = 0; i < m; i++) scand[i] = -aA * scand[i];
            if (check_dinf(P, tmp2, scand, Pxv, Ax, aA)) { *status = 3; break; }
            spmv(P->Ap, P->Ai, P->Av, m, n, x, scand);
            for (int i = 0; i < m; i++) scand[i] = -aA * scand[i];
            if (check_dinf(P, x, scand, Pxv, Ax, aA)) { *status = 3; break; }
        }
        if ((iter % 25) == 0) {
            memcpy(xprev, x, sizeof(double) * un);
            memcpy(zprev, z, sizeof(double) * um);
        }
        double el = now_sec() - t0;
        if (el > budget) break;
        /* periodic restart to escape stalls */
        {
            double cur = rp_n + rd_n;
            if (cur < best_res) { best_res = cur; stall = 0; }
            else if (++stall > 8000 && restarts < 30) {
                restarts++; stall = 0;
                rho = (restarts % 2) ? rho * 4.0 : rho / 4.0;
                if (rho > 1e2) rho = 1e2;
                if (rho < 1e-3) rho = 1e-3;
                C.rho = rho;
            }
        }
    }
    if (dbg) fprintf(stderr, "exit iter=%ld status=%d\n", iter, *status);
    free(Ax); free(Pxv); free(Atz); free(rhs); free(xnew); free(xh);
    free(tmp); free(tmp2); free(diag); free(r); free(pp); free(zz);
    free(Ap2); free(mv); free(Ld); free(xprev); free(zprev); free(scand); free(pdiag); free(col2);
    return 0;
}

/* ---------------- exported entry ---------------- */
typedef struct {
    long n, m, Pnnz, Annz, nblk;
    const long *Pp, *Pi, *Ap, *Ai, *bstart;
    const double *Pv, *Av, *q, *b;
    const long *btype, *bdim;
    double c0, budget;
} InArgs;

int run_solve(const long *dims, const double *Pv, const long *Pp, const long *Pi, long Pnnz,
              const double *Av, const long *Ap, const long *Ai, long Annz,
              const double *q, const double *b,
              const long *btype, const long *bstart, const long *bdim, long nblk,
              double c0, double budget,
              double *xo, double *so, double *zo, long *status, long *iters) {
    Prob P;
    memset(&P, 0, sizeof(P));
    P.n = (int)dims[0]; P.m = (int)dims[1];
    P.Pnnz = Pnnz; P.Annz = Annz; P.Pv = (double *)Pv; P.Av = (double *)Av;
    P.q = q; P.b = b; P.c0 = c0; P.budget = budget;
    P.nblk = (int)nblk;
    /* convert long index arrays to int */
    int np = P.n + 1, na = P.n + 1;
    P.Pp = (int *)malloc(sizeof(int) * (size_t)np);
    P.Ap = (int *)malloc(sizeof(int) * (size_t)na);
    P.Pi = (int *)malloc(sizeof(int) * (size_t)(Pnnz > 0 ? Pnnz : 1));
    P.Ai = (int *)malloc(sizeof(int) * (size_t)(Annz > 0 ? Annz : 1));
    P.btype = (int *)malloc(sizeof(int) * (size_t)(nblk > 0 ? nblk : 1));
    P.bdim = (int *)malloc(sizeof(int) * (size_t)(nblk > 0 ? nblk : 1));
    P.bstart = (long *)malloc(sizeof(long) * (size_t)(nblk > 0 ? nblk : 1));
    if (!P.Pp || !P.Ap || !P.Pi || !P.Ai || !P.btype || !P.bdim || !P.bstart) { *status = 0; return 1; }
    for (int i = 0; i < np; i++) P.Pp[i] = (int)Pp[i];
    for (int i = 0; i < na; i++) P.Ap[i] = (int)Ap[i];
    for (long i = 0; i < Pnnz; i++) P.Pi[i] = (int)Pi[i];
    for (long i = 0; i < Annz; i++) P.Ai[i] = (int)Ai[i];
    for (long i = 0; i < nblk; i++) { P.btype[i] = (int)btype[i]; P.bdim[i] = (int)bdim[i]; P.bstart[i] = bstart[i]; }
    int maxd = 0;
    for (long i = 0; i < nblk; i++) if (btype[i] == 4 && bdim[i] > maxd) maxd = (int)bdim[i];
    if (maxd > 0) {
        P.maxd = maxd;
        P.Wpsd = (double *)malloc(sizeof(double) * (size_t)maxd * maxd);
        P.Vpsd = (double *)malloc(sizeof(double) * (size_t)maxd * maxd);
        P.Epsd = (double *)malloc(sizeof(double) * (size_t)maxd);
        if (!P.Wpsd || !P.Vpsd || !P.Epsd) { *status = 0; return 1; }
    }
    int st = 0;
    solve_core(&P, xo, so, zo, &st, budget);
    *status = st;
    *iters = 0;
    free(P.Pp); free(P.Ap); free(P.Pi); free(P.Ai); free(P.btype); free(P.bdim); free(P.bstart);
    free(P.Wpsd); free(P.Vpsd); free(P.Epsd);
    return 0;
}
