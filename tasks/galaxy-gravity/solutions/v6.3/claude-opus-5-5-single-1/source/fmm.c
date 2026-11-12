/* Cartesian Taylor-expansion FMM for the Plummer-softened kernel.
 * Far field: dual tree traversal, mutual M2L of order P on an octree.
 * Near field: direct double-precision summation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef PORD
#define PORD 9
#endif
#define P PORD
#define P1 (P + 1)
#define NT ((P + 1) * (P + 2) * (P + 3) / 6)
#define ND (P1 * P1 * P1)
#ifndef NLEAF
#define NLEAF 48
#endif
#ifndef THETA
#define THETA 0.6
#endif

#define EPS2 1e-4
/* number of multiply-adds in one mutual M2L ~ 2*C(P+6,6) */
#define NPAIRM2L (2.0 * (P + 1) * (P + 2) * (P + 3) * (P + 4) * (P + 5) * (P + 6) / 720.0)

#ifdef __AVX2__
#define FN(x) x##_avx2
#else
#define FN(x) x##_gen
#endif

typedef struct {
    double z[3];
    double r;
    int start, end;
    int child[8];
    int nchild;
} Cell;

static int mi[NT][3], moff[NT], ordn[NT], cntUpTo[P + 2], plus1[NT][3];
static double sgn[NT];
/* pair table (n, m) with m <= n componentwise */
static int npair, *pr_n, *pr_m, *pr_off;

static int N, ncells, capcells;
static double *px, *py, *pz, *pm, *fx, *fy, *fz;
static int *idxbuf, *tmpbuf;
static unsigned char *octbuf;
static const double *ox, *oy, *oz, *om;
static Cell *cells;
static double *Mt, *Ms, *Lc;

static void init_tables(void)
{
    int t = 0;
    for (int o = 0; o <= P; o++)
        for (int a = o; a >= 0; a--)
            for (int b = o - a; b >= 0; b--) {
                int c = o - a - b;
                mi[t][0] = a; mi[t][1] = b; mi[t][2] = c;
                moff[t] = (a * P1 + b) * P1 + c;
                ordn[t] = o;
                sgn[t] = (o & 1) ? -1.0 : 1.0;
                t++;
            }
    for (int l = 0; l <= P; l++) cntUpTo[l] = (l + 1) * (l + 2) * (l + 3) / 6;
    cntUpTo[P + 1] = NT;
    static int lut[P1][P1][P1];
    for (int i = 0; i < NT; i++) lut[mi[i][0]][mi[i][1]][mi[i][2]] = i;
    for (int i = 0; i < NT; i++)
        for (int d = 0; d < 3; d++) {
            int a = mi[i][0] + (d == 0), b = mi[i][1] + (d == 1), c = mi[i][2] + (d == 2);
            plus1[i][d] = (a + b + c <= P) ? lut[a][b][c] : -1;
        }
    npair = 0;
    for (int n = 0; n < NT; n++) npair += (mi[n][0] + 1) * (mi[n][1] + 1) * (mi[n][2] + 1);
    pr_n = malloc(sizeof(int) * npair); pr_m = malloc(sizeof(int) * npair); pr_off = malloc(sizeof(int) * npair);
    int k = 0;
    for (int n = 0; n < NT; n++)
        for (int m = 0; m < NT; m++) {
            if (mi[m][0] <= mi[n][0] && mi[m][1] <= mi[n][1] && mi[m][2] <= mi[n][2]) {
                pr_n[k] = n; pr_m[k] = m;
                pr_off[k] = ((mi[n][0] - mi[m][0]) * P1 + (mi[n][1] - mi[m][1])) * P1 + (mi[n][2] - mi[m][2]);
                k++;
            }
        }
}

/* dense table of u^n / n! for |n| <= P */
static void powtab(double ux, double uy, double uz, double *T)
{
    double xa[P1], yb[P1], zc[P1];
    xa[0] = yb[0] = zc[0] = 1.0;
    for (int i = 1; i <= P; i++) {
        xa[i] = xa[i - 1] * ux / i;
        yb[i] = yb[i - 1] * uy / i;
        zc[i] = zc[i - 1] * uz / i;
    }
    for (int a = 0; a <= P; a++)
        for (int b = 0; a + b <= P; b++) {
            double ab = xa[a] * yb[b];
            for (int c = 0; a + b + c <= P; c++) T[(a * P1 + b) * P1 + c] = ab * zc[c];
        }
}

/* D^n G(x) for G = (|x|^2+eps^2)^(-1/2), dense layout, |n| <= P (McMurchie-Davidson) */
static void derivs(double x, double y, double z, double *D)
{
    double q = 1.0 / (x * x + y * y + z * z + EPS2);
    double f[P1];
    f[0] = sqrt(q);
    for (int j = 0; j < P; j++) f[j + 1] = -(2 * j + 1) * q * f[j];
    static double bufA[ND], bufB[ND];
    double *prev = bufA, *cur = bufB;
    prev[0] = f[P];
    for (int j = P - 1; j >= 0; j--) {
        int L = P - j;
        cur[0] = f[j];
        double *out = (j == 0) ? D : cur;
        out[0] = f[j];
        for (int a = 0; a <= L; a++)
            for (int b = 0; a + b <= L; b++)
                for (int c = 0; a + b + c <= L; c++) {
                    if (a + b + c == 0) continue;
                    int o = (a * P1 + b) * P1 + c;
                    double v;
                    if (a > 0) {
                        v = x * prev[o - P1 * P1];
                        if (a > 1) v += (a - 1) * prev[o - 2 * P1 * P1];
                    } else if (b > 0) {
                        v = y * prev[o - P1];
                        if (b > 1) v += (b - 1) * prev[o - 2 * P1];
                    } else {
                        v = z * prev[o - 1];
                        if (c > 1) v += (c - 1) * prev[o - 2];
                    }
                    out[o] = v;
                }
        double *tmp = prev; prev = out; cur = tmp;
        if (j == 0) break;
    }
    if (P == 0) D[0] = f[0];
}

static int newcell(void)
{
    if (ncells == capcells) {
        capcells = capcells * 2 + 64;
        cells = realloc(cells, sizeof(Cell) * capcells);
    }
    return ncells++;
}

static int build(int s, int e, double cx, double cy, double cz, double h, int depth)
{
    int c = newcell();
    cells[c].start = s; cells[c].end = e; cells[c].nchild = 0;
    for (;;) {
        if (e - s <= NLEAF || depth >= 60) return c;
        int cnt[8] = {0};
        for (int i = s; i < e; i++) {
            int id = idxbuf[i];
            int o = (ox[id] > cx) | ((oy[id] > cy) << 1) | ((oz[id] > cz) << 2);
            octbuf[i] = (unsigned char)o; cnt[o]++;
        }
        int ne = 0, last = 0;
        for (int k = 0; k < 8; k++) if (cnt[k]) { ne++; last = k; }
        double hh = 0.5 * h;
        if (ne == 1) {
            cx += (last & 1) ? hh : -hh; cy += (last & 2) ? hh : -hh; cz += (last & 4) ? hh : -hh;
            h = hh; depth++;
            continue;
        }
        int off[8], pos[8];
        off[0] = s;
        for (int k = 1; k < 8; k++) off[k] = off[k - 1] + cnt[k - 1];
        for (int k = 0; k < 8; k++) pos[k] = off[k];
        for (int i = s; i < e; i++) tmpbuf[pos[octbuf[i]]++] = idxbuf[i];
        memcpy(idxbuf + s, tmpbuf + s, sizeof(int) * (e - s));
        for (int k = 0; k < 8; k++)
            if (cnt[k]) {
                int ch = build(off[k], off[k] + cnt[k], cx + ((k & 1) ? hh : -hh), cy + ((k & 2) ? hh : -hh),
                               cz + ((k & 4) ? hh : -hh), hh, depth + 1);
                cells[c].child[cells[c].nchild++] = ch;
            }
        return c;
    }
}

/* near field: targets [ts,te) from sources [ss,se) */
static void p2p(int ts, int te, int ss, int se)
{
#ifdef __AVX2__
    const __m256d eps2 = _mm256_set1_pd(EPS2), one = _mm256_set1_pd(1.0);
    const __m256d c1 = _mm256_set1_pd(0.5), c2 = _mm256_set1_pd(0.375), c3 = _mm256_set1_pd(0.3125),
                  c4 = _mm256_set1_pd(0.2734375), c5 = _mm256_set1_pd(0.24609375);
#define RSQ(r2, y) do { \
        y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2))); \
        __m256d e_ = _mm256_fnmadd_pd(r2, _mm256_mul_pd(y, y), one); \
        __m256d p_ = _mm256_fmadd_pd(c5, e_, c4); \
        p_ = _mm256_fmadd_pd(p_, e_, c3); p_ = _mm256_fmadd_pd(p_, e_, c2); p_ = _mm256_fmadd_pd(p_, e_, c1); \
        y = _mm256_fmadd_pd(_mm256_mul_pd(y, e_), p_, y); } while (0)
    for (int i = ts; i < te; i += 8) {
        __m256d xi = _mm256_loadu_pd(px + i), yi = _mm256_loadu_pd(py + i), zi = _mm256_loadu_pd(pz + i);
        __m256d xk = _mm256_loadu_pd(px + i + 4), yk = _mm256_loadu_pd(py + i + 4), zk = _mm256_loadu_pd(pz + i + 4);
        __m256d axv = _mm256_setzero_pd(), ayv = _mm256_setzero_pd(), azv = _mm256_setzero_pd();
        __m256d bxv = _mm256_setzero_pd(), byv = _mm256_setzero_pd(), bzv = _mm256_setzero_pd();
        for (int j = ss; j < se; j++) {
            __m256d xj = _mm256_broadcast_sd(px + j), yj = _mm256_broadcast_sd(py + j), zj = _mm256_broadcast_sd(pz + j);
            __m256d mj = _mm256_broadcast_sd(pm + j);
            __m256d dx = _mm256_sub_pd(xj, xi), dy = _mm256_sub_pd(yj, yi), dz = _mm256_sub_pd(zj, zi);
            __m256d ex = _mm256_sub_pd(xj, xk), ey = _mm256_sub_pd(yj, yk), ez = _mm256_sub_pd(zj, zk);
            __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, eps2)));
            __m256d s2 = _mm256_fmadd_pd(ex, ex, _mm256_fmadd_pd(ey, ey, _mm256_fmadd_pd(ez, ez, eps2)));
            __m256d y, w;
            RSQ(r2, y);
            RSQ(s2, w);
            __m256d f = _mm256_mul_pd(_mm256_mul_pd(mj, y), _mm256_mul_pd(y, y));
            __m256d g = _mm256_mul_pd(_mm256_mul_pd(mj, w), _mm256_mul_pd(w, w));
            axv = _mm256_fmadd_pd(dx, f, axv); ayv = _mm256_fmadd_pd(dy, f, ayv); azv = _mm256_fmadd_pd(dz, f, azv);
            bxv = _mm256_fmadd_pd(ex, g, bxv); byv = _mm256_fmadd_pd(ey, g, byv); bzv = _mm256_fmadd_pd(ez, g, bzv);
        }
        double tx[8], ty[8], tz[8];
        _mm256_storeu_pd(tx, axv); _mm256_storeu_pd(ty, ayv); _mm256_storeu_pd(tz, azv);
        _mm256_storeu_pd(tx + 4, bxv); _mm256_storeu_pd(ty + 4, byv); _mm256_storeu_pd(tz + 4, bzv);
        int lim = te - i < 8 ? te - i : 8;
        for (int l = 0; l < lim; l++) { fx[i + l] += tx[l]; fy[i + l] += ty[l]; fz[i + l] += tz[l]; }
    }
#else
    for (int i = ts; i < te; i++) {
        double xi = px[i], yi = py[i], zi = pz[i], sx = 0, sy = 0, sz = 0;
        for (int j = ss; j < se; j++) {
            double dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + EPS2;
            double y = 1.0 / sqrt(r2);
            double f = pm[j] * y * y * y;
            sx += dx * f; sy += dy * f; sz += dz * f;
        }
        fx[i] += sx; fy[i] += sy; fz[i] += sz;
    }
#endif
}

static double Dbuf[ND];
static long nm2l, np2p;
static int *qa, *qb;
static long nq, capq;

static void m2l(int A, int B)
{
    if (nq == capq) {
        capq = capq * 2 + 1024;
        qa = realloc(qa, sizeof(int) * capq); qb = realloc(qb, sizeof(int) * capq);
    }
    qa[nq] = A; qb[nq] = B; nq++;
    nm2l++;
}

static void m2l_flush(void)
{
    static double D4[ND * 4] __attribute__((aligned(32)));
    static double MS4[NT * 4] __attribute__((aligned(32))), MT4[NT * 4] __attribute__((aligned(32)));
    static double LA4[NT * 4] __attribute__((aligned(32))), LB4[NT * 4] __attribute__((aligned(32)));
    for (long q = 0; q < nq; q += 4) {
        int cq = (nq - q < 4) ? (int)(nq - q) : 4;
        for (int l = 0; l < 4; l++) {
            if (l < cq) {
                int A = qa[q + l], B = qb[q + l];
                derivs(cells[A].z[0] - cells[B].z[0], cells[A].z[1] - cells[B].z[1], cells[A].z[2] - cells[B].z[2], Dbuf);
                const double *MsB = Ms + (size_t)B * NT, *MtA = Mt + (size_t)A * NT;
                for (int t = 0; t < NT; t++) {
                    D4[moff[t] * 4 + l] = Dbuf[moff[t]];
                    MS4[t * 4 + l] = MsB[t]; MT4[t * 4 + l] = MtA[t];
                }
            } else {
                for (int t = 0; t < NT; t++) { D4[moff[t] * 4 + l] = 0; MS4[t * 4 + l] = 0; MT4[t * 4 + l] = 0; }
            }
        }
        for (int k = 0; k < NT; k++) {
            const double *Dk = D4 + moff[k] * 4;
            int cnt = cntUpTo[P - ordn[k]];
#ifdef __AVX2__
            __m256d s1 = _mm256_setzero_pd(), s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd(), s4 = _mm256_setzero_pd();
            int t = 0;
            for (; t + 1 < cnt; t += 2) {
                __m256d d0 = _mm256_load_pd(Dk + moff[t] * 4), d1 = _mm256_load_pd(Dk + moff[t + 1] * 4);
                s1 = _mm256_fmadd_pd(_mm256_load_pd(MS4 + t * 4), d0, s1);
                s2 = _mm256_fmadd_pd(_mm256_load_pd(MT4 + t * 4), d0, s2);
                s3 = _mm256_fmadd_pd(_mm256_load_pd(MS4 + t * 4 + 4), d1, s3);
                s4 = _mm256_fmadd_pd(_mm256_load_pd(MT4 + t * 4 + 4), d1, s4);
            }
            if (t < cnt) {
                __m256d d0 = _mm256_load_pd(Dk + moff[t] * 4);
                s1 = _mm256_fmadd_pd(_mm256_load_pd(MS4 + t * 4), d0, s1);
                s2 = _mm256_fmadd_pd(_mm256_load_pd(MT4 + t * 4), d0, s2);
            }
            _mm256_store_pd(LA4 + k * 4, _mm256_add_pd(s1, s3));
            _mm256_store_pd(LB4 + k * 4, _mm256_add_pd(s2, s4));
#else
            double s1[4] = {0, 0, 0, 0}, s2[4] = {0, 0, 0, 0};
            for (int t = 0; t < cnt; t++) {
                const double *d = Dk + moff[t] * 4;
                for (int l = 0; l < 4; l++) { s1[l] += MS4[t * 4 + l] * d[l]; s2[l] += MT4[t * 4 + l] * d[l]; }
            }
            for (int l = 0; l < 4; l++) { LA4[k * 4 + l] = s1[l]; LB4[k * 4 + l] = s2[l]; }
#endif
        }
        for (int l = 0; l < cq; l++) {
            double *LA = Lc + (size_t)qa[q + l] * NT, *LB = Lc + (size_t)qb[q + l] * NT;
            for (int k = 0; k < NT; k++) { LA[k] += LA4[k * 4 + l]; LB[k] += sgn[k] * LB4[k * 4 + l]; }
        }
    }
    nq = 0;
}

static void p2p_pair(int A, int B)
{
    p2p(cells[A].start, cells[A].end, cells[B].start, cells[B].end);
    p2p(cells[B].start, cells[B].end, cells[A].start, cells[A].end);
    np2p += 2L * (cells[A].end - cells[A].start) * (cells[B].end - cells[B].start);
}

#ifndef KP2P
#define KP2P 10.0
#endif
static void interact(int A, int B)
{
    double nab = (double)(cells[A].end - cells[A].start) * (cells[B].end - cells[B].start);
    if (nab * KP2P < (double)NPAIRM2L) { p2p_pair(A, B); return; }
    double dx = cells[A].z[0] - cells[B].z[0], dy = cells[A].z[1] - cells[B].z[1], dz = cells[A].z[2] - cells[B].z[2];
    double d2 = dx * dx + dy * dy + dz * dz;
    double rs = cells[A].r + cells[B].r;
    if (rs * rs < THETA * THETA * d2) { m2l(A, B); return; }
    int la = cells[A].nchild == 0, lb = cells[B].nchild == 0;
    if (la && lb) { p2p_pair(A, B); return; }
    if (lb || (!la && cells[A].r >= cells[B].r)) {
        for (int k = 0; k < cells[A].nchild; k++) interact(cells[A].child[k], B);
    } else {
        for (int k = 0; k < cells[B].nchild; k++) interact(A, cells[B].child[k]);
    }
}

static void self(int A)
{
    double na = (double)(cells[A].end - cells[A].start);
    if (cells[A].nchild == 0 || na * na * KP2P * 0.5 < (double)NPAIRM2L) {
        p2p(cells[A].start, cells[A].end, cells[A].start, cells[A].end);
        np2p += (long)na * (long)na;
        return;
    }
    for (int i = 0; i < cells[A].nchild; i++) {
        self(cells[A].child[i]);
        for (int j = i + 1; j < cells[A].nchild; j++) interact(cells[A].child[i], cells[A].child[j]);
    }
}

void FN(compute)(int n, const double *in_m, const double *in_x, const double *in_y, const double *in_z, double *out)
{
    N = n;
    ox = in_x; oy = in_y; oz = in_z; om = in_m;
    init_tables();
    idxbuf = malloc(sizeof(int) * N);
    tmpbuf = malloc(sizeof(int) * N);
    octbuf = malloc(N);
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < N; i++) {
        idxbuf[i] = i;
        double v[3] = {ox[i], oy[i], oz[i]};
        for (int d = 0; d < 3; d++) { if (v[d] < lo[d]) lo[d] = v[d]; if (v[d] > hi[d]) hi[d] = v[d]; }
    }
    double h = 0;
    for (int d = 0; d < 3; d++) if (hi[d] - lo[d] > h) h = hi[d] - lo[d];
    h = 0.5 * h * 1.0001 + 1e-12;
    capcells = 4 * (N / NLEAF + 16);
    cells = malloc(sizeof(Cell) * capcells);
    ncells = 0;
    build(0, N, 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]), h, 0);

    int pad = N + 16;
    px = calloc(pad, sizeof(double)); py = calloc(pad, sizeof(double)); pz = calloc(pad, sizeof(double));
    pm = calloc(pad, sizeof(double));
    fx = calloc(pad, sizeof(double)); fy = calloc(pad, sizeof(double)); fz = calloc(pad, sizeof(double));
    for (int i = 0; i < N; i++) {
        int id = idxbuf[i];
        px[i] = ox[id]; py[i] = oy[id]; pz[i] = oz[id]; pm[i] = om[id];
    }
    Mt = calloc((size_t)ncells * NT, sizeof(double));
    Ms = calloc((size_t)ncells * NT, sizeof(double));
    Lc = calloc((size_t)ncells * NT, sizeof(double));
    /* centers, radii, multipoles: children have larger indices than parents */
    static double T[ND];
    for (int c = ncells - 1; c >= 0; c--) {
        Cell *C = &cells[c];
        double msum = 0, sx = 0, sy = 0, sz = 0;
        double gx = 0, gy = 0, gz = 0;
        for (int i = C->start; i < C->end; i++) {
            msum += pm[i]; sx += pm[i] * px[i]; sy += pm[i] * py[i]; sz += pm[i] * pz[i];
            gx += px[i]; gy += py[i]; gz += pz[i];
        }
        int cntp = C->end - C->start;
        if (msum > 0) { C->z[0] = sx / msum; C->z[1] = sy / msum; C->z[2] = sz / msum; }
        else { C->z[0] = gx / cntp; C->z[1] = gy / cntp; C->z[2] = gz / cntp; }
        double r2 = 0;
        for (int i = C->start; i < C->end; i++) {
            double dx = px[i] - C->z[0], dy = py[i] - C->z[1], dz = pz[i] - C->z[2];
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > r2) r2 = d2;
        }
        C->r = sqrt(r2);
        double *M = Mt + (size_t)c * NT;
        if (C->nchild == 0) {
            for (int i = C->start; i < C->end; i++) {
                powtab(px[i] - C->z[0], py[i] - C->z[1], pz[i] - C->z[2], T);
                for (int t = 0; t < NT; t++) M[t] += pm[i] * T[moff[t]];
            }
        } else {
            for (int k = 0; k < C->nchild; k++) {
                int ch = C->child[k];
                const double *Mc = Mt + (size_t)ch * NT;
                powtab(cells[ch].z[0] - C->z[0], cells[ch].z[1] - C->z[1], cells[ch].z[2] - C->z[2], T);
                for (int q = 0; q < npair; q++) M[pr_n[q]] += Mc[pr_m[q]] * T[pr_off[q]];
            }
        }
        double *S = Ms + (size_t)c * NT;
        for (int t = 0; t < NT; t++) S[t] = sgn[t] * M[t];
    }
    self(0);
    m2l_flush();
    /* downward pass */
    for (int c = 0; c < ncells; c++) {
        Cell *C = &cells[c];
        const double *Lp = Lc + (size_t)c * NT;
        for (int k = 0; k < C->nchild; k++) {
            int ch = C->child[k];
            double *Lch = Lc + (size_t)ch * NT;
            powtab(cells[ch].z[0] - C->z[0], cells[ch].z[1] - C->z[1], cells[ch].z[2] - C->z[2], T);
            for (int q = 0; q < npair; q++) Lch[pr_m[q]] += Lp[pr_n[q]] * T[pr_off[q]];
        }
        if (C->nchild == 0) {
            for (int i = C->start; i < C->end; i++) {
                powtab(px[i] - C->z[0], py[i] - C->z[1], pz[i] - C->z[2], T);
                double sx = 0, sy = 0, sz = 0;
                for (int t = 0; t < cntUpTo[P - 1]; t++) {
                    double u = T[moff[t]];
                    sx += Lp[plus1[t][0]] * u; sy += Lp[plus1[t][1]] * u; sz += Lp[plus1[t][2]] * u;
                }
                fx[i] += sx; fy[i] += sy; fz[i] += sz;
            }
        }
    }
    for (int i = 0; i < N; i++) {
        int id = idxbuf[i];
        out[3 * id] = fx[i]; out[3 * id + 1] = fy[i]; out[3 * id + 2] = fz[i];
    }
    if (getenv("FMM_STATS"))
        fprintf(stderr, "P=%d NLEAF=%d theta=%g cells=%d m2l=%ld p2p=%ld (%.1f per particle)\n", P, NLEAF, THETA, ncells,
                nm2l, np2p, (double)np2p / N);
}
