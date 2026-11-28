/* Gravitational acceleration on galaxy snapshots.
 *
 * a_i = sum_{j != i} m_j (x_j - x_i) / (|x_j-x_i|^2 + eps^2)^{3/2},  G = 1, eps = 0.01
 *
 * Method: Morton-sorted adaptive octree; every cell carries a Cartesian
 * multipole expansion of order P about its centre; particles are processed
 * leaf-by-leaf.  Cells passing the opening criterion are evaluated through
 * the multipole expansion, the rest are summed directly (exact near field).
 *
 * For a source cell with centre c and particles y_a (u = x - c, D=|u|^2+eps^2):
 *   Phi(x) = sum_{|j|<=P} M_j d^j g(u),  M_j = sum_a m_a (c-y_a)^j / j!
 *   d^j g(u) = Q_j(u) * D^(-1/2-|j|),  Q_j homogeneous of degree |j| (universal)
 * Grouping by degree n and folding Q into the moments:
 *   Phi(x) = sum_n t_n A_n(u),  A_n(u) = sum_m Ahat[n][m] u^m
 *   a(x)   = sum_n ( t_n * grad A_n - (2n+1) t_{n+1} * A_n * u ).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define P 3
#define NM ((P + 1) * (P + 2) * (P + 3) / 6)          /* |{m: |m|<=P}| */
#define NAH ((P + 1) * (P + 2) * (P + 3) * (P + 4) / 24) /* Ahat storage */
#define MAXLEV 20
#define QBITS 20

static int ds[P + 2], aoff[P + 2], degof[NM], jc3[3 * NM];
static double qt[NM * NM];
static double EPS2 = 1e-4;
static int NLEAF = 24;
static double THETA = 0.45;
static int PEFF = P;

static double cputime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void unindex(int j, int *c) { c[0] = jc3[3*j]; c[1] = jc3[3*j+1]; c[2] = jc3[3*j+2]; }
static int mindex(int i, int j, int k) {
    int d = i + j + k;
    return ds[d] + i * (d + 1) - i * (i - 1) / 2 + j;
}

static void init_tables(void) {
    int idx = 0, n = 0;
    for (int d = 0; d <= P; d++) {
        ds[d] = idx;
        for (int i = 0; i <= d; i++)
            for (int j = 0; j <= d - i; j++) {
                jc3[3*idx] = i; jc3[3*idx+1] = j; jc3[3*idx+2] = d - i - j;
                degof[idx] = d; idx++;
            }
    }
    ds[P + 1] = idx;
    if (idx != NM) { fprintf(stderr, "NM mismatch %d %d\n", idx, NM); exit(1); }
    for (int d = 0; d <= P; d++) { aoff[d] = n; n += ds[d + 1]; }
    if (n != NAH) { fprintf(stderr, "NAH mismatch\n"); exit(1); }

    /* Q_{j+e_i} = D * d_i Q_j - (2|j|+1) u_i Q_j,  D = |u|^2+eps^2,  Q_0 = 1.
     * (D * u^m' = u^{m'+ex} + u^{m'+ey} + u^{m'+ez} + eps^2 u^m') */
    double *q = calloc((size_t)NM * NM, sizeof(double));
    q[0] = 1.0; /* Q_{000} = 1 */
    for (int j = 0; j < NM; j++) {
        int dj = degof[j], jc[3], cc[3], mc[3];
        unindex(j, jc);
        for (int ax = 0; ax < 3; ax++) {
            if (dj + 1 > P) continue;
            cc[0] = jc[0]; cc[1] = jc[1]; cc[2] = jc[2]; cc[ax]++;
            int ci = mindex(cc[0], cc[1], cc[2]);
            double *row = q + (size_t)ci * NM;
            for (int m = 0; m < NM; m++) row[m] = 0.0;
            for (int m = 0; m < NM; m++) {
                unindex(m, mc);
                double v = q[(size_t)j * NM + m];
                if (v == 0.0) continue;
                if (mc[ax] >= 1) {
                    double coef = v * (double)mc[ax];
                    for (int i = 0; i < 3; i++) {
                        int t[3]; t[0] = mc[0]; t[1] = mc[1]; t[2] = mc[2];
                        t[ax]--; t[i] += 2; /* D = sum u_i^2 *eps^2 */
                        if (t[0] + t[1] + t[2] <= P) row[mindex(t[0], t[1], t[2])] += coef;
                    }
                    {
                        int t[3]; t[0] = mc[0]; t[1] = mc[1]; t[2] = mc[2]; t[ax]--;
                        row[mindex(t[0], t[1], t[2])] += coef * EPS2;
                    }
                }
                {
                    int t[3]; t[0] = mc[0]; t[1] = mc[1]; t[2] = mc[2]; t[ax]++;
                    if (t[0] + t[1] + t[2] <= P) row[mindex(t[0], t[1], t[2])] -= (double)(2 * dj + 1) * v;
                }
            }
        }
    }
    for (int j = 0; j < NM; j++) {
        int jc[3]; unindex(j, jc);
        double f = 1.0;
        for (int t = 1; t <= jc[0]; t++) f *= t;
        for (int t = 1; t <= jc[1]; t++) f *= t;
        for (int t = 1; t <= jc[2]; t++) f *= t;
        for (int m = 0; m < NM; m++) qt[(size_t)j * NM + m] = q[(size_t)j * NM + m] / f;
    }
    free(q);
}

/* ---------------- input data ---------------- */
static int N;
static double *XM, *XS, *YS, *ZS;   /* masses + sorted coordinates */
static double *AX, *AY, *AZ;
static uint64_t *CODE;
static int *ORD;                    /* ORD[s] = original index of sorted slot s */
static double BX0, BY0, BZ0, BLEN;

/* ---------------- tree ---------------- */
static int ncell, cellcap;
static double *Ccx, *Ccy, *Ccz, *Crho;
static int *Cch, *Clo, *Chi;
static unsigned char *Cleaf;
static double *CA;

static void grow_cells(void) {
    int cap = cellcap ? cellcap * 2 : 1024;
    Ccx = realloc(Ccx, cap * sizeof(double)); Ccy = realloc(Ccy, cap * sizeof(double));
    Ccz = realloc(Ccz, cap * sizeof(double)); Crho = realloc(Crho, cap * sizeof(double));
    Cch = realloc(Cch, cap * 8 * sizeof(int));
    Clo = realloc(Clo, cap * sizeof(int)); Chi = realloc(Chi, cap * sizeof(int));
    Cleaf = realloc(Cleaf, cap * sizeof(unsigned char));
    CA = realloc(CA, (size_t)cap * NAH * sizeof(double));
    if (!Ccx || !Ccy || !Ccz || !Crho || !Cch || !Clo || !Chi || !Cleaf || !CA) {
        fprintf(stderr, "out of memory\n"); exit(1);
    }
    cellcap = cap;
}

/* moments of cell [lo,hi) about its centre */
static void cell_moments(int id) {
    double M[NM];
    double cx = Ccx[id], cy = Ccy[id], cz = Ccz[id];
    int lo = Clo[id], hi = Chi[id];
    double rr = 0.0;
    for (int t = 0; t < NM; t++) M[t] = 0.0;
    for (int i = lo; i < hi; i++) {
        double ux = cx - XS[i], uy = cy - YS[i], uz = cz - ZS[i], m = XM[i];
        double px[P + 1], py[P + 1], pz[P + 1];
        px[0] = 1.0; py[0] = 1.0; pz[0] = 1.0;
        for (int d = 1; d <= P; d++) {
            px[d] = px[d - 1] * ux; py[d] = py[d - 1] * uy; pz[d] = pz[d - 1] * uz;
        }
        int t = 0;
        for (int d = 0; d <= P; d++)
            for (int a = 0; a <= d; a++)
                for (int b = 0; b <= d - a; b++) {
                    int c = d - a - b;
                    M[t++] += m * px[a] * py[b] * pz[c];
                }
        double d2 = ux * ux + uy * uy + uz * uz;
        if (d2 > rr) rr = d2;
    }
    Crho[id] = sqrt(rr);
    double *Ah = CA + (size_t)id * NAH;
    for (int t = 0; t < NAH; t++) Ah[t] = 0.0;
    for (int j = 0; j < NM; j++) {
        int dj = degof[j], lim = ds[dj + 1], base = aoff[dj];
        const double *row = qt + (size_t)j * NM;
        double mj = M[j];
        for (int m = 0; m < lim; m++) Ah[base + m] += mj * row[m];
    }
}

static int build(int lo, int hi, int lev, int ix, int iy, int iz) {
    if (ncell >= cellcap) grow_cells();
    int id = ncell++;
    double s = BLEN / (double)(1u << lev);
    Ccx[id] = BX0 + (ix + 0.5) * s;
    Ccy[id] = BY0 + (iy + 0.5) * s;
    Ccz[id] = BZ0 + (iz + 0.5) * s;
    Clo[id] = lo; Chi[id] = hi;
    for (int k = 0; k < 8; k++) Cch[8 * id + k] = -1;
    if (hi - lo > NLEAF && lev < MAXLEV) {
        int sh = 3 * (QBITS - 1 - lev);
        int cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0}, b[9];
        for (int i = lo; i < hi; i++) cnt[(int)((CODE[i] >> sh) & 7)]++;
        b[0] = lo;
        for (int k = 0; k < 8; k++) b[k + 1] = b[k] + cnt[k];
        for (int k = 0; k < 8; k++)
            if (cnt[k]) {
                int ch = build(b[k], b[k + 1], lev + 1,
                               2 * ix + ((k >> 2) & 1), 2 * iy + ((k >> 1) & 1),
                               2 * iz + (k & 1));
                Cch[8 * id + k] = ch;   /* after possible realloc inside build */
            }
        Cleaf[id] = 0;
    } else {
        Cleaf[id] = 1;
    }
    cell_moments(id);
    return id;
}

/* ---------------- evaluation ---------------- */
static int *farlist, *nearlist, *tstack;
static int farcap = 1 << 16, nearcap = 1 << 16;

/* one source cell -> all particles [i0,i1) */
static inline void m2p(int c, int i0, int i1) {
    const double *Ah = CA + (size_t)c * NAH;
    double ccx = Ccx[c], ccy = Ccy[c], ccz = Ccz[c];
    if (PEFF == 0) {
        for (int i = i0; i < i1; i++) {
            double ux = XS[i] - ccx, uy = YS[i] - ccy, uz = ZS[i] - ccz;
            double D = ux * ux + uy * uy + uz * uz + EPS2, w = Ah[0] / (D * sqrt(D));
            AX[i] -= ux * w; AY[i] -= uy * w; AZ[i] -= uz * w;
        }
        return;
    }
    for (int i = i0; i < i1; i++) {
        double ux = XS[i] - ccx, uy = YS[i] - ccy, uz = ZS[i] - ccz;
        double px[P + 1], py[P + 1], pz[P + 1];
        double mono[NM], monx[NM], mony[NM], monz[NM];
        px[0] = 1.0; py[0] = 1.0; pz[0] = 1.0;
        for (int d = 1; d <= P; d++) {
            px[d] = px[d - 1] * ux; py[d] = py[d - 1] * uy; pz[d] = pz[d - 1] * uz;
        }
        int t = 0;
        for (int d = 0; d <= P; d++)
            for (int a = 0; a <= d; a++)
                for (int b = 0; b <= d - a; b++) {
                    int cc = d - a - b;
                    mono[t] = px[a] * py[b] * pz[cc];
                    monx[t] = a ? a * px[a - 1] * py[b] * pz[cc] : 0.0;
                    mony[t] = b ? px[a] * b * py[b - 1] * pz[cc] : 0.0;
                    monz[t] = cc ? px[a] * py[b] * cc * pz[cc - 1] : 0.0;
                    t++;
                }
        double D = ux * ux + uy * uy + uz * uz + EPS2;
        double t0 = 1.0 / sqrt(D), invD = 1.0 / D, tn[P + 2];
        tn[0] = t0;
        for (int d = 1; d <= P + 1; d++) tn[d] = tn[d - 1] * invD;
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int n = 0; n <= P; n++) {
            const double *ah = Ah + aoff[n];
            int lim = ds[n + 1];
            double sa = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
            for (int m = 0; m < lim; m++) {
                sa += ah[m] * mono[m];
                sx += ah[m] * monx[m];
                sy += ah[m] * mony[m];
                sz += ah[m] * monz[m];
            }
            double c2 = (2 * n + 1) * tn[n + 1] * sa;
            ax += tn[n] * sx - c2 * ux;
            ay += tn[n] * sy - c2 * uy;
            az += tn[n] * sz - c2 * uz;
        }
        AX[i] += ax; AY[i] += ay; AZ[i] += az;
    }
}

static inline void p2p(int c, int i0, int i1) {
    int j0 = Clo[c], j1 = Chi[c];
    for (int i = i0; i < i1; i++) {
        double xi = XS[i], yi = YS[i], zi = ZS[i], axi = 0.0, ayi = 0.0, azi = 0.0;
        for (int j = j0; j < j1; j++) {
            double dx = XS[j] - xi, dy = YS[j] - yi, dz = ZS[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + EPS2;
            double w = XM[j] / (r2 * sqrt(r2));
            axi += dx * w; ayi += dy * w; azi += dz * w;
        }
        AX[i] += axi; AY[i] += ayi; AZ[i] += azi;
    }
}

/* returns #far, #near */
static void traverse(int leaf, int *nfar, int *nnear) {
    double tcx = Ccx[leaf], tcy = Ccy[leaf], tcz = Ccz[leaf], tr = Crho[leaf];
    int sp = 0, nf = 0, nn = 0;
    tstack[sp++] = 0;
    while (sp) {
        int c = tstack[--sp];
        double dx = Ccx[c] - tcx, dy = Ccy[c] - tcy, dz = Ccz[c] - tcz;
        double d = sqrt(dx * dx + dy * dy + dz * dz);
        if (Crho[c] + tr < THETA * d) {
            farlist[nf++] = c;
        } else if (Cleaf[c]) {
            nearlist[nn++] = c;
        } else {
            for (int k = 0; k < 8; k++) {
                int ch = Cch[8 * c + k];
                if (ch >= 0) tstack[sp++] = ch;
            }
        }
    }
    *nfar = nf; *nnear = nn;
}

static void eval_leaf(int leaf) {
    int i0 = Clo[leaf], i1 = Chi[leaf], nf, nn;
    traverse(leaf, &nf, &nn);
    for (int k = 0; k < nn; k++) p2p(nearlist[k], i0, i1);
    for (int k = 0; k < nf; k++) m2p(farlist[k], i0, i1);
}

/* list of leaf cell ids, in tree (morton) order */
static int *leaves; static int nleaves;
static void collect_leaves(int c) {
    if (Cleaf[c]) { leaves[nleaves++] = c; return; }
    for (int k = 0; k < 8; k++) if (Cch[8 * c + k] >= 0) collect_leaves(Cch[8 * c + k]);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: solve IN OUT\n"); return 2; }
    double t0 = cputime();
    double limit = 0.0;
    const char *e = getenv("GALAXY_CPU_LIMIT_S");
    if (e) limit = atof(e);
    if (limit <= 0) limit = 1e9;
    double budget = 0.55 * limit;
    if (getenv("GALAXY_NLEAF")) NLEAF = atoi(getenv("GALAXY_NLEAF"));
    if (getenv("GALAXY_THETA")) THETA = atof(getenv("GALAXY_THETA"));

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("input"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    N = (int)(sz / 32);
    double *buf = malloc((size_t)N * 32);
    if (!buf || fread(buf, 32, N, f) != (size_t)N) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(f);

    XM = malloc(N * sizeof(double)); XS = malloc(N * sizeof(double));
    YS = malloc(N * sizeof(double)); ZS = malloc(N * sizeof(double));
    CODE = malloc(N * sizeof(uint64_t)); ORD = malloc(N * sizeof(int));
    AX = calloc(N, sizeof(double)); AY = calloc(N, sizeof(double)); AZ = calloc(N, sizeof(double));
    if (!XM || !XS || !YS || !ZS || !CODE || !ORD || !AX || !AY || !AZ) { fprintf(stderr, "oom\n"); return 1; }

    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300, z0 = 1e300, z1 = -1e300;
    for (int i = 0; i < N; i++) {
        XM[i] = buf[4 * i]; XS[i] = buf[4 * i + 1]; YS[i] = buf[4 * i + 2]; ZS[i] = buf[4 * i + 3];
        if (XS[i] < x0) x0 = XS[i]; if (XS[i] > x1) x1 = XS[i];
        if (YS[i] < y0) y0 = YS[i]; if (YS[i] > y1) y1 = YS[i];
        if (ZS[i] < z0) z0 = ZS[i]; if (ZS[i] > z1) z1 = ZS[i];
    }
    BX0 = x0 - 1e-9; BY0 = y0 - 1e-9; BZ0 = z0 - 1e-9;
    double lx = x1 - x0 + 2e-9, ly = y1 - y0 + 2e-9, lz = z1 - z0 + 2e-9;
    BLEN = lx > ly ? (lx > lz ? lx : lz) : (ly > lz ? ly : lz);
    if (BLEN <= 0) BLEN = 1.0;
    double inv = (double)((1 << QBITS) - 1) / BLEN;
    for (int i = 0; i < N; i++) {
        int qx = (int)((XS[i] - BX0) * inv), qy = (int)((YS[i] - BY0) * inv), qz = (int)((ZS[i] - BZ0) * inv);
        if (qx < 0) qx = 0; if (qx >= (1 << QBITS)) qx = (1 << QBITS) - 1;
        if (qy < 0) qy = 0; if (qy >= (1 << QBITS)) qy = (1 << QBITS) - 1;
        if (qz < 0) qz = 0; if (qz >= (1 << QBITS)) qz = (1 << QBITS) - 1;
        uint64_t c = 0;
        for (int b = 0; b < QBITS; b++)
            c = (c << 3) | (uint64_t)(((qx >> (QBITS - 1 - b)) & 1) << 2 |
                                      ((qy >> (QBITS - 1 - b)) & 1) << 1 |
                                      ((qz >> (QBITS - 1 - b)) & 1));
        CODE[i] = c; ORD[i] = i;
    }
    /* LSD radix sort of (CODE, ORD) pairs, 4 passes of 16 bits */
    {
        uint64_t *ka = malloc(N * sizeof(uint64_t)), *kb = malloc(N * sizeof(uint64_t));
        int *ia = malloc(N * sizeof(int)), *ib = malloc(N * sizeof(int));
        if (!ka || !kb || !ia || !ib) { fprintf(stderr, "oom\n"); return 1; }
        for (int i = 0; i < N; i++) { ka[i] = CODE[i]; ia[i] = ORD[i]; }
        int cnt[65536];
        for (int pass = 0; pass < 4; pass++) {
            int sh = 16 * pass;
            for (int i = 0; i < 65536; i++) cnt[i] = 0;
            for (int i = 0; i < N; i++) cnt[(int)((ka[i] >> sh) & 0xffff)]++;
            int s = 0;
            for (int i = 0; i < 65536; i++) { int v = cnt[i]; cnt[i] = s; s += v; }
            for (int i = 0; i < N; i++) {
                int p = cnt[(int)((ka[i] >> sh) & 0xffff)]++;
                kb[p] = ka[i]; ib[p] = ia[i];
            }
            uint64_t *tk = ka; ka = kb; kb = tk;
            int *ti = ia; ia = ib; ib = ti;
        }
        double *tx = malloc(N * sizeof(double)), *ty = malloc(N * sizeof(double)), *tz = malloc(N * sizeof(double)), *tm = malloc(N * sizeof(double));
        if (!tx || !ty || !tz || !tm) { fprintf(stderr, "oom\n"); return 1; }
        for (int i = 0; i < N; i++) {
            int o = ia[i];
            CODE[i] = ka[i]; ORD[i] = o;
            tx[i] = XS[o]; ty[i] = YS[o]; tz[i] = ZS[o]; tm[i] = XM[o];
        }
        memcpy(XS, tx, N * sizeof(double)); memcpy(YS, ty, N * sizeof(double));
        memcpy(ZS, tz, N * sizeof(double)); memcpy(XM, tm, N * sizeof(double));
        free(tx); free(ty); free(tz); free(tm); free(ka); free(kb); free(ia); free(ib);
    }

    init_tables();
    cellcap = 0; grow_cells();
    build(0, N, 0, 0, 0, 0);

    leaves = malloc((ncell + 8) * sizeof(int));
    nleaves = 0; collect_leaves(0);
    farcap = ncell + 8; nearcap = ncell + 8;
    farlist = malloc(farcap * sizeof(int));
    nearlist = malloc(nearcap * sizeof(int));
    tstack = malloc(4096 * sizeof(int));
    if (!leaves || !farlist || !nearlist || !tstack) { fprintf(stderr, "oom\n"); return 1; }
    if (getenv("GALAXY_NLEAF")) NLEAF = atoi(getenv("GALAXY_NLEAF"));
    if (getenv("GALAXY_THETA")) THETA = atof(getenv("GALAXY_THETA"));
    /* pick theta (and order) so that the estimated cost fits the cpu budget */
    if (!getenv("GALAXY_THETA")) {
        double used = cputime() - t0;
        static const double thl[] = {0.32, 0.42, 0.54, 0.66, 0.80, 0.92};
        int nth = (int)(sizeof(thl) / sizeof(thl[0]));
        int pe = P;
        int nsamp = nleaves < 24 ? nleaves : 24;
        int step = nleaves / (nsamp ? nsamp : 1);
        if (step < 1) step = 1;
        int ok = 0;
        for (int it = 0; it < nth; it++) {
            THETA = thl[it]; PEFF = P;
            double ts = cputime();
            for (int s = 0; s < nsamp; s++) eval_leaf(leaves[s * step]);
            double dt = cputime() - ts;
            double est = used + dt * (double)nleaves / (double)nsamp;
            if (est * 1.6 < budget) { ok = 1; break; }
        }
        if (!ok) { THETA = thl[nth - 1]; PEFF = 0; }   /* last resort: monopole */
        if (getenv("GALAXY_DEBUG")) fprintf(stderr, "theta=%.3f peff=%d nleaf=%d cpu=%.3f est_ok\n", THETA, PEFF, NLEAF, cputime() - t0);
    }
    for (int i = 0; i < N; i++) { AX[i] = 0.0; AY[i] = 0.0; AZ[i] = 0.0; }
    for (int k = 0; k < nleaves; k++) eval_leaf(leaves[k]);
    if (getenv("GALAXY_DEBUG")) fprintf(stderr, "total cpu=%.3f theta=%.3f ncell=%d\n", cputime() - t0, THETA, ncell);

    /* write in original order */
    {
        double *ob = malloc((size_t)N * 24);
        if (!ob) { fprintf(stderr, "oom\n"); return 1; }
        for (int s = 0; s < N; s++) {
            int o = ORD[s];
            ob[3 * o] = AX[s]; ob[3 * o + 1] = AY[s]; ob[3 * o + 2] = AZ[s];
        }
        FILE *g = fopen(argv[2], "wb");
        if (!g) { perror("output"); return 1; }
        if (fwrite(ob, 24, N, g) != (size_t)N) { fprintf(stderr, "write fail\n"); return 1; }
        fclose(g);
        free(ob);
    }
    return 0;
}
