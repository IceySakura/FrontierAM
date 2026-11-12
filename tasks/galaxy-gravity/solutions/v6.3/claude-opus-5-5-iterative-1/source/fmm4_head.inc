// Cartesian Taylor FMM for Plummer-softened gravity (eps = 0.01, G = 1). AVX2 version.
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdint>

static double cpu_now() {
    timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static const double EPS2 = 1e-4;

// ---------------- multi-index tables ----------------
struct Tables {
    int p, nc;
    std::vector<int> ma, mb, mc, deg, ncdeg, idx3;
    std::vector<int> sumoff;
    std::vector<int16_t> sumidx;
    std::vector<int> sh_hi, sh_lo, sh_df;
    // derivative recursion plan: for each k>=1: dir, k1, k2, coef
    std::vector<int> rk1, rk2, rdir; std::vector<double> rcoef;
    std::vector<double> rinv;
    std::vector<int> l2p_x, l2p_y, l2p_z;
    std::vector<uint64_t> l2p_pk; std::vector<__m256d> sgnmask;
    std::vector<int> roff; int rtot;                 // compact R storage offsets (m-minor)
    std::vector<int> blk_n;                          // per block: 4 n indices (padded)
    std::vector<int> blk_nv, blk_cnt, blk_off;      // valid count, m count, offset into blk_idx
    std::vector<uint16_t> blk_idx;                    // [m][4] indices of n+m
    std::vector<int> b6_n, b6_cnt, b6_off;           // 6-wide blocks
    std::vector<int> s1_dst[3], s1_src[3], s1_pw[3];   // M2M 1D: out[dst] += in[src]*pw[pow]
    std::vector<int> s2_dst[3], s2_src[3], s2_pw[3];   // L2L 1D
    std::vector<uint16_t> b6_idx;                    // [m][8] byte offsets (6 used)
    int id(int a, int b, int c) const { return idx3[(a * (p + 1) + b) * (p + 1) + c]; }
    void init(int p_) {
        p = p_;
        nc = (p + 1) * (p + 2) * (p + 3) / 6;
        ma.clear(); mb.clear(); mc.clear(); deg.clear(); sumidx.clear(); sh_hi.clear(); sh_lo.clear(); sh_df.clear();
        idx3.assign((p + 2) * (p + 2) * (p + 2), -1);
        ncdeg.resize(p + 1);
        for (int n = 0; n <= p; n++) {
            for (int a = n; a >= 0; a--)
                for (int b = n - a; b >= 0; b--) {
                    int c = n - a - b;
                    idx3[(a * (p + 1) + b) * (p + 1) + c] = ma.size();
                    ma.push_back(a); mb.push_back(b); mc.push_back(c); deg.push_back(n);
                }
            ncdeg[n] = ma.size();
        }
        sumoff.resize(nc + 1);
        for (int k = 0; k < nc; k++) {
            sumoff[k] = sumidx.size();
            int q = p - deg[k];
            for (int m = 0; m < ncdeg[q]; m++) sumidx.push_back(id(ma[k] + ma[m], mb[k] + mb[m], mc[k] + mc[m]));
        }
        sumoff[nc] = sumidx.size();
        for (int hi = 0; hi < nc; hi++)
            for (int lo = 0; lo < nc; lo++)
                if (ma[lo] <= ma[hi] && mb[lo] <= mb[hi] && mc[lo] <= mc[hi]) {
                    sh_hi.push_back(hi); sh_lo.push_back(lo);
                    sh_df.push_back(id(ma[hi] - ma[lo], mb[hi] - mb[lo], mc[hi] - mc[lo]));
                }
        rk1.assign(nc, -1); rk2.assign(nc, -1); rdir.assign(nc, 0); rcoef.assign(nc, 0);
        rinv.assign(nc, 0.0);
        for (int k = 1; k < nc; k++) {
            int a = ma[k], b = mb[k], c = mc[k];
            if (a > 0) { rdir[k] = 0; rk1[k] = id(a - 1, b, c); if (a >= 2) { rk2[k] = id(a - 2, b, c); rcoef[k] = a - 1; } }
            else if (b > 0) { rdir[k] = 1; rk1[k] = id(a, b - 1, c); if (b >= 2) { rk2[k] = id(a, b - 2, c); rcoef[k] = b - 1; } }
            else { rdir[k] = 2; rk1[k] = id(a, b, c - 1); if (c >= 2) { rk2[k] = id(a, b, c - 2); rcoef[k] = c - 1; } }
            rinv[k] = 1.0 / (rdir[k] == 0 ? a : (rdir[k] == 1 ? b : c));
        }
        roff.resize(nc + 1); rtot = 0;
        for (int k = 0; k < nc; k++) { roff[k] = rtot; rtot += p - deg[k] + 1; }
        roff[nc] = rtot;
        blk_n.clear(); blk_nv.clear(); blk_cnt.clear(); blk_off.clear(); blk_idx.clear();
        for (int q2 = 0; q2 <= p; q2++) {
            int n0 = q2 == 0 ? 0 : ncdeg[q2 - 1], n1 = ncdeg[q2];
            int cntm = ncdeg[p - q2];
            for (int nb = n0; nb < n1; nb += 4) {
                int nv = std::min(4, n1 - nb);
                blk_nv.push_back(nv); blk_cnt.push_back(cntm); blk_off.push_back(blk_idx.size());
                for (int j = 0; j < 4; j++) blk_n.push_back(nb + std::min(j, nv - 1));
                for (int m = 0; m < cntm; m++)
                    for (int j = 0; j < 4; j++) {
                        int n = nb + std::min(j, nv - 1);
                        blk_idx.push_back(32 * id(ma[n] + ma[m], mb[n] + mb[m], mc[n] + mc[m]));
                    }
            }
        }
        for (int d = 0; d < 3; d++) {
            s1_dst[d].clear(); s1_src[d].clear(); s1_pw[d].clear(); s2_dst[d].clear(); s2_src[d].clear(); s2_pw[d].clear();
            for (int m = 0; m < nc; m++) {
                int e[3] = {ma[m], mb[m], mc[m]};
                for (int k = 0; k <= e[d]; k++) {   // M2M: src index with component d = k <= e[d]
                    int f[3] = {e[0], e[1], e[2]}; f[d] = k;
                    s1_dst[d].push_back(m); s1_src[d].push_back(id(f[0], f[1], f[2])); s1_pw[d].push_back(e[d] - k);
                }
                int rem = p - (e[0] + e[1] + e[2]);
                for (int k = e[d]; k <= e[d] + rem; k++) {   // L2L: src with component d = k >= e[d]
                    int f[3] = {e[0], e[1], e[2]}; f[d] = k;
                    s2_dst[d].push_back(m); s2_src[d].push_back(id(f[0], f[1], f[2])); s2_pw[d].push_back(k - e[d]);
                }
            }
        }
        b6_n.clear(); b6_cnt.clear(); b6_off.clear(); b6_idx.clear();
        for (int q2 = 0; q2 <= p; q2++) {
            int n0 = q2 == 0 ? 0 : ncdeg[q2 - 1], n1 = ncdeg[q2];
            int cntm = ncdeg[p - q2];
            for (int nb = n0; nb < n1; nb += 6) {
                int nv = std::min(6, n1 - nb);
                b6_cnt.push_back(cntm); b6_off.push_back(b6_idx.size());
                for (int j = 0; j < 6; j++) b6_n.push_back(nb + std::min(j, nv - 1));
                for (int m = 0; m < cntm; m++)
                    for (int j = 0; j < 8; j++) {
                        int n = nb + std::min(std::min(j, 5), nv - 1);
                        b6_idx.push_back(32 * id(ma[n] + ma[m], mb[n] + mb[m], mc[n] + mc[m]));
                    }
            }
        }
        int q = ncdeg[p - 1];
        l2p_x.resize(q); l2p_y.resize(q); l2p_z.resize(q);
        l2p_pk.resize(q);
        for (int k = 0; k < q; k++) l2p_pk[k] = (uint64_t)(32 * id(ma[k] + 1, mb[k], mc[k])) | ((uint64_t)(32 * id(ma[k], mb[k] + 1, mc[k])) << 16) | ((uint64_t)(32 * id(ma[k], mb[k], mc[k] + 1)) << 32);
        sgnmask.resize((nc + 3) / 4);
        for (int k4 = 0; k4 < (nc + 3) / 4; k4++) {
            double sg[4];
            for (int l = 0; l < 4; l++) { int k = 4 * k4 + l; sg[l] = (k < nc && (deg[k] & 1)) ? -0.0 : 0.0; }
            sgnmask[k4] = _mm256_setr_pd(sg[0], sg[1], sg[2], sg[3]);
        }
        for (int k = 0; k < q; k++) { l2p_x[k] = id(ma[k] + 1, mb[k], mc[k]); l2p_y[k] = id(ma[k], mb[k] + 1, mc[k]); l2p_z[k] = id(ma[k], mb[k], mc[k] + 1); }
    }
};
static Tables T;

static void monomials(double x, double y, double z, double* t) {
    // t[k] = v^k / k!  (precomputed reciprocals, no divisions)
    t[0] = 1.0;
    const double xyz[3] = {x, y, z};
    const int* rd = T.rdir.data(); const int* r1 = T.rk1.data(); const double* ri = T.rinv.data();
    for (int k = 1; k < T.nc; k++) t[k] = t[r1[k]] * (xyz[rd[k]] * ri[k]);
}

#include "gen_derivs.inc"
#include "gen_derivs_t.inc"
static derivs4_fn g_derivs = nullptr;
// SIMD derivatives for 4 separation vectors. D layout [k][4]; R compact scratch (m-minor)
static void derivs4_generic(__m256d x, __m256d y, __m256d z, __m256d* D, __m256d* R) {
    const int p = T.p, nc = T.nc;
    const int* roff = T.roff.data();
    __m256d u = _mm256_add_pd(_mm256_fmadd_pd(x, x, _mm256_fmadd_pd(y, y, _mm256_mul_pd(z, z))), _mm256_set1_pd(EPS2));
    __m256d s = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(u));
    __m256d iu = _mm256_mul_pd(s, s);
    __m256d v = s;
    for (int m = 0; m <= p; m++) {
        R[m] = v;
        v = _mm256_mul_pd(v, _mm256_mul_pd(_mm256_set1_pd(-(2.0 * m + 1)), iu));
    }
    D[0] = R[0];
    for (int k = 1; k < nc; k++) {
        const int n = T.deg[k];
        const int d = T.rdir[k];
        const __m256d w = d == 0 ? x : (d == 1 ? y : z);
        __m256d* Rk = R + roff[k];
        const __m256d* R1 = R + roff[T.rk1[k]] + 1;
        const int k2 = T.rk2[k];
        if (k2 >= 0) {
            const __m256d* R2 = R + roff[k2] + 1;
            const __m256d cf = _mm256_set1_pd(T.rcoef[k]);
            for (int m = 0; m <= p - n; m++) Rk[m] = _mm256_fmadd_pd(w, R1[m], _mm256_mul_pd(cf, R2[m]));
        } else {
            for (int m = 0; m <= p - n; m++) Rk[m] = _mm256_mul_pd(w, R1[m]);
        }
        D[k] = Rk[0];
    }
}

static inline void derivs4(__m256d x, __m256d y, __m256d z, __m256d* D, __m256d* R) {
    if (g_derivs) g_derivs(x, y, z, D, R); else derivs4_generic(x, y, z, D, R);
}
// ---------------- tree ----------------
struct Cell {
    double cx, cy, cz, r;
    int begin, end;
    int child0, nchild;
    int level;
};

static int N, NP;
static double *X, *Y, *Z, *M;
static double *AX, *AY, *AZ;
static std::vector<Cell> cells;
static double* Mexp; static double* Lexp;
static int MSTRIDE;
static int NLEAF = 48;
static double THETA = 0.5;

static void build_tree(std::vector<uint64_t>& keys) {
    cells.clear();
    cells.reserve(N + 100);
    Cell root; root.begin = 0; root.end = N; root.level = 0; root.child0 = -1; root.nchild = 0;
    cells.push_back(root);
    for (size_t ci = 0; ci < cells.size(); ci++) {
        Cell c = cells[ci];
        if (c.end - c.begin <= NLEAF || c.level >= 21) continue;
        int shift = 3 * (20 - c.level);
        int first = cells.size();
        int b = c.begin;
        int nch = 0;
        while (b < c.end) {
            uint64_t oct = (keys[b] >> shift) & 7;
            int lo = b, hi = c.end;
            while (lo < hi) { int mid = (lo + hi) / 2; if (((keys[mid] >> shift) & 7) <= oct) lo = mid + 1; else hi = mid; }
            Cell ch; ch.begin = b; ch.end = lo; ch.level = c.level + 1; ch.child0 = -1; ch.nchild = 0;
            cells.push_back(ch);
            nch++;
            b = lo;
        }
        if (nch == 1) {
            // collapse: replace the single child by re-processing this cell at the next level
            cells.pop_back();
            cells[ci].level++;
            ci--;
            continue;
        }
        cells[ci].child0 = first; cells[ci].nchild = nch;
    }
}

// ---------------- P2P ----------------
static inline __m256d rsqrt_acc(__m256d r2) {
    return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(r2));
}
static inline double hsum(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v), hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}

static const int MAXB = 1 << 14;
alignas(64) static double bx[MAXB], by[MAXB], bz[MAXB], bm[MAXB], bax[MAXB], bay[MAXB], baz[MAXB];

// mutual between disjoint ranges
static void p2p_mutual(int a0, int a1, int b0, int b1) {
    int nb = b1 - b0, nb4 = (nb + 3) & ~3;
    for (int j = 0; j < nb4; j++) {
        if (j < nb) { bx[j] = X[b0 + j]; by[j] = Y[b0 + j]; bz[j] = Z[b0 + j]; bm[j] = M[b0 + j]; }
        else { bx[j] = by[j] = bz[j] = 0; bm[j] = 0; }
        bax[j] = bay[j] = baz[j] = 0;
    }
    const __m256d e2 = _mm256_set1_pd(EPS2);
    for (int i = a0; i < a1; i++) {
        __m256d xi = _mm256_set1_pd(X[i]), yi = _mm256_set1_pd(Y[i]), zi = _mm256_set1_pd(Z[i]), mi = _mm256_set1_pd(M[i]);
        __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax;
        for (int j = 0; j < nb4; j += 4) {
            __m256d dx = _mm256_sub_pd(_mm256_load_pd(bx + j), xi);
            __m256d dy = _mm256_sub_pd(_mm256_load_pd(by + j), yi);
            __m256d dz = _mm256_sub_pd(_mm256_load_pd(bz + j), zi);
            __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
            __m256d y = rsqrt_acc(r2);
            __m256d y3 = _mm256_mul_pd(_mm256_mul_pd(y, y), y);
            __m256d fj = _mm256_mul_pd(_mm256_load_pd(bm + j), y3);
            __m256d fi = _mm256_mul_pd(mi, y3);
            ax = _mm256_fmadd_pd(fj, dx, ax); ay = _mm256_fmadd_pd(fj, dy, ay); az = _mm256_fmadd_pd(fj, dz, az);
            _mm256_store_pd(bax + j, _mm256_fnmadd_pd(fi, dx, _mm256_load_pd(bax + j)));
            _mm256_store_pd(bay + j, _mm256_fnmadd_pd(fi, dy, _mm256_load_pd(bay + j)));
            _mm256_store_pd(baz + j, _mm256_fnmadd_pd(fi, dz, _mm256_load_pd(baz + j)));
        }
        AX[i] += hsum(ax); AY[i] += hsum(ay); AZ[i] += hsum(az);
    }
    for (int j = 0; j < nb; j++) { AX[b0 + j] += bax[j]; AY[b0 + j] += bay[j]; AZ[b0 + j] += baz[j]; }
}
// self interaction within range (all ordered pairs; i==j contributes exactly zero)
static void p2p_self(int a0, int a1) {
    const int CH = MAXB - 8;
    for (int s0 = a0; s0 < a1; s0 += CH) {
        int s1 = std::min(a1, s0 + CH);
        int nb = s1 - s0, nb4 = (nb + 3) & ~3;
        for (int j = 0; j < nb4; j++) {
            if (j < nb) { bx[j] = X[s0 + j]; by[j] = Y[s0 + j]; bz[j] = Z[s0 + j]; bm[j] = M[s0 + j]; }
            else { bx[j] = by[j] = bz[j] = 0; bm[j] = 0; }
        }
        const __m256d e2 = _mm256_set1_pd(EPS2);
        for (int i = a0; i < a1; i++) {
            __m256d xi = _mm256_set1_pd(X[i]), yi = _mm256_set1_pd(Y[i]), zi = _mm256_set1_pd(Z[i]);
            __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax;
            for (int j = 0; j < nb4; j += 4) {
                __m256d dx = _mm256_sub_pd(_mm256_load_pd(bx + j), xi);
                __m256d dy = _mm256_sub_pd(_mm256_load_pd(by + j), yi);
                __m256d dz = _mm256_sub_pd(_mm256_load_pd(bz + j), zi);
                __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
                __m256d y3 = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, _mm256_sqrt_pd(r2)));
                __m256d fj = _mm256_mul_pd(_mm256_load_pd(bm + j), y3);
                ax = _mm256_fmadd_pd(fj, dx, ax); ay = _mm256_fmadd_pd(fj, dy, ay); az = _mm256_fmadd_pd(fj, dz, az);
            }
            AX[i] += hsum(ax); AY[i] += hsum(ay); AZ[i] += hsum(az);
        }
    }
}

// grouped mutual P2P: targets [a0,a1) vs gathered buffer of nb sources (g* arrays), 2 targets per pass
static double *gx, *gy, *gz, *gm, *gax, *gay, *gaz;
static void p2p_group_kernel(int a0, int a1, int nb4) {
    const __m256d e2 = _mm256_set1_pd(EPS2), one = _mm256_set1_pd(1.0);
    int i = a0;
    for (; i + 1 < a1; i += 2) {
        __m256d xi = _mm256_set1_pd(X[i]), yi = _mm256_set1_pd(Y[i]), zi = _mm256_set1_pd(Z[i]), mi = _mm256_set1_pd(M[i]);
        __m256d xk = _mm256_set1_pd(X[i + 1]), yk = _mm256_set1_pd(Y[i + 1]), zk = _mm256_set1_pd(Z[i + 1]), mk = _mm256_set1_pd(M[i + 1]);
        __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax, cx = ax, cy = ax, cz = ax;
        for (int j = 0; j < nb4; j += 4) {
            __m256d xj = _mm256_load_pd(gx + j), yj = _mm256_load_pd(gy + j), zj = _mm256_load_pd(gz + j), mj = _mm256_load_pd(gm + j);
            __m256d dx = _mm256_sub_pd(xj, xi), dy = _mm256_sub_pd(yj, yi), dz = _mm256_sub_pd(zj, zi);
            __m256d ex = _mm256_sub_pd(xj, xk), ey = _mm256_sub_pd(yj, yk), ez = _mm256_sub_pd(zj, zk);
            __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
            __m256d s2 = _mm256_fmadd_pd(ex, ex, _mm256_fmadd_pd(ey, ey, _mm256_fmadd_pd(ez, ez, e2)));
            __m256d y = _mm256_div_pd(one, _mm256_mul_pd(r2, _mm256_sqrt_pd(r2)));
            __m256d w = _mm256_div_pd(one, _mm256_mul_pd(s2, _mm256_sqrt_pd(s2)));
            __m256d fj = _mm256_mul_pd(mj, y), fi = _mm256_mul_pd(mi, y);
            __m256d gj = _mm256_mul_pd(mj, w), gk = _mm256_mul_pd(mk, w);
            ax = _mm256_fmadd_pd(fj, dx, ax); ay = _mm256_fmadd_pd(fj, dy, ay); az = _mm256_fmadd_pd(fj, dz, az);
            cx = _mm256_fmadd_pd(gj, ex, cx); cy = _mm256_fmadd_pd(gj, ey, cy); cz = _mm256_fmadd_pd(gj, ez, cz);
            __m256d tx = _mm256_fmadd_pd(fi, dx, _mm256_mul_pd(gk, ex)), ty = _mm256_fmadd_pd(fi, dy, _mm256_mul_pd(gk, ey)), tz = _mm256_fmadd_pd(fi, dz, _mm256_mul_pd(gk, ez));
            _mm256_store_pd(gax + j, _mm256_sub_pd(_mm256_load_pd(gax + j), tx));
            _mm256_store_pd(gay + j, _mm256_sub_pd(_mm256_load_pd(gay + j), ty));
            _mm256_store_pd(gaz + j, _mm256_sub_pd(_mm256_load_pd(gaz + j), tz));
        }
        AX[i] += hsum(ax); AY[i] += hsum(ay); AZ[i] += hsum(az);
        AX[i + 1] += hsum(cx); AY[i + 1] += hsum(cy); AZ[i + 1] += hsum(cz);
    }
    for (; i < a1; i++) {
        __m256d xi = _mm256_set1_pd(X[i]), yi = _mm256_set1_pd(Y[i]), zi = _mm256_set1_pd(Z[i]), mi = _mm256_set1_pd(M[i]);
        __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax;
        for (int j = 0; j < nb4; j += 4) {
            __m256d dx = _mm256_sub_pd(_mm256_load_pd(gx + j), xi), dy = _mm256_sub_pd(_mm256_load_pd(gy + j), yi), dz = _mm256_sub_pd(_mm256_load_pd(gz + j), zi);
            __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
            __m256d y = _mm256_div_pd(one, _mm256_mul_pd(r2, _mm256_sqrt_pd(r2)));
            __m256d fj = _mm256_mul_pd(_mm256_load_pd(gm + j), y), fi = _mm256_mul_pd(mi, y);
            ax = _mm256_fmadd_pd(fj, dx, ax); ay = _mm256_fmadd_pd(fj, dy, ay); az = _mm256_fmadd_pd(fj, dz, az);
            _mm256_store_pd(gax + j, _mm256_fnmadd_pd(fi, dx, _mm256_load_pd(gax + j)));
            _mm256_store_pd(gay + j, _mm256_fnmadd_pd(fi, dy, _mm256_load_pd(gay + j)));
            _mm256_store_pd(gaz + j, _mm256_fnmadd_pd(fi, dz, _mm256_load_pd(gaz + j)));
        }
        AX[i] += hsum(ax); AY[i] += hsum(ay); AZ[i] += hsum(az);
    }
}
// pairs must be sorted by first; processes all mutual pairs
static bool g_abort = false;
static void p2p_grouped(std::vector<std::pair<int, int>>& prs, double t0, double hardcap) {
    std::sort(prs.begin(), prs.end());
    size_t q = 0;
    int ngroups = 0;
    while (q < prs.size()) {
        if ((++ngroups & 63) == 0 && cpu_now() - t0 > hardcap) { g_abort = true; break; }
        int a = prs[q].first;
        size_t q1 = q;
        int nb = 0;
        while (q1 < prs.size() && prs[q1].first == a) {
            const Cell& B = cells[prs[q1].second];
            int n = B.end - B.begin;
            memcpy(gx + nb, X + B.begin, n * 8); memcpy(gy + nb, Y + B.begin, n * 8);
            memcpy(gz + nb, Z + B.begin, n * 8); memcpy(gm + nb, M + B.begin, n * 8);
            nb += n; q1++;
        }
        int nb4 = (nb + 3) & ~3;
        for (int j = nb; j < nb4; j++) { gx[j] = gy[j] = gz[j] = 1e50; gm[j] = 0; }
        memset(gax, 0, nb4 * 8); memset(gay, 0, nb4 * 8); memset(gaz, 0, nb4 * 8);
        const Cell& A = cells[a];
        p2p_group_kernel(A.begin, A.end, nb4);
        nb = 0;
        for (size_t t = q; t < q1; t++) {
            const Cell& B = cells[prs[t].second];
            for (int j = B.begin; j < B.end; j++, nb++) { AX[j] += gax[nb]; AY[j] += gay[nb]; AZ[j] += gaz[nb]; }
        }
        q = q1;
    }
}

// ---------------- M2L batched ----------------
static __m256d *Dv, *Rv, *MAv, *ABv, *LBv, *LAv;
static inline void transpose4(__m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3) {
    __m256d t0 = _mm256_unpacklo_pd(r0, r1), t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3), t3 = _mm256_unpackhi_pd(r2, r3);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20); r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31); r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
static void m2l_batch(const std::pair<int, int>* prs, int cnt, const Tables& TQ, derivs4_fn dfn) {
    const int nc = TQ.nc;
    const int nc4 = (nc + 3) & ~3;
    alignas(32) double dx[4], dy[4], dz[4];
    int A[4], B[4];
    for (int l = 0; l < 4; l++) {
        int q = l < cnt ? l : 0;
        A[l] = prs[q].first; B[l] = prs[q].second;
        const Cell& ca = cells[A[l]]; const Cell& cb = cells[B[l]];
        dx[l] = cb.cx - ca.cx; dy[l] = cb.cy - ca.cy; dz[l] = cb.cz - ca.cz;
    }
    if (dfn) dfn(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Dv, Rv);
    else derivs4(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Dv, Rv);
    const double* ma[4]; const double* mb[4];
    for (int l = 0; l < 4; l++) { ma[l] = Mexp + (size_t)A[l] * MSTRIDE; mb[l] = Mexp + (size_t)B[l] * MSTRIDE; }
    for (int m = 0; m < nc4; m += 4) {
        __m256d a0 = _mm256_loadu_pd(ma[0] + m), a1 = _mm256_loadu_pd(ma[1] + m), a2 = _mm256_loadu_pd(ma[2] + m), a3 = _mm256_loadu_pd(ma[3] + m);
        transpose4(a0, a1, a2, a3);
        MAv[m] = a0; MAv[m + 1] = a1; MAv[m + 2] = a2; MAv[m + 3] = a3;
        __m256d b0 = _mm256_loadu_pd(mb[0] + m), b1 = _mm256_loadu_pd(mb[1] + m), b2 = _mm256_loadu_pd(mb[2] + m), b3 = _mm256_loadu_pd(mb[3] + m);
        transpose4(b0, b1, b2, b3);
        ABv[m] = b0; ABv[m + 1] = b1; ABv[m + 2] = b2; ABv[m + 3] = b3;
    }
    for (int m = 0; m < nc; m++) if (TQ.deg[m] & 1) ABv[m] = _mm256_sub_pd(_mm256_setzero_pd(), ABv[m]);
    const int nblk = TQ.b6_cnt.size();
    const char* Db = (const char*)Dv;
    for (int b = 0; b < nblk; b++) {
        const int cntm = TQ.b6_cnt[b];
        const uint64_t* ix = (const uint64_t*)&TQ.b6_idx[TQ.b6_off[b]];
        __m256d b0 = _mm256_setzero_pd(), b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0;
        __m256d c0 = b0, c1 = b0, c2 = b0, c3 = b0, c4 = b0, c5 = b0;
        for (int m = 0; m < cntm; m++, ix += 2) {
            const __m256d a = MAv[m], bb = ABv[m];
            uint64_t w0 = ix[0], w1 = ix[1];
            __m256d d;
            d = *(const __m256d*)(Db + (w0 & 0xffff)); b0 = _mm256_fmadd_pd(a, d, b0); c0 = _mm256_fmadd_pd(bb, d, c0);
            d = *(const __m256d*)(Db + ((w0 >> 16) & 0xffff)); b1 = _mm256_fmadd_pd(a, d, b1); c1 = _mm256_fmadd_pd(bb, d, c1);
            d = *(const __m256d*)(Db + ((w0 >> 32) & 0xffff)); b2 = _mm256_fmadd_pd(a, d, b2); c2 = _mm256_fmadd_pd(bb, d, c2);
            d = *(const __m256d*)(Db + (w0 >> 48)); b3 = _mm256_fmadd_pd(a, d, b3); c3 = _mm256_fmadd_pd(bb, d, c3);
            d = *(const __m256d*)(Db + (w1 & 0xffff)); b4 = _mm256_fmadd_pd(a, d, b4); c4 = _mm256_fmadd_pd(bb, d, c4);
            d = *(const __m256d*)(Db + ((w1 >> 16) & 0xffff)); b5 = _mm256_fmadd_pd(a, d, b5); c5 = _mm256_fmadd_pd(bb, d, c5);
        }
        const int* nn = &TQ.b6_n[6 * b];
        LBv[nn[5]] = b5; LAv[nn[5]] = c5; LBv[nn[4]] = b4; LAv[nn[4]] = c4;
        LBv[nn[3]] = b3; LAv[nn[3]] = c3; LBv[nn[2]] = b2; LAv[nn[2]] = c2;
        LBv[nn[1]] = b1; LAv[nn[1]] = c1; LBv[nn[0]] = b0; LAv[nn[0]] = c0;
    }
    for (int n = 0; n < nc; n++) if (TQ.deg[n] & 1) LAv[n] = _mm256_sub_pd(_mm256_setzero_pd(), LAv[n]);
    for (int n = nc; n < nc4; n++) { LAv[n] = _mm256_setzero_pd(); LBv[n] = _mm256_setzero_pd(); }
    double* la[4]; double* lb[4];
    for (int l = 0; l < 4; l++) { la[l] = Lexp + (size_t)A[l] * MSTRIDE; lb[l] = Lexp + (size_t)B[l] * MSTRIDE; }
    for (int n = 0; n < nc4; n += 4) {
        __m256d r0 = LBv[n], r1 = LBv[n + 1], r2 = LBv[n + 2], r3 = LBv[n + 3];
        transpose4(r0, r1, r2, r3);
        __m256d s0 = LAv[n], s1 = LAv[n + 1], s2 = LAv[n + 2], s3 = LAv[n + 3];
        transpose4(s0, s1, s2, s3);
        __m256d rr[4] = {r0, r1, r2, r3}, ss[4] = {s0, s1, s2, s3};
        for (int l = 0; l < cnt; l++) {
            _mm256_storeu_pd(lb[l] + n, _mm256_add_pd(_mm256_loadu_pd(lb[l] + n), rr[l]));
            _mm256_storeu_pd(la[l] + n, _mm256_add_pd(_mm256_loadu_pd(la[l] + n), ss[l]));
        }
    }
}

static void m2l_batch(const std::pair<int, int>* prs, int cnt) { m2l_batch(prs, cnt, T, g_derivs); }

// factorized shifts. M2M: out += shift(in) by vector s (moments: uses (-s) powers handled by caller sign)
static void shift_m2m(const double* in, double* out, double sx, double sy, double sz, double* t1, double* t2) {
    // out[m] += sum_{k<=m} in[k] * prod_d s_d^{m_d-k_d}/(m_d-k_d)!
    const int nc = T.nc, p = T.p;
    double pw[3][32];
    double s[3] = {sx, sy, sz};
    for (int d = 0; d < 3; d++) { pw[d][0] = 1; for (int j = 1; j <= p; j++) pw[d][j] = pw[d][j - 1] * s[d] / j; }
    const double* src = in; double* dsts[3] = {t1, t2, nullptr};
    for (int d = 0; d < 3; d++) {
        double* dst = dsts[d];
        if (d < 2) for (int k = 0; k < nc; k++) dst[k] = 0;
        const int* D = T.s1_dst[d].data(); const int* S = T.s1_src[d].data(); const int* P = T.s1_pw[d].data();
        size_t ne = T.s1_dst[d].size();
        if (d < 2) { for (size_t q = 0; q < ne; q++) dst[D[q]] += src[S[q]] * pw[d][P[q]]; src = dst; }
        else { for (size_t q = 0; q < ne; q++) out[D[q]] += src[S[q]] * pw[d][P[q]]; }
    }
}
static void shift_l2l(const double* in, double* out, double sx, double sy, double sz, double* t1, double* t2) {
    const int nc = T.nc, p = T.p;
    double pw[3][32];
    double s[3] = {sx, sy, sz};
    for (int d = 0; d < 3; d++) { pw[d][0] = 1; for (int j = 1; j <= p; j++) pw[d][j] = pw[d][j - 1] * s[d] / j; }
    const double* src = in; double* dsts[3] = {t1, t2, nullptr};
    for (int d = 0; d < 3; d++) {
        double* dst = dsts[d];
        if (d < 2) for (int k = 0; k < nc; k++) dst[k] = 0;
        const int* D = T.s2_dst[d].data(); const int* S = T.s2_src[d].data(); const int* P = T.s2_pw[d].data();
        size_t ne = T.s2_dst[d].size();
        if (d < 2) { for (size_t q = 0; q < ne; q++) dst[D[q]] += src[S[q]] * pw[d][P[q]]; src = dst; }
        else { for (size_t q = 0; q < ne; q++) out[D[q]] += src[S[q]] * pw[d][P[q]]; }
    }
}
// ---------- generic scalar helpers parameterized by tables ----------
static void derivs_s(const Tables& t, double x, double y, double z, double* D, double* R) {
    const int p = t.p, nc = t.nc;
    double u = x * x + y * y + z * z + EPS2;
    double s = 1.0 / std::sqrt(u), iu = s * s, v = s;
    for (int m = 0; m <= p; m++) { R[m * nc] = v; v *= -(2.0 * m + 1) * iu; }
    for (int k = 1; k < nc; k++) {
        int n = t.deg[k], d = t.rdir[k];
        double w = d == 0 ? x : (d == 1 ? y : z);
        int k1 = t.rk1[k], k2 = t.rk2[k];
        for (int m = 0; m <= p - n; m++) {
            double r = w * R[(m + 1) * nc + k1];
            if (k2 >= 0) r += t.rcoef[k] * R[(m + 1) * nc + k2];
            R[m * nc + k] = r;
        }
    }
    for (int k = 0; k < nc; k++) D[k] = R[k];
}
static void monomials_t(const Tables& t, double x, double y, double z, double* o) {
    o[0] = 1.0;
    for (int k = 1; k < t.nc; k++) {
        int d = t.rdir[k];
        double v = d == 0 ? x / t.ma[k] : (d == 1 ? y / t.mb[k] : z / t.mc[k]);
        o[k] = o[t.rk1[k]] * v;
    }
}

// ---------- M2P + P2L: source/target cell pc, particles of cell qc ----------
static __m256d* Lacc;
static long long n_m2p = 0;
static void m2p_p2l(int pc, int qc, const Tables& TQ, derivs4_fn dfn) {
    const Cell& P = cells[pc]; const Cell& Q = cells[qc];
    const int nc = TQ.nc, p = TQ.p;
    const int nc4 = (nc + 3) & ~3;
    const double* MP = Mexp + (size_t)pc * MSTRIDE;
    double* LP = Lexp + (size_t)pc * MSTRIDE;
    for (int k = 0; k < nc4; k++) Lacc[k] = _mm256_setzero_pd();
    const int q1 = TQ.ncdeg[p - 1];
    const uint64_t* l2p = TQ.l2p_pk.data();
    const char* Db = (const char*)Dv;
    for (int j = Q.begin; j < Q.end; j += 4) {
        int nv = std::min(4, Q.end - j);
        alignas(32) double dx[4], dy[4], dz[4], mm[4];
        for (int l = 0; l < 4; l++) {
            int jj = l < nv ? j + l : j;
            dx[l] = X[jj] - P.cx; dy[l] = Y[jj] - P.cy; dz[l] = Z[jj] - P.cz; mm[l] = l < nv ? M[jj] : 0.0;
        }
        if (dfn) dfn(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Dv, Rv);
        else derivs4(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Dv, Rv);
        __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax;
        for (int m = 0; m < q1; m++) {
            __m256d a = _mm256_broadcast_sd(MP + m);
            uint64_t w = l2p[m];
            ax = _mm256_fmadd_pd(a, *(const __m256d*)(Db + (w & 0xffff)), ax);
            ay = _mm256_fmadd_pd(a, *(const __m256d*)(Db + ((w >> 16) & 0xffff)), ay);
            az = _mm256_fmadd_pd(a, *(const __m256d*)(Db + ((w >> 32) & 0xffff)), az);
        }
        alignas(32) double tx[4], ty[4], tz[4];
        _mm256_store_pd(tx, ax); _mm256_store_pd(ty, ay); _mm256_store_pd(tz, az);
        for (int l = 0; l < nv; l++) { AX[j + l] += tx[l]; AY[j + l] += ty[l]; AZ[j + l] += tz[l]; }
        __m256d mv = _mm256_load_pd(mm);
        for (int k = 0; k < nc; k++) Lacc[k] = _mm256_fmadd_pd(mv, Dv[k], Lacc[k]);
    }
    for (int k = 0; k < nc4; k += 4) {
        __m256d r0 = Lacc[k], r1 = Lacc[k + 1], r2 = Lacc[k + 2], r3 = Lacc[k + 3];
        transpose4(r0, r1, r2, r3);
        __m256d s = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));
        s = _mm256_xor_pd(s, TQ.sgnmask[k >> 2]);
        _mm256_storeu_pd(LP + k, _mm256_add_pd(_mm256_loadu_pd(LP + k), s));
    }
    n_m2p += Q.end - Q.begin;
}

static void m2p_p2l(int pc, int qc) { m2p_p2l(pc, qc, T, g_derivs); }
