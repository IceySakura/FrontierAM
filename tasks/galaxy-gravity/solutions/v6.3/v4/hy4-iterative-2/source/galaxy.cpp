// Gravitational accelerations for galaxy snapshots.
//
// Adaptive geometric octree; Cartesian multipole expansions of the softened
// kernel about each cell's centre of mass, translated into local Taylor
// expansions about each target leaf (direct M2L at every level of the tree),
// with exact direct summation for all non-well-separated leaf pairs.
//
// Build: g++ -O3 -mavx2 -mfma -o solve galaxy.cpp
//
// Environment overrides (for development):
//   GAL_P, GAL_Q, GAL_THETA, GAL_LEAF, GAL_MAXDEPTH, GAL_VERBOSE

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <time.h>

static constexpr double EPS2 = 1e-4; // softening^2

static double cpu_time() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static double getenvd(const char *name, double def) {
    const char *s = getenv(name);
    return s ? atof(s) : def;
}
static int getenvip(const char *name, int def) {
    const char *s = getenv(name);
    return s ? atoi(s) : def;
}

static inline int i3(int a, int b, int c, int side) {
    return (a * side + b) * side + c;
}
static double fact(int n) {
    double r = 1;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// ---------------------------------------------------------------------------
// Multi-index calculus up to order N = P + Q
// ---------------------------------------------------------------------------
struct Poly {
    int P, Q, N, side;
    int KP, KQ, KG, KU;
    std::vector<int> gid, mid, wid, uid;
    std::vector<int> mdeg, wdeg;
    std::vector<double> minvf, winvf;
    // G recurrence (derivatives of the softened kernel)
    std::vector<int> bOut, bStart, bCnt, tSlot, tDh;
    std::vector<double> tCo;
    // contraction M -> L
    std::vector<int> gW, gStart, gCnt, pM, pG;
    std::vector<double> gInv;
    // monomial evaluation at particles (degree <= Q-1)
    std::vector<int> uPar, uAxis, uW0, uW1, uW2;
    std::vector<double> uC0, uC1, uC2;
    // flattened evaluation list, including mu = 0
    std::vector<int> evMono, evW0, evW1, evW2;
    std::vector<double> evC0, evC1, evC2;
    // monomial enumeration for leaf moments (degree <= P)
    std::vector<int> mPar, mAxis, mOrder;

    Poly(int p, int q) : P(p), Q(q), N(p + q) {
        side = N + 2;
        int S3 = side * side * side;
        gid.assign(S3, -1); mid.assign(S3, -1);
        wid.assign(S3, -1); uid.assign(S3, -1);
        KP = KQ = KG = KU = 0;
        std::vector<std::array<int, 4>> all; // a,b,c,deg
        for (int n = 0; n <= N; ++n)
            for (int a = 0; a <= n; ++a)
                for (int b = 0; b <= n - a; ++b) {
                    int c = n - a - b;
                    all.push_back({a, b, c, n});
                }
        for (const auto &e : all) {
            int k = i3(e[0], e[1], e[2], side);
            double iv = 1.0 / (fact(e[0]) * fact(e[1]) * fact(e[2]));
            gid[k] = KG++;
            if (e[3] <= P) {
                mid[k] = KP++;
                mdeg.push_back(e[3]);
                minvf.push_back(iv);
            }
            if (e[3] <= Q) {
                wid[k] = KQ++;
                wdeg.push_back(e[3]);
                winvf.push_back(iv);
            }
            if (e[3] <= Q - 1) uid[k] = KU++;
        }
        mPar.assign(KP, -1); mAxis.assign(KP, -1);
        for (int n = 1; n <= P; ++n)
            for (int a = 0; a <= n; ++a)
                for (int b = 0; b <= n - a; ++b) {
                    int c = n - a - b;
                    int s = mid[i3(a, b, c, side)];
                    mOrder.push_back(s);
                    int ax = (a > 0 ? 0 : (b > 0 ? 1 : 2));
                    int t[3] = {a, b, c}; t[ax]--;
                    mPar[s] = mid[i3(t[0], t[1], t[2], side)];
                    mAxis[s] = ax;
                }
        // derivative recurrence
        for (const auto &e : all) {
            if (e[3] == 0) continue;
            int v[3] = {e[0], e[1], e[2]};
            int ii = (v[0] > 0 ? 0 : (v[1] > 0 ? 1 : 2));
            int al[3] = {v[0], v[1], v[2]};
            al[ii]--;
            bOut.push_back(gid[i3(v[0], v[1], v[2], side)]);
            bStart.push_back((int)tSlot.size());
            auto push = [&](int s, double co, int dh) {
                tSlot.push_back(s); tCo.push_back(co); tDh.push_back(dh);
            };
            push(gid[i3(al[0], al[1], al[2], side)], 1.0, ii);
            if (al[ii] > 0) {
                int t[3] = {al[0], al[1], al[2]}; t[ii]--;
                push(gid[i3(t[0], t[1], t[2], side)], (double)al[ii], -1);
            }
            for (int j = 0; j < 3; ++j) if (al[j] > 0) {
                int t[3] = {al[0], al[1], al[2]};
                t[j]--; t[ii]++;
                push(gid[i3(t[0], t[1], t[2], side)], 2.0 * al[j], j);
            }
            for (int j = 0; j < 3; ++j) if (al[j] > 1) {
                int t[3] = {al[0], al[1], al[2]};
                t[j] -= 2; t[ii]++;
                push(gid[i3(t[0], t[1], t[2], side)],
                     (double)al[j] * (al[j] - 1), -1);
            }
            bCnt.push_back((int)tSlot.size() - bStart.back());
        }
        // M2L contraction
        for (const auto &e : all) {
            if (e[3] > Q) continue;
            int w = wid[i3(e[0], e[1], e[2], side)];
            gW.push_back(w);
            gInv.push_back(winvf[w]);
            gStart.push_back((int)pM.size());
            for (const auto &f : all) {
                if (f[3] > P) continue;
                int ms = mid[i3(f[0], f[1], f[2], side)];
                int gs = gid[i3(f[0] + e[0], f[1] + e[1], f[2] + e[2], side)];
                pM.push_back(ms); pG.push_back(gs);
            }
            gCnt.push_back((int)pM.size() - gStart.back());
        }
        // monomial evaluation
        for (int n = 1; n <= Q - 1; ++n)
            for (int a = 0; a <= n; ++a)
                for (int b = 0; b <= n - a; ++b) {
                    int c = n - a - b;
                    int v[3] = {a, b, c};
                    int ax = (a > 0 ? 0 : (b > 0 ? 1 : 2));
                    int t[3] = {a, b, c}; t[ax]--;
                    uPar.push_back(uid[i3(t[0], t[1], t[2], side)]);
                    uAxis.push_back(ax);
                    for (int j = 0; j < 3; ++j) {
                        int s[3] = {a, b, c}; s[j]++;
                        (j == 0 ? uW0 : (j == 1 ? uW1 : uW2))
                            .push_back(wid[i3(s[0], s[1], s[2], side)]);
                    }
                    uC0.push_back(a + 1); uC1.push_back(b + 1); uC2.push_back(c + 1);
                }
        evMono.push_back(0);
        evW0.push_back(wid[i3(1, 0, 0, side)]);
        evW1.push_back(wid[i3(0, 1, 0, side)]);
        evW2.push_back(wid[i3(0, 0, 1, side)]);
        evC0.push_back(1); evC1.push_back(1); evC2.push_back(1);
        for (size_t u = 0; u < uPar.size(); ++u) {
            evMono.push_back((int)u + 1);
            evW0.push_back(uW0[u]); evW1.push_back(uW1[u]); evW2.push_back(uW2[u]);
            evC0.push_back(uC0[u]); evC1.push_back(uC1[u]); evC2.push_back(uC2[u]);
        }
    }
    int gidx(int a, int b, int c) const { return gid[i3(a, b, c, side)]; }
};

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------
struct alignas(64) CellQ {
    double cx, cy, cz, mass, r2;
    double qxx, qyy, qzz, qxy, qxz, qyz, tr;
    double rad;
};
struct Node {
    double cx, cy, cz, rad;
    int32_t cnt;
    int32_t ch[8];
    int32_t start, end;
};

struct Tree {
    int N;
    double m0;
    double *px, *py, *pz;
    uint32_t *perm, *tmp;
    std::vector<Node> nd;
    const Poly *poly;
    double *mom; // nd.size() * KP
    CellQ *cq;
    int momStride;
    int leafMax, maxDepth;
    double rootRad;

    Tree(int n, const Poly &po, int lm, int md)
        : N(n), poly(&po), leafMax(lm), maxDepth(md) {
        px = (double *)_mm_malloc(sizeof(double) * n, 64);
        py = (double *)_mm_malloc(sizeof(double) * n, 64);
        pz = (double *)_mm_malloc(sizeof(double) * n, 64);
        perm = (uint32_t *)malloc(sizeof(uint32_t) * n);
        tmp = (uint32_t *)malloc(sizeof(uint32_t) * n);
        for (int i = 0; i < n; ++i) perm[i] = (uint32_t)i;
        cq = nullptr;
        nd.reserve((size_t)(2.2 * n / lm) + 64);
    }
    ~Tree() {
        _mm_free(px); _mm_free(py); _mm_free(pz);
        free(perm); free(tmp);
        free(mom);
        _mm_free(cq);
    }
    void load(const double *raw) { // raw: N*4 (m,x,y,z)
        m0 = raw[0];
        for (int i = 0; i < N; ++i) {
            px[i] = raw[4 * i + 1]; py[i] = raw[4 * i + 2]; pz[i] = raw[4 * i + 3];
        }
    }

    // Morton keys, 40 bits per axis -> 120-bit interleaved key
    std::vector<unsigned __int128> keys;
    static const int BITS = 40;
    static unsigned __int128 spread8[256];
    static bool spreadInit;
    static void initSpread() {
        if (spreadInit) return;
        for (int v = 0; v < 256; ++v) {
            unsigned __int128 s = 0;
            for (int b = 0; b < 8; ++b)
                if (v & (1 << b)) s |= (unsigned __int128)1 << (3 * b);
            spread8[v] = s;
        }
        spreadInit = true;
    }
    static unsigned __int128 spread40(uint64_t u) {
        unsigned __int128 s = 0;
        for (int k = 0; k < 5; ++k)
            s |= spread8[(u >> (8 * k)) & 255] << (24 * k);
        return s;
    }
    void make_keys_and_sort(const double *raw) {
        initSpread();
        m0 = raw[0];
        double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300, z0 = 1e300, z1 = -1e300;
        for (int i = 0; i < N; ++i) {
            double x = raw[4 * i + 1], y = raw[4 * i + 2], z = raw[4 * i + 3];
            x0 = std::min(x0, x); x1 = std::max(x1, x);
            y0 = std::min(y0, y); y1 = std::max(y1, y);
            z0 = std::min(z0, z); z1 = std::max(z1, z);
        }
        double range = std::max(x1 - x0, std::max(y1 - y0, z1 - z0));
        if (!(range > 0)) range = 1.0;
        double scale = (double)((1ULL << 40) - 1) / range;
        struct Entry { unsigned __int128 key; uint32_t idx; };
        std::vector<Entry> e(N);
        for (int i = 0; i < N; ++i) {
            double u = (raw[4 * i + 1] - x0) * scale;
            double v = (raw[4 * i + 2] - y0) * scale;
            double w = (raw[4 * i + 3] - z0) * scale;
            uint64_t iu = (uint64_t)std::min(std::max(u, 0.0), (double)((1ULL << 40) - 1));
            uint64_t iv = (uint64_t)std::min(std::max(v, 0.0), (double)((1ULL << 40) - 1));
            uint64_t iw = (uint64_t)std::min(std::max(w, 0.0), (double)((1ULL << 40) - 1));
            e[i].key = spread40(iu) | (spread40(iv) << 1) | (spread40(iw) << 2);
            e[i].idx = (uint32_t)i;
        }
        std::sort(e.begin(), e.end(), [](const Entry &a, const Entry &b) {
            return a.key < b.key;
        });
        keys.resize(N);
        for (int i = 0; i < N; ++i) { keys[i] = e[i].key; perm[i] = e[i].idx; }
        // px/py/pz are still in original order here; reorder now
        double *nx = (double *)_mm_malloc(sizeof(double) * N, 64);
        double *ny = (double *)_mm_malloc(sizeof(double) * N, 64);
        double *nz = (double *)_mm_malloc(sizeof(double) * N, 64);
        for (int i = 0; i < N; ++i) {
            uint32_t j = perm[i];
            nx[i] = raw[4 * j + 1]; ny[i] = raw[4 * j + 2]; nz[i] = raw[4 * j + 3];
        }
        _mm_free(px); _mm_free(py); _mm_free(pz);
        px = nx; py = ny; pz = nz;
    }
    static int clz128(unsigned __int128 x) {
        uint64_t hi = (uint64_t)(x >> 64);
        if (hi) return __builtin_clzll(hi);
        uint64_t lo = (uint64_t)x;
        return lo ? 64 + __builtin_clzll(lo) : 128;
    }
    int build(int start, int end, int level) {
        int id = (int)nd.size();
        nd.push_back(Node{});
        nd[id].start = start; nd[id].end = end;
        nd[id].cnt = end - start;
        for (int k = 0; k < 8; ++k) nd[id].ch[k] = -1;
        if (end - start <= leafMax) return id;
        // `skip` is the absolute number of common 3-bit groups from the top
        unsigned __int128 xo = (keys[start] ^ keys[end - 1]) << 8; // keys use bits 0..119
        int skip = clz128(xo) / 3;
        if (skip >= BITS || level >= maxDepth) {
            // all keys identical (or depth cap): split by index
            int mid = (start + end) / 2;
            nd[id].ch[0] = build(start, mid, level);
            nd[id].ch[1] = build(mid, end, level);
            return id;
        }
        level = skip;
        int shift = 3 * (BITS - 1 - level);
        // keys are sorted and share the prefix above `shift`, so the octant
        // digit is non-decreasing over [start,end): just find the 8 boundaries.
        int off[9];
        off[0] = start;
        int k = start;
        for (int o = 0; o < 8; ++o) {
            while (k < end && (int)((keys[k] >> shift) & 7) == o) ++k;
            off[o + 1] = k;
        }
        for (int o = 0; o < 8; ++o) if (off[o + 1] > off[o]) {
            int cid = build(off[o], off[o + 1], level + 1);
            nd[id].ch[o] = cid;
        }
        return id;
    }

    void reorder() {
        double *nx = (double *)_mm_malloc(sizeof(double) * N, 64);
        double *ny = (double *)_mm_malloc(sizeof(double) * N, 64);
        double *nz = (double *)_mm_malloc(sizeof(double) * N, 64);
        for (int i = 0; i < N; ++i) {
            uint32_t j = perm[i];
            nx[i] = px[j]; ny[i] = py[j]; nz[i] = pz[j];
        }
        _mm_free(px); _mm_free(py); _mm_free(pz);
        px = nx; py = ny; pz = nz;
        free(tmp); tmp = nullptr;
    }

    void compute_moments() {
        const Poly &po = *poly;
        int KP = po.KP;
        momStride = KP;
        size_t nn = nd.size();
        mom = (double *)_mm_malloc(sizeof(double) * nn * KP, 64);
        std::vector<double> scratch(KP * 8);
        std::vector<double> rpow(po.P + 1), spow(po.P + 1), tp(po.P + 1);
        std::vector<double> mono(KP);
        // enumerate (a,b,c) for deg<=P in slot order
        std::vector<std::array<int, 3>> mabc(KP);
        for (int n = 0; n <= po.P; ++n)
            for (int a = 0; a <= n; ++a)
                for (int b = 0; b <= n - a; ++b) {
                    int c = n - a - b;
                    int s = po.mid[i3(a, b, c, po.side)];
                    mabc[s] = {a, b, c};
                }
        for (int id = (int)nn - 1; id >= 0; --id) {
            Node &ndref = nd[id];
            bool leaf = (ndref.ch[0] == -1 && ndref.ch[1] == -1 &&
                         ndref.ch[2] == -1 && ndref.ch[3] == -1 &&
                         ndref.ch[4] == -1 && ndref.ch[5] == -1 &&
                         ndref.ch[6] == -1 && ndref.ch[7] == -1);
            double cx, cy, cz, rad = 0;
            double *M = mom + (size_t)id * KP;
            if (leaf) {
                int s0 = ndref.start, s1 = ndref.end;
                double sx = 0, sy = 0, sz = 0;
                for (int k = s0; k < s1; ++k) { sx += px[k]; sy += py[k]; sz += pz[k]; }
                double inv = 1.0 / (s1 - s0);
                cx = sx * inv; cy = sy * inv; cz = sz * inv;
                for (int k = s0; k < s1; ++k) {
                    double dx = px[k] - cx, dy = py[k] - cy, dz = pz[k] - cz;
                    double d2 = dx * dx + dy * dy + dz * dz;
                    rad = std::max(rad, d2);
                }
                rad = sqrt(rad) * (1.0 + 1e-12);
                if (rad < 1e-100) rad = 0.0; // below representable moment scale
                ndref.cx = cx; ndref.cy = cy; ndref.cz = cz;
                ndref.rad = rad;
                // raw monomial sums of u = c - x
                double rbase = rad > 0 ? rad : 1.0;
                rpow[0] = 1;
                for (int d = 1; d <= po.P; ++d) rpow[d] = rpow[d - 1] / rbase;
                std::fill(M, M + KP, 0.0);
                M[0] = 1.0;
                if (rad > 0) {
                    for (int k = s0; k < s1; ++k) {
                        double u[3] = {cx - px[k], cy - py[k], cz - pz[k]};
                        mono[0] = 1.0;
                        for (int s : po.mOrder)
                            mono[s] = mono[po.mPar[s]] * u[po.mAxis[s]];
                        for (int s = 1; s < KP; ++s) M[s] += mono[s];
                    }
                    for (int s = 1; s < KP; ++s)
                        M[s] *= (inv * po.minvf[s] * rpow[po.mdeg[s]]);
                } else {
                    for (int s = 1; s < KP; ++s) M[s] = 0.0;
                }
            } else {
                double sx = 0, sy = 0, sz = 0;
                int tot = 0;
                for (int o = 0; o < 8; ++o) {
                    int c = ndref.ch[o];
                    if (c < 0) continue;
                    const Node &k = nd[c];
                    sx += k.cx * k.cnt; sy += k.cy * k.cnt; sz += k.cz * k.cnt;
                    tot += k.cnt;
                }
                double inv = 1.0 / tot;
                cx = sx * inv; cy = sy * inv; cz = sz * inv;
                // exact radius: scan all particles
                for (int k = ndref.start; k < ndref.end; ++k) {
                    double dx = px[k] - cx, dy = py[k] - cy, dz = pz[k] - cz;
                    double d2 = dx * dx + dy * dy + dz * dz;
                    rad = std::max(rad, d2);
                }
                rad = sqrt(rad) * (1.0 + 1e-12);
                if (rad < 1e-100) rad = 0.0;
                ndref.cx = cx; ndref.cy = cy; ndref.cz = cz;
                ndref.rad = rad;
                ndref.cnt = tot;
                std::fill(M, M + KP, 0.0);
                if (rad == 0.0) { M[0] = 1.0; continue; } // coincident: exact monopole
                for (int o = 0; o < 8; ++o) {
                    int c = ndref.ch[o];
                    if (c < 0) continue;
                    const Node &k = nd[c];
                    double *B = scratch.data() + o * KP;
                    const double *Mc = mom + (size_t)c * KP;
                    double s = k.rad / rad;
                    spow[0] = 1;
                    for (int d = 1; d <= po.P; ++d) spow[d] = spow[d - 1] * s;
                    for (int m = 0; m < KP; ++m) B[m] = Mc[m] * spow[po.mdeg[m]];
                    double t[3] = {(cx - k.cx) / rad, (cy - k.cy) / rad, (cz - k.cz) / rad};
                    // translate by t: B_alpha <- sum_beta B_beta t^(alpha-beta)/(alpha-beta)!
                    for (int ax = 0; ax < 3; ++ax) {
                        double tv = t[ax];
                        tp[0] = 1;
                        for (int d = 1; d <= po.P; ++d) tp[d] = tp[d - 1] * tv / d;
                        for (int na = po.P; na >= 1; --na) {
                            // all slots whose ax-component is na (descending so
                            // source slots have not been modified in this pass)
                            for (int m = 0; m < KP; ++m) {
                                if (mabc[m][ax] != na) continue;
                                double add = 0;
                                int cur[3] = {mabc[m][0], mabc[m][1], mabc[m][2]};
                                for (int q = 1; q <= na; ++q) {
                                    cur[ax] -= 1;
                                    int ms = po.mid[i3(cur[0], cur[1], cur[2], po.side)];
                                    add += B[ms] * tp[q];
                                }
                                B[m] += add;
                            }
                        }
                    }
                    double w = (double)k.cnt * inv;
                    for (int m = 0; m < KP; ++m) M[m] += w * B[m];
                }
            }
        }
        rootRad = nd[0].rad;
    }

    void build_cellq() {
        size_t nn = nd.size();
        cq = (CellQ*)_mm_malloc(sizeof(CellQ)*nn, 64);
        int sx[6] = {po_mid(2,0,0), po_mid(0,2,0), po_mid(0,0,2),
                     po_mid(1,1,0), po_mid(1,0,1), po_mid(0,1,1)};
        for (size_t i = 0; i < nn; ++i) {
            const Node &n = nd[i];
            const double *M = mom + i*momStride;
            CellQ &c = cq[i];
            c.cx=n.cx; c.cy=n.cy; c.cz=n.cz;
            c.mass=m0*n.cnt; c.r2=n.rad*n.rad;
            double q[6];
            for (int k=0;k<6;++k) q[k]=M[sx[k]]*(k<3?2.0:1.0);
            c.qxx=q[0]; c.qyy=q[1]; c.qzz=q[2];
            c.qxy=q[3]; c.qxz=q[4]; c.qyz=q[5];
            c.tr=q[0]+q[1]+q[2];
            c.rad=n.rad;
        }
    }
    int po_mid(int a,int b,int c) const { return poly->mid[i3(a,b,c,poly->side)]; }
};

unsigned __int128 Tree::spread8[256];
bool Tree::spreadInit = false;

#include "m2l_gen.h"

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------
struct Solver {
    Tree &t;
    const Poly &po;
    double theta, theta2;
    double *out;
    int P, Q, KP, KQ, KG, KU;
    // scratch
    std::vector<double> G, W, msc, T, kpow, epow, mono;
    std::vector<int> stack;
    double bTFloor;
    gfunc_t gf; cfunc_t cf;
    int momStride;
    double deadline; // absolute CPU time; 0 = no deadline
    bool aborted;
    long long nM2L = 0, nDirect = 0, nVisit = 0, nDPairs = 0;

    Solver(Tree &tr, const Poly &p, double th, double *o, double dl = 0.0, int grp = 1)
        : t(tr), po(p), theta(th), theta2(th * th), out(o), deadline(dl), aborted(false) {
        momStride = tr.momStride;
        groupSize = std::max(1, grp);
        P = po.P; Q = po.Q; KP = po.KP; KQ = po.KQ; KG = po.KG; KU = po.KU;
        G.resize(KG); W.resize(KQ); msc.resize(KP); T.resize(KQ);
        kpow.resize(P + 1); epow.resize(Q + 1); mono.resize(KU);
        stack.reserve(4096);
        bTFloor = 1e-12 * std::max(t.rootRad, 1e-12);
        gf = get_gfunc(P, Q); cf = get_cfunc(P, Q);
    }

    // M2L: accumulate source cell s contribution into Wout for target centre
    inline void m2l(int s, double tx, double ty, double tz, double bT, double *Wout) {
        const Node &nd = t.nd[s];
        double dx = tx - nd.cx, dy = ty - nd.cy, dz = tz - nd.cz;
        double R2 = dx * dx + dy * dy + dz * dz + EPS2;
        double invR = 1.0 / sqrt(R2);
        double dh[3] = {dx * invR, dy * invR, dz * invR};
        // derivative tensor G
        if (gf) gf(dh, G.data());
        else {
            G[0] = 1.0;
            for (size_t b = 0; b < po.bOut.size(); ++b) {
                double v = 0;
                int st = po.bStart[b], en = st + po.bCnt[b];
                for (int k = st; k < en; ++k) {
                    double co = po.tCo[k];
                    if (po.tDh[k] >= 0) co *= dh[po.tDh[k]];
                    v += co * G[po.tSlot[k]];
                }
                G[po.bOut[b]] = -v;
            }
        }
        double kap = nd.rad * invR, eta = bT * invR;
        kpow[0] = 1; for (int d = 1; d <= P; ++d) kpow[d] = kpow[d - 1] * kap;
        epow[0] = 1; for (int d = 1; d <= Q; ++d) epow[d] = epow[d - 1] * eta;
        double pref = (t.m0 * nd.cnt) * invR / bT;
        const double *M = t.mom + (size_t)s * momStride;
        for (int m = 0; m < KP; ++m) msc[m] = M[m] * kpow[po.mdeg[m]];
        if (cf) {
            cf(msc.data(), G.data(), T.data());
            for (int g = 0; g < KQ; ++g)
                Wout[g] += (pref * epow[po.wdeg[g]] * po.winvf[g]) * T[g];
        } else {
            for (size_t g = 0; g < po.gW.size(); ++g) {
                double v = 0;
                int st = po.gStart[g], en = st + po.gCnt[g];
                for (int k = st; k < en; ++k) v += msc[po.pM[k]] * G[po.pG[k]];
                int w = po.gW[g];
                Wout[w] += pref * epow[po.wdeg[w]] * po.gInv[g] * v;
            }
        }
    }

    // evaluate local expansion Wc about (cx,cy,cz) with radius bT at a particle
    inline void eval_poly(const double *Wc, double cx, double cy, double cz,
                          double bT, double xi, double yi, double zi,
                          double &ax, double &ay, double &az) {
        double invBT = 1.0 / bT;
        double dh[3] = {(xi - cx) * invBT, (yi - cy) * invBT, (zi - cz) * invBT};
        mono[0] = 1.0;
        for (size_t u = 0; u < po.uPar.size(); ++u)
            mono[u + 1] = mono[po.uPar[u]] * dh[po.uAxis[u]];
        double a0 = 0, a1 = 0, a2 = 0;
        for (size_t u = 0; u < po.evMono.size(); ++u) {
            double mv = mono[po.evMono[u]];
            a0 += po.evC0[u] * Wc[po.evW0[u]] * mv;
            a1 += po.evC1[u] * Wc[po.evW1[u]] * mv;
            a2 += po.evC2[u] * Wc[po.evW2[u]] * mv;
        }
        ax += a0; ay += a1; az += a2;
    }

    // exact direct interaction of target leaf [t0,t1) with source leaf [s0,s1)
    inline void direct(int t0, int t1, int s0, int s1, double *acc) {
        const __m256d veps = _mm256_set1_pd(EPS2);
        const __m256d vmas = _mm256_set1_pd(t.m0);
        for (int i = t0; i < t1; ++i) {
            __m256d vxi = _mm256_set1_pd(t.px[i]);
              __m256d vyi = _mm256_set1_pd(t.py[i]);
            __m256d vzi = _mm256_set1_pd(t.pz[i]);
            __m256d ax = _mm256_setzero_pd(), ay = _mm256_setzero_pd(), az = _mm256_setzero_pd();
            int j = s0;
            for (; j + 4 <= s1; j += 4) {
                __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(t.px + j), vxi);
                __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(t.py + j), vyi);
                __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(t.pz + j), vzi);
                __m256d r2 = _mm256_add_pd(veps,
                    _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                                  _mm256_mul_pd(dz, dz)));
                __m256d rr = _mm256_sqrt_pd(r2);
                __m256d f = _mm256_div_pd(vmas, _mm256_mul_pd(r2, rr));
                ax = _mm256_fmadd_pd(dx, f, ax);
                ay = _mm256_fmadd_pd(dy, f, ay);
                az = _mm256_fmadd_pd(dz, f, az);
            }
            double a[4], b[4], c[4];
            _mm256_storeu_pd(a, ax); _mm256_storeu_pd(b, ay); _mm256_storeu_pd(c, az);
            double sx = a[0] + a[1] + a[2] + a[3];
            double sy = b[0] + b[1] + b[2] + b[3];
            double sz = c[0] + c[1] + c[2] + c[3];
            for (; j < s1; ++j) {
                double dx = t.px[j] - t.px[i], dy = t.py[j] - t.py[i], dz = t.pz[j] - t.pz[i];
                double r2 = dx * dx + dy * dy + dz * dz + EPS2;
                double f = t.m0 / (r2 * sqrt(r2));
                sx += dx * f; sy += dy * f; sz += dz * f;
            }
            int o = 3 * (i - t0);
            acc[o] += sx; acc[o + 1] += sy; acc[o + 2] += sz;
        }
    }

    std::vector<int> leafIds;
    void buildLeafList() {
        leafIds.clear();
        for (int id = 0; id < (int)t.nd.size(); ++id) {
            bool leaf = true;
            for (int o = 0; o < 8; ++o) if (t.nd[id].ch[o] >= 0) { leaf = false; break; }
            if (leaf) leafIds.push_back(id);
        }
    }

    int groupSize;
    inline bool isLeaf(int s) const {
        const Node &n = t.nd[s];
        for (int o = 0; o < 8; ++o) if (n.ch[o] >= 0) return false;
        return true;
    }
    // squared distance and MAC test for a cell against a target centre/radius
    inline bool mac_ok(int s, double cx, double cy, double cz, double bT, double th2) const {
        const CellQ &sn = t.cq[s];
        double dx = cx - sn.cx, dy = cy - sn.cy, dz = cz - sn.cz;
        double d2 = dx * dx + dy * dy + dz * dz;
        double sr = sn.rad + bT;
        return sr * sr <= th2 * d2;
    }

    // Dry traversal on an evenly spaced sample of leaf groups; returns
    // estimated total M2L calls and direct pair evaluations.
    void count_sample(double th, double &m2l_out, double &dp_out) {
        double th2 = th * th;
        long long m2l = 0, dp = 0;
        int NL = (int)leafIds.size();
        int G = groupSize;
        int NG = (NL + G - 1) / G;
        int step = NG / 40;
        if (step < 1) step = 1;
        int ns = 0;
        std::vector<int> stk, pending;
        stk.reserve(8192); pending.reserve(8192);
        for (int gi = 0; gi < NG; gi += step) {
            int g0 = gi * G, g1 = std::min(g0 + G, NL);
            int p0 = t.nd[leafIds[g0]].start;
            int p1 = t.nd[leafIds[g1 - 1]].end;
            // group centre / radius
            double gx = 0, gy = 0, gz = 0;
            for (int i = p0; i < p1; ++i) { gx += t.px[i]; gy += t.py[i]; gz += t.pz[i]; }
            double inv = 1.0 / (p1 - p0);
            gx *= inv; gy *= inv; gz *= inv;
            double bG = 0;
            for (int i = p0; i < p1; ++i) {
                double dx = t.px[i]-gx, dy = t.py[i]-gy, dz = t.pz[i]-gz;
                bG = std::max(bG, dx*dx+dy*dy+dz*dz);
            }
            bG = std::max(sqrt(bG), bTFloor);
            pending.clear();
            if (G > 1) {
                stk.clear(); stk.push_back(0);
                while (!stk.empty()) {
                    int s = stk.back(); stk.pop_back();
                    if (mac_ok(s, gx, gy, gz, bG, th2)) { m2l++; continue; }
                    if (isLeaf(s)) pending.push_back(s);
                    else for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stk.push_back(t.nd[s].ch[o]);
                }
            } else {
                pending.push_back(0);
            }
            // now, for each leaf, refine pending
            for (int li = g0; li < g1; ++li) {
                const Node &nd = t.nd[leafIds[li]];
                int nl = nd.end - nd.start;
                double bT = std::max(nd.rad, bTFloor);
                stk = pending;
                while (!stk.empty()) {
                    int s = stk.back(); stk.pop_back();
                    if (mac_ok(s, nd.cx, nd.cy, nd.cz, bT, th2)) { m2l++; continue; }
                    if (isLeaf(s)) dp += (long long)nl * (t.nd[s].end - t.nd[s].start);
                    else for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stk.push_back(t.nd[s].ch[o]);
                }
            }
            ns++;
        }
        m2l_out = (double)m2l / ns * NG;
        dp_out = (double)dp / ns * NG;
    }

    // Dry traversal for the M2P (4-particle group) path.
    void count_sample_m2p(double th, double &cells_out, double &dp_out, double &vis_out) {
        double th2 = th * th;
        long long cells = 0, dp = 0, vis = 0;
        int NG = (t.N + 3) / 4;
        int want = (t.N >= 1048576) ? 1024 : 256;
        int step = NG / want;
        if (step < 1) step = 1;
        int ns = 0;
        std::vector<int> stk; stk.reserve(8192);
        for (int gi = 0; gi < NG; gi += step) {
            int g0 = gi * 4;
            double tx[4], ty[4], tz[4];
            for (int k = 0; k < 4; ++k) {
                int i = std::min(g0 + k, t.N - 1);
                tx[k] = t.px[i]; ty[k] = t.py[i]; tz[k] = t.pz[i];
            }
            double cx = 0.25*(tx[0]+tx[1]+tx[2]+tx[3]);
            double cy = 0.25*(ty[0]+ty[1]+ty[2]+ty[3]);
            double cz = 0.25*(tz[0]+tz[1]+tz[2]+tz[3]);
            double b2 = 0;
            for (int k = 0; k < 4; ++k) {
                double dx=tx[k]-cx, dy=ty[k]-cy, dz=tz[k]-cz;
                b2 = std::max(b2, dx*dx+dy*dy+dz*dz);
            }
            double bT = std::max(sqrt(b2), bTFloor);
            stk.clear(); stk.push_back(0);
            while (!stk.empty()) {
                int s = stk.back(); stk.pop_back(); vis++;
                if (mac_ok(s, cx, cy, cz, bT, th2)) cells++;
                else if (isLeaf(s)) dp += 4LL*(t.nd[s].end - t.nd[s].start);
                else for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stk.push_back(t.nd[s].ch[o]);
            }
            ns++;
        }
        cells_out = (double)cells / ns * NG;
        dp_out = (double)dp / ns * NG;
    }

    // u^{-3/2}, u^{-5/2}, u^{-7/2}, u^{-9/2} via float rsqrt + 2 Newton steps
    static inline void invsqrt_powers(__m256d u, __m256d &s3, __m256d &s5,
                                      __m256d &s7, __m256d &s9) {
        __m128 uf = _mm256_cvtpd_ps(u);
        __m128 yf = _mm_rsqrt_ps(uf);
        __m256d y = _mm256_cvtps_pd(yf);
        const __m256d half = _mm256_set1_pd(0.5), th = _mm256_set1_pd(1.5);
        __m256d hu = _mm256_mul_pd(half, u);
        for (int k = 0; k < 2; ++k) {
            __m256d yy = _mm256_mul_pd(y, y);
            __m256d e = _mm256_fnmadd_pd(hu, yy, th);
            y = _mm256_mul_pd(y, e);
        }
        __m256d iv = _mm256_mul_pd(y, y);
        s3 = _mm256_mul_pd(iv, y);
        s5 = _mm256_mul_pd(s3, iv);
        s7 = _mm256_mul_pd(s5, iv);
        s9 = _mm256_mul_pd(s7, iv);
    }

    // Quadrupole (order-2) multipole evaluated directly at 4 target particles.
    int qxx, qyy, qzz, qxy, qxz, qyz;
    void initQuad() {
        qxx = po.mid[i3(2,0,0,po.side)]; qyy = po.mid[i3(0,2,0,po.side)];
        qzz = po.mid[i3(0,0,2,po.side)]; qxy = po.mid[i3(1,1,0,po.side)];
        qxz = po.mid[i3(1,0,1,po.side)]; qyz = po.mid[i3(0,1,1,po.side)];
    }
    inline void m2p_quad(int s, const __m256d X, const __m256d Y, const __m256d Z,
                         __m256d &ax, __m256d &ay, __m256d &az) {
        const CellQ &c = t.cq[s];
        const __m256d vcx = _mm256_set1_pd(c.cx), vcy = _mm256_set1_pd(c.cy),
                      vcz = _mm256_set1_pd(c.cz), veps = _mm256_set1_pd(EPS2);
        __m256d rx = _mm256_sub_pd(X, vcx), ry = _mm256_sub_pd(Y, vcy), rz = _mm256_sub_pd(Z, vcz);
        __m256d u = _mm256_add_pd(veps, _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(rx,rx), _mm256_mul_pd(ry,ry)),
            _mm256_mul_pd(rz,rz)));
        __m256d iv = _mm256_div_pd(_mm256_set1_pd(1.0), u);
        __m256d sq = _mm256_sqrt_pd(iv);
        __m256d s3 = _mm256_mul_pd(iv, sq);
        __m256d s5 = _mm256_mul_pd(s3, iv);
        __m256d s7 = _mm256_mul_pd(s5, iv);
        __m256d mm = _mm256_set1_pd(c.mass);
        if (c.r2 == 0.0) {
            __m256d f3 = _mm256_mul_pd(mm, s3);
            ax = _mm256_fnmadd_pd(rx, f3, ax);
            ay = _mm256_fnmadd_pd(ry, f3, ay);
            az = _mm256_fnmadd_pd(rz, f3, az);
            return;
        }
        double vxx=c.qxx, vyy=c.qyy, vzz=c.qzz;
        double vxy=c.qxy, vxz=c.qxz, vyz=c.qyz;
        double tr=c.tr;
        double r2=c.r2;
        __m256d Sx = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(vxx), rx),
            _mm256_mul_pd(_mm256_set1_pd(vxy), ry)),
            _mm256_mul_pd(_mm256_set1_pd(vxz), rz));
        __m256d Sy = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(vxy), rx),
            _mm256_mul_pd(_mm256_set1_pd(vyy), ry)),
            _mm256_mul_pd(_mm256_set1_pd(vyz), rz));
        __m256d Sz = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(vxz), rx),
            _mm256_mul_pd(_mm256_set1_pd(vyz), ry)),
            _mm256_mul_pd(_mm256_set1_pd(vzz), rz));
        __m256d rSr = _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(rx,Sx), _mm256_mul_pd(ry,Sy)), _mm256_mul_pd(rz,Sz));
        __m256d vr2 = _mm256_set1_pd(r2);
        __m256d f3 = _mm256_mul_pd(mm, s3);
        __m256d f5 = _mm256_mul_pd(_mm256_mul_pd(mm, vr2), s5);
        __m256d f7 = _mm256_mul_pd(_mm256_mul_pd(mm, vr2), s7);
        ax = _mm256_sub_pd(ax, _mm256_mul_pd(rx, f3));
        ay = _mm256_sub_pd(ay, _mm256_mul_pd(ry, f3));
        az = _mm256_sub_pd(az, _mm256_mul_pd(rz, f3));
        ax = _mm256_fmadd_pd(_mm256_add_pd(
                 _mm256_mul_pd(_mm256_set1_pd(3.0), Sx),
                 _mm256_mul_pd(_mm256_set1_pd(1.5*tr), rx)), f5, ax);
        ay = _mm256_fmadd_pd(_mm256_add_pd(
                 _mm256_mul_pd(_mm256_set1_pd(3.0), Sy),
                 _mm256_mul_pd(_mm256_set1_pd(1.5*tr), ry)), f5, ay);
        az = _mm256_fmadd_pd(_mm256_add_pd(
                 _mm256_mul_pd(_mm256_set1_pd(3.0), Sz),
                 _mm256_mul_pd(_mm256_set1_pd(1.5*tr), rz)), f5, az);
        __m256d c7 = _mm256_mul_pd(_mm256_set1_pd(-7.5), f7);
        __m256d g = _mm256_mul_pd(rSr, c7);
        ax = _mm256_fmadd_pd(rx, g, ax);
        ay = _mm256_fmadd_pd(ry, g, ay);
        az = _mm256_fmadd_pd(rz, g, az);
    }
    int yz_slot() const { return qyz; }

    inline void direct_vec(int s0, int s1, const __m256d X, const __m256d Y, const __m256d Z,
                           __m256d &ax, __m256d &ay, __m256d &az) {
        const __m256d veps = _mm256_set1_pd(EPS2), vmas = _mm256_set1_pd(t.m0);
        for (int j = s0; j < s1; ++j) {
            __m256d rx = _mm256_sub_pd(_mm256_set1_pd(t.px[j]), X);
            __m256d ry = _mm256_sub_pd(_mm256_set1_pd(t.py[j]), Y);
            __m256d rz = _mm256_sub_pd(_mm256_set1_pd(t.pz[j]), Z);
            __m256d u = _mm256_add_pd(veps, _mm256_add_pd(
                _mm256_add_pd(_mm256_mul_pd(rx,rx), _mm256_mul_pd(ry,ry)),
                _mm256_mul_pd(rz,rz)));
            __m256d rr = _mm256_sqrt_pd(u);
            __m256d f = _mm256_div_pd(vmas, _mm256_mul_pd(u, rr));
            ax = _mm256_fmadd_pd(rx, f, ax);
            ay = _mm256_fmadd_pd(ry, f, ay);
            az = _mm256_fmadd_pd(rz, f, az);
        }
    }

    int abortGroup;
    void run_m2p(int startGroup = 0, int stride = 1, int maxGroups = 0x7fffffff) {
        initQuad();
        int ngroup = 0;
        abortGroup = (t.N + 3) / 4;
        for (int g0 = startGroup * 4; g0 < t.N && ngroup < maxGroups; g0 += 4 * stride) {
            if (deadline > 0 && (ngroup & 255) == 0 && cpu_time() > deadline) {
                aborted = true; abortGroup = g0 / 4; return;
            }
            ngroup++;
            int n = std::min(4, t.N - g0);
            double tx[4], ty[4], tz[4];
            for (int k = 0; k < 4; ++k) {
                int i = std::min(g0 + k, t.N - 1);
                tx[k] = t.px[i]; ty[k] = t.py[i]; tz[k] = t.pz[i];
            }
            double cx = 0.25*(tx[0]+tx[1]+tx[2]+tx[3]);
            double cy = 0.25*(ty[0]+ty[1]+ty[2]+ty[3]);
            double cz = 0.25*(tz[0]+tz[1]+tz[2]+tz[3]);
            double b2 = 0;
            for (int k = 0; k < n; ++k) {
                double dx=tx[k]-cx, dy=ty[k]-cy, dz=tz[k]-cz;
                b2 = std::max(b2, dx*dx+dy*dy+dz*dz);
            }
            double bT = std::max(sqrt(b2), bTFloor);
            __m256d X = _mm256_loadu_pd(tx), Y = _mm256_loadu_pd(ty), Z = _mm256_loadu_pd(tz);
            __m256d ax = _mm256_setzero_pd(), ay = _mm256_setzero_pd(), az = _mm256_setzero_pd();
            stack.clear(); stack.push_back(0);
            while (!stack.empty()) {
                int s = stack.back(); stack.pop_back(); nVisit++;
                if (mac_ok(s, cx, cy, cz, bT, theta2)) {
                    nM2L++; m2p_quad(s, X, Y, Z, ax, ay, az);
                } else if (isLeaf(s)) {
                    nDirect++; nDPairs += 4LL*(t.nd[s].end - t.nd[s].start);
                    direct_vec(t.nd[s].start, t.nd[s].end, X, Y, Z, ax, ay, az);
                } else {
                    for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stack.push_back(t.nd[s].ch[o]);
                }
            }
            double vx[4], vy[4], vz[4];
            _mm256_storeu_pd(vx, ax); _mm256_storeu_pd(vy, ay); _mm256_storeu_pd(vz, az);
            for (int k = 0; k < n; ++k) {
                uint32_t oi = t.perm[g0+k];
                out[3*oi] = vx[k]; out[3*oi+1] = vy[k]; out[3*oi+2] = vz[k];
            }
        }
    }

    void run() {
        const int L = 512;
        std::vector<double> acc(3 * L);
        std::vector<double> WG(KQ), WT(KQ);
        std::vector<int> pending; pending.reserve(8192);
        int NL = (int)leafIds.size();
        int G = groupSize;
        int ngroup = 0;
        for (int g0 = 0; g0 < NL; g0 += G) {
            int g1 = std::min(g0 + G, NL);
            if (deadline > 0 && (ngroup & 15) == 0 && cpu_time() > deadline) {
                aborted = true; return;
            }
            ngroup++;
            int p0 = t.nd[leafIds[g0]].start;
            int p1 = t.nd[leafIds[g1 - 1]].end;
            double gx = 0, gy = 0, gz = 0;
            for (int i = p0; i < p1; ++i) { gx += t.px[i]; gy += t.py[i]; gz += t.pz[i]; }
            double inv = 1.0 / (p1 - p0);
            gx *= inv; gy *= inv; gz *= inv;
            double bGsq = 0;
            for (int i = p0; i < p1; ++i) {
                double dx = t.px[i]-gx, dy = t.py[i]-gy, dz = t.pz[i]-gz;
                bGsq = std::max(bGsq, dx*dx+dy*dy+dz*dz);
            }
            double bG = std::max(sqrt(bGsq), bTFloor);
            bool useGroup = (G > 1);
            std::fill(WG.begin(), WG.end(), 0.0);
            pending.clear();
            if (useGroup) {
                stack.clear(); stack.push_back(0);
                while (!stack.empty()) {
                    int s = stack.back(); stack.pop_back();
                    nVisit++;
                    if (mac_ok(s, gx, gy, gz, bG, theta2)) {
                        nM2L++; m2l(s, gx, gy, gz, bG, WG.data());
                    } else if (isLeaf(s)) {
                        pending.push_back(s);
                    } else {
                        for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stack.push_back(t.nd[s].ch[o]);
                    }
                }
            } else {
                pending.push_back(0);
            }
            for (int li = g0; li < g1; ++li) {
                int id = leafIds[li];
                const Node &nd = t.nd[id];
                int t0 = nd.start, t1 = nd.end;
                int nl = t1 - t0;
                if (3 * nl > (int)acc.size()) acc.resize(3 * nl);
                double cx = nd.cx, cy = nd.cy, cz = nd.cz;
                double bT = std::max(nd.rad, bTFloor);
                double *Wuse = useGroup ? WT.data() : WG.data();
                std::fill(Wuse, Wuse + KQ, 0.0);
                std::fill(acc.data(), acc.data() + 3 * nl, 0.0);
                stack = pending;
                while (!stack.empty()) {
                    int s = stack.back(); stack.pop_back();
                    nVisit++;
                    if (mac_ok(s, cx, cy, cz, bT, theta2)) {
                        nM2L++; m2l(s, cx, cy, cz, bT, Wuse);
                    } else if (isLeaf(s)) {
                        nDirect++; nDPairs += (long long)nl * (t.nd[s].end - t.nd[s].start);
                        direct(t0, t1, t.nd[s].start, t.nd[s].end, acc.data());
                    } else {
                        for (int o = 0; o < 8; ++o) if (t.nd[s].ch[o] >= 0) stack.push_back(t.nd[s].ch[o]);
                    }
                }
                for (int i = t0; i < t1; ++i) {
                    double ax = 0, ay = 0, az = 0;
                    if (useGroup) eval_poly(WG.data(), gx, gy, gz, bG, t.px[i], t.py[i], t.pz[i], ax, ay, az);
                    eval_poly(Wuse, cx, cy, cz, bT, t.px[i], t.py[i], t.pz[i], ax, ay, az);
                    int o = 3 * (i - t0);
                    uint32_t oi = t.perm[i];
                    out[3 * oi] = ax + acc[o];
                    out[3 * oi + 1] = ay + acc[o + 1];
                    out[3 * oi + 2] = az + acc[o + 2];
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Measured per-interaction CPU costs on the development core (conservative).
static const double TM2L[7] = {0, 0, 0.12e-6, 0.24e-6, 0.65e-6, 1.45e-6, 12e-6};
static const double TPAIR = 1.6e-9;
static const double TCELL = 25e-9;   // one M2P cell evaluation (4 targets)
static const double TPAIR2 = 0.75e-9; // one direct pair in the M2P path
static const double TVISIT = 7.0e-9;  // one tree-node MAC test

static double default_limit(int N) {
    if (N <= 65536) return 0.9;
    if (N <= 262144) return 4.7;
    if (N <= 1048576) return 21.8;
    return 100.1;
}

int main(int argc, char **argv) {
    double t0cpu = cpu_time(), t0wall = wall_time();
    if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 1; }
    int verbose = getenvip("GAL_VERBOSE", 0);
    int forced = getenv("GAL_P") || getenv("GAL_Q") || getenv("GAL_THETA") ||
                 getenv("GAL_LEAF") || getenv("GAL_MODE");
    int P = getenvip("GAL_P", 2);
    int Q = getenvip("GAL_Q", 2);
    double theta = getenvd("GAL_THETA", 0.4);
    int leaf = getenvip("GAL_LEAF", 64);
    int maxDepth = getenvip("GAL_MAXDEPTH", 48);
    int group = getenvip("GAL_GROUP", 1);
    int mode = getenvip("GAL_MODE", 1);
    double limit = getenvd("GALAXY_CPU_LIMIT_S", -1.0);
    if (!(limit > 0)) limit = default_limit(65536);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int N = (int)(fsz / 32);
    if (N <= 0) return 1;
    double *raw = (double *)malloc((size_t)fsz);
    if (fread(raw, 1, fsz, f) != (size_t)fsz) { perror("read"); return 1; }
    fclose(f);
    if (getenv("GALAXY_CPU_LIMIT_S") == NULL) limit = default_limit(N);

    int PMAX = mode ? 2 : (forced ? std::max(P, 4) : 4);
    Poly poMax(PMAX, PMAX);
    Tree tr(N, poMax, leaf, maxDepth);
    tr.make_keys_and_sort(raw);
    tr.build(0, N, 0);
    tr.compute_moments();
    if (mode) tr.build_cellq();
    double buildTime = cpu_time() - t0cpu;

    double *out = (double *)malloc(sizeof(double) * 3 * N);
    double *tmp = (double *)malloc(sizeof(double) * 3 * N);
    if (!out || !tmp) return 1;
    memset(out, 0, sizeof(double) * 3 * N);

    int chosenP = 2, chosenQ = 2;
    double chosenTheta = theta;
    double writeReserve = 0.05 + 8.0e-8 * (double)N;
    double hardDeadline = 0.0;
    double cc[32]={0}, dd[32]={0}, vv[32]={0};
    double unit = -1.0;

    if (!forced) {
        const int NT = 25;
        const double TH[NT] = {0.14,0.155,0.17,0.185,0.20,0.22,0.24,0.26,0.28,
                               0.31,0.34,0.37,0.40,0.44,0.48,0.53,0.58,0.64,
                               0.70,0.78,0.87,0.98,1.10,1.25,1.40};
        Poly poProbe(2, 2);
        Solver probe(tr, poProbe, 0.5, tmp, 0.0, group);
        probe.buildLeafList();
        if (mode) {
            for (int k = 0; k < NT; ++k) probe.count_sample_m2p(TH[k], cc[k], dd[k], vv[k]);
        } else {
            for (int k = 0; k < NT; ++k) { probe.count_sample(TH[k], cc[k], dd[k]); vv[k] = 0; }
        }
        int NG = (N + 3) / 4;
        int stride = NG / 96;
        if (stride < 1) stride = 1;
        double nsamp = (double)((NG + stride - 1) / stride);
        double cacheFactor = 1.0 + 0.055 * log2(std::max(1.0, (double)N / 65536.0));
        // Time a few CONTIGUOUS blocks of groups: this reproduces the cache
        // behaviour of the real pass (strided samples are badly biased at
        // large N because every sample starts with cold caches).
        unit = TCELL * cacheFactor;
        if (mode) {
            double tsum = 0, csum = 0, dsum = 0, vsum = 0;
            const double OFF[4] = {0.10, 0.35, 0.60, 0.85};
            int blk = 48;
            for (int b = 0; b < 4; ++b) {
                int off = (int)(OFF[b] * NG);
                if (off + blk >= NG) off = NG - blk - 1;
                if (off < 0) off = 0;
                Solver p2(tr, poProbe, 0.45, tmp, 0.0, group);
                double tb = cpu_time();
                p2.run_m2p(off, 1, blk);
                tsum += cpu_time() - tb;
                csum += p2.nM2L; dsum += p2.nDPairs; vsum += p2.nVisit;
            }
            double w = csum + 0.03*dsum + 0.15*vsum;
            if (w > 0 && tsum > 0) unit = tsum / w;
        }
        double avail = 0.86 * limit - (cpu_time() - t0cpu) - writeReserve;
        int chosenIdx = NT - 1;
        for (int k = 0; k < NT; ++k) {
            double T = 1.12 * unit * (cc[k] + 0.03*dd[k] + 0.15*vv[k]);
            if (!mode) T = (cc[k]*TM2L[4] + dd[k]*TPAIR) * cacheFactor;
            if (T <= 0.95 * avail) { chosenIdx = k; break; }
        }
        chosenTheta = TH[chosenIdx];
        hardDeadline = t0cpu + 0.90 * limit - writeReserve;
        if (verbose)
            fprintf(stderr, "limit=%.2f build=%.2f nodes=%zu chosen theta=%.3f mode=%d\n",
                    limit, buildTime, tr.nd.size(), chosenTheta, mode);
    }

    Poly poMain(chosenP, chosenQ);
    Solver sv(tr, poMain, chosenTheta, out, hardDeadline, group);
    sv.buildLeafList();
    if (mode) sv.run_m2p(); else sv.run();
    if (sv.aborted) {
        // Safety net: finish unfinished groups with the best angle affordable
        // in the CPU time that remains.
        double thfb = 1.4;
        int NG = (N + 3) / 4;
        double frac = (double)(NG - sv.abortGroup) / (double)NG;
        double rem = (t0cpu + 0.93 * limit - writeReserve) - cpu_time();
        const double THS[12] = {0.25,0.31,0.38,0.47,0.53,0.60,0.70,0.85,1.05,1.40,1.8,2.5};
        for (int k = 0; k < 12; ++k) {
            // cost of a pass at THS[k], estimated from the sampled counts by
            // interpolating the ladder that was measured above
            double T = 0;
            if (!forced) {
                // find nearest sampled angle
                const double LAD[25] = {0.14,0.155,0.17,0.185,0.20,0.22,0.24,0.26,0.28,
                                        0.31,0.34,0.37,0.40,0.44,0.48,0.53,0.58,0.64,
                                        0.70,0.78,0.87,0.98,1.10,1.25,1.40};
                int best = 0; double bd = 1e9;
                for (int j = 0; j < 25; ++j) if (fabs(LAD[j]-THS[k]) < bd) { bd=fabs(LAD[j]-THS[k]); best=j; }
                double u = (unit > 0 ? unit : TCELL);
                T = 1.4 * frac * u * (cc[best] + 0.03 * dd[best] + 0.15 * vv[best]);
            }
            if (rem <= 0) { thfb = 2.5; break; }
            if (T <= rem || k == 11) { thfb = THS[k]; break; }
        }
        double fbDeadline = t0cpu + 0.965 * limit - writeReserve;
        Solver fb(tr, poMain, thfb, out, fbDeadline, group);
        if (mode) fb.run_m2p(sv.abortGroup);
        if (verbose) fprintf(stderr, "FALLBACK at group %d theta_fb=%.2f\n", sv.abortGroup, thfb);
    }

    FILE *g = fopen(argv[2], "wb");
    if (!g) { perror("open output"); return 1; }
    fwrite(out, 24, N, g);
    fclose(g);
    free(out); free(tmp); free(raw);
    if (verbose) {
        fprintf(stderr, "N=%d theta=%.3f leaf=%d nodes=%zu cpu=%.3f wall=%.3f"
                " | cells=%lld visit=%lld direct=%lld dpairs=%lld aborted=%d\n",
                N, chosenTheta, leaf, tr.nd.size(),
                cpu_time() - t0cpu, wall_time() - t0wall,
                sv.nM2L, sv.nVisit, sv.nDirect, sv.nDPairs, (int)sv.aborted);
    }
    return 0;
}
