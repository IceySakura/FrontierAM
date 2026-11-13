// Gravitational accelerations with Plummer softening via a Cartesian FMM
// (dual-tree traversal, Taylor expansions of the softened kernel).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sys/resource.h>

typedef double v4d __attribute__((vector_size(32)));

static const double EPS2 = 1e-4;  // softening 0.01^2

static double cpu_now() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec + ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
}

// ---------------------------------------------------------------- tables
static int P, NC, NCm1;
static std::vector<int> KA, KB, KC, DEG;
static int IDX[24][24][24];
static std::vector<int> R1x, R1y, R1z, R2x, R2y, R2z;
static std::vector<double> C1, C2, FACT, SGN;
static std::vector<int> M2Loff, M2Lcnt, M2Lidx;
static std::vector<int> MMoff, MMcnt, MMsrc, MMmono;  // M2M: target k <- (j, k-j)
static std::vector<int> LLoff, LLcnt, LLsrc, LLmono;  // L2L: target j <- (n, n-j)
static std::vector<int> MOpar, MOcomp;
static std::vector<double> MOinv;
static std::vector<int> GPx, GPy, GPz;

static inline int nc_of(int p) { return p < 0 ? 0 : (p + 1) * (p + 2) * (p + 3) / 6; }

static void init_tables(int p) {
    P = p;
    NC = nc_of(p);
    NCm1 = nc_of(p - 1);
    KA.clear(); KB.clear(); KC.clear(); DEG.clear();
    for (int n = 0; n <= P; n++)
        for (int a = n; a >= 0; a--)
            for (int b = n - a; b >= 0; b--) {
                int c = n - a - b;
                IDX[a][b][c] = (int)KA.size();
                KA.push_back(a); KB.push_back(b); KC.push_back(c); DEG.push_back(n);
            }
    auto id = [&](int a, int b, int c) -> int {
        if (a < 0 || b < 0 || c < 0) return NC;  // zero slot
        if (a + b + c > P) return -1;
        return IDX[a][b][c];
    };
    auto fac = [](int k) { double f = 1; for (int i = 2; i <= k; i++) f *= i; return f; };
    R1x.assign(NC, NC); R1y.assign(NC, NC); R1z.assign(NC, NC);
    R2x.assign(NC, NC); R2y.assign(NC, NC); R2z.assign(NC, NC);
    C1.assign(NC, 0); C2.assign(NC, 0); FACT.assign(NC, 1); SGN.assign(NC, 0);
    MOpar.assign(NC, 0); MOcomp.assign(NC, 0); MOinv.assign(NC, 0);
    for (int t = 0; t < NC; t++) {
        int a = KA[t], b = KB[t], c = KC[t], n = DEG[t];
        FACT[t] = fac(a) * fac(b) * fac(c);
        SGN[t] = (n & 1) ? 1.0 : -1.0;  // -(-1)^n
        if (n >= 1) {
            R1x[t] = id(a - 1, b, c); R1y[t] = id(a, b - 1, c); R1z[t] = id(a, b, c - 1);
            R2x[t] = id(a - 2, b, c); R2y[t] = id(a, b - 2, c); R2z[t] = id(a, b, c - 2);
            C1[t] = (2.0 * n - 1.0) / n;
            C2[t] = (n - 1.0) / n;
            if (a > 0) { MOpar[t] = id(a - 1, b, c); MOcomp[t] = 0; MOinv[t] = 1.0 / a; }
            else if (b > 0) { MOpar[t] = id(a, b - 1, c); MOcomp[t] = 1; MOinv[t] = 1.0 / b; }
            else { MOpar[t] = id(a, b, c - 1); MOcomp[t] = 2; MOinv[t] = 1.0 / c; }
        }
    }
    // M2L: for each n, all k with |k| <= P-|n|
    M2Loff.assign(NC, 0); M2Lcnt.assign(NC, 0); M2Lidx.clear();
    for (int t = 0; t < NC; t++) {
        M2Loff[t] = (int)M2Lidx.size();
        int cnt = nc_of(P - DEG[t]);
        M2Lcnt[t] = cnt;
        for (int k = 0; k < cnt; k++) M2Lidx.push_back(IDX[KA[t] + KA[k]][KB[t] + KB[k]][KC[t] + KC[k]]);
    }
    // M2M: Q'_k = sum_{j<=k} Q_j sf_{k-j}
    MMoff.assign(NC, 0); MMcnt.assign(NC, 0); MMsrc.clear(); MMmono.clear();
    for (int t = 0; t < NC; t++) {
        MMoff[t] = (int)MMsrc.size();
        for (int j = 0; j < NC; j++)
            if (KA[j] <= KA[t] && KB[j] <= KB[t] && KC[j] <= KC[t]) {
                MMsrc.push_back(j);
                MMmono.push_back(IDX[KA[t] - KA[j]][KB[t] - KB[j]][KC[t] - KC[j]]);
            }
        MMcnt[t] = (int)MMsrc.size() - MMoff[t];
    }
    // L2L: L'_j = sum_{n>=j} L_n sf_{n-j}
    LLoff.assign(NC, 0); LLcnt.assign(NC, 0); LLsrc.clear(); LLmono.clear();
    for (int t = 0; t < NC; t++) {
        LLoff[t] = (int)LLsrc.size();
        for (int n = 0; n < NC; n++)
            if (KA[n] >= KA[t] && KB[n] >= KB[t] && KC[n] >= KC[t]) {
                LLsrc.push_back(n);
                LLmono.push_back(IDX[KA[n] - KA[t]][KB[n] - KB[t]][KC[n] - KC[t]]);
            }
        LLcnt[t] = (int)LLsrc.size() - LLoff[t];
    }
    GPx.assign(NCm1, 0); GPy.assign(NCm1, 0); GPz.assign(NCm1, 0);
    for (int t = 0; t < NCm1; t++) {
        GPx[t] = IDX[KA[t] + 1][KB[t]][KC[t]];
        GPy[t] = IDX[KA[t]][KB[t] + 1][KC[t]];
        GPz[t] = IDX[KA[t]][KB[t]][KC[t] + 1];
    }
}

// sf_m = s^m / m!
static inline void monomials(double sx, double sy, double sz, double* sf) {
    double s[3] = {sx, sy, sz};
    sf[0] = 1.0;
    for (int t = 1; t < NC; t++) sf[t] = sf[MOpar[t]] * s[MOcomp[t]] * MOinv[t];
}

// ---------------------------------------------------------------- data
static int N;
static std::vector<double> X, Y, Z, M, AX, AY, AZ;
struct Node {
    int start, cnt, child, nchild;
    double cx, cy, cz, r;
};
static std::vector<Node> nodes;
static std::vector<uint64_t> keys;
static int NLEAF = 32;
static std::vector<double> Q, L;  // per-node coefficient arrays (NC each)

static int build(int s, int e, int level) {
    int id = (int)nodes.size();
    nodes.push_back(Node{s, e - s, -1, 0, 0, 0, 0, 0});
    if (e - s <= NLEAF || level >= 21) return id;
    int shift = 3 * (20 - level);
    int bounds[9];
    bounds[0] = s;
    for (int o = 1; o < 8; o++) {
        // first index with octant >= o
        int lo = bounds[o - 1], hi = e;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if ((int)((keys[mid] >> shift) & 7) < o) lo = mid + 1; else hi = mid;
        }
        bounds[o] = lo;
    }
    bounds[8] = e;
    int nch = 0;
    for (int o = 0; o < 8; o++) if (bounds[o + 1] > bounds[o]) nch++;
    if (nch == 1) {
        // single child: descend without creating a node level
        nodes.pop_back();
        return build(s, e, level + 1);
    }
    int first = (int)nodes.size();
    nodes[id].child = first;
    nodes[id].nchild = nch;
    for (int o = 0; o < nch; o++) nodes.push_back(Node{0, 0, -1, 0, 0, 0, 0, 0});
    int c = 0;
    for (int o = 0; o < 8; o++) {
        if (bounds[o + 1] <= bounds[o]) continue;
        // build child into slot first + c
        int before = (int)nodes.size();
        int cid = build(bounds[o], bounds[o + 1], level + 1);
        // move built node into slot
        nodes[first + c] = nodes[cid];
        if (cid == before) {
            // the node was appended at 'before'; leave a hole (mark unused)
            nodes[cid].cnt = -1;
        }
        c++;
    }
    return id;
}

// ---------------------------------------------------------------- traversal
static double THETA2 = 0.25;
static bool REL_MAC = false;
static double TOL = 1e-8;
static std::vector<double> AMIN;  // per-node min |a| estimate
static std::vector<double> MASSN;
static double P2P_THRESH = 1000;  // if na*nb below this, use direct instead of M2L
static bool COUNT_MODE = false;
static double cnt_m2l = 0, cnt_p2p = 0;

static int pendA[4], pendB[4], npend = 0;
static std::vector<v4d> bufB, bufQA, bufQB;

static void m2l_flush() {
    if (npend == 0) return;
    v4d Rx, Ry, Rz, inv;
    for (int l = 0; l < 4; l++) {
        if (l < npend) {
            const Node& A = nodes[pendA[l]];
            const Node& B = nodes[pendB[l]];
            Rx[l] = B.cx - A.cx; Ry[l] = B.cy - A.cy; Rz[l] = B.cz - A.cz;
        } else {
            Rx[l] = 1.0; Ry[l] = 0.0; Rz[l] = 0.0;
        }
    }
    v4d rho2 = Rx * Rx + Ry * Ry + Rz * Rz + EPS2;
    inv = 1.0 / rho2;
    v4d* Bf = bufB.data();
    v4d b0;
    for (int l = 0; l < 4; l++) b0[l] = std::sqrt(inv[l]);
    Bf[0] = b0;
    Bf[NC] = (v4d){0, 0, 0, 0};
    for (int t = 1; t < NC; t++) {
        v4d s1 = Rx * Bf[R1x[t]] + Ry * Bf[R1y[t]] + Rz * Bf[R1z[t]];
        v4d s2 = Bf[R2x[t]] + Bf[R2y[t]] + Bf[R2z[t]];
        Bf[t] = (C1[t] * s1 - C2[t] * s2) * inv;
    }
    for (int t = 1; t < NC; t++) Bf[t] *= FACT[t];
    v4d* QA = bufQA.data();
    v4d* QB = bufQB.data();
    for (int l = 0; l < 4; l++) {
        if (l < npend) {
            const double* qa = &Q[(size_t)pendA[l] * NC];
            const double* qb = &Q[(size_t)pendB[l] * NC];
            for (int k = 0; k < NC; k++) {
                QA[k][l] = qa[k];
                QB[k][l] = (DEG[k] & 1) ? -qb[k] : qb[k];
            }
        } else {
            for (int k = 0; k < NC; k++) { QA[k][l] = 0; QB[k][l] = 0; }
        }
    }
    for (int t = 0; t < NC; t++) {
        const int* S = &M2Lidx[M2Loff[t]];
        int cnt = M2Lcnt[t];
        v4d acc1 = {0, 0, 0, 0}, acc2 = {0, 0, 0, 0};
        for (int k = 0; k < cnt; k++) {
            v4d b = Bf[S[k]];
            acc1 += QA[k] * b;
            acc2 += QB[k] * b;
        }
        double sg = SGN[t];
        for (int l = 0; l < npend; l++) {
            L[(size_t)pendB[l] * NC + t] += sg * acc1[l];
            L[(size_t)pendA[l] * NC + t] -= acc2[l];
        }
    }
    npend = 0;
}

static void p2p_mutual(int sa, int na, int sb, int nb) {
    const double* __restrict x = X.data();
    const double* __restrict y = Y.data();
    const double* __restrict z = Z.data();
    const double* __restrict m = M.data();
    double* __restrict ax = AX.data();
    double* __restrict ay = AY.data();
    double* __restrict az = AZ.data();
    for (int i = sa; i < sa + na; i++) {
        double xi = x[i], yi = y[i], zi = z[i], mi = m[i];
        double sx = 0, sy = 0, sz = 0;
#pragma GCC ivdep
        for (int j = sb; j < sb + nb; j++) {
            double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + EPS2;
            double inv = 1.0 / std::sqrt(r2);
            double inv3 = inv * inv * inv;
            double fj = m[j] * inv3, fi = mi * inv3;
            sx += fj * dx; sy += fj * dy; sz += fj * dz;
            ax[j] -= fi * dx; ay[j] -= fi * dy; az[j] -= fi * dz;
        }
        ax[i] += sx; ay[i] += sy; az[i] += sz;
    }
}

static void p2p_self(int s, int n) {
    const double* __restrict x = X.data();
    const double* __restrict y = Y.data();
    const double* __restrict z = Z.data();
    const double* __restrict m = M.data();
    for (int i = s; i < s + n; i++) {
        double xi = x[i], yi = y[i], zi = z[i];
        double sx = 0, sy = 0, sz = 0;
        for (int j = s; j < s + n; j++) {
            double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + EPS2;
            double inv = 1.0 / std::sqrt(r2);
            double f = m[j] * inv * inv * inv;
            sx += f * dx; sy += f * dy; sz += f * dz;
        }
        AX[i] += sx; AY[i] += sy; AZ[i] += sz;
    }
}

static void interact(int a, int b) {
    const Node& A = nodes[a];
    const Node& B = nodes[b];
    double dx = A.cx - B.cx, dy = A.cy - B.cy, dz = A.cz - B.cz;
    double d2 = dx * dx + dy * dy + dz * dz;
    double rs = A.r + B.r;
    bool accept = rs * rs < THETA2 * d2;
    if (accept && REL_MAC) {
        double q = rs * rs / d2, qp = 1.0;
        for (int k = 0; k < P; k += 2) qp *= q;  // (rs/d)^(2*ceil(P/2))
        double e = qp / d2;
        double mA = MASSN[a], mB = MASSN[b];
        accept = (mA * e <= TOL * AMIN[b]) && (mB * e <= TOL * AMIN[a]);
    }
    if (accept) {
        if ((double)A.cnt * B.cnt <= P2P_THRESH) {
            if (COUNT_MODE) cnt_p2p += (double)A.cnt * B.cnt;
            else p2p_mutual(A.start, A.cnt, B.start, B.cnt);
        } else {
            if (COUNT_MODE) cnt_m2l += 1;
            else {
                pendA[npend] = a; pendB[npend] = b; npend++;
                if (npend == 4) m2l_flush();
            }
        }
        return;
    }
    bool la = A.nchild == 0, lb = B.nchild == 0;
    if (la && lb) {
        if (COUNT_MODE) cnt_p2p += (double)A.cnt * B.cnt;
        else p2p_mutual(A.start, A.cnt, B.start, B.cnt);
        return;
    }
    if (!la && (lb || A.r >= B.r)) {
        int c0 = A.child, nc = A.nchild;
        for (int c = 0; c < nc; c++) interact(c0 + c, b);
    } else {
        int c0 = B.child, nc = B.nchild;
        for (int c = 0; c < nc; c++) interact(a, c0 + c);
    }
}

static void self_interact(int a) {
    const Node& A = nodes[a];
    if (A.nchild == 0) {
        if (COUNT_MODE) cnt_p2p += 0.5 * (double)A.cnt * A.cnt;
        else p2p_self(A.start, A.cnt);
        return;
    }
    int c0 = A.child, nc = A.nchild;
    for (int i = 0; i < nc; i++) self_interact(c0 + i);
    for (int i = 0; i < nc; i++)
        for (int j = i + 1; j < nc; j++) interact(c0 + i, c0 + j);
}


static int nnodes;
static double cm2l_of(int p) {
    // per mutual M2L (seconds) on the development host
    switch (p) {
        case 4: return 0.25e-6;
        case 5: return 0.4e-6;
        case 6: return 0.7e-6;
        case 7: return 1.1e-6;
        case 8: return 1.7e-6;
        case 9: return 2.6e-6;
        case 10: return 3.8e-6;
        case 11: return 5.5e-6;
        case 12: return 7.8e-6;
        default: return 1e-5 * std::pow(p / 12.0, 5);
    }
}

static void upward() {
    Q.assign((size_t)nnodes * NC, 0.0);
    L.assign((size_t)nnodes * NC, 0.0);
    std::vector<double> sf(NC + 1);
    for (int i = nnodes - 1; i >= 0; i--) {
        Node& nd = nodes[i];
        double* q = &Q[(size_t)i * NC];
        if (nd.nchild == 0) {
            for (int j = nd.start; j < nd.start + nd.cnt; j++) {
                monomials(X[j] - nd.cx, Y[j] - nd.cy, Z[j] - nd.cz, sf.data());
                for (int t = 0; t < NC; t++) q[t] += M[j] * sf[t];
            }
        } else {
            for (int c = nd.child; c < nd.child + nd.nchild; c++) {
                const Node& ch = nodes[c];
                monomials(ch.cx - nd.cx, ch.cy - nd.cy, ch.cz - nd.cz, sf.data());
                const double* qc = &Q[(size_t)c * NC];
                for (int t = 0; t < NC; t++) {
                    double acc = 0;
                    const int* src = &MMsrc[MMoff[t]];
                    const int* mo = &MMmono[MMoff[t]];
                    for (int u = 0; u < MMcnt[t]; u++) acc += qc[src[u]] * sf[mo[u]];
                    q[t] += acc;
                }
            }
        }
    }
    bufB.assign(NC + 1, (v4d){0, 0, 0, 0}); bufQA.assign(NC, (v4d){0, 0, 0, 0}); bufQB.assign(NC, (v4d){0, 0, 0, 0});
}

static void evaluate() {
    AX.assign(N, 0); AY.assign(N, 0); AZ.assign(N, 0);
    npend = 0;
    self_interact(0);
    m2l_flush();
    std::vector<double> sf(NC + 1);
    for (int i = 0; i < nnodes; i++) {
        const Node& nd = nodes[i];
        const double* l = &L[(size_t)i * NC];
        if (nd.nchild > 0) {
            for (int c = nd.child; c < nd.child + nd.nchild; c++) {
                const Node& ch = nodes[c];
                monomials(ch.cx - nd.cx, ch.cy - nd.cy, ch.cz - nd.cz, sf.data());
                double* lc = &L[(size_t)c * NC];
                for (int t = 0; t < NC; t++) {
                    double acc = 0;
                    const int* src = &LLsrc[LLoff[t]];
                    const int* mo = &LLmono[LLoff[t]];
                    for (int u = 0; u < LLcnt[t]; u++) acc += l[src[u]] * sf[mo[u]];
                    lc[t] += acc;
                }
            }
        } else {
            for (int j = nd.start; j < nd.start + nd.cnt; j++) {
                monomials(X[j] - nd.cx, Y[j] - nd.cy, Z[j] - nd.cz, sf.data());
                double gx = 0, gy = 0, gz = 0;
                for (int t = 0; t < NCm1; t++) {
                    gx += l[GPx[t]] * sf[t];
                    gy += l[GPy[t]] * sf[t];
                    gz += l[GPz[t]] * sf[t];
                }
                AX[j] -= gx; AY[j] -= gy; AZ[j] -= gz;
            }
        }
    }
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 1; }
    double limit = 1.0;
    if (const char* s = getenv("GALAXY_CPU_LIMIT_S")) limit = atof(s);
    if (!(limit > 0)) limit = 1.0;

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    N = (int)(bytes / 32);
    std::vector<double> raw((size_t)N * 4);
    if (N > 0 && fread(raw.data(), 32, N, f) != (size_t)N) { perror("read"); return 1; }
    fclose(f);

    // parameters
    int p = 8;
    double theta = 0.75, p2pt = 256, budget_frac = 0.55;
    if (const char* s = getenv("FMM_P")) p = atoi(s);
    if (const char* s = getenv("FMM_THETA")) theta = atof(s);
    if (N > 600000) NLEAF = 64;
    if (const char* s = getenv("FMM_LEAF")) NLEAF = atoi(s);
    if (const char* s = getenv("FMM_P2PT")) p2pt = atof(s);
    if (const char* s = getenv("FMM_BUDGET")) budget_frac = atof(s);

    // bounding box and Morton keys
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < N; i++)
        for (int d = 0; d < 3; d++) {
            double v = raw[(size_t)i * 4 + 1 + d];
            lo[d] = std::min(lo[d], v); hi[d] = std::max(hi[d], v);
        }
    double size = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    if (!(size > 0)) size = 1.0;
    size *= 1.0000001;
    double scale = 2097152.0 / size;
    std::vector<std::pair<uint64_t, int>> kv(N);
    auto spread = [](uint64_t v) {
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    };
    for (int i = 0; i < N; i++) {
        uint64_t q[3];
        for (int d = 0; d < 3; d++) {
            double v = (raw[(size_t)i * 4 + 1 + d] - lo[d]) * scale;
            long long iv = (long long)v;
            if (iv < 0) iv = 0;
            if (iv > 2097151) iv = 2097151;
            q[d] = (uint64_t)iv;
        }
        kv[i].first = (spread(q[0]) << 2) | (spread(q[1]) << 1) | spread(q[2]);
        kv[i].second = i;
    }
    std::sort(kv.begin(), kv.end());
    X.resize(N); Y.resize(N); Z.resize(N); M.resize(N);
    keys.resize(N);
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++) {
        int o = kv[i].second;
        perm[i] = o;
        keys[i] = kv[i].first;
        M[i] = raw[(size_t)o * 4];
        X[i] = raw[(size_t)o * 4 + 1];
        Y[i] = raw[(size_t)o * 4 + 2];
        Z[i] = raw[(size_t)o * 4 + 3];
    }
    std::vector<std::pair<uint64_t, int>>().swap(kv);
    std::vector<double>().swap(raw);

    nodes.reserve(2 * (size_t)N / NLEAF + 1024);
    build(0, N, 0);
    // compact: remove holes (cnt == -1) while keeping indices of children valid
    {
        int nn = (int)nodes.size();
        std::vector<int> remap(nn, -1);
        int k = 0;
        for (int i = 0; i < nn; i++) if (nodes[i].cnt >= 0) remap[i] = k++;
        std::vector<Node> nn2(k);
        for (int i = 0; i < nn; i++)
            if (remap[i] >= 0) {
                Node nd = nodes[i];
                if (nd.nchild > 0) nd.child = remap[nd.child];
                nn2[remap[i]] = nd;
            }
        nodes.swap(nn2);
    }
    nnodes = (int)nodes.size();
    // centers of mass and radii (direct over particle ranges)
    MASSN.assign(nnodes, 0.0);
    for (int i = 0; i < nnodes; i++) {
        Node& nd = nodes[i];
        double ms = 0, sx = 0, sy = 0, sz = 0;
        for (int j = nd.start; j < nd.start + nd.cnt; j++) {
            ms += M[j]; sx += M[j] * X[j]; sy += M[j] * Y[j]; sz += M[j] * Z[j];
        }
        MASSN[i] = ms;
        if (ms > 0) { nd.cx = sx / ms; nd.cy = sy / ms; nd.cz = sz / ms; }
        else { nd.cx = X[nd.start]; nd.cy = Y[nd.start]; nd.cz = Z[nd.start]; }
        double r2 = 0;
        for (int j = nd.start; j < nd.start + nd.cnt; j++) {
            double dx = X[j] - nd.cx, dy = Y[j] - nd.cy, dz = Z[j] - nd.cz;
            r2 = std::max(r2, dx * dx + dy * dy + dz * dz);
        }
        nd.r = std::sqrt(r2);
    }
    // choose opening angle from the CPU budget using counting passes
    init_tables(p);
    P2P_THRESH = p2pt;
    REL_MAC = false;
    double c_m2l = cm2l_of(p), c_p2p = 3.7e-9;
    double c_rest = (double)N * (0.6e-6 + 2.5e-9 * NC) + (double)nnodes * 2.5e-9 * (double)MMsrc.size();
    const double cands[] = {0.8, 0.72, 0.65, 0.6, 0.55, 0.5, 0.46, 0.42, 0.39, 0.36, 0.33, 0.3, 0.27, 0.25};
    double chosen = cands[0], est = 0;
    bool fixed = getenv("FMM_THETA") != nullptr;
    for (double th : cands) {
        if (fixed) { chosen = theta; break; }
        THETA2 = th * th;
        COUNT_MODE = true; cnt_m2l = 0; cnt_p2p = 0;
        self_interact(0);
        COUNT_MODE = false;
        double e = c_m2l * cnt_m2l + c_p2p * cnt_p2p + c_rest;
        double remaining = budget_frac * limit - cpu_now() - 0.03 - 0.1e-6 * N;
        if (e > remaining && th != cands[0]) break;
        chosen = th; est = e;
        if (e > 0.8 * remaining) break;  // next candidate would most likely not fit
    }
    THETA2 = chosen * chosen;
    if (getenv("FMM_VERBOSE"))
        fprintf(stderr, "N=%d nodes=%d theta=%.3f est=%.3f now=%.3f\n", N, nnodes, chosen, est, cpu_now());
    upward();
    COUNT_MODE = true; cnt_m2l = 0; cnt_p2p = 0;
    if (getenv("FMM_VERBOSE")) self_interact(0);
    COUNT_MODE = false;
    double t1 = cpu_now();
    evaluate();
    double t3 = cpu_now();
    if (getenv("FMM_VERBOSE"))
        fprintf(stderr, "m2l=%.0f p2p=%.3g eval=%.3f total=%.3f\n", cnt_m2l, cnt_p2p, t3 - t1, t3);
    std::vector<double> out((size_t)N * 3);
    for (int i = 0; i < N; i++) {
        int o = perm[i];
        double a[3] = {AX[i], AY[i], AZ[i]};
        for (int d = 0; d < 3; d++) {
            if (!std::isfinite(a[d])) a[d] = 0.0;
            out[(size_t)o * 3 + d] = a[d];
        }
    }
    FILE* g = fopen(argv[2], "wb");
    if (!g) { perror("open output"); return 1; }
    if (N > 0) fwrite(out.data(), 24, N, g);
    fclose(g);
    return 0;
}
