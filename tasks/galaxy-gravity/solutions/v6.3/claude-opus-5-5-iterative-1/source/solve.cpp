#include "fmm4_head.inc"
#include <sys/mman.h>

static double DELTA = 4.5e-8, THMAX = 0.7, PPCUT = 4000;
static double THEST = 1.0;
static int M2PMODE = 1;
static std::vector<double> amin;   // per cell
static std::vector<double> mass;   // per cell
static std::vector<double> Bk, binom;   // radial moments sum m|v|^k per cell; binomials
static int MACMODE = 1;
// sum_j m_j (|v_j| + r)^p for cell ci
static int QMIN = 6;
static double QFAC = 0.1;
static int M2PADAPT = 1;
static bool SCALARSHIFT = false;
static double CMQ[32];
static Tables TT[32];
static double CQ[32];
static inline double radpow_q(int ci, double r, int q) {
    const int pb = T.p + 1;
    const double* b = &Bk[(size_t)ci * pb];
    const double* bn = &binom[q * pb];
    double s = 0, rp = 1;
    for (int k = q; k >= 0; k--) { s += bn[k] * b[k] * rp; rp *= r; }
    return s;
}
static inline double radpow(int ci, double r, int p) {
    const double* b = &Bk[(size_t)ci * (p + 1)];
    const double* bn = &binom[p * (p + 1)];
    double s = 0, rp = 1;
    for (int k = p; k >= 0; k--) { s += bn[k] * b[k] * rp; rp *= r; }
    return s;
}
static double C_M2L = 2.0e-6, C_PP = 2.65e-9, C_SELF = 1.0e-9, C_MP = 3.8e-7, C_DOWN = 0.2;
static const double REFPP = 1.47e-9, REFM2L11 = 2.3e-6, REFM2L12 = 3.2e-6;
static inline double ipow(double x, int n) { double r = 1; while (n) { if (n & 1) r *= x; x *= x; n >>= 1; } return r; }

// rough acceleration estimate: order-2 FMM, monopole for near leaf pairs, exact leaf self
static void estimate(std::vector<double>& aest) {
    Tables Tl; Tl.init(2);
    const int ncl = Tl.nc, ncm = T.nc;
    int ncell = cells.size();
    std::vector<double> Le((size_t)ncell * ncl, 0.0);
    std::vector<double> D(ncl), R(4 * ncl), tmp(ncl);
    std::vector<std::pair<int, int>> stack;
    std::vector<int> selfstack; selfstack.push_back(0);
    const double th2 = THEST * THEST;
    auto m2l_lo = [&](int a, int b, bool mono) {
        const Cell& A = cells[a]; const Cell& B = cells[b];
        derivs_s(Tl, B.cx - A.cx, B.cy - A.cy, B.cz - A.cz, D.data(), R.data());
        const double* MA = Mexp + (size_t)a * MSTRIDE; const double* MB = Mexp + (size_t)b * MSTRIDE;
        double* LA = &Le[(size_t)a * ncl]; double* LB = &Le[(size_t)b * ncl];
        int nlim = mono ? 4 : ncl;
        for (int n = 0; n < nlim; n++) {
            int cnt = mono ? 1 : Tl.ncdeg[Tl.p - Tl.deg[n]];
            const int16_t* si = &Tl.sumidx[Tl.sumoff[n]];
            double sb = 0, sa = 0;
            for (int m = 0; m < cnt; m++) { double d = D[si[m]]; sb += MA[m] * d; sa += ((Tl.deg[m] & 1) ? -MB[m] : MB[m]) * d; }
            LB[n] += sb; LA[n] += (Tl.deg[n] & 1) ? -sa : sa;
        }
    };
    while (!selfstack.empty()) {
        int a = selfstack.back(); selfstack.pop_back();
        const Cell& A = cells[a];
        if (A.nchild == 0) { p2p_self(A.begin, A.end); continue; }
        for (int i = 0; i < A.nchild; i++) {
            selfstack.push_back(A.child0 + i);
            for (int j = i + 1; j < A.nchild; j++) stack.push_back({A.child0 + i, A.child0 + j});
        }
        while (!stack.empty()) {
            auto pr = stack.back(); stack.pop_back();
            const Cell& P = cells[pr.first]; const Cell& Q = cells[pr.second];
            double dx = P.cx - Q.cx, dy = P.cy - Q.cy, dz = P.cz - Q.cz;
            double R2 = dx * dx + dy * dy + dz * dz + EPS2;
            double rs = P.r + Q.r;
            if (rs * rs < th2 * R2) { m2l_lo(pr.first, pr.second, false); continue; }
            if (P.nchild == 0 && Q.nchild == 0) { m2l_lo(pr.first, pr.second, true); continue; }
            bool splitP = Q.nchild == 0 || (P.nchild != 0 && P.r >= Q.r);
            if (splitP) for (int i = 0; i < P.nchild; i++) stack.push_back({P.child0 + i, pr.second});
            else for (int i = 0; i < Q.nchild; i++) stack.push_back({pr.first, Q.child0 + i});
        }
    }
    aest.assign(N, 0.0);
    for (int ci = 0; ci < ncell; ci++) {
        const Cell& c = cells[ci];
        double* Lc = &Le[(size_t)ci * ncl];
        if (c.nchild) {
            for (int ch = c.child0; ch < c.child0 + c.nchild; ch++) {
                const Cell& cc = cells[ch];
                double* Lch = &Le[(size_t)ch * ncl];
                monomials_t(Tl, cc.cx - c.cx, cc.cy - c.cy, cc.cz - c.cz, tmp.data());
                for (size_t q = 0; q < Tl.sh_hi.size(); q++) Lch[Tl.sh_lo[q]] += Lc[Tl.sh_hi[q]] * tmp[Tl.sh_df[q]];
            }
        } else {
            for (int i = c.begin; i < c.end; i++) {
                monomials_t(Tl, X[i] - c.cx, Y[i] - c.cy, Z[i] - c.cz, tmp.data());
                double ax = AX[i], ay = AY[i], az = AZ[i];
                for (int k = 0; k < Tl.ncdeg[1]; k++) { ax += Lc[Tl.l2p_x[k]] * tmp[k]; ay += Lc[Tl.l2p_y[k]] * tmp[k]; az += Lc[Tl.l2p_z[k]] * tmp[k]; }
                aest[i] = std::sqrt(ax * ax + ay * ay + az * az);
                AX[i] = AY[i] = AZ[i] = 0;
            }
        }
    }
}


struct Lists {
    std::vector<std::pair<int, int>> m2l, p2p, m2p;
    std::vector<std::pair<int, int>> m2lq[32];
    std::vector<std::pair<int, int>> m2pq[32]; double m2ppq[32];
    std::vector<int> p2pself;
    double pairs = 0, selfint = 0, m2ppart = 0;
    void clear() { m2l.clear(); p2p.clear(); m2p.clear(); p2pself.clear(); pairs = selfint = m2ppart = 0; for (int q = 0; q < 32; q++) { m2lq[q].clear(); m2pq[q].clear(); m2ppq[q] = 0; } }
    double m2pcost(const double* cq) const { double c = 0; for (int q = 0; q < 32; q++) c += cq[q] * m2ppq[q]; return c; }
    double m2lcost(const double* cq) const { double c = 0; for (int q = 0; q < 32; q++) c += cq[q] * m2lq[q].size(); return c; }
};
static void traverse(double delta, Lists& L) {
    L.clear();
    std::vector<std::pair<int, int>> stack;
    std::vector<int> selfstack; selfstack.push_back(0);
    const int p = T.p;
    while (!selfstack.empty()) {
        int a = selfstack.back(); selfstack.pop_back();
        const Cell& A = cells[a];
        double na = A.end - A.begin;
        if (A.nchild == 0 || na * na * 0.5 <= PPCUT) { L.p2pself.push_back(a); L.selfint += na * na; continue; }
        for (int i = 0; i < A.nchild; i++) {
            selfstack.push_back(A.child0 + i);
            for (int j = i + 1; j < A.nchild; j++) stack.push_back({A.child0 + i, A.child0 + j});
        }
        while (!stack.empty()) {
            auto pr = stack.back(); stack.pop_back();
            const Cell& P = cells[pr.first]; const Cell& Q = cells[pr.second];
            double dx = P.cx - Q.cx, dy = P.cy - Q.cy, dz = P.cz - Q.cz;
            double d2 = dx * dx + dy * dy + dz * dz;
            double R2 = d2 + EPS2;
            double th2 = (P.r + Q.r) * (P.r + Q.r) / R2;
            double nPQ = (double)(P.end - P.begin) * (Q.end - Q.begin);
            if (th2 < THMAX * THMAX) {
                double th = std::sqrt(th2);
                bool ok;
                if (MACMODE == 1) {
                    double c = 1.0 / (ipow(std::sqrt(R2), p) * R2 * (1 - th));
                    ok = radpow(pr.first, Q.r, p) * c <= delta * amin[pr.second] && radpow(pr.second, P.r, p) * c <= delta * amin[pr.first];
                } else {
                    double c = ipow(th, p) / (R2 * (1 - th));
                    ok = mass[pr.first] * c <= delta * amin[pr.second] && mass[pr.second] * c <= delta * amin[pr.first];
                }
                if (ok) {
                    if (nPQ <= PPCUT) { L.p2p.push_back(pr); L.pairs += nPQ; }
                    else if (MACMODE == 1 && QMIN < p) {
                        double Reff = std::sqrt(R2);
                        double tb = QFAC * delta * amin[pr.second], ta = QFAC * delta * amin[pr.first];
                        int q = QMIN;
                        double Rq = ipow(Reff, q) * R2 * (1 - th);
                        for (; q < p; q++, Rq *= Reff)
                            if (radpow_q(pr.first, Q.r, q) <= tb * Rq && radpow_q(pr.second, P.r, q) <= ta * Rq) break;
                        L.m2lq[q].push_back(pr);
                    }
                    else L.m2lq[p].push_back(pr);
                    continue;
                }
            }
            if (nPQ <= PPCUT) { L.p2p.push_back(pr); L.pairs += nPQ; continue; }
            if ((P.nchild == 0) != (Q.nchild == 0) && M2PMODE) {
                int cp = P.nchild ? pr.first : pr.second;
                int lq = P.nchild ? pr.second : pr.first;
                const Cell& C = cells[cp]; const Cell& Lf = cells[lq];
                double dmin = std::sqrt(d2) - Lf.r;
                if (dmin > 0 && (M2PMODE == 2 || Lf.r >= C.r)) {
                    double Rm2 = dmin * dmin + EPS2;
                    double thp = C.r / std::sqrt(Rm2);
                    if (thp < THMAX) {
                        double c = ipow(thp, p) / (Rm2 * (1 - thp));
                        if (mass[cp] * c <= delta * amin[lq] && mass[lq] * c <= delta * amin[cp]) {
                            int q = p;
                            if (M2PADAPT && QMIN < p) {
                                double tl = QFAC * delta * amin[lq] / mass[cp], tc = QFAC * delta * amin[cp] / mass[lq];
                                double tlim = std::min(tl, tc) * Rm2 * (1 - thp);
                                double tq = ipow(thp, QMIN);
                                for (q = QMIN; q < p; q++, tq *= thp) if (tq <= tlim) break;
                            }
                            L.m2pq[q].push_back({cp, lq}); L.m2ppq[q] += Lf.end - Lf.begin; L.m2ppart += Lf.end - Lf.begin;
                            continue;
                        }
                    }
                }
            }
            if (P.nchild == 0 && Q.nchild == 0) { L.p2p.push_back(pr); L.pairs += nPQ; continue; }
            bool splitP = Q.nchild == 0 || (P.nchild != 0 && P.r >= Q.r);
            if (splitP) for (int i = 0; i < P.nchild; i++) stack.push_back({P.child0 + i, pr.second});
            else for (int i = 0; i < Q.nchild; i++) stack.push_back({pr.first, Q.child0 + i});
        }
    }
}

// SIMD monomials for 4 points: t[k] = v^k/k!
static inline void monomials4(__m256d x, __m256d y, __m256d z, __m256d* t) {
    t[0] = _mm256_set1_pd(1.0);
    const __m256d v[3] = {x, y, z};
    const int* rd = T.rdir.data(); const int* r1 = T.rk1.data(); const double* ri = T.rinv.data();
    for (int k = 1; k < T.nc; k++) t[k] = _mm256_mul_pd(t[r1[k]], _mm256_mul_pd(v[rd[k]], _mm256_set1_pd(ri[k])));
}
static __m256d *Tq, *Macc;
static void p2m_simd(const Cell& c, double* Mc) {
    const int nc = T.nc, nc4 = (nc + 3) & ~3;
    for (int k = 0; k < nc4; k++) Macc[k] = _mm256_setzero_pd();
    for (int i = c.begin; i < c.end; i += 4) {
        int nv = std::min(4, c.end - i);
        alignas(32) double dx[4], dy[4], dz[4], mm[4];
        for (int l = 0; l < 4; l++) { int ii = l < nv ? i + l : i; dx[l] = c.cx - X[ii]; dy[l] = c.cy - Y[ii]; dz[l] = c.cz - Z[ii]; mm[l] = l < nv ? M[ii] : 0.0; }
        monomials4(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Tq);
        __m256d mv = _mm256_load_pd(mm);
        for (int k = 0; k < nc; k++) Macc[k] = _mm256_fmadd_pd(mv, Tq[k], Macc[k]);
    }
    for (int k = 0; k < nc4; k += 4) {
        __m256d r0 = Macc[k], r1 = Macc[k + 1], r2 = Macc[k + 2], r3 = Macc[k + 3];
        transpose4(r0, r1, r2, r3);
        __m256d sum = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));
        _mm256_storeu_pd(Mc + k, _mm256_add_pd(_mm256_loadu_pd(Mc + k), sum));
    }
}
static void l2p_simd(const Cell& c, const double* Lc) {
    const int q = T.ncdeg[T.p - 1];
    const uint64_t* l2p = T.l2p_pk.data();
    for (int i = c.begin; i < c.end; i += 4) {
        int nv = std::min(4, c.end - i);
        alignas(32) double dx[4], dy[4], dz[4];
        for (int l = 0; l < 4; l++) { int ii = l < nv ? i + l : i; dx[l] = X[ii] - c.cx; dy[l] = Y[ii] - c.cy; dz[l] = Z[ii] - c.cz; }
        monomials4(_mm256_load_pd(dx), _mm256_load_pd(dy), _mm256_load_pd(dz), Tq);
        __m256d ax = _mm256_setzero_pd(), ay = ax, az = ax;
        for (int k = 0; k < q; k++) {
            uint64_t w = l2p[k];
            const __m256d t = Tq[k];
            ax = _mm256_fmadd_pd(_mm256_broadcast_sd(Lc + ((w & 0xffff) >> 5)), t, ax);
            ay = _mm256_fmadd_pd(_mm256_broadcast_sd(Lc + (((w >> 16) & 0xffff) >> 5)), t, ay);
            az = _mm256_fmadd_pd(_mm256_broadcast_sd(Lc + (((w >> 32) & 0xffff) >> 5)), t, az);
        }
        alignas(32) double tx[4], ty[4], tz[4];
        _mm256_store_pd(tx, ax); _mm256_store_pd(ty, ay); _mm256_store_pd(tz, az);
        for (int l = 0; l < nv; l++) { AX[i + l] += tx[l]; AY[i + l] += ty[l]; AZ[i + l] += tz[l]; }
    }
}

// SIMD factorized shifts for up to 4 children at once (lanes = children)
static __m256d *SA, *SB, *SC;
static void shift4_core(const std::vector<int>* dsts, const std::vector<int>* srcs, const std::vector<int>* pws,
                        __m256d* in, __m256d* out, const double* sx, const double* sy, const double* sz) {
    // in: SoA [nc] vectors; out: SoA [nc] vectors (overwritten); uses SB as temp
    const int nc = T.nc, p = T.p;
    alignas(32) __m256d pw[3][32];
    for (int d = 0; d < 3; d++) {
        const double* s = d == 0 ? sx : (d == 1 ? sy : sz);
        __m256d sv = _mm256_loadu_pd(s);
        pw[d][0] = _mm256_set1_pd(1.0);
        for (int j = 1; j <= p; j++) pw[d][j] = _mm256_mul_pd(pw[d][j - 1], _mm256_mul_pd(sv, _mm256_set1_pd(1.0 / j)));
    }
    __m256d* src = in;
    __m256d* bufs[3] = {SB, SC, out};
    for (int d = 0; d < 3; d++) {
        __m256d* dst = bufs[d];
        for (int k = 0; k < nc; k++) dst[k] = _mm256_setzero_pd();
        const int* D = dsts[d].data(); const int* S = srcs[d].data(); const int* P = pws[d].data();
        size_t ne = dsts[d].size();
        for (size_t q = 0; q < ne; q++) dst[D[q]] = _mm256_fmadd_pd(src[S[q]], pw[d][P[q]], dst[D[q]]);
        src = dst;
    }
}
static void m2m_simd(const Cell& c, double* Mc) {
    const int nc = T.nc, nc4 = (nc + 3) & ~3;
    for (int g = c.child0; g < c.child0 + c.nchild; g += 4) {
        int nv = std::min(4, c.child0 + c.nchild - g);
        const double* src[4]; alignas(32) double sx[4], sy[4], sz[4];
        for (int l = 0; l < 4; l++) {
            int ch = g + (l < nv ? l : 0);
            const Cell& cc = cells[ch];
            src[l] = Mexp + (size_t)ch * MSTRIDE;
            sx[l] = c.cx - cc.cx; sy[l] = c.cy - cc.cy; sz[l] = c.cz - cc.cz;
        }
        for (int m = 0; m < nc4; m += 4) {
            __m256d a0 = _mm256_loadu_pd(src[0] + m), a1 = _mm256_loadu_pd(src[1] + m), a2 = _mm256_loadu_pd(src[2] + m), a3 = _mm256_loadu_pd(src[3] + m);
            transpose4(a0, a1, a2, a3);
            SA[m] = a0; SA[m + 1] = a1; SA[m + 2] = a2; SA[m + 3] = a3;
        }
        shift4_core(T.s1_dst, T.s1_src, T.s1_pw, SA, Macc, sx, sy, sz);
        // sum valid lanes into parent
        const __m256d lanemask = _mm256_castsi256_pd(_mm256_setr_epi64x(nv > 0 ? -1 : 0, nv > 1 ? -1 : 0, nv > 2 ? -1 : 0, nv > 3 ? -1 : 0));
        for (int k = nc; k < nc4; k++) Macc[k] = _mm256_setzero_pd();
        for (int k = 0; k < nc4; k += 4) {
            __m256d r0 = _mm256_and_pd(Macc[k], lanemask), r1 = _mm256_and_pd(Macc[k + 1], lanemask), r2 = _mm256_and_pd(Macc[k + 2], lanemask), r3 = _mm256_and_pd(Macc[k + 3], lanemask);
            transpose4(r0, r1, r2, r3);
            __m256d sum = _mm256_add_pd(_mm256_add_pd(r0, r1), _mm256_add_pd(r2, r3));
            _mm256_storeu_pd(Mc + k, _mm256_add_pd(_mm256_loadu_pd(Mc + k), sum));
        }
    }
}
static void l2l_simd(const Cell& c, const double* Lc) {
    const int nc = T.nc, nc4 = (nc + 3) & ~3;
    for (int k = 0; k < nc; k++) SA[k] = _mm256_broadcast_sd(Lc + k);
    for (int g = c.child0; g < c.child0 + c.nchild; g += 4) {
        int nv = std::min(4, c.child0 + c.nchild - g);
        double* dst[4]; alignas(32) double sx[4], sy[4], sz[4];
        for (int l = 0; l < 4; l++) {
            int ch = g + (l < nv ? l : 0);
            const Cell& cc = cells[ch];
            dst[l] = Lexp + (size_t)ch * MSTRIDE;
            sx[l] = cc.cx - c.cx; sy[l] = cc.cy - c.cy; sz[l] = cc.cz - c.cz;
        }
        shift4_core(T.s2_dst, T.s2_src, T.s2_pw, SA, Macc, sx, sy, sz);
        for (int k = nc; k < nc4; k++) Macc[k] = _mm256_setzero_pd();
        for (int k = 0; k < nc4; k += 4) {
            __m256d r0 = Macc[k], r1 = Macc[k + 1], r2 = Macc[k + 2], r3 = Macc[k + 3];
            transpose4(r0, r1, r2, r3);
            __m256d rr[4] = {r0, r1, r2, r3};
            for (int l = 0; l < nv; l++) _mm256_storeu_pd(dst[l] + k, _mm256_add_pd(_mm256_loadu_pd(dst[l] + k), rr[l]));
        }
    }
}

// zeroed allocation backed by transparent huge pages when available (fewer page faults)
static void* big_zalloc(size_t bytes) {
    const size_t HP = (size_t)2 << 20;
    size_t len = ((bytes + HP - 1) / HP + 1) * HP;
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { void* q = calloc(bytes, 1); return q; }
    uintptr_t a = ((uintptr_t)p + HP - 1) & ~(uintptr_t)(HP - 1);
#ifdef MADV_HUGEPAGE
    if (!getenv("GX_NOHUGE")) madvise((void*)a, len - (a - (uintptr_t)p), MADV_HUGEPAGE);
#endif
    return (void*)a;
}

// ---- machine speed calibration: times the real P2P and M2L kernels on synthetic data ----
static void calibrate(double& t_pp, double& t_m2l) {
    // P2P: 64 targets x 512 sources, mutual kernel
    const int na = 64, nb = 512;
    std::vector<double> sx(na), sy(na), sz(na), sm(na), sax(na), say(na), saz(na);
    double* oX = X; double* oY = Y; double* oZ = Z; double* oM = M; double* oAX = AX; double* oAY = AY; double* oAZ = AZ;
    uint64_t st = 12345;
    auto rnd = [&]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return (double)(st >> 11) / 9007199254740992.0 - 0.5; };
    for (int i = 0; i < na; i++) { sx[i] = rnd(); sy[i] = rnd(); sz[i] = rnd(); sm[i] = 1e-5; sax[i] = say[i] = saz[i] = 0; }
    for (int j = 0; j < nb; j++) { gx[j] = rnd(); gy[j] = rnd(); gz[j] = rnd(); gm[j] = 1e-5; gax[j] = gay[j] = gaz[j] = 0; }
    X = sx.data(); Y = sy.data(); Z = sz.data(); M = sm.data(); AX = sax.data(); AY = say.data(); AZ = saz.data();
    std::vector<double> tt;
    for (int r = 0; r < 15; r++) { double a = cpu_now(); p2p_group_kernel(0, na, nb); tt.push_back(cpu_now() - a); }
    std::sort(tt.begin(), tt.end());
    t_pp = tt[0] / ((double)na * nb);
    X = oX; Y = oY; Z = oZ; M = oM; AX = oAX; AY = oAY; AZ = oAZ;
    // M2L: 8 batches over the first cells of the real tree (results discarded by restoring L afterwards)
    int nce = std::min<int>(cells.size(), 64);
    std::vector<std::pair<int, int>> prs;
    for (int k = 0; k < 32; k++) { int a = (k * 7) % nce, b = (k * 13 + 5) % nce; if (a == b) b = (b + 1) % nce; prs.push_back({a, b}); }
    std::vector<double> Lsave(Lexp, Lexp + (size_t)nce * MSTRIDE);
    tt.clear();
    for (size_t q = 0; q < prs.size(); q += 4) m2l_batch(&prs[q], 4);   // warm-up
    for (int r = 0; r < 9; r++) {
        double a = cpu_now();
        for (size_t q = 0; q < prs.size(); q += 4) m2l_batch(&prs[q], 4);
        tt.push_back(cpu_now() - a);
    }
    std::copy(Lsave.begin(), Lsave.end(), Lexp);
    std::sort(tt.begin(), tt.end());
    t_m2l = tt[0] / prs.size();
}

int main(int argc, char** argv) {
    double t0 = cpu_now();
    const bool VERB = getenv("GX_VERBOSE") != nullptr;
    T.init(getenv("GX_P") ? atoi(getenv("GX_P")) : 15);
    if (getenv("GX_QMIN")) QMIN = atoi(getenv("GX_QMIN"));
    if (getenv("GX_QFAC")) QFAC = atof(getenv("GX_QFAC"));
    if (getenv("GX_M2PADAPT")) M2PADAPT = atoi(getenv("GX_M2PADAPT"));
    if (QMIN > T.p) QMIN = T.p;
    for (int q = QMIN; q < T.p; q++) TT[q].init(q);
    g_derivs = getenv("GX_NOGEN") ? nullptr : (getenv("GX_MD") ? derivs4_table(T.p) : derivs4t_table(T.p));
    if (getenv("GX_NLEAF")) NLEAF = atoi(getenv("GX_NLEAF"));
    if (getenv("GX_DELTA")) DELTA = atof(getenv("GX_DELTA"));
    if (getenv("GX_THMAX")) THMAX = atof(getenv("GX_THMAX"));
    if (getenv("GX_PPCUT")) PPCUT = atof(getenv("GX_PPCUT"));
    if (getenv("GX_M2PMODE")) M2PMODE = atoi(getenv("GX_M2PMODE"));
    if (getenv("GX_MACMODE")) MACMODE = atoi(getenv("GX_MACMODE"));
    if (getenv("GX_THEST")) THEST = atof(getenv("GX_THEST"));
    double FLOOR = getenv("GX_FLOOR") ? atof(getenv("GX_FLOOR")) : 0.2;
    FILE* f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    N = sz / 32;
    std::vector<double> buf(4 * (size_t)N);
    if (fread(buf.data(), 8, 4 * (size_t)N, f) != 4 * (size_t)N) return 1;
    fclose(f);
    NP = (N + 7) & ~7;
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < N; i++) for (int d = 0; d < 3; d++) { double v = buf[4 * i + 1 + d]; lo[d] = std::min(lo[d], v); hi[d] = std::max(hi[d], v); }
    double size = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2])) * 1.0001 + 1e-300;
    std::vector<std::pair<uint64_t, int>> kp(N);
    for (int i = 0; i < N; i++) {
        uint64_t key = 0;
        uint32_t q[3];
        for (int d = 0; d < 3; d++) {
            double t = (buf[4 * i + 1 + d] - lo[d]) / size * 2097152.0;
            if (!(t >= 0)) t = 0; if (t > 2097151) t = 2097151;
            q[d] = (uint32_t)t;
        }
        for (int bit = 20; bit >= 0; bit--) key = (key << 3) | (((q[0] >> bit) & 1) << 2) | (((q[1] >> bit) & 1) << 1) | ((q[2] >> bit) & 1);
        kp[i] = {key, i};
    }
    std::sort(kp.begin(), kp.end());
    std::vector<uint64_t> keys(N);
    std::vector<int> perm(N);
    for (int i = 0; i < N; i++) { keys[i] = kp[i].first; perm[i] = kp[i].second; }
    X = (double*)aligned_alloc(64, NP * 8); Y = (double*)aligned_alloc(64, NP * 8); Z = (double*)aligned_alloc(64, NP * 8); M = (double*)aligned_alloc(64, NP * 8);
    AX = (double*)aligned_alloc(64, NP * 8); AY = (double*)aligned_alloc(64, NP * 8); AZ = (double*)aligned_alloc(64, NP * 8);
    for (int i = 0; i < NP; i++) {
        if (i < N) { int j = perm[i]; M[i] = buf[4 * j]; X[i] = buf[4 * j + 1]; Y[i] = buf[4 * j + 2]; Z[i] = buf[4 * j + 3]; }
        else { M[i] = 0; X[i] = Y[i] = Z[i] = 0; }
        AX[i] = AY[i] = AZ[i] = 0;
    }
    if (N <= 3000 || getenv("GX_FORCE_DIRECT")) {
        std::vector<double> out(3 * (size_t)N);
        for (int i = 0; i < N; i++) {
            long double ax = 0, ay = 0, az = 0;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                double dx = buf[4 * j + 1] - buf[4 * i + 1], dy = buf[4 * j + 2] - buf[4 * i + 2], dz = buf[4 * j + 3] - buf[4 * i + 3];
                double r2 = dx * dx + dy * dy + dz * dz + EPS2;
                double fm = buf[4 * j] / (r2 * std::sqrt(r2));
                ax += dx * fm; ay += dy * fm; az += dz * fm;
            }
            out[3 * i] = (double)ax; out[3 * i + 1] = (double)ay; out[3 * i + 2] = (double)az;
        }
        f = fopen(argv[2], "wb");
        fwrite(out.data(), 8, 3 * (size_t)N, f);
        fclose(f);
        return 0;
    }
    build_tree(keys);
    int ncell = cells.size();
    mass.assign(ncell, 0.0);
    Bk.assign((size_t)ncell * (T.p + 1), 0.0);
    binom.assign((T.p + 1) * (T.p + 1), 0.0);
    for (int n = 0; n <= T.p; n++) { binom[n * (T.p + 1)] = 1; for (int k = 1; k <= n; k++) binom[n * (T.p + 1) + k] = binom[(n - 1) * (T.p + 1) + k - 1] + (k <= n - 1 ? binom[(n - 1) * (T.p + 1) + k] : 0); }
    for (int ci = ncell - 1; ci >= 0; ci--) {
        Cell& c = cells[ci];
        double mx = 0, my = 0, mz = 0, mt = 0;
        for (int i = c.begin; i < c.end; i++) { mx += M[i] * X[i]; my += M[i] * Y[i]; mz += M[i] * Z[i]; mt += M[i]; }
        if (mt > 0) { c.cx = mx / mt; c.cy = my / mt; c.cz = mz / mt; }
        else { c.cx = X[c.begin]; c.cy = Y[c.begin]; c.cz = Z[c.begin]; }
        double r2 = 0;
        for (int i = c.begin; i < c.end; i++) { double dx = X[i] - c.cx, dy = Y[i] - c.cy, dz = Z[i] - c.cz; r2 = std::max(r2, dx * dx + dy * dy + dz * dz); }
        c.r = std::sqrt(r2);
        mass[ci] = mt;
        {
            const int pb = T.p + 1;
            double* b = &Bk[(size_t)ci * pb];
            for (int k = 0; k < pb; k++) b[k] = 0;
            for (int i = c.begin; i < c.end; i++) {
                double dx = X[i] - c.cx, dy = Y[i] - c.cy, dz = Z[i] - c.cz;
                double rr = std::sqrt(dx * dx + dy * dy + dz * dz), t = M[i];
                for (int k = 0; k < pb; k++) { b[k] += t; t *= rr; }
            }
        }
    }
    double t1 = cpu_now();
    const int nc = T.nc;
    MSTRIDE = (nc + 3) & ~3;
    Mexp = (double*)big_zalloc(((size_t)ncell * MSTRIDE + 8) * 8);
    Lexp = (double*)big_zalloc(((size_t)ncell * MSTRIDE + 8) * 8);
    Dv = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); Rv = (__m256d*)aligned_alloc(32, (T.rtot + 4) * 32);
    MAv = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); ABv = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32);
    LAv = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); LBv = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32);
    Lacc = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32);
    std::vector<double> tmp(nc), tmp1(nc), tmp2(nc);
    Tq = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); Macc = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32);
    SA = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); SB = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32); SC = (__m256d*)aligned_alloc(32, (MSTRIDE + 4) * 32);
    SCALARSHIFT = getenv("GX_SCALARSHIFT") != nullptr;
    for (int ci = ncell - 1; ci >= 0; ci--) {
        Cell& c = cells[ci];
        double* Mc = Mexp + (size_t)ci * MSTRIDE;
        if (c.nchild == 0) {
            p2m_simd(c, Mc);
        } else {
            if (!SCALARSHIFT) m2m_simd(c, Mc);
            else for (int ch = c.child0; ch < c.child0 + c.nchild; ch++) {
                Cell& cc = cells[ch];
                double* Mch = Mexp + (size_t)ch * MSTRIDE;
                shift_m2m(Mch, Mc, c.cx - cc.cx, c.cy - cc.cy, c.cz - cc.cz, tmp1.data(), tmp2.data());
            }
        }
    }
    double t2 = cpu_now();
    std::vector<double> aest;
    estimate(aest);
    amin.assign(ncell, 0.0);
    {
        // sanitize estimates and apply a global floor (guards against zero tolerances)
        std::vector<double> tmpa;
        tmpa.reserve(N);
        for (int i = 0; i < N; i++) { if (!std::isfinite(aest[i])) aest[i] = 0.0; tmpa.push_back(aest[i]); }
        std::nth_element(tmpa.begin(), tmpa.begin() + N / 2, tmpa.end());
        double med = tmpa[N / 2];
        double gfloor = 1e-7 * med;
        if (!(gfloor > 0)) gfloor = 1e-300;
        for (int i = 0; i < N; i++) aest[i] = std::max(aest[i], gfloor);
    }
    for (int ci = ncell - 1; ci >= 0; ci--) {
        Cell& c = cells[ci];
        if (c.nchild == 0) {
            double mn = 1e300, s2 = 0;
            for (int i = c.begin; i < c.end; i++) { mn = std::min(mn, aest[i]); s2 += aest[i] * aest[i]; }
            double rms = std::sqrt(s2 / (c.end - c.begin));
            amin[ci] = std::max(mn, FLOOR * rms);
        } else {
            double mn = 1e300;
            for (int ch = c.child0; ch < c.child0 + c.nchild; ch++) mn = std::min(mn, amin[ch]);
            amin[ci] = mn;
        }
    }
    double t2b = cpu_now();
    {
        // per-order unit costs (seconds), measured on the dev machine
        const int p = T.p;
        double cm2l = p <= 8 ? 0.90e-6 : p <= 10 ? 1.85e-6 : p <= 11 ? 2.45e-6 : p <= 12 ? 3.45e-6 : p <= 13 ? 4.65e-6 : 6.2e-6;
        double cmp = p <= 8 ? 0.16e-6 : p <= 10 ? 0.28e-6 : p <= 11 ? 0.36e-6 : p <= 12 ? 0.46e-6 : p <= 13 ? 0.61e-6 : p <= 14 ? 0.72e-6 : 0.90e-6;
        double cdown = p <= 8 ? 0.06 : p <= 10 ? 0.09 : p <= 11 ? 0.11 : p <= 12 ? 0.15 : p <= 13 ? 0.15 : p <= 14 ? 0.17 : 0.22;
        C_M2L = cm2l; C_MP = cmp; C_DOWN = cdown; C_PP = 1.80e-9; C_SELF = 0.90e-9;
        if (getenv("GX_CSCALE")) { double f = atof(getenv("GX_CSCALE")); C_M2L *= f; C_MP *= f; C_DOWN *= f; C_PP *= f; C_SELF *= f; }
    }
    {
        size_t gsz = (size_t)NP + 1024;
        gx = (double*)aligned_alloc(64, gsz * 8); gy = (double*)aligned_alloc(64, gsz * 8); gz = (double*)aligned_alloc(64, gsz * 8); gm = (double*)aligned_alloc(64, gsz * 8);
        gax = (double*)aligned_alloc(64, gsz * 8); gay = (double*)aligned_alloc(64, gsz * 8); gaz = (double*)aligned_alloc(64, gsz * 8);
    }
    double speed_pp = 1.0, speed_m2l = 1.0;
    if (getenv("GX_CAL")) {
        double tpp, tm2l;
        calibrate(tpp, tm2l);
        const double REF_PP = REFPP, REF_M2L = (T.p <= 11 ? REFM2L11 : REFM2L12);
        double f = std::max(tpp / REF_PP, tm2l / REF_M2L);
        f = std::min(1.5, std::max(0.80, f));
        speed_pp = speed_m2l = f;
        if (VERB) fprintf(stderr, "calib: pp %.3g ns (x%.3f)  m2l %.3g us (x%.3f)\n", tpp * 1e9, speed_pp, tm2l * 1e6, speed_m2l);
        C_PP *= speed_pp; C_SELF *= speed_pp; C_M2L *= speed_m2l; C_MP *= speed_m2l; C_DOWN *= 0.5 * (speed_pp + speed_m2l);
    }
    for (int q = 0; q < 32; q++) {
        // measured relative cost of an order-q M2L batch vs order 12 (dev machine)
        // absolute per-order M2L cost (microseconds) measured on the dev machine
        static const double us[17] = {0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.72, 0.79, 1.10, 1.48, 2.0, 2.62, 3.45, 4.65, 6.2, 8.3, 10.8};
        CQ[q] = 1.05e-6 * (q <= 16 ? us[q] : 10.3 * std::pow(1.3, q - 16));
    }
    if (T.p != 12) for (int q = 0; q < 32; q++) CQ[q] *= 1.0;
    for (int q = 0; q < 32; q++) {
        double ncq = (q + 1.0) * (q + 2) * (q + 3) / 6.0, ncp = (T.p + 1.0) * (T.p + 2) * (T.p + 3) / 6.0;
        CMQ[q] = C_MP * std::max(0.3, std::pow(ncq / ncp, 0.85));
    }
    // main traversal with automatic tolerance selection
    Lists LL;
    double delta = DELTA;
    double budget = getenv("GX_BUDGET") ? atof(getenv("GX_BUDGET")) : 4.15;
    int ntrav = 0;
    double best_ok = -1;
    {
        Lists LB;   // best feasible list so far
        bool have = false;
        double tcost = 0.07;  // approx cost of one traversal
        double pd = 0, pv = 0;  // previous delta / var for local exponent estimate
        for (;;) {
            traverse(delta, LL);
            ntrav++;
            double now = cpu_now() - t0;
            double var = LL.m2lcost(CQ) + C_PP * LL.pairs + C_SELF * LL.selfint + LL.m2pcost(CMQ);
            double pred = now + var + C_DOWN + 0.02;
            if (VERB) fprintf(stderr, "trav %d delta=%.3g pred=%.3f (now %.3f var %.3f)\n", ntrav, delta, pred, now, var);
            if (getenv("GX_FIXDELTA")) break;
            double expo = 5.0;
            if (pd > 0 && std::fabs(std::log(pv / var)) > 0.02) expo = std::min(12.0, std::max(3.0, std::log(delta / pd) / std::log(pv / var)));
            double target = budget - now - C_DOWN - 0.02 - tcost;
            if (pred <= budget) {
                if (ntrav <= 3 && pred < budget - 0.2 && target > 0.1) {
                    double nd = delta * std::pow(var / target, expo) * 1.1;
                    if (nd < delta * 0.9) { std::swap(LB, LL); have = true; pd = delta; pv = var; delta = nd; continue; }
                }
                break;
            }
            if (have) { std::swap(LB, LL); break; }   // refinement overshot: fall back to feasible lists
            double ratio = var / std::max(target, 0.05);
            pd = delta; pv = var;
            delta *= std::min(1e6, std::max(1.3, std::pow(ratio, expo) * 1.2));
            if (ntrav >= 4) delta *= 1e3;
        }
    }
    std::vector<std::pair<int, int>>& m2l = LL.m2l; std::vector<std::pair<int, int>>& p2p = LL.p2p; std::vector<std::pair<int, int>>& m2p = LL.m2p;
    std::vector<int>& p2pself = LL.p2pself;
    long long npairs = (long long)LL.pairs;
    double t3 = cpu_now();
    const double hardcap = getenv("GX_HARDCAP") ? atof(getenv("GX_HARDCAP")) : 4.8;
    size_t n_m2l_total = 0;
    for (int qo = 0; qo <= T.p && !g_abort; qo++) {
        std::vector<std::pair<int, int>>& lst = LL.m2lq[qo];
        if (lst.empty()) continue;
        n_m2l_total += lst.size();
        const Tables& TQ = (qo == T.p) ? T : TT[qo];
        derivs4_fn dfn = (qo == T.p) ? g_derivs : derivs4t_table(qo);
        for (size_t q = 0; q < lst.size(); q += 4) {
            if ((q & 1023) == 0 && cpu_now() - t0 > hardcap) { g_abort = true; break; }
            m2l_batch(&lst[q], std::min<size_t>(4, lst.size() - q), TQ, dfn);
        }
    }
    double t4 = cpu_now();
    for (int qo = 0; qo <= T.p && !g_abort; qo++) {
        std::vector<std::pair<int, int>>& lst = LL.m2pq[qo];
        if (lst.empty()) continue;
        const Tables& TQ = (qo == T.p) ? T : TT[qo];
        derivs4_fn dfn = (qo == T.p) ? g_derivs : derivs4t_table(qo);
        for (size_t q = 0; q < lst.size(); q++) {
            if ((q & 255) == 0 && cpu_now() - t0 > hardcap) { g_abort = true; break; }
            m2p_p2l(lst[q].first, lst[q].second, TQ, dfn);
        }
    }
    double t4b = cpu_now();
    if (!g_abort) p2p_grouped(p2p, t0, hardcap);
    for (int a : p2pself) { const Cell& A = cells[a]; p2p_self(A.begin, A.end); }   // cheap: always done
    double t5 = cpu_now();
    for (int ci = 0; ci < ncell; ci++) {
        Cell& c = cells[ci];
        double* Lc = Lexp + (size_t)ci * MSTRIDE;
        if (c.nchild) {
            if (!SCALARSHIFT) l2l_simd(c, Lc);
            else for (int ch = c.child0; ch < c.child0 + c.nchild; ch++) {
                Cell& cc = cells[ch];
                double* Lch = Lexp + (size_t)ch * MSTRIDE;
                shift_l2l(Lc, Lch, cc.cx - c.cx, cc.cy - c.cy, cc.cz - c.cz, tmp1.data(), tmp2.data());
            }
        } else {
            l2p_simd(c, Lc);
        }
    }
    double t6 = cpu_now();
    std::vector<double> out(3 * (size_t)N);
    for (int i = 0; i < N; i++) {
        if (!(std::isfinite(AX[i]) && std::isfinite(AY[i]) && std::isfinite(AZ[i]))) {
            double ax = 0, ay = 0, az = 0;
            for (int j = 0; j < N; j++) {
                double dx = X[j] - X[i], dy = Y[j] - Y[i], dz = Z[j] - Z[i];
                double r2 = dx * dx + dy * dy + dz * dz + EPS2;
                double fm = M[j] / (r2 * std::sqrt(r2));
                ax += dx * fm; ay += dy * fm; az += dz * fm;
            }
            AX[i] = std::isfinite(ax) ? ax : 0.0; AY[i] = std::isfinite(ay) ? ay : 0.0; AZ[i] = std::isfinite(az) ? az : 0.0;
        }
        int j = perm[i]; out[3 * j] = AX[i]; out[3 * j + 1] = AY[i]; out[3 * j + 2] = AZ[i];
    }
    f = fopen(argv[2], "wb");
    fwrite(out.data(), 8, 3 * (size_t)N, f);
    fclose(f);
    double t7 = cpu_now();
    if (VERB) fprintf(stderr, "p=%d delta=%.1e thmax=%.2f nleaf=%d ppcut=%.0f cells=%d m2l=%zu m2p=%lld p2p=%zu self=%zu pairs=%lld (%.3f of N^2/2)\n", T.p, delta, THMAX, NLEAF, PPCUT, ncell, n_m2l_total, n_m2p, p2p.size(), p2pself.size(), npairs, npairs / (0.5 * N * (double)N));
    if (VERB) { fprintf(stderr, "m2l per order:"); for (int q = 0; q <= T.p; q++) if (LL.m2lq[q].size()) fprintf(stderr, " %d:%zu", q, LL.m2lq[q].size()); fprintf(stderr, "\n"); }
    if (VERB) fprintf(stderr, "units: m2l %.3g us  pp %.3g ns  mp %.3g ns/part  self %lld\n", (t4-t3)/std::max<size_t>(1,n_m2l_total)*1e6, (t5-t4b)/std::max(1.0,LL.pairs+0.5*LL.selfint)*1e9, (t4b-t4)/std::max(1.0,LL.m2ppart)*1e9, (long long)LL.selfint);
    if (VERB) fprintf(stderr, "time: tree %.3f up %.3f est %.3f trav %.3f m2l %.3f m2p %.3f p2p %.3f down %.3f out %.3f total %.3f\n", t1 - t0, t2 - t1, t2b - t2, t3 - t2b, t4 - t3, t4b - t4, t5 - t4b, t6 - t5, t7 - t6, t7 - t0);
    return 0;
}
