// Gravitational acceleration for galaxy snapshots.
// Adaptive dual-tree traversal with Cartesian multipole -> local (M2L)
// expansions of the softened potential, plus SIMD direct near field.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <time.h>
#include <sys/stat.h>

static const double EPS2 = 1e-4; // softening 0.01 squared

static double cputime() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static double walltime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static int dbg = 0;

// ---------------------------------------------------------------------------
// Expansion tables
// ---------------------------------------------------------------------------
static int P, Q, Q1, C, Cp;
static std::vector<int> mA, mB, mC;            // moment multi-indices (degree<=P)
static std::vector<double> COEF;               // Cp*Cp per (beta,alpha)
static std::vector<int> DIDX;                  // Cp*Cp flat index into D for alpha+beta
static std::vector<int> LUT;                   // (P+1)^3 -> compact moment index
static std::vector<double> BIN;                // binomial[n][k], n<=2P+2
static int P1;

struct DG {
    int out;        // flat D index of gamma
    int gi;         // chosen axis
    int ibeta;      // flat D index of beta = gamma - e_gi
    int ibme;       // flat D index of beta - e_gi, or -1
    int i1[3], i2[3];
    double f[3], g[3];
    double invgi;   // 1/gamma_gi
};
static std::vector<DG> dgs;                    // ordered by degree ascending
static std::vector<double> Darr;               // (Q+1)^3 dense derivative coefficients

static inline int idD(int a, int b, int c) { return (a * Q1 + b) * Q1 + c; }
static inline int idM(int a, int b, int c) { return (a * P1 + b) * P1 + c; }
static inline double binom(int n, int k) { return BIN[n * (2 * P + 3) + k]; }

static void initTables(int p) {
    P = p; Q = 2 * P; Q1 = Q + 1; P1 = P + 1;
    int NB = 2 * P + 3;
    BIN.assign(NB * NB, 0.0);
    for (int n = 0; n < NB; ++n) {
        BIN[n * NB + 0] = 1.0;
        for (int k = 1; k <= n; ++k)
            BIN[n * NB + k] = BIN[(n - 1) * NB + k - 1] + BIN[(n - 1) * NB + k];
    }
    mA.clear(); mB.clear(); mC.clear();
    for (int a = 0; a <= P; ++a)
        for (int b = 0; b <= P - a; ++b)
            for (int c = 0; c <= P - a - b; ++c) {
                mA.push_back(a); mB.push_back(b); mC.push_back(c);
            }
    C = (int)mA.size();
    Cp = ((C + 3) / 4) * 4;
    LUT.assign(P1 * P1 * P1, -1);
    for (int k = 0; k < C; ++k) LUT[idM(mA[k], mB[k], mC[k])] = k;

    COEF.assign((size_t)Cp * Cp, 0.0);
    DIDX.assign((size_t)Cp * Cp, 0);
    for (int ib = 0; ib < C; ++ib)
        for (int ia = 0; ia < C; ++ia) {
            int a = mA[ia], b = mB[ia], c = mC[ia];
            int A = mA[ib], B = mB[ib], Cc = mC[ib];
            double v = binom(a + A, a) * binom(b + B, b) * binom(c + Cc, c);
            if ((a + b + c) & 1) v = -v;
            COEF[(size_t)ib * Cp + ia] = v;
            DIDX[(size_t)ib * Cp + ia] = idD(a + A, b + B, c + Cc);
        }

    // derivative recurrence schedule: all gamma, 1<=|gamma|<=Q
    dgs.clear();
    for (int deg = 1; deg <= Q; ++deg) {
        for (int a = 0; a <= deg; ++a)
            for (int b = 0; b <= deg - a; ++b) {
                int c = deg - a - b;
                int gam[3] = {a, b, c};
                int gi = (a > 0) ? 0 : ((b > 0) ? 1 : 2);
                int beta[3] = {a, b, c};
                beta[gi]--;
                DG d;
                d.out = idD(a, b, c);
                d.gi = gi;
                d.ibeta = idD(beta[0], beta[1], beta[2]);
                d.invgi = 1.0 / (double)gam[gi];
                d.ibme = (beta[gi] > 0)
                    ? idD(beta[0] - (gi == 0), beta[1] - (gi == 1), beta[2] - (gi == 2))
                    : -1;
                for (int j = 0; j < 3; ++j) {
                    int b1[3] = {beta[0], beta[1], beta[2]};
                    b1[j]--; b1[gi]++;
                    int b2[3] = {beta[0], beta[1], beta[2]};
                    b2[j] -= 2; b2[gi]++;
                    d.i1[j] = (beta[j] > 0) ? idD(b1[0], b1[1], b1[2]) : -1;
                    d.i2[j] = (beta[j] > 1) ? idD(b2[0], b2[1], b2[2]) : -1;
                    d.f[j] = (double)(beta[gi] + 1 - (gi == j ? 1 : 0));
                    d.g[j] = (double)(beta[gi] + 1 - (gi == j ? 2 : 0));
                }
                dgs.push_back(d);
            }
    }
    Darr.assign((size_t)Q1 * Q1 * Q1, 0.0);
}

// Compute D[gamma] = d^gamma h / gamma! for h(w)=1/sqrt(|R+w|^2+eps^2)
static inline void computeD(double Rx, double Ry, double Rz) {
    double R[3] = {Rx, Ry, Rz};
    double s0 = Rx * Rx + Ry * Ry + Rz * Rz + EPS2;
    double invs0 = 1.0 / s0;
    Darr[0] = 1.0 / sqrt(s0);
    double* D = Darr.data();
    for (const DG& d : dgs) {
        double rhs = -R[d.gi] * D[d.ibeta];
        if (d.ibme >= 0) rhs -= D[d.ibme];
        for (int j = 0; j < 3; ++j) {
            if (d.f[j] != 0.0 && d.i1[j] >= 0)
                rhs -= 2.0 * R[j] * d.f[j] * D[d.i1[j]];
            if (d.i2[j] >= 0)
                rhs -= d.g[j] * D[d.i2[j]];
        }
        D[d.out] = rhs * d.invgi * invs0;
    }
}

// L[beta] += sum_alpha COEF[beta][alpha] * D[alpha+beta] * M[alpha]
static inline void m2lAdd(const double* M, const double* D, double* L) {
    for (int ib = 0; ib < C; ++ib) {
        __m256d acc = _mm256_setzero_pd();
        const double* cp = COEF.data() + (size_t)ib * Cp;
        const int* dp = DIDX.data() + (size_t)ib * Cp;
        for (int ia = 0; ia < C; ia += 4) {
            __m128i di = _mm_loadu_si128((const __m128i*)(dp + ia));
            __m256d dv = _mm256_i32gather_pd(D, di, 8);
            __m256d cv = _mm256_loadu_pd(cp + ia);
            __m256d mv = _mm256_loadu_pd(M + ia);
            acc = _mm256_fmadd_pd(_mm256_mul_pd(cv, dv), mv, acc);
        }
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        lo = _mm_add_pd(lo, hi);
        lo = _mm_hadd_pd(lo, lo);
        L[ib] += _mm_cvtsd_f64(lo);
    }
}

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    return _mm_cvtsd_f64(lo);
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------
struct Cell {
    double cx, cy, cz;   // center of mass (expansion center)
    double r;            // max distance of any particle to center
    double bx, by, bz;   // cube center
    double half;         // cube half width
    int start, count;
    int ch[8];
    int nch;
};

static int N;
static std::vector<double> PX, PY, PZ, PM;
static std::vector<double> AX, AY, AZ;
static std::vector<double> TX, TY, TZ, TM;   // temp for partition
static std::vector<Cell> cells;
static std::vector<double> MOM;              // numCells * Cp, padded with zeros
static int maxDepth = 26;

static int leafSize = 128;

static void buildCell(int idx, double bx, double by, double bz, double half, int depth) {
    cells[idx].bx = bx; cells[idx].by = by; cells[idx].bz = bz; cells[idx].half = half;
    cells[idx].nch = 0;
    int s = cells[idx].start, e = s + cells[idx].count;
    if (cells[idx].count <= leafSize || depth >= maxDepth) return;
    int cnt[8] = {0,0,0,0,0,0,0,0};
    for (int k = s; k < e; ++k) {
        int o = (PX[k] >= bx ? 1 : 0) | (PY[k] >= by ? 2 : 0) | (PZ[k] >= bz ? 4 : 0);
        cnt[o]++;
    }
    int off[8];
    int acc = s;
    for (int o = 0; o < 8; ++o) { off[o] = acc; acc += cnt[o]; }
    int cur[8];
    memcpy(cur, off, sizeof(cur));
    for (int k = s; k < e; ++k) {
        int o = (PX[k] >= bx ? 1 : 0) | (PY[k] >= by ? 2 : 0) | (PZ[k] >= bz ? 4 : 0);
        int d = cur[o]++;
        TX[d] = PX[k]; TY[d] = PY[k]; TZ[d] = PZ[k]; TM[d] = PM[k];
    }
    memcpy(PX.data() + s, TX.data() + s, (e - s) * sizeof(double));
    memcpy(PY.data() + s, TY.data() + s, (e - s) * sizeof(double));
    memcpy(PZ.data() + s, TZ.data() + s, (e - s) * sizeof(double));
    memcpy(PM.data() + s, TM.data() + s, (e - s) * sizeof(double));
    double h2 = half * 0.5;
    for (int o = 0; o < 8; ++o) {
        if (cnt[o] == 0) continue;
        Cell nc;
        nc.start = off[o]; nc.count = cnt[o]; nc.nch = 0;
        double nbx = bx + ((o & 1) ? h2 : -h2);
        double nby = by + ((o & 2) ? h2 : -h2);
        double nbz = bz + ((o & 4) ? h2 : -h2);
        int ci = (int)cells.size();
        cells.push_back(nc);
        cells[idx].ch[cells[idx].nch++] = ci;
        buildCell(ci, nbx, nby, nbz, h2, depth + 1);
    }
}

// shift moments by delta (binomial translation), separable passes
static std::vector<double> shTmp1, shTmp2;
static void shiftMom(double* M, double dx, double dy, double dz) {
    const double* src = M;
    double* dst = shTmp1.data();
    // x pass: dst[a,b,c] = sum_{i<=a} C(a,i) dx^{a-i} src[i,b,c]
    {
        double pw[P + 2];
        pw[0] = 1.0; for (int t = 1; t <= P; ++t) pw[t] = pw[t-1] * dx;
        for (int b = 0; b <= P; ++b)
            for (int c = 0; c <= P - b; ++c) {
                int ma = P - b - c;
                for (int a = 0; a <= ma; ++a) {
                    double s = 0.0;
                    for (int i = 0; i <= a; ++i)
                        s += binom(a, i) * pw[a - i] * src[LUT[idM(i, b, c)]];
                    dst[LUT[idM(a, b, c)]] = s;
                }
            }
    }
    // y pass
    {
        src = shTmp1.data();
        double pw[P + 2];
        pw[0] = 1.0; for (int t = 1; t <= P; ++t) pw[t] = pw[t-1] * dy;
        for (int a = 0; a <= P; ++a)
            for (int c = 0; c <= P - a; ++c) {
                int mb = P - a - c;
                for (int b = 0; b <= mb; ++b) {
                    double s = 0.0;
                    for (int j = 0; j <= b; ++j)
                        s += binom(b, j) * pw[b - j] * src[LUT[idM(a, j, c)]];
                    shTmp2[LUT[idM(a, b, c)]] = s;
                }
            }
    }
    // z pass
    {
        double pw[P + 2];
        pw[0] = 1.0; for (int t = 1; t <= P; ++t) pw[t] = pw[t-1] * dz;
        for (int a = 0; a <= P; ++a)
            for (int b = 0; b <= P - a; ++b) {
                int mc = P - a - b;
                for (int c = 0; c <= mc; ++c) {
                    double s = 0.0;
                    for (int k = 0; k <= c; ++k)
                        s += binom(c, k) * pw[c - k] * shTmp2[LUT[idM(a, b, k)]];
                    M[LUT[idM(a, b, c)]] = s;
                }
            }
    }
}

static void computeBottomUp() {
    // cells appended in pre-order: reverse gives children before parents
    static std::vector<double> childM;
    childM.resize(Cp);
    for (int ci = (int)cells.size() - 1; ci >= 0; --ci) {
        Cell& c = cells[ci];
        double sx = 0, sy = 0, sz = 0, sm = 0;
        if (c.nch == 0) {
            for (int k = c.start; k < c.start + c.count; ++k) {
                double m = PM[k];
                sx += m * PX[k]; sy += m * PY[k]; sz += m * PZ[k]; sm += m;
            }
        } else {
            for (int j = 0; j < c.nch; ++j) {
                Cell& d = cells[c.ch[j]];
                // degree-0 moment of child = its mass
                double m = MOM[(size_t)c.ch[j] * Cp + LUT[idM(0,0,0)]];
                sx += m * d.cx; sy += m * d.cy; sz += m * d.cz; sm += m;
            }
        }
        double im = (sm > 0) ? 1.0 / sm : 0.0;
        c.cx = sx * im; c.cy = sy * im; c.cz = sz * im;
        // exact radius
        double r2 = 0.0;
        for (int k = c.start; k < c.start + c.count; ++k) {
            double dx = PX[k] - c.cx, dy = PY[k] - c.cy, dz = PZ[k] - c.cz;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 > r2) r2 = d2;
        }
        c.r = sqrt(r2);
        double* M = MOM.data() + (size_t)ci * Cp;
        memset(M, 0, Cp * sizeof(double));
        if (c.nch == 0) {
            for (int k = c.start; k < c.start + c.count; ++k) {
                double dx = PX[k] - c.cx, dy = PY[k] - c.cy, dz = PZ[k] - c.cz;
                double wx[P + 2], wy[P + 2], wz[P + 2];
                wx[0] = wy[0] = wz[0] = 1.0;
                for (int t = 1; t <= P; ++t) {
                    wx[t] = wx[t-1]*dx; wy[t] = wy[t-1]*dy; wz[t] = wz[t-1]*dz;
                }
                double m = PM[k];
                for (int q = 0; q < C; ++q)
                    M[q] += m * wx[mA[q]] * wy[mB[q]] * wz[mC[q]];
            }
        } else {
            for (int j = 0; j < c.nch; ++j) {
                int di = c.ch[j];
                Cell& d = cells[di];
                const double* Md = MOM.data() + (size_t)di * Cp;
                memcpy(childM.data(), Md, Cp * sizeof(double));
                shiftMom(childM.data(), d.cx - c.cx, d.cy - c.cy, d.cz - c.cz);
                for (int q = 0; q < C; ++q) M[q] += childM[q];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main traversal
// ---------------------------------------------------------------------------
static double RHO = 0.5;
static std::vector<double> Lbuf;
static std::vector<int> nearList; // pairs (start,end)
static std::vector<int> stack;

static long long nFar = 0, nNearPairs = 0;

static void selfTestD() {
    double R[3] = {1.3, -0.7, 2.1};
    computeD(R[0], R[1], R[2]);
    double s0 = R[0]*R[0]+R[1]*R[1]+R[2]*R[2]+EPS2;
    for (double t = 0.01; t < 0.4; t *= 2) {
        double z[3] = {t*0.3, -t*0.5, t*0.7};
        double sv = s0;
        for (int d = 0; d < 3; ++d) sv += 2*R[d]*z[d] + z[d]*z[d];
        double exact = 1.0/sqrt(sv);
        double ser = 0.0;
        for (int a = 0; a <= Q; ++a)
            for (int b = 0; b <= Q-a; ++b)
                for (int c = 0; c <= Q-a-b; ++c) {
                    double p = 1.0;
                    for (int k=0;k<a;k++) p*=z[0];
                    for (int k=0;k<b;k++) p*=z[1];
                    for (int k=0;k<c;k++) p*=z[2];
                    ser += Darr[idD(a,b,c)]*p;
                }
        fprintf(stderr, "[dbg] Dtest t=%g exact=%.16g series=%.16g rel=%.3g\n",
                t, exact, ser, fabs(ser-exact)/exact);
    }
    // single particle monopole check
    double v[3] = {0.02,-0.03,0.01};
    double w[3] = {R[0]+v[0],R[1]+v[1],R[2]+v[2]};
    double s2 = w[0]*w[0]+w[1]*w[1]+w[2]*w[2]+EPS2;
    double inv = 1.0/(s2*sqrt(s2));
    double ax = -w[0]*inv, ay=-w[1]*inv, az=-w[2]*inv;
    double pwx[P+2],pwy[P+2],pwz[P+2];
    pwx[0]=pwy[0]=pwz[0]=1.0;
    for(int k=1;k<=P;k++){pwx[k]=pwx[k-1]*v[0];pwy[k]=pwy[k-1]*v[1];pwz[k]=pwz[k-1]*v[2];}
    double gx=0,gy=0,gz=0;
    for(int q=0;q<C;q++){
        int a=mA[q],b=mB[q],c=mC[q];
        double D=Darr[idD(a,b,c)];
        if(a)gx+=D*a*pwx[a-1]*pwy[b]*pwz[c];
        if(b)gy+=D*b*pwx[a]*pwy[b-1]*pwz[c];
        if(c)gz+=D*c*pwx[a]*pwy[b]*pwz[c-1];
    }
    fprintf(stderr, "[dbg] Dgrad exact=(%.10g,%.10g,%.10g) series=(%.10g,%.10g,%.10g)\n",
            ax,ay,az,gx,gy,gz);
}

static void doLeaf(int ti) {
    Cell& T = cells[ti];
    double rT = T.r;
    double cx = T.cx, cy = T.cy, cz = T.cz;
    memset(Lbuf.data(), 0, C * sizeof(double));
    nearList.clear();
    stack.clear();
    stack.push_back(0);
    while (!stack.empty()) {
        int si = stack.back(); stack.pop_back();
        Cell& S = cells[si];
        double dx = S.cx - cx, dy = S.cy - cy, dz = S.cz - cz;
        double R2 = dx*dx + dy*dy + dz*dz;
        double rr = (S.r + rT) / RHO;
        if (R2 > rr * rr) {
            // far: M2L ; note R vector used is target - source = -(dx,dy,dz)
            computeD(-dx, -dy, -dz);
            ++nFar;
            m2lAdd(MOM.data() + (size_t)si * Cp, Darr.data(), Lbuf.data());
        } else if (S.nch == 0) {
            nNearPairs += S.count;
            nearList.push_back(S.start);
            nearList.push_back(S.start + S.count);
        } else {
            for (int j = 0; j < S.nch; ++j) stack.push_back(S.ch[j]);
        }
    }
    const __m256d veps = _mm256_set1_pd(EPS2);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vthree = _mm256_set1_pd(1.5);
    for (int i = T.start; i < T.start + T.count; ++i) {
        double xi = PX[i], yi = PY[i], zi = PZ[i];
        __m256d axi = _mm256_setzero_pd(), ayi = _mm256_setzero_pd(), azi = _mm256_setzero_pd();
        __m256d vxi = _mm256_set1_pd(xi), vyi = _mm256_set1_pd(yi), vzi = _mm256_set1_pd(zi);
        for (size_t u = 0; u < nearList.size(); u += 2) {
            int s = nearList[u], e = nearList[u+1];
            int j = s;
            for (; j + 3 < e; j += 4) {
                __m256d ddx = _mm256_sub_pd(_mm256_loadu_pd(PX.data() + j), vxi);
                __m256d ddy = _mm256_sub_pd(_mm256_loadu_pd(PY.data() + j), vyi);
                __m256d ddz = _mm256_sub_pd(_mm256_loadu_pd(PZ.data() + j), vzi);
                __m256d s2 = _mm256_add_pd(veps,
                    _mm256_add_pd(_mm256_mul_pd(ddx, ddx),
                        _mm256_add_pd(_mm256_mul_pd(ddy, ddy), _mm256_mul_pd(ddz, ddz))));
                __m128 s2f = _mm256_cvtpd_ps(s2);
                __m128 rf = _mm_rsqrt_ps(s2f);
                __m256d r = _mm256_cvtps_pd(rf);
                for (int it = 0; it < 3; ++it) {
                    __m256d rr2 = _mm256_mul_pd(r, r);
                    r = _mm256_mul_pd(r, _mm256_fnmadd_pd(_mm256_mul_pd(s2, rr2), vhalf, vthree));
                }
                __m256d inv = _mm256_mul_pd(_mm256_mul_pd(r, r), r);
                __m256d f = _mm256_mul_pd(inv, _mm256_loadu_pd(PM.data() + j));
                axi = _mm256_fmadd_pd(ddx, f, axi);
                ayi = _mm256_fmadd_pd(ddy, f, ayi);
                azi = _mm256_fmadd_pd(ddz, f, azi);
            }
            for (; j < e; ++j) {
                double ddx = PX[j] - xi, ddy = PY[j] - yi, ddz = PZ[j] - zi;
                double s2 = ddx*ddx + ddy*ddy + ddz*ddz + EPS2;
                double inv = 1.0 / (s2 * sqrt(s2));
                double f = inv * PM[j];
                axi = _mm256_add_pd(axi, _mm256_set1_pd(ddx*f));
                ayi = _mm256_add_pd(ayi, _mm256_set1_pd(ddy*f));
                azi = _mm256_add_pd(azi, _mm256_set1_pd(ddz*f));
            }
        }
        double ax = hsum256(axi), ay = hsum256(ayi), az = hsum256(azi);
        // local expansion: a = grad_v sum_beta L[beta] v^beta
        double vx = xi - cx, vy = yi - cy, vz = zi - cz;
        double wx[P + 2], wy[P + 2], wz[P + 2];
        wx[0] = wy[0] = wz[0] = 1.0;
        for (int t = 1; t <= P; ++t) { wx[t]=wx[t-1]*vx; wy[t]=wy[t-1]*vy; wz[t]=wz[t-1]*vz; }
        for (int q = 0; q < C; ++q) {
            int a = mA[q], b = mB[q], c = mC[q];
            double L = Lbuf[q];
            if (a) ax += L * a * wx[a-1] * wy[b] * wz[c];
            if (b) ay += L * b * wx[a] * wy[b-1] * wz[c];
            if (c) az += L * c * wx[a] * wy[b] * wz[c-1];
        }
        AX[i] = ax; AY[i] = ay; AZ[i] = az;
    }
}

// debug: compare one far-cell interaction against direct summation
static void selfTestM2L() {
    // find a leaf with at least one far interaction
    int ti = -1, si = -1;
    for (int t = 0; t < (int)cells.size() && ti < 0; ++t) {
        if (cells[t].nch) continue;
        Cell& T = cells[t];
        std::vector<int> stk{0};
        while (!stk.empty() && si < 0) {
            int s = stk.back(); stk.pop_back();
            Cell& S = cells[s];
            double dx=S.cx-T.cx, dy=S.cy-T.cy, dz=S.cz-T.cz;
            double R2=dx*dx+dy*dy+dz*dz;
            double rr=(S.r+T.r)/RHO;
            if (R2 > rr*rr) { if (S.count > 4) { ti=t; si=s; } }
            else if (S.nch) for(int j=0;j<S.nch;j++) stk.push_back(S.ch[j]);
        }
    }
    if (ti < 0) { fprintf(stderr, "[dbg] no far pair found\n"); return; }
    Cell& T=cells[ti]; Cell& S=cells[si];
    int i = T.start;
    double ex=0,ey=0,ez=0;
    for (int j=S.start;j<S.start+S.count;j++) {
        double dx=PX[j]-PX[i],dy=PY[j]-PY[i],dz=PZ[j]-PZ[i];
        double s2=dx*dx+dy*dy+dz*dz+EPS2;
        double f=PM[j]/(s2*sqrt(s2));
        ex+=dx*f;ey+=dy*f;ez+=dz*f;
    }
    std::vector<double> L(Cp,0.0);
    computeD(T.cx-S.cx,T.cy-S.cy,T.cz-S.cz);
    m2lAdd(MOM.data()+(size_t)si*Cp,Darr.data(),L.data());
    double vx=PX[i]-T.cx,vy=PY[i]-T.cy,vz=PZ[i]-T.cz;
    double wx[P+2],wy[P+2],wz[P+2]; wx[0]=wy[0]=wz[0]=1.0;
    for(int k=1;k<=P;k++){wx[k]=wx[k-1]*vx;wy[k]=wy[k-1]*vy;wz[k]=wz[k-1]*vz;}
    double gx=0,gy=0,gz=0;
    for(int q=0;q<C;q++){
        int a=mA[q],b=mB[q],c=mC[q];
        double l=L[q];
        if(a)gx+=l*a*wx[a-1]*wy[b]*wz[c];
        if(b)gy+=l*b*wx[a]*wy[b-1]*wz[c];
        if(c)gz+=l*c*wx[a]*wy[b]*wz[c-1];
    }
    fprintf(stderr, "[dbg] M2Lpair T=%d(count=%d,r=%.4g) S=%d(count=%d,r=%.4g) R=%.4g exact=(%.8g,%.8g,%.8g) m2l=(%.8g,%.8g,%.8g)\n",
        ti,T.count,T.r,si,S.count,S.r,
        hypot(hypot(T.cx-S.cx,T.cy-S.cy),T.cz-S.cz),ex,ey,ez,gx,gy,gz);
}

// ---------------------------------------------------------------------------
// Debug: estimate error on a subset by direct summation
// ---------------------------------------------------------------------------
static void debugCheck(double t0) {
    int S = 192;
    unsigned long long seed = 88172645463325252ULL;
    double sum2 = 0.0;
    int cnt = 0;
    for (int t = 0; t < S; ++t) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        int i = (int)(seed % (unsigned long long)N);
        double ex = 0, ey = 0, ez = 0;
        double xi = PX[i], yi = PY[i], zi = PZ[i];
        for (int j = 0; j < N; ++j) {
            double dx = PX[j]-xi, dy = PY[j]-yi, dz = PZ[j]-zi;
            double s2 = dx*dx+dy*dy+dz*dz+EPS2;
            double f = PM[j] / (s2*sqrt(s2));
            ex += dx*f; ey += dy*f; ez += dz*f;
        }
        double dx = AX[i]-ex, dy = AY[i]-ey, dz = AZ[i]-ez;
        double den = ex*ex+ey*ey+ez*ez;
        sum2 += (dx*dx+dy*dy+dz*dz)/den;
        ++cnt;
    }
    fprintf(stderr, "[dbg] subset RMS rel err = %.6g  (cpu %.3fs)\n",
            sqrt(sum2/cnt), cputime()-t0);
}

int main(int argc, char** argv) {
    double t0 = cputime();
    if (argc < 3) return 1;
    dbg = getenv("GALAXY_DEBUG") ? 1 : 0;
    double limit = 1e9;
    if (getenv("GALAXY_CPU_LIMIT_S")) limit = atof(getenv("GALAXY_CPU_LIMIT_S"));
    const char* inp = argv[1], *outp = argv[2];

    FILE* f = fopen(inp, "rb");
    if (!f) return 2;
    struct stat st;
    fstat(fileno(f), &st);
    N = (int)(st.st_size / 32);
    std::vector<double> raw((size_t)4 * N);
    if (fread(raw.data(), 32, N, f) != (size_t)N) { fclose(f); return 3; }
    fclose(f);

    // ---- choose parameters from CPU budget ----
    // presets: (P, rho, leafSize). Estimated cost model calibrated on dev data.
    struct Preset { int p; double rho; int b; double cost; };
    static const Preset presets[] = {
        // cost ~ per-particle microseconds at N=65536 (to be calibrated)
        {2, 0.70,  64, 0.0},
        {3, 0.60,  64, 0.0},
        {4, 0.55,  64, 0.0},
        {4, 0.45,  96, 0.0},
        {6, 0.50,  96, 0.0},
        {6, 0.40, 128, 0.0},
        {8, 0.45, 128, 0.0},
        {8, 0.35, 192, 0.0},
        {10,0.40, 192, 0.0},
    };
    (void)presets; (void)limit;
    int pp = 6; double rr = 0.5; int bb = 96;
    if (getenv("GALAXY_P")) pp = atoi(getenv("GALAXY_P"));
    if (getenv("GALAXY_RHO")) rr = atof(getenv("GALAXY_RHO"));
    if (getenv("GALAXY_B")) bb = atoi(getenv("GALAXY_B"));
    P = pp; RHO = rr; leafSize = bb;

    initTables(P);
    if (dbg) selfTestD();
    shTmp1.resize(Cp); shTmp2.resize(Cp);
    Lbuf.resize(Cp);
    PX.resize(N); PY.resize(N); PZ.resize(N); PM.resize(N);
    TX.resize(N); TY.resize(N); TZ.resize(N); TM.resize(N);
    AX.resize(N); AY.resize(N); AZ.resize(N);
    for (int i = 0; i < N; ++i) {
        PM[i] = raw[(size_t)4*i+0];
        PX[i] = raw[(size_t)4*i+1];
        PY[i] = raw[(size_t)4*i+2];
        PZ[i] = raw[(size_t)4*i+3];
    }
    raw.clear(); raw.shrink_to_fit();

    double mn[3] = {PX[0], PY[0], PZ[0]}, mx[3] = {PX[0], PY[0], PZ[0]};
    for (int i = 1; i < N; ++i) {
        const double* p[3] = {&PX[i], &PY[i], &PZ[i]};
        for (int d = 0; d < 3; ++d) {
            if (*p[d] < mn[d]) mn[d] = *p[d];
            if (*p[d] > mx[d]) mx[d] = *p[d];
        }
    }
    double cx[3], half = 0;
    for (int d = 0; d < 3; ++d) {
        cx[d] = 0.5*(mn[d]+mx[d]);
        half = std::max(half, mx[d]-mn[d]);
    }
    half = half*0.5*(1.0+1e-9);
    cells.clear();
    cells.reserve((size_t)(4.0 * N / leafSize) + 64);
    Cell root; root.start = 0; root.count = N; root.nch = 0;
    cells.push_back(root);
    buildCell(0, cx[0], cx[1], cx[2], half, 0);
    MOM.assign((size_t)cells.size() * Cp, 0.0);
    computeBottomUp();
    if (dbg) selfTestM2L();
    if (getenv("GALAXY_SELFTEST")) return 0;

    if (dbg) fprintf(stderr, "[dbg] N=%d P=%d rho=%.3g B=%d cells=%zu t_build=%.3f\n",
                     N, P, RHO, leafSize, cells.size(), cputime()-t0);

    for (int ci = 0; ci < (int)cells.size(); ++ci)
        if (cells[ci].nch == 0) doLeaf(ci);

    if (dbg) fprintf(stderr, "[dbg] t_total=%.3f cpu, limit=%.3f nFar=%lld nNearPerTarget=%.1f\n",
                     cputime()-t0, limit, nFar, (double)nNearPairs / (double)(N ? N : 1));
    if (dbg) debugCheck(t0);

    std::vector<double> out((size_t)3 * N);
    for (int i = 0; i < N; ++i) {
        out[(size_t)3*i+0] = AX[i];
        out[(size_t)3*i+1] = AY[i];
        out[(size_t)3*i+2] = AZ[i];
    }
    FILE* g = fopen(outp, "wb");
    if (!g) return 4;
    size_t nw = fwrite(out.data(), 24, N, g);
    fclose(g);
    if (nw != (size_t)N) return 5;
    return 0;
}
