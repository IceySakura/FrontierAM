// Exponential propagation for a symmetric PSD sparse operator.
//
//   y = exp(-t L) b0 + t phi1(-t L) b1
//
// Two complementary strategies are used, both chosen from the input matrix:
//
//  * If the graph of L splits into connected components of small size (this is
//    the "diagonal/log-spectrum" family), each block is diagonalised exactly
//    (Jacobi in long double) and the two scalar functions are evaluated at its
//    eigenvalues.  This is O(nnz + n k) per request with k <= 32 and is
//    accurate to working precision for every precision level and every t.
//
//  * Otherwise a Chebyshev expansion on [0, B] is used.  B is a rigorous and
//    tight upper bound for lambda_max(L) from a scaled Gershgorin
//    (Collatz-Wielandt) bound on |L|.  With S = 2 L / B - I the coefficients
//    of exp(-t lambda) are a_0 = s_0, a_k = 2 (-1)^k s_k, where
//    s_k = exp(-z) I_k(z), z = t B / 2, obtained in O(K) by Miller's stable
//    backward recurrence normalised with I_0 + 2 sum I_k = exp(z).  For
//    t phi1(-t lambda) the exact tail-sum identity
//       c_0 = (4/B) M_0, c_k = (8/B) (-1)^k M_k,
//       M_k = sum_{j>k} (j-k) s_j
//    gives the coefficients, and both coefficient tails are available in the
//    same O(K) sweep.  A forward Chebyshev three-term recurrence applies both
//    series (one column per right-hand side) with a single sparse traversal
//    per degree, and stops as soon as the analytic tail bound is below a small
//    fraction of the acceptance threshold estimated from the current partial
//    sum.  Columns with an exactly zero right-hand side are omitted.
//
// All input-dependent work is inside the measured session.  The program prints
// READY before reading any input and writes exactly 8*n bytes per request.
// Only libc/libm/libstdc++ are used.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <unistd.h>

static bool read_exact(void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = read(0, p + got, len - got);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

static bool write_exact(const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t put = 0;
    while (put < len) {
        ssize_t r = write(1, p + put, len - put);
        if (r <= 0) return false;
        put += static_cast<size_t>(r);
    }
    return true;
}

static double json_number(const std::string& js, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    const char* p = std::strstr(js.c_str(), pat.c_str());
    if (!p) return 0.0;
    p += pat.size();
    while (*p == ':' || *p == ' ' || *p == '\t') ++p;
    return std::strtod(p, nullptr);
}

struct Session {
    uint32_t n = 0, nnz = 0, requests = 0;
    double t = 0.0, rtol = 0.0, atol = 0.0;
    std::vector<uint32_t> ptr, idx;
    std::vector<double> val, absval;

    // Exact small-block path.
    bool exact = false;
    std::vector<uint32_t> comp_start, comp_size, comp_vertices, qoff, pos;
    std::vector<double> eigQ, f0, f1;

    // Chebyshev path.
    bool use_half = false;
    std::vector<uint32_t> hi, hj;
    std::vector<double> ha, hdiag;
    double B = 0.0, alpha = 0.0, stop_c = 0.02;
    int K = 0;
    std::vector<double> cE, cH, tailE, tailH, tailBoth;
    bool use_taylor = false;
    std::vector<double> work;
};

// ---------------------------------------------------------------------------
// Exact diagonalisation of small symmetric blocks (long double for relative
// accuracy on tiny eigenvalues).
// ---------------------------------------------------------------------------
static void jacobi(long double* A, int k, long double* lam, long double* Q) {
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) Q[i * k + j] = (i == j) ? 1.0L : 0.0L;
    long double scale = 0.0L;
    for (int i = 0; i < k * k; ++i) scale = std::max(scale, std::fabs(A[i]));
    if (scale == 0.0L) scale = 1.0L;
    for (int sweep = 0; sweep < 40; ++sweep) {
        long double off = 0.0L;
        for (int p = 0; p < k; ++p)
            for (int q = p + 1; q < k; ++q)
                off = std::max(off, std::fabs(A[p * k + q]));
        if (off <= 4e-19L * scale) break;
        for (int p = 0; p < k; ++p) {
            for (int q = p + 1; q < k; ++q) {
                long double apq = A[p * k + q];
                if (std::fabs(apq) <= 4e-19L * scale) continue;
                long double app = A[p * k + p], aqq = A[q * k + q];
                long double theta = 0.5L * (aqq - app) / apq;
                long double tt = (theta >= 0 ? 1.0L : -1.0L) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0L));
                if (theta == 0.0L) tt = 1.0L;
                long double c = 1.0L / std::sqrt(1.0L + tt * tt);
                long double s = tt * c;
                for (int i = 0; i < k; ++i) {
                    if (i == p || i == q) continue;
                    long double aip = A[i * k + p], aiq = A[i * k + q];
                    A[i * k + p] = A[p * k + i] = c * aip - s * aiq;
                    A[i * k + q] = A[q * k + i] = s * aip + c * aiq;
                }
                A[p * k + p] = c * c * app - 2.0L * s * c * apq + s * s * aqq;
                A[q * k + q] = s * s * app + 2.0L * s * c * apq + c * c * aqq;
                A[p * k + q] = A[q * k + p] = 0.0L;
                for (int i = 0; i < k; ++i) {
                    long double qip = Q[i * k + p], qiq = Q[i * k + q];
                    Q[i * k + p] = c * qip - s * qiq;
                    Q[i * k + q] = s * qip + c * qiq;
                }
            }
        }
    }
    for (int i = 0; i < k; ++i) lam[i] = A[i * k + i];
}

static bool build_components(Session& S, int cap = 96) {
    const uint32_t n = S.n;
    std::vector<uint32_t> parent(n), rnk(n, 0), sz(n, 1);
    std::iota(parent.begin(), parent.end(), 0u);
    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p) {
            uint32_t j = S.idx[p];
            if (j == i || S.val[p] == 0.0) continue;
            uint32_t a = find(i), b = find(j);
            if (a == b) continue;
            if (rnk[a] < rnk[b]) std::swap(a, b);
            parent[b] = a;
            sz[a] += sz[b];
            if (sz[a] > static_cast<uint32_t>(cap)) return false;
            if (rnk[a] == rnk[b]) ++rnk[a];
        }
    }
    std::vector<uint32_t> root2comp(n, UINT32_MAX), sizes;
    uint32_t nc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t r = find(i);
        if (root2comp[r] == UINT32_MAX) {
            root2comp[r] = nc++;
            sizes.push_back(0);
        }
        ++sizes[root2comp[r]];
    }
    double cost = 0.0;
    for (uint32_t c = 0; c < nc; ++c) {
        if (sizes[c] > static_cast<uint32_t>(cap)) return false;
        double k = static_cast<double>(sizes[c]);
        cost += k * k * k;
    }
    // Keep the dense eigensolves cheap even when a few medium blocks occur.
    if (cost > 5e6) return false;
    S.comp_start.assign(nc + 1, 0);
    S.comp_size = sizes;
    for (uint32_t c = 0; c < nc; ++c) S.comp_start[c + 1] = S.comp_start[c] + sizes[c];
    S.comp_vertices.resize(n);
    S.pos.resize(n);
    std::vector<uint32_t> fill(S.comp_start.begin(), S.comp_start.end() - 1);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t c = root2comp[find(i)];
        uint32_t at = fill[c]++;
        S.comp_vertices[at] = i;
        S.pos[i] = at - S.comp_start[c];
    }
    S.qoff.assign(nc + 1, 0);
    for (uint32_t c = 0; c < nc; ++c)
        S.qoff[c + 1] = S.qoff[c] + sizes[c] * sizes[c];
    S.eigQ.assign(S.qoff[nc], 0.0L);
    S.f0.assign(n, 0.0L);
    S.f1.assign(n, 0.0L);
    const long double tl = static_cast<long double>(S.t);
    for (uint32_t c = 0; c < nc; ++c) {
        int k = static_cast<int>(sizes[c]);
        std::vector<long double> A(k * k, 0.0L), lam(k), Q(k * k);
        for (uint32_t ii = 0; ii < sizes[c]; ++ii) {
            uint32_t v = S.comp_vertices[S.comp_start[c] + ii];
            for (uint32_t p = S.ptr[v]; p < S.ptr[v + 1]; ++p) {
                uint32_t j = S.idx[p];
                uint32_t jj = S.pos[j];
                if (jj < sizes[c])
                    A[static_cast<size_t>(ii) * k + jj] = static_cast<long double>(S.val[p]);
            }
        }
        // Symmetrise (input is symmetric; this also tolerates rounding).
        for (int i = 0; i < k; ++i)
            for (int j = i + 1; j < k; ++j) {
                long double v = 0.5L * (A[i * k + j] + A[j * k + i]);
                A[i * k + j] = A[j * k + i] = v;
            }
        jacobi(A.data(), k, lam.data(), Q.data());
        for (int j = 0; j < k; ++j) {
            long double w = lam[j];
            if (w < 0.0L) w = 0.0L;           // L is PSD
            long double x = tl * w;
            long double e0 = (x > 700.0L) ? 0.0L : std::exp(-x);
            long double e1;
            if (w > 0.0L) {
                e1 = (x > 700.0L) ? 1.0L / w : -std::expm1(-x) / w;
            } else {
                e1 = tl;
            }
            size_t idx = S.comp_start[c] + j;
            S.f0[idx] = static_cast<double>(e0);
            S.f1[idx] = static_cast<double>(e1);
            for (int i = 0; i < k; ++i)
                S.eigQ[S.qoff[c] + static_cast<size_t>(i) * k + j] = static_cast<double>(Q[i * k + j]);
        }
    }
    S.exact = true;
    return true;
}

static void exact_solution(const Session& S, const double* b0, const double* b1,
                           double* y) {
    double tmp0[96], tmp1[96], q[96], out[96];
    for (uint32_t c = 0; c + 1 < S.comp_start.size(); ++c) {
        int k = static_cast<int>(S.comp_size[c]);
        const double* Q = &S.eigQ[S.qoff[c]];
        for (int i = 0; i < k; ++i) {
            uint32_t v = S.comp_vertices[S.comp_start[c] + i];
            tmp0[i] = b0[v]; tmp1[i] = b1[v];
        }
        for (int j = 0; j < k; ++j) {
            double a0 = 0.0, a1 = 0.0;
            for (int i = 0; i < k; ++i) {
                a0 += Q[i * k + j] * tmp0[i];
                a1 += Q[i * k + j] * tmp1[i];
            }
            q[j] = S.f0[S.comp_start[c] + j] * a0 + S.f1[S.comp_start[c] + j] * a1;
        }
        for (int i = 0; i < k; ++i) {
            double acc = 0.0;
            for (int j = 0; j < k; ++j) acc += Q[i * k + j] * q[j];
            out[i] = acc;
        }
        for (int i = 0; i < k; ++i)
            y[S.comp_vertices[S.comp_start[c] + i]] = out[i];
    }
}

// ---------------------------------------------------------------------------
// Chebyshev coefficients and spectral bound.
// ---------------------------------------------------------------------------
static void scaled_bessel(double z, int K, std::vector<double>& s) {
    s.assign(K + 1, 0.0);
    if (z < 1e-290) { s[0] = 1.0; return; }
    int extra = std::max(100, static_cast<int>(2.0 * std::sqrt(std::max(z, 1.0) * 40.0)));
    int M = K + extra;
    std::vector<double> r(M + 2, 0.0);
    r[M] = 1.0;
    const double kThreshold = 1e20, kScale = 1e-20;
    for (int k = M; k >= 1; --k) {
        double v = r[k + 1] + (2.0 * k / z) * r[k];
        if (!(v < 1e300) || v > kThreshold) {
            for (int j = 0; j <= M + 1; ++j) r[j] *= kScale;
            v = r[k + 1] + (2.0 * k / z) * r[k];
        }
        r[k - 1] = v;
    }
    double norm = r[0];
    for (int k = 1; k <= M; ++k) norm += 2.0 * r[k];
    double inv = 1.0 / norm;
    for (int k = 0; k <= K; ++k) {
        s[k] = r[k] * inv;
        if (!(s[k] >= 0.0)) s[k] = 0.0;
    }
}

static bool compute_spectral_bound(Session& S, double rowmax) {
    const uint32_t n = S.n;
    if (!(rowmax > 0.0)) return false;
    // For small t*||L|| the degree barely depends on B; a cheap bound suffices.
    double eta0 = S.t * rowmax;
    int iters = (eta0 < 20.0) ? 0 : (eta0 < 200.0 ? 4 : (eta0 < 2000.0 ? 6 : 10));
    double bound = rowmax;
    if (iters > 0) {
        std::vector<double> s(n, 1.0), sn(n, 1.0);
        for (int it = 0; it < iters; ++it) {
            double mx = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                double r = 0.0;
                for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p)
                    r += S.absval[p] * s[S.idx[p]];
                sn[i] = 1.0 + r;
                mx = std::max(mx, sn[i]);
            }
            double inv = 1.0 / mx;
            for (uint32_t i = 0; i < n; ++i) s[i] = sn[i] * inv;
        }
        bound = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            double r = 0.0;
            for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p)
                r += S.absval[p] * s[S.idx[p]];
            bound = std::max(bound, r / s[i]);
        }
    }
    S.B = std::min(rowmax, bound) * (1.0 + 1e-9);
    S.alpha = 2.0 / S.B;
    return true;
}

static void prepare_coefficients(Session& S) {
    double eta = S.t * S.B;
    if (eta <= 1e-6) { S.use_taylor = true; return; }
    int K = static_cast<int>(std::ceil(std::sqrt(eta * 50.0))) + 150;
    if (K > 12000) K = 12000;
    S.K = K;
    double z = eta / 2.0;
    std::vector<double> s;
    scaled_bessel(z, K, s);
    std::vector<double> M(K + 1, 0.0);
    double P = 0.0, Mv = 0.0;
    for (int k = K; k >= 1; --k) {
        P += s[k];
        Mv += P;
        M[k - 1] = Mv;
    }
    S.cE.assign(K + 1, 0.0);
    S.cH.assign(K + 1, 0.0);
    S.tailE.assign(K + 1, 0.0);
    S.tailH.assign(K + 1, 0.0);
    S.tailBoth.assign(K + 1, 0.0);
    double cumS = 0.0, cumM = 0.0;
    for (int k = K; k >= 0; --k) {
        S.tailE[k] = 2.0 * cumS;
        S.tailH[k] = (8.0 / S.B) * cumM;
        S.tailBoth[k] = S.tailE[k] + S.tailH[k];
        cumS += s[k];
        cumM += M[k];
    }
    for (int k = 0; k <= K; ++k) {
        double sign = (k & 1) ? -1.0 : 1.0;
        S.cE[k] = (k == 0 ? 1.0 : 2.0) * sign * s[k];
        S.cH[k] = ((k == 0 ? 4.0 : 8.0) / S.B) * sign * M[k];
    }
    S.work.assign(static_cast<size_t>(6) * 2 * S.n, 0.0);
}

// Build an upper-triangle (symmetric) representation of L.  Both stored
// triangles are averaged, so this works even if the two values differ by
// rounding.  It halves the matrix bytes read by each matvec.
static void build_half(Session& S) {
    const uint32_t n = S.n;
    S.hdiag.assign(n, 0.0);
    S.hi.reserve(S.nnz / 2 + 1);
    S.hj.reserve(S.nnz / 2 + 1);
    S.ha.reserve(S.nnz / 2 + 1);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p) {
            uint32_t j = S.idx[p];
            if (j == i) { S.hdiag[i] = S.val[p]; continue; }
            if (j < i) continue;
            // Find the stored (j,i) entry and average symmetric values.
            uint32_t lo = S.ptr[j], hi = S.ptr[j + 1];
            while (lo < hi) {
                uint32_t mid = (lo + hi) >> 1;
                if (S.idx[mid] < i) lo = mid + 1; else hi = mid;
            }
            double a = S.val[p];
            if (lo < S.ptr[j + 1] && S.idx[lo] == i)
                a = 0.5 * (a + S.val[lo]);
            S.hi.push_back(i);
            S.hj.push_back(j);
            S.ha.push_back(a);
        }
    }
    S.use_half = true;
}

// S*x = (2/B) L*x - x, with nrhs interleaved columns.
static void apply_S(const Session& S, int nrhs, const double* x, double* out) {
    const uint32_t n = S.n;
    const uint32_t* ptr = S.ptr.data();
    const uint32_t* idx = S.idx.data();
    const double* val = S.val.data();
    const double alpha = S.alpha;
    if (nrhs == 1 && S.use_half) {
        for (uint32_t i = 0; i < n; ++i) out[i] = alpha * S.hdiag[i] * x[i] - x[i];
        const size_t m = S.ha.size();
        for (size_t p = 0; p < m; ++p) {
            uint32_t i = S.hi[p], j = S.hj[p];
            double a = alpha * S.ha[p];
            out[i] += a * x[j];
            out[j] += a * x[i];
        }
    } else if (nrhs == 1) {
        for (uint32_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (uint32_t p = ptr[i]; p < ptr[i + 1]; ++p)
                acc += val[p] * x[idx[p]];
            out[i] = alpha * acc - x[i];
        }
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            double a0 = 0.0, a1 = 0.0;
            for (uint32_t p = ptr[i]; p < ptr[i + 1]; ++p) {
                uint32_t j = idx[p];
                double a = val[p];
                a0 += a * x[2 * j];
                a1 += a * x[2 * j + 1];
            }
            out[2 * i] = alpha * a0 - x[2 * i];
            out[2 * i + 1] = alpha * a1 - x[2 * i + 1];
        }
    }
}

// Forward Chebyshev recurrence with a self-tuning stopping rule.
// column 0 uses cE/tailE, column 1 uses cH/tailH.
static void forward_chebyshev(const Session& S, int nrhs, const double* c0,
                              const double* c1, const double* tail,
                              const double* rhs, double* result) {
    const uint32_t n = S.n;
    const size_t stride = static_cast<size_t>(nrhs) * n;
    const int K = S.K;
    double* buf = const_cast<double*>(S.work.data());
    double* prev = buf;                 // T_{k-1}
    double* cur = buf + stride;         // T_k
    double* next = buf + 2 * stride;    // T_{k+1}
    double* w = buf + 3 * stride;       // S * T_k
    double* acc = buf + 4 * stride;     // accumulated sum
    double* v = buf + 5 * stride;       // RHS (T_0)
    std::memcpy(v, rhs, stride * sizeof(double));
    std::memcpy(prev, v, stride * sizeof(double));
    // acc = c_0 * T_0
    for (uint32_t i = 0; i < n; ++i) {
        if (nrhs == 1) acc[i] = c0[0] * v[i];
        else { acc[2 * i] = c0[0] * v[2 * i]; acc[2 * i + 1] = c1[0] * v[2 * i + 1]; }
    }
    double ymax = 0.0;
    auto current_max = [&]() {
        double mx = 0.0;
        if (nrhs == 1) {
            for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, std::fabs(acc[i]));
        } else {
            for (uint32_t i = 0; i < n; ++i)
                mx = std::max(mx, std::fabs(acc[2 * i] + acc[2 * i + 1]));
        }
        return mx;
    };
    // T_1 = S T_0
    apply_S(S, nrhs, prev, cur);
    for (uint32_t i = 0; i < n; ++i) {
        if (nrhs == 1) acc[i] += c0[1] * cur[i];
        else { acc[2 * i] += c0[1] * cur[2 * i]; acc[2 * i + 1] += c1[1] * cur[2 * i + 1]; }
    }
    int consecutive = 0;
    for (int k = 2; k <= K; ++k) {
        apply_S(S, nrhs, cur, w);
        for (size_t q = 0; q < stride; ++q) next[q] = 2.0 * w[q] - prev[q];
        const double e = c0[k], h = (nrhs == 2 ? c1[k] : 0.0);
        ymax = 0.0;
        if (nrhs == 1) {
            for (uint32_t i = 0; i < n; ++i) {
                acc[i] += e * next[i];
                ymax = std::max(ymax, std::fabs(acc[i]));
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                acc[2 * i] += e * next[2 * i];
                acc[2 * i + 1] += h * next[2 * i + 1];
                ymax = std::max(ymax, std::fabs(acc[2 * i] + acc[2 * i + 1]));
            }
        }
        double thr = S.atol + S.rtol * ymax;
        if (k >= 4 && tail[k] <= S.stop_c * thr) {
            if (++consecutive >= 2) break;
        } else {
            consecutive = 0;
        }
        double* tmp = prev; prev = cur; cur = next; next = tmp;
    }
    if (nrhs == 1) {
        std::memcpy(result, acc, n * sizeof(double));
    } else {
        for (uint32_t i = 0; i < n; ++i) result[i] = acc[2 * i] + acc[2 * i + 1];
    }
}

// Tiny-eta Taylor/Horner expansion of the state u(t) for u'=-L u+b1.
static void taylor_solution(const Session& S, const double* b0, const double* b1,
                            double* y) {
    const uint32_t n = S.n;
    const int K = 6;
    std::vector<double> acc(n), vk(n), tmp(n);
    for (int k = K; k >= 0; --k) {
        double f = 1.0;
        for (int j = 2; j <= k; ++j) f *= j;
        double w0 = 1.0 / f, w1 = S.t / (f * (k + 1));
        for (uint32_t i = 0; i < n; ++i) vk[i] = w0 * b0[i] + w1 * b1[i];
        if (k == K) {
            std::memcpy(acc.data(), vk.data(), n * sizeof(double));
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p)
                    s += S.val[p] * acc[S.idx[p]];
                tmp[i] = -S.t * s;
            }
            for (uint32_t i = 0; i < n; ++i) acc[i] = vk[i] + tmp[i];
        }
    }
    std::memcpy(y, acc.data(), n * sizeof(double));
}

int main() {
    if (!write_exact("READY\n", 6)) return 1;
    char magic[8];
    if (!read_exact(magic, 8)) return 1;
    if (std::memcmp(magic, "FAMEXP2\n", 8) != 0) return 1;
    uint32_t hlen = 0;
    if (!read_exact(&hlen, 4)) return 1;
    std::string js(hlen, '\0');
    if (hlen && !read_exact(&js[0], hlen)) return 1;

    Session S;
    S.n = static_cast<uint32_t>(json_number(js, "n"));
    S.nnz = static_cast<uint32_t>(json_number(js, "nnz"));
    S.requests = static_cast<uint32_t>(json_number(js, "requests"));
    S.t = json_number(js, "t");
    S.rtol = json_number(js, "rtol");
    S.atol = json_number(js, "atol");
    if (const char* env = std::getenv("SOLVER_STOP")) S.stop_c = std::strtod(env, nullptr);
    if (S.n == 0 || S.requests == 0) return 1;

    std::vector<uint64_t> rp(static_cast<size_t>(S.n) + 1);
    if (!read_exact(rp.data(), 8ULL * (S.n + 1))) return 1;
    S.ptr.resize(static_cast<size_t>(S.n) + 1);
    for (size_t i = 0; i <= S.n; ++i) S.ptr[i] = static_cast<uint32_t>(rp[i]);
    S.idx.resize(S.nnz);
    S.val.resize(S.nnz);
    S.absval.resize(S.nnz);
    if (S.nnz) {
        if (!read_exact(S.idx.data(), 4ULL * S.nnz)) return 1;
        if (!read_exact(S.val.data(), 8ULL * S.nnz)) return 1;
    }
    double rowmax = 0.0;
    for (uint32_t i = 0; i < S.n; ++i) {
        double r = 0.0;
        for (uint32_t p = S.ptr[i]; p < S.ptr[i + 1]; ++p) {
            S.absval[p] = std::fabs(S.val[p]);
            r += S.absval[p];
        }
        rowmax = std::max(rowmax, r);
    }
    // For small t*||L|| a handful of Chebyshev steps is cheaper than either
    // component detection or a symmetric-matrix conversion.
    double eta0 = S.t * rowmax;
    bool use_exact = false;
    if (S.t > 0.0 && S.nnz > 0 && eta0 >= 20.0) use_exact = build_components(S);
    bool have_matrix = false;
    if (!use_exact && S.t > 0.0) {
        have_matrix = compute_spectral_bound(S, rowmax);
        if (have_matrix) {
            prepare_coefficients(S);
            if (!S.use_taylor) {
                // Upper bound on the degree actually used (the adaptive
                // threshold is always >= atol); only pay for the symmetric
                // conversion when the expansion is long enough to benefit.
                double tgt = S.stop_c * S.atol;
                int upper = S.K;
                for (int k = 0; k <= S.K; ++k)
                    if (S.tailBoth[k] <= tgt) { upper = k; break; }
                // Half-storage only pays off when the matrix is too large for
                // cache; on small matrices scattered updates are slower.
                if (upper >= 35 && S.nnz >= 150000 && S.n >= 10000
                    && !std::getenv("SOLVER_NO_HALF")) build_half(S);
            }
        }
    }

    std::vector<double> b0(S.n), b1(S.n), y(S.n), rhs(2ULL * S.n);
    for (uint32_t req = 0; req < S.requests; ++req) {
        if (!read_exact(b0.data(), 8ULL * S.n)) return 1;
        if (!read_exact(b1.data(), 8ULL * S.n)) return 1;
        bool z0 = true, z1 = true;
        for (uint32_t i = 0; i < S.n; ++i) {
            if (b0[i] != 0.0) z0 = false;
            if (b1[i] != 0.0) z1 = false;
        }
        if (S.t == 0.0) {
            std::memcpy(y.data(), b0.data(), 8ULL * S.n);
        } else if (use_exact) {
            exact_solution(S, b0.data(), b1.data(), y.data());
        } else if (!have_matrix) {
            for (uint32_t i = 0; i < S.n; ++i) y[i] = b0[i] + S.t * b1[i];
        } else if (S.use_taylor) {
            taylor_solution(S, b0.data(), b1.data(), y.data());
        } else {
            int nrhs = (z0 ? 0 : 1) + (z1 ? 0 : 1);
            if (nrhs == 0) {
                std::memset(y.data(), 0, 8ULL * S.n);
            } else if (nrhs == 1) {
                const double* rhs1 = z1 ? b0.data() : b1.data();
                const double* c = z1 ? S.cE.data() : S.cH.data();
                const double* tl = z1 ? S.tailE.data() : S.tailH.data();
                forward_chebyshev(S, 1, c, nullptr, tl, rhs1, y.data());
            } else {
                for (uint32_t i = 0; i < S.n; ++i) {
                    rhs[2 * i] = b0[i];
                    rhs[2 * i + 1] = b1[i];
                }
                forward_chebyshev(S, 2, S.cE.data(), S.cH.data(),
                                  S.tailBoth.data(), rhs.data(), y.data());
            }
        }
        if (!write_exact(y.data(), 8ULL * S.n)) return 1;
    }
    return 0;
}
