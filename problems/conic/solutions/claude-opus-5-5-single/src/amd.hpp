// amd.hpp - Approximate Minimum Degree (AMD) fill-reducing ordering for
// sparse symmetric matrices. Header-only, C++17.
//
// The algorithm is a port of the approximate minimum degree ordering of
// T. A. Davis' CSparse (cs_amd): quotient graph with element absorption,
// mass elimination, supervariable detection via hashing, aggressive
// absorption, dense row removal, and a final (non-recursive) postorder of
// the assembly tree. Adapted to build C = pattern(A + A^T) without the
// diagonal from an upper, lower or full CSC pattern (duplicates allowed),
// with int64 column pointers / int32 row indices.
//
// Reference: P. R. Amestoy, T. A. Davis, I. S. Duff, "An approximate minimum
// degree ordering algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996;
// T. A. Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace amd {

namespace detail {

inline int64_t flip(int64_t i) { return -i - 2; }

// Ensure w[0..n-1] < mark on return (w uses 64-bit marks, so a reset is
// essentially never needed after the first call).
inline int64_t wclear(int64_t mark, int64_t lemax, int64_t* w, int64_t n) {
    if (mark < 2 || mark >= (std::numeric_limits<int64_t>::max() >> 2) - lemax) {
        for (int64_t k = 0; k < n; ++k)
            if (w[k] != 0) w[k] = 1;
        mark = 2;
    }
    return mark;
}

// Non-recursive depth-first search / postorder of the tree rooted at j.
inline int64_t tdfs(int64_t j, int64_t k, int32_t* head, const int32_t* next, int32_t* post,
                    int32_t* stack) {
    int64_t top = 0;
    stack[0] = (int32_t)j;
    while (top >= 0) {
        int64_t p = stack[top];
        int64_t i = head[p];
        if (i == -1) {
            --top;
            post[k++] = (int32_t)p;
        } else {
            head[p] = next[i];
            stack[++top] = (int32_t)i;
        }
    }
    return k;
}

}  // namespace detail

// n: matrix dimension. Ap (size n+1), Ai: CSC sparsity pattern of a symmetric
// matrix; upper triangle, lower triangle, or both may be given; duplicates
// and diagonal entries are ignored (as are out-of-range row indices).
// On return perm has size n and perm[k] = i means original row/column i is
// the k-th pivot.
// dense_ratio: rows/cols with degree > max(16, dense_ratio*sqrt(n)) are
// treated as dense and ordered last. A negative value disables dense-row
// detection.
inline void order(int32_t n, const int64_t* Ap, const int32_t* Ai, std::vector<int32_t>& perm,
                  double dense_ratio = 10.0) {
    using detail::flip;
    perm.clear();
    if (n <= 0) return;
    if (n == 1) {
        perm.push_back(0);
        return;
    }
    const int64_t N = n;

    // ---------------------------------------------------------------------
    // Build C = pattern(A + A^T) without diagonal and without duplicates.
    // ---------------------------------------------------------------------
    std::vector<int64_t> Cp((size_t)N + 1, 0);
    for (int64_t j = 0; j < N; ++j) {
        const int64_t pe = Ap[j + 1];
        for (int64_t p = Ap[j]; p < pe; ++p) {
            const int64_t i = Ai[p];
            if (i < 0 || i >= N || i == j) continue;
            ++Cp[i];
            ++Cp[j];
        }
    }
    int64_t total = 0;
    for (int64_t j = 0; j < N; ++j) {
        total += Cp[j];
        Cp[j] = total;  // end of column j
    }
    Cp[N] = total;
    // elbow room as in cs_amd: nnz + nnz/5 + 2n (nnz counted with duplicates,
    // so this is >= the requirement for the de-duplicated pattern)
    const int64_t nzmax = total + total / 5 + 2 * N;
    std::vector<int32_t> Civ((size_t)nzmax);
    int32_t* Ci = Civ.data();
    for (int64_t j = 0; j < N; ++j) {
        const int64_t pe = Ap[j + 1];
        for (int64_t p = Ap[j]; p < pe; ++p) {
            const int64_t i = Ai[p];
            if (i < 0 || i >= N || i == j) continue;
            Ci[--Cp[i]] = (int32_t)j;
            Ci[--Cp[j]] = (int32_t)i;
        }
    }
    // now Cp[j] = start of column j. Remove duplicates, compacting in place.
    std::vector<int32_t> len((size_t)N + 1), nv((size_t)N + 1), next((size_t)N + 1),
        head((size_t)N + 1), elen((size_t)N + 1), degree((size_t)N + 1), hhead((size_t)N + 1);
    std::vector<int64_t> wv((size_t)N + 1);
    int64_t* w = wv.data();
    int64_t cnz = 0;
    {
        int32_t* seen = head.data();
        std::fill(head.begin(), head.end(), -1);
        for (int64_t j = 0; j < N; ++j) {
            const int64_t p1 = Cp[j], p2 = Cp[j + 1];
            Cp[j] = cnz;
            for (int64_t p = p1; p < p2; ++p) {
                const int32_t i = Ci[p];
                if (seen[i] != (int32_t)j) {
                    seen[i] = (int32_t)j;
                    Ci[cnz++] = i;
                }
            }
        }
        Cp[N] = cnz;
    }

    // dense threshold
    int64_t dense;
    if (!(dense_ratio >= 0.0)) {
        dense = N;  // disabled
    } else {
        double dd = std::max(16.0, dense_ratio * std::sqrt((double)N));
        dense = (dd >= (double)N) ? N : (int64_t)dd;
    }

    perm.assign((size_t)N + 1, -1);
    int32_t* last = perm.data();  // perm is used as workspace for 'last'

    // ---------------------------------------------------------------------
    // Initialize quotient graph
    // ---------------------------------------------------------------------
    for (int64_t k = 0; k < N; ++k) len[k] = (int32_t)(Cp[k + 1] - Cp[k]);
    len[N] = 0;
    for (int64_t i = 0; i <= N; ++i) {
        head[i] = -1;
        last[i] = -1;
        next[i] = -1;
        hhead[i] = -1;
        nv[i] = 1;
        w[i] = 1;
        elen[i] = 0;
        degree[i] = len[i];
    }
    int64_t lemax = 0, nel = 0, mindeg = 0;
    int64_t mark = detail::wclear(0, 0, w, N);
    elen[N] = -2;  // n is a dead element
    Cp[N] = -1;    // n is a root of assembly tree
    w[N] = 0;      // n is a dead element

    // ---------------------------------------------------------------------
    // Initialize degree lists
    // ---------------------------------------------------------------------
    for (int64_t i = 0; i < N; ++i) {
        const int64_t d = degree[i];
        if (d == 0) {  // node i is empty
            elen[i] = -2;
            ++nel;
            Cp[i] = -1;
            w[i] = 0;
        } else if (d > dense) {  // node i is dense: absorb into element n
            nv[i] = 0;
            elen[i] = -1;
            ++nel;
            Cp[i] = flip(N);
            ++nv[N];
        } else {
            if (head[d] != -1) last[head[d]] = (int32_t)i;
            next[i] = head[d];
            head[d] = (int32_t)i;
        }
    }

    while (nel < N) {
        // --- Select node of minimum approximate degree --------------------
        int64_t k = -1;
        for (; mindeg < N && (k = head[mindeg]) == -1; ++mindeg) {
        }
        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k];  // remove k from degree list
        const int64_t elenk = elen[k];
        int64_t nvk = nv[k];
        nel += nvk;

        // --- Garbage collection -------------------------------------------
        if (elenk > 0 && cnz + mindeg >= nzmax) {
            for (int64_t j = 0; j < N; ++j) {
                const int64_t p = Cp[j];
                if (p >= 0) {  // j is a live node or element
                    Cp[j] = Ci[p];
                    Ci[p] = (int32_t)flip(j);
                }
            }
            int64_t q = 0;
            for (int64_t p = 0; p < cnz;) {
                const int64_t j = flip(Ci[p++]);
                if (j >= 0) {  // found object j
                    Ci[q] = (int32_t)Cp[j];
                    Cp[j] = q++;
                    for (int64_t k3 = 0; k3 < (int64_t)len[j] - 1; ++k3) Ci[q++] = Ci[p++];
                }
            }
            cnz = q;
        }

        // --- Construct new element ----------------------------------------
        int64_t dk = 0;
        nv[k] = (int32_t)(-nvk);  // flag k as in Lk
        int64_t p = Cp[k];
        const int64_t pk1 = (elenk == 0) ? p : cnz;
        int64_t pk2 = pk1;
        for (int64_t k1 = 1; k1 <= elenk + 1; ++k1) {
            int64_t e, pj, ln;
            if (k1 > elenk) {
                e = k;
                pj = p;
                ln = len[k] - elenk;
            } else {
                e = Ci[p++];
                pj = Cp[e];
                ln = len[e];
            }
            for (int64_t k2 = 1; k2 <= ln; ++k2) {
                const int64_t i = Ci[pj++];
                const int64_t nvi = nv[i];
                if (nvi <= 0) continue;  // node i dead, or seen
                dk += nvi;
                nv[i] = (int32_t)(-nvi);
                Ci[pk2++] = (int32_t)i;
                if (next[i] != -1) last[next[i]] = last[i];
                if (last[i] != -1) {
                    next[last[i]] = next[i];
                } else {
                    head[degree[i]] = next[i];
                }
            }
            if (e != k) {
                Cp[e] = flip(k);  // absorb e into k
                w[e] = 0;
            }
        }
        if (elenk != 0) cnz = pk2;
        degree[k] = (int32_t)dk;
        Cp[k] = pk1;
        len[k] = (int32_t)(pk2 - pk1);
        elen[k] = -2;

        // --- Find set differences -----------------------------------------
        mark = detail::wclear(mark, lemax, w, N);
        for (int64_t pk = pk1; pk < pk2; ++pk) {
            const int64_t i = Ci[pk];
            const int64_t eln = elen[i];
            if (eln <= 0) continue;
            const int64_t nvi = -(int64_t)nv[i];
            const int64_t wnvi = mark - nvi;
            const int64_t pe = Cp[i] + eln;
            for (int64_t pp = Cp[i]; pp < pe; ++pp) {
                const int64_t e = Ci[pp];
                if (w[e] >= mark) {
                    w[e] -= nvi;
                } else if (w[e] != 0) {
                    w[e] = degree[e] + wnvi;
                }
            }
        }

        // --- Degree update -------------------------------------------------
        for (int64_t pk = pk1; pk < pk2; ++pk) {
            const int64_t i = Ci[pk];
            const int64_t p1 = Cp[i];
            const int64_t p2 = p1 + elen[i] - 1;
            int64_t pn = p1;
            int64_t h = 0, d = 0;
            for (int64_t pp = p1; pp <= p2; ++pp) {
                const int64_t e = Ci[pp];
                if (w[e] != 0) {
                    const int64_t dext = w[e] - mark;
                    if (dext > 0) {
                        d += dext;
                        Ci[pn++] = (int32_t)e;
                        h += e;
                    } else {
                        Cp[e] = flip(k);  // aggressive absorption
                        w[e] = 0;
                    }
                }
            }
            elen[i] = (int32_t)(pn - p1 + 1);
            const int64_t p3 = pn;
            const int64_t p4 = p1 + len[i];
            for (int64_t pp = p2 + 1; pp < p4; ++pp) {
                const int64_t j = Ci[pp];
                const int64_t nvj = nv[j];
                if (nvj <= 0) continue;
                d += nvj;
                Ci[pn++] = (int32_t)j;
                h += j;
            }
            if (d == 0) {  // mass elimination
                Cp[i] = flip(k);
                const int64_t nvi = -(int64_t)nv[i];
                dk -= nvi;
                nvk += nvi;
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1;
            } else {
                degree[i] = (int32_t)std::min<int64_t>(degree[i], d);
                Ci[pn] = Ci[p3];
                Ci[p3] = Ci[p1];
                Ci[p1] = (int32_t)k;
                len[i] = (int32_t)(pn - p1 + 1);
                h %= N;
                next[i] = hhead[h];
                hhead[h] = (int32_t)i;
                last[i] = (int32_t)h;
            }
        }
        degree[k] = (int32_t)dk;
        lemax = std::max(lemax, dk);
        mark = detail::wclear(mark + lemax, lemax, w, N);

        // --- Supernode detection -------------------------------------------
        for (int64_t pk = pk1; pk < pk2; ++pk) {
            int64_t i = Ci[pk];
            if (nv[i] >= 0) continue;
            const int64_t h = last[i];
            i = hhead[h];
            hhead[h] = -1;
            for (; i != -1 && next[i] != -1; i = next[i], ++mark) {
                const int64_t ln = len[i];
                const int64_t eln = elen[i];
                const int64_t pie = Cp[i] + ln;
                for (int64_t pp = Cp[i] + 1; pp < pie; ++pp) w[Ci[pp]] = mark;
                int64_t jlast = i;
                for (int64_t j = next[i]; j != -1;) {
                    bool ok = (len[j] == ln) && (elen[j] == eln);
                    const int64_t pje = Cp[j] + ln;
                    for (int64_t pp = Cp[j] + 1; ok && pp < pje; ++pp) {
                        if (w[Ci[pp]] != mark) ok = false;
                    }
                    if (ok) {  // i and j are identical: absorb j into i
                        Cp[j] = flip(i);
                        nv[i] += nv[j];
                        nv[j] = 0;
                        elen[j] = -1;
                        j = next[j];
                        next[jlast] = (int32_t)j;
                    } else {
                        jlast = j;
                        j = next[j];
                    }
                }
            }
        }

        // --- Finalize new element -------------------------------------------
        p = pk1;
        for (int64_t pk = pk1; pk < pk2; ++pk) {
            const int64_t i = Ci[pk];
            const int64_t nvi = -(int64_t)nv[i];
            if (nvi <= 0) continue;
            nv[i] = (int32_t)nvi;
            int64_t d = degree[i] + dk - nvi;
            d = std::min(d, N - nel - nvi);
            if (head[d] != -1) last[head[d]] = (int32_t)i;
            next[i] = head[d];
            last[i] = -1;
            head[d] = (int32_t)i;
            mindeg = std::min(mindeg, d);
            degree[i] = (int32_t)d;
            Ci[p++] = (int32_t)i;
        }
        nv[k] = (int32_t)nvk;
        if ((len[k] = (int32_t)(p - pk1)) == 0) {
            Cp[k] = -1;  // k is a root of the tree
            w[k] = 0;
        }
        if (elenk != 0) cnz = p;
    }

    // free the big workspace before postordering
    std::vector<int32_t>().swap(Civ);
    std::vector<int64_t>().swap(wv);
    std::vector<int32_t>().swap(hhead);

    // ---------------------------------------------------------------------
    // Postorder the assembly tree
    // ---------------------------------------------------------------------
    bool ok = true;
    for (int64_t i = 0; i < N; ++i) {
        Cp[i] = flip(Cp[i]);
        // sanity (should never trigger): parents must be valid tree nodes
        if (Cp[i] < -1 || Cp[i] > N || Cp[i] == i || (nv[i] <= 0 && Cp[i] < 0)) ok = false;
    }
    if (ok) {
        for (int64_t j = 0; j <= N; ++j) head[j] = -1;
        for (int64_t j = N; j >= 0; --j) {  // place non-principal nodes in lists
            if (nv[j] > 0) continue;
            next[j] = head[Cp[j]];
            head[Cp[j]] = (int32_t)j;
        }
        for (int64_t e = N; e >= 0; --e) {  // place elements in lists
            if (nv[e] <= 0) continue;
            if (Cp[e] != -1) {
                next[e] = head[Cp[e]];
                head[Cp[e]] = (int32_t)e;
            }
        }
        int32_t* post = perm.data();  // 'last' is no longer needed
        int32_t* stack = degree.data();
        int64_t k = 0;
        for (int64_t i = 0; i <= N; ++i) {
            if (Cp[i] == -1) k = detail::tdfs(i, k, head.data(), next.data(), post, stack);
        }
        ok = (k == N + 1) && (post[N] == N);
    }
    perm.resize((size_t)N);
    if (ok) {  // validate permutation
        std::vector<char> seen((size_t)N, 0);
        for (int64_t k = 0; k < N; ++k) {
            const int32_t i = perm[k];
            if (i < 0 || i >= n || seen[i]) {
                ok = false;
                break;
            }
            seen[i] = 1;
        }
    }
    if (!ok) {  // safety net (should be unreachable): identity ordering
        for (int64_t k = 0; k < N; ++k) perm[k] = (int32_t)k;
    }
}

}  // namespace amd
