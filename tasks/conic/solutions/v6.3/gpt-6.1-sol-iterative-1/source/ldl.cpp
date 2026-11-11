#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

// Sparse left-looking LDL for symmetrically permuted quasi-definite systems.
struct Factor {
    int n;
    int clips = 0;
    std::vector<int> order, inverse, parent, capacity, used, pointer, rows;
    std::vector<int> marks, pattern;
    std::vector<double> values, diagonal, work, solvework;

    Factor(int size, const int* permutation, const int* ap, const int* ai)
        : n(size), order(permutation, permutation + size), inverse(size),
          parent(size, -1), capacity(size, 0), used(size, 0),
          pointer(size + 1, 0), marks(size, -1), pattern(size),
          diagonal(size), work(size, 0.0), solvework(size) {
        for (int k = 0; k < n; ++k) inverse[order[k]] = k;
        for (int k = 0; k < n; ++k) {
            marks[k] = k;
            int old = order[k];
            for (int p = ap[old]; p < ap[old + 1]; ++p) {
                int i = inverse[ai[p]];
                if (i >= k) continue;
                while (marks[i] != k) {
                    if (parent[i] < 0) parent[i] = k;
                    ++capacity[i];
                    marks[i] = k;
                    i = parent[i];
                }
            }
        }
        int64_t total = 0;
        for (int j = 0; j < n; ++j) {
            total += capacity[j];
            if (total > 80000000) throw std::bad_alloc();
            pointer[j + 1] = static_cast<int>(total);
        }
        rows.resize(static_cast<size_t>(total));
        values.resize(static_cast<size_t>(total));
    }

    int factor(const int* ap, const int* ai, const double* ax, int positive,
               double threshold, double replacement) {
        std::fill(used.begin(), used.end(), 0);
        std::fill(marks.begin(), marks.end(), -1);
        std::fill(work.begin(), work.end(), 0.0);
        clips = 0;
        for (int k = 0; k < n; ++k) {
            int top = n;
            marks[k] = k;
            int old = order[k];
            for (int p = ap[old]; p < ap[old + 1]; ++p) {
                int i = inverse[ai[p]];
                if (i > k) continue;
                work[i] += ax[p];
                int length = 0;
                while (marks[i] != k) {
                    pattern[length++] = i;
                    marks[i] = k;
                    i = parent[i];
                    if (i < 0 || i >= n) return -2;
                }
                while (length > 0) pattern[--top] = pattern[--length];
            }
            double dk = work[k];
            work[k] = 0.0;
            for (; top < n; ++top) {
                int i = pattern[top];
                double yi = work[i];
                work[i] = 0.0;
                int end = pointer[i] + used[i];
                for (int p = pointer[i]; p < end; ++p)
                    work[rows[p]] -= values[p] * yi;
                double lki = yi / diagonal[i];
                dk -= lki * yi;
                if (used[i] >= capacity[i]) return -3;
                rows[end] = k;
                values[end] = lki;
                ++used[i];
            }
            double sign = old < positive ? 1.0 : -1.0;
            if (!std::isfinite(dk)) return -4;
            if (sign * dk < threshold) {
                dk = sign * replacement;
                ++clips;
            }
            diagonal[k] = dk;
        }
        return clips;
    }

    int solve(const double* rhs, double* out, int count) {
        std::vector<double>& y = solvework;
        for (int c = 0; c < count; ++c) {
            for (int j = 0; j < n; ++j) y[j] = rhs[order[j] * count + c];
            for (int j = 0; j < n; ++j) {
                double yj = y[j];
                int end = pointer[j] + used[j];
                for (int p = pointer[j]; p < end; ++p)
                    y[rows[p]] -= values[p] * yj;
            }
            for (int j = 0; j < n; ++j) y[j] /= diagonal[j];
            for (int j = n - 1; j >= 0; --j) {
                double yj = y[j];
                int end = pointer[j] + used[j];
                for (int p = pointer[j]; p < end; ++p)
                    yj -= values[p] * y[rows[p]];
                y[j] = yj;
            }
            for (int j = 0; j < n; ++j) {
                if (!std::isfinite(y[j])) return -1;
                out[order[j] * count + c] = y[j];
            }
        }
        return 0;
    }
};

struct Assembly {
    int n, m, nb, total;
    int64_t nnz;
    std::vector<int> ptr, row, diagonal, below, above;
    std::vector<double> initial;

    Assembly(int nx, int nz, const int* pp, const int* pi, const double* px,
             const int* bp, const int* bi, const double* cone_diagonal)
        : n(nx), m(nz), nb(bp[nx]), total(nx + nz), ptr(total + 1, 0),
          diagonal(total), below(nb), above(nb) {
        std::vector<int> rowcount(m, 0);
        std::vector<char> have_diag(n, 0);
        for (int j = 0; j < n; ++j) {
            for (int p = pp[j]; p < pp[j + 1]; ++p)
                if (pi[p] == j) have_diag[j] = 1;
            for (int p = bp[j]; p < bp[j + 1]; ++p) ++rowcount[bi[p]];
        }
        int64_t count = 0;
        for (int j = 0; j < n; ++j) {
            count += pp[j + 1] - pp[j] + bp[j + 1] - bp[j] + !have_diag[j];
            if (count > 180000000) throw std::bad_alloc();
            ptr[j + 1] = static_cast<int>(count);
        }
        for (int i = 0; i < m; ++i) {
            count += rowcount[i] + 1;
            if (count > 180000000) throw std::bad_alloc();
            ptr[n + i + 1] = static_cast<int>(count);
        }
        nnz = count;
        row.resize(static_cast<size_t>(count));
        initial.resize(static_cast<size_t>(count), 0.0);
        std::vector<int> cursor(m);
        for (int i = 0; i < m; ++i) cursor[i] = ptr[n + i];
        for (int j = 0; j < n; ++j) {
            int dest = ptr[j];
            bool inserted = false;
            for (int p = pp[j]; p < pp[j + 1]; ++p) {
                if (!inserted && pi[p] > j) {
                    diagonal[j] = dest;
                    row[dest++] = j;
                    inserted = true;
                }
                if (pi[p] == j) {
                    diagonal[j] = dest;
                    inserted = true;
                }
                row[dest] = pi[p];
                initial[dest++] = px[p];
            }
            if (!inserted) {
                diagonal[j] = dest;
                row[dest++] = j;
            }
            for (int p = bp[j]; p < bp[j + 1]; ++p) {
                int i = bi[p];
                below[p] = dest;
                row[dest++] = n + i;
                int position = cursor[i]++;
                above[p] = position;
                row[position] = j;
            }
        }
        for (int i = 0; i < m; ++i) {
            int position = ptr[n + i + 1] - 1;
            diagonal[n + i] = position;
            row[position] = n + i;
            initial[position] = -cone_diagonal[i];
        }
    }

    void export_arrays(int* op, int* oi, double* ox) {
        std::memcpy(op, ptr.data(), ptr.size() * sizeof(int));
        std::memcpy(oi, row.data(), row.size() * sizeof(int));
        std::memcpy(ox, initial.data(), initial.size() * sizeof(double));
        std::vector<int>().swap(ptr);
        std::vector<int>().swap(row);
        std::vector<double>().swap(initial);
    }

    void update(const double* bx, double* output) {
        for (int p = 0; p < nb; ++p)
            output[below[p]] = output[above[p]] = bx[p];
    }

    void shift(double* output, double delta) {
        for (int j = 0; j < total; ++j)
            output[diagonal[j]] += j < n ? delta : -delta;
    }
};

extern "C" {
void* conic_ldl_create(int n, const int* permutation, const int* ap, const int* ai) {
    try {
        return new Factor(n, permutation, ap, ai);
    } catch (...) {
        return nullptr;
    }
}
void conic_ldl_free(void* factor) {
    delete static_cast<Factor*>(factor);
}
int conic_ldl_factor(void* factor, const int* ap, const int* ai, const double* ax,
                     int positive, double threshold, double replacement) {
    try {
        return static_cast<Factor*>(factor)->factor(ap, ai, ax, positive, threshold, replacement);
    } catch (...) {
        return -1;
    }
}
int conic_ldl_solve(void* factor, const double* rhs, double* out, int count) {
    try {
        return static_cast<Factor*>(factor)->solve(rhs, out, count);
    } catch (...) {
        return -1;
    }
}
int64_t conic_ldl_nnz(void* factor) {
    return static_cast<Factor*>(factor)->pointer.back();
}
void* conic_kkt_create(int n, int m, const int* pp, const int* pi, const double* px,
                       const int* bp, const int* bi, const double* ci) {
    try {
        return new Assembly(n, m, pp, pi, px, bp, bi, ci);
    } catch (...) {
        return nullptr;
    }
}
void conic_kkt_free(void* assembly) {
    delete static_cast<Assembly*>(assembly);
}
int64_t conic_kkt_nnz(void* assembly) {
    return static_cast<Assembly*>(assembly)->nnz;
}
void conic_kkt_export(void* assembly, int* op, int* oi, double* ox) {
    static_cast<Assembly*>(assembly)->export_arrays(op, oi, ox);
}
void conic_kkt_update(void* assembly, const double* bx, double* output) {
    static_cast<Assembly*>(assembly)->update(bx, output);
}
void conic_kkt_shift(void* assembly, double* output, double delta) {
    static_cast<Assembly*>(assembly)->shift(output, delta);
}
int conic_state(int n, int m, const int* pp, const int* pi, const double* pv,
                const int* ap, const int* ai, const double* av,
                const double* q, const double* b, const double* D, const double* E,
                double cost, const double* x, const double* s, const double* z, double tau,
                double anorm, double pnorm, double qnorm, double bnorm,
                double* px, double* ax, double* rp, double* rd, double* out) {
    if (!(tau > 0.0) || !(cost > 0.0)) return -1;
    std::fill(px, px + n, 0.0);
    std::fill(ax, ax + m, 0.0);
    for (int j = 0; j < n; ++j)
        for (int p = pp[j]; p < pp[j + 1]; ++p)
            px[pi[p]] += pv[p] * x[j];
    double qx = 0.0, xp = 0.0, bz = 0.0, sz = 0.0;
    double nx = 0.0, nz = 0.0, npx = 0.0, natz = 0.0;
    double nax = 0.0, ns = 0.0, nrp = 0.0, nrd = 0.0, nas = 0.0;
    double ct = cost * tau;
    for (int j = 0; j < n; ++j) {
        double atz = 0.0;
        double xj = x[j];
        for (int p = ap[j]; p < ap[j + 1]; ++p) {
            int i = ai[p];
            ax[i] += av[p] * xj;
            atz += av[p] * z[i];
        }
        rd[j] = px[j] + q[j] * tau + atz;
        if (!std::isfinite(rd[j]) || !std::isfinite(atz)) return -1;
        double scale = ct * D[j];
        npx = std::max(npx, std::abs(px[j] / scale));
        natz = std::max(natz, std::abs(atz / scale));
        nrd = std::max(nrd, std::abs(rd[j] / scale));
        nx = std::max(nx, std::abs(D[j] * xj / tau));
        qx += q[j] * xj;
        xp += px[j] * xj;
    }
    for (int i = 0; i < m; ++i) {
        double scale = E[i] * tau;
        rp[i] = ax[i] + s[i] - b[i] * tau;
        if (!std::isfinite(rp[i]) || !std::isfinite(ax[i])) return -1;
        nax = std::max(nax, std::abs(ax[i] / scale));
        ns = std::max(ns, std::abs(s[i] / scale));
        nrp = std::max(nrp, std::abs(rp[i] / scale));
        nas = std::max(nas, std::abs((ax[i] + s[i]) / scale));
        nz = std::max(nz, std::abs(E[i] * z[i] / ct));
        bz += b[i] * z[i];
        sz += s[i] * z[i];
    }
    double primal = .5 * xp / (ct * tau) + qx / ct;
    double dual = -.5 * xp / (ct * tau) - bz / ct;
    out[0] = nrp / (1.0 + std::max(bnorm, std::max(nax, ns)));
    out[1] = nrd / (1.0 + std::max(qnorm, std::max(npx, natz)));
    out[2] = std::abs(primal - dual) / std::max(1.0, std::max(std::abs(primal), std::abs(dual)));
    out[3] = xp;
    out[4] = qx;
    out[5] = bz;
    out[6] = sz;
    out[7] = nz > 0.0 && bnorm > 0.0 ? -bz / ct / (bnorm * nz) : 0.0;
    out[8] = nx > 0.0 && qnorm > 0.0 ? -qx / ct / (qnorm * nx) : 0.0;
    out[9] = nz > 0.0 ? natz / (anorm * nz) : INFINITY;
    out[10] = nx > 0.0 ? npx / (pnorm * nx) : INFINITY;
    out[11] = nx > 0.0 ? nas / (anorm * nx) : INFINITY;
    for (int j = 0; j < 7; ++j)
        if (!std::isfinite(out[j])) return -1;
    return 0;
}
}
