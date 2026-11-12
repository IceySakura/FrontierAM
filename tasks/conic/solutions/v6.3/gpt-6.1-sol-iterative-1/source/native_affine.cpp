// Weighted affine union-find and dual recovery for sparse equality presolve.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

struct AffineMap {
    int n, nr;
    std::vector<int> parent, rank, vi, vj;
    std::vector<double> weight, shift, fixed_value, ca, cb;
    std::vector<std::uint8_t> fixed, status;

    int find(int x) {
        int p = parent[x];
        if (p != x) {
            int r = find(p);
            shift[x] += weight[x] * shift[p];
            weight[x] *= weight[p];
            parent[x] = r;
        }
        return parent[x];
    }
    static bool close(double a, double b) {
        return std::abs(a - b) <= 1e-12 * (1.0 + std::abs(a) + std::abs(b));
    }
    bool set_fixed(int root, double value) {
        if (!std::isfinite(value) || std::abs(value) > 1e100) return false;
        fixed[root] = 1;
        fixed_value[root] = value;
        return true;
    }
    bool link(int child, int target, double alpha, double beta) {
        if (!std::isfinite(alpha) || !std::isfinite(beta) ||
            std::abs(alpha) < 1e-14 || std::abs(alpha) > 1e14 ||
            std::abs(beta) > 1e100) return false;
        parent[child] = target;
        weight[child] = alpha;
        shift[child] = beta;
        if (rank[child] == rank[target]) ++rank[target];
        return true;
    }

    AffineMap(int size, int rows, const int* ap, const int* ai,
              const double* ax, const double* rhs) : n(size), nr(rows),
        parent(size), rank(size, 0), vi(rows, -1), vj(rows, -1),
        weight(size, 1.), shift(size, 0.), fixed_value(size, 0.),
        ca(rows, 0.), cb(rows, 0.), fixed(size, 0), status(rows, 0) {
        std::iota(parent.begin(), parent.end(), 0);
        for (int row = 0; row < nr; ++row) {
            int len = ap[row + 1] - ap[row];
            if (len < 1 || len > 2) continue;
            int pos = ap[row], i = ai[pos], j = len == 2 ? ai[pos + 1] : -1;
            double a = ax[pos], b = len == 2 ? ax[pos + 1] : 0.;
            vi[row] = i; vj[row] = j; ca[row] = a; cb[row] = b;
            int ri = find(i), rj = j < 0 ? -1 : find(j);
            double A = a * weight[i], B = j < 0 ? 0. : b * weight[j];
            double c = rhs[row] - a * shift[i] - (j < 0 ? 0. : b * shift[j]);
            if (!std::isfinite(A) || !std::isfinite(B) || !std::isfinite(c)) continue;
            if (j < 0 || ri == rj) {
                double coefficient = A + B;
                if (std::abs(coefficient) <= 1e-12 * (std::abs(A) + std::abs(B))) {
                    if (close(c, 0.)) status[row] = 3;
                } else if (fixed[ri]) {
                    if (close(coefficient * fixed_value[ri], c)) status[row] = 3;
                } else if (set_fixed(ri, c / coefficient)) {
                    status[row] = 2;
                }
            } else if (fixed[ri] && fixed[rj]) {
                if (close(A * fixed_value[ri] + B * fixed_value[rj], c)) status[row] = 3;
            } else {
                bool attach_i = fixed[rj] || (!fixed[ri] && rank[ri] <= rank[rj]);
                bool ok = attach_i ? link(ri, rj, -B / A, c / A)
                                   : link(rj, ri, -A / B, c / B);
                if (ok) status[row] = 1;
            }
        }
    }

    int map(int* roots, double* coefficients, double* offsets, std::uint8_t* removed) {
        for (int i = 0; i < n; ++i) {
            int root = find(i);
            roots[i] = fixed[root] ? -1 : root;
            coefficients[i] = fixed[root] ? 0. : weight[i];
            offsets[i] = shift[i] + (fixed[root] ? weight[i] * fixed_value[root] : 0.);
            if (!std::isfinite(coefficients[i]) || !std::isfinite(offsets[i])) return 1;
        }
        std::copy(status.begin(), status.end(), removed);
        return 0;
    }

    int dual(const double* gradient, double* multipliers) {
        std::vector<double> g(gradient, gradient + n), h(n, 0.), l0(nr, 0.), l1(nr, 0.);
        std::vector<int> component(n), head(n, -1), degree(n, 0), anchor(n, -1);
        std::vector<int> next, edge_row, neighbor, queue;
        std::vector<std::uint8_t> alive(nr, 0);
        next.reserve(2 * nr); edge_row.reserve(2 * nr); neighbor.reserve(2 * nr);
        for (int i = 0; i < n; ++i) component[i] = find(i);
        auto add_edge = [&](int from, int to, int row) {
            next.push_back(head[from]);
            edge_row.push_back(row);
            neighbor.push_back(to);
            head[from] = static_cast<int>(next.size()) - 1;
            ++degree[from];
        };
        for (int row = 0; row < nr; ++row) {
            if (status[row] == 1) {
                add_edge(vi[row], vj[row], row);
                add_edge(vj[row], vi[row], row);
                alive[row] = 1;
            } else if (status[row] == 2) {
                anchor[component[vi[row]]] = row;
                h[vi[row]] += ca[row];
                if (vj[row] >= 0) h[vj[row]] += cb[row];
            }
        }
        for (int i = 0; i < n; ++i)
            if (degree[i] <= 1 && component[i] != i) queue.push_back(i);
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            int node = queue[cursor], pos = head[node];
            while (pos >= 0 && !alive[edge_row[pos]]) pos = next[pos];
            if (pos < 0) continue;
            int row = edge_row[pos], other = neighbor[pos];
            double a = node == vi[row] ? ca[row] : cb[row];
            double b = node == vi[row] ? cb[row] : ca[row];
            l0[row] = -g[node] / a;
            l1[row] = -h[node] / a;
            g[other] += b * l0[row];
            h[other] += b * l1[row];
            alive[row] = 0;
            --degree[node];
            if (--degree[other] == 1 && component[other] != other) queue.push_back(other);
        }
        std::vector<double> parameter(n, 0.);
        for (int i = 0; i < n; ++i) {
            if (component[i] != i || anchor[i] < 0) continue;
            if (h[i] == 0. || !std::isfinite(h[i])) return 1;
            parameter[i] = -g[i] / h[i];
            l0[anchor[i]] = parameter[i];
        }
        for (int row = 0; row < nr; ++row) {
            multipliers[row] = l0[row] + (status[row] == 1 ? l1[row] * parameter[component[vi[row]]] : 0.);
            if (!std::isfinite(multipliers[row])) return 1;
        }
        return 0;
    }
};

extern "C" {
void* affine_create(int n, int nr, const int* ap, const int* ai,
                    const double* ax, const double* b) {
    try { return new AffineMap(n, nr, ap, ai, ax, b); } catch (...) { return nullptr; }
}
void affine_destroy(void* handle) { delete static_cast<AffineMap*>(handle); }
int affine_map(void* handle, int* roots, double* coefficients, double* offsets,
               std::uint8_t* status) {
    return static_cast<AffineMap*>(handle)->map(roots, coefficients, offsets, status);
}
int affine_dual(void* handle, const double* gradient, double* multipliers) {
    try { return static_cast<AffineMap*>(handle)->dual(gradient, multipliers); }
    catch (...) { return 1; }
}
}
