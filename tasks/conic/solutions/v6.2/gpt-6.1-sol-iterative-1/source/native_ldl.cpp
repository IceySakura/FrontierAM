// Sparse LDL^T with a fixed symmetric elimination ordering.
#include <algorithm>
#include <cmath>
#include <new>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct AffineForest {
    std::vector<int> parent, rank;
    std::vector<double> weight, shift, value;
    std::vector<unsigned char> fixed;
    explicit AffineForest(int n): parent(n), rank(n, 0), weight(n, 1.0),
        shift(n, 0.0), value(n, 0.0), fixed(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int i) {
        int p = parent[i];
        if (p == i) return i;
        int r = find(p);
        shift[i] += weight[i] * shift[p];
        weight[i] *= weight[p];
        parent[i] = r;
        return r;
    }
};

struct Factor {
    int n, np;
    std::vector<int> ap, ai, perm, inv, lp, parent, count, flag, li, pattern;
    std::vector<double> lx, d, y;
    Factor(int nn, int primal, const int* ptr, const int* idx, const int* order)
        : n(nn), np(primal), ap(ptr, ptr + nn + 1), ai(idx, idx + ptr[nn]),
          perm(order, order + nn), inv(nn), lp(nn + 1), parent(nn),
          count(nn), flag(nn), pattern(nn), d(nn), y(nn) {
        for (int k = 0; k < n; ++k) inv[perm[k]] = k;
        for (int k = 0; k < n; ++k) {
            parent[k] = -1;
            flag[k] = k;
            count[k] = 0;
            int col = perm[k];
            for (int p = ap[col]; p < ap[col + 1]; ++p) {
                int i = inv[ai[p]];
                if (i >= k) continue;
                for (; flag[i] != k; i = parent[i]) {
                    if (parent[i] == -1) parent[i] = k;
                    ++count[i];
                    flag[i] = k;
                }
            }
        }
        lp[0] = 0;
        for (int k = 0; k < n; ++k) {
            if (count[k] > 60000000 - lp[k]) throw std::bad_alloc();
            lp[k + 1] = lp[k] + count[k];
        }
        li.resize(lp[n]);
        lx.resize(lp[n]);
    }

    bool factor(const double* a) {
        std::fill(y.begin(), y.end(), 0.0);
        for (int k = 0; k < n; ++k) {
            int top = n, col = perm[k];
            flag[k] = k;
            count[k] = 0;
            for (int p = ap[col]; p < ap[col + 1]; ++p) {
                int i = inv[ai[p]];
                if (i > k) continue;
                y[i] += a[p];
                int len = 0;
                for (; flag[i] != k; i = parent[i]) {
                    pattern[len++] = i;
                    flag[i] = k;
                }
                while (len) pattern[--top] = pattern[--len];
            }
            double dk = y[k];
            y[k] = 0;
            for (; top < n; ++top) {
                int i = pattern[top];
                double yi = y[i];
                y[i] = 0;
                int end = lp[i] + count[i];
                for (int p = lp[i]; p < end; ++p) y[li[p]] -= lx[p] * yi;
                double l = yi / d[i];
                dk -= l * yi;
                li[end] = k;
                lx[end] = l;
                ++count[i];
            }
            if (!std::isfinite(dk)) return false;
            double sign = perm[k] < np ? 1.0 : -1.0;
            if (sign * dk < 1e-15) dk = sign * 1e-15;
            d[k] = dk;
        }
        return true;
    }

    void solve(const double* rhs, double* output) {
        for (int k = 0; k < n; ++k) y[k] = rhs[perm[k]];
        for (int j = 0; j < n; ++j)
            for (int p = lp[j]; p < lp[j + 1]; ++p) y[li[p]] -= lx[p] * y[j];
        for (int j = 0; j < n; ++j) y[j] /= d[j];
        for (int j = n - 1; j >= 0; --j)
            for (int p = lp[j]; p < lp[j + 1]; ++p) y[j] -= lx[p] * y[li[p]];
        for (int k = 0; k < n; ++k) output[perm[k]] = y[k];
    }
};

extern "C" {
void* ldl_create(int n, int primal, const int* ap, const int* ai, const int* order) {
    try { return new Factor(n, primal, ap, ai, order); }
    catch (...) { return nullptr; }
}
void ldl_free(void* ptr) { delete static_cast<Factor*>(ptr); }
int ldl_numeric(void* ptr, const double* a) {
    return static_cast<Factor*>(ptr)->factor(a) ? 0 : -1;
}
void ldl_solve(void* ptr, const double* b, double* x) {
    static_cast<Factor*>(ptr)->solve(b, x);
}

int affine_analyze(int n, int m, const int* ap, const int* ai, const double* av,
                   const double* b, unsigned char* selected, int* roots,
                   double* weight, double* offset) {
    try {
        AffineForest f(n);
        int eliminated = 0;
        for (int row = 0; row < m; ++row) {
            selected[row] = 0;
            int len = ap[row + 1] - ap[row];
            if (len < 1 || len > 2) continue;
            int p = ap[row], i = ai[p], ri = f.find(i);
            double a = av[p], wi = f.weight[i], di = f.shift[i];
            if (len == 1) {
                if (f.fixed[ri] || std::abs(a * wi) < 1e-12) continue;
                double val = (b[row] - a * di) / (a * wi);
                if (!std::isfinite(val) || std::abs(val) > 1e15) continue;
                f.fixed[ri] = 1;
                f.value[ri] = val;
                selected[row] = 1;
                ++eliminated;
                continue;
            }
            int j = ai[p + 1], rj = f.find(j);
            double c = av[p + 1], wj = f.weight[j], dj = f.shift[j];
            if (ri == rj || (f.fixed[ri] && f.fixed[rj])) continue;
            double aa = a * wi, cc = c * wj, rhs = b[row] - a * di - c * dj;
            if (std::abs(aa) < 1e-12 || std::abs(cc) < 1e-12) continue;
            if (std::abs(aa / cc) > 1e3 || std::abs(cc / aa) > 1e3) continue;
            if (f.rank[ri] > f.rank[rj]) {
                std::swap(ri, rj);
                std::swap(aa, cc);
            }
            double alpha = -cc / aa, beta = rhs / aa;
            if (!std::isfinite(beta) || std::abs(beta) > 1e12) continue;
            if (f.fixed[ri] && !f.fixed[rj]) {
                f.fixed[rj] = 1;
                f.value[rj] = (f.value[ri] - beta) / alpha;
            }
            f.parent[ri] = rj;
            f.weight[ri] = alpha;
            f.shift[ri] = beta;
            if (f.rank[ri] == f.rank[rj]) ++f.rank[rj];
            selected[row] = 1;
            ++eliminated;
        }
        for (int i = 0; i < n; ++i) {
            int r = f.find(i);
            if (f.fixed[r]) {
                roots[i] = -1;
                weight[i] = 0;
                offset[i] = f.shift[i] + f.weight[i] * f.value[r];
            } else {
                roots[i] = r;
                weight[i] = f.weight[i];
                offset[i] = f.shift[i];
            }
            if (!std::isfinite(weight[i]) || !std::isfinite(offset[i])) return -1;
        }
        return eliminated;
    } catch (...) { return -1; }
}

int affine_dual(int n, int m, const int* cp, const int* ci, const double* cv,
                const int* rp, const int* ri, const double* rv,
                const double* gradient, double* z) {
    try {
        std::vector<int> degree(n), queue;
        std::vector<unsigned char> done(m, 0);
        std::vector<double> g(gradient, gradient + n);
        queue.reserve(n);
        for (int i = 0; i < n; ++i) {
            degree[i] = cp[i + 1] - cp[i];
            if (degree[i] == 1) queue.push_back(i);
        }
        int solved = 0;
        for (size_t head = 0; head < queue.size(); ++head) {
            int i = queue[head];
            if (degree[i] != 1) continue;
            int pos = cp[i];
            while (pos < cp[i + 1] && done[ci[pos]]) ++pos;
            if (pos == cp[i + 1]) continue;
            int row = ci[pos];
            double val = -g[i] / cv[pos];
            if (!std::isfinite(val)) return -1;
            z[row] = val;
            done[row] = 1;
            ++solved;
            for (int p = rp[row]; p < rp[row + 1]; ++p) {
                int j = ri[p];
                g[j] += rv[p] * val;
                --degree[j];
                if (degree[j] == 1) queue.push_back(j);
            }
        }
        return solved == m ? 0 : -1;
    } catch (...) { return -1; }
}

int sparse_basis(int n, int m, const int* ap, const int* ai, const double* av,
                 const unsigned char* cheap, int* rowout, int* colout) {
    try {
        std::vector<std::unordered_map<int, double>> rows(m);
        std::vector<std::unordered_set<int>> cols(n);
        std::vector<unsigned char> active(m, 1);
        using Entry = std::pair<int, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        size_t nnz = 0;
        for (int r = 0; r < m; ++r) {
            double mx = 0;
            for (int p = ap[r]; p < ap[r + 1]; ++p) mx = std::max(mx, std::abs(av[p]));
            if (!mx) { active[r] = 0; continue; }
            for (int p = ap[r]; p < ap[r + 1]; ++p) {
                double val = av[p] / mx;
                if (std::abs(val) < 1e-14) continue;
                rows[r][ai[p]] = val;
                cols[ai[p]].insert(r);
                ++nnz;
            }
            queue.emplace(rows[r].size(), r);
        }
        int rank = 0;
        while (!queue.empty()) {
            auto entry = queue.top(); queue.pop();
            int r = entry.second, len = entry.first;
            if (!active[r] || static_cast<int>(rows[r].size()) != len || !len) continue;
            if (len > 512 || (len > 180 && rank >= n - 1000) || nnz > 5000000) break;
            double mx = 0;
            for (const auto& it : rows[r]) mx = std::max(mx, std::abs(it.second));
            int pivot = -1;
            double best = 1e300, value = 0;
            for (const auto& it : rows[r]) {
                if (std::abs(it.second) < 1e-7 * mx) continue;
                double score = std::max<size_t>(1, cols[it.first].size() - 1);
                if (!cheap[it.first]) score *= 50;
                score *= 1.0 + 0.01 * mx / std::abs(it.second);
                if (score < best) { best = score; pivot = it.first; value = it.second; }
            }
            if (pivot < 0) continue;
            rowout[rank] = r;
            colout[rank] = pivot;
            ++rank;
            std::vector<std::pair<int, double>> source(rows[r].begin(), rows[r].end());
            std::vector<int> neighbors(cols[pivot].begin(), cols[pivot].end());
            for (const auto& it : source) cols[it.first].erase(r);
            nnz -= rows[r].size();
            rows[r].clear();
            active[r] = 0;
            for (int other : neighbors) {
                if (other == r || !active[other]) continue;
                auto found = rows[other].find(pivot);
                if (found == rows[other].end()) continue;
                double factor = found->second / value;
                rows[other].erase(found);
                --nnz;
                cols[pivot].erase(other);
                for (const auto& it : source) {
                    int j = it.first;
                    if (j == pivot) continue;
                    auto old = rows[other].find(j);
                    double before = old == rows[other].end() ? 0 : old->second;
                    double change = factor * it.second;
                    double after = before - change;
                    double tol = 1e-12 * std::max(1.0, std::max(std::abs(before), std::abs(change)));
                    if (std::abs(after) <= tol) {
                        if (old != rows[other].end()) {
                            rows[other].erase(old);
                            cols[j].erase(other);
                            --nnz;
                        }
                    } else {
                        if (!std::isfinite(after)) return rank;
                        if (old == rows[other].end()) {
                            rows[other].emplace(j, after);
                            cols[j].insert(other);
                            ++nnz;
                        } else old->second = after;
                    }
                }
                queue.emplace(rows[other].size(), other);
            }
            cols[pivot].clear();
        }
        return rank;
    } catch (...) { return -1; }
}
}
