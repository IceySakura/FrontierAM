#pragma once
#include "tree.hpp"

namespace st {

inline double edge_reliability(double length, double support) {
    if (support < 0) return 1.0 - std::exp(-160.0 * length);
    return std::clamp((support - 1.0/3) * 1.5, 0.0, 1.0);
}

enum class EdgeMode { Length, Topology, Support, Soft, ShortLength, ReliableLength, InternalLength };

inline Matrix gene_distances(const Tree& gene, EdgeMode mode, double scale = 0.02) {
    Matrix d(gene.n);
    std::vector<std::array<double,3>> cost(gene.v.size());
    for (int u = 0; u < int(gene.v.size()); ++u) {
        for (int j = 0; j < gene.v[u].degree; ++j) {
            Edge e = gene.v[u].edges[j];
            double l = e.length;
            bool internal = u >= gene.n && e.to >= gene.n;
            if (mode != EdgeMode::Length) {
                double r = internal ? edge_reliability(l, e.support) : 1.0;
                switch (mode) {
                case EdgeMode::Topology: l = 1; break;
                case EdgeMode::Support: l = r; break;
                case EdgeMode::Soft: l = 0.04 + 0.96*r; break;
                case EdgeMode::ShortLength: l = std::log1p(l / scale); break;
                case EdgeMode::ReliableLength: if (internal) l *= std::sqrt(r); break;
                case EdgeMode::InternalLength: if (!internal) l = 0; break;
                default: break;
                }
            }
            cost[u][j] = l;
        }
    }
    for (int i = 0; i < gene.n; ++i) {
        auto walk = [&](auto& self, int u, int parent, double depth) -> void {
            if (u < gene.n) d(i,u) = depth;
            for (int j = 0; j < gene.v[u].degree; ++j) {
                Edge e = gene.v[u].edges[j];
                if (e.to == parent) continue;
                self(self, e.to, u, depth + cost[u][j]);
            }
        };
        walk(walk, i, -1, 0);
    }
    return d;
}

inline double quantile_sorted(const std::vector<double>& a, double p) {
    double x = std::clamp(p, 0.0, 1.0) * (a.size() - 1);
    int i = int(x);
    return a[i] + (a[std::min<int>(i + 1, a.size() - 1)] - a[i]) * (x - i);
}

inline double median(std::vector<double> a) {
    auto m = a.begin() + a.size()/2;
    std::nth_element(a.begin(), m, a.end());
    return a.size()%2 ? *m : (*m + *std::max_element(a.begin(), m))/2;
}

inline double pair_scale(const Matrix& m, int out) {
    std::vector<double> a;
    a.reserve(size_t(m.n) * m.n / 2);
    for (int i = 0; i < m.n; ++i)
        for (int j = 0; j < i; ++j)
            if (i != out && j != out) a.push_back(m(i,j));
    return std::max(1e-9, median(std::move(a)));
}

struct DistConfig {
    EdgeMode mode = EdgeMode::Length;
    double power = 1;
    double first = 0;
    double last = 1;
    double shift = 0.12;
    double norm = 0.85;
    double edge_scale = 0.02;
    double laplace = 0.;
};

inline Matrix aggregate_distances(const std::vector<Matrix>& mats, const std::vector<double>& scales,
                                  const DistConfig& c, const std::vector<double>* weights = nullptr) {
    int n = mats.front().n;
    int k = int(mats.size());
    Matrix d(n);
    std::vector<double> norm(k);
    double base = median(scales);
    for (int g = 0; g < k; ++g) norm[g] = 1 / std::pow(scales[g] / base, c.norm) / base;
    std::vector<std::pair<double,double>> p(k);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double value = 0, wsum = 0;
            for (int g = 0; g < k; ++g) {
                double x = mats[g](i,j) * norm[g];
                if (c.power == 0) x = std::log(x + c.shift);
                else if (c.power != 1) x = (std::pow(x + c.shift, c.power) - 1) / c.power;
                p[g] = {x, weights ? (*weights)[g] : 1.0};
            }
            if (c.first > 0 || c.last < 1) std::sort(p.begin(), p.end());
            double offset = 0;
            if (c.laplace > 0) {
                offset = p[0].first;
                if (c.first == 0 && c.last == 1) {
                    for (int g = 1; g < k; ++g) offset = std::min(offset,p[g].first);
                }
            }
            for (int g = 0; g < k; ++g) {
                double f = std::min(double(g + 1), c.last*k) - std::max(double(g), c.first*k);
                if (f > 0) {
                    double w = f * p[g].second;
                    value += w * (c.laplace > 0 ? std::exp(-c.laplace*(p[g].first-offset)) : p[g].first);
                    wsum += w;
                }
            }
            double x = value / wsum;
            d(i,j) = d(j,i) = c.laplace > 0 ? offset - std::log(std::max(1e-100,x)) / c.laplace : x;
        }
    }
    return d;
}

inline Tree join(Matrix d, int method = 0, double blend = 0) {
    // 0: neighbor joining, 1: BioNJ, 2: UPGMA, 3: WPGMA.
    int n = d.n;
    Tree t(n);
    std::vector<int> active(n), owner(n), size(n, 1);
    std::iota(active.begin(), active.end(), 0);
    std::iota(owner.begin(), owner.end(), 0);
    std::vector<double> row(n);
    Matrix var = d;
    while (active.size() > 3) {
        int count = int(active.size());
        if (method < 2) {
            for (int i : active) {
                row[i] = 0;
                for (int j : active) row[i] += d(i,j);
                row[i] /= count - 2;
            }
        }
        double best = 1e300;
        int bi = -1, bj = -1;
        for (int ii = 0; ii < count; ++ii) {
            int i = active[ii];
            for (int jj = 0; jj < ii; ++jj) {
                int j = active[jj];
                double score = d(i,j);
                if (method < 2) score -= (1-blend)*(row[i] + row[j]);
                if (score < best) best = score, bi = i, bj = j;
            }
        }
        int u = t.add();
        t.link(u, owner[bi]);
        t.link(u, owner[bj]);
        double lambda = 0.5;
        if (method == 1 && var(bi,bj) > 1e-10) {
            double z = 0;
            for (int i : active)
                if (i != bi && i != bj) z += var(bj,i) - var(bi,i);
            lambda = std::clamp(0.5 + z / (2*(count-2)*var(bi,bj)), 0.0, 1.0);
        } else if (method == 2) lambda = double(size[bi]) / (size[bj] + size[bi]);
        double div = d(bi,bj);
        for (int i : active) {
            if (i == bi || i == bj) continue;
            double x = lambda*d(i,bi) + (1-lambda)*d(i,bj);
            if (method < 2) x -= 0.5*div + (lambda-0.5)*(row[bi] - row[bj]);
            d(i,bi) = d(bi,i) = x;
            if (method == 1) {
                var(i,bi) = var(bi,i) = std::max(1e-10, lambda*var(i,bi) + (1-lambda)*var(i,bj)
                                                               - lambda*(1-lambda)*var(bi,bj));
            }
        }
        owner[bi] = u;
        size[bi] += size[bj];
        active.erase(std::find(active.begin(), active.end(), bj));
    }
    int u = t.add();
    for (int i : active) t.link(u, owner[i]);
    return t;
}

}  // namespace st
