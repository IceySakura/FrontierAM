#pragma once

#include "tree.hpp"
#include <numeric>

using Matrix = std::vector<float>;

inline float reliability(Arc e) {
    return e.support < 0 ? e.length / (e.length + 0.006f)
                         : std::clamp(1.5f * e.support - 0.5f, 0.0f, 1.0f);
}

struct GeneDistances {
    Matrix lengths, edges, confident;
    float scale = 1, mean_length = 0, mean_confident = 0;
    explicit GeneDistances(const Tree &t)
        : lengths(t.n*t.n), edges(t.n*t.n), confident(t.n*t.n) {
        for (int i = 0; i < t.n; ++i) dfs(t, i, i, -1, 0, 0, 0);
        std::vector<float> data;
        for (int i = 1; i < t.n; ++i) for (int j = 1; j < i; ++j) {
            int k = i*t.n+j;
            data.push_back(lengths[k]);
            mean_length += lengths[k];
            mean_confident += confident[k];
        }
        int mid = data.size() / 2;
        std::nth_element(data.begin(), data.begin() + mid, data.end());
        scale = std::max(0.001f, data[mid]);
        mean_length /= data.size();
        mean_confident /= data.size();
    }
private:
    void dfs(const Tree &t, int source, int u, int p, float d, float e, float c) {
        if (u < t.n) {
            int k = source*t.n+u;
            lengths[k] = d;
            edges[k] = e;
            confident[k] = c;
        }
        for (int i = 0; i < t.v[u].degree; ++i) {
            Arc a = t.v[u].edges[i];
            if (a.to != p)
                dfs(t, source, a.to, u, d+a.length, e+1,
                    c+((u < t.n || a.to < t.n) ? 0.0f : reliability(a)));
        }
    }
};

inline float power_distance(float x, float power) {
    if (power == 1) return x;
    if (power == 0.5f) return std::sqrt(x);
    if (power == 0) return std::log(x);
    return std::pow(x, power);
}

enum class Metric { Length, Topology, Confident, Hybrid };
enum class Join { NJ, BioNJ, UPGMA, WPGMA };

// The pseudocount represents uncertainty from a small number of sites.
inline Matrix aggregate(const std::vector<GeneDistances> &d, Metric m, float power = 1,
                        float normalization = 0.75f, float floor = 0.06f,
                        float trim = 0, const std::vector<float> &locus_weights = {}) {
    int size = d.front().lengths.size();
    int n = std::lround(std::sqrt(size));
    Matrix out(size);
    std::vector<float> w, s, z;
    for (size_t j = 0; j < d.size(); ++j) {
        const auto &g = d[j];
        s.push_back(std::pow(g.scale, normalization));
        w.push_back(locus_weights.empty() ? 1 : locus_weights[j]);
        if (m == Metric::Length || m == Metric::Hybrid) z.push_back(g.scale*floor + 0.0025f);
        else z.push_back(1 + 10*floor);
    }
    std::vector<float> work(d.size());
    int ktrim = std::clamp(static_cast<int>(trim*d.size()), 0, static_cast<int>(d.size())-1);
    float ws = std::accumulate(w.begin(), w.end(), 0.0f);
    for (int i = 0; i < n; ++i) for (int j = i+1; j < n; ++j) {
        int k = i*n+j;
        float sum = 0;
        for (size_t l = 0; l < d.size(); ++l) {
            const auto &g = d[l];
            float x = 0;
            switch (m) {
                case Metric::Length: x = (g.lengths[k] + z[l]) / s[l]; break;
                case Metric::Topology: x = (g.edges[k] + z[l]) / 15; break;
                case Metric::Confident: x = (g.confident[k] + z[l]) / 10; break;
                case Metric::Hybrid:
                    x = 0.5f*(g.lengths[k] + z[l])/s[l] +
                        0.5f*(g.confident[k]+1)*g.scale/(s[l]*(g.mean_confident+1));
                    break;
            }
            float val = power_distance(x, power);
            if (trim == 0) sum += w[l]*val;
            else work[l] = val;
        }
        if (trim != 0) {
            std::nth_element(work.begin(), work.begin()+ktrim, work.end());
            float threshold = work[ktrim];
            for (float val : work) sum += std::min(threshold, val);
        }
        out[i*n+j] = out[j*n+i] = sum / ws;
    }
    return out;
}

inline Tree join_tree(Matrix d, int n, Join method) {
    Tree t(n);
    Matrix var = d;
    std::vector<float> row(n), height(n);
    std::vector<int> id(n), count(n, 1), active(n);
    std::iota(id.begin(), id.end(), 0);
    std::iota(active.begin(), active.end(), 0);
    bool upgma = method == Join::UPGMA || method == Join::WPGMA;
    int last = upgma ? 2 : 3;
    for (int m = n; m > last; --m) {
        if (!upgma) {
            for (int a : active) {
                float s = 0;
                for (int b : active) s += d[a*n+b];
                row[a] = s / (m - 2);
            }
        }
        float best = std::numeric_limits<float>::infinity();
        int ai = 0, bi = 1;
        for (int i = 0; i < m; ++i) for (int j = i+1; j < m; ++j) {
            int a = active[i], b = active[j];
            float q = d[a*n+b] - (upgma ? 0 : row[a]+row[b]);
            if (q < best) { best = q; ai = i; bi = j; }
        }
        int a = active[ai], b = active[bi];
        float dab = d[a*n+b], h = 0.5f*dab;
        float la = upgma ? h-height[a] : 0.5f*(dab+row[a]-row[b]);
        float lb = upgma ? h-height[b] : dab-la;
        float lambda = 0.5f;
        if (method == Join::BioNJ && var[a*n+b] > 1e-8f) {
            float s = 0;
            for (int c : active) if (c != a && c != b) s += var[b*n+c]-var[a*n+c];
            lambda = std::clamp(0.5f + s/(2*(m-2)*var[a*n+b]), 0.0f, 1.0f);
        } else if (method == Join::UPGMA) {
            lambda = float(count[a]) / (count[a]+count[b]);
        }
        int u = t.node();
        t.link(u, id[a], std::max(0.0f, la));
        t.link(u, id[b], std::max(0.0f, lb));
        id[a] = u;
        active[bi] = active.back();
        active.pop_back();
        float oldvar = var[a*n+b];
        for (int c : active) if (c != a) {
            float x = upgma ? lambda*d[a*n+c]+(1-lambda)*d[b*n+c]
                            : lambda*(d[a*n+c]-la)+(1-lambda)*(d[b*n+c]-lb);
            d[a*n+c] = d[c*n+a] = x;
            if (method == Join::BioNJ)
                var[a*n+c] = var[c*n+a] = lambda*var[a*n+c] + (1-lambda)*var[b*n+c]
                                           - lambda*(1-lambda)*oldvar;
        }
        height[a] = h;
        count[a] += count[b];
    }
    if (upgma) {
        t.link(id[active[0]], id[active[1]]);
    } else {
        int u = t.node();
        for (int i = 0; i < 3; ++i) {
            int a = active[i], b = active[(i+1)%3], c = active[(i+2)%3];
            t.link(u, id[a], std::max(0.0f, 0.5f*(d[a*n+b]+d[a*n+c]-d[b*n+c])));
        }
    }
    return t;
}
