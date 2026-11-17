#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Edge {
    int to;
    double length;
    double support;
};

struct Tree {
    int n = 0;
    std::vector<std::vector<Edge>> adj;
    std::vector<std::string> names;

    void connect(int a, int b, double len = 0, double sup = -1) {
        adj[a].push_back({b, len, sup});
        adj[b].push_back({a, len, sup});
    }

    void disconnect(int a, int b) {
        auto &x = adj[a];
        auto &y = adj[b];
        x.erase(std::remove_if(x.begin(), x.end(),
                              [b](const Edge &e) { return e.to == b; }), x.end());
        y.erase(std::remove_if(y.begin(), y.end(),
                              [a](const Edge &e) { return e.to == a; }), y.end());
    }

    int add_node() {
        adj.emplace_back();
        return static_cast<int>(adj.size()) - 1;
    }

    std::string newick() const {
        int root = n;
        while (root < static_cast<int>(adj.size()) && adj[root].size() != 3) ++root;
        if (root == static_cast<int>(adj.size())) throw std::runtime_error("No internal root");
        auto quoted = [](const std::string &s) {
            bool simple = !s.empty();
            for (unsigned char c : s)
                if (std::isspace(c) || std::string("(),:;[]'").find(c) != std::string::npos)
                    simple = false;
            if (simple) return s;
            std::string q = "'";
            for (char c : s) {
                q += c;
                if (c == '\'') q += '\'';
            }
            return q + "'";
        };
        std::function<std::string(int, int)> visit = [&](int u, int p) -> std::string {
            if (u < n) return quoted(names[u]);
            std::string s = "(";
            bool first = true;
            for (const auto &e : adj[u]) if (e.to != p) {
                if (!first) s += ',';
                first = false;
                s += visit(e.to, u);
            }
            return s + ")";
        };
        return visit(root, -1) + ";\n";
    }
};

class NewickParser {
    struct RawNode {
        std::string label;
        double length = 0;
        double support = -1;
        std::vector<int> children;
    };
    const std::string &text;
    size_t pos = 0;
    std::vector<RawNode> raw;

    void space() {
        while (pos < text.size()) {
            if (std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            else if (text[pos] == '[') {
                int depth = 1;
                ++pos;
                while (pos < text.size() && depth) {
                    if (text[pos] == '[') ++depth;
                    if (text[pos] == ']') --depth;
                    ++pos;
                }
            } else break;
        }
    }

    std::string token() {
        space();
        std::string s;
        if (pos < text.size() && text[pos] == '\'') {
            ++pos;
            while (pos < text.size()) {
                if (text[pos] == '\'') {
                    ++pos;
                    if (pos < text.size() && text[pos] == '\'') {
                        s += '\'';
                        ++pos;
                    } else break;
                } else s += text[pos++];
            }
        } else {
            while (pos < text.size()) {
                char c = text[pos];
                if (std::isspace(static_cast<unsigned char>(c)) ||
                    std::string("(),:;[]").find(c) != std::string::npos) break;
                s += c;
                ++pos;
            }
        }
        space();
        return s;
    }

    int subtree() {
        space();
        int u = static_cast<int>(raw.size());
        raw.emplace_back();
        if (pos < text.size() && text[pos] == '(') {
            ++pos;
            int first_child = subtree();
            raw[u].children.push_back(first_child);
            space();
            while (pos < text.size() && text[pos] == ',') {
                ++pos;
                int v = subtree();
                raw[u].children.push_back(v);
                space();
            }
            if (pos >= text.size() || text[pos] != ')')
                throw std::runtime_error("Malformed Newick parenthesis");
            ++pos;
            std::string s = token();
            if (!s.empty()) {
                char *end = nullptr;
                double x = std::strtod(s.c_str(), &end);
                if (end && *end == '\0' && x >= 0 && x <= 1) raw[u].support = x;
            }
        } else {
            raw[u].label = token();
            if (raw[u].label.empty()) throw std::runtime_error("Missing leaf label");
        }
        space();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            std::string s = token();
            char *end = nullptr;
            double x = std::strtod(s.c_str(), &end);
            if (s.empty() || !end || end == s.c_str() || *end != '\0' || !std::isfinite(x) || x < 0)
                throw std::runtime_error("Invalid branch length");
            raw[u].length = x;
        }
        return u;
    }

public:
    explicit NewickParser(const std::string &s) : text(s) {
        if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xef &&
            static_cast<unsigned char>(s[1]) == 0xbb &&
            static_cast<unsigned char>(s[2]) == 0xbf) pos = 3;
    }

    Tree parse(const std::vector<std::string> &expected = {}) {
        int root = subtree();
        space();
        if (pos >= text.size() || text[pos] != ';')
            throw std::runtime_error("Missing Newick terminator");
        ++pos;
        space();
        if (pos != text.size()) throw std::runtime_error("Trailing Newick data");
        Tree t;
        if (expected.empty()) {
            for (const auto &u : raw) if (u.children.empty()) t.names.push_back(u.label);
            std::sort(t.names.begin(), t.names.end());
            if (std::adjacent_find(t.names.begin(), t.names.end()) != t.names.end())
                throw std::runtime_error("Duplicate leaves");
        } else t.names = expected;
        t.n = static_cast<int>(t.names.size());
        std::unordered_map<std::string, int> ids;
        for (int i = 0; i < t.n; ++i) ids[t.names[i]] = i;
        std::vector<int> index(raw.size(), -1);
        std::vector<bool> seen(t.n);
        int next = t.n;
        for (int u = 0; u < static_cast<int>(raw.size()); ++u) {
            if (raw[u].children.empty()) {
                auto it = ids.find(raw[u].label);
                if (it == ids.end() || seen[it->second])
                    throw std::runtime_error("Inconsistent gene-tree leaves");
                index[u] = it->second;
                seen[it->second] = true;
            } else index[u] = next++;
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::runtime_error("Missing gene-tree leaves");
        t.adj.resize(next);
        for (int u = 0; u < static_cast<int>(raw.size()); ++u)
            for (int v : raw[u].children)
                t.connect(index[u], index[v], raw[v].length, raw[v].support);
        int r = index[root];
        if (t.adj[r].size() == 2) {
            Edge a = t.adj[r][0], b = t.adj[r][1];
            t.disconnect(r, a.to);
            t.disconnect(r, b.to);
            t.connect(a.to, b.to, a.length + b.length, std::max(a.support, b.support));
        }
        Tree compact;
        compact.n = t.n;
        compact.names = t.names;
        std::vector<int> remap(t.adj.size(), -1);
        next = t.n;
        for (int u = 0; u < t.n; ++u) remap[u] = u;
        for (int u = t.n; u < static_cast<int>(t.adj.size()); ++u)
            if (!t.adj[u].empty()) remap[u] = next++;
        compact.adj.resize(next);
        for (int u = 0; u < static_cast<int>(t.adj.size()); ++u)
            for (const auto &e : t.adj[u]) if (u < e.to)
                compact.connect(remap[u], remap[e.to], e.length, e.support);
        for (int u = 0; u < next; ++u)
            if (compact.adj[u].size() != static_cast<size_t>(u < t.n ? 1 : 3))
                throw std::runtime_error("Input is not unrooted binary");
        return compact;
    }
};

struct Gene {
    int n, size, root;
    double rate_scale = 1;
    double weight = 1, base_weight = 1, leaf_multiplier = 1;
    double outgroup_weight = 1;
    std::vector<int> child0, child1, postorder;
    std::vector<double> length, evidence, attenuation, residual;
    std::vector<int> topodist;
    std::vector<double> lendist, evdist;

    explicit Gene(const Tree &t) : n(t.n), size(static_cast<int>(t.adj.size())) {
        child0.assign(size, -1);
        child1.assign(size, -1);
        length.assign(size, 0);
        evidence.assign(size, 0);
        attenuation.assign(size, 1);
        residual.assign(size, 0);
        root = t.adj[0][0].to;
        std::vector<double> positives;
        for (int u = n; u < size; ++u)
            for (const auto &e : t.adj[u]) if (e.to >= n && u < e.to && e.length > 0.00005)
                positives.push_back(e.length);
        double scale = 0.02;
        if (!positives.empty()) {
            std::sort(positives.begin(), positives.end());
            scale = positives[positives.size() / 2];
        }
        auto edge_evidence = [scale](const Edge &e) {
            if (e.support >= 0) return std::clamp((3.0 * e.support - 1.0) / 2.0, 0.0, 1.0);
            return -std::expm1(-e.length / std::max(0.001, scale));
        };
        std::function<void(int, int, const Edge &)> dfs =
            [&](int u, int p, const Edge &incoming) {
                length[u] = incoming.length;
                evidence[u] = edge_evidence(incoming);
                if (u < n) return;
                int k = 0;
                for (const auto &e : t.adj[u]) if (e.to != p) {
                    if (k++ == 0) child0[u] = e.to;
                    else child1[u] = e.to;
                    dfs(e.to, u, e);
                }
                postorder.push_back(u);
            };
        dfs(root, 0, t.adj[0][0]);
        length[0] = t.adj[0][0].length;
        evidence[0] = 1;
        topodist.assign(n * n, 0);
        lendist.assign(n * n, 0);
        evdist.assign(n * n, 0);
        for (int leaf = 0; leaf < n; ++leaf) {
            std::function<void(int, int, int, double, double)> walk =
                [&](int u, int p, int td, double ld, double ed) {
                    if (u < n) {
                        topodist[leaf * n + u] = td;
                        lendist[leaf * n + u] = ld;
                        evdist[leaf * n + u] = ed;
                    }
                    for (const auto &e : t.adj[u]) if (e.to != p)
                        walk(e.to, u, td + 1, ld + e.length,
                             ed + (u >= n && e.to >= n ? edge_evidence(e) : 1.0));
                };
            walk(leaf, -1, 0, 0, 0);
        }
        rate_scale = 0;
        for (int i = 1; i < n; ++i) rate_scale += lendist[i];
        rate_scale = std::max(1e-6, rate_scale / (n - 1));
    }

    void set_weights(int mode, double alpha, double beta) {
        for (int u = 0; u < size; ++u) {
            attenuation[u] = std::exp(-alpha * length[u]);
            if (mode == 0) residual[u] = 0;
            else if (mode == 1) residual[u] = 1.0 - evidence[u];
            else if (mode == 2) residual[u] = std::exp(-beta * length[u]);
            else if (mode == 3) residual[u] = (1.0 - evidence[u]) * std::exp(-beta * length[u]);
            else if (mode == 4) residual[u] = std::pow(1.0 - evidence[u], beta);
            else if (mode == 5) residual[u] = evidence[u] > (3 * beta - 1) / 2 ? 0.0 : 1.0;
            else if (mode == 6) {
                double floor = (3 * beta - 1) / 2;
                residual[u] = 1.0 - std::clamp((evidence[u] - floor) / (1 - floor), 0.0, 1.0);
            } else if (mode == 7) residual[u] = std::exp(-beta * length[u] / rate_scale);
            else if (mode == 8) {
                double e = evidence[u] * -std::expm1(-beta * length[u] / rate_scale);
                residual[u] = 1.0 - e;
            } else if (mode == 9) residual[u] = (2.0 / 3.0) * (1.0 - evidence[u]);
            else if (mode >= 10 && mode <= 13) {
                residual[u] = std::pow(1.0 - evidence[u], beta);
                if (u < n) attenuation[u] = 1;
                else if (mode == 10) attenuation[u] = std::pow((2 * evidence[u] + 1) / 3, alpha);
                else if (mode == 11) attenuation[u] = std::exp(-alpha);
                else if (mode == 12) attenuation[u] = std::exp(-alpha * (1.0 - evidence[u]));
                else attenuation[u] = std::exp(-alpha * length[u] / rate_scale);
            }
            else if (mode >= 14 && mode <= 17) {
                residual[u] = std::pow(1.0 - evidence[u], beta);
                attenuation[u] = 1;
            }
            else throw std::runtime_error("Unknown weighting mode");
        }
    }

    void set_locus_weight(double w) {
        weight = w;
        leaf_multiplier = std::sqrt(std::sqrt(w));
    }
};

Tree neighbor_joining(const std::vector<double> &distance,
                     const std::vector<std::string> &names, bool bionj = false) {
    int n = static_cast<int>(names.size()), capacity = 2 * n;
    std::vector<double> d(capacity * capacity), variance(capacity * capacity);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        variance[i * capacity + j] = d[i * capacity + j] = distance[i * n + j];
    std::vector<int> active(n);
    std::iota(active.begin(), active.end(), 0);
    Tree t;
    t.n = n;
    t.names = names;
    t.adj.resize(n);
    std::vector<double> sum(capacity);
    while (active.size() > 3) {
        int m = static_cast<int>(active.size());
        for (int i : active) {
            sum[i] = 0;
            for (int j : active) sum[i] += d[i * capacity + j];
        }
        double best = std::numeric_limits<double>::infinity();
        int ai = 0, aj = 1;
        for (int i = 0; i < m; ++i) for (int j = i + 1; j < m; ++j) {
            double q = (m - 2) * d[active[i] * capacity + active[j]]
                       - sum[active[i]] - sum[active[j]];
            if (q < best) { best = q; ai = i; aj = j; }
        }
        int a = active[ai], b = active[aj], u = t.add_node();
        double dab = d[a * capacity + b];
        double la = 0.5 * (dab + (sum[a] - sum[b]) / (m - 2));
        double lb = dab - la;
        t.connect(u, a);
        t.connect(u, b);
        double lambda = 0.5;
        if (bionj && variance[a * capacity + b] > 1e-12) {
            double v = 0;
            for (int k : active) if (k != a && k != b)
                v += variance[b * capacity + k] - variance[a * capacity + k];
            lambda = std::clamp(0.5 + v / (2 * (m - 2) * variance[a * capacity + b]), 0.0, 1.0);
        }
        for (int k : active) if (k != a && k != b) {
            double x = lambda * (d[a * capacity + k] - la)
                       + (1 - lambda) * (d[b * capacity + k] - lb);
            d[u * capacity + k] = d[k * capacity + u] = x;
            x = lambda * variance[a * capacity + k] + (1 - lambda) * variance[b * capacity + k]
                - lambda * (1 - lambda) * variance[a * capacity + b];
            variance[u * capacity + k] = variance[k * capacity + u] = std::max(0.0, x);
        }
        active.erase(active.begin() + aj);
        active.erase(active.begin() + ai);
        active.push_back(u);
    }
    int root = t.add_node();
    for (int u : active) t.connect(root, u);
    return t;
}

std::vector<double> calibrated_distance(const std::vector<Gene> &genes) {
    int n = genes.front().n;
    std::vector<double> distance(n * n);
    double total_weight = 0;
    for (const auto &g : genes) {
        total_weight += g.weight;
        if (g.weight == 0) continue;
        std::vector<int> parent(g.size, -1);
        parent[g.root] = 0;
        for (int u : g.postorder) {
            parent[g.child0[u]] = u;
            parent[g.child1[u]] = u;
        }
        std::vector<double> cost(g.size, 1);
        for (int u : g.postorder) if (u != g.root) cost[u] = 1.0 - g.residual[u];
        for (int leaf = 0; leaf < n; ++leaf) {
            std::function<void(int,int,double)> visit = [&](int u, int p, double d) {
                if (u < n) distance[leaf * n + u] += d * g.weight;
                int next[3], degree = 0;
                if (u == 0) next[degree++] = g.root;
                else {
                    next[degree++] = parent[u];
                    if (u >= n) {
                        next[degree++] = g.child0[u];
                        next[degree++] = g.child1[u];
                    }
                }
                for (int k = 0; k < degree; ++k) if (next[k] != p) {
                    int v = next[k], child = parent[u] == v ? u : v;
                    visit(v, u, d + cost[child]);
                }
            };
            visit(leaf, -1, 0);
        }
    }
    for (double &d : distance) d /= total_weight;
    return distance;
}

std::vector<double> average_distance(const std::vector<Gene> &genes, int type) {
    if (type == 5) return calibrated_distance(genes);
    int n = genes.front().n;
    std::vector<double> d(n * n);
    double total_weight = 0;
    for (const Gene &g : genes) {
        total_weight += g.weight;
        double scale = 1;
        if (type == 2 || type == 3) {
            scale = 0;
            for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j)
                scale += g.lendist[i * n + j];
            scale /= n * (n - 1) * 0.5;
        }
        for (int i = 0; i < n * n; ++i) {
            if (type == 0) d[i] += g.weight * g.topodist[i];
            else if (type == 1) d[i] += g.weight * g.evdist[i];
            else if (type == 2) d[i] += g.weight * g.lendist[i] / std::max(scale, 1e-10);
            else if (type == 3) d[i] += g.weight * std::min(g.lendist[i] / std::max(scale, 1e-10), 2.0);
            else if (type == 4) d[i] += g.weight * std::exp(-0.2 * g.evdist[i]);
        }
    }
    for (double &x : d) {
        x /= total_weight;
        if (type == 4) x = -std::log(std::max(x, 1e-15)) / 0.2;
    }
    for (int i = 0; i < n; ++i) d[i * n + i] = 0;
    return d;
}

template<int C, bool Diagonal>
struct Message {
    static constexpr int K = Diagonal ? C * (C + 1) / 2 : C * (C - 1) / 2;
    std::array<double, C> count{};
    std::array<double, K> pair{}, discounted{};
};

template<int C, bool D>
Message<C, D> combine(const Message<C, D> &x, const Message<C, D> &y,
                      double attenuation, double residual) {
    Message<C, D> z;
    for (int c = 0; c < C; ++c) z.count[c] = attenuation * (x.count[c] + y.count[c]);
    int k = 0;
    for (int i = 0; i < C; ++i) for (int j = i + (D ? 0 : 1); j < C; ++j, ++k) {
        double cross = x.count[i] * y.count[j];
        if (i != j) cross += x.count[j] * y.count[i];
        z.pair[k] = x.pair[k] + y.pair[k] + cross;
        z.discounted[k] = residual * (x.discounted[k] + y.discounted[k] + cross);
    }
    return z;
}

class QuartetScorer {
    std::vector<Gene> &genes;
    std::vector<Message<4, false>> down4, up4;
    std::vector<Message<3, true>> down3, up3;

    template<int C, bool D>
    void downpass(const Gene &g, const std::vector<int> &color,
                  std::vector<Message<C, D>> &down, std::vector<Message<C, D>> &up) {
        for (int u = 0; u < g.n; ++u) {
            down[u] = {};
            if (color[u] >= 0)
                down[u].count[color[u]] = g.attenuation[u] * g.leaf_multiplier
                                          * (u == 0 ? g.outgroup_weight : 1.0);
        }
        for (int u : g.postorder)
            down[u] = combine(down[g.child0[u]], down[g.child1[u]],
                              g.attenuation[u], g.residual[u]);
        up[g.root] = down[0];
    }

public:
    explicit QuartetScorer(std::vector<Gene> &g) : genes(g) {
        int size = 0;
        for (const auto &x : g) size = std::max(size, x.size);
        down4.resize(size); up4.resize(size);
        down3.resize(size); up3.resize(size);
    }

    std::array<double, 3> four(const std::vector<int> &color) {
        std::array<double, 3> total{};
        constexpr int pairs[3][2] = {{0,5}, {1,4}, {2,3}};
        constexpr int colors[3][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2}};
        for (const Gene &g : genes) {
            if (g.weight == 0) continue;
            downpass(g, color, down4, up4);
            for (auto it = g.postorder.rbegin(); it != g.postorder.rend(); ++it) {
                int u = *it, a = g.child0[u], b = g.child1[u];
                const Message<4, false> *side[3] = {&down4[a], &down4[b], &up4[u]};
                for (int s = 0; s < 3; ++s) {
                    const auto &x = *side[s], &y = *side[(s+1)%3], &z = *side[(s+2)%3];
                    for (int t = 0; t < 3; ++t) {
                        int i = colors[t][0], j = colors[t][1], k = colors[t][2], l = colors[t][3];
                        total[t] += (x.pair[pairs[t][0]] - x.discounted[pairs[t][0]])
                                    * (y.count[k] * z.count[l] + y.count[l] * z.count[k])
                                  + (x.pair[pairs[t][1]] - x.discounted[pairs[t][1]])
                                    * (y.count[i] * z.count[j] + y.count[j] * z.count[i]);
                    }
                }
                if (a >= g.n) up4[a] = combine(up4[u], down4[b], g.attenuation[a], g.residual[a]);
                if (b >= g.n) up4[b] = combine(up4[u], down4[a], g.attenuation[b], g.residual[b]);
            }
        }
        // Both branching vertices of every induced quartet contribute once.
        for (double &x : total) x *= 0.5;
        return total;
    }

    double three(const std::vector<int> &color) {
        double total = 0;
        constexpr int same[3] = {0,3,5};
        constexpr int other[3] = {4,2,1};
        for (const Gene &g : genes) {
            if (g.weight == 0) continue;
            downpass(g, color, down3, up3);
            for (auto it = g.postorder.rbegin(); it != g.postorder.rend(); ++it) {
                int u = *it, a = g.child0[u], b = g.child1[u];
                const Message<3, true> *side[3] = {&down3[a], &down3[b], &up3[u]};
                for (int s = 0; s < 3; ++s) {
                    const auto &x = *side[s], &y = *side[(s+1)%3], &z = *side[(s+2)%3];
                    for (int c = 0; c < 3; ++c) {
                        int i = (c+1)%3, j = (c+2)%3;
                        total += (x.pair[same[c]] - x.discounted[same[c]])
                                 * (y.count[i] * z.count[j] + y.count[j] * z.count[i])
                               + (x.pair[other[c]] - x.discounted[other[c]])
                                 * y.count[c] * z.count[c];
                    }
                }
                if (a >= g.n) up3[a] = combine(up3[u], down3[b], g.attenuation[a], g.residual[a]);
                if (b >= g.n) up3[b] = combine(up3[u], down3[a], g.attenuation[b], g.residual[b]);
            }
        }
        return total * 0.5;
    }
};

void color_component(const Tree &t, int u, int p, int c, std::vector<int> &color) {
    if (u < t.n) color[u] = c;
    else for (const auto &e : t.adj[u]) if (e.to != p)
        color_component(t, e.to, u, c, color);
}

double score_tree(const Tree &t, QuartetScorer &scorer) {
    std::vector<int> color(t.n);
    double score = 0;
    for (int u = t.n; u < static_cast<int>(t.adj.size()); ++u) if (t.adj[u].size() == 3) {
        for (int k = 0; k < 3; ++k) color_component(t, t.adj[u][k].to, u, k, color);
        score += scorer.three(color);
    }
    return 0.5 * score;
}

void nni_swap(Tree &t, int u, int v, int a, int b) {
    t.disconnect(u, a);
    t.disconnect(v, b);
    t.connect(u, b);
    t.connect(v, a);
}

struct SearchResult {
    int changes = 0;
    double gain = 0;
};

SearchResult refine_nni(Tree &t, QuartetScorer &scorer, std::mt19937_64 &rng,
                        Clock::time_point deadline, int rounds = 12) {
    SearchResult result;
    std::vector<int> color(t.n);
    for (int round = 0; round < rounds; ++round) {
        std::vector<std::pair<int, int>> edges;
        for (int u = t.n; u < static_cast<int>(t.adj.size()); ++u)
            for (const auto &e : t.adj[u]) if (e.to >= t.n && u < e.to)
                edges.emplace_back(u, e.to);
        std::shuffle(edges.begin(), edges.end(), rng);
        int changes = 0;
        for (auto [u,v] : edges) {
            if (Clock::now() >= deadline) return result;
            bool exists = false;
            for (const auto &e : t.adj[u]) if (e.to == v) exists = true;
            if (!exists) continue;
            int x[2], y[2], i = 0, j = 0;
            for (const auto &e : t.adj[u]) if (e.to != v) x[i++] = e.to;
            for (const auto &e : t.adj[v]) if (e.to != u) y[j++] = e.to;
            color_component(t, x[0], u, 0, color);
            color_component(t, x[1], u, 1, color);
            color_component(t, y[0], v, 2, color);
            color_component(t, y[1], v, 3, color);
            auto s = scorer.four(color);
            int best = s[1] > s[2] ? 1 : 2;
            double gain = s[best] - s[0];
            if (gain > 1e-7 * std::max(1.0, s[0])) {
                nni_swap(t, u, v, x[1], y[best == 1 ? 0 : 1]);
                ++changes;
                ++result.changes;
                result.gain += gain;
            }
        }
        std::cerr << "NNI round " << round + 1 << ": " << changes << " changes\n";
        if (!changes) break;
    }
    return result;
}

struct TaxonSet {
    std::array<uint64_t, 4> word{};
    bool operator==(const TaxonSet &other) const { return word == other.word; }
    bool contains(const TaxonSet &other) const {
        for (int k = 0; k < 4; ++k) if (other.word[k] & ~word[k]) return false;
        return true;
    }
    TaxonSet united(const TaxonSet &other) const {
        TaxonSet result;
        for (int k = 0; k < 4; ++k) result.word[k] = word[k] | other.word[k];
        return result;
    }
    TaxonSet minus(const TaxonSet &other) const {
        TaxonSet result;
        for (int k = 0; k < 4; ++k) result.word[k] = word[k] & ~other.word[k];
        return result;
    }
    int size() const {
        int s = 0;
        for (uint64_t w : word) s += __builtin_popcountll(w);
        return s;
    }
    void set(int leaf) { word[leaf / 64] |= uint64_t(1) << (leaf % 64); }
    void paint(int color, std::vector<int> &colors) const {
        for (int k = 0; k < 4; ++k) {
            uint64_t bits = word[k];
            while (bits) {
                int bit = __builtin_ctzll(bits);
                colors[64 * k + bit] = color;
                bits &= bits - 1;
            }
        }
    }
    int leaf() const {
        for (int k = 0; k < 4; ++k) if (word[k]) return 64 * k + __builtin_ctzll(word[k]);
        return -1;
    }
};

struct SetHash {
    size_t operator()(const TaxonSet &s) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t x : s.word) {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            h ^= (x ^ (x >> 31)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

double gene_concordance(const Tree &t, const std::vector<Gene> &genes, double gamma) {
    std::unordered_set<TaxonSet, SetHash> known;
    std::function<TaxonSet(int,int)> collect = [&](int u, int p) -> TaxonSet {
        TaxonSet s;
        if (u < t.n) s.set(u);
        else for (const auto &e : t.adj[u]) if (e.to != p)
            s = s.united(collect(e.to,u));
        if (u >= t.n && p != 0) known.insert(s);
        return s;
    };
    collect(t.adj[0][0].to,0);
    double matched = 0, total = 0;
    for (const auto &g : genes) {
        std::vector<TaxonSet> clade(g.size);
        for (int u = 1; u < g.n; ++u) clade[u].set(u);
        for (int u : g.postorder) {
            clade[u] = clade[g.child0[u]].united(clade[g.child1[u]]);
            if (u == g.root) continue;
            double w = g.weight * (1.0 - std::pow(1.0 - g.evidence[u], gamma));
            total += w;
            if (known.count(clade[u])) matched += w;
        }
    }
    return total > 0 ? matched / total : 0;
}

int main(int argc, char **argv) {
    auto start = Clock::now();
    try {
        if (argc != 3) throw std::runtime_error("Usage: infer INPUT OUTPUT");
        constexpr double support_power = 0.5;
        constexpr double locus_power = 0.5;
        constexpr double local_penalty = 0.3;
        constexpr double concordance_threshold = 0.6;
        constexpr int rounds = 30;
        uint64_t seed = 1;
        if (const char *s = std::getenv("SEED")) seed = std::strtoull(s, nullptr, 10);
        std::mt19937_64 rng(seed);
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("Cannot open input");
        std::string line;
        std::vector<Tree> trees;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            trees.push_back(NewickParser(line).parse(trees.empty() ? std::vector<std::string>{}
                                                                 : trees.front().names));
        }
        if (trees.empty() || trees.front().n < 4 || trees.front().n > 256)
            throw std::runtime_error("Unsupported or empty gene-tree input");
        std::vector<Gene> genes;
        for (const auto &t : trees) {
            genes.emplace_back(t);
            genes.back().set_weights(4, 0, support_power);
        }
        {
            double mean_weight = 0;
            for (auto &g : genes) {
                double quality = 0;
                for (int u : g.postorder) if (u != g.root) quality += g.evidence[u];
                quality = 0.05 + 0.95 * quality / std::max(1, g.n - 3);
                g.base_weight = std::pow(quality, locus_power);
                mean_weight += g.base_weight;
            }
            mean_weight /= genes.size();
            for (auto &g : genes) {
                g.base_weight /= mean_weight;
                g.set_locus_weight(g.base_weight);
            }
        }
        QuartetScorer scorer(genes);
        auto deadline = start + std::chrono::seconds(120);
        bool check_score = std::getenv("CHECK_SCORE") != nullptr;
        auto distance = average_distance(genes, 1);
        Tree initial = neighbor_joining(distance, trees.front().names);
        Tree t = initial;
        double predicted_score = check_score ? score_tree(t, scorer) : 0;
        auto preliminary = refine_nni(t, scorer, rng, deadline, rounds);
        predicted_score += preliminary.gain;
        double concordance = gene_concordance(t, genes, support_power);
        std::cerr << "Supported gene-split concordance " << concordance << "\n";
        if (Clock::now() < deadline) {
            if (concordance >= concordance_threshold) {
                for (auto &g : genes) g.set_weights(11, local_penalty, support_power);
                t = initial;
            } else {
                distance = calibrated_distance(genes);
                t = neighbor_joining(distance, trees.front().names);
            }
            rng.seed(seed);
            predicted_score = check_score ? score_tree(t, scorer) : 0;
            auto refined = refine_nni(t, scorer, rng, deadline, rounds);
            predicted_score += refined.gain;
        }
        if (check_score) {
            double actual = score_tree(t, scorer);
            std::cerr << "Quartet score " << actual << ", accumulated " << predicted_score << "\n";
            if (std::abs(actual - predicted_score) > 1e-6 * std::max(1.0, actual))
                throw std::runtime_error("Inconsistent topology-search score");
        }
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Cannot open output");
        out << t.newick();
        out.close();
        if (!out) throw std::runtime_error("Cannot write output");
        std::cerr << "Elapsed " << std::chrono::duration<double>(Clock::now() - start).count() << " s\n";
    } catch (const std::exception &e) {
        std::cerr << "Inference error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
