#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
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
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
constexpr int MAXN = 256;
constexpr int MAXV = 2 * MAXN;
constexpr int WORDS = 4;
int N;
vector<string> labels;
unordered_map<string, int> taxon;
mt19937_64 rng(1);
Clock::time_point started;
double elapsed() { return chrono::duration<double>(Clock::now() - started).count(); }
double time_limit = 125.0;
double branch_alpha = 0.0;
double support_power = 1.0;
double quality_power = 0.0;
double rate_power = 0.0;
int branch_mode = 0;
double length_kappa = 20.0;
int missing_strategy = 0;
bool expired() { return elapsed() > time_limit; }

struct Bits {
    array<uint64_t, WORDS> w{};
    void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    int size() const {
        int s = 0;
        for (auto x : w) s += __builtin_popcountll(x);
        return s;
    }
    bool operator==(const Bits &b) const { return w == b.w; }
    bool operator<(const Bits &b) const { return w < b.w; }
    Bits operator|(const Bits &b) const {
        Bits r;
        for (int i = 0; i < WORDS; ++i) r.w[i] = w[i] | b.w[i];
        return r;
    }
    Bits operator-(const Bits &b) const {
        Bits r;
        for (int i = 0; i < WORDS; ++i) r.w[i] = w[i] & ~b.w[i];
        return r;
    }
    bool subset(const Bits &b) const {
        for (int i = 0; i < WORDS; ++i) if (w[i] & ~b.w[i]) return false;
        return true;
    }
};
Bits all_taxa;
struct BitsHash {
    size_t operator()(const Bits &b) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (auto x : b.w) {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            h ^= (x ^ (x >> 31)) + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct Edge {
    int to;
    double length, support;
};
struct ParsedTree {
    vector<vector<Edge>> adj;
    vector<int> leaf;
    int newnode(int id = -1) {
        adj.emplace_back();
        leaf.push_back(id);
        return int(adj.size()) - 1;
    }
    void connect(int a, int b, double length, double support) {
        adj[a].push_back({b, length, support});
        adj[b].push_back({a, length, support});
    }
};
struct Parser {
    const string &s;
    size_t p = 0;
    ParsedTree tree;
    explicit Parser(const string &text) : s(text) {}
    void space() {
        while (p < s.size()) {
            if (isspace(static_cast<unsigned char>(s[p]))) ++p;
            else if (s[p] == '[') {
                int level = 1;
                ++p;
                while (p < s.size() && level) {
                    if (s[p] == '[') ++level;
                    if (s[p] == ']') --level;
                    ++p;
                }
            } else break;
        }
    }
    string token() {
        space();
        string out;
        if (p < s.size() && s[p] == '\'') {
            ++p;
            while (p < s.size()) {
                if (s[p] == '\'') {
                    ++p;
                    if (p < s.size() && s[p] == '\'') {
                        out.push_back('\'');
                        ++p;
                    } else break;
                } else out.push_back(s[p++]);
            }
        } else {
            while (p < s.size() && !isspace(static_cast<unsigned char>(s[p])) &&
                   string("(),:;[]").find(s[p]) == string::npos)
                out.push_back(s[p++]);
        }
        return out;
    }
    struct Node { int id; double length, support; };
    Node subtree() {
        space();
        bool internal = p < s.size() && s[p] == '(';
        int u;
        double length = 0, support = -1;
        if (internal) {
            ++p;
            u = tree.newnode();
            while (true) {
                auto c = subtree();
                tree.connect(u, c.id, c.length, c.support);
                space();
                if (p < s.size() && s[p] == ',') ++p;
                else break;
            }
            if (p >= s.size() || s[p++] != ')') throw runtime_error("Invalid Newick");
            space();
            if (p < s.size() && string("(),:;").find(s[p]) == string::npos) {
                string t = token();
                char *end;
                double v = strtod(t.c_str(), &end);
                if (*end == 0 && v >= 0 && v <= 1) support = v;
            }
        } else {
            string name = token();
            if (name.empty()) throw runtime_error("Missing taxon");
            auto it = taxon.find(name);
            if (it == taxon.end()) {
                int id = int(labels.size());
                labels.push_back(name);
                taxon[name] = id;
                u = tree.newnode(id);
            } else u = tree.newnode(it->second);
        }
        space();
        if (p < s.size() && s[p] == ':') {
            ++p;
            string t = token();
            length = max(0.0, strtod(t.c_str(), nullptr));
        }
        return {u, length, support};
    }
    ParsedTree parse() {
        auto root = subtree();
        space();
        if (p >= s.size() || s[p] != ';') throw runtime_error("Missing semicolon");
        if (tree.adj[root.id].size() == 2) {
            auto a = tree.adj[root.id][0], b = tree.adj[root.id][1];
            auto remove = [&](int u) {
                auto &v = tree.adj[u];
                v.erase(remove_if(v.begin(), v.end(), [&](Edge e) { return e.to == root.id; }), v.end());
            };
            remove(a.to);
            remove(b.to);
            double support = a.support < 0 ? b.support :
                             b.support < 0 ? a.support : max(a.support, b.support);
            tree.connect(a.to, b.to, a.length + b.length, support);
            tree.adj[root.id].clear();
        }
        return move(tree);
    }
};

struct Gene {
    int nodes, root;
    double weight = 1.0;
    array<int, MAXV> left{}, right{};
    array<double, MAXV> length{}, support{}, f{}, attenuation{}, estimated_support{};
    double mean_support = 1;
    vector<Bits> clades;
    explicit Gene(const ParsedTree &t) {
        nodes = N;
        left.fill(-1);
        right.fill(-1);
        support.fill(-1);
        int anchor = -1;
        for (int i = 0; i < int(t.leaf.size()); ++i)
            if (t.leaf[i] == 0) anchor = i;
        if (anchor < 0) throw runtime_error("Missing leaf");
        function<int(int, int, double, double)> visit = [&](int u, int p, double len, double sup) {
            if (t.leaf[u] >= 0) {
                int v = t.leaf[u];
                length[v] = len;
                support[v] = sup;
                return v;
            }
            vector<int> children;
            for (auto e : t.adj[u]) if (e.to != p)
                children.push_back(visit(e.to, u, e.length, e.support));
            if (children.size() != 2) throw runtime_error("Nonbinary gene tree");
            int v = nodes++;
            left[v] = children[0];
            right[v] = children[1];
            length[v] = len;
            support[v] = sup;
            return v;
        };
        auto e = t.adj[anchor].at(0);
        int rest = visit(e.to, anchor, e.length, -1);
        root = nodes++;
        left[root] = 0;
        right[root] = rest;
        length[0] = 0;
        length[rest] = e.length;
        vector<Bits> bits(nodes);
        for (int i = 0; i < N; ++i) bits[i].set(i);
        for (int i = N; i < nodes; ++i) {
            bits[i] = bits[left[i]] | bits[right[i]];
            if (bits[i].size() < N - 1) clades.push_back(bits[i]);
        }
        vector<pair<double, double>> observed;
        for (int i = N; i < root - 1; ++i) if (support[i] >= 0)
            observed.push_back({log(max(1e-9, length[i])), max(0.0, (support[i] - 1.0 / 3) * 1.5)});
        sort(observed.begin(), observed.end());
        double avg = 0;
        for (auto v : observed) avg += v.second;
        avg = observed.empty() ? 1 : avg / observed.size();
        mean_support = (1 + 2 * avg) / 3;
        struct Block { double x, y, weight; };
        vector<Block> blocks;
        for (auto [x, y] : observed) {
            blocks.push_back({x, y, 1});
            while (blocks.size() > 1 && blocks[blocks.size() - 2].y > blocks.back().y) {
                auto b = blocks.back(); blocks.pop_back();
                auto &a = blocks.back();
                double w = a.weight + b.weight;
                a.x = (a.x * a.weight + b.x * b.weight) / w;
                a.y = (a.y * a.weight + b.y * b.weight) / w;
                a.weight = w;
            }
        }
        double total_length = 0;
        for (int i = 0; i < nodes; ++i) total_length += length[i];
        for (int i = 0; i < nodes; ++i) {
            estimated_support[i] = support[i] >= 0 ? support[i] : 1;
            if (i < N || i >= root - 1 || support[i] >= 0) continue;
            double p = total_length > 1e-8 ? 1 - exp(-200 * length[i]) : 1;
            if (!blocks.empty()) {
                double x = log(max(1e-9, length[i]));
                auto it = lower_bound(blocks.begin(), blocks.end(), x,
                                      [](const Block &b, double v) { return b.x < v; });
                if (it == blocks.begin()) p = it->y;
                else if (it == blocks.end()) p = blocks.back().y;
                else {
                    auto &a = *(it - 1);
                    double fraction = (x - a.x) / max(1e-12, it->x - a.x);
                    p = a.y + fraction * (it->y - a.y);
                }
            }
            estimated_support[i] = (1 + 2 * clamp(p, 0.0, 1.0)) / 3;
        }
    }
    double edge_support(int i) const {
        if (support[i] >= 0) return support[i];
        if (i < N || i >= root - 1 || missing_strategy == 0) return 1;
        return missing_strategy == 1 ? mean_support : estimated_support[i];
    }
    void weights(double mode) {
        double quality = 0, rate = 0;
        for (int i = 0; i < nodes; ++i) rate += length[i];
        for (int i = N; i < root - 1; ++i) {
            double s = max(0.0, (edge_support(i) - 1.0 / 3) * 1.5);
            quality += s;
        }
        quality /= max(1, N - 3);
        weight = pow(max(0.01, quality), quality_power) * pow(max(0.001, rate), rate_power);
        double alpha = branch_alpha;
        if (branch_mode == 1) alpha *= N / max(0.001, rate);
        for (int i = 0; i < nodes; ++i) {
            double s = edge_support(i);
            if (mode < 0) s = 1;
            else if (mode == 0) s = max(0.0, (s - 1.0 / 3) * 1.5);
            else if (mode == 1) s = max(0.0, (s - 1.0 / 3) * 1.5) >= 0.5 ? 1 : 0;
            else if (mode == 2) s = max(0.0, (s - 1.0 / 3) * 1.5) >= 0.8 ? 1 : 0;
            else if (mode == 3) s = edge_support(i);
            else if (mode == 5) s = 1 - exp(-length_kappa * length[i]);
            else if (mode == 6) s = max(0.0, (s - 1.0 / 3) * 1.5) *
                                    (1 - exp(-length_kappa * length[i]));
            else if (mode == 7) s = 0.5 * max(0.0, (s - 1.0 / 3) * 1.5) +
                                    0.5 * (1 - exp(-length_kappa * length[i]));
            s = pow(clamp(s, 0.0, 1.0), support_power);
            f[i] = 1 - clamp(s, 0.0, 1.0);
            double external_length = length[i];
            if (branch_mode == 2) external_length = i >= N && i < root - 1 ? 1.0 : 0.0;
            attenuation[i] = exp(-alpha * external_length);
            if (branch_mode == 3) attenuation[i] = exp(-alpha * f[i]);
            if (branch_mode == 4) attenuation[i] = pow(max(1e-6, 1 - (2.0 / 3) * f[i]), alpha);
        }
    }
};
vector<Gene> genes;

struct Tree {
    int nodes;
    array<vector<int>, MAXV> adj;
    explicit Tree(int size = 0) : nodes(size) {}
    void add(int a, int b) { adj[a].push_back(b); adj[b].push_back(a); }
    void del(int a, int b) {
        auto &v = adj[a];
        v.erase(find(v.begin(), v.end(), b));
        auto &w = adj[b];
        w.erase(find(w.begin(), w.end(), a));
    }
    void swap_branches(int u, int a, int v, int b) {
        del(u, a);
        del(v, b);
        add(u, b);
        add(v, a);
    }
    Bits component(int u, int parent) const {
        Bits b;
        function<void(int, int)> dfs = [&](int v, int p) {
            if (v < N) b.set(v);
            for (int w : adj[v]) if (w != p) dfs(w, v);
        };
        dfs(u, parent);
        return b;
    }
    vector<Bits> splits() const {
        vector<Bits> out;
        function<Bits(int, int)> dfs = [&](int v, int p) {
            Bits b;
            if (v < N) b.set(v);
            for (int w : adj[v]) if (w != p) b = b | dfs(w, v);
            if (v >= N && b.size() > 1 && b.size() < N - 1) out.push_back(b);
            return b;
        };
        dfs(adj[0][0], 0);
        return out;
    }
    string newick() const {
        function<string(int, int)> dfs = [&](int v, int p) -> string {
            if (v < N) {
                string s = labels[v];
                if (s.find_first_of(" \t\n(),:;[]'") == string::npos) return s;
                string q = "'";
                for (char c : s) { q.push_back(c); if (c == '\'') q.push_back(c); }
                return q + "'";
            }
            string s = "(";
            for (int w : adj[v]) if (w != p) {
                if (s.size() > 1) s += ',';
                s += dfs(w, v);
            }
            return s + ')';
        };
        return dfs(N, -1) + ";\n";
    }
    bool valid() const {
        if (nodes != 2 * N - 2) return false;
        vector<int> seen(nodes);
        function<bool(int, int)> visit = [&](int u, int parent) {
            if (u < 0 || u >= nodes || seen[u]) return false;
            seen[u] = 1;
            if (adj[u].size() != size_t(u < N ? 1 : 3)) return false;
            for (int v : adj[u]) if (v != parent) {
                if (v < 0 || v >= nodes ||
                    count(adj[v].begin(), adj[v].end(), u) != 1 || !visit(v, u)) return false;
            }
            return true;
        };
        if (!visit(0, -1)) return false;
        return accumulate(seen.begin(), seen.end(), 0) == nodes;
    }
};

using Matrix = vector<vector<double>>;
Matrix distances(int mode, const vector<double> &gene_weights = {}) {
    Matrix d(N, vector<double>(N));
    double decay = mode >= 8 ? array<double, 5>{0.1, 0.2, 0.4, 0.7, 1.0}[min(4, mode - 8)] : 0;
    double sum_weights = 0;
    for (int gi = 0; gi < int(genes.size()); ++gi) {
        const auto &g = genes[gi];
        vector<vector<pair<int, double>>> adj(g.nodes);
        double norm = 0;
        for (int u = 0; u < g.nodes; ++u) norm += g.length[u];
        norm = norm > 0 ? N / norm : 1;
        for (int u = N; u < g.nodes; ++u) for (int v : {g.left[u], g.right[u]}) {
            double s = g.edge_support(v);
            double val = 1;
            if (mode == 1) val = max(0.0, (s - 1.0 / 3) * 1.5);
            if (mode == 2) val = max(0.0, (s - 1.0 / 3) * 1.5) >= 0.5;
            if (mode == 3) val = g.length[v] * norm;
            if (mode == 4) val = s;
            if (mode == 5) val = sqrt(g.length[v] * norm);
            if (mode == 6) val = max(0.0, (s - 1.0 / 3) * 1.5) >= 0.8;
            if (mode == 7) val = pow(max(0.0, (s - 1.0 / 3) * 1.5), 2);
            adj[u].push_back({v, val});
            adj[v].push_back({u, val});
        }
        double gw = gene_weights.empty() ? 1 : gene_weights[gi];
        sum_weights += gw;
        for (int a = 0; a < N; ++a) {
            function<void(int, int, double)> dfs = [&](int u, int p, double dist) {
                if (u < a) d[a][u] += gw * (decay > 0 ? exp(-decay * dist) : dist);
                for (auto [v, len] : adj[u]) if (v != p) dfs(v, u, dist + len);
            };
            dfs(a, -1, 0);
        }
    }
    for (int a = 0; a < N; ++a) for (int b = 0; b < a; ++b) {
        if (decay > 0) d[a][b] = -log(max(1e-100, d[a][b] / sum_weights)) / decay;
        d[b][a] = d[a][b];
    }
    return d;
}

Tree neighbor_joining(const Matrix &input) {
    Matrix d(2 * N, vector<double>(2 * N));
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) d[i][j] = input[i][j];
    Tree t(N);
    vector<int> active(N);
    iota(active.begin(), active.end(), 0);
    vector<double> sum(2 * N);
    while (active.size() > 3) {
        int k = int(active.size());
        for (int a : active) {
            sum[a] = 0;
            for (int b : active) sum[a] += d[a][b];
        }
        double best = numeric_limits<double>::infinity();
        int ai = -1, bi = -1;
        for (int i = 0; i < k; ++i) for (int j = 0; j < i; ++j) {
            int a = active[i], b = active[j];
            double q = (k - 2) * d[a][b] - sum[a] - sum[b];
            if (q < best) { best = q; ai = i; bi = j; }
        }
        int a = active[ai], b = active[bi], u = t.nodes++;
        t.add(a, u);
        t.add(b, u);
        for (int c : active) if (c != a && c != b)
            d[c][u] = d[u][c] = (d[a][c] + d[b][c] - d[a][b]) * 0.5;
        active.erase(active.begin() + ai);
        active.erase(active.begin() + bi);
        active.push_back(u);
    }
    int u = t.nodes++;
    for (int a : active) t.add(u, a);
    return t;
}

void bme_nni(Tree &t, const Matrix &d, int rounds = 30) {
    for (int iteration = 0; iteration < rounds && !expired(); ++iteration) {
        vector<pair<int, int>> edges;
        for (int u = N; u < t.nodes; ++u) for (int v : t.adj[u]) if (v > u) edges.push_back({u, v});
        shuffle(edges.begin(), edges.end(), rng);
        int moves = 0;
        for (auto [u, v] : edges) {
            if (find(t.adj[u].begin(), t.adj[u].end(), v) == t.adj[u].end()) continue;
            array<int, 4> branches;
            int k = 0;
            for (int x : t.adj[u]) if (x != v) branches[k++] = x;
            for (int x : t.adj[v]) if (x != u) branches[k++] = x;
            array<vector<pair<int, double>>, 4> leaves;
            for (int c = 0; c < 4; ++c) {
                function<void(int, int, double)> dfs = [&](int x, int p, double w) {
                    if (x < N) { leaves[c].push_back({x, w}); return; }
                    for (int y : t.adj[x]) if (y != p) dfs(y, x, w * 0.5);
                };
                dfs(branches[c], c < 2 ? u : v, 1.0);
            }
            double dist[4][4]{};
            for (int a = 0; a < 4; ++a) for (int b = 0; b < a; ++b) {
                double s = 0;
                for (auto [i, wi] : leaves[a]) for (auto [j, wj] : leaves[b])
                    s += wi * wj * d[i][j];
                dist[a][b] = dist[b][a] = s;
            }
            double score[3] = {dist[0][1] + dist[2][3], dist[0][2] + dist[1][3],
                               dist[0][3] + dist[1][2]};
            int best = int(min_element(score, score + 3) - score);
            if (score[best] < score[0] - 1e-9) {
                t.swap_branches(u, branches[1], v, branches[best + 1]);
                ++moves;
            }
        }
        if (!moves) break;
    }
}

constexpr int pa[6] = {0, 0, 0, 1, 1, 2};
constexpr int pb[6] = {1, 2, 3, 2, 3, 3};
constexpr int comp[6] = {5, 4, 3, 2, 1, 0};
constexpr int topology[6] = {0, 1, 2, 2, 1, 0};
struct QuadKey {
    array<Bits, 3> bits;
    bool operator==(const QuadKey &k) const { return bits == k.bits; }
};
struct QuadHash {
    size_t operator()(const QuadKey &k) const {
        BitsHash bh;
        size_t h = bh(k.bits[0]);
        for (int i = 1; i < 3; ++i) h ^= bh(k.bits[i]) + (h << 6) + (h >> 2);
        return h;
    }
};
unordered_map<QuadKey, array<double, 3>, QuadHash> quartet_cache;
long long evaluations = 0;

template<int K>
array<double, 3> hybrid_score(const array<Bits, K> &groups) {
    constexpr int a3[6] = {0, 1, 2, 0, 0, 1};
    constexpr int b3[6] = {0, 1, 2, 1, 2, 2};
    const int *aa = K == 4 ? pa : a3;
    const int *bb = K == 4 ? pb : b3;
    array<double, 3> score{};
    alignas(64) double cnt[MAXV][K], ext[MAXV][K];
    alignas(64) double pairs[MAXV][6], down[MAXV][6];
    alignas(64) double outpairs[MAXV][6], up[MAXV][6];
    int color[MAXN];
    for (int i = 0; i < N; ++i) {
        color[i] = -1;
        for (int c = 0; c < K; ++c) if (groups[c].test(i)) color[i] = c;
        fill(pairs[i], pairs[i] + 6, 0);
        fill(down[i], down[i] + 6, 0);
    }
    for (const auto &g : genes) {
        array<double, 3> gscore{};
        for (int i = 0; i < N; ++i) {
            fill(cnt[i], cnt[i] + K, 0);
            if (color[i] >= 0) cnt[i][color[i]] = g.attenuation[i];
        }
        for (int u = N; u < g.nodes; ++u) {
            int l = g.left[u], r = g.right[u];
            for (int c = 0; c < K; ++c)
                cnt[u][c] = g.attenuation[u] * (cnt[l][c] + cnt[r][c]);
            double f = g.f[u], s = 1 - f;
            for (int p = 0; p < 6; ++p) {
                int a = aa[p], b = bb[p];
                double cross = cnt[l][a] * cnt[r][b];
                if (a != b) cross += cnt[l][b] * cnt[r][a];
                pairs[u][p] = pairs[l][p] + pairs[r][p] + cross;
                down[u][p] = f * (down[l][p] + down[r][p]) + s * pairs[u][p];
            }
        }
        fill(ext[g.root], ext[g.root] + K, 0);
        fill(outpairs[g.root], outpairs[g.root] + 6, 0);
        fill(up[g.root], up[g.root] + 6, 0);
        for (int u = g.root; u >= N; --u) {
            int l = g.left[u], r = g.right[u];
            for (int p = 0; p < 6; ++p) {
                if constexpr (K == 4) {
                    int c = pa[comp[p]], d = pb[comp[p]];
                    gscore[topology[p]] +=
                        down[l][p] * (cnt[r][c] * ext[u][d] + cnt[r][d] * ext[u][c]) +
                        down[r][p] * (cnt[l][c] * ext[u][d] + cnt[l][d] * ext[u][c]) +
                        up[u][p] * (cnt[l][c] * cnt[r][d] + cnt[l][d] * cnt[r][c]);
                } else {
                    if (p < 3) {
                        int c = (p + 1) % 3, d = (p + 2) % 3;
                        gscore[0] +=
                            down[l][p] * (cnt[r][c] * ext[u][d] + cnt[r][d] * ext[u][c]) +
                            down[r][p] * (cnt[l][c] * ext[u][d] + cnt[l][d] * ext[u][c]) +
                            up[u][p] * (cnt[l][c] * cnt[r][d] + cnt[l][d] * cnt[r][c]);
                    } else {
                        int c = 3 - aa[p] - bb[p];
                        gscore[0] += down[l][p] * cnt[r][c] * ext[u][c] +
                                    down[r][p] * cnt[l][c] * ext[u][c] +
                                    up[u][p] * cnt[l][c] * cnt[r][c];
                    }
                }
            }
            for (int side = 0; side < 2; ++side) {
                int v = side ? r : l, sibling = side ? l : r;
                if (v < N) continue;
                for (int c = 0; c < K; ++c)
                    ext[v][c] = g.attenuation[v] * (ext[u][c] + cnt[sibling][c]);
                double f = g.f[v], s = 1 - f;
                for (int p = 0; p < 6; ++p) {
                    int a = aa[p], b = bb[p];
                    double cross = ext[u][a] * cnt[sibling][b];
                    if (a != b) cross += ext[u][b] * cnt[sibling][a];
                    outpairs[v][p] = outpairs[u][p] + pairs[sibling][p] + cross;
                    up[v][p] = f * (up[u][p] + down[sibling][p]) + s * outpairs[v][p];
                }
            }
        }
        for (int k = 0; k < 3; ++k) score[k] += g.weight * gscore[k];
    }
    for (auto &s : score) s *= 0.5;
    return score;
}

array<double, 3> quartet_score(const array<Bits, 4> &groups) {
    QuadKey key{{groups[1], groups[2], groups[3]}};
    auto it = quartet_cache.find(key);
    if (it != quartet_cache.end()) return it->second;
    ++evaluations;
    if (branch_alpha != 0) {
        auto score = hybrid_score<4>(groups);
        quartet_cache.emplace(key, score);
        return score;
    }
    array<double, 3> score{};
    alignas(64) double counts[MAXV][4];
    alignas(64) double down[MAXV][6];
    alignas(64) double up[MAXV][6];
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 4; ++c) counts[i][c] = groups[c].test(i);
        fill(down[i], down[i] + 6, 0);
    }
    double total[4];
    for (int c = 0; c < 4; ++c) total[c] = groups[c].size();
    for (const auto &g : genes) {
        array<double, 3> gscore{};
        for (int u = N; u < g.nodes; ++u) {
            int l = g.left[u], r = g.right[u];
            for (int c = 0; c < 4; ++c) counts[u][c] = counts[l][c] + counts[r][c];
            double f = g.f[u], s = 1 - f;
            for (int p = 0; p < 6; ++p)
                down[u][p] = f * (down[l][p] + down[r][p]) +
                             s * counts[u][pa[p]] * counts[u][pb[p]];
        }
        fill(up[g.root], up[g.root] + 6, 0);
        for (int u = g.root; u >= N; --u) {
            int l = g.left[u], r = g.right[u];
            double outside[4];
            for (int c = 0; c < 4; ++c) outside[c] = total[c] - counts[u][c];
            for (int p = 0; p < 6; ++p) {
                int c = pa[comp[p]], d = pb[comp[p]];
                gscore[topology[p]] +=
                    down[l][p] * (counts[r][c] * outside[d] + counts[r][d] * outside[c]) +
                    down[r][p] * (counts[l][c] * outside[d] + counts[l][d] * outside[c]) +
                    up[u][p] * (counts[l][c] * counts[r][d] + counts[l][d] * counts[r][c]);
            }
            if (l >= N) {
                double f = g.f[l], s = 1 - f;
                for (int p = 0; p < 6; ++p)
                    up[l][p] = f * (up[u][p] + down[r][p]) +
                               s * (total[pa[p]] - counts[l][pa[p]]) *
                                   (total[pb[p]] - counts[l][pb[p]]);
            }
            if (r >= N) {
                double f = g.f[r], s = 1 - f;
                for (int p = 0; p < 6; ++p)
                    up[r][p] = f * (up[u][p] + down[l][p]) +
                               s * (total[pa[p]] - counts[r][pa[p]]) *
                                   (total[pb[p]] - counts[r][pb[p]]);
            }
        }
        for (int k = 0; k < 3; ++k) score[k] += g.weight * gscore[k];
    }
    for (auto &x : score) x *= 0.5;
    quartet_cache.emplace(key, score);
    return score;
}

struct TripKey {
    array<Bits, 2> bits;
    bool operator==(const TripKey &k) const { return bits == k.bits; }
};
struct TripHash {
    size_t operator()(const TripKey &k) const {
        BitsHash bh;
        size_t h = bh(k.bits[0]);
        return h ^ (bh(k.bits[1]) + (h << 6) + (h >> 2));
    }
};
unordered_map<TripKey, double, TripHash> tripart_cache;

double tripart_score(Bits a, Bits b) {
    if (b < a) swap(a, b);
    TripKey key{{a, b}};
    auto it = tripart_cache.find(key);
    if (it != tripart_cache.end()) return it->second;
    array<Bits, 3> groups{{all_taxa - (a | b), a, b}};
    if (branch_alpha != 0) {
        double score = hybrid_score<3>(groups)[0];
        tripart_cache.emplace(key, score);
        return score;
    }
    constexpr int aa[6] = {0, 1, 2, 0, 0, 1};
    constexpr int bb[6] = {0, 1, 2, 1, 2, 2};
    double score = 0;
    alignas(64) double counts[MAXV][3], down[MAXV][6], up[MAXV][6];
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) counts[i][c] = groups[c].test(i);
        fill(down[i], down[i] + 6, 0);
    }
    double total[3];
    for (int c = 0; c < 3; ++c) total[c] = groups[c].size();
    for (const auto &g : genes) {
        double gscore = 0;
        for (int u = N; u < g.nodes; ++u) {
            int l = g.left[u], r = g.right[u];
            for (int c = 0; c < 3; ++c) counts[u][c] = counts[l][c] + counts[r][c];
            double f = g.f[u], s = 1 - f;
            for (int p = 0; p < 6; ++p) {
                double pairs = p < 3 ? counts[u][p] * (counts[u][p] - 1) * 0.5 :
                                      counts[u][aa[p]] * counts[u][bb[p]];
                down[u][p] = f * (down[l][p] + down[r][p]) + s * pairs;
            }
        }
        fill(up[g.root], up[g.root] + 6, 0);
        for (int u = g.root; u >= N; --u) {
            int l = g.left[u], r = g.right[u];
            double out[3];
            for (int c = 0; c < 3; ++c) out[c] = total[c] - counts[u][c];
            for (int p = 0; p < 3; ++p) {
                int c = (p + 1) % 3, d = (p + 2) % 3;
                gscore += down[l][p] * (counts[r][c] * out[d] + counts[r][d] * out[c]) +
                         down[r][p] * (counts[l][c] * out[d] + counts[l][d] * out[c]) +
                         up[u][p] * (counts[l][c] * counts[r][d] + counts[l][d] * counts[r][c]);
            }
            for (int p = 3; p < 6; ++p) {
                int c = 3 - aa[p] - bb[p];
                gscore += down[l][p] * counts[r][c] * out[c] +
                         down[r][p] * counts[l][c] * out[c] +
                         up[u][p] * counts[l][c] * counts[r][c];
            }
            for (int side = 0; side < 2; ++side) {
                int v = side ? r : l, sibling = side ? l : r;
                if (v < N) continue;
                double f = g.f[v], s = 1 - f;
                for (int p = 0; p < 6; ++p) {
                    double ca = total[aa[p]] - counts[v][aa[p]];
                    double cb = total[bb[p]] - counts[v][bb[p]];
                    double pairs = p < 3 ? ca * (ca - 1) * 0.5 : ca * cb;
                    up[v][p] = f * (up[u][p] + down[sibling][p]) + s * pairs;
                }
            }
        }
        score += g.weight * gscore;
    }
    score *= 0.5;
    tripart_cache.emplace(key, score);
    return score;
}

double tree_score(const Tree &t) {
    double score = 0;
    function<Bits(int, int)> visit = [&](int u, int p) {
        Bits b;
        if (u < N) { b.set(u); return b; }
        vector<Bits> children;
        for (int v : t.adj[u]) if (v != p) children.push_back(visit(v, u));
        score += tripart_score(children[0], children[1]);
        return children[0] | children[1];
    };
    visit(t.adj[0][0], 0);
    return score * 0.5;
}

struct Neighborhood {
    array<Bits, 4> groups;
    array<int, 4> branch, owner;
    int current;
};
Neighborhood neighborhood(const Tree &t, int u, int v) {
    vector<pair<pair<int, int>, Bits>> pieces;
    for (int x : t.adj[u]) if (x != v) pieces.push_back({{x, u}, t.component(x, u)});
    for (int x : t.adj[v]) if (x != u) pieces.push_back({{x, v}, t.component(x, v)});
    auto smallest = [](const Bits &b) {
        for (int j = 0; j < WORDS; ++j) if (b.w[j]) return j * 64 + __builtin_ctzll(b.w[j]);
        return MAXN;
    };
    sort(pieces.begin(), pieces.end(), [&](const auto &a, const auto &b) {
        return smallest(a.second) < smallest(b.second);
    });
    Neighborhood n;
    for (int i = 0; i < 4; ++i) {
        n.groups[i] = pieces[i].second;
        n.branch[i] = pieces[i].first.first;
        n.owner[i] = pieces[i].first.second;
    }
    n.current = 0;
    for (int i = 1; i < 4; ++i) if (n.owner[i] == n.owner[0]) n.current = i - 1;
    return n;
}
double nni(Tree &t, int max_rounds = 20, bool verbose = true) {
    double gain = 0;
    for (int round = 0; round < max_rounds && !expired(); ++round) {
        vector<pair<int, int>> edges;
        for (int u = N; u < t.nodes; ++u) for (int v : t.adj[u]) if (v > u) edges.push_back({u, v});
        shuffle(edges.begin(), edges.end(), rng);
        int changed = 0;
        for (auto [u, v] : edges) {
            if (expired()) break;
            if (find(t.adj[u].begin(), t.adj[u].end(), v) == t.adj[u].end()) continue;
            auto n = neighborhood(t, u, v);
            auto score = quartet_score(n.groups);
            int best = int(max_element(score.begin(), score.end()) - score.begin());
            if (score[best] > score[n.current] + 1e-7) {
                int a = n.current + 1, b = best + 1;
                t.swap_branches(n.owner[a], n.branch[a], n.owner[b], n.branch[b]);
                gain += score[best] - score[n.current];
                ++changed;
            }
        }
        if (verbose) cerr << "NNI " << round << " changes=" << changed << " t=" << elapsed() << "\n";
        if (!changed) break;
    }
    return gain;
}

double spr(Tree &t, int radius, int max_size = MAXN) {
    vector<pair<int, int>> prunes;
    for (int u = N; u < t.nodes; ++u) for (int x : t.adj[u])
        if (t.component(x, u).size() <= min(max_size, N / 2)) prunes.push_back({u, x});
    shuffle(prunes.begin(), prunes.end(), rng);
    int changes = 0;
    double total_gain = 0;
    for (auto [u, x] : prunes) {
        if (expired()) break;
        if (find(t.adj[u].begin(), t.adj[u].end(), x) == t.adj[u].end()) continue;
        double best_gain = 1e-5;
        Tree best;
        function<void(int, int, double)> slide = [&](int v, int depth, double gain) {
            if (depth == radius || v < N || expired()) return;
            auto n = neighborhood(t, u, v);
            auto q = quartet_score(n.groups);
            int back = -1;
            for (int y : t.adj[u]) if (y != v && y != x) back = y;
            vector<int> forward;
            for (int y : t.adj[v]) if (y != u) forward.push_back(y);
            for (int y : forward) {
                int cback = -1, cy = -1;
                for (int c = 0; c < 4; ++c) {
                    if (n.branch[c] == back) cback = c;
                    if (n.branch[c] == y) cy = c;
                }
                array<int, 4> owners = n.owner;
                swap(owners[cback], owners[cy]);
                int next = -1;
                for (int c = 1; c < 4; ++c) if (owners[c] == owners[0]) next = c - 1;
                double delta = q[next] - q[n.current];
                t.swap_branches(u, back, v, y);
                double newgain = gain + delta;
                if (newgain > best_gain) { best_gain = newgain; best = t; }
                slide(y, depth + 1, newgain);
                t.swap_branches(u, y, v, back);
            }
        };
        vector<int> directions;
        for (int v : t.adj[u]) if (v != x) directions.push_back(v);
        for (int v : directions) slide(v, 0, 0);
        if (best_gain > 1e-5) {
            t = move(best);
            total_gain += best_gain;
            ++changes;
        }
    }
    cerr << "SPR radius=" << radius << " changes=" << changes << " gain=" << total_gain
         << " t=" << elapsed() << "\n";
    return total_gain;
}

Tree candidate_dp(const vector<Bits> &input, const Tree &fallback,
                  const unordered_map<Bits, double, BitsHash> *frequency = nullptr) {
    unordered_map<Bits, int, BitsHash> ids;
    vector<Bits> clusters;
    auto add = [&](const Bits &b) {
        if (!ids.count(b)) { ids[b] = int(clusters.size()); clusters.push_back(b); }
    };
    for (int i = 1; i < N; ++i) { Bits b; b.set(i); add(b); }
    for (const auto &b : input) if (!b.test(0) && b.size() > 0) add(b);
    for (const auto &b : fallback.splits()) add(b);
    Bits full = all_taxa;
    full.w[0] &= ~1ULL;
    add(full);
    sort(clusters.begin(), clusters.end(), [](const Bits &a, const Bits &b) {
        int x = a.size(), y = b.size();
        return x != y ? x < y : a < b;
    });
    ids.clear();
    int count = int(clusters.size());
    vector<int> size(count);
    for (int i = 0; i < count; ++i) { ids[clusters[i]] = i; size[i] = clusters[i].size(); }
    vector<double> dp(count, -numeric_limits<double>::infinity());
    vector<pair<int, int>> choice(count, {-1, -1});
    long long partitions = 0;
    for (int i = 0; i < count; ++i) {
        if (size[i] == 1) { dp[i] = 0; continue; }
        if (frequency ? elapsed() > 130 : expired()) return fallback;
        const auto &c = clusters[i];
        double reward = 0;
        if (frequency) {
            auto f = frequency->find(c);
            if (f != frequency->end()) reward = f->second;
        }
        for (int j = 0; j < i && size[j] * 2 <= size[i]; ++j) {
            if (!isfinite(dp[j]) || !clusters[j].subset(c)) continue;
            auto it = ids.find(c - clusters[j]);
            if (it == ids.end()) continue;
            int k = it->second;
            if (size[j] == size[k] && j > k) continue;
            if (!isfinite(dp[k])) continue;
            ++partitions;
            double s = dp[j] + dp[k] +
                       (frequency ? reward : tripart_score(clusters[j], clusters[k]));
            if (s > dp[i]) { dp[i] = s; choice[i] = {j, k}; }
        }
    }
    int root = ids.at(full);
    if (!isfinite(dp[root])) return fallback;
    Tree t(N);
    function<int(int)> reconstruct = [&](int c) {
        if (size[c] == 1) {
            for (int i = 1; i < N; ++i) if (clusters[c].test(i)) return i;
        }
        auto [a, b] = choice[c];
        int l = reconstruct(a), r = reconstruct(b);
        int u = t.nodes++;
        t.add(u, l);
        t.add(u, r);
        return u;
    };
    int r = reconstruct(root);
    t.add(0, r);
    cerr << "DP clusters=" << count << " partitions=" << partitions << " score=" << dp[root] * 0.5
         << " t=" << elapsed() << "\n";
    return t;
}

Tree exact_consensus(const vector<Tree> &trees, const Tree &reference) {
    unordered_map<Bits, double, BitsHash> frequency;
    vector<Bits> pool;
    for (const auto &t : trees) for (const auto &b : t.splits()) {
        frequency[b] += 1;
        pool.push_back(b);
    }
    for (const auto &b : reference.splits()) frequency[b] += 0.01;
    return candidate_dp(pool, reference, &frequency);
}

Tree consensus(const vector<Tree> &trees, const Tree &reference) {
    unordered_map<Bits, double, BitsHash> frequency;
    for (const auto &t : trees) for (const auto &b : t.splits()) frequency[b] += 1;
    vector<pair<double, Bits>> ordered;
    for (const auto &item : frequency) ordered.push_back({item.second, item.first});
    sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first > b.first;
        if (a.second.size() != b.second.size()) return a.second.size() < b.second.size();
        return a.second < b.second;
    });
    vector<Bits> compatible;
    auto add = [&](const Bits &b) {
        for (const auto &a : compatible) {
            bool intersection = false;
            for (int i = 0; i < WORDS; ++i) intersection |= bool(a.w[i] & b.w[i]);
            if (intersection && !a.subset(b) && !b.subset(a)) return;
            if (a == b) return;
        }
        compatible.push_back(b);
    };
    for (const auto &p : ordered) add(p.second);
    for (const auto &b : reference.splits()) add(b);
    Bits full = all_taxa;
    full.w[0] &= ~1ULL;
    add(full);
    for (int i = 1; i < N; ++i) { Bits b; b.set(i); add(b); }
    sort(compatible.begin(), compatible.end(), [](const Bits &a, const Bits &b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
    });
    vector<vector<int>> child(compatible.size());
    for (int i = 0; i < int(compatible.size()) - 1; ++i) {
        for (int j = i + 1; j < int(compatible.size()); ++j) if (compatible[i].subset(compatible[j])) {
            child[j].push_back(i);
            break;
        }
    }
    Tree out(N);
    function<int(int)> build = [&](int c) {
        if (compatible[c].size() == 1) {
            for (int i = 1; i < N; ++i) if (compatible[c].test(i)) return i;
        }
        vector<int> parts;
        for (int x : child[c]) parts.push_back(build(x));
        while (parts.size() > 1) {
            int a = parts.back(); parts.pop_back();
            int b = parts.back(); parts.pop_back();
            int u = out.nodes++;
            out.add(u, a);
            out.add(u, b);
            parts.push_back(u);
        }
        return parts[0];
    };
    int r = build(int(compatible.size()) - 1);
    out.add(r, 0);
    return out;
}

Tree bootstrap_consensus(const Tree &base, int replicates, double concentration,
                         bool mixed_parameters = false, bool exact = false) {
    vector<double> original;
    for (const auto &g : genes) original.push_back(g.weight);
    vector<Tree> trees;
    double original_power = support_power, original_quality = quality_power, original_rate = rate_power;
    double mode = 0;
    if (const char *s = getenv("WEIGHT_MODE")) mode = atof(s);
    gamma_distribution<double> gamma(concentration, 1.0 / concentration);
    for (int it = 0; it < replicates && !expired(); ++it) {
        if (mixed_parameters || getenv("BOOT_PARAMS")) {
            support_power = it % 2 ? 0.5 : 1.0;
            quality_power = (it / 2) % 3;
            for (auto &g : genes) g.weights(mode);
            for (auto &g : genes) g.weight *= gamma(rng);
        } else {
            for (int g = 0; g < int(genes.size()); ++g) genes[g].weight = original[g] * gamma(rng);
        }
        quartet_cache.clear();
        tripart_cache.clear();
        Tree t = base;
        if (getenv("BOOT_NJ")) {
            vector<double> gw;
            for (const auto &g : genes) gw.push_back(g.weight);
            t = neighbor_joining(distances(it % 2 ? 0 : 5, gw));
        }
        nni(t, 30, false);
        if (const char *r = getenv("BOOT_SPR")) {
            spr(t, atoi(r));
            nni(t, 20, false);
        }
        trees.push_back(t);
        if ((it + 1) % 10 == 0) cerr << "Resamples=" << it + 1 << " t=" << elapsed() << "\n";
    }
    support_power = original_power;
    quality_power = original_quality;
    rate_power = original_rate;
    for (auto &g : genes) g.weights(mode);
    quartet_cache.clear();
    tripart_cache.clear();
    if (trees.empty()) return base;
    if (exact || getenv("CONSENSUS_DP")) return exact_consensus(trees, base);
    return consensus(trees, base);
}

Tree method_ensemble(const Tree &base, bool fixed_locality = false, bool exact = false) {
    double original_alpha = branch_alpha, original_power = support_power, original_quality = quality_power;
    int original_mode = branch_mode;
    double mode = 0;
    if (const char *s = getenv("WEIGHT_MODE")) mode = atof(s);
    vector<Tree> trees;
    vector<double> alphas{0.3, 0.4, 0.5};
    if (fixed_locality || getenv("ENSEMBLE_FIXED")) alphas = {original_alpha};
    for (double a : alphas) for (double p : {0.5, 1.0}) for (double q : {0.0, 1.0, 2.0}) {
        if (expired()) break;
        branch_alpha = a;
        support_power = p;
        quality_power = q;
        branch_mode = 2;
        for (auto &g : genes) g.weights(mode);
        quartet_cache.clear();
        tripart_cache.clear();
        Tree t = base;
        nni(t, 30, false);
        if (getenv("ENSEMBLE_SPR")) spr(t, 3);
        trees.push_back(t);
        cerr << "Ensemble alpha=" << a << " power=" << p << " quality=" << q
             << " t=" << elapsed() << "\n";
    }
    branch_alpha = original_alpha;
    support_power = original_power;
    quality_power = original_quality;
    branch_mode = original_mode;
    for (auto &g : genes) g.weights(mode);
    quartet_cache.clear();
    tripart_cache.clear();
    return (exact || getenv("CONSENSUS_DP")) ? exact_consensus(trees, base) : consensus(trees, base);
}

void write_tree(const Tree &t, const string &path) {
    if (!t.valid()) throw runtime_error("Invalid inferred topology");
    auto parent = filesystem::path(path).parent_path();
    if (!parent.empty()) filesystem::create_directories(parent);
    ofstream out(path);
    if (!out) throw runtime_error("Cannot open output");
    out << t.newick();
    out.close();
    if (!out) throw runtime_error("Cannot write output");
}

Tree infer(const string &output) {
    for (auto &g : genes) g.weights(0);
    Tree best = neighbor_joining(distances(5));
    write_tree(best, output);
    nni(best, 30, false);
    double best_score = tree_score(best);
    for (int mode : {0, 1, 2, 3, 4, 5, 6, 7}) {
        if (expired()) break;
        Tree t = neighbor_joining(distances(mode));
        nni(t, 30, false);
        double score = tree_score(t);
        if (score > best_score) { best = move(t); best_score = score; }
    }
    write_tree(best, output);
    if (!expired()) best = method_ensemble(best, true, true);
    write_tree(best, output);
    int replicates = 0;
    if (const char *s = getenv("BOOTSTRAPS")) replicates = max(0, atoi(s));
    if (replicates && !expired()) best = bootstrap_consensus(best, replicates, 1.0, true, true);
    return best;
}

int main(int argc, char **argv) {
    started = Clock::now();
    try {
        if (argc < 3) throw runtime_error("Usage: run.sh INPUT OUTPUT");
        bool experimental = getenv("EXPERIMENTAL") != nullptr;
        if (!experimental) {
            branch_alpha = 0.4;
            branch_mode = 2;
            missing_strategy = 2;
            time_limit = 118;
        }
        if (const char *seed = getenv("SEED")) rng.seed(strtoull(seed, nullptr, 10));
        if (const char *s = getenv("SOLVER_TIME")) time_limit = atof(s);
        if (const char *s = getenv("BRANCH_ALPHA")) branch_alpha = atof(s);
        if (const char *s = getenv("BRANCH_MODE")) branch_mode = atoi(s);
        if (const char *s = getenv("SUPPORT_POWER")) support_power = atof(s);
        if (const char *s = getenv("QUALITY_POWER")) quality_power = atof(s);
        if (const char *s = getenv("RATE_POWER")) rate_power = atof(s);
        if (const char *s = getenv("LENGTH_KAPPA")) length_kappa = atof(s);
        if (const char *s = getenv("MISSING_SUPPORT")) missing_strategy = atoi(s);
        labels.push_back("0");
        taxon["0"] = 0;
        ifstream in(argv[1]);
        if (!in) throw runtime_error("Cannot open input");
        string line, text;
        vector<ParsedTree> parsed;
        while (getline(in, line)) {
            text += line;
            if (text.find(';') != string::npos) {
                parsed.push_back(Parser(text).parse());
                text.clear();
            }
        }
        N = int(labels.size());
        if (N > MAXN || N < 4 || parsed.empty()) throw runtime_error("Unsupported input size");
        for (const auto &t : parsed) genes.emplace_back(t);
        for (int i = 0; i < N; ++i) all_taxa.set(i);
        if (!experimental) {
            Tree result = infer(argv[2]);
            write_tree(result, argv[2]);
            cerr << "Done taxa=" << N << " genes=" << genes.size() << " time=" << elapsed() << "\n";
            return 0;
        }
        double mode = 0;
        int dist_mode = 1;
        if (const char *s = getenv("WEIGHT_MODE")) mode = atof(s);
        if (const char *s = getenv("DIST_MODE")) dist_mode = atoi(s);
        for (auto &g : genes) g.weights(mode);
        Matrix dist = distances(dist_mode);
        Tree best = neighbor_joining(dist);
        if (getenv("BME")) bme_nni(best, dist);
        write_tree(best, argv[2]);
        if (!getenv("NO_NNI")) {
            double before = tree_score(best);
            double gain = nni(best);
            double after = tree_score(best);
            cerr << setprecision(14) << "Score " << before << " -> " << after
                 << " change=" << after - before << " gain=" << gain << "\n";
        }
        if (getenv("MULTISTART")) {
            vector<Bits> candidates;
            for (const auto &g : genes) candidates.insert(candidates.end(), g.clades.begin(), g.clades.end());
            double best_score = tree_score(best);
            for (int mode : {0, 1, 2, 3, 4, 5, 6, 7}) {
                Matrix d = distances(mode);
                Tree t = neighbor_joining(d);
                if (getenv("BME")) bme_nni(t, d);
                auto s = t.splits();
                candidates.insert(candidates.end(), s.begin(), s.end());
                nni(t, 30, false);
                s = t.splits();
                candidates.insert(candidates.end(), s.begin(), s.end());
                double score = tree_score(t);
                cerr << "Start " << mode << " score=" << score << " t=" << elapsed() << "\n";
                if (score > best_score) { best = t; best_score = score; }
            }
            if (!getenv("NO_DP")) best = candidate_dp(candidates, best);
        }
        if (const char *s = getenv("SPR_RADIUS")) {
            int rounds = 2;
            if (const char *r = getenv("SPR_ROUNDS")) rounds = atoi(r);
            for (int i = 0; i < rounds && !expired(); ++i) {
                double before = tree_score(best);
                double gain = spr(best, atoi(s));
                double after = tree_score(best);
                cerr << "SPR check change=" << after - before << " gain=" << gain << "\n";
                nni(best, 30, false);
                if (gain < 1e-5) break;
            }
        }
        if (const char *s = getenv("BOOTSTRAPS")) {
            double concentration = 1;
            if (const char *v = getenv("BOOT_CONCENTRATION")) concentration = atof(v);
            best = bootstrap_consensus(best, atoi(s), concentration);
        }
        if (getenv("ENSEMBLE")) best = method_ensemble(best);
        cerr << "Final score=" << tree_score(best) << "\n";
        write_tree(best, argv[2]);
        cerr << "Done taxa=" << N << " genes=" << genes.size() << " evaluations=" << evaluations
             << " time=" << elapsed() << "\n";
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
