#include <algorithm>
#include <array>
#include <chrono>
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
static Clock::time_point started = Clock::now();
static double seconds() {
    return chrono::duration<double>(Clock::now() - started).count();
}

struct Edge {
    int to;
    double length;
    double support;
};

struct Tree {
    int n = 0;
    vector<vector<Edge>> adj;

    explicit Tree(int leaves = 0) : n(leaves), adj(leaves) {}
    int node() {
        adj.emplace_back();
        return int(adj.size()) - 1;
    }
    void connect(int a, int b, double length = 1, double support = -1) {
        adj[a].push_back({b, length, support});
        adj[b].push_back({a, length, support});
    }
    void disconnect(int a, int b) {
        auto remove = [](vector<Edge>& es, int x) {
            es.erase(remove_if(es.begin(), es.end(),
                               [x](const Edge& e) { return e.to == x; }), es.end());
        };
        remove(adj[a], b);
        remove(adj[b], a);
    }
    void swapBranches(int u, int a, int v, int b) {
        disconnect(u, a);
        disconnect(v, b);
        connect(u, b);
        connect(v, a);
    }
    vector<int> side(int u, int p) const {
        vector<int> result;
        function<void(int, int)> visit = [&](int v, int parent) {
            if (v < n) result.push_back(v);
            for (const auto& e : adj[v])
                if (e.to != parent) visit(e.to, v);
        };
        visit(u, p);
        return result;
    }
    string newick(const vector<string>& labels) const {
        int root = n;
        while (root < int(adj.size()) && adj[root].size() != 3) ++root;
        if (root == int(adj.size())) throw runtime_error("No binary tree root");
        function<string(int, int)> write = [&](int v, int p) -> string {
            if (v < n) {
                const string& s = labels[v];
                if (s.find_first_of(" ()[],:;'\t\r\n") == string::npos) return s;
                string q = "'";
                for (char c : s) {
                    q += c;
                    if (c == '\'') q += c;
                }
                return q + "'";
            }
            string s = "(";
            bool first = true;
            for (const auto& e : adj[v]) if (e.to != p) {
                if (!first) s += ",";
                s += write(e.to, v);
                first = false;
            }
            return s + ")";
        };
        return write(root, -1) + ";\n";
    }
    void validate() const {
        int edges = 0, internals = 0;
        for (int v = 0; v < int(adj.size()); ++v) {
            if (v < n && adj[v].size() != 1)
                throw runtime_error("Invalid leaf degree");
            if (v >= n && !adj[v].empty()) {
                if (adj[v].size() != 3) throw runtime_error("Invalid internal degree");
                ++internals;
            }
            edges += int(adj[v].size());
        }
        if (internals != n - 2 || edges != 4 * n - 6)
            throw runtime_error("Invalid binary tree");
        if (int(side(0, -1).size()) != n)
            throw runtime_error("Disconnected tree");
    }
};

class Parser {
    const string& text;
    size_t pos = 0;
    vector<string>& labels;
    unordered_map<string, int>& ids;
    bool addLabels;
    struct Parsed {
        vector<int> children;
        string label;
        double length = 0, support = -1;
    };
    vector<Parsed> nodes;

    void space() {
        while (pos < text.size()) {
            if (isspace(static_cast<unsigned char>(text[pos]))) ++pos;
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
    string token() {
        space();
        string s;
        if (pos < text.size() && text[pos] == '\'') {
            ++pos;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '\'') {
                    if (pos < text.size() && text[pos] == '\'') ++pos;
                    else break;
                }
                s += c;
            }
        } else {
            while (pos < text.size() &&
                   !isspace(static_cast<unsigned char>(text[pos])) &&
                   string("(),:;[]").find(text[pos]) == string::npos)
                s += text[pos++];
        }
        return s;
    }
    int subtree() {
        space();
        int v = int(nodes.size());
        nodes.emplace_back();
        if (pos < text.size() && text[pos] == '(') {
            ++pos;
            vector<int> ch;
            ch.push_back(subtree());
            space();
            while (pos < text.size() && text[pos] == ',') {
                ++pos;
                ch.push_back(subtree());
                space();
            }
            if (pos >= text.size() || text[pos] != ')')
                throw runtime_error("Malformed Newick");
            ++pos;
            nodes[v].children = move(ch);
            string s = token();
            if (!s.empty()) {
                char* end;
                double x = strtod(s.c_str(), &end);
                if (*end == '\0' && isfinite(x)) nodes[v].support = x;
            }
        } else {
            string s = token();
            if (s.empty()) throw runtime_error("Missing leaf label");
            nodes[v].label = s;
            if (!ids.count(s)) {
                if (!addLabels) throw runtime_error("Different gene-tree taxa");
                int id = int(labels.size());
                ids[s] = id;
                labels.push_back(s);
            }
        }
        space();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            string s = token();
            char* end;
            nodes[v].length = strtod(s.c_str(), &end);
            if (*end != '\0' || !isfinite(nodes[v].length) || nodes[v].length < 0)
                throw runtime_error("Invalid branch length");
        }
        return v;
    }

public:
    Parser(const string& s, vector<string>& names, unordered_map<string, int>& map,
           bool allowNew) : text(s), labels(names), ids(map), addLabels(allowNew) {}
    Tree parse() {
        int root = subtree();
        space();
        if (pos >= text.size() || text[pos] != ';')
            throw runtime_error("Missing Newick terminator");
        ++pos;
        space();
        if (pos != text.size()) throw runtime_error("Extra Newick text");
        Tree t(int(labels.size()));
        vector<int> index(nodes.size(), -1);
        vector<int> seen(t.n, 0);
        for (int v = 0; v < int(nodes.size()); ++v) {
            if (nodes[v].children.empty()) {
                index[v] = ids.at(nodes[v].label);
                if (seen[index[v]]++) throw runtime_error("Duplicate leaf");
            } else index[v] = t.node();
        }
        if (find(seen.begin(), seen.end(), 0) != seen.end())
            throw runtime_error("Missing taxa");
        for (int v = 0; v < int(nodes.size()); ++v)
            for (int c : nodes[v].children)
                t.connect(index[v], index[c], nodes[c].length, nodes[c].support);
        int r = index[root];
        if (t.adj[r].size() == 2) {
            Edge a = t.adj[r][0], b = t.adj[r][1];
            t.disconnect(r, a.to);
            t.disconnect(r, b.to);
            double support = max(a.support, b.support);
            t.connect(a.to, b.to, a.length + b.length, support);
        }
        t.validate();
        return t;
    }
};

static pair<double, bool> supportCalibration(const Tree& t) {
    vector<double> rates;
    bool positiveLengths = false;
    for (int v = t.n; v < int(t.adj.size()); ++v)
        for (const Edge& e : t.adj[v]) if (e.to > v && e.to >= t.n) {
            if (e.length > 1e-5) positiveLengths = true;
            if (e.support < 0 || e.length <= 1e-5) continue;
            double q = max(0.0, min(1.0, (3 * e.support - 1) / 2));
            if (q > .05 && q < .995) rates.push_back(-log1p(-q) / e.length);
        }
    double scale = 150;
    if (!rates.empty()) {
        nth_element(rates.begin(), rates.begin() + rates.size() / 2, rates.end());
        scale = max(20.0, min(2000.0, rates[rates.size() / 2]));
    }
    return {scale, positiveLengths};
}

static double supportReliability(double support, double length,
                                 double scale, bool positiveLengths) {
    if (support >= 0) return max(0.0, min(1.0, (3 * support - 1) / 2));
    if (!positiveLengths) return 1;
    if (length <= 1e-5) return 0;
    return -expm1(-scale * length);
}

struct Gene {
    int n, m, root;
    vector<int> order, parent;
    vector<array<int, 2>> children;
    vector<double> failure, length, support;
    double missingScale = 150;
    bool positiveLengths = false;
    array<int, 3> rootChildren{};

    explicit Gene(const Tree& t) : n(t.n), m(int(t.adj.size())),
        parent(m, -2), children(m, {-1, -1}), failure(m, 1),
        length(m, 0), support(m, -1) {
        root = n;
        while (root < m && t.adj[root].size() != 3) ++root;
        function<void(int, int)> visit = [&](int v, int p) {
            parent[v] = p;
            order.push_back(v);
            int k = 0;
            for (const auto& e : t.adj[v]) if (e.to != p) {
                if (v == root) rootChildren[k++] = e.to;
                else children[v][k++] = e.to;
                length[e.to] = e.length;
                support[e.to] = e.support;
                visit(e.to, v);
            }
        };
        visit(root, -1);
        tie(missingScale, positiveLengths) = supportCalibration(t);
    }
    void weights(int mode, double power, double scale, double threshold = -1) {
        for (int v : order) {
            if (v == root || v < n) { failure[v] = 1; continue; }
            double q;
            if (mode == 0) q = 1;
            else if (mode == 1 || mode == 4) {
                q = supportReliability(support[v], length[v], missingScale, positiveLengths);
                if (threshold >= 0) q = q >= (3 * threshold - 1) / 2 ? 1 : 0;
                q = pow(q, power);
            } else if (mode == 2) q = -expm1(-scale * length[v]);
            else {
                double s = supportReliability(support[v], length[v], missingScale, positiveLengths);
                double l = -expm1(-scale * length[v]);
                q = 1 - (1 - pow(s, power)) * (1 - l);
            }
            failure[v] = 1 - q;
        }
    }
};

using Matrix = vector<vector<double>>;

struct Bits {
    array<uint64_t, 4> words{};
    bool operator==(const Bits& b) const { return words == b.words; }
    bool operator<(const Bits& b) const { return words < b.words; }
    void set(int x) { words[x >> 6] |= uint64_t(1) << (x & 63); }
    bool has(int x) const { return (words[x >> 6] >> (x & 63)) & 1; }
    int size() const {
        int n = 0;
        for (uint64_t w : words) n += __builtin_popcountll(w);
        return n;
    }
    bool subset(const Bits& b) const {
        for (int i = 0; i < 4; ++i) if (words[i] & ~b.words[i]) return false;
        return true;
    }
    Bits operator|(const Bits& b) const {
        Bits out;
        for (int i = 0; i < 4; ++i) out.words[i] = words[i] | b.words[i];
        return out;
    }
    Bits operator-(const Bits& b) const {
        Bits out;
        for (int i = 0; i < 4; ++i) out.words[i] = words[i] & ~b.words[i];
        return out;
    }
};

static uint64_t mixHash(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct BitsHash {
    size_t operator()(const Bits& b) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t w : b.words) h = mixHash(h ^ w);
        return h;
    }
};

struct PartitionKey {
    array<Bits, 4> parts{};
    bool operator==(const PartitionKey& b) const { return parts == b.parts; }
};

struct PartitionHash {
    size_t operator()(const PartitionKey& b) const {
        uint64_t h = 0;
        for (const Bits& p : b.parts)
            for (uint64_t w : p.words) h = mixHash(h ^ w);
        return h;
    }
};

static Matrix distances(const vector<Tree>& genes, int mode, double power,
                         int anchor, const vector<double>* weights,
                         double zeroCut = 0);

static double poissonShift(const vector<double>& histogram, double coalescence) {
    double mass = 0, mean = 0;
    for (int k = 0; k < int(histogram.size()); ++k) {
        mass += histogram[k];
        mean += k * histogram[k];
    }
    if (mass <= 0) return 0;
    mean /= mass;
    if (coalescence <= 1e-8) return mean;
    double q = coalescence / (1 + coalescence);
    if ((mass - histogram[0]) / q <= mass) return 0;
    double low = 0, high = mean;
    for (int iteration = 0; iteration < 24; ++iteration) {
        double a = .5 * (low + high), z = a / q;
        double last = 1, derivative = -mass;
        for (int k = 1; k < int(histogram.size()); ++k) {
            double ratio = 1 / (1 + z * last / k);
            last = 1 - ratio;
            derivative += histogram[k] * ratio / q;
        }
        if (derivative > 0) low = a;
        else high = a;
    }
    return .5 * (low + high);
}

static Matrix coalescentDistances(const vector<Tree>& genes, int mode, int anchor,
                                  const vector<double>* weights) {
    int n = genes[0].n, k = int(genes.size()), pairs = n * (n - 1) / 2;
    double sites = mode == 11 ? 150 : mode == 12 ? 250 : 200;
    vector<uint16_t> observations(pairs * k);
    bool saturated = false;
    for (int gi = 0; gi < k; ++gi) {
        const Tree& t = genes[gi];
        for (int a = 0; a < n; ++a) {
            function<void(int, int, double)> visit = [&](int v, int p, double path) {
                if (v < n && v > a) {
                    int pair = a * (2 * n - a - 1) / 2 + v - a - 1;
                    if (!isfinite(path) || sites * path > 500) saturated = true;
                    else observations[pair * k + gi] = uint16_t(lround(sites * path));
                }
                for (Edge e : t.adj[v]) if (e.to != p) visit(e.to, v, path + e.length);
            };
            visit(a, -1, 0);
        }
    }
    if (saturated) return distances(genes, 3, 1, anchor, weights);
    vector<double> w(k, 1);
    if (weights) w = *weights;
    double mass = accumulate(w.begin(), w.end(), 0.0), squared = 0;
    for (double value : w) squared += value * value;
    if (mass <= 0) return distances(genes, 3, 1, anchor, nullptr);
    double correction = mass * mass > squared + 1e-8 ?
                        mass * mass / (mass * mass - squared) : 1;
    vector<double> residual(pairs);
    for (int p = 0; p < pairs; ++p) {
        double mean = 0, second = 0;
        for (int gi = 0; gi < k; ++gi) {
            double value = observations[p * k + gi];
            mean += w[gi] * value;
            second += w[gi] * value * value;
        }
        mean /= mass;
        double variance = max(0.0, second / mass - mean * mean) * correction;
        residual[p] = variance - mean;
    }
    nth_element(residual.begin(), residual.begin() + pairs / 2, residual.end());
    double coalescence = sqrt(max(0.0, residual[pairs / 2]));
    cerr << "Pairwise coalescent scale: " << coalescence << " substitutions / locus\n";
    if (coalescence < .5 || coalescence > 50)
        return distances(genes, 3, 1, anchor, weights);
    Matrix result(n, vector<double>(n));
    int pair = 0;
    vector<double> histogram;
    for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b, ++pair) {
        int maximum = 0;
        for (int gi = 0; gi < k; ++gi) maximum = max(maximum, int(observations[pair * k + gi]));
        histogram.assign(maximum + 1, 0);
        for (int gi = 0; gi < k; ++gi) histogram[observations[pair * k + gi]] += w[gi];
        result[a][b] = result[b][a] = poissonShift(histogram, coalescence) / sites;
    }
    return result;
}

static Matrix clockDistances(const vector<Tree>& genes, int mode, int anchor,
                             const vector<double>* weights = nullptr) {
    int n = genes[0].n;
    Matrix d(n, vector<double>(n));
    for (int gi = 0; gi < int(genes.size()); ++gi) {
        const Tree& t = genes[gi];
        double geneWeight = weights ? (*weights)[gi] : 1;
        if (geneWeight == 0) continue;
        int m = int(t.adj.size()), root = t.adj[anchor][0].to;
        vector<int> parent(m, -1), order;
        vector<array<int, 2>> children(m, {-1, -1});
        vector<double> length(m), height(m), aa(m), bb(m), ma(m), mb(m);
        vector<vector<int>> leaves(m);
        function<void(int, int)> visit = [&](int v, int p) {
            order.push_back(v);
            parent[v] = p;
            int k = 0;
            for (Edge e : t.adj[v]) if (e.to != p) {
                children[v][k++] = e.to;
                length[e.to] = e.length;
                visit(e.to, v);
            }
        };
        visit(root, anchor);
        for (int oi = int(order.size()) - 1; oi >= 0; --oi) {
            int v = order[oi];
            if (v < n) {
                leaves[v].push_back(v);
                ma[v] = 1;
                mb[v] = length[v];
                continue;
            }
            int a = children[v][0], b = children[v][1];
            leaves[v] = leaves[a];
            leaves[v].insert(leaves[v].end(), leaves[b].begin(), leaves[b].end());
            if (mode == 6 || mode == 9) {
                aa[v] = ma[a] + ma[b];
                bb[v] = mb[a] + mb[b];
                ma[v] = aa[v] / (1 + aa[v]);
                mb[v] = (bb[v] + aa[v] * length[v]) / (1 + aa[v]);
            } else {
                height[v] = ((height[a] + length[a]) * leaves[a].size() +
                             (height[b] + length[b]) * leaves[b].size()) / leaves[v].size();
            }
        }
        if (mode == 6 || mode == 9) {
            height[root] = bb[root] / aa[root];
            for (int v : order) if (v >= n && v != root)
                height[v] = max(0.0, (height[parent[v]] - length[v] + bb[v]) / (1 + aa[v]));
        }
        for (int oi = int(order.size()) - 1; oi >= 0; --oi) {
            int v = order[oi];
            if (v >= n) height[v] = max(height[v], max(height[children[v][0]],
                                                       height[children[v][1]]));
        }
        double scale = 1;
        if ((mode == 8 || mode == 9) && height[root] > 1e-12) scale = 1 / height[root];
        for (int v : order) if (v >= n) {
            int a = children[v][0], b = children[v][1];
            for (int i : leaves[a]) for (int j : leaves[b]) {
                double value = 2 * height[v] * scale * geneWeight / genes.size();
                d[i][j] += value;
                d[j][i] += value;
            }
        }
        double outgroup = max(t.adj[anchor][0].length + height[root], 2.2 * height[root]);
        for (int x = 0; x < n; ++x) if (x != anchor) {
            d[x][anchor] += outgroup * scale * geneWeight / genes.size();
            d[anchor][x] = d[x][anchor];
        }
    }
    return d;
}

static Matrix distances(const vector<Tree>& genes, int mode, double power = 1,
                         int anchor = 0, const vector<double>* weights = nullptr,
                         double zeroCut) {
    if (mode >= 10 && mode <= 12) return coalescentDistances(genes, mode, anchor, weights);
    if (mode >= 5 && mode <= 9) return clockDistances(genes, mode, anchor, weights);
    int n = genes[0].n;
    Matrix d(n, vector<double>(n, 0));
    for (int gi = 0; gi < int(genes.size()); ++gi) {
        const Tree& t = genes[gi];
        double geneWeight = weights ? (*weights)[gi] : 1;
        if (geneWeight == 0) continue;
        double scale = 1;
        auto [missingScale, positiveLengths] = supportCalibration(t);
        if (mode == 2 || mode == 3) {
            vector<double> lengths;
            for (int v = n; v < int(t.adj.size()); ++v)
                for (const Edge& e : t.adj[v])
                    if (e.to > v && e.length > 0.00001) lengths.push_back(e.length);
            if (!lengths.empty()) {
                nth_element(lengths.begin(), lengths.begin() + lengths.size() / 2,
                            lengths.end());
                scale = 1 / lengths[lengths.size() / 2];
            }
        }
        for (int a = 0; a < n; ++a) {
            function<void(int, int, double)> visit = [&](int v, int p, double x) {
                if (v < n) d[a][v] += x * geneWeight / genes.size();
                for (const Edge& e : t.adj[v]) if (e.to != p) {
                    double w = 0;
                    if (mode == 0) w = 1;
                    else if (mode == 1) {
                        w = (v < n || e.to < n) ? 1 :
                            pow(supportReliability(e.support, e.length, missingScale,
                                                   positiveLengths), power);
                    } else if (mode == 2) w = e.length * scale;
                    else if (mode == 3) {
                        w = (v >= n && e.to >= n && e.length <= zeroCut) ? 0 :
                            sqrt(e.length * scale);
                    }
                    else w = e.length;
                    visit(e.to, v, x + w);
                }
            };
            visit(a, -1, 0);
        }
    }
    return d;
}

static Tree neighborJoining(const Matrix& input, bool bio = false) {
    int n = int(input.size()), maxn = 2 * n;
    Tree t(n);
    Matrix d(maxn, vector<double>(maxn)), var(maxn, vector<double>(maxn));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) d[i][j] = var[i][j] = input[i][j];
    vector<int> active(n);
    iota(active.begin(), active.end(), 0);
    while (active.size() > 3) {
        int r = int(active.size());
        vector<double> sum(maxn, 0);
        for (int i : active) for (int j : active) sum[i] += d[i][j];
        double best = numeric_limits<double>::infinity();
        int a = -1, b = -1;
        for (int i = 0; i < r; ++i)
            for (int j = i + 1; j < r; ++j) {
                int u = active[i], v = active[j];
                double q = (r - 2) * d[u][v] - sum[u] - sum[v];
                if (q < best) { best = q; a = u; b = v; }
            }
        int u = t.node();
        double la = .5 * (d[a][b] + (sum[a] - sum[b]) / (r - 2));
        double lb = d[a][b] - la;
        t.connect(u, a, max(0.0, la));
        t.connect(u, b, max(0.0, lb));
        double lambda = .5;
        if (bio && var[a][b] > 1e-12) {
            double s = 0;
            for (int v : active) if (v != a && v != b) s += var[b][v] - var[a][v];
            lambda = max(0.0, min(1.0, .5 + s / (2 * (r - 2) * var[a][b])));
        }
        vector<int> next;
        for (int v : active) if (v != a && v != b) {
            d[u][v] = d[v][u] =
                lambda * (d[a][v] - la) + (1 - lambda) * (d[b][v] - lb);
            var[u][v] = var[v][u] = lambda * var[a][v] +
                (1 - lambda) * var[b][v] - lambda * (1 - lambda) * var[a][b];
            next.push_back(v);
        }
        next.push_back(u);
        active = move(next);
    }
    int root = t.node();
    for (int a : active) t.connect(root, a);
    t.validate();
    return t;
}

static Tree upgma(const Matrix& input) {
    int n = int(input.size());
    Tree t(n);
    Matrix d(2 * n, vector<double>(2 * n));
    vector<int> active(n), size(2 * n, 1);
    iota(active.begin(), active.end(), 0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) d[i][j] = input[i][j];
    while (active.size() > 2) {
        int a = -1, b = -1;
        double best = numeric_limits<double>::infinity();
        for (int i = 0; i < int(active.size()); ++i)
            for (int j = i + 1; j < int(active.size()); ++j) {
                int u = active[i], v = active[j];
                if (d[u][v] < best) { best = d[u][v]; a = u; b = v; }
            }
        int u = t.node();
        size[u] = size[a] + size[b];
        t.connect(u, a);
        t.connect(u, b);
        vector<int> next;
        for (int v : active) if (v != a && v != b) {
            d[u][v] = d[v][u] = (size[a] * d[a][v] + size[b] * d[b][v]) / size[u];
            next.push_back(v);
        }
        next.push_back(u);
        active = move(next);
    }
    t.connect(active[0], active[1]);
    t.validate();
    return t;
}

static array<double, 3> localDistanceScores(const Tree& t, const Matrix& d,
                                           int u, const array<int, 2>& a,
                                           int v, const array<int, 2>& b,
                                           bool uniform = false) {
    using Profile = vector<pair<int, double>>;
    auto profile = [&](int start, int parent) {
        Profile result;
        function<void(int, int, double)> visit = [&](int v, int p, double weight) {
            if (v < t.n) result.emplace_back(v, weight);
            else for (Edge e : t.adj[v]) if (e.to != p) visit(e.to, v, .5 * weight);
        };
        visit(start, parent, 1);
        if (uniform)
            for (auto& entry : result) entry.second = 1.0 / result.size();
        return result;
    };
    auto mean = [&](const Profile& a, const Profile& b) {
        double value = 0;
        for (auto [i, wi] : a) for (auto [j, wj] : b) value += wi * wj * d[i][j];
        return value;
    };
    auto pa = profile(a[0], u), pb = profile(a[1], u);
    auto pc = profile(b[0], v), pd = profile(b[1], v);
    return {mean(pa, pb) + mean(pc, pd), mean(pa, pc) + mean(pb, pd),
            mean(pa, pd) + mean(pb, pc)};
}

static void optimizeBME(Tree& t, const Matrix& d, int maxSweeps = 20) {
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        vector<pair<int, int>> edges;
        for (int u = t.n; u < int(t.adj.size()); ++u)
            for (Edge e : t.adj[u]) if (e.to > u && e.to >= t.n) edges.emplace_back(u, e.to);
        int changed = 0;
        for (auto [u, v] : edges) {
            if (none_of(t.adj[u].begin(), t.adj[u].end(),
                        [v](const Edge& e) { return e.to == v; })) continue;
            array<int, 2> a{}, b{};
            int k = 0;
            for (Edge e : t.adj[u]) if (e.to != v) a[k++] = e.to;
            k = 0;
            for (Edge e : t.adj[v]) if (e.to != u) b[k++] = e.to;
            auto score = localDistanceScores(t, d, u, a, v, b);
            int best = int(min_element(score.begin(), score.end()) - score.begin());
            if (score[best] + 1e-10 < score[0]) {
                t.swapBranches(u, a[1], v, b[best == 1 ? 0 : 1]);
                ++changed;
            }
        }
        if (!changed) break;
    }
}

class QuartetScorer {
    vector<Gene>& genes;
    double externalAlpha;
    bool hybrid;
    double hybridScale;
    int maximumBins;
    double normalization;
    double tipAlpha;
    struct Work {
        vector<array<int, 4>> count;
        vector<array<double, 6>> up, down;
        explicit Work(int m) : count(m), up(m), down(m) {}
    };
    struct ExternalWork {
        vector<array<double, 4>> singleUp, singleDown;
        vector<array<double, 6>> pairUp, failUp, pairDown, failDown;
        vector<array<double, 6>> lengthUp, bothUp, lengthDown, bothDown;
        vector<double> attenuation, lengthFailure;
        ExternalWork(const Gene& g, double alpha, bool hybrid, double scale,
                     double tips, int protectedLeaf) :
            singleUp(g.m), singleDown(g.m), pairUp(g.m), failUp(g.m),
            pairDown(g.m), failDown(g.m), attenuation(g.m), lengthFailure(g.m) {
            if (hybrid) {
                lengthUp.resize(g.m);
                bothUp.resize(g.m);
                lengthDown.resize(g.m);
                bothDown.resize(g.m);
            }
            for (int v = 0; v < g.m; ++v) {
                attenuation[v] = exp(-alpha * g.length[v]);
                if (v < g.n && v != protectedLeaf)
                    attenuation[v] *= exp(-tips * g.length[v]);
                lengthFailure[v] = exp(-scale * g.length[v]);
            }
        }
    };
    struct MaximumWork {
        int width;
        int leafScale;
        vector<int> up, down, level, leafWeight;
        vector<double> lengthUp, lengthDown, lengthFailure;
        vector<array<double, 6>> valueUp, valueDown;
        MaximumWork(const Gene& g, int bins, bool hybrid, double scale,
                    double tips, int protectedLeaf) :
            width(bins + 1), up(g.m * 6 * width), down(g.m * 6 * width),
            level(g.m), leafWeight(g.n), lengthFailure(g.m), valueUp(g.m), valueDown(g.m) {
            leafScale = tips > 0 ? 256 : 1;
            if (hybrid) {
                lengthUp.resize(g.m * 6 * width);
                lengthDown.resize(g.m * 6 * width);
            }
            for (int v = 0; v < g.m; ++v) {
                level[v] = max(0, min(bins, int(lround((1 - g.failure[v]) * bins))));
                lengthFailure[v] = exp(-scale * g.length[v]);
            }
            for (int v = 0; v < g.n; ++v)
                leafWeight[v] = v == protectedLeaf ? leafScale :
                    max(0, min(leafScale, int(lround(leafScale * exp(-tips * g.length[v])))));
        }
    };
    vector<Work> work;
    vector<ExternalWork> externalWork;
    vector<MaximumWork> maximumWork;
    struct FourRecord {
        array<double, 3> total;
        vector<array<double, 3>> perGene;
    };
    struct ThreeRecord {
        double total;
        double all;
        double divisor;
        vector<double> perGene;
        vector<double> perGeneAll;
    };
    unordered_map<PartitionKey, FourRecord, PartitionHash> fourCache;
    unordered_map<PartitionKey, ThreeRecord, PartitionHash> threeCache;
    vector<double> locusWeights;
    bool reweighted = false;
    static constexpr int pa[6] = {0, 0, 0, 1, 1, 2};
    static constexpr int pb[6] = {1, 2, 3, 2, 3, 3};

    array<double, 3> weighted(const FourRecord& r) const {
        if (!reweighted) return r.total;
        array<double, 3> result{};
        for (int i = 0; i < int(r.perGene.size()); ++i)
            for (int q = 0; q < 3; ++q) result[q] += locusWeights[i] * r.perGene[i][q];
        return result;
    }
    double weighted(const ThreeRecord& r) const {
        double result = r.total, all = r.all;
        if (reweighted) {
            result = 0;
            all = 0;
            for (int i = 0; i < int(r.perGene.size()); ++i) {
                result += locusWeights[i] * r.perGene[i];
                if (normalization > 0) all += locusWeights[i] * r.perGeneAll[i];
            }
        }
        return normalization > 0 ? (result - all / 3) / r.divisor : result;
    }
    array<double, 3> saveFour(const PartitionKey& key, const array<int, 3>& qrank,
                              const array<double, 3>& total,
                              vector<array<double, 3>> perGene) {
        FourRecord rec;
        for (int q = 0; q < 3; ++q) rec.total[qrank[q]] = total[q];
        for (auto& contribution : perGene) {
            auto original = contribution;
            for (int q = 0; q < 3; ++q) contribution[qrank[q]] = original[q];
        }
        rec.perGene = move(perGene);
        auto [it, inserted] = fourCache.emplace(key, move(rec));
        (void)inserted;
        auto resolved = weighted(it->second);
        return {resolved[qrank[0]], resolved[qrank[1]], resolved[qrank[2]]};
    }
    double saveThree(const PartitionKey& key, double total, vector<double> perGene,
                     double all = 0, vector<double> perGeneAll = {}) {
        double a = key.parts[0].size(), b = key.parts[1].size(), c = key.parts[2].size();
        double count = .5 * (a * (a - 1) * b * c + b * (b - 1) * a * c +
                            c * (c - 1) * a * b);
        double divisor = normalization > 0 ? pow(max(1.0, count), normalization) : 1;
        auto [it, inserted] = threeCache.emplace(key,
            ThreeRecord{total, all, divisor, move(perGene), move(perGeneAll)});
        (void)inserted;
        return weighted(it->second);
    }

    template<int K>
    array<double, 3> scoreMaximum(const vector<int>& color,
                                 vector<array<double, 3>>& perGene) {
        constexpr array<int, 6> ax = {0, 0, 0, 1, 1, 2};
        constexpr array<int, 6> by = K == 4 ? array<int, 6>{1, 2, 3, 2, 3, 3} :
                                                          array<int, 6>{0, 1, 2, 1, 2, 2};
        constexpr array<int, 3> repeated = {0, 3, 5};
        constexpr array<int, 3> opposite = {4, 2, 1};
        array<double, 3> scores{};
        for (int gi = 0; gi < int(genes.size()); ++gi) {
            auto previous = scores;
            const Gene& g = genes[gi];
            auto& count = work[gi].count;
            MaximumWork& w = maximumWork[gi];
            array<int, 4> totals{};
            for (int v = 0; v < g.n; ++v)
                if (color[v] >= 0) totals[color[v]] += w.leafWeight[v];
            auto merge = [&](int target, bool targetDown, int a, bool aDown,
                             int b, bool bDown, const array<int, 4>& ca,
                             const array<int, 4>& cb) {
                int level = w.level[target], width = w.width;
                double attenuation = w.lengthFailure[target];
                auto& value = targetDown ? w.valueDown[target] : w.valueUp[target];
                for (int p = 0; p < 6; ++p) {
                    const int* ha = (aDown ? w.down.data() : w.up.data()) + (a * 6 + p) * width;
                    const int* hb = (bDown ? w.down.data() : w.up.data()) + (b * 6 + p) * width;
                    int* out = (targetDown ? w.down.data() : w.up.data()) + (target * 6 + p) * width;
                    const double* fa = nullptr;
                    const double* fb = nullptr;
                    double* fout = nullptr;
                    if (hybrid) {
                        fa = (aDown ? w.lengthDown.data() : w.lengthUp.data()) + (a * 6 + p) * width;
                        fb = (bDown ? w.lengthDown.data() : w.lengthUp.data()) + (b * 6 + p) * width;
                        fout = (targetDown ? w.lengthDown.data() : w.lengthUp.data()) +
                               (target * 6 + p) * width;
                    }
                    int cross = ca[ax[p]] * cb[by[p]];
                    if (ax[p] != by[p]) cross += cb[ax[p]] * ca[by[p]];
                    int low = cross;
                    double lowFailure = cross;
                    for (int bin = 0; bin < width; ++bin) {
                        if (bin < level) {
                            low += ha[bin] + hb[bin];
                            out[bin] = 0;
                            if (hybrid) {
                                lowFailure += fa[bin] + fb[bin];
                                fout[bin] = 0;
                            }
                        } else {
                            out[bin] = ha[bin] + hb[bin];
                            if (hybrid) fout[bin] = attenuation * (fa[bin] + fb[bin]);
                        }
                    }
                    out[level] += low;
                    if (hybrid) fout[level] += attenuation * lowFailure;
                    double expected = 0;
                    for (int bin = 1; bin < width; ++bin)
                        expected += bin * (hybrid ? out[bin] - fout[bin] : out[bin]);
                    value[p] = expected / maximumBins;
                }
            };
            for (int v = 0; v < g.n; ++v) {
                count[v] = {0, 0, 0, 0};
                if (color[v] >= 0) count[v][color[v]] = w.leafWeight[v];
            }
            for (int oi = int(g.order.size()) - 1; oi > 0; --oi) {
                int v = g.order[oi];
                if (v < g.n) continue;
                int a = g.children[v][0], b = g.children[v][1];
                for (int c = 0; c < K; ++c) count[v][c] = count[a][c] + count[b][c];
                merge(v, false, a, false, b, false, count[a], count[b]);
            }
            for (int v : g.order) {
                if (v < g.n) continue;
                array<int, 3> source{};
                array<bool, 3> isDown{};
                array<array<int, 4>, 3> c{};
                array<array<double, 6>, 3> pair{};
                if (v == g.root) source = g.rootChildren;
                else {
                    source = {g.children[v][0], g.children[v][1], v};
                    isDown[2] = true;
                }
                for (int i = 0; i < 3; ++i) {
                    if (isDown[i]) {
                        for (int j = 0; j < K; ++j) c[i][j] = totals[j] - count[v][j];
                        pair[i] = w.valueDown[v];
                    } else {
                        c[i] = count[source[i]];
                        pair[i] = w.valueUp[source[i]];
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    int i = (k + 1) % 3, j = (k + 2) % 3;
                    array<int, 6> cross{};
                    for (int p = 0; p < 6; ++p) {
                        cross[p] = c[i][ax[p]] * c[j][by[p]];
                        if (ax[p] != by[p]) cross[p] += c[j][ax[p]] * c[i][by[p]];
                    }
                    if constexpr (K == 4) {
                        scores[0] += cross[0] * pair[k][5] + cross[5] * pair[k][0];
                        scores[1] += cross[1] * pair[k][4] + cross[4] * pair[k][1];
                        scores[2] += cross[2] * pair[k][3] + cross[3] * pair[k][2];
                    } else {
                        double agree = 0;
                        for (int p = 0; p < 3; ++p)
                            agree += cross[repeated[p]] * pair[k][opposite[p]] +
                                     cross[opposite[p]] * pair[k][repeated[p]];
                        scores[0] += agree;
                        if (normalization > 0)
                            scores[1] += agree + cross[1] * pair[k][2] +
                                cross[2] * pair[k][1] + cross[1] * pair[k][4] +
                                cross[4] * pair[k][1] + cross[2] * pair[k][4] +
                                cross[4] * pair[k][2];
                    }
                    if (v == g.root || k < 2) {
                        int child = source[k];
                        if (child >= g.n)
                            merge(child, true, source[i], isDown[i], source[j],
                                  isDown[j], c[i], c[j]);
                    }
                }
            }
            double factor = double(w.leafScale) * w.leafScale * w.leafScale * w.leafScale;
            for (int q = 0; q < 3; ++q)
                perGene[gi][q] = .5 * (scores[q] - previous[q]) / factor;
        }
        double scale = tipAlpha > 0 ? 256 : 1;
        double factor = scale * scale * scale * scale;
        for (double& x : scores) x *= .5 / factor;
        return scores;
    }

    template<int K>
    array<double, 3> scoreExternal(const vector<int>& color,
                                   vector<array<double, 3>>& perGene) {
        constexpr array<int, 6> ax = {0, 0, 0, 1, 1, 2};
        constexpr array<int, 6> by = K == 4 ? array<int, 6>{1, 2, 3, 2, 3, 3} :
                                                          array<int, 6>{0, 1, 2, 1, 2, 2};
        constexpr array<int, 3> repeated = {0, 3, 5};
        constexpr array<int, 3> opposite = {4, 2, 1};
        array<double, 3> scores{};
        for (int gi = 0; gi < int(genes.size()); ++gi) {
            auto previous = scores;
            const Gene& g = genes[gi];
            ExternalWork& w = externalWork[gi];
            for (int v = 0; v < g.n; ++v) {
                w.singleUp[v] = {0, 0, 0, 0};
                if (color[v] >= 0) w.singleUp[v][color[v]] = w.attenuation[v];
                w.pairUp[v] = w.failUp[v] = {0, 0, 0, 0, 0, 0};
                if (hybrid) w.lengthUp[v] = w.bothUp[v] = {0, 0, 0, 0, 0, 0};
            }
            for (int oi = int(g.order.size()) - 1; oi > 0; --oi) {
                int v = g.order[oi];
                if (v < g.n) continue;
                int a = g.children[v][0], b = g.children[v][1];
                const auto& ca = w.singleUp[a];
                const auto& cb = w.singleUp[b];
                for (int c = 0; c < K; ++c)
                    w.singleUp[v][c] = w.attenuation[v] * (ca[c] + cb[c]);
                for (int p = 0; p < 6; ++p) {
                    double cross = ca[ax[p]] * cb[by[p]];
                    if (ax[p] != by[p]) cross += cb[ax[p]] * ca[by[p]];
                    w.pairUp[v][p] = w.pairUp[a][p] + w.pairUp[b][p] + cross;
                    w.failUp[v][p] = g.failure[v] *
                        (w.failUp[a][p] + w.failUp[b][p] + cross);
                    if (hybrid) {
                        w.lengthUp[v][p] = w.lengthFailure[v] *
                            (w.lengthUp[a][p] + w.lengthUp[b][p] + cross);
                        w.bothUp[v][p] = w.lengthFailure[v] * g.failure[v] *
                            (w.bothUp[a][p] + w.bothUp[b][p] + cross);
                    }
                }
            }
            w.singleDown[g.root] = {0, 0, 0, 0};
            w.pairDown[g.root] = w.failDown[g.root] = {0, 0, 0, 0, 0, 0};
            if (hybrid) w.lengthDown[g.root] = w.bothDown[g.root] = {0, 0, 0, 0, 0, 0};
            for (int v : g.order) {
                if (v < g.n) continue;
                array<array<double, 4>, 3> c{};
                array<array<double, 6>, 3> base{}, fail{}, len{}, both{};
                array<int, 3> neighbor{};
                if (v == g.root) neighbor = g.rootChildren;
                else neighbor = {g.children[v][0], g.children[v][1], g.parent[v]};
                for (int k = 0; k < 3; ++k) {
                    if (v != g.root && k == 2) {
                        c[k] = w.singleDown[v];
                        base[k] = w.pairDown[v];
                        fail[k] = w.failDown[v];
                        if (hybrid) { len[k] = w.lengthDown[v]; both[k] = w.bothDown[v]; }
                    } else {
                        c[k] = w.singleUp[neighbor[k]];
                        base[k] = w.pairUp[neighbor[k]];
                        fail[k] = w.failUp[neighbor[k]];
                        if (hybrid) {
                            len[k] = w.lengthUp[neighbor[k]];
                            both[k] = w.bothUp[neighbor[k]];
                        }
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    int i = (k + 1) % 3, j = (k + 2) % 3;
                    array<double, 6> cross, pair;
                    for (int p = 0; p < 6; ++p) {
                        cross[p] = c[i][ax[p]] * c[j][by[p]];
                        if (ax[p] != by[p]) cross[p] += c[j][ax[p]] * c[i][by[p]];
                        pair[p] = base[k][p] - fail[k][p];
                        if (hybrid) pair[p] += both[k][p] - len[k][p];
                    }
                    if constexpr (K == 4) {
                        scores[0] += cross[0] * pair[5] + cross[5] * pair[0];
                        scores[1] += cross[1] * pair[4] + cross[4] * pair[1];
                        scores[2] += cross[2] * pair[3] + cross[3] * pair[2];
                    } else {
                        double agree = 0;
                        for (int x = 0; x < 3; ++x)
                            agree += cross[repeated[x]] * pair[opposite[x]] +
                                     cross[opposite[x]] * pair[repeated[x]];
                        scores[0] += agree;
                        if (normalization > 0)
                            scores[1] += agree + cross[1] * pair[2] + cross[2] * pair[1] +
                                cross[1] * pair[4] + cross[4] * pair[1] +
                                cross[2] * pair[4] + cross[4] * pair[2];
                    }
                    if (v == g.root || k < 2) {
                        int child = neighbor[k];
                        if (child < g.n) continue;
                        for (int p = 0; p < 6; ++p) {
                            w.pairDown[child][p] = base[i][p] + base[j][p] + cross[p];
                            w.failDown[child][p] = g.failure[child] *
                                (fail[i][p] + fail[j][p] + cross[p]);
                            if (hybrid) {
                                w.lengthDown[child][p] = w.lengthFailure[child] *
                                    (len[i][p] + len[j][p] + cross[p]);
                                w.bothDown[child][p] = w.lengthFailure[child] * g.failure[child] *
                                    (both[i][p] + both[j][p] + cross[p]);
                            }
                        }
                        for (int x = 0; x < K; ++x)
                            w.singleDown[child][x] = w.attenuation[child] * (c[i][x] + c[j][x]);
                    }
                }
            }
            for (int q = 0; q < 3; ++q) perGene[gi][q] = .5 * (scores[q] - previous[q]);
        }
        for (double& x : scores) x *= .5;
        return scores;
    }

public:
    long long evaluations = 0;
    long long cacheHits = 0;
    explicit QuartetScorer(vector<Gene>& gs, double alpha = 0, bool useHybrid = false,
                           double scale = 100, int maxBins = 0, double norm = 0,
                           double tips = 0, int protectedLeaf = -1) :
        genes(gs), externalAlpha(alpha), hybrid(useHybrid), hybridScale(scale),
        maximumBins(maxBins), normalization(norm), tipAlpha(tips) {
        if (maxBins < 0 || maxBins > 32 || (maxBins && alpha > 0))
            throw runtime_error("Invalid maximum-support settings");
        for (const Gene& g : genes) work.emplace_back(g.m);
        if (maxBins)
            for (const Gene& g : genes)
                maximumWork.emplace_back(g, maxBins, hybrid, scale, tips, protectedLeaf);
        else if (externalAlpha > 0 || hybrid || tips > 0)
            for (const Gene& g : genes)
                externalWork.emplace_back(g, alpha, hybrid, scale, tips, protectedLeaf);
        fourCache.reserve(32768);
        threeCache.reserve(32768);
        locusWeights.resize(genes.size(), 1);
    }
    void setWeights(const vector<double>& weights) {
        if (weights.size() != genes.size()) throw runtime_error("Incorrect locus weights");
        locusWeights = weights;
        reweighted = true;
    }
    void resetWeights() { reweighted = false; }
    int loci() const { return int(genes.size()); }

    array<double, 3> topologyScores(const vector<int>& color) {
        if (normalization <= 0) return four(color);
        array<double, 3> score{};
        vector<int> trip(color.size());
        for (int q = 0; q < 3; ++q) {
            int partner = q + 1;
            array<int, 2> other{};
            int k = 0;
            for (int c = 1; c < 4; ++c) if (c != partner) other[k++] = c;
            for (int x = 0; x < int(color.size()); ++x)
                trip[x] = color[x] == 0 ? 0 : color[x] == partner ? 1 : 2;
            score[q] = tripartition(trip);
            for (int x = 0; x < int(color.size()); ++x)
                trip[x] = color[x] == other[0] ? 0 : color[x] == other[1] ? 1 : 2;
            score[q] = .5 * (score[q] + tripartition(trip));
        }
        return score;
    }

    array<double, 3> four(const vector<int>& color) {
        PartitionKey key;
        for (int x = 0; x < int(color.size()); ++x) if (color[x] >= 0)
            key.parts[color[x]].set(x);
        array<int, 4> perm{0, 1, 2, 3};
        sort(perm.begin(), perm.end(), [&](int a, int b) { return key.parts[a] < key.parts[b]; });
        PartitionKey canonical;
        array<int, 4> rank{};
        for (int i = 0; i < 4; ++i) {
            canonical.parts[i] = key.parts[perm[i]];
            rank[perm[i]] = i;
        }
        array<int, 3> qrank{};
        for (int i = 0; i < 3; ++i) {
            int bits = (1 << rank[0]) | (1 << rank[i + 1]);
            if (!(bits & 1)) bits ^= 15;
            qrank[i] = bits == 3 ? 0 : bits == 5 ? 1 : 2;
        }
        auto cached = fourCache.find(canonical);
        if (cached != fourCache.end()) {
            ++cacheHits;
            auto resolved = weighted(cached->second);
            return {resolved[qrank[0]], resolved[qrank[1]], resolved[qrank[2]]};
        }
        ++evaluations;
        vector<array<double, 3>> perGene(genes.size());
        if (maximumBins) {
            auto scores = scoreMaximum<4>(color, perGene);
            return saveFour(canonical, qrank, scores, move(perGene));
        }
        if (externalAlpha > 0 || hybrid || tipAlpha > 0) {
            auto scores = scoreExternal<4>(color, perGene);
            return saveFour(canonical, qrank, scores, move(perGene));
        }
        array<int, 4> totals{};
        for (int c : color) if (c >= 0 && c < 4) ++totals[c];
        array<double, 3> scores{};
        for (int gi = 0; gi < int(genes.size()); ++gi) {
            auto previous = scores;
            const Gene& g = genes[gi];
            Work& w = work[gi];
            for (int v = 0; v < g.n; ++v) {
                w.count[v] = {0, 0, 0, 0};
                if (color[v] >= 0) w.count[v][color[v]] = 1;
                w.up[v] = {0, 0, 0, 0, 0, 0};
            }
            for (int oi = int(g.order.size()) - 1; oi > 0; --oi) {
                int v = g.order[oi];
                if (v < g.n) continue;
                int a = g.children[v][0], b = g.children[v][1];
                const auto& ca = w.count[a];
                const auto& cb = w.count[b];
                for (int c = 0; c < 4; ++c) w.count[v][c] = ca[c] + cb[c];
                for (int p = 0; p < 6; ++p)
                    w.up[v][p] = g.failure[v] * (w.up[a][p] + w.up[b][p] +
                        double(ca[pa[p]]) * cb[pb[p]] + double(cb[pa[p]]) * ca[pb[p]]);
            }
            w.down[g.root] = {0, 0, 0, 0, 0, 0};
            for (int v : g.order) {
                if (v < g.n) continue;
                array<array<int, 4>, 3> c{};
                array<array<double, 6>, 3> f{};
                array<int, 3> neighbor{};
                if (v == g.root) {
                    neighbor = g.rootChildren;
                    for (int i = 0; i < 3; ++i) {
                        c[i] = w.count[neighbor[i]];
                        f[i] = w.up[neighbor[i]];
                    }
                } else {
                    neighbor = {g.children[v][0], g.children[v][1], g.parent[v]};
                    for (int i = 0; i < 2; ++i) {
                        c[i] = w.count[neighbor[i]];
                        f[i] = w.up[neighbor[i]];
                    }
                    for (int k = 0; k < 4; ++k) c[2][k] = totals[k] - w.count[v][k];
                    f[2] = w.down[v];
                }
                for (int k = 0; k < 3; ++k) {
                    int i = (k + 1) % 3, j = (k + 2) % 3;
                    array<double, 6> cross, pair;
                    for (int p = 0; p < 6; ++p) {
                        cross[p] = double(c[i][pa[p]]) * c[j][pb[p]] +
                                   double(c[j][pa[p]]) * c[i][pb[p]];
                        pair[p] = double(c[k][pa[p]]) * c[k][pb[p]] - f[k][p];
                    }
                    scores[0] += cross[0] * pair[5] + cross[5] * pair[0];
                    scores[1] += cross[1] * pair[4] + cross[4] * pair[1];
                    scores[2] += cross[2] * pair[3] + cross[3] * pair[2];
                    if (v == g.root || k < 2) {
                        int child = neighbor[k];
                        if (child >= g.n)
                            for (int p = 0; p < 6; ++p)
                                w.down[child][p] = g.failure[child] *
                                    (f[i][p] + f[j][p] + cross[p]);
                    }
                }
            }
            for (int q = 0; q < 3; ++q) perGene[gi][q] = .5 * (scores[q] - previous[q]);
        }
        for (double& x : scores) x *= .5;
        return saveFour(canonical, qrank, scores, move(perGene));
    }

    double tripartition(const vector<int>& color) {
        PartitionKey key;
        for (int x = 0; x < int(color.size()); ++x) if (color[x] >= 0)
            key.parts[color[x]].set(x);
        sort(key.parts.begin(), key.parts.begin() + 3);
        auto cached = threeCache.find(key);
        if (cached != threeCache.end()) {
            ++cacheHits;
            return weighted(cached->second);
        }
        ++evaluations;
        vector<double> perGene(genes.size());
        if (maximumBins) {
            vector<array<double, 3>> contributions(genes.size());
            auto scores = scoreMaximum<3>(color, contributions);
            vector<double> perGeneAll(genes.size());
            for (int i = 0; i < int(genes.size()); ++i) perGene[i] = contributions[i][0];
            if (normalization > 0)
                for (int i = 0; i < int(genes.size()); ++i) perGeneAll[i] = contributions[i][1];
            return saveThree(key, scores[0], move(perGene), scores[1], move(perGeneAll));
        }
        if (externalAlpha > 0 || hybrid || tipAlpha > 0) {
            vector<array<double, 3>> contributions(genes.size());
            auto scores = scoreExternal<3>(color, contributions);
            vector<double> perGeneAll(genes.size());
            for (int i = 0; i < int(genes.size()); ++i) perGene[i] = contributions[i][0];
            if (normalization > 0)
                for (int i = 0; i < int(genes.size()); ++i) perGeneAll[i] = contributions[i][1];
            return saveThree(key, scores[0], move(perGene), scores[1], move(perGeneAll));
        }
        static constexpr int aidx[6] = {0, 0, 0, 1, 1, 2};
        static constexpr int bidx[6] = {0, 1, 2, 1, 2, 2};
        static constexpr int repeated[3] = {0, 3, 5};
        static constexpr int opposite[3] = {4, 2, 1};
        array<int, 4> totals{};
        for (int c : color) if (c >= 0) ++totals[c];
        double score = 0, all = 0;
        vector<double> perGeneAll(genes.size());
        for (int gi = 0; gi < int(genes.size()); ++gi) {
            double previous = score;
            double previousAll = all;
            const Gene& g = genes[gi];
            Work& w = work[gi];
            for (int v = 0; v < g.n; ++v) {
                w.count[v] = {0, 0, 0, 0};
                if (color[v] >= 0) w.count[v][color[v]] = 1;
                w.up[v] = {0, 0, 0, 0, 0, 0};
            }
            for (int oi = int(g.order.size()) - 1; oi > 0; --oi) {
                int v = g.order[oi];
                if (v < g.n) continue;
                int a = g.children[v][0], b = g.children[v][1];
                const auto& ca = w.count[a];
                const auto& cb = w.count[b];
                for (int c = 0; c < 3; ++c) w.count[v][c] = ca[c] + cb[c];
                for (int p = 0; p < 6; ++p) {
                    int x = aidx[p], y = bidx[p];
                    double cross = double(ca[x]) * cb[y];
                    if (x != y) cross += double(cb[x]) * ca[y];
                    w.up[v][p] = g.failure[v] * (w.up[a][p] + w.up[b][p] + cross);
                }
            }
            w.down[g.root] = {0, 0, 0, 0, 0, 0};
            for (int v : g.order) {
                if (v < g.n) continue;
                array<array<int, 4>, 3> c{};
                array<array<double, 6>, 3> f{};
                array<int, 3> neighbor{};
                if (v == g.root) {
                    neighbor = g.rootChildren;
                    for (int i = 0; i < 3; ++i) {
                        c[i] = w.count[neighbor[i]];
                        f[i] = w.up[neighbor[i]];
                    }
                } else {
                    neighbor = {g.children[v][0], g.children[v][1], g.parent[v]};
                    for (int i = 0; i < 2; ++i) {
                        c[i] = w.count[neighbor[i]];
                        f[i] = w.up[neighbor[i]];
                    }
                    for (int k = 0; k < 3; ++k) c[2][k] = totals[k] - w.count[v][k];
                    f[2] = w.down[v];
                }
                for (int k = 0; k < 3; ++k) {
                    int i = (k + 1) % 3, j = (k + 2) % 3;
                    array<double, 6> cross, pair;
                    for (int p = 0; p < 6; ++p) {
                        int x = aidx[p], y = bidx[p];
                        cross[p] = double(c[i][x]) * c[j][y];
                        pair[p] = double(c[k][x]) * (c[k][y] - (x == y));
                        if (x != y) cross[p] += double(c[j][x]) * c[i][y];
                        else pair[p] *= .5;
                        pair[p] -= f[k][p];
                    }
                    double agree = 0;
                    for (int x = 0; x < 3; ++x)
                        agree += cross[repeated[x]] * pair[opposite[x]] +
                                 cross[opposite[x]] * pair[repeated[x]];
                    score += agree;
                    if (normalization > 0)
                        all += agree + cross[1] * pair[2] + cross[2] * pair[1] +
                            cross[1] * pair[4] + cross[4] * pair[1] +
                            cross[2] * pair[4] + cross[4] * pair[2];
                    if (v == g.root || k < 2) {
                        int child = neighbor[k];
                        if (child >= g.n)
                            for (int p = 0; p < 6; ++p)
                                w.down[child][p] = g.failure[child] *
                                    (f[i][p] + f[j][p] + cross[p]);
                    }
                }
            }
            perGene[gi] = .5 * (score - previous);
            if (normalization > 0) perGeneAll[gi] = .5 * (all - previousAll);
        }
        score *= .5;
        all *= .5;
        return saveThree(key, score, move(perGene), all, move(perGeneAll));
    }

    double treeScore(const Tree& t) {
        double s = 0;
        vector<int> color(t.n, 0);
        for (int v = t.n; v < int(t.adj.size()); ++v) if (t.adj[v].size() == 3) {
            for (int k = 0; k < 3; ++k)
                for (int x : t.side(t.adj[v][k].to, v)) color[x] = k;
            s += tripartition(color);
        }
        return s * .5;
    }
};

static vector<pair<Bits, pair<Bits, Bits>>> rootedPartitions(const Tree& t, int anchor) {
    vector<pair<Bits, pair<Bits, Bits>>> result;
    function<Bits(int, int)> visit = [&](int v, int p) -> Bits {
        if (v < t.n) { Bits b; b.set(v); return b; }
        array<Bits, 2> children{};
        int k = 0;
        for (Edge e : t.adj[v]) if (e.to != p) children[k++] = visit(e.to, v);
        Bits b = children[0] | children[1];
        result.push_back({b, {children[0], children[1]}});
        return b;
    };
    visit(t.adj[anchor][0].to, anchor);
    return result;
}

class ClusterSearch {
    struct Transition {
        int a, b;
        double score = numeric_limits<double>::quiet_NaN();
        bool initial = false;
    };
    struct Cluster {
        Bits bits;
        int size;
        int frequency = 0;
        vector<Transition> transitions;
        double score = -numeric_limits<double>::infinity();
        pair<int, int> best{-1, -1};
    };
    int n, anchor, fullId = -1;
    int maxGeneSize, minFrequency;
    Bits full;
    vector<Cluster> clusters;
    unordered_map<Bits, int, BitsHash> index;
    vector<pair<Bits, pair<Bits, Bits>>> initial;
    vector<int> bySize;
    vector<pair<int, int>> todo;
    bool prepared = false;

    int add(const Bits& b) {
        auto it = index.find(b);
        if (it != index.end()) return it->second;
        int id = int(clusters.size());
        index[b] = id;
        clusters.push_back({b, b.size(), 0, {}, -numeric_limits<double>::infinity(),
                            {-1, -1}});
        return id;
    }
    void transition(int c, int a, int b, bool init = false) {
        if (a > b) swap(a, b);
        for (auto& t : clusters[c].transitions)
            if (t.a == a && t.b == b) { t.initial |= init; return; }
        clusters[c].transitions.push_back({a, b, numeric_limits<double>::quiet_NaN(), init});
    }
    void prepare() {
        if (prepared) return;
        for (const auto& p : initial)
            transition(index.at(p.first), index.at(p.second.first),
                       index.at(p.second.second), true);
        bySize.resize(clusters.size());
        iota(bySize.begin(), bySize.end(), 0);
        sort(bySize.begin(), bySize.end(), [&](int a, int b) {
            if (clusters[a].size != clusters[b].size)
                return clusters[a].size < clusters[b].size;
            return clusters[a].bits < clusters[b].bits;
        });
        long long count = 0;
        for (int c : bySize) {
            if (clusters[c].size == 1) continue;
            const Bits& cb = clusters[c].bits;
            for (int a : bySize) {
                if (clusters[a].size * 2 > clusters[c].size) break;
                if (!clusters[a].bits.subset(cb)) continue;
                auto it = index.find(cb - clusters[a].bits);
                if (it == index.end()) continue;
                int b = it->second;
                if (clusters[a].size == clusters[b].size && a > b) continue;
                transition(c, a, b);
            }
            count += clusters[c].transitions.size();
        }
        (void)count;
        todo.clear();
        for (int c : bySize)
            for (int j = 0; j < int(clusters[c].transitions.size()); ++j)
                todo.emplace_back(c, j);
        stable_sort(todo.begin(), todo.end(), [&](auto a, auto b) {
            const auto& ta = clusters[a.first].transitions[a.second];
            const auto& tb = clusters[b.first].transitions[b.second];
            if (ta.initial != tb.initial) return ta.initial > tb.initial;
            if (clusters[a.first].size != clusters[b.first].size)
                return clusters[a.first].size < clusters[b.first].size;
            return clusters[a.first].frequency > clusters[b.first].frequency;
        });
        prepared = true;
    }

public:
    ClusterSearch(int leaves, int outgroup, int maxSize, int minCount)
        : n(leaves), anchor(outgroup), maxGeneSize(maxSize), minFrequency(minCount) {
        for (int x = 0; x < n; ++x) if (x != anchor) {
            full.set(x);
            Bits leaf;
            leaf.set(x);
            add(leaf);
        }
        fullId = add(full);
    }
    void addGenes(const vector<Tree>& genes) {
        unordered_map<Bits, int, BitsHash> frequencies;
        for (const Tree& g : genes)
            for (const auto& p : rootedPartitions(g, anchor)) ++frequencies[p.first];
        for (const auto& [b, freq] : frequencies) {
            if (b.size() <= maxGeneSize || freq >= minFrequency || b == full) {
                int c = add(b);
                clusters[c].frequency = freq;
            }
        }
        prepared = false;
    }
    void addTree(const Tree& t) {
        for (const auto& p : rootedPartitions(t, anchor)) {
            add(p.first);
            add(p.second.first);
            add(p.second.second);
            initial.push_back(p);
        }
        prepared = false;
    }
    Tree solve(QuartetScorer& scorer, double deadline, bool quiet = false) {
        prepare();
        if (!quiet)
            cerr << "Cluster search: " << clusters.size() << " clusters, "
                 << todo.size() << " transitions, elapsed " << seconds() << '\n';
        for (auto& cl : clusters) {
            cl.score = -numeric_limits<double>::infinity();
            cl.best = {-1, -1};
            for (auto& t : cl.transitions) t.score = numeric_limits<double>::quiet_NaN();
        }
        vector<int> color(n);
        long long computed = 0;
        for (auto [c, j] : todo) {
            auto& t = clusters[c].transitions[j];
            if (!t.initial && seconds() >= deadline) break;
            for (int x = 0; x < n; ++x)
                color[x] = clusters[t.a].bits.has(x) ? 0 :
                           clusters[t.b].bits.has(x) ? 1 : 2;
            t.score = scorer.tripartition(color);
            ++computed;
        }
        for (int c : bySize) {
            auto& cl = clusters[c];
            if (cl.size == 1) { cl.score = 0; continue; }
            for (const auto& t : cl.transitions) if (isfinite(t.score)) {
                double score = t.score + clusters[t.a].score + clusters[t.b].score;
                if (score > cl.score) {
                    cl.score = score;
                    cl.best = {t.a, t.b};
                }
            }
        }
        Tree result(n);
        function<int(int)> build = [&](int c) {
            const auto& cl = clusters[c];
            if (cl.size == 1)
                for (int x = 0; x < n; ++x) if (cl.bits.has(x)) return x;
            if (cl.best.first < 0) throw runtime_error("Incomplete cluster search");
            int u = result.node();
            int a = build(cl.best.first), b = build(cl.best.second);
            result.connect(u, a);
            result.connect(u, b);
            return u;
        };
        int root = build(fullId);
        result.connect(root, anchor);
        result.validate();
        if (!quiet)
            cerr << "Cluster score: " << setprecision(16) << clusters[fullId].score * .5
                 << ", " << computed << " transitions evaluated, elapsed " << seconds() << '\n';
        return result;
    }
};

static Tree resolvePosterior(const vector<pair<Bits, double>>& frequencies,
                             const Tree& backbone, int anchor, double deadline) {
    struct Region {
        Bits bits;
        int size;
        vector<int> children;
        vector<uint32_t> split;
        bool exact = false;
    };
    int n = backbone.n;
    vector<Region> regions;
    unordered_map<Bits, int, BitsHash> index;
    auto add = [&](const Bits& bits) {
        auto it = index.find(bits);
        if (it != index.end()) return it->second;
        int id = int(regions.size());
        regions.push_back({bits, bits.size(), {}, {}, false});
        index[bits] = id;
        return id;
    };
    Bits full;
    for (int x = 0; x < n; ++x) if (x != anchor) full.set(x);
    int root = add(full);
    double mass = 0;
    for (const auto& [bits, frequency] : frequencies)
        if (bits == full) mass = frequency;
    for (int x = 0; x < n; ++x) if (x != anchor) {
        Bits leaf;
        leaf.set(x);
        add(leaf);
    }
    for (const auto& [bits, frequency] : frequencies)
        if (frequency >= mass - 1e-8 && bits.size() > 1) add(bits);
    for (int c = 0; c < int(regions.size()); ++c) if (c != root) {
        int parent = root;
        for (int p = 0; p < int(regions.size()); ++p)
            if (regions[p].size > regions[c].size &&
                regions[p].size < regions[parent].size &&
                regions[c].bits.subset(regions[p].bits)) parent = p;
        regions[parent].children.push_back(c);
    }
    int exactCount = 0, largest = 0;
    for (auto& region : regions) {
        int k = int(region.children.size());
        largest = max(largest, k);
        if (k <= 2 || k > 18 || seconds() >= deadline) continue;
        uint32_t count = uint32_t(1) << k;
        vector<double> weight(count, 0), dp(count, 0);
        vector<uint32_t> split(count, 0);
        for (const auto& [bits, frequency] : frequencies) {
            if (!bits.subset(region.bits)) continue;
            uint32_t mask = 0;
            bool partial = false;
            for (int i = 0; i < k; ++i) {
                const Bits& atom = regions[region.children[i]].bits;
                if (atom.subset(bits)) mask |= uint32_t(1) << i;
                else {
                    for (int w = 0; w < 4; ++w)
                        if (atom.words[w] & bits.words[w]) { partial = true; break; }
                }
                if (partial) break;
            }
            if (!partial && __builtin_popcount(mask) >= 2) weight[mask] = frequency;
        }
        bool complete = true;
        for (uint32_t mask = 1; mask < count; ++mask) {
            if ((mask & (mask - 1)) == 0) continue;
            if ((mask & 1023) == 0 && seconds() >= deadline) { complete = false; break; }
            uint32_t first = mask & -mask, rest = mask ^ first;
            double best = -numeric_limits<double>::infinity();
            uint32_t bestSplit = 0;
            for (uint32_t sub = rest;; sub = (sub - 1) & rest) {
                uint32_t a = sub | first, b = mask ^ a;
                if (b && dp[a] + dp[b] > best) {
                    best = dp[a] + dp[b];
                    bestSplit = a;
                }
                if (!sub) break;
            }
            dp[mask] = best + weight[mask];
            split[mask] = bestSplit;
        }
        if (complete) {
            region.exact = true;
            region.split = move(split);
            ++exactCount;
        }
    }
    Tree result(n);
    function<int(int)> build = [&](int c) -> int {
        const Region& region = regions[c];
        if (region.size == 1)
            for (int x = 0; x < n; ++x) if (region.bits.has(x)) return x;
        vector<int> atoms;
        for (int child : region.children) atoms.push_back(build(child));
        int k = int(atoms.size());
        if (k == 2) {
            int u = result.node();
            result.connect(u, atoms[0]);
            result.connect(u, atoms[1]);
            return u;
        }
        if (region.exact) {
            function<int(uint32_t)> assemble = [&](uint32_t mask) -> int {
                if ((mask & (mask - 1)) == 0) return atoms[__builtin_ctz(mask)];
                uint32_t a = region.split[mask], b = mask ^ a;
                int u = result.node();
                int left = assemble(a), right = assemble(b);
                result.connect(u, left);
                result.connect(u, right);
                return u;
            };
            return assemble((uint32_t(1) << k) - 1);
        }
        vector<int> representatives(n, -1);
        for (int i = 0; i < k; ++i)
            for (int x = 0; x < n; ++x) if (regions[region.children[i]].bits.has(x)) {
                representatives[x] = atoms[i];
                break;
            }
        function<int(int, int)> induce = [&](int v, int p) -> int {
            if (v < n) return representatives[v];
            array<int, 2> child{-1, -1};
            int j = 0;
            for (Edge e : backbone.adj[v]) if (e.to != p) child[j++] = induce(e.to, v);
            if (child[0] < 0) return child[1];
            if (child[1] < 0) return child[0];
            int u = result.node();
            result.connect(u, child[0]);
            result.connect(u, child[1]);
            return u;
        };
        return induce(backbone.adj[anchor][0].to, anchor);
    };
    int top = build(root);
    result.connect(top, anchor);
    result.validate();
    cerr << "Posterior resolution: " << exactCount << " exact regions, largest "
         << largest << " atoms, elapsed " << seconds() << '\n';
    return result;
}

static Tree splitConsensus(const vector<Tree>& trees, const Tree& point,
                           int anchor, double prior,
                           double deadline = numeric_limits<double>::infinity()) {
    struct Clade {
        Bits bits;
        int size;
        double frequency = 0;
        double score = -numeric_limits<double>::infinity();
        pair<int, int> best{-1, -1};
    };
    int n = point.n;
    vector<Clade> clades;
    unordered_map<Bits, int, BitsHash> index;
    auto add = [&](const Bits& b, double frequency) {
        auto it = index.find(b);
        if (it != index.end()) { clades[it->second].frequency += frequency; return it->second; }
        int id = int(clades.size());
        index[b] = id;
        clades.push_back({b, b.size(), frequency, -numeric_limits<double>::infinity(),
                          {-1, -1}});
        return id;
    };
    Bits full;
    for (int x = 0; x < n; ++x) if (x != anchor) {
        Bits leaf;
        leaf.set(x);
        add(leaf, 0);
        full.set(x);
    }
    int rootId = add(full, 0);
    auto include = [&](const Tree& t, double weight) {
        for (const auto& p : rootedPartitions(t, anchor)) add(p.first, weight);
    };
    for (const Tree& t : trees) include(t, 1);
    include(point, prior);
    vector<int> order(clades.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (clades[a].size != clades[b].size) return clades[a].size < clades[b].size;
        return clades[a].bits < clades[b].bits;
    });
    for (int c : order) {
        Clade& cl = clades[c];
        if (cl.size == 1) { cl.score = 0; continue; }
        for (int a : order) {
            if (clades[a].size * 2 > cl.size) break;
            if (!clades[a].bits.subset(cl.bits)) continue;
            auto it = index.find(cl.bits - clades[a].bits);
            if (it == index.end()) continue;
            int b = it->second;
            if (clades[a].size == clades[b].size && a > b) continue;
            double score = cl.frequency + clades[a].score + clades[b].score;
            if (score > cl.score) { cl.score = score; cl.best = {a, b}; }
        }
    }
    Tree result(n);
    function<int(int)> build = [&](int c) {
        const Clade& cl = clades[c];
        if (cl.size == 1)
            for (int x = 0; x < n; ++x) if (cl.bits.has(x)) return x;
        if (cl.best.first < 0) throw runtime_error("Incomplete split consensus");
        int u = result.node();
        int a = build(cl.best.first), b = build(cl.best.second);
        result.connect(u, a);
        result.connect(u, b);
        return u;
    };
    result.connect(anchor, build(rootId));
    result.validate();
    vector<pair<Bits, double>> frequencies;
    for (const auto& cl : clades) if (cl.size > 1)
        frequencies.emplace_back(cl.bits, cl.frequency);
    Tree refined = resolvePosterior(frequencies, result, anchor, deadline);
    double score = 0;
    for (const auto& p : rootedPartitions(refined, anchor)) {
        auto it = index.find(p.first);
        if (it != index.end()) score += clades[it->second].frequency;
    }
    if (score > clades[rootId].score + 1e-8) result = move(refined);
    cerr << "Split consensus: " << trees.size() << " replicates, "
         << clades.size() << " clades, elapsed " << seconds() << '\n';
    return result;
}

static int optimizeNNI(Tree& t, QuartetScorer& scorer, mt19937_64& rng,
                       double deadline, int maxSweeps = 20, bool quiet = false,
                       const Matrix* guide = nullptr, double prior = 0,
                       bool uniformGuide = false) {
    int moves = 0;
    double distanceScale = 0;
    if (guide) {
        for (int i = 0; i < t.n; ++i)
            for (int j = i + 1; j < t.n; ++j) distanceScale += (*guide)[i][j];
        distanceScale /= t.n * (t.n - 1) / 2;
        distanceScale = max(1e-12, distanceScale);
    }
    vector<pair<int, int>> edges;
    vector<int> color(t.n);
    for (int sweep = 0; sweep < maxSweeps && seconds() < deadline; ++sweep) {
        edges.clear();
        for (int u = t.n; u < int(t.adj.size()); ++u)
            for (const Edge& e : t.adj[u])
                if (e.to > u && e.to >= t.n) edges.emplace_back(u, e.to);
        shuffle(edges.begin(), edges.end(), rng);
        int changed = 0;
        for (auto [u, v] : edges) {
            if (seconds() >= deadline) break;
            if (none_of(t.adj[u].begin(), t.adj[u].end(),
                        [v](const Edge& e) { return e.to == v; })) continue;
            array<int, 2> a{}, b{};
            int k = 0;
            for (const Edge& e : t.adj[u]) if (e.to != v) a[k++] = e.to;
            k = 0;
            for (const Edge& e : t.adj[v]) if (e.to != u) b[k++] = e.to;
            for (int x : t.side(a[0], u)) color[x] = 0;
            for (int x : t.side(a[1], u)) color[x] = 1;
            for (int x : t.side(b[0], v)) color[x] = 2;
            for (int x : t.side(b[1], v)) color[x] = 3;
            auto q = scorer.topologyScores(color);
            int best = int(max_element(q.begin(), q.end()) - q.begin());
            if (guide) {
                array<double, 3> distance = localDistanceScores(t, *guide, u, a, v, b,
                                                              uniformGuide);
                array<int, 4> counts{};
                for (int c : color) ++counts[c];
                double capacity = double(scorer.loci()) * counts[0] * counts[1] *
                                  counts[2] * counts[3];
                array<double, 3> score{};
                for (int j = 0; j < 3; ++j)
                    score[j] = q[j] / capacity - prior * distance[j] / distanceScale;
                best = int(max_element(score.begin(), score.end()) - score.begin());
                if (score[best] <= score[0] + 1e-10) best = 0;
            }
            if (q[best] > q[0] + 1e-5) {
                if (best == 1) t.swapBranches(u, a[1], v, b[0]);
                else t.swapBranches(u, a[1], v, b[1]);
                ++changed;
            }
        }
        moves += changed;
        if (!quiet)
            cerr << "NNI sweep " << sweep + 1 << ": " << changed
                 << " moves, " << fixed << setprecision(2) << seconds() << " s\n";
        if (!changed) break;
    }
    return moves;
}

static double placeSubtree(Tree& t, int x, int v, QuartetScorer& scorer,
                           double deadline, int radius) {
    vector<int> leaves = t.side(x, v);
    if (leaves.size() + 3 > size_t(t.n)) return 0;
    array<int, 2> other{};
    int k = 0;
    for (Edge e : t.adj[v]) if (e.to != x) other[k++] = e.to;
    int a = other[0], b = other[1];
    t.disconnect(v, x);
    t.disconnect(v, a);
    t.disconnect(v, b);
    t.connect(a, b);
    double bestGain = 0;
    pair<int, int> bestEdge{a, b};
    vector<int> color(t.n, 1);
    function<void(int, int, double, int)> visit =
        [&](int u, int p, double gain, int depth) {
            if (u < t.n || depth >= radius || seconds() >= deadline) return;
            array<int, 2> ch{};
            int j = 0;
            for (Edge e : t.adj[u]) if (e.to != p) ch[j++] = e.to;
            fill(color.begin(), color.end(), 1);
            for (int z : leaves) color[z] = 0;
            for (int z : t.side(ch[0], u)) color[z] = 2;
            for (int z : t.side(ch[1], u)) color[z] = 3;
            auto q = scorer.topologyScores(color);
            array<double, 2> gains{gain + q[1] - q[0], gain + q[2] - q[0]};
            for (int i = 0; i < 2; ++i) if (gains[i] > bestGain + 1e-5) {
                bestGain = gains[i];
                bestEdge = {u, ch[i]};
            }
            int first = gains[0] > gains[1] ? 0 : 1;
            visit(ch[first], u, gains[first], depth + 1);
            visit(ch[1 - first], u, gains[1 - first], depth + 1);
        };
    visit(a, b, 0, 0);
    visit(b, a, 0, 0);
    t.disconnect(bestEdge.first, bestEdge.second);
    t.connect(v, bestEdge.first);
    t.connect(v, bestEdge.second);
    t.connect(v, x);
    return bestGain;
}

static int optimizeSPR(Tree& t, QuartetScorer& scorer, mt19937_64& rng,
                       double deadline, int radius, int maxSubtree, int passes) {
    int moves = 0;
    for (int pass = 0; pass < passes && seconds() < deadline; ++pass) {
        vector<pair<int, int>> prunes;
        for (int u = 0; u < int(t.adj.size()); ++u) {
            for (Edge e : t.adj[u]) if (u < e.to) {
                int v = e.to;
                int count = int(t.side(u, v).size());
                if (count <= maxSubtree && v >= t.n) prunes.emplace_back(u, v);
                if (t.n - count <= maxSubtree && u >= t.n) prunes.emplace_back(v, u);
            }
        }
        shuffle(prunes.begin(), prunes.end(), rng);
        int changed = 0, tried = 0;
        double totalGain = 0;
        for (auto [x, v] : prunes) {
            if (seconds() >= deadline) break;
            if (none_of(t.adj[v].begin(), t.adj[v].end(),
                        [x](const Edge& e) { return e.to == x; })) continue;
            if (int(t.side(x, v).size()) > maxSubtree) continue;
            double gain = placeSubtree(t, x, v, scorer, deadline, radius);
            if (gain > 1e-5) { ++changed; totalGain += gain; }
            ++tried;
        }
        moves += changed;
        cerr << "SPR pass " << pass + 1 << ": " << changed << " moves / "
             << tried << " subtrees, gain " << totalGain << ", elapsed " << seconds() << '\n';
        optimizeNNI(t, scorer, rng, deadline, 10);
        if (!changed) break;
    }
    return moves;
}

static Tree bootstrapInference(const Tree& point, const vector<Tree>& candidates,
                               const vector<Tree>& input, QuartetScorer& scorer,
                               mt19937_64& rng, int anchor, int replicates,
                               bool useDP, double prior, double deadline,
                               int maxGeneSize, int minFrequency, bool bayesian,
                               double shape, bool njStart, int distanceMode,
                               double power, bool bio, bool bme,
                               const vector<double>* quality = nullptr) {
    ClusterSearch search(point.n, anchor, maxGeneSize, minFrequency);
    if (useDP) {
        search.addGenes(input);
        for (const Tree& t : candidates) search.addTree(t);
        search.addTree(point);
    }
    vector<Tree> trees;
    int k = int(input.size());
    vector<double> weights(k);
    gamma_distribution<double> draw(shape, 1);
    for (int rep = 0; rep < replicates && seconds() < deadline; ++rep) {
        fill(weights.begin(), weights.end(), 0);
        if (bayesian) {
            double total = 0;
            for (double& w : weights) { w = draw(rng); total += w; }
            for (double& w : weights) w *= k / total;
        } else {
            for (int i = 0; i < k; ++i) ++weights[rng() % k];
        }
        if (quality) {
            double total = 0;
            for (int i = 0; i < k; ++i) { weights[i] *= (*quality)[i]; total += weights[i]; }
            for (double& w : weights) w *= k / total;
        }
        scorer.setWeights(weights);
        Tree t = useDP ? search.solve(scorer, deadline, true) : point;
        if (!useDP && njStart) {
            Matrix d = distances(input, distanceMode, power, anchor, &weights);
            t = neighborJoining(d, bio);
            if (bme) optimizeBME(t, d);
        }
        int moves = optimizeNNI(t, scorer, rng, deadline, 30, true);
        trees.push_back(move(t));
        cerr << "Bootstrap " << rep + 1 << ": " << moves
             << " NNI moves, elapsed " << seconds() << '\n';
    }
    if (quality) scorer.setWeights(*quality);
    else scorer.resetWeights();
    if (trees.empty()) return point;
    return splitConsensus(trees, point, anchor, prior, deadline + 1);
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw runtime_error("Usage: infer INPUT OUTPUT [options]");
        int distanceMode = 1, weightMode = 1;
        double supportPower = 1, lengthScale = 100, budget = 120, externalAlpha = 0;
        int starts = 3, sweeps = 20, dpMaxSize = 256, dpFrequency = 1;
        int sprRadius = 8, sprSize = 8, sprPasses = 1;
        int bootstrap = 0;
        int maximumBins = 0;
        double normalization = 0;
        double distancePrior = 0;
        double distanceZero = 0;
        bool uniformGuide = false;
        double tipAlpha = 0;
        double threshold = -1;
        double geneWeightAlpha = 0;
        double geneSupportPower = 0;
        bool geneTerminalOnly = false;
        double bootPrior = 1;
        double bootShape = 1;
        bool bio = false, upg = false, nni = true, bme = false;
        bool dp = false, spr = false;
        bool bootDP = false, bootBayes = false, bootNJ = false;
        bool sameStarts = false, orderConsensus = false;
        for (int i = 3; i < argc; ++i) {
            string a = argv[i];
            auto value = [&]() -> string {
                if (++i == argc) throw runtime_error("Missing option value");
                return argv[i];
            };
            if (a == "--distance") distanceMode = stoi(value());
            else if (a == "--weight") weightMode = stoi(value());
            else if (a == "--power") supportPower = stod(value());
            else if (a == "--scale") lengthScale = stod(value());
            else if (a == "--external") externalAlpha = stod(value());
            else if (a == "--budget") budget = stod(value());
            else if (a == "--starts") starts = stoi(value());
            else if (a == "--same-starts") sameStarts = true;
            else if (a == "--order-consensus") orderConsensus = true;
            else if (a == "--sweeps") sweeps = stoi(value());
            else if (a == "--dp") dp = true;
            else if (a == "--dp-size") dpMaxSize = stoi(value());
            else if (a == "--dp-frequency") dpFrequency = stoi(value());
            else if (a == "--spr") spr = true;
            else if (a == "--spr-radius") sprRadius = stoi(value());
            else if (a == "--spr-size") sprSize = stoi(value());
            else if (a == "--spr-passes") sprPasses = stoi(value());
            else if (a == "--bootstrap") bootstrap = stoi(value());
            else if (a == "--max-support") maximumBins = stoi(value());
            else if (a == "--normalize") normalization = stod(value());
            else if (a == "--distance-prior") distancePrior = stod(value());
            else if (a == "--distance-zero") distanceZero = stod(value());
            else if (a == "--uniform-prior") uniformGuide = true;
            else if (a == "--tips") tipAlpha = stod(value());
            else if (a == "--threshold") threshold = stod(value());
            else if (a == "--gene-weight") geneWeightAlpha = stod(value());
            else if (a == "--gene-support") geneSupportPower = stod(value());
            else if (a == "--gene-terminal") geneTerminalOnly = true;
            else if (a == "--boot-prior") bootPrior = stod(value());
            else if (a == "--boot-dp") bootDP = true;
            else if (a == "--boot-bayes") bootBayes = true;
            else if (a == "--boot-nj") bootNJ = true;
            else if (a == "--boot-shape") bootShape = stod(value());
            else if (a == "--bio") bio = true;
            else if (a == "--bme") bme = true;
            else if (a == "--upgma") upg = true;
            else if (a == "--no-nni") nni = false;
            else throw runtime_error("Unknown option: " + a);
        }
        ifstream in(argv[1]);
        if (!in) throw runtime_error("Cannot open input");
        vector<string> labels;
        unordered_map<string, int> ids;
        vector<Tree> trees;
        string line;
        while (getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == string::npos) continue;
            Parser p(line, labels, ids, trees.empty());
            trees.push_back(p.parse());
        }
        if (trees.empty()) throw runtime_error("No gene trees");
        if (labels.size() > 256) throw runtime_error("At most 256 taxa supported");
        vector<Gene> genes;
        for (const Tree& t : trees) {
            genes.emplace_back(t);
            genes.back().weights(weightMode, supportPower, lengthScale, threshold);
        }
        int protectedLeaf = ids.count("0") ? ids.at("0") : -1;
        QuartetScorer scorer(genes, externalAlpha, weightMode == 4, lengthScale,
                             maximumBins, normalization, tipAlpha, protectedLeaf);
        vector<double> quality(trees.size(), 1);
        const vector<double>* qualityPointer = nullptr;
        if (geneWeightAlpha > 0 || geneSupportPower > 0) {
            double total = 0;
            for (int gi = 0; gi < int(trees.size()); ++gi) {
                const Tree& t = trees[gi];
                double sum = 0;
                int count = 0;
                for (int v = 0; v < int(t.adj.size()); ++v)
                    for (Edge e : t.adj[v]) if (v < e.to) {
                        if (v == protectedLeaf || e.to == protectedLeaf) continue;
                        if (geneTerminalOnly && v >= t.n && e.to >= t.n) continue;
                        sum += e.length;
                        ++count;
                    }
                quality[gi] = exp(-min(700.0, geneWeightAlpha * sum / max(1, count)));
                if (geneSupportPower > 0) {
                    const Gene& gene = genes[gi];
                    double support = 0;
                    int edges = 0;
                    for (int v : gene.order) if (v >= gene.n && v != gene.root) {
                        support += 1 - gene.failure[v];
                        ++edges;
                    }
                    quality[gi] *= pow(max(.001, support / max(1, edges)), geneSupportPower);
                }
                total += quality[gi];
            }
            for (double& w : quality) w *= trees.size() / total;
            qualityPointer = &quality;
            scorer.setWeights(quality);
            double sumSquared = 0;
            for (double w : quality) sumSquared += w * w;
            cerr << "Gene weighting: effective loci "
                 << trees.size() * trees.size() / sumSquared << '\n';
        }
        unsigned long long seed = 1;
        if (const char* s = getenv("SEED")) seed = strtoull(s, nullptr, 10);
        mt19937_64 rng(seed);
        if (distancePrior > 0 && (dp || spr || bootstrap || normalization > 0))
            throw runtime_error("Distance-prior search requires plain NNI");
        Matrix guide;
        if (distancePrior > 0) {
            int anchor = ids.count("0") ? ids.at("0") : 0;
            guide = distances(trees, 3, supportPower, anchor, qualityPointer, distanceZero);
        }
        Tree bestTree;
        vector<Tree> candidates;
        double bestScore = -numeric_limits<double>::infinity();
        for (int st = 0; st < starts; ++st) {
            if (st && seconds() + 5 >= budget) break;
            int mode = distanceMode;
            if (st > 0 && !sameStarts) {
                int remaining = st;
                for (int candidate : {2, 0, 3, 1, 4}) if (candidate != distanceMode)
                    if (--remaining == 0) { mode = candidate; break; }
            }
            int anchor = ids.count("0") ? ids.at("0") : 0;
            Matrix d = distances(trees, mode, supportPower, anchor, qualityPointer, distanceZero);
            Tree t = upg ? upgma(d) : neighborJoining(d, bio);
            if (bme) optimizeBME(t, d);
            candidates.push_back(t);
            cerr << "Starting tree " << st + 1 << ", distance " << mode
                 << ", elapsed " << seconds() << '\n';
            if (nni) optimizeNNI(t, scorer, rng, budget - 3, sweeps, false,
                                 distancePrior > 0 ? &guide : nullptr, distancePrior,
                                 uniformGuide);
            candidates.push_back(t);
            double score = scorer.treeScore(t);
            cerr << setprecision(16) << "Tree score: " << score
                 << ", elapsed " << seconds() << '\n';
            if (score > bestScore) { bestScore = score; bestTree = move(t); }
        }
        if (orderConsensus && candidates.size() >= 4) {
            vector<Tree> orders;
            for (int i = 1; i < int(candidates.size()); i += 2) orders.push_back(candidates[i]);
            int anchor = ids.count("0") ? ids.at("0") : 0;
            bestTree = splitConsensus(orders, candidates[1], anchor, 1, budget - 1);
        }
        if (dp && seconds() + 3 < budget) {
            int anchor = ids.count("0") ? ids.at("0") : 0;
            ClusterSearch search(int(labels.size()), anchor, dpMaxSize, dpFrequency);
            search.addGenes(trees);
            for (const Tree& t : candidates) search.addTree(t);
            bestTree = search.solve(scorer, budget - 5);
            if (nni && seconds() + 2 < budget)
                optimizeNNI(bestTree, scorer, rng, budget - 2, 10);
        }
        if (spr && seconds() + 2 < budget)
            optimizeSPR(bestTree, scorer, rng, budget - 2, sprRadius, sprSize, sprPasses);
        if (bootstrap && seconds() + 3 < budget) {
            int anchor = ids.count("0") ? ids.at("0") : 0;
            bestTree = bootstrapInference(bestTree, candidates, trees, scorer, rng,
                                          anchor, bootstrap, bootDP, bootPrior,
                                          budget - 3, dpMaxSize, dpFrequency,
                                          bootBayes, bootShape, bootNJ, distanceMode,
                                          supportPower, bio, bme, qualityPointer);
        }
        bestTree.validate();
        ofstream out(argv[2]);
        if (!out) throw runtime_error("Cannot open output");
        out << bestTree.newick(labels);
        out.close();
        if (!out) throw runtime_error("Cannot write output");
        cerr << "Finished: " << scorer.evaluations << " quartet evaluations, "
             << scorer.cacheHits << " cached, " << seconds() << " s\n";
        return 0;
    } catch (const exception& e) {
        cerr << "Inference failed: " << e.what() << '\n';
        return 1;
    }
}
