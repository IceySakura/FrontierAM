#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
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
static auto start_time = Clock::now();
static double elapsed() {
    return chrono::duration<double>(Clock::now() - start_time).count();
}

struct Edge {
    int to;
    double length = 0.0;
    double support = -1.0;
};

struct Tree {
    vector<vector<Edge>> adj;
    vector<int> taxon;
    int node(int label = -1) {
        adj.emplace_back();
        taxon.push_back(label);
        return (int)adj.size() - 1;
    }
    void link(int a, int b, double length = 0.0, double support = -1.0) {
        adj[a].push_back({b, length, support});
        adj[b].push_back({a, length, support});
    }
    void unlink(int a, int b) {
        auto &v = adj[a];
        v.erase(remove_if(v.begin(), v.end(), [b](Edge e) { return e.to == b; }), v.end());
        auto &w = adj[b];
        w.erase(remove_if(w.begin(), w.end(), [a](Edge e) { return e.to == a; }), w.end());
    }
    void suppress_root(int r) {
        if (taxon[r] == -1 && adj[r].size() == 2) {
            Edge a = adj[r][0], b = adj[r][1];
            unlink(r, a.to);
            unlink(r, b.to);
            double s = max(a.support, b.support);
            link(a.to, b.to, a.length + b.length, s);
        }
    }
};

struct Labels {
    vector<string> names;
    unordered_map<string, int> index;
    int get(const string &s) {
        auto it = index.find(s);
        if (it != index.end()) return it->second;
        int i = (int)names.size();
        names.push_back(s);
        index[s] = i;
        return i;
    }
};

struct Parser {
    const string &text;
    size_t pos = 0;
    Labels &labels;
    Tree tree;
    void ws() {
        while (pos < text.size()) {
            if (isspace((unsigned char)text[pos])) { ++pos; continue; }
            if (text[pos] == '[') {
                int depth = 1;
                ++pos;
                while (pos < text.size() && depth) {
                    if (text[pos] == '[') ++depth;
                    if (text[pos] == ']') --depth;
                    ++pos;
                }
                continue;
            }
            break;
        }
    }
    string token() {
        ws();
        string s;
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
            return s;
        }
        while (pos < text.size() && !isspace((unsigned char)text[pos]) &&
               string("(),:;[]").find(text[pos]) == string::npos)
            s += text[pos++];
        return s;
    }
    struct Sub { int node; double length, support; };
    Sub subtree() {
        ws();
        if (pos >= text.size()) throw runtime_error("Unexpected end of Newick");
        int u;
        double support = -1.0, length = 0.0;
        if (text[pos] == '(') {
            ++pos;
            u = tree.node();
            while (true) {
                Sub s = subtree();
                tree.link(u, s.node, s.length, s.support);
                ws();
                if (pos >= text.size()) throw runtime_error("Unclosed Newick group");
                if (text[pos] == ')') { ++pos; break; }
                if (text[pos++] != ',') throw runtime_error("Expected comma");
            }
            string s = token();
            if (!s.empty()) {
                char *end = nullptr;
                double x = strtod(s.c_str(), &end);
                if (end == s.c_str() + s.size() && isfinite(x) && x >= 0 && x <= 1)
                    support = x;
            }
        } else {
            string s = token();
            if (s.empty()) throw runtime_error("Missing leaf name");
            u = tree.node(labels.get(s));
        }
        ws();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            string s = token();
            char *end = nullptr;
            length = strtod(s.c_str(), &end);
            if (end != s.c_str() + s.size() || !isfinite(length) || length < 0)
                throw runtime_error("Invalid branch length");
        }
        return {u, length, support};
    }
    Tree parse() {
        int r = subtree().node;
        ws();
        if (pos >= text.size() || text[pos] != ';') throw runtime_error("Missing semicolon");
        tree.suppress_root(r);
        return move(tree);
    }
};

static string quote_name(const string &s) {
    if (!s.empty() && s.find_first_of(" \t\r\n(),:;[]'") == string::npos) return s;
    string out = "'";
    for (char c : s) { out += c; if (c == '\'') out += '\''; }
    return out + "'";
}

static string newick(const Tree &t, const Labels &labels) {
    int root = -1;
    for (int i = 0; i < (int)t.adj.size(); ++i)
        if (t.adj[i].size() >= 3) { root = i; break; }
    if (root < 0) throw runtime_error("No internal tree node");
    function<string(int,int)> visit = [&](int u, int p) -> string {
        if (t.taxon[u] >= 0) return quote_name(labels.names.at(t.taxon[u]));
        string out = "(";
        bool first = true;
        for (auto e : t.adj[u]) if (e.to != p) {
            if (!first) out += ',';
            first = false;
            out += visit(e.to, u);
        }
        return out + ")";
    };
    return visit(root, -1) + ";\n";
}

using Matrix = vector<vector<double>>;

struct Options {
    string distance = "support";
    string method = "nj";
    double power = 1.0;
    double blend = 0.0;
    double threshold = 0.0;
    double normalize = 0.0;
    int nni = 0;
    double qpower = 1.0;
    double qblend = 0.0;
    double qthreshold = 0.0;
    double qexternal = 0.0;
    double qlength = 0.0;
    double qsupport = 1.0;
    double qnormalize = 0.0;
    int spr = 0;
    int radius = 5;
    bool report_score = false;
    int starts = 1;
    bool cluster_dp = false;
    double dp_fraction = 0.65;
    double minz = 0.0;
    int bootstrap = 0;
    double bootstrap_fraction = 1.0;
    double bootstrap_prior = 1.0;
    bool qproduct = false;
    int qmax = 0;
    int bme = 0;
    double trim = 0.0;
    double quantile = -1.0;
    int ensemble = 0;
    bool legacy_missing = false;
    bool qand = false;
    double dquality = 0.0;
    bool zero_floor = false;
    int cv = 0;
    double cv_z = 1.0;
    double qmixture = 1.0;
    double limit = 115.0;
    string initial;
};

static double reliability(double s) {
    return s < 0 ? 1.0 : max(0.0, min(1.0, (3.0 * s - 1.0) / 2.0));
}

static double support_rate(const Tree &tree) {
    vector<double> rates;
    double length = 0;
    int leaves = 0;
    for (int u = 0; u < (int)tree.adj.size(); ++u) {
        if (tree.taxon[u] >= 0) ++leaves;
        for (Edge e : tree.adj[u]) if (u < e.to) {
            length += e.length;
            double r = reliability(e.support);
            if (tree.taxon[u] < 0 && tree.taxon[e.to] < 0 && e.support >= 0 &&
                r > 0.05 && r < 0.98 && e.length > 1e-5)
                rates.push_back(-log1p(-r)/e.length);
        }
    }
    if (rates.size() >= 3) {
        nth_element(rates.begin(), rates.begin()+rates.size()/2, rates.end());
        return rates[rates.size()/2];
    }
    return 8.0*leaves/max(length, 1e-8);
}

static double edge_reliability(double support, double length, double rate, bool legacy) {
    if (support >= 0 || legacy) return reliability(support);
    // Unlabeled, effectively zero-length resolutions have no support evidence.
    if (length <= 1e-5) return 0;
    return -expm1(-rate*length);
}

static Matrix distances(const vector<Tree> &genes, int n, const Options &opt) {
    Matrix result(n, vector<double>(n));
    vector<Matrix> history;
    for (const Tree &t : genes) {
        Matrix local(n, vector<double>(n));
        double total = 0;
        int edges = 0;
        for (int i = 0; i < (int)t.adj.size(); ++i)
            for (auto e : t.adj[i]) { total += e.length; ++edges; }
        double average_edge = max(1e-12, total/edges);
        double rate = support_rate(t);
        total *= 0.5;
        vector<vector<pair<int,double>>> weighted(t.adj.size());
        double information = 0;
        for (int u = 0; u < (int)t.adj.size(); ++u) {
            for (Edge e : t.adj[u]) {
                bool internal = t.taxon[u] < 0 && t.taxon[e.to] < 0;
                double r = edge_reliability(e.support,e.length,rate,opt.legacy_missing);
                double support = e.support < 0 ? (1+2*r)/3 : e.support;
                if (internal) information += max(0.0,e.length-1e-5)*r*0.5;
                double w = 1.0;
                if (opt.distance == "length") w = pow(e.length, opt.power);
                if (opt.distance == "support" && internal)
                    w = opt.blend + (1-opt.blend)*pow(r,opt.power);
                if (opt.distance == "rawsupport" && internal)
                    w = pow(support, opt.power);
                if (opt.distance == "contract" && internal)
                    w = support > opt.threshold ? 1.0 : 0.0;
                if (opt.distance == "lengthsupport")
                    w = pow(e.length,opt.power)*(internal ? r : 1.0);
                if (opt.distance == "mix")
                    w = opt.blend+(1-opt.blend)*e.length/average_edge;
                if (opt.distance == "loglength")
                    w = log1p(opt.power*e.length/average_edge);
                weighted[u].emplace_back(e.to,w);
            }
        }
        function<void(int,int,int,double)> walk = [&](int u, int p, int from, double d) {
            if (t.taxon[u] >= 0) local[from][t.taxon[u]] = d;
            for (auto [v,w] : weighted[u]) if (v != p) walk(v, u, from, d+w);
        };
        for (int u = 0; u < (int)t.adj.size(); ++u)
            if (t.taxon[u] >= 0) walk(u, -1, t.taxon[u], 0);
        double scale = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) scale += local[i][j];
        scale /= n * (n - 1) / 2;
        scale = pow(max(1e-12, scale), opt.normalize);
        double quality = opt.dquality > 0 ? information/(information+opt.dquality) : 1;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                local[i][j] /= scale;
                result[i][j] += quality*local[i][j]/genes.size();
            }
        if (opt.trim > 0 || opt.quantile >= 0) history.push_back(move(local));
    }
    if (!history.empty()) {
        vector<double> samples(history.size());
        for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) {
            for (int g = 0; g < (int)history.size(); ++g) samples[g] = history[g][i][j];
            sort(samples.begin(), samples.end());
            double value = 0;
            if (opt.quantile >= 0) {
                double index = clamp(opt.quantile,0.0,1.0)*(samples.size()-1);
                int a = int(index), b = min(a+1,int(samples.size())-1);
                value = samples[a]*(b-index)+samples[b]*(index-a);
                if (a == b) value = samples[a];
            } else {
                int skip = min(int(samples.size()*opt.trim),int(samples.size()/2)-1);
                for (int g = skip; g < (int)samples.size()-skip; ++g) value += samples[g];
                value /= samples.size()-2*skip;
            }
            result[i][j] = result[j][i] = value;
        }
    }
    return result;
}

static Tree join(const Matrix &input, const string &method) {
    int n = (int)input.size();
    int cap = 2 * n;
    Matrix d(cap, vector<double>(cap));
    Matrix v(cap, vector<double>(cap));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) d[i][j] = v[i][j] = input[i][j];
    Tree tree;
    vector<int> active(n), size(cap, 1);
    for (int i = 0; i < n; ++i) active[i] = tree.node(i);
    while (active.size() > 3) {
        int m = (int)active.size(), a = -1, b = -1;
        vector<double> r(cap);
        for (int i : active) for (int j : active) r[i] += d[i][j];
        double best = numeric_limits<double>::infinity();
        for (int ii = 0; ii < m; ++ii) for (int jj = 0; jj < ii; ++jj) {
            int i = active[ii], j = active[jj];
            double q = method == "upgma" ? d[i][j] : (m - 2) * d[i][j] - r[i] - r[j];
            if (q < best) { best = q; a = i; b = j; }
        }
        int u = tree.node();
        tree.link(u, a); tree.link(u, b);
        double lambda = 0.5;
        if (method == "bionj" && v[a][b] > 1e-12) {
            double s = 0;
            for (int k : active) if (k != a && k != b) s += v[b][k] - v[a][k];
            lambda = clamp(0.5 + s / (2 * (m - 2) * v[a][b]), 0.0, 1.0);
        }
        if (method == "upgma") lambda = double(size[a]) / (size[a] + size[b]);
        double la = 0.5 * (d[a][b] + (r[a] - r[b]) / (m - 2));
        double lb = d[a][b] - la;
        for (int k : active) if (k != a && k != b) {
            double x = lambda * (d[a][k] - la) + (1 - lambda) * (d[b][k] - lb);
            if (method == "upgma") x = lambda * d[a][k] + (1 - lambda) * d[b][k];
            d[u][k] = d[k][u] = x;
            v[u][k] = v[k][u] = lambda * v[a][k] + (1 - lambda) * v[b][k]
                                       - lambda * (1 - lambda) * v[a][b];
        }
        size[u] = size[a] + size[b];
        active.erase(remove(active.begin(), active.end(), a), active.end());
        active.erase(remove(active.begin(), active.end(), b), active.end());
        active.push_back(u);
    }
    int u = tree.node();
    for (int a : active) tree.link(u, a);
    return tree;
}

struct RNode {
    int left = -1, right = -1, taxon = -1;
    double failure = 0;
    double attenuation = 1;
    int max_cutoff = 0;
    double independent_failure = 0;
};

static vector<RNode> root_gene(const Tree &t, const Options &opt, bool maximum = false) {
    vector<RNode> nodes;
    double scale = 0;
    int leaf_count = 0;
    for (int u = 0; u < (int)t.adj.size(); ++u) {
        if (t.taxon[u] >= 0) ++leaf_count;
        for (Edge e : t.adj[u]) scale += e.length;
    }
    scale = pow(max(1e-12, scale / leaf_count), opt.qnormalize);
    double rate = support_rate(t);
    vector<double> thresholds;
    if (maximum) for (int i = 0; i < opt.qmax; ++i)
        thresholds.push_back((1+2*pow((i+0.5)/opt.qmax,1/opt.qpower))/3);
    function<int(int,int,double,double)> visit = [&](int u, int p, double support, double length) {
        RNode x;
        double r = edge_reliability(support,length,rate,opt.legacy_missing);
        if (maximum) {
            double s = support < 0 ? (1+2*r)/3 : support;
            x.max_cutoff = lower_bound(thresholds.begin(),thresholds.end(),s)-thresholds.begin();
        } else if (opt.qthreshold > 0) {
            double s = support < 0 ? (1+2*r)/3 : support;
            r = s > opt.qthreshold ? 1 : 0;
        }
        double powered = pow(r,opt.qpower);
        double support_weight = opt.qblend+(1-opt.qblend)*opt.qsupport*powered;
        double length_failure = exp(-opt.qlength*length/scale);
        x.failure = (maximum ? 1.0 : opt.qproduct ? support_weight : 1-support_weight) *
                    length_failure;
        x.independent_failure = maximum ? (1-powered)*length_failure : x.failure;
        x.attenuation = exp(-opt.qexternal * length / scale);
        x.taxon = t.taxon[u];
        vector<int> ch;
        for (Edge e : t.adj[u]) if (e.to != p) ch.push_back(visit(e.to, u, e.support, e.length));
        if (!ch.empty()) {
            if (ch.size() != 2) throw runtime_error("Gene tree is not binary");
            x.left = ch[0]; x.right = ch[1];
        }
        nodes.push_back(x);
        return int(nodes.size()) - 1;
    };
    int leaf = 0;
    while (leaf < (int)t.adj.size() && t.taxon[leaf] < 0) ++leaf;
    int neighbor = t.adj[leaf][0].to;
    int a = visit(leaf, neighbor, -1, t.adj[leaf][0].length);
    int b = visit(neighbor, leaf, -1, 0);
    nodes.push_back({a, b, -1, 0});
    return nodes;
}

struct QuartetScorer {
    // Pairs carry the probability that every edge from their LCA failed.
    // Triples stop that path where the third leaf branches off.
    struct State {
        double c[4] = {};
        double u[6] = {};
        double p[6] = {};
        double t[12] = {};
        unsigned mask = 0;
    };
    vector<vector<RNode>> genes;
    vector<State> state;
    struct State3 {
        double c[3] = {};
        double u[6] = {};
        double p[6] = {};
        double t[9] = {};
        unsigned mask = 0;
    };
    vector<State3> state3;
    vector<double> weights;
    vector<double> base_weights;
    vector<int> group;
    int loci = 0;
    vector<array<double,3>> last_gene_scores;
    bool product = false;
    int maximum_levels = 0;
    double maximum_weight = 1.0;
    vector<double> maximum_pairs;
    long calls = 0;
    explicit QuartetScorer(const vector<Tree> &input, const Options &opt, bool fast = true) {
        product = opt.qproduct;
        loci = input.size();
        bool maximum_model = opt.qmax > 0 && opt.qpower > 0 && !opt.qand &&
                             !opt.qproduct && opt.qsupport == 1;
        if (maximum_model) maximum_weight = clamp(opt.qmixture,0.0,1.0);
        if (fast && maximum_model) maximum_levels = opt.qmax;
        size_t size = 0;
        for (int g = 0; g < (int)input.size(); ++g) {
            const auto &t = input[g];
            if (maximum_levels) {
                genes.push_back(root_gene(t,opt,true));
                weights.push_back(1);
                group.push_back(g);
                size = max(size,genes.back().size());
                continue;
            }
            if (maximum_model && maximum_weight < 1) {
                Options independent = opt;
                independent.qmax = 0;
                independent.qthreshold = independent.qblend = 0;
                genes.push_back(root_gene(t,independent));
                weights.push_back(1-maximum_weight);
                group.push_back(g);
                size = max(size,genes.back().size());
            }
            int levels = max(1, opt.qmax);
            for (int level = 0; level < levels; ++level) {
                Options o = opt;
                if (opt.qmax) {
                    double r = pow((level+0.5)/levels, 1/opt.qpower);
                    o.qthreshold = (1+2*r)/3;
                    o.qblend = 0;
                    o.qpower = 1;
                }
                for (int part = 0; part < (opt.qand ? 3 : 1); ++part) {
                    Options part_opt = o;
                    double weight = maximum_weight/levels;
                    if (opt.qand) {
                        if (part == 0) part_opt.qlength = 0;
                        if (part == 1) { part_opt.qsupport = 0; part_opt.qblend = 0; }
                        if (part == 2) weight = -weight;
                    }
                    if (!weight) continue;
                    genes.push_back(root_gene(t, part_opt));
                    weights.push_back(weight);
                    group.push_back(g);
                    size = max(size, genes.back().size());
                }
            }
        }
        state.resize(size);
        state3.resize(size);
        last_gene_scores.resize(genes.size());
        base_weights = weights;
        maximum_pairs.resize(size*6*maximum_levels);
    }
    array<double,3> maximum_score(const vector<int> &color, int duplicated) {
        ++calls;
        constexpr int pa[6] = {0,0,0,1,1,2};
        constexpr int pb[6] = {1,2,3,2,3,3};
        constexpr int tc[6][2] = {{2,3},{1,3},{1,2},{0,3},{0,2},{0,1}};
        constexpr int qp[3][2] = {{0,5},{1,4},{2,3}};
        int levels = maximum_levels;
        double inverse = 1.0/levels;
        auto row = [&](int node, int pair) {
            return maximum_pairs.data()+(size_t(node)*6+pair)*levels;
        };
        array<double,3> result{};
        for (int gene = 0; gene < (int)genes.size(); ++gene) {
            const auto &nodes = genes[gene];
            last_gene_scores[gene] = {};
            if (!weights[gene]) continue;
            array<double,3> local{};
            for (int u = 0; u < (int)nodes.size(); ++u) {
                const RNode &node = nodes[u];
                State &s = state[u];
                s = {};
                if (node.taxon >= 0) {
                    int c = color[node.taxon];
                    if (c >= 0) {
                        if (duplicated < 0) s.c[c] = 1;
                        else if (c == duplicated) s.c[0] = s.c[1] = 1;
                        else s.c[c == (duplicated+1)%3 ? 2 : 3] = 1;
                    }
                    for (int i = 0; i < 4; ++i) if (s.c[i]) s.mask |= 1U<<i;
                    continue;
                }
                int li = node.left, ri = node.right;
                const State &l = state[li], &r = state[ri];
                const RNode &ln = nodes[li], &rn = nodes[ri];
                s.mask = l.mask | r.mask;
                double cl[4], cr[4];
                for (int i = 0; i < 4; ++i) {
                    cl[i] = ln.attenuation*l.c[i]; cr[i] = rn.attenuation*r.c[i];
                    s.c[i] = cl[i]+cr[i];
                }
                if (!(s.mask & (s.mask-1))) continue;
                double wl[6] = {}, wr[6] = {};
                bool active_left[6] = {}, active_right[6] = {};
                for (int p = 0; p < 6; ++p) {
                    int a = pa[p], b = pb[p];
                    unsigned mask = (1U<<a)|(1U<<b);
                    if ((s.mask & mask) != mask) continue;
                    active_left[p] = ln.taxon < 0 && (l.mask & mask) == mask;
                    active_right[p] = rn.taxon < 0 && (r.mask & mask) == mask;
                    double cross = cl[a]*cr[b]+cl[b]*cr[a];
                    s.u[p] = l.u[p]+r.u[p]+cross;
                    const double *pl = row(li,p), *pr = row(ri,p);
                    double *current = row(u,p);
                    double sum_l = 0, sum_r = 0;
                    for (int k = 0; k < levels; ++k) {
                        double x = active_left[p] && k >= ln.max_cutoff ? ln.failure*pl[k] : 0;
                        double y = active_right[p] && k >= rn.max_cutoff ? rn.failure*pr[k] : 0;
                        current[k] = cross+x+y;
                        sum_l += x; sum_r += y;
                    }
                    double il = ln.independent_failure*l.p[p];
                    double ir = rn.independent_failure*r.p[p];
                    wl[p] = maximum_weight*sum_l*inverse+(1-maximum_weight)*il;
                    wr[p] = maximum_weight*sum_r*inverse+(1-maximum_weight)*ir;
                    s.p[p] = cross+il+ir;
                }
                unsigned rest = s.mask & (s.mask-1);
                if (!(rest & (rest-1))) continue;
                // Only pair failures depend on thresholds; triples can be averaged now.
                for (int p = 0; p < 6; ++p) for (int j = 0; j < 2; ++j) {
                    int k = tc[p][j], index = 2*p+j;
                    s.t[index] = ln.attenuation*l.t[index]+rn.attenuation*r.t[index]+
                                 (l.u[p]-wl[p])*cr[k]+(r.u[p]-wr[p])*cl[k];
                }
                if (s.mask != 15) continue;
                auto product_mean = [&](int p, int z) {
                    if (!active_left[p] || !active_right[z] ||
                        ln.failure == 0 || rn.failure == 0) return 0.0;
                    const double *pl = row(li,p), *pr = row(ri,z);
                    double sum = 0;
                    for (int k = max(ln.max_cutoff,rn.max_cutoff); k < levels; ++k)
                        sum += pl[k]*pr[k];
                    return ln.failure*rn.failure*sum*inverse;
                };
                for (int q = 0; q < 3; ++q) {
                    int p = qp[q][0], z = qp[q][1];
                    // A balanced quartet must use the same threshold for both pairs.
                    double independent = ln.independent_failure*rn.independent_failure*
                                         (l.p[p]*r.p[z]+l.p[z]*r.p[p]);
                    double value = l.u[p]*r.u[z]+r.u[p]*l.u[z]-
                                   maximum_weight*(product_mean(p,z)+product_mean(z,p))-
                                   (1-maximum_weight)*independent;
                    for (int x : {p,z})
                        value += ln.attenuation*(l.t[2*x]*cr[tc[x][1]]+l.t[2*x+1]*cr[tc[x][0]])+
                                 rn.attenuation*(r.t[2*x]*cl[tc[x][1]]+r.t[2*x+1]*cl[tc[x][0]]);
                    local[q] += value;
                }
            }
            last_gene_scores[gene] = local;
            for (int q = 0; q < 3; ++q) result[q] += weights[gene]*local[q];
        }
        return result;
    }
    array<double,3> score(const vector<int> &color, int duplicated = -1) {
        if (maximum_levels) return maximum_score(color,duplicated);
        ++calls;
        constexpr int pa[6] = {0,0,0,1,1,2};
        constexpr int pb[6] = {1,2,3,2,3,3};
        constexpr int tc[6][2] = {{2,3},{1,3},{1,2},{0,3},{0,2},{0,1}};
        constexpr int qp[3][2] = {{0,5},{1,4},{2,3}};
        array<double,3> result{};
        for (int gene = 0; gene < (int)genes.size(); ++gene) {
            const auto &nodes = genes[gene];
            last_gene_scores[gene] = {};
            if (!weights[gene]) continue;
            array<double,3> local{};
            for (int u = 0; u < (int)nodes.size(); ++u) {
                const RNode &node = nodes[u];
                State &s = state[u];
                if (node.taxon >= 0) {
                    s = {};
                    int c = color[node.taxon];
                    if (c >= 0) {
                        if (duplicated < 0) s.c[c] = 1;
                        else if (c == duplicated) s.c[0] = s.c[1] = 1;
                        else s.c[c == (duplicated + 1) % 3 ? 2 : 3] = 1;
                    }
                    for (int i = 0; i < 4; ++i) if (s.c[i]) s.mask |= 1U<<i;
                    continue;
                }
                const State &l = state[node.left], &r = state[node.right];
                double fl = nodes[node.left].failure, fr = nodes[node.right].failure;
                double el = nodes[node.left].attenuation, er = nodes[node.right].attenuation;
                unsigned mask = l.mask | r.mask;
                unsigned rest = mask & (mask-1);
                if (!(rest & (rest-1))) {
                    s = {};
                    s.mask = mask;
                    for (int i = 0; i < 4; ++i) s.c[i] = el*l.c[i]+er*r.c[i];
                    if (rest) {
                        constexpr int index[4][4] = {{-1,0,1,2},{0,-1,3,4},{1,3,-1,5},{2,4,5,-1}};
                        int a = __builtin_ctz(mask), b = __builtin_ctz(rest);
                        int p = index[a][b];
                        double cross = el*er*(l.c[a]*r.c[b]+l.c[b]*r.c[a]);
                        s.u[p] = l.u[p]+r.u[p]+cross;
                        s.p[p] = fl*l.p[p]+fr*r.p[p]+cross;
                    }
                    continue;
                }
                s.mask = mask;
                double pl[6], pr[6], wl[6], wr[6];
                double cl[4], cr[4];
                for (int i = 0; i < 4; ++i) {
                    cl[i] = el * l.c[i]; cr[i] = er * r.c[i];
                    s.c[i] = cl[i] + cr[i];
                }
                for (int p = 0; p < 6; ++p) {
                    int a = pa[p], b = pb[p];
                    pl[p] = l.u[p];
                    pr[p] = r.u[p];
                    wl[p] = fl * l.p[p];
                    wr[p] = fr * r.p[p];
                    double cross = cl[a]*cr[b] + cl[b]*cr[a];
                    s.p[p] = wl[p] + wr[p] + cross;
                    s.u[p] = pl[p] + pr[p] + cross;
                    for (int j = 0; j < 2; ++j) {
                        int k = tc[p][j], idx = 2*p + j;
                        double vl = product ? wl[p] : pl[p]-wl[p];
                        double vr = product ? wr[p] : pr[p]-wr[p];
                        s.t[idx] = el*l.t[idx] + er*r.t[idx] + vl*cr[k] + vr*cl[k];
                    }
                }
                if (mask == 15) for (int q = 0; q < 3; ++q) {
                    int p = qp[q][0], z = qp[q][1];
                    double path = wl[p]*wr[z] + wr[p]*wl[z];
                    double v = product ? path : pl[p]*pr[z]+pr[p]*pl[z]-path;
                    for (int x : {p,z}) {
                        v += el*(l.t[2*x]*cr[tc[x][1]] + l.t[2*x+1]*cr[tc[x][0]]) +
                             er*(r.t[2*x]*cl[tc[x][1]] + r.t[2*x+1]*cl[tc[x][0]]);
                    }
                    local[q] += v;
                }
            }
            last_gene_scores[gene] = local;
            for (int q = 0; q < 3; ++q) result[q] += weights[gene]*local[q];
        }
        return result;
    }
    double trip_score(const vector<int> &color) {
        if (maximum_levels) {
            double result = 0;
            for (int i = 0; i < 3; ++i) result += score(color,i)[0]/2;
            return result;
        }
        ++calls;
        constexpr int pa[6] = {0,1,2,0,0,1};
        constexpr int pb[6] = {0,1,2,1,2,2};
        constexpr int tp[9] = {0,0,1,1,2,2,3,4,5};
        constexpr int tc[9] = {1,2,0,2,0,1,2,1,0};
        double result = 0;
        for (int gene = 0; gene < (int)genes.size(); ++gene) {
            if (!weights[gene]) continue;
            const auto &nodes = genes[gene];
            double local = 0;
            for (int u = 0; u < (int)nodes.size(); ++u) {
                const RNode &node = nodes[u];
                State3 &s = state3[u];
                if (node.taxon >= 0) {
                    s = {};
                    int c = color[node.taxon];
                    if (c >= 0) { s.c[c] = 1; s.mask = 1U<<c; }
                    continue;
                }
                const State3 &l = state3[node.left], &r = state3[node.right];
                double fl = nodes[node.left].failure, fr = nodes[node.right].failure;
                double el = nodes[node.left].attenuation, er = nodes[node.right].attenuation;
                unsigned mask = l.mask | r.mask;
                if (!(mask & (mask-1))) {
                    s = {};
                    s.mask = mask;
                    if (mask) {
                        int c = __builtin_ctz(mask);
                        double cross = el*er*l.c[c]*r.c[c];
                        s.c[c] = el*l.c[c]+er*r.c[c];
                        s.u[c] = l.u[c]+r.u[c]+cross;
                        s.p[c] = fl*l.p[c]+fr*r.p[c]+cross;
                    }
                    continue;
                }
                s.mask = mask;
                double wl[6], wr[6], cl[3], cr[3];
                for (int i = 0; i < 3; ++i) {
                    cl[i] = el*l.c[i]; cr[i] = er*r.c[i];
                    s.c[i] = cl[i]+cr[i];
                }
                for (int p = 0; p < 6; ++p) {
                    int a = pa[p], b = pb[p];
                    double cross = cl[a]*cr[b];
                    if (a != b) cross += cl[b]*cr[a];
                    wl[p] = fl*l.p[p]; wr[p] = fr*r.p[p];
                    s.p[p] = wl[p]+wr[p]+cross;
                    s.u[p] = l.u[p]+r.u[p]+cross;
                }
                for (int j = 0; j < 9; ++j) {
                    int p = tp[j], k = tc[j];
                    double vl = product ? wl[p] : l.u[p]-wl[p];
                    double vr = product ? wr[p] : r.u[p]-wr[p];
                    s.t[j] = el*l.t[j]+er*r.t[j]+vl*cr[k]+vr*cl[k];
                }
                if (mask == 7) for (int i = 0; i < 3; ++i) {
                    int z = 5-i;
                    double path = wl[i]*wr[z]+wr[i]*wl[z];
                    local += product ? path : l.u[i]*r.u[z]+r.u[i]*l.u[z]-path;
                    int a = 2*i, b = 2*i+1, c = 8-i;
                    local += el*(l.t[a]*cr[tc[b]]+l.t[b]*cr[tc[a]]+l.t[c]*cr[i])+
                              er*(r.t[a]*cl[tc[b]]+r.t[b]*cl[tc[a]]+r.t[c]*cl[i]);
                }
            }
            result += weights[gene]*local;
        }
        return result;
    }
};

static void paint(const Tree &t, int u, int p, int color, vector<int> &colors) {
    if (t.taxon[u] >= 0) colors[t.taxon[u]] = color;
    for (Edge e : t.adj[u]) if (e.to != p) paint(t, e.to, u, color, colors);
}

static void improve_bme(Tree &tree, const Matrix &d, int rounds) {
    using Profile = vector<pair<int,double>>;
    function<void(int,int,double,Profile&)> collect = [&](int u, int p, double weight, Profile &v) {
        if (tree.taxon[u] >= 0) { v.emplace_back(tree.taxon[u],weight); return; }
        for (Edge e : tree.adj[u]) if (e.to != p) collect(e.to,u,weight/2,v);
    };
    auto mean = [&](const Profile &a, const Profile &b) {
        double total = 0;
        for (auto [i,x] : a) for (auto [j,y] : b) total += x*y*d[i][j];
        return total;
    };
    for (int rep = 0; rep < rounds; ++rep) {
        int moves = 0;
        vector<pair<int,int>> edges;
        for (int u = 0; u < (int)tree.adj.size(); ++u) if (tree.adj[u].size() == 3)
            for (Edge e : tree.adj[u]) if (e.to > u && tree.adj[e.to].size() == 3)
                edges.emplace_back(u,e.to);
        for (auto [u,v] : edges) {
            vector<int> a,b;
            for (Edge e : tree.adj[u]) if (e.to != v) a.push_back(e.to);
            for (Edge e : tree.adj[v]) if (e.to != u) b.push_back(e.to);
            if (a.size() != 2 || b.size() != 2) continue;
            Profile p[4];
            collect(a[0],u,1,p[0]); collect(a[1],u,1,p[1]);
            collect(b[0],v,1,p[2]); collect(b[1],v,1,p[3]);
            array<double,3> sums = {mean(p[0],p[1])+mean(p[2],p[3]),
                                    mean(p[0],p[2])+mean(p[1],p[3]),
                                    mean(p[0],p[3])+mean(p[1],p[2])};
            int best = min_element(sums.begin(), sums.end())-sums.begin();
            if (sums[best] < sums[0]-1e-12) {
                int x = a[1], y = b[best-1];
                tree.unlink(u,x); tree.unlink(v,y); tree.link(u,y); tree.link(v,x);
                ++moves;
            }
        }
        if (!moves) break;
    }
}

static double tree_score(const Tree &tree, QuartetScorer &scorer, int n) {
    vector<int> colors(n);
    double score = 0;
    for (int u = 0; u < (int)tree.adj.size(); ++u) if (tree.adj[u].size() == 3) {
        for (int i = 0; i < 3; ++i) paint(tree, tree.adj[u][i].to, u, i, colors);
        score += scorer.trip_score(colors);
    }
    return score * 0.5;
}

static double improve_nni(Tree &tree, QuartetScorer &scorer, int n,
                          const Options &opt, mt19937 &rng) {
    vector<int> colors(n);
    double improvement = 0;
    for (int round = 0; round < opt.nni; ++round) {
        vector<pair<int,int>> edges;
        for (int u = 0; u < (int)tree.adj.size(); ++u)
            if (tree.adj[u].size() == 3)
                for (Edge e : tree.adj[u]) if (e.to > u && tree.adj[e.to].size() == 3)
                    edges.emplace_back(u, e.to);
        shuffle(edges.begin(), edges.end(), rng);
        int moves = 0;
        for (auto [u,v] : edges) {
            if (elapsed() > opt.limit) break;
            vector<int> us, vs;
            for (Edge e : tree.adj[u]) if (e.to != v) us.push_back(e.to);
            for (Edge e : tree.adj[v]) if (e.to != u) vs.push_back(e.to);
            if (us.size() != 2 || vs.size() != 2) continue;
            paint(tree, us[0], u, 0, colors); paint(tree, us[1], u, 1, colors);
            paint(tree, vs[0], v, 2, colors); paint(tree, vs[1], v, 3, colors);
            auto s = scorer.score(colors);
            int best = max_element(s.begin(), s.end()) - s.begin();
            double threshold = max(1e-6,1e-12*max(abs(s[best]),abs(s[0])));
            if (opt.minz > 0 && best) {
                double sum = 0, squares = 0;
                int g = scorer.loci;
                vector<double> differences(g);
                for (int k = 0; k < (int)scorer.genes.size(); ++k) {
                    const auto &v = scorer.last_gene_scores[k];
                    differences[scorer.group[k]] += scorer.base_weights[k]*(v[best]-v[0]);
                }
                for (double d : differences) {
                    sum += d; squares += d*d;
                }
                threshold = max(threshold, opt.minz*sqrt(max(0.0, (squares-sum*sum/g)*g/(g-1))));
            }
            if (s[best] > s[0] + threshold) {
                int a = us[1], b = vs[best - 1];
                tree.unlink(u, a); tree.unlink(v, b);
                tree.link(u, b); tree.link(v, a);
                improvement += s[best] - s[0];
                ++moves;
            }
        }
        cerr << "NNI round=" << round << " moves=" << moves << " seconds=" << elapsed() << '\n';
        if (!moves || elapsed() > opt.limit) break;
    }
    return improvement;
}

static double improve_spr(Tree &tree, QuartetScorer &scorer, int n,
                          const Options &opt, mt19937 &rng) {
    vector<int> colors(n);
    double improvement = 0;
    for (int round = 0; round < opt.spr; ++round) {
        vector<pair<int,int>> prunes;
        for (int u = 0; u < (int)tree.adj.size(); ++u) if (tree.adj[u].size() == 3)
            for (Edge e : tree.adj[u]) prunes.emplace_back(u, e.to);
        shuffle(prunes.begin(), prunes.end(), rng);
        int moves = 0;
        for (auto [u,sub] : prunes) {
            if (elapsed() > opt.limit) break;
            if (tree.adj[u].size() != 3) continue;
            bool found = false;
            vector<int> other;
            for (Edge e : tree.adj[u]) {
                if (e.to == sub) found = true;
                else other.push_back(e.to);
            }
            if (!found || other.size() != 2) continue;
            fill(colors.begin(), colors.end(), -1);
            paint(tree, sub, u, 0, colors);
            int count = std::count(colors.begin(), colors.end(), 0);
            if (count > n/2) continue;
            int a = other[0], b = other[1];
            tree.unlink(u, sub); tree.unlink(u, a); tree.unlink(u, b);
            tree.link(a, b);
            double best = 1e-6;
            int best_a = -1, best_b = -1;
            function<void(int,int,int,double)> search = [&](int x, int prev, int depth, double gain) {
                if (tree.adj[x].size() != 3 || depth >= opt.radius || elapsed() > opt.limit) return;
                int y = -1, z = -1;
                for (Edge e : tree.adj[x]) if (e.to != prev) {
                    if (y < 0) y = e.to; else z = e.to;
                }
                paint(tree, prev, x, 1, colors);
                paint(tree, y, x, 2, colors);
                paint(tree, z, x, 3, colors);
                auto scores = scorer.score(colors);
                double gain_y = gain + scores[1] - scores[0];
                double gain_z = gain + scores[2] - scores[0];
                if (gain_y > best) { best = gain_y; best_a = x; best_b = y; }
                if (gain_z > best) { best = gain_z; best_a = x; best_b = z; }
                search(y, x, depth + 1, gain_y);
                search(z, x, depth + 1, gain_z);
            };
            search(a, b, 0, 0); search(b, a, 0, 0);
            if (best_a >= 0) {
                tree.unlink(best_a, best_b);
                tree.link(u, best_a); tree.link(u, best_b); tree.link(u, sub);
                improvement += best;
                ++moves;
            } else {
                tree.unlink(a, b);
                tree.link(u, a); tree.link(u, b); tree.link(u, sub);
            }
        }
        cerr << "SPR round=" << round << " moves=" << moves << " seconds=" << elapsed() << '\n';
        if (!moves || elapsed() > opt.limit) break;
    }
    return improvement;
}

struct Bits {
    array<uint64_t,4> w{};
    bool operator==(const Bits &b) const { return w == b.w; }
    bool operator<(const Bits &b) const { return w < b.w; }
    Bits operator|(const Bits &b) const {
        Bits c; for (int i = 0; i < 4; ++i) c.w[i] = w[i] | b.w[i]; return c;
    }
    Bits operator^(const Bits &b) const {
        Bits c; for (int i = 0; i < 4; ++i) c.w[i] = w[i] ^ b.w[i]; return c;
    }
    bool subset(const Bits &b) const {
        return !(w[0]&~b.w[0]) && !(w[1]&~b.w[1]) && !(w[2]&~b.w[2]) && !(w[3]&~b.w[3]);
    }
    bool has(int x) const { return (w[x/64] >> (x%64)) & 1; }
    void set(int x) { w[x/64] |= uint64_t(1) << (x%64); }
    int size() const {
        int s = 0; for (uint64_t x : w) s += __builtin_popcountll(x); return s;
    }
    int first() const {
        for (int i = 0; i < 4; ++i) if (w[i]) return i*64 + __builtin_ctzll(w[i]);
        return -1;
    }
};
struct BitsHash {
    size_t operator()(const Bits &b) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t x : b.w) {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            h ^= x + (h << 6) + (h >> 2);
        }
        return h;
    }
};

static Tree cluster_search(const vector<Tree> &genes, const vector<Tree> &seeds,
                           const Tree &fallback, QuartetScorer &scorer, int n,
                           int anchor, const Options &opt) {
    if (n > 256) return fallback;
    Bits all, full;
    for (int i = 0; i < n; ++i) { all.set(i); if (i != anchor) full.set(i); }
    unordered_map<Bits,int,BitsHash> ids;
    auto add = [&](Bits b) {
        if (b.has(anchor)) b = all ^ b;
        if (b.size()) ids.emplace(b, -1);
    };
    auto extract = [&](const Tree &tree, bool expand) {
        int root = -1;
        for (int u = 0; u < (int)tree.taxon.size(); ++u)
            if (tree.taxon[u] == anchor) { root = u; break; }
        vector<int> parent(tree.adj.size(), -1);
        vector<Bits> below(tree.adj.size());
        function<Bits(int,int)> visit = [&](int u, int p) {
            parent[u] = p;
            Bits b;
            if (tree.taxon[u] >= 0) b.set(tree.taxon[u]);
            for (Edge e : tree.adj[u]) if (e.to != p) b = b | visit(e.to, u);
            below[u] = b;
            if (u != root) add(b);
            return b;
        };
        visit(root, -1);
        if (expand) {
            for (int u = 0; u < (int)tree.adj.size(); ++u) if (tree.adj[u].size() == 3)
                for (Edge e : tree.adj[u]) {
                    int v = e.to;
                    if (parent[v] != u || tree.adj[v].size() != 3) continue;
                    vector<Bits> a, b;
                    for (Edge f : tree.adj[u]) if (f.to != v)
                        a.push_back(parent[f.to] == u ? below[f.to] : all ^ below[u]);
                    for (Edge f : tree.adj[v]) if (f.to != u) b.push_back(below[f.to]);
                    add(a[0] | b[0]); add(a[0] | b[1]);
                }
        }
    };
    for (const Tree &tree : genes) extract(tree, false);
    for (const Tree &tree : seeds) extract(tree, true);
    add(full);
    vector<Bits> clades;
    for (auto &kv : ids) clades.push_back(kv.first);
    sort(clades.begin(), clades.end(), [](const Bits &a, const Bits &b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    });
    int k = clades.size();
    vector<int> size(k), left(k,-1), right(k,-1);
    vector<double> dp(k, -numeric_limits<double>::infinity());
    for (int i = 0; i < k; ++i) { ids[clades[i]] = i; size[i] = clades[i].size(); }
    vector<int> colors(n);
    long evaluations = 0;
    double deadline = min(opt.limit - 1.0, elapsed() + opt.limit * opt.dp_fraction);
    for (int i = 0; i < k; ++i) {
        if (size[i] == 1) { dp[i] = 0; continue; }
        if (elapsed() > deadline) {
            cerr << "cluster search deadline, retaining fallback; clades=" << k
                 << " evaluated=" << evaluations << '\n';
            return fallback;
        }
        for (int j = 0; j < i && size[j] <= size[i]/2; ++j) {
            if (!isfinite(dp[j]) || !clades[j].subset(clades[i])) continue;
            Bits other = clades[i] ^ clades[j];
            auto it = ids.find(other);
            if (it == ids.end()) continue;
            int z = it->second;
            if ((size[j] == size[z] && j > z) || !isfinite(dp[z])) continue;
            for (int t = 0; t < n; ++t)
                colors[t] = clades[j].has(t) ? 0 : other.has(t) ? 1 : 2;
            double score = dp[j]+dp[z]+scorer.trip_score(colors);
            ++evaluations;
            if (evaluations % 64 == 0 && elapsed() > deadline) return fallback;
            if (score > dp[i]) { dp[i] = score; left[i] = j; right[i] = z; }
        }
    }
    int root = ids.at(full);
    if (!isfinite(dp[root])) return fallback;
    Tree answer;
    function<int(int)> build = [&](int i) {
        if (size[i] == 1) return answer.node(clades[i].first());
        int u = answer.node();
        answer.link(u, build(left[i])); answer.link(u, build(right[i]));
        return u;
    };
    int u = build(root);
    answer.link(u, answer.node(anchor));
    cerr << setprecision(15) << "cluster clades=" << k << " resolutions=" << evaluations
         << " score=" << dp[root]/2 << " seconds=" << elapsed() << '\n';
    return answer;
}

static Tree split_consensus(const vector<Tree> &trees, const Tree &base,
                            int n, int anchor, double prior) {
    Bits all, full;
    for (int i = 0; i < n; ++i) { all.set(i); if (i != anchor) full.set(i); }
    unordered_map<Bits,double,BitsHash> support;
    auto add_tree = [&](const Tree &tree, double weight) {
        int root = -1;
        for (int u = 0; u < (int)tree.taxon.size(); ++u)
            if (tree.taxon[u] == anchor) { root = u; break; }
        function<Bits(int,int)> visit = [&](int u, int p) {
            Bits b;
            if (tree.taxon[u] >= 0) b.set(tree.taxon[u]);
            for (Edge e : tree.adj[u]) if (e.to != p) b = b | visit(e.to, u);
            if (u != root) support[b] += weight;
            return b;
        };
        visit(root, -1);
    };
    add_tree(base, prior + 1e-6);
    for (const Tree &tree : trees) add_tree(tree, 1);
    vector<Bits> clades;
    for (auto &kv : support) clades.push_back(kv.first);
    sort(clades.begin(), clades.end(), [](const Bits &a, const Bits &b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    });
    unordered_map<Bits,int,BitsHash> ids;
    int k = clades.size();
    vector<double> dp(k, -numeric_limits<double>::infinity());
    vector<int> size(k), left(k,-1), right(k,-1);
    for (int i = 0; i < k; ++i) { ids[clades[i]] = i; size[i] = clades[i].size(); }
    for (int i = 0; i < k; ++i) {
        if (size[i] == 1) { dp[i] = 0; continue; }
        for (int j = 0; j < i && size[j] <= size[i]/2; ++j) {
            if (!isfinite(dp[j]) || !clades[j].subset(clades[i])) continue;
            auto it = ids.find(clades[i]^clades[j]);
            if (it == ids.end()) continue;
            int z = it->second;
            double score = dp[j]+dp[z]+support.at(clades[i]);
            if (score > dp[i]) { dp[i] = score; left[i] = j; right[i] = z; }
        }
    }
    Tree answer;
    function<int(int)> build = [&](int i) {
        if (size[i] == 1) return answer.node(clades[i].first());
        int u = answer.node();
        answer.link(u, build(left[i])); answer.link(u, build(right[i]));
        return u;
    };
    int root = build(ids.at(full));
    answer.link(root, answer.node(anchor));
    return answer;
}

static Tree cross_validate(const vector<Tree> &genes, const Matrix &distance,
                           const Tree &base, int n, const Options &opt, unsigned seed) {
    const vector<double> grid = {0.25,0.5,1.0};
    int baseline = 0;
    for (int i = 1; i < (int)grid.size(); ++i)
        if (abs(grid[i]-opt.qexternal) < abs(grid[baseline]-opt.qexternal)) baseline = i;
    int folds = min(opt.cv, int(genes.size()));
    if (folds < 2) return base;
    vector<int> order(genes.size());
    iota(order.begin(), order.end(), 0);
    mt19937 order_rng(seed+1973);
    shuffle(order.begin(), order.end(), order_rng);
    vector<vector<double>> values(grid.size());
    double deadline = min(opt.limit*0.8, opt.limit-8);
    for (int fold = 0; fold < folds && elapsed() < deadline; ++fold) {
        vector<Tree> train, test;
        for (int i = 0; i < (int)order.size(); ++i)
            (i%folds == fold ? test : train).push_back(genes[order[i]]);
        Options criterion = opt;
        criterion.qexternal = criterion.qlength = criterion.qblend = criterion.qthreshold = 0;
        criterion.qpower = criterion.qsupport = 1;
        criterion.qmax = 0;
        criterion.qand = criterion.qproduct = false;
        QuartetScorer heldout(test, criterion);
        Tree initial = join(distances(train,n,opt), opt.method);
        if (opt.bme) improve_bme(initial,distances(train,n,opt),opt.bme);
        vector<double> scores(grid.size());
        bool finished = true;
        for (int model = 0; model < (int)grid.size(); ++model) {
            Options fit = opt;
            fit.qexternal = grid[model];
            fit.limit = deadline;
            fit.minz = 0;
            Tree candidate = initial;
            QuartetScorer training(train,fit);
            mt19937 fit_rng(seed+1009*(fold+1));
            improve_nni(candidate,training,n,fit,fit_rng);
            if (elapsed() > deadline) { finished = false; break; }
            scores[model] = tree_score(candidate,heldout,n);
        }
        if (!finished) break;
        for (int model = 0; model < (int)grid.size(); ++model)
            values[model].push_back(scores[model]);
        cerr << "validation fold=" << fold << " seconds=" << elapsed() << '\n';
    }
    int completed = values[0].size(), chosen = baseline;
    double best_gain = 0;
    if (completed >= 2) for (int model = 0; model < (int)grid.size(); ++model) {
        double sum = 0, squares = 0;
        for (int fold = 0; fold < completed; ++fold) {
            double delta = values[model][fold]-values[baseline][fold];
            sum += delta;
            squares += delta*delta;
        }
        double se = sqrt(max(0.0,(squares-sum*sum/completed)*completed/(completed-1)));
        cerr << "validation external=" << grid[model] << " gain=" << sum << " se=" << se << '\n';
        if (sum > best_gain && sum > opt.cv_z*se) { best_gain = sum; chosen = model; }
    }
    cerr << "validation chosen=" << grid[chosen] << " folds=" << completed << '\n';
    if (chosen == baseline) return base;
    Options fit = opt;
    fit.qexternal = grid[chosen];
    Tree answer = join(distance, opt.method);
    if (opt.bme) improve_bme(answer,distance,opt.bme);
    QuartetScorer scoring(genes,fit);
    mt19937 fit_rng(seed);
    improve_nni(answer,scoring,n,fit,fit_rng);
    return answer;
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) throw runtime_error("Usage: infer INPUT OUTPUT [options]");
        Options opt;
        for (int i = 3; i < argc; ++i) {
            string arg = argv[i];
            auto val = [&]() -> string {
                if (++i == argc) throw runtime_error("Missing option value");
                return argv[i];
            };
            if (arg == "--distance") opt.distance = val();
            else if (arg == "--method") opt.method = val();
            else if (arg == "--power") opt.power = stod(val());
            else if (arg == "--blend") opt.blend = stod(val());
            else if (arg == "--threshold") opt.threshold = stod(val());
            else if (arg == "--normalize") opt.normalize = stod(val());
            else if (arg == "--nni") opt.nni = stoi(val());
            else if (arg == "--qpower") opt.qpower = stod(val());
            else if (arg == "--qblend") opt.qblend = stod(val());
            else if (arg == "--qthreshold") opt.qthreshold = stod(val());
            else if (arg == "--qexternal") opt.qexternal = stod(val());
            else if (arg == "--qlength") opt.qlength = stod(val());
            else if (arg == "--qsupport") opt.qsupport = stod(val());
            else if (arg == "--qnormalize") opt.qnormalize = stod(val());
            else if (arg == "--spr") opt.spr = stoi(val());
            else if (arg == "--radius") opt.radius = stoi(val());
            else if (arg == "--score") opt.report_score = true;
            else if (arg == "--starts") opt.starts = stoi(val());
            else if (arg == "--cluster-dp") opt.cluster_dp = true;
            else if (arg == "--dp-fraction") opt.dp_fraction = stod(val());
            else if (arg == "--minz") opt.minz = stod(val());
            else if (arg == "--bootstrap") opt.bootstrap = stoi(val());
            else if (arg == "--bootstrap-fraction") opt.bootstrap_fraction = stod(val());
            else if (arg == "--bootstrap-prior") opt.bootstrap_prior = stod(val());
            else if (arg == "--qproduct") opt.qproduct = true;
            else if (arg == "--qmax") opt.qmax = stoi(val());
            else if (arg == "--bme") opt.bme = stoi(val());
            else if (arg == "--trim") opt.trim = stod(val());
            else if (arg == "--quantile") opt.quantile = stod(val());
            else if (arg == "--ensemble") opt.ensemble = stoi(val());
            else if (arg == "--legacy-missing") opt.legacy_missing = true;
            else if (arg == "--qand") opt.qand = true;
            else if (arg == "--dquality") opt.dquality = stod(val());
            else if (arg == "--zero-floor") opt.zero_floor = true;
            else if (arg == "--cv") opt.cv = stoi(val());
            else if (arg == "--cv-z") opt.cv_z = stod(val());
            else if (arg == "--qmixture") opt.qmixture = stod(val());
            else if (arg == "--limit") opt.limit = stod(val());
            else if (arg == "--initial") opt.initial = val();
            else throw runtime_error("Unknown option: " + arg);
        }
        ifstream in(argv[1]);
        if (opt.ensemble) opt.starts = max(opt.starts, opt.ensemble);
        if (!in) throw runtime_error("Cannot open input");
        Labels labels;
        vector<Tree> genes;
        string line;
        while (getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == string::npos) continue;
            Parser parser{line, 0, labels, {}};
            genes.push_back(parser.parse());
        }
        if (genes.empty() || labels.names.size() < 4) throw runtime_error("Empty input");
        int n = labels.names.size();
        for (const Tree &t : genes) {
            vector<int> count(n);
            for (int x : t.taxon) if (x >= 0) ++count.at(x);
            for (int x : count) if (x != 1) throw runtime_error("Gene leaf sets differ");
        }
        if (opt.zero_floor)
            for (Tree &t : genes) for (auto &edges : t.adj) for (Edge &e : edges)
                if (e.length <= 1e-5) e.length = 0;
        Matrix d = distances(genes, n, opt);
        Tree answer = join(d, opt.method);
        if (opt.bme) improve_bme(answer,d,opt.bme);
        if (!opt.initial.empty()) {
            ifstream initial(opt.initial);
            string text((istreambuf_iterator<char>(initial)), istreambuf_iterator<char>());
            Parser parser{text, 0, labels, {}};
            answer = parser.parse();
        }
        unsigned seed = 1;
        if (getenv("SEED")) seed = strtoul(getenv("SEED"), nullptr, 10);
        mt19937 rng(seed);
        if (opt.nni || opt.spr || opt.report_score || opt.cluster_dp || opt.bootstrap) {
            QuartetScorer scorer(genes, opt);
            double before = 0, gain = 0;
            if (opt.report_score) before = tree_score(answer, scorer, n);
            if (opt.nni) gain += improve_nni(answer, scorer, n, opt, rng);
            vector<Tree> seeds = {answer};
            vector<Tree> optimized = {answer};
            if (opt.starts > 1) {
                double best = opt.ensemble ? 0 : tree_score(answer, scorer, n);
                cerr << setprecision(15) << "start=0 score=" << best << '\n';
                for (int rep = 1; rep < opt.starts && elapsed() < opt.limit * 0.4; ++rep) {
                    Options alt = opt;
                    if (rep % 4 == 1) { alt.distance = "topo"; alt.normalize = 0; }
                    if (rep % 4 == 2) { alt.distance = "rawsupport"; alt.normalize = 0; alt.power = 1; }
                    if (rep % 4 == 3) { alt.distance = "length"; alt.normalize = 0.5; alt.power = 0.5; }
                    if (rep % 4 == 0) { alt.distance = "length"; alt.normalize = 1; alt.power = 1; }
                    string method = rep % 2 ? "bionj" : "nj";
                    if (opt.ensemble) {
                        constexpr double powers[] = {0.5,0.25,0.75,1};
                        alt.distance = "length"; alt.power = powers[rep%4];
                        alt.normalize = 1; method = "bionj";
                        if (rep/4 == 1) { alt.qexternal = 0; alt.qlength = 0; }
                        if (rep/4 == 2) { alt.qmax = 8; alt.qexternal = 0; alt.qlength = 0; }
                    }
                    Tree candidate = join(distances(genes, n, alt), method);
                    seeds.push_back(candidate);
                    if (opt.nni) {
                        if (opt.ensemble && rep/4) {
                            QuartetScorer other(genes, alt);
                            improve_nni(candidate, other, n, alt, rng);
                        } else improve_nni(candidate, scorer, n, opt, rng);
                    }
                    seeds.push_back(candidate);
                    optimized.push_back(candidate);
                    double score = opt.ensemble ? 0 : tree_score(candidate, scorer, n);
                    cerr << "start=" << rep << " score=" << score << '\n';
                    if (score > best) { best = score; answer = candidate; }
                }
                gain = best - before;
            }
            if (opt.ensemble) {
                int anchor = labels.index.count("0") ? labels.index.at("0") : 0;
                answer = split_consensus(optimized, optimized[0], n, anchor, 0);
            }
            if (opt.cluster_dp) {
                int anchor = labels.index.count("0") ? labels.index.at("0") : 0;
                answer = cluster_search(genes, seeds, answer, scorer, n, anchor, opt);
            }
            if (opt.spr) gain += improve_spr(answer, scorer, n, opt, rng);
            if (opt.spr && opt.nni) gain += improve_nni(answer, scorer, n, opt, rng);
            if (opt.report_score) {
                double after = tree_score(answer, scorer, n);
                cerr << setprecision(15) << "score before=" << before << " after=" << after
                     << " gain=" << gain << " discrepancy=" << after-before-gain << '\n';
            }
            if (opt.bootstrap && n <= 256) {
                vector<Tree> replicates;
                int g = scorer.loci;
                Options bootopt = opt;
                bootopt.minz = 0;
                bootopt.nni = max(opt.nni, 10);
                for (int rep = 0; rep < opt.bootstrap && elapsed() < opt.limit - 2; ++rep) {
                    vector<int> count(g);
                    int draws = max(1, int(g*opt.bootstrap_fraction + 0.5));
                    for (int j = 0; j < draws; ++j) ++count[rng()%g];
                    for (int j = 0; j < (int)scorer.genes.size(); ++j)
                        scorer.weights[j] = count[scorer.group[j]]*scorer.base_weights[j];
                    Tree candidate = answer;
                    improve_nni(candidate, scorer, n, bootopt, rng);
                    replicates.push_back(candidate);
                    cerr << "bootstrap=" << rep << " seconds=" << elapsed() << '\n';
                }
                int anchor = labels.index.count("0") ? labels.index.at("0") : 0;
                if (!replicates.empty()) answer = split_consensus(replicates, answer, n, anchor,
                                                                 opt.bootstrap_prior);
            }
            cerr << "quartet evaluations=" << scorer.calls << '\n';
        }
        if (opt.cv) answer = cross_validate(genes,d,answer,n,opt,seed);
        ofstream out(argv[2]);
        if (!out) throw runtime_error("Cannot open output");
        out << newick(answer, labels);
        out.close();
        if (!out) throw runtime_error("Output write failed");
        cerr << "taxa=" << n << " genes=" << genes.size() << " seconds=" << elapsed() << '\n';
        return 0;
    } catch (const exception &e) {
        cerr << "infer: " << e.what() << '\n';
        return 1;
    }
}
