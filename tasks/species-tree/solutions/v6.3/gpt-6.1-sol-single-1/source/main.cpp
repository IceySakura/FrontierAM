#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
static const auto started = Clock::now();
static double elapsed() {
    return chrono::duration<double>(Clock::now() - started).count();
}
static constexpr double deadline = 122;

struct RawNode {
    vector<int> child;
    string label;
    double length = 0, support = -1;
};

struct Parser {
    const string &text;
    size_t pos = 0;
    vector<RawNode> nodes;
    explicit Parser(const string &s) : text(s) {}
    void space() {
        for (;;) {
            while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos == text.size() || text[pos] != '[') break;
            int depth = 0;
            do {
                if (text[pos] == '[') ++depth;
                if (text[pos] == ']') --depth;
                ++pos;
            } while (pos < text.size() && depth);
        }
    }
    string token() {
        space();
        string result;
        if (pos < text.size() && text[pos] == '\'') {
            ++pos;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '\'') {
                    if (pos < text.size() && text[pos] == '\'') ++pos;
                    else break;
                }
                result += c;
            }
        } else {
            size_t begin = pos;
            while (pos < text.size() && !isspace(static_cast<unsigned char>(text[pos])) &&
                   string("(),:;[]").find(text[pos]) == string::npos) ++pos;
            result = text.substr(begin, pos - begin);
        }
        return result;
    }
    int subtree() {
        space();
        int v = nodes.size();
        nodes.emplace_back();
        if (pos < text.size() && text[pos] == '(') {
            ++pos;
            for (;;) {
                int c = subtree();
                nodes[v].child.push_back(c);
                space();
                if (pos == text.size()) throw runtime_error("Truncated Newick");
                if (text[pos] == ')') { ++pos; break; }
                if (text[pos++] != ',') throw runtime_error("Expected comma");
            }
            string label = token();
            if (!label.empty()) {
                char *end = nullptr;
                double value = strtod(label.c_str(), &end);
                if (*end == 0 && value >= 0 && value <= 1) nodes[v].support = value;
            }
        } else {
            nodes[v].label = token();
            if (nodes[v].label.empty()) throw runtime_error("Empty leaf label");
        }
        space();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            nodes[v].length = stod(token());
        }
        return v;
    }
    int parse() {
        int root = subtree();
        space();
        if (pos == text.size() || text[pos++] != ';') throw runtime_error("Missing semicolon");
        space();
        if (pos != text.size()) throw runtime_error("Trailing Newick text");
        return root;
    }
};

struct Edge {
    int v;
    double length, support;
};
struct Tree {
    int n;
    vector<vector<Edge>> adj;
    explicit Tree(int leaves = 0) : n(leaves), adj(leaves) {}
    int addnode() { adj.emplace_back(); return int(adj.size()) - 1; }
    void add(int a, int b, double length = 1, double support = 1) {
        adj[a].push_back({b, length, support});
        adj[b].push_back({a, length, support});
    }
    void del(int a, int b) {
        for (auto [v, other] : {pair<int,int>{a,b}, pair<int,int>{b,a}}) {
            auto &neighbors = adj[v];
            auto it = find_if(neighbors.begin(), neighbors.end(),
                              [other](const Edge &e) { return e.v == other; });
            if (it == neighbors.end()) throw runtime_error("Missing tree edge");
            neighbors.erase(it);
        }
    }
    vector<int> leaves(int v, int from) const {
        vector<int> result;
        function<void(int,int)> visit = [&](int x, int p) {
            if (x < n) result.push_back(x);
            for (auto e : adj[x]) if (e.v != p) visit(e.v, x);
        };
        visit(v, from);
        return result;
    }
    void swap_branches(int a, int x, int b, int y) {
        del(a,x); del(b,y); add(a,y); add(b,x);
    }
};

static Tree convert(const Parser &p, int root, const unordered_map<string,int> &ids) {
    Tree tree(ids.size());
    vector<int> index(p.nodes.size());
    for (size_t i = 0; i < p.nodes.size(); ++i) {
        if (p.nodes[i].child.empty()) {
            auto it = ids.find(p.nodes[i].label);
            if (it == ids.end()) throw runtime_error("Inconsistent leaf sets");
            index[i] = it->second;
        } else index[i] = tree.addnode();
    }
    for (size_t i = 0; i < p.nodes.size(); ++i)
        for (int c : p.nodes[i].child)
            tree.add(index[i], index[c], p.nodes[c].length, p.nodes[c].support);
    int r = index[root];
    if (tree.adj[r].size() == 2) {
        Edge a = tree.adj[r][0], b = tree.adj[r][1];
        tree.del(r,a.v); tree.del(r,b.v);
        tree.add(a.v,b.v,a.length+b.length,max(a.support,b.support));
    }
    for (int i = 0; i < tree.n; ++i)
        if (tree.adj[i].size() != 1) throw runtime_error("Duplicate or absent leaf");
    for (int i = tree.n; i < int(tree.adj.size()); ++i)
        if (!tree.adj[i].empty() && tree.adj[i].size() != 3)
            throw runtime_error("Nonbinary input");
    return tree;
}

static double reliability(double support, double length, int mode) {
    if (mode == 0) return 1;
    if (support < 0) return length > 1e-5 ? 1 : 0;
    double p = max(0.0, min(1.0, (3*support-1)/2));
    if (mode == 4) p *= p;
    if (mode == 6) p = sqrt(p);
    return p;
}

using Matrix = vector<vector<double>>;
static vector<Matrix> distances(const vector<Tree> &genes, bool raw) {
    int n = genes[0].n;
    vector<Matrix> result;
    for (const auto &tree : genes) {
        Matrix d(n, vector<double>(n));
        for (int s = 0; s < n; ++s) {
            function<void(int,int,double)> visit = [&](int v, int p, double depth) {
                if (v < n) d[s][v] = depth;
                for (auto e : tree.adj[v]) if (e.v != p)
                    visit(e.v, v, depth + reliability(e.support,e.length,raw ? 0 : 1));
            };
            visit(s,-1,0);
        }
        result.push_back(move(d));
    }
    return result;
}

static Matrix average(const vector<Matrix> &data, const vector<double> &weights) {
    int n = data[0].size();
    Matrix result(n,vector<double>(n));
    for (size_t g = 0; g < data.size(); ++g) if (weights[g] != 0)
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            result[i][j] += data[g][i][j]*weights[g];
    return result;
}

static Tree nj(const Matrix &input) {
    int n = input.size();
    Tree tree(n);
    Matrix d(2*n,vector<double>(2*n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) d[i][j] = input[i][j];
    vector<int> active(n);
    iota(active.begin(),active.end(),0);
    while (active.size() > 3) {
        int k = active.size();
        vector<double> sums(2*n,0);
        for (int i : active) for (int j : active) sums[i] += d[i][j];
        double best = numeric_limits<double>::infinity();
        int a = -1, b = -1;
        for (size_t x = 0; x < active.size(); ++x) for (size_t y = 0; y < x; ++y) {
            int i = active[x], j = active[y];
            double score = (k-2)*d[i][j]-sums[i]-sums[j];
            if (score < best) { best=score; a=i; b=j; }
        }
        int v = tree.addnode();
        tree.add(v,a); tree.add(v,b);
        double la = (d[a][b]+(sums[a]-sums[b])/(k-2))/2, lb = d[a][b]-la;
        for (int j : active) if (j != a && j != b)
            d[v][j] = d[j][v] = 0.5*(d[a][j]-la)+0.5*(d[b][j]-lb);
        active.erase(remove(active.begin(),active.end(),a),active.end());
        active.erase(remove(active.begin(),active.end(),b),active.end());
        active.push_back(v);
    }
    int v = tree.addnode();
    for (int a : active) tree.add(v,a);
    return tree;
}

struct Gene {
    int n, m;
    vector<int> parent, left, right, order;
    vector<double> support, length, q;
    explicit Gene(const Tree &tree) : n(tree.n), m(tree.adj.size()), parent(m,-2),
        left(m,-1), right(m,-1), support(m), length(m), q(m) {
        parent[0] = -1;
        order.push_back(0);
        for (size_t i = 0; i < order.size(); ++i) {
            int v = order[i];
            for (auto e : tree.adj[v]) if (e.v != parent[v]) {
                parent[e.v] = v;
                if (left[v] < 0) left[v] = e.v;
                else right[v] = e.v;
                support[e.v] = e.support;
                length[e.v] = e.length;
                order.push_back(e.v);
            }
        }
    }
    void setmode(int mode) {
        for (int v : order) q[v] = 1-reliability(support[v],length[v],mode);
    }
};

struct QuartetScorer {
    vector<Gene> genes;
    vector<double> geneweight;
    int n;
    double localinfo;
    struct Work {
        array<double,4> c{};
        array<double,3> down{}, up{};
    };
    vector<Work> work;
    explicit QuartetScorer(const vector<Tree> &trees, double info = 0) :
        n(trees[0].n), localinfo(info) {
        size_t m = 0;
        for (const auto &tree : trees) {
            genes.emplace_back(tree);
            m = max(m,size_t(genes.back().m));
        }
        geneweight.assign(trees.size(),1);
        work.resize(m);
    }
    void setmode(int mode) {
        for (auto &g : genes) g.setmode(mode);
    }
    array<double,3> score(const vector<int> &color) {
        array<double,3> output{};
        array<double,4> total{};
        for (int c : color) ++total[c];
        double potential = total[0]*total[1]*total[2]*total[3];
        const int px[3] = {1,1,2}, py[3] = {2,3,3};
        for (size_t gi = 0; gi < genes.size(); ++gi) {
            if (geneweight[gi] == 0) continue;
            const auto &g = genes[gi];
            array<double,3> result{};
            for (int v : g.order) {
                auto &w = work[v];
                w.c = {0,0,0,0}; w.down = {0,0,0}; w.up = {0,0,0};
                if (v < n) w.c[color[v]] = 1;
            }
            // Pair messages propagate the probability that every edge on
            // the internal quartet path is uninformative.
            for (int z = int(g.order.size())-1; z > 0; --z) {
                int v = g.order[z];
                if (g.left[v] < 0) continue;
                auto &w = work[v];
                const auto &l = work[g.left[v]], &r = work[g.right[v]];
                for (int i = 0; i < 4; ++i) w.c[i] = l.c[i]+r.c[i];
                for (int i = 0; i < 3; ++i) {
                    double cross = l.c[px[i]]*r.c[py[i]]+l.c[py[i]]*r.c[px[i]];
                    w.down[i] = g.q[v]*(l.down[i]+r.down[i]+cross);
                }
            }
            for (size_t z = 1; z < g.order.size(); ++z) {
                int v = g.order[z], p = g.parent[v];
                auto &w = work[v];
                if (p != 0) {
                    int s = g.left[p] == v ? g.right[p] : g.left[p];
                    const auto &pw = work[p], &sw = work[s];
                    for (int i = 0; i < 3; ++i) {
                        double cross = (total[px[i]]-pw.c[px[i]])*sw.c[py[i]]+
                                       (total[py[i]]-pw.c[py[i]])*sw.c[px[i]];
                        w.up[i] = g.q[v]*(pw.up[i]+sw.down[i]+cross);
                    }
                }
                if (g.left[v] < 0) continue;
                const auto &l = work[g.left[v]], &r = work[g.right[v]];
                array<double,4> u;
                for (int i = 0; i < 4; ++i) u[i] = total[i]-w.c[i];
                const array<double,4> *c[3] = {&l.c,&r.c,&u};
                const array<double,3> *f[3] = {&l.down,&r.down,&w.up};
                for (int side = 0; side < 3; ++side) {
                    const auto &a = *c[(side+1)%3], &b = *c[(side+2)%3], &d = *c[side];
                    const auto &discount = *f[side];
                    result[0] += (a[0]*b[1]+a[1]*b[0])*(d[2]*d[3]-discount[2]);
                    result[1] += (a[0]*b[2]+a[2]*b[0])*(d[1]*d[3]-discount[1]);
                    result[2] += (a[0]*b[3]+a[3]*b[0])*(d[1]*d[2]-discount[0]);
                }
            }
            double weight = geneweight[gi];
            if (localinfo != 0 && potential > 0) {
                double density = (result[0]+result[1]+result[2])/potential;
                weight *= pow(max(0.2,density),-localinfo);
            }
            for (int i = 0; i < 3; ++i) output[i] += result[i]*weight;
        }
        return output;
    }
};

static void nni(Tree &tree, QuartetScorer &scorer, int maxpasses, mt19937 &rng) {
    for (int pass = 0; pass < maxpasses && elapsed() < deadline; ++pass) {
        vector<pair<int,int>> edges;
        for (int a = tree.n; a < int(tree.adj.size()); ++a)
            for (auto e : tree.adj[a]) if (e.v > a) edges.push_back({a,e.v});
        shuffle(edges.begin(),edges.end(),rng);
        int changes = 0;
        for (auto [a,b] : edges) {
            if (elapsed() > deadline) break;
            if (find_if(tree.adj[a].begin(),tree.adj[a].end(),
                        [b](const Edge &e) { return e.v == b; }) == tree.adj[a].end()) continue;
            int x[2], y[2], k = 0;
            for (auto e : tree.adj[a]) if (e.v != b) x[k++] = e.v;
            k = 0;
            for (auto e : tree.adj[b]) if (e.v != a) y[k++] = e.v;
            vector<int> color(tree.n);
            for (int i = 0; i < 2; ++i) {
                for (int leaf : tree.leaves(x[i],a)) color[leaf] = i;
                for (int leaf : tree.leaves(y[i],b)) color[leaf] = i+2;
            }
            auto s = scorer.score(color);
            int best = max_element(s.begin(),s.end())-s.begin();
            if (best > 0 && s[best] > s[0]+1e-5) {
                tree.swap_branches(a,x[1],b,y[best == 1 ? 0 : 1]);
                ++changes;
            }
        }
        if (!changes) break;
    }
}

struct Bits {
    array<uint64_t,4> a{};
    bool operator==(const Bits &b) const { return a == b.a; }
    bool operator<(const Bits &b) const { return a < b.a; }
    int size() const {
        int result = 0;
        for (uint64_t x : a) result += __builtin_popcountll(x);
        return result;
    }
    void set(int i) { a[i/64] |= uint64_t(1) << (i%64); }
    bool contains(const Bits &b) const {
        for (int i = 0; i < 4; ++i) if ((a[i]&b.a[i]) != b.a[i]) return false;
        return true;
    }
    Bits minus(const Bits &b) const {
        Bits result;
        for (int i = 0; i < 4; ++i) result.a[i] = a[i]&~b.a[i];
        return result;
    }
    Bits &operator|=(const Bits &b) {
        for (int i = 0; i < 4; ++i) a[i] |= b.a[i];
        return *this;
    }
};
struct BitsHash {
    size_t operator()(const Bits &b) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t x : b.a) {
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31;
            h ^= x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        }
        return h;
    }
};

static vector<Bits> clusters(const Tree &tree) {
    vector<Bits> result;
    function<Bits(int,int)> visit = [&](int v, int p) {
        Bits b;
        if (v < tree.n) b.set(v);
        for (auto e : tree.adj[v]) if (e.v != p) b |= visit(e.v,v);
        if (v >= tree.n) result.push_back(b);
        return b;
    };
    visit(tree.adj[0][0].v,0);
    return result;
}

static Tree consensus(const vector<Tree> &trees) {
    int n = trees[0].n;
    if (n > 256) throw runtime_error("At most 256 leaves supported");
    struct Cluster {
        Bits b;
        int size, left = -1, right = -1;
        double frequency = 0, value = -1e100;
    };
    unordered_map<Bits,double,BitsHash> frequencies;
    for (const auto &tree : trees)
        for (const auto &b : clusters(tree)) if (b.size() < n-1) ++frequencies[b];
    for (int i = 1; i < n; ++i) { Bits b; b.set(i); frequencies.emplace(b,0); }
    Bits all;
    for (int i = 1; i < n; ++i) all.set(i);
    frequencies.emplace(all,0);
    vector<Cluster> clades;
    for (auto [b,f] : frequencies) clades.push_back({b,b.size(),-1,-1,f,-1e100});
    sort(clades.begin(),clades.end(),[](const Cluster &a, const Cluster &b) {
        return a.size != b.size ? a.size < b.size : a.b < b.b;
    });
    unordered_map<Bits,int,BitsHash> ids;
    for (int i = 0; i < int(clades.size()); ++i) ids[clades[i].b] = i;
    // Maximize the sum of recovered split frequencies while retaining a
    // compatible, fully binary tree, rather than greedily resolving ties.
    for (int i = 0; i < int(clades.size()); ++i) {
        if ((i&63) == 0 && elapsed() > 130) return trees[0];
        auto &c = clades[i];
        if (c.size == 1) { c.value=0; continue; }
        for (int j = 0; j < i && clades[j].size <= c.size/2; ++j) {
            if (!c.b.contains(clades[j].b)) continue;
            auto it = ids.find(c.b.minus(clades[j].b));
            if (it == ids.end() || it->second < j) continue;
            int k = it->second;
            double value = c.frequency+clades[j].value+clades[k].value;
            if (value > c.value+1e-10) { c.value=value; c.left=j; c.right=k; }
        }
        if (c.left < 0) throw runtime_error("Incomplete consensus search space");
    }
    Tree result(n);
    function<int(int)> build = [&](int i) {
        const auto &c = clades[i];
        if (c.size == 1)
            for (int j = 0; j < 4; ++j)
                if (c.b.a[j]) return 64*j+__builtin_ctzll(c.b.a[j]);
        int v = result.addnode(), l = build(c.left), r = build(c.right);
        result.add(v,l); result.add(v,r);
        return v;
    };
    int root = build(ids.at(all));
    result.add(root,0);
    return result;
}

static string quote(const string &s) {
    if (s.find_first_of(" \t\r\n(),:;[]'") == string::npos) return s;
    string result = "'";
    for (char c : s) { result += c; if (c == '\'') result += '\''; }
    return result+"'";
}
static string newick(const Tree &tree, const vector<string> &labels) {
    int root = tree.n;
    while (root < int(tree.adj.size()) && tree.adj[root].empty()) ++root;
    function<string(int,int)> write = [&](int v, int p) {
        if (v < tree.n) return quote(labels[v]);
        string result = "(";
        bool first = true;
        for (auto e : tree.adj[v]) if (e.v != p) {
            if (!first) result += ",";
            first = false;
            result += write(e.v,v);
        }
        return result+")";
    };
    return write(root,-1)+";\n";
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw runtime_error("Usage: solver INPUT OUTPUT");
        ifstream in(argv[1]);
        if (!in) throw runtime_error("Cannot open input");
        vector<Tree> genes;
        vector<string> labels;
        unordered_map<string,int> ids;
        string line;
        while (getline(in,line)) {
            if (line.find_first_not_of(" \t\r\n") == string::npos) continue;
            Parser parser(line);
            int root = parser.parse();
            if (genes.empty()) {
                for (const auto &v : parser.nodes) if (v.child.empty()) labels.push_back(v.label);
                sort(labels.begin(),labels.end());
                if (adjacent_find(labels.begin(),labels.end()) != labels.end())
                    throw runtime_error("Duplicate leaves");
                for (size_t i = 0; i < labels.size(); ++i) ids[labels[i]] = i;
            }
            genes.push_back(convert(parser,root,ids));
        }
        if (genes.empty() || labels.size() < 4 || labels.size() > 256)
            throw runtime_error("Invalid taxon count or empty input");
        double information = 0;
        for (const auto &tree : genes)
            for (int v = tree.n; v < int(tree.adj.size()); ++v)
                for (auto e : tree.adj[v]) if (e.v > v)
                    information += reliability(e.support,e.length,1);
        bool raw = information < 1e-8;
        auto gd = distances(genes,raw);
        QuartetScorer scorer(genes,0.5);
        scorer.setmode(raw ? 0 : 1);
        const char *seed = getenv("SEED");
        mt19937 rng(seed ? static_cast<unsigned>(strtoul(seed,nullptr,10)) : 1);
        Tree best = nj(average(gd,scorer.geneweight));
        nni(best,scorer,40,rng);

        constexpr int replicates = 60, prior = 5;
        vector<Tree> ensemble;
        for (int i = 0; i < prior; ++i) ensemble.push_back(best);
        uniform_int_distribution<int> choose(0,int(genes.size())-1);
        for (int rep = 0; rep < replicates && elapsed() < deadline-3; ++rep) {
            const int modecycle[4] = {1,4,1,6};
            scorer.setmode(raw ? 0 : modecycle[rep%4]);
            scorer.geneweight.assign(genes.size(),0);
            for (size_t i = 0; i < genes.size(); ++i) ++scorer.geneweight[choose(rng)];
            Matrix d = average(gd,scorer.geneweight);
            Tree tree = rep%3 ? best : nj(d);
            nni(tree,scorer,20,rng);
            ensemble.push_back(move(tree));
        }
        best = consensus(ensemble);
        ofstream out(argv[2]);
        if (!out) throw runtime_error("Cannot open output");
        out << newick(best,labels);
        out.close();
        if (!out) throw runtime_error("Cannot write output");
        cerr << "Inferred " << labels.size() << " taxa from " << genes.size()
             << " loci; " << ensemble.size()-prior << " replicates; " << elapsed() << "s\n";
        return 0;
    } catch (const exception &error) {
        cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
