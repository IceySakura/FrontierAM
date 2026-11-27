// Species-tree inference from estimated gene trees.
//
// Method (v1): summarise each gene tree by internode / support-weighted
// topological distances after contracting low-support edges, run neighbour
// joining, refine with balanced-minimum-evolution NNI, and combine several
// variants by weighted split consensus.
//
// Single-threaded, no external dependencies beyond the C++ standard library.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

static const int NW = 4;  // 4 * 64 = 256 bits for up to 250 taxa

struct Bits {
    array<uint64_t, NW> w;
    Bits() { w.fill(0); }
    bool operator==(const Bits& o) const { return w == o.w; }
};

struct BitsHash {
    size_t operator()(const Bits& b) const {
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < NW; ++i) {
            h ^= b.w[i];
            h *= 1099511628211ULL;
        }
        return (size_t)h;
    }
};

static inline void setbit(Bits& b, int i) { b.w[i >> 6] |= 1ULL << (i & 63); }
static inline bool getbit(const Bits& b, int i) { return (b.w[i >> 6] >> (i & 63)) & 1ULL; }
static inline int popcnt4(const Bits& b) {
    int c = 0;
    for (int i = 0; i < NW; ++i) c += __builtin_popcountll(b.w[i]);
    return c;
}
static inline Bits band(const Bits& a, const Bits& b) {
    Bits r;
    for (int i = 0; i < NW; ++i) r.w[i] = a.w[i] & b.w[i];
    return r;
}
static inline Bits bor(const Bits& a, const Bits& b) {
    Bits r;
    for (int i = 0; i < NW; ++i) r.w[i] = a.w[i] | b.w[i];
    return r;
}
static inline bool subset(const Bits& a, const Bits& b) {  // a subset of b
    for (int i = 0; i < NW; ++i)
        if (a.w[i] & ~b.w[i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Gene tree parsing
// ---------------------------------------------------------------------------

struct GEdge {
    int to;
    double len;
    double sup;  // negative: no support reported
};

struct GeneTree {
    int n = 0;                     // number of leaves
    vector<vector<GEdge>> adj;     // node -> neighbours
    vector<int> label;             // node -> leaf id, -1 for internal
    vector<int> leafNode;          // leaf id -> node
};

struct SubTree {
    int node;
    double len;
    double sup;
};

static bool gParentAttr = false;

struct NewickParser {
    const string& s;
    size_t pos = 0;
    GeneTree g;
    vector<int> nodes;             // node creation order

    explicit NewickParser(const string& str) : s(str) {
        g.n = 250;
        g.label.reserve(512);
        g.adj.reserve(512);
    }

    int newNode(int lab) {
        int id = (int)g.adj.size();
        g.adj.emplace_back();
        g.label.push_back(lab);
        if (lab >= 0) {
            if ((int)g.leafNode.size() <= lab) g.leafNode.resize(lab + 1, -1);
            g.leafNode[lab] = id;
        }
        return id;
    }

    void skipWs() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) ++pos;
    }

    double readNumber() {
        skipWs();
        size_t st = pos;
        while (pos < s.size() && (isdigit((unsigned char)s[pos]) || s[pos] == '.' || s[pos] == '-' ||
                                  s[pos] == '+' || s[pos] == 'e' || s[pos] == 'E'))
            ++pos;
        if (st == pos) return 0.0;
        return strtod(s.substr(st, pos - st).c_str(), nullptr);
    }

    string readLabel() {
        skipWs();
        if (pos < s.size() && s[pos] == '\'') {
            ++pos;
            size_t st = pos;
            while (pos < s.size() && s[pos] != '\'') ++pos;
            string r = s.substr(st, pos - st);
            if (pos < s.size()) ++pos;
            return r;
        }
        size_t st = pos;
        while (pos < s.size() && s[pos] != ',' && s[pos] != ')' && s[pos] != ':' && s[pos] != ';') ++pos;
        return s.substr(st, pos - st);
    }

    SubTree parse() {
        skipWs();
        if (pos < s.size() && s[pos] == '(') {
            ++pos;
            int v = newNode(-1);
            vector<SubTree> kids;
            while (true) {
                kids.push_back(parse());
                skipWs();
                if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
                if (pos < s.size() && s[pos] == ')') { ++pos; break; }
                break;  // tolerate malformed input
            }
            double sup = -1.0;
            skipWs();
            if (pos < s.size() && s[pos] != ':' && s[pos] != ',' && s[pos] != ')' && s[pos] != ';') {
                string tok = readLabel();
                char* end = nullptr;
                double val = strtod(tok.c_str(), &end);
                if (end && *end == '\0' && !tok.empty()) sup = val;
            }
            double len = 0.0;
            skipWs();
            if (pos < s.size() && s[pos] == ':') { ++pos; len = readNumber(); }
            for (const SubTree& k : kids) {
                double el = gParentAttr ? len : k.len;
                double es = gParentAttr ? sup : k.sup;
                g.adj[v].push_back({k.node, el, es});
                g.adj[k.node].push_back({v, el, es});
            }
            return {v, len, sup};
        } else {
            string lab = readLabel();
            int id = -1;
            if (!lab.empty() && all_of(lab.begin(), lab.end(), [](unsigned char c) { return isdigit(c); }))
                id = atoi(lab.c_str());
            int v = newNode(id);
            double len = 0.0;
            skipWs();
            if (pos < s.size() && s[pos] == ':') { ++pos; len = readNumber(); }
            return {v, len, -1.0};
        }
    }
};

// Collapse degree-2 internal nodes so that every internal node has degree 3.
static void suppressDegree2(GeneTree& g) {
    int m = (int)g.adj.size();
    vector<int> alive(m, 1);
    // Process iteratively: a degree-2 node's two neighbours get linked directly.
    for (int v = 0; v < m; ++v) {
        if (g.label[v] >= 0 || !alive[v]) continue;
        if (g.adj[v].size() != 2) continue;
        int a = g.adj[v][0].to, b = g.adj[v][1].to;
        double la = g.adj[v][0].len, lb = g.adj[v][1].len;
        double sa = g.adj[v][0].sup, sb = g.adj[v][1].sup;
        double sup = (sa < 0) ? sb : (sb < 0 ? sa : min(sa, sb));
        double len = la + lb;
        for (auto& e : g.adj[a]) if (e.to == v) { e.to = b; e.len = len; e.sup = sup; }
        for (auto& e : g.adj[b]) if (e.to == v) { e.to = a; e.len = len; e.sup = sup; }
        alive[v] = 0;
        g.adj[v].clear();
    }
}

// ---------------------------------------------------------------------------
// Distance matrices from gene trees
// ---------------------------------------------------------------------------

struct DistConfig {
    int mode = 0;        // 0 = hard contraction, 1 = soft support weights, 2 = all edges
    double supThr = 0.5; // hard: contract edges below this support
    double lenEps = 1e-5;
    double softPow = 1.0;
    double softBase = 0.333;  // support value treated as "no information"
    bool useSup = true;
    bool useLen = true;
    double lenScale = 0.01;   // soft length weight: min(1,(len/lenScale)^lenPow)
    double lenPow = 0.0;
    double lenFloor = 0.0;
    double patristicMix = 0.0;
    double missingSup = 0.5;
    int bmeIter = 12;
    string name;
};

static double softWeight(double sup, double len, const DistConfig& c) {
    double v = 1.0;
    if (c.useSup) {
        double p = (sup < 0) ? c.missingSup : sup;
        double x = max(0.0, min(1.0, (p - c.softBase) / (1.0 - c.softBase)));
        v *= pow(x, c.softPow);
    }
    if (c.useLen) {
        if (len < c.lenEps) v *= c.lenFloor;
        else {
            double l = (c.lenPow == 0.0) ? 1.0 : min(1.0, pow(len / c.lenScale, c.lenPow));
            v *= (c.lenFloor + (1.0 - c.lenFloor) * l);
        }
    }
    if (c.patristicMix > 0.0) {
        double pl = min(1.0, len / c.lenScale);
        v = v * (1.0 - c.patristicMix) + c.patristicMix * pl;
    }
    return v;
}

// Compute the gene-tree distance contribution for one tree.
// Returns D[i*n+j].
static void geneDistances(const GeneTree& g, const DistConfig& c, vector<double>& out,
                          const unordered_map<int, double>* fac = nullptr) {
    int n = g.n;
    int m = (int)g.adj.size();
    vector<int> uf(m);
    iota(uf.begin(), uf.end(), 0);
    function<int(int)> find = [&](int x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    if (c.mode == 0) {
        for (int v = 0; v < m; ++v) {
            for (const auto& e : g.adj[v]) {
                if (v >= e.to) continue;
                bool internalEdge = (g.label[v] < 0 && g.label[e.to] < 0);
                if (!internalEdge) continue;
                bool bad = false;
                if (c.useSup && e.sup >= 0 && e.sup < c.supThr) bad = true;
                if (c.useSup && e.sup < 0 && c.missingSup < c.supThr) bad = true;
                if (c.useLen && e.len < c.lenEps) bad = true;
                if (bad) {
                    int a = find(v), b = find(e.to);
                    if (a != b) uf[a] = b;
                }
            }
        }
    }
    // Build contracted adjacency
    vector<int> sn(m, -1);
    int ns = 0;
    for (int v = 0; v < m; ++v) {
        int r = find(v);
        if (sn[r] < 0) sn[r] = ns++;
    }
    struct CE { int to; double w; };
    vector<vector<CE>> ca(ns);
    for (int v = 0; v < m; ++v) {
        int a = sn[find(v)];
        for (const auto& e : g.adj[v]) {
            if (v >= e.to) continue;
            int b = sn[find(e.to)];
            if (a == b) continue;
            bool internalEdge = (g.label[v] < 0 && g.label[e.to] < 0);
            double w = 0.0;
            if (getenv("INFER_DEBUG_EDGES") && internalEdge && c.mode == 1 && g.leafNode[0] >= 0
                && getenv("INFER_FIRSTGENE")) {
                cerr << "EDGE\t" << e.sup << "\t" << e.len << "\t" << softWeight(e.sup, e.len, c) << "\n";
            }
            if (internalEdge) {
                if (c.mode == 0 || c.mode == 2) w = 1.0;
                else w = softWeight(e.sup, e.len, c);
                if (fac) {
                    auto it = fac->find(v * 512 + e.to);
                    if (it != fac->end()) w *= it->second;
                }
            }
            if (getenv("INFER_DEBUG_WSUM") && internalEdge && (v == 0)) {
                // never reached; placeholder
            }
            ca[a].push_back({b, w});
            ca[b].push_back({a, w});
        }
    }
    if (getenv("INFER_DEBUG_WSUM")) {
        double tot = 0.0; int cnt = 0, nz = 0;
        for (int v = 0; v < (int)ca.size(); ++v)
            for (const auto& e : ca[v]) if (v < e.to) { tot += e.w; cnt++; if (e.w > 0) nz++; }
        cerr << "WSUM\t" << tot/2 << "\t" << cnt/2 << "\t" << nz/2 << "\n";
    }
    // BFS from each leaf supernode
    vector<int> lsn(n, -1);
    for (int i = 0; i < n; ++i) lsn[i] = sn[find(g.leafNode[i])];
    vector<double> dist(ns);
    vector<int> seen(ns, 0);
    int stamp = 0;
    for (int i = 0; i < n; ++i) {
        ++stamp;
        int src = lsn[i];
        vector<int> st;
        st.push_back(src);
        seen[src] = stamp;
        dist[src] = 0.0;
        while (!st.empty()) {
            int v = st.back(); st.pop_back();
            for (const auto& e : ca[v]) {
                if (seen[e.to] == stamp) continue;
                seen[e.to] = stamp;
                dist[e.to] = dist[v] + e.w;
                st.push_back(e.to);
            }
        }
        for (int j = 0; j < n; ++j) out[(size_t)i * n + j] += dist[lsn[j]];
    }
}

// ---------------------------------------------------------------------------
// Neighbour joining
// ---------------------------------------------------------------------------

struct UTree {
    int n;
    vector<vector<int>> adj;  // node ids: 0..n-1 leaves, >= n internal
};

static UTree neighborJoining(const vector<double>& D, int n) {
    int maxNodes = 2 * n + 5;
    vector<vector<double>> d(maxNodes, vector<double>(maxNodes, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) d[i][j] = D[(size_t)i * n + j];
    UTree t;
    t.n = n;
    t.adj.assign(maxNodes, {});
    vector<int> active(n);
    iota(active.begin(), active.end(), 0);
    int nextId = n;
    while ((int)active.size() > 3) {
        int m = (int)active.size();
        vector<double> r(m, 0.0);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) r[i] += d[active[i]][active[j]];
        double best = numeric_limits<double>::infinity();
        int bi = -1, bj = -1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i; ++j) {
                double q = (m - 2) * d[active[i]][active[j]] - r[i] - r[j];
                if (q < best) { best = q; bi = i; bj = j; }
            }
        }
        int a = active[bi], b = active[bj];
        int u = nextId++;
        t.adj[u].push_back(a); t.adj[a].push_back(u);
        t.adj[u].push_back(b); t.adj[b].push_back(u);
        double dab = d[a][b];
        vector<int> keep;
        for (int k = 0; k < m; ++k) if (k != bi && k != bj) keep.push_back(active[k]);
        for (int k : keep) {
            double val = 0.5 * (d[a][k] + d[b][k] - dab);
            d[u][k] = d[k][u] = val;
        }
        d[u][u] = 0.0;
        active = keep;
        active.push_back(u);
    }
    int u = nextId++;
    for (int a : active) { t.adj[u].push_back(a); t.adj[a].push_back(u); }
    t.adj.resize(nextId);
    return t;
}

// Root an unrooted tree at leaf 0. Internal nodes get exactly two children.
struct RTree {
    int n;
    int root = -1;
    vector<array<int, 2>> ch;  // indexed by node id (internal only)
    vector<int> par;           // parent node id (-2 for unused, -1 for ingroup root)
    vector<int> nodes;         // list of internal node ids
};

static RTree rootTree(const UTree& u) {
    int n = u.n;
    int M = (int)u.adj.size();
    RTree r;
    r.n = n;
    r.ch.assign(M, {-1, -1});
    r.par.assign(M, -2);
    int leaf0 = 0;
    int root = -1;
    for (int v : u.adj[leaf0]) { root = v; break; }
    r.root = root;
    r.par[root] = -1;
    r.par[leaf0] = root;  // leaf 0 sits above the ingroup root
    vector<int> st = {root};
    vector<int> order;
    while (!st.empty()) {
        int v = st.back(); st.pop_back();
        order.push_back(v);
        int k = 0;
        for (int w : u.adj[v]) {
            if (w == r.par[v] || w == 0) continue;
            if (k < 2) r.ch[v][k] = w;
            ++k;
            r.par[w] = v;
            if (w >= n) st.push_back(w);
        }
    }
    for (int v : order) r.nodes.push_back(v);
    return r;
}

// ---------------------------------------------------------------------------
// BME (Pauplin) score and NNI refinement
// ---------------------------------------------------------------------------

static vector<double> pathLengths(const RTree& t) {
    int n = t.n;
    int M = (int)t.ch.size();
    vector<vector<int>> adj(M);
    for (int v : t.nodes) {
        for (int k = 0; k < 2; ++k) {
            int c = t.ch[v][k];
            if (c < 0) continue;
            adj[v].push_back(c);
            adj[c].push_back(v);
        }
    }
    adj[t.root].push_back(0);
    adj[0].push_back(t.root);
    vector<double> P((size_t)n * n, 0.0);
    vector<int> dist(M, -1);
    vector<int> st;
    for (int i = 0; i < n; ++i) {
        fill(dist.begin(), dist.end(), -1);
        st.clear();
        st.push_back(i);
        dist[i] = 0;
        while (!st.empty()) {
            int v = st.back(); st.pop_back();
            for (int w : adj[v]) {
                if (dist[w] < 0) { dist[w] = dist[v] + 1; st.push_back(w); }
            }
        }
        for (int j = 0; j < n; ++j) P[(size_t)i * n + j] = dist[j];
    }
    return P;
}

static double bmeScore(const vector<double>& P, const vector<double>& D, int n,
                       const vector<double>& wtable) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double* Di = &D[(size_t)i * n];
        const double* Pi = &P[(size_t)i * n];
        for (int j = i + 1; j < n; ++j) {
            int p = (int)Pi[j];
            if (p < 0) p = 0;
            if (p >= (int)wtable.size()) p = (int)wtable.size() - 1;
            s += Di[j] * wtable[p];
        }
    }
    return s;
}

static RTree nniVariant(const RTree& t, int p, int c, int which) {
    RTree r = t;
    int a = (t.ch[p][0] == c) ? t.ch[p][1] : t.ch[p][0];
    int b = t.ch[c][0], d = t.ch[c][1];
    if (which == 0) {
        r.ch[p] = {c, b}; r.ch[c] = {a, d};
    } else {
        r.ch[p] = {c, d}; r.ch[c] = {b, a};
    }
    // Update parents of moved nodes
    for (int k = 0; k < 2; ++k) {
        if (r.ch[p][k] >= 0) r.par[r.ch[p][k]] = p;
        if (r.ch[c][k] >= 0) r.par[r.ch[c][k]] = c;
    }
    return r;
}

static double gBmeAlpha = 0.5;
static RTree bmeRefine(const RTree& start, const vector<double>& D, int maxIter, int n) {
    vector<double> wtable(1024);
    wtable[0] = 1.0;
    for (int i = 1; i < 1024; ++i) wtable[i] = wtable[i-1] * gBmeAlpha;
    RTree cur = start;
    double curS = bmeScore(pathLengths(cur), D, n, wtable);
    for (int it = 0; it < maxIter; ++it) {
        double bestS = curS;
        RTree best;
        bool found = false;
        for (int c : cur.nodes) {
            int p = cur.par[c];
            if (p < 0 || p < n) continue;  // only internal edges
            for (int which = 0; which < 2; ++which) {
                RTree cand = nniVariant(cur, p, c, which);
                double s = bmeScore(pathLengths(cand), D, n, wtable);
                if (s < bestS - 1e-9) { bestS = s; best = cand; found = true; }
            }
        }
        if (!found) break;
        cur = best;
        curS = bestS;
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Splits
// ---------------------------------------------------------------------------

static vector<Bits> treeSplits(const RTree& t) {
    int n = t.n;
    int M = (int)t.ch.size();
    vector<Bits> clade(M);
    // compute depths
    vector<int> depth(M, 0);
    vector<int> st = {t.root};
    vector<int> ord;
    while (!st.empty()) {
        int v = st.back(); st.pop_back(); ord.push_back(v);
        for (int k = 0; k < 2; ++k) {
            int c = t.ch[v][k];
            if (c >= n) { depth[c] = depth[v] + 1; st.push_back(c); }
        }
    }
    sort(ord.begin(), ord.end(), [&](int a, int b) { return depth[a] > depth[b]; });
    for (int v : ord) {
        Bits b;
        for (int k = 0; k < 2; ++k) {
            int c = t.ch[v][k];
            if (c < n) setbit(b, c);
            else b = bor(b, clade[c]);
        }
        clade[v] = b;
    }
    vector<Bits> res;
    for (int v : t.nodes) {
        if (v == t.root) continue;
        res.push_back(clade[v]);
    }
    return res;
}

static bool compatible(const Bits& a, const Bits& b) {
    Bits i = band(a, b);
    if (popcnt4(i) == 0) return true;
    return subset(a, b) || subset(b, a);
}

// Build an RTree from a compatible set of splits (oriented away from leaf 0).
// If splits do not fully resolve the tree, complete it arbitrarily using the
// reference tree resolution.
static RTree treeFromSplits(const vector<Bits>& splits, int n) {
    Bits full;
    for (int i = 1; i < n; ++i) setbit(full, i);
    vector<Bits> cl;
    unordered_map<Bits, int, BitsHash> id;
    auto add = [&](const Bits& b) {
        if (id.find(b) == id.end()) { int k = (int)cl.size(); id[b] = k; cl.push_back(b); }
    };
    add(full);
    for (int i = 0; i < n; ++i) { Bits b; setbit(b, i); add(b); }
    for (const auto& s : splits) add(s);

    RTree r;
    r.n = n;
    r.ch.assign(n + 2 * n, {-1, -1});
    r.par.assign(n + 2 * n, -2);
    int nextId = n;

    auto leafId = [&](const Bits& b) {
        for (int i = 0; i < n; ++i) if (getbit(b, i)) return i;
        return 0;
    };

    function<int(const Bits&)> build = [&](const Bits& C) -> int {
        int sz = popcnt4(C);
        if (sz <= 1) return leafId(C);
        // maximal proper subclusters
        vector<Bits> kids;
        for (const Bits& b : cl) {
            if (b == C || popcnt4(b) >= sz) continue;
            if (subset(b, C)) kids.push_back(b);
        }
        vector<Bits> maxs;
        for (const Bits& a : kids) {
            bool contained = false;
            for (const Bits& b : kids)
                if (!(a == b) && subset(a, b)) { contained = true; break; }
            if (!contained) maxs.push_back(a);
        }
        if (maxs.empty()) {  // fallback: split leaves arbitrarily
            vector<int> ls;
            for (int i = 0; i < n; ++i) if (getbit(C, i)) ls.push_back(i);
            function<int(int,int)> buildList = [&](int a, int b) -> int {
                if (b - a == 1) return ls[a];
                int v = nextId++;
                int m = (a + b) / 2;
                r.ch[v] = {buildList(a, m), buildList(m, b)};
                r.par[r.ch[v][0]] = v; r.par[r.ch[v][1]] = v;
                r.nodes.push_back(v);
                return v;
            };
            return buildList(0, (int)ls.size());
        }
        // resolve polytomies by a caterpillar over maxs
        function<int(int)> buildKids = [&](int start) -> int {
            if (start == (int)maxs.size() - 1) return build(maxs[start]);
            int v = nextId++;
            int a = build(maxs[start]);
            int b = buildKids(start + 1);
            r.ch[v] = {a, b};
            r.par[a] = v; r.par[b] = v;
            r.nodes.push_back(v);
            return v;
        };
        if (maxs.size() == 1) return build(maxs[0]);
        return buildKids(0);
    };
    // set sizes for capacity safety
    int root = build(full);
    r.root = root;
    r.par[root] = -1;
    r.ch.resize(nextId);
    r.par.resize(nextId);
    return r;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static string newickFrom(const RTree& t, int v) {
    if (v < t.n) return to_string(v);
    return "(" + newickFrom(t, t.ch[v][0]) + "," + newickFrom(t, t.ch[v][1]) + ")";
}

static string treeNewick(const RTree& t) {
    return "(" + newickFrom(t, t.ch[t.root][0]) + "," + newickFrom(t, t.ch[t.root][1]) + ",0);";
}


// ---------------------------------------------------------------------------
// Sampled exact-quartet votes (ASTRAL-style quartet score)
// ---------------------------------------------------------------------------

struct QuartetVotes {
    vector<array<int, 4>> q;
    vector<array<double, 3>> v;   // accumulated gene-tree votes per topology
    int n = 250;
};

static uint64_t rngState = 88172645463325252ULL;
static inline uint64_t xorshift64() {
    uint64_t x = rngState;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rngState = x;
    return x;
}

static QuartetVotes buildQuartetVotes(const vector<GeneTree>& genes, int ngenes, int Q,
                                      int voteMode = 0) {
    QuartetVotes QV;
    QV.n = 250;
    QV.q.resize(Q);
    QV.v.assign(Q, {0.0, 0.0, 0.0});
    for (int i = 0; i < Q; ++i) {
        int a, b, c, d;
        do {
            a = (int)(xorshift64() % 250);
            b = (int)(xorshift64() % 250);
            c = (int)(xorshift64() % 250);
            d = (int)(xorshift64() % 250);
        } while (a == b || a == c || a == d || b == c || b == d || c == d);
        QV.q[i] = {a, b, c, d};
    }
    // Gene tree quartet topology from unweighted internal-path distances.
    DistConfig cfg;
    cfg.mode = 2; cfg.useSup = false; cfg.useLen = false;
    for (int gi = 0; gi < ngenes; ++gi) {
        vector<double> D((size_t)250 * 250, 0.0);
        geneDistances(genes[gi], cfg, D);
        for (int i = 0; i < Q; ++i) {
            const auto& z = QV.q[i];
            double s0 = D[z[0] * 250 + z[1]] + D[z[2] * 250 + z[3]];
            double s1 = D[z[0] * 250 + z[2]] + D[z[1] * 250 + z[3]];
            double s2 = D[z[0] * 250 + z[3]] + D[z[1] * 250 + z[2]];
            int code = 0;
            if (s1 < s0 - 1e-9) { code = 1; s0 = s1; }
            if (s2 < s0 - 1e-9) { code = 2; }
            QV.v[i][code] += 1.0;
        }
    }
    return QV;
}

static double quartetScore(const RTree& t, const QuartetVotes& QV) {
    vector<double> P = pathLengths(t);
    double s = 0.0;
    int Q = (int)QV.q.size();
    for (int i = 0; i < Q; ++i) {
        const auto& z = QV.q[i];
        double s0 = P[z[0] * 250 + z[1]] + P[z[2] * 250 + z[3]];
        double s1 = P[z[0] * 250 + z[2]] + P[z[1] * 250 + z[3]];
        double s2 = P[z[0] * 250 + z[3]] + P[z[1] * 250 + z[2]];
        int code = 0;
        if (s1 < s0 - 1e-9) { code = 1; s0 = s1; }
        if (s2 < s0 - 1e-9) { code = 2; }
        s += QV.v[i][code];
    }
    return s;
}

static RTree qscoreRefine(const RTree& start, const QuartetVotes& QV, int maxIter) {
    RTree cur = start;
    double curS = quartetScore(cur, QV);
    for (int it = 0; it < maxIter; ++it) {
        double bestS = curS;
        RTree best;
        bool found = false;
        for (int c : cur.nodes) {
            int p = cur.par[c];
            if (p < 0 || p < 250) continue;
            for (int which = 0; which < 2; ++which) {
                RTree cand = nniVariant(cur, p, c, which);
                double s = quartetScore(cand, QV);
                if (s > bestS + 1e-9) { bestS = s; best = cand; found = true; }
            }
        }
        if (!found) break;
        cur = best; curS = bestS;
    }
    return cur;
}


// ---------------------------------------------------------------------------
// Exact (all-quartets) ASTRAL-style score, with an O(#gene nodes) NNI update.
//
// For an unrooted binary tree rooted at leaf 0, every internal node has three
// incident branches (the two child clades and the complement, which contains
// leaf 0).  For a gene node u with branches (A,B,C) and a species node l with
// branches (X,Y,Z), rooting both trees at any leaf r in C &cap; Z makes u and
// l the cherry / LCA nodes of a rooted triple.  Summing over all root choices
// counts every matching quartet exactly four times (verified by brute force).
// ---------------------------------------------------------------------------

struct GeneNode {
    array<Bits, 3> b;   // the three branch leaf sets, partitioning all taxa
    double w;
};

static vector<GeneNode> buildGeneNodes(const vector<GeneTree>& genes, int mode = 0) {
    vector<GeneNode> out;
    Bits all;
    for (int i = 0; i < 250; ++i) setbit(all, i);
    auto reliability = [&](double sup, double len) {
        double p = (sup < 0) ? 0.5 : sup;
        double x = max(0.0, min(1.0, (p - 0.333) / 0.667));
        if (len < 1e-5) x = 0.0;
        return x;
    };
    for (const GeneTree& g : genes) {
        int M = (int)g.adj.size();
        int root = g.leafNode[0];
        vector<int> par(M, -2), order;
        par[root] = -1;
        order.push_back(root);
        for (size_t k = 0; k < order.size(); ++k) {
            int v = order[k];
            for (const auto& e : g.adj[v]) {
                if (par[e.to] == -2) { par[e.to] = v; order.push_back(e.to); }
            }
        }
        vector<array<int, 2>> ch(M, {-1, -1});
        for (int v : order) {
            if (par[v] < 0) continue;
            // children of v = neighbours other than its parent
            int k = 0;
            for (const auto& e : g.adj[v]) {
                if (e.to == par[v]) continue;
                if (k < 2) ch[v][k] = e.to;
                ++k;
            }
        }
        vector<Bits> clade(M);
        vector<int> sz(M, 0);
        for (int ii = (int)order.size() - 1; ii >= 0; --ii) {
            int v = order[ii];
            Bits b;
            if (g.label[v] >= 0) setbit(b, g.label[v]);
            else {
                for (int k = 0; k < 2; ++k) if (ch[v][k] >= 0) b = bor(b, clade[ch[v][k]]);
            }
            clade[v] = b;
        }
        for (int v : order) {
            if (g.label[v] >= 0 || par[v] < 0) continue;
            if (ch[v][0] < 0 || ch[v][1] < 0) continue;
            GeneNode gn;
            gn.b[0] = clade[ch[v][0]];
            gn.b[1] = clade[ch[v][1]];
            Bits up;
            for (int q = 0; q < NW; ++q) up.w[q] = all.w[q] & ~clade[v].w[q];
            gn.b[2] = up;
            double w = 1.0;
            if (mode > 0) {
                double mn = 1.0, prod = 1.0;
                int cnt = 0;
                for (const auto& e : g.adj[v]) {
                    if (g.label[e.to] >= 0) continue;  // pendant edge: reliable
                    double r = reliability(e.sup, e.len);
                    mn = min(mn, r); prod *= r; ++cnt;
                }
                if (cnt == 0) { mn = 1.0; prod = 1.0; }
                w = (mode == 1) ? mn : (mode == 2 ? prod : (mn >= 0.5 ? 1.0 : 0.0));
            }
            if (w <= 0.0) continue;
            gn.w = w;
            out.push_back(gn);
        }
    }
    return out;
}

static inline int pcount(const Bits& a, const Bits& b) {
    int c = 0;
    for (int i = 0; i < NW; ++i) c += __builtin_popcountll(a.w[i] & b.w[i]);
    return c;
}

static double nodeContribution(const array<Bits, 3>& sb, const vector<GeneNode>& gn) {
    int M[3][3];
    double total = 0.0;
    for (const GeneNode& u : gn) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M[i][j] = pcount(u.b[i], sb[j]);
        double f = 0.0;
        for (int i = 0; i < 3; ++i) {
            int a = (i + 1) % 3, b = (i + 2) % 3;
            for (int j = 0; j < 3; ++j) {
                int x = (j + 1) % 3, y = (j + 2) % 3;
                f += (double)M[i][j] * ((double)M[a][x] * M[b][x] * M[i][y] +
                                        (double)M[a][y] * M[b][y] * M[i][x]);
            }
        }
        total += u.w * f;
    }
    return total;
}

static vector<array<Bits, 3>> speciesBranches(const RTree& t) {
    int M = (int)t.ch.size();
    Bits all;
    for (int i = 0; i < t.n; ++i) setbit(all, i);
    vector<Bits> clade(M);
    vector<int> depth(M, 0);
    vector<int> st = {t.root};
    vector<int> ord;
    while (!st.empty()) {
        int v = st.back(); st.pop_back(); ord.push_back(v);
        for (int k = 0; k < 2; ++k) {
            int c = t.ch[v][k];
            if (c >= t.n) { depth[c] = depth[v] + 1; st.push_back(c); }
        }
    }
    sort(ord.begin(), ord.end(), [&](int a, int b) { return depth[a] > depth[b]; });
    for (int v : ord) {
        Bits b;
        for (int k = 0; k < 2; ++k) {
            int c = t.ch[v][k];
            if (c < t.n) setbit(b, c);
            else b = bor(b, clade[c]);
        }
        clade[v] = b;
    }
    vector<array<Bits, 3>> res(t.nodes.size());
    for (size_t k = 0; k < t.nodes.size(); ++k) {
        int v = t.nodes[k];
        int c0 = t.ch[v][0], c1 = t.ch[v][1];
        Bits b0 = (c0 < t.n) ? Bits() : clade[c0];
        Bits b1 = (c1 < t.n) ? Bits() : clade[c1];
        if (c0 < t.n) { setbit(b0, c0); }
        if (c1 < t.n) { setbit(b1, c1); }
        Bits up;
        for (int q = 0; q < NW; ++q) up.w[q] = all.w[q] & ~clade[v].w[q];
        res[k] = {b0, b1, up};
    }
    return res;
}

static double qexactScore(const RTree& t, const vector<GeneNode>& gn) {
    vector<array<Bits, 3>> sb = speciesBranches(t);
    double s = 0.0;
    for (const auto& b : sb) s += nodeContribution(b, gn);
    return s;
}

static RTree qexactRefine(const RTree& start, const vector<GeneNode>& gn, int maxIter) {
    RTree cur = start;
    vector<array<Bits, 3>> sb = speciesBranches(cur);
    vector<double> cl(sb.size());
    for (size_t k = 0; k < sb.size(); ++k) cl[k] = nodeContribution(sb[k], gn);
    // node id -> index in t.nodes
    int M = (int)cur.ch.size();
    vector<int> idx(M, -1);
    for (size_t k = 0; k < cur.nodes.size(); ++k) idx[cur.nodes[k]] = (int)k;
    for (int it = 0; it < maxIter; ++it) {
        double bestDelta = 1e-9;
        int bestC = -1, bestW = -1;
        array<Bits, 3> bestBp, bestBc;
        double bestCp = 0, bestCc = 0;
        for (int c : cur.nodes) {
            int p = cur.par[c];
            if (p < 0 || p < 250) continue;
            for (int which = 0; which < 2; ++which) {
                RTree cand = nniVariant(cur, p, c, which);
                vector<array<Bits, 3>> nb = speciesBranches(cand);
                // indices of p and c are unchanged (nodes list identical)
                int ip = idx[p], ic = idx[c];
                double cp = nodeContribution(nb[ip], gn);
                double cc = nodeContribution(nb[ic], gn);
                double delta = (cp + cc) - (cl[ip] + cl[ic]);
                if (delta > bestDelta) {
                    bestDelta = delta; bestC = c; bestW = which;
                    bestBp = nb[ip]; bestBc = nb[ic]; bestCp = cp; bestCc = cc;
                }
            }
        }
        if (bestC < 0) break;
        int p = cur.par[bestC];
        cur = nniVariant(cur, p, bestC, bestW);
        cl[idx[p]] = bestCp;
        cl[idx[bestC]] = bestCc;
        // branch sets of other nodes unchanged
    }
    return cur;
}


static RTree geneTreeToRTree(const GeneTree& g) {
    int n = g.n;
    int M = (int)g.adj.size();
    vector<int> newId(M, -1);
    int next = n;
    for (int v = 0; v < M; ++v) {
        if (g.label[v] >= 0) newId[v] = g.label[v];
        else newId[v] = next++;
    }
    RTree r;
    r.n = n;
    r.ch.assign(next, {-1, -1});
    r.par.assign(next, -2);
    int root = -1;
    for (const auto& e : g.adj[g.leafNode[0]]) { root = newId[e.to]; break; }
    r.root = root;
    r.par[root] = -1;
    vector<int> st = {root};
    vector<int> origOf(next, -1);
    for (int v = 0; v < M; ++v) origOf[newId[v]] = v;
    while (!st.empty()) {
        int v = st.back(); st.pop_back();
        r.nodes.push_back(v);
        int ov = origOf[v];
        int k = 0;
        for (const auto& e : g.adj[ov]) {
            if (newId[e.to] == r.par[v] || e.to == g.leafNode[0]) continue;
            if (k < 2) r.ch[v][k] = newId[e.to];
            ++k;
            r.par[newId[e.to]] = v;
            if (g.label[e.to] < 0) st.push_back(newId[e.to]);
        }
    }
    return r;
}


// Split (oriented away from leaf 0) of every internal edge of a gene tree.
static vector<tuple<int, int, Bits>> geneEdgeSplits(const GeneTree& g) {
    int M = (int)g.adj.size();
    int root = g.leafNode[0];
    vector<int> par(M, -2), order;
    par[root] = -1;
    order.push_back(root);
    for (size_t k = 0; k < order.size(); ++k) {
        int v = order[k];
        for (const auto& e : g.adj[v])
            if (par[e.to] == -2) { par[e.to] = v; order.push_back(e.to); }
    }
    vector<array<int, 2>> ch(M, {-1, -1});
    for (int v : order) {
        if (par[v] < 0) continue;
        int k = 0;
        for (const auto& e : g.adj[v]) {
            if (e.to == par[v]) continue;
            if (k < 2) ch[v][k] = e.to;
            ++k;
        }
    }
    vector<Bits> clade(M);
    for (int ii = (int)order.size() - 1; ii >= 0; --ii) {
        int v = order[ii];
        Bits b;
        if (g.label[v] >= 0) setbit(b, g.label[v]);
        else for (int k = 0; k < 2; ++k) if (ch[v][k] >= 0) b = bor(b, clade[ch[v][k]]);
        clade[v] = b;
    }
    vector<tuple<int, int, Bits>> res;
    for (int v : order) {
        if (g.label[v] >= 0 || par[v] < 0) continue;
        if (ch[v][0] < 0 || ch[v][1] < 0) continue;
        int p = par[v];
        if (g.label[p] >= 0) continue;   // pendant edge to leaf 0
        res.push_back({v, p, clade[v]});
    }
    return res;
}

static unordered_map<int, double> edgeFactors(const GeneTree& g,
                                              const vector<Bits>& refSplits,
                                              double alpha) {
    unordered_map<int, double> f;
    for (const auto& [v, p, sp] : geneEdgeSplits(g)) {
        bool ok = true;
        for (const Bits& r : refSplits) {
            if (!compatible(sp, r)) { ok = false; break; }
        }
        double a = ok ? 1.0 : alpha;
        f[v * 512 + p] = a;
        f[p * 512 + v] = a;
    }
    return f;
}


// UPGMA (average-linkage) tree from a distance matrix; returns split clusters.
static vector<Bits> upgmaSplits(const vector<double>& D, int n) {
    int maxNodes = 2 * n + 5;
    vector<vector<double>> d(maxNodes, vector<double>(maxNodes, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) d[i][j] = D[(size_t)i * n + j];
    vector<Bits> cl(maxNodes);
    vector<int> sz(maxNodes, 0);
    for (int i = 0; i < n; ++i) { setbit(cl[i], i); sz[i] = 1; }
    vector<int> active(n);
    iota(active.begin(), active.end(), 0);
    int nextId = n;
    vector<Bits> splits;
    while ((int)active.size() > 2) {
        double best = numeric_limits<double>::infinity();
        int bi = -1, bj = -1;
        for (int i = 0; i < (int)active.size(); ++i)
            for (int j = 0; j < i; ++j)
                if (d[active[i]][active[j]] < best) {
                    best = d[active[i]][active[j]]; bi = i; bj = j;
                }
        int a = active[bi], b = active[bj];
        int u = nextId++;
        cl[u] = bor(cl[a], cl[b]);
        sz[u] = sz[a] + sz[b];
        if (2 <= sz[u] && sz[u] <= n - 2) splits.push_back(cl[u]);
        vector<int> keep;
        for (int k = 0; k < (int)active.size(); ++k) if (k != bi && k != bj) keep.push_back(active[k]);
        for (int k : keep) {
            double v = (sz[a] * d[a][k] + sz[b] * d[b][k]) / (double)(sz[a] + sz[b]);
            d[u][k] = d[k][u] = v;
        }
        active = keep;
        active.push_back(u);
    }
    return splits;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "usage: infer INPUT OUTPUT\n";
        return 1;
    }
    bool debug = (getenv("INFER_DEBUG") != nullptr);
    ifstream in(argv[1]);
    if (!in) { cerr << "cannot open input\n"; return 1; }
    vector<string> lines;
    string line;
    while (getline(in, line)) {
        if (line.find('(') != string::npos) lines.push_back(line);
    }
    in.close();
    if (lines.empty()) { cerr << "no gene trees\n"; return 1; }

    int n = 250;
    if (const char* e = getenv("INFER_BME_ALPHA")) gBmeAlpha = atof(e);
    if (getenv("INFER_PARENT_ATTR")) gParentAttr = true;
    vector<GeneTree> genes;
    genes.reserve(lines.size());
    vector<GeneTree> genesB;
    for (const string& s : lines) {
        gParentAttr = false;
        NewickParser p(s);
        SubTree root = p.parse();
        GeneTree g = std::move(p.g);
        gParentAttr = true;
        NewickParser pb(s);
        pb.parse();
        GeneTree gb = std::move(pb.g);
        suppressDegree2(gb);
        genesB.push_back(std::move(gb));
        gParentAttr = false;
        if (debug && getenv("INFER_DEBUG_DEG") && genes.empty()) {
            map<int,int> deg0;
            for (int v = 0; v < (int)g.adj.size(); ++v) deg0[(int)g.adj[v].size()]++;
            cerr << "DEG_BEFORE\t"; for (auto& kv : deg0) cerr << kv.first << ":" << kv.second << " "; cerr << "\n";
        }
        suppressDegree2(g);
        if (debug && getenv("INFER_DEBUG_DEG") && genes.empty()) {
            map<int,int> deg1;
            for (int v = 0; v < (int)g.adj.size(); ++v) deg1[(int)g.adj[v].size()]++;
            cerr << "DEG_AFTER\t"; for (auto& kv : deg1) cerr << kv.first << ":" << kv.second << " "; cerr << "\n";
        }
        genes.push_back(std::move(g));
    }

    // Build several distance matrices.
    vector<DistConfig> configs;
    if (getenv("INFER_SWEEP3")) {
        int k = 0;
        for (double thr : {0.7, 0.85, 0.9, 0.95, 0.98, 1.0}) {
            DistConfig c; c.mode = 0; c.supThr = thr; c.name = "ht" + to_string(k++); configs.push_back(c);
        }
        for (double thr : {0.9, 0.95, 0.98}) {
            DistConfig c; c.mode = 0; c.supThr = thr; c.useLen = false; c.lenEps = 0.0; c.name = "hn" + to_string(k++); configs.push_back(c);
        }
    } else if (getenv("INFER_SWEEP2")) {
        int k = 0;
        for (double sp : {0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5}) {
            DistConfig c; c.mode = 1; c.softPow = sp; c.name = "sp" + to_string(k++); configs.push_back(c);
        }
        for (double b : {0.15, 0.25, 0.333, 0.42, 0.5}) {
            DistConfig c; c.mode = 1; c.softBase = b; c.name = "sb" + to_string(k++); configs.push_back(c);
        }
        for (double ms : {0.2, 0.35, 0.5, 0.7, 0.9}) {
            DistConfig c; c.mode = 1; c.missingSup = ms; c.name = "ms" + to_string(k++); configs.push_back(c);
        }
        for (double le : {1e-6, 3e-6, 1e-5, 3e-5, 1e-4}) {
            DistConfig c; c.mode = 1; c.lenEps = le; c.name = "le" + to_string(k++); configs.push_back(c);
        }
    } else if (getenv("INFER_SWEEP")) {
        auto addHard = [&](double thr, double eps, const char* nm) {
            DistConfig c; c.mode = 0; c.supThr = thr; c.lenEps = eps; c.name = nm; configs.push_back(c);
        };
        auto addSoft = [&](double spow, double lp, double scale, double floorv, double base, double mix, const char* nm) {
            DistConfig c; c.mode = 1; c.softPow = spow; c.lenPow = lp; c.lenScale = scale;
            c.lenFloor = floorv; c.softBase = base; c.patristicMix = mix; c.name = nm; configs.push_back(c);
        };
        addHard(0.34, 1e-5, "h34");
        addHard(0.5, 1e-5, "h50");
        addHard(0.65, 1e-5, "h65");
        addHard(0.8, 1e-5, "h80");
        addHard(0.9, 1e-5, "h90");
        addHard(0.5, 0.0, "h50nolen");
        addHard(0.8, 0.0, "h80nolen");
        addSoft(0.5, 0.0, 0.01, 0.0, 0.333, 0.0, "s05");
        addSoft(1.0, 0.0, 0.01, 0.0, 0.333, 0.0, "s10");
        addSoft(2.0, 0.0, 0.01, 0.0, 0.333, 0.0, "s20");
        addSoft(1.0, 0.0, 0.01, 0.2, 0.333, 0.0, "s10f02");
        addSoft(1.0, 0.0, 0.01, 0.5, 0.333, 0.0, "s10f05");
        addSoft(1.0, 1.0, 0.005, 0.0, 0.333, 0.0, "s10l1");
        addSoft(1.0, 1.0, 0.02, 0.0, 0.333, 0.0, "s10l2");
        addSoft(1.0, 0.5, 0.005, 0.0, 0.333, 0.0, "s10l3");
        addSoft(1.0, 0.0, 0.01, 0.0, 0.0, 0.0, "s10b0");
        addSoft(1.0, 0.0, 0.01, 0.0, 0.2, 0.0, "s10b2");
        addSoft(1.0, 0.0, 0.01, 0.0, 0.333, 0.3, "s10mix3");
        addSoft(1.0, 0.0, 0.01, 0.0, 0.333, 0.6, "s10mix6");
        addSoft(1.0, 0.0, 0.01, 1.0, 0.333, 0.0, "s10nol");
        addSoft(0.0, 0.0, 0.01, 0.0, 0.333, 0.0, "s00");
    } else {
    {
        DistConfig c; c.mode = 0; c.supThr = 0.65; c.name = "hard65"; configs.push_back(c);
    }
    {
        DistConfig c; c.mode = 1; c.softPow = 1.0; c.name = "soft1"; configs.push_back(c);
    }
    {
        DistConfig c; c.mode = 0; c.supThr = 0.80; c.name = "hard80"; configs.push_back(c);
    }
    {
        DistConfig c; c.mode = 0; c.supThr = 0.90; c.name = "hard90"; configs.push_back(c);
    }
    }

    int qIter = 0;
    if (const char* e = getenv("INFER_QSCORE")) qIter = atoi(e);
    int qeIter = 0;
    if (const char* e = getenv("INFER_QEXACT")) qeIter = atoi(e);
    int gnMode = 0;
    if (const char* e = getenv("INFER_GNMODE")) gnMode = atoi(e);
    vector<GeneNode> geneNodes;
    if (qeIter > 0) {
        geneNodes = buildGeneNodes(genes, gnMode);
        if (debug) cerr << "gene nodes " << geneNodes.size() << "\n";
    }
    if (const char* sf = getenv("INFER_SCORE_TREES")) {
        // Score arbitrary Newick trees listed as name<TAB>newick per line.
        vector<GeneNode> gn = buildGeneNodes(genes, gnMode);
        ifstream sfstream(sf);
        string sl;
        while (getline(sfstream, sl)) {
            size_t tab = sl.find('\t');
            if (tab == string::npos) continue;
            string nm = sl.substr(0, tab), nw = sl.substr(tab + 1);
            NewickParser p2(nw);
            p2.parse();
            GeneTree gt = std::move(p2.g);
            suppressDegree2(gt);
            RTree rt = geneTreeToRTree(gt);
            cerr << "SCORE\t" << nm << "\t" << qexactScore(rt, gn) << "\n";
        }
    }
    if (debug && qeIter > 0 && getenv("INFER_DEBUG_TRUTH")) {
        ifstream tf(getenv("INFER_DEBUG_TRUTH"));
        string ts((istreambuf_iterator<char>(tf)), istreambuf_iterator<char>());
        NewickParser tp(ts);
        tp.parse();
        GeneTree tg = std::move(tp.g);
        suppressDegree2(tg);
        RTree tr = geneTreeToRTree(tg);
        cerr << "TRUTH_INFO\t" << tg.adj.size() << "\t" << tr.nodes.size() << "\t" << tr.root
             << "\tleaf0=" << tg.leafNode[0] << "\n";
        cerr << "TRUTH_ADJ\t";
        for (const auto& e : tg.adj[tg.leafNode[0]]) cerr << e.to << "(deg" << tg.adj[e.to].size() << ") ";
        cerr << "\nTRUTH_ROOTCH\t" << tr.ch[tr.root][0] << "\t" << tr.ch[tr.root][1] << "\n";
        vector<array<Bits, 3>> tsb = speciesBranches(tr);
        cerr << "TRUTH_BRANCHES\t" << tsb.size() << "\t"
             << popcnt4(tsb[0][0]) << "\t" << popcnt4(tsb[0][1]) << "\t" << popcnt4(tsb[0][2]) << "\n";
        cerr << "TRUTH_QSCORE\t" << qexactScore(tr, geneNodes) << "\n";
    }
    int Qsize = 100000;
    if (const char* e = getenv("INFER_QSIZE")) Qsize = atoi(e);
    QuartetVotes QV;
    if (qIter > 0) {
        const char* seedstr = getenv("SEED");
        if (seedstr) rngState ^= (uint64_t)atol(seedstr) * 0x9e3779b97f4a7c15ULL + 1;
        QV = buildQuartetVotes(genes, (int)genes.size(), Qsize);
        if (debug) cerr << "quartet votes built Q=" << Qsize << "\n";
    }
    if (const char* ld = getenv("INFER_LOAD_D")) {
        ifstream df(ld);
        vector<double> D((size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) df >> D[(size_t)i*n+j];
        UTree u = neighborJoining(D, n);
        RTree r = rootTree(u);
        int bmeIter = 12;
        if (const char* e = getenv("INFER_BME_ITER")) bmeIter = atoi(e);
        RTree rr = bmeRefine(r, D, bmeIter, n);
        ofstream o(argv[2]); o << treeNewick(rr) << "\n"; o.close();
        return 0;
    }
    const char* single = getenv("INFER_SINGLE");
    vector<vector<double>> Ds;
    vector<RTree> trees;
    vector<vector<Bits>> splitSets;
    for (const auto& cfg : configs) {
        if (single && cfg.name != single) continue;
        vector<double> D((size_t)n * n, 0.0);
        if (getenv("INFER_NORM_ALL")) {
            for (const auto& g : genes) {
                vector<double> one((size_t)n * n, 0.0);
                geneDistances(g, cfg, one);
                double s = 0.0; int cnt = 0;
                for (int i = 0; i < n; ++i)
                    for (int j = i + 1; j < n; ++j) { s += one[(size_t)i*n+j]; ++cnt; }
                double m = s / max(cnt, 1);
                if (m > 0) for (size_t q = 0; q < D.size(); ++q) D[q] += 100.0 * one[q] / m;
            }
        } else if (const char* gqe = getenv("INFER_GENEQUALITY")) {
            double k = atof(gqe);
            vector<double> wgt;
            double sum = 0.0;
            for (const auto& g : genes) {
                double q = 0.0; int cnt = 0;
                for (int v = 0; v < (int)g.adj.size(); ++v)
                    for (const auto& e : g.adj[v]) {
                        if (v >= e.to || g.label[v] >= 0 || g.label[e.to] >= 0) continue;
                        q += softWeight(e.sup, e.len, cfg); ++cnt;
                    }
                q = cnt ? q / cnt : 0.5;
                double w = pow(max(q, 0.01), k);
                wgt.push_back(w); sum += w;
            }
            for (double& w : wgt) w = w * genes.size() / max(sum, 1e-9);
            for (size_t gi = 0; gi < genes.size(); ++gi) {
                vector<double> one((size_t)n * n, 0.0);
                geneDistances(genes[gi], cfg, one);
                for (size_t q = 0; q < D.size(); ++q) D[q] += wgt[gi] * one[q];
            }
        } else {
            for (const auto& g : genes) geneDistances(g, cfg, D);
        }
        UTree u = neighborJoining(D, n);
        RTree r = rootTree(u);
        int bmeIter = cfg.bmeIter;
        if (const char* e = getenv("INFER_BME_ITER")) bmeIter = atoi(e);
        RTree rr = bmeRefine(r, D, bmeIter, n);
        if (qIter > 0) {
            RTree rq = qscoreRefine(rr, QV, qIter);
            if (debug) cerr << cfg.name << "_q\t" << treeNewick(rq) << "\n";
            rr = rq;
        }
        if (debug) {
            cerr << cfg.name << "_qs\t" << (qIter > 0 ? quartetScore(rr, QV) : 0.0) << "\n";
        }
        if (qeIter > 0) {
            double s0 = qexactScore(rr, geneNodes);
            RTree rq = qexactRefine(rr, geneNodes, qeIter);
            double s1 = qexactScore(rq, geneNodes);
            if (debug) {
                cerr << cfg.name << "_qe\t" << treeNewick(rq) << "\n";
                cerr << cfg.name << "_qes\t" << s0 << "\t" << s1 << "\n";
            }
            rr = rq;
        }
        if (getenv("INFER_UPGMA")) {
            vector<Bits> us = upgmaSplits(D, n);
            RTree ur = treeFromSplits(us, n);
            if (debug) cerr << cfg.name << "_upgma\t" << treeNewick(ur) << "\n";
        }
        Ds.push_back(std::move(D));
        trees.push_back(rr);
        splitSets.push_back(treeSplits(rr));
        if (debug) {
            cerr << cfg.name << "\t" << treeNewick(rr) << "\n";
        }
    }

    // Species-tree-guided denoising: downweight gene-tree edges whose split
    // is incompatible with a provisional species tree, then re-estimate.
    double compatAlpha = 1.0;
    if (const char* e = getenv("INFER_COMPAT")) compatAlpha = atof(e);
    int compatIters = 1;
    if (const char* e = getenv("INFER_COMPAT_ITER")) compatIters = atoi(e);
    if (compatAlpha < 1.0) {
        size_t refIdx = 0;
        for (size_t k = 0; k < configs.size(); ++k) {
            // find the soft1 config among those actually computed
            if (single == nullptr && k == 3) refIdx = k;
        }
        if (const char* rtf = getenv("INFER_REF_TREE")) {
            ifstream rstream(rtf);
            string rts((istreambuf_iterator<char>(rstream)), istreambuf_iterator<char>());
            NewickParser rp(rts); rp.parse();
            GeneTree rgt = std::move(rp.g); suppressDegree2(rgt);
            // uses correct-attribute parse; if INFER_PARENT_ATTR is set the gene
            // trees differ, but topology is the same
            trees[refIdx] = geneTreeToRTree(rgt);
        }
        if (trees.size() > refIdx) {
            RTree ref = trees[refIdx];
            DistConfig cfg;
            cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "compat";
            for (int it = 0; it < compatIters; ++it) {
                vector<Bits> refSplits = treeSplits(ref);
                vector<double> D((size_t)n * n, 0.0);
                int nbad = 0, ntot = 0;
                for (const auto& g : genes) {
                    auto f = edgeFactors(g, refSplits, compatAlpha);
                    for (const auto& kv : f) if (kv.second < 1.0) ++nbad;
                    ntot += (int)f.size();
                    geneDistances(g, cfg, D, &f);
                }
                if (debug) cerr << "COMPAT_FACTORS\t" << nbad << "\t" << ntot << "\n";
                if (const char* dd = getenv("INFER_DUMP_D")) {
                    ofstream df(dd);
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) df << D[(size_t)i*n+j] << (j+1<n ? " " : "\n");
                    }
                }
                UTree u = neighborJoining(D, n);
                RTree r = rootTree(u);
                int bmeIter = 12;
                if (const char* e = getenv("INFER_BME_ITER")) bmeIter = atoi(e);
                RTree rr = bmeRefine(r, D, bmeIter, n);
                if (debug) cerr << "compat" << it << "\t" << treeNewick(rr) << "\n";
                ref = rr;
                trees.push_back(rr);
                splitSets.push_back(treeSplits(rr));
            }
        }
    }

    // Gene-tree-level weighting by concordance with a provisional species tree.
    double genePower = -1e9;
    if (const char* e = getenv("INFER_GENEWEIGHT")) genePower = atof(e);
    if (genePower > -1e8) {
        size_t refIdx = (single == nullptr && trees.size() > 3) ? 3 : 0;
        RTree ref = trees[refIdx];
        vector<Bits> refSplits = treeSplits(ref);
        DistConfig cfg;
        cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "genew";
        vector<double> D((size_t)n * n, 0.0);
        vector<double> one((size_t)n * n, 0.0);
        double sumw = 0.0;
        for (const auto& g : genes) {
            fill(one.begin(), one.end(), 0.0);
            geneDistances(g, cfg, one);
            int tot = 0, ok = 0;
            for (const auto& [v, p, sp] : geneEdgeSplits(g)) {
                ++tot;
                bool good = true;
                for (const Bits& r : refSplits) if (!compatible(sp, r)) { good = false; break; }
                if (good) ++ok;
            }
            double frac = (tot > 0) ? (double)ok / tot : 0.5;
            double w = pow(max(frac, 0.01), genePower);
            sumw += w;
            for (size_t q = 0; q < D.size(); ++q) D[q] += w * one[q];
        }
        if (debug) cerr << "GENEW_SUM\t" << sumw << "\n";
        UTree u = neighborJoining(D, n);
        RTree r = rootTree(u);
        int bmeIter = 12;
        if (const char* e = getenv("INFER_BME_ITER")) bmeIter = atoi(e);
        RTree rr = bmeRefine(r, D, bmeIter, n);
        if (debug) cerr << "genew\t" << treeNewick(rr) << "\n";
        trees.push_back(rr);
        splitSets.push_back(treeSplits(rr));
    }

    // Species-tree-guided denoising with a smoothed (parent-clade) reliability.
    // Several downweighting factors are used and their trees combined.
    vector<double> c2alphas = {0.25, 0.30, 0.35, 0.40};
    if (const char* e = getenv("INFER_ALPHAS")) {
        c2alphas.clear();
        stringstream ss(e);
        string item;
        while (getline(ss, item, ',')) c2alphas.push_back(atof(item.c_str()));
    }
    double c2bme = 0;
    if (const char* e = getenv("INFER_COMPAT2_BME")) c2bme = atoi(e);
    vector<RTree> compatTrees;
    bool needCompat = (getenv("INFER_C2_VOTE") != nullptr) ||
                      (getenv("INFER_BOOT") != nullptr) ||
                      (getenv("INFER_STAGE3") != nullptr) ||
                      (getenv("INFER_C2_REFTREE") != nullptr) ||
                      (getenv("INFER_C2_REFCONS") != nullptr) ||
                      (getenv("INFER_ALPHAS") != nullptr);
    size_t softIdx = 0;
    for (size_t k = 0; k < configs.size(); ++k) {
        if (!single && configs[k].name == "soft1") softIdx = k;
        else if (single && configs[k].name == single) softIdx = k;
    }
    if (softIdx >= trees.size()) softIdx = 0;
    {
        RTree ref = trees[softIdx];
        if (const char* rtf = getenv("INFER_C2_REFTREE")) {
            ifstream rstream(rtf);
            string rts((istreambuf_iterator<char>(rstream)), istreambuf_iterator<char>());
            NewickParser rp(rts); rp.parse();
            GeneTree rgt = std::move(rp.g); suppressDegree2(rgt);
            ref = geneTreeToRTree(rgt);
        } else if (getenv("INFER_C2_REFCONS")) {
            // greedy consensus of all initial trees
            unordered_map<Bits, double, BitsHash> vv;
            for (const auto& ss : splitSets) for (const Bits& b : ss) vv[b] += 1.0;
            vector<pair<double, Bits>> rr;
            for (const auto& kv : vv) rr.push_back({kv.second, kv.first});
            sort(rr.begin(), rr.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            vector<Bits> ch;
            for (const auto& kv : rr) {
                if ((int)ch.size() >= n - 3) break;
                bool ok = true;
                for (const Bits& c : ch) if (!compatible(c, kv.second)) { ok = false; break; }
                if (ok) ch.push_back(kv.second);
            }
            if ((int)ch.size() == n - 3) ref = treeFromSplits(ch, n);
        }
        vector<Bits> refSplits = treeSplits(ref);
        if (debug) {
            ofstream rf("/tmp/c2ref.tre"); rf << treeNewick(ref) << "\n";
        }
        DistConfig cfg;
        cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "compat2";
        for (size_t ai = 0; needCompat && ai < c2alphas.size(); ++ai) {
            double alpha = c2alphas[ai];
            vector<double> D((size_t)n * n, 0.0);
            for (const auto& g : genesB) {
                auto f = edgeFactors(g, refSplits, alpha);
                geneDistances(g, cfg, D, &f);
            }
            UTree u = neighborJoining(D, n);
            RTree r = rootTree(u);
            RTree rr = bmeRefine(r, D, (int)c2bme, n);
            if (debug) cerr << "compat2_" << ai << "\t" << treeNewick(rr) << "\n";
            compatTrees.push_back(rr);
        }
    }

    // Bootstrap over gene trees of the denoised distance estimate.
    int nBoot = 0;
    if (const char* e = getenv("INFER_BOOT")) nBoot = atoi(e);
    double bootTotal = 4.0;
    if (const char* e = getenv("INFER_BOOT_TOTAL")) bootTotal = atof(e);
    if (nBoot > 0) {
        RTree ref = trees[softIdx];
        vector<Bits> refSplits = treeSplits(ref);
        DistConfig cfg;
        cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "boot";
        vector<vector<float>> gD(genesB.size(), vector<float>((size_t)n * n));
        for (size_t gi = 0; gi < genesB.size(); ++gi) {
            auto f = edgeFactors(genesB[gi], refSplits, 0.3);
            vector<double> one((size_t)n * n, 0.0);
            geneDistances(genesB[gi], cfg, one, &f);
            for (size_t q = 0; q < one.size(); ++q) gD[gi][q] = (float)one[q];
        }
        vector<double> D((size_t)n * n, 0.0);
        rngState = 0x9e3779b97f4a7c15ULL;
        if (const char* sd = getenv("SEED")) rngState ^= (uint64_t)atol(sd) * 0x1234567ULL + 7;
        for (int b = 0; b < nBoot; ++b) {
            fill(D.begin(), D.end(), 0.0);
            for (int k = 0; k < (int)genesB.size(); ++k) {
                int gi = (int)(xorshift64() % genesB.size());
                for (size_t q = 0; q < D.size(); ++q) D[q] += gD[gi][q];
            }
            UTree u = neighborJoining(D, n);
            RTree r = rootTree(u);
            if (debug) cerr << "boot_" << b << "\t" << treeNewick(r) << "\n";
            compatTrees.push_back(r);
        }
        // Normalise bootstrap votes so that their total equals bootTotal.
        bootTotal /= max(1, nBoot);
    }

    // Optional third stage: re-denoise with the denoised tree as reference.
    double s3alpha = -1.0;
    if (const char* e = getenv("INFER_STAGE3")) s3alpha = atof(e);
    if (s3alpha >= 0.0 && !compatTrees.empty()) {
        vector<Bits> refSplits = treeSplits(compatTrees[0]);
        DistConfig cfg;
        cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "stage3";
        bool useParent = (getenv("INFER_STAGE3_PARENT") != nullptr);
        const vector<GeneTree>& gs = useParent ? genesB : genes;
        vector<double> D((size_t)n * n, 0.0);
        for (const auto& g : gs) {
            auto f = edgeFactors(g, refSplits, s3alpha);
            geneDistances(g, cfg, D, &f);
        }
        UTree u = neighborJoining(D, n);
        RTree r = rootTree(u);
        int bi = 12;
        if (const char* e = getenv("INFER_STAGE3_BME")) bi = atoi(e);
        RTree rr = bmeRefine(r, D, bi, n);
        if (debug) cerr << "stage3\t" << treeNewick(rr) << "\n";
        compatTrees.push_back(rr);
    }

    // Robust (trimmed / median) aggregation of per-gene distance matrices.
    if (const char* e = getenv("INFER_ROBUST")) {
        int trim = atoi(e);   // percentage trimmed from each tail
        DistConfig cfg;
        cfg.mode = 1; cfg.softPow = 1.0; cfg.lenEps = 1e-5; cfg.name = "robust";
        int G = (int)genes.size();
        vector<vector<float>> allD(G, vector<float>((size_t)n * n));
        for (int gi = 0; gi < G; ++gi) {
            vector<double> one((size_t)n * n, 0.0);
            geneDistances(genes[gi], cfg, one);
            for (size_t q = 0; q < one.size(); ++q) allD[gi][q] = (float)one[q];
        }
        vector<double> D((size_t)n * n, 0.0);
        vector<float> vals(G);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                for (int gi = 0; gi < G; ++gi) vals[gi] = allD[gi][(size_t)i*n+j];
                sort(vals.begin(), vals.end());
                double s = 0.0; int cnt = 0;
                int lo = (int)(G * trim / 200.0);
                for (int k = lo; k < G - lo; ++k) { s += vals[k]; ++cnt; }
                D[(size_t)i*n+j] = D[(size_t)j*n+i] = s / max(cnt, 1);
            }
        UTree u = neighborJoining(D, n);
        RTree r = rootTree(u);
        int bi = 12;
        if (const char* e2 = getenv("INFER_ROBUST_BME")) bi = atoi(e2);
        RTree rr = bmeRefine(r, D, bi, n);
        if (debug) cerr << "robust\t" << treeNewick(rr) << "\n";
        trees.push_back(rr);
        splitSets.push_back(treeSplits(rr));
    }

    // Average of the normalised distance matrices -> one more candidate tree.
    if (getenv("INFER_DAVG")) {
        vector<double> Dav((size_t)n * n, 0.0);
        double wsum = 0.0;
        for (const auto& D : Ds) {
            double s = 0.0;
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) s += D[(size_t)i*n+j];
            double m = s / (n * (n - 1) / 2.0);
            if (m <= 0) continue;
            double w = 1.0;
            for (size_t q = 0; q < Dav.size(); ++q) Dav[q] += w * D[q] / m;
            wsum += w;
        }
        for (double& x : Dav) x /= max(wsum, 1e-9);
        UTree u = neighborJoining(Dav, n);
        RTree r = rootTree(u);
        int bi = 12;
        if (const char* e = getenv("INFER_DAVG_BME")) bi = atoi(e);
        RTree rr = bmeRefine(r, Dav, bi, n);
        if (debug) cerr << "davg\t" << treeNewick(rr) << "\n";
        trees.push_back(rr);
        splitSets.push_back(treeSplits(rr));
    }

    // Iterated local search on the BME criterion for the best variant.
    if (const char* e = getenv("INFER_ILS")) {
        int R = atoi(e);
        int perturb = 4;
        if (const char* p2 = getenv("INFER_ILS_PERTURB")) perturb = atoi(p2);
        const vector<double>& D = Ds[softIdx];
        vector<double> wtable(1024);
        wtable[0]=1.0; for (int i=1;i<1024;++i) wtable[i]=wtable[i-1]*gBmeAlpha;
        RTree best = trees[softIdx];
        double bestS = bmeScore(pathLengths(best), D, n, wtable);
        rngState = 0x2545F4914F6CDD1DULL;
        for (int r = 0; r < R; ++r) {
            RTree t = best;
            for (int k = 0; k < perturb; ++k) {
                int c = t.nodes[(size_t)(xorshift64() % t.nodes.size())];
                int p = t.par[c];
                if (p < 0 || p < 250) continue;
                t = nniVariant(t, p, c, (int)(xorshift64() & 1));
            }
            t = bmeRefine(t, D, 12, n);
            double s = bmeScore(pathLengths(t), D, n, wtable);
            if (s < bestS - 1e-9) { bestS = s; best = t; }
            if (debug) cerr << "ils_" << r << "\t" << s << "\t" << treeNewick(t) << "\n";
        }
        if (debug) cerr << "ils_best\t" << bestS << "\t" << treeNewick(best) << "\n";
        trees.push_back(best);
        splitSets.push_back(treeSplits(best));
    }

    // Combine by split votes.  By default this is a plain majority consensus
    // over the distance-variant trees (robust across ILS levels).
    unordered_map<Bits, double, BitsHash> votes;
    unordered_map<Bits, double, BitsHash> geneFreq;
    double conflict = 0.0;
    {
        for (const auto& g : genes) {
            for (const auto& [v, p, sp] : geneEdgeSplits(g)) {
                geneFreq[sp] += 1.0;
            }
        }
        if (getenv("INFER_NO_ADAPT") == nullptr) {
            int m25 = 0;
            for (const auto& kv : geneFreq) if (kv.second >= 25.0) ++m25;
            conflict = max(0.0, min(0.6, (95.0 - m25) / 40.0));
            if (debug) cerr << "ADAPT\tm25=" << m25 << "\tconflict=" << conflict << "\n";
        }
    }
    auto configWeight = [&](const string& nm) -> double {
        if (nm == "soft1") return 3.0 - 1.5 * conflict;
        if (nm == "hard65") return 1.0;
        if (nm == "hard80") return 1.0 + conflict;
        if (nm == "hard90") return 1.0 + 2.0 * conflict;
        return 0.0;
    };
    if (const char* e = getenv("INFER_VOTE_UNIFORM")) {
        for (size_t k = 0; k < trees.size(); ++k)
            for (const Bits& s : splitSets[k]) votes[s] += 1.0;
    } else {
        // match each computed tree with its config by position
        size_t ci = 0;
        for (size_t k = 0; k < trees.size(); ++k) {
            while (ci < configs.size() && (single && configs[ci].name != single)) ++ci;
            if (ci >= configs.size()) break;
            double w = configWeight(configs[ci].name);
            ++ci;
            for (const Bits& s : splitSets[k]) votes[s] += w;
        }
    }
    // Optional: let gene-tree splits vote directly (weighted by reliability).
    double geneVoteW = 0.0;
    if (const char* e = getenv("INFER_GENE_VOTE")) geneVoteW = atof(e);
    if (geneVoteW > 0.0) {
        unordered_map<Bits, double, BitsHash> gv;
        for (const auto& g : genes) {
            for (const auto& [v, p, sp] : geneEdgeSplits(g)) {
                // reliability from the edge attribute
                double r = 1.0;
                for (const auto& e2 : g.adj[v]) if (e2.to == p) {
                    double sup = (e2.sup < 0) ? 0.5 : e2.sup;
                    r = max(0.0, min(1.0, (sup - 0.333) / 0.667));
                    if (e2.len < 1e-5) r = 0.0;
                    break;
                }
                gv[sp] += r;
            }
        }
        for (const auto& kv : gv) votes[kv.first] += geneVoteW * kv.second / (double)genes.size();
    }

    double c2Vote = 0.0;
    if (const char* e = getenv("INFER_C2_VOTE")) c2Vote = atof(e);
    if (c2Vote > 0.0) {
        for (const RTree& t : compatTrees) {
            vector<Bits> sp = treeSplits(t);
            for (const Bits& s : sp) votes[s] += c2Vote;
        }
    }
    if (const char* e = getenv("INFER_SOFT_VOTE")) {
        double w = atof(e);
        for (const Bits& s : splitSets[softIdx]) votes[s] += w;
    }
    vector<pair<double, Bits>> ranked;
    ranked.reserve(votes.size());
    for (const auto& kv : votes) ranked.push_back({kv.second, kv.first});
    bool geneTie = (getenv("INFER_NO_GENE_TIE") == nullptr);
    sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        if (geneTie) {
            auto ia = geneFreq.find(a.second), ib = geneFreq.find(b.second);
            double ga = (ia == geneFreq.end()) ? 0.0 : ia->second;
            double gb = (ib == geneFreq.end()) ? 0.0 : ib->second;
            if (ga != gb) return ga > gb;
        }
        for (int i = NW - 1; i >= 0; --i)
            if (a.second.w[i] != b.second.w[i]) return a.second.w[i] > b.second.w[i];
        return false;
    });
    vector<Bits> chosen;
    for (const auto& kv : ranked) {
        if ((int)chosen.size() >= n - 3) break;
        if (popcnt4(kv.second) < 2) continue;
        bool ok = true;
        for (const Bits& c : chosen) if (!compatible(c, kv.second)) { ok = false; break; }
        if (ok) chosen.push_back(kv.second);
    }
    // Complete the tree using the reference (first) tree if necessary.
    RTree finalTree = trees[0];
    if ((int)chosen.size() == n - 3) {
        RTree built = treeFromSplits(chosen, n);
        // Validate: must have exactly n-3 splits
        if ((int)treeSplits(built).size() == n - 3) finalTree = built;
    } else {
        // add splits from the first tree to force a complete resolution
        for (const Bits& s : splitSets[0]) {
            if ((int)chosen.size() >= n - 3) break;
            bool ok = true;
            for (const Bits& c : chosen) if (!compatible(c, s)) { ok = false; break; }
            if (ok) chosen.push_back(s);
        }
        if ((int)chosen.size() == n - 3) {
            RTree built = treeFromSplits(chosen, n);
            if ((int)treeSplits(built).size() == n - 3) finalTree = built;
        }
    }
    if (debug) cerr << "consensus\t" << treeNewick(finalTree) << "\n";

    if (getenv("INFER_SOFT_ONLY")) finalTree = trees[softIdx];
    if (const char* e = getenv("INFER_POST_BME")) {
        int iters = atoi(e);
        // refine the consensus tree on the soft1 distance matrix
        size_t di = softIdx;
        if (di < Ds.size()) {
            RTree rr = bmeRefine(finalTree, Ds[di], iters, n);
            if (debug) cerr << "postbme\t" << treeNewick(rr) << "\n";
            finalTree = rr;
        }
    }
    ofstream out(argv[2]);
    if (!out) { cerr << "cannot open output\n"; return 1; }
    out << treeNewick(finalTree) << "\n";
    out.close();
    return 0;
}
