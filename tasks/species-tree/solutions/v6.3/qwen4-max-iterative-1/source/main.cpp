#include <bits/stdc++.h>
#include <cstdint>
using namespace std;

static constexpr int MAXW = 4;

struct Bits {
    uint64_t w[MAXW]{};
    void set(int i) { w[i >> 6] |= uint64_t(1) << (i & 63); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    bool subsetOf(const Bits& o) const {
        for (int k = 0; k < MAXW; ++k) if (w[k] & ~o.w[k]) return false;
        return true;
    }
    Bits operator&(const Bits& o) const {
        Bits r; for (int k = 0; k < MAXW; ++k) r.w[k] = w[k] & o.w[k]; return r;
    }
    Bits operator|(const Bits& o) const {
        Bits r; for (int k = 0; k < MAXW; ++k) r.w[k] = w[k] | o.w[k]; return r;
    }
    Bits andnot(const Bits& o) const {
        Bits r; for (int k = 0; k < MAXW; ++k) r.w[k] = w[k] & ~o.w[k]; return r;
    }
    bool operator==(const Bits& o) const {
        for (int k = 0; k < MAXW; ++k) if (w[k] != o.w[k]) return false;
        return true;
    }
};

struct BitsHash {
    size_t operator()(const Bits& b) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int k = 0; k < MAXW; ++k) {
            uint64_t x = b.w[k] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27; x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return (size_t)h;
    }
};

struct Node {
    vector<int> adj;
    vector<double> adjW;
    vector<double> adjL;
    string label;
    int taxon = -1;
    double support = -1.0;
    double length = 0.0;
};

struct Graph {
    vector<Node> nodes;
    int addLeaf(const string& label) {
        int id = (int)nodes.size();
        nodes.push_back(Node{{}, {}, {}, label, -1, -1.0, 0.0});
        return id;
    }
    int addInternal() {
        int id = (int)nodes.size();
        nodes.push_back(Node{{}, {}, {}, "", -1, -1.0, 0.0});
        return id;
    }
    void edge(int a, int b, double w = 1.0, double len = 0.0) {
        nodes[a].adj.push_back(b);
        nodes[b].adj.push_back(a);
        nodes[a].adjW.push_back(w);
        nodes[b].adjW.push_back(w);
        nodes[a].adjL.push_back(len);
        nodes[b].adjL.push_back(len);
    }
    void removeEdge(int a, int b) {
        auto& va = nodes[a].adj;
        auto ia = find(va.begin(), va.end(), b);
        size_t pa = ia - va.begin();
        va.erase(ia);
        nodes[a].adjW.erase(nodes[a].adjW.begin() + pa);
        nodes[a].adjL.erase(nodes[a].adjL.begin() + pa);
        auto& vb = nodes[b].adj;
        auto ib = find(vb.begin(), vb.end(), a);
        size_t pb = ib - vb.begin();
        vb.erase(ib);
        nodes[b].adjW.erase(nodes[b].adjW.begin() + pb);
        nodes[b].adjL.erase(nodes[b].adjL.begin() + pb);
    }
};

struct Parser {
    const string& s;
    size_t p = 0;
    explicit Parser(const string& s) : s(s) {}
    void skip() {
        while (p < s.size()) {
            char c = s[p];
            if (isspace((unsigned char)c)) { ++p; }
            else if (c == '[') {
                int depth = 1; ++p;
                while (p < s.size() && depth) {
                    if (s[p] == '[') ++depth;
                    else if (s[p] == ']') --depth;
                    ++p;
                }
            } else break;
        }
    }
    char peek() { skip(); return p < s.size() ? s[p] : '\0'; }
    bool consume(char c) { skip(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    string label() {
        skip();
        if (p < s.size() && s[p] == '\'') {
            ++p; string out;
            while (p < s.size()) {
                if (s[p] == '\'') {
                    if (p + 1 < s.size() && s[p + 1] == '\'') { out += '\''; p += 2; }
                    else { ++p; break; }
                } else out += s[p++];
            }
            return out;
        }
        size_t start = p;
        while (p < s.size()) {
            char c = s[p];
            if (isspace((unsigned char)c) || c == '(' || c == ')' || c == '[' || c == ']' ||
                c == ',' || c == ':' || c == ';') break;
            ++p;
        }
        return s.substr(start, p - start);
    }
    void branchLength() {
        if (consume(':')) {
            skip();
            while (p < s.size()) {
                char c = s[p];
                if (isspace((unsigned char)c) || c == ',' || c == ')' || c == ';' || c == '[') break;
                ++p;
            }
        }
    }
    int parseNode(Graph& g) {
        skip();
        if (consume('(')) {
            vector<int> ch;
            ch.push_back(parseNode(g));
            while (consume(',')) ch.push_back(parseNode(g));
            if (!consume(')')) throw runtime_error("expected )");
            double support = -1.0;
            if (peek() != ':' && peek() != ',' && peek() != ')' && peek() != ';' && peek() != '\0') {
                string lab = label();
                char* end = nullptr;
                double v = strtod(lab.c_str(), &end);
                if (end && *end == '\0') support = v;
            }
            int id = g.addInternal();
            g.nodes[id].support = support;
            for (int c : ch) g.edge(id, c, support >= 0 ? support : 1.0, g.nodes[c].length);
            if (consume(':')) {
                skip(); size_t start = p;
                while (p < s.size()) {
                    char c = s[p];
                    if (isspace((unsigned char)c) || c == ',' || c == ')' || c == ';' || c == '[') break;
                    ++p;
                }
                g.nodes[id].length = strtod(s.substr(start, p - start).c_str(), nullptr);
            }
            return id;
        } else {
            string lab = label();
            if (lab.empty()) throw runtime_error("empty leaf label");
            int id = g.addLeaf(lab);
            if (consume(':')) {
                skip(); size_t start = p;
                while (p < s.size()) {
                    char c = s[p];
                    if (isspace((unsigned char)c) || c == ',' || c == ')' || c == ';' || c == '[') break;
                    ++p;
                }
                g.nodes[id].length = strtod(s.substr(start, p - start).c_str(), nullptr);
            }
            return id;
        }
    }
    Graph parse() {
        Graph g;
        int root = parseNode(g);
        if (!consume(';')) throw runtime_error("expected ;");
        if (g.nodes[root].adj.size() == 2) {
            int a = g.nodes[root].adj[0], b = g.nodes[root].adj[1];
            g.removeEdge(root, a);
            g.removeEdge(root, b);
            g.edge(a, b);
        }
        return g;
    }
};

struct Rooted {
    const Graph* g;
    int root = -1;
    vector<int> parent, post, pre, subsize, taxon;
    vector<vector<int>> children;
    vector<Bits> bits;
    vector<double> edgeW;
    vector<double> edgeL;
    void build(const Graph& gr, int outgroup) {
        g = &gr;
        int m = (int)gr.nodes.size();
        parent.assign(m, -1);
        children.assign(m, {});
        taxon.assign(m, -1);
        subsize.assign(m, 0);
        bits.assign(m, Bits{});
        edgeW.assign(m, 1.0);
        edgeL.assign(m, 0.0);
        for (int i = 0; i < m; ++i) {
            taxon[i] = gr.nodes[i].taxon;
            if (taxon[i] == outgroup && !gr.nodes[i].adj.empty()) root = i;
        }
        if (root < 0) throw runtime_error("outgroup not found");
        pre.clear(); post.clear();
        vector<int> st; st.push_back(root); parent[root] = -2;
        while (!st.empty()) {
            int v = st.back(); st.pop_back();
            pre.push_back(v);
            for (int u : gr.nodes[v].adj) if (u != parent[v]) {
                parent[u] = v;
                children[v].push_back(u);
                for (size_t k = 0; k < gr.nodes[v].adj.size(); ++k)
                    if (gr.nodes[v].adj[k] == u) { edgeW[u] = gr.nodes[v].adjW[k]; edgeL[u] = gr.nodes[v].adjL[k]; break; }
                st.push_back(u);
            }
        }
        post = pre; reverse(post.begin(), post.end());
        for (int v : post) {
            int sz = (taxon[v] >= 0 ? 1 : 0);
            Bits b;
            if (taxon[v] >= 0) b.set(taxon[v]);
            for (int u : children[v]) { sz += subsize[u]; b = b | bits[u]; }
            subsize[v] = sz; bits[v] = b;
        }
    }
};

static int C2[256];
static bool g_localW = false;
static bool g_lengthW = false;
static double g_lengthP = 1.0, g_lengthC = 0.0;
static double g_lengthSat = 0.0;
static double g_supportP = 0.0;
static double g_localP = 0.0;
static bool g_ingroupOnly = false;

static int64_t splitWeight(const Rooted& rt, const uint8_t* mem, int totalC) {
    int n = (int)rt.taxon.size(); // node count, not taxon count; use subsize root instead
    (void)n;
    int totalD = rt.subsize[rt.root] - totalC;
    int m = (int)rt.parent.size();
    vector<int> c(m), pc(m), pd(m), anc(m);
    for (int v : rt.post) {
        int cv = (rt.taxon[v] >= 0 ? mem[rt.taxon[v]] : 0);
        int sumc2 = 0, sumd2 = 0;
        for (int u : rt.children[v]) {
            cv += c[u];
            sumc2 += C2[c[u]];
            sumd2 += C2[rt.subsize[u] - c[u]];
        }
        c[v] = cv;
        int dv = rt.subsize[v] - cv;
        pc[v] = C2[cv] - sumc2;
        pd[v] = C2[dv] - sumd2;
    }
    int64_t weight = 0;
    anc[rt.root] = 0;
    for (int v : rt.pre) {
        int dv = rt.subsize[v] - c[v];
        weight += (int64_t)pd[v] * C2[totalC - c[v]];
        weight += (int64_t)pc[v] * ((int64_t)anc[v] - (int64_t)dv * (totalD - dv));
        for (int u : rt.children[v]) anc[u] = anc[v] + pd[v];
    }
    return weight;
}

static int64_t tripWeight(const Rooted& rt, const uint8_t* memB, const uint8_t* memC,
                          int totalB, int totalC, int ntax, int mode) {
    int totalD = ntax - totalB - totalC - (g_ingroupOnly ? 1 : 0);
    int m = (int)rt.parent.size();
    static vector<int> b, c, d;
    static vector<double> pd, anc, pairs, pb, pc, ancC;
    b.resize(m); c.resize(m); d.resize(m); pd.resize(m); anc.resize(m); pairs.resize(m);
    pb.resize(m); pc.resize(m); ancC.resize(m);
    double A = 0, B = 0, CA = 0, CB = 0;
    for (int v : rt.post) {
        int bv = (rt.taxon[v] >= 0 ? memB[rt.taxon[v]] : 0);
        int cv = (rt.taxon[v] >= 0 ? memC[rt.taxon[v]] : 0);
        int sumd2 = 0, sumbc = 0, sumb2 = 0, sumc2 = 0;
        for (int u : rt.children[v]) {
            bv += b[u]; cv += c[u];
            sumd2 += C2[rt.subsize[u] - b[u] - c[u]];
            sumbc += b[u] * c[u];
            sumb2 += C2[b[u]]; sumc2 += C2[c[u]];
        }
        b[v] = bv; c[v] = cv;
        d[v] = rt.subsize[v] - bv - cv;
        double lw = g_lengthSat > 0 ? rt.edgeL[v] / (rt.edgeL[v] + g_lengthSat) : pow(rt.edgeL[v] + g_lengthC, g_lengthP);
        double w = g_localP > 0 ? pow(rt.edgeW[v], g_localP) : (g_localW ? rt.edgeW[v] : (g_lengthW ? lw : 1.0));
        if (g_supportP > 0) w *= pow(rt.edgeW[v], g_supportP);
        pd[v] = (C2[d[v]] - sumd2) * w;
        if (g_ingroupOnly && v == rt.root) pd[v] = 0;
        pairs[v] = (bv * cv - sumbc) * w;
        pb[v] = C2[bv] - sumb2;
        pc[v] = C2[cv] - sumc2;
        A += pairs[v] * C2[totalD - d[v]];
        B += pd[v] * (totalB - bv) * (totalC - cv);
        CA += pb[v] * C2[totalC - cv];
        CB += pc[v] * C2[totalB - bv];
    }
    double I = 0, CI = 0;
    anc[rt.root] = 0; ancC[rt.root] = 0;
    for (int v : rt.pre) {
        I += pairs[v] * (C2[totalD] - C2[d[v]] - anc[v]);
        CI += pb[v] * (C2[totalC] - C2[c[v]] - ancC[v]);
        for (int u : rt.children[v]) { anc[u] = anc[v] + pd[v]; ancC[u] = ancC[v] + pc[v]; }
    }
    double CC = CA + CB - CI;
    double r;
    if (mode == 1) r = A;
    else if (mode == 2) r = A - I;
    else if (mode == 3) r = B;
    else if (mode == 4) r = A - I / 2;
    else if (mode == 5) r = A - I / 4;
    else if (mode == 6) r = A - 2 * I;
    else if (mode == 7) r = A - I + CC;
    else if (mode == 8) r = A + CC;
    else if (mode == 9) r = A - I / 2 + CC;
    else r = A + B - I;
    return (int64_t)llround(r);
}

struct Cluster {
    Bits b;
    int size = 0;
    int64_t weight = 0;
    int64_t score = -1;
    int left = -1, right = -1;
};

static void addCluster(unordered_set<Bits, BitsHash>& seen, vector<Cluster>& clusters, const Bits& b, int sz) {
    if (seen.insert(b).second) {
        Cluster c; c.b = b; c.size = sz; clusters.push_back(c);
    }
}

static void collectRooted(const Rooted& rt, int ntax, int outgroup,
                          unordered_set<Bits, BitsHash>& seen, vector<Cluster>& clusters) {
    for (int v = 0; v < (int)rt.parent.size(); ++v) {
        if (v == rt.root || rt.subsize[v] < 2 || rt.subsize[v] > ntax - 2) continue;
        addCluster(seen, clusters, rt.bits[v], rt.subsize[v]);
    }
    Bits full;
    for (int i = 0; i < ntax; ++i) if (i != outgroup) full.set(i);
    addCluster(seen, clusters, full, ntax - 1);
    for (int i = 0; i < ntax; ++i) {
        Bits leaf; leaf.set(i);
        addCluster(seen, clusters, leaf, 1);
    }
}

static Graph neighborJoining(const vector<vector<double>>& dist, const vector<string>& labels) {
    int n = (int)labels.size();
    Graph g;
    vector<int> nodeOf(n);
    for (int i = 0; i < n; ++i) {
        nodeOf[i] = g.addLeaf(labels[i]);
        g.nodes[nodeOf[i]].taxon = i;
    }
    vector<vector<double>> D = dist;
    vector<int> active(n);
    iota(active.begin(), active.end(), 0);
    vector<double> r(n);
    while (active.size() > 2) {
        int m = (int)active.size();
        for (int i : active) {
            r[i] = 0;
            for (int j : active) if (i != j) r[i] += D[i][j];
        }
        double best = 1e300; int bi = -1, bj = -1;
        for (int ai = 0; ai < m; ++ai) for (int aj = ai + 1; aj < m; ++aj) {
            int i = active[ai], j = active[aj];
            double q = (m - 2) * D[i][j] - r[i] - r[j];
            if (q < best) { best = q; bi = i; bj = j; }
        }
        int u = g.addInternal();
        g.edge(u, nodeOf[bi]); g.edge(u, nodeOf[bj]);
        D.resize(n + 1);
        for (auto& row : D) row.resize(n + 1, 0.0);
        r.resize(n + 1, 0.0);
        for (int k : active) if (k != bi && k != bj) {
            double nd = (D[bi][k] + D[bj][k] - D[bi][bj]) / 2.0;
            D[u][k] = D[k][u] = nd;
        }
        D[u][u] = 0;
        nodeOf.resize(n + 1);
        nodeOf[u] = u;
        active.erase(remove(active.begin(), active.end(), bi), active.end());
        active.erase(remove(active.begin(), active.end(), bj), active.end());
        active.push_back(u);
        n = (int)nodeOf.size();
    }
    if (active.size() == 2) {
        g.edge(nodeOf[active[0]], nodeOf[active[1]]);
    }
    // Suppress any degree-two serialization root if present (NJ connects two nodes directly here).
    return g;
}

static Graph upgma(const vector<vector<double>>& dist, const vector<string>& labels) {
    int n = (int)labels.size();
    Graph g;
    vector<int> nodeOf(n), sz(n, 1);
    for (int i = 0; i < n; ++i) { nodeOf[i] = g.addLeaf(labels[i]); g.nodes[nodeOf[i]].taxon = i; }
    vector<vector<double>> D = dist;
    vector<int> active(n); iota(active.begin(), active.end(), 0);
    while (active.size() > 1) {
        int m = (int)active.size();
        double best = 1e300; int bi = -1, bj = -1;
        for (int ai = 0; ai < m; ++ai) for (int aj = ai + 1; aj < m; ++aj) {
            int i = active[ai], j = active[aj];
            if (D[i][j] < best) { best = D[i][j]; bi = i; bj = j; }
        }
        int u = g.addInternal();
        g.edge(u, nodeOf[bi]); g.edge(u, nodeOf[bj]);
        int newSz = sz[bi] + sz[bj];
        D.resize(n + 1); for (auto& row : D) row.resize(n + 1, 0.0);
        nodeOf.resize(n + 1); sz.resize(n + 1);
        for (int k : active) if (k != bi && k != bj) {
            double nd = (sz[bi] * D[bi][k] + sz[bj] * D[bj][k]) / newSz;
            D[u][k] = D[k][u] = nd;
        }
        nodeOf[u] = u; sz[u] = newSz;
        active.erase(remove(active.begin(), active.end(), bi), active.end());
        active.erase(remove(active.begin(), active.end(), bj), active.end());
        active.push_back(u);
        n = (int)nodeOf.size();
    }
    int root = nodeOf[active[0]];
    if (g.nodes[root].adj.size() == 2) {
        int a = g.nodes[root].adj[0], b = g.nodes[root].adj[1];
        g.removeEdge(root, a); g.removeEdge(root, b); g.edge(a, b);
    }
    return g;
}

static Graph bionj(const vector<vector<double>>& dist, const vector<string>& labels) {
    int n = (int)labels.size();
    Graph g;
    vector<int> nodeOf(n);
    for (int i = 0; i < n; ++i) { nodeOf[i] = g.addLeaf(labels[i]); g.nodes[nodeOf[i]].taxon = i; }
    vector<vector<double>> D = dist;
    vector<int> active(n); iota(active.begin(), active.end(), 0);
    vector<double> r(n);
    while (active.size() > 2) {
        int m = (int)active.size();
        for (int i : active) { r[i] = 0; for (int j : active) if (i != j) r[i] += D[i][j]; }
        double best = 1e300; int bi = -1, bj = -1;
        for (int ai = 0; ai < m; ++ai) for (int aj = ai + 1; aj < m; ++aj) {
            int i = active[ai], j = active[aj];
            double q = (m - 2) * D[i][j] - r[i] - r[j];
            if (q < best) { best = q; bi = i; bj = j; }
        }
        double lambda = 0.5;
        if (D[bi][bj] > 1e-12) lambda = 0.5 + (r[bi] - r[bj]) / (2.0 * (m - 2) * D[bi][bj]);
        lambda = max(0.0, min(1.0, lambda));
        int u = g.addInternal();
        g.edge(u, nodeOf[bi]); g.edge(u, nodeOf[bj]);
        D.resize(n + 1); for (auto& row : D) row.resize(n + 1, 0.0);
        r.resize(n + 1, 0.0);
        for (int k : active) if (k != bi && k != bj) {
            double nd = lambda * (D[bi][k] - D[bi][bj] / 2) + (1 - lambda) * (D[bj][k] - D[bi][bj] / 2);
            D[u][k] = D[k][u] = nd;
        }
        nodeOf.resize(n + 1); nodeOf[u] = u;
        active.erase(remove(active.begin(), active.end(), bi), active.end());
        active.erase(remove(active.begin(), active.end(), bj), active.end());
        active.push_back(u);
        n = (int)nodeOf.size();
    }
    if (active.size() == 2) g.edge(nodeOf[active[0]], nodeOf[active[1]]);
    return g;
}

static string quoteLabel(const string& s) {
    bool need = false;
    for (char c : s) {
        if (isspace((unsigned char)c) || c == '(' || c == ')' || c == '[' || c == ']' ||
            c == ',' || c == ':' || c == ';' || c == '\'') { need = true; break; }
    }
    if (!need) return s;
    string out = "'";
    for (char c : s) { if (c == '\'') out += "''"; else out += c; }
    out += "'";
    return out;
}

static void writeCluster(const vector<Cluster>& clusters, const vector<string>& labels, int idx, string& out) {
    const auto& c = clusters[idx];
    if (c.size == 1) {
        for (int i = 0; i < (int)labels.size(); ++i) if (c.b.test(i)) { out += quoteLabel(labels[i]); return; }
        return;
    }
    out += '(';
    writeCluster(clusters, labels, c.left, out);
    out += ',';
    writeCluster(clusters, labels, c.right, out);
    out += ')';
}

static string guideNewick(const Graph& g, const Rooted& rt, const vector<string>& labels, int outgroup) {
    string out;
    function<void(int)> write = [&](int v) {
        if (g.nodes[v].taxon >= 0 && g.nodes[v].adj.size() <= 1) {
            out += quoteLabel(labels[g.nodes[v].taxon]); return;
        }
        vector<int> ch;
        for (int u : g.nodes[v].adj) if (u != rt.parent[v]) ch.push_back(u);
        if (ch.empty()) {
            if (g.nodes[v].taxon >= 0) out += quoteLabel(labels[g.nodes[v].taxon]);
            return;
        }
        out += '(';
        for (size_t k = 0; k < ch.size(); ++k) { if (k) out += ','; write(ch[k]); }
        out += ')';
    };
    out += '(';
    out += quoteLabel(labels[outgroup]);
    for (int u : g.nodes[rt.root].adj) { out += ','; write(u); }
    out += ");\n";
    return out;
}

static void collectTaxa(const Graph& g, int start, int avoid, vector<int>& taxa);

static Graph contractGraph(const Graph& g, double threshold, bool byLength = false) {
    int m = (int)g.nodes.size();
    vector<int> dsu(m); iota(dsu.begin(), dsu.end(), 0);
    function<int(int)> find = [&](int x) { return dsu[x] == x ? x : dsu[x] = find(dsu[x]); };
    auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) dsu[b] = a; };
    for (int u = 0; u < m; ++u) {
        if (g.nodes[u].adj.size() != 3) continue;
        for (size_t k = 0; k < g.nodes[u].adj.size(); ++k) {
            int v = g.nodes[u].adj[k];
            if (v <= u || g.nodes[v].adj.size() != 3) continue;
            double val = byLength ? g.nodes[u].adjL[k] : g.nodes[u].adjW[k];
            if (val < threshold) unite(u, v);
        }
    }
    vector<int> compId(m, -1);
    Graph out;
    for (int i = 0; i < m; ++i) {
        int r = find(i);
        if (compId[r] < 0) {
            compId[r] = (int)out.nodes.size();
            out.nodes.push_back(Node{{}, {}, {}, "", -1, -1.0, 0.0});
        }
        compId[i] = compId[r];
        if (g.nodes[i].taxon >= 0) {
            out.nodes[compId[i]].taxon = g.nodes[i].taxon;
            out.nodes[compId[i]].label = g.nodes[i].label;
        }
    }
    map<pair<int,int>, pair<double,double>> edges;
    for (int u = 0; u < m; ++u) for (size_t k = 0; k < g.nodes[u].adj.size(); ++k) {
        int v = g.nodes[u].adj[k];
        if (u >= v) continue;
        int a = compId[u], b = compId[v];
        if (a != b) {
            auto key = make_pair(min(a,b), max(a,b));
            edges[key] = {g.nodes[u].adjW[k], g.nodes[u].adjL[k]};
        }
    }
    for (auto& [ab, wl] : edges) out.edge(ab.first, ab.second, wl.first, wl.second);
    return out;
}

static vector<vector<double>> computeDistances(const vector<Graph>& genes, const vector<Rooted>& rooted,
                                               int ntax, int outgroup, int mode) {
    vector<vector<double>> d(ntax, vector<double>(ntax, 0.0));
    if (mode == 0 || mode == 1 || mode == 5) {
        for (const auto& g : genes) {
            vector<int> ln(ntax, -1);
            for (int i = 0; i < (int)g.nodes.size(); ++i)
                if (g.nodes[i].taxon >= 0 && !g.nodes[i].adj.empty()) ln[g.nodes[i].taxon] = i;
            for (int s = 0; s < ntax; ++s) {
                vector<tuple<int,int,double>> st;
                st.emplace_back(ln[s], -1, 0.0);
                while (!st.empty()) {
                    auto [v, par, dist] = st.back(); st.pop_back();
                    if (g.nodes[v].taxon >= 0 && g.nodes[v].taxon != s)
                        d[s][g.nodes[v].taxon] += dist;
                    for (size_t k = 0; k < g.nodes[v].adj.size(); ++k) {
                        int u = g.nodes[v].adj[k];
                        if (u == par) continue;
                        double add;
                        if (mode == 0) add = (g.nodes[u].adj.size() >= 3 ? 1.0 : 0.0);
                        else if (mode == 1) add = (g.nodes[u].adj.size() >= 3 && g.nodes[v].adj.size() >= 3 ? g.nodes[v].adjW[k] : 0.0);
                        else add = g.nodes[v].adjL[k];
                        st.emplace_back(u, v, dist + add);
                    }
                }
            }
        }
    } else if (mode == 4) {
        for (const auto& g : genes) {
            for (int v = 0; v < (int)g.nodes.size(); ++v) {
                int deg = (int)g.nodes[v].adj.size();
                if (deg < 3) continue;
                vector<vector<int>> comp(deg);
                vector<int> sz(deg);
                for (int k = 0; k < deg; ++k) {
                    collectTaxa(g, g.nodes[v].adj[k], v, comp[k]);
                    sz[k] = (int)comp[k].size();
                }
                for (int a = 0; a < deg; ++a) for (int b = a + 1; b < deg; ++b) {
                    int side = 0;
                    for (int c = 0; c < deg; ++c) if (c != a && c != b) side += C2[sz[c]];
                    for (int i : comp[a]) for (int j : comp[b]) { d[i][j] += side; d[j][i] += side; }
                }
            }
        }
        double total = C2[ntax - 2];
        for (int i = 0; i < ntax; ++i) for (int j = 0; j < ntax; ++j)
            d[i][j] = total - d[i][j] / (double)genes.size();
        return d;
    } else {
        for (const auto& rt : rooted) {
            vector<int> height(rt.parent.size(), 0);
            for (int v : rt.post) {
                int h = 0;
                for (int u : rt.children[v]) h += height[u];
                height[v] = h + (rt.children[v].empty() ? 0 : 1);
            }
            auto rank = [&](int v) { return mode == 2 ? rt.subsize[v] : height[v]; };
            for (int i = 0; i < ntax; ++i) if (i != outgroup) {
                d[outgroup][i] += rank(rt.root);
                d[i][outgroup] += rank(rt.root);
            }
            for (int v : rt.pre) {
                if (v == rt.root || rt.children[v].size() < 2) continue;
                double r = rank(v);
                int deg = (int)rt.children[v].size();
                vector<vector<int>> leaves(deg);
                for (int a = 0; a < deg; ++a) {
                    for (int k = 0; k < MAXW; ++k) {
                        uint64_t x = rt.bits[rt.children[v][a]].w[k];
                        while (x) { int b = __builtin_ctzll(x); leaves[a].push_back((k << 6) + b); x &= x - 1; }
                    }
                }
                for (int a = 0; a < deg; ++a) for (int b = a + 1; b < deg; ++b) {
                    for (int i : leaves[a]) for (int j : leaves[b]) {
                        d[i][j] += r; d[j][i] += r;
                    }
                }
            }
        }
    }
    for (int i = 0; i < ntax; ++i) for (int j = 0; j < ntax; ++j) d[i][j] /= (double)genes.size();
    return d;
}

static void collectTaxa(const Graph& g, int start, int avoid, vector<int>& taxa) {
    vector<pair<int,int>> st;
    st.emplace_back(start, avoid);
    while (!st.empty()) {
        auto [v, par] = st.back(); st.pop_back();
        if (g.nodes[v].taxon >= 0) taxa.push_back(g.nodes[v].taxon);
        for (int u : g.nodes[v].adj) if (u != par) st.emplace_back(u, v);
    }
}

static double refineNNI(Graph& g, const vector<vector<double>>& D, int ntax) {
    vector<vector<int>> T(ntax, vector<int>(ntax, 0));
    vector<int> leafNode(ntax, -1);
    for (int i = 0; i < (int)g.nodes.size(); ++i)
        if (g.nodes[i].taxon >= 0 && !g.nodes[i].adj.empty()) leafNode[g.nodes[i].taxon] = i;
    for (int s = 0; s < ntax; ++s) {
        vector<tuple<int,int,int>> st;
        st.emplace_back(leafNode[s], -1, 0);
        while (!st.empty()) {
            auto [v, par, dist] = st.back(); st.pop_back();
            if (g.nodes[v].taxon >= 0 && g.nodes[v].taxon != s) T[s][g.nodes[v].taxon] = dist;
            for (int u : g.nodes[v].adj) if (u != par)
                st.emplace_back(u, v, dist + (g.nodes[u].adj.size() == 3 ? 1 : 0));
        }
    }
    double sse = 0;
    for (int i = 0; i < ntax; ++i) for (int j = i + 1; j < ntax; ++j) {
        double r = D[i][j] - T[i][j]; sse += r * r;
    }
    auto crossSum = [&](const vector<int>& A, const vector<int>& B) {
        double s = 0;
        for (int i : A) for (int j : B) s += D[i][j] - T[i][j];
        return s;
    };
    auto update = [&](const vector<int>& A, const vector<int>& B, int delta) {
        for (int i : A) for (int j : B) { T[i][j] += delta; T[j][i] += delta; }
    };
    while (true) {
        double bestDelta = -1e-9;
        int bu = -1, bv = -1, ba = -1, bc = -1;
        bool swapAD = false;
        vector<int> bestA, bestB, bestC, bestD;
        for (int u = 0; u < (int)g.nodes.size(); ++u) {
            if (g.nodes[u].adj.size() != 3) continue;
            for (int v : g.nodes[u].adj) {
                if (v <= u || g.nodes[v].adj.size() != 3) continue;
                vector<int> au, av;
                for (int x : g.nodes[u].adj) if (x != v) au.push_back(x);
                for (int x : g.nodes[v].adj) if (x != u) av.push_back(x);
                if (au.size() != 2 || av.size() != 2) continue;
                vector<int> A, B, C, DD;
                collectTaxa(g, au[0], u, A);
                collectTaxa(g, au[1], u, B);
                collectTaxa(g, av[0], v, C);
                collectTaxa(g, av[1], v, DD);
                double RAB = crossSum(A, B), RCD = crossSum(C, DD);
                double RAD = crossSum(A, DD), RBC = crossSum(B, C);
                double cnt = (double)(A.size() * B.size() + C.size() * DD.size() + A.size() * DD.size() + B.size() * C.size());
                double d1 = -2 * (RAB + RCD) + 2 * (RAD + RBC) + cnt;
                double d2 = 2 * (RAB + RCD) - 2 * (RAD + RBC) + cnt;
                if (d1 < bestDelta) {
                    bestDelta = d1; bu = u; bv = v; ba = au[0]; bc = av[0]; swapAD = false;
                    bestA = A; bestB = B; bestC = C; bestD = DD;
                }
                if (d2 < bestDelta) {
                    bestDelta = d2; bu = u; bv = v; ba = au[0]; bc = av[1]; swapAD = true;
                    bestA = A; bestB = B; bestC = C; bestD = DD;
                }
            }
        }
        if (bu < 0) break;
        g.removeEdge(bu, ba);
        g.removeEdge(bv, bc);
        g.edge(bu, bc);
        g.edge(bv, ba);
        if (!swapAD) {
            update(bestA, bestB, +1); update(bestC, bestD, +1);
            update(bestA, bestD, -1); update(bestB, bestC, -1);
        } else {
            update(bestA, bestD, +1); update(bestB, bestC, +1);
            update(bestA, bestB, -1); update(bestC, bestD, -1);
        }
        sse += bestDelta;
    }
    return sse;
}

static int64_t quartetAScore(const Graph& g, const vector<Rooted>& genes, int ntax, int outgroup) {
    Rooted rt; rt.build(g, outgroup);
    int m = (int)genes[0].parent.size();
    vector<int> cL(m), cR(m);
    vector<uint8_t> mL(ntax), mR(ntax);
    int64_t score = 0;
    for (int v : rt.pre) {
        if (rt.children[v].size() != 2) continue;
        int L = rt.children[v][0], R = rt.children[v][1];
        int totalD = ntax - rt.subsize[v];
        fill(mL.begin(), mL.end(), 0); fill(mR.begin(), mR.end(), 0);
        for (int t = 0; t < ntax; ++t) {
            if (rt.bits[L].test(t)) mL[t] = 1;
            if (rt.bits[R].test(t)) mR[t] = 1;
        }
        for (const auto& gt : genes) {
            for (int u : gt.post) {
                int l = (gt.taxon[u] >= 0 ? mL[gt.taxon[u]] : 0);
                int r = (gt.taxon[u] >= 0 ? mR[gt.taxon[u]] : 0);
                for (int ch : gt.children[u]) { l += cL[ch]; r += cR[ch]; }
                cL[u] = l; cR[u] = r;
            }
            for (int u : gt.pre) {
                if (gt.children[u].size() != 2) continue;
                int a = gt.children[u][0], b = gt.children[u][1];
                int pairs = cL[a] * cR[b] + cR[a] * cL[b];
                if (!pairs) continue;
                int dout = totalD - (gt.subsize[u] - cL[u] - cR[u]);
                score += (int64_t)pairs * C2[dout];
            }
        }
    }
    return score;
}

static void astralNNI(Graph& g, const vector<Rooted>& genes, int ntax, int outgroup, int maxPass) {
    int64_t cur = quartetAScore(g, genes, ntax, outgroup);
    for (int pass = 0; pass < maxPass; ++pass) {
        int64_t best = cur;
        int bu = -1, bv = -1, ba = -1, bc = -1;
        for (int u = 0; u < (int)g.nodes.size(); ++u) {
            if (g.nodes[u].adj.size() != 3) continue;
            for (int v : g.nodes[u].adj) {
                if (v <= u || g.nodes[v].adj.size() != 3) continue;
                vector<int> au, av;
                for (int x : g.nodes[u].adj) if (x != v) au.push_back(x);
                for (int x : g.nodes[v].adj) if (x != u) av.push_back(x);
                if (au.size() != 2 || av.size() != 2) continue;
                for (int k = 0; k < 2; ++k) {
                    int a = au[0], c = av[k];
                    g.removeEdge(u, a); g.removeEdge(v, c);
                    g.edge(u, c); g.edge(v, a);
                    int64_t s = quartetAScore(g, genes, ntax, outgroup);
                    g.removeEdge(u, c); g.removeEdge(v, a);
                    g.edge(u, a); g.edge(v, c);
                    if (s > best) { best = s; bu = u; bv = v; ba = a; bc = c; }
                }
            }
        }
        if (bu < 0) break;
        g.removeEdge(bu, ba); g.removeEdge(bv, bc);
        g.edge(bu, bc); g.edge(bv, ba);
        cur = best;
        cerr << "A_NNI pass " << pass << " score " << cur << "\n";
    }
}

static int64_t tripScore(const Graph& g, const vector<Rooted>& genes, int ntax, int outgroup, int mode) {
    Rooted rt; rt.build(g, outgroup);
    vector<uint8_t> mL(ntax), mR(ntax);
    int64_t score = 0;
    for (int v : rt.pre) {
        if (rt.children[v].size() != 2) continue;
        int L = rt.children[v][0], R = rt.children[v][1];
        fill(mL.begin(), mL.end(), 0); fill(mR.begin(), mR.end(), 0);
        for (int t = 0; t < ntax; ++t) {
            if (rt.bits[L].test(t)) mL[t] = 1;
            if (rt.bits[R].test(t)) mR[t] = 1;
        }
        for (const auto& gt : genes)
            score += tripWeight(gt, mL.data(), mR.data(), rt.subsize[L], rt.subsize[R], ntax, mode);
    }
    return score;
}

static void tripNNI(Graph& g, const vector<Rooted>& genes, int ntax, int outgroup, int maxPass, int mode) {
    int64_t cur = tripScore(g, genes, ntax, outgroup, mode);
    for (int pass = 0; pass < maxPass; ++pass) {
        int64_t best = cur;
        int bu = -1, bv = -1, ba = -1, bc = -1;
        for (int u = 0; u < (int)g.nodes.size(); ++u) {
            if (g.nodes[u].adj.size() != 3) continue;
            for (int v : g.nodes[u].adj) {
                if (v <= u || g.nodes[v].adj.size() != 3) continue;
                vector<int> au, av;
                for (int x : g.nodes[u].adj) if (x != v) au.push_back(x);
                for (int x : g.nodes[v].adj) if (x != u) av.push_back(x);
                if (au.size() != 2 || av.size() != 2) continue;
                for (int k = 0; k < 2; ++k) {
                    int a = au[0], c = av[k];
                    g.removeEdge(u, a); g.removeEdge(v, c);
                    g.edge(u, c); g.edge(v, a);
                    int64_t s = tripScore(g, genes, ntax, outgroup, mode);
                    g.removeEdge(u, c); g.removeEdge(v, a);
                    g.edge(u, a); g.edge(v, c);
                    if (s > best) { best = s; bu = u; bv = v; ba = a; bc = c; }
                }
            }
        }
        if (bu < 0) break;
        g.removeEdge(bu, ba); g.removeEdge(bv, bc);
        g.edge(bu, bc); g.edge(bv, ba);
        cur = best;
        cerr << "TRIP_NNI pass " << pass << " score " << cur << "\n";
    }
}

static Graph graphFromClusters(const vector<Cluster>& clusters, const vector<string>& labels,
                               int fullIdx, int outgroup) {
    Graph g;
    function<int(int)> build = [&](int idx) -> int {
        const auto& c = clusters[idx];
        if (c.size == 1) {
            for (int t = 0; t < (int)labels.size(); ++t) if (c.b.test(t)) {
                int id = g.addLeaf(labels[t]); g.nodes[id].taxon = t; return id;
            }
        }
        int id = g.addInternal();
        int l = build(c.left), r = build(c.right);
        g.edge(id, l); g.edge(id, r);
        return id;
    };
    int root = g.addInternal();
    int og = g.addLeaf(labels[outgroup]); g.nodes[og].taxon = outgroup;
    g.edge(root, og);
    g.edge(root, build(clusters[fullIdx].left));
    g.edge(root, build(clusters[fullIdx].right));
    return g;
}

static int solveTripDP(vector<Cluster>& clusters, const vector<Rooted>& rooted,
                       int ntax, int outgroup, int tripMode, const vector<double>& geneW) {
    sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
        if (a.size != b.size) return a.size < b.size;
        for (int k = MAXW - 1; k >= 0; --k) if (a.b.w[k] != b.b.w[k]) return a.b.w[k] < b.b.w[k];
        return false;
    });
    int nc = (int)clusters.size();
    unordered_map<Bits,int,BitsHash> index;
    index.reserve(nc * 2);
    for (int i = 0; i < nc; ++i) {
        index[clusters[i].b] = i;
        clusters[i].score = -1; clusters[i].left = -1; clusters[i].right = -1;
    }
    vector<vector<int>> bySize(ntax + 1);
    for (int i = 0; i < nc; ++i) bySize[clusters[i].size].push_back(i);
    vector<uint8_t> memB(ntax), memC(ntax);
    long long dpPairs = 0;
    for (int sz = 1; sz <= ntax - 1; ++sz) {
        for (int ai : bySize[sz]) {
            Cluster& a = clusters[ai];
            if (sz == 1) { a.score = 0; continue; }
            int64_t best = -1; int bl = -1, br = -1;
            for (int bs = 1; bs <= sz / 2; ++bs) {
                for (int bi : bySize[bs]) {
                    const Cluster& b = clusters[bi];
                    if (!b.b.subsetOf(a.b)) continue;
                    Bits cb = a.b.andnot(b.b);
                    auto it = index.find(cb);
                    if (it == index.end()) continue;
                    int ci = it->second;
                    ++dpPairs;
                    if (clusters[bi].score < 0 || clusters[ci].score < 0) continue;
                    fill(memB.begin(), memB.end(), 0); fill(memC.begin(), memC.end(), 0);
                    for (int t = 0; t < ntax; ++t) {
                        if (clusters[bi].b.test(t)) memB[t] = 1;
                        if (clusters[ci].b.test(t)) memC[t] = 1;
                    }
                    int64_t w = 0;
                    for (size_t gi = 0; gi < rooted.size(); ++gi)
                        w += (int64_t)(geneW[gi] * tripWeight(rooted[gi], memB.data(), memC.data(), bs, sz - bs, ntax, tripMode));
                    int64_t val = clusters[bi].score + clusters[ci].score + w;
                    if (val > best) { best = val; bl = bi; br = ci; }
                }
            }
            a.score = best; a.left = bl; a.right = br;
        }
    }
    cerr << "dpPairs=" << dpPairs << "\n";
    Bits full; for (int i = 0; i < ntax; ++i) if (i != outgroup) full.set(i);
    return index[full];
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc != 3) {
        cerr << "usage: run.sh INPUT OUTPUT\n";
        return 2;
    }
    for (int i = 0; i <= 255; ++i) C2[i] = i * (i - 1) / 2;
    if (getenv("LOCAL_W")) g_localW = true;
    g_lengthW = !getenv("NO_LENGTHW");
    g_lengthP = getenv("LENGTH_P") ? atof(getenv("LENGTH_P")) : 0.3;
    if (getenv("LENGTH_C")) g_lengthC = atof(getenv("LENGTH_C"));
    if (getenv("LENGTH_SAT")) g_lengthSat = atof(getenv("LENGTH_SAT"));
    g_supportP = getenv("SUPPORT_P") ? atof(getenv("SUPPORT_P")) : 0.5;
    g_localP = getenv("LOCAL_P") ? atof(getenv("LOCAL_P")) : 0.0;
    g_ingroupOnly = !getenv("NO_INGROUP");
    string inpath = argv[1], outpath = argv[2];
    ifstream in(inpath);
    if (!in) { cerr << "cannot open input\n"; return 2; }
    vector<Graph> genes;
    vector<string> labels;
    unordered_map<string,int> labelId;
    string line;
    while (getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == string::npos) continue;
        Parser parser(line);
        Graph g = parser.parse();
        for (auto& node : g.nodes) {
            if (!node.label.empty() && node.adj.size() <= 1) {
                auto it = labelId.find(node.label);
                if (it == labelId.end()) {
                    int id = (int)labels.size();
                    labelId[node.label] = id;
                    labels.push_back(node.label);
                    node.taxon = id;
                } else node.taxon = it->second;
            }
        }
        genes.push_back(move(g));
    }
    int ntax = (int)labels.size();
    if (ntax < 4) { cerr << "too few taxa\n"; return 2; }
    int outgroup = -1;
    for (int i = 0; i < ntax; ++i) if (labels[i] == "0") outgroup = i;
    if (outgroup < 0) outgroup = 0;
    // Ensure every gene has the same taxon set.
    for (auto& g : genes) {
        vector<int> seen(ntax, 0);
        for (auto& node : g.nodes) if (node.taxon >= 0) {
            if (node.taxon >= ntax) { cerr << "taxon mismatch\n"; return 2; }
            seen[node.taxon]++;
        }
        for (int x : seen) if (x != 1) { cerr << "taxon mismatch\n"; return 2; }
    }
    {
        if (getenv("CONTRACT_LEN")) {
            double t = atof(getenv("CONTRACT_LEN"));
            for (auto& g : genes) g = contractGraph(g, t, true);
        } else {
            double t = getenv("CONTRACT") ? atof(getenv("CONTRACT")) : (getenv("NO_CONTRACT") ? 1.1 : 0.1);
            if (t < 1.0) for (auto& g : genes) g = contractGraph(g, t);
        }
    }
    vector<Rooted> rooted(genes.size());
    unordered_set<Bits, BitsHash> seen;
    vector<Cluster> clusters;
    for (size_t gi = 0; gi < genes.size(); ++gi) {
        rooted[gi].build(genes[gi], outgroup);
        collectRooted(rooted[gi], ntax, outgroup, seen, clusters);
    }
    if (getenv("LENGTH_NORM")) {
        for (size_t gi = 0; gi < genes.size(); ++gi) {
            double mean = 0; int cnt = 0;
            for (int v = 0; v < (int)genes[gi].nodes.size(); ++v)
                if (genes[gi].nodes[v].adj.size() == 3) { mean += rooted[gi].edgeL[v]; ++cnt; }
            if (cnt && mean > 0) {
                mean /= cnt;
                for (auto& x : rooted[gi].edgeL) x /= mean;
            }
        }
    }
    int guideMode = getenv("GUIDE_MODE") ? atoi(getenv("GUIDE_MODE")) : 0;
    vector<vector<double>> dist;
    if (guideMode == 10) {
        auto d0 = computeDistances(genes, rooted, ntax, outgroup, 0);
        auto d4 = computeDistances(genes, rooted, ntax, outgroup, 4);
        double m0 = 0, m4 = 0; int cnt = 0;
        for (int i = 0; i < ntax; ++i) for (int j = i + 1; j < ntax; ++j) { m0 += d0[i][j]; m4 += d4[i][j]; ++cnt; }
        m0 /= cnt; m4 /= cnt;
        double alpha = getenv("MIX_ALPHA") ? atof(getenv("MIX_ALPHA")) : 0.25;
        dist.assign(ntax, vector<double>(ntax, 0.0));
        for (int i = 0; i < ntax; ++i) for (int j = 0; j < ntax; ++j)
            dist[i][j] = alpha * d0[i][j] / m0 + (1 - alpha) * d4[i][j] / m4;
    } else {
        dist = computeDistances(genes, rooted, ntax, outgroup, guideMode);
    }
    Graph guide = neighborJoining(dist, labels);
    if (getenv("REFINE")) refineNNI(guide, dist, ntax);
    if (getenv("A_NNI")) astralNNI(guide, rooted, ntax, outgroup, atoi(getenv("A_NNI")));
    Rooted guideRooted; guideRooted.build(guide, outgroup);
    collectRooted(guideRooted, ntax, outgroup, seen, clusters);
    if (!getenv("NO_ADAPTIVE") && !getenv("LENGTH_P")) {
        vector<Bits> gsplits;
        for (int v = 0; v < (int)guideRooted.parent.size(); ++v)
            if (guideRooted.subsize[v] >= 2 && guideRooted.subsize[v] <= ntax - 2)
                gsplits.push_back(guideRooted.bits[v]);
        double miss = 0;
        for (const auto& rt : rooted) {
            unordered_set<Bits, BitsHash> sset;
            for (int v = 0; v < (int)rt.parent.size(); ++v)
                if (rt.subsize[v] >= 2 && rt.subsize[v] <= ntax - 2) sset.insert(rt.bits[v]);
            for (const auto& b : gsplits) if (!sset.count(b)) miss += 1;
        }
        double metric = miss / (gsplits.size() * rooted.size());
        g_lengthP = max(0.2, min(0.5, 0.5 - 2.5 * (metric - 0.6)));
        cerr << "conflict=" << metric << " lengthP=" << g_lengthP << "\n";
    }
    vector<Graph> allGuides;
    allGuides.push_back(guide);
    if (getenv("GUIDE_ONLY")) {
        ofstream os(outpath);
        if (!os) { cerr << "cannot open output\n"; return 2; }
        os << guideNewick(guide, guideRooted, labels, outgroup);
        return 0;
    }
    if (getenv("GUIDE_OUTPUT")) {
        ofstream os(outpath);
        if (!os) { cerr << "cannot open output\n"; return 2; }
        os << guideNewick(guide, guideRooted, labels, outgroup);
        cerr << "taxa=" << ntax << " genes=" << genes.size() << " guide\n";
        return 0;
    }
    if (!getenv("NO_EXTRA")) {
        for (int m : {2, 4, 5}) {
            auto dd = computeDistances(genes, rooted, ntax, outgroup, m);
            Graph gg = neighborJoining(dd, labels);
            Rooted rr; rr.build(gg, outgroup);
            collectRooted(rr, ntax, outgroup, seen, clusters);
            allGuides.push_back(gg);
        }
        auto d0 = computeDistances(genes, rooted, ntax, outgroup, 0);
        auto d4 = computeDistances(genes, rooted, ntax, outgroup, 4);
        double m0 = 0, m4 = 0; int cnt = 0;
        for (int i = 0; i < ntax; ++i) for (int j = i + 1; j < ntax; ++j) { m0 += d0[i][j]; m4 += d4[i][j]; ++cnt; }
        m0 /= cnt; m4 /= cnt;
        vector<vector<double>> dm(ntax, vector<double>(ntax, 0.0));
        for (int i = 0; i < ntax; ++i) for (int j = 0; j < ntax; ++j)
            dm[i][j] = 0.25 * d0[i][j] / m0 + 0.75 * d4[i][j] / m4;
        Graph gm = neighborJoining(dm, labels);
        Rooted rm; rm.build(gm, outgroup);
        collectRooted(rm, ntax, outgroup, seen, clusters);
        allGuides.push_back(gm);
        Graph gu = upgma(dm, labels);
        Rooted ru; ru.build(gu, outgroup);
        collectRooted(ru, ntax, outgroup, seen, clusters);
        allGuides.push_back(gu);
        Graph gb = bionj(dm, labels);
        Rooted rb; rb.build(gb, outgroup);
        collectRooted(rb, ntax, outgroup, seen, clusters);
        allGuides.push_back(gb);
    }
    if (getenv("ALL_PAIRS")) {
        for (int i = 0; i < ntax; ++i) for (int j = i + 1; j < ntax; ++j) {
            if (i == outgroup || j == outgroup) continue;
            Bits p; p.set(i); p.set(j);
            addCluster(seen, clusters, p, 2);
        }
    }
    if (!getenv("NO_NNI_EXPAND")) {
        vector<int> sel;
        string nset = getenv("NNI_SET") ? getenv("NNI_SET") : "nmbq";
        if (nset == "primary") sel = {0};
        else if (nset == "nmq") sel = {0, 2, 4};
        else if (nset == "nmbq") sel = {0, 2, 4, 6};
        else for (int i = 0; i < (int)allGuides.size(); ++i) sel.push_back(i);
        for (int si : sel) {
        const Graph& base = allGuides[si];
        Graph gg = base;
        for (int u = 0; u < (int)gg.nodes.size(); ++u) {
            if (gg.nodes[u].adj.size() != 3) continue;
            for (int v : gg.nodes[u].adj) {
                if (v <= u || gg.nodes[v].adj.size() != 3) continue;
                vector<int> au, av;
                for (int x : gg.nodes[u].adj) if (x != v) au.push_back(x);
                for (int x : gg.nodes[v].adj) if (x != u) av.push_back(x);
                if (au.size() != 2 || av.size() != 2) continue;
                for (int k = 0; k < 2; ++k) {
                    int a = au[0], c = av[k];
                    gg.removeEdge(u, a); gg.removeEdge(v, c);
                    gg.edge(u, c); gg.edge(v, a);
                    Rooted rr; rr.build(gg, outgroup);
                    collectRooted(rr, ntax, outgroup, seen, clusters);
                    gg.removeEdge(u, c); gg.removeEdge(v, a);
                    gg.edge(u, a); gg.edge(v, c);
                }
            }
        }
        }
    }

    sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
        if (a.size != b.size) return a.size < b.size;
        for (int k = MAXW - 1; k >= 0; --k) if (a.b.w[k] != b.b.w[k]) return a.b.w[k] < b.b.w[k];
        return false;
    });
    int nc = (int)clusters.size();
    unordered_map<Bits,int,BitsHash> index;
    index.reserve(nc * 2);
    for (int i = 0; i < nc; ++i) index[clusters[i].b] = i;
    vector<vector<int>> bySize(ntax + 1);
    for (int i = 0; i < nc; ++i) bySize[clusters[i].size].push_back(i);

    if (getenv("GREEDY")) {
        vector<uint8_t> mem(ntax);
        for (int i = 0; i < nc; ++i) {
            Cluster& c = clusters[i];
            if (c.size <= 1 || c.size >= ntax - 1) { c.weight = 0; continue; }
            fill(mem.begin(), mem.end(), 0);
            for (int t = 0; t < ntax; ++t) if (c.b.test(t)) mem[t] = 1;
            int64_t w = 0;
            for (const auto& rt : rooted) w += splitWeight(rt, mem.data(), c.size);
            c.weight = w;
        }
    }
    Bits full; for (int i = 0; i < ntax; ++i) if (i != outgroup) full.set(i);
    int fullIdx = index[full];
    if (getenv("GREEDY")) {
        int gmode = atoi(getenv("GREEDY"));
        function<void(int)> build = [&](int ai) {
            Cluster& a = clusters[ai];
            if (a.size == 1) return;
            int64_t best = -1; int bl = -1, br = -1;
            for (int bs = 1; bs <= a.size / 2; ++bs) {
                for (int bi : bySize[bs]) {
                    const Cluster& b = clusters[bi];
                    if (!b.b.subsetOf(a.b)) continue;
                    Bits cb = a.b.andnot(b.b);
                    auto it = index.find(cb);
                    if (it == index.end()) continue;
                    int ci = it->second;
                    int64_t val;
                    if (gmode == 1) val = max(clusters[bi].weight, clusters[ci].weight);
                    else if (gmode == 2) val = clusters[bi].weight + clusters[ci].weight - a.weight;
                    else val = clusters[bi].weight + clusters[ci].weight;
                    if (val > best) { best = val; bl = bi; br = ci; }
                }
            }
            if (bl < 0) {
                // split off any leaf if no candidate complement exists
                for (int t = 0; t < ntax; ++t) if (a.b.test(t)) {
                    Bits leaf; leaf.set(t);
                    Bits rest = a.b.andnot(leaf);
                    auto itl = index.find(leaf), itr = index.find(rest);
                    if (itl != index.end() && itr != index.end()) { bl = itl->second; br = itr->second; break; }
                }
            }
            a.left = bl; a.right = br;
            if (bl >= 0) { build(bl); build(br); }
        };
        build(fullIdx);
        string out;
        out += '('; out += quoteLabel(labels[outgroup]); out += ',';
        writeCluster(clusters, labels, clusters[fullIdx].left, out);
        out += ',';
        writeCluster(clusters, labels, clusters[fullIdx].right, out);
        out += ");\n";
        ofstream os(outpath); os << out;
        cerr << "taxa=" << ntax << " greedy\n";
        return 0;
    }
    int tripMode = getenv("TRIP_MODE") ? atoi(getenv("TRIP_MODE")) : 1;
    vector<double> geneW(genes.size(), 1.0);
    if (getenv("GENE_WEIGHT")) {
        double mean = 0;
        for (size_t gi = 0; gi < genes.size(); ++gi) {
            double s = 0; int cnt = 0;
            const auto& g = genes[gi];
            for (int u = 0; u < (int)g.nodes.size(); ++u) {
                if (g.nodes[u].adj.size() != 3) continue;
                for (size_t k = 0; k < g.nodes[u].adj.size(); ++k) {
                    int v = g.nodes[u].adj[k];
                    if (v > u && g.nodes[v].adj.size() == 3) { s += g.nodes[u].adjW[k]; ++cnt; }
                }
            }
            geneW[gi] = cnt ? s / cnt : 1.0;
            mean += geneW[gi];
        }
        mean /= genes.size();
        for (auto& w : geneW) w /= mean;
    }
    if (getenv("MULTI_P")) {
        double savedP = g_lengthP;
        for (double p : {0.2, 0.35, 0.5}) {
            g_lengthP = p;
            int fi = solveTripDP(clusters, rooted, ntax, outgroup, tripMode, geneW);
            if (clusters[fi].score >= 0 && clusters[fi].left >= 0) {
                Graph tg = graphFromClusters(clusters, labels, fi, outgroup);
                Rooted rr; rr.build(tg, outgroup);
                collectRooted(rr, ntax, outgroup, seen, clusters);
            }
        }
        g_lengthP = savedP;
    }
    fullIdx = solveTripDP(clusters, rooted, ntax, outgroup, tripMode, geneW);
    if (getenv("ITER_EXPAND") && clusters[fullIdx].score >= 0 && clusters[fullIdx].left >= 0) {
        Graph tg = graphFromClusters(clusters, labels, fullIdx, outgroup);
        for (int u = 0; u < (int)tg.nodes.size(); ++u) {
            if (tg.nodes[u].adj.size() != 3) continue;
            for (int v : tg.nodes[u].adj) {
                if (v <= u || tg.nodes[v].adj.size() != 3) continue;
                vector<int> au, av;
                for (int x : tg.nodes[u].adj) if (x != v) au.push_back(x);
                for (int x : tg.nodes[v].adj) if (x != u) av.push_back(x);
                if (au.size() != 2 || av.size() != 2) continue;
                for (int k = 0; k < 2; ++k) {
                    int a = au[0], c = av[k];
                    tg.removeEdge(u, a); tg.removeEdge(v, c);
                    tg.edge(u, c); tg.edge(v, a);
                    Rooted rr; rr.build(tg, outgroup);
                    collectRooted(rr, ntax, outgroup, seen, clusters);
                    tg.removeEdge(u, c); tg.removeEdge(v, a);
                    tg.edge(u, a); tg.edge(v, c);
                }
            }
        }
        fullIdx = solveTripDP(clusters, rooted, ntax, outgroup, tripMode, geneW);
    }
    nc = (int)clusters.size();
    string out;
    if (clusters[fullIdx].score >= 0 && clusters[fullIdx].left >= 0) {
        if (getenv("TRIP_NNI")) {
            Graph tg = graphFromClusters(clusters, labels, fullIdx, outgroup);
            tripNNI(tg, rooted, ntax, outgroup, atoi(getenv("TRIP_NNI")), tripMode);
            Rooted tr; tr.build(tg, outgroup);
            out = guideNewick(tg, tr, labels, outgroup);
        } else {
            out += '(';
            out += quoteLabel(labels[outgroup]);
            out += ',';
            writeCluster(clusters, labels, clusters[fullIdx].left, out);
            out += ',';
            writeCluster(clusters, labels, clusters[fullIdx].right, out);
            out += ");\n";
        }
    } else {
        // Fallback: guide tree, rooted at outgroup for serialization.
        out = guideNewick(guide, guideRooted, labels, outgroup);
    }
    ofstream os(outpath);
    if (!os) { cerr << "cannot open output\n"; return 2; }
    os << out;
    os.close();
    cerr << "taxa=" << ntax << " genes=" << genes.size() << " clusters=" << nc
         << " score=" << clusters[fullIdx].score << "\n";
    return 0;
}
