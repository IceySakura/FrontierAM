// Weighted-quartet species tree inference (ASTRAL-style exact DP over a
// constrained bipartition set, with support/length weighted quartets in the
// spirit of weighted ASTRAL).  Single threaded, deterministic.
#include <bits/stdc++.h>
using namespace std;

typedef double v4d __attribute__((vector_size(32)));
typedef long long v4i __attribute__((vector_size(32)));

static chrono::steady_clock::time_point T0;
static double elapsed() { return chrono::duration<double>(chrono::steady_clock::now() - T0).count(); }

static int NT;          // number of taxa
static int OUTG;        // outgroup taxon index
static vector<string> names;

struct BS {
    uint64_t w[4];
    BS() { w[0] = w[1] = w[2] = w[3] = 0; }
    void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    bool get(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    int cnt() const { return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1]) + __builtin_popcountll(w[2]) + __builtin_popcountll(w[3]); }
    bool operator==(const BS& o) const { return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2] && w[3] == o.w[3]; }
    BS operator|(const BS& o) const { BS r; for (int k = 0; k < 4; k++) r.w[k] = w[k] | o.w[k]; return r; }
    BS operator&(const BS& o) const { BS r; for (int k = 0; k < 4; k++) r.w[k] = w[k] & o.w[k]; return r; }
    BS andnot(const BS& o) const { BS r; for (int k = 0; k < 4; k++) r.w[k] = w[k] & ~o.w[k]; return r; }
    bool subsetOf(const BS& o) const { for (int k = 0; k < 4; k++) if (w[k] & ~o.w[k]) return false; return true; }
    int lowest() const { for (int k = 0; k < 4; k++) if (w[k]) return k * 64 + __builtin_ctzll(w[k]); return -1; }
};
struct BSH { size_t operator()(const BS& b) const { uint64_t h = 1469598103934665603ULL; for (int k = 0; k < 4; k++) { h ^= b.w[k] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); } return h; } };

// ---------------------------------------------------------------- parsing
struct RawTree { vector<int> par; vector<vector<int>> ch; vector<double> len, sup; vector<string> lab; };

static void skipws(const string& s, size_t& i) { while (i < s.size() && isspace((unsigned char)s[i])) i++; }
static string readLabel(const string& s, size_t& i) {
    skipws(s, i);
    string r;
    if (i < s.size() && s[i] == '\'') {
        i++;
        while (i < s.size()) {
            if (s[i] == '\'') { if (i + 1 < s.size() && s[i + 1] == '\'') { r += '\''; i += 2; continue; } i++; break; }
            r += s[i++];
        }
        return r;
    }
    while (i < s.size() && s[i] != ':' && s[i] != ',' && s[i] != ')' && s[i] != '(' && s[i] != ';' && s[i] != '[' && !isspace((unsigned char)s[i])) r += s[i++];
    return r;
}
static void skipComment(const string& s, size_t& i) {
    skipws(s, i);
    while (i < s.size() && s[i] == '[') { while (i < s.size() && s[i] != ']') i++; if (i < s.size()) i++; skipws(s, i); }
}
static int parseSub(const string& s, size_t& i, RawTree& t, int parent) {
    int id = t.par.size();
    t.par.push_back(parent); t.ch.emplace_back(); t.len.push_back(0); t.sup.push_back(-1); t.lab.emplace_back();
    skipComment(s, i);
    if (i < s.size() && s[i] == '(') {
        i++;
        while (true) {
            int c = parseSub(s, i, t, id);
            t.ch[id].push_back(c);
            skipComment(s, i);
            if (i < s.size() && s[i] == ',') { i++; continue; }
            if (i < s.size() && s[i] == ')') { i++; break; }
            throw runtime_error("bad newick");
        }
        skipComment(s, i);
        string l = readLabel(s, i);
        if (!l.empty()) { char* e; double v = strtod(l.c_str(), &e); if (e != l.c_str() && *e == 0 && v >= 0 && v <= 1.0000001) t.sup[id] = v; }
    } else {
        t.lab[id] = readLabel(s, i);
    }
    skipComment(s, i);
    if (i < s.size() && s[i] == ':') {
        i++; skipws(s, i);
        size_t j = i;
        while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' || s[j] == '-' || s[j] == '+')) j++;
        double v = atof(s.substr(i, j - i).c_str());
        if (!(v >= 0) || !isfinite(v)) v = 0;
        t.len[id] = v; i = j;
    }
    return id;
}

// Unrooted gene tree as edge list on nodes; leaves carry taxon ids.
struct UTree {
    int nn;
    vector<int> taxon;                     // -1 internal
    vector<vector<pair<int, int>>> adj;    // (neighbor, edge id)
    vector<double> elen, esup;             // per edge
};

// Gene tree rooted at the outgroup leaf, binary, nodes in post-order.
struct Gene {
    int m;                 // nodes excluding outgroup root; index m-1 is the outgroup's neighbor r
    vector<int> c1, c2;    // children (-1 for leaves)
    vector<int> tax;       // taxon for leaves
    vector<double> e, f;   // edge to parent: exp(-lambda*len), failure factor 1-s'
    vector<double> len, sup; // raw
    vector<char> isTerm;
    double eo, leno;       // edge r-outgroup
    vector<int> parent;
};

static vector<UTree> utrees;
static vector<Gene> genes;
static double LAMBDA = 0.0;
static double SPOWV = 3.0;
static double SMIN = getenv("STREE_SMINV") ? atof(getenv("STREE_SMINV")) : 1.0 / 3.0;

static bool buildUTree(const RawTree& t, unordered_map<string, int>& nameIdx, bool first, UTree& u) {
    int n = t.par.size();
    // count children of root
    int root = 0;
    vector<int> mapId(n, -1);
    u.nn = 0;
    u.taxon.clear(); u.adj.clear(); u.elen.clear(); u.esup.clear();
    auto newNode = [&](int tax) { u.taxon.push_back(tax); u.adj.emplace_back(); return u.nn++; };
    vector<char> seen(NT > 0 ? NT : 1, 0);
    for (int v = 0; v < n; v++) {
        if (t.ch[v].empty()) {
            int tax;
            auto it = nameIdx.find(t.lab[v]);
            if (it == nameIdx.end()) {
                if (!first) return false;
                tax = names.size(); names.push_back(t.lab[v]); nameIdx[t.lab[v]] = tax;
            } else tax = it->second;
            mapId[v] = newNode(tax);
        } else {
            if (v == root && t.ch[v].size() == 2) continue; // suppressed
            mapId[v] = newNode(-1);
        }
    }
    auto addEdge = [&](int a, int b, double l, double s) {
        int id = u.elen.size(); u.elen.push_back(l); u.esup.push_back(s);
        u.adj[a].push_back({b, id}); u.adj[b].push_back({a, id});
    };
    for (int v = 0; v < n; v++) {
        if (v == root) continue;
        int p = t.par[v];
        double s = t.ch[v].empty() ? -1 : t.sup[v];
        if (p == root && t.ch[root].size() == 2) continue;
        addEdge(mapId[v], mapId[p], t.len[v], s);
    }
    if (t.ch[root].size() == 2) {
        int x = t.ch[root][0], y = t.ch[root][1];
        double sx = t.ch[x].empty() ? -1 : t.sup[x], sy = t.ch[y].empty() ? -1 : t.sup[y];
        addEdge(mapId[x], mapId[y], t.len[x] + t.len[y], max(sx, sy));
    }
    return true;
}

// root UTree at outgroup, binarize, produce Gene
static bool makeGene(const UTree& u, Gene& g) {
    int o = -1;
    for (int v = 0; v < u.nn; v++) if (u.taxon[v] == OUTG) { o = v; break; }
    if (o < 0) return false;
    // iterative DFS to get parent / children
    int nn = u.nn;
    vector<int> par(nn, -2), pedge(nn, -1), order;
    vector<vector<int>> kids(nn);
    vector<int> st = {o}; par[o] = -1;
    while (!st.empty()) {
        int v = st.back(); st.pop_back(); order.push_back(v);
        for (auto& pr : u.adj[v]) if (par[pr.first] == -2) { par[pr.first] = v; pedge[pr.first] = pr.second; kids[v].push_back(pr.first); st.push_back(pr.first); }
    }
    if ((int)order.size() != nn) return false;
    if (kids[o].size() != 1) return false;
    // build binary nodes: new arrays
    vector<int> C1, C2, TX; vector<double> L, S; vector<char> TERM;
    // build binary post-order arrays (polytomies resolved with zero-support edges)
    function<int(int)> build2 = [&](int v) -> int {
        vector<int> cids;
        for (int c : kids[v]) {
            int cid = build2(c);
            int eid = pedge[c];
            L[cid] = u.elen[eid];
            S[cid] = TERM[cid] ? -1 : u.esup[eid];
            cids.push_back(cid);
        }
        int id;
        if (cids.empty()) {
            id = C1.size(); C1.push_back(-1); C2.push_back(-1); TX.push_back(u.taxon[v]); L.push_back(0); S.push_back(-1); TERM.push_back(1);
            return id;
        }
        int cur = cids[0];
        for (size_t k = 1; k < cids.size(); k++) {
            id = C1.size(); C1.push_back(cur); C2.push_back(cids[k]); TX.push_back(-1); L.push_back(0); S.push_back(SMIN); TERM.push_back(0);
            cur = id;
        }
        return cur;
    };
    int r = build2(kids[o][0]);
    g.m = C1.size();
    if (r != g.m - 1) return false;
    g.c1 = C1; g.c2 = C2; g.tax = TX; g.len = L; g.sup = S; g.isTerm = TERM;
    g.leno = u.elen[pedge[kids[o][0]]];
    g.parent.assign(g.m, -1);
    for (int v = 0; v < g.m; v++) if (g.c1[v] >= 0) { g.parent[g.c1[v]] = v; g.parent[g.c2[v]] = v; }
    return true;
}

static void setWeights(Gene& g) {
    g.e.resize(g.m); g.f.resize(g.m);
    for (int v = 0; v < g.m; v++) {
        g.e[v] = exp(-LAMBDA * g.len[v]);
        if (g.isTerm[v]) g.f[v] = 0;
        else {
            double s = g.sup[v];
            double sp = (s - SMIN) / (1.0 - SMIN);
            if (getenv("STREE_UNW")) sp = 1;
            sp = pow(max(0.0, sp), getenv("STREE_SPOW") ? atof(getenv("STREE_SPOW")) : SPOWV);
            if (sp < 0) sp = 0; if (sp > 1) sp = 1;
            g.f[v] = 1 - sp;
        }
    }
    g.eo = exp(-LAMBDA * g.leno);
}

// ---------------------------------------------------------------- scoring
static vector<v4d> bN, bG, bH, bUM, bUG, bUH;
static vector<int> partOf;
static long long nScored = 0;

static inline v4d rot1(v4d x) { return __builtin_shuffle(x, (v4i){1, 2, 0, 3}); }
static inline v4d rot2(v4d x) { return __builtin_shuffle(x, (v4i){2, 0, 1, 3}); }
static inline v4d cross(v4d x, v4d y) { return rot1(x) * rot2(y) + rot2(x) * rot1(y); }
static inline v4d bc(double x) { return (v4d){x, x, x, x}; }

static double scoreTri(const BS& A1, const BS& A2) {
    nScored++;
    static const v4d oh[3] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    v4d pv[256];
    for (int t = 0; t < NT; t++) pv[t] = A1.get(t) ? oh[0] : (A2.get(t) ? oh[1] : oh[2]);
    double total = 0;
    for (const Gene& g : genes) {
        int m = g.m;
        v4d* N = bN.data(); v4d* G = bG.data(); v4d* H = bH.data();
        v4d* UM = bUM.data(); v4d* UG = bUG.data(); v4d* UH = bUH.data();
        const int* c1 = g.c1.data(); const int* c2 = g.c2.data();
        const double* e = g.e.data(); const double* f = g.f.data();
        for (int v = 0; v < m; v++) {
            int a = c1[v];
            if (a < 0) { N[v] = pv[g.tax[v]]; G[v] = bc(0); H[v] = bc(0); continue; }
            int b = c2[v];
            v4d Ma = N[a] * e[a], Mb = N[b] * e[b];
            N[v] = Ma + Mb;
            v4d MM = Ma * Mb;
            G[v] = MM + G[a] + G[b];
            H[v] = MM + f[a] * H[a] + f[b] * H[b];
        }
        int r = m - 1;
        UM[r] = pv[OUTG] * g.eo; UG[r] = bc(0); UH[r] = bc(0);
        v4d acc = bc(0);
        for (int v = m - 1; v >= 0; v--) {
            int a = c1[v];
            if (a < 0) continue;
            int b = c2[v];
            v4d M0 = N[a] * e[a], G0 = G[a], H0 = f[a] * H[a];
            v4d M1 = N[b] * e[b], G1 = G[b], H1 = f[b] * H[b];
            v4d M2 = UM[v], G2 = UG[v], H2 = UH[v];
            acc += (G0 - H0) * cross(M1, M2) + (G1 - H1) * cross(M0, M2) + (G2 - H2) * cross(M0, M1);
            if (c1[a] >= 0) { v4d MM = M1 * M2; UM[a] = e[a] * (M1 + M2); UG[a] = MM + G1 + G2; UH[a] = f[a] * (MM + H1 + H2); }
            if (c1[b] >= 0) { v4d MM = M0 * M2; UM[b] = e[b] * (M0 + M2); UG[b] = MM + G0 + G2; UH[b] = f[b] * (MM + H0 + H2); }
        }
        total += acc[0] + acc[1] + acc[2];
    }
    return total;
}

// ---------------------------------------------------------------- cluster set + DP
static vector<BS> X;
static vector<int> Xsz;
static unordered_map<BS, int, BSH> Xidx;
static unordered_map<uint64_t, double> scoreCache;

static int addCluster(const BS& b) {
    auto it = Xidx.find(b);
    if (it != Xidx.end()) return it->second;
    int id = X.size(); X.push_back(b); Xsz.push_back(b.cnt()); Xidx[b] = id;
    return id;
}
static BS ingroupAll() { BS b; for (int t = 0; t < NT; t++) if (t != OUTG) b.set(t); return b; }

static void addGeneClusters(const Gene& g) {
    vector<BS> cl(g.m);
    for (int v = 0; v < g.m; v++) {
        if (g.c1[v] < 0) { cl[v] = BS(); cl[v].set(g.tax[v]); }
        else cl[v] = cl[g.c1[v]] | cl[g.c2[v]];
        addCluster(cl[v]);
    }
}

struct DPRes { double score; vector<int> bestSplitA1; vector<int> bestSplitA2; };

static vector<vector<pair<int, int>>> splits; // per cluster: (i1,i2)
static size_t splitsBuiltFor = 0;

static void buildSplits() {
    int K = X.size();
    // per taxon list of clusters containing it
    vector<vector<int>> byTax(NT);
    for (int i = 0; i < K; i++) {
        const BS& b = X[i];
        for (int k = 0; k < 4; k++) { uint64_t w = b.w[k]; while (w) { int t = k * 64 + __builtin_ctzll(w); w &= w - 1; byTax[t].push_back(i); } }
    }
    splits.assign(K, {});
    for (int i = 0; i < K; i++) {
        if (Xsz[i] < 2) continue;
        const BS& A = X[i];
        int lo = A.lowest();
        for (int j : byTax[lo]) {
            if (Xsz[j] >= Xsz[i]) continue;
            if (!X[j].subsetOf(A)) continue;
            BS rest = A.andnot(X[j]);
            auto it = Xidx.find(rest);
            if (it == Xidx.end()) continue;
            splits[i].push_back({j, it->second});
        }
    }
    splitsBuiltFor = K;
}

static double runDP(vector<int>& choice) {
    int K = X.size();
    vector<int> ord(K);
    iota(ord.begin(), ord.end(), 0);
    stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return Xsz[a] < Xsz[b]; });
    vector<double> best(K, -1e300);
    choice.assign(K, -1);
    for (int i : ord) {
        if (Xsz[i] == 1) { best[i] = 0; continue; }
        double bv = -1e300; int bc_ = -1;
        for (size_t s = 0; s < splits[i].size(); s++) {
            int a = splits[i][s].first, b = splits[i][s].second;
            if (best[a] < -1e299 || best[b] < -1e299) continue;
            uint64_t key = ((uint64_t)min(a, b) << 32) | (uint64_t)max(a, b);
            double sc;
            auto it = scoreCache.find(key);
            if (it == scoreCache.end()) { sc = scoreTri(X[a], X[b]); scoreCache[key] = sc; }
            else sc = it->second;
            double v = best[a] + best[b] + sc;
            if (v > bv) { bv = v; bc_ = s; }
        }
        best[i] = bv; choice[i] = bc_;
    }
    int root = Xidx[ingroupAll()];
    return best[root];
}

// extract tree: list of internal clusters (as children pairs)
static void collectTree(int i, const vector<int>& choice, vector<array<int, 3>>& nodes) {
    if (Xsz[i] == 1) return;
    auto pr = splits[i][choice[i]];
    nodes.push_back({i, pr.first, pr.second});
    collectTree(pr.first, choice, nodes);
    collectTree(pr.second, choice, nodes);
}

static string newickOf(int i, const vector<int>& choice) {
    if (Xsz[i] == 1) return names[X[i].lowest()];
    auto pr = splits[i][choice[i]];
    return "(" + newickOf(pr.first, choice) + "," + newickOf(pr.second, choice) + ")";
}

// ---------------------------------------------------------------- NJ on distance matrix
static void njClusters(const vector<vector<double>>& D0) {
    int n = NT;
    vector<vector<double>> D = D0;
    // node ids: 0..n-1 leaves, new internal from n
    vector<int> active(n); iota(active.begin(), active.end(), 0);
    vector<int> nodeOf(n); iota(nodeOf.begin(), nodeOf.end(), 0);
    int nextId = n;
    vector<vector<int>> adjT(2 * n);
    D.resize(n);
    while (active.size() > 3) {
        int k = active.size();
        vector<double> R(k, 0);
        for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) if (a != b) R[a] += D[active[a]][active[b]];
        double bq = 1e300; int ba = -1, bb = -1;
        for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) {
            double q = (k - 2) * D[active[a]][active[b]] - R[a] - R[b];
            if (q < bq) { bq = q; ba = a; bb = b; }
        }
        int i = active[ba], j = active[bb];
        int u = nextId++;
        adjT[u].push_back(nodeOf[i]); adjT[nodeOf[i]].push_back(u);
        adjT[u].push_back(nodeOf[j]); adjT[nodeOf[j]].push_back(u);
        // reuse row i for new node
        for (int c = 0; c < k; c++) {
            int x = active[c];
            if (x == i || x == j) continue;
            double d = 0.5 * (D[i][x] + D[j][x] - D[i][j]);
            D[i][x] = D[x][i] = d;
        }
        nodeOf[i] = u;
        active.erase(active.begin() + bb);
    }
    int u = nextId++;
    for (int x : active) { adjT[u].push_back(nodeOf[x]); adjT[nodeOf[x]].push_back(u); }
    // root at outgroup leaf OUTG (node id OUTG)
    int tot = nextId;
    vector<int> par(tot, -2); vector<int> order; vector<int> st = {OUTG}; par[OUTG] = -1;
    while (!st.empty()) { int v = st.back(); st.pop_back(); order.push_back(v); for (int w : adjT[v]) if (par[w] == -2) { par[w] = v; st.push_back(w); } }
    vector<BS> cl(tot);
    for (int idx = order.size() - 1; idx >= 0; idx--) {
        int v = order[idx];
        if (v < n) { if (v != OUTG) cl[v].set(v); }
        if (v == OUTG) continue;
        if (par[v] >= 0 && par[v] != OUTG) cl[par[v]] = cl[par[v]] | cl[v];
        if (cl[v].cnt() > 0) addCluster(cl[v]);
    }
}

static vector<vector<float>> geneD; // per gene flattened n*n support-weighted distances
static void geneDistances() {
    int n = NT;
    geneD.assign(genes.size(), vector<float>(n * n, 0));
    for (size_t gi = 0; gi < genes.size(); gi++) {
        const Gene& g = genes[gi];
        int m = g.m;
        vector<vector<pair<int, double>>> adj(m + 1);
        for (int v = 0; v < m; v++) {
            int p = (v == m - 1) ? m : g.parent[v];
            double w = g.isTerm[v] ? 1.0 : (1 - g.f[v]);
            adj[v].push_back({p, w}); adj[p].push_back({v, w});
        }
        vector<int> leafNode(n, -1);
        for (int v = 0; v < m; v++) if (g.c1[v] < 0) leafNode[g.tax[v]] = v;
        leafNode[OUTG] = m;
        vector<double> dist(m + 1); vector<int> st;
        for (int t = 0; t < n; t++) {
            fill(dist.begin(), dist.end(), -1);
            dist[leafNode[t]] = 0; st.assign(1, leafNode[t]);
            while (!st.empty()) { int v = st.back(); st.pop_back(); for (auto& pr : adj[v]) if (dist[pr.first] < 0) { dist[pr.first] = dist[v] + pr.second; st.push_back(pr.first); } }
            for (int t2 = 0; t2 < n; t2++) geneD[gi][t * n + t2] = dist[leafNode[t2]];
        }
    }
}
static void bootstrapNJ(int reps, uint64_t seed, double tlim) {
    int n = NT, k = genes.size();
    mt19937_64 rng(seed);
    vector<vector<double>> D(n, vector<double>(n));
    for (int r = 0; r < reps && elapsed() < tlim; r++) {
        for (auto& row : D) fill(row.begin(), row.end(), 0.0);
        for (int j = 0; j < k; j++) {
            const vector<float>& gd = geneD[rng() % k];
            for (int a = 0; a < n; a++) { double* row = D[a].data(); const float* src = gd.data() + a * n; for (int b = 0; b < n; b++) row[b] += src[b]; }
        }
        njClusters(D);
    }
}

static void distanceMatrix(int mode, vector<vector<double>>& D) {
    // mode 0: topological internode distance; mode 1: support weighted internal edges
    int n = NT;
    D.assign(n, vector<double>(n, 0));
    for (const Gene& g : genes) {
        int m = g.m;
        // build adjacency incl outgroup as node m
        vector<vector<pair<int, double>>> adj(m + 1);
        for (int v = 0; v < m; v++) {
            int p = (v == m - 1) ? m : g.parent[v];
            double w = g.isTerm[v] ? 1.0 : (mode == 0 ? 1.0 : (1 - g.f[v]));
            adj[v].push_back({p, w}); adj[p].push_back({v, w});
        }
        vector<int> leafNode(n, -1);
        for (int v = 0; v < m; v++) if (g.c1[v] < 0) leafNode[g.tax[v]] = v;
        leafNode[OUTG] = m;
        vector<double> dist(m + 1);
        vector<int> st;
        for (int t = 0; t < n; t++) {
            int s = leafNode[t];
            fill(dist.begin(), dist.end(), -1);
            dist[s] = 0; st.assign(1, s);
            while (!st.empty()) { int v = st.back(); st.pop_back(); for (auto& pr : adj[v]) if (dist[pr.first] < 0) { dist[pr.first] = dist[v] + pr.second; st.push_back(pr.first); } }
            for (int t2 = 0; t2 < n; t2++) D[t][t2] += dist[leafNode[t2]];
        }
    }
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) D[a][b] /= genes.size();
}

// add clusters of all SPR neighbours within tree distance R of the current tree
static void sprClusters(const vector<array<int, 3>>& nodes, int root, int R) {
    // build tree over cluster ids
    unordered_map<int, int> loc; vector<int> cid; vector<int> par; vector<vector<int>> kids;
    function<int(int, int)> mk = [&](int c, int p) -> int {
        int id = cid.size(); cid.push_back(c); par.push_back(p); kids.emplace_back(); loc[c] = id;
        return id;
    };
    unordered_map<int, pair<int, int>> kidsOf;
    for (auto& nd : nodes) kidsOf[nd[0]] = {nd[1], nd[2]};
    vector<int> st = {mk(root, -1)};
    while (!st.empty()) {
        int v = st.back(); st.pop_back();
        auto it = kidsOf.find(cid[v]);
        if (it == kidsOf.end()) continue;
        int a = mk(it->second.first, v), b = mk(it->second.second, v);
        kids[v] = {a, b}; st.push_back(a); st.push_back(b);
    }
    int M = cid.size();
    vector<int> depth(M, 0);
    for (int v = 1; v < M; v++) depth[v] = depth[par[v]] + 1; // parents created before children
    for (int s = 1; s < M; s++) {
        const BS& S = X[cid[s]];
        // climb ancestors up to R steps
        int a = par[s], prev = s, k = 1;
        while (a >= 0 && k <= R) {
            // removing s: clusters from par[s] (excl) up to a (excl) lose s; handled per target
            // targets: a itself (if a != par[s]) and nodes in the other subtree of a within R-k
            vector<pair<int, int>> todo; // (node, dist from a)
            if (a != par[s]) todo.push_back({a, 0}); // w = a (ancestor case)
            for (int c : kids[a]) if (c != prev) todo.push_back({c, 1});
            while (!todo.empty()) {
                auto [w, dw] = todo.back(); todo.pop_back();
                if (k + dw > R) continue;
                if (w == a) {
                    // ancestor target: clusters from par(par[s]) up to a inclusive lose s
                    for (int x = par[par[s]] >= 0 ? par[s] : -1; x >= 0; x = par[x]) {
                        if (x != par[s]) addCluster(X[cid[x]].andnot(S));
                        if (x == a) break;
                    }
                    continue;
                }
                // w in other subtree of a: path w..(child of a) gains s, path par[s]..(child of a) loses s
                for (int x = par[s]; x != a; x = par[x]) if (x != par[s]) addCluster(X[cid[x]].andnot(S));
                for (int x = w; x != a; x = par[x]) addCluster(X[cid[x]] | S);
                if (w != (int)(kids[a][0] == prev ? kids[a][1] : kids[a][0]) || true)
                    for (int c : kids[w]) todo.push_back({c, dw + 1});
            }
            prev = a; a = par[a]; k++;
        }
    }
}

int main(int argc, char** argv) {
    T0 = chrono::steady_clock::now();
    if (argc < 3) { fprintf(stderr, "usage: %s genes.tre out.tre\n", argv[0]); return 1; }
    if (getenv("STREE_LAMBDA")) LAMBDA = atof(getenv("STREE_LAMBDA"));
    double TLIM = getenv("STREE_TLIM") ? atof(getenv("STREE_TLIM")) : 75.0;
    int maxIter = getenv("STREE_MAXITER") ? atoi(getenv("STREE_MAXITER")) : 1000;
    ifstream in(argv[1]);
    string line;
    unordered_map<string, int> nameIdx;
    vector<RawTree> raws;
    string all((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    // split by ';'
    size_t pos = 0;
    while (true) {
        size_t sc = all.find(';', pos);
        if (sc == string::npos) break;
        string s = all.substr(pos, sc - pos + 1);
        pos = sc + 1;
        size_t k = 0; skipComment(s, k);
        if (k >= s.size() || s[k] != '(') continue;
        try { RawTree t; size_t i = k; parseSub(s, i, t, -1); raws.push_back(move(t)); } catch (...) { fprintf(stderr, "parse error, skipping tree\n"); }
    }
    if (raws.empty()) { fprintf(stderr, "no trees\n"); return 1; }
    // taxa from first tree
    NT = 0;
    {
        bool first = true;
        for (auto& t : raws) {
            UTree u;
            if (first) {
                NT = 0; for (auto& c : t.ch) if (c.empty()) NT++;
                // set names via buildUTree with first=true
                names.clear();
                // temporary NT for seen vector
                if (!buildUTree(t, nameIdx, true, u)) continue;
                NT = names.size();
                first = false;
                utrees.push_back(u);
            } else {
                int nl = 0; for (auto& c : t.ch) if (c.empty()) nl++;
                if (nl != NT) { fprintf(stderr, "tree with different taxa skipped\n"); continue; }
                if (!buildUTree(t, nameIdx, false, u)) { fprintf(stderr, "tree with unknown taxa skipped\n"); continue; }
                // check all taxa distinct
                vector<char> sn(NT, 0); bool ok = true;
                for (int v = 0; v < u.nn; v++) if (u.taxon[v] >= 0) { if (sn[u.taxon[v]]) ok = false; sn[u.taxon[v]] = 1; }
                if (!ok) continue;
                utrees.push_back(u);
            }
        }
    }
    OUTG = 0;
    if (nameIdx.count("0")) OUTG = nameIdx["0"];
    fprintf(stderr, "taxa %d trees %zu\n", NT, utrees.size());
    if (NT < 4) {
        ofstream out(argv[2]);
        string s = "(";
        for (int t = 0; t < NT; t++) { if (t) s += ","; s += names[t]; }
        s += ");\n"; out << s; return 0;
    }
    // impute missing supports from branch length
    {
        vector<pair<double, double>> ls;
        for (auto& u : utrees) for (int v = 0; v < u.nn; v++) for (auto& pr : u.adj[v]) if (v < pr.first) {
            int a = v, b = pr.first;
            if (u.taxon[a] >= 0 || u.taxon[b] >= 0) continue;
            if (u.esup[pr.second] >= 0) ls.push_back({u.elen[pr.second], u.esup[pr.second]});
        }
        sort(ls.begin(), ls.end());
        int nmiss = 0;
        for (auto& u : utrees) for (int v = 0; v < u.nn; v++) for (auto& pr : u.adj[v]) if (v < pr.first) {
            int a = v, b = pr.first;
            if (u.taxon[a] >= 0 || u.taxon[b] >= 0) continue;
            double& s = u.esup[pr.second];
            if (s >= 0) continue;
            nmiss++;
            double L = u.elen[pr.second];
            if (ls.size() < 20) { s = 1.0 - (2.0 / 3.0) * exp(-200.0 * L); continue; }
            size_t p = lower_bound(ls.begin(), ls.end(), make_pair(L, -1.0)) - ls.begin();
            size_t lo = p >= 25 ? p - 25 : 0, hi = min(ls.size(), lo + 50);
            if (hi - lo < 50) lo = hi >= 50 ? hi - 50 : 0;
            double sm = 0; for (size_t q = lo; q < hi; q++) sm += ls[q].second;
            s = sm / (hi - lo);
        }
        fprintf(stderr, "missing supports imputed: %d\n", nmiss);
    }
    for (auto& u : utrees) { Gene g; if (makeGene(u, g)) { setWeights(g); genes.push_back(move(g)); } }
    fprintf(stderr, "genes %zu\n", genes.size());
    int maxm = 0; for (auto& g : genes) maxm = max(maxm, g.m);
    bN.resize(maxm); bG.resize(maxm); bH.resize(maxm); bUM.resize(maxm); bUG.resize(maxm); bUH.resize(maxm);

    // cluster set
    for (int t = 0; t < NT; t++) if (t != OUTG) { BS b; b.set(t); addCluster(b); }
    addCluster(ingroupAll());
    for (auto& g : genes) addGeneClusters(g);
    fprintf(stderr, "clusters from genes %zu\n", X.size());
    for (int mode = 0; mode < 2; mode++) {
        vector<vector<double>> D;
        distanceMatrix(mode, D);
        njClusters(D);
    }
    fprintf(stderr, "clusters after NJ %zu  (%.2fs)\n", X.size(), elapsed());
    {
        int reps = getenv("STREE_BOOT") ? atoi(getenv("STREE_BOOT")) : 100;
        uint64_t seed = getenv("SEED") ? strtoull(getenv("SEED"), 0, 10) : 1;
        if (reps > 0) { geneDistances(); bootstrapNJ(reps, seed * 7919 + 17, 20.0); }
        geneD.clear(); geneD.shrink_to_fit();
        fprintf(stderr, "clusters after bootstrap NJ %zu  (%.2fs)\n", X.size(), elapsed());
    }

    if (getenv("STREE_DIAG_TREE")) {
        ifstream dt(getenv("STREE_DIAG_TREE")); string s((istreambuf_iterator<char>(dt)), istreambuf_iterator<char>());
        RawTree t; size_t i = 0; skipComment(s, i); parseSub(s, i, t, -1);
        UTree u; buildUTree(t, nameIdx, false, u); Gene g; makeGene(u, g); addGeneClusters(g);
        fprintf(stderr, "added diag tree clusters -> %zu\n", X.size());
    }

    vector<int> choice;
    double bestScore = -1e300;
    string bestNewick;
    for (int iter = 0; iter < maxIter; iter++) {
        buildSplits();
        size_t ns = 0; for (auto& s : splits) ns += s.size();
        double sc = runDP(choice);
        int root = Xidx[ingroupAll()];
        fprintf(stderr, "iter %d clusters %zu splits %zu scored %lld score %.6g (%.2fs)\n", iter, X.size(), ns, nScored, sc, elapsed());
        if (sc <= bestScore * (1 + 1e-12) && iter > 0) break;
        bestScore = sc;
        // output tree
        {
            auto pr = splits[root][choice[root]];
            bestNewick = "(" + names[OUTG] + "," + newickOf(pr.first, choice) + "," + newickOf(pr.second, choice) + ");";
            ofstream out(argv[2]); out << bestNewick << "\n";
        }
        if (elapsed() > TLIM) break;
        // NNI enrichment
        vector<array<int, 3>> nodes;
        collectTree(root, choice, nodes);
        // map cluster -> (children)
        unordered_map<int, pair<int, int>> kidsOf;
        for (auto& nd : nodes) kidsOf[nd[0]] = {nd[1], nd[2]};
        size_t before = X.size();
        for (auto& nd : nodes) {
            int p = nd[0];
            int ch[2] = {nd[1], nd[2]};
            for (int k = 0; k < 2; k++) {
                int v = ch[k], c = ch[1 - k];
                auto it = kidsOf.find(v);
                if (it == kidsOf.end()) continue;
                addCluster(X[it->second.first] | X[c]);
                addCluster(X[it->second.second] | X[c]);
            }
            (void)p;
        }
        int SPRR = getenv("STREE_SPRR") ? atoi(getenv("STREE_SPRR")) : 3;
        if (SPRR > 0) sprClusters(nodes, root, SPRR);
        if (X.size() == before) break;
    }
    fprintf(stderr, "done score %.6g time %.2fs\n", bestScore, elapsed());
    return 0;
}
