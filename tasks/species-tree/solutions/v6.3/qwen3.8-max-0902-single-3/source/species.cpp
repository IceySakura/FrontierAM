// Species tree inference from gene trees under the multi-species coalescent.
//
// Pipeline:
//   1. Parse the 50 estimated gene trees (Newick, unrooted, optional support labels).
//   2. ASTRID-style step: average internode (edge-count) distances over gene trees,
//      build a starting species tree with neighbour joining.
//   3. Optional NNI hill climbing on a weighted-quartet objective, disabled by
//      default (ST_BUDGET=0) -- see README "Known limitation".
//   4. Optional randomised NNI-perturbation restarts to escape local optima.
//
// Single threaded, deterministic given SEED.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

// ------------------------------------------------------------------ options
static int    OPT_WEIGHT = 1;      // 0 none, 1 support, 2 max(0,(s-1/3)*1.5), 3 s^2, 4 (s>0.5)
static double OPT_BUDGET = 0.0;   // seconds of extra hill climbing (0 = ASTRID/NJ tree only)
static int    OPT_RESTARTS = 0;
static int    OPT_VERBOSE = 0;

static double wall0;
static double now() {
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}
static double deadline_ = 1e18;
static inline bool outOfTime() { return now() > deadline_; }

// ------------------------------------------------------------------ Newick
struct PNode {
    vector<int> kids;
    int tax = -1;
    double bl = 0.0;
    double sup = -1.0;
};

struct Parser {
    const string &s;
    size_t p = 0;
    vector<PNode> &nd;
    unordered_map<string,int> &tmap;
    vector<string> &tnames;
    Parser(const string &str, vector<PNode> &n, unordered_map<string,int> &m, vector<string> &t)
        : s(str), nd(n), tmap(m), tnames(t) {}

    void skipWS() { while (p < s.size() && isspace((unsigned char)s[p])) ++p; }
    void skipNoise() {
        for (;;) {
            skipWS();
            if (p < s.size() && s[p] == '[') {
                int d = 0;
                while (p < s.size()) {
                    char c = s[p++];
                    if (c == '[') ++d;
                    else if (c == ']') { if (--d == 0) break; }
                }
            } else break;
        }
    }
    int newNode() { nd.push_back(PNode()); return (int)nd.size() - 1; }

    string readLabel() {
        string out;
        if (p < s.size() && s[p] == '\'') {
            ++p;
            while (p < s.size()) {
                if (s[p] == '\'') {
                    if (p + 1 < s.size() && s[p+1] == '\'') { out += '\''; p += 2; continue; }
                    ++p; break;
                }
                out += s[p++];
            }
        } else {
            while (p < s.size()) {
                char c = s[p];
                if (c==',' || c==':' || c==')' || c=='(' || c==';' || isspace((unsigned char)c)) break;
                out += c; ++p;
            }
        }
        return out;
    }

    int parseSub() {
        skipNoise();
        int id = newNode();
        if (p < s.size() && s[p] == '(') {
            ++p;
            for (;;) {
                int k = parseSub();
                nd[id].kids.push_back(k);
                skipNoise();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                break;
            }
            skipNoise();
            if (p < s.size() && s[p] == ')') ++p;
            skipNoise();
            string lab = readLabel();
            if (!lab.empty()) nd[id].sup = atof(lab.c_str());
        } else {
            string lab = readLabel();
            auto it = tmap.find(lab);
            if (it == tmap.end()) {
                tmap[lab] = (int)tnames.size();
                tnames.push_back(lab);
                it = tmap.find(lab);
            }
            nd[id].tax = it->second;
        }
        skipNoise();
        if (p < s.size() && s[p] == ':') {
            ++p; skipNoise();
            nd[id].bl = atof(readLabel().c_str());
        }
        return id;
    }
};

// ------------------------------------------------------------------ gene tree
struct GTree {
    int nn = 0, root = -1;
    vector<int> par, c0, c1, c2, tax, post;
    vector<double> w;
    vector<int> ca, cb;
};

static double edgeWeight(double sup) {
    switch (OPT_WEIGHT) {
        case 0: return 1.0;
        case 1: return sup < 0 ? 1.0 : max(sup, 0.0);
        case 2: return sup < 0 ? 1.0 : max(0.0, (sup - 1.0/3.0) * 1.5);
        case 3: return sup < 0 ? 1.0 : sup * sup;
        case 4: return sup < 0 ? 1.0 : (sup >= 0.5 ? 1.0 : 0.0);
        default: return 1.0;
    }
}

// Turn a parsed Newick tree into a rooted-at-trifurcation structure.
static bool buildGTree(const string &line, GTree &g, unordered_map<string,int> &tmap,
                       vector<string> &tnames, bool allowNewTaxa)
{
    vector<PNode> nd;
    nd.reserve(600);
    Parser ps(line, nd, tmap, tnames);
    ps.skipNoise();
    if (ps.p >= line.size() || line[ps.p] != '(') return false;
    int root = ps.parseSub();

    // Ensure the root is a trifurcation (unrooted binary tree semantics).
    if ((int)nd[root].kids.size() == 2) {
        int x = nd[root].kids[0], y = nd[root].kids[1];
        int nr = -1, other = -1;
        if (!nd[x].kids.empty()) { nr = x; other = y; }
        else if (!nd[y].kids.empty()) { nr = y; other = x; }
        else return false;
        vector<int> kk = nd[nr].kids;
        nd[nr].kids.clear();
        nd[nr].kids.push_back(kk[0]);
        nd[nr].kids.push_back(kk[1]);
        nd[nr].kids.push_back(other);
        root = nr;
    } else if ((int)nd[root].kids.size() != 3) {
        // Multifurcation in the input: keep as is but only the first three matter
        // for a binary assumption; handle generically by chaining.
        if (nd[root].kids.size() < 2) return false;
        while (nd[root].kids.size() > 3) {
            int a = nd[root].kids.back(); nd[root].kids.pop_back();
            int b = nd[root].kids.back(); nd[root].kids.pop_back();
            int h = (int)nd.size(); nd.push_back(PNode());
            nd[h].kids.push_back(a); nd[h].kids.push_back(b);
            nd[root].kids.push_back(h);
        }
    }
    (void)allowNewTaxa;

    int nn = (int)nd.size();
    g.nn = nn; g.root = root;
    g.par.assign(nn, -1); g.c0.assign(nn, -1); g.c1.assign(nn, -1); g.c2.assign(nn, -1);
    g.tax.assign(nn, -1); g.w.assign(nn, 0.0);
    g.post.clear(); g.post.reserve(nn);
    g.ca.assign(nn, 0); g.cb.assign(nn, 0);

    // iterative pre-order to set parents/children, then post-order
    vector<int> order; order.reserve(nn);
    vector<int> stk; stk.push_back(root);
    g.par[root] = -1;
    while (!stk.empty()) {
        int v = stk.back(); stk.pop_back();
        order.push_back(v);
        const vector<int> &k = nd[v].kids;
        if (!k.empty()) {
            g.c0[v] = k[0]; g.par[k[0]] = v; stk.push_back(k[0]);
            if (k.size() > 1) { g.c1[v] = k[1]; g.par[k[1]] = v; stk.push_back(k[1]); }
            if (k.size() > 2) { g.c2[v] = k[2]; g.par[k[2]] = v; stk.push_back(k[2]); }
        }
        g.tax[v] = nd[v].tax;
    }
    for (int i = (int)order.size() - 1; i >= 0; --i) {
        int v = order[i];
        if (v == root) continue;            // root has no parent edge
        g.w[v] = edgeWeight(nd[v].sup);
        g.post.push_back(v);
    }
    // reverse pre-order == valid post-order (every node after its descendants)
    return true;
}

// ------------------------------------------------------------------ globals
static int n = 0;                 // number of taxa
static vector<string> taxNames;
static vector<GTree> GT;
static int NN = 0;                // nodes of the search tree (2n-2)
static vector<array<int,3>> nb;   // neighbours in the search tree
static vector<double> sc;         // cached split scores, NN*NN
static vector<int>    sv;         // cache validity stamp
static int stampGen = 0;
static vector<unsigned char> col;
static vector<int> stkN, stkP;

static inline int SIDX(int a, int b) { return a * NN + b; }

// Score of the bipartition induced by removing edge (u,v): component of v vs rest.
static double computeSplitScore(int u, int v) {
    fill(col.begin(), col.end(), (unsigned char)0);
    stkN.clear(); stkP.clear();
    stkN.push_back(v); stkP.push_back(u);
    int nL = 0;
    while (!stkN.empty()) {
        int x = stkN.back(), px = stkP.back();
        stkN.pop_back(); stkP.pop_back();
        if (x < n) { col[x] = 1; ++nL; }
        for (int k = 0; k < 3; ++k) {
            int y = nb[x][k];
            if (y >= 0 && y != px) { stkN.push_back(y); stkP.push_back(x); }
        }
    }
    if (nL < 1) return 0.0;
    const int nR = n - nL;
    double total = 0.0;
    for (size_t gi = 0; gi < GT.size(); ++gi) {
        GTree &g = GT[gi];
        const vector<int> &post = g.post;
        for (size_t pi = 0; pi < post.size(); ++pi) {
            int t = post[pi];
            int a, b;
            int tx = g.tax[t];
            if (tx >= 0) { a = col[tx]; b = 1 - a; }
            else {
                int x1 = g.c0[t], x2 = g.c1[t], x3 = g.c2[t];
                a = g.ca[x1] + g.ca[x2];
                b = g.cb[x1] + g.cb[x2];
                if (x3 >= 0) { a += g.ca[x3]; b += g.cb[x3]; }
            }
            g.ca[t] = a; g.cb[t] = b;
            int c = nL - a, d = nR - b;
            if (a < 2 && b < 2) continue;
            double t1 = 0.0, t2 = 0.0;
            if (a >= 2 && d >= 2) t1 = (double)(a * (a - 1) / 2) * (double)(d * (d - 1) / 2);
            if (c >= 2 && b >= 2) t2 = (double)(c * (c - 1) / 2) * (double)(b * (b - 1) / 2);
            if (t1 != 0.0 || t2 != 0.0) total += g.w[t] * (t1 + t2);
        }
    }
    return total;
}

static inline double getScore(int u, int v) {
    int i = SIDX(u, v);
    if (sv[i] != stampGen) {
        double s = computeSplitScore(u, v);
        sc[i] = s; sc[SIDX(v, u)] = s;
        sv[i] = stampGen; sv[SIDX(v, u)] = stampGen;
    }
    return sc[i];
}
static inline void invalidate(int u, int v) { sv[SIDX(u,v)] = -1; sv[SIDX(v,u)] = -1; }

static double fullScore() {
    double s = 0.0;
    for (int u = n; u < NN; ++u)
        for (int k = 0; k < 3; ++k) {
            int v = nb[u][k];
            if (v > u && v >= n) s += getScore(u, v);
        }
    return s;
}

// ------------------------------------------------------------------ tree ops
static inline void replaceNb(int x, int oldv, int newv) {
    for (int k = 0; k < 3; ++k) if (nb[x][k] == oldv) { nb[x][k] = newv; return; }
}

struct NNIMove { int u, v, a, c; double delta; };

// Apply / revert swap of neighbour a (at u) with c (at v) across edge (u,v).
static void applySwap(int u, int v, int a, int c) {
    replaceNb(u, a, c);
    replaceNb(v, c, a);
    replaceNb(a, u, v);
    replaceNb(c, v, u);
    static const int P[4] = {0,0,0,0}; (void)P;
    int q[4] = {u, v, a, c};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (i != j) invalidate(q[i], q[j]);
}

// Enumerate internal edges
static void internalEdges(vector<pair<int,int>> &out) {
    out.clear();
    for (int u = n; u < NN; ++u)
        for (int k = 0; k < 3; ++k) {
            int v = nb[u][k];
            if (v > u && v >= n) out.push_back({u, v});
        }
}

// ------------------------------------------------------------------ NJ
struct NJTree { vector<array<int,3>> adj; int nodes; };

static bool neighborJoining(const vector<double> &D0, vector<array<int,3>> &adjOut) {
    int m = n;
    vector<double> D = D0;                     // m x m
    vector<char> active(m, 1);
    vector<int> act(m); for (int i = 0; i < m; ++i) act[i] = i;
    vector<int> nodeId(m); for (int i = 0; i < m; ++i) nodeId[i] = i;
    int total = 2 * m - 2;
    vector<array<int,3>> adj(total);
    for (int i = 0; i < total; ++i) adj[i] = {-1,-1,-1};
    int nextNode = m;
    vector<double> rowsum(m, 0.0);
    int nact = m;

    auto addAdj = [&](int x, int y) {
        for (int k = 0; k < 3; ++k) if (adj[x][k] == -1) { adj[x][k] = y; return; }
    };

    while (nact > 3) {
        int r = nact;
        for (int ii = 0; ii < r; ++ii) {
            int i = act[ii];
            double s = 0.0;
            for (int jj = 0; jj < r; ++jj) { int j = act[jj]; if (i != j) s += D[i*m+j]; }
            rowsum[i] = s;
        }
        double bestQ = 1e300; int bi = -1, bj = -1;
        double inv = 1.0 / (r - 2);
        for (int ii = 0; ii < r; ++ii) {
            int i = act[ii];
            double ui = rowsum[i] * inv;
            for (int jj = ii + 1; jj < r; ++jj) {
                int j = act[jj];
                double q = D[i*m+j] - ui - rowsum[j] * inv;
                if (q < bestQ) { bestQ = q; bi = i; bj = j; }
            }
        }
        if (bi < 0) return false;
        int u = nextNode++;
        double dij = D[bi*m+bj];
        double dui = dij * 0.5 + (rowsum[bi] - rowsum[bj]) * inv * 0.5;
        addAdj(u, nodeId[bi]); addAdj(nodeId[bi], u);
        addAdj(u, nodeId[bj]); addAdj(nodeId[bj], u);
        // update distances
        for (int kk = 0; kk < r; ++kk) {
            int k = act[kk];
            if (k == bi || k == bj) continue;
            double nd = 0.5 * (D[bi*m+k] + D[bj*m+k] - dij);
            D[bi*m+k] = nd; D[k*m+bi] = nd;
        }
        nodeId[bi] = u;
        // deactivate bj
        vector<int> na; na.reserve(r - 1);
        for (int kk = 0; kk < r; ++kk) if (act[kk] != bj) na.push_back(act[kk]);
        act.swap(na);
        active[bj] = 0;
        nact = r - 1;
    }
    if (nact != 3) return false;
    int rt = nextNode++;
    (void)rt;
    int root = 2 * m - 3;      // last node index
    addAdj(root, nodeId[act[0]]); addAdj(nodeId[act[0]], root);
    addAdj(root, nodeId[act[1]]); addAdj(nodeId[act[1]], root);
    addAdj(root, nodeId[act[2]]); addAdj(nodeId[act[2]], root);
    adjOut = adj;
    return true;
}

// ------------------------------------------------------------------ search tree from adjacency
static void loadTree(const vector<array<int,3>> &adj) {
    NN = 2 * n - 2;
    nb.assign(NN, {-1,-1,-1});
    for (int v = 0; v < NN; ++v) nb[v] = adj[v];
    sc.assign(NN * NN, 0.0);
    sv.assign(NN * NN, -1);
    stampGen = 1;
}
static void bumpStamp() { ++stampGen; }   // cheap full invalidation

// ------------------------------------------------------------------ NNI hill climb
static double nniClimb() {
    vector<pair<int,int>> edges;
    double cur = fullScore();
    while (!outOfTime()) {
        internalEdges(edges);
        double bestDelta = 1e-6;
        int bu = -1, bv = -1, ba = -1, bc = -1;
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            int u = edges[ei].first, v = edges[ei].second;
            double s0 = getScore(u, v);
            int A[2], C[2], na = 0, nc = 0;
            for (int k = 0; k < 3; ++k) { int x = nb[u][k]; if (x != v) A[na++] = x; }
            for (int k = 0; k < 3; ++k) { int x = nb[v][k]; if (x != u) C[nc++] = x; }
            if (na != 2 || nc != 2) continue;
            for (int t = 0; t < 2; ++t) {
                int a = A[0], c = C[t];
                applySwap(u, v, a, c);
                double s1 = getScore(u, v);
                applySwap(u, v, c, a);   // revert (swap back)
                double d = s1 - s0;
                if (d > bestDelta) { bestDelta = d; bu = u; bv = v; ba = a; bc = c; }
            }
        }
        if (bu < 0) break;
        applySwap(bu, bv, ba, bc);
        cur += bestDelta;
    }
    return cur;
}

// ------------------------------------------------------------------ output
static string outBuf;
static void emitSub(int v, int parent) {
    if (v < n) { outBuf += taxNames[v]; return; }
    outBuf += "(";
    bool first = true;
    for (int k = 0; k < 3; ++k) {
        int w = nb[v][k];
        if (w < 0 || w == parent) continue;
        if (!first) outBuf += ",";
        first = false;
        emitSub(w, v);
    }
    outBuf += ")";
}
static void writeNewick(const char *path) {
    int root = -1;
    for (int v = n; v < NN; ++v) {
        int cnt = 0; for (int k = 0; k < 3; ++k) if (nb[v][k] >= 0) ++cnt;
        if (cnt == 3) { root = v; break; }
    }
    if (root < 0) root = n;
    outBuf.clear(); outBuf.reserve(16384);
    outBuf += "(";
    bool first = true;
    for (int k = 0; k < 3; ++k) {
        int w = nb[root][k];
        if (w < 0) continue;
        if (!first) outBuf += ",";
        first = false;
        emitSub(w, root);
    }
    outBuf += ");";
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "cannot open output %s\n", path); exit(2); }
    fwrite(outBuf.data(), 1, outBuf.size(), fp);
    fclose(fp);
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv) {
    wall0 = now();
    if (argc < 3) { fprintf(stderr, "usage: %s input output\n", argv[0]); return 2; }
    const char *env;
    if ((env = getenv("ST_WEIGHT")))   OPT_WEIGHT   = atoi(env);
    if ((env = getenv("ST_BUDGET")))   OPT_BUDGET   = atof(env);
    if ((env = getenv("ST_RESTARTS"))) OPT_RESTARTS = atoi(env);
    if ((env = getenv("ST_VERBOSE")))  OPT_VERBOSE  = atoi(env);
    unsigned seed = 1;
    if ((env = getenv("SEED"))) seed = (unsigned)atoi(env);
    deadline_ = wall0 + OPT_BUDGET;

    // ---- read input -------------------------------------------------------
    FILE *fp = fopen(argv[1], "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    vector<string> lines;
    {
        string buf; buf.reserve(1 << 20);
        int c;
        while ((c = fgetc(fp)) != EOF) {
            if (c == '\n') { if (!buf.empty()) { lines.push_back(buf); buf.clear(); } }
            else buf.push_back((char)c);
        }
        if (!buf.empty()) lines.push_back(buf);
    }
    fclose(fp);
    if (lines.empty()) { fprintf(stderr, "empty input\n"); return 2; }

    unordered_map<string,int> tmap;
    taxNames.reserve(300);
    GT.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        string L = lines[i];
        size_t a = L.find_first_not_of(" \t\r\n");
        if (a == string::npos) continue;
        size_t b = L.find_last_not_of(" \t\r\n");
        L = L.substr(a, b - a + 1);
        if (L.empty()) continue;
        GTree g;
        if (!buildGTree(L, g, tmap, taxNames, i == 0)) { fprintf(stderr, "parse fail line %zu\n", i); continue; }
        GT.push_back(std::move(g));
    }
    n = (int)taxNames.size();
    if (n < 4 || GT.empty()) { fprintf(stderr, "bad input n=%d genes=%zu\n", n, GT.size()); return 2; }
    NN = 2 * n - 2;
    col.assign(n, 0);
    if (OPT_VERBOSE) fprintf(stderr, "taxa=%d genes=%zu parse=%.2fs\n", n, GT.size(), now() - wall0);

    // ---- average internode distances (ASTRID) -----------------------------
    vector<double> Dsum((size_t)n * n, 0.0);
    {
        vector<int> dist;
        vector<array<int,3>> gadj;
        vector<int> q;
        for (size_t gi = 0; gi < GT.size(); ++gi) {
            GTree &g = GT[gi];
            gadj.assign(g.nn, {-1,-1,-1});
            for (int v = 0; v < g.nn; ++v) {
                int slot = 0;
                if (g.c0[v] >= 0) gadj[v][slot++] = g.c0[v];
                if (g.c1[v] >= 0) gadj[v][slot++] = g.c1[v];
                if (g.c2[v] >= 0) gadj[v][slot++] = g.c2[v];
                if (g.par[v] >= 0 && slot < 3) gadj[v][slot++] = g.par[v];
            }
            dist.assign(g.nn, -1);
            for (int src = 0; src < g.nn; ++src) {
                if (g.tax[src] < 0) continue;
                fill(dist.begin(), dist.end(), -1);
                q.clear(); q.push_back(src); dist[src] = 0;
                for (size_t h = 0; h < q.size(); ++h) {
                    int v = q[h];
                    for (int k = 0; k < 3; ++k) {
                        int w = gadj[v][k];
                        if (w >= 0 && dist[w] < 0) { dist[w] = dist[v] + 1; q.push_back(w); }
                    }
                }
                double *row = &Dsum[(size_t)g.tax[src] * n];
                for (int v = 0; v < g.nn; ++v)
                    if (g.tax[v] >= 0 && v != src) row[g.tax[v]] += (double)dist[v];
            }
        }
    }
    double invG = 1.0 / (double)GT.size();
    for (size_t i = 0; i < (size_t)n * n; ++i) Dsum[i] *= invG;
    if (OPT_VERBOSE) fprintf(stderr, "distances %.2fs\n", now() - wall0);

    // ---- NJ starting tree -------------------------------------------------
    vector<array<int,3>> adj;
    if (!neighborJoining(Dsum, adj)) { fprintf(stderr, "NJ failed\n"); return 2; }
    loadTree(adj);
    double bestScore = fullScore();
    vector<array<int,3>> bestNb = nb;
    if (OPT_VERBOSE) fprintf(stderr, "NJ score=%.0f at %.2fs\n", bestScore, now() - wall0);

    // ---- NNI hill climb ---------------------------------------------------
    double cur = nniClimb();
    if (cur > bestScore + 1e-9) { bestScore = cur; bestNb = nb; }
    if (OPT_VERBOSE) fprintf(stderr, "NNI score=%.0f at %.2fs\n", cur, now() - wall0);

    // ---- perturbation restarts -------------------------------------------
    if (OPT_RESTARTS > 0) {
        mt19937 rng(seed ? seed : 1u);
        vector<pair<int,int>> edges;
        vector<array<int,3>> startNb = nb;
        for (int r = 0; r < OPT_RESTARTS && !outOfTime(); ++r) {
            nb = startNb;
            bumpStamp();
            // random NNI perturbations
            int npert = 3 + (int)(rng() % 12);
            for (int t = 0; t < npert; ++t) {
                internalEdges(edges);
                if (edges.empty()) break;
                const auto &e = edges[rng() % edges.size()];
                int u = e.first, v = e.second;
                int A[2], C[2], na = 0, nc = 0;
                for (int k = 0; k < 3; ++k) { int x = nb[u][k]; if (x != v) A[na++] = x; }
                for (int k = 0; k < 3; ++k) { int x = nb[v][k]; if (x != u) C[nc++] = x; }
                if (na == 2 && nc == 2) applySwap(u, v, A[0], C[rng() & 1]);
            }
            bumpStamp();
            double s = nniClimb();
            if (OPT_VERBOSE) fprintf(stderr, "restart %d score=%.0f (best %.0f) %.2fs\n", r, s, bestScore, now() - wall0);
            if (s > bestScore + 1e-9) { bestScore = s; bestNb = nb; startNb = nb; }
        }
    }

    nb = bestNb;
    bumpStamp();
    double fs = fullScore();
    if (OPT_VERBOSE) fprintf(stderr, "final score=%.0f at %.2fs\n", fs, now() - wall0);
    writeNewick(argv[2]);
    return 0;
}
