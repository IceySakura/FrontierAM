// Weighted-quartet species tree inference (ASTRAL-style constrained dynamic
// programming with support-weighted quartets and search-space expansion).
#include <bits/stdc++.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;
typedef uint64_t u64;
static const int W = 4;  // up to 256 taxa

static double T0;
static double now() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}
static double elapsed() { return now() - T0; }

struct BS {
    u64 w[W];
    BS() { memset(w, 0, sizeof w); }
    void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    int count() const { int c = 0; for (int i = 0; i < W; i++) c += __builtin_popcountll(w[i]); return c; }
    bool operator==(const BS& o) const { for (int i = 0; i < W; i++) if (w[i] != o.w[i]) return false; return true; }
    bool subsetOf(const BS& o) const { for (int i = 0; i < W; i++) if (w[i] & ~o.w[i]) return false; return true; }
    BS operator|(const BS& o) const { BS r; for (int i = 0; i < W; i++) r.w[i] = w[i] | o.w[i]; return r; }
    BS minus(const BS& o) const { BS r; for (int i = 0; i < W; i++) r.w[i] = w[i] & ~o.w[i]; return r; }
};
struct BSH {
    size_t operator()(const BS& b) const {
        u64 h = 1469598103934665603ULL;
        for (int i = 0; i < W; i++) {
            u64 x = b.w[i] + 0x9e3779b97f4a7c15ULL * (i + 1);
            x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
            h ^= x; h *= 0x100000001b3ULL;
        }
        return h;
    }
};

// ---------------------------------------------------------------- parsing
struct PN { int par = -1; vector<int> ch; int leaf = -1; double sup = -1; double len = 0; };
static vector<string> names;
static unordered_map<string, int> nameId;

static bool parseNewick(const string& s, vector<PN>& nodes) {
    nodes.clear();
    size_t i = 0, n = s.size();
    int cur = -1, last = -1;
    bool justClosed = false;
    while (i < n) {
        char c = s[i];
        if (isspace((unsigned char)c)) { i++; continue; }
        if (c == '[') { while (i < n && s[i] != ']') i++; i++; continue; }
        if (c == '(') {
            PN nd; nd.par = cur; nodes.push_back(nd);
            int id = nodes.size() - 1;
            if (cur >= 0) nodes[cur].ch.push_back(id);
            cur = id; last = -1; justClosed = false; i++;
        } else if (c == ',') {
            i++; last = -1; justClosed = false;
        } else if (c == ')') {
            if (cur < 0) return false;
            last = cur; cur = nodes[cur].par; justClosed = true; i++;
        } else if (c == ':') {
            i++;
            size_t j = i;
            while (j < n && (isdigit((unsigned char)s[j]) || s[j] == '.' || s[j] == 'e' || s[j] == 'E' || s[j] == '-' || s[j] == '+')) j++;
            if (last >= 0) nodes[last].len = atof(s.substr(i, j - i).c_str());
            i = j; justClosed = false;
        } else if (c == ';') {
            break;
        } else {
            string lab;
            if (c == '\'') {
                i++;
                while (i < n) {
                    if (s[i] == '\'') { if (i + 1 < n && s[i + 1] == '\'') { lab += '\''; i += 2; continue; } i++; break; }
                    lab += s[i++];
                }
            } else {
                while (i < n && !strchr("(),:;[", s[i]) && !isspace((unsigned char)s[i])) lab += s[i++];
            }
            if (justClosed && last >= 0) {
                char* e; double v = strtod(lab.c_str(), &e);
                if (e != lab.c_str() && *e == 0) nodes[last].sup = v;
                justClosed = false;
            } else {
                PN nd; nd.par = cur; nodes.push_back(nd);
                int id = nodes.size() - 1;
                if (cur >= 0) nodes[cur].ch.push_back(id);
                auto it = nameId.find(lab);
                int tid;
                if (it == nameId.end()) { tid = names.size(); nameId[lab] = tid; names.push_back(lab); }
                else tid = it->second;
                nodes[id].leaf = tid;
                last = id; justClosed = false;
            }
        }
    }
    return !nodes.empty();
}

// ---------------------------------------------------------------- gene trees
static int NT;     // number of taxa
static int ROOT;   // taxon used as root anchor
struct Gene {
    int nI;                 // internal nodes, post-order, top = nI-1
    vector<int> c1, c2;     // >=0 internal, <0 leaf -(taxon+1)
    vector<float> sp;       // normalized support of edge to parent
    vector<float> ln;       // branch length of edge to parent
    vector<BS> clus;        // leaf set below each internal node
    vector<float> l1, l2;   // lengths of edges to children
    vector<double> sz, wpre; // subtree size; path-weighted pair sum if monochromatic
    vector<double> d1, d2;  // decay factors of edges to children
    double topLen = 0, dTop = 1;
};
static double BETA = 0.0, GAMMA = 1.0;
static vector<Gene> genes;

struct UEdge { int to; double sup; double len; };

static bool buildGene(const vector<PN>& pn, Gene& g, double smin) {
    int m = pn.size();
    vector<vector<UEdge>> adj(m);
    for (int i = 0; i < m; i++) if (pn[i].par >= 0) {
        adj[i].push_back({pn[i].par, pn[i].sup, pn[i].len});
        adj[pn[i].par].push_back({i, pn[i].sup, pn[i].len});
    }
    // suppress degree-2 internal nodes
    for (int i = 0; i < m; i++) if (pn[i].leaf < 0 && adj[i].size() == 2) {
        UEdge a = adj[i][0], b = adj[i][1];
        double s = max(a.sup, b.sup);
        double l = a.len + b.len;
        for (auto& e : adj[a.to]) if (e.to == i) { e.to = b.to; e.sup = s; e.len = l; }
        for (auto& e : adj[b.to]) if (e.to == i) { e.to = a.to; e.sup = s; e.len = l; }
        adj[i].clear();
    }
    int rnode = -1;
    for (int i = 0; i < m; i++) if (pn[i].leaf == ROOT) rnode = i;
    if (rnode < 0 || adj[rnode].size() != 1) return false;
    g.c1.clear(); g.c2.clear(); g.sp.clear(); g.ln.clear();
    // iterative DFS producing post-order of internal nodes
    // returns code for node
    struct Fr { int node, parent, idx; vector<int> kids; vector<double> kl; double sup, len; };
    g.l1.clear(); g.l2.clear();
    int top = adj[rnode][0].to;
    if (pn[top].leaf >= 0) return false;
    vector<Fr> st;
    st.push_back({top, rnode, 0, {}, {}, -1, adj[rnode][0].len});
    g.topLen = adj[rnode][0].len;
    int retCode = 0;
    double retLen = 0;
    bool haveRet = false;
    while (!st.empty()) {
        Fr& f = st.back();
        if (haveRet) { f.kids.push_back(retCode); f.kl.push_back(retLen); haveRet = false; }
        // find next neighbor
        bool pushed = false;
        while (f.idx < (int)adj[f.node].size()) {
            UEdge e = adj[f.node][f.idx++];
            if (e.to == f.parent) continue;
            if (pn[e.to].leaf >= 0) { f.kids.push_back(-(pn[e.to].leaf + 1)); f.kl.push_back(e.len); continue; }
            st.push_back({e.to, f.node, 0, {}, {}, e.sup, e.len});
            pushed = true;
            break;
        }
        if (pushed) continue;
        // finalize node: resolve polytomy as caterpillar with zero-support edges
        Fr fr = st.back(); st.pop_back();
        vector<int> k = fr.kids;
        if (k.size() < 2) return false;
        int code = k[0];
        double clen = fr.kl[0];
        for (size_t q = 1; q < k.size(); q++) {
            g.c1.push_back(code); g.c2.push_back(k[q]);
            g.l1.push_back(clen); g.l2.push_back(fr.kl[q]);
            clen = 0;
            bool last = (q + 1 == k.size());
            double sv = last ? fr.sup : -1;
            double s = 0;
            if (sv >= 0) s = (sv - smin) / (1.0 - smin);
            if (s < 0) s = 0; if (s > 1) s = 1;
            if (GAMMA != 1.0) s = pow(s, GAMMA);
            g.sp.push_back(s);
            g.ln.push_back(last ? fr.len : 0.0);
            code = g.c1.size() - 1;
        }
        retCode = code; retLen = fr.len; haveRet = true;
    }
    g.nI = g.c1.size();
    g.sp[g.nI - 1] = 0;
    g.d1.resize(g.nI); g.d2.resize(g.nI);
    for (int j = 0; j < g.nI; j++) { g.d1[j] = exp(-BETA * g.l1[j]); g.d2[j] = exp(-BETA * g.l2[j]); }
    g.dTop = exp(-BETA * g.topLen);
    g.sz.assign(g.nI, 0); g.wpre.assign(g.nI, 0);
    for (int j = 0; j < g.nI; j++) {
        int a = g.c1[j], c = g.c2[j];
        double sa = a >= 0 ? g.sz[a] : 1, sc = c >= 0 ? g.sz[c] : 1;
        double wa = a >= 0 ? g.wpre[a] : 0, wc = c >= 0 ? g.wpre[c] : 0;
        g.sz[j] = sa + sc;
        g.wpre[j] = (1.0 - g.sp[j]) * (sa * sc + wa + wc);
    }
    // clusters
    g.clus.assign(g.nI, BS());
    for (int j = 0; j < g.nI; j++) {
        BS b;
        int a = g.c1[j], c = g.c2[j];
        if (a >= 0) b = b | g.clus[a]; else b.set(-a - 1);
        if (c >= 0) b = b | g.clus[c]; else b.set(-c - 1);
        g.clus[j] = b;
    }
    return true;
}

// ---------------------------------------------------------------- cluster store
static vector<BS> X;
static vector<int> Xsz;
static unordered_map<BS, int, BSH> Xid;
static vector<uint8_t> CNT;  // counts per cluster per gene edge
static int E;                // number of scored gene edges
static vector<double> PHI, SZE;
static vector<int> edgeGene, edgeNode;
static int MODE = 2;         // 0 unweighted node-based, 1 additive support, 2 noisy-or support

static void computeCounts(const BS& b, uint8_t* out) {
    static vector<int> tmp;
    int e = 0;
    for (auto& g : genes) {
        tmp.resize(g.nI);
        for (int j = 0; j < g.nI; j++) {
            int a = g.c1[j], c = g.c2[j];
            int v = (a >= 0 ? tmp[a] : b.test(-a - 1)) + (c >= 0 ? tmp[c] : b.test(-c - 1));
            tmp[j] = v;
            if (j < g.nI - 1) out[e++] = v;
        }
    }
}

static int addCluster(const BS& b) {
    auto it = Xid.find(b);
    if (it != Xid.end()) return it->second;
    int id = X.size();
    X.push_back(b); Xsz.push_back(b.count()); Xid[b] = id;
    if (MODE == 1) {
        CNT.resize((size_t)(id + 1) * E);
        computeCounts(b, &CNT[(size_t)id * E]);
    }
    return id;
}

// ---------------------------------------------------------------- scoring
static double scoreAdd(int i1, int ia, double n1, double n2, double nc) {
    const uint8_t* p1 = &CNT[(size_t)i1 * E];
    const uint8_t* pa = &CNT[(size_t)ia * E];
    const double* phi = PHI.data();
    const double* sz = SZE.data();
    double acc = 0;
    for (int e = 0; e < E; e++) {
        double a1 = p1[e], a = pa[e], a2 = a - a1, c = sz[e] - a;
        double a1o = n1 - a1, a2o = n2 - a2, co = nc - c;
        double F = a1 * (a1 - 1) * a2o * co + a2 * (a2 - 1) * a1o * co + c * (c - 1) * a1o * a2o
                 + a1o * (a1o - 1) * a2 * c + a2o * (a2o - 1) * a1 * c + co * (co - 1) * a1 * a2;
        acc += phi[e] * F;
    }
    return 0.5 * acc;
}

// node-based: weighted = N - M where M uses path products of t = 1 - s;
// leaf-to-endpoint decays exp(-BETA * length) multiply quartet weights.
static vector<int> colorBuf;
static double scoreNode(const BS& A1, const BS& A2, bool weighted, float* pg) {
    colorBuf.resize(NT);
    for (int i = 0; i < NT; i++) colorBuf[i] = A1.test(i) ? 0 : (A2.test(i) ? 1 : 2);
    static vector<array<double, 3>> S;
    static vector<array<double, 6>> Nv, Mv;
    static vector<array<double, 3>> U;
    double SN = 0, SM = 0;
    int gi = 0;
    for (auto& g : genes) {
        double SN0 = SN, SM0 = SM;
        int nI = g.nI;
        S.resize(nI); Nv.resize(nI); Mv.resize(nI); U.resize(nI);
        // bottom-up
        for (int j = 0; j < nI; j++) {
            double x1[3] = {0, 0, 0}, x2[3] = {0, 0, 0};
            double n1[6] = {0, 0, 0, 0, 0, 0}, n2[6] = {0, 0, 0, 0, 0, 0}, m1[6] = {0, 0, 0, 0, 0, 0}, m2[6] = {0, 0, 0, 0, 0, 0};
            int a = g.c1[j], b = g.c2[j];
            double da = g.d1[j], db = g.d2[j];
            if (a >= 0) { for (int q = 0; q < 3; q++) x1[q] = S[a][q] * da; for (int q = 0; q < 6; q++) { n1[q] = Nv[a][q]; m1[q] = Mv[a][q]; } }
            else x1[colorBuf[-a - 1]] = da;
            if (b >= 0) { for (int q = 0; q < 3; q++) x2[q] = S[b][q] * db; for (int q = 0; q < 6; q++) { n2[q] = Nv[b][q]; m2[q] = Mv[b][q]; } }
            else x2[colorBuf[-b - 1]] = db;
            double P[6];
            P[0] = x1[0] * x2[0]; P[1] = x1[1] * x2[1]; P[2] = x1[2] * x2[2];
            P[3] = x1[1] * x2[2] + x1[2] * x2[1]; P[4] = x1[0] * x2[2] + x1[2] * x2[0]; P[5] = x1[0] * x2[1] + x1[1] * x2[0];
            for (int q = 0; q < 3; q++) S[j][q] = x1[q] + x2[q];
            double t = 1.0 - g.sp[j];
            for (int q = 0; q < 6; q++) { Nv[j][q] = P[q] + n1[q] + n2[q]; Mv[j][q] = weighted ? t * (P[q] + m1[q] + m2[q]) : 0.0; }
        }
        // top-down: up counts and contributions
        int top = nI - 1;
        for (int q = 0; q < 3; q++) U[top][q] = 0;
        U[top][colorBuf[ROOT]] = g.dTop;
        for (int j = nI - 1; j >= 0; j--) {
            double x1[3] = {0, 0, 0}, x2[3] = {0, 0, 0};
            const double* n1 = nullptr; const double* n2 = nullptr; const double* m1 = nullptr; const double* m2 = nullptr;
            static const double Z[6] = {0, 0, 0, 0, 0, 0};
            int a = g.c1[j], b = g.c2[j];
            double da = g.d1[j], db = g.d2[j];
            if (a >= 0) { for (int q = 0; q < 3; q++) x1[q] = S[a][q] * da; n1 = Nv[a].data(); m1 = Mv[a].data(); }
            else { x1[colorBuf[-a - 1]] = da; n1 = Z; m1 = Z; }
            if (b >= 0) { for (int q = 0; q < 3; q++) x2[q] = S[b][q] * db; n2 = Nv[b].data(); m2 = Mv[b].data(); }
            else { x2[colorBuf[-b - 1]] = db; n2 = Z; m2 = Z; }
            const double* u = U[j].data();
            if (a >= 0) for (int q = 0; q < 3; q++) U[a][q] = da * (u[q] + x2[q]);
            if (b >= 0) for (int q = 0; q < 3; q++) U[b][q] = db * (u[q] + x1[q]);
            double v2s[3], v1s[3], v2c[3], v1c[3];
            v2s[0] = x2[1] * u[2] + x2[2] * u[1]; v2s[1] = x2[0] * u[2] + x2[2] * u[0]; v2s[2] = x2[0] * u[1] + x2[1] * u[0];
            v1s[0] = x1[1] * u[2] + x1[2] * u[1]; v1s[1] = x1[0] * u[2] + x1[2] * u[0]; v1s[2] = x1[0] * u[1] + x1[1] * u[0];
            for (int q = 0; q < 3; q++) { v2c[q] = x2[q] * u[q]; v1c[q] = x1[q] * u[q]; }
            double cN = 0, cM = 0;
            for (int P = 0; P < 3; P++) {
                cN += n1[P] * n2[3 + P] + n2[P] * n1[3 + P] + n1[P] * v2s[P] + n1[3 + P] * v2c[P] + n2[P] * v1s[P] + n2[3 + P] * v1c[P];
                cM += m1[P] * m2[3 + P] + m2[P] * m1[3 + P] + m1[P] * v2s[P] + m1[3 + P] * v2c[P] + m2[P] * v1s[P] + m2[3 + P] * v1c[P];
            }
            SN += cN; SM += cM;
        }
        pg[gi++] = (float)((SN - SN0) - (SM - SM0));
    }
    return SN - SM;
}

// fast single-pass variant for BETA == 0 (up counts = totals - subtree counts)
struct NodeVal { double x[3]; double m[6]; };
static double scoreNodeFast(const BS& A1, const BS& A2, float* pg) {
    colorBuf.resize(NT);
    double T[3] = {0, 0, 0};
    for (int i = 0; i < NT; i++) { int c = A1.test(i) ? 0 : (A2.test(i) ? 1 : 2); colorBuf[i] = c; T[c] += 1; }
    static vector<NodeVal> nv;
    static vector<int> mono;
    double tot = 0;
    int gi = 0;
    for (auto& g : genes) {
        int nI = g.nI;
        nv.resize(nI); mono.resize(nI);
        double SN = 0, SM = 0;
        for (int j = 0; j < nI; j++) {
            int a = g.c1[j], b = g.c2[j];
            int ca = a >= 0 ? mono[a] : colorBuf[-a - 1];
            int cb = b >= 0 ? mono[b] : colorBuf[-b - 1];
            if (ca == cb && ca >= 0) {
                mono[j] = ca;
                NodeVal& J = nv[j];
                J.x[0] = J.x[1] = J.x[2] = 0; J.x[ca] = g.sz[j];
                for (int q = 0; q < 6; q++) J.m[q] = 0;
                J.m[ca] = g.wpre[j];
                continue;
            }
            mono[j] = -1;
            double x1[3], x2[3], m1[6], m2[6];
            if (a >= 0) { const NodeVal& A = nv[a]; x1[0] = A.x[0]; x1[1] = A.x[1]; x1[2] = A.x[2]; for (int q = 0; q < 6; q++) m1[q] = A.m[q]; }
            else { x1[0] = x1[1] = x1[2] = 0; x1[colorBuf[-a - 1]] = 1; for (int q = 0; q < 6; q++) m1[q] = 0; }
            if (b >= 0) { const NodeVal& B = nv[b]; x2[0] = B.x[0]; x2[1] = B.x[1]; x2[2] = B.x[2]; for (int q = 0; q < 6; q++) m2[q] = B.m[q]; }
            else { x2[0] = x2[1] = x2[2] = 0; x2[colorBuf[-b - 1]] = 1; for (int q = 0; q < 6; q++) m2[q] = 0; }
            NodeVal& J = nv[j];
            double u[3];
            for (int q = 0; q < 3; q++) { J.x[q] = x1[q] + x2[q]; u[q] = T[q] - J.x[q]; }
            double n1[6], n2[6];
            for (int q = 0; q < 3; q++) { n1[q] = x1[q] * (x1[q] - 1) * 0.5; n2[q] = x2[q] * (x2[q] - 1) * 0.5; }
            n1[3] = x1[1] * x1[2]; n1[4] = x1[0] * x1[2]; n1[5] = x1[0] * x1[1];
            n2[3] = x2[1] * x2[2]; n2[4] = x2[0] * x2[2]; n2[5] = x2[0] * x2[1];
            double v2s[3], v1s[3], v2c[3], v1c[3];
            v2s[0] = x2[1] * u[2] + x2[2] * u[1]; v2s[1] = x2[0] * u[2] + x2[2] * u[0]; v2s[2] = x2[0] * u[1] + x2[1] * u[0];
            v1s[0] = x1[1] * u[2] + x1[2] * u[1]; v1s[1] = x1[0] * u[2] + x1[2] * u[0]; v1s[2] = x1[0] * u[1] + x1[1] * u[0];
            for (int q = 0; q < 3; q++) { v2c[q] = x2[q] * u[q]; v1c[q] = x1[q] * u[q]; }
            for (int P = 0; P < 3; P++) {
                SN += n1[P] * n2[3 + P] + n2[P] * n1[3 + P] + n1[P] * v2s[P] + n1[3 + P] * v2c[P] + n2[P] * v1s[P] + n2[3 + P] * v1c[P];
                SM += m1[P] * m2[3 + P] + m2[P] * m1[3 + P] + m1[P] * v2s[P] + m1[3 + P] * v2c[P] + m2[P] * v1s[P] + m2[3 + P] * v1c[P];
            }
            double t = 1.0 - g.sp[j];
            J.m[0] = t * (x1[0] * x2[0] + m1[0] + m2[0]);
            J.m[1] = t * (x1[1] * x2[1] + m1[1] + m2[1]);
            J.m[2] = t * (x1[2] * x2[2] + m1[2] + m2[2]);
            J.m[3] = t * (x1[1] * x2[2] + x1[2] * x2[1] + m1[3] + m2[3]);
            J.m[4] = t * (x1[0] * x2[2] + x1[2] * x2[0] + m1[4] + m2[4]);
            J.m[5] = t * (x1[0] * x2[1] + x1[1] * x2[0] + m1[5] + m2[5]);
        }
        pg[gi++] = (float)(SN - SM);
        tot += SN - SM;
    }
    return tot;
}

static long long nScored = 0;
static unordered_map<u64, int> scoreCache;
static vector<float> PG;          // per-gene tripartition scores
static vector<double> tripTot;
static vector<int> tripB, tripD;
static int NG;
static double triScoreIdx(int ib, int id, int* idxOut) {
    int lo = min(ib, id), hi = max(ib, id);
    u64 key = ((u64)lo << 32) | (u64)hi;
    auto it = scoreCache.find(key);
    if (it != scoreCache.end()) { *idxOut = it->second; return tripTot[it->second]; }
    nScored++;
    int t = tripTot.size();
    PG.resize((size_t)(t + 1) * NG);
    double s;
    if (MODE == 1) {
        BS A = X[ib] | X[id];
        int ia = Xid[A];
        double n1 = Xsz[ib], n2 = Xsz[id];
        s = scoreAdd(ib, ia, n1, n2, NT - n1 - n2);
        for (int g = 0; g < NG; g++) PG[(size_t)t * NG + g] = s / NG;
    } else {
        if (MODE == 2 && BETA == 0.0) s = scoreNodeFast(X[ib], X[id], &PG[(size_t)t * NG]);
        else s = scoreNode(X[ib], X[id], MODE == 2, &PG[(size_t)t * NG]);
    }
    tripTot.push_back(s); tripB.push_back(lo); tripD.push_back(hi);
    scoreCache[key] = t;
    *idxOut = t;
    return s;
}
static double triScore(int ib, int id) { int t; return triScoreIdx(ib, id, &t); }

// ---------------------------------------------------------------- DP
static vector<double> V;
static vector<pair<int, int>> bestSplit;
static int TOPID;

static vector<int> dpOrder;
static vector<vector<int>> candTrips;
static double HARD_LIMIT = 95.0;
static bool dpAborted = false;
// incremental maintenance of candidate splits (B,D) for every cluster A = B u D
static int processedUpTo = 0;
static vector<vector<int>> sizeBuckets;
static void updateCands() {
    int nx = X.size();
    candTrips.resize(nx);
    if ((int)sizeBuckets.size() < NT + 1) sizeBuckets.resize(NT + 1);
    for (int n = processedUpTo; n < nx; n++) sizeBuckets[Xsz[n]].push_back(n);
    for (int n = processedUpTo; n < nx; n++) {
        if ((n & 15) == 0 && elapsed() > HARD_LIMIT) { dpAborted = true; break; }
        const BS& N = X[n];
        int k = Xsz[n];
        // (1) N as parent
        if (k > 1) {
            for (int kb = 1; 2 * kb <= k; kb++) {
                for (int B : sizeBuckets[kb]) {
                    if (!X[B].subsetOf(N)) continue;
                    BS d = N.minus(X[B]);
                    auto it = Xid.find(d);
                    if (it == Xid.end()) continue;
                    int D = it->second;
                    if (2 * kb == k && D < B) continue;
                    int tix;
                    triScoreIdx(B, D, &tix);
                    candTrips[n].push_back(tix);
                }
            }
        }
        // (2) N as a child of an older cluster
        for (int ka = k + 1; ka < NT; ka++) {
            for (int A : sizeBuckets[ka]) {
                if (A >= processedUpTo) continue;
                if (!N.subsetOf(X[A])) continue;
                BS d = X[A].minus(N);
                auto it = Xid.find(d);
                if (it == Xid.end()) continue;
                int D = it->second;
                if (D >= processedUpTo && D < n) continue;
                int tix;
                triScoreIdx(n, D, &tix);
                candTrips[A].push_back(tix);
            }
        }
    }
    if (!dpAborted) processedUpTo = nx;
}

static double runDP(const vector<double>* gw = nullptr) {
    dpAborted = false;
    updateCands();
    if (dpAborted) return -1e300;
    int nx = X.size();
    vector<int> order;
    order.reserve(nx);
    for (int k = 1; k <= NT; k++) if (k < (int)sizeBuckets.size()) for (int id : sizeBuckets[k]) order.push_back(id);
    V.assign(nx, -1e300);
    bestSplit.assign(nx, {-1, -1});
    for (int A : order) {
        if (Xsz[A] == 1) { V[A] = 0; continue; }
        double best = -1e300; pair<int, int> bs = {-1, -1};
        for (int t : candTrips[A]) {
            int B = tripB[t], D = tripD[t];
            if (V[B] < -1e299 || V[D] < -1e299) continue;
            double sc;
            if (gw) {
                const float* pg = &PG[(size_t)t * NG];
                double a2 = 0;
                for (int g = 0; g < NG; g++) a2 += (*gw)[g] * pg[g];
                sc = a2;
            } else sc = tripTot[t];
            double val = V[B] + V[D] + sc;
            if (val > best) { best = val; bs = {B, D}; }
        }
        V[A] = best; bestSplit[A] = bs;
    }
    dpOrder = order;
    return V[TOPID];
}

// DP over recorded candidates with per-cluster bonus and trip weights
static void dpGeneric(const vector<double>& tw, const vector<double>* bonus, vector<pair<int, int>>& split, vector<double>& val) {
    int nx = X.size();
    val.assign(nx, -1e300); split.assign(nx, {-1, -1});
    for (int A : dpOrder) {
        double bA = bonus ? (*bonus)[A] : 0.0;
        if (Xsz[A] == 1) { val[A] = bA; continue; }
        double best = -1e300; pair<int, int> bs = {-1, -1};
        for (int t : candTrips[A]) {
            int B = tripB[t], D = tripD[t];
            if (val[B] < -1e299 || val[D] < -1e299) continue;
            double v = val[B] + val[D] + tw[t];
            if (v > best) { best = v; bs = {B, D}; }
        }
        if (bs.first >= 0) { val[A] = best + bA; split[A] = bs; }
    }
}
static void collectFrom(const vector<pair<int, int>>& split, int id, vector<int>& out) {
    out.push_back(id);
    if (Xsz[id] == 1) return;
    collectFrom(split, split[id].first, out);
    collectFrom(split, split[id].second, out);
}

struct TNode { int a = -1, b = -1; int leaf = -1; };
// rooted tree (anchored at ROOT) as list of clusters with children
static void collectClusters(int id, vector<int>& out) {
    out.push_back(id);
    if (Xsz[id] == 1) return;
    collectClusters(bestSplit[id].first, out);
    collectClusters(bestSplit[id].second, out);
}

static string newickOf(int id) {
    if (Xsz[id] == 1) {
        for (int i = 0; i < NT; i++) if (X[id].test(i)) return names[i];
    }
    return "(" + newickOf(bestSplit[id].first) + "," + newickOf(bestSplit[id].second) + ")";
}

// ---------------------------------------------------------------- distance tree (NJ)
static void njTree(const vector<vector<double>>& D, vector<BS>& clustersOut, string* newickOut = nullptr) {
    // standard NJ; produce clusters (sides not containing ROOT)
    int n = NT;
    vector<vector<double>> d = D;
    vector<int> active(n);
    iota(active.begin(), active.end(), 0);
    vector<BS> nodeSet(n);
    for (int i = 0; i < n; i++) nodeSet[i].set(i);
    // dynamic arrays
    vector<vector<double>> dd(2 * n, vector<double>(0));
    // We'll keep matrix indexed by node id up to 2n
    int maxN = 2 * n;
    vector<vector<double>> M(maxN, vector<double>(maxN, 0));
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) M[i][j] = d[i][j];
    vector<BS> sets(maxN);
    for (int i = 0; i < n; i++) sets[i] = nodeSet[i];
    vector<string> nw(newickOut ? maxN : 0);
    if (newickOut) for (int i = 0; i < n; i++) nw[i] = names[i];
    int next = n;
    vector<int> act = active;
    while (act.size() > 3) {
        int m = act.size();
        vector<double> r(m, 0);
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) r[i] += M[act[i]][act[j]];
        double best = 1e300; int bi = -1, bj = -1;
        for (int i = 0; i < m; i++) for (int j = i + 1; j < m; j++) {
            double q = (m - 2) * M[act[i]][act[j]] - r[i] - r[j];
            if (q < best) { best = q; bi = i; bj = j; }
        }
        int u = act[bi], v = act[bj];
        int w = next++;
        sets[w] = sets[u] | sets[v];
        if (newickOut) { nw[w] = "(" + nw[u] + "," + nw[v] + ")"; nw[u].clear(); nw[v].clear(); }
        for (int k = 0; k < m; k++) {
            int x = act[k];
            if (x == u || x == v) continue;
            double val = 0.5 * (M[u][x] + M[v][x] - M[u][v]);
            M[w][x] = M[x][w] = val;
        }
        vector<int> na;
        for (int k = 0; k < m; k++) if (k != bi && k != bj) na.push_back(act[k]);
        na.push_back(w);
        act = na;
        clustersOut.push_back(sets[w]);
    }
    // remaining 3 nodes: their sets are clusters too
    for (int x : act) clustersOut.push_back(sets[x]);
    if (newickOut && act.size() == 3) *newickOut = "(" + nw[act[0]] + "," + nw[act[1]] + "," + nw[act[2]] + ");\n";
}

static void addTreeClusters(const vector<BS>& cl) {
    BS all;
    for (int i = 0; i < NT; i++) all.set(i);
    for (auto b : cl) {
        if (b.test(ROOT)) b = all.minus(b);
        int k = b.count();
        if (k < 1) continue;
        addCluster(b);
    }
}

// ---------------------------------------------------------------- SPR expansion
static void addSPRClusters(int radius) {
    vector<int> cl;
    collectClusters(TOPID, cl);
    int nx = X.size();
    vector<int> parent(nx, -2), depth(nx, 0);
    parent[TOPID] = -1;
    for (int id : cl) {
        if (Xsz[id] == 1) continue;
        auto pr = bestSplit[id];
        parent[pr.first] = id; parent[pr.second] = id;
        depth[pr.first] = depth[id] + 1; depth[pr.second] = depth[id] + 1;
    }
    vector<BS> newCl;
    vector<int> mark(nx, -1);
    for (int v : cl) {
        if (v == TOPID) continue;
        int p = parent[v];
        int sib = (bestSplit[p].first == v) ? bestSplit[p].second : bestSplit[p].first;
        const BS S = X[v];
        vector<pair<int, int>> q;
        q.push_back({p, 0});
        mark[p] = v; mark[v] = v;
        for (size_t h = 0; h < q.size(); h++) {
            int x = q[h].first, d = q[h].second;
            if (d >= radius) continue;
            int nb[3], nn = 0;
            if (parent[x] >= 0) nb[nn++] = parent[x];
            if (Xsz[x] > 1) { nb[nn++] = bestSplit[x].first; nb[nn++] = bestSplit[x].second; }
            for (int k = 0; k < nn; k++) {
                int y = nb[k];
                if (mark[y] == v) continue;
                mark[y] = v;
                q.push_back({y, d + 1});
            }
        }
        for (auto& qq : q) {
            int w = qq.first;
            if (w == p || w == sib) continue;
            int a = p, b = w;
            vector<int> pa, pb;
            while (depth[a] > depth[b]) { pa.push_back(a); a = parent[a]; }
            while (depth[b] > depth[a]) { pb.push_back(b); b = parent[b]; }
            while (a != b) { pa.push_back(a); pb.push_back(b); a = parent[a]; b = parent[b]; }
            int L = a;
            for (int x : pa) if (x != p) newCl.push_back(X[x].minus(S));
            for (int x : pb) newCl.push_back(X[x] | S);
            if (w == L) newCl.push_back(X[w].minus(S));
        }
    }
    for (auto& b : newCl) if (b.count() >= 1) addCluster(b);
}

// emergency output (async-signal-safe double buffer)
static char* treeBuf[2] = {nullptr, nullptr};
static size_t treeLen[2] = {0, 0};
static volatile sig_atomic_t treeCur = -1;
static const char* outPath = nullptr;
static void publishTree(const string& t) {
    int nxt = (treeCur == 0) ? 1 : 0;
    free(treeBuf[nxt]);
    treeBuf[nxt] = (char*)malloc(t.size() + 1);
    memcpy(treeBuf[nxt], t.data(), t.size());
    treeLen[nxt] = t.size();
    treeCur = nxt;
}
static void onAlarm(int) {
    int c = treeCur;
    if (c >= 0 && outPath) {
        int fd = open(outPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            size_t off = 0;
            while (off < treeLen[c]) { ssize_t k = write(fd, treeBuf[c] + off, treeLen[c] - off); if (k <= 0) break; off += k; }
            close(fd);
        }
    }
    _exit(0);
}
static string currentTreeString() {
    return "(" + names[ROOT] + "," + newickOf(bestSplit[TOPID].first) + "," + newickOf(bestSplit[TOPID].second) + ");\n";
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    T0 = now();
    if (argc < 3) { fprintf(stderr, "usage: stree in out\n"); return 1; }
    const char* env = getenv("STREE_MODE");
    if (env) MODE = atoi(env);
    double smin = 1.0 / 3.0;
    if (getenv("STREE_SMIN")) smin = atof(getenv("STREE_SMIN"));
    int rounds = 100;
    if (getenv("STREE_ROUNDS")) rounds = atoi(getenv("STREE_ROUNDS"));
    double tlimit = 50;
    if (getenv("STREE_TLIMIT")) tlimit = atof(getenv("STREE_TLIMIT"));
    int useNJ = 1;
    if (getenv("STREE_BETA")) BETA = atof(getenv("STREE_BETA"));
    if (getenv("STREE_GAMMA")) GAMMA = atof(getenv("STREE_GAMMA"));
    if (getenv("STREE_NJ")) useNJ = atoi(getenv("STREE_NJ"));

    ifstream fin(argv[1]);
    string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    vector<string> trees;
    {
        string cur;
        for (char c : content) {
            cur += c;
            if (c == ';') { trees.push_back(cur); cur.clear(); }
        }
    }
    vector<vector<PN>> parsed;
    for (auto& t : trees) {
        vector<PN> pn;
        if (parseNewick(t, pn)) parsed.push_back(pn);
    }
    NT = names.size();
    ROOT = 0;
    if (nameId.count("0")) ROOT = nameId["0"];
    outPath = argv[2];
    {
        // fallback: caterpillar over all taxa
        string t;
        if (NT >= 3) {
            t = "(" + names[0] + "," + names[1] + ",";
            for (int i = 2; i < NT - 1; i++) t += "(" + names[i] + ",";
            t += names[NT - 1];
            for (int i = 2; i < NT - 1; i++) t += ")";
            t += ");\n";
        } else if (NT == 2) t = "(" + names[0] + "," + names[1] + ");\n";
        else if (NT == 1) t = names[0] + ";\n";
        publishTree(t);
        signal(SIGALRM, onAlarm);
        struct itimerval it;
        memset(&it, 0, sizeof it);
        double emerg = 108.0;
        if (getenv("STREE_EMERG")) emerg = atof(getenv("STREE_EMERG"));
        it.it_value.tv_sec = (long)emerg;
        it.it_value.tv_usec = (long)((emerg - (long)emerg) * 1e6);
        setitimer(ITIMER_REAL, &it, nullptr);
        if (NT > 64 * W || NT < 4) { onAlarm(0); }
    }
    for (auto& pn : parsed) {
        // require all taxa present
        int cnt = 0;
        for (auto& p : pn) if (p.leaf >= 0) cnt++;
        if (cnt != NT) continue;
        Gene g;
        if (buildGene(pn, g, smin)) genes.push_back(g);
    }
    fprintf(stderr, "taxa %d genes %zu\n", NT, genes.size());
    NG = genes.size();
    if (NG == 0) onAlarm(0);
    // gene edges
    E = 0;
    for (auto& g : genes) {
        for (int j = 0; j < g.nI - 1; j++) {
            PHI.push_back(g.sp[j]);
            SZE.push_back(g.clus[j].count());
            E++;
        }
    }
    // pad E to multiple of 8 with zero-weight edges
    while (E % 8) { PHI.push_back(0); SZE.push_back(0); E++; }
    if (MODE == 1) {
        // padded edges must have count 0: computeCounts writes only real edges; zero-init by resize
    }
    BS all;
    for (int i = 0; i < NT; i++) if (i != ROOT) all.set(i);
    for (int i = 0; i < NT; i++) if (i != ROOT) { BS b; b.set(i); addCluster(b); }
    TOPID = addCluster(all);
    for (auto& g : genes) for (int j = 0; j < g.nI; j++) addCluster(g.clus[j]);
    fprintf(stderr, "gene clusters: %zu  t=%.2f\n", X.size(), elapsed());
    if (useNJ) {
        // ASTRID-like internode distances
        vector<vector<double>> D(NT, vector<double>(NT, 0));
        for (auto& g : genes) {
            // build adjacency over nodes: leaves 0..NT-1, internal NT + j
            int nn = NT + g.nI;
            vector<vector<int>> adj(nn);
            for (int j = 0; j < g.nI; j++) {
                int a = g.c1[j], b = g.c2[j];
                int na = a >= 0 ? NT + a : -a - 1, nb = b >= 0 ? NT + b : -b - 1;
                adj[NT + j].push_back(na); adj[na].push_back(NT + j);
                adj[NT + j].push_back(nb); adj[nb].push_back(NT + j);
            }
            adj[NT + g.nI - 1].push_back(ROOT); adj[ROOT].push_back(NT + g.nI - 1);
            vector<int> dist(nn), q(nn);
            for (int s = 0; s < NT; s++) {
                fill(dist.begin(), dist.end(), -1);
                int h = 0, t = 0; q[t++] = s; dist[s] = 0;
                while (h < t) { int x = q[h++]; for (int y : adj[x]) if (dist[y] < 0) { dist[y] = dist[x] + 1; q[t++] = y; } }
                for (int i = 0; i < NT; i++) D[s][i] += dist[i];
            }
        }
        vector<BS> cl;
        string njNewick;
        njTree(D, cl, &njNewick);
        addTreeClusters(cl);
        if (!njNewick.empty()) publishTree(njNewick);
        fprintf(stderr, "after NJ clusters: %zu  t=%.2f\n", X.size(), elapsed());
    }
    double best = runDP();
    fprintf(stderr, "DP0 score %.6g scored %lld clusters %zu t=%.2f\n", best, nScored, X.size(), elapsed());
    if (dpAborted || bestSplit[TOPID].first < 0) { fprintf(stderr, "DP0 failed\n"); onAlarm(0); }
    publishTree(currentTreeString());
    int sprR = 0;
    if (getenv("STREE_SPR")) sprR = atoi(getenv("STREE_SPR"));
    auto addNNI = [&]() {
        vector<int> cl;
        collectClusters(TOPID, cl);
        for (int id : cl) {
            if (Xsz[id] == 1) continue;
            auto [p, q] = bestSplit[id];
            for (int side = 0; side < 2; side++) {
                int v = side ? q : p, s = side ? p : q;
                if (Xsz[v] == 1) continue;
                auto [v1, v2] = bestSplit[v];
                addCluster(X[v1] | X[s]);
                addCluster(X[v2] | X[s]);
            }
        }
        if (sprR > 0) addSPRClusters(sprR);
    };
    // local search: NNI-closure rounds on (weighted) data; returns final score
    auto localRounds = [&](const vector<double>* gw, int maxR, double cur, bool publish) {
        for (int r = 0; r < maxR && elapsed() < tlimit; r++) {
            size_t before = X.size();
            addNNI();
            if (X.size() == before) break;
            auto saveSplit = bestSplit;
            double sc = runDP(gw);
            if (dpAborted) { bestSplit = saveSplit; break; }
            if (publish) publishTree(currentTreeString());
            if (sc <= cur * (1 + 1e-12)) { cur = max(cur, sc); break; }
            cur = sc;
        }
        return cur;
    };
    best = localRounds(nullptr, rounds, best, true);
    fprintf(stderr, "main search score %.6g scored %lld clusters %zu t=%.2f\n", best, nScored, X.size(), elapsed());
    int bexp = 24;
    if (getenv("STREE_BEXP")) bexp = atoi(getenv("STREE_BEXP"));
    double texp = 70;
    if (getenv("STREE_TEXP")) texp = atof(getenv("STREE_TEXP"));
    if (bexp > 0) {
        auto mainSplit = bestSplit;
        mt19937_64 rngE(4242ULL + 7919ULL * (getenv("SEED") ? (unsigned)atoi(getenv("SEED")) : 1u));
        vector<double> w(NG);
        for (int r = 0; r < bexp && elapsed() < texp; r++) {
            fill(w.begin(), w.end(), 0.0);
            for (int k = 0; k < NG; k++) w[rngE() % NG] += 1.0;
            double sc = runDP(&w);
            if (dpAborted) break;
            localRounds(&w, 3, sc, false);
            if (dpAborted) break;
        }
        bestSplit = mainSplit;
        auto saveSplit = bestSplit;
        double sc = runDP(nullptr);
        if (dpAborted) bestSplit = saveSplit;
        else {
            best = sc;
            publishTree(currentTreeString());
            best = localRounds(nullptr, rounds, best, true);
        }
        fprintf(stderr, "after bootstrap expansion score %.6g scored %lld clusters %zu t=%.2f\n", best, nScored, X.size(), elapsed());
    }
    // ---- bootstrap consensus: maximise summed replicate support of the clusters
    unsigned seedBase = 1;
    if (getenv("SEED")) seedBase = (unsigned)atoi(getenv("SEED"));
    int nboot = 800;
    if (getenv("STREE_BOOT")) nboot = atoi(getenv("STREE_BOOT"));
    int bkind = 1;
    if (getenv("STREE_BKIND")) bkind = atoi(getenv("STREE_BKIND"));
    double lam = 0.1;
    if (getenv("STREE_LAMBDA")) lam = atof(getenv("STREE_LAMBDA"));
    if (nboot > 0 && elapsed() < 100) {
        int nt = tripTot.size();
        vector<double> freq(X.size(), 0.0);
        mt19937_64 rng(1000003ULL * seedBase + 17);
        vector<double> w(NG), tw(nt);
        vector<int> perm(NG);
        vector<pair<int, int>> sp; vector<double> val;
        int done = 0;
        for (int r = 0; r < nboot && elapsed() < 103; r++) {
            fill(w.begin(), w.end(), 0.0);
            if (bkind == 0) { for (int k = 0; k < NG; k++) w[rng() % NG] += 1.0; }
            else if (bkind == 1) { exponential_distribution<double> ex(1.0); for (int k = 0; k < NG; k++) w[k] = ex(rng); }
            else { iota(perm.begin(), perm.end(), 0); shuffle(perm.begin(), perm.end(), rng); for (int k = 0; k < NG / 2; k++) w[perm[k]] = 1.0; }
            for (int t = 0; t < nt; t++) {
                const float* pg = &PG[(size_t)t * NG];
                double a = 0;
                for (int g = 0; g < NG; g++) a += w[g] * pg[g];
                tw[t] = a;
            }
            dpGeneric(tw, nullptr, sp, val);
            vector<int> cl;
            collectFrom(sp, TOPID, cl);
            for (int c : cl) freq[c] += 1.0;
            done++;
        }
        if (done > 0) {
            for (auto& f : freq) f /= done;
            double scale = best / max(1, NT - 2);
            for (int t = 0; t < nt; t++) tw[t] = lam * tripTot[t] / scale;
            dpGeneric(tw, &freq, sp, val);
            if (sp[TOPID].first >= 0) bestSplit = sp;
        }
        fprintf(stderr, "bootstrap consensus (%d reps) t=%.2f\n", done, elapsed());
    }
    if (getenv("STREE_VARIANTS")) {
        // diagnostics: write consensus variants to OUT.v<k>
        int nt = tripTot.size();
        vector<double> tw(nt);
        vector<pair<int, int>> sp; vector<double> val;
        auto makeFreq = [&](int kind, int nb, unsigned seed) {
            vector<double> freq(X.size(), 0.0);
            mt19937_64 rng(seed);
            vector<double> w(NG);
            vector<int> perm(NG);
            for (int r = 0; r < nb; r++) {
                fill(w.begin(), w.end(), 0.0);
                if (kind == 0) { for (int k = 0; k < NG; k++) w[rng() % NG] += 1.0; }
                else if (kind == 1) { exponential_distribution<double> ex(1.0); for (int k = 0; k < NG; k++) w[k] = ex(rng); }
                else { iota(perm.begin(), perm.end(), 0); shuffle(perm.begin(), perm.end(), rng); for (int k = 0; k < NG / 2; k++) w[perm[k]] = 1.0; }
                for (int t = 0; t < nt; t++) {
                    const float* pg = &PG[(size_t)t * NG];
                    double a = 0;
                    for (int g = 0; g < NG; g++) a += w[g] * pg[g];
                    tw[t] = a;
                }
                dpGeneric(tw, nullptr, sp, val);
                vector<int> cl;
                collectFrom(sp, TOPID, cl);
                for (int c : cl) freq[c] += 1.0 / nb;
            }
            return freq;
        };
        double scale = best / max(1, NT - 2);
        auto saved = bestSplit;
        auto emit = [&](int k, const vector<double>& freq, double lam) {
            for (int t = 0; t < nt; t++) tw[t] = lam * tripTot[t] / scale;
            dpGeneric(tw, &freq, sp, val);
            bestSplit = sp;
            string o = currentTreeString();
            string fn = string(argv[2]) + ".v" + to_string(k);
            FILE* f = fopen(fn.c_str(), "w"); fputs(o.c_str(), f); fclose(f);
        };
        auto f0 = makeFreq(0, 100, 777);
        emit(0, f0, 0.001);
        auto f1 = makeFreq(0, 400, 778);
        emit(1, f1, 0.3);
        emit(4, f1, 3.0);
        auto f2 = makeFreq(1, 400, 779);
        emit(2, f2, 0.3);
        auto f3 = makeFreq(2, 400, 780);
        emit(3, f3, 0.3);
        emit(5, f2, 0.001);
        emit(6, f1, 0.001);
        bestSplit = saved;
        fprintf(stderr, "variants done t=%.2f\n", elapsed());
    }
    if (getenv("STREE_TRUTH")) {
        ifstream ft(getenv("STREE_TRUTH"));
        string tc((istreambuf_iterator<char>(ft)), istreambuf_iterator<char>());
        vector<PN> pn;
        parseNewick(tc, pn);
        Gene g;
        if (buildGene(pn, g, smin)) {
            int inX = 0, inTree = 0;
            vector<int> cl;
            collectClusters(TOPID, cl);
            unordered_set<int> cs(cl.begin(), cl.end());
            for (int j = 0; j < g.nI; j++) {
                auto it = Xid.find(g.clus[j]);
                if (it != Xid.end()) { inX++; if (cs.count(it->second)) inTree++; }
            }
            double ts = 0;
            for (int j = 0; j < g.nI; j++) {
                BS b1, b2;
                if (g.c1[j] >= 0) b1 = g.clus[g.c1[j]]; else b1.set(-g.c1[j] - 1);
                if (g.c2[j] >= 0) b2 = g.clus[g.c2[j]]; else b2.set(-g.c2[j] - 1);
                int i1 = addCluster(b1), i2 = addCluster(b2);
                addCluster(b1 | b2);
                ts += triScore(i1, i2);
            }
            fprintf(stderr, "TRUTH: clusters %d inX %d inTree %d truthScore %.8g ourScore %.8g ratio %.6f\n", g.nI, inX, inTree, ts, best, ts / best);
        }
    }
    string out = currentTreeString();
    publishTree(out);
    {
        struct itimerval it;
        memset(&it, 0, sizeof it);
        setitimer(ITIMER_REAL, &it, nullptr);
    }
    FILE* fo = fopen(argv[2], "w");
    if (!fo) { fprintf(stderr, "cannot open output\n"); return 1; }
    fputs(out.c_str(), fo);
    fclose(fo);
    fprintf(stderr, "done t=%.2f\n", elapsed());
    return 0;
}
