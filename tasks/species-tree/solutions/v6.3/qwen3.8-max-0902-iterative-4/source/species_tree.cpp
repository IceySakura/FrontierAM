// species_tree.cpp -- ASTRAL-style species tree inference from gene trees.
//
// Objective (ASTRAL): maximize |Q_G ∩ Q_T|, the number of quartet topologies
// shared between the predicted species tree T and the input gene trees G.
//
// The score decomposes over the levels of T's recursive (clade) decomposition:
// for a level that splits clade S into A|B (R = V\S), each quartet is owned by
// exactly one level, and
//   levelScore(A,B;S) = M(A) + M(B) - N(A,B) - N(A,R) - N(B,R)
//                     = N(A,B) + Str(A;B,R) + Str(B;A,R)
// where N(X,Y) = # quartets {x1,x2}|{y1,y2} displayed by the gene trees with
// x's in X and y's in Y (X,Y disjoint), M(A) = N(A, V\A), and Str counts
// quartets whose second pair straddles (B,R).
//
// N(g;X,Y) per gene tree is computed exactly in O(#nodes) by a postorder
// recursion (valid for arbitrary node degrees / multifurcations).
//
// DP: v(S) over subsets S that are sides of bipartitions in X (the split set of
// the gene trees), maximizing the sum of levelScores. Optional filters:
// minimum split frequency, contraction of low-support/short gene-tree edges.
//
// Single file, C++17, no dependencies, deterministic, single-threaded.
//
// Build:   g++ -O2 -std=c++17 -o species_tree species_tree.cpp
// Usage:   species_tree INPUT OUTPUT [options]
// Options: --min-freq F       only keep splits seen in >= F gene trees (default 1)
//          --collapse-sup S   contract gene-tree edges with support < S (default 0=off)
//          --collapse-len L   contract gene-tree edges with length < L (default 0=off)
//          --time-guard SEC   abort optional phases after SEC seconds (default 105)
//          --selftest         run internal correctness tests and exit

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

using namespace std;

typedef long long i64;
typedef unsigned long long u64;

// ---------------------------------------------------------------- bitsets ---
static const int W = 4; // 256 bits, enough for n <= 256 taxa
struct BS { u64 w[W]; };

static int NTAX = 0;
static u64 VMASK[W];

static inline void bsInit(int n) {
    NTAX = n;
    for (int i = 0; i < W; i++) VMASK[i] = 0;
    for (int t = 0; t < n && t < 256; t++) VMASK[t >> 6] |= 1ULL << (t & 63);
}
static inline void bsClear(BS &b) { memset(b.w, 0, sizeof(BS)); }
static inline bool bsTest(const BS &b, int t) { return (b.w[t >> 6] >> (t & 63)) & 1ULL; }
static inline void bsSet(BS &b, int t) { b.w[t >> 6] |= 1ULL << (t & 63); }
static inline BS bsOr(const BS &a, const BS &b) { BS r; for (int i = 0; i < W; i++) r.w[i] = a.w[i] | b.w[i]; return r; }
static inline BS bsAnd(const BS &a, const BS &b) { BS r; for (int i = 0; i < W; i++) r.w[i] = a.w[i] & b.w[i]; return r; }
static inline BS bsComp(const BS &a) { BS r; for (int i = 0; i < W; i++) r.w[i] = ~a.w[i] & VMASK[i]; return r; }
static inline bool bsEq(const BS &a, const BS &b) { for (int i = 0; i < W; i++) if (a.w[i] != b.w[i]) return false; return true; }
static inline bool bsDisjoint(const BS &a, const BS &b) { for (int i = 0; i < W; i++) if (a.w[i] & b.w[i]) return false; return true; }
static inline int bsPop(const BS &a) { int c = 0; for (int i = 0; i < W; i++) c += __builtin_popcountll(a.w[i]); return c; }

static inline u64 mix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline u64 bsHash(const BS &b) {
    return mix64(b.w[0] ^ mix64(b.w[1] ^ mix64(b.w[2] ^ mix64(b.w[3]))));
}
static inline i64 C2(i64 x) { return x * (x - 1) / 2; }

// ------------------------------------------------------------ open hashes ---
struct BSMap { // BS -> int index
    vector<BS> keys;
    vector<int> vals; // -1 = empty
    size_t mask = 0;
    size_t n = 0;
    void init(size_t cap = 1024) {
        size_t c = 1;
        while (c < cap * 2) c <<= 1;
        keys.assign(c, BS{{0, 0, 0, 0}});
        vals.assign(c, -1);
        mask = c - 1;
        n = 0;
    }
    void grow() {
        vector<BS> ok = move(keys);
        vector<int> ov = move(vals);
        size_t ocap = mask + 1, c = ocap * 2;
        keys.assign(c, BS{{0, 0, 0, 0}});
        vals.assign(c, -1);
        mask = c - 1;
        n = 0;
        for (size_t j = 0; j < ocap; j++)
            if (ov[j] != -1) { *find(ok[j]) = ov[j]; n--; }
        n += 0; // count fixed below
        // recompute n
        n = 0;
        for (size_t j = 0; j <= mask; j++) if (vals[j] != -1) n++;
    }
    int *find(const BS &k) {
        if (n * 2 > mask) grow();
        size_t i = (size_t)(bsHash(k) & mask);
        while (vals[i] != -1) {
            if (bsEq(keys[i], k)) return &vals[i];
            i = (i + 1) & mask;
        }
        keys[i] = k;
        n++;
        return &vals[i];
    }
    int get(const BS &k) const {
        if (!mask) return -1;
        size_t i = (size_t)(bsHash(k) & mask);
        while (vals[i] != -1) {
            if (bsEq(keys[i], k)) return vals[i];
            i = (i + 1) & mask;
        }
        return -1;
    }
};

struct U64i64Map { // u64 key -> i64 value, open addressing, growable
    vector<u64> keys;
    vector<i64> vals;
    vector<char> used;
    size_t mask = 0, n = 0;
    void init(size_t cap = 1024) {
        size_t c = 1;
        while (c < cap * 2) c <<= 1;
        keys.assign(c, 0); vals.assign(c, 0); used.assign(c, 0);
        mask = c - 1; n = 0;
    }
    void grow() {
        vector<u64> ok = move(keys); vector<i64> ov = move(vals); vector<char> ou = move(used);
        size_t ocap = mask + 1, c = ocap * 2;
        keys.assign(c, 0); vals.assign(c, 0); used.assign(c, 0);
        mask = c - 1; n = 0;
        for (size_t i = 0; i < ocap; i++) if (ou[i]) { *find(ok[i]) = ov[i]; }
    }
    i64 *find(u64 k) {
        if (n * 2 > mask) grow();
        size_t i = (size_t)(mix64(k) & mask);
        while (used[i]) {
            if (keys[i] == k) return &vals[i];
            i = (i + 1) & mask;
        }
        keys[i] = k; used[i] = 1; vals[i] = 0; n++;
        return &vals[i];
    }
    const i64 *get(u64 k) const {
        if (!mask) return nullptr;
        size_t i = (size_t)(mix64(k) & mask);
        while (used[i]) {
            if (keys[i] == k) return &vals[i];
            i = (i + 1) & mask;
        }
        return nullptr;
    }
};

struct BSi64Map { // BS -> i64, open addressing, growable
    vector<BS> keys;
    vector<i64> vals;
    vector<char> used;
    size_t mask = 0, n = 0;
    void init(size_t cap = 1024) {
        size_t c = 1;
        while (c < cap * 2) c <<= 1;
        keys.assign(c, BS{{0,0,0,0}}); vals.assign(c, 0); used.assign(c, 0);
        mask = c - 1; n = 0;
    }
    void grow() {
        vector<BS> ok = move(keys); vector<i64> ov = move(vals); vector<char> ou = move(used);
        size_t ocap = mask + 1, c = ocap * 2;
        keys.assign(c, BS{{0,0,0,0}}); vals.assign(c, 0); used.assign(c, 0);
        mask = c - 1; n = 0;
        for (size_t i = 0; i < ocap; i++) if (ou[i]) { *find(ok[i]) = ov[i]; }
    }
    i64 *find(const BS &k) {
        if (n * 2 > mask) grow();
        size_t i = (size_t)(bsHash(k) & mask);
        while (used[i]) {
            if (bsEq(keys[i], k)) return &vals[i];
            i = (i + 1) & mask;
        }
        keys[i] = k; used[i] = 1; vals[i] = 0; n++;
        return &vals[i];
    }
    const i64 *get(const BS &k) const {
        if (!mask) return nullptr;
        size_t i = (size_t)(bsHash(k) & mask);
        while (used[i]) {
            if (bsEq(keys[i], k)) return &vals[i];
            i = (i + 1) & mask;
        }
        return nullptr;
    }
};

// ------------------------------------------------------------- timing -------
static double nowSec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double T0;
static double GUARD = 105;

// ------------------------------------------------------------ newick --------
struct EAttr { double sup = 0, len = 0; char hasSup = 0, hasLen = 0; };

struct PTree {
    vector<vector<int>> kids;
    vector<int> par;
    vector<int> taxon; // leaf -> taxon id, internal -> -1
    vector<EAttr> attr; // edge (par[v],v)
    int root = -1;
};

static bool isSpecial(char c) {
    return c == '(' || c == ')' || c == ',' || c == ':' || c == ';' || c == '\0' ||
           c == '\n' || c == '\r' || c == '\t' || c == ' ';
}

static bool parseNewick(const char *s, PTree &pt, vector<string> &labels) {
    int nn = 0;
    pt.kids.clear(); pt.par.clear(); pt.taxon.clear(); pt.attr.clear();
    auto newNode = [&]() -> int {
        pt.kids.push_back(vector<int>());
        pt.par.push_back(-1);
        pt.taxon.push_back(-1);
        pt.attr.push_back(EAttr());
        return nn++;
    };
    int dummy = newNode();
    vector<int> st;
    st.push_back(dummy);
    size_t i = 0;
    bool expectChild = true;
    auto registerLeaf = [&](const string &lab) -> int {
        int id = -1;
        for (size_t t = 0; t < labels.size(); t++)
            if (labels[t] == lab) { id = (int)t; break; }
        if (id < 0) { id = (int)labels.size(); labels.push_back(lab); }
        return id;
    };
    while (s[i]) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { i++; continue; }
        if (c == '(') {
            int u = newNode();
            int p = st.back();
            pt.par[u] = p;
            pt.kids[p].push_back(u);
            st.push_back(u);
            i++;
            expectChild = true;
            continue;
        }
        if (c == ',') { i++; expectChild = true; continue; }
        if (c == ')') {
            if (st.size() <= 1) return false;
            int v = st.back();
            st.pop_back();
            i++;
            size_t j = i;
            while (s[j] && !isSpecial(s[j])) j++;
            if (j > i) {
                string tok(s + i, j - i);
                char *end = nullptr;
                double d = strtod(tok.c_str(), &end);
                if (end && *end == '\0') { pt.attr[v].sup = d; pt.attr[v].hasSup = 1; }
                i = j;
            }
            if (s[i] == ':') {
                i++;
                size_t k = i;
                while (s[k] && !isSpecial(s[k])) k++;
                string tok(s + i, k - i);
                char *end = nullptr;
                double d = strtod(tok.c_str(), &end);
                if (end && *end == '\0' && !tok.empty()) { pt.attr[v].len = d; pt.attr[v].hasLen = 1; }
                i = k;
            }
            expectChild = false;
            continue;
        }
        if (c == ';') { i++; break; }
        if (c == ':') {
            i++;
            size_t k = i;
            while (s[k] && !isSpecial(s[k])) k++;
            i = k;
            continue;
        }
        if (!expectChild) return false;
        string lab;
        if (c == '\'') {
            i++;
            while (s[i]) {
                if (s[i] == '\'') {
                    if (s[i + 1] == '\'') { lab += '\''; i += 2; continue; }
                    i++;
                    break;
                }
                lab += s[i++];
            }
        } else {
            size_t j = i;
            while (s[j] && !isSpecial(s[j])) j++;
            if (j == i) return false;
            lab.assign(s + i, j - i);
            i = j;
        }
        int id = registerLeaf(lab);
        int u = newNode();
        pt.taxon[u] = id;
        int p = st.back();
        pt.par[u] = p;
        pt.kids[p].push_back(u);
        if (s[i] == ':') {
            i++;
            size_t k = i;
            while (s[k] && !isSpecial(s[k])) k++;
            string tok(s + i, k - i);
            char *end = nullptr;
            double d = strtod(tok.c_str(), &end);
            if (end && *end == '\0' && !tok.empty()) { pt.attr[u].len = d; pt.attr[u].hasLen = 1; }
            i = k;
        }
        expectChild = false;
    }
    if (st.size() != 1) return false;
    int r = dummy;
    while ((int)pt.kids[r].size() == 1) {
        int u = pt.kids[r][0];
        pt.attr[u] = EAttr();
        pt.par[u] = -1;
        r = u;
    }
    if (pt.kids[r].empty()) return false;
    pt.root = r;
    return true;
}

// ------------------------------------------------------- unrooted + rooted --
struct GTree {
    int nNodes = 0;
    int root = -1;
    vector<int> par;
    vector<int> chOff, ch;
    vector<int> post;      // internal nodes incl. root, children before parent
    vector<int> nLeaves;
    vector<int> leafOf;    // taxon -> leaf node, -1 if absent
    vector<BS> clade;
    vector<EAttr> attr;
    vector<int> splitId;   // internal node -> split id in X or -1
    vector<char> isLeaf;
    BS mask;
    int nTax = 0;
};

static bool buildGTree(const PTree &pt, int outTaxon, double supT, double lenT, GTree &gt) {
    int pn = (int)pt.kids.size();
    vector<vector<pair<int, EAttr>>> adj(pn);
    int pr = pt.root;
    vector<char> dead(pn, 0);
    for (int v = 0; v < pn; v++) if (v != pr && pt.par[v] < 0) dead[v] = 1;
    if ((int)pt.kids[pr].size() == 2) {
        dead[pr] = 1;
        int a = pt.kids[pr][0], b = pt.kids[pr][1];
        EAttr e;
        if (pt.attr[a].hasLen && pt.attr[b].hasLen) { e.len = pt.attr[a].len + pt.attr[b].len; e.hasLen = 1; }
        else if (pt.attr[a].hasLen) e = pt.attr[a];
        else if (pt.attr[b].hasLen) e = pt.attr[b];
        if (pt.attr[a].hasSup && pt.attr[b].hasSup) { e.sup = max(pt.attr[a].sup, pt.attr[b].sup); e.hasSup = 1; }
        else if (pt.attr[a].hasSup) { e.sup = pt.attr[a].sup; e.hasSup = 1; }
        else if (pt.attr[b].hasSup) { e.sup = pt.attr[b].sup; e.hasSup = 1; }
        adj[a].push_back({b, e});
        adj[b].push_back({a, e});
        for (int v = 0; v < pn; v++) {
            if (v == pr || v == a || v == b || dead[v]) continue;
            EAttr at = pt.attr[v];
            adj[v].push_back({pt.par[v], at});
            adj[pt.par[v]].push_back({v, at});
        }
    } else {
        for (int v = 0; v < pn; v++) {
            if (v == pr || dead[v]) continue;
            EAttr at = pt.attr[v];
            adj[v].push_back({pt.par[v], at});
            adj[pt.par[v]].push_back({v, at});
        }
    }

    vector<int> uf(pn);
    for (int i = 0; i < pn; i++) uf[i] = i;
    function<int(int)> find = [&](int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
    if (supT > 0 || lenT > 0) {
        for (int u = 0; u < pn; u++) {
            if (dead[u] || pt.taxon[u] >= 0) continue;
            for (auto &ea : adj[u]) {
                int v = ea.first;
                if (v < u || dead[v] || pt.taxon[v] >= 0) continue;
                const EAttr &e = ea.second;
                bool cut = false;
                if (supT > 0 && e.hasSup && e.sup < supT) cut = true;
                if (lenT > 0 && e.hasLen && e.len < lenT) cut = true;
                if (cut) {
                    int ru = find(u), rv = find(v);
                    if (ru != rv) uf[rv] = ru;
                }
            }
        }
    }
    vector<int> repId(pn, -1);
    int nn = 0;
    vector<int> taxonNew;
    vector<vector<pair<int, EAttr>>> adj2;
    auto getId = [&](int r) -> int {
        if (repId[r] < 0) {
            repId[r] = nn++;
            adj2.push_back({});
            taxonNew.push_back(pt.taxon[r]);
        }
        return repId[r];
    };
    for (int u = 0; u < pn; u++) {
        if (dead[u]) continue;
        int id = getId(find(u));
        if (pt.taxon[u] >= 0) taxonNew[id] = pt.taxon[u];
    }
    for (int u = 0; u < pn; u++) {
        if (dead[u]) continue;
        for (auto &ea : adj[u]) {
            int v = ea.first;
            if (dead[v]) continue;
            int ru = find(u), rv = find(v);
            if (ru == rv) continue;
            adj2[getId(ru)].push_back({getId(rv), ea.second});
        }
    }

    int outNode = -1;
    for (int u = 0; u < nn; u++) if (taxonNew[u] == outTaxon) { outNode = u; break; }
    if (outNode < 0 || adj2[outNode].size() != 1) return false;
    int rootNode = adj2[outNode][0].first;

    gt.nNodes = nn;
    gt.root = rootNode;
    gt.par.assign(nn, -1);
    gt.attr.assign(nn, EAttr());
    gt.isLeaf.assign(nn, 0);
    gt.leafOf.assign(NTAX, -1);
    bsClear(gt.mask);
    vector<int> order;
    order.reserve(nn);
    {
        vector<vector<int>> chList(nn);
        vector<pair<int, int>> stack;
        stack.push_back({rootNode, 0});
        gt.par[rootNode] = rootNode;
        while (!stack.empty()) {
            int v = stack.back().first;
            int ci = stack.back().second;
            if (ci < (int)adj2[v].size()) {
                stack.back().second++;
                int w = adj2[v][ci].first;
                if (w == gt.par[v]) continue;
                gt.par[w] = v;
                gt.attr[w] = adj2[v][ci].second;
                chList[v].push_back(w);
                stack.push_back({w, 0});
            } else {
                order.push_back(v);
                stack.pop_back();
            }
        }
        if ((int)order.size() != nn) return false;
        int tot = 0;
        gt.chOff.assign(nn + 1, 0);
        for (int v = 0; v < nn; v++) { gt.chOff[v] = tot; tot += (int)chList[v].size(); }
        gt.chOff[nn] = tot;
        gt.ch.assign(tot, -1);
        for (int v = 0; v < nn; v++)
            for (size_t j = 0; j < chList[v].size(); j++)
                gt.ch[gt.chOff[v] + (int)j] = chList[v][j];
    }
    gt.nLeaves.assign(nn, 0);
    gt.clade.assign(nn, BS{{0, 0, 0, 0}});
    gt.post.clear();
    gt.post.reserve(nn / 2 + 1);
    for (int v : order) {
        if (gt.chOff[v + 1] - gt.chOff[v] == 0) {
            int t = taxonNew[v];
            if (t < 0 || t >= NTAX) return false;
            if (gt.leafOf[t] >= 0) return false;
            gt.isLeaf[v] = 1;
            gt.leafOf[t] = v;
            bsClear(gt.clade[v]);
            bsSet(gt.clade[v], t);
            bsSet(gt.mask, t);
            gt.nLeaves[v] = 1;
        } else {
            BS acc; bsClear(acc);
            int nl = 0;
            for (int j = gt.chOff[v]; j < gt.chOff[v + 1]; j++) {
                acc = bsOr(acc, gt.clade[gt.ch[j]]);
                nl += gt.nLeaves[gt.ch[j]];
            }
            gt.clade[v] = acc;
            gt.nLeaves[v] = nl;
            gt.post.push_back(v);
        }
    }
    gt.nTax = bsPop(gt.mask);
    gt.splitId.assign(nn, -1);
    return true;
}

// --------------------------------------------- generic N(g; X, Y) scorer ----
// Counts quartets {x1,x2} subset X, {y1,y2} subset Y whose topology x1x2|y1y2
// is displayed by gt. X and Y must be disjoint. O(#nodes), any node degree.
static vector<int> s_xa, s_xb;
static vector<i64> s_sab, s_sba;

static i64 NcountGen(const GTree &gt, const BS &X, const BS &Y) {
    int nn = gt.nNodes;
    if ((int)s_xa.size() < nn) { s_xa.resize(nn); s_xb.resize(nn); s_sab.resize(nn); s_sba.resize(nn); }
    int *xa = s_xa.data();
    int *xb = s_xb.data();
    i64 *sab = s_sab.data();
    i64 *sba = s_sba.data();
    const int *leafOf = gt.leafOf.data();
    for (int t = 0; t < NTAX; t++) {
        int nd = leafOf[t];
        if (nd >= 0) {
            xa[nd] = bsTest(X, t) ? 1 : 0;
            xb[nd] = bsTest(Y, t) ? 1 : 0;
            sab[nd] = 0; sba[nd] = 0;
        }
    }
    i64 N = 0;
    const int *ch = gt.ch.data();
    const int *off = gt.chOff.data();
    const int *nl = gt.nLeaves.data();
    for (size_t pi = 0; pi < gt.post.size(); pi++) {
        int v = gt.post[pi];
        int o = off[v], oe = off[v + 1];
        i64 xv = 0, yv = 0;
        i64 SCX = 0, SCY = 0, SCXY = 0, SX2 = 0, SY2 = 0;
        i64 sabSum = 0, sbaSum = 0;
        for (int j = o; j < oe; j++) {
            int c = ch[j];
            i64 xi = xa[c], yi = xb[c];
            i64 CX = C2(xi), CY = C2(yi);
            xv += xi; yv += yi;
            SCX += CX; SCY += CY; SCXY += CX * CY;
            SX2 += xi * xi; SY2 += yi * yi;
            sabSum += sab[c];
            sbaSum += sba[c];
        }
        xa[v] = (int)xv;
        xb[v] = (int)yv;
        if (xv == 0 || yv == 0) { sab[v] = 0; sba[v] = 0; continue; }
        i64 pX = (xv * xv - SX2) / 2; // # X-pairs with LCA exactly v
        i64 pY = (yv * yv - SY2) / 2; // # Y-pairs with LCA exactly v
        sab[v] = pY * xv + sabSum;
        sba[v] = pX * yv + sbaSum;
        i64 T1 = SCX * SCY - SCXY;
        i64 P1 = pX * SCY, P2 = 0, Q1 = pY * SCX, Q2 = 0;
        for (int j = o; j < oe; j++) {
            int c = ch[j];
            i64 xi = xa[c], yi = xb[c];
            i64 CX = C2(xi), CY = C2(yi);
            P1 -= CY * xi * (xv - xi);
            P2 += (xv - xi) * (xi * CY - sab[c]);
            Q1 -= CX * yi * (yv - yi);
            Q2 += (yv - yi) * (yi * CX - sba[c]);
        }
        N += T1 + P1 + P2 + Q1 + Q2;
    }
    return N;
}

// brute-force reference for selftests
static i64 NcountBrute(const GTree &gt, const BS &X, const BS &Y) {
    vector<BS> sides;
    for (int v = 0; v < gt.nNodes; v++) {
        if (v == gt.root) continue;
        sides.push_back(gt.clade[v]);
    }
    vector<int> xs, ys;
    for (int t = 0; t < NTAX; t++) {
        if (!bsTest(gt.mask, t)) continue;
        if (bsTest(X, t)) xs.push_back(t);
        else if (bsTest(Y, t)) ys.push_back(t);
    }
    i64 cnt = 0;
    for (size_t i1 = 0; i1 < xs.size(); i1++)
        for (size_t i2 = i1 + 1; i2 < xs.size(); i2++)
            for (size_t j1 = 0; j1 < ys.size(); j1++)
                for (size_t j2 = j1 + 1; j2 < ys.size(); j2++) {
                    for (const BS &s : sides) {
                        bool s1 = bsTest(s, xs[i1]), s2 = bsTest(s, xs[i2]);
                        bool t1 = bsTest(s, ys[j1]), t2 = bsTest(s, ys[j2]);
                        if ((s1 && s2 && !t1 && !t2) || (!s1 && !s2 && t1 && t2)) { cnt++; break; }
                    }
                }
    return cnt;
}


// ------------------------------------------------- level-score evaluation ---
// Cached evaluation of M(A) and N(X,Y) for arbitrary taxon sets, used by the
// NNI local search (and to rescore trees). Sets are registered by index.
struct EvalCache {
    BSMap bmap;
    vector<BS> bsets;
    U64i64Map pairs;
    const vector<GTree> *gts = nullptr;
    void init(const vector<GTree> &g) { gts = &g; bmap.init(1 << 12); pairs.init(1 << 15); }
    int idx(const BS &b) {
        int *s = bmap.find(b);
        if (*s == -1) { *s = (int)bsets.size(); bsets.push_back(b); }
        return *s;
    }
    static u64 keyOf(int ix, int iy) {
        if (ix > iy) { int t = ix; ix = iy; iy = t; }
        return (u64)ix * 1000003ULL + (u64)iy;
    }
    void putN(int ix, int iy, i64 val) { *pairs.find(keyOf(ix, iy)) = val; }
    i64 Npair(int ix, int iy) {
        u64 key = keyOf(ix, iy);
        const i64 *h = pairs.get(key);
        if (h) return *h;
        i64 tot = 0;
        if (bsPop(bsets[ix]) >= 2 && bsPop(bsets[iy]) >= 2) {
            for (const GTree &gt : *gts) tot += NcountGen(gt, bsets[ix], bsets[iy]);
        }
        *pairs.find(key) = tot;
        return tot;
    }
    i64 M(int i) { return Npair(i, idx(bsComp(bsets[i]))); }
    // levelScore(A,B;S) = M(A)+M(B)-N(A,B)-N(A,R)-N(B,R), R = V\S
    i64 LS(int iA, int iB, int iS) {
        int iR = idx(bsComp(bsets[iS]));
        return M(iA) + M(iB) - Npair(iA, iB) - Npair(iA, iR) - Npair(iB, iR);
    }
};

// ------------------------------------------------------------ globals -------
static vector<string> LABELS;
static int OUTID = -1;

struct SplitRec { BS side; int sz; int freq; };

// ================================= selftest part 1: N formula ===============
static int selftestN() {
    srand(12345);
    int fails = 0;
    for (int trial = 0; trial < 400; trial++) {
        int n = 4 + rand() % 6;
        bsInit(n);
        LABELS.clear();
        for (int t = 0; t < n; t++) LABELS.push_back(to_string(t));
        int cap = 2 * n + 2;
        vector<vector<int>> adj2(cap);
        auto addEdge = [&](int u, int v) { adj2[u].push_back(v); adj2[v].push_back(u); };
        auto delEdge = [&](int u, int v) {
            adj2[u].erase(remove(adj2[u].begin(), adj2[u].end(), v), adj2[u].end());
        };
        int i0 = n;
        addEdge(0, i0); addEdge(1, i0); addEdge(2, i0);
        int nInternals = i0 + 1;
        for (int leaf = 3; leaf < n; leaf++) {
            vector<pair<int, int>> edges;
            for (int u = 0; u < nInternals; u++)
                for (int v : adj2[u]) if (u < v) edges.push_back({u, v});
            auto e = edges[rand() % edges.size()];
            int mid = nInternals++;
            delEdge(e.first, e.second);
            delEdge(e.second, e.first);
            addEdge(e.first, mid); addEdge(e.second, mid); addEdge(leaf, mid);
        }
        int ncontract = (trial % 3 == 0) ? 1 + rand() % 2 : 0;
        for (int q = 0; q < ncontract; q++) {
            vector<pair<int, int>> iedges;
            for (int u = n; u < nInternals; u++)
                for (int v : adj2[u]) if (v > u && v >= n) iedges.push_back({u, v});
            if (iedges.empty()) break;
            auto e = iedges[rand() % iedges.size()];
            for (int w : adj2[e.second]) {
                if (w == e.first) continue;
                delEdge(w, e.second);
                addEdge(w, e.first);
            }
            delEdge(e.first, e.second);
            adj2[e.second].clear();
        }
        int r0 = adj2[0][0];
        string nwk;
        function<void(int, int)> rec2 = [&](int v, int p) {
            if (v < n) { nwk += to_string(v); return; }
            nwk += "(";
            bool first = true;
            for (int w : adj2[v]) {
                if (w == p) continue;
                if (!first) nwk += ",";
                first = false;
                rec2(w, v);
            }
            nwk += ")";
        };
        nwk += "(0";
        for (int w : adj2[r0]) {
            if (w == 0) continue;
            nwk += ",";
            rec2(w, r0);
        }
        nwk += ");";
        PTree pt;
        if (!parseNewick(nwk.c_str(), pt, LABELS)) { fails++; continue; }
        GTree gt;
        if (!buildGTree(pt, 0, 0, 0, gt)) { fails++; continue; }
        for (int rep = 0; rep < 8; rep++) {
            BS X, Y; bsClear(X); bsClear(Y);
            for (int t = 0; t < n; t++) {
                int r = rand() % 4;
                if (r == 0) bsSet(X, t);
                else if (r == 1) bsSet(Y, t);
            }
            if (bsPop(X) < 2 || bsPop(Y) < 2) continue;
            i64 f = NcountGen(gt, X, Y);
            i64 b = NcountBrute(gt, X, Y);
            if (f != b) {
                fprintf(stderr, "SELFTEST N MISMATCH trial=%d n=%d nw=%s f=%lld b=%lld\n",
                        trial, n, nwk.c_str(), (long long)f, (long long)b);
                fails++;
            }
        }
    }
    fprintf(stderr, "selftest N done, fails=%d\n", fails);
    return fails;
}

// ================= selftest part 2: full DP vs brute force ==================
// (declared here, implemented after the pipeline pieces)
static int selftestDP();

// ------------------------------------------------------------- pipeline ----
// Shared structures so selftestDP can reuse the pipeline.
struct Pipeline {
    vector<GTree> gts;
    vector<SplitRec> splits;
    vector<int> kept;          // split ids kept after filtering
    vector<int> keptSideIdx;   // side idx for each kept split
    vector<BS> sideBS;
    vector<int> sideSz;
    vector<i64> sideM;         // M(A) = N(A, V\A) summed over gene trees
    BSMap smap;
    int S0idx = -1;
    vector<i64> v;
    vector<pair<int,int>> bestPair;
    long long nLevels = 0;
};

int main(int argc, char **argv) {
    T0 = nowSec();
    if (argc < 3) {
        fprintf(stderr, "usage: %s INPUT OUTPUT [opts]\n", argv[0]);
        return 2;
    }
    const char *inPath = argv[1];
    const char *outPath = argv[2];

    int minFreq = 1;
    double supT = 0, lenT = 0;
    int nniRounds = 0;
    int rexpand = 2;
    bool selftest = false;
    const char *scoreTreePath = nullptr;
    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--min-freq") minFreq = atoi(next());
        else if (a == "--collapse-sup") supT = atof(next());
        else if (a == "--collapse-len") lenT = atof(next());
        else if (a == "--time-guard") GUARD = atof(next());
        else if (a == "--selftest") selftest = true;
        else if (a == "--nni") nniRounds = atoi(next());
        else if (a == "--rex") rexpand = atoi(next());
        else if (a == "--score-tree") scoreTreePath = next();
    }

    if (selftest) {
        int f = selftestN();
        f += selftestDP();
        return f ? 1 : 0;
    }

    // ---------------------------------------------------------------- read ----
    vector<string> lines;
    {
        FILE *fin = fopen(inPath, "rb");
        if (!fin) { fprintf(stderr, "cannot open input %s\n", inPath); return 1; }
        string all;
        char buf[1 << 16];
        while (size_t rd = fread(buf, 1, sizeof(buf), fin)) all.append(buf, rd);
        fclose(fin);
        size_t pos = 0;
        while (pos < all.size()) {
            size_t nl = all.find('\n', pos);
            if (nl == string::npos) nl = all.size();
            string ln = all.substr(pos, nl - pos);
            pos = nl + 1;
            while (!ln.empty() && (ln.back() == '\r' || ln.back() == ' ' || ln.back() == '\t')) ln.pop_back();
            size_t b = 0;
            while (b < ln.size() && (ln[b] == ' ' || ln[b] == '\t')) b++;
            if (b < ln.size()) lines.push_back(ln.substr(b));
        }
    }
    auto writeRaw = [&](const string &txt) {
        FILE *fo = fopen(outPath, "wb");
        if (!fo) {
            string dir(outPath);
            size_t sl = dir.find_last_of('/');
            if (sl != string::npos) {
                dir = dir.substr(0, sl);
                string cmd = "mkdir -p '" + dir + "'";
                if (system(cmd.c_str()) != 0) fprintf(stderr, "mkdir failed\n");
            }
            fo = fopen(outPath, "wb");
        }
        if (!fo) { fprintf(stderr, "cannot open output %s\n", outPath); exit(1); }
        fwrite(txt.data(), 1, txt.size(), fo);
        fputc('\n', fo);
        fclose(fo);
    };
    auto writeFallback = [&](const char *msg) {
        fprintf(stderr, "FALLBACK (%s)\n", msg);
        string best;
        for (auto &ln : lines) if (ln.find(';') != string::npos) { best = ln; break; }
        if (best.empty()) best = "(0,1,2);";
        writeRaw(best);
        exit(0);
    };
    if (lines.empty()) { fprintf(stderr, "empty input\n"); return 1; }

    LABELS.clear();
    vector<PTree> pts(lines.size());
    vector<char> parsedOk(lines.size(), 0);
    int nParsed = 0;
    for (size_t li = 0; li < lines.size(); li++) {
        if (parseNewick(lines[li].c_str(), pts[li], LABELS)) { parsedOk[li] = 1; nParsed++; }
        else fprintf(stderr, "warn: failed to parse line %zu\n", li + 1);
    }
    if (nParsed == 0) writeFallback("no parseable tree");

    OUTID = -1;
    for (size_t t = 0; t < LABELS.size(); t++) if (LABELS[t] == "0") { OUTID = (int)t; break; }
    if (OUTID < 0) { fprintf(stderr, "warn: no outgroup label '0'; using first label\n"); OUTID = 0; }

    NTAX = (int)LABELS.size();
    if (NTAX > 256) writeFallback("more than 256 taxa");
    if (NTAX < 4) writeFallback("fewer than 4 taxa");
    bsInit(NTAX);

    vector<GTree> gts;
    gts.reserve(lines.size());
    for (size_t li = 0; li < lines.size(); li++) {
        if (!parsedOk[li]) continue;
        GTree gt;
        if (buildGTree(pts[li], OUTID, supT, lenT, gt)) gts.push_back(move(gt));
        else fprintf(stderr, "warn: cannot root tree from line %zu\n", li + 1);
    }
    vector<PTree>().swap(pts);
    if (gts.empty()) writeFallback("no usable gene tree");
    fprintf(stderr, "taxa=%d trees=%zu (parsed %d)\n", NTAX, gts.size(), nParsed);

    // ------------------------------------------------------------- build X --
    Pipeline P;
    P.gts = move(gts);
    BSMap xmap;
    xmap.init(16384);
    for (size_t g = 0; g < P.gts.size(); g++) {
        GTree &gt = P.gts[g];
        for (int v : gt.post) {
            if (v == gt.root) continue;
            BS side = gt.clade[v];
            int sz = bsPop(side);
            if (sz < 2 || sz > NTAX - 2) continue;
            if (bsTest(side, OUTID)) side = bsComp(side);
            int *slot = xmap.find(side);
            if (*slot == -1) {
                *slot = (int)P.splits.size();
                SplitRec r; r.side = side; r.sz = sz; r.freq = 1;
                P.splits.push_back(r);
            } else {
                P.splits[*slot].freq++;
            }
            gt.splitId[v] = *slot;
        }
    }
    int nX = (int)P.splits.size();
    fprintf(stderr, "|X|=%d collapse(sup<%g,len<%g)\n", nX, supT, lenT);

    for (int s = 0; s < nX; s++) if (P.splits[s].freq >= minFreq) P.kept.push_back(s);
    if (P.kept.empty()) writeFallback("no splits passed filter");
    if (minFreq > 1) fprintf(stderr, "kept=%zu (minFreq=%d)\n", P.kept.size(), minFreq);

    // ------------------------------------------------------------ sides -----
    P.smap.init(16384);
    auto addSide = [&](const BS &b, int sz) -> int {
        int *slot = P.smap.find(b);
        if (*slot == -1) {
            *slot = (int)P.sideBS.size();
            P.sideBS.push_back(b);
            P.sideSz.push_back(sz);
            P.sideM.push_back(0);
        }
        return *slot;
    };
    for (int s : P.kept) {
        int idx = addSide(P.splits[s].side, P.splits[s].sz);
        P.keptSideIdx.push_back(idx);
    }
    for (int t = 0; t < NTAX; t++) {
        if (t == OUTID) continue;
        BS b; bsClear(b); bsSet(b, t);
        addSide(b, 1);
    }
    BS ob; bsClear(ob); bsSet(ob, OUTID);
    BS S0 = bsComp(ob);
    P.S0idx = addSide(S0, NTAX - 1);
    int m = (int)P.sideBS.size();
    fprintf(stderr, "sides m=%d\n", m);

    // -------------------------------------------- M phase (full-split sigma) --
    vector<char> inG(nX, 0);
    for (size_t g = 0; g < P.gts.size(); g++) {
        const GTree &gt = P.gts[g];
        for (int v : gt.post) { int id = gt.splitId[v]; if (id >= 0) inG[id] = 1; }
        bool full = (gt.nTax == NTAX);
        for (size_t ki = 0; ki < P.kept.size(); ki++) {
            int s = P.kept[ki];
            i64 add;
            if (inG[s]) {
                int szX = full ? P.splits[s].sz : bsPop(bsAnd(P.splits[s].side, gt.mask));
                add = C2(szX) * C2(gt.nTax - szX);
            } else {
                add = NcountGen(gt, P.splits[s].side, bsComp(P.splits[s].side));
            }
            P.sideM[P.keptSideIdx[ki]] += add;
        }
        for (int v : gt.post) { int id = gt.splitId[v]; if (id >= 0) inG[id] = 0; }
        if ((g & 3) == 3 && nowSec() - T0 > GUARD) {
            fprintf(stderr, "time guard in M phase at tree %zu\n", g);
            break;
        }
    }

    // ------------------------------------------------------ N with cache ----
    // token: sideIdx*2 + compFlag (comp of a side); sizes derived
    auto tokSize = [&](long long tok) -> int {
        int idx = (int)(tok >> 1);
        return (tok & 1) ? (NTAX - P.sideSz[idx]) : P.sideSz[idx];
    };
    auto tokBS = [&](long long tok) -> BS {
        int idx = (int)(tok >> 1);
        return (tok & 1) ? bsComp(P.sideBS[idx]) : P.sideBS[idx];
    };
    U64i64Map ncache;
    ncache.init(1 << 15);
    long long nNcalls = 0;
    function<i64(long long, long long)> NcountAll = [&](long long tx, long long ty) -> i64 {
        if (tx > ty) swap(tx, ty);
        u64 key = (u64)tx * 1000003ULL + (u64)ty;
        const i64 *hit = ncache.get(key);
        if (hit) return *hit;
        i64 tot = 0;
        if (tokSize(tx) >= 2 && tokSize(ty) >= 2) {
            BS Xs = tokBS(tx), Ys = tokBS(ty);
            for (const GTree &gt : P.gts) tot += NcountGen(gt, Xs, Ys);
            nNcalls++;
        }
        *ncache.find(key) = tot;
        return tot;
    };

    // ---------------------------------------------------------------- DP ----
    int maxSz = NTAX - 1;
    bool dpAborted = false;

    auto levelScore = [&](int iA, int iB, int iS) -> i64 {
        long long tA = (long long)iA * 2, tB = (long long)iB * 2;
        long long tR = (long long)iS * 2 + 1; // R = V \ S
        i64 s = P.sideM[iA] + P.sideM[iB];
        s -= NcountAll(tA, tB);
        s -= NcountAll(tA, tR);
        s -= NcountAll(tB, tR);
        return s;
    };

    auto runDP = [&]() {
        dpAborted = false;
        int mNow = (int)P.sideBS.size();
        vector<vector<int>> buckets(maxSz + 1);
        for (int i = 0; i < mNow; i++) buckets[P.sideSz[i]].push_back(i);
        P.v.assign(mNow, -1);
        P.bestPair.assign(mNow, {-1, -1});
        for (int i = 0; i < mNow; i++) if (P.sideSz[i] == 1) P.v[i] = 0;
        for (int t = 2; t <= maxSz; t++) {
            for (int s1 = 1; s1 * 2 <= t; s1++) {
                int s2 = t - s1;
                if (s2 > maxSz) continue;
                const vector<int> &B1 = buckets[s1];
                if (B1.empty()) continue;
                const vector<int> &B2 = buckets[s2];
                if (s1 != s2 && B2.empty()) continue;
                if (s1 == s2) {
                    for (size_t ii = 0; ii < B1.size(); ii++) {
                        int i = B1[ii];
                        if (P.v[i] < 0) continue;
                        const BS &bi = P.sideBS[i];
                        for (size_t jj = ii + 1; jj < B1.size(); jj++) {
                            int j = B1[jj];
                            if (P.v[j] < 0) continue;
                            if (!bsDisjoint(bi, P.sideBS[j])) continue;
                            int idx = P.smap.get(bsOr(bi, P.sideBS[j]));
                            if (idx < 0) continue;
                            P.nLevels++;
                            i64 cand = P.v[i] + P.v[j] + levelScore(i, j, idx);
                            if (cand > P.v[idx]) { P.v[idx] = cand; P.bestPair[idx] = {i, j}; }
                        }
                    }
                } else {
                    for (int i : B1) {
                        if (P.v[i] < 0) continue;
                        const BS &bi = P.sideBS[i];
                        for (int j : B2) {
                            if (P.v[j] < 0) continue;
                            if (!bsDisjoint(bi, P.sideBS[j])) continue;
                            int idx = P.smap.get(bsOr(bi, P.sideBS[j]));
                            if (idx < 0) continue;
                            P.nLevels++;
                            i64 cand = P.v[i] + P.v[j] + levelScore(i, j, idx);
                            if (cand > P.v[idx]) { P.v[idx] = cand; P.bestPair[idx] = {i, j}; }
                        }
                    }
                }
            }
            if ((t & 7) == 0 && nowSec() - T0 > 95.0) {
                dpAborted = true;
                fprintf(stderr, "DP aborted (time) at t=%d\n", t);
                return;
            }
        }
    };
    runDP();
    fprintf(stderr, "DP score=%lld levels=%lld Ncalls=%lld (%.1fs)\n",
            (long long)P.v[P.S0idx], P.nLevels, nNcalls, nowSec() - T0);

    // debug mode: score an arbitrary tree file under the quartet objective
    if (scoreTreePath) {
        FILE *f = fopen(scoreTreePath, "rb");
        if (!f) { fprintf(stderr, "cannot open score-tree file\n"); return 1; }
        string all;
        char buf[1 << 16];
        while (size_t rd = fread(buf, 1, sizeof(buf), f)) all.append(buf, rd);
        fclose(f);
        PTree qt;
        vector<string> qlabels = LABELS;
        if (!parseNewick(all.c_str(), qt, qlabels)) { fprintf(stderr, "score-tree: parse fail\n"); return 1; }
        GTree qg;
        if (!buildGTree(qt, OUTID, 0, 0, qg)) { fprintf(stderr, "score-tree: build fail\n"); return 1; }
        BSi64Map mcache; mcache.init(4096);
        U64i64Map ncache2; ncache2.init(4096);
        // generic M and N evaluators with cache keyed by bitset pair
        BSMap bmap; bmap.init(4096);
        vector<BS> bsets;
        auto bidx = [&](const BS &b) -> int {
            int *sl = bmap.find(b);
            if (*sl == -1) { *sl = (int)bsets.size(); bsets.push_back(b); }
            return *sl;
        };
        function<i64(int,int)> Npair = [&](int ix, int iy) -> i64 {
            if (ix > iy) swap(ix, iy);
            u64 key = (u64)ix * 1000003ULL + (u64)iy;
            const i64 *h = ncache2.get(key);
            if (h) return *h;
            i64 tot = 0;
            int px = bsPop(bsets[ix]), py = bsPop(bsets[iy]);
            if (px >= 2 && py >= 2) {
                for (const GTree &gt : P.gts) tot += NcountGen(gt, bsets[ix], bsets[iy]);
            }
            *ncache2.find(key) = tot;
            return tot;
        };
        i64 total = 0;
        int nlev = 0;
        // levels: every internal node of qg except the root u0 gives (S=clade(u); C1|C2...),
        // plus the top level S0 = V\{0} split by root's two non-outgroup children.
        // For multifurcations, pick any binary resolution of the level? Input tree is binary.
        // root children: leaf0 + A1 + A2 (binary tree assumption)
        {
            int o = qg.chOff[qg.root], oe = qg.chOff[qg.root + 1];
            vector<int> bigs;
            for (int j = o; j < oe; j++) {
                int c = qg.ch[j];
                if (!bsTest(qg.clade[c], OUTID)) bigs.push_back(c);
            }
            if (bigs.size() != 2) { fprintf(stderr, "score-tree: root must have outgroup leaf + 2 subtrees\n"); return 1; }
            int iA = bidx(qg.clade[bigs[0]]), iB = bidx(qg.clade[bigs[1]]);
            int iS = bidx(S0);
            int iR = bidx(ob);
            i64 ls = Npair(iA, bsPop(bsets[iA]) ? iA : iA) ; (void)ls;
            i64 MA = Npair(iA, bidx(bsComp(bsets[iA])));
            i64 MB = Npair(iB, bidx(bsComp(bsets[iB])));
            i64 lv = MA + MB - Npair(iA, iB) - Npair(iA, iR) - Npair(iB, iR);
            total += lv; nlev++;
        }
        for (int v : qg.post) {
            if (v == qg.root) continue;
            int o = qg.chOff[v], oe = qg.chOff[v + 1];
            if (oe - o != 2) { fprintf(stderr, "score-tree: non-binary node\n"); return 1; }
            int c1 = qg.ch[o], c2 = qg.ch[o+1];
            BS A = qg.clade[c1], B = qg.clade[c2], S = qg.clade[v], R = bsComp(S);
            int iA = bidx(A), iB = bidx(B), iS = bidx(S), iR = bidx(R);
            int iAc = bidx(bsComp(A)), iBc = bidx(bsComp(B));
            i64 MA = Npair(iA, iAc);
            i64 MB = Npair(iB, iBc);
            i64 lv = MA + MB - Npair(iA, iB) - Npair(iA, iR) - Npair(iB, iR);
            total += lv; nlev++;
        }
        fprintf(stderr, "SCORE-TREE %s: |Q_G∩Q_T| = %lld over %d levels (%.1fs)\n",
                scoreTreePath, (long long)total, nlev, nowSec() - T0);
        if (P.v[P.S0idx] >= 0)
            fprintf(stderr, "DP score for comparison = %lld\n", (long long)P.v[P.S0idx]);
        return 0;
    }
    if (P.v[P.S0idx] < 0) writeFallback("DP found no tree");

    // ------------------------------------- working tree (for NNI + output) ---
    struct WNode { int ch[3]; int nch; int par; BS clade; char isLeaf; int taxon; };
    vector<WNode> wn;
    int rootW = -1;
    int Wn = 0;
    function<int(int)> buildRec = [&](int sidx) -> int {
        if (P.sideSz[sidx] == 1) {
            int t = -1;
            for (int tt = 0; tt < NTAX; tt++) if (bsTest(P.sideBS[sidx], tt)) { t = tt; break; }
            WNode w; w.ch[0]=w.ch[1]=w.ch[2]=-1; w.nch=0; w.par=-1;
            bsClear(w.clade); bsSet(w.clade, t); w.isLeaf=1; w.taxon=t;
            wn.push_back(w);
            return (int)wn.size() - 1;
        }
        int a = buildRec(P.bestPair[sidx].first);
        int b = buildRec(P.bestPair[sidx].second);
        WNode w; w.ch[0]=a; w.ch[1]=b; w.ch[2]=-1; w.nch=2; w.par=-1; w.isLeaf=0; w.taxon=-1;
        w.clade = bsOr(wn[a].clade, wn[b].clade);
        wn.push_back(w);
        int id = (int)wn.size() - 1;
        wn[a].par = id; wn[b].par = id;
        return id;
    };
    auto buildW = [&]() {
        wn.clear();
        wn.reserve(2 * NTAX + 2);
        rootW = -1;
        int a = buildRec(P.bestPair[P.S0idx].first);
        int b = buildRec(P.bestPair[P.S0idx].second);
        WNode ol; ol.ch[0]=ol.ch[1]=ol.ch[2]=-1; ol.nch=0; ol.par=-1;
        bsClear(ol.clade); bsSet(ol.clade, OUTID); ol.isLeaf=1; ol.taxon=OUTID;
        wn.push_back(ol);
        int o = (int)wn.size() - 1;
        WNode r; r.ch[0]=o; r.ch[1]=a; r.ch[2]=b; r.nch=3; r.par=-1; r.isLeaf=0; r.taxon=-1;
        bsClear(r.clade);
        for (int t = 0; t < NTAX; t++) bsSet(r.clade, t);
        wn.push_back(r);
        rootW = (int)wn.size() - 1;
        wn[o].par = rootW; wn[a].par = rootW; wn[b].par = rootW;
        Wn = (int)wn.size();
    };
    buildW();

    // evaluation cache, warmed with precomputed M values for X sides
    EvalCache EC;
    EC.init(P.gts);
    int iS0 = EC.idx(S0);
    for (size_t ki = 0; ki < P.kept.size(); ki++) {
        int s = P.kept[ki];
        int i1 = EC.idx(P.splits[s].side);
        int i2 = EC.idx(bsComp(P.splits[s].side));
        EC.putN(i1, i2, P.sideM[P.keptSideIdx[ki]]);
    }
    auto nodeLS = [&](int u) -> i64 {
        if (u == rootW) {
            int a = -1, b = -1;
            for (int j = 0; j < wn[u].nch; j++) {
                int c = wn[u].ch[j];
                if (wn[c].isLeaf && wn[c].taxon == OUTID) continue;
                if (a < 0) a = c; else b = c;
            }
            return EC.LS(EC.idx(wn[a].clade), EC.idx(wn[b].clade), iS0);
        }
        int c1 = wn[u].ch[0], c2 = wn[u].ch[1];
        return EC.LS(EC.idx(wn[c1].clade), EC.idx(wn[c2].clade), EC.idx(wn[u].clade));
    };
    auto scoreW = [&]() -> i64 {
        i64 s = 0;
        for (int u = 0; u < Wn; u++) if (!wn[u].isLeaf) s += nodeLS(u);
        return s;
    };
    i64 curScore = scoreW();
    fprintf(stderr, "working tree score=%lld (DP %lld) (%.1fs)\n",
            (long long)curScore, (long long)P.v[P.S0idx], nowSec() - T0);

    // ------------------------------------------------------------- NNI -------
    auto runNNI = [&]() {
        if (nniRounds <= 0) return;
        for (int round = 1; round <= nniRounds; round++) {
            if (nowSec() - T0 > GUARD) { fprintf(stderr, "time guard stops NNI at round %d\n", round); break; }
            i64 bestD = 0; int bestV = -1, bestMode = -1;
            for (int x = 0; x < Wn; x++) {
                if (x == rootW || wn[x].isLeaf) continue;
                int u = wn[x].par;
                int c1 = wn[x].ch[0], c2 = wn[x].ch[1];
                int o = -1;
                for (int j = 0; j < wn[u].nch; j++) {
                    int c = wn[u].ch[j];
                    if (c == x) continue;
                    if (u == rootW && wn[c].isLeaf && wn[c].taxon == OUTID) continue;
                    o = c; break;
                }
                if (o < 0) continue;
                int iSu = (u == rootW) ? iS0 : EC.idx(wn[u].clade);
                int io = EC.idx(wn[o].clade);
                int iC1 = EC.idx(wn[c1].clade), iC2 = EC.idx(wn[c2].clade);
                int iV = EC.idx(wn[x].clade);
                i64 old = EC.LS(iV, io, iSu) + EC.LS(iC1, iC2, iV);
                int iX1 = EC.idx(bsOr(wn[o].clade, wn[c2].clade));
                int iX2 = EC.idx(bsOr(wn[o].clade, wn[c1].clade));
                i64 d1 = EC.LS(iX1, iC1, iSu) + EC.LS(io, iC2, iX1) - old;
                i64 d2 = EC.LS(iX2, iC2, iSu) + EC.LS(io, iC1, iX2) - old;
                if (d1 > bestD) { bestD = d1; bestV = x; bestMode = 1; }
                if (d2 > bestD) { bestD = d2; bestV = x; bestMode = 2; }
                // first-improvement: apply immediately and rescan
                if (bestD > 0) break;
            }
            if (bestV < 0 || bestD <= 0) {
                fprintf(stderr, "NNI converged after %d improving rounds, score=%lld\n", round - 1, (long long)curScore);
                break;
            }
            int x = bestV;
            int u = wn[x].par;
            int c1 = wn[x].ch[0], c2 = wn[x].ch[1];
            int o = -1, oPos = -1;
            for (int j = 0; j < wn[u].nch; j++) {
                int c = wn[u].ch[j];
                if (c == x) continue;
                if (u == rootW && wn[c].isLeaf && wn[c].taxon == OUTID) continue;
                o = c; oPos = j; break;
            }
            int swapC = (bestMode == 1) ? c1 : c2;
            int keepC = (bestMode == 1) ? c2 : c1;
            int swapPos = (wn[x].ch[0] == swapC) ? 0 : 1;
            wn[u].ch[oPos] = swapC;
            wn[x].ch[swapPos] = o;
            wn[swapC].par = u;
            wn[o].par = x;
            wn[x].clade = bsOr(wn[o].clade, wn[keepC].clade);
            curScore += bestD;
            if (round <= 10 || (round % 50) == 0)
                fprintf(stderr, "NNI round %d: delta=%lld score=%lld (%.1fs)\n", round, (long long)bestD, (long long)curScore, nowSec() - T0);
        }
    };
    runNNI();

    // ------- re-expansion: add the current tree's splits to X, redo DP + NNI --
    for (int rex = 1; rex <= rexpand; rex++) {
        if (nowSec() - T0 > GUARD - 15) { fprintf(stderr, "time guard: no re-expansion\n"); break; }
        int added = 0;
        for (int x = 0; x < Wn; x++) {
            if (wn[x].isLeaf) continue;
            BS sd = wn[x].clade;
            int sz = bsPop(sd);
            if (sz < 2 || sz > NTAX - 2) continue;
            if (bsTest(sd, OUTID)) { sd = bsComp(sd); sz = NTAX - sz; }
            if (P.smap.get(sd) >= 0) continue;
            i64 Mv = EC.Npair(EC.idx(sd), EC.idx(bsComp(sd)));
            int *slot = P.smap.find(sd);
            *slot = (int)P.sideBS.size();
            P.sideBS.push_back(sd);
            P.sideSz.push_back(sz);
            P.sideM.push_back(Mv);
            added++;
        }
        if (added == 0) { fprintf(stderr, "rex %d: no new splits, stop\n", rex); break; }
        fprintf(stderr, "rex %d: +%d splits (%.1fs)\n", rex, added, nowSec() - T0);
        vector<i64> oldV = P.v;
        vector<pair<int,int>> oldBP = P.bestPair;
        i64 oldScore = curScore;
        runDP();
        if (dpAborted || P.v[P.S0idx] < 0 || P.v[P.S0idx] < oldScore) {
            P.v = move(oldV); P.bestPair = move(oldBP);
            fprintf(stderr, "rex %d: DP not better, keep previous tree\n", rex);
            break;
        }
        buildW();
        curScore = scoreW();
        fprintf(stderr, "rex %d: DP=%lld tree=%lld (%.1fs)\n", rex, (long long)P.v[P.S0idx], (long long)curScore, nowSec() - T0);
        runNNI();
        fprintf(stderr, "rex %d: after NNI score=%lld (%.1fs)\n", rex, (long long)curScore, nowSec() - T0);
    }


    // --------------------------------------------------------- write output --
    string out;
    out.reserve(16 * 1024);
    function<void(int)> emit = [&](int x) {
        if (wn[x].isLeaf) { out += LABELS[wn[x].taxon]; return; }
        out += "(";
        for (int j = 0; j < wn[x].nch; j++) {
            if (j) out += ",";
            emit(wn[x].ch[j]);
        }
        out += ")";
    };
    out += "(";
    emit(wn[rootW].ch[0]);
    out += ",";
    emit(wn[rootW].ch[1]);
    out += ",";
    emit(wn[rootW].ch[2]);
    out += ");";
    writeRaw(out);
    fprintf(stderr, "done in %.2fs; wrote %zu bytes; final score=%lld\n", nowSec() - T0, out.size(), (long long)curScore);
    return 0;
}

// ======================= selftest part 2 implementation ====================
// Runs the same DP as main() on small random instances and compares against
// brute-force enumeration of all binary trees (restricted to splits in X).
static void runSmallDP(vector<GTree> &gts, int minFreq, i64 &dpScore, vector<BS> &dpSplits, int n);

static int selftestDP() {
    srand(987654);
    int fails = 0;
    for (int trial = 0; trial < 60; trial++) {
        int n = 5 + rand() % 3; // 5..7 taxa
        int k = 1 + rand() % 3; // 1..3 gene trees
        bsInit(n);
        LABELS.clear();
        for (int t = 0; t < n; t++) LABELS.push_back(to_string(t));
        // build random gene trees as Newick strings (rooted at 0's neighbour)
        vector<string> nwcs;
        for (int gi = 0; gi < k; gi++) {
            int cap = 2 * n + 2;
            vector<vector<int>> adj2(cap);
            auto addEdge = [&](int u, int v) { adj2[u].push_back(v); adj2[v].push_back(u); };
            auto delEdge = [&](int u, int v) {
                adj2[u].erase(remove(adj2[u].begin(), adj2[u].end(), v), adj2[u].end());
            };
            int i0 = n;
            addEdge(0, i0); addEdge(1, i0); addEdge(2, i0);
            int nInternals = i0 + 1;
            for (int leaf = 3; leaf < n; leaf++) {
                vector<pair<int, int>> edges;
                for (int u = 0; u < nInternals; u++)
                    for (int v : adj2[u]) if (u < v) edges.push_back({u, v});
                auto e = edges[rand() % edges.size()];
                int mid = nInternals++;
                delEdge(e.first, e.second);
                delEdge(e.second, e.first);
                addEdge(e.first, mid); addEdge(e.second, mid); addEdge(leaf, mid);
            }
            int r0 = adj2[0][0];
            string nwk;
            function<void(int, int)> rec2 = [&](int v, int p) {
                if (v < n) { nwk += to_string(v); return; }
                nwk += "(";
                bool first = true;
                for (int w : adj2[v]) {
                    if (w == p) continue;
                    if (!first) nwk += ",";
                    first = false;
                    rec2(w, v);
                }
                nwk += ")";
            };
            nwk += "(0";
            for (int w : adj2[r0]) {
                if (w == 0) continue;
                nwk += ",";
                rec2(w, r0);
            }
            nwk += ");";
            nwcs.push_back(nwk);
        }
        vector<GTree> gts;
        bool ok = true;
        for (auto &nwk : nwcs) {
            PTree pt;
            if (!parseNewick(nwk.c_str(), pt, LABELS)) { ok = false; break; }
            GTree gt;
            if (!buildGTree(pt, 0, 0, 0, gt)) { ok = false; break; }
            gts.push_back(move(gt));
        }
        if (!ok || gts.empty()) { fails++; continue; }

        i64 dpScore = -1;
        vector<BS> dpSplits;
        runSmallDP(gts, 1, dpScore, dpSplits, n);

        // ---- brute force: enumerate all binary trees on n leaves
        // collect X
        BSMap xmap; xmap.init(256);
        vector<BS> xs;
        for (auto &gt : gts)
            for (int v : gt.post) {
                if (v == gt.root) continue;
                BS side = gt.clade[v];
                int sz = bsPop(side);
                if (sz < 2 || sz > n - 2) continue;
                if (bsTest(side, 0)) side = bsComp(side);
                int *slot = xmap.find(side);
                if (*slot == -1) { *slot = (int)xs.size(); xs.push_back(side); }
            }
        // enumerate trees via leaf insertion; keep split sets
        // tree rep: parent/children arrays; here simply store split sets
        vector<vector<BS>> trees; // all binary trees' nontrivial split sets (canonical, 0 not in side)
        {
            // start: n=3 star with leaves 0,1,2 -> no nontrivial splits
            // represent trees as adjacency for insertion
            struct T { vector<vector<int>> adj; int nn; };
            vector<T> cur;
            T t0; t0.nn = n + 1; t0.adj.assign(2 * n + 2, {});
            t0.adj[0].push_back(n); t0.adj[1].push_back(n); t0.adj[2].push_back(n);
            t0.adj[n].push_back(0); t0.adj[n].push_back(1); t0.adj[n].push_back(2);
            cur.push_back(t0);
            for (int leaf = 3; leaf < n; leaf++) {
                vector<T> nxt;
                for (auto &t : cur) {
                    vector<pair<int,int>> edges;
                    for (int u = 0; u < t.nn; u++)
                        for (int v : t.adj[u]) if (u < v) edges.push_back({u, v});
                    for (auto &e : edges) {
                        T nt; nt.adj.assign(2 * n + 2, {}); nt.nn = t.nn + 1;
                        for (int u = 0; u < t.nn; u++) for (int v : t.adj[u]) nt.adj[u].push_back(v);
                        int mid = t.nn;
                        auto rm = [&](int a, int b) {
                            nt.adj[a].erase(remove(nt.adj[a].begin(), nt.adj[a].end(), b), nt.adj[a].end());
                        };
                        rm(e.first, e.second); rm(e.second, e.first);
                        nt.adj[e.first].push_back(mid); nt.adj[mid].push_back(e.first);
                        nt.adj[e.second].push_back(mid); nt.adj[mid].push_back(e.second);
                        nt.adj[leaf].push_back(mid); nt.adj[mid].push_back(leaf);
                        nxt.push_back(nt);
                    }
                }
                cur = move(nxt);
            }
            // convert to split sets
            for (auto &t : cur) {
                vector<BS> sp;
                // root at neighbour of leaf 0
                int r0 = t.adj[0][0];
                // DFS
                vector<int> par(t.nn, -1);
                vector<BS> cl(t.nn);
                vector<int> order;
                vector<int> stk = {r0};
                par[r0] = r0;
                while (!stk.empty()) {
                    int v = stk.back(); stk.pop_back();
                    order.push_back(v);
                    for (int w : t.adj[v]) if (w != par[v]) { par[w] = v; stk.push_back(w); }
                }
                for (int ii = (int)order.size() - 1; ii >= 0; ii--) {
                    int v = order[ii];
                    bsClear(cl[v]);
                    bool leaf = true;
                    for (int w : t.adj[v]) if (w != par[v]) { cl[v] = bsOr(cl[v], cl[w]); leaf = false; }
                    if (leaf) bsSet(cl[v], v < n ? v : 0);
                }
                for (int v = 0; v < t.nn; v++) {
                    if (v == r0) continue;
                    BS s = cl[v];
                    int sz = bsPop(s);
                    if (sz < 2 || sz > n - 2) continue;
                    if (bsTest(s, 0)) s = bsComp(s);
                    sp.push_back(s);
                }
                trees.push_back(sp);
            }
        }
        // quartet topologies per gene tree: for each 4-subset, which pairing
        auto topOf = [&](const GTree &gt, int q[4]) -> int {
            // returns 0: (01|23), 1: (02|13), 2: (12|03)... use pair indices
            for (int v = 0; v < gt.nNodes; v++) {
                if (v == gt.root) continue;
                const BS &s = gt.clade[v];
                bool in[4];
                int cnt = 0;
                for (int j = 0; j < 4; j++) { in[j] = bsTest(s, q[j]); cnt += in[j]; }
                if (cnt == 2) {
                    // which pairing
                    if ((in[0] && in[1]) || (in[2] && in[3])) return 0; // q0q1 | q2q3
                    if ((in[0] && in[2]) || (in[1] && in[3])) return 1; // q0q2 | q1q3
                    return 2;                                            // q0q3 | q1q2
                }
            }
            return -1; // unresolved
        };
        vector<vector<int>> tops; // [quartet][gene tree]
        vector<int> q4(4);
        vector<array<int,4>> quads;
        for (int a = 0; a < n; a++) for (int b = a+1; b < n; b++) for (int c = b+1; c < n; c++) for (int d = c+1; d < n; d++)
            quads.push_back({a,b,c,d});
        tops.assign(quads.size(), vector<int>(gts.size(), -1));
        for (size_t qi = 0; qi < quads.size(); qi++)
            for (size_t g = 0; g < gts.size(); g++)
                tops[qi][g] = topOf(gts[g], quads[qi].data());

        i64 bestBF = -1;
        for (auto &sp : trees) {
            // splits(sp) must be subset of X
            bool allowed = true;
            for (auto &s : sp) if (xmap.get(s) < 0) { allowed = false; break; }
            if (!allowed) continue;
            // score = sum over quartets, over gene trees: topology match
            // topology of quartet in candidate tree: use its splits
            i64 sc = 0;
            for (size_t qi = 0; qi < quads.size(); qi++) {
                int q[4] = {quads[qi][0], quads[qi][1], quads[qi][2], quads[qi][3]};
                int tt = -1;
                for (auto &s : sp) {
                    bool in[4]; int cnt = 0;
                    for (int j = 0; j < 4; j++) { in[j] = bsTest(s, q[j]); cnt += in[j]; }
                    if (cnt == 2) {
                        if ((in[0] && in[1]) || (in[2] && in[3])) { tt = 0; break; }
                        if ((in[0] && in[2]) || (in[1] && in[3])) { tt = 1; break; }
                        tt = 2; break;
                    }
                }
                for (size_t g = 0; g < gts.size(); g++)
                    if (tops[qi][g] == tt && tt >= 0) sc++;
            }
            if (sc > bestBF) bestBF = sc;
        }
        if (dpScore != bestBF) {
            fprintf(stderr, "SELFTEST DP MISMATCH trial=%d n=%d k=%d dp=%lld bf=%lld\n",
                    trial, n, k, (long long)dpScore, (long long)bestBF);
            fails++;
        }
    }
    fprintf(stderr, "selftest DP done, fails=%d\n", fails);
    return fails;
}

// mini pipeline replica for selftest (same math as main)
static void runSmallDP(vector<GTree> &gts, int minFreq, i64 &dpScore, vector<BS> &dpSplits, int n) {
    (void)minFreq;
    OUTID = 0;
    BSMap xmap; xmap.init(256);
    vector<SplitRec> splits;
    for (auto &gt : gts) {
        for (int v : gt.post) {
            if (v == gt.root) continue;
            BS side = gt.clade[v];
            int sz = bsPop(side);
            if (sz < 2 || sz > n - 2) continue;
            if (bsTest(side, OUTID)) side = bsComp(side);
            int *slot = xmap.find(side);
            if (*slot == -1) {
                *slot = (int)splits.size();
                SplitRec r; r.side = side; r.sz = sz; r.freq = 1;
                splits.push_back(r);
            } else splits[*slot].freq++;
        }
    }
    // sides
    vector<BS> sideBS; vector<int> sideSz; vector<i64> sideM;
    BSMap smap; smap.init(256);
    auto addSide = [&](const BS &b, int sz) -> int {
        int *slot = smap.find(b);
        if (*slot == -1) {
            *slot = (int)sideBS.size();
            sideBS.push_back(b); sideSz.push_back(sz); sideM.push_back(0);
        }
        return *slot;
    };
    for (size_t si = 0; si < splits.size(); si++) addSide(splits[si].side, splits[si].sz);
    for (int t = 0; t < n; t++) {
        if (t == OUTID) continue;
        BS b; bsClear(b); bsSet(b, t);
        addSide(b, 1);
    }
    BS ob; bsClear(ob); bsSet(ob, OUTID);
    int S0idx = addSide(bsComp(ob), n - 1);
    int m = (int)sideBS.size();
    // M values
    for (size_t si = 0; si < splits.size(); si++) {
        int idx = smap.get(splits[si].side);
        i64 tot = 0;
        BS cp = bsComp(splits[si].side);
        for (auto &gt : gts) tot += NcountGen(gt, splits[si].side, cp);
        sideM[idx] = tot;
    }
    auto Nall = [&](int iA, int iB, bool bComp) -> i64 {
        BS Ys = bComp ? bsComp(sideBS[iB]) : sideBS[iB];
        if (bsPop(sideBS[iA]) < 2 || bsPop(Ys) < 2) return 0;
        i64 tot = 0;
        for (auto &gt : gts) tot += NcountGen(gt, sideBS[iA], Ys);
        return tot;
    };
    // DP
    int maxSz = n - 1;
    vector<vector<int>> buckets(maxSz + 1);
    for (int i = 0; i < m; i++) buckets[sideSz[i]].push_back(i);
    vector<i64> v(m, -1);
    vector<pair<int,int>> bestPair(m, {-1,-1});
    for (int i = 0; i < m; i++) if (sideSz[i] == 1) v[i] = 0;
    for (int t = 2; t <= maxSz; t++) {
        for (int s1 = 1; s1 * 2 <= t; s1++) {
            int s2 = t - s1;
            if (s2 > maxSz || buckets[s1].empty()) continue;
            if (s1 != s2 && buckets[s2].empty()) continue;
            auto tryPair = [&](int i, int j) {
                if (v[i] < 0 || v[j] < 0) return;
                if (!bsDisjoint(sideBS[i], sideBS[j])) return;
                BS S = bsOr(sideBS[i], sideBS[j]);
                int idx = smap.get(S);
                if (idx < 0) return;
                i64 ls = sideM[i] + sideM[j];
                ls -= Nall(i, j, false);          // N(A,B)
                ls -= Nall(i, idx, true);         // N(A,R)
                ls -= Nall(j, idx, true);         // N(B,R)
                i64 cand = v[i] + v[j] + ls;
                if (cand > v[idx]) { v[idx] = cand; bestPair[idx] = {i, j}; }
            };
            if (s1 == s2) {
                auto &B = buckets[s1];
                for (size_t ii = 0; ii < B.size(); ii++)
                    for (size_t jj = ii + 1; jj < B.size(); jj++)
                        tryPair(B[ii], B[jj]);
            } else {
                for (int i : buckets[s1]) for (int j : buckets[s2]) tryPair(i, j);
            }
        }
    }
    dpScore = v[S0idx];
    dpSplits.clear();
}
