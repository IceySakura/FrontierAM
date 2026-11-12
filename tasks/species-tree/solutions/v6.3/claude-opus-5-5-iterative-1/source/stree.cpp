// Species tree inference from estimated gene trees: weighted-quartet DP (ASTRAL-like)
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;
static const int MW = 4;  // up to 256 taxa
static int NT = 0;        // number of taxa
static int NWORDS = 4;

struct Bits {
    u64 w[MW];
    Bits() { memset(w, 0, sizeof(w)); }
    inline void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    inline bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
    inline int count() const { int c = 0; for (int k = 0; k < MW; k++) c += __builtin_popcountll(w[k]); return c; }
    inline bool operator==(const Bits& o) const { for (int k = 0; k < MW; k++) if (w[k] != o.w[k]) return false; return true; }
    inline Bits operator|(const Bits& o) const { Bits r; for (int k = 0; k < MW; k++) r.w[k] = w[k] | o.w[k]; return r; }
    inline Bits operator&(const Bits& o) const { Bits r; for (int k = 0; k < MW; k++) r.w[k] = w[k] & o.w[k]; return r; }
    inline Bits operator^(const Bits& o) const { Bits r; for (int k = 0; k < MW; k++) r.w[k] = w[k] ^ o.w[k]; return r; }
    inline bool subsetOf(const Bits& o) const { for (int k = 0; k < MW; k++) if (w[k] & ~o.w[k]) return false; return true; }
    inline int first() const { for (int k = 0; k < MW; k++) if (w[k]) return k * 64 + __builtin_ctzll(w[k]); return -1; }
};
struct BitsHash {
    size_t operator()(const Bits& b) const {
        u64 h = 0x9E3779B97F4A7C15ULL;
        for (int k = 0; k < MW; k++) { h ^= b.w[k] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xff51afd7ed558ccdULL; }
        return h ^ (h >> 29);
    }
};

static double now_s() {
    static auto t0 = chrono::steady_clock::now();
    return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
}

// ---------------- Newick parsing ----------------
struct PNode { vector<int> ch; int par = -1; string label; double len = 0; double sup = NAN; bool hasLen = false; };
struct PTree { vector<PNode> nodes; int root = 0; };

static bool parseNewick(const string& s, PTree& t) {
    t.nodes.clear();
    size_t pos = 0, n = s.size();
    auto skipws = [&]() { while (pos < n && (isspace((unsigned char)s[pos]) || s[pos] == '[')) { if (s[pos] == '[') { while (pos < n && s[pos] != ']') pos++; if (pos < n) pos++; } else pos++; } };
    auto readLabel = [&]() -> string {
        skipws();
        string r;
        if (pos < n && s[pos] == '\'') {
            pos++;
            while (pos < n) { if (s[pos] == '\'') { if (pos + 1 < n && s[pos + 1] == '\'') { r += '\''; pos += 2; } else { pos++; break; } } else r += s[pos++]; }
            return r;
        }
        while (pos < n && !strchr("(),:;[", s[pos]) && !isspace((unsigned char)s[pos])) r += s[pos++];
        return r;
    };
    vector<int> stack;
    t.nodes.push_back(PNode()); int cur = 0; t.root = 0;
    skipws();
    if (pos >= n || s[pos] != '(') {  // single leaf tree?
        return false;
    }
    // iterative parse
    // cur is the node being built
    pos++;  // consume '('
    stack.push_back(0);
    // create first child
    auto newChild = [&](int p) { PNode nd; nd.par = p; t.nodes.push_back(nd); int id = (int)t.nodes.size() - 1; t.nodes[p].ch.push_back(id); return id; };
    cur = newChild(0);
    while (true) {
        skipws();
        if (pos >= n) return false;
        char c = s[pos];
        if (c == '(') { pos++; stack.push_back(cur); cur = newChild(cur); continue; }
        // leaf or end of internal: read label for cur
        if (c != ')' && c != ',' && c != ':') {
            string lab = readLabel();
            t.nodes[cur].label = lab;
        }
        skipws();
        if (pos < n && s[pos] == ':') {
            pos++; skipws(); size_t st = pos;
            while (pos < n && !strchr("(),:;[", s[pos]) && !isspace((unsigned char)s[pos])) pos++;
            double L = atof(s.substr(st, pos - st).c_str());
            if (!std::isfinite(L) || L < 0) L = 0;
            t.nodes[cur].len = L; t.nodes[cur].hasLen = true;
        }
        skipws();
        if (pos >= n) return false;
        c = s[pos];
        if (c == ',') { pos++; int p = stack.back(); cur = newChild(p); continue; }
        if (c == ')') {
            pos++;
            cur = stack.back(); stack.pop_back();
            // internal label
            skipws();
            if (pos < n && !strchr("(),:;[", s[pos])) {
                string lab = readLabel();
                char* e = nullptr; double v = strtod(lab.c_str(), &e);
                if (e && *e == 0 && !lab.empty() && std::isfinite(v)) t.nodes[cur].sup = v;
            }
            if (cur == 0) {
                skipws();
                if (pos < n && s[pos] == ':') { pos++; while (pos < n && !strchr("(),;", s[pos])) pos++; }
                skipws();
                if (pos < n && s[pos] != ';') return false;
                return true;
            }
            continue;
        }
        if (c == ';') return stack.empty();
        return false;
    }
}

// ---------------- Gene tree (binary rooted, postorder numbered) ----------------
struct Gene {
    int nn = 0;
    vector<int> L, R, par, taxon, sz;
    vector<double> om, keep, len, D, sup, h, Dc, DU;
    double gw = 1.0, totlen = 0;
    vector<int> leafNode;  // taxon -> node
    int root = -1;
};

static map<string, int> taxIndex;
static vector<string> taxName;

// Build gene from parsed tree. support->omega mapping is applied later.
static bool buildGene(const PTree& t, Gene& g) {
    // Collect structure: unroot/binarize. First compute children lists, handle degree-2 nodes and polytomies.
    int N = (int)t.nodes.size();
    // Create working arrays
    struct WN { vector<int> ch; int taxon = -1; double len = 0, sup = NAN; };
    vector<WN> w(N);
    for (int i = 0; i < N; i++) {
        w[i].ch = t.nodes[i].ch; w[i].len = t.nodes[i].len; w[i].sup = t.nodes[i].sup;
        if (t.nodes[i].ch.empty()) {
            auto it = taxIndex.find(t.nodes[i].label);
            if (it == taxIndex.end()) return false;
            w[i].taxon = it->second;
        }
    }
    // contract internal nodes with one child (non-root)
    function<int(int)> fix = [&](int v) -> int {
        // returns node id representing v after contraction
        while (w[v].taxon < 0 && w[v].ch.size() == 1) {
            int c = w[v].ch[0];
            w[c].len += w[v].len;
            if (std::isnan(w[c].sup)) w[c].sup = w[v].sup; else if (!std::isnan(w[v].sup)) w[c].sup = max(w[c].sup, w[v].sup);
            v = c;
        }
        for (auto& c : w[v].ch) c = fix(c);
        return v;
    };
    int root = fix(t.root);
    // if root has 2 children: unrooted edge split in two; move support to one side
    if (w[root].ch.size() == 2) {
        int a = w[root].ch[0], b = w[root].ch[1];
        double s = NAN;
        if (!std::isnan(w[a].sup)) s = w[a].sup;
        if (!std::isnan(w[b].sup)) s = std::isnan(s) ? w[b].sup : max(s, w[b].sup);
        // keep combined edge on the internal one
        if (w[a].taxon < 0) { w[a].sup = s; w[a].len += w[b].len; w[b].len = 0; w[b].sup = -1; }
        else { w[b].sup = s; w[b].len += w[a].len; w[a].len = 0; w[a].sup = -1; }
    }
    // binarize: any node with >2 children: join children pairwise (new edges have zero weight)
    vector<WN> out;  // new list
    // build final arrays via recursion producing postorder
    g.L.clear(); g.R.clear(); g.par.clear(); g.taxon.clear(); g.len.clear(); g.sup.clear();
    function<int(int)> emit;
    auto addNode = [&](int l, int r, int tax, double ln, double sp) {
        g.L.push_back(l); g.R.push_back(r); g.taxon.push_back(tax); g.len.push_back(ln); g.sup.push_back(sp); g.par.push_back(-1);
        int id = (int)g.L.size() - 1;
        if (l >= 0) g.par[l] = id;
        if (r >= 0) g.par[r] = id;
        return id;
    };
    // iterative-ish recursion fine for 250 taxa (depth <= 500)
    emit = [&](int v) -> int {
        if (w[v].taxon >= 0) return addNode(-1, -1, w[v].taxon, w[v].len, NAN);
        vector<int> kids;
        for (int c : w[v].ch) kids.push_back(emit(c));
        while (kids.size() > 2) {
            int a = kids[kids.size() - 2], b = kids[kids.size() - 1];
            kids.pop_back(); kids.pop_back();
            int id = addNode(a, b, -1, 0.0, -1.0);  // artificial edge: zero weight
            kids.push_back(id);
        }
        if (kids.size() == 1) return kids[0];
        return addNode(kids[0], kids[1], -1, w[v].len, w[v].sup);
    };
    g.root = emit(root);
    g.nn = (int)g.L.size();
    g.leafNode.assign(NT, -1);
    for (int i = 0; i < g.nn; i++) if (g.taxon[i] >= 0) { if (g.leafNode[g.taxon[i]] >= 0) return false; g.leafNode[g.taxon[i]] = i; }
    for (int i = 0; i < NT; i++) if (g.leafNode[i] < 0) return false;
    g.sz.assign(g.nn, 0);
    for (int i = 0; i < g.nn; i++) g.sz[i] = g.taxon[i] >= 0 ? 1 : g.sz[g.L[i]] + g.sz[g.R[i]];
    return true;
}

// parameters
static double P_SMIN = 1.0 / 3.0;  // support lower bound
static double P_POW = 1.0;         // power on normalized support
static double P_MISS = 0.0;        // omega for missing support
static double P_ZEROLEN = 1e-5;    // edges shorter than this get omega 0... (if P_ZEROMODE)
static double P_LAM = 4.0;         // terminal discount rate (internal edges)
static double P_LAMLEAF = 4.0;     // terminal discount rate (pendant edges)
static int P_HFORM = 0;            // 0 exp(-lam L), 1 1/(1+lam L)
static int P_LNORM = 1;            // 0 raw lengths, 1 normalised by gene mean edge length
static int P_GW = 0;               // gene weight mode
static double P_CAP = 1.0;         // cap multiplier on edge weight
static double P_MU = 0.0;          // internal-length factor 1-exp(-mu*Lnorm) (off)
static int P_WMODE = 3;            // 0 support, 1 length-imputed, 2 mean, 3 geometric mean, 4 min
static double P_L0 = 0.010;
static double P_ABSK = 0.0;        // absolute-substitution factor scale (0 = off)
static double P_ABSB = 1.0;        // its exponent
static double P_ALPHA = 0.5;       // weight of support in geometric blend
static double P_GTAU = 4.0;        // gene weight scale

static void setWeights(Gene& g) {
    double tl0 = 0; int ne0 = 0;
    for (int i = 0; i < g.nn; i++) if (i != g.root) { tl0 += g.len[i]; ne0++; }
    double meanE0 = ne0 ? tl0 / ne0 : 1.0; if (meanE0 <= 0) meanE0 = 1e-9;
    g.om.assign(g.nn, 0); g.keep.assign(g.nn, 1);
    for (int i = 0; i < g.nn; i++) {
        double o = 0;
        if (g.taxon[i] < 0 && i != g.root) {
            double s = g.sup[i];
            if (s < 0) o = 0;  // artificial
            else if (std::isnan(s)) {
                // impute from normalised length (fit on sample data: mean omega ~ 1-exp(-Lnorm/0.010))
                double Ln = max(0.0, g.len[i]) / meanE0 * 0.03;
                if (tl0 <= 0) o = 1.0;  // no length information at all: unweighted
                else o = (g.len[i] < 1e-5) ? 0.0 : P_CAP * (1.0 - exp(-Ln / 0.010));
            }
            else {
                double x = (s - P_SMIN) / (1.0 - P_SMIN);
                x = min(1.0, max(0.0, x));
                o = P_CAP * pow(x, P_POW);
            }
        }
        if (P_WMODE > 0 && g.taxon[i] < 0 && i != g.root && g.sup[i] >= 0 && tl0 > 0) {
            double Ln = max(0.0, g.len[i]) / meanE0 * 0.03;
            double ol = (g.len[i] < 1e-5) ? 0.0 : P_CAP * (1.0 - exp(-Ln / P_L0));
            if (P_WMODE == 1) o = ol;
            else if (P_WMODE == 2) o = 0.5 * (o + ol);
            else if (P_WMODE == 3) o = (o > 0 && ol > 0) ? exp(P_ALPHA * log(o) + (1 - P_ALPHA) * log(ol)) : 0.0;
            else if (P_WMODE == 4) o = min(o, ol);
        }
        if (P_ABSK > 0 && g.taxon[i] < 0 && i != g.root && tl0 > 0) o *= pow(1.0 - exp(-max(0.0, g.len[i]) * 200.0 / P_ABSK), P_ABSB);
        if (P_MU > 0 && g.taxon[i] < 0 && i != g.root) o *= 1 - exp(-P_MU * max(0.0, g.len[i]) / meanE0 * 0.03);
        g.om[i] = o; g.keep[i] = 1 - o;
    }
    double tl = 0; int ne = 0;
    for (int i = 0; i < g.nn; i++) if (i != g.root) { tl += g.len[i]; ne++; }
    g.totlen = tl;
    double meanE = ne ? tl / ne : 1.0; if (meanE <= 0) meanE = 1e-9;
    vector<double> tlen; for (int i = 0; i < g.nn; i++) if (g.taxon[i] >= 0) tlen.push_back(g.len[i]);
    sort(tlen.begin(), tlen.end());
    double medT = tlen.empty() ? 1.0 : tlen[tlen.size() / 2]; if (medT <= 0) medT = meanE;
    double tli = 0; int nei = 0;
    for (int i = 0; i < g.nn; i++) if (i != g.root && g.taxon[i] < 0) { tli += g.len[i]; nei++; }
    double meanI = nei ? tli / nei : meanE; if (meanI <= 0) meanI = meanE;
    g.h.assign(g.nn, 1.0);
    for (int i = 0; i < g.nn; i++) {
        if (i == g.root) continue;
        double L = g.len[i]; if (L < 0) L = 0;
        if (P_LNORM == 1) L = L / meanE * 0.03;
        else if (P_LNORM == 2) L = 0.03;  // topological
        else if (P_LNORM == 3) L = L / medT * 0.03;  // by median terminal length
        else if (P_LNORM == 4) L = 0.5 * (L / meanE * 0.03 + 0.03);  // mix
        else if (P_LNORM == 5) L = L / meanI * 0.03;  // by mean internal edge length
        if (P_HFORM == 1) g.h[i] = 1.0 / (1.0 + (g.taxon[i] >= 0 ? P_LAMLEAF : P_LAM) * L);
        else g.h[i] = exp(-(g.taxon[i] >= 0 ? P_LAMLEAF : P_LAM) * L);
    }
    g.gw = 1.0;
    if (P_GW == 1) g.gw = min(1.0, tl / P_GTAU);
    else if (P_GW == 2 || P_GW == 3) {
        int ni = 0, nz = 0;
        for (int i = 0; i < g.nn; i++) if (g.taxon[i] < 0 && i != g.root && g.sup[i] >= -0.5) { ni++; if (g.len[i] > 1e-5) nz++; }
        double f = ni ? (double)nz / ni : 1.0;
        if (P_GW == 2) g.gw = pow(f, P_GTAU);
        else g.gw = min(1.0, f / P_GTAU);
    }
    g.D.assign(g.nn, 0); g.Dc.assign(g.nn, 0); g.DU.assign(g.nn, 0);
    for (int i = 0; i < g.nn; i++) {
        if (g.taxon[i] >= 0) { g.D[i] = 0; g.DU[i] = 0; g.Dc[i] = 1; continue; }
        int l = g.L[i], r = g.R[i];
        double nl = g.h[l] * g.Dc[l], nr = g.h[r] * g.Dc[r];
        g.Dc[i] = nl + nr;
        g.D[i] = g.keep[l] * g.D[l] + g.keep[r] * g.D[r] + nl * nr;
        g.DU[i] = g.DU[l] + g.DU[r] + nl * nr;
    }
}

static vector<Gene> genes;
static int OUTG = 0;

// ---------------- tripartition scoring ----------------
struct Scratch {
    vector<u64> mark;  // bitmap over nodes
    vector<double> st; // 18 per node
    vector<double> out; // 3 per node
};
static vector<Scratch> scr;

// col: color per taxon (0,1,2); nb: list of non-background taxa
// state per relevant node (18): c[3], PS[6], UPS[6], K[3]
static double scoreGene(int gi, const unsigned char* col, int bcol, const double* /*tot*/, const vector<int>& nb) {
    const Gene& g = genes[gi];
    Scratch& S = scr[gi];
    int nwm = (g.nn + 63) >> 6;
    u64* mk = S.mark.data();
    for (int k = 0; k < nwm; k++) mk[k] = 0;
    for (int t : nb) {
        int v = g.par[g.leafNode[t]];
        while (v >= 0 && !((mk[v >> 6] >> (v & 63)) & 1)) { mk[v >> 6] |= 1ULL << (v & 63); v = g.par[v]; }
    }
    double total = 0;
    double* st = S.st.data();
    const int SS = 18;
    for (int k = 0; k < nwm; k++) {
        u64 word = mk[k];
        while (word) {
            int v = k * 64 + __builtin_ctzll(word); word &= word - 1;
            double cl[3], cr[3], pl[6], pr[6], ul[6], ur[6];
            int ch[2] = {g.L[v], g.R[v]};
            for (int side = 0; side < 2; side++) {
                int x = ch[side];
                double* c = side ? cr : cl; double* p = side ? pr : pl; double* u = side ? ur : ul;
                if (g.taxon[x] >= 0) {
                    c[0] = c[1] = c[2] = 0; c[col[g.taxon[x]]] = g.h[x];
                    for (int q = 0; q < 6; q++) p[q] = u[q] = 0;
                } else if ((mk[x >> 6] >> (x & 63)) & 1) {
                    const double* s = st + SS * x; double kp = g.keep[x], hx = g.h[x];
                    c[0] = hx * s[0]; c[1] = hx * s[1]; c[2] = hx * s[2];
                    for (int q = 0; q < 6; q++) { p[q] = kp * s[3 + q]; u[q] = s[9 + q]; }
                } else {
                    c[0] = c[1] = c[2] = 0; c[bcol] = g.h[x] * g.Dc[x];
                    for (int q = 0; q < 6; q++) p[q] = u[q] = 0;
                    p[bcol] = g.keep[x] * g.D[x]; u[bcol] = g.DU[x];
                }
            }
            double* sv = st + SS * v;
            double acc = 0;
            for (int i = 0; i < 3; i++) {
                acc += ul[i] * ur[3 + i] - pl[i] * pr[3 + i];
                acc += ul[3 + i] * ur[i] - pl[3 + i] * pr[i];
            }
            total += acc;
            double a[6], b[6];
            for (int q = 0; q < 6; q++) { a[q] = ul[q] - pl[q]; b[q] = ur[q] - pr[q]; }
            sv[0] = cl[0] + cr[0]; sv[1] = cl[1] + cr[1]; sv[2] = cl[2] + cr[2];
            double n0 = cl[0] * cr[0], n1 = cl[1] * cr[1], n2 = cl[2] * cr[2];
            double n3 = cl[1] * cr[2] + cl[2] * cr[1], n4 = cl[0] * cr[2] + cl[2] * cr[0], n5 = cl[0] * cr[1] + cl[1] * cr[0];
            sv[3] = pl[0] + pr[0] + n0; sv[4] = pl[1] + pr[1] + n1; sv[5] = pl[2] + pr[2] + n2;
            sv[6] = pl[3] + pr[3] + n3; sv[7] = pl[4] + pr[4] + n4; sv[8] = pl[5] + pr[5] + n5;
            sv[9] = ul[0] + ur[0] + n0; sv[10] = ul[1] + ur[1] + n1; sv[11] = ul[2] + ur[2] + n2;
            sv[12] = ul[3] + ur[3] + n3; sv[13] = ul[4] + ur[4] + n4; sv[14] = ul[5] + ur[5] + n5;
            sv[15] = a[3] * cr[0] + a[1] * cr[2] + a[2] * cr[1] + b[3] * cl[0] + b[1] * cl[2] + b[2] * cl[1];
            sv[16] = a[0] * cr[2] + a[4] * cr[1] + a[2] * cr[0] + b[0] * cl[2] + b[4] * cl[1] + b[2] * cl[0];
            sv[17] = a[0] * cr[1] + a[1] * cr[0] + a[5] * cr[2] + b[0] * cl[1] + b[1] * cl[0] + b[5] * cl[2];
        }
    }
    // pass 2: top-down outside counts. o stored in S.out (3 per node)
    double* ob = S.out.data();
    int rt = g.root;
    ob[3 * rt] = ob[3 * rt + 1] = ob[3 * rt + 2] = 0;
    for (int k = nwm - 1; k >= 0; k--) {
        u64 word = mk[k];
        while (word) {
            int hb = 63 - __builtin_clzll(word); word &= ~(1ULL << hb);
            int v = k * 64 + hb;
            const double* sv = st + SS * v;
            const double* o = ob + 3 * v;
            total += sv[15] * o[0] + sv[16] * o[1] + sv[17] * o[2];
            int ch[2] = {g.L[v], g.R[v]};
            for (int side = 0; side < 2; side++) {
                int x = ch[side];
                if (!((mk[x >> 6] >> (x & 63)) & 1)) continue;  // only relevant internal children
                int y = ch[1 - side];
                double cy[3];
                if (g.taxon[y] >= 0) { cy[0] = cy[1] = cy[2] = 0; cy[col[g.taxon[y]]] = g.h[y]; }
                else if ((mk[y >> 6] >> (y & 63)) & 1) { const double* s = st + SS * y; double hy = g.h[y]; cy[0] = hy * s[0]; cy[1] = hy * s[1]; cy[2] = hy * s[2]; }
                else { cy[0] = cy[1] = cy[2] = 0; cy[bcol] = g.h[y] * g.Dc[y]; }
                double hx = g.h[x];
                double* ox = ob + 3 * x;
                ox[0] = hx * (o[0] + cy[0]); ox[1] = hx * (o[1] + cy[1]); ox[2] = hx * (o[2] + cy[2]);
            }
        }
    }
    return total * g.gw;
}

static long long nScoreCalls = 0;
static vector<unsigned char> colbuf;
// score tripartition (A, B, rest) where A, B disjoint
static double* perGeneOut = nullptr;
static double scoreTrip(const Bits& A, const Bits& B) {
    nScoreCalls++;
    Bits AB = A | B;
    double tot[3] = {(double)A.count(), (double)B.count(), (double)(NT - AB.count())};
    int bcol = 0; if (tot[1] > tot[bcol]) bcol = 1; if (tot[2] > tot[bcol]) bcol = 2;
    static vector<int> nb; nb.clear();
    for (int t = 0; t < NT; t++) {
        unsigned char c = A.test(t) ? 0 : (B.test(t) ? 1 : 2);
        colbuf[t] = c;
        if (c != bcol) nb.push_back(t);
    }
    double s = 0;
    for (int gi = 0; gi < (int)genes.size(); gi++) { double x = scoreGene(gi, colbuf.data(), bcol, tot, nb); if (perGeneOut) perGeneOut[gi] = x; s += x; }
    return s;
}

// ---------------- cluster set & DP ----------------
static vector<Bits> clus;
static vector<int> csize;
static unordered_map<Bits, int, BitsHash> cidx;
static Bits INGROUP;

static int addCluster(const Bits& b) {
    auto it = cidx.find(b);
    if (it != cidx.end()) return it->second;
    int id = (int)clus.size();
    clus.push_back(b); csize.push_back(b.count()); cidx[b] = id;
    return id;
}
// add a bipartition given by one side (any side)
static void addSplit(Bits side) {
    if (side.test(OUTG)) {
        Bits all; for (int t = 0; t < NT; t++) all.set(t);
        side = side ^ all;
    }
    int c = side.count();
    if (c < 1 || c > NT - 1) return;
    addCluster(side);
}

static unordered_map<u64, int> memo;   // tripartition -> row in pgTable
static vector<double> pgTable;          // per-gene scores, G per row
static vector<double> totTable;         // total score per row
static const double* DPW = nullptr;     // gene weights for DP (nullptr = all ones)
static const unordered_map<Bits, double, BitsHash>* MEAF = nullptr;  // if set: MEA objective
static double MEA_EPS = 0.0;
static unordered_map<Bits, double, BitsHash> njFreq;  // bootstrap NJ frequency per cluster
static double P_BONUS = 0.0;  // relative bonus weight
static double BONUS_SCALE = 0.0;  // absolute multiplier (set after first DP)
static int P_NBOOT = 30;
static int P_NNI = 30;       // max NNI-enrichment rounds
static int P_SPR = 0;        // SPR enrichment radius (0 = off)
static int P_BAG = 0;        // number of bagging replicates (0 = off)
static int P_BAGMODE = 0;    // 0 multinomial bootstrap, 1 Bayesian bootstrap
static double P_FULLW = 0.0; // extra frequency weight for full-data optimum
static int P_BAGFINAL = 0;   // 0: replicates only diversify the search; 1: MEA consensus
static double P_TLIM = 90.0; // soft time limit (s)
static vector<double> V;
static vector<int> bestA, bestB;

static double runDP() {
    int M = (int)clus.size();
    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return csize[a] < csize[b]; });
    vector<vector<int>> byTax(NT);
    for (int id : order) { const Bits& b = clus[id]; for (int t = 0; t < NT; t++) if (b.test(t)) byTax[t].push_back(id); }
    V.assign(M, -1e300); bestA.assign(M, -1); bestB.assign(M, -1);
    for (int id : order) {
        if (csize[id] == 1) { V[id] = 0; continue; }
        const Bits& A = clus[id];
        int a = A.first();
        double best = -1e300; int ba = -1, bb = -1;
        double mult = 1.0;
        if (P_BONUS > 0) { auto fi = njFreq.find(A); if (fi != njFreq.end()) mult += P_BONUS * fi->second; }
        double meaBonus = 0;
        if (MEAF) { auto fi = MEAF->find(A); if (fi != MEAF->end()) meaBonus = fi->second; }
        for (int bid : byTax[a]) {
            if (csize[bid] >= csize[id]) break;
            if (V[bid] < -1e299) continue;
            const Bits& B = clus[bid];
            if (!B.subsetOf(A)) continue;
            Bits C = A ^ B;
            auto it = cidx.find(C);
            if (it == cidx.end()) continue;
            int cid = it->second;
            if (V[cid] < -1e299) continue;
            u64 key = ((u64)min(bid, cid) << 32) | (u64)max(bid, cid);
            int row;
            auto mi = memo.find(key);
            if (mi != memo.end()) row = mi->second;
            else {
                int G = genes.size();
                row = (int)totTable.size();
                pgTable.resize(pgTable.size() + G);
                perGeneOut = pgTable.data() + (size_t)row * G;
                double t = scoreTrip(B, clus[cid]);
                perGeneOut = nullptr;
                totTable.push_back(t);
                memo[key] = row;
            }
            double w;
            if (MEAF) w = MEA_EPS * totTable[row];
            else if (!DPW) w = totTable[row];
            else { int G = genes.size(); const double* pg = pgTable.data() + (size_t)row * G; w = 0; for (int g = 0; g < G; g++) w += DPW[g] * pg[g]; }
            double val = V[bid] + V[cid] + w * mult;
            if (val > best) { best = val; ba = bid; bb = cid; }
        }
        if (best > -1e299) best += meaBonus;
        V[id] = best; bestA[id] = ba; bestB[id] = bb;
    }
    return V[cidx[INGROUP]];
}

// rooted tree from DP: list of clusters (internal) with children
struct RT { vector<int> kids[2]; };
static void collectTree(int id, vector<int>& internals) {
    if (csize[id] == 1) return;
    internals.push_back(id);
    collectTree(bestA[id], internals);
    collectTree(bestB[id], internals);
}
static string qlabel(const string& s) {
    bool need = s.empty();
    for (char c : s) if (strchr("()[]':;, \t\n", c)) need = true;
    if (!need) return s;
    string r = "'";
    for (char c : s) { if (c == '\'') r += "''"; else r += c; }
    return r + "'";
}
static void writeNewick(int id, string& out) {
    if (csize[id] == 1) { out += qlabel(taxName[clus[id].first()]); return; }
    out += '(';
    writeNewick(bestA[id], out);
    out += ',';
    writeNewick(bestB[id], out);
    out += ')';
}

// ---------------- distance + NJ ----------------
// computes per-gene leaf-leaf path weights with edge weight function, accumulates into D
static void geneDistances(const Gene& g, const vector<double>& ew, vector<double>& Dm) {
    // ew: weight of edge (v,parent v) for each node. root edge none.
    // adjacency
    int nn = g.nn;
    vector<vector<pair<int, double>>> adj(nn);
    for (int v = 0; v < nn; v++) if (g.par[v] >= 0) { adj[v].push_back({g.par[v], ew[v]}); adj[g.par[v]].push_back({v, ew[v]}); }
    vector<double> dist(nn);
    vector<int> stk; vector<int> from(nn);
    for (int t = 0; t < NT; t++) {
        int s = g.leafNode[t];
        stk.clear(); stk.push_back(s); dist[s] = 0; from[s] = -1;
        while (!stk.empty()) {
            int v = stk.back(); stk.pop_back();
            for (auto& e : adj[v]) if (e.first != from[v]) { from[e.first] = v; dist[e.first] = dist[v] + e.second; stk.push_back(e.first); }
        }
        for (int u = 0; u < NT; u++) Dm[t * NT + u] += dist[g.leafNode[u]];
    }
}

// Neighbor joining; returns list of splits (as Bits) of resulting tree
static vector<Bits> neighborJoin(const vector<double>& D0) {
    int n = NT;
    vector<double> D = D0;
    vector<int> active(n); iota(active.begin(), active.end(), 0);
    vector<Bits> cl(2 * n);
    for (int i = 0; i < n; i++) { cl[i] = Bits(); cl[i].set(i); }
    vector<int> nodeOf(n); iota(nodeOf.begin(), nodeOf.end(), 0);
    int nextNode = n;
    vector<Bits> splits;
    // D as n x n matrix with indices of slots; slot i holds node nodeOf[i]
    vector<char> alive(n, 1);
    int m = n;
    vector<double> rsum(n, 0);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) rsum[i] += D[i * n + j];
    while (m > 3) {
        double best = 1e300; int bi = -1, bj = -1;
        for (int i = 0; i < n; i++) if (alive[i]) for (int j = i + 1; j < n; j++) if (alive[j]) {
            double q = (m - 2) * D[i * n + j] - rsum[i] - rsum[j];
            if (q < best) { best = q; bi = i; bj = j; }
        }
        // join bi,bj into slot bi
        Bits nb = cl[nodeOf[bi]] | cl[nodeOf[bj]];
        cl[nextNode] = nb; splits.push_back(nb);
        double dij = D[bi * n + bj];
        alive[bj] = 0;
        for (int k = 0; k < n; k++) if (alive[k] && k != bi) {
            double dk = 0.5 * (D[bi * n + k] + D[bj * n + k] - dij);
            rsum[k] += dk - D[k * n + bi] - D[k * n + bj];
            D[bi * n + k] = D[k * n + bi] = dk;
        }
        rsum[bi] = 0;
        for (int k = 0; k < n; k++) if (alive[k] && k != bi) rsum[bi] += D[bi * n + k];
        nodeOf[bi] = nextNode++;
        m--;
    }
    return splits;
}

static string EVALTREE;
// score of a rooted-at-outgroup tree given as Newick; returns 0.5*sum W; also fills its clusters
static double evalTreeFile(const string& fn, vector<Bits>& tclus) {
    ifstream f(fn); stringstream ss; ss << f.rdbuf(); string s = ss.str();
    PTree t; if (!parseNewick(s, t)) return -1;
    // compute leaf sets, then reroot at outgroup: clusters = sides without OUTG
    int N = t.nodes.size();
    vector<Bits> sub(N);
    function<void(int)> rec = [&](int v) { if (t.nodes[v].ch.empty()) { sub[v].set(taxIndex[t.nodes[v].label]); return; } for (int c : t.nodes[v].ch) { rec(c); sub[v] = sub[v] | sub[c]; } };
    rec(t.root);
    Bits all; for (int i = 0; i < NT; i++) all.set(i);
    // Build adjacency for unrooted tree and compute tripartitions around each internal node, oriented with OUTG
    // Simpler: for each internal node, its 3 neighbor sides
    double tot = 0;
    for (int v = 0; v < N; v++) {
        if (t.nodes[v].ch.empty()) continue;
        vector<Bits> sides;
        for (int c : t.nodes[v].ch) sides.push_back(sub[c]);
        if (v != t.root) sides.push_back(all ^ sub[v]);
        if (sides.size() != 3) continue;  // skip degree-2 root
        // the side containing OUTG is "rest"
        int r = -1; for (int k = 0; k < 3; k++) if (sides[k].test(OUTG)) r = k;
        int a = (r + 1) % 3, b = (r + 2) % 3;
        tot += scoreTrip(sides[a], sides[b]);
    }
    for (int v = 0; v < N; v++) { Bits s = sub[v]; if (s.test(OUTG)) s = s ^ all; int c = s.count(); if (c >= 2 && c <= NT - 2) tclus.push_back(s); }
    return 0.5 * tot;
}


#ifdef BRUTE
static int bruteMain(const char* fn) {
    ifstream in(fn); stringstream ss; ss << in.rdbuf(); string all = ss.str();
    vector<string> lines; string cur; for (char c : all) { cur += c; if (c == ';') { lines.push_back(cur); cur.clear(); } }
    vector<PTree> pts; for (auto& l : lines) { PTree t; if (parseNewick(l, t)) pts.push_back(t); }
    for (auto& pt : pts) for (auto& nd : pt.nodes) if (nd.ch.empty()) { if (!taxIndex.count(nd.label)) { taxIndex[nd.label] = (int)taxName.size(); taxName.push_back(nd.label); } }
    NT = taxName.size(); OUTG = 0;
    for (auto& t : pts) { Gene g; if (buildGene(t, g)) { setWeights(g); genes.push_back(std::move(g)); } }
    scr.resize(genes.size());
    for (size_t i = 0; i < genes.size(); i++) { scr[i].mark.assign((genes[i].nn + 63) / 64 + 1, 0); scr[i].st.assign(18 * genes[i].nn, 0); scr[i].out.assign(3 * genes[i].nn, 0); }
    colbuf.assign(NT, 0);
    mt19937 rng(7);
    double maxrel = 0;
    for (int trial = 0; trial < 20; trial++) {
        vector<int> col(NT); for (int t = 0; t < NT; t++) col[t] = rng() % 3;
        Bits A, B; for (int t = 0; t < NT; t++) { if (col[t] == 0) A.set(t); else if (col[t] == 1) B.set(t); }
        double fast = scoreTrip(A, B);
        double brute = 0;
        for (auto& g : genes) {
            // edge sets of leaf-leaf paths as vectors of node ids (edge = node->parent)
            auto path = [&](int x, int y) { // returns set of edge nodes
                vector<int> ax, ay; int u = g.leafNode[x]; while (u >= 0) { ax.push_back(u); u = g.par[u]; }
                u = g.leafNode[y]; while (u >= 0) { ay.push_back(u); u = g.par[u]; }
                set<int> sx(ax.begin(), ax.end()); int lca = -1; for (int v : ay) if (sx.count(v)) { lca = v; break; }
                set<int> e; for (int v : ax) { if (v == lca) break; e.insert(v); } for (int v : ay) { if (v == lca) break; e.insert(v); }
                return e;
            };
            auto inter = [](const set<int>& a, const set<int>& b) { set<int> r; for (int x : a) if (b.count(x)) r.insert(x); return r; };
            for (int a = 0; a < NT; a++) for (int b = a + 1; b < NT; b++) for (int c = b + 1; c < NT; c++) for (int d = c + 1; d < NT; d++) {
                int q[4] = {a, b, c, d};
                int cnt[3] = {0, 0, 0}; for (int k = 0; k < 4; k++) cnt[col[q[k]]]++;
                int pairc = -1; bool ok = true;
                for (int k = 0; k < 3; k++) { if (cnt[k] == 2) pairc = k; else if (cnt[k] != 1) ok = false; }
                if (!ok || pairc < 0) continue;
                // species pairing: two leaves of color pairc together
                int p1 = -1, p2 = -1, o1 = -1, o2 = -1;
                for (int k = 0; k < 4; k++) { if (col[q[k]] == pairc) { if (p1 < 0) p1 = q[k]; else p2 = q[k]; } else { if (o1 < 0) o1 = q[k]; else o2 = q[k]; } }
                // gene topology: p1p2|o1o2 iff path(p1,p2) and path(o1,o2) disjoint
                set<int> e12 = path(p1, p2), e34 = path(o1, o2);
                if (!inter(e12, e34).empty()) continue;
                set<int> internal = inter(path(p1, o1), path(p2, o2));
                double prodk = 1; for (int e : internal) prodk *= g.keep[e];
                double hw = 1;
                set<int> term;
                auto tp = [&](int x, int y, int z, int w) { set<int> r = inter(inter(path(x, y), path(x, z)), path(x, w)); for (int e : r) term.insert(e); };
                tp(p1, p2, o1, o2); tp(p2, p1, o1, o2); tp(o1, o2, p1, p2); tp(o2, o1, p1, p2);
                for (int e : term) hw *= g.h[e];
                brute += (1 - prodk) * hw * g.gw;
            }
        }
        double rel = fabs(fast - brute) / max(1e-12, fabs(brute));
        maxrel = max(maxrel, rel);
        printf("trial %d fast %.10g brute %.10g rel %.2e\n", trial, fast, brute, rel);
    }
    printf("max rel err %.3e\n", maxrel);
    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc >= 2 && string(argv[1]) == "--selftest") { Bits b; b.set(3); return b.count() == 1 ? 0 : 1; }
#ifdef BRUTE
    if (argc >= 3 && string(argv[1]) == "--brute") { return bruteMain(argv[2]); }
#endif
    if (argc < 3) { fprintf(stderr, "usage: stree in out\n"); return 1; }
    now_s();
    for (int i = 3; i + 1 < argc; i += 2) {
        string k = argv[i]; double v = atof(argv[i + 1]);
        if (k == "-eval") EVALTREE = argv[i + 1];
        if (k == "-smin") P_SMIN = v;
        else if (k == "-lam") P_LAM = v; else if (k == "-lnorm") P_LNORM = (int)v; else if (k == "-gw") P_GW = (int)v; else if (k == "-gtau") P_GTAU = v; else if (k == "-cap") P_CAP = v; else if (k == "-bonus") P_BONUS = v; else if (k == "-bag") P_BAG = (int)v; else if (k == "-bagmode") P_BAGMODE = (int)v; else if (k == "-fullw") P_FULLW = v; else if (k == "-tlim") P_TLIM = v; else if (k == "-nboot") P_NBOOT = (int)v; else if (k == "-mu") P_MU = v; else if (k == "-wmode") P_WMODE = (int)v; else if (k == "-l0") P_L0 = v; else if (k == "-alpha") P_ALPHA = v; else if (k == "-nni") P_NNI = (int)v; else if (k == "-spr") P_SPR = (int)v; else if (k == "-absk") P_ABSK = v; else if (k == "-lamleaf") P_LAMLEAF = v; else if (k == "-bagfinal") P_BAGFINAL = (int)v; else if (k == "-hform") P_HFORM = (int)v; else if (k == "-absb") P_ABSB = v; else if (k == "-pow") P_POW = v; else if (k == "-miss") P_MISS = v;
    }
    ifstream in(argv[1]);
    string line; vector<string> lines;
    {
        stringstream ss; ss << in.rdbuf(); string all = ss.str();
        // split by ';'
        string cur;
        for (char c : all) { cur += c; if (c == ';') { lines.push_back(cur); cur.clear(); } }
    }
    vector<PTree> pts;
    for (auto& l : lines) { PTree t; if (parseNewick(l, t)) pts.push_back(t); }
    if (pts.empty()) { fprintf(stderr, "no trees\n"); return 1; }
    // taxa from first tree
    for (auto& pt : pts) for (auto& nd : pt.nodes) if (nd.ch.empty()) { if (!taxIndex.count(nd.label)) { taxIndex[nd.label] = (int)taxName.size(); taxName.push_back(nd.label); } }
    NT = (int)taxName.size();
    if (NT > 64 * MW) { fprintf(stderr, "too many taxa\n"); return 1; }
    if (NT < 4) {
        string o = "(";
        for (int t = 0; t < NT; t++) { if (t) o += ","; o += qlabel(taxName[t]); }
        o += ");\n";
        FILE* f = fopen(argv[2], "w"); fputs(o.c_str(), f); fclose(f); return 0;
    }
    OUTG = taxIndex.count("0") ? taxIndex["0"] : 0;
    {
        double mx = 0;
        for (auto& t : pts) for (auto& nd : t.nodes) if (!std::isnan(nd.sup)) mx = max(mx, nd.sup);
        if (mx > 1.0 + 1e-6) {
            for (auto& t : pts) for (auto& nd : t.nodes) if (!std::isnan(nd.sup)) nd.sup /= (mx > 100.0 + 1e-6 ? mx : 100.0);
        }
    }
    for (auto& t : pts) { Gene g; if (buildGene(t, g)) { setWeights(g); genes.push_back(std::move(g)); } }
    fprintf(stderr, "[%.2fs] parsed %d genes, %d taxa\n", now_s(), (int)genes.size(), NT);
    if (genes.empty()) { fprintf(stderr, "no usable gene trees\n"); return 1; }
    scr.resize(genes.size());
    for (size_t i = 0; i < genes.size(); i++) { scr[i].mark.assign((genes[i].nn + 63) / 64 + 1, 0); scr[i].st.assign(18 * genes[i].nn, 0); scr[i].out.assign(3 * genes[i].nn, 0); }
    colbuf.assign(NT, 0);
    for (int t = 0; t < NT; t++) if (t != OUTG) INGROUP.set(t);
    // clusters: singletons, ingroup
    for (int t = 0; t < NT; t++) if (t != OUTG) { Bits b; b.set(t); addCluster(b); }
    addCluster(INGROUP);
    // gene tree splits
    for (auto& g : genes) {
        vector<Bits> sub(g.nn);
        for (int v = 0; v < g.nn; v++) {
            if (g.taxon[v] >= 0) sub[v].set(g.taxon[v]); else sub[v] = sub[g.L[v]] | sub[g.R[v]];
            addSplit(sub[v]);
        }
    }
    fprintf(stderr, "[%.2fs] gene clusters: %d\n", now_s(), (int)clus.size());
    // distance trees
    {
        mt19937_64 rng(1);
        const char* se = getenv("SEED"); if (se) rng.seed(atoll(se));
        int G = genes.size();
        vector<vector<double>> perGene(G, vector<double>(NT * NT, 0));
        for (int gi = 0; gi < G; gi++) {
            const Gene& g = genes[gi];
            vector<double> ew(g.nn, 0);
            for (int v = 0; v < g.nn; v++) ew[v] = (g.taxon[v] >= 0) ? 0.5 : (g.om[v] + 0.0);
            geneDistances(g, ew, perGene[gi]);
        }
        vector<double> Dm(NT * NT, 0);
        for (int gi = 0; gi < G; gi++) for (int k = 0; k < NT * NT; k++) Dm[k] += perGene[gi][k];
        auto sp = neighborJoin(Dm);
        Bits allT; for (int t = 0; t < NT; t++) allT.set(t);
        auto norm = [&](Bits s) { if (s.test(OUTG)) s = s ^ allT; return s; };
        for (auto& b : sp) addSplit(b);
        int nboot = P_NBOOT;
        for (int r = 0; r < nboot; r++) {
            vector<double> Db(NT * NT, 0);
            for (int k2 = 0; k2 < G; k2++) { int gi = rng() % G; for (int k = 0; k < NT * NT; k++) Db[k] += perGene[gi][k]; }
            auto sp2 = neighborJoin(Db);
            for (auto& b : sp2) { addSplit(b); Bits nb2 = norm(b); if (nb2.count() >= 2) njFreq[nb2] += 1.0 / nboot; }
        }
    }
    fprintf(stderr, "[%.2fs] clusters after NJ: %d\n", now_s(), (int)clus.size());
    double sc = runDP();
    fprintf(stderr, "[%.2fs] DP score %.6g, clusters %d, tripartitions %lld\n", now_s(), sc, (int)clus.size(), nScoreCalls);

    auto enrich = [&](double sc, int maxIt) {
        for (int it = 0; it < maxIt && now_s() < P_TLIM; it++) {
            vector<int> internals; collectTree(cidx[INGROUP], internals);
            int before = clus.size();
            unordered_map<int, int> parentOf;
            for (int u : internals) { parentOf[bestA[u]] = u; parentOf[bestB[u]] = u; }
            for (int u : internals) {
                auto pit = parentOf.find(u);
                if (pit == parentOf.end()) continue;
                int p = pit->second; int s = bestA[p] == u ? bestB[p] : bestA[p];
                addCluster(clus[bestA[u]] | clus[s]);
                addCluster(clus[bestB[u]] | clus[s]);
            }
            if ((int)clus.size() == before) break;
            double sc2 = runDP();
            if (sc2 <= sc + 1e-12 * fabs(sc)) { sc = sc2; break; }
            sc = sc2;
        }
        return sc;
    };
    sc = enrich(sc, P_NNI);
    // SPR-neighbourhood enrichment: prune subtree x, regraft on edge above y (path length <= P_SPR)
    for (int round = 0; round < 10 && P_SPR > 0 && now_s() < P_TLIM; round++) {
        vector<int> internals; collectTree(cidx[INGROUP], internals);
        // node list: internals + leaves; parent pointers
        unordered_map<int, int> parentOf;
        vector<int> nodes;
        for (int u : internals) { parentOf[bestA[u]] = u; parentOf[bestB[u]] = u; nodes.push_back(u); nodes.push_back(bestA[u]); nodes.push_back(bestB[u]); }
        sort(nodes.begin(), nodes.end()); nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
        int rootId = cidx[INGROUP];
        int before = clus.size();
        for (int x : nodes) {
            if (x == rootId) continue;
            const Bits& X = clus[x];
            // walk up ancestors of x up to P_SPR steps; from each ancestor a, descend into other child up to remaining steps
            vector<int> upPath;  // ancestors of x (excluding x)
            int cur = x;
            for (int d = 1; d <= P_SPR; d++) {
                auto it = parentOf.find(cur); if (it == parentOf.end()) break;
                int a = it->second; upPath.push_back(a);
                int other = bestA[a] == cur ? bestB[a] : bestA[a];
                // targets: other and its descendants within P_SPR - d steps
                vector<pair<int, int>> stk = {{other, d}};
                while (!stk.empty()) {
                    auto [y, dist] = stk.back(); stk.pop_back();
                    if (dist >= 2 || y != other) {
                        // regraft x onto edge above y: Y|X new cluster; ancestors of y below a gain X; upPath nodes below a lose X
                        addCluster(clus[y] | X);
                        // y's ancestors up to (excluding) a gain X
                        int z = y;
                        while (true) { auto pz = parentOf.find(z); if (pz == parentOf.end()) break; int pp = pz->second; if (pp == a) break; addCluster(clus[pp] | X); z = pp; }
                        for (size_t k = 0; k + 1 < upPath.size(); k++) { Bits r = clus[upPath[k]] ^ X; if (r.count() >= 1) addCluster(r); }
                    }
                    if (dist < P_SPR && csize[y] > 1) { stk.push_back({bestA[y], dist + 1}); stk.push_back({bestB[y], dist + 1}); }
                }
                cur = a;
            }
        }
        if ((int)clus.size() == before) break;
        double sc2 = runDP();
        sc2 = enrich(sc2, P_NNI);
        fprintf(stderr, "[%.2fs] SPR round %d: score %.6g clusters %d trips %lld\n", now_s(), round, sc2, (int)clus.size(), nScoreCalls);
        if (sc2 <= sc + 1e-12 * fabs(sc)) { sc = sc2; break; }
        sc = sc2;
    }
    fprintf(stderr, "[%.2fs] after NNI enrichment: score %.6g clusters %d trips %lld\n", now_s(), sc, (int)clus.size(), nScoreCalls);
    if (P_BAG > 0) {
        // remember full-data optimum
        vector<int> fullInternals; collectTree(cidx[INGROUP], fullInternals);
        vector<Bits> fullClus; for (int u : fullInternals) fullClus.push_back(clus[u]);
        unordered_map<Bits, double, BitsHash> freq;
        int G = genes.size();
        vector<double> w(G);
        mt19937_64 brng(12345);
        const char* se = getenv("SEED"); if (se) brng.seed(atoll(se) * 7919 + 17);
        int done = 0;
        for (int b = 0; b < P_BAG && now_s() < P_TLIM; b++) {
            if (P_BAGMODE == 0) { fill(w.begin(), w.end(), 0.0); for (int k = 0; k < G; k++) w[brng() % G] += 1.0; }
            else { exponential_distribution<double> ex(1.0); double s = 0; for (int k = 0; k < G; k++) { w[k] = ex(brng); s += w[k]; } for (int k = 0; k < G; k++) w[k] *= G / s; }
            DPW = w.data();
            double s1 = runDP();
            s1 = enrich(s1, 10);
            vector<int> ints; collectTree(cidx[INGROUP], ints);
            for (int u : ints) freq[clus[u]] += 1.0;
            done++;
        }
        DPW = nullptr;
        for (auto& kv : freq) kv.second /= max(1, done);
        // full-data optimum gets weight too
        for (auto& b : fullClus) freq[b] += P_FULLW;
        if (P_BAGFINAL == 0) {
            // diversification only: final DP on full data over the enlarged cluster set
            double s2 = runDP();
            s2 = enrich(s2, P_NNI);
            fprintf(stderr, "[%.2fs] diversified search (%d reps): full-data score %.6g (before %.6g), clusters %d\n", now_s(), done, s2, sc, (int)clus.size());
            sc = s2;
        } else {
        // MEA DP
        double fullScore = runDP();
        MEAF = &freq; MEA_EPS = 1e-3 / max(1e-300, fullScore);
        double m = runDP();
        MEAF = nullptr;
        vector<int> ints; collectTree(cidx[INGROUP], ints);
        int same = 0; unordered_set<Bits, BitsHash> fs(fullClus.begin(), fullClus.end());
        for (int u : ints) if (fs.count(clus[u])) same++;
        fprintf(stderr, "[%.2fs] bagging %d reps: MEA value %.4f, shares %d/%d clusters with full optimum\n", now_s(), done, m, same, (int)ints.size());
        // keep MEA tree in bestA/bestB (already set by last runDP)
        }
    }
    if (!EVALTREE.empty()) {
        vector<Bits> tc; double ts = evalTreeFile(EVALTREE, tc);
        int inX = 0, inBest = 0;
        vector<int> internals; collectTree(cidx[INGROUP], internals);
        set<int> bestSet(internals.begin(), internals.end());
        for (auto& b : tc) { auto it = cidx.find(b); if (it != cidx.end()) { inX++; if (bestSet.count(it->second)) inBest++; } }
        fprintf(stderr, "EVAL true-tree score %.6g vs found %.6g (ratio %.6f); true clusters %d, in X %d, in found %d\n", ts, 0.5 * sc, ts / (0.5 * sc), (int)tc.size(), inX, inBest);
    }
    string out;
    if (bestA[cidx[INGROUP]] >= 0) {
        out = "(" + qlabel(taxName[OUTG]) + ",";
        writeNewick(cidx[INGROUP], out);
        out += ");\n";
    } else {  // fallback: caterpillar
        out = "(" + qlabel(taxName[0]) + "," + qlabel(taxName[1]) + ")";
        for (int t = 2; t < NT; t++) out = "(" + out + "," + qlabel(taxName[t]) + ")";
        out += ";\n";
    }
    FILE* f = fopen(argv[2], "w");
    if (!f) { fprintf(stderr, "cannot open output\n"); return 1; }
    fputs(out.c_str(), f);
    if (fclose(f) != 0) return 1;
    fprintf(stderr, "[%.2fs] done\n", now_s());
    return 0;
}
