// Species-tree inference from 50 estimated gene trees (250 taxa, unrooted binary).
// Method: contract unreliable gene-tree edges, compute rooted-triplet (quartet-with-outgroup)
// votes from gene trees, build NJ / support-weighted consensus starting trees, then
// hill-climb with NNI + perturbation restarts, maximizing weighted triplet support.
#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

static const int W = 4;                     // 4*64 = 256 bits >= 250 taxa
struct M {
    uint64_t w[W];
    M(uint64_t v=0){ for(int i=0;i<W;i++) w[i]= (i==0? v:0); }
};
static inline M mor(M a, M b){ M c; for(int i=0;i<W;i++) c.w[i]=a.w[i]|b.w[i]; return c; }
static inline M mand(M a, M b){ M c; for(int i=0;i<W;i++) c.w[i]=a.w[i]&b.w[i]; return c; }
static inline M mxor(M a, M b){ M c; for(int i=0;i<W;i++) c.w[i]=a.w[i]^b.w[i]; return c; }
static inline M mandnot(M a, M b){ M c; for(int i=0;i<W;i++) c.w[i]=a.w[i]&~b.w[i]; return c; }
static inline bool mzero(M a){ uint64_t s=0; for(int i=0;i<W;i++) s|=a.w[i]; return s==0; }
static inline bool meq(M a, M b){ for(int i=0;i<W;i++) if(a.w[i]!=b.w[i]) return false; return true; }
static inline bool msubset(M a, M b){ for(int i=0;i<W;i++) if(a.w[i]&~b.w[i]) return false; return true; }
static inline int mpop(M a){ int s=0; for(int i=0;i<W;i++) s+=__builtin_popcountll(a.w[i]); return s; }
static inline int minterpop(M a, M b){ int s=0; for(int i=0;i<W;i++) s+=__builtin_popcountll(a.w[i]&b.w[i]); return s; }
struct MHash { size_t operator()(M const&a) const { uint64_t h=1469598103934665603ULL; for(int i=0;i<W;i++){ h^=a.w[i]+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); } return (size_t)h; } };
struct MEq { bool operator()(M const&a,M const&b) const { return meq(a,b); } };
static inline M bitmask(int i){ M m; m.w[i>>6] = 1ULL<<(i&63); return m; }

struct Params { double st=0.5; double lt=-1.0; double alpha=1.0; bool centered=false; };

// ------------------------------ Newick parsing ------------------------------
struct RawEdge { int to; double len; double sup; };
struct RawTree {
    int n=0;                                  // number of taxa
    vector<vector<int>> ch;                   // children in Newick order
    vector<double> len, sup;                  // edge to parent: length, support
    vector<int> tax;                          // leaf taxon index, -1 otherwise
    vector<char> hasSup;
    vector<vector<RawEdge>> adj;              // unrooted graph (all parsed vertices)
    vector<array<int,2>> edges;               // directed parent->child (v,u) list
    vector<vector<double>> elen, esup;        // parallel to adj
};

static inline void skipws(const string&s, size_t&p){ while(p<s.size() && isspace((unsigned char)s[p])) p++; }
static inline string readLabel(const string&s, size_t&p){
    size_t a=p; while(p<s.size() && !strchr("(),:;",s[p]) && !isspace((unsigned char)s[p])) p++;
    return s.substr(a,p-a);
}
static inline double readNum(const string&s, size_t&p){
    skipws(s,p); size_t a=p;
    while(p<s.size() && (isdigit((unsigned char)s[p])||s[p]=='.'||s[p]=='-'||s[p]=='+'||s[p]=='e'||s[p]=='E')) p++;
    return strtod(s.substr(a,p-a).c_str(),nullptr);
}
static bool isNumber(const string&t){
    if(t.empty()) return false;
    char*end=nullptr; strtod(t.c_str(),&end); return end==t.c_str()+t.size();
}
struct NameMap { unordered_map<string,int> id; vector<string> names; bool hasZero=false;
    int get(const string&s){ auto it=id.find(s); if(it!=id.end()) return it->second; int k=(int)names.size(); id[s]=k; names.push_back(s); return k; }
};
static int parseNode(const string&s, size_t&p, RawTree&t, NameMap&nm, bool first){
    skipws(s,p);
    int v=(int)t.ch.size();
    t.ch.push_back({}); t.len.push_back(0); t.sup.push_back(1.0); t.tax.push_back(-1); t.hasSup.push_back(0);
    if(p<s.size() && s[p]=='('){
        p++;
        int u=parseNode(s,p,t,nm,first); t.ch[v].push_back(u);
        while(p<s.size() && s[p]==','){ p++; u=parseNode(s,p,t,nm,first); t.ch[v].push_back(u); }
        if(p<s.size() && s[p]==')') p++;
        skipws(s,p);
        if(p<s.size() && !strchr("),:;",s[p])){
            string lab=readLabel(s,p);
            if(isNumber(lab)){ t.sup[v]=strtod(lab.c_str(),nullptr); t.hasSup[v]=1; }
        }
        if(p<s.size() && s[p]==':'){ p++; t.len[v]=readNum(s,p); }
    } else {
        string lab=readLabel(s,p);
        int id;
        if(first) id=nm.get(lab);
        else { auto it=nm.id.find(lab); if(it==nm.id.end()){ /* new taxon: append */ id=nm.get(lab);} else id=it->second; }
        t.tax[v]=id;
        if(p<s.size() && s[p]==':'){ p++; t.len[v]=readNum(s,p); }
    }
    return v;
}
static RawTree parseTree(const string&s, NameMap&nm, bool first){
    RawTree t; size_t p=0; int root=parseNode(s,p,t,nm,first);
    int n=(int)nm.names.size();
    t.n=n;
    int N=(int)t.ch.size();
    t.adj.assign(N,{}); t.elen.assign(N,{}); t.esup.assign(N,{});
    auto addEdge=[&](int v,int u,double L,double S){
        t.adj[v].push_back({u,L,S}); t.adj[u].push_back({v,L,S});
        t.elen[v].push_back(L); t.elen[u].push_back(L);
        t.esup[v].push_back(S); t.esup[u].push_back(S);
        t.edges.push_back({v,u});
    };
    // build graph: parent v -> child u, edge carries child's support/length
    for(int v=0;v<N;v++) for(int u: t.ch[v])
        addEdge(v,u,t.len[u], (t.ch[u].empty()?1.0:(t.hasSup[u]?t.sup[u]:1.0)));
    // leaf vertices must be 0..n-1; currently they are arbitrary ids. Build remap.
    vector<int> rid(N,-1); int c=0;
    for(int v=0;v<N;v++) if(t.ch[v].empty()) rid[v]=t.tax[v];
    for(int v=0;v<N;v++) if(!t.ch[v].empty()) rid[v]=n+(c++);
    RawTree o; o.n=n;
    int NN=n+c;
    o.ch.assign(NN,{}); o.len.assign(NN,0); o.sup.assign(NN,1); o.tax.assign(NN,-1); o.hasSup.assign(NN,0);
    o.adj.assign(NN,{}); o.elen.assign(NN,{}); o.esup.assign(NN,{});
    for(int v=0;v<N;v++){
        int a=rid[v];
        o.tax[a]=t.tax[v]; o.hasSup[a]=t.hasSup[v]; o.sup[a]=t.sup[v]; o.len[a]=t.len[v];
    }
    for(size_t e=0;e<t.edges.size();e++){
        int v=t.edges[e][0], u=t.edges[e][1];
        int a=rid[v], b=rid[u];
        double L=t.len[u];
        double S=(t.ch[u].empty()?1.0:(t.hasSup[u]?t.sup[u]:1.0));
        o.adj[a].push_back({b,L,S}); o.adj[b].push_back({a,L,S});
        o.elen[a].push_back(L); o.elen[b].push_back(L);
        o.esup[a].push_back(S); o.esup[b].push_back(S);
    }
    (void)root;
    return o;
}

// --------------------------- contracted component graph ---------------------------
struct CEdge { int to; double len; double sup; };
struct CompGraph {
    int n=0, C=0;
    vector<vector<CEdge>> adj;
};
struct DSU { vector<int> p; DSU(int n=0):p(n){ iota(p.begin(),p.end(),0);} int f(int x){while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x;} void u(int a,int b){a=f(a);b=f(b);if(a!=b)p[b]=a;} };
static CompGraph makeCompGraph(const RawTree&t, const Params&pr){
    int n=t.n, N=(int)t.adj.size();
    DSU d(N);
    // suppress degree-2 internal vertices (Newick serialization root)
    for(int v=n;v<N;v++) if((int)t.adj[v].size()==2){
        int u=t.adj[v][0].to;
        if(u<n) u=t.adj[v][1].to;
        d.u(v,u);
    }
    // contract unreliable internal edges
    for(int v=n;v<N;v++) for(size_t k=0;k<t.adj[v].size();k++){
        int u=t.adj[v][k].to;
        if(u<n || u<=v) continue;
        double L=t.elen[v][k], S=t.esup[v][k];
        bool bad=false;
        if(pr.st>0 && S<pr.st) bad=true;
        if(pr.lt>0 && L<pr.lt) bad=true;
        if(bad) d.u(v,u);
    }
    vector<int> cid(N,-1); int c=n;
    for(int i=0;i<n;i++) cid[d.f(i)]=i;
    for(int v=0;v<N;v++){ int r=d.f(v); if(cid[r]<0) cid[r]=c++; }
    CompGraph g; g.n=n; g.C=c; g.adj.assign(c,{});
    unordered_map<uint64_t,pair<double,double>> em; // key=(a,b) -> (len,sup)
    for(int v=0;v<N;v++) for(size_t k=0;k<t.adj[v].size();k++){
        int u=t.adj[v][k].to; if(u<v) continue;
        int a=cid[d.f(v)], b=cid[d.f(u)];
        if(a==b) continue;
        if(a>b) swap(a,b);
        uint64_t key=((uint64_t)a<<32)|(uint64_t)b;
        double L=t.elen[v][k], S=t.esup[v][k];
        auto it=em.find(key);
        if(it==em.end()) em[key]={L,S};
        else { it->second.first+=L; it->second.second=min(it->second.second,S); }
    }
    for(auto&kv: em){
        int a=(int)(kv.first>>32), b=(int)(kv.first&0xffffffffULL);
        g.adj[a].push_back({b,kv.second.first,kv.second.second});
        g.adj[b].push_back({a,kv.second.first,kv.second.second});
    }
    return g;
}

// ------------------------------ vote forest ------------------------------
struct Forest {
    int n=0, ning=0, N=0;
    vector<M> masks;
    vector<int> parent;
    vector<int> intIds;
    vector<double> sz, wt;
    vector<int> childIds, childPar;
    double scale=1.0;
    mutable vector<int> x1,x2;
    mutable vector<double> cs;
    double h(const M&A,const M&B,int sa,int sb) const {
        if(x1.empty()){ x1.assign(N,0); x2.assign(N,0); cs.assign(N,0.0); }
        for(int i=0;i<N;i++){
            int a=0,b=0;
            for(int k=0;k<W;k++){ a+=__builtin_popcountll(masks[i].w[k]&A.w[k]); b+=__builtin_popcountll(masks[i].w[k]&B.w[k]); }
            x1[i]=a; x2[i]=b;
        }
        fill(cs.begin(),cs.end(),0.0);
        for(size_t j=0;j<childIds.size();j++){
            int c=childIds[j], p=childPar[j];
            cs[p]+=(double)x1[c]*x2[c];
        }
        double tot=0;
        int sab=sa+sb;
        for(size_t j=0;j<intIds.size();j++){
            int id=intIds[j];
            double P=(double)x1[id]*x2[id]-cs[id];
            if(P<=0) continue;
            double Q=(double)ning-sab-sz[j]+x1[id]+x2[id];
            if(Q<=0) continue;
            tot+=wt[j]*P*Q;
        }
        return tot*scale;
    }
};
struct RootInfo { vector<int> par, nchild; vector<M> mask; vector<double> sup; vector<int> sz; vector<vector<int>> ch; vector<int> order; };
static RootInfo rootGraph(const CompGraph&g,int root){
    int C=g.C;
    RootInfo r;
    r.par.assign(C,-2); r.sup.assign(C,1.0); r.ch.assign(C,{}); r.nchild.assign(C,0);
    r.par[root]=-1;
    vector<int> st={root};
    while(!st.empty()){
        int v=st.back(); st.pop_back(); r.order.push_back(v);
        for(auto&e:g.adj[v]) if(r.par[e.to]==-2){ r.par[e.to]=v; r.sup[e.to]=e.sup; r.nchild[v]++; r.ch[v].push_back(e.to); st.push_back(e.to); }
    }
    r.mask.assign(C,M()); r.sz.assign(C,0);
    for(int i=0;i<g.n;i++){ r.mask[i]=bitmask(i); r.sz[i]=1; }
    for(int ii=(int)r.order.size()-1;ii>=0;ii--){
        int v=r.order[ii];
        if(v>=g.n){
            M m; int s=0;
            for(int u:r.ch[v]){ m=mor(m,r.mask[u]); s+=r.sz[u]; }
            r.mask[v]=m; r.sz[v]=s;
        }
    }
    return r;
}
static Forest makeForest(const CompGraph&g,int root,const Params&pr){
    RootInfo r=rootGraph(g,root);
    Forest f; f.n=g.n; f.ning=g.n-1; f.N=g.C;
    f.masks=r.mask; f.parent=r.par;
    f.x1.clear();
    double V=0;
    for(int v=0;v<g.C;v++){
        if(r.par[v]>=0) { f.childIds.push_back(v); f.childPar.push_back(r.par[v]); }
        if(r.nchild[v]>=2){
            f.intIds.push_back(v);
            double s=r.sup[v];
            double w;
            if(pr.centered) w=pow(max(0.0,(s-1.0/3.0)/(2.0/3.0)),pr.alpha);
            else w=pow(max(0.0,s),pr.alpha);
            if(r.par[v]<0) w=0;
            f.wt.push_back(w); f.sz.push_back((double)r.sz[v]);
            double pairs=(double)r.sz[v]*(r.sz[v]-1)/2.0;
            for(int u:r.ch[v]) pairs-=(double)r.sz[u]*(r.sz[u]-1)/2.0;
            V+=w*pairs*max(0.0,(double)f.ning-r.sz[v]);
        }
    }
    f.scale = V>0 ? 1e7/V : 0.0;
    return f;
}

// --------------------------- distances / NJ / consensus ---------------------------
typedef vector<vector<double>> DMat;
static DMat distances(const CompGraph&g, bool patristic){
    int n=g.n, C=g.C;
    DMat D(n, vector<double>(n,0.0));
    vector<double> d(C);
    vector<char> vis(C);
    for(int s=0;s<n;s++){
        fill(d.begin(),d.end(),0.0); fill(vis.begin(),vis.end(),0);
        vis[s]=1; vector<int> st={s};
        while(!st.empty()){
            int v=st.back(); st.pop_back();
            for(auto&e:g.adj[v]) if(!vis[e.to]){
                vis[e.to]=1;
                double w = patristic? e.len : ((v>=n && e.to>=n)?1.0:((v>=n||e.to>=n)?0.5:0.0));
                d[e.to]=d[v]+w; st.push_back(e.to);
            }
        }
        for(int j=0;j<n;j++) D[s][j]=d[j];
    }
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){ double v=0.5*(D[i][j]+D[j][i]); D[i][j]=D[j][i]=v; }
    return D;
}
static bool compatible(const M&A,const M&B,const M&full){
    M na=mandnot(full,A), nb=mandnot(full,B);
    if(mzero(mand(A,B))) return true;
    if(mzero(mand(A,nb))) return true;
    if(mzero(mand(na,B))) return true;
    if(mzero(mand(na,nb))) return true;
    return false;
}
static M orientAway0(M m,const M&full){ return (m.w[0]&1ULL)? mandnot(full,m) : m; }
static vector<M> njTree(const DMat&D,int n){
    int SZ=2*n+5;
    vector<vector<double>> A(SZ, vector<double>(SZ,0.0));
    vector<M> cl(SZ,M());
    for(int i=0;i<n;i++){ cl[i]=bitmask(i); for(int j=0;j<n;j++) A[i][j]=D[i][j]; }
    vector<int> active(n); iota(active.begin(),active.end(),0);
    vector<M> res; int nxt=n; M full; for(int i=0;i<n;i++) full=mor(full,bitmask(i));
    while((int)active.size()>3){
        int m=(int)active.size();
        vector<double> r(m,0);
        for(int i=0;i<m;i++) for(int j=0;j<m;j++) r[i]+=A[active[i]][active[j]];
        double best=1e300; int bi=-1,bj=-1;
        for(int i=0;i<m;i++) for(int j=i+1;j<m;j++){
            double q=(m-2)*A[active[i]][active[j]]-r[i]-r[j];
            if(q<best){best=q;bi=i;bj=j;}
        }
        int i=active[bi], j=active[bj];
        M nm=mor(cl[i],cl[j]);
        M o=orientAway0(nm,full);
        if(mpop(o)>=2 && mpop(o)<=n-2) res.push_back(o);
        cl[nxt]=nm;
        for(int k: active) if(k!=i&&k!=j){
            double v=0.5*(A[i][k]+A[j][k]-A[i][j]);
            A[nxt][k]=A[k][nxt]=v;
        }
        A[nxt][nxt]=0;
        vector<int> na; for(int k:active) if(k!=i&&k!=j) na.push_back(k);
        na.push_back(nxt); active.swap(na);
        nxt++;
    }
    return res;
}
static vector<pair<M,double>> consensus(const vector<CompGraph>&graphs,int n,const M&full){
    unordered_map<M,double,MHash,MEq> acc;
    for(const auto&g: graphs){
        RootInfo r=rootGraph(g,0);
        for(int v=0;v<g.C;v++){
            if(r.par[v]<0 || r.nchild[v]<2) continue;
            int s=r.sz[v];
            if(s<2 || s>n-2) continue;
            M m=orientAway0(r.mask[v],full);
            double w=pow(max(0.0,r.sup[v]),1.0);
            acc[m]+=w;
        }
    }
    vector<pair<M,double>> v(acc.begin(),acc.end());
    sort(v.begin(),v.end(),[](auto&a,auto&b){ return a.second>b.second; });
    vector<M> chosen;
    for(auto&p:v){
        if((int)chosen.size()>=n-3) break;
        bool ok=true;
        for(auto&c:chosen) if(!compatible(p.first,c,full)){ok=false;break;}
        if(ok) chosen.push_back(p.first);
    }
    vector<pair<M,double>> out;
    for(auto&c:chosen) out.push_back({c,1.0});
    return out;
}

// ------------------------------ species tree ------------------------------
struct STree {
    int n; M full, rootMask;
    const vector<Forest>* fs;
    vector<array<int,2>> ch;
    vector<int> par;
    vector<M> mask;
    vector<int> sz;
    vector<double> hv;
    vector<int> ints;
    int root=-1;
    STree(int n_,const vector<Forest>* f):n(n_),fs(f){
        for(int i=0;i<n;i++) full=mor(full,bitmask(i));
        rootMask=mandnot(full,bitmask(0));
    }
    double hq(const M&A,const M&B,int sa,int sb) const {
        double s=0; for(const auto&f:*fs) s+=f.h(A,B,sa,sb); return s;
    }
    int newNode(const M&m,int p){ int v=(int)ch.size(); ch.push_back({-1,-1}); par.push_back(p); mask.push_back(m); sz.push_back(mpop(m)); hv.push_back(0); return v; }
    void build(const vector<M>&clades){
        vector<M> cs;
        unordered_map<M,char,MHash,MEq> seen;
        for(auto&c:clades){
            if(meq(c,rootMask)||c.w[0]&1ULL) continue;
            int s=mpop(c); if(s<2||s>n-2) continue;
            if(seen.find(c)==seen.end()){ seen[c]=1; cs.push_back(c); }
        }
        std::function<int(M)> rec=[&](M L)->int{
            int v=newNode(L,-1);
            if(sz[v]==1) return v;
            M best; int bs=0;
            for(auto&c:cs){
                int s=mpop(c);
                if(s<=bs||s>=sz[v]) continue;
                if(meq(c,L)) continue;
                if(!mzero(mandnot(c,L))) continue;      // c subset of L?
                if(mzero(mandnot(L,c))) continue;
                best=c; bs=s;
            }
            int a,b;
            if(bs>=2){
                a=rec(best); b=rec(mandnot(L,best));
            } else {
                uint64_t low=0; int idx=-1;
                for(int k=0;k<W;k++) if(L.w[k]){ low=L.w[k]&(~L.w[k]+1); idx=k*64+__builtin_ctzll(L.w[k]); break; }
                M leaf=bitmask(idx);
                a=rec(leaf); b=rec(mxor(L,leaf));
            }
            ch[v]={a,b}; par[a]=v; par[b]=v;
            return v;
        };
        root=rec(rootMask);
        for(int v=0;v<(int)ch.size();v++) if(ch[v][0]>=0) ints.push_back(v);
        for(int v:ints) hv[v]=hq(mask[ch[v][0]],mask[ch[v][1]],sz[ch[v][0]],sz[ch[v][1]]);
    }
    double total() const { double s=0; for(int v:ints) s+=hv[v]; return s; }
    bool tryNNI(int v,double&delta,bool apply){
        int p=par[v]; if(p<0||ch[v][0]<0||ch[p][0]<0) return false;
        int sib=(ch[p][0]==v?ch[p][1]:ch[p][0]);
        for(int k=0;k<2;k++){
            int moved=ch[v][k], other=ch[v][1-k];
            M nm=mor(mask[other],mask[sib]);
            double hV=hq(mask[other],mask[sib],sz[other],sz[sib]);
            double hP=hq(nm,mask[moved],mpop(nm),sz[moved]);
            double d=(hV+hP)-hv[v]-hv[p];
            if(d>1e-7){
                if(apply){
                    ch[v]={other,sib}; par[sib]=v;
                    ch[p]={v,moved}; par[moved]=p;
                    mask[v]=nm; sz[v]=mpop(nm);
                    hv[v]=hV; hv[p]=hP;
                }
                delta=d; return true;
            }
        }
        return false;
    }
    int nniPass(int maxRounds, mt19937&rng, double timeLimit, steady_clock::time_point t0){
        int moves=0;
        for(int round=0;round<maxRounds;round++){
            bool any=false;
            vector<int> ord=ints; shuffle(ord.begin(),ord.end(),rng);
            for(int v:ord){ double d=0; if(tryNNI(v,d,true)){ moves++; any=true; } }
            if(!any) break;
            if(duration_cast<duration<double>>(steady_clock::now()-t0).count()>timeLimit) break;
        }
        return moves;
    }
    void forceNNI(int v,int kk){
        int p=par[v]; if(p<0||ch[p][0]<0) return;
        int sib=(ch[p][0]==v?ch[p][1]:ch[p][0]);
        int moved=ch[v][kk], other=ch[v][1-kk];
        M nm=mor(mask[other],mask[sib]);
        double hV=hq(mask[other],mask[sib],sz[other],sz[sib]);
        double hP=hq(nm,mask[moved],mpop(nm),sz[moved]);
        ch[v]={other,sib}; par[sib]=v;
        ch[p]={v,moved}; par[moved]=p;
        mask[v]=nm; sz[v]=mpop(nm);
        hv[v]=hV; hv[p]=hP;
    }
    void perturb(int k,mt19937&rng){
        for(int i=0;i<k;i++){
            int v=ints[rng()%ints.size()];
            forceNNI(v,rng()%2);
        }
    }
    vector<M> getClades() const {
        vector<M> v;
        for(int u:ints) if(u!=root){ M m=mask[u]; if(mpop(m)>=2 && mpop(m)<=n-2) v.push_back(m); }
        return v;
    }
    string newick(const vector<string>&names) const {
        std::function<string(int)> rec=[&](int v)->string{
            if(ch[v][0]<0){
                int i=-1; for(int k=0;k<W;k++) if(mask[v].w[k]){ i=k*64+__builtin_ctzll(mask[v].w[k]); break; }
                const string&nm=names[i];
                if(nm.find_first_of("(),:;'[] \t\n")==string::npos) return nm;
                string q="'"; for(char c:nm){ q+=c; if(c=='\'') q+=c; } q+="'"; return q;
            }
            return "("+rec(ch[v][0])+","+rec(ch[v][1])+")";
        };
        const string&n0=names[0];
        string r0=n0;
        if(n0.find_first_of("(),:;'[] \t\n")!=string::npos){ r0="'"; for(char c:n0){r0+=c; if(c=='\'')r0+=c;} r0+="'"; }
        return "("+r0+","+rec(ch[root][0])+","+rec(ch[root][1])+");\n";
    }
};

// ------------------------------ main ------------------------------
int main(int argc,char**argv){
    auto t0=steady_clock::now();
    if(argc<3){ fprintf(stderr,"usage: species_tree INPUT OUTPUT [MODE]\n"); return 2; }
    int mode = argc>3? atoi(argv[3]) : 0;
    double timeLimit = argc>4? atof(argv[4]) : 55.0;
    ifstream in(argv[1]);
    if(!in){ fprintf(stderr,"cannot open input\n"); return 2; }
    string all((istreambuf_iterator<char>(in)),istreambuf_iterator<char>());
    vector<string> lines;
    { string cur; for(char c: all){ if(c==';'){ if(!cur.empty()) lines.push_back(cur); cur.clear(); } else cur+=c; } }
    if(lines.empty()){ fprintf(stderr,"no trees\n"); return 2; }
    NameMap nm;
    nm.get("0");                  // outgroup is always taxon 0
    vector<RawTree> raws;
    raws.reserve(lines.size());
    for(size_t i=0;i<lines.size();i++) raws.push_back(parseTree(lines[i],nm,i==0));
    int n=(int)nm.names.size();
    M full; for(int i=0;i<n;i++) full=mor(full,bitmask(i));
    Params p1,p2;
    // mode variants: different gene-tree contraction / weighting ensembles
    bool ensemble=true;
    switch(mode){
        case 1: p1={0.5,-1,1.0,false}; ensemble=false; break;
        case 2: p1={0.5,-1,2.0,false}; ensemble=false; break;
        case 3: p1={0.9,-1,0.0,false}; ensemble=false; break;
        case 4: p1={-1,-1,1.0,false}; ensemble=false; break;
        case 5: p1={-1,-1,2.0,false}; ensemble=false; break;
        case 6: p1={0.7,-1,0.0,false}; ensemble=false; break;
        case 7: p1={0.5,-1,1.0,true}; ensemble=false; break;
        default: p1={0.5,-1,1.0,false}; p2={0.9,-1,0.0,false}; ensemble=true; break;
    }
    vector<CompGraph> graphs; graphs.reserve(raws.size());
    for(auto&r:raws) graphs.push_back(makeCompGraph(r,p1));
    // build a combined Forest by merging per-tree forests
    auto mergeForests=[&](const vector<CompGraph>&gs,int root,const Params&pr){
        Forest F; F.n=n; F.ning=n-1; F.N=0; F.scale=0;
        double sumScale=0;
        for(const auto&g:gs){
            Forest f=makeForest(g,root,pr);
            int off=F.N;
            for(auto&m:f.masks) F.masks.push_back(m);
            for(int p:f.parent) F.parent.push_back(p<0?-1:p+off);
            for(int id:f.intIds) F.intIds.push_back(id+off);
            for(double s:f.sz) F.sz.push_back(s);
            for(double w:f.wt) F.wt.push_back(w);
            for(size_t q=0;q<f.childIds.size();q++){ F.childIds.push_back(f.childIds[q]+off); F.childPar.push_back(f.childPar[q]+off); }
            // scale of each forest must be global, not per-tree; recompute below
            F.N+=f.N;
        }
        // uniform global normalization: use average of per-tree scale factors
        double sc=0; int cnt=0;
        for(const auto&g:gs){ Forest f=makeForest(g,root,pr); sc+=f.scale; cnt++; }
        F.scale = cnt? sc/cnt : 0;
        return F;
    };
    Forest F1=mergeForests(graphs,0,p1);
    vector<Forest> fs; fs.push_back(F1);
    if(ensemble){
        vector<CompGraph> g2; g2.reserve(raws.size());
        for(auto&r:raws) g2.push_back(makeCompGraph(r,p2));
        Forest F2=mergeForests(g2,0,p2);
        fs.push_back(F2);
    }
    // distance-based starting trees
    DMat Dp(n,vector<double>(n,0)), Dt(n,vector<double>(n,0));
    for(const auto&g:graphs){
        DMat a=distances(g,true), b=distances(g,false);
        for(int i=0;i<n;i++) for(int j=0;j<n;j++){ Dp[i][j]+=a[i][j]/graphs.size(); Dt[i][j]+=b[i][j]/graphs.size(); }
    }
    vector<M> njp=njTree(Dp,n), njt=njTree(Dt,n);
    vector<pair<M,double>> cons=consensus(graphs,n,full);
    vector<M> consc; for(auto&p:cons) consc.push_back(p.first);
    vector<vector<M>> starts;
    starts.push_back(njt); starts.push_back(njp); starts.push_back(consc);
    // a hybrid: consensus clades first, fill with NJt
    { vector<M> h=consc; for(auto&m:njt) h.push_back(m); starts.push_back(h); }
    if(getenv("DUMP_STARTS")){
        for(int si=0;si<3;si++){
            STree T(n,&fs); T.build(starts[si]);
            ofstream o(string("/workspace/work/dump")+to_string(si)+".tre"); o<<T.newick(nm.names);
        }
    }
    mt19937 rng(1);
    double bestScore=-1e300;
    vector<M> bestClades;
    vector<pair<double,vector<M>>> optima;
    auto consider=[&](STree&T){
        double sc=T.total();
        optima.push_back({sc,T.getClades()});
        if(sc>bestScore){ bestScore=sc; bestClades=T.getClades(); }
    };
    for(auto&st:starts){
        STree T(n,&fs); T.build(st);
        T.nniPass(40,rng,timeLimit,t0);
        consider(T);
        fprintf(stderr,"start %.0f score %.0f time %.1f\n",(double)optima.size(),T.total(),duration_cast<duration<double>>(steady_clock::now()-t0).count());
    }
    // consensus of optima (weighted) as another start
    {
        unordered_map<M,double,MHash,MEq> acc;
        for(auto&o:optima) for(auto&c:o.second) acc[c]+=1.0;
        vector<pair<M,double>> v(acc.begin(),acc.end());
        sort(v.begin(),v.end(),[](auto&a,auto&b){return a.second>b.second;});
        vector<M> chosen;
        for(auto&p:v){ if((int)chosen.size()>=n-3) break; bool ok=true; for(auto&c:chosen) if(!compatible(p.first,c,full)){ok=false;break;} if(ok) chosen.push_back(p.first); }
        for(auto&m:njt) chosen.push_back(m);
        STree T(n,&fs); T.build(chosen); T.nniPass(40,rng,timeLimit,t0); consider(T);
    }
    // iterated local search from best tree
    {
        STree T(n,&fs); T.build(bestClades);
        double sc=T.total();
        for(int it=0;it<200;it++){
            if(duration_cast<duration<double>>(steady_clock::now()-t0).count()>timeLimit) break;
            STree C=T;
            int k=2+rng()%8;
            C.perturb(k,rng);
            C.nniPass(40,rng,timeLimit,t0);
            double s=C.total();
            if(s>sc+1e-7){ T=C; sc=s; }
            if(s>bestScore){ bestScore=s; bestClades=C.getClades(); }
            if(s>0.985*bestScore) optima.push_back({s,C.getClades()});
            if(it%10==0) fprintf(stderr,"ILS %d score %.0f best %.0f time %.1f\n",it,sc,bestScore,duration_cast<duration<double>>(steady_clock::now()-t0).count());
        }
    }
    // Majority consensus of the best local optima: reduces overfitting to the
    // surrogate objective. Ties are broken by the highest-scoring tree's clades.
    sort(optima.begin(),optima.end(),[](auto&a,auto&b){return a.first>b.first;});
    if((int)optima.size()>40) optima.resize(40);
    unordered_map<M,double,MHash,MEq> acc;
    for(auto&o:optima) for(auto&c:o.second) acc[c]+=1.0;
    vector<pair<M,double>> cv(acc.begin(),acc.end());
    sort(cv.begin(),cv.end(),[](auto&a,auto&b){return a.second>b.second;});
    vector<M> voted;
    for(auto&p:cv){ if((int)voted.size()>=n-3) break; bool ok=true; for(auto&c:voted) if(!compatible(p.first,c,full)){ok=false;break;} if(ok) voted.push_back(p.first); }
    for(auto&c:bestClades) voted.push_back(c);
    for(auto&c:njt) voted.push_back(c);
    STree Cons(n,&fs); Cons.build(voted);
    STree Best(n,&fs); Best.build(bestClades);
    if(getenv("DUMP_CONS")){
        ofstream o(string(argv[2])+".cons"); o<<Cons.newick(nm.names);
        ofstream ob(string(argv[2])+".best"); ob<<Best.newick(nm.names);
        fprintf(stderr,"cons score %.0f best score %.0f optima %zu\n",Cons.total(),Best.total(),optima.size());
    }
    STree &Final = (Cons.total() > Best.total()*1.001) ? Cons : Best;
    ofstream out(argv[2]);
    if(!out){ fprintf(stderr,"cannot open output\n"); return 2; }
    out<<Final.newick(nm.names);
    out.close();
    fprintf(stderr,"done score %.0f time %.1f\n",Final.total(),duration_cast<duration<double>>(steady_clock::now()-t0).count());
    return 0;
}
