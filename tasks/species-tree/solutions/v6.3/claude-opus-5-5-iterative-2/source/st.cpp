// Species tree inference from estimated gene trees: weighted bipartition-quartet
// score optimised by an ASTRAL-style dynamic programme over a cluster search space
// that is iteratively expanded with rearrangement neighbourhoods of the incumbent.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;
static const int NW = 4;               // up to 256 taxa
struct BS { u64 w[NW]; };
static inline bool operator==(const BS&a,const BS&b){ for(int i=0;i<NW;i++) if(a.w[i]!=b.w[i]) return false; return true; }
static inline u64 mix64(u64 x){ x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }
struct BSHash { size_t operator()(const BS&b) const { u64 h=0; for(int i=0;i<NW;i++) h=mix64(h^b.w[i]); return h; } };
static inline int pc(const BS&a){ int s=0; for(int i=0;i<NW;i++) s+=__builtin_popcountll(a.w[i]); return s; }
static inline int pcand(const BS&a,const BS&b){ int s=0; for(int i=0;i<NW;i++) s+=__builtin_popcountll(a.w[i]&b.w[i]); return s; }
static inline BS bor(const BS&a,const BS&b){ BS r; for(int i=0;i<NW;i++) r.w[i]=a.w[i]|b.w[i]; return r; }
static inline BS bandn(const BS&a,const BS&b){ BS r; for(int i=0;i<NW;i++) r.w[i]=a.w[i]&~b.w[i]; return r; }
static inline bool subset(const BS&a,const BS&b){ for(int i=0;i<NW;i++) if(a.w[i]&~b.w[i]) return false; return true; }
static inline bool isempty(const BS&a){ for(int i=0;i<NW;i++) if(a.w[i]) return false; return true; }
static inline int lowbit(const BS&a){ for(int i=0;i<NW;i++) if(a.w[i]) return i*64+__builtin_ctzll(a.w[i]); return -1; }
static inline void setb(BS&a,int i){ a.w[i>>6]|=1ULL<<(i&63); }
static inline BS zero(){ BS r; memset(&r,0,sizeof r); return r; }

static chrono::steady_clock::time_point T0;
static double elapsed(){ return chrono::duration<double>(chrono::steady_clock::now()-T0).count(); }
static double envd(const char*k,double d){ const char*v=getenv(k); return v?atof(v):d; }

// ---------------- Newick parsing ----------------
struct PNode { vector<int> ch; string label; double len=NAN; int parent=-1; };
struct PTree { vector<PNode> nd; int root=-1; };
static bool parseNewick(const string&s, PTree&t){
    size_t p=0; t.nd.clear();
    auto skip=[&](){ while(p<s.size() && isspace((unsigned char)s[p])) p++; };
    function<int()> sub=[&]()->int{
        skip(); int id=t.nd.size(); t.nd.push_back(PNode());
        if(p<s.size() && s[p]=='('){
            p++;
            while(true){
                int c=sub(); if(c<0) return -1;
                t.nd[c].parent=id; t.nd[id].ch.push_back(c);
                skip();
                if(p<s.size() && s[p]==','){ p++; continue; }
                if(p<s.size() && s[p]==')'){ p++; break; }
                return -1;
            }
        }
        skip();
        string lab;
        if(p<s.size() && s[p]=='\''){
            p++; while(p<s.size()){ if(s[p]=='\''){ if(p+1<s.size()&&s[p+1]=='\''){lab+='\'';p+=2;continue;} p++; break;} lab+=s[p++]; }
        } else {
            while(p<s.size() && !strchr("(),:;[",s[p]) && !isspace((unsigned char)s[p])) lab+=s[p++];
        }
        t.nd[id].label=lab;
        skip();
        if(p<s.size() && s[p]=='['){ while(p<s.size() && s[p]!=']') p++; if(p<s.size()) p++; skip(); }
        if(p<s.size() && s[p]==':'){
            p++; skip(); size_t q=p; while(p<s.size() && (isdigit((unsigned char)s[p])||strchr("+-.eE",s[p]))) p++;
            t.nd[id].len = (p>q)? atof(s.substr(q,p-q).c_str()) : NAN;
        }
        skip();
        if(p<s.size() && s[p]=='['){ while(p<s.size() && s[p]!=']') p++; if(p<s.size()) p++; skip(); }
        return id;
    };
    int r=sub(); if(r<0) return false; t.root=r; return true;
}

// ---------------- global data ----------------
static int n=0, R=0;
static vector<string> names;
static unordered_map<string,int> name2id;
static BS UALL;  // all taxa except R

struct GEdge { BS cl; double sup, len; int gene; double p; double praw; };
static vector<GEdge> gedges;   // internal edges of gene trees, cluster oriented away from R
static int ngenes=0;
struct GTree { vector<BS> cl; vector<int> par, leaf, edge; vector<vector<int>> kids; vector<double> plen; }; // node 0 = R
static vector<GTree> gtrees;

// gene-tree -> unrooted adjacency, then rooted at leaf R; collect clusters of internal edges
static bool processGene(const PTree&t, int gi){
    int m=t.nd.size();
    vector<vector<pair<int,int>>> adj(m); // (nbr, edge id)
    vector<double> esup, elen;
    auto addE=[&](int a,int b,double s,double l){ int id=esup.size(); esup.push_back(s); elen.push_back(l); adj[a].push_back({b,id}); adj[b].push_back({a,id}); };
    auto supOf=[&](int v)->double{ if(t.nd[v].ch.empty()) return NAN; const string&L=t.nd[v].label; if(L.empty()) return NAN; char*e; double x=strtod(L.c_str(),&e); if(e==L.c_str()) return NAN; return x; };
    int root=t.root;
    for(int v=0;v<m;v++){ if(v==root) continue; int p=t.nd[v].parent; if(p==root && t.nd[root].ch.size()==2) continue; addE(p,v,supOf(v),t.nd[v].len); }
    if(t.nd[root].ch.size()==2){
        int a=t.nd[root].ch[0], b=t.nd[root].ch[1];
        double sa=supOf(a), sb=supOf(b), s=NAN;
        if(!std::isnan(sa)) s=sa; if(!std::isnan(sb)) s=std::isnan(s)?sb:max(s,sb);
        double la=t.nd[a].len, lb=t.nd[b].len, l=NAN;
        if(!std::isnan(la)||!std::isnan(lb)) l=(std::isnan(la)?0:la)+(std::isnan(lb)?0:lb);
        addE(a,b,s,l);
    }
    // map leaves
    vector<int> leafTax(m,-1); int rnode=-1;
    for(int v=0;v<m;v++) if(t.nd[v].ch.empty() && !(v==root && t.nd[root].ch.size()==2)){
        auto it=name2id.find(t.nd[v].label); if(it==name2id.end()) return false;
        leafTax[v]=it->second; if(it->second==R) rnode=v;
    }
    if(rnode<0) return false;
    // iterative DFS from rnode
    vector<int> order, par(m,-1), pe(m,-1); vector<char> vis(m,0);
    vector<int> st={rnode}; vis[rnode]=1;
    while(!st.empty()){ int v=st.back(); st.pop_back(); order.push_back(v); for(auto&q:adj[v]) if(!vis[q.first]){ vis[q.first]=1; par[q.first]=v; pe[q.first]=q.second; st.push_back(q.first);} }
    vector<BS> cl(m, zero());
    for(int i=order.size()-1;i>=0;i--){ int v=order[i]; if(leafTax[v]>=0 && v!=rnode) setb(cl[v],leafTax[v]); if(par[v]>=0) cl[par[v]]=bor(cl[par[v]],cl[v]); }
    GTree gt;
    // compact node numbering in DFS preorder
    vector<int> nid(m,-1);
    for(int v:order){ nid[v]=gt.cl.size(); gt.cl.push_back(cl[v]); gt.par.push_back(par[v]>=0?nid[par[v]]:-1); gt.leaf.push_back(leafTax[v]); gt.edge.push_back(-1); gt.plen.push_back(pe[v]>=0? elen[pe[v]] : 0.0); }
    gt.kids.assign(gt.cl.size(),{});
    for(size_t i=1;i<gt.cl.size();i++) gt.kids[gt.par[i]].push_back(i);
    for(int v=0;v<m;v++){
        if(v==rnode || par[v]<0 || leafTax[v]>=0) continue;
        if(par[v]==rnode) continue;
        int k=pc(cl[v]); if(k<2 || k>n-2) continue;
        gt.edge[nid[v]]=gedges.size();
        gedges.push_back({cl[v], esup[pe[v]], elen[pe[v]], gi, 0.0});
    }
    gtrees.push_back(move(gt));
    return true;
}

// ---------------- weighted bipartitions ----------------
static vector<BS> Ub; static vector<int> Usz; static vector<double> Uw;

static double scoreTrip(const BS&A1,int n1,const BS&A2,int n2){
    const long n3=n-n1-n2;
    double tot=0; size_t U=Ub.size();
    for(size_t u=0;u<U;u++){
        const BS&S=Ub[u];
        long x=pcand(A1,S), y=pcand(A2,S), z=Usz[u]-x-y;
        long X=n1-x, Y=n2-y, Z=n3-z;
        long N = x*(x-1)/2*Y*Z + X*(X-1)/2*y*z + y*(y-1)/2*X*Z + Y*(Y-1)/2*x*z + z*(z-1)/2*X*Y + Z*(Z-1)/2*x*y;
        tot += Uw[u]*(double)N;
    }
    return tot;
}

// ---------------- node-based (quartet) scoring ----------------
static int SMODE=1; // 0 = edge-sum, 1 = node/quartet
static double MU=0; // weight of edge-sum term mixed into node score
static double HMIX=0; // weight of plain quartet score mixed into hybrid score
static double HTLIM=75, HQRATIO=0; // hybrid time guard and hybrid/quartet magnitude ratio
static vector<BS> Pb; static vector<int> Psz; static unordered_map<BS,int,BSHash> Pid;
static int poolId(const BS&b){ auto it=Pid.find(b); if(it!=Pid.end()) return it->second; int id=Pb.size(); Pb.push_back(b); Psz.push_back(pc(b)); Pid[b]=id; return id; }
struct QNode { int own; int k0, k1; double w; }; // own pool id (-1 => top node, up={R}); kids [k0,k1) in Qkids
static vector<QNode> Qn; static vector<int> Qkids;
static unordered_map<u64,int> Qkey;
static vector<vector<pair<int,double>>> QnContrib; // per Q node: (gene, weight)
static void addQNode(int own, vector<int>&kids, double w, int gene){
    sort(kids.begin(),kids.end());
    u64 h=mix64((u64)(own+2)); for(int k:kids) h=mix64(h^(u64)k);
    auto it=Qkey.find(h);
    if(it!=Qkey.end()){ int id=it->second; Qn[id].w+=w; auto&cv=QnContrib[id]; if(!cv.empty()&&cv.back().first==gene) cv.back().second+=w; else cv.push_back({gene,w}); return; }
    Qkey[h]=Qn.size(); Qn.push_back({own,(int)Qkids.size(),(int)(Qkids.size()+kids.size()),w}); for(int k:kids) Qkids.push_back(k);
    QnContrib.push_back({{gene,w}});
}
// contract gene tree: keep[e] for internal edges (by node index); add contracted nodes with weight w
static void addContracted(const GTree&g, const vector<char>&keepNode, double w, int gene){
    // representative nodes: node 1 (top, child of R) and internal nodes whose edge is kept
    int m=g.cl.size();
    vector<int> kids; vector<int> st;
    for(int v=1;v<m;v++){
        if(g.leaf[v]>=0) continue;
        bool rep = (v==1) || keepNode[v];
        if(!rep) continue;
        kids.clear(); st.clear();
        for(int c:g.kids[v]) st.push_back(c);
        while(!st.empty()){ int c=st.back(); st.pop_back(); if(g.leaf[c]>=0) continue; if(keepNode[c]) kids.push_back(poolId(g.cl[c])); else for(int cc:g.kids[c]) st.push_back(cc); }
        int own = (v==1)? -1 : poolId(g.cl[v]);
        addQNode(own,kids,w,gene);
    }
}
static vector<int> pa, pb2; // per pool cluster counts in A1, A2
static double scoreTripQ(const BS&A1,int n1,const BS&A2,int n2){
    const double T0_=n1, T1_=n2, T2_=n-n1-n2;
    size_t P=Pb.size(); pa.resize(P); pb2.resize(P);
    for(size_t i=0;i<P;i++){ pa[i]=pcand(A1,Pb[i]); pb2[i]=pcand(A2,Pb[i]); }
    double tot=0;
    double c0[64],c1[64],c2[64];
    for(const QNode&q:Qn){
        int d=0;
        double s0=0,s1=0,s2=0;
        for(int k=q.k0;k<q.k1;k++){ int id=Qkids[k]; double a=pa[id], b=pb2[id], c=Psz[id]-a-b; if(d<63){ c0[d]=a;c1[d]=b;c2[d]=c; d++; } }
        // up direction
        double ua,ub,uc;
        if(q.own<0){ ua=0; ub=0; uc=1; }
        else { double a=pa[q.own], b=pb2[q.own], c=Psz[q.own]-a-b; ua=T0_-a; ub=T1_-b; uc=T2_-c; }
        c0[d]=ua; c1[d]=ub; c2[d]=uc; d++;
        double S01=0,S02=0,S12=0,Q0=0,Q1=0,Q2=0;
        for(int j=0;j<d;j++){ S01+=c0[j]*c1[j]; S02+=c0[j]*c2[j]; S12+=c1[j]*c2[j]; Q0+=c0[j]*(c0[j]-1)*0.5; Q1+=c1[j]*(c1[j]-1)*0.5; Q2+=c2[j]*(c2[j]-1)*0.5; }
        double r=0;
        for(int j=0;j<d;j++){
            double a=c0[j], b=c1[j], c=c2[j];
            double Ca=a*(a-1)*0.5, Cb=b*(b-1)*0.5, Cc=c*(c-1)*0.5;
            // alpha=0 (beta=1,gamma=2)
            r += Ca*((T1_-b)*(T2_-c)-S12+b*c) + b*c*(((T0_-a)*(T0_-a-1))*0.5-Q0+Ca);
            r += Cb*((T0_-a)*(T2_-c)-S02+a*c) + a*c*(((T1_-b)*(T1_-b-1))*0.5-Q1+Cb);
            r += Cc*((T0_-a)*(T1_-b)-S01+a*b) + a*b*(((T2_-c)*(T2_-c-1))*0.5-Q2+Cc);
        }
        tot+=q.w*r;
    }
    return tot;
}

// per-gene decomposition of the node score (out has ngenes entries)
static void scoreTripQG(const BS&A1,int n1,const BS&A2,int n2, double*out){
    const double T0_=n1, T1_=n2, T2_=n-n1-n2;
    size_t P=Pb.size(); pa.resize(P); pb2.resize(P);
    for(size_t i=0;i<P;i++){ pa[i]=pcand(A1,Pb[i]); pb2[i]=pcand(A2,Pb[i]); }
    for(int g=0;g<ngenes;g++) out[g]=0;
    double c0[64],c1[64],c2[64];
    for(size_t qi=0; qi<Qn.size(); qi++){
        const QNode&q=Qn[qi];
        int d=0;
        for(int k=q.k0;k<q.k1;k++){ int id=Qkids[k]; double a=pa[id], b=pb2[id], c=Psz[id]-a-b; if(d<63){ c0[d]=a;c1[d]=b;c2[d]=c; d++; } }
        double ua,ub,uc;
        if(q.own<0){ ua=0; ub=0; uc=1; }
        else { double a=pa[q.own], b=pb2[q.own], c=Psz[q.own]-a-b; ua=T0_-a; ub=T1_-b; uc=T2_-c; }
        c0[d]=ua; c1[d]=ub; c2[d]=uc; d++;
        double S01=0,S02=0,S12=0,Q0=0,Q1=0,Q2=0;
        for(int j=0;j<d;j++){ S01+=c0[j]*c1[j]; S02+=c0[j]*c2[j]; S12+=c1[j]*c2[j]; Q0+=c0[j]*(c0[j]-1)*0.5; Q1+=c1[j]*(c1[j]-1)*0.5; Q2+=c2[j]*(c2[j]-1)*0.5; }
        double r=0;
        for(int j=0;j<d;j++){
            double a=c0[j], b=c1[j], c=c2[j];
            double Ca=a*(a-1)*0.5, Cb=b*(b-1)*0.5, Cc=c*(c-1)*0.5;
            r += Ca*((T1_-b)*(T2_-c)-S12+b*c) + b*c*(((T0_-a)*(T0_-a-1))*0.5-Q0+Ca);
            r += Cb*((T0_-a)*(T2_-c)-S02+a*c) + a*c*(((T1_-b)*(T1_-b-1))*0.5-Q1+Cb);
            r += Cc*((T0_-a)*(T1_-b)-S01+a*b) + a*b*(((T2_-c)*(T2_-c-1))*0.5-Q2+Cc);
        }
        if(r==0) continue;
        for(auto&gw:QnContrib[qi]) out[gw.first]+=gw.second*r;
    }
}

// flat, vectorisable version of scoreTripQG
static vector<int> Fpool; static vector<double> Fsgn, Fb0, Fb1, Fb2; static vector<int> Fnode, Foff; static int FuallPool=-1;
static vector<double> FA,FB,FC,FS01,FS02,FS12,FQ0,FQ1,FQ2,FR,FNr; static vector<double> pad, pbd;
static vector<int> CgOff, CgGene; static vector<double> CgW;
static void buildFlat(){
    FuallPool=poolId(UALL);
    Foff.clear(); Fpool.clear(); Fsgn.clear(); Fb0.clear(); Fb1.clear(); Fb2.clear(); Fnode.clear();
    for(size_t qi=0;qi<Qn.size();qi++){
        const QNode&q=Qn[qi]; Foff.push_back(Fpool.size());
        for(int k=q.k0;k<q.k1;k++){ Fpool.push_back(Qkids[k]); Fsgn.push_back(1); Fb0.push_back(0); Fb1.push_back(0); Fb2.push_back(0); Fnode.push_back(qi); }
        Fpool.push_back(q.own<0? FuallPool: q.own); Fsgn.push_back(-1); Fb0.push_back(1); Fb1.push_back(1); Fb2.push_back(1); Fnode.push_back(qi);
    }
    Foff.push_back(Fpool.size());
    CgOff.clear(); CgGene.clear(); CgW.clear();
    for(size_t qi=0;qi<Qn.size();qi++){ CgOff.push_back(CgGene.size()); for(auto&gw:QnContrib[qi]){ CgGene.push_back(gw.first); CgW.push_back(gw.second);} }
    CgOff.push_back(CgGene.size());
    size_t K=Fpool.size(); FA.resize(K); FB.resize(K); FC.resize(K); FS01.resize(K); FS02.resize(K); FS12.resize(K); FQ0.resize(K); FQ1.resize(K); FQ2.resize(K); FR.resize(K);
    FNr.resize(Qn.size());
}
static void scoreTripQF(const BS&A1,int n1,const BS&A2,int n2, double*out){
    const double T0_=n1, T1_=n2, T2_=n-n1-n2;
    size_t P=Pb.size(); pad.resize(P); pbd.resize(P);
    for(size_t i=0;i<P;i++){ pad[i]=pcand(A1,Pb[i]); pbd[i]=pcand(A2,Pb[i]); }
    const size_t K=Fpool.size();
    double*__restrict a=FA.data(),*__restrict b=FB.data(),*__restrict c=FC.data();
    for(size_t k=0;k<K;k++){
        int p=Fpool[k]; double s=Fsgn[k]; double x=pad[p], y=pbd[p], z=Psz[p]-x-y;
        a[k]=Fb0[k]*T0_+s*x; b[k]=Fb1[k]*T1_+s*y; c[k]=Fb2[k]*T2_+s*z;
    }
    const size_t NQ=Qn.size();
    double*__restrict s01=FS01.data(),*__restrict s02=FS02.data(),*__restrict s12=FS12.data(),*__restrict q0=FQ0.data(),*__restrict q1=FQ1.data(),*__restrict q2=FQ2.data();
    for(size_t qi=0;qi<NQ;qi++){
        int o0=Foff[qi], o1=Foff[qi+1];
        double S01=0,S02=0,S12=0,Q0=0,Q1=0,Q2=0;
        for(int k=o0;k<o1;k++){ S01+=a[k]*b[k]; S02+=a[k]*c[k]; S12+=b[k]*c[k]; Q0+=a[k]*(a[k]-1); Q1+=b[k]*(b[k]-1); Q2+=c[k]*(c[k]-1); }
        Q0*=0.5; Q1*=0.5; Q2*=0.5;
        for(int k=o0;k<o1;k++){ s01[k]=S01; s02[k]=S02; s12[k]=S12; q0[k]=Q0; q1[k]=Q1; q2[k]=Q2; }
    }
    double*__restrict r=FR.data();
    for(size_t k=0;k<K;k++){
        double x=a[k], y=b[k], z=c[k];
        double Ca=x*(x-1)*0.5, Cb=y*(y-1)*0.5, Cc=z*(z-1)*0.5;
        double ra=T0_-x, rb=T1_-y, rc=T2_-z;
        r[k] = Ca*(rb*rc-s12[k]+y*z) + y*z*(ra*(ra-1)*0.5-q0[k]+Ca)
             + Cb*(ra*rc-s02[k]+x*z) + x*z*(rb*(rb-1)*0.5-q1[k]+Cb)
             + Cc*(ra*rb-s01[k]+x*y) + x*y*(rc*(rc-1)*0.5-q2[k]+Cc);
    }
    for(int g=0;g<ngenes;g++) out[g]=0;
    for(size_t qi=0;qi<NQ;qi++){
        double rr=0; for(int k=Foff[qi];k<Foff[qi+1];k++) rr+=r[k];
        if(rr==0) continue;
        for(int j=CgOff[qi];j<CgOff[qi+1];j++) out[CgGene[j]]+=CgW[j]*rr;
    }
}

// ---------------- hybrid scoring (support levels x terminal-length decay) ----------------
struct HGene {
    int m; vector<int> par, leaf, kidOff, kids; vector<double> dec; // node arrays (preorder), dec = decay of parent edge
    vector<int> cnOff, cnTop, cnDirOff, cnDirs; vector<double> cnW; // contracted nodes over all levels
};
static vector<HGene> hgenes;
static void buildHGenes(double lam, int normMode, const vector<double>&qth, const vector<double>&qw){
    for(auto&g:gtrees){
        HGene h; h.m=g.cl.size(); h.par=g.par; h.leaf=g.leaf; h.kidOff.assign(h.m+1,0);
        for(int v=0;v<h.m;v++) h.kidOff[v+1]=h.kidOff[v]+g.kids[v].size();
        for(int v=0;v<h.m;v++) for(int c:g.kids[v]) h.kids.push_back(c);
        double tl=0; int ne=0; for(int v=1;v<h.m;v++){ double l=g.plen[v]; if(!std::isnan(l)){ tl+=l; ne++; } }
        double L = normMode? max(1e-9, tl/max(1,ne)) : 1.0;
        if(normMode==2){ vector<double> ls; for(int v=1;v<h.m;v++){ double l=g.plen[v]; if(!std::isnan(l) && l>1e-5) ls.push_back(l); }
            if(!ls.empty()){ nth_element(ls.begin(),ls.begin()+ls.size()/2,ls.end()); L=max(1e-9,ls[ls.size()/2]); } }
        h.dec.assign(h.m,1.0);
        int htopo=(int)envd("HTOPO",0);
        for(int v=1;v<h.m;v++){ double l=g.plen[v]; if(std::isnan(l)) l=0; if(htopo==1) { l=1; } else if(htopo==2){ l = (l>1e-5)? 1:0; }
            h.dec[v]= (htopo? exp(-lam*l) : exp(-lam*l/L)); }
        // contracted nodes for each level
        vector<char> keep(h.m,0); vector<int> st;
        for(size_t k=0;k<qth.size();k++){
            for(int v=0;v<h.m;v++){ int e=g.edge[v]; keep[v]= e>=0 && gedges[e].p>=qth[k]; }
            for(int v=1;v<h.m;v++){
                if(g.leaf[v]>=0) continue;
                if(!(v==1 || keep[v])) continue;
                h.cnTop.push_back(v); h.cnW.push_back(qw[k]); h.cnDirOff.push_back(h.cnDirs.size());
                st.clear(); for(int c:g.kids[v]) st.push_back(c);
                while(!st.empty()){ int c=st.back(); st.pop_back(); if(g.leaf[c]>=0 || keep[c]) h.cnDirs.push_back(c); else for(int cc:g.kids[c]) st.push_back(cc); }
            }
        }
        h.cnDirOff.push_back(h.cnDirs.size());
        hgenes.push_back(move(h));
    }
}
static vector<unsigned char> partOf;
static vector<double> Hs, Hps, Hcps, Hu, Hpu, Hcpu; // per node x3
static double scoreTripH(const BS&A1,int n1,const BS&A2,int n2){
    partOf.assign(n,2);
    for(int i=0;i<n;i++){ if(A1.w[i>>6]>>(i&63)&1) partOf[i]=0; else if(A2.w[i>>6]>>(i&63)&1) partOf[i]=1; }
    double tot=0;
    for(const HGene&h:hgenes){
        int m=h.m;
        Hs.resize(3*m); Hps.resize(3*m); Hcps.resize(3*m); Hu.resize(3*m); Hpu.resize(3*m); Hcpu.resize(3*m);
        double*s=Hs.data(),*ps=Hps.data(),*cps=Hcps.data(),*u=Hu.data(),*pu=Hpu.data(),*cpu=Hcpu.data();
        // bottom-up (cps index: 0=(0,1) 1=(0,2) 2=(1,2))
        for(int v=m-1;v>=1;v--){
            double*sv=s+3*v,*pv=ps+3*v,*cv=cps+3*v;
            if(h.leaf[v]>=0){ sv[0]=sv[1]=sv[2]=0; sv[partOf[h.leaf[v]]]=1; pv[0]=pv[1]=pv[2]=0; cv[0]=cv[1]=cv[2]=0; continue; }
            double a0=0,a1=0,a2=0,p0=0,p1=0,p2=0,c01=0,c02=0,c12=0;
            for(int k=h.kidOff[v];k<h.kidOff[v+1];k++){
                int c=h.kids[k]; double d=h.dec[c]; double x0=d*s[3*c],x1=d*s[3*c+1],x2=d*s[3*c+2];
                p0+=ps[3*c]+a0*x0; p1+=ps[3*c+1]+a1*x1; p2+=ps[3*c+2]+a2*x2;
                c01+=cps[3*c]+a0*x1+a1*x0; c02+=cps[3*c+1]+a0*x2+a2*x0; c12+=cps[3*c+2]+a1*x2+a2*x1;
                a0+=x0; a1+=x1; a2+=x2;
            }
            sv[0]=a0; sv[1]=a1; sv[2]=a2; pv[0]=p0; pv[1]=p1; pv[2]=p2; cv[0]=c01; cv[1]=c02; cv[2]=c12;
        }
        // top-down: up quantities decayed to node v (include v's parent edge)
        { int v=1; double d=h.dec[1]; u[3]=0; u[4]=0; u[5]=d; pu[3]=pu[4]=pu[5]=0; cpu[3]=cpu[4]=cpu[5]=0; (void)v; }
        for(int p=1;p<m;p++){
            if(h.leaf[p]>=0) continue;
            // sources at p: up(p) [already decayed to p] and children
            double T0=u[3*p],T1=u[3*p+1],T2=u[3*p+2];
            double P0=pu[3*p],P1=pu[3*p+1],P2=pu[3*p+2];
            double C01=cpu[3*p],C02=cpu[3*p+1],C12=cpu[3*p+2];
            double X0=0,X1=0,X2=0,Y01=0,Y02=0,Y12=0;
            for(int k=h.kidOff[p];k<h.kidOff[p+1];k++){
                int c=h.kids[k]; double d=h.dec[c]; double x0=d*s[3*c],x1=d*s[3*c+1],x2=d*s[3*c+2];
                X0+=T0*x0; X1+=T1*x1; X2+=T2*x2; Y01+=T0*x1+T1*x0; Y02+=T0*x2+T2*x0; Y12+=T1*x2+T2*x1;
                T0+=x0; T1+=x1; T2+=x2; P0+=ps[3*c]; P1+=ps[3*c+1]; P2+=ps[3*c+2]; C01+=cps[3*c]; C02+=cps[3*c+1]; C12+=cps[3*c+2];
            }
            for(int k=h.kidOff[p];k<h.kidOff[p+1];k++){
                int c=h.kids[k]; if(h.leaf[c]>=0) continue;
                double d=h.dec[c]; double x0=d*s[3*c],x1=d*s[3*c+1],x2=d*s[3*c+2];
                double r0=T0-x0,r1=T1-x1,r2=T2-x2;
                u[3*c]=d*r0; u[3*c+1]=d*r1; u[3*c+2]=d*r2;
                pu[3*c]=P0-ps[3*c]+X0-x0*r0; pu[3*c+1]=P1-ps[3*c+1]+X1-x1*r1; pu[3*c+2]=P2-ps[3*c+2]+X2-x2*r2;
                cpu[3*c]=C01-cps[3*c]+Y01-x0*r1-x1*r0; cpu[3*c+1]=C02-cps[3*c+1]+Y02-x0*r2-x2*r0; cpu[3*c+2]=C12-cps[3*c+2]+Y12-x1*r2-x2*r1;
            }
        }
        // contracted nodes
        int ncn=h.cnTop.size();
        double sg0[512],sg1[512],sg2[512],pi0[512],pi1[512],pi2[512],ch01[512],ch02[512],ch12[512];
        for(int q=0;q<ncn;q++){
            int top=h.cnTop[q]; int d=0;
            for(int k=h.cnDirOff[q];k<h.cnDirOff[q+1] && d<511;k++){
                int c=h.cnDirs[k]; double dd=h.dec[c];
                sg0[d]=dd*s[3*c]; sg1[d]=dd*s[3*c+1]; sg2[d]=dd*s[3*c+2];
                pi0[d]=ps[3*c]; pi1[d]=ps[3*c+1]; pi2[d]=ps[3*c+2]; ch01[d]=cps[3*c]; ch02[d]=cps[3*c+1]; ch12[d]=cps[3*c+2]; d++;
            }
            sg0[d]=u[3*top]; sg1[d]=u[3*top+1]; sg2[d]=u[3*top+2]; pi0[d]=pu[3*top]; pi1[d]=pu[3*top+1]; pi2[d]=pu[3*top+2];
            ch01[d]=cpu[3*top]; ch02[d]=cpu[3*top+1]; ch12[d]=cpu[3*top+2]; d++;
            double S0=0,S1=0,S2=0,S01=0,S02=0,S12=0,Q0=0,Q1=0,Q2=0;
            for(int j=0;j<d;j++){ S0+=sg0[j]; S1+=sg1[j]; S2+=sg2[j]; S01+=sg0[j]*sg1[j]; S02+=sg0[j]*sg2[j]; S12+=sg1[j]*sg2[j]; Q0+=sg0[j]*sg0[j]; Q1+=sg1[j]*sg1[j]; Q2+=sg2[j]*sg2[j]; }
            double r=0;
            for(int j=0;j<d;j++){
                double a=sg0[j],b=sg1[j],c=sg2[j];
                if(pi0[j]!=0) r+=pi0[j]*((S1-b)*(S2-c)-(S12-b*c));
                if(pi1[j]!=0) r+=pi1[j]*((S0-a)*(S2-c)-(S02-a*c));
                if(pi2[j]!=0) r+=pi2[j]*((S0-a)*(S1-b)-(S01-a*b));
                if(ch12[j]!=0) r+=ch12[j]*(((S0-a)*(S0-a))-(Q0-a*a))*0.5;
                if(ch02[j]!=0) r+=ch02[j]*(((S1-b)*(S1-b))-(Q1-b*b))*0.5;
                if(ch01[j]!=0) r+=ch01[j]*(((S2-c)*(S2-c))-(Q2-c*c))*0.5;
            }
            tot+=h.cnW[q]*r;
        }
    }
    return tot;
}

// ---------------- cluster search space + DP ----------------
static vector<BS> C; static vector<int> Csz; static unordered_map<BS,int,BSHash> Cid;
static vector<vector<int>> bucket; // by lowbit
static vector<int> avail; static int curStamp=-1; static long availChanges=0; // avail[id]==-1: always usable
static int addCluster(const BS&b){
    auto it=Cid.find(b);
    if(it!=Cid.end()){ int id=it->second; if(curStamp>=0 && avail[id]!=-1 && avail[id]!=curStamp){ avail[id]=curStamp; availChanges++; } return id; }
    int id=C.size(); C.push_back(b); Csz.push_back(pc(b)); Cid[b]=id; bucket[lowbit(b)].push_back(id);
    avail.push_back(curStamp>=0? curStamp : -1); availChanges++;
    return id;
}
static inline bool usable(int id){ return curStamp<0 || avail[id]==-1 || avail[id]==curStamp; }
static unordered_map<u64,double> scache;
static long nScored=0;
static bool PERGENE=false; static unordered_map<u64,int> pgIdx; static vector<double> pgStore;
static vector<double> curMult; static double FREQW=0, QSCALE=1.0; static vector<double> cfreqById;
static vector<double> best; static vector<int> bestL, bestR;

struct Split { int b, d, pg; };
static vector<vector<Split>> splitsOf; static size_t integrated=0; static vector<int> bySize;
static void integrateNewClusters(){
    size_t m=C.size(); if(integrated==m) return;
    splitsOf.resize(m);
    for(size_t x=integrated; x<m; x++){
        const BS&A=C[x]; int sz=Csz[x];
        if(sz>=2){
            int lb=lowbit(A);
            for(int bid: bucket[lb]){
                if((size_t)bid>=x || Csz[bid]>=sz) continue;
                if(!subset(C[bid],A)) continue;
                auto it=Cid.find(bandn(A,C[bid])); if(it==Cid.end()) continue;
                if((size_t)it->second>=x) continue;
                splitsOf[x].push_back({bid,it->second,-1});
            }
        }
        // x as a part of an older, larger cluster
        for(size_t a=0;a<x;a++){
            if(Csz[a]<=sz) continue;
            if(!subset(A,C[a])) continue;
            auto it=Cid.find(bandn(C[a],A)); if(it==Cid.end()) continue;
            int d=it->second; if((size_t)d>=x) continue;
            int lb=lowbit(C[a]);
            if(A.w[lb>>6]>>(lb&63)&1) splitsOf[a].push_back({(int)x,d,-1}); else splitsOf[a].push_back({d,(int)x,-1});
        }
    }
    integrated=m;
    bySize.resize(m); iota(bySize.begin(),bySize.end(),0);
    stable_sort(bySize.begin(),bySize.end(),[&](int a,int b){return Csz[a]<Csz[b];});
}
static double runDP2(){
    integrateNewClusters();
    int m=C.size();
    best.assign(m,-1e300); bestL.assign(m,-1); bestR.assign(m,-1);
    for(int id:bySize){
        if(Csz[id]==1){ best[id]=0; continue; }
        if(!usable(id)) continue;
        double bv=-1e300; int bl=-1, br=-1;
        for(Split&s: splitsOf[id]){
            if(!usable(s.b) || !usable(s.d)) continue;
            if(best[s.b]<-1e299 || best[s.d]<-1e299) continue;
            if(s.pg<0){ s.pg=pgStore.size()/ngenes; pgStore.resize(pgStore.size()+ngenes); scoreTripQG(C[s.b],Csz[s.b],C[s.d],Csz[s.d],&pgStore[(size_t)s.pg*ngenes]); nScored++; }
            const double*pg=&pgStore[(size_t)s.pg*ngenes]; double q=0; for(int g=0;g<ngenes;g++) q+=curMult[g]*pg[g];
            double sc=QSCALE*q;
            if(FREQW>0){ double f=0; if((size_t)s.b<cfreqById.size()) f+=cfreqById[s.b]; if((size_t)s.d<cfreqById.size()) f+=cfreqById[s.d]; sc+=FREQW*f; }
            double v=best[s.b]+best[s.d]+sc;
            if(v>bv){ bv=v; bl=s.b; br=s.d; }
        }
        best[id]=bv; bestL[id]=bl; bestR[id]=br;
    }
    return best[Cid[UALL]];
}
static long countSplits(){
    long cnt=0;
    for(size_t id=0; id<C.size(); id++){
        if(Csz[id]<2) continue; const BS&A=C[id]; int lb=lowbit(A); int sz=Csz[id];
        for(int bid: bucket[lb]){ if(Csz[bid]>=sz) continue; if(!subset(C[bid],A)) continue; if(Cid.count(bandn(A,C[bid]))) cnt++; }
    }
    return cnt;
}
static double runDP(){
    if(PERGENE) return runDP2();
    int m=C.size();
    vector<int> ord(m); iota(ord.begin(),ord.end(),0);
    sort(ord.begin(),ord.end(),[&](int a,int b){return Csz[a]<Csz[b];});
    best.assign(m,-1e300); bestL.assign(m,-1); bestR.assign(m,-1);
    for(int id:ord){
        if(Csz[id]==1){ best[id]=0; continue; }
        if(!usable(id)) continue;
        const BS&A=C[id]; int lb=lowbit(A); int sz=Csz[id];
        double bv=-1e300; int bl=-1, br=-1;
        for(int bid: bucket[lb]){
            if(Csz[bid]>=sz) continue;
            if(!usable(bid)) continue;
            if(!subset(C[bid],A)) continue;
            BS D=bandn(A,C[bid]);
            auto it=Cid.find(D); if(it==Cid.end()) continue;
            int did=it->second;
            if(!usable(did)) continue;
            if(best[bid]<-1e299 || best[did]<-1e299) continue;
            u64 key=((u64)bid<<32)|(u64)did;
            double sc;
            if(PERGENE){
                auto pi=pgIdx.find(key); int ix;
                if(pi==pgIdx.end()){ ix=pgStore.size()/ngenes; pgIdx[key]=ix; pgStore.resize(pgStore.size()+ngenes); scoreTripQG(C[bid],Csz[bid],D,Csz[did],&pgStore[(size_t)ix*ngenes]); nScored++; }
                else ix=pi->second;
                const double*pg=&pgStore[(size_t)ix*ngenes]; double q=0; for(int g=0;g<ngenes;g++) q+=curMult[g]*pg[g];
                sc=QSCALE*q;
                if(FREQW>0){ double f=0; if((size_t)bid<cfreqById.size()) f+=cfreqById[bid]; if((size_t)did<cfreqById.size()) f+=cfreqById[did]; sc+=FREQW*f; }
                double v=best[bid]+best[did]+sc;
                if(v>bv){ bv=v; bl=bid; br=did; }
                continue;
            }
            auto ci=scache.find(key);
            if(ci==scache.end()){
                sc= SMODE==2? scoreTripH(C[bid],Csz[bid],D,Csz[did]) : SMODE? scoreTripQ(C[bid],Csz[bid],D,Csz[did]) : scoreTrip(C[bid],Csz[bid],D,Csz[did]); if(SMODE==2 && HMIX>0) sc+=HMIX*scoreTripQ(C[bid],Csz[bid],D,Csz[did]); if(SMODE && MU>0) sc+=MU*scoreTrip(C[bid],Csz[bid],D,Csz[did]); scache[key]=sc; nScored++; }
            else sc=ci->second;
            double v=best[bid]+best[did]+sc;
            if(v>bv){ bv=v; bl=bid; br=did; }
        }
        best[id]=bv; bestL[id]=bl; bestR[id]=br;
    }
    return best[Cid[UALL]];
}

// tree as list of clusters (rooted at R): node = cluster id with children
static void collectTree(int id, vector<int>&out){ out.push_back(id); if(Csz[id]>1){ collectTree(bestL[id],out); collectTree(bestR[id],out);} }

static void writeNewick(int id, string&s){
    if(Csz[id]==1){ s+=names[lowbit(C[id])]; return; }
    s+='('; writeNewick(bestL[id],s); s+=','; writeNewick(bestR[id],s); s+=')';
}


// score a binary tree given as Newick (diagnostics)
static double scoreTreeFile(const string&path){
    ifstream in(path); string s((istreambuf_iterator<char>(in)),istreambuf_iterator<char>());
    PTree t; if(!parseNewick(s,t)) return NAN;
    int m=t.nd.size();
    vector<vector<int>> adj(m);
    for(int v=0;v<m;v++){ if(t.nd[v].parent>=0){ adj[v].push_back(t.nd[v].parent); adj[t.nd[v].parent].push_back(v);} }
    // remove degree-2 root
    int root=t.root;
    if(t.nd[root].ch.size()==2){ int a=t.nd[root].ch[0], b=t.nd[root].ch[1]; adj[root].clear();
        for(auto&x:adj[a]) if(x==root) x=b; for(auto&x:adj[b]) if(x==root) x=a; }
    vector<int> leafTax(m,-1); int rnode=-1;
    for(int v=0;v<m;v++) if(t.nd[v].ch.empty()){ leafTax[v]=name2id[t.nd[v].label]; if(leafTax[v]==R) rnode=v; }
    vector<int> order, par(m,-1); vector<char> vis(m,0); vector<int> st={rnode}; vis[rnode]=1;
    while(!st.empty()){ int v=st.back(); st.pop_back(); order.push_back(v); for(int q:adj[v]) if(!vis[q]){ vis[q]=1; par[q]=v; st.push_back(q);} }
    vector<BS> cl(m,zero()); vector<vector<int>> kids(m);
    for(int i=order.size()-1;i>=0;i--){ int v=order[i]; if(leafTax[v]>=0 && v!=rnode) setb(cl[v],leafTax[v]); if(par[v]>=0){ cl[par[v]]=bor(cl[par[v]],cl[v]); kids[par[v]].push_back(v);} }
    double tot=0;
    auto SC=[&](const BS&a,const BS&b){ return (SMODE==2?scoreTripH:SMODE?scoreTripQ:scoreTrip)(a,pc(a),b,pc(b)) + ((SMODE&&MU>0)? MU*scoreTrip(a,pc(a),b,pc(b)):0.0); };
    for(int v=0;v<m;v++){ if(v==rnode||kids[v].size()!=2) continue; tot+=SC(cl[kids[v][0]],cl[kids[v][1]]); }
    if(getenv("NNITEST")){
        int good=0, tot2=0; double margin=0;
        for(int v=0;v<m;v++){
            if(v==rnode||kids[v].size()!=2||par[v]<0||par[v]==rnode) continue;
            int p=par[v]; if(kids[p].size()!=2) continue;
            int s = kids[p][0]==v? kids[p][1]: kids[p][0];
            const BS&A1=cl[kids[v][0]], &A2=cl[kids[v][1]], &S=cl[s];
            double t = SC(A1,A2)+SC(cl[v],S);
            double a1 = SC(A1,S)+SC(bor(A1,S),A2);
            double a2 = SC(A2,S)+SC(bor(A2,S),A1);
            tot2++; if(t>a1 && t>a2) good++;
            margin += (t-max(a1,a2))/max(1e-9,fabs(t));
        }
        fprintf(stderr,"NNITEST good=%d / %d\n",good,tot2);
    }
    return tot;
}

// add clusters of all SPR neighbours (radius <= r) of the current DP tree; returns #added
static size_t sprExpand(int r){
    vector<int> tr; collectTree(Cid[UALL],tr);
    int top=Cid[UALL];
    unordered_map<int,int> idx; for(size_t i=0;i<tr.size();i++) idx[tr[i]]=i;
    int m=tr.size(); vector<int> par(m,-1), L(m,-1), Rr(m,-1), dep(m,0);
    for(int i=0;i<m;i++){ int id=tr[i]; if(Csz[id]>1){ L[i]=idx[bestL[id]]; Rr[i]=idx[bestR[id]]; par[L[i]]=i; par[Rr[i]]=i; } }
    // depth (tr is preorder from collectTree)
    for(int i=1;i<m;i++) dep[i]=dep[par[i]]+1;
    size_t before=C.size();
    int rootI=idx[top];
    for(int c=0;c<m;c++){
        if(c==rootI) continue; int P=par[c]; if(P<0) continue;
        int S = L[P]==c? Rr[P]: L[P];
        const BS&Cc=C[tr[c]];
        // BFS over nodes reachable from P without entering subtree c, radius r (distance measured from P)
        vector<pair<int,int>> q; q.push_back({P,0}); vector<int> seen; seen.push_back(P);
        auto inSeen=[&](int x){ for(int y:seen) if(y==x) return true; return false; };
        for(size_t qi=0; qi<q.size(); qi++){
            int x=q[qi].first, d=q[qi].second;
            if(x!=P && x!=S){
                // target x: new clusters
                // ancestors chain
                int a=x, b=P;
                vector<int> upx, upP; // nodes strictly below LCA
                while(dep[a]>dep[b]){ upx.push_back(a); a=par[a]; }
                while(dep[b]>dep[a]){ upP.push_back(b); b=par[b]; }
                while(a!=b){ upx.push_back(a); upP.push_back(b); a=par[a]; b=par[b]; }
                // x's ancestors (incl x itself) below LCA gain C; new node x∪C
                for(int y:upx){ int id=tr[y]; addCluster(bor(C[id],Cc)); }
                for(int y:upP){ if(y==P) continue; int id=tr[y]; BS nb=bandn(C[id],Cc); if(!isempty(nb)) addCluster(nb); }
                if(upx.empty()){ /* x is ancestor of P (x==LCA) */ if(x!=rootI) addCluster(bor(C[tr[x]],Cc)); }
            }
            if(d>=r) continue;
            int nb[3]={par[x], L[x], Rr[x]};
            for(int y:nb){ if(y<0||y==c||inSeen(y)) continue; seen.push_back(y); q.push_back({y,d+1}); }
        }
    }
    return C.size()-before;
}

// ---------------- distance-based starting trees (NJ) ----------------
// per-gene leaf-to-leaf path sums with internal-edge weight f(p) and leaf-edge weight 1
static void geneDistances(const GTree&g, int mode, double th, vector<float>&D){
    int m=g.cl.size();
    vector<vector<pair<int,double>>> adj(m);
    for(int v=1;v<m;v++){
        int p=g.par[v]; double w;
        if(g.leaf[v]>=0) w=1.0;
        else { int e=g.edge[v]; double pv = e>=0? gedges[e].praw : 0.0;
               if(mode==0) w = pv>=th? 1.0: 0.0; else if(mode==1) w=pv; else w = std::isnan(g.plen[v])?0:g.plen[v]; }
        adj[p].push_back({v,w}); adj[v].push_back({p,w});
    }
    D.assign((size_t)n*n,0.f);
    vector<double> dist(m); vector<int> st; vector<int> from(m);
    for(int s=0;s<m;s++){
        if(g.leaf[s]<0) continue;
        int ts=g.leaf[s];
        st.clear(); st.push_back(s); dist[s]=0; from[s]=-1;
        while(!st.empty()){ int v=st.back(); st.pop_back(); if(g.leaf[v]>=0) D[(size_t)ts*n+g.leaf[v]]=dist[v];
            for(auto&q:adj[v]) if(q.first!=from[v]){ from[q.first]=v; dist[q.first]=dist[v]+q.second; st.push_back(q.first);} }
    }
}
// neighbour joining; returns clusters (oriented away from R) of the resulting tree
static void njClusters(const vector<double>&Din, vector<BS>&out){
    int N=n; vector<double> D(Din);
    vector<int> act(N); iota(act.begin(),act.end(),0);
    vector<BS> cl(N); for(int i=0;i<N;i++){ cl[i]=zero(); setb(cl[i],i); }
    // D is N x N, grows by reusing slots
    vector<double> r(N);
    while(act.size()>3){
        int k=act.size();
        for(int i:act){ double s=0; for(int j:act) s+=D[(size_t)i*N+j]; r[i]=s; }
        double bq=1e300; int bi=-1,bj=-1;
        for(int x=0;x<k;x++){ int i=act[x]; for(int y=x+1;y<k;y++){ int j=act[y]; double q=(k-2)*D[(size_t)i*N+j]-r[i]-r[j]; if(q<bq){bq=q;bi=i;bj=j;} } }
        // merge bj into bi
        double dij=D[(size_t)bi*N+bj];
        for(int x:act){ if(x==bi||x==bj) continue; double v=0.5*(D[(size_t)bi*N+x]+D[(size_t)bj*N+x]-dij); D[(size_t)bi*N+x]=v; D[(size_t)x*N+bi]=v; }
        cl[bi]=bor(cl[bi],cl[bj]);
        act.erase(find(act.begin(),act.end(),bj));
        out.push_back(cl[bi]);
    }
    // orient away from R
    for(auto&b:out){ if(b.w[R>>6]>>(R&63)&1){ BS c=bandn(UALL,b); BS rr=zero(); setb(rr,R); c=bandn(c,rr); b=c; } }
}

int main(int argc,char**argv){
    T0=chrono::steady_clock::now();
    if(argc<3){ fprintf(stderr,"usage: st in out\n"); return 1; }
    double TLIM=envd("TLIM",60.0);
    ifstream in(argv[1]); string line; vector<string> lines;
    while(getline(in,line)){ bool any=false; for(char c:line) if(!isspace((unsigned char)c)){any=true;break;} if(any) lines.push_back(line); }
    // newick may span lines; join and split by ';'
    string all; for(auto&l:lines){ all+=l; all+='\n'; }
    vector<string> trees; { string cur; for(char c:all){ cur+=c; if(c==';'){ trees.push_back(cur); cur.clear(); } } }
    vector<PTree> pts;
    for(auto&ts:trees){ PTree t; if(parseNewick(ts,t)) pts.push_back(move(t)); }
    // taxa from first tree
    {
        set<string> labs;
        for(auto&pt:pts) for(auto&nd:pt.nd) if(nd.ch.empty()) labs.insert(nd.label);
        vector<string> v(labs.begin(),labs.end());
        sort(v.begin(),v.end(),[](const string&a,const string&b){ bool da=!a.empty()&&all_of(a.begin(),a.end(),::isdigit), db=!b.empty()&&all_of(b.begin(),b.end(),::isdigit); if(da&&db&&a.size()!=b.size()) return a.size()<b.size(); return a<b; });
        names=v; n=v.size(); for(int i=0;i<n;i++) name2id[v[i]]=i;
        R = name2id.count("0")? name2id["0"] : 0;
    }
    fprintf(stderr,"n=%d trees=%zu\n",n,pts.size());
    UALL=zero(); for(int i=0;i<n;i++) if(i!=R) setb(UALL,i);
    // map so that R has index? keep; bucket by lowbit
    bucket.assign(n,{});
    int gi=0; for(auto&pt:pts){ if(processGene(pt,gi)) gi++; }
    ngenes=gi;
    fprintf(stderr,"genes=%d edges=%zu\n",ngenes,gedges.size());
    // ---- weights ----
    int wmode=(int)envd("WMODE",1); double wexp=envd("WEXP",1.0);
    // ---- sanity: support scale (0-100 -> 0-1) and non-negative lengths ----
    {
        double mx=0; for(auto&e:gedges) if(!std::isnan(e.sup)) mx=max(mx,e.sup);
        if(mx>1.5) for(auto&e:gedges) if(!std::isnan(e.sup)) e.sup/=100.0;
        for(auto&e:gedges) if(!std::isnan(e.len) && e.len<0) e.len=0;
        for(auto&g:gtrees) for(auto&l:g.plen) if(!std::isnan(l) && l<0) l=0;
    }
    // ---- support imputation for edges without a reported support ----
    {
        vector<pair<double,double>> lp; // (length, p)
        int nmiss=0;
        for(auto&e:gedges){
            if(std::isnan(e.sup)){ nmiss++; continue; }
            if(!std::isnan(e.len)) lp.push_back({e.len, min(1.0,max(0.0,(e.sup-1.0/3)/(2.0/3)))});
        }
        int nsup=0; double psum=0; for(auto&e:gedges) if(!std::isnan(e.sup)){ nsup++; psum+=min(1.0,max(0.0,(e.sup-1.0/3)/(2.0/3))); }
        double pmean = nsup? psum/nsup : 1.0;
        if(nmiss>0){
            sort(lp.begin(),lp.end());
            // monotone (non-decreasing) step map from binned means
            vector<double> bl, bp;
            if(lp.size()>=200){
                int B=min<int>(40, lp.size()/50);
                for(int b=0;b<B;b++){
                    size_t lo=lp.size()*b/B, hi=lp.size()*(b+1)/B; double s=0; for(size_t i=lo;i<hi;i++) s+=lp[i].second;
                    bl.push_back(lp[hi-1].first); bp.push_back(s/(hi-lo));
                }
                // pool adjacent violators
                vector<double> v=bp, w(bp.size(),1.0); vector<double> ov, ow; vector<int> ocnt;
                for(size_t i=0;i<v.size();i++){ ov.push_back(v[i]); ow.push_back(w[i]); ocnt.push_back(1);
                    while(ov.size()>1 && ov[ov.size()-2]>ov.back()){ double nw=ow[ow.size()-2]+ow.back(); double nv=(ov[ov.size()-2]*ow[ow.size()-2]+ov.back()*ow.back())/nw; int nc=ocnt[ocnt.size()-2]+ocnt.back(); ov.pop_back(); ow.pop_back(); ocnt.pop_back(); ov.back()=nv; ow.back()=nw; ocnt.back()=nc; } }
                bp.clear(); for(size_t i=0;i<ov.size();i++) for(int k=0;k<ocnt[i];k++) bp.push_back(ov[i]);
            }
            int imp=0;
            for(auto&e:gedges){
                if(!std::isnan(e.sup)) continue;
                double p;
                if(std::isnan(e.len)) p=pmean;
                else if(!bl.empty()){ size_t b=lower_bound(bl.begin(),bl.end(),e.len)-bl.begin(); if(b>=bp.size()) b=bp.size()-1; p=bp[b]; }
                else p=1.0-exp(-e.len/0.007); // no supports at all: length-based pseudo support
                e.sup = 1.0/3 + p*(2.0/3); imp++;
            }
            fprintf(stderr,"imputed support for %d edges (map bins=%zu)\n",imp,bl.size());
        }
    }
    double ilenMu=envd("ILEN",0);
    vector<double> geneMeanLen(ngenes,0.0);
    for(int g=0;g<ngenes;g++){ double tl=0; int ne=0; for(size_t v=1;v<gtrees[g].cl.size();v++){ double l=gtrees[g].plen[v]; if(!std::isnan(l)){ tl+=l; ne++; } } geneMeanLen[g]= ne? tl/ne : 0; }
    for(auto&e:gedges){
        double s=e.sup;
        double p;
        if(std::isnan(s)) p=0; else p=min(1.0,max(0.0,(s-1.0/3)/(2.0/3)));
        e.praw=p;
        if(ilenMu>0 && !std::isnan(e.len) && e.gene>=0 && e.gene<(int)geneMeanLen.size() && geneMeanLen[e.gene]>0) p*= 1.0-exp(-ilenMu*e.len/geneMeanLen[e.gene]);
        if(wmode==0) p=1.0;
        else if(wmode==1) p=pow(p,wexp);
        else if(wmode==2) p=min(envd("WCAP",5.0),-log(max(1e-12,1-p)))*(p>0);
        else if(wmode==3) p=(p>=envd("WTH",0.5))?1.0:0.0;
        e.p=p;
    }
    {
        unordered_map<BS,int,BSHash> uid;
        for(auto&e:gedges){ if(e.p<=0) continue; auto it=uid.find(e.cl); if(it==uid.end()){ uid[e.cl]=Ub.size(); Ub.push_back(e.cl); Usz.push_back(pc(e.cl)); Uw.push_back(e.p);} else Uw[it->second]+=e.p; }
    }
    fprintf(stderr,"unique weighted bipartitions=%zu\n",Ub.size());
    SMODE=(int)envd("SMODE",2); MU=envd("MU",0);
    if(SMODE){
        int K=(int)envd("QK",3); int qmode=(int)envd("QMODE",0);
        vector<double> qth, qw;
        if(getenv("QTH")){ // explicit thresholds on raw normalised support p and level weights
            string a=getenv("QTH"), b=getenv("QW"); stringstream sa(a), sb(b); string x;
            while(getline(sa,x,',')) qth.push_back(atof(x.c_str())); while(getline(sb,x,',')) qw.push_back(atof(x.c_str()));
            K=qth.size();
        }
        mt19937_64 rng(1234567);
        for(int gidx=0; gidx<(int)gtrees.size(); gidx++){ auto&g=gtrees[gidx];
            int m=g.cl.size(); vector<char> keep(m,0);
            for(int k=0;k<K;k++){
                double t=(k+0.5)/K;
                if(!qth.empty()){
                    for(int v=0;v<m;v++){ int e=g.edge[v]; keep[v]= e>=0 && gedges[e].praw>=qth[k]; }
                    addContracted(g,keep,qw[k],gidx); continue;
                }
                for(int v=0;v<m;v++){ int e=g.edge[v]; if(e<0){ keep[v]=0; continue; } double p=gedges[e].p;
                    if(qmode==0) keep[v]= p>=t; else keep[v]= (rng()>>11)*(1.0/9007199254740992.0) < p; }
                addContracted(g,keep,1.0/K,gidx);
            }
        }
        fprintf(stderr,"Q nodes=%zu pool=%zu kids=%zu t=%.2f\n",Qn.size(),Pb.size(),Qkids.size(),elapsed());
        if(SMODE==2){
            vector<double> th, w; for(int k=0;k<K;k++){ th.push_back((k+0.5)/K); w.push_back(1.0/K); }
            buildHGenes(envd("HLAM",0.2),(int)envd("HNORM",1),th,w);
            fprintf(stderr,"hybrid genes built t=%.2f\n",elapsed());
            double hm=envd("HMIXF",0.2); HTLIM=envd("HTLIM",60.0);
            { // calibrate magnitudes of hybrid vs plain quartet score on a fixed tripartition
                BS A1=zero(),A2=zero(); int k=0; for(int i=0;i<n;i++){ if(i==R) continue; if(k%3==0) setb(A1,i); else if(k%3==1) setb(A2,i); k++; }
                double h=scoreTripH(A1,pc(A1),A2,pc(A2)), q=scoreTripQ(A1,pc(A1),A2,pc(A2));
                HQRATIO = (q>0)? h/q : 0;
                HMIX = (q>0 && hm>0)? hm*h/q : 0; fprintf(stderr,"HMIX=%.3g (h=%.3g q=%.3g)\n",HMIX,h,q);
            }
        }
    }
    // ---- search space ----
    for(int i=0;i<n;i++) if(i!=R){ BS b=zero(); setb(b,i); addCluster(b); }
    addCluster(UALL);
    vector<vector<int>> geneClusters(ngenes);
    {
        double xminp=envd("XMINP",0.2);
        unordered_map<BS,double,BSHash> maxp;
        if(xminp>=0) for(auto&e:gedges){ auto&v=maxp[e.cl]; v=max(v,e.praw); }
        for(auto&e:gedges){ if(xminp>=0 && maxp[e.cl]<xminp) continue; int id=addCluster(e.cl); if(e.gene>=0 && e.gene<ngenes) geneClusters[e.gene].push_back(id); }
    }
    // complements w.r.t. UALL of gene clusters are not clusters (they contain R) -> skip
    fprintf(stderr,"clusters=%zu\n",C.size());
    if(getenv("ADDTREE")){ // diagnostic: add clusters of a given tree
        ifstream in2(getenv("ADDTREE")); string s((istreambuf_iterator<char>(in2)),istreambuf_iterator<char>());
        PTree t; parseNewick(s,t); size_t g0=gedges.size(); size_t gt0=gtrees.size(); processGene(t,-1);
        for(size_t i=g0;i<gedges.size();i++) addCluster(gedges[i].cl);
        gedges.resize(g0); gtrees.resize(gt0);
    }
    // ---- NJ starting trees from several gene-tree distance definitions (+ gene bootstrap) ----
    vector<vector<int>> njTrees;
    int NJN=(int)envd("NJN",1);
    if(NJN>0){
        vector<pair<int,double>> modes={{0,0.05},{1,0.0},{0,0.5},{2,0.0}};
        if(envd("NJMORE",0)>0){ modes.push_back({0,0.2}); modes.push_back({0,0.8}); modes.push_back({0,0.95}); }
        int nboot=(int)envd("NJBOOT",0);
        mt19937_64 rng2(12345);
        size_t before=C.size();
        vector<float> Dg;
        for(auto&md:modes){
            // per-gene matrices (normalised for lengths)
            vector<vector<float>> per(ngenes);
            for(int g=0;g<ngenes;g++){ geneDistances(gtrees[g],md.first,md.second,Dg);
                if(md.first==2){ double s=0; size_t c=0; for(size_t i=0;i<Dg.size();i++){ s+=Dg[i]; c++; } double mean=s/max<size_t>(1,c); if(mean>0) for(auto&x:Dg) x/=mean; }
                per[g]=Dg; }
            for(int b=0;b<=nboot;b++){
                vector<double> mult(ngenes,1.0);
                if(b>0){ fill(mult.begin(),mult.end(),0.0); for(int g=0;g<ngenes;g++) mult[uniform_int_distribution<int>(0,ngenes-1)(rng2)]+=1; }
                vector<double> D((size_t)n*n,0.0);
                for(int g=0;g<ngenes;g++) if(mult[g]>0){ const float*p=per[g].data(); for(size_t i=0;i<D.size();i++) D[i]+=mult[g]*p[i]; }
                vector<BS> cls; njClusters(D,cls);
                vector<int> ids; for(auto&c:cls){ int k=pc(c); if(k>=2 && k<=n-2) ids.push_back(addCluster(c)); }
                njTrees.push_back(ids);
            }
        }
        fprintf(stderr,"NJ clusters added=%zu total=%zu t=%.2f\n",C.size()-before,C.size(),elapsed());
    }
    PERGENE = (SMODE==1) && envd("PERGENE",1)>0;
    if(PERGENE && getenv("BENCH")) buildFlat();
    if(getenv("BENCH")){
        size_t nc=0; for(auto&v:QnContrib) nc+=v.size();
        fprintf(stderr,"Qn=%zu contribs=%zu pool=%zu\n",Qn.size(),nc,Pb.size());
        vector<double> out(ngenes);
        BS A1=zero(),A2=zero(); for(int i=0;i<n;i++){ if(i==R) continue; if(i%3==0) setb(A1,i); else if(i%3==1) setb(A2,i); }
        double t0=elapsed(); for(int k=0;k<1000;k++) scoreTripQG(A1,pc(A1),A2,pc(A2),out.data()); double t1=elapsed();
        for(int k=0;k<1000;k++) scoreTripQ(A1,pc(A1),A2,pc(A2)); double t2=elapsed();
        vector<double> out2(ngenes); for(int k=0;k<1000;k++) scoreTripQF(A1,pc(A1),A2,pc(A2),out2.data()); double t3=elapsed();
        double md=0; for(int g=0;g<ngenes;g++) md=max(md,fabs(out[g]-out2[g])/max(1.0,fabs(out[g])));
        fprintf(stderr,"QG %.1f us  Q %.1f us  QF %.1f us maxreldiff %.3g\n",(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,md);
        return 0;
    }
    curMult.assign(ngenes,1.0);
    auto nniSearch=[&](double sc, int maxIt)->double{
        for(int it=0; it<maxIt && elapsed()<TLIM; it++){
            vector<int> tr; collectTree(Cid[UALL],tr);
            long before=availChanges;
            unordered_map<int,int> parent;
            for(int id:tr) if(Csz[id]>1){ parent[bestL[id]]=id; parent[bestR[id]]=id; }
            for(int id:tr){
                if(Csz[id]<2) continue;
                auto pi=parent.find(id); if(pi==parent.end()) continue;
                int p=pi->second; int sib = bestL[p]==id? bestR[p]: bestL[p];
                addCluster(bor(C[bestL[id]],C[sib]));
                addCluster(bor(C[bestR[id]],C[sib]));
            }
            if(availChanges==before) break;
            double ns=runDP();
            if(ns<=sc*(1+1e-12)) { sc=max(sc,ns); break; }
            sc=ns;
        }
        return sc;
    };
    if(SMODE==2){ // time guard: fall back to the fast plain quartet score if hybrid scoring would be too slow
        long ns=countSplits();
        BS A1=zero(),A2=zero(); int k=0; for(int i=0;i<n;i++){ if(i==R) continue; if(k%3==0) setb(A1,i); else if(k%3==1) setb(A2,i); k++; }
        double t0=elapsed(); for(int r=0;r<5;r++) scoreTripH(A1,pc(A1),A2,pc(A2)); double per=(elapsed()-t0)/5;
        double est=ns*per*1.4;
        fprintf(stderr,"splits=%ld est hybrid DP time=%.1fs\n",ns,est);
        if(est>HTLIM){ SMODE=1; HMIX=0; fprintf(stderr,"switching to plain quartet scoring\n"); }
    }
    double sc=runDP();
    if(sc<-1e299){ // safety: make sure a complete tree exists in the search space
        for(auto&e:gedges) addCluster(e.cl);
        sc=runDP();
    }
    fprintf(stderr,"DP0 score=%.6g scored=%ld t=%.2f\n",sc,nScored,elapsed());
    sc=nniSearch(sc,(int)envd("NNIIT",50));
    fprintf(stderr,"full-data search score=%.8g clusters=%zu scored=%ld t=%.2f\n",sc,C.size(),nScored,elapsed());
    int sprR=(int)envd("SPRR",0);
    if(sprR>0){ sprExpand(sprR); double ns=runDP(); sc=nniSearch(max(sc,ns),50); fprintf(stderr,"spr score=%.8g clusters=%zu t=%.2f\n",sc,C.size(),elapsed()); }
    // ---- multi-start: trajectories from gene clusters + a single NJ tree, optionally reweighted ----
    int NMS=(int)envd("NMS",0); double MSLIM=envd("MSLIM",60.0); int msBoot=(int)envd("MSBOOT",0);
    if(PERGENE && NMS>0 && !njTrees.empty()){
        vector<vector<int>> geneCl(ngenes);
        for(auto&e:gedges){ auto it=Cid.find(e.cl); if(it!=Cid.end() && e.gene>=0) geneCl[e.gene].push_back(it->second); }
        mt19937_64 rng3((u64)envd("SEED",1)*104729+3);
        for(size_t id=0; id<avail.size(); id++) avail[id] = (Csz[id]==1 || (int)id==Cid[UALL]) ? -1 : -2;
        int done=0;
        for(int t=0;t<NMS && elapsed()<MSLIM;t++){
            curStamp=t;
            vector<double> mult(ngenes,1.0);
            if(msBoot){ fill(mult.begin(),mult.end(),0.0); for(int g=0;g<ngenes;g++) mult[uniform_int_distribution<int>(0,ngenes-1)(rng3)]+=1; }
            curMult=mult;
            for(int g=0;g<ngenes;g++) if(mult[g]>0) for(int id:geneCl[g]) if(avail[id]!=-1) avail[id]=t;
            const vector<int>&nj=njTrees[uniform_int_distribution<int>(0,njTrees.size()-1)(rng3)];
            for(int id:nj) if(avail[id]!=-1) avail[id]=t;
            double s0=runDP(); nniSearch(s0,(int)envd("MSNNI",50));
            done++;
        }
        curStamp=-1; curMult.assign(ngenes,1.0);
        double ns=runDP(); sc=nniSearch(ns,50);
        fprintf(stderr,"multistart runs=%d clusters=%zu final score=%.8g scored=%ld t=%.2f\n",done,C.size(),sc,nScored,elapsed());
        if(sprR>0){ sprExpand(sprR); double s2=runDP(); sc=nniSearch(max(sc,s2),50); fprintf(stderr,"post-ms spr score=%.8g t=%.2f\n",sc,elapsed()); }
    }
    // ---- iterative gene reweighting by agreement with the current tree ----
    int RW=(int)envd("REWEIGHT",0);
    if(PERGENE && RW>0){
        // per-gene max: score of each gene's own tree
        vector<double> own(ngenes,0.0), cur(ngenes,0.0), tmp(ngenes);
        for(int g=0; g<ngenes; g++){
            const GTree&gt=gtrees[g];
            for(size_t v=1; v<gt.cl.size(); v++){ if(gt.leaf[v]>=0 || gt.kids[v].size()!=2) continue;
                const BS&a=gt.cl[gt.kids[v][0]], &b=gt.cl[gt.kids[v][1]];
                scoreTripQG(a,pc(a),b,pc(b),tmp.data()); own[g]+=tmp[g]; }
        }
        double gam=envd("RWGAM",1.0);
        for(int it=0; it<RW; it++){
            vector<int> tr; collectTree(Cid[UALL],tr);
            fill(cur.begin(),cur.end(),0.0);
            for(int id:tr){ if(Csz[id]<2) continue; scoreTripQG(C[bestL[id]],Csz[bestL[id]],C[bestR[id]],Csz[bestR[id]],tmp.data()); for(int g=0;g<ngenes;g++) cur[g]+=tmp[g]; }
            vector<double> m(ngenes); double s=0;
            for(int g=0;g<ngenes;g++){ double a= own[g]>0? cur[g]/own[g] : 0; m[g]=pow(max(0.0,a),gam); s+=m[g]; }
            for(int g=0;g<ngenes;g++) m[g]*= s>0? ngenes/s : 1.0;
            curMult=m;
            double s0=runDP(); s0=nniSearch(s0,50);
            fprintf(stderr,"reweight it %d score=%.6g t=%.2f\n",it,s0,elapsed());
        }
        if(envd("RWKEEP",1)==0){ curMult.assign(ngenes,1.0); double s2=runDP(); nniSearch(s2,50); }
    }
    // full-data tree clusters
    vector<int> fullTree; collectTree(Cid[UALL],fullTree);
    double fullScore=sc;
    // ---- bootstrap over genes: split frequencies ----
    int NB=(int)envd("NBOOT",0); double BTLIM=envd("BTLIM",80.0);
    if(PERGENE && NB>0){
        mt19937_64 rng((u64)envd("SEED",1)*7919+17);
        vector<double> freqCount; int done=0;
        int bootNNI=(int)envd("BOOTNNI",50); int bayes=(int)envd("BAYESBOOT",0); int restrictX=(int)envd("RESTRICTX",1);
        // make every non-singleton cluster replicate-specific (singletons and the root stay always usable)
        for(size_t id=0; id<avail.size(); id++) avail[id] = (Csz[id]==1 || (int)id==Cid[UALL]) ? -1 : -2;
        for(int b=0;b<NB && elapsed()<BTLIM;b++){
            vector<double> mult(ngenes,0.0);
            if(bayes){ exponential_distribution<double> ex(1.0); double s=0; for(int g=0;g<ngenes;g++){ mult[g]=ex(rng); s+=mult[g]; } for(int g=0;g<ngenes;g++) mult[g]*=ngenes/s; }
            else for(int g=0;g<ngenes;g++) mult[uniform_int_distribution<int>(0,ngenes-1)(rng)]+=1.0;
            curMult=mult;
            curStamp=b;
            if(restrictX){ for(int g=0;g<ngenes;g++) if(mult[g]>0) for(int id:geneClusters[g]) if(avail[id]!=-1) avail[id]=b; }
            else { for(size_t id=0; id<avail.size(); id++) if(avail[id]!=-1) avail[id]=b; }
            double s0=runDP(); nniSearch(s0,bootNNI);
            vector<int> tr; collectTree(Cid[UALL],tr);
            if(freqCount.size()<C.size()) freqCount.resize(C.size(),0.0);
            for(int id:tr) if(Csz[id]>1 && id!=Cid[UALL]) freqCount[id]+=1.0;
            done++;
        }
        fprintf(stderr,"bootstrap replicates=%d clusters=%zu scored=%ld t=%.2f\n",done,C.size(),nScored,elapsed());
        // final: maximise sum of split frequencies (+ small quartet-score tie-break) over all clusters
        curStamp=-1;
        curMult.assign(ngenes,1.0);
        cfreqById.assign(C.size(),0.0); for(size_t i=0;i<freqCount.size();i++) cfreqById[i]=freqCount[i]/max(1,done);
        double fq=envd("FQW",0.02); // weight of normalised quartet score relative to split frequency
        int finalMode=(int)envd("FINALMODE",0);
        if(finalMode==1){ FREQW=0; QSCALE=1.0; }
        else if(finalMode==2){ FREQW=envd("FRW",0.002)*fabs(fullScore)/247.0; QSCALE=1.0; } // score + small frequency bonus
        else { FREQW=1.0; QSCALE=fq/max(1e-300,fabs(fullScore)); }
        double fs=runDP();
        if(finalMode>=1){ fs=nniSearch(fs,50); }
        vector<int> tr; collectTree(Cid[UALL],tr);
        double sf=0; int common=0; unordered_set<int> fset(fullTree.begin(),fullTree.end());
        for(int id:tr){ if(Csz[id]>1 && id!=Cid[UALL]){ sf+=cfreqById[id]; if(fset.count(id)) common++; } }
        fprintf(stderr,"consensus DP obj=%.6g sumfreq=%.3f shared-with-full=%d t=%.2f\n",fs,sf,common,elapsed());
    }
    string out="("+names[R]+",";
    int top=Cid[UALL];
    string s1; writeNewick(bestL[top],s1); string s2; writeNewick(bestR[top],s2);
    out+=s1+","+s2+");\n";
    FILE*f=fopen(argv[2],"w"); fputs(out.c_str(),f); fclose(f);
    fprintf(stderr,"done t=%.2f\n",elapsed());
    if(getenv("TRUTHX")){
        ifstream in2(getenv("TRUTHX")); string s((istreambuf_iterator<char>(in2)),istreambuf_iterator<char>());
        PTree t; parseNewick(s,t); size_t g0=gedges.size(); size_t gt0=gtrees.size(); processGene(t,-1);
        int inX=0, tot=0; vector<int> fin; collectTree(Cid[UALL],fin); unordered_set<int> fs(fin.begin(),fin.end());
        int missedInX=0, missedNotX=0;
        for(size_t i=g0;i<gedges.size();i++){ tot++; auto it=Cid.find(gedges[i].cl); bool x=it!=Cid.end(); inX+=x; bool f= x && fs.count(it->second); if(!f){ if(x) missedInX++; else missedNotX++; } }
        fprintf(stderr,"TRUTHX: truth clusters in X: %d/%d ; missed in X: %d ; missed not in X: %d\n",inX,tot,missedInX,missedNotX);
        gedges.resize(g0); gtrees.resize(gt0);
    }
    if(getenv("SCORETREE")){ fprintf(stderr,"found=%.8g truth=%.8g\n",scoreTreeFile(argv[2]),scoreTreeFile(getenv("SCORETREE"))); }
    return 0;
}
