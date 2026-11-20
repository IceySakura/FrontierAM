#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
constexpr int N = 250, V = 2 * N;

using Clock=chrono::steady_clock;
static Clock::time_point deadline=Clock::time_point::max();
static bool timeLeft() { return Clock::now()<deadline; }

struct Tree {
    array<array<int, 3>, V> adj{};
    array<int, V> deg{};
    int next = N;
    void link(int a, int b) {
        adj[a][deg[a]++] = b;
        adj[b][deg[b]++] = a;
    }
    void del(int a, int b) {
        int i = 0;
        while (adj[a][i] != b) ++i;
        adj[a][i] = adj[a][--deg[a]];
    }
    void unlink(int a, int b) { del(a,b); del(b,a); }
    void swapBranches(int u, int v, int a, int b) {
        unlink(u,a); unlink(v,b); link(u,b); link(v,a);
    }
    array<int,2> children(int u, int p) const {
        array<int,2> ans{};
        int i=0;
        for (int j=0; j<deg[u]; ++j) if(adj[u][j]!=p) ans[i++]=adj[u][j];
        return ans;
    }
    void mark(int u, int p, int c, int *out) const {
        if (u < N) out[u] = c;
        else for (int i=0; i<3; ++i) if(adj[u][i]!=p) mark(adj[u][i],u,c,out);
    }
    void clade(int u, int p, vector<int> &out) const {
        if (u < N) out.push_back(u);
        else for (int i=0; i<3; ++i) if(adj[u][i]!=p) clade(adj[u][i],u,out);
    }
    string emit(int u, int p) const {
        if (u < N) return to_string(u);
        string s = "(";
        for (int i=0; i<3; ++i) if(adj[u][i]!=p) {
            if (s.size()>1) s += ',';
            s += emit(adj[u][i],u);
        }
        return s + ')';
    }
    string newick() const { return emit(adj[0][0],-1) + ";\n"; }
    bool valid() const {
        if(next!=V-2) return false;
        array<bool,V> seen{};
        int visited=0;
        auto check=[&](auto &&self,int u,int from)->bool {
            if(u<0 || u>=next || seen[u] || deg[u]!=(u<N?1:3)) return false;
            seen[u]=true;
            ++visited;
            bool connected=(from<0);
            for(int i=0;i<deg[u];++i) {
                int v=adj[u][i];
                if(v==from) connected=true;
                else if(!self(self,v,u)) return false;
            }
            return connected;
        };
        return check(check,0,-1) && visited==next;
    }
};

struct Mask {
    array<uint64_t,4> w{};
    explicit Mask(int x=-1) { if(x>=0) w[x/64]=uint64_t(1)<<(x%64); }
    Mask &operator|=(const Mask &b) { for(int i=0;i<4;++i) w[i]|=b.w[i]; return *this; }
    Mask operator|(const Mask &b) const { Mask a=*this; return a|=b; }
    Mask operator&(const Mask &b) const {
        Mask a;
        for(int i=0;i<4;++i) a.w[i]=w[i]&b.w[i];
        return a;
    }
    Mask operator^(const Mask &b) const {
        Mask a;
        for(int i=0;i<4;++i) a.w[i]=w[i]^b.w[i];
        return a;
    }
    bool operator==(const Mask &b) const { return w==b.w; }
    bool operator!=(const Mask &b) const { return w!=b.w; }
    bool operator<(const Mask &b) const { return w<b.w; }
    bool has(int x) const { return (w[x/64]>>(x%64))&1; }
    bool empty() const { return (w[0]|w[1]|w[2]|w[3])==0; }
    int count() const { int n=0; for(auto x:w) n+=__builtin_popcountll(x); return n; }
    int first() const {
        for(int i=0;i<4;++i) if(w[i]) return i*64+__builtin_ctzll(w[i]);
        return -1;
    }
    template<class F> void each(F f) const {
        for(int i=0;i<4;++i) for(uint64_t x=w[i];x;x&=x-1) f(i*64+__builtin_ctzll(x));
    }
    bool inside(const Mask &b) const { return ((*this)&b)==*this; }
};
struct MaskHash {
    size_t operator()(const Mask &a) const {
        uint64_t h=0x9e3779b97f4a7c15ULL;
        for(uint64_t x:a.w) {
            h^=x + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h^=h>>30; h*=0xbf58476d1ce4e5b9ULL;
        }
        return h^(h>>27);
    }
};
static vector<Mask> treeSplits(const Tree &t) {
    vector<Mask> ans;
    auto visit=[&](auto &&self,int u,int p)->Mask {
        if(u<N) return Mask(u);
        auto [a,b]=t.children(u,p);
        Mask s=self(self,a,u)|self(self,b,u);
        if(p) ans.push_back(s);
        return s;
    };
    visit(visit,t.adj[0][0],0);
    sort(ans.begin(),ans.end());
    return ans;
}

struct Edge { float len=0, sup=-1; };
struct Gene {
    struct Node { int l=-1,r=-1; Edge le,re; };
    // The binary postorder is rooted by subdividing the outgroup's edge.
    array<Node,V-1> node;
    vector<float> dist;
    float rate=0, resolution=0, q=1;
};

struct NewickReader {
    string s;
    size_t pos=0;
    array<char,N> seen{};
    Tree t;
    array<array<Edge,3>,V> edge;
    explicit NewickReader(string text):s(std::move(text)) {}
    void skip() {
        if(pos==0 && s.compare(0,3,"\xef\xbb\xbf")==0) pos=3;
        while(pos<s.size()) {
            if (static_cast<unsigned char>(s[pos])<=32) ++pos;
            else if(s[pos]=='[') {
                do { ++pos; } while(pos<s.size() && s[pos]!=']');
                if(pos<s.size()) ++pos;
            } else break;
        }
    }
    char peek() { skip(); return pos<s.size()?s[pos]:'\0'; }
    string word() {
        char c=peek();
        if(c=='\'') {
            ++pos;
            string w;
            while(pos<s.size()) {
                c=s[pos++];
                if(c=='\'') {
                    if(pos==s.size() || s[pos]!='\'') break;
                    ++pos;
                }
                w+=c;
            }
            return w;
        }
        size_t st=pos;
        while(pos<s.size()) {
            c=s[pos];
            if (static_cast<unsigned char>(c)<=32 || c=='(' || c==')' || c==':' ||
                c==',' || c==';' || c=='[') break;
            ++pos;
        }
        return s.substr(st,pos-st);
    }
    void add(int a, int b, Edge e) {
        if (t.deg[a]>=3 || t.deg[b]>=3) throw runtime_error("non-binary Newick");
        edge[a][t.deg[a]]=edge[b][t.deg[b]]=e; t.link(a,b);
    }
    Edge get(int a,int b) const {
        for(int i=0;i<t.deg[a];++i) if(t.adj[a][i]==b) return edge[a][i];
        throw runtime_error("missing edge");
    }
    void replace(int a,int b,int to,Edge e) {
        for(int i=0;i<t.deg[a];++i) if(t.adj[a][i]==b) {
            t.adj[a][i]=to; edge[a][i]=e; return;
        }
        throw runtime_error("missing edge");
    }
    pair<int,Edge> subtree() {
        int u;
        Edge e;
        if(peek()=='(') {
            ++pos;
            if (t.next>=V) throw runtime_error("too many nodes");
            u=t.next++;
            for (;;) {
                auto [v,w]=subtree();
                add(u,v,w);
                if(peek()!=',') break;
                ++pos;
            }
            if(peek()!=')') throw runtime_error("expected ')'");
            ++pos;
            string w=word();
            if(!w.empty()) {
                char *end=nullptr;
                float sup=strtof(w.c_str(),&end);
                if(*end==0 && sup>=0 && sup<=1) e.sup=sup;
            }
        } else {
            string w=word();
            if(w.empty()) throw runtime_error("missing tip label");
            char *end=nullptr;
            long k=strtol(w.c_str(),&end,10);
            if(*end || k<0 || k>=N || seen[k]++ || w!=to_string(k))
                throw runtime_error("unexpected tip label: "+w);
            u=int(k);
        }
        if(peek()==':') {
            ++pos;
            string w=word();
            e.len=clamp(strtof(w.c_str(),nullptr),0.f,100.f);
        }
        return {u,e};
    }
    int order(Gene &g,int &idx,int u,int p) const {
        if(u<N) return u;
        auto [l,r]=t.children(u,p);
        int il=order(g,idx,l,u), ir=order(g,idx,r,u);
        g.node[idx]={il,ir,get(u,l),get(u,r)};
        return idx++;
    }
    Gene read() {
        auto [r,ignored]=subtree();
        if(peek()!=';' || count(seen.begin(),seen.end(),1)!=N)
            throw runtime_error("incomplete Newick tree");
        ++pos;
        if (peek()) throw runtime_error("extra Newick data");
        if (t.deg[r]==2) {
            auto [a,b]=t.children(r,-1);
            Edge ea=get(r,a), eb=get(r,b);
            Edge joint={ea.len+eb.len,max(ea.sup,eb.sup)};
            replace(a,r,b,joint); replace(b,r,a,joint); t.deg[r]=0;
        } else if (t.deg[r]!=3) throw runtime_error("invalid Newick root");
        for (int i=N;i<t.next;++i) if(t.deg[i]!=3 && i!=r)
            throw runtime_error("invalid internal degree");
        Gene g;
        int idx=N;
        int neighbor=t.adj[0][0], inner=order(g,idx,neighbor,0);
        Edge x=get(0,neighbor);
        x.len*=0.5f;
        g.node[idx++]={0,inner,x,x};
        if(idx!=V-1) throw runtime_error("invalid number of nodes");
        return g;
    }
};

static float support(Edge e) {
    if(e.sup<0) return 1-expf(-e.len*75);
    return clamp(1.5f*(e.sup-1.f/3),0.f,1.f);
}
static float median(vector<float> a) {
    if(a.empty()) return 0;
    auto m=a.begin()+a.size()/2;
    nth_element(a.begin(),m,a.end());
    return *m;
}

using Matrix=vector<double>;
static constexpr double INF=1e99;

struct Distances {
    vector<Matrix> d;
    vector<array<vector<float>,5>> boot;
    struct Observation { float distance; uint16_t locus; };
    vector<Observation> sorted;
    explicit Distances(vector<Gene> &gs) : d(38) {
        constexpr int kinds[]={0,2,19,20,24,25,29,32,34,35,37};
        for(int i:kinds) d[i].resize(N*N);
        for(Gene &g:gs) {
            array<array<int,3>,V> adj{};
            array<array<Edge,3>,V> es{};
            array<int,V> deg{};
            auto link=[&](int a,int b,Edge w) {
                adj[a][deg[a]]=b; es[a][deg[a]++]=w;
                adj[b][deg[b]]=a; es[b][deg[b]++]=w;
            };
            for(int v=N;v<V-1;++v) {
                const auto &p=g.node[v]; link(v,p.l,p.le); link(v,p.r,p.re);
            }
            struct Arc {
                int v; float l, a, b, c, d, e;
            };
            array<array<Arc,3>,V> arcs{};
            int resolved=0;
            for(int u=N; u<V-2; ++u) {
                const auto &p=g.node[u];
                if(p.l>=N && p.le.len>0.002) ++resolved;
                if(p.r>=N && p.re.len>0.002) ++resolved;
            }
            g.resolution=float(resolved)/(N-3);
            for(int u=0;u<V-1;++u) for(int j=0;j<deg[u];++j) {
                int v=adj[u][j];
                Edge e=es[u][j];
                bool inner=(u>=N && v>=N && u!=V-2 && v!=V-2);
                float p=support(e);
                float l=e.len;
                arcs[u][j]={v,l,float(inner),inner?p:0.f,
                    inner?sqrtf(p*clamp(l/(l+.003f),0.f,1.f)):0.f,
                    inner?p*p:0.f,
                    inner?1-expf(-200*l):0.f};
            }
            struct Cell { float l, a, b, c, d, e; };
            vector<Cell> raw(N*N);
            vector<float> lengths, topo;
            lengths.reserve(N*N/2);
            topo.reserve(N*N/2);
            g.dist.resize(N*N);
            for(int x=0;x<N;++x) {
                auto rec=[&](auto &&self,int u,int p,Cell z)->void {
                    if(u<N && u!=x) {
                        g.dist[x*N+u]=z.l;
                        if(u>x) {
                            raw[x*N+u]=z;
                            if(x!=0) { lengths.push_back(z.l); topo.push_back(z.b); }
                        }
                    }
                    for(int i=0;i<deg[u];++i) {
                        const Arc &a=arcs[u][i]; if(a.v==p) continue;
                        self(self,a.v,u,{z.l+a.l,z.a+a.a,z.b+a.b,z.c+a.c,z.d+a.d,z.e+a.e});
                    }
                };
                rec(rec,x,-1,{});
            }
            float rate=max(median(std::move(lengths)),.005f);
            float srate=max(median(std::move(topo)),.5f);
            g.rate=rate;
            float qw=powf(g.resolution,.65f);
            float good=qw/(1+powf(rate/.85f,2));
            g.q=good;
            boot.emplace_back();
            for(auto &a:boot.back()) a.resize(N*N);
            const double wt=powf(g.resolution,.44f)/(1+powf(rate/1.45f,2));
            const double sqrtResolution=sqrt(g.resolution);
            const double ps=max(.04,1.-exp(-4./3.*rate));
            for(int x=0;x<N;++x) for(int y=x+1;y<N;++y) {
                int i=x*N+y;
                Cell e=raw[i];
                float r=e.l/rate, sr=e.b/srate;
                double pd=-expm1(-4./3.*e.l);
                auto &b=boot.back();
                b[0][i]=.65f*e.c+.25f*sr*5+.10f*e.d;
                b[1][i]=expf(-sr*.6f-r*.95f)*wt;
                b[2][i]=expf(-sr/.42f)*(.45f+.55f*g.resolution);
                b[3][i]=min(2.5f,r)*wt;
                b[4][i]=.54f*sr+.46f*min(2.9f,r);
                d[0][i]+=e.a;
                d[2][i]+=e.c;
                d[19][i]+=pd/ps;
                d[20][i]+=sr;
                d[24][i]+=exp(-r/.30)*wt;
                d[25][i]+=exp(-r/.65)*wt;
                d[29][i]+=exp(-sr/.40)*sqrtResolution;
                d[34][i]+=.52*sr+.48*r;
                d[35][i]+=exp(-sr*.6)*exp(-r/.8)*wt;
                d[37][i]+=.55*e.b+.35*e.d+.1*e.e;
            }
        }
        vector<Observation> v;
        v.resize(gs.size());
        sorted.reserve(N*(N-1)/2*gs.size());
        for(int x=0;x<N;++x) for(int y=x+1;y<N;++y) {
            int i=x*N+y;
            for(size_t k=0;k<gs.size();++k)
                v[k]={gs[k].dist[i]/max(.025f,gs[k].rate),uint16_t(k)};
            sort(v.begin(),v.end(),[](const auto &a,const auto &b) { return a.distance<b.distance; });
            sorted.insert(sorted.end(),v.begin(),v.end());
            int n=int(v.size());
            double low=0;
            for(int k=0;k<n/2;++k) low+=v[k].distance;
            d[32][i]=low/max(1,n/2);
            for(int j:{24,25,29,35}) d[j][i]=-log(max(1e-40,d[j][i]/gs.size()));
        }
        for(int k:kinds) {
            auto &a=d[k];
            double mn=INF, mean=0;
            for(int x=1;x<N;++x) for(int y=x+1;y<N;++y) { mn=min(mn,a[x*N+y]); mean+=a[x*N+y]; }
            if(mn<0) for(int x=0;x<N;++x) for(int y=x+1;y<N;++y) a[x*N+y]-=mn;
            double scale=max(1e-10,mean/((N-1)*(N-2)/2)-min(0.,mn));
            for(int x=0;x<N;++x) for(int y=x+1;y<N;++y) a[y*N+x]=(a[x*N+y]/=scale);
        }
    }
    Matrix trimmed(const vector<float> &mult,double fraction) const {
        Matrix a(N*N);
        size_t i=0, m=mult.size();
        double total=accumulate(mult.begin(),mult.end(),0.)*fraction;
        double avg=0;
        for(int x=0;x<N;++x) for(int y=x+1;y<N;++y,i+=m) {
            double taken=0, sum=0;
            for(size_t j=i;j<i+m && taken<total;++j) {
                const auto [d,g]=sorted[j];
                double w=min<double>(total-taken,mult[g]);
                taken+=w; sum+=w*d;
            }
            a[x*N+y]=a[y*N+x]=sum/max(1e-8,total);
            if(x) avg+=a[x*N+y];
        }
        double scale=max(1e-8,avg/((N-1)*(N-2)/2));
        for(double &x:a) x/=scale;
        return a;
    }
    Matrix sample(const vector<float> &mult,int m) const {
        Matrix a(N*N);
        for(size_t i=0;i<boot.size();++i) {
            const auto &b=boot[i][m];
            double w=mult[i];
            for(int k=0;k<N*N;++k) a[k]+=w*b[k];
        }
        for(int i=0;i<N;++i) for(int j=i+1;j<N;++j) {
            double &z=a[i*N+j];
            if(m==1 || m==2) z=-log(max(1e-40,z/boot.size()));
            a[j*N+i]=z;
        }
        return a;
    }
};

// Slots for joined groups are reused; node IDs record the resulting tree.
static void distanceJoin(Tree &t,const Matrix &source,int method,bool excludeOut,
                         vector<int> roots,const vector<int> &freeNodes) {
    int n=roots.size();
    vector<int> live;
    array<int,N> sizes{};
    array<double,N> sums{};
    Matrix d=source, var=source;
    for(int x=excludeOut;x<n;++x) { live.push_back(x); sizes[x]=1; }
    size_t ni=0;
    while(live.size()>(excludeOut?2:3)) {
        int m=int(live.size());
        if(method!=2) for(int a:live) {
            double s=0;
            for(int b:live) s+=d[a*n+b];
            sums[a]=s;
        }
        int a=-1,b=-1;
        double best=INF;
        for(int i=0;i<m;++i) {
            int x=live[i];
            for(int j=i+1;j<m;++j) {
                int y=live[j];
                double s=d[x*n+y];
                if(method!=2) s=(m-2)*s-sums[x]-sums[y];
                if(s<best) { best=s; a=x; b=y; }
            }
        }
        double l=.5, dab=d[a*n+b];
        if(method==1 && var[a*n+b]>1e-12) {
            double diff=0;
            for(int k:live) if(k!=a && k!=b) diff+=var[b*n+k]-var[a*n+k];
            l=clamp(.5+diff/(2*(m-2)*var[a*n+b]),0.,1.);
        }
        double xlen=(dab+(sums[a]-sums[b])/(m-2))*.5, ylen=dab-xlen;
        if(method==2) { l=double(sizes[a])/(sizes[a]+sizes[b]); xlen=ylen=0; }
        double cross=l*(1-l)*var[a*n+b];
        for(int k:live) if(k!=a && k!=b) {
            double z=l*(d[a*n+k]-xlen)+(1-l)*(d[b*n+k]-ylen);
            d[a*n+k]=d[k*n+a]=z;
            if(method==1) var[a*n+k]=var[k*n+a]=l*var[a*n+k]+(1-l)*var[b*n+k]-cross;
        }
        int u=freeNodes[ni++];
        t.link(u,roots[a]); t.link(u,roots[b]);
        roots[a]=u; sizes[a]+=sizes[b];
        live.erase(find(live.begin(),live.end(),b));
    }
    int u=freeNodes[ni];
    for(int a:live) t.link(u,roots[a]);
    if(excludeOut) t.link(u,roots.front());
}
static Tree distTree(const Matrix &source,int method,bool excludeOut=false) {
    Tree t;
    vector<int> roots(N),freeNodes(N-2);
    iota(roots.begin(),roots.end(),0);
    iota(freeNodes.begin(),freeNodes.end(),N);
    t.next=V-2;
    distanceJoin(t,source,method,excludeOut,std::move(roots),freeNodes);
    return t;
}

struct Quartets {
    struct Step { int l,r; float fl,fr,al,ar,cl,cr; };
    vector<array<Step,N-1>> steps;
    vector<float> weights;
    float arm;
    float representative;
    explicit Quartets(const vector<Gene> &genes,int mode=0,int locus=0,float atten=0,
                      float rep=0): arm(atten),representative(rep) {
        for(const Gene &g:genes) {
            array<Step,N-1> s;
            auto fail=[&](Edge e) {
                float p=support(e);
                float l=e.len, r=e.len/max(g.rate,.01f);
                if(mode==1) return 0.f;
                if(mode==2) return 1.f-p;
                if(mode==3) return expf(-130.f*l);
                if(mode==4) return expf(-14.f*r);
                if(mode==5) return 1.f-(l>.001f);
                if(mode==6) return expf(-1.25f*p);
                if(mode==7) return expf(-2.5f*p*sqrtf(r));
                if(mode==8) return expf(-11.f*p*r);
                if(mode==9) return 1.f-p*p*l/(l+.003f);
                if(mode==10) return 1.f-sqrtf(p)*l/(l+.002f);
                if(mode==11) return expf(-.55f*p);
                if(mode==12) return expf(-45.f*l);
                if(mode==13) return 1.f-min(.99f,p*1.23f)*l/(l+.001f);
                if(mode==14) return 1.f-(l>.001f)*(.25f+.75f*p);
                if(mode==15) return expf(-p*min(4.f,95.f*l));
                if(mode==16) return 1.f-powf(p*l/(l+.0015f),.4f);
                if(mode==17) return expf(-p*min(3.5f,35.f*r));
                if(mode==18) return .57f*(1.f-powf(p,1.45f)*l/(l+.0019f))
                                    +.43f*expf(-11.f*p*r);
                if(mode==19) return expf(-1.7f*(.76f*p+.24f*l/(l+.004f)));
                return 1.f-powf(p*l/(l+.0015f),.8f);
            };
            float a=atten<0 ? -atten/max(g.rate,.025f) : atten;
            auto affinity=[&](Edge e, bool inner) {
                double l=min(2.f,e.len/max(g.rate,.025f));
                return expf(-.9f*l-(inner?.26f*support(e):0.f));
            };
            for(int i=N;i<V-1;++i) {
                const auto &t=g.node[i];
                s[i-N]={t.l,t.r,fail(t.le),fail(t.re),
                        expf(-a*t.le.len),expf(-a*t.re.len),
                        affinity(t.le,t.l>=N),affinity(t.re,t.r>=N)};
            }
            float w=1;
            if(locus==1) w=g.q;
            if(locus==2) w=powf(g.resolution,1.4f)/(1+powf(g.rate/.55f,2));
            if(locus==3) w=1/(1+powf(g.rate/.42f,2));
            if(locus==4) w=powf(g.resolution,1.7f);
            if(locus==5) w=sqrtf(g.resolution)/(1+powf(g.rate/1.3f,2));
            if(locus==6) w=1/(1+powf(.1f/max(g.rate,.001f),2)+powf(g.rate/.75f,2));
            weights.push_back(w);
            steps.push_back(s);
        }
    }
    static constexpr int x[6]={0,0,0,2,1,1}, y[6]={1,2,3,3,3,2};
    void adjust(const array<Step,N-1> &g,const int *color,const float *ws,
                const float *tot,float *wNew) const {
        array<array<float,4>,V-1> down,up;
        for(int i=0;i<N;++i) {
            auto &a=down[i]; a={};
            float w=ws?ws[i]:1.f;
            if(w) a[color[i]]=w;
        }
        for(int k=0;k<N-1;++k) {
            const auto &b=g[k];
            const auto &l=down[b.l], &r=down[b.r];
            auto &a=down[N+k];
            for(int i=0;i<4;++i) a[i]=b.cl*l[i]+b.cr*r[i];
        }
        up[V-2]={};
        for(int k=N-2;k>=0;--k) {
            const auto &b=g[k];
            const auto &u=up[N+k];
            const auto &l=down[b.l], &r=down[b.r];
            for(int i=0;i<4;++i) {
                up[b.l][i]=b.cl*(u[i]+b.cr*r[i]);
                up[b.r][i]=b.cr*(u[i]+b.cl*l[i]);
            }
        }
        float norm[4]{};
        for(int i=0;i<N;++i) {
            float w=ws?ws[i]:1.f;
            int c=color[i];
            float d=(tot[c]-w>1e-9f)?up[i][c]/(tot[c]-w):.5f;
            // A bounded density factor favors representatives from coherent
            // parts of the gene tree without discarding outlying lineages.
            float p=(.12f+max(0.f,d));
            if(representative==.5f) p=sqrtf(p);
            else if(representative==2.f) p*=p;
            wNew[i]=w*p;
            norm[c]+=wNew[i];
        }
        for(int i=0;i<4;++i) norm[i]=norm[i]>1e-12?tot[i]/norm[i]:0.f;
        for(int i=0;i<N;++i) wNew[i]*=norm[color[i]];
    }
    array<double,3> scoreArms(const int *color,const float *ws,
                              vector<array<float,3>> *per=nullptr) const {
        struct State { float f[4], out[4], p[6], miss[6]; };
        array<State,V-1> st;
        for(int i=0;i<N;++i) {
            auto &a=st[i]; a={};
            float w=ws?ws[i]:1.f;
            if(w) a.f[color[i]]=w;
        }
        array<double,3> ans{};
        if(per) per->clear();
        for(size_t gi=0;gi<steps.size();++gi) {
            const auto &g=steps[gi];
            float sc[3]{};
            for(int k=0;k<N-1;++k) {
                const Step &b=g[k];
                const State &a=st[b.l], &c=st[b.r];
                State &f=st[N+k];
                float lf[4],rf[4];
                for(int j=0;j<4;++j) {
                    lf[j]=b.al*a.f[j]; rf[j]=b.ar*c.f[j];
                    f.f[j]=lf[j]+rf[j];
                }
                for(int j=0;j<6;++j) {
                    const int u=x[j],v=y[j];
                    float n=lf[u]*rf[v]+lf[v]*rf[u];
                    f.p[j]=a.p[j]+c.p[j]+n;
                    f.miss[j]=b.fl*a.miss[j]+b.fr*c.miss[j]+n;
                }
            }
            for(float &z:st[V-2].out) z=0;
            for(int k=N-2;k>=0;--k) {
                const Step &b=g[k];
                State &a=st[b.l], &c=st[b.r];
                const State &f=st[N+k];
                float lf[4],rf[4];
                for(int j=0;j<4;++j) {
                    lf[j]=b.al*a.f[j]; rf[j]=b.ar*c.f[j];
                    a.out[j]=b.al*(f.out[j]+rf[j]); c.out[j]=b.ar*(f.out[j]+lf[j]);
                }
                float h[6],i[6],la[6],ra[6];
                for(int j=0;j<6;++j) {
                    const int u=x[j],v=y[j];
                    h[j]=b.fl*a.miss[j]; i[j]=b.fr*c.miss[j];
                    la[j]=lf[u]*f.out[v]+lf[v]*f.out[u];
                    ra[j]=rf[u]*f.out[v]+rf[v]*f.out[u];
                }
                for(int j=0;j<3;++j) {
                    const int l=j,r=j+3;
                    sc[j]+=a.p[l]*c.p[r]+a.p[r]*c.p[l]-h[l]*i[r]-h[r]*i[l]
                        +(a.p[l]-h[l])*ra[r]+(c.p[l]-i[l])*la[r]
                        +(a.p[r]-h[r])*ra[l]+(c.p[r]-i[r])*la[l];
                }
            }
            for(int j=0;j<3;++j) ans[j]+=weights[gi]*sc[j];
            if(per) per->push_back({sc[0],sc[1],sc[2]});
        }
        return ans;
    }
    // Counts each quartet at its rooted 2+2 or 3+1 branching pattern.
    // The loss term is the product of lack-of-support values on its middle path.
    array<double,3> score(const int *color,const float *ws,
                          vector<array<float,3>> *per=nullptr) const {
        if(arm!=0) return scoreArms(color,ws,per);
        struct State { float f[4], miss[6]; };
        array<State,V-1> st;
        float tot[4]{};
        for(int i=0;i<N;++i) {
            auto &a=st[i]; a={};
            float w=ws?ws[i]:1.f;
            if(w) { a.f[color[i]]=w; tot[color[i]]+=w; }
        }
        array<double,3> ans{};
        if(per) per->clear();
        int gi=0;
        for(const auto &g:steps) {
            if(representative>0) {
                float wNew[N];
                adjust(g,color,ws,tot,wNew);
                for(int i=0;i<N;++i) st[i].f[color[i]]=wNew[i];
            }
            float sc[3]{};
            for(int k=0;k<N-1;++k) {
                const Step &b=g[k];
                const State &a=st[b.l], &c=st[b.r];
                State &f=st[N+k];
                float out[4];
                for(int j=0;j<4;++j) {
                    f.f[j]=a.f[j]+c.f[j]; out[j]=tot[j]-f.f[j];
                }
                float h[6],i[6],p[6],q[6];
                for(int j=0;j<6;++j) {
                    const int u=x[j],v=y[j];
                    p[j]=a.f[u]*a.f[v]; q[j]=c.f[u]*c.f[v];
                    h[j]=b.fl*a.miss[j]; i[j]=b.fr*c.miss[j];
                    f.miss[j]=h[j]+i[j]+a.f[u]*c.f[v]+a.f[v]*c.f[u];
                }
                for(int j=0;j<3;++j) {
                    const int l=j,r=j+3;
                    sc[j]+=p[l]*q[r]+p[r]*q[l]-h[l]*i[r]-h[r]*i[l]
                        +(p[l]-h[l])*(c.f[x[r]]*out[y[r]]+c.f[y[r]]*out[x[r]])
                        +(q[l]-i[l])*(a.f[x[r]]*out[y[r]]+a.f[y[r]]*out[x[r]])
                        +(p[r]-h[r])*(c.f[x[l]]*out[y[l]]+c.f[y[l]]*out[x[l]])
                        +(q[r]-i[r])*(a.f[x[l]]*out[y[l]]+a.f[y[l]]*out[x[l]]);
                }
            }
            for(int j=0;j<3;++j) ans[j]+=weights[gi]*sc[j];
            if(per) per->push_back({sc[0],sc[1],sc[2]});
            ++gi;
        }
        return ans;
    }
};

static vector<pair<int,int>> internalEdges(const Tree &t) {
    vector<pair<int,int>> e;
    for(int i=N;i<t.next;++i) for(int k=0;k<t.deg[i];++k) {
        int j=t.adj[i][k];
        if(j>i) e.emplace_back(i,j);
    }
    return e;
}

static void colorEdge(const Tree &t,int u,int v,int *color,float *ws,float local) {
    auto [a,b]=t.children(u,v);
    auto [c,d]=t.children(v,u);
    if(local>0) {
        auto mark=[&](auto &&self,int x,int p,int group,float w)->void {
            if(x<N) { color[x]=group; ws[x]=w; }
            else for(int y:t.adj[x]) if(y!=p) self(self,y,x,group,w*local);
        };
        mark(mark,a,u,0,1); mark(mark,b,u,1,1);
        mark(mark,c,v,2,1); mark(mark,d,v,3,1);
    } else {
        t.mark(a,u,0,color); t.mark(b,u,1,color);
        t.mark(c,v,2,color); t.mark(d,v,3,color);
    }
}

static array<double,3> distanceScore(const Matrix &d,const int *color,const float *ws) {
    array<vector<pair<int,double>>,4> group;
    double mass[4]{};
    for(int i=0;i<N;++i) {
        int c=color[i];
        double w=ws?ws[i]:1;
        group[c].emplace_back(i,w); mass[c]+=w;
    }
    double m[6]{};
    for(int i=0;i<6;++i) {
        int a=Quartets::x[i],b=Quartets::y[i];
        double z=0;
        for(auto [x,w]:group[a]) {
            double s=0;
            const double *row=&d[x*N];
            for(auto [y,p]:group[b]) s+=row[y]*p;
            z+=w*s;
        }
        m[i]=z/(mass[a]*mass[b]);
    }
    return {-m[0]-m[3],-m[1]-m[4],-m[2]-m[5]};
}

static int refine(Tree &t,const Quartets &qt,mt19937 &rng,int rounds,float local=0,
                  const Matrix *matrix=nullptr,double mix=0) {
    int moves=0;
    for(int r=0;r<rounds;++r) {
        int flips=0;
        auto order=internalEdges(t);
        shuffle(order.begin(),order.end(),rng);
        for(auto [u,v]:order) {
            if(!timeLeft()) return moves+flips;
            if(find(t.adj[u].begin(),t.adj[u].end(),v)==t.adj[u].end()) continue;
            auto [a,b]=t.children(u,v);
            auto [c,d]=t.children(v,u);
            int color[N];
            float ws[N];
            colorEdge(t,u,v,color,ws,local);
            auto s=qt.score(color,local?ws:nullptr);
            if(matrix && mix) {
                auto z=distanceScore(*matrix,color,local?ws:nullptr);
                double w[4]{};
                for(int i=0;i<N;++i) w[color[i]]+=(local?ws[i]:1.f);
                double f=mix*w[0]*w[1]*w[2]*w[3];
                double a=(z[0]+z[1]+z[2])/3;
                for(int j=0;j<3;++j) s[j]+=f*(z[j]-a);
            }
            int best=int(max_element(s.begin(),s.end())-s.begin());
            if(best!=0 && s[best]>s[0]+1e-5*max(1.,s[best])) {
                if(best==1) t.swapBranches(u,v,b,c);
                else t.swapBranches(u,v,b,d);
                ++flips;
            }
        }
        moves+=flips;
        if(!flips) break;
    }
    return moves;
}

static void refineDistance(Tree &t,const Matrix &d,mt19937 &rng,int rounds,float locality=.5f) {
    for(int r=0;r<rounds;++r) {
        int flips=0;
        auto order=internalEdges(t);
        shuffle(order.begin(),order.end(),rng);
        for(auto [u,v]:order) {
            if(!timeLeft()) return;
            if(find(t.adj[u].begin(),t.adj[u].end(),v)==t.adj[u].end()) continue;
            int color[N];
            float ws[N];
            colorEdge(t,u,v,color,ws,locality);
            auto s=distanceScore(d,color,locality?ws:nullptr);
            int j=int(max_element(s.begin(),s.end())-s.begin());
            if(j && s[j]>s[0]+1e-9) {
                auto [a,b]=t.children(u,v);
                auto [c,e]=t.children(v,u);
                t.swapBranches(u,v,b,j==1?c:e);
                ++flips;
            }
        }
        if(!flips) return;
    }
}

#include "clock.hpp"
#include "calibration.hpp"

struct PatchModel {
    const Quartets *primary;
    const Quartets *secondary=nullptr;
    double mix=.25;
    // Optional additive internode contrast.
    const Matrix *dist=nullptr;
    double distMix=0;
    float locality=.87;
    double posterior=0;
    double splitMix=0;
    int boot=0;
    double edgeMix=0;
    const Triplets *trip=nullptr;
    double tripMix=0;
    function<double(const Mask &)> prior={};
};

using SplitWeights=unordered_map<Mask,double,MaskHash>;

struct Patch {
    struct Tip { int old, outside; vector<pair<int,float>> leaves; };
    vector<int> nodes;
    vector<Tip> tips;
    void collect(const Tree &t,float local) {
        auto append=[&](auto &&self,Tip &tip,int u,int p,float w)->void {
            if(u<N) tip.leaves.emplace_back(u,w);
            else for(int v:t.adj[u]) if(v!=p) self(self,tip,v,u,w*local);
        };
        for(int u:nodes) for(int v:t.adj[u]) if(find(nodes.begin(),nodes.end(),v)==nodes.end()) {
            Tip tip{u,v,{}};
            append(append,tip,v,u,1.f);
            sort(tip.leaves.begin(),tip.leaves.end());
            double sum=0; for(auto xy:tip.leaves) sum+=xy.second;
            for(auto &xy:tip.leaves) xy.second/=sum;
            tips.push_back(std::move(tip));
        }
        sort(tips.begin(),tips.end(),[](const Tip &a,const Tip &b) {
            // Place the outgroup in the final (anchor) slot.
            int x=a.leaves.front().first, y=b.leaves.front().first;
            return (x?x:N)<(y?y:N);
        });
    }
    struct Q {
        int a,b,c,d;
        array<double,3> s;
        vector<array<float,3>> per;
    };
    struct Trip { int a,b,c; array<double,3> s; };
    vector<Trip> tripEvidence(const PatchModel &model) const {
        vector<Trip> ans;
        if(!model.trip) return ans;
        int k=int(tips.size());
        double p=model.tripMix;
        for(int a=0;a<k-1;++a) for(int b=a+1;b<k-1;++b) for(int c=b+1;c<k-1;++c) {
            int color[N]{};
            float w[N]{};
            int ids[]={a,b,c};
            for(int i=0;i<3;++i) for(auto [x,y]:tips[ids[i]].leaves) {
                color[x]=i; w[x]=y;
            }
            auto s=model.trip->score(color,w);
            double sum=s[0]+s[1]+s[2];
            for(double &x:s) x=(x-sum/3.)*(p*36./max(4.,sum));
            ans.push_back({a,b,c,s});
        }
        return ans;
    }
    vector<Q> evidence(const PatchModel &model) const {
        int k=int(tips.size());
        double dist[12][12]{};
        if(model.dist) {
            for(int a=0;a<k;++a) for(int b=a+1;b<k;++b) {
                double s=0;
                for(auto [i,p]:tips[a].leaves) for(auto [j,q]:tips[b].leaves)
                    s+=(*model.dist)[i*N+j]*p*q;
                dist[a][b]=dist[b][a]=s;
            }
        }
        vector<Q> res;
        for(int a=0;a<k;++a) for(int b=a+1;b<k;++b) for(int c=b+1;c<k;++c)
            for(int d=c+1;d<k;++d) {
                int color[N]{};
                float w[N]{};
                int ids[]={a,b,c,d};
                for(int i=0;i<4;++i) for(auto [x,y]:tips[ids[i]].leaves) {
                    color[x]=i; w[x]=y;
                }
                vector<array<float,3>> per,other;
                auto s=model.primary->score(color,w,model.boot?&per:nullptr);
                if(model.boot) for(size_t j=0;j<per.size();++j)
                    for(float &p:per[j]) p*=model.primary->weights[j];
                if(model.secondary) {
                    auto t=model.secondary->score(color,w,model.boot?&other:nullptr);
                    for(int j=0;j<3;++j) s[j]=(1-model.mix)*s[j]+model.mix*t[j];
                    if(model.boot) for(size_t g=0;g<per.size();++g) for(int j=0;j<3;++j)
                        per[g][j]=(1-model.mix)*per[g][j]+model.mix*model.secondary->weights[g]*other[g][j];
                }
                if(model.dist) {
                    s[0]-=model.distMix*(dist[a][b]+dist[c][d]);
                    s[1]-=model.distMix*(dist[a][c]+dist[b][d]);
                    s[2]-=model.distMix*(dist[a][d]+dist[b][c]);
                }
                double avg=(s[0]+s[1]+s[2])/3;
                for(double &x:s) x-=avg;
                if(model.boot) for(auto &a:per) {
                    float mean=(a[0]+a[1]+a[2])/3;
                    for(float &x:a) x-=mean;
                }
                res.push_back({a,b,c,d,s,std::move(per)});
            }
        return res;
    }
    vector<double> splitEvidence(const Quartets &q) const {
        constexpr int K=11;
        int k=int(tips.size()), top=(1<<k)-1, all=top>>1, anchor=1<<(k-1);
        vector<double> support(all+1,0);
        array<array<float,K>,V-1> mass{};
        for(int i=0;i<k;++i) for(auto [x,w]:tips[i].leaves) mass[x][i]=w;
        for(size_t gi=0;gi<q.steps.size();++gi) {
            vector<float> accum(all+1);
            auto inspect=[&](int u,float weight) {
                if(u<N || weight<.001f) return;
                int forced=0;
                array<int,K> bits{};
                array<float,K> frac{};
                int m=0;
                for(int j=0;j<k;++j) {
                    float p=mass[u][j];
                    if(p>1.f-1e-6f) forced|=1<<j;
                    else if(p>1e-6f) { bits[m]=1<<j; frac[m++]=p; }
                }
                auto sum=[&](auto &&self,int i,int mask,float prob)->void {
                    if(prob<1e-7f) return;
                    if(i==m) {
                        if(mask&anchor) mask^=top;
                        if(mask!=all && (mask&(mask-1))) accum[mask]+=prob;
                        return;
                    }
                    self(self,i+1,mask,prob*(1-frac[i]));
                    self(self,i+1,mask|bits[i],prob*frac[i]);
                };
                sum(sum,0,forced,weight);
            };
            for(int s=0;s<N-1;++s) {
                const auto &e=q.steps[gi][s];
                auto &z=mass[N+s];
                for(int a=0;a<k;++a) z[a]=mass[e.l][a]+mass[e.r][a];
                if(s!=N-2) {
                    inspect(e.l,1-e.fl); inspect(e.r,1-e.fr);
                }
            }
            for(int s=1;s<all;++s) support[s]+=q.weights[gi]*(-expm1f(-accum[s]));
        }
        return support;
    }
    static double logAdd(double a,double b) {
        if(a<b) swap(a,b);
        if(a-b>40) return a;
        return a+log1p(exp(b-a));
    }
    static array<int,81> quartetRelations() {
        array<int,81> relation{};
        for(int i=0;i<81;++i) {
            int x[4]={i%3,i/3%3,i/9%3,i/27%3}, counts[3]{};
            for(int v:x) ++counts[v];
            if(counts[0] && counts[1] && counts[2]) {
                for(int j=1;j<4;++j) if(x[0]==x[j]) relation[i]=j;
                for(int j=1;j<4;++j) if(x[0]!=x[j] && x[(j%3)+1]==x[(j+1)%3+1])
                    relation[i]=j;
            }
        }
        return relation;
    }
    static int pairing(const Q &q,const int *c,const array<int,81>& relation) {
        return relation[c[q.a]+3*c[q.b]+9*c[q.c]+27*c[q.d]];
    }
    static double tripScore(const vector<Trip> &ts,const int *c) {
        double v=0;
        for(const auto &t:ts) {
            int a=c[t.a],b=c[t.b],d=c[t.c];
            if(a==2 || b==2 || d==2 || (a==b && b==d)) continue;
            v+=t.s[a==b?0:a==d?1:2];
        }
        return v;
    }
    static vector<int> chooseBoot(const vector<Q> &qs,const vector<double> &split,
                                  const vector<Trip> &ts,int k,int boot,mt19937 &rng,
                                  vector<double> *marginal) {
        constexpr int B=32;
        using Lane=array<float,B>;
        int all=(1<<(k-1))-1, m=qs.front().per.size();
        vector<Lane> w(m);
        gamma_distribution<float> gamma(boot==2?.8f:1.6f,boot==2?1.25f:.625f);
        for(auto &a:w) for(float &x:a) x=gamma(rng);
        for(int j=0;j<B;++j) {
            float sum=0;
            for(int i=0;i<m;++i) sum+=w[i][j];
            float scale=m/max(sum,.001f);
            for(int i=0;i<m;++i) w[i][j]=w[i][j]*scale-1;
        }
        vector<array<Lane,4>> support(qs.size());
        for(size_t qi=0;qi<qs.size();++qi) for(int j=0;j<3;++j) {
            const Q &q=qs[qi];
            Lane &p=support[qi][j+1]; p.fill(float(q.s[j]*.5));
            for(int i=0;i<m;++i) for(int r=0;r<B;++r) p[r]+=.5f*w[i][r]*q.per[i][j];
        }
        vector<Lane> dp(all+1);
        vector<array<uint16_t,B>> choose(all+1);
        const auto relation=quartetRelations();
        for(int s=1;s<=all;++s) {
            int rem=s&(s-1);
            if(!rem) continue;
            Lane &v=dp[s]; v.fill(-1e30f);
            for(int b=rem;b;b=(b-1)&rem) {
                int a=s^b;
                int c[12];
                for(int i=0;i<k;++i) c[i]=(b>>i&1)+2*int(!(s>>i&1));
                Lane cost;
                float extra=split[s]+tripScore(ts,c);
                for(int h=0;h<B;++h) cost[h]=extra+dp[a][h]+dp[b][h];
                for(size_t i=0;i<qs.size();++i) {
                    const auto &z=support[i][pairing(qs[i],c,relation)];
                    for(int h=0;h<B;++h) cost[h]+=z[h];
                }
                for(int h=0;h<B;++h) if(cost[h]>v[h]) {
                    v[h]=cost[h]; choose[s][h]=b;
                }
            }
        }
        vector<int> count(all+1), choice(all+1);
        auto tally=[&](auto &&self,int s,int h)->void {
            if(!(s&(s-1))) return;
            ++count[s];
            int b=choose[s][h];
            self(self,b,h); self(self,s^b,h);
        };
        for(int h=0;h<B;++h) tally(tally,all,h);
        if(marginal) {
            marginal->resize(all+1);
            for(int s=1;s<=all;++s) (*marginal)[s]=double(count[s])/B;
        }
        vector<double> score(all+1);
        for(int s=1;s<=all;++s) {
            int rem=s&(s-1);
            if(!rem) continue;
            double best=-INF;
            for(int b=rem;b;b=(b-1)&rem) {
                int a=s^b;
                double val=score[a]+score[b];
                if(val>best) { best=val; choice[s]=b; }
            }
            score[s]=best+count[s];
        }
        return choice;
    }
    vector<int> choosePosterior(const PatchModel &model,const vector<Q> &qs,
                                const vector<double> &split,const vector<Trip> &ts,int k,
                                vector<double> *marginal) const {
        int all=(1<<(k-1))-1;
        vector<double> dp(all+1,0), outside(all+1,-INF);
        vector<int> choice(all+1,0);
        struct Part { int a,b; double val; };
        vector<vector<Part>> parts(all+1);
        auto relation=quartetRelations();
        for(int s=1;s<=all;++s) {
            int rem=s&(s-1);
            if(!rem) continue;
            double best=-INF, zz=-INF;
            for(int b=rem;b;b=(b-1)&rem) {
                int a=s^b;
                int c[12];
                for(int i=0;i<k;++i) c[i]=(b>>i&1)+2*int(!(s>>i&1));
                double cost=0;
                for(const Q &q:qs) {
                    int typ=pairing(q,c,relation);
                    if(typ) cost+=q.s[typ-1];
                }
                cost=.5*cost+split[s]+tripScore(ts,c);
                if(model.posterior>0) {
                    double val=cost/(model.posterior*max(1,k-3));
                    parts[s].push_back({a,b,val});
                    zz=logAdd(zz,val+dp[a]+dp[b]);
                } else if(double v=cost+dp[a]+dp[b];v>best) {
                    best=v; choice[s]=b;
                }
            }
            dp[s]=(model.posterior>0)?zz:best;
        }
        if(model.posterior>0) {
            if(marginal) marginal->assign(all+1,1.);
            outside[all]=0;
            for(int s=all;s;--s) for(auto [a,b,x]:parts[s]) {
                outside[a]=logAdd(outside[a],outside[s]+x+dp[b]);
                outside[b]=logAdd(outside[b],outside[s]+x+dp[a]);
            }
            double z=dp[all];
            for(int s=1;s<=all;++s) {
                if(!(s&(s-1))) { dp[s]=0; continue; }
                double p=exp(outside[s]+dp[s]-z), best=-INF;
                if(marginal) (*marginal)[s]=p;
                for(auto [a,b,x]:parts[s]) if(double v=dp[a]+dp[b];v>best) {
                    best=v; choice[s]=b;
                }
                dp[s]=p+best;
            }
        }
        return choice;
    }
    static vector<int> chooseLocal(const vector<Q> &qs,const vector<double> &split,
                                   const vector<Trip> &ts,int k,double mix) {
        int n=1<<(k-1), all=n-1, anchor=n, top=2*n-1;
        auto offset=[&](int a,int b,int c,int d) { return ((a*k+b)*k+c)*k+d; };
        vector<float> table(k*k*k*k);
        for(const Q &q:qs) {
            for(int j=0;j<3;++j) {
                int a=q.a,b=q.b,c=q.c,d=q.d;
                if(j==1) swap(b,c);
                if(j==2) swap(b,d);
                for(int p=0;p<2;++p) {
                    table[offset(a,b,c,d)]=table[offset(b,a,c,d)]=
                        table[offset(a,b,d,c)]=table[offset(b,a,d,c)]=q.s[j];
                    swap(a,c); swap(b,d);
                }
            }
        }
        vector<vector<int>> leaves(2*n);
        for(int s=1;s<=top;++s) {
            int a=s&-s;
            leaves[s]=leaves[s^a];
            leaves[s].push_back(__builtin_ctz(unsigned(a)));
        }
        const auto relation=quartetRelations();
        vector<float> dp(n*n,0);
        vector<uint16_t> trace(n*n,0);
        double scale=mix*k*(k-1)*(k-2)/24.;
        for(int s=1;s<=all;++s) {
            int rem=s&(s-1);
            if(!rem) continue;
            int ext=all^s;
            auto at=[&](int y)->float& { return dp[s*n+y]; };
            for(int y=ext;y;y=(y-1)&ext) at(y)=-1e30f;
            at(0)=-1e30f;
            for(int b=rem;b;b=(b-1)&rem) {
                int a=s^b;
                int colors[12];
                for(int i=0;i<k;++i) colors[i]=(b>>i&1)+2*int(!(s>>i&1));
                double cost=0;
                for(const auto &q:qs) {
                    int j=pairing(q,colors,relation);
                    if(j) cost+=q.s[j-1];
                }
                float base=float(.5*(1-mix)*cost+split[s]+tripScore(ts,colors)+dp[a*n+b]+dp[b*n+a]);
                if(!ext && base>at(0)) { at(0)=base; trace[s*n]=b; }
                for(int y=ext;y;y=(y-1)&ext) {
                    int z=(ext^y)|anchor;
                    double v=0, d=0;
                    for(int i:leaves[a]) for(int j:leaves[b])
                        for(int p:leaves[y]) for(int q:leaves[z]) {
                            v+=table[offset(i,j,p,q)]; d+=table[offset(i,p,j,q)];
                        }
                    double cnt=double(leaves[a].size()*leaves[b].size()*leaves[y].size()*leaves[z].size());
                    double signal=(1.5*v-.17*abs(2*d+v))/cnt;
                    double edge=copysign(9*log1p(abs(signal)/9),signal);
                    float value=base+scale*edge;
                    if(value>at(y)) { at(y)=value; trace[s*n+y]=b; }
                }
            }
        }
        vector<int> choice(n);
        auto visit=[&](auto &&self,int s,int y)->void {
            if(!(s&(s-1))) return;
            int b=trace[s*n+y];
            choice[s]=b;
            self(self,b,s^b); self(self,s^b,b);
        };
        visit(visit,all,0);
        return choice;
    }
    void solve(Tree &t,const PatchModel &model,mt19937 &rng,SplitWeights *uncertainty) const {
        int k=int(tips.size()), all=(1<<(k-1))-1;
        auto qs=evidence(model);
        auto ts=tripEvidence(model);
        vector<double> split(all+1,0);
        if(model.splitMix>0 && k>4) {
            split=splitEvidence(*model.primary);
            for(double &r:split) r*=model.splitMix*(k-3);
        }
        vector<Mask> subtree;
        if(model.prior || uncertainty) {
            vector<Mask> group(k);
            for(int i=0;i<k;++i) {
                for(auto [x,w]:tips[i].leaves) group[i]|=Mask(x);
            }
            subtree.resize(all+1);
            for(int s=1;s<=all;++s) {
                int b=s&-s;
                subtree[s]=subtree[s^b]|group[__builtin_ctz(unsigned(b))];
                if(model.prior && s!=all && (s&(s-1)))
                    split[s]+=(k-3)*model.prior(subtree[s]);
            }
        }
        vector<double> support;
        auto dst=uncertainty?&support:nullptr;
        auto choice=model.boot?chooseBoot(qs,split,ts,k,model.boot,rng,dst):
            model.edgeMix>0?chooseLocal(qs,split,ts,k,model.edgeMix):
            choosePosterior(model,qs,split,ts,k,dst);
        if(uncertainty && !support.empty()) for(int s=1;s<all;++s) {
            if(s&(s-1)) (*uncertainty)[subtree[s]]=clamp(support[s],0.,1.);
        }
        for(const auto &tip:tips) t.unlink(tip.old,tip.outside);
        for(int u:nodes) while(t.deg[u]) t.unlink(u,t.adj[u][t.deg[u]-1]);
        int ni=0;
        auto rebuild=[&](auto &&self,int s)->int {
            if(!(s&(s-1))) return tips[__builtin_ctz(unsigned(s))].outside;
            int b=choice[s],a=s^b, u=nodes[ni++];
            t.link(u,self(self,a)); t.link(u,self(self,b));
            return u;
        };
        int at=rebuild(rebuild,all);
        t.link(at,tips.back().outside);
    }
};

struct Reliability {
    int u,v;
    double z, margin;
};
static vector<Reliability> edgeReliability(const Tree &t,const Quartets &q,float local) {
    vector<Reliability> conf;
    vector<array<float,3>> per;
    for(auto [u,v]:internalEdges(t)) {
        if(!timeLeft()) break;
        int color[N];
        float w[N];
        colorEdge(t,u,v,color,w,local);
        auto s=q.score(color,local?w:nullptr,&per);
        int b=1+(s[2]>s[1]);
        double n=0,sum=0,ss=0;
        for(size_t i=0;i<per.size();++i) {
            double x=q.weights[i]*(per[i][0]-per[i][b]);
            sum+=x; ss+=x*x; n+=1;
        }
        double z=sum/sqrt(max(.00000000001,ss-sum*sum/n));
        conf.push_back({u,v,clamp(z,-99.,99.),(s[0]-s[b])/max(1e-7,s[0]+s[1]+s[2])});
    }
    return conf;
}

static int patches(Tree &t,const PatchModel &model,mt19937 &rng,int maxTips,double maxZ,
                   double variation=0,SplitWeights *uncertainty=nullptr) {
    auto conf=edgeReliability(t,*model.primary,model.locality);
    uniform_real_distribution<double> noise(-variation,variation);
    for(auto &e:conf) e.z+=noise(rng);
    sort(conf.begin(),conf.end(),[](const auto &a,const auto &b) { return a.z<b.z; });
    array<int,V> parent{}, sizes{};
    iota(parent.begin(),parent.end(),0);
    sizes.fill(1);
    auto find=[&](int u) {
        while(u!=parent[u]) { parent[u]=parent[parent[u]]; u=parent[u]; }
        return u;
    };
    for(auto e:conf) if(e.z<maxZ) {
        int a=find(e.u), b=find(e.v);
        if(a!=b && sizes[a]+sizes[b]<=maxTips-2) {
            parent[b]=a; sizes[a]+=sizes[b];
        }
    }
    array<vector<int>,V> group;
    for(int u=N;u<t.next;++u) group[find(u)].push_back(u);
    int nt=0;
    for(auto &g:group) if(g.size()>1) {
        if(!timeLeft()) break;
        Patch p;
        p.nodes=std::move(g);
        p.collect(t,model.locality);
        p.solve(t,model,rng,uncertainty);
        nt+=int(p.nodes.size())-1;
    }
    if(uncertainty) for(const auto &s:treeSplits(t)) uncertainty->emplace(s,1.);
    return nt;
}

// A regraft moving along one backbone edge changes only quartets with
// one taxon in the pruned clade and three backbones at the crossed vertex.
static bool placeClade(Tree &t,int a,int root,const Quartets &q,int depth,double balance) {
    auto [b,c]=t.children(root,a);
    Tree rest=t;
    rest.unlink(root,a); rest.unlink(root,b); rest.unlink(root,c); rest.link(b,c);
    int base[N]{};
    t.mark(a,root,1,base);
    int sizeA=accumulate(base,base+N,0);
    int winB=b, winC=c;
    double best=1e-7;
    auto search=[&](auto &&self,int u,int p,int d,double score)->void {
        if(u<N || d<=0) return;
        auto [y,z]=rest.children(u,p);
        int color[N], sz[4]{};
        for(int i=0;i<N;++i) color[i]=base[i]?0:1;
        rest.mark(y,u,2,color); rest.mark(z,u,3,color);
        for(int v:color) ++sz[v];
        auto v=q.score(color,nullptr);
        double n=double(sz[0])*sz[1]*sz[2]*sz[3];
        double factor=balance==0?1.:pow(n,-balance);
        double s=score+(v[1]-v[0])*factor, r=score+(v[2]-v[0])*factor;
        if(s>best) { best=s; winB=u; winC=y; }
        if(r>best) { best=r; winB=u; winC=z; }
        // The expansive searches stay around uncertain attachment points.
        double loss=6*q.steps.size()/50.*pow(sizeA*max(1,int((N-sizeA)*.17)),1-balance);
        if(s>best-loss) self(self,y,u,d-1,s);
        if(r>best-loss) self(self,z,u,d-1,r);
    };
    search(search,b,c,depth,0); search(search,c,b,depth,0);
    if(winB==b && winC==c) return false;
    rest.unlink(winB,winC);
    rest.link(root,winB); rest.link(root,winC); rest.link(root,a);
    t=rest;
    return true;
}

static int sprPass(Tree &t,const Quartets &q,mt19937 &rng,int attempts,int depth,double balance) {
    auto conf=edgeReliability(t,q,0);
    array<double,V> rating;
    rating.fill(99.);
    for(auto [a,b,z,m]:conf) {
        rating[a]=min(rating[a],z); rating[b]=min(rating[b],z);
    }
    struct Cut { int a,b; double priority; };
    vector<Cut> cuts;
    for(int b=N;b<t.next;++b) {
        if(rating[b]>3.9) continue;
        for(int a:t.adj[b]) {
            double v=rating[b]+.16*(a>=N);
            cuts.push_back({a,b,v});
        }
    }
    shuffle(cuts.begin(),cuts.end(),rng);
    stable_sort(cuts.begin(),cuts.end(),[](const Cut &a,const Cut &b) { return a.priority<b.priority; });
    int moves=0;
    for(auto [a,b,z]:cuts) {
        if(!timeLeft()) break;
        if(find(t.adj[b].begin(),t.adj[b].end(),a)==t.adj[b].end()) continue;
        if(!attempts--) break;
        moves+=placeClade(t,a,b,q,depth,balance);
    }
    return moves;
}

struct SplitPool {
    SplitWeights frequency;
    double total=0;
    void add(const Tree &t,double weight) {
        for(const auto &s:treeSplits(t)) frequency[s]+=weight;
        total+=weight;
    }
    void add(const SplitWeights &p,double weight) {
        for(const auto &[s,f]:p) if(f>0.006) frequency[s]+=weight*f;
        total+=weight;
    }
    double support(const Mask &s) const {
        auto it=frequency.find(s);
        return it==frequency.end()?0:it->second/total;
    }
    void combine(const SplitPool &other,double weight) {
        if(weight<=0 || other.total<=0) return;
        for(const auto &[s,w]:other.frequency) frequency[s]+=weight*w/other.total;
        total+=weight;
    }
    Tree consensus() const {
        struct Entry {
            Mask s;
            double w;
            int size, first;
        };
        vector<Entry> all;
        Mask complete;
        for(int i=1;i<N;++i) {
            Mask s(i); complete|=s;
            all.push_back({s,0,1,i});
        }
        for(const auto &[s,w]:frequency) all.push_back({s,w,s.count(),s.first()});
        all.push_back({complete,0,N-1,1});
        sort(all.begin(),all.end(),[](const Entry &a,const Entry &b) {
            if(a.size!=b.size) return a.size<b.size;
            return a.s<b.s;
        });
        unordered_map<Mask,int,MaskHash> id;
        id.reserve(all.size()*2);
        array<vector<int>,N> byFirst;
        vector<double> score(all.size(),-INF);
        vector<pair<int,int>> choice(all.size(),{-1,-1});
        for(int i=0;i<int(all.size());++i) {
            const auto &[s,w,n,f]=all[i];
            if(n==1) score[i]=0;
            else for(int a:byFirst[f]) {
                if(all[a].size==n) break;
                if(!all[a].s.inside(s)) continue;
                auto b=id.find(s^all[a].s);
                if(b==id.end()) continue;
                double x=score[a]+score[b->second]+w;
                if(x>score[i]) { score[i]=x; choice[i]={a,b->second}; }
            }
            id.emplace(s,i);
            byFirst[f].push_back(i);
        }
        Tree t;
        auto make=[&](auto &&self,int i)->int {
            if(all[i].size==1) return all[i].first;
            auto [a,b]=choice[i];
            if(a<0 || b<0) throw runtime_error("incompatible consensus clades");
            int u=t.next++;
            t.link(u,self(self,a)); t.link(u,self(self,b));
            return u;
        };
        int r=make(make,int(all.size()-1));
        t.link(r,0);
        return t;
    }
};

static Tree skeleton(Tree t,const SplitPool &pool,const Matrix &d,int cap,
                     double conf,float locality,int method) {
    array<Mask,V> masks;
    auto fill=[&](auto &&self,int u,int p)->Mask {
        if(u<N) return masks[u]=Mask(u);
        auto [a,b]=t.children(u,p);
        return masks[u]=self(self,a,u)|self(self,b,u);
    };
    int r=t.adj[0][0];
    fill(fill,r,0);
    Patch patch;
    auto freeze=[&](auto &&self,int u,int p)->void {
        if(u<N) return;
        if(p!=0 && masks[u].count()<=cap && pool.support(masks[u])>=conf) return;
        patch.nodes.push_back(u);
        auto [a,b]=t.children(u,p);
        self(self,a,u); self(self,b,u);
    };
    freeze(freeze,r,0);
    patch.collect(t,locality);
    const int k=patch.tips.size();
    if(k<4 || k>N-4) return t;
    rotate(patch.tips.begin(),patch.tips.end()-1,patch.tips.end());
    Matrix m(k*k);
    vector<int> roots;
    for(int a=0;a<k;++a) {
        const auto &x=patch.tips[a];
        roots.push_back(x.outside);
        for(int b=a+1;b<k;++b) {
            const auto &y=patch.tips[b];
            double val=0;
            for(auto [i,u]:x.leaves) for(auto [j,v]:y.leaves) val+=d[i*N+j]*u*v;
            m[a*k+b]=m[b*k+a]=val;
        }
    }
    for(const auto &x:patch.tips) t.unlink(x.old,x.outside);
    for(int u:patch.nodes) while(t.deg[u]) t.unlink(u,t.adj[u][t.deg[u]-1]);
    distanceJoin(t,m,method,method==2,std::move(roots),patch.nodes);
    return t;
}

static Matrix weightedDistance(const Distances &ds,initializer_list<pair<int,double>> terms) {
    Matrix d(N*N);
    for(auto [i,w]:terms) for(size_t j=0;j<d.size();++j) d[j]+=w*ds.d[i][j];
    return d;
}

static Matrix blend(const Matrix &a,const Matrix &b,double w) {
    Matrix d(a.size());
    for(size_t k=0;k<d.size();++k) d[k]=(1-w)*a[k]+w*b[k];
    return d;
}

// Repeatability of the distance summaries across disjoint sets of loci.
static array<double,5> repeatability(const Distances &ds,mt19937 rng) {
    int m=int(ds.boot.size());
    vector<int> order(m);
    iota(order.begin(),order.end(),0);
    vector<float> a(m),b(m);
    array<double,5> score{};
    for(int r=0;r<3 && timeLeft();++r) {
        shuffle(order.begin(),order.end(),rng);
        for(int g=0;g<m;++g) { a[order[g]]=g<m/2; b[order[g]]=g>=m/2; }
        for(int k=0;k<5;++k) {
            auto x=treeSplits(distTree(ds.sample(a,k),r%2));
            auto y=treeSplits(distTree(ds.sample(b,k),r%2));
            size_t u=0,v=0;
            int n=0;
            while(u<x.size() && v<y.size()) {
                if(x[u]==y[v]) { ++u; ++v; ++n; }
                else if(x[u]<y[v]) ++u;
                else ++v;
            }
            score[k]+=n/(3.*(N-3));
        }
    }
    return score;
}

// Distant pairs and almost-unresolved short alignments behave differently.
// The ensemble uses independent starting summaries and modifies the balance
// of length and quartet evidence according to observed locus agreement.
static Tree infer(const vector<Gene> &genes,const Distances &ds,mt19937 &rng,
                  const Matrix &topo,Tree first,double spread,double signal) {
    size_t ng=genes.size();
    const bool highConflict=signal<.235;
    const bool calibrated=!highConflict && spread<1.85;
    double lengthTrust=1/(1+max(0.,spread-1.25)*1.6);
    array<double,5> stableScores{};
    if(ng>=8) {
        stableScores=repeatability(ds,rng);
        double gain=stableScores[3]-stableScores[0];
        lengthTrust=.62*lengthTrust+.38/(1+exp(-15*(gain+.037)));
    }
    Quartets q(genes,18), base(genes,0), fine(genes,9), length(genes,8);
    Quartets reps(genes,18,0,0,.5f), arms(genes,0,0,-.42f);
    auto near=weightedDistance(ds,{{29,.49},{35,.36},{24,.15}});
    auto paths=weightedDistance(ds,{{2,.64},{25,.22},{19,.14}});
    if(!timeLeft()) return first;
    ClockMatrices cm(genes);
    Triplets ts(cm,0,.68f), anchored(cm,1,.55f);
    auto rank=blend(cm.rank[0],cm.rank[1],.65);
    auto clock=blend(near,rank,.60);
    auto high=blend(topo,paths,.60*lengthTrust);
    auto soft=blend(clock,topo,.26);
    double discord=clamp((.37-signal)/.25,0.,1.);
    double severe=clamp((.25-signal)/.15,0.,1.);
    double strength=max(stableScores[1],stableScores[2])-stableScores[0];
    discord=clamp(discord+min(.24,max(0.,3.7*strength)),0.,1.);
    auto joint=blend(topo,soft,.25+.58*discord);
    auto lower=blend(ds.d[32],ds.d[25],.42);
    auto dcombo=blend(ds.d[34],ds.d[35],.23);
    auto stable=blend(ds.d[29],ds.d[35],.30);
    auto early=blend(stable,rank,.30+.34*discord);
    if(calibrated) {
        auto ref=blend(ds.d[25],ds.d[32],.46);
        CalibratedMatrices cal(genes,first,ref);
        lower=blend(lower,blend(cal.d[3],cal.d[6],.61),.32);
        dcombo=blend(dcombo,cal.d[2],.14+.22*lengthTrust);
    }
    joint=blend(joint,blend(early,lower,.64*lengthTrust),.22+.23*discord);
    double qScale=ng/50.;
    double prior=qScale*(8+46*discord);
    prior+=qScale*(7+17*lengthTrust);
    SplitPool pool;
    pool.add(first,.55+1.4*discord);
    struct Start {
        const Matrix *d;
        const Quartets *q;
        int method;
        float focus;
        double distance, vote;
    };
    const Start variants[]={
        {&topo,&q,0,0, .26,3.4},
        {&topo,&fine,1,.92f,.13,3.0},
        {&high,&base,0,.82f,.38,2.7},
        {&near,&q,1,.91f,.83,2.8},
        {&joint,&reps,1,.84f,.65,3.0},
        {&soft,&base,0,.78f,1.25,2.4},
        {&ds.d[35],spread<1.8?&arms:&length,0,0, .50,2.5},
        {&clock,&fine,1,.88f,.92,2.5},
        {&rank,&reps,discord>.57?2:1,.95f,1.15,2.3},
        {&joint,&q,0,.71f,1.4,2.35},
        {&lower,&q,0,.82f,1.1,2.7+2.5*lengthTrust},
        {&dcombo,&fine,0,.86f,1.03,3.7},
        {&stable,&base,1,.81f,1.36,2.75+1.8*discord},
        {&early,&q,discord>.70?2:1,.81f,1.62,2.3+2.1*discord},
        {&ds.d[32],&q,0,.82f,1.05,1.6+4.3*lengthTrust},
        {&dcombo,&q,1,.79f,1.12,3.9},
        {&stable,&base,1,.83f,1.55,2.4+2.0*discord},
    };
    for(size_t i=0;i<size(variants) && timeLeft();++i) {
        const auto &st=variants[i];
        Tree t=distTree(*st.d,st.method,st.method==2);
        if(i==3 || i==5 || i>=7) {
            refineDistance(t,*st.d,rng,3,i==8?.74f:.50f);
            pool.add(t,.22+discord*(st.d==&rank?2.8:1.1));
        }
        const Matrix *metric=i>=10 ? st.d : &joint;
        if(i>=10) {
            Tree purely=distTree(*st.d,st.method,st.method==2);
            refineDistance(purely,*st.d,rng,5,.82f);
            pool.add(purely,.75+2.85*discord+1.19*lengthTrust);
            if(i%2) t=purely;
        }
        double mult=i>=10 ? qScale*(44+61*discord) : prior;
        refine(t,*st.q,rng,5,st.focus,metric,st.distance*mult);
        int type=i>=10?(i==10||i==14?3:i==11||i==15?4:2):i<=2?0:1;
        double vote=st.vote*clamp(exp(4.3*(stableScores[type]-stableScores[0])),.52,1.58);
        pool.add(t,vote);
        if(i==0 && timeLeft()) {
            Tree s=t;
            sprPass(s,q,rng,31,6,.26);
            pool.add(s,1.8);
        }
    }
    gamma_distribution<float> resample(1.65f,1.f/1.65f);
    vector<float> weights(ng);
    int repsCount=highConflict?10:12;
    for(int b=0;b<repsCount && timeLeft();++b) {
        for(float &x:weights) x=.15f+.85f*resample(rng);
        int metric=(b+int(discord>.5))%3;
        Matrix dist=ds.sample(weights,metric);
        if(metric) {
            double mean=0;
            for(int i=1;i<N;++i) for(int j=i+1;j<N;++j) mean+=dist[i*N+j];
            mean=max(1e-6,mean/((N-1)*(N-2)/2));
            for(double &x:dist) x/=mean;
            dist=blend(dist,joint,.12+discord*.14);
        }
        Tree t=distTree(dist,(b/3)%2);
        Quartets qm(genes,b%4==0?0:b%4==1?9:18,0,0,b%7==3?.5f:0);
        for(size_t i=0;i<ng;++i) qm.weights[i]*=weights[i];
        refine(t,qm,rng,4,b%4==1?.86f:0.f,&joint,prior*.5);
        pool.add(t,1.12);
    }
    if(highConflict) {
        double wLen=1.0+1.5*lengthTrust;
        for(int b=0;b<10 && timeLeft();++b) {
            for(float &x:weights) x=.30f+.70f*resample(rng);
            Matrix d;
            int m=b%4;
            if(m==0 || m==2) {
                d=ds.trimmed(weights,m==2?.60:.46);
                d=blend(d,stable,.10+.27*(1-lengthTrust)+.15*discord);
            } else {
                d=ds.sample(weights,m==1?2:1);
                double mean=0;
                for(int i=1;i<N;++i) for(int j=i+1;j<N;++j) mean+=d[i*N+j];
                mean=max(1e-6,mean/((N-1)*(N-2)/2));
                for(double &x:d) x/=mean;
                d=blend(d,rank,.21+.25*discord);
            }
            Tree t=distTree(d,b%3==0);
            if((b%2) || discord>.43) {
                refineDistance(t,d,rng,4,.81f);
                if(discord>.33) pool.add(t,1.6*discord);
            }
            const auto &wq=b%3==1?base:q;
            Quartets qm=wq;
            for(size_t i=0;i<ng;++i) qm.weights[i]*=weights[i];
            refine(t,qm,rng,4,.82f,&d,qScale*(42+105*discord));
            pool.add(t,m%2?(1.38+2.0*discord):wLen);
        }
        if(severe>.06) {
            const Matrix *d[]={&ds.d[29],&rank,&early,&stable,&clock,&ds.d[35]};
            for(int i=0;i<6 && timeLeft();++i) {
                bool useClock=(i==1 && severe>.32);
                int method=useClock?2:i%2;
                Tree t=distTree(*d[i],method,useClock);
                refineDistance(t,*d[i],rng,6,i==1?.68f:.82f);
                double w=severe*(i==0?7.0:i==1?5.2:3.4);
                pool.add(t,w);
                if(i<4) {
                    refine(t,q,rng,4,.83f,d[i],qScale*(90+140*severe));
                    pool.add(t,.76*w);
                }
            }
        }
    }
    Tree center=pool.consensus();
    if((calibrated || highConflict) && timeLeft()) {
        const Matrix *candidate[]={&joint,&dcombo,&early,&lower};
        for(int i=0;i<4 && timeLeft();++i) {
            int cap[]={5,13,22,8};
            double conf=.81+.025*i;
            Tree t=skeleton(center,pool,*candidate[i],cap[i],conf,i%2?.92f:.79f,
                            i==2 && discord>.6?2:i%2);
            if(i!=3) refineDistance(t,*candidate[i],rng,4,.77f);
            refine(t,i%2?fine:q,rng,4,.84f,candidate[i],prior*1.09);
            pool.add(t,1.3+1.5*discord);
        }
        center=pool.consensus();
    }
    if(!timeLeft()) return center;
    SplitPool local;
    local.add(center,1.95);
    PatchModel models[]={
        {&q,&fine,.30,&joint,prior,.94f,.65*qScale,.74,0,0,&ts,.14+.36*discord},
        {&q,&base,.25,&dcombo,prior*.8,.88f,.55*qScale,.68,1,0,&ts,.26+.31*discord},
        {&reps,&fine,.32,&joint,prior*1.1,.98f,.65*qScale,.65,0,.22,&anchored,.19+.27*discord},
        {&fine,&base,.29,&joint,prior*.75,.93f,.60*qScale,.87,0,0,nullptr,0},
        {&base,&q,.38,&joint,prior*1.12,.82f,.70*qScale,.37,0,0,&ts,.37+.35*discord},
    };
    for(auto &model:models)
        model.prior=[&](const Mask &s) { return qScale*1.38*sqrt(pool.support(s)); };
    models[1].dist=spread<2.0?&lower:&dcombo;
    models[3].dist=&dcombo;
    models[4].dist=discord>.48?&early:&joint;
    models[4].distMix*=1.36;
    auto active=center;
    for(int i=0;i<int(size(models)) && timeLeft();++i) {
        Tree t=(i==3||i==4)?active:center;
        int cap=i==4?7:i==1?8:9;
        double z=(i==4?2.9:3.12)+discord*.2;
        SplitWeights post;
        patches(t,models[i],rng,cap,z,i==0?0:.28+i*.14,&post);
        double w=i<=1?2.3:1.8;
        local.add(post,.77*w);
        local.add(t,.23*w);
        if(i==1 || i==3) active=local.consensus();
    }
    SplitPool blended;
    double w=highConflict?min(.86,.58+.12*discord+.13*severe):.54;
    blended.combine(pool,w);
    blended.combine(local,1-w);
    Tree result=blended.consensus();
    return result.valid()?result:center;
}

static Tree inferHeterogeneous(const vector<Gene> &genes,const Distances &ds,mt19937 &rng) {
    Quartets strong(genes,9), q(genes,18), soft(genes,0), branch(genes,8), density(genes,18,0,0,.5);
    auto topo=weightedDistance(ds,{{2,.69},{37,.15},{20,.16}});
    auto robust=weightedDistance(ds,{{2,.38},{34,.44},{35,.18}});
    auto near=weightedDistance(ds,{{29,.66},{2,.17},{35,.17}});
    double n=genes.size()/50.;
    struct Start {
        const Matrix *d;
        const Quartets *q;
        int method;
        float local, mix;
    };
    const Start variants[]={
        {&ds.d[2],&strong,0,0,0}, {&ds.d[2],&branch,0,0,0},
        {&topo,&soft,1,.92f,7}, {&ds.d[34],&q,0,.82f,48},
        {&near,&density,1,.86f,22}, {&robust,&strong,0,.90f,37},
        {&ds.d[2],&soft,0,0,0},
    };
    SplitPool pool;
    Tree special=distTree(ds.d[2],0);
    for(size_t i=0;i<size(variants) && timeLeft();++i) {
        const auto &v=variants[i];
        Tree t=distTree(*v.d,v.method);
        refine(t,*v.q,rng,6,v.local,v.d,v.mix*n);
        pool.add(t,i==3?3.:2.2);
        if(i==6) special=t;
    }
    gamma_distribution<float> gamma(1.45f,1.f/1.45f);
    for(int b=0;b<10 && timeLeft();++b) {
        vector<float> w(genes.size());
        for(float &f:w) f=.12f+.88f*gamma(rng);
        int metric=b%3==0?2:0;
        Tree t=distTree(ds.sample(w,metric),b%2);
        Quartets c=b%3==0?branch:b%3==1?strong:q;
        for(size_t i=0;i<w.size();++i) c.weights[i]*=w[i];
        refine(t,c,rng,4,b%3==0?.91f:0,&robust,n*13);
        pool.add(t,1.13);
    }
    if(pool.total==0) return special;
    auto center=pool.consensus();
    SplitPool local;
    local.add(center,1.6);
    const PatchModel pm[]={
        {&soft,&branch,.35,nullptr,0,.90f,.6*n,.55},
        {&strong,&q,.29,&robust,n*18,.96f,.66*n,.79,1},
        {&q,&strong,.31,&near,n*14,.91f,.56*n,.74},
        {&branch,&soft,.35,&robust,n*12,1.0f,.62*n,.87},
    };
    for(int i=0;i<int(size(pm)) && timeLeft();++i) {
        Tree t=i==0?special:center;
        SplitWeights post;
        int repeat=i==0?2:1;
        for(int j=0;j<repeat && timeLeft();++j) {
            if(j) post.clear();
            patches(t,pm[i],rng,i==0?7:i==2?9:8,i==0?3.1:3.3,.25*j+.12*i,&post);
        }
        local.add(t,i==0?1.7:1.27);
        if(!post.empty()) local.add(post,1.);
    }
    SplitPool result;
    result.combine(pool,.37);
    result.combine(local,.63);
    return result.consensus();
}

static vector<Gene> readGenes(const char *path) {
    ifstream input(path);
    if(!input) throw runtime_error("cannot open input");
    string line;
    vector<Gene> g;
    while(getline(input,line)) if(line.find_first_not_of(" \t\r")!=string::npos)
        g.push_back(NewickReader(std::move(line)).read());
    if(g.empty()) throw runtime_error("empty tree file");
    return g;
}

static Tree solve(const vector<Gene> &genes,const Distances &ds,mt19937 &rng) {
    auto topo=weightedDistance(ds,{{2,.59},{20,.22},{37,.19}});
    Tree guide=distTree(topo,0);
    Quartets q(genes,18);
    auto conf=edgeReliability(guide,q,0);
    vector<double> margins;
    vector<float> rates;
    for(auto [u,v,z,m]:conf) margins.push_back(m);
    for(const auto &g:genes) rates.push_back(g.rate);
    sort(rates.begin(),rates.end());
    size_t n=rates.size();
    if(!timeLeft() || margins.empty()) return guide;
    sort(margins.begin(),margins.end());
    double signal=margins[margins.size()/2], spread=log(rates[n*9/10]/rates[n/10]);
    double resolved=0;
    for(const auto &g:genes) resolved+=g.resolution;
    if(resolved/n<.035) {
        Quartets raw(genes,1);
        Tree t=distTree(ds.d[0],1);
        refine(t,raw,rng,7);
        return t;
    }
    if(signal>=.235 && spread>2.55) return inferHeterogeneous(genes,ds,rng);
    return infer(genes,ds,rng,topo,guide,spread,signal);
}

int main(int argc,char **argv) {
    try {
        deadline=Clock::now()+chrono::seconds(110);
        if(argc!=3) throw runtime_error("usage: solver INPUT OUTPUT");
        auto genes=readGenes(argv[1]);
        Distances ds(genes);
        unsigned seed=1;
        if(const char *s=getenv("SEED")) seed=unsigned(strtoul(s,nullptr,10));
        mt19937 rng(seed);
        Tree t=solve(genes,ds,rng);
        if(!t.valid()) throw runtime_error("invalid inferred topology");
        ofstream out(argv[2]);
        out<<t.newick();
        out.close();
        if(!out) throw runtime_error("cannot write output");
        return 0;
    } catch(const exception &e) {
        cerr<<"solver: "<<e.what()<<'\n'; return 1;
    }
}
