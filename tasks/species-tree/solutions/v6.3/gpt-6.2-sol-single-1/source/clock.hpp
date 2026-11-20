// Included by solver.cpp, after the core tree and matrix definitions.
struct ClockMatrices {
    struct RNode { int l,r; Edge le,re; float age; };
    array<Matrix,2> rank;
    array<vector<array<RNode,N-1>>,2> rooted;
    explicit ClockMatrices(const vector<Gene> &genes) : rank{Matrix(N*N),Matrix(N*N)} {
        for(const Gene &g:genes) {
            struct Arc { int to; double length; float sup; };
            array<array<Arc,3>,V> adj{};
            int deg[V]{};
            auto link=[&](int u,int v,double w,float sup) {
                adj[u][deg[u]++]={v,w,sup}; adj[v][deg[v]++]={u,w,sup};
            };
            for(int u=N;u<V-2;++u) {
                auto v=g.node[u];
                link(u,v.l,v.le.len,v.le.sup); link(u,v.r,v.re.len,v.re.sup);
            }
            int base=g.node[V-2].r;
            double outlen=g.node[V-2].le.len+g.node[V-2].re.len;
            link(0,base,outlen,-1);
            struct Down { double mass, s, ss; };
            array<Down,V> down{};
            auto aggregate=[&](auto &&self,int u,int p)->Down {
                if(u<N) {
                    double l=adj[u][0].length;
                    double w=1/(1+pow(l/max(.08,g.rate*.75),2));
                    if(u==0) w*=.6;
                    return down[u]={w,0,0};
                }
                Down d{};
                for(auto [v,l,s]:adj[u]) if(v!=p) {
                    Down x=self(self,v,u);
                    d.mass+=x.mass;
                    d.s+=x.s+x.mass*l;
                    d.ss+=x.ss+x.mass*l*l+2*l*x.s;
                }
                return down[u]=d;
            };
            auto d=aggregate(aggregate,base,0);
            auto out=aggregate(aggregate,0,base);
            double tot=d.mass+out.mass, sum=d.s+out.mass*outlen, sq=d.ss+out.mass*outlen*outlen;
            double best=INF, zBest=0, zOut=0;
            int aa=base, bb=0;
            double abLen=outlen;
            float abSup=-1;
            auto root=[&](auto &&self,int u,int p,double sum,double sq)->void {
                for(auto [v,l,s]:adj[u]) if(v!=p) {
                    const auto &a=down[v];
                    double m=1-2*a.mass/tot, t=a.s+a.mass*l;
                    double cov=(sum*(1-m)-2*t)/tot, vs=max(1e-9,1-m*m);
                    double z=clamp(-cov/vs,0.,l);
                    double val=sq/tot-pow(sum/tot,2)+2*z*cov+z*z*vs;
                    if(v==0) zOut=z;
                    else val+=.0008*g.rate*g.rate;
                    if(val<best) { best=val; aa=u; bb=v; zBest=z; abLen=l; abSup=s; }
                    if(v>=N) self(self,v,u,sum+l*(tot-2*a.mass),sq+2*l*(sum-2*t)+tot*l*l);
                }
            };
            root(root,base,-1,sum,sq);
            for(int rt=0;rt<2;++rt) {
                int a=rt?base:aa,b=rt?0:bb;
                double len=rt?outlen:abLen, z=rt?zOut:zBest;
                struct State { double h, var, edge; vector<int> leaves; };
                array<State,V> st;
                vector<int> nodes;
                auto variance=[](double l) { return .007+.65*l+1.3*l*l; };
                auto inward=[&](auto &&self,int u,int p,double l)->void {
                    auto &s=st[u]; s.edge=l;
                    if(u<N) { s.h=s.var=0; s.leaves={u}; return; }
                    double accum=0, precision=0;
                    for(auto [v,w,sup]:adj[u]) if(v!=p) {
                        self(self,v,u,w);
                        const auto &x=st[v]; double pr=1/(x.var+variance(w));
                        accum+=pr*(x.h+w); precision+=pr;
                        s.leaves.insert(s.leaves.end(),x.leaves.begin(),x.leaves.end());
                    }
                    s.var=1/precision; s.h=accum*s.var;
                    nodes.push_back(u);
                };
                inward(inward,a,b,z);
                inward(inward,b,a,len-z);
                double pa=1/(st[a].var+variance(z)), pb=1/(st[b].var+variance(len-z));
                double height=(pa*(st[a].h+z)+pb*(st[b].h+len-z))/(pa+pb);
                auto outward=[&](auto &&self,int u,int p,double h)->void {
                    if(u<N) return;
                    auto &s=st[u];
                    double l=s.edge;
                    s.h=clamp(s.h+s.var/(s.var+variance(l))*(h-s.h-l),0.,h);
                    for(auto [v,w,sup]:adj[u]) if(v!=p) self(self,v,u,s.h);
                };
                outward(outward,a,b,height);
                outward(outward,b,a,height);
                vector<float> heights;
                for(int u:nodes) heights.push_back(st[u].h);
                sort(heights.begin(),heights.end());
                Matrix &ranks=rank[rt];
                double rate=max(.022,double(g.rate));
                double quality=.53+.47*g.resolution;
                auto pairs=[&](const vector<int> &x,const vector<int> &y,double h) {
                    double r=(lower_bound(heights.begin(),heights.end(),h)-heights.begin())/
                        double(heights.size());
                    double k=2*h/rate;
                    for(int i:x) for(int j:y) {
                        int ij=min(i,j)*N+max(i,j);
                        ranks[ij]+=quality*(.27*k+.73*r);
                    }
                };
                pairs(st[a].leaves,st[b].leaves,height);
                rooted[rt].emplace_back();
                auto &order=rooted[rt].back();
                int next=0;
                auto fill=[&](auto &&self,int u,int p)->int {
                    if(u<N) return u;
                    Arc arcs[2];
                    int i=0;
                    for(auto x:adj[u]) if(x.to!=p) arcs[i++]=x;
                    pairs(st[arcs[0].to].leaves,st[arcs[1].to].leaves,st[u].h);
                    int x=self(self,arcs[0].to,u), y=self(self,arcs[1].to,u);
                    int idx=next++;
                    order[idx]={x,y,{float(arcs[0].length),arcs[0].sup},
                                    {float(arcs[1].length),arcs[1].sup},float(2*st[u].h/rate)};
                    return N+idx;
                };
                int x=fill(fill,a,b), y=fill(fill,b,a);
                float sp=rt?-1:abSup;
                order[next++]={x,y,{float(z),sp},{float(len-z),sp},float(2*height/rate)};
            }
        }
        for(Matrix &m:rank) {
            double high=0,lo=INF;
            for(int a=0;a<N;++a) for(int b=a+1;b<N;++b) {
                double &v=m[a*N+b];
                lo=min(lo,v);
                if(a) high+=v;
            }
            double scale=max(1e-8,high/((N-1)*(N-2)/2)-min(0.,lo));
            for(int a=0;a<N;++a) for(int b=a+1;b<N;++b)
                m[b*N+a]=m[a*N+b]=(m[a*N+b]-min(0.,lo))/scale;
        }
    }
};

struct Triplets {
    struct Step { int l,r; float fl,fr,early; };
    vector<array<Step,N-1>> step;
    explicit Triplets(const ClockMatrices &cm,int root,float tau) {
        for(const auto &g:cm.rooted[root]) {
            step.emplace_back();
            auto f=[](Edge e) {
                return 1-powf(support(e),1.18f)*e.len/(e.len+.0016f);
            };
            for(int i=0;i<N-1;++i) {
                const auto &x=g[i];
                float early=tau>0?.08f+expf(-x.age/tau):1.f;
                step.back()[i]={x.l,x.r,f(x.le),f(x.re),early};
            }
        }
    }
    array<double,3> score(const int *c,const float *w) const {
        struct State { float f[3], pair[3], miss[3]; };
        array<State,V-1> st;
        for(int x=0;x<N;++x) {
            st[x]={};
            if(w[x]) st[x].f[c[x]]=w[x];
        }
        array<double,3> result{};
        // AB|C, AC|B, and BC|A, respectively.
        constexpr int x[]={0,0,1}, y[]={1,2,2}, z[]={2,1,0};
        for(const auto &g:step) {
            float s[3]{};
            for(int i=0;i<N-1;++i) {
                auto e=g[i];
                const auto &a=st[e.l],&b=st[e.r];
                auto &r=st[N+i];
                for(int j=0;j<3;++j) r.f[j]=a.f[j]+b.f[j];
                for(int j=0;j<3;++j) {
                    float v=e.early*(a.f[x[j]]*b.f[y[j]]+a.f[y[j]]*b.f[x[j]]);
                    float h=e.fl*a.miss[j], k=e.fr*b.miss[j];
                    r.pair[j]=a.pair[j]+b.pair[j]+v;
                    r.miss[j]=h+k+v;
                    s[j]+=(a.pair[j]-h)*b.f[z[j]]+(b.pair[j]-k)*a.f[z[j]];
                }
            }
            for(int j=0;j<3;++j) result[j]+=s[j];
        }
        return result;
    }
};
