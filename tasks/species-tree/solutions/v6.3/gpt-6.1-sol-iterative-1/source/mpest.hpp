#pragma once

struct RootedTriple {
    uint8_t a,b,c;
    array<float,3> q{};
};

vector<RootedTriple> makeRootedTriples(const vector<Tree>& genes,int mode,double param,double lambda,
                                     bool relative,const vector<GTree>& gs,double deadline) {
    int n=genes[0].n,m=genes.size();
    vector<RootedTriple> data;
    data.reserve(size_t(n-1)*(n-2)*(n-3)/6);
    for(int c=3;c<n;++c)for(int b=2;b<c;++b)for(int a=1;a<b;++a)
        data.push_back({uint8_t(a),uint8_t(b),uint8_t(c),{}});
    vector<double> lut(32769);
    for(int i=0;i<(int)lut.size();++i)lut[i]=-expm1(-i/1024.);
    for(int g=0;g<m;++g) {
        if(elapsed()>deadline)return {};
        const Tree& t=genes[g];
        double sum=0;
        for(int v=0;v<(int)t.adj.size();++v)for(auto e:t.adj[v])if(e.to>v)sum+=e.len;
        double scale=mode==5?param*sum/(2*n-3):param;
        double localLambda=lambda*(relative?(2*n-3)/max(1e-20,sum):1);
        double weight=gs.size()==genes.size()?gs[g].weight:1;
        vector<vector<int>> d(n,vector<int>(n));
        Matrix bl(n,vector<double>(n));
        for(int a=0;a<n;++a) {
            function<void(int,int,int,double)> dfs=[&](int v,int p,int distance,double length) {
                if(v<n) {d[a][v]=distance;bl[a][v]=length;}
                for(auto e:t.adj[v])if(e.to!=p) {
                    double prob=(v<n||e.to<n)?0:retention(e.sup,e.len,mode,scale);
                    int w=lround(-log(max(1e-14,1-prob))*1024);
                    dfs(e.to,v,distance+w,length+e.len);
                }
            };
            dfs(a,-1,0,0);
        }
        for(auto& triple:data) {
            int a=triple.a,b=triple.b,c=triple.c;
            int s[3]={d[0][a]+d[b][c],d[0][b]+d[a][c],d[0][c]+d[a][b]};
            int top=min_element(s,s+3)-s;
            int delta=min((*max_element(s,s+3)-s[top]+1)/2,32768);
            double w=weight*lut[delta];
            if(localLambda!=0) {
                double pendant=top==0?bl[0][a]+bl[b][c]:top==1?bl[0][b]+bl[a][c]:bl[0][c]+bl[a][b];
                w*=exp(-localLambda*pendant);
            }
            triple.q[2-top]+=w;
        }
    }
    cerr<<"Triplet data "<<data.size()<<" time "<<elapsed()<<"\n";
    return data;
}

struct TripleStats {
    int m;
    vector<double> w,n;
    explicit TripleStats(int size):m(size),w(m*m),n(m*m) {}
    void add(int r,int c,double ww,double nn) {w[r*m+c]+=ww;n[r*m+c]+=nn;}
};

array<int,3> triplePosition(int ab,int ac,int bc) {
    if(ac==bc)return {ac,ab,0};
    if(ab==bc)return {ab,ac,1};
    return {ab,bc,2};
}

TripleStats aggregateTriples(const vector<RootedTriple>& data,const Routing& route) {
    int m=route.nn-route.n,n=route.n;
    TripleStats stats(m);
    for(auto& t:data) {
        auto p=triplePosition(route.lc[t.a][t.b],route.lc[t.a][t.c],route.lc[t.b][t.c]);
        stats.add(p[0]-n,p[1]-n,t.q[p[2]],double(t.q[0])+t.q[1]+t.q[2]);
    }
    return stats;
}

double tripleLogLikelihood(double w,double n,double t) {
    if(n<1e-14)return 0;
    double x=exp(-max(0.,t));
    return w*log(3-2*x)-(n-w)*max(0.,t);
}
pair<double,double> tripleDerivatives(double w,double n,double t) {
    double x=exp(-max(0.,t)),den=3-2*x;
    return {2*w*x/den-(n-w),-6*w*x/(den*den)};
}

double fullTripleLikelihood(const TripleStats& s,const vector<double>& h) {
    double ll=0;
    for(int r=0;r<s.m;++r)for(int c=0;c<s.m;++c)
        if(s.n[r*s.m+c]>0)ll+=tripleLogLikelihood(s.w[r*s.m+c],s.n[r*s.m+c],h[r]-h[c]);
    return ll;
}

template<class Gradient>
double boundedHeight(double initial,double lo,double hi,Gradient gradient) {
    if(hi<=lo+1e-10)return lo;
    auto dl=gradient(lo),dh=gradient(hi);
    if(dl.first<=0)return lo;
    if(dh.first>=0)return hi;
    double x=max(lo,min(hi,initial));
    for(int it=0;it<14;++it) {
        auto [d,dd]=gradient(x);
        if(fabs(d)<1e-7)break;
        if(d>0)lo=x;else hi=x;
        double next=x-d/dd;
        if(!(next>lo&&next<hi))next=(lo+hi)*.5;
        if(fabs(next-x)<1e-8) {x=next;break;}
        x=next;
    }
    return x;
}

double fitTripleHeights(const Tree& tree,const Routing& route,const TripleStats& stats,vector<double>& h,
                        int maxPasses,double deadline) {
    int n=tree.n,m=stats.m;
    double previous=fullTripleLikelihood(stats,h);
    auto updateGroup=[&](const vector<int>& group) {
        vector<bool> inside(m);
        for(int i:group)inside[i]=true;
        double lo=0,hi=100,initial=0;
        for(int i:group) {
            int v=i+n;
            initial+=h[i];
            int par=route.parent[v];
            if(par>=n&&!inside[par-n])hi=min(hi,h[par-n]);
            for(auto e:tree.adj[v])if(e.to!=par&&e.to>=n&&!inside[e.to-n])lo=max(lo,h[e.to-n]);
        }
        if(lo>hi)lo=hi;
        auto gradient=[&](double x) {
            double d=0,dd=0;
            for(int i:group)for(int j=0;j<m;++j)if(!inside[j]) {
                int a=i*m+j,b=j*m+i;
                if(stats.n[a]>0) {
                    auto v=tripleDerivatives(stats.w[a],stats.n[a],x-h[j]);d+=v.first;dd+=v.second;
                }
                if(stats.n[b]>0) {
                    auto v=tripleDerivatives(stats.w[b],stats.n[b],h[j]-x);d-=v.first;dd+=v.second;
                }
            }
            return make_pair(d,dd);
        };
        double x=boundedHeight(initial/group.size(),lo,hi,gradient);
        for(int i:group)h[i]=x;
    };
    vector<int> order(m);
    iota(order.begin(),order.end(),0);
    sort(order.begin(),order.end(),[&](int a,int b){return route.depth[a+n]>route.depth[b+n];});
    for(int pass=0;pass<maxPasses && elapsed()<deadline;++pass) {
        if(pass%2)reverse(order.begin(),order.end());
        for(int i:order)updateGroup({i});
        if(pass%3==2) {
            vector<int> parent(m);iota(parent.begin(),parent.end(),0);
            function<int(int)> find=[&](int i){return parent[i]==i?i:parent[i]=find(parent[i]);};
            for(int i=0;i<m;++i) {
                int p=route.parent[n+i];
                if(p>=n&&h[p-n]-h[i]<1e-5)parent[find(i)]=find(p-n);
            }
            vector<vector<int>> groups(m);
            for(int i=0;i<m;++i)groups[find(i)].push_back(i);
            for(auto& group:groups)if(group.size()>1)updateGroup(group);
        }
        double minimum=*min_element(h.begin(),h.end());
        for(double& x:h)x-=minimum;
        double value=fullTripleLikelihood(stats,h);
        if(value<previous-1e-5*max(1.,fabs(previous)))cerr<<"Warning: triplet coordinate decrease\n";
        if(pass>=3&&value-previous<1e-9*max(1.,fabs(value)))return value;
        previous=value;
    }
    return fullTripleLikelihood(stats,h);
}

struct RootedNniEdge {
    int u,v,a,b,c,parent;
    vector<uint8_t> color;
};
vector<RootedNniEdge> rootedNniEdges(const Tree& t,const Routing& route) {
    vector<RootedNniEdge> out;
    for(int u=t.n;u<(int)t.adj.size();++u) {
        int v=route.parent[u];
        if(v<t.n)continue;
        vector<int> child;
        for(auto e:t.adj[u])if(e.to!=v)child.push_back(e.to);
        int c=-1;
        for(auto e:t.adj[v])if(e.to!=u&&e.to!=route.parent[v]&&e.to!=0)c=e.to;
        if(child.size()!=2||c<0)continue;
        vector<int> col(t.n,3);
        colorSub(t,child[0],u,0,col);colorSub(t,child[1],u,1,col);colorSub(t,c,v,2,col);
        vector<uint8_t> color(col.begin(),col.end());
        out.push_back({u,v,child[0],child[1],c,route.parent[v],move(color)});
    }
    return out;
}

struct NniTripleDelta {
    vector<double> w,n;
    explicit NniTripleDelta(int size=0):w(size),n(size) {}
};
int deltaSlot(int r,int c,int u,int v,int m) {
    if(r==u)return c;
    if(r==v)return m+c;
    if(c==u)return 2*m+r;
    if(c==v)return 3*m+r;
    return -1;
}

bool buildTripleDeltas(const vector<RootedTriple>& data,const Routing& route,const vector<RootedNniEdge>& edges,
                      vector<array<NniTripleDelta,2>>& delta,double deadline) {
    int n=route.n,m=route.nn-n;
    vector<vector<int>> incident(m);
    delta.clear();delta.reserve(edges.size());
    for(int e=0;e<(int)edges.size();++e) {
        incident[edges[e].u-n].push_back(e);incident[edges[e].v-n].push_back(e);
        delta.push_back({NniTripleDelta(4*m),NniTripleDelta(4*m)});
    }
    size_t counter=0;
    for(auto& t:data) {
        if(++counter%32768==0 && elapsed()>deadline)return false;
        int ab=route.lc[t.a][t.b],ac=route.lc[t.a][t.c],bc=route.lc[t.b][t.c];
        auto old=triplePosition(ab,ac,bc);
        double total=double(t.q[0])+t.q[1]+t.q[2],ow=t.q[old[2]];
        if(total<=0)continue;
        int candidates[6],count=0;
        for(int e:incident[old[0]-n])candidates[count++]=e;
        for(int e:incident[old[1]-n]) {
            bool duplicate=false;
            for(int k=0;k<count;++k)if(candidates[k]==e)duplicate=true;
            if(!duplicate)candidates[count++]=e;
        }
        for(int k=0;k<count;++k) {
            int eid=candidates[k];
            const auto& e=edges[eid];
            int ca=e.color[t.a],cb=e.color[t.b],cc=e.color[t.c];
            for(int alternative=0;alternative<2;++alternative) {
                auto updated=[&](int x,int y,int lca) {
                    return x<3&&y<3&&x!=y?(x+y==alternative+2?e.u:e.v):lca;
                };
                auto fresh=triplePosition(updated(ca,cb,ab),updated(ca,cc,ac),updated(cb,cc,bc));
                if(fresh==old)continue;
                int u=e.u-n,v=e.v-n;
                int from=deltaSlot(old[0]-n,old[1]-n,u,v,m);
                int to=deltaSlot(fresh[0]-n,fresh[1]-n,u,v,m);
                if(from<0||to<0)throw runtime_error("Triplet delta routing failed");
                auto& d=delta[eid][alternative];
                d.n[from]-=total;d.w[from]-=ow;
                d.n[to]+=total;d.w[to]+=t.q[fresh[2]];
            }
        }
    }
    return true;
}

struct LocalTriplePair {int r,c;double w,n;};
struct TripleMove {double gain=-1e300,hu=0,hv=0;int edge=-1,alternative=0;};

TripleMove scoreTripleMove(const Tree& tree,const TripleStats& base,const vector<double>& h,const RootedNniEdge& edge,
                           const NniTripleDelta& delta,int alternative) {
    int m=base.m,n=tree.n,u=edge.u-n,v=edge.v-n;
    vector<LocalTriplePair> pairs;
    double before=0;
    for(int slot=0;slot<4*m;++slot) {
        int type=slot/m,x=slot%m,r,c;
        if(type==0) {r=u;c=x;}
        else if(type==1) {r=v;c=x;}
        else if(type==2) {r=x;c=u;if(x==u||x==v)continue;}
        else {r=x;c=v;if(x==u||x==v)continue;}
        int id=r*m+c;
        if(base.n[id]>0)before+=tripleLogLikelihood(base.w[id],base.n[id],h[r]-h[c]);
        double nn=max(0.,base.n[id]+delta.n[slot]),ww=max(0.,base.w[id]+delta.w[slot]);
        if(nn>1e-8)pairs.push_back({r,c,min(nn,ww),nn});
    }
    auto age=[&](int a){return a>=n?h[a-n]:0.;};
    int child0=alternative==0?edge.a:edge.b;
    int sibling=alternative==0?edge.b:edge.a;
    double childBound=max(age(child0),age(edge.c)),parentBound=edge.parent>=n?h[edge.parent-n]:100;
    double hu=max(childBound,min(h[u],h[v])),hv=h[v];
    auto optimize=[&](int node,bool both) {
        double lo,hi;
        if(both) {lo=max(childBound,age(sibling));hi=parentBound;}
        else if(node==u) {lo=childBound;hi=hv;}
        else {lo=max(hu,age(sibling));hi=parentBound;}
        auto gradient=[&](double value) {
            double d=0,dd=0;
            for(auto p:pairs) {
                bool inR=both?(p.r==u||p.r==v):p.r==node;
                bool inC=both?(p.c==u||p.c==v):p.c==node;
                if(inR==inC)continue;
                double hr=p.r==u?hu:p.r==v?hv:h[p.r];
                double hc=p.c==u?hu:p.c==v?hv:h[p.c];
                if(inR)hr=value;else hc=value;
                auto g=tripleDerivatives(p.w,p.n,hr-hc);
                d+=(inR?1:-1)*g.first;dd+=g.second;
            }
            return make_pair(d,dd);
        };
        double value=boundedHeight(both?(hu+hv)*.5:node==u?hu:hv,lo,hi,gradient);
        if(both)hu=hv=value;
        else if(node==u)hu=value;
        else hv=value;
    };
    for(int rep=0;rep<5;++rep) {
        optimize(u,false);optimize(v,false);
        if(hv-hu<1e-5)optimize(u,true);
    }
    double after=0;
    for(auto p:pairs) {
        double hr=p.r==u?hu:p.r==v?hv:h[p.r];
        double hc=p.c==u?hu:p.c==v?hv:h[p.c];
        after+=tripleLogLikelihood(p.w,p.n,hr-hc);
    }
    return {after-before,hu,hv,-1,alternative};
}

Tree mpestRefine(const Tree& initial,const vector<RootedTriple>& data,int iterations,double prior,double deadline) {
    Tree t=initial;
    int n=t.n,m=t.adj.size()-n;
    Routing first(t);
    vector<double> heights(m);
    for(int i=0;i<m;++i)heights[i]=-.1*first.depth[n+i];
    double minH=*min_element(heights.begin(),heights.end());
    for(double& x:heights)x-=minH;
    for(int it=0;it<iterations && elapsed()<deadline-2;++it) {
        Routing route(t);
        auto stats=aggregateTriples(data,route);
        double ll=fitTripleHeights(t,route,stats,heights,it==0?60:30,deadline-2);
        auto edges=rootedNniEdges(t,route);
        vector<array<NniTripleDelta,2>> deltas;
        if(!buildTripleDeltas(data,route,edges,deltas,deadline-2))break;
        TripleMove best;
        for(int e=0;e<(int)edges.size()&&elapsed()<deadline-1;++e)for(int alt=0;alt<2;++alt) {
            auto move=scoreTripleMove(t,stats,heights,edges[e],deltas[e][alt],alt);
            if(prior!=0) {
                double oldSize=0,newSize=0;
                for(int c:edges[e].color) {
                    if(c==0||c==1)++oldSize;
                    if(c==alt||c==2)++newSize;
                }
                move.gain+=prior*log((oldSize-1)/(newSize-1));
            }
            if(move.gain>best.gain) {best=move;best.edge=e;}
        }
        cerr<<"MP-EST "<<it<<" likelihood "<<ll<<" gain "<<best.gain<<" time "<<elapsed()<<"\n";
        if(best.edge<0||best.gain<=max(1e-5,1e-9*fabs(ll)))break;
        auto e=edges[best.edge];
        t.swapBranches(e.u,best.alternative==0?e.b:e.a,e.v,e.c);
        heights[e.u-n]=best.hu;heights[e.v-n]=best.hv;
    }
    return t;
}
