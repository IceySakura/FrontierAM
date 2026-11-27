#pragma once
#include <fftw3.h>
#include <unordered_map>
#include "fft_support.hpp"
#if __has_include("fft_wisdom.hpp")
#include "fft_wisdom.hpp"
#endif

namespace grid {
using namespace galaxy;
constexpr int max_order=20, max_pitch=20, max_volume=max_order*max_order*max_pitch;
constexpr int tile_width=64;

static int fft_dimension(int order){
    return galaxy_fft::dimension(order);
}

struct Cell {
    int t, level, ix, iy, iz, parity;
    int count=0, biggest=0, child[8];
    int field=-1;
    bool live=false,below=false;
};

struct GridTree {
    Tree &t;
    std::vector<Cell> cell;
    double origin[3],width;
    int root,height=0;

    Node bound(int lo,int hi) const {
        Node a;
        a.lo=lo; a.hi=hi;
        double lo3[3],hi3[3];
        for(int j=0;j<3;++j) lo3[j]=hi3[j]=t.pts[lo].x[j];
        for(int i=lo+1;i<hi;++i) for(int j=0;j<3;++j){
            lo3[j]=std::min(lo3[j],t.pts[i].x[j]);
            hi3[j]=std::max(hi3[j],t.pts[i].x[j]);
        }
        for(int j=0;j<3;++j) a.c[j]=0.5*(lo3[j]+hi3[j]);
        double rr=0.;
        for(int i=lo;i<hi;++i){
            double dx=t.pts[i].x[0]-a.c[0],dy=t.pts[i].x[1]-a.c[1],dz=t.pts[i].x[2]-a.c[2];
            rr=std::max(rr,dx*dx+dy*dy+dz*dz);
        }
        a.radius=std::sqrt(rr);
        return a;
    }
    int router(const int *a, int n) {
        if (n==1) return a[0];
        int l=router(a,n/2),r=router(a+n/2,n-n/2);
        Node b=bound(t.nodes[l].lo,t.nodes[r].hi);
        b.left=l; b.right=r;
        int v=t.nodes.size();
        t.nodes.push_back(b);
        return v;
    }
    int build(int lo,int hi,int level,int ix,int iy,int iz,int parity,int leaf) {
        const int v=cell.size();
        Cell c{};
        c.level=level;c.ix=ix;c.iy=iy;c.iz=iz;c.parity=parity;
        cell.push_back(c);
        height=std::max(height,level);
        if (hi-lo<=leaf || level==29) {
            cell[v].t=t.subdivide(lo,hi,48,0);
            return v;
        }
        const double h=std::ldexp(width,-level);
        const double m[3]={origin[0]+(ix+0.5)*h,origin[1]+(iy+0.5)*h,origin[2]+(iz+0.5)*h};
        int cuts[9];
        cuts[0]=lo; cuts[8]=hi;
        for(int k=0;k<3;++k){
            const int gap=4>>k;
            for (int j=0;j<8;j+=2*gap){
                cuts[j+gap]=std::partition(t.pts.begin()+cuts[j],t.pts.begin()+cuts[j+2*gap],
                    [k,&m](const Particle &p){return p.x[k]<m[k];})-t.pts.begin();
            }
        }
        int children[8],count=0;
        for(int j=0;j<8;++j) if(cuts[j]<cuts[j+1]){
            const int u=build(cuts[j],cuts[j+1],level+1,2*ix+(j>>2),2*iy+((j>>1)&1),
                2*iz+(j&1),j,leaf);
            cell[v].child[count]=u;
            cell[v].biggest=std::max(cell[v].biggest,cuts[j+1]-cuts[j]);
            children[count++]=cell[u].t;
        }
        cell[v].count=count;
        cell[v].t=router(children,count);
        return v;
    }
    int depth(int v) const {
        const Node &a=t.nodes[v];
        return a.leaf()?0:1+std::max(depth(a.left),depth(a.right));
    }
    GridTree(Tree &tree,int leaf):t(tree) {
        double low[3],high[3];
        for(int j=0;j<3;++j)low[j]=high[j]=t.pts[0].x[j];
        for(const Particle &p:t.pts)for(int j=0;j<3;++j){
            low[j]=std::min(low[j],p.x[j]); high[j]=std::max(high[j],p.x[j]);
        }
        const double factor=std::max(1.000002,env("GALAXY_GRID_SCALE",1.000002));
        const double shift=std::clamp(env("GALAXY_GRID_SHIFT",0.),-.99,.99);
        width=std::max({high[0]-low[0],high[1]-low[1],high[2]-low[2],.01})*factor;
        const double phases[3]={1.,-.71,.38};
        for(int j=0;j<3;++j)
            origin[j]=.5*(low[j]+high[j]-width+phases[j]*shift*(width-high[j]+low[j]));
        cell.reserve(2*(tree.n/leaf+1));
        root=build(0,t.n,0,0,0,0,0,leaf);
        t.height=depth(cell[root].t);
        for(int j=0;j<3;++j){
            t.xyz[j].resize(t.n+4); t.acc[j].resize(t.n+4);
        }
        for(int i=0;i<t.n;++i){
            t.order[i]=t.pts[i].original;
            for(int j=0;j<3;++j)t.xyz[j][i]=t.pts[i].x[j];
        }
        std::vector<Particle>().swap(t.pts);
    }
    double len(const Cell &a) const {return std::ldexp(width,-a.level);}
    std::array<double,3> center(const Cell &a) const {
        const double w=len(a);
        return {origin[0]+(a.ix+.5)*w,origin[1]+(a.iy+.5)*w,origin[2]+(a.iz+.5)*w};
    }
};

struct Link {
    int a,b,dx,dy,dz;
};
struct Settings {
    int order;
    double extent, minprod, nearprod;
    int small, leaf;
    bool axial=false;
};
struct GridPlan {
    GridTree &g;
    Plan others;
    std::vector<std::vector<Link>> levels;
    std::vector<Link> deferred;
    std::vector<std::vector<std::pair<int,int>>> point_links;
    int minfft, leaf;
    double minprod, nearprod;
    double defer=env("GALAXY_GRID_DEFER",.42);
    const int gap=std::clamp(int(env("GALAXY_GRID_GAP",1.)),1,3);
    const bool split_weak=env("GALAXY_GRID_SPLIT_WEAK",1.)!=0.;
    const bool axis_refine;
    const double soft_open=env("GALAXY_GRID_SOFT",0.);
    const int order;
    double workspace_mb=256.;
    bool sort_links=env("GALAXY_GRID_SORT",1.)!=0.;
    bool recycle=env("GALAXY_GRID_RECYCLE",1.)!=0.;
    size_t count=0,nfields=0,point_terms=0;
    explicit GridPlan(GridTree &grid,const Algebra &al,const Settings &spec)
        :g(grid),others(grid.t,al,-16.,20,.23),
        levels(grid.height+1), point_links(grid.height+1),
        minfft(spec.small),leaf(spec.leaf),minprod(spec.minprod),nearprod(spec.nearprod),
        axis_refine(env("GALAXY_GRID_AXIS",spec.axial?1.:0.)!=0.),order(spec.order) {}
    void fallback(int a,int b) {
        others.near_pair(a,b);
    }
    void add(int u,int v,int dx,int dy,int dz){
        Cell &a=g.cell[u],&b=g.cell[v];
        levels[a.level].push_back({u,v,dx,dy,dz});
        a.live=b.live=true;
        ++count;
    }
    bool grid_point(int u,int v,bool sep){
        Cell *a=&g.cell[u],*b=&g.cell[v];
        if(g.t.nodes[a->t].size()<g.t.nodes[b->t].size()){
            std::swap(a,b);std::swap(u,v);
        }
        const Node &na=g.t.nodes[a->t], &nb=g.t.nodes[b->t];
        if (na.size() < 1.24*order*order*((order+3)&-4) || nb.size()>160)return false;
        if(!sep){
            const auto c=g.center(*a);
            double distance2=0.;
            for(int j=0;j<3;++j)distance2+=(c[j]-nb.c[j])*(c[j]-nb.c[j]);
            const double w=g.len(*a);
            if (std::sqrt(distance2)-nb.radius<1.53*w)return false;
        }
        point_links[a->level].push_back({u,v});
        a->live=true;
        point_terms+=nb.size()*size_t(order*order*((order+3)&-4));
        return true;
    }
    void uneven(int a,int b){
        if(grid_point(a,b,false))return;
        const Cell &x=g.cell[a],&y=g.cell[b];
        if(!env_unbal) {
            fallback(x.t,y.t);
            return;
        }
        const Node &na=g.t.nodes[x.t],&nb=g.t.nodes[y.t];
        if (std::max(na.size(),nb.size()) < 2.0*order*order*((order+3)&-4) ||
                (!x.count && !y.count)){
            fallback(x.t,y.t); return;
        }
        if (x.count && (!y.count || na.size()>nb.size())) {
            for(int j=0;j<x.count;++j)uneven(x.child[j],b);
        } else {
            for(int j=0;j<y.count;++j)uneven(a,y.child[j]);
        }
    }
    const bool env_unbal=env("GALAXY_GRID_UNEVEN",1.)!=0.;
    const bool unsplit=env("GALAXY_GRID_UNSPLIT",1.)!=0.;
    void plan(int u,int v) {
        if(others.stopped)return;
        Cell &a=g.cell[u],&b=g.cell[v];
        const int na=g.t.nodes[a.t].size(),nb=g.t.nodes[b.t].size();
        const bool unable=unsplit && double(a.biggest)*b.biggest<defer*minprod &&
            std::max(a.biggest,b.biggest)<1.24*order*order*((order+3)&-4);
        if(u==v){
            if(na<=leaf || !a.count || unable){
                fallback(a.t,b.t);
            } else {
                for(int i=0;i<a.count;++i)
                    for(int j=i;j<a.count;++j)plan(a.child[i],a.child[j]);
            }
            return;
        }
        const int dx=a.ix-b.ix,dy=a.iy-b.iy,dz=a.iz-b.iz;
        const int d=std::max({std::abs(dx),std::abs(dy),std::abs(dz)});
        const double work=double(na)*nb;
        const int u0=std::max(0,std::abs(dx)-1),u1=std::max(0,std::abs(dy)-1),
                  u2=std::max(0,std::abs(dz)-1);
        const int gap2=u0*u0+u1*u1+u2*u2;
        const double side=g.len(a);
        const bool smooth=soft_open>0. && d<=1 && side*soft_open<=.01;
        const bool axial=axis_refine && d==2 &&
            std::abs(dx)+std::abs(dy)+std::abs(dz)==2 && side>.011;
        if((gap2>=gap && !axial) || smooth){
            if(std::min(na,nb)>=minfft && work>=minprod){
                add(u,v,dx,dy,dz);
            } else if (std::min(na,nb)>=8 && work>=defer*minprod) {
                deferred.push_back({u,v,dx,dy,dz});
            } else if(!grid_point(u,v,d>=2)) fallback(a.t,b.t);
        } else if(d>=2 && !split_weak){
            if(!grid_point(u,v,true))fallback(a.t,b.t);
        } else if(!a.count || !b.count || work<=nearprod || unable ||
                   (std::max(na,nb)<=leaf && work<=1.5*nearprod)){
            uneven(u,v);
        } else {
            for(int i=0;i<a.count;++i)for(int j=0;j<b.count;++j)plan(a.child[i],b.child[j]);
        }
    }
    void complete(){
        for(const Link &e:deferred){
            const Cell &a=g.cell[e.a],&b=g.cell[e.b];
            if(a.live && b.live)add(e.a,e.b,e.dx,e.dy,e.dz);
            else if(!grid_point(e.a,e.b,true)) fallback(a.t,b.t);
        }
        std::vector<Link>().swap(deferred);
    }
    bool mark(int v) {
        Cell &a=g.cell[v];
        bool yes=false;
        for(int i=0;i<a.count;++i)yes|=mark(a.child[i]);
        a.below=yes;
        if(a.live)a.field=nfields++;
        return yes||a.live;
    }
    void mark_again() {
        nfields=0;
        for(Cell &a:g.cell){a.live=a.below=false;a.field=-1;}
        for(const auto &level:levels)for(const Link &q:level){
            g.cell[q.a].live=g.cell[q.b].live=true;
        }
        for(const auto &level:point_links)for(auto q:level)g.cell[q.first].live=true;
        mark(g.root);
    }
    void move_close_to(GridPlan &high,double minwork) {
        for(size_t j=0;j<levels.size();++j){
            auto &lev=levels[j];
            size_t dst=0;
            for(const Link &q:lev){
                const int dx=std::max(0,std::abs(q.dx)-1),
                          dy=std::max(0,std::abs(q.dy)-1),
                          dz=std::max(0,std::abs(q.dz)-1);
                const bool close=dx*dx+dy*dy+dz*dz<=1;
                const double pairs=double(g.t.nodes[g.cell[q.a].t].size())*
                                          g.t.nodes[g.cell[q.b].t].size();
                if(close&&pairs>=minwork){
                    high.levels[j].push_back(q);
                    --count;++high.count;
                } else lev[dst++]=q;
            }
            lev.resize(dst);
        }
    }
    double refine(double seconds) {
        if(seconds<=0.)return 0.;
        struct Weak {
            double value, work;
            int level,index;
        };
        std::vector<Weak> cand;
        std::vector<double> norms(g.cell.size(),0.);
        const int root=g.cell[g.root].t;
        auto inv_norm=[&](int v){
            double &e=norms[v];
            if(e==0.){
                double f[5]{};
                g.t.approx_norm(g.cell[v].t,root,f);
                e=1./std::max(std::hypot(f[0],std::hypot(f[1],f[2]))+.14*f[4],.022*f[3]);
            }
            return e;
        };
        const double nfft=fft_dimension(order);
        const double saving=.52e-9*nfft*nfft*(int(nfft)/2+1)*2;
        std::vector<std::vector<char>> remove(levels.size());
        for(size_t l=0;l<levels.size();++l){
            const auto &v=levels[l];
            remove[l].resize(v.size());
            for(size_t i=0;i<v.size();++i){
                const Link &k=v[i];
                const int ax=std::abs(k.dx),ay=std::abs(k.dy),az=std::abs(k.dz);
                if(std::max({ax,ay,az})!=2)continue;
                const int d2=ax*ax+ay*ay+az*az;
                if(d2>6)continue;
                const Node &a=g.t.nodes[g.cell[k.a].t],&b=g.t.nodes[g.cell[k.b].t];
                const double work=std::max(.35e-6,2.38e-9*a.size()*b.size()-saving);
                if(work>seconds)continue;
                const double h=g.len(g.cell[k.a]);
                const double q=d2*h*h+eps2;
                const double wa=inv_norm(k.a),wb=inv_norm(k.b);
                const double intensity=d2==4?1.:d2==5?.20:.065;
                const double quality=intensity*(a.size()*wb*wb+b.size()*wa*wa)/
                        (q*q*(1.+4.*eps2/(h*h)));
                cand.push_back({quality*a.size()*b.size()/work,work,int(l),int(i)});
            }
        }
        std::sort(cand.begin(),cand.end(),[](const Weak &a,const Weak &b){return a.value>b.value;});
        double extra=0.;
        for(const Weak &w:cand)if(extra+w.work<=seconds){
            const Link &q=levels[w.level][w.index];
            if(others.direct.size()>=16999999 ||
               others.p2pcost+double(g.t.nodes[g.cell[q.a].t].size())*
                      g.t.nodes[g.cell[q.b].t].size()>=others.limit_pairs)break;
            fallback(g.cell[q.a].t,g.cell[q.b].t);
            remove[w.level][w.index]=1;
            extra+=w.work;
            --count;
        }
        for(size_t l=0;l<levels.size();++l){
            auto &v=levels[l];
            size_t k=0;
            for(size_t i=0;i<v.size();++i)if(!remove[l][i])v[k++]=v[i];
            v.resize(k);
        }
        mark_again();
        return extra;
    }
    double estimate(double observed_fft=0.,double near_time=0.,double cross_time=0.) {
        if(others.stopped)return 1e100;
        const int d=fft_dimension(order);
        const size_t vol=order*order*((order+3)&-4);
        const size_t freq=(size_t(d)*d*(d/2+1)*2+tile_width-1)&-size_t(tile_width);
        std::vector<int> used(g.cell.size(),-1), fcounts(levels.size()), mcounts(levels.size());
        int maxfield=0;
        size_t offsets=0,maxoffset=0;
        std::vector<int> first,last;
        if(recycle && g.t.n>1048576){
            first.resize(g.cell.size(),-1);
            last.resize(g.cell.size(),-1);
            if(sort_links)for(auto &v:levels)
                if(!std::is_sorted(v.begin(),v.end(),link_order))
                    std::sort(v.begin(),v.end(),link_order);
        }
        for(size_t lv=0;lv<levels.size();++lv){
            bool pattern[11][11][11]{};
            size_t noff=0;
            for(const Link &l:levels[lv]){
                if(!pattern[5+l.dx][5+l.dy][5+l.dz]){
                    ++noff;
                    pattern[5+l.dx][5+l.dy][5+l.dz]=true;
                }
                for(int v:{l.a,l.b})if(used[v]!=int(lv)){
                    ++fcounts[lv];
                    used[v]=lv;
                }
            }
            offsets+=noff;
            maxoffset=std::max(maxoffset,noff);
            mcounts[lv]=fcounts[lv];
            for(auto q:point_links[lv])if(used[q.first]!=int(lv)){
                ++mcounts[lv];
                used[q.first]=lv;
            }
            int work_fields=mcounts[lv];
            if(!first.empty() && work_fields>1200){
                const auto &v=levels[lv];
                std::vector<int> idx;
                idx.reserve(fcounts[lv]);
                for(size_t i=0;i<v.size();++i)for(int u:{v[i].a,v[i].b}){
                    if(first[u]<0){
                        first[u]=i;
                        idx.push_back(u);
                    }
                    last[u]=i;
                }
                std::vector<int> events(v.size()+1);
                for(int u:idx){
                    events[first[u]]++;
                    events[last[u]]--;
                }
                int nb=0,peak=0;
                for(int a:events){
                    nb+=a;
                    peak=std::max(peak,nb);
                }
                work_fields=std::min(work_fields,peak+int(point_links[lv].size())+8);
            }
            maxfield=std::max(maxfield,work_fields);
        }
        const size_t base = (nfields+maxfield)*vol*8 +
            g.cell.capacity()*sizeof(Cell)+g.t.nodes.capacity()*sizeof(Node)+
            size_t(g.t.n)*52+others.direct.capacity()*sizeof(Pair)+
            count*40+freq*(maxoffset+90)*8;
        const double base_mb=base/double(1<<20);
        if(base_mb>1575.)return 1e100;
        workspace_mb=std::min(310.,1670.-base_mb);
        workspace_mb=env("GALAXY_GRID_MB",workspace_mb);
        const double nb=std::max(32.,workspace_mb*(1<<20)/(16.*freq));
        double nfft=0;
        for(int m:fcounts)
            nfft+=m*(m+4<nb?1.:std::min(1.75,1.16+.065*std::log1p(m/nb)));
        if(g.t.n>1048576){
            const int capacity=int(nb);
            std::vector<int> seen(g.cell.size());
            int batch=1;
            size_t exact=0;
            for(size_t lv=0;lv<levels.size();++lv){
                auto &edges=levels[lv];
                if(fcounts[lv]+2<capacity){
                    exact+=fcounts[lv];
                    continue;
                }
                if(sort_links && !std::is_sorted(edges.begin(),edges.end(),link_order))
                    std::sort(edges.begin(),edges.end(),link_order);
                int used=0;
                ++batch;
                for(const Link &e:edges){
                    if(used+2>=capacity){
                        exact+=used;
                        used=0;
                        ++batch;
                    }
                    for(int v:{e.a,e.b})if(seen[v]!=batch){
                        seen[v]=batch;
                        ++used;
                    }
                }
                exact+=used;
            }
            nfft=std::max(nfft,double(exact));
        }
        const double fft_ns=d<=16?16.8:d==18?23.5:d<=21?18.5:d<=27?22.:
                             d==30?22.0:d==32?20.0:25.0;
        const double near_ns=near_time>0.?near_time:2.33e-9;
        const double cross_ns=cross_time>0.?cross_time:.665e-9;
        const double basis_ns=near_time>0. && g.t.n>=1048576?.775e-9:
            order>=13?.89e-9:.98e-9;
        return near_ns*others.p2pcost+50e-9*others.direct.size()+
            2.25e-9*point_terms+cross_ns*freq*count+
            (observed_fft>0.?observed_fft:1e-9*fft_ns*d*d*d)*nfft+
            basis_ns*vol*g.t.n+5.05e-9*freq*offsets+
            28e-9*g.t.n;
    }
    static bool link_order(const Link &p,const Link &q) {
        return p.a<q.a||(p.a==q.a&&p.b<q.b);
    }
};

static double time_near(GridPlan &p){
    const size_t n=p.others.direct.size();
    const int groups=13;
    if(n<groups*20)return 2.33e-9;
    Tree &t=p.g.t;
    Direct d(t);
    d.version=8;
    double work=0.;
    const double amount=t.n>=1048576?9.0e6:1.8e6;
    std::vector<const Pair*> done;
    done.reserve(4096);
    const double start=cpu_time();
    for(int j=0;j<groups;++j){
        size_t i=(size_t(2*j+1)*n)/(2*groups);
        double local=0.;
        for(size_t step=0;step<1024 && i<n && local<amount/groups;++step,++i){
            const Pair &pair=p.others.direct[i];
            const Node &a=t.nodes[pair.a],&b=t.nodes[pair.b];
            local+=pair.a==pair.b?.5*a.size()*(a.size()-1.):double(a.size())*b.size();
            d.pair(a,b,pair.a==pair.b);
            done.push_back(&pair);
        }
        work+=local;
    }
    const double elapsed=cpu_time()-start;
    for(const Pair *p:done)for(int u:{p->a,p->b}){
        const Node &a=t.nodes[u];
        for(int k=0;k<3;++k)std::fill_n(t.acc[k].data()+a.lo,a.size(),0.);
    }
    return work>4.e5?std::clamp(1.052*elapsed/work,2.16e-9,2.72e-9):2.33e-9;
}

class FFT {
    double *real, *fourier;
    fftw_plan fwd,bwd;
    std::unique_ptr<galaxy_fft::Pruned> pruned;
public:
    int n, c, nv, nf, nvpad, nt;
    explicit FFT(int p) {
        n=fft_dimension(p);
        c=n/2+1;
        nv=n*n*n;
        nf=n*n*c*2;
        nvpad=(nf+tile_width-1)&-tile_width;
        nt=nvpad/tile_width;
        real=fftw_alloc_real(nv);
        fourier=fftw_alloc_real(nvpad);
        if(!real || !fourier){
            fftw_free(real);
            fftw_free(fourier);
            throw std::bad_alloc();
        }
        fwd=fftw_plan_dft_r2c_3d(n,n,n,real,(fftw_complex*)fourier,FFTW_ESTIMATE);
        bwd=fftw_plan_dft_c2r_3d(n,n,n,(fftw_complex*)fourier,real,FFTW_ESTIMATE);
        if(!fwd || !bwd){
            if(fwd)fftw_destroy_plan(fwd);
            if(bwd)fftw_destroy_plan(bwd);
            fftw_free(real);
            fftw_free(fourier);
            throw std::bad_alloc();
        }
        if(env("GALAXY_PRUNED",1.)){
            pruned=std::make_unique<galaxy_fft::Pruned>(p,n,real,(fftw_complex*)fourier,FFTW_ESTIMATE);
            if(!pruned->okay())pruned.reset();
        }
    }
    ~FFT(){
        pruned.reset();
        fftw_destroy_plan(fwd); fftw_destroy_plan(bwd);
        fftw_free(real); fftw_free(fourier);
    }
    void expand(const double *a, double *r, int p,int stride) {
        std::fill_n(r,nv,0.);
        for(int z=0;z<p;++z)for(int y=0;y<p;++y)
            std::copy_n(a+(z*p+y)*stride,p,r+(z*n+y)*n);
    }
    static void pack(const double *src,double *dst){
        for(int i=0;i<tile_width/2;++i){
            dst[i]=src[2*i];
            dst[i+tile_width/2]=src[2*i+1];
        }
    }
    static void unpack(const double *src,double *dst){
        for(int i=0;i<tile_width/2;++i){
            dst[2*i]=src[i];
            dst[2*i+1]=src[i+tile_width/2];
        }
    }
    void spec(const double *a,int p,int stride,double *o,int no) {
        if(pruned){
            for(int j=0;j<p*p;++j){
                std::copy_n(a+j*stride,p,real+j*n);
                std::fill_n(real+j*n+p,n-p,0.);
            }
            for(int z=0;z<p;++z)
                std::fill_n(fourier+2*(z*n+p)*c,2*(n-p)*c,0.);
            std::fill_n(fourier+2*p*n*c,2*(n-p)*n*c,0.);
            pruned->forward();
        } else {
            expand(a,real,p,stride);
            fftw_execute(fwd);
        }
        std::fill(fourier+nf,fourier+nvpad,0.);
        for(int i=0;i<nt;++i)pack(fourier+i*tile_width,o+i*no*tile_width);
    }
    void poly(const double *a,int no,double *l,int p,int stride) {
        for(int i=0;i<nt;++i)unpack(a+i*no*tile_width,fourier+i*tile_width);
        if(pruned)pruned->backward();
        else fftw_execute(bwd);
        const int pitch=pruned?p:n;
        for(int z=0;z<p;++z)for(int y=0;y<p;++y)for(int x=0;x<p;++x)
            l[(z*p+y)*stride+x]+=real[(z*pitch+y)*n+x];
    }
    void kernel(double *dst,int p,double scale,double h,int dx,int dy,int dz) {
        const double step=scale*h/(p-1);
        const double df[3]={dx*h,dy*h,dz*h};
        auto rv=[&](int k){return k<p?k:k-n;};
        double squares[3][40];
        for(int j=0;j<3;++j)for(int i=0;i<n;++i){
            const double r=df[j]+step*rv(i);
            squares[j][i]=r*r;
        }
        const double factor=1./nv;
        const double level0=env("GALAXY_GAUGE",1.)?
            factor/std::sqrt(df[0]*df[0]+df[1]*df[1]+df[2]*df[2]+eps2):0.;
        std::fill_n(real,nv,0.);
        for(int iz=0;iz<n;++iz)if(std::abs(rv(iz))<p)
            for(int iy=0;iy<n;++iy)if(std::abs(rv(iy))<p){
                double *row=real+(iz*n+iy)*n;
                const double q=eps2+squares[1][iy]+squares[2][iz];
                for(int ix=0;ix<p;++ix)row[ix]=factor/std::sqrt(squares[0][ix]+q)-level0;
                for(int ix=n+1-p;ix<n;++ix)row[ix]=factor/std::sqrt(squares[0][ix]+q)-level0;
            }
        fftw_execute(fwd);
        std::copy_n(fourier,nf,dst);
        std::fill(dst+nf,dst+nvpad,0.);
    }
};

static double time_fft(int p){
    FFT ft(p);
    const int stride=(p+3)&-4;
    const int volume=p*p*stride;
    const int batch=36;
    std::vector<double> m(volume,0.161),l(volume);
    std::vector<double> f(size_t(batch)*ft.nvpad);
    ft.spec(m.data(),p,stride,f.data(),batch);
    ft.poly(f.data(),batch,l.data(),p,stride);
    double begin=cpu_time();
    for(int j=0;j<batch;++j)ft.spec(m.data(),p,stride,f.data()+j*tile_width,batch);
    for(int j=0;j<batch;++j)ft.poly(f.data()+j*tile_width,batch,l.data(),p,stride);
    return (cpu_time()-begin)/batch;
}

struct Poly {
    int p, sx, volume;
    double extent,den[max_order];
    double u[2][2][max_order][max_pitch]{};
    explicit Poly(int order,double extent):p(order),sx((p+3)&-4),volume(p*p*sx),extent(extent){
        for(int i=0;i<p;++i){
            den[i]=1.;
            for(int j=0;j<p;++j)if(i!=j)den[i]/=(i-j);
        }
        for(int side=0;side<2;++side)for(int i=0;i<p;++i){
            double v=(side?0.25:-0.25)*(p-1)/extent+.5*(i-.5*(p-1));
            basis(v,u[0][side][i],nullptr);
            for(int j=0;j<p;++j)u[1][side][j][i]=u[0][side][i][j];
        }
    }
    void basis(double v,double *w,double *d) const {
        double f[max_order+1],fd[max_order+1],q[max_order];
        f[0]=1.;fd[0]=0.;
        for(int i=0;i<p;++i){
            const double a=v+(.5*(p-1)-i);
            q[i]=a;
            if(d)fd[i+1]=a*fd[i]+f[i];
            f[i+1]=a*f[i];
        }
        double b=1.,bd=0.;
        for(int i=p-1;i>=0;--i){
            w[i]=(b*f[i])*den[i];
            if(d){d[i]=(bd*f[i]+b*fd[i])*den[i];bd=q[i]*bd+b;}
            b*=q[i];
        }
        for(int i=p;i<sx;++i){
            w[i]=0.;
            if(d)d[i]=0.;
        }
    }
    void p2m(const GridTree &g,int v,double *m,int lo,int hi) const {
        const Cell &c=g.cell[v];
        const auto ctr=g.center(c);
        const double h=(p-1)/(extent*g.len(c));
        double x[max_pitch],y[max_pitch],z[max_pitch];
        for(int i=lo;i<hi;++i){
            basis((g.t.xyz[0][i]-ctr[0])*h,x,nullptr);
            basis((g.t.xyz[1][i]-ctr[1])*h,y,nullptr);
            basis((g.t.xyz[2][i]-ctr[2])*h,z,nullptr);
            for(int k=0;k<p;++k)for(int j=0;j<p;++j){
                const V v=splat(z[k]*y[j]);
                double *r=m+(k*p+j)*sx;
                store(r,fmadd(v,load(x),load(r)));
                store(r+4,fmadd(v,load(x+4),load(r+4)));
                if(sx>=12)store(r+8,fmadd(v,load(x+8),load(r+8)));
                if(sx>=16)store(r+12,fmadd(v,load(x+12),load(r+12)));
                if(sx>=20)store(r+16,fmadd(v,load(x+16),load(r+16)));
            }
        }
    }
    void l2p(GridTree &g,int v,const double *l,int lo,int hi) const {
        const Cell &c=g.cell[v];
        const auto ctr=g.center(c);
        const double h=(p-1)/(extent*g.len(c));
        double x[max_pitch],y[max_pitch],z[max_pitch],dx[max_pitch],dy[max_pitch],dz[max_pitch];
        for(int i=lo;i<hi;++i){
            basis((g.t.xyz[0][i]-ctr[0])*h,x,dx);
            basis((g.t.xyz[1][i]-ctr[1])*h,y,dy);
            basis((g.t.xyz[2][i]-ctr[2])*h,z,dz);
            V ax=zero(),ay=zero(),az=zero();
            for(int k=0;k<p;++k)for(int j=0;j<p;++j){
                const double *r=l+(k*p+j)*sx;
                V u=zero(),v=zero();
                for(int b=0;b<sx;b+=4){
                    V l=load(r+b);
                    u=fmadd(l,load(x+b),u);
                    v=fmadd(l,load(dx+b),v);
                }
                ax=fmadd(v,splat(z[k]*y[j]),ax);
                ay=fmadd(u,splat(z[k]*dy[j]),ay);
                az=fmadd(u,splat(dz[k]*y[j]),az);
            }
            g.t.acc[0][i]+=h*sum(ax); g.t.acc[1][i]+=h*sum(ay); g.t.acc[2][i]+=h*sum(az);
        }
    }
    void shift(const double *a,double *b,int parity,bool up,double *tmp,double *tmp2) const {
        const auto &mx=u[up?0:1][parity>>2];
        const auto &my=u[up?1:0][(parity>>1)&1];
        const auto &mz=u[up?1:0][parity&1];
        for(int r=0;r<p*p;++r){
            const double *f=a+r*sx;
            double *d=tmp+r*sx;
            for(int j=0;j<sx;j+=4){
                V s=zero();
                for(int k=0;k<p;++k)s=fmadd(splat(f[k]),load(mx[k]+j),s);
                store(d+j,s);
            }
        }
        for(int z=0;z<p;++z)for(int y=0;y<p;++y)for(int x=0;x<sx;x+=4){
            V s=zero();
            for(int k=0;k<p;++k)s=fmadd(splat(my[y][k]),load(tmp+(z*p+k)*sx+x),s);
            store(tmp2+(z*p+y)*sx+x,s);
        }
        for(int z=0;z<p;++z)for(int j=0;j<p*sx;j+=4){
            V s=load(b+z*p*sx+j);
            for(int k=0;k<p;++k)s=fmadd(splat(mz[z][k]),load(tmp2+k*p*sx+j),s);
            store(b+z*p*sx+j,s);
        }
    }
};

struct Engine {
    GridPlan &plan;
    GridTree &g;
    Poly poly;
    FFT fft;
    std::vector<double> m,scratch,tmp1,tmp2;
    int width;
    int moment_limit=env("GALAXY_GRID_MOMENT",160.);
    bool trim=env("GALAXY_GRID_TRIM",1.)!=0.;
    Engine(GridPlan &p,int order,double extent)
        :plan(p),g(p.g),poly(order,extent),fft(order),
         m(size_t(p.nfields)*poly.volume),scratch(size_t(g.height+2)*poly.volume),
         tmp1(poly.volume),tmp2(poly.volume),width(poly.volume){}
    void upward(int v,double *dst) {
        const Cell &c=g.cell[v];
        if(!dst && !c.live && !c.below)return;
        double *buf=c.live?m.data()+c.field*size_t(width):
            scratch.data()+c.level*size_t(width);
        const bool need=dst||c.live;
        if(need)std::fill_n(buf,width,0.);
        const Node &n=g.t.nodes[c.t];
        if(!c.count || (need && ((trim&&!c.below)||n.size()<=moment_limit))) {
            if(need)poly.p2m(g,v,buf,n.lo,n.hi);
            if(c.below)for(int i=0;i<c.count;++i)upward(c.child[i],nullptr);
        } else for(int i=0;i<c.count;++i){
            const int k=c.child[i];
            const Cell &u=g.cell[k];
            if(need && trim && !u.live && !u.below){
                const Node &a=g.t.nodes[u.t];
                poly.p2m(g,v,buf,a.lo,a.hi);
            } else upward(k,need?buf:nullptr);
        }
        if(dst)poly.shift(buf,dst,c.parity,true,tmp1.data(),tmp2.data());
    }
    void downward(int v,const double *parent) {
        const Cell &c=g.cell[v];
        if(!parent && !c.live && !c.below)return;
        double *buf=c.live?m.data()+c.field*size_t(width):scratch.data()+c.level*size_t(width);
        if(parent){
            if(!c.live)std::fill_n(buf,width,0.);
            poly.shift(parent,buf,c.parity,false,tmp1.data(),tmp2.data());
        }
        const bool need=parent||c.live;
        if(!c.count || !c.below){
            if(need){
                const Node &a=g.t.nodes[c.t];
                poly.l2p(g,v,buf,a.lo,a.hi);
            }
        } else for(int j=0;j<c.count;++j){
            const int k=c.child[j];
            const Cell &u=g.cell[k];
            if(need && trim && !u.live && !u.below){
                const Node &a=g.t.nodes[u.t];
                poly.l2p(g,v,buf,a.lo,a.hi);
            } else downward(k,need?buf:nullptr);
        }
    }
    static int key(int dx,int dy,int dz) {
        return (dx+16)+(dy+16)*33+(dz+16)*1089;
    }
    static void translate_pair(double *a,double *b,const double *sa,const double *sb,
                               const double *ker) {
        constexpr int e=tile_width/2;
        for(int i=0;i<e;i+=4){
            const V kr=load(ker+i),ki=load(ker+e+i);
            const V ar=load(sa+i),ai=load(sa+e+i),br=load(sb+i),bi=load(sb+e+i);
            store(a+i,fnmadd(bi,ki,fmadd(br,kr,load(a+i))));
            store(a+e+i,fmadd(br,ki,fmadd(bi,kr,load(a+e+i))));
            store(b+i,fmadd(ai,ki,fmadd(ar,kr,load(b+i))));
            store(b+e+i,fnmadd(ar,ki,fmadd(ai,kr,load(b+e+i))));
        }
    }
    static double time_translations(){
        constexpr int n=384,nk=60,trials=12,counts=8297;
        std::vector<double> src(n*tile_width,.03),local(n*tile_width),ker(nk*tile_width,.028);
        int a[counts],b[counts],k[counts];
        uint32_t state=479393;
        auto rnd=[&](){state=1664525*state+1013904223;return state>>9;};
        for(int i=0;i<counts;++i){
            a[i]=((i/24)%n)*tile_width;
            b[i]=((i/24+1+rnd()%57)%n)*tile_width;
            k[i]=(rnd()%nk)*tile_width;
        }
        const double start=cpu_time();
        for(int j=0;j<trials;++j)for(int i=0;i<counts;++i){
            translate_pair(local.data()+a[i],local.data()+b[i],src.data()+a[i],src.data()+b[i],ker.data()+k[i]);
        }
        asm volatile("" : : "g"(local.data()) : "memory");
        const double spent=cpu_time()-start;
        return std::clamp(1.10*spent/(trials*counts*tile_width),.60e-9,.78e-9);
    }
    struct Neighbour { int src,k; int64_t flip; };
    static void translate_row(double *out,const double *src,const double *kern,
                              const Neighbour *row,int count){
        constexpr int e=tile_width/2;
        for(int u=0;u<e;u+=16){
            V ar[4]={zero(),zero(),zero(),zero()};
            V ai[4]={zero(),zero(),zero(),zero()};
            for(int j=0;j<count;++j){
                const Neighbour &p=row[j];
                const double *h=kern+p.k+u,*q=src+p.src+u;
                const V sign=_mm256_castsi256_pd(_mm256_set1_epi64x(p.flip));
                #pragma GCC unroll 4
                for(int k=0;k<4;++k){
                    const int l=k*4;
                    const V kr=load(h+l),ki=_mm256_xor_pd(load(h+e+l),sign);
                    const V br=load(q+l),bi=load(q+e+l);
                    ar[k]=fnmadd(bi,ki,fmadd(br,kr,ar[k]));
                    ai[k]=fmadd(br,ki,fmadd(bi,kr,ai[k]));
                }
            }
            for(int k=0;k<4;++k){
                store(out+u+4*k,ar[k]);
                store(out+u+e+4*k,ai[k]);
            }
        }
    }
    void to_points(int a,int b,double *l) const {
        const Cell &c=g.cell[a];
        const Node &t=g.t.nodes[g.cell[b].t];
        const double *m0=m.data()+size_t(c.field)*width;
        const auto ac=g.center(c);
        const double h=g.len(c)*poly.extent/(poly.p-1);
        double pxyz[3][max_pitch];
        for(int j=0;j<3;++j)for(int k=0;k<poly.sx;++k)
            pxyz[j][k]=ac[j]+(k-.5*(poly.p-1))*h;
        for(int i=t.lo;i<t.hi;++i){
            const V ix=splat(g.t.xyz[0][i]);
            const double y=g.t.xyz[1][i],z=g.t.xyz[2][i];
            V tx=zero(),ty=zero(),tz=zero();
            for(int k=0;k<poly.p;++k) {
                const V dz=splat(pxyz[2][k]-z);
                for(int j=0;j<poly.p;++j){
                    const V dy=splat(pxyz[1][j]-y);
                    const V yzs=fmadd(dy,dy,fmadd(dz,dz,splat(eps2)));
                    const int row=(k*poly.p+j)*poly.sx;
                    for(int u=0;u<poly.sx;u+=4){
                        const V dx=sub(load(pxyz[0]+u),ix);
                        const V q=fmadd(dx,dx,yzs);
                        const V f=galaxy::inv_cube(q);
                        const V fq=mul(f,load(m0+row+u));
                        store(l+row+u,fmadd(f,q,load(l+row+u)));
                        tx=fmadd(dx,fq,tx); ty=fmadd(dy,fq,ty); tz=fmadd(dz,fq,tz);
                    }
                }
            }
            g.t.acc[0][i]+=sum(tx);g.t.acc[1][i]+=sum(ty);g.t.acc[2][i]+=sum(tz);
        }
    }
    void kernel(double *dst, std::unordered_map<int,std::vector<double>> &base,
                const Link &e) {
        const int d[3]={e.dx,e.dy,e.dz};
        int axes[3]={0,1,2};
        std::sort(axes,axes+3,[&](int i,int j){return std::abs(d[i])>std::abs(d[j]);});
        const int a=std::abs(d[axes[0]]),b=std::abs(d[axes[1]]),c=std::abs(d[axes[2]]);
        const int k=key(a,b,c);
        auto it=base.find(k);
        if(it==base.end()){
            auto &f=base[k];
            f.resize(fft.nvpad);
            fft.kernel(f.data(),poly.p,poly.extent,g.len(g.cell[e.a]),a,b,c);
            it=base.find(k);
        }
        const double *src=it->second.data();
        const int n=fft.n, csz=fft.c;
        const int pitch[3]={2,2*csz,2*n*csz};
        int off[2][3][40];
        bool flip[40]{};
        for(int j=0;j<3;++j)for(int k=0;k<n;++k){
            const int ax=axes[j];
            const int i=d[ax]>=0 || !k?k:n-k;
            off[0][ax][k]=pitch[j]*i;
            off[1][ax][k]=pitch[j]*(i?n-i:0);
            if(!j)flip[k]=i>n/2;
        }
        for(int z=0;z<n;++z)for(int y=0;y<n;++y){
            double *w=dst+2*(z*n+y)*csz;
            if(axes[0]==0 && d[0]>=0){
                std::copy_n(src+off[0][2][z]+off[0][1][y],2*csz,w);
            } else if(axes[0]!=0){
                const int conj=flip[axes[0]==1?y:z];
                const double *r=src+off[conj][2][z]+off[conj][1][y];
                const __m128d sign=_mm_set_pd(conj?-0.:0.,0.);
                for(int x=0;x<csz;++x)
                    _mm_storeu_pd(w+2*x,_mm_xor_pd(_mm_loadu_pd(r+off[conj][0][x]),sign));
            } else {
                const double *r[2]={src+off[0][2][z]+off[0][1][y],
                                    src+off[1][2][z]+off[1][1][y]};
                for(int x=0;x<csz;++x){
                    const int conj=flip[x];
                    _mm_storeu_pd(w+2*x,_mm_xor_pd(_mm_loadu_pd(r[conj]+off[conj][0][x]),
                                                    _mm_set_pd(conj?-0.:0.,0.)));
                }
            }
        }
        std::fill(dst+fft.nf,dst+fft.nvpad,0.);
    }
    void translate() {
        const int capacity=std::max(32,int(plan.workspace_mb*(1<<20)/(16.*fft.nvpad)));
        std::vector<int> ids(g.cell.size(),-1),level_ids(g.cell.size(),-1),keys(35937,-1);
        const bool keep_fft=env("GALAXY_FFT_KEEP",1.)!=0.;
        const size_t length=size_t(capacity)*fft.nvpad;
        std::unique_ptr<double[]> source_fft,local_fft;
        if(keep_fft){
            source_fft.reset(new double[length]);
            local_fft.reset(new double[length]);
        }
        double tker=0., tfwd=0., tm2l=0., tbwd=0.,tpt=0.;
        for(size_t level=0;level<plan.levels.size();++level){
            auto &links=plan.levels[level];
            const auto &point=plan.point_links[level];
            if(links.empty()&&point.empty())continue;
            const double time0=cpu_time();
            if(plan.sort_links && !std::is_sorted(links.begin(),links.end(),GridPlan::link_order))
                std::sort(links.begin(),links.end(),GridPlan::link_order);
            std::vector<Link> offsets;
            std::vector<int> kmap(links.size());
            std::vector<int> level_cells;
            std::unordered_map<int,std::vector<double>> base;
            std::fill(keys.begin(),keys.end(),-1);
            int nk=0;
            auto add_cell=[&](int v){
                if(level_ids[v]<0){
                    level_ids[v]=level_cells.size();level_cells.push_back(v);
                }
            };
            for(size_t i=0;i<links.size();++i){
                const Link &a=links[i];
                int &id=keys[key(a.dx,a.dy,a.dz)];
                if(id<0){
                    id=nk++;
                    offsets.push_back(a);
                }
                add_cell(a.a);add_cell(a.b);
                kmap[i]=id;
            }
            for(auto q:point)add_cell(q.first);
            std::vector<double> kt(size_t(nk)*fft.nvpad),kernel_buffer(fft.nvpad);
            for(int k=0;k<nk;++k){
                kernel(kernel_buffer.data(),base,offsets[k]);
                for(int j=0;j<fft.nt;++j)
                    FFT::pack(kernel_buffer.data()+j*tile_width,
                              kt.data()+(size_t(j)*nk+k)*tile_width);
            }
            std::vector<double> ll(plan.recycle?0:level_cells.size()*width);
            std::vector<std::unique_ptr<double[]>> saved(plan.recycle?level_cells.size():0);
            std::vector<int> last(plan.recycle?level_cells.size():0,-1);
            if(plan.recycle)for(size_t i=0;i<links.size();++i){
                last[level_ids[links[i].a]]=i;
                last[level_ids[links[i].b]]=i;
            }
            auto local=[&](int f)->double*{
                if(!plan.recycle)return ll.data()+size_t(f)*width;
                auto &r=saved[f];
                if(!r)r=std::make_unique<double[]>(width);
                return r.get();
            };
            auto finish=[&](int f)->double*{
                const int v=level_cells[f];
                double *dst=m.data()+size_t(g.cell[v].field)*width;
                if(saved[f]){
                    std::copy_n(saved[f].get(),width,dst);
                    saved[f].reset();
                } else std::fill_n(dst,width,0.);
                return dst;
            };
            tker+=cpu_time()-time0;
            const double tp=cpu_time();
            for(auto q:point)to_points(q.first,q.second,local(level_ids[q.first]));
            if(plan.recycle)for(size_t k=0;k<last.size();++k)if(last[k]<0)finish(k);
            tpt+=cpu_time()-tp;
            for(size_t i=0;i<links.size();){
                const double time1=cpu_time();
                std::vector<int> use;
                size_t j=i;
                for(;j<links.size() && int(use.size())+2<capacity;++j){
                    for(int u:{links[j].a,links[j].b})if(ids[u]<0){
                        ids[u]=use.size();use.push_back(u);
                    }
                }
                const int n=use.size();
                const size_t len=size_t(n)*fft.nvpad;
                std::vector<double> tmp_m(keep_fft?0:len),tmp_l(keep_fft?0:len);
                double *sm=keep_fft?source_fft.get():tmp_m.data();
                double *sl=keep_fft?local_fft.get():tmp_l.data();
                if(keep_fft)std::fill_n(sl,len,0.);
                for(int k=0;k<n;++k){
                    int f=g.cell[use[k]].field;
                    fft.spec(m.data()+size_t(f)*width,poly.p,poly.sx,sm+k*tile_width,n);
                }
                struct E{int a,b,k;};
                std::vector<E> e(j-i);
                for(size_t k=i;k<j;++k)e[k-i]={ids[links[k].a]*tile_width,ids[links[k].b]*tile_width,
                                              kmap[k]*tile_width};
                const double time2=cpu_time();
                if(env("GALAXY_GRID_ROWS",0.)){
                    std::vector<int> first(n+1);
                    for(const E &q:e){++first[q.a/tile_width+1];++first[q.b/tile_width+1];}
                    for(int v=1;v<=n;++v)first[v]+=first[v-1];
                    auto next=first;
                    std::vector<Neighbour> adj(2*e.size());
                    for(const E &q:e){
                        adj[next[q.a/tile_width]++]={q.b,q.k,0};
                        adj[next[q.b/tile_width]++]={q.a,q.k,std::numeric_limits<int64_t>::min()};
                    }
                    for(int b=0;b<fft.nt;++b){
                        const double *m=sm+size_t(b)*n*tile_width,
                                     *k=kt.data()+size_t(b)*nk*tile_width;
                        double *l=sl+size_t(b)*n*tile_width;
                        for(int v=0;v<n;++v)
                            translate_row(l+v*tile_width,m,k,adj.data()+first[v],first[v+1]-first[v]);
                    }
                } else for(int b=0;b<fft.nt;++b){
                    const double *m=sm+size_t(b)*n*tile_width,
                                 *k=kt.data()+size_t(b)*nk*tile_width;
                    double *l=sl+size_t(b)*n*tile_width;
                    for(const E &q:e)translate_pair(l+q.a,l+q.b,m+q.a,m+q.b,k+q.k);
                }
                const double time3=cpu_time();
                if(plan.recycle){
                    for(bool terminal:{true,false})for(int k=0;k<n;++k){
                        const int f=level_ids[use[k]];
                        if((last[f]<int(j))!=terminal)continue;
                        fft.poly(sl+k*tile_width,n,terminal?finish(f):local(f),poly.p,poly.sx);
                    }
                    for(int k=0;k<n;++k)ids[use[k]]=-1;
                } else for(int k=0;k<n;++k){
                    const int f=level_ids[use[k]];
                    fft.poly(sl+k*tile_width,n,local(f),poly.p,poly.sx);
                    ids[use[k]]=-1;
                }
                tfwd+=time2-time1;tm2l+=time3-time2;tbwd+=cpu_time()-time3;
                i=j;
            }
            if(!plan.recycle)for(size_t i=0;i<level_cells.size();++i)
                std::copy_n(ll.data()+i*width,width,m.data()+size_t(g.cell[level_cells[i]].field)*width);
        }
        if(env("GALAXY_VERBOSE",0.))
            std::fprintf(stderr,"fft kernels %.4f forward %.4f m2l %.4f backward %.4f point %.4f\n",
                         tker,tfwd,tm2l,tbwd,tpt);
    }
};

int solve_grid(const char *input,const char *output){
    int fd=open(input,O_RDONLY);
    struct stat info{};
    if(fd<0)return 1;
    if(fstat(fd,&info)||info.st_size%32||info.st_size<=0){
        close(fd);
        return 1;
    }
    const size_t bytes=info.st_size;
    if(bytes<65536*32 || bytes>4194304LL*32){
        close(fd);
        return 2;
    }
    const int n=bytes/32;
    const double limit=env("GALAXY_CPU_LIMIT_S",2.0*n/65536.);
    const double *raw=(const double*)mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(raw==MAP_FAILED)return 1;
    Tree t(n);t.mass=raw[0];
    for(int i=0;i<n;++i)t.pts[i]={{raw[4*i+1],raw[4*i+2],raw[4*i+3]},i};
    munmap((void*)raw,bytes);
    if(env("GALAXY_ALIGN",0.))t.align_thin();
    const int bottom=(int)env("GALAXY_GRID_MIN_LEAF",n<=262144?24.:n<=1048576?32.:38.);
    GridTree tree(t,bottom);
#if __has_include("fft_wisdom.hpp")
    if(env("GALAXY_WISDOM",1.))fftw_import_wisdom_from_string(fft_wisdom);
#endif
    const double prep=cpu_time();
    Algebra alg;
    static const Settings options[] = {
        {18,1.47,30000.,15500.,27,270},
        {17,1.45,27300.,13500.,25,246},
        {16,1.43,21400.,11000.,24,228},
        {15,1.41,17500.,9500.,22,210},
        {14,1.37,15700.,8700.,20,199},
        {13,1.32,14200.,7900.,19,190},
        {12,1.31,12000.,7000.,18,180},
        {11,1.23,10500.,6400.,16,167},
        {10,1.14,6500.,5900.,12,150},
        { 9,1.14,5400.,4100.,11,130},
        { 8,1.08,3900.,2750.,8,111},
        { 7,1.06,3200.,2200.,8,95}
    };
    const int force_order=(int)env("GALAXY_GRID_ORDER",0.);
    Settings selected{};
    std::unique_ptr<GridPlan> best;
    double predicted=0.;
    const bool verbose=env("GALAXY_VERBOSE",0.);
    const bool tune=env("GALAXY_GRID_TUNE",1.)!=0. && !force_order;
    // At the intermediate size, leave more headroom for the translation batches.
    const bool calibrate=env("GALAXY_CALIBRATE",n==1048576?0.:1.)!=0. && tune;
    const double target=(calibrate?.933:.917)*limit-.006-45e-9*n;
    double near_cost=0.,cross_cost=0.;
    for(Settings spec:options){
        if(force_order && spec.order!=force_order)continue;
        if(!force_order && spec.order>=17 && limit<28e-6*n)continue;
        spec.minprod=env("GALAXY_GRID_PROD",spec.minprod);
        spec.nearprod=env("GALAXY_GRID_NEAR",spec.nearprod);
        spec.leaf=int(env("GALAXY_GRID_LEAF",spec.leaf));
        spec.small=int(env("GALAXY_GRID_MIN",spec.small));
        spec.extent=env("GALAXY_GRID_EXTENT",spec.extent);
        double fft_cost=0.;
        auto trial=[&](Settings s){
            for(Cell &a:tree.cell){a.live=a.below=false; a.field=-1;}
            auto pl=std::make_unique<GridPlan>(tree,alg,s);
            pl->others.limit_pairs=force_order?1e100:limit/2.15e-9;
            pl->plan(tree.root,tree.root);
            if(!pl->others.stopped)pl->complete();
            if(!pl->others.stopped)pl->mark(tree.root);
            double cost=pl->estimate(fft_cost,near_cost,cross_cost);
            if(calibrate && !near_cost && cost<(target-cpu_time())*1.45){
                near_cost=time_near(*pl);
                cross_cost=Engine::time_translations();
                cost=pl->estimate(fft_cost,near_cost,cross_cost);
                if(verbose)
                    std::fprintf(stderr,"grid rates %.4f ns direct, %.4f ns spectral\n",
                                 1e9*near_cost,1e9*cross_cost);
            }
            if(!fft_cost && cost<(target-cpu_time())*1.35 && env("GALAXY_GRID_PROBE",1.)){
                const double elapsed=time_fft(s.order);
                const double d=fft_dimension(s.order);
                fft_cost=elapsed*1.12+2.8e-9*d*d*d;
                cost=pl->estimate(fft_cost,near_cost,cross_cost);
            }
            if(verbose)
                std::fprintf(stderr,"grid trial p=%d prod=%.0f axis=%d forecast %.4g room %.4f [%zu cells, %zu links, %.4g direct]\n",
                    s.order,s.minprod,s.axial,cost,target-cpu_time(),pl->nfields,pl->count,pl->others.p2pcost);
            if(cost<1e99 && (force_order || cost<target-cpu_time()) &&
                  (!best || s.axial || cost<predicted)){
                predicted=cost;
                selected=s;
                best=std::move(pl);
            }
            return cost;
        };
        const double base=trial(spec);
        if(tune && !best && base<(target-cpu_time())*1.18 &&
           cpu_time()-prep<.085*limit){
            Settings alt=spec;
            const double ratio=(n>=262144 || spec.order<=11)?.77:1.28;
            alt.minprod*=ratio;
            alt.nearprod*=ratio;
            alt.small=std::max(7,int(spec.small*std::sqrt(ratio)+.5));
            alt.leaf=std::max(70,int(spec.leaf*std::sqrt(ratio)+.5));
            trial(alt);
            if(!best && base<(target-cpu_time())*1.09 && cpu_time()-prep<.075*limit){
                alt=spec;
                alt.minprod/=ratio;
                alt.nearprod/=ratio;
                alt.leaf=int(spec.leaf/std::sqrt(ratio)+.5);
                trial(alt);
            }
        }
        if(best){
            if(tune && selected.order>=9 && predicted<(target-cpu_time())*.864 &&
               cpu_time()-prep<.079*limit){
                Settings alt=selected;
                alt.axial=true;
                trial(alt);
            }
            best->mark_again();
            break;
        }
    }
    if(!best)return 2;
    GridPlan &pl=*best;
    const int higher=int(env("GALAXY_GRID_HIGH",0.));
    Settings highspec{};
    std::unique_ptr<GridPlan> high;
    for(const Settings &s:options)if(higher==s.order && s.order>selected.order)highspec=s;
    if(highspec.order){
        high=std::make_unique<GridPlan>(tree,alg,highspec);
        pl.move_close_to(*high,env("GALAXY_GRID_HIGH_MIN",0.));
        high->mark_again();
        const double high_cost=high->estimate();
        pl.mark_again();
        predicted=pl.estimate()+high_cost;
        if(verbose)std::fprintf(stderr,"grid extra order=%d pairs=%zu expected combined %.4f\n",
                               highspec.order,high->count,predicted);
    }
    const double extra=env("GALAXY_GRID_REFINE",0.);
    if(extra>0.&&!high){
        const double maxextra=force_order?extra:
            std::max(0.,std::min(extra*limit,.91*limit-predicted-cpu_time()-.008-60e-9*n));
        const double spent=pl.refine(maxextra);
        predicted+=spent;
        if(verbose)std::fprintf(stderr,"grid exact refinement %.5f planned seconds\n",spent);
    }
    const double planned=cpu_time();
    if(verbose)
        std::fprintf(stderr,"grid=%zu topo=%zu depth=%d order=%d links=%zu fields=%zu, direct=%.5g (%zu) point=%zu, prep=%.4f plan=%.4f forecast=%.4f\n",
            tree.cell.size(),t.nodes.size(),tree.height,selected.order,pl.count,pl.nfields,
            pl.others.p2pcost,pl.others.direct.size(),pl.point_terms,prep,planned-prep,predicted);
    if(pl.count || pl.point_terms){
        Engine en(pl,selected.order,selected.extent);
        en.upward(tree.root,nullptr);
        const double a=cpu_time();
        en.translate();
        const double b=cpu_time();
        en.downward(tree.root,nullptr);
        if(verbose)std::fprintf(stderr,"grid up %.4f, translate %.4f, down %.4f\n",
                                a-planned,b-a,cpu_time()-b);
    }
    if(high && high->count){
        high->mark_again();
        Engine en(*high,highspec.order,highspec.extent);
        en.upward(tree.root,nullptr);
        en.translate();
        en.downward(tree.root,nullptr);
    }
    const double now=cpu_time();
    Direct direct(t);
    direct.version=8;
    execute_near(t,pl.others,direct,limit-.009-70e-9*n,!force_order&&!high);
    if(verbose)std::fprintf(stderr,"grid direct %.4f\n",cpu_time()-now);
    return write_output(t,output)?0:1;
}

} // namespace grid
