#define main cartesian_main
#include "solve.cpp"
#undef main

struct BatchSums {
    V a0{},a1{},a2{},a3{},b0{},b1{},b2{},b3{};
};

template<int K,int N>
__attribute__((always_inline)) static inline
void dot_row_step(const V* A,const V* B,const V* D,BatchSums& s) {
    V d=D[K];
    if constexpr(K%4==0){s.a0+=B[K]*d;s.b0+=A[K]*d;}
    if constexpr(K%4==1){s.a1+=B[K]*d;s.b1+=A[K]*d;}
    if constexpr(K%4==2){s.a2+=B[K]*d;s.b2+=A[K]*d;}
    if constexpr(K%4==3){s.a3+=B[K]*d;s.b3+=A[K]*d;}
    if constexpr(K+1<N)dot_row_step<K+1,N>(A,B,D,s);
}

template<int DA=3>
__attribute__((always_inline)) static inline
void hdot_rows(const V* A,const V* B,const V* D,int limit,int db,int ib,int kb,
               const int dbase[MAXP+1][3],BatchSums& s) {
    if(limit>=DA) {
        constexpr int m0=DA*DA,m1=m0+DA+1;
        dot_row_step<0,DA+1>(A+m0,B+m0,D+dbase[DA+db][kb]+ib,s);
        if constexpr(DA>0)dot_row_step<0,DA>(A+m1,B+m1,D+dbase[DA+db][kb+1]+ib,s);
        if constexpr(DA<MAXP-1)hdot_rows<DA+1>(A,B,D,limit,db,ib,kb,dbase,s);
    }
}

template<int DA,int IA=0>
__attribute__((always_inline)) static inline
void cdot_degree(const V* A,const V* B,const V* D,int db,int ib,int jb,
                 const int rowbase[MAXP+1][MAXP+1],BatchSums& s) {
    constexpr int m0=DA*(DA+1)*(DA+2)/6+IA*(DA+1)-IA*(IA-1)/2;
    dot_row_step<0,DA-IA+1>(A+m0,B+m0,D+rowbase[DA+db][IA+ib]+jb,s);
    if constexpr(IA<DA)cdot_degree<DA,IA+1>(A,B,D,db,ib,jb,rowbase,s);
}

template<int DA=3>
__attribute__((always_inline)) static inline
void cdot_rows(const V* A,const V* B,const V* D,int limit,int db,int ib,int jb,
               const int rowbase[MAXP+1][MAXP+1],BatchSums& s) {
    if(limit>=DA) {
        cdot_degree<DA>(A,B,D,db,ib,jb,rowbase,s);
        if constexpr(DA<MAXP-1)cdot_rows<DA+1>(A,B,D,limit,db,ib,jb,rowbase,s);
    }
}

struct Hybrid : Solver {
    static constexpr int HH=(MAXP+1)*(MAXP+1)+1;
    static constexpr int HZERO=HH-1;
    static constexpr int HDZERO=1+3*MAXP*(MAXP+1)/2;
    int ph, pc, ps, cc;
    double switch_r;
    double deadline=4.8;
    bool variable=false;
    int refinement=0;
    bool lagrange=false;
    bool partial_refinement=false;
    bool weighted_targets=false;
    bool batched_translations=true;
    double fixed_work=0;
    double base_budget_fraction=.9;
    bool refine_particles=true;
    int refined_particles=0;
    long long horders[MAXP+1]{};
    long long corders[MAXP+1]{};
    std::vector<int> hlist, hid, hc;
    std::vector<std::vector<int>> hsums;
    std::vector<std::vector<int>> hmx,hmy,hlx,hly;
    std::vector<int> hpx,hpy,hpz0,hpz1,huz0,huz1;
    std::vector<double> hsign,hinvd;
    std::vector<int> hd,hDx,hDy,hDz,hDxx,hDyy;
    std::vector<double> hfac;
    std::vector<int> dprev[6];
    std::vector<double> dfac;
    int dbase[MAXP+1][3]{},rowbase[MAXP+1][MAXP+1]{};
    std::vector<double> hmom, hlocal;
    std::vector<Pair> outer, inner;
    struct AccuracyPlan {
        Pair pair;
        double hqa,hqb,hia,hib,cqa,cqb,cia,cib;
        double fia,fib,fcia,fcib,direct_cost,low_cost;
        int child0=-1,child1=-1;
        bool convergent=true;
    };
    std::vector<AccuracyPlan> plans;
    std::vector<int> plan_roots;
    std::vector<double> radius_h,radius_c,acc_scale;
    std::vector<double> target_h,target_c;
    std::vector<std::vector<int>> levels;
    std::vector<int> leaf_nodes;
    std::vector<Pair> hgroup[MAXP+1],cgroup[MAXP+1];
    std::vector<Pair> hfull[MAXP+1],cfull[MAXP+1],hybrid_pairs;
    std::vector<Pair> replace_direct;
    double chosen_tolerance=0,predicted_work=0;
    double predicted_error=0;

    Hybrid(int p,double theta,int leaf,int pc,int ps,double sw)
        : Solver(p,theta,leaf,2),ph(p),pc(pc),ps(ps),switch_r(sw) {
        cc=count[std::max(pc,ps)];
        hid.resize(nc,-1);hc.resize(P+1);
        for(int d=0;d<=ph;++d) {
            for(int k=0;k<=std::min(d,1);++k) {
                for(int i=0;i<=d-k;++i) {
                    int j=d-k-i,n=idx[i][j][k];
                    hid[n]=hlist.size();hlist.push_back(n);
                }
            }
            hc[d]=hlist.size();
            int base=d?1+3*(d-1)*d/2:0;
            for(int k=0;k<3;++k)dbase[d][k]=base+k*(d+1)-k*(k-1)/2;
            for(int i=0;i<=d;++i)
                rowbase[d][i]=d*(d+1)*(d+2)/6+i*(d+1)-i*(i-1)/2;
        }
        hsums.resize(hlist.size());
        int hh=hlist.size();
        hpx.resize(hh,HZERO);hpy.resize(hh,HZERO);
        hpz0.resize(hh,HZERO);hpz1.resize(hh,HZERO);
        huz0.resize(hh,HZERO);huz1.resize(hh,HZERO);
        hsign.resize(hh);hinvd.resize(hh);
        hd.resize(hh);hDx.resize(hh,HDZERO);hDy.resize(hh,HDZERO);hDz.resize(hh,HDZERO);
        hDxx.resize(hh,HDZERO);hDyy.resize(hh,HDZERO);hfac.resize(hh);
        hmx.resize(hh);hmy.resize(hh);hlx.resize(hh);hly.resize(hh);
        for(int h=0;h<hh;++h) {
            int n=hlist[h],i=ix[n],j=iy[n],k=iz[n],d=deg[n];
            hd[h]=dbase[d][k]+i;
            hfac[h]=fact[i]*fact[j];
            if(i)hDx[h]=dbase[d-1][k]+i-1;
            if(j)hDy[h]=dbase[d-1][k]+i;
            if(k)hDz[h]=dbase[d-1][0]+i;
            if(i>1)hDxx[h]=dbase[d-2][k]+i-2;
            if(j>1)hDyy[h]=dbase[d-2][k]+i;
            if(i)hpx[h]=hid[idx[i-1][j][k]];
            if(j)hpy[h]=hid[idx[i][j-1][k]];
            if(k)hpz0[h]=hid[idx[i][j][0]];
            else {
                if(i>1)hpz0[h]=hid[idx[i-2][j][1]];
                if(j>1)hpz1[h]=hid[idx[i][j-2][1]];
            }
            if(d<ph) {
                if(!k)huz0[h]=hid[idx[i][j][1]];
                else {
                    huz0[h]=hid[idx[i+2][j][0]];
                    huz1[h]=hid[idx[i][j+2][0]];
                }
            }
            hsign[h]=k?1:-1;hinvd[h]=d?1./d:0;
            for(int t=0;t<=i;++t)hmx[h].push_back(hid[idx[i-t][j][k]]);
            for(int t=0;t<=j;++t)hmy[h].push_back(hid[idx[i][j-t][k]]);
            for(int t=0;t<=ph-d;++t) {
                hlx[h].push_back(hid[idx[i+t][j][k]]);
                hly[h].push_back(hid[idx[i][j+t][k]]);
            }
        }
        for(size_t b=1;b<hlist.size();++b) {
            int bi=hlist[b],na=hc[P-deg[bi]];
            hsums[b].resize(na);
            for(int a=0;a<na;++a) {
                int ai=hlist[a];
                hsums[b][a]=idx[ix[ai]+ix[bi]][iy[ai]+iy[bi]][iz[ai]+iz[bi]];
            }
        }
        P=std::max(pc,ps);
        nc=count[P];nm=count[P-1];
        for(auto& v:dprev)v.resize(nc,MAXC);
        dfac.resize(nc);
        for(int n=0;n<nc;++n) {
            int i=ix[n],j=iy[n],k=iz[n];
            if(i)dprev[0][n]=idx[i-1][j][k];
            if(j)dprev[1][n]=idx[i][j-1][k];
            if(k)dprev[2][n]=idx[i][j][k-1];
            if(i>1)dprev[3][n]=idx[i-2][j][k];
            if(j>1)dprev[4][n]=idx[i][j-2][k];
            if(k>1)dprev[5][n]=idx[i][j][k-2];
            dfac[n]=fact[i]*fact[j]*fact[k];
        }
    }

    void hpoly(V* H,V x,V y,V z,V mass,int order) {
        H[0]=mass;H[HZERO]=V{};
        for(int h=1;h<hc[order];++h) {
            H[h]=(x*H[hpx[h]]+y*H[hpy[h]]
                +z*(H[hpz0[h]]+H[hpz1[h]])*hsign[h])*hinvd[h];
        }
    }

    template<class T>
    void hshift_m(T* dst,const T* src,T x,T y,T z) {
        T aa[HH],bb[HH],f[MAXP+1];
        int hn=hc[ph-1];
        T* a=aa;T* b=bb;
        a[HZERO]=b[HZERO]=T{};
        memcpy(a,src,hn*sizeof(T));
        for(int dim=0;dim<2;++dim) {
            f[0]=T{}+1.0;T s=dim?y:x;
            for(int k=1;k<ph;++k)f[k]=f[k-1]*s/k;
            for(int h=0;h<hn;++h) {
                T v{};
                const auto& ids=dim?hmy[h]:hmx[h];
                for(size_t k=0;k<ids.size();++k)v+=a[ids[k]]*f[k];
                b[h]=v;
            }
            std::swap(a,b);
        }
        for(int h=0;h<hn;++h)dst[h]+=a[h];
        for(int d=1;d<ph;++d) {
            T s=z/d;
            for(int h=hc[d-1];h<hn;++h) {
                b[h]=(a[hpz0[h]]+a[hpz1[h]])*(hsign[h]*s);
                dst[h]+=b[h];
            }
            std::swap(a,b);
        }
    }

    __attribute__((noinline))
    void hform_leaf(int ni) {
        const Node& nd=nodes[ni];
        int hn=hc[ph-1];
        V accum[HH]{},H[HH];
        for(int q=nd.begin;q<nd.end;q+=4) {
            V x{},y{},z{},m{};
            for(int l=0;l<4;++l) {
                int qi=std::min(q+l,nd.end-1);
                x[l]=nd.x-px[qi];y[l]=nd.y-py[qi];z[l]=nd.z-pz[qi];
                m[l]=q+l<nd.end?mass[qi]:0;
            }
            hpoly(H,x,y,z,m,ph-1);
            for(int h=0;h<hn;++h)accum[h]+=H[h];
        }
        double* dst=&hmom[ni*hn];
        for(int h=0;h<hn;++h)
            dst[h]=accum[h][0]+accum[h][1]+accum[h][2]+accum[h][3];
    }

    void h_upward(int ni) {
        const Node& nd=nodes[ni];
        if(nd.left<0) {
            hform_leaf(ni);
        } else {
            h_upward(nd.left);h_upward(nd.right);
            int hn=hc[ph-1];
            for(int c:{nd.left,nd.right}) {
                const Node& ch=nodes[c];
                hshift_m(&hmom[ni*hn],&hmom[c*hn],nd.x-ch.x,nd.y-ch.y,nd.z-ch.z);
            }
        }
    }

    template<class T>
    void hshift_l(T* dst,const T* src,T x,T y,T z) {
        T aa[HH],bb[HH],f[MAXP+1];
        int hn=hc[ph];
        T* a=aa;T* b=bb;
        a[HZERO]=b[HZERO]=T{};
        memcpy(a,src,hn*sizeof(T));
        for(int dim=0;dim<2;++dim) {
            f[0]=T{}+1.0;T s=dim?y:x;
            for(int k=1;k<=ph;++k)f[k]=f[k-1]*s/k;
            b[0]=T{};
            for(int h=1;h<hn;++h) {
                T v{};
                const auto& ids=dim?hly[h]:hlx[h];
                for(size_t k=0;k<ids.size();++k)v+=a[ids[k]]*f[k];
                b[h]=v;
            }
            std::swap(a,b);
        }
        for(int h=1;h<hn;++h)dst[h]+=a[h];
        for(int d=1;d<ph;++d) {
            T s=z/d;
            for(int h=1;h<hc[ph-d];++h) {
                b[h]=(a[huz0[h]]+a[huz1[h]])*(-hsign[h]*s);
                dst[h]+=b[h];
            }
            std::swap(a,b);
        }
    }

    __attribute__((noinline))
    void h_eval_leaf(int ni) {
        const Node& nd=nodes[ni];
        const double* L=&hlocal[ni*hc[ph]];
        for(int q=nd.begin;q<nd.end;q+=4) {
            V x{},y{},z{},H[HH];
            for(int l=0;l<4;++l) {
                int qi=std::min(q+l,nd.end-1);
                x[l]=px[qi]-nd.x;y[l]=py[qi]-nd.y;z[l]=pz[qi]-nd.z;
            }
            hpoly(H,x,y,z,V{1,1,1,1},ph-1);
            V a{},b{},c{};
            for(int h=1;h<hc[ph];++h) {
                a+=L[h]*H[hpx[h]];
                b+=L[h]*H[hpy[h]];
                c+=(L[h]*hsign[h])*(H[hpz0[h]]+H[hpz1[h]]);
            }
            for(int l=0;l<4&&q+l<nd.end;++l) {
                ax[q+l]+=a[l];ay[q+l]+=b[l];az[q+l]+=c[l];
            }
        }
    }

    void h_downward(int ni) {
        const Node& nd=nodes[ni];
        if(nd.left<0)h_eval_leaf(ni);
        else {
            int hn=hc[ph];
            for(int c:{nd.left,nd.right}) {
                const Node& ch=nodes[c];
                hshift_l(&hlocal[c*hn],&hlocal[ni*hn],ch.x-nd.x,ch.y-nd.y,ch.z-nd.z);
                h_downward(c);
            }
        }
    }

    void prepare_levels() {
        std::vector<int> depth(nodes.size());
        for(size_t n=0;n<nodes.size();++n) {
            const Node& nd=nodes[n];
            if(nd.left<0)leaf_nodes.push_back(n);
            else {
                int d=depth[n];
                if(int(levels.size())<=d)levels.resize(d+1);
                levels[d].push_back(n);
                depth[nd.left]=depth[nd.right]=d+1;
            }
        }
    }

    void h_upward_batched() {
        for(int n:leaf_nodes)hform_leaf(n);
        int hn=hc[ph-1];
        alignas(32) V source[HH],destination[HH];
        for(int d=int(levels.size())-1;d>=0;--d) {
            const auto& list=levels[d];
            for(size_t base=0;base<list.size();base+=4) {
                int ids[4],n=std::min(size_t(4),list.size()-base);
                for(int l=0;l<4;++l)ids[l]=list[base+std::min(l,n-1)];
                std::fill_n(destination,hn,V{});
                for(int side=0;side<2;++side) {
                    int children[4];V x{},y{},z{};
                    for(int l=0;l<4;++l) {
                        const Node& parent=nodes[ids[l]];
                        children[l]=side?parent.right:parent.left;
                        const Node& child=nodes[children[l]];
                        x[l]=parent.x-child.x;y[l]=parent.y-child.y;z[l]=parent.z-child.z;
                    }
                    for(int h=0;h<hn;++h)
                        for(int l=0;l<4;++l)source[h][l]=hmom[children[l]*hn+h];
                    hshift_m(destination,source,x,y,z);
                }
                for(int h=0;h<hn;++h)
                    for(int l=0;l<n;++l)hmom[ids[l]*hn+h]=destination[h][l];
            }
        }
    }

    void cart_upward_batched() {
        for(int n:leaf_nodes)form_leaf(n);
        alignas(32) V source[MAXC],destination[MAXC];
        for(int d=int(levels.size())-1;d>=0;--d) {
            const auto& list=levels[d];
            for(size_t base=0;base<list.size();base+=4) {
                int ids[4],n=std::min(size_t(4),list.size()-base);
                for(int l=0;l<4;++l)ids[l]=list[base+std::min(l,n-1)];
                std::fill_n(destination,nm,V{});
                for(int side=0;side<2;++side) {
                    int children[4];V x{},y{},z{};
                    for(int l=0;l<4;++l) {
                        const Node& parent=nodes[ids[l]];
                        children[l]=side?parent.right:parent.left;
                        const Node& child=nodes[children[l]];
                        x[l]=parent.x-child.x;y[l]=parent.y-child.y;z[l]=parent.z-child.z;
                    }
                    for(int a=0;a<nm;++a)
                        for(int l=0;l<4;++l)source[a][l]=moments[children[l]*nm+a];
                    shift_m(destination,source,x,y,z);
                }
                for(int a=0;a<nm;++a)
                    for(int l=0;l<n;++l)moments[ids[l]*nm+a]=destination[a][l];
            }
        }
    }

    void cart_downward_batched() {
        alignas(32) V source[MAXC],destination[MAXC];
        for(const auto& list:levels) {
            for(size_t base=0;base<list.size();base+=4) {
                int ids[4],n=std::min(size_t(4),list.size()-base);
                for(int l=0;l<4;++l)ids[l]=list[base+std::min(l,n-1)];
                for(int a=0;a<nc;++a)
                    for(int l=0;l<4;++l)source[a][l]=locals[ids[l]*nc+a];
                for(int side=0;side<2;++side) {
                    int children[4];V x{},y{},z{};
                    for(int l=0;l<4;++l) {
                        const Node& parent=nodes[ids[l]];
                        children[l]=side?parent.right:parent.left;
                        const Node& child=nodes[children[l]];
                        x[l]=child.x-parent.x;y[l]=child.y-parent.y;z[l]=child.z-parent.z;
                    }
                    std::fill_n(destination,nc,V{});
                    shift_l(destination,source,x,y,z);
                    for(int a=1;a<nc;++a)
                        for(int l=0;l<n;++l)locals[children[l]*nc+a]+=destination[a][l];
                }
            }
        }
        for(int n:leaf_nodes)evaluate_leaf(n);
    }

    void h_downward_batched() {
        int hn=hc[ph];
        alignas(32) V source[HH],destination[HH];
        for(const auto& list:levels) {
            for(size_t base=0;base<list.size();base+=4) {
                int ids[4],n=std::min(size_t(4),list.size()-base);
                for(int l=0;l<4;++l)ids[l]=list[base+std::min(l,n-1)];
                for(int h=0;h<hn;++h)
                    for(int l=0;l<4;++l)source[h][l]=hlocal[ids[l]*hn+h];
                for(int side=0;side<2;++side) {
                    int children[4];V x{},y{},z{};
                    for(int l=0;l<4;++l) {
                        const Node& parent=nodes[ids[l]];
                        children[l]=side?parent.right:parent.left;
                        const Node& child=nodes[children[l]];
                        x[l]=child.x-parent.x;y[l]=child.y-parent.y;z[l]=child.z-parent.z;
                    }
                    std::fill_n(destination,hn,V{});
                    hshift_l(destination,source,x,y,z);
                    for(int h=1;h<hn;++h)
                        for(int l=0;l<n;++l)hlocal[children[l]*hn+h]+=destination[h][l];
                }
            }
        }
        for(int n:leaf_nodes)h_eval_leaf(n);
    }

    void deriv_h(V* D,V x,V y,V z,int order) {
        V u=x*x+y*y+z*z;
        V inv=V{1,1,1,1}/u;
        D[0]=V{1,1,1,1}/V(_mm256_sqrt_pd(u));D[HDZERO]=V{};
        V xx=x*inv,yy=y*inv,zz=z*inv;
        for(int h=1;h<hc[order];++h) {
            int d=deg[hlist[h]];
            V a=xx*D[hDx[h]]+yy*D[hDy[h]]+zz*D[hDz[h]];
            V b=D[hDxx[h]]+D[hDyy[h]];
            D[hd[h]]=a*((1.-2*d)/d)+b*inv*((1.-d)/d);
        }
        for(int h=1;h<hc[order];++h)D[hd[h]]*=hfac[h];
        for(int d=2;d<=order;++d)
            for(int i=0;i<=d-2;++i)
                D[dbase[d][2]+i]=-D[dbase[d][0]+i]-D[dbase[d][0]+i+2];
    }

    template<bool FULL=false>
    void harmonic_m2l(const std::vector<Pair>& pairs,int cap,double deadline) {
        alignas(32) V D[HDZERO+1],MA[HH],MB[HH];
        int hn=hc[ph-1],start=FULL?0:hc[2];
        constexpr int first=FULL?0:3;
        double work[MAXP+1]{},done=0,begin=cpu_time();
        int order=cap;
        for(int p=4;p<=ph;++p) {
            work[p]=16*hc[p];
            for(int b=1;b<hc[p-first];++b)work[p]+=hc[p-deg[hlist[b]]]-start;
        }
        for(size_t base=0;base<pairs.size();base+=4) {
            if(base%1024==0) {
                double now=cpu_time();
                if(now>deadline+.025) {
                    if constexpr(FULL)order=4;
                    else break;
                }
                if(base) {
                double rate=(now-begin)/done;
                double remain=pairs.size()-base,left=(deadline-now)*.94;
                order=cap;
                while(order>4 && rate*remain*work[order]>left)--order;
                }
            }
            int as[4],bs[4];V x{},y{},z{};
            int n=std::min(size_t(4),pairs.size()-base);
            for(int l=0;l<4;++l) {
                Pair pr=pairs[base+std::min(l,n-1)];
                as[l]=pr.a;bs[l]=pr.b;
                x[l]=nodes[pr.a].x-nodes[pr.b].x;
                y[l]=nodes[pr.a].y-nodes[pr.b].y;
                z[l]=nodes[pr.a].z-nodes[pr.b].z;
            }
            deriv_h(D,x,y,z,order);
            for(int a=start;a<hc[order-1];++a) {
                double sign=(deg[hlist[a]]&1)?-1:1;
                for(int l=0;l<4;++l) {
                    MA[a][l]=sign*hmom[as[l]*hn+a];
                    MB[a][l]=hmom[bs[l]*hn+a];
                }
            }
            for(int b=1;b<hc[order-first];++b) {
                int bi=hlist[b];
                BatchSums s;
                hdot_rows<first>(MA,MB,D,order-deg[bi],deg[bi],ix[bi],iz[bi],dbase,s);
                V la=(s.a0+s.a1)+(s.a2+s.a3),lb=(s.b0+s.b1)+(s.b2+s.b3);
                double sign=(deg[bi]&1)?-1:1;
                for(int l=0;l<n;++l) {
                    hlocal[as[l]*hc[ph]+b]+=la[l];
                    hlocal[bs[l]*hc[ph]+b]+=sign*lb[l];
                }
            }
            done+=n*work[order];
            horders[order]+=n;
        }
    }

    void deriv_full(V* D,V x,V y,V z,double eps,int order) {
        V u=x*x+y*y+z*z+V{eps,eps,eps,eps};
        V inv=V{1,1,1,1}/u;
        D[0]=V{1,1,1,1}/V(_mm256_sqrt_pd(u));D[MAXC]=V{};
        V xx=x*inv,yy=y*inv,zz=z*inv;
        for(int d=1;d<=order;++d) {
            double f=(1.-2*d)/d;
            V xc=xx*f,yc=yy*f,zc=zz*f,bc=inv*((1.-d)/d);
            for(int n=count[d-1];n<count[d];++n)
                D[n]=xc*D[dprev[0][n]]+yc*D[dprev[1][n]]+zc*D[dprev[2][n]]
                    +bc*(D[dprev[3][n]]+D[dprev[4][n]]+D[dprev[5][n]]);
        }
        for(int n=1;n<count[order];++n)D[n]*=dfac[n];
    }

    void deriv_newton(V* D,V x,V y,V z,int order) {
        V u=x*x+y*y+z*z,inv=V{1,1,1,1}/u;
        D[0]=V{1,1,1,1}/V(_mm256_sqrt_pd(u));D[MAXC]=V{};
        V xx=x*inv,yy=y*inv,zz=z*inv;
        for(int d=1;d<=order;++d) {
            double f=(1.-2*d)/d;
            V xc=xx*f,yc=yy*f,zc=zz*f,bc=inv*((1.-d)/d);
            for(int h=hc[d-1];h<hc[d];++h) {
                int n=hlist[h];
                D[n]=xc*D[dprev[0][n]]+yc*D[dprev[1][n]]+zc*D[dprev[2][n]]
                    +bc*(D[dprev[3][n]]+D[dprev[4][n]]);
            }
        }
        for(int h=1;h<hc[order];++h){int n=hlist[h];D[n]*=dfac[n];}
        for(int d=2;d<=order;++d)
            for(int k=2;k<=d;++k)
                for(int i=0;i<=d-k;++i) {
                    int j=d-k-i;
                    D[idx[i][j][k]]=-D[idx[i+2][j][k-2]]-D[idx[i][j+2][k-2]];
                }
    }

    template<bool FULL=false>
    void correction_m2l(const std::vector<Pair>& pairs,int order,bool difference) {
        alignas(32) V D[MAXC+1],DN[MAXC+1],MA[MAXC],MB[MAXC];
        int nn=count[order],mm=count[order-1],start=FULL?0:count[2];
        constexpr int first=FULL?0:3;
        for(size_t base=0;base<pairs.size();base+=4) {
            if(base%1024==0 && cpu_time()>deadline-.22) {
                if constexpr(FULL){order=4;nn=count[order];mm=count[order-1];}
                else break;
            }
            int as[4],bs[4];V x{},y{},z{};
            int n=std::min(size_t(4),pairs.size()-base);
            for(int l=0;l<4;++l) {
                Pair pr=pairs[base+std::min(l,n-1)];
                as[l]=pr.a;bs[l]=pr.b;
                x[l]=nodes[pr.a].x-nodes[pr.b].x;
                y[l]=nodes[pr.a].y-nodes[pr.b].y;
                z[l]=nodes[pr.a].z-nodes[pr.b].z;
            }
            deriv_full(D,x,y,z,EPS2,order);
            if(difference) {
                deriv_newton(DN,x,y,z,order);
                for(int b=1;b<nn;++b)D[b]-=DN[b];
            }
            for(int a=start;a<mm;++a) {
                double sign=(deg[a]&1)?-1:1;
                for(int l=0;l<4;++l) {
                    MA[a][l]=sign*moments[as[l]*nm+a];
                    MB[a][l]=moments[bs[l]*nm+a];
                }
            }
            for(int b=1;b<nn;++b) {
                int na=count[order-deg[b]];
                if(na<=start)continue;
                BatchSums s;
                cdot_rows<first>(MA,MB,D,order-deg[b],deg[b],ix[b],iy[b],rowbase,s);
                V la=(s.a0+s.a1)+(s.a2+s.a3),lb=(s.b0+s.b1)+(s.b2+s.b3);
                double sign=(deg[b]&1)?-1:1;
                for(int l=0;l<n;++l) {
                    locals[as[l]*nc+b]+=la[l];
                    locals[bs[l]*nc+b]+=sign*lb[l];
                }
            }
            corders[order]+=n;
        }
    }

    void predict_accelerations() {
        constexpr int order=6;
        int lc=count[order],mc=count[order-1];
        std::vector<double> preliminary(nodes.size()*lc);
        alignas(32) V D[MAXC+1],MA[MAXC],MB[MAXC];
        for(size_t base=0;base<far.size();base+=4) {
            int as[4],bs[4];V x{},y{},z{};
            int n=std::min(size_t(4),far.size()-base);
            for(int l=0;l<4;++l) {
                Pair pr=far[base+std::min(l,n-1)];
                as[l]=pr.a;bs[l]=pr.b;
                x[l]=nodes[pr.a].x-nodes[pr.b].x;
                y[l]=nodes[pr.a].y-nodes[pr.b].y;
                z[l]=nodes[pr.a].z-nodes[pr.b].z;
            }
            deriv_full(D,x,y,z,EPS2,order);
            for(int a=0;a<mc;++a) {
                double sign=(deg[a]&1)?-1:1;
                for(int l=0;l<4;++l) {
                    MA[a][l]=sign*moments[as[l]*nm+a];
                    MB[a][l]=moments[bs[l]*nm+a];
                }
            }
            for(int b=1;b<lc;++b) {
                int na=count[order-deg[b]];
                V la{},lb{};
                const auto& ids=sums[b];
                for(int a=0;a<na;++a) {
                    V d=D[ids[a]];
                    la+=MB[a]*d;lb+=MA[a]*d;
                }
                double sign=(deg[b]&1)?-1:1;
                for(int l=0;l<n;++l) {
                    preliminary[as[l]*lc+b]+=la[l];
                    preliminary[bs[l]*lc+b]+=sign*lb[l];
                }
            }
        }
        int save_p=P,save_nc=nc;
        std::vector<double> saved=std::move(locals);
        locals=std::move(preliminary);P=order;nc=lc;
        downward(0);
        P=save_p;nc=save_nc;locals=std::move(saved);
    }

    void make_accuracy_plans() {
        const int directions=16,ang_order=std::min(12,ph-1);
        std::vector<double> angular(directions*hc[ang_order]);
        V D[HDZERO+1];
        for(int start=0;start<directions;start+=4) {
            V x{},y{},z{};
            for(int l=0;l<4;++l) {
                int q=start+l;
                double zz=1.-2.*(q+.5)/directions;
                double angle=q*2.39996322972865332;
                double rr=std::sqrt(1-zz*zz);
                x[l]=rr*std::cos(angle);y[l]=rr*std::sin(angle);z[l]=zz;
            }
            deriv_h(D,x,y,z,ang_order);
            for(int h=0;h<hc[ang_order];++h)
                for(int l=0;l<4;++l)angular[(start+l)*hc[ang_order]+h]=D[hd[h]][l];
        }
        size_t nn=nodes.size();
        radius_h.resize(nn);radius_c.resize(nn);acc_scale.resize(nn);
        target_h.resize(nn);target_c.resize(nn);
        auto& rh=radius_h;auto& rc=radius_c;auto& scale=acc_scale;
        std::vector<double> weight(nn);
        int hn=hc[ph-1];
        for(int c=int(nn)-1;c>=0;--c) {
            const Node& nd=nodes[c];
            if(nd.left<0) {
                double w=0;
                for(int i=nd.begin;i<nd.end;++i)
                    w+=1./std::max(1.e-60,ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i]);
                weight[c]=w;
            } else weight[c]=weight[nd.left]+weight[nd.right];
            scale[c]=std::sqrt((nd.end-nd.begin)/weight[c]);
            const double* H=&hmom[c*hn];
            const double* M=&moments[c*nm];
            double r=0;
            for(int degree:{3,4,6,8,12}) {
                if(degree>ang_order)continue;
                double sum2=0;
                for(int u=0;u<directions;++u) {
                    const double* A=&angular[u*hc[ang_order]];
                    double val=0;
                    for(int h=hc[degree-1];h<hc[degree];++h)val+=H[h]*A[h];
                    sum2+=val*val;
                }
                double power=std::sqrt(sum2/directions);
                r=std::max(r,std::pow(power/M[0],1./degree));
            }
            rh[c]=std::min(nd.r,r);
            int kmax=std::min(4,(P-1)/2);
            double radial=0;
            for(int i=0;i<=kmax;++i)
                for(int j=0;j<=kmax-i;++j) {
                    int k=kmax-i-j,mi=idx[2*i][2*j][2*k];
                    double coef=fact[kmax]/(fact[i]*fact[j]*fact[k]);
                    radial+=coef*dfac[mi]*M[mi];
                }
            rc[c]=std::min(nd.r,std::pow(std::max(0.,radial/M[0]),1./(2*kmax)));
            if(weighted_targets && nd.r>0) {
                double hsum=0,csum=0,invr2=1./(nd.r*nd.r);
                for(int q=nd.begin;q<nd.end;++q) {
                    double dx=px[q]-nd.x,dy=py[q]-nd.y,dz=pz[q]-nd.z;
                    double s=(dx*dx+dy*dy+dz*dz)*invr2;
                    double s2=s*s,s4=s2*s2,s8=s4*s4,s16=s8*s8;
                    double w=1./std::max(1.e-60,ax[q]*ax[q]+ay[q]*ay[q]+az[q]*az[q]);
                    hsum+=w*s16*s4;csum+=w*s8*s4;
                }
                target_h[c]=nd.r*std::pow(std::clamp(hsum/weight[c],0.,1.),1./40);
                target_c[c]=nd.r*std::pow(std::clamp(csum/weight[c],0.,1.),1./24);
            } else target_h[c]=target_c[c]=.9*nd.r;
        }
        plans.clear();plan_roots.clear();
        plans.reserve(outer.size()*(refinement?3:1));
        for(Pair pr:outer)plan_roots.push_back(append_plan(pr,partial_refinement?0:refinement));
    }

    AccuracyPlan make_plan(Pair pr) const {
        auto log_influence=[](double influence,double source,double target) {
            double mu=20.*source/std::max(1.e-100,source+target);
            double fraction;
            if(mu<.03)fraction=mu*mu*mu/6;
            else fraction=1-std::exp(-mu)*(1+mu+.5*mu*mu);
            return std::log(std::max(1.e-300,influence*fraction));
        };
        const double invn=1./pts.size();
            const Node& A=nodes[pr.a];const Node& B=nodes[pr.b];
            double x=A.x-B.x,y=A.y-B.y,z=A.z-B.z,d2=x*x+y*y+z*z;
            double distance=std::sqrt(d2),sdistance=std::sqrt(d2+EPS2);
            double ta=target_h[pr.a],tb=target_h[pr.b];
            double tca=target_c[pr.a],tcb=target_c[pr.b];
            double fa=moments[pr.b*nm]/(d2*acc_scale[pr.a])*std::sqrt((A.end-A.begin)*invn);
            double fb=moments[pr.a*nm]/(d2*acc_scale[pr.b])*std::sqrt((B.end-B.begin)*invn);
            AccuracyPlan p;p.pair=pr;
            p.hqa=std::log(std::clamp((ta+radius_h[pr.b])/distance,1.e-20,.999));
            p.hqb=std::log(std::clamp((tb+radius_h[pr.a])/distance,1.e-20,.999));
            p.hia=log_influence(fa,radius_h[pr.b],ta);
            p.hib=log_influence(fb,radius_h[pr.a],tb);
            p.fia=std::log(std::max(1.e-300,fa));
            p.fib=std::log(std::max(1.e-300,fb));
            double soft=std::min(1.,196.*EPS2/(d2+EPS2));
            p.cqa=std::log(std::clamp((tca+radius_c[pr.b])/sdistance,1.e-20,.999));
            p.cqb=std::log(std::clamp((tcb+radius_c[pr.a])/sdistance,1.e-20,.999));
            p.cia=log_influence(fa*soft,radius_c[pr.b],tca);
            p.cib=log_influence(fb*soft,radius_c[pr.a],tcb);
            p.fcia=p.fia+std::log(soft);p.fcib=p.fib+std::log(soft);
            double na=A.end-A.begin,nb=B.end-B.begin;
#ifdef USE_EXACT_FORCE
            p.direct_cost=(uniform_mass?2.35e-9:3.5e-9)*na*nb+1.e-7;
#else
            p.direct_cost=3.5e-9*na*nb+1.e-7;
#endif
            p.low_cost=6.2e-9*(na+nb)+1.e-7;
            p.convergent=(A.r+B.r)*(A.r+B.r)<.96*.96*d2;
            return p;
    }

    int append_plan(Pair pr,int depth) {
        int index=plans.size();
        plans.push_back(make_plan(pr));
        const Node& A=nodes[pr.a];const Node& B=nodes[pr.b];
        long long product=(long long)(A.end-A.begin)*(B.end-B.begin);
        if(depth>0 && product>=4096 && (A.left>=0 || B.left>=0)) {
            Pair left,right;
            if(B.left<0 || (A.left>=0 && A.r>B.r)) {
                left={A.left,pr.b};right={A.right,pr.b};
            } else {
                left={pr.a,B.left};right={pr.a,B.right};
            }
            int c0=append_plan(left,depth-1),c1=append_plan(right,depth-1);
            plans[index].child0=c0;plans[index].child1=c1;
        }
        return index;
    }

    static int choose_order(double tolerance,double qa,double qb,double ia,double ib,int maximum) {
        double p=std::max((tolerance-ia)/qa,(tolerance-ib)/qb);
        return int(std::clamp(std::ceil(p),4.,double(maximum)));
    }

    void cost_tables(double* hcost,double* ccost,double* hfcost,double* cfcost) const {
        for(int p=4;p<=ph;++p) {
            double hw=1000+14*hc[p],cw=0,hwf=hw,cwf=0;
            for(int b=1;b<hc[p-3];++b)hw+=hc[p-deg[hlist[b]]]-hc[2];
            for(int b=1;b<count[p-3];++b)cw+=count[p-deg[b]]-count[2];
            for(int b=1;b<hc[p];++b)hwf+=hc[p-deg[hlist[b]]];
            for(int b=1;b<count[p];++b)cwf+=count[p-deg[b]];
            hcost[p]=1.9e-10*hw;
            ccost[p]=1.8e-10*cw+3.e-9*count[p];
            hfcost[p]=1.9e-10*hwf;
            cfcost[p]=1.8e-10*cwf+3.e-9*count[p];
        }
    }

    void select_orders() {
        if(lagrange){select_lagrange();return;}
        double hcost[MAXP+1]{},ccost[MAXP+1]{},hfcost[MAXP+1]{},cfcost[MAXP+1]{};
        cost_tables(hcost,ccost,hfcost,cfcost);
        std::vector<double> values(plans.size());
        std::vector<unsigned char> choices(plans.size()),hp(plans.size()),cp(plans.size());
        auto cost=[&](double tolerance,bool record) {
            for(int n=int(plans.size())-1;n>=0;--n) {
                const AccuracyPlan& p=plans[n];
                double v=p.direct_cost;int choice=0,h=0,c=0;
                double hr=std::ceil(std::max((tolerance-p.hia)/p.hqa,(tolerance-p.hib)/p.hqb));
                double cr=std::ceil(std::max((tolerance-p.cia)/p.cqa,(tolerance-p.cib)/p.cqb));
                if(p.convergent) {
                    int hh=int(std::clamp(hr,4.,double(ph)));
                    int cc=int(std::clamp(cr,4.,double(pc)));
                    double hybrid_cost=hcost[hh]+ccost[cc]+p.low_cost;
                    if(hybrid_cost<v){v=hybrid_cost;choice=1;h=hh;c=cc;}
                }
                hr=std::ceil(std::max((tolerance-p.fia)/p.hqa,(tolerance-p.fib)/p.hqb));
                cr=std::ceil(std::max((tolerance-p.fcia)/p.cqa,(tolerance-p.fcib)/p.cqb));
                if(p.convergent && hr<=ph && cr<=pc) {
                    int hh=int(std::max(4.,hr)),cc=int(std::max(4.,cr));
                    double full_cost=hfcost[hh]+cfcost[cc];
                    if(full_cost<v){v=full_cost;choice=2;h=hh;c=cc;}
                }
                if(p.child0>=0 && values[p.child0]+values[p.child1]<v) {
                    v=values[p.child0]+values[p.child1];choice=3;
                }
                values[n]=v;
                if(record){choices[n]=choice;hp[n]=h;cp[n]=c;}
            }
            double total=0;
            for(int root:plan_roots)total+=values[root];
            return total;
        };
        double calibration_start=cpu_time();
        cost(-25,false);
        double selection_reserve=(cpu_time()-calibration_start)*20;
        double all_available=std::max(.02,(deadline-cpu_time()-selection_reserve)*.98);
        double available=all_available*(partial_refinement?base_budget_fraction:1.);
        double lo=-60,hi=-2;
        if(deadline>10)hi=-27.631021115928547;
        else {
            for(int k=0;k<16;++k) {
                double mid=(lo+hi)*.5;
                if(cost(mid,false)>available)lo=mid;
                else hi=mid;
            }
        }
        chosen_tolerance=hi;predicted_work=cost(hi,true);
        auto emit=[&](auto&& self,int n)->void {
            const AccuracyPlan& p=plans[n];
            int h=hp[n],c=cp[n];
            if(choices[n]==3) {
                self(self,p.child0);self(self,p.child1);
            } else if(choices[n]==0)replace_direct.push_back(p.pair);
            else if(choices[n]==2) {
                hfull[h].push_back(p.pair);cfull[c].push_back(p.pair);
            } else {
                hybrid_pairs.push_back(p.pair);hgroup[h].push_back(p.pair);cgroup[c].push_back(p.pair);
            }
        };
        if(!partial_refinement) {
            for(int root:plan_roots)emit(emit,root);
        } else {
            std::vector<Pair> candidates;
            double fixed=0;
            for(int root:plan_roots) {
                bool capped=(choices[root]==1 || choices[root]==2) &&
                    (hp[root]>=ph-1 || cp[root]>=pc-1);
                if(capped)candidates.push_back(plans[root].pair);
                else {emit(emit,root);fixed+=values[root];}
            }
            if(!candidates.empty()) {
                std::vector<AccuracyPlan> saved_plans=std::move(plans);
                plans.clear();plan_roots.clear();
                plans.reserve(candidates.size()*3);
                for(Pair p:candidates)plan_roots.push_back(append_plan(p,std::max(1,refinement)));
                fixed_work=fixed;
                select_lagrange();
                predicted_work+=fixed;
            }
        }
    }

    void select_lagrange() {
        double hcost[MAXP+1]{},ccost[MAXP+1]{},hfcost[MAXP+1]{},cfcost[MAXP+1]{};
        cost_tables(hcost,ccost,hfcost,cfcost);
        const int hstride=ph+1,cstride=pc+1,stride=2*hstride+2*cstride;
        std::vector<double> errors(plans.size()*stride);
        std::vector<std::array<float,12>> order_estimate(plans.size());
        for(size_t i=0;i<plans.size();++i) {
            const AccuracyPlan& p=plans[i];
            double* e=&errors[i*stride];
            double ha=std::exp(2*p.fia),hb=std::exp(2*p.fib);
            double ca=std::exp(2*p.fcia),cb=std::exp(2*p.fcib);
            double hqa=std::exp(2*p.hqa),hqb=std::exp(2*p.hqb);
            double cqa=std::exp(2*p.cqa),cqb=std::exp(2*p.cqb);
            auto& g=order_estimate[i];
            g[0]=.5/p.hqa;g[1]=.5/p.hqb;g[2]=.5/p.cqa;g[3]=.5/p.cqb;
            double bh0=std::log1p(-hqa),bh1=std::log1p(-hqb);
            double bc0=std::log1p(-cqa),bc1=std::log1p(-cqb);
            g[4]=-(2*p.hia+bh0)*g[0];g[5]=-(2*p.hib+bh1)*g[1];
            g[6]=-(2*p.fia+bh0)*g[0];g[7]=-(2*p.fib+bh1)*g[1];
            g[8]=-(2*p.cia+bc0)*g[2];g[9]=-(2*p.cib+bc1)*g[3];
            g[10]=-(2*p.fcia+bc0)*g[2];g[11]=-(2*p.fcib+bc1)*g[3];
            double hfa=std::exp(2*(p.hia-p.fia)),hfb=std::exp(2*(p.hib-p.fib));
            double cfa=std::exp(2*(p.cia-p.fcia)),cfb=std::exp(2*(p.cib-p.fcib));
            for(int k=0;k<=ph;++k) {
                e[k]=ha*hfa+hb*hfb;e[hstride+k]=ha+hb;
                ha*=hqa;hb*=hqb;
            }
            for(int k=0;k<=pc;++k) {
                e[2*hstride+k]=ca*cfa+cb*cfb;
                e[2*hstride+cstride+k]=ca+cb;
                ca*=cqa;cb*=cqb;
            }
        }
        auto optimal_order=[](const double* error,const double* cost,int maximum,double mu,float guess) {
            int p=int(std::clamp(std::ceil(double(guess)),4.,double(maximum)));
            while(p<maximum && error[p]-error[p+1]>mu*(cost[p+1]-cost[p]))++p;
            while(p>4 && error[p-1]-error[p]<=mu*(cost[p]-cost[p-1]))--p;
            return p;
        };
        std::vector<double> utility(plans.size()),times(plans.size());
        std::vector<unsigned char> choices(plans.size()),hp(plans.size()),cp(plans.size());
        int href=std::min(ph-1,18),cref=std::min(pc-1,12);
        double hd=std::log(hcost[href+1]-hcost[href]);
        double hfd=std::log(hfcost[href+1]-hfcost[href]);
        double cd=std::log(ccost[cref+1]-ccost[cref]);
        double cfd=std::log(cfcost[cref+1]-cfcost[cref]);
        auto cost=[&](double log_mu,bool record) {
            double mu=std::exp(log_mu);
            float hsignal=log_mu+hd,hfsignal=log_mu+hfd;
            float csignal=log_mu+cd,cfsignal=log_mu+cfd;
            for(int n=int(plans.size())-1;n>=0;--n) {
                const AccuracyPlan& p=plans[n];const double* e=&errors[n*stride];
                const auto& g=order_estimate[n];
                double t=p.direct_cost,u=mu*t;int choice=0,h=0,c=0;
                if(p.convergent) {
                    float guess=std::max(hsignal*g[0]+g[4],hsignal*g[1]+g[5]);
                    int hh=optimal_order(e,hcost,ph,mu,guess);
                    guess=std::max(csignal*g[2]+g[8],csignal*g[3]+g[9]);
                    int cc=optimal_order(e+2*hstride,ccost,pc,mu,guess);
                    double tt=hcost[hh]+ccost[cc]+p.low_cost;
                    double uu=e[hh]+e[2*hstride+cc]+mu*tt;
                    if(uu<u){u=uu;t=tt;h=hh;c=cc;choice=1;}
                    guess=std::max(hfsignal*g[0]+g[6],hfsignal*g[1]+g[7]);
                    hh=optimal_order(e+hstride,hfcost,ph,mu,guess);
                    guess=std::max(cfsignal*g[2]+g[10],cfsignal*g[3]+g[11]);
                    cc=optimal_order(e+2*hstride+cstride,cfcost,pc,mu,guess);
                    tt=hfcost[hh]+cfcost[cc];
                    uu=e[hstride+hh]+e[2*hstride+cstride+cc]+mu*tt;
                    if(uu<u){u=uu;t=tt;h=hh;c=cc;choice=2;}
                }
                if(p.child0>=0 && utility[p.child0]+utility[p.child1]<u) {
                    u=utility[p.child0]+utility[p.child1];
                    t=times[p.child0]+times[p.child1];choice=3;
                }
                utility[n]=u;times[n]=t;
                if(record){choices[n]=choice;hp[n]=h;cp[n]=c;}
            }
            double total=0,total_utility=0;
            for(int root:plan_roots){total+=times[root];total_utility+=utility[root];}
            if(record)predicted_error=std::sqrt(std::max(0.,total_utility-mu*total));
            return total;
        };
        double begin=cpu_time(),initial=cost(-35,false);
        double selection_reserve=(cpu_time()-begin)*12;
        double available=std::max(.02,(deadline-cpu_time()-selection_reserve-fixed_work)*.97);
        double lo=initial>available?-35:-65,hi=initial>available?5:-35;
        if(deadline>10)hi=-38;
        else {
            for(int k=0;k<9;++k) {
                double mid=(lo+hi)*.5;
                if(cost(mid,false)>available)lo=mid;
                else hi=mid;
            }
        }
        chosen_tolerance=hi;predicted_work=cost(hi,true);
        auto emit=[&](auto&& self,int n)->void {
            const AccuracyPlan& p=plans[n];int h=hp[n],c=cp[n];
            if(choices[n]==3){self(self,p.child0);self(self,p.child1);}
            else if(choices[n]==0)replace_direct.push_back(p.pair);
            else if(choices[n]==2){hfull[h].push_back(p.pair);cfull[c].push_back(p.pair);}
            else {
                hybrid_pairs.push_back(p.pair);
                hgroup[h].push_back(p.pair);cgroup[c].push_back(p.pair);
            }
        };
        for(int root:plan_roots)emit(emit,root);
    }

    void finish_particles() {
        if(!variable || cpu_time()>4.78 || far.empty())return;
        std::vector<double> variance(nodes.size()),node_mass(nodes.size());
        for(size_t n=0;n<nodes.size();++n)node_mass[n]=moments[n*nm];
        auto power=[](double x,int n) {
            double value=1;
            while(n){if(n&1)value*=x;x*=x;n>>=1;}
            return value;
        };
        auto add=[&](const std::vector<Pair>& pairs,int order,bool harmonic,bool full,bool difference) {
            for(size_t i=0;i<pairs.size();++i) {
                if(i%1024==0 && cpu_time()>4.81)break;
                Pair pr=pairs[i];const Node& A=nodes[pr.a];const Node& B=nodes[pr.b];
                double x=A.x-B.x,y=A.y-B.y,z=A.z-B.z,r2=x*x+y*y+z*z;
                double u=harmonic?r2:r2+EPS2;
                double inv=1/std::sqrt(u);
                double soft=difference?std::min(1.,196.*EPS2/(r2+EPS2)):1.;
                double ta=harmonic?target_h[pr.a]:target_c[pr.a];
                double tb=harmonic?target_h[pr.b]:target_c[pr.b];
                double sa=harmonic?radius_h[pr.a]:radius_c[pr.a];
                double sb=harmonic?radius_h[pr.b]:radius_c[pr.b];
                auto fraction=[&](double s,double t) {
                    if(full)return 1.;
                    double mu=20*s/std::max(1.e-100,s+t);
                    return mu<.03?mu*mu*mu/6:
                        std::max(0.,1-std::exp(-mu)*(1+mu+.5*mu*mu));
                };
                double ea=(order+1)*node_mass[pr.b]/u*soft*fraction(sb,ta)*
                    power(std::clamp((ta+sb)*inv,0.,.999),order);
                double eb=(order+1)*node_mass[pr.a]/u*soft*fraction(sa,tb)*
                    power(std::clamp((tb+sa)*inv,0.,.999),order);
                variance[pr.a]+=ea*ea;variance[pr.b]+=eb*eb;
            }
        };
        for(int p=4;p<=ph;++p) {
            add(hgroup[p],p,true,false,false);add(hfull[p],p,true,true,false);
            add(cgroup[p],p,false,false,true);add(cfull[p],p,false,true,true);
        }
        add(inner,ps,false,true,false);
        for(size_t n=0;n<nodes.size();++n)if(nodes[n].left>=0) {
            variance[nodes[n].left]+=variance[n];variance[nodes[n].right]+=variance[n];
        }
        std::vector<double> score(pts.size());
        bool any=false;
        for(size_t n=0;n<nodes.size();++n)if(nodes[n].left<0 && variance[n]>0) {
            any=true;
            for(int i=nodes[n].begin;i<nodes[n].end;++i)
                score[i]=variance[n]/std::max(1.e-60,ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i]);
        }
        if(!any || cpu_time()>4.84)return;
        std::vector<int> order(pts.size());std::iota(order.begin(),order.end(),0);
        auto compare=[&](int a,int b){return score[a]>score[b];};
        size_t count=std::min(size_t(4096),pts.size());
        if(count<order.size())std::nth_element(order.begin(),order.begin()+count,order.end(),compare);
        std::sort(order.begin(),order.begin()+count,compare);
        // Spend only the remaining time on complete direct sums.
        for(size_t k=0;k<count && cpu_time()<4.86;++k) {
            int i=order[k];Node target{};target.begin=i;target.end=i+1;
            ax[i]=ay[i]=az[i]=0;
            if(uniform_mass)direct_pair_ordered<true,true>(target,nodes[0]);
            else direct_pair_ordered<false,true>(target,nodes[0]);
            ++refined_particles;
        }
    }

    void run(bool debug) {
        double t0=cpu_time();
        nodes.reserve(pts.size()/leaf*4);
        build_tree(0,pts.size());
        int n=pts.size();
        px.resize(n);py.resize(n);pz.resize(n);mass.resize(n);
        ax.resize(n);ay.resize(n);az.resize(n);
        for(int i=0;i<n;++i) {
            px[i]=pts[i].x;py[i]=pts[i].y;pz[i]=pts[i].z;mass[i]=pts[i].m;
            if(i && mass[i]!=mass[0])uniform_mass=false;
        }
        interact(0,0);
        long long nlow=0;
        for(int pass=0;pass<8;++pass) {
            long long ndirect=0;nlow=0;
            for(Pair pr:near) {
                long long na=nodes[pr.a].end-nodes[pr.a].begin;
                long long nb=nodes[pr.b].end-nodes[pr.b].begin;
                ndirect+=na*nb*(pr.a==pr.b?1:2);
            }
            for(Pair pr:far)nlow+=nodes[pr.a].end-nodes[pr.a].begin+nodes[pr.b].end-nodes[pr.b].begin;
#ifdef USE_EXACT_FORCE
            double direct_rate=uniform_mass?1.25e-9:1.8e-9;
#else
            double direct_rate=1.8e-9;
#endif
            double baseline=variable?.9+direct_rate*ndirect:.7+direct_rate*ndirect+6.e-9*nlow;
            if(deadline>10 || baseline<=(variable?3.5:3.2) || theta>.85)break;
            theta*=1.04;far.clear();near.clear();interact(0,0);
        }
        double t2=cpu_time();
        moments.resize(nodes.size()*nm);locals.resize(nodes.size()*nc);
        hmom.resize(nodes.size()*hc[ph-1]);hlocal.resize(nodes.size()*hc[ph]);
        if(batched_translations){prepare_levels();cart_upward_batched();h_upward_batched();}
        else {upward(0);h_upward(0);}
        double t1=cpu_time();
        if(deadline<10)deadline=std::min(deadline,4.82-1.1*(t1-t2));
        for(Pair pr:far) {
            const Node& A=nodes[pr.a];const Node& B=nodes[pr.b];
            double x=A.x-B.x,y=A.y-B.y,z=A.z-B.z;
            ((x*x+y*y+z*z>=switch_r*switch_r)?outer:inner).push_back(pr);
        }
        if(!variable)low_all();
        double tl=cpu_time();
        for(Pair pr:near)direct_pair(nodes[pr.a],nodes[pr.b],pr.a==pr.b);
        double t4=cpu_time();
        double tc,t3;
        double predict_seconds=0,plan_seconds=0,select_seconds=0;
        if(variable) {
            std::vector<double> saved_ax=ax,saved_ay=ay,saved_az=az;
            double tp=cpu_time();
            predict_accelerations();double tm=cpu_time();
            make_accuracy_plans();double te=cpu_time();
            predict_seconds=tm-tp;plan_seconds=te-tm;
            ax.swap(saved_ax);ay.swap(saved_ay);az.swap(saved_az);
            correction_m2l<true>(inner,ps,false);
            double ts=cpu_time();select_orders();select_seconds=cpu_time()-ts;
            for(Pair pr:hybrid_pairs) {
                low_m2p<2>(nodes[pr.a],pr.b);low_m2p<2>(nodes[pr.b],pr.a);
            }
            for(Pair pr:replace_direct) {
                direct_pair(nodes[pr.a],nodes[pr.b],false);
            }
            for(int p=pc;p>=4;--p) {
                correction_m2l<true>(cfull[p],p,true);
                correction_m2l(cgroup[p],p,true);
            }
            tc=cpu_time();t3=tc;
            for(int p=ph;p>=4;--p) {
                harmonic_m2l<true>(hfull[p],p,deadline);
                harmonic_m2l(hgroup[p],p,deadline);
            }
        } else {
            correction_m2l(outer,pc,true);
            tc=cpu_time();
            correction_m2l(inner,ps,false);
            t3=cpu_time();
            harmonic_m2l(outer,ph,deadline);
        }
        double th=cpu_time();
        if(batched_translations){cart_downward_batched();h_downward_batched();}
        else {downward(0);h_downward(0);}
        double td=cpu_time();
        if(refine_particles)finish_particles();
        double t5=cpu_time();
        if(debug)
            fprintf(stderr,"H P=%d pc=%d ps=%d theta=%g leaf=%d switch=%g nodes=%zu far=%zu (%zu+%zu) near=%zu direct=%lld low=%lld\n"
                "up %.4f tree %.4f H %.4f C %.4f S %.4f low %.4f P2P %.4f down %.4f total %.4f\n",
                ph,pc,ps,theta,leaf,switch_r,nodes.size(),far.size(),outer.size(),inner.size(),near.size(),np2p,nlow,
                t1-t2,t2-t0,th-t3,tc-t4,t3-tc,tl-t1,t4-tl,td-th,t5-t0);
        if(debug) {
            fprintf(stderr,"H orders");
            for(int p=4;p<=ph;++p)if(horders[p])fprintf(stderr," %d:%lld",p,horders[p]);
            fprintf(stderr,"\n");
            fprintf(stderr,"C orders");
            for(int p=4;p<=ph;++p)if(corders[p])fprintf(stderr," %d:%lld",p,corders[p]);
            size_t full_count=0;for(int p=4;p<=ph;++p)full_count+=hfull[p].size();
            fprintf(stderr," tolerance %.4g predicted %.4f direct %zu hybrid %zu full %zu plans %zu/%zu refine %d\n",
                chosen_tolerance,predicted_work,replace_direct.size(),hybrid_pairs.size(),
                full_count,plans.size(),plan_roots.size(),refinement);
            fprintf(stderr,"predict %.4f plans %.4f select %.4f estimated_error %.4g\n",
                predict_seconds,plan_seconds,select_seconds,predicted_error);
            fprintf(stderr,"direct_finish %.4f particles %d\n",t5-td,refined_particles);
        }
    }
};

static bool flat_distribution(const std::vector<std::array<double,4>>& rows) {
    double mean[3]{},c[3][3]{};
    for(const auto& p:rows)for(int j=0;j<3;++j)mean[j]+=p[j+1];
    for(double& v:mean)v/=rows.size();
    for(const auto& p:rows) {
        double x[3]={p[1]-mean[0],p[2]-mean[1],p[3]-mean[2]};
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)c[i][j]+=x[i]*x[j];
    }
    for(int iter=0;iter<12;++iter) {
        int p=0,q=1;
        if(std::abs(c[0][2])>std::abs(c[p][q])){p=0;q=2;}
        if(std::abs(c[1][2])>std::abs(c[p][q])){p=1;q=2;}
        if(std::abs(c[p][q])<1.e-14*(c[0][0]+c[1][1]+c[2][2]))break;
        double angle=.5*std::atan2(2*c[p][q],c[q][q]-c[p][p]);
        double co=std::cos(angle),si=std::sin(angle);
        double pp=c[p][p],qq=c[q][q],pq=c[p][q];
        c[p][p]=co*co*pp-2*si*co*pq+si*si*qq;
        c[q][q]=si*si*pp+2*si*co*pq+co*co*qq;
        c[p][q]=c[q][p]=0;
        for(int r=0;r<3;++r)if(r!=p&&r!=q) {
            double x=c[r][p],y=c[r][q];
            c[r][p]=c[p][r]=co*x-si*y;
            c[r][q]=c[q][r]=si*x+co*y;
        }
    }
    double lo=std::min({c[0][0],c[1][1],c[2][2]});
    double hi=std::max({c[0][0],c[1][1],c[2][2]});
    return lo<.06*hi;
}

int main(int argc,char**argv) {
    if(argc<3)return 1;
    int p=argc>3?atoi(argv[3]):26;
    double theta=argc>4?atof(argv[4]):.65;
    int leaf=argc>5?atoi(argv[5]):64;
    int pc=argc>6?atoi(argv[6]):17;
    int ps=argc>7?atoi(argv[7]):17;
    double sw=argc>8?atof(argv[8]):.15;
    if(p<4||p>MAXP||pc<4||ps<4||pc>p||ps>p||leaf<1)return 1;
    FILE* in=fopen(argv[1],"rb");if(!in)return 1;
    struct stat st;if(fstat(fileno(in),&st)||st.st_size%32)return 1;
    size_t n=st.st_size/32;
    std::vector<std::array<double,4>> raw(n);
    if(fread(raw.data(),32,n,in)!=n)return 1;
    fclose(in);
    if(!n){FILE*out=fopen(argv[2],"wb");if(!out)return 1;return fclose(out)!=0;}
    bool flat=argc==3 && flat_distribution(raw);
    if(flat){p=28;pc=18;leaf=128;}
    Hybrid solver(p,theta,leaf,pc,ps,sw);
    solver.variable=true;solver.inner_radius=sw;solver.inner_theta=.15;
    solver.refinement=1;solver.partial_refinement=!flat;
    if(flat){solver.lagrange=true;solver.weighted_targets=true;}
    if(argc>9)solver.deadline=atof(argv[9]);
    if(argc>10) {
        solver.inner_radius=sw;
        solver.inner_theta=atof(argv[10]);
    }
    if(argc>11)solver.variable=atoi(argv[11])!=0;
    if(argc>12)solver.refinement=std::clamp(atoi(argv[12]),0,3);
    if(argc>13)solver.lagrange=atoi(argv[13])!=0;
    if(argc>14)solver.weighted_targets=atoi(argv[14])!=0;
    if(argc>15)solver.batched_translations=atoi(argv[15])!=0;
    if(argc>16)solver.partial_refinement=atoi(argv[16])!=0;
    if(argc>17)solver.base_budget_fraction=std::clamp(atof(argv[17]),.5,1.);
    if(argc>18)solver.refine_particles=atoi(argv[18])!=0;
    solver.pts.resize(n);
    for(size_t i=0;i<n;++i)
        solver.pts[i]={raw[i][0],raw[i][1],raw[i][2],raw[i][3],int(i)};
    raw.clear();raw.shrink_to_fit();
    solver.run(argc>3);
    std::vector<std::array<double,3>> result(n);
    for(size_t i=0;i<n;++i)
        result[solver.pts[i].id]={solver.ax[i],solver.ay[i],solver.az[i]};
    FILE*out=fopen(argv[2],"wb");if(!out)return 1;
    if(fwrite(result.data(),24,n,out)!=n)return 1;
    return fclose(out)!=0;
}
