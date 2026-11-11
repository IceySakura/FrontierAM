#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <numeric>
#include <sys/stat.h>
#include <vector>

using V = double __attribute__((vector_size(32)));
constexpr int MAXP = 28;
constexpr int MAXC = (MAXP + 1) * (MAXP + 2) * (MAXP + 3) / 6;
constexpr double EPS2 = 0.0001;

static inline __m256d inverse_sqrt(__m256d u) {
#ifdef USE_EXACT_SQRT
    return _mm256_div_pd(_mm256_set1_pd(1),_mm256_sqrt_pd(u));
#else
    __m256d inv=_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(u)));
    __m256d half=_mm256_mul_pd(u,_mm256_set1_pd(.5));
    inv=_mm256_mul_pd(inv,_mm256_fnmadd_pd(half,_mm256_mul_pd(inv,inv),_mm256_set1_pd(1.5)));
    return _mm256_mul_pd(inv,_mm256_fnmadd_pd(half,_mm256_mul_pd(inv,inv),_mm256_set1_pd(1.5)));
#endif
}

static double cpu_time() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1.e-9;
}

struct Particle {
    double m, x, y, z;
    int id;
};
static inline double coordinate(const Particle& p,int dim) {
    return dim==0?p.x:(dim==1?p.y:p.z);
}
struct Node {
    double x, y, z, r;
    int begin, end, left, right;
};
struct Pair { int a, b; };

struct Solver {
    int P, nc, nm, leaf, low;
    double theta;
    double inner_radius=0,inner_theta=0;
    bool uniform_mass=true;
    std::vector<Particle> pts;
    std::vector<Node> nodes;
    std::vector<double> moments, locals;
    std::vector<double> px, py, pz, mass, ax, ay, az;
    std::vector<Pair> far, near;
    int ix[MAXC], iy[MAXC], iz[MAXC], deg[MAXC], count[MAXP + 1];
    int idx[MAXP + 1][MAXP + 1][MAXP + 1];
    double fact[MAXP + 1];
    std::vector<std::vector<int>> sums;
    std::vector<std::vector<int>> mshift[3], lshift[3];
    long long np2p = 0;

    Solver(int p, double t, int l, int lo) : P(p), leaf(l), low(lo), theta(t) {
        std::memset(idx, -1, sizeof(idx));
        int n = 0;
        fact[0] = 1;
        for (int i = 1; i <= P; ++i) fact[i] = fact[i-1] * i;
        for (int d = 0; d <= P; ++d) {
            for (int i = 0; i <= d; ++i)
                for (int j = 0; j <= d-i; ++j) {
                    int k = d-i-j;
                    ix[n] = i; iy[n] = j; iz[n] = k; deg[n] = d;
                    idx[i][j][k] = n++;
                }
            count[d] = n;
        }
        nc = n; nm = count[P-1];
        sums.resize(nc);
        for (int b = 1; b < nc; ++b) {
            int na = count[P - deg[b]];
            sums[b].resize(na);
            for (int a = 0; a < na; ++a)
                sums[b][a] = idx[ix[a]+ix[b]][iy[a]+iy[b]][iz[a]+iz[b]];
        }
        for(int dim=0;dim<3;++dim) {
            mshift[dim].resize(nc);lshift[dim].resize(nc);
            for(int n=0;n<nc;++n) {
                int q[3]={ix[n],iy[n],iz[n]};
                for(int k=0;k<=q[dim];++k) {
                    q[dim]-=k;
                    mshift[dim][n].push_back(idx[q[0]][q[1]][q[2]]);
                    q[dim]+=k;
                }
                for(int k=0;k<=P-deg[n];++k) {
                    q[dim]+=k;
                    lshift[dim][n].push_back(idx[q[0]][q[1]][q[2]]);
                    q[dim]-=k;
                }
            }
        }
    }

    int build_tree(int begin, int end) {
        int ni = nodes.size();
        nodes.emplace_back();
        Node nd{};
        nd.begin = begin; nd.end = end;
        nd.left = nd.right = -1;
        double lo[3] = {1e300,1e300,1e300}, hi[3] = {-1e300,-1e300,-1e300};
        double m = 0;
        for (int i = begin; i < end; ++i) {
            const auto& q = pts[i];
            m += q.m;
            nd.x += q.m*q.x; nd.y += q.m*q.y; nd.z += q.m*q.z;
            lo[0] = std::min(lo[0], q.x); hi[0] = std::max(hi[0], q.x);
            lo[1] = std::min(lo[1], q.y); hi[1] = std::max(hi[1], q.y);
            lo[2] = std::min(lo[2], q.z); hi[2] = std::max(hi[2], q.z);
        }
        nd.x /= m; nd.y /= m; nd.z /= m;
        double r2 = 0;
        for (int i = begin; i < end; ++i) {
            double x = pts[i].x-nd.x, y = pts[i].y-nd.y, z = pts[i].z-nd.z;
            r2 = std::max(r2, x*x+y*y+z*z);
        }
        nd.r = std::sqrt(r2);
        if (end - begin > leaf) {
            int dim = 0;
            for (int k = 1; k < 3; ++k)
                if (hi[k]-lo[k] > hi[dim]-lo[dim]) dim = k;
            double cut = (lo[dim]+hi[dim])*0.5;
            auto split = std::partition(pts.begin()+begin, pts.begin()+end,
                [dim,cut](const Particle& a) { return coordinate(a,dim) < cut; });
            int mid = split-pts.begin();
            if (mid==begin || mid==end) {
                mid=(begin+end)/2;
                std::nth_element(pts.begin()+begin, pts.begin()+mid, pts.begin()+end,
                    [dim](const Particle& a,const Particle& b) {
                        return coordinate(a,dim)<coordinate(b,dim);
                    });
            }
            nd.left = build_tree(begin, mid);
            nd.right = build_tree(mid, end);
        }
        nodes[ni] = nd;
        return ni;
    }

    template<class T>
    void shift_m(T* dst, const T* src, T x, T y, T z) {
        T f[MAXP+1], a[MAXC], b[MAXC];
        const T s[3] = {x,y,z};
        std::memcpy(a, src, nm*sizeof(T));
        for (int dim = 0; dim < 3; ++dim) {
            f[0]=T{}+1.0;
            for (int i=1; i<P; ++i) f[i]=f[i-1]*s[dim]/i;
            for (int n=0; n<nm; ++n) {
                T v{};
                const auto& ids=mshift[dim][n];
                for (size_t k=0;k<ids.size();++k) v+=a[ids[k]]*f[k];
                b[n]=v;
            }
            std::memcpy(a,b,nm*sizeof(T));
        }
        for (int n=0;n<nm;++n) dst[n]+=a[n];
    }

    __attribute__((noinline))
    void form_leaf(int ni) {
        const Node& nd=nodes[ni];
        double* M = &moments[ni*nm];
        alignas(32) V mv[MAXC];
        std::fill_n(mv,nm,V{});
        for (int q=nd.begin;q<nd.end;q+=4) {
            V xp[MAXP+1],yp[MAXP+1],zp[MAXP+1];
            V x{},y{},z{};
            xp[0]=V{};yp[0]=zp[0]=V{1,1,1,1};
            for(int l=0;l<4;++l) {
                int qi=std::min(q+l,nd.end-1);
                xp[0][l]=q+l<nd.end?pts[qi].m:0;
                x[l]=nd.x-pts[qi].x;y[l]=nd.y-pts[qi].y;z[l]=nd.z-pts[qi].z;
            }
            for (int i=1;i<P;++i) {
                xp[i]=xp[i-1]*x*(1./i);yp[i]=yp[i-1]*y*(1./i);zp[i]=zp[i-1]*z*(1./i);
            }
            for (int a=0;a<nm;++a)
                mv[a]+=xp[ix[a]]*yp[iy[a]]*zp[iz[a]];
        }
        for(int a=0;a<nm;++a) M[a]=mv[a][0]+mv[a][1]+mv[a][2]+mv[a][3];
    }

    void upward(int ni) {
        const Node& nd=nodes[ni];
        if (nd.left<0)form_leaf(ni);
        else {
            upward(nd.left); upward(nd.right);
            double* M=&moments[ni*nm];
            for (int c : {nd.left,nd.right}) {
                const Node& ch=nodes[c];
                shift_m(M,&moments[c*nm],nd.x-ch.x,nd.y-ch.y,nd.z-ch.z);
            }
        }
    }

    void interact(int a, int b) {
        const Node& A=nodes[a]; const Node& B=nodes[b];
        if (a==b) {
            if (A.left<0) { near.push_back({a,b}); return; }
            interact(A.left,A.left); interact(A.left,A.right); interact(A.right,A.right);
            return;
        }
        double x=A.x-B.x,y=A.y-B.y,z=A.z-B.z;
        double r=A.r+B.r;
        double theta_here=theta;
        if(inner_radius>0 && x*x+y*y+z*z<inner_radius*inner_radius)theta_here=inner_theta;
        if (r*r < theta_here*theta_here*(x*x+y*y+z*z+EPS2)) {
            far.push_back({a,b}); return;
        }
        if (A.left<0 && B.left<0) { near.push_back({a,b}); return; }
        if (B.left<0 || (A.left>=0 && A.r>B.r)) {
            interact(A.left,b); interact(A.right,b);
        } else {
            interact(a,B.left); interact(a,B.right);
        }
    }

    void deriv(V* D, V x, V y, V z) {
        V u=x*x+y*y+z*z+V{EPS2,EPS2,EPS2,EPS2};
        V inv=V{1,1,1,1}/u;
        V v=_mm256_sqrt_pd(u);
        D[0]=V{1,1,1,1}/v;
        V xx=x*inv,yy=y*inv,zz=z*inv;
        for (int n=1;n<nc;++n) {
            int i=ix[n],j=iy[n],k=iz[n],d=deg[n];
            V a{},b{};
            if (i) a+=xx*D[idx[i-1][j][k]];
            if (j) a+=yy*D[idx[i][j-1][k]];
            if (k) a+=zz*D[idx[i][j][k-1]];
            if (i>1) b+=D[idx[i-2][j][k]];
            if (j>1) b+=D[idx[i][j-2][k]];
            if (k>1) b+=D[idx[i][j][k-2]];
            D[n]=a*((1.-2*d)/d)+b*inv*((1.-d)/d);
        }
        for (int n=1;n<nc;++n) D[n]*=fact[ix[n]]*fact[iy[n]]*fact[iz[n]];
    }

    void m2l_all() {
        alignas(32) V D[MAXC], MA[MAXC], MB[MAXC];
        for (size_t base=0;base<far.size();base+=4) {
            int as[4],bs[4]; V x{},y{},z{};
            int n=std::min(size_t(4),far.size()-base);
            for (int l=0;l<4;++l) {
                auto pair=far[base+std::min(l,n-1)];
                as[l]=pair.a;bs[l]=pair.b;
                x[l]=nodes[pair.a].x-nodes[pair.b].x;
                y[l]=nodes[pair.a].y-nodes[pair.b].y;
                z[l]=nodes[pair.a].z-nodes[pair.b].z;
            }
            deriv(D,x,y,z);
            for (int a=0;a<nm;++a) {
                double s=(deg[a]&1)?-1:1;
                for (int l=0;l<4;++l) {
                    MA[a][l]=s*moments[as[l]*nm+a];
                    MB[a][l]=moments[bs[l]*nm+a];
                }
            }
            for (int b=1;b<nc;++b) {
                V la{},lb{},la1{},lb1{},la2{},lb2{},la3{},lb3{};
                const auto& inds=sums[b];
                size_t a=low<0?0:count[low];
                for (;a+4<=inds.size();a+=4) {
                    V d=D[inds[a]];
                    la+=MB[a]*d;lb+=MA[a]*d;
                    d=D[inds[a+1]];
                    la1+=MB[a+1]*d;lb1+=MA[a+1]*d;
                    d=D[inds[a+2]];
                    la2+=MB[a+2]*d;lb2+=MA[a+2]*d;
                    d=D[inds[a+3]];
                    la3+=MB[a+3]*d;lb3+=MA[a+3]*d;
                }
                la=(la+la1)+(la2+la3);lb=(lb+lb1)+(lb2+lb3);
                for (;a<inds.size();++a) {
                    V d=D[inds[a]];
                    la+=MB[a]*d;lb+=MA[a]*d;
                }
                double s=(deg[b]&1)?-1:1;
                for (int l=0;l<n;++l) {
                    locals[as[l]*nc+b]+=la[l];
                    locals[bs[l]*nc+b]+=s*lb[l];
                }
            }
        }
    }

    template<int LOW,bool NEGATIVE=false>
    void low_m2p(const Node& T, int si) {
        const Node& S=nodes[si];
        const double* M=&moments[si*nm];
        auto splat=[](double x)->V{return _mm256_set1_pd(x);};
        V sx=splat(S.x),sy=splat(S.y),sz=splat(S.z),m=splat(M[0]);
        V tx{},ty{},tz{},txy{},txz{},tyz{},trace{};
        V dx0{},dy0{},dz0{};
        if constexpr(LOW>=1) {
            dx0=splat(M[idx[1][0][0]]);dy0=splat(M[idx[0][1][0]]);dz0=splat(M[idx[0][0][1]]);
        }
        if constexpr(LOW>=2) {
            tx=splat(2*M[idx[2][0][0]]);ty=splat(2*M[idx[0][2][0]]);tz=splat(2*M[idx[0][0][2]]);
            txy=splat(M[idx[1][1][0]]);txz=splat(M[idx[1][0][1]]);tyz=splat(M[idx[0][1][1]]);
            trace=tx+ty+tz;
        }
        for(int i=T.begin;i<T.end;i+=4) {
            V x{},y{},z{};
            if(i+4<=T.end) {
                x=_mm256_loadu_pd(&px[i]);y=_mm256_loadu_pd(&py[i]);z=_mm256_loadu_pd(&pz[i]);
            } else {
                for(int l=0;l<4;++l) {
                    int q=std::min(i+l,T.end-1);
                    x[l]=px[q];y[l]=py[q];z[l]=pz[q];
                }
            }
            x-=sx;y-=sy;z-=sz;
            V u=x*x+y*y+z*z+splat(EPS2);
            V inv=inverse_sqrt(u);
            V inv2=inv*inv,inv3=inv*inv2;
            V f=-m*inv3;
            V a=x*f,b=y*f,c=z*f;
            if constexpr(LOW>=1) {
                V q=(dx0*x+dy0*y+dz0*z)*(3.0*inv2);
                a+=(q*x-dx0)*inv3;b+=(q*y-dy0)*inv3;c+=(q*z-dz0)*inv3;
            }
            if constexpr(LOW>=2) {
                V qx=tx*x+txy*y+txz*z;
                V qy=txy*x+ty*y+tyz*z;
                V qz=txz*x+tyz*y+tz*z;
                V q=x*qx+y*qy+z*qz;
                V f2=trace*1.5-q*inv2*7.5;
                V inv5=inv3*inv2;
                a+=(3.0*qx+f2*x)*inv5;
                b+=(3.0*qy+f2*y)*inv5;
                c+=(3.0*qz+f2*z)*inv5;
            }
            if constexpr(NEGATIVE){a=-a;b=-b;c=-c;}
            if(i+4<=T.end) {
                _mm256_storeu_pd(&ax[i],a+V(_mm256_loadu_pd(&ax[i])));
                _mm256_storeu_pd(&ay[i],b+V(_mm256_loadu_pd(&ay[i])));
                _mm256_storeu_pd(&az[i],c+V(_mm256_loadu_pd(&az[i])));
            } else {
                for(int l=0;l<4&&i+l<T.end;++l) {
                    ax[i+l]+=a[l];ay[i+l]+=b[l];az[i+l]+=c[l];
                }
            }
        }
    }

    void low_all() {
        if(low<0)return;
        for(auto pair:far) {
            if(low==0) {
                low_m2p<0>(nodes[pair.a],pair.b);low_m2p<0>(nodes[pair.b],pair.a);
            } else if(low==1) {
                low_m2p<1>(nodes[pair.a],pair.b);low_m2p<1>(nodes[pair.b],pair.a);
            } else {
                low_m2p<2>(nodes[pair.a],pair.b);low_m2p<2>(nodes[pair.b],pair.a);
            }
        }
    }

    template<bool EQUAL,bool SAME>
    void direct_pair_ordered(const Node& A,const Node& B) {
        int a0=A.begin,a1=A.end,b0=B.begin,b1=B.end;
        np2p+=(long long)(a1-a0)*(b1-b0)*(SAME?1:2);
        for (int i=a0;i<a1;++i) {
            __m256d xi=_mm256_set1_pd(px[i]),yi=_mm256_set1_pd(py[i]),zi=_mm256_set1_pd(pz[i]);
            __m256d mi=_mm256_set1_pd(mass[i]);
            __m256d ai{},bi{},ci{};
            int j=b0;
            for (;j+4<=b1;j+=4) {
                __m256d dx=_mm256_sub_pd(_mm256_loadu_pd(&px[j]),xi);
                __m256d dy=_mm256_sub_pd(_mm256_loadu_pd(&py[j]),yi);
                __m256d dz=_mm256_sub_pd(_mm256_loadu_pd(&pz[j]),zi);
                __m256d r2=_mm256_fmadd_pd(dx,dx,_mm256_set1_pd(EPS2));
                r2=_mm256_fmadd_pd(dy,dy,r2);r2=_mm256_fmadd_pd(dz,dz,r2);
                __m256d f{},fj;
#ifdef USE_EXACT_FORCE
                if constexpr(EQUAL) {
                    fj=_mm256_div_pd(mi,_mm256_mul_pd(r2,_mm256_sqrt_pd(r2)));
                } else {
                    __m256d inv=inverse_sqrt(r2);
                    f=_mm256_mul_pd(_mm256_mul_pd(inv,inv),inv);
                    fj=_mm256_mul_pd(f,_mm256_loadu_pd(&mass[j]));
                }
#else
                __m256d inv=inverse_sqrt(r2);
                f=_mm256_mul_pd(_mm256_mul_pd(inv,inv),inv);
                if constexpr(EQUAL)fj=_mm256_mul_pd(f,mi);
                else fj=_mm256_mul_pd(f,_mm256_loadu_pd(&mass[j]));
#endif
                ai=_mm256_fmadd_pd(dx,fj,ai);bi=_mm256_fmadd_pd(dy,fj,bi);ci=_mm256_fmadd_pd(dz,fj,ci);
                if constexpr(!SAME) {
                    __m256d fi;
                    if constexpr(EQUAL)fi=fj;
                    else fi=_mm256_mul_pd(f,mi);
                    _mm256_storeu_pd(&ax[j],_mm256_fnmadd_pd(dx,fi,_mm256_loadu_pd(&ax[j])));
                    _mm256_storeu_pd(&ay[j],_mm256_fnmadd_pd(dy,fi,_mm256_loadu_pd(&ay[j])));
                    _mm256_storeu_pd(&az[j],_mm256_fnmadd_pd(dz,fi,_mm256_loadu_pd(&az[j])));
                }
            }
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp,ai); double sx=tmp[0]+tmp[1]+tmp[2]+tmp[3];
            _mm256_store_pd(tmp,bi); double sy=tmp[0]+tmp[1]+tmp[2]+tmp[3];
            _mm256_store_pd(tmp,ci); double sz=tmp[0]+tmp[1]+tmp[2]+tmp[3];
            for (;j<b1;++j) {
                double dx=px[j]-px[i],dy=py[j]-py[i],dz=pz[j]-pz[i];
                double r2=dx*dx+dy*dy+dz*dz+EPS2;
                double inv=1/std::sqrt(r2),f=inv*inv*inv;
                sx+=dx*f*mass[j];sy+=dy*f*mass[j];sz+=dz*f*mass[j];
                if constexpr(!SAME) {
                    ax[j]-=dx*f*mass[i];ay[j]-=dy*f*mass[i];az[j]-=dz*f*mass[i];
                }
            }
            ax[i]+=sx;ay[i]+=sy;az[i]+=sz;
        }
    }

    void direct_pair(const Node& A,const Node& B,bool same) {
        if(same) {
            if(uniform_mass)direct_pair_ordered<true,true>(A,B);
            else direct_pair_ordered<false,true>(A,B);
        } else {
            const Node* a=&A;const Node* b=&B;
            if(b->end-b->begin<a->end-a->begin)std::swap(a,b);
            if(uniform_mass)direct_pair_ordered<true,false>(*a,*b);
            else direct_pair_ordered<false,false>(*a,*b);
        }
    }

    template<class T>
    void shift_l(T* dst,const T* src,T x,T y,T z) {
        T f[MAXP+1],a[MAXC],b[MAXC];
        const T s[3]={x,y,z};
        std::memcpy(a,src,nc*sizeof(T));
        for (int dim=0;dim<3;++dim) {
            f[0]=T{}+1.0;
            for (int i=1;i<=P;++i) f[i]=f[i-1]*s[dim]/i;
            for (int n=1;n<nc;++n) {
                T v{};
                const auto& ids=lshift[dim][n];
                for (int k=0;k<=P-deg[n];++k) v+=a[ids[k]]*f[k];
                b[n]=v;
            }
            b[0]=T{};
            std::memcpy(a,b,nc*sizeof(T));
        }
        for (int n=1;n<nc;++n) dst[n]+=a[n];
    }

    __attribute__((noinline))
    void evaluate_leaf(int ni) {
        const Node& nd=nodes[ni];
        double* L=&locals[ni*nc];
            for (int q=nd.begin;q<nd.end;q+=4) {
                V xp[MAXP+1],yp[MAXP+1],zp[MAXP+1];
                xp[0]=yp[0]=zp[0]=V{1,1,1,1};
                V x{},y{},z{};
                for(int l=0;l<4;++l) {
                    int qi=std::min(q+l,nd.end-1);
                    x[l]=px[qi]-nd.x;y[l]=py[qi]-nd.y;z[l]=pz[qi]-nd.z;
                }
                for (int i=1;i<P;++i) {
                    xp[i]=xp[i-1]*x*(1./i);yp[i]=yp[i-1]*y*(1./i);zp[i]=zp[i-1]*z*(1./i);
                }
                V a{},b{},c{};
                for (int n=1;n<nc;++n) {
                    int i=ix[n],j=iy[n],k=iz[n];
                    if (i) a+=L[n]*xp[i-1]*yp[j]*zp[k];
                    if (j) b+=L[n]*xp[i]*yp[j-1]*zp[k];
                    if (k) c+=L[n]*xp[i]*yp[j]*zp[k-1];
                }
                for(int l=0;l<4&&q+l<nd.end;++l) {
                    ax[q+l]+=a[l];ay[q+l]+=b[l];az[q+l]+=c[l];
                }
            }
    }

    void downward(int ni) {
        const Node& nd=nodes[ni];
        double* L=&locals[ni*nc];
        if (nd.left>=0) {
            for (int c : {nd.left,nd.right}) {
                const Node& ch=nodes[c];
                shift_l(&locals[c*nc],L,ch.x-nd.x,ch.y-nd.y,ch.z-nd.z);
                downward(c);
            }
        } else {
            evaluate_leaf(ni);
        }
    }

    void run(bool debug) {
        double t0=cpu_time();
        nodes.reserve(pts.size()/leaf*3);
        build_tree(0,pts.size());
        int n=pts.size();
        px.resize(n);py.resize(n);pz.resize(n);mass.resize(n);
        ax.resize(n);ay.resize(n);az.resize(n);
        for(int i=0;i<n;++i) {
            px[i]=pts[i].x;py[i]=pts[i].y;pz[i]=pts[i].z;mass[i]=pts[i].m;
            if(i && mass[i]!=mass[0])uniform_mass=false;
        }
        moments.resize(nodes.size()*nm);locals.resize(nodes.size()*nc);
        upward(0);
        double t1=cpu_time();
        interact(0,0);
        double t2=cpu_time();
        m2l_all();
        double t3=cpu_time();
        low_all();
        double tl=cpu_time();
        for(auto pair:near) direct_pair(nodes[pair.a],nodes[pair.b],pair.a==pair.b);
        double t4=cpu_time();
        downward(0);
        double t5=cpu_time();
        if(debug)
            std::fprintf(stderr,"P=%d theta=%g leaf=%d low=%d nodes=%zu far=%zu near=%zu direct=%lld\n"
                "build/up %.4f tree %.4f M2L %.4f low %.4f P2P %.4f down %.4f total %.4f\n",
                P,theta,leaf,low,nodes.size(),far.size(),near.size(),np2p,
                t1-t0,t2-t1,t3-t2,tl-t3,t4-tl,t5-t4,t5-t0);
    }
};

int main(int argc,char** argv) {
    if(argc<3) return 1;
    int p=argc>3?std::atoi(argv[3]):10;
    double theta=argc>4?std::atof(argv[4]):0.5;
    int leaf=argc>5?std::atoi(argv[5]):64;
    int low=argc>6?std::atoi(argv[6]):-1;
    if(p<2||p>MAXP||leaf<1) return 1;
    FILE* in=std::fopen(argv[1],"rb");
    if(!in) return 1;
    struct stat st;
    if(fstat(fileno(in),&st)||st.st_size%32) return 1;
    size_t n=st.st_size/32;
    std::vector<std::array<double,4>> raw(n);
    if(fread(raw.data(),32,n,in)!=n) return 1;
    fclose(in);
    if(!n) {
        FILE* out=fopen(argv[2],"wb");if(!out) return 1;fclose(out);return 0;
    }
    Solver solver(p,theta,leaf,low);
    solver.pts.resize(n);
    for(size_t i=0;i<n;++i)
        solver.pts[i]={raw[i][0],raw[i][1],raw[i][2],raw[i][3],int(i)};
    raw.clear();raw.shrink_to_fit();
    solver.run(argc>3);
    std::vector<std::array<double,3>> result(n);
    for(size_t i=0;i<n;++i)
        result[solver.pts[i].id]={solver.ax[i],solver.ay[i],solver.az[i]};
    FILE* out=std::fopen(argv[2],"wb");if(!out) return 1;
    if(fwrite(result.data(),24,n,out)!=n) return 1;
    return fclose(out)!=0;
}
