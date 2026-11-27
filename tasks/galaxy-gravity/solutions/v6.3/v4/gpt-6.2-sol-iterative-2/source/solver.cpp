#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace galaxy {

constexpr double eps2 = 0.0001;
constexpr int cap = 24;
constexpr int stride = cap + 1;
constexpr int maxcoef = 4600;

double cpu_time() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

double env(const char *key, double def) {
    const char *s = std::getenv(key);
    return s ? std::atof(s) : def;
}

struct Particle {
    double x[3];
    int original;
};

struct Node {
    double c[3], radius;
    int lo, hi, left = -1, right = -1;
    int size() const { return hi - lo; }
    bool leaf() const { return left < 0; }
};

struct Tree {
    int n;
    double mass;
    std::vector<Particle> pts;
    std::vector<int> order;
    std::vector<Node> nodes;
    std::vector<double> xyz[3], acc[3];
    int height = 0;
    int geometry = (int)env("GALAXY_GEOMETRY",2.);
    bool rotated = false;
    double frame[3][3]{};
    explicit Tree(int n) : n(n), pts(n), order(n) {
        nodes.reserve(4*(n/32+1));
    }
    void align_thin() {
        const int ns=std::min(n,4096);
        double c[3]{};
        for(int i=0;i<ns;++i){
            const Particle &p=pts[size_t(i)*n/ns];
            for(int j=0;j<3;++j)c[j]+=p.x[j];
        }
        for(double &v:c)v/=ns;
        double a[3][3]{},q[3][3]{};
        for(int i=0;i<ns;++i){
            const Particle &p=pts[size_t(i)*n/ns];
            double d[3]={p.x[0]-c[0],p.x[1]-c[1],p.x[2]-c[2]};
            for(int j=0;j<3;++j)for(int k=0;k<=j;++k)a[j][k]+=d[j]*d[k];
        }
        for(int j=0;j<3;++j){
            q[j][j]=1.;
            for(int k=0;k<j;++k)a[k][j]=a[j][k];
        }
        for(int i=0;i<20;++i){
            int j=0,k=1;
            if(std::abs(a[0][2])>std::abs(a[j][k]))j=0,k=2;
            if(std::abs(a[1][2])>std::abs(a[j][k]))j=1,k=2;
            const double b=a[j][k];
            if(std::abs(b)<=1e-15*(a[0][0]+a[1][1]+a[2][2]))break;
            const double tau=(a[k][k]-a[j][j])/(2.*b);
            const double s0=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau));
            const double f=1./std::sqrt(1.+s0*s0),s=f*s0;
            a[j][j]-=s0*b;
            a[k][k]+=s0*b;
            a[j][k]=a[k][j]=0.;
            for(int l=0;l<3;++l){
                if(l!=j&&l!=k){
                    const double x=a[l][j],y=a[l][k];
                    a[j][l]=a[l][j]=f*x-s*y;
                    a[k][l]=a[l][k]=s*x+f*y;
                }
                const double u=q[l][j],v=q[l][k];
                q[l][j]=f*u-s*v; q[l][k]=s*u+f*v;
            }
        }
        int k[3]={0,1,2};
        std::sort(k,k+3,[&](int i,int j){return a[i][i]>a[j][j];});
        if(a[k[2]][k[2]]>=.082*a[k[1]][k[1]])return;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)frame[i][j]=q[j][k[i]];
        rotated=true;
        for(Particle &p:pts){
            const double d[3]={p.x[0]-c[0],p.x[1]-c[1],p.x[2]-c[2]};
            for(int j=0;j<3;++j)
                p.x[j]=frame[j][0]*d[0]+frame[j][1]*d[1]+frame[j][2]*d[2];
        }
    }
    int subdivide(int lo, int hi, int leaf, int depth) {
        height = std::max(height, depth);
        Node box;
        box.lo = lo;
        box.hi = hi;
        double low[3], high[3];
        double avg[3]={pts[lo].x[0],pts[lo].x[1],pts[lo].x[2]};
        int imin[3]={lo,lo,lo},imax[3]={lo,lo,lo};
        for (int k = 0; k < 3; ++k) low[k] = high[k] = pts[lo].x[k];
        for (int i = lo+1; i < hi; ++i) {
            for (int k = 0; k < 3; ++k) {
                const double v=pts[i].x[k];
                if(v<low[k]){low[k]=v;imin[k]=i;}
                if(v>high[k]){high[k]=v;imax[k]=i;}
                avg[k]+=v;
            }
        }
        int kmax = 0;
        for (int k = 0; k < 3; ++k) {
            box.c[k] = low[k] + 0.5*(high[k]-low[k]);
            if (high[k]-low[k] > high[kmax]-low[kmax]) kmax = k;
        }
        const double mid=box.c[kmax];
        if(geometry>=1) for (int k=0;k<3;++k) avg[k]/=hi-lo;
        const double *x=pts[imin[kmax]].x,*y=pts[imax[kmax]].x;
        double ctr[3]={0.5*(x[0]+y[0]),0.5*(x[1]+y[1]),0.5*(x[2]+y[2])};
        double rad2 = 0.0, rmean=0., rfar=0.;
        for (int i = lo; i < hi; ++i) {
            const double *p=pts[i].x;
            const double dx=p[0]-box.c[0],dy=p[1]-box.c[1],dz=p[2]-box.c[2];
            rad2=std::max(rad2,dx*dx+dy*dy+dz*dz);
            if(geometry>=1){
                const double a=p[0]-avg[0],b=p[1]-avg[1],c=p[2]-avg[2];
                rmean=std::max(rmean,a*a+b*b+c*c);
            }
            if(geometry>=2){
                const double a=p[0]-ctr[0],b=p[1]-ctr[1],c=p[2]-ctr[2];
                rfar=std::max(rfar,a*a+b*b+c*c);
            }
        }
        if(geometry>=2 && rfar<rad2){
            rad2=rfar;
            std::copy_n(ctr,3,box.c);
        }
        if(geometry>=1 && rmean<=1.015*rad2){
            rad2=rmean;
            std::copy_n(avg,3,box.c);
        }
        box.radius = std::sqrt(rad2);
        int a = (int)nodes.size();
        nodes.push_back(box);
        if (hi-lo > leaf && depth < 70) {
            int sep = std::partition(pts.begin()+lo, pts.begin()+hi,
                [kmax,mid](const Particle &p) { return p.x[kmax] < mid; })-pts.begin();
            if (sep == lo || sep == hi) sep = (lo+hi)/2;
            const int l = subdivide(lo, sep, leaf, depth+1);
            const int r = subdivide(sep, hi, leaf, depth+1);
            nodes[a].left = l;
            nodes[a].right = r;
        }
        return a;
    }
    void build(int leaf) {
        subdivide(0, n, leaf, 0);
        for (int k = 0; k < 3; ++k) {
            xyz[k].resize(n+4);
            acc[k].resize(n+4);
        }
        for (int i = 0; i < n; ++i) {
            order[i] = pts[i].original;
            for (int k = 0; k < 3; ++k) xyz[k][i] = pts[i].x[k];
        }
        std::vector<Particle>().swap(pts);
    }
    void approx_norm(int target,int src,double *sum) const {
        const Node &a=nodes[target],&b=nodes[src];
        const double x=b.c[0]-a.c[0],y=b.c[1]-a.c[1],z=b.c[2]-a.c[2];
        const double q=x*x+y*y+z*z+eps2;
        if(b.leaf() || b.radius*b.radius<.46*q) {
            const double qsafe=std::max(q,1.75*b.radius*b.radius);
            const double f=b.size()/qsafe, fi=f/std::sqrt(qsafe);
            sum[0]+=x*fi; sum[1]+=y*fi; sum[2]+=z*fi;
            sum[3]+=f;
            sum[4]+=std::min(f, a.radius*fi);
        } else {
            approx_norm(target,b.left,sum);
            approx_norm(target,b.right,sum);
        }
    }
    std::vector<double> inverse_norms() const {
        std::vector<double> out(nodes.size());
        for(size_t i=0;i<nodes.size();++i){
            double v[5]{};
            approx_norm(i,0,v);
            const double field=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            out[i]=1./std::max(field+.28*v[4],.045*v[3]);
        }
        return out;
    }
};

using V = __m256d;
inline V splat(double x) { return _mm256_set1_pd(x); }
inline V zero() { return _mm256_setzero_pd(); }
inline V load(const double *x) { return _mm256_loadu_pd(x); }
inline void store(double *x, V a) { _mm256_storeu_pd(x, a); }
inline V add(V a, V b) { return _mm256_add_pd(a,b); }
inline V sub(V a, V b) { return _mm256_sub_pd(a,b); }
inline V mul(V a, V b) { return _mm256_mul_pd(a,b); }
inline V fmadd(V a, V b, V c) { return _mm256_fmadd_pd(a,b,c); }
inline V fnmadd(V a, V b, V c) { return _mm256_fnmadd_pd(a,b,c); }
inline double sum(V x) {
    const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x,1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s,s)));
}

template<int kind=0> inline V inv_cube(V q) {
    if constexpr(kind==2){
        const V r=_mm256_div_pd(splat(1.),_mm256_sqrt_pd(q));
        return mul(mul(r,r),r);
    }
    const V r = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(q)));
    const V r2 = mul(r,r);
    const V e = fnmadd(r2, q, splat(1.));
    // (1-e)^(-3/2) through fourth order; |e| < 0.0008.
    const V a = kind==1 ? splat(35./16.) : fmadd(e, splat(315./128.), splat(35./16.));
    const V b = fmadd(e, a, splat(15./8.));
    const V c = fmadd(e, b, splat(1.5));
    return mul(mul(r2,r), fmadd(e,c,splat(1.)));
}

struct Direct {
    const double *x, *y, *z;
    double *ax, *ay, *az;
    int version;
    explicit Direct(Tree &t) : x(t.xyz[0].data()), y(t.xyz[1].data()), z(t.xyz[2].data()),
        ax(t.acc[0].data()), ay(t.acc[1].data()), az(t.acc[2].data()),
        version(int(env("GALAXY_DIRECT",1))) {}
    template<int kind, bool masked=false>
    __attribute__((noinline)) static void solve_block(const double * __restrict x,
           const double * __restrict y,const double * __restrict z,
           double * __restrict ax,double * __restrict ay,double * __restrict az,
           int al,int ah,int bl,int bh,bool same){
        for(int i=al;i<ah;++i){
            const double xx=x[i], yy=y[i], zz=z[i];
            const V xi=splat(xx),yi=splat(yy),zi=splat(zz);
            V tx=zero(),ty=zero(),tz=zero();
            int j=same?i+1:bl;
            for(;j+4<=bh;j+=4){
                const V dx=sub(load(x+j),xi),dy=sub(load(y+j),yi),dz=sub(load(z+j),zi);
                const V q=fmadd(dx,dx,fmadd(dy,dy,fmadd(dz,dz,splat(eps2))));
                const V f=inv_cube<kind>(q);
                tx=fmadd(dx,f,tx); ty=fmadd(dy,f,ty); tz=fmadd(dz,f,tz);
                store(ax+j,fnmadd(dx,f,load(ax+j)));
                store(ay+j,fnmadd(dy,f,load(ay+j)));
                store(az+j,fnmadd(dz,f,load(az+j)));
            }
            if constexpr(masked){
                if(j<bh){
                    const int left=bh-j;
                    const V mask=_mm256_castsi256_pd(_mm256_set_epi64x(
                        0,left>=3?-1LL:0LL,left>=2?-1LL:0LL,-1LL));
                    const V dx=sub(load(x+j),xi),dy=sub(load(y+j),yi),dz=sub(load(z+j),zi);
                    const V q=fmadd(dx,dx,fmadd(dy,dy,fmadd(dz,dz,splat(eps2))));
                    const V f=_mm256_and_pd(mask,inv_cube<kind>(q));
                    tx=fmadd(dx,f,tx);ty=fmadd(dy,f,ty);tz=fmadd(dz,f,tz);
                    store(ax+j,fnmadd(dx,f,load(ax+j)));
                    store(ay+j,fnmadd(dy,f,load(ay+j)));
                    store(az+j,fnmadd(dz,f,load(az+j)));
                }
                ax[i]+=sum(tx);ay[i]+=sum(ty);az[i]+=sum(tz);
                continue;
            }
            double ux=sum(tx),uy=sum(ty),uz=sum(tz);
            for(;j<bh;++j){
                const double dx=x[j]-xx,dy=y[j]-yy,dz=z[j]-zz;
                const double q=dx*dx+dy*dy+dz*dz+eps2, f=1./(q*std::sqrt(q));
                ux+=dx*f; uy+=dy*f; uz+=dz*f;
                ax[j]-=dx*f; ay[j]-=dy*f; az[j]-=dz*f;
            }
            ax[i]+=ux; ay[i]+=uy; az[i]+=uz;
        }
    }
    __attribute__((noinline)) void block(int al, int ah, int bl, int bh, bool same) const {
        for (int i = al; i < ah; ++i) {
            const V xi = splat(x[i]), yi = splat(y[i]), zi = splat(z[i]);
            V tx = zero(), ty = zero(), tz = zero();
            int j = same ? i+1 : bl;
            for (; j+4 <= bh; j+=4) {
                const V dx = sub(load(x+j), xi), dy = sub(load(y+j), yi),
                        dz = sub(load(z+j), zi);
                const V q = fmadd(dx,dx,fmadd(dy,dy,fmadd(dz,dz,splat(eps2))));
                const V f = inv_cube(q);
                tx = fmadd(dx,f,tx);
                ty = fmadd(dy,f,ty);
                tz = fmadd(dz,f,tz);
                store(ax+j, fnmadd(dx,f,load(ax+j)));
                store(ay+j, fnmadd(dy,f,load(ay+j)));
                store(az+j, fnmadd(dz,f,load(az+j)));
            }
            double sx = sum(tx), sy = sum(ty), sz = sum(tz);
            for (; j < bh; ++j) {
                const double dx = x[j]-x[i], dy = y[j]-y[i], dz = z[j]-z[i];
                const double q = dx*dx+dy*dy+dz*dz+eps2;
                const double f = 1.0/(q*std::sqrt(q));
                sx += dx*f; sy += dy*f; sz += dz*f;
                ax[j] -= dx*f; ay[j] -= dy*f; az[j] -= dz*f;
            }
            ax[i] += sx; ay[i] += sy; az[i] += sz;
        }
    }
    static __m128d pair_sum(V a) {
        return _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a,1));
    }
    static __m128d adjacent_sum(V a) {
        const V s = _mm256_hadd_pd(a,a);
        return _mm_unpacklo_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s,1));
    }
    static V duplicate2(const double *p) {
        V a = _mm256_castpd128_pd256(_mm_loadu_pd(p));
        return _mm256_permute4x64_pd(a, _MM_SHUFFLE(1,1,0,0));
    }
    __attribute__((noinline)) void block2(int al, int ah, int bl, int bh, bool same) const {
        int i=al;
        for (; i+1<ah; i+=2) {
            const V xi=duplicate2(x+i), yi=duplicate2(y+i), zi=duplicate2(z+i);
            V tx=zero(), ty=zero(), tz=zero();
            int j=same?i+2:bl;
            for (; j+2<=bh; j+=2) {
                const V dx=sub(_mm256_broadcast_pd((const __m128d*)(x+j)),xi);
                const V dy=sub(_mm256_broadcast_pd((const __m128d*)(y+j)),yi);
                const V dz=sub(_mm256_broadcast_pd((const __m128d*)(z+j)),zi);
                const V q=fmadd(dx,dx,fmadd(dy,dy,fmadd(dz,dz,splat(eps2))));
                const V f=inv_cube(q);
                const V ux=mul(dx,f), uy=mul(dy,f), uz=mul(dz,f);
                tx=add(ux,tx); ty=add(uy,ty); tz=add(uz,tz);
                _mm_storeu_pd(ax+j,_mm_sub_pd(_mm_loadu_pd(ax+j),pair_sum(ux)));
                _mm_storeu_pd(ay+j,_mm_sub_pd(_mm_loadu_pd(ay+j),pair_sum(uy)));
                _mm_storeu_pd(az+j,_mm_sub_pd(_mm_loadu_pd(az+j),pair_sum(uz)));
            }
            _mm_storeu_pd(ax+i,_mm_add_pd(_mm_loadu_pd(ax+i),adjacent_sum(tx)));
            _mm_storeu_pd(ay+i,_mm_add_pd(_mm_loadu_pd(ay+i),adjacent_sum(ty)));
            _mm_storeu_pd(az+i,_mm_add_pd(_mm_loadu_pd(az+i),adjacent_sum(tz)));
            if (j<bh) block(i,i+2,j,bh,false);
            if (same) block(i,i+2,i,i+2,true);
        }
        if (i<ah && !same) block(i,ah,bl,bh,false);
    }
    template<int k> static inline V perm(V a) {
        return _mm256_permute4x64_pd(a,_MM_SHUFFLE(3^k,2^k,1^k,0^k));
    }
    template<int k> static inline void step(V xi, V yi, V zi, V xj, V yj, V zj,
            V &tx, V &ty, V &tz, V &sx, V &sy, V &sz) {
        const V dx=sub(perm<k>(xj),xi), dy=sub(perm<k>(yj),yi), dz=sub(perm<k>(zj),zi);
        const V q=fmadd(dx,dx,fmadd(dy,dy,fmadd(dz,dz,splat(eps2))));
        const V f=inv_cube(q);
        const V ux=mul(dx,f), uy=mul(dy,f), uz=mul(dz,f);
        tx=add(ux,tx); ty=add(uy,ty); tz=add(uz,tz);
        sx=sub(sx,perm<k>(ux)); sy=sub(sy,perm<k>(uy)); sz=sub(sz,perm<k>(uz));
    }
    __attribute__((noinline)) void block4(int al, int ah, int bl, int bh, bool same) const {
        int i=al;
        for (; i+4<=ah; i+=4) {
            const V xi=load(x+i), yi=load(y+i), zi=load(z+i);
            V tx=zero(), ty=zero(), tz=zero();
            int j=same?i+4:bl;
            for (; j+4<=bh; j+=4) {
                const V xj=load(x+j), yj=load(y+j), zj=load(z+j);
                V sx=load(ax+j),sy=load(ay+j),sz=load(az+j);
                step<0>(xi,yi,zi,xj,yj,zj,tx,ty,tz,sx,sy,sz);
                step<1>(xi,yi,zi,xj,yj,zj,tx,ty,tz,sx,sy,sz);
                step<2>(xi,yi,zi,xj,yj,zj,tx,ty,tz,sx,sy,sz);
                step<3>(xi,yi,zi,xj,yj,zj,tx,ty,tz,sx,sy,sz);
                store(ax+j,sx); store(ay+j,sy); store(az+j,sz);
            }
            store(ax+i,add(load(ax+i),tx));
            store(ay+i,add(load(ay+i),ty));
            store(az+i,add(load(az+i),tz));
            if (j<bh) block(i,i+4,j,bh,false);
            if (same) block(i,i+4,i,i+4,true);
        }
        if (i<ah) block(i,ah,same?i:bl,bh,same);
    }
    void interact(int al,int ah,int bl,int bh,bool same) const {
        if (version==2) block2(al,ah,bl,bh,same);
        else if (version==4) block4(al,ah,bl,bh,same);
        else if (version==5) solve_block<0>(x,y,z,ax,ay,az,al,ah,bl,bh,same);
        else if (version==6) solve_block<1>(x,y,z,ax,ay,az,al,ah,bl,bh,same);
        else if (version==7) solve_block<2>(x,y,z,ax,ay,az,al,ah,bl,bh,same);
        else if (version>=8) solve_block<0,true>(x,y,z,ax,ay,az,al,ah,bl,bh,same);
        else block(al,ah,bl,bh,same);
    }
    void pair(const Node &a, const Node &b, bool same) const {
        if(version>=8 && !same && a.size()>b.size()){
            pair(b,a,false);
            return;
        }
        if (same && a.size() > 512) {
            for (int i = a.lo; i < a.hi; i+=384) {
                const int ie = std::min(i+384,a.hi);
                interact(i,ie,i,ie,true);
                for (int j = ie; j < a.hi; j+=384)
                    interact(i,ie,j,std::min(j+384,a.hi),false);
            }
        } else if (!same && b.size() > 512) {
            for (int j = b.lo; j < b.hi; j+=384)
                interact(a.lo,a.hi,j,std::min(j+384,b.hi),false);
        } else {
            interact(a.lo,a.hi,b.lo,b.hi,same);
        }
    }
    __attribute__((noinline)) void fast_block(int al,int ah,int bl,int bh,bool same,
                                             const double *c) const {
        alignas(32) float bx[400]{},by[400]{},bz[400]{},ox[400]{},oy[400]{},oz[400]{};
        const int nn=bh-bl;
        for(int j=0;j<nn;++j){
            bx[j]=float(x[bl+j]-c[0]);by[j]=float(y[bl+j]-c[1]);bz[j]=float(z[bl+j]-c[2]);
        }
        const __m256 e2=_mm256_set1_ps(float(eps2)),one=_mm256_set1_ps(1.f),
                     onehalf=_mm256_set1_ps(1.5f),two=_mm256_set1_ps(1.875f);
        const __m256i idx=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
        for(int i=al;i<ah;++i){
            const __m256 xi=_mm256_set1_ps(float(x[i]-c[0])),
                         yi=_mm256_set1_ps(float(y[i]-c[1])),
                         zi=_mm256_set1_ps(float(z[i]-c[2]));
            __m256 tx=_mm256_setzero_ps(),ty=tx,tz=tx;
            for(int j=same?i+1-bl:0;j<nn;j+=8){
                const __m256 dx=_mm256_sub_ps(_mm256_loadu_ps(bx+j),xi),
                             dy=_mm256_sub_ps(_mm256_loadu_ps(by+j),yi),
                             dz=_mm256_sub_ps(_mm256_loadu_ps(bz+j),zi);
                const __m256 q=_mm256_fmadd_ps(dx,dx,_mm256_fmadd_ps(dy,dy,_mm256_fmadd_ps(dz,dz,e2)));
                const __m256 r=_mm256_rsqrt_ps(q),r2=_mm256_mul_ps(r,r),
                             e=_mm256_fnmadd_ps(q,r2,one);
                __m256 f=_mm256_mul_ps(_mm256_mul_ps(r,r2),
                    _mm256_fmadd_ps(e,_mm256_fmadd_ps(e,two,onehalf),one));
                if(j+8>nn)f=_mm256_and_ps(f,_mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(_mm256_set1_epi32(nn-j),idx)));
                tx=_mm256_fmadd_ps(dx,f,tx);ty=_mm256_fmadd_ps(dy,f,ty);tz=_mm256_fmadd_ps(dz,f,tz);
                _mm256_storeu_ps(ox+j,_mm256_fnmadd_ps(dx,f,_mm256_loadu_ps(ox+j)));
                _mm256_storeu_ps(oy+j,_mm256_fnmadd_ps(dy,f,_mm256_loadu_ps(oy+j)));
                _mm256_storeu_ps(oz+j,_mm256_fnmadd_ps(dz,f,_mm256_loadu_ps(oz+j)));
            }
            auto accumulate=[](__m256 v){
                return sum(_mm256_cvtps_pd(_mm256_castps256_ps128(v)))+
                       sum(_mm256_cvtps_pd(_mm256_extractf128_ps(v,1)));
            };
            ax[i]+=accumulate(tx);ay[i]+=accumulate(ty);az[i]+=accumulate(tz);
        }
        for(int j=0;j<nn;++j){
            ax[bl+j]+=ox[j];ay[bl+j]+=oy[j];az[bl+j]+=oz[j];
        }
    }
    void pair_fast(const Node &a,const Node &b,bool same) const {
        if(!same&&a.size()>b.size()){
            pair_fast(b,a,false);
            return;
        }
        if(!same){
            for(int j=b.lo;j<b.hi;j+=384)
                fast_block(a.lo,a.hi,j,std::min(j+384,b.hi),false,b.c);
        } else {
            for(int i=a.lo;i<a.hi;i+=384){
                const int hi=std::min(i+384,a.hi);
                fast_block(i,hi,i,hi,true,a.c);
                for(int j=hi;j<a.hi;j+=384)
                    fast_block(i,hi,j,std::min(j+384,a.hi),false,a.c);
            }
        }
    }
};

struct Row {
    int y, z, n, off;
};
struct Layout {
    int p, size = 0, nz;
    int off[stride][stride]{};
    std::vector<Row> rows;
    void init(int degree, int zmax=cap) {
        p = degree;
        nz = std::min(p,zmax);
        for (int z = 0; z <= nz; ++z)
            for (int y = 0; y <= p-z; ++y) {
                const int nx = p+1-z-y;
                off[z][y] = size;
                rows.push_back({y,z,nx,size});
                size += (nx+6)&~3;
            }
    }
};

struct Algebra {
    Layout layout[stride], harmonic[stride], harmonic_d[stride];
    double inv[stride], choose6[stride], chooseH[stride];
    Algebra() {
        for (int i=0; i<=cap; ++i) {
            layout[i].init(i);
            harmonic[i].init(i,1);
            harmonic_d[i].init(i,2);
            inv[i] = i ? 1.0/i : 0.0;
            double v = 1.;
            for (int j=1; j<=6; ++j) v = v*(i+j)/j;
            choose6[i] = v;
            v=0.;
            for (int k=0;k<=std::min(i,2);++k) {
                double a=k==1?2.:1.;
                for (int j=1;j<=4;++j) a=a*(i-k+j)/j;
                v+=a;
            }
            chooseH[i]=v;
        }
    }
    void powers(double x, double *f, int p) const {
        f[0] = 1.;
        for (int i=1; i<=p; ++i) f[i] = f[i-1]*x*inv[i];
        for (int i=p+1; i<p+5; ++i) f[i] = 0.;
    }
    void p2m(const Tree &t, const Node &a, int p, double *out) const {
        const Layout &s = layout[p];
        alignas(32) double x[stride+5], y[stride+5], z[stride+5];
        for (int i=a.lo; i<a.hi; ++i) {
            powers(t.xyz[0][i]-a.c[0], x, p);
            powers(t.xyz[1][i]-a.c[1], y, p);
            powers(t.xyz[2][i]-a.c[2], z, p);
            for (const Row &r : s.rows) {
                const V f = splat(y[r.y]*z[r.z]);
                for (int j=0; j<r.n; j+=4)
                    store(out+r.off+j, fmadd(f,load(x+j),load(out+r.off+j)));
            }
        }
    }
    void shift(const double *src, double *dst, int p, int ps, const double *delta,
               bool local) const {
        const Layout &a = layout[p], &b = layout[ps];
        alignas(32) double tmp1[maxcoef]{}, tmp2[maxcoef]{};
        double f[stride+4];
        // Shift along x, y and z separately. Moment coefficients include 1/alpha!;
        // local coefficients store ordinary, unscaled derivatives.
        powers(delta[0], f, p);
        for (const Row &r : a.rows) {
            const double *s = src+b.off[r.z][r.y];
            double *d = tmp1+r.off;
            for (int k=0; k<r.n; ++k) {
                const V w = splat(f[k]);
                for (int i=0; i<r.n-k; i+=4) {
                    if (local) {
                        if (i+3 < r.n-k)
                            store(d+i, fmadd(w,load(s+i+k),load(d+i)));
                        else
                            for (int j=i;j<r.n-k;++j) d[j] += f[k]*s[j+k];
                    } else {
                        store(d+i+k, fmadd(w,load(s+i),load(d+i+k)));
                    }
                }
            }
        }
        powers(delta[1], f, p);
        for (const Row &r : a.rows) {
            double *d = tmp2+r.off;
            for (int k=0, lim=local?p-r.y-r.z:r.y; k<=lim; ++k) {
                const int sy = local ? r.y+k : r.y-k;
                const double *s = tmp1+a.off[r.z][sy];
                const V w = splat(f[k]);
                const int end = local ? r.n-k : r.n;
                for (int i=0; i<end; i+=4) {
                    if (local && i+3 >= end) {
                        for (int j=i;j<end;++j) d[j] += f[k]*s[j];
                    } else {
                        store(d+i, fmadd(w,load(s+i),load(d+i)));
                    }
                }
            }
        }
        powers(delta[2], f, p);
        for (const Row &r : a.rows) {
            double *d = dst+r.off;
            for (int k=0, lim=local?p-r.y-r.z:r.z; k<=lim; ++k) {
                const int sz = local ? r.z+k : r.z-k;
                const double *s = tmp2+a.off[sz][r.y];
                const V w = splat(f[k]);
                const int end = local ? r.n-k : r.n;
                for (int i=0; i<end; i+=4) {
                    if (local && i+3 >= end) {
                        for (int j=i;j<end;++j) d[j] += f[k]*s[j];
                    } else {
                        store(d+i, fmadd(w,load(s+i),load(d+i)));
                    }
                }
            }
        }
    }
    void derivatives(double x, double y, double z, const Layout &s, double *buf,
                     bool reduce) const {
        const int p=s.p;
        const double q = x*x+y*y+z*z+eps2, iq = 1./q;
        double f1[stride], f2[stride], zz[stride+4]{};
        for (int i=1; i<=p; ++i) {
            f1[i] = -(2*i-1)*inv[i]*iq;
            f2[i] = -(i-1)*inv[i]*iq;
        }
        // Coefficient recurrence from Q(u) (u . grad K) = -(R . u + u.u) K.
        for (const Row &r : s.rows) {
            const int j=r.y, k=r.z;
            double *d = buf+r.off;
            const double *dy = j ? buf+s.off[k][j-1] : zz;
            const double *dz = k ? buf+s.off[k-1][j] : zz;
            const double *dyy = j>=2 ? buf+s.off[k][j-2] : zz;
            const double *dzz = k>=2 ? buf+s.off[k-2][j] : zz;
            const double a = j*y, b = k*z, c = j*(j-1), e = k*(k-1);
            double prev=0., prev2=0.;
            if (reduce && k==2) {
                const double *dd = buf+s.off[0][j+2];
                for (int i=0;i<r.n;++i) d[i]=-dzz[i+2]-dd[i];
            } else {
                for (int i=0; i<r.n; ++i) {
                    int n = i+j+k;
                    double w;
                    if (!n) w = std::sqrt(iq);
                    else w = f1[n]*(i*x*prev+a*dy[i]+b*dz[i])
                           + f2[n]*(i*(i-1)*prev2+c*dyy[i]+e*dzz[i]);
                    d[i]=w;
                    prev2=prev;
                    prev=w;
                }
            }
            for (int i=r.n, end=(r.n+6)&~3; i<end; ++i) d[i]=0.;
        }
    }
    void contract(const Layout &s, const Layout &sd, const Layout &sa,
                  const Layout &sb, const double *buf, const double *ma,
                  const double *mb, double *la, double *lb, bool residual) const {
        alignas(32) double neg[maxcoef];
        for (const Row &r:s.rows) {
            const double *f=mb+sb.off[r.z][r.y];
            double *d=neg+r.off;
            const double sign=(r.y+r.z)%2 ? -1. : 1.;
            const V sg=_mm256_setr_pd(sign,-sign,sign,-sign);
            for (int i=0;i<r.n;i+=4) store(d+i,mul(sg,load(f+i)));
        }
        const V odd = _mm256_setr_pd(1.,-1.,1.,-1.);
        for (const Row &l : s.rows) {
            double *aa = la+sa.off[l.z][l.y], *bb=lb+sb.off[l.z][l.y];
            const V sg = (l.y+l.z)%2 ? _mm256_setr_pd(-1.,1.,-1.,1.) : odd;
            for (int ii=0; ii<l.n; ii+=4) {
                V a0=zero(), a1=zero(), b0=zero(), b1=zero();
                const int lim = l.n-1-ii;
                for (int z=residual?std::max(0,2-l.z):0; z<=std::min(lim,s.nz); ++z) {
                    for (int y=0; y<=lim-z; ++y) {
                        const double *d=buf+sd.off[l.z+z][l.y+y]+ii;
                        const double *m0=ma+sa.off[z][y], *m1=neg+s.off[z][y];
                        const int k=lim-y-z;
                        int i=0;
                        for (; i<k; i+=2) {
                            const V u=load(d+i), v=load(d+i+1);
                            a0=fmadd(u,splat(m1[i]),a0);
                            a1=fmadd(v,splat(m1[i+1]),a1);
                            b0=fmadd(u,splat(m0[i]),b0);
                            b1=fmadd(v,splat(m0[i+1]),b1);
                        }
                        if (i<=k) {
                            const V u=load(d+i);
                            a0=fmadd(u,splat(m1[i]),a0);
                            b0=fmadd(u,splat(m0[i]),b0);
                        }
                    }
                }
                store(aa+ii,add(load(aa+ii),add(a0,a1)));
                store(bb+ii,fmadd(sg,add(b0,b1),load(bb+ii)));
            }
        }
    }
    void expand_harmonic(const double *src, const Layout &from, double *out,
                         const Layout &to) const {
        for (const Row &r : to.rows) {
            double *d=out+r.off;
            if (r.z<2) {
                std::copy_n(src+from.off[r.z][r.y],r.n,d);
            } else {
                const double *a=out+to.off[r.z-2][r.y]+2;
                const double *b=out+to.off[r.z-2][r.y+2];
                for (int x=0;x<r.n;++x) d[x]=-a[x]-b[x];
            }
        }
    }
    void reduce_harmonic(double *src, const Layout &from, double *dst, int p) const {
        for (auto it=from.rows.rbegin();it!=from.rows.rend();++it) {
            const Row &r=*it;
            int nx=std::min(r.n,p+1-r.y-r.z);
            if (nx<=0 || r.z<2) continue;
            double *a=src+from.off[r.z-2][r.y]+2;
            double *b=src+from.off[r.z-2][r.y+2];
            const double *s=src+r.off;
            for (int i=0;i<nx;++i) {
                a[i]-=s[i];
                b[i]-=s[i];
            }
        }
        for (const Row &r:harmonic[p].rows)
            std::copy_n(src+from.off[r.z][r.y],r.n,dst+r.off);
    }
    void m2l(const Node &a, const Node &b, int p, int pa, int pb, const double *ma,
             const double *mb, double *la, double *lb, const double *hd=nullptr,
             int ph=-1) const {
        alignas(32) double buf[maxcoef], hb[maxcoef];
        const Layout &s=layout[p];
        derivatives(a.c[0]-b.c[0], a.c[1]-b.c[1], a.c[2]-b.c[2],s,buf,false);
        if (hd) {
            expand_harmonic(hd,harmonic_d[ph],hb,s);
            for (const Row &r:s.rows) {
                if (r.z<2) std::fill_n(buf+r.off,r.n,0.);
                else for (int i=0;i<r.n;++i) buf[r.off+i]-=hb[r.off+i];
            }
        }
        contract(s,s,layout[pa],layout[pb],buf,ma,mb,la,lb,hd!=nullptr);
    }
    void point_field(const Layout &s, const Layout &sd, const Layout &sb,
                     const double *buf, const double *m, double *l, double *a) const {
        V ax=zero(),ay=zero(),az=zero();
        for(const Row &r:s.rows){
            const int off=sb.off[r.z][r.y];
            const double *dx=buf+sd.off[r.z][r.y];
            const double *dy=r.n>1?buf+sd.off[r.z][r.y+1]:nullptr;
            const double *dz=r.n>1?buf+sd.off[r.z+1][r.y]:nullptr;
            for(int i=0;i<r.n;i+=4)store(l+off+i,add(load(l+off+i),load(dx+i)));
            for(int i=0;i+1<r.n;i+=4){
                const V v=load(m+off+i);
                ax=fmadd(v,load(dx+i+1),ax); ay=fmadd(v,load(dy+i),ay);
                az=fmadd(v,load(dz+i),az);
            }
        }
        a[0]-=sum(ax); a[1]-=sum(ay); a[2]-=sum(az);
    }
    void p2l_m2p(const Tree &t,const Node &a,int j,int p,int h,int palloc,int halloc,
                 const double *m,const double *hm,double *l,double *hl,double *acc) const {
        alignas(32) double buf[maxcoef],hd[maxcoef],res[maxcoef];
        const double dx=a.c[0]-t.xyz[0][j],dy=a.c[1]-t.xyz[1][j],dz=a.c[2]-t.xyz[2][j];
        if(h>=0){
            derivatives(dx,dy,dz,harmonic_d[h],hd,true);
            point_field(harmonic[h],harmonic_d[h],harmonic[halloc],hd,hm,hl,acc);
        }
        if(p>=0){
            const Layout &s=layout[p];
            derivatives(dx,dy,dz,s,buf,false);
            if(h>=0){
                expand_harmonic(hd,harmonic_d[h],res,s);
                for(const Row &r:s.rows){
                    if(r.z<2)std::fill_n(buf+r.off,r.n,0.);
                    else for(int i=0;i<r.n;++i)buf[r.off+i]-=res[r.off+i];
                }
            }
            point_field(s,s,layout[palloc],buf,m,l,acc);
        }
    }
    void l2p(Tree &t, const Node &a, int p, const double *l) const {
        const Layout &s=layout[p];
        alignas(32) double x[stride+5], y[stride+5], z[stride+5];
        for (int i=a.lo; i<a.hi; ++i) {
            powers(t.xyz[0][i]-a.c[0], x+1, p);
            powers(t.xyz[1][i]-a.c[1], y+1, p);
            powers(t.xyz[2][i]-a.c[2], z+1, p);
            x[0]=y[0]=z[0]=0.;
            V ax=zero(), ay=zero(), az=zero();
            for (const Row &r : s.rows) {
                V v=zero(), u=zero();
                int j=0;
                for (; j+4<=r.n; j+=4) {
                    V q=load(l+r.off+j);
                    v=fmadd(q,load(x+1+j),v);
                    u=fmadd(q,load(x+j),u);
                }
                if (j<r.n) {
                    const V mask=_mm256_castsi256_pd(_mm256_set_epi64x(
                        0, r.n-j>=3?-1LL:0LL, r.n-j>=2?-1LL:0LL, -1LL));
                    V q=_mm256_and_pd(mask,load(l+r.off+j));
                    v=fmadd(q,load(x+1+j),v);
                    u=fmadd(q,load(x+j),u);
                }
                ax=fmadd(u,splat(y[r.y+1]*z[r.z+1]),ax);
                ay=fmadd(v,splat(y[r.y]*z[r.z+1]),ay);
                az=fmadd(v,splat(y[r.y+1]*z[r.z]),az);
            }
            t.acc[0][i]+=sum(ax);
            t.acc[1][i]+=sum(ay);
            t.acc[2][i]+=sum(az);
        }
    }
};

struct Pair { int a,b; int16_t p,h; };
struct Plan {
    const Tree &t;
    const Algebra &g;
    std::vector<Pair> direct, far, points;
    std::vector<int> degree,hdegree;
    double tol, cut, p2pcost=0, m2lcost=0;
    int pmax;
    bool hybrid = env("GALAXY_HARMONIC",1.)!=0.;
    double hmargin = env("GALAXY_HMARGIN",20.);
    double nearprod=env("GALAXY_NEAR_PROD",1900.);
    const std::vector<double> *inorm=nullptr;
    double weight=env("GALAXY_WEIGHT",.6);
    bool use_points=env("GALAXY_POINTS",1.)!=0.;
    double pointwork=0.;
    double limit_pairs = 1e100;
    bool stopped = false;
    size_t field_count = 0;
    double leaf_work = 0., shift_work = 0., predicted = 0.;
    explicit Plan(const Tree &t, const Algebra &g, double tol, int pmax, double cut,
                  const std::vector<double> *inorm=nullptr)
        : t(t), g(g), degree(t.nodes.size(),-1), hdegree(t.nodes.size(),-1),
          tol(tol), cut(cut), pmax(pmax), inorm(inorm) {}
    void near_pair(int a, int b) {
        direct.push_back({a,b,0,0});
        const double na=t.nodes[a].size(), nb=t.nodes[b].size();
        p2pcost += (a==b ? na*(na-1)*.5 : na*nb);
        if (p2pcost>limit_pairs || direct.size()>17000000) stopped=true;
    }
    bool point_pair(int a,int b,double best_cost=1e100) {
        if(!use_points || a==b)return false;
        if(t.nodes[a].size()<t.nodes[b].size())std::swap(a,b);
        const Node &x=t.nodes[a],&y=t.nodes[b];
        if(x.size()<240 || x.size()<6*y.size() || y.size()>96)return false;
        const double dx=x.c[0]-y.c[0],dy=x.c[1]-y.c[1],dz=x.c[2]-y.c[2];
        const double dr=std::max(0.,std::sqrt(dx*dx+dy*dy+dz*dz)-y.radius);
        const double q=dr*dr+eps2;
        if(x.radius*x.radius>=.68*q)return false;
        const double ratio=.5*std::log(x.radius*x.radius/q);
        double tol_pair=tol;
        if(inorm && weight!=0.){
            const double w=std::max(y.size()*(*inorm)[a],x.size()*(*inorm)[b])/q;
            tol_pair=std::min(-1.3,tol_pair-weight*std::log(std::min(4.,w)));
        }
        const int full=std::max(3,(int)std::ceil(tol_pair/ratio));
        if(full>pmax)return false;
        int p=full,h=-1;
        double coef=g.layout[p].size;
        if(hybrid && q>hmargin*eps2){
            const int correction=std::max(1,(int)std::ceil((tol_pair-std::log(hmargin*eps2/q))/ratio));
            if(correction<full){
                const double reduced=g.harmonic_d[full].size+
                    (correction<2?0.:1.16*g.layout[correction].size);
                if(reduced<.9*coef){
                    p=correction<2?-1:correction;
                    h=full;
                    coef=reduced;
                }
            }
        }
        const double work=(coef+24.)*y.size();
        if(work*5.3e-9>=std::min(best_cost,double(x.size())*y.size()*2.12e-9))return false;
        pointwork+=work;
        points.push_back({a,b,(int16_t)p,(int16_t)h});
        degree[a]=std::max(degree[a],p);
        hdegree[a]=std::max(hdegree[a],h);
        return true;
    }
    void visit(int a, int b) {
        if (stopped) return;
        const Node &x=t.nodes[a], &y=t.nodes[b];
        if (a==b) {
            if (x.leaf() || x.size()<=80) near_pair(a,b);
            else {
                visit(x.left,x.left);
                visit(x.left,x.right);
                visit(x.right,x.right);
            }
            return;
        }
        const double prod=double(x.size())*y.size();
        if (prod > 180.) {
            const double rx=x.c[0]-y.c[0], ry=x.c[1]-y.c[1], rz=x.c[2]-y.c[2];
            const double q=rx*rx+ry*ry+rz*rz+eps2, r=x.radius+y.radius;
            if (r*r < .8*q) {
                // Lowest order meeting rho^p <= tolerance, rho = (rA+rB)/sqrt(q).
                const double logR=0.5*std::log((r*r)/q);
                double tol_pair=tol;
                if(inorm && weight!=0.){
                    const double w=std::max(y.size()*(*inorm)[a],x.size()*(*inorm)[b])/q;
                    tol_pair-=weight*std::log(std::min(4.,w));
                    tol_pair=std::min(-1.3,tol_pair);
                }
                const int full=std::max(3,(int)std::ceil(tol_pair/logR));
                if (full<=pmax) {
                    int p=full, h=-1;
                    double cost=g.choose6[p];
                    if (hybrid && q>hmargin*eps2) {
                        const double logfrac=std::log(hmargin*eps2/q);
                        const int correction=std::max(1,(int)std::ceil((tol_pair-logfrac)/logR));
                        if (correction<full) {
                            const double reduced=1.05*g.chooseH[full]+
                                (correction<2 ? 0. : .72*g.choose6[correction]);
                            if (reduced<.90*cost) {
                                p=correction<2 ? -1 : correction;
                                h=full;
                                cost=reduced;
                            }
                        }
                    }
                    if (prod>=cut*cost+40.*full) {
                        if(point_pair(a,b,0.77e-9*cost))return;
                        far.push_back({a,b,(int16_t)p,(int16_t)h});
                        if (far.size()>17000000) stopped=true;
                        m2lcost+=cost;
                        degree[a]=std::max(degree[a],p);
                        degree[b]=std::max(degree[b],p);
                        hdegree[a]=std::max(hdegree[a],h);
                        hdegree[b]=std::max(hdegree[b],h);
                        return;
                    }
                }
            }
        }
        if(point_pair(a,b))return;
        if (prod<=nearprod || (x.leaf() && y.leaf())) {
            near_pair(a,b);
        } else if (y.leaf() || (!x.leaf() && x.radius>=y.radius)) {
            visit(x.left,b); visit(x.right,b);
        } else {
            visit(a,y.left); visit(a,y.right);
        }
    }
    void estimate(int v,int up) {
        const int p=std::max({up,degree[v],hdegree[v]});
        if (degree[v]>=0) field_count+=g.layout[degree[v]].size;
        if (hdegree[v]>=0) field_count+=g.harmonic[hdegree[v]].size;
        const Node &a=t.nodes[v];
        if (a.leaf()) {
            if (p>=0) leaf_work+=double(g.layout[p].size)*a.size();
        } else {
            if (p>=0) shift_work+=double(g.layout[p].size)*(p+4);
            estimate(a.left,p); estimate(a.right,p);
        }
    }
    double cost() {
        if (stopped) return 1e100;
        estimate(0,-1);
        if (field_count>70000000 || (field_count*16+12*(direct.capacity()+far.capacity()+points.capacity())+
                size_t(t.n)*56+t.nodes.size()*64)>size_t(1550)*(1<<20)) {
            stopped=true;
            return 1e100;
        }
        predicted=2.53e-9*p2pcost+0.83e-9*m2lcost+
            1.65e-9*leaf_work+0.94e-9*shift_work+0.24e-6*far.size()+
            0.044e-6*direct.size()+32e-9*t.n+5.7e-9*pointwork;
        return predicted;
    }
};

struct FMM {
    Tree &t;
    const Algebra &g;
    const Plan &plan;
    struct Field { size_t off=0; int p=-1; };
    std::vector<Field> fields;
    std::vector<Field> hfields;
    std::vector<double> m,l,hm,hl;
    struct alignas(32) Workspace { double temp[maxcoef], child[maxcoef]; };
    std::unique_ptr<Workspace[]> scratch;
    explicit FMM(Tree &t, const Algebra &g, const Plan &plan)
        : t(t), g(g), plan(plan), fields(plan.degree.size()), hfields(plan.degree.size()),
          scratch(new Workspace[t.height+1]) {
        size_t count=0,hcount=0;
        for (size_t i=0;i<fields.size();++i) {
            const int p=plan.degree[i];
            if (p>=0) {
                fields[i]={count,p};
                count+=g.layout[p].size;
            }
            const int h=plan.hdegree[i];
            if (h>=0) {
                hfields[i]={hcount,h};
                hcount+=g.harmonic[h].size;
            }
        }
        m.resize(count);
        l.resize(count);
        hm.resize(hcount);
        hl.resize(hcount);
    }
    void upward(int v, int parent, double *out, int depth=0) {
        const Field &s=fields[v];
        const Field &h=hfields[v];
        const Node &a=t.nodes[v];
        const int p=std::max({s.p,h.p,parent});
        if (p<0) {
            if (!a.leaf()) {
                upward(a.left,-1,nullptr,depth+1);
                upward(a.right,-1,nullptr,depth+1);
            }
            return;
        }
        double *temp=scratch[depth].temp, *child=scratch[depth].child;
        double *val=(p==s.p?m.data()+s.off:temp);
        if (val==temp) std::fill_n(temp,g.layout[p].size,0.);
        if (a.leaf()) g.p2m(t,a,p,val);
        else {
            for (const int k : {a.left,a.right}) {
                const Node &b=t.nodes[k];
                upward(k,p,child,depth+1);
                const double d[3]={b.c[0]-a.c[0], b.c[1]-a.c[1], b.c[2]-a.c[2]};
                g.shift(child,val,p,p,d,false);
            }
        }
        if (s.p>=0 && val==temp) {
            for (const Row &r : g.layout[s.p].rows)
                std::copy_n(val+g.layout[p].off[r.z][r.y],r.n,m.data()+s.off+r.off);
        }
        if (parent>=0) {
            for (const Row &r : g.layout[parent].rows) {
                std::copy_n(val+g.layout[p].off[r.z][r.y],r.n,out+r.off);
                std::fill_n(out+r.off+r.n,((r.n+6)&~3)-r.n,0.);
            }
        }
        if (h.p>=0) {
            if (val!=temp) for (const Row &r:g.layout[p].rows)
                std::copy_n(val+r.off,r.n,temp+r.off);
            g.reduce_harmonic(temp,g.layout[p],hm.data()+h.off,h.p);
        }
    }
    void translate(const Pair &q) {
        const Field &a=fields[q.a], &b=fields[q.b];
        const Node &na=t.nodes[q.a], &nb=t.nodes[q.b];
        if (q.h>=0) {
            alignas(32) double hd[maxcoef];
            const Field &ha=hfields[q.a], &hb=hfields[q.b];
            const Layout &h=g.harmonic[q.h], &d=g.harmonic_d[q.h];
            g.derivatives(na.c[0]-nb.c[0],na.c[1]-nb.c[1],na.c[2]-nb.c[2],d,hd,true);
            g.contract(h,d,g.harmonic[ha.p],g.harmonic[hb.p],hd,hm.data()+ha.off,
                       hm.data()+hb.off,hl.data()+ha.off,hl.data()+hb.off,false);
            if (q.p>=0) g.m2l(na,nb,q.p,a.p,b.p,m.data()+a.off,m.data()+b.off,l.data()+a.off,
                              l.data()+b.off,hd,q.h);
        } else {
            g.m2l(na,nb,q.p,a.p,b.p,m.data()+a.off,m.data()+b.off,l.data()+a.off,l.data()+b.off);
        }
    }
    void point(const Pair &q){
        const Field &a=fields[q.a],&h=hfields[q.a];
        const Node &na=t.nodes[q.a],&nb=t.nodes[q.b];
        const double *m0=q.p>=0?m.data()+a.off:nullptr,*mh=q.h>=0?hm.data()+h.off:nullptr;
        double *l0=q.p>=0?l.data()+a.off:nullptr,*lh=q.h>=0?hl.data()+h.off:nullptr;
        for(int i=nb.lo;i<nb.hi;++i){
            double v[3]{};
            g.p2l_m2p(t,na,i,q.p,q.h,a.p,h.p,m0,mh,l0,lh,v);
            for(int k=0;k<3;++k)t.acc[k][i]+=v[k];
        }
    }
    void downward(int v, int parent, int pp, const double *given, int depth=0) {
        const Field &s=fields[v];
        const Field &h=hfields[v];
        const Node &a=t.nodes[v];
        const int p=std::max({s.p,h.p,pp});
        double *tmp=scratch[depth].temp;
        const double *total=nullptr;
        if (p>=0) {
            if (pp<0 && h.p<0) total=l.data()+s.off;
            else {
                std::fill_n(tmp,g.layout[p].size,0.);
                if (s.p>=0) for (const Row &r:g.layout[s.p].rows)
                    std::copy_n(l.data()+s.off+r.off,r.n,tmp+g.layout[p].off[r.z][r.y]);
                if (pp>=0) {
                    const Node &b=t.nodes[parent];
                    const double d[3]={a.c[0]-b.c[0],a.c[1]-b.c[1],a.c[2]-b.c[2]};
                    if (pp==p) g.shift(given,tmp,p,pp,d,true);
                    else {
                        double *sub=scratch[depth].child;
                        std::fill_n(sub,g.layout[pp].size,0.);
                        g.shift(given,sub,pp,pp,d,true);
                        for (const Row &r:g.layout[pp].rows) {
                            double *to=tmp+g.layout[p].off[r.z][r.y];
                            for (int i=0;i<r.n;++i) to[i]+=sub[r.off+i];
                        }
                    }
                }
                if (h.p>=0) {
                    double *sub=scratch[depth].child;
                    g.expand_harmonic(hl.data()+h.off,g.harmonic[h.p],sub,g.layout[h.p]);
                    for (const Row &r:g.layout[h.p].rows) {
                        double *to=tmp+g.layout[p].off[r.z][r.y];
                        for (int i=0;i<r.n;++i) to[i]+=sub[r.off+i];
                    }
                }
                total=tmp;
            }
        }
        if (a.leaf()) {
            if (p>0) g.l2p(t,a,p,total);
        } else {
            downward(a.left,v,p,total,depth+1);
            downward(a.right,v,p,total,depth+1);
        }
    }
};

void execute_near(const Tree &t,const Plan &plan,const Direct &d,double deadline,bool watch){
    const double block_work=std::max(150000.,std::min(8.0e7,6.0e6*t.n/65536.));
    double remaining=plan.p2pcost,rate=2.39e-9,dc=0.,dw=0.,extra=0.;
    for(size_t i=0;i<plan.direct.size();){
        const double start=cpu_time();
        const bool fast=watch && remaining*rate > (deadline-start)*1.008;
        size_t j=i;
        double work=0.;
        for(;j<plan.direct.size() && work<block_work;++j){
            const Pair &q=plan.direct[j];
            const Node &a=t.nodes[q.a],&b=t.nodes[q.b];
            const double u=a.size(),v=b.size();
            work+=q.a==q.b?.5*u*(u-1):u*v;
            if(fast)d.pair_fast(a,b,q.a==q.b);
            else d.pair(a,b,q.a==q.b);
        }
        i=j;
        remaining-=work;
        if(fast)extra+=work;
        else if(work>0.){
            dc+=cpu_time()-start;
            dw+=work;
            rate=std::clamp(1.025*dc/dw,1.9e-9,4.2e-9);
        }
    }
    if(extra>0. && env("GALAXY_VERBOSE",0.))
        std::fprintf(stderr,"deadline reserve used for %.4g direct pairs\n",extra);
}

bool write_output(Tree &tree, const char *name) {
    std::vector<double> a(3*size_t(tree.n));
    for (int i=0;i<tree.n;++i) {
        double *dst=a.data()+3*size_t(tree.order[i]);
        if(!tree.rotated)for (int k=0;k<3;++k) dst[k]=tree.mass*tree.acc[k][i];
        else for(int k=0;k<3;++k)
            dst[k]=tree.mass*(tree.frame[0][k]*tree.acc[0][i]+
                              tree.frame[1][k]*tree.acc[1][i]+tree.frame[2][k]*tree.acc[2][i]);
    }
    FILE *f=std::fopen(name,"wb");
    if (!f) return false;
    const bool ok=std::fwrite(a.data(),sizeof(double),a.size(),f)==a.size();
    return std::fclose(f)==0 && ok;
}

int solve(const char *input, const char *output) {
    int fd=open(input,O_RDONLY);
    struct stat info;
    if (fd<0 || fstat(fd,&info) || info.st_size<32 || info.st_size%32 ||
            info.st_size>32LL*4194304) return 1;
    const size_t bytes=info.st_size;
    const int n=bytes/32;
    void *map=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if (map==MAP_FAILED) return 1;
    const double *raw=(const double*)map;
    Tree t(n);
    t.mass=raw[0];
    for (int i=0;i<n;++i) {
        t.pts[i]={{raw[4*i+1],raw[4*i+2],raw[4*i+3]},i};
    }
    munmap(map,bytes);
    const double limit=env("GALAXY_CPU_LIMIT_S",2.0*n/65536.);
    const double leaf_default=n<=65536?84.:n<=262144?114.:n<=1048576?150.:190.;
    t.build(std::clamp((int)env("GALAXY_LEAF",leaf_default),8,8192));
    Direct direct(t);
    if (n<=128 || env("GALAXY_EXACT",0.)) {
        direct.pair(t.nodes[0],t.nodes[0],true);
        return write_output(t,output)?0:1;
    }
    const auto inorm=t.inverse_norms();
    const double prep=cpu_time();
    const Algebra g;
    const double cut=env("GALAXY_CUT",.25);
    std::unique_ptr<Plan> best;
    const double specified=env("GALAXY_TOL",0.);
    double strength=specified>0. ? -std::log(specified) : 9.6;
    double low=2.6,high=25.;
    double best_strength=0.;
    const double out_slack=.007+50.e-9*n;
    const double deadline=.90*limit-out_slack;
    for (int i=0;i<(specified>0.?1:8);++i) {
        const int pmax=std::clamp((int)env("GALAXY_PMAX",7+.85*strength),5,cap);
        auto trial=std::make_unique<Plan>(t,g,-strength,pmax,cut,&inorm);
        if (specified<=0.) trial->limit_pairs=limit/2.28e-9;
        trial->visit(0,0);
        const double work=trial->cost();
        const double now=cpu_time();
        const double room=deadline-now;
        if (specified>0. || work<room) {
            best=std::move(trial);
            best_strength=strength;
            low=strength;
            if (work>room*.977 || high-low<.20 ||
                    (i>=2 && now-prep>.065*limit)) break;
            if (high==25. && work>0.)
                strength=std::min(24.9,strength+std::max(.30,5.8*std::log(room/work)));
            else
                strength=(low+high)*.5;
        } else {
            high=strength;
            if (best && (high-low<.22 || now-prep>.065*limit)) break;
            strength=best ? ((low+high)*.5) :
                std::max(low+.01,strength-std::max(.7,std::min(4.,5.8*std::log(work/std::max(1e-6,room)))));
        }
    }
    if (!best) {
        best_strength=2.55;
        best=std::make_unique<Plan>(t,g,-best_strength,8,.18,&inorm);
        best->visit(0,0);
        best->cost();
    }
    const Plan &plan=*best;
    if (plan.stopped) return 1;
    const double planned=cpu_time();
    double up, far, down;
    size_t alloc;
    {
        FMM fmm(t,g,plan);
        alloc=fmm.m.size()+fmm.hm.size();
        fmm.upward(0,-1,nullptr);
        up=cpu_time();
        for (const Pair &p : plan.far) fmm.translate(p);
        for (const Pair &p : plan.points) fmm.point(p);
        far=cpu_time();
        fmm.downward(0,-1,-1,nullptr);
        down=cpu_time();
    }
    execute_near(t,plan,direct,limit-out_slack,specified<=0.);
    const double d=cpu_time();
    const bool ok=write_output(t,output);
    if (env("GALAXY_VERBOSE",0.)) {
        std::fprintf(stderr,"n=%d nodes=%zu depth=%d near=%zu (%.4g pairs) far=%zu (%.4g terms) mem=%.1f MiB\n",
            n,t.nodes.size(),t.height,plan.direct.size(),plan.p2pcost,plan.far.size(),plan.m2lcost,
            alloc*16./(1<<20));
        std::fprintf(stderr,"CPU prepare %.4f plan %.4f up %.4f far %.4f down %.4f direct %.4f out %.4f\n",
            prep,planned-prep,up-planned,far-up,down-far,d-down,cpu_time()-d);
        std::fprintf(stderr,"budget %.4f strength %.3f pmax %d predict %.4f, leaf %.4g shift %.4g\n",
            limit,best_strength,plan.pmax,plan.predicted,plan.leaf_work,plan.shift_work);
        std::fprintf(stderr,"point interactions %zu (%.4g coefficients)\n",plan.points.size(),plan.pointwork);
    }
    return ok?0:1;
}

} // namespace galaxy

#include "grid.hpp"

int main(int argc, char **argv) {
    if(argc!=3)return 1;
    if(galaxy::env("GALAXY_GRID",1.)){
        try {
            const int r=grid::solve_grid(argv[1],argv[2]);
            if(r!=2)return r;
        } catch(const std::bad_alloc &) {
            // The fallback rebuilds from the input with the remaining CPU budget.
        }
    }
    try {
        return galaxy::solve(argv[1],argv[2]);
    } catch(const std::bad_alloc &) {
        return 1;
    }
}
