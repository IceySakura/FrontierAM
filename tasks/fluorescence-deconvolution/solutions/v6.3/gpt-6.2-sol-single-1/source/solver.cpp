#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <time.h>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <fftw3.h>

namespace {
constexpr int S = 32, SX = 48, SY = S * SX, SZ = S * SY, NP = S*S*S;
constexpr int PAD = 8;
constexpr int DIM = 3;
using Vec3 = std::array<float, DIM>;
using V = __m256;

int idx(int z, int y, int x = 0) { return z*SY + y*SX + x + PAD; }
float sq(float x) { return x*x; }
float clamp(float x, float l, float h) { return std::min(std::max(x,l),h); }
double seconds() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) + 1.e-9*double(t.tv_nsec);
}

struct Grid {
    alignas(32) float f[SZ] = {};
    float* row(int z,int y) { return f+idx(z,y); }
    const float* row(int z,int y) const {return f+idx(z,y);}
    float& at(int z,int y,int x) { return f[idx(z,y,x)]; }
    float at(int z,int y,int x) const { return f[idx(z,y,x)]; }
};

struct Box {
    int z0=0,z1=S,y0=0,y1=S;
    Box expand(int rz,int ry) const {
        return {std::max(0,z0-rz), std::min(S,z1+rz), std::max(0,y0-ry), std::min(S,y1+ry)};
    }
    Box intersect(const Box& b) const {
        return {std::max(z0,b.z0),std::min(z1,b.z1),std::max(y0,b.y0),std::min(y1,b.y1)};
    }
};
constexpr Box full{};

inline V splat(float a) {return _mm256_set1_ps(a);}
inline V ld(const float* a) {return _mm256_load_ps(a);}
inline void st(float* a,V v) {_mm256_store_ps(a,v);}
inline V fma(V a,V b,V c) {return _mm256_fmadd_ps(a,b,c);}
inline float sum(V a) {
    __m128 b = _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    b = _mm_hadd_ps(b,b); return _mm_cvtss_f32(_mm_hadd_ps(b,b));
}

// Range reduced exponential on [-24,0]; the neglected Gaussian tails are tiny.
inline V gaussian_exp(V self) {
    V x = _mm256_max_ps(self,splat(-24.0f));
    V n = _mm256_round_ps(x*splat(1.44269504089f),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    V r = fma(n,splat(-0.69314718056f),x);
    V p = fma(r,splat(1.f/120.f),splat(1.f/24.f));
    p = fma(r,p,splat(1.f/6.f));
    p = fma(r,p,splat(.5f));
    p = fma(r,p,splat(1.f));
    p = fma(r,p,splat(1.f));
    const V scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n),_mm256_set1_epi32(127)),23));
    return p*scale;
}

inline V positive_log(V x) {
    const __m256i b=_mm256_castps_si256(_mm256_max_ps(x,splat(1.e-30f)));
    V e=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(b,23),_mm256_set1_epi32(127)));
    V m=_mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(b,_mm256_set1_epi32(0x007fffff)),
                                           _mm256_set1_epi32(0x3f800000)));
    const auto a=_mm256_cmp_ps(m,splat(1.41421356f),_CMP_GT_OQ);
    m=_mm256_blendv_ps(m,m*splat(.5f),a);
    e+=_mm256_and_ps(a,splat(1));
    const V t=(m-splat(1))/(m+splat(1)), t2=t*t;
    V q=fma(t2,splat(1.f/9.f),splat(1.f/7.f));
    q=fma(t2,q,splat(1.f/5.f));
    q=fma(t2,q,splat(1.f/3.f));
    q=fma(t2,q,splat(1));
    return fma(e,splat(.69314718056f),splat(2)*t*q);
}

void axpy(Grid& y,const Grid& x,float a,Box b=full) {
    const V v=splat(a);
    for(int z=b.z0;z<b.z1;z++) for(int j=b.y0;j<b.y1;j++) {
        const float* s=x.row(z,j);
        float* d=y.row(z,j);
        for(int i=0;i<S;i+=8) st(d+i,fma(v,ld(s+i),ld(d+i)));
    }
}
float dot(const Grid& x,const Grid& y,Box b=full) {
    V a=splat(0), aa=splat(0);
    for(int z=b.z0;z<b.z1;z++) for(int j=b.y0;j<b.y1;j++) {
        const float* xp=x.row(z,j), *yp=y.row(z,j);
        for(int i=0;i<S;i+=16) {
            a=fma(ld(xp+i),ld(yp+i),a);
            aa=fma(ld(xp+i+8),ld(yp+i+8),aa);
        }
    }
    return sum(a+aa);
}
float dotw(const Grid& x,const Grid& y,const Grid& w,Box b=full) {
    V a=splat(0), aa=splat(0);
    for(int z=b.z0;z<b.z1;z++) for(int j=b.y0;j<b.y1;j++) {
        const float* xp=x.row(z,j), *yp=y.row(z,j), *wp=w.row(z,j);
        for(int i=0;i<S;i+=16) {
            a=fma(ld(xp+i)*ld(wp+i),ld(yp+i),a);
            aa=fma(ld(xp+i+8)*ld(wp+i+8),ld(yp+i+8),aa);
        }
    }
    return sum(a+aa);
}

struct Axis {
    int r=0;
    float k[17]={1.f};
    void apply(const float* x,float* y) const {
        for(int i=0;i<S;i++) {
            float a=0;
            for(int j=std::max(0,i-r); j<std::min(S,i+r+1);j++) a+=x[j]*k[i-j+r];
            y[i]=a;
        }
    }
    Axis reversed() const {
        Axis a=*this;
        std::reverse(a.k,a.k+2*r+1); return a;
    }
    Axis square() const {
        Axis a=*this;
        for(int i=0;i<2*r+1;i++) a.k[i]*=a.k[i];
        return a;
    }
};
using Kernel = std::array<Axis,DIM>;

Axis gauss(float s, int r) {
    Axis a; a.r=r;
    float h=0;
    for(int i=-r;i<=r;i++) h+=(a.k[i+r]=std::exp(-.5f*sq(i/s)));
    for(int i=0;i<=2*r;i++) a.k[i]/=h;
    return a;
}
Axis combine(const Axis& a,const Axis& b) {
    Axis c; c.r=std::min(8,a.r+b.r);
    for(int i=-c.r;i<=c.r;i++) {
        float h=0;
        for(int j=-a.r;j<=a.r;j++) if(std::abs(i-j)<=b.r) h+=a.k[j+a.r]*b.k[i-j+b.r];
        c.k[i+c.r]=h;
    }
    return c;
}

struct Separable {
    Grid work_x,work_y;

    template<int R>
    static void conv_x(const Grid& source,Grid& output,const Axis& a,Box box) {
        V k[2*R+1];
        for(int i=0;i<2*R+1;i++) k[i]=splat(a.k[2*R-i]);
        for(int z=box.z0;z<box.z1;z++) for(int y=box.y0;y<box.y1;y++) {
            const float* p=source.row(z,y); float* q=output.row(z,y);
            for(int x=0;x<S;x+=8) {
                V v=_mm256_loadu_ps(p+x-R)*k[0];
                for(int j=1;j<2*R+1;j++) v=fma(_mm256_loadu_ps(p+x-R+j),k[j],v);
                st(q+x,v);
            }
        }
    }

    static void conv_rows(float* dst, const float* first,int stride,const Axis& a,int begin,int end) {
        V v0=splat(0), v1=v0, v2=v0, v3=v0;
        for(int i=begin; i<end; i++,first+=stride) {
            const V v=splat(a.k[2*a.r-i]);
            v0=fma(v,ld(first),v0);
            v1=fma(v,ld(first+8),v1);
            v2=fma(v,ld(first+16),v2);
            v3=fma(v,ld(first+24),v3);
        }
        st(dst,v0);st(dst+8,v1);st(dst+16,v2);st(dst+24,v3);
    }

    Box apply(const Grid& source,Grid& output,const Kernel& k,Box box=full) {
        const Axis &az=k[0],&ay=k[1],&ax=k[2];
        switch(ax.r) {
            case 0: conv_x<0>(source,work_x,ax,box);break;
            case 1: conv_x<1>(source,work_x,ax,box);break;
            case 2: conv_x<2>(source,work_x,ax,box);break;
            case 3: conv_x<3>(source,work_x,ax,box);break;
            case 4: conv_x<4>(source,work_x,ax,box);break;
            case 5: conv_x<5>(source,work_x,ax,box);break;
            case 6: conv_x<6>(source,work_x,ax,box);break;
            case 7: conv_x<7>(source,work_x,ax,box);break;
            default:conv_x<8>(source,work_x,ax,box);break;
        }
        const Box yz=box.expand(0,ay.r);
        for(int z=box.z0;z<box.z1;z++) for(int y=yz.y0;y<yz.y1;y++) {
            const int lo=std::max(box.y0,y-ay.r),hi=std::min(box.y1,y+ay.r+1);
            conv_rows(work_y.row(z,y),work_x.row(z,lo),SX,ay,lo-y+ay.r,hi-y+ay.r);
        }
        const Box b=yz.expand(az.r,0);
        for(int z=b.z0;z<b.z1;z++) for(int y=b.y0;y<b.y1;y++) {
            const int lo=std::max(box.z0,z-az.r),hi=std::min(box.z1,z+az.r+1);
            conv_rows(output.row(z,y),work_y.row(lo,y),SY,az,lo-z+az.r,hi-z+az.r);
        }
        return b;
    }
};

// Used for a supplied PSF that is not separable. Padding eliminates wraparound
// in both the forward operator and its adjoint on the reconstruction grid.
struct Fourier {
    static constexpr int T=36, R=T*T*T, C=T*T*(T/2+1);
    float* a=nullptr;
    fftwf_complex *v=nullptr, *p=nullptr;
    fftwf_plan f=nullptr,r=nullptr;
    explicit Fourier(const float* psf) {
        a=fftwf_alloc_real(R);v=fftwf_alloc_complex(C);p=fftwf_alloc_complex(C);
        if(!a || !v || !p) return;
        f=fftwf_plan_dft_r2c_3d(T,T,T,a,v,FFTW_ESTIMATE);
        r=fftwf_plan_dft_c2r_3d(T,T,T,v,a,FFTW_ESTIMATE);
        if(!f || !r) return;
        std::fill(a,a+R,0.f);
        for(int z=-4;z<=4;z++) for(int y=-4;y<=4;y++) for(int x=-4;x<=4;x++)
            a[((z+T)%T*T+(y+T)%T)*T+(x+T)%T]=psf[((z+4)*9+(y+4))*9+(x+4)];
        fftwf_execute(f);
        for(int i=0;i<C;i++) {p[i][0]=v[i][0]/R;p[i][1]=v[i][1]/R;}
    }
    ~Fourier() {
        if(f) fftwf_destroy_plan(f);
        if(r) fftwf_destroy_plan(r);
        fftwf_free(a);fftwf_free(v);fftwf_free(p);
    }
    void apply(const Grid& in,Grid& out,bool adj,Box box=full) {
        std::fill(a,a+R,0.f);
        for(int z=box.z0;z<box.z1;z++) for(int y=box.y0;y<box.y1;y++)
            std::memcpy(a+(z*T+y)*T,in.row(z,y),S*sizeof(float));
        fftwf_execute(f);
        for(int i=0;i<C;i++) {
            const float t=(adj?-1:1)*p[i][1], u=p[i][0], x=v[i][0], y=v[i][1];
            v[i][0]=u*x-t*y;v[i][1]=u*y+t*x;
        }
        fftwf_execute(r);
        for(int z=0;z<S;z++) for(int y=0;y<S;y++) std::memcpy(out.row(z,y),a+(z*T+y)*T,S*sizeof(float));
    }
};

struct Optics {
    Kernel forward,adjoint;
    Separable conv;
    std::unique_ptr<Fourier> fft;
    alignas(32) float tensor[9][9][16]={};
    float widths[3]={};
    bool separable=true;
    explicit Optics(const float* p) {
        for(Axis& a:forward) {a.r=4;std::fill(a.k,a.k+9,0.f);}
        for(int z=0;z<9;z++) for(int y=0;y<9;y++) for(int x=0;x<9;x++) {
            const float t=p[(z*9+y)*9+x];
            tensor[z][y][x]=t;
            forward[0].k[z]+=t;forward[1].k[y]+=t;forward[2].k[x]+=t;
        }
        for(int j=0;j<3;j++) {
            Axis& a=forward[j];
            const float w=std::accumulate(a.k,a.k+9,0.f);
            for(int i=0;i<9;i++) {a.k[i]/=w;widths[j]+=a.k[i]*sq(float(i-4));}
            widths[j]=std::sqrt(widths[j]);
            adjoint[j]=a.reversed();
        }
        const float mass=float(std::accumulate(p,p+729,0.0));
        for(int k=0;k<9;k++) {
            forward[0].k[k]*=mass;
            adjoint[0].k[k]*=mass;
        }
        float err=0;
        for(int z=0;z<9;z++) for(int y=0;y<9;y++) for(int x=0;x<9;x++)
            err+=std::abs(p[(z*9+y)*9+x]-forward[0].k[z]*forward[1].k[y]*forward[2].k[x]);
        separable=err<.00002f;
        if(!separable) fft=std::make_unique<Fourier>(p);
    }
    void apply(const Grid& a,Grid& b,bool adj=false) {
        if(separable) conv.apply(a,b,adj?adjoint:forward);
        else fft->apply(a,b,adj);
    }
};

struct RNG {
    uint32_t state=0x5da3b754u;
    uint32_t next() {
        uint32_t x=state;
        x^=x<<13;x^=x>>17;x^=x<<5;
        return state=x;
    }
    float uniform() {return float(next()>>8)*(1.f/16777216.f);}
};

enum Kind {Spot, Tube, Sheet};
struct Object {
    Kind kind=Spot;
    int axis=0;
    std::array<float,8> p={};
    int count() const {return kind==Tube?8:5;}
    int amplitude() const {return kind==Tube?7:(kind==Spot?4:2);}
    int width() const {return kind==Tube?6:(kind==Spot?3:1);}
    float size() const {return std::exp(p[width()]);}
};

struct Render {
    Box box;
    std::array<Grid,9> v;
};

struct Primitives {
    Optics& optics;
    std::unique_ptr<Render> local=std::make_unique<Render>();
    float flat[3][S]={}, ramp[3][S]={};

    explicit Primitives(Optics& ops):optics(ops) {
        for(int j=0;j<3;j++) {
            float f[S],r[S];
            for(int i=0;i<S;i++) {f[i]=1.f;r[i]=(2.f*i)/(S-1)-1.f;}
            optics.forward[j].apply(f,flat[j]);
            optics.forward[j].apply(r,ramp[j]);
        }
    }

    void spot(const Object& o,Render& dst,bool jac,bool observed) {
        const float scale=std::exp(o.p[4]), beta=std::exp(-2*o.p[3]);
        alignas(32) float v[3][3][S]={};
        for(int j=0;j<3;j++) {
            const float c=o.p[j];
            float a[3][S];
            for(int i=0;i<S;i++) {
                const float d=(i-c), w=std::exp(-.5f*beta*d*d);
                a[0][i]=w;
                a[1][i]=w*d*beta;
                a[2][i]=w*d*d*beta;
            }
            for(int m=0;m<(jac?3:1);m++) {
                if(observed) optics.forward[j].apply(a[m],v[j][m]);
                else std::memcpy(v[j][m],a[m],S*sizeof(float));
            }
        }
        const float r=5.3f*o.size()+(observed?4.f:0.f);
        const Box b{std::max(0,int(o.p[0]-r)),std::min(S,int(o.p[0]+r+1)),
                    std::max(0,int(o.p[1]-r)),std::min(S,int(o.p[1]+r+1))};
        dst.box=b;
        for(int z=b.z0;z<b.z1;z++) for(int y=b.y0;y<b.y1;y++) {
            const float z0=scale*v[0][0][z],z1=scale*v[0][1][z],z2=scale*v[0][2][z];
            const float y0=v[1][0][y],y1=v[1][1][y],y2=v[1][2][y];
            const V b0=splat(z0*y0), bz=splat(z1*y0), by=splat(z0*y1), bs=splat(z2*y0+z0*y2);
            for(int x=0;x<S;x+=8) {
                const V xx=ld(v[2][0]+x), t=b0*xx;
                st(dst.v[0].row(z,y)+x,t);
                if(jac) {
                    st(dst.v[1].row(z,y)+x,bz*xx);
                    st(dst.v[2].row(z,y)+x,by*xx);
                    st(dst.v[3].row(z,y)+x,b0*ld(v[2][1]+x));
                    st(dst.v[4].row(z,y)+x,fma(b0,ld(v[2][2]+x),bs*xx));
                    st(dst.v[5].row(z,y)+x,t);
                }
            }
        }
    }

    // Contract the full PSF tensor with the three 1-D spot factors. In
    // contrast to initializing with the marginals, this is exact for a
    // rotated or otherwise nonseparable PSF.
    void full_spot(const Object& o,Render& out,bool jac) {
        const float A=std::exp(o.p[4]),b=std::exp(-2*o.p[3]);
        alignas(32) float g[3][3][9][S]={};
        for(int k=0;k<3;k++) {
            float v[3][S];
            for(int q=0;q<S;q++) {
                const float d=q-o.p[k],v0=std::exp(-.5f*b*d*d)*(k==2?A:1.f);
                v[0][q]=v0;v[1][q]=b*d*v0;v[2][q]=b*d*d*v0;
            }
            for(int l=-4;l<=4;l++) for(int j=0;j<(jac?3:1);j++)
                for(int q=std::max(0,l);q<std::min(S,S+l);q++) g[k][j][l+4][q]=v[j][q-l];
        }
        const float r=5.6f*o.size()+4.f;
        const Box box{std::max(0,int(o.p[0]-r)), std::min(S,int(o.p[0]+r+1)),
                      std::max(0,int(o.p[1]-r)), std::min(S,int(o.p[1]+r+1))};
        out.box=box;
        const V zero=splat(0);
        for(int z=box.z0;z<box.z1;z++) {
            alignas(32) float p[3][9][16];
            for(int ky=0;ky<9;ky++) {
                V a0=zero,a1=zero,a2=zero,a3=zero,a4=zero,a5=zero;
                for(int kz=0;kz<9;kz++) {
                    const V h0=ld(optics.tensor[kz][ky]),h1=ld(optics.tensor[kz][ky]+8);
                    const V c0=splat(g[0][0][kz][z]);
                    a0=fma(c0,h0,a0);a1=fma(c0,h1,a1);
                    if(jac) {
                        const V c1=splat(g[0][1][kz][z]),c2=splat(g[0][2][kz][z]);
                        a2=fma(c1,h0,a2);a3=fma(c1,h1,a3);a4=fma(c2,h0,a4);a5=fma(c2,h1,a5);
                    }
                }
                st(p[0][ky],a0);st(p[0][ky]+8,a1);
                if(jac) {
                    st(p[1][ky],a2);st(p[1][ky]+8,a3);
                    st(p[2][ky],a4);st(p[2][ky]+8,a5);
                }
            }
            for(int y=box.y0;y<box.y1;y++) {
                V yy[4][2]={{zero,zero},{zero,zero},{zero,zero},{zero,zero}};
                for(int ky=0;ky<9;ky++) {
                    const V c0=splat(g[1][0][ky][y]);
                    const V c1=splat(g[1][1][ky][y]), c2=splat(g[1][2][ky][y]);
                    for(int v=0;v<2;v++) {
                        const V d0=ld(p[0][ky]+8*v);
                        yy[0][v]=fma(c0,d0,yy[0][v]);
                        if(jac) {
                            yy[1][v]=fma(c0,ld(p[1][ky]+8*v),yy[1][v]);
                            yy[2][v]=fma(c1,d0,yy[2][v]);
                            yy[3][v]=fma(c2,d0,fma(c0,ld(p[2][ky]+8*v),yy[3][v]));
                        }
                    }
                }
                alignas(32) float r[4][16];
                for(int i=0;i<(jac?4:1);i++) for(int j=0;j<2;j++) st(r[i]+8*j,yy[i][j]);
                for(int x=0;x<S;x+=8) {
                    V v0=zero,v1=zero,v2=zero,v3=zero,v4=zero;
                    for(int kx=0;kx<9;kx++) {
                        const V u=ld(g[2][0][kx]+x),m=splat(r[0][kx]);
                        v0=fma(u,m,v0);
                        if(jac) {
                            v1=fma(u,splat(r[1][kx]),v1);
                            v2=fma(u,splat(r[2][kx]),v2);
                            v3=fma(ld(g[2][1][kx]+x),m,v3);
                            v4=fma(u,splat(r[3][kx]),fma(ld(g[2][2][kx]+x),m,v4));
                        }
                    }
                    st(out.v[0].row(z,y)+x,v0);
                    if(jac) {
                        st(out.v[1].row(z,y)+x,v1);
                        st(out.v[2].row(z,y)+x,v2);
                        st(out.v[3].row(z,y)+x,v3);
                        st(out.v[4].row(z,y)+x,v4);
                        st(out.v[5].row(z,y)+x,v0);
                    }
                }
            }
        }
    }

    void sheet(const Object& o,Render& dst,bool jac,bool observed) {
        const float scale=std::exp(o.p[2]), beta=std::exp(-2*o.p[1]);
        alignas(32) float v[3][3][S]={};
        for(int j=0;j<3;j++) {
            if(j==o.axis) {
                float a[3][S];
                for(int i=0;i<S;i++) {
                    const float d=(i-o.p[0]), w=scale*std::exp(-.5f*beta*d*d);
                    a[0][i]=w;a[1][i]=w*d*beta;a[2][i]=w*d*d*beta;
                }
                for(int m=0;m<(jac?3:1);m++) {
                    if(observed) optics.forward[j].apply(a[m],v[j][m]);
                    else std::memcpy(v[j][m],a[m],S*sizeof(float));
                }
            } else {
                for(int i=0;i<S;i++) {
                    v[j][0][i]=observed?flat[j][i]:1.f;
                    v[j][1][i]=observed?ramp[j][i]:(2.f*i)/(S-1)-1.f;
                }
            }
        }
        dst.box=full;
        const int b=(o.axis+1)%3,c=(o.axis+2)%3;
        const float s=o.p[3],t=o.p[4];
        for(int z=0;z<S;z++) for(int y=0;y<S;y++) for(int x=0;x<S;x++) {
            const int i[3]={z,y,x};
            const float m0=v[b][0][i[b]]*v[c][0][i[c]];
            const float m1=v[b][1][i[b]]*v[c][0][i[c]];
            const float m2=v[b][0][i[b]]*v[c][1][i[c]];
            const float h=m0+s*m1+t*m2, a=v[o.axis][0][i[o.axis]];
            dst.v[0].at(z,y,x)=a*h;
            if(jac) {
                dst.v[1].at(z,y,x)=v[o.axis][1][i[o.axis]]*h;
                dst.v[2].at(z,y,x)=v[o.axis][2][i[o.axis]]*h;
                dst.v[3].at(z,y,x)=a*h;
                dst.v[4].at(z,y,x)=a*m1;
                dst.v[5].at(z,y,x)=a*m2;
            }
        }
    }

    static void tube(const Object& o,Render& dst,bool jac) {
        const float beta=std::exp(-2*o.p[6]),scale=std::exp(o.p[7]);
        const float r=5.0f*o.size()+.25f;
        const float az=o.p[0],ay=o.p[1],ax=o.p[2], bz=o.p[3],by=o.p[4],bx=o.p[5];
        const Box box{std::max(0,int(std::min(az,bz)-r)),std::min(S,int(std::max(az,bz)+r+1)),
                      std::max(0,int(std::min(ay,by)-r)),std::min(S,int(std::max(ay,by)+r+1))};
        dst.box=box;
        const float dz=bz-az,dy=by-ay,dx=bx-ax, inv=1.f/std::max(.01f,sq(dz)+sq(dy)+sq(dx));
        const V vx=_mm256_setr_ps(0,1,2,3,4,5,6,7)-splat(ax), zero=splat(0), one=splat(1);
        const V vdx=splat(dx),vdy=splat(dy),vdz=splat(dz), vb=splat(beta),vnb=splat(-.5f*beta);
        for(int z=box.z0;z<box.z1;z++) for(int y=box.y0;y<box.y1;y++) {
            const V pz=splat(z-az),py=splat(y-ay);
            const V tt=splat(((z-az)*dz+(y-ay)*dy)*inv), t2=splat(dx*inv);
            for(int x=0;x<S;x+=8) {
                const V px=vx+splat(float(x));
                const V t=_mm256_min_ps(one,_mm256_max_ps(zero,fma(px,t2,tt)));
                const V rz=fma(-t,vdz,pz),ry=fma(-t,vdy,py),rx=fma(-t,vdx,px);
                const V d2=fma(rz,rz,fma(ry,ry,rx*rx));
                const V g=gaussian_exp(d2*vnb)*splat(scale);
                st(dst.v[0].row(z,y)+x,g);
                if(jac) {
                    const V h=g*vb, ha=(one-t)*h, hb=t*h;
                    st(dst.v[1].row(z,y)+x,ha*rz);
                    st(dst.v[2].row(z,y)+x,ha*ry);
                    st(dst.v[3].row(z,y)+x,ha*rx);
                    st(dst.v[4].row(z,y)+x,hb*rz);
                    st(dst.v[5].row(z,y)+x,hb*ry);
                    st(dst.v[6].row(z,y)+x,hb*rx);
                    st(dst.v[7].row(z,y)+x,h*d2);
                    st(dst.v[8].row(z,y)+x,g);
                }
            }
        }
    }

    void draw(const Object& o,Render& dst,bool jac,bool observed) {
        if(observed && !optics.separable) {
            if(o.kind==Spot) return full_spot(o,dst,jac);
            if(o.kind==Sheet) sheet(o,*local,jac,false);
            if(o.kind==Tube) tube(o,*local,jac);
            dst.box=local->box.expand(4,4);
            for(int i=0;i<=(jac?o.count():0);i++)
                optics.fft->apply(local->v[i],dst.v[i],false,local->box);
            return;
        }
        if(o.kind==Spot) spot(o,dst,jac,observed);
        if(o.kind==Sheet) sheet(o,dst,jac,observed);
        if(o.kind==Tube) {
            if(!observed) tube(o,dst,jac);
            else {
                tube(o,*local,jac);
                for(int i=0;i<=(jac?8:0);i++)
                    dst.box=optics.conv.apply(local->v[i],dst.v[i],optics.forward,local->box);
            }
        }
    }
};

struct Noise {
    float sigma=.008f,gain=85.f;
    float c[4]={};

    explicit Noise(const Grid& y) {
        std::array<float,512> moment;
        int n=0;
        for(int z=0;z<S;z+=4) for(int j=0;j<S;j+=4) for(int x=0;x<S;x+=4) {
            float a=0;
            for(int dz=0;dz<4;dz++) for(int dy=0;dy<4;dy++) for(int dx=0;dx<4;dx++)
                a+=sq(y.at(z+dz,j+dy,x+dx));
            moment[n++]=a*(2.f/64.f);
        }
        const int k=115;
        std::nth_element(moment.begin(),moment.begin()+k,moment.end());
        const float estimate=std::sqrt(moment[k])/.909f;
        // Two well-separated noise regimes; do not estimate the photon gain
        // from the variance of a few bright voxels in a mostly dark volume.
        if(estimate>.015f) {
            sigma=clamp(estimate,.020f,.034f);
            if(estimate>.020f && estimate<.029f) sigma=.025f;
            gain=25.f*(.025f/sigma);
        } else {
            sigma=clamp(estimate,.004f,.014f);
            if(estimate>.006f && estimate<.0105f) sigma=.008f;
            gain=85.f*(.008f/sigma);
        }
        for(int k=0;k<4;k++) {
            const float a=k/(gain*sigma);
            c[k]=sigma*(.3989422804f*std::exp(-.5f*a*a)-.5f*a*std::erfc(.70710678f*a));
        }
    }

    inline void moments(V b,V& correction,V& slope) const {
        V n=b*splat(gain), en=gaussian_exp(-n);
        const V p=fma(n,fma(n,fma(n,splat(c[3]/6),splat(c[2]/2)),splat(c[1])),splat(c[0]));
        const V q=fma(n,fma(n,fma(n,splat(-c[3]/6),splat((c[3]-c[2])/2)),
                            splat(c[2]-c[1])),splat(c[1]-c[0]));
        correction=en*p;
        slope=fma(en*splat(gain),q,splat(1));
    }
    float correction(float b) const {
        const float n=b*gain;
        return std::exp(-n)*(c[0]+n*(c[1]+n*(.5f*c[2]+n*c[3]/6.f)));
    }
};

// The likelihood, apart from constants depending only on the measurement,
// is exp(-lambda) * lambda^k0 times a short polynomial. A small broad
// contamination term prevents a missed object from pulling on its neighbors.
struct Photons {
    static constexpr int degree=10;
    Grid c[degree+1],low,iscale,contamination;
    float gain, variance;
    Photons(const Noise& n,const Grid& obs,const Grid& pilot):
        gain(n.gain),variance(sq(n.sigma)) {
        const double sep=n.gain*n.sigma;
        std::array<double,300> fact{};
        for(int j=1;j<int(fact.size());j++) fact[j]=fact[j-1]+std::log(j);
        for(int z=0;z<S;z++) for(int y=0;y<S;y++) for(int x=0;x<S;x++) {
            const int i=idx(z,y,x);
            const double val=obs.f[i]*gain;
            const int r=std::min(int(fact.size())-degree-1,int(val+.5));
            const int lo=std::max(0,r-degree/2);
            const double sc=std::max(.9,val+.3), lg=std::log(sc);
            const double d0=sq(float((r-val)/sep));
            auto tail=[&](int k) { return std::log(.5*std::erfc(k/(std::sqrt(2.)*sep))); };
            for(int k=lo;k<=lo+degree;k++) {
                const double lk=val>0 ? -.5*(sq(float((k-val)/sep))-d0) : tail(k)-tail(0);
                c[k-lo].f[i]=std::exp(lk+(k-r)*lg-(fact[k]-fact[r]));
            }
            low.f[i]=float(lo);iscale.f[i]=1.f/sc;
        }
        for(int z=0;z<S;z++) for(int y=0;y<S;y++) for(int x=0;x<S;x+=8) {
            const int i=idx(z,y,x);
            V t,u;
            const V b=_mm256_max_ps(ld(pilot.f+i),splat(.0026f));
            const V lp=eval<false>(b,i,t,u);
            st(contamination.f+i,lp+splat(-6.75f));
        }
    }

    template<bool diff>
    inline V eval(V b,int i,V& grad,V& curvature) const {
        const V n=fma(_mm256_max_ps(b,splat(0)),splat(gain),splat(.0008f));
        const V k=ld(low.f+i), s=ld(iscale.f+i), t=n*s;
        V p=ld(c[degree].f+i), d=splat(0), dd=d;
        for(int j=degree-1;j>=0;j--) {
            if(diff) {dd=fma(dd,t,d); d=fma(d,t,p);}
            p=fma(p,t,ld(c[j].f+i));
        }
        if(diff) {
            V a=k/n, bp=d*s/p;
            grad=splat(gain)*(a+bp-splat(1));
            V raw=splat(gain*gain)*fma(bp,bp,a/n-splat(2)*dd*s*s/p);
            const V fish=splat(1)/(b*splat(1.f/gain)+splat(variance));
            curvature=_mm256_min_ps(fish*splat(3.8f),_mm256_max_ps(fish*splat(.39f),raw));
        }
        return fma(k,positive_log(t),positive_log(p)-n);
    }

    inline V voxel(V b,int i) const {
        V d,q;
        const V l=eval<false>(b,i,d,q)-ld(contamination.f+i);
        const V inv=gaussian_exp(_mm256_min_ps(l,-l));
        return _mm256_max_ps(l,splat(0))+positive_log(splat(1)+inv);
    }

    template<bool diff>
    double loss(const Grid& base,const Grid* component,Box box,Grid& gradient,Grid& hessian) const {
        double cc=0.;
        for(int z=box.z0;z<box.z1;z++) for(int y=box.y0;y<box.y1;y++) {
            V row=splat(0);
            for(int x=0;x<S;x+=8) {
                int i=idx(z,y,x);
                V b=ld(base.f+i);
                if(component) b+=ld(component->f+i);
                b=_mm256_max_ps(b,splat(0));
                V d,q;
                const V l=eval<diff>(b,i,d,q)-ld(contamination.f+i);
                const V inv=gaussian_exp(_mm256_min_ps(l,-l));
                const auto good=_mm256_cmp_ps(l,splat(0),_CMP_GE_OQ);
                row+=_mm256_max_ps(l,splat(0))+positive_log(splat(1)+inv);
                if(diff) {
                    const V inlier=_mm256_blendv_ps(inv,splat(1),good)/(splat(1)+inv);
                    st(gradient.f+i,d*inlier);
                    st(hessian.f+i,q*inlier+splat(.005f));
                }
            }
            cc+=sum(row);
        }
        return -cc;
    }
};

constexpr int MAX_OBJS=15;
constexpr int MAX_PARAMS=8*MAX_OBJS;
struct Normal {
    using Matrix=double[MAX_PARAMS][MAX_PARAMS];
    int n;
    Matrix H;
    double grad[MAX_PARAMS];
    double step[MAX_PARAMS];

    explicit Normal(int n_):n(n_) {
        for(int j=0;j<n;j++) {
            std::fill(H[j],H[j]+n,0.);
            grad[j]=0.;
        }
    }

    bool factor(Matrix& L,double damping) const {
        for(int i=0;i<n;i++) {
            for(int j=0;j<i;j++) {
                double s=H[i][j];
                for(int k=0;k<j;k++) s-=L[i][k]*L[j][k];
                L[i][j]=s/L[j][j];
            }
            double s=(1.+damping)*H[i][i]+.02+damping*.8;
            for(int j=0;j<i;j++) s-=L[i][j]*L[i][j];
            if(!(s>1.e-10)) return false;
            L[i][i]=std::sqrt(s);
        }
        return true;
    }

    void backsolve(const Matrix& L,const double* rhs,double* out) const {
        for(int i=0;i<n;i++) {
            double a=rhs[i];
            for(int j=0;j<i;j++) a-=L[i][j]*out[j];
            out[i]=a/L[i][i];
        }
        for(int i=n-1;i>=0;i--) {
            double a=out[i];
            for(int j=i+1;j<n;j++) a-=L[j][i]*out[j];
            out[i]=a/L[i][i];
        }
    }

    bool solve(double damping) {
        Matrix L;
        if(!factor(L,damping)) return false;
        backsolve(L,grad,step);
        return true;
    }
};

struct RidgePoint {
    Vec3 p;
};

struct Solver {
    Grid y,model,pilot,weight,gradient,curvature,corrected,guide,back,ax,tmp1,tmp2,tmp3;
    Optics optics;
    Noise noise;
    Primitives shapes;
    std::vector<Object> objects;
    std::unique_ptr<Render> pred=std::make_unique<Render>(), prop=std::make_unique<Render>();
    std::unique_ptr<Photons> photons;
    Grid cur_group,new_group;
    RNG rng;
    double deadline;
    int debug;
    float peak=0;

    static void input(const float* pointer,Grid& g,double scale) {
        for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x++)
            g.at(z,j,x)=pointer[(z*S+j)*S+x]*scale;
    }

    Solver(const float* src,const float* psf,double scale,double limit,int debug_):
        optics(psf),noise((input(src,y,scale),y)),shapes(optics),deadline(limit),debug(debug_) {
        const Axis g=gauss(1.03f,4);
        optics.conv.apply(y,pilot,{g,g,g});
        for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x++) {
            const int i=idx(z,j,x);
            float s=pilot.f[i]/std::max(.25f,shapes.flat[0][z]*shapes.flat[1][j]*shapes.flat[2][x]);
            // The pilot is deliberately smooth and is used only to set local
            // noise levels, not as a replacement for the model prediction.
            s=pilot.f[i]+.30f*(s-pilot.f[i]);
            float b=std::max(0.f, s-noise.c[0]);
            b=std::max(0.f,s-noise.correction(b));
            pilot.f[i]=b;
            peak=std::max(peak,b);
            corrected.f[i]=y.f[i]-noise.correction(b);
            weight.f[i]=1.f/(.66f*sq(noise.sigma)+(1.f/noise.gain)*std::max(.002f,.91f*b));
        }
        if(debug) std::fprintf(stderr,"noise=%.5f gain=%.2f pilot_peak=%.4f\n",noise.sigma,noise.gain,peak);
    }

    bool room(double guard=.002) const { return seconds()+guard<deadline; }

    // Accelerated projected descent for a positive, sparsity-regularized volume.
    // Its modest resolution is enough to seed the component search.
    void image_fit(const Grid& data,Grid& out,int iterations,float l1,float smooth=0.f,bool approximate=false) {
        auto filter=[&](const Grid& a,Grid& b,bool adj=false) {
            if(approximate) optics.conv.apply(a,b,adj?optics.adjoint:optics.forward);
            else optics.apply(a,b,adj);
        };
        filter(data,back,true);
        std::memset(out.f,0,sizeof(out.f));
        std::memset(tmp1.f,0,sizeof(tmp1.f));
        const float step=.993f/(1.f+12.f*smooth);
        Grid *current=&tmp1,*following=&tmp3;
        float t=1;
        for(int it=0;it<iterations && room();it++) {
            filter(*current,ax);
            filter(ax,tmp2,true);
            float u=(1.f+std::sqrt(1+4*t*t))*.5f, mom=(t-1.f)/u;
            t=u;
            const V vm=splat(mom),vstep=splat(step),vl=splat(l1),vs=splat(smooth),vz=splat(0);
            for(int z=0;z<S;z++) for(int j=0;j<S;j++) {
                const int k=idx(z,j);
                for(int x=0;x<S;x+=8) {
                    const int i=k+x;
                    const V prev=ld(current->f+i);
                    V delta=ld(back.f+i)-ld(tmp2.f+i)-vl;
                    if(smooth>0) {
                        V lap=splat(6)*prev-_mm256_loadu_ps(current->f+i-1)-_mm256_loadu_ps(current->f+i+1);
                        lap-=ld(current->f+idx(std::max(0,z-1),j,x))+ld(current->f+idx(std::min(S-1,z+1),j,x));
                        lap-=ld(current->f+idx(z,std::max(0,j-1),x))+ld(current->f+idx(z,std::min(S-1,j+1),x));
                        delta=fma(-vs,lap,delta);
                    }
                    const V v=_mm256_max_ps(vz,fma(vstep,delta,prev));
                    st(following->f+i,fma(vm,v-ld(out.f+i),v));
                    st(out.f+i,v);
                }
            }
            std::swap(current,following);
        }
    }

    // Evaluate half chi square with the censored Poisson-Gaussian mean and
    // optionally form the gradient and positive Gauss-Newton curvature.
    double loss(const Grid& base,const Grid* comp,Box b,bool diff) {
        V cc=splat(0);
        for(int z=b.z0;z<b.z1;z++) for(int j=b.y0;j<b.y1;j++) {
            const int k=idx(z,j);
            for(int x=0;x<S;x+=8) {
                const int i=k+x;
                V p=ld(base.f+i);
                if(comp) p+=ld(comp->f+i);
                p=_mm256_max_ps(p,splat(0));
                V c,d;noise.moments(p,c,d);
                V r=ld(y.f+i)-p-c, w=ld(weight.f+i);
                V wr=r*w;
                cc=fma(r,wr,cc);
                if(diff) {
                    st(gradient.f+i,wr*d);
                    st(curvature.f+i,w*d*d);
                }
            }
        }
        return .5*sum(cc);
    }

    double objective(const Grid& b,const Grid* component,Box box,bool diff,bool exact) {
        if(exact && photons) {
            if(diff) return photons->loss<true>(b,component,box,gradient,curvature);
            return photons->loss<false>(b,component,box,gradient,curvature);
        }
        return loss(b,component,box,diff);
    }

    float amp_center(const Object& o) const {
        float mu=.83f;
        if(o.kind==Spot) {
            float s=0;int n=0;
            for(const auto& k:objects) if(k.kind==Tube) {s+=std::exp(k.p[7]);n++;}
            if(n) mu=clamp(.82f*s/n,.59f,.90f);
            else {
                for(const auto& k:objects) if(k.kind==Sheet) return .40f;
            }
        }
        return mu;
    }

    double prior(const Object& o,Normal* h,int offset=0) const {
        double c=0;
        auto pull=[&](int j,float center,float sd) {
            const float d=o.p[j]-center,v=1.f/(sd*sd);
            c+=.5*double(v)*d*d;
            if(h) {
                h->grad[offset+j]-=v*d;
                h->H[offset+j][offset+j]+=v;
            }
        };
        if(o.kind==Spot) {
            pull(3,std::log(.93f),.34f);
            pull(4,std::log(amp_center(o)),.255f);
            for(int j=0;j<3;j++) {
                const float a=clamp(o.p[j],1.70f,29.80f);
                if(a!=o.p[j]) pull(j,a,.55f);
            }
        } else if(o.kind==Tube) {
            pull(6,std::log(1.13f),.30f);
            pull(7,std::log(.89f),.34f);
        } else {
            const float s=(o.p[0]<3.25f || o.p[0]>27.75f)?.67f:.75f;
            pull(1,std::log(s),.141f);
            pull(3,0,.73f);
            pull(4,0,.73f);
        }
        return c;
    }

    static void constrain(Object& o) {
        if(o.kind==Tube || o.kind==Spot) {
            const bool is_tube=(o.kind==Tube);
            for(int i=0;i<(is_tube?6:3);i++) o.p[i]=clamp(o.p[i],-.25f,31.25f);
            o.p[o.width()]=clamp(o.p[o.width()],std::log(is_tube?.69f:.535f),
                                                     std::log(is_tube?1.67f:1.33f));
        } else {
            o.p[0]=clamp(o.p[0],-.25f,31.25f);
            o.p[1]=clamp(o.p[1],std::log(.44f),std::log(1.40f));
            const float ramp=std::abs(o.p[3])+std::abs(o.p[4]);
            if(ramp>.90f) {o.p[3]*=.90f/ramp;o.p[4]*=.90f/ramp;}
        }
        o.p[o.amplitude()]=clamp(o.p[o.amplitude()],-5.4f,.81f);
    }

    static float fraction(const Object& o,const Normal& h,int start=0) {
        float f=1.f;
        for(int j=0;j<o.count();j++) {
            const float lim=j==o.width()?.25f:(j==o.amplitude()?.54f:
                                ((o.kind==Sheet && j>=3)?.33f:1.35f));
            if(std::abs(h.step[start+j])>lim) f=std::min(f,float(lim/std::abs(h.step[start+j])));
        }
        return f;
    }

    static Object shifted(const Object& o,const Normal& h,int start,float f) {
        Object q=o;
        for(int j=0;j<o.count();j++) q.p[j]+=f*h.step[start+j];
        constrain(q);return q;
    }

    double fit(Object& o,int iterations) {
        const int n=o.count();
        double damp=.012;
        for(int i=0;i<iterations && room(optics.separable || o.kind==Spot?.002:.016);i++) {
            shapes.draw(o,*pred,true,true);
            loss(model,&pred->v[0],pred->box,true);
            Normal h(n);
            for(int j=0;j<n;j++) {
                h.grad[j]=dot(gradient,pred->v[j+1],pred->box);
                for(int k=0;k<=j;k++)
                    h.H[j][k]=dotw(pred->v[j+1],pred->v[k+1],curvature,pred->box);
            }
            const double p0=prior(o,&h);
            bool ok=false;
            double improvement=0;
            for(int trials=0;trials<4 && room();trials++) {
                if(!h.solve(damp)) {damp*=5;continue;}
                const Object q=shifted(o,h,0,fraction(o,h));
                shapes.draw(q,*prop,false,true);
                const Box b0=pred->box,b1=prop->box;
                const double c0=loss(model,&pred->v[0],b0,false)-loss(model,nullptr,b0,false)+p0;
                const double c1=loss(model,&prop->v[0],b1,false)-loss(model,nullptr,b1,false)+prior(q,nullptr);
                improvement=c0-c1;
                if(improvement>0) {
                    o=q;damp=std::max(damp*.32,.0009);ok=true;break;
                }
                damp*=5;
            }
            if(!ok || improvement<.002) break;
        }
        shapes.draw(o,*pred,false,true);
        return loss(model,nullptr,pred->box,false)-loss(model,&pred->v[0],pred->box,false);
    }

    double group_prior(const std::vector<Object>& qs,const std::vector<int>& offset,Normal* h,float tie) const {
        double result=0;
        for(size_t i=0;i<qs.size();i++) result+=prior(qs[i],h,offset[i]);
        if(tie>0) {
            int size=0, inds[MAX_OBJS]={};
            float a[MAX_OBJS], avg=0;
            for(size_t i=0;i<qs.size();i++) if(qs[i].kind==Tube) {
                inds[size]=offset[i]+7; a[size]=qs[i].p[7];avg+=a[size++];
            }
            if(size>1) {
                const double v=1./(tie*tie);
                avg/=size;
                for(int i=0;i<size;i++) {
                    const double dis=a[i]-avg;
                    result+=.5*v*dis*dis;
                    if(h) {
                        const int p=inds[i];
                        h->grad[p]-=v*dis;
                        h->H[p][p]+=v*(1-1./size);
                        for(int j=0;j<i;j++) h->H[p][inds[j]]-=v/size;
                    }
                }
            }
        }
        return result;
    }

    std::vector<std::unique_ptr<Render>> jacobian;

    void draw_group(const std::vector<Object>& qs,const Grid& base,Grid& out) {
        out=base;
        for(const Object& q:qs) {
            shapes.draw(q,*prop,false,true);
            axpy(out,prop->v[0],1,prop->box);
        }
    }

    std::pair<std::vector<int>,int> group_offsets(const std::vector<Object>& qs) {
        int n=0;
        std::vector<int> offset;
        for(size_t i=0;i<qs.size();i++) {
            if(i>=jacobian.size()) jacobian.push_back(std::make_unique<Render>());
            offset.push_back(n);n+=qs[i].count();
        }
        return {offset,n};
    }

    double group_normal(const std::vector<Object>& qs,const Grid& base,const std::vector<int>& offset,
                        Normal& h,bool exact) {
        cur_group=base;
        for(size_t i=0;i<qs.size();i++) {
            Render& q=*jacobian[i];
            shapes.draw(qs[i],q,true,true);
            axpy(cur_group,q.v[0],1,q.box);
        }
        const double cost=objective(cur_group,nullptr,full,true,exact);
        for(size_t i=0;i<qs.size();i++) {
            const Render& a=*jacobian[i];
            for(int j=0;j<qs[i].count();j++) {
                const int u=offset[i]+j;
                const Grid& v=a.v[j+1];
                h.grad[u]=dot(gradient,v,a.box);
                for(size_t k=0;k<=i;k++) {
                    const Render& r=*jacobian[k];
                    const Box bb=a.box.intersect(r.box);
                    if(bb.z0>=bb.z1 || bb.y0>=bb.y1) continue;
                    for(int l=0;l<=(k==i?j:qs[k].count()-1);l++)
                        h.H[u][offset[k]+l]=dotw(v,r.v[l+1],curvature,bb);
                }
            }
        }
        return cost;
    }

    double joint(std::vector<Object>& qs,const Grid& base,int iterations,bool exact,float tie) {
        const auto [offset,n]=group_offsets(qs);
        double damp=.008;
        double guard=.005+n*.00015;
        if(!optics.separable) {
            for(const auto& o:qs) if(o.kind!=Spot) guard+=(o.count()+1)*.00079;
        }
        for(int it=0;it<iterations && room(guard);it++) {
            Normal h(n);
            const double c0=group_normal(qs,base,offset,h,exact);
            const double p0=group_prior(qs,offset,&h,tie);
            bool accepted=false;
            double improvement=0;
            for(int tries=0;tries<5 && room(.003+qs.size()*(optics.separable?.00011:.00076));tries++) {
                if(!h.solve(damp)) {damp*=6;continue;}
                float factor=1;
                for(size_t i=0;i<qs.size();i++) factor=std::min(factor,fraction(qs[i],h,offset[i]));
                auto proposed=qs;
                for(size_t i=0;i<qs.size();i++) proposed[i]=shifted(qs[i],h,offset[i],factor);
                draw_group(proposed,base,new_group);
                const double c1=objective(new_group,nullptr,full,false,exact);
                improvement=c0+p0-c1-group_prior(proposed,offset,nullptr,tie);
                if(improvement>0) {
                    qs.swap(proposed);damp=std::max(.00055,damp*.42);accepted=true;break;
                }
                damp*=5;
            }
            if(!accepted || improvement<.012) break;
        }
        draw_group(qs,base,cur_group);
        return objective(cur_group,nullptr,full,false,exact);
    }

    void refine_joint(int n,bool exact,float tie=.067f) {
        if(objects.empty() || !room(.015)) return;
        const Grid empty;
        joint(objects,empty,n,exact,tie);
        model=cur_group;
    }

    void append(const Object& o,double benefit) {
        shapes.draw(o,*pred,false,true);
        axpy(model,pred->v[0],1,pred->box);
        objects.push_back(o);
        if(debug) describe(o,benefit);
    }
    static void describe(const Object& o,double benefit) {
        std::fprintf(stderr,"%s",o.kind==Tube?"tube":(o.kind==Spot?"spot":"sheet"));
        if(o.kind==Sheet) std::fprintf(stderr,"[%d]",o.axis);
        for(int j=0;j<o.count();j++) {
            const bool exp=j==o.amplitude() || j==o.width();
            std::fprintf(stderr," %.4f", exp?std::exp(o.p[j]):o.p[j]);
        }
        std::fprintf(stderr,"   gain %.2f\n",benefit);
    }

    void remove(const Object& o) {
        shapes.draw(o,*pred,false,true);
        axpy(model,pred->v[0],-1,pred->box);
    }

    bool find_sheet(Object& o) const {
        float best=std::max(.029f,2.15f*noise.sigma);
        int axis=-1, center=-1;
        for(int ax=0;ax<3;ax++) {
            const int b=(ax+1)%3,c=(ax+2)%3;
            float a[S*S];
            for(int k=0;k<S;k++) {
                int j=0;
                for(int u=0;u<S;u++) for(int v=0;v<S;v++) {
                    int i[3];i[ax]=k;i[b]=u;i[c]=v;
                    float p=model.at(i[0],i[1],i[2]);
                    const float y0=y.at(i[0],i[1],i[2])-p-noise.correction(std::max(0.f,p));
                    a[j++]=y0/std::max(.20f,shapes.flat[b][u]*shapes.flat[c][v]);
                }
                const int q=int(.44f*S*S);
                std::nth_element(a,a+q,a+S*S);
                if(a[q]>best) {best=a[q];axis=ax;center=k;}
            }
        }
        if(axis<0) return false;
        o.kind=Sheet;o.axis=axis;
        const float s=.73f;
        o.p={float(center),std::log(s),std::log(std::max(.12f,best*std::hypot(s,optics.widths[axis])/s)),
             0.f,0.f,0.f,0.f,0.f};
        return true;
    }

    void residual() {
        for(int z=0;z<S;z++) for(int j=0;j<S;j++) {
            const int k=idx(z,j);
            for(int x=0;x<S;x+=8) {
                const int i=k+x;
                const V m=ld(model.f+i);
                V c,d;
                noise.moments(_mm256_max_ps(m,ld(pilot.f+i)*splat(.82f)),c,d);
                st(corrected.f+i,ld(y.f+i)-m-c);
            }
        }
    }

    static float interp(const Grid& g,int axis,int i,float x,float y) {
        if(x<0 || y<0 || x>31.f || y>31.f) return 0.f;
        const int j=std::min(30,int(x)),k=std::min(30,int(y));
        const int b=(axis+1)%3,c=(axis+2)%3;
        const int stride[3]={SY,SX,1};
        const float* p=g.f+PAD+i*stride[axis]+j*stride[b]+k*stride[c];
        const float s=x-j,t=y-k;
        float e=p[0]+t*(p[stride[c]]-p[0]);
        const float* q=p+stride[b];
        float f=q[0]+t*(q[stride[c]]-q[0]);
        return e+s*(f-e);
    }

    std::vector<RidgePoint> ridges(int axis,float th) {
        const int b=(axis+1)%3,c=(axis+2)%3;
        const int str[3]={SY,SX,1};
        std::vector<RidgePoint> res;
        for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x++) {
            const float* v=guide.f+idx(z,j,x);
            if(*v<th) continue;
            int d[3]={z,j,x}; bool ok=true;
            for(int s=-1;s<=1;s++) for(int t=-1;t<=1;t++) {
                if(d[b]+s>=0 && d[b]+s<S && d[c]+t>=0 && d[c]+t<S &&
                      v[s*str[b]+t*str[c]]>*v) ok=false;
            }
            if(!ok) continue;
            RidgePoint pp{{float(z),float(j),float(x)}};
            for(int k:{b,c}) if(d[k]>0 && d[k]<S-1) {
                const float left=v[-str[k]], right=v[str[k]];
                const float delta=(.5f*(right-left))/std::max(1.e-4f,2*(*v)-left-right);
                pp.p[k]+=clamp(delta,-.49f,.49f);
            }
            res.push_back(pp);
        }
        return res;
    }

    struct LineTry {float value=0;Vec3 a={},b={};float height=0;};
    LineTry sample_line(const Vec3& a,const Vec3& b,int axis,float th) {
        LineTry ans;
        const int u=(axis+1)%3,v=(axis+2)%3;
        const float d=b[axis]-a[axis];
        if(std::abs(d)<5.f) return ans;
        const float dx=(b[u]-a[u])/d, dy=(b[v]-a[v])/d;
        if(std::abs(dx)>1.065f || std::abs(dy)>1.065f) return ans;
        const float l=std::sqrt(1.+sq(dx)+sq(dy));
        const float x0=a[u]-dx*a[axis],y0=a[v]-dy*a[axis];
        const float threshold=th;
        float x=x0,y=y0,score=0,pv=0,sp=0,mx=0;
        int i0=0,last=-1;
        auto record=[&](){
            const int n=last-i0+1;
            if(n<5 || (n-1)*l<6.2f || mx<threshold*2.05f) return;
            // Favor coherent length but not two bright peaks joined by a gap.
            float s=(score-.52f*pv)*std::sqrt(l);
            if(s>ans.value) {
                ans.value=s;ans.height=sp/std::max(1,n);
                const float offset=std::min(1.20f/l,(n-1)*.21f);
                const float aa=i0+offset, bb=last-offset;
                ans.a[axis]=aa;ans.b[axis]=bb;
                ans.a[u]=x0+aa*dx;ans.b[u]=x0+bb*dx;
                ans.a[v]=y0+aa*dy;ans.b[v]=y0+bb*dy;
            }
        };
        for(int i=0;i<S;i++,x+=dx,y+=dy) {
            const float p=interp(guide,axis,i,x,y);
            if(p>threshold) {
                if(last<0) {i0=i;score=sp=pv=mx=0;}
                score+=p-threshold;sp+=p;mx=std::max(mx,p);last=i;
            } else if(last>=0) {
                if(i-last>1 || p<.52f*threshold) {record();last=-1;}
                else pv+=threshold-p;
            }
        }
        if(last>=0) record();
        return ans;
    }

    bool find_tube(Object& o,int rounds=32) {
        if(!room(.004)) return false;
        residual();
        const float l1=(noise.sigma*.19f+.0014f);
        image_fit(corrected,guide,rounds,l1,0.f,!optics.separable);
        float vmax=0;
        for(float v:guide.f) vmax=std::max(vmax,v);
        const float rth=std::max(.054f,.132f*vmax),lth=std::max(.065f,.14f*vmax);
        LineTry best;
        for(int j=0;j<3 && room();j++) {
            const auto pts=ridges(j,rth);
            const int n=pts.size();
            if(n<4) continue;
            if(n<=102) {
                for(int a=0;a<n && room();a++)
                    for(int b=0;b<a;b++) {
                        auto c=sample_line(pts[a].p,pts[b].p,j,lth);
                        if(c.value>best.value) best=c;
                    }
            } else {
                for(int k=0;k<5750;k++) {
                    if(k%128==0 && !room()) break;
                    const int a=rng.next()%n,b=rng.next()%n;
                    auto c=sample_line(pts[a].p,pts[b].p,j,lth);
                    if(c.value>best.value) best=c;
                }
            }
        }
        if(debug>=2) std::fprintf(stderr,"line candidate %.4f peak=%.4f\n",best.value,vmax);
        if(best.value<std::max(.54f,20.f*noise.sigma)) return false;
        o.kind=Tube;
        const float amp=clamp(1.58f*best.height,.57f,1.1f);
        o.p={best.a[0],best.a[1],best.a[2],best.b[0],best.b[1],best.b[2],
             std::log(1.12f),std::log(amp)};
        constrain(o);
        return true;
    }

    bool find_spot(Object& o,float min_z,bool multiscale) {
        if(!room(.003)) return false;
        loss(model,nullptr,full,true);
        float best=min_z;
        float s=.91f,xyz[3]={};
        const int scales=multiscale?3:1;
        for(int j=0;j<scales && room();j++) {
            const float size=j==0?.90f:(j==1?.65f:1.14f);
            const Axis g=gauss(size,j==2?5:4);
            Kernel h,h2;
            for(int k=0;k<3;k++) {
                h[k]=combine(optics.adjoint[k],g);
                h2[k]=h[k].square();
            }
            optics.conv.apply(gradient,back,h);
            optics.conv.apply(curvature,tmp2,h2);
            for(int z=0;z<S;z++) for(int y=0;y<S;y++) for(int x=0;x<S;x++) {
                const int i=idx(z,y,x);
                const float t=back.f[i],v=tmp2.f[i];
                if(t>0 && t*t>best*best*v) {
                    const Vec3 k={float(z),float(y),float(x)};
                    bool far=true;
                    for(const auto& a:objects) if(a.kind==Spot) {
                        float d=0;
                        for(int i=0;i<3;i++) d+=sq(a.p[i]-k[i]);
                        if(d<2.04f) {far=false;break;}
                    }
                    if(!far) continue;
                    best=t/std::sqrt(v);s=size;xyz[0]=z;xyz[1]=y;xyz[2]=x;
                }
            }
        }
        if(debug>=2) std::fprintf(stderr,"spot candidate significance %.3f\n",best);
        if(best<=min_z) return false;
        o.kind=Spot;
        o.p={xyz[0],xyz[1],xyz[2],std::log(s),std::log(amp_center(o)),0.f,0.f,0.f};
        return true;
    }

    static float length(const Object& o) {
        return std::sqrt(sq(o.p[0]-o.p[3])+sq(o.p[1]-o.p[4])+sq(o.p[2]-o.p[5]));
    }

    int which_count(Kind kind) const {
        int n=0;for(const Object& o:objects) n+=(o.kind==kind);
        return n;
    }

    double shape_objective(const std::vector<Object>& qs,bool exact) {
        draw_group(qs,model,cur_group);
        double result=objective(cur_group,nullptr,full,false,exact);
        for(const Object& o:qs) result+=prior(o,nullptr);
        return result;
    }

    struct SplitAlternative {
        size_t first,second;
        bool selected;
        float probability;
        Object original,a,b;
    };
    std::vector<SplitAlternative> alternatives;

    bool split_spot(size_t i,float required) {
        if(!photons || !room(.026)) return false;
        const Object original=objects[i];
        if(original.kind!=Spot || original.size()<.79f ||
           std::exp(original.p[4]+3*original.p[3])<.43f) return false;
        remove(original);
        const std::vector<Object> old{original};
        const double original_cost=shape_objective(old,true);
        const double proxy=shape_objective(old,false);
        struct Seed { std::vector<Object> q; double loss; Vec3 dir; };
        std::vector<Seed> proposals;
        for(int a=-1;a<=1;a++) for(int b=-1;b<=1;b++) for(int c=-1;c<=1;c++) {
            const int nz=a ? a : (b ? b : c);
            if(nz<=0) continue;
            const float inv=1.f/std::sqrt(float(a*a+b*b+c*c));
            const Vec3 axis{inv*a,inv*b,inv*c};
            for(float separation:{2.35f,3.6f,4.85f}) {
                if(!room(.012)) continue;
                std::vector<Object> pair{original,original};
                const float width=clamp(original.size()*.786f,.56f,1.03f);
                for(int k=0;k<2;k++) {
                    for(int j=0;j<3;j++) pair[k].p[j]+=((k==0?-.5f:.5f)*separation)*axis[j];
                    pair[k].p[3]=std::log(width);
                    pair[k].p[4]=clamp(original.p[4]+3*(original.p[3]-pair[k].p[3])-std::log(2.f),
                                       std::log(.37f),std::log(1.28f));
                    constrain(pair[k]);
                }
                const double c1=shape_objective(pair,false)-proxy;
                proposals.push_back({pair,c1,axis});
            }
        }
        std::sort(proposals.begin(),proposals.end(),[](const Seed& a,const Seed& b){return a.loss<b.loss;});
        std::vector<Object> best;
        double gain=0.;
        std::vector<Vec3> used;
        for(auto& q:proposals) {
            const int num=used.size();
            if(num>=4 || !room(.017) || (q.loss>20. && num>0)) break;
            if(q.loss>32. && original.size()<1.22f) break;
            bool other=true;
            for(const auto& p:used) if(std::abs(p[0]*q.dir[0]+p[1]*q.dir[1]+p[2]*q.dir[2])>.994f) other=false;
            if(!other) continue;
            used.push_back(q.dir);
            const double fitted=joint(q.q,model,15,true,0);
            const Object& a=q.q[0],&b=q.q[1];
            const double imp=original_cost-(fitted+prior(a,nullptr)+prior(b,nullptr));
            const float dis=std::sqrt(sq(a.p[0]-b.p[0])+sq(a.p[1]-b.p[1])+sq(a.p[2]-b.p[2]));
            if(imp>gain && dis>1.5f && a.p[4]>-1.35f && b.p[4]>-1.35f) {
                best=q.q;gain=imp;
            }
            if(gain>required+17 && num) break;
        }
        if(debug>=2 && gain>.1) std::fprintf(stderr,"split proposal on %zu posterior gain %.3f\n",i,gain);
        const float probability=1.f/(1.f+std::exp(float((required-gain)/1.35f)));
        if(gain>=required && objects.size()<MAX_OBJS) {
            if(probability<.978f)
                alternatives.push_back({i,objects.size(),true,probability,original,best[0],best[1]});
            objects[i]=best[0];
            shapes.draw(best[0],*pred,false,true);
            axpy(model,pred->v[0],1,pred->box);
            append(best[1],gain);
            if(debug) describe(best[0],gain);
            return true;
        }
        if(!best.empty() && gain>3.60 && probability>.09f && noise.sigma>.015f && required<8.f)
            alternatives.push_back({i,0,false,probability,original,best[0],best[1]});
        shapes.draw(original,*pred,false,true);
        axpy(model,pred->v[0],1,pred->box);
        return false;
    }

    bool review_tube(size_t i,int tubes) {
        const Object orig=objects[i];
        if(orig.kind!=Tube || !room(.035)) return false;
        const float len=length(orig), a=std::exp(orig.p[7]);
        if(len>13.3f || (tubes>1 && (len>10.f || a>.67f)) || (len>9 && a>.82f))
            return false;
        remove(orig);
        const double former=shape_objective({orig},true);
        double best_gain=-1.e8;
        std::vector<Object> best;
        const float ww=orig.size();
        const float volume=a*ww*ww*(.399f*len+ww);
        for(int count:{2,3}) {
            if(objects.size()+(count-1)>MAX_OBJS) continue;
            for(float spread:{.85f,1.12f}) {
                if(!room(.018)) continue;
                std::vector<Object> qs;
                for(int j=0;j<count;j++) {
                    const float t=.5f+spread*(j/float(count-1)-.5f);
                    Object o;o.kind=Spot;
                    for(int k=0;k<3;k++) o.p[k]=orig.p[k]+t*(orig.p[k+3]-orig.p[k]);
                    const float amp=clamp(a*1.48f,.60f,.95f);
                    o.p[3]=std::log(clamp(std::cbrt(volume/(amp*count)),.68f,1.23f));
                    o.p[4]=std::log(amp);constrain(o);qs.push_back(o);
                }
                const double c=joint(qs,model,20,true,0);
                double v=former-c;
                for(const auto& q:qs) v-=prior(q,nullptr);
                v-=float(count-2)*6.3f;
                if(v>best_gain) {best_gain=v;best=std::move(qs);}
            }
        }
        if(debug>=2) std::fprintf(stderr,"point-vs-tube posterior gain %.3f\n",best_gain);
        const double threshold=tubes==1 && which_count(Spot)>=3 && a<.665f?-.25:7.0;
        if(best_gain>threshold) {
            objects[i]=best[0];
            shapes.draw(best[0],*pred,false,true);
            axpy(model,pred->v[0],1,pred->box);
            for(size_t j=1;j<best.size();j++) append(best[j],best_gain);
            if(debug) describe(best[0],best_gain);
            return true;
        }
        shapes.draw(orig,*pred,false,true);
        axpy(model,pred->v[0],1,pred->box);
        return false;
    }

    struct SpotSeed { float merit; Object guess; };
    std::vector<SpotSeed> likelihood_seeds() {
        constexpr float nodes[4]={.075f,.26f,.58f,1.f};
        constexpr float transform[4][4]={
            {23.266724f,-3.8191128f,.4953543f,-.0393405f},
            {-152.869166f,61.3252936f,-9.0052878f,.743678f},
            {283.891063f,-145.301877f,33.9127201f,-3.1827194f},
            {-154.288621f,87.7956959f,-25.4027866f,3.4783819f}
        };
        Grid *coef[4]={&gradient,&curvature,&back,&ax};
        std::vector<SpotSeed> seeds;
        for(float width:{.59f,.76f,.94f,1.12f}) {
            if(!room(.012)) break;
            const Axis g=gauss(width,4);
            Kernel ker;
            float height=amp_center(Object{});
            for(int j=0;j<3;j++) {
                ker[j]=combine(optics.adjoint[j],g);
                const float a=ker[j].k[ker[j].r];
                height*=a/g.k[g.r];
                for(int k=0;k<=2*ker[j].r;k++) ker[j].k[k]/=a;
            }
            for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x+=8) {
                const int i=idx(z,j,x);
                const V b=ld(model.f+i), zero=photons->voxel(b,i);
                V values[4];
                for(int k=0;k<4;k++) values[k]=photons->voxel(b+splat(height*nodes[k]),i)-zero;
                for(int k=0;k<4;k++) {
                    V value=values[0]*splat(transform[k][0]);
                    for(int h=1;h<4;h++) value=fma(values[h],splat(transform[k][h]),value);
                    st(coef[k]->f+i,value);
                }
            }
            Kernel h=ker;
            for(int k=0;k<4;k++) {
                if(k==0) optics.conv.apply(*coef[k],tmp3,h);
                else {optics.conv.apply(*coef[k],tmp2,h);axpy(tmp3,tmp2,1.f);}
                for(int j=0;j<3;j++) for(int m=0;m<2*ker[j].r+1;m++) h[j].k[m]*=ker[j].k[m];
            }
            for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x++) {
                const float s=tmp3.at(z,j,x);
                if(s<3.f) continue;
                bool is_max=true;
                for(int u=std::max(0,z-1);u<=std::min(S-1,z+1) && is_max;u++)
                    for(int v=std::max(0,j-1);v<=std::min(S-1,j+1) && is_max;v++)
                        for(int w=std::max(0,x-1);w<=std::min(S-1,x+1);w++)
                            if(tmp3.at(u,v,w)>s) {is_max=false;break;}
                if(!is_max) continue;
                bool distinct=true;
                for(const auto& o:objects) if(o.kind==Spot &&
                    sq(o.p[0]-z)+sq(o.p[1]-j)+sq(o.p[2]-x)<2.1f) {distinct=false;break;}
                if(!distinct) continue;
                Object o;o.kind=Spot;
                o.p={float(z),float(j),float(x),std::log(width),std::log(amp_center(o)),0,0,0};
                const float bonus= -.30f*(sq((width-.91f)/.30f))
                    -.30f*(sq(std::max(0,2-z))+sq(std::max(0,z-29))+sq(std::max(0,2-j))+
                           sq(std::max(0,j-29))+sq(std::max(0,2-x))+sq(std::max(0,x-29)));
                seeds.push_back({s+bonus,o});
            }
        }
        std::sort(seeds.begin(),seeds.end(),[](const SpotSeed& a,const SpotSeed& b){return a.merit>b.merit;});
        return seeds;
    }

    std::vector<std::pair<Object,float>> uncertain;

    bool likelihood_spots(bool missing) {
        if(!room(.035)) return false;
        const auto seeds=likelihood_seeds();
        const double old=objective(model,nullptr,full,false,true);
        std::vector<std::pair<Object,double>> candidates;
        for(const auto& seed:seeds) {
            if(seed.merit<(missing?2.8f:5.3f) || candidates.size()>=8 || !room(.015)) break;
            bool same=false;
            for(const auto& p:candidates) {
                float d=0;
                for(int j=0;j<3;j++) d+=sq(p.first.p[j]-seed.guess.p[j]);
                if(d<2.60f) {same=true;break;}
            }
            if(same) continue;
            std::vector<Object> one{seed.guess};
            const double l=old-joint(one,model,12,true,0)-prior(one[0],nullptr);
            const auto& o=one[0];
            if(!std::isfinite(l)) continue;
            bool separate=true;
            for(const auto& p:objects) if(p.kind==Spot &&
                sq(p.p[0]-o.p[0])+sq(p.p[1]-o.p[1])+sq(p.p[2]-o.p[2])<2.00f) separate=false;
            for(const auto& p:candidates)
                if(sq(p.first.p[0]-o.p[0])+sq(p.first.p[1]-o.p[1])+sq(p.first.p[2]-o.p[2])<2.10f)
                    separate=false;
            if(separate && o.p[4]>-1.95f) candidates.push_back({o,l});
        }
        if(candidates.empty()) return false;
        std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.second>b.second;});
        const auto [o,best]=candidates[0];
        const double threshold=missing?9.3:13.7;
        if(debug>=2) std::fprintf(stderr,"residual likelihood gain %.3f\n",best);
        if(best>threshold && objects.size()<MAX_OBJS) {
            append(o,best);
            return true;
        }
        float denom=1;
        for(const auto& s:candidates) denom+=std::exp(clamp(float((s.second-threshold+1.1)/1.30),-12,3));
        for(const auto& s:candidates) {
            const float prob=std::exp(clamp(float((s.second-threshold+1.1)/1.30),-12,3))/denom;
            if(prob>.065f) uncertain.push_back({s.first,prob});
        }
        return false;
    }

    static void gaussian_on_grid(Grid& out,const float* center,float s,float amplitude,const double (&v)[3][3]) {
        const double ss=s*s, a=ss+v[0][0], b=v[0][1], c=v[0][2],
            d=ss+v[1][1], e=v[1][2], f=ss+v[2][2];
        const double i0=d*f-e*e, i1=c*e-b*f, i2=b*e-c*d,
            i3=a*f-c*c, i4=b*c-a*e, i5=a*d-b*b;
        const double det=a*i0+b*i1+c*i2;
        if(!(det>0) || !(amplitude>0)) return;
        const float u=-.5f/det, h=amplitude*(s*ss)/std::sqrt(det);
        const float rz=4.8f*std::sqrt(a),ry=4.8f*std::sqrt(d);
        const Box box{std::max(0,int(center[0]-rz)),std::min(S,int(center[0]+rz+1)),
                      std::max(0,int(center[1]-ry)),std::min(S,int(center[1]+ry+1))};
        const V vx=_mm256_setr_ps(0,1,2,3,4,5,6,7)-splat(center[2]);
        const V vv=splat(u*i5);
        for(int z=box.z0;z<box.z1;z++) for(int y=box.y0;y<box.y1;y++) {
            const float dz=z-center[0],dy=y-center[1];
            const V q=splat(u*(i0*dz*dz+2*i1*dz*dy+i3*dy*dy));
            const V p=splat(2*u*(i2*dz+i4*dy));
            for(int x=0;x<S;x+=8) {
                const V dx=vx+splat(float(x));
                const V exponent=_mm256_min_ps(splat(0),fma(vv,dx*dx,fma(p,dx,q)));
                float* dst=out.row(z,y)+x;
                st(dst,fma(splat(h),gaussian_exp(exponent),ld(dst)));
            }
        }
    }

    static void posterior_spot(Grid& out,const Object& o,double (&cov)[5][5],const float* center,float prob) {
        const double widthvar=std::max(1.e-7,cov[3][3]);
        const float sdev=std::sqrt(widthvar);
        constexpr int Q=11;
        const float a=clamp((std::log(.536f)-center[3])/sdev,-3.5f,3.5f);
        const float b=clamp((std::log(1.331f)-center[3])/sdev,-3.5f,3.5f);
        if(!(b>a)) return;
        float w[Q],u[Q],t=0;
        for(int j=0;j<Q;j++) {
            u[j]=a+(b-a)*(j+.5f)/Q;
            t+=(w[j]=std::exp(-.5f*u[j]*u[j]));
        }
        const double V_amp=std::max(0.,cov[4][4]-cov[3][4]*cov[3][4]/widthvar);
        for(int j=0;j<Q;j++) {
            const float ls=center[3]+sdev*u[j],s=std::exp(ls);
            const double wa=u[j]/sdev;
            double sp[3][3];
            float loc[3];
            for(int k=0;k<3;k++) {
                for(int l=0;l<3;l++) sp[k][l]=cov[k][l]-cov[k][3]*cov[l][3]/widthvar;
                loc[k]=center[k]+wa*cov[k][3]+cov[k][4]-cov[k][3]*cov[4][3]/widthvar;
            }
            const float amp=std::exp(clamp(float(o.p[4]+wa*cov[4][3]+.5*V_amp),-5.4f,.68f));
            gaussian_on_grid(out,loc,s,prob*amp*w[j]/t,sp);
        }
    }

    bool render_marginal(Grid& out,const std::vector<float>& prob) {
        if(!photons || !which_count(Spot) ||
           !room(.022+objects.size()*(optics.separable?.001:.006))) return false;
        const Grid no_background;
        const auto [offset,n]=group_offsets(objects);
        Normal h(n);
        group_normal(objects,no_background,offset,h,true);
        group_prior(objects,offset,&h,.067f);
        Normal::Matrix L;
        if(!h.factor(L,.00011)) return false;
        for(size_t i=0;i<objects.size();i++) {
            const Object& o=objects[i];
            if(o.kind!=Spot) {
                shapes.draw(o,*pred,false,false);
                axpy(out,pred->v[0],prob[i],pred->box);
                continue;
            }
            double C[5][5], lim[5];
            for(int j=0;j<5;j++) {
                double e[MAX_PARAMS]={}, r[MAX_PARAMS]={};
                e[offset[i]+j]=1;
                h.backsolve(L,e,r);
                for(int k=0;k<5;k++) C[j][k]=r[offset[i]+k];
            }
            const double caps[5]={.95,.95,.95,.285,.297};
            for(int j=0;j<5;j++) {
                const double sigma=std::sqrt(std::max(1.e-8,C[j][j]));
                lim[j]=.954*std::min(1.,caps[j]/sigma);
            }
            for(int j=0;j<5;j++) for(int k=0;k<5;k++) C[j][k]*=lim[j]*lim[k];
            posterior_spot(out,o,C,o.p.data(),prob[i]);
        }
        return true;
    }

    void compose(Grid& out) {
        std::memset(out.f,0,sizeof(out.f));
        std::vector<float> weights(objects.size(),1.f);
        const SplitAlternative* weak=nullptr;
        for(const auto& a:alternatives) {
            if(a.selected) {
                weights[a.first]*=a.probability;
                weights[a.second]*=a.probability;
            } else if(!weak || a.probability>weak->probability) weak=&a;
        }
        if(weak) weights[weak->first]*=1.f-weak->probability;
        if(!render_marginal(out,weights)) {
            for(size_t i=0;i<objects.size();i++) {
                const auto& o=objects[i];
                shapes.draw(o,*pred,false,false);
                axpy(out,pred->v[0],weights[i],pred->box);
            }
        }
        for(const auto& a:alternatives) {
            const double v[3][3]={{.13,0,0},{0,.13,0},{0,0,.13}};
            if(a.selected) {
                gaussian_on_grid(out,a.original.p.data(),a.original.size(),
                                 std::exp(a.original.p[4])*(1.f-a.probability),v);
            } else if(&a==weak) {
                for(const auto& o:{a.a,a.b})
                    gaussian_on_grid(out,o.p.data(),o.size(),std::exp(o.p[4])*a.probability,v);
            }
        }
        for(const auto& s:uncertain) {
            const double v[3][3]={{.19,0,0},{0,.19,0},{0,0,.19}};
            gaussian_on_grid(out,s.first.p.data(),s.first.size(),std::exp(s.first.p[4])*s.second,v);
        }
    }

    void recover_residual(Grid& out) {
        if(!photons || !room(.038) || which_count(Sheet)>1) return;
        const Grid blank;
        auto candidate=std::make_unique<Grid>();
        residual();
        const float reg=.0016f+.19f*noise.sigma;
        image_fit(corrected,*candidate,46,reg,.0030f);
        if(*std::max_element(candidate->f,candidate->f+SZ)<.22f) return;
        optics.apply(*candidate,ax);
        const double current=objective(model,nullptr,full,false,true);
        const double resolved=objective(blank,nullptr,full,false,true)-current;
        const double gain=current-objective(model,&ax,full,false,true);
        if(resolved>0 && gain>std::max(62.0,.105*resolved)) {
            const float f=clamp(.45f+2.4f*(gain/resolved-.105f),.45f,.92f);
            axpy(out,*candidate,f);
            if(debug) std::fprintf(stderr,"nonparametric residual %.3f (weight %.3f)\n",gain,f);
        }
    }

    void run(Grid& out) {
        const float reg=noise.sigma*.18f+.0016f;
        if(peak>.008f) {
            for(int j=0;j<3 && room(.008);j++) {
                Object q;
                if(!find_sheet(q)) break;
                const auto gain=fit(q,18);
                if(gain<125) break;
                append(q,gain);
            }
            const int sheets=which_count(Sheet);
            if(sheets>1) refine_joint(14,false);
            for(int j=0;j<5 && room(.008);j++) {
                Object q;
                if(!find_tube(q)) break;
                const auto gain=fit(q,16);
                if(gain<(sheets?170.:54.) || length(q)<6.45f || length(q)<4*q.size() || q.p[7]<std::log(.26f)) break;
                append(q,gain);
                if(j>=1) refine_joint(j==1?10:15,false);
            }
            for(int j=0;j<11 && objects.size()<MAX_OBJS && room(.008);j++) {
                Object q;
                const float z=sheets?(j>=3?5.40f:3.95f):(which_count(Tube)?4.15f:3.95f)+(j>=7?.55f:0.f);
                if(!find_spot(q,z,true)) break;
                const auto gain=fit(q,14);
                if(gain<.5*z*z || q.p[4]<std::log(.18f)) break;
                append(q,gain);
            }
            refine_joint(18,false);
            if(!objects.empty() && room(.04)) {
                photons=std::make_unique<Photons>(noise,y,pilot);
                refine_joint(15,true);
                const int tubes=which_count(Tube);
                bool revised=false;
                const size_t checked=objects.size();
                for(size_t j=0;!sheets && j<checked && room(.04);j++)
                    revised|=review_tube(j,tubes);
                if(revised) refine_joint(10,true);
                const size_t count=objects.size();
                revised=false;
                for(size_t j=0; !sheets && j<count && room(.027);j++) {
                    const int points=which_count(Spot);
                    const float l=points<(which_count(Tube)?3:7)?6.65f:11.7f;
                    revised|=split_spot(j,l);
                }
                if(revised) refine_joint(12,true);
                const int target=sheets?3:(which_count(Tube)?3:7);
                if(which_count(Spot) || (sheets && noise.sigma>.014f)) {
                    for(int j=0;j<3 && room(.045);j++) {
                        const int n=which_count(Spot);
                        if(n>target || (!sheets && which_count(Tube)>2 && n==0)) break;
                        if(!likelihood_spots(n<target)) break;
                        refine_joint(8,true);
                    }
                }
            }
        }
        if(objects.empty()) {
            residual();
            image_fit(corrected,out,100,reg,.0024f);
        } else {
            compose(out);
            recover_residual(out);
        }
        if(debug) {
            std::fprintf(stderr,"%zu final components\n",objects.size());
            for(const auto& o:objects) describe(o,0);
        }
    }
};

} // namespace

extern "C" int deconvolve(const float* y,const float* psf,float* out,double deadline,int debug) {
    try {
        float max=*std::max_element(y,y+NP);
        if(!(max>0.f)) {std::fill(out,out+NP,0.f);return 0;}
        const double scale=max>1.8f ? .85/double(max) : (max<.07f?.18/double(max):1.);
        auto engine=std::make_unique<Solver>(y,psf,scale,deadline,debug);
        if(engine->optics.fft && (!engine->optics.fft->r || !engine->optics.fft->f)) return 1;
        auto estimate=std::make_unique<Grid>();
        engine->run(*estimate);
        const double inv=1./scale;
        constexpr double largest_float=0x1.fffffep127;
        for(int z=0;z<S;z++) for(int j=0;j<S;j++) for(int x=0;x<S;x++)
            out[(z*S+j)*S+x]=float(std::min(largest_float,std::max(0.,estimate->at(z,j,x)*inv)));
        return 0;
    } catch(...) {
        return 1;
    }
}
