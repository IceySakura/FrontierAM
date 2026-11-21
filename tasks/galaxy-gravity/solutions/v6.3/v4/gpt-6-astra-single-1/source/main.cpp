#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <vector>
#include <utility>
#include <complex>
#include <sys/stat.h>
#include <immintrin.h>

using V = __m256d;
static constexpr double EPS2 = 0.0001;
static inline V splat(double x) { return _mm256_set1_pd(x); }
static inline double horizontal_sum(V v) {
    __m128d s=_mm_add_pd(_mm256_castpd256_pd128(v),_mm256_extractf128_pd(v,1));
    return _mm_cvtsd_f64(_mm_add_sd(s,_mm_unpackhi_pd(s,s)));
}
static inline double cpu_time() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double setting(const char* name, double fallback) {
    const char* p = std::getenv(name);
    return p ? std::atof(p) : fallback;
}
#include "derivatives.h"

struct Particle { double m, x, y, z; uint32_t id; };
struct Node {
    double x, y, z, r;
    int begin, end, left, right;
    bool leaf() const { return left < 0; }
};
struct Pair { int a, b; };
struct Index { int x,y,z,n; };
struct ShiftTerm { uint16_t b, d; };
struct HarmTerm { uint16_t b; double re,im; };

struct Basis {
    int p, nm, nl;
    int idx[13][13][13];
    std::vector<Index> v;
    std::vector<uint16_t> gx,gy,gz;
    std::vector<std::vector<uint16_t>> sums;
    std::vector<std::vector<ShiftTerm>> shifts;
    std::vector<std::vector<HarmTerm>> harmonics;
    std::vector<std::pair<int,int>> harm_range;
    explicit Basis(int order) : p(order) {
        std::memset(idx, -1, sizeof(idx));
        for(int n=0;n<=p;n++)
            for(int x=0;x<=n;x++)
                for(int y=0;y<=n-x;y++) {
                    idx[x][y][n-x-y] = int(v.size());
                    v.push_back({x,y,n-x-y,n});
                }
        nl = v.size();
        nm = p*(p+1)*(p+2)/6;
        for(int a=0;a<nm;a++) {
            auto u=v[a];
            gx.push_back(idx[u.x+1][u.y][u.z]);
            gy.push_back(idx[u.x][u.y+1][u.z]);
            gz.push_back(idx[u.x][u.y][u.z+1]);
        }
        sums.resize(nl);
        shifts.resize(nl);
        for(int a=0;a<nl;a++) {
            const auto u=v[a];
            for(int b=0;b<nl && u.n+v[b].n<=p;b++)
                sums[a].push_back(idx[u.x+v[b].x][u.y+v[b].y][u.z+v[b].z]);
            for(int b=0;b<=a;b++) {
                const auto w=v[b];
                if(w.x<=u.x && w.y<=u.y && w.z<=u.z)
                    shifts[a].push_back({uint16_t(b),uint16_t(idx[u.x-w.x][u.y-w.y][u.z-w.z])});
            }
        }
        using C=std::complex<double>;
        std::vector<std::vector<C>> h(p*(p+1)/2,std::vector<C>(nl));
        double fact[13];fact[0]=1;
        for(int i=1;i<=12;i++)fact[i]=fact[i-1]*i;
        for(int l=0;l<p;l++) {
            int first=harmonics.size();
            for(int m=0;m<=l;m++) {
                auto& out=h[l*(l+1)/2+m];
                if(l==0)out[0]=1;
                else if(m==l) {
                    const auto& prev=h[(l-1)*l/2+m-1];
                    double c=-std::sqrt(double(2*m-1)/(2*m));
                    for(int a=0;a<nl;a++)if(v[a].n==l-1) {
                        auto u=v[a];
                        out[idx[u.x+1][u.y][u.z]]+=c*prev[a];
                        out[idx[u.x][u.y+1][u.z]]+=C(0,c)*prev[a];
                    }
                } else {
                    double den=double((l-m)*(l+m));
                    double c=(2*l-1)/std::sqrt(den);
                    const auto& prev=h[(l-1)*l/2+m];
                    for(int a=0;a<nl;a++)if(v[a].n==l-1) {
                        auto u=v[a];
                        out[idx[u.x][u.y][u.z+1]]+=c*prev[a];
                    }
                    if(m<=l-2) {
                        double d=-std::sqrt(double((l-1-m)*(l-1+m))/den);
                        const auto& pp=h[(l-2)*(l-1)/2+m];
                        for(int a=0;a<nl;a++)if(v[a].n==l-2) {
                            auto u=v[a];
                            out[idx[u.x+2][u.y][u.z]]+=d*pp[a];
                            out[idx[u.x][u.y+2][u.z]]+=d*pp[a];
                            out[idx[u.x][u.y][u.z+2]]+=d*pp[a];
                        }
                    }
                }
                std::vector<HarmTerm> terms;
                for(int a=0;a<nl;a++)if(std::abs(out[a])>1e-30) {
                    auto u=v[a];
                    double c=fact[u.x]*fact[u.y]*fact[u.z]*(m?std::sqrt(2.0):1.0);
                    terms.push_back({uint16_t(a),c*out[a].real(),c*out[a].imag()});
                }
                harmonics.push_back(std::move(terms));
            }
            harm_range.push_back({first,int(harmonics.size())});
        }
    }
    void powers(double x,double y,double z,double* f) const {
        double xp[13],yp[13],zp[13];
        xp[0]=yp[0]=zp[0]=1.0;
        for(int i=1;i<=p;i++) {
            xp[i]=xp[i-1]*x/i;
            yp[i]=yp[i-1]*y/i;
            zp[i]=zp[i-1]*z/i;
        }
        for(int a=0;a<nl;a++)
            f[a]=xp[v[a].x]*yp[v[a].y]*zp[v[a].z];
    }
    void powers4(V x,V y,V z,V* f) const {
        V xp[13],yp[13],zp[13];
        xp[0]=yp[0]=zp[0]=splat(1.0);
        for(int i=1;i<=p;i++) {
            V inv=splat(1.0/i);
            xp[i]=xp[i-1]*x*inv;
            yp[i]=yp[i-1]*y*inv;
            zp[i]=zp[i-1]*z*inv;
        }
        for(int a=0;a<nm;a++)
            f[a]=xp[v[a].x]*yp[v[a].y]*zp[v[a].z];
    }
};

class Solver {
public:
    int n, leafsize;
    double theta;
    double cx=0,cy=0,cz=0,scale=1,leaf_geom=1e30;
    bool rotated=false;
    double rotation[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    std::vector<Particle> part;
    std::vector<Node> nodes;
    std::vector<double> x,y,z,m,ax,ay,az;
    std::vector<Pair> far,near;
    std::vector<Pair> point_far;
    std::vector<double> moments,locals;
    std::vector<double> error_moments,acc_est;
    std::vector<double> harmonic_errors;
    std::vector<double> error_bound;
    double pair_error_a=0,pair_error_b=0,point_rate=5e-7;
    double tolerance=0.0;
    bool use_harmonic=false;
    bool fast_direct=false;
    bool mutual_direct=false;
    int mode=0;
    uint64_t far_count=0,point_count=0,near_count=0,sample_direct_work=0;
    double far_seconds=0,point_seconds=0,near_seconds=0;
    double deadline=1e100,coarse_deadline=1e100;
    bool emergency=false;
    Basis basis;
    uint64_t direct_work=0;
    Solver(int order,int leaf,double angle) : leafsize(leaf),theta(angle),basis(order) {}

    int make_node(int begin,int end) {
        double lo[3]={part[begin].x,part[begin].y,part[begin].z};
        double hi[3]={lo[0],lo[1],lo[2]};
        for(int i=begin+1;i<end;i++) {
            const double a[3]={part[i].x,part[i].y,part[i].z};
            for(int j=0;j<3;j++) { lo[j]=std::min(lo[j],a[j]); hi[j]=std::max(hi[j],a[j]); }
        }
        Node t;
        t.x=(lo[0]+hi[0])*0.5;
        t.y=(lo[1]+hi[1])*0.5;
        t.z=(lo[2]+hi[2])*0.5;
        double r2=0;
        for(int i=begin;i<end;i++) {
            double dx=part[i].x-t.x,dy=part[i].y-t.y,dz=part[i].z-t.z;
            r2=std::max(r2,dx*dx+dy*dy+dz*dz);
        }
        t.r=std::sqrt(r2);
        t.begin=begin;t.end=end;t.left=t.right=-1;
        int id=nodes.size();
        nodes.push_back(t);
        double dc=std::sqrt((t.x-cx)*(t.x-cx)+(t.y-cy)*(t.y-cy)+(t.z-cz)*(t.z-cz));
        if((end-begin>leafsize || (end-begin>1 && t.r>leaf_geom*(dc+scale))) && t.r>1e-15) {
            int axis=0;
            for(int j=1;j<3;j++) if(hi[j]-lo[j]>hi[axis]-lo[axis]) axis=j;
            double mid=(hi[axis]+lo[axis])*0.5;
            auto coord=[axis](const Particle& a){ return axis==0 ? a.x : axis==1 ? a.y : a.z; };
            int split=std::partition(part.begin()+begin,part.begin()+end,
                                   [&](const Particle& a){return coord(a)<mid;})-part.begin();
            if(split==begin || split==end) {
                split=(begin+end)/2;
                std::nth_element(part.begin()+begin,part.begin()+split,part.begin()+end,
                                 [&](const Particle& a,const Particle& b){return coord(a)<coord(b);});
            }
            int left=make_node(begin,split);
            int right=make_node(split,end);
            nodes[id].left=left;nodes[id].right=right;
        }
        return id;
    }
    void align_flat_distribution() {
        if(!setting("FMM_ROTATE",1))return;
        double mean[3]={},cov[3][3]={},v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
        int stride=std::max(1,n/8192),count=0;
        for(int i=0;i<n;i+=stride) {
            mean[0]+=part[i].x;mean[1]+=part[i].y;mean[2]+=part[i].z;count++;
        }
        for(auto& a:mean)a/=count;
        for(int i=0;i<n;i+=stride) {
            double d[3]={part[i].x-mean[0],part[i].y-mean[1],part[i].z-mean[2]};
            for(int a=0;a<3;a++)for(int b=0;b<3;b++)cov[a][b]+=d[a]*d[b];
        }
        if(!(cov[0][0]+cov[1][1]+cov[2][2]>0))return;
        for(int step=0;step<24;step++) {
            int a=0,b=1;
            for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)
                if(std::abs(cov[i][j])>std::abs(cov[a][b])){a=i;b=j;}
            if(std::abs(cov[a][b])<1e-15*(cov[0][0]+cov[1][1]+cov[2][2]))break;
            double tau=(cov[b][b]-cov[a][a])/(2*cov[a][b]);
            double t=std::copysign(1.0,tau)/(std::abs(tau)+std::sqrt(1+tau*tau));
            double c=1/std::sqrt(1+t*t),s=t*c;
            double cross=cov[a][b];
            cov[a][a]-=t*cross;cov[b][b]+=t*cross;
            cov[a][b]=cov[b][a]=0;
            for(int k=0;k<3;k++) {
                if(k!=a && k!=b) {
                    double va=cov[k][a],vb=cov[k][b];
                    cov[k][a]=cov[a][k]=c*va-s*vb;
                    cov[k][b]=cov[b][k]=s*va+c*vb;
                }
                double va=v[k][a],vb=v[k][b];
                v[k][a]=c*va-s*vb;v[k][b]=s*va+c*vb;
            }
        }
        int axes[3]={0,1,2};
        std::sort(axes,axes+3,[&](int a,int b){return cov[a][a]>cov[b][b];});
        if(cov[axes[2]][axes[2]]>=0.12*cov[axes[1]][axes[1]])return;
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)rotation[i][j]=v[j][axes[i]];
        for(auto& pt:part) {
            double d[3]={pt.x-mean[0],pt.y-mean[1],pt.z-mean[2]},r[3]={};
            for(int i=0;i<3;i++)for(int j=0;j<3;j++)r[i]+=rotation[i][j]*d[j];
            pt.x=r[0];pt.y=r[1];pt.z=r[2];
        }
        double center[3]={cx-mean[0],cy-mean[1],cz-mean[2]},next[3]={};
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)next[i]+=rotation[i][j]*center[j];
        cx=next[0];cy=next[1];cz=next[2];
        rotated=true;
    }
    bool read(const char* path) {
        FILE* f=std::fopen(path,"rb");
        if(!f)return false;
        struct stat st;
        if(fstat(fileno(f),&st)!=0 || st.st_size%32 || st.st_size<=0) {fclose(f);return false;}
        n=st.st_size/32;
        part.resize(n);
        std::vector<double> buf(4*std::min(n,65536));
        for(int i=0;i<n;) {
            int count=std::min(n-i,65536);
            if(fread(buf.data(),32,count,f)!=size_t(count)) {fclose(f);return false;}
            for(int j=0;j<count;j++)part[i+j]={buf[4*j],buf[4*j+1],buf[4*j+2],buf[4*j+3],uint32_t(i+j)};
            i+=count;
        }
        fclose(f);
        std::vector<double> sample;
        for(int axis=0;axis<3;axis++) {
            sample.clear();
            for(int i=0;i<n;i+=std::max(1,n/4096))
                sample.push_back(axis==0?part[i].x:axis==1?part[i].y:part[i].z);
            std::nth_element(sample.begin(),sample.begin()+sample.size()/2,sample.end());
            (axis==0?cx:axis==1?cy:cz)=sample[sample.size()/2];
        }
        sample.clear();
        for(int i=0;i<n;i+=std::max(1,n/4096))
            sample.push_back(std::sqrt((part[i].x-cx)*(part[i].x-cx)+(part[i].y-cy)*(part[i].y-cy)+(part[i].z-cz)*(part[i].z-cz)));
        std::nth_element(sample.begin(),sample.begin()+sample.size()/2,sample.end());
        scale=sample[sample.size()/2];
        align_flat_distribution();
        nodes.reserve(std::max(64,n/leafsize*4));
        make_node(0,n);
        while(120.0*n+nodes.size()*(8.0*(basis.nm+basis.nl+2*(basis.p+1))+64)>1.70*1073741824.0) {
            nodes.clear();
            leafsize=leafsize*3/2;
            leaf_geom*=1.2;
            make_node(0,n);
        }
        x.resize(n+4);y.resize(n+4);z.resize(n+4);m.resize(n+4);
        ax.resize(n+4);ay.resize(n+4);az.resize(n+4);
        for(int i=0;i<n;i++) {x[i]=part[i].x;y[i]=part[i].y;z[i]=part[i].z;m[i]=part[i].m;}
        return true;
    }
    void upward() {
        const int nm=basis.nm;
        moments.assign(size_t(nodes.size())*nm,0.0);
        locals.assign(size_t(nodes.size())*basis.nl,0.0);
        error_bound.assign(nodes.size(),0.0);
        double f[455];
        // Cartesian moments store sum(m * displacement^index / index!).
        for(int k=int(nodes.size())-1;k>=0;k--) {
            const Node& t=nodes[k];
            double* out=&moments[size_t(k)*nm];
            if(t.end-t.begin==1) {
                out[0]=m[t.begin];
            } else if(t.leaf() || t.end-t.begin < basis.p*basis.p/2) {
                for(int i=t.begin;i<t.end;i++) {
                    basis.powers(x[i]-t.x,y[i]-t.y,z[i]-t.z,f);
                    for(int a=0;a<nm;a++)out[a]+=m[i]*f[a];
                }
            } else {
                for(int child : {t.left,t.right}) {
                    const Node& c=nodes[child];
                    basis.powers(c.x-t.x,c.y-t.y,c.z-t.z,f);
                    const double* in=&moments[size_t(child)*nm];
                    for(int a=0;a<nm;a++) {
                        double sum=0;
                        for(auto term:basis.shifts[a])sum+=in[term.b]*f[term.d];
                        out[a]+=sum;
                    }
                }
            }
        }
    }
    double estimate_acceleration(double tx,double ty,double tz,int self) {
        int stack[256],top=0;stack[top++]=0;
        double sx=0,sy=0,sz=0;
        while(top) {
            int k=stack[--top];
            if(k==self)continue;
            const Node& t=nodes[k];
            const double* mm=&moments[size_t(k)*basis.nm];
            double cx=t.x+mm[3]/mm[0],cy=t.y+mm[2]/mm[0],cz=t.z+mm[1]/mm[0];
            double dx=cx-tx,dy=cy-ty,dz=cz-tz;
            double q=dx*dx+dy*dy+dz*dz+EPS2;
            if(t.leaf() || t.r*t.r<0.15*q) {
                double s=mm[0]/(q*std::sqrt(q));
                sx+=dx*s;sy+=dy*s;sz+=dz*s;
            } else {
                stack[top++]=t.left;stack[top++]=t.right;
            }
        }
        return std::max(1e-30,std::sqrt(sx*sx+sy*sy+sz*sz));
    }
    void make_error_estimates() {
        int p=basis.p;
        double fact[13];fact[0]=1;
        for(int i=1;i<=12;i++)fact[i]=fact[i-1]*i;
        error_moments.resize(size_t(nodes.size())*(p+1));
        if(use_harmonic)harmonic_errors.resize(size_t(nodes.size())*(p+1));
        acc_est.resize(nodes.size());
        for(int k=int(nodes.size())-1;k>=0;k--) {
            double* e=&error_moments[size_t(k)*(p+1)];
            const double* mm=&moments[size_t(k)*basis.nm];
            const Node& t=nodes[k];
            e[0]=mm[0];
            for(int j=2;j<p;j+=2) {
                double sum=0;
                for(int x=0;x<=j;x+=2)
                    for(int y=0;y<=j-x;y+=2) {
                        int z=j-x-y;
                        double c=fact[j/2]/(fact[x/2]*fact[y/2]*fact[z/2]);
                        sum+=c*fact[x]*fact[y]*fact[z]*mm[basis.idx[x][y][z]];
                    }
                e[j]=sum;
            }
            e[p]=e[p-2]*t.r*t.r;
            for(int j=1;j<p;j+=2)e[j]=std::sqrt(e[j-1]*e[j+1]);
            if(use_harmonic) {
                double* h=&harmonic_errors[size_t(k)*(p+1)];
                for(int j=0;j<p;j++) {
                    double sum=0;
                    for(int a=basis.harm_range[j].first;a<basis.harm_range[j].second;a++) {
                        double re=0,im=0;
                        for(auto term:basis.harmonics[a]) {
                            re+=term.re*mm[term.b];im+=term.im*mm[term.b];
                        }
                        sum+=re*re+im*im;
                    }
                    h[j]=std::sqrt(sum);
                }
                h[p]=std::max(0.03*e[p],std::min(e[p],h[p-2]*t.r*t.r));
                for(int j=0;j<=p;j++)h[j]*=fact[p]/(fact[j]*fact[p-j]);
            }
            for(int j=0;j<=p;j++)e[j]*=fact[p]/(fact[j]*fact[p-j]);
            if(t.leaf()) {
                acc_est[k]=estimate_acceleration(t.x+mm[3]/mm[0],t.y+mm[2]/mm[0],t.z+mm[1]/mm[0],k);
            } else {
                acc_est[k]=std::min(acc_est[t.left],acc_est[t.right]);
            }
        }
    }
    bool acceptable(int a,int b,double q) {
        if(!tolerance)return true;
        int p=basis.p;
        const double* ea=&error_moments[size_t(a)*(p+1)];
        const double* eb=&error_moments[size_t(b)*(p+1)];
        double sa=eb[0],sb=ea[0],ra=nodes[a].r,rb=nodes[b].r;
        if(use_harmonic) {
            const double* ha=&harmonic_errors[size_t(a)*(p+1)];
            const double* hb=&harmonic_errors[size_t(b)*(p+1)];
            double blend=std::min(1.0,0.5*p*p*EPS2/q);
            for(int j=1;j<=p;j++) {
                sa=sa*ra+(hb[j]+blend*(eb[j]-hb[j]));
                sb=sb*rb+(ha[j]+blend*(ea[j]-ha[j]));
            }
        } else {
            for(int j=1;j<=p;j++) {
                sa=sa*ra+eb[j];sb=sb*rb+ea[j];
            }
        }
        double qp=q;
        for(int j=0;j<p/2;j++)qp*=q;
        pair_error_a=sa/qp;pair_error_b=sb/qp;
        return sa<tolerance*acc_est[a]*qp && sb<tolerance*acc_est[b]*qp;
    }
    void reset_lists(int newmode) {
        mode=newmode;
        far.clear();near.clear();point_far.clear();
        far_count=point_count=near_count=direct_work=sample_direct_work=0;
        far_seconds=point_seconds=near_seconds=0;
    }
    void flush_far() {
        if(far.empty())return;
        double t=cpu_time();
        switch(basis.p) {
            case 4:far_field<4>();break;case 6:far_field<6>();break;
            case 8:far_field<8>();break;case 10:far_field<10>();break;case 12:far_field<12>();break;
        }
        far_seconds+=cpu_time()-t;
        if(mode==2)far.clear();
    }
    void flush_point() {
        if(point_far.empty())return;
        double t=cpu_time();
        switch(basis.p) {
            case 4:point_field<4>();break;case 6:point_field<6>();break;
            case 8:point_field<8>();break;case 10:point_field<10>();break;case 12:point_field<12>();break;
        }
        point_seconds+=cpu_time()-t;
        if(mode==2)point_far.clear();
    }
    void flush_near() {
        if(near.empty())return;
        double t=cpu_time();
        near_field();
        near_seconds+=cpu_time()-t;
        if(mode==2)near.clear();
    }
    void emit_far(int a,int b) {
        far_count++;
        if(mode==1 && far.size()>=2048)return;
        far.push_back({a,b});
        if(mode==2 && far.size()>=1024) {flush_far();check_deadline();}
    }
    void emit_point(int a,int b) {
        for(int i=nodes[a].begin;i<nodes[a].end;i++) {
            point_count++;
            if(mode==1 && point_far.size()>=2048)continue;
            point_far.push_back({i,b});
            if(mode==2 && point_far.size()>=2048) {flush_point();check_deadline();}
        }
    }
    void emit_near(int a,int b) {
        near_count++;
        uint64_t work=uint64_t(nodes[a].end-nodes[a].begin)*(nodes[b].end-nodes[b].begin)*(a==b?1:2);
        direct_work+=work;
        if(mode==1 && (sample_direct_work>=8000000 || near.size()>=4096 || work<1024))return;
        near.push_back({a,b});
        if(mode==1)sample_direct_work+=work;
        if(mode==2 && near.size()>=512) {flush_near();check_deadline();}
    }
    void check_deadline() {
        if(cpu_time()>coarse_deadline)emergency=true;
    }
    double down_estimate() const {
        return std::pow(basis.p/10.0,5)*(5e-6*nodes.size()+1.5e-7*n);
    }
    void tune(double start,double limit) {
        double t=cpu_time();
        reset_lists(1);
        interactions(0,0);
        double list_time=cpu_time()-t;
        double rf=2e-6*std::pow(basis.p/10.0,5),rp=4e-7*std::pow(basis.p/10.0,3),rn=1.2e-9;
        if(!far.empty()) {
            double begin=cpu_time();int reps=0;
            do {flush_far();reps++;} while(cpu_time()-begin<0.005);
            rf=(cpu_time()-begin)/(reps*far.size());
        }
        if(!point_far.empty()) {
            double begin=cpu_time();int reps=0;
            do {flush_point();reps++;} while(cpu_time()-begin<0.002);
            rp=(cpu_time()-begin)/(reps*point_far.size());
        }
        point_rate=rp;
        if(sample_direct_work) {
            double begin=cpu_time();int reps=0;
            do {flush_near();reps++;} while(cpu_time()-begin<0.005);
            rn=(cpu_time()-begin)/(reps*sample_direct_work);
        }
        std::fill(locals.begin(),locals.end(),0.0);
        std::fill(ax.begin(),ax.end(),0.0);
        std::fill(ay.begin(),ay.end(),0.0);
        std::fill(az.begin(),az.end(),0.0);
        double previous_tol=0,previous_work=0;
        double write_reserve=std::max(0.02,0.025*limit);
        double target_fraction=setting("FMM_FRACTION",0.92);
        for(int iteration=0;iteration<4;iteration++) {
            double work=rf*far_count+rp*point_count+rn*direct_work+1e-8*near_count;
            double budget=std::max(0.01,target_fraction*limit-(cpu_time()-start)-down_estimate()-write_reserve-1.4*list_time);
            if(work<=budget && budget-work<std::max(0.025*budget,1.25*list_time))break;
            double slope=0.24;
            if(previous_tol && std::abs(std::log(previous_tol/tolerance))>0.1)
                slope=std::clamp(std::log(work/previous_work)/std::log(previous_tol/tolerance),0.12,0.5);
            double ratio=std::clamp(work/budget,0.1,10.0);
            double next=std::clamp(tolerance*std::pow(ratio,1/slope),1e-14,1e-1);
            if(std::abs(std::log(next/tolerance))<0.10)break;
            previous_tol=tolerance;previous_work=work;tolerance=next;
            if(iteration==3)break;
            t=cpu_time();
            reset_lists(1);
            interactions(0,0);
            list_time=cpu_time()-t;
        }
        deadline=start+limit-write_reserve;
        coarse_deadline=deadline-1.4*down_estimate()-std::max(0.01,0.015*limit);
        std::fprintf(stderr,"tune tol %.4g rates %.3g %.3g %.3g elapsed %.4f budget %.3f\n",
                     tolerance,rf,rp,rn,cpu_time()-start,limit);
        reset_lists(2);
    }
    void coarse_pair(int a,int b) {
        if(a==b)return;
        const Node& u=nodes[a];const Node& v=nodes[b];
        double dx=v.x-u.x,dy=v.y-u.y,dz=v.z-u.z,q=dx*dx+dy*dy+dz*dz+EPS2;
        double inv=1/(q*std::sqrt(q)),ma=moments[size_t(a)*basis.nm],mb=moments[size_t(b)*basis.nm];
        double* la=&locals[size_t(a)*basis.nl];double* lb=&locals[size_t(b)*basis.nl];
        la[3]+=mb*inv*dx;la[2]+=mb*inv*dy;la[1]+=mb*inv*dz;
        lb[3]-=ma*inv*dx;lb[2]-=ma*inv*dy;lb[1]-=ma*inv*dz;
    }
    void interactions(int a,int b) {
        if(mode==2 && emergency) {coarse_pair(a,b);return;}
        const Node& u=nodes[a];const Node& v=nodes[b];
        double dx=u.x-v.x,dy=u.y-v.y,dz=u.z-v.z;
        double r=u.r+v.r;
        double q=dx*dx+dy*dy+dz*dz+EPS2;
        if(r*r<theta*theta*q && acceptable(a,b,q)) {
            int small=std::max(1,basis.p-6);
            int na=u.end-u.begin,nb=v.end-v.begin;
            if(na<=small && nb<=small) {
                emit_near(a,b);
            } else if(std::min(na,nb)<=small) {
                if(mode==2) {
                    error_bound[a]+=pair_error_a;error_bound[b]+=pair_error_b;
                }
                if(na>nb)std::swap(a,b);
                emit_point(a,b);
            } else {
                if(mode==2) {
                    error_bound[a]+=pair_error_a;
                    if(a!=b)error_bound[b]+=pair_error_b;
                }
                emit_far(a,b);
            }
        } else if(u.leaf() && v.leaf()) {
            emit_near(a,b);
        } else if(a==b) {
            interactions(u.left,u.left);
            interactions(u.left,u.right);
            interactions(u.right,u.right);
        } else if(v.leaf() || (!u.leaf() && u.r>=v.r)) {
            interactions(u.left,b);
            interactions(u.right,b);
        } else {
            interactions(a,v.left);
            interactions(a,v.right);
        }
    }
    template<int P> void far_field() {
        constexpr int NL=(P+1)*(P+2)*(P+3)/6;
        constexpr int NM=P*(P+1)*(P+2)/6;
        alignas(32) V ma[NM],mb[NM],d[NL];
        alignas(32) double xa[4],ya[4],za[4],outa[4],outb[4];
        // Four independent cell pairs share SIMD lanes; both directions use
        // the same kernel derivatives, with the corresponding parity signs.
        for(size_t k=0;k<far.size();k+=4) {
            int count=std::min(size_t(4),far.size()-k);
            const double* srca[4];const double* srcb[4];
            double* dsta[4];double* dstb[4];
            double factor[4];
            for(int j=0;j<4;j++) {
                Pair pr=far[k+std::min(j,count-1)];
                const Node& a=nodes[pr.a];const Node& b=nodes[pr.b];
                xa[j]=a.x-b.x;ya[j]=a.y-b.y;za[j]=a.z-b.z;
                srca[j]=&moments[size_t(pr.a)*NM];srcb[j]=&moments[size_t(pr.b)*NM];
                dsta[j]=&locals[size_t(pr.a)*NL];dstb[j]=&locals[size_t(pr.b)*NL];
                factor[j]=pr.a==pr.b?0.5:1.0;
            }
            V fac=_mm256_loadu_pd(factor);
            for(int b=0;b<NM;b++) {
                ma[b]=_mm256_set_pd(srca[3][b],srca[2][b],srca[1][b],srca[0][b])*fac;
                mb[b]=_mm256_set_pd(srcb[3][b],srcb[2][b],srcb[1][b],srcb[0][b])*fac;
                if(basis.v[b].n&1)mb[b]=-mb[b];
            }
            derivatives<P>(_mm256_load_pd(xa),_mm256_load_pd(ya),_mm256_load_pd(za),d);
            for(int a=1;a<NL;a++) {
                const auto& sums=basis.sums[a];
                int size=sums.size(),b=0;
                V sa0=_mm256_setzero_pd(),sa1=sa0,sa2=sa0,sa3=sa0;
                V sb0=sa0,sb1=sa0,sb2=sa0,sb3=sa0;
                for(;b+3<size;b+=4) {
                    V d0=d[sums[b]],d1=d[sums[b+1]],d2=d[sums[b+2]],d3=d[sums[b+3]];
                    sa0=_mm256_fmadd_pd(d0,mb[b],sa0);
                    sb0=_mm256_fmadd_pd(d0,ma[b],sb0);
                    sa1=_mm256_fmadd_pd(d1,mb[b+1],sa1);
                    sb1=_mm256_fmadd_pd(d1,ma[b+1],sb1);
                    sa2=_mm256_fmadd_pd(d2,mb[b+2],sa2);
                    sb2=_mm256_fmadd_pd(d2,ma[b+2],sb2);
                    sa3=_mm256_fmadd_pd(d3,mb[b+3],sa3);
                    sb3=_mm256_fmadd_pd(d3,ma[b+3],sb3);
                }
                sa0=(sa0+sa1)+(sa2+sa3);sb0=(sb0+sb1)+(sb2+sb3);
                for(;b<size;b++) {
                    sa0=_mm256_fmadd_pd(d[sums[b]],mb[b],sa0);
                    sb0=_mm256_fmadd_pd(d[sums[b]],ma[b],sb0);
                }
                if(basis.v[a].n&1)sb0=-sb0;
                _mm256_store_pd(outa,sa0);_mm256_store_pd(outb,sb0);
                for(int j=0;j<count;j++) {dsta[j][a]+=outa[j];dstb[j][a]+=outb[j];}
            }
        }
    }
    template<int P> void point_field() {
        constexpr int NL=(P+1)*(P+2)*(P+3)/6;
        constexpr int NM=P*(P+1)*(P+2)/6;
        alignas(32) V d[NL];
        alignas(32) double xx[4],yy[4],zz[4],mm[4],sx[4],sy[4],sz[4],out[4];
        for(size_t k=0;k<point_far.size();k+=4) {
            int count=std::min(size_t(4),point_far.size()-k),ids[4];
            const double* src[4];double* dst[4];
            for(int j=0;j<4;j++) {
                Pair pr=point_far[k+std::min(j,count-1)];
                ids[j]=pr.a;const Node& node=nodes[pr.b];
                xx[j]=x[pr.a]-node.x;yy[j]=y[pr.a]-node.y;zz[j]=z[pr.a]-node.z;
                mm[j]=m[pr.a];
                src[j]=&moments[size_t(pr.b)*NM];
                dst[j]=&locals[size_t(pr.b)*NL];
            }
            derivatives<P>(_mm256_load_pd(xx),_mm256_load_pd(yy),_mm256_load_pd(zz),d);
            V vx=_mm256_setzero_pd(),vy=vx,vz=vx;
            for(int b=0;b<NM;b++) {
                V mass=_mm256_set_pd(src[3][b],src[2][b],src[1][b],src[0][b]);
                if(basis.v[b].n&1)mass=-mass;
                vx=_mm256_fmadd_pd(d[basis.gx[b]],mass,vx);
                vy=_mm256_fmadd_pd(d[basis.gy[b]],mass,vy);
                vz=_mm256_fmadd_pd(d[basis.gz[b]],mass,vz);
            }
            _mm256_store_pd(sx,vx);_mm256_store_pd(sy,vy);_mm256_store_pd(sz,vz);
            for(int j=0;j<count;j++) {ax[ids[j]]+=sx[j];ay[ids[j]]+=sy[j];az[ids[j]]+=sz[j];}
            V mass=_mm256_load_pd(mm);
            for(int a=1;a<NL;a++) {
                V value=d[a]*mass;
                if(basis.v[a].n&1)value=-value;
                _mm256_store_pd(out,value);
                for(int j=0;j<count;j++)dst[j][a]+=out[j];
            }
        }
    }
    template<bool Fast> void direct_one(const Node& a,const Node& b) {
        int i=a.begin;
        for(;i+3<a.end;i+=4) {
            V xi=_mm256_loadu_pd(&x[i]),yi=_mm256_loadu_pd(&y[i]),zi=_mm256_loadu_pd(&z[i]);
            V sx=_mm256_setzero_pd(),sy=sx,sz=sx;
            for(int j=b.begin;j<b.end;j++) {
                V dx=splat(x[j])-xi,dy=splat(y[j])-yi,dz=splat(z[j])-zi;
                V q=_mm256_fmadd_pd(dx,dx,splat(EPS2));
                q=_mm256_fmadd_pd(dy,dy,q);q=_mm256_fmadd_pd(dz,dz,q);
                V scale;
                if constexpr(Fast) {
                    V ri=_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(q)));
                    V halfq=splat(0.5)*q;
                    ri=ri*_mm256_fnmadd_pd(halfq,ri*ri,splat(1.5));
                    ri=ri*_mm256_fnmadd_pd(halfq,ri*ri,splat(1.5));
                    scale=splat(m[j])*ri*ri*ri;
                } else scale=splat(m[j])/(q*_mm256_sqrt_pd(q));
                sx=_mm256_fmadd_pd(dx,scale,sx);
                sy=_mm256_fmadd_pd(dy,scale,sy);
                sz=_mm256_fmadd_pd(dz,scale,sz);
            }
            _mm256_storeu_pd(&ax[i],_mm256_loadu_pd(&ax[i])+sx);
            _mm256_storeu_pd(&ay[i],_mm256_loadu_pd(&ay[i])+sy);
            _mm256_storeu_pd(&az[i],_mm256_loadu_pd(&az[i])+sz);
        }
        for(;i<a.end;i++) {
            double sx=0,sy=0,sz=0;
            for(int j=b.begin;j<b.end;j++) {
                double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
                double q=dx*dx+dy*dy+dz*dz+EPS2;
                double scale=m[j]/(q*std::sqrt(q));
                sx+=dx*scale;sy+=dy*scale;sz+=dz*scale;
            }
            ax[i]+=sx;ay[i]+=sy;az[i]+=sz;
        }
    }
    void near_field() {
        for(auto pr:near) {
            if(mutual_direct && pr.a!=pr.b) {
                direct_mutual(nodes[pr.a],nodes[pr.b]);
            } else if(fast_direct) {
                direct_one<true>(nodes[pr.a],nodes[pr.b]);
                if(pr.a!=pr.b)direct_one<true>(nodes[pr.b],nodes[pr.a]);
            } else {
                direct_one<false>(nodes[pr.a],nodes[pr.b]);
                if(pr.a!=pr.b)direct_one<false>(nodes[pr.b],nodes[pr.a]);
            }
        }
    }
    void direct_mutual(const Node& a,const Node& b) {
        int i=a.begin;
        for(;i+3<a.end;i+=4) {
            V xi=_mm256_loadu_pd(&x[i]),yi=_mm256_loadu_pd(&y[i]),zi=_mm256_loadu_pd(&z[i]),mi=_mm256_loadu_pd(&m[i]);
            V sx=_mm256_setzero_pd(),sy=sx,sz=sx;
            for(int j=b.begin;j<b.end;j++) {
                V dx=splat(x[j])-xi,dy=splat(y[j])-yi,dz=splat(z[j])-zi;
                V q=_mm256_fmadd_pd(dx,dx,splat(EPS2));
                q=_mm256_fmadd_pd(dy,dy,q);q=_mm256_fmadd_pd(dz,dz,q);
                V s=splat(1.0)/(q*_mm256_sqrt_pd(q)),smj=s*splat(m[j]),smi=s*mi;
                sx=_mm256_fmadd_pd(dx,smj,sx);
                sy=_mm256_fmadd_pd(dy,smj,sy);
                sz=_mm256_fmadd_pd(dz,smj,sz);
                ax[j]-=horizontal_sum(dx*smi);
                ay[j]-=horizontal_sum(dy*smi);
                az[j]-=horizontal_sum(dz*smi);
            }
            _mm256_storeu_pd(&ax[i],_mm256_loadu_pd(&ax[i])+sx);
            _mm256_storeu_pd(&ay[i],_mm256_loadu_pd(&ay[i])+sy);
            _mm256_storeu_pd(&az[i],_mm256_loadu_pd(&az[i])+sz);
        }
        for(;i<a.end;i++) {
            double sx=0,sy=0,sz=0;
            for(int j=b.begin;j<b.end;j++) {
                double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
                double q=dx*dx+dy*dy+dz*dz+EPS2,s=1.0/(q*std::sqrt(q));
                sx+=dx*s*m[j];sy+=dy*s*m[j];sz+=dz*s*m[j];
                ax[j]-=dx*s*m[i];ay[j]-=dy*s*m[i];az[j]-=dz*s*m[i];
            }
            ax[i]+=sx;ay[i]+=sy;az[i]+=sz;
        }
    }
    void downward() {
        const int nl=basis.nl;
        double f[455];
        bool simple=false;
        for(size_t k=0;k<nodes.size();k++) {
            if(k%512==0 && cpu_time()>deadline-0.005)simple=true;
            const Node& t=nodes[k];
            const double* in=&locals[k*nl];
            if(!t.leaf()) {
                error_bound[t.left]+=error_bound[k];error_bound[t.right]+=error_bound[k];
            }
            if(simple) {
                if(t.leaf()) {
                    for(int i=t.begin;i<t.end;i++) {ax[i]+=in[3];ay[i]+=in[2];az[i]+=in[1];}
                } else {
                    for(int child:{t.left,t.right}) {
                        double* out=&locals[size_t(child)*nl];
                        out[3]+=in[3];out[2]+=in[2];out[1]+=in[1];
                    }
                }
                continue;
            }
            if(t.leaf()) {
                int i=t.begin;
                alignas(32) V vf[455];
                for(;i+3<t.end;i+=4) {
                    basis.powers4(_mm256_loadu_pd(&x[i])-splat(t.x),
                                  _mm256_loadu_pd(&y[i])-splat(t.y),
                                  _mm256_loadu_pd(&z[i])-splat(t.z),vf);
                    V sx=_mm256_setzero_pd(),sy=sx,sz=sx;
                    for(int a=0;a<basis.nm;a++) {
                        sx=_mm256_fmadd_pd(splat(in[basis.gx[a]]),vf[a],sx);
                        sy=_mm256_fmadd_pd(splat(in[basis.gy[a]]),vf[a],sy);
                        sz=_mm256_fmadd_pd(splat(in[basis.gz[a]]),vf[a],sz);
                    }
                    _mm256_storeu_pd(&ax[i],_mm256_loadu_pd(&ax[i])+sx);
                    _mm256_storeu_pd(&ay[i],_mm256_loadu_pd(&ay[i])+sy);
                    _mm256_storeu_pd(&az[i],_mm256_loadu_pd(&az[i])+sz);
                }
                for(;i<t.end;i++) {
                    if(t.end-t.begin==1) {
                        ax[i]+=in[3];ay[i]+=in[2];az[i]+=in[1];
                        continue;
                    }
                    basis.powers(x[i]-t.x,y[i]-t.y,z[i]-t.z,f);
                    double sx=0,sy=0,sz=0;
                    for(int a=0;a<basis.nm;a++) {
                        sx+=in[basis.gx[a]]*f[a];
                        sy+=in[basis.gy[a]]*f[a];
                        sz+=in[basis.gz[a]]*f[a];
                    }
                    ax[i]+=sx;ay[i]+=sy;az[i]+=sz;
                }
            } else {
                for(int child : {t.left,t.right}) {
                    const Node& c=nodes[child];
                    basis.powers(c.x-t.x,c.y-t.y,c.z-t.z,f);
                    double* out=&locals[size_t(child)*nl];
                    if(c.end-c.begin==1) {
                        for(int a=0;a<basis.nm;a++) {
                            auto u=basis.v[a];
                            out[3]+=in[basis.idx[u.x+1][u.y][u.z]]*f[a];
                            out[2]+=in[basis.idx[u.x][u.y+1][u.z]]*f[a];
                            out[1]+=in[basis.idx[u.x][u.y][u.z+1]]*f[a];
                        }
                    } else {
                        for(int a=1;a<nl;a++) {
                            const auto& sums=basis.sums[a];
                            int size=sums.size(),b=0;
                            double s0=0,s1=0,s2=0,s3=0;
                            for(;b+3<size;b+=4) {
                                s0+=in[sums[b]]*f[b];
                                s1+=in[sums[b+1]]*f[b+1];
                                s2+=in[sums[b+2]]*f[b+2];
                                s3+=in[sums[b+3]]*f[b+3];
                            }
                            double sum=(s0+s1)+(s2+s3);
                            for(;b<size;b++)sum+=in[sums[b]]*f[b];
                            out[a]+=sum;
                        }
                    }
                }
            }
        }
    }
    int refine() {
        if(basis.p!=10 || emergency)return 0;
        double stop=deadline-0.01-2e-9*n;
        if(cpu_time()+0.02+2e-8*n>stop)return 0;
        std::vector<std::pair<double,int>> rank;
        rank.reserve(n);
        for(size_t k=0;k<nodes.size();k++)if(nodes[k].leaf()) {
            const auto& t=nodes[k];
            for(int i=t.begin;i<t.end;i++) {
                double norm=std::sqrt(ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i]);
                rank.push_back({-error_bound[k]/std::max(1e-30,norm),i});
            }
        }
        int maximum=std::min(n/10,20000);
        if(maximum<int(rank.size()))std::nth_element(rank.begin(),rank.begin()+maximum,rank.end());
        std::sort(rank.begin(),rank.begin()+maximum);
        std::vector<Pair> source_far,source_near;
        source_far.reserve(32768);source_near.reserve(4096);
        constexpr int P=10,NL=286,NM=220;
        int refined=0;
        for(int base=0;base<maximum;base+=8) {
            int count=std::min(8,maximum-base),ids[8];
            double result[8][3]={};
            source_far.clear();source_near.clear();
            uint64_t nearwork=0;
            bool abort=false;
            for(int j=0;j<count && !abort;j++) {
                int i=ids[j]=rank[base+j].second;
                double anorm=std::sqrt(ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i]);
                double tol=std::min(1e-11,tolerance*0.002)*std::max(anorm,1e-30);
                int stack[256],top=0,visits=0;stack[top++]=0;
                while(top) {
                    if((++visits&2047)==0 && cpu_time()>stop-0.005) {abort=true;break;}
                    int k=stack[--top];const auto& node=nodes[k];
                    double dx=x[i]-node.x,dy=y[i]-node.y,dz=z[i]-node.z;
                    double q=dx*dx+dy*dy+dz*dz+EPS2,qp=q*q*q;qp*=qp;
                    if(node.r*node.r<0.36*q && error_moments[size_t(k)*11+10]<tol*qp) {
                        source_far.push_back({j,k});
                    } else if(node.leaf()) {
                        source_near.push_back({j,k});
                        nearwork+=node.end-node.begin;
                    } else {stack[top++]=node.left;stack[top++]=node.right;}
                }
            }
            double predicted=source_far.size()*point_rate+nearwork*1.5e-9;
            if(abort || cpu_time()+1.5*predicted+0.003>stop)break;
            alignas(32) V d[NL];
            alignas(32) double xx[4],yy[4],zz[4],sx[4],sy[4],sz[4];
            for(size_t k=0;k<source_far.size();k+=4) {
                int lanes=std::min(size_t(4),source_far.size()-k),target[4];
                const double* src[4];
                for(int j=0;j<4;j++) {
                    Pair pr=source_far[k+std::min(j,lanes-1)];
                    target[j]=pr.a;int i=ids[pr.a];const auto& node=nodes[pr.b];
                    xx[j]=x[i]-node.x;yy[j]=y[i]-node.y;zz[j]=z[i]-node.z;
                    src[j]=&moments[size_t(pr.b)*NM];
                }
                derivatives<P>(_mm256_load_pd(xx),_mm256_load_pd(yy),_mm256_load_pd(zz),d);
                V vx=_mm256_setzero_pd(),vy=vx,vz=vx;
                for(int b=0;b<NM;b++) {
                    V mass=_mm256_set_pd(src[3][b],src[2][b],src[1][b],src[0][b]);
                    if(basis.v[b].n&1)mass=-mass;
                    vx=_mm256_fmadd_pd(d[basis.gx[b]],mass,vx);
                    vy=_mm256_fmadd_pd(d[basis.gy[b]],mass,vy);
                    vz=_mm256_fmadd_pd(d[basis.gz[b]],mass,vz);
                }
                _mm256_store_pd(sx,vx);_mm256_store_pd(sy,vy);_mm256_store_pd(sz,vz);
                for(int j=0;j<lanes;j++) {
                    result[target[j]][0]+=sx[j];result[target[j]][1]+=sy[j];result[target[j]][2]+=sz[j];
                }
            }
            for(auto pr:source_near) {
                int i=ids[pr.a];const auto& node=nodes[pr.b];
                V xi=splat(x[i]),yi=splat(y[i]),zi=splat(z[i]),sxv=_mm256_setzero_pd(),syv=sxv,szv=sxv;
                int j=node.begin;
                for(;j+3<node.end;j+=4) {
                    V dx=_mm256_loadu_pd(&x[j])-xi,dy=_mm256_loadu_pd(&y[j])-yi,dz=_mm256_loadu_pd(&z[j])-zi;
                    V q=_mm256_fmadd_pd(dx,dx,splat(EPS2));q=_mm256_fmadd_pd(dy,dy,q);q=_mm256_fmadd_pd(dz,dz,q);
                    V s=_mm256_loadu_pd(&m[j])/(q*_mm256_sqrt_pd(q));
                    sxv=_mm256_fmadd_pd(dx,s,sxv);syv=_mm256_fmadd_pd(dy,s,syv);szv=_mm256_fmadd_pd(dz,s,szv);
                }
                result[pr.a][0]+=horizontal_sum(sxv);result[pr.a][1]+=horizontal_sum(syv);result[pr.a][2]+=horizontal_sum(szv);
                for(;j<node.end;j++) {
                    double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i],q=dx*dx+dy*dy+dz*dz+EPS2;
                    double s=m[j]/(q*std::sqrt(q));
                    result[pr.a][0]+=dx*s;result[pr.a][1]+=dy*s;result[pr.a][2]+=dz*s;
                }
            }
            for(int j=0;j<count;j++) {int i=ids[j];ax[i]=result[j][0];ay[i]=result[j][1];az[i]=result[j][2];}
            refined+=count;
        }
        return refined;
    }
    template<int P> void debug_target(int original) {
        int i=0;
        while(i<n && part[i].id!=uint32_t(original))i++;
        if(i==n)return;
        double accum[3]={},exact[3]={};
        double maxe=0;
        for(auto pr:far) {
            for(int side=0;side<2;side++) {
                int a=side?pr.b:pr.a,b=side?pr.a:pr.b;
                const Node& u=nodes[a];const Node& v=nodes[b];
                if(i<u.begin || i>=u.end || (side && a==b))continue;
                alignas(32) V deriv[455];
                double loc[455]={},f[455],dd[455];
                derivatives<P>(splat(u.x-v.x),splat(u.y-v.y),splat(u.z-v.z),deriv);
                for(int j=0;j<basis.nl;j++)dd[j]=deriv[j][0];
                const double* mm=&moments[size_t(b)*basis.nm];
                for(int j=1;j<basis.nl;j++)
                    for(size_t k=0;k<basis.sums[j].size();k++)
                        loc[j]+=((basis.v[k].n&1)?-1:1)*mm[k]*dd[basis.sums[j][k]];
                basis.powers(x[i]-u.x,y[i]-u.y,z[i]-u.z,f);
                double pred[3]={},ref[3]={};
                for(int j=0;j<basis.nm;j++) {
                    auto ij=basis.v[j];
                    pred[0]+=loc[basis.idx[ij.x+1][ij.y][ij.z]]*f[j];
                    pred[1]+=loc[basis.idx[ij.x][ij.y+1][ij.z]]*f[j];
                    pred[2]+=loc[basis.idx[ij.x][ij.y][ij.z+1]]*f[j];
                }
                for(int j=v.begin;j<v.end;j++) {
                    double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
                    double q=dx*dx+dy*dy+dz*dz+EPS2,s=m[j]/(q*std::sqrt(q));
                    ref[0]+=dx*s;ref[1]+=dy*s;ref[2]+=dz*s;
                }
                double e=0,norm=0;
                for(int j=0;j<3;j++) {accum[j]+=pred[j];exact[j]+=ref[j];e+=(pred[j]-ref[j])*(pred[j]-ref[j]);norm+=ref[j]*ref[j];}
                e=std::sqrt(e);
                if(e>maxe) {
                    maxe=e;
                    double dx=u.x-v.x,dy=u.y-v.y,dz=u.z-v.z,q=dx*dx+dy*dy+dz*dz+EPS2;
                    double est=0;
                    if(tolerance) {
                        auto ee=&error_moments[size_t(b)*(P+1)];
                        est=ee[0];
                        for(int j=1;j<=P;j++)est=est*u.r+ee[j];
                        est/=std::pow(q,(P+2)/2);
                    }
                    fprintf(stderr,"dbg pair %d %d counts %d %d radii %.4g %.4g dist %.4g err %.9g force %.4g est %.9g acc %.4g\n",
                            a,b,u.end-u.begin,v.end-v.begin,u.r,v.r,std::sqrt(q),e,std::sqrt(norm),est,tolerance?acc_est[a]:0);
                }
            }
        }
        fprintf(stderr,"dbg far predicted %.15g %.15g %.15g exact %.15g %.15g %.15g full %.15g %.15g %.15g\n",
                accum[0],accum[1],accum[2],exact[0],exact[1],exact[2],ax[i],ay[i],az[i]);
    }
    bool write(const char* path) {
        std::vector<std::array<double,3>> out(n);
        for(int i=0;i<n;i++) {
            double a[3]={ax[i],ay[i],az[i]},b[3]={};
            if(rotated) {
                for(int j=0;j<3;j++)for(int k=0;k<3;k++)b[j]+=rotation[k][j]*a[k];
            } else std::copy(a,a+3,b);
            out[part[i].id]={std::isfinite(b[0])?b[0]:0.0,std::isfinite(b[1])?b[1]:0.0,std::isfinite(b[2])?b[2]:0.0};
        }
        FILE* f=std::fopen(path,"wb");if(!f)return false;
        bool ok=fwrite(out.data(),24,n,f)==size_t(n);
        return fclose(f)==0 && ok;
    }
};

int main(int argc,char** argv) {
    if(argc!=3)return 1;
    double start=cpu_time();
    int p=setting("FMM_ORDER",10);
    if(p!=4 && p!=6 && p!=8 && p!=10 && p!=12)p=10;
    int leaf=setting("FMM_LEAF",64);
    double theta=setting("FMM_THETA",0.8);
    Solver solver(p,leaf,theta);
    solver.leaf_geom=setting("FMM_GEOM",0.12);
    solver.tolerance=setting("FMM_TOL",1e-7);
    solver.use_harmonic=setting("FMM_HARMONIC",0);
    solver.fast_direct=setting("FMM_FAST",0);
    solver.mutual_direct=setting("FMM_MUTUAL",0);
    if(!solver.read(argv[1]))return 2;
    double t1=cpu_time();
    if(solver.n<=1024) {
        solver.direct_one<false>(solver.nodes[0],solver.nodes[0]);
        return solver.write(argv[2])?0:3;
    }
    solver.upward();
    if(solver.tolerance)solver.make_error_estimates();
    double t2=cpu_time();
    bool automatic=std::getenv("GALAXY_CPU_LIMIT_S") && !std::getenv("FMM_TOL");
    if(automatic)solver.tune(start,setting("GALAXY_CPU_LIMIT_S",2.0));
    else solver.reset_lists(2);
    double tplan=cpu_time();
    solver.interactions(0,0);
    solver.flush_far();solver.flush_point();solver.flush_near();
    double t3=cpu_time();
    solver.downward();
    int refined=0;
    if(automatic && setting("FMM_REFINE",1))refined=solver.refine();
    double t6=cpu_time();
    if(!solver.write(argv[2]))return 3;
    std::fprintf(stderr,"n %d p %d leaf %d theta %.4f nodes %zu far %llu point %llu near %llu direct %.3g | tree %.4f up %.4f plan %.4f far %.4f point %.4f near %.4f down %.4f total %.4f emergency %d\n",
                 solver.n,p,solver.leafsize,theta,solver.nodes.size(),
                 (unsigned long long)solver.far_count,(unsigned long long)solver.point_count,(unsigned long long)solver.near_count,double(solver.direct_work),
                 t1-start,t2-t1,tplan-t2,solver.far_seconds,solver.point_seconds,solver.near_seconds,t6-t3,cpu_time()-start,solver.emergency);
    if(refined)std::fprintf(stderr,"refined %d\n",refined);
    return 0;
}
