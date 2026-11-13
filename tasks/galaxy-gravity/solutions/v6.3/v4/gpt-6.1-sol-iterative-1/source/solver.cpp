#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <sys/resource.h>
#include <vector>

static constexpr int MAXP = 12;
static constexpr int MAXC = (MAXP+1)*(MAXP+2)*(MAXP+3)/6;
static constexpr int MAXH = 16;
static constexpr int MAXHC = (MAXH+1)*(MAXH+1);
static constexpr int MAXVEC = MAXHC>MAXC ? MAXHC : MAXC;
static constexpr double EPS2 = 0.0001;
static constexpr auto RECIP=[] {
    std::array<double,2*MAXH+1> v={};
    for (int k=1;k<=2*MAXH;k++) v[k]=1./k;
    return v;
}();

static double cpu_time() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

struct Particle {
    double x[3], m;
    uint32_t id;
};

struct Node {
    double c[3], radius, mass;
    double acc_scale;
    int begin, end, left, right;
};

struct Table {
    struct Row { uint16_t source,derivative; uint8_t length,flags; };
    int p, count, moments;
    int index[MAXP+1][MAXP+1][MAXP+1];
    int ex[MAXC][3], degree[MAXC];
    int prev1[MAXC][3], prev2[MAXC][3];
    double factorial[MAXC];
    double inv_factorial[MAXC];
    std::vector<Row> rows;
    std::vector<int> row_begin;
    explicit Table(int order) : p(order) {
        std::memset(index, -1, sizeof(index));
        count=0;
        for (int n=0;n<=p;n++)
            for (int a=n;a>=0;a--)
                for (int b=n-a;b>=0;b--) {
                    int c=n-a-b;
                    index[a][b][c]=count;
                    ex[count][0]=a;
                    ex[count][1]=b;
                    ex[count][2]=c;
                    degree[count++]=n;
                }
        moments=p*(p+1)*(p+2)/6;
        double fac[MAXP+1];
        fac[0]=1.;
        for (int n=1;n<=p;n++) fac[n]=fac[n-1]*n;
        for (int i=0;i<count;i++) {
            int a=ex[i][0], b=ex[i][1], c=ex[i][2];
            factorial[i]=fac[a]*fac[b]*fac[c];
            inv_factorial[i]=1./factorial[i];
            for (int k=0;k<3;k++) {
                int e[3]={a,b,c};
                e[k]--;
                prev1[i][k]=e[k]>=0 ? index[e[0]][e[1]][e[2]] : -1;
                e[k]--;
                prev2[i][k]=e[k]>=0 ? index[e[0]][e[1]][e[2]] : -1;
            }
        }
        row_begin.resize(count+1);
        for (int j=1;j<count;j++) {
            row_begin[j]=rows.size();
            int a=ex[j][0],b=ex[j][1],c=ex[j][2];
            for (int m=0;m<=p-degree[j];m++) {
                if (m==1) continue;
                for (int aa=m;aa>=0;aa--) {
                    int bb=m-aa;
                    rows.push_back({uint16_t(index[aa][bb][0]),
                                    uint16_t(index[a+aa][b+bb][c]),
                                    uint8_t(bb+1),0});
                }
            }
        }
        row_begin[count]=rows.size();
    }
};

class Solver {
public:
    int n, leaf_size, order, nc, nm;
    double theta2;
    Table tab;
    std::vector<Node> tree;
    std::vector<Particle> particles;
    std::vector<double> x, y, z, mass, ax, ay, az;
    std::vector<uint32_t> ids;
    std::vector<double> moments, locals;
    uint64_t translations=0, direct_pairs=0;
    double started;
    struct Interaction { int a,b; double dx,dy,dz; };
    Interaction batch[4];
    int batch_count=0;
    static constexpr int BINS=128;
    struct Bucket { Interaction data[4]; int count=0; };
    Bucket pending[BINS];
    int batch_horder=0,batch_soft=0,batch_bin=0;
    bool calibrating=false;
    double bin_time[BINS]={},bin_cost[BINS]={};
    uint64_t bin_calls[BINS]={};
    int horder=0, hn=0;
    std::vector<double> hmoments, hlocals;
    double near_cut=0.02;
    double leaf_radius_factor=0.;
    double global_c[3]={0.,0.,0.},global_scale=1.;
    bool max_mac=false;
    bool compact=false;
    bool balanced=false;
    bool variable_soft=false;
    double variable_h=0.;
    bool newton_only=false;
    bool full_h=false;
    bool relative_mac=false;
    bool power_mac=false;
    bool target_mac=false;
    bool adaptive_h=false;
    bool adaptive_c=false;
    double adaptive_safety=.25;
    uint64_t direct_cutoff=256;
    bool near_soft=false;
    bool align_flat=true,rotated=false;
    bool pipelined_direct=true;
    double rotation[3][3]={{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
    bool pade=false;
    double force_tol=1e-6;
    double soft_weight=1.;
    double coarse_theta2=.64;
    bool coarse_max=false;
    std::vector<double> multipole_power;
    std::vector<double> target_norm,radial_power,inverse_acc2;
    double choose_h[MAXH+1]={};
    double choose_c[MAXP+1]={};
    double choose_all[MAXH+1][MAXH+1]={};
    uint64_t selected_horders[MAXH+1]={};
    uint64_t selected_corders[MAXP+1]={};
    double deadline=INFINITY, predicted_down=0.;
    bool emergency_mode=false;
    bool hard_emergency=false;
    double hard_deadline=INFINITY;
    uint64_t emergency_calls=0;
    bool runtime_control=true,tracking_work=false;
    double processed_work=0.,expected_work=0.,pair_model=0.,walk_model=0.;
    double control_time=0.,control_work=0.,control_slope=.25;
    int control_adjustments=0;
    double timing_bias=1.;
    double budget_fraction=.87;
    uint64_t calls=0;
    struct Stats {
        uint64_t m2l=0;
        double pairs=0.;
        uint64_t direct=0;
        uint64_t bins[BINS]={};
        std::vector<Interaction> far;
        std::vector<Interaction> far_bins[BINS];
        std::vector<std::array<int,2>> close;
    };

    Solver(int n_, int p, int leaf, double theta, int hp=0)
        : n(n_), leaf_size(leaf), order(p), theta2(theta*theta), tab(p),
          horder(hp), hn((hp+1)*(hp+1)) {
        nc=tab.count;
        nm=tab.moments;
        for (int p=0;p<=MAXH;p++) {
            choose_all[p][0]=1.;
            for (int k=1;k<=p;k++)
                choose_all[p][k]=choose_all[p][k-1]*(p-k+1)/k;
        }
        started=cpu_time();
        tree.reserve((n/leaf_size+1)*3);
    }

    int build_node(int begin, int end) {
        Node nd;
        nd.begin=begin; nd.end=end; nd.left=nd.right=-1;
        double lo[3]={INFINITY,INFINITY,INFINITY};
        double hi[3]={-INFINITY,-INFINITY,-INFINITY};
        double c[3]={0.,0.,0.}, m=0.;
        for (int i=begin;i<end;i++) {
            const Particle& q=particles[i];
            m+=q.m;
            for (int k=0;k<3;k++) {
                lo[k]=std::min(lo[k],q.x[k]);
                hi[k]=std::max(hi[k],q.x[k]);
                c[k]+=q.m*q.x[k];
            }
        }
        nd.mass=m;
        nd.acc_scale=1.;
        for (int k=0;k<3;k++) nd.c[k]=c[k]/m;
        double r2=0.;
        for (int i=begin;i<end;i++) {
            double dx=particles[i].x[0]-nd.c[0];
            double dy=particles[i].x[1]-nd.c[1];
            double dz=particles[i].x[2]-nd.c[2];
            r2=std::max(r2,dx*dx+dy*dy+dz*dz);
        }
        nd.radius=std::sqrt(r2);
        int index=tree.size();
        tree.push_back(nd);
        double dc0=nd.c[0]-global_c[0],dc1=nd.c[1]-global_c[1],dc2=nd.c[2]-global_c[2];
        double limit=leaf_radius_factor*(std::sqrt(dc0*dc0+dc1*dc1+dc2*dc2)+global_scale);
        if (end-begin>leaf_size ||
            (leaf_radius_factor>0. && end-begin>1 && nd.radius>limit)) {
            int axis=0;
            if (hi[1]-lo[1]>hi[axis]-lo[axis]) axis=1;
            if (hi[2]-lo[2]>hi[axis]-lo[axis]) axis=2;
            double split=(lo[axis]+hi[axis])*0.5;
            int middle=std::partition(particles.begin()+begin,particles.begin()+end,
                                     [axis,split](const Particle& q) {
                                         return q.x[axis]<split;
                                     })-particles.begin();
            if (balanced) {
                middle=(begin+end)/2;
                std::nth_element(particles.begin()+begin,particles.begin()+middle,
                                 particles.begin()+end,[axis](const Particle& a,const Particle& b) {
                                     return a.x[axis]<b.x[axis];
                                 });
            }
            if (middle==begin || middle==end) middle=(begin+end)/2;
            int l=build_node(begin,middle);
            int r=build_node(middle,end);
            tree[index].left=l;
            tree[index].right=r;
        }
        return index;
    }

    void prepare() {
        int samples=std::min(n,4096);
        std::vector<double> values(samples);
        for (int k=0;k<3;k++) {
            for (int i=0;i<samples;i++) values[i]=particles[size_t(i)*n/samples].x[k];
            std::nth_element(values.begin(),values.begin()+samples/2,values.end());
            global_c[k]=values[samples/2];
        }
        for (int i=0;i<samples;i++) {
            const Particle& q=particles[size_t(i)*n/samples];
            double dx=q.x[0]-global_c[0],dy=q.x[1]-global_c[1],dz=q.x[2]-global_c[2];
            values[i]=std::sqrt(dx*dx+dy*dy+dz*dz);
        }
        std::nth_element(values.begin(),values.begin()+samples/2,values.end());
        global_scale=values[samples/2];
        if (align_flat && n>1024) align_distribution(samples);
        build_node(0,n);
        for (;;) {
            size_t scalars=nm+nc+(horder?2*hn:0);
            if (power_mac) scalars+=horder+1;
            if (target_mac) scalars+=horder+order+2;
            double estimate=double(tree.size())*scalars*8.+double(n)*84.+
                            double(tree.capacity())*sizeof(Node)+(128ULL<<20);
            if (estimate<=1.85*(1ULL<<30)) break;
            size_t before=tree.size();
            leaf_size=std::min(n,(leaf_size*3+1)/2);
            tree.clear();
            build_node(0,n);
            if (tree.size()>.95*before) leaf_radius_factor*=1.5;
        }
        x.resize(n); y.resize(n); z.resize(n); mass.resize(n); ids.resize(n);
        ax.assign(n,0.); ay.assign(n,0.); az.assign(n,0.);
        for (int i=0;i<n;i++) {
            x[i]=particles[i].x[0];
            y[i]=particles[i].x[1];
            z[i]=particles[i].x[2];
            mass[i]=particles[i].m;
            ids[i]=particles[i].id;
        }
        std::vector<Particle>().swap(particles);
        moments.assign(size_t(tree.size())*nm,0.);
        locals.assign(size_t(tree.size())*nc,0.);
        if (target_mac) inverse_acc2.resize(n);
        if (relative_mac) {
            coarse_interact(0,0);
            coarse_down(0);
            clear_forces();
        }
        if (horder) {
            hmoments.assign(size_t(tree.size())*hn,0.);
            hlocals.assign(size_t(tree.size())*hn,0.);
            upward_harmonic(0);
        } else {
            upward(0);
        }
        if (power_mac && horder) prepare_power();
        if (target_mac && horder) prepare_target_norms();
    }

    void align_distribution(int samples) {
        double cov[3][3]={},vectors[3][3]={{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
        double cutoff=4.*global_scale;
        for (int i=0;i<samples;i++) {
            const Particle& p=particles[size_t(i)*n/samples];
            double r[3]={p.x[0]-global_c[0],p.x[1]-global_c[1],p.x[2]-global_c[2]};
            double d2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
            if (d2>cutoff*cutoff) continue;
            for (int a=0;a<3;a++)
                for (int b=0;b<3;b++) cov[a][b]+=r[a]*r[b];
        }
        double trace=cov[0][0]+cov[1][1]+cov[2][2];
        for (int iter=0;iter<24;iter++) {
            int p=0,q=1;
            if (std::abs(cov[0][2])>std::abs(cov[p][q])) p=0,q=2;
            if (std::abs(cov[1][2])>std::abs(cov[p][q])) p=1,q=2;
            double off=cov[p][q];
            if (std::abs(off)<=trace*1e-14) break;
            double tau=(cov[q][q]-cov[p][p])/(2.*off);
            double t=std::copysign(1./(std::abs(tau)+std::sqrt(1.+tau*tau)),tau);
            double c=1./std::sqrt(1.+t*t),s=t*c;
            cov[p][p]-=t*off;cov[q][q]+=t*off;
            cov[p][q]=cov[q][p]=0.;
            for (int k=0;k<3;k++) {
                if (k!=p && k!=q) {
                    double ap=cov[k][p],aq=cov[k][q];
                    cov[k][p]=cov[p][k]=c*ap-s*aq;
                    cov[k][q]=cov[q][k]=s*ap+c*aq;
                }
                double vp=vectors[k][p],vq=vectors[k][q];
                vectors[k][p]=c*vp-s*vq;vectors[k][q]=s*vp+c*vq;
            }
        }
        std::array<int,3> axes={0,1,2};
        std::sort(axes.begin(),axes.end(),[&](int a,int b) {
            return cov[a][a]>cov[b][b];
        });
        if (!(cov[axes[2]][axes[2]]<.1*cov[axes[1]][axes[1]])) return;
        for (int j=0;j<3;j++)
            for (int k=0;k<3;k++) rotation[j][k]=vectors[k][axes[j]];
        for (Particle& p:particles) {
            double r[3]={p.x[0]-global_c[0],p.x[1]-global_c[1],p.x[2]-global_c[2]};
            for (int j=0;j<3;j++)
                p.x[j]=rotation[j][0]*r[0]+rotation[j][1]*r[1]+rotation[j][2]*r[2];
        }
        global_c[0]=global_c[1]=global_c[2]=0.;
        rotated=true;
    }

    void prepare_target_norms() {
        target_norm.resize(size_t(tree.size())*(horder+1));
        radial_power.resize(size_t(tree.size())*(order+1));
        for (size_t node=0;node<tree.size();node++) {
            const Node& nd=tree[node];
            alignas(32) __m256d sums[MAXH+1],rsums[MAXP/2+1];
            for (int k=0;k<=horder;k++) sums[k]=_mm256_setzero_pd();
            for (int k=0;k<=order/2;k++) rsums[k]=_mm256_setzero_pd();
            __m256d cx=_mm256_set1_pd(nd.c[0]),cy=_mm256_set1_pd(nd.c[1]),
                    cz=_mm256_set1_pd(nd.c[2]);
            double invr2=nd.radius>0.?1./(nd.radius*nd.radius):0.;
            __m256d ir2=_mm256_set1_pd(invr2);
            int i=nd.begin;
            for (;i+3<nd.end;i+=4) {
                __m256d dx=_mm256_loadu_pd(&x[i])-cx;
                __m256d dy=_mm256_loadu_pd(&y[i])-cy;
                __m256d dz=_mm256_loadu_pd(&z[i])-cz;
                __m256d r2=(dx*dx+dy*dy+dz*dz)*ir2;
                r2=_mm256_min_pd(r2,_mm256_set1_pd(1.));
                __m256d wa=_mm256_loadu_pd(&inverse_acc2[i]);
                __m256d mi=_mm256_loadu_pd(&mass[i]);
                __m256d rp=_mm256_set1_pd(1.);
                for (int k=0;k<=horder;k++) {
                    sums[k]+=wa*rp;
                    if (k<=order/2) rsums[k]+=mi*rp;
                    rp*=r2;
                }
            }
            double ws[MAXH+1],qs[MAXP/2+1];
            for (int k=0;k<=horder;k++) ws[k]=hsum(sums[k]);
            for (int k=0;k<=order/2;k++) qs[k]=hsum(rsums[k]);
            for (;i<nd.end;i++) {
                double dx=x[i]-nd.c[0],dy=y[i]-nd.c[1],dz=z[i]-nd.c[2];
                double r2=std::min(1.,(dx*dx+dy*dy+dz*dz)*invr2),rp=1.;
                for (int k=0;k<=horder;k++) {
                    ws[k]+=inverse_acc2[i]*rp;
                    if (k<=order/2) qs[k]+=mass[i]*rp;
                    rp*=r2;
                }
            }
            double* wn=target_norm.data()+node*(horder+1);
            double* pn=radial_power.data()+node*(order+1);
            double radius_power=1.;
            for (int k=0;k<=horder;k++) {
                wn[k]=std::sqrt(ws[k]/(nd.end-nd.begin))*radius_power;
                if (!(k&1) && k<=order) pn[k]=qs[k/2]*radius_power;
                radius_power*=nd.radius;
            }
            for (int k=1;k<order;k+=2) pn[k]=std::sqrt(pn[k-1]*pn[k+1]);
        }
        std::vector<double>().swap(inverse_acc2);
        choose_c[0]=1.;
        for (int k=1;k<=order;k++) choose_c[k]=choose_c[k-1]*(order-k+1)/k;
    }

    void prepare_power() {
        using Complex=std::complex<double>;
        std::vector<std::vector<Complex>> poly((horder+1)*(horder+2)/2);
        auto pi=[](int n,int m) { return n*(n+1)/2+m; };
        poly[0]={Complex(1.,0.)};
        auto each=[&](int degree,auto fn) {
            for (int c=0;c<=std::min(degree,1);c++)
                for (int a=degree-c;a>=0;a--) {
                    int b=degree-c-a;
                    fn(a,b,c,hi(a,b,c)-degree*degree);
                }
        };
        for (int m=0;m<=horder;m++) {
            if (m) {
                auto& dst=poly[pi(m,m)];
                const auto& src=poly[pi(m-1,m-1)];
                dst.resize(2*m+1);
                double f=-std::sqrt(double(2*m-1)/(2*m));
                each(m-1,[&](int a,int b,int c,int j) {
                    dst[hi(a+1,b,c)-m*m]+=f*src[j];
                    dst[hi(a,b+1,c)-m*m]+=Complex(0.,f)*src[j];
                });
            }
            if (m<horder) {
                int n=m+1;
                auto& dst=poly[pi(n,m)];
                const auto& src=poly[pi(m,m)];
                dst.resize(2*n+1);
                double f=std::sqrt(2*m+1.);
                each(m,[&](int a,int b,int c,int j) {
                    if (!c) dst[hi(a,b,1)-n*n]+=f*src[j];
                });
            }
            for (int n=m+2;n<=horder;n++) {
                auto& dst=poly[pi(n,m)];
                const auto& s1=poly[pi(n-1,m)];
                const auto& s2=poly[pi(n-2,m)];
                dst.resize(2*n+1);
                double den=double(n*n-m*m);
                double f1=(2*n-1)/std::sqrt(den);
                double f2=-std::sqrt(((n-1.)*(n-1.)-m*m)/den);
                each(n-1,[&](int a,int b,int c,int j) {
                    if (!c) dst[hi(a,b,1)-n*n]+=f1*s1[j];
                });
                each(n-2,[&](int a,int b,int c,int j) {
                    dst[hi(a+2,b,c)-n*n]+=f2*s2[j];
                    dst[hi(a,b+2,c)-n*n]+=f2*s2[j];
                });
            }
        }
        std::vector<double> gr,gi;
        std::vector<size_t> offsets(horder+1);
        double fac[MAXH+1];fac[0]=1.;
        for (int k=1;k<=horder;k++) fac[k]=fac[k-1]*k;
        for (int n=0;n<=horder;n++) {
            offsets[n]=gr.size();
            for (int m=0;m<=n;m++) {
                const auto& src=poly[pi(n,m)];
                each(n,[&](int a,int b,int c,int j) {
                    Complex v=src[j]*fac[a]*fac[b];
                    gr.push_back(v.real());gi.push_back(v.imag());
                });
            }
        }
        multipole_power.resize(size_t(tree.size())*(horder+1));
        for (size_t i=0;i<tree.size();i++) {
            const double* hm=hmoments.data()+i*hn;
            double* pn=multipole_power.data()+i*(horder+1);
            for (int n=0;n<=horder;n++) {
                int size=2*n+1;
                double sum=0.;
                size_t off=offsets[n];
                const double* v=hm+n*n;
                for (int m=0;m<=n;m++,off+=size) {
                    double real=0.,imag=0.;
                    for (int j=0;j<size;j++) {
                        real+=gr[off+j]*v[j];
                        imag+=gi[off+j]*v[j];
                    }
                    sum+=(m?2.:1.)*(real*real+imag*imag);
                }
                pn[n]=std::sqrt(sum);
            }
        }
        choose_h[0]=1.;
        for (int k=1;k<=horder;k++) choose_h[k]=choose_h[k-1]*(horder-k+1)/k;
    }

    void coarse_translate(int a,int b,double dx,double dy,double dz,double d2) {
        const Node& na=tree[a];
        const Node& nb=tree[b];
        double inv=1./std::sqrt(d2),w=inv*inv*inv;
        double r[3]={dx,dy,dz};
        double* la=locals.data()+size_t(a)*nc;
        double* lb=locals.data()+size_t(b)*nc;
        for (int k=0;k<3;k++) {
            la[1+k]-=nb.mass*w*r[k];
            if (a!=b) lb[1+k]+=na.mass*w*r[k];
        }
        for (int i=0;i<3;i++) {
            for (int j=i;j<3;j++) {
                double h=w*(3.*r[i]*r[j]/d2-(i==j?1.:0.));
                int e[3]={0,0,0};
                e[i]++;e[j]++;
                int k=tab.index[e[0]][e[1]][e[2]];
                la[k]+=nb.mass*h;
                if (a!=b) lb[k]+=na.mass*h;
            }
        }
    }

    void coarse_interact(int a,int b) {
        const Node& na=tree[a];
        const Node& nb=tree[b];
        double dx=na.c[0]-nb.c[0],dy=na.c[1]-nb.c[1],dz=na.c[2]-nb.c[2];
        double r=coarse_max?std::max(na.radius,nb.radius):na.radius+nb.radius;
        double d2=dx*dx+dy*dy+dz*dz+EPS2;
        if (r*r<coarse_theta2*d2) {
            coarse_translate(a,b,dx,dy,dz,d2);
        } else if (na.left<0 && nb.left<0) {
            direct(a,b);
        } else if (a==b) {
            coarse_interact(na.left,na.left);
            coarse_interact(na.left,na.right);
            coarse_interact(na.right,na.right);
        } else if (nb.left<0 || (na.left>=0 && na.radius>=nb.radius)) {
            coarse_interact(na.left,b);
            coarse_interact(na.right,b);
        } else {
            coarse_interact(a,nb.left);
            coarse_interact(a,nb.right);
        }
    }

    void coarse_down(int node) {
        Node& nd=tree[node];
        const double* l=locals.data()+size_t(node)*nc;
        double h[3][3];
        for (int i=0;i<3;i++) {
            for (int j=i;j<3;j++) {
                int e[3]={0,0,0};e[i]++;e[j]++;
                h[i][j]=h[j][i]=l[tab.index[e[0]][e[1]][e[2]]];
            }
        }
        if (nd.left<0) {
            double sum=0.;
            for (int i=nd.begin;i<nd.end;i++) {
                double r[3]={x[i]-nd.c[0],y[i]-nd.c[1],z[i]-nd.c[2]};
                double v[3]={ax[i]+l[1],ay[i]+l[2],az[i]+l[3]};
                for (int k=0;k<3;k++)
                    for (int j=0;j<3;j++) v[k]+=h[k][j]*r[j];
                double a2=v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
                sum+=1./std::max(a2,1e-60);
                if (target_mac) inverse_acc2[i]=1./std::max(a2,1e-60);
            }
            nd.acc_scale=std::sqrt((nd.end-nd.begin)/sum);
        } else {
            for (int ch : {nd.left,nd.right}) {
                double* cl=locals.data()+size_t(ch)*nc;
                double r[3];
                for (int k=0;k<3;k++) r[k]=tree[ch].c[k]-nd.c[k];
                for (int k=0;k<3;k++) {
                    double v=l[1+k];
                    for (int j=0;j<3;j++) v+=h[k][j]*r[j];
                    cl[1+k]+=v;
                }
                for (int j=4;j<10;j++) cl[j]+=l[j];
                coarse_down(ch);
            }
            double na=tree[nd.left].end-tree[nd.left].begin;
            double nb=tree[nd.right].end-tree[nd.right].begin;
            double aa=tree[nd.left].acc_scale,ab=tree[nd.right].acc_scale;
            nd.acc_scale=std::sqrt((na+nb)/(na/(aa*aa)+nb/(ab*ab)));
        }
    }

    void powers(double v, double* p) const {
        p[0]=1.;
        for (int i=1;i<=order;i++) p[i]=p[i-1]*v*RECIP[i];
    }

    void shift_moment(double* v, const double* shift) const {
        double ps[MAXP+1];
        for (int dim=0;dim<3;dim++) {
            powers(shift[dim],ps);
            for (int j=nm-1;j>=1;j--) {
                int a=tab.ex[j][0],b=tab.ex[j][1],c=tab.ex[j][2];
                int e=tab.ex[j][dim];
                double s=v[j];
                for (int k=1;k<=e;k++) {
                    int aa=a-(dim==0?k:0);
                    int bb=b-(dim==1?k:0);
                    int cc=c-(dim==2?k:0);
                    s+=v[tab.index[aa][bb][cc]]*ps[k];
                }
                v[j]=s;
            }
        }
    }

    static int hi(int a,int b,int c) {
        int n=a+b+c;
        return n*n+(c ? 2*n-a : n-a);
    }

    static void table_shift(double* v, const double* shift, const Table& tb, int count,
                            bool local) {
        double ps[MAXP+1];
        for (int dim=0;dim<3;dim++) {
            ps[0]=1.;
            for (int j=1;j<=tb.p;j++) ps[j]=ps[j-1]*shift[dim]*RECIP[j];
            for (int q=local?1:count-1; local ? q<count : q>=1; local?q++:q--) {
                int a=tb.ex[q][0],b=tb.ex[q][1],c=tb.ex[q][2];
                int maxk=local ? tb.p-tb.degree[q] : tb.ex[q][dim];
                double s=v[q];
                for (int k=1;k<=maxk;k++) {
                    int off=local?k:-k;
                    s+=v[tb.index[a+(dim==0?off:0)][b+(dim==1?off:0)]
                                   [c+(dim==2?off:0)]]*ps[k];
                }
                v[q]=s;
            }
        }
    }

    void harmonic_basis(double dx,double dy,double dz,double* v,int p) const {
        v[0]=1.;
        for (int n=1;n<=p;n++) {
            double inv=RECIP[n];
            for (int a=n;a>=0;a--) {
                int b=n-a;
                double s=0.;
                if (a) s+=dx*v[hi(a-1,b,0)];
                if (b) s+=dy*v[hi(a,b-1,0)];
                if (a>=2) s-=dz*v[hi(a-2,b,1)];
                if (b>=2) s-=dz*v[hi(a,b-2,1)];
                v[hi(a,b,0)]=s*inv;
            }
            for (int a=n-1;a>=0;a--) {
                int b=n-1-a;
                double s=dz*v[hi(a,b,0)];
                if (a) s+=dx*v[hi(a-1,b,1)];
                if (b) s+=dy*v[hi(a,b-1,1)];
                v[hi(a,b,1)]=s*inv;
            }
        }
    }

    void harmonic_basis_vec(__m256d dx,__m256d dy,__m256d dz,__m256d* v,int p) const {
        v[0]=_mm256_set1_pd(1.);
        for (int n=1;n<=p;n++) {
            __m256d inv=_mm256_set1_pd(RECIP[n]);
            for (int a=n;a>=0;a--) {
                int b=n-a;
                __m256d s={};
                if (a) s+=dx*v[hi(a-1,b,0)];
                if (b) s+=dy*v[hi(a,b-1,0)];
                if (a>=2) s-=dz*v[hi(a-2,b,1)];
                if (b>=2) s-=dz*v[hi(a,b-2,1)];
                v[hi(a,b,0)]=s*inv;
            }
            for (int a=n-1;a>=0;a--) {
                int b=n-1-a;
                __m256d s=dz*v[hi(a,b,0)];
                if (a) s+=dx*v[hi(a-1,b,1)];
                if (b) s+=dy*v[hi(a,b-1,1)];
                v[hi(a,b,1)]=s*inv;
            }
        }
    }

    void powers_vec(__m256d x,__m256d* v) const {
        v[0]=_mm256_set1_pd(1.);
        for (int k=1;k<=order;k++) v[k]=v[k-1]*x*_mm256_set1_pd(RECIP[k]);
    }

    void shift_harmonic(double* v,const double* shift,bool local) const {
        double ps[MAXH+1];
        for (int dim=0;dim<2;dim++) {
            ps[0]=1.;
            for (int k=1;k<=horder;k++) ps[k]=ps[k-1]*shift[dim]*RECIP[k];
            for (int n=local?1:horder;local?n<=horder:n>=1;local?n++:n--) {
                for (int c=0;c<=1;c++)
                    for (int a=n-c;a>=0;a--) {
                        int b=n-c-a,j=hi(a,b,c);
                        int maxk=local?horder-n:(dim?b:a);
                        double s=v[j];
                        for (int k=1;k<=maxk;k++) {
                            int offset=local?k:-k;
                            s+=v[hi(a+(dim==0?offset:0),b+(dim==1?offset:0),c)]*ps[k];
                        }
                        v[j]=s;
                    }
            }
        }
        double tmp[MAXHC];
        std::memcpy(tmp,v,hn*sizeof(double));
        for (int k=1;k<=horder;k++) {
            double factor=shift[2]*RECIP[k];
            for (int n=local?0:horder;local?n<=horder:n>=0;local?n++:n--) {
                for (int c=0;c<=std::min(n,1);c++) {
                    for (int a=n-c;a>=0;a--) {
                        int b=n-c-a,j=hi(a,b,c);
                        double s=0.;
                        if (local) {
                            if (n<horder) {
                                if (!c) s=tmp[hi(a,b,1)];
                                else s=-tmp[hi(a+2,b,0)]-tmp[hi(a,b+2,0)];
                            }
                        } else {
                            if (c) s=tmp[hi(a,b,0)];
                            else {
                                if (a>=2) s-=tmp[hi(a-2,b,1)];
                                if (b>=2) s-=tmp[hi(a,b-2,1)];
                            }
                        }
                        tmp[j]=factor*s;
                    }
                }
            }
            for (int j=local?1:0;j<hn;j++) v[j]+=tmp[j];
        }
    }

    __attribute__((noinline))
    void leaf_up_harmonic(int node) {
        const Node& nd=tree[node];
        double* m=moments.data()+size_t(node)*nm;
        double* hm=hmoments.data()+size_t(node)*hn;
        int i=nd.begin;
        if (i+3<nd.end) {
            alignas(32) __m256d hb[MAXHC],mh[MAXHC],ms[MAXC];
            alignas(32) __m256d px[MAXP+1],py[MAXP+1],pz[MAXP+1];
            for (int j=0;j<hn;j++) mh[j]=_mm256_setzero_pd();
            for (int j=0;j<nm;j++) ms[j]=_mm256_setzero_pd();
            for (;i+3<nd.end;i+=4) {
                __m256d dx=_mm256_loadu_pd(&x[i])-_mm256_set1_pd(nd.c[0]);
                __m256d dy=_mm256_loadu_pd(&y[i])-_mm256_set1_pd(nd.c[1]);
                __m256d dz=_mm256_loadu_pd(&z[i])-_mm256_set1_pd(nd.c[2]);
                __m256d mi=_mm256_loadu_pd(&mass[i]);
                powers_vec(dx,px); powers_vec(dy,py); powers_vec(dz,pz);
                harmonic_basis_vec(dx,dy,dz,hb,horder);
                for (int j=0;j<nm;j++)
                    ms[j]+=mi*px[tab.ex[j][0]]*py[tab.ex[j][1]]*pz[tab.ex[j][2]];
                for (int j=0;j<hn;j++) mh[j]+=mi*hb[j];
            }
            for (int j=0;j<nm;j++) m[j]=hsum(ms[j]);
            for (int j=0;j<hn;j++) hm[j]=hsum(mh[j]);
        }
        {
            double px[MAXP+1],py[MAXP+1],pz[MAXP+1];
            double hb[MAXHC];
            for (;i<nd.end;i++) {
                double dx=x[i]-nd.c[0],dy=y[i]-nd.c[1],dz=z[i]-nd.c[2];
                powers(dx,px); powers(dy,py); powers(dz,pz);
                harmonic_basis(dx,dy,dz,hb,horder);
                double mi=mass[i];
                for (int j=0;j<nm;j++)
                    m[j]+=mi*px[tab.ex[j][0]]*py[tab.ex[j][1]]*pz[tab.ex[j][2]];
                for (int j=0;j<hn;j++) hm[j]+=mi*hb[j];
            }
        }
    }

    void upward_harmonic(int node) {
        const Node& nd=tree[node];
        double* m=moments.data()+size_t(node)*nm;
        double* hm=hmoments.data()+size_t(node)*hn;
        if (nd.left<0) {
            leaf_up_harmonic(node);
        } else {
            double buf[MAXC],hbuf[MAXHC],shift[3];
            for (int ch : {nd.left,nd.right}) {
                upward_harmonic(ch);
                for (int k=0;k<3;k++) shift[k]=tree[ch].c[k]-nd.c[k];
                std::memcpy(buf,moments.data()+size_t(ch)*nm,nm*sizeof(double));
                shift_moment(buf,shift);
                for (int j=0;j<nm;j++) m[j]+=buf[j];
                std::memcpy(hbuf,hmoments.data()+size_t(ch)*hn,hn*sizeof(double));
                shift_harmonic(hbuf,shift,false);
                for (int j=0;j<hn;j++) hm[j]+=hbuf[j];
            }
        }
        m[0]=hm[0]=nd.mass;
        m[1]=m[2]=m[3]=hm[1]=hm[2]=hm[3]=0.;
    }

    void upward(int node) {
        const Node& nd=tree[node];
        double* m=moments.data()+size_t(node)*nm;
        if (nd.left<0) {
            double px[MAXP+1],py[MAXP+1],pz[MAXP+1];
            for (int i=nd.begin;i<nd.end;i++) {
                powers(x[i]-nd.c[0],px);
                powers(y[i]-nd.c[1],py);
                powers(z[i]-nd.c[2],pz);
                double mi=mass[i];
                for (int j=0;j<nm;j++)
                    m[j]+=mi*px[tab.ex[j][0]]*py[tab.ex[j][1]]*pz[tab.ex[j][2]];
            }
        } else {
            double buf[MAXC],shift[3];
            for (int ch : {nd.left,nd.right}) {
                upward(ch);
                std::memcpy(buf,moments.data()+size_t(ch)*nm,nm*sizeof(double));
                for (int k=0;k<3;k++) shift[k]=tree[ch].c[k]-nd.c[k];
                shift_moment(buf,shift);
                for (int j=0;j<nm;j++) m[j]+=buf[j];
            }
        }
        m[0]=nd.mass;
        m[1]=m[2]=m[3]=0.;
    }

    static const Table& get_table(int p) {
        static const Table t4(4),t6(6),t8(8),t10(10),t12(12);
        switch(p) {
            case 4:return t4;
            case 6:return t6;
            case 8:return t8;
            case 10:return t10;
            default:return t12;
        }
    }

    void derivatives(__m256d dx, __m256d dy, __m256d dz, __m256d* d,
                     double epsilon2=EPS2,int p=0) const {
        const Table& tb=p?get_table(p):tab;
        __m256d t[MAXC];
        __m256d one=_mm256_set1_pd(1.);
        __m256d s=dx*dx+dy*dy+dz*dz+_mm256_set1_pd(epsilon2);
        __m256d inv=one/s;
        t[0]=one/_mm256_sqrt_pd(s);
        d[0]=t[0];
        const __m256d r[3]={dx,dy,dz};
        int j=1;
        for (int degree=1;degree<=tb.p;degree++) {
            __m256d f1=_mm256_set1_pd(2.-RECIP[degree]);
            __m256d f2=_mm256_set1_pd(1.-RECIP[degree]);
            int end=(degree+1)*(degree+2)*(degree+3)/6;
            for (;j<end;j++) {
                __m256d a={},b={};
                for (int k=0;k<3;k++) {
                    int q=tb.prev1[j][k];
                    if (q>=0) a+=r[k]*t[q];
                    q=tb.prev2[j][k];
                    if (q>=0) b+=t[q];
                }
                t[j]=-inv*(f1*a+f2*b);
                d[j]=t[j]*_mm256_set1_pd(tb.factorial[j]);
            }
        }
    }

    template<bool Z2>
    __attribute__((always_inline))
    static inline void harmonic_row(int length,const __m256d* ma,const __m256d* mb,
                                     const __m256d* d,
                                     __m256d& u0,__m256d& u1,__m256d& u2,__m256d& u3,
                                     __m256d& v0,__m256d& v1,__m256d& v2,__m256d& v3) {
        int i=0;
        for (;i+3<length;i+=4) {
            __m256d d0=Z2 ? -d[i]-d[i+2] : d[i];
            __m256d d1=Z2 ? -d[i+1]-d[i+3] : d[i+1];
            __m256d d2=Z2 ? -d[i+2]-d[i+4] : d[i+2];
            __m256d d3=Z2 ? -d[i+3]-d[i+5] : d[i+3];
            u0+=mb[i]*d0; v0+=ma[i]*d0;
            u1+=mb[i+1]*d1; v1+=ma[i+1]*d1;
            u2+=mb[i+2]*d2; v2+=ma[i+2]*d2;
            u3+=mb[i+3]*d3; v3+=ma[i+3]*d3;
        }
        if (i<length) {
            __m256d dd=Z2 ? -d[i]-d[i+2] : d[i];
            u0+=mb[i]*dd; v0+=ma[i]*dd; i++;
        }
        if (i<length) {
            __m256d dd=Z2 ? -d[i]-d[i+2] : d[i];
            u1+=mb[i]*dd; v1+=ma[i]*dd; i++;
        }
        if (i<length) {
            __m256d dd=Z2 ? -d[i]-d[i+2] : d[i];
            u2+=mb[i]*dd; v2+=ma[i]*dd;
        }
    }

    void cartesian_contract(const __m256d* ma,const __m256d* mb,
                            __m256d* la,__m256d* lb,const __m256d* d,int p=0) const {
        const Table& tb=p?get_table(p):tab;
        for (int j=1;j<tb.count;j++) {
            __m256d u0={},u1={},u2={},u3={},v0={},v1={},v2={},v3={};
            for (int row=tb.row_begin[j];row<tb.row_begin[j+1];row++) {
                const Table::Row& rr=tb.rows[row];
                harmonic_row<false>(rr.length,ma+rr.source,mb+rr.source,d+rr.derivative,
                                    u0,u1,u2,u3,v0,v1,v2,v3);
            }
            la[j]+=u0+u1+u2+u3;
            __m256d val=v0+v1+v2+v3;
            if (tb.degree[j]&1) lb[j]-=val;
            else lb[j]+=val;
        }
    }

    void harmonic_derivatives(__m256d dx,__m256d dy,__m256d dz,
                              __m256d* d,int derivative_order) const {
        constexpr int HD=(2*MAXH+1)*(2*MAXH+1);
        alignas(32) __m256d t[HD];
        __m256d inv=_mm256_set1_pd(1.)/(dx*dx+dy*dy+dz*dz);
        t[0]=_mm256_sqrt_pd(inv);
        d[0]=t[0];
        double fac[2*MAXH+1];
        fac[0]=1.;
        for (int k=1;k<=derivative_order;k++) fac[k]=fac[k-1]*k;
        for (int n=1;n<=derivative_order;n++) {
            __m256d f1=_mm256_set1_pd(2.-RECIP[n]);
            __m256d f2=_mm256_set1_pd(1.-RECIP[n]);
            for (int c=0;c<=1;c++) {
                for (int a=n-c;a>=0;a--) {
                    int b=n-c-a,j=hi(a,b,c);
                    __m256d s1={},s2={};
                    if (a) s1+=dx*t[hi(a-1,b,c)];
                    if (b) s1+=dy*t[hi(a,b-1,c)];
                    if (c) s1+=dz*t[hi(a,b,0)];
                    if (a>=2) s2+=t[hi(a-2,b,c)];
                    if (b>=2) s2+=t[hi(a,b-2,c)];
                    t[j]=-inv*(f1*s1+f2*s2);
                    d[j]=t[j]*_mm256_set1_pd(fac[a]*fac[b]);
                }
            }
        }
    }

    void correction_derivatives(__m256d dx,__m256d dy,__m256d dz,__m256d* d,
                                 const __m256d* dh,int pc) const {
        const Table& tb=get_table(pc);
        alignas(32) __m256d t[MAXC],dn[MAXC];
        __m256d s=dx*dx+dy*dy+dz*dz+_mm256_set1_pd(EPS2);
        __m256d inv=_mm256_set1_pd(1.)/s;
        __m256d r0=_mm256_set1_pd(1.)/dh[0],rs=_mm256_sqrt_pd(s);
        dn[0]=dh[0];
        t[0]=-_mm256_set1_pd(EPS2)/(r0*rs*(r0+rs));
        d[0]=t[0];
        const __m256d r[3]={dx,dy,dz};
        int j=1;
        for (int degree=1;degree<=pc;degree++) {
            __m256d f1=_mm256_set1_pd(2.-RECIP[degree]);
            __m256d f2=_mm256_set1_pd(1.-RECIP[degree]);
            int end=(degree+1)*(degree+2)*(degree+3)/6;
            for (;j<end;j++) {
                int a=tb.ex[j][0],b=tb.ex[j][1],c=tb.ex[j][2];
                if (c<2) dn[j]=dh[hi(a,b,c)];
                else dn[j]=-dn[tb.index[a+2][b][c-2]]-dn[tb.index[a][b+2][c-2]];
                __m256d s1={},s2={};
                for (int k=0;k<3;k++) {
                    int q=tb.prev1[j][k];
                    if (q>=0) s1+=r[k]*t[q];
                    q=tb.prev2[j][k];
                    if (q>=0) s2+=t[q];
                }
                t[j]=-inv*(f1*s1+f2*s2+dn[j]*_mm256_set1_pd(EPS2*tb.inv_factorial[j]));
                d[j]=t[j]*_mm256_set1_pd(tb.factorial[j]);
            }
        }
    }

    void harmonic_contract(const __m256d* ma,const __m256d* mb,
                           __m256d* la,__m256d* lb,
                           const __m256d* d,int hp) const {
        for (int n=1;n<=hp;n++) {
            for (int c=0;c<=1;c++) {
                for (int a=n-c;a>=0;a--) {
                    int b=n-c-a,j=hi(a,b,c);
                    __m256d u0={},u1={},u2={},u3={},v0={},v1={},v2={},v3={};
                    for (int m=0;m<=(full_h?hp:hp-n);m++) {
                        if (m==1) continue;
                        int off=m*m;
                        int dd=hi(a+m,b,c);
                        harmonic_row<false>(m+1,ma+off,mb+off,d+dd,
                                            u0,u1,u2,u3,v0,v1,v2,v3);
                        if (m) {
                            off+=m+1;
                            if (c) {
                                dd=hi(a+m+1,b,0);
                                harmonic_row<true>(m,ma+off,mb+off,d+dd,
                                                   u0,u1,u2,u3,v0,v1,v2,v3);
                            } else {
                                dd=hi(a+m-1,b,1);
                                harmonic_row<false>(m,ma+off,mb+off,d+dd,
                                                    u0,u1,u2,u3,v0,v1,v2,v3);
                            }
                        }
                    }
                    la[j]+=u0+u1+u2+u3;
                    __m256d val=v0+v1+v2+v3;
                    if (n&1) lb[j]-=val;
                    else lb[j]+=val;
                }
            }
        }
    }

    void flush_translations() {
        if (!batch_count) return;
        double t0=calibrating?cpu_time():0.;
        int actual_count=batch_count;
        for (int i=batch_count;i<4;i++) batch[i]=batch[0];
        int pc=batch_soft?batch_soft:order;
        int hp=batch_horder;
        int hcount=(hp+1)*(hp+1);
        const Table& ct=get_table(pc);
        int cnc=ct.count,cnm=ct.moments;
        const double *ma[4],*mb[4];
        double zero[MAXC]={0.};
        for (int i=0;i<4;i++) {
            ma[i]=moments.data()+size_t(batch[i].a)*nm;
            mb[i]=batch[i].a==batch[i].b ? zero :
                  moments.data()+size_t(batch[i].b)*nm;
        }
        constexpr int HD=(2*MAXH+1)*(2*MAXH+1);
        alignas(32) __m256d va[MAXVEC],vb[MAXVEC],la[MAXVEC],lb[MAXVEC],d[MAXC],dh[HD];
        for (int j=0;j<cnc;j++) la[j]=lb[j]=_mm256_setzero_pd();
        for (int j=0;j<cnm;j++) {
            va[j]=_mm256_set_pd(ma[3][j],ma[2][j],ma[1][j],ma[0][j]);
            vb[j]=_mm256_set_pd(mb[3][j],mb[2][j],mb[1][j],mb[0][j]);
        }
        __m256d dx=_mm256_set_pd(batch[3].dx,batch[2].dx,batch[1].dx,batch[0].dx);
        __m256d dy=_mm256_set_pd(batch[3].dy,batch[2].dy,batch[1].dy,batch[0].dy);
        __m256d dz=_mm256_set_pd(batch[3].dz,batch[2].dz,batch[1].dz,batch[0].dz);
        if (hp>0) {
            harmonic_derivatives(dx,dy,dz,dh,std::max(pc,full_h?2*hp:hp));
            if (newton_only) {
                for (int j=0;j<cnc;j++) d[j]=_mm256_setzero_pd();
            } else correction_derivatives(dx,dy,dz,d,dh,pc);
        } else {
            derivatives(dx,dy,dz,d,newton_only?0.:EPS2,pc);
        }
        if (compact) {
            for (int j=0;j<cnm;j++) if (ct.degree[j]&1) vb[j]=-vb[j];
            cartesian_contract(va,vb,la,lb,d,pc);
        }
        for (int j=1;j<cnc;j++) {
            alignas(32) double ta[4],tb[4];
            _mm256_store_pd(ta,la[j]);
            _mm256_store_pd(tb,lb[j]);
            for (int k=0;k<batch_count;k++) {
                locals[size_t(batch[k].a)*nc+j]+=ta[k];
                locals[size_t(batch[k].b)*nc+j]+=tb[k];
            }
        }
        if (hp>0) {
            for (int i=0;i<4;i++) {
                ma[i]=hmoments.data()+size_t(batch[i].a)*hn;
                mb[i]=hmoments.data()+size_t(batch[i].b)*hn;
            }
            for (int j=0;j<hcount;j++) {
                va[j]=_mm256_set_pd(ma[3][j],ma[2][j],ma[1][j],ma[0][j]);
                vb[j]=_mm256_set_pd(mb[3][j],mb[2][j],mb[1][j],mb[0][j]);
                la[j]=lb[j]=_mm256_setzero_pd();
            }
            for (int degree=1;degree<=hp;degree+=2)
                for (int j=degree*degree;j<(degree+1)*(degree+1);j++) vb[j]=-vb[j];
            harmonic_contract(va,vb,la,lb,dh,hp);
            for (int j=1;j<hcount;j++) {
                alignas(32) double ta[4],tb[4];
                _mm256_store_pd(ta,la[j]);
                _mm256_store_pd(tb,lb[j]);
                for (int k=0;k<batch_count;k++) {
                    hlocals[size_t(batch[k].a)*hn+j]+=ta[k];
                    hlocals[size_t(batch[k].b)*hn+j]+=tb[k];
                }
            }
        }
        if (calibrating) {
            bin_time[batch_bin]+=cpu_time()-t0;
            bin_calls[batch_bin]+=actual_count;
        }
        batch_count=0;
    }

    int interaction_bin(int a,int b,double dx,double dy,double dz) const {
        int pc=order,hp=horder;
        double d2=dx*dx+dy*dy+dz*dz;
        if (near_soft && relative_mac && d2<16.*EPS2) {
            pc=order;
            if (adaptive_c)
                for (int p=4;p<order;p+=2)
                    if (full_soft_error(a,b,d2,p)<force_tol*adaptive_safety) {
                        pc=p;break;
                    }
            return pc/2;
        }
        if (variable_soft && horder) {
            double dist=std::sqrt(d2);
            pc=dist>3.?4:dist>1.?6:dist>.3?8:dist>.1?10:12;
            pc=std::min(pc,order);
        }
        if (variable_h>0. && horder) {
            double q=(tree[a].radius+tree[b].radius)/std::sqrt(d2);
            q=std::min(.95,std::max(.001,q*.7));
            int wanted=2*int(std::ceil(std::log(variable_h)/std::log(q)*.5));
            hp=std::min(horder,std::max(6,wanted));
        }
        if (adaptive_h && power_mac && horder) {
            for (int p=6;p<horder;p+=2)
                if (harmonic_error(a,b,d2,p)<force_tol*adaptive_safety) {
                    hp=p;break;
                }
        }
        if (adaptive_c && relative_mac && horder) {
            for (int p=4;p<order;p+=2)
                if (soft_error(a,b,d2,p)<force_tol*adaptive_safety) {
                    pc=p;break;
                }
        }
        return (hp/2)*7+pc/2;
    }

    void flush_pending() {
        for (int key=0;key<BINS;key++) {
            Bucket& bucket=pending[key];
            if (!bucket.count) continue;
            batch_count=bucket.count;
            std::memcpy(batch,bucket.data,sizeof(Interaction)*batch_count);
            batch_horder=(key/7)*2;
            batch_soft=(key%7)*2;
            batch_bin=key;
            flush_translations();
            bucket.count=0;
        }
    }

    void queue_translation(int key,int a,int b,double dx,double dy,double dz) {
        Bucket& bucket=pending[key];
        bucket.data[bucket.count++]={a,b,dx,dy,dz};
        translations++;
        selected_horders[(key/7)*2]++;
        selected_corders[(key%7)*2]++;
        if (tracking_work) processed_work+=bin_cost[key]+walk_model;
        if (bucket.count==4) {
            batch_count=4;
            std::memcpy(batch,bucket.data,sizeof(batch));
            batch_horder=(key/7)*2;
            batch_soft=(key%7)*2;
            batch_bin=key;
            flush_translations();
            bucket.count=0;
        }
    }

    void translate(int a, int b, double dx, double dy, double dz) {
        queue_translation(interaction_bin(a,b,dx,dy,dz),a,b,dx,dy,dz);
    }

    static inline __m256d reciprocal_root(__m256d r2) {
        __m128 f=_mm256_cvtpd_ps(r2);
        __m256d v=_mm256_cvtps_pd(_mm_rsqrt_ps(f));
        __m256d half=_mm256_set1_pd(0.5),three=_mm256_set1_pd(3.);
        v=_mm256_mul_pd(_mm256_mul_pd(half,v),
                       _mm256_fnmadd_pd(r2,_mm256_mul_pd(v,v),three));
        v=_mm256_mul_pd(_mm256_mul_pd(half,v),
                       _mm256_fnmadd_pd(r2,_mm256_mul_pd(v,v),three));
        return v;
    }

    __attribute__((always_inline))
    static inline void reciprocal_roots2(__m256d ra,__m256d rb,
                                         __m256d& a,__m256d& b) {
        a=_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(ra)));
        b=_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(rb)));
        __m256d half=_mm256_set1_pd(.5),three=_mm256_set1_pd(3.);
        __m256d ca=_mm256_fnmadd_pd(ra,a*a,three);
        __m256d cb=_mm256_fnmadd_pd(rb,b*b,three);
        a=(half*a)*ca;b=(half*b)*cb;
        ca=_mm256_fnmadd_pd(ra,a*a,three);
        cb=_mm256_fnmadd_pd(rb,b*b,three);
        a=(half*a)*ca;b=(half*b)*cb;
    }

    static inline double hsum(__m256d v) {
        __m128d s=_mm_add_pd(_mm256_castpd256_pd128(v),_mm256_extractf128_pd(v,1));
        s=_mm_add_sd(s,_mm_unpackhi_pd(s,s));
        return _mm_cvtsd_f64(s);
    }

    void direct(int a, int b) {
        if (tree[a].end-tree[a].begin>tree[b].end-tree[b].begin) std::swap(a,b);
        const Node& na=tree[a];
        const Node& nb=tree[b];
        const double* __restrict xx=x.data();
        const double* __restrict yy=y.data();
        const double* __restrict zz=z.data();
        const double* __restrict mm=mass.data();
        double* __restrict xa=ax.data();
        double* __restrict ya=ay.data();
        double* __restrict za=az.data();
        bool same=a==b;
        uint64_t pairs=same ? uint64_t(na.end-na.begin)*(na.end-na.begin-1)/2 :
                             uint64_t(na.end-na.begin)*(nb.end-nb.begin);
        direct_pairs+=pairs;
        if (tracking_work) processed_work+=pairs*pair_model+walk_model;
        for (int i=na.begin;i<na.end;i++) {
            __m256d xi=_mm256_set1_pd(xx[i]),yi=_mm256_set1_pd(yy[i]),zi=_mm256_set1_pd(zz[i]);
            __m256d mi=_mm256_set1_pd(mm[i]);
            __m256d sx=_mm256_setzero_pd(),sy=sx,sz=sx;
            int j=same ? i+1 : nb.begin;
            if (pipelined_direct) {
                for (;j+7<nb.end;j+=8) {
                    __m256d ra,rb;
                    {
                        __m256d dx=_mm256_loadu_pd(xx+j)-xi;
                        __m256d dy=_mm256_loadu_pd(yy+j)-yi;
                        __m256d dz=_mm256_loadu_pd(zz+j)-zi;
                        ra=_mm256_fmadd_pd(dx,dx,_mm256_set1_pd(newton_only?0.:EPS2));
                        ra=_mm256_fmadd_pd(dy,dy,ra);
                        ra=_mm256_fmadd_pd(dz,dz,ra);
                    }
                    {
                        __m256d dx=_mm256_loadu_pd(xx+j+4)-xi;
                        __m256d dy=_mm256_loadu_pd(yy+j+4)-yi;
                        __m256d dz=_mm256_loadu_pd(zz+j+4)-zi;
                        rb=_mm256_fmadd_pd(dx,dx,_mm256_set1_pd(newton_only?0.:EPS2));
                        rb=_mm256_fmadd_pd(dy,dy,rb);
                        rb=_mm256_fmadd_pd(dz,dz,rb);
                    }
                    __m256d ia,ib;
                    reciprocal_roots2(ra,rb,ia,ib);
                    __m256d wa=ia*(ia*ia),wb=ib*(ib*ib);
                    // Reload displacements after the root pipeline to limit live vectors.
                    asm volatile("" ::: "memory");
#pragma GCC unroll 2
                    for (int k=0;k<2;k++) {
                        int q=j+4*k;
                        __m256d dx=_mm256_loadu_pd(xx+q)-xi;
                        __m256d dy=_mm256_loadu_pd(yy+q)-yi;
                        __m256d dz=_mm256_loadu_pd(zz+q)-zi;
                        __m256d w=k?wb:wa;
                        __m256d wi=w*mi,wj=w*_mm256_loadu_pd(mm+q);
                        sx=_mm256_fmadd_pd(wj,dx,sx);
                        sy=_mm256_fmadd_pd(wj,dy,sy);
                        sz=_mm256_fmadd_pd(wj,dz,sz);
                        _mm256_storeu_pd(xa+q,_mm256_fnmadd_pd(wi,dx,_mm256_loadu_pd(xa+q)));
                        _mm256_storeu_pd(ya+q,_mm256_fnmadd_pd(wi,dy,_mm256_loadu_pd(ya+q)));
                        _mm256_storeu_pd(za+q,_mm256_fnmadd_pd(wi,dz,_mm256_loadu_pd(za+q)));
                    }
                }
            }
#pragma GCC unroll 4
            for (;j+3<nb.end;j+=4) {
                __m256d dx=_mm256_sub_pd(_mm256_loadu_pd(xx+j),xi);
                __m256d dy=_mm256_sub_pd(_mm256_loadu_pd(yy+j),yi);
                __m256d dz=_mm256_sub_pd(_mm256_loadu_pd(zz+j),zi);
                __m256d r2=_mm256_fmadd_pd(dx,dx,_mm256_set1_pd(newton_only?0.:EPS2));
                r2=_mm256_fmadd_pd(dy,dy,r2);
                r2=_mm256_fmadd_pd(dz,dz,r2);
                __m256d ir=reciprocal_root(r2);
                __m256d ir3=_mm256_mul_pd(ir,_mm256_mul_pd(ir,ir));
                __m256d wi=_mm256_mul_pd(ir3,mi);
                __m256d wj=_mm256_mul_pd(ir3,_mm256_loadu_pd(mm+j));
                sx=_mm256_fmadd_pd(wj,dx,sx);
                sy=_mm256_fmadd_pd(wj,dy,sy);
                sz=_mm256_fmadd_pd(wj,dz,sz);
                _mm256_storeu_pd(xa+j,_mm256_fnmadd_pd(wi,dx,_mm256_loadu_pd(xa+j)));
                _mm256_storeu_pd(ya+j,_mm256_fnmadd_pd(wi,dy,_mm256_loadu_pd(ya+j)));
                _mm256_storeu_pd(za+j,_mm256_fnmadd_pd(wi,dz,_mm256_loadu_pd(za+j)));
            }
            double tx=hsum(sx),ty=hsum(sy),tz=hsum(sz);
            for (;j<nb.end;j++) {
                double dx=xx[j]-xx[i],dy=yy[j]-yy[i],dz=zz[j]-zz[i];
                double r2=dx*dx+dy*dy+dz*dz+(newton_only?0.:EPS2);
                double ir=1./std::sqrt(r2);
                double w=ir*ir*ir;
                tx+=mm[j]*w*dx; ty+=mm[j]*w*dy; tz+=mm[j]*w*dz;
                xa[j]-=mm[i]*w*dx; ya[j]-=mm[i]*w*dy; za[j]-=mm[i]*w*dz;
            }
            xa[i]+=tx; ya[i]+=ty; za[i]+=tz;
        }
    }

    void count_interactions(int a,int b,Stats& stats,bool sample) const {
        const Node& na=tree[a];
        const Node& nb=tree[b];
        double dx=na.c[0]-nb.c[0],dy=na.c[1]-nb.c[1],dz=na.c[2]-nb.c[2];
        double r=max_mac ? std::max(na.radius,nb.radius) : na.radius+nb.radius;
        double d2=dx*dx+dy*dy+dz*dz+(horder?0.:EPS2);
        bool small_pair=uint64_t(na.end-na.begin)*(nb.end-nb.begin)<=direct_cutoff;
        if (!small_pair && acceptable(a,b,dx,dy,dz,r,d2)) {
            stats.m2l++;
            int bin=interaction_bin(a,b,dx,dy,dz);
            stats.bins[bin]++;
            if (sample) {
                auto& samples=stats.far_bins[bin];
                if (samples.size()<256) samples.push_back({a,b,dx,dy,dz});
                else {
                    uint64_t hash=(uint64_t(a)*0x94d049bb133111ebULL)^
                                  (uint64_t(b)*0x9e3779b97f4a7c15ULL);
                    hash^=hash>>30;hash*=0xbf58476d1ce4e5b9ULL;
                    hash^=hash>>27;hash*=0x94d049bb133111ebULL;hash^=hash>>31;
                    uint64_t slot=hash%stats.bins[bin];
                    if (slot<256) samples[slot]={a,b,dx,dy,dz};
                }
            }
        } else if (small_pair || (na.left<0 && nb.left<0)) {
            stats.direct++;
            double sa=na.end-na.begin,sb=nb.end-nb.begin;
            stats.pairs+=a==b ? sa*(sa-1)*.5 : sa*sb;
            if (sample) {
                if (stats.close.size()<2048) stats.close.push_back({a,b});
                else {
                    uint64_t hash=(uint64_t(a)*0x9e3779b97f4a7c15ULL)^
                                  (uint64_t(b)*0xbf58476d1ce4e5b9ULL);
                    hash^=hash>>30;hash*=0xbf58476d1ce4e5b9ULL;
                    hash^=hash>>27;hash*=0x94d049bb133111ebULL;
                    hash^=hash>>31;
                    uint64_t slot=hash%stats.direct;
                    if (slot<2048) stats.close[slot]={a,b};
                }
            }
        } else if (a==b) {
            count_interactions(na.left,na.left,stats,sample);
            count_interactions(na.left,na.right,stats,sample);
            count_interactions(na.right,na.right,stats,sample);
        } else if (nb.left<0 || (na.left>=0 && na.radius>=nb.radius)) {
            count_interactions(na.left,b,stats,sample);
            count_interactions(na.right,b,stats,sample);
        } else {
            count_interactions(a,nb.left,stats,sample);
            count_interactions(a,nb.right,stats,sample);
        }
    }

    static double int_power(double q,int p) {
        double v=1.;
        while(p) {
            if(p&1)v*=q;
            q*=q;p>>=1;
        }
        return v;
    }

    double harmonic_error(int a,int b,double d2,int p) const {
        const double* pa=multipole_power.data()+size_t(a)*(horder+1);
        const double* pb=multipole_power.data()+size_t(b)*(horder+1);
        const double* choose=choose_all[p];
        double sa=0.,sb=0.;
        if (target_mac) {
            const double* wa=target_norm.data()+size_t(a)*(horder+1);
            const double* wb=target_norm.data()+size_t(b)*(horder+1);
            for (int k=0;k<=p;k++) {
                sa+=choose[k]*pb[k]*wa[p-k];
                sb+=choose[k]*pa[k]*wb[p-k];
            }
        } else {
            sa=pb[0];sb=pa[0];
            for (int k=1;k<=p;k++) {
                sa=sa*tree[a].radius+choose[k]*pb[k];
                sb=sb*tree[b].radius+choose[k]*pa[k];
            }
            sa/=tree[a].acc_scale;sb/=tree[b].acc_scale;
        }
        return std::max(sa,sb)*int_power(1./d2,p/2)/d2;
    }

    double soft_error(int a,int b,double d2,int p) const {
        double estimate;
        if (target_mac) {
            const double* wa=target_norm.data()+size_t(a)*(horder+1);
            const double* wb=target_norm.data()+size_t(b)*(horder+1);
            const double* qa=radial_power.data()+size_t(a)*(order+1);
            const double* qb=radial_power.data()+size_t(b)*(order+1);
            double sa=0.,sb=0.;
            const double* choose=choose_all[p];
            for (int k=0;k<=p;k++) {
                sa+=choose[k]*qb[k]*wa[p-k];
                sb+=choose[k]*qa[k]*wb[p-k];
            }
            estimate=std::max(sa,sb)*int_power(1./d2,p/2)/d2;
        } else {
            double radius=tree[a].radius+tree[b].radius;
            double q2=radius*radius/d2;
            double scale=std::max(tree[b].mass/tree[a].acc_scale,
                                 tree[a].mass/tree[b].acc_scale)/d2;
            estimate=scale*int_power(q2,p/2);
        }
        double factor=soft_weight*double((p+2)*(p+3))/((order+2)*(order+3));
        return estimate*factor*(EPS2/(d2+EPS2));
    }

    double full_soft_error(int a,int b,double d2,int p) const {
        double s=d2+EPS2,estimate;
        if (target_mac) {
            const double* wa=target_norm.data()+size_t(a)*(horder+1);
            const double* wb=target_norm.data()+size_t(b)*(horder+1);
            const double* qa=radial_power.data()+size_t(a)*(order+1);
            const double* qb=radial_power.data()+size_t(b)*(order+1);
            double sa=0.,sb=0.;
            const double* choose=choose_all[p];
            for (int k=0;k<=p;k++) {
                sa+=choose[k]*qb[k]*wa[p-k];
                sb+=choose[k]*qa[k]*wb[p-k];
            }
            estimate=std::max(sa,sb)*int_power(1./s,p/2)/s;
        } else {
            double radius=tree[a].radius+tree[b].radius;
            double q2=radius*radius/s;
            double scale=std::max(tree[b].mass/tree[a].acc_scale,
                                 tree[a].mass/tree[b].acc_scale)/s;
            estimate=scale*int_power(q2,p/2);
        }
        return (p+1)*estimate;
    }

    bool acceptable(int a,int b,double dx,double dy,double dz,double r,double d2) const {
        if (!relative_mac)
            return r*r<theta2*d2 && (!horder || d2>near_cut*near_cut);
        double rr=tree[a].radius+tree[b].radius;
        if (near_soft && d2<16.*EPS2) {
            if (rr*rr>.81*(d2+EPS2)) return false;
            return full_soft_error(a,b,d2,order)<force_tol;
        }
        if (a==b || d2<=near_cut*near_cut) return false;
        double q2=rr*rr/d2;
        if (q2>.81) return false;
        double herror;
        if (power_mac && horder) herror=harmonic_error(a,b,d2,horder);
        else {
            double scale=std::max(tree[b].mass/tree[a].acc_scale,
                                  tree[a].mass/tree[b].acc_scale)/d2;
            herror=scale*int_power(q2,horder/2);
        }
        double serror=soft_error(a,b,d2,order);
        return std::max(herror,serror)<force_tol;
    }

    void clear_forces() {
        std::fill(locals.begin(),locals.end(),0.);
        std::fill(hlocals.begin(),hlocals.end(),0.);
        std::fill(ax.begin(),ax.end(),0.);
        std::fill(ay.begin(),ay.end(),0.);
        std::fill(az.begin(),az.end(),0.);
        batch_count=0;
        for (Bucket& bucket:pending) bucket.count=0;
        translations=direct_pairs=0;
        tracking_work=false;processed_work=0.;
        std::fill(selected_horders,selected_horders+MAXH+1,0);
        std::fill(selected_corders,selected_corders+MAXP+1,0);
    }

    void tune(double budget,double process_start) {
        theta2=0.36*0.36;
        if (relative_mac) force_tol=1e-6;
        Stats initial;
        double count_started=cpu_time();
        count_interactions(0,0,initial,true);
        double count_cost=cpu_time()-count_started;
        for (auto& samples:initial.far_bins) {
            initial.far.insert(initial.far.end(),samples.begin(),samples.end());
            std::vector<Interaction>().swap(samples);
        }
        for (size_t i=0;i<std::min(size_t(32),initial.far.size());i++) {
            const Interaction& in=initial.far[i];
            translate(in.a,in.b,in.dx,in.dy,in.dz);
        }
        flush_pending();
        std::fill(bin_time,bin_time+BINS,0.);
        std::fill(bin_calls,bin_calls+BINS,0);
        double t0=cpu_time();
        calibrating=true;
        for (const Interaction& in : initial.far)
            translate(in.a,in.b,in.dx,in.dy,in.dz);
        flush_pending();
        std::vector<Interaction> ordinary;
        for (const Interaction& in:initial.far) {
            double d2=in.dx*in.dx+in.dy*in.dy+in.dz*in.dz;
            if (in.a!=in.b && d2>=16.*EPS2) ordinary.push_back(in);
            if (ordinary.size()==4) break;
        }
        for (int hp=0;hp<=horder;hp+=2) {
            if (hp>0 && hp<6) continue;
            for (int pc=4;pc<=order;pc+=2) {
                int key=(hp/2)*7+pc/2;
                if (bin_calls[key]>=32) continue;
                const auto& probes=hp?ordinary:initial.far;
                if (probes.empty()) continue;
                bin_time[key]=0.;bin_calls[key]=0;
                for (int repeat=0;repeat<8;repeat++) {
                    for (int k=0;k<4;k++) batch[k]=probes[k%probes.size()];
                    batch_count=4;batch_horder=hp;batch_soft=pc;batch_bin=key;
                    flush_translations();
                }
            }
        }
        calibrating=false;
        double far_time=0.,far_samples=0.;
        for (int key=0;key<BINS;key++) {
            far_time+=bin_time[key];far_samples+=bin_calls[key];
        }
        double far_cost=far_samples?far_time/far_samples:0.;
        for (int key=0;key<BINS;key++)
            bin_cost[key]=timing_bias*(bin_calls[key]?bin_time[key]/bin_calls[key]:far_cost);
        t0=cpu_time();
        uint64_t pairs_before=direct_pairs;
        for (auto in : initial.close) {
            direct(in[0],in[1]);
            if (direct_pairs-pairs_before>=1500000) break;
        }
        double pair_time=cpu_time()-t0;
        double pair_cost=direct_pairs==pairs_before?3.e-9:
                         pair_time/(direct_pairs-pairs_before);
        pair_cost=std::max(pair_cost,pipelined_direct?1.5e-9:2.5e-9);
        pair_cost*=timing_bias;
        t0=cpu_time();
        int parts=0;
        std::vector<int> leaf_ids;
        for (int j=0;j<int(tree.size());j++)
            if (tree[j].left<0) leaf_ids.push_back(j);
        int ns=std::min(size_t(128),leaf_ids.size());
        for (int k=0;k<ns;k++) {
            int j=leaf_ids[size_t(k)*leaf_ids.size()/ns];
            if (horder) downward_harmonic(j);
            else downward(j);
            parts+=tree[j].end-tree[j].begin;
        }
        double eval_time=cpu_time()-t0;
        double eval_cost=parts ? eval_time/parts : 0.;
        t0=cpu_time();
        double hb[MAXHC],sb[MAXC],shift[3]={0.031,0.027,-0.018};
        for (int j=0;j<128;j++) {
            std::memset(hb,0,hn*sizeof(double));
            std::memset(sb,0,nc*sizeof(double));
            if (horder) shift_harmonic(hb,shift,true);
            shift_local(sb,shift);
            volatile double sink=hb[1]+sb[1];
            (void)sink;
        }
        double shift_cost=(cpu_time()-t0)/128;
        predicted_down=eval_cost*n+shift_cost*(tree.size()-1);
        predicted_down*=1.15;
        clear_forces();
        double target=budget*budget_fraction;
        pair_model=pair_cost;
        walk_model=.8*count_cost/std::max(uint64_t(1),initial.m2l+initial.direct);
        auto work_estimate=[&](const Stats& stats) {
            double estimate=stats.pairs*pair_cost;
            for (int key=0;key<BINS;key++) estimate+=stats.bins[key]*bin_cost[key];
            estimate+=.8*count_cost*(stats.m2l+stats.direct)/
                      std::max(uint64_t(1),initial.m2l+initial.direct);
            return estimate;
        };
        if (relative_mac) {
            Stats stats=initial;
            double eta=force_tol,slope=.25;
            for (int iter=0;iter<3;iter++) {
                double work=work_estimate(stats);
                double available=std::max(budget*.06,target-(cpu_time()-process_start)-
                                 predicted_down-n*1.8e-8-(2-iter)*count_cost);
                double step=std::clamp(std::log(work/available)/slope,-5.,5.);
                if (std::abs(step)<.08) break;
                double next_eta=std::clamp(eta*std::exp(step),1e-14,.05);
                force_tol=next_eta;
                Stats next;
                count_interactions(0,0,next,false);
                double next_work=work_estimate(next);
                if (std::abs(std::log(next_eta/eta))>.01) {
                    double measured=-std::log(next_work/work)/std::log(next_eta/eta);
                    slope=.3*slope+.7*std::clamp(measured,.08,.6);
                }
                eta=next_eta;stats=std::move(next);
            }
            double available=std::max(budget*.06,target-(cpu_time()-process_start)-
                              predicted_down-n*1.8e-8);
            double factor=std::pow(work_estimate(stats)/available,1./slope);
            force_tol=std::clamp(eta*std::clamp(factor,.5,2.),1e-14,.05);
            Stats final_stats;
            double final_started=cpu_time();
            count_interactions(0,0,final_stats,false);
            double final_walk=cpu_time()-final_started;
            walk_model=final_walk/std::max(uint64_t(1),final_stats.m2l+final_stats.direct);
            expected_work=work_estimate(final_stats);
            expected_work+=final_walk-.8*count_cost*(final_stats.m2l+final_stats.direct)/
                           std::max(uint64_t(1),initial.m2l+initial.direct);
            control_slope=slope;
        } else {
            double lo=.20,hi=.62;
            for (int iter=0;iter<7;iter++) {
                double theta=(lo+hi)*.5;
                theta2=theta*theta;
                Stats stats;
                count_interactions(0,0,stats,false);
                double estimate=cpu_time()-process_start+predicted_down+
                                work_estimate(stats)+n*1.8e-8;
                if (estimate>target) lo=theta;
                else hi=theta;
            }
            theta2=hi*hi;
        }
        deadline=process_start+budget-predicted_down*1.12-
                 std::max(0.015,budget*.045)-n*1.5e-8;
        hard_deadline=process_start+budget-predicted_down*1.20-
                      std::max(.008,budget*.01)-n*1.5e-8;
        std::fprintf(stderr,"tuned theta=%.5f tol=%.7g farcost=%.9f paircost=%.11f down=%.4f "
                     "used=%.4f budget=%.4f\n",std::sqrt(theta2),force_tol,
                     far_cost,pair_cost,predicted_down,
                     cpu_time()-process_start,budget);
        control_time=cpu_time();control_work=processed_work=0.;
        tracking_work=runtime_control && relative_mac;
    }

    void control_budget(double now) {
        double work=processed_work-control_work;
        double expected=expected_work-control_work;
        double elapsed=now-control_time;
        double remaining=expected_work-processed_work;
        double available=deadline-now;
        if (work<.20*expected || elapsed<.025 || remaining<=0. || available<=.005)
            return;
        double forecast=remaining*(elapsed/work);
        if (forecast<=.94*available) return;
        double factor=std::clamp(std::pow(forecast/(.86*available),1./control_slope),
                                 1.05,64.);
        double next=std::min(.05,force_tol*factor);
        if (next<=force_tol*1.01) return;
        factor=next/force_tol;
        force_tol=next;
        expected_work=processed_work+remaining*std::pow(factor,-control_slope);
        control_time=now;control_work=processed_work;
        control_adjustments++;
    }

    void coarse_emergency(int a,int b) {
        const Node& na=tree[a];
        const Node& nb=tree[b];
        if (a==b) {
            if (na.left<0) {
                const double* m=moments.data()+size_t(a)*nm;
                double r2=2.*(m[tab.index[2][0][0]]+m[tab.index[0][2][0]]+
                              m[tab.index[0][0][2]])/na.mass;
                double s=EPS2+std::max(0.,r2)*(5./3.);
                double w=-na.mass/(s*std::sqrt(s));
                double* l=locals.data()+size_t(a)*nc;
                l[tab.index[2][0][0]]+=w;
                l[tab.index[0][2][0]]+=w;
                l[tab.index[0][0][2]]+=w;
            }
            else {
                coarse_emergency(na.left,na.left);
                coarse_emergency(na.left,na.right);
                coarse_emergency(na.right,na.right);
            }
            return;
        }
        double dx=nb.c[0]-na.c[0],dy=nb.c[1]-na.c[1],dz=nb.c[2]-na.c[2];
        double r2=dx*dx+dy*dy+dz*dz+EPS2;
        double inv=1./std::sqrt(r2);
        double w=inv*inv*inv;
        double* la=locals.data()+size_t(a)*nc;
        double* lb=locals.data()+size_t(b)*nc;
        la[1]+=nb.mass*w*dx; la[2]+=nb.mass*w*dy; la[3]+=nb.mass*w*dz;
        lb[1]-=na.mass*w*dx; lb[2]-=na.mass*w*dy; lb[3]-=na.mass*w*dz;
    }

    void emergency_interaction(int a,int b) {
        if ((++emergency_calls&31)==0 && cpu_time()>hard_deadline) hard_emergency=true;
        if (hard_emergency) {
            coarse_emergency(a,b);
            return;
        }
        const Node& na=tree[a];
        const Node& nb=tree[b];
        double dx=na.c[0]-nb.c[0],dy=na.c[1]-nb.c[1],dz=na.c[2]-nb.c[2];
        double radius=na.radius+nb.radius;
        double d2=dx*dx+dy*dy+dz*dz+EPS2;
        bool small_pair=uint64_t(na.end-na.begin)*(nb.end-nb.begin)<=direct_cutoff;
        if (!small_pair && radius*radius<.36*d2) {
            queue_translation(2,a,b,dx,dy,dz);
        } else if (small_pair || (na.left<0 && nb.left<0)) {
            direct(a,b);
        } else if (a==b) {
            emergency_interaction(na.left,na.left);
            emergency_interaction(na.left,na.right);
            emergency_interaction(na.right,na.right);
        } else if (nb.left<0 || (na.left>=0 && na.radius>=nb.radius)) {
            emergency_interaction(na.left,b);
            emergency_interaction(na.right,b);
        } else {
            emergency_interaction(a,nb.left);
            emergency_interaction(a,nb.right);
        }
    }

    void interact(int a, int b) {
        if ((++calls&1023)==0) {
            double now=cpu_time();
            if (tracking_work && !emergency_mode) control_budget(now);
            if (now>deadline) emergency_mode=true;
        }
        if (emergency_mode) {
            emergency_interaction(a,b);
            return;
        }
        const Node& na=tree[a];
        const Node& nb=tree[b];
        double dx=na.c[0]-nb.c[0],dy=na.c[1]-nb.c[1],dz=na.c[2]-nb.c[2];
        double r=max_mac ? std::max(na.radius,nb.radius) : na.radius+nb.radius;
        double d2=dx*dx+dy*dy+dz*dz+(horder?0.:EPS2);
        bool small_pair=uint64_t(na.end-na.begin)*(nb.end-nb.begin)<=direct_cutoff;
        if (!small_pair && acceptable(a,b,dx,dy,dz,r,d2)) {
            translate(a,b,dx,dy,dz);
        } else if (small_pair || (na.left<0 && nb.left<0)) {
            direct(a,b);
        } else if (a==b) {
            interact(na.left,na.left);
            interact(na.left,na.right);
            interact(na.right,na.right);
        } else if (nb.left<0 || (na.left>=0 && na.radius>=nb.radius)) {
            interact(na.left,b);
            interact(na.right,b);
        } else {
            interact(a,nb.left);
            interact(a,nb.right);
        }
    }

    static __m256d pade_sum(const __m256d* terms,int n) {
        __m256d prev[MAXH],cur[MAXH],next[MAXH];
        __m256d pv[MAXH],cv[MAXH],nv[MAXH];
        __m256d raw={},base={};
        int start=std::max(0,n-12),len=n-start;
        for (int j=0;j<start;j++) base+=terms[j];
        __m256d valid=_mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        for (int j=0;j<len;j++) {
            raw+=terms[start+j];
            cur[j]=raw;
            prev[j]=_mm256_setzero_pd();
            pv[j]=cv[j]=valid;
        }
        __m256d estimate=raw,ok=valid;
        const __m256d signmask=_mm256_set1_pd(-0.);
        for (int k=1;k<len;k++) {
            for (int j=0;j<len-k;j++) {
                __m256d diff=cur[j+1]-cur[j];
                __m256d absdiff=_mm256_andnot_pd(signmask,diff);
                __m256d safe=_mm256_cmp_pd(absdiff,_mm256_set1_pd(1e-280),_CMP_GT_OQ);
                safe=_mm256_and_pd(safe,_mm256_and_pd(pv[j+1],
                                      _mm256_and_pd(cv[j],cv[j+1])));
                __m256d denom=_mm256_blendv_pd(_mm256_set1_pd(1.),diff,safe);
                next[j]=prev[j+1]+_mm256_set1_pd(1.)/denom;
                nv[j]=safe;
            }
            if (!(k&1)) {
                estimate=next[len-k-1];
                ok=nv[len-k-1];
            }
            for (int j=0;j<len-k+1;j++) {
                prev[j]=cur[j];pv[j]=cv[j];
            }
            for (int j=0;j<len-k;j++) {
                cur[j]=next[j];cv[j]=nv[j];
            }
        }
        __m256d bound={};
        for (int j=std::max(0,n-3);j<n;j++)
            bound+=_mm256_andnot_pd(signmask,terms[j]);
        bound=bound*_mm256_set1_pd(8.)+_mm256_set1_pd(1e-100);
        __m256d deviation=_mm256_andnot_pd(signmask,estimate-raw);
        ok=_mm256_and_pd(ok,_mm256_cmp_pd(deviation,bound,_CMP_LT_OQ));
        return base+_mm256_blendv_pd(raw,estimate,ok);
    }

    __attribute__((noinline))
    int leaf_down_vec(int node) {
        const Node& nd=tree[node];
        const double* sl=locals.data()+size_t(node)*nc;
        const double* hl=hlocals.data()+size_t(node)*hn;
        alignas(32) __m256d hb[MAXHC],px[MAXP+1],py[MAXP+1],pz[MAXP+1];
        alignas(32) __m256d cx[MAXH],cy[MAXH],cz[MAXH];
        int i=nd.begin;
        for (;i+3<nd.end;i+=4) {
            __m256d dx=_mm256_loadu_pd(&x[i])-_mm256_set1_pd(nd.c[0]);
            __m256d dy=_mm256_loadu_pd(&y[i])-_mm256_set1_pd(nd.c[1]);
            __m256d dz=_mm256_loadu_pd(&z[i])-_mm256_set1_pd(nd.c[2]);
            powers_vec(dx,px); powers_vec(dy,py); powers_vec(dz,pz);
            harmonic_basis_vec(dx,dy,dz,hb,horder-1);
            __m256d tx={},ty={},tz={};
            for (int n=1;n<=horder;n++) {
                __m256d nx={},ny={},nz={};
                for (int c=0;c<=1;c++)
                    for (int a=n-c;a>=0;a--) {
                        int b=n-c-a,j=hi(a,b,c);
                        __m256d l=_mm256_set1_pd(hl[j]);
                        if (a) nx+=l*hb[hi(a-1,b,c)];
                        if (b) ny+=l*hb[hi(a,b-1,c)];
                        if (c) nz+=l*hb[hi(a,b,0)];
                        else {
                            if (a>=2) nz-=l*hb[hi(a-2,b,1)];
                            if (b>=2) nz-=l*hb[hi(a,b-2,1)];
                        }
                    }
                tx+=nx;ty+=ny;tz+=nz;
                if (pade) { cx[n-1]=nx;cy[n-1]=ny;cz[n-1]=nz; }
            }
            if (pade) {
                tx=pade_sum(cx,horder);ty=pade_sum(cy,horder);tz=pade_sum(cz,horder);
            }
            for (int j=1;j<nc;j++) {
                int a=tab.ex[j][0],b=tab.ex[j][1],c=tab.ex[j][2];
                __m256d l=_mm256_set1_pd(sl[j]);
                if (a) tx+=l*px[a-1]*py[b]*pz[c];
                if (b) ty+=l*px[a]*py[b-1]*pz[c];
                if (c) tz+=l*px[a]*py[b]*pz[c-1];
            }
            _mm256_storeu_pd(&ax[i],_mm256_loadu_pd(&ax[i])+tx);
            _mm256_storeu_pd(&ay[i],_mm256_loadu_pd(&ay[i])+ty);
            _mm256_storeu_pd(&az[i],_mm256_loadu_pd(&az[i])+tz);
        }
        return i;
    }

    void downward_harmonic(int node) {
        const Node& nd=tree[node];
        const double* sl=locals.data()+size_t(node)*nc;
        const double* hl=hlocals.data()+size_t(node)*hn;
        if (nd.left<0) {
            double px[MAXP+1],py[MAXP+1],pz[MAXP+1];
            double hb[MAXHC];
            for (int i=leaf_down_vec(node);i<nd.end;i++) {
                double dx=x[i]-nd.c[0],dy=y[i]-nd.c[1],dz=z[i]-nd.c[2];
                powers(dx,px); powers(dy,py); powers(dz,pz);
                harmonic_basis(dx,dy,dz,hb,horder-1);
                double tx=0.,ty=0.,tz=0.;
                for (int n=1;n<=horder;n++) {
                    for (int c=0;c<=1;c++)
                        for (int a=n-c;a>=0;a--) {
                            int b=n-c-a,j=hi(a,b,c);
                            if (a) tx+=hl[j]*hb[hi(a-1,b,c)];
                            if (b) ty+=hl[j]*hb[hi(a,b-1,c)];
                            if (c) tz+=hl[j]*hb[hi(a,b,0)];
                            else {
                                if (a>=2) tz-=hl[j]*hb[hi(a-2,b,1)];
                                if (b>=2) tz-=hl[j]*hb[hi(a,b-2,1)];
                            }
                        }
                }
                for (int j=1;j<nc;j++) {
                    int a=tab.ex[j][0],b=tab.ex[j][1],c=tab.ex[j][2];
                    if (a) tx+=sl[j]*px[a-1]*py[b]*pz[c];
                    if (b) ty+=sl[j]*px[a]*py[b-1]*pz[c];
                    if (c) tz+=sl[j]*px[a]*py[b]*pz[c-1];
                }
                ax[i]+=tx; ay[i]+=ty; az[i]+=tz;
            }
        } else {
            double buf[MAXHC],shift[3],sbuf[MAXC];
            for (int ch : {nd.left,nd.right}) {
                for (int k=0;k<3;k++) shift[k]=tree[ch].c[k]-nd.c[k];
                std::memcpy(buf,hl,hn*sizeof(double));
                shift_harmonic(buf,shift,true);
                double* cl=hlocals.data()+size_t(ch)*hn;
                for (int j=1;j<hn;j++) cl[j]+=buf[j];
                std::memcpy(sbuf,sl,nc*sizeof(double));
                shift_local(sbuf,shift);
                double* sc=locals.data()+size_t(ch)*nc;
                for (int j=1;j<nc;j++) sc[j]+=sbuf[j];
                downward_harmonic(ch);
            }
        }
    }

    void shift_local(double* v, const double* shift) const {
        double ps[MAXP+1];
        for (int dim=0;dim<3;dim++) {
            powers(shift[dim],ps);
            for (int j=1;j<nc;j++) {
                int a=tab.ex[j][0],b=tab.ex[j][1],c=tab.ex[j][2];
                int maxk=order-tab.degree[j];
                double s=v[j];
                for (int k=1;k<=maxk;k++) {
                    int aa=a+(dim==0?k:0);
                    int bb=b+(dim==1?k:0);
                    int cc=c+(dim==2?k:0);
                    s+=v[tab.index[aa][bb][cc]]*ps[k];
                }
                v[j]=s;
            }
        }
    }

    void downward(int node) {
        const Node& nd=tree[node];
        const double* l=locals.data()+size_t(node)*nc;
        if (nd.left<0) {
            double px[MAXP+1],py[MAXP+1],pz[MAXP+1];
            for (int i=nd.begin;i<nd.end;i++) {
                powers(x[i]-nd.c[0],px);
                powers(y[i]-nd.c[1],py);
                powers(z[i]-nd.c[2],pz);
                double tx=0.,ty=0.,tz=0.;
                for (int j=1;j<nc;j++) {
                    int a=tab.ex[j][0],b=tab.ex[j][1],c=tab.ex[j][2];
                    if (a) tx+=l[j]*px[a-1]*py[b]*pz[c];
                    if (b) ty+=l[j]*px[a]*py[b-1]*pz[c];
                    if (c) tz+=l[j]*px[a]*py[b]*pz[c-1];
                }
                ax[i]+=tx; ay[i]+=ty; az[i]+=tz;
            }
        } else {
            double buf[MAXC],shift[3];
            for (int ch : {nd.left,nd.right}) {
                std::memcpy(buf,l,nc*sizeof(double));
                for (int k=0;k<3;k++) shift[k]=tree[ch].c[k]-nd.c[k];
                shift_local(buf,shift);
                double* lc=locals.data()+size_t(ch)*nc;
                for (int j=1;j<nc;j++) lc[j]+=buf[j];
                downward(ch);
            }
        }
    }
};

int main(int argc, char** argv) {
    double started=cpu_time();
    if (argc!=3) return 1;
    struct stat st;
    if (stat(argv[1],&st)!=0 || st.st_size%32!=0) return 2;
    int n=st.st_size/32;
    if (!n) {
        FILE* f=std::fopen(argv[2],"wb");
        if (f) std::fclose(f);
        return f?0:3;
    }
    int order=8, leaf=64,horder=16;
    double theta=0.35;
    double budget=2.;
    if (const char* s=std::getenv("GALAXY_CPU_LIMIT_S")) budget=std::atof(s);
    order=budget>=1e-6*n*std::log2(double(n))?10:8;
    if (const char* s=std::getenv("FMM_ORDER")) order=std::atoi(s);
    if (const char* s=std::getenv("FMM_LEAF")) leaf=std::atoi(s);
    if (const char* s=std::getenv("FMM_THETA")) theta=std::atof(s);
    if (const char* s=std::getenv("FMM_HORDER")) horder=std::atoi(s);
    if (order!=4 && order!=6 && order!=8 && order!=10 && order!=12) order=8;
    if (horder<4 || horder>MAXH || (horder&1)) horder=16;
    if (n<=1024) { order=4;horder=0; }
    Solver solver(n,order,leaf,theta,horder);
    solver.leaf_radius_factor=0.24;
    solver.max_mac=true;
    solver.relative_mac=true;
    solver.power_mac=true;
    solver.target_mac=true;
    solver.adaptive_h=true;
    solver.adaptive_c=true;
    solver.adaptive_safety=1.;
    solver.near_soft=true;
    solver.soft_weight=32.*(order+2)*(order+3)/110.;
    solver.coarse_theta2=.65*.65;
    solver.coarse_max=true;
    if (const char* s=std::getenv("FMM_LRAD")) solver.leaf_radius_factor=std::atof(s);
    if (const char* s=std::getenv("FMM_MAXMAC")) solver.max_mac=std::atoi(s);
    if (const char* s=std::getenv("FMM_COMPACT")) solver.compact=std::atoi(s);
    if (const char* s=std::getenv("FMM_BALANCED")) solver.balanced=std::atoi(s);
    if (const char* s=std::getenv("FMM_VSOFT")) solver.variable_soft=std::atoi(s);
    if (const char* s=std::getenv("FMM_VH")) solver.variable_h=std::atof(s);
    if (const char* s=std::getenv("FMM_NEWTON")) solver.newton_only=std::atoi(s);
    if (const char* s=std::getenv("FMM_FULLH")) solver.full_h=std::atoi(s);
    if (const char* s=std::getenv("FMM_PADE")) solver.pade=std::atoi(s);
    if (const char* s=std::getenv("FMM_TOL")) {
        solver.relative_mac=true;
        solver.force_tol=std::atof(s);
    }
    if (const char* s=std::getenv("FMM_POWER")) {
        solver.power_mac=std::atoi(s);
        solver.relative_mac=solver.power_mac || std::getenv("FMM_TOL");
    }
    if (const char* s=std::getenv("FMM_COARSE")) {
        double t=std::atof(s);
        solver.coarse_theta2=t*t;
        solver.coarse_max=true;
    }
    if (const char* s=std::getenv("FMM_TARGET")) {
        solver.target_mac=std::atoi(s);
        if (solver.target_mac) solver.power_mac=solver.relative_mac=true;
    }
    if (const char* s=std::getenv("FMM_ADAPTIVE_H")) solver.adaptive_h=std::atoi(s);
    if (const char* s=std::getenv("FMM_ADAPTIVE_C")) solver.adaptive_c=std::atoi(s);
    if (const char* s=std::getenv("FMM_ADAPTIVE_SAFETY")) solver.adaptive_safety=std::atof(s);
    if (const char* s=std::getenv("FMM_DIRECT_CUTOFF")) solver.direct_cutoff=std::strtoull(s,nullptr,10);
    if (const char* s=std::getenv("FMM_NEAR_SOFT")) solver.near_soft=std::atoi(s);
    if (const char* s=std::getenv("FMM_CONTROL")) solver.runtime_control=std::atoi(s);
    if (const char* s=std::getenv("FMM_PROBE_SCALE")) solver.timing_bias=std::atof(s);
    if (const char* s=std::getenv("FMM_ALIGN")) solver.align_flat=std::atoi(s);
    if (const char* s=std::getenv("FMM_PIPELINE")) solver.pipelined_direct=std::atoi(s);
    if (const char* s=std::getenv("FMM_BUDGET_FRACTION")) solver.budget_fraction=std::atof(s);
    if (const char* s=std::getenv("FMM_SOFT_WEIGHT")) solver.soft_weight=std::atof(s);
    solver.compact=true;
    if (const char* s=std::getenv("FMM_NEAR")) solver.near_cut=std::atof(s);
    if (!solver.relative_mac) {
        solver.power_mac=false;
        solver.target_mac=false;
        solver.adaptive_h=false;
        solver.adaptive_c=false;
    }
    if (n<=1024) {
        solver.leaf_size=n;
        solver.leaf_radius_factor=0.;
        solver.relative_mac=false;
        solver.power_mac=false;
        solver.target_mac=false;
    }
    solver.particles.resize(n);
    FILE* f=std::fopen(argv[1],"rb");
    if (!f) return 4;
    double row[4];
    for (int i=0;i<n;i++) {
        if (std::fread(row,sizeof(double),4,f)!=4) return 5;
        solver.particles[i]={{row[1],row[2],row[3]},row[0],uint32_t(i)};
    }
    std::fclose(f);
    solver.prepare();
    if (n>1024 && !std::getenv("FMM_THETA")) solver.tune(budget,started);
    double prepared=cpu_time();
    solver.interact(0,0);
    solver.flush_pending();
    double interacted=cpu_time();
    if (horder) solver.downward_harmonic(0);
    else solver.downward(0);
    double computed=cpu_time();
    std::vector<std::array<double,3>> out(n);
    for (int i=0;i<n;i++) {
        double a=solver.ax[i],b=solver.ay[i],c=solver.az[i];
        if (solver.rotated) {
            double u=a,v=b,w=c;
            a=solver.rotation[0][0]*u+solver.rotation[1][0]*v+solver.rotation[2][0]*w;
            b=solver.rotation[0][1]*u+solver.rotation[1][1]*v+solver.rotation[2][1]*w;
            c=solver.rotation[0][2]*u+solver.rotation[1][2]*v+solver.rotation[2][2]*w;
        }
        if (!std::isfinite(a)) a=0.;
        if (!std::isfinite(b)) b=0.;
        if (!std::isfinite(c)) c=0.;
        out[solver.ids[i]]={a,b,c};
    }
    f=std::fopen(argv[2],"wb");
    if (!f) return 6;
    size_t wrote=std::fwrite(out.data(),sizeof(double)*3,n,f);
    std::fclose(f);
    rusage usage;
    getrusage(RUSAGE_SELF,&usage);
    double average_order=0.;
    for (int p=0;p<=horder;p++) average_order+=p*solver.selected_horders[p];
    if (solver.translations) average_order/=solver.translations;
    double average_cart=0.;
    for (int p=0;p<=order;p++) average_cart+=p*solver.selected_corders[p];
    if (solver.translations) average_cart/=solver.translations;
    std::fprintf(stderr,"n=%d p=%d hp=%d leaf=%d theta=%.4f nodes=%zu m2l=%llu pairs=%llu "
                 "prep=%.4f interact=%.4f down=%.4f total=%.4f emergency=%d hard=%d rss=%.1fMB avgp=%.2f avgc=%.2f controls=%d endtol=%.4g rotated=%d\n",
                 n,order,horder,solver.leaf_size,std::sqrt(solver.theta2),solver.tree.size(),
                 (unsigned long long)solver.translations,
                 (unsigned long long)solver.direct_pairs,
                 prepared-started,interacted-prepared,computed-interacted,cpu_time()-started,
                 int(solver.emergency_mode),int(solver.hard_emergency),usage.ru_maxrss/1024.,average_order,average_cart,
                 solver.control_adjustments,solver.force_tol,int(solver.rotated));
    return wrote==size_t(n)?0:7;
}
