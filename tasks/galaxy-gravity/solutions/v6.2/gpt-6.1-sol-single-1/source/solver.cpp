#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <sys/stat.h>
#include <immintrin.h>

#include "kernels.hpp"
#include "harmonic.hpp"

struct Particle { double m,x,y,z; int id; };
struct Node {
    double x,y,z,r,m;
    int begin,end,left,right;
    bool leaf() const { return left<0; }
};
struct Pair { int a,b; };
struct Costs {
    double pp=2.2e-9,h=5.8e-6,res=4.8e-6,body=1.0e-6,down=0.4;
};

static std::vector<Particle> particles;
static std::vector<Node> tree;
static std::vector<std::array<double,NC>> moments,locals;
static std::vector<std::array<double,H::NC>> hmoments,hlocals;
static std::vector<double> px,py,pz,pm,ax,ay,az,force_scale;
static std::vector<Pair> far_pairs,near_pairs,body_pairs;
static int leaf_size=32,direct_threshold=6000,body_min=512;
static double tolerance=5e-7,body_theta=0.45,near_cut=0.05;
static double cpu_limit=4.72,target_cpu=4.65;
static double down_reserve=0.5,h_deadline=3,res_deadline=4;
static bool equal_mass=true,emergency=false;
static Costs costs;
static double direct_work=0,body_work=0;
static long long direct_pairs=0,coarse_near=0,coarse_h=0,coarse_res=0,coarse_body=0;
static volatile double calibration_sink=0;

static double cpu() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);
    return t.tv_sec+1e-9*t.tv_nsec;
}

static int build_tree(int begin,int end) {
    int id=tree.size();
    tree.emplace_back();
    double mass=0,x=0,y=0,z=0;
    double lo[3]={1e300,1e300,1e300},hi[3]={-1e300,-1e300,-1e300};
    for(int i=begin;i<end;i++) {
        const auto &p=particles[i];
        mass+=p.m; x+=p.m*p.x; y+=p.m*p.y; z+=p.m*p.z;
        const double v[3]={p.x,p.y,p.z};
        for(int k=0;k<3;k++) { lo[k]=std::min(lo[k],v[k]); hi[k]=std::max(hi[k],v[k]); }
    }
    x/=mass; y/=mass; z/=mass;
    double r2=0;
    for(int i=begin;i<end;i++) {
        const auto &p=particles[i];
        double dx=p.x-x,dy=p.y-y,dz=p.z-z;
        r2=std::max(r2,dx*dx+dy*dy+dz*dz);
    }
    tree[id]={x,y,z,std::sqrt(r2),mass,begin,end,-1,-1};
    if(end-begin>leaf_size) {
        int k=0;
        if(hi[1]-lo[1]>hi[k]-lo[k]) k=1;
        if(hi[2]-lo[2]>hi[k]-lo[k]) k=2;
        double middle=0.5*(lo[k]+hi[k]);
        auto split=std::partition(particles.begin()+begin,particles.begin()+end,[=](const Particle &p) {
            return (k==0?p.x:k==1?p.y:p.z)<middle;
        });
        int mid=split-particles.begin();
        if(mid==begin || mid==end) mid=begin+(end-begin)/2;
        int a=build_tree(begin,mid),b=build_tree(mid,end);
        tree[id].left=a; tree[id].right=b;
    }
    return id;
}

static void upward(int id) {
    const Node &a=tree[id];
    double *m=moments[id].data(),*h=hmoments[id].data();
    if(a.leaf()) {
        double v[NC],w[H::NC];
        for(int j=a.begin;j<a.end;j++) {
            monomials(a.x-px[j],a.y-py[j],a.z-pz[j],v);
            H::monomials(a.x-px[j],a.y-py[j],a.z-pz[j],w);
            for(int k=0;k<NM;k++) m[k]+=pm[j]*v[k];
            for(int k=0;k<H::NM;k++) h[k]+=pm[j]*w[k];
        }
        m[1]=m[2]=m[3]=0;
        h[H::GX]=h[H::GY]=h[H::GZ]=0;
    } else {
        upward(a.left); upward(a.right);
        for(int child:{a.left,a.right}) {
            const Node &b=tree[child];
            shift_m(moments[child].data(),m,b.x-a.x,b.y-a.y,b.z-a.z);
            H::shift_m(hmoments[child].data(),h,b.x-a.x,b.y-a.y,b.z-a.z);
        }
    }
    m[0]=h[0]=a.m;
}

static void estimate_force(int id) {
    const Node &a=tree[id];
    if(!a.leaf()) {
        estimate_force(a.left); estimate_force(a.right);
        const Node &b=tree[a.left],&c=tree[a.right];
        double fb=force_scale[a.left],fc=force_scale[a.right];
        // An inverse-RMS scale protects particles with weak accelerations.
        force_scale[id]=std::sqrt((a.end-a.begin)/((b.end-b.begin)/(fb*fb)+(c.end-c.begin)/(fc*fc)));
        return;
    }
    double fx=0,fy=0,fz=0;
    int stack[256],top=0;
    stack[top++]=0;
    while(top) {
        int k=stack[--top];
        if(k==id) continue;
        const Node &b=tree[k];
        double x=b.x-a.x,y=b.y-a.y,z=b.z-a.z,r2=x*x+y*y+z*z+0.0001;
        bool contains=b.begin<=a.begin && b.end>=a.end;
        if(b.leaf() || (!contains && b.r*b.r<0.64*r2)) {
            double f=b.m/(r2*std::sqrt(r2));
            fx+=f*x; fy+=f*y; fz+=f*z;
        } else {
            stack[top++]=b.left; stack[top++]=b.right;
        }
    }
    double x=a.x-tree[0].x,y=a.y-tree[0].y,z=a.z-tree[0].z;
    double dc2=x*x+y*y+z*z,outer=std::sqrt(dc2)+a.r;
    double correction=(dc2+0.0001)/(outer*outer+0.0001);
    force_scale[id]=std::max(std::sqrt(fx*fx+fy*fy+fz*fz)*correction,1e-12);
}

static double pp_work(const Node &a,const Node &b,bool self) {
    int na=a.end-a.begin,nb=b.end-b.begin;
    if(!self) return double(na)*(((nb+3)&~3)+5);
    double sum=5*na;
    for(int k=1;k<na;k++) sum+=(k+3)&~3;
    return sum;
}

static void add_near(int ia,int ib) {
    const Node &a=tree[ia],&b=tree[ib];
    bool self=ia==ib;
    near_pairs.push_back({ia,ib});
    direct_pairs+=self?(long long)(a.end-a.begin)*(a.end-a.begin-1)/2:
        (long long)(a.end-a.begin)*(b.end-b.begin);
    direct_work+=pp_work(a,b,self);
}

static bool error_ok(double rho2,double ratio,double r2) {
    double eh=std::pow(rho2,0.5*H::ORDER)*ratio;
    double es=0.0001/r2*std::pow(rho2,0.5*RES_ORDER)*ratio*(RES_ORDER+1)*(RES_ORDER+2)*0.5;
    return std::max(eh,es)<tolerance;
}

static bool body_ok(int ia,int ib,double distance) {
    const Node &a=tree[ia],&b=tree[ib];
    if(!a.leaf() || a.r<=b.r || b.end-b.begin<body_min) return false;
    double d=distance-a.r;
    if(!(b.r<body_theta*d)) return false;
    double d2=d*d;
    double rho2=b.r*b.r/d2;
    double ratio=std::max(b.m/force_scale[ia],a.m/force_scale[ib])/d2;
    return error_ok(rho2,ratio,d2);
}

static void interactions(int ia,int ib) {
    const Node &a=tree[ia],&b=tree[ib];
    if(ia==ib) {
        if(a.leaf()) add_near(ia,ib);
        else {
            interactions(a.left,a.left);
            interactions(a.left,a.right);
            interactions(a.right,a.right);
        }
        return;
    }
    long long count=(long long)(a.end-a.begin)*(b.end-b.begin);
    if(count<=direct_threshold) { add_near(ia,ib); return; }
    double x=a.x-b.x,y=a.y-b.y,z=a.z-b.z,d2=x*x+y*y+z*z;
    double distance=std::sqrt(d2);
    bool separated=distance-a.r-b.r>near_cut;
    if(separated) {
        double radius=a.r+b.r;
        if((a.r>b.r && b.m>8*a.m) || (b.r>a.r && a.m>8*b.m))
            radius=2*std::max(a.r,b.r);
        double rho2=radius*radius/d2;
        double ratio=std::max(b.m/force_scale[ia],a.m/force_scale[ib])/d2;
        if(rho2<0.64 && error_ok(rho2,ratio,d2)) {
            far_pairs.push_back({ia,ib});
            return;
        }
        if(body_ok(ia,ib,distance)) {
            body_pairs.push_back({ia,ib});
            body_work+=((a.end-a.begin+3)&~3)+8;
            return;
        }
        if(body_ok(ib,ia,distance)) {
            body_pairs.push_back({ib,ia});
            body_work+=((b.end-b.begin+3)&~3)+8;
            return;
        }
    }
    if(a.leaf() && b.leaf()) add_near(ia,ib);
    else if(b.leaf() || (!a.leaf() && a.r>b.r)) {
        interactions(a.left,ib); interactions(a.right,ib);
    } else {
        interactions(ia,b.left); interactions(ia,b.right);
    }
}

static void make_lists() {
    far_pairs.clear(); near_pairs.clear(); body_pairs.clear();
    direct_pairs=0; direct_work=body_work=0;
    interactions(0,0);
}

template<bool Equal>
static void direct(const Node &a,const Node &b,bool self) {
    alignas(32) static const long long masks[4][4]={{0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0}};
    const V soft={0.0001,0.0001,0.0001,0.0001},one={1,1,1,1};
    for(int i=a.begin;i<a.end;i++) {
        V xi={px[i],px[i],px[i],px[i]},yi={py[i],py[i],py[i],py[i]},zi={pz[i],pz[i],pz[i],pz[i]};
        V mass={pm[i],pm[i],pm[i],pm[i]};
        V sx={0,0,0,0},sy=sx,sz=sx;
        for(int j=self?i+1:b.begin;j<b.end;j+=4) {
            V x=_mm256_loadu_pd(&px[j])-xi,y=_mm256_loadu_pd(&py[j])-yi,z=_mm256_loadu_pd(&pz[j])-zi;
            V r2=_mm256_fmadd_pd(x,x,_mm256_fmadd_pd(y,y,_mm256_fmadd_pd(z,z,soft)));
            V f=one/(r2*_mm256_sqrt_pd(r2));
            if(j+4>b.end) f=_mm256_and_pd(f,_mm256_castsi256_pd(_mm256_load_si256(reinterpret_cast<const __m256i *>(masks[b.end-j]))));
            V fb=f*mass,fa;
            if constexpr(Equal) fa=fb;
            else fa=f*_mm256_loadu_pd(&pm[j]);
            sx=_mm256_fmadd_pd(fa,x,sx); sy=_mm256_fmadd_pd(fa,y,sy); sz=_mm256_fmadd_pd(fa,z,sz);
            _mm256_storeu_pd(&ax[j],_mm256_fnmadd_pd(fb,x,_mm256_loadu_pd(&ax[j])));
            _mm256_storeu_pd(&ay[j],_mm256_fnmadd_pd(fb,y,_mm256_loadu_pd(&ay[j])));
            _mm256_storeu_pd(&az[j],_mm256_fnmadd_pd(fb,z,_mm256_loadu_pd(&az[j])));
        }
        ax[i]+=sx[0]+sx[1]+sx[2]+sx[3];
        ay[i]+=sy[0]+sy[1]+sy[2]+sy[3];
        az[i]+=sz[0]+sz[1]+sz[2]+sz[3];
    }
}

static void monopoles(int ia,int ib) {
    const Node &a=tree[ia],&b=tree[ib];
    double x=b.x-a.x,y=b.y-a.y,z=b.z-a.z,r2=x*x+y*y+z*z+0.0001;
    double f=1/(r2*std::sqrt(r2));
    locals[ia][3]+=b.m*f*x; locals[ia][2]+=b.m*f*y; locals[ia][1]+=b.m*f*z;
    locals[ib][3]-=a.m*f*x; locals[ib][2]-=a.m*f*y; locals[ib][1]-=a.m*f*z;
}

static void apply_body(const Node &a,int ib) {
    const Node &b=tree[ib];
    alignas(32) V l[NC]={},h[H::NC]={};
    const V bx={b.x,b.x,b.x,b.x},by={b.y,b.y,b.y,b.y},bz={b.z,b.z,b.z,b.z};
    int i=a.begin;
    for(;i+4<=a.end;i+=4) {
        V fx=_mm256_loadu_pd(&ax[i]),fy=_mm256_loadu_pd(&ay[i]),fz=_mm256_loadu_pd(&az[i]);
        V x=_mm256_loadu_pd(&px[i])-bx,y=_mm256_loadu_pd(&py[i])-by,z=_mm256_loadu_pd(&pz[i])-bz;
        V m=_mm256_loadu_pd(&pm[i]);
        H::m2p4(hmoments[ib].data(),h,x,y,z,m,fx,fy,fz);
        residual_m2p4(moments[ib].data(),l,x,y,z,m,fx,fy,fz);
        _mm256_storeu_pd(&ax[i],fx); _mm256_storeu_pd(&ay[i],fy); _mm256_storeu_pd(&az[i],fz);
    }
    if(i<a.end) {
        double xx[4]={1,1,1,1},yy[4]={},zz[4]={},mm[4]={};
        for(int k=0;i+k<a.end;k++) {
            xx[k]=px[i+k]-b.x; yy[k]=py[i+k]-b.y; zz[k]=pz[i+k]-b.z; mm[k]=pm[i+k];
        }
        V x=_mm256_loadu_pd(xx),y=_mm256_loadu_pd(yy),z=_mm256_loadu_pd(zz),m=_mm256_loadu_pd(mm);
        V fx={0,0,0,0},fy=fx,fz=fx;
        H::m2p4(hmoments[ib].data(),h,x,y,z,m,fx,fy,fz);
        residual_m2p4(moments[ib].data(),l,x,y,z,m,fx,fy,fz);
        for(int k=0;i+k<a.end;k++) { ax[i+k]+=fx[k]; ay[i+k]+=fy[k]; az[i+k]+=fz[k]; }
    }
    for(int k=1;k<NC;k++) locals[ib][k]+=l[k][0]+l[k][1]+l[k][2]+l[k][3];
    for(int k=1;k<H::NC;k++) hlocals[ib][k]+=h[k][0]+h[k][1]+h[k][2]+h[k][3];
}

static size_t sample_index(size_t q,size_t count,size_t size,size_t block=128) {
    block=std::min({block,count,size});
    size_t groups=(count+block-1)/block;
    size_t start=groups>1?(q/block)*(size-block)/(groups-1):0;
    return std::min(size-1,start+q%block);
}

static void calibrate() {
    // All timing probes use this snapshot; their accumulators are cleared below.
    int count=std::min<size_t>(2048,near_pairs.size());
    double work=0,t=cpu();
    for(int q=0;q<count;q++) {
        auto pair=near_pairs[sample_index(q,count,near_pairs.size())];
        const Node &a=tree[pair.a],&b=tree[pair.b];
        if(equal_mass) direct<true>(a,b,pair.a==pair.b);
        else direct<false>(a,b,pair.a==pair.b);
        work+=pp_work(a,b,pair.a==pair.b);
    }
    if(work>0) costs.pp=(cpu()-t)/work;
    if(!far_pairs.empty()) {
        count=std::min<size_t>(1536,far_pairs.size());
        t=cpu();
        for(int q=0;q<count;q++) {
            auto pair=far_pairs[sample_index(q,count,far_pairs.size())];
            const Node &a=tree[pair.a],&b=tree[pair.b];
            H::m2l_mutual(hmoments[pair.a].data(),hmoments[pair.b].data(),hlocals[pair.a].data(),hlocals[pair.b].data(),a.x-b.x,a.y-b.y,a.z-b.z);
        }
        costs.h=(cpu()-t)/count;
        alignas(32) V l[NC]={};
        t=cpu();
        for(int q=0;q<count;q++) {
            double xx[4],yy[4],zz[4]; const double *m[4];
            for(int k=0;k<4;k++) {
                auto pair=far_pairs[sample_index(size_t(q)*4+k,size_t(count)*4,far_pairs.size())];
                if(k&1) std::swap(pair.a,pair.b);
                const Node &a=tree[pair.a],&b=tree[pair.b];
                xx[k]=a.x-b.x; yy[k]=a.y-b.y; zz[k]=a.z-b.z;
                m[k]=moments[pair.b].data();
            }
            residual_m2l4(m[0],m[1],m[2],m[3],l,_mm256_loadu_pd(xx),_mm256_loadu_pd(yy),_mm256_loadu_pd(zz));
        }
        costs.res=0.5*(cpu()-t)/count;
        calibration_sink+=l[1][0]*1e-100;
    }
    if(!body_pairs.empty()) {
        count=std::min<size_t>(128,body_pairs.size());
        work=0; t=cpu();
        for(int q=0;q<count;q++) {
            auto pair=body_pairs[sample_index(q,count,body_pairs.size(),32)];
            apply_body(tree[pair.a],pair.b);
            work+=((tree[pair.a].end-tree[pair.a].begin+3)&~3)+8;
        }
        costs.body=(cpu()-t)/work;
    }
    std::vector<int> parents(tree.size(),0);
    for(size_t i=0;i<tree.size();i++) if(!tree[i].leaf()) {
        parents[tree[i].left]=parents[tree[i].right]=i;
    }
    double l[NC]={},h[H::NC]={};
    count=std::min<size_t>(256,tree.size()-1);
    t=cpu();
    for(int q=0;q<count;q++) {
        int child=1+sample_index(q,count,tree.size()-1,64),parent=parents[child];
        const Node &a=tree[parent],&b=tree[child];
        shift_l(moments[parent].data(),l,b.x-a.x,b.y-a.y,b.z-a.z);
        H::shift_l(hmoments[parent].data(),h,b.x-a.x,b.y-a.y,b.z-a.z);
    }
    double shift_cost=count?(cpu()-t)/count:0;
    calibration_sink+=(l[1]+h[1])*1e-100;
    std::vector<int> leaf_ids;
    for(size_t i=0;i<tree.size();i++) if(tree[i].leaf()) leaf_ids.push_back(i);
    count=std::min<size_t>(64,leaf_ids.size());
    double fx=0,fy=0,fz=0;
    int evaluated=0;
    t=cpu();
    for(int q=0;q<count;q++) {
        int id=leaf_ids[size_t(q)*leaf_ids.size()/count];
        const Node &a=tree[id];
        for(int j=a.begin;j<std::min(a.end,a.begin+16);j++) {
            evaluate_l(moments[id].data(),px[j]-a.x,py[j]-a.y,pz[j]-a.z,fx,fy,fz);
            H::evaluate_l(hmoments[id].data(),px[j]-a.x,py[j]-a.y,pz[j]-a.z,fx,fy,fz);
            evaluated++;
        }
    }
    double eval_cost=evaluated?(cpu()-t)/evaluated:0;
    costs.down=shift_cost*(tree.size()-1)+eval_cost*particles.size()+0.006;
    calibration_sink+=(fx+fy+fz)*1e-100;
    std::memset(locals.data(),0,locals.size()*sizeof(locals[0]));
    std::memset(hlocals.data(),0,hlocals.size()*sizeof(hlocals[0]));
    std::fill(ax.begin(),ax.end(),0); std::fill(ay.begin(),ay.end(),0); std::fill(az.begin(),az.end(),0);
    costs.pp*=1.045; costs.h*=1.08; costs.res*=1.10; costs.body*=1.10; costs.down*=1.12;
}

static double predicted_work() {
    return direct_work*costs.pp+far_pairs.size()*(costs.h+costs.res)+body_work*costs.body;
}

static void choose_tolerance() {
    if(far_pairs.empty()) return;
    calibrate();
    direct_threshold=std::clamp<int>(1.40*(costs.h+costs.res)/costs.pp,500,24000);
    body_min=std::clamp<int>(1.25*costs.body/costs.pp,64,4096);
    for(int iteration=0;iteration<4;iteration++) {
        make_lists();
        double available=target_cpu-cpu()-costs.down-0.025;
        if(available<0.1) { tolerance=std::min(1e-2,tolerance*100); continue; }
        double ratio=predicted_work()/available;
        if(ratio>0.94 && ratio<1.005) break;
        double factor=std::clamp(std::pow(ratio,5.5),0.12,12.0);
        tolerance=std::clamp(tolerance*factor,1e-14,1e-2);
    }
    make_lists();
}

template<bool Equal>
static void all_direct() {
    size_t q=0;
    for(auto pair:near_pairs) {
        if((q++&127)==0 && cpu()>cpu_limit-down_reserve-0.65) emergency=true;
        if(emergency && pair.a!=pair.b) { monopoles(pair.a,pair.b); coarse_near++; }
        else direct<Equal>(tree[pair.a],tree[pair.b],pair.a==pair.b);
    }
}

static void all_bodies() {
    size_t q=0;
    for(auto pair:body_pairs) {
        if((q++&63)==0 && cpu()>cpu_limit-down_reserve-0.45) emergency=true;
        if(emergency) { monopoles(pair.a,pair.b); coarse_body++; }
        else apply_body(tree[pair.a],pair.b);
    }
}

static void residuals() {
    std::vector<int> offsets(tree.size()+1,0);
    for(auto pair:far_pairs) { offsets[pair.a+1]++; offsets[pair.b+1]++; }
    for(size_t i=1;i<offsets.size();i++) offsets[i]+=offsets[i-1];
    std::vector<int> sources(offsets.back()),next=offsets;
    for(auto pair:far_pairs) { sources[next[pair.a]++]=pair.b; sources[next[pair.b]++]=pair.a; }
    alignas(32) V l[NC];
    alignas(32) double zero[NC]={};
    bool low=false;
    for(size_t i=0;i<tree.size();i++) {
        if(offsets[i]==offsets[i+1]) continue;
        if(cpu()>res_deadline) low=true;
        const Node &a=tree[i];
        for(int k=0;k<NC;k++) l[k]=V{0,0,0,0};
        for(int q=offsets[i];q<offsets[i+1];q+=4) {
            double xx[4],yy[4],zz[4]; const double *m[4];
            for(int k=0;k<4;k++) {
                if(q+k<offsets[i+1]) {
                    int j=sources[q+k]; const Node &b=tree[j];
                    xx[k]=a.x-b.x; yy[k]=a.y-b.y; zz[k]=a.z-b.z;
                    m[k]=moments[j].data();
                } else { xx[k]=1; yy[k]=zz[k]=0; m[k]=zero; }
            }
            if(low) {
                residual_m2l4_coarse(m[0],m[1],m[2],m[3],l,_mm256_loadu_pd(xx),_mm256_loadu_pd(yy),_mm256_loadu_pd(zz));
                coarse_res++;
            } else residual_m2l4(m[0],m[1],m[2],m[3],l,_mm256_loadu_pd(xx),_mm256_loadu_pd(yy),_mm256_loadu_pd(zz));
        }
        for(int k=1;k<NC;k++) locals[i][k]+=l[k][0]+l[k][1]+l[k][2]+l[k][3];
    }
}

static void all_far() {
    if(emergency) {
        for(auto pair:far_pairs) { monopoles(pair.a,pair.b); coarse_h++; }
        return;
    }
    bool low=false;
    size_t q=0;
    for(auto pair:far_pairs) {
        if((q++&63)==0 && cpu()>h_deadline) low=true;
        const Node &a=tree[pair.a],&b=tree[pair.b];
        if(low) {
            // Keep complete interactions when the high-order budget is exhausted.
            H::m2l_coarse(hmoments[pair.a].data(),hmoments[pair.b].data(),hlocals[pair.a].data(),hlocals[pair.b].data(),a.x-b.x,a.y-b.y,a.z-b.z);
            coarse_h++;
        } else H::m2l_mutual(hmoments[pair.a].data(),hmoments[pair.b].data(),hlocals[pair.a].data(),hlocals[pair.b].data(),a.x-b.x,a.y-b.y,a.z-b.z);
    }
    residuals();
}

static void downward(int id) {
    const Node &a=tree[id];
    double now=cpu();
    if(a.leaf()) {
        if(now>cpu_limit+0.07) {
            double fx=locals[id][3]+hlocals[id][H::GX],fy=locals[id][2]+hlocals[id][H::GY],fz=locals[id][1]+hlocals[id][H::GZ];
            for(int j=a.begin;j<a.end;j++) { ax[j]+=fx; ay[j]+=fy; az[j]+=fz; }
        } else for(int j=a.begin;j<a.end;j++) {
            evaluate_l(locals[id].data(),px[j]-a.x,py[j]-a.y,pz[j]-a.z,ax[j],ay[j],az[j]);
            H::evaluate_l(hlocals[id].data(),px[j]-a.x,py[j]-a.y,pz[j]-a.z,ax[j],ay[j],az[j]);
        }
    } else for(int child:{a.left,a.right}) {
        const Node &b=tree[child];
        if(now>cpu_limit-0.04) {
            double fx=0,fy=0,fz=0;
            evaluate_l(locals[id].data(),b.x-a.x,b.y-a.y,b.z-a.z,fx,fy,fz);
            H::evaluate_l(hlocals[id].data(),b.x-a.x,b.y-a.y,b.z-a.z,fx,fy,fz);
            locals[child][3]+=fx; locals[child][2]+=fy; locals[child][1]+=fz;
        } else {
            shift_l(locals[id].data(),locals[child].data(),b.x-a.x,b.y-a.y,b.z-a.z);
            H::shift_l(hlocals[id].data(),hlocals[child].data(),b.x-a.x,b.y-a.y,b.z-a.z);
        }
        downward(child);
    }
}

static bool flat_distribution(const std::vector<std::array<double,4>> &input) {
    double cx=0,cy=0,cz=0;
    for(auto p:input) { cx+=p[1]; cy+=p[2]; cz+=p[3]; }
    cx/=input.size(); cy/=input.size(); cz/=input.size();
    double xx=0,yy=0,zz=0,xy=0,xz=0,yz=0;
    for(auto p:input) {
        double x=p[1]-cx,y=p[2]-cy,z=p[3]-cz;
        xx+=x*x; yy+=y*y; zz+=z*z; xy+=x*y; xz+=x*z; yz+=y*z;
    }
    double tr=xx+yy+zz,det=xx*yy*zz+2*xy*xz*yz-xx*yz*yz-yy*xz*xz-zz*xy*xy;
    return det<0.003*tr*tr*tr;
}

static double finite_value(double x) {
    uint64_t bits;
    std::memcpy(&bits,&x,8);
    return (bits&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000)?0:x;
}

int main(int argc,char **argv) {
    if(argc!=3) return 1;
    double t0=cpu();
    if(const char *s=std::getenv("FMM_BUDGET")) {
        double limit=std::atof(s);
        if(limit>0 && limit<cpu_limit) cpu_limit=limit;
    }
    struct stat st;
    if(stat(argv[1],&st) || st.st_size%32) return 2;
    int n=st.st_size/32;
    std::vector<std::array<double,4>> input(n);
    FILE *f=std::fopen(argv[1],"rb");
    if(!f || std::fread(input.data(),32,n,f)!=(size_t)n) return 3;
    std::fclose(f);
    particles.resize(n);
    for(int i=0;i<n;i++) {
        particles[i]={input[i][0],input[i][1],input[i][2],input[i][3],i};
        if(input[i][0]!=input[0][0]) equal_mass=false;
    }
    if(n>1024 && flat_distribution(input)) leaf_size=96;
    if(n>1024) { tree.reserve(n); build_tree(0,n); }
    px.resize(n+4); py.resize(n+4); pz.resize(n+4); pm.resize(n+4);
    ax.resize(n+4); ay.resize(n+4); az.resize(n+4);
    for(int i=0;i<n;i++) {
        const Particle &p=particles[i];
        px[i]=p.x; py[i]=p.y; pz[i]=p.z; pm[i]=p.m;
    }
    double t1=cpu(),t2=t1,t3=t1,t4=t1,t5=t1,t6=t1;
    if(n<=1024) {
        Node all={0,0,0,0,1,0,n,-1,-1};
        if(equal_mass) direct<true>(all,all,true);
        else direct<false>(all,all,true);
    } else {
        moments.resize(tree.size()); locals.resize(tree.size());
        hmoments.resize(tree.size()); hlocals.resize(tree.size());
        force_scale.resize(tree.size());
        upward(0); estimate_force(0);
        t2=cpu();
        make_lists();
        choose_tolerance();
        t3=cpu();
        down_reserve=std::max(0.12,costs.down*1.08+0.015);
        res_deadline=cpu_limit-down_reserve-0.06;
        h_deadline=res_deadline-costs.res*far_pairs.size()-0.05;
        if(equal_mass) all_direct<true>(); else all_direct<false>();
        t4=cpu();
        all_bodies();
        t5=cpu();
        all_far();
        t6=cpu();
        downward(0);
    }
    std::vector<std::array<double,3>> output(n);
    for(int i=0;i<n;i++) output[particles[i].id]={finite_value(ax[i]),finite_value(ay[i]),finite_value(az[i])};
    f=std::fopen(argv[2],"wb");
    if(!f || std::fwrite(output.data(),24,n,f)!=(size_t)n) return 4;
    if(std::fclose(f)) return 5;
    double end=cpu();
    if(std::getenv("FMM_STATS")) std::fprintf(stderr,
        "H=%d C=%d leaf=%d tau=%.6g threshold=%d bodymin=%d nodes=%zu far=%zu near=%zu body=%zu pairs=%lld\n"
        "read+tree %.4f up+scale %.4f tune %.4f pp %.4f body %.4f far %.4f down+io %.4f total %.4f\n"
        "cost pp %.3gns h %.3gus res %.3gus body %.3gus down %.4f predicted %.4f coarse %lld %lld %lld %lld\n",
        H::ORDER,ORDER,leaf_size,tolerance,direct_threshold,body_min,tree.size(),far_pairs.size(),near_pairs.size(),body_pairs.size(),direct_pairs,
        t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,end-t6,end-t0,
        costs.pp*1e9,costs.h*1e6,costs.res*1e6,costs.body*1e6,costs.down,predicted_work(),coarse_near,coarse_body,coarse_h,coarse_res);
    return 0;
}
