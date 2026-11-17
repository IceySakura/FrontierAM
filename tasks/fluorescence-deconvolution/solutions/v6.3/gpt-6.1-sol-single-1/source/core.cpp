#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace {

double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}

using Vec = std::vector<float>;
using DVec = std::vector<double>;
constexpr double pi = 3.14159265358979323846;

struct Conv {
    int z, y, x, n;
    std::array<Vec, 3> p;
    Vec a, b;

    Conv(int zz, int yy, int xx, std::array<Vec, 3> pp)
        : z(zz), y(yy), x(xx), n(zz*yy*xx), p(std::move(pp)), a(n), b(n) {}

    void axis(const Vec& src, Vec& dst, int j, bool adj, bool square) const {
        std::fill(dst.begin(), dst.end(), 0.f);
        int dim = j == 0 ? z : (j == 1 ? y : x);
        int stride = j == 0 ? y*x : (j == 1 ? x : 1);
        int m = int(p[j].size()), mid = m/2;
        for (int k = 0; k < m; ++k) {
            int shift = k - mid;
            float v = p[j][adj ? k : m-1-k];
            if (square) v *= v;
            int lo = std::max(0, -shift), hi = std::min(dim, dim-shift);
            if (j == 0) {
                int begin = lo*y*x, end = hi*y*x, offset = shift*stride;
                for (int i = begin; i < end; ++i) dst[i] += v*src[i+offset];
            } else if (j == 1) {
                for (int zz = 0; zz < z; ++zz) {
                    int begin = (zz*y+lo)*x, end = (zz*y+hi)*x, offset = shift*x;
                    for (int i = begin; i < end; ++i) dst[i] += v*src[i+offset];
                }
            } else {
                for (int row = 0; row < z*y; ++row) {
                    int begin = row*x+lo, end = row*x+hi;
                    for (int i = begin; i < end; ++i) dst[i] += v*src[i+shift];
                }
            }
        }
    }

    void apply(const Vec& src, Vec& dst, bool adj=false, bool square=false) {
        axis(src, a, 0, adj, square);
        axis(a, b, 1, adj, square);
        axis(b, dst, 2, adj, square);
    }
};

struct Noise {
    double sigma, alpha;
    std::array<double, 3> c;
    Noise(double s) : sigma(s), alpha(s*1.62) {
        for (int k = 0; k < 3; ++k) {
            double u = k*alpha/sigma;
            c[k] = sigma*(std::exp(-u*u/2)/std::sqrt(2*pi)
                         - u*.5*std::erfc(u/std::sqrt(2.)));
        }
    }
    inline void response(double b, double& pred, double& deriv) const {
        double v = std::max(b, 0.)/alpha;
        double e = std::exp(-std::min(v, 80.));
        pred = b+e*(c[0]+c[1]*v+.5*c[2]*v*v);
        deriv = 1+e*((c[1]-c[0])+(c[2]-c[1])*v-.5*c[2]*v*v)/alpha;
    }
};

double estimate_noise(const Vec& obs, Conv& conv) {
    std::array<Vec,3> p;
    for (int j=0; j<3; ++j) {
        p[j].resize(13);
        double sum=0;
        for(int k=0;k<13;++k) sum += p[j][k]=float(std::exp(-.5*std::pow((k-6)/1.5,2)));
        for(float& v:p[j]) v/=float(sum);
    }
    Conv smooth(conv.z,conv.y,conv.x,p);
    Vec sm(conv.n);
    smooth.apply(obs,sm);
    float center=p[0][6]*p[1][6]*p[2][6];
    Vec vals;
    for(int z=3;z<conv.z-3;++z) for(int y=3;y<conv.y-3;++y) for(int x=3;x<conv.x-3;++x) {
        int i=(z*conv.y+y)*conv.x+x;
        sm[i]-=center*obs[i];
        vals.push_back(sm[i]);
    }
    if(vals.empty()) return std::max(1e-6, double(std::accumulate(obs.begin(),obs.end(),0.))/conv.n*.1);
    size_t qi=size_t(.35*vals.size());
    std::nth_element(vals.begin(),vals.begin()+qi,vals.end());
    float threshold=vals[qi];
    double sum=0; int count=0;
    for(int z=3;z<conv.z-3;++z) for(int y=3;y<conv.y-3;++y) for(int x=3;x<conv.x-3;++x) {
        int i=(z*conv.y+y)*conv.x+x;
        if(sm[i]<=threshold) {sum+=obs[i];++count;}
    }
    return std::max(1e-6,sum/std::max(count,1)*std::sqrt(2*pi));
}

void laplacian(const Vec& src, Vec& dst, const Conv& c) {
    std::fill(dst.begin(),dst.end(),0.f);
    for(int z=0;z<c.z;++z) for(int y=0;y<c.y;++y) for(int x=0;x<c.x;++x) {
        int i=(z*c.y+y)*c.x+x;
        if(z+1<c.z) {int k=i+c.y*c.x; float d=src[k]-src[i]; dst[k]+=d;dst[i]-=d;}
        if(y+1<c.y) {int k=i+c.x; float d=src[k]-src[i]; dst[k]+=d;dst[i]-=d;}
        if(x+1<c.x) {int k=i+1; float d=src[k]-src[i]; dst[k]+=d;dst[i]-=d;}
    }
}

Vec baseline(const Vec& obs, Conv& conv, const Noise& noise, int count, double deadline) {
    int n=conv.n;
    Vec x(n), z(n), old(n), f(n), grad(n), lap(n);
    double l2=.0025*std::pow(std::max(noise.sigma/.008,.2),1.05);
    l2=std::min(l2,.035);
    double l1=.055*noise.sigma;
    double step=1/(1+12*l2), t=1;
    for(int i=0;i<n;++i) x[i]=z[i]=std::max(obs[i]-float(.5*noise.sigma),0.f);
    for(int k=0;k<count;++k) {
        if(now()>deadline) break;
        conv.apply(z,f);
        for(int i=0;i<n;++i) {
            // A slight extra floor is useful for the initialization's sparsity.
            f[i]-=obs[i]-float(.5*noise.sigma);
        }
        conv.apply(f,grad,true);
        laplacian(z,lap,conv);
        double tn=.5*(1+std::sqrt(1+4*t*t)), momentum=(t-1)/tn;
        for(int i=0;i<n;++i) {
            float v=std::max(float(z[i]-step*(grad[i]+l2*lap[i]+l1)),0.f);
            z[i]=float(v+momentum*(v-x[i]));
            x[i]=v;
        }
        t=tn;
    }
    return x;
}

struct Atom {
    int type; // 0 point, 1 tube, 2 sheet
    int axis=0;
    DVec p, lo, hi;
    int size() const {return int(p.size());}
};

void generate_atom(const Atom& atom, const DVec& p, int off, const Conv& conv, Vec& image, Vec* total) {
    int i=0;
    if(atom.type==0) {
        double r=p[off+3], amp=p[off+4], inv=-.5/(r*r);
        DVec gz(conv.z),gy(conv.y),gx(conv.x);
        for(int z=0;z<conv.z;++z)gz[z]=std::exp((z-p[off])*(z-p[off])*inv);
        for(int y=0;y<conv.y;++y)gy[y]=std::exp((y-p[off+1])*(y-p[off+1])*inv);
        for(int x=0;x<conv.x;++x)gx[x]=std::exp((x-p[off+2])*(x-p[off+2])*inv);
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            float v=float(gz[z]*gy[y]*gx[x]);
            image[i]=v;
            if(total) (*total)[i]+=float(amp*v);
        }
    } else if(atom.type==1) {
        double az=p[off],ay=p[off+1],ax=p[off+2];
        double uz=p[off+3]-az,uy=p[off+4]-ay,ux=p[off+5]-ax;
        double invlen=1/std::max(uz*uz+uy*uy+ux*ux,1e-8);
        double r=p[off+6],amp=p[off+7],inv=-.5/(r*r);
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            double vz=z-az,vy=y-ay,vx=x-ax;
            double t=std::max(0.,std::min(1.,(vz*uz+vy*uy+vx*ux)*invlen));
            double dz=vz-t*uz,dy=vy-t*uy,dx=vx-t*ux;
            float v=float(std::exp((dz*dz+dy*dy+dx*dx)*inv));
            image[i]=v;
            if(total) (*total)[i]+=float(amp*v);
        }
    } else {
        int j=atom.axis, a=(j==0?1:0),b=(j==2?1:2);
        double dims[3]={double(conv.z-1),double(conv.y-1),double(conv.x-1)};
        double r=p[off+1], inv=-.5/(r*r);
        DVec base(int(dims[j])+1);
        for(int k=0;k<int(base.size());++k)base[k]=std::exp((k-p[off])*(k-p[off])*inv);
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            double coord[3]={double(z),double(y),double(x)};
            double u1=2*coord[a]/dims[a]-1,u2=2*coord[b]/dims[b]-1;
            float v=float(base[int(coord[j])]);
            image[i]=v;
            if(total) (*total)[i]+=float(v*std::max(0.,p[off+2]+p[off+3]*u1+p[off+4]*u2));
        }
    }
}

void atom_gradient(const Atom& atom,const DVec& p,int off,const Conv& conv,
                   const Vec& image,const Vec& adj,DVec& grad) {
    int i=0;
    double g[8]={};
    if(atom.type==0) {
        double r=p[off+3],amp=p[off+4],ir2=1/(r*r),ir3=ir2/r;
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            double dz=z-p[off],dy=y-p[off+1],dx=x-p[off+2],v=adj[i]*double(image[i]);
            g[0]+=v*amp*dz*ir2;g[1]+=v*amp*dy*ir2;g[2]+=v*amp*dx*ir2;
            g[3]+=v*amp*(dz*dz+dy*dy+dx*dx)*ir3;g[4]+=v;
        }
    } else if(atom.type==1) {
        double az=p[off],ay=p[off+1],ax=p[off+2];
        double uz=p[off+3]-az,uy=p[off+4]-ay,ux=p[off+5]-ax;
        double invlen=1/std::max(uz*uz+uy*uy+ux*ux,1e-8);
        double r=p[off+6],amp=p[off+7],ir2=1/(r*r),ir3=ir2/r;
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            double vz=z-az,vy=y-ay,vx=x-ax;
            double t=std::max(0.,std::min(1.,(vz*uz+vy*uy+vx*ux)*invlen));
            double dz=vz-t*uz,dy=vy-t*uy,dx=vx-t*ux;
            double v=adj[i]*double(image[i]),s=v*amp*ir2;
            g[0]+=s*dz*(1-t);g[1]+=s*dy*(1-t);g[2]+=s*dx*(1-t);
            g[3]+=s*dz*t;g[4]+=s*dy*t;g[5]+=s*dx*t;
            g[6]+=v*amp*(dz*dz+dy*dy+dx*dx)*ir3;g[7]+=v;
        }
    } else {
        int j=atom.axis,a=(j==0?1:0),b=(j==2?1:2);
        double dims[3]={double(conv.z-1),double(conv.y-1),double(conv.x-1)};
        double r=p[off+1],ir2=1/(r*r),ir3=ir2/r;
        for(int z=0;z<conv.z;++z) for(int y=0;y<conv.y;++y) for(int x=0;x<conv.x;++x,++i) {
            double coord[3]={double(z),double(y),double(x)};
            double d=coord[j]-p[off],u1=2*coord[a]/dims[a]-1,u2=2*coord[b]/dims[b]-1;
            double field=p[off+2]+p[off+3]*u1+p[off+4]*u2;
            if(field<=0) continue;
            double v=adj[i]*double(image[i]);
            g[0]+=v*field*d*ir2;g[1]+=v*field*d*d*ir3;
            g[2]+=v;g[3]+=v*u1;g[4]+=v*u2;
        }
    }
    for(int j=0;j<atom.size();++j) grad[off+j]=g[j]/256.;
}

struct Objective {
    const Vec& obs;
    Conv& conv;
    const Noise& noise;
    const Vec& weight;
    const std::vector<Atom>& atoms;
    Vec fixed,total,blur,resid,adj;
    std::vector<Vec> images;
    double deadline;
    int evals=0;
    double prior=1;
    double amp_prior=0;

    Objective(const Vec& o,Conv& c,const Noise& no,const Vec& w,const std::vector<Atom>& aa,
              const Vec& f,double d)
        :obs(o),conv(c),noise(no),weight(w),atoms(aa),fixed(f),total(c.n),blur(c.n),
         resid(c.n),adj(c.n),images(aa.size(),Vec(c.n)),deadline(d) {
        double density=std::accumulate(obs.begin(),obs.end(),0.)/conv.n-.4*noise.sigma;
        amp_prior=std::max(0.,std::min(1.,(.006-density)/.003));
    }

    double eval(const DVec& p,DVec* g) {
        ++evals;
        total=fixed;
        int off=0;
        for(size_t k=0;k<atoms.size();++k) {
            generate_atom(atoms[k],p,off,conv,images[k],&total);
            off+=atoms[k].size();
        }
        conv.apply(total,blur);
        double value=0;
        for(int i=0;i<conv.n;++i) {
            double pred,dp;
            noise.response(blur[i],pred,dp);
            double r=pred-obs[i];
            value+=.5*r*r*weight[i];
            resid[i]=float(r*weight[i]*dp);
        }
        value/=256.;
        if(g) {
            g->assign(p.size(),0.);
            conv.apply(resid,adj,true);
            off=0;
            for(size_t k=0;k<atoms.size();++k) {
                atom_gradient(atoms[k],p,off,conv,images[k],adj,*g);
                off+=atoms[k].size();
            }
        }
        if(prior>0 || amp_prior>0) {
            off=0;
            for(const Atom& a:atoms) {
                if(a.type==0) {
                    double r=p[off+3],amp=p[off+4],d2=0,d[3];
                    for(int j=0;j<3;++j){d[j]=std::round(p[off+j])-p[off+j];d2+=d[j]*d[j];}
                    double base=std::exp(-d2/(2*r*r)),peak=amp*base;
                    double dr=r-.9;
                    value+=prior*.5*dr*dr/(.3*.3)/256.;
                    if(g)(*g)[off+3]+=prior*dr/(.3*.3)/256.;
                    double da=amp-.85;
                    value+=amp_prior*.5*da*da/(.25*.25)/256.;
                    if(g)(*g)[off+4]+=amp_prior*da/(.25*.25)/256.;
                    if(peak>1.) {
                        double dp=peak-1.,gg=prior*dp/(.07*.07)/256.;
                        value+=prior*.5*dp*dp/(.07*.07)/256.;
                        if(g) {
                            for(int j=0;j<3;++j)(*g)[off+j]+=gg*peak*d[j]/(r*r);
                            (*g)[off+3]+=gg*peak*d2/(r*r*r);
                            (*g)[off+4]+=gg*base;
                        }
                    }
                }
                off+=a.size();
            }
        }
        return value;
    }
};

double dot(const DVec& a,const DVec& b) {
    double s=0;for(size_t i=0;i<a.size();++i)s+=a[i]*b[i];return s;
}

double optimize(Objective& obj,std::vector<Atom>& atoms,int iterations) {
    DVec p,lo,hi,scale;
    for(const Atom& a:atoms) {
        p.insert(p.end(),a.p.begin(),a.p.end());
        lo.insert(lo.end(),a.lo.begin(),a.lo.end());
        hi.insert(hi.end(),a.hi.begin(),a.hi.end());
        for(int j=0;j<a.size();++j) {
            bool coord=(a.type==0&&j<3)||(a.type==1&&j<6)||(a.type==2&&j==0);
            bool radius=(a.type==0&&j==3)||(a.type==1&&j==6)||(a.type==2&&j==1);
            scale.push_back(coord?2.:(radius?.3:.5));
        }
    }
    int d=int(p.size());
    if(!d) return 0;
    DVec H(d*d,0),g(d),gnew(d),dir(d),trial(d),s(d),v(d),Hv(d);
    auto reset=[&]() {std::fill(H.begin(),H.end(),0.);for(int i=0;i<d;++i)H[i*d+i]=scale[i]*scale[i]*.15;};
    reset();
    double value=obj.eval(p,&g);
    for(int iter=0;iter<iterations;++iter) {
        if(now()>obj.deadline) break;
        for(int i=0;i<d;++i) {
            dir[i]=0;
            for(int j=0;j<d;++j) dir[i]-=H[i*d+j]*g[j];
            if((p[i]<=lo[i]+1e-9&&dir[i]<0)||(p[i]>=hi[i]-1e-9&&dir[i]>0))dir[i]=0;
        }
        if(dot(g,dir)>=-1e-12) {
            reset();
            for(int i=0;i<d;++i) {
                dir[i]=-H[i*d+i]*g[i];
                if((p[i]<=lo[i]+1e-9&&dir[i]<0)||(p[i]>=hi[i]-1e-9&&dir[i]>0))dir[i]=0;
            }
        }
        double ratio=1;
        for(int i=0;i<d;++i) ratio=std::max(ratio,std::abs(dir[i])/(scale[i]*.65));
        for(double& v:dir)v/=ratio;
        double gd=dot(g,dir);
        if(gd>=-1e-11)break;
        double step=1,newvalue=value;
        bool accepted=false;
        for(int ls=0;ls<14;++ls) {
            if(now()>obj.deadline)break;
            for(int i=0;i<d;++i)trial[i]=std::max(lo[i],std::min(hi[i],p[i]+step*dir[i]));
            newvalue=obj.eval(trial,&gnew);
            if(std::isfinite(newvalue)&&newvalue<=value+1e-4*step*gd) {accepted=true;break;}
            step*=.5;
        }
        if(!accepted)break;
        for(int i=0;i<d;++i){s[i]=trial[i]-p[i];v[i]=gnew[i]-g[i];}
        double sy=dot(s,v);
        if(sy>1e-12) {
            for(int i=0;i<d;++i){Hv[i]=0;for(int j=0;j<d;++j)Hv[i]+=H[i*d+j]*v[j];}
            double yhy=dot(v,Hv),f=(sy+yhy)/(sy*sy);
            for(int i=0;i<d;++i)for(int j=0;j<d;++j)
                H[i*d+j]+=f*s[i]*s[j]-(Hv[i]*s[j]+s[i]*Hv[j])/sy;
        }
        double improvement=value-newvalue;
        p=trial;g=gnew;value=newvalue;
        if(iter>12 && improvement<1e-8*std::max(1.,value)) break;
    }
    int off=0;
    for(Atom& a:atoms) {for(double& v:a.p)v=p[off++];}
    obj.eval(p,nullptr);
    return value;
}

void render(const std::vector<Atom>& atoms,Conv& conv,Vec& out) {
    std::fill(out.begin(),out.end(),0.f);
    Vec image(conv.n);
    for(const Atom& a:atoms)generate_atom(a,a.p,0,conv,image,&out);
}

void eig3(double a[3][3],double values[3],double vectors[3][3]) {
    for(int j=0;j<3;++j)for(int k=0;k<3;++k)vectors[j][k]=(j==k?1.:0.);
    for(int iter=0;iter<24;++iter) {
        int p=0,q=1;
        if(std::abs(a[0][2])>std::abs(a[p][q])){p=0;q=2;}
        if(std::abs(a[1][2])>std::abs(a[p][q])){p=1;q=2;}
        if(std::abs(a[p][q])<1e-12)break;
        double phi=.5*std::atan2(2*a[p][q],a[q][q]-a[p][p]),c=std::cos(phi),s=std::sin(phi);
        double app=a[p][p],aqq=a[q][q],apq=a[p][q];
        a[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
        a[q][q]=s*s*app+2*c*s*apq+c*c*aqq;
        a[p][q]=a[q][p]=0;
        for(int k=0;k<3;++k)if(k!=p&&k!=q) {
            double ap=a[k][p],aq=a[k][q];
            a[k][p]=a[p][k]=c*ap-s*aq;
            a[k][q]=a[q][k]=s*ap+c*aq;
        }
        for(int k=0;k<3;++k){double vp=vectors[k][p],vq=vectors[k][q];vectors[k][p]=c*vp-s*vq;vectors[k][q]=s*vp+c*vq;}
    }
    std::array<int,3> order={0,1,2};
    std::sort(order.begin(),order.end(),[&](int j,int k){return a[j][j]<a[k][k];});
    double tmp[3][3];
    for(int j=0;j<3;++j){values[j]=a[order[j]][order[j]];for(int k=0;k<3;++k)tmp[k][j]=vectors[k][order[j]];}
    for(int j=0;j<3;++j)for(int k=0;k<3;++k)vectors[j][k]=tmp[j][k];
}

Atom make_atom(bool tube,const double center[3],const double direction[3],Conv& conv,double margin=0.) {
    Atom a;
    a.type=tube?1:0;
    double dims[3]={double(conv.z-1),double(conv.y-1),double(conv.x-1)};
    if(tube) {
        for(int j=0;j<3;++j)a.p.push_back(std::max(0.,std::min(dims[j],center[j]-7*direction[j])));
        for(int j=0;j<3;++j)a.p.push_back(std::max(0.,std::min(dims[j],center[j]+7*direction[j])));
        a.p.push_back(1.1);a.p.push_back(.6);
        for(int j=0;j<6;++j){a.lo.push_back(std::max(-2.,a.p[j]-7));a.hi.push_back(std::min(dims[j%3]+2,a.p[j]+7));}
        a.lo.insert(a.lo.end(),{.65,0.});a.hi.insert(a.hi.end(),{2.,1.3});
    } else {
        for(int j=0;j<3;++j) {
            double global_low=margin?margin:-2.;
            double global_high=margin?dims[j]+1-margin:dims[j]+2;
            double initial=std::max(global_low,std::min(global_high,center[j]));
            double low=std::max(global_low,initial-3);
            double high=std::min(global_high,initial+3);
            a.lo.push_back(low);a.hi.push_back(high);
            a.p.push_back(initial);
        }
        a.p.push_back(.9);a.p.push_back(.6);
        a.lo.insert(a.lo.end(),{.6,0.});a.hi.insert(a.hi.end(),{1.6,2.});
    }
    return a;
}

void reconstruct(const Vec& obs,Conv& conv,double deadline,Vec& output,double* diag) {
    double start=now();
    double sigma=estimate_noise(obs,conv);
    Noise noise(sigma);
    double density=std::accumulate(obs.begin(),obs.end(),0.)/conv.n-.4*sigma;
    double margin=2*std::max(0.,std::min(1.,(.006-density)/.003));
    double detection=(density<.006 && sigma>.015)?6.:4.5;
    double separation=3.;
    Vec init=baseline(obs,conv,noise,40,std::min(deadline,start+.13));
    output=init;
    if(now()>deadline-.025)return;
    Vec weight(conv.n),blur(conv.n),score(conv.n),sd(conv.n),res(conv.n),xx(conv.n,0.f),zero(conv.n,0.f);
    conv.apply(init,blur);
    for(int i=0;i<conv.n;++i)weight[i]=float(1/std::max(noise.alpha*blur[i]+sigma*sigma,sigma*sigma));
    std::vector<Atom> atoms;
    int dims[3]={conv.z,conv.y,conv.x};
    for(int j=0;j<3;++j) {
        Vec profile(dims[j]),slice;
        for(int k=0;k<dims[j];++k) {
            slice.clear();
            for(int z=0;z<conv.z;++z)for(int y=0;y<conv.y;++y)for(int x=0;x<conv.x;++x) {
                int coord[3]={z,y,x};
                if(coord[j]==k)slice.push_back(init[(z*conv.y+y)*conv.x+x]);
            }
            size_t q=size_t(.35*slice.size());
            std::nth_element(slice.begin(),slice.begin()+q,slice.end());
            profile[k]=slice[q];
        }
        for(int k=0;k<dims[j];++k)if(profile[k]>.06) {
            bool peak=true;
            for(int l=std::max(0,k-3);l<=std::min(dims[j]-1,k+3);++l)if(profile[l]>profile[k])peak=false;
            if(!peak)continue;
            Atom a;a.type=2;a.axis=j;a.p={double(k),.8,.6,0.,0.};
            a.lo={-2.,.4,0.,-1.,-1.};a.hi={double(dims[j]+1),2.5,2.,1.,1.};
            atoms.push_back(a);
        }
    }
    int evals=0;
    if(!atoms.empty()) {
        Objective obj(obs,conv,noise,weight,atoms,zero,deadline);
        optimize(obj,atoms,65);evals+=obj.evals;
        render(atoms,conv,xx);
    }
    int added=0;
    for(int iteration=0;iteration<12;++iteration) {
        if(now()>deadline-.018)break;
        conv.apply(xx,blur);
        for(int i=0;i<conv.n;++i) {
            double pred,dp;noise.response(blur[i],pred,dp);
            res[i]=float(obs[i]-pred);
            sd[i]=float(noise.alpha*blur[i]+.340845*sigma*sigma);
        }
        conv.apply(res,score,true);
        conv.apply(sd,res,false,true);
        int peak=0;double maxscore=-1e30;
        for(int i=0;i<conv.n;++i) {
            double v=score[i]/std::sqrt(std::max(double(res[i]),1e-20));
            if(separation>0 && density<.006) {
                double pos[3]={double(i/(conv.y*conv.x)),double((i/conv.x)%conv.y),double(i%conv.x)};
                for(const Atom& a:atoms)if(a.type==0) {
                    double d2=0;
                    for(int j=0;j<3;++j)d2+=std::pow(pos[j]-a.p[j],2);
                    if(d2<separation*separation){v=-1e30;break;}
                }
            }
            if(v>maxscore){maxscore=v;peak=i;}
        }
        if(maxscore<detection)break;
        double seed[3]={double(peak/(conv.y*conv.x)),double((peak/conv.x)%conv.y),double(peak%conv.x)};
        double sum=0,center[3]={},cov[3][3]={};
        double threshold=.1*std::max(double(init[peak]-xx[peak]),.05);
        for(int z=std::max(0,int(seed[0])-7);z<=std::min(conv.z-1,int(seed[0])+7);++z)
        for(int y=std::max(0,int(seed[1])-7);y<=std::min(conv.y-1,int(seed[1])+7);++y)
        for(int x=std::max(0,int(seed[2])-7);x<=std::min(conv.x-1,int(seed[2])+7);++x) {
            double coord[3]={double(z),double(y),double(x)};
            double d2=0;for(int j=0;j<3;++j)d2+=(coord[j]-seed[j])*(coord[j]-seed[j]);
            if(d2>=49)continue;
            int i=(z*conv.y+y)*conv.x+x;
            double w=std::max(double(init[i]-xx[i])-threshold,0.);
            sum+=w;
            for(int j=0;j<3;++j){center[j]+=w*coord[j];for(int k=0;k<3;++k)cov[j][k]+=w*coord[j]*coord[k];}
        }
        if(sum>1e-12) {
            for(double& v:center)v/=sum;
            for(int j=0;j<3;++j)for(int k=0;k<3;++k)cov[j][k]=cov[j][k]/sum-center[j]*center[k];
        } else {for(int j=0;j<3;++j){center[j]=seed[j];cov[j][j]=1;}}
        double eig[3],vectors[3][3];eig3(cov,eig,vectors);
        bool tube=eig[2]>(iteration==0?1.55:2.5)*std::max(eig[1],.3)&&eig[2]>7;
        double direction[3]={vectors[0][2],vectors[1][2],vectors[2][2]};
        std::vector<Atom> active={make_atom(tube,tube?center:seed,direction,conv,margin)};
        Objective obj(obs,conv,noise,weight,active,xx,deadline-.007);
        DVec initialp=active[0].p;
        double old=obj.eval(initialp,nullptr);
        double value=optimize(obj,active,60);
        evals+=obj.evals;
        if(tube && (active[0].p.back()<.08 || old-value<.03)) {
            active={make_atom(false,seed,direction,conv,margin)};
            Objective retry(obs,conv,noise,weight,active,xx,deadline-.007);
            optimize(retry,active,45);evals+=retry.evals;
        }
        if(!tube && eig[2]>2.2*std::max(eig[1],.3)
            && eig[2]>2.0&&eig[2]<5.0&&now()<deadline-.09) {
            double offset=std::min(3.5,std::sqrt(std::max(eig[2]-eig[0],1.)));
            double ca[3],cb[3];
            for(int j=0;j<3;++j){ca[j]=center[j]-offset*direction[j];cb[j]=center[j]+offset*direction[j];}
            std::vector<Atom> pair={make_atom(false,ca,direction,conv,margin),make_atom(false,cb,direction,conv,margin)};
            for(Atom& a:pair){a.p[3]=.8;a.p[4]=.65;}
            Objective paired(obs,conv,noise,weight,pair,xx,deadline-.015);
            double pairvalue=optimize(paired,pair,65);evals+=paired.evals;
            double separation=0;
            for(int j=0;j<3;++j)separation+=std::pow(pair[0].p[j]-pair[1].p[j],2);
            if(pairvalue<value-.016 && separation>1.5*1.5
                && pair[0].p[4]>.2 && pair[1].p[4]>.2)active=pair;
        }
        for(const Atom& a:active)atoms.push_back(a);
        added+=int(active.size());
        render(atoms,conv,xx);
        if(atoms.size()>1&&now()<deadline-.03) {
            Objective joint(obs,conv,noise,weight,atoms,zero,deadline-.007);
            optimize(joint,atoms,iteration<3?25:16);evals+=joint.evals;
            render(atoms,conv,xx);
        }
    }
    if(!atoms.empty()&&now()<deadline-.02) {
        Objective joint(obs,conv,noise,weight,atoms,zero,deadline-.007);
        optimize(joint,atoms,55);evals+=joint.evals;
        render(atoms,conv,xx);
    }
    if(!atoms.empty()) {
        output=xx;
        // Preserve statistically significant regions not yet explained by the fit.
        conv.apply(xx,blur);
        for(int i=0;i<conv.n;++i) {
            double pred,dp;noise.response(blur[i],pred,dp);
            res[i]=float(obs[i]-pred);
            sd[i]=float(noise.alpha*blur[i]+.340845*sigma*sigma);
        }
        conv.apply(res,score,true);
        conv.apply(sd,res,false,true);
        for(int i=0;i<conv.n;++i) {
            double zs=score[i]/std::sqrt(std::max(double(res[i]),1e-20));
            double gate=std::max(0.,std::min(1.,(zs-7)/10));
            output[i]+=float(gate*std::max(init[i]-xx[i],0.f));
        }
    }
    diag[0]=sigma;diag[1]=noise.alpha;diag[2]=atoms.size();diag[3]=evals;diag[4]=now()-start;diag[5]=added;
}
}

extern "C" int solve(const float* observed,const float* psf,int z,int y,int x,
                     int hz,int hy,int hx,double deadline,float* output,double* diag) {
    try {
        int n=z*y*x;
        Vec obs(observed,observed+n);
        std::array<Vec,3> p={Vec(hz,0.f),Vec(hy,0.f),Vec(hx,0.f)};
        for(int k=0;k<hz;++k)for(int j=0;j<hy;++j)for(int i=0;i<hx;++i) {
            float v=psf[(k*hy+j)*hx+i];p[0][k]+=v;p[1][j]+=v;p[2][i]+=v;
        }
        double err=0;
        for(int k=0;k<hz;++k)for(int j=0;j<hy;++j)for(int i=0;i<hx;++i) {
            double v=p[0][k]*double(p[1][j])*p[2][i];
            err=std::max(err,std::abs(v-psf[(k*hy+j)*hx+i]));
        }
        if(err>1e-5)return 2;
        Conv conv(z,y,x,p);
        Vec out(n);
        reconstruct(obs,conv,deadline,out,diag);
        for(int i=0;i<n;++i)output[i]=std::isfinite(out[i])?std::max(out[i],0.f):0.f;
        return 0;
    } catch(...) {return 1;}
}
