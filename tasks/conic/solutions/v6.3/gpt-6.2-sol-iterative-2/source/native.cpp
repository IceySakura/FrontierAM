#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <sys/time.h>
#include <unistd.h>

namespace {

struct Ldl {
    int n;
    std::vector<int> perm, ptr, row, map, parent, lp, li, count, flag, pattern, path;
    std::vector<double> d, sign, lx, y, work;
    bool usable = true;

    Ldl(int n_, const int* ap, const int* ai, const int* order,
        const double* signs, int maxfill, double maxops)
        : n(n_), perm(order,order+n), ptr(n+1,0), parent(n,-1), lp(n+1,0),
          count(n,0), flag(n,-1), pattern(n), path(n), d(n), sign(n), y(n), work(n) {
        for (int j=0; j<n; ++j) {
            const int jc=perm[j];
            sign[jc]=signs[j];
            for (int t=ap[j]; t<ap[j+1]; ++t)
                if (perm[ai[t]]<=jc) ++ptr[jc+1];
        }
        for (int j=0; j<n; ++j) ptr[j+1]+=ptr[j];
        row.resize(ptr[n]); map.resize(ptr[n]);
        std::vector<int> next(ptr);
        for (int j=0; j<n; ++j) {
            const int jc=perm[j];
            for (int t=ap[j]; t<ap[j+1]; ++t) {
                int i=perm[ai[t]];
                if (i<=jc) {
                    int off=next[jc]++;
                    row[off]=i; map[off]=t;
                }
            }
        }
        // Elimination-tree reach gives the column counts without numeric fill.
        int64_t fill=0;
        double ops=0;
        for (int k=0; k<n; ++k) {
            flag[k]=k;
            for (int t=ptr[k]; t<ptr[k+1]; ++t) {
                for (int i=row[t]; flag[i]!=k; i=parent[i]) {
                    if (parent[i]<0) parent[i]=k;
                    ops += count[i]+1;
                    ++count[i]; ++fill;
                    flag[i]=k;
                }
            }
            if (fill>maxfill || ops>maxops) {
                usable=false;
                return;
            }
        }
        for (int i=0; i<n; ++i) lp[i+1]=lp[i]+count[i];
        li.resize(fill); lx.resize(fill);
    }

    bool factor(const double* a) {
        std::fill(count.begin(),count.end(),0);
        std::fill(flag.begin(),flag.end(),-1);
        std::fill(y.begin(),y.end(),0.);
        int modifications=0;
        for (int k=0; k<n; ++k) {
            int top=n;
            flag[k]=k;
            for (int t=ptr[k]; t<ptr[k+1]; ++t) {
                int i=row[t];
                y[i]+=a[map[t]];
                int len=0;
                for (; flag[i]!=k; i=parent[i]) {
                    path[len++]=i;
                    flag[i]=k;
                }
                while (len) pattern[--top]=path[--len];
            }
            double dk=y[k]; y[k]=0.;
            for (int j=top; j<n; ++j) {
                const int i=pattern[j], end=lp[i]+count[i];
                const double yi=y[i];
                y[i]=0.;
                for (int t=lp[i]; t<end; ++t) y[li[t]] -= lx[t]*yi;
                const double lki=yi*d[i];
                dk-=lki*yi;
                li[end]=k; lx[end]=lki; ++count[i];
            }
            if (!std::isfinite(dk) || std::abs(dk)>1.e20 || sign[k]*dk < -1.e-6)
                return false;
            if (dk*sign[k]<8.e-15) {
                if (++modifications>std::max(25,n/10)) return false;
                dk=sign[k]*1.e-11;
            }
            d[k]=1./dk;
        }
        return true;
    }

    void solve(const double* b, double* x) {
        auto* v=work.data();
        for (int k=0;k<n;++k) v[perm[k]]=b[k];
        for (int k=0;k<n;++k) {
            const double z=v[k];
            for (int t=lp[k];t<lp[k+1];++t) v[li[t]]-=lx[t]*z;
        }
        for (int k=0;k<n;++k) v[k]*=d[k];
        for (int k=n-1;k>=0;--k) {
            double z=v[k];
            for (int t=lp[k];t<lp[k+1];++t) z-=lx[t]*v[li[t]];
            v[k]=z;
        }
        for (int k=0;k<n;++k) x[k]=v[perm[k]];
    }
};

void exp_project(double* p) {
    const double scale=std::max({std::abs(p[0]),std::abs(p[1]),std::abs(p[2])});
    if (scale==0. || !std::isfinite(scale)) return;
    const double a=p[0]/scale, b=p[1]/scale, c=p[2]/scale;
    if ((b==0. && a<=0. && c>=0.) ||
        (b>0. && c>0. && a<=b*(std::log(c)-std::log(b)))) return;
    if ((a==0. && b<=0. && c<=0.) ||
        (a>0. && c<0. && b<=a*(1.+std::log(-c)-std::log(a)))) {
        p[0]=p[1]=p[2]=0.;
        return;
    }
    static constexpr double grid[]={
        -80.,-40.,-24.,-16.,-10.,-7.,-5.,-4.,-3.,-2.,-1.,-.5,0.,
        .5,1.,1.5,2.,2.5,3.,4.,5.,6.,7.,9.,12.,16.,22.,32.,48.
    };
    auto score=[&](double t) {
        if (t<=0.) {
            const double e=std::exp(t);
            return (a*t+b+c*e)/std::sqrt(t*t+1.+e*e);
        }
        const double e=std::exp(-t), f=t*e;
        return (a*f+b*e+c)/std::sqrt(1.+e*e+f*f);
    };
    const double floorA=std::min(a,0.), floorC=std::max(c,0.);
    double best=std::sqrt(floorA*floorA+floorC*floorC);
    double pos=0., mid=-1.e100;
    int idx=0;
    constexpr int n=sizeof(grid)/sizeof(*grid);
    for (int j=0;j<n;++j) {
        double f=score(grid[j]);
        if (f>mid) { mid=f; idx=j; }
    }
    if (mid>best) { best=mid; pos=grid[idx]; }
    else pos=1.e100;
    const double left=grid[std::max(0,idx-1)], right=grid[std::min(n-1,idx+1)];
    double lo=left, hi=right;
    // Use the derivative of the normalized directional support.
    auto slope=[&](double t) {
        if (t<=0.) {
            const double e=std::exp(t);
            return a-b*t+c*e*(t*t-t+1.)+e*e*(a*(1.-t)-b);
        }
        const double e=std::exp(-t);
        return e*e*(a-b*t)+c*e*(t*t-t+1.)+a*(1.-t)-b;
    };
    if (slope(lo)>0. && slope(hi)<0.) {
        for (int k=0;k<42;++k) {
            const double t=.5*(lo+hi);
            if (slope(t)>0.) lo=t; else hi=t;
        }
        const double t=.5*(lo+hi), f=score(t);
        if (f>=best-2.*std::numeric_limits<double>::epsilon() && f>0.) {
            best=f; pos=t;
        }
    }
    if (a<0. && b>0. && a/b<-70.) {
        const double t=std::max(-1.e15,a/b);
        double f=score(t);
        if (f>=best && f>0.) { best=f; pos=t; }
    }
    if (pos>100.) {
        p[0]=scale*floorA;
        p[1]=0.;
        p[2]=scale*floorC;
    } else if (pos<=0.) {
        const double e=std::exp(pos);
        const double r=scale*best/std::sqrt(pos*pos+1.+e*e);
        p[0]=pos*r; p[1]=r;
        p[2]=std::max(r*e,std::numeric_limits<double>::min());
    } else {
        const double e=std::exp(-pos), f=pos*e;
        const double r=scale*best/std::sqrt(1.+e*e+f*f);
        p[0]=f*r; p[1]=e*r; p[2]=r;
    }
}

}

extern "C" {

void conic_watchdog(int milliseconds) {
    struct sigaction a{};
    a.sa_handler=[](int) { _exit(0); };
    sigemptyset(&a.sa_mask);
    sigaction(SIGALRM,&a,nullptr);
    itimerval timer{};
    if (milliseconds>0) {
        timer.it_value.tv_sec=milliseconds/1000;
        timer.it_value.tv_usec=(milliseconds%1000)*1000;
    }
    setitimer(ITIMER_REAL,&timer,nullptr);
}

void* conic_ldl_create(int n, const int* ap, const int* ai, const int* order,
                       const double* signs, int maxfill, double maxops) {
    try {
        auto p=std::make_unique<Ldl>(n,ap,ai,order,signs,maxfill,maxops);
        return p->usable ? p.release() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

int conic_ldl_factor(void* self,const double* a) {
    return static_cast<Ldl*>(self)->factor(a);
}

void conic_ldl_solve(void* self,const double* b,double* x) {
    static_cast<Ldl*>(self)->solve(b,x);
}

void conic_ldl_free(void* self) {
    delete static_cast<Ldl*>(self);
}

void conic_schur(int n,int d,const int* ptr,const int* ri,const int* rj,
                 const double* values,const double* g,double* h) {
    for (int j=0;j<n;++j) {
        for (int i=0;i<=j;++i) {
            double v=0.;
            for (int p=ptr[j];p<ptr[j+1];++p) {
                const double* a=g+std::size_t(ri[p])*d;
                const double* b=g+std::size_t(rj[p])*d;
                double s=0.;
                for (int q=ptr[i];q<ptr[i+1];++q) {
                    const int k=ri[q], l=rj[q];
                    s+=values[q]*(a[k]*b[l]+a[l]*b[k]);
                }
                v+=values[p]*s;
            }
            h[std::size_t(i)*n+j]=h[std::size_t(j)*n+i]=v*.5;
        }
    }
}

void conic_psd_columns(int n,int d,const int* ptr,const int* ri,const int* rj,
                       const double* values,const double* rtr,double* h) {
    const int m=d*(d+1)/2;
    constexpr double root2=1.4142135623730950488;
    for (int j=0;j<n;++j) {
        double* hj=h+std::size_t(j)*m;
        std::fill(hj,hj+m,0.);
        for (int p=ptr[j];p<ptr[j+1];++p) {
            const int i=ri[p], j=rj[p];
            const double* a=rtr+std::size_t(i)*d;
            const double* b=rtr+std::size_t(j)*d;
            const double val=values[p];
            int at=0;
            if (i==j) {
                for (int l=0;l<d;++l) {
                    const double e=val*a[l];
                    for (int k=0;k<=l;++k) hj[at+k]+=e*a[k];
                    at+=l+1;
                }
            } else {
                for (int l=0;l<d;++l) {
                    const double e=val*a[l], f=val*b[l];
                    for (int k=0;k<=l;++k) hj[at+k]+=e*b[k]+f*a[k];
                    at+=l+1;
                }
            }
        }
        for (int l=0,at=0;l<d;++l) {
            for (int k=0;k<l;++k) hj[at+k]*=root2;
            at+=l+1;
        }
    }
}

void conic_project_exp(int n,double* x) {
    for (int j=0;j<n;++j) exp_project(x+3*std::size_t(j));
}

void conic_gram(int m,int lda,const int* ptr,const int* idx,const double* val,
                const double* weight,double* mat) {
    for (int i=0;i<m;++i) {
        for (int p=ptr[i];p<ptr[i+1];++p) {
            const auto j=idx[p];
            double* col=mat+std::size_t(j)*lda;
            const double a=val[p]*weight[i];
            for (int q=p;q<ptr[i+1];++q)
                col[idx[q]]+=a*val[q];
        }
    }
}

int conic_eq_pivots(int n,int count,const int* eq,const int* rp,const int* ci,
                    const double* a,const int* ap,const int* pp,int* piv) {
    try {
        std::vector<unsigned char> mark(n,0);
        int chosen=0;
        for (int k=0;k<count;++k) {
            const int r=eq[k], start=rp[r], end=rp[r+1], nnz=end-start;
            piv[k]=-1;
            if (nnz<2 || nnz>128) continue;
            double biggest=0.;
            bool usable=true;
            for (int t=start;t<end;++t) {
                biggest=std::max(biggest,std::abs(a[t]));
                if (mark[ci[t]]==1) usable=false;
            }
            if (!usable) continue;
            int best=-1, cost=99999999;
            for (int t=start;t<end;++t) {
                int j=ci[t];
                if (mark[j] || std::abs(a[t])<.14*biggest) continue;
                const int da=ap[j+1]-ap[j], dp=pp[j+1]-pp[j];
                if (nnz>2 && (da>1 || dp)) continue;
                if (nnz==2 && da+dp>60) continue;
                const int c=da+3*dp;
                if (c<cost || (c==cost && best>=0 && std::abs(a[t])>std::abs(a[best]))) {
                    best=t;
                    cost=c;
                }
            }
            if (best<0) continue;
            for (int t=start;t<end;++t) mark[ci[t]]= (t==best ? 1 : 2);
            piv[k]=best;
            ++chosen;
        }
        return chosen;
    } catch (...) {
        return 0;
    }
}

void conic_scatter(int n,const int* slot,const int* row,const double* a,
                   const double* w,double* dst) {
    for (int k=0;k<n;++k) dst[slot[k]]+=a[k]*w[row[k]];
}

int conic_group_rows(int m,const int* ptr,const int* idx,const double* a,
                     int* which,int* chosen,double* mul) {
    try {
        using H=std::uint64_t;
        std::unordered_multimap<H,int> table;
        table.reserve(m);
        int g=0;
        auto combine=[](H h,H a) {
            return (h^(a+UINT64_C(0x9e3779b97f4a7c15)+(h<<6)+(h>>2)))*UINT64_C(1099511628211);
        };
        for (int i=0;i<m;++i) {
            const int begin=ptr[i], end=ptr[i+1], len=end-begin;
            which[i]=-1; mul[i]=1.;
            if (!len) {
                which[i]=g;
                chosen[g++]=i;
                continue;
            }
            const double p=a[begin];
            H h=static_cast<H>(len)*UINT64_C(11400714819323198485);
            bool finite=true;
            for (int k=begin;k<end;++k) {
                double x=a[k]/p;
                std::uint64_t bits=0;
                std::memcpy(&bits,&x,sizeof(bits));
                finite &= std::isfinite(x);
                h=combine(combine(h,idx[k]),bits);
            }
            if (finite) {
                auto same=table.equal_range(h);
                for (auto it=same.first;it!=same.second;++it) {
                    const int j=it->second,t=ptr[j];
                    if (ptr[j+1]-t!=len) continue;
                    const double f=p/a[t];
                    if (f==0. || !std::isfinite(f)) continue;
                    bool ok=true;
                    for (int r=0;r<len;++r)
                        if (idx[begin+r]!=idx[t+r] || a[begin+r]/p!=a[t+r]/a[t]) {
                            ok=false;
                            break;
                        }
                    if (ok) {
                        which[i]=which[j];
                        mul[i]=f;
                        break;
                    }
                }
            }
            if (which[i]<0) {
                which[i]=g;
                chosen[g++]=i;
                if (finite) table.emplace(h,i);
            }
        }
        return g;
    } catch (...) {
        return 0;
    }
}

}
