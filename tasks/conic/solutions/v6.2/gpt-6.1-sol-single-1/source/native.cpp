#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <limits>
#include <chrono>

extern "C" {
// Hessian of -logdet in the isometric upper-triangle coordinates.
void psd_schur(int n, int d, int nz, const int32_t *ri,
               const int32_t *rj, const int32_t *col, const double *val,
               const double *r, double *out) {
    const double rt2 = std::sqrt(2.0);
    for (int a = 0; a < nz; ++a) {
        const int i = ri[a], j = rj[a], ca = col[a];
        const double va = val[a] * (i == j ? 1.0 : rt2);
        double *dest = out + static_cast<size_t>(ca) * n;
        for (int b = 0; b < nz; ++b) {
            const int k = ri[b], l = rj[b];
            const double vb = val[b] * (k == l ? 1.0 : rt2);
            dest[col[b]] += 0.5 * va * vb *
                (r[i*d+k]*r[j*d+l] + r[i*d+l]*r[j*d+k]);
        }
    }
}

int select_pivots(int n, int m, const int32_t *ptr, const int32_t *row,
                  const double *value, const int32_t *row_degree,
                  const int32_t *col_degree, const uint8_t *allowed,
                  const double *q, int maxdegree, int32_t *pcols,
                  int32_t *prows, double *pvalues) {
    std::vector<int> candidates;
    std::vector<uint8_t> blocked(m,0), selected(m,0);
    for (int j=0; j<n; ++j) {
        int deg=ptr[j+1]-ptr[j];
        if (allowed[j] && deg>0 && deg<=maxdegree)
            candidates.push_back(j);
    }
    std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b) {
        int sa=col_degree[a]+3*(ptr[a+1]-ptr[a])+(q[a]!=0.0 ? 1000 : 0);
        int sb=col_degree[b]+3*(ptr[b+1]-ptr[b])+(q[b]!=0.0 ? 1000 : 0);
        return sa<sb;
    });
    int count=0;
    for (int j:candidates) {
        bool conflict=false;
        int best=-1;
        double score=1e100;
        double colmax=0.0;
        for (int k=ptr[j]; k<ptr[j+1]; ++k)
            colmax=std::max(colmax,std::abs(value[k]));
        for (int k=ptr[j]; k<ptr[j+1]; ++k) {
            int r=row[k];
            if (selected[r]) { conflict=true; break; }
            if (blocked[r] || std::abs(value[k])<1e-8 ||
                std::abs(value[k])>1e8 || std::abs(value[k])<0.5*colmax ||
                row_degree[r]>8) continue;
            double sc=row_degree[r] + 0.01/std::abs(value[k]);
            if (sc<score) { score=sc; best=k; }
        }
        if (conflict || best<0) continue;
        pcols[count]=j;
        prows[count]=row[best];
        pvalues[count]=value[best];
        ++count;
        selected[row[best]]=1;
        for (int k=ptr[j]; k<ptr[j+1]; ++k) blocked[row[k]]=1;
    }
    return count;
}
}

struct LDLBase {
    virtual ~LDLBase() = default;
    virtual bool factor(const double *,bool) = 0;
    virtual void solve(const double *,double *) = 0;
    virtual void residual(const double *,const double *,const double *,double *) = 0;
    virtual int64_t nonzeros() const = 0;
};

template<class Real> struct SparseLDL : LDLBase {
    int n;
    std::vector<int32_t> perm, ap, ai, mapping, parent, flag, pattern, used;
    std::vector<int64_t> lp;
    std::vector<int32_t> li;
    std::vector<Real> ax, lx, diag, y, work;
    SparseLDL(int n_, const int32_t *op, const int32_t *oi, const int32_t *p)
        : n(n_), perm(p,p+n), ap(n+1,0), mapping(op[n],-1),
          parent(n,-1), flag(n,-1), pattern(n), used(n), lp(n+1,0),
          diag(n),y(n),work(n) {
        std::vector<int32_t> inverse(n);
        for (int j=0;j<n;++j) inverse[perm[j]]=j;
        for (int j=0;j<n;++j)
            for (int k=op[j];k<op[j+1];++k)
                if (inverse[oi[k]]<=inverse[j]) ++ap[inverse[j]+1];
        for (int j=0;j<n;++j) ap[j+1]+=ap[j];
        ai.resize(ap[n]);
        ax.resize(ap[n]);
        std::vector<int32_t> next=ap;
        for (int j=0;j<n;++j) {
            int col=inverse[j];
            for (int k=op[j];k<op[j+1];++k) {
                int row=inverse[oi[k]];
                if (row<=col) {
                    int pos=next[col]++;
                    ai[pos]=row;
                    mapping[k]=pos;
                }
            }
        }
        std::vector<int64_t> counts(n,0);
        for (int k=0;k<n;++k) {
            flag[k]=k;
            for (int p0=ap[k];p0<ap[k+1];++p0) {
                int i=ai[p0];
                if (i>=k) continue;
                while (flag[i]!=k) {
                    if (parent[i]<0) parent[i]=k;
                    ++counts[i];
                    flag[i]=k;
                    i=parent[i];
                }
            }
        }
        long double flops=0;
        for (int i=0;i<n;++i) {
            lp[i+1]=lp[i]+counts[i];
            flops+=static_cast<long double>(counts[i])*counts[i];
        }
        if (lp[n]>6000000 || flops>2e7L) throw std::bad_alloc();
        li.resize(lp[n]);
        lx.resize(lp[n]);
    }
    bool factor(const double *values, bool enforce_sign) override {
        for (size_t k=0;k<mapping.size();++k)
            if (mapping[k]>=0) ax[mapping[k]]=values[k];
        std::fill(flag.begin(),flag.end(),-1);
        std::fill(used.begin(),used.end(),0);
        std::fill(y.begin(),y.end(),0.0);
        for (int k=0;k<n;++k) {
            int top=n;
            Real expected=1;
            flag[k]=k;
            for (int p=ap[k];p<ap[k+1];++p) {
                int i=ai[p],len=0;
                y[i]+=ax[p];
                if (i==k) expected=(ax[p]<0 ? -1 : 1);
                while (flag[i]!=k) {
                    pattern[len++]=i;
                    flag[i]=k;
                    i=parent[i];
                }
                while (len>0) pattern[--top]=pattern[--len];
            }
            Real dk=y[k];
            y[k]=0.0;
            for (int p=top;p<n;++p) {
                int i=pattern[p];
                Real yi=y[i];
                y[i]=0.0;
                int64_t stop=lp[i]+used[i];
                for (int64_t j=lp[i];j<stop;++j)
                    y[li[j]]-=lx[j]*yi;
                Real v=yi/diag[i];
                dk-=v*yi;
                li[stop]=k;
                lx[stop]=v;
                ++used[i];
            }
            Real magnitude=(dk<0 ? -dk : dk);
            if (dk!=dk || magnitude>static_cast<Real>(1e200L)) return false;
            if (enforce_sign) {
                if (dk*expected<1e-14L) dk=expected*1e-7L;
            } else if (magnitude<static_cast<Real>(1e-30L)) {
                dk=(dk<0 ? -1e-20L : 1e-20L);
            }
            diag[k]=dk;
        }
        return true;
    }
    void solve(const double *rhs,double *out) override {
        for (int i=0;i<n;++i) work[i]=rhs[perm[i]];
        for (int j=0;j<n;++j) {
            Real v=work[j];
            for (int64_t p=lp[j];p<lp[j+1];++p)
                work[li[p]]-=lx[p]*v;
        }
        for (int j=0;j<n;++j) work[j]/=diag[j];
        for (int j=n-1;j>=0;--j)
            for (int64_t p=lp[j];p<lp[j+1];++p)
                work[j]-=lx[p]*work[li[p]];
        for (int j=0;j<n;++j) out[perm[j]]=work[j];
    }
    void residual(const double *rhs,const double *x,const double *reg,double *out) override {
        for (int i=0;i<n;++i)
            work[i]=static_cast<Real>(rhs[perm[i]])+
                static_cast<Real>(reg[perm[i]])*x[perm[i]];
        for (int j=0;j<n;++j) {
            Real xj=x[perm[j]];
            for (int p=ap[j];p<ap[j+1];++p) {
                int i=ai[p];
                work[i]-=ax[p]*xj;
                if (i!=j) work[j]-=ax[p]*static_cast<Real>(x[perm[i]]);
            }
        }
        for (int j=0;j<n;++j) out[perm[j]]=work[j];
    }
    int64_t nonzeros() const override { return lp[n]; }
};

extern "C" {
void *ldl_create(int n,const int32_t *ap,const int32_t *ai,const int32_t *perm,int quad) {
    try {
        if (quad) return new SparseLDL<__float128>(n,ap,ai,perm);
        return new SparseLDL<long double>(n,ap,ai,perm);
    }
    catch (...) { return nullptr; }
}
int ldl_factor(void *p,const double *ax,int enforce_sign) {
    return static_cast<LDLBase*>(p)->factor(ax,enforce_sign!=0) ? 1 : 0;
}
void ldl_solve(void *p,const double *rhs,double *out) {
    static_cast<LDLBase*>(p)->solve(rhs,out);
}
void ldl_residual(void *p,const double *rhs,const double *x,const double *reg,double *out) {
    static_cast<LDLBase*>(p)->residual(rhs,x,reg,out);
}
int64_t ldl_nnz(void *p) {
    return static_cast<LDLBase*>(p)->nonzeros();
}
void ldl_free(void *p) { delete static_cast<LDLBase*>(p); }

int lasso_cd(int m,int n,const int32_t *ap,const int32_t *ai,const double *av,
             const double *target,const double *penalty,double *x,double *r,
             double seconds,double tol,double offset,double *diagnostics) {
    using clock=std::chrono::steady_clock;
    auto started=clock::now();
    std::vector<double> lips(n),correlation(n);
    std::vector<int> active;
    double largest=0.0;
    for (int i=0;i<m;++i) r[i]=target[i];
    for (int j=0;j<n;++j) {
        x[j]=0.0;
        long double l=0,c=0;
        for (int p=ap[j];p<ap[j+1];++p) {
            l+=static_cast<long double>(av[p])*av[p];
            c+=static_cast<long double>(av[p])*r[ai[p]];
        }
        lips[j]=2.0*l;
        correlation[j]=2.0*c;
        largest=std::max(largest,penalty[j]);
        if (std::abs(correlation[j])>penalty[j]*(1.-1e-12))
            active.push_back(j);
    }
    for (int epoch=0;epoch<30000;++epoch) {
        for (int j:active) {
            if (lips[j]<=0.0) continue;
            double dot=0;
            for (int p=ap[j];p<ap[j+1];++p) dot+=av[p]*r[ai[p]];
            double value=x[j]+2.0*dot/lips[j];
            double th=penalty[j]/lips[j];
            double next=(value>th ? value-th : (value< -th ? value+th : 0.0));
            double delta=next-x[j];
            if (delta!=0.0) {
                x[j]=next;
                for (int p=ap[j];p<ap[j+1];++p) r[ai[p]]-=av[p]*delta;
            }
        }
        if (epoch%5!=0) continue;
        double error=0.0,ratio=1.0;
        active.clear();
        long double cost=0,ynorm=0,yr=0;
        for (int i=0;i<m;++i) {
            cost+=static_cast<long double>(r[i])*r[i];
            ynorm+=static_cast<long double>(target[i])*r[i];
            yr+=static_cast<long double>(r[i])*r[i];
        }
        for (int j=0;j<n;++j) {
            double dot=0;
            for (int p=ap[j];p<ap[j+1];++p) dot+=av[p]*r[ai[p]];
            double c=2.*dot;
            ratio=std::max(ratio,std::abs(c)/penalty[j]);
            if (x[j]!=0.0) {
                cost+=static_cast<long double>(penalty[j])*std::abs(x[j]);
                error=std::max(error,std::abs(c-std::copysign(penalty[j],x[j])));
            } else error=std::max(error,std::max(0.0,std::abs(c)-penalty[j]));
            if (x[j]!=0.0 || std::abs(c)>penalty[j]*(1.-1e-7))
                active.push_back(j);
        }
        long double scale=1.0/ratio;
        long double dual=2.0*scale*ynorm-scale*scale*yr;
        double gap=static_cast<double>(cost-dual);
        diagnostics[0]=error;
        diagnostics[1]=gap;
        diagnostics[2]=epoch;
        diagnostics[3]=static_cast<double>(cost-offset);
        if (error<=tol*(1.+largest) &&
            gap<=tol*std::max(1.L,std::abs(cost-offset))) return 1;
        if (std::chrono::duration<double>(clock::now()-started).count()>seconds)
            return 0;
    }
    return 0;
}
}

struct SOCOracle {
    int n,d;
    std::vector<int32_t> start;
    std::vector<long double> v,lambda,eta,scratch;
    SOCOracle(int n_,int d_,const int32_t *p): n(n_),d(d_),start(p,p+n),
        v(static_cast<size_t>(n)*d),lambda(static_cast<size_t>(n)*d),
        eta(n),scratch(5*d) {}
    void update(const double *s,const double *z,double *h,double *hi,
                double *u,double *diagonal,int dense,int inverse) {
        for (int k=0;k<n;++k) {
            int st=start[k];
            long double sn=0,zn=0,sz=0;
            for (int j=1;j<d;++j) {
                sn+=static_cast<long double>(s[st+j])*s[st+j];
                zn+=static_cast<long double>(z[st+j])*z[st+j];
            }
            sn=std::sqrt(sn); zn=std::sqrt(zn);
            long double ds=(s[st]-sn)*(s[st]+sn);
            long double dz=(z[st]-zn)*(z[st]+zn);
            ds=std::max(ds,1e-300L); dz=std::max(dz,1e-300L);
            eta[k]=std::sqrt(std::sqrt(ds/dz));
            long double e=eta[k], e2=e*e;
            for (int j=0;j<d;++j) sz+=static_cast<long double>(s[st+j])*z[st+j];
            long double denom=std::sqrt(2.L*(sz+std::sqrt(ds*dz)));
            long double *w=scratch.data();
            for (int j=0;j<d;++j)
                w[j]=(s[st+j]/e+e*(j==0 ? z[st+j] : -z[st+j]))/denom;
            long double *vv=v.data()+static_cast<size_t>(k)*d;
            vv[0]=std::sqrt((w[0]+1.L)*0.5L);
            for (int j=1;j<d;++j) vv[j]=w[j]/(2.L*vv[0]);
            long double dot=0;
            for (int j=0;j<d;++j) dot+=vv[j]*z[st+j];
            for (int j=0;j<d;++j) {
                lambda[static_cast<size_t>(k)*d+j]=e*(2.L*vv[j]*dot-
                    (j==0 ? z[st+j] : -z[st+j]));
                u[static_cast<size_t>(k)*d+j]=std::sqrt(2.L)*e*w[j];
                diagonal[static_cast<size_t>(k)*d+j]=(j==0 ? -e2 : e2);
            }
            if (dense || inverse) {
                for (int i=0;i<d;++i) for (int j=0;j<d;++j) {
                    long double ji=(i==0 ? 1.L : -1.L);
                    long double jj=(j==0 ? 1.L : -1.L);
                    long double entry=2.L*w[i]*w[j]-(i==j ? ji : 0.L);
                    size_t p=(static_cast<size_t>(k)*d+i)*d+j;
                    if (dense) h[p]=e2*entry;
                    if (inverse) hi[p]=(2.L*ji*jj*w[i]*w[j]-
                        (i==j ? ji : 0.L))/e2;
                }
            }
        }
    }
    void correct(const double *s,const double *ds,const double *dz,
                 long double sigma_mu,double *out) {
        long double *a=scratch.data(),*b=a+d,*r=b+d,*yy=r+d;
        for (int k=0;k<n;++k) {
            int st=start[k];
            long double *vv=v.data()+static_cast<size_t>(k)*d;
            long double *lam=lambda.data()+static_cast<size_t>(k)*d;
            long double e=eta[k],dp=0,dd=0;
            for (int j=0;j<d;++j) {
                dp+=(j==0 ? vv[j] : -vv[j])*ds[st+j];
                dd+=vv[j]*dz[st+j];
            }
            for (int j=0;j<d;++j) {
                long double sign=(j==0 ? 1.L : -1.L);
                a[j]=(2.L*sign*vv[j]*dp-sign*ds[st+j])/e;
                b[j]=e*(2.L*vv[j]*dd-sign*dz[st+j]);
            }
            long double prod=0;
            for (int j=0;j<d;++j) prod+=a[j]*b[j];
            r[0]=2.L*sigma_mu-prod;
            for (int j=1;j<d;++j) r[j]=-a[0]*b[j]-b[0]*a[j];
            long double lv=0,lr=0;
            for (int j=1;j<d;++j) { lv+=lam[j]*lam[j]; lr+=lam[j]*r[j]; }
            long double ln=std::sqrt(lv);
            long double det=(lam[0]-ln)*(lam[0]+ln);
            yy[0]=(lam[0]*r[0]-lr)/det;
            for (int j=1;j<d;++j) yy[j]=(r[j]-lam[j]*yy[0])/lam[0];
            long double dot=0;
            for (int j=0;j<d;++j) dot+=vv[j]*yy[j];
            for (int j=0;j<d;++j)
                out[static_cast<size_t>(k)*d+j]=-s[st+j]+e*(2.L*vv[j]*dot-
                    (j==0 ? yy[j] : -yy[j]));
        }
    }
    double step(const double *x,const double *dx) {
        long double alpha=1;
        for (int k=0;k<n;++k) {
            int st=start[k];
            long double aa=static_cast<long double>(dx[st])*dx[st];
            long double bb=static_cast<long double>(x[st])*dx[st];
            long double sn=0;
            for (int j=1;j<d;++j) {
                aa-=static_cast<long double>(dx[st+j])*dx[st+j];
                bb-=static_cast<long double>(x[st+j])*dx[st+j];
                sn+=static_cast<long double>(x[st+j])*x[st+j];
            }
            long double vn=std::sqrt(sn);
            long double cc=(x[st]-vn)*(x[st]+vn);
            bb*=2;
            long double disc=std::max(0.L,bb*bb-4.L*aa*cc);
            long double den=-bb+std::sqrt(disc);
            if (den>0 && (aa<0 || (bb<0 && disc>0)))
                alpha=std::min(alpha,2.L*cc/den);
            if (dx[st]<0) alpha=std::min(alpha,-static_cast<long double>(x[st])/dx[st]);
        }
        return static_cast<double>(std::max(0.L,alpha));
    }
    bool neighborhood(const double *s,const double *z,long double mu) {
        for (int k=0;k<n;++k) {
            int st=start[k];
            long double sn=0,zn=0,gamma=0;
            for (int j=1;j<d;++j) {
                sn+=static_cast<long double>(s[st+j])*s[st+j];
                zn+=static_cast<long double>(z[st+j])*z[st+j];
            }
            sn=std::sqrt(sn); zn=std::sqrt(zn);
            long double ds=(s[st]-sn)*(s[st]+sn);
            long double dz=(z[st]-zn)*(z[st]+zn);
            if (ds<=0 || dz<=0) return false;
            for (int j=0;j<d;++j) gamma+=static_cast<long double>(s[st+j])*z[st+j];
            long double prod=ds*dz;
            long double small=prod/(gamma+std::sqrt(std::max(0.L,gamma*gamma-prod)));
            if (small<2e-2L*mu) return false;
        }
        return true;
    }
};

extern "C" {
void *soc_create(int n,int d,const int32_t *starts) {
    try {return new SOCOracle(n,d,starts);} catch (...) {return nullptr;}
}
void soc_free(void *p) {delete static_cast<SOCOracle*>(p);}
void soc_update(void *p,const double *s,const double *z,double *h,double *hi,
                double *u,double *diag,int dense,int inverse) {
    static_cast<SOCOracle*>(p)->update(s,z,h,hi,u,diag,dense,inverse);
}
void soc_correct(void *p,const double *s,const double *ds,const double *dz,
                 double sigma_mu,double *out) {
    static_cast<SOCOracle*>(p)->correct(s,ds,dz,sigma_mu,out);
}
double soc_step(void *p,const double *s,const double *ds) {
    return static_cast<SOCOracle*>(p)->step(s,ds);
}
int soc_neighborhood(void *p,const double *s,const double *z,double mu) {
    return static_cast<SOCOracle*>(p)->neighborhood(s,z,mu) ? 1 : 0;
}
}
