#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace {
using V=std::vector<double>;
using Clock=std::chrono::steady_clock;

int solve(int n,int m,const int* row,const int* col,const double* a,
          const double* diagonal,const double* q,const double* b,const unsigned char* eq,
          double* solx,double* solz,double seconds) {
    auto end=Clock::now()+std::chrono::microseconds(static_cast<long long>(seconds*1.e6));
    V inv(n), linear(n), x(n), norm(m), rhs(m), lam(m), at(row[m]);
    double xs=1., bs=1., zs=1.;
    for (int j=0;j<n;++j) {
        if (!(diagonal[j]>0.)) return 0;
        inv[j]=1./std::sqrt(diagonal[j]);
        linear[j]=x[j]=-q[j]*inv[j];
        xs=std::max(xs,std::abs(x[j]));
    }
    for (int i=0;i<m;++i) {
        double v=0.;
        for (int k=row[i];k<row[i+1];++k) {
            double t=a[k]*inv[col[k]];
            v+=t*t;
            at[k]=t;
        }
        if (!(v>0.)) {
            if (eq[i]?b[i]!=0.:b[i]<0.) return 0;
            continue;
        }
        norm[i]=1./std::sqrt(v);
        rhs[i]=b[i]*norm[i];
        bs=std::max(bs,std::abs(rhs[i]));
        for (int k=row[i];k<row[i+1];++k) at[k]*=norm[i];
    }
    if (!std::isfinite(xs+bs)) return 0;
    auto rowdot=[&](int i,const V& x) {
        double t=0.;
        for (int k=row[i];k<row[i+1];++k) t+=at[k]*x[col[k]];
        return t;
    };
    auto rebuild=[&]() {
        x=linear;
        for (int i=0;i<m;++i) if (lam[i]!=0.)
            for (int k=row[i];k<row[i+1];++k) x[col[k]]-=lam[i]*at[k];
    };
    auto output=[&]() {
        for (int j=0;j<n;++j) solx[j]=x[j]*inv[j];
        for (int i=0;i<m;++i) solz[i]=lam[i]*norm[i];
    };
    std::vector<int> active;
    V fixed,xsave,zsave;
    double best=1.e90;
    bool improved=false;
    for (int it=0;it<180;++it) {
        if (Clock::now()>end) break;
        double scale=0.;
        const double relax=it<80?1.56:1.23;
        for (int st=0;st<m;++st) {
            int i=it%2==0?st:m-1-st;
            if (norm[i]==0.) continue;
            double d=(rowdot(i,x)-rhs[i])*relax, y=lam[i]+d;
            if (!eq[i]) y=std::max(0.,y);
            d=y-lam[i];
            lam[i]=y;
            scale=std::max(scale,std::abs(d));
            for (int k=row[i];k<row[i+1];++k) x[col[k]]-=d*at[k];
        }
        if (it%7!=6 && scale>1.e-11*std::max(xs,bs)) continue;
        rebuild();
        active.clear();
        double pn=0., pg=0.;
        for (int i=0;i<m;++i) if (norm[i]>0.) {
            double g=rowdot(i,x)-rhs[i];
            if (eq[i] || lam[i]>1.e-13) {
                active.push_back(i);
                pn=std::max(pn,std::abs(g));
            } else pn=std::max(pn,g);
            pg+=std::abs(g*lam[i]);
            zs=std::max(zs,std::abs(lam[i]));
        }
        const double measure=std::max(pn/(1.+std::max(xs,bs)),
                                      pg/(1.+n*std::max(xs,bs)*zs));
        if (measure<best) { best=measure; output(); improved=true; }
        if (measure<7.e-11) return 1;
        if (it%14!=6 || active.empty() || measure>.16 || Clock::now()>end) continue;
        const auto na=active.size();
        V r(na), dir(na), delta(na), vec(na), adj(n);
        for (std::size_t k=0;k<na;++k)
            dir[k]=r[k]=rowdot(active[k],x)-rhs[active[k]];
        double rr=std::inner_product(r.begin(),r.end(),r.begin(),0.);
        const double initial=rr;
        for (int j=0;j<37 && Clock::now()<end;++j) {
            std::fill(adj.begin(),adj.end(),0.);
            for (std::size_t k=0;k<na;++k) {
                int i=active[k];
                for (int h=row[i];h<row[i+1];++h) adj[col[h]]+=dir[k]*at[h];
            }
            double den=0.;
            for (std::size_t k=0;k<na;++k) {
                vec[k]=rowdot(active[k],adj)+1.e-12*dir[k];
                den+=vec[k]*dir[k];
            }
            if (!(den>1.e-28*rr)) break;
            const double step=rr/den;
            double nr=0.;
            for (std::size_t k=0;k<na;++k) {
                r[k]-=step*vec[k];
                delta[k]+=step*dir[k];
                nr+=r[k]*r[k];
            }
            if (nr<1.e-21*initial || nr<1.e-25) { rr=nr; break; }
            for (std::size_t k=0;k<na;++k) dir[k]=r[k]+(nr/rr)*dir[k];
            rr=nr;
        }
        if (rr>.36*initial) continue;
        double step=1.;
        for (std::size_t k=0;k<na;++k) {
            int i=active[k];
            if (!eq[i] && delta[k]<0.) step=std::min(step,-lam[i]/delta[k]);
        }
        if (!(step>.16)) continue;
        for (std::size_t k=0;k<na;++k)
            lam[active[k]]+=step*delta[k];
        rebuild();
    }
    return improved;
}
}

extern "C" int conic_diagonal_qp(int n,int m,const int* p,const int* i,
                      const double* a,const double* d,const double* q,const double* b,
                      const unsigned char* eq,double* x,double* z,double seconds) {
    try { return solve(n,m,p,i,a,d,q,b,eq,x,z,seconds); }
    catch (...) { return 0; }
}
