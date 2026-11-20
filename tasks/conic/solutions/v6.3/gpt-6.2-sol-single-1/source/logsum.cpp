#include "optimizer.hpp"

#include <cstdio>
#include <limits>

extern "C" {
void scipy_dgemv_(const char*,const int*,const int*,const double*,const double*,const int*,
                  const double*,const int*,const double*,double*,const int*);
void scipy_dsyrk_(const char*,const char*,const int*,const int*,const double*,const double*,
                  const int*,const double*,double*,const int*);
void scipy_dpotrf_(const char*,const int*,double*,const int*,int*);
void scipy_dpotrs_(const char*,const int*,const int*,const double*,const int*,double*,const int*,int*);
}

namespace {
double mv(const Vec& a,int n,const Vec& x,Vec& y) {
    if(n){
        const int one=1;
        const double d=1.,z=0.;
        scipy_dgemv_("N",&n,&n,&d,a.data(),&n,x.data(),&one,&z,y.data(),&one);
    }
    return dot(y,x);
}
struct Eval {
    double value=0,lse=0;
    Vec p,grad,px,mean;
    Eval(int d,int n):p(n),grad(d),px(d),mean(d){}
};
}

bool logsum_form(const Model& p,Verifier& verify) {
    const int n=p.n(),m=p.m();
    if(p.cones.size()<7 || n<8) return false;
    int at=0,sumrow=-1;
    Ind pos;
    for(auto c:p.cones) {
        if(c.kind==Orthant && c.dim==1 && sumrow<0)sumrow=at;
        else if(c.kind==Exponential)pos.push_back(at);
        else return false;
        at+=c.width();
    }
    const int k=pos.size(),d=n-k-1;
    if(sumrow<0 || d<0 || d>430 || k*(long long)std::max(1,d)>2'800'000) return false;
    auto ar=p.A.transpose();
    if(ar.p[sumrow+1]-ar.p[sumrow]!=k) return false;
    Ind wcol(k),map(n,-1),uind(m,-1);
    Vec rowcoef(k),ratio(k),vb(k),wi(k);
    long double hb=p.b[sumrow];
    for(int i=0;i<k;++i){
        int j=pos[i], w=j+2;
        uind[j]=i;
        if(ar.p[j+1]!=ar.p[j+2] || !(p.b[j+1]>0) || ar.p[w+1]!=ar.p[w]+1)return false;
        int c=ar.i[ar.p[w]];
        if(map[c]!=-1 || p.q[c]!=0 || p.P.p[c]!=p.P.p[c+1] || p.A.p[c+1]!=p.A.p[c]+2)return false;
        map[c]=-2;wcol[i]=c;
        rowcoef[i]=ar.x[ar.p[w]];
        double s=0;
        for(int l=p.A.p[c];l<p.A.p[c+1];++l)if(p.A.i[l]==sumrow)s=p.A.x[l];
        ratio[i]=-s/rowcoef[i];vb[i]=p.b[j+1];
        if(!(ratio[i]>0) || !std::isfinite(ratio[i]))return false;
        hb+=(long double)ratio[i]*p.b[w];
    }
    const double B=hb;
    if(!(B>0) || !std::isfinite(B))return false;
    int t=-1;
    double epicoef=0;
    for(int j=0;j<n;++j){
        if(map[j]!=-1 || !(p.q[j]>0) || p.P.p[j]!=p.P.p[j+1] || p.A.p[j+1]-p.A.p[j]!=k)continue;
        bool ok=true;
        double r=0;
        for(int l=p.A.p[j];l<p.A.p[j+1];++l){
            int i=uind[p.A.i[l]];
            if(i<0){ok=false;break;}
            double a=p.A.x[l]/vb[i];
            if(!(a>0) || !std::isfinite(a)){ok=false;break;}
            if(!r)r=a;
            else if(std::abs(a/r-1)>2e-13){ok=false;break;}
        }
        if(ok){t=j;epicoef=r;break;}
    }
    if(t<0)return false;
    map[t]=-2;
    Ind idx;
    Vec b(k),C(d*(size_t)k),q;
    for(int j=0;j<n;++j)if(map[j]==-1){map[j]=idx.size();idx.push_back(j);q.push_back(p.q[j]);}
    Vec P(d*(size_t)d);
    for(int j=0;j<d;++j){
        int i=idx[j];
        for(int l=p.P.p[i];l<p.P.p[i+1];++l){
            int u=map[p.P.i[l]];if(u<0)return false;
            P[u+d*j]=p.P.x[l];
        }
    }
    for(int i=0;i<k;++i){
        int u=pos[i];
        b[i]=p.b[u]/vb[i]+std::log(ratio[i])+std::log(vb[i])-std::log(B);
        for(int l=ar.p[u];l<ar.p[u+1];++l){
            int a=ar.i[l], j=map[a];
            if(j>=0) C[j+d*i]=-ar.x[l]/vb[i];
            else if(a!=t)return false;
        }
    }
    if(!std::isfinite(norm(b))||!std::isfinite(norm(C)))return false;
    const double fac=p.q[t]/epicoef;
    if(!(fac>0) || !std::isfinite(fac))return false;
    auto evaluate=[&](const Vec& y,Eval& e){
        const double one=1.;
        const int i1=1;
        e.p=b;
        if(d)scipy_dgemv_("T",&d,&k,&one,C.data(),&d,y.data(),&i1,&one,e.p.data(),&i1);
        double top=*std::max_element(e.p.begin(),e.p.end());
        if(!std::isfinite(top))return false;
        double sum=0;
        for(double& a:e.p){a=std::exp(a-top);sum+=a;}
        e.lse=top+std::log(sum);
        for(double& a:e.p)a/=sum;
        e.value=.5*mv(P,d,y,e.px)+dot(q,y)+fac*e.lse;
        const double z=0.;
        if(d)scipy_dgemv_("N",&d,&k,&one,C.data(),&d,e.p.data(),&i1,&z,e.mean.data(),&i1);
        for(int j=0;j<d;++j)e.grad[j]=q[j]+e.px[j]+fac*e.mean[j];
        return std::isfinite(e.value);
    };
    auto recover=[&](const Vec& y,const Eval& e){
        Vec x(n),z(m);
        for(int j=0;j<d;++j)x[idx[j]]=y[j];
        x[t]=e.lse/epicoef;
        const double zy=fac/B;
        z[sumrow]=zy;
        auto s=p.A.mul(x);
        for(int i=0;i<k;++i){
            int u=pos[i],w=u+2;
            double sw=std::max(std::numeric_limits<double>::denorm_min(),B/ratio[i]*e.p[i]);
            x[wcol[i]]=(p.b[w]-sw)/rowcoef[i];
            double r=(p.b[u]-s[u])/vb[i];
            double a=fac*e.p[i]/vb[i];
            z[u]=-a;z[u+1]=(r-1)*a;z[w]=zy*ratio[i];
        }
        p.A.mul(x.data(),s.data());
        for(int j=0;j<m;++j)s[j]=p.b[j]-s[j];
        // Preserve tiny exp slacks when subtracting two almost equal offsets.
        for(int i=0;i<k;++i)s[pos[i]+2]=std::max(std::numeric_limits<double>::denorm_min(),B/ratio[i]*e.p[i]);
        return verify.solved(x,s,z,false);
    };
    Vec y(d),trial(d),R(C.size()),H(P.size()),dx(d);
    Eval current(d,k),test(d,k);
    if(!evaluate(y,current))return false;
    if(!d)return recover(y,current);
    int one=1;
    const double unity=1.;
    for(int it=0;it<40;++it){
        double gn=norm(current.grad),scale=1+std::max(fastnorm(q),fastnorm(current.px));
        if(gn<2e-7*scale && recover(y,current))return true;
        for(int i=0;i<k;++i){
            double sc=std::sqrt(fac*current.p[i]);
            for(int j=0;j<d;++j) R[j+d*i]=(C[j+d*i]-current.mean[j])*sc;
        }
        H=P;
        scipy_dsyrk_("U","N",&d,&k,&unity,R.data(),&d,&unity,H.data(),&d);
        double sc=0;
        for(int j=0;j<d;++j)sc=std::max(sc,std::abs(H[j+d*j]));
        sc=std::max(sc,1e-8);
        Vec orig=H;
        int info=-1;
        for(double a:{1e-13,1e-9,1e-6}) {
            H=orig;
            for(int j=0;j<d;++j)H[j+d*j]+=sc*a;
            scipy_dpotrf_("U",&d,H.data(),&d,&info);
            if(!info)break;
        }
        if(info)return false;
        for(int j=0;j<d;++j)dx[j]=-current.grad[j];
        scipy_dpotrs_("U",&d,&one,H.data(),&d,dx.data(),&d,&info);
        double der=dot(dx,current.grad);
        if(info || !(der<0) || norm(dx)>1e10*(1+norm(y)))break;
        double a=1;
        bool ok=false;
        for(int j=0;j<19;++j,a*=.5){
            for(int i=0;i<d;++i)trial[i]=y[i]+a*dx[i];
            if(evaluate(trial,test) && (test.value<=current.value+a*der*1e-4+
                 1e-14*(1+std::abs(current.value)) || (gn<1e-5*scale && norm(test.grad)<.7*gn))){
                y.swap(trial);std::swap(current,test);ok=true;break;
            }
        }
        if(!ok)break;
    }
    return recover(y,current);
}
