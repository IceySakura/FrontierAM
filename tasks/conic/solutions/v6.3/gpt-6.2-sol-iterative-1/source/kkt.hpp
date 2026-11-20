#pragma once
#include "cones.hpp"
#include "extlu.hpp"
#include <memory>

namespace conic {

struct SOCScale {
    Vec v, lam, a, b, t, axis, house;
    double eta=1., det=1., tail=0.;
    double hh=1.,sgn=1.;
    explicit SOCScale(int d):v(d),lam(d),a(d),b(d),t(d),axis(d-1),house(d-1) {}
    bool update(const double* s,const double* z) {
        int d=int(v.size());
        long double sn=std::sqrt(soc_inner(s,s,d)), zn=std::sqrt(soc_inner(z,z,d));
        if (!(sn>0 && zn>0 && s[0]>0 && z[0]>0)) return false;
        long double sz=0;
        for (int i=0;i<d;++i) sz+=(long double)s[i]*z[i];
        long double den=std::sqrt(2*(1+sz/sn/zn));
        if (!(den>0)) return false;
        v[0]=double((s[0]/sn+z[0]/zn)/den);
        double r2=0;
        for(int i=1;i<d;++i) {
            v[i]=double((s[i]/sn-z[i]/zn)/den);
            r2+=v[i]*v[i];
        }
        tail=std::sqrt(r2);
        v[0]=std::hypot(1.,tail); // Enforce the Lorentz normalization.
        eta=double(std::sqrt(sn/zn));
        det=double(sn*zn);
        // A cancellation-free formula for the common scaled point's scalar.
        lam[0]=double(std::sqrt(.5L*(sz+sn*zn)));
        long double div=v[0]+1.L;
        for (int i=1;i<d;++i)
            lam[i]=double(((long double)eta*z[i]+s[i]/(long double)eta
                          + v[i]*((long double)eta*z[0]-s[0]/(long double)eta)/div)*.5L);
        return eta>0 && det>0 && lam[0]>0 && std::isfinite(v[0]);
    }
    void apply_w(const double* x,double* out,bool inv=false) const {
        int d=int(v.size());
        long double r=0;
        for(int i=1;i<d;++i) r+=(long double)v[i]*x[i];
        long double a=inv ? -1.L : 1.L, f=inv ? 1.L/eta : eta;
        long double c=a*x[0]+r/(v[0]+1.L);
        for(int i=1;i<d;++i) out[i]=double(f*(x[i]+v[i]*c));
        out[0]=double(f*((long double)v[0]*x[0]+a*r));
    }
    void correct(double* out,const double* ds,const double* dz,double center,double damp) {
        int d=int(v.size());
        if (damp!=0) {
            apply_w(ds,a.data(),true);
            apply_w(dz,b.data());
            t[0]=center-damp*dot(a,b);
            for(int i=1;i<d;++i) t[i]=-damp*(a[0]*b[i]+b[0]*a[i]);
        } else {
            t[0]=center;
            std::fill(t.begin()+1,t.end(),0);
        }
        t[0]=double(soc_inner(lam.data(),t.data(),d)/det);
        for(int i=1;i<d;++i) t[i]=(t[i]-lam[i]*t[0])/lam[0];
        apply_w(t.data(),a.data());
        for(int i=0;i<d;++i) out[i]+=a[i];
    }
    double energy(const double* z) {
        apply_w(z,t.data());
        return dot(t,t);
    }
    void rotation() {
        int d=int(v.size());
        for(int i=1;i<d;++i) axis[i-1]=tail>0?v[i]/tail:double(i==1);
        sgn=axis[0]>=0 ? 1. : -1.;
        house=axis;
        house[0]+=sgn;
        hh=1./(1.+std::abs(axis[0]));
    }
    void rotate(const double* x,double* y) const {
        int d=int(v.size());
        long double r=0, q=0;
        for(int j=1;j<d;++j) {
            r+=(long double)axis[j-1]*x[j];
            q+=(long double)house[j-1]*x[j];
        }
        y[0]=double((x[0]+r)*invroot2);
        y[1]=double((x[0]-r)*invroot2);
        for(int j=2;j<d;++j) y[j]=double(x[j]-house[j-1]*(hh*q));
    }
    void unrotate(const double* x,double* y) const {
        int d=int(v.size());
        const long double y1=-sgn*((long double)x[0]-x[1])*invroot2;
        long double q=house[0]*y1;
        for(int j=2;j<d;++j) q+=(long double)house[j-1]*x[j];
        y[0]=double(((long double)x[0]+x[1])*invroot2);
        y[1]=double(y1-house[0]*(hh*q));
        for(int j=2;j<d;++j) y[j]=double(x[j]-house[j-1]*(hh*q));
    }
};

struct Block {
    ConeInfo c;
    Int ids, ri, rj;
    std::unique_ptr<SOCScale> soc;
    std::unique_ptr<PSDScale> psd;
    ExpScale exp;
    struct Term {int row; double value;};
    Int cols, ap;
    std::vector<Term> av;
    int first=-1, aux=-1;
    bool eliminated=false, rotated=false;
    Vec ratio;
    explicit Block(ConeInfo v):c(v) {}
};

struct Entry { int i,j; double v; };

// Quotient-graph minimum-degree ordering. Eliminated vertices hold elements,
// keeping dense fill cliques implicit.
inline Int minimum_degree(int n,const Int& p,const Int& ii) {
    std::vector<Int> adj(n);
    Int count(n,0), stamp(n,-1), degree(n), result;
    result.reserve(n);
    for(int j=0;j<n;++j)
        for(int a=p[j];a<p[j+1];++a)
            if(ii[a]!=j) { ++count[ii[a]]; ++count[j]; }
    for(int i=0;i<n;++i) adj[i].reserve(count[i]);
    for(int j=0;j<n;++j)
        for(int a=p[j];a<p[j+1];++a) {
            int i=ii[a];
            if(i!=j) {adj[i].push_back(j);adj[j].push_back(i);}
        }
    // Intrusive buckets avoid heap growth on frontier updates.
    Int next(n,-1), prev(n,-1), head(n,-1);
    auto insert = [&](int v,int d){
        degree[v]=d; prev[v]=-1; next[v]=head[d];
        if(next[v]!=-1) prev[next[v]]=v;
        head[d]=v;
    };
    auto erase = [&](int v){
        if(prev[v]==-1) head[degree[v]]=next[v]; else next[prev[v]]=next[v];
        if(next[v]!=-1) prev[next[v]]=prev[v];
    };
    for(int i=n-1;i>=0;--i) insert(i,count[i]);
    Int touch, elements, in, ext(n,0), estmp(n,-1);
    int low=0;
    for(int k=0;k<n;++k) {
        while(low<n && head[low]<0) ++low;
        if(low==n) return {};
        int v=head[low];
        erase(v); degree[v]=-1; result.push_back(v);
        touch.clear(); elements.clear(); stamp[v]=k;
        for(int u:adj[v]) {
            if(degree[u]>=0) {
                if(stamp[u]!=k) {stamp[u]=k; touch.push_back(u);}
            } else if(!adj[u].empty()) {
                for(int a:adj[u])
                    if(degree[a]>=0 && stamp[a]!=k) {stamp[a]=k;touch.push_back(a);}
                adj[u].clear();
            }
        }
        adj[v].swap(touch);
        const int width=int(adj[v].size());
        if(width==0) continue;
        // External degree of each adjacent old element, amortized over overlaps.
        for(int u:adj[v]) {
            Int& row=adj[u];
            int w=0;
            for(int j:row) {
                if(degree[j]>=0) {
                    if(stamp[j]!=k) row[w++]=j;
                } else if(j!=v && !adj[j].empty()) {
                    row[w++]=j;
                    if(estmp[j]!=k) {
                        estmp[j]=k; ext[j]=int(adj[j].size()); elements.push_back(j);
                    }
                    --ext[j];
                }
            }
            row.resize(w);
        }
        for(int j:elements) if(ext[j]==0) adj[j].clear();
        for(int u:adj[v]) {
            Int& row=adj[u];
            int d=width-1, w=0;
            for(int j:row) {
                if(degree[j]>=0) { ++d; row[w++]=j; }
                else if(!adj[j].empty()) { d+=ext[j]; row[w++]=j; }
            }
            row.resize(w);
            row.push_back(v);
            d=std::min({d,n-k-2,degree[u]+width-1});
            d=std::max(0,d);
            if(d!=degree[u]) {
                erase(u); insert(u,d);
                low=std::min(low,d);
            }
        }
    }
    return result;
}

struct LinearSystem {
    int n=0;
    int split=0, trailing=0;
    double work_est=0.;
    bool dense=false, spd=false, ok=false, extra=false, use_lu=false;
    int64_t lu_token=0;
    Int p,i, perm, pinv, entrymap, di, sign, pivot;
    Int parent, lp, li, mark, pattern, walk, next, reachp, reachi, border;
    Vec mat, val, lv, dinv, scale, maxrow, y, fac, lwork, panel, sx, sr, su;
    Vec gm_v,gm_z,gm_rhs,gm_work,gm_corr;
    std::vector<long double> llv, ldinv, ly;

    LinearSystem(int size,const Int& signs,const std::vector<Entry>& es,
                 bool may_spd,bool force_dense=false,bool external=false):n(size),spd(may_spd) {
        // Assemble unpermuted column structure, preserving numeric assembly slots.
        if (n==0) {ok=true; dense=true;return;}
        if (es.size()>35000000) return;
        Int cp(n+1,0), idx(es.size()), ci;
        for(const auto& e:es) ++cp[std::max(e.i,e.j)+1];
        for(int j=0;j<n;++j) cp[j+1]+=cp[j];
        Int head=cp;
        for(int a=0;a<(int)es.size();++a) idx[head[std::max(es[a].i,es[a].j)]++]=a;
        Int up(n+1,0);
        for(int j=0;j<n;++j) {
            std::sort(idx.begin()+cp[j],idx.begin()+cp[j+1],
                [&](int a,int b){return std::min(es[a].i,es[a].j)<std::min(es[b].i,es[b].j);});
            int prev=-1;
            for(int a=cp[j];a<cp[j+1];++a) {
                int t=idx[a], r=std::min(es[t].i,es[t].j);
                if(r!=prev) {ci.push_back(r);prev=r;}
            }
            up[j+1]=int(ci.size());
        }
        dense = n<=95 || (n<=2400 && (force_dense || ci.size()>.075*double(n)*n));
        use_lu=!dense && external && external_lu && n<=45000 && es.size()<1300000;
        if(dense || use_lu) {
            perm.resize(n);std::iota(perm.begin(),perm.end(),0);
        } else {
            perm=minimum_degree(n,up,ci);
            if(perm.empty()) return;
        }
        pinv.resize(n);
        for(int j=0;j<n;++j) pinv[perm[j]]=j;
        sign.resize(n);
        p.assign(n+1,0);
        for(const auto&e:es) ++p[std::max(pinv[e.i],pinv[e.j])+1];
        for(int j=0;j<n;++j) p[j+1]+=p[j];
        head=p;
        for(int a=0;a<(int)es.size();++a) idx[head[std::max(pinv[es[a].i],pinv[es[a].j])]++]=a;
        entrymap.resize(es.size());di.resize(n);
        for(int j=0;j<n;++j) {
            sign[j]=signs[perm[j]];
            int start=p[j], end=p[j+1], previous=-1;
            p[j]=int(i.size());
            std::sort(idx.begin()+start,idx.begin()+end,
                [&](int a,int b){return std::min(pinv[es[a].i],pinv[es[a].j])<
                                      std::min(pinv[es[b].i],pinv[es[b].j]);});
            for(int a=start;a<end;++a){
                int k=idx[a], r=std::min(pinv[es[k].i],pinv[es[k].j]);
                if(r!=previous) {i.push_back(r);previous=r;}
                entrymap[k]=int(i.size())-1;
            }
            di[j]=int(i.size())-1; // A diagonal structural entry is always supplied.
        }
        p[n]=int(i.size());
        mat.resize(i.size());y.resize(n);scale.resize(n);maxrow.resize(n);
        if(use_lu) {
            val.resize(i.size());
            work_est=1.;
        } else if(dense) {
            val.resize(i.size());
            fac.resize(size_t(n)*n);pivot.resize(n);lwork.resize(size_t(96)*n);
            work_est=double(n)*n*n/3;
        } else {
            parent.assign(n,-1); next.assign(n,0); mark.assign(n,-1);
            lp.assign(n+1,0); pattern.resize(n); walk.resize(n); dinv.resize(n);
            for(int k=0;k<n;++k) {
                mark[k]=k;
                for(int a=p[k];a<di[k];++a) {
                    for(int j=i[a];mark[j]!=k;j=parent[j]){
                        if(parent[j]<0) parent[j]=k;
                        ++next[j]; mark[j]=k;
                    }
                }
            }
            int64_t total=0;
            int nd=n;
            while(nd>0 && next[nd-1]==n-nd) --nd;
            if(n-nd>=85) {
                trailing=std::min(2800,n-nd);
                split=n-trailing;
                fac.resize(size_t(trailing)*trailing);
                pivot.resize(trailing);lwork.resize(size_t(96)*trailing);
                panel.resize(size_t(48)*trailing);
            } else split=n;
            for(int j=0;j<n;++j) {
                if(j<split)total+=next[j];
                if(total>49000000) return;
                lp[j+1]=int(total);
                if(j<split)work_est+=.5*double(next[j])*(next[j]+3);
            }
            if(work_est>1.6e9) return;
            if(trailing)work_est+=double(trailing)*trailing*trailing/75.;
            li.resize(total);lv.resize(total);
            // Record the row reach once for all the subsequent numerical factors.
            reachp.resize(n+1);reachi.resize(total);
            int at=0;
            std::fill(mark.begin(),mark.end(),-1);
            std::copy(lp.begin(),lp.begin()+n,next.begin());
            for(int k=0;k<n;++k) {
                reachp[k]=at;
                mark[k]=k;
                int top=n;
                for(int a=p[k];a<di[k];++a) {
                    int j=i[a],len=0;
                    for(;mark[j]!=k;j=parent[j]) {walk[len++]=j;mark[j]=k;}
                    while(len)pattern[--top]=walk[--len];
                }
                for(int b=top;b<n;++b) {
                    int j=pattern[b];
                    if(j<split) {li[next[j]++]=k;reachi[at++]=j;}
                }
            }
            reachp[n]=at;
            if(trailing) {
                border.resize(split);
                for(int j=0;j<split;++j)
                    border[j]=int(std::lower_bound(li.begin()+lp[j],li.begin()+lp[j+1],split)-li.begin());
            }
            Int().swap(mark);Int().swap(walk);Int().swap(pattern);Int().swap(parent);
        }
        ok = true;
    }

    ~LinearSystem() {
        if(lu_token && external_lu.release)external_lu.release(lu_token);
    }

    template<class V>
    bool finish_trailing(const V& lv,const V& dinv,double reg) {
        const int d=trailing;
        if(d==0)return true;
        std::fill(fac.begin(),fac.end(),0.);
        for(int j=split;j<n;++j) {
            for(int a=p[j];a<p[j+1];++a)
                if(i[a]>=split)fac[(j-split)+size_t(i[a]-split)*d]=mat[a]*scale[j]*scale[i[a]];
            fac[(j-split)+size_t(j-split)*d]+=reg*sign[j];
        }
        for(int sgn:{1,-1}) {
            int count=0;
            auto flush=[&]() {
                if(count==0)return;
                const double a=-sgn,b=1.;
                scipy_dsyrk_("L","N",&d,&count,&a,panel.data(),&d,&b,fac.data(),&d);
                count=0;
            };
            for(int j=0;j<split;++j) {
                const int lo=border[j],hi=lp[j+1];
                if(lo==hi || dinv[j]*sgn<=0)continue;
                const double d_j=double(1./dinv[j]);
                if(hi-lo<d/5) {
                    for(int k=lo;k<hi;++k) {
                        const double w=double(lv[k])*d_j;
                        double* dst=fac.data()+size_t(li[k]-split)*d;
                        for(int a=k;a<hi;++a)dst[li[a]-split]-=w*double(lv[a]);
                    }
                } else {
                    double* out=panel.data()+size_t(count)*d;
                    std::fill(out,out+d,0.);
                    const double w=std::sqrt(std::abs(d_j));
                    for(int k=lo;k<hi;++k)out[li[k]-split]=double(lv[k])*w;
                    if(++count==48)flush();
                }
            }
            flush();
        }
        int info=0,lw=int(lwork.size());
        scipy_dsytrf_("L",&d,fac.data(),&d,pivot.data(),lwork.data(),&lw,&info);
        return info==0;
    }

    template<class F,class V>
    bool sparse_factor(V& ly,V& lv,V& dinv,double reg) {
        std::fill(ly.begin(),ly.end(),F(0));
        std::copy(lp.begin(),lp.begin()+n,next.begin());
        for(int k=0;k<n;++k) {
            for(int a=p[k];a<di[k];++a) {
                int j=i[a];
                if(j>=split)break;
                ly[j]=F(mat[a])*scale[j]*scale[k];
            }
            F d=F(mat[di[k]])*scale[k]*scale[k]+sign[k]*reg;
            for(int b=reachp[k];b<reachp[k+1];++b) {
                int j=reachi[b],end=next[j];
                F t=ly[j]; ly[j]=0;
                const int iend=k<split?end:border[j];
                for(int c=lp[j];c<iend;++c) ly[li[c]]-=lv[c]*t;
                F a=t*dinv[j];
                lv[end]=a;next[j]=end+1;
                d-=a*t;
            }
            if(k<split) {
                if(!std::isfinite(d)) return false;
                constexpr bool low=std::is_same_v<F,double>;
                if(d*sign[k] < (low?2e-14:2e-18))
                    d=std::max(low?3e-11:1e-17,reg)*sign[k];
                dinv[k]=1./d;
            }
        }
        return finish_trailing(lv,dinv,reg);
    }

    bool factor(const Vec& input,double reg,bool extended=false) {
        if(n==0) return true;
        extra=extended && !dense && work_est<1.e8;
        std::fill(mat.begin(),mat.end(),0.);
        for(size_t k=0;k<input.size();++k) mat[entrymap[k]]+=input[k];
        std::fill(maxrow.begin(),maxrow.end(),0.);
        for(int j=0;j<n;++j) for(int a=p[j];a<p[j+1];++a) {
            double v=std::abs(mat[a]);
            maxrow[j]=std::max(maxrow[j],v);
            maxrow[i[a]]=std::max(maxrow[i[a]],v);
        }
        for(int j=0;j<n;++j)
            scale[j]=1./std::sqrt(std::max({std::abs(mat[di[j]]), .03*maxrow[j], 1e-6}));
        if(use_lu) {
            for(int j=0;j<n;++j) {
                for(int a=p[j];a<p[j+1];++a)val[a]=mat[a]*scale[i[a]]*scale[j];
                val[di[j]]+=sign[j]*reg;
            }
            lu_token=external_lu.make(lu_token,n,p.data(),i.data(),val.data());
            return lu_token>0;
        }
        if(dense) {
            for(int j=0;j<n;++j) {
                for(int a=p[j];a<p[j+1];++a) val[a]=mat[a]*scale[i[a]]*scale[j];
                val[di[j]]+=sign[j]*reg;
            }
            std::fill(fac.begin(),fac.end(),0.);
            for(int j=0;j<n;++j)
                for(int a=p[j];a<p[j+1];++a) fac[j+size_t(i[a])*n]=val[a];
            if(spd && chol(n,fac.data())) return true;
            if(spd) {
                for(int j=0;j<n;++j)
                    for(int a=p[j];a<p[j+1];++a) fac[j+size_t(i[a])*n]=val[a];
            }
            spd=false;
            int info=0, lw=int(lwork.size());
            scipy_dsytrf_("L",&n,fac.data(),&n,pivot.data(),lwork.data(),&lw,&info);
            return info==0;
        }
        if(extra) {
            ly.resize(n);ldinv.resize(n);llv.resize(lv.size());
            return sparse_factor<long double>(ly,llv,ldinv,reg);
        }
        return sparse_factor<double>(y,lv,dinv,reg);
    }

    template<class V>
    void sparse_raw(V& x,const V& lv,const V& dinv) const {
        for(int j=0;j<split;++j) {
            auto v=x[j];
            for(int a=lp[j];a<lp[j+1];++a) x[li[a]]-=lv[a]*v;
        }
        for(int j=0;j<split;++j) x[j]*=dinv[j];
        if(trailing) {
            const int one=1;
            int info=0;
            if constexpr (std::is_same_v<typename V::value_type,double>) {
                scipy_dsytrs_("L",&trailing,&one,fac.data(),&trailing,pivot.data(),x.data()+split,&trailing,&info);
            } else {
                Vec w(trailing);
                for(int j=0;j<trailing;++j)w[j]=double(x[j+split]);
                scipy_dsytrs_("L",&trailing,&one,fac.data(),&trailing,pivot.data(),w.data(),&trailing,&info);
                for(int j=0;j<trailing;++j)x[j+split]=w[j];
            }
        }
        for(int j=split-1;j>=0;--j) {
            auto v=x[j];
            for(int a=lp[j];a<lp[j+1];++a) v-=lv[a]*x[li[a]];
            x[j]=v;
        }
    }

    void raw(Vec& x) {
        if(n==0) return;
        if(use_lu) {
            if(!external_lu.solve(lu_token,n,x.data()))std::fill(x.begin(),x.end(),inf);
            return;
        }
        if(dense) {
            const int one=1;
            int info;
            if(spd) scipy_dpotrs_("L",&n,&one,fac.data(),&n,x.data(),&n,&info);
            else scipy_dsytrs_("L",&n,&one,fac.data(),&n,pivot.data(),x.data(),&n,&info);
            return;
        }
        if(extra) {
            std::copy(x.begin(),x.end(),ly.begin());
            sparse_raw(ly,llv,ldinv);
            std::copy(ly.begin(),ly.end(),x.begin());
        } else sparse_raw(x,lv,dinv);
    }

    void residual(const Vec& rhs,const Vec& x,Vec& r) const {
        for(int j=0;j<n;++j)r[j]=rhs[perm[j]];
        for(int j=0;j<n;++j) {
            double a=x[j],b=r[j]-a*mat[di[j]];
            for(int k=p[j];k<di[j];++k) {
                r[i[k]]-=mat[k]*a;
                b-=mat[k]*x[i[k]];
            }
            r[j]=b;
        }
    }

    bool krylov(const Vec& rhs,Vec& answer,double previous,double target) {
        if(n<8 || n>95000 || i.size()>1700000 || !std::isfinite(previous))return false;
        constexpr int cap=10;
        gm_v.resize(size_t(n)*(cap+1));gm_z.resize(size_t(n)*cap);
        gm_work.resize(n);gm_corr.resize(n);gm_rhs.resize(n);
        auto& rr=gm_rhs; auto& work=gm_work; auto& corr=gm_corr;
        for(int j=0;j<n;++j)corr[j]=answer[perm[j]];
        residual(rhs,corr,rr);
        double ss=0.;
        for(int j=0;j<n;++j){work[j]=scale[j]*rr[j];ss+=work[j]*work[j];}
        double first=std::sqrt(ss);
        if(!(first>0.)||!std::isfinite(first))return false;
        for(int j=0;j<n;++j)gm_v[j]=work[j]/first;
        double B[(cap+1)*cap]{},cs[cap]{},sn[cap]{},g[cap+1]{};
        g[0]=first;
        int last=-1;
        for(int k=0;k<cap;++k) {
            const double* current=gm_v.data()+size_t(k)*n;
            std::copy(current,current+n,work.begin());
            raw(work);
            if(!finite(work))break;
            double* dz=gm_z.data()+size_t(k)*n;
            std::copy(work.begin(),work.end(),dz);
            for(int j=0;j<n;++j)corr[j]=work[j]*scale[j];
            std::fill(work.begin(),work.end(),0.);
            for(int j=0;j<n;++j) {
                double a=corr[j],b=work[j]+a*mat[di[j]];
                for(int t=p[j];t<di[j];++t) {
                    work[i[t]]+=mat[t]*a;
                    b+=mat[t]*corr[i[t]];
                }
                work[j]=b;
            }
            for(int j=0;j<n;++j)work[j]*=scale[j];
            double* col=B+k*(cap+1);
            for(int j=0;j<=k;++j) {
                const double* v=gm_v.data()+size_t(j)*n;
                double a=col[j]=dot(v,work.data(),n);
                for(int i=0;i<n;++i)work[i]-=a*v[i];
            }
            double d=col[k+1]=std::sqrt(dot(work,work));
            if(!std::isfinite(d))break;
            if(d>1e-30) {
                double* v=gm_v.data()+size_t(k+1)*n;
                for(int i=0;i<n;++i)v[i]=work[i]/d;
            }
            for(int j=0;j<k;++j) {
                double a=col[j],b=col[j+1];
                col[j]=cs[j]*a+sn[j]*b;col[j+1]=-sn[j]*a+cs[j]*b;
            }
            double a=col[k],h=std::hypot(a,d);
            if(!(h>1e-30))break;
            cs[k]=a/h;sn[k]=d/h;col[k]=h;
            g[k+1]=-sn[k]*g[k];g[k]*=cs[k];
            last=k;
            if(std::abs(g[k+1])<first*2e-8 || d<=1e-30)break;
        }
        if(last<0)return false;
        for(int j=last;j>=0;--j) {
            g[j]/=B[j+(cap+1)*j];
            for(int i=0;i<j;++i)g[i]-=B[i+(cap+1)*j]*g[j];
        }
        for(int j=0;j<n;++j) {
            double s=0.;
            for(int i=0;i<=last;++i)s+=g[i]*gm_z[size_t(i)*n+j];
            corr[j]=answer[perm[j]]+s*scale[j];
        }
        if(!finite(corr))return false;
        residual(rhs,corr,rr);
        double err=norm(rr);
        if(!(err<std::max(.8*previous,target)))return false;
        for(int j=0;j<n;++j)answer[perm[j]]=corr[j];
        return true;
    }

    bool solve(const Vec& rhs,Vec& answer,int refine=4) {
        if(n==0) return true;
        for(int j=0;j<n;++j) y[j]=rhs[perm[j]]*scale[j];
        raw(y);
        for(int j=0;j<n;++j) answer[perm[j]]=y[j]*scale[j];
        if(!refine) return finite(answer);
        auto& x=sx;auto& r=sr;auto& u=su;
        x.resize(n);r.resize(n);u.resize(n);
        for(int j=0;j<n;++j) x[j]=y[j]*scale[j];
        double previous=inf;
        const double goal=2e-13*std::max(1e-10,norm(rhs));
        for(int it=0;it<=refine;++it) {
            residual(rhs,x,r);
            double err=norm(r);
            if(!std::isfinite(err)||err>=previous*.99) break;
            if(it) for(int j=0;j<n;++j) answer[perm[j]]=x[j];
            previous=err;
            if(err<=goal||it==refine) break;
            for(int j=0;j<n;++j) u[j]=r[j]*scale[j];
            raw(u);
            for(int j=0;j<n;++j) x[j]+=u[j]*scale[j];
        }
        if(refine>=3 && previous>600*goal)krylov(rhs,answer,previous,goal);
        return finite(answer);
    }
};

struct KKT {
    const Problem& p;
    const int n,m;
    int dim;
    std::vector<Block> blocks;
    std::unique_ptr<LinearSystem> ls;
    Vec values,rr,ans,temp,temp2,schurwork;
    Vec qrf, qrwork, qrtau, qrvec, qrx, qrw;
    Int qrpiv;
    bool qr_allowed=false, qr_active=false;
    int qrrank=0;
    Int map, amap, pmap;
    bool ok=false, eliminates=false;
    int degree=1;
    int maxpsd=0;
    KKT(const Problem& prob,bool noelim=false,bool robust=false,bool dense=false,bool rotate_soc=true,
        bool external=false):
          p(prob),n(p.n()),m(p.m()),dim(n),map(m,-1) {
        std::vector<Entry> data;
        Int signs(n,1);
        // The CSR view of A is only needed to collect columns touching a PSD block.
        Int rowstart, csrcol;
        Vec csrv;
        auto collect=[&](Block& B) {
            const auto c=B.c;
            if(rowstart.empty()) {
                rowstart.assign(m+1,0);csrcol.resize(p.A.nz());csrv.resize(p.A.nz());
                for(int k=0;k<p.A.nz();++k) ++rowstart[p.A.i[k]+1];
                for(int k=0;k<m;++k) rowstart[k+1]+=rowstart[k];
                Int pos=rowstart;
                for(int j=0;j<n;++j)
                    for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                        int t=pos[p.A.i[k]]++;
                        csrcol[t]=j;csrv[t]=p.A.v[k];
                    }
            }
            std::vector<std::tuple<int,int,double>> tmp;
            for(int j=0;j<c.width;++j)
                for(int k=rowstart[c.off+j];k<rowstart[c.off+j+1];++k)
                    tmp.emplace_back(csrcol[k],j,csrv[k]);
            std::sort(tmp.begin(),tmp.end());
            int last=-1;
            for(const auto& item:tmp) {
                const auto [j,k,v]=item;
                if(j!=last){B.cols.push_back(j);B.ap.push_back(int(B.av.size()));last=j;}
                B.av.push_back({k,v});
            }
            B.ap.push_back(int(B.av.size()));
        };
        for(auto c:p.cones) {
            blocks.emplace_back(c);
            Block& B=blocks.back();
            if(c.kind==PSD) {
                const int d=c.dim;
                maxpsd=std::max(maxpsd,d);
                degree+=d;
                if(d>=7 && !noelim) {
                    collect(B);
                    const double nz=B.av.size(), nc=B.cols.size();
                    // Sparse basis products or a dense transformed-basis Gram matrix.
                    B.eliminated=nc<=2400 && (nc<.90*c.width || c.width>=2500) &&
                       (nz*nz<1.8e8 || nc*c.width<1.5e7);
                    if(!B.eliminated) {B.cols.clear();B.ap.clear();B.av.clear();}
                }
                if(!B.eliminated && double(c.width)*c.width>26000000) return;
                B.psd=std::make_unique<PSDScale>(d);
                for(int j=0;j<d;++j) for(int i=0;i<=j;++i){B.ri.push_back(i);B.rj.push_back(j);}
            } else if(c.kind==Soc) {
                ++degree;
                B.soc=std::make_unique<SOCScale>(c.width);
                if(rotate_soc && c.width<=24) {
                    collect(B);
                    B.rotated=true;
                    B.ratio.resize(c.width);
                }
            } else if(c.kind==Linear || c.kind==Exp) {
                degree+=c.width;
                if(c.kind==Linear) B.ratio.resize(c.width);
            }
            if(!B.eliminated) {
                B.first=dim;
                for(int j=0;j<c.width;++j) map[c.off+j]=B.rotated?-1:dim+j;
                dim+=c.width;
                signs.insert(signs.end(),c.width,-1);
                if(c.kind==Soc && !B.rotated && c.width>(robust?2:12)) {
                    B.aux=dim; dim+=2; signs.push_back(1);signs.push_back(-1);
                }
            } else eliminates=true;
        }
        size_t reserve=size_t(dim)+p.A.nz()+p.P.nz()/2+1;
        for(auto& B:blocks) {
            int k=B.eliminated ? int(B.cols.size()) : B.c.width;
            if(B.rotated) reserve+=B.cols.size()*k;
            else if(B.aux>=0) reserve+=2*size_t(k);
            else if(B.c.kind>=Soc) reserve+=size_t(k)*(k+1)/2;
        }
        if(reserve>28000000) return;
        data.reserve(reserve);
        auto add=[&](int i,int j,double x=0.) {
            data.push_back({i,j,x});return int(data.size())-1;
        };
        for(int j=0;j<dim;++j) add(j,j);
        for(int j=0;j<n;++j) {
            for(int k=p.P.p[j];k<p.P.p[j+1];++k)
                if(p.P.i[k]==j) data[j].v+=p.P.v[k];
                else if(p.P.i[k]<j) add(p.P.i[k],j,p.P.v[k]);
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)
                if(map[p.A.i[k]]>=0) add(j,map[p.A.i[k]],p.A.v[k]);
        }
        for(auto& B:blocks) {
            auto c=B.c;
            if(c.kind==Linear || c.kind==Zero) continue;
            if(B.rotated) {
                B.ids.reserve(B.cols.size()*c.width);
                for(int j:B.cols) for(int i=0;i<c.width;++i) B.ids.push_back(add(j,B.first+i));
            } else if(B.eliminated) {
                const int k=int(B.cols.size());
                B.ids.reserve(size_t(k)*(k+1)/2);
                for(int j=0;j<k;++j) for(int i=0;i<=j;++i) B.ids.push_back(add(B.cols[i],B.cols[j]));
            } else if(B.aux>=0) {
                B.ids.reserve(2*size_t(c.width));
                for(int j=0;j<c.width;++j) {
                    B.ids.push_back(add(B.first+j,B.aux));
                    B.ids.push_back(add(B.first+j,B.aux+1));
                }
                data[B.aux].v=1.;
                data[B.aux+1].v=-1.;
            } else {
                B.ids.reserve(size_t(c.width)*(c.width+1)/2);
                for(int j=0;j<c.width;++j)
                    for(int i=0;i<=j;++i) B.ids.push_back(i==j ? B.first+i : add(B.first+i,B.first+j));
            }
        }
        ls=std::make_unique<LinearSystem>(dim,signs,data,dim==n,dense,external);
        if(!ls->ok) return;
        values.reserve(data.size());
        for(const auto& e:data) values.push_back(e.v);
        rr.resize(dim);ans.resize(dim);
        ok=true;
        qr_allowed=eliminates && dim==n && p.P.nz()==0 && n>0 && m>=n &&
            double(m)*n*n<=3.0e8;
    }

    static double psd_entry(const Block& B,const Vec& w,int i,int j) {
        const int a=B.ri[i], b=B.rj[i], c=B.ri[j], d=B.rj[j], n=B.c.dim;
        double f=w[a+c*n]*w[b+d*n];
        if(a==b && c==d) return f;
        if(a==b||c==d) return root2*f;
        return f+w[a+d*n]*w[b+c*n];
    }

    static void transformed_columns(Block& B,double* T,int leading) {
        const size_t cols=B.cols.size(), d=B.c.dim, sq=d*d;
        Vec full(sq),tmp(sq),a(sq);
        const auto& gi=B.psd->gi;
        for(size_t j=0;j<cols;++j) {
            if(B.ap[j+1]-B.ap[j]<int(sq/5)) {
                std::fill(tmp.begin(),tmp.end(),0.);
                for(int k=B.ap[j];k<B.ap[j+1];++k) {
                    int row=B.av[k].row, i=B.ri[row], l=B.rj[row];
                    const double v=B.av[k].value*(i==l?1.:invroot2);
                    const double* u=gi.data()+i*d;
                    double* t=tmp.data()+l*d;
                    for(size_t a=0;a<d;++a)t[a]+=v*u[a];
                    if(i!=l) {
                        u=gi.data()+l*d;t=tmp.data()+i*d;
                        for(size_t a=0;a<d;++a)t[a]+=v*u[a];
                    }
                }
                gemm("N","T",int(d),int(d),int(d),tmp.data(),d,gi.data(),d,full.data(),d);
            } else {
                std::fill(a.begin(),a.end(),0.);
                for(int k=B.ap[j];k<B.ap[j+1];++k) {
                    int row=B.av[k].row, i=B.ri[row], l=B.rj[row];
                    double v=B.av[k].value*(i==l?1.:invroot2);
                    a[i+l*d]=a[l+i*d]=v;
                }
                congruence(int(d),gi.data(),a.data(),full.data(),tmp.data());
            }
            svec(int(d),full.data(),T+j*leading);
        }
    }

    void schur(Block& B) {
        const int cols=int(B.cols.size()), wid=B.c.width, d=B.c.dim;
        if(cols==0) return;
        const double nz=B.av.size();
        const bool gramroute=(nz*nz>.25*cols*cols*wid+.35*cols*d*d*d || nz*nz>2e7)
            && (double(cols)*wid<1.6e7);
        if(gramroute) {
            schurwork.resize(size_t(cols)*wid + size_t(cols)*cols);
            double* T=schurwork.data(),*M=T+size_t(cols)*wid;
            transformed_columns(B,T,wid);
            double one=1,zero=0;
            scipy_dsyrk_("U","T",&cols,&wid,&one,T,&wid,&zero,M,&cols);
            int k=0;
            for(int j=0;j<cols;++j) for(int i=0;i<=j;++i) values[B.ids[k++]]=M[i+size_t(j)*cols];
        } else {
            int k=0;
            for(int j=0;j<cols;++j) for(int i=0;i<=j;++i) {
                double a=0;
                for(int t=B.ap[j];t<B.ap[j+1];++t)
                    for(int u=B.ap[i];u<B.ap[i+1];++u)
                        a+=B.av[t].value*B.av[u].value*
                            psd_entry(B,B.psd->wi,B.av[t].row,B.av[u].row);
                values[B.ids[k++]]=a;
            }
        }
    }

    bool factor_qr() {
        qrf.assign(size_t(n)*m,0.);
        qrpiv.assign(n,0);qrtau.resize(n);qrvec.resize(m);qrx.resize(n);qrw.resize(n);
        qrwork.resize(size_t(160)*n+256);
        for(auto& B:blocks) {
            const int w=B.c.width;
            if(int(B.cols.size())==n) transformed_columns(B,qrf.data()+B.c.off,m);
            else {
                schurwork.resize(size_t(w)*B.cols.size());
                transformed_columns(B,schurwork.data(),w);
                for(size_t j=0;j<B.cols.size();++j) {
                    auto t=schurwork.data()+j*w;
                    std::copy(t,t+w,qrf.data()+size_t(m)*B.cols[j]+B.c.off);
                }
            }
        }
        int info=0,lw=int(qrwork.size());
        scipy_dgeqp3_(&m,&n,qrf.data(),&m,qrpiv.data(),qrtau.data(),qrwork.data(),&lw,&info);
        if(info) return false;
        double threshold=std::max(1e-30,std::abs(qrf[0])*1e-14);
        for(qrrank=0;qrrank<n;++qrrank)
            if(!(std::abs(qrf[qrrank+size_t(qrrank)*m])>threshold))break;
        return qrrank>0;
    }

    bool solve_qr(const Vec& rd,const Vec& rp,Vec& dx,Vec& dz) {
        int one=1, info=0, lw=int(qrwork.size());
        for(auto& B:blocks) {
            auto& S=*B.psd;
            int d=S.d, off=B.c.off;
            smat(d,rp.data()+off,S.a.data());
            congruence(d,S.gi.data(),S.a.data(),S.b.data(),S.tmp.data());
            svec(d,S.b.data(),qrvec.data()+off);
        }
        scipy_dormqr_("L","T",&m,&one,&qrrank,qrf.data(),&m,qrtau.data(),qrvec.data(),
                     &m,qrwork.data(),&lw,&info);
        if(info) return false;
        for(int j=0;j<qrrank;++j) {
            int k=qrpiv[j]-1;
            const double* col=qrf.data()+size_t(j)*m;
            qrw[j]=(rd[k]-dot(qrw.data(),col,j))/col[j];
            qrx[j]=qrvec[j]+qrw[j];
        }
        std::fill(dx.begin(),dx.end(),0.);
        for(int j=qrrank-1;j>=0;--j) {
            const double* col=qrf.data()+size_t(j)*m;
            const double v=qrx[j]/col[j];
            dx[qrpiv[j]-1]=v;
            for(int k=0;k<j;++k) qrx[k]-=col[k]*v;
        }
        for(int j=0;j<qrrank;++j) qrvec[j]=qrw[j];
        for(int j=qrrank;j<m;++j) qrvec[j]=-qrvec[j];
        scipy_dormqr_("L","N",&m,&one,&qrrank,qrf.data(),&m,qrtau.data(),qrvec.data(),
                     &m,qrwork.data(),&lw,&info);
        if(info) return false;
        for(auto& B:blocks) {
            auto& S=*B.psd;
            int d=S.d, off=B.c.off;
            smat(d,qrvec.data()+off,S.a.data());
            congruence(d,S.gi.data(),S.a.data(),S.b.data(),S.tmp.data(),true);
            svec(d,S.b.data(),dz.data()+off);
        }
        return finite(dx) && finite(dz);
    }

    bool update(const Vec& s,const Vec& z,double mu,bool secant,double reg,bool extended=false,
                bool prefer_qr=false,bool two_secant=false) {
        qr_active=qr_allowed && prefer_qr;
        for(auto& B:blocks) {
            auto c=B.c;
            const double* ss=s.data()+c.off,*zz=z.data()+c.off;
            switch(c.kind) {
            case Zero: break;
            case Linear:
                for(int i=0;i<c.width;++i) {
                    if(!(ss[i]>0&&zz[i]>0)) return false;
                    B.ratio[i]=ss[i]/zz[i];
                    values[B.first+i]=-B.ratio[i];
                }
                break;
            case Soc: {
                if(!B.soc->update(ss,zz)) return false;
                auto& S=*B.soc;
                double eta2=S.eta*S.eta;
                if(B.rotated) {
                    S.rotation();
                    double h=S.v[0]+S.tail;
                    B.ratio[0]=eta2*h*h;
                    B.ratio[1]=eta2/h/h;
                    for(int j=2;j<c.width;++j) B.ratio[j]=eta2;
                    for(int j=0;j<c.width;++j) values[B.first+j]=-B.ratio[j];
                    int pos=0;
                    for(size_t j=0;j<B.cols.size();++j) {
                        std::fill(S.a.begin(),S.a.end(),0.);
                        for(int k=B.ap[j];k<B.ap[j+1];++k) S.a[B.av[k].row]=B.av[k].value;
                        S.rotate(S.a.data(),S.b.data());
                        for(double a:S.b) values[B.ids[pos++]]=a;
                    }
                } else if(B.aux>=0) {
                    double h=S.v[0]+S.tail;
                    double t=S.eta*std::sqrt(S.tail*h);
                    double u=S.eta*std::sqrt(std::max(0.,.5-.5/(h*h)));
                    values[B.ids[0]]=t;values[B.ids[1]]=u;
                    values[B.first]=-eta2;
                    for(int i=1;i<c.width;++i) {
                        double a=S.tail>0?S.v[i]/S.tail:0.;
                        values[B.first+i]=-eta2;
                        values[B.ids[2*i]]=t*a;
                        values[B.ids[2*i+1]]=-u*a;
                    }
                } else {
                    int k=0;
                    for(int j=0;j<c.width;++j) for(int i=0;i<=j;++i)
                        values[B.ids[k++]]=-eta2*(2*S.v[i]*S.v[j]+(i==j ? (i==0?-1.:1.) : 0.));
                }
                break;
            }
            case Exp: {
                if(!B.exp.update(ss,zz,mu,secant,two_secant)) return false;
                int k=0;
                for(int j=0;j<3;++j) for(int i=0;i<=j;++i) values[B.ids[k++]]=-B.exp.h[i+j*3];
                break;
            }
            default:
                if(!B.psd->update(ss,zz)) return false;
                if(B.eliminated) { if(!qr_active)schur(B); }
                else {
                    int k=0;
                    for(int j=0;j<c.width;++j) for(int i=0;i<=j;++i)
                        values[B.ids[k++]]=-psd_entry(B,B.psd->w,i,j);
                }
            }
        }
        if(qr_active) {
            if(factor_qr()) return true;
            qr_active=false;
            for(auto& B:blocks)schur(B);
        }
        if(ls->factor(values,reg,extended))return true;
        return ls->factor(values,std::max(1e-9,reg*30),true) ||
            ls->factor(values,std::max(5e-8,reg*300),true);
    }

    bool solve(const Vec& rd,const Vec& rp,Vec& dx,Vec& dz,int refine=4) {
        if(qr_active)return solve_qr(rd,rp,dx,dz);
        std::copy(rd.begin(),rd.end(),rr.begin());
        for(auto& B:blocks) {
            auto c=B.c;
            if(B.eliminated) {
                temp.resize(c.width);
                B.psd->apply(rp.data()+c.off,temp.data(),true);
                for(size_t j=0;j<B.cols.size();++j) {
                    double a=0;
                    for(int k=B.ap[j];k<B.ap[j+1];++k) a+=B.av[k].value*temp[B.av[k].row];
                    rr[B.cols[j]]+=a;
                }
            } else if(B.rotated) {
                B.soc->rotate(rp.data()+c.off,rr.data()+B.first);
            } else {
                std::copy(rp.begin()+c.off,rp.begin()+c.off+c.width,rr.begin()+B.first);
                if(B.aux>=0)rr[B.aux]=rr[B.aux+1]=0;
            }
        }
        if(!ls->solve(rr,ans,refine)) return false;
        std::copy(ans.begin(),ans.begin()+n,dx.begin());
        for(auto& B:blocks) {
            auto c=B.c;
            if(B.eliminated) {
                temp.resize(c.width);
                for(int j=0;j<c.width;++j) temp[j]=-rp[c.off+j];
                for(size_t j=0;j<B.cols.size();++j) {
                    const double a=dx[B.cols[j]];
                    for(int k=B.ap[j];k<B.ap[j+1];++k) temp[B.av[k].row]+=B.av[k].value*a;
                }
                B.psd->apply(temp.data(),dz.data()+c.off,true);
            } else if(B.rotated) B.soc->unrotate(ans.data()+B.first,dz.data()+c.off);
            else std::copy(ans.begin()+B.first,ans.begin()+B.first+c.width,dz.begin()+c.off);
        }
        return finite(dz);
    }

    double energy(const Vec& z) {
        double answer=0;
        for(auto& B:blocks) {
            auto c=B.c;
            const double* zz=z.data()+c.off;
            switch(c.kind) {
            case Zero: break;
            case Linear:
                for(int j=0;j<c.width;++j) answer+=zz[j]*zz[j]*B.ratio[j];
                break;
            case Soc:
                if(B.rotated) {
                    B.soc->rotate(zz,B.soc->t.data());
                    for(int j=0;j<c.width;++j)
                        answer+=B.soc->t[j]*B.soc->t[j]*B.ratio[j];
                } else answer+=B.soc->energy(zz);
                break;
            case PSD:
                B.psd->apply(zz,B.psd->tmp2.data());
                answer+=dot(zz,B.psd->tmp2.data(),c.width);
                break;
            default:
                for(int j=0;j<3;++j)
                    for(int i=0;i<3;++i) answer+=zz[i]*zz[j]*B.exp.h[i+3*j];
            }
        }
        return answer;
    }

    void direction_slack(const Vec& z, const Vec& cone, Vec& s) {
        for(auto& B:blocks) {
            auto c=B.c;
            const double* zz=z.data()+c.off;
            const double* cc=cone.data()+c.off;
            double* ss=s.data()+c.off;
            switch(c.kind) {
            case Zero:
                std::fill(ss,ss+c.width,0.);
                break;
            case Linear:
                for(int i=0;i<c.width;++i) ss[i]=cc[i]-B.ratio[i]*zz[i];
                break;
            case Soc:
                if(B.rotated) {
                    B.soc->rotate(zz,B.soc->t.data());
                    for(int j=0;j<c.width;++j) B.soc->t[j]*=B.ratio[j];
                    B.soc->unrotate(B.soc->t.data(),ss);
                } else {
                    B.soc->apply_w(zz,B.soc->t.data());
                    B.soc->apply_w(B.soc->t.data(),ss);
                }
                for(int i=0;i<c.width;++i) ss[i]=cc[i]-ss[i];
                break;
            case Exp:
                for(int i=0;i<3;++i) {
                    long double v=cc[i];
                    for(int j=0;j<3;++j) v-=(long double)B.exp.h[i+3*j]*zz[j];
                    ss[i]=double(v);
                }
                break;
            default:
                if(!B.eliminated) {
                    B.psd->apply(zz,ss);
                    for(int i=0;i<c.width;++i) ss[i]=cc[i]-ss[i];
                }
                break;
            }
        }
    }
};

} // namespace conic
