#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;

struct Arc {
    int to,rev;
    double cap,cost;
};
struct Original {
    int a,b,first=-1,second=-1;
    double base=0.,sign=1.,mag=1.,cap=0.,cost=0.;
};
struct Flow {
    std::vector<std::vector<Arc>> G;
    std::vector<double> p,e;
    std::vector<int> curr;
    std::vector<unsigned char> seen;
    std::deque<int> queue;
    Clock::time_point stop;
    long long work=0,limit;
    double tol;
    Flow(int n,long long ops,double f,Clock::time_point d)
        : G(n),p(n),e(n),curr(n),seen(n),stop(d),limit(ops),tol(f) {}
    int edge(int x,int y,double c,double w) {
        const int j=G[x].size(),k=G[y].size();
        G[x].push_back({y,k,c,w});
        G[y].push_back({x,j,0.,-w});
        return j;
    }
    double flow(int u,int k) const {
        const auto& a=G[u][k];
        return G[a.to][a.rev].cap;
    }
    void push(int u,Arc& a,double val) {
        e[u]-=val;
        e[a.to]+=val;
        a.cap-=val;
        G[a.to][a.rev].cap+=val;
    }
    void activate(int j) {
        if (e[j]>tol && !seen[j]) {
            queue.push_back(j);
            seen[j]=1;
        }
    }
    bool alive() {
        return work<limit && Clock::now()<stop;
    }
    bool refine(double eps) {
        const int n=G.size();
        for (int i=0;i<n;++i) {
            work+=G[i].size();
            for (auto& a:G[i])
                if (a.cap>tol && a.cost+p[i]-p[a.to]<0.) push(i,a,a.cap);
        }
        std::fill(curr.begin(),curr.end(),0);
        for (int i=0;i<n;++i) activate(i);
        while (!queue.empty()) {
            const int u=queue.front();
            queue.pop_front();
            seen[u]=0;
            int& k=curr[u];
            while (e[u]>tol) {
                auto& a=G[u];
                for (; k<static_cast<int>(a.size()); ++k) {
                    if (a[k].cap<=tol) continue;
                    if (a[k].cost+p[u]-p[a[k].to]<0.) {
                        push(u,a[k],std::min(e[u],a[k].cap));
                        activate(a[k].to);
                        if (e[u]<=tol) break;
                    }
                }
                if (e[u]<=tol) break;
                double val=-std::numeric_limits<double>::infinity();
                for (const auto& a:G[u]) if (a.cap>tol)
                    val=std::max(val,p[a.to]-a.cost);
                if (!std::isfinite(val)) return false;
                p[u]=val-std::max(eps,1.e-15*(1.+std::abs(val)));
                k=0;
                work+=2*G[u].size()+1;
                if (work>limit || ((work&4095)<2*a.size()+1 && !alive())) return false;
            }
            if ((++work&1023)==0 && !alive()) return false;
        }
        return alive();
    }
};

int solve(int m,int n,const int* ptr,const int* idx,const double* v,
          const double* cost,const double* rhs,const double* lower,const double* upper,
          double* answer,double* prices,double seconds) {
    const auto now=Clock::now();
    const auto stop=now+std::chrono::microseconds(static_cast<long long>(seconds*1.e6));
    const auto inf=std::numeric_limits<double>::infinity();
    std::vector<std::vector<std::pair<int,signed char>>> links(m);
    for (int j=0;j<n;++j) {
        const int d=ptr[j+1]-ptr[j];
        if (d==2) {
            const int k=ptr[j], u=idx[k], w=idx[k+1];
            const double s=std::abs(v[k]), t=std::abs(v[k+1]);
            if (std::abs(s-t)>1.e-13*std::max(s,t)) return 0;
            const signed char rel=(std::signbit(v[k])==std::signbit(v[k+1]))?-1:1;
            links[u].emplace_back(w,rel);
            links[w].emplace_back(u,rel);
        } else if (d>2) return 0;
    }
    std::vector<signed char> sgn(m,0);
    std::vector<int> stack;
    std::vector<double> b(m+1,0.);
    for (int i=0;i<m;++i) if (!sgn[i]) {
        sgn[i]=1; stack.push_back(i);
        long double total=0., scale=0.;
        const auto before=stack.size();
        for (std::size_t k=0;k<stack.size();++k) {
            int u=stack[k];
            b[u]=sgn[u]*rhs[u];
            total+=b[u]; scale+=std::abs(b[u]);
            for (auto [j,r]:links[u]) {
                int s=sgn[u]*r;
                if (!sgn[j]) { sgn[j]=s; stack.push_back(j); }
                else if (sgn[j]!=s) return 0;
            }
        }
        (void)before;
        stack.clear();
    }
    std::vector<Original> arcs(n);
    double maxcost=0.,sumcap=0.,bmax=0.;
    for (int j=0;j<n;++j) {
        const int d=ptr[j+1]-ptr[j];
        const int k=ptr[j];
        auto& e=arcs[j];
        if (d==0) {
            answer[j]=cost[j]>0.?lower[j]:(cost[j]<0.?upper[j]:std::clamp(0.,lower[j],upper[j]));
            if (!std::isfinite(answer[j])) return 0;
            continue;
        }
        const int a=idx[k],c=d==2?idx[k+1]:m;
        const double f=v[k]*sgn[a];
        e.a=(f>0.?a:c);
        e.b=(f>0.?c:a);
        e.mag=std::abs(f);
        e.cap=inf;
        e.cost=cost[j]/e.mag;
        e.first=-2;
        if (std::isfinite(lower[j])) {
            e.base=lower[j];
            if (std::isfinite(upper[j])) {
                e.cap=(upper[j]-lower[j])*e.mag;
                sumcap+=e.cap;
            }
        } else if (std::isfinite(upper[j])) {
            e.base=upper[j];
            std::swap(e.a,e.b);
            e.sign=-1.;
            e.cost=-e.cost;
        } else e.second=-2;
        b[e.a]-=e.base*e.mag*e.sign;
        b[e.b]+=e.base*e.mag*e.sign;
        maxcost=std::max(maxcost,std::abs(e.cost));
    }
    long double bal=0.;
    for (int i=0;i<m;++i) bal+=b[i];
    b[m]=-bal;
    double sumb=0.;
    for (double f:b) {
        bmax=std::max(bmax,std::abs(f));
        sumb+=std::abs(f);
    }
    if (!std::isfinite(sumb+sumcap+maxcost) || Clock::now()>stop) return 0;
    maxcost=std::max(maxcost,1.e-20);
    const double large=std::max(1.,sumb+sumcap);
    Flow ff(m+1,std::max(250000LL,230LL*n),3.e-14*std::max(1.,bmax),stop);
    ff.e=b;
    for (auto& e:arcs) if (e.first==-2 && e.cap>ff.tol) {
        e.first=ff.edge(e.a,e.b,std::min(large,e.cap),e.cost/maxcost);
        if (e.second==-2) e.second=ff.edge(e.b,e.a,large,-e.cost/maxcost);
    }
    double eps=.23;
    for (int it=0;it<24;++it) {
        if (!ff.refine(eps)) return 0;
        eps*=.23;
        if (eps<3.e-14) break;
    }
    for (int j=0;j<n;++j) {
        auto& e=arcs[j];
        if (e.first==-1) continue;
        double f=(e.first>=0)?ff.flow(e.a,e.first):0.;
        if (e.second>=0) f-=ff.flow(e.b,e.second);
        answer[j]=e.base+e.sign*f/e.mag;
        if (std::isfinite(lower[j]) && std::abs(answer[j]-lower[j])<5.e-13*(1.+std::abs(lower[j])))
            answer[j]=lower[j];
        if (std::isfinite(upper[j]) && std::abs(answer[j]-upper[j])<5.e-13*(1.+std::abs(upper[j])))
            answer[j]=upper[j];
    }
    for (int i=0;i<m;++i) prices[i]=(ff.p[i]-ff.p[m])*maxcost*sgn[i];
    return 1;
}
}

extern "C" int conic_network(int m,int n,const int* ptr,const int* idx,const double* v,
                             const double* cost,const double* rhs,const double* lo,const double* up,
                             double* answer,double* prices,double seconds) {
    try { return solve(m,n,ptr,idx,v,cost,rhs,lo,up,answer,prices,seconds); }
    catch (...) { return 0; }
}
