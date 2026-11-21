#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
static Clock::time_point start_time;
static double elapsed() {
    return chrono::duration<double>(Clock::now() - start_time).count();
}
static double option(const char* name, double def) {
    const char* s = getenv(name);
    return s ? atof(s) : def;
}

struct Edge {
    int v;
    double len, support;
};
struct Tree {
    int n = 0;
    vector<vector<Edge>> adj;
    explicit Tree(int n_ = 0) : n(n_), adj(n_) {}
    int node() { adj.emplace_back(); return (int)adj.size()-1; }
    void add(int a, int b, double len=1, double support=1) {
        adj[a].push_back({b,len,support});
        adj[b].push_back({a,len,support});
    }
    void remove(int a, int b) {
        auto& v = adj[a];
        v.erase(remove_if(v.begin(),v.end(),[&](const Edge& e){return e.v==b;}),v.end());
        auto& w = adj[b];
        w.erase(remove_if(w.begin(),w.end(),[&](const Edge& e){return e.v==a;}),w.end());
    }
    void swap_branches(int u, int a, int v, int b) {
        remove(u,a); remove(v,b);
        add(u,b); add(v,a);
    }
    void suppress_root(int r) {
        if ((int)adj[r].size()!=2) return;
        Edge a=adj[r][0], b=adj[r][1];
        remove(r,a.v); remove(r,b.v);
        add(a.v,b.v,a.len+b.len,max(a.support,b.support));
    }
};

struct Parser {
    const string& s;
    size_t pos=0;
    Tree t;
    unordered_map<string,int>& ids;
    bool create;
    vector<unsigned char> seen;
    int leaf_count=0,depth=0;
    Parser(const string& s_, unordered_map<string,int>& ids_, bool create_)
        : s(s_), t((int)ids_.size()), ids(ids_), create(create_),seen(ids_.size()) {}
    void space() {
        while (pos<s.size()) {
            if (isspace((unsigned char)s[pos])) {++pos; continue;}
            if (s[pos]=='[') {
                int depth=1; ++pos;
                while(pos<s.size() && depth) {
                    if(s[pos]=='[') ++depth;
                    if(s[pos]==']') --depth;
                    ++pos;
                }
                continue;
            }
            break;
        }
    }
    string label() {
        space();
        string z;
        if(pos<s.size() && s[pos]=='\'') {
            ++pos;
            while(pos<s.size()) {
                char c=s[pos++];
                if(c=='\'') {
                    if(pos<s.size() && s[pos]=='\'') {z+='\''; ++pos;}
                    else break;
                } else z+=c;
            }
        } else {
            while(pos<s.size() && !isspace((unsigned char)s[pos]) &&
                  string("(),:;[]").find(s[pos])==string::npos) z+=s[pos++];
        }
        return z;
    }
    pair<int,Edge> subtree() {
        if(++depth>512)throw runtime_error("Newick nesting is too deep");
        space();
        int v;
        bool internal = pos<s.size() && s[pos]=='(';
        if (internal) {
            ++pos;
            v=t.node();
            while(true) {
                auto ch=subtree();
                t.add(v,ch.first,ch.second.len,ch.second.support);
                space();
                if(pos>=s.size()) throw runtime_error("Unfinished Newick tree");
                if(s[pos]==')') {++pos; break;}
                if(s[pos++]!=',') throw runtime_error("Expected comma");
            }
        } else {
            string name=label();
            auto it=ids.find(name);
            if(it==ids.end()) throw runtime_error("Unknown taxon: "+name);
            v=it->second;
            if(seen[v])throw runtime_error("Duplicate taxon: "+name);
            seen[v]=1;leaf_count++;
        }
        double support=-1, len=0;
        space();
        if(internal && pos<s.size() && string("(),:;").find(s[pos])==string::npos) {
            string a=label();
            char* end=nullptr;
            double x=strtod(a.c_str(),&end);
            if(end!=a.c_str() && *end==0 && x>=0 && x<=1) support=x;
        }
        space();
        if(pos<s.size() && s[pos]==':') {
            ++pos; string a=label();
            char* end=nullptr;
            double x=strtod(a.c_str(),&end);
            if(end==a.c_str() || *end!=0 || !isfinite(x) || x<0)
                throw runtime_error("Invalid branch length");
            len=x==0?0:x;
        }
        depth--;
        return {v,{v,len,support}};
    }
    Tree parse() {
        auto r=subtree();
        space();
        if(pos>=s.size() || s[pos]!=';') throw runtime_error("Missing semicolon");
        t.suppress_root(r.first);
        if(leaf_count!=t.n)throw runtime_error("Gene trees have different taxon sets");
        for(int v=0;v<(int)t.adj.size();++v) {
            size_t degree=t.adj[v].size();
            if((v<t.n && degree!=1) || (v>=t.n && degree && degree!=3))
                throw runtime_error("Expected an unrooted binary tree");
        }
        return t;
    }
};

static vector<string> read_labels(const string& s) {
    vector<string> names;
    bool leaf=true;
    size_t p=0;
    while(p<s.size()) {
        if(isspace((unsigned char)s[p])) {++p;continue;}
        char c=s[p++];
        if(c=='[') {
            int depth=1;
            while(p<s.size() && depth) {
                if(s[p]=='[')depth++;
                if(s[p]==']')depth--;
                ++p;
            }
            continue;
        }
        if(c=='(' || c==',') {leaf=true;continue;}
        if(c==')' || c==':' || c==';') {leaf=false;continue;}
        string a;
        if(c=='\'') {
            while(p<s.size()) {
                char d=s[p++];
                if(d=='\'') {
                    if(p<s.size() && s[p]=='\'') {a+='\''; ++p;}
                    else break;
                } else a+=d;
            }
        } else {
            a+=c;
            while(p<s.size() && !isspace((unsigned char)s[p]) &&
                  string("(),:;[]").find(s[p])==string::npos) a+=s[p++];
        }
        if(leaf) {
            if(a.empty())throw runtime_error("Empty taxon label");
            names.push_back(a);
        }
        leaf=false;
    }
    sort(names.begin(),names.end(),[](const string& a,const string& b){
        if(a=="0")return b!="0";
        if(b=="0")return false;
        return a<b;
    });
    if(adjacent_find(names.begin(),names.end())!=names.end()) throw runtime_error("Duplicate taxa");
    return names;
}

static string quote_name(const string& name) {
    if(name.find_first_of(" \t\n\r(),:;[]'")==string::npos)return name;
    string out="'";
    for(char c:name){out+=c; if(c=='\'')out+='\'';}
    return out+"'";
}
static string newick(const Tree& t,const vector<string>& labels) {
    int r=-1;
    for(int i=t.n;i<(int)t.adj.size();++i)if(t.adj[i].size()==3){r=i;break;}
    if(r<0)throw runtime_error("No internal node");
    function<string(int,int)> rec=[&](int v,int p) {
        if(v<t.n)return quote_name(labels[v]);
        string s="("; bool first=true;
        for(auto e:t.adj[v])if(e.v!=p) {
            if(!first)s+=",";
            s+=rec(e.v,v);first=false;
        }
        return s+")";
    };
    return rec(r,-1)+";\n";
}

static void write_tree(const string& path,const Tree& tree,const vector<string>& labels) {
    auto parent=filesystem::path(path).parent_path();
    if(!parent.empty())filesystem::create_directories(parent);
    ofstream out(path);
    if(!out)throw runtime_error("Cannot open output");
    out<<newick(tree,labels);out.close();
    if(!out)throw runtime_error("Cannot write output");
}

static double branching_log_prior(const Tree& t) {
    double value=0;
    function<int(int,int)> visit=[&](int u,int parent) {
        if(u<t.n)return 1;
        int size=0;
        for(auto e:t.adj[u])if(e.v!=parent)size+=visit(e.v,u);
        value-=log(max(1,size-1));
        return size;
    };
    visit(t.adj[0][0].v,0);
    return value;
}

static void impute_support(vector<Tree>& genes) {
    int zero=0,other=0;
    for(Tree& t:genes) {
        vector<pair<double,double>> known;
        double norm=0;int edges=0;
        for(int u=0;u<(int)t.adj.size();++u)for(const auto& e:t.adj[u])if(u<e.v) {
            norm+=e.len;edges++;
            if(u>=t.n && e.v>=t.n && e.support>=0)
                known.emplace_back(log(max(e.len,1e-8)),clamp((e.support-1.0/3)*1.5,0.0,1.0));
        }
        norm=max(1e-10,norm/max(1,edges));
        for(int u=t.n;u<(int)t.adj.size();++u)for(auto& e:t.adj[u])
            if(e.v>u && e.support<0) {
                double p=0;
                if(e.len<=1e-5)zero++;
                else {
                    other++;
                    double x=log(e.len),sum=0,denom=0;
                    for(auto [y,s]:known) {
                        double w=exp(-.5*pow((x-y)/.6,2));
                        sum+=w*s;denom+=w;
                    }
                    p=denom>1e-10?sum/denom:1-exp(-4*e.len/norm);
                }
                e.support=(1+2*p)/3;
                for(auto& back:t.adj[e.v])if(back.v==u)back.support=e.support;
            }
    }
    cerr<<"Unreported supports: "<<zero<<" near-zero edges, "<<other<<" calibrated edges\n";
}

struct Gene {
    int n,m;
    vector<int> parent, left, right, order, merge_order;
    vector<double> len, sup, q, decay;
    explicit Gene(const Tree& t) : n(t.n), m((int)t.adj.size()), parent(m,-1),
        left(m,-1),right(m,-1),len(m),sup(m),q(m),decay(m) {
        order.push_back(0); parent[0]=0;
        for(size_t k=0;k<order.size();++k) {
            int u=order[k];
            for(auto e:t.adj[u])if(e.v!=parent[u]) {
                int v=e.v; parent[v]=u;len[v]=e.len;sup[v]=e.support;
                if(left[u]<0)left[u]=v;else right[u]=v;
                order.push_back(v);
            }
        }
    }
    void weights(double lambda,double power,double floor) {
        double norm=0;
        int count=0;
        for(int v:order)if(v){norm+=len[v];count++;}
        norm=max(1e-10,norm/max(1,count));
        int mode=(int)option("WEIGHT_MODE",0);
        double rate=option("LENGTH_RATE",1);
        bool relative=option("RELATIVE_DECAY",0)>0;
        double support_decay=clamp(option("SUP_DECAY",0),0.0,1.0);
        for(int v:order) {
            double p=sup[v]<0 ? 1 : max(0.0,min(1.0,(sup[v]-1.0/3)*1.5));
            double terminal_confidence=p;
            if(mode==1)p=1-exp(-rate*len[v]/norm);
            if(mode==2)p*=1-exp(-rate*len[v]/norm);
            if(mode==3)p=1-exp(-rate*len[v]);
            if(mode==4)p=p>=option("SUP_THRESHOLD",.5)?1:0;
            if(mode==5) {
                double beta=option("LENGTH_MIX",.5);
                p=(1-beta)*p+beta*(1-exp(-rate*len[v]/norm));
            }
            p=pow(p,power);
            p=floor+(1-floor)*p;
            if(option("ZERO_COLLAPSE",0)>0 && len[v]<=1e-10)p=0;
            q[v]=1-p;
            decay[v]=exp(-lambda*len[v]/(relative?norm:1));
            if(v>=n && parent[v]>=n)decay[v]*=1-support_decay*(1-terminal_confidence);
        }
        merge_order.clear();
        for(int v:order)if(v>=n && parent[v]>=n)merge_order.push_back(v);
        sort(merge_order.begin(),merge_order.end(),[&](int a,int b){return q[a]>q[b];});
    }
};

using Matrix = vector<vector<double>>;
static Matrix distance_matrix(const vector<Tree>& genes,int mode,double param,
                              const vector<double>& weights={}) {
    int n=genes[0].n;
    Matrix d(n,vector<double>(n,0));
    for(int gi=0;gi<(int)genes.size();++gi) {
        const Tree& t=genes[gi];
        double weight=weights.empty()?1:weights[gi];
        if(weight==0)continue;
        double norm=0;int num=0;
        for(int u=0;u<(int)t.adj.size();++u)for(auto e:t.adj[u])if(u<e.v){
            norm+=e.len;num++;
        }
        norm=max(1e-10,norm/num);
        for(int i=0;i<n;++i) {
            function<void(int,int,double)> dfs=[&](int u,int par,double sum) {
                if(u<n)d[i][u]+=sum*weight;
                for(auto e:t.adj[u])if(e.v!=par){
                    double p=e.support<0?1:max(0.0,min(1.0,(e.support-1.0/3)*1.5));
                    double w;
                    if(mode==0)w=1;
                    else if(mode==1)w=param+(1-param)*p;
                    else if(mode==2)w=e.len/norm;
                    else if(mode==3)w=pow(e.len/norm,param);
                    else w=(1-param)*p+param*e.len/norm;
                    dfs(e.v,u,sum+w);
                }
            };
            dfs(i,-1,0);
        }
    }
    double norm=weights.empty()?genes.size():accumulate(weights.begin(),weights.end(),0.0);
    for(auto& row:d)for(double& x:row)x/=norm;
    return d;
}

static void quantile_distances(Matrix& d,const vector<Tree>& genes,int mode,double param,
                               double quantile,double mix) {
    vector<Matrix> distances;
    for(const auto& g:genes)distances.push_back(distance_matrix({g},mode,param));
    vector<double> sample(genes.size());
    double position=clamp(quantile,0.0,1.0)*(genes.size()-1);
    int low=(int)floor(position),high=(int)ceil(position);
    for(int i=0;i<(int)d.size();++i)for(int j=0;j<i;++j) {
        for(int k=0;k<(int)genes.size();++k)sample[k]=distances[k][i][j];
        sort(sample.begin(),sample.end());
        double q=sample[low]+(position-low)*(sample[high]-sample[low]);
        d[i][j]=d[j][i]=(1-mix)*d[i][j]+mix*q;
    }
}

static Tree nj(const Matrix& input,bool bionj=false) {
    int n=(int)input.size(),lim=2*n;
    Matrix d(lim,vector<double>(lim)),v(lim,vector<double>(lim));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)d[i][j]=v[i][j]=input[i][j];
    Tree t(n);
    vector<int> active(n);iota(active.begin(),active.end(),0);
    vector<double> sums(lim);
    while(active.size()>3) {
        int m=(int)active.size();
        for(int a:active) {
            sums[a]=0;
            for(int b:active)sums[a]+=d[a][b];
        }
        double best=1e300;
        int ai=-1,bi=-1;
        for(int x=0;x<m;++x)for(int y=0;y<x;++y) {
            int a=active[x],b=active[y];
            double z=(m-2)*d[a][b]-sums[a]-sums[b];
            if(z<best){best=z;ai=x;bi=y;}
        }
        int a=active[ai],b=active[bi],u=t.node();
        double la=0.5*(d[a][b]+(sums[a]-sums[b])/(m-2)),lb=d[a][b]-la;
        t.add(u,a,max(0.0,la));t.add(u,b,max(0.0,lb));
        double lambda=0.5;
        if(bionj && v[a][b]>1e-10) {
            double diff=0;
            for(int c:active)if(c!=a && c!=b)diff+=v[b][c]-v[a][c];
            lambda=clamp(0.5+diff/(2*(m-2)*v[a][b]),0.0,1.0);
        }
        for(int c:active)if(c!=a && c!=b) {
            d[u][c]=d[c][u]=lambda*(d[a][c]-la)+(1-lambda)*(d[b][c]-lb);
            v[u][c]=v[c][u]=lambda*v[a][c]+(1-lambda)*v[b][c]-lambda*(1-lambda)*v[a][b];
        }
        active.erase(active.begin()+ai);active.erase(active.begin()+bi);active.push_back(u);
    }
    int r=t.node();
    for(int a:active)t.add(r,a);
    return t;
}

static Tree upgma(const Matrix& input,bool equal=false) {
    int n=(int)input.size();
    Matrix d(2*n,vector<double>(2*n));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)d[i][j]=input[i][j];
    Tree t(n);
    vector<int> active(n),sizes(2*n,1);iota(active.begin(),active.end(),0);
    while(active.size()>1) {
        int ai=-1,bi=-1;
        double best=1e300;
        for(int i=0;i<(int)active.size();++i)for(int j=0;j<i;++j)
            if(d[active[i]][active[j]]<best){best=d[active[i]][active[j]];ai=i;bi=j;}
        int a=active[ai],b=active[bi],u=t.node();
        t.add(u,a);t.add(u,b);
        sizes[u]=sizes[a]+sizes[b];
        double lambda=equal?.5:double(sizes[a])/sizes[u];
        for(int c:active)if(c!=a && c!=b)d[u][c]=d[c][u]=lambda*d[a][c]+(1-lambda)*d[b][c];
        active.erase(active.begin()+ai);active.erase(active.begin()+bi);active.push_back(u);
    }
    t.suppress_root(active[0]);return t;
}

static constexpr int PA[6]={0,0,0,1,1,2};
static constexpr int PB[6]={1,2,3,2,3,3};
struct State {
    double c[4]{}, f[6]{}, p[6]{};
};
static inline State merge_state(const State& a,const State& b,double q,double decay) {
    State out;
    for(int k=0;k<4;++k)out.c[k]=(a.c[k]+b.c[k])*decay;
    for(int k=0;k<6;++k) {
        double cross=a.c[PA[k]]*b.c[PB[k]]+a.c[PB[k]]*b.c[PA[k]];
        out.f[k]=a.f[k]+b.f[k]+cross;
        out.p[k]=(a.p[k]+b.p[k]+cross)*q;
    }
    return out;
}
struct QuartetKey {
    array<uint64_t,8> x{};
    bool operator==(const QuartetKey& b) const {return x==b.x;}
};
struct QuartetKeyHash {
    size_t operator()(const QuartetKey& k) const {
        uint64_t h=0x243f6a8885a308d3ULL;
        for(auto v:k.x) {
            v^=v>>30;v*=0xbf58476d1ce4e5b9ULL;v^=v>>27;
            h^=v+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        }
        return h;
    }
};
struct QuartetScorer {
    vector<Gene> genes;
    vector<double> weights;
    vector<double> leaf_weights;
    const Matrix* bme=nullptr;
    const Matrix* hybrid_distance=nullptr;
    double hybrid_scale=0;
    double balance=0;
    vector<State> down,up;
    bool max_mode;
    vector<int> repr;
    vector<array<double,6>> boundary;
    bool caching=false;
    size_t cache_hits=0,cache_misses=0;
    unordered_map<QuartetKey,vector<double>,QuartetKeyHash> cache;
    explicit QuartetScorer(const vector<Tree>& trees,double lambda,double power,double floor) {
        max_mode=option("MAX_WEIGHT",0)>0;
        for(const auto& t:trees) {
            genes.emplace_back(t);
            genes.back().weights(lambda,power,floor);
        }
        weights.assign(genes.size(),1);
        leaf_weights.assign(trees[0].n,1);
        int m=0;for(auto& g:genes)m=max(m,g.m);
        down.resize(m);up.resize(m);
        repr.resize(m);boundary.resize(m);
    }
    template<bool TRI> array<double,3> score_max(const vector<int>& color) {
        array<double,3> ans{};
        double total[4]{};
        for(int i=0;i<(int)color.size();++i)total[color[i]]+=leaf_weights[i];
        auto pairs=[](const double* c) {
            array<double,6> p;
            if constexpr(TRI) {
                p={c[0]*(c[0]-1)*.5,c[1]*(c[1]-1)*.5,c[2]*(c[2]-1)*.5,
                   c[0]*c[1],c[0]*c[2],c[1]*c[2]};
            } else {
                for(int k=0;k<6;++k)p[k]=c[PA[k]]*c[PB[k]];
            }
            return p;
        };
        auto find=[&](int v) {
            int r=v;
            while(repr[r]!=r)r=repr[r];
            while(repr[v]!=v){int w=repr[v];repr[v]=r;v=w;}
            return r;
        };
        for(int gi=0;gi<(int)genes.size();++gi) {
            if(weights[gi]==0)continue;
            const auto& g=genes[gi];
            auto before=ans;
            for(int v=0;v<g.n;++v) {
                for(int k=0;k<4;++k)down[v].c[k]=0;
                down[v].c[color[v]]=leaf_weights[v];
            }
            for(int i=(int)g.order.size()-1;i>0;--i) {
                int u=g.order[i];if(u<g.n)continue;
                int a=g.left[u],b=g.right[u];
                for(int k=0;k<4;++k)down[u].c[k]=down[a].c[k]+down[b].c[k];
                double outside[4];
                for(int k=0;k<4;++k)outside[k]=total[k]-down[u].c[k];
                auto pa=pairs(down[a].c),pb=pairs(down[b].c),pc=pairs(outside);
                for(int k=0;k<6;++k)boundary[u][k]=pa[k]+pb[k]+pc[k];
                repr[u]=u;
            }
            // Contract edges in increasing confidence. A quartet disappears
            // exactly when the most confident edge of its central path is removed.
            for(int v:g.merge_order) {
                int u=g.parent[v],ru=find(u),rv=find(v);
                double outside[4];
                for(int k=0;k<4;++k)outside[k]=total[k]-down[v].c[k];
                auto a=pairs(outside),b=pairs(down[v].c);
                double pu[6],pv[6];
                for(int k=0;k<6;++k) {
                    pu[k]=a[k]+b[k]-boundary[ru][k];
                    pv[k]=a[k]+b[k]-boundary[rv][k];
                    boundary[ru][k]+=boundary[rv][k]-a[k]-b[k];
                }
                double w=1-g.q[v];
                if constexpr(TRI) {
                    for(int k=0;k<6;++k)ans[0]+=w*pu[k]*pv[5-k];
                } else {
                    ans[0]+=w*(pu[0]*pv[5]+pu[5]*pv[0]);
                    ans[1]+=w*(pu[1]*pv[4]+pu[4]*pv[1]);
                    ans[2]+=w*(pu[2]*pv[3]+pu[3]*pv[2]);
                }
                repr[rv]=ru;
            }
            if(weights[gi]!=1)for(int k=0;k<3;++k)ans[k]=before[k]+(ans[k]-before[k])*weights[gi];
        }
        return ans;
    }
    array<double,3> score_gene(const Gene& g,const vector<int>& color) {
        array<double,3> ans{};
        for(int v=0;v<g.n;++v) {
            down[v]=State{};
            down[v].c[color[v]]=g.decay[v]*leaf_weights[v];
        }
        for(int i=(int)g.order.size()-1;i>0;--i) {
            int u=g.order[i];
            if(u<g.n)continue;
            down[u]=merge_state(down[g.left[u]],down[g.right[u]],g.q[u],g.decay[u]);
        }
        int root=g.left[0];
        up[root]=State{};
        up[root].c[color[0]]=g.decay[root]*leaf_weights[0];
        for(int u:g.order) {
            if(u<g.n)continue;
            int a=g.left[u],b=g.right[u];
            const State* sides[3]={&down[a],&down[b],&up[u]};
            for(int x=0;x<3;++x) {
                const State& s=*sides[x];
                const State& y=*sides[(x+1)%3];
                const State& z=*sides[(x+2)%3];
                double cross[6],r[6];
                for(int k=0;k<6;++k) {
                    cross[k]=y.c[PA[k]]*z.c[PB[k]]+y.c[PB[k]]*z.c[PA[k]];
                    r[k]=s.f[k]-s.p[k];
                }
                ans[0]+=r[0]*cross[5]+r[5]*cross[0];
                ans[1]+=r[1]*cross[4]+r[4]*cross[1];
                ans[2]+=r[2]*cross[3]+r[3]*cross[2];
            }
            if(a>=g.n)up[a]=merge_state(up[u],down[b],g.q[a],g.decay[a]);
            if(b>=g.n)up[b]=merge_state(up[u],down[a],g.q[b],g.decay[b]);
        }
        for(double& a:ans)a*=.5;
        return ans;
    }
    array<double,3> distance_score(const vector<int>& color,const Matrix& matrix) const {
        double pair[4][4]{};
        for(int i=0;i<(int)color.size();++i)for(int j=0;j<i;++j)
            if(color[i]!=color[j]) {
                int a=color[i],b=color[j];if(a>b)swap(a,b);
                pair[a][b]+=leaf_weights[i]*leaf_weights[j]*matrix[i][j];
            }
        return {-.25*(pair[0][1]+pair[2][3]),-.25*(pair[0][2]+pair[1][3]),
                -.25*(pair[0][3]+pair[1][2])};
    }
    array<double,3> finish_score(array<double,3> ans,const vector<int>& color) const {
        if(hybrid_distance) {
            auto d=distance_score(color,*hybrid_distance);
            for(int k=0;k<3;++k)ans[k]+=hybrid_scale*d[k];
        }
        return ans;
    }
    array<double,3> score(const vector<int>& color) {
        if(bme)return distance_score(color,*bme);
        if(max_mode)return finish_score(score_max<false>(color),color);
        array<double,3> ans{};
        if(caching && balance==0) {
            QuartetKey key;
            int map[4]={-1,-1,-1,-1},next=0;
            for(int i=0;i<(int)color.size();++i) {
                int c=color[i];
                if(map[c]<0)map[c]=next++;
                key.x[i/32]|=uint64_t(map[c])<<(2*(i%32));
            }
            auto it=cache.find(key);
            if(it==cache.end() && cache.size()<250000) {
                vector<int> canonical(color.size());
                for(int i=0;i<(int)color.size();++i)canonical[i]=map[color[i]];
                vector<double> scores(genes.size()*3);
                for(int i=0;i<(int)genes.size();++i) {
                    auto s=score_gene(genes[i],canonical);
                    for(int k=0;k<3;++k)scores[i*3+k]=s[k];
                }
                it=cache.emplace(key,move(scores)).first;cache_misses++;
            } else cache_hits++;
            if(it!=cache.end()) {
                auto index=[](int a,int b) {
                    if(a>b)swap(a,b);
                    return a==0?b-1:5-a-b;
                };
                int perm[3]={index(map[0],map[1]),index(map[0],map[2]),index(map[0],map[3])};
                const auto& values=it->second;
                for(int i=0;i<(int)genes.size();++i)for(int k=0;k<3;++k)
                    ans[k]+=weights[i]*values[i*3+perm[k]];
                return finish_score(ans,color);
            }
        }
        for(int gi=0;gi<(int)genes.size();++gi) {
            if(weights[gi]==0)continue;
            auto s=score_gene(genes[gi],color);
            for(int k=0;k<3;++k)ans[k]+=s[k]*weights[gi];
        }
        return finish_score(ans,color);
    }
    double tripartition(const vector<int>& color) {
        if(max_mode)return score_max<true>(color)[0]*.5;
        double ans=0;
        auto combine=[](const State& a,const State& b,double q,double decay) {
            State out;
            for(int k=0;k<3;++k)out.c[k]=(a.c[k]+b.c[k])*decay;
            double cross[6]={a.c[0]*b.c[0],a.c[1]*b.c[1],a.c[2]*b.c[2],
                a.c[0]*b.c[1]+a.c[1]*b.c[0],a.c[0]*b.c[2]+a.c[2]*b.c[0],
                a.c[1]*b.c[2]+a.c[2]*b.c[1]};
            for(int k=0;k<6;++k) {
                out.f[k]=a.f[k]+b.f[k]+cross[k];
                out.p[k]=(a.p[k]+b.p[k]+cross[k])*q;
            }
            return out;
        };
        for(int gi=0;gi<(int)genes.size();++gi) {
            if(weights[gi]==0)continue;
            const auto& g=genes[gi];
            double before=ans;
            for(int v=0;v<g.n;++v) {
                down[v]=State{};
                down[v].c[color[v]]=g.decay[v]*leaf_weights[v];
            }
            for(int i=(int)g.order.size()-1;i>0;--i) {
                int u=g.order[i];
                if(u<g.n)continue;
                down[u]=combine(down[g.left[u]],down[g.right[u]],g.q[u],g.decay[u]);
            }
            int root=g.left[0];
            up[root]=State{};
            up[root].c[color[0]]=g.decay[root]*leaf_weights[0];
            for(int u:g.order) {
                if(u<g.n)continue;
                int a=g.left[u],b=g.right[u];
                const State* sides[3]={&down[a],&down[b],&up[u]};
                for(int x=0;x<3;++x) {
                    const State& s=*sides[x];
                    const State& y=*sides[(x+1)%3];
                    const State& z=*sides[(x+2)%3];
                    double cross[6]={y.c[0]*z.c[0],y.c[1]*z.c[1],y.c[2]*z.c[2],
                        y.c[0]*z.c[1]+y.c[1]*z.c[0],y.c[0]*z.c[2]+y.c[2]*z.c[0],
                        y.c[1]*z.c[2]+y.c[2]*z.c[1]};
                    for(int k=0;k<6;++k)ans+=(s.f[k]-s.p[k])*cross[5-k];
                }
                if(a>=g.n)up[a]=combine(up[u],down[b],g.q[a],g.decay[a]);
                if(b>=g.n)up[b]=combine(up[u],down[a],g.q[b],g.decay[b]);
            }
            if(weights[gi]!=1)ans=before+(ans-before)*weights[gi];
        }
        // Each agreement has two endpoints in each of the two trees.
        return ans*.25;
    }
};

static void color_side(const Tree& t,int v,int par,int col,vector<int>& colors) {
    if(v<t.n){colors[v]=col;return;}
    for(auto e:t.adj[v])if(e.v!=par)color_side(t,e.v,v,col,colors);
}
static int color_balanced(const Tree& t,int v,int par,int col,vector<int>& colors,
                          vector<double>& weights,double weight=1) {
    if(v<t.n){colors[v]=col;weights[v]=weight;return 1;}
    int count=0;
    int children=0;for(auto e:t.adj[v])if(e.v!=par)children++;
    double child_weight=weight/children;
    for(auto e:t.adj[v])if(e.v!=par)
        count+=color_balanced(t,e.v,v,col,colors,weights,child_weight);
    return count;
}
static int nni_refine(Tree& t,QuartetScorer& scorer,int rounds,double deadline,bool quiet=false) {
    int changes=0;
    vector<int> color(t.n);
    for(int iter=0;iter<rounds && elapsed()<deadline;++iter) {
        int edits=0;
        vector<pair<int,int>> edges;
        for(int u=t.n;u<(int)t.adj.size();++u)for(auto e:t.adj[u])
            if(e.v>u)edges.emplace_back(u,e.v);
        for(auto [u,v]:edges) {
            if(elapsed()>deadline)break;
            vector<int> a,b;
            for(auto e:t.adj[u])if(e.v!=v)a.push_back(e.v);
            for(auto e:t.adj[v])if(e.v!=u)b.push_back(e.v);
            if(a.size()!=2 || b.size()!=2)continue;
            if(scorer.balance>0) {
                int size[4];
                size[0]=color_balanced(t,a[0],u,0,color,scorer.leaf_weights);
                size[1]=color_balanced(t,a[1],u,1,color,scorer.leaf_weights);
                size[2]=color_balanced(t,b[0],v,2,color,scorer.leaf_weights);
                size[3]=color_balanced(t,b[1],v,3,color,scorer.leaf_weights);
                if(scorer.balance<1)for(int i=0;i<t.n;++i)
                    scorer.leaf_weights[i]=scorer.balance*scorer.leaf_weights[i]+(1-scorer.balance)/size[color[i]];
            } else {
                color_side(t,a[0],u,0,color);
                color_side(t,a[1],u,1,color);
                color_side(t,b[0],v,2,color);
                color_side(t,b[1],v,3,color);
            }
            auto s=scorer.score(color);
            int best=(int)(max_element(s.begin(),s.end())-s.begin());
            double tolerance=1e-7+1e-10*max(abs(s[0]),abs(s[best]));
            if(best && s[best]>s[0]+tolerance) {
                t.swap_branches(u,a[1],v,b[best==1?0:1]);
                edits++;changes++;
            }
        }
        if(!quiet)cerr<<"NNI "<<iter+1<<": "<<edits<<" changes, "<<elapsed()<<" s\n";
        if(!edits)break;
    }
    fill(scorer.leaf_weights.begin(),scorer.leaf_weights.end(),1);
    return changes;
}

static double tree_score(const Tree& t,QuartetScorer& scorer) {
    if(scorer.bme) {
        double score=0;
        for(int i=0;i<t.n;++i) {
            function<void(int,int,int)> dfs=[&](int u,int par,int depth) {
                if(u<t.n && u>i)score-=ldexp((*scorer.bme)[i][u],1-depth);
                for(auto e:t.adj[u])if(e.v!=par)dfs(e.v,u,depth+1);
            };
            dfs(i,-1,0);
        }
        return score;
    }
    vector<int> color(t.n);
    double score=0;
    for(int u=t.n;u<(int)t.adj.size();++u)if(t.adj[u].size()==3) {
        for(int k=0;k<3;++k)color_side(t,t.adj[u][k].v,u,k,color);
        score+=scorer.tripartition(color);
    }
    return score;
}

static int spr_refine(Tree& t,QuartetScorer& scorer,int rounds,int radius,double deadline,
                      mt19937& rng) {
    int changes=0;
    vector<int> color(t.n);
    for(int iter=0;iter<rounds && elapsed()<deadline;++iter) {
        vector<int> parent(t.adj.size(),-1),sizes(t.adj.size(),0),order={0};
        parent[0]=0;
        for(size_t i=0;i<order.size();++i) {
            int u=order[i];
            for(auto e:t.adj[u])if(e.v!=parent[u]) {
                parent[e.v]=u;order.push_back(e.v);
            }
        }
        for(int i=(int)order.size()-1;i>=0;--i) {
            int v=order[i];
            if(v<t.n)sizes[v]++;
            if(v) sizes[parent[v]]+=sizes[v];
        }
        vector<int> candidates;
        for(int v:order)if(v && parent[v]>=t.n && sizes[v]<t.n-3)candidates.push_back(v);
        shuffle(candidates.begin(),candidates.end(),rng);
        int edits=0,checks=0;
        double gain_sum=0;
        for(int s:candidates) {
            if(elapsed()>deadline)break;
            int r=parent[s];
            bool adjacent=false;
            for(auto e:t.adj[s])if(e.v==r)adjacent=true;
            if(!adjacent || t.adj[r].size()!=3)continue;
            int b=-1,c=-1;
            for(auto e:t.adj[r])if(e.v!=s) {if(b<0)b=e.v;else c=e.v;}
            t.remove(r,s);t.remove(r,b);t.remove(r,c);t.add(b,c);
            fill(color.begin(),color.end(),0);
            if(scorer.balance>0)color_balanced(t,s,r,0,color,scorer.leaf_weights);
            double best=0;
            int best_a=b,best_b=c;
            function<void(int,int,double,int)> visit=[&](int v,int par,double current,int depth) {
                if(v<t.n || depth>=radius || elapsed()>deadline)return;
                int x=-1,y=-1;
                for(auto e:t.adj[v])if(e.v!=par){if(x<0)x=e.v;else y=e.v;}
                if(scorer.balance>0) {
                    color_balanced(t,par,v,1,color,scorer.leaf_weights);
                    color_balanced(t,x,v,2,color,scorer.leaf_weights);
                    color_balanced(t,y,v,3,color,scorer.leaf_weights);
                } else {
                    color_side(t,par,v,1,color);
                    color_side(t,x,v,2,color);
                    color_side(t,y,v,3,color);
                }
                auto q=scorer.score(color);++checks;
                double sx=current+q[1]-q[0],sy=current+q[2]-q[0];
                double tolerance=1e-7+1e-10*max({abs(q[0]),abs(q[1]),abs(q[2])});
                if(sx>best+tolerance){best=sx;best_a=v;best_b=x;}
                if(sy>best+tolerance){best=sy;best_a=v;best_b=y;}
                visit(x,v,sx,depth+1);visit(y,v,sy,depth+1);
            };
            visit(b,c,0,0);visit(c,b,0,0);
            t.remove(best_a,best_b);
            t.add(r,s);t.add(r,best_a);t.add(r,best_b);
            if(best>1e-7) {
                edits++;changes++;gain_sum+=best;
                // The next move must use the current, not the original, orientation.
                fill(parent.begin(),parent.end(),-1);order={0};parent[0]=0;
                for(size_t i=0;i<order.size();++i) {
                    int u=order[i];
                    for(auto e:t.adj[u])if(e.v!=parent[u]) {
                        parent[e.v]=u;order.push_back(e.v);
                    }
                }
            }
        }
        cerr<<"SPR "<<iter+1<<": "<<edits<<" changes, "<<checks<<" scores, gain "
            <<gain_sum<<", "<<elapsed()<<" s\n";
        if(!edits)break;
    }
    fill(scorer.leaf_weights.begin(),scorer.leaf_weights.end(),1);
    return changes;
}

struct Mask {
    array<uint64_t,4> x{};
    bool operator==(const Mask& b) const {return x==b.x;}
    bool operator<(const Mask& b) const {return x<b.x;}
    int size() const {
        int n=0;for(auto a:x)n+=__builtin_popcountll(a);return n;
    }
    bool subset(const Mask& b) const {
        for(int k=0;k<4;++k)if(x[k]&~b.x[k])return false;
        return true;
    }
    Mask operator|(const Mask& b) const {
        Mask c;for(int k=0;k<4;++k)c.x[k]=x[k]|b.x[k];return c;
    }
    Mask operator^(const Mask& b) const {
        Mask c;for(int k=0;k<4;++k)c.x[k]=x[k]^b.x[k];return c;
    }
    void color(vector<int>& colors,int c) const {
        for(int k=0;k<4;++k) {
            uint64_t w=x[k];
            while(w){int j=__builtin_ctzll(w);colors[k*64+j]=c;w&=w-1;}
        }
    }
};
struct MaskHash {
    size_t operator()(const Mask& m) const {
        uint64_t h=0x9e3779b97f4a7c15ULL;
        for(uint64_t a:m.x) {
            a^=a>>30;a*=0xbf58476d1ce4e5b9ULL;
            a^=a>>27;a*=0x94d049bb133111ebULL;a^=a>>31;
            h^=a+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        }
        return h;
    }
};
struct Cluster {
    Mask mask;
    int size=0, a=-1, b=-1;
    double frequency=0, score=-1e300;
    vector<pair<int,int>> splits;
};
struct ClusterPool {
    int n;
    bool canonical_order=false;
    vector<Cluster> clusters;
    unordered_map<Mask,int,MaskHash> index;
    explicit ClusterPool(int n_) : n(n_) {index.reserve(20000);}
    int add(const Mask& mask,double weight) {
        auto it=index.find(mask);
        if(it!=index.end()){clusters[it->second].frequency+=weight;return it->second;}
        int id=(int)clusters.size();
        Cluster c;c.mask=mask;c.size=mask.size();c.frequency=weight;
        clusters.push_back(c);index.emplace(mask,id);return id;
    }
    void add_tree(const Tree& t,double weight=1) {
        function<Mask(int,int)> rec=[&](int u,int par) {
            Mask m;
            if(u<t.n)m.x[u/64]|=1ULL<<(u%64);
            else for(auto e:t.adj[u])if(e.v!=par)m=m|rec(e.v,u);
            add(m,weight);return m;
        };
        rec(t.adj[0][0].v,0);
    }
    int root_id() const {
        Mask full;
        for(int i=1;i<n;++i)full.x[i/64]|=1ULL<<(i%64);
        return index.at(full);
    }
    void count_current(double weight=1) {
        function<void(int)> rec=[&](int i) {
            auto& c=clusters[i];
            if(c.size==1)return;
            c.frequency+=weight;
            rec(c.a);rec(c.b);
        };
        rec(root_id());
    }
    double current_log_prior() const {
        double value=0;
        function<void(int)> rec=[&](int i) {
            const auto& c=clusters[i];
            if(c.size==1)return;
            value-=log(c.size-1);
            rec(c.a);rec(c.b);
        };
        rec(root_id());return value;
    }
    vector<int> prepare() {
        vector<int> order(clusters.size());iota(order.begin(),order.end(),0);
        stable_sort(order.begin(),order.end(),[&](int a,int b){
            if(clusters[a].size!=clusters[b].size)return clusters[a].size<clusters[b].size;
            return canonical_order && clusters[a].mask<clusters[b].mask;
        });
        long long count=0;
        for(int ci:order) {
            auto& c=clusters[ci];
            c.splits.clear();
            if(c.size==1)continue;
            for(int ai:order) {
                const auto& a=clusters[ai];
                if(a.size>c.size/2)break;
                if(!a.mask.subset(c.mask))continue;
                auto it=index.find(c.mask^a.mask);
                if(it==index.end())continue;
                int bi=it->second;
                if(a.size*2==c.size && ai>bi)continue;
                c.splits.emplace_back(ai,bi);++count;
            }
        }
        cerr<<"Cluster search: "<<clusters.size()<<" clades, "<<count<<" resolutions\n";
        return order;
    }
    bool optimize(Tree& result,QuartetScorer& scorer,bool consensus,double deadline) {
        auto order=prepare();
        vector<int> color(n);
        for(int ci:order) {
            auto& c=clusters[ci];
            c.score=c.size==1?0:-1e300;c.a=c.b=-1;
            for(auto [ai,bi]:c.splits) {
                if(elapsed()>deadline)return false;
                double score=clusters[ai].score+clusters[bi].score;
                if(consensus)score+=c.frequency;
                else {
                    fill(color.begin(),color.end(),2);
                    clusters[ai].mask.color(color,0);clusters[bi].mask.color(color,1);
                    score+=scorer.tripartition(color);
                }
                if(score>c.score){c.score=score;c.a=ai;c.b=bi;}
            }
        }
        Mask full;
        for(int i=1;i<n;++i)full.x[i/64]|=1ULL<<(i%64);
        int root=index.at(full);
        if(clusters[root].a<0)return false;
        Tree answer(n);
        function<int(int)> rec=[&](int i) {
            const auto& c=clusters[i];
            if(c.size==1) {
                for(int k=0;k<4;++k)if(c.mask.x[k])return 64*k+__builtin_ctzll(c.mask.x[k]);
            }
            int u=answer.node(),a=rec(c.a),b=rec(c.b);
            answer.add(u,a);answer.add(u,b);return u;
        };
        int u=rec(root);answer.add(0,u);
        result=move(answer);
        cerr<<"Cluster optimum: "<<setprecision(14)<<clusters[root].score
            <<", "<<elapsed()<<" s\n";
        return true;
    }
};

static void bootstrap_consensus(Tree& result,const vector<Tree>& genes,QuartetScorer& scorer,
                                int reps,int mode,double param,double deadline) {
    ClusterPool pool(result.n);
    pool.add_tree(result,option("BOOT_PRIOR",1));
    Tree base=result;
    double tree_prior=option("BOOT_YULE",0);
    double log_scale=tree_prior*branching_log_prior(base);
    auto importance_weight=[&](double log_prior) {
        double value=tree_prior*log_prior;
        if(value>log_scale) {
            double scale=exp(log_scale-value);
            for(auto& c:pool.clusters)c.frequency*=scale;
            log_scale=value;
        }
        return exp(value-log_scale);
    };
    mt19937 rng((unsigned)option("SEED",1)+1729);
    exponential_distribution<double> exponential(1);
    scorer.caching=option("BOOT_CACHE",1)>0;
    if(scorer.caching)scorer.cache.reserve(100000);
    vector<Matrix> distances;
    if(option("PRECOMPUTE_DIST",1)>0) {
        distances.reserve(genes.size());
        for(const auto& g:genes)distances.push_back(distance_matrix({g},mode,param));
    }
    int completed=0;
    auto resample=[&]() {
        scorer.weights.assign(genes.size(),0);
        if(option("BOOT_TYPE",0)==0) {
            for(size_t k=0;k<genes.size();++k)scorer.weights[rng()%genes.size()]++;
        } else {
            for(double& w:scorer.weights)w=exponential(rng);
            double scale=genes.size()/accumulate(scorer.weights.begin(),scorer.weights.end(),0.0);
            for(double& w:scorer.weights)w*=scale;
        }
    };
    bool dp_boot=option("BOOT_DP",0)>0;
    int initial_reps=dp_boot?min(reps,(int)option("BOOT_WARMUP",30)):reps;
    for(int r=0;r<initial_reps && elapsed()<deadline-1;++r) {
        resample();
        Tree sample=base;
        int start=(int)option("BOOT_START",0);
        if(start==0 || (start==2 && r%2==0)) {
            Matrix d;
            if(distances.empty())d=distance_matrix(genes,mode,param,scorer.weights);
            else {
                int n=result.n;
                d.assign(n,vector<double>(n,0));
                double sum=accumulate(scorer.weights.begin(),scorer.weights.end(),0.0);
                for(int gi=0;gi<(int)genes.size();++gi)if(scorer.weights[gi]>0) {
                    for(int i=0;i<n;++i)for(int j=0;j<n;++j)
                        d[i][j]+=scorer.weights[gi]*distances[gi][i][j];
                }
                for(auto& row:d)for(double& x:row)x/=sum;
            }
            sample=nj(d);
        }
        nni_refine(sample,scorer,(int)option("BOOT_NNI",12),deadline-1,true);
        if(option("BOOT_BALANCE",0)>0) {
            scorer.balance=option("BOOT_BALANCE",0);
            nni_refine(sample,scorer,(int)option("BOOT_BALANCE_ROUNDS",3),deadline-1,true);
            scorer.balance=0;
        }
        if(option("BOOT_SPR",0)>0)
            spr_refine(sample,scorer,1,(int)option("BOOT_SPR",0),deadline-1,rng);
        pool.add_tree(sample,tree_prior==0?1:importance_weight(branching_log_prior(sample)));
        completed++;
        if(completed%10==0 || completed==initial_reps)
            cerr<<"Resample "<<completed<<"/"<<initial_reps<<", "<<elapsed()<<" s\n";
    }
    scorer.weights.assign(genes.size(),1);
    cerr<<"Quartet cache: "<<scorer.cache_hits<<" hits, "<<scorer.cache_misses<<" entries\n";
    if(dp_boot && elapsed()<deadline-5) {
        if(option("BOOT_DP_GENES",1)>0)for(const auto& t:genes)pool.add_tree(t,0);
        if(option("BOOT_POOL_EXTRA",0)>0) {
            for(auto [m,p]:vector<pair<int,double>>{{1,.1},{2,0},{3,.25},{3,.5}}) {
                if(elapsed()>deadline-5)break;
                Matrix d=distance_matrix(genes,m,p);
                if(m==3 && p==.5)quantile_distances(d,genes,m,p,.25,.5);
                Tree init=nj(d);
                pool.add_tree(init,0);
                nni_refine(init,scorer,12,deadline-4,true);
                pool.add_tree(init,0);
            }
        }
        auto order=pool.prepare();
        vector<size_t> offset(pool.clusters.size());
        size_t resolutions=0;
        for(int ci:order) {
            offset[ci]=resolutions;resolutions+=pool.clusters[ci].splits.size();
        }
        size_t ng=genes.size();
        vector<double> scores(resolutions*ng);
        vector<int> color(result.n);
        bool ready=true;
        scorer.weights.assign(ng,0);
        for(int ci:order) {
            const auto& c=pool.clusters[ci];
            for(size_t si=0;si<c.splits.size();++si) {
                if(elapsed()>deadline-4){ready=false;break;}
                auto [a,b]=c.splits[si];
                fill(color.begin(),color.end(),2);
                pool.clusters[a].mask.color(color,0);
                pool.clusters[b].mask.color(color,1);
                size_t base_index=(offset[ci]+si)*ng;
                for(size_t gi=0;gi<ng;++gi) {
                    scorer.weights[gi]=1;
                    scores[base_index+gi]=scorer.tripartition(color);
                    scorer.weights[gi]=0;
                }
            }
            if(!ready)break;
        }
        scorer.weights.assign(ng,1);
        if(ready) {
            cerr<<"Resampling DP: "<<resolutions<<" cached tripartitions, "<<elapsed()<<" s\n";
            for(auto& c:pool.clusters)c.frequency=0;
            pool.add_tree(base,option("BOOT_PRIOR",1));
            log_scale=tree_prior*branching_log_prior(base);
            rng.seed((unsigned)option("SEED",1)+1729);
            int samples=(int)option("BOOT_DP_REPS",reps);
            completed=0;
            for(int r=0;r<samples && elapsed()<deadline-1;++r) {
                resample();
                for(int ci:order) {
                    auto& c=pool.clusters[ci];
                    c.score=c.size==1?0:-1e300;c.a=c.b=-1;
                    for(size_t si=0;si<c.splits.size();++si) {
                        auto [a,b]=c.splits[si];
                        double z=pool.clusters[a].score+pool.clusters[b].score;
                        const double* s=&scores[(offset[ci]+si)*ng];
                        for(size_t gi=0;gi<ng;++gi)z+=scorer.weights[gi]*s[gi];
                        if(z>c.score){c.score=z;c.a=a;c.b=b;}
                    }
                }
                pool.count_current(tree_prior==0?1:importance_weight(pool.current_log_prior()));
                completed++;
            }
            cerr<<"Exact constrained resamples: "<<completed<<", "<<elapsed()<<" s\n";
        }
        scorer.weights.assign(ng,1);
    }
    if(option("BOOT_QUARTET",0)>0) {
        for(const auto& t:genes)pool.add_tree(t,0);
        pool.optimize(result,scorer,false,deadline);
    } else {
        pool.optimize(result,scorer,true,deadline+3);
    }
}

int main(int argc,char** argv) {
    start_time=Clock::now();
    try {
        if(argc>=4 && string(argv[1])=="--consensus") {
            vector<Tree> predictions;
            vector<string> labels;
            unordered_map<string,int> ids;
            for(int i=3;i<argc;++i) {
                ifstream in(argv[i]);
                if(!in)throw runtime_error("Cannot open consensus input");
                string text((istreambuf_iterator<char>(in)),istreambuf_iterator<char>());
                if(text.compare(0,3,"\xEF\xBB\xBF")==0)text.erase(0,3);
                if(predictions.empty()) {
                    labels=read_labels(text);
                    if(labels.size()<4 || labels.size()>256)throw runtime_error("Unsupported taxon count");
                    for(int j=0;j<(int)labels.size();++j)ids[labels[j]]=j;
                }
                predictions.push_back(Parser(text,ids,false).parse());
            }
            Tree result=predictions[0];
            QuartetScorer scorer({result},0,1,0);
            ClusterPool pool(result.n);
            pool.canonical_order=true;
            for(int i=0;i<(int)predictions.size();++i)pool.add_tree(predictions[i],i==0?1.000001:1);
            pool.optimize(result,scorer,true,120);
            write_tree(argv[2],result,labels);
            return 0;
        }
        if(argc!=3)throw runtime_error("Usage: run.sh INPUT OUTPUT");
        ifstream in(argv[1]);
        if(!in)throw runtime_error("Cannot open input");
        vector<Tree> trees;
        vector<string> labels;
        unordered_map<string,int> ids;
        string text((istreambuf_iterator<char>(in)),istreambuf_iterator<char>());
        if(text.compare(0,3,"\xEF\xBB\xBF")==0)text.erase(0,3);
        size_t begin=0;
        bool quoted=false;
        int comment_depth=0;
        for(size_t p=0;p<text.size();++p) {
            if(!quoted && text[p]=='['){comment_depth++;continue;}
            if(comment_depth) {
                if(text[p]==']')comment_depth--;
                continue;
            }
            if(text[p]=='\'')quoted=!quoted;
            if(text[p]!=';' || quoted)continue;
            string line=text.substr(begin,p-begin+1);begin=p+1;
            if(trees.empty()) {
                labels=read_labels(line);
                if(labels.size()<4 || labels.size()>256)throw runtime_error("Unsupported taxon count");
                for(int i=0;i<(int)labels.size();++i)ids[labels[i]]=i;
            }
            trees.push_back(Parser(line,ids,false).parse());
        }
        if(trees.empty() || labels.size()<4)throw runtime_error("No valid gene trees");
        if(option("MISSING_SUPPORT",1)>0)impute_support(trees);
        int mode=(int)option("DIST_MODE",3);
        double param=option("DIST_PARAM",0.5);
        Matrix dist=distance_matrix(trees,mode,param);
        if(option("DIST_QUANTILE",-1)>=0)
            quantile_distances(dist,trees,mode,param,option("DIST_QUANTILE",.25),option("QUANTILE_MIX",1));
        if(option("FARRIS",0)>0) {
            vector<double> outgroup=dist[0];
            for(int i=0;i<(int)dist.size();++i)for(int j=0;j<(int)dist.size();++j)
                if(i!=j)dist[i][j]-=outgroup[i]+outgroup[j];
        }
        Tree result=option("UPGMA",0)>0?upgma(dist,option("UPGMA",0)>1):nj(dist,option("BIONJ",0)>0);
        if(option("BME_ROUNDS",0)>0) {
            QuartetScorer scorer(trees,0,1,0);
            scorer.bme=&dist;scorer.balance=1;
            double deadline=clamp(option("TIME_LIMIT",120),1.0,120.0);
            if(option("DEBUG_SCORE",0))cerr<<"Before BME: "<<setprecision(14)<<tree_score(result,scorer)<<"\n";
            nni_refine(result,scorer,(int)option("BME_ROUNDS",20),deadline);
            mt19937 rng((unsigned)option("SEED",1));
            spr_refine(result,scorer,(int)option("BME_SPR",2),(int)option("BME_RADIUS",6),deadline,rng);
            nni_refine(result,scorer,(int)option("BME_ROUNDS",20),deadline);
            if(option("DEBUG_SCORE",0))cerr<<"After BME: "<<setprecision(14)<<tree_score(result,scorer)<<"\n";
        }
        if(option("NNI_ROUNDS",30)>0) {
            QuartetScorer scorer(trees,option("LAMBDA",0),option("SUP_POWER",1),option("SUP_FLOOR",.25));
            double deadline=clamp(option("TIME_LIMIT",120),1.0,120.0);
            nni_refine(result,scorer,(int)option("NNI_ROUNDS",30),deadline);
            if(option("SPR_ROUNDS",2)>0) {
                mt19937 rng((unsigned)option("SEED",1));
                if(option("DEBUG_SCORE",0))cerr<<"Before SPR score: "<<setprecision(14)<<tree_score(result,scorer)<<"\n";
                spr_refine(result,scorer,(int)option("SPR_ROUNDS",2),(int)option("SPR_RADIUS",4),deadline,rng);
                nni_refine(result,scorer,(int)option("NNI_ROUNDS",30),deadline);
            }
            if(option("DP_MODE",0)>0 && elapsed()<deadline-2) {
                ClusterPool pool((int)labels.size());
                for(const Tree& t:trees)pool.add_tree(t);
                pool.add_tree(result);
                for(auto [m,p]:vector<pair<int,double>>{{0,0},{1,.1},{2,0},{3,.25},{3,.5}}) {
                    Tree init=nj(distance_matrix(trees,m,p));
                    if(option("DP_INITIAL_NNI",1)>0)nni_refine(init,scorer,12,deadline);
                    pool.add_tree(init);
                }
                bool completed=pool.optimize(result,scorer,option("DP_MODE",0)==2,deadline);
                cerr<<"Cluster search "<<(completed?"finished":"stopped at deadline")<<"\n";
                if(completed)nni_refine(result,scorer,(int)option("NNI_ROUNDS",30),deadline);
            }
            if(option("BOOT_REPS",0)>0 && elapsed()<deadline-2)
                bootstrap_consensus(result,trees,scorer,(int)option("BOOT_REPS",0),mode,param,
                                    deadline-(option("FINAL_BALANCE",0)>0?3:0));
            if(option("FINAL_BALANCE",0)>0) {
                scorer.balance=option("FINAL_BALANCE",0);
                if(option("DISTANCE_PRIOR",0)>0) {
                    double mean=0;
                    for(const auto& row:dist)mean+=accumulate(row.begin(),row.end(),0.0);
                    mean/=labels.size()*(labels.size()-1);
                    scorer.hybrid_distance=&dist;
                    scorer.hybrid_scale=option("DISTANCE_PRIOR",0)*trees.size()/max(mean,1e-10);
                }
                nni_refine(result,scorer,(int)option("BALANCE_ROUNDS",4),deadline);
                if(option("BALANCE_SPR",0)>0 && scorer.balance==1) {
                    mt19937 rng((unsigned)option("SEED",1));
                    spr_refine(result,scorer,(int)option("BALANCE_SPR",0),
                               (int)option("BALANCE_RADIUS",4),deadline,rng);
                    nni_refine(result,scorer,(int)option("BALANCE_ROUNDS",4),deadline);
                }
                scorer.balance=0;
                scorer.hybrid_distance=nullptr;
            }
            if(option("DEBUG_SCORE",0))cerr<<"Final score: "<<setprecision(14)<<tree_score(result,scorer)<<"\n";
        }
        write_tree(argv[2],result,labels);
        cerr<<"Finished "<<trees.size()<<" genes, "<<labels.size()<<" taxa in "<<elapsed()<<" s\n";
    } catch(const exception& e) {
        cerr<<"species-tree: "<<e.what()<<"\n";
        return 1;
    }
    return 0;
}
