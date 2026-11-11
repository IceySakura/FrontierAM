#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
static auto startTime = Clock::now();
static double evidenceScale=1;
static int likelihoodMoments=0;
static double momentCap=.5;
static double maximumNormalization=1e300;
double elapsed() { return chrono::duration<double>(Clock::now() - startTime).count(); }

struct Edge {
    int to;
    double len = 0, sup = -1;
};
struct Tree {
    int n = 0;
    vector<vector<Edge>> adj;
    Tree() = default;
    explicit Tree(int n_) : n(n_), adj(n_) {}
    int node() { adj.emplace_back(); return (int)adj.size() - 1; }
    void add(int a, int b, double len = 0, double sup = -1) {
        adj[a].push_back({b,len,sup});
        adj[b].push_back({a,len,sup});
    }
    Edge remove(int a, int b) {
        Edge ret{b,0,-1};
        for (int i = 0; i < (int)adj[a].size(); ++i) if (adj[a][i].to == b) {
            ret = adj[a][i]; adj[a].erase(adj[a].begin() + i); break;
        }
        for (int i = 0; i < (int)adj[b].size(); ++i) if (adj[b][i].to == a) {
            adj[b].erase(adj[b].begin() + i); break;
        }
        return ret;
    }
    void swapBranches(int u, int a, int v, int b) {
        Edge ea = remove(u,a), eb = remove(v,b);
        add(u,b,eb.len,eb.sup); add(v,a,ea.len,ea.sup);
    }
    void compact() {
        vector<int> id(adj.size(),-1);
        for (int i=0;i<n;++i) id[i]=i;
        int k=n;
        for (int i=n;i<(int)adj.size();++i) if (!adj[i].empty()) id[i]=k++;
        vector<vector<Edge>> out(k);
        for (int i=0;i<(int)adj.size();++i) if(id[i]>=0)
            for (auto e:adj[i]) { e.to=id[e.to]; out[id[i]].push_back(e); }
        adj.swap(out);
    }
};

struct RawNode {
    vector<int> ch;
    string name;
    double len=0, sup=-1;
};
struct Parser {
    string s;
    size_t p=0;
    vector<RawNode> nodes;
    explicit Parser(string x) : s(move(x)) {}
    void space() {
        while (p<s.size()) {
            if (isspace((unsigned char)s[p])) { ++p; continue; }
            if (s[p]=='[') {
                int d=1; ++p;
                while(p<s.size() && d) { if(s[p]=='[')++d; if(s[p]==']')--d; ++p; }
                continue;
            }
            break;
        }
    }
    string token() {
        space();
        string t;
        if(p<s.size() && s[p]=='\'') {
            ++p;
            while(p<s.size()) {
                if(s[p]=='\'') { ++p; if(p<s.size() && s[p]=='\'') {t+='\'';++p;} else break; }
                else t+=s[p++];
            }
        } else {
            while(p<s.size() && !isspace((unsigned char)s[p]) && string("(),:;[]").find(s[p])==string::npos)
                t+=s[p++];
        }
        return t;
    }
    int sub() {
        space();
        int v=nodes.size(); nodes.emplace_back();
        if(p<s.size() && s[p]=='(') {
            ++p;
            for(;;) {
                int c=sub(); nodes[v].ch.push_back(c);
                space();
                if(p<s.size() && s[p]==',') {++p;continue;}
                if(p>=s.size() || s[p]!=')') throw runtime_error("Invalid Newick");
                ++p; break;
            }
            string x=token();
            if(!x.empty()) {
                try { nodes[v].sup=stod(x); } catch(...) {}
            }
        } else {
            nodes[v].name=token();
            if(nodes[v].name.empty()) throw runtime_error("Missing leaf label");
        }
        space();
        if(p<s.size() && s[p]==':') {++p; string x=token(); nodes[v].len=stod(x);}
        return v;
    }
    Tree parse(vector<string>& labels, unordered_map<string,int>& ids) {
        int root=sub();
        if(labels.empty()) {
            for(auto& v:nodes) if(v.ch.empty()) labels.push_back(v.name);
            sort(labels.begin(), labels.end(), [](const string& a,const string& b) {
                if(a.size()!=b.size()) return a.size()<b.size();
                return a<b;
            });
            for(int i=0;i<(int)labels.size();++i) ids[labels[i]]=i;
        }
        Tree t(labels.size());
        vector<int> index(nodes.size());
        vector<bool> seen(labels.size(),false);
        for(int i=0;i<(int)nodes.size();++i) {
            if(nodes[i].ch.empty()) {
                auto it=ids.find(nodes[i].name);
                if(it==ids.end() || seen[it->second]) throw runtime_error("Inconsistent leaf labels");
                index[i]=it->second; seen[it->second]=true;
            } else index[i]=t.node();
        }
        for(int i=0;i<(int)nodes.size();++i) for(int c:nodes[i].ch)
            t.add(index[i],index[c],nodes[c].len,nodes[c].sup);
        int r=index[root];
        if(t.adj[r].size()==2) {
            auto a=t.adj[r][0], b=t.adj[r][1];
            t.remove(r,a.to); t.remove(r,b.to);
            t.add(a.to,b.to,a.len+b.len,max(a.sup,b.sup));
        }
        for(bool x:seen) if(!x) throw runtime_error("Missing taxon");
        t.compact();
        return t;
    }
};

string quoteLabel(const string& s) {
    if(s.find_first_of(" \t\n(),:;[]'")==string::npos) return s;
    string out="'";
    for(char c:s) {out+=c;if(c=='\'')out+='\'';}
    return out+"'";
}
string newickSub(const Tree& t,int v,int par,const vector<string>& labels) {
    if(v<t.n) return quoteLabel(labels[v]);
    string x="(";
    for(auto e:t.adj[v]) if(e.to!=par) {
        if(x.size()>1)x+=",";
        x+=newickSub(t,e.to,v,labels);
    }
    return x+")";
}
void writeTree(const Tree& t,const vector<string>& labels,const string& path) {
    int root=t.n;
    while(root<(int)t.adj.size() && t.adj[root].empty())++root;
    auto parent=filesystem::path(path).parent_path();
    if(!parent.empty())filesystem::create_directories(parent);
    ofstream f(path);
    if(!f)throw runtime_error("Cannot open output");
    f<<newickSub(t,root,-1,labels)<<";\n";
}

using Matrix=vector<vector<double>>;
double retention(double s,double len,int mode,double scale) {
    if(mode==0) return 1;
    if(mode==1) return s<0 ? (len<1e-5?0:1) : max(0.,min(1.,(s-1./3)*1.5));
    if(mode==2) return s<0 ? (len<1e-5?0:1) : max(0.,min(1.,s));
    if(mode==3) return len>1e-5?1.:0.;
    if(mode==4) return s<0 ? (len<1e-5?0:1) : max(0.,min(1.,(s-.5)*2));
    if(mode==5) return 1-exp(-len/max(1e-30,scale));
    if(mode==6) return s<0 ? (len<1e-5?0:1) : pow(max(0.,min(1.,(s-1./3)*1.5)),2);
    if(mode==7) return s<0 ? (len<1e-5?0:1) : sqrt(max(0.,min(1.,(s-1./3)*1.5)));
    if(mode==8) return s<0 ? (len<1e-5?0:1) : pow(max(0.,min(1.,(s-1./3)*1.5)),scale);
    if(mode==9) return s<0 ? (len<1e-5?0:1) : (s>scale?1.:0.);
    return 1;
}

Matrix distances(const vector<Tree>& genes,int mode,double param=1.) {
    int n=genes[0].n;
    Matrix D(n,vector<double>(n));
    for(const auto& t:genes) {
        double norm=1;
        if(mode==10||mode==11||mode==12) {
            norm=0;
            for(int v=0;v<(int)t.adj.size();++v)for(auto e:t.adj[v])if(e.to>v)norm+=e.len;
            norm=pow(norm,param);
            norm=max(norm,1e-30);
        }
        for(int a=0;a<n;++a) {
            function<void(int,int,double)> dfs=[&](int v,int p,double d) {
                if(v<n) D[a][v]+=d;
                for(auto e:t.adj[v])if(e.to!=p) {
                    double w;
                    if(mode>=10) {
                        w=e.len/norm;
                        if(mode==11)w=sqrt(e.len)/norm;
                        if(mode==12)w=log1p(e.len*param)/norm;
                    } else w=retention(e.sup,e.len,mode,param);
                    dfs(e.to,v,d+w);
                }
            };
            dfs(a,-1,0);
        }
    }
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)D[i][j]/=genes.size();
    return D;
}

Matrix geneCorrelations(const vector<Tree>& genes,double decay) {
    int n=genes[0].n;
    Matrix covariance(n,vector<double>(n));
    for(const auto& t:genes)for(int a=0;a<n;++a) {
        function<void(int,int,double)> dfs=[&](int v,int parent,double distance) {
            if(v<n)covariance[a][v]+=exp(-decay*distance)/genes.size();
            for(auto e:t.adj[v])if(e.to!=parent) {
                double d=(v<n||e.to<n)?0:retention(e.sup,e.len,1,1);
                dfs(e.to,v,distance+d);
            }
        };
        dfs(a,-1,0);
    }
    return covariance;
}

Tree nj(const Matrix& input,bool upgma=false,double alpha=1.,bool bionj=false,const Matrix* variance=nullptr) {
    int n=input.size(), cap=2*n;
    Tree t(n);
    Matrix D(cap,vector<double>(cap));
    Matrix V;
    if(bionj) {
        V.assign(cap,vector<double>(cap));
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)V[i][j]=variance?(*variance)[i][j]:input[i][j];
    }
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)D[i][j]=input[i][j];
    vector<int> active(n),sz(cap,1);
    iota(active.begin(),active.end(),0);
    while(active.size()>3) {
        int m=active.size();
        vector<double> sum(cap,0);
        if(!upgma)for(int i:active)for(int j:active)sum[i]+=D[i][j];
        double best=1e300;
        int a=-1,b=-1;
        for(int i=0;i<m;++i)for(int j=i+1;j<m;++j) {
            int x=active[i],y=active[j];
            double z=upgma ? D[x][y] : D[x][y]*(m-2)-alpha*(sum[x]+sum[y]);
            if(z<best){best=z;a=x;b=y;}
        }
        int v=t.node();
        t.add(v,a);t.add(v,b);
        double mix=.5,la=.5*D[a][b]+(sum[a]-sum[b])/(2*(m-2)),lb=D[a][b]-la;
        if(bionj) {
            double numerator=0;
            for(int k:active)if(k!=a&&k!=b)numerator+=V[b][k]-V[a][k];
            mix=max(0.,min(1.,.5+numerator/(2*(m-2)*max(1e-12,V[a][b]))));
        }
        for(int k:active) if(k!=a && k!=b) {
            D[v][k]=D[k][v]=upgma?(D[a][k]*sz[a]+D[b][k]*sz[b])/(sz[a]+sz[b])
                          :bionj?mix*(D[a][k]-la)+(1-mix)*(D[b][k]-lb):(D[a][k]+D[b][k]-D[a][b])*.5;
            if(bionj)V[v][k]=V[k][v]=max(0.,mix*V[a][k]+(1-mix)*V[b][k]-mix*(1-mix)*V[a][b]);
        }
        sz[v]=sz[a]+sz[b];
        active.erase(remove(active.begin(),active.end(),a),active.end());
        active.erase(remove(active.begin(),active.end(),b),active.end());
        active.push_back(v);
    }
    int v=t.node();
    for(int a:active)t.add(v,a);
    return t;
}

using LeafWeights=vector<pair<int,double>>;
void weightedLeaves(const Tree& t,int v,int parent,double weight,LeafWeights& leaves) {
    if(v<t.n) {leaves.push_back({v,weight});return;}
    int k=(int)t.adj[v].size()-1;
    for(auto e:t.adj[v])if(e.to!=parent)weightedLeaves(t,e.to,v,weight/k,leaves);
}
double meanDistance(const Matrix& d,const LeafWeights& a,const LeafWeights& b) {
    double value=0;
    for(auto [i,wi]:a) {
        double row=0;
        for(auto [j,wj]:b)row+=d[i][j]*wj;
        value+=row*wi;
    }
    return value;
}
int bmeNNI(Tree& t,const Matrix& d,int passes=100) {
    int total=0;
    for(int pass=0;pass<passes;++pass) {
        int changes=0;
        vector<pair<int,int>> edges;
        for(int u=t.n;u<(int)t.adj.size();++u)for(auto e:t.adj[u])if(e.to>u&&e.to>=t.n)edges.push_back({u,e.to});
        for(auto [u,v]:edges) {
            vector<int> a,b;
            for(auto e:t.adj[u])if(e.to!=v)a.push_back(e.to);
            for(auto e:t.adj[v])if(e.to!=u)b.push_back(e.to);
            if(a.size()!=2||b.size()!=2)continue;
            LeafWeights w[4];
            weightedLeaves(t,a[0],u,1,w[0]);weightedLeaves(t,a[1],u,1,w[1]);
            weightedLeaves(t,b[0],v,1,w[2]);weightedLeaves(t,b[1],v,1,w[3]);
            double q[3]={meanDistance(d,w[0],w[1])+meanDistance(d,w[2],w[3]),
                         meanDistance(d,w[0],w[2])+meanDistance(d,w[1],w[3]),
                         meanDistance(d,w[0],w[3])+meanDistance(d,w[1],w[2])};
            int best=min_element(q,q+3)-q;
            if(best && q[best]<q[0]-1e-10) {
                if(best==1)t.swapBranches(u,a[1],v,b[0]);
                else t.swapBranches(u,a[1],v,b[1]);
                ++changes;
            }
        }
        total+=changes;
        if(!changes)break;
    }
    return total;
}

struct GNode {
    int leaf=-1,l=-1,r=-1;
    double ql=0,qr=0,hl=1,hr=1;
};
struct GTree {
    vector<GNode> ns;
    double weight=1;
    bool pendant=false;
    int index=0;
    int moment=0;
};

vector<GTree> makeGtrees(const vector<Tree>& trees,int mode,double param,double lambda=0,double genePower=0,bool pendant=false,bool relative=false,
                         double absoluteLambda=0) {
    vector<GTree> gs;
    for(const auto& t:trees) {
        GTree gt;
        gt.pendant=pendant;
        gt.index=gs.size();
        double total=0;
        for(int v=0;v<(int)t.adj.size();++v)for(auto e:t.adj[v])if(e.to>v)total+=e.len;
        double localLambda=lambda*(relative?(2*t.n-3)/max(total,1e-20):1)+absoluteLambda;
        double scale=param;
        if(mode==5)scale=param*total/(2*t.n-3);
        function<int(int,int)> dfs=[&](int v,int p) {
            if(v<t.n) {int u=gt.ns.size();gt.ns.push_back({v});return u;}
            vector<pair<int,Edge>> ch;
            for(auto e:t.adj[v]) if(e.to!=p) ch.push_back({dfs(e.to,v),e});
            if(ch.size()!=2)throw runtime_error("Nonbinary gene tree");
            GNode gn;
            gn.l=ch[0].first;gn.r=ch[1].first;
            gn.ql=1-retention(ch[0].second.sup,ch[0].second.len,mode,scale);
            gn.qr=1-retention(ch[1].second.sup,ch[1].second.len,mode,scale);
            gn.hl=exp(-localLambda*ch[0].second.len);
            gn.hr=exp(-localLambda*ch[1].second.len);
            int u=gt.ns.size();gt.ns.push_back(gn);return u;
        };
        int a=0,b=t.adj[0][0].to;
        int left=dfs(a,b),right=dfs(b,a);
        auto e=t.adj[a][0];
        GNode gn;
        gn.l=left;gn.r=right;
        gn.ql=0;
        gn.qr=1-retention(e.sup,e.len,mode,scale);
        gn.hl=1;gn.hr=exp(-localLambda*e.len);
        gt.ns.push_back(gn);
        gt.weight=pow(max(total/(2*t.n-3),1e-30),genePower);
        gs.push_back(move(gt));
    }
    double norm=0;for(auto& g:gs)norm+=g.weight;
    for(auto& g:gs)g.weight*=gs.size()/norm;
    return gs;
}

vector<GTree> makeMaxGtrees(const vector<Tree>& trees,int levels,double lambda=0,double genePower=0,bool pendant=false,bool relative=false,
                            double absoluteLambda=0) {
    vector<GTree> out;
    for(int k=0;k<levels;++k) {
        double threshold=1./3+2./3*(k+.5)/levels;
        auto gs=makeGtrees(trees,9,threshold,lambda,genePower,pendant,relative,absoluteLambda);
        for(auto& gt:gs) {
            gt.weight/=levels;gt.index=out.size();
            out.push_back(move(gt));
        }
    }
    return out;
}

vector<GTree> makeMomentGtrees(const vector<Tree>& trees,int degree,int mode,double param) {
    auto original=makeGtrees(trees,mode,param);
    vector<GTree> out;
    for(int k=1;k<=degree;++k)for(auto gt:original) {
        for(auto& nd:gt.ns) {nd.ql=pow(nd.ql,k);nd.qr=pow(nd.qr,k);}
        gt.moment=k-1;gt.index=out.size();
        out.push_back(move(gt));
    }
    return out;
}

// State values describe one, two, and resolved three colors below a gene node.
struct QState {
    double s[4],p[6],z[6],t[12];
};
const int pa[6]={0,0,0,1,1,2};
const int pb[6]={1,2,3,2,3,3};
const int pi4[4][4]={{-1,0,1,2},{0,-1,3,4},{1,3,-1,5},{2,4,5,-1}};
const int pmatch[6]={0,1,2,2,1,0};
struct TripleType {int a, pair,missing,top;};
vector<TripleType> triples;
void initTriples() {
    if(!triples.empty())return;
    for(int p=0;p<6;++p)for(int a=0;a<4;++a) if(a!=pa[p]&&a!=pb[p]) {
        int missing=6-a-pa[p]-pb[p];
        triples.push_back({a,p,missing,pmatch[p]});
    }
}

array<double,3> quartetScores(const vector<GTree>& gs,const vector<int>& color,bool masks=false,array<double,2>* variance=nullptr,
                             vector<array<double,3>>* perGene=nullptr,const vector<double>* leafWeights=nullptr) {
    initTriples();
    array<double,3> result{};
    array<double,2> squares{};
    if(perGene)perGene->clear();
    vector<QState> st(2*color.size()+4);
    for(const auto& gt:gs) {
        array<double,3> q{};
        for(int v=0;v<(int)gt.ns.size();++v) {
            const auto& nd=gt.ns[v];
            QState& o=st[v];
            if(nd.leaf>=0) {
                o={};
                double weight=leafWeights?(*leafWeights)[nd.leaf]:1;
                if(masks) {
                    for(int k=0;k<4;++k)if(color[nd.leaf]&(1<<k))o.s[k]=weight;
                } else if(color[nd.leaf]>=0) o.s[color[nd.leaf]]=weight;
                continue;
            }
            const QState& x=st[nd.l];const QState& y=st[nd.r];
            double xs[4],ys[4],xp[6],yp[6],xz[6],yz[6];
            for(int a=0;a<4;++a) {
                xs[a]=x.s[a]*nd.hl;ys[a]=y.s[a]*nd.hr;
                o.s[a]=xs[a]+ys[a];
            }
            double phl=gt.pendant?1.:nd.hl,phr=gt.pendant?1.:nd.hr;
            for(int p=0;p<6;++p) {
                xp[p]=x.p[p]*phl;yp[p]=y.p[p]*phr;
                xz[p]=x.z[p]*phl*nd.ql;yz[p]=y.z[p]*phr*nd.qr;
                double cross=xs[pa[p]]*ys[pb[p]]+xs[pb[p]]*ys[pa[p]];
                o.p[p]=xp[p]+yp[p]+cross;
                o.z[p]=xz[p]+yz[p]+cross;
            }
            for(int i=0;i<12;++i) {
                auto tr=triples[i];
                double xt=x.t[i]*nd.hl,yt=y.t[i]*nd.hr;
                q[tr.top]+=xt*ys[tr.missing]+yt*xs[tr.missing];
                o.t[i]=xt+yt+(xp[tr.pair]-xz[tr.pair])*ys[tr.a]+(yp[tr.pair]-yz[tr.pair])*xs[tr.a];
            }
            for(int p=0;p<6;++p) {
                int other=5-p;
                q[pmatch[p]]+=xp[p]*yp[other]-xz[p]*yz[other];
            }
        }
        if(perGene)perGene->push_back(q);
        for(int k=0;k<3;++k)result[k]+=q[k]*gt.weight;
        if(variance)for(int k=0;k<2;++k) {
            double delta=(q[k+1]-q[0])*gt.weight;
            squares[k]+=delta*delta;
        }
    }
    if(variance)for(int k=0;k<2;++k) {
        double delta=result[k+1]-result[0],m=gs.size();
        (*variance)[k]=m>1?max(0.,(squares[k]-delta*delta/m)*m/(m-1)):0;
    }
    return result;
}

void normalizeLengthWeights(vector<GTree>& gs,int n) {
    vector<int> mask(n,15);
    for(auto& gt:gs) {
        GTree plain=gt;plain.weight=1;
        auto q=quartetScores({plain},mask,true);
        double penalized=q[0]+q[1]+q[2];
        for(auto& nd:plain.ns)nd.hl=nd.hr=1;
        q=quartetScores({plain},mask,true);
        double original=q[0]+q[1]+q[2];
        if(penalized>1e-100)gt.weight*=min(maximumNormalization,original/penalized);
    }
}

vector<GTree> makeInformationGtrees(const vector<Tree>& trees,int mode,double param,double lambda,double genePower,
                                    bool pendant,bool relative,double scale,double mix,bool normalize,double absoluteLambda=0) {
    auto support=makeGtrees(trees,mode,param,lambda,genePower,pendant,relative,absoluteLambda);
    auto length=makeGtrees(trees,5,scale,lambda,genePower,pendant,relative,absoluteLambda);
    auto joint=support;
    vector<int> mask(trees[0].n,15);
    for(int g=0;g<(int)trees.size();++g) {
        for(int v=0;v<(int)joint[g].ns.size();++v) {
            joint[g].ns[v].ql*=length[g].ns[v].ql;
            joint[g].ns[v].qr*=length[g].ns[v].qr;
        }
        double factor=1;
        if(normalize) {
            GTree a=support[g],b=length[g],c=joint[g];
            a.weight=1;b.weight=mix;c.weight=-mix;
            auto q=quartetScores({a,b,c},mask,true);
            double penalized=q[0]+q[1]+q[2];
            for(auto& nd:a.ns)nd.hl=nd.hr=1;
            q=quartetScores({a},mask,true);
            double original=q[0]+q[1]+q[2];
            if(penalized>1e-100)factor=min(maximumNormalization,original/penalized);
        }
        support[g].weight*=factor;
        length[g].weight*=factor*mix;
        joint[g].weight*=-factor*mix;
    }
    vector<GTree> out;
    for(auto* model:{&support,&length,&joint})for(auto& gt:*model) {
        gt.index=out.size();out.push_back(move(gt));
    }
    return out;
}

vector<GTree> makeAdaptiveGtrees(const vector<Tree>& trees,double amount,double powerStrength,double absoluteLambda) {
    auto plain=makeGtrees(trees,1,1);
    int n=trees[0].n;
    double denominator=double(n)*(n-1)*(n-2)*(n-3),meanQuality=0,meanPenalty=0;
    vector<int> mask(n,15);
    vector<GTree> out;
    for(int g=0;g<(int)trees.size();++g) {
        auto q=quartetScores({plain[g]},mask,true);
        double quality=max(0.,min(1.,(q[0]+q[1]+q[2])/denominator));
        double penalty=max(.025,min(.5,amount*quality/max(.05,1-quality)));
        double power=1+powerStrength*(1-quality);
        auto gs=makeGtrees({trees[g]},8,power,penalty,0,true,true,absoluteLambda);
        normalizeLengthWeights(gs,n);
        gs[0].index=g;out.push_back(move(gs[0]));
        meanQuality+=quality;meanPenalty+=penalty;
    }
    cerr<<"Gene quality "<<meanQuality/trees.size()<<" relative penalty "<<meanPenalty/trees.size()<<" time "<<elapsed()<<"\n";
    return out;
}

void colorSub(const Tree& t,int v,int p,int c,vector<int>& colors) {
    if(v<t.n) {colors[v]=c;return;}
    for(auto e:t.adj[v])if(e.to!=p)colorSub(t,e.to,v,c,colors);
}
struct NniResult {int moves=0,sweeps=0;double gain=0;};
struct QuartetCache;
array<double,3> cachedQuartets(QuartetCache& cache,const vector<GTree>& gs,const vector<int>& colors,array<double,2>* variance=nullptr,
                              double* effective=nullptr,double geneLikelihood=-1);
NniResult nniOptimize(Tree& t,const vector<GTree>& gs,int maxSweeps=20,double deadline=1e9,bool shuffle=false,double gate=0,double rooted=0,
                      QuartetCache* cache=nullptr,double geneLikelihood=-1,double balance=0) {
    NniResult result;
    vector<int> colors(t.n,-1);
    mt19937 rng(1);
    for(int it=0;it<maxSweeps && elapsed()<deadline;++it) {
        vector<pair<int,int>> edges;
        for(int u=t.n;u<(int)t.adj.size();++u)for(auto e:t.adj[u])
            if(e.to>u && e.to>=t.n)edges.push_back({u,e.to});
        if(shuffle)std::shuffle(edges.begin(),edges.end(),rng);
        int moves=0;
        for(auto [u,v]:edges) {
            if(elapsed()>deadline)break;
            vector<int>a,b;
            for(auto e:t.adj[u])if(e.to!=v)a.push_back(e.to);
            for(auto e:t.adj[v])if(e.to!=u)b.push_back(e.to);
            if(a.size()!=2||b.size()!=2)continue;
            colorSub(t,a[0],u,0,colors);colorSub(t,a[1],u,1,colors);
            colorSub(t,b[0],v,2,colors);colorSub(t,b[1],v,3,colors);
            array<double,2> variance{};
            array<double,3> q{};
            if(rooted<1) {
                if(balance>0) {
                    vector<double> weights(t.n);
                    LeafWeights lists[4];
                    weightedLeaves(t,a[0],u,1,lists[0]);weightedLeaves(t,a[1],u,1,lists[1]);
                    weightedLeaves(t,b[0],v,1,lists[2]);weightedLeaves(t,b[1],v,1,lists[3]);
                    for(int k=0;k<4;++k) {
                        double norm=0;
                        for(auto& p:lists[k]) {p.second=pow(p.second,balance);norm+=p.second;}
                        for(auto p:lists[k])weights[p.first]=p.second/norm;
                    }
                    q=quartetScores(gs,colors,false,gate>0?&variance:nullptr,nullptr,&weights);
                } else if(cache)q=cachedQuartets(*cache,gs,colors,gate>0?&variance:nullptr,nullptr,geneLikelihood);
                else q=quartetScores(gs,colors,false,gate>0?&variance:nullptr);
            }
            if(rooted>0) {
                int anchorColor=colors[0],count=0;
                for(int i=0;i<t.n;++i)if(colors[i]==anchorColor) {
                    ++count;if(i!=0)colors[i]=-1;
                }
                auto rq=quartetScores(gs,colors);
                for(int k=0;k<3;++k)q[k]=(1-rooted)*q[k]+rooted*rq[k]*count;
            }
            array<double,3> adjusted=q;
            if(gate>0)for(int k=1;k<3;++k)adjusted[k]-=gate*sqrt(variance[k-1]);
            int best=max_element(adjusted.begin(),adjusted.end())-adjusted.begin();
            if(best!=0 && adjusted[best]>q[0]+1e-7*max(1.,fabs(q[0]))) {
                if(best==1)t.swapBranches(u,a[1],v,b[0]);
                else t.swapBranches(u,a[1],v,b[1]);
                ++moves;result.gain+=q[best]-q[0];
            }
        }
        result.moves+=moves;++result.sweeps;
        if(!moves)break;
    }
    return result;
}

double totalScoreSlow(const Tree& t,const vector<GTree>& gs) {
    double score=0;
    vector<int> col(t.n,-1),mask(t.n);
    for(int v=t.n;v<(int)t.adj.size();++v)if(t.adj[v].size()==3) {
        for(int k=0;k<3;++k)colorSub(t,t.adj[v][k].to,v,k,col);
        for(int k=0;k<3;++k) {
            for(int a=0;a<t.n;++a) {
                if(col[a]==k)mask[a]=3;
                else if(col[a]==(k+1)%3)mask[a]=4;
                else mask[a]=8;
            }
            score+=quartetScores(gs,mask,true)[0];
        }
    }
    return score*.25;
}

struct Q3State {
    double s[3],p[6],z[6],t[9];
};
double tripartitionScore(const vector<GTree>& gs,const vector<int>& color) {
    static const int aa[6]={0,0,0,1,1,2},bb[6]={0,1,2,1,2,2};
    static const int repeat[3]={0,3,5},other[3]={4,2,1};
    static const int ta[9]={1,2,0,2,0,1,0,1,2};
    static const int tp[9]={0,0,3,3,5,5,4,2,1};
    static const int tm[9]={2,1,2,0,1,0,0,1,2};
    vector<Q3State> st(2*color.size()+4);
    double result=0;
    for(const auto& gt:gs) {
        double q=0;
        for(int v=0;v<(int)gt.ns.size();++v) {
            const auto& nd=gt.ns[v];
            auto& o=st[v];
            if(nd.leaf>=0) {
                o={};
                o.s[color[nd.leaf]]=1;
                continue;
            }
            const auto& x=st[nd.l];const auto& y=st[nd.r];
            double xs[3],ys[3],xp[6],yp[6],xz[6],yz[6];
            for(int a=0;a<3;++a) {
                xs[a]=x.s[a]*nd.hl;ys[a]=y.s[a]*nd.hr;
                o.s[a]=xs[a]+ys[a];
            }
            double phl=gt.pendant?1.:nd.hl,phr=gt.pendant?1.:nd.hr;
            for(int p=0;p<6;++p) {
                xp[p]=x.p[p]*phl;yp[p]=y.p[p]*phr;
                xz[p]=x.z[p]*phl*nd.ql;yz[p]=y.z[p]*phr*nd.qr;
                double cross=xs[aa[p]]*ys[bb[p]];
                if(aa[p]!=bb[p])cross+=xs[bb[p]]*ys[aa[p]];
                o.p[p]=xp[p]+yp[p]+cross;
                o.z[p]=xz[p]+yz[p]+cross;
            }
            for(int i=0;i<9;++i) {
                double xt=x.t[i]*nd.hl,yt=y.t[i]*nd.hr;
                q+=xt*ys[tm[i]]+yt*xs[tm[i]];
                o.t[i]=xt+yt+(xp[tp[i]]-xz[tp[i]])*ys[ta[i]]+(yp[tp[i]]-yz[tp[i]])*xs[ta[i]];
            }
            for(int a=0;a<3;++a) {
                int p=repeat[a],r=other[a];
                q+=xp[p]*yp[r]+xp[r]*yp[p]-xz[p]*yz[r]-xz[r]*yz[p];
            }
        }
        result+=q*gt.weight;
    }
    return result;
}

double totalScore(const Tree& t,const vector<GTree>& gs) {
    double score=0;
    vector<int> col(t.n,-1);
    for(int v=t.n;v<(int)t.adj.size();++v)if(t.adj[v].size()==3) {
        for(int k=0;k<3;++k)colorSub(t,t.adj[v][k].to,v,k,col);
        score+=tripartitionScore(gs,col);
    }
    return score*.5;
}

struct Bits {
    array<uint64_t,4> w{};
    bool operator==(const Bits& o) const {return w==o.w;}
    bool operator<(const Bits& o) const {return w<o.w;}
    void add(int i) {w[i>>6]|=uint64_t(1)<<(i&63);}
    bool has(int i) const {return (w[i>>6]>>(i&63))&1;}
    int size() const {int k=0;for(auto x:w)k+=__builtin_popcountll(x);return k;}
    int first() const {for(int k=0;k<4;++k)if(w[k])return k*64+__builtin_ctzll(w[k]);return -1;}
    bool subset(const Bits& o) const {
        for(int i=0;i<4;++i)if(w[i]&~o.w[i])return false;
        return true;
    }
    Bits operator^(const Bits& o)const {Bits b;for(int i=0;i<4;++i)b.w[i]=w[i]^o.w[i];return b;}
    Bits operator|(const Bits& o)const {Bits b;for(int i=0;i<4;++i)b.w[i]=w[i]|o.w[i];return b;}
};
struct BitsHash {
    size_t operator()(const Bits& x)const {
        uint64_t h=0x9e3779b97f4a7c15ULL;
        for(auto a:x.w) {a^=a>>30;a*=0xbf58476d1ce4e5b9ULL;a^=a>>27;a*=0x94d049bb133111ebULL;a^=a>>31;h^=a+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);}
        return h;
    }
};
struct QuartetKey {
    array<Bits,4> group;
    bool operator==(const QuartetKey& k)const {return group==k.group;}
};
struct QuartetKeyHash {
    size_t operator()(const QuartetKey& k)const {
        size_t h=0;
        for(auto& b:k.group)h^=BitsHash{}(b)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        return h;
    }
};
struct QuartetCache {
    const vector<GTree>& original;
    unordered_map<QuartetKey,vector<array<double,3>>,QuartetKeyHash> values;
    int hits=0,misses=0;
    const Matrix* covariance=nullptr;
    double densityPower=.5;
    bool gls=false;
    double ridge=.05;
    unordered_map<Bits,vector<pair<int,double>>,BitsHash> leafWeightCache;
    explicit QuartetCache(const vector<GTree>& gs):original(gs) {}
};

struct MomentKernel {
    int degree=0;
    double cap=0;
    vector<vector<double>> cp,cm;
    MomentKernel()=default;
    MomentKernel(int d,double c):degree(d),cap(c),cp(256,vector<double>(d)),cm(256,vector<double>(d)) {
        vector<vector<double>> binomial(d+1,vector<double>(d+1));
        for(int r=0;r<=d;++r) {
            binomial[r][0]=binomial[r][r]=1;
            for(int k=1;k<r;++k)binomial[r][k]=binomial[r-1][k-1]+binomial[r-1][k];
        }
        for(int g=0;g<256;++g) {
            double z=cap*g/255,a=2*z/(1+2*z);
            vector<double> power(d+1,1);
            for(int r=1;r<=d;++r)power[r]=power[r-1]*z;
            for(int k=1;k<=d;++k) {
                cp[g][k-1]=pow(a,k)/k;
                double coeff=0;
                for(int r=k;r<=d;++r)coeff+=binomial[r][k]*power[r]/r;
                cm[g][k-1]=(k%2?-1:1)*coeff;
            }
        }
    }
    array<double,3> score(const vector<array<double,3>>& s,double blend)const {
        array<double,3> best{};
        for(int g=1;g<256;++g) {
            array<double,3> value{};
            for(int k=0;k<degree;++k) {
                double sum=s[k][0]+s[k][1]+s[k][2],base=cm[g][k]*sum,diff=cp[g][k]-cm[g][k];
                for(int t=0;t<3;++t)value[t]+=base+diff*s[k][t];
            }
            for(int t=0;t<3;++t)best[t]=max(best[t],value[t]);
        }
        if(blend<1) {
            double mean=(s[0][0]+s[0][1]+s[0][2])/3;
            for(int t=0;t<3;++t)best[t]=blend*best[t]+(1-blend)*(s[0][t]-mean);
        }
        return best;
    }
};

array<double,3> cachedQuartets(QuartetCache& cache,const vector<GTree>& gs,const vector<int>& colors,array<double,2>* variance,double* effective,
                              double geneLikelihood) {
    QuartetKey key;
    for(int a=0;a<(int)colors.size();++a)key.group[colors[a]].add(a);
    int order[4]={0,1,2,3},rank[4];
    sort(order,order+4,[&](int a,int b){return key.group[a]<key.group[b];});
    QuartetKey canon;
    for(int k=0;k<4;++k) {canon.group[k]=key.group[order[k]];rank[order[k]]=k;}
    auto it=cache.values.find(canon);
    if(it==cache.values.end()) {
        vector<int> canonicalColors(colors.size());
        for(int a=0;a<(int)colors.size();++a)canonicalColors[a]=rank[colors[a]];
        vector<array<double,3>> counts;
        vector<double> leafWeights;
        if(cache.covariance) {
            leafWeights.resize(colors.size());
            for(int k=0;k<4;++k) {
                auto wi=cache.leafWeightCache.find(canon.group[k]);
                if(wi==cache.leafWeightCache.end()) {
                    vector<int> taxa;
                    for(int i=0;i<(int)colors.size();++i)if(canonicalColors[i]==k)taxa.push_back(i);
                    int count=taxa.size();
                    vector<double> weights(count,1./count);
                    if(cache.gls && count>1) {
                        vector<double> gradient(count);
                        for(int i=0;i<count;++i) {
                            for(int j=0;j<count;++j)gradient[i]+=(*cache.covariance)[taxa[i]][taxa[j]]/count;
                            gradient[i]+=cache.ridge/count;
                        }
                        for(int iter=0;iter<500;++iter) {
                            double f=0;
                            for(int i=0;i<count;++i)f+=weights[i]*gradient[i];
                            int j=min_element(gradient.begin(),gradient.end())-gradient.begin();
                            double gap=f-gradient[j];
                            if(gap<1e-7)break;
                            double den=(*cache.covariance)[taxa[j]][taxa[j]]+cache.ridge-2*gradient[j]+f;
                            double step=max(0.,min(1.,gap/max(1e-20,den)));
                            for(int i=0;i<count;++i) {
                                weights[i]*=1-step;
                                gradient[i]=(1-step)*gradient[i]+step*((*cache.covariance)[taxa[i]][taxa[j]]+(i==j?cache.ridge:0));
                            }
                            weights[j]+=step;
                        }
                        for(double& x:weights)x=pow(max(1e-20,x),cache.densityPower);
                    } else if(!cache.gls) {
                        for(int i=0;i<count;++i) {
                            double density=0;
                            for(int j:taxa)density+=(*cache.covariance)[taxa[i]][j];
                            weights[i]=pow(max(1e-12,density),-cache.densityPower);
                        }
                    }
                    double norm=accumulate(weights.begin(),weights.end(),0.);
                    vector<pair<int,double>> values;
                    for(int i=0;i<count;++i)values.push_back({taxa[i],weights[i]/norm});
                    wi=cache.leafWeightCache.emplace(canon.group[k],move(values)).first;
                }
                for(auto [i,w]:wi->second)leafWeights[i]=w;
            }
        }
        quartetScores(cache.original,canonicalColors,false,nullptr,&counts,cache.covariance?&leafWeights:nullptr);
        it=cache.values.emplace(move(canon),move(counts)).first;
        ++cache.misses;
    } else ++cache.hits;
    int topology[3];
    for(int k=0;k<3;++k)topology[k]=pmatch[pi4[rank[0]][rank[k+1]]];
    if(likelihoodMoments>0) {
        vector<array<double,3>> sums(likelihoodMoments);
        for(const auto& gt:gs) {
            const auto& q=it->second[gt.index];
            for(int k=0;k<3;++k)sums[gt.moment][k]+=q[topology[k]]*gt.weight;
        }
        static MomentKernel kernel;
        if(kernel.degree!=likelihoodMoments || kernel.cap!=momentCap)kernel=MomentKernel(likelihoodMoments,momentCap);
        return kernel.score(sums,geneLikelihood>=0?geneLikelihood:1);
    }
    array<double,3> total{};
    array<double,2> squares{};
    for(const auto& gt:gs) {
        const auto& q=it->second[gt.index];
        for(int k=0;k<3;++k)total[k]+=q[topology[k]]*gt.weight;
        if(variance)for(int k=0;k<2;++k) {
            double delta=(q[topology[k+1]]-q[topology[0]])*gt.weight;
            squares[k]+=delta*delta;
        }
    }
    if(variance)for(int k=0;k<2;++k) {
        double delta=total[k+1]-total[0],m=gs.size();
        (*variance)[k]=m>1?max(0.,(squares[k]-delta*delta/m)*m/(m-1)):0;
    }
    if(effective) {
        double all=total[0]+total[1]+total[2],p[3];
        if(all<=1e-20) {*effective=gs.size();return total;}
        double binomial=1;
        for(int k=0;k<3;++k) {p[k]=total[k]/all;binomial-=p[k]*p[k];}
        double sumsq=0;
        for(const auto& gt:gs) {
            const auto& q=it->second[gt.index];
            double qt=(q[0]+q[1]+q[2])*gt.weight;
            for(int k=0;k<3;++k) {
                double r=q[topology[k]]*gt.weight-p[k]*qt;
                sumsq+=r*r;
            }
        }
        double m=gs.size();
        double var=sumsq*m/max(1.,m-1)/(all*all);
        *effective=var>1e-20?max(2.,min(5000.,binomial/var)):m;
    }
    if(geneLikelihood>=0) {
        double sizes[4]={0,0,0,0};
        for(int c:colors)++sizes[c];
        double count=sizes[0]*sizes[1]*sizes[2]*sizes[3];
        vector<array<double,3>> evidence(gs.size());
        for(int g=0;g<(int)gs.size();++g) {
            const auto& q=it->second[gs[g].index];
            double sum=q[0]+q[1]+q[2];
            for(int k=0;k<3;++k)evidence[g][k]=max(-1.,min(2.,evidenceScale*(3*q[topology[k]]-sum)/count));
        }
        double grand=total[0]+total[1]+total[2];
        for(int k=0;k<3;++k) {
            double derivative0=0,curvature0=0,derivative1=0;
            for(int g=0;g<(int)gs.size();++g) {
                double a=evidence[g][k],w=gs[g].weight;
                derivative0+=w*a;curvature0+=w*a*a;
                derivative1+=w*a/max(1e-12,1+a*(1-1e-10));
            }
            double loglik=0;
            if(derivative0>1e-10) {
                double z=1-1e-10;
                if(derivative1<0) {
                    double lo=0,hi=1;
                    z=max(1e-7,min(.999999,derivative0/max(1e-20,curvature0)));
                    for(int iter=0;iter<18;++iter) {
                        double d=0,dd=0;
                        for(int g=0;g<(int)gs.size();++g) {
                            double a=evidence[g][k],v=a/max(1e-12,1+a*z),w=gs[g].weight;
                            d+=w*v;dd-=w*v*v;
                        }
                        if(fabs(d)<1e-10)break;
                        if(d>0)lo=z;else hi=z;
                        double next=z-d/dd;
                        if(!(next>lo&&next<hi))next=(lo+hi)*.5;
                        if(fabs(next-z)<1e-10) {z=next;break;}
                        z=next;
                    }
                }
                for(int g=0;g<(int)gs.size();++g)loglik+=gs[g].weight*log1p(evidence[g][k]*z);
            }
            total[k]=(1-geneLikelihood)*(total[k]-grand/3)/count+geneLikelihood*max(0.,loglik);
        }
    }
    return total;
}

void treeClades(const Tree& t,unordered_map<Bits,int,BitsHash>& clades) {
    function<Bits(int,int)> dfs=[&](int v,int p) {
        Bits b;
        if(v<t.n)b.add(v);
        else for(auto e:t.adj[v])if(e.to!=p)b=b|dfs(e.to,v);
        clades.emplace(b,0);
        return b;
    };
    dfs(t.adj[0][0].to,0);
}

vector<Tree> recurrenceWeights(const vector<Tree>& genes,double power) {
    unordered_map<Bits,int,BitsHash> counts;
    for(const auto& t:genes) {
        unordered_map<Bits,int,BitsHash> clades;
        treeClades(t,clades);
        for(auto& kv:clades)++counts[kv.first];
    }
    vector<Tree> out=genes;
    for(auto& t:out) {
        function<Bits(int,int)> dfs=[&](int v,int par) {
            Bits bits;
            if(v<t.n) {bits.add(v);return bits;}
            for(auto e:t.adj[v])if(e.to!=par)bits=bits|dfs(e.to,v);
            if(par>=t.n) {
                double f=(counts.at(bits)+.5)/(genes.size()+.5);
                for(auto& e:t.adj[v])if(e.to==par) {
                    double p=retention(e.sup,e.len,1,1)*pow(f,power);
                    e.sup=(1+2*min(1.,p))/3;
                    for(auto& back:t.adj[par])if(back.to==v)back.sup=e.sup;
                }
            }
            return bits;
        };
        dfs(t.adj[0][0].to,0);
    }
    return out;
}

Tree constrainedDP(const vector<Tree>& genes,const vector<Tree>& extra,const vector<GTree>& gs,const Tree& fallback,double deadline,
                   const unordered_map<Bits,double,BitsHash>* supports=nullptr) {
    int n=fallback.n;
    Tree backup=fallback;
    if(supports) {
        auto score=[&](const Tree& tree) {
            unordered_map<Bits,int,BitsHash> clades;
            treeClades(tree,clades);
            double value=0;
            for(auto& kv:clades)if(kv.first.size()>1) {
                auto it=supports->find(kv.first);
                if(it!=supports->end())value+=it->second;
            }
            return value;
        };
        double best=score(backup);
        for(const auto& tree:genes) {
            double value=score(tree);
            if(value>best) {best=value;backup=tree;}
        }
    }
    unordered_map<Bits,int,BitsHash> id;
    for(const auto& t:genes)treeClades(t,id);
    for(const auto& t:extra)treeClades(t,id);
    if(elapsed()>deadline) {
        cerr<<"Consensus medoid fallback "<<id.size()<<" clades time "<<elapsed()<<"\n";
        return backup;
    }
    if(supports&&id.size()>12000) {
        vector<pair<double,Bits>> ranked;
        ranked.reserve(id.size());
        for(auto& kv:id) {
            auto it=supports->find(kv.first);
            ranked.push_back({it==supports->end()?0:it->second,kv.first});
        }
        auto better=[](const auto& a,const auto& b) {
            return a.first==b.first?a.second<b.second:a.first>b.first;
        };
        nth_element(ranked.begin(),ranked.begin()+11000,ranked.end(),better);
        id.clear();
        for(int i=0;i<11000;++i)id.emplace(ranked[i].second,0);
        treeClades(backup,id);treeClades(fallback,id);
    }
    vector<Bits> bs;
    for(const auto& kv:id)bs.push_back(kv.first);
    sort(bs.begin(),bs.end(),[](const Bits& a,const Bits& b) {
        int sa=a.size(),sb=b.size();
        return sa==sb?a<b:sa<sb;
    });
    int m=bs.size();
    vector<int> sizes(m);
    vector<vector<int>> containing(n);
    for(int i=0;i<m;++i) {
        if(i%1024==0&&elapsed()>deadline)return backup;
        id[bs[i]]=i;sizes[i]=bs[i].size();
        for(int a=1;a<n;++a)if(bs[i].has(a))containing[a].push_back(i);
    }
    vector<vector<pair<int,int>>> choices(m);
    int total=0;
    uint64_t inspected=0;
    for(int x=0;x<m;++x)if(sizes[x]>1) {
        if(elapsed()>deadline)return backup;
        for(int a:containing[bs[x].first()]) {
            if((++inspected%1024)==0&&elapsed()>deadline)return backup;
            if(sizes[a]>=sizes[x])break;
            if(!bs[a].subset(bs[x]))continue;
            auto it=id.find(bs[x]^bs[a]);
            if(it!=id.end())choices[x].push_back({a,it->second});
        }
        total+=choices[x].size();
    }
    cerr<<"DP clades "<<m<<" partitions "<<total<<" time "<<elapsed()<<"\n";
    vector<double> dp(m,-1e300);
    vector<pair<int,int>> best(m,{-1,-1});
    vector<int> cols(n,2);
    int calls=0;
    for(int x=0;x<m;++x) {
        if(sizes[x]==1) {dp[x]=0;continue;}
        double nodeWeight=0;
        if(supports) {
            auto it=supports->find(bs[x]);
            if(it!=supports->end())nodeWeight=it->second;
        }
        for(auto [a,b]:choices[x]) {
            if(elapsed()>deadline) {
                cerr<<"DP deadline after "<<calls<<" partitions\n";
                return backup;
            }
            if(dp[a]<-1e290||dp[b]<-1e290)continue;
            double value=dp[a]+dp[b]+nodeWeight;
            if(!supports) {
                for(int i=0;i<n;++i)cols[i]=bs[a].has(i)?0:bs[b].has(i)?1:2;
                value+=tripartitionScore(gs,cols);
            }
            ++calls;
            if(value>dp[x]) {dp[x]=value;best[x]={a,b};}
        }
    }
    Bits full;for(int i=1;i<n;++i)full.add(i);
    int root=id.at(full);
    if(dp[root]<-1e290)return backup;
    Tree out(n);
    function<int(int)> construct=[&](int x) {
        if(sizes[x]==1)return bs[x].first();
        int u=out.node();
        auto [a,b]=best[x];
        if(a<0||b<0)throw runtime_error("Incomplete DP resolution");
        out.add(u,construct(a));out.add(u,construct(b));
        return u;
    };
    int r=construct(root);out.add(0,r);
    cerr<<setprecision(14)<<"DP score "<<dp[root]*(supports?1:.5)<<" calls "<<calls<<" time "<<elapsed()<<"\n";
    return out;
}

Tree geneSplitConsensus(const vector<Tree>& genes,const Tree& primary,double fraction,double deadline) {
    unordered_map<Bits,double,BitsHash> counts;
    unordered_map<Bits,int,BitsHash> current;
    treeClades(primary,current);
    for(auto& kv:current)counts[kv.first]+=fraction*genes.size()+.001;
    for(const auto& tree:genes) {
        function<Bits(int,int)> dfs=[&](int v,int p) {
            Bits b;
            if(v<tree.n) {b.add(v);return b;}
            for(auto e:tree.adj[v])if(e.to!=p)b=b|dfs(e.to,v);
            if(p>=tree.n)for(auto e:tree.adj[v])if(e.to==p)
                counts[b]+=retention(e.sup,e.len,1,1);
            return b;
        };
        dfs(tree.adj[0][0].to,0);
    }
    return constrainedDP(genes,{primary},{},primary,deadline,&counts);
}

Tree modelConsensus(const vector<Tree>& genes,const Tree& primary,double primaryWeight,double deadline,double absoluteLambda=0) {
    struct Model {
        double lambda,balance;
        int support;
        double power;
        bool bme,gls;
        int passes;
        double information;
        bool splits=false;
    };
    const vector<Model> models={
        {.05,.25,1,1,false,false,20,0},
        {.20,.25,1,1,false,false,20,0},
        {.10,0,1,1,false,false,20,0},
        {.10,.5,1,1,false,false,20,0},
        {.10,.25,1,1,true,false,20,0},
        {.10,0,1,1,false,true,20,0},
        {.10,.25,8,1.5,false,false,20,0},
        {.10,.25,1,1,false,false,20,1},
        {.10,0,1,1,false,false,2,0},
        {.10,0,1,1,false,false,0,0,true}
    };
    vector<Tree> trees{primary};
    unordered_map<Bits,double,BitsHash> counts;
    auto add=[&](const Tree& t,double weight) {
        unordered_map<Bits,int,BitsHash> clades;
        treeClades(t,clades);
        for(auto& kv:clades)counts[kv.first]+=weight;
    };
    add(primary,primaryWeight);
    Matrix distance=distances(genes,1),covariance;
    int index=0;
    for(auto model:models) {
        if(elapsed()>deadline-3)break;
        if(model.splits) {
            Tree t=geneSplitConsensus(genes,primary,.1,deadline-2);
            add(t,1);trees.push_back(move(t));
            cerr<<"Reliability model "<<++index<<" time "<<elapsed()<<"\n";
            continue;
        }
        Tree t=nj(distance);
        if(model.bme)bmeNNI(t,distance);
        auto gs=model.information>0
            ?makeInformationGtrees(genes,model.support,model.power,model.lambda,0,true,true,model.information,.5,true,absoluteLambda)
            :makeGtrees(genes,model.support,model.power,model.lambda,0,true,true,absoluteLambda);
        if(model.information<=0)normalizeLengthWeights(gs,t.n);
        QuartetCache cache(gs);
        if(model.gls) {
            if(covariance.empty())covariance=geneCorrelations(genes,1);
            cache.covariance=&covariance;cache.gls=true;cache.densityPower=.5;
        }
        nniOptimize(t,gs,model.passes,deadline-2,false,0,0,&cache,-1,model.balance);
        add(t,1);trees.push_back(move(t));
        cerr<<"Reliability model "<<++index<<" time "<<elapsed()<<"\n";
    }
    return constrainedDP(trees,{primary},{},primary,deadline,&counts);
}

Tree bootstrapConsensus(const vector<Tree>& genes,const vector<GTree>& gs,const Tree& initial,int samples,int dm,double dparam,
                        bool bme,double prior,double deadline,double geneLikelihood=-1,double balance=0) {
    int m=genes.size(),n=initial.n;
    vector<Matrix> perGene;
    for(const auto& g:genes)perGene.push_back(distances({g},dm,dparam));
    vector<Tree> trees;
    unordered_map<Bits,double,BitsHash> counts;
    auto addCounts=[&](const Tree& t,double weight) {
        unordered_map<Bits,int,BitsHash> clades;
        treeClades(t,clades);
        for(auto& kv:clades)counts[kv.first]+=weight;
    };
    addCounts(initial,prior);
    unsigned seed=1;
    if(const char* s=getenv("SEED"))seed=(unsigned)strtoul(s,nullptr,10);
    mt19937 rng(seed);
    QuartetCache cache(gs);
    double lastTime=0;
    for(int rep=0;rep<samples;++rep) {
        if(elapsed()+max(1.,lastTime*1.15)>deadline-2)break;
        double now=elapsed();
        vector<int> multiplicity(m,0);
        for(int k=0;k<m;++k)++multiplicity[rng()%m];
        Matrix d(n,vector<double>(n,0));
        vector<GTree> bg;
        for(int k=0;k<m;++k)if(multiplicity[k]) {
            double w=double(multiplicity[k])/m;
            for(int i=0;i<n;++i)for(int j=0;j<n;++j)d[i][j]+=w*perGene[k][i][j];
            for(int j=k;j<(int)gs.size();j+=m) {
                bg.push_back(gs[j]);bg.back().weight*=multiplicity[k];
            }
        }
        Tree t=nj(d);
        if(bme)bmeNNI(t,d);
        nniOptimize(t,bg,30,deadline-2,false,0,0,&cache,geneLikelihood,balance);
        addCounts(t,1);
        trees.push_back(move(t));
        lastTime=elapsed()-now;
        if((rep+1)%10==0)cerr<<"Bootstrap "<<rep+1<<" time "<<elapsed()<<"\n";
    }
    if(trees.empty())return initial;
    cerr<<"Bootstrap done "<<trees.size()<<" cache "<<cache.hits<<" / "<<cache.misses<<" time "<<elapsed()<<"\n";
    return constrainedDP(trees,{initial},{},initial,deadline,&counts);
}

struct PosteriorKernel {
    array<double,32> lm,la,lw;
    explicit PosteriorKernel(double rate) {
        const int n=32;
        for(int i=0;i<n/2;++i) {
            double z=cos(acos(-1.)*(i+.75)/(n+.5)),pp=0;
            for(int iter=0;iter<20;++iter) {
                double p1=1,p2=0;
                for(int j=1;j<=n;++j) {
                    double p3=p2;p2=p1;
                    p1=((2*j-1)*z*p2-(j-1)*p3)/j;
                }
                pp=n*(z*p1-p2)/(z*z-1);
                double nz=z-p1/pp;
                if(fabs(nz-z)<1e-15) {z=nz;break;}
                z=nz;
            }
            double weight=1/((1-z*z)*pp*pp);
            for(int side=0;side<2;++side) {
                int k=side?n-1-i:i;
                double y=(1+(side?z:-z))*.5;
                double logx=log(y)/rate,x=exp(logx);
                lm[k]=log1p(-2*x/3);
                la[k]=logx-log(3.);
                lw[k]=log(weight);
            }
        }
    }
    double logLikelihood(double major,double rest)const {
        double logs[32],largest=-1e300;
        for(int k=0;k<32;++k) {logs[k]=lw[k]+major*lm[k]+rest*la[k];largest=max(largest,logs[k]);}
        double sum=0;for(double x:logs)sum+=exp(x-largest);
        return largest+log(sum);
    }
};

Tree posteriorConsensus(const vector<GTree>& gs,const Tree& initial,int sweeps,double temperature,double countScale,
                        double branchPrior,double prior,double deadline,bool adaptive=false,double yule=0,
                        double sizePrior=0,bool autoPrior=false,bool normalizeEffective=false) {
    Tree t=initial;
    int n=t.n;
    unsigned seed=1;
    if(const char* s=getenv("SEED"))seed=(unsigned)strtoul(s,nullptr,10);
    mt19937 rng(seed);
    uniform_real_distribution<double> uniform(0,1);
    QuartetCache cache(gs);
    if(autoPrior) {
        vector<int> col(n);
        double total=0,count=0;
        for(int u=n;u<(int)t.adj.size();++u)for(auto e:t.adj[u])if(e.to>u&&e.to>=n) {
            int v=e.to;
            vector<int> a,b;
            for(auto f:t.adj[u])if(f.to!=v)a.push_back(f.to);
            for(auto f:t.adj[v])if(f.to!=u)b.push_back(f.to);
            if(a.size()!=2||b.size()!=2)continue;
            colorSub(t,a[0],u,0,col);colorSub(t,a[1],u,1,col);
            colorSub(t,b[0],v,2,col);colorSub(t,b[1],v,3,col);
            double effective=gs.size();
            auto q=cachedQuartets(cache,gs,col,nullptr,&effective);
            double sum=q[0]+q[1]+q[2];
            if(sum<1e-12)continue;
            effective=max(10.,min(200.,effective));
            double frequency=(q[0]/sum*effective+1)/(effective+3);
            double length=max(0.,-log(max(1e-6,1.5*(1-frequency))));
            int clade=0;
            for(int c:col)if(c==0||c==1)++clade;
            if(col[0]==0||col[0]==1)clade=n-clade;
            total+=pow(clade/2.,sizePrior)*length;
            ++count;
        }
        if(total>0)branchPrior=max(.05,min(50.,count/total));
    }
    vector<PosteriorKernel> kernels;
    for(int s=0;s<=n;++s)kernels.emplace_back(branchPrior*pow(max(2,s)/2.,sizePrior));
    cerr<<"Posterior branch prior "<<branchPrior<<" size power "<<sizePrior<<" time "<<elapsed()<<"\n";
    vector<Tree> trees;
    unordered_map<Bits,double,BitsHash> counts;
    auto addCounts=[&](const Tree& x,double weight) {
        unordered_map<Bits,int,BitsHash> clades;
        treeClades(x,clades);
        for(auto& kv:clades)counts[kv.first]+=weight;
    };
    addCounts(initial,prior);
    vector<int> colors(n);
    int burn=min(20,sweeps/5),moves=0;
    for(int it=0;it<sweeps && elapsed()<deadline-4;++it) {
        vector<pair<int,int>> edges;
        for(int u=n;u<(int)t.adj.size();++u)for(auto e:t.adj[u])if(e.to>u&&e.to>=n)edges.push_back({u,e.to});
        shuffle(edges.begin(),edges.end(),rng);
        for(auto [u,v]:edges) {
            if(elapsed()>deadline-4)break;
            vector<int> a,b;
            for(auto e:t.adj[u])if(e.to!=v)a.push_back(e.to);
            for(auto e:t.adj[v])if(e.to!=u)b.push_back(e.to);
            if(a.size()!=2||b.size()!=2)continue;
            colorSub(t,a[0],u,0,colors);colorSub(t,a[1],u,1,colors);
            colorSub(t,b[0],v,2,colors);colorSub(t,b[1],v,3,colors);
            double effective=gs.size();
            auto q=cachedQuartets(cache,gs,colors,nullptr,adaptive?&effective:nullptr);
            double size[4]={0,0,0,0};
            for(int c:colors)++size[c];
            double scale=countScale/(size[0]*size[1]*size[2]*size[3]),total=0;
            if(adaptive)scale*=effective/gs.size();
            if(normalizeEffective) {
                double sum=q[0]+q[1]+q[2];
                scale=sum>1e-20?countScale*effective/sum:0;
            }
            for(int k=0;k<3;++k) {q[k]=max(0.,q[k]*scale);total+=q[k];}
            double logp[3];
            for(int k=0;k<3;++k) {
                double clade=size[0]+size[k+1];
                if(colors[0]==0||colors[0]==k+1)clade=n-clade;
                int cladeSize=max(2,min(n,int(clade)));
                logp[k]=kernels[cladeSize].logLikelihood(q[k],total-q[k])/temperature;
                if(yule!=0) {
                    logp[k]-=yule*log(max(1.,clade-1));
                }
            }
            double top=*max_element(logp,logp+3),prob[3],sum=0;
            for(int k=0;k<3;++k) {prob[k]=exp(logp[k]-top);sum+=prob[k];}
            double choice=uniform(rng)*sum;
            int best=0;while(best<2 && choice>prob[best]) {choice-=prob[best];++best;}
            if(best) {
                if(best==1)t.swapBranches(u,a[1],v,b[0]);
                else t.swapBranches(u,a[1],v,b[1]);
                ++moves;
            }
        }
        if(it>=burn && (it-burn)%2==0) {addCounts(t,1);trees.push_back(t);}
        if((it+1)%50==0)cerr<<"Posterior "<<it+1<<" moves "<<moves<<" samples "<<trees.size()<<" cache "<<cache.hits<<" / "<<cache.misses<<" time "<<elapsed()<<"\n";
    }
    if(trees.empty())return initial;
    cerr<<"Posterior done "<<trees.size()<<" time "<<elapsed()<<"\n";
    return constrainedDP(trees,{initial},{},initial,deadline,&counts);
}

struct Routing {
    int n,nn,root;
    vector<int> depth,parent;
    vector<vector<int>> up;
    vector<vector<uint16_t>> lc;
    vector<vector<int16_t>> edgeId;
    vector<array<int,6>> edgeNodes;
    vector<vector<uint8_t>> colors;
    vector<double> combinations;
    explicit Routing(const Tree& t):n(t.n),nn(t.adj.size()),root(t.adj[0][0].to),
        depth(nn),parent(nn,-1),up(10,vector<int>(nn)),lc(n,vector<uint16_t>(n)),
        edgeId(nn,vector<int16_t>(nn,-1)) {
        function<void(int,int)> dfs=[&](int v,int p) {
            parent[v]=p;
            up[0][v]=p<0?v:p;
            for(int k=1;k<10;++k)up[k][v]=up[k-1][up[k-1][v]];
            for(auto e:t.adj[v])if(e.to!=p) {depth[e.to]=depth[v]+1;dfs(e.to,v);}
        };
        dfs(root,-1);
        auto lca=[&](int a,int b) {
            if(depth[a]<depth[b])swap(a,b);
            int diff=depth[a]-depth[b];
            for(int k=0;k<10;++k)if(diff&(1<<k))a=up[k][a];
            if(a==b)return a;
            for(int k=9;k>=0;--k)if(up[k][a]!=up[k][b]) {a=up[k][a];b=up[k][b];}
            return up[0][a];
        };
        for(int a=0;a<n;++a)for(int b=0;b<n;++b)lc[a][b]=lca(a,b);
        vector<int> col(n);
        for(int u=n;u<nn;++u)for(auto e:t.adj[u])if(e.to>u&&e.to>=n) {
            int v=e.to;
            vector<int> a,b;
            for(auto f:t.adj[u])if(f.to!=v)a.push_back(f.to);
            for(auto f:t.adj[v])if(f.to!=u)b.push_back(f.to);
            if(a.size()!=2||b.size()!=2)continue;
            int id=edgeNodes.size();
            edgeId[u][v]=edgeId[v][u]=id;
            edgeNodes.push_back({u,v,a[0],a[1],b[0],b[1]});
            colorSub(t,a[0],u,0,col);colorSub(t,a[1],u,1,col);
            colorSub(t,b[0],v,2,col);colorSub(t,b[1],v,3,col);
            vector<uint8_t> row(n);
            double sz[4]={0,0,0,0};
            for(int i=0;i<n;++i) {row[i]=col[i];++sz[col[i]];}
            colors.push_back(move(row));
            combinations.push_back(sz[0]*sz[1]*sz[2]*sz[3]);
        }
    }
    int median(int ab,int ac,int bc)const {
        int x=depth[ab]>depth[ac]?ab:ac;
        return depth[x]>depth[bc]?x:bc;
    }
    int quartetEdge(int a,int b,int c,int d)const {
        int ab=lc[a][b],ac=lc[a][c],ad=lc[a][d],bc=lc[b][c],bd=lc[b][d];
        int x=median(ab,ac,bc),y=median(ab,ad,bd);
        if(x==y)y=median(ac,ad,lc[c][d]);
        return edgeId[x][y];
    }
    int topology(int a,int b,int c,int d)const {
        int x[3]={depth[lc[a][b]]+depth[lc[c][d]],
                  depth[lc[a][c]]+depth[lc[b][d]],
                  depth[lc[a][d]]+depth[lc[b][c]]};
        return max_element(x,x+3)-x;
    }
};

struct QuartetPotential {
    array<uint8_t,4> taxon;
    array<float,3> value{};
    float importance=1;
};
struct Combinadics {
    uint32_t c[257][5]{};
    explicit Combinadics(int n) {
        for(int i=0;i<=n;++i) {
            c[i][0]=1;
            for(int k=1;k<=min(4,i);++k)c[i][k]=c[i-1][k]+c[i-1][k-1];
        }
    }
    uint32_t rank(array<int,4> a)const {
        sort(a.begin(),a.end());
        return a[0]+c[a[1]][2]+c[a[2]][3]+c[a[3]][4];
    }
    array<uint8_t,4> decode(uint32_t r,int n)const {
        array<uint8_t,4> a;
        int hi=n;
        for(int k=4;k>=2;--k) {
            int lo=k-1,top=hi;
            while(lo+1<top) {int m=(lo+top)/2;if(c[m][k]<=r)lo=m;else top=m;}
            a[k-1]=lo;r-=c[lo][k];hi=lo;
        }
        a[0]=r;
        return a;
    }
};

vector<QuartetPotential> sampledPotentials(const vector<Tree>& genes,const Tree& initial,int uniformSamples,int edgeSamples,
                                           int mode,double param,double nonlinearity,double weightPower,double deadline,bool marginal=false) {
    int n=initial.n;
    Combinadics comb(n);
    uint32_t totalQuartets=comb.c[n][4];
    unsigned seed=1;
    if(const char* s=getenv("SEED"))seed=(unsigned)strtoul(s,nullptr,10);
    mt19937 rng(seed);
    Routing base(initial);
    vector<uint32_t> ranks;
    ranks.reserve(uniformSamples+edgeSamples*base.edgeNodes.size());
    for(int rep=0;rep<uniformSamples;++rep) {
        array<int,4> a;
        for(int k=0;k<4;++k) {
            bool duplicate;
            do {
                a[k]=rng()%n;duplicate=false;
                for(int j=0;j<k;++j)if(a[j]==a[k])duplicate=true;
            } while(duplicate);
        }
        ranks.push_back(comb.rank(a));
    }
    for(int e=0;e<(int)base.edgeNodes.size();++e) {
        vector<int> group[4];
        for(int i=0;i<n;++i)group[base.colors[e][i]].push_back(i);
        uint32_t count=base.combinations[e],take=min(uint32_t(edgeSamples),count);
        auto push=[&](uint32_t index) {
            array<int,4> a;
            for(int k=0;k<4;++k) {a[k]=group[k][index%group[k].size()];index/=group[k].size();}
            ranks.push_back(comb.rank(a));
        };
        if(take==count)for(uint32_t i=0;i<count;++i)push(i);
        else {
            unordered_set<uint32_t> chosen;
            chosen.reserve(take*2);
            for(uint32_t j=count-take;j<count;++j) {
                uint32_t v=rng()%(j+1);
                if(!chosen.insert(v).second)chosen.insert(j);
            }
            for(uint32_t i:chosen)push(i);
        }
    }
    sort(ranks.begin(),ranks.end());
    vector<QuartetPotential> data;
    data.reserve(ranks.size());
    for(size_t i=0;i<ranks.size();) {
        size_t j=i+1;while(j<ranks.size() && ranks[j]==ranks[i])++j;
        auto a=comb.decode(ranks[i],n);
        int e=base.quartetEdge(a[0],a[1],a[2],a[3]);
        double inclusion=-expm1(uniformSamples*log1p(-1./totalQuartets));
        if(e>=0)inclusion=1-(1-inclusion)*(1-min(1.,edgeSamples/base.combinations[e]));
        double weight=1/pow(max(1e-30,inclusion),weightPower);
        data.push_back({a,{},float(weight)});
        i=j;
    }
    ranks.clear();ranks.shrink_to_fit();
    cerr<<"Quartet sample "<<data.size()<<" time "<<elapsed()<<"\n";
    vector<uint16_t> observations;
    if(marginal)observations.resize(data.size()*genes.size());
    vector<double> lut(32769);
    const double resolution=1024;
    for(int i=0;i<(int)lut.size();++i)lut[i]=-expm1(-i/resolution);
    for(int g=0;g<(int)genes.size();++g) {
        if(elapsed()>deadline) {cerr<<"Quartet construction deadline\n";return {};}
        const Tree& t=genes[g];
        double scale=param;
        if(mode==5) {
            double total=0;
            for(int v=0;v<(int)t.adj.size();++v)for(auto e:t.adj[v])if(e.to>v)total+=e.len;
            scale*=total/(2*n-3);
        }
        vector<vector<int>> d(n,vector<int>(n));
        for(int a=0;a<n;++a) {
            function<void(int,int,int)> dfs=[&](int v,int parent,int distance) {
                if(v<n)d[a][v]=distance;
                for(auto e:t.adj[v])if(e.to!=parent) {
                    double p=(v<n||e.to<n)?0:retention(e.sup,e.len,mode,scale);
                    int w=lround(-log(max(1e-14,1-p))*resolution);
                    dfs(e.to,v,distance+w);
                }
            };
            dfs(a,-1,0);
        }
        for(size_t index=0;index<data.size();++index) {
            auto& q=data[index];
            int a=q.taxon[0],b=q.taxon[1],c=q.taxon[2],z=q.taxon[3];
            int s0=d[a][b]+d[c][z],s1=d[a][c]+d[b][z],s2=d[a][z]+d[b][c];
            int top=s0<=s1?(s0<=s2?0:2):(s1<=s2?1:2);
            int low=min(s0,min(s1,s2)),high=max(s0,max(s1,s2));
            int delta=min((high-low+1)/2,32768);
            q.value[top]+=lut[delta];
            if(marginal)observations[index*genes.size()+g]=top*256+max(0,min(255,int(lround(evidenceScale*lut[delta]*255))));
        }
    }
    const int grid=32;
    float positive[256][grid],negative[256][grid];
    if(marginal)for(int w=0;w<256;++w)for(int z=0;z<grid;++z) {
        double x=double(z)/(grid-1);
        if(z==grid-1)x=1-1e-8;
        positive[w][z]=log1p(2*x*w/255);
        negative[w][z]=log1p(-x*w/255);
    }
    for(size_t index=0;index<data.size();++index) {
        if(index%16384==0 && elapsed()>deadline) {cerr<<"Quartet likelihood deadline\n";return {};}
        auto& q=data[index];
        double total=q.value[0]+q.value[1]+q.value[2];
        array<double,3> likelihood{};
        if(marginal) {
            float values[3][grid]{};
            for(int g=0;g<(int)genes.size();++g) {
                int code=observations[index*genes.size()+g],top=code/256,w=code%256;
                const float* p0=top==0?positive[w]:negative[w];
                const float* p1=top==1?positive[w]:negative[w];
                const float* p2=top==2?positive[w]:negative[w];
                for(int z=0;z<grid;++z) {
                    values[0][z]+=p0[z];values[1][z]+=p1[z];values[2][z]+=p2[z];
                }
            }
            for(int k=0;k<3;++k)likelihood[k]=*max_element(values[k],values[k]+grid);
        }
        for(int k=0;k<3;++k) {
            double raw=q.value[k],f=0;
            if(marginal)f=likelihood[k];
            else if(total>1e-8 && raw>total/3) {
                double p=raw/total;
                f=raw*log(3*p);
                if(total-raw>1e-8)f+=(total-raw)*log(1.5*(1-p));
            }
            q.value[k]=q.importance*((1-nonlinearity)*(raw-total/3)+nonlinearity*f);
        }
    }
    cerr<<"Quartet potentials ready time "<<elapsed()<<"\n";
    return data;
}

int potentialNNI(Tree& t,const vector<QuartetPotential>& data,int passes,double deadline) {
    static const int pairTopology[4][4]={{-1,0,1,2},{0,-1,2,1},{1,2,-1,0},{2,1,0,-1}};
    int total=0;
    for(int pass=0;pass<passes && elapsed()<deadline;++pass) {
        Routing route(t);
#ifndef NDEBUG
        double before=0;
        for(auto& q:data)before+=q.value[route.topology(q.taxon[0],q.taxon[1],q.taxon[2],q.taxon[3])];
#endif
        int m=route.edgeNodes.size();
        vector<array<double,3>> score(m);
        for(auto& q:data) {
            int a=q.taxon[0],b=q.taxon[1],c=q.taxon[2],d=q.taxon[3];
            int e=route.quartetEdge(a,b,c,d);
            if(e<0)continue;
            const auto& col=route.colors[e];
            int first=col[a];
            score[e][pairTopology[first][col[b]]]+=q.value[0];
            score[e][pairTopology[first][col[c]]]+=q.value[1];
            score[e][pairTopology[first][col[d]]]+=q.value[2];
        }
        vector<array<double,3>> moves;
        for(int e=0;e<m;++e) {
            int best=max_element(score[e].begin(),score[e].end())-score[e].begin();
            double gain=score[e][best]-score[e][0];
            if(best && gain>1e-8*max(1.,fabs(score[e][0])))moves.push_back({gain,double(e),double(best)});
        }
        sort(moves.begin(),moves.end(),[](const auto& a,const auto& b){return a[0]>b[0];});
        vector<bool> blocked(t.adj.size(),false);
        int changes=0;
        for(auto move:moves) {
            int e=move[1],best=move[2];
            auto z=route.edgeNodes[e];
            int u=z[0],v=z[1];
            if(blocked[u]||blocked[v])continue;
            if(best==1)t.swapBranches(u,z[3],v,z[4]);
            else t.swapBranches(u,z[3],v,z[5]);
            for(int x:z)blocked[x]=true;
            ++changes;
        }
        total+=changes;
#ifndef NDEBUG
        Routing after(t);
        double value=0;
        for(auto& q:data)value+=q.value[after.topology(q.taxon[0],q.taxon[1],q.taxon[2],q.taxon[3])];
        assert(value>=before-1e-6*max(1.,fabs(before)));
#endif
        cerr<<"Potential NNI "<<pass<<" moves "<<changes<<" time "<<elapsed()<<"\n";
        if(!changes)break;
    }
    return total;
}

#include "mpest.hpp"

int leafCount(const Tree& t,int v,int par) {
    if(v<t.n)return 1;
    int count=0;
    for(auto e:t.adj[v])if(e.to!=par)count+=leafCount(t,e.to,v);
    return count;
}

struct SprResult {int moves=0,calls=0;double gain=0;};
SprResult sprOptimize(Tree& t,const vector<GTree>& gs,int radius,int rounds,double deadline,QuartetCache* cache=nullptr) {
    SprResult result;
    mt19937 rng(1);
    for(int it=0;it<rounds && elapsed()<deadline;++it) {
        vector<pair<int,int>> prunes;
        for(int u=0;u<(int)t.adj.size();++u)for(auto e:t.adj[u])if(e.to>u) {
            int cnt=leafCount(t,u,e.to);
            if(cnt<=t.n-cnt && e.to>=t.n)prunes.push_back({u,e.to});
            else if(u>=t.n)prunes.push_back({e.to,u});
        }
        shuffle(prunes.begin(),prunes.end(),rng);
        int changed=0;
        for(auto [a,p]:prunes) {
            if(elapsed()>=deadline)break;
            vector<int> others;
            bool present=false;
            for(auto e:t.adj[p]) {
                if(e.to==a)present=true;
                else others.push_back(e.to);
            }
            if(!present || others.size()!=2)continue;
            int u=others[0],v=others[1];
            vector<int> base(t.n,1),cols(t.n);
            colorSub(t,a,p,0,base);
            double bestGain=0;
            int bx=-1,by=-1;
            function<void(int,int,int,double)> search=[&](int x,int parent,int depth,double gain) {
                if(x<t.n || depth>=radius || elapsed()>=deadline)return;
                vector<int> children;
                for(auto e:t.adj[x])if(e.to!=parent)children.push_back(e.to);
                if(children.size()!=2)return;
                cols=base;
                colorSub(t,children[0],x,2,cols);
                colorSub(t,children[1],x,3,cols);
                auto q=cache?cachedQuartets(*cache,gs,cols):quartetScores(gs,cols);++result.calls;
                double gains[2]={gain+q[1]-q[0],gain+q[2]-q[0]};
                for(int j=0;j<2;++j)if(gains[j]>bestGain+1e-6) {
                    bestGain=gains[j];bx=x;by=children[j];
                }
                for(int j=0;j<2;++j)search(children[j],x,depth+1,gains[j]);
            };
            search(u,p,0,0);
            search(v,p,0,0);
            if(bestGain>1e-5 && bx>=0) {
                t.remove(a,p);t.remove(p,u);t.remove(p,v);
                t.add(u,v);
                t.remove(bx,by);
                t.add(p,bx);t.add(p,by);t.add(p,a);
                ++changed;++result.moves;result.gain+=bestGain;
            }
        }
        cerr<<"SPR round "<<it<<" moves "<<changed<<" calls "<<result.calls<<" time "<<elapsed()<<"\n";
        if(!changed)break;
    }
    return result;
}

int main(int argc,char**argv) {
    try {
        if(argc<3)throw runtime_error("Usage: infer INPUT OUTPUT [options]");
        int dm=1,qm=1,sweeps=20,spr=0,rounds=2,extraStarts=0,bootstraps=0,posterior=0,maxSupport=0;
        int uniformQuartets=500000,edgeQuartets=3000;
        int mpest=0;
        bool upgma=false,score=false,pendant=false,dpSearch=false,bme=false,normalizeLength=false,adaptive=false,relative=false,marginal=false;
        bool bionj=false,empirical=false;
        bool gls=false;
        bool autoPrior=false;
        bool normalizeEffective=false;
        bool ensemble=false;
        double dp=1,qp=1,lambda=0,gpow=0,deadline=125,alpha=1,freq=0,prior=.001,gate=0,rooted=0;
        double temperature=1,countScale=1,branchPrior=.5;
        double nonlinearity=-1,weightPower=1;
        double geneLikelihood=-1;
        double balance=0,yule=0;
        double mpestPrior=0;
        double correlation=0,densityPower=.5;
        double ridge=.05;
        double lateGate=0;
        int freeSweeps=2;
        double sizePrior=0;
        double informationScale=0,informationMix=.5;
        double ensemblePrior=3.001;
        double absoluteLambda=0;
        double adaptiveWeights=0,adaptivePower=1;
        double splitPrior=0;
        string startPath;
        for(int i=3;i<argc;++i) {
            string o=argv[i];
            if(o=="--dist")dm=stoi(argv[++i]);
            else if(o=="--dparam")dp=stod(argv[++i]);
            else if(o=="--quartet")qm=stoi(argv[++i]);
            else if(o=="--qparam")qp=stod(argv[++i]);
            else if(o=="--lambda")lambda=stod(argv[++i]);
            else if(o=="--gene-power")gpow=stod(argv[++i]);
            else if(o=="--upgma")upgma=true;
            else if(o=="--sweeps")sweeps=stoi(argv[++i]);
            else if(o=="--time")deadline=stod(argv[++i]);
            else if(o=="--alpha")alpha=stod(argv[++i]);
            else if(o=="--start")startPath=argv[++i];
            else if(o=="--spr")spr=stoi(argv[++i]);
            else if(o=="--rounds")rounds=stoi(argv[++i]);
            else if(o=="--score")score=true;
            else if(o=="--pendant")pendant=true;
            else if(o=="--dp")dpSearch=true;
            else if(o=="--extra-starts")extraStarts=stoi(argv[++i]);
            else if(o=="--bme")bme=true;
            else if(o=="--freq")freq=stod(argv[++i]);
            else if(o=="--bootstrap")bootstraps=stoi(argv[++i]);
            else if(o=="--prior")prior=stod(argv[++i]);
            else if(o=="--gate")gate=stod(argv[++i]);
            else if(o=="--rooted")rooted=stod(argv[++i]);
            else if(o=="--posterior")posterior=stoi(argv[++i]);
            else if(o=="--temperature")temperature=stod(argv[++i]);
            else if(o=="--count-scale")countScale=stod(argv[++i]);
            else if(o=="--branch-prior")branchPrior=stod(argv[++i]);
            else if(o=="--normalize")normalizeLength=true;
            else if(o=="--adaptive")adaptive=true;
            else if(o=="--max-support")maxSupport=stoi(argv[++i]);
            else if(o=="--relative")relative=true;
            else if(o=="--nonlinear")nonlinearity=stod(argv[++i]);
            else if(o=="--uniform-quartets")uniformQuartets=stoi(argv[++i]);
            else if(o=="--edge-quartets")edgeQuartets=stoi(argv[++i]);
            else if(o=="--weight-power")weightPower=stod(argv[++i]);
            else if(o=="--gene-likelihood")geneLikelihood=stod(argv[++i]);
            else if(o=="--likelihood")marginal=true;
            else if(o=="--confidence-scale")evidenceScale=stod(argv[++i]);
            else if(o=="--moments")likelihoodMoments=stoi(argv[++i]);
            else if(o=="--moment-cap")momentCap=stod(argv[++i]);
            else if(o=="--balance")balance=stod(argv[++i]);
            else if(o=="--yule")yule=stod(argv[++i]);
            else if(o=="--bionj")bionj=true;
            else if(o=="--variance")empirical=true;
            else if(o=="--mpest")mpest=stoi(argv[++i]);
            else if(o=="--mpest-prior")mpestPrior=stod(argv[++i]);
            else if(o=="--correlation")correlation=stod(argv[++i]);
            else if(o=="--density-power")densityPower=stod(argv[++i]);
            else if(o=="--gls")gls=true;
            else if(o=="--ridge")ridge=stod(argv[++i]);
            else if(o=="--late-gate")lateGate=stod(argv[++i]);
            else if(o=="--free-sweeps")freeSweeps=stoi(argv[++i]);
            else if(o=="--size-prior")sizePrior=stod(argv[++i]);
            else if(o=="--auto-prior")autoPrior=true;
            else if(o=="--effective-normalize")normalizeEffective=true;
            else if(o=="--information-scale")informationScale=stod(argv[++i]);
            else if(o=="--information-mix")informationMix=stod(argv[++i]);
            else if(o=="--ensemble")ensemble=true;
            else if(o=="--ensemble-prior")ensemblePrior=stod(argv[++i]);
            else if(o=="--absolute-lambda")absoluteLambda=stod(argv[++i]);
            else if(o=="--adaptive-weights")adaptiveWeights=stod(argv[++i]);
            else if(o=="--adaptive-power")adaptivePower=stod(argv[++i]);
            else if(o=="--split-prior")splitPrior=stod(argv[++i]);
            else if(o=="--normalization-cap")maximumNormalization=stod(argv[++i]);
            else throw runtime_error("Unknown option "+o);
        }
        ifstream in(argv[1]);if(!in)throw runtime_error("Cannot open input");
        vector<Tree> genes;vector<string> labels;unordered_map<string,int> ids;
        string line,all;
        while(getline(in,line))all+=line+"\n";
        if(all.compare(0,3,"\xEF\xBB\xBF")==0)all.erase(0,3);
        size_t prev=0;
        int commentDepth=0;
        bool quoted=false;
        for(size_t pos=0;pos<all.size();++pos) {
            char c=all[pos];
            if(quoted) {
                if(c=='\'') {
                    if(pos+1<all.size()&&all[pos+1]=='\'')++pos;
                    else quoted=false;
                }
            } else if(commentDepth>0) {
                if(c=='[')++commentDepth;
                if(c==']')--commentDepth;
            } else if(c=='\'')quoted=true;
            else if(c=='[')commentDepth=1;
            else if(c==';') {
                genes.push_back(Parser(all.substr(prev,pos-prev+1)).parse(labels,ids));
                prev=pos+1;
            }
        }
        if(genes.empty())throw runtime_error("No input trees");
        Tree t;
        Matrix dmat;
        if(startPath.empty()) {
            dmat=distances(genes,dm,dp);
            Matrix variance;
            if(empirical) {
                variance.assign(labels.size(),vector<double>(labels.size()));
                for(const auto& gene:genes) {
                    auto d=distances({gene},dm,dp);
                    for(int i=0;i<(int)labels.size();++i)for(int j=0;j<(int)labels.size();++j) {
                        double diff=d[i][j]-dmat[i][j];
                        variance[i][j]+=diff*diff/genes.size();
                    }
                }
            }
            t=nj(dmat,upgma,alpha,bionj,empirical?&variance:nullptr);
        } else {
            ifstream f(startPath);string x((istreambuf_iterator<char>(f)),{});
            t=Parser(x).parse(labels,ids);
        }
        if(bme) {
            if(dmat.empty())dmat=distances(genes,dm,dp);
            cerr<<"BME "<<bmeNNI(t,dmat)<<" moves time "<<elapsed()<<"\n";
        }
        cerr<<"start "<<elapsed()<<"\n";
        if(qm>=0) {
            vector<Tree> qgenes;
            if(freq!=0)qgenes=recurrenceWeights(genes,freq);
            auto gs=adaptiveWeights>0?makeAdaptiveGtrees(freq==0?genes:qgenes,adaptiveWeights,adaptivePower,absoluteLambda)
                     :informationScale>0?makeInformationGtrees(freq==0?genes:qgenes,qm,qp,lambda,gpow,pendant,relative,informationScale,informationMix,normalizeLength,absoluteLambda)
                     :likelihoodMoments>0?makeMomentGtrees(freq==0?genes:qgenes,likelihoodMoments,qm,qp)
                     :maxSupport>0?makeMaxGtrees(freq==0?genes:qgenes,maxSupport,lambda,gpow,pendant,relative,absoluteLambda)
                                 :makeGtrees(freq==0?genes:qgenes,qm,qp,lambda,gpow,pendant,relative,absoluteLambda);
            if(normalizeLength&&informationScale<=0&&adaptiveWeights<=0)normalizeLengthWeights(gs,labels.size());
            if(score) {
                vector<double> weights;
                for(auto& g:gs)weights.push_back(g.weight);
                sort(weights.begin(),weights.end());
                cerr<<"Gene weights "<<weights.front()<<" "<<weights[weights.size()/2]<<" "<<weights.back()<<"\n";
            }
            Matrix covariance;
            if(correlation>0)covariance=geneCorrelations(genes,correlation);
            QuartetCache cache(gs);
            if(correlation>0) {
                cache.covariance=&covariance;cache.densityPower=densityPower;cache.gls=gls;cache.ridge=ridge;
            }
            int firstSweeps=lateGate>0?min(sweeps,freeSweeps):sweeps;
            auto res=nniOptimize(t,gs,firstSweeps,deadline,false,gate,rooted,&cache,geneLikelihood,balance);
            if(lateGate>0&&sweeps>firstSweeps) {
                auto rest=nniOptimize(t,gs,sweeps-firstSweeps,deadline,false,lateGate,rooted,&cache,geneLikelihood,balance);
                res.moves+=rest.moves;res.sweeps+=rest.sweeps;res.gain+=rest.gain;
            }
            cerr<<"NNI "<<res.moves<<" moves "<<res.sweeps<<" sweeps gain "<<res.gain<<" time "<<elapsed()<<"\n";
            if(score)cerr<<setprecision(14)<<"score "<<totalScore(t,gs)<<" time "<<elapsed()<<"\n";
            if(dpSearch) {
                vector<Tree> extras{t};
                const int modes[8]={0,2,3,4,6,10,1,10};
                for(int i=0;i<min(extraStarts,8) && elapsed()<deadline-5;++i) {
                    Tree other=nj(distances(genes,modes[i],1),i>=6);
                    extras.push_back(other);
                    nniOptimize(other,gs,sweeps,deadline-3,false,0,0,&cache);
                    extras.push_back(other);
                }
                t=constrainedDP(genes,extras,gs,t,deadline-2);
                auto nr=nniOptimize(t,gs,sweeps,deadline,false,gate,rooted,&cache,geneLikelihood,balance);
                cerr<<"DP NNI "<<nr.moves<<" moves time "<<elapsed()<<"\n";
                if(score)cerr<<setprecision(14)<<"score "<<totalScore(t,gs)<<" time "<<elapsed()<<"\n";
            }
            if(spr>0) {
                auto sr=sprOptimize(t,gs,spr,rounds,deadline,&cache);
                cerr<<"SPR "<<sr.moves<<" moves gain "<<sr.gain<<" calls "<<sr.calls<<" time "<<elapsed()<<"\n";
                auto nr=nniOptimize(t,gs,sweeps,deadline,false,gate,rooted,&cache,geneLikelihood,balance);
                cerr<<"NNI "<<nr.moves<<" moves "<<nr.sweeps<<" sweeps time "<<elapsed()<<"\n";
                if(score)cerr<<setprecision(14)<<"score "<<totalScore(t,gs)<<" time "<<elapsed()<<"\n";
            }
            if(bootstraps>0)t=bootstrapConsensus(genes,gs,t,bootstraps,dm,dp,bme,prior,deadline,geneLikelihood,balance);
            if(posterior>0)t=posteriorConsensus(gs,t,posterior,temperature,countScale,branchPrior,prior,deadline,adaptive,yule,sizePrior,autoPrior,normalizeEffective);
            if(nonlinearity>=0 || marginal) {
                auto potentials=sampledPotentials(freq==0?genes:qgenes,t,uniformQuartets,edgeQuartets,qm,qp,nonlinearity>=0?nonlinearity:1,weightPower,deadline-5,marginal);
                if(!potentials.empty())potentialNNI(t,potentials,50,deadline);
            }
            if(mpest>0) {
                auto triples=makeRootedTriples(freq==0?genes:qgenes,qm,qp,lambda,relative,gs,deadline-3);
                if(!triples.empty())t=mpestRefine(t,triples,mpest,mpestPrior,deadline);
            }
            if(splitPrior>0)t=geneSplitConsensus(genes,t,splitPrior,deadline);
            if(ensemble)t=modelConsensus(genes,t,ensemblePrior,deadline,absoluteLambda);
        }
        writeTree(t,labels,argv[2]);
        cerr<<"time "<<elapsed()<<"\n";
    } catch(exception& e) {cerr<<"error: "<<e.what()<<"\n";return 1;}
    return 0;
}
