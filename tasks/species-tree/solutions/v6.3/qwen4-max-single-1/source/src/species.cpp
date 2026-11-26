#include <bits/stdc++.h>
using namespace std;

static const int MAXN = 256;
static const int W = 4;
struct Mask { uint64_t w[W]; };
static inline Mask operator&(const Mask&a,const Mask&b){Mask r;for(int i=0;i<W;i++)r.w[i]=a.w[i]&b.w[i];return r;}
static inline Mask operator^(const Mask&a,const Mask&b){Mask r;for(int i=0;i<W;i++)r.w[i]=a.w[i]^b.w[i];return r;}
static inline Mask operator|(const Mask&a,const Mask&b){Mask r;for(int i=0;i<W;i++)r.w[i]=a.w[i]|b.w[i];return r;}
static inline bool operator==(const Mask&a,const Mask&b){for(int i=0;i<W;i++)if(a.w[i]!=b.w[i])return false;return true;}
static inline int popcount(const Mask&a){int s=0;for(int i=0;i<W;i++)s+=__builtin_popcountll(a.w[i]);return s;}
static inline void setbit(Mask&a,int i){a.w[i>>6]|=1ULL<<(i&63);}
static inline bool getbit(const Mask&a,int i){return (a.w[i>>6]>>(i&63))&1ULL;}
static inline Mask fullmask(int n){Mask m{};for(int i=0;i<n;i++)setbit(m,i);return m;}

struct GeneEdge { Mask s; int sz; double wt; };
struct Gene { vector<GeneEdge> edges; vector<vector<pair<int,double>>> adj; int n; };
struct PEdge { int to; double sup; double len; };

// Newick parser -> unrooted weighted adjacency
struct Parser {
    const string& text; size_t pos=0;
    unordered_map<string,int>& labelMap; vector<string>& labels;
    vector<vector<PEdge>> adj;
    vector<int> degree;
    int nextInternal;
    int n;
    Parser(const string& t, unordered_map<string,int>& lm, vector<string>& lb, int nleaves)
        : text(t), labelMap(lm), labels(lb), n(nleaves) {
        adj.resize(2*n); degree.assign(2*n,0); nextInternal=n;
    }
    void skip(){ while(pos<text.size() && isspace((unsigned char)text[pos])) pos++; }
    string readLabel(){ skip(); size_t st=pos; while(pos<text.size() && !strchr(",():; \t\n\r",text[pos])) pos++; return text.substr(st,pos-st); }
    double readNum(){ skip(); size_t st=pos; while(pos<text.size() && (isdigit((unsigned char)text[pos])||text[pos]=='.'||text[pos]=='e'||text[pos]=='E'||text[pos]=='+'||text[pos]=='-')) pos++; if(pos==st)return 0; return atof(text.substr(st,pos-st).c_str()); }
    void addEdge(int u,int v,double sup,double len){ adj[u].push_back({v,sup,len}); adj[v].push_back({u,sup,len}); degree[u]++; degree[v]++; }
    int parseNode(){
        skip();
        if(pos<text.size() && text[pos]=='('){
            int id=nextInternal++;
            pos++; // (
            vector<int> children;
            while(true){ children.push_back(parseNode()); skip(); if(pos<text.size()&&text[pos]==','){pos++;continue;} break; }
            skip(); if(pos<text.size()&&text[pos]==')')pos++;
            // optional internal label (support)
            double sup=1.0; bool hasSup=false;
            skip();
            if(pos<text.size() && text[pos]!=':' && text[pos]!=',' && text[pos]!=')' && text[pos]!=';' && text[pos]!='('){
                string lab=readLabel(); if(!lab.empty()){ sup=atof(lab.c_str()); hasSup=true; }
            }
            double len=0; skip(); if(pos<text.size()&&text[pos]==':'){pos++;len=readNum();}
            for(int c:children){ addEdge(id,c,hasSup?sup:1.0,len); }
            return id;
        } else {
            string lab=readLabel();
            auto it=labelMap.find(lab);
            int id;
            if(it==labelMap.end()){ id=(int)labels.size(); labelMap[lab]=id; labels.push_back(lab); }
            else id=it->second;
            double len=0; skip(); if(pos<text.size()&&text[pos]==':'){pos++;len=readNum();}
            return id;
        }
    }
    void suppressDegree2(){
        for(int v=n; v<nextInternal; v++){
            if(degree[v]==2){
                int a=adj[v][0].to, b=adj[v][1].to;
                double sup=0.5*(adj[v][0].sup+adj[v][1].sup);
                double len=adj[v][0].len+adj[v][1].len;
                // remove v-a
                adj[a].erase(remove_if(adj[a].begin(),adj[a].end(),[&](auto&p){return p.to==v;}),adj[a].end());
                adj[b].erase(remove_if(adj[b].begin(),adj[b].end(),[&](auto&p){return p.to==v;}),adj[b].end());
                degree[a]--; degree[b]--; degree[v]=0; adj[v].clear();
                addEdge(a,b,sup,len);
            }
        }
    }
};

Gene parseGene(const string& text, unordered_map<string,int>& labelMap, vector<string>& labels, int n){
    Parser p(text,labelMap,labels,n);
    p.parseNode();
    p.suppressDegree2();
    Gene g; g.n=n; g.adj.resize(2*n);
    // compact nodes: leaves 0..n-1, internals with degree>0
    vector<int> remap(2*n,-1);
    for(int i=0;i<n;i++)remap[i]=i;
    int cnt=n;
    for(int i=n;i<p.nextInternal;i++)if(p.degree[i]>0)remap[i]=cnt++;
    g.adj.resize(cnt);
    vector<unordered_map<int,double>> sup(cnt);
    for(int u=0;u<p.nextInternal;u++)if(remap[u]>=0){
        for(auto&e:p.adj[u])if(remap[e.to]>=0){g.adj[remap[u]].push_back({remap[e.to],e.len}); sup[remap[u]][remap[e.to]]=e.sup;}
    }
    // compute splits via DFS rooted at n (first internal)
    int root=n;
    vector<int> parent(cnt,-1); vector<Mask> sub(cnt); vector<double> pw(cnt,1.0); vector<double> pl(cnt,0.0);
    vector<int> order; order.reserve(cnt);
    stack<int> st; st.push(root); parent[root]=-2;
    while(!st.empty()){int u=st.top();st.pop();order.push_back(u);for(auto&e:g.adj[u])if(e.first!=parent[u]){parent[e.first]=u;pw[e.first]=sup[u][e.first];st.push(e.first);} }
    Mask full=fullmask(n);
    for(int i=cnt-1;i>=0;i--){int u=order[i]; if(u<n){setbit(sub[u],u);} else {Mask m{};for(auto&e:g.adj[u])if(e.first!=parent[u])m=m|sub[e.first];sub[u]=m;} }
    for(int u=0;u<cnt;u++){ if(parent[u]==-2)continue; Mask m=sub[u]; int sz=popcount(m); if(sz>=2&&sz<=n-2){ if(getbit(m,0))m=m^full; g.edges.push_back({m,sz,pw[u]}); } }
    return g;
}

// distance matrix (internode or branch length)
vector<vector<double>> geneDistance(const Gene& g, bool weighted){
    int n=g.n; vector<vector<double>> d(n,vector<double>(n,0));
    for(int src=0;src<n;src++){
        vector<double> dist(g.adj.size(),-1); vector<int> par(g.adj.size(),-1);
        queue<int>q; q.push(src); dist[src]=0;
        while(!q.empty()){int u=q.front();q.pop();for(auto&e:g.adj[u])if(dist[e.first]<0){dist[e.first]=dist[u]+(weighted?e.second:1.0);par[e.first]=u;q.push(e.first);} }
        for(int j=0;j<n;j++)d[src][j]=dist[j];
    }
    return d;
}

string nj(const vector<vector<double>>& dist, const vector<string>& labels){
    int n=dist.size(); vector<vector<double>> d=dist;
    vector<int> active(n); iota(active.begin(),active.end(),0);
    vector<string> names=labels;
    while(active.size()>3){
        int m=active.size(); vector<double> r(m,0);
        for(int a=0;a<m;a++)for(int b=0;b<m;b++)if(a!=b)r[a]+=d[active[a]][active[b]];
        double best=1e300; int bi=-1,bj=-1;
        for(int a=0;a<m;a++)for(int b=0;b<a;b++){double q=(m-2)*d[active[a]][active[b]]-r[a]-r[b];if(q<best){best=q;bi=a;bj=b;}}
        int i=active[bi],j=active[bj];
        int k=d.size(); vector<double> row(k+1,0);
        for(int h:active)if(h!=i&&h!=j)row[h]=0.5*(d[i][h]+d[j][h]-d[i][j]);
        d.push_back(row); for(int h=0;h<k;h++)d[h].push_back(row[h]);
        names.push_back("("+names[i]+","+names[j]+")");
        active.erase(remove(active.begin(),active.end(),i),active.end());
        active.erase(remove(active.begin(),active.end(),j),active.end());
        active.push_back(k);
    }
    string s="("+names[active[0]]+","+names[active[1]]+","+names[active[2]]+");";
    return s;
}

// Species tree (unrooted, degree 3 internals)
struct STree {
    int n; vector<vector<int>> adj; vector<int> parent; vector<Mask> sub; Mask full;
    STree(){}
    STree(int n):n(n),adj(2*n-2),parent(2*n-2),sub(2*n-2),full(fullmask(n)){}
    void buildFromNewick(const string& text, unordered_map<string,int>& labelMap, vector<string>& labels){
        Parser p(text,labelMap,labels,n); p.parseNode(); p.suppressDegree2();
        vector<int> remap(2*n,-1); for(int i=0;i<n;i++)remap[i]=i; int cnt=n;
        for(int i=n;i<p.nextInternal;i++)if(p.degree[i]>0)remap[i]=cnt++;
        adj.assign(cnt,{});
        for(int u=0;u<p.nextInternal;u++)if(remap[u]>=0)for(auto&e:p.adj[u])if(remap[e.to]>=0)adj[remap[u]].push_back(remap[e.to]);
        recompute();
    }
    void recompute(){
        int root=n; fill(parent.begin(),parent.end(),-1);
        vector<int> order; order.reserve(adj.size());
        vector<char> vis(adj.size(),0);
        stack<int>st; st.push(root); parent[root]=-2; vis[root]=1;
        while(!st.empty()){int u=st.top();st.pop();order.push_back(u);for(int v:adj[u])if(!vis[v]){vis[v]=1;parent[v]=u;st.push(v);} }
        for(int i=(int)order.size()-1;i>=0;i--){int u=order[i]; if(u<n){Mask m{};setbit(m,u);sub[u]=m;} else {Mask m{};for(int v:adj[u])if(v!=parent[u])m=m|sub[v];sub[u]=m;} }
    }
    Mask getMask(int u,int v) const { if(parent[v]==u)return sub[v]; Mask r=full^sub[u]; return r; }
    string toNewick(const vector<string>& labels) const {
        // root at n
        function<string(int,int)> rec=[&](int u,int par)->string{
            if(u<n)return labels[u];
            vector<int> ch; for(int v:adj[u])if(v!=par)ch.push_back(v);
            if(ch.size()==2)return "("+rec(ch[0],u)+","+rec(ch[1],u)+")";
            return "("+rec(ch[0],u)+","+rec(ch[1],u)+","+rec(ch[2],u)+")";
        };
        return rec(n,-1)+";";
    }
};

static inline double C2(int x){return x<2?0.0:0.5*x*(x-1);}

// NNI deltas for edge u-v. Returns {delta swap B<->C, delta swap B<->D}
pair<double,double> nniDelta(const STree& T,int u,int v,const vector<GeneEdge>& edges){
    int a=-1,b=-1,c=-1,d=-1;
    for(int x:T.adj[u])if(x!=v){if(a<0)a=x;else b=x;}
    for(int x:T.adj[v])if(x!=u){if(c<0)c=x;else d=x;}
    Mask A=T.getMask(u,a),B=T.getMask(u,b),C=T.getMask(v,c),D=T.getMask(v,d);
    int As=popcount(A),Bs=popcount(B),Cs=popcount(C),Ds=popcount(D);
    double d1=0,d2=0;
    for(const auto&e:edges){
        int as=popcount(A&e.s), bs=popcount(B&e.s), cs=popcount(C&e.s), ds=popcount(D&e.s);
        int at=As-as,bt=Bs-bs,ct=Cs-cs,dt=Ds-ds;
        // current AB|CD ; swap B<->C -> AC|BD ; swap B<->D -> AD|BC
        double cur = (double)as*bs*ct*dt + (double)at*bt*cs*ds;
        double s1 = (double)as*cs*bt*dt + (double)at*ct*bs*ds;
        double s2 = (double)as*ds*bt*ct + (double)at*dt*bs*cs;
        d1 += e.wt*(s1-cur);
        d2 += e.wt*(s2-cur);
    }
    return {d1,d2};
}

void doSwap(STree& T,int u,int v,int b,int c){
    T.adj[u].erase(remove(T.adj[u].begin(),T.adj[u].end(),b),T.adj[u].end());
    T.adj[v].erase(remove(T.adj[v].begin(),T.adj[v].end(),c),T.adj[v].end());
    T.adj[b].erase(remove(T.adj[b].begin(),T.adj[b].end(),u),T.adj[b].end());
    T.adj[c].erase(remove(T.adj[c].begin(),T.adj[c].end(),v),T.adj[c].end());
    T.adj[u].push_back(c); T.adj[v].push_back(b);
    T.adj[c].push_back(u); T.adj[b].push_back(v);
    T.recompute();
}

void leafMove(STree& T, mt19937& rng){
    int l=rng()%T.n;
    int p=T.adj[l][0];
    int a=-1,b=-1;
    for(int x:T.adj[p])if(x!=l){if(a<0)a=x;else b=x;}
    // detach p (suppress)
    T.adj[p].clear();
    T.adj[a].erase(remove(T.adj[a].begin(),T.adj[a].end(),p),T.adj[a].end());
    T.adj[b].erase(remove(T.adj[b].begin(),T.adj[b].end(),p),T.adj[b].end());
    T.adj[a].push_back(b); T.adj[b].push_back(a);
    // pick random edge not involving p or l
    vector<pair<int,int>> el;
    for(int u=0;u<(int)T.adj.size();u++)if(u!=p&&u!=l)for(int v:T.adj[u])if(v>u&&v!=p&&v!=l)el.push_back({u,v});
    if(el.empty()){ // revert
        T.adj[a].erase(remove(T.adj[a].begin(),T.adj[a].end(),b),T.adj[a].end());
        T.adj[b].erase(remove(T.adj[b].begin(),T.adj[b].end(),a),T.adj[b].end());
        T.adj[a].push_back(p);T.adj[b].push_back(p);T.adj[p]={a,b,l};T.recompute();return;
    }
    auto [x,y]=el[rng()%el.size()];
    T.adj[x].erase(remove(T.adj[x].begin(),T.adj[x].end(),y),T.adj[x].end());
    T.adj[y].erase(remove(T.adj[y].begin(),T.adj[y].end(),x),T.adj[y].end());
    T.adj[x].push_back(p);T.adj[y].push_back(p);
    T.adj[p]={x,y,l}; T.adj[l]={p};
    T.recompute();
}

void subtreeMove(STree& T, mt19937& rng){
    int s;
    if(rng()%2){ s=rng()%T.n; }
    else { s=T.n+(rng()%(T.adj.size()-T.n)); }
    if(s==T.n)return;
    int p=T.parent[s]; if(p<0)return;
    Mask S=T.sub[s];
    vector<char> inS(T.adj.size(),0);
    {stack<int>st;st.push(s);inS[s]=1;while(!st.empty()){int u=st.top();st.pop();for(int v:T.adj[u])if(T.parent[v]==u&&!inS[v]){inS[v]=1;st.push(v);}}}
    int a=-1,b=-1;
    for(int x:T.adj[p])if(x!=s){if(a<0)a=x;else b=x;}
    if(a<0||b<0)return;
    // detach s from p, suppress p
    T.adj[s].erase(remove(T.adj[s].begin(),T.adj[s].end(),p),T.adj[s].end());
    T.adj[p].clear();
    T.adj[a].erase(remove(T.adj[a].begin(),T.adj[a].end(),p),T.adj[a].end());
    T.adj[b].erase(remove(T.adj[b].begin(),T.adj[b].end(),p),T.adj[b].end());
    T.adj[a].push_back(b);T.adj[b].push_back(a);
    // pick target edge outside S, not involving p
    vector<pair<int,int>> el;
    for(int u=0;u<(int)T.adj.size();u++)if(u!=p&&!inS[u])for(int v:T.adj[u])if(v>u&&v!=p&&!inS[v])el.push_back({u,v});
    if(el.empty()){
        T.adj[a].erase(remove(T.adj[a].begin(),T.adj[a].end(),b),T.adj[a].end());
        T.adj[b].erase(remove(T.adj[b].begin(),T.adj[b].end(),a),T.adj[b].end());
        T.adj[a].push_back(p);T.adj[b].push_back(p);T.adj[p]={a,b,s};T.adj[s].push_back(p);T.recompute();return;
    }
    auto [x,y]=el[rng()%el.size()];
    T.adj[x].erase(remove(T.adj[x].begin(),T.adj[x].end(),y),T.adj[x].end());
    T.adj[y].erase(remove(T.adj[y].begin(),T.adj[y].end(),x),T.adj[y].end());
    T.adj[x].push_back(p);T.adj[y].push_back(p);
    T.adj[p]={x,y,s};T.adj[s].push_back(p);
    T.recompute();
}

double fullScore(const STree& T,const vector<GeneEdge>& edges){
    double total=0;
    for(int u=T.n;u<(int)T.adj.size();u++){
        if(T.adj[u].size()!=3)continue;
        Mask A=T.getMask(u,T.adj[u][0]),B=T.getMask(u,T.adj[u][1]),C=T.getMask(u,T.adj[u][2]);
        int As=popcount(A),Bs=popcount(B),Cs=popcount(C);
        double q=0;
        for(const auto&e:edges){
            int as=popcount(A&e.s),bs=popcount(B&e.s),cs=popcount(C&e.s);
            int at=As-as,bt=Bs-bs,ct=Cs-cs;
            double t = C2(as)*bt*ct + C2(at)*bs*cs
                     + C2(bs)*at*ct + C2(bt)*as*cs
                     + C2(cs)*at*bt + C2(ct)*as*bs;
            q += e.wt*t;
        }
        total += q;
    }
    return 0.5*total;
}

double hillClimb(STree& T,const vector<GeneEdge>& edges, double score, mt19937& rng, int maxSweeps=10000, double deadline=1e300, chrono::steady_clock::time_point t0=chrono::steady_clock::now()){
    for(int sweep=0;sweep<maxSweeps;sweep++){
        if(chrono::duration<double>(chrono::steady_clock::now()-t0).count()>deadline)break;
        bool improved=false;
        vector<pair<int,int>> edgesList;
        for(int u=T.n;u<(int)T.adj.size();u++)for(int v:T.adj[u])if(v>u&&v>=T.n)edgesList.push_back({u,v});
        shuffle(edgesList.begin(),edgesList.end(),rng);
        int ec=0;
        for(auto [u,v]:edgesList){
            if((++ec&31)==0 && chrono::duration<double>(chrono::steady_clock::now()-t0).count()>deadline)return score;
            // verify edge still exists
            bool found=false; for(int x:T.adj[u])if(x==v)found=true; if(!found)continue;
            auto [d1,d2]=nniDelta(T,u,v,edges);
            if(d1>1e-9 || d2>1e-9){
                int a=-1,b=-1,c=-1,d=-1;
                for(int x:T.adj[u])if(x!=v){if(a<0)a=x;else b=x;}
                for(int x:T.adj[v])if(x!=u){if(c<0)c=x;else d=x;}
                if(d1>=d2){ doSwap(T,u,v,b,c); score+=d1; }
                else { doSwap(T,u,v,b,d); score+=d2; }
                improved=true;
            }
        }
        if(!improved)break;
    }
    return score;
}

double leafSearch(STree& T,const vector<GeneEdge>& edges,double score,mt19937& rng,double deadline,chrono::steady_clock::time_point t0,int maxIter=2000){
    int stale=0;
    for(int it=0;it<maxIter;it++){
        if(chrono::duration<double>(chrono::steady_clock::now()-t0).count()>deadline)break;
        STree cand=T;
        if(rng()%3==0)subtreeMove(cand,rng);else leafMove(cand,rng);
        double sc=fullScore(cand,edges);
        if(sc>score+1e-9){T=cand;score=sc;stale=0;}else{stale++;}
        if(stale>400)break;
    }
    return score;
}

int main(int argc,char**argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(argc<3){cerr<<"usage: species INPUT OUTPUT\n";return 1;}
    auto t0=chrono::steady_clock::now();
    double timeLimit=120.0;
    if(getenv("SPECIES_TIME"))timeLimit=atof(getenv("SPECIES_TIME"));
    string inpath=argv[1],outpath=argv[2];
    ifstream fin(inpath); if(!fin){cerr<<"cannot open input\n";return 1;}
    vector<string> lines; string line;
    while(getline(fin,line)){ if(!line.empty())lines.push_back(line); }
    fin.close();
    unordered_map<string,int> labelMap; vector<string> labels;
    // first pass to get labels
    {
        // parse first tree to collect labels
        vector<Gene> tmp;
        // We'll parse all genes; labelMap built incrementally
    }
    vector<Gene> genes;
    int n=0;
    for(size_t i=0;i<lines.size();i++){
        if(i==0){
            // determine n by counting leaves: parse with a temporary map
            unordered_map<string,int> lm; vector<string> lb;
            // count leaves by scanning labels? parse with n guess 250
            Gene g=parseGene(lines[i],lm,lb,250);
            n=lb.size(); labelMap=lm; labels=lb;
            // reparse with correct n? parseGene used 250 but labels size n; adj sized 2*250, fine. edges use n=250? bug: g.n=250 but actual n.
            // We'll reparse all with correct n below.
            genes.clear();
        }
        Gene g=parseGene(lines[i],labelMap,labels,n);
        genes.push_back(move(g));
    }
    if(n==0){cerr<<"no leaves\n";return 1;}
    // aggregate edges
    vector<GeneEdge> allEdges;
    for(auto&g:genes)for(auto&e:g.edges)allEdges.push_back(e);
    int wmode=0; if(getenv("SPECIES_WEIGHT"))wmode=atoi(getenv("SPECIES_WEIGHT"));
    if(wmode==0)for(auto&e:allEdges)e.wt=1.0;
    else if(wmode==2)for(auto&e:allEdges)e.wt=e.wt*e.wt;
    else if(wmode==3)for(auto&e:allEdges)e.wt=sqrt(e.wt);
    // distance matrices
    vector<vector<double>> avgTop(n,vector<double>(n,0)), avgLen(n,vector<double>(n,0));
    for(auto&g:genes){
        auto dt=geneDistance(g,false); auto dl=geneDistance(g,true);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){avgTop[i][j]+=dt[i][j]/genes.size();avgLen[i][j]+=dl[i][j]/genes.size();}
    }
    mt19937 rng(1);
    string seedEnv=getenv("SEED")?getenv("SEED"):"1"; rng.seed((unsigned)stoul(seedEnv));
    vector<STree> starts;
    {
        { string s1=nj(avgTop,labels); STree T(n); T.buildFromNewick(s1,labelMap,labels); starts.push_back(move(T)); }
        { string s2=nj(avgLen,labels); STree T(n); T.buildFromNewick(s2,labelMap,labels); starts.push_back(move(T)); }
    }
    // greedy consensus start
    {
        map<Mask,int> cnt;
        auto cmp=[](const Mask&a,const Mask&b){for(int i=0;i<W;i++)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;};
        map<Mask,int,decltype(cmp)> counts(cmp);
        for(auto&g:genes)for(auto&e:g.edges)counts[e.s]++;
        vector<pair<Mask,int>> splits(counts.begin(),counts.end());
        sort(splits.begin(),splits.end(),[](auto&a,auto&b){return a.second>b.second;});
        // build compatible set greedily
        vector<Mask> chosen;
        auto compatible=[](const Mask&a,const Mask&b){
            Mask ab=a&b, anb=a; // check a subset b, b subset a, or disjoint
            // a&b ==0 or a&b==a or a&b==b
            if((a&b)==a||(a&b)==b)return true;
            Mask z{}; if((a&b)==z)return true; return false;
        };
        for(auto&[m,c]:splits){ bool ok=true; for(auto&x:chosen)if(!compatible(m,x)){ok=false;break;} if(ok)chosen.push_back(m); }
        // build tree from chosen splits + leaves
        Mask full=fullmask(n);
        function<string(Mask)> rec=[&](Mask m)->string{
            if(popcount(m)==1){for(int i=0;i<n;i++)if(getbit(m,i))return labels[i];}
            // find best split subset of m
            int bestc=-1; Mask bestm{};
            for(auto&[sm,c]:splits){ if(popcount(sm)>=2&&popcount(sm)<popcount(m)&&(sm&m)==sm){ if(c>bestc){bestc=c;bestm=sm;} } }
            if(bestc>=0){ return "("+rec(bestm)+","+rec(m^bestm)+")"; }
            // fallback: split off one leaf
            for(int i=0;i<n;i++)if(getbit(m,i)){Mask leaf{};setbit(leaf,i);return "("+rec(leaf)+","+rec(m^leaf)+")";}
            return "";
        };
        string s=rec(full)+";";
        STree T(n); T.buildFromNewick(s,labelMap,labels); starts.push_back(move(T));
    }
    vector<GeneEdge> unEdges=allEdges; for(auto&e:unEdges)e.wt=1.0;
    double bestScore=-1e300; STree best;
    auto consider=[&](STree& T){ double us=fullScore(T,unEdges); if(us>bestScore){bestScore=us;best=T;} };
    for(auto& T:starts){
        double sc=fullScore(T,allEdges);
        sc=hillClimb(T,allEdges,sc,rng,10000,timeLimit,t0);
        consider(T);
    }
    { double sc=fullScore(best,allEdges); sc=leafSearch(best,allEdges,sc,rng,timeLimit,t0,3000); sc=hillClimb(best,allEdges,sc,rng,10000,timeLimit,t0); consider(best); }
    // iterated local search
    int iter=0;
    while(true){
        auto now=chrono::steady_clock::now(); double elapsed=chrono::duration<double>(now-t0).count();
        if(elapsed>timeLimit)break;
        STree T=best;
        int k=1+(rng()%12);
        if(rng()%3==0){
            int m=1+(rng()%4);
            for(int i=0;i<m;i++){ if(rng()%2)subtreeMove(T,rng);else leafMove(T,rng); }
        } else {
            for(int i=0;i<k;i++){
                vector<pair<int,int>> el; for(int u=T.n;u<(int)T.adj.size();u++)for(int v:T.adj[u])if(v>u&&v>=T.n)el.push_back({u,v});
                if(el.empty())break;
                auto [u,v]=el[rng()%el.size()];
                int a=-1,b=-1,c=-1,d=-1;
                for(int x:T.adj[u])if(x!=v){if(a<0)a=x;else b=x;}
                for(int x:T.adj[v])if(x!=u){if(c<0)c=x;else d=x;}
                if(rng()%2)doSwap(T,u,v,b,c);else doSwap(T,u,v,b,d);
            }
        }
        double sc=fullScore(T,allEdges);
        sc=hillClimb(T,allEdges,sc,rng,10000,timeLimit,t0);
        if(rng()%4==0){ sc=leafSearch(T,allEdges,sc,rng,timeLimit,t0,800); sc=hillClimb(T,allEdges,sc,rng,10000,timeLimit,t0); }
        consider(T);
        iter++;
    }
    string out=best.toNewick(labels);
    ofstream fout(outpath); fout<<out<<"\n"; fout.close();
    cerr<<"n="<<n<<" genes="<<genes.size()<<" score="<<bestScore<<" iters="<<iter<<"\n";
    return 0;
}
