#include <bits/stdc++.h>
using namespace std;
struct Bits { uint64_t w[4]; bool operator==(const Bits&o)const{return w[0]==o.w[0]&&w[1]==o.w[1]&&w[2]==o.w[2]&&w[3]==o.w[3];} bool operator!=(const Bits&o)const{return !(*this==o);} };
struct Hash { size_t operator()(const Bits&b)const{ size_t h=b.w[0]; h^=b.w[1]+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); h^=b.w[2]+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); h^=b.w[3]+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2); return h; } };
static inline Bits operator&(const Bits&a,const Bits&b){return {a.w[0]&b.w[0],a.w[1]&b.w[1],a.w[2]&b.w[2],a.w[3]&b.w[3]};}
static inline Bits operator^(const Bits&a,const Bits&b){return {a.w[0]^b.w[0],a.w[1]^b.w[1],a.w[2]^b.w[2],a.w[3]^b.w[3]};}
static inline Bits operator|(const Bits&a,const Bits&b){return {a.w[0]|b.w[0],a.w[1]|b.w[1],a.w[2]|b.w[2],a.w[3]|b.w[3]};}
static inline int pc(const Bits&a){return __builtin_popcountll(a.w[0])+__builtin_popcountll(a.w[1])+__builtin_popcountll(a.w[2])+__builtin_popcountll(a.w[3]);}
static inline void setbit(Bits&a,int i){a.w[i>>6]|=1ULL<<(i&63);}
static Bits full, ing;
struct Tree { vector<vector<pair<int,double>>> adj; vector<double> sup; vector<int> leaf,taxon; int n=250; int newNode(){adj.push_back({});sup.push_back(1);leaf.push_back(-1);taxon.push_back(-1);return adj.size()-1;} };
void suppress(Tree&t){bool ch=true;while(ch){ch=false;for(int v=0;v<(int)t.adj.size();v++){if(t.taxon[v]>=0||t.adj[v].size()!=2)continue;int a=t.adj[v][0].first,b=t.adj[v][1].first;double la=t.adj[v][0].second,lb=t.adj[v][1].second;auto rem=[&](int x,int y){auto&vec=t.adj[x];for(size_t k=0;k<vec.size();k++)if(vec[k].first==y){vec[k]=vec.back();vec.pop_back();break;}};rem(a,v);rem(b,v);t.adj[a].push_back({b,la+lb});t.adj[b].push_back({a,la+lb});t.adj[v].clear();ch=true;break;}}}
Tree parse(const string&s){Tree t;size_t i=0;t.leaf.assign(250,-1);function<pair<int,double>()> sub=[&]()->pair<int,double>{int v;double len=0;if(i<s.size()&&s[i]=='('){i++;v=t.newNode();while(true){auto[c,l]=sub();t.adj[v].push_back({c,l});t.adj[c].push_back({v,l});if(i<s.size()&&s[i]==','){i++;continue;}break;}if(i<s.size()&&s[i]==')')i++;string lab;while(i<s.size()&&s[i]!=':'&&s[i]!=','&&s[i]!=')'&&s[i]!=';'&&!isspace((unsigned char)s[i]))lab+=s[i++];if(!lab.empty()){char*end;double sv=strtod(lab.c_str(),&end);if(end!=lab.c_str())t.sup[v]=sv;}}else{string lab;while(i<s.size()&&s[i]!=':'&&s[i]!=','&&s[i]!=')'&&s[i]!=';'&&!isspace((unsigned char)s[i]))lab+=s[i++];v=t.newNode();int id=stoi(lab);t.leaf[id]=v;t.taxon[v]=id;}if(i<s.size()&&s[i]==':'){i++;string num;while(i<s.size()&&s[i]!=','&&s[i]!=')'&&s[i]!=';'&&!isspace((unsigned char)s[i]))num+=s[i++];len=strtod(num.c_str(),nullptr);}return{v,len};};auto[r,rl]=sub();suppress(t);return t;}
struct Rooted { vector<int> parent,order; vector<Bits> below; vector<vector<int>> children; };
Rooted rootTree(const Tree&t){Rooted r;int root=t.leaf[0];r.parent.assign(t.adj.size(),-1);r.children.assign(t.adj.size(),{});r.order.push_back(root);r.parent[root]=-2;for(size_t k=0;k<r.order.size();k++){int v=r.order[k];for(auto[w,l]:t.adj[v])if(w!=r.parent[v]){r.parent[w]=v;r.children[v].push_back(w);r.order.push_back(w);}}r.below.assign(t.adj.size(),Bits{0,0,0,0});for(int k=r.order.size()-1;k>=0;k--){int v=r.order[k];if(t.taxon[v]>=0)setbit(r.below[v],t.taxon[v]);else for(int c:r.children[v])r.below[v]=r.below[v]|r.below[c];}return r;}
struct GNode { Bits X,Y,Z; int szZ; double w; };
vector<GNode> geneNodes(const Tree&t,const Rooted&r,int mode,double alpha,double betaOverride=0){vector<GNode> ns;for(int v:r.order){if(t.taxon[v]>=0)continue;if(r.children[v].size()!=2)continue;Bits X=r.below[r.children[v][0]],Y=r.below[r.children[v][1]];Bits Z=full^(X|Y);double w=1;if(mode==1)w=pow(t.sup[v],alpha);else if(mode==2)w=t.sup[v];
        else if(mode==21)w=0.5+0.5*pow(t.sup[v],2);
        else if(mode>=32&&mode<=34){double len=0;for(auto[x,l]:t.adj[v])if(x==r.parent[v])len=l;double beta=betaOverride>0?betaOverride:0.01;double f=len/(len+beta);double sf=mode==32?0.5+0.5*f:mode==33?0.25+0.75*f:sqrt(f);w=pow(t.sup[v],alpha)*sf;}
        else if(mode==22)w=0.25+0.75*pow(t.sup[v],2);
        else if(mode==43){double ms=0;int cn=0;for(int x:r.order){if(t.taxon[x]>=0||r.parent[x]<0||t.taxon[r.parent[x]]>=0)continue;ms+=t.sup[x];cn++;}ms=cn?ms/cn:1;w=pow(t.sup[v],alpha)*ms*ms;}
        else if(mode==35)w=pow(t.sup[v],3);
        else if(mode==36)w=pow(t.sup[v],2)/(1.1-t.sup[v]);
        else if(mode==37)w=pow(t.sup[v],4);
        else if(mode==39||mode==40){double len=0;for(auto[x,l]:t.adj[v])if(x==r.parent[v])len=l;double beta=betaOverride>0?betaOverride:0.01;w=pow(t.sup[v],alpha)*(mode==39?exp(-len/beta):1.0/(1+len/beta));}
        else if((mode>=4&&mode<=20)||(mode>=44&&mode<=52)){double len=0;for(auto[x,l]:t.adj[v])if(x==r.parent[v])len=l;double beta=betaOverride>0?betaOverride:(mode==4?0.0:mode==5?0.01:mode==6?0.005:mode==7?0.02:mode==8?0.05:mode==9?0.008:mode==10?0.012:mode==44?0.002:mode==45?0.003:mode==46?0.004:mode==47?0.006:mode==48?0.007:mode==51?0.0065:mode==52?0.0075:0.015);w=pow(t.sup[v],alpha)*(mode==4?len:len/(len+beta));if(mode==19&&t.sup[v]<0.5)w=0;if(mode==20&&t.sup[v]<0.7)w=0;}
        ns.push_back({X,Y,Z,pc(Z),w});}return ns;}
static inline long long C2(long long x){return x*(x-1)/2;}
double score2Idx(int ia,int ib,int sa,int sb,int sc,const vector<GNode>&all,const vector<int>&szX,const vector<int>&szY,const vector<uint8_t>&AX,const vector<uint8_t>&AY,int N,int normMode=0){
    double total=0; const uint8_t*axp=&AX[ia*N]; const uint8_t*ayp=&AY[ia*N]; const uint8_t*bxp=&AX[ib*N]; const uint8_t*byp=&AY[ib*N];
    for(int n=0;n<N;n++){
        int ax=axp[n], ay=ayp[n], bx=bxp[n], by=byp[n];
        int az=sa-ax-ay, bz=sb-bx-by;
        int cx=szX[n]-ax-bx, cy=szY[n]-ay-by, cz=sc-cx-cy;
        long long q211=C2(ax)*(by*cz+bz*cy)+C2(ay)*(bx*cz+bz*cx)+C2(az)*(bx*cy+by*cx);
        long long q121=C2(bx)*(ay*cz+az*cy)+C2(by)*(ax*cz+az*cx)+C2(bz)*(ax*cy+ay*cx);
        long long q112=C2(cx)*(ay*bz+az*by)+C2(cy)*(ax*bz+az*bx)+C2(cz)*(ax*by+ay*bx);
        total+=all[n].w*(q211+q121+q112);
    }
    if(normMode==1){double den=C2(sa)*sb*sc+C2(sb)*sa*sc+C2(sc)*sa*sb; if(den>0)total/=den;}
    else if(normMode==2){double den=C2(sa)*sb*sc+C2(sb)*sa*sc+C2(sc)*sa*sb; if(den>0)total/=sqrt(den);}
    return total;
}
// NJ for initial clusters
vector<vector<double>> avgDist(const vector<Tree>&trees){int N=250;vector<vector<double>>D(N,vector<double>(N,0)),D1(N,vector<double>(N,0)),D2(N,vector<double>(N,0));for(auto&t:trees){for(int a=0;a<N;a++){int s=t.leaf[a];vector<double>d(t.adj.size(),-1),d1(t.adj.size(),-1),d2(t.adj.size(),-1);queue<int>q;d[s]=d1[s]=d2[s]=0;q.push(s);while(!q.empty()){int v=q.front();q.pop();for(auto[w,l]:t.adj[v])if(d[w]<0){d[w]=d[v]+1;d1[w]=d1[v]+l;d2[w]=d2[v]+t.sup[w];q.push(w);}}for(int b=0;b<N;b++){D[a][b]+=d[t.leaf[b]]/trees.size();D1[a][b]+=d1[t.leaf[b]]/trees.size();D2[a][b]+=d2[t.leaf[b]]/trees.size();}}}auto norm=[&](vector<vector<double>>&M){double s=0;int c=0;for(int i=0;i<N;i++)for(int j=0;j<N;j++)if(i!=j){s+=M[i][j];c++;}double m=s/c;for(int i=0;i<N;i++)for(int j=0;j<N;j++)M[i][j]/=m;};norm(D);norm(D1);norm(D2);for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i][j]=0.35*D[i][j]+0.6*D1[i][j]+0.05*D2[i][j];return D;}
string nj(vector<vector<double>>D){int n=D.size();vector<int>active(n);iota(active.begin(),active.end(),0);vector<string>node(n);for(int i=0;i<n;i++)node[i]=to_string(i);auto d=D;while(active.size()>2){int k=active.size();vector<double>r(k,0);for(int i=0;i<k;i++)for(int j=0;j<k;j++)if(i!=j)r[i]+=d[active[i]][active[j]];double best=1e300;int bi=-1,bj=-1;for(int i=0;i<k;i++)for(int j=i+1;j<k;j++){double q=(k-2)*d[active[i]][active[j]]-r[i]-r[j];if(q<best){best=q;bi=i;bj=j;}}int a=active[bi],b=active[bj];double da=0.5*d[a][b]+(r[bi]-r[bj])/(2*(k-2));double db=d[a][b]-da;int nw=node.size();node.push_back("("+node[a]+","+node[b]+")");d.push_back(vector<double>(nw+1,0));for(auto&row:d)row.resize(nw+1,0);active.erase(active.begin()+bj);active.erase(active.begin()+bi);for(int h:active){double v=0.5*(d[a][h]+d[b][h]-d[a][b]);d[nw][h]=d[h][nw]=v;}active.push_back(nw);}if(active.size()==2)return "("+node[active[0]]+","+node[active[1]]+");";return node[active[0]]+";";}
void addClusters(Tree&t,vector<Bits>&cl){Rooted r=rootTree(t);for(int v:r.order){if(t.taxon[v]>=0)continue;Bits s=r.below[v];if(pc(s)>=2&&pc(s)<=248)cl.push_back(s);}}

vector<vector<double>> avgDistType(const vector<Tree>&trees,int type){int N=250;vector<vector<double>>D(N,vector<double>(N,0));for(auto&t:trees){for(int a=0;a<N;a++){int st=t.leaf[a];vector<double>d(t.adj.size(),-1);queue<int>q;d[st]=0;q.push(st);while(!q.empty()){int v=q.front();q.pop();for(auto[w,l]:t.adj[v])if(d[w]<0){d[w]=d[v]+(type==0?1.0:type==1?l:pow(t.sup[w],2.0));q.push(w);}}for(int b=0;b<N;b++)D[a][b]+=d[t.leaf[b]]/trees.size();}}return D;}
Tree contractTree(Tree t,double thresh){Rooted r=rootTree(t);for(int k=r.order.size()-1;k>=0;k--){int v=r.order[k];if(t.taxon[v]>=0||v==t.leaf[0])continue;if(t.sup[v]>=thresh)continue;int p=r.parent[v];if(p<0)continue;auto rem=[&](int x,int y){auto&vec=t.adj[x];for(size_t j=0;j<vec.size();j++)if(vec[j].first==y){vec[j]=vec.back();vec.pop_back();break;}};vector<pair<int,double>> ch;for(auto[w,l]:t.adj[v])if(w!=p)ch.push_back({w,l});rem(p,v);for(auto[w,l]:ch){rem(w,v);t.adj[p].push_back({w,l});t.adj[w].push_back({p,l});}t.adj[v].clear();}return t;}
void addContracted(const vector<Tree>&trees,vector<Bits>&cl,double thresh){for(auto&t:trees){Tree c=contractTree(t,thresh);Rooted r=rootTree(c);for(int v:r.order){if(c.taxon[v]>=0)continue;Bits s=r.below[v];int sz=pc(s);if(sz>=2&&sz<=248)cl.push_back(s);for(int ch:r.children[v]){Bits cs=r.below[ch];int sz2=pc(cs);if(sz2>=2&&sz2<=248)cl.push_back(cs);}}}}

void addContractedSubsets(const vector<Tree>&trees,vector<Bits>&cl,double thresh){for(auto&t:trees){Tree c=contractTree(t,thresh);Rooted r=rootTree(c);for(int v:r.order){if(c.taxon[v]>=0)continue;auto&ch=r.children[v];int k=ch.size();if(k<3||k>8)continue;vector<Bits>cs;for(int x:ch)cs.push_back(r.below[x]);for(int mask=1;mask<(1<<k)-1;mask++){Bits u{0,0,0,0};for(int i=0;i<k;i++)if(mask>>i&1)u=u|cs[i];int sz=pc(u);if(sz>=2&&sz<=248)cl.push_back(u);}}}}

void addContractedPairs(const vector<Tree>&trees,vector<Bits>&cl,double thresh){for(auto&t:trees){Tree c=contractTree(t,thresh);Rooted r=rootTree(c);for(int v:r.order){if(c.taxon[v]>=0)continue;auto&ch=r.children[v];int k=ch.size();if(k<3)continue;vector<Bits>cs;for(int x:ch)cs.push_back(r.below[x]);for(int i=0;i<k;i++)for(int j=i+1;j<k;j++){Bits u=cs[i]|cs[j];int sz=pc(u);if(sz>=2&&sz<=248)cl.push_back(u);}}}}
int main(int argc,char**argv){
    full=Bits{~0ULL,~0ULL,~0ULL,(1ULL<<58)-1}; ing=full; ing.w[0]&=~1ULL;
    string inp=argv[1],out=argv[2]; int mode=argc>3?atoi(argv[3]):0; double alpha=argc>4?atof(argv[4]):1;
    ifstream fin(inp);string line;vector<Tree>trees;while(getline(fin,line)){if(!line.empty())trees.push_back(parse(line));}
    double cth=argc>7?atof(argv[7]):0.0;
    vector<Tree> gtrees;
    if(cth>0){for(auto&t:trees)gtrees.push_back(contractTree(t,cth));}
    else gtrees=trees;
    double betaOverride=0;
    if((mode>=12&&mode<=20)||(mode>=32&&mode<=40)||mode==49||mode==50){vector<double>lens;for(auto&t:trees){Rooted r=rootTree(t);for(int v:r.order){if(t.taxon[v]>=0||r.parent[v]<0||t.taxon[r.parent[v]]>=0)continue;for(auto[x,l]:t.adj[v])if(x==r.parent[v])lens.push_back(l);}}
      sort(lens.begin(),lens.end());double med=lens[lens.size()/2];double mean=accumulate(lens.begin(),lens.end(),0.0)/lens.size();
      betaOverride=(mode==12||mode==19||mode==20||(mode>=32&&mode<=40))?med:mode==49?0.6*med:mode==50?0.7*med:mode==13?0.5*mean:mode==14?0.3*mean:mode==15?0.8*med:mode==16?1.2*med:mode==17?0.5*med:1.5*med;}
    vector<vector<GNode>>genes;for(auto&t:gtrees)genes.push_back(geneNodes(t,rootTree(t),mode,alpha,betaOverride));
    vector<Bits>cl;
    for(int k=1;k<250;k++){Bits b{0,0,0,0};setbit(b,k);cl.push_back(b);}
    double clth=argc>8?atof(argv[8]):0.0;
    int addcomp=argc>9?atoi(argv[9]):0;
    int ignoreOut=argc>10?atoi(argv[10]):0;
    if(ignoreOut)full=ing;
    for(auto&t:trees){Rooted r=rootTree(t);for(int v:r.order){if(t.taxon[v]>=0)continue;for(int c:r.children[v]){if(t.sup[c]<clth)continue;Bits s=r.below[c];int sz=pc(s);if(sz>=2&&sz<=248){cl.push_back(s);if(addcomp){Bits comp=ing^s;int sz2=pc(comp);if(sz2>=2&&sz2<=248)cl.push_back(comp);}}}}}
    auto D=avgDist(trees); string init=nj(D); Tree it=parse(init); addClusters(it,cl); cl.push_back(ing);
    int expand=argc>5?atoi(argv[5]):0;
    if(expand==1||expand==5){auto D0=avgDistType(trees,0);Tree t0=parse(nj(D0));addClusters(t0,cl);}
    if(expand==1||expand==6){auto D1=avgDistType(trees,1);Tree t1=parse(nj(D1));addClusters(t1,cl);}
    if(expand>=2&&expand<=4){addContracted(trees,cl,0.5);}
    if(expand>=3&&expand<=4){addContracted(trees,cl,0.7);}
    if(expand>=4&&expand<=4){addContracted(trees,cl,0.3);}
    if(expand==26){addContracted(trees,cl,0.5);addContracted(trees,cl,0.7);}
    if(expand==28){addContractedSubsets(trees,cl,0.5);}
    if(expand==30){addContracted(trees,cl,0.001);}
    if(expand==38){for(auto&t:trees){vector<Tree>one{t};auto Dg=avgDistType(one,0);Tree tg=parse(nj(Dg));addClusters(tg,cl);}}
    if(expand==41){addContractedPairs(trees,cl,0.5);}
    if(expand==42){addContractedPairs(trees,cl,0.7);}
    if(expand==31){addContracted(trees,cl,0.01);}
    if(expand==29){addContractedSubsets(trees,cl,0.7);}
    if((expand>=8&&expand<=14)||(expand>=15&&expand<=42)){auto D0=avgDistType(trees,0);Tree t0=parse(nj(D0));addClusters(t0,cl);}
    if(expand==14){for(int o=0;o<(int)trees.size();o++){vector<Tree>bs;for(int i=0;i<(int)trees.size();i++)if(i!=o)bs.push_back(trees[i]);auto Db=avgDistType(bs,0);Tree tb=parse(nj(Db));addClusters(tb,cl);}}
    if((expand>=7&&expand<=12)||(expand>=15&&expand<=19)||(expand>=23&&expand<=42)||expand==22){mt19937 rng(argc>12?atoi(argv[12]):1);int nb=(expand==9?20:expand==10?50:expand==15?5:expand==16?8:expand==17?12:expand==18?15:expand==23?3:expand==24?4:expand==25?6:expand==26?5:expand==27?5:expand==28?5:expand==29?5:expand==30?5:expand==31?5:expand==38?5:expand==41?5:expand==42?5:10);for(int b=0;b<nb;b++){vector<Tree>bs;for(int i=0;i<(int)trees.size();i++)bs.push_back(trees[rng()%trees.size()]);int bt=expand==11?2:expand==12?1:expand==19?3:0;
            auto Db=bt==2?avgDist(bs):avgDistType(bs,bt);Tree tb=parse(nj(Db));addClusters(tb,cl);}}
    if(expand==22){auto D1=avgDistType(trees,1);Tree t1=parse(nj(D1));addClusters(t1,cl);}
    if(expand==27){auto D3=avgDistType(trees,3);Tree t3=parse(nj(D3));addClusters(t3,cl);}
    if(expand==20||expand==21){mt19937 rng(argc>12?atoi(argv[12]):1);int nb=expand==20?5:10;
      for(int b=0;b<nb;b++){vector<Tree>bs;for(int i=0;i<(int)trees.size();i++)bs.push_back(trees[rng()%trees.size()]);auto Db=avgDistType(bs,0);Tree tb=parse(nj(Db));addClusters(tb,cl);}
      for(int b=0;b<nb;b++){vector<Tree>bs;for(int i=0;i<(int)trees.size();i++)bs.push_back(trees[rng()%trees.size()]);auto Db=avgDistType(bs,3);Tree tb=parse(nj(Db));addClusters(tb,cl);}}
    if(addcomp==2){auto addCompTree=[&](Tree&t){Rooted r=rootTree(t);for(int v:r.order){if(t.taxon[v]>=0)continue;Bits s=r.below[v];int sz=pc(s);if(sz>=2&&sz<=248){Bits comp=ing^s;int sz2=pc(comp);if(sz2>=2&&sz2<=248)cl.push_back(comp);}}};addCompTree(it);auto D0=avgDistType(trees,0);Tree t0=parse(nj(D0));addCompTree(t0);}
    if(argc>6){ifstream extra(argv[6]);string el;while(getline(extra,el)){if(!el.empty()){Tree et=parse(el);addClusters(et,cl);}}}
    sort(cl.begin(),cl.end(),[](const Bits&a,const Bits&b){if(a.w[0]!=b.w[0])return a.w[0]<b.w[0];if(a.w[1]!=b.w[1])return a.w[1]<b.w[1];if(a.w[2]!=b.w[2])return a.w[2]<b.w[2];return a.w[3]<b.w[3];});
    cl.erase(unique(cl.begin(),cl.end()),cl.end());
    sort(cl.begin(),cl.end(),[](const Bits&a,const Bits&b){return pc(a)<pc(b);});
    unordered_map<Bits,int,Hash> idx; for(int i=0;i<(int)cl.size();i++)idx[cl[i]]=i;
    int K=cl.size(); vector<double>dp(K,0); vector<int>ba(K,-1),bb(K,-1);
    vector<int>sz(K);for(int i=0;i<K;i++)sz[i]=pc(cl[i]);
    vector<GNode> all; vector<int> szX,szY;
    for(auto&gn:genes)for(auto&n:gn){all.push_back(n);szX.push_back(pc(n.X));szY.push_back(pc(n.Y));}
    unordered_map<Bits,int,Hash> freq;
    for(auto&t:trees){Rooted r=rootTree(t);for(int v:r.order){if(t.taxon[v]>=0)continue;for(int c:r.children[v]){Bits s=r.below[c];if(pc(s)>=2&&pc(s)<=248)freq[s]++;}}}
    int N=all.size(); vector<uint8_t> AX((size_t)K*N),AY((size_t)K*N);
    for(int i=0;i<K;i++){for(int n=0;n<N;n++){AX[(size_t)i*N+n]=pc(cl[i]&all[n].X);AY[(size_t)i*N+n]=pc(cl[i]&all[n].Y);}}
    // singletons implicit not in cl; dp for size2 splits singletons score0
    for(int i=0;i<K;i++){
        Bits X=cl[i]; int sx=sz[i];
        if(sx==1){int a=-1;for(int k=0;k<250;k++)if((X.w[k>>6]>>(k&63))&1){a=k;break;} dp[i]=0;ba[i]=-2-a;bb[i]=-1;continue;}
        double best=-1; int ca=-1,cb=-1;
        for(int j=0;j<i;j++){
            Bits A=cl[j]; if(sz[j]>=sx)continue;
            if((A&X)!=A)continue;
            Bits B=X^A; auto it=idx.find(B); if(it==idx.end())continue;
            int k=it->second; if(k>=i)continue;
            Bits C=full^X; int sc=pc(C);
            int normMode=argc>11?atoi(argv[11]):0; double sc2=score2Idx(j,k,sz[j],sz[k],sc,all,szX,szY,AX,AY,N,normMode);
            double bonus=0; int bm=argc>11?atoi(argv[11]):0; if(bm>0){auto itf=freq.find(cl[i]); if(itf!=freq.end()){ if(bm<100)bonus=bm*10000.0*itf->second; else if(sz[i]==2)bonus=(bm-100)*100000.0*itf->second; }}
            double val=dp[j]+dp[k]+sc2+bonus;
            if(val>best){best=val;ca=j;cb=k;}
        }
        if(ca<0){ // fallback: split off smallest taxon
            int a=-1;for(int k=0;k<250;k++)if((X.w[k>>6]>>(k&63))&1){a=k;break;}
            Bits A{0,0,0,0};setbit(A,a);Bits B=X^A;
            // B may not be in cl; add? skip by creating singleton split not in DP; use recursive fallback later
            dp[i]=0;ba[i]=-2-a;bb[i]=-3; // marker
        } else {dp[i]=best;ba[i]=ca;bb[i]=cb;}
    }
    int root=idx[ing];
    function<string(int)> rec=[&](int i)->string{
        if(ba[i]<-1){int a=-2-ba[i]; if(bb[i]==-3){
                Bits X=cl[i]; vector<int>tax;for(int k=0;k<250;k++)if((X.w[k>>6]>>(k&63))&1)tax.push_back(k);
                string s=to_string(tax[0]); for(int j=1;j<(int)tax.size();j++)s="("+s+","+to_string(tax[j])+")"; return s;
            } if(bb[i]<-1){int b=-2-bb[i];return "("+to_string(a)+","+to_string(b)+")";} return to_string(a);}
        if(ba[i]<0){int a=-2-ba[i],b=-2-bb[i];return "("+to_string(a)+","+to_string(b)+")";}
        return "("+rec(ba[i])+","+rec(bb[i])+")";
    };
    string res="(0,"+rec(root)+");";
    ofstream fout(out);fout<<res<<"\n";
    return 0;
}
