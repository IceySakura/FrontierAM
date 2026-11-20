#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
struct DSU {
    std::vector<int> p;
    std::vector<unsigned char> depth;
    DSU(int n):p(n),depth(n) {std::iota(p.begin(),p.end(),0);}
    int root(int x) {
        int j=x;
        while (p[j]!=j) j=p[j];
        while (p[x]!=x) {int k=p[x]; p[x]=j; x=k;}
        return j;
    }
    void join(int a,int b) {
        a=root(a); b=root(b);
        if (a==b) return;
        if (depth[a]<depth[b]) std::swap(a,b);
        p[b]=a;
        if (depth[a]==depth[b]) ++depth[a];
    }
};

std::uint64_t bits(double x) {
    std::uint64_t w;
    std::memcpy(&w,&x,sizeof(w));
    return w;
}
void feed(std::uint64_t* a,std::uint64_t* b,int k,std::uint64_t x) {
    a[k]=(a[k]^x)*UINT64_C(1099511628211);
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    b[k]^=(x^(x>>31))+(b[k]<<7)+(b[k]>>3);
}
}

extern "C" int conic_components(int n,int m,const int* ap,const int* ai,const double* av,
             const int* pp,const int* pi,const double* pv,const double* q,const double* rhs,
             int blocks,const int* starts,const int* stops,const int* row_kind,
             int* varpart,int* rowpart,std::uint64_t* first,std::uint64_t* second) {
    try {
        DSU d(n);
        std::fill(rowpart,rowpart+m,-1);
        for (int j=0;j<n;++j) {
            for (int k=pp[j];k<pp[j+1];++k) d.join(j,pi[k]);
            for (int k=ap[j];k<ap[j+1];++k) {
                int& owner=rowpart[ai[k]];
                if (owner>=0) d.join(j,owner);
                else owner=j;
            }
        }
        for (int k=0;k<blocks;++k) {
            int fst=-1;
            for (int i=starts[k];i<stops[k];++i) if (rowpart[i]>=0) {
                if (fst>=0) d.join(fst,rowpart[i]);
                else fst=rowpart[i];
            }
            std::fill(rowpart+starts[k],rowpart+stops[k],fst);
        }
        std::vector<int> label(n,-1),vloc(n),rloc(m),nc(n),mc(n);
        int num=0;
        for (int j=0;j<n;++j) {
            int& id=label[d.root(j)];
            if (id<0) id=num++;
            varpart[j]=id;
            vloc[j]=nc[id]++;
        }
        if (num<2) return num;
        for (int i=0;i<m;++i) if (rowpart[i]>=0) {
            int k=rowpart[i]=varpart[rowpart[i]];
            rloc[i]=mc[k]++;
        }
        for (int k=0;k<num;++k) {
            first[k]=UINT64_C(14695981039346656037);
            second[k]=UINT64_C(0xa76c9d2bfb1fd345);
            feed(first,second,k,nc[k]);
            feed(first,second,k,mc[k]);
        }
        for (int i=0;i<m;++i) if (rowpart[i]>=0) {
            int k=rowpart[i];
            feed(first,second,k,row_kind[i]);
            feed(first,second,k,bits(rhs[i]));
        }
        for (int j=0;j<n;++j) {
            const int k=varpart[j];
            feed(first,second,k,bits(q[j]));
            feed(first,second,k,ap[j+1]-ap[j]);
            for (int u=ap[j];u<ap[j+1];++u) {
                feed(first,second,k,rloc[ai[u]]);
                feed(first,second,k,bits(av[u]));
            }
            feed(first,second,k,pp[j+1]-pp[j]);
            for (int u=pp[j];u<pp[j+1];++u) {
                feed(first,second,k,vloc[pi[u]]);
                feed(first,second,k,bits(pv[u]));
            }
        }
        return num;
    } catch (...) {return 0;}
}
