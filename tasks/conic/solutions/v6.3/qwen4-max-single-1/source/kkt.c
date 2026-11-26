#include "kkt.h"

/* ---- minimum-degree ordering on symmetric pattern (full CSC, no diag needed) ---- */
typedef struct { idx_t *a; idx_t n,cap; } Adj;

static void adj_add(Adj *A,idx_t v){
    if(A->n==A->cap){A->cap=A->cap?A->cap*2:4;A->a=xrealloc(A->a,A->cap*sizeof(idx_t));}
    A->a[A->n++]=v;
}

static void md_order(idx_t N,const idx_t *Ap,const idx_t *Ai,idx_t *perm){
    Adj *adj=xcalloc(N,sizeof(Adj));
    for(idx_t c=0;c<N;c++)for(idx_t k=Ap[c];k<Ap[c+1];k++){idx_t r=Ai[k];if(r!=c)adj_add(&adj[c],r);}
    idx_t *deg=xmalloc(N*sizeof(idx_t));
    idx_t *nxt=xmalloc(N*sizeof(idx_t)),*prv=xmalloc(N*sizeof(idx_t));
    idx_t *head=xmalloc((N+1)*sizeof(idx_t));
    for(idx_t d=0;d<=N;d++)head[d]=-1;
    char *live=xcalloc(N,1);
    for(idx_t i=0;i<N;i++){deg[i]=adj[i].n;live[i]=1;nxt[i]=head[deg[i]];prv[i]=-1;if(head[deg[i]]>=0)prv[head[deg[i]]]=i;head[deg[i]]=i;}
    idx_t *tag=xcalloc(N,sizeof(idx_t)); idx_t stamp=0;
    idx_t *nbrs=xmalloc(N*sizeof(idx_t));
    idx_t mindeg=0;
    for(idx_t step=0;step<N;step++){
        while(mindeg<=N&&head[mindeg]<0)mindeg++;
        idx_t p=head[mindeg];
        /* unlink p */
        head[mindeg]=nxt[p]; if(nxt[p]>=0)prv[nxt[p]]=-1;
        live[p]=0; perm[step]=p;
        idx_t nn=adj[p].n;
        memcpy(nbrs,adj[p].a,nn*sizeof(idx_t));
        /* for each neighbor i: remove p, merge nbrs */
        for(idx_t ii=0;ii<nn;ii++){
            idx_t i=nbrs[ii];
            if(!live[i])continue;
            /* unlink i from its bucket */
            idx_t d=deg[i];
            if(prv[i]>=0)nxt[prv[i]]=nxt[i]; else head[d]=nxt[i];
            if(nxt[i]>=0)prv[nxt[i]]=prv[i];
            stamp++;
            for(idx_t jj=0;jj<nn;jj++){idx_t j=nbrs[jj];if(j!=i)tag[j]=stamp;}
            idx_t wl=0;
            for(idx_t k=0;k<adj[i].n;k++){
                idx_t v=adj[i].a[k];
                if(v==p)continue;
                if(tag[v]==stamp){tag[v]=-stamp;}
                adj[i].a[wl++]=v;
            }
            for(idx_t jj=0;jj<nn;jj++){idx_t j=nbrs[jj];if(j!=i&&tag[j]==stamp){adj_add(&adj[i],j);tag[j]=0;}}
            adj[i].n=wl;
            /* recompute degree, reinsert */
            idx_t nd=0;
            for(idx_t k=0;k<wl;k++)if(live[adj[i].a[k]])nd++;
            /* compact live only */
            idx_t w2=0; for(idx_t k=0;k<wl;k++)if(live[adj[i].a[k]])adj[i].a[w2++]=adj[i].a[k];
            adj[i].n=w2; nd=w2;
            deg[i]=nd;
            nxt[i]=head[nd];prv[i]=-1;if(head[nd]>=0)prv[head[nd]]=i;head[nd]=i;
            if(nd<mindeg)mindeg=nd;
        }
        adj[p].n=0; free(adj[p].a); adj[p].a=NULL; adj[p].cap=0;
    }
    for(idx_t i=0;i<N;i++)free(adj[i].a);
    free(adj);free(deg);free(nxt);free(prv);free(head);free(live);free(tag);free(nbrs);
}

/* ---- KKT ---- */
static int cmp_trip(const void *a,const void *b){
    const long *x=a,*y=b;
    if(x[0]!=y[0])return x[0]<y[0]?-1:1;
    return x[1]<y[1]?-1:x[1]>y[1]?1:0;
}

void kkt_setup(KKT *K,idx_t N,long nnz,const idx_t *ti,const idx_t *tj,const double *tv){
    memset(K,0,sizeof(*K));
    K->N=N;K->nnz=nnz;K->ti=(idx_t*)ti;K->tj=(idx_t*)tj;K->tv=(double*)tv;
    /* full symmetric adjacency for ordering */
    idx_t *cnt=xcalloc(N,sizeof(idx_t));
    for(long k=0;k<nnz;k++){idx_t i=ti[k],j=tj[k];if(i!=j){cnt[i]++;cnt[j]++;}}
    idx_t *Ap=xmalloc((N+1)*sizeof(idx_t));
    Ap[0]=0;for(idx_t i=0;i<N;i++)Ap[i+1]=Ap[i]+cnt[i];
    idx_t *Ai=xmalloc((Ap[N]?Ap[N]:1)*sizeof(idx_t));
    idx_t *wp=xmalloc(N*sizeof(idx_t)); memcpy(wp,Ap,N*sizeof(idx_t));
    for(long k=0;k<nnz;k++){idx_t i=ti[k],j=tj[k];if(i!=j){Ai[wp[i]++]=j;Ai[wp[j]++]=i;}}
    free(cnt);free(wp);
    K->perm=xmalloc(N*sizeof(idx_t));K->pinv=xmalloc(N*sizeof(idx_t));
    md_order(N,Ap,Ai,K->perm);
    for(idx_t i=0;i<N;i++)K->pinv[K->perm[i]]=i;
    free(Ap);free(Ai);
    /* permuted FULL symmetric CSC (both triangles), diagonal once */
    long fullnnz=0;
    for(long k=0;k<nnz;k++)fullnnz+=(ti[k]==tj[k])?1:2;
    K->fullnnz=fullnnz;
    long (*sortkey)[3]=xmalloc(fullnnz*sizeof(*sortkey));
    long fk=0;
    for(long k=0;k<nnz;k++){
        idx_t a=K->pinv[ti[k]],b=K->pinv[tj[k]];
        sortkey[fk][0]=a;sortkey[fk][1]=b;sortkey[fk][2]=k;fk++;
        if(a!=b){sortkey[fk][0]=b;sortkey[fk][1]=a;sortkey[fk][2]=k;fk++;}
    }
    qsort(sortkey,fullnnz,sizeof(*sortkey),cmp_trip);
    K->Ap=xmalloc((N+1)*sizeof(idx_t));
    K->Ai=xmalloc(fullnnz*sizeof(idx_t));
    K->Ax=xmalloc(fullnnz*sizeof(double));
    K->slot=xmalloc(fullnnz*sizeof(long));
    idx_t cur=0; K->Ap[0]=0;
    for(long k=0;k<fullnnz;k++){
        idx_t c=(idx_t)sortkey[k][0],r=(idx_t)sortkey[k][1];
        while(cur<c){cur++;K->Ap[cur]=k;}
        K->Ai[k]=r;K->slot[k]=sortkey[k][2];
    }
    while(cur<N){cur++;K->Ap[cur]=fullnnz;}
    free(sortkey);
    /* symbolic: true etree (parent) + separate union-find for fast root queries */
    K->etree=xmalloc(N*sizeof(idx_t));
    idx_t *parent=K->etree;
    idx_t *uf=xmalloc(N*sizeof(idx_t));
    for(idx_t i=0;i<N;i++){parent[i]=-1;uf[i]=i;}
    for(idx_t k=0;k<N;k++){
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++){
            idx_t i=K->Ai[kk];
            if(i>=k)continue;
            idx_t r=i;
            while(uf[r]!=r){uf[r]=uf[uf[r]];r=uf[r];}
            if(r!=k){parent[r]=k;uf[r]=k;}
        }
    }
    free(uf);
    /* F (upper) pattern via reach from i<k, then transpose to L (lower) */
    idx_t *mark=xmalloc(N*sizeof(idx_t));
    for(idx_t i=0;i<N;i++)mark[i]=-1;
    idx_t *Fp=xmalloc((N+1)*sizeof(idx_t));
    long total=0;
    for(idx_t k=0;k<N;k++){
        idx_t cnt=0;
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++){
            idx_t i=K->Ai[kk];
            if(i>=k)continue;
            idx_t j=i;
            while(j>=0&&j<k&&mark[j]!=k){mark[j]=k;cnt++;j=parent[j];}
        }
        Fp[k]=cnt;total+=cnt;
    }
    for(idx_t i=0;i<N;i++)mark[i]=-1;
    idx_t *Fi=xmalloc((total?total:1)*sizeof(idx_t));
    idx_t w=0;
    for(idx_t k=0;k<N;k++){
        idx_t st=w;
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++){
            idx_t i=K->Ai[kk];
            if(i>=k)continue;
            idx_t j=i;
            while(j>=0&&j<k&&mark[j]!=k){mark[j]=k;Fi[w++]=j;j=parent[j];}
        }
        Fp[k]=st;
    }
    Fp[N]=w;
    /* transpose F -> L : Lcol[j] = { i : j in Fcol[i] } */
    K->Lp=xmalloc((N+1)*sizeof(idx_t));
    idx_t *cnt2=xcalloc(N,sizeof(idx_t));
    for(idx_t i=0;i<N;i++)for(idx_t t=Fp[i];t<Fp[i+1];t++)cnt2[Fi[t]]++;
    K->Lp[0]=0;for(idx_t j=0;j<N;j++)K->Lp[j+1]=K->Lp[j]+cnt2[j];
    K->Li=xmalloc((total?total:1)*sizeof(idx_t));
    idx_t *wp2=xmalloc(N*sizeof(idx_t));memcpy(wp2,K->Lp,N*sizeof(idx_t));
    for(idx_t i=0;i<N;i++)for(idx_t t=Fp[i];t<Fp[i+1];t++){idx_t j=Fi[t];K->Li[wp2[j]++]=i;}
    /* sort each L column ascending */
    for(idx_t j=0;j<N;j++){
        idx_t st=K->Lp[j],en=K->Lp[j+1];
        for(idx_t a=st+1;a<en;a++){idx_t v=K->Li[a];idx_t bb=a-1;while(bb>=st&&K->Li[bb]>v){K->Li[bb+1]=K->Li[bb];bb--;}K->Li[bb+1]=v;}
    }
    K->Lx=xmalloc((total?total:1)*sizeof(double));
    K->D=xmalloc(N*sizeof(double));
    K->work=xcalloc(N,sizeof(double));
    free(mark);free(Fp);free(Fi);free(cnt2);free(wp2);
    kkt_refill(K,tv);
}

void kkt_refill(KKT *K,const double *tv){
    for(long k=0;k<K->fullnnz;k++)K->Ax[k]=tv[K->slot[k]];
}

int kkt_factor(KKT *K){
    idx_t N=K->N;
    double *y=K->work;
    /* row k of L pattern (= j<k with L[k,j]!=0) via etree reach, reused buffer */
    idx_t *rowpat=xmalloc(N*sizeof(idx_t));
    idx_t *mark=xmalloc(N*sizeof(idx_t));
    for(idx_t i=0;i<N;i++)mark[i]=-1;
    for(idx_t k=0;k<N;k++){
        /* scatter full column k of A */
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++){idx_t i=K->Ai[kk];y[i]=K->Ax[kk];}
        /* enumerate row k of L: walk etree from i<k in col k */
        idx_t top=0;
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++){
            idx_t i=K->Ai[kk];
            if(i>=k)continue;
            idx_t j=i;
            while(j>=0&&j<k&&mark[j]!=k){mark[j]=k;rowpat[top++]=j;j=K->etree[j];}
        }
        /* rowpat is in decreasing-ish order; sort ascending */
        for(idx_t a=1;a<top;a++){idx_t v=rowpat[a];idx_t bb=a-1;while(bb>=0&&rowpat[bb]>v){rowpat[bb+1]=rowpat[bb];bb--;}rowpat[bb+1]=v;}
        for(idx_t t=0;t<top;t++){
            idx_t j=rowpat[t];
            double l=y[j]/K->D[j];
            double dl=l*K->D[j];
            y[k]-=l*dl;
            if(dl!=0.0){
                for(idx_t tt=K->Lp[j];tt<K->Lp[j+1];tt++){
                    idx_t i=K->Li[tt];
                    if(i>=k)break;
                    y[i]-=dl*K->Lx[tt];
                }
            }
            /* store L[k,j] into column j (append; pattern preallocated) */
            /* find slot: column j pattern contains k */
            /* we pre-fill Lx in pattern order; locate via binary search */
            idx_t lo=K->Lp[j],hi=K->Lp[j+1];
            while(lo<hi){idx_t mid=(lo+hi)/2;if(K->Li[mid]<k)lo=mid+1;else hi=mid;}
            K->Lx[lo]=l;
        }
        double dk=y[k];
        K->D[k]=dk;
        y[k]=0;
        for(idx_t kk=K->Ap[k];kk<K->Ap[k+1];kk++)y[K->Ai[kk]]=0;
        for(idx_t t=0;t<top;t++)y[rowpat[t]]=0;
        if(!(fabs(dk)>1e-300)){free(rowpat);free(mark);return -1;}
    }
    free(rowpat);free(mark);
    return 0;
}

void kkt_solve(KKT *K,const double *rhs,double *sol){
    idx_t N=K->N;
    double *y=K->work;
    for(idx_t i=0;i<N;i++)y[i]=rhs[K->perm[i]];
    /* forward: L w = b (column-oriented) */
    for(idx_t j=0;j<N;j++){
        double yj=y[j];
        for(idx_t t=K->Lp[j];t<K->Lp[j+1];t++)y[K->Li[t]]-=K->Lx[t]*yj;
    }
    for(idx_t i=0;i<N;i++)y[i]/=K->D[i];
    /* backward: L^T x = w (row-oriented) */
    for(idx_t j=N-1;j>=0;j--){
        double acc=y[j];
        for(idx_t t=K->Lp[j];t<K->Lp[j+1];t++)acc-=K->Lx[t]*y[K->Li[t]];
        y[j]=acc;
    }
    for(idx_t i=0;i<N;i++)sol[K->perm[i]]=y[i];
}

/* symmetric matvec using triplets: y = M x */
void kkt_matvec(const KKT *K,const double *x,double *y){
    idx_t N=K->N;
    for(idx_t i=0;i<N;i++)y[i]=0;
    for(long k=0;k<K->nnz;k++){
        idx_t i=K->ti[k],j=K->tj[k];double v=K->tv[k];
        if(i==j)y[i]+=v*x[i];
        else {y[i]+=v*x[j];y[j]+=v*x[i];}
    }
}
