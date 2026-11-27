/* solver.c -- species tree inference from a set of estimated gene trees.
 *
 * Objective (the ASTRAL objective):   score(T) = sum_g |Q(T) & Q(g)|,
 * the number of (gene tree, quartet) pairs where the species tree T and the
 * gene tree g induce the same topology on those four taxa.
 *
 * That quantity is not additive over the branches of T (a quartet whose central
 * path spans several branches of T would be counted several times), so it is
 * optimised by local search with *exact* move deltas.
 *
 * A nearest-neighbour interchange around the internal edge whose four incident
 * components are A,B,C,D changes the induced topology only for quartets having
 * exactly one leaf in each of A,B,C,D.  With
 *
 *   N(P,Q|R,S) = sum_g #{ (p,q,r,s) in PxQxRxS : g displays pq|rs }
 *
 *   delta(swap B,C) = N(A,C|B,D) - N(A,B|C,D)
 *   delta(swap B,D) = N(A,D|B,C) - N(A,B|C,D)
 *
 * N(P,Q|R,S) is evaluated in one pass over the internal nodes of all gene
 * trees: for an internal node v of g whose three incident branches have leaf
 * sets L1,L2,L3, the quartets pq|rs whose p/q junction is v are exactly those
 * with p,q in two different branches and r,s together in the third:
 *
 *   (p1 q2 + p2 q1) r3 s3 + (p1 q3 + p3 q1) r2 s2 + (p2 q3 + p3 q2) r1 s1
 *
 * with p_i = |P & Li| etc.  Only six bitset intersections are needed per node
 * because L1,L2,L3 partition the taxon set (and so do P,Q,R,S).  Term i is
 * weighted by the weight of the gene-tree edge leaving v towards branch i, so
 * support/branch-length weighting is possible (all weights 1 = plain ASTRAL).
 *
 * Search: ASTRID seed (average internode distance over gene trees + neighbour
 * joining), then NNI hill climbing plus a ratchet restart loop.  Scores are
 * tracked relative to the seed tree, which is all local search needs.
 *
 * Single threaded; libc/libm only.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

typedef uint64_t u64;

#define MAXN 512
#define MAXNODES (2*MAXN+8)
static int NW;                   /* words per bitset */
static int NTAX;                 /* number of taxa   */
static u64 ALLW[MAXN/64+2];

/* ------------------------------------------------------------------- timing */
static double T0, T_LIMIT;
static double DEADLINE = 40.0;
static double EPS = 1e-3;   /* ignore score improvements below this (float noise floor) */
static double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}
static double elapsed(void){ return now_sec()-T0; }

/* ---------------------------------------------------------------------- rng */
static u64 rng_s = 88172645463325252ULL;
static inline u64 rnd64(void){
    u64 z = (rng_s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z>>30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z>>27)) * 0x94D049BB133111EBULL;
    return z ^ (z>>31);
}
static inline int rndint(int m){ return m>0 ? (int)(rnd64()%(u64)m) : 0; }

/* ------------------------------------------------------------------ bitsets */
static inline void bs_copy(u64*a,const u64*b){ for(int i=0;i<NW;i++) a[i]=b[i]; }
static inline void bs_comp(u64*a,const u64*b){ for(int i=0;i<NW;i++) a[i]=ALLW[i]&~b[i]; }
static inline int bs_pop(const u64*a){ int c=0; for(int i=0;i<NW;i++) c+=__builtin_popcountll(a[i]); return c; }
static inline int bs_test(const u64*a,int i){ return (int)((a[i>>6]>>(i&63))&1ULL); }
static inline void bs_set(u64*a,int i){ a[i>>6] |= ((u64)1)<<(i&63); }
static inline void bs_zero(u64*a){ for(int i=0;i<NW;i++) a[i]=0; }
static inline void bs_canon(u64*a){ if(bs_test(a,0)) bs_comp(a,a); }

/* --------------------------------------------------------------------- tree */
typedef struct {
    int nn;
    int deg[MAXNODES];
    int adj[MAXNODES][3];
    int leaf[MAXNODES];
    int root;
    u64 *clade;
    int *csz;
    int *par;
    int *buf;
} Tree;

static void tree_alloc(Tree*t,int nn){
    t->nn=nn;
    t->clade=(u64*)calloc((size_t)nn*NW,sizeof(u64));
    t->csz  =(int*)calloc(nn,sizeof(int));
    t->par  =(int*)malloc(nn*sizeof(int));
    t->buf  =(int*)malloc(4*(size_t)nn*sizeof(int));
    for(int i=0;i<nn;i++){ t->deg[i]=0; t->leaf[i]=-1; t->par[i]=-1;
        t->adj[i][0]=t->adj[i][1]=t->adj[i][2]=-1; }
}
static void tree_free(Tree*t){ free(t->clade); free(t->csz); free(t->par); free(t->buf); t->clade=NULL; }
static inline void tree_link(Tree*t,int a,int b){
    if(t->deg[a]>=3||t->deg[b]>=3){ fprintf(stderr,"degree overflow\n"); exit(3); }
    t->adj[a][t->deg[a]++]=b; t->adj[b][t->deg[b]++]=a;
}
static int tree_pick_root(Tree*t){
    for(int i=0;i<t->nn;i++) if(t->deg[i]==3 && t->leaf[i]<0){ t->root=i; return i; }
    for(int i=0;i<t->nn;i++) if(t->deg[i]==3){ t->root=i; return i; }
    t->root=0; return 0;
}
static void tree_root_compute(Tree*t){
    int r=t->root, nn=t->nn;
    int *st=t->buf, *rev=t->buf+2*nn;
    int sp=0,no=0;
    t->par[r]=-1; st[sp++]=r;
    while(sp>0){
        int v=st[--sp];
        if(no>=nn||sp>=2*nn) break;          /* defensive */
        rev[no++]=v;
        for(int k=0;k<t->deg[v];k++){
            int w=t->adj[v][k];
            if(w==t->par[v]) continue;
            t->par[w]=v; st[sp++]=w;
        }
    }
    for(int i=no-1;i>=0;i--){
        int v=rev[i];
        u64 *cv=t->clade+(size_t)v*NW;
        bs_zero(cv);
        int sz=(t->leaf[v]>=0)?1:0;
        if(sz) bs_set(cv,t->leaf[v]);
        for(int k=0;k<t->deg[v];k++){
            int w=t->adj[v][k];
            if(w==t->par[v]) continue;
            const u64 *cw=t->clade+(size_t)w*NW;
            for(int q=0;q<NW;q++) cv[q]|=cw[q];
            sz+=t->csz[w];
        }
        t->csz[v]=sz;
    }
}
static void tree_copy(const Tree*src,Tree*dst){
    memcpy(dst->deg,src->deg,sizeof(int)*dst->nn);
    memcpy(dst->adj,src->adj,sizeof(int)*dst->nn*3);
    memcpy(dst->leaf,src->leaf,sizeof(int)*dst->nn);
    dst->root=src->root;
}
static int tree_ok(Tree*t){
    int nn=t->nn; long sd=0;
    for(int i=0;i<nn;i++){
        sd+=t->deg[i];
        if(t->leaf[i]>=0){ if(t->deg[i]!=1) return 0; } else if(t->deg[i]!=3) return 0;
        for(int k=0;k<t->deg[i];k++){
            int w=t->adj[i][k];
            if(w<0||w>=nn||w==i) return 0;
            int ok=0; for(int j=0;j<t->deg[w];j++) if(t->adj[w][j]==i) ok=1;
            if(!ok) return 0;
        }
    }
    if(sd!=2L*(nn-1)) return 0;
    int *vis=(int*)calloc(nn,sizeof(int)); int *q=(int*)malloc(nn*sizeof(int));
    int qh=0,qt=0,cnt=0; q[qt++]=t->root; vis[t->root]=1;
    while(qh<qt){ int v=q[qh++]; cnt++;
        for(int k=0;k<t->deg[v];k++){ int w=t->adj[v][k]; if(vis[w]) continue; vis[w]=1; q[qt++]=w; } }
    free(vis); free(q);
    return cnt==nn;
}

/* --------------------------------------------------------- taxon name table */
static char **TNAMES;
static int NNAME, TNAMESCAP;
static u64 *NAMEH; static int *NAMEI; static int NAMECAP;
static void names_init(int cap){
    NAMECAP=1; while(NAMECAP<cap*4) NAMECAP<<=1;
    if(NAMECAP<(1<<16)) NAMECAP=1<<16;   /* keep open addressing far from saturation */
    TNAMESCAP=cap; TNAMES=(char**)malloc((size_t)cap*sizeof(char*));
    NAMEH=(u64*)malloc(NAMECAP*sizeof(u64));
    NAMEI=(int*)malloc(NAMECAP*sizeof(int));
    memset(NAMEI,0xff,NAMECAP*sizeof(int));
    NNAME=0;
}
static u64 str_hash(const char*s){ u64 h=1469598103934665603ULL;
    for(;*s;s++){ h^=(unsigned char)*s; h*=1099511628211ULL; } return h; }
static int name_intern(const char*s){
    u64 h=str_hash(s); u64 m=(u64)NAMECAP-1; u64 p=h&m;
    while(NAMEI[p]>=0){ if(NAMEH[p]==h && strcmp(TNAMES[NAMEI[p]],s)==0) return NAMEI[p]; p=(p+1)&m; }
    int id=NNAME++;
    if(id>=TNAMESCAP){ TNAMESCAP*=2; TNAMES=(char**)realloc(TNAMES,(size_t)TNAMESCAP*sizeof(char*)); }
    TNAMES[id]=(char*)malloc(strlen(s)+1); memcpy(TNAMES[id],s,strlen(s)+1);
    NAMEH[p]=h; NAMEI[p]=id; return id;
}

/* ------------------------------------------------------------------- newick */
typedef struct { int n; int *nch; int *ch; int *leaf; double *bl; double *sup; } PTree;
static const char *P; static char TOKB[512]; static int TOKT;
static void tok_next(void){
    while(*P){
        char c=*P;
        if(c==' '||c=='\t'||c=='\r'||c=='\n'){ P++; continue; }
        if(c=='['){ while(*P && *P!=']') P++; if(*P) P++; continue; }
        if(c=='('){ P++; TOKT=1; return; }
        if(c==')'){ P++; TOKT=2; return; }
        if(c==','){ P++; TOKT=3; return; }
        if(c==':'){ P++; TOKT=4; return; }
        if(c==';'){ P++; TOKT=5; return; }
        if(c=='\''){ P++; int k=0;
            while(*P){ if(*P=='\''){ if(P[1]=='\''){ if(k<511)TOKB[k++]='\''; P+=2; continue; } P++; break; }
                       if(k<511) TOKB[k++]=*P; P++; }
            TOKB[k]=0; TOKT=6; return; }
        int k=0;
        while(*P && !strchr(" \t\r\n()[],:;",*P)){ if(k<511) TOKB[k++]=*P; P++; }
        TOKB[k]=0; TOKT=6; return;
    }
    TOKT=0;
}
static PTree *pt_new(int cap){
    PTree *t=(PTree*)malloc(sizeof(PTree));
    t->n=0; t->nch=(int*)malloc(cap*sizeof(int)); t->ch=(int*)malloc(cap*3*sizeof(int));
    t->leaf=(int*)malloc(cap*sizeof(int)); t->bl=(double*)malloc(cap*sizeof(double));
    t->sup=(double*)malloc(cap*sizeof(double)); return t;
}
static int pt_node(PTree*t){ int v=t->n++; t->nch[v]=0; t->leaf[v]=-1; t->bl[v]=0.0; t->sup[v]=NAN; return v; }
static void pt_free(PTree*t){ free(t->nch);free(t->ch);free(t->leaf);free(t->bl);free(t->sup);free(t); }
static int parse_sub(PTree*t){
    int v=pt_node(t);
    if(TOKT==1){
        tok_next();
        int c=parse_sub(t); if(t->nch[v]<3) t->ch[v*3+t->nch[v]++]=c;
        while(TOKT==3){ tok_next(); c=parse_sub(t); if(t->nch[v]<3) t->ch[v*3+t->nch[v]++]=c; }
        if(TOKT==2) tok_next();
        if(TOKT==6){ char *e; double d=strtod(TOKB,&e); if(e!=TOKB&&*e==0) t->sup[v]=d; tok_next(); }
        if(TOKT==4){ tok_next(); if(TOKT==6){ t->bl[v]=strtod(TOKB,NULL); tok_next(); } }
    } else if(TOKT==6){
        t->leaf[v]=name_intern(TOKB); tok_next();
        if(TOKT==4){ tok_next(); if(TOKT==6){ t->bl[v]=strtod(TOKB,NULL); tok_next(); } }
    }
    return v;
}

typedef struct { Tree t; double *ebl; double *esup; } GTree;

static GTree *to_utree(PTree *p){
    int m=p->n, root0=0, nch0=p->nch[root0];
    int drop=(nch0==2);
    int *map=(int*)malloc(m*sizeof(int));
    int nn=0;
    for(int i=0;i<m;i++) map[i]=(drop&&i==root0)?-1:nn++;
    GTree *g=(GTree*)malloc(sizeof(GTree));
    tree_alloc(&g->t,nn);
    g->ebl=(double*)malloc(nn*sizeof(double));
    g->esup=(double*)malloc(nn*sizeof(double));
    for(int i=0;i<nn;i++){ g->ebl[i]=0.0; g->esup[i]=NAN; }
    for(int i=0;i<m;i++){ int v=map[i]; if(v>=0) g->t.leaf[v]=p->leaf[i]; }
    /* link every parsed edge whose parent survived the drop */
    for(int i=0;i<m;i++){
        if(drop && i==root0) continue;
        int v=map[i]; if(v<0) continue;
        for(int k=0;k<p->nch[i] && k<3;k++){
            int c=p->ch[i*3+k];
            tree_link(&g->t,v,map[c]);
            g->ebl[map[c]]=p->bl[c]; g->esup[map[c]]=p->sup[c];
        }
    }
    if(drop){
        /* the two children of the serialisation root lose their parent: join
         * them directly so the tree becomes unrooted with nn = m-1 nodes */
        int c1=p->ch[root0*3+0], c2=p->ch[root0*3+1];
        int parc,chdc;
        if(p->leaf[c1]<0 && p->leaf[c2]>=0){ parc=c1; chdc=c2; }
        else if(p->leaf[c2]<0 && p->leaf[c1]>=0){ parc=c2; chdc=c1; }
        else { parc=c1; chdc=c2; }
        tree_link(&g->t,map[parc],map[chdc]);
        g->ebl[map[chdc]]=p->bl[c1]+p->bl[c2];
        g->esup[map[chdc]]= !isnan(p->sup[c1]) ? p->sup[c1] : p->sup[c2];
        g->t.root=map[parc];
    } else {
        g->t.root=map[root0];
    }
    free(map);
    if(g->t.deg[g->t.root]!=3) tree_pick_root(&g->t);
    return g;
}

/* ------------------------------------------------------- weighting of edges */
static int WT_MODE = 0;   /* 0 = plain ASTRAL (all gene-tree edge weights 1) */
static double WT_B0 = 0.01;
static double WT_SUPSHIFT = 1.0/3.0;
static double WT_POW = 1.0;
static double WT_LAM = 0.5;
static double WT_SUPCUT = 0.5;
static double edge_weight(double bl,double sup){
    double w=1.0, s;
    switch(WT_MODE){
    case 0: w=1.0; break;
    case 1: w = isnan(sup)?1.0:sup; break;
    case 2: w = isnan(sup)?1.0:sup*sup; break;
    case 3: s = isnan(sup)?1.0:sup; w=(s>WT_SUPSHIFT)?(s-WT_SUPSHIFT)/(1.0-WT_SUPSHIFT):0.0; break;
    case 4: w = bl/(bl+WT_B0); break;
    case 5: s = isnan(sup)?1.0:sup; w = s*bl/(bl+WT_B0); break;
    case 6: w = (isnan(sup)||sup>=0.5)?1.0:0.0; break;
    case 7: w = (isnan(sup)||sup>=0.7)?1.0:0.0; break;
    case 8: s = isnan(sup)?1.0:sup; w = sqrt(s); break;
    case 9: s = isnan(sup)?1.0:sup; w=((s>WT_SUPSHIFT)?(s-WT_SUPSHIFT)/(1.0-WT_SUPSHIFT):0.0)*bl/(bl+WT_B0); break;
    case 10: w = bl; break;
    case 11: w = 1.0-exp(-bl/WT_B0); break;
    case 12: s = isnan(sup)?1.0:sup; w = pow(s,WT_POW); break;          /* support^p   */
    case 13: s = isnan(sup)?1.0:sup; w = (1.0-WT_LAM) + WT_LAM*s; break; /* mixture    */
    case 14: s = isnan(sup)?1.0:sup; w = pow(s,WT_POW)*bl/(bl+WT_B0); break;
    case 15: s = isnan(sup)?1.0:sup; w = (s>=WT_SUPCUT)?1.0:(s/WT_SUPCUT); break;
    default: w=1.0;
    }
    return w<0?0:w;
}

/* --------------------------------------------------------- gene node table */
static u64 *GB1,*GB2;
static int *GS1,*GS2,*GS3;
static double *GW1,*GW2,*GW3;
static int NGN;

static void build_gene_nodes(GTree **gt,int ng){
    int cap=ng*(NTAX-2)+8;
    GB1=(u64*)malloc((size_t)cap*NW*sizeof(u64));
    GB2=(u64*)malloc((size_t)cap*NW*sizeof(u64));
    GS1=(int*)malloc(cap*sizeof(int)); GS2=(int*)malloc(cap*sizeof(int)); GS3=(int*)malloc(cap*sizeof(int));
    GW1=(double*)malloc(cap*sizeof(double)); GW2=(double*)malloc(cap*sizeof(double)); GW3=(double*)malloc(cap*sizeof(double));
    NGN=0;
    u64 *rest=(u64*)malloc(NW*sizeof(u64));
    for(int gi=0;gi<ng;gi++){
        Tree *t=&gt[gi]->t;
        tree_root_compute(t);
        for(int v=0;v<t->nn;v++){
            if(t->leaf[v]>=0 || t->deg[v]!=3) continue;
            const u64 *L[3]; int sz[3]; double wt[3];
            int isroot=(t->par[v]<0);
            if(!isroot) bs_comp(rest,t->clade+(size_t)v*NW);
            for(int k=0;k<3;k++){
                int w=t->adj[v][k];
                if(!isroot && w==t->par[v]){
                    L[k]=rest; sz[k]=NTAX-t->csz[v];
                    wt[k]=edge_weight(gt[gi]->ebl[v],gt[gi]->esup[v]);
                } else {
                    L[k]=t->clade+(size_t)w*NW; sz[k]=t->csz[w];
                    wt[k]=edge_weight(gt[gi]->ebl[w],gt[gi]->esup[w]);
                }
            }
            int o=NGN++;
            bs_copy(GB1+(size_t)o*NW,L[0]);
            bs_copy(GB2+(size_t)o*NW,L[1]);
            GS1[o]=sz[0]; GS2[o]=sz[1]; GS3[o]=sz[2];
            GW1[o]=wt[0]; GW2[o]=wt[1]; GW3[o]=wt[2];
        }
    }
    free(rest);
}

/* ------------------------------------------------------- four-taxon counts */
/* All three counts needed for one NNI, in a single pass over the gene nodes.
 * out[0]=N(A,B|C,D)  out[1]=N(A,C|B,D)  out[2]=N(A,D|B,C)               */
static void nni_counts(const u64*A,const u64*B,const u64*C,
                       int na,int nb,int nc,double out[3]){
    double n0=0,n1=0,n2=0;
    const u64 *L1=GB1, *L2=GB2;
    for(int i=0;i<NGN;i++,L1+=NW,L2+=NW){
        int a1=0,a2=0,b1=0,b2=0,c1=0,c2=0;
        for(int w=0;w<NW;w++){
            u64 x1=L1[w], x2=L2[w], aw=A[w], bw=B[w], cw=C[w];
            a1+=__builtin_popcountll(aw&x1); a2+=__builtin_popcountll(aw&x2);
            b1+=__builtin_popcountll(bw&x1); b2+=__builtin_popcountll(bw&x2);
            c1+=__builtin_popcountll(cw&x1); c2+=__builtin_popcountll(cw&x2);
        }
        int a3=na-a1-a2, b3=nb-b1-b2, c3=nc-c1-c2;
        int s1=GS1[i], s2=GS2[i], s3=GS3[i];
        int d1=s1-a1-b1-c1, d2=s2-a2-b2-c2, d3=s3-a3-b3-c3;
        n0 += GW3[i]*(double)(a1*b2+a2*b1)*(double)(c3*d3)
            + GW2[i]*(double)(a1*b3+a3*b1)*(double)(c2*d2)
            + GW1[i]*(double)(a2*b3+a3*b2)*(double)(c1*d1);
        n1 += GW3[i]*(double)(a1*c2+a2*c1)*(double)(b3*d3)
            + GW2[i]*(double)(a1*c3+a3*c1)*(double)(b2*d2)
            + GW1[i]*(double)(a2*c3+a3*c2)*(double)(b1*d1);
        n2 += GW3[i]*(double)(a1*d2+a2*d1)*(double)(b3*c3)
            + GW2[i]*(double)(a1*d3+a3*d1)*(double)(b2*c2)
            + GW1[i]*(double)(a2*d3+a3*d2)*(double)(b1*c1);
    }
    out[0]=n0; out[1]=n1; out[2]=n2;
}

/* --------------------------------------------------------------- ASTRID */
static double *DMAT_I, *DMAT_B;
static void astrid_accum(GTree *g){
    Tree *t=&g->t; int nn=t->nn;
    int *dist=(int*)malloc(nn*sizeof(int));
    double *dbl=(double*)malloc(nn*sizeof(double));
    int *qq=(int*)malloc(nn*sizeof(int));
    tree_root_compute(t);
    for(int s=0;s<nn;s++){
        if(t->leaf[s]<0) continue;
        for(int i=0;i<nn;i++) dist[i]=-1;
        dist[s]=0; dbl[s]=0.0;
        int qh=0,qt=0; qq[qt++]=s;
        while(qh<qt){
            int v=qq[qh++];
            for(int k=0;k<t->deg[v];k++){
                int w=t->adj[v][k];
                if(dist[w]>=0) continue;
                dist[w]=dist[v]+1;
                dbl[w]=dbl[v]+((t->par[w]==v)? g->ebl[w] : g->ebl[v]);
                qq[qt++]=w;
            }
        }
        int si=t->leaf[s];
        double *ri=DMAT_I+(size_t)si*NTAX, *rb=DMAT_B+(size_t)si*NTAX;
        for(int v=0;v<nn;v++){
            if(t->leaf[v]<0||v==s) continue;
            int j=t->leaf[v];
            ri[j]+=dist[v]; rb[j]+=dbl[v];
        }
    }
    free(dist);free(qq);free(dbl);
}

static void nj_tree(const double *Din,int n,Tree *T){
    double *D=(double*)malloc((size_t)n*n*sizeof(double));
    memcpy(D,Din,(size_t)n*n*sizeof(double));
    int *act=(int*)malloc(n*sizeof(int));
    int *slot=(int*)malloc((2*n+2)*sizeof(int));
    double *S=(double*)malloc(n*sizeof(double));
    tree_alloc(T,2*n-2);
    for(int i=0;i<n;i++){ T->leaf[i]=i; act[i]=i; slot[i]=i; }
    for(int i=n;i<2*n-2;i++) T->leaf[i]=-1;
    int r=n,next=n;
    while(r>3){
        for(int ii=0;ii<r;ii++){
            const double *Di=D+(size_t)slot[act[ii]]*n; double acc=0.0;
            for(int jj=0;jj<r;jj++) acc+=Di[slot[act[jj]]];
            S[ii]=acc;
        }
        double best=1e300; int bi=0,bj=1;
        for(int ii=0;ii<r;ii++){
            const double *Di=D+(size_t)slot[act[ii]]*n;
            for(int jj=ii+1;jj<r;jj++){
                double qv=(double)(r-2)*Di[slot[act[jj]]]-S[ii]-S[jj];
                if(qv<best){ best=qv; bi=ii; bj=jj; }
            }
        }
        int ni=act[bi], nj=act[bj];
        int si=slot[ni], sj=slot[nj];
        int u=next++;
        tree_link(T,u,ni); tree_link(T,u,nj);
        double dij=D[(size_t)si*n+sj];
        for(int k=0;k<n;k++){
            if(k==si||k==sj) continue;
            double nv=0.5*(D[(size_t)si*n+k]+D[(size_t)sj*n+k]-dij);
            D[(size_t)si*n+k]=nv; D[(size_t)k*n+si]=nv;
        }
        slot[u]=si; act[bi]=u;
        int w=0; for(int ii=0;ii<r;ii++) if(act[ii]!=nj) act[w++]=act[ii];
        r--;
    }
    { int u=next++; tree_link(T,u,act[0]); tree_link(T,u,act[1]); tree_link(T,u,act[2]); }
    T->nn=next; tree_pick_root(T);
    free(D);free(act);free(S);free(slot);
}

/* ------------------------------------------------------------------- NNI */
static int *EU,*EV,*NUA,*NUB,*NVA,*NVB,*NCH,*NUSED;
static double *NDEL;
static u64 *NA,*NB,*NC,*ND;
static int NEDGE;

static int collect_internal_edges(Tree*t,int *eu,int *ev){
    tree_root_compute(t);
    int r=t->root,k=0;
    for(int v=0;v<t->nn;v++){
        if(v==r||t->leaf[v]>=0) continue;
        int sz=t->csz[v];
        if(sz<2||sz>NTAX-2) continue;
        eu[k]=t->par[v]; ev[k]=v; k++;
    }
    return k;
}
static void comp_towards(Tree*t,int u,int w,u64*out){
    if(t->par[u]==w) bs_comp(out,t->clade+(size_t)u*NW);
    else             bs_copy(out,t->clade+(size_t)w*NW);
}
static inline void relink(Tree*t,int node,int oldn,int newn){
    for(int k=0;k<t->deg[node];k++) if(t->adj[node][k]==oldn){ t->adj[node][k]=newn; return; }
}
static inline void apply_nni(Tree*t,int u,int v,int opt){
    int ua=-1,ub=-1;
    for(int j=0;j<t->deg[u];j++){ int w=t->adj[u][j]; if(w==v) continue; if(ua<0) ua=w; else ub=w; }
    int va=-1,vb=-1;
    for(int j=0;j<t->deg[v];j++){ int w=t->adj[v][j]; if(w==u) continue; if(va<0) va=w; else vb=w; }
    if(ua<0||ub<0||va<0||vb<0) return;
    if(opt==1){                                   /* swap B and D */
        t->adj[u][0]=v; t->adj[u][1]=ua; t->adj[u][2]=vb;
        t->adj[v][0]=u; t->adj[v][1]=va; t->adj[v][2]=ub;
        relink(t,ub,u,v); relink(t,vb,v,u);
    } else {                                      /* swap B and C */
        t->adj[u][0]=v; t->adj[u][1]=ua; t->adj[u][2]=va;
        t->adj[v][0]=u; t->adj[v][1]=vb; t->adj[v][2]=ub;
        relink(t,ub,u,v); relink(t,va,v,u);
    }
    t->deg[u]=3; t->deg[v]=3;
}
/* fill NA..ND and sizes for internal edge k; returns 0 on failure */
static int edge_components(Tree*t,int k,int *na,int *nb,int *nc,int *nd){
    int u=EU[k],v=EV[k];
    int ua=-1,ub=-1,va=-1,vb=-1;
    for(int j=0;j<t->deg[u];j++){ int w=t->adj[u][j]; if(w==v) continue; if(ua<0) ua=w; else ub=w; }
    for(int j=0;j<t->deg[v];j++){ int w=t->adj[v][j]; if(w==u) continue; if(va<0) va=w; else vb=w; }
    if(ua<0||ub<0||va<0||vb<0) return 0;
    NUA[k]=ua; NUB[k]=ub; NVA[k]=va; NVB[k]=vb;
    comp_towards(t,u,ua,NA);
    comp_towards(t,u,ub,NB);
    bs_copy(NC,t->clade+(size_t)va*NW);
    bs_copy(ND,t->clade+(size_t)vb*NW);
    *na=bs_pop(NA); *nb=bs_pop(NB); *nc=bs_pop(NC); *nd=bs_pop(ND);
    return 1;
}

/* Evaluate all internal edges.  Improving moves are stored; *totdelta receives
 * the exact sum of deltas of a node-disjoint improving set (moves that share
 * any of the six incident components cannot both be taken, and moves that do
 * not share one provably affect disjoint quartet sets, so their deltas add). */
static double APPLIED_DELTA=0.0;
static int nni_sweep(Tree*t,double *bestdelta,int *bestedge,int *bestopt,int applybatch){
    APPLIED_DELTA=0.0;
    NEDGE=collect_internal_edges(t,EU,EV);
    double best=0.0; int bestk=-1,besto=-1,nimp=0;
    for(int k=0;k<NEDGE;k++){
        NCH[k]=-1; NDEL[k]=0.0;
        int na,nb,nc,nd;
        if(!edge_components(t,k,&na,&nb,&nc,&nd)) continue;
        double out[3];
        nni_counts(NA,NB,NC,na,nb,nc,out);
        double d1=out[1]-out[0], d2=out[2]-out[0];
        double d=0.0; int opt=-1;
        if(d1>=d2 && d1>EPS){ d=d1; opt=0; } else if(d2>EPS){ d=d2; opt=1; }
        if(opt>=0){ NCH[k]=opt; NDEL[k]=d; nimp++;
            if(d>best){ best=d; bestk=k; besto=opt; } }
        if((k&7)==7 && now_sec()>T_LIMIT) break;
    }
    *bestdelta=best; *bestedge=bestk; *bestopt=besto;
    if(nimp==0||bestk<0) return 0;
    if(!applybatch){ apply_nni(t,EU[bestk],EV[bestk],besto); APPLIED_DELTA=best; return 1; }
    int *ordx=(int*)malloc(NEDGE*sizeof(int)); int no=0;
    for(int k=0;k<NEDGE;k++) if(NCH[k]>=0) ordx[no++]=k;
    for(int i=1;i<no;i++){ int key=ordx[i]; double kd=NDEL[key]; int j=i-1;
        while(j>=0 && NDEL[ordx[j]]<kd){ ordx[j+1]=ordx[j]; j--; } ordx[j+1]=key; }
    memset(NUSED,0,sizeof(int)*t->nn);
    int nap=0;
    for(int i=0;i<no;i++){
        int k=ordx[i],u=EU[k],v=EV[k];
        if(NUSED[u]||NUSED[v]||NUSED[NUA[k]]||NUSED[NUB[k]]||NUSED[NVA[k]]||NUSED[NVB[k]]) continue;
        apply_nni(t,u,v,NCH[k]);
        NUSED[u]=NUSED[v]=NUSED[NUA[k]]=NUSED[NUB[k]]=NUSED[NVA[k]]=NUSED[NVB[k]]=1;
        APPLIED_DELTA+=NDEL[k];
        nap++;
    }
    free(ordx);
    return nap;
}

/* ------------------------------------------------------------ hill climbing */
static Tree *WORK,*BEST,*CAND,*BAK;
static double REL_BEST=0.0;

static double hill_nni(Tree*t,double rel,int batch){
    while(now_sec()<T_LIMIT){
        double bd; int be,bo;
        int r=nni_sweep(t,&bd,&be,&bo,batch);
        if(r==0||bd<=1e-6) break;
        rel += batch ? APPLIED_DELTA : bd;
    }
    return rel;
}

static double perturb(Tree*t,double rel,int nmoves){
    for(int m=0;m<nmoves && now_sec()<T_LIMIT;m++){
        NEDGE=collect_internal_edges(t,EU,EV);
        if(NEDGE<=0) break;
        int k=rndint(NEDGE);
        int na,nb,nc,nd;
        if(!edge_components(t,k,&na,&nb,&nc,&nd)) continue;
        double out[3];
        nni_counts(NA,NB,NC,na,nb,nc,out);
        int opt=rndint(2);
        apply_nni(t,EU[k],EV[k],opt);
        rel += (opt==1? out[2]-out[0] : out[1]-out[0]);
    }
    return rel;
}

/* ------------------------------------------------------------- newick out */
static char *OUTBUF; static size_t OUTLEN,OUTCAP;
static void out_reserve(size_t l){ while(OUTLEN+l+1>OUTCAP){ OUTCAP=OUTCAP?OUTCAP*2:(1<<16); OUTBUF=(char*)realloc(OUTBUF,OUTCAP);} }
static void out_puts(const char*s){ size_t l=strlen(s); out_reserve(l); memcpy(OUTBUF+OUTLEN,s,l); OUTLEN+=l; OUTBUF[OUTLEN]=0; }
static void out_char(char c){ out_reserve(1); OUTBUF[OUTLEN++]=c; OUTBUF[OUTLEN]=0; }
static void write_sub(Tree*t,int v,int p){
    if(t->leaf[v]>=0){ out_puts(TNAMES[t->leaf[v]]); return; }
    out_char('(');
    int first=1;
    for(int k=0;k<t->deg[v];k++){ int w=t->adj[v][k]; if(w==p) continue;
        if(!first) out_char(','); first=0; write_sub(t,w,v); }
    out_char(')');
}
static void write_newick(Tree*t){
    OUTLEN=0; out_reserve(64); OUTBUF[0]=0;
    int r=tree_pick_root(t);
    out_char('(');
    int first=1;
    for(int k=0;k<t->deg[r];k++){ if(!first) out_char(','); first=0; write_sub(t,t->adj[r][k],r); }
    out_puts(");");
}

/* ------------------------------------------------------- brute self test */
static int G_NG;
static u64 *BF_TS; static int BF_NT;
static u64 *BF_GS; static int *BF_GN; static int BF_STRIDE;
static int collect_splits(Tree*t,u64*dst){
    tree_root_compute(t);
    int r=t->root,k=0;
    for(int v=0;v<t->nn;v++){
        if(v==r) continue;
        int sz=t->csz[v];
        if(sz<2||sz>NTAX-2) continue;
        bs_copy(dst+(size_t)k*NW,t->clade+(size_t)v*NW);
        bs_canon(dst+(size_t)k*NW);
        k++;
    }
    return k;
}
/* pairing index of a 4-subset in a split list: 0 = (01|23), 1 = (02|13), 2 = (03|12) */
static int pairing_index(const u64*base,int ns,const int idx[4]){
    for(int s=0;s<ns;s++){
        const u64*S=base+(size_t)s*NW;
        int p[4],cnt=0;
        for(int q=0;q<4;q++){ p[q]=bs_test(S,idx[q])?1:0; cnt+=p[q]; }
        if(cnt!=2) continue;
        if(p[0]==p[1]&&p[2]==p[3]) return 0;
        if(p[0]==p[2]&&p[1]==p[3]) return 1;
        return 2;
    }
    return -1;
}
/* brute force N(P,Q|R,S) = sum_g #{(p,q,r,s) in PxQxRxS : g displays pq|rs} */
static double brute_n4(const u64*Pp,const u64*Qq,const u64*Rr,const u64*Ss){
    double tot=0; int idx[4];
    for(int p=0;p<NTAX;p++){ if(!bs_test(Pp,p)) continue;
     for(int q=0;q<NTAX;q++){ if(!bs_test(Qq,q)) continue;
      for(int r=0;r<NTAX;r++){ if(!bs_test(Rr,r)) continue;
       for(int t4=0;t4<NTAX;t4++){ if(!bs_test(Ss,t4)) continue;
        idx[0]=p;idx[1]=q;idx[2]=r;idx[3]=t4;
        for(int gi=0;gi<G_NG;gi++)
            if(pairing_index(BF_GS+(size_t)gi*BF_STRIDE*NW,BF_GN[gi],idx)==0) tot+=1.0;
       }}}}
    return tot;
}
static double brute_score(Tree*t){
    BF_NT=collect_splits(t,BF_TS);
    int idx[4]; double tot=0;
    for(int a=0;a<NTAX;a++) for(int b=a+1;b<NTAX;b++) for(int c=b+1;c<NTAX;c++) for(int d=c+1;d<NTAX;d++){
        idx[0]=a;idx[1]=b;idx[2]=c;idx[3]=d;
        int tp=pairing_index(BF_TS,BF_NT,idx);
        if(tp<0) continue;
        for(int gi=0;gi<G_NG;gi++)
            if(pairing_index(BF_GS+(size_t)gi*BF_STRIDE*NW,BF_GN[gi],idx)==tp) tot+=1.0;
    }
    return tot;
}
/* ------------------------------------------------------------ config/main */
static int OPT_BATCH=1, OPT_RATCHET=1, OPT_SEEDDIST=0, OPT_SELFTEST=0;
static double OPT_SEARCH_FRAC=0.85;
static int OPT_KICK0=2, OPT_KICKMAX=12;

int main(int argc,char**argv){
    T0=now_sec();
    if(argc<3){ fprintf(stderr,"usage: %s in.tre out.tre\n",argv[0]); return 2; }
    const char *e;
    if((e=getenv("SEED"))){ long s=strtol(e,NULL,10); rng_s=(u64)(s?s:1)*0x9E3779B97F4A7C15ULL+12345678ULL; if(!rng_s) rng_s=1; }
    if((e=getenv("WT"))) WT_MODE=atoi(e);
    if((e=getenv("WT_B0"))) WT_B0=atof(e);
    if((e=getenv("WT_SHIFT"))) WT_SUPSHIFT=atof(e);
    if((e=getenv("WT_POW"))) WT_POW=atof(e);
    if((e=getenv("WT_LAM"))) WT_LAM=atof(e);
    if((e=getenv("WT_CUT"))) WT_SUPCUT=atof(e);
    if((e=getenv("DEADLINE"))) DEADLINE=atof(e);
    if((e=getenv("BATCH"))) OPT_BATCH=atoi(e);
    if((e=getenv("RATCHET"))) OPT_RATCHET=atoi(e);
    if((e=getenv("SFRAC"))) OPT_SEARCH_FRAC=atof(e);
    if((e=getenv("DIST"))) OPT_SEEDDIST=atoi(e);
    if((e=getenv("SELFTEST"))) OPT_SELFTEST=atoi(e);
    if((e=getenv("KICK0"))) OPT_KICK0=atoi(e);
    if((e=getenv("KICKMAX"))) OPT_KICKMAX=atoi(e);
    if((e=getenv("EPS"))) EPS=atof(e);

    FILE *f=fopen(argv[1],"rb");
    if(!f){ perror("open"); return 2; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=(char*)malloc((size_t)fsz+2);
    size_t rd=fread(buf,1,(size_t)fsz,f); buf[rd]=0; fclose(f);
    int nlines=0; for(size_t i=0;i<rd;i++) if(buf[i]=='\n') nlines++;
    if(rd>0 && buf[rd-1]!='\n') nlines++;
    if(nlines<1) nlines=1;

    names_init(4096);
    PTree **pts=(PTree**)malloc(sizeof(PTree*)*(size_t)nlines);
    int ng=0;
    { char *line=buf;
      while(*line){
        char *nl=strchr(line,'\n'); if(nl) *nl=0;
        char *s=line; while(*s==' '||*s=='\t'||*s=='\r') s++;
        if(*s && strchr(s,'(')){
            P=s; tok_next();
            PTree *pt=pt_new(1200);
            parse_sub(pt);
            pts[ng++]=pt;
        }
        if(!nl) break;
        line=nl+1;
      } }
    if(ng<1){ fprintf(stderr,"no trees parsed\n"); return 2; }
    NTAX=NNAME;
    if(NTAX>MAXN){ fprintf(stderr,"too many taxa: %d\n",NTAX); return 2; }
    NW=(NTAX+63)/64;
    for(int i=0;i<NW;i++) ALLW[i]=~0ULL;
    if(NTAX%64) ALLW[NW-1]=((u64)1<<(NTAX%64))-1;

    GTree **gts=(GTree**)malloc(sizeof(GTree*)*ng);
    for(int i=0;i<ng;i++){ gts[i]=to_utree(pts[i]); pt_free(pts[i]); }
    G_NG=ng;

    DMAT_I=(double*)calloc((size_t)NTAX*NTAX,sizeof(double));
    DMAT_B=(double*)calloc((size_t)NTAX*NTAX,sizeof(double));
    for(int i=0;i<ng;i++) astrid_accum(gts[i]);
    double inv=1.0/(double)ng;
    for(size_t i=0;i<(size_t)NTAX*NTAX;i++){ DMAT_I[i]*=inv; DMAT_B[i]*=inv; }

    build_gene_nodes(gts,ng);
    fprintf(stderr,"NTAX=%d genes=%d nw=%d gene-nodes=%d t=%.2f\n",NTAX,ng,NW,NGN,elapsed());

    int nnodes=2*NTAX-2;
    EU=(int*)malloc((NTAX+8)*sizeof(int)); EV=(int*)malloc((NTAX+8)*sizeof(int));
    NUA=(int*)malloc((NTAX+8)*sizeof(int)); NUB=(int*)malloc((NTAX+8)*sizeof(int));
    NVA=(int*)malloc((NTAX+8)*sizeof(int)); NVB=(int*)malloc((NTAX+8)*sizeof(int));
    NCH=(int*)malloc((NTAX+8)*sizeof(int)); NDEL=(double*)malloc((NTAX+8)*sizeof(double));
    NUSED=(int*)malloc(MAXNODES*sizeof(int));
    NA=(u64*)malloc(NW*sizeof(u64)); NB=(u64*)malloc(NW*sizeof(u64));
    NC=(u64*)malloc(NW*sizeof(u64)); ND=(u64*)malloc(NW*sizeof(u64));

    WORK=(Tree*)malloc(sizeof(Tree)); BEST=(Tree*)malloc(sizeof(Tree));
    CAND=(Tree*)malloc(sizeof(Tree)); BAK=(Tree*)malloc(sizeof(Tree));
    tree_alloc(BEST,nnodes); tree_alloc(CAND,nnodes); tree_alloc(BAK,nnodes);

    nj_tree(OPT_SEEDDIST?DMAT_B:DMAT_I,NTAX,WORK);
    tree_pick_root(WORK);
    if(!tree_ok(WORK)){ fprintf(stderr,"bad seed tree\n"); return 3; }
    tree_copy(WORK,BEST); tree_pick_root(BEST);

    if(OPT_SELFTEST){
        BF_TS=(u64*)malloc((size_t)(NTAX+8)*NW*sizeof(u64));
        BF_STRIDE=NTAX+8;
        BF_GS=(u64*)malloc((size_t)ng*BF_STRIDE*NW*sizeof(u64));
        BF_GN=(int*)malloc(ng*sizeof(int));
        for(int i=0;i<ng;i++) BF_GN[i]=collect_splits(&gts[i]->t,BF_GS+(size_t)i*BF_STRIDE*NW);
        { int bad=0; u64 *L3=(u64*)malloc(NW*sizeof(u64));
          for(int i=0;i<NGN;i++){
            const u64*L1=GB1+(size_t)i*NW,*L2=GB2+(size_t)i*NW;
            for(int w=0;w<NW;w++) L3[w]=ALLW[w]&~(L1[w]|L2[w]);
            if(bs_pop(L1)!=GS1[i]||bs_pop(L2)!=GS2[i]||bs_pop(L3)!=GS3[i]) bad++;
            if(GS1[i]+GS2[i]+GS3[i]!=NTAX||GS1[i]<1||GS2[i]<1||GS3[i]<1) bad++;
          }
          fprintf(stderr,"GENETABLE check bad=%d of %d\n",bad,NGN); free(L3); }
        double bs0=brute_score(BEST);
        fprintf(stderr,"SELFTEST brute(seed)=%.1f\n",bs0);
        Tree *S=(Tree*)malloc(sizeof(Tree)); tree_alloc(S,nnodes);
        tree_copy(BEST,S); tree_pick_root(S);
        double rel=0.0; int bad=0;
        for(int step=0;step<400;step++){
            NEDGE=collect_internal_edges(S,EU,EV);
            if(NEDGE<=0) break;
            int k=rndint(NEDGE);
            int na,nb,nc,nd;
            if(!edge_components(S,k,&na,&nb,&nc,&nd)) continue;
            double out[3]; nni_counts(NA,NB,NC,na,nb,nc,out);
            if(step<6){
                u64 *S4=(u64*)malloc(NW*sizeof(u64));
                for(int w=0;w<NW;w++) S4[w]=ALLW[w]&~(NA[w]|NB[w]|NC[w]);
                double b0=brute_n4(NA,NB,NC,S4);
                double b1=brute_n4(NA,NC,NB,S4);
                double b2=brute_n4(NA,S4,NB,NC);
                fprintf(stderr,"N4 step=%d fast=[%.0f %.0f %.0f] brute=[%.0f %.0f %.0f] sizes a=%d b=%d c=%d d=%d\n",
                        step,out[0],out[1],out[2],b0,b1,b2,na,nb,nc,bs_pop(S4));
                free(S4);
            }
            int opt=rndint(2);
            apply_nni(S,EU[k],EV[k],opt);
            tree_pick_root(S);
            rel += (opt==1? out[2]-out[0] : out[1]-out[0]);
            double bs=brute_score(S);
            if(fabs((bs0+rel)-bs)>1e-6){
                fprintf(stderr,"SELFTEST MISMATCH step=%d tracked=%.1f brute=%.1f\n",step,bs0+rel,bs);
                if(++bad>3) break;
            }
        }
        fprintf(stderr,"SELFTEST %s (bad=%d)\n",bad?"FAILED":"PASSED",bad);
        return bad?4:0;
    }

    T_LIMIT = T0 + DEADLINE*OPT_SEARCH_FRAC;

    tree_copy(BEST,CAND); tree_pick_root(CAND);
    REL_BEST=hill_nni(CAND,0.0,OPT_BATCH);
    fprintf(stderr,"NNI climb: rel=%+.1f t=%.2f\n",REL_BEST,elapsed());
    if(tree_ok(CAND)){ tree_copy(CAND,BEST); tree_pick_root(BEST); } else REL_BEST=0.0;

    if(OPT_RATCHET){
        long rounds=0; int kick=OPT_KICK0;
        while(now_sec()<T_LIMIT){
            rounds++;
            tree_copy(BEST,CAND); tree_pick_root(CAND);
            double relc=perturb(CAND,REL_BEST,kick);
            relc=hill_nni(CAND,relc,OPT_BATCH);
            if(relc>REL_BEST+EPS && tree_ok(CAND)){
                REL_BEST=relc; tree_copy(CAND,BEST); tree_pick_root(BEST);
                fprintf(stderr,"ratchet %ld: rel=%+.1f kick=%d t=%.2f\n",rounds,relc,kick,elapsed());
                kick=OPT_KICK0;
            } else if(kick<OPT_KICKMAX) kick+=2;
        }
        fprintf(stderr,"ratchet done rounds=%ld rel=%+.1f t=%.2f\n",rounds,REL_BEST,elapsed());
    }

    if(!tree_ok(BEST)){ tree_copy(WORK,BEST); tree_pick_root(BEST); }
    for(int i=0;i<ng;i++){ free(gts[i]->ebl); free(gts[i]->esup); tree_free(&gts[i]->t); free(gts[i]); }
    write_newick(BEST);
    FILE *o=fopen(argv[2],"wb");
    if(!o){ perror("open out"); return 2; }
    fwrite(OUTBUF,1,OUTLEN,o);
    if(fclose(o)!=0){ perror("fclose"); return 2; }
    fprintf(stderr,"done rel=%+.1f t=%.2f\n",REL_BEST,elapsed());
    return 0;
}
