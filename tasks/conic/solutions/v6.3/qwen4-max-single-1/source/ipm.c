#include "npzio.h"
#include "kkt.h"

void eig_sym(int n,double *A,double *w,double *V);
void mmul(int n,const double *A,const double *B,double *C);
void mmulT(int n,const double *A,const double *B,double *C);

#define SQRT2 1.4142135623730951
#define INV_SQRT2 0.7071067811865475

typedef struct {
    int type; idx_t dim,start,width;
    idx_t *diag_slot;      /* width entries: triplet idx of diagonal */
    idx_t aux;             /* soc aux var index or -1 */
    idx_t *aux_slot;       /* soc: width triplet idx for u_k */
    idx_t *off_slot;       /* exp/psd: off-diagonal triplet indices */
    int noff;
    /* psd workspace */
    double *G,*W,*Win,*Zhalf,*Zinvhalf,*M,*Vd,*Ud,*tmp1,*tmp2,*tmp3,*ev,*EV,*S,*Z;
} Blk;

typedef struct {
    Problem *P;
    idx_t N,n,m,nsoc;
    long nnz;
    idx_t *ti,*tj; double *tv;
    idx_t *diag_slot;
    Blk *blk; int nblk;
    KKT K;
    double *x,*z,*s,*tau,*kappa;
    double *r1,*r2,*rc,*u,*v,*dx,*dz,*ds,*work,*rhs;
    double *dxa,*dza,*dsa; /* affine directions */
    double dtau,dkappa,dtaua,dkappaa;
    double mu;
    idx_t *soc_aux_row;
    double delta1,delta2;
    double t_start;
} Solver;

/* ---- sparse matvec helpers ---- */
static void Pmv(const Problem *P,const double *x,double *y){
    idx_t n=P->n; for(idx_t i=0;i<n;i++)y[i]=0;
    const CSC *M=&P->P;
    for(idx_t c=0;c<n;c++)for(idx_t k=M->p[c];k<M->p[c+1];k++)y[M->i[k]]+=M->x[k]*x[c];
}
static void Amv(const Problem *P,const double *x,double *y){
    idx_t m=P->m,n=P->n; for(idx_t i=0;i<m;i++)y[i]=0;
    const CSC *M=&P->A;
    for(idx_t c=0;c<n;c++)for(idx_t k=M->p[c];k<M->p[c+1];k++)y[M->i[k]]+=M->x[k]*x[c];
}
static void ATmv(const Problem *P,const double *z,double *y){
    idx_t m=P->m,n=P->n; for(idx_t i=0;i<n;i++)y[i]=0;
    const CSC *M=&P->A;
    for(idx_t c=0;c<n;c++){double acc=0;for(idx_t k=M->p[c];k<M->p[c+1];k++)acc+=M->x[k]*z[M->i[k]];y[c]=acc;}
}
static double dot(const double *a,const double *b,idx_t n){double s=0;for(idx_t i=0;i<n;i++)s+=a[i]*b[i];return s;}
static double ninf(const double *a,idx_t n){double s=0;for(idx_t i=0;i<n;i++){double v=fabs(a[i]);if(v>s)s=v;}return s;}

/* ---- exp cone barrier ---- */
static void exp_grad(const double *s,double *g){
    double u=s[0],v=s[1],w=s[2];
    double t=log(w/v), r=v*t-u;
    g[0]=1.0/r;
    g[1]=-(t-1)/r-1/v;
    g[2]=-(v/r+1)/w;
}
static void exp_hess(const double *s,double *H){
    double u=s[0],v=s[1],w=s[2];
    double t=log(w/v), r=v*t-u;
    double g0=-1,g1=t-1,g2=v/w;
    double ir=1/r, ir2=ir*ir;
    H[0]=g0*g0*ir2;
    H[1]=g0*g1*ir2; H[2]=g0*g2*ir2;
    H[3]=g1*g0*ir2;
    H[4]=g1*g1*ir2+ir/v+1/(v*v);
    H[5]=g1*g2*ir2-ir/w;
    H[6]=g2*g0*ir2;
    H[7]=g2*g1*ir2-ir/w;
    H[8]=g2*g2*ir2+v/(r*w*w)+1/(w*w);
}
/* inverse of 3x3 PD */
static int inv3(const double *H,double *Hi){
    double a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7],i=H[8];
    double A=e*i-f*h,B=f*g-d*i,C=d*h-e*g;
    double det=a*A+b*B+c*C;
    if(fabs(det)<1e-300)return -1;
    double id=1/det;
    Hi[0]=A*id;Hi[1]=(c*h-b*i)*id;Hi[2]=(b*f-c*e)*id;
    Hi[3]=B*id;Hi[4]=(a*i-c*g)*id;Hi[5]=(c*d-a*f)*id;
    Hi[6]=C*id;Hi[7]=(b*g-a*h)*id;Hi[8]=(a*e-b*d)*id;
    return 0;
}

/* ---- svec helpers (column-major upper, off-diag *sqrt2) ---- */
static void smat(int d,const double *v,double *S){
    int pos=0;
    for(int j=0;j<d;j++){
        for(int i=0;i<=j;i++){
            double val=v[pos++];
            if(i<j){S[i*d+j]=val*INV_SQRT2;S[j*d+i]=val*INV_SQRT2;}
            else S[i*d+j]=val;
        }
    }
}
static void svec(int d,const double *S,double *v){
    int pos=0;
    for(int j=0;j<d;j++)for(int i=0;i<=j;i++)v[pos++]=(i<j)?S[i*d+j]*SQRT2:S[i*d+j];
}

/* ---- assembly ---- */
static long add_trip(Solver *S,idx_t i,idx_t j,double val,long *cap){
    if(S->nnz>=*cap){*cap*=2;S->ti=xrealloc(S->ti,*cap*sizeof(idx_t));S->tj=xrealloc(S->tj,*cap*sizeof(idx_t));S->tv=xrealloc(S->tv,*cap*sizeof(double));}
    long k=S->nnz++;S->ti[k]=i;S->tj[k]=j;S->tv[k]=val;return k;
}

static void assemble(Solver *S){
    Problem *P=S->P; idx_t n=P->n,m=P->m;
    S->nsoc=0;
    for(int b=0;b<P->nblk;b++)if(P->blk[b].type==2)S->nsoc++;
    S->N=n+m+S->nsoc; S->n=n;S->m=m;
    long cap=1024+(P->P.p[n]+P->A.p[n])*2+ (long)m*4;
    for(int b=0;b<P->nblk;b++){Block*B=&P->blk[b];if(B->type==4)cap+=(long)B->width*B->width;}
    S->ti=xmalloc(cap*sizeof(idx_t));S->tj=xmalloc(cap*sizeof(idx_t));S->tv=xmalloc(cap*sizeof(double));
    S->nnz=0;
    S->diag_slot=xmalloc(S->N*sizeof(idx_t));
    /* diagonals */
    for(idx_t k=0;k<S->N;k++)S->diag_slot[k]=add_trip(S,k,k,0,&cap);
    /* P strictly upper */
    const CSC *PM=&P->P;
    for(idx_t c=0;c<n;c++)for(idx_t k=PM->p[c];k<PM->p[c+1];k++){idx_t r=PM->i[k];if(r<c)add_trip(S,r,c,PM->x[k],&cap);}
    /* A -> (j, n+i) */
    const CSC *AM=&P->A;
    for(idx_t c=0;c<n;c++)for(idx_t k=AM->p[c];k<AM->p[c+1];k++){idx_t i=AM->i[k];add_trip(S,c,n+i,AM->x[k],&cap);}
    /* blocks */
    S->nblk=P->nblk;S->blk=xcalloc(S->nblk,sizeof(Blk));
    idx_t aux=n+m;
    for(int b=0;b<P->nblk;b++){
        Block *B=&P->blk[b]; Blk *E=&S->blk[b];
        E->type=B->type;E->dim=B->dim;E->start=B->start;E->width=B->width;E->aux=-1;
        E->diag_slot=xmalloc(B->width*sizeof(idx_t));
        for(idx_t k=0;k<B->width;k++)E->diag_slot[k]=S->diag_slot[n+B->start+k];
        if(B->type==2){
            E->aux=aux++;
            E->aux_slot=xmalloc(B->width*sizeof(idx_t));
            for(idx_t k=0;k<B->width;k++)E->aux_slot[k]=add_trip(S,n+B->start+k,E->aux,0,&cap);
        } else if(B->type==3){
            E->noff=3;E->off_slot=xmalloc(3*sizeof(idx_t));
            idx_t st=n+B->start;
            E->off_slot[0]=add_trip(S,st,st+1,0,&cap);
            E->off_slot[1]=add_trip(S,st,st+2,0,&cap);
            E->off_slot[2]=add_trip(S,st+1,st+2,0,&cap);
        } else if(B->type==4){
            idx_t p=B->width; long noff=p*(p-1)/2;
            E->noff=noff;E->off_slot=xmalloc((noff?noff:1)*sizeof(idx_t));
            idx_t st=n+B->start; long t=0;
            for(idx_t a=0;a<p;a++)for(idx_t bb=a+1;bb<p;bb++)E->off_slot[t++]=add_trip(S,st+a,st+bb,0,&cap);
            int d=B->dim;
            E->G=xmalloc(d*d*sizeof(double));E->W=xmalloc(d*d*sizeof(double));E->Win=xmalloc(d*d*sizeof(double));
            E->Zhalf=xmalloc(d*d*sizeof(double));E->Zinvhalf=xmalloc(d*d*sizeof(double));
            E->M=xmalloc(d*d*sizeof(double));E->tmp1=xmalloc(d*d*sizeof(double));E->tmp2=xmalloc(d*d*sizeof(double));E->tmp3=xmalloc(d*d*sizeof(double));
            E->ev=xmalloc(d*sizeof(double));E->EV=xmalloc(d*d*sizeof(double));
            E->S=xmalloc(d*d*sizeof(double));E->Z=xmalloc(d*d*sizeof(double));
        }
    }
    /* set fixed P/A values already in tv; diagonals filled per iteration */
    kkt_setup(&S->K,S->N,S->nnz,S->ti,S->tj,S->tv);
}

/* ---- cone init ---- */
static void init_cones(Solver *S){
    Problem *P=S->P;
    for(int b=0;b<P->nblk;b++){
        Block *B=&P->blk[b]; double *s=S->s+B->start,*z=S->z+B->start;
        if(B->type==0){for(idx_t k=0;k<B->width;k++){s[k]=0;z[k]=0;}}
        else if(B->type==1){for(idx_t k=0;k<B->width;k++){s[k]=1;z[k]=1;}}
        else if(B->type==2){s[0]=1;z[0]=1;for(idx_t k=1;k<B->width;k++){s[k]=0;z[k]=0;}}
        else if(B->type==3){s[0]=0;s[1]=1;s[2]=2;double g[3];exp_grad(s,g);z[0]=-g[0];z[1]=-g[1];z[2]=-g[2];}
        else {int d=B->dim;for(int j=0;j<d;j++)for(int i=0;i<=j;i++){idx_t pos=j*(j+1)/2+i;s[pos]=(i==j)?1:0;z[pos]=(i==j)?1:0;}}
    }
}

/* ---- compute H blocks and fill KKT (2,2) + aux ---- */
static void fill_H(Solver *S){
    Problem *P=S->P;
    double d1=S->delta1,d2=S->delta2;
    /* x diagonals: P_ii + d1 */
    const CSC *PM=&P->P;
    for(idx_t c=0;c<S->n;c++){
        double pii=0;
        for(idx_t k=PM->p[c];k<PM->p[c+1];k++)if(PM->i[k]==c)pii=PM->x[k];
        S->tv[S->diag_slot[c]]=pii+d1;
    }
    for(idx_t a=S->n+S->m;a<S->N;a++)S->tv[S->diag_slot[a]]=-1.0;
    for(int b=0;b<S->nblk;b++){
        Blk *E=&S->blk[b]; idx_t st=E->start; double *s=S->s+st,*z=S->z+st;
        if(E->type==0){
            for(idx_t k=0;k<E->width;k++)S->tv[E->diag_slot[k]]=-d2;
        } else if(E->type==1){
            for(idx_t k=0;k<E->width;k++){double h=s[k]/z[k];if(!(h>1e-300))h=1e-300;S->tv[E->diag_slot[k]]=-(h+d2);}
        } else if(E->type==2){
            idx_t d=E->width;
            double s0=s[0],z0=z[0];
            double ss=0,zz=0,sz=0;
            for(idx_t k=1;k<d;k++){ss+=s[k]*s[k];zz+=z[k]*z[k];sz+=s[k]*z[k];}
            double ds=s0*s0-ss, dz=z0*z0-zz;
            if(ds<1e-300)ds=1e-300; if(dz<1e-300)dz=1e-300;
            double mus=sqrt(ds),muz=sqrt(dz);
            double g2=mus/muz;
            double c=(s0*z0-sz)/(mus*muz);
            double denom=sqrt(2*(1+c)); if(denom<1e-300)denom=1e-300;
            /* abar = (stilde + J ztilde)/denom */
            double ab0=(s0/mus+z0/muz)/denom;
            S->tv[E->diag_slot[0]]=g2-d2;
            for(idx_t k=1;k<d;k++)S->tv[E->diag_slot[k]]=-g2-d2;
            double u0=sqrt(2*g2)*ab0;
            S->tv[E->aux_slot[0]]=u0;
            for(idx_t k=1;k<d;k++){
                double abk=(s[k]/mus-z[k]/muz)/denom;
                S->tv[E->aux_slot[k]]=sqrt(2*g2)*abk;
            }
        } else if(E->type==3){
            double H[9],Hi[9];
            exp_hess(s,H);
            if(inv3(H,Hi)){for(int k=0;k<3;k++)S->tv[E->diag_slot[k]]=-d2;S->tv[E->off_slot[0]]=0;S->tv[E->off_slot[1]]=0;S->tv[E->off_slot[2]]=0;continue;}
            double sz=dot(s,z,3); if(sz<1e-300)sz=1e-300;
            double alpha=3.0/sz;
            /* Hscale = alpha*Hi ; KKT block = -(Hscale+d2 I) */
            S->tv[E->diag_slot[0]]=-(alpha*Hi[0]+d2);
            S->tv[E->diag_slot[1]]=-(alpha*Hi[4]+d2);
            S->tv[E->diag_slot[2]]=-(alpha*Hi[8]+d2);
            S->tv[E->off_slot[0]]=-alpha*Hi[1];
            S->tv[E->off_slot[1]]=-alpha*Hi[2];
            S->tv[E->off_slot[2]]=-alpha*Hi[5];
        } else {
            int d=E->dim; idx_t p=E->width;
            smat(d,s,E->S); smat(d,z,E->Z);
            /* eig Z */
            double *Zc=xmalloc(d*d*sizeof(double));memcpy(Zc,E->Z,d*d*sizeof(double));
            eig_sym(d,Zc,E->ev,E->EV);
            /* Zhalf = EV sqrt(ev) EV^T ; Zinvhalf = EV 1/sqrt(ev) EV^T */
            for(int i=0;i<d;i++)for(int j=0;j<d;j++){
                double sh=0,ih=0;
                for(int k=0;k<d;k++){double e=E->ev[k];if(e<1e-300)e=1e-300;double se=sqrt(e);sh+=E->EV[i*d+k]*se*E->EV[j*d+k];ih+=E->EV[i*d+k]*(1/se)*E->EV[j*d+k];}
                E->Zhalf[i*d+j]=sh;E->Zinvhalf[i*d+j]=ih;
            }
            free(Zc);
            /* M = Zhalf S Zhalf */
            mmul(d,E->Zhalf,E->S,E->tmp1); mmul(d,E->tmp1,E->Zhalf,E->M);
            double *Mc=xmalloc(d*d*sizeof(double));memcpy(Mc,E->M,d*d*sizeof(double));
            double *ev2=xmalloc(d*sizeof(double)),*EV2=xmalloc(d*d*sizeof(double));
            eig_sym(d,Mc,ev2,EV2);
            /* Mhalf = EV2 sqrt(ev2) EV2^T */
            for(int i=0;i<d;i++)for(int j=0;j<d;j++){
                double sh=0;for(int k=0;k<d;k++){double e=ev2[k];if(e<1e-300)e=1e-300;sh+=EV2[i*d+k]*sqrt(e)*EV2[j*d+k];}
                E->tmp1[i*d+j]=sh;
            }
            /* G = Zinvhalf Mhalf Zinvhalf */
            mmul(d,E->Zinvhalf,E->tmp1,E->tmp2); mmul(d,E->tmp2,E->Zinvhalf,E->G);
            free(Mc);free(ev2);free(EV2);
            /* W = G^{1/2} */
            double *Gc=xmalloc(d*d*sizeof(double));memcpy(Gc,E->G,d*d*sizeof(double));
            double *ev3=xmalloc(d*sizeof(double)),*EV3=xmalloc(d*d*sizeof(double));
            eig_sym(d,Gc,ev3,EV3);
            for(int i=0;i<d;i++)for(int j=0;j<d;j++){
                double sh=0,ih=0;for(int k=0;k<d;k++){double e=ev3[k];if(e<1e-300)e=1e-300;double se=sqrt(e);sh+=EV3[i*d+k]*se*EV3[j*d+k];ih+=EV3[i*d+k]*(1/se)*EV3[j*d+k];}
                E->W[i*d+j]=sh;E->Win[i*d+j]=ih;
            }
            free(Gc);free(ev3);free(EV3);
            /* H_svec from G: fill diag + off */
            /* diag entries (a=a): H[a,a] = c_ij^2 * ... use formula */
            idx_t *pos=xmalloc(p*sizeof(idx_t));
            /* map svec index -> (i,j) */
            int *ii=xmalloc(p*sizeof(int)),*jj=xmalloc(p*sizeof(int));
            idx_t t=0;
            for(int j=0;j<d;j++)for(int i=0;i<=j;i++){ii[t]=i;jj[t]=j;t++;}
            for(idx_t a=0;a<p;a++){
                int i=ii[a],j=jj[a];double cij=(i<j)?SQRT2:1.0;
                double diag=cij*cij*E->G[i*d+i]*E->G[j*d+j]; /* k=l=i? no: H[a,a] source=(i,j) */
                /* H[(ij),(ij)]: if i<j: c_ij^2*(G_ii G_jj + G_ij G_ji)/2 ; if i==j: G_ii^2 */
                if(i<j)diag=cij*cij*(E->G[i*d+i]*E->G[j*d+j]+E->G[i*d+j]*E->G[j*d+i])*0.5;
                else diag=E->G[i*d+i]*E->G[i*d+i];
                S->tv[E->diag_slot[a]]=-(diag+d2);
            }
            long tt=0;
            for(idx_t a=0;a<p;a++)for(idx_t bb=a+1;bb<p;bb++){
                int i=ii[a],j=jj[a],k=ii[bb],l=jj[bb];
                double cij=(i<j)?SQRT2:1.0, ckl=(k<l)?SQRT2:1.0;
                double val;
                if(k==l)val=cij*E->G[i*d+k]*E->G[j*d+k];
                else val=cij*(E->G[i*d*k]*E->G[j*d+l]+E->G[i*d+l]*E->G[j*d*k])/ckl;
                /* note: source (k,l) k<l uses /ckl ; but formula H[a,b]=cij*(...)/ckl */
                S->tv[E->off_slot[tt++]]=-val;
            }
            free(pos);free(ii);free(jj);
        }
    }
}

/* ---- H*dz per block, and rc ---- */
static void apply_Hdz(Solver *S,const double *dz,double *Hdz){
    for(int b=0;b<S->nblk;b++){
        Blk *E=&S->blk[b];idx_t st=E->start;const double *dv=dz+st;double *o=Hdz+st;double *s=S->s+st,*z=S->z+st;
        if(E->type==0){for(idx_t k=0;k<E->width;k++)o[k]=0;}
        else if(E->type==1){for(idx_t k=0;k<E->width;k++){double h=s[k]/z[k];o[k]=h*dv[k];}}
        else if(E->type==2){
            idx_t dd=E->width;
            double s0=s[0],z0=z[0],ss=0,zz=0,sz=0;
            for(idx_t k=1;k<dd;k++){ss+=s[k]*s[k];zz+=z[k]*z[k];sz+=s[k]*z[k];}
            double ds=s0*s0-ss,dz2=z0*z0-zz;if(ds<1e-300)ds=1e-300;if(dz2<1e-300)dz2=1e-300;
            double mus=sqrt(ds),muz=sqrt(dz2),g2=mus/muz;
            double c=(s0*z0-sz)/(mus*muz);double denom=sqrt(2*(1+c));if(denom<1e-300)denom=1e-300;
            double ab0=(s0/mus+z0/muz)/denom;
            double ad=ab0*dv[0];
            for(idx_t k=1;k<dd;k++){double abk=(s[k]/mus-z[k]/muz)/denom;ad+=abk*dv[k];}
            o[0]=g2*(2*ab0*ad-dv[0]);
            for(idx_t k=1;k<dd;k++){double abk=(s[k]/mus-z[k]/muz)/denom;o[k]=g2*(2*abk*ad+dv[k]);}
        }
        else if(E->type==3){
            double H[9],Hi[9];exp_hess(s,H);inv3(H,Hi);
            double sz=dot(s,z,3);if(sz<1e-300)sz=1e-300;double alpha=3.0/sz;
            for(int i=0;i<3;i++)o[i]=alpha*(Hi[i*3]*dv[0]+Hi[i*3+1]*dv[1]+Hi[i*3+2]*dv[2]);
        }
        else {
            int d=E->dim;
            double *dS=xmalloc(d*d*sizeof(double));smat(d,dv,dS);
            mmul(d,E->G,dS,E->tmp1);mmul(d,E->tmp1,E->G,E->tmp2);
            svec(d,E->tmp2,o);
            free(dS);
        }
    }
}

static void soc_P(const double *a,double deta,const double *x,double *y,idx_t d){
    double ax=0;for(idx_t k=0;k<d;k++)ax+=a[k]*x[k];
    for(idx_t k=0;k<d;k++)y[k]=2*a[k]*ax-deta*(k==0?x[k]:-x[k]);
}
/* compute rc for corrector/affine into S->rc, and rc_tau */
static void compute_rc(Solver *S,int affine,double sigma,double *rc_tau){
    double mu=S->mu;
    double target=affine?0.0:sigma*mu;
    for(int b=0;b<S->nblk;b++){
        Blk *E=&S->blk[b];idx_t st=E->start;double *s=S->s+st,*z=S->z+st,*rc=S->rc+st;
        if(E->type==0){for(idx_t k=0;k<E->width;k++)rc[k]=0;}
        else if(E->type==1){
            for(idx_t k=0;k<E->width;k++){
                double meh=affine?0.0:S->dsa[st+k]*S->dza[st+k];
                rc[k]=(target-s[k]*z[k]-meh)/z[k];
            }
        }
        else if(E->type==2){
            idx_t d=E->width;
            /* z^{-1} = Jz/det(z) */
            double zz=0;for(idx_t k=1;k<d;k++)zz+=z[k]*z[k];
            double dz=z[0]*z[0]-zz;if(dz<1e-300)dz=1e-300;
            /* rc = target*zinv - s - meh ; affine: -s */
            if(affine){for(idx_t k=0;k<d;k++)rc[k]=-s[k];}
            else {
                double s0=s[0],z0=z[0],ss=0,sz=0;
                for(idx_t k=1;k<d;k++){ss+=s[k]*s[k];sz+=s[k]*z[k];}
                double ds=s0*s0-ss;if(ds<1e-300)ds=1e-300;
                double mus=sqrt(ds),muz=sqrt(dz),g2=mus/muz;
                double c=(s0*z0-sz)/(mus*muz);double denom=sqrt(2*(1+c));if(denom<1e-300)denom=1e-300;
                double ab0=(s0/mus+z0/muz)/denom;
                double sg=sqrt(g2);
                double *wt=xmalloc(d*sizeof(double));
                double sw=sqrt(2*(ab0+1));
                wt[0]=sg*(ab0+1)/sw;
                for(idx_t k=1;k<d;k++){double abk=(s[k]/mus-z[k]/muz)/denom;wt[k]=sg*abk/sw;}
                double *wti=xmalloc(d*sizeof(double));
                for(idx_t k=0;k<d;k++)wti[k]=(k==0?wt[0]:-wt[k])/g2;
                double *dsp=xmalloc(d*sizeof(double)),*dzp=xmalloc(d*sizeof(double)),*vv=xmalloc(d*sizeof(double));
                soc_P(wti,1.0/g2,S->dsa+st,dsp,d);
                soc_P(wt,g2,S->dza+st,dzp,d);
                soc_P(wt,g2,z,vv,d);
                double *prod=xmalloc(d*sizeof(double));
                double dd=0;for(idx_t k=0;k<d;k++)dd+=dsp[k]*dzp[k];
                prod[0]=dd;
                for(idx_t k=1;k<d;k++)prod[k]=dsp[0]*dzp[k]+dzp[0]*dsp[k];
                double v0=vv[0],vbar2=0;for(idx_t k=1;k<d;k++)vbar2+=vv[k]*vv[k];
                double detv=v0*v0-vbar2;if(detv<1e-300)detv=1e-300;
                double vbr=0;for(idx_t k=1;k<d;k++)vbr+=vv[k]*prod[k];
                double y0=(v0*prod[0]-vbr)/detv;
                double *yy=xmalloc(d*sizeof(double));yy[0]=y0;
                for(idx_t k=1;k<d;k++)yy[k]=(prod[k]-y0*vv[k])/v0;
                double *meh=xmalloc(d*sizeof(double));
                soc_P(wt,g2,yy,meh,d);
                for(idx_t k=0;k<d;k++)rc[k]=target*(k==0?z[0]:-z[k])/dz - s[k] - meh[k];
                free(wt);free(wti);free(dsp);free(dzp);free(vv);free(prod);free(yy);free(meh);
            }
        }
        else if(E->type==3){
            if(affine){for(int k=0;k<3;k++)rc[k]=-s[k];}
            else {
                /* g = -grad F*(z) = (F_v(Tz),F_u(Tz),-e F_w(Tz)), Tz=(-z2,-z1,e z3) */
                double tz[3]={-z[1],-z[0],M_E*z[2]};
                double gg[3];exp_grad(tz,gg);
                double g[3]={gg[1],gg[0],-M_E*gg[2]};
                double sz=dot(s,z,3);if(fabs(sz)<1e-300)sz=1e-300;
                double meh=dot(S->dsa+st,S->dza+st,3)/sz;
                for(int k=0;k<3;k++)rc[k]=target*g[k]-s[k]-meh*s[k];
            }
        }
        else {
            int d=E->dim;
            if(affine){for(idx_t k=0;k<E->width;k++)rc[k]=-s[k];}
            else {
                /* Zinv */
                double *Zc=xmalloc(d*d*sizeof(double));memcpy(Zc,E->Z,d*d*sizeof(double));
                double *ev=xmalloc(d*sizeof(double)),*EV=xmalloc(d*d*sizeof(double));
                eig_sym(d,Zc,ev,EV);
                double *Zinv=xmalloc(d*d*sizeof(double));
                for(int i=0;i<d;i++)for(int j=0;j<d;j++){double s2=0;for(int k=0;k<d;k++){double e=ev[k];if(e<1e-300)e=1e-300;s2+=EV[i*d+k]*(1/e)*EV[j*d+k];}Zinv[i*d+j]=s2;}
                /* v = W Z W */
                mmul(d,E->W,E->Z,E->tmp1);mmul(d,E->tmp1,E->W,E->tmp2); /* tmp2=v */
                /* dS' = Win dS_aff Win ; dZ' = W dZ_aff W */
                double *dSa=xmalloc(d*d*sizeof(double)),*dZa=xmalloc(d*d*sizeof(double));
                smat(d,S->dsa+st,dSa);smat(d,S->dza+st,dZa);
                mmul(d,E->Win,dSa,E->tmp1);mmul(d,E->tmp1,E->Win,E->tmp3); /* dS' */
                double *dSp=xmalloc(d*d*sizeof(double));memcpy(dSp,E->tmp3,d*d*sizeof(double));
                mmul(d,E->W,dZa,E->tmp1);mmul(d,E->tmp1,E->W,E->tmp3); /* dZ' */
                /* P = (dS' dZ' + dZ' dS')/2 */
                double *Pm=xmalloc(d*d*sizeof(double)),*Qm=xmalloc(d*d*sizeof(double));
                mmul(d,dSp,E->tmp3,Pm);mmul(d,E->tmp3,dSp,Qm);
                for(int i=0;i<d*d;i++)Pm[i]=(Pm[i]+Qm[i]); /* =2*sym product; solve vY+Yv=Pm */
                /* eig v */
                double *vc=xmalloc(d*d*sizeof(double));memcpy(vc,E->tmp2,d*d*sizeof(double));
                double *evv=xmalloc(d*sizeof(double)),*EVv=xmalloc(d*d*sizeof(double));
                eig_sym(d,vc,evv,EVv);
                /* Y = EVv [ (EVv^T Pm EVv)_ij/(evv_i+evv_j) ] EVv^T */
                double *T1=xmalloc(d*d*sizeof(double)),*T2=xmalloc(d*d*sizeof(double));
                mmulT(d,EVv,Pm,T1);mmul(d,T1,EVv,T2);
                for(int i=0;i<d;i++)for(int j=0;j<d;j++){double de=evv[i]+evv[j];if(de<1e-300)de=1e-300;T2[i*d+j]/=de;}
                mmul(d,EVv,T2,T1);mmul(d,T1,EVv,E->tmp3); /* Y in tmp3 */
                /* term = W Y W */
                mmul(d,E->W,E->tmp3,E->tmp1);mmul(d,E->tmp1,E->W,E->tmp2);
                double *rcm=xmalloc(E->width*sizeof(double));
                svec(d,E->tmp2,rcm);
                double *zinv_s=xmalloc(E->width*sizeof(double));svec(d,Zinv,zinv_s);
                for(idx_t k=0;k<E->width;k++)rc[k]=target*zinv_s[k]-s[k]-rcm[k];
                free(Zc);free(ev);free(EV);free(Zinv);free(dSa);free(dZa);free(dSp);free(Pm);free(Qm);free(vc);free(evv);free(EVv);free(T1);free(T2);free(rcm);free(zinv_s);
            }
        }
    }
    /* tau complementarity */
    double tau=*S->tau,kappa=*S->kappa;
    double meh=affine?0.0:S->dtaua*S->dkappaa;
    *rc_tau=target-tau*kappa-meh;
}

/* ---- step length ---- */
static double step_nonneg(const double *s,const double *ds,idx_t n){
    double a=1e300;
    for(idx_t i=0;i<n;i++)if(ds[i]<0){double t=-s[i]/ds[i];if(t<a)a=t;}
    return a>1?1:a;
}
static double step_soc(const double *s,const double *ds,idx_t d){
    double s0=s[0],d0=ds[0];
    double ss=0,dd=0,sd=0;
    for(idx_t k=1;k<d;k++){ss+=s[k]*s[k];dd+=ds[k]*ds[k];sd+=s[k]*ds[k];}
    double A=d0*d0-dd,B=2*(s0*d0-sd),C=s0*s0-ss;
    double amax=1e300;
    if(d0<0){double t=-s0/d0;if(t<amax)amax=t;}
    if(fabs(A)>1e-300){
        double disc=B*B-4*A*C;
        if(disc>=0){
            double sq=sqrt(disc);
            double r1=(-B-sq)/(2*A),r2=(-B+sq)/(2*A);
            double r;
            if(A>0){r=r1>0?r1:r2;} else {r=r2>0?r2:r1;}
            /* smallest positive root */
            double rp=1e300;
            if(r1>1e-12&&r1<rp)rp=r1;
            if(r2>1e-12&&r2<rp)rp=r2;
            if(rp<amax)amax=rp;
        }
    } else if(fabs(B)>1e-300){
        double r=-C/B;if(r>1e-12&&r<amax)amax=r;
    }
    return amax>1?1:amax;
}
static int exp_member(const double *v){
    double u=v[0],vv=v[1],w=v[2];
    if(vv>1e-300&&w>1e-300){return u < vv*log(w/vv);}
    if(fabs(vv)<=1e-300)return u<=0&&w>=0;
    return 0;
}
static double step_exp(const double *s,const double *ds){
    double t[3]={s[0]+ds[0],s[1]+ds[1],s[2]+ds[2]};
    if(exp_member(t))return 1.0;
    double lo=0,hi=1;
    for(int it=0;it<60;it++){
        double mid=0.5*(lo+hi);
        double tt[3]={s[0]+mid*ds[0],s[1]+mid*ds[1],s[2]+mid*ds[2]};
        if(exp_member(tt))lo=mid;else hi=mid;
    }
    return lo;
}
static double step_psd(Blk *E,const double *s,const double *ds){
    int d=E->dim;
    smat(d,s,E->S);smat(d,ds,E->tmp1);
    /* chol S = L L^T ; then eig L^{-1} dS L^{-T} */
    double *L=xmalloc(d*d*sizeof(double));
    for(int i=0;i<d*d;i++)L[i]=0;
    for(int i=0;i<d;i++){
        for(int j=0;j<=i;j++){
            double sum=E->S[i*d+j];
            for(int k=0;k<j;k++)sum-=L[i*d+k]*L[j*d+k];
            if(i==j){if(sum<1e-300)sum=1e-300;L[i*d+i]=sqrt(sum);}
            else L[i*d+j]=sum/L[j*d+j];
        }
    }
    /* T = L^{-1} dS L^{-T} : solve L X = dS -> X ; then X L^{-T} */
    double *X=xmalloc(d*d*sizeof(double));
    for(int col=0;col<d;col++){
        for(int i=0;i<d;i++){
            double sum=E->tmp1[i*d+col];
            for(int k=0;k<i;k++)sum-=L[i*d+k]*X[k*d+col];
            X[i*d+col]=sum/L[i*d+i];
        }
    }
    /* T = X L^{-T} : T[i,j] = sum_k X[i,k] (L^{-T})[k,j] = solve L^T ... T = X * inv(L)^T */
    double *T=xmalloc(d*d*sizeof(double));
    for(int row=0;row<d;row++){
        for(int j=0;j<d;j++){
            double sum=X[row*d+j];
            for(int k=j+1;k<d;k++)sum-=T[row*d+k]*L[k*d+j];
            T[row*d+j]=sum/L[j*d+j];
        }
    }
    /* symmetrize */
    for(int i=0;i<d;i++)for(int j=0;j<i;j++){double av=0.5*(T[i*d+j]+T[j*d+i]);T[i*d+j]=av;T[j*d+i]=av;}
    double *ev=xmalloc(d*sizeof(double)),*EV=xmalloc(d*d*sizeof(double));
    eig_sym(d,T,ev,EV);
    double lmin=ev[0];for(int i=1;i<d;i++)if(ev[i]<lmin)lmin=ev[i];
    double amax=1.0;
    if(lmin<-1e-300)amax=-1.0/lmin;
    free(L);free(X);free(T);free(ev);free(EV);
    return amax>1?1:amax;
}

static double primal_step(Solver *S,const double *ds,double dtau){
    double a=1.0;
    for(int b=0;b<S->nblk;b++){
        Blk *E=&S->blk[b];idx_t st=E->start;
        if(E->type==0)continue;
        double t;
        if(E->type==1)t=step_nonneg(S->s+st,ds+st,E->width);
        else if(E->type==2)t=step_soc(S->s+st,ds+st,E->width);
        else if(E->type==3)t=step_exp(S->s+st,ds+st);
        else t=step_psd(E,S->s+st,ds+st);
        if(t<a)a=t;
    }
    if(dtau<0){double t=-(*S->tau)/dtau;if(t<a)a=t;}
    return a;
}
static double dual_step(Solver *S,const double *dz,double dkappa){
    double a=1.0;
    for(int b=0;b<S->nblk;b++){
        Blk *E=&S->blk[b];idx_t st=E->start;
        if(E->type==0)continue;
        double t;
        if(E->type==1)t=step_nonneg(S->z+st,dz+st,E->width);
        else if(E->type==2)t=step_soc(S->z+st,dz+st,E->width);
        else if(E->type==3)t=step_exp(S->z+st,dz+st);
        else t=step_psd(E,S->z+st,dz+st);
        if(t<a)a=t;
    }
    if(dkappa<0){double t=-(*S->kappa)/dkappa;if(t<a)a=t;}
    return a;
}

/* ---- termination checks (mirror checker) ---- */
static double exp_violation(double a,double b,double c){
    double boundary=fabs(b);if(a>boundary)boundary=a;if(-c>boundary)boundary=-c;
    if(b>0&&c>0){double interior=a-b*(log(c)-log(b));if(interior<0)interior=0;if(interior<boundary)return interior;return boundary;}
    return boundary;
}
static double cone_defect(Solver *S,const double *v,int dual){
    Problem *P=S->P;double worst=0;
    for(int b=0;b<P->nblk;b++){
        Block *B=&P->blk[b];const double *blk=v+B->start;
        double nrm=ninf(blk,B->width);double sc=nrm>1?nrm:1;
        double viol=0;
        if(B->type==0){viol=dual?0:ninf(blk,B->width)/sc;}
        else if(B->type==1){double mn=1e300;for(idx_t k=0;k<B->width;k++){double u=blk[k]/sc;if(u<mn)mn=u;}viol=mn<0?-mn:0;}
        else if(B->type==2){double u0=blk[0]/sc;double n2=0;for(idx_t k=1;k<B->width;k++){double u=blk[k]/sc;n2+=u*u;}viol=sqrt(n2)-u0;if(viol<0)viol=0;}
        else if(B->type==3){double u[3]={blk[0]/sc,blk[1]/sc,blk[2]/sc};
            if(dual)viol=exp_violation(-u[1],-u[0],M_E*u[2]);else viol=exp_violation(u[0],u[1],u[2]);}
        else {
            int d=B->dim;double *M=xmalloc(d*d*sizeof(double));
            double *uu=xmalloc(B->width*sizeof(double));
            for(idx_t k=0;k<B->width;k++)uu[k]=blk[k]/sc;
            smat(d,uu,M);
            double *ev=xmalloc(d*sizeof(double)),*EV=xmalloc(d*d*sizeof(double));
            eig_sym(d,M,ev,EV);
            double lmin=ev[0];for(int i=1;i<d;i++)if(ev[i]<lmin)lmin=ev[i];
            viol=lmin<0?-lmin:0;
            free(M);free(uu);free(ev);free(EV);
        }
        if(viol>worst)worst=viol;
    }
    return worst;
}

static int check_solved(Solver *S,const double *x,const double *s,const double *z){
    Problem *P=S->P;idx_t n=P->n,m=P->m;
    double *Ax=xmalloc(m*sizeof(double)),*Px=xmalloc(n*sizeof(double)),*ATz=xmalloc(n*sizeof(double));
    Amv(P,x,Ax);Pmv(P,x,Px);ATmv(P,z,ATz);
    double rp=0,rd=0;
    double nAx=ninf(Ax,m),ns=ninf(s,m),nb=ninf(P->b,m);
    double den=1+(nAx>ns?(nAx>nb?nAx:nb):(ns>nb?ns:nb));
    for(idx_t i=0;i<m;i++){double r=Ax[i]+s[i]-P->b[i];if(fabs(r)>rp)rp=fabs(r);}
    rp/=den;
    double nPx=ninf(Px,n),nq=ninf(P->q,n),nATz=ninf(ATz,n);
    double den2=1+(nPx>nq?(nPx>nATz?nPx:nATz):(nq>nATz?nq:nATz));
    for(idx_t i=0;i<n;i++){double r=Px[i]+P->q[i]+ATz[i];if(fabs(r)>rd)rd=fabs(r);}
    rd/=den2;
    double xPx=dot(x,Px,n),qx=dot(P->q,x,n),bz=dot(P->b,z,m);
    double p0=0.5*xPx+qx,d0=-0.5*xPx-bz;
    double gap=fabs(p0-d0),scale=fabs(p0)>fabs(d0)?fabs(p0):fabs(d0);
    double gaprel=scale>0?gap/scale:0;
    free(Ax);free(Px);free(ATz);
    if(getenv("DBG"))fprintf(stderr,"  chk rp=%.2e rd=%.2e gap=%.2e gaprel=%.2e cdp=%.2e cdd=%.2e\n",rp,rd,gap,gaprel,cone_defect(S,s,0),cone_defect(S,z,1));
    if(rp>8e-9||rd>8e-9)return 0;
    if(!(gap<=8e-9||gaprel<=8e-9))return 0;
    if(cone_defect(S,s,0)>8e-9)return 0;
    if(cone_defect(S,z,1)>8e-9)return 0;
    return 1;
}

static int check_primal_infeas(Solver *S,const double *z){
    Problem *P=S->P;idx_t m=P->m,n=P->n;
    double nz=ninf(z,m),nb=ninf(P->b,m);
    if(!(nz>0&&nb>0))return 0;
    double *zh=xmalloc(m*sizeof(double));for(idx_t i=0;i<m;i++)zh[i]=z[i]/nz;
    double margin=0;for(idx_t i=0;i<m;i++)margin-= (P->b[i]/nb)*zh[i];
    if(!(margin>1e-8)){free(zh);return 0;}
    double a=0;const CSC*AM=&P->A;for(idx_t k=0;k<AM->p[n];k++){double v=fabs(AM->x[k]);if(v>a)a=v;}if(a==0)a=1;
    double *ATz=xmalloc(n*sizeof(double));ATmv(P,zh,ATz);
    double lin=ninf(ATz,n)/a;
    free(ATz);
    double cd=cone_defect(S,zh,1);
    free(zh);
    return lin/margin<=1e-8 && cd/margin<=1e-8;
}
static int check_dual_infeas(Solver *S,const double *x,const double *s){
    Problem *P=S->P;idx_t m=P->m,n=P->n;
    double nx=ninf(x,n),nq=ninf(P->q,n);
    if(!(nx>0&&nq>0))return 0;
    double *xh=xmalloc(n*sizeof(double)),*sh=xmalloc(m*sizeof(double));
    for(idx_t i=0;i<n;i++)xh[i]=x[i]/nx;
    double margin=0;for(idx_t i=0;i<n;i++)margin-=(P->q[i]/nq)*xh[i];
    if(!(margin>1e-8)){free(xh);free(sh);return 0;}
    double h=0;const CSC*PM=&P->P;for(idx_t k=0;k<PM->p[n];k++){double v=fabs(PM->x[k]);if(v>h)h=v;}if(h==0)h=1;
    double a=0;const CSC*AM=&P->A;for(idx_t k=0;k<AM->p[n];k++){double v=fabs(AM->x[k]);if(v>a)a=v;}if(a==0)a=1;
    for(idx_t i=0;i<m;i++)sh[i]=s[i]/nx/a;
    double *Px=xmalloc(n*sizeof(double));Pmv(P,xh,Px);
    double quad=ninf(Px,n)/h;
    double *Ax=xmalloc(m*sizeof(double));Amv(P,xh,Ax);
    double lin=0;for(idx_t i=0;i<m;i++){double r=Ax[i]/a+sh[i];if(fabs(r)>lin)lin=fabs(r);}
    double cd=cone_defect(S,sh,0);
    /* note: checker scales s by a then cone defect; sh here = s/nx, need sh/a */
    free(Px);free(Ax);free(xh);free(sh);
    return quad/margin<=1e-8 && lin/margin<=1e-8 && cd/margin<=1e-8;
}

/* ---- solve linear system for direction ---- */
static void solve_dir(Solver *S,const double *r1,const double *r2p,double r3p,double rc_tau,
                      double *dx,double *dz,double *dtau,double *dkappa){
    idx_t n=S->n,m=S->m,N=S->N;
    double *rhs=S->rhs;
    for(idx_t i=0;i<N;i++)rhs[i]=0;
    for(idx_t i=0;i<n;i++)rhs[i]=r1[i];
    for(idx_t i=0;i<m;i++)rhs[n+i]=r2p[i];
    kkt_solve(&S->K,rhs,S->u);
    /* refinement */
    kkt_matvec(&S->K,S->u,S->work);
    for(idx_t i=0;i<N;i++)rhs[i]-=S->work[i];
    kkt_solve(&S->K,rhs,S->work);
    for(idx_t i=0;i<N;i++)S->u[i]+=S->work[i];
    /* v solve: rhs = [q;-b;0] */
    for(idx_t i=0;i<N;i++)rhs[i]=0;
    for(idx_t i=0;i<n;i++)rhs[i]=S->P->q[i];
    for(idx_t i=0;i<m;i++)rhs[n+i]=-S->P->b[i];
    kkt_solve(&S->K,rhs,S->v);
    kkt_matvec(&S->K,S->v,S->work);
    for(idx_t i=0;i<N;i++)rhs[i]-=S->work[i];
    kkt_solve(&S->K,rhs,S->work);
    for(idx_t i=0;i<N;i++)S->v[i]+=S->work[i];
    double tau=*S->tau,kappa=*S->kappa;
    double zeta=kappa/tau;
    double qu=0,bu=0,qv=0,bv=0;
    for(idx_t i=0;i<n;i++){qu+=S->P->q[i]*S->u[i];qv+=S->P->q[i]*S->v[i];}
    for(idx_t i=0;i<m;i++){bu+=S->P->b[i]*S->u[n+i];bv+=S->P->b[i]*S->v[n+i];}
    double dt=(r3p+qu+bu)/(qv+bv+zeta);
    *dtau=dt;
    for(idx_t i=0;i<n;i++)dx[i]=S->u[i]-S->v[i]*dt;
    for(idx_t i=0;i<m;i++)dz[i]=S->u[n+i]-S->v[n+i]*dt;
    *dkappa=(rc_tau-kappa*dt)/tau;
}

static void compute_residuals(Solver *S,double *r1,double *r2,double *r3){
    Problem *P=S->P;idx_t n=S->n,m=S->m;
    double tau=*S->tau;
    double *Px=xmalloc(n*sizeof(double)),*ATz=xmalloc(n*sizeof(double)),*Ax=xmalloc(m*sizeof(double));
    Pmv(P,S->x,Px);ATmv(P,S->z,ATz);Amv(P,S->x,Ax);
    for(idx_t i=0;i<n;i++)r1[i]=-(Px[i]+P->q[i]*tau+ATz[i]);
    for(idx_t i=0;i<m;i++)r2[i]=-(Ax[i]+S->s[i]-P->b[i]*tau);
    double qxx=dot(P->q,S->x,n),bz=dot(P->b,S->z,m);
    *r3=qxx+bz+*S->kappa;
    free(Px);free(ATz);free(Ax);
}

static double compute_mu(Solver *S){
    Problem *P=S->P;
    double sz=dot(S->s,S->z,S->m);
    double nu=0;
    for(int b=0;b<P->nblk;b++){Block*B=&P->blk[b];
        if(B->type==1)nu+=B->dim;else if(B->type==2)nu+=2;else if(B->type==3)nu+=3;else if(B->type==4)nu+=B->dim;}
    return (sz+(*S->tau)*(*S->kappa))/(nu+1);
}


static int solve_direct(Problem *P,const char *outpath,double time_cap){
    Solver S;memset(&S,0,sizeof(S));S.P=P;
    S.delta1=1e-8;S.delta2=1e-8;S.t_start=now_sec();
    idx_t n=P->n,m=P->m;
    S.x=xcalloc(n,sizeof(double));S.z=xmalloc(m*sizeof(double));S.s=xmalloc(m*sizeof(double));
    S.tau=xmalloc(sizeof(double));S.kappa=xmalloc(sizeof(double));*S.tau=1;*S.kappa=1;
    init_cones(&S);assemble(&S);
    S.r1=xmalloc(n*sizeof(double));S.r2=xmalloc(m*sizeof(double));S.rc=xmalloc(m*sizeof(double));
    S.u=xmalloc(S.N*sizeof(double));S.v=xmalloc(S.N*sizeof(double));
    S.dx=xmalloc(n*sizeof(double));S.dz=xmalloc(m*sizeof(double));S.ds=xmalloc(m*sizeof(double));
    S.dxa=xmalloc(n*sizeof(double));S.dza=xmalloc(m*sizeof(double));S.dsa=xmalloc(m*sizeof(double));
    S.work=xmalloc(S.N*sizeof(double));S.rhs=xmalloc(S.N*sizeof(double));
    double *Hdz=xmalloc(m*sizeof(double)),*r2p=xmalloc(m*sizeof(double));
    double nu=0;for(int b=0;b<P->nblk;b++){Block*B=&P->blk[b];if(B->type==1)nu+=B->dim;else if(B->type==2)nu+=2;else if(B->type==3)nu+=3;else if(B->type==4)nu+=B->dim;}
    if(nu<1)nu=1;
    const char *status="unresolved";double *outx=NULL,*outs=NULL,*outz=NULL;
    for(int iter=0;iter<200;iter++){
        if(now_sec()-S.t_start>time_cap)break;
        S.mu=dot(S.s,S.z,m)/nu;
        fill_H(&S);kkt_refill(&S.K,S.tv);
        if(kkt_factor(&S.K))break;
        double *Px=xmalloc(n*sizeof(double)),*ATz=xmalloc(n*sizeof(double)),*Ax=xmalloc(m*sizeof(double));
        Pmv(P,S.x,Px);ATmv(P,S.z,ATz);Amv(P,S.x,Ax);
        for(idx_t i=0;i<n;i++)S.r1[i]=-(Px[i]+P->q[i]+ATz[i]);
        for(idx_t i=0;i<m;i++)S.r2[i]=-(Ax[i]+S.s[i]-P->b[i]);
        free(Px);free(ATz);free(Ax);
        double rc_tau;
        compute_rc(&S,1,0,&rc_tau);
        for(idx_t i=0;i<m;i++)r2p[i]=S.r2[i]-S.rc[i];
        for(idx_t i=0;i<S.N;i++)S.rhs[i]=0;
        for(idx_t i=0;i<n;i++)S.rhs[i]=S.r1[i];
        for(idx_t i=0;i<m;i++)S.rhs[n+i]=r2p[i];
        kkt_solve(&S.K,S.rhs,S.dxa);
        /* dxa holds [dx;dz;dy]; extract */
        double *dxa_x=xmalloc(n*sizeof(double)),*dxa_z=xmalloc(m*sizeof(double));
        for(idx_t i=0;i<n;i++)dxa_x[i]=S.dxa[i];
        for(idx_t i=0;i<m;i++)dxa_z[i]=S.dxa[n+i];
        apply_Hdz(&S,dxa_z,Hdz);
        for(idx_t i=0;i<m;i++)S.dsa[i]=S.rc[i]-Hdz[i];
        memcpy(S.dza,dxa_z,m*sizeof(double));
        double ap=primal_step(&S,S.dsa,0),ad=dual_step(&S,S.dza,0);
        double apf=0.99*ap,adf=0.99*ad;if(apf>1)apf=1;if(adf>1)adf=1;
        double szaff=0;for(idx_t i=0;i<m;i++)szaff+=(S.s[i]+apf*S.dsa[i])*(S.z[i]+adf*S.dza[i]);
        double mu_aff=szaff/nu;double sigma=mu_aff/S.mu;sigma=sigma*sigma*sigma;if(sigma>1)sigma=1;if(sigma<0)sigma=0;
        compute_rc(&S,0,sigma,&rc_tau);
        for(idx_t i=0;i<m;i++)r2p[i]=S.r2[i]-S.rc[i];
        for(idx_t i=0;i<S.N;i++)S.rhs[i]=0;
        for(idx_t i=0;i<n;i++)S.rhs[i]=S.r1[i];
        for(idx_t i=0;i<m;i++)S.rhs[n+i]=r2p[i];
        kkt_solve(&S.K,S.rhs,S.u);
        kkt_matvec(&S.K,S.u,S.work);
        for(idx_t i=0;i<S.N;i++)S.rhs[i]-=S.work[i];
        kkt_solve(&S.K,S.rhs,S.work);
        for(idx_t i=0;i<S.N;i++)S.u[i]+=S.work[i];
        for(idx_t i=0;i<n;i++)S.dx[i]=S.u[i];
        for(idx_t i=0;i<m;i++)S.dz[i]=S.u[n+i];
        apply_Hdz(&S,S.dz,Hdz);
        for(idx_t i=0;i<m;i++)S.ds[i]=S.rc[i]-Hdz[i];
        double cp=primal_step(&S,S.ds,0),cd=dual_step(&S,S.dz,0);
        double cpf=0.995*cp,cdf=0.995*cd;if(cpf>1)cpf=1;if(cdf>1)cdf=1;
        for(idx_t i=0;i<n;i++)S.x[i]+=cpf*S.dx[i];
        for(idx_t i=0;i<m;i++){S.s[i]+=cpf*S.ds[i];S.z[i]+=cdf*S.dz[i];}
        free(dxa_x);free(dxa_z);
        if(check_solved(&S,S.x,S.s,S.z)){status="solved";outx=xmalloc(n*sizeof(double));memcpy(outx,S.x,n*sizeof(double));outs=xmalloc(m*sizeof(double));memcpy(outs,S.s,m*sizeof(double));outz=xmalloc(m*sizeof(double));memcpy(outz,S.z,m*sizeof(double));break;}
    }
    write_result(outpath,status,outx,outs,outz,n,m);
    return 0;
}
int solve_problem(Problem *P,const char *outpath,double time_cap){
    Solver S;memset(&S,0,sizeof(S));S.P=P;
    S.delta1=1e-8;S.delta2=1e-8;
    S.t_start=now_sec();
    idx_t n=P->n,m=P->m;
    S.x=xcalloc(n,sizeof(double));S.z=xmalloc(m*sizeof(double));S.s=xmalloc(m*sizeof(double));
    S.tau=xmalloc(sizeof(double));S.kappa=xmalloc(sizeof(double));
    *S.tau=1;*S.kappa=1;
    init_cones(&S);
    assemble(&S);
    S.r1=xmalloc(n*sizeof(double));S.r2=xmalloc(m*sizeof(double));S.rc=xmalloc(m*sizeof(double));
    S.u=xmalloc(S.N*sizeof(double));S.v=xmalloc(S.N*sizeof(double));
    S.dx=xmalloc(n*sizeof(double));S.dz=xmalloc(m*sizeof(double));S.ds=xmalloc(m*sizeof(double));
    S.dxa=xmalloc(n*sizeof(double));S.dza=xmalloc(m*sizeof(double));S.dsa=xmalloc(m*sizeof(double));
    S.work=xmalloc(S.N*sizeof(double));S.rhs=xmalloc(S.N*sizeof(double));
    double *Hdz=xmalloc(m*sizeof(double));
    double *r2p=xmalloc(m*sizeof(double));
    const char *status="unresolved";
    double *outx=NULL,*outs=NULL,*outz=NULL;
    for(int iter=0;iter<300;iter++){
        if(now_sec()-S.t_start>time_cap)break;
        S.mu=compute_mu(&S);
        fill_H(&S);
        kkt_refill(&S.K,S.tv);
        if(kkt_factor(&S.K)){break;}
        double r3;
        compute_residuals(&S,S.r1,S.r2,&r3);
        /* affine */
        double rc_tau_a;
        compute_rc(&S,1,0,&rc_tau_a);
        for(idx_t i=0;i<m;i++)r2p[i]=S.r2[i]-S.rc[i];
        double r3p_a=r3+rc_tau_a/(*S.tau);
        solve_dir(&S,S.r1,r2p,r3p_a,rc_tau_a,S.dxa,S.dza,&S.dtaua,&S.dkappaa);
        apply_Hdz(&S,S.dza,Hdz);
        for(idx_t i=0;i<m;i++)S.dsa[i]=S.rc[i]-Hdz[i];
        /* affine step & sigma */
        double ap=primal_step(&S,S.dsa,S.dtaua),ad=dual_step(&S,S.dza,S.dkappaa);
        double apf=0.99*ap,adf=0.99*ad; if(apf>1)apf=1;if(adf>1)adf=1;
        /* mu_aff via complementarity */
        double szaff=0;
        for(idx_t i=0;i<m;i++)szaff+=(S.s[i]+apf*S.dsa[i])*(S.z[i]+adf*S.dza[i]);
        double tauaff=(*S.tau)+apf*S.dtaua,kapaff=(*S.kappa)+adf*S.dkappaa;
        double nu=0;
        for(int b=0;b<P->nblk;b++){Block*B=&P->blk[b];if(B->type==1)nu+=B->dim;else if(B->type==2)nu+=2;else if(B->type==3)nu+=3;else if(B->type==4)nu+=B->dim;}
        double mu_aff=(szaff+tauaff*kapaff)/(nu+1);
        double sigma=mu_aff/S.mu;sigma=sigma*sigma*sigma;if(sigma>1)sigma=1;if(sigma<0)sigma=0;
        /* corrector */
        double rc_tau_c;
        compute_rc(&S,0,sigma,&rc_tau_c);
        for(idx_t i=0;i<m;i++)r2p[i]=S.r2[i]-S.rc[i];
        double r3p_c=r3+rc_tau_c/(*S.tau);
        solve_dir(&S,S.r1,r2p,r3p_c,rc_tau_c,S.dx,S.dz,&S.dtau,&S.dkappa);
        apply_Hdz(&S,S.dz,Hdz);
        for(idx_t i=0;i<m;i++)S.ds[i]=S.rc[i]-Hdz[i];
        double cp=primal_step(&S,S.ds,S.dtau),cd=dual_step(&S,S.dz,S.dkappa);
        double cpf=0.995*cp,cdf=0.995*cd;if(cpf>1)cpf=1;if(cdf>1)cdf=1;
        /* update */
        for(idx_t i=0;i<n;i++)S.x[i]+=cpf*S.dx[i];
        for(idx_t i=0;i<m;i++){S.s[i]+=cpf*S.ds[i];S.z[i]+=cdf*S.dz[i];}
        *S.tau+=cpf*S.dtau;*S.kappa+=cdf*S.dkappa;
        /* termination */
        if(getenv("DBG")){
            double *Ax=xmalloc(m*sizeof(double)),*Px=xmalloc(n*sizeof(double)),*ATz=xmalloc(n*sizeof(double));
            Amv(P,S.x,Ax);Pmv(P,S.x,Px);ATmv(P,S.z,ATz);
            double rp=0,rd=0;for(idx_t i=0;i<m;i++){double r=Ax[i]+S.s[i]-P->b[i]*(*S.tau);if(fabs(r)>rp)rp=fabs(r);}
            for(idx_t i=0;i<n;i++){double r=Px[i]+P->q[i]*(*S.tau)+ATz[i];if(fabs(r)>rd)rd=fabs(r);}
            fprintf(stderr,"it%d mu=%.3e tau=%.3e kap=%.3e rp=%.2e rd=%.2e sz=%.3e cp=%.3f cd=%.3f sig=%.3f\n",iter,S.mu,*S.tau,*S.kappa,rp,rd,dot(S.s,S.z,m),cp,cd,sigma);
            free(Ax);free(Px);free(ATz);
        }
        double tau=*S.tau,kappa=*S.kappa;
        if(tau>1e-12){
            double *xh=xmalloc(n*sizeof(double)),*sh=xmalloc(m*sizeof(double)),*zh=xmalloc(m*sizeof(double));
            for(idx_t i=0;i<n;i++)xh[i]=S.x[i]/tau;
            for(idx_t i=0;i<m;i++){sh[i]=S.s[i]/tau;zh[i]=S.z[i]/tau;}
            if(check_solved(&S,xh,sh,zh)){status="solved";outx=xh;outs=sh;outz=zh;break;}
            free(xh);free(sh);free(zh);
        }
        if(kappa>1e-12 && tau<1e-6*kappa){
            if(check_primal_infeas(&S,S.z)){status="primal_infeasible";outz=xmalloc(m*sizeof(double));memcpy(outz,S.z,m*sizeof(double));break;}
            if(check_dual_infeas(&S,S.x,S.s)){status="dual_infeasible";outx=xmalloc(n*sizeof(double));memcpy(outx,S.x,n*sizeof(double));outs=xmalloc(m*sizeof(double));memcpy(outs,S.s,m*sizeof(double));break;}
        }
    }
    write_result(outpath,status,outx,outs,outz,n,m);
    return 0;
}

int main(int argc,char **argv){
    const char *in=NULL,*out=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--input")==0&&i+1<argc)in=argv[++i];
        else if(strcmp(argv[i],"--output")==0&&i+1<argc)out=argv[++i];
    }
    if(!in||!out){fprintf(stderr,"usage: --input IN --output OUT\n");return 2;}
    Problem P;
    if(load_problem(in,&P)){fprintf(stderr,"load failed\n");return 2;}
    int only_lin=1;
    for(int i=0;i<P.nblk;i++)if(P.blk[i].type>1)only_lin=0;
    if(P.P.p[P.n]>0)solve_direct(&P,out,280.0);
    else if(only_lin)solve_problem(&P,out,280.0);
    else solve_direct(&P,out,280.0);
    return 0;
}
