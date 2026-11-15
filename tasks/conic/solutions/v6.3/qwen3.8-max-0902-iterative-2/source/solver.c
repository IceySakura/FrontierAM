#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "npz.h"
#include "sparse.h"

extern void dsyevd_(const char *, const char *, const int *, double *, const int *, double *,
                    double *, const int *, int *, const int *, int *);
extern void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *,
                   const double *, const int *, const double *, const int *, const double *, double *, const int *);

#define CZ 0
#define CNONNEG 1
#define CSOC 2
#define CEXP 3
#define CPSD 4
#define SQ2 1.41421356237309504880
#define ISQ2 0.70710678118654752440
#define BIG 1e30

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static double t_start=0.0, time_limit=1e30;
static int dbg=0;
static void *xmalloc(size_t n){void*p=malloc(n?n:1);if(!p){fprintf(stderr,"oom\n");exit(3);}return p;}
static void *xcalloc(size_t n,size_t s){void*p=calloc(n?n:1,s);if(!p){fprintf(stderr,"oom\n");exit(3);}return p;}

static void dmm(int d,const double*A,const double*B,double*C){char nn='N';double o=1,z=0;dgemm_(&nn,&nn,&d,&d,&d,&o,A,&d,B,&d,&z,C,&d);}
static void dmmT(int d,const double*A,const double*B,double*C){char nn='N',tt='T';double o=1,z=0;dgemm_(&nn,&tt,&d,&d,&d,&o,A,&d,B,&d,&z,C,&d);}
static int eig_sym(int d,double*A,double*ev,double*w,int lw,int*iw,int liw){int info=0;char j='V',u='L';dsyevd_(&j,&u,&d,A,&d,ev,w,&lw,iw,&liw,&info);return info;}
static void eig_wsq(int d,int*lw,int*liw){double wk;int iwk,info=0,a=-1,b=-1;char j='V',u='L';int dd=d<1?1:d;
  double*t=(double*)xcalloc((size_t)dd*dd,sizeof(double));double*e=(double*)xcalloc(dd,sizeof(double));
  dsyevd_(&j,&u,&dd,t,&dd,e,&wk,&a,&iwk,&b,&info);*lw=(int)wk;*liw=iwk;free(t);free(e);}

/* ---------- cone primitives ---------- */
static void soc_inv(const double*a,int d,double*o){double det=a[0]*a[0];for(int i=1;i<d;i++)det-=a[i]*a[i];
  if(fabs(det)<1e-300)det=det<0?-1e-300:1e-300; o[0]=a[0]/det;for(int i=1;i<d;i++)o[i]=-a[i]/det;}
static void soc_jord(const double*a,const double*b,int d,double*o){double dot=0;for(int i=1;i<d;i++)dot+=a[i]*b[i];
  o[0]=a[0]*b[0]+dot;for(int i=1;i<d;i++)o[i]=a[0]*b[i]+b[0]*a[i];}
static int soc_sqrt(const double*a,int d,double*o){double nb=0;for(int i=1;i<d;i++)nb+=a[i]*a[i];nb=sqrt(nb);
  double p=a[0]+nb,q=a[0]-nb;if(p<0||q<-1e-10*(p>0?p:1))return -1;if(q<0)q=0;
  double r=sqrt(p),t=sqrt(q);o[0]=0.5*(r+t);double cf=nb>1e-300?0.5*(r-t)/nb:0.0;
  for(int i=1;i<d;i++)o[i]=cf*a[i];return 0;}
static double soc_maxstep(const double*s,const double*ds,int d){
  double c=s[0]*s[0];for(int i=1;i<d;i++)c-=s[i]*s[i];
  double a=ds[0]*ds[0];for(int i=1;i<d;i++)a-=ds[i]*ds[i];
  double bb=s[0]*ds[0];for(int i=1;i<d;i++)bb-=s[i]*ds[i];bb*=2.0;
  if(c<=0)return 0.0;
  if(a<=1e-300)return bb>=0?BIG:-c/bb;
  double D=bb*bb-4*a*c;if(D<0)return BIG;double den=-bb+sqrt(D);if(den<=1e-300)return BIG;return 2*c/den;}
static double exp_viol(double a,double b,double c){double B=fabs(b);double ap=a>0?a:0,cm=(-c)>0?(-c):0;
  if(ap>B)B=ap;if(cm>B)B=cm;if(b>0&&c>0){double inn=a-b*(log(c)-log(b));if(inn<0)inn=0;return inn<B?inn:B;}return B;}
static int exp_inside(const double*t){double u=t[0],v=t[1],w=t[2];if(v<=0)return 0;double r=u/v;if(r>700)return 0;
  return (w-v*exp(r))>0;}
static double exp_maxstep(const double*s,const double*ds){
  if(!exp_inside(s))return 0.0;
  double hi=BIG;
  if(ds[1]<0)hi=-s[1]/ds[1];
  if(ds[2]<0){double h2=-s[2]/ds[2];if(h2<hi)hi=h2;}
  if(hi>=1e29){double t[3]={s[0]+1e18*ds[0],s[1]+1e18*ds[1],s[2]+1e18*ds[2]};if(exp_inside(t))return BIG;hi=1e18;}
  double lo=0.0;
  for(int it=0;it<70;it++){double mid=0.5*(lo+hi);double t[3]={s[0]+mid*ds[0],s[1]+mid*ds[1],s[2]+mid*ds[2]};
    if(exp_inside(t))lo=mid;else hi=mid; if(hi-lo<1e-13*(1+lo))break;}
  return lo;}
static void svec_to_mat(int d,const double*v,double*M){int p=0;for(int c=0;c<d;c++)for(int r=0;r<=c;r++){
  double val=v[p++];if(r!=c)val*=ISQ2;M[c*d+r]=val;M[r*d+c]=val;}}
static void mat_to_svec(int d,const double*M,double*v){int p=0;for(int c=0;c<d;c++)for(int r=0;r<=c;r++){
  double val=M[c*d+r];if(r!=c)val*=SQ2;v[p++]=val;}}
static double inf_norm(const double*v,int n){double m=0;for(int i=0;i<n;i++){double a=fabs(v[i]);if(a>m)m=a;}return m;}

/* ---------- blocks ---------- */
typedef struct{
  int type,dim,len,off,d,nu;
  int *ucols;
  double *lam_s,*lam_z;
  double *Ad,*Abd;
  double *wv,*wiv;
  double *H,*Hs,*Hsi,*evw,*V3,*Wm,*Winvm;
  double *Gm,*Ginv,*Sm,*evpsd,*evvec,*work,*Xm,*Ym,*Zm;
  int lwork,liwork;int *iwork;
  double mu;
}Blk;

typedef struct Ctx Ctx;
struct Ctx{
  int n,m,mE,mI,nblk,N,nu_total;
  Sp *P,*A,*Ai,*Ae;SpR AiR,AeR;
  double *q,*b,offset;
  Blk *blk;int *blkOf;
  SpR AbR;Sp *AbC;int *ab2csr,*abarSrc;
  Sp *K;int *pos;
  LDL ldl;int *perm;int orderdone;
  double rho,delta;
  double *x,*s,*z,*dx,*ds,*dz;
  double *sI,*zI;
  double *rd,*rpE,*rpI,*rc,*dshat,*dzhat,*w1,*w2,*w3,*Abdx,*winvrp,*rhs,*sol;
};

/* ---------- global matvecs ---------- */
static void Ab_mul(const Ctx*C,const double*x,double*out){
  for(int i=0;i<C->mI;i++){double a=0;for(int p=C->AbR.p[i];p<C->AbR.p[i+1];p++)a+=C->AbR.x[p]*x[C->AbR.j[p]];out[i]=a;}}
static void AbT_mul(const Ctx*C,const double*v,double*out){
  memset(out,0,C->n*sizeof(double));
  for(int i=0;i<C->mI;i++){double vi=v[i];if(vi==0.0)continue;for(int p=C->AbR.p[i];p<C->AbR.p[i+1];p++)out[C->AbR.j[p]]+=C->AbR.x[p]*vi;}}
static void Ae_mul(const Ctx*C,const double*x,double*out){
  for(int e=0;e<C->mE;e++){double a=0;for(int p=C->AeR.p[e];p<C->AeR.p[e+1];p++)a+=C->AeR.x[p]*x[C->AeR.j[p]];out[e]=a;}}
static void A_mul(const Ctx*C,const double*x,double*out){
  memset(out,0,C->m*sizeof(double));
  for(int j=0;j<C->n;j++){double xj=x[j];if(xj==0.0)continue;for(int p=C->A->p[j];p<C->A->p[j+1];p++)out[C->A->i[p]]+=C->A->x[p]*xj;}}
static void AT_mul(const Ctx*C,const double*v,double*out){
  memset(out,0,C->n*sizeof(double));
  for(int j=0;j<C->n;j++)for(int p=C->A->p[j];p<C->A->p[j+1];p++)out[j]+=C->A->x[p]*v[C->A->i[p]];}
static void P_mul(const Ctx*C,const double*x,double*out){
  memset(out,0,C->n*sizeof(double));
  for(int j=0;j<C->n;j++){double xj=x[j];if(xj==0.0)continue;for(int p=C->P->p[j];p<C->P->p[j+1];p++)out[C->P->i[p]]+=C->P->x[p]*xj;}}

/* ---------- per-block ops ---------- */
static int exp_grad_hess_local(const double*t,double*g,double*Hm);
static void blk_Winv_vec(Blk*B,const double*in,double*out){
  int len=B->len;
  if(B->type==CNONNEG){for(int i=0;i<len;i++)out[i]=in[i]*B->Winvm[i];}
  else if(B->type==CSOC){int d=B->dim;double a=B->wiv[0];const double*bv=B->wiv+1;
    double t0=a*in[0];for(int k=1;k<d;k++)t0+=bv[k-1]*in[k];
    out[0]=t0;for(int k=1;k<d;k++)out[k]=bv[k-1]*in[0]+a*in[k];}
  else if(B->type==CEXP){for(int i=0;i<3;i++)out[i]=B->Winvm[i*3+0]*in[0]+B->Winvm[i*3+1]*in[1]+B->Winvm[i*3+2]*in[2];}
  else{int d=B->d;svec_to_mat(d,in,B->Xm);dmm(d,B->Ginv,B->Xm,B->Ym);dmm(d,B->Ym,B->Ginv,B->Xm);mat_to_svec(d,B->Xm,out);}
}
static void blk_W_vec(Blk*B,const double*in,double*out){
  int len=B->len;
  if(B->type==CNONNEG){for(int i=0;i<len;i++)out[i]=in[i]*B->Wm[i];}
  else if(B->type==CSOC){int d=B->dim;double a=B->wv[0];const double*bv=B->wv+1;
    double t0=a*in[0];for(int k=1;k<d;k++)t0+=bv[k-1]*in[k];
    out[0]=t0;for(int k=1;k<d;k++)out[k]=bv[k-1]*in[0]+a*in[k];}
  else if(B->type==CEXP){for(int i=0;i<3;i++)out[i]=B->Wm[i*3+0]*in[0]+B->Wm[i*3+1]*in[1]+B->Wm[i*3+2]*in[2];}
  else{int d=B->d;svec_to_mat(d,in,B->Xm);dmm(d,B->Gm,B->Xm,B->Ym);dmm(d,B->Ym,B->Gm,B->Xm);mat_to_svec(d,B->Xm,out);}
}
/* r_c = lam_z - T (op) lam_s^{-1}, T = sigma*mu*e - dshat (op) dzhat ; affine: dshat=dzhat=NULL,sigma=0 */
static void blk_rc(Blk*B,double mu,double sigma,const double*dsh,const double*dzh,double*out){
  int len=B->len;
  if(B->type==CNONNEG||B->type==CEXP){
    for(int i=0;i<len;i++){
      double T=sigma*mu;
      if(dsh)T-=dsh[i]*dzh[i];
      double ls=B->lam_s[i];if(fabs(ls)<1e-300)ls=ls<0?-1e-300:1e-300;
      out[i]=B->lam_z[i]-T/ls;}
  }else if(B->type==CSOC){
    int d=B->dim;double *T=(double*)xmalloc(d*sizeof(double)),*Ti=(double*)xmalloc(d*sizeof(double));
    if(dsh){soc_jord(dsh,dzh,d,T);for(int i=0;i<d;i++)T[i]*=-1;}else memset(T,0,d*sizeof(double));
    T[0]+=sigma*mu;
    soc_inv(B->lam_s,d,Ti);
    soc_jord(Ti,T,d,out);
    for(int i=0;i<d;i++)out[i]=B->lam_z[i]-out[i];
    free(T);free(Ti);
  }else{
    int d=B->d;
    /* T = sigma*mu*I - (dS*dZ + dZ*dS)/2  in matrix space */
    if(dsh){
      svec_to_mat(d,dsh,B->Xm);svec_to_mat(d,dzh,B->Ym);
      dmm(d,B->Xm,B->Ym,B->Zm);
      dmm(d,B->Ym,B->Xm,B->Sm);
      for(int i=0;i<d*d;i++)B->Zm[i]=-0.5*(B->Zm[i]+B->Sm[i]);
    }else memset(B->Zm,0,(size_t)d*d*sizeof(double));
    for(int i=0;i<d;i++)B->Zm[i*d+i]+=sigma*mu;
    double ismu=1.0/sqrt(mu);
    for(int i=0;i<d*d;i++)B->Zm[i]*=ismu;
    /* out = lam_z - that */
    for(int i=0;i<len;i++)out[i]=B->lam_z[i];
    mat_to_svec(d,B->Zm,B->Xm);
    for(int i=0;i<len;i++)out[i]-=B->Xm[i];
    /* restore Sm (used as workspace elsewhere) - recompute lazily */
  }
}
static double blk_maxstep_s(Blk*B,const double*s,const double*ds){
  if(B->type==CNONNEG){double a=BIG;for(int i=0;i<B->len;i++)if(ds[i]<0){double t=-s[i]/ds[i];if(t<a)a=t;}return a;}
  if(B->type==CSOC)return soc_maxstep(s,ds,B->dim);
  if(B->type==CEXP)return exp_maxstep(s,ds);
  {int d=B->d;svec_to_mat(d,ds,B->Xm);dmm(d,B->Ginv,B->Xm,B->Ym);dmm(d,B->Ym,B->Ginv,B->Xm);
   /* eigen of symmetric Xm */
   memcpy(B->Zm,B->Xm,(size_t)d*d*sizeof(double));
   eig_sym(d,B->Zm,B->evpsd,B->work,B->lwork,B->iwork,B->liwork);
   double lmin=B->evpsd[0];return lmin<0?-sqrt(B->mu)/lmin:BIG;}
}
static double blk_maxstep_z(Blk*B,const double*z,const double*dz){
  if(B->type==CEXP){
    /* map to primal cone: (u,v,w)->(-v,-u,e*w) */
    double s[3]={-z[1],-z[0],2.718281828459045*z[2]};
    double d[3]={-dz[1],-dz[0],2.718281828459045*dz[2]};
    return exp_maxstep(s,d);}
  return blk_maxstep_s(B,z,dz);
}
/* cone violation of a block vector (primal or dual) with normalization */
static double blk_viol(Blk*B,const double*v,int dual){
  int len=B->len;double mx=inf_norm(v,len);if(mx<1)mx=1;
  double *u=(double*)xmalloc(len*sizeof(double));
  for(int i=0;i<len;i++)u[i]=v[i]/mx;
  double r=0;
  if(B->type==CNONNEG){double mn=0;for(int i=0;i<len;i++)if(u[i]<mn)mn=u[i];r=mn<0?-mn:0;}
  else if(B->type==CSOC){int d=B->dim;double nb=0;for(int i=1;i<d;i++)nb+=u[i]*u[i];nb=sqrt(nb);r=nb-u[0];if(r<0)r=0;}
  else if(B->type==CEXP){if(dual)r=exp_viol(-u[1],-u[0],2.718281828459045*u[2]);else r=exp_viol(u[0],u[1],u[2]);}
  else{int d=B->d;svec_to_mat(d,u,B->Xm);memcpy(B->Zm,B->Xm,(size_t)d*d*sizeof(double));
    eig_sym(d,B->Zm,B->evpsd,B->work,B->lwork,B->iwork,B->liwork);r=-B->evpsd[0];if(r<0)r=0;}
  free(u);return r;}

/* ---------- scaling computation per block ---------- */
static void blk_scaling(Ctx*C,Blk*B,double mu){
  const double*s=C->sI+B->off,*z=C->zI+B->off;
  int len=B->len;B->mu=mu;
  if(B->type==CNONNEG){
    for(int i=0;i<len;i++){
      double sv=s[i]>1e-300?s[i]:1e-300,zv=z[i]>1e-300?z[i]:1e-300;
      double w=sqrt(sv/zv);B->Wm[i]=w;B->Winvm[i]=1.0/w;B->lam_s[i]=sqrt(sv*zv);B->lam_z[i]=B->lam_s[i];}
  }else if(B->type==CSOC){
    int d=B->dim;double *tmpa=(double*)xmalloc(2*d*sizeof(double)),*tmpb=tmpa+d;
    soc_inv(z,d,tmpa);soc_jord(s,tmpa,d,tmpb);
    if(soc_sqrt(tmpb,d,B->wv)!=0){B->wv[0]=1;for(int i=1;i<d;i++)B->wv[i]=0;}
    soc_inv(B->wv,d,B->wiv);
    soc_jord(B->wv,z,d,B->lam_z);
    soc_jord(B->wiv,s,d,B->lam_s);
    free(tmpa);
  }else if(B->type==CEXP){
    double g[3];
    if(exp_grad_hess_local(s,g,B->H)!=0){memset(B->H,0,9*sizeof(double));B->H[0]=B->H[4]=B->H[8]=1.0;}
    memcpy(B->V3,B->H,9*sizeof(double));
    eig_sym(3,B->V3,B->evw,B->work,B->lwork,B->iwork,B->liwork);
    for(int i=0;i<3;i++)if(B->evw[i]<1e-14)B->evw[i]=1e-14;
    for(int c=0;c<3;c++)for(int r=0;r<3;r++)B->Xm[c*3+r]=B->V3[c*3+r]*sqrt(B->evw[c]);
    dmmT(3,B->Xm,B->V3,B->Hs);
    for(int c=0;c<3;c++)for(int r=0;r<3;r++)B->Xm[c*3+r]=B->V3[c*3+r]/sqrt(B->evw[c]);
    dmmT(3,B->Xm,B->V3,B->Hsi);
    double sm=sqrt(mu),ism=1.0/sm;
    for(int i=0;i<9;i++){B->Wm[i]=ism*B->Hsi[i];B->Winvm[i]=sm*B->Hs[i];}
    for(int i=0;i<3;i++){double a=0,b=0;
      for(int k=0;k<3;k++){a+=B->Winvm[i*3+k]*s[k];b+=B->Wm[i*3+k]*z[k];}
      B->lam_s[i]=a;B->lam_z[i]=b;}
  }else{
    int d=B->d;
    svec_to_mat(d,s,B->Sm);
    memcpy(B->evvec,B->Sm,(size_t)d*d*sizeof(double));
    eig_sym(d,B->evvec,B->evpsd,B->work,B->lwork,B->iwork,B->liwork);
    for(int i=0;i<d;i++)if(B->evpsd[i]<1e-14)B->evpsd[i]=1e-14;
    double m14=pow(mu,-0.25),m4=pow(mu,0.25);
    for(int c=0;c<d;c++)for(int r=0;r<d;r++)B->Xm[c*d+r]=B->evvec[c*d+r]*sqrt(B->evpsd[c]);
    dmmT(d,B->Xm,B->evvec,B->Gm);
    for(int i=0;i<d*d;i++)B->Gm[i]*=m14;
    for(int c=0;c<d;c++)for(int r=0;r<d;r++)B->Xm[c*d+r]=B->evvec[c*d+r]/sqrt(B->evpsd[c]);
    dmmT(d,B->Xm,B->evvec,B->Ginv);
    for(int i=0;i<d*d;i++)B->Ginv[i]*=m4;
    int p=0;double sm=sqrt(mu);
    for(int c=0;c<d;c++)for(int r=0;r<=c;r++)B->lam_s[p++]=(r==c)?sm:0.0;
    svec_to_mat(d,z,B->Zm);
    dmm(d,B->Gm,B->Zm,B->Xm);dmm(d,B->Xm,B->Gm,B->Ym);
    mat_to_svec(d,B->Ym,B->lam_z);
  }
}
static int exp_grad_hess_local(const double*t,double*g,double*Hm){
  double u=t[0],v=t[1],w=t[2];
  if(v<=0)return -1;double r=u/v;if(r>700)return -1;double E=exp(r);double gg=w-v*E;if(gg<=0)return -1;
  double ig=1.0/gg,iv=1.0/v,iw=1.0/w;
  double gu=-E,gv=E*(r-1.0),gw=1.0;
  if(g){g[0]=-gu*ig;g[1]=-iv-gv*ig;g[2]=-iw-gw*ig;}
  if(Hm){double ig2=ig*ig;
    double hguu=-E*iv,hguv=E*r*iv,hgvv=-E*r*r*iv;
    Hm[0]=gu*gu*ig2-hguu*ig;Hm[1]=gu*gv*ig2-hguv*ig;Hm[2]=gu*gw*ig2;
    Hm[4]=gv*gv*ig2-hgvv*ig+iv*iv;Hm[5]=gv*gw*ig2;Hm[8]=gw*gw*ig2+iw*iw;
    Hm[3]=Hm[1];Hm[6]=Hm[2];Hm[7]=Hm[5];}
  return 0;}

/* ---------- Abar fill ---------- */
static void abar_fill(Ctx*C){
  for(int bi=0;bi<C->nblk;bi++){
    Blk*B=&C->blk[bi];
    if(B->type==CNONNEG){
      for(int r=0;r<B->len;r++){int rr=B->off+r;
        double wi=B->Winvm[rr-B->off];
        for(int p=C->AbR.p[rr];p<C->AbR.p[rr+1];p++){int src=C->abarSrc[p];
          C->AbR.x[p]=src>=0?C->AiR.x[src]*wi:0.0;}}
    }else{
      int nu=B->nu,len=B->len;
      memset(B->Ad,0,(size_t)len*nu*sizeof(double));
      int *up=(int*)xmalloc(C->n*sizeof(int));for(int i=0;i<C->n;i++)up[i]=-1;
      for(int c=0;c<nu;c++)up[B->ucols[c]]=c;
      for(int r=0;r<len;r++){int rr=B->off+r;
        for(int p=C->AiR.p[rr];p<C->AiR.p[rr+1];p++){int c=up[C->AiR.j[p]];if(c>=0)B->Ad[(size_t)r*nu+c]=C->AiR.x[p];}}
      free(up);
      if(B->type==CSOC){
        int d=B->dim;double a=B->wiv[0];const double*bv=B->wiv+1;
        for(int c=0;c<nu;c++){
          double t0=a*B->Ad[c];for(int k=1;k<d;k++)t0+=bv[k-1]*B->Ad[(size_t)k*nu+c];
          B->Abd[c]=t0;
          for(int k=1;k<d;k++)B->Abd[(size_t)k*nu+c]=bv[k-1]*B->Ad[c]+a*B->Ad[(size_t)k*nu+c];}
      }else if(B->type==CEXP){
        for(int c=0;c<nu;c++){
          double c0=B->Ad[c],c1=B->Ad[nu+c],c2=B->Ad[2*nu+c];
          B->Abd[c]=B->Winvm[0]*c0+B->Winvm[1]*c1+B->Winvm[2]*c2;
          B->Abd[nu+c]=B->Winvm[3]*c0+B->Winvm[4]*c1+B->Winvm[5]*c2;
          B->Abd[2*nu+c]=B->Winvm[6]*c0+B->Winvm[7]*c1+B->Winvm[8]*c2;}
      }else{
        int d=B->d;
        for(int c=0;c<nu;c++){
          svec_to_mat(d,B->Ad+(size_t)c*len,B->Xm);
          dmm(d,B->Ginv,B->Xm,B->Ym);dmm(d,B->Ym,B->Ginv,B->Xm);
          mat_to_svec(d,B->Xm,B->Abd+(size_t)c*len);}
      }
      for(int r=0;r<len;r++){int rr=B->off+r;int q0=C->AbR.p[rr];
        for(int c=0;c<nu;c++)C->AbR.x[q0+c]=B->Abd[(size_t)r*nu+c];}
    }
  }
  for(int q=0;q<C->AbC->nnz;q++)C->AbC->x[q]=C->AbR.x[C->ab2csr[q]];
}

/* ---------- KKT assembly ---------- */
static void kkt_assemble(Ctx*C){
  Sp*K=C->K;int n=C->n,mE=C->mE,N=C->N;int *pos=C->pos;
  for(int j=0;j<N;j++){
    int a=K->p[j],b=K->p[j+1];
    for(int q=a;q<b;q++){pos[K->i[q]]=q;K->x[q]=0.0;}
    if(j<n){
      for(int p=C->P->p[j];p<C->P->p[j+1];p++)K->x[pos[C->P->i[p]]]+=C->P->x[p];
      for(int p=C->AbC->p[j];p<C->AbC->p[j+1];p++){
        int i=C->AbC->i[p];double v=C->AbC->x[p];
        for(int q=C->AbR.p[i];q<C->AbR.p[i+1];q++)K->x[pos[C->AbR.j[q]]]+=v*C->AbR.x[q];}
      for(int p=C->Ae->p[j];p<C->Ae->p[j+1];p++)K->x[pos[n+C->Ae->i[p]]]+=C->Ae->x[p];
      K->x[pos[j]]+=C->rho;
    }else{
      int e=j-n;
      for(int p=C->AeR.p[e];p<C->AeR.p[e+1];p++)K->x[pos[C->AeR.j[p]]]+=C->AeR.x[p];
      K->x[pos[j]]-=C->delta;
    }
    for(int q=a;q<b;q++)pos[K->i[q]]=-1;
  }
}

/* ---------- vector-level block ops ---------- */
static void apply_Winv_all(Ctx*C,const double*in,double*out){
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];blk_Winv_vec(B,in+B->off,out+B->off);}}
static void apply_W_all(Ctx*C,const double*in,double*out){
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];blk_W_vec(B,in+B->off,out+B->off);}}
static void rc_all(Ctx*C,double mu,double sigma,const double*dsh,const double*dzh,double*out){
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];blk_rc(B,mu,sigma,dsh?dsh+B->off:NULL,dzh?dzh+B->off:NULL,out+B->off);}}

static void solve_dir(Ctx*C,const double*rc,double*Abdx,double*winvrp){
  /* rhs1 = -rd + Ab'(rc - W^-1 rpI) ; rhs2 = -rpE */
  for(int i=0;i<C->mI;i++)C->w1[i]=rc[i]-winvrp[i];
  AbT_mul(C,C->w1,C->rhs);
  for(int j=0;j<C->n;j++)C->rhs[j]+= -C->rd[j];
  for(int e=0;e<C->mE;e++)C->rhs[C->n+e]=-C->rpE[e];
  ldl_solve(&C->ldl,C->rhs,C->sol);
  memcpy(C->dx,C->sol,C->n*sizeof(double));
  Ab_mul(C,C->dx,Abdx);
  for(int i=0;i<C->mI;i++){
    C->dzhat[i]=Abdx[i]-rc[i]+winvrp[i];
    C->dshat[i]=-Abdx[i]-winvrp[i];}
  apply_W_all(C,C->dshat,C->ds+C->mE);
  apply_Winv_all(C,C->dzhat,C->dz+C->mE);
  for(int e=0;e<C->mE;e++){C->ds[e]=0.0;C->dz[e]=C->sol[C->n+e];}
}

static double step_len(Ctx*C,const double*ds,const double*dz){
  double ap=BIG,ad=BIG;
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];int o=C->mE+B->off;
    double t=blk_maxstep_s(B,C->sI+B->off,ds+o);if(t<ap)ap=t;
    t=blk_maxstep_z(B,C->zI+B->off,dz+o);if(t<ad)ad=t;}
  double a=ap<ad?ap:ad;
  if(!(a>0))a=0;
  if(a>1.0)a=1.0;
  return a;
}

/* ---------- metrics ---------- */
static void metrics(Ctx*C,double*rp,double*rd,double*pk,double*dk,double*gapabs,double*gaprel){
  double *Ax=C->w1,*Px=C->w2,*ATz=C->w3;
  A_mul(C,C->x,Ax);
  double na=inf_norm(Ax,C->m),ns=inf_norm(C->s,C->m),nb=inf_norm(C->b,C->m);
  double den=1.0+(na>ns?(na>nb?na:nb):(ns>nb?ns:nb));
  double mx=0;
  for(int i=0;i<C->m;i++){double v=Ax[i]+C->s[i]-C->b[i];if(fabs(v)>mx)mx=fabs(v);}
  *rp=mx/den;
  P_mul(C,C->x,Px);AT_mul(C,C->z,ATz);
  double n1=inf_norm(Px,C->n),n2=inf_norm(C->q,C->n),n3=inf_norm(ATz,C->n);
  double den2=1.0+(n1>n2?(n1>n3?n1:n3):(n2>n3?n2:n3));
  mx=0;for(int j=0;j<C->n;j++){double v=Px[j]+C->q[j]+ATz[j];if(fabs(v)>mx)mx=fabs(v);}
  *rd=mx/den2;
  double xPx=0;for(int j=0;j<C->n;j++)xPx+=C->x[j]*Px[j];
  double qx=0;for(int j=0;j<C->n;j++)qx+=C->q[j]*C->x[j];
  double bz=0;for(int i=0;i<C->m;i++)bz+=C->b[i]*C->z[i];
  double p0=0.5*xPx+qx,d0=-0.5*xPx-bz;
  *gapabs=fabs(p0-d0);
  double mxx=fabs(p0)>fabs(d0)?fabs(p0):fabs(d0);
  *gaprel=mxx>0?*gapabs/mxx:0.0;
  C->sol[0]=p0;C->sol[1]=d0;
  *pk=0;*dk=0;
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];
    double v=blk_viol(B,C->sI+B->off,0);if(v>*pk)*pk=v;
    v=blk_viol(B,C->zI+B->off,1);if(v>*dk)*dk=v;}
  if(C->mE>0){double v=inf_norm(C->s,C->mE);if(v>*pk)*pk=v;}
}

static int point_inside(Ctx*C,const double*s,const double*z){
  for(int i=0;i<C->m;i++){if(!isfinite(s[i])||!isfinite(z[i]))return 0;}
  for(int j=0;j<C->n;j++)if(!isfinite(C->x[j]))return 0;
  if(C->mE>0){double v=inf_norm(s,C->mE);if(!(v<=1e-12))return 0;}
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];
    double*a=(double*)xmalloc(B->len*sizeof(double));double*c=(double*)xmalloc(B->len*sizeof(double));
    memcpy(a,s+C->mE+B->off,B->len*sizeof(double));
    memcpy(c,z+C->mE+B->off,B->len*sizeof(double));
    double v1=blk_viol(B,a,0),v2=blk_viol(B,c,1);
    free(a);free(c);
    if(!(v1<=0.0)||!(v2<=0.0))return 0;}
  (void)0;
  return 1;
}

/* ---------- infeasibility certificates ---------- */
static int check_pinf(Ctx*C,double*zout){
  double nb=inf_norm(C->b,C->m);if(nb<=0)return 0;
  double nz=inf_norm(C->z,C->m);if(nz<=0)return 0;
  double *zh=C->w1;for(int i=0;i<C->m;i++)zh[i]=C->z[i]/nz;
  double bz=0;for(int i=0;i<C->m;i++)bz+=(C->b[i]/nb)*zh[i];
  double margin=-bz;if(margin<=1e-12)return 0;
  double *t=C->w2;AT_mul(C,zh,t);
  double a=C->rho>0?0:0; (void)a;
  double scA=0;for(int p=0;p<C->A->nnz;p++){double v=fabs(C->A->x[p]);if(v>scA)scA=v;}if(scA==0)scA=1;
  double lin=inf_norm(t,C->n)/scA/margin;
  double cv=0;
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];double v=blk_viol(B,zh+C->blk[bi].off+C->mE,1);if(v>cv)cv=v;}
  {double v=0;(void)v;}
  if(lin<=1e-9&&cv/margin<=1e-9){memcpy(zout,zh,C->m*sizeof(double));return 1;}
  return 0;
}
static int check_dinf(Ctx*C,double*xout,double*sout){
  double nq=inf_norm(C->q,C->n);if(nq<=0)return 0;
  double nx=inf_norm(C->x,C->n);if(nx<=0)return 0;
  double *xh=C->w1;for(int j=0;j<C->n;j++)xh[j]=C->x[j]/nx;
  double *sh=C->w2;for(int i=0;i<C->m;i++)sh[i]=C->s[i]/nx;
  double qx=0;for(int j=0;j<C->n;j++)qx+=(C->q[j]/nq)*xh[j];
  double margin=-qx;if(margin<=1e-12)return 0;
  double scA=0;for(int p=0;p<C->A->nnz;p++){double v=fabs(C->A->x[p]);if(v>scA)scA=v;}if(scA==0)scA=1;
  double scP=0;for(int p=0;p<C->P->nnz;p++){double v=fabs(C->P->x[p]);if(v>scP)scP=v;}if(scP==0)scP=1;
  double *t=C->w3;P_mul(C,xh,t);
  double pq=inf_norm(t,C->n)/scP/margin;
  /* (A xh + sh)/scA */
  A_mul(C,xh,t);
  double mx=0;for(int i=0;i<C->m;i++){double v=(t[i]+sh[i])/scA;if(fabs(v)>mx)mx=fabs(v);}
  double pl=mx/margin;
  double cv=0;
  for(int bi=0;bi<C->nblk;bi++){Blk*B=&C->blk[bi];
    double *tmp=(double*)xmalloc(B->len*sizeof(double));
    for(int i=0;i<B->len;i++)tmp[i]=sh[B->off+i+C->mE]/scA;
    double v=blk_viol(B,tmp,0);free(tmp);if(v>cv)cv=v;}
  cv/=margin;
  double ez=inf_norm(sh,C->mE)/scA;
  if(pq<=1e-9&&pl<=1e-9&&cv<=1e-9&&ez<=1e-9*margin){memcpy(xout,xh,C->n*sizeof(double));memcpy(sout,sh,C->m*sizeof(double));return 1;}
  return 0;
}

/* ---------- JSON cones ---------- */
static int parse_cones(const char*js,int**types,int**dims,int*nc){
  const char*p=strstr(js,"\"cones\"");if(!p)return -1;p=strchr(p,'[');if(!p)return -1;p++;
  int cap=16,n=0;int*tt=(int*)xmalloc(cap*sizeof(int)),*dd=(int*)xmalloc(cap*sizeof(int));
  while(*p&&*p!=']'){
    if(*p=='{'){const char*e=strchr(p,'}');if(!e)break;
      char buf[256];int L=(int)(e-p);if(L>250)L=250;memcpy(buf,p,L);buf[L]=0;
      char*tp=strstr(buf,"\"type\""),*dp=strstr(buf,"\"dim\"");
      int type=-1,dim=0;
      if(tp){char*q=strchr(tp,':');q=q?strchr(q,'"'):NULL;if(q){q++;char*q2=strchr(q,'"');
        if(q2){int l=(int)(q2-q);char nm[64];if(l>60)l=60;memcpy(nm,q,l);nm[l]=0;
          if(!strcmp(nm,"zero"))type=CZ;else if(!strcmp(nm,"nonnegative"))type=CNONNEG;
          else if(!strcmp(nm,"soc"))type=CSOC;else if(!strcmp(nm,"exponential"))type=CEXP;
          else if(!strcmp(nm,"psd"))type=CPSD;}}}
      if(dp){char*q=strchr(dp,':');if(q)dim=(int)strtol(q+1,NULL,10);}
      if(type<0||dim<=0){free(tt);free(dd);return -2;}
      if(n==cap){cap*=2;tt=realloc(tt,cap*sizeof(int));dd=realloc(dd,cap*sizeof(int));}
      tt[n]=type;dd[n]=dim;n++;p=e+1;}
    else p++;}
  *types=tt;*dims=dd;*nc=n;return 0;}

static int *arr_to_int(NpyArr*a){
  if(a->itemsize==8){int64_t*s=(int64_t*)a->data;long c=a->count;int*o=(int*)xmalloc((c?c:1)*sizeof(int));
    for(long i=0;i<c;i++)o[i]=(int)s[i];return o;}
  if(a->itemsize==4){int*s=(int*)a->data;long c=a->count;int*o=(int*)xmalloc((c?c:1)*sizeof(int));
    for(long i=0;i<c;i++)o[i]=s[i];return o;}
  return NULL;
}
static double *arr_to_dbl(NpyArr*a){
  long c=a->count;double*o=(double*)xmalloc((c?c:1)*sizeof(double));
  if(a->itemsize==8)memcpy(o,a->data,c*sizeof(double));
  else if(a->itemsize==4){float*f=(float*)a->data;for(long i=0;i<c;i++)o[i]=f[i];}
  return o;
}

int main(int argc,char**argv){
  t_start=now_sec();
  const char*inpath=NULL,*outpath=NULL;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--input")&&i+1<argc)inpath=argv[++i];
    else if(!strcmp(argv[i],"--output")&&i+1<argc)outpath=argv[++i];
    else if(!strncmp(argv[i],"--input=",8))inpath=argv[i]+8;
    else if(!strncmp(argv[i],"--output=",9))outpath=argv[i]+9;
    else if(!strcmp(argv[i],"--limit")&&i+1<argc)time_limit=atof(argv[++i]);}
  if(getenv("SOLVER_DEBUG"))dbg=1;
  {const char*l=getenv("SOLVER_LIMIT");if(l)time_limit=atof(l);}
  if(!inpath||!outpath){fprintf(stderr,"usage: solver --input F --output F\n");return 2;}

  NpzW W;npzw_init(&W);
  const char*STATUS="unresolved";
  double *outx=NULL,*outs=NULL,*outz=NULL;int oxn=0,osn=0,ozn=0;

  Npz*nz=npz_read(inpath);
  if(!nz){fprintf(stderr,"read fail\n");goto done;}
  {
  NpyArr*aPd=npz_get(nz,"P_data"),*aPi=npz_get(nz,"P_indices"),*aPp=npz_get(nz,"P_indptr"),*aPs=npz_get(nz,"P_shape");
  NpyArr*aAd=npz_get(nz,"A_data"),*aAi=npz_get(nz,"A_indices"),*aAp=npz_get(nz,"A_indptr"),*aAs=npz_get(nz,"A_shape");
  NpyArr*aQ=npz_get(nz,"q"),*aB=npz_get(nz,"b"),*aM=npz_get(nz,"metadata_json");
  if(!aPd||!aPi||!aPp||!aPs||!aAd||!aAi||!aAp||!aAs||!aQ||!aB||!aM){fprintf(stderr,"missing\n");goto done;}
  int *Pi=arr_to_int(aPi),*Pp=arr_to_int(aPp),*Ai_=arr_to_int(aAi),*Ap_=arr_to_int(aAp);
  int n,m;
  {if(aPs->itemsize==8){int64_t*s=(int64_t*)aPs->data;n=(int)s[1];}else{int*s=(int*)aPs->data;n=s[1];}}
  {if(aAs->itemsize==8){int64_t*s=(int64_t*)aAs->data;m=(int)s[0];}else{int*s=(int*)aAs->data;m=s[0];}}
  double *Pd=arr_to_dbl(aPd),*Ad=arr_to_dbl(aAd),*Qd=arr_to_dbl(aQ),*Bd=arr_to_dbl(aB);
  long nP=aPd->count,nA=aAd->count;

  Sp*P=sp_alloc(n,n,(int)nP);memcpy(P->p,Pp,(n+1)*sizeof(int));memcpy(P->i,Pi,nP*sizeof(int));memcpy(P->x,Pd,nP*sizeof(double));P->nnz=(int)nP;
  Sp*A0=sp_alloc(m,n,(int)nA);memcpy(A0->p,Ap_,(n+1)*sizeof(int));memcpy(A0->i,Ai_,nA*sizeof(int));memcpy(A0->x,Ad,nA*sizeof(double));A0->nnz=(int)nA;

  /* metadata -> utf8 */
  char*meta;
  {unsigned char*d=(unsigned char*)aM->data;long len=(aM->itemsize/4)*(aM->count>0?aM->count:1);if(len<1)len=1;meta=(char*)xmalloc(4*len+1);long o=0;
   for(long i=0;i<len;i++){uint32_t cp=d[4*i]|(d[4*i+1]<<8)|(d[4*i+2]<<16)|((uint32_t)d[4*i+3]<<24);
     if(cp==0)break;
     if(cp<0x80)meta[o++]=(char)cp;
     else if(cp<0x800){meta[o++]=0xC0|(cp>>6);meta[o++]=0x80|(cp&63);}
     else if(cp<0x10000){meta[o++]=0xE0|(cp>>12);meta[o++]=0x80|((cp>>6)&63);meta[o++]=0x80|(cp&63);}
     else{meta[o++]=0xF0|(cp>>18);meta[o++]=0x80|((cp>>12)&63);meta[o++]=0x80|((cp>>6)&63);meta[o++]=0x80|(cp&63);}}
   meta[o]=0;}
  double objoff=0;{char*p=strstr(meta,"\"objective_offset\"");if(p){p=strchr(p,':');if(p)objoff=strtod(p+1,NULL);}}
  int *ct,*cd,nc;
  if(dbg){fprintf(stderr,"META len=%ld isz=%d cnt=%ld descr=%s\n",(long)strlen(meta),aM->itemsize,aM->count,aM->descr);
    unsigned char*dd=(unsigned char*)aM->data;for(int i=0;i<24;i++)fprintf(stderr,"%02x ",dd[i]);fprintf(stderr,"\n");
    fprintf(stderr,"META=[%s]\n",meta);}
  int pcr=parse_cones(meta,&ct,&cd,&nc);
  if(dbg)fprintf(stderr,"parse_cones=%d nc=%d\n",pcr,nc);
  if(pcr!=0){fprintf(stderr,"cones\n");goto done;}

  /* ---- reorder blocks: zero cones first ---- */
  int*border=(int*)xmalloc((nc?nc:1)*sizeof(int));int nb=0;
  for(int i=0;i<nc;i++)if(ct[i]==CZ)border[nb++]=i;
  for(int i=0;i<nc;i++)if(ct[i]!=CZ)border[nb++]=i;
  int*width=(int*)xmalloc((nc?nc:1)*sizeof(int));
  for(int i=0;i<nc;i++)width[i]=ct[i]==CPSD?cd[i]*(cd[i]+1)/2:cd[i];
  int*ost=(int*)xmalloc((nc+1)*sizeof(int));ost[0]=0;for(int i=0;i<nc;i++)ost[i+1]=ost[i]+width[i];
  int mE=0,mI=0;for(int i=0;i<nc;i++){if(ct[i]==CZ)mE+=width[i];else mI+=width[i];}
  int*rowperm=(int*)xmalloc((m?m:1)*sizeof(int));
  int*nbs=(int*)xmalloc((nc+1)*sizeof(int));
  {int r=0;for(int k=0;k<nc;k++){int bi=border[k];nbs[k]=r;for(int t=0;t<width[bi];t++)rowperm[r++]=ost[bi]+t;}nbs[nc]=r;}

  Ctx C;memset(&C,0,sizeof C);
  C.n=n;C.m=m;C.mE=mE;C.mI=mI;C.offset=objoff;C.N=n+mE;
  C.P=P;C.q=Qd;
  /* permute rows of A and b */
  {int*inv=(int*)xmalloc((m?m:1)*sizeof(int));for(int i=0;i<m;i++)inv[rowperm[i]]=i;
   Sp*Ap=sp_alloc(m,n,A0->nnz);
   for(int j=0;j<n;j++)Ap->p[j+1]=Ap->p[j]+(A0->p[j+1]-A0->p[j]);
   for(int j=0;j<n;j++){int q=Ap->p[j];int cnt=0;
     for(int p=A0->p[j];p<A0->p[j+1];p++){Ap->i[q+cnt]=inv[A0->i[p]];Ap->x[q+cnt]=A0->x[p];cnt++;}
     for(int u=0;u<cnt;u++)for(int v=u+1;v<cnt;v++)if(Ap->i[q+v]<Ap->i[q+u]){int ti=Ap->i[q+u];Ap->i[q+u]=Ap->i[q+v];Ap->i[q+v]=ti;double tx=Ap->x[q+u];Ap->x[q+u]=Ap->x[q+v];Ap->x[q+v]=tx;}}
   C.A=Ap;sp_free(A0);}
  C.b=(double*)xmalloc((m?m:1)*sizeof(double));
  for(int i=0;i<m;i++)C.b[i]=Bd[rowperm[i]];

  C.blk=(Blk*)xcalloc(nc?nc:1,sizeof(Blk));
  C.blkOf=(int*)xmalloc((mI?mI:1)*sizeof(int));
  {int ioff=0;C.nblk=0;
   for(int k=0;k<nc;k++){int bi=border[k];if(ct[bi]==CZ)continue;
     Blk*B=&C.blk[C.nblk];memset(B,0,sizeof*B);
     B->type=ct[bi];B->dim=cd[bi];B->len=width[bi];B->off=ioff;B->d=(B->type==CPSD)?B->dim:0;
     for(int r=0;r<B->len;r++)C.blkOf[ioff+r]=C.nblk;
     ioff+=B->len;C.nblk++;}}
  C.Ae=mE>0?sp_rows(C.A,0,mE):sp_alloc(0,n,0);
  C.Ai=mI>0?sp_rows(C.A,mE,m):sp_alloc(0,n,0);
  spr_from_sp(&C.AiR,C.Ai);spr_from_sp(&C.AeR,C.Ae);
  C.x=(double*)xcalloc(n?n:1,sizeof(double));
  C.s=(double*)xcalloc(m?m:1,sizeof(double));
  C.z=(double*)xcalloc(m?m:1,sizeof(double));
  C.dx=(double*)xcalloc(n?n:1,sizeof(double));C.ds=(double*)xcalloc(m?m:1,sizeof(double));C.dz=(double*)xcalloc(m?m:1,sizeof(double));
  C.sI=C.s+mE;C.zI=C.z+mE;
  #define ALLOCV(f,sz) C.f=(double*)xcalloc((sz)?(sz):1,sizeof(double))
  ALLOCV(rd,n);ALLOCV(rpE,mE);ALLOCV(rpI,mI);ALLOCV(rc,mI);ALLOCV(dshat,mI);ALLOCV(dzhat,mI);
  ALLOCV(w1,mI>n?mI:n);ALLOCV(w2,mI>n?mI:n);ALLOCV(w3,mI>n?mI:n);ALLOCV(Abdx,mI);ALLOCV(winvrp,mI);
  ALLOCV(rhs,C.N);ALLOCV(sol,C.N);
  #undef ALLOCV
  C.w1=(double*)realloc(C.w1,(size_t)(m>n?m:n)*sizeof(double)+8);
  C.w2=(double*)realloc(C.w2,(size_t)(m>n?m:n)*sizeof(double)+8);
  C.w3=(double*)realloc(C.w3,(size_t)(m>n?m:n)*sizeof(double)+sizeof(double)*8);
  C.pos=(int*)xmalloc((C.N?C.N:1)*sizeof(int));
  for(int i=0;i<C.N;i++)C.pos[i]=-1;
  C.perm=(int*)xmalloc((C.N?C.N:1)*sizeof(int));

  /* nu_total */
  {long nu=0;for(int bi=0;bi<C.nblk;bi++){Blk*B=&C.blk[bi];
     nu+= B->type==CNONNEG?B->len : B->type==CSOC?2 : B->type==CEXP?3 : B->d;}
   C.nu_total=(int)(nu>0?nu:1);}

  /* block workspaces */
  for(int bi=0;bi<C.nblk;bi++){
    Blk*B=&C.blk[bi];
    B->lam_s=(double*)xcalloc(B->len,sizeof(double));
    B->lam_z=(double*)xcalloc(B->len,sizeof(double));
    if(B->type==CNONNEG){B->Wm=(double*)xcalloc(B->len,sizeof(double));B->Winvm=(double*)xcalloc(B->len,sizeof(double));}
    else if(B->type==CSOC){B->wv=(double*)xcalloc(B->dim,sizeof(double));B->wiv=(double*)xcalloc(B->dim,sizeof(double));
      B->Xm=(double*)xcalloc(4,sizeof(double));}
    else if(B->type==CEXP){
      B->H=(double*)xcalloc(9,sizeof(double));B->Hs=(double*)xcalloc(9,sizeof(double));B->Hsi=(double*)xcalloc(9,sizeof(double));
      B->Wm=(double*)xcalloc(9,sizeof(double));B->Winvm=(double*)xcalloc(9,sizeof(double));
      B->evw=(double*)xcalloc(3,sizeof(double));B->V3=(double*)xcalloc(9,sizeof(double));B->Xm=(double*)xcalloc(9,sizeof(double));
      eig_wsq(3,&B->lwork,&B->liwork);
      B->work=(double*)xcalloc(B->lwork>0?B->lwork:1,sizeof(double));B->iwork=(int*)xcalloc(B->liwork>0?B->liwork:1,sizeof(int));}
    else{int d=B->d;
      B->Gm=(double*)xcalloc((size_t)d*d,sizeof(double));B->Ginv=(double*)xcalloc((size_t)d*d,sizeof(double));
      B->Sm=(double*)xcalloc((size_t)d*d,sizeof(double));B->Xm=(double*)xcalloc((size_t)d*d,sizeof(double));
      B->Ym=(double*)xcalloc((size_t)d*d,sizeof(double));B->Zm=(double*)xcalloc((size_t)d*d,sizeof(double));
      B->evvec=(double*)xcalloc((size_t)d*d,sizeof(double));B->evpsd=(double*)xcalloc(d,sizeof(double));
      eig_wsq(d,&B->lwork,&B->liwork);
      B->work=(double*)xcalloc(B->lwork>0?B->lwork:1,sizeof(double));B->iwork=(int*)xcalloc(B->liwork>0?B->liwork:1,sizeof(int));}
  }

  /* ---- Abar CSR pattern ---- */
  {int tot=0;int*rowp=(int*)xmalloc((mI+1)*sizeof(int));int*cnts=(int*)xmalloc((mI?mI:1)*sizeof(int));
   for(int bi=0;bi<C.nblk;bi++){Blk*B=&C.blk[bi];
     if(B->type==CNONNEG){for(int r=0;r<B->len;r++)cnts[B->off+r]=C.AiR.p[B->off+r+1]-C.AiR.p[B->off+r];}
     else{int*mark=(int*)xcalloc(n?n:1,sizeof(int));int cnt=0;int*list=(int*)xmalloc((n+1)*sizeof(int));
       for(int r=0;r<B->len;r++){int rr=B->off+r;
         for(int p=C.AiR.p[rr];p<C.AiR.p[rr+1];p++){int j=C.AiR.j[p];if(!mark[j]){mark[j]=1;list[cnt++]=j;}}}
       for(int i=0;i<cnt;i++)for(int k=i+1;k<cnt;k++)if(list[k]<list[i]){int t=list[i];list[i]=list[k];list[k]=t;}
       B->nu=cnt;B->ucols=(int*)xmalloc((cnt?cnt:1)*sizeof(int));memcpy(B->ucols,list,cnt*sizeof(int));
       B->Ad=(double*)xcalloc((size_t)(cnt?cnt:1)*B->len+1,sizeof(double));
       B->Abd=(double*)xcalloc((size_t)(cnt?cnt:1)*B->len+1,sizeof(double));
       for(int r=0;r<B->len;r++)cnts[B->off+r]=cnt;
       free(mark);free(list);}}
   rowp[0]=0;for(int r=0;r<mI;r++){rowp[r+1]=rowp[r]+cnts[r];tot+=cnts[r];}
   int*rowj=(int*)xmalloc((tot?tot:1)*sizeof(int));
   C.abarSrc=(int*)xmalloc((tot?tot:1)*sizeof(int));
   C.AbR.m=mI;C.AbR.n=n;C.AbR.nnz=tot;C.AbR.p=rowp;C.AbR.j=rowj;C.AbR.x=(double*)xcalloc(tot?tot:1,sizeof(double));
   for(int bi=0;bi<C.nblk;bi++){Blk*B=&C.blk[bi];
     if(B->type==CNONNEG){for(int r=0;r<B->len;r++){int rr=B->off+r,q0=rowp[rr],t=0;
       for(int p=C.AiR.p[rr];p<C.AiR.p[rr+1];p++,t++){rowj[q0+t]=C.AiR.j[p];C.abarSrc[q0+t]=p;}}}
     else{int*up=(int*)xmalloc((n?n:1)*sizeof(int));for(int i=0;i<n;i++)up[i]=-1;
       for(int c=0;c<B->nu;c++)up[B->ucols[c]]=c;
       for(int r=0;r<B->len;r++){int rr=B->off+r,q0=rowp[rr];
         for(int c=0;c<B->nu;c++){rowj[q0+c]=B->ucols[c];C.abarSrc[q0+c]=-1;}
         for(int p=C.AiR.p[rr];p<C.AiR.p[rr+1];p++){int c=up[C.AiR.j[p]];if(c>=0)C.abarSrc[q0+c]=p;}}
       free(up);}}
   free(cnts);
   /* CSC */
   Sp*AbC=sp_alloc(mI,n,tot);int*cnt=(int*)xcalloc(n+1,sizeof(int));
   for(int p=0;p<tot;p++)cnt[rowj[p]]++;
   for(int j=0;j<n;j++)AbC->p[j+1]=AbC->p[j]+cnt[j];
   memcpy(cnt,AbC->p,(n+1)*sizeof(int));
   int*csrOf=(int*)xmalloc((tot?tot:1)*sizeof(int));
   for(int i=0;i<mI;i++)for(int p=rowp[i];p<rowp[i+1];p++){int q=cnt[rowj[p]]++;AbC->i[q]=i;csrOf[q]=p;}
   free(cnt);C.AbC=AbC;C.ab2csr=csrOf;}

  /* ---- K pattern ---- */
  {int N=C.N;int*markRow=(int*)xmalloc((N?N:1)*sizeof(int));for(int i=0;i<N;i++)markRow[i]=-1;
   int**cl=(int**)xcalloc((N?N:1),sizeof(int*));int*cn=(int*)xcalloc(N?N:1,sizeof(int)),*cc=(int*)xcalloc(N?N:1,sizeof(int));
   #define INS(ii,jj) do{int _i=(ii),_j=(jj);if(markRow[_i]!=_j){markRow[_i]=_j;\
     if(cn[_j]==cc[_j]){cc[_j]=cc[_j]?cc[_j]*2:8;cl[_j]=(int*)realloc(cl[_j],cc[_j]*sizeof(int));}\
     cl[_j][cn[_j]++]=_i;}}while(0)
   for(int j=0;j<n;j++)for(int p=P->p[j];p<P->p[j+1];p++)INS(P->i[p],j);
   for(int j=0;j<n;j++)INS(j,j);
   for(int e=0;e<mE;e++)INS(n+e,n+e);
   for(int j=0;j<n;j++)for(int p=C.Ae->p[j];p<C.Ae->p[j+1];p++){INS(n+C.Ae->i[p],j);INS(j,n+C.Ae->i[p]);}
   for(int i=0;i<mI;i++){int a0=C.AbR.p[i],a1=C.AbR.p[i+1];
     for(int p=a0;p<a1;p++){int ja=C.AbR.j[p];for(int q2=a0;q2<a1;q2++)INS(C.AbR.j[q2],ja);}}
   #undef INS
   int tot=0;for(int j=0;j<N;j++)tot+=cn[j];
   Sp*K=sp_alloc(N,N,tot);
   for(int j=0;j<N;j++)K->p[j+1]=K->p[j]+cn[j];
   int kk=0;
   for(int j=0;j<N;j++){
     for(int u=0;u<cn[j];u++)for(int v=u+1;v<cn[j];v++)if(cl[j][v]<cl[j][u]){int t=cl[j][u];cl[j][u]=cl[j][v];cl[j][v]=t;}
     for(int u=0;u<cn[j];u++)K->i[kk++]=cl[j][u];}
   memset(K->x,0,(size_t)tot*sizeof(double));
   C.K=K;
   for(int j=0;j<N;j++)free(cl[j]);
   free(cl);free(cn);free(cc);free(markRow);}

  /* ---- initial point ---- */
  for(int bi=0;bi<C.nblk;bi++){
    Blk*B=&C.blk[bi];double*s=C.sI+B->off,*zz=C.zI+B->off;const double*bb=C.b+mE+B->off;
    if(B->type==CNONNEG){for(int i=0;i<B->len;i++){s[i]=(bb[i]>0?bb[i]:0)+1.0;zz[i]=1.0;}}
    else if(B->type==CSOC){int d=B->dim;double nb=0;for(int i=1;i<d;i++)nb+=bb[i]*bb[i];nb=sqrt(nb);
      s[0]=(bb[0]>nb?bb[0]:nb)+1.0;for(int i=1;i<d;i++)s[i]=bb[i];
      zz[0]=1.0;for(int i=1;i<d;i++)zz[i]=0.0;}
    else if(B->type==CEXP){double u=bb[0],v=bb[1],w=bb[2];
      double vp=fabs(u)+fabs(v)+fabs(w)+1.0;
      s[0]=u;s[1]=vp;s[2]=vp*exp(u/vp)+1.0;
      zz[0]=-1.0;zz[1]=1.0;zz[2]=1.0;}
    else{int d=B->d;svec_to_mat(d,bb,B->Sm);memcpy(B->evvec,B->Sm,(size_t)d*d*sizeof(double));
      eig_sym(d,B->evvec,B->evpsd,B->work,B->lwork,B->iwork,B->liwork);
      double sh=1.0-B->evpsd[0];if(sh<1.0)sh=1.0;
      memcpy(B->evvec,B->Sm,(size_t)d*d*sizeof(double));
      for(int i=0;i<d;i++)B->evvec[i*d+i]+=sh;
      mat_to_svec(d,B->evvec,s);
      int p=0;for(int c=0;c<d;c++)for(int r=0;r<=c;r++)zz[p++]=(r==c)?1.0:0.0;}
  }
  for(int e=0;e<mE;e++){C.s[e]=0.0;C.z[e]=0.0;}

  ldl_init(&C.ldl,C.N);
  C.rho=1e-8;C.delta=1e-8;

  int STATUSi=3; /* 0 solved, 1 pinf, 2 dinf, 3 unresolved */
  double *zpinf=NULL,*xdinf=NULL,*sdinf=NULL;
  double pinf_margin=0,dinf_margin=0;

  int maxit=250;
  for(int it=0;it<maxit;it++){
    if(now_sec()-t_start>time_limit)break;
    /* residuals */
    {double*Ax=C.w1;A_mul(&C,C.x,Ax);
     for(int e=0;e<mE;e++)C.rpE[e]=Ax[e]-C.b[e];
     for(int i=0;i<mI;i++)C.rpI[i]=Ax[mE+i]+C.sI[i]-C.b[mE+i];}
    {double*Px=C.w2,*ATz=C.w3;P_mul(&C,C.x,Px);AT_mul(&C,C.z,ATz);
     for(int j=0;j<n;j++)C.rd[j]=Px[j]+C.q[j]+ATz[j];}
    double sz=0;for(int i=0;i<mI;i++)sz+=C.sI[i]*C.zI[i];
    double mu=sz/C.nu_total;
    if(!(mu>0))mu=1e-16;

    /* metrics */
    double rp,rd,pk,dk,ga,gr;
    metrics(&C,&rp,&rd,&pk,&dk,&ga,&gr);
    int ok=(rp<=5e-9&&rd<=5e-9&&pk<=5e-9&&dk<=5e-9&&(ga<=5e-9||gr<=5e-9));
    if(dbg)fprintf(stderr,"it %d mu=%.3e rp=%.2e rd=%.2e pk=%.2e dk=%.2e gap=%.2e/%.2e%s\n",it,mu,rp,rd,pk,dk,ga,gr,ok?" OK":"");
    if(ok){STATUSi=0;break;}
    /* infeasibility probes */
    if(it>0){
      if(!zpinf){zpinf=(double*)xmalloc((m?m:1)*sizeof(double));if(check_pinf(&C,zpinf)){STATUSi=1;break;}}
      if(!xdinf){xdinf=(double*)xmalloc((n?n:1)*sizeof(double));sdinf=(double*)xmalloc((m?m:1)*sizeof(double));
        if(check_dinf(&C,xdinf,sdinf)){STATUSi=2;break;}}
    }

    /* scaling */
    for(int bi=0;bi<C.nblk;bi++)blk_scaling(&C,&C.blk[bi],mu);
    abar_fill(&C);

    /* assemble + factor */
    kkt_assemble(&C);
    if(!C.orderdone){
      long best=-1;int bestk=-1;
      for(int cand=0;cand<2;cand++){
        int*pm=(int*)xmalloc(C.N*sizeof(int));
        if(cand==0)ord_natural(pm,C.N);else ord_rcm(C.K,C.N,pm);
        ldl_setperm(&C.ldl,pm);
        int r=ldl_factor(&C.ldl,C.K);
        if(dbg)fprintf(stderr,"order cand %d nnzL=%ld r=%d\n",cand,C.ldl.nnzL,r);
        if(r==0&&(best<0||C.ldl.nnzL<best)){best=C.ldl.nnzL;bestk=cand;memcpy(C.perm,pm,C.N*sizeof(int));}
        free(pm);
      }
      if(bestk<0){
        C.rho*=1e4;C.delta*=1e4;
        ord_natural(C.perm,C.N);ldl_setperm(&C.ldl,C.perm);
        if(ldl_factor(&C.ldl,C.K)!=0){break;}
      }else{
        ldl_setperm(&C.ldl,C.perm);
        if(ldl_factor(&C.ldl,C.K)!=0){C.rho*=1e4;C.delta*=1e4;kkt_assemble(&C);if(ldl_factor(&C.ldl,C.K)!=0)break;}
      }
      C.orderdone=1;
    }else{
      int r=ldl_factor(&C.ldl,C.K);
      if(r!=0){
        for(int try1=0;try1<6;try1++){
          C.rho*=10;C.delta*=10;kkt_assemble(&C);
          if(ldl_factor(&C.ldl,C.K)==0){r=0;break;}}
        if(r!=0)break;
      }
    }

    /* affine step */
    apply_Winv_all(&C,C.rpI,C.winvrp);
    rc_all(&C,mu,0.0,NULL,NULL,C.rc);
    solve_dir(&C,C.rc,C.Abdx,C.winvrp);
    memcpy(C.dshat,C.dshat,mI*sizeof(double));
    double*ds_aff=(double*)xmalloc((m?m:1)*sizeof(double));
    double*dz_aff=(double*)xmalloc((m?m:1)*sizeof(double));
    memcpy(ds_aff,C.ds,m*sizeof(double));memcpy(dz_aff,C.dz,m*sizeof(double));
    double*dsh_aff=(double*)xmalloc((mI?mI:1)*sizeof(double));
    double*dzh_aff=(double*)xmalloc((mI?mI:1)*sizeof(double));
    memcpy(dsh_aff,C.dshat,mI*sizeof(double));memcpy(dzh_aff,C.dzhat,mI*sizeof(double));
    double aaff=step_len(&C,ds_aff,dz_aff);
    double aa=0.995*aaff;if(aa>1)aa=1;
    /* mu at affine step */
    double sza=0;
    for(int bi=0;bi<C.nblk;bi++){Blk*B=&C.blk[bi];
      for(int i=0;i<B->len;i++)sza+=(C.sI[B->off+i]+aa*ds_aff[mE+B->off+i])*(C.zI[B->off+i]+aa*dz_aff[mE+B->off+i]);}
    double mua=sza/C.nu_total;
    double sig=mua/mu;if(!(sig>0))sig=0;
    sig=sig*sig*sig;if(sig>1)sig=1;if(sig<1e-8)sig=1e-8;

    /* corrector with retry over sigma */
    int done=0;
    for(int rr=0;rr<4&&!done;rr++){
      double sg = rr==0?sig:(rr==1?0.3:(rr==2?0.7:0.95));
      rc_all(&C,mu,sg,dsh_aff,dzh_aff,C.rc);
      solve_dir(&C,C.rc,C.Abdx,C.winvrp);
      int bad=0;
      for(int i=0;i<mI;i++)if(!isfinite(C.dshat[i])||!isfinite(C.dzhat[i]))bad=1;
      for(int i=0;i<m;i++)if(!isfinite(C.ds[i])||!isfinite(C.dz[i]))bad=1;
      for(int j=0;j<n;j++)if(!isfinite(C.dx[j]))bad=1;
      if(bad)continue;
      double al=step_len(&C,C.ds,C.dz);
      double alpha=0.995*al;if(alpha>1)alpha=1;
      if(!(alpha>0)||!isfinite(alpha))continue;
      {double *xs=(double*)xmalloc((n?n:1)*sizeof(double));
       double *ss=(double*)xmalloc((m?m:1)*sizeof(double));
       double *zz=(double*)xmalloc((m?m:1)*sizeof(double));
       memcpy(xs,C.x,n*sizeof(double));memcpy(ss,C.s,m*sizeof(double));memcpy(zz,C.z,m*sizeof(double));
       for(int tr=0;tr<30;tr++){
         for(int j=0;j<n;j++)C.x[j]=xs[j]+alpha*C.dx[j];
         for(int i=0;i<m;i++){C.s[i]=ss[i]+alpha*C.ds[i];C.z[i]=zz[i]+alpha*C.dz[i];}
         for(int e=0;e<mE;e++)C.s[e]=0.0;
         if(point_inside(&C,C.s,C.z)){done=1;break;}
         alpha*=0.5;
         if(alpha<1e-16)break;
       }
       if(!done){memcpy(C.x,xs,n*sizeof(double));memcpy(C.s,ss,m*sizeof(double));memcpy(C.z,zz,m*sizeof(double));}
       free(xs);free(ss);free(zz);}
    }
    if(!done)break;
    free(ds_aff);free(dz_aff);free(dsh_aff);free(dzh_aff);
  }

  /* final metrics */
  {double rp,rd,pk,dk,ga,gr;metrics(&C,&rp,&rd,&pk,&dk,&ga,&gr);
   if(dbg)fprintf(stderr,"final rp=%.3e rd=%.3e pk=%.3e dk=%.3e gap=%.3e/%.3e\n",rp,rd,pk,dk,ga,gr);
   if(STATUSi==3&&rp<=9e-9&&rd<=9e-9&&pk<=9e-9&&dk<=9e-9&&(ga<=9e-9||gr<=9e-9))STATUSi=0;
   if(STATUSi==3){
     /* polish variant: enforce s = b - A x exactly (kills primal residual) */
     double *sold=(double*)xmalloc((m?m:1)*sizeof(double));memcpy(sold,C.s,m*sizeof(double));
     double *Ax=(double*)xmalloc((m?m:1)*sizeof(double));A_mul(&C,C.x,Ax);
     for(int i=0;i<m;i++)C.s[i]=C.b[i]-Ax[i];
     for(int e=0;e<mE;e++)C.s[e]=0.0;
     double rp2,rd2,pk2,dk2,ga2,gr2;metrics(&C,&rp2,&rd2,&pk2,&dk2,&ga2,&gr2);
     if(dbg)fprintf(stderr,"polish rp=%.3e pk=%.3e gap=%.3e/%.3e\n",rp2,pk2,ga2,gr2);
     if(rp2<=9e-9&&rd2<=9e-9&&pk2<=9e-9&&dk2<=9e-9&&(ga2<=9e-9||gr2<=9e-9))STATUSi=0;
     else memcpy(C.s,sold,m*sizeof(double));
     free(sold);free(Ax);
   }
   if(STATUSi==3){
     double*zt=(double*)xmalloc((m?m:1)*sizeof(double));
     if(check_pinf(&C,zt)){STATUSi=1;free(zpinf);zpinf=zt;}else{free(zt);
       double*xt=(double*)xmalloc((n?n:1)*sizeof(double)),*st=(double*)xmalloc((m?m:1)*sizeof(double));
       if(check_dinf(&C,xt,st)){STATUSi=2;free(xdinf);free(sdinf);xdinf=xt;sdinf=st;}else{free(xt);free(st);}}}
  }

  if(STATUSi==0){
    STATUS="solved";
    /* unpermute s,z */
    outx=(double*)xmalloc((n?n:1)*sizeof(double));memcpy(outx,C.x,n*sizeof(double));oxn=n;
    outs=(double*)xmalloc((m?m:1)*sizeof(double));outz=(double*)xmalloc((m?m:1)*sizeof(double));
    for(int i=0;i<m;i++){outs[rowperm[i]]=C.s[i];outz[rowperm[i]]=C.z[i];}
    osn=m;ozn=m;
  }else if(STATUSi==1){
    STATUS="primal_infeasible";
    outz=(double*)xmalloc((m?m:1)*sizeof(double));
    for(int i=0;i<m;i++)outz[rowperm[i]]=zpinf[i];
    ozn=m;
  }else if(STATUSi==2){
    STATUS="dual_infeasible";
    outx=(double*)xmalloc((n?n:1)*sizeof(double));memcpy(outx,xdinf,n*sizeof(double));oxn=n;
    outs=(double*)xmalloc((m?m:1)*sizeof(double));
    for(int i=0;i<m;i++)outs[rowperm[i]]=sdinf[i];
    osn=m;
  }
  }
done:
  {char mj[256];
   if(!strcmp(STATUS,"solved"))snprintf(mj,sizeof mj,"{\"schema_version\": 1, \"status\": \"solved\"}");
   else snprintf(mj,sizeof mj,"{\"schema_version\": 1, \"status\": \"%s\"}",STATUS);
   npzw_add_ustr(&W,"metadata_json",mj);
   if(outx)npzw_add_f8(&W,"x",outx,oxn);else if(osn||ozn){}
   if(outs)npzw_add_f8(&W,"s",outs,osn);
   if(outz)npzw_add_f8(&W,"z",outz,ozn);
   if(!outx&&osn==0&&ozn==0){/* unresolved: no vectors needed */}
   if(npzw_write(&W,outpath)!=0){fprintf(stderr,"write fail\n");return 2;}
  }
  npzw_free(&W);
  return 0;
}
