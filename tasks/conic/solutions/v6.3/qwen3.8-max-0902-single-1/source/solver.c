/* Conic solver: homogeneous self-dual ADMM (Douglas-Rachford) + Anderson accel.
 * min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K   (K = zero/nonneg/soc/exp/psd blocks)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "npz.h"
#include "expcone.h"

void dpotrf_(const char *, const int *, double *, const int *, int *);
void dpotrs_(const char *, const int *, const int *, const double *, const int *, double *, const int *, int *);
void dsyevd_(const char *, const char *, const int *, double *, const int *, double *, double *, int *, int *, int *, int *);
void dposv_(const char *, const int *, const int *, double *, const int *, double *, const int *, int *);

#define CZERO 0
#define CNONNEG 1
#define CSOC 2
#define CEXP 3
#define CPSD 4
#define EPS 1e-8

static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+1e-9*ts.tv_nsec;}
static void *xmalloc(size_t k){void*p=calloc(1,k?k:1);if(!p){fprintf(stderr,"oom\n");exit(3);}return p;}

typedef struct {
  long n,m; long *Ap; long *Ai; double *Av; long *Pp; long *Pi; double *Pv;
  double *q,*b; long nnzA,nnzP;
  int nc; int *cty; long *cst,*cdim;
  double maxA,maxP,normb,normq;
  double *exs;
} Prob;

static void mv_A(const Prob*p,const double*x,double*o){long j,k;memset(o,0,sizeof(double)*p->m);
  for(j=0;j<p->n;j++){double t=x[j];if(t==0.0)continue;for(k=p->Ap[j];k<p->Ap[j+1];k++)o[p->Ai[k]]+=p->Av[k]*t;}}
static void mv_AT(const Prob*p,const double*y,double*o){long j,k;
  for(j=0;j<p->n;j++){double s=0;for(k=p->Ap[j];k<p->Ap[j+1];k++)s+=p->Av[k]*y[p->Ai[k]];o[j]=s;}}
static void mv_P(const Prob*p,const double*x,double*o){long j,k;memset(o,0,sizeof(double)*p->n);if(!p->Pv)return;
  for(j=0;j<p->n;j++){double t=x[j];if(t==0.0)continue;for(k=p->Pp[j];k<p->Pp[j+1];k++)o[p->Pi[k]]+=p->Pv[k]*t;}}
static double nrmi(const double*v,long n){double m=0;long i;for(i=0;i<n;i++){double a=fabs(v[i]);if(a>m)m=a;}return m;}
static double dot(const double*a,const double*b,long n){double s=0;long i;for(i=0;i<n;i++)s+=a[i]*b[i];return s;}

static void smat(const double*v,long d,double*M){long j,i,k=0;const double ir=1.0/sqrt(2.0);
  for(j=0;j<d;j++)for(i=0;i<=j;i++){double val=(i==j)?v[k]:v[k]*ir;M[i*d+j]=val;M[j*d+i]=val;k++;}}
static void vecmat(const double*M,long d,double*v){long j,i,k=0;const double r2=sqrt(2.0);
  for(j=0;j<d;j++)for(i=0;i<=j;i++){v[k++]=(i==j)?M[i*d+j]:M[i*d+j]*r2;}}

typedef struct{double*M,*w,*work;int*iwork;int lw;long dmax;}PsdScratch;
static void psd_init(PsdScratch*s,long dmax){s->dmax=dmax;if(dmax<=0){s->M=NULL;return;}
  s->lw=(int)(2*dmax*dmax+8*dmax+64);
  s->M=(double*)xmalloc(sizeof(double)*(dmax*dmax+dmax+s->lw));s->w=s->M+dmax*dmax;s->work=s->w+dmax;
  s->iwork=(int*)xmalloc(sizeof(int)*(3+5*dmax+8));}
static int eig_sym(double*M,int n,double*w,char jobz,double*work,int lw,int*iwork){
  int info,liw=3+5*n;char up='U';dsyevd_(&jobz,&up,&n,M,&n,w,work,&lw,iwork,&liw,&info);return info;}
static void proj_psd(double*v,long d,PsdScratch*ps){int n=(int)d,info,i;long j;char jz='V';
  if(!ps->M)return; smat(v,d,ps->M); info=eig_sym(ps->M,n,ps->w,jz,ps->work,ps->lw,ps->iwork); if(info)return;
  for(i=0;i<n;i++)if(ps->w[i]<0)ps->w[i]=0;
  for(i=0;i<n;i++)for(j=0;j<n;j++)ps->M[(long)i*n+j]*=ps->w[j];
  {double*T=(double*)xmalloc(sizeof(double)*n*n);long a,b;
   for(a=0;a<n;a++)for(b=a;b<n;b++){double s=0;long t;for(t=0;t<n;t++)s+=ps->M[t*n+a]*ps->M[t*n+b];T[a*n+b]=s;T[b*n+a]=s;}
   memcpy(ps->M,T,sizeof(double)*n*n);free(T);}
  vecmat(ps->M,d,v);}
static double psd_defect(const double*v,long d,PsdScratch*ps){int n=(int)d,info;double me;
  if(!ps->M)return 0.0; smat(v,d,ps->M); info=eig_sym(ps->M,n,ps->w,'N',ps->work,ps->lw,ps->iwork);
  if(info)return 0.0; me=ps->w[0]; return me<0?-me:0;}

static void proj_dual_block(const Prob*p,int ci,double*v,PsdScratch*ps){
  long st=p->cst[ci],d=p->cdim[ci],i;
  switch(p->cty[ci]){
    case CZERO:break;
    case CNONNEG:for(i=st;i<st+d;i++)if(v[i]<0)v[i]=0;break;
    case CSOC:{double t=v[st],nn=0,a;for(i=st+1;i<st+d;i++)nn+=v[i]*v[i];nn=sqrt(nn);
      if(t>=nn)break; if(t<=-nn){for(i=st;i<st+d;i++)v[i]=0;break;}
      a=0.5*(t+nn)/(nn>1e-300?nn:1.0); v[st]=0.5*(t+nn); for(i=st+1;i<st+d;i++)v[i]*=a; break;}
    case CEXP:for(i=st;i+3<=st+d;i+=3){double tmp[3];proj_exp_dual(v+i,tmp);v[i]=tmp[0];v[i+1]=tmp[1];v[i+2]=tmp[2];}break;
    case CPSD:proj_psd(v+st,d,ps);break;}}
static void project_dual(const Prob*p,double*v,PsdScratch*ps){int ci;for(ci=0;ci<p->nc;ci++)proj_dual_block(p,ci,v,ps);}
static double expdefect(double a,double b,double c){double B=fabs(b),t;
  if(a>B)B=a; if(-c>B)B=-c;
  if(b>0&&c>0){t=a-b*(log(c)-log(b));if(t<0)t=0;return t<B?t:B;} return B;}
static double cone_defect(const Prob*p,const double*v,int dual,PsdScratch*ps){
  int ci;long i;double worst=0;
  for(ci=0;ci<p->nc;ci++){long st=p->cst[ci],d=p->cdim[ci];double dv=0,mx=0;
    for(i=st;i<st+d;i++){double a=fabs(v[i]);if(a>mx)mx=a;} if(mx<1.0)mx=1.0;
    switch(p->cty[ci]){
      case CZERO:if(!dual){dv=0;for(i=st;i<st+d;i++){double a=fabs(v[i])/mx;if(a>dv)dv=a;}}break;
      case CNONNEG:{double mn=0;for(i=st;i<st+d;i++){double u=v[i]/mx;if(u<mn)mn=u;}dv=mn<0?-mn:0;break;}
      case CSOC:{double t=v[st]/mx,nn=0;for(i=st+1;i<st+d;i++){double u=v[i]/mx;nn+=u*u;}nn=sqrt(nn);dv=nn-t;if(dv<0)dv=0;break;}
      case CEXP:for(i=st;i+3<=st+d;i+=3){double a=v[i]/mx,b=v[i+1]/mx,c=v[i+2]/mx,e;
        e=dual?expdefect(-b,-a,M_E*c):expdefect(a,b,c); if(e>dv)dv=e;}break;
      case CPSD:{double me=psd_defect(v+st,d,ps)/mx;dv=me>0?me:0;break;}}
    if(dv>worst)worst=dv;}
  return worst;}

static const char*memstr(const char*h,size_t n,const char*pat){size_t pl=strlen(pat);size_t i;
  if(pl>n)return NULL;for(i=0;i+pl<=n;i++)if(!memcmp(h+i,pat,pl))return h+i;return NULL;}
static int parse_cones(const char*js,Prob*p){
  const char*c=strstr(js,"\"cones\"");const char*q,*e;char*sub;long pos=0;int cap=16,cnt=0;
  if(!c)return -1; c=strchr(c,'['); if(!c)return -1;
  {int depth=0;e=c;while(*e){if(*e=='[')depth++;else if(*e==']'){depth--;if(!depth){e++;break;}}e++;}}
  sub=(char*)xmalloc(e-c+1);memcpy(sub,c,e-c);sub[e-c]=0;
  p->cty=(int*)xmalloc(sizeof(int)*cap);p->cst=(long*)xmalloc(sizeof(long)*cap);p->cdim=(long*)xmalloc(sizeof(long)*cap);
  q=sub;
  while((q=strchr(q,'{'))!=NULL){
    const char*rb=strchr(q,'}');const char*tq,*dm;char tb[32];size_t L;long dim=0;int ty=-1;
    if(!rb)break;
    tq=memstr(q,rb-q,"\"type\"");
    dm=memstr(q,rb-q,"\"dim\"");
    if(tq&&dm){
      const char*q1=strchr(tq+6,'\"'),*q2;
      if(q1&&q1<rb&&(q2=strchr(q1+1,'\"'))&&q2<=rb){L=q2-q1-1;memcpy(tb,q1+1,L);tb[L]=0;
        if(!strcmp(tb,"zero"))ty=CZERO;else if(!strcmp(tb,"nonnegative"))ty=CNONNEG;
        else if(!strcmp(tb,"soc"))ty=CSOC;else if(!strcmp(tb,"exponential"))ty=CEXP;
        else if(!strcmp(tb,"psd"))ty=CPSD;else{fprintf(stderr,"bad cone\n");free(sub);return -1;}}
      dim=strtol(strchr(dm,':')+1,NULL,10);
    } else { q=rb+1; continue; }
    if(cnt>=cap){cap*=2;p->cty=realloc(p->cty,sizeof(int)*cap);p->cst=realloc(p->cst,sizeof(long)*cap);p->cdim=realloc(p->cdim,sizeof(long)*cap);}
    p->cty[cnt]=ty;p->cst[cnt]=pos;p->cdim[cnt]=dim;cnt++;
    pos+=(ty==CPSD)?dim*(dim+1)/2:dim;
    q=rb+1;}
  free(sub);p->nc=cnt;
  if(pos!=p->m){fprintf(stderr,"cone rows %ld != m %ld\n",pos,p->m);return -1;}
  return 0;}

typedef struct{int dense;long n;double*G,*dg;const Prob*p;double gam;double*t1,*t2,*t3,*Ap;}LinSys;
static void Gmvp(const LinSys*L,const double*x,double*o){const Prob*p=L->p;double g=L->gam,g2=g*g;long i;
  mv_A(p,x,L->t1);mv_AT(p,L->t1,o);
  for(i=0;i<p->n;i++)o[i]*=g2;
  if(p->Pv){mv_P(p,x,L->t2);for(i=0;i<p->n;i++)o[i]+=g*L->t2[i];}
  for(i=0;i<p->n;i++)o[i]+=x[i];}
static int pcg(const LinSys*L,const double*rhs,double*x,double tol,int maxit){
  long n=L->p->n,i;double*r=L->t3,*zz=L->t1,*pp=L->t2,*Ap=L->Ap;
  double rs0=0,rsold,alpha,rsnew,beta_;int it=0;
  Gmvp(L,x,Ap);
  for(i=0;i<n;i++)r[i]=rhs[i]-Ap[i];
  rs0=nrmi(rhs,n);if(rs0<1e-300)rs0=1;
  if(nrmi(r,n)<=tol*rs0)return 0;
  for(i=0;i<n;i++)zz[i]=r[i]/L->dg[i];
  memcpy(pp,zz,sizeof(double)*n);rsold=dot(r,zz,n);
  for(it=0;it<maxit;it++){
    Gmvp(L,pp,Ap);alpha=dot(pp,Ap,n);
    if(!(alpha>1e-300))break;alpha=rsold/alpha;
    for(i=0;i<n;i++){x[i]+=alpha*pp[i];r[i]-=alpha*Ap[i];}
    if(nrmi(r,n)<=tol*rs0)break;
    for(i=0;i<n;i++)zz[i]=r[i]/L->dg[i];
    rsnew=dot(r,zz,n);beta_=rsnew/(rsold+1e-300);
    for(i=0;i<n;i++)pp[i]=zz[i]+beta_*pp[i];rsold=rsnew;}
  return it;}

static Prob P; static NpzProblem np;
static double *ut,*u,*zv;
static LinSys L;
static double *cc,*ee,*gc,*wf,*t1,*t2;
static double beta_,delta_,gam;
static PsdScratch ps;

static void SOLVE(void){long i,n=P.n,m=P.m;double sg=dot(P.b,u+n,m),den=beta_-delta_;
  mv_AT(&P,u+n,t1);
  for(i=0;i<n;i++)wf[i]=u[i]-gam*t1[i];
  if(L.dense){int ni=(int)n,one=1,info;char up='L';dpotrs_(&up,&ni,&one,L.G,&ni,wf,&ni,&info);}
  else pcg(&L,wf,wf,1e-11,3000);
  ut[n+m]=(u[n+m]+gam*sg+dot(ee,wf,n))/den;
  for(i=0;i<n;i++)ut[i]=wf[i]+ut[n+m]*gc[i];
  mv_A(&P,ut,t2);
  for(i=0;i<m;i++)ut[n+i]=u[n+i]+gam*(t2[i]-P.b[i]*ut[n+m]);}

int main(int argc,char**argv){
  const char*inpath=NULL,*outpath=NULL;int i;double t0=now();
  long n,m,N;double hardcap=280.0;long maxit=3000000;int status=3;
  double *xout,*sout,*zout,*axb,*pxp,*atz,*xs;double qnorig=0;static double fin_rp,fin_rd,fin_gap,fin_cds,fin_cdz;
  char jsonbuf[256];
  for(i=1;i<argc;i++){if(!strcmp(argv[i],"--input")&&i+1<argc)inpath=argv[++i];
    else if(!strcmp(argv[i],"--output")&&i+1<argc)outpath=argv[++i];}
  if(!inpath||!outpath){fprintf(stderr,"usage\n");return 2;}
  if(getenv("SOLVER_CAP"))hardcap=atof(getenv("SOLVER_CAP"));
  if(getenv("SOLVER_MAXIT"))maxit=atol(getenv("SOLVER_MAXIT"));
  if(getenv("SOLVER_GAM"))gam=atof(getenv("SOLVER_GAM"));
  if(npz_read(inpath,&np)){fprintf(stderr,"read fail\n");return 0;}
  memset(&P,0,sizeof(P));
  n=np.n>0?np.n:np.An;m=np.m>0?np.m:np.Am;
  P.n=n;P.m=m;P.Ap=np.A_indptr;P.Ai=np.A_indices;P.Av=np.A_data;
  P.Pp=np.P_indptr;P.Pi=np.P_indices;P.Pv=np.P_data;P.q=np.q;P.b=np.b;
  P.nnzA=P.Ap?P.Ap[n]:0;P.nnzP=P.Pp?P.Pp[n]:0;
  if(!P.Ap){P.Ap=(long*)xmalloc(sizeof(long)*(n+1));P.Ai=(long*)xmalloc(sizeof(long));}
  if(!P.Pp){P.Pp=(long*)xmalloc(sizeof(long)*(n+1));P.Pv=NULL;}
  if(!P.q)P.q=(double*)xmalloc(sizeof(double)*(n?n:1));
  if(!P.b)P.b=(double*)xmalloc(sizeof(double)*(m?m:1));
  if(parse_cones(np.metadata_json?np.metadata_json:"",&P)){fprintf(stderr,"cones fail\n");return 0;}
  P.maxA=0;for(i=0;i<P.nnzA;i++)if(fabs(P.Av[i])>P.maxA)P.maxA=fabs(P.Av[i]);if(P.maxA==0)P.maxA=1;
  P.maxP=0;for(i=0;i<P.nnzP;i++)if(fabs(P.Pv[i])>P.maxP)P.maxP=fabs(P.Pv[i]);if(P.maxP==0)P.maxP=1;
  P.normb=nrmi(P.b,m);P.normq=nrmi(P.q,n);
  {long npsd=0,dmax=0;for(i=0;i<P.nc;i++)if(P.cty[i]==CPSD){npsd++;if(P.cdim[i]>dmax)dmax=P.cdim[i];}psd_init(&ps,dmax);}

  /* ---- column scaling (exact): A'=A Ex, P'=Ex P Ex, q'=Ex q ; x=Ex x', s,z unchanged ---- */
  {double*exs=(double*)xmalloc(sizeof(double)*(n+1));long j;int r;
   for(j=0;j<n;j++)exs[j]=1.0;
   for(r=0;r<2;r++){double*cm=(double*)xmalloc(sizeof(double)*(n?n:1));
     for(j=0;j<n;j++){double mx=0;long t;
       for(t=P.Ap[j];t<P.Ap[j+1];t++){double a=fabs(P.Av[t]);if(a>mx)mx=a;}
       if(P.Pv)for(t=P.Pp[j];t<P.Pp[j+1];t++){double a=sqrt(fabs(P.Pv[t]));if(a>mx)mx=a;}
       cm[j]=mx>1e-12?1.0/sqrt(mx):1.0;}
     for(j=0;j<n;j++){double f=cm[j];exs[j]*=f;long t;
       for(t=P.Ap[j];t<P.Ap[j+1];t++)P.Av[t]*=f;
       if(P.Pv)for(t=P.Pp[j];t<P.Pp[j+1];t++)P.Pv[t]*=f;
       P.q[j]*=f;}
     if(P.Pv)for(j=0;j<n;j++){long t;for(t=P.Pp[j];t<P.Pp[j+1];t++)P.Pv[t]*=cm[P.Pi[t]];}
     free(cm);}
   P.exs=exs;}

  if(!getenv("SOLVER_GAM"))gam=1.0;
  if(gam<=0)gam=1.0;

  N=n+m+1;L.p=&P;L.gam=gam;L.n=n;
  L.t1=(double*)xmalloc(sizeof(double)*(m?m:1));L.t2=(double*)xmalloc(sizeof(double)*(n?n:1));
  L.t3=(double*)xmalloc(sizeof(double)*(n?n:1));L.Ap=(double*)xmalloc(sizeof(double)*(n?n:1));
  L.dg=(double*)xmalloc(sizeof(double)*(n?n:1));
  for(i=0;i<n;i++)L.dg[i]=1.0+1e-12;
  if(P.Pv){long j;for(j=0;j<n;j++){long t;for(t=P.Pp[j];t<P.Pp[j+1];t++)if(P.Pi[t]==j)L.dg[j]+=gam*P.Pv[t];}}
  {double*cs=(double*)xmalloc(sizeof(double)*(n?n:1));long j,k;
   for(j=0;j<n;j++){double s=0;for(k=P.Ap[j];k<P.Ap[j+1];k++)s+=P.Av[k]*P.Av[k];cs[j]=s;}
   for(i=0;i<n;i++){L.dg[i]+=gam*gam*cs[i];if(!(L.dg[i]>1e-30))L.dg[i]=1.0;}free(cs);}
  L.dense=(n<=1500);L.G=NULL;
  if(L.dense&&n>0){double*G=(double*)xmalloc(sizeof(double)*n*n);int ni=(int)n,info;char up='L';long j,k;
    double*col=(double*)xmalloc(sizeof(double)*(m?m:1));
    memset(G,0,sizeof(double)*n*n);
    for(j=0;j<n;j++){memset(col,0,sizeof(double)*m);
      for(k=P.Ap[j];k<P.Ap[j+1];k++)col[P.Ai[k]]=P.Av[k];
      for(k=P.Ap[j];k<P.Ap[j+1];k++){long r=P.Ai[k];if(r>=j){double vv=gam*gam*P.Av[k]*col[r];G[r*n+j]+=vv;if(r!=j)G[j*n+r]+=vv;}}}
    free(col);
    if(P.Pv)for(j=0;j<n;j++)for(k=P.Pp[j];k<P.Pp[j+1];k++)G[(long)P.Pi[k]*n+j]+=gam*P.Pv[k];
    for(i=0;i<n;i++)G[i*n+i]+=1.0+1e-12;
    dpotrf_(&up,&ni,G,&ni,&info);
    if(info){L.dense=0;free(G);}else L.G=G;}

  cc=(double*)xmalloc(sizeof(double)*(n?n:1));ee=(double*)xmalloc(sizeof(double)*(n?n:1));
  gc=(double*)xmalloc(sizeof(double)*(n?n:1));wf=(double*)xmalloc(sizeof(double)*(n?n:1));
  t1=(double*)xmalloc(sizeof(double)*(n?n:1));t2=(double*)xmalloc(sizeof(double)*(m?m:1));
  u=(double*)xmalloc(sizeof(double)*N);ut=(double*)xmalloc(sizeof(double)*N);zv=(double*)xmalloc(sizeof(double)*N);
  mv_AT(&P,P.b,t1);
  for(i=0;i<n;i++){cc[i]=gam*gam*t1[i]-gam*P.q[i];ee[i]=gam*P.q[i]+gam*gam*t1[i];}
  memcpy(gc,cc,sizeof(double)*n);
  if(n>0){if(L.dense){int ni=(int)n,one=1,info;char up='L';dpotrs_(&up,&ni,&one,L.G,&ni,gc,&ni,&info);}
    else pcg(&L,cc,gc,1e-12,20000);}
  beta_=1.0+gam*gam*dot(P.b,P.b,m);delta_=dot(ee,gc,n);
  if(fabs(beta_-delta_)<1e-300){fprintf(stderr,"singular\n");beta_+=1e-12;}

  xout=(double*)xmalloc(sizeof(double)*(n?n:1));sout=(double*)xmalloc(sizeof(double)*(m?m:1));
  zout=(double*)xmalloc(sizeof(double)*(m?m:1));axb=(double*)xmalloc(sizeof(double)*(m?m:1));
  pxp=(double*)xmalloc(sizeof(double)*(n?n:1));atz=(double*)xmalloc(sizeof(double)*(n?n:1));xs=(double*)xmalloc(sizeof(double)*(n?n:1));
  for(i=0;i<n;i++){double a=fabs(P.q[i]/P.exs[i]);if(a>qnorig)qnorig=a;}
  memset(u,0,sizeof(double)*N);u[n+m]=1.0;memset(ut,0,sizeof(double)*N);

  { /* main loop with Anderson acceleration */
    int AAmem=(N<=300000)?10:(N<=2000000?4:0);
    double *hx=NULL,*hg=NULL,*AA=NULL,*rhsa=NULL,*sol=NULL;
    double tolr=3e-9,best=1e300;long it;int hist=0,slot=0,restart=0;int final_eval=0,pending=0;
    double prevg=1e300;
    if(getenv("SOLVER_NOAA"))AAmem=0;
    if(AAmem){hx=(double*)xmalloc(sizeof(double)*N*AAmem);hg=(double*)xmalloc(sizeof(double)*N*AAmem);
      AA=(double*)xmalloc(sizeof(double)*AAmem*AAmem);rhsa=(double*)xmalloc(sizeof(double)*AAmem);
      sol=(double*)xmalloc(sizeof(double)*AAmem);}
    for(it=0;it<maxit;){
      double gn=0;long Nl=N;
      if(it>=maxit-1)final_eval=1;
      SOLVE();
      for(i=0;i<Nl;i++)zv[i]=2*ut[i]-u[i];
      project_dual(&P,zv+n,&ps);
      if(zv[n+m]<0)zv[n+m]=0;
      /* g = T(u)-u = z - ut ; u += g */
      {double s2=0;for(i=0;i<Nl;i++){double g=zv[i]-ut[i];u[i]+=g;s2+=g*g;}gn=sqrt(s2);}
      if(getenv("SOLVER_DBG2")&&(it<4||it%50==49))fprintf(stderr,"k=%ld u=(%.6f,%.6f,%.6f) ut=(%.6f,%.6f,%.6f) gam=%.4f G00=%.4f\n",it,u[0],u[1],u[2],ut[0],ut[1],ut[2],gam,L.dense?L.G[0]*L.G[0]:-1.0);
      /* Anderson */
      if(AAmem&&hist>=1){
        int mk=hist<AAmem?hist:AAmem,j2,k2,ok=1;double ridge=0;
        for(j2=0;j2<mk;j2++){
          double*c1=hg+((slot-j2-1+2*AAmem)%AAmem)*N,*c0=hg+((slot-j2-2+2*AAmem)%AAmem)*N;
          double*d=hg+((slot-j2-1+2*AAmem)%AAmem)*N;(void)c1;(void)c0;(void)d;}
        /* build normal equations: columns dG_j = g_{k-j} - g_{k-j-1} */
        {double *gk=zv; /* current residual stored at slot */
         double *gcur=(double*)xmalloc(sizeof(double)*N);
         for(i=0;i<Nl;i++)gcur[i]=zv[i]-ut[i];
         /* store into history at position slot */
         memcpy(hx+((slot)%AAmem)*N,u,sizeof(double)*N);
         memcpy(hg+((slot)%AAmem)*N,gcur,sizeof(double)*N);
         for(j2=0;j2<mk;j2++){
           int i1=(slot-j2+AAmem)%AAmem,i0=(slot-j2-1+AAmem)%AAmem;
           double*d1=hg+i1*N,*d0=hg+i0*N,*x1=hx+i1*N,*x0=hx+i0*N;
           for(k2=j2;k2<mk;k2++){
             int i1b=(slot-k2+AAmem)%AAmem,i0b=(slot-k2-1+AAmem)%AAmem;
             double s=0;for(i=0;i<Nl;i++)s+=(d1[i]-d0[i])*(hg[i1b*N+i]-hg[i0b*N+i]);
             AA[j2*mk+k2]=s;AA[k2*mk+j2]=s;}
           rhsa[j2]=0;for(i=0;i<Nl;i++)rhsa[j2]+=(d1[i]-d0[i])*gcur[i];
           (void)x1;(void)x0;}
         for(j2=0;j2<mk;j2++)if(AA[j2*mk+j2]>ridge)ridge=AA[j2*mk+j2];
         ridge*=1e-12;if(ridge<1e-300)ridge=1e-300;
         for(j2=0;j2<mk;j2++)AA[j2*mk+j2]+=ridge;
         {int info;char up='L';memcpy(sol,rhsa,sizeof(double)*mk);
          dpotrf_(&up,&mk,AA,&mk,&info);
          if(info)ok=0;else{int one=1;dpotrs_(&up,&mk,&one,AA,&mk,sol,&mk,&info);if(info)ok=0;}}
         if(ok){double st=0;for(j2=0;j2<mk;j2++)st+=fabs(sol[j2]);
           if(st<1e6){
             for(j2=0;j2<mk;j2++){
               int i1=(slot-j2+AAmem)%AAmem,i0=(slot-j2-1+AAmem)%AAmem;
               double b_=sol[j2];
               for(i=0;i<Nl;i++){
                 double dg=hg[i1*N+i]-hg[i0*N+i];
                 double dx=hx[i1*N+i]-hx[i0*N+i];
                 u[i]-=b_*(dx+dg);}}}
           else ok=0;}
         slot=(slot+1)%AAmem;if(hist<AAmem)hist++;else slot=slot;
         free(gcur);
         if(gn>prevg*1.5){hist=0;slot=0;}
        }
      } else if(AAmem){
        memcpy(hx+(slot%AAmem)*N,u,sizeof(double)*N);
        for(i=0;i<Nl;i++)hg[(slot%AAmem)*N+i]=zv[i]-ut[i];
        slot=(slot+1)%AAmem;hist=1;
      }
      prevg=gn;
      /* convergence / certificate evaluation */
      if((it%10==0&&it<500)||it%200==0||final_eval){
        double tau=ut[n+m];
        if(tau>1e-300){
          double sc=1.0/tau,rp,rd,gapv,p0,d0,cds,cdz,dmax2=0,xpx;
          double a1,a2,a3,dn;
          for(i=0;i<n;i++)xs[i]=ut[i]*sc;
          mv_A(&P,xs,axb);mv_P(&P,xs,pxp);
          for(i=0;i<m;i++){sout[i]=P.b[i]-axb[i];zout[i]=ut[n+i]*sc;}
          mv_AT(&P,zout,atz);
          for(i=0;i<n;i++)atz[i]/=P.exs[i];
          a1=nrmi(axb,m);a2=nrmi(sout,m);a3=P.normb;dn=1+(a1>a2?a1:(a2>a3?a2:a3));
          for(i=0;i<m;i++)t2[i]=axb[i]+sout[i]-P.b[i];
          rp=nrmi(t2,m)/dn;
          xpx=dot(xs,pxp,n);
          p0=0.5*xpx+dot(P.q,xs,n);
          for(i=0;i<n;i++)pxp[i]/=P.exs[i];
          a1=nrmi(pxp,n);a2=qnorig;a3=nrmi(atz,n);dn=1+(a1>a2?a1:(a2>a3?a2:a3));
          for(i=0;i<n;i++)t1[i]=pxp[i]+P.q[i]/P.exs[i]+atz[i];
          rd=nrmi(t1,n)/dn;
          d0=-0.5*xpx-dot(P.b,zout,m);
          gapv=fabs(p0-d0);
          cds=cone_defect(&P,sout,0,&ps);cdz=cone_defect(&P,zout,1,&ps);
          dmax2=rp;if(rd>dmax2)dmax2=rd;if(gapv>dmax2)dmax2=gapv;if(cds>dmax2)dmax2=cds;if(cdz>dmax2)dmax2=cdz;
          fin_rp=rp;fin_rd=rd;fin_gap=gapv;fin_cds=cds;fin_cdz=cdz;
          if(dmax2<best){best=dmax2;}
          if(getenv("SOLVER_DBG")&&(it%20==0))fprintf(stderr,"it=%ld rp=%.3g rd=%.3g gap=%.3g cds=%.3g cdz=%.3g tau=%.3g gn=%.3g\n",it,rp,rd,gapv,cds,cdz,tau,gn);
          if(rp<=tolr&&rd<=tolr&&cds<=tolr&&cdz<=tolr&&(gapv<=tolr||gapv<=tolr*(fabs(p0)>fabs(d0)?fabs(p0):fabs(d0))))status=0;
          if(status==0){for(i=0;i<n;i++)xout[i]=xs[i]*P.exs[i];}
        }
        if(status!=0&&(it%1000==0)){
          double nz=nrmi(ut+n,m);
          if(nz>1e-300&&P.normb>0){
            double zh=1.0/nz,bt,at2,cd;
            for(i=0;i<m;i++)zout[i]=ut[n+i]*zh;
            bt=-dot(P.b,zout,m)/P.normb;
            mv_AT(&P,zout,atz);for(i=0;i<n;i++)atz[i]/=P.exs[i];
            at2=nrmi(atz,n)/P.maxA;
            cd=cone_defect(&P,zout,1,&ps);
            if(bt>1e-6&&at2/bt<=tolr&&cd/bt<=tolr)status=1;
          }
        }
        if(status==0||status==1)break;
      }
      (void)restart;
      it++;if(final_eval)break;
      if(final_eval)break;
      if(pending){final_eval=1;continue;}
      if(now()-t0>hardcap){pending=1;continue;};
      if((gn<1e-15&&it>50)||best<tolr){pending=1;continue;}
    }
    fprintf(stderr,"iters=%ld best=%g status=%d t=%.2f rp=%g rd=%g gap=%g cds=%g cdz=%g\n",it,best,status,now()-t0,fin_rp,fin_rd,fin_gap,fin_cds,fin_cdz);
  }
  if(status==0)strcpy(jsonbuf,"{\"schema_version\":1,\"status\":\"solved\"}");
  else if(status==1)strcpy(jsonbuf,"{\"schema_version\":1,\"status\":\"primal_infeasible\"}");
  else strcpy(jsonbuf,"{\"schema_version\":1,\"status\":\"unresolved\"}");
  npz_write(outpath,NULL,0,jsonbuf,
            status==0?xout:NULL, status==0?n:-1,
            status==0?sout:NULL, status==0?m:-1,
            (status==0||status==1)?zout:NULL, (status==0||status==1)?m:-1);
  return 0;
}
