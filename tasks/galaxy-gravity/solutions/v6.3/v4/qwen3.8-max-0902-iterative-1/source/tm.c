#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "dertab.h"
typedef int32_t i32; typedef int64_t i64;
#define MAXP 6
#define NTERM(p) (((p)+1)*((p)+2)*((p)+3)/6)
static i32 g_alpha[MAXP+1][NTERM(MAXP)][3];
static double g_invfact[MAXP+1][NTERM(MAXP)];
static double g_sign[MAXP+1][NTERM(MAXP)];
static int g_nalpha[MAXP+1];
static void plan_init(void){
  for(int p=0;p<=MAXP;p++){int k=0;
    for(int a=0;a<=p;a++)for(int b=0;a+b<=p;b++)for(int c=0;a+b+c<=p;c++){
      g_alpha[p][k][0]=a;g_alpha[p][k][1]=b;g_alpha[p][k][2]=c;
      g_invfact[p][k]=1.0/((double)tgamma(a+1)*tgamma(b+1)*tgamma(c+1));
      g_sign[p][k]=1.0; k++;}
    g_nalpha[p]=k;}
}
static inline double eval_D(const double R[3], int key, double rpow, int extra){
  const int o=DER_OFF[key],o1=DER_OFF[key+1];
  double s=0.0,xp[DER_MAXP+2],yp[DER_MAXP+2],zp[DER_MAXP+2];
  xp[0]=yp[0]=zp[0]=1.0;
  for(int i=1;i<=DER_MAXP+1;i++){xp[i]=xp[i-1]*R[0];yp[i]=yp[i-1]*R[1];zp[i]=zp[i-1]*R[2];}
  for(int i=o;i<o1;i++){const DerMon*d=&DER_MON[i];s+=(double)d->coeff*xp[d->ea]*yp[d->eb]*zp[d->ec];}
  (void)extra;
  return s/rpow;
}
int main(){
  plan_init();
  /* one source cluster far from the target */
  double com[3]={10.0,-4.0,2.0};
  double pt[3]={-3.0,5.0,1.0};
  int ns=12; double src[12][3], ms[12];
  srand(7);
  for(int i=0;i<ns;i++){
    for(int d=0;d<3;d++) src[i][d]=com[d]+((double)rand()/RAND_MAX-0.5)*0.6;
    ms[i]=0.05/ns;
  }
  /* exact */
  double ex[3]={0,0,0};
  for(int i=0;i<ns;i++){
    double dx=src[i][0]-pt[0],dy=src[i][1]-pt[1],dz=src[i][2]-pt[2];
    double r2=dx*dx+dy*dy+dz*dz, rr=sqrt(r2);
    double w=ms[i]/(r2*rr);
    ex[0]+=dx*w;ex[1]+=dy*w;ex[2]+=dz*w;
  }
  for(int p=0;p<=4;p++){
    /* moments */
    double Mo[NTERM(MAXP)]; memset(Mo,0,sizeof(Mo));
    for(int i=0;i<ns;i++){
      double dx=src[i][0]-com[0],dy=src[i][1]-com[1],dz=src[i][2]-com[2];
      double pw[MAXP+1],qw[MAXP+1],rw[MAXP+1];
      pw[0]=ms[i];qw[0]=ms[i];rw[0]=ms[i];
      for(int a=1;a<=p;a++)pw[a]=pw[a-1]*dx;
      for(int b=1;b<=p;b++)qw[b]=qw[b-1]*dy;
      for(int c=1;c<=p;c++)rw[c]=rw[c-1]*dz;
      for(int a=0;a<=p;a++)for(int b=0;a+b<=p;b++){double ab=pw[a]*qw[b];
        for(int c=0;a+b+c<=p;c++) Mo[der_key(a,b,c)]+=ab*rw[c];}
    }
    double R[3]={pt[0]-com[0],pt[1]-com[1],pt[2]-com[2]};
    double r2=R[0]*R[0]+R[1]*R[1]+R[2]*R[2],r=sqrt(r2),ri=1.0/r;
    double rpow[MAXP+3]; rpow[0]=r; for(int n=1;n<=MAXP+2;n++) rpow[n]=rpow[n-1]*r2;
    double f[3]={0,0,0};
    int n=g_nalpha[p];
    for(int k=0;k<n;k++){
      double mk=Mo[k]; if(mk==0.0) continue;
      int a=g_alpha[p][k][0],b=g_alpha[p][k][1],c=g_alpha[p][k][2];
      double cf=mk*g_invfact[p][k]*g_sign[p][k];
      int ord=a+b+c;
      if(ord+1<=DER_MAXP){
        double pw_=rpow[ord+1];
      if(p==0){ fprintf(stderr,"  dbg: k=%d Mo=%.6g cf=%.6g ord=%d rpow=%.6g Dx=%.6e\n",
        k, mk, cf, ord, pw_, eval_D(R,der_key(a+1,b,c),pw_,0)); }
        f[0]+=cf*eval_D(R,der_key(a+1,b,c),pw_,0);
        f[1]+=cf*eval_D(R,der_key(a,b+1,c),pw_,0);
        f[2]+=cf*eval_D(R,der_key(a,b,c+1),pw_,0);
      }
    }
    double e=sqrt((f[0]-ex[0])*(f[0]-ex[0])+(f[1]-ex[1])*(f[1]-ex[1])+(f[2]-ex[2])*(f[2]-ex[2]));
    double en=sqrt(ex[0]*ex[0]+ex[1]*ex[1]+ex[2]*ex[2]);
    printf("p=%d sgn=+ rel=%.4e | sgn=- rel=%.4e\n", p, e/en,
      sqrt((f[0]+ex[0])*(f[0]+ex[0])+(f[1]+ex[1])*(f[1]+ex[1])+(f[2]+ex[2])*(f[2]+ex[2]))/en);
  }
  printf("g_alpha[2] order check: ");
  for(int k=0;k<g_nalpha[2];k++) printf("(%d%d%d)",g_alpha[2][k][0],g_alpha[2][k][1],g_alpha[2][k][2]);
  printf("\nder_key(0,0,0)=%d der_key(1,0,0)=%d der_key(0,1,0)=%d der_key(0,0,1)=%d\n",
     der_key(0,0,0),der_key(1,0,0),der_key(0,1,0),der_key(0,0,1));
  return 0;
}
