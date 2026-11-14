#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "dertab.h"
typedef int32_t i32; typedef int64_t i64;
static inline double eval_D(const double R[3], double invr2, int key, double rpow) {
  const int o = DER_OFF[key], o1 = DER_OFF[key + 1];
  double s = 0.0;
  double xp[8], yp[8], zp[8];
  xp[0]=yp[0]=zp[0]=1.0;
  for (int i=1;i<8;i++){xp[i]=xp[i-1]*R[0];yp[i]=yp[i-1]*R[1];zp[i]=zp[i-1]*R[2];}
  (void)invr2;
  for (int i=o;i<o1;i++){
    const DerMon *dm=&DER_MON[i];
    s += (double)dm->coeff * xp[dm->ea]*yp[dm->eb]*zp[dm->ec];
  }
  return s*rpow;
}
int main(){
  double R[3]={0.7,-1.3,0.4};
  double r2=R[0]*R[0]+R[1]*R[1]+R[2]*R[2], r=sqrt(r2);
  double x=R[0],y=R[1],z=R[2];
  /* rpow[n] = r^{-(2n+1)} */
  double rpow[12]; rpow[0]=1.0/r; for(int n=1;n<12;n++) rpow[n]=rpow[n-1]/r2;
  struct { int a,b,c; double expect; } chk[] = {
    {0,0,0, 1.0/r},
    {1,0,0, -x/pow(r,3)},
    {2,0,0, (2*x*x-y*y-z*z)/pow(r,5)},
    {1,1,0, 3*x*y/pow(r,5)},
    {3,0,0, (-6*x*x*x+9*x*(y*y+z*z))/pow(r,7)},
    {2,1,0, (-12*x*x*y+3*y*y*y+3*y*z*z)/pow(r,7)},
    {1,1,1, -15*x*y*z/pow(r,7)},
    {0,1,2, (3*x*x*y+3*y*y*y-12*y*z*z)/pow(r,7)},
  };
  for(unsigned i=0;i<sizeof(chk)/sizeof(chk[0]);i++){
    int a=chk[i].a,b=chk[i].b,c=chk[i].c;
    int n=a+b+c;
    double got=eval_D(R,0.0,der_key(a,b,c),rpow[n]);
    printf("(%d,%d,%d) key=%d got=%.12f expect=%.12f rel=%.2e\n",a,b,c,der_key(a,b,c),got,chk[i].expect,fabs(got-chk[i].expect)/fabs(chk[i].expect));
  }
  return 0;
}
