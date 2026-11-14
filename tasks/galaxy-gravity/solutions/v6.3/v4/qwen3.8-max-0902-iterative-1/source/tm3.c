#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "dertab.h"
int main(){
  double R[3]={-13.0, 9.0, -1.0};   /* pt - com from the failing case */
  double r2=R[0]*R[0]+R[1]*R[1]+R[2]*R[2], r=sqrt(r2);
  printf("r=%.6f r2=%.1f r^3=%.6g\n",r,r2,r2*r);
  int k=der_key(1,0,0);
  printf("der_key(1,0,0)=%d OFF=[%d,%d)\n",k,DER_OFF[k],DER_OFF[k+1]);
  double xp[10],yp[10],zp[10]; xp[0]=yp[0]=zp[0]=1.0;
  for(int i=1;i<10;i++){xp[i]=xp[i-1]*R[0];yp[i]=yp[i-1]*R[1];zp[i]=zp[i-1]*R[2];}
  double s=0;
  for(int i=DER_OFF[k];i<DER_OFF[k+1];i++){
    const DerMon*d=&DER_MON[i];
    printf("  mon {%d,%d,%d,%d} xp=%.6g yp=%.6g zp=%.6g\n",d->ea,d->eb,d->ec,d->coeff,
      xp[d->ea],yp[d->eb],zp[d->ec]);
    s+=(double)d->coeff*xp[d->ea]*yp[d->eb]*zp[d->ec];
  }
  printf("P=%.6g  D=P/r^3=%.6e  expected -x/r^3=%.6e\n",s,s/(r2*r),-R[0]/(r2*r));
  return 0;
}
