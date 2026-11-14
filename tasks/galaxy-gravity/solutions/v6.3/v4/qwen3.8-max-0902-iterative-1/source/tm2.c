#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "dertab.h"
typedef int32_t i32; typedef int64_t i64;
int main(){
  double com[3]={10.0,-4.0,2.0}, pt[3]={-3.0,5.0,1.0};
  double R[3]={pt[0]-com[0],pt[1]-com[1],pt[2]-com[2]};
  double r2=R[0]*R[0]+R[1]*R[1]+R[2]*R[2], r=sqrt(r2);
  double M=0.05;
  printf("R=(%g,%g,%g) r=%.10g r2=%.10g M=%g\n",R[0],R[1],R[2],r,r2,M);
  /* monopole: expected a = M*(pt-com)/r^3 = M*R/r^3 */
  printf("expected a = (%.8e, %.8e, %.8e)\n", M*R[0]/(r2*r), M*R[1]/(r2*r), M*R[2]/(r2*r));
  /* table path */
  printf("der_key(0,0,0)=%d der_key(1,0,0)=%d\n",der_key(0,0,0),der_key(1,0,0));
  printf("DER_OFF[0]=%d DER_OFF[1]=%d DER_OFF[3]=%d DER_OFF[4]=%d\n",
     DER_OFF[0],DER_OFF[1],DER_OFF[3],DER_OFF[4]);
  printf("DER_MON[0]={%d,%d,%d,%d}\n",DER_MON[0].ea,DER_MON[0].eb,DER_MON[0].ec,DER_MON[0].coeff);
  for(int i=DER_OFF[der_key(1,0,0)];i<DER_OFF[der_key(1,0,0)+1];i++)
    printf("  D_(1,0,0) monomial: {%d,%d,%d,%d}\n",DER_MON[i].ea,DER_MON[i].eb,DER_MON[i].ec,DER_MON[i].coeff);
  /* eval_D for key (1,0,0) with rpow=r^3 */
  double rpow=r2*r;
  int key=der_key(1,0,0); double s=0;
  for(int i=DER_OFF[key];i<DER_OFF[key+1];i++){
    const DerMon*d=&DER_MON[i];
    double xp=1,yp=1,zp=1;
    for(int t=0;t<d->ea;t++) xp*=R[0];
    for(int t=0;t<d->eb;t++) yp*=R[1];
    for(int t=0;t<d->ec;t++) zp*=R[2];
    s+=(double)d->coeff*xp*yp*zp;
    printf("  term {%d,%d,%d,%d} -> P contrib %.10g\n",d->ea,d->eb,d->ec,d->coeff,(double)d->coeff*xp*yp*zp);
  }
  printf("P_(1,0,0)(R)=%.10g  D=%.10g\n",s,s/rpow);
  double cf=M*1.0*1.0;  /* invfact=1, sign=+1 for |alpha|=0 */
  printf("f_x = cf * D_(1,0,0) = %.10g\n", cf*s/rpow);
  return 0;
}
