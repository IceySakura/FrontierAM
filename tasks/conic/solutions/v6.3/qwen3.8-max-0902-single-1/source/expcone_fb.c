/* Fallback exponential-cone projections (used if expcone.c is absent). */
#include <math.h>
#include <float.h>
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
/* minimize 2-D smooth objective by damped Newton with backtracking */
typedef double (*F2)(double,double,void*,double*,double*,double*,double*,double*);
static void newton2(F2 f, void *ctx, double *x, double *y, int maxit){
  int k;
  for(k=0;k<maxit;k++){
    double g0,g1,h00,h01,h11,dx,dy,det,fx,fy,t=1.0,v0;
    (*f)(*x,*y,ctx,&g0,&g1,&h00,&h01,&h11);
    det=h00*h11-h01*h01;
    if(!(fabs(det)>1e-300)){dx=-g0*1e-3;dy=-g1*1e-3;}
    else{dx=(-h11*g0+h01*g1)/det;dy=(h01*g0-h00*g1)/det;}
    if(!isfinite(dx)||!isfinite(dy))return;
    {double nn=fabs(dx)+fabs(dy);double lim=1e3*(fabs(*x)+fabs(*y)+1.0);if(nn>lim){dx*=lim/nn;dy*=lim/nn;}}
    v0=(*f)(*x,*y,ctx,0,0,0,0,0);
    for(;;){double nx=*x+t*dx,ny=*y+t*dy;
      if(nx>1e-14&&ny>1e-14){double v=(*f)(nx,ny,ctx,0,0,0,0,0);if(v<=v0){*x=nx;*y=ny;break;}}
      t*=0.5;if(t<1e-12)break;}
    if(t<1e-12)break;
  }
}
typedef struct{double a0,a1,a2;}Ctx;
/* primal: boundary u = v ln(v/w); objective in (v,w) */
static double fp(double v,double w,void*c_,double*g0,double*g1,double*h00,double*h01,double*h11){
  Ctx*c=(Ctx*)c_;double r=log(v/w),u=v*r,d0=u-c->a0,d1=v-c->a1,d2=w-c->a2;
  double F=d0*d0+d1*d1+d2*d2;
  if(g0){*g0=2*d0*(r+1)+2*d1;*g1=-2*d0*(v/w)+2*d2;
    *h00=2*(r+1)*(r+1)+2*d0/v+2;*h01=-2*(r+1)*(v/w)-2*d0/w;*h11=2*(v/w)*(v/w)+2*d0*v/(w*w);}
  return F;}
/* dual: boundary y = x(1-ln(-x/z)), x<0,z>0; variables (p=-x>0, z) */
static double fd(double p,double z,void*c_,double*g0,double*g1,double*h00,double*h01,double*h11){
  Ctx*c=(Ctx*)c_;double r=log(p/z),y=-p*(1.0-r),d0=-p-c->a0,d1=y-c->a1,d2=z-c->a2;
  double F=d0*d0+d1*d1+d2*d2;
  /* dy/dp = -(1-r) - p*(1/p)... r=log p - log z, dr/dp=1/p; y=-p+p*r -> dy/dp=-1+r+1=r ; dy/dz=-p*(-1/z)=p/z */
  if(g0){*g0=2*d0*(-1)+2*d1*r;*g1=2*d1*(p/z)+2*d2;
    *h00=2+2*r*r+2*d1/p;   /* d/dp(2d1 r)=2 r*r + 2 d1 /p */
    *h01=2*r*(p/z)+2*d1*(1/z);
    *h11=2*(p/z)*(p/z)-2*d1*p/(z*z);}
  return F;}
void proj_exp(const double *v,double *out){
  Ctx c;double bv[3],F,bF;double V,W;
  double a0=v[0],a1=v[1],a2=v[2];
  if(a1>0&&a2>0){double t=a1*exp(a0/a1);if(isfinite(t)?(t<=a2):1){out[0]=a0;out[1]=a1;out[2]=a2;return;}}
  if(a1<=0&&a0<=0&&a2>=0){out[0]=a0;out[1]=0;out[2]=a2;return;}
  /* face candidate */
  bv[0]=a0<0?a0:0;bv[1]=0;bv[2]=a2>0?a2:0;
  bF=(bv[0]-a0)*(bv[0]-a0)+(bv[1]-a1)*(bv[1]-a1)+(bv[2]-a2)*(bv[2]-a2);
  c.a0=a0;c.a1=a1;c.a2=a2;
  V=a1>1e-9?a1:1e-3*(fabs(a0)+fabs(a1)+fabs(a2)+1);
  W=a2>1e-9?a2:1e-3*(fabs(a0)+fabs(a1)+fabs(a2)+1);
  newton2(fp,&c,&V,&W,80);
  {double u=V*log(V/W);F=(u-a0)*(u-a0)+(V-a1)*(V-a1)+(W-a2)*(W-a2);
   if(F<bF&&isfinite(F)){out[0]=u;out[1]=V;out[2]=W;}else{out[0]=bv[0];out[1]=bv[1];out[2]=bv[2];}}
}
void proj_exp_dual(const double *v,double *out){
  Ctx c;double bv[3],bF,Pp,Z,F;
  double a0=v[0],a1=v[1],a2=v[2];
  /* inside dual cone? check defect E(-a1,-a0,e*a2)==0 */
  {double a=-a1,b=-a0,cc=M_E*a2;double B=fabs(b);double t;
   if(a>B)B=a;if(-cc>B)B=-cc;
   if(b>0&&cc>0){t=a-b*(log(cc)-log(b));if(t<0)t=0;if(t<B)B=t;}
   if(B<=0){out[0]=a0;out[1]=a1;out[2]=a2;return;}}
  bv[0]=0;bv[1]=a1>0?a1:0;bv[2]=a2>0?a2:0;
  bF=(bv[0]-a0)*(bv[0]-a0)+(bv[1]-a1)*(bv[1]-a1)+(bv[2]-a2)*(bv[2]-a2);
  c.a0=a0;c.a1=a1;c.a2=a2;
  Pp=(-a0>1e-9)?-a0:1e-3*(fabs(a0)+fabs(a1)+fabs(a2)+1);
  Z=(a2>1e-9)?a2:1e-3*(fabs(a0)+fabs(a1)+fabs(a2)+1);
  newton2(fd,&c,&Pp,&Z,80);
  {double x=-Pp,y=-Pp*(1.0-log(Pp/Z));
   F=(x-a0)*(x-a0)+(y-a1)*(y-a1)+(Z-a2)*(Z-a2);
   if(F<bF&&isfinite(F)){out[0]=x;out[1]=y;out[2]=Z;}else{out[0]=bv[0];out[1]=bv[1];out[2]=bv[2];}}
}
