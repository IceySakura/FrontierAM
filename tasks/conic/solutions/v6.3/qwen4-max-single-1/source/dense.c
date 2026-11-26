#include "common.h"

/* symmetric eigendecomposition via cyclic Jacobi. A row-major n x n (destroyed).
   w[n] eigenvalues (ascending not guaranteed), V row-major: V[i*n+k]=eigvec k comp i */
void eig_sym(int n,double *A,double *w,double *V){
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)V[i*n+j]=(i==j)?1.0:0.0;
    if(n==1){w[0]=A[0];return;}
    for(int sweep=0;sweep<60;sweep++){
        double off=0;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++)off+=A[p*n+q]*A[p*n+q];
        if(off<1e-30)break;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            double apq=A[p*n+q];
            if(fabs(apq)<1e-300)continue;
            double app=A[p*n+p],aqq=A[q*n+q];
            double theta=(aqq-app)/(2*apq);
            double t;
            if(fabs(theta)>1e150)t=1.0/(2*theta);
            else t=(theta>=0?1.0:-1.0)/(fabs(theta)+sqrt(theta*theta+1));
            double c=1.0/sqrt(t*t+1),s=t*c;
            for(int k=0;k<n;k++){
                double akp=A[k*n+p],akq=A[k*n+q];
                A[k*n+p]=c*akp-s*akq;
                A[k*n+q]=s*akp+c*akq;
            }
            for(int k=0;k<n;k++){
                double apk=A[p*n+k],aqk=A[q*n+k];
                A[p*n+k]=c*apk-s*aqk;
                A[q*n+k]=s*apk+c*aqk;
            }
            for(int k=0;k<n;k++){
                double vkp=V[k*n+p],vkq=V[k*n+q];
                V[k*n+p]=c*vkp-s*vkq;
                V[k*n+q]=s*vkp+c*vkq;
            }
        }
    }
    for(int i=0;i<n;i++)w[i]=A[i*n+i];
}

/* C = A*B, all n x n row-major */
void mmul(int n,const double *A,const double *B,double *C){
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){
        double s=0;for(int k=0;k<n;k++)s+=A[i*n+k]*B[k*n+j];
        C[i*n+j]=s;
    }
}
/* C = A^T B */
void mmulT(int n,const double *A,const double *B,double *C){
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){
        double s=0;for(int k=0;k<n;k++)s+=A[k*n+i]*B[k*n+j];
        C[i*n+j]=s;
    }
}
