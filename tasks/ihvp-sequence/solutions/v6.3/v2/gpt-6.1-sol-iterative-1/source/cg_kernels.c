#include <stddef.h>

/* Independent accumulators enable SIMD without fast-math reassociation. */
double cg_step(int n, double alpha, double old_rr,
               double *x, double *r, double *p, const double *hp) {
    double s0 = 0., s1 = 0., s2 = 0., s3 = 0.;
    int i = 0;
    for (; i+3 < n; i += 4) {
        double r0 = r[i] - alpha*hp[i];
        double r1 = r[i+1] - alpha*hp[i+1];
        double r2 = r[i+2] - alpha*hp[i+2];
        double r3 = r[i+3] - alpha*hp[i+3];
        x[i] += alpha*p[i]; x[i+1] += alpha*p[i+1];
        x[i+2] += alpha*p[i+2]; x[i+3] += alpha*p[i+3];
        r[i] = r0; r[i+1] = r1; r[i+2] = r2; r[i+3] = r3;
        s0 += r0*r0; s1 += r1*r1; s2 += r2*r2; s3 += r3*r3;
    }
    for (; i < n; ++i) {
        x[i] += alpha*p[i];
        r[i] -= alpha*hp[i];
        s0 += r[i]*r[i];
    }
    const double rr = (s0+s1)+(s2+s3);
    const double beta = rr/old_rr;
    for (i = 0; i < n; ++i) p[i] = r[i]+beta*p[i];
    return rr;
}
