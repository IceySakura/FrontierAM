/* Two-steps-per-sweep Chebyshev kernels, instantiated for two matrix encodings.
 * Included from expkern.c with:
 *   SUFFIX           name suffix
 *   MATARGS          kernel parameters describing the matrix
 *   DOT1(X, i)       (L X)_i for a vector X
 *   DOT2(X, i, s0, s1)  (L X)_i for two interleaved vectors
 * Contract: see cheb1 / cheb2.  Each sweep computes row i of T_{j+1} and row i-lag of
 * T_{j+2}; requires lag >= bandwidth (any lag >= n degenerates to two plain sweeps). */
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

i64 CAT(cheb1s, SUFFIX)(i64 n, MATARGS,
                        double sa, double sb, const double *coef, const double *tail, i64 K,
                        const double *b, double *y, double *A, double *B,
                        double atol, double rtol, double safety, double growlim, double *info, i64 lag,
                        const double *pfirst)
{
    /* pfirst (optional): L b computed beforehand, used for the first step */
    double bn2 = 0.0, binf = 0.0, ymax = 0.0;
    double c0 = coef[0];
    for (i64 i = 0; i < n; i++) {
        double v = b[i];
        A[i] = v;
        bn2 += v * v;
        if (fabs(v) > binf) binf = fabs(v);
        double yy = c0 * v;
        y[i] = yy;
        if (fabs(yy) > ymax) ymax = fabs(yy);
    }
    info[0] = 0.0; info[1] = binf; info[2] = ymax;
    if (bn2 == 0.0) return 0;
    double M = binf;
    double glim2 = growlim * growlim * bn2;
    if (K < 1 || safety * tail[1] * M <= atol + rtol * ymax) {
        info[0] = tail[1] * M; info[2] = ymax;
        return 0;
    }
    {
        double c1 = coef[1], wn2 = 0.0, wmax = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = pfirst ? pfirst[i] : DOT1(A, i);
            double w = sa * s + sb * A[i];
            B[i] = w;
            wn2 += w * w;
            if (fabs(w) > wmax) wmax = fabs(w);
            double yy = y[i] + c1 * w;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (!(wn2 <= glim2)) return -2;
        if (wmax > M) M = wmax;
        if (K < 2 || safety * tail[2] * M <= atol + rtol * ymax) {
            info[0] = tail[2] * M; info[1] = M; info[2] = ymax;
            return 1;
        }
    }
    double sa2 = 2.0 * sa, sb2 = 2.0 * sb;
    i64 j = 1;
    if (lag < 1) lag = 1;
    i64 pre = lag < n ? lag : n;
    while (j + 2 <= K) {
        double c1 = coef[j + 1], c2 = coef[j + 2];
        double n1 = 0.0, n2 = 0.0, x1 = 0.0, x2 = 0.0;
        ymax = 0.0;
        i64 i = 0;
        for (; i < pre; i++) {
            double s = DOT1(B, i);
            double w = sa2 * s + sb2 * B[i] - A[i];
            A[i] = w;
            n1 += w * w;
            if (fabs(w) > x1) x1 = fabs(w);
            y[i] += c1 * w;
        }
        for (; i < n; i++) {
            double s = DOT1(B, i);
            double w = sa2 * s + sb2 * B[i] - A[i];
            A[i] = w;
            n1 += w * w;
            if (fabs(w) > x1) x1 = fabs(w);
            y[i] += c1 * w;
            i64 r = i - pre;
            double s2 = DOT1(A, r);
            double w2 = sa2 * s2 + sb2 * A[r] - B[r];
            B[r] = w2;
            n2 += w2 * w2;
            if (fabs(w2) > x2) x2 = fabs(w2);
            double yy = y[r] + c2 * w2;
            y[r] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        for (i64 r = n - pre; r < n; r++) {
            double s2 = DOT1(A, r);
            double w2 = sa2 * s2 + sb2 * A[r] - B[r];
            B[r] = w2;
            n2 += w2 * w2;
            if (fabs(w2) > x2) x2 = fabs(w2);
            double yy = y[r] + c2 * w2;
            y[r] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (!(n1 <= glim2) || !(n2 <= glim2)) return -(j + 2);
        if (x1 > M) M = x1;
        if (x2 > M) M = x2;
        j += 2;
        if (safety * tail[j + 1] * M <= atol + rtol * ymax) {
            info[0] = tail[j + 1] * M; info[1] = M; info[2] = ymax;
            return j;
        }
    }
    if (j < K) {
        double cj = coef[j + 1], wn2 = 0.0, wmax = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = DOT1(B, i);
            double w = sa2 * s + sb2 * B[i] - A[i];
            A[i] = w;
            wn2 += w * w;
            if (fabs(w) > wmax) wmax = fabs(w);
            double yy = y[i] + cj * w;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (!(wn2 <= glim2)) return -(j + 2);
        if (wmax > M) M = wmax;
        j++;
    }
    info[0] = tail[j + 1] * M; info[1] = M; info[2] = ymax;
    return j;
}

i64 CAT(cheb2s, SUFFIX)(i64 n, MATARGS,
                        double sa, double sb, const double *c, const double *tc,
                        const double *d, const double *td, i64 K,
                        const double *b0, const double *b1, double *y, double *A, double *B,
                        double atol, double rtol, double safety, double growlim, double *info, i64 lag,
                        const double *pre0, const double *pre1)
{
    double q0 = 0.0, q1 = 0.0, m0 = 0.0, m1 = 0.0, ymax = 0.0;
    for (i64 i = 0; i < n; i++) {
        double u = b0[i], v = b1[i];
        A[2 * i] = u; A[2 * i + 1] = v;
        q0 += u * u; q1 += v * v;
        if (fabs(u) > m0) m0 = fabs(u);
        if (fabs(v) > m1) m1 = fabs(v);
        double yy = c[0] * u + d[0] * v;
        y[i] = yy;
        if (fabs(yy) > ymax) ymax = fabs(yy);
    }
    info[0] = 0.0; info[1] = m0; info[2] = ymax;
    double M0 = m0, M1 = m1;
    double g0 = growlim * growlim * q0, g1 = growlim * growlim * q1;
    if (K < 1 || safety * (tc[1] * M0 + td[1] * M1) <= atol + rtol * ymax) {
        info[0] = tc[1] * M0 + td[1] * M1; info[2] = ymax;
        return 0;
    }
    {
        double e0 = 0.0, e1 = 0.0, x0 = 0.0, x1 = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s0, s1;
            if (pre0) { s0 = pre0[i]; s1 = pre1[i]; }
            else DOT2(A, i, s0, s1);
            double w0 = sa * s0 + sb * A[2 * i], w1 = sa * s1 + sb * A[2 * i + 1];
            B[2 * i] = w0; B[2 * i + 1] = w1;
            e0 += w0 * w0; e1 += w1 * w1;
            if (fabs(w0) > x0) x0 = fabs(w0);
            if (fabs(w1) > x1) x1 = fabs(w1);
            double yy = y[i] + c[1] * w0 + d[1] * w1;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (!(e0 <= g0) || !(e1 <= g1)) return -2;
        if (x0 > M0) M0 = x0;
        if (x1 > M1) M1 = x1;
        if (K < 2 || safety * (tc[2] * M0 + td[2] * M1) <= atol + rtol * ymax) {
            info[0] = tc[2] * M0 + td[2] * M1; info[1] = M0; info[2] = ymax;
            return 1;
        }
    }
    double sa2 = 2.0 * sa, sb2 = 2.0 * sb;
    i64 j = 1;
    if (lag < 1) lag = 1;
    i64 pre = lag < n ? lag : n;
    while (j + 2 <= K) {
        double ca = c[j + 1], da = d[j + 1], cb = c[j + 2], db = d[j + 2];
        double e0 = 0.0, e1 = 0.0, f0 = 0.0, f1 = 0.0, x0 = 0.0, x1 = 0.0, z0 = 0.0, z1 = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n + pre; i++) {
            if (i < n) {
                double s0, s1;
                DOT2(B, i, s0, s1);
                double w0 = sa2 * s0 + sb2 * B[2 * i] - A[2 * i];
                double w1 = sa2 * s1 + sb2 * B[2 * i + 1] - A[2 * i + 1];
                A[2 * i] = w0; A[2 * i + 1] = w1;
                e0 += w0 * w0; e1 += w1 * w1;
                if (fabs(w0) > x0) x0 = fabs(w0);
                if (fabs(w1) > x1) x1 = fabs(w1);
                y[i] += ca * w0 + da * w1;
            }
            i64 r = i - pre;
            if (r >= 0) {
                double s0, s1;
                DOT2(A, r, s0, s1);
                double w0 = sa2 * s0 + sb2 * A[2 * r] - B[2 * r];
                double w1 = sa2 * s1 + sb2 * A[2 * r + 1] - B[2 * r + 1];
                B[2 * r] = w0; B[2 * r + 1] = w1;
                f0 += w0 * w0; f1 += w1 * w1;
                if (fabs(w0) > z0) z0 = fabs(w0);
                if (fabs(w1) > z1) z1 = fabs(w1);
                double yy = y[r] + cb * w0 + db * w1;
                y[r] = yy;
                if (fabs(yy) > ymax) ymax = fabs(yy);
            }
        }
        if (!(e0 <= g0) || !(e1 <= g1) || !(f0 <= g0) || !(f1 <= g1)) return -(j + 2);
        if (x0 > M0) M0 = x0;
        if (z0 > M0) M0 = z0;
        if (x1 > M1) M1 = x1;
        if (z1 > M1) M1 = z1;
        j += 2;
        if (safety * (tc[j + 1] * M0 + td[j + 1] * M1) <= atol + rtol * ymax) {
            info[0] = tc[j + 1] * M0 + td[j + 1] * M1; info[1] = M0; info[2] = ymax;
            return j;
        }
    }
    if (j < K) {
        double e0 = 0.0, e1 = 0.0, x0 = 0.0, x1 = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s0, s1;
            DOT2(B, i, s0, s1);
            double w0 = sa2 * s0 + sb2 * B[2 * i] - A[2 * i];
            double w1 = sa2 * s1 + sb2 * B[2 * i + 1] - A[2 * i + 1];
            A[2 * i] = w0; A[2 * i + 1] = w1;
            e0 += w0 * w0; e1 += w1 * w1;
            if (fabs(w0) > x0) x0 = fabs(w0);
            if (fabs(w1) > x1) x1 = fabs(w1);
            double yy = y[i] + c[j + 1] * w0 + d[j + 1] * w1;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (!(e0 <= g0) || !(e1 <= g1)) return -(j + 2);
        if (x0 > M0) M0 = x0;
        if (x1 > M1) M1 = x1;
        j++;
    }
    info[0] = tc[j + 1] * M0 + td[j + 1] * M1; info[1] = M0; info[2] = ymax;
    return j;
}

#undef CAT
#undef CAT_
