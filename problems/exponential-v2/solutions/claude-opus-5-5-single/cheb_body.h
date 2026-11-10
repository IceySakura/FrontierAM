/* Chebyshev recurrence kernels, instantiated for REAL = double and long double.
 * Included twice from expo.c with REAL and SUF() defined. */

/* fused step: vn = p*(L x)_i + q*x_i + r*xm_i ; xm <- vn ; y += c*vn */
static void SUF(step1)(const REAL *restrict x, REAL *restrict xm, REAL *restrict y,
                       REAL p, REAL q, REAL r, REAL c,
                       double *nrm2, double *vmax, double *ymax) {
    REAL s2 = 0, vm = 0, ym = 0;
    const uint32_t *restrict RP = rp; const uint32_t *restrict CI = ci; const double *restrict VA = va;
    for (int i = 0; i < n; i++) {
        REAL s = 0;
        uint32_t e = RP[i + 1];
        for (uint32_t k = RP[i]; k < e; k++) s += VA[k] * x[CI[k]];
        REAL vn = p * s + q * x[i] + r * xm[i];
        xm[i] = vn;
        REAL yi = y[i] + c * vn;
        y[i] = yi;
        s2 += vn * vn;
        REAL av = RABS(vn); vm = av > vm ? av : vm;
        REAL ay = RABS(yi); ym = ay > ym ? ay : ym;
    }
    *nrm2 = (double)s2; *vmax = (double)vm; *ymax = (double)ym;
}

/* two interleaved vectors x[2i], x[2i+1]; y += c0*vn0 + c1*vn1 */
static void SUF(step2)(const REAL *restrict x, REAL *restrict xm, REAL *restrict y,
                       REAL p, REAL q, REAL r, REAL c0, REAL c1,
                       double *nrm2a, double *nrm2b, double *vmaxa, double *vmaxb, double *ymax) {
    REAL sa = 0, sb = 0, vma = 0, vmb = 0, ym = 0;
    const uint32_t *restrict RP = rp; const uint32_t *restrict CI = ci; const double *restrict VA = va;
    for (int i = 0; i < n; i++) {
        REAL s0 = 0, s1 = 0;
        uint32_t e = RP[i + 1];
        for (uint32_t k = RP[i]; k < e; k++) {
            double a = VA[k]; size_t j = 2 * (size_t)CI[k];
            s0 += a * x[j]; s1 += a * x[j + 1];
        }
        size_t i2 = 2 * (size_t)i;
        REAL v0 = p * s0 + q * x[i2] + r * xm[i2];
        REAL v1 = p * s1 + q * x[i2 + 1] + r * xm[i2 + 1];
        xm[i2] = v0; xm[i2 + 1] = v1;
        REAL yi = y[i] + c0 * v0 + c1 * v1;
        y[i] = yi;
        sa += v0 * v0; sb += v1 * v1;
        REAL a0 = RABS(v0); vma = a0 > vma ? a0 : vma;
        REAL a1 = RABS(v1); vmb = a1 > vmb ? a1 : vmb;
        REAL ay = RABS(yi); ym = ay > ym ? ay : ym;
    }
    *nrm2a = (double)sa; *nrm2b = (double)sb; *vmaxa = (double)vma; *vmaxb = (double)vmb; *ymax = (double)ym;
}


/* SELL-4 variants: chunks of 4 consecutive (permuted) rows, column-major
   within a chunk, padded with zero entries; 4 independent accumulators. */
static void SUF(step1s)(const REAL *restrict x, REAL *restrict xm, REAL *restrict y,
                        REAL p, REAL q, REAL r, REAL c,
                        double *nrm2, double *vmax, double *ymax) {
    REAL s2 = 0, vm = 0, ym = 0;
    const uint32_t *restrict CS = scs; const uint32_t *restrict CI = sci; const double *restrict VA = sva;
    for (int ch = 0; ch < nch; ch++) {
        uint32_t k1 = CS[ch + 1];
        REAL a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (uint32_t k = CS[ch]; k < k1; k += 4) {
            a0 += VA[k] * x[CI[k]];
            a1 += VA[k + 1] * x[CI[k + 1]];
            a2 += VA[k + 2] * x[CI[k + 2]];
            a3 += VA[k + 3] * x[CI[k + 3]];
        }
        REAL acc[4] = {a0, a1, a2, a3};
        size_t i0 = 4 * (size_t)ch;
        for (int rr = 0; rr < 4; rr++) {
            size_t i = i0 + rr;
            REAL vn = p * acc[rr] + q * x[i] + r * xm[i];
            xm[i] = vn;
            REAL yi = y[i] + c * vn;
            y[i] = yi;
            s2 += vn * vn;
            REAL av = RABS(vn); vm = av > vm ? av : vm;
            REAL ay = RABS(yi); ym = ay > ym ? ay : ym;
        }
    }
    *nrm2 = (double)s2; *vmax = (double)vm; *ymax = (double)ym;
}

static void SUF(step2s)(const REAL *restrict x, REAL *restrict xm, REAL *restrict y,
                        REAL p, REAL q, REAL r, REAL c0, REAL c1,
                        double *nrm2a, double *nrm2b, double *vmaxa, double *vmaxb, double *ymax) {
    REAL sa = 0, sb = 0, vma = 0, vmb = 0, ym = 0;
    const uint32_t *restrict CS = scs; const uint32_t *restrict CI = sci; const double *restrict VA = sva;
    for (int ch = 0; ch < nch; ch++) {
        uint32_t k1 = CS[ch + 1];
        REAL a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (uint32_t k = CS[ch]; k < k1; k += 4) {
            size_t j0 = 2 * (size_t)CI[k], j1 = 2 * (size_t)CI[k + 1], j2 = 2 * (size_t)CI[k + 2], j3 = 2 * (size_t)CI[k + 3];
            double v0 = VA[k], v1 = VA[k + 1], v2 = VA[k + 2], v3 = VA[k + 3];
            a0 += v0 * x[j0]; b0 += v0 * x[j0 + 1];
            a1 += v1 * x[j1]; b1 += v1 * x[j1 + 1];
            a2 += v2 * x[j2]; b2 += v2 * x[j2 + 1];
            a3 += v3 * x[j3]; b3 += v3 * x[j3 + 1];
        }
        REAL aa[4] = {a0, a1, a2, a3}, bb[4] = {b0, b1, b2, b3};
        size_t i0 = 4 * (size_t)ch;
        for (int rr = 0; rr < 4; rr++) {
            size_t i2 = 2 * (i0 + rr);
            REAL w0 = p * aa[rr] + q * x[i2] + r * xm[i2];
            REAL w1 = p * bb[rr] + q * x[i2 + 1] + r * xm[i2 + 1];
            xm[i2] = w0; xm[i2 + 1] = w1;
            REAL yi = y[i0 + rr] + c0 * w0 + c1 * w1;
            y[i0 + rr] = yi;
            sa += w0 * w0; sb += w1 * w1;
            REAL e0 = RABS(w0); vma = e0 > vma ? e0 : vma;
            REAL e1 = RABS(w1); vmb = e1 > vmb ? e1 : vmb;
            REAL ay = RABS(yi); ym = ay > ym ? ay : ym;
        }
    }
    *nrm2a = (double)sa; *nrm2b = (double)sb; *vmaxa = (double)vma; *vmaxb = (double)vmb; *ymax = (double)ym;
}

/* Returns 0 ok, 1 if the 2-norm growth monitor fired (interval too small).
 * w1, w2: 2n REALs each; yacc: n REALs (may alias yout when REAL is double).
 * cheb_solve_x: the right-hand sides are multiplied by `scale` on input (and
 * the result divided by it); fixed_deg > 0 forces exactly that degree. */
static int SUF(cheb_solve_x)(const Coef *C, const double *b0, const double *b1, double scale, int fixed_deg,
                             double *yout, REAL *w1, REAL *w2, REAL *yacc, int monitor, int *deg_out, double *ymax_out);
static int SUF(cheb_solve)(const Coef *C, const double *b0, const double *b1, double *yout,
                           REAL *w1, REAL *w2, REAL *yacc, int monitor, int *deg_out, double *ymax_out) {
    return SUF(cheb_solve_x)(C, b0, b1, 1.0, 0, yout, w1, w2, yacc, monitor, deg_out, ymax_out);
}
static int SUF(cheb_solve_x)(const Coef *C, const double *b0, const double *b1, double scale, int fixed_deg,
                             double *yout, REAL *w1, REAL *w2, REAL *yacc, int monitor, int *deg_out, double *ymax_out) {
    REAL alpha2 = (REAL)2.0 / (REAL)C->lh, alpha4 = (REAL)4.0 / (REAL)C->lh;
    double n0 = 0, n1 = 0, m0 = 0, m1 = 0;
    for (int i = 0; i < nk; i++) {
        double a = b0[i] * scale, b = b1[i] * scale;
        n0 += a * a; n1 += b * b;
        double fa = fabs(a), fb = fabs(b);
        m0 = fa > m0 ? fa : m0; m1 = fb > m1 ? fb : m1;
    }
    int has0 = m0 > 0, has1 = m1 > 0;
    const double SF = 3.0;
    *ymax_out = 0;
    if (!has0 && !has1) { memset(yout, 0, sizeof(double) * nk); *deg_out = 0; return 0; }
    int M = C->M;
    int k;
    double ym = 0;
    if (has0 && has1) {
        REAL *X = w1, *Xm = w2;
        for (int i = 0; i < nk; i++) {
            double a = b0[i] * scale, b = b1[i] * scale;
            X[2 * (size_t)i] = a; X[2 * (size_t)i + 1] = b;
            yacc[i] = (REAL)C->c[0] * a + (REAL)C->d[0] * b;
        }
        memset(Xm, 0, sizeof(REAL) * 2 * (size_t)nk);   /* r = 0 in the first step, but 0*NaN-garbage = NaN */
        double V0 = m0, V1 = m1, lim0 = n0 * (1.0 + 1e-6) + 1e-300, lim1 = n1 * (1.0 + 1e-6) + 1e-300;
        double s0, s1, vm0, vm1;
        PICK2(X, Xm, yacc, alpha2, -1.0, 0.0, C->c[1], C->d[1], &s0, &s1, &vm0, &vm1, &ym);
        { REAL *t = X; X = Xm; Xm = t; }
        V0 = fmax(V0, vm0); V1 = fmax(V1, vm1);
        for (k = 1; k < M - 1; k++) {
            if (fixed_deg > 0) { if (k >= fixed_deg) break; }
            else {
                double err = SF * (C->tc[k] * V0 + C->td[k] * V1);
                if (err <= atol_ * scale + rtol * ym * 0.98) break;
            }
            PICK2(X, Xm, yacc, alpha4, -2.0, -1.0, C->c[k + 1], C->d[k + 1], &s0, &s1, &vm0, &vm1, &ym);
            { REAL *t = X; X = Xm; Xm = t; }
            if (monitor && (s0 > lim0 || s1 > lim1)) { *deg_out = k + 1; return 1; }
            V0 = fmax(V0, vm0); V1 = fmax(V1, vm1);
        }
    } else {
        const double *b = has0 ? b0 : b1;
        const double *cf = has0 ? C->c : C->d;
        const double *tl = has0 ? C->tc : C->td;
        double nb = has0 ? n0 : n1, V = has0 ? m0 : m1;
        REAL *X = w1, *Xm = w2;
        for (int i = 0; i < nk; i++) { double a = b[i] * scale; X[i] = a; yacc[i] = (REAL)cf[0] * a; }
        memset(Xm, 0, sizeof(REAL) * (size_t)nk);
        double lim = nb * (1.0 + 1e-6) + 1e-300;
        double s, vm;
        PICK1(X, Xm, yacc, alpha2, -1.0, 0.0, cf[1], &s, &vm, &ym);
        { REAL *t = X; X = Xm; Xm = t; }
        V = fmax(V, vm);
        for (k = 1; k < M - 1; k++) {
            if (fixed_deg > 0) { if (k >= fixed_deg) break; }
            else {
                double err = SF * tl[k] * V;
                if (err <= atol_ * scale + rtol * ym * 0.98) break;
            }
            PICK1(X, Xm, yacc, alpha4, -2.0, -1.0, cf[k + 1], &s, &vm, &ym);
            { REAL *t = X; X = Xm; Xm = t; }
            if (monitor && s > lim) { *deg_out = k + 1; return 1; }
            V = fmax(V, vm);
        }
    }
    if (scale != 1.0) {
        REAL is = (REAL)1.0 / (REAL)scale;
        for (int i = 0; i < nk; i++) yout[i] = (double)(yacc[i] * is);
        ym /= scale;
    } else if ((void *)yacc != (void *)yout)
        for (int i = 0; i < nk; i++) yout[i] = (double)yacc[i];
    *deg_out = k;
    *ymax_out = ym;
    return 0;
}
