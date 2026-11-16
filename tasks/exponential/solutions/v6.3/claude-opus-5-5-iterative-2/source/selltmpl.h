/* SELL-4 (AVX2) two-steps-per-sweep Chebyshev kernel template.
 * Included from expkern.c with SPAT (0: values, 1: pattern = dg + cval*pattern) and
 * SNV (1 or 2 vectors).  Matrix: chunk c holds rows 4c..4c+3; entry k of row 4c+r is at
 * col/val[4*(cp[c]+k)+r], k < cl[c] (zero-padded).  Vectors are padded to 4*nch entries.
 * For SNV == 2 the two vectors live in separate arrays (A0/B0 and A1/B1).
 * Same contract as cheb1/cheb2; lagc (chunks) must be >= ceil(bandwidth/4)+1. */
#define SCAT_(a, b, c) a##b##c
#define SCAT(a, b, c) SCAT_(a, b, c)
#if SPAT
#define SNAME SCAT(sell, SNV, _pat)
#else
#define SNAME SCAT(sell, SNV, _csr)
#endif

/* acc = (L X)[4c..4c+3] */
#if SPAT
#define CHUNK_DOT1(X, c, acc) do {                                                   \
        const i32 *ci_ = col + 4 * cp[c];                                            \
        int L_ = cl[c], k_ = 0;                                                      \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                \
        for (; k_ + 1 < L_; k_ += 2) {                                               \
            a0_ = _mm256_add_pd(a0_, gath4(X, ci_ + 4 * k_));                        \
            a1_ = _mm256_add_pd(a1_, gath4(X, ci_ + 4 * k_ + 4));                    \
        }                                                                            \
        if (k_ < L_) a0_ = _mm256_add_pd(a0_, gath4(X, ci_ + 4 * k_));               \
        acc = _mm256_fmadd_pd(_mm256_loadu_pd(dg + 4 * (c)), _mm256_loadu_pd(X + 4 * (c)), \
                              _mm256_mul_pd(vcval, _mm256_add_pd(a0_, a1_)));       \
    } while (0)
#define CHUNK_DOT2(X0, X1, c, acc0, acc1) do {                                       \
        const i32 *ci_ = col + 4 * cp[c];                                            \
        int L_ = cl[c];                                                              \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                \
        for (int k_ = 0; k_ < L_; k_++) {                                            \
            a0_ = _mm256_add_pd(a0_, gath4(X0, ci_ + 4 * k_));                       \
            a1_ = _mm256_add_pd(a1_, gath4(X1, ci_ + 4 * k_));                       \
        }                                                                            \
        __m256d dv_ = _mm256_loadu_pd(dg + 4 * (c));                                 \
        acc0 = _mm256_fmadd_pd(dv_, _mm256_loadu_pd(X0 + 4 * (c)), _mm256_mul_pd(vcval, a0_)); \
        acc1 = _mm256_fmadd_pd(dv_, _mm256_loadu_pd(X1 + 4 * (c)), _mm256_mul_pd(vcval, a1_)); \
    } while (0)
#else
#define CHUNK_DOT1(X, c, acc) do {                                                   \
        const i32 *ci_ = col + 4 * cp[c];                                            \
        const double *vi_ = val + 4 * cp[c];                                         \
        int L_ = cl[c], k_ = 0;                                                      \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                \
        for (; k_ + 1 < L_; k_ += 2) {                                               \
            a0_ = _mm256_fmadd_pd(_mm256_loadu_pd(vi_ + 4 * k_), gath4(X, ci_ + 4 * k_), a0_);         \
            a1_ = _mm256_fmadd_pd(_mm256_loadu_pd(vi_ + 4 * k_ + 4), gath4(X, ci_ + 4 * k_ + 4), a1_); \
        }                                                                            \
        if (k_ < L_) a0_ = _mm256_fmadd_pd(_mm256_loadu_pd(vi_ + 4 * k_), gath4(X, ci_ + 4 * k_), a0_); \
        acc = _mm256_add_pd(a0_, a1_);                                               \
    } while (0)
#define CHUNK_DOT2(X0, X1, c, acc0, acc1) do {                                       \
        const i32 *ci_ = col + 4 * cp[c];                                            \
        const double *vi_ = val + 4 * cp[c];                                         \
        int L_ = cl[c];                                                              \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                \
        for (int k_ = 0; k_ < L_; k_++) {                                            \
            __m256d v_ = _mm256_loadu_pd(vi_ + 4 * k_);                              \
            a0_ = _mm256_fmadd_pd(v_, gath4(X0, ci_ + 4 * k_), a0_);                 \
            a1_ = _mm256_fmadd_pd(v_, gath4(X1, ci_ + 4 * k_), a1_);                 \
        }                                                                            \
        acc0 = a0_; acc1 = a1_;                                                      \
    } while (0)
#endif

i64 SNAME(i64 nch, const i64 *cp, const i32 *cl, const i32 *col,
#if SPAT
          const double *dg, double cval,
#else
          const double *val,
#endif
          double sa, double sb, const double *c, const double *tc, const double *d, const double *td, i64 K,
          const double *b0, const double *b1, double *y,
          double *A0, double *B0, double *A1, double *B1,
          double atol, double rtol, double safety, double growlim, double *info, i64 lagc)
{
#if SPAT
    const __m256d vcval = _mm256_set1_pd(cval);
#endif
    const __m256d absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    i64 n4 = 4 * nch;
    (void)d; (void)td; (void)b1; (void)A1; (void)B1;
    double q0 = 0.0, m0 = 0.0, ymax = 0.0;
#if SNV == 2
    double q1 = 0.0, m1 = 0.0;
#endif
    for (i64 i = 0; i < n4; i++) {
        double u = b0[i];
        A0[i] = u;
        q0 += u * u;
        if (fabs(u) > m0) m0 = fabs(u);
        double yy = c[0] * u;
#if SNV == 2
        double v = b1[i];
        A1[i] = v;
        q1 += v * v;
        if (fabs(v) > m1) m1 = fabs(v);
        yy += d[0] * v;
#endif
        y[i] = yy;
        if (fabs(yy) > ymax) ymax = fabs(yy);
    }
    info[0] = 0.0; info[1] = m0; info[2] = ymax;
#if SNV == 1
    if (q0 == 0.0) return 0;
#define TAILEST(j) (tc[j] * M0)
#else
#define TAILEST(j) (tc[j] * M0 + td[j] * M1)
    double M1 = m1, g1 = growlim * growlim * q1;
#endif
    double M0 = m0, g0 = growlim * growlim * q0;
    if (K < 1 || safety * TAILEST(1) <= atol + rtol * ymax) {
        info[0] = TAILEST(1); info[2] = ymax;
        return 0;
    }
    const __m256d vsa = _mm256_set1_pd(sa), vsb = _mm256_set1_pd(sb);
    const __m256d vsa2 = _mm256_set1_pd(2.0 * sa), vsb2 = _mm256_set1_pd(2.0 * sb);
    __m256d e0, e1, x0, x1, ym;
#if SNV == 2
    __m256d f0, f1, z0, z1;
#endif
    /* hsum / hmax helpers */
#define HSUM(v) ({ __m128d lo_ = _mm256_castpd256_pd128(v), hi_ = _mm256_extractf128_pd(v, 1); \
                   lo_ = _mm_add_pd(lo_, hi_); _mm_cvtsd_f64(_mm_add_sd(lo_, _mm_unpackhi_pd(lo_, lo_))); })
#define HMAX(v) ({ __m128d lo_ = _mm256_castpd256_pd128(v), hi_ = _mm256_extractf128_pd(v, 1); \
                   lo_ = _mm_max_pd(lo_, hi_); _mm_cvtsd_f64(_mm_max_sd(lo_, _mm_unpackhi_pd(lo_, lo_))); })
    {
        /* step 1: B = S A */
        const __m256d vc1 = _mm256_set1_pd(c[1]);
        e0 = x0 = ym = _mm256_setzero_pd();
#if SNV == 2
        const __m256d vd1 = _mm256_set1_pd(d[1]);
        e1 = x1 = _mm256_setzero_pd();
#endif
        for (i64 ch = 0; ch < nch; ch++) {
#if SNV == 1
            __m256d s0;
            CHUNK_DOT1(A0, ch, s0);
            __m256d w0 = _mm256_fmadd_pd(vsa, s0, _mm256_mul_pd(vsb, _mm256_loadu_pd(A0 + 4 * ch)));
            _mm256_storeu_pd(B0 + 4 * ch, w0);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            __m256d yv = _mm256_fmadd_pd(vc1, w0, _mm256_loadu_pd(y + 4 * ch));
#else
            __m256d s0, s1;
            CHUNK_DOT2(A0, A1, ch, s0, s1);
            __m256d w0 = _mm256_fmadd_pd(vsa, s0, _mm256_mul_pd(vsb, _mm256_loadu_pd(A0 + 4 * ch)));
            __m256d w1 = _mm256_fmadd_pd(vsa, s1, _mm256_mul_pd(vsb, _mm256_loadu_pd(A1 + 4 * ch)));
            _mm256_storeu_pd(B0 + 4 * ch, w0);
            _mm256_storeu_pd(B1 + 4 * ch, w1);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            e1 = _mm256_fmadd_pd(w1, w1, e1);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
            __m256d yv = _mm256_fmadd_pd(vd1, w1, _mm256_fmadd_pd(vc1, w0, _mm256_loadu_pd(y + 4 * ch)));
#endif
            _mm256_storeu_pd(y + 4 * ch, yv);
            ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
        }
        if (!(HSUM(e0) <= g0)) return -2;
        double t0 = HMAX(x0);
        if (t0 > M0) M0 = t0;
#if SNV == 2
        if (!(HSUM(e1) <= g1)) return -2;
        double t1 = HMAX(x1);
        if (t1 > M1) M1 = t1;
#endif
        ymax = HMAX(ym);
        if (K < 2 || safety * TAILEST(2) <= atol + rtol * ymax) {
            info[0] = TAILEST(2); info[1] = M0; info[2] = ymax;
            return 1;
        }
    }
    i64 j = 1;
    if (lagc < 1) lagc = 1;
    i64 pre = lagc < nch ? lagc : nch;
    while (j + 2 <= K) {
        const __m256d vca = _mm256_set1_pd(c[j + 1]), vcb = _mm256_set1_pd(c[j + 2]);
#if SNV == 2
        const __m256d vda = _mm256_set1_pd(d[j + 1]), vdb = _mm256_set1_pd(d[j + 2]);
        f0 = f1 = z0 = z1 = e1 = x1 = _mm256_setzero_pd();
#endif
        e0 = x0 = ym = _mm256_setzero_pd();
        __m256d e0b = _mm256_setzero_pd(), x0b = _mm256_setzero_pd();
        for (i64 ch = 0; ch < nch + pre; ch++) {
            if (ch < nch) {
                /* first half: A <- 2 S B - A  (T_{j+1}) */
#if SNV == 1
                __m256d s0;
                CHUNK_DOT1(B0, ch, s0);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B0 + 4 * ch))),
                                           _mm256_loadu_pd(A0 + 4 * ch));
                _mm256_storeu_pd(A0 + 4 * ch, w0);
                e0 = _mm256_fmadd_pd(w0, w0, e0);
                x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
                _mm256_storeu_pd(y + 4 * ch, _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch)));
#else
                __m256d s0, s1;
                CHUNK_DOT2(B0, B1, ch, s0, s1);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B0 + 4 * ch))),
                                           _mm256_loadu_pd(A0 + 4 * ch));
                __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B1 + 4 * ch))),
                                           _mm256_loadu_pd(A1 + 4 * ch));
                _mm256_storeu_pd(A0 + 4 * ch, w0);
                _mm256_storeu_pd(A1 + 4 * ch, w1);
                e0 = _mm256_fmadd_pd(w0, w0, e0);
                e1 = _mm256_fmadd_pd(w1, w1, e1);
                x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
                x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
                _mm256_storeu_pd(y + 4 * ch, _mm256_fmadd_pd(vda, w1, _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch))));
#endif
            }
            i64 r = ch - pre;
            if (r >= 0) {
                /* second half: B <- 2 S A - B  (T_{j+2}) */
#if SNV == 1
                __m256d s0;
                CHUNK_DOT1(A0, r, s0);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(A0 + 4 * r))),
                                           _mm256_loadu_pd(B0 + 4 * r));
                _mm256_storeu_pd(B0 + 4 * r, w0);
                e0b = _mm256_fmadd_pd(w0, w0, e0b);
                x0b = _mm256_max_pd(x0b, _mm256_and_pd(w0, absm));
                __m256d yv = _mm256_fmadd_pd(vcb, w0, _mm256_loadu_pd(y + 4 * r));
#else
                __m256d s0, s1;
                CHUNK_DOT2(A0, A1, r, s0, s1);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(A0 + 4 * r))),
                                           _mm256_loadu_pd(B0 + 4 * r));
                __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, _mm256_loadu_pd(A1 + 4 * r))),
                                           _mm256_loadu_pd(B1 + 4 * r));
                _mm256_storeu_pd(B0 + 4 * r, w0);
                _mm256_storeu_pd(B1 + 4 * r, w1);
                e0b = _mm256_fmadd_pd(w0, w0, e0b);
                f1 = _mm256_fmadd_pd(w1, w1, f1);
                x0b = _mm256_max_pd(x0b, _mm256_and_pd(w0, absm));
                z1 = _mm256_max_pd(z1, _mm256_and_pd(w1, absm));
                __m256d yv = _mm256_fmadd_pd(vdb, w1, _mm256_fmadd_pd(vcb, w0, _mm256_loadu_pd(y + 4 * r)));
#endif
                _mm256_storeu_pd(y + 4 * r, yv);
                ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
            }
        }
        if (!(HSUM(e0) <= g0) || !(HSUM(e0b) <= g0)) return -(j + 2);
        double t0 = HMAX(_mm256_max_pd(x0, x0b));
        if (t0 > M0) M0 = t0;
#if SNV == 2
        if (!(HSUM(e1) <= g1) || !(HSUM(f1) <= g1)) return -(j + 2);
        double t1 = HMAX(_mm256_max_pd(x1, z1));
        if (t1 > M1) M1 = t1;
#endif
        ymax = HMAX(ym);
        j += 2;
        if (safety * TAILEST(j + 1) <= atol + rtol * ymax) {
            info[0] = TAILEST(j + 1); info[1] = M0; info[2] = ymax;
            return j;
        }
    }
    if (j < K) {
        /* one final single step: A <- 2 S B - A */
        const __m256d vca = _mm256_set1_pd(c[j + 1]);
#if SNV == 2
        const __m256d vda = _mm256_set1_pd(d[j + 1]);
        e1 = x1 = _mm256_setzero_pd();
#endif
        e0 = x0 = ym = _mm256_setzero_pd();
        for (i64 ch = 0; ch < nch; ch++) {
#if SNV == 1
            __m256d s0;
            CHUNK_DOT1(B0, ch, s0);
            __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B0 + 4 * ch))),
                                       _mm256_loadu_pd(A0 + 4 * ch));
            _mm256_storeu_pd(A0 + 4 * ch, w0);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            __m256d yv = _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch));
#else
            __m256d s0, s1;
            CHUNK_DOT2(B0, B1, ch, s0, s1);
            __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B0 + 4 * ch))),
                                       _mm256_loadu_pd(A0 + 4 * ch));
            __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, _mm256_loadu_pd(B1 + 4 * ch))),
                                       _mm256_loadu_pd(A1 + 4 * ch));
            _mm256_storeu_pd(A0 + 4 * ch, w0);
            _mm256_storeu_pd(A1 + 4 * ch, w1);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            e1 = _mm256_fmadd_pd(w1, w1, e1);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
            __m256d yv = _mm256_fmadd_pd(vda, w1, _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch)));
#endif
            _mm256_storeu_pd(y + 4 * ch, yv);
            ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
        }
        if (!(HSUM(e0) <= g0)) return -(j + 2);
        double t0 = HMAX(x0);
        if (t0 > M0) M0 = t0;
#if SNV == 2
        if (!(HSUM(e1) <= g1)) return -(j + 2);
        double t1 = HMAX(x1);
        if (t1 > M1) M1 = t1;
#endif
        ymax = HMAX(ym);
        j++;
    }
    info[0] = TAILEST(j + 1); info[1] = M0; info[2] = ymax;
    return j;
#undef TAILEST
#undef HSUM
#undef HMAX
}

#undef CHUNK_DOT1
#undef CHUNK_DOT2
#undef SNAME
#undef SCAT
#undef SCAT_
