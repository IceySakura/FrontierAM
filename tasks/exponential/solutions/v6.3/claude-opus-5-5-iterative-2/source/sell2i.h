/* SELL-4 (AVX2) two-vector Chebyshev kernel with interleaved vector storage:
 * X[2*i] = vector 0, X[2*i+1] = vector 1 (one 16-byte load per gathered column).
 * Included from expkern.c with SPAT (0: values, 1: pattern).  Same contract as the SNV == 2
 * kernels of selltmpl.h; A and B hold 2*(4*nch) + 16 doubles and the zero slot is column 4*nch. */
#define S2CAT_(a, b) a##b
#define S2CAT(a, b) S2CAT_(a, b)
#if SPAT
#define S2NAME sell2i_pat
#else
#define S2NAME sell2i_csr
#endif

/* own values of chunk c: v0 = vector 0 of rows 4c..4c+3, v1 = vector 1 */
#define LOAD2(X, c, v0, v1) do {                                                      \
        __m256d lo_ = _mm256_loadu_pd((X) + 8 * (c)), hi_ = _mm256_loadu_pd((X) + 8 * (c) + 4); \
        __m256d t0_ = _mm256_permute2f128_pd(lo_, hi_, 0x20);                         \
        __m256d t1_ = _mm256_permute2f128_pd(lo_, hi_, 0x31);                         \
        v0 = _mm256_unpacklo_pd(t0_, t1_);                                            \
        v1 = _mm256_unpackhi_pd(t0_, t1_);                                            \
    } while (0)
#define STORE2(X, c, w0, w1) do {                                                     \
        __m256d t0_ = _mm256_unpacklo_pd(w0, w1), t1_ = _mm256_unpackhi_pd(w0, w1);   \
        _mm256_storeu_pd((X) + 8 * (c), _mm256_permute2f128_pd(t0_, t1_, 0x20));      \
        _mm256_storeu_pd((X) + 8 * (c) + 4, _mm256_permute2f128_pd(t0_, t1_, 0x31));  \
    } while (0)
/* gathered columns ci[0..3] of both vectors */
#define GATH2(X, ci, x0, x1) do {                                                     \
        __m128d p0_ = _mm_loadu_pd((X) + 2 * (i64)(ci)[0]), p1_ = _mm_loadu_pd((X) + 2 * (i64)(ci)[1]); \
        __m128d p2_ = _mm_loadu_pd((X) + 2 * (i64)(ci)[2]), p3_ = _mm_loadu_pd((X) + 2 * (i64)(ci)[3]); \
        __m256d t02_ = _mm256_insertf128_pd(_mm256_castpd128_pd256(p0_), p2_, 1);    \
        __m256d t13_ = _mm256_insertf128_pd(_mm256_castpd128_pd256(p1_), p3_, 1);    \
        x0 = _mm256_unpacklo_pd(t02_, t13_);                                          \
        x1 = _mm256_unpackhi_pd(t02_, t13_);                                          \
    } while (0)

#if SPAT
#define CDOT2I(X, c, own0, own1, acc0, acc1) do {                                     \
        const i32 *ci_ = col + 4 * cp[c];                                             \
        int L_ = cl[c];                                                               \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                 \
        for (int k_ = 0; k_ < L_; k_++) {                                             \
            __m256d g0_, g1_;                                                         \
            GATH2(X, ci_ + 4 * k_, g0_, g1_);                                         \
            a0_ = _mm256_add_pd(a0_, g0_);                                            \
            a1_ = _mm256_add_pd(a1_, g1_);                                            \
        }                                                                             \
        __m256d dv_ = _mm256_loadu_pd(dg + 4 * (c));                                  \
        acc0 = _mm256_fmadd_pd(dv_, own0, _mm256_mul_pd(vcval, a0_));                 \
        acc1 = _mm256_fmadd_pd(dv_, own1, _mm256_mul_pd(vcval, a1_));                 \
    } while (0)
#else
#define CDOT2I(X, c, own0, own1, acc0, acc1) do {                                     \
        const i32 *ci_ = col + 4 * cp[c];                                             \
        const double *vi_ = val + 4 * cp[c];                                          \
        int L_ = cl[c];                                                               \
        __m256d a0_ = _mm256_setzero_pd(), a1_ = _mm256_setzero_pd();                 \
        for (int k_ = 0; k_ < L_; k_++) {                                             \
            __m256d g0_, g1_, v_ = _mm256_loadu_pd(vi_ + 4 * k_);                     \
            GATH2(X, ci_ + 4 * k_, g0_, g1_);                                         \
            a0_ = _mm256_fmadd_pd(v_, g0_, a0_);                                      \
            a1_ = _mm256_fmadd_pd(v_, g1_, a1_);                                      \
        }                                                                             \
        (void)own0; (void)own1;                                                       \
        acc0 = a0_; acc1 = a1_;                                                       \
    } while (0)
#endif

i64 S2NAME(i64 nch, const i64 *cp, const i32 *cl, const i32 *col,
#if SPAT
           const double *dg, double cval,
#else
           const double *val,
#endif
           double sa, double sb, const double *c, const double *tc, const double *d, const double *td, i64 K,
           const double *b0, const double *b1, double *y, double *A, double *B,
           double atol, double rtol, double safety, double growlim, double *info, i64 lagc)
{
#if SPAT
    const __m256d vcval = _mm256_set1_pd(cval);
#endif
    const __m256d absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
    i64 n4 = 4 * nch;
    double q0 = 0.0, q1 = 0.0, m0 = 0.0, m1 = 0.0, ymax = 0.0;
    for (i64 i = 0; i < n4; i++) {
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
#define TAIL2(j) (tc[j] * M0 + td[j] * M1)
    if (K < 1 || safety * TAIL2(1) <= atol + rtol * ymax) {
        info[0] = TAIL2(1); info[2] = ymax;
        return 0;
    }
#define HSUM2(v) ({ __m128d lo_ = _mm256_castpd256_pd128(v), hi_ = _mm256_extractf128_pd(v, 1); \
                    lo_ = _mm_add_pd(lo_, hi_); _mm_cvtsd_f64(_mm_add_sd(lo_, _mm_unpackhi_pd(lo_, lo_))); })
#define HMAX2(v) ({ __m128d lo_ = _mm256_castpd256_pd128(v), hi_ = _mm256_extractf128_pd(v, 1); \
                    lo_ = _mm_max_pd(lo_, hi_); _mm_cvtsd_f64(_mm_max_sd(lo_, _mm_unpackhi_pd(lo_, lo_))); })
    const __m256d vsa = _mm256_set1_pd(sa), vsb = _mm256_set1_pd(sb);
    const __m256d vsa2 = _mm256_set1_pd(2.0 * sa), vsb2 = _mm256_set1_pd(2.0 * sb);
    {
        /* step 1: B = S A */
        const __m256d vc1 = _mm256_set1_pd(c[1]), vd1 = _mm256_set1_pd(d[1]);
        __m256d e0 = _mm256_setzero_pd(), e1 = e0, x0 = e0, x1 = e0, ym = e0;
        for (i64 ch = 0; ch < nch; ch++) {
            __m256d o0, o1, s0, s1;
            LOAD2(A, ch, o0, o1);
            CDOT2I(A, ch, o0, o1, s0, s1);
            __m256d w0 = _mm256_fmadd_pd(vsa, s0, _mm256_mul_pd(vsb, o0));
            __m256d w1 = _mm256_fmadd_pd(vsa, s1, _mm256_mul_pd(vsb, o1));
            STORE2(B, ch, w0, w1);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            e1 = _mm256_fmadd_pd(w1, w1, e1);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
            __m256d yv = _mm256_fmadd_pd(vd1, w1, _mm256_fmadd_pd(vc1, w0, _mm256_loadu_pd(y + 4 * ch)));
            _mm256_storeu_pd(y + 4 * ch, yv);
            ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
        }
        if (!(HSUM2(e0) <= g0) || !(HSUM2(e1) <= g1)) return -2;
        double t0 = HMAX2(x0), t1 = HMAX2(x1);
        if (t0 > M0) M0 = t0;
        if (t1 > M1) M1 = t1;
        ymax = HMAX2(ym);
        if (K < 2 || safety * TAIL2(2) <= atol + rtol * ymax) {
            info[0] = TAIL2(2); info[1] = M0; info[2] = ymax;
            return 1;
        }
    }
    i64 j = 1;
    if (lagc < 1) lagc = 1;
    i64 pre = lagc < nch ? lagc : nch;
    while (j + 2 <= K) {
        const __m256d vca = _mm256_set1_pd(c[j + 1]), vcb = _mm256_set1_pd(c[j + 2]);
        const __m256d vda = _mm256_set1_pd(d[j + 1]), vdb = _mm256_set1_pd(d[j + 2]);
        __m256d e0 = _mm256_setzero_pd(), e1 = e0, f0 = e0, f1 = e0, x0 = e0, x1 = e0, z0 = e0, z1 = e0, ym = e0;
        for (i64 ch = 0; ch < nch + pre; ch++) {
            if (ch < nch) {
                /* first half: A <- 2 S B - A  (T_{j+1}) */
                __m256d o0, o1, p0, p1, s0, s1;
                LOAD2(B, ch, o0, o1);
                LOAD2(A, ch, p0, p1);
                CDOT2I(B, ch, o0, o1, s0, s1);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, o0)), p0);
                __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, o1)), p1);
                STORE2(A, ch, w0, w1);
                e0 = _mm256_fmadd_pd(w0, w0, e0);
                e1 = _mm256_fmadd_pd(w1, w1, e1);
                x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
                x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
                _mm256_storeu_pd(y + 4 * ch, _mm256_fmadd_pd(vda, w1, _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch))));
            }
            i64 r = ch - pre;
            if (r >= 0) {
                /* second half: B <- 2 S A - B  (T_{j+2}) */
                __m256d o0, o1, p0, p1, s0, s1;
                LOAD2(A, r, o0, o1);
                LOAD2(B, r, p0, p1);
                CDOT2I(A, r, o0, o1, s0, s1);
                __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, o0)), p0);
                __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, o1)), p1);
                STORE2(B, r, w0, w1);
                f0 = _mm256_fmadd_pd(w0, w0, f0);
                f1 = _mm256_fmadd_pd(w1, w1, f1);
                z0 = _mm256_max_pd(z0, _mm256_and_pd(w0, absm));
                z1 = _mm256_max_pd(z1, _mm256_and_pd(w1, absm));
                __m256d yv = _mm256_fmadd_pd(vdb, w1, _mm256_fmadd_pd(vcb, w0, _mm256_loadu_pd(y + 4 * r)));
                _mm256_storeu_pd(y + 4 * r, yv);
                ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
            }
        }
        if (!(HSUM2(e0) <= g0) || !(HSUM2(f0) <= g0) || !(HSUM2(e1) <= g1) || !(HSUM2(f1) <= g1))
            return -(j + 2);
        double t0 = HMAX2(_mm256_max_pd(x0, z0)), t1 = HMAX2(_mm256_max_pd(x1, z1));
        if (t0 > M0) M0 = t0;
        if (t1 > M1) M1 = t1;
        ymax = HMAX2(ym);
        j += 2;
        if (safety * TAIL2(j + 1) <= atol + rtol * ymax) {
            info[0] = TAIL2(j + 1); info[1] = M0; info[2] = ymax;
            return j;
        }
    }
    if (j < K) {
        /* one final single step: A <- 2 S B - A */
        const __m256d vca = _mm256_set1_pd(c[j + 1]), vda = _mm256_set1_pd(d[j + 1]);
        __m256d e0 = _mm256_setzero_pd(), e1 = e0, x0 = e0, x1 = e0, ym = e0;
        for (i64 ch = 0; ch < nch; ch++) {
            __m256d o0, o1, p0, p1, s0, s1;
            LOAD2(B, ch, o0, o1);
            LOAD2(A, ch, p0, p1);
            CDOT2I(B, ch, o0, o1, s0, s1);
            __m256d w0 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s0, _mm256_mul_pd(vsb2, o0)), p0);
            __m256d w1 = _mm256_sub_pd(_mm256_fmadd_pd(vsa2, s1, _mm256_mul_pd(vsb2, o1)), p1);
            STORE2(A, ch, w0, w1);
            e0 = _mm256_fmadd_pd(w0, w0, e0);
            e1 = _mm256_fmadd_pd(w1, w1, e1);
            x0 = _mm256_max_pd(x0, _mm256_and_pd(w0, absm));
            x1 = _mm256_max_pd(x1, _mm256_and_pd(w1, absm));
            __m256d yv = _mm256_fmadd_pd(vda, w1, _mm256_fmadd_pd(vca, w0, _mm256_loadu_pd(y + 4 * ch)));
            _mm256_storeu_pd(y + 4 * ch, yv);
            ym = _mm256_max_pd(ym, _mm256_and_pd(yv, absm));
        }
        if (!(HSUM2(e0) <= g0) || !(HSUM2(e1) <= g1)) return -(j + 2);
        double t0 = HMAX2(x0), t1 = HMAX2(x1);
        if (t0 > M0) M0 = t0;
        if (t1 > M1) M1 = t1;
        ymax = HMAX2(ym);
        j++;
    }
    info[0] = TAIL2(j + 1); info[1] = M0; info[2] = ymax;
    return j;
#undef TAIL2
#undef HSUM2
#undef HMAX2
}

#undef CDOT2I
#undef LOAD2
#undef STORE2
#undef GATH2
#undef S2NAME
#undef S2CAT
#undef S2CAT_
