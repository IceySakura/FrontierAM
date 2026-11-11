// Solid Spherical Harmonic + Smooth Plummer Residual FMM Kernels (Variant C)
//
// Mathematical formulation from W. Dehnen (2014), Comput. Astrophys. Cosmol. 1:1,
// Section 2.1 (Eqs. 2a-2c, 3a-3f) and Appendix A.3-A.4 (Eqs. A.10-A.34, Table 2):
//   - Split Plummer Green's function:
//       psi_Plummer(r) = 1 / sqrt(r^2 + eps^2) = 1 / r + [ 1 / sqrt(r^2 + eps^2) - 1 / r ]
//   - Newtonian 1/r part is harmonic (nabla^2(1/r) = 0) and is expanded to high order P_harm (up to 16)
//     in real-valued solid spherical harmonics U_n^m(r) and T_n^m(r) with only (P_harm + 1)^2 terms.
//   - Smooth residual delta_psi(r) = (r^2 + eps^2)^{-1/2} - r^{-1} is expanded to low order P_res
//     using the Cartesian FMM of Dehnen (2002) with radial derivatives D_k^res = D_k(eps) - D_k(0)
//     evaluated without catastrophic cancellation via rationalized difference.
//   - Near pairs (P2P) use the exact softened Plummer kernel (r^2 + eps^2)^{-3/2}.

#ifndef FMM_HARMONIC_HPP
#define FMM_HARMONIC_HPP

#include "fmm_kernels.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fmm {

constexpr int P_HARM_MAX = 16;

constexpr int harm_total(int P) {
    return (P + 1) * (P + 1);
}

constexpr int nm_idx(int n, int m) {
    return n * (n + 1) + m;
}

struct HarmTables {
    double inv_n2_m2[P_HARM_MAX + 1][P_HARM_MAX + 1];
    double fact[2 * P_HARM_MAX + 2];
    double inv_fact[2 * P_HARM_MAX + 2];

    constexpr HarmTables() : inv_n2_m2{}, fact{}, inv_fact{} {
        fact[0] = 1.0;
        inv_fact[0] = 1.0;
        for (int i = 1; i <= 2 * P_HARM_MAX + 1; ++i) {
            fact[i] = fact[i - 1] * i;
            inv_fact[i] = 1.0 / fact[i];
        }
        for (int n = 1; n <= P_HARM_MAX; ++n) {
            for (int m = 0; m < n; ++m) {
                inv_n2_m2[n][m] = 1.0 / static_cast<double>(n * n - m * m);
            }
        }
    }
};

constexpr HarmTables HTAB{};

// Evaluate real solid spherical harmonics U_n^m(x, y, z) for n = 0..P (Dehnen 2014 Eqs. 2b, A.20, A.21, A.25a)
template <int P>
inline void eval_U_real(const Vec3& r, double* __restrict__ U) {
    const double x = r.x, y = r.y, z = r.z;
    const double r2 = x * x + y * y + z * z;

    U[nm_idx(0, 0)] = 1.0;
    if constexpr (P >= 1) {
        U[nm_idx(1, 0)] = z;
        for (int n = 2; n <= P; ++n) {
            U[nm_idx(n, 0)] = ((2 * n - 1) * z * U[nm_idx(n - 1, 0)] - r2 * U[nm_idx(n - 2, 0)]) * HTAB.inv_n2_m2[n][0];
        }
        double u_re = 1.0, u_im = 0.0;
        for (int m = 1; m <= P; ++m) {
            const double inv_2m = 0.5 / static_cast<double>(m);
            const double next_re = (x * u_re - y * u_im) * inv_2m;
            const double next_im = (x * u_im + y * u_re) * inv_2m;
            u_re = next_re;
            u_im = next_im;
            U[nm_idx(m,  m)] = u_re;
            U[nm_idx(m, -m)] = u_im;

            if (m + 1 <= P) {
                const int n1 = m + 1;
                const double c1 = static_cast<double>(2 * n1 - 1) * z * HTAB.inv_n2_m2[n1][m];
                U[nm_idx(n1,  m)] = c1 * u_re;
                U[nm_idx(n1, -m)] = c1 * u_im;
                for (int n = m + 2; n <= P; ++n) {
                    const double inv_d = HTAB.inv_n2_m2[n][m];
                    const double cz = static_cast<double>(2 * n - 1) * z;
                    U[nm_idx(n,  m)] = (cz * U[nm_idx(n - 1,  m)] - r2 * U[nm_idx(n - 2,  m)]) * inv_d;
                    U[nm_idx(n, -m)] = (cz * U[nm_idx(n - 1, -m)] - r2 * U[nm_idx(n - 2, -m)]) * inv_d;
                }
            }
        }
    }
}

// Evaluate real irregular solid spherical harmonics T_n^m(x, y, z) for n = 0..P (Dehnen 2014 Eqs. 2a, A.19, A.21, A.25b)
template <int P>
inline void eval_T_real(const Vec3& r, double inv_r, double inv_r2, double* __restrict__ T) {
    const double x = r.x, y = r.y, z = r.z;

    T[nm_idx(0, 0)] = inv_r;
    if constexpr (P >= 1) {
        T[nm_idx(1, 0)] = z * inv_r2 * inv_r;
        for (int n = 2; n <= P; ++n) {
            const double c_prev = static_cast<double>((n - 1) * (n - 1));
            T[nm_idx(n, 0)] = inv_r2 * (static_cast<double>(2 * n - 1) * z * T[nm_idx(n - 1, 0)] - c_prev * T[nm_idx(n - 2, 0)]);
        }
        double t_re = inv_r, t_im = 0.0;
        for (int m = 1; m <= P; ++m) {
            const double fac = static_cast<double>(2 * m - 1) * inv_r2;
            const double next_re = fac * (x * t_re - y * t_im);
            const double next_im = fac * (x * t_im + y * t_re);
            t_re = next_re;
            t_im = next_im;
            T[nm_idx(m,  m)] = t_re;
            T[nm_idx(m, -m)] = t_im;

            if (m + 1 <= P) {
                const int n1 = m + 1;
                const double c1 = static_cast<double>(2 * n1 - 1) * z * inv_r2;
                T[nm_idx(n1,  m)] = c1 * t_re;
                T[nm_idx(n1, -m)] = c1 * t_im;
                for (int n = m + 2; n <= P; ++n) {
                    const double c_prev = static_cast<double>((n - 1) * (n - 1) - m * m);
                    const double cz = static_cast<double>(2 * n - 1) * z;
                    T[nm_idx(n,  m)] = inv_r2 * (cz * T[nm_idx(n - 1,  m)] - c_prev * T[nm_idx(n - 2,  m)]);
                    T[nm_idx(n, -m)] = inv_r2 * (cz * T[nm_idx(n - 1, -m)] - c_prev * T[nm_idx(n - 2, -m)]);
                }
            }
        }
    }
}

// 4-way AVX2 evaluation of irregular solid harmonics T_n^m(x, y, z) with dynamic power-of-two distance scaling
// to prevent IEEE-754 double overflow when r < 0.01 and P >= 10!
// For each lane j, scale_r[j] = 1.0 when r >= 1.0, or a power of two >= 1/r when r < 1.0,
// so that r_scaled = r * scale_r >= 1.0 and T_n^m(r_scaled) never overflows!
template <int P>
inline void eval_T_real_batch4_avx2(
    const __m256d& vx, const __m256d& vy, const __m256d& vz,
    const __m256d& vinv_r, const __m256d& vinv_r2,
    __m256d* __restrict__ T
) {
    // l = 0 vertical recurrence:
    T[nm_idx(0, 0)] = vinv_r;
    if constexpr (P >= 1) {
        T[nm_idx(1, 0)] = _mm256_mul_pd(vz, _mm256_mul_pd(vinv_r2, vinv_r));
        for (int n = 2; n <= P; ++n) {
            const __m256d c_prev = _mm256_set1_pd(static_cast<double>((n - 1) * (n - 1)));
            const __m256d cz = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(2 * n - 1)), vz);
            const __m256d num = _mm256_sub_pd(_mm256_mul_pd(cz, T[nm_idx(n - 1, 0)]), _mm256_mul_pd(c_prev, T[nm_idx(n - 2, 0)]));
            T[nm_idx(n, 0)] = _mm256_mul_pd(vinv_r2, num);
        }
        __m256d t_re = vinv_r, t_im = _mm256_setzero_pd();
        for (int m = 1; m <= P; ++m) {
            const __m256d fac = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(2 * m - 1)), vinv_r2);
            const __m256d next_re = _mm256_mul_pd(fac, _mm256_sub_pd(_mm256_mul_pd(vx, t_re), _mm256_mul_pd(vy, t_im)));
            const __m256d next_im = _mm256_mul_pd(fac, _mm256_add_pd(_mm256_mul_pd(vx, t_im), _mm256_mul_pd(vy, t_re)));
            t_re = next_re;
            t_im = next_im;
            T[nm_idx(m,  m)] = t_re;
            T[nm_idx(m, -m)] = t_im;

            if (m + 1 <= P) {
                const int n1 = m + 1;
                const __m256d c1 = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(2 * n1 - 1)), _mm256_mul_pd(vz, vinv_r2));
                T[nm_idx(n1,  m)] = _mm256_mul_pd(c1, t_re);
                T[nm_idx(n1, -m)] = _mm256_mul_pd(c1, t_im);
                for (int n = m + 2; n <= P; ++n) {
                    const __m256d c_prev = _mm256_set1_pd(static_cast<double>((n - 1) * (n - 1) - m * m));
                    const __m256d cz = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(2 * n - 1)), vz);
                    const __m256d num_re = _mm256_sub_pd(_mm256_mul_pd(cz, T[nm_idx(n - 1,  m)]), _mm256_mul_pd(c_prev, T[nm_idx(n - 2,  m)]));
                    const __m256d num_im = _mm256_sub_pd(_mm256_mul_pd(cz, T[nm_idx(n - 1, -m)]), _mm256_mul_pd(c_prev, T[nm_idx(n - 2, -m)]));
                    T[nm_idx(n,  m)] = _mm256_mul_pd(vinv_r2, num_re);
                    T[nm_idx(n, -m)] = _mm256_mul_pd(vinv_r2, num_im);
                }
            }
        }
    }
}

inline void get_complex_harm(const double* __restrict__ A, int n, int m, double& re, double& im) {
    if (m > 0) {
        re = A[nm_idx(n,  m)];
        im = A[nm_idx(n, -m)];
    } else if (m == 0) {
        re = A[nm_idx(n, 0)];
        im = 0.0;
    } else {
        const int am = -m;
        const double s = (am & 1) ? -1.0 : 1.0;
        re =  s * A[nm_idx(n,  am)];
        im = -s * A[nm_idx(n, -am)];
    }
}

template <int P>
inline void harm_p2m_add_body(const Vec3& d, double m_a, double* __restrict__ M) {
    double U[harm_total(P)];
    eval_U_real<P>(d, U);
    for (int i = 0; i < harm_total(P); ++i) {
        M[i] += m_a * U[i];
    }
}

template <int P>
inline void harm_m2m_add_cell(const Vec3& v, const double* __restrict__ M_child, double* __restrict__ M_parent) {
    double U[harm_total(P)];
    eval_U_real<P>(v, U);

    for (int n = 0; n <= P; ++n) {
        for (int m = 0; m <= n; ++m) {
            double sum_re = 0.0, sum_im = 0.0;
            for (int k = 0; k <= n; ++k) {
                const int nk = n - k;
                const int l_lo = std::max(-k, m - nk);
                const int l_hi = std::min( k, m + nk);
                for (int l = l_lo; l <= l_hi; ++l) {
                    double u_re, u_im, mc_re, mc_im;
                    get_complex_harm(U, k, l, u_re, u_im);
                    get_complex_harm(M_child, nk, m - l, mc_re, mc_im);
                    sum_re += u_re * mc_re - u_im * mc_im;
                    sum_im += u_re * mc_im + u_im * mc_re;
                }
            }
            M_parent[nm_idx(n, m)] += sum_re;
            if (m > 0) {
                M_parent[nm_idx(n, -m)] += sum_im;
            }
        }
    }
}

// 4-way AVX2 batched mutual harmonic M2L kernel
// Exploits:
//   1. Center-of-mass expansion implies dipole k = 1 vanishes (M_1^l = 0).
//   2. Highest multipole k = P only affects n = 0 (potential), not accelerations (n = 1..P).
//   3. Conjugate symmetry of l and -l folds the inner loop to l = 1..k (halving inner loop iterations!).
//   4. Parity factorization: for fixed (n, m), k_even and k_odd contributions are accumulated in registers
//      and added/subtracted once at the end!
template <int P>
inline void harm_m2l_batch4_avx2(
    const Vec3 r_AB[4], // z_B - z_A
    double* const FA[4], const double* const MB[4],
    double* const FB[4], const double* const MA[4]
) {
    constexpr int H_LEN = harm_total(P);
    constexpr int M_USED = harm_total(P - 1);

    const __m256d vx = _mm256_set_pd(r_AB[3].x, r_AB[2].x, r_AB[1].x, r_AB[0].x);
    const __m256d vy = _mm256_set_pd(r_AB[3].y, r_AB[2].y, r_AB[1].y, r_AB[0].y);
    const __m256d vz = _mm256_set_pd(r_AB[3].z, r_AB[2].z, r_AB[1].z, r_AB[0].z);
    const __m256d vr2 = _mm256_fmadd_pd(vz, vz, _mm256_fmadd_pd(vy, vy, _mm256_mul_pd(vx, vx)));
    const __m256d vinv_r2 = _mm256_div_pd(_mm256_set1_pd(1.0), vr2);
    const __m256d vinv_r  = _mm256_sqrt_pd(vinv_r2);

    __m256d T[H_LEN];
    eval_T_real_batch4_avx2<P>(vx, vy, vz, vinv_r, vinv_r2, T);

    // Gather MA and MB for k = 0 and k = 2..P-1
    __m256d vMA[M_USED];
    __m256d vMB[M_USED];
    const double* ma0 = MA[0]; const double* ma1 = MA[1]; const double* ma2 = MA[2]; const double* ma3 = MA[3];
    const double* mb0 = MB[0]; const double* mb1 = MB[1]; const double* mb2 = MB[2]; const double* mb3 = MB[3];
    vMA[0] = _mm256_set_pd(ma3[0], ma2[0], ma1[0], ma0[0]);
    vMB[0] = _mm256_set_pd(mb3[0], mb2[0], mb1[0], mb0[0]);
    for (int i = 4; i < M_USED; ++i) {
        vMA[i] = _mm256_set_pd(ma3[i], ma2[i], ma1[i], ma0[i]);
        vMB[i] = _mm256_set_pd(mb3[i], mb2[i], mb1[i], mb0[i]);
    }

    alignas(32) double tmpB_re[4], tmpB_im[4], tmpA_re[4], tmpA_im[4];

    for (int n = 1; n <= P; ++n) {
        const bool odd_n = (n & 1) != 0;
        // m = 0 (imaginary part of F_n^0 is 0):
        {
            const __m256d t_n0 = T[nm_idx(n, 0)];
            __m256d fb_re = _mm256_mul_pd(vMA[0], t_n0);
            __m256d fa_even = _mm256_mul_pd(vMB[0], t_n0);
            __m256d fa_odd  = _mm256_setzero_pd();

            for (int k = 2; k <= P - n; ++k) {
                const int nk = n + k;
                const int base_k = nm_idx(k, 0);
                const int base_nk = nm_idx(nk, 0);
                const __m256d t0 = T[base_nk];
                __m256d sb = _mm256_mul_pd(vMA[base_k], t0);
                __m256d sa = _mm256_mul_pd(vMB[base_k], t0);
                // For m = 0, l and -l give 2 * (ma_re * t_re + ma_im * t_im)
                __m256d sb_l = _mm256_setzero_pd();
                __m256d sa_l = _mm256_setzero_pd();
                for (int l = 1; l <= k; ++l) {
                    const __m256d t_re = T[base_nk + l];
                    const __m256d t_im = T[base_nk - l];
                    const __m256d ma_re = vMA[base_k + l], ma_im = vMA[base_k - l];
                    const __m256d mb_re = vMB[base_k + l], mb_im = vMB[base_k - l];
                    sb_l = _mm256_fmadd_pd(ma_re, t_re, _mm256_fmadd_pd(ma_im, t_im, sb_l));
                    sa_l = _mm256_fmadd_pd(mb_re, t_re, _mm256_fmadd_pd(mb_im, t_im, sa_l));
                }
                sb = _mm256_add_pd(sb, _mm256_add_pd(sb_l, sb_l));
                sa = _mm256_add_pd(sa, _mm256_add_pd(sa_l, sa_l));
                fb_re = _mm256_add_pd(fb_re, sb);
                if (k & 1) fa_odd  = _mm256_add_pd(fa_odd, sa);
                else       fa_even = _mm256_add_pd(fa_even, sa);
            }
            const __m256d fa_re = odd_n ? _mm256_sub_pd(fa_odd, fa_even) : _mm256_sub_pd(fa_even, fa_odd);
            const int idx0 = nm_idx(n, 0);
            _mm256_store_pd(tmpB_re, fb_re);
            _mm256_store_pd(tmpA_re, fa_re);
            FB[0][idx0] += tmpB_re[0]; FA[0][idx0] += tmpA_re[0];
            FB[1][idx0] += tmpB_re[1]; FA[1][idx0] += tmpA_re[1];
            FB[2][idx0] += tmpB_re[2]; FA[2][idx0] += tmpA_re[2];
            FB[3][idx0] += tmpB_re[3]; FA[3][idx0] += tmpA_re[3];
        }

        // m = 1 .. n
        for (int m = 1; m <= n; ++m) {
            const int idx_re = nm_idx(n,  m);
            const int idx_im = nm_idx(n, -m);
            const __m256d t_nm_re = T[idx_re];
            const __m256d t_nm_im = T[idx_im];

            __m256d fb_re = _mm256_mul_pd(vMA[0], t_nm_re);
            __m256d fb_im = _mm256_mul_pd(vMA[0], t_nm_im);
            __m256d fa_even_re = _mm256_mul_pd(vMB[0], t_nm_re);
            __m256d fa_even_im = _mm256_mul_pd(vMB[0], t_nm_im);
            __m256d fa_odd_re  = _mm256_setzero_pd();
            __m256d fa_odd_im  = _mm256_setzero_pd();

            for (int k = 2; k <= P - n; ++k) {
                const int nk = n + k;
                const int base_k = nm_idx(k, 0);
                const int base_nk = nm_idx(nk, 0);

                // l = 0 term:
                const __m256d t0_re = T[base_nk + m];
                const __m256d t0_im = T[base_nk - m];
                __m256d sb_re = _mm256_mul_pd(vMA[base_k], t0_re);
                __m256d sb_im = _mm256_mul_pd(vMA[base_k], t0_im);
                __m256d sa_re = _mm256_mul_pd(vMB[base_k], t0_re);
                __m256d sa_im = _mm256_mul_pd(vMB[base_k], t0_im);

                // Combine +l and -l for l = 1..k:
                // For +l: m+l > 0, so T_{nk}^{m+l} = (T[base_nk + m + l], T[base_nk - (m + l)]).
                // For -l: M_k^{-l*} = (-1)^l (ma_re + i ma_im).
                // Let tp_re = T_{nk}^{m+l}.re, tp_im = T_{nk}^{m+l}.im,
                // and tm_re = (-1)^l T_{nk}^{m-l}.re, tm_im = (-1)^l T_{nk}^{m-l}.im.
                // Then:
                //   (+l term) + (-l term) =
                //     Re: ma_re * (tp_re + tm_re) + ma_im * (tp_im - tm_im)
                //     Im: ma_re * (tp_im + tm_im) - ma_im * (tp_re - tm_re)!
                for (int l = 1; l <= k; ++l) {
                    const __m256d tp_re = T[base_nk + (m + l)];
                    const __m256d tp_im = T[base_nk - (m + l)];
                    const int ml = m - l;
                    __m256d tm_re, tm_im;
                    if (ml > 0) {
                        if (l & 1) {
                            tm_re = _mm256_sub_pd(_mm256_setzero_pd(), T[base_nk + ml]);
                            tm_im = _mm256_sub_pd(_mm256_setzero_pd(), T[base_nk - ml]);
                        } else {
                            tm_re = T[base_nk + ml];
                            tm_im = T[base_nk - ml];
                        }
                    } else if (ml == 0) {
                        tm_im = _mm256_setzero_pd();
                        if (l & 1) tm_re = _mm256_sub_pd(_mm256_setzero_pd(), T[base_nk]);
                        else       tm_re = T[base_nk];
                    } else {
                        const int aml = -ml;
                        // (-1)^{l + aml} = (-1)^m!
                        if (m & 1) {
                            tm_re = _mm256_sub_pd(_mm256_setzero_pd(), T[base_nk + aml]);
                            tm_im = T[base_nk - aml];
                        } else {
                            tm_re = T[base_nk + aml];
                            tm_im = _mm256_sub_pd(_mm256_setzero_pd(), T[base_nk - aml]);
                        }
                    }

                    const __m256d sum_re  = _mm256_add_pd(tp_re, tm_re);
                    const __m256d diff_re = _mm256_sub_pd(tp_re, tm_re);
                    const __m256d sum_im  = _mm256_add_pd(tp_im, tm_im);
                    const __m256d diff_im = _mm256_sub_pd(tp_im, tm_im);

                    const __m256d ma_re = vMA[base_k + l], ma_im = vMA[base_k - l];
                    const __m256d mb_re = vMB[base_k + l], mb_im = vMB[base_k - l];

                    sb_re = _mm256_fmadd_pd(ma_re, sum_re, _mm256_fmadd_pd(ma_im, diff_im, sb_re));
                    sb_im = _mm256_fnmadd_pd(ma_im, diff_re, _mm256_fmadd_pd(ma_re, sum_im, sb_im));

                    sa_re = _mm256_fmadd_pd(mb_re, sum_re, _mm256_fmadd_pd(mb_im, diff_im, sa_re));
                    sa_im = _mm256_fnmadd_pd(mb_im, diff_re, _mm256_fmadd_pd(mb_re, sum_im, sa_im));
                }

                fb_re = _mm256_add_pd(fb_re, sb_re);
                fb_im = _mm256_add_pd(fb_im, sb_im);
                if (k & 1) {
                    fa_odd_re = _mm256_add_pd(fa_odd_re, sa_re);
                    fa_odd_im = _mm256_add_pd(fa_odd_im, sa_im);
                } else {
                    fa_even_re = _mm256_add_pd(fa_even_re, sa_re);
                    fa_even_im = _mm256_add_pd(fa_even_im, sa_im);
                }
            }

            const __m256d fa_re = odd_n ? _mm256_sub_pd(fa_odd_re, fa_even_re) : _mm256_sub_pd(fa_even_re, fa_odd_re);
            const __m256d fa_im = odd_n ? _mm256_sub_pd(fa_odd_im, fa_even_im) : _mm256_sub_pd(fa_even_im, fa_odd_im);

            _mm256_store_pd(tmpB_re, fb_re); _mm256_store_pd(tmpB_im, fb_im);
            _mm256_store_pd(tmpA_re, fa_re); _mm256_store_pd(tmpA_im, fa_im);
            for (int j = 0; j < 4; ++j) {
                FB[j][idx_re] += tmpB_re[j]; FB[j][idx_im] += tmpB_im[j];
                FA[j][idx_re] += tmpA_re[j]; FA[j][idx_im] += tmpA_im[j];
            }
        }
    }
}

// Single-pair mutual harmonic M2L (for remaining <4 interactions in buffer)
template <int P>
inline void harm_m2l_mutual(
    const Vec3& r_AB,
    double inv_r, double inv_r2,
    const double* __restrict__ MA, double* __restrict__ FB,
    const double* __restrict__ MB, double* __restrict__ FA
) {
    double T[harm_total(P)];
    eval_T_real<P>(r_AB, inv_r, inv_r2, T);

    for (int n = 1; n <= P; ++n) {
        const bool odd_n = (n & 1) != 0;
        {
            const double t_n0 = T[nm_idx(n, 0)];
            double fb_re = MA[0] * t_n0;
            double fa_even = MB[0] * t_n0, fa_odd = 0.0;
            for (int k = 2; k <= P - n; ++k) {
                const int nk = n + k;
                const int base_k = nm_idx(k, 0), base_nk = nm_idx(nk, 0);
                double sb = MA[base_k] * T[base_nk];
                double sa = MB[base_k] * T[base_nk];
                double sb_l = 0.0, sa_l = 0.0;
                for (int l = 1; l <= k; ++l) {
                    const double t_re = T[base_nk + l], t_im = T[base_nk - l];
                    sb_l += MA[base_k + l] * t_re + MA[base_k - l] * t_im;
                    sa_l += MB[base_k + l] * t_re + MB[base_k - l] * t_im;
                }
                sb += 2.0 * sb_l;
                sa += 2.0 * sa_l;
                fb_re += sb;
                if (k & 1) fa_odd += sa;
                else       fa_even += sa;
            }
            FB[nm_idx(n, 0)] += fb_re;
            FA[nm_idx(n, 0)] += odd_n ? (fa_odd - fa_even) : (fa_even - fa_odd);
        }
        for (int m = 1; m <= n; ++m) {
            const int idx_re = nm_idx(n, m), idx_im = nm_idx(n, -m);
            double fb_re = MA[0] * T[idx_re], fb_im = MA[0] * T[idx_im];
            double fa_even_re = MB[0] * T[idx_re], fa_even_im = MB[0] * T[idx_im];
            double fa_odd_re = 0.0, fa_odd_im = 0.0;
            for (int k = 2; k <= P - n; ++k) {
                const int nk = n + k;
                const int base_k = nm_idx(k, 0), base_nk = nm_idx(nk, 0);
                double sb_re = MA[base_k] * T[base_nk + m], sb_im = MA[base_k] * T[base_nk - m];
                double sa_re = MB[base_k] * T[base_nk + m], sa_im = MB[base_k] * T[base_nk - m];
                for (int l = 1; l <= k; ++l) {
                    const double tp_re = T[base_nk + m + l], tp_im = T[base_nk - (m + l)];
                    const int ml = m - l;
                    double tm_re, tm_im;
                    if (ml > 0) {
                        const double s = (l & 1) ? -1.0 : 1.0;
                        tm_re = s * T[base_nk + ml]; tm_im = s * T[base_nk - ml];
                    } else if (ml == 0) {
                        tm_re = ((l & 1) ? -1.0 : 1.0) * T[base_nk]; tm_im = 0.0;
                    } else {
                        const int aml = -ml;
                        const double s = (m & 1) ? -1.0 : 1.0;
                        tm_re = s * T[base_nk + aml]; tm_im = -s * T[base_nk - aml];
                    }
                    const double sum_re = tp_re + tm_re, diff_re = tp_re - tm_re;
                    const double sum_im = tp_im + tm_im, diff_im = tp_im - tm_im;
                    const double ma_re = MA[base_k + l], ma_im = MA[base_k - l];
                    const double mb_re = MB[base_k + l], mb_im = MB[base_k - l];
                    sb_re += ma_re * sum_re + ma_im * diff_im;
                    sb_im += ma_re * sum_im - ma_im * diff_re;
                    sa_re += mb_re * sum_re + mb_im * diff_im;
                    sa_im += mb_re * sum_im - mb_im * diff_re;
                }
                fb_re += sb_re; fb_im += sb_im;
                if (k & 1) { fa_odd_re += sa_re; fa_odd_im += sa_im; }
                else       { fa_even_re += sa_re; fa_even_im += sa_im; }
            }
            FB[idx_re] += fb_re; FB[idx_im] += fb_im;
            FA[idx_re] += odd_n ? (fa_odd_re - fa_even_re) : (fa_even_re - fa_odd_re);
            FA[idx_im] += odd_n ? (fa_odd_im - fa_even_im) : (fa_even_im - fa_odd_im);
        }
    }
}

template <int P>
inline void harm_l2l_shift(const Vec3& d_parent_minus_child, const double* __restrict__ F_parent, double* __restrict__ F_child) {
    double U[harm_total(P)];
    eval_U_real<P>(d_parent_minus_child, U);

    for (int n = 1; n <= P; ++n) {
        for (int m = 0; m <= n; ++m) {
            double sum_re = 0.0, sum_im = 0.0;
            for (int k = 0; k <= P - n; ++k) {
                const int nk = n + k;
                for (int l = -k; l <= k; ++l) {
                    double u_re, u_im, f_re, f_im;
                    get_complex_harm(U, k, l, u_re, u_im);
                    get_complex_harm(F_parent, nk, m + l, f_re, f_im);
                    sum_re += u_re * f_re + u_im * f_im;
                    sum_im += u_re * f_im - u_im * f_re;
                }
            }
            F_child[nm_idx(n, m)] += sum_re;
            if (m > 0) {
                F_child[nm_idx(n, -m)] += sum_im;
            }
        }
    }
}

template <int P>
inline void harm_l2p_eval(const Vec3& d_cell_minus_leaf, const double* __restrict__ F_cell, double& ax, double& ay, double& az) {
    double U[harm_total(P - 1)];
    eval_U_real<P - 1>(d_cell_minus_leaf, U);

    double psi10_re = F_cell[nm_idx(1, 0)];
    double psi11_re = F_cell[nm_idx(1, 1)];
    double psi11_im = F_cell[nm_idx(1, -1)];

    for (int k = 1; k <= P - 1; ++k) {
        const int nk = 1 + k;
        const int base_k = nm_idx(k, 0);
        const int base_nk = nm_idx(nk, 0);

        // l = 0 term:
        const double u0 = U[base_k];
        psi10_re += u0 * F_cell[base_nk];
        psi11_re += u0 * F_cell[base_nk + 1];
        psi11_im += u0 * F_cell[base_nk - 1];

        // Combine +l and -l for l = 1..k:
        for (int l = 1; l <= k; ++l) {
            const double u_re = U[base_k + l];
            const double u_im = U[base_k - l];
            // For psi10 (m = 0): +l and -l are conjugates, so 2 * (u_re * f_re + u_im * f_im)
            psi10_re += 2.0 * (u_re * F_cell[base_nk + l] + u_im * F_cell[base_nk - l]);

            // For psi11 (m = 1): +l has index 1+l > 0; -l has index 1-l
            const double fp_re = F_cell[base_nk + (1 + l)];
            const double fp_im = F_cell[base_nk - (1 + l)];
            const int ml = 1 - l;
            double fm_re, fm_im;
            if (ml == 0) {
                // l = 1: (-1)^1 F_{nk}^0 = -F_cell[base_nk]
                fm_re = -F_cell[base_nk];
                fm_im = 0.0;
            } else {
                // ml < 0: aml = l - 1 > 0, (-1)^{l + aml} = -1
                const int aml = l - 1;
                fm_re = -F_cell[base_nk + aml];
                fm_im =  F_cell[base_nk - aml];
            }
            psi11_re += u_re * (fp_re + fm_re) + u_im * (fp_im - fm_im);
            psi11_im += u_re * (fp_im + fm_im) - u_im * (fp_re - fm_re);
        }
    }
    ax -= psi11_re;
    ay -= psi11_im;
    az -= psi10_re;
}

// Compute residual radial derivatives D_k^res = mAmB * (2k-1)!! * [ (r^2 + eps^2)^{-(k+1/2)} - r^{-(2k+1)} ]
// using cancellation-free identity:
//   d0_eps - d0_0 = -d0_0 * eps^2 / ( (r^2 + eps^2) + r * sqrt(r^2 + eps^2) )
//   D_k^res = (2k-1) * [ inv_r2_eps * D_{k-1}^res - (eps^2 * inv_r2_eps) * D_{k-1}^0 * inv_r2_0 ]
template <int P_RES>
inline void set_dPhi_residual(const Vec3& R, double mAmB, double r2, double eps2, double* __restrict__ F_res) {
    const double r2_eps = r2 + eps2;
    const double inv_r2_0 = 1.0 / r2;
    const double inv_r0 = std::sqrt(inv_r2_0);
    const double inv_r2_e = 1.0 / r2_eps;
    const double inv_re = std::sqrt(inv_r2_e);

    double D0_newt = mAmB * inv_r0;
    double D_res[P_RES + 1];
    // Cancellation-free difference: 1/sqrt(r^2+eps^2) - 1/r = -eps^2 / (r * sqrt(r^2+eps^2) * (r + sqrt(r^2+eps^2)))
    D_res[0] = -D0_newt * (eps2 * inv_r2_e) / (1.0 + inv_r0 / inv_re);

    double Dk_newt = D0_newt;
    const double eps2_inv_r2e = eps2 * inv_r2_e;
    for (int k = 1; k <= P_RES; ++k) {
        const double fk = static_cast<double>(2 * k - 1);
        D_res[k] = fk * (inv_r2_e * D_res[k - 1] - eps2_inv_r2e * (Dk_newt * inv_r2_0));
        Dk_newt *= fk * inv_r2_0;
    }

    double G[symset_total(P_RES)];
    compute_outer_powers<P_RES>(R, G);

    for (int k = 0; k <= P_RES; ++k) {
        const double sdk = (k & 1) ? -D_res[k] : D_res[k];
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            F_res[off + i] = sdk * G[off + i];
        }
    }

    for (int j = 1; j <= P_RES / 2; ++j) {
        const double inv_j = COMB.inv_int[j];
        for (int n = P_RES; n >= 2 * j; --n) {
            const int k = n - 2 * j;
            const double sd = ((k + j) & 1) ? -D_res[k + j] : D_res[k + j];
            const int off_n = symset_offset(n);
            const int off_n2 = symset_offset(n - 2);
            double* __restrict__ Gn = G + off_n;
            double* __restrict__ Fn = F_res + off_n;
            const double* __restrict__ Gn2 = G + off_n2;

            const int sz_n = sym3d_size(n);
            for (int i = 0; i < sz_n; ++i) Gn[i] = 0.0;

            const int sz_n2 = sym3d_size(n - 2);
            for (int i = 0; i < sz_n2; ++i) {
                const double val = Gn2[i] * inv_j;
                const int idx2 = off_n2 + i;
                const int nx = META<P_RES>.nx_of[idx2];
                const int ny = META<P_RES>.ny_of[idx2];
                const int nz = META<P_RES>.nz_of[idx2];
                const int l = ny + nz, m = nz;
                Gn[lm_idx(l, m)]         += COMB.binom[nx + 2][2] * val;
                Gn[lm_idx(l + 2, m)]     += COMB.binom[ny + 2][2] * val;
                Gn[lm_idx(l + 2, m + 2)] += COMB.binom[nz + 2][2] * val;
            }
            for (int i = 0; i < sz_n; ++i) {
                Fn[i] += sd * Gn[i];
            }
        }
    }
}

// 4-way AVX2 evaluation of residual derivative tensors F_res for P_RES
template <int P_RES>
inline void approx_cc_residual_batch4_avx2(
    const Vec3 dX[4], const double mAmB[4], const double r2_0[4], double eps2,
    double* const CA[4], const double* const MB[4],
    double* const CB[4], const double* const MA[4]
) {
    constexpr int C_LEN = symset_total(P_RES);
    constexpr int M_LEN = (P_RES >= 3) ? (symset_total(P_RES - 1) - 4) : 1;

    const __m256d R_x = _mm256_set_pd(dX[3].x, dX[2].x, dX[1].x, dX[0].x);
    const __m256d R_y = _mm256_set_pd(dX[3].y, dX[2].y, dX[1].y, dX[0].y);
    const __m256d R_z = _mm256_set_pd(dX[3].z, dX[2].z, dX[1].z, dX[0].z);
    const __m256d vm  = _mm256_loadu_pd(mAmB);
    const __m256d vr2_0 = _mm256_loadu_pd(r2_0);
    const __m256d veps2 = _mm256_set1_pd(eps2);
    const __m256d vr2_e = _mm256_add_pd(vr2_0, veps2);

    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vinv_r2_0 = _mm256_div_pd(vone, vr2_0);
    const __m256d vinv_r0   = _mm256_sqrt_pd(vinv_r2_0);
    const __m256d vinv_r2_e = _mm256_div_pd(vone, vr2_e);
    const __m256d vinv_re   = _mm256_sqrt_pd(vinv_r2_e);

    __m256d D0_newt = _mm256_mul_pd(vm, vinv_r0);
    __m256d D_res[P_RES + 1];
    const __m256d veps2_inv_r2e = _mm256_mul_pd(veps2, vinv_r2_e);
    const __m256d vdenom = _mm256_add_pd(vone, _mm256_div_pd(vinv_r0, vinv_re));
    D_res[0] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_div_pd(_mm256_mul_pd(D0_newt, veps2_inv_r2e), vdenom));

    __m256d Dk_newt = D0_newt;
    for (int k = 1; k <= P_RES; ++k) {
        const __m256d vfk = _mm256_set1_pd(static_cast<double>(2 * k - 1));
        const __m256d term1 = _mm256_mul_pd(vinv_r2_e, D_res[k - 1]);
        const __m256d term2 = _mm256_mul_pd(veps2_inv_r2e, _mm256_mul_pd(Dk_newt, vinv_r2_0));
        D_res[k] = _mm256_mul_pd(vfk, _mm256_sub_pd(term1, term2));
        Dk_newt = _mm256_mul_pd(Dk_newt, _mm256_mul_pd(vfk, vinv_r2_0));
    }

    __m256d G[C_LEN];
    __m256d F[C_LEN];
    G[0] = vone;
    if constexpr (P_RES >= 1) {
        G[1] = R_x; G[2] = R_y; G[3] = R_z;
    }
    for (int k = 2; k <= P_RES; ++k) {
        const __m256d* prev = G + symset_offset(k - 1);
        __m256d* cur = G + symset_offset(k);
        const int k0 = sym3d_size(k - 1);
        const int k1 = sym3d_size(k - 2);
        for (int i = 0; i < k0; ++i) cur[i] = _mm256_mul_pd(prev[i], R_x);
        for (int i = 0; i < k; ++i)  cur[k0 + i] = _mm256_mul_pd(prev[k1 + i], R_y);
        cur[sym3d_size(k) - 1] = _mm256_mul_pd(prev[k0 - 1], R_z);
    }

    for (int k = 0; k <= P_RES; ++k) {
        __m256d sdk = (k & 1) ? _mm256_sub_pd(_mm256_setzero_pd(), D_res[k]) : D_res[k];
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            F[off + i] = _mm256_mul_pd(sdk, G[off + i]);
        }
    }

    for (int j = 1; j <= P_RES / 2; ++j) {
        const __m256d vinv_j = _mm256_set1_pd(COMB.inv_int[j]);
        for (int n = P_RES; n >= 2 * j; --n) {
            const int k = n - 2 * j;
            __m256d sd = ((k + j) & 1) ? _mm256_sub_pd(_mm256_setzero_pd(), D_res[k + j]) : D_res[k + j];
            const int off_n = symset_offset(n);
            const int off_n2 = symset_offset(n - 2);
            __m256d* Gn = G + off_n;
            __m256d* Fn = F + off_n;
            const __m256d* Gn2 = G + off_n2;

            const int sz_n = sym3d_size(n);
            for (int i = 0; i < sz_n; ++i) Gn[i] = _mm256_setzero_pd();

            const int sz_n2 = sym3d_size(n - 2);
            for (int i = 0; i < sz_n2; ++i) {
                __m256d val = _mm256_mul_pd(Gn2[i], vinv_j);
                const int idx2 = off_n2 + i;
                const int nx = META<P_RES>.nx_of[idx2];
                const int ny = META<P_RES>.ny_of[idx2];
                const int nz = META<P_RES>.nz_of[idx2];
                const int l = ny + nz, m = nz;
                Gn[lm_idx(l, m)]         = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[nx + 2][2]), val, Gn[lm_idx(l, m)]);
                Gn[lm_idx(l + 2, m)]     = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[ny + 2][2]), val, Gn[lm_idx(l + 2, m)]);
                Gn[lm_idx(l + 2, m + 2)] = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[nz + 2][2]), val, Gn[lm_idx(l + 2, m + 2)]);
            }
            for (int i = 0; i < sz_n; ++i) {
                Fn[i] = _mm256_fmadd_pd(sd, Gn[i], Fn[i]);
            }
        }
    }

    __m256d vMA[M_LEN];
    __m256d vMB[M_LEN];
    if constexpr (P_RES >= 3) {
        const double* ma0 = MA[0]; const double* ma1 = MA[1]; const double* ma2 = MA[2]; const double* ma3 = MA[3];
        const double* mb0 = MB[0]; const double* mb1 = MB[1]; const double* mb2 = MB[2]; const double* mb3 = MB[3];
        for (int m = 2; m <= P_RES - 1; ++m) {
            const int off = symset_offset(m) - 4;
            const int sz = sym3d_size(m);
            if (m & 1) {
                for (int i = 0; i < sz; ++i) {
                    const int idx = off + i;
                    vMA[idx] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_set_pd(ma3[idx], ma2[idx], ma1[idx], ma0[idx]));
                    vMB[idx] = _mm256_set_pd(mb3[idx], mb2[idx], mb1[idx], mb0[idx]);
                }
            } else {
                for (int i = 0; i < sz; ++i) {
                    const int idx = off + i;
                    vMA[idx] = _mm256_set_pd(ma3[idx], ma2[idx], ma1[idx], ma0[idx]);
                    vMB[idx] = _mm256_set_pd(mb3[idx], mb2[idx], mb1[idx], mb0[idx]);
                }
            }
        }
    }

    alignas(32) double tmpA[4], tmpB[4];

    if constexpr (P_RES >= 3) {
        __m256d ca0 = F[1], cb0 = ca0;
        __m256d ca1 = F[2], cb1 = ca1;
        __m256d ca2 = F[3], cb2 = ca2;
        if constexpr (P_RES >= 3) contract_k1_m2(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(3), vMB + (symset_offset(2) - 4), vMA + (symset_offset(2) - 4));
        if constexpr (P_RES >= 4) contract_k1_m3(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(4), vMB + (symset_offset(3) - 4), vMA + (symset_offset(3) - 4));
        if constexpr (P_RES >= 5) contract_k1_m4(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(5), vMB + (symset_offset(4) - 4), vMA + (symset_offset(4) - 4));
        if constexpr (P_RES >= 6) contract_k1_m5(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(6), vMB + (symset_offset(5) - 4), vMA + (symset_offset(5) - 4));
        if constexpr (P_RES >= 7) contract_k1_m6(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(7), vMB + (symset_offset(6) - 4), vMA + (symset_offset(6) - 4));
        if constexpr (P_RES >= 8) contract_k1_m7(ca0, cb0, ca1, cb1, ca2, cb2, F + symset_offset(8), vMB + (symset_offset(7) - 4), vMA + (symset_offset(7) - 4));
        const __m256d cas[3] = {ca0, ca1, ca2};
        const __m256d cbs[3] = {cb0, cb1, cb2};
        for (int bi = 0; bi < 3; ++bi) {
            const int idx = 1 + bi;
            _mm256_store_pd(tmpA, cas[bi]); _mm256_store_pd(tmpB, cbs[bi]);
            CA[0][idx] += tmpA[0]; CB[0][idx] -= tmpB[0];
            CA[1][idx] += tmpA[1]; CB[1][idx] -= tmpB[1];
            CA[2][idx] += tmpA[2]; CB[2][idx] -= tmpB[2];
            CA[3][idx] += tmpA[3]; CB[3][idx] -= tmpB[3];
        }
    }

    if constexpr (P_RES >= 4) {
        __m256d ca0 = F[4], cb0 = ca0;
        __m256d ca1 = F[5], cb1 = ca1;
        __m256d ca2 = F[6], cb2 = ca2;
        __m256d ca3 = F[7], cb3 = ca3;
        __m256d ca4 = F[8], cb4 = ca4;
        __m256d ca5 = F[9], cb5 = ca5;
        if constexpr (P_RES >= 4) contract_k2_m2(ca0, cb0, ca1, cb1, ca2, cb2, ca3, cb3, ca4, cb4, ca5, cb5, F + symset_offset(4), vMB + (symset_offset(2) - 4), vMA + (symset_offset(2) - 4));
        if constexpr (P_RES >= 5) contract_k2_m3(ca0, cb0, ca1, cb1, ca2, cb2, ca3, cb3, ca4, cb4, ca5, cb5, F + symset_offset(5), vMB + (symset_offset(3) - 4), vMA + (symset_offset(3) - 4));
        if constexpr (P_RES >= 6) contract_k2_m4(ca0, cb0, ca1, cb1, ca2, cb2, ca3, cb3, ca4, cb4, ca5, cb5, F + symset_offset(6), vMB + (symset_offset(4) - 4), vMA + (symset_offset(4) - 4));
        if constexpr (P_RES >= 7) contract_k2_m5(ca0, cb0, ca1, cb1, ca2, cb2, ca3, cb3, ca4, cb4, ca5, cb5, F + symset_offset(7), vMB + (symset_offset(5) - 4), vMA + (symset_offset(5) - 4));
        if constexpr (P_RES >= 8) contract_k2_m6(ca0, cb0, ca1, cb1, ca2, cb2, ca3, cb3, ca4, cb4, ca5, cb5, F + symset_offset(8), vMB + (symset_offset(6) - 4), vMA + (symset_offset(6) - 4));
        const __m256d cas[6] = {ca0, ca1, ca2, ca3, ca4, ca5};
        const __m256d cbs[6] = {cb0, cb1, cb2, cb3, cb4, cb5};
        for (int bi = 0; bi < 6; ++bi) {
            const int idx = 4 + bi;
            _mm256_store_pd(tmpA, cas[bi]); _mm256_store_pd(tmpB, cbs[bi]);
            CA[0][idx] += tmpA[0]; CB[0][idx] += tmpB[0];
            CA[1][idx] += tmpA[1]; CB[1][idx] += tmpB[1];
            CA[2][idx] += tmpA[2]; CB[2][idx] += tmpB[2];
            CA[3][idx] += tmpA[3]; CB[3][idx] += tmpB[3];
        }
    }

    __m256d vCA[C_LEN];
    __m256d vCB[C_LEN];
    const int k_start = (P_RES >= 4) ? 3 : ((P_RES >= 3) ? 2 : 1);
    for (int i = symset_offset(k_start); i < C_LEN; ++i) {
        vCA[i] = F[i];
        vCB[i] = F[i];
    }

    if constexpr (P_RES >= 5) {
        for (int k = 3; k <= P_RES - 2; ++k) {
            __m256d* ca_k = vCA + symset_offset(k);
            __m256d* cb_k = vCB + symset_offset(k);
            for (int m = 2; m <= P_RES - k; ++m) {
                const __m256d* Fn = F + symset_offset(k + m);
                const __m256d* mb = vMB + (symset_offset(m) - 4);
                const __m256d* ma = vMA + (symset_offset(m) - 4);
                switch (m) {
                    case 2: m2l_contract_m2(ca_k, cb_k, k, Fn, mb, ma); break;
                    case 3: if constexpr (P_RES > 5) m2l_contract_m3(ca_k, cb_k, k, Fn, mb, ma); break;
                    case 4: if constexpr (P_RES > 6) m2l_contract_m4(ca_k, cb_k, k, Fn, mb, ma); break;
                    case 5: if constexpr (P_RES > 7) m2l_contract_m5(ca_k, cb_k, k, Fn, mb, ma); break;
                }
            }
        }
    }

    for (int k = k_start; k <= P_RES; ++k) {
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        if (k & 1) {
            for (int i = 0; i < sz; ++i) {
                const int idx = off + i;
                _mm256_store_pd(tmpA, vCA[idx]);
                _mm256_store_pd(tmpB, vCB[idx]);
                CA[0][idx] += tmpA[0]; CB[0][idx] -= tmpB[0];
                CA[1][idx] += tmpA[1]; CB[1][idx] -= tmpB[1];
                CA[2][idx] += tmpA[2]; CB[2][idx] -= tmpB[2];
                CA[3][idx] += tmpA[3]; CB[3][idx] -= tmpB[3];
            }
        } else {
            for (int i = 0; i < sz; ++i) {
                const int idx = off + i;
                _mm256_store_pd(tmpA, vCA[idx]);
                _mm256_store_pd(tmpB, vCB[idx]);
                CA[0][idx] += tmpA[0]; CB[0][idx] += tmpB[0];
                CA[1][idx] += tmpA[1]; CB[1][idx] += tmpB[1];
                CA[2][idx] += tmpA[2]; CB[2][idx] += tmpB[2];
                CA[3][idx] += tmpA[3]; CB[3][idx] += tmpB[3];
            }
        }
    }
}

} // namespace fmm

#endif // FMM_HARMONIC_HPP
