// High-order Cartesian Fast Multipole Method (FMM) Kernels for Plummer-softened gravity
// Implements:
//   - W. Dehnen (2000), ApJ, 536, L39
//   - W. Dehnen (2002), J. Comput. Phys., 179, 27
//   - W. Dehnen (2014), Comput. Astrophys. Cosmol., 1:1 (Section 6.1.1, Section 7.2 & Appendix A.1)
// Symmetric 3D tensor layout adapted with attribution from the public falcON code
// (Copyright (C) 2000-2012 Walter Dehnen, GPL v2+).

#ifndef FMM_KERNELS_HPP
#define FMM_KERNELS_HPP

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fmm {

constexpr int P_MAX = 12;

struct Vec3 {
    double x, y, z;
};

// Number of independent elements in a 3D symmetric tensor of order K: (K+1)(K+2)/2
constexpr int sym3d_size(int K) {
    return ((K + 1) * (K + 2)) / 2;
}

// Offset of order-K tensor in a set of 3D symmetric tensors of orders 0..P: K(K+1)(K+2)/6
constexpr int symset_offset(int K) {
    return (K * (K + 1) * (K + 2)) / 6;
}

// Total elements in a set of 3D symmetric tensors of orders 0..P: (P+1)(P+2)(P+3)/6
constexpr int symset_total(int P) {
    return ((P + 1) * (P + 2) * (P + 3)) / 6;
}

// Index of (ny, nz) inside an order-k tensor, where l = ny + nz (0..k), m = nz (0..l), nx = k - l
constexpr int lm_idx(int l, int m) {
    return (l * (l + 1)) / 2 + m;
}

struct CombTables {
    double fact[P_MAX + 2];
    double inv_fact[P_MAX + 2];
    double inv_int[P_MAX + 2];
    double binom[P_MAX + 2][P_MAX + 2];
    double trinom[P_MAX + 2][P_MAX + 2][P_MAX + 2]; // k! / (nx! ny! nz!) indexed by [k][l][m]

    constexpr CombTables() : fact{}, inv_fact{}, inv_int{}, binom{}, trinom{} {
        fact[0] = 1.0;
        inv_fact[0] = 1.0;
        inv_int[0] = 1.0;
        for (int i = 1; i <= P_MAX + 1; ++i) {
            fact[i] = fact[i - 1] * i;
            inv_fact[i] = 1.0 / fact[i];
            inv_int[i] = 1.0 / static_cast<double>(i);
        }
        for (int n = 0; n <= P_MAX + 1; ++n) {
            binom[n][0] = 1.0;
            for (int k = 1; k <= n; ++k) {
                binom[n][k] = (k == n) ? 1.0 : binom[n - 1][k - 1] + binom[n - 1][k];
            }
        }
        for (int k = 0; k <= P_MAX + 1; ++k) {
            for (int l = 0; l <= k; ++l) {
                for (int m = 0; m <= l; ++m) {
                    trinom[k][l][m] = binom[k][l] * binom[l][m];
                }
            }
        }
    }
};

constexpr CombTables COMB{};

template <int P>
struct OrderMeta {
    static constexpr int TOTAL = symset_total(P);
    static constexpr int POLES_TOTAL = (P >= 3) ? (symset_total(P - 1) - 4) : 1;

    int nx_of[TOTAL];
    int ny_of[TOTAL];
    int nz_of[TOTAL];
    int idx_plus_x[TOTAL];
    int idx_plus_y[TOTAL];
    int idx_plus_z[TOTAL];
    double mult[TOTAL];

    constexpr OrderMeta()
        : nx_of{}, ny_of{}, nz_of{}, idx_plus_x{}, idx_plus_y{}, idx_plus_z{}, mult{} {
        for (int k = 0; k <= P; ++k) {
            const int off = symset_offset(k);
            for (int l = 0; l <= k; ++l) {
                for (int m = 0; m <= l; ++m) {
                    const int idx = off + lm_idx(l, m);
                    const int nx = k - l;
                    const int ny = l - m;
                    const int nz = m;
                    nx_of[idx] = nx;
                    ny_of[idx] = ny;
                    nz_of[idx] = nz;
                    if (k < P) {
                        const int next_off = symset_offset(k + 1);
                        idx_plus_x[idx] = next_off + lm_idx(l, m);
                        idx_plus_y[idx] = next_off + lm_idx(l + 1, m);
                        idx_plus_z[idx] = next_off + lm_idx(l + 1, m + 1);
                    }
                    mult[idx] = COMB.trinom[k][l][m];
                }
            }
        }
    }
};

template <int P>
constexpr OrderMeta<P> META{};

// Compute outer product powers of vector v = (vx, vy, vz) for orders k = 0..P
template <int P>
inline void compute_outer_powers(const Vec3& v, double* __restrict__ out) {
    out[0] = 1.0;
    if constexpr (P >= 1) {
        out[1] = v.x;
        out[2] = v.y;
        out[3] = v.z;
    }
    for (int k = 2; k <= P; ++k) {
        const double* __restrict__ prev = out + symset_offset(k - 1);
        double* __restrict__ cur = out + symset_offset(k);
        const int k0 = sym3d_size(k - 1);
        const int k1 = sym3d_size(k - 2);
        for (int i = 0; i < k0; ++i) cur[i] = prev[i] * v.x;
        for (int i = 0; i < k; ++i)  cur[k0 + i] = prev[k1 + i] * v.y;
        cur[sym3d_size(k) - 1] = prev[k0 - 1] * v.z;
    }
}

// P2M: Add point mass m at displacement v = x_i - z_cell to unnormalized multipoles M (orders 2..P-1)
// (Dehnen 2002 Eq. 5; falcON::poles3D::add_body)
template <int P>
inline void p2m_add_body(const Vec3& v, double m, double* __restrict__ M) {
    if constexpr (P < 3) return;
    double pow_v[symset_total(P - 1)];
    pow_v[0] = m;
    pow_v[1] = m * v.x;
    pow_v[2] = m * v.y;
    pow_v[3] = m * v.z;
    for (int k = 2; k <= P - 1; ++k) {
        const double* __restrict__ prev = pow_v + symset_offset(k - 1);
        double* __restrict__ cur = pow_v + symset_offset(k);
        double* __restrict__ dst = M + (symset_offset(k) - 4);
        const int k0 = sym3d_size(k - 1);
        const int k1 = sym3d_size(k - 2);
        const int sz = sym3d_size(k);
        for (int i = 0; i < k0; ++i) cur[i] = prev[i] * v.x;
        for (int i = 0; i < k; ++i)  cur[k0 + i] = prev[k1 + i] * v.y;
        cur[sz - 1] = prev[k0 - 1] * v.z;
        for (int i = 0; i < sz; ++i) dst[i] += cur[i];
    }
}

// M2M: Shift child multipoles M_child (unnormalized, orders 2..P-1) and mass m_child
// at v = z_child - z_parent into parent's unnormalized multipoles M_parent
// (Dehnen 2002 / Dehnen 2014 Eq. A.4d; falcON::poles3D::add_cell)
template <int P>
inline void m2m_add_cell(const Vec3& v, double m_child, const double* __restrict__ M_child, double* __restrict__ M_parent) {
    if constexpr (P < 3) return;
    p2m_add_body<P>(v, m_child, M_parent);

    constexpr int NPOLES = symset_total(P - 1) - 4;
    for (int i = 0; i < NPOLES; ++i) {
        M_parent[i] += M_child[i];
    }
    if constexpr (P >= 4) {
        double U_prev[symset_total(P - 1)];
        double U_curr[symset_total(P - 1)];
        for (int k = 2; k <= P - 1; ++k) {
            const int off = symset_offset(k);
            const int sz = sym3d_size(k);
            for (int i = 0; i < sz; ++i) {
                U_prev[off + i] = M_child[off - 4 + i];
            }
        }
        for (int j = 1; j <= P - 3; ++j) {
            const double inv_j = COMB.inv_int[j];
            const double vx = v.x * inv_j;
            const double vy = v.y * inv_j;
            const double vz = v.z * inv_j;
            for (int k = P - 1; k >= j + 2; --k) {
                const int off_k = symset_offset(k);
                const int sz_k = sym3d_size(k);
                double* __restrict__ uk = U_curr + off_k;
                for (int i = 0; i < sz_k; ++i) uk[i] = 0.0;

                const int off_prev = symset_offset(k - 1);
                const int sz_prev = sym3d_size(k - 1);
                const double* __restrict__ up = U_prev + off_prev;
                for (int i = 0; i < sz_prev; ++i) {
                    const double val = up[i];
                    const int idx = off_prev + i;
                    const int ix = META<P>.idx_plus_x[idx];
                    const int iy = META<P>.idx_plus_y[idx];
                    const int iz = META<P>.idx_plus_z[idx];
                    uk[ix - off_k] += META<P>.nx_of[ix] * vx * val;
                    uk[iy - off_k] += META<P>.ny_of[iy] * vy * val;
                    uk[iz - off_k] += META<P>.nz_of[iz] * vz * val;
                }
                double* __restrict__ dst = M_parent + (off_k - 4);
                for (int i = 0; i < sz_k; ++i) {
                    dst[i] += uk[i];
                    U_prev[off_k + i] = uk[i];
                }
            }
        }
    }
}

// Normalize multipoles of cell by dividing order-k tensor by (mass * k!)
template <int P>
inline void normalize_poles(double mass, double* __restrict__ M) {
    if constexpr (P < 3) return;
    if (mass <= 0.0) return;
    const double inv_m = 1.0 / mass;
    for (int k = 2; k <= P - 1; ++k) {
        const double scale = inv_m * COMB.inv_fact[k];
        double* __restrict__ mk = M + (symset_offset(k) - 4);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            mk[i] *= scale;
        }
    }
}

// Compute F^{(n)} = m_A * m_B * nabla^{(n)} g(R) for n = 0..P (Dehnen 2002 Eq. 6; falcON::set_dPhi)
template <int P>
inline void set_dPhi(const Vec3& R, double mAmB, double r2_eps2, double* __restrict__ F) {
    double D[P + 1];
    const double inv_r2 = 1.0 / r2_eps2;
    D[0] = mAmB * std::sqrt(inv_r2);
    for (int k = 1; k <= P; ++k) {
        D[k] = static_cast<double>(2 * k - 1) * inv_r2 * D[k - 1];
    }

    double G[symset_total(P)];
    compute_outer_powers<P>(R, G);

    for (int k = 0; k <= P; ++k) {
        const double sdk = (k & 1) ? -D[k] : D[k];
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            F[off + i] = sdk * G[off + i];
        }
    }

    for (int j = 1; j <= P / 2; ++j) {
        const double inv_j = COMB.inv_int[j];
        for (int n = P; n >= 2 * j; --n) {
            const int k = n - 2 * j;
            const double sd = ((k + j) & 1) ? -D[k + j] : D[k + j];
            const int off_n = symset_offset(n);
            const int off_n2 = symset_offset(n - 2);
            double* __restrict__ Gn = G + off_n;
            double* __restrict__ Fn = F + off_n;
            const double* __restrict__ Gn2 = G + off_n2;

            const int sz_n = sym3d_size(n);
            for (int i = 0; i < sz_n; ++i) Gn[i] = 0.0;

            const int sz_n2 = sym3d_size(n - 2);
            for (int i = 0; i < sz_n2; ++i) {
                const double val = Gn2[i] * inv_j;
                const int idx2 = off_n2 + i;
                const int nx = META<P>.nx_of[idx2];
                const int ny = META<P>.ny_of[idx2];
                const int nz = META<P>.nz_of[idx2];
                const int l = ny + nz;
                const int m = nz;
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

// Simultaneous mutual M2L contraction for a single pair of cells (A, B)
// (Dehnen 2000 Section 2.2; Dehnen 2002 Eq. 4 & Section 2.2)
template <int P>
inline void add_C_C2C_mutual(
    double* __restrict__ CA, const double* __restrict__ MB,
    double* __restrict__ CB, const double* __restrict__ MA,
    const double* __restrict__ F
) {
    for (int k = 0; k <= P; ++k) {
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        double* __restrict__ ca = CA + off;
        double* __restrict__ cb = CB + off;
        const double* __restrict__ fk = F + off;
        if (k & 1) {
            for (int i = 0; i < sz; ++i) {
                ca[i] += fk[i];
                cb[i] -= fk[i];
            }
        } else {
            for (int i = 0; i < sz; ++i) {
                ca[i] += fk[i];
                cb[i] += fk[i];
            }
        }
    }
    if constexpr (P < 3) return;

    for (int n = 2; n <= P; ++n) {
        const double* __restrict__ Fn = F + symset_offset(n);
        const double sign_n = (n & 1) ? -1.0 : 1.0;
        const int max_m = (n < P) ? n : (P - 1);
        for (int m = 2; m <= max_m; ++m) {
            const int k = n - m;
            double* __restrict__ ca = CA + symset_offset(k);
            double* __restrict__ cb = CB + symset_offset(k);
            const double* __restrict__ mb = MB + (symset_offset(m) - 4);
            const double* __restrict__ ma = MA + (symset_offset(m) - 4);

            for (int l1 = 0; l1 <= k; ++l1) {
                for (int l2 = 0; l2 <= m; ++l2) {
                    const double* __restrict__ mb_row = mb + lm_idx(l2, 0);
                    const double* __restrict__ ma_row = ma + lm_idx(l2, 0);
                    const double* __restrict__ fn_base = Fn + lm_idx(l1 + l2, 0);
                    double* __restrict__ ca_row = ca + lm_idx(l1, 0);
                    double* __restrict__ cb_row = cb + lm_idx(l1, 0);
                    for (int m1 = 0; m1 <= l1; ++m1) {
                        const double* __restrict__ f_row = fn_base + m1;
                        double sum_a = 0.0;
                        double sum_b = 0.0;
                        for (int m2 = 0; m2 <= l2; ++m2) {
                            const double fv = f_row[m2];
                            sum_a += fv * mb_row[m2];
                            sum_b += fv * ma_row[m2];
                        }
                        ca_row[m1] += sum_a;
                        cb_row[m1] += sign_n * sum_b;
                    }
                }
            }
        }
    }
}

// 4-way AVX2 batched mutual M2L kernel (Dehnen 2014 Section 6.1.1; falcON::GravKernBase::acc_block)
// Computes 4 mutual cell-cell interactions simultaneously using 256-bit AVX2 + FMA registers.
template <int P>
inline void approx_cc_batch4_avx2(
    const Vec3 dX[4], const double mAmB[4], const double r2_eps2[4],
    double* const CA[4], const double* const MB[4],
    double* const CB[4], const double* const MA[4]
) {
    constexpr int C_LEN = symset_total(P);
    constexpr int M_LEN = (P >= 3) ? (symset_total(P - 1) - 4) : 1;

    __m256d R_x = _mm256_set_pd(dX[3].x, dX[2].x, dX[1].x, dX[0].x);
    __m256d R_y = _mm256_set_pd(dX[3].y, dX[2].y, dX[1].y, dX[0].y);
    __m256d R_z = _mm256_set_pd(dX[3].z, dX[2].z, dX[1].z, dX[0].z);
    __m256d vm  = _mm256_loadu_pd(mAmB);
    __m256d vr2 = _mm256_loadu_pd(r2_eps2);

    // 1. Compute radial derivatives D[0..P] for 4 interactions
    __m256d D[P + 1];
    __m256d inv_r2 = _mm256_div_pd(_mm256_set1_pd(1.0), vr2);
    D[0] = _mm256_mul_pd(vm, _mm256_sqrt_pd(inv_r2));
    for (int k = 1; k <= P; ++k) {
        D[k] = _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(2 * k - 1)), _mm256_mul_pd(inv_r2, D[k - 1]));
    }

    // 2. Compute 4 derivative tensors F^{(0..P)} in AVX2 registers
    __m256d G[C_LEN];
    __m256d F[C_LEN];
    G[0] = _mm256_set1_pd(1.0);
    if constexpr (P >= 1) {
        G[1] = R_x;
        G[2] = R_y;
        G[3] = R_z;
    }
    for (int k = 2; k <= P; ++k) {
        const __m256d* prev = G + symset_offset(k - 1);
        __m256d* cur = G + symset_offset(k);
        const int k0 = sym3d_size(k - 1);
        const int k1 = sym3d_size(k - 2);
        for (int i = 0; i < k0; ++i) cur[i] = _mm256_mul_pd(prev[i], R_x);
        for (int i = 0; i < k; ++i)  cur[k0 + i] = _mm256_mul_pd(prev[k1 + i], R_y);
        cur[sym3d_size(k) - 1] = _mm256_mul_pd(prev[k0 - 1], R_z);
    }

    for (int k = 0; k <= P; ++k) {
        __m256d sdk = (k & 1) ? _mm256_sub_pd(_mm256_setzero_pd(), D[k]) : D[k];
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            F[off + i] = _mm256_mul_pd(sdk, G[off + i]);
        }
    }

    for (int j = 1; j <= P / 2; ++j) {
        const __m256d vinv_j = _mm256_set1_pd(COMB.inv_int[j]);
        for (int n = P; n >= 2 * j; --n) {
            const int k = n - 2 * j;
            __m256d sd = ((k + j) & 1) ? _mm256_sub_pd(_mm256_setzero_pd(), D[k + j]) : D[k + j];
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
                const int nx = META<P>.nx_of[idx2];
                const int ny = META<P>.ny_of[idx2];
                const int nz = META<P>.nz_of[idx2];
                const int l = ny + nz;
                const int m = nz;
                Gn[lm_idx(l, m)]         = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[nx + 2][2]), val, Gn[lm_idx(l, m)]);
                Gn[lm_idx(l + 2, m)]     = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[ny + 2][2]), val, Gn[lm_idx(l + 2, m)]);
                Gn[lm_idx(l + 2, m + 2)] = _mm256_fmadd_pd(_mm256_set1_pd(COMB.binom[nz + 2][2]), val, Gn[lm_idx(l + 2, m + 2)]);
            }
            for (int i = 0; i < sz_n; ++i) {
                Fn[i] = _mm256_fmadd_pd(sd, Gn[i], Fn[i]);
            }
        }
    }

    // 3. Gather 4 multipole arrays MA[0..3] and MB[0..3] into transposed __m256d buffers
    __m256d vMA[M_LEN];
    __m256d vMB[M_LEN];
    if constexpr (P >= 3) {
        const double* ma0 = MA[0]; const double* ma1 = MA[1]; const double* ma2 = MA[2]; const double* ma3 = MA[3];
        const double* mb0 = MB[0]; const double* mb1 = MB[1]; const double* mb2 = MB[2]; const double* mb3 = MB[3];
        for (int i = 0; i < M_LEN; ++i) {
            vMA[i] = _mm256_set_pd(ma3[i], ma2[i], ma1[i], ma0[i]);
            vMB[i] = _mm256_set_pd(mb3[i], mb2[i], mb1[i], mb0[i]);
        }
    }

    // 4. Simultaneous M2L contraction in AVX2 registers
    __m256d vCA[C_LEN];
    __m256d vCB[C_LEN];
    for (int k = 0; k <= P; ++k) {
        const int off = symset_offset(k);
        const int sz = sym3d_size(k);
        if (k & 1) {
            for (int i = 0; i < sz; ++i) {
                vCA[off + i] = F[off + i];
                vCB[off + i] = _mm256_sub_pd(_mm256_setzero_pd(), F[off + i]);
            }
        } else {
            for (int i = 0; i < sz; ++i) {
                vCA[off + i] = F[off + i];
                vCB[off + i] = F[off + i];
            }
        }
    }

    if constexpr (P >= 3) {
        for (int n = 2; n <= P; ++n) {
            const __m256d* Fn = F + symset_offset(n);
            const bool odd_n = (n & 1) != 0;
            const int max_m = (n < P) ? n : (P - 1);
            for (int m = 2; m <= max_m; ++m) {
                const int k = n - m;
                __m256d* ca = vCA + symset_offset(k);
                __m256d* cb = vCB + symset_offset(k);
                const __m256d* mb = vMB + (symset_offset(m) - 4);
                const __m256d* ma = vMA + (symset_offset(m) - 4);

                for (int l1 = 0; l1 <= k; ++l1) {
                    __m256d* ca_row = ca + lm_idx(l1, 0);
                    __m256d* cb_row = cb + lm_idx(l1, 0);
                    for (int m1 = 0; m1 <= l1; ++m1) {
                        __m256d sum_a = _mm256_setzero_pd();
                        __m256d sum_b = _mm256_setzero_pd();
                        for (int l2 = 0; l2 <= m; ++l2) {
                            const __m256d* mb_row = mb + lm_idx(l2, 0);
                            const __m256d* ma_row = ma + lm_idx(l2, 0);
                            const __m256d* f_row = Fn + lm_idx(l1 + l2, m1);
                            for (int m2 = 0; m2 <= l2; ++m2) {
                                const __m256d fv = f_row[m2];
                                sum_a = _mm256_fmadd_pd(fv, mb_row[m2], sum_a);
                                sum_b = _mm256_fmadd_pd(fv, ma_row[m2], sum_b);
                            }
                        }
                        ca_row[m1] = _mm256_add_pd(ca_row[m1], sum_a);
                        if (odd_n) {
                            cb_row[m1] = _mm256_sub_pd(cb_row[m1], sum_b);
                        } else {
                            cb_row[m1] = _mm256_add_pd(cb_row[m1], sum_b);
                        }
                    }
                }
            }
        }
    }

    // 5. Scatter-add vCA and vCB back to CA[0..3] and CB[0..3]
    alignas(32) double tmpA[4], tmpB[4];
    for (int i = 0; i < C_LEN; ++i) {
        _mm256_store_pd(tmpA, vCA[i]);
        _mm256_store_pd(tmpB, vCB[i]);
        CA[0][i] += tmpA[0]; CB[0][i] += tmpB[0];
        CA[1][i] += tmpA[1]; CB[1][i] += tmpB[1];
        CA[2][i] += tmpA[2]; CB[2][i] += tmpB[2];
        CA[3][i] += tmpA[3]; CB[3][i] += tmpB[3];
    }
}

// Flip sign of odd-order tensors in F (orders 0..P) (Dehnen 2002 / falcON::symset3D::flip_sign_odd)
template <int P>
inline void flip_sign_odd(double* __restrict__ F) {
    for (int k = 1; k <= P; k += 2) {
        double* __restrict__ Fk = F + symset_offset(k);
        const int sz = sym3d_size(k);
        for (int i = 0; i < sz; ++i) {
            Fk[i] = -Fk[i];
        }
    }
}

// Cell-to-body contraction (Dehnen 2002 / falcON::add_C_C2B):
// Accumulates into leaf's (pot, ax, ay, az) = C_leaf[0..3]
template <int P>
inline void add_C_C2B(double* __restrict__ C_leaf, const double* __restrict__ F, const double* __restrict__ M_scaled) {
    double c0 = F[0];
    double c1 = F[1];
    double c2 = F[2];
    double c3 = F[3];
    if constexpr (P >= 3) {
        for (int m = 2; m <= P - 1; ++m) {
            const double* __restrict__ Mm = M_scaled + (symset_offset(m) - 4);
            const double* __restrict__ Fm = F + symset_offset(m);
            const int sz_m = sym3d_size(m);
            for (int i = 0; i < sz_m; ++i) {
                c0 += Fm[i] * Mm[i];
            }
            const double* __restrict__ Fm1 = F + symset_offset(m + 1);
            for (int l2 = 0; l2 <= m; ++l2) {
                const double* __restrict__ m_row = Mm + lm_idx(l2, 0);
                const double* __restrict__ fx_row = Fm1 + lm_idx(l2, 0);
                const double* __restrict__ fy_row = Fm1 + lm_idx(l2 + 1, 0);
                const double* __restrict__ fz_row = Fm1 + lm_idx(l2 + 1, 1);
                for (int m2 = 0; m2 <= l2; ++m2) {
                    const double mv = m_row[m2];
                    c1 += fx_row[m2] * mv;
                    c2 += fy_row[m2] * mv;
                    c3 += fz_row[m2] * mv;
                }
            }
        }
    }
    C_leaf[0] -= c0;
    C_leaf[1] += c1;
    C_leaf[2] += c2;
    C_leaf[3] += c3;
}

// Body-to-cell accumulation (Dehnen 2002 / falcON::add_C_B2C):
template <int P>
inline void add_C_B2C(double* __restrict__ C, const double* __restrict__ F) {
    constexpr int TOTAL = symset_total(P);
    for (int i = 0; i < TOTAL; ++i) {
        C[i] += F[i];
    }
}

// Complete inner product of every tensor in symset R (orders 1..K) with vector v,
// storing result in L (orders 0..K-1) (Dehnen 2002 / falcON::symset3D::ass_inn_prd)
inline void symset_inner_prod_vec(int K, const double* __restrict__ R, const Vec3& v, double* __restrict__ L) {
    for (int k = 0; k < K; ++k) {
        double* __restrict__ Lk = L + symset_offset(k);
        const double* __restrict__ Rk1 = R + symset_offset(k + 1);
        for (int l1 = 0; l1 <= k; ++l1) {
            const double* __restrict__ rx = Rk1 + lm_idx(l1, 0);
            const double* __restrict__ ry = Rk1 + lm_idx(l1 + 1, 0);
            const double* __restrict__ rz = Rk1 + lm_idx(l1 + 1, 1);
            double* __restrict__ lk = Lk + lm_idx(l1, 0);
            for (int m1 = 0; m1 <= l1; ++m1) {
                lk[m1] = v.x * rx[m1] + v.y * ry[m1] + v.z * rz[m1];
            }
        }
    }
}

// L2L: Shift local Taylor series coefficients C (orders 0..P) by displacement dX = X_child - X_parent
// (Dehnen 2002 Eq. 7; falcON::shift_by in tensor_set.cc)
template <int P>
inline void l2l_shift_by(double* __restrict__ C, const Vec3& dX) {
    double o[symset_total(P - 1)];
    symset_inner_prod_vec(P, C, dX, o);
    for (int K = P - 1, I = 1; K >= 2; --K, ++I) {
        const int cd = symset_total(K);
        for (int j = 0; j < cd; ++j) {
            C[j] += o[j];
        }
        const double inv = COMB.inv_int[I + 1];
        const Vec3 v_scaled{dX.x * inv, dX.y * inv, dX.z * inv};
        symset_inner_prod_vec(K, o, v_scaled, o);
    }
    C[0] += o[0] + COMB.inv_int[P] * (dX.x * o[1] + dX.y * o[2] + dX.z * o[3]);
    C[1] += o[1];
    C[2] += o[2];
    C[3] += o[3];
}

// L2P: Evaluate local Taylor series C (orders 0..P) at displacement dX = X_leaf - X_cell,
// accumulating into leaf's C_leaf[0..3] = (pot, ax, ay, az)
// (Dehnen 2002 Eq. 3; falcON::eval_expn in tensor_set.cc)
template <int P>
inline void l2p_eval_expn(double* __restrict__ C_leaf, const double* __restrict__ C, const Vec3& dX) {
    C_leaf[0] -= C[0];
    C_leaf[1] += C[1];
    C_leaf[2] += C[2];
    C_leaf[3] += C[3];
    double o[symset_total(P - 1)];
    symset_inner_prod_vec(P, C, dX, o);
    for (int K = P - 1, I = 1; K >= 2; --K, ++I) {
        C_leaf[0] -= o[0];
        C_leaf[1] += o[1];
        C_leaf[2] += o[2];
        C_leaf[3] += o[3];
        const double inv = COMB.inv_int[I + 1];
        const Vec3 v_scaled{dX.x * inv, dX.y * inv, dX.z * inv};
        symset_inner_prod_vec(K, o, v_scaled, o);
    }
    C_leaf[0] -= o[0] + COMB.inv_int[P] * (dX.x * o[1] + dX.y * o[2] + dX.z * o[3]);
    C_leaf[1] += o[1];
    C_leaf[2] += o[2];
    C_leaf[3] += o[3];
}

} // namespace fmm

#endif // FMM_KERNELS_HPP
