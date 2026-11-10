#include "fmm_unrolled.hpp"
#include "fmm_harmonic.hpp"
#include <cstdio>
#include <ctime>
#include <cmath>

using namespace fmm;

template <int P>
void bench_cart(int niters) {
    Vec3 dX[4] = {{2.3, -1.5, 3.0}, {1.8, 2.2, -1.1}, {-2.5, 1.3, 2.0}, {3.1, -0.8, -1.9}};
    double mAmB[4] = {1.0, 1.0, 1.0, 1.0};
    double r2[4];
    for (int k = 0; k < 4; ++k) r2[k] = dX[k].x * dX[k].x + dX[k].y * dX[k].y + dX[k].z * dX[k].z + 0.0001;

    constexpr int C_LEN = symset_total(P);
    constexpr int M_LEN = symset_total(P - 1) - 4;

    double C1_ref[4][C_LEN]{}, C2_ref[4][C_LEN]{};
    double C1_got[4][C_LEN]{}, C2_got[4][C_LEN]{};
    double M1[4][M_LEN]{}, M2[4][M_LEN]{};
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < M_LEN; ++i) {
            M1[k][i] = 0.01 * ((i % 7) + 1);
            M2[k][i] = -0.005 * ((i % 5) + 1);
        }
    }
    double* CA_ref[4] = {C1_ref[0], C1_ref[1], C1_ref[2], C1_ref[3]};
    double* CB_ref[4] = {C2_ref[0], C2_ref[1], C2_ref[2], C2_ref[3]};
    double* CA_got[4] = {C1_got[0], C1_got[1], C1_got[2], C1_got[3]};
    double* CB_got[4] = {C2_got[0], C2_got[1], C2_got[2], C2_got[3]};
    const double* MA[4] = {M1[0], M1[1], M1[2], M1[3]};
    const double* MB[4] = {M2[0], M2[1], M2[2], M2[3]};

    // 1. Scalar loop-based mutual M2L
    clock_t t0 = std::clock();
    double F[C_LEN];
    for (int it = 0; it < niters; ++it) {
        int k = it & 3;
        dX[k].x += 1e-12;
        set_dPhi<P>(dX[k], mAmB[k], r2[k], F);
        add_C_C2C_mutual<P>(CA_ref[k], MB[k], CB_ref[k], MA[k], F);
    }
    clock_t t1 = std::clock();
    double ns_scalar = double(t1 - t0) / CLOCKS_PER_SEC * 1e9 / niters;

    // 2. Baseline loop-based 4-way AVX2 batch M2L
    clock_t t2 = std::clock();
    for (int it = 0; it < niters / 4; ++it) {
        dX[0].x += 1e-12;
        approx_cc_batch4_avx2<P>(dX, mAmB, r2, CA_ref, MB, CB_ref, MA);
    }
    clock_t t3 = std::clock();
    double ns_batch_loop = double(t3 - t2) / CLOCKS_PER_SEC * 1e9 / niters;

    // 3. Variant A code-generated unrolled 4-way AVX2 batch M2L
    clock_t t4 = std::clock();
    for (int it = 0; it < niters / 4; ++it) {
        dX[0].x += 1e-12;
        approx_cc_batch4_unrolled<P>(dX, mAmB, r2, CA_got, MB, CB_got, MA);
    }
    clock_t t5 = std::clock();
    double ns_batch_unrolled = double(t5 - t4) / CLOCKS_PER_SEC * 1e9 / niters;

    std::printf("Cartesian M2L P=%2d: scalar_loop=%8.1f ns | baseline_batch4_loop=%7.1f ns | variantA_unrolled=%7.1f ns (speedup vs scalar: %.2fx, vs baseline_loop: %.2fx)\n",
                P, ns_scalar, ns_batch_loop, ns_batch_unrolled,
                ns_scalar / ns_batch_unrolled, ns_batch_loop / ns_batch_unrolled);
}

template <int P_HARM, int P_RES>
void bench_split(int niters) {
    Vec3 r_AB[4] = {{2.3, -1.5, 3.0}, {1.8, 2.2, -1.1}, {-2.5, 1.3, 2.0}, {3.1, -0.8, -1.9}};
    double mAmB[4] = {1.0, 1.0, 1.0, 1.0};
    double r2[4];
    for (int k = 0; k < 4; ++k) r2[k] = r_AB[k].x * r_AB[k].x + r_AB[k].y * r_AB[k].y + r_AB[k].z * r_AB[k].z;

    constexpr int H_LEN = harm_total(P_HARM);
    constexpr int C_LEN = symset_total(P_RES);
    constexpr int M_LEN = symset_total(P_RES - 1) - 4;

    double FA[4][H_LEN]{}, FB[4][H_LEN]{}, MA_H[4][H_LEN]{}, MB_H[4][H_LEN]{};
    double CA[4][C_LEN]{}, CB[4][C_LEN]{}, MA_C[4][M_LEN]{}, MB_C[4][M_LEN]{};
    double* pFA[4] = {FA[0], FA[1], FA[2], FA[3]};
    double* pFB[4] = {FB[0], FB[1], FB[2], FB[3]};
    const double* pMAH[4] = {MA_H[0], MA_H[1], MA_H[2], MA_H[3]};
    const double* pMBH[4] = {MB_H[0], MB_H[1], MB_H[2], MB_H[3]};
    double* pCA[4] = {CA[0], CA[1], CA[2], CA[3]};
    double* pCB[4] = {CB[0], CB[1], CB[2], CB[3]};
    const double* pMAC[4] = {MA_C[0], MA_C[1], MA_C[2], MA_C[3]};
    const double* pMBC[4] = {MB_C[0], MB_C[1], MB_C[2], MB_C[3]};

    clock_t t0 = std::clock();
    for (int it = 0; it < niters / 4; ++it) {
        r_AB[0].x += 1e-12;
        harm_m2l_batch4_avx2<P_HARM>(r_AB, pFA, pMBH, pFB, pMAH);
        approx_cc_residual_batch4_avx2<P_RES>(r_AB, mAmB, r2, 0.0001, pCA, pMBC, pCB, pMAC);
    }
    clock_t t1 = std::clock();
    double ns_split = double(t1 - t0) / CLOCKS_PER_SEC * 1e9 / niters;
    std::printf("Split Harmonic+Res M2L (P_harm=%2d, P_res=%2d): %7.1f ns / mutual M2L\n", P_HARM, P_RES, ns_split);
}

int main() {
    bench_cart<8>(200000);
    bench_cart<10>(100000);
    bench_cart<12>(40000);
    bench_split<8, 6>(200000);
    bench_split<10, 6>(100000);
    bench_split<12, 6>(100000);
    bench_split<14, 6>(80000);
    bench_split<16, 6>(40000);
    return 0;
}
