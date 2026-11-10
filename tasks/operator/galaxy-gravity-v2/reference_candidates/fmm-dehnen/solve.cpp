// High-order Cartesian Fast Multipole Method (FMM) for Plummer-softened gravity
// Implements:
//   - W. Dehnen (2000), ApJ, 536, L39: "A Very Fast and Momentum-Conserving Tree Code"
//   - W. Dehnen (2002), J. Comput. Phys., 179, 27: "A Hierarchical O(N) Force Calculation Algorithm"
//   - W. Dehnen (2014), Comput. Astrophys. Cosmol., 1:1: "A fast multipole method for stellar dynamics"
//     (Section 7.2: softened gravity via high-order Cartesian FMM; Section 4-5: multipole-power error estimator)
//   - Tree structure, cell geometry, MAC table inversion, and symmetric 3D tensor conventions adapted
//     with attribution from the public falcON code (Copyright (C) 2000-2012 Walter Dehnen, GPL v2+).

#include "fmm_kernels.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace fmm {

enum class MacType : int {
    ConstTheta = 0, // Dehnen (2000) Eq. (2): r_crit = r_max / theta
    ThetaOfM   = 1, // Dehnen (2002) Eq. (12): theta(M)^(p+2)/(1-theta(M))^2 = (M/M_tot)^(-1/3) * theta_0^(p+2)/(1-theta_0)^2
    Dehnen2014 = 2  // Dehnen (2014) Eq. (12)-(14): multipole-power error estimator E_{A->B}
};

struct Config {
    int order = 8;
    MacType mac = MacType::ConstTheta;
    double theta = 0.48;
    double eps = 0.01;
    int ncrit = 96;
    int n_cb_pre = 16;
    int n_cb_post = 256;
    int n_cc_post = 96;
    int n_cs = 256;
    bool verbose = false;
};

// Table-based inversion of z^2 (1 + z)^p = y^{1/a} for z = 1/theta - 1
// Directly follows falcON::InvertZ (Dehnen 2002 Eq. 12; gravity.cc lines 57-100)
class InvertZ {
    static constexpr int N = 1000;
    static constexpr int N1 = N - 1;
    int P;
    double A, hA, sA;
    double Z[N];
    double Y[N];

    // Polynomial/linear interpolation in table (falcON::Polev)
    double interp(double y) const {
        int lo = 0, hi = N1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (Y[mid] <= y) lo = mid;
            else hi = mid;
        }
        double t = (y - Y[lo]) / (Y[hi] - Y[lo]);
        return Z[lo] + t * (Z[hi] - Z[lo]);
    }

    double z(double y) const {
        if (y <= Y[0]) return std::pow(y, hA);
        if (y >= Y[N1]) return std::pow(y, sA);
        return interp(y);
    }

public:
    InvertZ(double a, int p) : P(p), A(a), hA(0.5 * a), sA(a / (p + 2.0)) {
        const double iA = 1.0 / A;
        const double zmin = 1.0e-4;
        const double zmax = 1.0e4;
        const double lmin = std::log(zmin);
        const double dlz = (std::log(zmax) - lmin) / static_cast<double>(N1);
        for (int i = 0; i < N; ++i) {
            double zv = std::exp(lmin + i * dlz);
            Z[i] = zv;
            Y[i] = std::pow(zv * zv * std::pow(1.0 + zv, P), iA);
        }
    }

    double invtheta(double y) const {
        return 1.0 + z(y);
    }
};

struct Cell {
    Vec3 centre;        // Geometric center of cubic octree cell
    Vec3 cofm;          // Center of mass z_A
    double radius;      // Half-side length of cubic cell
    double mass;        // Cell mass M_A
    double rmax;        // Bounding sphere radius around cofm (Dehnen 2000 Eq. 3)
    double rcrit;       // Critical opening radius rmax / theta_A
    double rcrit2;      // rcrit^2
    uint32_t fcleaf;    // Start index in leaf array
    uint32_t nleafs;    // Number of direct leaf children (non-zero for twig cells)
    uint32_t number;    // Total number of leaf descendants
    uint32_t fccell;    // Start index of child cells
    uint16_t ncells;    // Number of child cells (0..8)
    uint8_t level;      // Tree level
    uint8_t has_coeffs; // Whether local field tensor C has been accumulated
};

// Build octree and reorder particles into contiguous tree-leaf order
// Follows Dehnen (2002) Section 3.1 (falcON::OctTree / BoxDotTree)
struct OctTree {
    int N = 0;
    int ncrit = 16;
    // Leaf arrays in tree order (SoA for AVX2 direct summation, plus AoS for fast single leaf access)
    std::vector<double> lx, ly, lz, lm;
    std::vector<double> lpot, lax, lay, laz;
    std::vector<uint32_t> orig_idx;
    std::vector<Cell> cells;

    struct BuildNode {
        Vec3 centre;
        double radius;
        uint32_t start;
        uint32_t count;
        uint8_t level;
    };

    void build(int n_part, const double* raw_mxyz, int nc) {
        N = n_part;
        ncrit = std::max(1, nc);
        lx.resize(N);
        ly.resize(N);
        lz.resize(N);
        lm.resize(N);
        lpot.assign(N, 0.0);
        lax.assign(N, 0.0);
        lay.assign(N, 0.0);
        laz.assign(N, 0.0);
        orig_idx.resize(N);

        Vec3 xave{0.0, 0.0, 0.0};
        Vec3 xmin{raw_mxyz[1], raw_mxyz[2], raw_mxyz[3]};
        Vec3 xmax = xmin;
        for (int i = 0; i < N; ++i) {
            const double m = raw_mxyz[4 * i + 0];
            const double x = raw_mxyz[4 * i + 1];
            const double y = raw_mxyz[4 * i + 2];
            const double z = raw_mxyz[4 * i + 3];
            lm[i] = m;
            lx[i] = x;
            ly[i] = y;
            lz[i] = z;
            orig_idx[i] = static_cast<uint32_t>(i);
            xave.x += x;
            xave.y += y;
            xave.z += z;
            if (x < xmin.x) xmin.x = x;
            if (x > xmax.x) xmax.x = x;
            if (y < xmin.y) xmin.y = y;
            if (y > xmax.y) xmax.y = y;
            if (z < xmin.z) xmin.z = z;
            if (z > xmax.z) xmax.z = z;
        }
        xave.x /= N;
        xave.y /= N;
        xave.z /= N;

        // Root center rounded to nearest integer as in falcON::TreeBuilder::root_centre (tree.cc line 160)
        Vec3 root_c{
            std::floor(xave.x + 0.5),
            std::floor(xave.y + 0.5),
            std::floor(xave.z + 0.5)
        };
        double max_d = 0.0;
        max_d = std::max(max_d, std::max(std::abs(xmax.x - root_c.x), std::abs(xmin.x - root_c.x)));
        max_d = std::max(max_d, std::max(std::abs(xmax.y - root_c.y), std::abs(xmin.y - root_c.y)));
        max_d = std::max(max_d, std::max(std::abs(xmax.z - root_c.z), std::abs(xmin.z - root_c.z)));
        // Power-of-two root half-size R = 2^(1 + floor(log2(max_d))) (tree.cc line 998)
        double root_r = (max_d > 0.0) ? std::ldexp(1.0, 1 + static_cast<int>(std::floor(std::log2(max_d)))) : 1.0;

        cells.clear();
        cells.reserve(N / 2 + 64);

        // Temporary buffers for in-place octant partitioning
        std::vector<double> tx(N), ty(N), tz(N), tm(N);
        std::vector<uint32_t> tidx(N);
        std::vector<uint8_t> oct_buf(N);

        Cell root{};
        root.centre = root_c;
        root.radius = root_r;
        root.fcleaf = 0;
        root.nleafs = 0;
        root.number = static_cast<uint32_t>(N);
        root.fccell = 0;
        root.ncells = 0;
        root.level = 0;
        root.has_coeffs = 0;
        cells.push_back(root);

        // Breadth-first / pre-order recursive subdivision
        auto split_cell = [&](auto& self, uint32_t c_idx) -> void {
            const uint32_t start = cells[c_idx].fcleaf;
            const uint32_t count = cells[c_idx].number;
            const uint8_t lev = cells[c_idx].level;
            if (count <= static_cast<uint32_t>(ncrit) || lev >= 50) {
                cells[c_idx].nleafs = count;
                cells[c_idx].ncells = 0;
                return;
            }
            const Vec3 cen = cells[c_idx].centre;
            const double hrad = 0.5 * cells[c_idx].radius;

            uint32_t counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t p = start + i;
                uint8_t o = 0;
                if (lx[p] > cen.x) o |= 1;
                if (ly[p] > cen.y) o |= 2;
                if (lz[p] > cen.z) o |= 4;
                oct_buf[i] = o;
                counts[o]++;
            }

            // Count non-empty octants and separate single-leaf octants as in falcON::BoxDotTree
            // (tree.cc lines 640-720: octants with 1 body are direct leaf children of the branch cell;
            //  octants with >1 bodies become daughter cells).
            int non_empty = 0;
            int single_oct = -1;
            int num_subboxes = 0;
            int num_direct_leafs = 0;
            for (int o = 0; o < 8; ++o) {
                if (counts[o] > 0) {
                    non_empty++;
                    single_oct = o;
                    if (counts[o] == 1) num_direct_leafs++;
                    else num_subboxes++;
                }
            }
            if (non_empty == 1) {
                // Shrink cell to the single populated octant and retry (falcON split_box do-while(ne==1))
                cells[c_idx].centre.x += (single_oct & 1) ? hrad : -hrad;
                cells[c_idx].centre.y += (single_oct & 2) ? hrad : -hrad;
                cells[c_idx].centre.z += (single_oct & 4) ? hrad : -hrad;
                cells[c_idx].radius = hrad;
                cells[c_idx].level = lev + 1;
                self(self, c_idx);
                return;
            }

            // Place direct leaf children (from octants with count == 1) first at [start, start + num_direct_leafs),
            // followed by octants with count > 1 (which become child cells).
            // This matches falcON::BoxDotTree::link_cells_N (tree.cc lines 909-960) where fcleaf..fcleaf+nleafs
            // are the direct leaf children of the cell, and fcleaf..fcleaf+number are all leaf descendants!
            uint32_t offsets[8];
            uint32_t cur_direct = 0;
            uint32_t cur_box = static_cast<uint32_t>(num_direct_leafs);
            for (int o = 0; o < 8; ++o) {
                if (counts[o] == 1) {
                    offsets[o] = cur_direct++;
                } else if (counts[o] > 1) {
                    offsets[o] = cur_box;
                    cur_box += counts[o];
                } else {
                    offsets[o] = 0;
                }
            }
            for (uint32_t i = 0; i < count; ++i) {
                const uint8_t o = oct_buf[i];
                const uint32_t dst = offsets[o]++;
                const uint32_t src = start + i;
                tx[dst] = lx[src];
                ty[dst] = ly[src];
                tz[dst] = lz[src];
                tm[dst] = lm[src];
                tidx[dst] = orig_idx[src];
            }
            std::memcpy(lx.data() + start, tx.data(), count * sizeof(double));
            std::memcpy(ly.data() + start, ty.data(), count * sizeof(double));
            std::memcpy(lz.data() + start, tz.data(), count * sizeof(double));
            std::memcpy(lm.data() + start, tm.data(), count * sizeof(double));
            std::memcpy(orig_idx.data() + start, tidx.data(), count * sizeof(uint32_t));

            const uint32_t first_child = static_cast<uint32_t>(cells.size());
            cells[c_idx].fccell = (num_subboxes > 0) ? first_child : 0;
            cells[c_idx].ncells = static_cast<uint16_t>(num_subboxes);
            cells[c_idx].nleafs = static_cast<uint32_t>(num_direct_leafs);

            uint32_t sub_start = start + static_cast<uint32_t>(num_direct_leafs);
            for (int o = 0; o < 8; ++o) {
                if (counts[o] <= 1) continue;
                Cell child{};
                child.centre.x = cen.x + ((o & 1) ? hrad : -hrad);
                child.centre.y = cen.y + ((o & 2) ? hrad : -hrad);
                child.centre.z = cen.z + ((o & 4) ? hrad : -hrad);
                child.radius = hrad;
                child.fcleaf = sub_start;
                child.nleafs = 0;
                child.number = counts[o];
                child.fccell = 0;
                child.ncells = 0;
                child.level = lev + 1;
                child.has_coeffs = 0;
                cells.push_back(child);
                sub_start += counts[o];
            }

            for (uint32_t k = 0; k < static_cast<uint32_t>(num_subboxes); ++k) {
                self(self, first_child + k);
            }
        };

        split_cell(split_cell, 0);
    }
};

// AVX2 + FMA Direct Summation Kernels (P2P)
// Computes mutual gravitational interactions between leaf ranges [a0, a0+na) and [b0, b0+nb)
inline void p2p_self_avx2(
    uint32_t a0, uint32_t na, double eps2,
    const double* __restrict__ lx, const double* __restrict__ ly, const double* __restrict__ lz,
    const double* __restrict__ lm,
    double* __restrict__ lax, double* __restrict__ lay, double* __restrict__ laz
) {
    const uint32_t an = a0 + na;
    const __m256d veps2 = _mm256_set1_pd(eps2);
    for (uint32_t i = a0; i + 1 < an; ++i) {
        const double xi = lx[i], yi = ly[i], zi = lz[i], mi = lm[i];
        const __m256d vxi = _mm256_set1_pd(xi);
        const __m256d vyi = _mm256_set1_pd(yi);
        const __m256d vzi = _mm256_set1_pd(zi);
        const __m256d vmi = _mm256_set1_pd(mi);

        __m256d vaxi = _mm256_setzero_pd();
        __m256d vayi = _mm256_setzero_pd();
        __m256d vazi = _mm256_setzero_pd();

        uint32_t j = i + 1;
        for (; j + 4 <= an; j += 4) {
            __m256d vxj = _mm256_loadu_pd(lx + j);
            __m256d vyj = _mm256_loadu_pd(ly + j);
            __m256d vzj = _mm256_loadu_pd(lz + j);
            __m256d vmj = _mm256_loadu_pd(lm + j);

            __m256d dx = _mm256_sub_pd(vxi, vxj);
            __m256d dy = _mm256_sub_pd(vyi, vyj);
            __m256d dz = _mm256_sub_pd(vzi, vzj);

            __m256d r2 = _mm256_fmadd_pd(dx, dx, veps2);
            r2 = _mm256_fmadd_pd(dy, dy, r2);
            r2 = _mm256_fmadd_pd(dz, dz, r2);

            __m256d inv_r = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(r2));
            __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
            __m256d inv_r3 = _mm256_mul_pd(inv_r, inv_r2);

            __m256d fi = _mm256_mul_pd(vmj, inv_r3);
            __m256d fj = _mm256_mul_pd(vmi, inv_r3);

            vaxi = _mm256_fnmadd_pd(dx, fi, vaxi);
            vayi = _mm256_fnmadd_pd(dy, fi, vayi);
            vazi = _mm256_fnmadd_pd(dz, fi, vazi);

            __m256d axj = _mm256_loadu_pd(lax + j);
            __m256d ayj = _mm256_loadu_pd(lay + j);
            __m256d azj = _mm256_loadu_pd(laz + j);

            axj = _mm256_fmadd_pd(dx, fj, axj);
            ayj = _mm256_fmadd_pd(dy, fj, ayj);
            azj = _mm256_fmadd_pd(dz, fj, azj);

            _mm256_storeu_pd(lax + j, axj);
            _mm256_storeu_pd(lay + j, ayj);
            _mm256_storeu_pd(laz + j, azj);
        }

        // Horizontal sum of vaxi, vayi, vazi
        auto hsum = [](__m256d v) -> double {
            __m128d lo = _mm256_castpd256_pd128(v);
            __m128d hi = _mm256_extractf128_pd(v, 1);
            lo = _mm_add_pd(lo, hi);
            __m128d shuf = _mm_unpackhi_pd(lo, lo);
            return _mm_cvtsd_f64(_mm_add_sd(lo, shuf));
        };
        double axi = hsum(vaxi);
        double ayi = hsum(vayi);
        double azi = hsum(vazi);

        for (; j < an; ++j) {
            const double dx = xi - lx[j];
            const double dy = yi - ly[j];
            const double dz = zi - lz[j];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double fi = lm[j] * inv_r3;
            const double fj = mi * inv_r3;
            axi -= dx * fi;
            ayi -= dy * fi;
            azi -= dz * fi;
            lax[j] += dx * fj;
            lay[j] += dy * fj;
            laz[j] += dz * fj;
        }

        lax[i] += axi;
        lay[i] += ayi;
        laz[i] += azi;
    }
}

inline void p2p_pair_avx2(
    uint32_t a0, uint32_t na, uint32_t b0, uint32_t nb, double eps2,
    const double* __restrict__ lx, const double* __restrict__ ly, const double* __restrict__ lz,
    const double* __restrict__ lm,
    double* __restrict__ lax, double* __restrict__ lay, double* __restrict__ laz
) {
    if (na > nb) {
        std::swap(a0, b0);
        std::swap(na, nb);
    }
    const uint32_t an = a0 + na;
    const uint32_t bn = b0 + nb;
    const __m256d veps2 = _mm256_set1_pd(eps2);
    const __m256d vone  = _mm256_set1_pd(1.0);

    auto hsum = [](__m256d v) -> double {
        __m128d lo = _mm256_castpd256_pd128(v);
        __m128d hi = _mm256_extractf128_pd(v, 1);
        lo = _mm_add_pd(lo, hi);
        __m128d shuf = _mm_unpackhi_pd(lo, lo);
        return _mm_cvtsd_f64(_mm_add_sd(lo, shuf));
    };

    uint32_t i = a0;
    // 2x4 register-blocked outer loop
    for (; i + 2 <= an; i += 2) {
        const double xi0 = lx[i],     yi0 = ly[i],     zi0 = lz[i],     mi0 = lm[i];
        const double xi1 = lx[i + 1], yi1 = ly[i + 1], zi1 = lz[i + 1], mi1 = lm[i + 1];
        const __m256d vxi0 = _mm256_set1_pd(xi0), vyi0 = _mm256_set1_pd(yi0), vzi0 = _mm256_set1_pd(zi0), vmi0 = _mm256_set1_pd(mi0);
        const __m256d vxi1 = _mm256_set1_pd(xi1), vyi1 = _mm256_set1_pd(yi1), vzi1 = _mm256_set1_pd(zi1), vmi1 = _mm256_set1_pd(mi1);

        __m256d vaxi0 = _mm256_setzero_pd(), vayi0 = _mm256_setzero_pd(), vazi0 = _mm256_setzero_pd();
        __m256d vaxi1 = _mm256_setzero_pd(), vayi1 = _mm256_setzero_pd(), vazi1 = _mm256_setzero_pd();

        uint32_t j = b0;
        for (; j + 4 <= bn; j += 4) {
            __m256d vxj = _mm256_loadu_pd(lx + j);
            __m256d vyj = _mm256_loadu_pd(ly + j);
            __m256d vzj = _mm256_loadu_pd(lz + j);
            __m256d vmj = _mm256_loadu_pd(lm + j);

            __m256d dx0 = _mm256_sub_pd(vxi0, vxj);
            __m256d dy0 = _mm256_sub_pd(vyi0, vyj);
            __m256d dz0 = _mm256_sub_pd(vzi0, vzj);
            __m256d r2_0 = _mm256_fmadd_pd(dz0, dz0, _mm256_fmadd_pd(dy0, dy0, _mm256_fmadd_pd(dx0, dx0, veps2)));

            __m256d dx1 = _mm256_sub_pd(vxi1, vxj);
            __m256d dy1 = _mm256_sub_pd(vyi1, vyj);
            __m256d dz1 = _mm256_sub_pd(vzi1, vzj);
            __m256d r2_1 = _mm256_fmadd_pd(dz1, dz1, _mm256_fmadd_pd(dy1, dy1, _mm256_fmadd_pd(dx1, dx1, veps2)));

            __m256d inv_r0 = _mm256_div_pd(vone, _mm256_sqrt_pd(r2_0));
            __m256d inv_r1 = _mm256_div_pd(vone, _mm256_sqrt_pd(r2_1));
            __m256d inv_r3_0 = _mm256_mul_pd(inv_r0, _mm256_mul_pd(inv_r0, inv_r0));
            __m256d inv_r3_1 = _mm256_mul_pd(inv_r1, _mm256_mul_pd(inv_r1, inv_r1));

            __m256d fi0 = _mm256_mul_pd(vmj, inv_r3_0);
            __m256d fj0 = _mm256_mul_pd(vmi0, inv_r3_0);
            __m256d fi1 = _mm256_mul_pd(vmj, inv_r3_1);
            __m256d fj1 = _mm256_mul_pd(vmi1, inv_r3_1);

            vaxi0 = _mm256_fnmadd_pd(dx0, fi0, vaxi0);
            vayi0 = _mm256_fnmadd_pd(dy0, fi0, vayi0);
            vazi0 = _mm256_fnmadd_pd(dz0, fi0, vazi0);

            vaxi1 = _mm256_fnmadd_pd(dx1, fi1, vaxi1);
            vayi1 = _mm256_fnmadd_pd(dy1, fi1, vayi1);
            vazi1 = _mm256_fnmadd_pd(dz1, fi1, vazi1);

            __m256d axj = _mm256_loadu_pd(lax + j);
            __m256d ayj = _mm256_loadu_pd(lay + j);
            __m256d azj = _mm256_loadu_pd(laz + j);

            axj = _mm256_fmadd_pd(dx1, fj1, _mm256_fmadd_pd(dx0, fj0, axj));
            ayj = _mm256_fmadd_pd(dy1, fj1, _mm256_fmadd_pd(dy0, fj0, ayj));
            azj = _mm256_fmadd_pd(dz1, fj1, _mm256_fmadd_pd(dz0, fj0, azj));

            _mm256_storeu_pd(lax + j, axj);
            _mm256_storeu_pd(lay + j, ayj);
            _mm256_storeu_pd(laz + j, azj);
        }

        double axi0 = hsum(vaxi0), ayi0 = hsum(vayi0), azi0 = hsum(vazi0);
        double axi1 = hsum(vaxi1), ayi1 = hsum(vayi1), azi1 = hsum(vazi1);

        for (; j < bn; ++j) {
            const double xj = lx[j], yj = ly[j], zj = lz[j], mj = lm[j];
            const double dx0 = xi0 - xj, dy0 = yi0 - yj, dz0 = zi0 - zj;
            const double r2_0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps2;
            const double inv_r0 = 1.0 / std::sqrt(r2_0);
            const double inv_r3_0 = inv_r0 * inv_r0 * inv_r0;

            const double dx1 = xi1 - xj, dy1 = yi1 - yj, dz1 = zi1 - zj;
            const double r2_1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps2;
            const double inv_r1 = 1.0 / std::sqrt(r2_1);
            const double inv_r3_1 = inv_r1 * inv_r1 * inv_r1;

            axi0 -= dx0 * (mj * inv_r3_0);
            ayi0 -= dy0 * (mj * inv_r3_0);
            azi0 -= dz0 * (mj * inv_r3_0);

            axi1 -= dx1 * (mj * inv_r3_1);
            ayi1 -= dy1 * (mj * inv_r3_1);
            azi1 -= dz1 * (mj * inv_r3_1);

            lax[j] += dx0 * (mi0 * inv_r3_0) + dx1 * (mi1 * inv_r3_1);
            lay[j] += dy0 * (mi0 * inv_r3_0) + dy1 * (mi1 * inv_r3_1);
            laz[j] += dz0 * (mi0 * inv_r3_0) + dz1 * (mi1 * inv_r3_1);
        }

        lax[i]     += axi0; lay[i]     += ayi0; laz[i]     += azi0;
        lax[i + 1] += axi1; lay[i + 1] += ayi1; laz[i + 1] += azi1;
    }

    for (; i < an; ++i) {
        const double xi = lx[i], yi = ly[i], zi = lz[i], mi = lm[i];
        const __m256d vxi = _mm256_set1_pd(xi);
        const __m256d vyi = _mm256_set1_pd(yi);
        const __m256d vzi = _mm256_set1_pd(zi);
        const __m256d vmi = _mm256_set1_pd(mi);

        __m256d vaxi = _mm256_setzero_pd();
        __m256d vayi = _mm256_setzero_pd();
        __m256d vazi = _mm256_setzero_pd();

        uint32_t j = b0;
        for (; j + 4 <= bn; j += 4) {
            __m256d vxj = _mm256_loadu_pd(lx + j);
            __m256d vyj = _mm256_loadu_pd(ly + j);
            __m256d vzj = _mm256_loadu_pd(lz + j);
            __m256d vmj = _mm256_loadu_pd(lm + j);

            __m256d dx = _mm256_sub_pd(vxi, vxj);
            __m256d dy = _mm256_sub_pd(vyi, vyj);
            __m256d dz = _mm256_sub_pd(vzi, vzj);

            __m256d r2 = _mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dx, dx, veps2)));
            __m256d inv_r = _mm256_div_pd(vone, _mm256_sqrt_pd(r2));
            __m256d inv_r3 = _mm256_mul_pd(inv_r, _mm256_mul_pd(inv_r, inv_r));

            __m256d fi = _mm256_mul_pd(vmj, inv_r3);
            __m256d fj = _mm256_mul_pd(vmi, inv_r3);

            vaxi = _mm256_fnmadd_pd(dx, fi, vaxi);
            vayi = _mm256_fnmadd_pd(dy, fi, vayi);
            vazi = _mm256_fnmadd_pd(dz, fi, vazi);

            __m256d axj = _mm256_loadu_pd(lax + j);
            __m256d ayj = _mm256_loadu_pd(lay + j);
            __m256d azj = _mm256_loadu_pd(laz + j);

            axj = _mm256_fmadd_pd(dx, fj, axj);
            ayj = _mm256_fmadd_pd(dy, fj, ayj);
            azj = _mm256_fmadd_pd(dz, fj, azj);

            _mm256_storeu_pd(lax + j, axj);
            _mm256_storeu_pd(lay + j, ayj);
            _mm256_storeu_pd(laz + j, azj);
        }

        double axi = hsum(vaxi);
        double ayi = hsum(vayi);
        double azi = hsum(vazi);

        for (; j < bn; ++j) {
            const double dx = xi - lx[j];
            const double dy = yi - ly[j];
            const double dz = zi - lz[j];
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double fi = lm[j] * inv_r3;
            const double fj = mi * inv_r3;
            axi -= dx * fi;
            ayi -= dy * fi;
            azi -= dz * fi;
            lax[j] += dx * fj;
            lay[j] += dy * fj;
            laz[j] += dz * fj;
        }

        lax[i] += axi;
        lay[i] += ayi;
        laz[i] += azi;
    }
}

// Templated FMM solver at expansion order P
template <int P>
class Solver {
    static constexpr int C_LEN = symset_total(P);
    static constexpr int M_LEN = (P >= 3) ? (symset_total(P - 1) - 4) : 1;
    static constexpr int M1_LEN = symset_total(P - 1); // orders 0..P-1 for Step 1 contracted tensor B

    const Config& cfg;
    OctTree& tree;
    double eps2;
    std::vector<double> cell_M;       // Multipoles M_scaled for each cell (size ncells * M_LEN)
    std::vector<double> cell_C;       // Local field tensors C for each cell (size ncells * C_LEN)
    std::vector<double> cell_power;   // Multipole power P_k (k=0..P) for Dehnen 2014 MAC (size ncells * (P+1))
    std::vector<double> cell_a_est;   // Estimated min acceleration scale a_B per cell for Dehnen 2014 MAC (Eq. 15a)
    std::vector<double> leaf_a_est;   // Estimated acceleration scale a_i per leaf for Dehnen 2014 MAC (Eq. 15a)

    uint64_t stat_m2l = 0;
    uint64_t stat_cb_approx = 0;
    uint64_t stat_p2p_cc = 0;
    uint64_t stat_p2p_cs = 0;
    uint64_t stat_p2p_pairs = 0;

public:
    Solver(const Config& c, OctTree& t)
        : cfg(c), tree(t), eps2(c.eps * c.eps) {}

    // Low-order (p=1, theta=1.0) mutual FMM pass to estimate per-particle acceleration scale a_b
    // as described in Dehnen (2014) Section 5.4.2 ("Estimating a_b or f_b using low-order FMM").
    void estimate_accel_scale_low_order() {
        const size_t nc = tree.cells.size();
        const int n = tree.N;
        std::vector<Vec3> c_acc(nc, {0.0, 0.0, 0.0});
        std::vector<double> ax_est(n, 0.0), ay_est(n, 0.0), az_est(n, 0.0);

        const double* __restrict__ lx = tree.lx.data();
        const double* __restrict__ ly = tree.ly.data();
        const double* __restrict__ lz = tree.lz.data();
        const double* __restrict__ lm = tree.lm.data();

        struct Pair { uint32_t a, b; };
        std::vector<uint32_t> cx_stack;
        std::vector<Pair> cc_stack;
        cx_stack.reserve(128);
        cc_stack.reserve(512);

        auto try_cc_p1 = [&](uint32_t a, uint32_t b) {
            const Cell& A = tree.cells[a];
            const Cell& B = tree.cells[b];
            const double dx = A.cofm.x - B.cofm.x;
            const double dy = A.cofm.y - B.cofm.y;
            const double dz = A.cofm.z - B.cofm.z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double rmax_sum = A.rmax + B.rmax;
            // Criterion theta < 1 (Dehnen 2014 Section 5.4.2)
            if (r2 > rmax_sum * rmax_sum) {
                const double inv_r = 1.0 / std::sqrt(r2 + eps2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double fA = B.mass * inv_r3;
                const double fB = A.mass * inv_r3;
                c_acc[a].x -= dx * fA; c_acc[a].y -= dy * fA; c_acc[a].z -= dz * fA;
                c_acc[b].x += dx * fB; c_acc[b].y += dy * fB; c_acc[b].z += dz * fB;
                return;
            }
            if ((A.ncells == 0 && B.ncells == 0) || (A.number <= 16 && B.number <= 16)) {
                p2p_pair_avx2(A.fcleaf, A.number, B.fcleaf, B.number, eps2,
                              lx, ly, lz, lm, ax_est.data(), ay_est.data(), az_est.data());
                return;
            }
            cc_stack.push_back({a, b});
        };

        auto try_cb_p1 = [&](auto& self, uint32_t a, uint32_t l_idx) -> void {
            const Cell& A = tree.cells[a];
            const double dx = A.cofm.x - lx[l_idx];
            const double dy = A.cofm.y - ly[l_idx];
            const double dz = A.cofm.z - lz[l_idx];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > A.rmax * A.rmax) {
                const double inv_r = 1.0 / std::sqrt(r2 + eps2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double fA = lm[l_idx] * inv_r3;
                const double fL = A.mass * inv_r3;
                c_acc[a].x -= dx * fA; c_acc[a].y -= dy * fA; c_acc[a].z -= dz * fA;
                ax_est[l_idx] += dx * fL; ay_est[l_idx] += dy * fL; az_est[l_idx] += dz * fL;
                return;
            }
            if (A.ncells == 0 || A.number <= 16) {
                p2p_pair_avx2(l_idx, 1, A.fcleaf, A.number, eps2,
                              lx, ly, lz, lm, ax_est.data(), ay_est.data(), az_est.data());
                return;
            }
            if (A.nleafs > 0) {
                p2p_pair_avx2(l_idx, 1, A.fcleaf, A.nleafs, eps2,
                              lx, ly, lz, lm, ax_est.data(), ay_est.data(), az_est.data());
            }
            const uint32_t fc = A.fccell;
            const uint32_t ec = fc + A.ncells;
            for (uint32_t c1 = fc; c1 < ec; ++c1) {
                self(self, c1, l_idx);
            }
        };

        auto drain_cc_p1 = [&]() {
            while (!cc_stack.empty()) {
                Pair p = cc_stack.back();
                cc_stack.pop_back();
                const Cell& A = tree.cells[p.a];
                const Cell& B = tree.cells[p.b];
                const bool split_a = (B.ncells == 0) || (A.ncells != 0 && A.rmax > B.rmax);
                if (split_a) {
                    for (uint32_t l1 = A.fcleaf; l1 < A.fcleaf + A.nleafs; ++l1)
                        try_cb_p1(try_cb_p1, p.b, l1);
                    for (uint32_t c1 = A.fccell; c1 < A.fccell + A.ncells; ++c1)
                        try_cc_p1(c1, p.b);
                } else {
                    for (uint32_t l2 = B.fcleaf; l2 < B.fcleaf + B.nleafs; ++l2)
                        try_cb_p1(try_cb_p1, p.a, l2);
                    for (uint32_t c2 = B.fccell; c2 < B.fccell + B.ncells; ++c2)
                        try_cc_p1(p.a, c2);
                }
            }
        };

        cx_stack.push_back(0);
        while (!cx_stack.empty()) {
            const uint32_t a = cx_stack.back();
            cx_stack.pop_back();
            const Cell& A = tree.cells[a];
            if (A.ncells == 0 || A.number <= 32) {
                p2p_self_avx2(A.fcleaf, A.number, eps2, lx, ly, lz, lm,
                              ax_est.data(), ay_est.data(), az_est.data());
                continue;
            }
            if (A.nleafs > 1) {
                p2p_self_avx2(A.fcleaf, A.nleafs, eps2, lx, ly, lz, lm,
                              ax_est.data(), ay_est.data(), az_est.data());
            }
            const uint32_t fl = A.fcleaf, el = fl + A.nleafs;
            const uint32_t fc = A.fccell, ec = fc + A.ncells;
            for (uint32_t c1 = fc; c1 < ec; ++c1) {
                cx_stack.push_back(c1);
                for (uint32_t l2 = fl; l2 < el; ++l2) try_cb_p1(try_cb_p1, c1, l2);
                for (uint32_t c2 = c1 + 1; c2 < ec; ++c2) try_cc_p1(c1, c2);
            }
            drain_cc_p1();
        }

        // Pass down p=1 accelerations and compute a_b = |a_est|
        leaf_a_est.resize(n);
        auto down_p1 = [&](auto& self, uint32_t ci, Vec3 acc_p) -> void {
            const Cell& c = tree.cells[ci];
            acc_p.x += c_acc[ci].x;
            acc_p.y += c_acc[ci].y;
            acc_p.z += c_acc[ci].z;
            for (uint32_t i = c.fcleaf; i < c.fcleaf + c.nleafs; ++i) {
                const double ax = ax_est[i] + acc_p.x;
                const double ay = ay_est[i] + acc_p.y;
                const double az = az_est[i] + acc_p.z;
                leaf_a_est[i] = std::sqrt(ax * ax + ay * ay + az * az);
            }
            for (uint32_t k = c.fccell; k < c.fccell + c.ncells; ++k) {
                self(self, k, acc_p);
            }
        };
        down_p1(down_p1, 0, {0.0, 0.0, 0.0});

        // Compute mean acceleration a_bar as a floor (1e-3 * a_bar) to prevent zero scale at exact symmetry center
        double a_sum = 0.0;
        for (int i = 0; i < n; ++i) a_sum += leaf_a_est[i];
        const double a_floor = 1e-3 * (a_sum / n);
        for (int i = 0; i < n; ++i) {
            if (leaf_a_est[i] < a_floor) leaf_a_est[i] = a_floor;
        }

        // Pass up min_{b in B} a_b for each cell B (Dehnen 2014 Eq. 15a)
        cell_a_est.resize(nc);
        for (int64_t ci = static_cast<int64_t>(nc) - 1; ci >= 0; --ci) {
            const Cell& c = tree.cells[ci];
            double amin = 1e300;
            for (uint32_t i = c.fcleaf; i < c.fcleaf + c.nleafs; ++i) {
                if (leaf_a_est[i] < amin) amin = leaf_a_est[i];
            }
            for (uint32_t k = c.fccell; k < c.fccell + c.ncells; ++k) {
                if (cell_a_est[k] < amin) amin = cell_a_est[k];
            }
            cell_a_est[ci] = amin;
        }
    }
    // Follows Dehnen (2000) Section 2.1, Dehnen (2002) Section 2.4 & 3.1 (falcON::GravEstimator::pass_up)
    void pass_up() {
        const size_t nc = tree.cells.size();
        cell_M.assign(nc * M_LEN, 0.0);
        cell_C.assign(nc * C_LEN, 0.0);
        if (cfg.mac == MacType::Dehnen2014) {
            cell_power.assign(nc * (P + 1), 0.0);
        }

        const double* __restrict__ lx = tree.lx.data();
        const double* __restrict__ ly = tree.ly.data();
        const double* __restrict__ lz = tree.lz.data();
        const double* __restrict__ lm = tree.lm.data();

        // Loop cells bottom-up (since children always have larger indices than parents)
        // Follows falcON::GravEstimator::pass_up (gravity.cc lines 818-860)
        for (int64_t ci = static_cast<int64_t>(nc) - 1; ci >= 0; --ci) {
            Cell& c = tree.cells[ci];
            double* __restrict__ Mc = cell_M.data() + ci * M_LEN;

            double m_sum = 0.0;
            Vec3 com{0.0, 0.0, 0.0};
            const uint32_t fc = c.fccell;
            const uint32_t ec = fc + c.ncells;
            for (uint32_t k = fc; k < ec; ++k) {
                const Cell& sub = tree.cells[k];
                m_sum += sub.mass;
                com.x += sub.mass * sub.cofm.x;
                com.y += sub.mass * sub.cofm.y;
                com.z += sub.mass * sub.cofm.z;
            }
            const uint32_t fl = c.fcleaf;
            const uint32_t el = fl + c.nleafs;
            for (uint32_t i = fl; i < el; ++i) {
                const double mi = lm[i];
                m_sum += mi;
                com.x += mi * lx[i];
                com.y += mi * ly[i];
                com.z += mi * lz[i];
            }
            c.mass = m_sum;
            const double inv_m = (m_sum > 0.0) ? (1.0 / m_sum) : 0.0;
            com.x *= inv_m;
            com.y *= inv_m;
            com.z *= inv_m;
            c.cofm = com;

            double dmax = 0.0;
            if (c.nleafs > 0) {
                double dmax2 = 0.0;
                for (uint32_t i = fl; i < el; ++i) {
                    const Vec3 dv{lx[i] - com.x, ly[i] - com.y, lz[i] - com.z};
                    const double d2 = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
                    if (d2 > dmax2) dmax2 = d2;
                    p2m_add_body<P>(dv, lm[i], Mc);
                }
                dmax = std::sqrt(dmax2);
            }
            for (uint32_t k = fc; k < ec; ++k) {
                const Cell& sub = tree.cells[k];
                const Vec3 dv{sub.cofm.x - com.x, sub.cofm.y - com.y, sub.cofm.z - com.z};
                const double d2 = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
                const double x = dmax - sub.rmax;
                if (x < 0.0 || d2 > x * x) {
                    dmax = std::sqrt(d2) + sub.rmax;
                }
                const double* __restrict__ Msub = cell_M.data() + k * M_LEN;
                m2m_add_cell<P>(dv, sub.mass, Msub, Mc);
            }
            const double bmax = std::sqrt(
                (c.radius + std::abs(com.x - c.centre.x)) * (c.radius + std::abs(com.x - c.centre.x)) +
                (c.radius + std::abs(com.y - c.centre.y)) * (c.radius + std::abs(com.y - c.centre.y)) +
                (c.radius + std::abs(com.z - c.centre.z)) * (c.radius + std::abs(com.z - c.centre.z))
            );
            c.rmax = std::min(dmax, bmax);
        }

        // Normalize multipoles and pre-multiply by (-1)^m * trinom(m; nx, ny, nz) for O(1) inner product
        for (size_t ci = 0; ci < nc; ++ci) {
            double* __restrict__ Mc = cell_M.data() + ci * M_LEN;
            const double mass = tree.cells[ci].mass;
            if (cfg.mac == MacType::Dehnen2014) {
                // Compute Cartesian multipole power P_k (Dehnen 2002 Appendix A Eq. A.4 / Dehnen 2014 Eq. 10)
                // ||M^{(k)}|| / M_A = sqrt( sum_{|m|=k} trinom(k; m) (M_m / M_A)^2 )
                double* __restrict__ pk = cell_power.data() + ci * (P + 1);
                pk[0] = 1.0;
                pk[1] = 0.0; // dipole vanishes around center of mass
                if constexpr (P >= 3) {
                    const double inv_m = (mass > 0.0) ? (1.0 / mass) : 0.0;
                    for (int k = 2; k <= P - 1; ++k) {
                        const int off = symset_offset(k);
                        double s2 = 0.0;
                        for (int i = 0; i < sym3d_size(k); ++i) {
                            const double val = Mc[off - 4 + i] * inv_m;
                            s2 += META<P>.mult[off + i] * val * val;
                        }
                        pk[k] = std::sqrt(s2);
                    }
                }
                pk[P] = std::pow(tree.cells[ci].rmax, P);
            }
            normalize_poles<P>(mass, Mc);
            if constexpr (P >= 3) {
                for (int m = 2; m <= P - 1; ++m) {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    const int off = symset_offset(m);
                    for (int i = 0; i < sym3d_size(m); ++i) {
                        Mc[off - 4 + i] *= sign * META<P>.mult[off + i];
                    }
                }
            }
        }

        // Set rcrit for each cell according to MAC (Dehnen 2000 Eq. 2 / Dehnen 2002 Eq. 12)
        if (cfg.mac == MacType::ConstTheta) {
            const double inv_th = 1.0 / cfg.theta;
            for (size_t ci = 0; ci < nc; ++ci) {
                tree.cells[ci].rcrit = tree.cells[ci].rmax * inv_th;
                tree.cells[ci].rcrit2 = tree.cells[ci].rcrit * tree.cells[ci].rcrit;
            }
        } else if (cfg.mac == MacType::ThetaOfM) {
            // Dehnen (2002) Eq. (12); falcON::GravMAC::set_rcrit (gravity.cc lines 179-185)
            InvertZ iz(1.0 / 3.0, P);
            const double th0 = std::min(1.0, std::abs(cfg.theta));
            const double m0 = tree.cells[0].mass;
            const double fac = (1.0 - th0) * (1.0 - th0) / std::pow(th0, P + 2);
            const double iF = (fac * fac * fac) / m0;
            for (size_t ci = 0; ci < nc; ++ci) {
                const double inv_th = iz.invtheta(tree.cells[ci].mass * iF);
                tree.cells[ci].rcrit = tree.cells[ci].rmax * inv_th;
                tree.cells[ci].rcrit2 = tree.cells[ci].rcrit * tree.cells[ci].rcrit;
            }
        } else if (cfg.mac == MacType::Dehnen2014) {
            // In Dehnen (2014) Eq. (15a), theta < 1 is the convergence guard, so rcrit = rmax,
            // and cfg.theta serves as the relative force error parameter epsilon!
            for (size_t ci = 0; ci < nc; ++ci) {
                tree.cells[ci].rcrit = tree.cells[ci].rmax;
                tree.cells[ci].rcrit2 = tree.cells[ci].rmax * tree.cells[ci].rmax;
            }
            estimate_accel_scale_low_order();
        }
    }

    // Check whether two cells A and B are well-separated
    inline bool well_separated(uint32_t a, uint32_t b, double r2) const {
        const Cell& A = tree.cells[a];
        const Cell& B = tree.cells[b];
        const double rsum = A.rcrit + B.rcrit;
        if (r2 <= rsum * rsum) return false;
        if (cfg.mac == MacType::Dehnen2014) {
            // Dehnen (2014) Eq. (11)-(15a) multipole-power error test:
            // theta < 1 and E_tilde_{A->B} * M_A / r^2 < eps * min_{b in B} a_b
            // and E_tilde_{B->A} * M_B / r^2 < eps * min_{a in A} a_a
            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double inv_rp2 = std::pow(inv_r, P + 2);
            const double* __restrict__ pA = cell_power.data() + a * (P + 1);
            const double* __restrict__ pB = cell_power.data() + b * (P + 1);
            double sumAB = 0.0, sumBA = 0.0;
            double rB_p[P + 1], rA_p[P + 1];
            rB_p[0] = 1.0; rA_p[0] = 1.0;
            for (int k = 1; k <= P; ++k) {
                rB_p[k] = rB_p[k - 1] * B.rmax;
                rA_p[k] = rA_p[k - 1] * A.rmax;
            }
            for (int k = 0; k <= P; ++k) {
                const double bin = COMB.binom[P][k];
                sumAB += bin * pA[k] * rB_p[P - k];
                sumBA += bin * pB[k] * rA_p[P - k];
            }
            const double scale = 8.0 * std::max(A.rmax, B.rmax) / (A.rmax + B.rmax + 1e-300) * inv_rp2;
            const double dA_on_B = scale * sumAB * A.mass;
            const double dB_on_A = scale * sumBA * B.mass;
            const double eps_tol = cfg.theta;
            return (dA_on_B < eps_tol * cell_a_est[b]) && (dB_on_A < eps_tol * cell_a_est[a]);
        }
        return true;
    }

    struct AccBlock {
        int nr = 0;
        uint32_t a[4];
        uint32_t b[4];
        Vec3 dX[4];
        double mAmB[4];
        double r2_eps2[4];
    } acc_buf;

    inline void flush_acc_buf() {
        if (acc_buf.nr == 0) return;
        if (acc_buf.nr == 4) {
            double* CA[4];
            double* CB[4];
            const double* MA[4];
            const double* MB[4];
            for (int k = 0; k < 4; ++k) {
                const uint32_t a = acc_buf.a[k];
                const uint32_t b = acc_buf.b[k];
                tree.cells[a].has_coeffs = 1;
                tree.cells[b].has_coeffs = 1;
                CA[k] = cell_C.data() + a * C_LEN;
                CB[k] = cell_C.data() + b * C_LEN;
                MA[k] = cell_M.data() + a * M_LEN;
                MB[k] = cell_M.data() + b * M_LEN;
            }
            approx_cc_batch4_avx2<P>(acc_buf.dX, acc_buf.mAmB, acc_buf.r2_eps2, CA, MB, CB, MA);
        } else {
            for (int k = 0; k < acc_buf.nr; ++k) {
                const uint32_t a = acc_buf.a[k];
                const uint32_t b = acc_buf.b[k];
                tree.cells[a].has_coeffs = 1;
                tree.cells[b].has_coeffs = 1;
                double F[C_LEN];
                set_dPhi<P>(acc_buf.dX[k], acc_buf.mAmB[k], acc_buf.r2_eps2[k], F);
                double* __restrict__ CA = cell_C.data() + a * C_LEN;
                double* __restrict__ CB = cell_C.data() + b * C_LEN;
                const double* __restrict__ MA = cell_M.data() + a * M_LEN;
                const double* __restrict__ MB = cell_M.data() + b * M_LEN;
                add_C_C2C_mutual<P>(CA, MB, CB, MA, F);
            }
        }
        stat_m2l += acc_buf.nr;
        acc_buf.nr = 0;
    }

    // Mutual cell-cell approximate interaction (M2L) (Dehnen 2002 Eq. 4-5; Dehnen 2014 Section 6.1.1)
    inline void approx_cc(uint32_t a, uint32_t b, const Vec3& dX, double r2) {
        const int idx = acc_buf.nr++;
        acc_buf.a[idx] = a;
        acc_buf.b[idx] = b;
        acc_buf.dX[idx] = dX;
        acc_buf.mAmB[idx] = tree.cells[a].mass * tree.cells[b].mass;
        acc_buf.r2_eps2[idx] = r2 + eps2;
        if (acc_buf.nr == 4) {
            flush_acc_buf();
        }
    }

    inline bool well_separated_cb(uint32_t a, uint32_t l_idx, double r2) const {
        const Cell& A = tree.cells[a];
        if (r2 <= A.rcrit2) return false;
        if (cfg.mac == MacType::Dehnen2014) {
            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double inv_rp2 = std::pow(inv_r, P + 2);
            const double* __restrict__ pA = cell_power.data() + a * (P + 1);
            const double dA_on_B = 8.0 * pA[P] * inv_rp2 * A.mass;
            const double dB_on_A = 8.0 * pA[P] * inv_rp2 * tree.lm[l_idx];
            const double eps_tol = cfg.theta;
            return (dA_on_B < eps_tol * leaf_a_est[l_idx]) && (dB_on_A < eps_tol * cell_a_est[a]);
        }
        return true;
    }

    // Mutual cell-leaf approximate interaction (P2L + M2P) (Dehnen 2002; kernel.cc CellLeafAll)
    inline void approx_cb(uint32_t a, uint32_t l_idx, const Vec3& dX, double r2) {
        Cell& A = tree.cells[a];
        A.has_coeffs = 1;
        const double m_leaf = tree.lm[l_idx];
        double F[C_LEN];
        set_dPhi<P>(dX, A.mass * m_leaf, r2 + eps2, F);
        double* __restrict__ CA = cell_C.data() + a * C_LEN;
        add_C_B2C<P>(CA, F);
        flip_sign_odd<P>(F);
        const double* __restrict__ MA = cell_M.data() + a * M_LEN;
        double leaf_C[4] = {0.0, 0.0, 0.0, 0.0};
        add_C_C2B<P>(leaf_C, F, MA);
        const double inv_ml = 1.0 / m_leaf;
        tree.lax[l_idx] += leaf_C[1] * inv_ml;
        tree.lay[l_idx] += leaf_C[2] * inv_ml;
        tree.laz[l_idx] += leaf_C[3] * inv_ml;
    }

    // Dual tree walk (Dehnen 2002 Section 3.2.1; interact.h MutualInteractor)
    void interact_dual_tree() {
        double* __restrict__ lx = tree.lx.data();
        double* __restrict__ ly = tree.ly.data();
        double* __restrict__ lz = tree.lz.data();
        double* __restrict__ lm = tree.lm.data();
        double* __restrict__ lax = tree.lax.data();
        double* __restrict__ lay = tree.lay.data();
        double* __restrict__ laz = tree.laz.data();

        struct Pair { uint32_t a, b; };
        std::vector<uint32_t> cx_stack;
        std::vector<Pair> cc_stack;
        std::vector<Pair> cb_stack; // {cell_idx, leaf_idx}
        cx_stack.reserve(256);
        cc_stack.reserve(1024);
        cb_stack.reserve(256);

        auto try_cb = [&](uint32_t a, uint32_t l_idx) {
            const Cell& A = tree.cells[a];
            if (A.number < static_cast<uint32_t>(cfg.n_cb_pre)) {
                p2p_pair_avx2(l_idx, 1, A.fcleaf, A.number, eps2, lx, ly, lz, lm, lax, lay, laz);
                stat_p2p_pairs += A.number;
                return;
            }
            const Vec3 dX{A.cofm.x - lx[l_idx], A.cofm.y - ly[l_idx], A.cofm.z - lz[l_idx]};
            const double r2 = dX.x * dX.x + dX.y * dX.y + dX.z * dX.z;
            if (well_separated_cb(a, l_idx, r2)) {
                approx_cb(a, l_idx, dX, r2);
                stat_cb_approx++;
                return;
            }
            if (A.ncells == 0 || A.number < static_cast<uint32_t>(cfg.n_cb_post)) {
                p2p_pair_avx2(l_idx, 1, A.fcleaf, A.number, eps2, lx, ly, lz, lm, lax, lay, laz);
                stat_p2p_pairs += A.number;
                return;
            }
            cb_stack.push_back({a, l_idx});
        };

        auto drain_cb = [&]() {
            while (!cb_stack.empty()) {
                Pair p = cb_stack.back();
                cb_stack.pop_back();
                const Cell& A = tree.cells[p.a];
                if (A.nleafs > 0) {
                    p2p_pair_avx2(p.b, 1, A.fcleaf, A.nleafs, eps2, lx, ly, lz, lm, lax, lay, laz);
                    stat_p2p_pairs += A.nleafs;
                }
                const uint32_t fc = A.fccell;
                const uint32_t ec = fc + A.ncells;
                for (uint32_t c1 = fc; c1 < ec; ++c1) {
                    try_cb(c1, p.b);
                }
            }
        };

        auto try_cc = [&](uint32_t a, uint32_t b) {
            const Cell& A = tree.cells[a];
            const Cell& B = tree.cells[b];
            const Vec3 dX{A.cofm.x - B.cofm.x, A.cofm.y - B.cofm.y, A.cofm.z - B.cofm.z};
            const double r2 = dX.x * dX.x + dX.y * dX.y + dX.z * dX.z;
            if (well_separated(a, b, r2)) {
                approx_cc(a, b, dX, r2);
                return;
            }
            if ((A.ncells == 0 && B.ncells == 0) ||
                (A.number < static_cast<uint32_t>(cfg.n_cc_post) &&
                 B.number < static_cast<uint32_t>(cfg.n_cc_post))) {
                p2p_pair_avx2(A.fcleaf, A.number, B.fcleaf, B.number, eps2, lx, ly, lz, lm, lax, lay, laz);
                stat_p2p_cc++;
                stat_p2p_pairs += static_cast<uint64_t>(A.number) * B.number;
                return;
            }
            cc_stack.push_back({a, b});
        };

        auto drain_cc = [&]() {
            while (!cc_stack.empty()) {
                Pair p = cc_stack.back();
                cc_stack.pop_back();
                const Cell& A = tree.cells[p.a];
                const Cell& B = tree.cells[p.b];
                // Split larger cell (or the non-twig cell if one is a twig) (gravity.cc split_first line 311)
                const bool split_a = (B.ncells == 0) || (A.ncells != 0 && A.rmax > B.rmax);
                if (split_a) {
                    const uint32_t fl = A.fcleaf;
                    const uint32_t el = fl + A.nleafs;
                    for (uint32_t l1 = fl; l1 < el; ++l1) {
                        try_cb(p.b, l1);
                    }
                    drain_cb();
                    const uint32_t fc = A.fccell;
                    const uint32_t ec = fc + A.ncells;
                    for (uint32_t c1 = fc; c1 < ec; ++c1) {
                        try_cc(c1, p.b);
                    }
                } else {
                    const uint32_t fl = B.fcleaf;
                    const uint32_t el = fl + B.nleafs;
                    for (uint32_t l2 = fl; l2 < el; ++l2) {
                        try_cb(p.a, l2);
                    }
                    drain_cb();
                    const uint32_t fc = B.fccell;
                    const uint32_t ec = fc + B.ncells;
                    for (uint32_t c2 = fc; c2 < ec; ++c2) {
                        try_cc(p.a, c2);
                    }
                }
            }
        };

        cx_stack.push_back(0);
        while (!cx_stack.empty()) {
            const uint32_t a = cx_stack.back();
            cx_stack.pop_back();
            const Cell& A = tree.cells[a];
            if (A.ncells == 0 || A.number < static_cast<uint32_t>(cfg.n_cs)) {
                p2p_self_avx2(A.fcleaf, A.number, eps2, lx, ly, lz, lm, lax, lay, laz);
                stat_p2p_cs++;
                stat_p2p_pairs += static_cast<uint64_t>(A.number) * (A.number - 1) / 2;
                continue;
            }
            if (A.nleafs > 1) {
                p2p_self_avx2(A.fcleaf, A.nleafs, eps2, lx, ly, lz, lm, lax, lay, laz);
                stat_p2p_pairs += static_cast<uint64_t>(A.nleafs) * (A.nleafs - 1) / 2;
            }
            const uint32_t fl = A.fcleaf;
            const uint32_t el = fl + A.nleafs;
            const uint32_t fc = A.fccell;
            const uint32_t ec = fc + A.ncells;
            for (uint32_t c1 = fc; c1 < ec; ++c1) {
                cx_stack.push_back(c1);
                for (uint32_t l2 = fl; l2 < el; ++l2) {
                    try_cb(c1, l2);
                }
                for (uint32_t c2 = c1 + 1; c2 < ec; ++c2) {
                    try_cc(c1, c2);
                }
            }
            drain_cb();
            drain_cc();
        }
        flush_acc_buf();
    }

    // Downward evaluation pass (L2L and L2P)
    // Follows Dehnen (2002) Section 3.2.2 (falcON::GravKernBase::eval_grav_all, kernel.cc lines 102-150)
    void pass_down() {
        const double* __restrict__ lx = tree.lx.data();
        const double* __restrict__ ly = tree.ly.data();
        const double* __restrict__ lz = tree.lz.data();
        double* __restrict__ lax = tree.lax.data();
        double* __restrict__ lay = tree.lay.data();
        double* __restrict__ laz = tree.laz.data();

        auto eval_cell = [&](auto& self, uint32_t ci, Vec3 X_exp, const double* C_parent, bool parent_nonempty) -> void {
            const Cell& c = tree.cells[ci];
            double C_loc[C_LEN];
            bool nonempty = parent_nonempty;
            if (nonempty) {
                std::memcpy(C_loc, C_parent, C_LEN * sizeof(double));
            } else {
                std::memset(C_loc, 0, C_LEN * sizeof(double));
            }

            if (c.has_coeffs) {
                if (nonempty) {
                    Vec3 dX{c.cofm.x - X_exp.x, c.cofm.y - X_exp.y, c.cofm.z - X_exp.z};
                    if (dX.x != 0.0 || dX.y != 0.0 || dX.z != 0.0) {
                        l2l_shift_by<P>(C_loc, dX);
                    }
                }
                X_exp = c.cofm;
                const double inv_m = 1.0 / c.mass;
                const double* __restrict__ Cc = cell_C.data() + ci * C_LEN;
                for (int i = 0; i < C_LEN; ++i) {
                    C_loc[i] += Cc[i] * inv_m;
                }
                nonempty = true;
            }

            if (nonempty && c.nleafs > 0) {
                const uint32_t s = c.fcleaf;
                const uint32_t e = s + c.nleafs;
                for (uint32_t i = s; i < e; ++i) {
                    double leaf_C[4] = {0.0, 0.0, 0.0, 0.0};
                    const Vec3 dX{lx[i] - X_exp.x, ly[i] - X_exp.y, lz[i] - X_exp.z};
                    l2p_eval_expn<P>(leaf_C, C_loc, dX);
                    lax[i] += leaf_C[1];
                    lay[i] += leaf_C[2];
                    laz[i] += leaf_C[3];
                }
            }
            if (c.ncells > 0) {
                const uint32_t fc = c.fccell;
                const uint32_t ec = fc + c.ncells;
                for (uint32_t k = fc; k < ec; ++k) {
                    self(self, k, X_exp, C_loc, nonempty);
                }
            }
        };

        double C_zero[C_LEN]{};
        eval_cell(eval_cell, 0, tree.cells[0].cofm, C_zero, false);

        if (cfg.verbose) {
            std::fprintf(stderr,
                "FMM P=%d theta=%.4f mac=%d ncells=%zu M2L=%lu P2P_CC=%lu P2P_CS=%lu P2P_pairs=%lu\n",
                P, cfg.theta, static_cast<int>(cfg.mac), tree.cells.size(),
                stat_m2l, stat_p2p_cc, stat_p2p_cs, stat_p2p_pairs);
        }
    }

    void run() {
        pass_up();
        interact_dual_tree();
        pass_down();
    }
};

inline void run_fmm(const Config& cfg, OctTree& tree) {
    switch (cfg.order) {
        case 2: { Solver<2> s(cfg, tree); s.run(); break; }
        case 3: { Solver<3> s(cfg, tree); s.run(); break; }
        case 4: { Solver<4> s(cfg, tree); s.run(); break; }
        case 5: { Solver<5> s(cfg, tree); s.run(); break; }
        case 6: { Solver<6> s(cfg, tree); s.run(); break; }
        case 7: { Solver<7> s(cfg, tree); s.run(); break; }
        case 8: { Solver<8> s(cfg, tree); s.run(); break; }
        case 9: { Solver<9> s(cfg, tree); s.run(); break; }
        case 10: { Solver<10> s(cfg, tree); s.run(); break; }
        case 11: { Solver<11> s(cfg, tree); s.run(); break; }
        case 12: { Solver<12> s(cfg, tree); s.run(); break; }
        default: { Solver<6> s(cfg, tree); s.run(); break; }
    }
}

} // namespace fmm

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]);
        return 2;
    }

    FILE* in = std::fopen(argv[1], "rb");
    if (!in) return 1;
    if (std::fseek(in, 0, SEEK_END) != 0) { std::fclose(in); return 1; }
    const long bytes = std::ftell(in);
    if (bytes < 0 || (bytes % 32) != 0) { std::fclose(in); return 1; }
    if (std::fseek(in, 0, SEEK_SET) != 0) { std::fclose(in); return 1; }

    const int n = static_cast<int>(bytes / 32);
    std::vector<double> rows(4 * static_cast<size_t>(n));
    if (std::fread(rows.data(), 32, static_cast<size_t>(n), in) != static_cast<size_t>(n)) {
        std::fclose(in);
        return 1;
    }
    std::fclose(in);

    fmm::Config cfg{};
#ifdef FMM_DEFAULT_ORDER
    cfg.order = FMM_DEFAULT_ORDER;
#endif
#ifdef FMM_DEFAULT_THETA
    cfg.theta = FMM_DEFAULT_THETA;
#endif
#ifdef FMM_DEFAULT_MAC
    cfg.mac = static_cast<fmm::MacType>(FMM_DEFAULT_MAC);
#endif
#ifdef FMM_DEFAULT_NCRIT
    cfg.ncrit = FMM_DEFAULT_NCRIT;
#endif
#ifdef FMM_DEFAULT_NCC_POST
    cfg.n_cc_post = FMM_DEFAULT_NCC_POST;
#endif
#ifdef FMM_DEFAULT_NCS
    cfg.n_cs = FMM_DEFAULT_NCS;
#endif

    if (const char* s = std::getenv("FMM_ORDER")) cfg.order = std::atoi(s);
    if (const char* s = std::getenv("FMM_THETA")) cfg.theta = std::atof(s);
    if (const char* s = std::getenv("FMM_MAC")) cfg.mac = static_cast<fmm::MacType>(std::atoi(s));
    if (const char* s = std::getenv("FMM_EPS")) cfg.eps = std::atof(s);
    if (const char* s = std::getenv("FMM_NCRIT")) cfg.ncrit = std::atoi(s);
    if (const char* s = std::getenv("FMM_NCC_POST")) cfg.n_cc_post = std::atoi(s);
    if (const char* s = std::getenv("FMM_NCS")) cfg.n_cs = std::atoi(s);
    if (const char* s = std::getenv("FMM_VERBOSE")) cfg.verbose = (std::atoi(s) != 0);

    fmm::OctTree tree;
    tree.build(n, rows.data(), cfg.ncrit);
    fmm::run_fmm(cfg, tree);

    std::vector<double> out_acc(3 * static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const uint32_t orig = tree.orig_idx[i];
        out_acc[3 * orig + 0] = tree.lax[i];
        out_acc[3 * orig + 1] = tree.lay[i];
        out_acc[3 * orig + 2] = tree.laz[i];
    }

    FILE* out = std::fopen(argv[2], "wb");
    if (!out) return 1;
    if (std::fwrite(out_acc.data(), 24, static_cast<size_t>(n), out) != static_cast<size_t>(n)) {
        std::fclose(out);
        return 1;
    }
    if (std::fclose(out) != 0) return 1;

    if (cfg.verbose) {
        std::fprintf(stderr, "cpu %.4f\n", static_cast<double>(std::clock()) / CLOCKS_PER_SEC);
    }
    return 0;
}
