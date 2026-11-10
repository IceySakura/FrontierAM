# High-Order Cartesian Fast Multipole Method (`fmm-dehnen`) for `galaxy-gravity-v2`

## Overview

This directory contains an independent C++17 / AVX2 implementation of Walter Dehnen's Cartesian Fast Multipole Method (FMM) for computing Plummer-softened gravitational accelerations:

$$\boldsymbol{a}_i = G \sum_{j \ne i} m_j \frac{\boldsymbol{x}_j - \boldsymbol{x}_i}{\bigl(\|\boldsymbol{x}_j - \boldsymbol{x}_i\|_2^2 + \epsilon^2\bigr)^{3/2}}, \qquad G = 1, \quad \epsilon = 0.01.$$

As discussed in Section 7.2 and Appendix A.1 of Dehnen (2014), the Plummer-softened Green's function $\psi(\boldsymbol{r}) = (\|\boldsymbol{r}\|_2^2 + \epsilon^2)^{-1/2}$ is not harmonic ($\nabla^2 \psi \ne 0$), so spherical-harmonic FMM expansions cannot be used directly; instead, the general Cartesian FMM of Dehnen (2000, 2002) extended to higher expansion order $p$ (Dehnen 2014, Appendix A.1) is the exact mathematical formulation for softened gravity.

## Literature Citations for Every Technique

1. **Hierarchical Octree and Bounding Sphere Radii**
   - **Citations**: Barnes & Hut (1986, *Nature*, 324, 446); Salmon & Warren (1994, *J. Comput. Phys.*, 111, 136); Benz et al. (1990, *ApJ*, 348, 647); Dehnen (2000, *ApJ*, 536, L39, Section 2.1, Eq. 3); Dehnen (2002, *J. Comput. Phys.*, 179, 27, Section 2.4 & Section 3.1).
   - **Implementation** (`solve.cpp`: `OctTree::build`, `Solver::pass_up`):
     - Root center rounded to the nearest integer of the mean particle position and root half-side length chosen as a power of two enclosing all particles (`falcON` `tree.cc` lines 160, 998).
     - Adaptive octree subdivision down to leaf/twig cells with at most $N_{\text{crit}}$ particles; octants of branch cells containing a single particle are stored as direct leaf children (`falcON` `BoxDotTree::split_box`, `tree.cc` lines 640–720), and all descendant leaves of any cell occupy a contiguous slice `[fcleaf, fcleaf + number)`.
     - Cell center of mass $\boldsymbol{z}_A = M_A^{-1} \sum_{i \in A} m_i \boldsymbol{x}_i$ and bounding radius $r_{\max, A} = \min\{d_{\max, A}, b_{\max, A}\}$, where $d_{\max, A} = \max_{c \in \text{children}(A)} (\|\boldsymbol{z}_c - \boldsymbol{z}_A\|_2 + r_{\max, c})$ and $b_{\max, A}$ is the distance from $\boldsymbol{z}_A$ to the most distant corner of the cubic cell (`falcON` `gravity.cc` line 231).

2. **Symmetric 3D Tensor Representation and Indexing**
   - **Citations**: Warren & Salmon (1995, *Comput. Phys. Commun.*, 87, 266); Dehnen (2002, Section 2, Eqs. 3–7); Dehnen (2014, Appendix A.1, Eqs. A.4a–e); `falcON` (`tensor.h`, `tensor.cc`, `tensor_set.h`, `tensor_set.cc`).
   - **Implementation** (`fmm_kernels.hpp`: `sym3d_size`, `symset_offset`, `lm_idx`, `OrderMeta<P>`):
     - A symmetric 3D Cartesian tensor of order $k$ has $\binom{k+2}{2} = \frac{(k+1)(k+2)}{2}$ independent components indexed by $l = n_y + n_z \in [0, k]$ and $m = n_z \in [0, l]$ (with $n_x = k - l$, $n_y = l - m$, $n_z = m$) at flat offset $\frac{l(l+1)}{2} + m$.
     - A complete set of symmetric 3D tensors of orders $0 \dots p$ (`symset3D<p>`) has $\binom{p+3}{3} = \frac{(p+1)(p+2)(p+3)}{6}$ elements.
     - Because cell multipoles are expanded around the cell center of mass $\boldsymbol{z}_A$, the dipole ($\boldsymbol{\mathsf{M}}^{(1)}$) vanishes identically, and specific multipole moments of order $p$ only contribute a constant to the potential and do not affect the acceleration (Dehnen 2002 Section 2). Thus each cell stores multipole moments of orders $2 \dots p-1$ (`poles3D<p-1>`).

3. **Upward Pass: Particle-to-Multipole (P2M) and Multipole-to-Multipole (M2M)**
   - **Citations**: Dehnen (2002, Eq. 5); Dehnen (2014, Eq. A.4c–d); `falcON` (`poles3D::add_body`, `poles3D::add_cell`, `poles3D::normalize` in `tensor_set.cc`).
   - **Implementation** (`fmm_kernels.hpp`: `p2m_add_body`, `m2m_add_cell`, `normalize_poles`):
     - `p2m_add_body<P>` accumulates $m_i (\boldsymbol{x}_i - \boldsymbol{z}_A)^{(k)}$ for $k = 2 \dots p-1$.
     - `m2m_add_cell<P>` shifts child multipoles from $\boldsymbol{z}_{\text{child}}$ to $\boldsymbol{z}_{\text{parent}}$ via $\boldsymbol{\mathsf{M}}_{\text{parent}}^{(k)} \mathrel{+}= m_{\text{child}} \boldsymbol{v}^{(k)} + \sum_{j=0}^{k-2} \boldsymbol{v}^{(j)} \odot \boldsymbol{\mathsf{M}}_{\text{child}}^{(k-j)}$ where $\boldsymbol{v} = \boldsymbol{z}_{\text{child}} - \boldsymbol{z}_{\text{parent}}$, factored into $p-3$ directional outer-product steps requiring $\mathcal{O}(p^4)$ operations instead of $\mathcal{O}(p^6)$.
     - `normalize_poles<P>` normalizes order-$k$ multipoles by $(M_A k!)^{-1}$ and pre-multiplies by $(-1)^k \frac{k!}{n_x! n_y! n_z!}$ so that complete inner products during M2L reduce to unweighted sums.

4. **Mutual Cell-Cell Interaction (M2L) and Radial Derivative Recursion for Plummer Kernel**
   - **Citations**: Dehnen (2000, Section 2.2, Eqs. 5–7); Dehnen (2002, Section 2.1–2.2, Eqs. 4 & 8); Dehnen (2014, Appendix A.1, Eq. A.4b); `falcON` (`kernel.cc` `_block<p0, K>` and `tensor_set.cc` `set_dPhi`, `add_C_C2C`).
   - **Implementation** (`fmm_kernels.hpp`: `set_dPhi`, `add_C_C2C_mutual`, `approx_cc_batch4_avx2`):
     - Scalar derivatives $D_k = M_A M_B (2k-1)!! / (R^2 + \epsilon^2)^{k + 1/2}$ are computed recursively for $k = 0 \dots p$.
     - The Cartesian derivative tensors $\boldsymbol{\mathsf{F}}^{(n)} = M_A M_B \boldsymbol{\nabla}^{(n)} g(\boldsymbol{R}) = \sum_{j=0}^{\lfloor n/2 \rfloor} (-1)^{n-j} D_{n-j} (\boldsymbol{R}^{(n-2j)} \odot \boldsymbol{\delta}^{(j)})$ are constructed in $\mathcal{O}(p^3)$ operations via the exact trace-injection recurrence $\boldsymbol{\mathsf{G}}_n^{(j)} = \frac{1}{j} (\boldsymbol{\mathsf{G}}_{n-2}^{(j-1)} \odot \boldsymbol{\delta}^{(1)})$.
     - Both interaction directions $B \to A$ and $A \to B$ are contracted simultaneously using the parity relation $\boldsymbol{\mathsf{F}}_{A \to B}^{(n)} = (-1)^n \boldsymbol{\mathsf{F}}_{B \to A}^{(n)}$ (Dehnen 2000 Section 2.2; Dehnen 2002 Section 2.2).

5. **Dual Tree Walk and Multipole-Acceptance Criteria (MAC)**
   - **Citations**: Dehnen (2000, Section 2.1 & 2.3, Eq. 2); Dehnen (2002, Section 2.4, 3.2.1 & Appendix B, Eqs. 10–12); Dehnen (2014, Section 2.2.2, Section 4 & Section 5, Eqs. 10–15a); `falcON` (`interact.h` `MutualInteractor`, `gravity.cc` `InvertZ`, `GravMAC`).
   - **Implementation** (`solve.cpp`: `InvertZ`, `Solver::well_separated`, `Solver::interact_dual_tree`):
     - Non-recursive stack-based dual tree walk processing mutual cell self-interactions, cell-cell interactions, and cell-leaf interactions, splitting the larger cell when an interaction cannot be approximated or handled by direct summation.
     - Supports three opening criteria selectable at build or run time (`FMM_MAC`):
       - `FMM_MAC=0`: Constant opening angle $r_{\text{crit}, A} = r_{\max, A} / \theta$, well-separated if $\|\boldsymbol{z}_A - \boldsymbol{z}_B\|_2 > r_{\text{crit}, A} + r_{\text{crit}, B}$ (Dehnen 2000 Eq. 2).
       - `FMM_MAC=1`: Mass-dependent opening angle $\frac{\theta(M)^{p+2}}{(1-\theta(M))^2} = \frac{\theta_0^{p+2}}{(1-\theta_0)^2} \left(\frac{M}{M_{\text{tot}}}\right)^{-1/3}$ inverted via log-spaced lookup table `InvertZ` (Dehnen 2002 Eq. 12; `falcON` `gravity.cc` lines 57–185).
       - `FMM_MAC=2`: Multipole-power error estimator $\tilde{E}_{A \to B} \frac{M_A}{r^2} < \epsilon \min_{b \in B} a_b$ with $a_b$ estimated by a preliminary $p=1, \theta=1$ mutual FMM pass (Dehnen 2014 Eqs. 10–15a & Section 5.4.2).

6. **Downward Pass: Local-to-Local (L2L) and Local-to-Particle (L2P)**
   - **Citations**: Dehnen (2002, Section 2.3 & 3.2.2, Eqs. 3 & 7); Dehnen (2014, Appendix A.1, Eqs. A.4a & A.4e); `falcON` (`tensor_set.cc` `shift_by`, `eval_expn`).
   - **Implementation** (`fmm_kernels.hpp`: `l2l_shift_by`, `l2p_eval_expn`; `solve.cpp`: `Solver::pass_down`):
     - Normalizes each cell's accumulated field tensor by $M_A^{-1}$, shifts the parent's Taylor expansion to the child's center of mass via Horner's scheme in $\mathcal{O}(p^4)$ operations (`l2l_shift_by`), and evaluates the local Taylor series at each leaf particle (`l2p_eval_expn`).

7. **Engineering Optimizations (AVX2 Vectorization and Memory Layout)**
   - **4-Way Batched AVX2 M2L Kernel** (`fmm_kernels.hpp`: `approx_cc_batch4_avx2`; `solve.cpp`: `AccBlock`): Following Dehnen (2014, Section 6.1.1) and `falcON` (`kernel.h` `acc_block`), well-separated cell-cell interactions are buffered in groups of $K=4$ and evaluated simultaneously in 256-bit AVX2 + FMA registers (`__m256d`).
   - **2x4 Register-Blocked AVX2 P2P Kernel** (`solve.cpp`: `p2p_pair_avx2`, `p2p_self_avx2`): Leaf particles are reordered into contiguous Structure-of-Arrays (`lx, ly, lz, lm, lax, lay, laz`) following tree-leaf order, and direct-summation interactions below $(N_{\text{cc}}^{\text{post}}, N_{\text{cs}})$ are computed with 2x4 AVX2 + FMA register blocking.

## Building and Running

Build the executable `./solve`:
```sh
./build.sh
```

Run on a snapshot:
```sh
./solve INPUT OUTPUT
```

### Configuration Parameters

The frozen default parameters compiled into `./solve` (selected by the calibration protocol in `CALIBRATION.md`) are:
- Expansion order: `FMM_ORDER = 8`
- Multipole-acceptance criterion: `FMM_MAC = 0` (Dehnen 2000 Eq. 2 / Dehnen 2014 Section 5.2 constant $\theta$)
- Opening angle: `FMM_THETA = 0.48`
- Softening length: `FMM_EPS = 0.01`
- Tree / direct-summation thresholds: `FMM_NCRIT = 96`, `FMM_NCC_POST = 96`, `FMM_NCS = 256`

All parameters can be overridden at compile time via `-DFMM_DEFAULT_ORDER=...` or at run time via environment variables (`FMM_ORDER`, `FMM_THETA`, `FMM_MAC`, `FMM_EPS`, `FMM_NCRIT`, `FMM_NCC_POST`, `FMM_NCS`, `FMM_UNROLLED`, `FMM_AUTOTUNE`, `FMM_SPLIT`, `FMM_RES_ORDER`, `FMM_TARGET_CPU`, `FMM_VERBOSE`).

## Switchable Ablation Variants (`ABLATION.md`)

In addition to the frozen paper-based reference `./solve` ($p = 8, \theta = 0.48$, kept intact), `build.sh` builds four switchable variant entrypoints for the ablation study documented in `ABLATION.md`:

- `./solve-A` (**Variant A — Engineering Only: Code-Generated Unrolled Fixed-Order Kernels**):
  Uses `gen_unrolled.py` $\to$ `fmm_unrolled.hpp` to emit straight-line, parity-factorized, register-blocked AVX2 M2L micro-kernels (`contract_k1_m*`, `contract_k2_m*`, `m2l_row_b4/b2/b1_l*`, `approx_cc_batch4_unrolled<P>`) and an $\mathcal{O}(p^3)$ direct monomial L2P kernel (`l2p_eval_expn_unrolled<P>`).
- `./solve-AB` (**Variant A+B — Unrolled Kernels + Literature Runtime Cost Model Autotuner**):
  Combines Variant A with a runtime cost model (`autotune_parameters` in `solve.cpp`) that counts $(N_{\text{M2L}}(\theta), N_{\text{P2P}}(\theta))$ at three anchor opening angles on the current input snapshot's octree and selects $(p, \theta)$ to minimize predicted multipole truncation error within the CPU budget (Dehnen 2014 Sections 5–6).
- `./solve-C` and `./solve-ABC` (**Variant C — Kernel Splitting, an Idea Taken from AI Submissions**):
  *Note: Kernel splitting is an idea taken from an AI submission on this task (not from Dehnen's papers, which treat softened gravity purely via Cartesian FMM in Section 7.2).* It decomposes the Plummer Green's function as $\psi_{\text{Plummer}}(r) = \frac{1}{r} + \bigl(\frac{1}{\sqrt{r^2+\epsilon^2}} - \frac{1}{r}\bigr)$, expanding the harmonic Newtonian $1/r$ part to high order $p_{\text{harm}}$ (up to $16$) in real solid spherical harmonics $U_n^m(\boldsymbol{r}), T_n^m(\boldsymbol{r})$ with $(p_{\text{harm}}+1)^2$ coefficients (`fmm_harmonic.hpp`, Dehnen 2014 Section 2.1 & Appendix A.3), and the fast-decaying smooth residual $\mathcal{O}(\epsilon^2/r^3)$ with a lower-order ($p_{\text{res}} = 6$) Cartesian expansion (`set_dPhi_residual`, `approx_cc_residual_batch4_avx2`), while keeping near pairs direct with the exact softened Plummer kernel. See `ABLATION.md` for full results and analysis.
