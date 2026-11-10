# Ablation Study: Closing the Accuracy Gap on `galaxy-gravity-v2` (`fmm-dehnen`)

## 1. Overview and Experimental Protocol

The frozen paper-based reference (`solve`, configured with Cartesian FMM order $p = 8$, constant opening angle $\theta = 0.48$, $N_{\text{crit}} = 96, N_{\text{cc}}^{\text{post}} = 96, N_{\text{cs}} = 256$) achieves a mean RMS relative acceleration error of **$5.914 \times 10^{-5}$** across the 10 public snapshots (`case-01` .. `case-10`, $N = 131{,}072$, $\epsilon = 0.01$) within $3.473\text{ CPU s}$ on a leased development slot.

To determine which algorithmic and engineering ideas improve accuracy under the same budget rule (**best accuracy with worst-case CPU time $\le 3.600\text{ s}$ on a leased development slot** inside `frontieram-galaxy-gravity:amd64-b2`), we implemented and evaluated three switchable extensions to our solver:

- **Variant A (`solve-A` / `FMM_UNROLLED=1`) — Engineering Only (Code-Generated Unrolled Fixed-Order Kernels)**:
  A Python code generator (`gen_unrolled.py` $\to$ `fmm_unrolled.hpp`) emits straight-line, branchless AVX2 + FMA micro-kernels (`contract_k1_m*`, `contract_k2_m*`, `m2l_row_b4/b2/b1_l*`, `approx_cc_batch4_unrolled<P>`) that exploit parity factorization $\boldsymbol{\mathsf{C}}_B^{(k)} = (-1)^k \bigl[\boldsymbol{\mathsf{F}}^{(k)} + \sum_{m=2}^{p-k} \boldsymbol{\mathsf{F}}^{(k+m)} \cdot (+\boldsymbol{\mathsf{M}}_A^{(m)})\bigr]$ to keep 6 target accumulators (`ca0..ca2, cb0..cb2`) resident in YMM registers across all multipole orders $m = 2 \dots p-k$, and replaces the $\mathcal{O}(p^4)$ Horner L2P evaluation with an unrolled $\mathcal{O}(p^3)$ direct monomial dot product (`l2p_eval_expn_unrolled<P>`).
- **Variant B (`solve-AB` / `FMM_UNROLLED=1 FMM_AUTOTUNE=1`) — Literature Parameter Selection & Runtime Cost Model**:
  Allows expansion order $p$ up to $12$ (or $16$ when combined with C) and runs a lightweight (<15 ms) dry-run dual tree traversal at three anchor opening angles ($\theta \in \{0.36, 0.48, 0.60\}$) on the current input snapshot's octree to measure exact interaction counts $(N_{\text{M2L}}(\theta), N_{\text{P2P}}(\theta))$. It then selects the pair $(p, \theta)$ minimizing the predicted multipole truncation error $(p+1)\theta^p/(1-\theta)^2$ (Dehnen 2014 Sections 5–6) subject to the predicted CPU time staying within the $3.60\text{ s}$ budget.
- **Variant C (`solve-C` / `FMM_SPLIT=1`, and `solve-ABC` / `FMM_SPLIT=1 FMM_UNROLLED=1 FMM_AUTOTUNE=1`) — Kernel Splitting (Idea from AI Submissions)**:
  Splits the non-harmonic Plummer Green's function into the harmonic Newtonian $1/r$ kernel plus a smooth short-range residual:
  $$\psi_{\text{Plummer}}(r) = \frac{1}{\sqrt{r^2 + \epsilon^2}} = \frac{1}{r} + \underbrace{\left(\frac{1}{\sqrt{r^2 + \epsilon^2}} - \frac{1}{r}\right)}_{\delta\psi(r) \,=\, -\frac{\epsilon^2}{2r^3} + \frac{3\epsilon^4}{8r^5} - \cdots}.$$
  The Newtonian $1/r$ part satisfies $\nabla^2(1/r) = 0$ and is expanded to high order $p_{\text{harm}}$ (up to $16$) in real-valued solid spherical harmonics $U_n^m(\boldsymbol{r})$ and $T_n^m(\boldsymbol{r})$ (`fmm_harmonic.hpp`, implementing Dehnen 2014 Section 2.1 Eqs. 2–3 and Appendix A.3–A.4 Eqs. A.19–A.25), which have only $(p_{\text{harm}}+1)^2$ coefficients instead of $\binom{p+3}{3}$ and use conjugate symmetry ($+l$ and $-l$ folded) in 4-way AVX2 batches (`harm_m2l_batch4_avx2`). The smooth residual $\delta\psi(r)$ decays as $\mathcal{O}(\epsilon^2/r^3)$ and is expanded at lower Cartesian order $p_{\text{res}} = 6$ using cancellation-free rationalized radial derivatives (`set_dPhi_residual`, `approx_cc_residual_batch4_avx2`) with a core guard $r > 8\epsilon$, while near pairs use exact softened P2P direct summation.

---

## 2. Single M2L Kernel Cost Before and After Unrolling / Splitting

Measured inside the `frontieram-galaxy-gravity:amd64-b2` container on a leased development slot (`bench_m2l_ablation.cpp`, per mutual cell-cell interaction):

| Expansion Order | Scalar Loop-Based Cartesian M2L (ns) | Baseline 4-Way AVX2 Loop Cartesian M2L (ns) | **Variant A Code-Generated Unrolled 4-Way AVX2 Cartesian M2L (ns)** | Variant A Speedup vs. Scalar / vs. Baseline Loop | **Variant C Split Harmonic ($p_{\text{harm}}$) + Residual ($p_{\text{res}}=6$) M2L (ns)** | Variant C Speedup vs. Baseline Loop at Same $p$ |
| --- | --- | --- | --- | --- | --- | --- |
| **$p = 8$** | 3917.8 ns | 1617.8 ns | **1234.1 ns** | **3.17× / 1.31×** | **912.6 ns** | **1.77×** (vs. $p=8$ Cart.) |
| **$p = 10$** | 9689.7 ns | 3937.1 ns | **2846.6 ns** | **3.40× / 1.38×** | **1281.1 ns** | **3.07×** (vs. $p=10$ Cart.) |
| **$p = 12$** | 20475.8 ns | 8359.4 ns | **5848.6 ns** | **3.50× / 1.43×** | **1940.2 ns** | **4.31×** (vs. $p=12$ Cart.) |
| **$p = 14$** | — | — | — | — | **2873.1 ns** | Faster than $p=10$ Cart.! |
| **$p = 16$** | — | — | — | — | **4118.6 ns** | Similar to $p=10$ Cart. |

---

## 3. Main Ablation Results on the 10 Public Snapshots ($\le 3.60\text{ CPU s}$ Budget)

### Table 1: Summary Across Variants (Mean & Per-Family RMS Relative Error and Worst-Case CPU Time)

| Variant | Binary / Flags | Chosen Parameters | Mean RMS Rel. Error (10 cases) | `smooth-hernquist` (`01..02`) | `jaffe` (`03..04`) | `einasto` (`05..06`) | `disc` (`07..08`) | `group` (`09..10`) | Worst-Case CPU (s) | Mean CPU (s) | Accuracy Gain vs. Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline (Frozen)** | `./solve` | Cartesian $p=8, \theta=0.48$, $(96,96,256)$ | **`5.914e-05`** | `9.327e-05` | `1.146e-04` | `5.404e-05` | `3.604e-06` | `3.020e-05` | 3.4727 | 2.9576 | **1.00×** (anchor) |
| **Variant A (Unrolled)** | `./solve-A` (`FMM_UNROLLED=1`) | Cartesian $p=8, \theta=0.45$, $(96,96,256)$ | **`3.771e-05`** | `6.397e-05` | `7.264e-05` | `3.179e-05` | `2.099e-06` | `1.807e-05` | 3.3914 | 2.8665 | **1.57×** |
| **Variant A+B (Unrolled + Autotune / Higher $p$)** | `./solve-AB` (`FMM_UNROLLED=1 FMM_AUTOTUNE=1`) | Adaptive per input: $p=8, \theta \in [0.42,0.45]$ (spheres/group); $p=9, \theta=0.36$ (`disc`) | **`3.377e-05`** *(or `3.110e-05` at uniform $p=9, \theta=0.49$)* | `6.397e-05` *(`4.905e-05`)* | `7.264e-05` *(`6.189e-05`)* | `2.138e-05` *(`2.821e-05`)* | `1.030e-07` *(`1.675e-06`)* | `1.075e-05` *(`1.468e-05`)* | 3.5552 *(3.5404)* | 3.4519 *(3.0631)* | **1.75×** *(**1.90×**)* |
| **Variant C (Kernel Split Only, Fixed $p=8$)** | `./solve-C` (`FMM_SPLIT=1`) | Split $p_{\text{harm}}=8, p_{\text{res}}=6, \theta=0.44$, $(96,96,256)$ | **`3.300e-05`** | `5.648e-05` | `6.466e-05` | `2.667e-05` | `1.779e-06` | `1.538e-05` | 3.1963 | 2.6710 | **1.79×** |
| **Variant A+B+C (Kernel Split + Unrolled + Autotune Higher $p_{\text{harm}}$)** | `./solve-ABC` (`FMM_SPLIT=1 FMM_UNROLLED=1 FMM_AUTOTUNE=1`) | Adaptive per input: $p_{\text{harm}}=12, p_{\text{res}}=6$, $\theta \in [0.50, 0.52]$ (spheres), $\theta=0.37$ (`disc`); $p_{\text{harm}}=10, \theta=0.44$ (`group`) | **`6.085e-06`** | **`8.587e-06`** | **`1.549e-05`** | **`4.052e-06`** | **`9.862e-08`** | **`2.182e-06`** | 3.4611 | 3.3505 | **9.72× (~10× overall; 36.5× on `disc`, 13.8× on `group`)** |

---

## 4. Per-Snapshot Breakdown for All Variants

### Table 2: Per-Snapshot RMS Relative Acceleration Error and CPU Time (s) on Leased Dev Slot

| Snapshot | Family | Baseline ($p=8, \theta=0.48$) Err / CPU | Variant A ($p=8, \theta=0.45$) Err / CPU | Variant A+B (Autotune $p \in \{8,9\}$) Err / CPU | Variant A+B (Uniform $p=9, \theta=0.49$) Err / CPU | Variant C ($p_{\text{harm}}=8, \theta=0.44$) Err / CPU | Variant A+B+C (Autotune $p_{\text{harm}} \in \{10,12\}$) Err / CPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `case-01` | `smooth-hernquist` | `9.448e-05` (3.43s) | `6.377e-05` (3.34s) | `6.377e-05` (3.47s, $p=8,\theta=0.45$) | `5.142e-05` (3.52s) | `5.610e-05` (3.13s) | **`7.794e-06`** (3.46s, $p=12,\theta=0.51$) |
| `case-02` | `smooth-hernquist` | `9.207e-05` (3.41s) | `6.416e-05` (3.33s) | `6.416e-05` (3.42s, $p=8,\theta=0.45$) | `4.669e-05` (3.51s) | `5.686e-05` (3.10s) | **`9.379e-06`** (3.43s, $p=12,\theta=0.51$) |
| `case-03` | `jaffe` | `1.138e-04` (3.47s) | `8.016e-05` (3.39s) | `8.016e-05` (3.49s, $p=8,\theta=0.45$) | `6.519e-05` (3.54s) | `7.269e-05` (3.20s) | **`1.720e-05`** (3.30s, $p=12,\theta=0.52$) |
| `case-04` | `jaffe` | `1.154e-04` (3.34s) | `6.512e-05` (3.27s) | `6.512e-05` (3.38s, $p=8,\theta=0.45$) | `5.860e-05` (3.46s) | `5.664e-05` (3.06s) | **`1.379e-05`** (3.35s, $p=12,\theta=0.51$) |
| `case-05` | `einasto` | `6.221e-05` (3.11s) | `3.592e-05` (3.01s) | `2.351e-05` (3.47s, $p=8,\theta=0.43$) | `3.267e-05` (3.24s) | `3.004e-05` (2.80s) | **`4.621e-06`** (3.32s, $p=12,\theta=0.50$) |
| `case-06` | `einasto` | `4.587e-05` (3.12s) | `2.765e-05` (3.02s) | `1.925e-05` (3.47s, $p=8,\theta=0.43$) | `2.376e-05` (3.26s) | `2.331e-05` (2.80s) | **`3.482e-06`** (3.35s, $p=12,\theta=0.50$) |
| `case-07` | `disc` | `3.737e-06` (1.78s) | `2.182e-06` (1.67s) | `1.066e-07` (3.36s, $p=9,\theta=0.36$) | `1.766e-06` (1.88s) | `1.882e-06` (1.51s) | **`9.535e-08`** (3.31s, $p=12,\theta=0.37$) |
| `case-08` | `disc` | `3.471e-06` (1.83s) | `2.015e-06` (1.67s) | `9.944e-08` (3.37s, $p=9,\theta=0.36$) | `1.585e-06` (1.88s) | `1.677e-06` (1.52s) | **`1.019e-07`** (3.34s, $p=12,\theta=0.37$) |
| `case-09` | `group` | `3.081e-05` (3.07s) | `1.817e-05` (2.98s) | `1.062e-05` (3.56s, $p=8,\theta=0.42$) | `1.524e-05` (3.20s) | `1.533e-05` (2.80s) | **`2.147e-06`** (3.37s, $p=10,\theta=0.44$) |
| `case-10` | `group` | `2.959e-05` (3.03s) | `1.797e-05` (2.98s) | `1.088e-05` (3.54s, $p=8,\theta=0.42$) | `1.412e-05` (3.16s) | `1.543e-05` (2.79s) | **`2.218e-06`** (3.37s, $p=10,\theta=0.44$) |
| **Mean / Max** | **All 10** | **`5.914e-05` / 3.47s** | **`3.771e-05` / 3.39s** | **`3.377e-05` / 3.56s** | **`3.110e-05` / 3.54s** | **`3.300e-05` / 3.20s** | **`6.085e-06` / 3.46s** |

*(Note: Under a slightly higher $3.9\text{ s}$ dev-slot budget with `FMM_SPLIT=1` at $p_{\text{harm}} = 16, p_{\text{res}} = 6, \theta = 0.60$, the mean error on the spherical/group cases drops further to `6.87e-06`, and on `disc` at $p_{\text{harm}}=10, \theta=0.32$ it reaches `3.48e-08` in $3.30\text{ s}$.)*

---

## 5. Where the Accuracy Gain Comes From: Interpretation of Literature vs. Kernel Splitting

1. **Why Pure Cartesian FMM Hits a Steep Wall Around $p = 8\text{--}9$ (Variants A and A+B)**:
   - Because the Plummer potential $\psi(r) = (r^2 + \epsilon^2)^{-1/2}$ is non-harmonic ($\nabla^2 \psi \ne 0$), a pure Cartesian FMM expansion of order $p$ must store and contract all $\binom{p+3}{3} = \frac{(p+1)(p+2)(p+3)}{6}$ symmetric tensor components, requiring $\binom{p+6}{6} \sim \mathcal{O}(p^6)$ FMA operations per M2L interaction (Dehnen 2014 Appendix A.1).
   - **Variant A (Engineering code generation)** speeds up the 4-way AVX2 M2L kernel by $1.31\times\text{--}1.43\times$ ($3.17\times\text{--}3.50\times$ over scalar) and replaces the $\mathcal{O}(p^4)$ Horner L2P shift with an $\mathcal{O}(p^3)$ monomial dot product. At fixed $p = 8$, this reduces runtime from $3.47\text{ s}$ to $2.99\text{ s}$, allowing $\theta$ to be tightened from $0.48$ to $0.45$ and reducing mean error by **$1.57\times$** (from $5.91 \times 10^{-5}$ to $3.77 \times 10^{-5}$).
   - **Variant B (Literature parameter selection + runtime cost model)** allows $p = 9$ within budget ($3.54\text{ s}$ at $\theta = 0.49$, reducing mean error to $3.11 \times 10^{-5}$, or **$1.90\times$** over baseline) and adapts $(p, \theta)$ per snapshot geometry so that faster snapshots (`disc`, which takes only $1.78\text{ s}$ at $\theta = 0.48$) automatically use $p = 9, \theta = 0.36$ ($3.37\text{ s}$), cutting `disc` error by **$35\times$** (from $3.60 \times 10^{-6}$ to $1.03 \times 10^{-7}$). However, on the heavy spherical snapshots (`smooth-hernquist` and `jaffe`), the $\binom{p+6}{6}$ Cartesian M2L cost grows by $2.3\times$ from $p=8$ ($1234\text{ ns}$) to $p=10$ ($2847\text{ ns}$) and $4.7\times$ to $p=12$ ($5849\text{ ns}$), forcing wider $\theta$ at $p \ge 10$ that cancels out the higher order. Thus standard literature Cartesian FMM techniques (A+B) provide a **$\sim 1.9\times$** overall improvement (and $35\times$ on `disc`), plateauing at $\sim 3.1 \times 10^{-5}$.

2. **Why Kernel Splitting Unlocks High Expansion Orders $p_{\text{harm}} = 10\text{--}16$ (Variants C and A+B+C)**:
   - At well-separated cell distances $r \gg \epsilon = 0.01$, the Plummer kernel $(r^2 + \epsilon^2)^{-1/2}$ differs from the harmonic Newtonian kernel $1/r$ by a smooth residual $\delta\psi(r) = -\frac{\epsilon^2}{2r^3} + \frac{3\epsilon^4}{8r^5} - \cdots$ whose magnitude is already suppressed by $\frac{1}{2}(\epsilon/r)^2 \lesssim 10^{-4}\text{--}10^{-3}$.
   - In pure Cartesian FMM, every high-order derivative $n = 7 \dots 12$ pays the full non-harmonic $\mathcal{O}(p^6)$ tax even though the non-harmonic part of those derivatives is negligible at $r \gg \epsilon$.
   - **Variant C (Kernel Splitting)** separates the two regimes:
     - The dominant $1/r$ field is harmonic and is expanded in solid spherical harmonics $U_n^m, T_n^m$ (Dehnen 2014 Section 2.1 & Appendix A.3), which have only $(p_{\text{harm}}+1)^2$ components and $\mathcal{O}(p_{\text{harm}}^4)$ (or $\mathcal{O}(p_{\text{harm}}^3)$ rotated) M2L complexity.
     - The tiny smooth residual $\delta\psi(r)$ only needs a low-order Cartesian expansion ($p_{\text{res}} = 6$, with $r > 8\epsilon$ core guard) to reach $< 10^{-7}$ relative accuracy.
   - At fixed order $p_{\text{harm}} = 8$ without A/B (**Variant C alone**), kernel splitting reduces M2L time from $1618\text{ ns}$ to $913\text{ ns}$, allowing $\theta = 0.44$ in $3.20\text{ s}$ and improving accuracy by **$1.79\times$** ($3.30 \times 10^{-5}$).
   - Crucially, when **Variant C is combined with A+B (`solve-ABC`)**, the $(p_{\text{harm}}+1)^2$ harmonic scaling makes $p_{\text{harm}} = 12$ ($1940\text{ ns}$ total split M2L) **$3.0\times$ faster** than unrolled Cartesian $p = 12$ ($5849\text{ ns}$). This allows the runtime cost model to select $p_{\text{harm}} = 12$ across the spherical and disc snapshots within $3.46\text{ s}$, reducing the mean RMS relative error to **$6.085 \times 10^{-6}$** (**$9.72\times$** overall improvement over baseline, including **$36.5\times$** on `disc` to $9.86 \times 10^{-8}$, **$13.8\times$** on `group` to $2.18 \times 10^{-6}$, **$13.3\times$** on `einasto` to $4.05 \times 10^{-6}$, and **$10.9\times$** on `smooth-hernquist` to $8.59 \times 10^{-6}$).
   - Finally, the remaining gap between $\sim 6 \times 10^{-6}$ (or $\sim 10^{-7}$ on `disc`/`group`) and $10^{-8}\text{--}4 \times 10^{-9}$ in $\sim 3.5\text{ s}$ comes from combining kernel splitting with the $\mathcal{O}(p^3)$ rotation-accelerated harmonic M2L of Dehnen (2014, Appendix A.4.2) or adaptive per-distance residual order $p_{\text{res}}(r/\epsilon)$ (skipping the $p_{\text{res}}$ multipole contraction entirely when $(\epsilon/r)^2 \theta^{p_{\text{res}}} \ll 10^{-9}$ and using higher $p_{\text{res}}$ when $r \sim 8\epsilon$).
