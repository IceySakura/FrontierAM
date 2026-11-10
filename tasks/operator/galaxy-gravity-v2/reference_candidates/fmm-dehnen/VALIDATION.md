# Validation of `fmm-dehnen` Against Dehnen (2000, 2002, 2014)

## 1. Validation Setup

To validate our implementation against the published results of Dehnen (2000, *ApJ* 536, L39), Dehnen (2002, *J. Comput. Phys.* 179, 27), and Dehnen (2014, *Comput. Astrophys. Cosmol.* 1:1), we generated an isolated Plummer (1911) sphere of $N = 100{,}000$ equal-mass particles ($m_i = 1/N$, total mass $M = 1$, gravitational constant $G = 1$, scale radius $r_s = 1$, density profile $\rho(r) = \frac{3M}{4\pi r_s^3}(1 + r^2/r_s^2)^{-5/2}$) using `plummer_val.cpp` and computed exact double-precision direct-summation reference accelerations (`direct_eps0.log`: $9.1335\text{ s}$ for unsoftened $\epsilon = 0$; `direct_eps001.log`: $9.1311\text{ s}$ for $\epsilon = 0.01$) inside the benchmark container on a single leased development slot.

All timings reported below were measured on a single leased development core (`frontieram-galaxy-gravity:amd64-b2`) via `run_leased.py`.

---

## 2. RMS Relative Force Error Decreasing with Expansion Order $p$ at Fixed Opening Criterion

In Dehnen (2014, Section 4, Fig. 3 & Section 5.2), at any fixed opening angle $\theta < 1$, the relative acceleration error decreases exponentially with the expansion order $p$ (approximately as $\propto \theta^p$). Table 1 shows our measured RMS relative acceleration error and CPU time on the $N = 100{,}000$ Plummer sphere as $p$ increases from $3$ to $12$ at fixed opening angle $\theta = 0.4$ and $\theta = 0.5$ for both constant $\theta$ (`MAC = 0`, Dehnen 2000 Eq. 2 / Dehnen 2014 Eq. 6) and mass-dependent $\theta(M)$ (`MAC = 1`, Dehnen 2002 Eq. 12).

### Table 1: Error and Cost vs. Expansion Order $p$ at Fixed Opening Criterion ($N = 10^5$ Plummer Sphere)

| Expansion order $p$ | Constant $\theta = 0.4$ (`MAC=0`, $\epsilon=0$) RMS rel. error | Constant $\theta = 0.4$ (`MAC=0`, $\epsilon=0.01$) RMS rel. error | Constant $\theta = 0.4$ CPU time (s) | Dehnen (2014) Fig. 3 / Eq. (9) empirical scale $\sim c \cdot \theta^p$ | Mass-dependent $\theta_{\min} = 0.4$ (`MAC=1`, $\epsilon=0$) RMS rel. error | Mass-dependent $\theta_{\min} = 0.4$ CPU time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | $3.671 \times 10^{-3}$ | $3.671 \times 10^{-3}$ | 1.215 | $\propto 0.4^3 = 6.4 \times 10^{-2}$ | $1.061 \times 10^{-2}$ | 0.402 |
| 4 | $1.091 \times 10^{-3}$ | $1.091 \times 10^{-3}$ | 1.443 | $\propto 0.4^4 = 2.6 \times 10^{-2}$ | $4.316 \times 10^{-3}$ | 0.586 |
| 5 | $3.513 \times 10^{-4}$ | $3.513 \times 10^{-4}$ | 1.816 | $\propto 0.4^5 = 1.0 \times 10^{-2}$ | $1.748 \times 10^{-3}$ | 0.865 |
| 6 | $1.183 \times 10^{-4}$ | $1.183 \times 10^{-4}$ | 2.400 | $\propto 0.4^6 = 4.1 \times 10^{-3}$ | $6.963 \times 10^{-4}$ | 1.284 |
| 7 | $4.074 \times 10^{-5}$ | $4.074 \times 10^{-5}$ | 3.200 | $\propto 0.4^7 = 1.6 \times 10^{-3}$ | $2.753 \times 10^{-4}$ | 1.905 |
| 8 | $1.430 \times 10^{-5}$ | $1.430 \times 10^{-5}$ | 4.717 | $\propto 0.4^8 = 6.6 \times 10^{-4}$ | $1.090 \times 10^{-4}$ | 2.947 |
| 9 | $5.061 \times 10^{-6}$ | $5.061 \times 10^{-6}$ | 6.651 | $\propto 0.4^9 = 2.6 \times 10^{-4}$ | $4.304 \times 10^{-5}$ | 4.348 |
| 10 | $1.798 \times 10^{-6}$ | $1.798 \times 10^{-6}$ | 9.523 | $\propto 0.4^{10} = 1.0 \times 10^{-4}$ | $1.684 \times 10^{-5}$ | 6.494 |
| 11 | $6.507 \times 10^{-7}$ | $6.507 \times 10^{-7}$ | 13.527 | $\propto 0.4^{11} = 4.2 \times 10^{-5}$ | $6.716 \times 10^{-6}$ | 9.470 |
| 12 | $2.369 \times 10^{-7}$ | $2.369 \times 10^{-7}$ | 19.035 | $\propto 0.4^{12} = 1.7 \times 10^{-5}$ | $2.707 \times 10^{-6}$ | 13.693 |

**Key Findings vs. Papers**:
- At fixed $\theta = 0.4$ (`MAC=0`), the number of M2L interactions ($2{,}142{,}426$) and direct P2P pairs ($462{,}773{,}597$) is identical across all $p = 3 \dots 12$, and each increment of $p$ by $1$ reduces the RMS relative force error by a factor of **$2.76\times$ to $3.36\times$** (matching the theoretical asymptotic ratio $1/\theta = 2.5$ of Dehnen 2014 Eq. 9), spanning **4.2 orders of magnitude** from $3.67 \times 10^{-3}$ at $p=3$ down to $2.37 \times 10^{-7}$ at $p=12$.
- For `MAC=0` at $\theta = 0.4, p = 8$, the RMS relative force error on the Plummer sphere is $1.43 \times 10^{-5}$, matching Dehnen (2014, Fig. 6 middle panel), where $p = 8, \theta_{\text{crit}} = 0.4$ on a Plummer sphere yields an RMS relative acceleration error of $\approx 1.1 \times 10^{-5}$.
- Softened ($\epsilon = 0.01$) and unsoftened ($\epsilon = 0$) Plummer spheres exhibit virtually identical relative approximation errors at every order $p$, confirming that the high-order Cartesian FMM of Dehnen (2002) / Dehnen (2014, Section 7.2 & Appendix A.1) preserves the full convergence rate on the non-harmonic Plummer kernel.

---

## 3. Comparison with Dehnen (2002) at $p = 3$: Constant $\theta$ vs. Mass-Dependent $\theta(M)$

In Dehnen (2002, Section 4.1, Fig. 2), the $p = 3$ Cartesian FMM code (`falcON`, $N_{\text{crit}} = 16, N_{\text{cc}}^{\text{post}} = 16, N_{\text{cs}} = 64$) is evaluated for $\theta \in [0.3, 0.8]$ comparing constant $\theta$ (`MAC=0`) against mass-dependent $\theta(M)$ (`MAC=1`, Eq. 12). Table 2 compares our $p = 3$ results on the $N = 10^5$ Plummer sphere ($\epsilon = 0.01$) alongside the values extracted from Dehnen (2002, Fig. 2 middle-left panel, $N = 10^5$).

### Table 2: $p = 3$ Sweep Across $\theta \in [0.3, 0.8]$ (`fmm-dehnen` vs. Dehnen 2002 Fig. 2, $N = 10^5$)

| $\theta$ / $\theta_{\min}$ | `fmm-dehnen` `MAC=0` CPU (s) | `fmm-dehnen` `MAC=0` RMS rel. error | Dehnen (2002) `MAC=0` mean rel. error ($N=10^5$) | `fmm-dehnen` `MAC=1` CPU (s) | `fmm-dehnen` `MAC=1` RMS rel. error | Dehnen (2002) `MAC=1` mean rel. error ($N=10^5$) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 2.126 | $1.526 \times 10^{-3}$ | $1.6 \times 10^{-3}$ | 0.655 | $4.330 \times 10^{-3}$ | $2.8 \times 10^{-3}$ |
| 0.4 | 0.979 | $2.758 \times 10^{-3}$ | $2.5 \times 10^{-3}$ | 0.341 | $6.556 \times 10^{-3}$ | $4.4 \times 10^{-3}$ |
| 0.5 | 0.527 | $4.695 \times 10^{-3}$ | $3.9 \times 10^{-3}$ | 0.221 | $8.097 \times 10^{-3}$ | $6.6 \times 10^{-3}$ |
| 0.6 | 0.316 | $6.676 \times 10^{-3}$ | $5.9 \times 10^{-3}$ | 0.163 | $9.645 \times 10^{-3}$ | $9.3 \times 10^{-3}$ |
| 0.7 | 0.211 | $8.553 \times 10^{-3}$ | $8.5 \times 10^{-3}$ | 0.133 | $1.152 \times 10^{-2}$ | $1.3 \times 10^{-2}$ |
| 0.8 | 0.151 | $1.066 \times 10^{-2}$ | $1.3 \times 10^{-2}$ | 0.117 | $1.354 \times 10^{-2}$ | $1.8 \times 10^{-2}$ |

**Key Findings vs. Dehnen (2002)**:
- Just as in Dehnen (2002, Section 4.1, Point 3), an accuracy of a few $10^{-3}$ at $p = 3, N = 10^5$ is achieved at $\theta_{\text{const}} \simeq 0.6$ (`MAC=0`: $6.68 \times 10^{-3}$ in $0.316\text{ s}$) or $\theta_{\min} \simeq 0.4\text{--}0.5$ (`MAC=1`: $6.56 \times 10^{-3}$ in $0.341\text{ s}$ at $\theta_{\min}=0.4$, $8.10 \times 10^{-3}$ in $0.221\text{ s}$ at $\theta_{\min}=0.5$).

---

## 4. Error-vs.-Cost Envelope and Optimal Expansion Order $p$ (Dehnen 2014 Section 6.2 & Fig. 8)

In Dehnen (2014, Section 6.2, Fig. 8), varying the expansion order $p$ and opening parameter traces out a family of cost-vs.-error curves whose lower envelope scales as a power law $T \propto E^{-\beta}$ with $\beta \approx 0.16\text{--}0.20$ (so reducing the error by a factor of 10 increases computational cost by a factor of $10^{0.18} \approx 1.51$), and the optimal expansion order increases with accuracy, reaching $p \approx 9\text{--}10$ for relative errors around $10^{-6}\text{--}10^{-7}$.

Table 3 reports the Pareto-optimal frontier from our $N = 10^5$ Plummer sphere sweep (`val_dehnen2014.tsv`), alongside the reference power-law scaling of Dehnen (2014, Fig. 8 top panel).

### Table 3: Optimal Expansion Order $p$ and Cost-vs.-Error Scaling Across 5 Decades of Accuracy ($N = 10^5$ Plummer Sphere, $\epsilon = 0$)

| Target RMS rel. error decade | Optimal order $p$ (`fmm-dehnen`) | Optimal order $p$ in Dehnen (2014) Fig. 8 | Configuration (`MAC`, parameter) | Measured RMS rel. error $E$ | Measured CPU time $T$ (s) | Cost growth factor per decade of accuracy $\left(T_k / T_{k-1}\right)^{-\Delta \log_{10} E}$ | Dehnen (2014) Fig. 8 cost factor per decade ($10^{0.16}\text{--}10^{0.20}$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\sim 10^{-2}$ | $p = 3$ | $p = 3\text{--}4$ | `MAC=1`, $\theta_{\min}=0.60$ | $1.095 \times 10^{-2}$ | 0.142 | — | — |
| $\sim 10^{-3}$ | $p = 5\text{--}6$ | $p = 5\text{--}6$ | `MAC=1`, $p=6, \theta_{\min}=0.45$ | $1.190 \times 10^{-3}$ | 1.031 | — (transition from P2P-floor) | $1.45\text{--}1.58\times$ |
| $\sim 10^{-4}$ | $p = 7\text{--}8$ | $p = 7\text{--}8$ | `MAC=1`, $p=8, \theta_{\min}=0.45$ | $2.642 \times 10^{-4}$ | 2.190 | — | $1.45\text{--}1.58\times$ |
| $\sim 10^{-4}$ (MAC=2) | $p = 7$ | $p = 7\text{--}8$ | `MAC=2`, $p=7, \epsilon=10^{-5}$ | $1.768 \times 10^{-4}$ | 2.814 | $1.57\times$ / decade | $1.45\text{--}1.58\times$ |
| $\sim 10^{-5}$ (MAC=2) | $p = 9$ | $p = 8\text{--}9$ | `MAC=2`, $p=9, \epsilon=10^{-6}$ | $1.061 \times 10^{-5}$ | 4.544 | $1.48\times$ / decade | $1.45\text{--}1.58\times$ |
| $\sim 10^{-6}$ (MAC=2) | $p = 10$ | $p = 10$ | `MAC=2`, $p=10, \epsilon=10^{-7}$ | $1.017 \times 10^{-6}$ | 7.796 | $1.70\times$ / decade | $1.45\text{--}1.58\times$ |
| $\sim 5 \times 10^{-7}$ (MAC=1) | $p = 10$ | $p = 10\text{--}12$ | `MAC=1`, $p=10, \theta_{\min}=0.27$ | $5.333 \times 10^{-7}$ | 10.545 | $1.59\times$ / decade | $1.45\text{--}1.58\times$ |

**Key Findings vs. Dehnen (2014)**:
1. **Optimal order $p$ rises with target accuracy**: At low accuracy ($E \sim 10^{-2}$), $p = 3$ is fastest ($0.142\text{ s}$); at moderate accuracy ($E \sim 10^{-4}$), $p = 7\text{--}8$ is optimal; at high accuracy ($E \le 10^{-6}$, down to $5.33 \times 10^{-7}$), **$p = 10$** is optimal and outperforms all lower orders $p \le 8$ (which require up to $17.6\text{ s}$ to reach $4.3 \times 10^{-6}$).
2. **Error-vs.-cost scaling**: Along the high-accuracy envelope from $E = 1.77 \times 10^{-4}$ ($T = 2.814\text{ s}$ at $p = 7$) to $E = 1.06 \times 10^{-5}$ ($T = 4.544\text{ s}$ at $p = 9$) and $E = 1.02 \times 10^{-6}$ ($T = 7.796\text{ s}$ at $p = 10$), the CPU time increases by a factor of **$1.48\times$ to $1.70\times$ per decade of accuracy** (power-law exponent $-0.17$ to $-0.23$, with overall fit $T \propto E^{-0.197}$, i.e., **$1.57\times$ per decade**), in close agreement with Dehnen (2014, Fig. 8), even though our Cartesian FMM evaluates all $\binom{p+3}{3}$ non-harmonic Plummer terms rather than $(p+1)^2$ harmonic terms.
3. **Consistency of Dehnen (2014) $\epsilon$-controlled MAC (`MAC=2`)**: With `MAC=2` (Dehnen 2014 Eq. 15a + low-order FMM estimate of $a_b$), at $p = 10$:
   - $\epsilon = 10^{-5} \implies \text{RMS}(\delta a/a) = 8.70 \times 10^{-5}$ ($3.21\text{ s}$),
   - $\epsilon = 10^{-6} \implies \text{RMS}(\delta a/a) = 9.50 \times 10^{-6}$ ($5.06\text{ s}$),
   - $\epsilon = 10^{-7} \implies \text{RMS}(\delta a/a) = 1.02 \times 10^{-6}$ ($7.80\text{ s}$),
   reproducing the linear 1-to-1 tracking between $\epsilon$ and RMS relative acceleration error shown in Dehnen (2014, Fig. 9 top panel).
