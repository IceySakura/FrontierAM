# Parameter Calibration Protocol and Results for `galaxy-gravity-v2` (`fmm-dehnen`)

## 1. Pre-Declared Parameter-Selection Rule

Before evaluating the 10 public benchmark snapshots (`case-01` .. `case-10`), we declared our parameter-selection rule:

1. **Candidate Space**:
   We consider configurations of our Cartesian Fast Multipole Method (`fmm-dehnen`) parameterized by:
   - Expansion order $p \in \{4, 5, 6, 7, 8, 9, 10\}$;
   - Multipole-acceptance criterion $\text{MAC} \in \{0\text{ (constant }\theta\text{, Dehnen 2000 Eq. 2)}, 1\text{ (mass-dependent }\theta(M)\text{, Dehnen 2002 Eq. 12)}, 2\text{ (multipole-power error estimator, Dehnen 2014 Eq. 15a)}\}$;
   - Tolerance parameter $\theta_0$ (for $\text{MAC} \in \{0, 1\}$) or relative error parameter $\epsilon$ (for $\text{MAC} = 2$);
   - Tree/direct-summation crossover thresholds $(N_{\text{crit}}, N_{\text{cc}}^{\text{post}}, N_{\text{cs}}) \in \{(32, 32, 96), (48, 48, 128), (64, 64, 192), (96, 96, 256)\}$ (following Dehnen 2002 Appendix B and `falcON` `default.h`).

2. **Timing Budget Constraint**:
   - The formal grading limit is $5.000$ CPU seconds per call on the grading cores.
   - As required by the benchmark specification, we enforce at least a **25% safety margin** on a leased development slot:
     $$\max_{c \in \{\text{case-01}, \dots, \text{case-10}\}} T_{\text{CPU}}(c) \le 3.600\text{ CPU seconds},$$
     where $T_{\text{CPU}}(c)$ is the full process CPU time (including start-up, binary file read, tree construction, upward pass, dual tree walk, downward pass, and binary file write) measured inside the `frontieram-galaxy-gravity:amd64-b2` container on a single leased development core.

3. **Objective Function**:
   - Among all candidate configurations that satisfy $\max_{c} T_{\text{CPU}}(c) \le 3.600\text{ s}$ across all 10 public snapshots (2 snapshots each of the 5 galaxy families: `smooth-hernquist`, `jaffe`, `einasto`, `disc`, `group`), we select and freeze the single configuration that minimizes the mean RMS relative acceleration error across the 10 public snapshots:
     $$E_{\text{pub}} = \frac{1}{10} \sum_{k=1}^{10} e(\text{case-}k), \qquad e = \sqrt{\frac{1}{N}\sum_{i=1}^N \frac{\|\hat{\boldsymbol{a}}_i - \boldsymbol{a}_i^{\text{ref}}\|_2^2}{\|\boldsymbol{a}_i^{\text{ref}}\|_2^2}}.$$
   - Every call uses fixed uniform parameters across all snapshots without snapshot-specific tuning or cross-call state.

---

## 2. Summary of Calibration Sweep on the 10 Public Snapshots

All runs were executed inside the `frontieram-galaxy-gravity:amd64-b2` container on a single leased development core via `scripts/ops/run_leased.py`. Table 1 summarizes the top candidate configurations across `MAC = 0`, `MAC = 1`, and `MAC = 2` that satisfy the $\max_{c} T_{\text{CPU}}(c) \le 3.600\text{ s}$ constraint (plus neighboring configurations just above $3.600\text{ s}$ for context).

### Table 1: Candidate Configurations Evaluated on the 10 Public Snapshots ($N = 131{,}072, \epsilon = 0.01$)

| Rank ($\le 3.6\text{ s}$) | MAC | Order $p$ | Parameter ($\theta$ or $\epsilon$) | $(N_{\text{crit}}, N_{\text{cc}}^{\text{post}}, N_{\text{cs}})$ | Mean RMS Rel. Error $E_{\text{pub}}$ | Max CPU Time (s) | Mean CPU Time (s) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1 (Frozen)** | **0 (`const_theta`)** | **8** | **0.48** | **(96, 96, 256)** | **$5.914 \times 10^{-5}$** | **3.5003** | **2.9846** | **Selected & Frozen** |
| 2 | 0 (`const_theta`) | 7 | 0.43 | (96, 96, 256) | $7.094 \times 10^{-5}$ | 3.4492 | 2.8753 | Feasible ($\le 3.6\text{ s}$) |
| 3 | 0 (`const_theta`) | 7 | 0.44 | (64, 64, 192) | $7.779 \times 10^{-5}$ | 3.4975 | 3.0850 | Feasible ($\le 3.6\text{ s}$) |
| 4 | 1 (`theta(M)`) | 8 | 0.39 | (96, 96, 256) | $8.094 \times 10^{-5}$ | 3.4611 | 2.9706 | Best `MAC=1` ($\le 3.6\text{ s}$) |
| 5 | 0 (`const_theta`) | 8 | 0.50 | (96, 96, 256) | $8.283 \times 10^{-5}$ | 3.1773 | 2.7208 | Feasible ($\le 3.6\text{ s}$) |
| 6 | 0 (`const_theta`) | 7 | 0.45 | (64, 64, 192) | $8.754 \times 10^{-5}$ | 3.3227 | 2.9335 | Feasible ($\le 3.6\text{ s}$) |
| 7 | 0 (`const_theta`) | 9 | 0.54 | (96, 96, 256) | $8.859 \times 10^{-5}$ | 3.3584 | 2.9564 | Feasible ($\le 3.6\text{ s}$) |
| 8 | 1 (`theta(M)`) | 9 | 0.44 | (96, 96, 256) | $9.129 \times 10^{-5}$ | 3.5888 | 3.1541 | Feasible ($\le 3.6\text{ s}$) |
| 9 | 1 (`theta(M)`) | 7 | 0.34 | (96, 96, 256) | $9.546 \times 10^{-5}$ | 3.3467 | 2.8082 | Feasible ($\le 3.6\text{ s}$) |
| 10 | 1 (`theta(M)`) | 8 | 0.41 | (64, 64, 192) | $1.176 \times 10^{-4}$ | 3.5815 | 3.2635 | Feasible ($\le 3.6\text{ s}$) |
| 11 | 1 (`theta(M)`) | 8 | 0.41 | (96, 96, 256) | $1.199 \times 10^{-4}$ | 3.0586 | 2.6407 | Feasible ($\le 3.6\text{ s}$) |
| 12 | 1 (`theta(M)`) | 5 | 0.25 | (48, 48, 128) | $2.115 \times 10^{-4}$ | 3.5503 | 2.9898 | Feasible ($\le 3.6\text{ s}$) |
| 13 | 2 (`Dehnen2014`) | 9 | $10^{-5}$ | (96, 96, 256) | $2.179 \times 10^{-4}$ | 3.5295 | 3.1487 | Best `MAC=2` ($\le 3.6\text{ s}$) |
| — | 0 (`const_theta`) | 8 | 0.47 | (96, 96, 256) | $5.030 \times 10^{-5}$ | 3.6631 | 3.1200 | Exceeds $3.6\text{ s}$ ($3.66\text{ s}$) |
| — | 1 (`theta(M)`) | 8 | 0.38 | (96, 96, 256) | $6.617 \times 10^{-5}$ | 3.6531 | 3.1366 | Exceeds $3.6\text{ s}$ ($3.65\text{ s}$) |

### Why `MAC = 0` ($p = 8, \theta = 0.48$) Outperforms `MAC = 1` and `MAC = 2` on the Benchmark Suite

- **Opening angle near theoretical optimum**: In Dehnen (2014, Section 5.2, unnumbered equation on line 909), minimizing computational cost $p^\alpha / \theta_{\text{crit}}^3$ at fixed target error $\theta_{\text{crit}}^p = \epsilon$ yields an optimal constant opening angle of $\theta_{\text{crit}} = e^{-\alpha/3} \approx 0.46\text{--}0.48$ independent of $p$. Our empirically selected $\theta = 0.48$ at $p = 8$ coincides with this theoretical optimum.
- **Steep central cusp in `jaffe` snapshots (`case-03`, `case-04`)**: The Jaffe model has an inner density profile $\rho(r) \propto r^{-2}$ where a large fraction of particles reside at very small radii ($r \lesssim \epsilon = 0.01$) with tiny cell sizes and small cell masses $M_A \ll M_{\text{tot}}$. Both `MAC = 1` (which relaxes $\theta(M) \propto (M/M_{\text{tot}})^{-1/(3(p+2))}$ for small-mass cells) and `MAC = 2` (which relaxes opening when the multipole power $\mathcal{P}_{k,A} \ll M_A r_{\max,A}^k$, where non-harmonic Plummer softening derivatives at $r \sim \epsilon$ exceed the unsoftened harmonic multipole power bound) allow wider opening angles in the dense Jaffe core, increasing the error on `case-03` and `case-04`. Constant $\theta = 0.48$ (`MAC = 0`) uniformly enforces $\|\boldsymbol{z}_A - \boldsymbol{z}_B\|_2 > (r_{\max,A} + r_{\max,B}) / 0.48$ across all scales.

---

## 3. Frozen Configuration and 3-Repetition Verification on Leased Dev Slot

The frozen configuration compiled as defaults into `solve.cpp` is:
- `order = 8` ($p = 8$)
- `mac = MacType::ConstTheta` (`MAC = 0`, Dehnen 2000 Eq. 2)
- `theta = 0.48`
- `eps = 0.01`
- `ncrit = 96`, `n_cb_pre = 16`, `n_cb_post = 256`, `n_cc_post = 96`, `n_cs = 256`

Table 2 reports the per-snapshot RMS relative acceleration error and full process CPU time (user + sys, including binary start-up, file I/O, tree build, FMM passes, and output write) across 3 independent repetitions of `./solve INPUT OUTPUT` on a leased development slot.

### Table 2: Per-Snapshot Performance of Frozen `./solve` on the 10 Public Snapshots

| Snapshot | Galaxy Family | RMS Relative Acceleration Error $e$ | CPU Time Rep 1 (s) | CPU Time Rep 2 (s) | CPU Time Rep 3 (s) | Mean CPU Time (s) | Max CPU Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `case-01` | `smooth-hernquist` | $9.447564 \times 10^{-5}$ | 3.4314 | 3.4141 | 3.4445 | 3.4300 | 3.4445 |
| `case-02` | `smooth-hernquist` | $9.206611 \times 10^{-5}$ | 3.4014 | 3.4104 | 3.4288 | 3.4135 | 3.4288 |
| `case-03` | `jaffe` | $1.137683 \times 10^{-4}$ | 3.4628 | 3.4727 | 3.4678 | 3.4678 | **3.4727** |
| `case-04` | `jaffe` | $1.154244 \times 10^{-4}$ | 3.3428 | 3.3296 | 3.3361 | 3.3362 | 3.3428 |
| `case-05` | `einasto` | $6.221391 \times 10^{-5}$ | 3.1055 | 3.1054 | 3.1166 | 3.1092 | 3.1166 |
| `case-06` | `einasto` | $4.586642 \times 10^{-5}$ | 3.1152 | 3.1139 | 3.1288 | 3.1193 | 3.1288 |
| `case-07` | `disc` | $3.736836 \times 10^{-6}$ | 1.7720 | 1.7767 | 1.7774 | 1.7754 | 1.7774 |
| `case-08` | `disc` | $3.471492 \times 10^{-6}$ | 1.7906 | 1.7971 | 1.8953 | 1.8277 | 1.8953 |
| `case-09` | `group` | $3.081097 \times 10^{-5}$ | 3.0694 | 3.0698 | 3.0651 | 3.0681 | 3.0698 |
| `case-10` | `group` | $2.958994 \times 10^{-5}$ | 3.0329 | 3.0301 | 3.0244 | 3.0291 | 3.0329 |
| **Overall** | **5 families (10 cases)** | **Mean $E_{\text{pub}} = 5.914240 \times 10^{-5}$** | **Max 3.4628** | **Max 3.4727** | **Max 3.4678** | **Mean 2.9576** | **Max 3.4727** |

### Family-Mean RMS Relative Errors:
- `smooth-hernquist` (`case-01`, `case-02`): $9.327088 \times 10^{-5}$
- `jaffe` (`case-03`, `case-04`): $1.145964 \times 10^{-4}$
- `einasto` (`case-05`, `case-06`): $5.404017 \times 10^{-5}$
- `disc` (`case-07`, `case-08`): $3.604164 \times 10^{-6}$
- `group` (`case-09`, `case-10`): $3.020045 \times 10^{-5}$
- **Equal-weighted 5-family mean error $E_{\text{pub}}$**: **$5.914240 \times 10^{-5}$**
- **Worst-case CPU time across all 30 calls**: **$3.4727\text{ CPU s}$** ($\le 3.600\text{ CPU s}$ target, providing a **30.5% margin** below the $5.000\text{ s}$ limit).
