# Dataset-v2 truth computation

Implemented in `truth.py`; run as `truth.py MANIFEST --jobs N --cpus LIST`. Scored cases whose
`truth_status` is not `chebyshev_audited_qualified` must not be used for correctness or scoring.
Analytic controls keep their dense-eigendecomposition truth.

## Method

* **Primary:** `exp(-t x)` and `t*phi1(-t x)` are expanded separately in Chebyshev series on
  `[0, ||L||_inf]`. Coefficients come from Chebyshev-point samples; the series is truncated where the
  coefficient envelope drops below `1e-3` of the request's acceptance threshold. The Clenshaw recurrence
  and the sparse products run in x87 extended precision (eps ~1.1e-19).
* **Audit:** Al-Mohy–Higham truncated Taylor (`scipy.sparse.linalg.expm_multiply`, binary64) on the
  augmented operator `t*[[-L, b1], [0, 0]]` applied to `[b0; 1]`.
* **Qualification:** for every request, `||primary - audit||_inf <= 0.02*(atol + rtol*||y||_inf)`.
* **Arbitration:** when the audit disagrees and `n <= 6000`, a dense eigendecomposition decides. The case
  qualifies if the primary is within the same `0.02` of it. Otherwise the case stays unqualified.

## Approaches that were rejected

* `y = b0 + phi1(L)(b1 - L b0)` cancels catastrophically at large `t*lambda`.
* A binary64 Chebyshev primary reached `0.108` of the threshold against dense eigh at `eta = 1e5`, p10:
  rounding is amplified by `max f = t`.
* A Lanczos audit stagnated on stiff operators, with errors 2.7x and 13x the threshold.

## Public smoke result (48 scored cases)

* 48/48 qualified. The worst primary–audit disagreement on 47 cases is within 0.02.
* `syn_logspectrum_p10_k16`, request 14: the audit is off by 0.237 of the threshold, while the primary
  matches dense eigh to 0.000 (case-wide arbitration 0.0064). The binary64 Taylor audit is the inaccurate
  side on this ill-conditioned operator.
* Cost: 600 CPU-s primary+audit for all 48 cases (max Chebyshev degree 1827), plus 58 CPU-s arbitration.
  Extended precision is roughly 5x the binary64 planning estimate (1.9 CPU-min primary).

## Full dataset

The binary64 planning budget was 20–60 CPU-h primary plus 20–80 CPU-h audit. With the measured
extended-precision factor, plan for about 100–300 CPU-h primary. Operators above `n = 6000` have no
dense arbitration, so an audit disagreement there removes the case from scoring. The count of such
cases must be reported with the release.
