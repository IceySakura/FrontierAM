# Public calibration protocol (declared before the first run)

The implementation has no case-family switch and no fitted numerical constant.
Before running any of the 29 public development cases, the policy was frozen as:

1. use the rounded-up PSD Gershgorin interval, without Lanczos tightening;
2. generate scaled-Bessel coefficients through index
   `ceil(12*sqrt(a+1)+128)`, where `a=t*lambda_upper/2`;
3. retain the smallest degree whose absolute coefficient tail, multiplied by
   `||b1-L*b0||_2`, is no more than one eighth of
   `atol + rtol*max(||b0||_inf,t*||b1||_inf)`;
4. use binary64 vectors, long-double coefficient construction, and no
   case-specific exceptions except the exact mathematical degeneracies `t=0`,
   `L=0`, and `b1-L*b0=0`.

The factor 12 places the Bessel cutoff beyond the Gaussian large-argument tail
needed for binary64; 128 protects the small/intermediate-argument transition.
The factor 8 is a declared safety split between polynomial truncation and
floating-point recurrence/SpMV error. These are analytic safety choices, not
timing fits. Public validation may reveal a defect and motivate a documented
uniform correction, but it may not select from a parameter grid by timing.

## Correction 1 — 2026-11-05

Dataset-v2 public smoke validation revealed that the original input-relative
degree rule was not sufficient when `exp(-tL)` strongly damps the input. In
particular, `syn_heavytail_p7_k1` and `syn_heavytail_p7_k16` exceeded the
acceptance threshold even though their coefficient tails met the preregistered
rule. The formulation `b0 + g(L)(b1-L*b0)` also lost accuracy through
cancellation on tighter, strongly damped public cases.

The correction is uniform and analytic:

1. evaluate `exp(-tL)b0` and `g(L)b1`, where
   `g(x)=t*phi_1(-t*x)`, as separate Chebyshev actions;
2. evaluate both series by matrix Clenshaw recurrence;
3. initially choose the smallest common degree `d` for which
   `F_tail[d+1]*||b0||_2 + G_tail[d+1]*||b1||_2` is at most one eighth of
   `atol + rtol*max(||b0||_inf,t*||b1||_inf)`;
4. after forming `y_d`, compute the combined omitted-tail bound `E_d` and use
   `max(||y_d||_inf-E_d,0)` as a certified lower bound on the exact output
   norm. If the same tail rule with this output scale requires a larger
   degree, recompute both Clenshaw actions at that degree and repeat.

This is conservative because `||y||_inf >= ||y_d||_inf-E_d`. It has no
case-family switch, fitted threshold, timing measurement, or RHS-dependent
state reuse. The exact zero-operator branch was also repaired by preserving a
zero Gershgorin bound as exactly zero rather than rounding it to the smallest
positive subnormal.

Required one-repeat, low-priority native validation after rebuilding both
binaries:

- dataset-v2 public smoke: 56/56 cases valid; maximum scaled error
  `0.11312275805683136`;
- v1 public development set: 29/29 cases valid; maximum scaled error
  `0.07536607430372355`.

No parameter grid or timing comparison was used to select this correction.
