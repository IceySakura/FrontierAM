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
