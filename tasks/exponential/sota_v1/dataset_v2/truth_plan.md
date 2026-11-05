# Dataset-v2 truth computation plan

Scored files currently contain finite zero placeholders in `y_ref`; they are
explicitly **unqualified** and must not be used for correctness or scoring.
Analytic controls already contain qualified dense-eigendecomposition truth.

## Primary computation

Use 256-bit MPFR/Arb coefficients for both `exp(-t x)` and
`(1-exp(-t x))/x` on `[0, lambda_upper]`, where `lambda_upper` is the smaller
of the infinity norm and a safety-inflated verified Lanczos upper bound.
Choose the Chebyshev tail below `1e-3` of the p10 acceptance threshold and use
one compensated binary64/long-double Clenshaw recurrence for both functions.
Reuse sessions use block SpMM over all 16 right-hand sides.

## Independent audit

Audit every case with restarted high-accuracy Lanczos/KIOPS at `1e-3*rtol`.
For `eta >= 1e4`, add rational Krylov or scaling-and-Taylor augmented action.
Audit controls and sampled principal subproblems with arbitrary precision dense
eigendecomposition. Require infinity-norm disagreement no greater than
`0.02*(atol + rtol*||y||inf)`; otherwise classify the case as evaluator error.

## Cost model

The generator estimates degree as
`ceil(2*sqrt(eta*log(2/(atol+rtol))))`, sparse work as
`2*nnz*degree*requests`, and CPU time at a conservative 50 million sparse
floating operations/second. This is a planning estimate, not a timing result.

* Cases estimated: 48
* Primary total: 1.9 CPU-minutes (0.03 CPU-hours)
* Hardest case: `syn_signed_p10_k16`, degree 3081,
  0.4 CPU-minutes
* Independent audit budget: 1–2x primary for ordinary cases and 2–3x for
  `eta >= 1e4`.

Machine-readable per-case estimates are in `truth_costs.json`.
