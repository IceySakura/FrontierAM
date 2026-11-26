import Theory.Defs

/-!
# Statement layer

The formal statements of the results R1–R5 of `THEORY.md` §3, each with the proof `sorry`. The definitions used
(`AdaptiveAlg`, `AdaptiveAlg.info`, `NonadaptiveAlg`, `NonadaptiveAlg.info`, `StepClass`, `IntervalClass`) are in
`Theory/Defs.lean`. `Theory/Proofs.lean` repeats every statement verbatim with its proof.

Setting: `γ = gaussianReal 0 1` is the standard Gaussian measure on `ℝ`; the integral of `f` is
`∫ t, f t ∂(gaussianReal 0 1)`; an algorithm `A` with `n` function values estimates it by `A.output (A.info f)`, so
its error on `f` is `|A.output (A.info f) - ∫ t, f t ∂(gaussianReal 0 1)|`.

* `J = StepClass` (one crossing): R1, R2 (adaptive, exact value `2^{-(n+1)}`), R3 (nonadaptive, exact value
  `1/(2(n+1))`), R5 (randomized lower bound).
* `P = IntervalClass` (two crossings): R4 (adaptive lower bound `1/(2(n+1))`, nonadaptive upper bound `1/(n+1)`).
-/

open MeasureTheory ProbabilityTheory

namespace Cubature

/-- **R1 (adaptive lower bound, one crossing).** For every deterministic adaptive algorithm that uses `n` function
values there is a step function `f ∈ J` on which its error is at least `2^{-(n+1)}`. -/
theorem adaptive_lower_bound_step (n : ℕ) (A : AdaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 : ℝ) ^ (n + 1) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  sorry

/-- **R2 (bisection in Gaussian quantiles, one crossing).** There is a deterministic adaptive algorithm that uses `n`
function values and has error at most `2^{-(n+1)}` on every `f ∈ J` (in the proof: bisection of the interval of
Gaussian levels `[lo, hi] ∋ 1 - ∫ f`, querying the quantile of the midpoint level). -/
theorem bisection_upper_bound_step (n : ℕ) :
    ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / (2 : ℝ) ^ (n + 1) := by
  sorry

/-- **R1 + R2 (exact minimal error).** `2^{-(n+1)}` is the least worst-case error on `J` that a deterministic
adaptive algorithm with `n` function values can guarantee (the infimum is attained). -/
theorem minimal_adaptive_error_step (n : ℕ) :
    IsLeast {e : ℝ | ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ e} (1 / (2 : ℝ) ^ (n + 1)) := by
  sorry

/-- **R3 (nonadaptive lower bound, one crossing).** For every deterministic nonadaptive algorithm with `n` nodes
(and an arbitrary output map) there is `f ∈ J` on which its error is at least `1 / (2(n+1))`. -/
theorem nonadaptive_lower_bound_step (n : ℕ) (A : NonadaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  sorry

/-- **R3, matching upper bound.** There is a deterministic nonadaptive algorithm with `n` nodes whose error is at
most `1 / (2(n+1))` on every `f ∈ J` (in the proof: nodes at the Gaussian quantiles of levels `1/(n+1), …, n/(n+1)`,
estimate `1 - (k + 1/2)/(n+1)` where `k` is the number of zero values). -/
theorem nonadaptive_upper_bound_step (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / (2 * ((n : ℝ) + 1)) := by
  sorry

/-- **R3 (exact minimal nonadaptive error).** `1 / (2(n+1))` is the least worst-case error on `J` that a
deterministic nonadaptive algorithm with `n` function values can guarantee (compare `2^{-(n+1)}` for adaptive
algorithms, `minimal_adaptive_error_step`). -/
theorem minimal_nonadaptive_error_step (n : ℕ) :
    IsLeast {e : ℝ | ∃ A : NonadaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ e} (1 / (2 * ((n : ℝ) + 1))) := by
  sorry

/-- **R4 (adaptive lower bound, two crossings).** For every deterministic adaptive algorithm that uses `n` function
values there is an interval indicator `f ∈ P` on which its error is at least `1 / (2(n+1))`. -/
theorem adaptive_lower_bound_interval (n : ℕ) (A : AdaptiveAlg n) :
    ∃ f ∈ IntervalClass,
      1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  sorry

/-- **R4, upper bound (adaption helps at most by a factor 2 on `P`).** There is a deterministic nonadaptive
algorithm with `n` nodes whose error is at most `1/(n+1)` on every `f ∈ P` (in the proof: nodes at the Gaussian
quantiles of levels `1/(n+1), …, n/(n+1)`, estimate `k/(n+1)` where `k` is the number of values equal to `1`). -/
theorem nonadaptive_upper_bound_interval (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ IntervalClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / ((n : ℝ) + 1) := by
  sorry

/-- **R5 (randomized lower bound, one crossing).** Let `(Ω, P)` be a probability space and `A ω` (`ω ∈ Ω`) a
family of deterministic adaptive algorithms that use `n` function values, such that for every `f ∈ J` the estimate
`ω ↦ (A ω).output ((A ω).info f)` is measurable (a random variable). Then for some `f ∈ J` the expected error
`∫⁻ ω, |(A ω).output ((A ω).info f) - ∫ f dγ| ∂P` (a Lebesgue integral with values in `[0, ∞]`) is at least
`1 / (2 (2^{n+1} - 1))`, which exceeds `2^{-(n+2)}`: compared with the deterministic optimum `2^{-(n+1)}`
(`minimal_adaptive_error_step`), randomization gains at most a factor of about `2`. -/
theorem randomized_lower_bound_step (n : ℕ) {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → AdaptiveAlg n)
    (hA : ∀ f ∈ StepClass, Measurable fun ω => (A ω).output ((A ω).info f)) :
    ∃ f ∈ StepClass, ENNReal.ofReal (1 / (2 * ((2 : ℝ) ^ (n + 1) - 1))) ≤
      ∫⁻ ω, ENNReal.ofReal |(A ω).output ((A ω).info f) - ∫ t, f t ∂gaussianReal 0 1| ∂P := by
  sorry

end Cubature
