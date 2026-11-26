import Theory.Defs
import Theory.Lemmas

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof (no `sorry`, no new
axioms). The arguments are carried out in `Theory/Lemmas.lean` for an arbitrary probability measure on `ℝ` without
atoms; here they are applied to the standard Gaussian measure `gaussianReal 0 1`.
-/

open MeasureTheory ProbabilityTheory

namespace Cubature

/-- The standard Gaussian measure has no atoms. -/
lemma gaussian_nullSingleton : NullSingletonClass (gaussianReal 0 1) :=
  nullSingletonClass_gaussianReal one_ne_zero

theorem adaptive_lower_bound_step (n : ℕ) (A : AdaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 : ℝ) ^ (n + 1) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  have := gaussian_nullSingleton
  exact adaptive_lower_bound_general A

theorem bisection_upper_bound_step (n : ℕ) :
    ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / (2 : ℝ) ^ (n + 1) := by
  have := gaussian_nullSingleton
  exact bisection_upper_bound_general n

theorem minimal_adaptive_error_step (n : ℕ) :
    IsLeast {e : ℝ | ∃ A : AdaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ e} (1 / (2 : ℝ) ^ (n + 1)) := by
  refine ⟨bisection_upper_bound_step n, ?_⟩
  rintro e ⟨A, hA⟩
  obtain ⟨f, hf, hlow⟩ := adaptive_lower_bound_step n A
  exact hlow.trans (hA f hf)

theorem randomized_lower_bound_step (n : ℕ) {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → AdaptiveAlg n)
    (hA : ∀ f ∈ StepClass, Measurable fun ω => (A ω).output ((A ω).info f)) :
    ∃ f ∈ StepClass, ENNReal.ofReal (1 / (2 * ((2 : ℝ) ^ (n + 1) - 1))) ≤
      ∫⁻ ω, ENNReal.ofReal |(A ω).output ((A ω).info f) - ∫ t, f t ∂gaussianReal 0 1| ∂P := by
  have := gaussian_nullSingleton
  exact randomized_lower_bound_general P A hA

theorem nonadaptive_lower_bound_step (n : ℕ) (A : NonadaptiveAlg n) :
    ∃ f ∈ StepClass, 1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  have := gaussian_nullSingleton
  exact nonadaptive_lower_bound_general A

theorem nonadaptive_upper_bound_step (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / (2 * ((n : ℝ) + 1)) := by
  have := gaussian_nullSingleton
  exact nonadaptive_upper_bound_general n

theorem minimal_nonadaptive_error_step (n : ℕ) :
    IsLeast {e : ℝ | ∃ A : NonadaptiveAlg n, ∀ f ∈ StepClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ e} (1 / (2 * ((n : ℝ) + 1))) := by
  refine ⟨nonadaptive_upper_bound_step n, ?_⟩
  rintro e ⟨A, hA⟩
  obtain ⟨f, hf, hlow⟩ := nonadaptive_lower_bound_step n A
  exact hlow.trans (hA f hf)

theorem adaptive_lower_bound_interval (n : ℕ) (A : AdaptiveAlg n) :
    ∃ f ∈ IntervalClass,
      1 / (2 * ((n : ℝ) + 1)) ≤ |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| := by
  have := gaussian_nullSingleton
  exact adaptive_lower_bound_interval_general A

theorem nonadaptive_upper_bound_interval (n : ℕ) :
    ∃ A : NonadaptiveAlg n, ∀ f ∈ IntervalClass,
      |A.output (A.info f) - ∫ t, f t ∂gaussianReal 0 1| ≤ 1 / ((n : ℝ) + 1) := by
  have := gaussian_nullSingleton
  exact nonadaptive_upper_bound_interval_general n

end Cubature
