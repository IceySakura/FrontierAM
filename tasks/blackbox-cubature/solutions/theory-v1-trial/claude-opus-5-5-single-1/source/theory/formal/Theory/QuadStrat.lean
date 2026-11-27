import Theory.Defs
import Theory.Quadric
import Theory.Tube
import Theory.Refined

/-!
# Quadric jump surfaces in an arbitrary direction: stratified rate up to a logarithm
(auxiliary proofs for R10)
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology
open scoped ENNReal InnerProductSpace

namespace Cubature.Aux

/-- Refined 1-D estimate keeping the square roots of the discriminants (concave case). -/
lemma quad_mass_diff_sqrt_of_neg {a β γ β' γ' : ℝ} (ha : a < 0) :
    (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β * t + γ} -
        (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'} ≤
      (|β - β'| + |√(max (β ^ 2 - 4 * a * γ) 0) - √(max (β' ^ 2 - 4 * a * γ') 0)|) /
        (√(2 * π) * |a|) := by
  rw [quad_pos_set_of_neg ha, quad_pos_set_of_neg ha]
  refine (gauss_Ioo_sub_le _ _ _ _).trans ?_
  set D := β ^ 2 - 4 * a * γ
  set D' := β' ^ 2 - 4 * a * γ'
  have habs : 0 < |a| := abs_pos.mpr ha.ne
  have hm : |-β / (2 * a) - -β' / (2 * a)| = |β - β'| / (2 * |a|) := by
    rw [show -β / (2 * a) - -β' / (2 * a) = -(β - β') / (2 * a) by ring, abs_div, abs_neg,
      abs_mul, abs_two]
  have hw : |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| =
      |√(max D 0) - √(max D' 0)| / (2 * |a|) := by
    rw [← sub_div, abs_div, abs_of_pos (by positivity : (0:ℝ) < 2 * |a|)]
  have e1 : |(-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))|
      ≤ |-β / (2 * a) - -β' / (2 * a)| +
        |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| := by
    rw [show (-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))
      = (-β / (2 * a) - -β' / (2 * a)) - (√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|))
      by ring]
    exact abs_sub _ _
  have e2 : |(-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))|
      ≤ |-β / (2 * a) - -β' / (2 * a)| +
        |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| := by
    rw [show (-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))
      = (-β / (2 * a) - -β' / (2 * a)) + (√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|))
      by ring]
    exact abs_add_le _ _
  rw [hm, hw] at e1 e2
  have hsum : |(-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))|
      + |(-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))|
      ≤ (|β - β'| + |√(max D 0) - √(max D' 0)|) / |a| := by
    have : (|β - β'| + |√(max D 0) - √(max D' 0)|) / |a| =
        2 * (|β - β'| / (2 * |a|)) + 2 * (|√(max D 0) - √(max D' 0)| / (2 * |a|)) := by
      field_simp
    rw [this]; linarith
  calc (√(2 * π))⁻¹ * (_ + _) ≤ (√(2 * π))⁻¹ * ((|β - β'| + |√(max D 0) - √(max D' 0)|) / |a|) := by
        gcongr
    _ = (|β - β'| + |√(max D 0) - √(max D' 0)|) / (√(2 * π) * |a|) := by field_simp

theorem quad_mass_diff_sqrt {a β γ β' γ' : ℝ} (ha : a ≠ 0) :
    |(gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β * t + γ} -
        (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'}| ≤
      (|β - β'| + |√(max (β ^ 2 - 4 * a * γ) 0) - √(max (β' ^ 2 - 4 * a * γ') 0)|) /
        (√(2 * π) * |a|) := by
  rcases lt_or_gt_of_ne ha with hneg | hpos
  · rw [abs_le]
    constructor
    · have := quad_mass_diff_sqrt_of_neg (β := β') (γ := γ') (β' := β) (γ' := γ) hneg
      rw [abs_sub_comm β' β, abs_sub_comm (√(max (β' ^ 2 - 4 * a * γ') 0))] at this
      linarith
    · exact quad_mass_diff_sqrt_of_neg hneg
  · rw [quad_mass_neg ha, quad_mass_neg ha]
    have hn : -a < 0 := by linarith
    have key1 := quad_mass_diff_sqrt_of_neg (β := -β') (γ := -γ') (β' := -β) (γ' := -γ) hn
    have key2 := quad_mass_diff_sqrt_of_neg (β := -β) (γ := -γ) (β' := -β') (γ' := -γ') hn
    have e1 : |-β' - -β| = |β - β'| := by rw [show -β' - -β = β - β' by ring]
    have e2 : |-β - -β'| = |β - β'| := by rw [show -β - -β' = -(β - β') by ring, abs_neg]
    have d1 : (-β') ^ 2 - 4 * -a * -γ' = β' ^ 2 - 4 * a * γ' := by ring
    have d2 : (-β) ^ 2 - 4 * -a * -γ = β ^ 2 - 4 * a * γ := by ring
    rw [e1, d1, d2, abs_neg] at key1
    rw [e2, d1, d2, abs_neg] at key2
    rw [abs_sub_comm (√(max (β' ^ 2 - 4 * a * γ') 0))] at key1
    rw [abs_le]
    constructor <;> linarith

/-- Square-root increments of discriminants: profile bound. -/
noncomputable def sqrtProfile (η Dmax : ℝ) (r : ℝ) : ℝ :=
  (Icc (2 * η) Dmax).indicator (fun r => η ^ 2 / r) r + (Ico (-η) (2 * η)).indicator (fun _ => η) r

lemma sqrtProfile_nonneg {η Dmax : ℝ} (hη : 0 < η) (r : ℝ) : 0 ≤ sqrtProfile η Dmax r := by
  unfold sqrtProfile
  refine add_nonneg ?_ ?_
  · by_cases h : r ∈ Icc (2 * η) Dmax
    · rw [indicator_of_mem h]
      have : 0 < r := by linarith [h.1]
      positivity
    · rw [indicator_of_notMem h]
  · by_cases h : r ∈ Ico (-η) (2 * η)
    · rw [indicator_of_mem h]; positivity
    · rw [indicator_of_notMem h]

lemma sqrt_incr_sq_le {D D' η Dmax : ℝ} (hη : 0 < η) (hD : |D - D'| ≤ η) (hDm : D ≤ Dmax) :
    |√(max D 0) - √(max D' 0)| ^ 2 ≤ sqrtProfile η Dmax D := by
  unfold sqrtProfile
  rcases le_or_gt (2 * η) D with h1 | h1
  · rw [indicator_of_mem (show D ∈ Icc (2 * η) Dmax from ⟨h1, hDm⟩),
      indicator_of_notMem (show D ∉ Ico (-η) (2 * η) from fun h => by linarith [h.2]), add_zero]
    have hD0 : 0 < D := by linarith
    have hD'0 : 0 < D' := by linarith [(abs_le.mp hD).2]
    rw [max_eq_left hD0.le, max_eq_left hD'0.le]
    have hs : √D - √D' = (D - D') / (√D + √D') := by
      have hpos : 0 < √D + √D' := by positivity
      rw [eq_div_iff hpos.ne']
      have e : (√D - √D') * (√D + √D') = √D ^ 2 - √D' ^ 2 := by ring
      rw [e, Real.sq_sqrt hD0.le, Real.sq_sqrt hD'0.le]
    have hb : |√D - √D'| ≤ η / √D := by
      rw [hs, abs_div, abs_of_pos (by positivity : 0 < √D + √D')]
      calc |D - D'| / (√D + √D') ≤ η / (√D + √D') := by gcongr
        _ ≤ η / √D := by
          apply div_le_div_of_nonneg_left hη.le (Real.sqrt_pos.mpr hD0)
          linarith [Real.sqrt_nonneg D']
    calc |√D - √D'| ^ 2 ≤ (η / √D) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) hb 2
      _ = η ^ 2 / D := by rw [div_pow, Real.sq_sqrt hD0.le]
  · rcases le_or_gt (-η) D with h2 | h2
    · rw [indicator_of_notMem (show D ∉ Icc (2 * η) Dmax from fun h => by linarith [h.1]),
        indicator_of_mem (show D ∈ Ico (-η) (2 * η) from ⟨h2, h1⟩), zero_add]
      have := abs_sqrt_sub_sqrt_le (le_max_right D 0) (le_max_right D' 0)
      have hmm : |max D 0 - max D' 0| ≤ η := (abs_max_sub_max_le_abs D D' 0).trans hD
      calc |√(max D 0) - √(max D' 0)| ^ 2 ≤ (√|max D 0 - max D' 0|) ^ 2 :=
            pow_le_pow_left₀ (abs_nonneg _) this 2
        _ = |max D 0 - max D' 0| := Real.sq_sqrt (abs_nonneg _)
        _ ≤ η := hmm
    · have hD' : D' < 0 := by linarith [(abs_le.mp hD).1]
      rw [max_eq_right (by linarith), max_eq_right hD'.le, sub_self, abs_zero]
      simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow]
      exact sqrtProfile_nonneg hη D

lemma sqrtProfile_measurable {η Dmax : ℝ} : Measurable (sqrtProfile η Dmax) := by
  unfold sqrtProfile
  refine Measurable.add ?_ ?_
  · exact (measurable_const.div measurable_id).indicator measurableSet_Icc
  · exact measurable_const.indicator measurableSet_Ico

lemma sqrtProfile_integrable {η Dmax : ℝ} (hη : 0 < η) : Integrable (sqrtProfile η Dmax) := by
  unfold sqrtProfile
  refine Integrable.add ?_ ?_
  · refine (integrable_indicator_iff measurableSet_Icc).mpr ?_
    refine ContinuousOn.integrableOn_Icc ?_
    refine ContinuousOn.div continuousOn_const continuousOn_id ?_
    intro r hr
    have : 0 < r := by linarith [hr.1]
    exact this.ne'
  · exact (integrable_indicator_iff measurableSet_Ico).mpr (integrableOn_const (by simp))

lemma sqrtProfile_integral {η Dmax : ℝ} (hη : 0 < η) (hD : 2 * η ≤ Dmax) :
    ∫ r, sqrtProfile η Dmax r = η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2 := by
  unfold sqrtProfile
  have h1 : Integrable ((Icc (2 * η) Dmax).indicator (fun r : ℝ => η ^ 2 / r)) := by
    refine (integrable_indicator_iff measurableSet_Icc).mpr ?_
    refine ContinuousOn.integrableOn_Icc ?_
    refine ContinuousOn.div continuousOn_const continuousOn_id ?_
    intro r hr
    have : 0 < r := by linarith [hr.1]
    exact this.ne'
  have h2 : Integrable ((Ico (-η) (2 * η)).indicator (fun _ : ℝ => η)) :=
    (integrable_indicator_iff measurableSet_Ico).mpr (integrableOn_const (by simp))
  rw [integral_add h1 h2, integral_indicator measurableSet_Icc, integral_indicator measurableSet_Ico,
    setIntegral_const, integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le hD]
  have e : ∀ r : ℝ, η ^ 2 / r = η ^ 2 * r⁻¹ := fun r => div_eq_mul_inv _ _
  simp_rw [e]
  rw [intervalIntegral.integral_const_mul, integral_inv_of_pos (by positivity) (by linarith),
    Measure.real, Real.volume_Ico, ENNReal.toReal_ofReal (by linarith), smul_eq_mul]
  ring

end Cubature.Aux

namespace Cubature.Aux

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma quadric_incr (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z) {R : ℝ} (x x' : E) (hx : ‖x‖ ≤ R)
    (hx' : ‖x'‖ ≤ R) :
    |β x - β x'| ≤ 2 * ‖A‖ * ‖x - x'‖ ∧
      |D x - D x'| ≤ 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * ‖x - x'‖ := by
  set a := ⟪u, A u⟫_ℝ with ha
  set δ := ‖x - x'‖
  have hA : 0 ≤ ‖A‖ := norm_nonneg _
  have hδ : 0 ≤ δ := norm_nonneg _
  have hR : 0 ≤ R := (norm_nonneg _).trans hx
  have hAu : ‖A u‖ ≤ ‖A‖ := by
    have := A.le_opNorm u; rwa [hu1, mul_one] at this
  have hβd : |β x - β x'| ≤ 2 * ‖A‖ * δ := by
    have e : β x - β x' = ⟪x - x', A u⟫_ℝ + ⟪u, A (x - x')⟫_ℝ := by
      rw [hβ x, hβ x']
      simp only [map_sub, inner_sub_left, inner_sub_right]; ring
    rw [e]
    have h1 : |⟪x - x', A u⟫_ℝ| ≤ δ * ‖A‖ :=
      (abs_real_inner_le_norm _ _).trans (by gcongr)
    have h2 : |⟪u, A (x - x')⟫_ℝ| ≤ ‖A‖ * δ := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hu1, one_mul]; exact A.le_opNorm _
    calc |⟪x - x', A u⟫_ℝ + ⟪u, A (x - x')⟫_ℝ| ≤ |⟪x - x', A u⟫_ℝ| + |⟪u, A (x - x')⟫_ℝ| :=
          abs_add_le _ _
      _ ≤ δ * ‖A‖ + ‖A‖ * δ := add_le_add h1 h2
      _ = 2 * ‖A‖ * δ := by ring
  have hβb : ∀ z : E, ‖z‖ ≤ R → |β z| ≤ 2 * ‖A‖ * R + ‖b‖ := by
    intro z hz
    have h1 : |⟪z, A u⟫_ℝ| ≤ R * ‖A‖ := (abs_real_inner_le_norm _ _).trans (by gcongr)
    have h2 : |⟪u, A z⟫_ℝ| ≤ ‖A‖ * R := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hu1, one_mul]; exact (A.le_opNorm _).trans (by gcongr)
    have h3 : |⟪b, u⟫_ℝ| ≤ ‖b‖ := by
      refine (abs_real_inner_le_norm _ _).trans ?_; rw [hu1, mul_one]
    calc |β z| ≤ |⟪z, A u⟫_ℝ| + |⟪u, A z⟫_ℝ| + |⟪b, u⟫_ℝ| := by
          rw [hβ z]; exact (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ ≤ R * ‖A‖ + ‖A‖ * R + ‖b‖ := by gcongr
      _ = 2 * ‖A‖ * R + ‖b‖ := by ring
  have hφd : |φ x - φ x'| ≤ (2 * ‖A‖ * R + ‖b‖) * δ := by
    have e : φ x - φ x' = ⟪x - x', A x⟫_ℝ + ⟪x', A (x - x')⟫_ℝ + ⟪b, x - x'⟫_ℝ := by
      rw [hφ x, hφ x']
      simp only [map_sub, inner_sub_left, inner_sub_right]; ring
    rw [e]
    have h1 : |⟪x - x', A x⟫_ℝ| ≤ δ * (‖A‖ * R) :=
      (abs_real_inner_le_norm _ _).trans (by gcongr; exact (A.le_opNorm _).trans (by gcongr))
    have h2 : |⟪x', A (x - x')⟫_ℝ| ≤ R * (‖A‖ * δ) :=
      (abs_real_inner_le_norm _ _).trans (by gcongr; exact A.le_opNorm _)
    have h3 : |⟪b, x - x'⟫_ℝ| ≤ ‖b‖ * δ := abs_real_inner_le_norm _ _
    calc |⟪x - x', A x⟫_ℝ + ⟪x', A (x - x')⟫_ℝ + ⟪b, x - x'⟫_ℝ|
        ≤ |⟪x - x', A x⟫_ℝ| + |⟪x', A (x - x')⟫_ℝ| + |⟪b, x - x'⟫_ℝ| :=
          (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ ≤ δ * (‖A‖ * R) + R * (‖A‖ * δ) + ‖b‖ * δ := by gcongr
      _ = (2 * ‖A‖ * R + ‖b‖) * δ := by ring
  have hab : |a| ≤ ‖A‖ := by
    refine (abs_real_inner_le_norm _ _).trans ?_; rw [hu1, one_mul]; exact hAu
  refine ⟨hβd, ?_⟩
  have e : D x - D x' = (β x - β x') * (β x + β x') - 4 * a * (φ x - φ x') := by
    rw [hD x, hD x']; ring
  rw [e]
  have hs : |β x + β x'| ≤ 2 * (2 * ‖A‖ * R + ‖b‖) := by
    have := abs_add_le (β x) (β x'); linarith [hβb x hx, hβb x' hx']
  calc |(β x - β x') * (β x + β x') - 4 * a * (φ x - φ x')|
      ≤ |β x - β x'| * |β x + β x'| + 4 * |a| * |φ x - φ x'| := by
        refine (abs_sub _ _).trans ?_
        rw [abs_mul, abs_mul, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 4)]
    _ ≤ (2 * ‖A‖ * δ) * (2 * (2 * ‖A‖ * R + ‖b‖)) +
          4 * ‖A‖ * ((2 * ‖A‖ * R + ‖b‖) * δ) := by gcongr
    _ = 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * δ := by ring

end Cubature.Aux

namespace Cubature.Aux

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma sqrtProfile_le {η Dmax : ℝ} (hη : 0 < η) (r : ℝ) : sqrtProfile η Dmax r ≤ η := by
  unfold sqrtProfile
  by_cases h1 : r ∈ Icc (2 * η) Dmax
  · rw [indicator_of_mem h1, indicator_of_notMem (show r ∉ Ico (-η) (2 * η) from
      fun h => by linarith [h.2, h1.1]), add_zero]
    have hr : 0 < r := by linarith [h1.1]
    rw [div_le_iff₀ hr]; nlinarith [h1.1]
  · rw [indicator_of_notMem h1, zero_add]
    by_cases h2 : r ∈ Ico (-η) (2 * η)
    · rw [indicator_of_mem h2]
    · rw [indicator_of_notMem h2]; exact hη.le

theorem quadric_stratified {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace E]
    [OpensMeasurableSpace E] (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z)
    {ι : Type*} [Fintype ι] {μ : Measure E} [IsProbabilityMeasure μ]
    {C : ι → Set E} (hCm : ∀ i, MeasurableSet (C i))
    (hCd : Pairwise (Disjoint on C)) (hCcov : μ (⋃ i, C i)ᶜ = 0)
    {R h Dmax K : ℝ} (hR : 0 < R) (hh : 0 < h)
    (hCR : ∀ i, ∀ x ∈ C i, ‖x‖ ≤ R) (hCD : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, dist x x' ≤ h)
    (hDmax : ∀ i, ∀ x ∈ C i, D x ≤ Dmax) (hK0 : 0 ≤ K)
    (hK : μ.map D ≤ ENNReal.ofReal K • volume)
    (h2 : 2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ≤ Dmax)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → E} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (C i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump (gaussianReal 0 1)
          (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) (X i ω) -
        ∫ x, preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x ∂μ) ^ 2
        ∂P ≤
      1 / Fintype.card ι * ((2 * ‖A‖ * h) ^ 2 +
        K * ((8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2 *
          Real.log (Dmax / (2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h))) +
          3 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2)) / (π * ⟪u, A u⟫_ℝ ^ 2) := by
  set a := ⟪u, A u⟫_ℝ with ha
  set F := preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) with hF
  have hapos : 0 < |a| := abs_pos.mpr hu
  have hA : 0 < ‖A‖ := by
    rcases eq_or_lt_of_le (norm_nonneg A) with h0 | h0
    · exfalso; apply hu
      have : A = 0 := norm_eq_zero.mp h0.symm
      simp [ha, this]
    · exact h0
  set η := 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h with hη_def
  have hη : 0 < η := by positivity
  -- explicit form of the preintegrated function
  have hP : ∀ z, F z = (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β z * t + φ z} := by
    intro z
    have hset : {t : ℝ | 0 < φ (z + t • u)} = {t : ℝ | 0 < a * t ^ 2 + β z * t + φ z} := by
      ext t
      simp only [mem_ofPred_eq]
      rw [hφ (z + t • u), quadric_line, hβ z, hφ z]
    have hm : MeasurableSet {t : ℝ | 0 < a * t ^ 2 + β z * t + φ z} :=
      measurableSet_lt measurable_const (by fun_prop)
    rw [hF, preintJump, hset, integral_indicator hm, setIntegral_const, smul_eq_mul, mul_one]
  -- continuity of β, D and F
  have hβc : Continuous β := by
    have : β = fun z => ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ := funext hβ
    rw [this]; fun_prop
  have hφc : Continuous φ := by
    have : φ = fun z => ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c := funext hφ
    rw [this]; fun_prop
  have hDc : Continuous D := by
    have : D = fun z => β z ^ 2 - 4 * a * φ z := funext hD
    rw [this]; fun_prop
  have hFc : Continuous F := by
    rw [continuous_iff_continuousAt]
    intro x0
    rw [ContinuousAt, tendsto_iff_dist_tendsto_zero]
    have hbound : ∀ x, dist (F x) (F x0) ≤
        (|β x - β x0| + √|D x - D x0|) / (√(2 * π) * |a|) := by
      intro x
      rw [Real.dist_eq]
      exact quadric_preint A b c φ hφ u hu β D hβ hD x x0
    refine squeeze_zero (fun x => dist_nonneg) hbound ?_
    have h1 : Tendsto (fun x => |β x - β x0|) (𝓝 x0) (𝓝 0) := by
      have := ((hβc.tendsto x0).sub_const (β x0)).abs
      simpa using this
    have h2 : Tendsto (fun x => √|D x - D x0|) (𝓝 x0) (𝓝 0) := by
      have := (((hDc.tendsto x0).sub_const (D x0)).abs).sqrt
      simpa using this
    have := (h1.add h2).div_const (√(2 * π) * |a|)
    simpa using this
  have hFm : Measurable F := hFc.measurable
  -- the oscillation weight
  set g : E → ℝ := fun x => (2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) / (√(2 * π) * |a|)
    with hg
  have hgm : Measurable g := by
    refine Measurable.div_const (measurable_const.add ?_) _
    exact (sqrtProfile_measurable.comp hDc.measurable).sqrt
  have hgC : ∀ x, |g x| ≤ (2 * ‖A‖ * h + √η) / (√(2 * π) * |a|) := by
    intro x
    have h0 : 0 ≤ √(sqrtProfile η Dmax (D x)) := Real.sqrt_nonneg _
    have hs : √(sqrtProfile η Dmax (D x)) ≤ √η := Real.sqrt_le_sqrt (sqrtProfile_le hη _)
    have hden : 0 < √(2 * π) * |a| := by positivity
    rw [abs_of_nonneg (div_nonneg (by positivity) hden.le)]
    exact div_le_div_of_nonneg_right (by linarith) hden.le
  have hFg : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, |F x - F x'| ≤ g x := by
    intro i x hx x' hx'
    obtain ⟨hb1, hb2⟩ := quadric_incr A b c φ hφ u hu1 β D hβ hD x x' (hCR i x hx) (hCR i x' hx')
    have hd : ‖x - x'‖ ≤ h := by rw [← dist_eq_norm]; exact hCD i x hx x' hx'
    have hbe : |β x - β x'| ≤ 2 * ‖A‖ * h := hb1.trans (by gcongr)
    have hDe : |D x - D x'| ≤ η := hb2.trans (by rw [hη_def]; gcongr)
    have hsq := sqrt_incr_sq_le hη hDe (hDmax i x hx)
    have hsq' : |√(max (D x) 0) - √(max (D x') 0)| ≤ √(sqrtProfile η Dmax (D x)) := by
      rw [sq_abs] at hsq; exact Real.abs_le_sqrt hsq
    rw [hP x, hP x']
    have key := quad_mass_diff_sqrt (a := a) (β := β x) (γ := φ x) (β' := β x') (γ' := φ x') hu
    rw [← hD x, ← hD x'] at key
    refine key.trans ?_
    simp only [hg]
    gcongr
  have main := stratified_mse_weighted hCm hCd hCcov hXm hXind hXlaw hFm (B := 1)
    (fun x => by
      rw [hP x]
      have h0 := measureReal_nonneg (μ := gaussianReal 0 1)
        (s := {t : ℝ | 0 < a * t ^ 2 + β x * t + φ x})
      have h1 := measureReal_le_one (μ := gaussianReal 0 1)
        (s := {t : ℝ | 0 < a * t ^ 2 + β x * t + φ x})
      rw [abs_le]; constructor <;> linarith) hgm hgC hFg
  refine main.trans ?_
  -- bound `∫ g²`
  set Ψ := sqrtProfile η Dmax with hΨ
  have hΨ0 : ∀ r, 0 ≤ Ψ r := sqrtProfile_nonneg hη
  have hΨi : Integrable Ψ := sqrtProfile_integrable hη
  have hc2 : (√(2 * π) * |a|) ^ 2 = 2 * π * a ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by positivity), sq_abs]
  have hpt : ∀ x, g x ^ 2 ≤ (2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) / (2 * π * a ^ 2) := by
    intro x
    simp only [hg]
    rw [div_pow, hc2]
    gcongr
    have h1 := Real.sq_sqrt (hΨ0 (D x))
    nlinarith [sq_nonneg (2 * ‖A‖ * h - √(Ψ (D x)))]
  have hΨD_int : Integrable (fun x => Ψ (D x)) μ := by
    have : Integrable Ψ (μ.map D) := by
      refine Integrable.mono_measure ?_ hK
      exact hΨi.smul_measure ENNReal.ofReal_ne_top
    exact (integrable_map_measure sqrtProfile_measurable.aestronglyMeasurable
      hDc.measurable.aemeasurable).mp this
  have hint_rhs : Integrable (fun x => (2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) / (2 * π * a ^ 2)) μ :=
    ((integrable_const _).add (hΨD_int.const_mul _)).div_const _
  have hint_g2 : Integrable (fun x => g x ^ 2) μ :=
    Integrable.of_bound (hgm.pow_const 2).aestronglyMeasurable
      (((2 * ‖A‖ * h + √η) / (√(2 * π) * |a|)) ^ 2)
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC x) 2)
  have hΨint : ∫ x, Ψ (D x) ∂μ ≤ K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2) := by
    calc ∫ x, Ψ (D x) ∂μ = ∫ r, Ψ r ∂(μ.map D) :=
          (integral_map hDc.measurable.aemeasurable sqrtProfile_measurable.aestronglyMeasurable).symm
      _ ≤ ∫ r, Ψ r ∂(ENNReal.ofReal K • volume) :=
          integral_mono_measure hK (ae_of_all _ hΨ0) (hΨi.smul_measure ENNReal.ofReal_ne_top)
      _ = K * ∫ r, Ψ r := by
          rw [integral_smul_measure, ENNReal.toReal_ofReal hK0, smul_eq_mul]
      _ = K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2) := by
          rw [hΨ, sqrtProfile_integral hη h2]
  have hg2 : ∫ x, g x ^ 2 ∂μ ≤ ((2 * ‖A‖ * h) ^ 2 +
      K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) / (π * a ^ 2) := by
    calc ∫ x, g x ^ 2 ∂μ ≤ ∫ x, (2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) / (2 * π * a ^ 2) ∂μ :=
          integral_mono hint_g2 hint_rhs hpt
      _ = (2 * (2 * ‖A‖ * h) ^ 2 + 2 * ∫ x, Ψ (D x) ∂μ) / (2 * π * a ^ 2) := by
          rw [integral_div, integral_add (integrable_const _) (hΨD_int.const_mul _),
            integral_const, integral_const_mul]
          simp
      _ ≤ (2 * (2 * ‖A‖ * h) ^ 2 + 2 * (K * (η ^ 2 * Real.log (Dmax / (2 * η)) +
            3 * η ^ 2))) / (2 * π * a ^ 2) := by gcongr
      _ = ((2 * ‖A‖ * h) ^ 2 + K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) /
            (π * a ^ 2) := by
          have : (0 : ℝ) < π * a ^ 2 := by positivity
          field_simp
  have hn0 : 0 ≤ (1 / Fintype.card ι : ℝ) := by positivity
  calc (1 / Fintype.card ι : ℝ) * ∫ x, g x ^ 2 ∂μ
      ≤ (1 / Fintype.card ι : ℝ) * (((2 * ‖A‖ * h) ^ 2 +
          K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) / (π * a ^ 2)) :=
        mul_le_mul_of_nonneg_left hg2 hn0
    _ = _ := by rw [hη_def]; ring

end Cubature.Aux
