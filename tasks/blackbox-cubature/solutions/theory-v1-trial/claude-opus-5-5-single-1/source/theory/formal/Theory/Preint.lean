import Theory.Defs
import Theory.Sublevel

/-!
# Regularity of preintegrated jump and kink integrands (auxiliary proofs for R2, R3)
-/

open MeasureTheory ProbabilityTheory Real Set

namespace Cubature.Aux

lemma continuous_of_strong {φ : ℝ → ℝ} {α : ℝ} (hα : 0 < α)
    (hφ : StrongConvexOn univ α φ ∨ StrongConcaveOn univ α φ) : Continuous φ := by
  have hmod : (0 : ℝ → ℝ) ≤ fun r => α / 2 * r ^ 2 := fun r => by
    simp only [Pi.zero_apply]; positivity
  rcases hφ with h | h
  · exact continuousOn_univ.mp ((h.convexOn hmod).continuousOn isOpen_univ)
  · exact continuousOn_univ.mp ((h.concaveOn hmod).continuousOn isOpen_univ)

lemma volume_sublevel_le_of_strong {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η)
    (hφ : StrongConvexOn univ α φ ∨ StrongConcaveOn univ α φ) :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(η / α)) := by
  rcases hφ with h | h
  · exact volume_sublevel_le_of_strongConvexOn hα hη h
  · have h' : StrongConvexOn univ α (-φ) := UniformConcaveOn.neg h
    have := volume_sublevel_le_of_strongConvexOn hα hη h'
    simpa only [Pi.neg_apply, abs_neg] using this

lemma measureReal_sublevel_le {ν : Measure ℝ} [IsFiniteMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η)
    (hφ : StrongConvexOn univ α φ ∨ StrongConcaveOn univ α φ) :
    ν.real {t : ℝ | |φ t| ≤ η} ≤ ρ * (4 * √(η / α)) := by
  have h2 := volume_sublevel_le_of_strong hα hη hφ
  have h3 : ν {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (ρ * (4 * √(η / α))) := by
    rw [ENNReal.ofReal_mul hρ]
    exact (hν _).trans (by gcongr)
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) h3

lemma gaussianPDFReal_le (x : ℝ) : gaussianPDFReal 0 1 x ≤ (√(2 * π))⁻¹ := by
  rw [gaussianPDFReal]
  simp only [NNReal.coe_one, mul_one, sub_zero]
  have : rexp (-x ^ 2 / 2) ≤ 1 := by
    rw [Real.exp_le_one_iff]; have := sq_nonneg x; linarith
  have h0 : 0 ≤ (√(2 * π))⁻¹ := by positivity
  calc (√(2 * π))⁻¹ * rexp (-x ^ 2 / 2) ≤ (√(2 * π))⁻¹ * 1 :=
        mul_le_mul_of_nonneg_left this h0
    _ = (√(2 * π))⁻¹ := mul_one _

lemma gaussianReal_le_volume (S : Set ℝ) :
    gaussianReal 0 1 S ≤ ENNReal.ofReal (√(2 * π))⁻¹ * volume S := by
  rw [gaussianReal_apply 0 one_ne_zero S]
  calc ∫⁻ x in S, gaussianPDF 0 1 x ≤ ∫⁻ _ in S, ENNReal.ofReal (√(2 * π))⁻¹ := by
        refine lintegral_mono fun x => ?_
        rw [gaussianPDF]
        exact ENNReal.ofReal_le_ofReal (gaussianPDFReal_le x)
    _ = ENNReal.ofReal (√(2 * π))⁻¹ * volume S := setLIntegral_const _ _

theorem jump_holder {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ} [IsProbabilityMeasure ν]
    {ρ : ℝ} (hρ : 0 ≤ ρ) (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y))
    (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump ν φ θ y - preintJump ν φ θ y'| ≤
      Lθ * dist y y' + 4 * ρ * B * √(L * dist y y' / α) := by
  set δ := dist y y'
  have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0 y)
  have hLδ : 0 ≤ L * δ := (abs_nonneg _).trans (hφL 0 y y')
  have hcont : ∀ z, Continuous (φ · z) := fun z => continuous_of_strong hα (hφ z)
  have hmeasA : ∀ z, MeasurableSet {s : ℝ | 0 < φ s z} := fun z =>
    measurableSet_lt measurable_const (hcont z).measurable
  set S := {t : ℝ | |φ t y| ≤ L * δ} with hS
  have hSm : MeasurableSet S :=
    measurableSet_le ((hcont y).measurable.abs) measurable_const
  have hint : ∀ z, Integrable (fun t => {s : ℝ | 0 < φ s z}.indicator (fun s => θ s z) t) ν := by
    intro z
    refine Integrable.of_bound (((hθm z).indicator (hmeasA z)).aestronglyMeasurable) B
      (ae_of_all _ fun t => ?_)
    rw [Real.norm_eq_abs]
    by_cases h : 0 < φ t z
    · rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s z} from h)]; exact hθB t z
    · rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s z} from h), abs_zero]; exact hB
  have hpt : ∀ t, |{s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
      {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t| ≤ Lθ * δ + B * S.indicator 1 t := by
    intro t
    have hθ1 := hθL t y y'
    have hφ1 := hφL t y y'
    by_cases h1 : 0 < φ t y <;> by_cases h2 : 0 < φ t y'
    · rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y'} from h2)]
      have : 0 ≤ B * S.indicator 1 t := mul_nonneg hB (indicator_nonneg (fun _ _ => zero_le_one) t)
      linarith
    · have htS : t ∈ S := by
        show |φ t y| ≤ L * δ
        rw [abs_of_pos h1]; push Not at h2
        have := (abs_le.mp hφ1).2; linarith
      rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y'} from h2), indicator_of_mem htS]
      simp only [sub_zero, Pi.one_apply, mul_one]
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      linarith [hθB t y]
    · have htS : t ∈ S := by
        show |φ t y| ≤ L * δ
        push Not at h1
        rw [abs_of_nonpos h1]
        have := (abs_le.mp hφ1).1; linarith
      rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y'} from h2), indicator_of_mem htS]
      simp only [zero_sub, abs_neg, Pi.one_apply, mul_one]
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      linarith [hθB t y']
    · rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y'} from h2)]
      have : 0 ≤ B * S.indicator 1 t := mul_nonneg hB (indicator_nonneg (fun _ _ => zero_le_one) t)
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      simp only [sub_self, abs_zero]
      linarith
  have hI : Integrable (fun t => B * S.indicator (1 : ℝ → ℝ) t) ν :=
    ((integrable_const (1:ℝ)).indicator hSm).const_mul B
  have hbound := measureReal_sublevel_le hρ hν (φ := (φ · y)) hα hLδ (hφ y)
  calc |preintJump ν φ θ y - preintJump ν φ θ y'|
      = |∫ t, ({s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
          {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t) ∂ν| := by
        rw [preintJump, preintJump, integral_sub (hint y) (hint y')]
    _ ≤ ∫ t, |{s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
          {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t| ∂ν :=
        abs_integral_le_integral_abs
    _ ≤ ∫ t, (Lθ * δ + B * S.indicator 1 t) ∂ν := by
        refine integral_mono ((hint y).sub (hint y')).abs ?_ hpt
        exact (integrable_const _).add hI
    _ = Lθ * δ + B * ν.real S := by
        rw [integral_add (integrable_const _) hI, integral_const, integral_const_mul,
          integral_indicator_one hSm]
        simp
    _ ≤ Lθ * δ + B * (ρ * (4 * √(L * δ / α))) := by
        gcongr
    _ = Lθ * δ + 4 * ρ * B * √(L * δ / α) := by ring

lemma kink_pt_far {a b c l m : ℝ} (hca : |c - b| ≤ l) (hab : |a - b| ≤ l)
    (hm : |c + a - 2 * b| ≤ m) (hb : l < |b|) :
    |max c 0 + max a 0 - 2 * max b 0| ≤ m := by
  have h0 : 0 ≤ m := (abs_nonneg _).trans hm
  rw [abs_le] at hca hab hm
  rcases lt_or_gt_of_ne (show b ≠ 0 by rintro rfl; simp at hb; linarith [abs_nonneg (c - 0)])
    with hbn | hbp
  · rw [abs_of_neg hbn] at hb
    rw [max_eq_right (by linarith), max_eq_right (by linarith), max_eq_right (by linarith)]
    simpa using h0
  · rw [abs_of_pos hbp] at hb
    rw [max_eq_left (by linarith), max_eq_left (by linarith), max_eq_left (by linarith), abs_le]
    constructor <;> linarith

lemma kink_pt_near {a b c l m : ℝ} (hca : |c - b| ≤ l) (hab : |a - b| ≤ l)
    (hm : |c + a - 2 * b| ≤ m) :
    |max c 0 + max a 0 - 2 * max b 0| ≤ m + l := by
  rw [abs_le] at hca hab hm
  rw [abs_le]
  rcases le_total 0 a with ha | ha <;> rcases le_total 0 b with hb | hb <;>
    rcases le_total 0 c with hc | hc <;>
    simp only [max_eq_left, max_eq_right, ha, hb, hc] <;>
    constructor <;> linarith

theorem kink_second_difference {E : Type*} [SeminormedAddCommGroup E] {ν : Measure ℝ}
    [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ : ℝ → E → ℝ} {α L M : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφint : ∀ y, Integrable (φ · y) ν)
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * ‖y - y'‖)
    (hφM : ∀ t y h, |φ t (y + h) + φ t (y - h) - 2 * φ t y| ≤ M * ‖h‖ ^ 2) (y h : E) :
    |preintKink ν φ (y + h) + preintKink ν φ (y - h) - 2 * preintKink ν φ y| ≤
      M * ‖h‖ ^ 2 + 4 * ρ * L * ‖h‖ * √(L * ‖h‖ / α) := by
  set l := L * ‖h‖
  set m := M * ‖h‖ ^ 2
  have hl : 0 ≤ l := (abs_nonneg _).trans ((hφL 0 y (y + h)).trans (by
    rw [show y - (y + h) = -h by abel, norm_neg]))
  have hcont : Continuous (φ · y) := continuous_of_strong hα (hφ y)
  set S := {t : ℝ | |φ t y| ≤ l} with hS
  have hSm : MeasurableSet S := measurableSet_le (hcont.measurable.abs) measurable_const
  have hint : ∀ z, Integrable (fun t => max (φ t z) 0) ν := fun z => (hφint z).pos_part
  have hI : Integrable (fun t => l * S.indicator (1 : ℝ → ℝ) t) ν :=
    ((integrable_const (1:ℝ)).indicator hSm).const_mul l
  have hpt : ∀ t, |max (φ t (y + h)) 0 + max (φ t (y - h)) 0 - 2 * max (φ t y) 0| ≤
      m + l * S.indicator 1 t := by
    intro t
    have hca : |φ t (y + h) - φ t y| ≤ l := by
      have := hφL t (y + h) y; rwa [show y + h - y = h by abel] at this
    have hab : |φ t (y - h) - φ t y| ≤ l := by
      have := hφL t (y - h) y; rwa [show y - h - y = -h by abel, norm_neg] at this
    have hmm : |φ t (y + h) + φ t (y - h) - 2 * φ t y| ≤ m := hφM t y h
    by_cases htS : t ∈ S
    · rw [indicator_of_mem htS, Pi.one_apply, mul_one]
      exact kink_pt_near hca hab hmm
    · rw [indicator_of_notMem htS, mul_zero, add_zero]
      exact kink_pt_far hca hab hmm (lt_of_not_ge htS)
  have hbound := measureReal_sublevel_le hρ hν (φ := (φ · y)) hα hl (hφ y)
  calc |preintKink ν φ (y + h) + preintKink ν φ (y - h) - 2 * preintKink ν φ y|
      = |∫ t, (max (φ t (y + h)) 0 + max (φ t (y - h)) 0 - 2 * max (φ t y) 0) ∂ν| := by
        have h1 : Integrable (fun t => max (φ t (y + h)) 0 + max (φ t (y - h)) 0) ν :=
          (hint _).add (hint _)
        have h2 : Integrable (fun t => 2 * max (φ t y) 0) ν := (hint _).const_mul 2
        rw [integral_sub h1 h2, integral_add (hint _) (hint _), integral_const_mul]
        rfl
    _ ≤ ∫ t, |max (φ t (y + h)) 0 + max (φ t (y - h)) 0 - 2 * max (φ t y) 0| ∂ν :=
        abs_integral_le_integral_abs
    _ ≤ ∫ t, (m + l * S.indicator 1 t) ∂ν := by
        refine integral_mono (((hint _).add (hint _)).sub ((hint _).const_mul 2)).abs ?_ hpt
        exact (integrable_const _).add hI
    _ = m + l * ν.real S := by
        rw [integral_add (integrable_const _) hI, integral_const, integral_const_mul,
          integral_indicator_one hSm]
        simp
    _ ≤ m + l * (ρ * (4 * √(l / α))) := by gcongr
    _ = M * ‖h‖ ^ 2 + 4 * ρ * L * ‖h‖ * √(L * ‖h‖ / α) := by ring

end Cubature.Aux
