import Theory.Model

open MeasureTheory Set Filter
open scoped BigOperators ENNReal Pointwise

namespace Cubature

theorem density_pos {d : ℕ} (x : Point d) : 0 < density x := by
  exact Finset.prod_pos fun i _ ↦ ProbabilityTheory.gaussianPDFReal_pos 0 1 (x i) one_ne_zero

theorem continuous_normSq {d : ℕ} : Continuous (normSq (d := d)) := by
  unfold normSq
  fun_prop

theorem continuous_polynomial {d : ℕ} (A : Coefficients d) (b : Point d) :
    Continuous (polynomial A b) := by
  unfold polynomial quadratic linear
  fun_prop

theorem measurable_region {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) :
    MeasurableSet (region ρ E) :=
  (isOpen_lt ((continuous_polynomial E.A E.b).comp (continuous_const_smul _))
    continuous_const).measurableSet

theorem density_measurable {d : ℕ} : Measurable (density (d := d)) := by
  unfold density
  exact Finset.measurable_prod _ fun i _ ↦
    (ProbabilityTheory.measurable_gaussianPDFReal 0 1).comp (measurable_pi_apply i)

theorem gaussian_eq_pi (d : ℕ) :
    gaussian d = Measure.pi (fun _ : Fin d ↦ ProbabilityTheory.gaussianReal 0 1) := by
  symm
  apply Measure.pi_eq
  intro s hs
  let p : Fin d → ℝ → ℝ := fun i ↦ (s i).indicator (ProbabilityTheory.gaussianPDFReal 0 1)
  have hp : ∀ i, Integrable (p i) volume :=
    fun i ↦ (ProbabilityTheory.integrable_gaussianPDFReal 0 1).indicator (hs i)
  have hprod : Integrable (fun x : Point d ↦ ∏ i, p i (x i)) volume := by
    rw [volume_pi]
    exact Integrable.fintype_prod hp
  have hval (x : Point d) :
      (univ.pi s).indicator (fun x : Point d ↦ ENNReal.ofReal (density x)) x =
        ENNReal.ofReal (∏ i, p i (x i)) := by
    by_cases hx : x ∈ univ.pi s
    · simp only [indicator_of_mem hx]
      congr 1
      exact Finset.prod_congr rfl (fun i _ ↦ (indicator_of_mem (hx i (mem_univ i)) _).symm)
    · obtain ⟨i, hi⟩ : ∃ i, x i ∉ s i := by simpa [Set.mem_pi, not_forall] using hx
      rw [indicator_of_notMem hx, Finset.prod_eq_zero (Finset.mem_univ i),
        ENNReal.ofReal_zero]
      exact indicator_of_notMem hi _
  have hnonneg : ∀ x : Point d, 0 ≤ ∏ i, p i (x i) := fun x ↦
    Finset.prod_nonneg fun i _ ↦ by
      by_cases h : x i ∈ s i
      · simpa [p, h] using ProbabilityTheory.gaussianPDFReal_nonneg 0 1 (x i)
      · simp [p, h]
  have hmass (i : Fin d) :
      ProbabilityTheory.gaussianReal 0 1 (s i) = ENNReal.ofReal (∫ x, p i x) := by
    rw [ProbabilityTheory.gaussianReal_apply_eq_integral 0 one_ne_zero,
      integral_indicator (hs i)]
  have hmeas : MeasurableSet (univ.pi s) :=
    MeasurableSet.pi countable_univ (fun i _ => hs i)
  rw [gaussian, withDensity_apply _ hmeas, ← lintegral_indicator hmeas]
  simp_rw [hval]
  rw [← ofReal_integral_eq_lintegral_ofReal hprod (ae_of_all _ hnonneg),
    integral_fintype_prod_volume_eq_prod]
  simp_rw [hmass]
  exact ENNReal.ofReal_prod_of_nonneg fun i _ ↦ integral_nonneg (fun x ↦ by
    by_cases h : x ∈ s i
    · simpa [p, h] using ProbabilityTheory.gaussianPDFReal_nonneg 0 1 x
    · simp [p, h])

instance gaussian_isProbability (d : ℕ) : IsProbabilityMeasure (gaussian d) := by
  rw [gaussian_eq_pi]
  infer_instance

theorem probability_nonneg {d : ℕ} (s : Set (Point d)) : 0 ≤ probability s :=
  ENNReal.toReal_nonneg

theorem probability_le_one {d : ℕ} (s : Set (Point d)) : probability s ≤ 1 := by
  unfold probability
  simpa using (measureReal_mono (μ := gaussian d) (subset_univ s))

theorem probability_mono {d : ℕ} {s t : Set (Point d)} (hst : s ⊆ t) :
    probability s ≤ probability t := measureReal_mono hst

theorem probability_eq_payoff_integral {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) :
    probability (region ρ E) = ∫ x, payoff ρ E x ∂(gaussian d) := by
  classical
  have h : payoff ρ E = (region ρ E).indicator (fun _ ↦ (1 : ℝ)) := by
    funext x
    by_cases hx : x ∈ region ρ E <;> simp [payoff, answerBit, hx]
  rw [h, integral_indicator (measurable_region ρ E), setIntegral_const]
  simp [probability]

theorem density_smul_le {d : ℕ} (x : Point d) {t : ℝ} (ht : 1 ≤ t) :
    density (t • x) ≤ density x := by
  unfold density
  apply Finset.prod_le_prod₀ (fun i _ => ProbabilityTheory.gaussianPDFReal_nonneg 0 1 ((t • x) i))
  intro i _
  simp only [ProbabilityTheory.gaussianPDFReal_def, Pi.smul_apply, smul_eq_mul,
    sub_zero, NNReal.coe_one, mul_one]
  apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by positivity)
  have ht2 : 1 ≤ t^2 := by nlinarith [sq_nonneg (t-1)]
  nlinarith [mul_le_mul_of_nonneg_right ht2 (sq_nonneg (x i))]

theorem measurableSet_dilate {d : ℕ} {s : Set (Point d)} (hs : MeasurableSet s)
    {t : ℝ} (ht : t ≠ 0) : MeasurableSet (t • s) := by
  rw [← preimage_smul_inv₀ ht]
  exact hs.preimage (measurable_const_smul _)

/-- Change of variables for a dilation in the coordinates used by the oracle. -/
theorem gaussian_dilate {d : ℕ} {t : ℝ} (ht : 0 < t)
    {s : Set (Point d)} (hs : MeasurableSet s) :
    gaussian d (t • s) =
      ENNReal.ofReal (t^d) * ∫⁻ x in s, ENNReal.ofReal (density (t • x)) := by
  let g : Point d → ℝ≥0∞ := s.indicator (fun x ↦ ENNReal.ofReal (density (t • x)))
  have hg : Measurable g := (density_measurable.comp (measurable_const_smul t)).ennreal_ofReal.indicator hs
  have hv :
      Measure.map (fun x : Point d ↦ t⁻¹ • x) volume = ENNReal.ofReal (t^d) • volume := by
    rw [Measure.map_addHaar_smul volume (inv_ne_zero ht.ne')]
    simp [Module.finrank_fintype_fun_eq_card, inv_pow, abs_of_pos (pow_pos ht d)]
  have hmap := lintegral_map (μ := (volume : Measure (Point d))) hg
    (measurable_const_smul (t⁻¹ : ℝ))
  have hcomp : (fun x : Point d ↦ g (t⁻¹ • x)) =
      (t • s).indicator (fun x ↦ ENNReal.ofReal (density x)) := by
    rw [← preimage_smul_inv₀ ht.ne']
    funext x
    by_cases hx : t⁻¹ • x ∈ s
    · simp [g, hx, smul_smul, mul_inv_cancel₀ ht.ne']
    · simp [g, hx]
  rw [hv, lintegral_smul_measure, hcomp, lintegral_indicator (measurableSet_dilate hs ht.ne')] at hmap
  rw [gaussian, withDensity_apply _ (measurableSet_dilate hs ht.ne'), ← hmap,
    smul_eq_mul, lintegral_indicator hs]

theorem probability_dilate_le {d : ℕ} {t : ℝ} (ht : 1 ≤ t)
    {s : Set (Point d)} (hs : MeasurableSet s) :
    probability (t • s) ≤ t^d * probability s := by
  have ht0 : 0 < t := by linarith
  have hp := gaussian_dilate ht0 hs
  have hb : (gaussian d) (t • s) ≤ ENNReal.ofReal (t^d) * gaussian d s := by
    rw [hp, gaussian, withDensity_apply _ hs]
    exact mul_le_mul_right (lintegral_mono fun x ↦ ENNReal.ofReal_le_ofReal
      (density_smul_le x ht)) _
  have hr := ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top (measure_ne_top _ _)) hb
  simpa [probability, Measure.real, ENNReal.toReal_mul, ENNReal.toReal_ofReal (pow_nonneg ht0.le d)]
    using hr

theorem probability_dilate_ge {d : ℕ} {t : ℝ} (ht : 0 < t) (ht1 : t ≤ 1)
    {s : Set (Point d)} (hs : MeasurableSet s) :
    t^d * probability s ≤ probability (t • s) := by
  have h := probability_dilate_le (one_le_inv₀ ht |>.mpr ht1)
    (measurableSet_dilate hs ht.ne')
  rw [smul_smul, inv_mul_cancel₀ ht.ne', one_smul, inv_pow] at h
  have h' := mul_le_mul_of_nonneg_left h (pow_nonneg ht.le d)
  rwa [← mul_assoc, mul_inv_cancel₀ (pow_ne_zero d ht.ne'), one_mul] at h'

theorem isOpen_arithmeticBall {d : ℕ} (r : ℝ) :
    IsOpen (arithmeticBall (d := d) r) := isOpen_lt continuous_normSq continuous_const

theorem volume_absolutelyContinuous_gaussian (d : ℕ) :
    (volume : Measure (Point d)) ≪ gaussian d := by
  unfold gaussian
  apply withDensity_absolutelyContinuous'
  · exact density_measurable.ennreal_ofReal.aemeasurable
  · exact ae_of_all _ fun x ↦ (ENNReal.ofReal_pos.mpr (density_pos x)).ne'

end Cubature
