import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.MeasureTheory.Measure.Haar.NormedSpace
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators ENNReal

namespace Cubature

abbrev Point (d : ℕ) := Fin d → ℝ

noncomputable def gamma (d : ℕ) : Measure (Point d) :=
  Measure.pi fun _ ↦ gaussianReal 0 1

noncomputable instance gammaProbability (d : ℕ) : IsProbabilityMeasure (gamma d) := by
  unfold gamma
  infer_instance

noncomputable def density (d : ℕ) (x : Point d) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) ^ d * Real.exp (-(∑ i, (x i)^2) / 2)

lemma density_nonneg (x : Point d) : 0 ≤ density d x := by
  unfold density
  positivity

lemma density_eq_product (x : Point d) :
    density d x = ∏ i, gaussianPDFReal 0 1 (x i) := by
  simp only [density, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero, one_div]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_fin, ← Real.exp_sum]
  rw [← Finset.sum_div, Finset.sum_neg_distrib]

lemma measurable_density (d : ℕ) : Measurable (density d) := by
  unfold density
  fun_prop

lemma gamma_eq_density (d : ℕ) :
    gamma d = volume.withDensity (fun x ↦ ENNReal.ofReal (density d x)) := by
  unfold gamma
  apply Measure.pi_eq
  intro s hs
  rw [withDensity_apply _ (MeasurableSet.pi countable_univ (fun i _ ↦ hs i))]
  let h (i : Fin d) := (s i).indicator (gaussianPDFReal 0 1)
  have hI : ∀ i, Integrable (h i) volume :=
    fun i ↦ (integrable_gaussianPDFReal 0 1).indicator (hs i)
  have hN : ∀ i t, 0 ≤ h i t :=
    fun i t ↦ indicator_nonneg (fun _ _ ↦ gaussianPDFReal_nonneg _ _ _) _
  have hb (x : Point d) :
      (Set.pi univ s).indicator (density d) x = ∏ i, h i (x i) := by
    classical
    by_cases hx : x ∈ Set.pi univ s
    · rw [indicator_of_mem hx, density_eq_product]
      apply Finset.prod_congr rfl
      intro i _
      exact (indicator_of_mem (mem_univ_pi.mp hx i) _).symm
    · rw [indicator_of_notMem hx]
      obtain ⟨i, hi⟩ := not_forall.mp (mt mem_univ_pi.mpr hx)
      symm
      exact Finset.prod_eq_zero (Finset.mem_univ i) (indicator_of_notMem hi _)
  calc
    ∫⁻ x in Set.pi univ s, ENNReal.ofReal (density d x)
        = ∫⁻ x : Point d, ENNReal.ofReal (∏ i, h i (x i)) := by
      rw [← lintegral_indicator (MeasurableSet.pi countable_univ (fun i _ ↦ hs i))]
      congr with x
      rw [← hb]
      by_cases h : x ∈ Set.pi univ s <;> simp [h]
    _ = ENNReal.ofReal (∫ x : Point d, ∏ i, h i (x i)) := by
      rw [← ofReal_integral_eq_lintegral_ofReal]
      · rw [volume_pi]
        exact Integrable.fintype_prod hI
      · exact ae_of_all _ (fun x ↦ Finset.prod_nonneg (fun i _ ↦ hN i (x i)))
    _ = ∏ i, gaussianReal 0 1 (s i) := by
      rw [integral_fintype_prod_volume_eq_prod]
      simp_rw [gaussianReal_apply_eq_integral _ one_ne_zero,
        ← integral_indicator (hs _)]
      exact ENNReal.ofReal_prod_of_nonneg
        (fun i _ ↦ integral_nonneg (hN i))

/-- Homothety with center `a`; the coordinates are in physical space. -/
def dilate (a : Point d) (t : ℝ) (x : Point d) : Point d :=
  a + t • (x - a)

lemma measurable_dilate (a : Point d) (t : ℝ) : Measurable (dilate a t) := by
  unfold dilate
  fun_prop

lemma dilate_cancel (a : Point d) {t : ℝ} (ht : t ≠ 0) (x : Point d) :
    dilate a t (dilate a t⁻¹ x) = x := by
  ext i
  change a i + t * (a i + t⁻¹ * (x i-a i) - a i) = x i
  rw [add_sub_cancel_left, ← mul_assoc, mul_inv_cancel₀ ht, one_mul, add_sub_cancel]

noncomputable def dilationFactor (a : Point d) (t : ℝ) : ℝ :=
  t^d * Real.exp (((t-1)/(t+1)) * (∑ i, (a i)^2) / 2)

lemma one_le_dilationFactor (a : Point d) {t : ℝ} (ht : 1 ≤ t) :
    1 ≤ dilationFactor a t := by
  unfold dilationFactor
  have ht' : 1 ≤ t^d := one_le_pow₀ ht
  have hz : 0 ≤ ((t-1)/(t+1)) * (∑ i, (a i)^2) / 2 := by
    have : 0 ≤ t-1 := sub_nonneg.mpr ht
    have : 0 < t+1 := by linarith
    positivity
  exact one_le_mul_of_one_le_of_one_le ht' (Real.one_le_exp_iff.mpr hz)

private lemma affine_square (a x t : ℝ) (ht : 1 ≤ t) :
    x^2 - (a+t*(x-a))^2 ≤ ((t-1)/(t+1))*a^2 := by
  have ht' : 0 < t+1 := by linarith
  have hh : 0 ≤ (t-1)*((t+1)*x-t*a)^2 :=
    mul_nonneg (sub_nonneg.mpr ht) (sq_nonneg _)
  have H : (t+1)*(x^2-(a+t*(x-a))^2)+(t-1)*((t+1)*x-t*a)^2 =
      (t-1)*a^2 := by ring
  rw [div_mul_eq_mul_div]
  exact (le_div_iff₀ ht').2 (by nlinarith only [hh, H])

private lemma density_dilate (a x : Point d) (t : ℝ) (ht : 1 ≤ t) :
    density d (dilate a t x) ≤
      Real.exp (((t-1)/(t+1)) * (∑ i, (a i)^2) / 2) * density d x := by
  have h := Finset.sum_le_sum
    (s := (Finset.univ : Finset (Fin d))) (fun i _ ↦ affine_square (a i) (x i) t ht)
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum] at h
  unfold density
  have hc : 0 ≤ (1 / Real.sqrt (2 * Real.pi)) ^ d := by positivity
  rw [mul_left_comm, ← Real.exp_add]
  apply mul_le_mul_of_nonneg_left _ hc
  apply Real.exp_le_exp.mpr
  change -(∑ i, (a i+t*(x i-a i))^2)/2 ≤ _
  linarith

private lemma volume_dilate_inv (a : Point d) {t : ℝ} (ht : 0 < t) :
    (volume : Measure (Point d)).map (dilate a t⁻¹) =
      ENNReal.ofReal (t^d) • volume := by
  have h1 : Measurable (fun x : Point d ↦ x + -a) := by fun_prop
  have h2 : Measurable (fun x : Point d ↦ t⁻¹ • x) := by fun_prop
  have h3 : Measurable (fun x : Point d ↦ x + a) := by fun_prop
  have e : dilate a t⁻¹ = (fun x : Point d ↦ x + a) ∘
      ((fun x : Point d ↦ t⁻¹ • x) ∘ (fun x : Point d ↦ x + -a)) := by
    ext x i
    change a i + t⁻¹*(x i-a i) = t⁻¹*(x i + -(a i))+a i
    ring
  rw [e, ← Measure.map_map h3 (h2.comp h1), ← Measure.map_map h2 h1,
    map_add_right_eq_self (volume : Measure (Point d)) (-a),
    Measure.map_addHaar_smul _ (inv_ne_zero ht.ne'),
    Measure.map_smul _ h3.aemeasurable,
    map_add_right_eq_self (volume : Measure (Point d)) a]
  simp [inv_pow, abs_of_nonneg (pow_nonneg ht.le _)]

/-- Upper Gaussian mass distortion for dilation about an arbitrary center. -/
theorem gaussian_preimage_dilation_le
    (a : Point d) {t : ℝ} (ht : 1 ≤ t) {s : Set (Point d)} (hs : MeasurableSet s) :
    (gamma d).real ((dilate a t⁻¹) ⁻¹' s) ≤ dilationFactor a t * (gamma d).real s := by
  have hp : 0 < t := by linarith
  let k := Real.exp (((t-1)/(t+1)) * (∑ i, (a i)^2)/2)
  have hk : 0 ≤ k := Real.exp_nonneg _
  have hm : Measurable (fun x ↦ ENNReal.ofReal (density d (dilate a t x))) :=
    (measurable_density d |>.ennreal_ofReal).comp (measurable_dilate _ _)
  have h :
      gamma d ((dilate a t⁻¹) ⁻¹' s) ≤
        ENNReal.ofReal (dilationFactor a t) * gamma d s := by
    rw [gamma_eq_density, withDensity_apply _ (hs.preimage (measurable_dilate _ _)),
      withDensity_apply _ hs, dilationFactor]
    rw [← show _ = ∫⁻ x in (dilate a t⁻¹) ⁻¹' s,
        ENNReal.ofReal (density d x) from
      (setLIntegral_map hs hm (measurable_dilate _ _)).trans
        (lintegral_congr (fun x ↦ congrArg (fun y ↦ ENNReal.ofReal (density d y))
          (dilate_cancel a hp.ne' x)))]
    rw [volume_dilate_inv a hp, Measure.restrict_smul, lintegral_smul_measure]
    change ENNReal.ofReal (t^d) * _ ≤ ENNReal.ofReal (t^d*k) * _
    rw [ENNReal.ofReal_mul (pow_nonneg hp.le _), mul_assoc, ← lintegral_const_mul _ (measurable_density d).ennreal_ofReal]
    gcongr with x
    rw [← ENNReal.ofReal_mul hk]
    apply ENNReal.ofReal_le_ofReal
    exact density_dilate a x t ht
  unfold Measure.real
  have hfinite : ENNReal.ofReal (dilationFactor a t) * gamma d s ≠ ∞ :=
    ENNReal.mul_ne_top ENNReal.ofReal_ne_top (measure_ne_top _ _)
  simpa [ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (le_trans zero_le_one (one_le_dilationFactor a ht))]
    using ENNReal.toReal_mono hfinite h

end Cubature
