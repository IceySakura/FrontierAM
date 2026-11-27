import Theory.BallGeometry
import Mathlib.MeasureTheory.Measure.Haar.NormedSpace
import Mathlib.Algebra.Order.Ring.Pow

/-! Relative mass bounds for Gaussian dilations about the origin. -/

open MeasureTheory ProbabilityTheory Set
open scoped Pointwise ENNReal

namespace Cubature

noncomputable section

def gaussianSetMass {d : ℕ} (S : Set (GaussianSpace d)) : ℝ :=
  (stdGaussian (GaussianSpace d)).real S

theorem gaussianSetMass_nonneg {d : ℕ} (S : Set (GaussianSpace d)) :
    0 ≤ gaussianSetMass S := measureReal_nonneg

theorem gaussianSetMass_mono {d : ℕ} {S T : Set (GaussianSpace d)} (h : S ⊆ T) :
    gaussianSetMass S ≤ gaussianSetMass T :=
  ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono h)

theorem gaussianSetMass_integral {d : ℕ} {S : Set (GaussianSpace d)} (hS : MeasurableSet S) :
    gaussianSetMass S = ∫ x in S, gaussianDensity d ‖x‖ := by
  have hi : IntegrableOn (fun x : GaussianSpace d => gaussianDensity d ‖x‖) S :=
    (gaussianDensity_norm_integrable d).integrableOn
  have hn : ∀ᵐ x ∂(volume : Measure (GaussianSpace d)).restrict S, 0 ≤ gaussianDensity d ‖x‖ :=
    Filter.Eventually.of_forall (fun _ => gaussianDensity_nonneg d _)
  rw [gaussianSetMass, measureReal_def, stdGaussian_density, withDensity_apply _ hS,
    ← ofReal_integral_eq_lintegral_ofReal hi hn, ENNReal.toReal_ofReal (integral_nonneg_of_ae hn)]

private theorem gaussianSetMass_dilate {d : ℕ} {S : Set (GaussianSpace d)} (hS : MeasurableSet S)
    {t : ℝ} (ht : 0 < t) :
    gaussianSetMass (t • S) = t^d * ∫ x in S, gaussianDensity d ‖t • x‖ := by
  have hT : MeasurableSet (t • S) := by
    change MeasurableSet ((fun x : GaussianSpace d => t • x) '' S)
    exact (Homeomorph.smul (Units.mk0 t ht.ne')).measurableEmbedding.measurableSet_image.mpr hS
  have hf : Module.finrank ℝ (GaussianSpace d) = d := by simp [GaussianSpace]
  have he := Measure.setIntegral_comp_smul (volume : Measure (GaussianSpace d))
    (fun x : GaussianSpace d => gaussianDensity d ‖x‖) S ht.ne'
  rw [hf, abs_of_nonneg (inv_nonneg.mpr (pow_nonneg ht.le _)), smul_eq_mul] at he
  rw [gaussianSetMass_integral hT, he, ← mul_assoc, mul_inv_cancel₀ (pow_ne_zero _ ht.ne'), one_mul]

private theorem density_antitone {d : ℕ} {x y : ℝ} (hx : 0 ≤ x) (h : x ≤ y) :
    gaussianDensity d y ≤ gaussianDensity d x := by
  unfold gaussianDensity
  refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by unfold gaussianConstant; positivity)
  have he := pow_le_pow_left₀ hx h 2
  linarith

private theorem gaussianDensity_dilate_integrable {d : ℕ} {t : ℝ} (ht : t ≠ 0) :
    Integrable (fun x : GaussianSpace d => gaussianDensity d ‖t • x‖) :=
  (integrable_comp_smul_iff volume (fun x : GaussianSpace d => gaussianDensity d ‖x‖) ht).2
    (gaussianDensity_norm_integrable d)

theorem gaussianSetMass_expand_le {d : ℕ} {S : Set (GaussianSpace d)} (hS : MeasurableSet S)
    {t : ℝ} (ht : 1 ≤ t) :
    gaussianSetMass (t • S) ≤ t^d * gaussianSetMass S := by
  have hpos : 0 < t := zero_lt_one.trans_le ht
  rw [gaussianSetMass_dilate hS hpos, gaussianSetMass_integral hS]
  refine mul_le_mul_of_nonneg_left (integral_mono
    (gaussianDensity_dilate_integrable hpos.ne').restrict (gaussianDensity_norm_integrable d).restrict
    (fun x => ?_)) (pow_nonneg hpos.le _)
  rw [norm_smul, Real.norm_of_nonneg hpos.le]
  exact density_antitone (norm_nonneg x) (le_mul_of_one_le_left (norm_nonneg x) ht)

theorem gaussianSetMass_contract_ge {d : ℕ} {S : Set (GaussianSpace d)} (hS : MeasurableSet S)
    {t : ℝ} (ht₀ : 0 < t) (ht : t ≤ 1) :
    t^d * gaussianSetMass S ≤ gaussianSetMass (t • S) := by
  rw [gaussianSetMass_dilate hS ht₀, gaussianSetMass_integral hS]
  refine mul_le_mul_of_nonneg_left (integral_mono
    (gaussianDensity_norm_integrable d).restrict (gaussianDensity_dilate_integrable ht₀.ne').restrict
    (fun x => ?_)) (pow_nonneg ht₀.le _)
  rw [norm_smul, Real.norm_of_nonneg ht₀.le]
  exact density_antitone (mul_nonneg ht₀.le (norm_nonneg x))
    (mul_le_of_le_one_left (norm_nonneg x) ht)

/-- Small multiplicative geometric uncertainty implies a relative, not absolute, mass bound. -/
theorem gaussianSetMass_relative {d : ℕ} (hd : 0 < d) {S T : Set (GaussianSpace d)}
    (hS : MeasurableSet S) {η : ℝ} (hη : 0 ≤ η) (hsmall : (d:ℝ)*η ≤ 1/2)
    (hlo : (1-η) • S ⊆ T) (hhi : T ⊆ (1-η)⁻¹ • S) :
    |gaussianSetMass T - gaussianSetMass S| ≤ (2*(d:ℝ)*η) * gaussianSetMass S := by
  let t := 1-η
  let b := (d:ℝ)*η
  have hb : 0 ≤ b := mul_nonneg (Nat.cast_nonneg _) hη
  have hb' : b ≤ 1/2 := hsmall
  have hd' : (1:ℝ) ≤ d := by exact_mod_cast Nat.succ_le_of_lt hd
  have hle : η ≤ b := le_mul_of_one_le_left hη hd'
  have ht : 0 < t := by dsimp [t]; linarith
  have ht' : t ≤ 1 := by dsimp [t]; linarith
  have hInv : 1 ≤ t⁻¹ := one_le_inv₀ ht |>.mpr ht'
  have hBern : 1-b ≤ t^d := by
    have h := one_add_mul_le_pow (a := -η) (by linarith : (-2:ℝ) ≤ -η) d
    simpa only [t,b,sub_eq_add_neg,mul_neg] using h
  have hPowInv : (t⁻¹)^d ≤ 1+2*b := by
    rw [inv_pow]
    have hp : 0 < 1-b := by linarith
    refine ((inv_le_inv₀ (pow_pos ht _) hp).mpr hBern).trans ?_
    rw [inv_eq_one_div, div_le_iff₀ hp]
    nlinarith [mul_nonneg hb (show 0 ≤ 1-2*b by linarith)]
  have hM := gaussianSetMass_nonneg S
  have h1 : (1-b) * gaussianSetMass S ≤ gaussianSetMass T :=
    (mul_le_mul_of_nonneg_right hBern hM).trans
      ((gaussianSetMass_contract_ge hS ht ht').trans (gaussianSetMass_mono hlo))
  have h2 : gaussianSetMass T ≤ (1+2*b) * gaussianSetMass S :=
    (gaussianSetMass_mono hhi).trans
      ((gaussianSetMass_expand_le hS hInv).trans (mul_le_mul_of_nonneg_right hPowInv hM))
  have hmul : 0 ≤ b * gaussianSetMass S := mul_nonneg hb hM
  rw [abs_le]
  rw [show (2:ℝ)*(d:ℝ)*η = 2*b by unfold b; ring]
  constructor <;> nlinarith

end

end Cubature
