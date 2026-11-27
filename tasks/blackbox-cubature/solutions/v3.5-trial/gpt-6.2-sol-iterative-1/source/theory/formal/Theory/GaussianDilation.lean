import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Relative Gaussian mass under a dilation about an arbitrary anchor. -/

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators ENNReal NNReal

namespace Cubature

abbrev Vec (d : ℕ) := Fin d → ℝ

/-- The joint measure of independent standard normal coordinates. -/
noncomputable def gaussian (d : ℕ) : Measure (Vec d) :=
  Measure.pi fun _ : Fin d => gaussianReal 0 1

instance (d : ℕ) : IsProbabilityMeasure (gaussian d) := by
  unfold gaussian
  infer_instance

noncomputable def contract {d : ℕ} (a : Vec d) (s : ℝ) (x : Vec d) : Vec d :=
  fun i => a i + (x i - a i) / s

noncomputable def dilationFactor {d : ℕ} (a : Vec d) (s : ℝ) : ℝ :=
  s ^ d * Real.exp ((s - 1) / (2 * (s + 1)) * ∑ i, (a i) ^ 2)

private lemma gaussian_map_contract_one (a s : ℝ) (hs : 1 ≤ s) :
    (gaussianReal 0 1).map (fun x : ℝ => a + (x - a) / s) ≤
      ENNReal.ofReal (s * Real.exp ((s - 1) / (2 * (s + 1)) * a ^ 2)) • gaussianReal 0 1 := by
  have hs0 : 0 < s := lt_of_lt_of_le zero_lt_one hs
  let v : ℝ≥0 := ⟨(s⁻¹) ^ 2, sq_nonneg _⟩
  have hv : v ≠ 0 := by
    intro hz
    have hh := congrArg NNReal.toReal hz
    change (s⁻¹) ^ 2 = 0 at hh
    exact (by positivity : 0 < (s⁻¹) ^ 2).ne' hh
  have hm : (fun x : ℝ => a + (x - a) / s) =
      (fun y : ℝ => y + (a - a / s)) ∘ (fun x : ℝ => s⁻¹ * x) := by
    funext x
    simp only [Function.comp_apply, div_eq_mul_inv]
    ring
  rw [hm, ← Measure.map_map (by fun_prop) (by fun_prop),
    gaussianReal_map_const_mul, mul_zero, mul_one, gaussianReal_map_add_const, zero_add]
  change gaussianReal (a - a / s) v ≤ _
  rw [gaussianReal_of_var_ne_zero _ hv, gaussianReal_of_var_ne_zero _ one_ne_zero,
    ← withDensity_smul _ (measurable_gaussianPDF 0 1)]
  apply withDensity_mono
  filter_upwards with x
  simp only [Pi.smul_apply, smul_eq_mul, gaussianPDF_def]
  rw [← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  have root : Real.sqrt (2 * Real.pi * (v : ℝ)) = Real.sqrt (2 * Real.pi) * s⁻¹ := by
    change Real.sqrt (2 * Real.pi * (s⁻¹) ^ 2) = _
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
  have normFactor : (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ =
      s * (Real.sqrt (2 * Real.pi))⁻¹ := by rw [root, mul_inv_rev, inv_inv]
  have hexp : -(x - (a - a / s)) ^ 2 / (2 * (v : ℝ)) =
      -(s * x + (1 - s) * a) ^ 2 / 2 := by
    change -(x - (a - a / s)) ^ 2 / (2 * (s⁻¹) ^ 2) = _
    field_simp
    <;> ring
  have hpoly :
      (s + 1) * ((s * x + (1 - s) * a) ^ 2 - x ^ 2) + (s - 1) * a ^ 2 =
        (s - 1) * ((s + 1) * x - s * a) ^ 2 := by ring
  have hsq :
      0 ≤ (s + 1) * ((s * x + (1 - s) * a) ^ 2 - x ^ 2) + (s - 1) * a ^ 2 := by
    rw [hpoly]
    exact mul_nonneg (sub_nonneg.mpr hs) (sq_nonneg _)
  have he : -(s * x + (1 - s) * a) ^ 2 / 2 ≤
      (s - 1) / (2 * (s + 1)) * a ^ 2 + (-x ^ 2 / 2) := by
    have hs1 : 0 < 2 * (s + 1) := by positivity
    apply (sub_le_iff_le_add).mp
    rw [show (s - 1) / (2 * (s + 1)) * a ^ 2 =
      ((s - 1) * a ^ 2) / (2 * (s + 1)) by ring]
    -- The division is safe since s >= 1.
    apply (le_div_iff₀ hs1).2
    nlinarith only [hsq]
  simp only [gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
  rw [normFactor, hexp]
  calc
    s * (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-(s * x + (1 - s) * a) ^ 2 / 2) ≤
        s * (Real.sqrt (2 * Real.pi))⁻¹ *
          Real.exp ((s - 1) / (2 * (s + 1)) * a ^ 2 + (-x ^ 2 / 2)) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr he) (by positivity)
    _ = _ := by rw [Real.exp_add]; ring

lemma pi_mono_coord {d : ℕ} {μ ν : Fin d → Measure ℝ}
    (h : ∀ i, μ i ≤ ν i) : Measure.pi μ ≤ Measure.pi ν := by
  rw [Measure.pi, Measure.pi]
  apply Measure.le_iff.mpr
  intro set hset
  rw [toMeasure_apply _ _ hset, toMeasure_apply _ _ hset]
  apply (OuterMeasure.le_pi.mpr ?_) set
  intro s _
  exact (OuterMeasure.pi_pi_le _ _).trans
    (Finset.prod_le_prod' fun i _ => h i (s i))

lemma pi_smul_coord {d : ℕ} (μ : Fin d → Measure ℝ)
    [∀ i, SigmaFinite (μ i)] (c : Fin d → ℝ≥0∞) (hc : ∀ i, c i ≠ ∞) :
    Measure.pi (fun i => c i • μ i) = (∏ i, c i) • Measure.pi μ := by
  have (i : Fin d) : SigmaFinite (c i • μ i) := by
    rw [← ENNReal.coe_toNNReal (hc i)]
    change SigmaFinite ((c i).toNNReal • μ i)
    infer_instance
  apply Measure.pi_eq
  intro s _
  simp only [Measure.smul_apply, smul_eq_mul, Measure.pi_pi, Finset.prod_mul_distrib]

lemma gaussian_map_contract {d : ℕ} (a : Vec d) (s : ℝ) (hs : 1 ≤ s) :
    (gaussian d).map (contract a s) ≤ ENNReal.ofReal (dilationFactor a s) • gaussian d := by
  let C : Fin d → ℝ := fun i => s * Real.exp ((s - 1) / (2 * (s + 1)) * (a i) ^ 2)
  unfold gaussian contract
  rw [Measure.pi_map_pi (fun _ => (show Measurable (fun x : ℝ => _ + (x - _) / s)
    from by fun_prop).aemeasurable)]
  refine (pi_mono_coord (fun i => gaussian_map_contract_one (a i) s hs)).trans ?_
  rw [pi_smul_coord _ (fun i => ENNReal.ofReal (C i)) (fun _ => ENNReal.ofReal_ne_top)]
  have hc (i : Fin d) : 0 ≤ C i :=
    mul_nonneg (zero_le_one.trans hs) (Real.exp_nonneg _)
  rw [← ENNReal.ofReal_prod_of_nonneg (fun i _ => hc i)]
  have prod_eq : (∏ i, C i) = dilationFactor a s := by
    simp only [C, dilationFactor, Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ,
      Fintype.card_fin, ← Real.exp_sum, Finset.mul_sum]
  rw [prod_eq]

lemma gaussian_mass_contract_le {d : ℕ} {a : Vec d} {s : ℝ} (hs : 1 ≤ s)
    {E : Set (Vec d)} (hE : MeasurableSet E) :
    (gaussian d).real (contract a s ⁻¹' E) ≤ dilationFactor a s * (gaussian d).real E := by
  have hm : Measurable (contract a s) := by
    apply Measurable.of_eval
    intro i
    unfold contract
    fun_prop
  have hp : 0 ≤ dilationFactor a s := by
    unfold dilationFactor
    exact mul_nonneg (pow_nonneg (zero_le_one.trans hs) _) (Real.exp_nonneg _)
  have h := (gaussian_map_contract a s hs) E
  rw [Measure.map_apply hm hE, Measure.smul_apply, smul_eq_mul] at h
  apply ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top (measure_ne_top _ _)) at h
  simpa [measureReal_def, ENNReal.toReal_mul, ENNReal.toReal_ofReal hp] using h

end Cubature
