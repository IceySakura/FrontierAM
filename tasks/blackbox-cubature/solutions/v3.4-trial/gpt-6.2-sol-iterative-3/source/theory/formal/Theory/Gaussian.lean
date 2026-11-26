import Theory.Model

open scoped BigOperators ENNReal
open Set MeasureTheory

namespace Cubature

variable {d : ℕ}

theorem measurable_gaussianDensity : Measurable (gaussianDensity (d := d)) := by
  unfold gaussianDensity
  fun_prop

theorem gaussianDensity_pos (x : Coordinates d) : 0 < gaussianDensity x := by
  unfold gaussianDensity
  rw [ENNReal.ofReal_pos]
  have h : 0 < Real.sqrt (2 * Real.pi) := Real.sqrt_pos.2 (by positivity)
  positivity

theorem contraction_density (a x : Coordinates d) {t : ℝ} (ht : 0 < t) (ht₁ : t ≤ 1) :
    ENNReal.ofReal (Real.exp (-((1-t)/(2*(1+t))*∑ i, (a i)^2))) *
      gaussianDensity x ≤ gaussianDensity (contract a t x) := by
  let c := (Real.sqrt (2 * Real.pi))⁻¹ ^ d
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have hden : 0 < 1+t := by linarith
  have hb : 0 ≤ (1-t)*(1+t) := mul_nonneg (by linarith) hden.le
  have pointwise (i : Fin d) :
      (contract a t x i)^2 - (x i)^2 ≤ ((1-t)/(1+t))*(a i)^2 := by
    have hid : (contract a t x i)^2 - (x i)^2 =
        ((1-t)/(1+t))*(a i)^2 - (1-t)*(1+t)*(x i-t/(1+t)*a i)^2 := by
      simp only [contract, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      field_simp
      ring
    rw [hid]
    exact sub_le_self _ (mul_nonneg hb (sq_nonneg _))
  have h := Finset.sum_le_sum (s := Finset.univ) fun i _ => pointwise i
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum] at h
  have hexp : -((1-t)/(2*(1+t))*∑ i, (a i)^2) - (∑ i, (x i)^2)/2 ≤
      -(∑ i, (contract a t x i)^2)/2 := by
    have aux : (1-t)/(2*(1+t)) = ((1-t)/(1+t))/2 := by
      field_simp
    rw [aux]
    linarith
  change ENNReal.ofReal (Real.exp _) *
    ENNReal.ofReal (c*Real.exp (-(∑ i, (x i)^2)/2)) ≤
    ENNReal.ofReal (c*Real.exp (-(∑ i, (contract a t x i)^2)/2))
  rw [← ENNReal.ofReal_mul (Real.exp_pos _).le]
  apply ENNReal.ofReal_le_ofReal
  rw [show ∀ u v : ℝ, Real.exp u * (c*Real.exp v) = c*Real.exp (u+v) by
    intro u v; rw [Real.exp_add]; ring]
  exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (by linarith)) hc

theorem contract_mass (a : Coordinates d) {t : ℝ} (ht : 0 < t)
    {S : Set (Coordinates d)} (hS : MeasurableSet S) :
    gaussian (contract a t '' S) =
      ENNReal.ofReal (t^d) * ∫⁻ x in S, gaussianDensity (contract a t x) := by
  let F := contract a t
  let G : Coordinates d → Coordinates d := fun x => t⁻¹ • (x-(1-t) • a)
  have hLeft : Function.LeftInverse G F := by
    intro x
    simp [F, G, contract, smul_smul, inv_mul_cancel₀ ht.ne']
  have hRight : Function.RightInverse G F := by
    intro x
    simp [F, G, contract, smul_smul, mul_inv_cancel₀ ht.ne']
  have hpre : F ⁻¹' (F '' S) = S := preimage_image_eq _ hLeft.injective
  have hF : Measurable F := by
    change Measurable (fun z : Coordinates d => t • z + (1-t) • a)
    fun_prop
  have hG : Measurable G := by dsimp [G]; fun_prop
  have hs : MeasurableSet (F '' S) := by
    have he : F '' S = G ⁻¹' S := by
      ext x
      constructor
      · rintro ⟨y,hy,rfl⟩
        change G (F y) ∈ S
        rwa [hLeft y]
      · intro hx
        exact ⟨G x,hx,hRight x⟩
    rw [he]
    exact hS.preimage hG
  have hmap : (volume : Measure (Coordinates d)).map F =
      ENNReal.ofReal ((t^d)⁻¹) • (volume : Measure (Coordinates d)) := by
    have hf : F = (fun x : Coordinates d => x+(1-t) • a) ∘ (fun x => t • x) := rfl
    have h₁ : Measurable (fun x : Coordinates d => x + (1-t) • a) := by fun_prop
    have h₂ : Measurable (fun x : Coordinates d => t • x) := by fun_prop
    rw [hf, ← Measure.map_map h₁ h₂,
      Measure.map_addHaar_smul _ ht.ne', Measure.map_smul _ h₁.aemeasurable, map_add_right_eq_self]
    simp [abs_of_nonneg ((inv_pos.mpr (pow_pos ht d)).le)]
  have h := setLIntegral_map (μ := (volume : Measure (Coordinates d)))
    hs measurable_gaussianDensity hF
  rw [hpre, hmap, Measure.restrict_smul, lintegral_smul_measure] at h
  change gaussian (F '' S) = ENNReal.ofReal (t^d) * ∫⁻ x in S, gaussianDensity (F x)
  rw [gaussian, withDensity_apply _ hs, ← h, smul_eq_mul, ← mul_assoc, ← ENNReal.ofReal_mul (pow_nonneg ht.le d),
    mul_inv_cancel₀ (pow_ne_zero d ht.ne'), ENNReal.ofReal_one, one_mul]

theorem gaussian_contraction_lower (a : Coordinates d) {t : ℝ} (ht : 0 < t) (ht₁ : t ≤ 1)
    {S : Set (Coordinates d)} (hS : MeasurableSet S) :
    ENNReal.ofReal (massFactor a t) * gaussian S ≤ gaussian (contract a t '' S) := by
  rw [contract_mass a ht hS, massFactor, ENNReal.ofReal_mul (pow_nonneg ht.le d),
    mul_assoc, gaussian, withDensity_apply _ hS]
  have h :
      ENNReal.ofReal (Real.exp (-((1-t)/(2*(1+t))*∑ i, (a i)^2))) *
        (∫⁻ x in S, gaussianDensity x) ≤ ∫⁻ x in S, gaussianDensity (contract a t x) := by
    rw [← lintegral_const_mul _ measurable_gaussianDensity]
    exact lintegral_mono (contraction_density a · ht ht₁)
  gcongr

end Cubature
