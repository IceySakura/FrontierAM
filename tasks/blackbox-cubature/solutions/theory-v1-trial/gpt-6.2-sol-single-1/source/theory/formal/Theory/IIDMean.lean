import Theory.Slab

/-! Statistics of a mean of bounded iid residuals plus a known baseline. -/

open MeasureTheory ProbabilityTheory Set

namespace Cubature

noncomputable section

theorem shifted_iid_mean {X : Type*} [MeasurableSpace X] (μ : Measure X) [IsProbabilityMeasure μ]
    {Z : X → ℝ} (hz : Measurable Z) {D : ℝ} (hb : ∀ x, |Z x| ≤ D)
    {n : ℕ} (hn : 0 < n) (C : ℝ) :
    let law := Measure.pi (fun _ : Fin n => μ)
    let F := fun s : Fin n → X => C + (n : ℝ)⁻¹ * ∑ i, Z (s i)
    let target := C + ∫ x, Z x ∂μ
    Measurable F ∧ (∫ s, F s ∂law) = target ∧
      (∫ s, (F s - target)^2 ∂law) ≤ D^2 / (n : ℝ) := by
  let law := Measure.pi (fun _ : Fin n => μ)
  let S : (Fin n → X) → ℝ := ∑ i : Fin n, fun v => Z (v i)
  have hsFun (v : Fin n → X) : S v = ∑ i, Z (v i) := by simp [S, Finset.sum_apply]
  have hrange : ∀ᵐ x ∂μ, Z x ∈ Icc (-D) D := Filter.Eventually.of_forall
    (fun x => abs_le.mp (hb x))
  have hLp := memLp_of_bounded hrange hz.aestronglyMeasurable 2
  have hZint : Integrable Z μ := hLp.integrable (by norm_num)
  have hS : Measurable S := by
    simp_rw [funext hsFun]
    exact Finset.measurable_sum _ (fun i _ => hz.comp (measurable_pi_apply i))
  have hzi (i : Fin n) : Integrable (fun s : Fin n → X => Z (s i)) law :=
    integrable_comp_eval (μ := fun _ : Fin n => μ) (i := i) hZint
  have hSint : Integrable S law := by
    rw [funext hsFun]
    exact integrable_finsetSum _ (fun i _ => hzi i)
  have hInt : ∫ s, S s ∂law = (n : ℝ) * ∫ x, Z x ∂μ := by
    simp_rw [hsFun]
    rw [integral_finsetSum _ (fun i _ => hzi i)]
    have he (i : Fin n) : (∫ s : Fin n → X, Z (s i) ∂law) = ∫ x, Z x ∂μ :=
      integral_comp_eval (μ := fun _ : Fin n => μ) (i := i) hz.aestronglyMeasurable
    simp [he]
  have hF : Measurable fun s => C + (n : ℝ)⁻¹ * S s := by fun_prop
  have hne : (n:ℝ) ≠ 0 := by positivity
  have hMean : (∫ s, (C + (n : ℝ)⁻¹ * S s) ∂law) = C + ∫ x, Z x ∂μ := by
    rw [integral_add (integrable_const _) (hSint.const_mul _), integral_const,
      integral_const_mul, hInt, ← mul_assoc, inv_mul_cancel₀ hne, one_mul]
    simp [law]
  have hVar : variance Z μ ≤ D^2 := by
    have ha := variance_le_sq_of_bounded hrange hz.aemeasurable
    rwa [show (D - -D)/2 = D by ring] at ha
  dsimp only
  simp_rw [← hsFun]
  refine ⟨hF,hMean,?_⟩
  rw [← hMean, ← variance_eq_integral hF.aemeasurable,
    variance_const_add (hS.const_mul _).aestronglyMeasurable, variance_const_mul]
  change (n : ℝ)⁻¹ ^ 2 * variance S (Measure.pi fun _ => μ) ≤ _
  rw [variance_sum_pi (X := fun _ => Z) (fun _ => hLp)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc
    _ ≤ (n : ℝ)⁻¹ ^ 2 * ((n : ℝ) * D^2) := by gcongr
    _ = _ := by field_simp

end

end Cubature
