import Theory.Recovery

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

theorem gaussian_map_neg (d : ℕ) :
    (gaussian d).map (fun x : Point d ↦ -x) = gaussian d := by
  rw [gaussian_eq_pi]
  let μ : Fin d → Measure ℝ := fun _ ↦ ProbabilityTheory.gaussianReal 0 1
  have hm (i : Fin d) : (μ i).map (fun x : ℝ ↦ -x) = μ i := by
    simp [μ, ProbabilityTheory.gaussianReal_map_neg]
  letI (i : Fin d) : SigmaFinite ((μ i).map (fun x : ℝ ↦ -x)) := by
    rw [hm i]
    infer_instance
  have H := Measure.pi_map_pi (μ := μ) (f := fun _ x ↦ -x)
    (fun _ ↦ (measurable_neg : Measurable (fun x : ℝ ↦ -x)).aemeasurable)
  change (Measure.pi μ).map (fun (x : Point d) i ↦ -x i) = Measure.pi μ
  simpa only [hm] using H

theorem integral_augmentedPayoff {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (h : Point d → ℝ)
    (ho : ∀ x, h (-x) = -h x) (hi : Integrable h (gaussian d)) :
    (∫ x, augmentedPayoff ρ E h x ∂(gaussian d)) = probability (region ρ E) := by
  classical
  have hm := integral_map (μ := gaussian d)
    ((measurable_neg : Measurable (fun x : Point d ↦ -x)).aemeasurable)
    (show AEStronglyMeasurable h ((gaussian d).map (fun x : Point d ↦ -x)) by
      rw [gaussian_map_neg]
      exact hi.aestronglyMeasurable)
  rw [gaussian_map_neg] at hm
  simp_rw [ho, integral_neg] at hm
  have hz : ∫ x, h x ∂(gaussian d) = 0 := by linarith
  have hf : payoff ρ E = (region ρ E).indicator (fun _ ↦ (1 : ℝ)) := by
    funext x
    by_cases hx : x ∈ region ρ E <;> simp [payoff, answerBit, hx]
  have hj : Integrable (payoff ρ E) (gaussian d) :=
    hf ▸ (integrable_const _).indicator (measurable_region _ _)
  change (∫ x, h x + payoff ρ E x ∂(gaussian d)) = probability (region ρ E)
  rw [integral_add hi hj, hz, zero_add, ← probability_eq_payoff_integral]

theorem pairedBit_iff {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (h : Point d → ℝ)
    (ho : ∀ x, h (-x) = -h x) (tag : Bool) (x : Point d) :
    pairedBit (augmentedPayoff ρ E h) (tag,x) = true ↔
      if tag then x ∈ region ρ E ∧ -x ∈ region ρ E
      else x ∈ region ρ E ∨ -x ∈ region ρ E := by
  classical
  have hf : augmentedPayoff ρ E h x + augmentedPayoff ρ E h (-x) =
      payoff ρ E x + payoff ρ E (-x) := by simp only [augmentedPayoff,ho]; ring
  have hb : pairedBit (augmentedPayoff ρ E h) (tag,x) = true ↔
      (if tag then (3/2 : ℝ) else 1/2) <
        augmentedPayoff ρ E h x + augmentedPayoff ρ E h (-x) := by
    unfold pairedBit
    exact decide_eq_true_iff
  rw [hb,hf]
  cases tag <;> by_cases hx : x ∈ region ρ E <;> by_cases hn : -x ∈ region ρ E <;>
    norm_num [payoff,answerBit,hx,hn]

namespace QueryTree

def mapPoints {X X' Y : Type*} (g : X → X') : QueryTree X Y → QueryTree X' Y
  | .ret y => .ret y
  | .ask x l r => .ask (g x) (mapPoints g l) (mapPoints g r)

theorem run_mapPoints {X X' Y : Type*} (t : QueryTree X Y) (g : X → X') (f : X' → Bool) :
    (t.mapPoints g).run f = t.run (fun x ↦ f (g x)) := by
  induction t with
  | ret y => rfl
  | ask x l r hl hr => simp [mapPoints,run,hl,hr]

theorem depth_mapPoints {X X' Y : Type*} (t : QueryTree X Y) (g : X → X') :
    (t.mapPoints g).depth = t.depth := by
  induction t with
  | ret y => rfl
  | ask x l r hl hr => simp [mapPoints,depth,hl,hr]

theorem admissible_mapPoints {X X' Y : Type*} (t : QueryTree X Y) (g : X → X')
    (P : X → Prop) (P' : X' → Prop) (h : ∀ x, P x → P' (g x))
    (ht : t.admissible P) : (t.mapPoints g).admissible P' := by
  induction t with
  | ret y => trivial
  | ask x l r hl hr => exact ⟨h _ ht.1, hl ht.2.1, hr ht.2.2⟩

end QueryTree

theorem quadratic_neg {d : ℕ} (A : Coefficients d) (x : Point d) :
    quadratic A (-x) = quadratic A x := by
  simpa [neg_smul] using quadratic_smul A (-1) x

theorem linear_neg {d : ℕ} (b x : Point d) : linear b (-x) = -linear b x := by
  simpa [neg_smul] using linear_smul b (-1) x

theorem linear_neg_coeffs {d : ℕ} (b x : Point d) : linear (-b) x = -linear b x := by
  simp [linear, Finset.sum_neg_distrib]

def reflected {d : ℕ} (E : Ellipsoid d) : Ellipsoid d where
  A := E.A
  b := -E.b
  symm := E.symm
  low := E.low
  high := E.high
  bias := fun x ↦ by rw [linear_neg_coeffs,abs_neg]; exact E.bias x

theorem reflected_probability {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) :
    probability (region ρ (reflected E)) = probability (region ρ E) := by
  have hre : region ρ (reflected E) = (fun x : Point d ↦ -x) ⁻¹' region ρ E := by
    ext x
    simp [region,polynomial,reflected,smul_neg,quadratic_neg,linear_neg,linear_neg_coeffs]
  unfold probability Measure.real
  rw [hre, ← Measure.map_apply measurable_neg (measurable_region ρ E), gaussian_map_neg]

end Cubature
