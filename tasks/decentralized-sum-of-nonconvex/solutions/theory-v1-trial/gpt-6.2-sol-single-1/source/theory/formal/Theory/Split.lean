import Theory.Chain

open scoped RealInnerProductSpace

set_option maxHeartbeats 1000000

noncomputable section

namespace ULB63
namespace SplitChain

open Chain

variable {d : ℕ}

def selected (p : ℕ) (j : Fin d) : Prop := j.val ≠ 0 ∧ j.val % 2 = p

instance selectedDec (p : ℕ) (j : Fin d) : Decidable (selected p j) := by
  unfold selected
  infer_instance

def edge (d p : ℕ) : E d →L[ℝ] E d :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 fun j =>
        if selected p j then x j - x (prev j) else 0
      map_add' := fun x y => by
        ext j
        by_cases h : selected p j <;> simp [h]
        all_goals ring
      map_smul' := fun a x => by
        ext j
        by_cases h : selected p j <;> simp [h]
        all_goals ring }

@[simp] theorem edge_apply (p : ℕ) (x : E d) (j : Fin d) :
    edge d p x j = if selected p j then x j-x (prev j) else 0 := rfl

theorem edge_norm (p : ℕ) (x : E d) : ‖edge d p x‖ ≤ 2*‖x‖ := by
  have hb : ‖edge d p x‖ ≤ ‖x-lag d x‖ := by
    apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
    apply Finset.sum_le_sum
    intro j _
    by_cases h : selected p j
    · simp [h, h.1]
    · simp [h, sq_nonneg]
  have hs := (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1 (lag_sq x)
  linarith [hb.trans (norm_sub_le x (lag d x))]

def rawQuad (G : E d →L[ℝ] E d) (a : ℝ) (v x : E d) : ℝ :=
  a/2*‖G x‖^2 + ⟪v,x⟫

private theorem rawQuad_deriv (G : E d →L[ℝ] E d) (a : ℝ) (v x : E d) :
    HasFDerivAt (rawQuad G a v)
      ((a/2) • (2:ℝ) • (innerSL ℝ (G x)).comp G + innerSL ℝ v) x := by
  have H := ((G.hasFDerivAt (x := x)).norm_sq.const_mul (a/2)).fun_add (innerSL ℝ v).hasFDerivAt
  unfold rawQuad
  simpa only [innerSL_apply_apply, ← Nat.cast_smul_eq_nsmul ℝ, Nat.cast_ofNat] using H

theorem rawQuad_grad (G : E d →L[ℝ] E d) (a : ℝ) (v x : E d) :
    gradient (rawQuad G a v) x = a • G.adjoint (G x) + v := by
  apply ext_inner_right ℝ
  intro y
  rw [inner_gradient_left, (rawQuad_deriv G a v x).fderiv, inner_add_left, real_inner_smul_left,
    ContinuousLinearMap.adjoint_inner_left]
  change (a/2) * (2 * ⟪G x,G y⟫) + ⟪v,y⟫ = a * ⟪G x,G y⟫ + ⟪v,y⟫
  ring

theorem rawQuad_smooth (G : E d →L[ℝ] E d) (hG : ‖G‖ ≤ 2) {a L : ℝ}
    (ha : 0 ≤ a) (hL : 4*a ≤ L) (v : E d) : LSmooth L (rawQuad G a v) := by
  refine ⟨fun x => (rawQuad_deriv G a v x).differentiableAt, fun x y => ?_⟩
  rw [rawQuad_grad, rawQuad_grad, add_sub_add_right_eq_sub, ← smul_sub, ← map_sub, ← map_sub,
    norm_smul, Real.norm_eq_abs, abs_of_nonneg ha]
  have hGa : ‖G.adjoint‖ ≤ 2 := by rw [ContinuousLinearMap.adjoint.norm_map]; exact hG
  calc
    _ ≤ a*(‖G.adjoint‖*‖G (x-y)‖) :=
      mul_le_mul_of_nonneg_left (G.adjoint.le_opNorm _) ha
    _ ≤ a*(2*(2*‖x-y‖)) :=
      mul_le_mul_of_nonneg_left (mul_le_mul hGa ((G.le_opNorm _).trans
        (mul_le_mul_of_nonneg_right hG (norm_nonneg _))) (norm_nonneg _) (by norm_num)) ha
    _ = (4*a)*‖x-y‖ := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_right hL (norm_nonneg _)

theorem rawQuad_convex (G : E d →L[ℝ] E d) {a : ℝ} (ha : 0 ≤ a) (v : E d) :
    ConvexOn ℝ Set.univ (rawQuad G a v) := by
  refine ⟨convex_univ, fun x _ y _ t u ht hu hsum => ?_⟩
  obtain rfl : u = 1-t := by linarith
  have ht₁ : t ≤ 1 := by linarith
  have H := normSq_bary t ht ht₁ (G x) (G y)
  have hnon : 0 ≤ (a/2) * (t*(1-t)*‖G x-G y‖^2) := by positivity
  have h := congrArg (fun z : ℝ => (a/2)*z) H
  simp only [rawQuad, map_add, map_smul, inner_add_right, real_inner_smul_right, smul_eq_mul]
  linarith

def diagValue (a : Fin d → ℝ) (x : E d) : ℝ := ∑ k, a k / 2 * (x k)^2

def diagProx (a : Fin d → ℝ) (v : E d) (η : ℝ) : E d :=
  WithLp.toLp 2 (fun k => (1+η*a k)⁻¹ * v k)

theorem diag_convex (a : Fin d → ℝ) (ha : ∀ k, 0 ≤ a k) :
    StronglyConvexE 0 (fun x => (diagValue a x : EReal)) := by
  apply (stronglyConvexE_coe_iff _ _).2
  apply strongConvexOn_zero.mpr
  refine ⟨convex_univ, fun x _ y _ t u ht hu hh => ?_⟩
  obtain rfl : u = 1-t := by linarith
  have ht₁ : 0 ≤ 1-t := by linarith
  simp only [diagValue, smul_eq_mul]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro k _
  simp only [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  have hnon := mul_nonneg (mul_nonneg ht ht₁) (sq_nonneg (x k-y k))
  have H : (t*x k + (1-t)*y k)^2 ≤ t*(x k)^2+(1-t)*(y k)^2 := by nlinarith
  have HH := mul_le_mul_of_nonneg_left H (show 0 ≤ a k/2 by have := ha k; positivity)
  linarith

theorem diag_closed (a : Fin d → ℝ) : LowerSemicontinuous (fun x => (diagValue a x : EReal)) := by
  apply Continuous.lowerSemicontinuous
  apply continuous_coe_real_ereal.comp
  unfold diagValue
  apply continuous_finsetSum
  intro k _
  fun_prop

private theorem prox_scalar (c v η u : ℝ) (hc : 0 ≤ c) (hη : 0 < η) :
    let p := (1+η*c)⁻¹*v
    c/2*p^2+(p-v)^2/(2*η) ≤ c/2*u^2+(u-v)^2/(2*η) := by
  intro p
  have hp : (1+η*c)*p = v := by
    dsimp [p]
    field_simp
  have hcancel : c*p+(p-v)/η = 0 := by
    have he : p-v = -(η*(c*p)) := by linarith
    rw [he, neg_div, mul_div_cancel_left₀ _ hη.ne']
    ring
  have H : c/2*u^2+(u-v)^2/(2*η) - (c/2*p^2+(p-v)^2/(2*η)) =
      (c/2+1/(2*η))*(u-p)^2 + (c*p+(p-v)/η)*(u-p) := by ring
  rw [← sub_nonneg, H, hcancel, zero_mul, add_zero]
  positivity

theorem diag_exact (a : Fin d → ℝ) (ha : ∀ k, 0 ≤ a k) (v : E d)
    {η : ℝ} (hη : 0 < η) :
    IsProx (fun x => (diagValue a x : EReal)) η v (diagProx a v η) := by
  intro u
  rw [← EReal.coe_add, ← EReal.coe_add, EReal.coe_le_coe_iff]
  simp only [diagValue, EuclideanSpace.real_norm_sq_eq, PiLp.sub_apply]
  rw [Finset.sum_div, Finset.sum_div, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum (fun k _ => prox_scalar (a k) (v k) η (u k) (ha k) hη)

theorem diag_support (a : Fin d → ℝ) (v : E d) (η : ℝ) : supp (diagProx a v η) ⊆ supp v := by
  intro k hk
  change (1+η*a k)⁻¹ * v k ≠ 0 at hk
  exact fun h => hk (by rw [show v k = 0 from h, mul_zero])

end SplitChain
end ULB63
