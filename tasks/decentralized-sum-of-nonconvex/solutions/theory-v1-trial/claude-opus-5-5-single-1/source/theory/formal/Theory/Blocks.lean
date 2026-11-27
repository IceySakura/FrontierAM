import Theory.ChainInstance

/-!
# Block structure: `ℝ^{m b}` as `m` blocks of `ℝ^b`

Block projections `P ξ`, block embeddings `Em ξ` (the adjoint of `P ξ`), and composition of
quadratic chain functions with the block projections.
-/

open scoped RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Blocks

variable {m b : ℕ}

/-- The coordinate code `(ξ, j) ↦ p`. -/
abbrev code (m b : ℕ) : Fin m × Fin b ≃ Fin (m * b) := finProdFinEquiv

/-- Block projection. -/
def P (ξ : Fin m) (x : E (m * b)) : E b := WithLp.toLp 2 (fun j => x (code m b (ξ, j)))

@[simp] lemma P_apply (ξ : Fin m) (x : E (m * b)) (j : Fin b) : P ξ x j = x (code m b (ξ, j)) := rfl

/-- Block embedding. -/
def Em (ξ : Fin m) (y : E b) : E (m * b) :=
  WithLp.toLp 2 (fun p => if ((code m b).symm p).1 = ξ then y ((code m b).symm p).2 else 0)

@[simp] lemma Em_apply (ξ : Fin m) (y : E b) (p : Fin (m * b)) :
    Em ξ y p = if ((code m b).symm p).1 = ξ then y ((code m b).symm p).2 else 0 := rfl

/-- `P ξ` as a linear map. -/
def Plin (ξ : Fin m) : E (m * b) →ₗ[ℝ] E b where
  toFun := P ξ
  map_add' x y := by ext j; simp
  map_smul' c x := by ext j; simp

/-- `P ξ` as a continuous linear map. -/
def Pclm (ξ : Fin m) : E (m * b) →L[ℝ] E b := LinearMap.toContinuousLinearMap (Plin ξ)

lemma Pclm_apply (ξ : Fin m) (x : E (m * b)) : Pclm ξ x = P ξ x := rfl

lemma sum_code {M : Type*} [AddCommMonoid M] (g : Fin (m * b) → M) :
    ∑ p, g p = ∑ ξ : Fin m, ∑ j : Fin b, g (code m b (ξ, j)) := by
  rw [← (code m b).sum_comp, Fintype.sum_prod_type]

lemma inner_Em (ξ : Fin m) (y : E b) (x : E (m * b)) : ⟪Em ξ y, x⟫ = ⟪y, P ξ x⟫ := by
  simp only [PiLp.inner_apply, RCLike.inner_apply, conj_trivial]
  rw [sum_code]
  simp only [Em_apply, Equiv.symm_apply_apply, P_apply]
  rw [Finset.sum_eq_single ξ]
  · simp
  · intro ξ' _ hξ'
    simp [hξ']
  · intro h; exact absurd (Finset.mem_univ ξ) h

lemma norm_sq_eq_sum_P (x : E (m * b)) : ‖x‖ ^ 2 = ∑ ξ, ‖P ξ x‖ ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq, sum_code]
  refine Finset.sum_congr rfl fun ξ _ => ?_
  rw [EuclideanSpace.real_norm_sq_eq]
  rfl

lemma norm_P_le (ξ : Fin m) (x : E (m * b)) : ‖P ξ x‖ ≤ ‖x‖ := by
  have h : ‖P ξ x‖ ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [norm_sq_eq_sum_P x]
    exact Finset.single_le_sum (f := fun ξ => ‖P ξ x‖ ^ 2) (fun _ _ => sq_nonneg _)
      (Finset.mem_univ ξ)
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) (by norm_num)).1 h

lemma norm_Em (ξ : Fin m) (y : E b) : ‖Em ξ y‖ = ‖y‖ := by
  have h : ‖Em ξ y‖ ^ 2 = ‖y‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, sum_code, EuclideanSpace.real_norm_sq_eq]
    simp only [Em_apply, Equiv.symm_apply_apply]
    rw [Finset.sum_eq_single ξ]
    · simp
    · intro ξ' _ hξ'; simp [hξ']
    · intro h; exact absurd (Finset.mem_univ ξ) h
  exact (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).1 h

lemma P_sub (ξ : Fin m) (x y : E (m * b)) : P ξ (x - y) = P ξ x - P ξ y := by ext j; simp

lemma Em_sub (ξ : Fin m) (y z : E b) : Em ξ y - Em ξ z = Em ξ (y - z) := by
  ext p; simp only [PiLp.sub_apply, Em_apply]; split_ifs <;> simp

lemma P_zero (ξ : Fin m) : P ξ (0 : E (m * b)) = 0 := by ext j; simp

/-- Gradient of a function composed with a block projection. -/
lemma hasGradientAt_comp_P (g : E b → ℝ) (g' : E b) (ξ : Fin m) (x : E (m * b))
    (hg : HasGradientAt g g' (P ξ x)) : HasGradientAt (fun z => g (P ξ z)) (Em ξ g') x := by
  rw [hasGradientAt_iff_hasFDerivAt] at hg ⊢
  have hP : HasFDerivAt (fun z => P ξ z) (Pclm ξ) x := (Pclm ξ).hasFDerivAt
  have := hg.comp x hP
  refine this.congr_fderiv ?_
  ext v
  simp [InnerProductSpace.toDual_apply_apply, inner_Em, Pclm_apply]

/-- A chain quadratic on block `ξ`. -/
def bq (L : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (ξ : Fin m) (x : E (m * b)) : ℝ :=
  Chain.quad (Chain.u b) (L / 4) S w c (P ξ x)

lemma gradient_bq (L : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (ξ : Fin m) (x : E (m * b)) :
    gradient (bq L S w c ξ) x = Em ξ (Chain.quadGrad (Chain.u b) (L / 4) S w c (P ξ x)) :=
  (hasGradientAt_comp_P _ _ ξ x (Chain.hasGradientAt_quad _ _ _ _ _ _)).gradient

lemma lsmooth_bq {L : ℝ} (hL : 0 ≤ L) (S : Finset ℕ) (hS : S ⊆ range (b + 1)) (w c : ℕ → ℝ)
    (hw : ∀ k ∈ S, 0 ≤ w k ∧ w k ≤ 1) (ξ : Fin m) :
    LSmooth (d := m * b) L (bq (m := m) (b := b) L S w c ξ) := by
  have hg := Chain.lsmooth_quad (Chain.u b) (by positivity : 0 ≤ L / 4) S w c hw (by norm_num)
    (Chain.sum_inner_u_sq_le (d := b) S hS) (show L / 4 * 4 ≤ L by linarith)
  refine ⟨fun x => (hasGradientAt_comp_P _ _ ξ x (Chain.hasGradientAt_quad _ _ _ _ _ _)).differentiableAt,
    fun x y => ?_⟩
  rw [gradient_bq, gradient_bq, Em_sub, norm_Em]
  have h1 := hg.2 (P ξ x) (P ξ y)
  rw [Chain.gradient_quad, Chain.gradient_quad] at h1
  calc _ ≤ L * ‖P ξ x - P ξ y‖ := h1
    _ = L * ‖P ξ (x - y)‖ := by rw [P_sub]
    _ ≤ L * ‖x - y‖ := mul_le_mul_of_nonneg_left (norm_P_le ξ _) hL

lemma bq_convex {L : ℝ} (hL : 0 ≤ L) (S : Finset ℕ) (w c : ℕ → ℝ) (hw : ∀ k, 0 ≤ w k)
    (ξ : Fin m) : ConvexOn ℝ Set.univ (bq (m := m) (b := b) L S w c ξ) := by
  have h := Chain.quad_convex (Chain.u b) (by positivity : 0 ≤ L / 4) S w c hw
  have := h.comp_linearMap (Plin ξ)
  rw [Set.preimage_univ] at this
  exact this

end Blocks
end ULB63
