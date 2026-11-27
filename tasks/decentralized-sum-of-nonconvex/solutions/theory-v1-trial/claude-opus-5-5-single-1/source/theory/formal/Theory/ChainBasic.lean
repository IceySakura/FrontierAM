import Theory.Model

/-!
# Quadratic building blocks on `ℝ^d`

Coordinates indexed by `ℕ` (zero beyond `d`), coordinate vectors, the chain link vectors
`u k = e_{k-1} - e_k`, and quadratic functions `(β/2) ∑_{k∈S} w_k (⟪U k, x⟫ + c_k)^2` with their
gradients, smoothness constants and support properties.
-/

open scoped RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {d : ℕ}

/-- Coordinate `k` of `x`, and `0` when `k ≥ d`. -/
def cx (x : E d) (k : ℕ) : ℝ := if h : k < d then x ⟨k, h⟩ else 0

/-- The coordinate vector `e_k`, and `0` when `k ≥ d`. -/
def ev (d : ℕ) (k : ℕ) : E d := if h : k < d then EuclideanSpace.single ⟨k, h⟩ 1 else 0

lemma cx_of_ge {x : E d} {k : ℕ} (hk : d ≤ k) : cx x k = 0 := by
  simp [cx, show ¬ k < d by omega]

lemma cx_fin (x : E d) (p : Fin d) : cx x p = x p := by
  simp [cx, p.isLt]

lemma inner_ev (k : ℕ) (x : E d) : ⟪ev d k, x⟫ = cx x k := by
  unfold ev cx
  split_ifs with h
  · simp [EuclideanSpace.inner_single_left]
  · simp

lemma ev_apply (k : ℕ) (p : Fin d) : (ev d k) p = if (p : ℕ) = k then 1 else 0 := by
  unfold ev
  split_ifs with h h' h'
  · subst h'; simp
  · simp only [PiLp.single_apply]
    rw [ite_eq_right_iff.2]
    intro hp; exact absurd (by rw [hp]) h'
  · omega
  · simp

lemma cx_add (x y : E d) (k : ℕ) : cx (x + y) k = cx x k + cx y k := by
  unfold cx; split_ifs <;> simp

lemma cx_sub (x y : E d) (k : ℕ) : cx (x - y) k = cx x k - cx y k := by
  unfold cx; split_ifs <;> simp

lemma cx_smul (a : ℝ) (x : E d) (k : ℕ) : cx (a • x) k = a * cx x k := by
  unfold cx; split_ifs <;> simp

lemma cx_zero (k : ℕ) : cx (0 : E d) k = 0 := by
  unfold cx; split_ifs <;> simp

lemma inner_eq_sum (x y : E d) : ⟪x, y⟫ = ∑ k ∈ range d, cx x k * cx y k := by
  rw [← Fin.sum_univ_eq_sum_range (fun k => cx x k * cx y k) d]
  simp only [cx_fin, PiLp.inner_apply, RCLike.inner_apply, conj_trivial]
  exact Finset.sum_congr rfl fun p _ => by ring

lemma norm_sq_eq_sum (x : E d) : ‖x‖ ^ 2 = ∑ k ∈ range d, cx x k ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_eq_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

lemma sq_cx_le_norm_sq (x : E d) (k : ℕ) : cx x k ^ 2 ≤ ‖x‖ ^ 2 := by
  by_cases hk : k < d
  · rw [norm_sq_eq_sum]
    exact Finset.single_le_sum (f := fun j => cx x j ^ 2) (fun j _ => sq_nonneg _)
      (Finset.mem_range.2 hk)
  · rw [cx_of_ge (by omega)]; simp

/-- The chain link vector `u_k = e_{k-1} - e_k` (`u_0 = -e_0`); `u_d = e_{d-1}`. -/
def u (d : ℕ) (k : ℕ) : E d := (if k = 0 then 0 else ev d (k - 1)) - ev d k

lemma inner_u (k : ℕ) (x : E d) :
    ⟪u d k, x⟫ = (if k = 0 then 0 else cx x (k - 1)) - cx x k := by
  unfold u
  rw [inner_sub_left]
  split_ifs <;> simp [inner_ev]

lemma u_apply (k : ℕ) (p : Fin d) :
    (u d k) p = (if k = 0 then 0 else (if (p : ℕ) = k - 1 then 1 else 0)) -
      (if (p : ℕ) = k then 1 else 0) := by
  unfold u
  rw [PiLp.sub_apply]
  by_cases h0 : k = 0
  · simp only [h0, if_true, PiLp.zero_apply, ev_apply]
  · simp only [h0, if_false, ev_apply]

/-! ## Quadratic functions and their gradients -/

/-- `(β/2) ∑_{k ∈ S} w_k (⟪U k, x⟫ + c_k)^2`. -/
def quad (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x : E d) : ℝ :=
  β / 2 * ∑ k ∈ S, w k * (⟪U k, x⟫ + c k) ^ 2

/-- The gradient `β ∑_{k ∈ S} w_k (⟪U k, x⟫ + c_k) U k`. -/
def quadGrad (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x : E d) : E d :=
  β • ∑ k ∈ S, (w k * (⟪U k, x⟫ + c k)) • U k

lemma hasGradientAt_quad (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x : E d) :
    HasGradientAt (quad U β S w c) (quadGrad U β S w c x) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have hk : ∀ k ∈ S, HasFDerivAt (fun y : E d => w k * (⟪U k, y⟫ + c k) ^ 2)
      (w k • ((2 : ℕ) • (⟪U k, x⟫ + c k) ^ (2 - 1)) • innerSL ℝ (U k)) x := by
    intro k _
    have h1 : HasFDerivAt (fun y : E d => ⟪U k, y⟫ + c k) (innerSL ℝ (U k)) x :=
      ((innerSL ℝ (U k)).hasFDerivAt).add_const (c k)
    exact (h1.pow 2).const_mul (w k)
  have hsum := HasFDerivAt.fun_sum hk
  have hall := hsum.const_mul (β / 2)
  refine hall.congr_fderiv ?_
  ext v
  simp only [ContinuousLinearMap.smul_apply, ContinuousLinearMap.coe_sum', Finset.sum_apply,
    innerSL_apply_apply, InnerProductSpace.toDual_apply_apply, quadGrad, inner_smul_left,
    sum_inner, smul_eq_mul, nsmul_eq_mul, conj_trivial, ContinuousLinearMap.coe_smul',
    Pi.smul_apply]
  rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  push_cast
  ring

lemma gradient_quad (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x : E d) :
    gradient (quad U β S w c) x = quadGrad U β S w c x :=
  (hasGradientAt_quad U β S w c x).gradient

lemma differentiable_quad (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) :
    Differentiable ℝ (quad U β S w c) :=
  fun x => (hasGradientAt_quad U β S w c x).differentiableAt

lemma quadGrad_sub (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x y : E d) :
    quadGrad U β S w c x - quadGrad U β S w c y = β • ∑ k ∈ S, (w k * ⟪U k, x - y⟫) • U k := by
  unfold quadGrad
  rw [← smul_sub, ← Finset.sum_sub_distrib]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [← sub_smul, inner_sub_right]
  congr 1; ring

/-- Norm bound for `∑ w_k ⟪U k, v⟫ U k` when `∑ ⟪U k, z⟫² ≤ K ‖z‖²` and `0 ≤ w ≤ 1`. -/
lemma norm_sum_le (U : ℕ → E d) (S : Finset ℕ) (w : ℕ → ℝ) (hw : ∀ k ∈ S, 0 ≤ w k ∧ w k ≤ 1)
    (K : ℝ) (hK0 : 0 ≤ K) (hK : ∀ z : E d, ∑ k ∈ S, ⟪U k, z⟫ ^ 2 ≤ K * ‖z‖ ^ 2) (v : E d) :
    ‖∑ k ∈ S, (w k * ⟪U k, v⟫) • U k‖ ≤ K * ‖v‖ := by
  set z := ∑ k ∈ S, (w k * ⟪U k, v⟫) • U k with hz
  have h1 : ‖z‖ ^ 2 = ∑ k ∈ S, (w k * ⟪U k, v⟫) * ⟪U k, z⟫ := by
    rw [← real_inner_self_eq_norm_sq]
    have : ⟪z, z⟫ = ⟪∑ k ∈ S, (w k * ⟪U k, v⟫) • U k, z⟫ := by rw [← hz]
    rw [this, sum_inner]
    simp [inner_smul_left]
  have h2 : (∑ k ∈ S, (w k * ⟪U k, v⟫) * ⟪U k, z⟫) ^ 2 ≤
      (∑ k ∈ S, (w k * ⟪U k, v⟫) ^ 2) * ∑ k ∈ S, ⟪U k, z⟫ ^ 2 :=
    Finset.sum_mul_sq_le_sq_mul_sq S _ _
  have h3 : ∑ k ∈ S, (w k * ⟪U k, v⟫) ^ 2 ≤ ∑ k ∈ S, ⟪U k, v⟫ ^ 2 := by
    refine Finset.sum_le_sum fun k hk => ?_
    obtain ⟨hw0, hw1⟩ := hw k hk
    rw [mul_pow]
    have : w k ^ 2 ≤ 1 := by nlinarith
    nlinarith [sq_nonneg ⟪U k, v⟫]
  have h4 := hK v
  have h5 := hK z
  have hA : 0 ≤ ∑ k ∈ S, (w k * ⟪U k, v⟫) ^ 2 := Finset.sum_nonneg fun k _ => sq_nonneg _
  have hB : 0 ≤ ∑ k ∈ S, ⟪U k, z⟫ ^ 2 := Finset.sum_nonneg fun k _ => sq_nonneg _
  have key : (‖z‖ ^ 2) ^ 2 ≤ (K * ‖v‖ ^ 2) * (K * ‖z‖ ^ 2) := by
    calc (‖z‖ ^ 2) ^ 2 = (∑ k ∈ S, (w k * ⟪U k, v⟫) * ⟪U k, z⟫) ^ 2 := by rw [← h1]
      _ ≤ _ := h2
      _ ≤ (∑ k ∈ S, ⟪U k, v⟫ ^ 2) * ∑ k ∈ S, ⟪U k, z⟫ ^ 2 :=
          mul_le_mul_of_nonneg_right h3 hB
      _ ≤ (K * ‖v‖ ^ 2) * (K * ‖z‖ ^ 2) :=
          mul_le_mul h4 h5 hB (by positivity)
  by_contra hcon
  push_neg at hcon
  have hz0 : 0 < ‖z‖ := lt_of_le_of_lt (by positivity) hcon
  have hKv : 0 ≤ K * ‖v‖ := by positivity
  have : (K * ‖v‖) ^ 2 < ‖z‖ ^ 2 := by nlinarith
  have hzz : 0 < ‖z‖ ^ 2 := by positivity
  nlinarith

lemma lsmooth_quad (U : ℕ → E d) {β : ℝ} (hβ : 0 ≤ β) (S : Finset ℕ) (w c : ℕ → ℝ)
    (hw : ∀ k ∈ S, 0 ≤ w k ∧ w k ≤ 1) {K : ℝ} (hK0 : 0 ≤ K)
    (hK : ∀ z : E d, ∑ k ∈ S, ⟪U k, z⟫ ^ 2 ≤ K * ‖z‖ ^ 2) {L : ℝ} (hL : β * K ≤ L) :
    LSmooth L (quad U β S w c) := by
  refine ⟨differentiable_quad U β S w c, fun x y => ?_⟩
  rw [gradient_quad, gradient_quad, quadGrad_sub, norm_smul, Real.norm_of_nonneg hβ]
  calc β * ‖∑ k ∈ S, (w k * ⟪U k, x - y⟫) • U k‖ ≤ β * (K * ‖x - y‖) :=
        mul_le_mul_of_nonneg_left (norm_sum_le U S w hw K hK0 hK _) hβ
    _ = (β * K) * ‖x - y‖ := by ring
    _ ≤ L * ‖x - y‖ := mul_le_mul_of_nonneg_right hL (norm_nonneg _)

/-- `∑_{k ≤ d} ⟪u_k, z⟫² ≤ 4 ‖z‖²`. -/
lemma sum_inner_u_sq_le (S : Finset ℕ) (hS : S ⊆ range (d + 1)) (z : E d) :
    ∑ k ∈ S, ⟪u d k, z⟫ ^ 2 ≤ 4 * ‖z‖ ^ 2 := by
  have hsub : ∑ k ∈ S, ⟪u d k, z⟫ ^ 2 ≤ ∑ k ∈ range (d + 1), ⟪u d k, z⟫ ^ 2 :=
    Finset.sum_le_sum_of_subset_of_nonneg hS fun k _ _ => sq_nonneg _
  refine hsub.trans ?_
  have hpt : ∀ k ∈ range (d + 1), ⟪u d k, z⟫ ^ 2 ≤
      2 * (if k = 0 then 0 else cx z (k - 1)) ^ 2 + 2 * cx z k ^ 2 := by
    intro k _
    rw [inner_u]
    nlinarith [sq_nonneg ((if k = 0 then 0 else cx z (k - 1)) + cx z k)]
  refine (Finset.sum_le_sum hpt).trans ?_
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have ha : ∑ k ∈ range (d + 1), (if k = 0 then 0 else cx z (k - 1)) ^ 2 =
      ∑ k ∈ range d, cx z k ^ 2 := by
    rw [Finset.sum_range_succ']
    simp
  have hb : ∑ k ∈ range (d + 1), cx z k ^ 2 = ∑ k ∈ range d, cx z k ^ 2 := by
    rw [Finset.sum_range_succ, cx_of_ge le_rfl]
    simp
  rw [ha, hb, ← norm_sq_eq_sum]
  linarith

/-- Support propagation for the chain: if `x` vanishes from coordinate `j` on, the gradient of
`quad (u d) β S w c` vanishes from coordinate `j + 1` on, and from `j` on when `j ∉ S`. -/
lemma quadGrad_u_vanish (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (hc : ∀ k, k ≠ 0 → c k = 0)
    (j : ℕ) (x : E d) (hx : ∀ k, j ≤ k → cx x k = 0) (p : Fin d)
    (hp : j + (if j ∈ S then 1 else 0) ≤ (p : ℕ)) : quadGrad (u d) β S w c x p = 0 := by
  unfold quadGrad
  rw [PiLp.smul_apply, smul_eq_mul]
  have hsum : (∑ k ∈ S, (w k * (⟪u d k, x⟫ + c k)) • u d k) p = 0 := by
    rw [WithLp.ofLp_sum, Finset.sum_apply]
    refine Finset.sum_eq_zero fun k hk => ?_
    rw [PiLp.smul_apply, smul_eq_mul, u_apply]
    by_cases hup : ((if k = 0 then 0 else (if (p : ℕ) = k - 1 then 1 else 0)) -
        (if (p : ℕ) = k then 1 else 0) : ℝ) = 0
    · rw [hup, mul_zero]
    · have hpk : (p : ℕ) = k ∨ (k ≠ 0 ∧ (p : ℕ) = k - 1) := by
        by_contra hcon
        push_neg at hcon
        apply hup
        by_cases hk0 : k = 0
        · subst hk0; simp [hcon.1]
        · simp [hk0, hcon.1, hcon.2 hk0]
      have hjk : j < k := by
        rcases hpk with h | ⟨hk0, h⟩
        · by_contra hcon
          push_neg at hcon
          have : k = j := by
            split_ifs at hp <;> omega
          subst this
          rw [if_pos hk] at hp
          omega
        · split_ifs at hp <;> omega
      have hk0 : k ≠ 0 := by omega
      rw [inner_u, if_neg hk0, hx (k - 1) (by omega), hx k (by omega), hc k hk0]
      simp
  rw [hsum, mul_zero]

end Chain
end ULB63
