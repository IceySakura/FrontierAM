import Theory.ChainBasic

/-!
# The chain objective, its minimizer, and the gap bound

`φc x = (α/2) [ (x₀ - R)² + ∑_{k=1}^{d-1} (x_{k-1} - x_k)² + (1 - q) x_{d-1}² ] + (μ/2) ‖x‖²`
with `μ q = α (1 - q)²` has the minimizer `x⋆_k = R q^{k+1}`, and
`φc x - φc x⋆ = (α/2) ∑ w_k ⟪u_k, x - x⋆⟫² + (μ/2) ‖x - x⋆‖²`.
-/

open scoped RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {d : ℕ}

/-- Weights: `1` on the links, `1 - q` on the terminal term `k = d`. -/
def wch (d : ℕ) (q : ℝ) (k : ℕ) : ℝ := if k < d then 1 else 1 - q

/-- Offsets: the anchor `-(x₀ - R)` for `k = 0`. -/
def cch (R : ℝ) (k : ℕ) : ℝ := if k = 0 then R else 0

/-- The chain objective. -/
def φc (d : ℕ) (α μ R q : ℝ) (x : E d) : ℝ :=
  quad (u d) α (range (d + 1)) (wch d q) (cch R) x + μ / 2 * ‖x‖ ^ 2

/-- The minimizer `x⋆_k = R q^{k+1}`. -/
def xstar (d : ℕ) (R q : ℝ) : E d := WithLp.toLp 2 (fun p : Fin d => R * q ^ ((p : ℕ) + 1))

lemma cx_xstar (R q : ℝ) (k : ℕ) : cx (xstar d R q) k = if k < d then R * q ^ (k + 1) else 0 := by
  unfold cx xstar
  split_ifs <;> simp

lemma quad_add (U : ℕ → E d) (β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (y v : E d) :
    quad U β S w c (y + v) = quad U β S w c y +
      β * ∑ k ∈ S, w k * (⟪U k, y⟫ + c k) * ⟪U k, v⟫ + β / 2 * ∑ k ∈ S, w k * ⟪U k, v⟫ ^ 2 := by
  unfold quad
  simp only [inner_add_right]
  have : ∀ k ∈ S, w k * (⟪U k, y⟫ + ⟪U k, v⟫ + c k) ^ 2 = w k * (⟪U k, y⟫ + c k) ^ 2 +
      2 * (w k * (⟪U k, y⟫ + c k) * ⟪U k, v⟫) + w k * ⟪U k, v⟫ ^ 2 := fun k _ => by ring
  rw [Finset.sum_congr rfl this, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

lemma wg_xstar (hd : 1 ≤ d) (R q : ℝ) (k : ℕ) (hk : k ∈ range (d + 1)) :
    wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) = R * (1 - q) * q ^ k := by
  rw [Finset.mem_range] at hk
  rw [inner_u, cx_xstar, cx_xstar]
  unfold wch cch
  by_cases hk0 : k = 0
  · subst hk0
    simp [show 0 < d by omega]
    ring
  · by_cases hkd : k < d
    · rw [if_pos hkd, if_neg hk0, if_neg hk0, if_pos (by omega : k - 1 < d), if_pos hkd]
      have : k - 1 + 1 = k := by omega
      rw [this]
      ring
    · have hkd' : k = d := by omega
      subst hkd'
      rw [if_neg hkd, if_neg hk0, if_neg hk0, if_pos (by omega : k - 1 < k), if_neg hkd]
      have : k - 1 + 1 = k := by omega
      rw [this]
      ring

lemma lin_part (hd : 1 ≤ d) {α μ R q : ℝ} (hq : μ * q = α * (1 - q) ^ 2) (v : E d) :
    α * ∑ k ∈ range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) * ⟪u d k, v⟫ +
      μ * ⟪xstar d R q, v⟫ = 0 := by
  have h1 : ∑ k ∈ range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) * ⟪u d k, v⟫ =
      R * (1 - q) * ∑ k ∈ range (d + 1), q ^ k * ⟪u d k, v⟫ := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [wg_xstar hd R q k hk]
    ring
  have h2 : ∑ k ∈ range (d + 1), q ^ k * ⟪u d k, v⟫ =
      (q - 1) * ∑ k ∈ range d, q ^ k * cx v k := by
    simp only [inner_u, mul_sub, Finset.sum_sub_distrib]
    rw [Finset.sum_range_succ', Finset.sum_range_succ (fun k => q ^ k * cx v k), cx_of_ge le_rfl]
    simp only [if_true, mul_zero, add_zero, Nat.add_sub_cancel, Nat.succ_ne_zero, if_false]
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  have h3 : ⟪xstar d R q, v⟫ = R * q * ∑ k ∈ range d, q ^ k * cx v k := by
    rw [inner_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [cx_xstar, if_pos (Finset.mem_range.1 hk)]
    ring
  rw [h1, h2, h3]
  have : α * (R * (1 - q) * ((q - 1) * ∑ k ∈ range d, q ^ k * cx v k)) +
      μ * (R * q * ∑ k ∈ range d, q ^ k * cx v k) =
      R * (μ * q - α * (1 - q) ^ 2) * ∑ k ∈ range d, q ^ k * cx v k := by ring
  rw [this, hq, sub_self, mul_zero, zero_mul]

/-- The exact quadratic expansion of `φc` around `x⋆`. -/
lemma φc_eq (hd : 1 ≤ d) {α μ R q : ℝ} (hq : μ * q = α * (1 - q) ^ 2) (x : E d) :
    φc d α μ R q x = φc d α μ R q (xstar d R q) +
      α / 2 * ∑ k ∈ range (d + 1), wch d q k * ⟪u d k, x - xstar d R q⟫ ^ 2 +
      μ / 2 * ‖x - xstar d R q‖ ^ 2 := by
  have hx : x = xstar d R q + (x - xstar d R q) := by abel
  conv_lhs => rw [hx]
  unfold φc
  rw [quad_add, norm_add_sq_real]
  have := lin_part hd (R := R) hq (x - xstar d R q)
  linear_combination (1 : ℝ) * this

lemma wch_nonneg {q : ℝ} (hq1 : q ≤ 1) (k : ℕ) : 0 ≤ wch d q k := by
  unfold wch; split_ifs <;> linarith

lemma wch_le_one {q : ℝ} (hq0 : 0 ≤ q) (k : ℕ) : wch d q k ≤ 1 := by
  unfold wch; split_ifs <;> linarith

lemma quad_nonneg (U : ℕ → E d) {β : ℝ} (hβ : 0 ≤ β) (S : Finset ℕ) (w c : ℕ → ℝ)
    (hw : ∀ k, 0 ≤ w k) (x : E d) : 0 ≤ quad U β S w c x := by
  unfold quad
  exact mul_nonneg (by linarith) (Finset.sum_nonneg fun k _ => mul_nonneg (hw k) (sq_nonneg _))

lemma φc_xstar_le (hd : 1 ≤ d) {α μ R q : ℝ} (hα : 0 ≤ α) (hμ : 0 ≤ μ) (hq1 : q ≤ 1)
    (hq : μ * q = α * (1 - q) ^ 2) (x : E d) :
    φc d α μ R q (xstar d R q) ≤ φc d α μ R q x := by
  rw [φc_eq hd hq x]
  have h1 : 0 ≤ α / 2 * ∑ k ∈ range (d + 1), wch d q k * ⟪u d k, x - xstar d R q⟫ ^ 2 :=
    mul_nonneg (by linarith) (Finset.sum_nonneg fun k _ =>
      mul_nonneg (wch_nonneg hq1 k) (sq_nonneg _))
  have h2 : 0 ≤ μ / 2 * ‖x - xstar d R q‖ ^ 2 := by positivity
  linarith

/-- Gap lower bound: if `x` vanishes at coordinate `K < d`, then
`φc x - φc x⋆ ≥ (μ/2) R² q^{2K+2}`. -/
lemma φc_gap_ge (hd : 1 ≤ d) {α μ R q : ℝ} (hα : 0 ≤ α) (hμ : 0 ≤ μ) (hq1 : q ≤ 1)
    (hq : μ * q = α * (1 - q) ^ 2) (x : E d) (K : ℕ) (hK : K < d) (hxK : cx x K = 0) :
    μ / 2 * (R ^ 2 * q ^ (2 * K + 2)) ≤ φc d α μ R q x - φc d α μ R q (xstar d R q) := by
  rw [φc_eq hd hq x]
  have h1 : 0 ≤ α / 2 * ∑ k ∈ range (d + 1), wch d q k * ⟪u d k, x - xstar d R q⟫ ^ 2 :=
    mul_nonneg (by linarith) (Finset.sum_nonneg fun k _ =>
      mul_nonneg (wch_nonneg hq1 k) (sq_nonneg _))
  have h2 : cx (x - xstar d R q) K ^ 2 ≤ ‖x - xstar d R q‖ ^ 2 := sq_cx_le_norm_sq _ K
  rw [cx_sub, hxK, cx_xstar, if_pos hK] at h2
  have h3 : (0 - R * q ^ (K + 1)) ^ 2 = R ^ 2 * q ^ (2 * K + 2) := by ring
  rw [h3] at h2
  have h4 : μ / 2 * (R ^ 2 * q ^ (2 * K + 2)) ≤ μ / 2 * ‖x - xstar d R q‖ ^ 2 :=
    mul_le_mul_of_nonneg_left h2 (by linarith)
  linarith

lemma φc_zero (hd : 1 ≤ d) (α μ R q : ℝ) : φc d α μ R q 0 = α / 2 * R ^ 2 := by
  unfold φc quad
  rw [Finset.sum_eq_single 0]
  · simp [wch, cch, show 0 < d by omega]
  · intro k _ hk
    simp [cch, hk]
  · intro h; exact absurd (Finset.mem_range.2 (by omega)) h

end Chain
end ULB63

namespace ULB63
namespace Chain

variable {d : ℕ}

/-- The optimal value: `φc x⋆ = (α/2) R² (1 - q)`. -/
lemma φc_xstar_val (hd : 1 ≤ d) {α μ R q : ℝ} (hq : μ * q = α * (1 - q) ^ 2) :
    φc d α μ R q (xstar d R q) = α / 2 * R ^ 2 * (1 - q) := by
  have hlin := lin_part hd (R := R) hq (xstar d R q)
  rw [real_inner_self_eq_norm_sq] at hlin
  -- `⟪u_k, x⋆⟫ = g_k - c_k`
  have hsplit : ∑ k ∈ Finset.range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) *
      ⟪u d k, xstar d R q⟫ =
      ∑ k ∈ Finset.range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) ^ 2 -
      ∑ k ∈ Finset.range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) * cch R k := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  have hc : ∑ k ∈ Finset.range (d + 1), wch d q k * (⟪u d k, xstar d R q⟫ + cch R k) * cch R k =
      R ^ 2 * (1 - q) := by
    rw [Finset.sum_eq_single 0]
    · rw [wg_xstar hd R q 0 (Finset.mem_range.2 (by omega))]
      simp [cch]; ring
    · intro k _ hk; simp [cch, hk]
    · intro h; exact absurd (Finset.mem_range.2 (by omega)) h
  rw [hsplit, hc] at hlin
  unfold φc quad
  linarith

end Chain
end ULB63
