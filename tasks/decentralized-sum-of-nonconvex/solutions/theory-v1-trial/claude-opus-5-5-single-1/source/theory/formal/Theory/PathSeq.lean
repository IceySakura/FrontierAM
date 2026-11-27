import Mathlib

/-!
# Sequence inequalities for the path graph

For a sequence `U` on `{0, …, N-1}`: the Laplacian quadratic form, the bound on the squared
Laplacian, and the discrete Poincaré inequality `∑ U_i² ≤ (N²/2) ∑ (U_k - U_{k+1})²` for
mean-zero `U`.
-/

open Finset

noncomputable section

namespace ULB63
namespace Path

/-- The path Laplacian applied to `U` at node `i` of `{0, …, N-1}`. -/
def lap (U : ℕ → ℝ) (N i : ℕ) : ℝ :=
  (if i = 0 then 0 else U i - U (i - 1)) + (if i + 1 = N then 0 else U i - U (i + 1))

/-- `∑_k (U_k - U_{k+1})²` over the `N - 1` edges. -/
def edgeSum (U : ℕ → ℝ) (N : ℕ) : ℝ := ∑ k ∈ range (N - 1), (U k - U (k + 1)) ^ 2

lemma sum_first (U : ℕ → ℝ) (M : ℕ) (g : ℝ → ℝ → ℝ) :
    ∑ i ∈ range (M + 1), (if i = 0 then 0 else g (U i) (U (i - 1))) =
      ∑ k ∈ range M, g (U (k + 1)) (U k) := by
  rw [Finset.sum_range_succ']
  simp

lemma sum_second (U : ℕ → ℝ) (M : ℕ) (g : ℝ → ℝ → ℝ) :
    ∑ i ∈ range (M + 1), (if i + 1 = M + 1 then 0 else g (U i) (U (i + 1))) =
      ∑ k ∈ range M, g (U k) (U (k + 1)) := by
  rw [Finset.sum_range_succ]
  simp only [if_true, add_zero]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.mem_range] at hk
  rw [if_neg (by omega)]

lemma lap_inner (U : ℕ → ℝ) (N : ℕ) :
    ∑ i ∈ range N, U i * lap U N i = edgeSum U N := by
  rcases N with _ | M
  · simp [edgeSum]
  · unfold lap edgeSum
    simp only [mul_add, Finset.sum_add_distrib, Nat.add_sub_cancel]
    have h1 : ∑ i ∈ range (M + 1), U i * (if i = 0 then 0 else U i - U (i - 1)) =
        ∑ i ∈ range (M + 1), (if i = 0 then 0 else U i * (U i - U (i - 1))) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      split_ifs <;> ring
    have h2 : ∑ i ∈ range (M + 1), U i * (if i + 1 = M + 1 then 0 else U i - U (i + 1)) =
        ∑ i ∈ range (M + 1), (if i + 1 = M + 1 then 0 else U i * (U i - U (i + 1))) := by
      refine Finset.sum_congr rfl fun i _ => ?_
      split_ifs <;> ring
    rw [h1, h2, sum_first U M (fun a b => a * (a - b)), sum_second U M (fun a b => a * (a - b)),
      ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring

lemma lap_sq_le (U : ℕ → ℝ) (N : ℕ) :
    ∑ i ∈ range N, lap U N i ^ 2 ≤ 4 * edgeSum U N := by
  rcases N with _ | M
  · simp [edgeSum]
  · have hpt : ∀ i ∈ range (M + 1), lap U (M + 1) i ^ 2 ≤
        2 * (if i = 0 then 0 else (U i - U (i - 1)) ^ 2) +
        2 * (if i + 1 = M + 1 then 0 else (U i - U (i + 1)) ^ 2) := by
      intro i _
      unfold lap
      split_ifs <;> nlinarith [sq_nonneg (U i - U (i - 1) - (U i - U (i + 1)))]
    refine (Finset.sum_le_sum hpt).trans ?_
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      sum_first U M (fun a b => (a - b) ^ 2), sum_second U M (fun a b => (a - b) ^ 2)]
    unfold edgeSum
    simp only [Nat.add_sub_cancel]
    have : ∑ k ∈ range M, (U (k + 1) - U k) ^ 2 = ∑ k ∈ range M, (U k - U (k + 1)) ^ 2 :=
      Finset.sum_congr rfl fun k _ => by ring
    rw [this]
    linarith

/-- Telescoping and Cauchy–Schwarz: `(U_j - U_i)² ≤ N · edgeSum` for `i ≤ j < N`. -/
lemma diff_sq_le (U : ℕ → ℝ) (N i j : ℕ) (hij : i ≤ j) (hj : j < N) :
    (U j - U i) ^ 2 ≤ N * edgeSum U N := by
  have htel : U j - U i = ∑ k ∈ Ico i j, (U (k + 1) - U k) := by
    rw [Finset.sum_Ico_eq_sum_range]
    have := Finset.sum_range_sub (fun k => U (i + k)) (j - i)
    rw [show i + (j - i) = j by omega, add_zero] at this
    rw [← this]
    exact Finset.sum_congr rfl fun k _ => by rw [add_assoc]
  rw [htel]
  have hcs := sq_sum_le_card_mul_sum_sq (s := Ico i j) (f := fun k => U (k + 1) - U k)
  refine hcs.trans ?_
  rw [Nat.card_Ico]
  have hsub : ∑ k ∈ Ico i j, (U (k + 1) - U k) ^ 2 ≤ edgeSum U N := by
    unfold edgeSum
    have : ∑ k ∈ Ico i j, (U (k + 1) - U k) ^ 2 = ∑ k ∈ Ico i j, (U k - U (k + 1)) ^ 2 :=
      Finset.sum_congr rfl fun k _ => by ring
    rw [this]
    apply Finset.sum_le_sum_of_subset_of_nonneg
    · intro k hk
      rw [Finset.mem_Ico] at hk
      rw [Finset.mem_range]
      omega
    · intro k _ _; exact sq_nonneg _
  have hE : 0 ≤ edgeSum U N := Finset.sum_nonneg fun k _ => sq_nonneg _
  have hc : ((j - i : ℕ) : ℝ) ≤ N := by exact_mod_cast (by omega : j - i ≤ N)
  calc ((j - i : ℕ) : ℝ) * ∑ k ∈ Ico i j, (U (k + 1) - U k) ^ 2 ≤ (N : ℝ) * edgeSum U N :=
        mul_le_mul hc hsub (Finset.sum_nonneg fun k _ => sq_nonneg _) (Nat.cast_nonneg N)

lemma pair_sq_le (U : ℕ → ℝ) (N i j : ℕ) (hi : i < N) (hj : j < N) :
    (U i - U j) ^ 2 ≤ N * edgeSum U N := by
  rcases le_total i j with h | h
  · have := diff_sq_le U N i j h hj
    calc (U i - U j) ^ 2 = (U j - U i) ^ 2 := by ring
      _ ≤ _ := this
  · exact diff_sq_le U N j i h hi

/-- The discrete Poincaré inequality on the path. -/
lemma poincare (U : ℕ → ℝ) (N : ℕ) (hN : 1 ≤ N) (h0 : ∑ i ∈ range N, U i = 0) :
    ∑ i ∈ range N, U i ^ 2 ≤ (N : ℝ) ^ 2 / 2 * edgeSum U N := by
  have hid : ∑ i ∈ range N, ∑ j ∈ range N, (U i - U j) ^ 2 =
      2 * N * ∑ i ∈ range N, U i ^ 2 - 2 * (∑ i ∈ range N, U i) ^ 2 := by
    have : ∀ i j, (U i - U j) ^ 2 = U i ^ 2 + U j ^ 2 - 2 * (U i * U j) := fun i j => by ring
    simp only [this, Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul, ← Finset.mul_sum, ← Finset.sum_mul]
    ring
  rw [h0] at hid
  have hle : ∑ i ∈ range N, ∑ j ∈ range N, (U i - U j) ^ 2 ≤
      ∑ i ∈ range N, ∑ j ∈ range N, (N : ℝ) * edgeSum U N :=
    Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun j hj =>
      pair_sq_le U N i j (Finset.mem_range.1 hi) (Finset.mem_range.1 hj)
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hle
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have : 2 * (N : ℝ) * ∑ i ∈ range N, U i ^ 2 ≤ (N : ℝ) * ((N : ℝ) * ((N : ℝ) * edgeSum U N)) := by
    linarith
  have h2 : ∑ i ∈ range N, U i ^ 2 ≤ (N : ℝ) * ((N : ℝ) * edgeSum U N) / 2 := by
    rw [le_div_iff₀ (by norm_num)]
    nlinarith
  calc ∑ i ∈ range N, U i ^ 2 ≤ (N : ℝ) * ((N : ℝ) * edgeSum U N) / 2 := h2
    _ = (N : ℝ) ^ 2 / 2 * edgeSum U N := by ring

end Path
end ULB63
