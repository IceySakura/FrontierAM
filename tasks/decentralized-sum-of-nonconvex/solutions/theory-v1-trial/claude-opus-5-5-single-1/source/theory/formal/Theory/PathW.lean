import Theory.PathSeq
import Theory.Model

/-!
# The lazy path gossip matrix

`pathW n` puts weight `1/4` on each edge `{k, k+1}` of the path `0 - 1 - ⋯ - (n-1)` and the rest on
the diagonal. It belongs to `𝓦_{n,ρ}` for every `ρ ≥ √(1 - 1/(2n²))`.
-/

open Finset

noncomputable section

namespace ULB63
namespace Path

/-- A vector of `ℝ^n` as a sequence (zero beyond `n`). -/
def vn {n : ℕ} (v : Fin n → ℝ) (k : ℕ) : ℝ := if h : k < n then v ⟨k, h⟩ else 0

lemma vn_fin {n : ℕ} (v : Fin n → ℝ) (i : Fin n) : vn v i = v i := by simp [vn, i.isLt]

/-- The lazy path gossip matrix. -/
def pathW (n : ℕ) : Matrix (Fin n) (Fin n) ℝ := fun i j =>
  if (i : ℕ) = j then 1 - (if (i : ℕ) = 0 then 0 else 1 / 4) - (if (i : ℕ) + 1 = n then 0 else 1 / 4)
  else if (i : ℕ) + 1 = j ∨ (j : ℕ) + 1 = i then 1 / 4 else 0

lemma pathW_mulVec (n : ℕ) (v : Fin n → ℝ) (i : Fin n) :
    (pathW n).mulVec v i = v i - 1 / 4 * lap (vn v) n i := by
  have hsplit : ∀ j : Fin n, pathW n i j * v j =
      (if (i : ℕ) = j then (1 - (if (i : ℕ) = 0 then 0 else 1 / 4) -
        (if (i : ℕ) + 1 = n then 0 else 1 / 4)) * v j else 0) +
      (if (i : ℕ) + 1 = j then 1 / 4 * v j else 0) +
      (if (j : ℕ) + 1 = i then 1 / 4 * v j else 0) := by
    intro j
    by_cases h1 : (i : ℕ) = j
    · have hij : i = j := Fin.ext h1
      subst hij
      simp [pathW]
    · by_cases h2 : (i : ℕ) + 1 = j
      · have h3 : ¬ ((j : ℕ) + 1 = i) := by omega
        simp [pathW, h1, h2, h3]
      · by_cases h3 : (j : ℕ) + 1 = i
        · simp [pathW, h1, h2, h3]
        · simp [pathW, h1, h2, h3]
  simp only [Matrix.mulVec, dotProduct]
  rw [Finset.sum_congr rfl fun j _ => hsplit j, Finset.sum_add_distrib, Finset.sum_add_distrib]
  have hA : ∑ j : Fin n, (if (i : ℕ) = j then (1 - (if (i : ℕ) = 0 then 0 else 1 / 4) -
      (if (i : ℕ) + 1 = n then 0 else 1 / 4)) * v j else 0) =
      (1 - (if (i : ℕ) = 0 then 0 else 1 / 4) - (if (i : ℕ) + 1 = n then 0 else 1 / 4)) * v i := by
    rw [Finset.sum_eq_single i]
    · simp
    · intro j _ hj; rw [if_neg (fun h => hj (Fin.ext h).symm)]
    · intro h; exact absurd (Finset.mem_univ i) h
  have hB : ∑ j : Fin n, (if (i : ℕ) + 1 = j then 1 / 4 * v j else 0) =
      1 / 4 * vn v (i + 1) := by
    by_cases hi : (i : ℕ) + 1 < n
    · rw [Finset.sum_eq_single ⟨(i : ℕ) + 1, hi⟩]
      · simp [vn, hi]
      · intro j _ hj
        rw [if_neg]
        intro h; apply hj; exact Fin.ext h.symm
      · intro h; exact absurd (Finset.mem_univ _) h
    · rw [Finset.sum_eq_zero]
      · simp [vn, hi]
      · intro j _
        rw [if_neg]
        intro h; have := j.isLt; omega
  have hC : ∑ j : Fin n, (if (j : ℕ) + 1 = i then 1 / 4 * v j else 0) =
      if (i : ℕ) = 0 then 0 else 1 / 4 * vn v (i - 1) := by
    by_cases hi : (i : ℕ) = 0
    · rw [if_pos hi, Finset.sum_eq_zero]
      intro j _
      rw [if_neg (by omega)]
    · rw [if_neg hi, Finset.sum_eq_single ⟨(i : ℕ) - 1, by omega⟩]
      · rw [if_pos (by simp; omega)]
        simp [vn, show (i : ℕ) - 1 < n by omega]
      · intro j _ hj
        rw [if_neg]
        intro h; apply hj; ext; simp; omega
      · intro h; exact absurd (Finset.mem_univ _) h
  rw [hA, hB, hC]
  unfold lap
  rw [vn_fin]
  by_cases hn : (i : ℕ) + 1 = n
  · have hvn0 : vn v ((i : ℕ) + 1) = 0 := by simp [vn, show ¬ ((i : ℕ) + 1 < n) by omega]
    by_cases h0 : (i : ℕ) = 0
    · simp only [if_pos h0, if_pos hn, hvn0]; ring
    · simp only [if_neg h0, if_pos hn, hvn0]; ring
  · by_cases h0 : (i : ℕ) = 0
    · simp only [if_pos h0, if_neg hn]; ring
    · simp only [if_neg h0, if_neg hn]; ring

lemma lap_const (n : ℕ) (a : ℝ) (i : Fin n) : lap (vn (fun _ : Fin n => a)) n i = 0 := by
  unfold lap
  have h1 : (i : ℕ) ≠ 0 → vn (fun _ : Fin n => a) ((i : ℕ) - 1) = a := fun _ => by
    simp [vn, show (i : ℕ) - 1 < n by omega]
  have h2 : (i : ℕ) + 1 ≠ n → vn (fun _ : Fin n => a) ((i : ℕ) + 1) = a := fun h => by
    have := i.isLt
    simp [vn, show (i : ℕ) + 1 < n by omega]
  have h3 : vn (fun _ : Fin n => a) (i : ℕ) = a := by simp [vn, i.isLt]
  split_ifs with ha hb hb
  · simp
  · rw [h2 hb, h3]; simp
  · rw [h1 ha, h3]; simp
  · rw [h1 ha, h2 hb, h3]; simp

lemma lap_shift (n : ℕ) (v : Fin n → ℝ) (a : ℝ) (i : Fin n) :
    lap (vn (fun j => v j - a)) n i = lap (vn v) n i := by
  unfold lap
  have h1 : (i : ℕ) ≠ 0 → vn (fun j => v j - a) ((i : ℕ) - 1) = vn v ((i : ℕ) - 1) - a := fun _ => by
    simp [vn, show (i : ℕ) - 1 < n by omega]
  have h2 : (i : ℕ) + 1 ≠ n → vn (fun j => v j - a) ((i : ℕ) + 1) = vn v ((i : ℕ) + 1) - a :=
    fun h => by
      have := i.isLt
      simp [vn, show (i : ℕ) + 1 < n by omega]
  have h3 : vn (fun j => v j - a) (i : ℕ) = vn v i - a := by simp [vn, i.isLt]
  split_ifs with ha hb hb
  · simp
  · rw [h2 hb, h3]; ring
  · rw [h1 ha, h3]; ring
  · rw [h1 ha, h2 hb, h3]; ring

/-- `pathW n ∈ 𝓦_{n,ρ}` for `ρ ≥ √(1 - 1/(2n²))`. -/
theorem pathW_commClass {n : ℕ} (hn : 1 ≤ n) {ρ : ℝ} (hρ : Real.sqrt (1 - 1 / (2 * n ^ 2)) ≤ ρ) :
    CommClass n ρ (pathW n) where
  nonneg i j := by unfold pathW; split_ifs <;> norm_num
  symm := by
    ext i j
    simp only [Matrix.transpose_apply, pathW]
    by_cases h : (i : ℕ) = j
    · have : i = j := Fin.ext h
      subst this; rfl
    · rw [if_neg (fun h' => h h'.symm), if_neg h]
      by_cases h2 : (i : ℕ) + 1 = j ∨ (j : ℕ) + 1 = i
      · rw [if_pos (by tauto), if_pos h2]
      · rw [if_neg (by tauto), if_neg h2]
  row_sum i := by
    have := pathW_mulVec n (fun _ => 1) i
    simp only [Matrix.mulVec, dotProduct, mul_one, lap_const, mul_zero, sub_zero] at this
    exact this
  contraction v := by
    have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    set vbar : ℝ := (∑ j, v j) / n with hvbar
    set u : Fin n → ℝ := fun j => v j - vbar with hu
    have hJ : ∀ i, (J n).mulVec v i = vbar := by
      intro i
      simp only [J, Matrix.mulVec, dotProduct, hvbar]
      rw [← Finset.mul_sum]; field_simp
    have hcomp : ∀ i, ((pathW n - J n).mulVec v) i = u i - 1 / 4 * lap (vn u) n i := by
      intro i
      rw [Matrix.sub_mulVec, Pi.sub_apply, pathW_mulVec, hJ, hu, lap_shift]
      ring
    -- sums over `range n`
    set U : ℕ → ℝ := vn u with hU
    have hfin : ∀ (g : ℕ → ℝ), ∑ i : Fin n, g i = ∑ i ∈ range n, g i :=
      fun g => Fin.sum_univ_eq_sum_range g n
    have hsumU : ∑ i ∈ range n, U i = 0 := by
      rw [← hfin]
      simp only [hU, vn_fin, hu, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul, hvbar]
      field_simp
      ring
    have hUv : ∑ i ∈ range n, U i ^ 2 ≤ ∑ i, v i ^ 2 := by
      rw [← hfin]
      simp only [hU, vn_fin, hu]
      have : ∑ i, (v i - vbar) ^ 2 = ∑ i, v i ^ 2 - n * vbar ^ 2 := by
        have h1 : ∀ i, (v i - vbar) ^ 2 = v i ^ 2 - 2 * vbar * v i + vbar ^ 2 := fun i => by ring
        simp only [h1, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
          Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        have : ∑ i, v i = n * vbar := by rw [hvbar]; field_simp
        rw [this]; ring
      rw [this]
      nlinarith [sq_nonneg vbar]
    have hnorm : ∑ i, (((pathW n - J n).mulVec v) i) ^ 2 =
        ∑ i ∈ range n, (U i - 1 / 4 * lap U n i) ^ 2 := by
      rw [← hfin]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [hcomp, hU, vn_fin]
    have hexp : ∑ i ∈ range n, (U i - 1 / 4 * lap U n i) ^ 2 =
        ∑ i ∈ range n, U i ^ 2 - 1 / 2 * ∑ i ∈ range n, U i * lap U n i +
          1 / 16 * ∑ i ∈ range n, lap U n i ^ 2 := by
      have : ∀ i, (U i - 1 / 4 * lap U n i) ^ 2 =
          U i ^ 2 - 1 / 2 * (U i * lap U n i) + 1 / 16 * lap U n i ^ 2 := fun i => by ring
      simp only [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    have hP := poincare U n hn hsumU
    have hI := lap_inner U n
    have hS := lap_sq_le U n
    have hE : 0 ≤ edgeSum U n := Finset.sum_nonneg fun k _ => sq_nonneg _
    have hkey : ∑ i, (((pathW n - J n).mulVec v) i) ^ 2 ≤
        (1 - 1 / (2 * n ^ 2)) * ∑ i, v i ^ 2 := by
      rw [hnorm, hexp, hI]
      have h1 : ∑ i ∈ range n, U i ^ 2 - 1 / 2 * edgeSum U n +
          1 / 16 * ∑ i ∈ range n, lap U n i ^ 2 ≤ ∑ i ∈ range n, U i ^ 2 - 1 / 4 * edgeSum U n := by
        linarith
      have h2 : (∑ i ∈ range n, U i ^ 2) / (2 * n ^ 2) ≤ 1 / 4 * edgeSum U n := by
        rw [div_le_iff₀ (by positivity)]
        nlinarith
      have h3 : 0 ≤ 1 - 1 / (2 * (n : ℝ) ^ 2) := by
        have h1n : (1 : ℝ) ≤ n := by exact_mod_cast hn
        have : (1 : ℝ) ≤ n ^ 2 := by nlinarith
        rw [sub_nonneg, div_le_one (by positivity)]
        linarith
      calc _ ≤ ∑ i ∈ range n, U i ^ 2 - 1 / 4 * edgeSum U n := h1
        _ ≤ ∑ i ∈ range n, U i ^ 2 - (∑ i ∈ range n, U i ^ 2) / (2 * n ^ 2) := by linarith
        _ = (1 - 1 / (2 * n ^ 2)) * ∑ i ∈ range n, U i ^ 2 := by ring
        _ ≤ (1 - 1 / (2 * n ^ 2)) * ∑ i, v i ^ 2 := mul_le_mul_of_nonneg_left hUv h3
    have hρ0 : 0 ≤ ρ := (Real.sqrt_nonneg _).trans hρ
    unfold norm2
    rw [show ρ * Real.sqrt (∑ i, v i ^ 2) = Real.sqrt (ρ ^ 2 * ∑ i, v i ^ 2) by
      rw [Real.sqrt_mul (sq_nonneg ρ), Real.sqrt_sq hρ0]]
    apply Real.sqrt_le_sqrt
    have hρ2 : 1 - 1 / (2 * (n : ℝ) ^ 2) ≤ ρ ^ 2 := by
      have h3 : 0 ≤ 1 - 1 / (2 * (n : ℝ) ^ 2) := by
        have h1n : (1 : ℝ) ≤ n := by exact_mod_cast hn
        have : (1 : ℝ) ≤ n ^ 2 := by nlinarith
        rw [sub_nonneg, div_le_one (by positivity)]
        linarith
      calc 1 - 1 / (2 * (n : ℝ) ^ 2) = Real.sqrt (1 - 1 / (2 * n ^ 2)) ^ 2 := (Real.sq_sqrt h3).symm
        _ ≤ ρ ^ 2 := pow_le_pow_left₀ (Real.sqrt_nonneg _) hρ 2
    calc _ ≤ (1 - 1 / (2 * n ^ 2)) * ∑ i, v i ^ 2 := hkey
      _ ≤ ρ ^ 2 * ∑ i, v i ^ 2 :=
        mul_le_mul_of_nonneg_right hρ2 (Finset.sum_nonneg fun i _ => sq_nonneg _)

end Path
end ULB63
