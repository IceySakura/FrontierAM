import Theory.Combined
import Theory.PathW
import Theory.Transfer

/-!
# Network term in the minimax bound: the lazy path, `ρ ≥ √(1 - 1/(2n²))`
-/

open MeasureTheory
open scoped ENNReal
open Finset

noncomputable section

namespace ULB63
namespace Path

variable {n m c : ℕ}

lemma pathW_ne_zero {n : ℕ} {i j : Fin n} (h : pathW n i j ≠ 0) :
    (i : ℕ) = j ∨ (i : ℕ) + 1 = j ∨ (j : ℕ) + 1 = i := by
  unfold pathW at h
  by_cases h1 : (i : ℕ) = j
  · exact Or.inl h1
  · rw [if_neg h1] at h
    by_cases h2 : (i : ℕ) + 1 = j ∨ (j : ℕ) + 1 = i
    · exact Or.inr h2
    · rw [if_neg h2] at h; exact absurd rfl h

theorem minimax_ge_path (hn : 4 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : Real.sqrt (1 - 1 / (2 * n ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ n / 2 * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  classical
  refine Chain.le_minimax_of _ fun A hA => ⟨pathW n, pathW_commClass (by omega) hρ, ?_⟩
  set a := n / 4 with ha
  have ha1 : 1 ≤ a := by omega
  have h4a : 4 * a ≤ n := by omega
  have h7a : n ≤ 7 * a := by omega
  set D := n - 2 * a + 1 with hD
  have hD1 : 1 ≤ D := by omega
  let SA : Finset (Fin n) := univ.filter (fun i => (i : ℕ) < a)
  let SB : Finset (Fin n) := univ.filter (fun i => n - a ≤ (i : ℕ))
  have hSAcard : SA.card = a := by
    simp only [SA, Fin.card_filter_val_lt]; omega
  have hSBcard : SB.card = a := by
    have h1 := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Fin n)))
      (fun i : Fin n => (i : ℕ) < n - a)
    rw [Fin.card_filter_val_lt, Finset.card_univ, Fintype.card_fin] at h1
    have h2 : (univ.filter (fun i : Fin n => ¬ ((i : ℕ) < n - a))) = SB := by
      ext i; simp only [SB, mem_filter, mem_univ, true_and]; omega
    rw [h2] at h1
    omega
  have hne : SA.Nonempty := by rw [← Finset.card_pos, hSAcard]; omega
  let δ : Fin n → ℕ := fun i => if (i : ℕ) < a then 0 else min ((i : ℕ) - (a - 1)) D
  refine (fun ⟨d, I, hI, _, hT⟩ => ⟨d, I, hI, hT⟩ :
    (∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I (pathW n) ε) → _) ?_
  refine Chain.lower_bound hm hL hμ hΔ hε A hA (pathW n) δ D hD1
    (fun i => by simp only [δ]; split_ifs <;> omega)
    (fun i j hij => by
      have := pathW_ne_zero hij
      simp only [δ]
      constructor <;> split_ifs <;> omega)
    SA SB (fun i hi => by simp only [SA, mem_filter] at hi; simp [δ, hi.2])
    (fun i hi => by
      simp only [SB, mem_filter] at hi
      simp only [δ]
      rw [if_neg (by omega)]
      omega)
    (by rw [hSAcard, hSBcard]) hne T ?_
  rw [hSAcard]
  have hn' : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha1
  have hr1 : L / (7 * μ) ≤ a * L / (n * μ) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have : (n : ℝ) ≤ 7 * a := by exact_mod_cast h7a
    have hLμ : 0 < L * μ := by positivity
    nlinarith [mul_le_mul_of_nonneg_left this hLμ.le]
  have hr2 : 16 * μ * Δ / (L * ε) ≤ 4 * n * μ * Δ / (a * L * ε) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have : (4 * a : ℝ) ≤ n := by exact_mod_cast h4a
    have hp : 0 < μ * Δ * L * ε := by positivity
    nlinarith
  have hs0 : 1 < Real.sqrt (1 + L / (7 * μ)) := Transfer.sqrt_gt_one (by positivity)
  have hsa : Real.sqrt (1 + L / (7 * μ)) ≤ Real.sqrt (1 + a * L / (n * μ)) :=
    Real.sqrt_le_sqrt (by linarith)
  have hΛ : Real.log (16 * μ * Δ / (L * ε)) ≤ Real.log (4 * n * μ * Δ / (a * L * ε)) :=
    Real.log_le_log (by positivity) hr2
  have hDn : (n : ℝ) / 2 ≤ D := by
    have : n ≤ 2 * D := by omega
    have : (n : ℝ) ≤ 2 * D := by exact_mod_cast this
    linarith
  exact Transfer.transfer (Transfer.ell_pos (lt_of_lt_of_le hs0 hsa)) (Transfer.ell_anti hs0 hsa)
    hΛ (by positivity) hDn (Nat.cast_nonneg T) hT

/-- Combined: for `ρ ≥ √(1 - 1/(2n²))`, the data terms and the network term together. -/
theorem minimax_ge_path_combined (hn : 4 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : Real.sqrt (1 - 1 / (2 * n ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (n / 2 * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2))) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  rcases le_max_iff.1 hT with h1 | h2
  · refine Chain.le_minimax_of _ fun A hA => ⟨pathW n, pathW_commClass (by omega) hρ, ?_⟩
    obtain ⟨d, I, hI, -, hT'⟩ :=
      Blocks.lower_bound_data (by omega) hm hL hμ hΔ hε A hA (pathW n) T h1
    exact ⟨d, I, hI, hT'⟩
  · exact minimax_ge_path hn hm hL hμ hΔ hε hρ T h2

end Path
end ULB63
