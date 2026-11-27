import Theory.LowerChain
import Theory.Transfer

/-!
# Corollaries: the complete graph `W = J`
-/

open MeasureTheory
open scoped ENNReal
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {n m c : ℕ}

/-- `minimax ≥ T` follows from a hard instance for every admissible algorithm with a matrix of the
class. -/
lemma le_minimax_of {L μ Δ ρ ε : ℝ} (T : ℕ∞)
    (h : ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ →
      ∃ W, CommClass n ρ W ∧ ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ T ≤ A.hittingTime I W ε) :
    T ≤ minimax n m c L μ Δ ρ ε := by
  unfold minimax
  refine le_iInf₂ fun A hA => ?_
  obtain ⟨W, hW, d, I, hI, hT⟩ := h A hA
  exact le_iSup_of_le d (le_iSup_of_le I (le_iSup_of_le hI (le_iSup_of_le W
    (le_iSup_of_le hW hT))))

theorem minimax_ge_chain_J (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ Real.log (8 * μ * Δ / (L * ε)) /
      (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  classical
  refine le_minimax_of _ fun A hA => ⟨J n, J_commClass (by omega) hρ, ?_⟩
  set a := n / 2 with ha
  have ha1 : 1 ≤ a := by omega
  have h2a : 2 * a ≤ n := by omega
  have h3a : n ≤ 3 * a := by omega
  let SA : Finset (Fin n) := univ.filter (fun i => (i : ℕ) < a)
  let SB : Finset (Fin n) := univ.filter (fun i => a ≤ (i : ℕ) ∧ (i : ℕ) < 2 * a)
  have hSAcard : SA.card = a := by
    simp only [SA, Fin.card_filter_val_lt]; omega
  have hSBcard : SB.card = a := by
    have hsplit : (univ.filter (fun i : Fin n => (i : ℕ) < 2 * a)) =
        SA ∪ SB := by
      ext i; simp only [SA, SB, mem_filter, mem_univ, true_and, mem_union]; omega
    have hdisj : Disjoint SA SB := by
      rw [Finset.disjoint_left]
      intro i hi hi'
      simp only [SA, SB, mem_filter, mem_univ, true_and] at hi hi'
      omega
    have := Finset.card_union_of_disjoint hdisj
    rw [← hsplit, Fin.card_filter_val_lt, hSAcard] at this
    omega
  have hne : SA.Nonempty := by
    rw [← Finset.card_pos, hSAcard]; omega
  refine (fun ⟨d, I, hI, _, hT⟩ => ⟨d, I, hI, hT⟩ :
    (∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I (J n) ε) → _) ?_
  refine lower_bound hm hL hμ hΔ hε A hA (J n) (fun i => if (i : ℕ) < a then 0 else 1) 1 le_rfl
    (fun i => by split_ifs <;> omega) (fun i j _ => by constructor <;> split_ifs <;> omega)
    SA SB (fun i hi => by simp only [SA, mem_filter] at hi; simp [hi.2])
    (fun i hi => by simp only [SB, mem_filter] at hi; simp [show ¬ (i : ℕ) < a by omega])
    (by rw [hSAcard, hSBcard]) hne T ?_
  rw [hSAcard]
  -- monotonicity in the ratio `a / n ∈ [1/3, 1/2]`
  have hn' : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha1
  have hr1 : L / (3 * μ) ≤ a * L / (n * μ) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have : (n : ℝ) ≤ 3 * a := by exact_mod_cast h3a
    have hLμ : 0 < L * μ := by positivity
    nlinarith [mul_le_mul_of_nonneg_left this hLμ.le]
  have hr2 : 8 * μ * Δ / (L * ε) ≤ 4 * n * μ * Δ / (a * L * ε) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have : (2 * a : ℝ) ≤ n := by exact_mod_cast h2a
    have hp : 0 < μ * Δ * L * ε := by positivity
    nlinarith
  have hs0 : 1 < Real.sqrt (1 + L / (3 * μ)) := Transfer.sqrt_gt_one (by positivity)
  have hsa : Real.sqrt (1 + L / (3 * μ)) ≤ Real.sqrt (1 + a * L / (n * μ)) :=
    Real.sqrt_le_sqrt (by linarith)
  have hΛ : Real.log (8 * μ * Δ / (L * ε)) ≤ Real.log (4 * n * μ * Δ / (a * L * ε)) :=
    Real.log_le_log (by positivity) hr2
  have := Transfer.transfer (Transfer.ell_pos (lt_of_lt_of_le hs0 hsa))
    (Transfer.ell_anti hs0 hsa) hΛ one_pos le_rfl (Nat.cast_nonneg T) (by linarith)
  simpa using this

end Chain
end ULB63
