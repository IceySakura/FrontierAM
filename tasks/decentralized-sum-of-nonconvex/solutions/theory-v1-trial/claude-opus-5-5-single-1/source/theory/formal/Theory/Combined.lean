import Theory.Corollaries
import Theory.LowerData3

/-!
# The combined minimax lower bound
-/

open MeasureTheory
open scoped ENNReal
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {n m c : ℕ}

theorem minimax_ge_combined (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (Real.log (8 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  rcases le_max_iff.1 hT with h1 | h2
  · refine le_minimax_of _ fun A hA => ⟨J n, J_commClass (by omega) hρ, ?_⟩
    obtain ⟨d, I, hI, -, hT'⟩ := Blocks.lower_bound_data (by omega) hm hL hμ hΔ hε A hA (J n) T h1
    exact ⟨d, I, hI, hT'⟩
  · exact minimax_ge_chain_J hn hm hL hμ hΔ hε hρ T h2

end Chain
end ULB63
