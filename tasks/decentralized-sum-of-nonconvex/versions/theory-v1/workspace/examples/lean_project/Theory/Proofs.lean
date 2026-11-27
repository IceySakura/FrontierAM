import Theory.Sanity

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof (no `sorry`, no new
axioms). Helper lemmas and other submission modules may be used, including the fixed `Theory/Model.lean` and
`Theory/Sanity.lean`; `Theory.Statements` may not be imported.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

theorem minimax_eq_zero_example {n m c : ℕ} (hn : 1 ≤ n) {L μ Δ ρ ε : ℝ} (hΔε : Δ ≤ ε) :
    minimax n m c L μ Δ ρ ε = 0 :=
  minimax_eq_zero_of_le hn hΔε

end ULB63
