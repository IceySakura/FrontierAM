import Theory.Model

/-!
# Statement layer

The formal statements of the results claimed in `THEORY.md`, each with the proof `sorry`. Reviewers read this file
(and the submission modules it imports, such as shared definitions) to decide whether each statement says what
`THEORY.md` claims. Results about the problem are stated with the fixed model `Theory/Model.lean`: a lower bound
bounds `minimax` (or `Algorithm.hittingTime` for every admissible algorithm) from below, an upper bound exhibits an
admissible algorithm. `Theory/Proofs.lean` repeats every statement verbatim with its proof and must not import this
file. The entry below only shows the layout; replace it.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

/-- Example: when `ε ≥ Δ` the initial point already meets the target, so the minimax time is `0`. -/
theorem minimax_eq_zero_example {n m c : ℕ} (hn : 1 ≤ n) {L μ Δ ρ ε : ℝ} (hΔε : Δ ≤ ε) :
    minimax n m c L μ Δ ρ ε = 0 := by
  sorry

end ULB63
