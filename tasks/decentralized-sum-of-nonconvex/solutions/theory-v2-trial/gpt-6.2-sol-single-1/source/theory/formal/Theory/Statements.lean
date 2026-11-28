import Theory.RelaxedLower
import Theory.TwoNodeUpper
import Theory.Disconnected

/-!
# Statement layer

The formal statements of the results claimed in `THEORY.md`, each with the proof `sorry`. Reviewers read this file
(and the submission modules it imports, such as shared definitions) to decide whether each statement says what
`THEORY.md` claims. Results about the problem are stated with the fixed model `Theory/Model.lean`: a lower bound
bounds `minimax` (or `Algorithm.hittingTime` for every admissible algorithm) from below, an upper bound exhibits an
admissible algorithm. `Theory/Proofs.lean` repeats every statement verbatim with its proof and must not import this
file.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

/-- R1: individually smooth block chains with a normalized objective-gap lower bound. -/
theorem minimax_data_no_accuracy_prefactor {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hρ : 0 ≤ ρ) (_hρ1 : ρ < 1) (hε : 0 < ε) (_hεΔ : ε < Δ) :
    let q := dataRatio (L / (5 * ((n : ℝ) * m))) μ
    (⌈max ((m : ℝ) * (1 - ε / Δ))
        ((m : ℝ) * Real.log (Δ / ε) / (2 * Real.log (1 / q)))⌉₊ : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  sorry

/-- R1, in the high-condition-number range in which the data acceleration term has square-root scaling. -/
theorem minimax_data_high_condition {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : 5 * ((n : ℝ) * m) * μ ≤ L) (hΔ : 0 < Δ)
    (hρ : 0 ≤ ρ) (hρ1 : ρ < 1) (hε : 0 < ε) (hεΔ : ε ≤ Δ / 2) :
    (⌈(m : ℝ) / 4 +
        (m : ℝ) * Real.sqrt (L / (μ * ((n : ℝ) * m))) * Real.log (Δ / ε) / 20⌉₊ : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  sorry

/-- R2: lower bound with a local-average smoothness bound and a separate component inflation cap. -/
theorem local_average_data_lower {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε θ : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hθ1 : 1 ≤ θ) (hθm : θ ≤ (m : ℝ)) (hε : 0 < ε) (_hεΔ : ε < Δ) :
    let q := dataRatio (θ * L / (5 * ((n : ℝ) * m))) μ
    let B := ⌈max ((m : ℝ) * (1 - ε / Δ))
      ((m : ℝ) * Real.log (Δ / ε) / (2 * Real.log (1 / q)))⌉₊
    ∃ (d : ℕ) (I : Instance n m d), I.InClass (θ * L) μ Δ ∧
      (∀ i, LSmooth L (I.localAverage i)) ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ ρ →
        ∀ W : Matrix (Fin n) (Fin n) ℝ, (B : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

/-- R2: no H-UB order with the local-average bound substituted for the individual one,
even up to arbitrarily many fixed powers of logarithms and on the exact averaging network. -/
theorem local_average_only_breaks_hub (c : ℕ) (C : ℝ) (hC : 0 < C) (s : ℕ) :
    ∃ (m : ℕ), 10 ≤ m ∧
      ∃ (d : ℕ) (I : Instance 2 m d), I.InClass ((m : ℝ)^2) 1 1 ∧
        (∀ i, LSmooth (m : ℝ) (I.localAverage i)) ∧
        ∀ (A : Algorithm 2 m c), A.Admissible (m : ℝ) 1 1 0 →
          (⌈C * (m : ℝ)^(5/4 : ℝ) * (1 + Real.log m)^s⌉₊ : ℕ∞) <
            A.hittingTime I (J 2) (1 / (m : ℝ)) := by
  sorry

/-- R3: a single deterministic algorithm for the two-node local-average-only class. -/
theorem two_node_local_average_upper {m : ℕ} (hm : 1 ≤ m)
    {L μ Δ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ A : Algorithm 2 m 1, (∀ M ρ, A.Admissible M μ Δ ρ) ∧
      ∀ (d : ℕ) (I : Instance 2 m d), I.InLocalClass L μ Δ →
        ∀ W : Matrix (Fin 2) (Fin 2) ℝ, W 0 1 ≠ 0 → W 1 0 ≠ 0 →
          A.hittingTime I W ε ≤
            (((m+2) * ⌈Real.sqrt (16*(L/μ))⌉₊ * ⌈Real.log (Δ/ε)/Real.log 2⌉₊ : ℕ) : ℕ∞) := by
  sorry

/-- R3 lower bound in the very same local-average-only class, even on `J`. -/
theorem two_node_local_average_lower {m c : ℕ} (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε)
    (_hεΔ : ε < Δ) :
    let q := dataRatio (L/10) μ
    ∃ (d : ℕ) (I : Instance 2 m d), I.InLocalClass L μ Δ ∧
      ∀ (A : Algorithm 2 m c), A.Admissible L μ Δ ρ →
        (⌈max ((m : ℝ) * (1 - ε/Δ)) ((m : ℝ) * Real.log (Δ/ε) / (2*Real.log (1/q)))⌉₊ : ℕ∞) ≤
          A.hittingTime I (J 2) ε := by
  sorry

/-- R3 specialized to the standard class at two nodes, uniformly in the spectral gap. -/
theorem two_node_standard_upper {m : ℕ} (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : ρ < 1) :
    minimax 2 m 1 L μ Δ ρ ε ≤
      (((m+2) * ⌈Real.sqrt (16*(L/μ))⌉₊ * ⌈Real.log (Δ/ε)/Real.log 2⌉₊ : ℕ) : ℕ∞) := by
  sorry

/-- R4: allowing a disconnected network can make the averaged-output target impossible. -/
theorem minimax_infinite_at_disconnected_endpoint {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    let l := ((n : ℝ) - 1) * L / (2 * (n : ℝ))
    ε < Δ * (l / (μ + l))^2 → minimax n m c L μ Δ 1 ε = ⊤ := by
  sorry

end ULB63
