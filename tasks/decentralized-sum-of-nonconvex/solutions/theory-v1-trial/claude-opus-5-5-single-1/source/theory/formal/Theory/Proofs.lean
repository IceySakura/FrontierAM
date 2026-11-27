import Theory.Sanity
import Theory.CliquesPSD

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof.
The proofs live in the modules `Theory.ChainBasic`, `Theory.ChainValue`, `Theory.ChainInstance`,
`Theory.Support`, `Theory.LowerChain`, `Theory.Corollaries`.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

theorem lower_bound_network {n m c : ℕ} (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ)
    (W : Matrix (Fin n) (Fin n) ℝ) (δ : Fin n → ℕ) (D : ℕ) (hD : 1 ≤ D) (hδD : ∀ i, δ i ≤ D)
    (hδ : ∀ i j, W i j ≠ 0 → δ i ≤ δ j + 1 ∧ δ j ≤ δ i + 1)
    (SA SB : Finset (Fin n)) (hSA : ∀ i ∈ SA, δ i = 0) (hSB : ∀ i ∈ SB, δ i = D)
    (hcard : SA.card = SB.card) (hne : SA.Nonempty) (T : ℕ)
    (hT : (T : ℝ) ≤ D * (Real.log (4 * n * μ * Δ / (SA.card * L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + SA.card * L / (n * μ)) + 1) /
          (Real.sqrt (1 + SA.card * L / (n * μ)) - 1))) - 2)) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I W ε :=
  Chain.lower_bound hm hL hμ hΔ hε A hA W δ D hD hδD hδ SA SB hSA hSB hcard hne T hT

theorem minimax_ge_sqrt_kappa_log {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ}
    (hL : 0 < L) (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ Real.log (8 * μ * Δ / (L * ε)) /
      (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  Chain.minimax_ge_chain_J hn hm hL hμ hΔ hε hρ T hT

theorem lower_bound_data {n m c : ℕ} (hn : 1 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ)
    (W : Matrix (Fin n) (Fin n) ℝ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1)))) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I W ε :=
  Blocks.lower_bound_data hn hm hL hμ hΔ hε A hA W T hT

theorem minimax_ge_combined {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (Real.log (8 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  Chain.minimax_ge_combined hn hm hL hμ hΔ hε hρ T hT

theorem minimax_ge_path_combined {n m c : ℕ} (hn : 4 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ}
    (hL : 0 < L) (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε)
    (hρ : Real.sqrt (1 - 1 / (2 * n ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (n / 2 * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2))) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  Path.minimax_ge_path_combined hn hm hL hμ hΔ hε hρ T hT

theorem minimax_ge_all {n m c : ℕ} (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (K : ℕ) (hK : 8 ≤ K) (hKn : K ≤ n)
    (hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (max (Real.log (8 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2)
        (((K : ℝ) / 2 - 2) * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)))) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  Cliques.minimax_ge_all hm hL hμ hΔ hε K hK hKn hρ T hT

theorem minimax_ge_explicit_rho {n m c : ℕ} (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hK : 8 ≤ min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊) (T : ℕ)
    (hT : (T : ℝ) ≤ (((min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊ : ℕ) : ℝ) / 2 - 2) *
      (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  Cliques.minimax_ge_explicit_rho hm hL hμ hΔ hε hρ0 hρ1 hK T hT

theorem lower_bound_psd {n m c : ℕ} (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (K : ℕ) (hK : 8 ≤ K) (hKn : K ≤ n)
    (hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (((K : ℝ) / 2 - 2) * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)))
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) :
    ∃ W : Matrix (Fin n) (Fin n) ℝ, CommClass n ρ W ∧ W.PosSemidef ∧
      ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
        (T : ℕ∞) ≤ A.hittingTime I W ε :=
  Cliques.lower_bound_psd hm hL hμ hΔ hε K hK hKn hρ T hT A hA

end ULB63
