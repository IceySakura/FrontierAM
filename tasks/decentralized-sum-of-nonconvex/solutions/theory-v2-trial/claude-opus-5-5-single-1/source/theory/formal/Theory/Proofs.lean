import Theory.Sanity
import Theory.Main
import Theory.Unified

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof.
The constructions and lemmas live in `Theory/Chain.lean` (the separate-chains instance),
`Theory/Support.lean` (support propagation for admissible algorithms), `Theory/LowerBound.lean`
(the instance with convex components and the gap bounds), `Theory/StronglyConvex.lean` (the
instance with strongly convex components) and `Theory/Main.lean` (the final bounds) and `Theory/Unified.lean` (the bound in HLY form).
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

theorem lower_bound_transient {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε :=
  Chain.lower_bound_transient_SC_aux hn hm hμ hμL hΔ hε

theorem lower_bound_chain {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε :=
  Chain.lower_bound_chain_SC_aux hn hm hμ hΔ hε hq hq1 hqL

theorem lower_bound_small_kappa {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ < L) (hL8 : L - μ ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / (L - μ))⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε :=
  Chain.lower_bound_small_kappa_aux hn hm hμ hμL hL8 hΔ hε

theorem lower_bound_large_kappa {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL16 : 16 * n * m * μ ≤ L - μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 8 * Real.sqrt ((L - μ) / (n * m * μ)) *
            Real.log (4 * n * m * μ * Δ / ((L - μ) * ε))⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε :=
  Chain.lower_bound_large_kappa_aux hn hm hμ hL16 hΔ hε

theorem lower_bound_chain_convex {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε :=
  Chain.lower_bound_chain_aux hn hm hμ hμL hΔ hε hq hq1 hqL

theorem minimax_lower_bound {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    max ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞)
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  refine max_le (Chain.le_minimax_of hn hρ _ ?_) (Chain.le_minimax_of hn hρ _ ?_)
  · obtain ⟨d, I, hI, -, hT⟩ := Chain.lower_bound_transient_SC_aux (c := c) (ρ := ρ) hn hm hμ hμL hΔ hε
    exact ⟨d, I, hI, hT⟩
  · obtain ⟨d, I, hI, -, hT⟩ := Chain.lower_bound_chain_aux (c := c) (ρ := ρ) hn hm hμ hμL hΔ hε hq hq1 hqL
    exact ⟨d, I, hI, hT⟩

theorem lower_bound_chain_pathwise {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q) →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) :=
  Chain.lower_bound_chain_pathwise_aux hn hm hμ hΔ hε hq hq1 hqL

theorem lower_bound_small_kappa_convex {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hL8 : L ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / L)⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε :=
  Chain.lower_bound_small_kappa_convex_aux hn hm hμ hμL hL8 hΔ hε

theorem lower_bound_hly_form {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε :=
  Chain.lower_bound_hly_form_aux hn hm hμ hL2 hε hR hRκ

theorem minimax_lower_bound_hly_form {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) (hρ : 0 ≤ ρ) :
    ((⌈((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
        Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100⌉₊ : ℕ) : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  refine Chain.le_minimax_of hn hρ _ ?_
  obtain ⟨d, I, hI, -, hT⟩ :=
    Chain.lower_bound_hly_form_aux (c := c) (ρ := ρ) hn hm hμ hL2 hε hR hRκ
  exact ⟨d, I, hI, hT⟩

theorem lower_bound_hly_form_pathwise {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < ((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100 →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) :=
  Chain.lower_bound_hly_form_pathwise_aux hn hm hμ hL2 hε hR hRκ

end ULB63
