import Theory.Model

/-!
# Statement layer

The formal statements of the results R1–R11 claimed in `THEORY.md` (table in §2, proofs in §4),
each with the proof `sorry`;
`Theory/Proofs.lean` repeats them verbatim with complete proofs. All of them are lower bounds stated
with the fixed model `Theory/Model.lean` only.

Common shape: there is one instance `I` of `𝓕_{L,μ,Δ,m}` (`I.InClass L μ Δ`, in a dimension `d` of
our choice) whose components have an extra property (convex, or `μ`-strongly convex), such that for
**every** admissible algorithm `A` (`A.Admissible L μ Δ ρ`: randomized, zero-respecting, adaptive
component selection, at most one component gradient and one prox per node and iteration, any message
width `c`) and **every** mixing matrix `W`, the hitting time `T_ε = A.hittingTime I W ε` is at least
the stated bound. The instance does not depend on `A` or `W`. R6 and R10 bound `minimax` itself,
and R7 and R11 are pathwise (every seed) forms. Notation in the docstrings: `N = n m`, `κ = L/μ`.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

/-- **R1 (transient data term).** With `μ`-strongly convex components:
`T_ε ≥ ⌈m (1 - ε/Δ)⌉` for every admissible algorithm and every `W`. -/
theorem lower_bound_transient {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

/-- **R2 (separate-chains data term, general form).** For every `q ∈ (0,1)` with
`4 N μ q ≤ (L - μ)(1 - q)²`, with `μ`-strongly convex components:
`T_ε ≥ ⌈(m/2) · log((1-q)² Δ/ε) / log(1/q)⌉` for every admissible algorithm and every `W`. -/
theorem lower_bound_chain {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  sorry

/-- **R3 (regime `0 < κ - 1 ≤ 8N`).** With `μ`-strongly convex components:
`T_ε ≥ ⌈(m/2) · log(Δ/(4ε)) / log(16 N/(κ - 1))⌉` for every admissible algorithm and every `W`. -/
theorem lower_bound_small_kappa {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ < L) (hL8 : L - μ ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / (L - μ))⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε := by
  sorry

/-- **R4 (regime `κ - 1 ≥ 16N`).** With `μ`-strongly convex components:
`T_ε ≥ ⌈(m/8) · √((κ - 1)/N) · log(4 N Δ/((κ - 1) ε))⌉` for every admissible algorithm and every
`W`. -/
theorem lower_bound_large_kappa {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL16 : 16 * n * m * μ ≤ L - μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 8 * Real.sqrt ((L - μ) / (n * m * μ)) *
            Real.log (4 * n * m * μ * Δ / ((L - μ) * ε))⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

/-- **R5 (convex components, `κ` down to `1`).** For every `q ∈ (0,1)` with
`4 N μ q ≤ L (1 - q)²`, with convex components (the strong convexity sits in `h = (μ/2)‖·‖²`):
`T_ε ≥ ⌈(m/2) · log((1-q)² Δ/ε) / log(1/q)⌉` for every admissible algorithm and every `W`. -/
theorem lower_bound_chain_convex {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  sorry

/-- **R6 (the minimax quantity).** For `ρ ≥ 0` and every `q ∈ (0,1)` with `4 N μ q ≤ L (1 - q)²`:
`minimax ≥ max(⌈m (1 - ε/Δ)⌉, ⌈(m/2) · log((1-q)² Δ/ε) / log(1/q)⌉)`. -/
theorem minimax_lower_bound {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    max ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞)
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R7 (pathwise form of R2).** Same hypotheses and instance class as R2. For every admissible
algorithm, every `W`, every seed `ω` and every iteration `t < (m/2) · log((1-q)² Δ/ε) / log(1/q)`,
the averaged output `x̂_t` has gap `φ(x̂_t) - φ⋆ > ε` (so the bound also holds for
success-probability criteria, not only in expectation). -/
theorem lower_bound_chain_pathwise {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q) →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) := by
  sorry

/-- **R8 (regime `1 ≤ κ ≤ 8N`, convex components).** With convex components (the strong convexity
sits in `h = (μ/2)‖·‖²`): `T_ε ≥ ⌈(m/2) · log(Δ/(4ε)) / log(16 N/κ)⌉` for every admissible
algorithm and every `W`. -/
theorem lower_bound_small_kappa_convex {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hL8 : L ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / L)⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε := by
  sorry

/-- **R9 (the Hannah–Liu–O'Connor–Yin form, all `κ ≥ 2`).** If `L ≥ 2μ`, `Δ/ε ≥ 16` and
`Δ/ε ≥ ((κ - 1)/(4N))²`, then on one instance with `μ`-strongly convex components, for every
admissible algorithm and every `W`,
`T_ε ≥ ⌈(m + (m/(1 + (ln(N/κ))₊) + √(m(κ - 1)/n)) · ln(Δ/ε)) / 100⌉`. -/
theorem lower_bound_hly_form {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  sorry

/-- **R10 (the minimax quantity in Hannah–Liu–O'Connor–Yin form).** Under the hypotheses of R9 and
`ρ ≥ 0`: `minimax ≥ ⌈(m + (m/(1 + (ln(N/κ))₊) + √(m(κ - 1)/n)) · ln(Δ/ε)) / 100⌉`. -/
theorem minimax_lower_bound_hly_form {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) (hρ : 0 ≤ ρ) :
    ((⌈((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
        Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100⌉₊ : ℕ) : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R11 (pathwise form of R9).** Under the hypotheses of R9, on one instance with `μ`-strongly
convex components: for every admissible algorithm, every `W`, every seed `ω` and every iteration
`t < (m + (m/(1 + (ln(N/κ))₊) + √(m(κ - 1)/n)) · ln(Δ/ε)) / 100`, the averaged output has gap
`φ(x̂_t) - φ⋆ > ε`. -/
theorem lower_bound_hly_form_pathwise {n m c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < ((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100 →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) := by
  sorry

end ULB63
