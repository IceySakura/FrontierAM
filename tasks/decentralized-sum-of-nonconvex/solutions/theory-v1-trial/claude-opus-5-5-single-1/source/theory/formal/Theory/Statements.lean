import Theory.Model

/-!
# Statement layer

Formal statements of the results of `THEORY.md` (proofs: `Theory/Proofs.lean`). Everything is
stated with the fixed model `Theory/Model.lean` only. All results are lower bounds.

Reading guide (`κ = L/μ` throughout):
* `(T : ℕ∞) ≤ A.hittingTime I W ε` means `T_ε ≥ T`, i.e. `𝔼[φ(x̂_t) - φ⋆] > ε` for all `t < T`;
* `(T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε` means that every admissible algorithm needs at least `T`
  iterations on some instance of the class and some `W ∈ 𝓦_{n,ρ}`; each theorem gives this for
  every natural `T` below an explicit real expression;
* `Real.log ((Real.sqrt (1 + x) + 1) / (Real.sqrt (1 + x) - 1))` is `ℓ(x) = log (1/q)` for the decay
  rate `q` of the hard chain; `ℓ(x) ≤ 2/(√(1+x) - 1)`, so `log(·)/ℓ(x) ≳ √x · log(·)` for `x ≥ 1`;
* the message width `c` is arbitrary in every statement.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

/-- **R1 (network-aware chain lower bound, every gossip matrix).**
Let `W` be any matrix and `δ : Fin n → ℕ` a potential that changes by at most `1` along every
edge of `W` (`W i j ≠ 0`), with `δ ≤ D`, `δ = 0` on the node set `SA` and `δ = D` on `SB`
(so `SA` and `SB` are at graph distance `≥ D` in the support graph of `W`; for instance
`δ i = min (dist_W(i, SA)) D`). Then for every admissible algorithm there is an instance of the
class `𝓕_{L,μ,Δ,m}`, all of whose components are convex, on which the algorithm, run over `W`,
needs at least `T` iterations (`T ≤ T_ε`) whenever
`T ≤ D · (log (4 n μ Δ / (|SA| L ε)) / (2 log ((s+1)/(s-1))) - 2)` with `s = √(1 + |SA| L / (n μ))`. -/
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
      (T : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

/-- **R2 (minimax lower bound, `√κ · log` term).** For `n ≥ 2`, `m ≥ 1`, `ρ ≥ 0` and every `T`
with `T ≤ log (8 Δ / (κ ε)) / (2 log ((s+1)/(s-1))) - 2`, `s = √(1 + κ/3)` (where `κ = L/μ`), the
minimax iteration complexity is at least `T`. -/
theorem minimax_ge_sqrt_kappa_log {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ}
    (hL : 0 < L) (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ Real.log (8 * μ * Δ / (L * ε)) /
      (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R3 (data lower bound, every gossip matrix).** For every admissible algorithm and every
matrix `W` there is an instance of `𝓕_{L,μ,Δ,m}`, all of whose components are convex, on which
the algorithm needs at least `T` iterations (`T ≤ T_ε`) whenever `T` is at most the larger of
`(m/5) · (1 - (1 + κ/(2nm)) ε/Δ)` and
`(m/10) · log (Δ / ((1 + κ/(2nm)) ε)) / log ((s+1)/(s-1))` with `s = √(1 + κ/(nm))`, where
`κ = L/μ`. -/
theorem lower_bound_data {n m c : ℕ} (hn : 1 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ)
    (W : Matrix (Fin n) (Fin n) ℝ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1)))) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

/-- **R4 (combined minimax lower bound).** For `n ≥ 2`, `m ≥ 1`, `ρ ≥ 0`: the minimax iteration
complexity is at least every `T` bounded by the largest of the three terms
`(m/5) (1 - (1 + κ/(2nm)) ε/Δ)`, `(m/10) log (Δ/((1 + κ/(2nm)) ε)) / log ((s+1)/(s-1))`
(`s = √(1 + κ/(nm))`) and `log (8 Δ/(κ ε)) / (2 log ((r+1)/(r-1))) - 2` (`r = √(1 + κ/3)`). -/
theorem minimax_ge_combined {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : 0 ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (Real.log (8 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (3 * μ)) + 1) / (Real.sqrt (1 + L / (3 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R5 (minimax lower bound with the network term).** For `n ≥ 4`, `m ≥ 1` and
`ρ ≥ √(1 - 1/(2n²))`: the minimax iteration complexity is at least every `T` bounded by the
largest of `(m/5) (1 - (1 + κ/(2nm)) ε/Δ)`, `(m/10) log (Δ/((1 + κ/(2nm)) ε)) / log ((s+1)/(s-1))`
(`s = √(1 + κ/(nm))`) and `(n/2) (log (16 Δ/(κ ε)) / (2 log ((r+1)/(r-1))) - 2)` (`r = √(1 + κ/7)`). -/
theorem minimax_ge_path_combined {n m c : ℕ} (hn : 4 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ}
    (hL : 0 < L) (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε)
    (hρ : Real.sqrt (1 - 1 / (2 * n ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (n / 2 * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2))) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R6 (minimax lower bound for every `ρ`: data, `√κ log` and network terms).** Let `m ≥ 1`
and let `K` be any integer with `8 ≤ K ≤ n` and `ρ ≥ √(1 - 1/(4K²))` (for instance the largest
`K ≤ n` with `K ≤ 1/(2√(1-ρ²))`). Then the minimax iteration complexity is at least every `T`
bounded by the largest of `(m/5) (1 - (1 + κ/(2nm)) ε/Δ)`,
`(m/10) log (Δ/((1 + κ/(2nm)) ε)) / log ((s+1)/(s-1))` (`s = √(1 + κ/(nm))`),
`log (8Δ/(κε)) / (2 log ((r+1)/(r-1))) - 2` (`r = √(1 + κ/3)`) and
`(K/2 - 2) (log (16 Δ/(κ ε)) / (2 log ((r'+1)/(r'-1))) - 2)` (`r' = √(1 + κ/7)`). -/
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
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R7 (the network term with explicit `ρ`).** For `0 ≤ ρ < 1` put
`K_ρ = min n ⌊1/(2√(1-ρ²))⌋` (so `K_ρ ≍ min {n, (1-ρ)^{-1/2}}`). If `K_ρ ≥ 8`, the minimax iteration
complexity is at least every `T ≤ (K_ρ/2 - 2) (log (16 Δ/(κ ε)) / (2 log ((r+1)/(r-1))) - 2)`,
`r = √(1 + κ/7)`. -/
theorem minimax_ge_explicit_rho {n m c : ℕ} (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L)
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hK : 8 ≤ min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊) (T : ℕ)
    (hT : (T : ℝ) ≤ (((min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊ : ℕ) : ℝ) / 2 - 2) *
      (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  sorry

/-- **R8 (the same bound under the restrictions `W ⪰ 0` and convex components).** With `K` as in
R6, for every admissible algorithm there are a positive semidefinite matrix `W ∈ 𝓦_{n,ρ}` and an
instance of `𝓕_{L,μ,Δ,m}` with convex components on which the algorithm needs at least `T`
iterations, for every `T` bounded by the largest of the two data terms of R3 and the network term
`(K/2 - 2) (log (16 Δ/(κ ε)) / (2 log ((r+1)/(r-1))) - 2)`, `r = √(1 + κ/7)`. -/
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
        (T : ℕ∞) ≤ A.hittingTime I W ε := by
  sorry

end ULB63
