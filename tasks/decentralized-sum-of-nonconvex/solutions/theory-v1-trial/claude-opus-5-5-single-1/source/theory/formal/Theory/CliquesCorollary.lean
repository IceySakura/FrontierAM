import Theory.PathCorollary
import Theory.Cliques

/-!
# Network term for every `ρ`: the path of `K` cliques
-/

open MeasureTheory
open scoped ENNReal
open Finset

noncomputable section

namespace ULB63
namespace Cliques

variable {n m c : ℕ}

theorem minimax_ge_cliques (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (K : ℕ) (hK : 8 ≤ K) (hKn : K ≤ n)
    (hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ ((K : ℝ) / 2 - 2) * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  classical
  refine Chain.le_minimax_of _ fun A hA => ⟨cqW n K, cqW_commClass (by omega) hKn hρ, ?_⟩
  have hn : 8 ≤ n := le_trans hK hKn
  set a := n / 4 with ha
  have ha1 : 2 ≤ a := by omega
  have h4a : 4 * a ≤ n := by omega
  have h7a : n ≤ 7 * a := by omega
  set s := n / K with hs
  have hs1 : 1 ≤ s := (Nat.one_le_div_iff (by omega)).2 hKn
  have hsK : s * K ≤ n := Nat.div_mul_le_self n K
  have hnK : n < K * (s + 1) := Nat.lt_mul_div_succ n (by omega)
  let iA : Fin n := ⟨a - 1, by omega⟩
  let iB : Fin n := ⟨n - a, by omega⟩
  set gA := grp n K iA with hgA
  set gB := grp n K iB with hgB
  have hgAB : gA ≤ gB := grp_mono (by simp [iA, iB]; omega)
  have hgA_le : gA ≤ (a - 1) / s := min_le_left _ _
  have hsgA : s * gA ≤ a - 1 := by
    calc s * gA ≤ s * ((a - 1) / s) := Nat.mul_le_mul_left _ hgA_le
      _ ≤ a - 1 := Nat.mul_div_le (a - 1) s
  have hgB_cases : gB = (n - a) / s ∨ gB = K - 1 := by
    have : gB = min ((n - a) / s) (K - 1) := rfl
    rcases le_total ((n - a) / s) (K - 1) with h | h
    · left; rw [this, min_eq_left h]
    · right; rw [this, min_eq_right h]
  set D := gB - gA with hD
  -- the distance between the two node sets
  have hDK : (K : ℝ) < 2 * D + 4 := by
    have hDr : (D : ℝ) = gB - gA := by rw [hD, Nat.cast_sub hgAB]
    rw [hDr]
    have hsr : (1 : ℝ) ≤ s := by exact_mod_cast hs1
    have hsK' : (s : ℝ) * K ≤ n := by exact_mod_cast hsK
    have hnK' : (n : ℝ) < K * (s + 1) := by exact_mod_cast hnK
    have h4a' : (4 * a : ℝ) ≤ n := by exact_mod_cast h4a
    have hsgA' : (s : ℝ) * gA ≤ a - 1 := by
      have : s * gA ≤ a - 1 := hsgA
      have h' : ((s * gA : ℕ) : ℝ) ≤ ((a - 1 : ℕ) : ℝ) := by exact_mod_cast this
      push_cast [Nat.cast_sub (by omega : 1 ≤ a)] at h'
      exact h'
    rcases hgB_cases with h | h
    · have hlt : n - a < s * ((n - a) / s + 1) := Nat.lt_mul_div_succ (n - a) (by omega)
      rw [← h] at hlt
      have hlt' : (n : ℝ) - a < s * (gB + 1) := by
        have h' : ((n - a : ℕ) : ℝ) < ((s * (gB + 1) : ℕ) : ℝ) := by exact_mod_cast hlt
        push_cast [Nat.cast_sub (by omega : a ≤ n)] at h'
        exact h'
      by_contra hcon
      push_neg at hcon
      have := mul_le_mul_of_nonneg_left hcon (by positivity : (0 : ℝ) ≤ s)
      nlinarith
    · have hgB' : (gB : ℝ) = K - 1 := by
        rw [h]; push_cast [Nat.cast_sub (by omega : 1 ≤ K)]; ring
      rw [hgB']
      by_contra hcon
      push_neg at hcon
      have := mul_le_mul_of_nonneg_left hcon (by positivity : (0 : ℝ) ≤ s)
      nlinarith
  have hD1 : 1 ≤ D := by
    have : (8 : ℝ) ≤ K := by exact_mod_cast hK
    have : (1 : ℝ) < D := by linarith
    exact_mod_cast this.le
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
  let δ : Fin n → ℕ := fun i => min (grp n K i - gA) D
  refine (fun ⟨d, I, hI, _, hT⟩ => ⟨d, I, hI, hT⟩ :
    (∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I (cqW n K) ε) → _) ?_
  refine Chain.lower_bound hm hL hμ hΔ hε A hA (cqW n K) δ D hD1
    (fun i => min_le_right _ _)
    (fun i j hij => by
      have := cqW_ne_zero hij
      simp only [δ]
      constructor <;> omega)
    SA SB
    (fun i hi => by
      simp only [SA, mem_filter] at hi
      have : grp n K i ≤ gA := grp_mono (by simp [iA]; omega)
      simp only [δ]
      omega)
    (fun i hi => by
      simp only [SB, mem_filter] at hi
      have : gB ≤ grp n K i := grp_mono (by simp [iB]; omega)
      simp only [δ]
      omega)
    (by rw [hSAcard, hSBcard]) hne T ?_
  rw [hSAcard]
  have hn' : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have ha' : (0 : ℝ) < a := by exact_mod_cast (by omega : 0 < a)
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
  have hK8 : (8 : ℝ) ≤ K := by exact_mod_cast hK
  have hDn : (K : ℝ) / 2 - 2 ≤ D := by linarith
  exact Transfer.transfer (Transfer.ell_pos (lt_of_lt_of_le hs0 hsa)) (Transfer.ell_anti hs0 hsa)
    hΛ (by linarith) hDn (Nat.cast_nonneg T) hT

/-- Combined: data terms and the network term of the path of `K` cliques. -/
theorem minimax_ge_cliques_combined (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (K : ℕ) (hK : 8 ≤ K) (hKn : K ≤ n)
    (hρ : Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1))))
      (((K : ℝ) / 2 - 2) * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2))) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  rcases le_max_iff.1 hT with h1 | h2
  · refine Chain.le_minimax_of _ fun A hA =>
      ⟨cqW n K, cqW_commClass (by omega) hKn hρ, ?_⟩
    obtain ⟨d, I, hI, -, hT'⟩ :=
      Blocks.lower_bound_data (by omega) hm hL hμ hΔ hε A hA (cqW n K) T h1
    exact ⟨d, I, hI, hT'⟩
  · exact minimax_ge_cliques hm hL hμ hΔ hε K hK hKn hρ T h2

end Cliques
end ULB63

namespace ULB63
namespace Cliques

variable {n m c : ℕ}

/-- All four terms: the two data terms, the `√κ log` term, and the network term of the path of
`K` cliques (any `8 ≤ K ≤ n` with `ρ ≥ √(1 - 1/(4K²))`). -/
theorem minimax_ge_all (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
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
  rcases le_max_iff.1 hT with h1 | h2
  · exact minimax_ge_cliques_combined hm hL hμ hΔ hε K hK hKn hρ T (le_max_of_le_left h1)
  · rcases le_max_iff.1 h2 with h3 | h4
    · exact Chain.minimax_ge_chain_J (by omega) hm hL hμ hΔ hε
        ((Real.sqrt_nonneg _).trans hρ) T h3
    · exact minimax_ge_cliques_combined hm hL hμ hΔ hε K hK hKn hρ T (le_max_of_le_right h4)

end Cliques
end ULB63

namespace ULB63
namespace Cliques

variable {n m c : ℕ}

/-- The choice `K_ρ = min n ⌊1/(2√(1-ρ²))⌋` satisfies `ρ ≥ √(1 - 1/(4K_ρ²))`. -/
lemma rho_ok {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) {K : ℕ}
    (hK : K ≤ ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊) (hK1 : 1 ≤ K) :
    Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ ρ := by
  have hr : 0 < 1 - ρ ^ 2 := by nlinarith
  have hsq : 0 < Real.sqrt (1 - ρ ^ 2) := Real.sqrt_pos.2 hr
  have hK' : (K : ℝ) ≤ 1 / (2 * Real.sqrt (1 - ρ ^ 2)) :=
    le_trans (by exact_mod_cast hK) (Nat.floor_le (by positivity))
  have hK1' : (1 : ℝ) ≤ K := by exact_mod_cast hK1
  have h2 : 2 * K * Real.sqrt (1 - ρ ^ 2) ≤ 1 := by
    rw [le_div_iff₀ (by positivity)] at hK'
    linarith
  have h3 : 4 * (K : ℝ) ^ 2 * (1 - ρ ^ 2) ≤ 1 := by
    have h0 : 0 ≤ 2 * (K : ℝ) * Real.sqrt (1 - ρ ^ 2) := by positivity
    have h3' : (2 * (K : ℝ) * Real.sqrt (1 - ρ ^ 2)) ^ 2 ≤ 1 := by nlinarith
    have he : (2 * (K : ℝ) * Real.sqrt (1 - ρ ^ 2)) ^ 2 = 4 * (K : ℝ) ^ 2 * (1 - ρ ^ 2) := by
      rw [mul_pow, Real.sq_sqrt hr.le]; ring
    linarith
  have h4 : 1 - 1 / (4 * (K : ℝ) ^ 2) ≤ ρ ^ 2 := by
    have hpos : 0 < 4 * (K : ℝ) ^ 2 := by positivity
    have : 1 - ρ ^ 2 ≤ 1 / (4 * (K : ℝ) ^ 2) := by
      rw [le_div_iff₀ hpos]; linarith
    linarith
  calc Real.sqrt (1 - 1 / (4 * K ^ 2)) ≤ Real.sqrt (ρ ^ 2) := Real.sqrt_le_sqrt h4
    _ = ρ := Real.sqrt_sq hρ0

/-- R6 with the explicit choice `K_ρ = min n ⌊1/(2√(1-ρ²))⌋`. -/
theorem minimax_ge_explicit_rho (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hK : 8 ≤ min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊) (T : ℕ)
    (hT : (T : ℝ) ≤ (((min n ⌊1 / (2 * Real.sqrt (1 - ρ ^ 2))⌋₊ : ℕ) : ℝ) / 2 - 2) *
      (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2)) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε :=
  minimax_ge_cliques hm hL hμ hΔ hε _ hK (min_le_left _ _)
    (rho_ok hρ0 hρ1 (min_le_right _ _) (by omega)) T hT

end Cliques
end ULB63
