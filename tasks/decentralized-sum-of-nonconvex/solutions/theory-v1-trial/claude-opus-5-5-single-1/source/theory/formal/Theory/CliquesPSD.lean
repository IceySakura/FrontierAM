import Theory.CliquesCorollary

/-!
# The path of cliques is positive semidefinite

So the network lower bound also holds for the restricted communication class `W ⪰ 0`
(an alternative listed in `PROFILE.md` §3, and the assumption `0 ⪯ W ⪯ I` of Liu–Low).
-/

open MeasureTheory
open scoped ENNReal
open Finset

noncomputable section

namespace ULB63
namespace Cliques

open _root_.ULB63.Path

variable {n m c K : ℕ}

lemma edgeSum_le_four (U : ℕ → ℝ) (N : ℕ) : edgeSum U N ≤ 4 * ∑ k ∈ range N, U k ^ 2 := by
  rcases N with _ | M
  · simp [edgeSum]
  · unfold edgeSum
    simp only [Nat.add_sub_cancel]
    have hpt : ∀ k ∈ range M, (U k - U (k + 1)) ^ 2 ≤ 2 * U k ^ 2 + 2 * U (k + 1) ^ 2 :=
      fun k _ => by nlinarith [sq_nonneg (U k + U (k + 1))]
    refine (Finset.sum_le_sum hpt).trans ?_
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have h1 : ∑ k ∈ range M, U k ^ 2 ≤ ∑ k ∈ range (M + 1), U k ^ 2 := by
      rw [Finset.sum_range_succ]; linarith [sq_nonneg (U M)]
    have h2 : ∑ k ∈ range M, U (k + 1) ^ 2 ≤ ∑ k ∈ range (M + 1), U k ^ 2 := by
      rw [Finset.sum_range_succ' (fun k => U k ^ 2)]; linarith [sq_nonneg (U 0)]
    linarith

/-- The path-of-cliques gossip matrix is positive semidefinite. -/
theorem cqW_posSemidef (hK : 1 ≤ K) (hKn : K ≤ n) : (cqW n K).PosSemidef := by
  rw [Matrix.posSemidef_iff_dotProduct_mulVec]
  refine ⟨?_, fun v => ?_⟩
  · ext i j
    simp only [Matrix.conjTranspose_apply, star_trivial, cqW]
    exact coef_symm _ _
  · simp only [star_trivial]
    set s : ℕ := n / K with hs
    have hs1 : 1 ≤ s := s_pos hK hKn
    have hs' : (0 : ℝ) < s := by exact_mod_cast hs1
    have hθ : θ n K = (s : ℝ) / 4 := rfl
    set y := ybar K v with hy
    have hszpos : ∀ g ∈ range K, (0 : ℝ) < sz n K g := fun g hg => by
      exact_mod_cast sz_pos hK hKn (Finset.mem_range.1 hg)
    have hszge : ∀ g ∈ range K, (s : ℝ) ≤ sz n K g := fun g hg => by
      exact_mod_cast sz_ge hK (Finset.mem_range.1 hg)
    -- regroup the quadratic form by groups
    have h1 : v ⬝ᵥ ((cqW n K).mulVec v) =
        ∑ g ∈ range K, gsum K v g * (y g - θ n K / sz n K g * lap y K g) := by
      simp only [dotProduct]
      rw [Finset.sum_congr rfl fun i _ => by rw [cqW_mulVec hK hKn v i]]
      rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := range K) (g := grp n K)
        (fun i _ => Finset.mem_range.2 (grp_lt hK i))]
      refine Finset.sum_congr rfl fun g _ => ?_
      rw [gsum, Finset.sum_mul]
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [(Finset.mem_filter.1 hi).2]
    have h2 : ∑ g ∈ range K, gsum K v g * (y g - θ n K / sz n K g * lap y K g) =
        ∑ g ∈ range K, (sz n K g : ℝ) * y g ^ 2 - θ n K * ∑ g ∈ range K, y g * lap y K g := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun g hg => ?_
      rw [gsum_eq hK hKn v (Finset.mem_range.1 hg)]
      have := hszpos g hg
      field_simp
      ring
    rw [h1, h2, lap_inner]
    have h3 := edgeSum_le_four y K
    have h4 : ∑ k ∈ range K, y k ^ 2 ≤ (∑ g ∈ range K, (sz n K g : ℝ) * y g ^ 2) / s := by
      rw [le_div_iff₀ hs', Finset.sum_mul]
      refine Finset.sum_le_sum fun g hg => ?_
      have := hszge g hg
      nlinarith [sq_nonneg (y g)]
    have h5 : θ n K * edgeSum y K ≤ ∑ g ∈ range K, (sz n K g : ℝ) * y g ^ 2 := by
      rw [hθ]
      have h6 : (s : ℝ) / 4 * edgeSum y K ≤ (s : ℝ) / 4 * (4 * ∑ k ∈ range K, y k ^ 2) :=
        mul_le_mul_of_nonneg_left h3 (by positivity)
      have h7 : (s : ℝ) / 4 * (4 * ∑ k ∈ range K, y k ^ 2) ≤
          ∑ g ∈ range K, (sz n K g : ℝ) * y g ^ 2 := by
        have := mul_le_mul_of_nonneg_left h4 hs'.le
        rw [mul_div_cancel₀ _ hs'.ne'] at this
        linarith
      linarith
    linarith


lemma cliques_hard (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (K : ℕ) (hK : 8 ≤ K) (hKn : K ≤ n)
    (T : ℕ)
    (hT : (T : ℝ) ≤ ((K : ℝ) / 2 - 2) * (Real.log (16 * μ * Δ / (L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + L / (7 * μ)) + 1) / (Real.sqrt (1 + L / (7 * μ)) - 1))) - 2))
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I (cqW n K) ε := by
  classical
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


/-- **PSD version.** The lower bound of R6 (data terms and the network term) is attained with a
positive semidefinite gossip matrix of `𝓦_{n,ρ}` and an instance with convex components. -/
theorem lower_bound_psd (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
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
  refine ⟨cqW n K, cqW_commClass (by omega) hKn hρ, cqW_posSemidef (by omega) hKn, ?_⟩
  rcases le_max_iff.1 hT with h1 | h2
  · exact Blocks.lower_bound_data (by omega) hm hL hμ hΔ hε A hA (cqW n K) T h1
  · exact cliques_hard hm hL hμ hΔ hε K hK hKn T h2 A hA

end Cliques
end ULB63
