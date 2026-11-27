import Theory.ChainInstance

/-!
# The chain lower bound (network-aware zero-chain argument)
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {n m c d : ℕ}

/-- Raise rule of the chain instance. -/
def raise (SA SB : Finset (Fin n)) (i : Fin n) (j : ℕ) : ℕ :=
  if (i ∈ SA ∧ j % 2 = 0) ∨ (i ∈ SB ∧ j % 2 = 1) then 1 else 0

lemma chainInst_grad_supp (SA SB : Finset (Fin n)) (L μ R q : ℝ) (i : Fin n) (ξ : Fin m)
    (x : E d) (j : ℕ) (hx : supp x ⊆ Pre d j) :
    supp (gradient ((chainInst n m d SA SB L μ R q).F i ξ) x) ⊆ Pre d (j + raise SA SB i j) := by
  simp only [chainInst]
  rw [gradient_quad, supp_subset_Pre]
  intro p hp
  have hx' : ∀ k, j ≤ k → cx x k = 0 := by
    intro k hk
    by_cases hkd : k < d
    · simp only [cx, dif_pos hkd]
      exact supp_subset_Pre.1 hx ⟨k, hkd⟩ hk
    · exact cx_of_ge (by omega)
  refine quadGrad_u_vanish _ _ _ _ (fun k hk => by simp [cch, hk]) j x hx' p ?_
  have : (if j ∈ nodeS SA SB d i then 1 else 0) ≤ raise SA SB i j := by
    by_cases hmem : j ∈ nodeS SA SB d i
    · rw [if_pos hmem]
      unfold nodeS at hmem
      unfold raise
      by_cases hA : i ∈ SA
      · rw [if_pos hA] at hmem
        have hj : j % 2 = 0 := (Finset.mem_filter.1 hmem).2
        rw [if_pos (Or.inl ⟨hA, hj⟩)]
      · rw [if_neg hA] at hmem
        by_cases hB : i ∈ SB
        · rw [if_pos hB] at hmem
          have hj : ¬ j % 2 = 0 := (Finset.mem_filter.1 hmem).2
          rw [if_pos (Or.inr ⟨hB, by omega⟩)]
        · rw [if_neg hB] at hmem
          exact absurd hmem (Finset.notMem_empty _)
    · rw [if_neg hmem]; exact Nat.zero_le _
  omega

lemma chainInst_prox_supp (SA SB : Finset (Fin n)) (L μ R q : ℝ) (v : E d) (η : ℝ) (j : ℕ)
    (hv : supp v ⊆ Pre d j) : supp ((chainInst n m d SA SB L μ R q).prox v η) ⊆ Pre d j := by
  intro p hp
  apply hv
  simp only [chainInst, supp, Set.mem_setOf_eq, PiLp.smul_apply, smul_eq_mul] at hp ⊢
  exact right_ne_zero_of_mul hp

/-- Real-variable core: `ε < G₀ q^{2K+2}` under the hypothesis on `T`, with `q = (s-1)/(s+1)`. -/
lemma key_ineq {G0 ε s : ℝ} (hG0 : 0 < G0) (hε : 0 < ε) (hs1 : 1 < s) {D t T K : ℕ} (hD : 1 ≤ D)
    (htT : t < T) (hK : K ≤ t / D + 1)
    (hT : (T : ℝ) ≤ D * (Real.log (G0 / ε) / (2 * Real.log ((s + 1) / (s - 1))) - 2)) :
    ε < G0 * ((s - 1) / (s + 1)) ^ (2 * K + 2) := by
  set q := (s - 1) / (s + 1) with hq
  have hq0 : 0 < q := div_pos (by linarith) (by linarith)
  have hs0 : 0 < s - 1 := by linarith
  set ℓ := Real.log ((s + 1) / (s - 1)) with hℓ
  have hℓ0 : 0 < ℓ := by
    apply Real.log_pos
    rw [one_lt_div hs0]; linarith
  have hD' : (0 : ℝ) < D := by exact_mod_cast hD
  have htD : ((t / D : ℕ) : ℝ) ≤ (t : ℝ) / D := Nat.cast_div_le
  have htT' : (t : ℝ) + 1 ≤ T := by exact_mod_cast htT
  have hKr : ((2 * K + 2 : ℕ) : ℝ) ≤ 2 * ((t : ℝ) / D) + 4 := by
    have : (K : ℝ) ≤ ((t / D : ℕ) : ℝ) + 1 := by exact_mod_cast hK
    push_cast; linarith
  -- `(2t/D + 4) ℓ < log (G0/ε)`
  have hmain : (2 * ((t : ℝ) / D) + 4) * ℓ < Real.log (G0 / ε) := by
    have h1 : (t : ℝ) / D < Real.log (G0 / ε) / (2 * ℓ) - 2 := by
      rw [div_lt_iff₀ hD']
      nlinarith
    have h2 : 2 * ((t : ℝ) / D) + 4 < Real.log (G0 / ε) / ℓ := by
      have : Real.log (G0 / ε) / (2 * ℓ) = Real.log (G0 / ε) / ℓ / 2 := by
        field_simp
      linarith
    rwa [lt_div_iff₀ hℓ0] at h2
  have hNlog : ((2 * K + 2 : ℕ) : ℝ) * ℓ < Real.log (G0 / ε) :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right hKr hℓ0.le) hmain
  have hpow : Real.log (q ^ (2 * K + 2)) = -(((2 * K + 2 : ℕ) : ℝ) * ℓ) := by
    rw [Real.log_pow, hq, hℓ, show (s - 1) / (s + 1) = ((s + 1) / (s - 1))⁻¹ by rw [inv_div],
      Real.log_inv]
    ring
  have hlt : Real.log (ε / G0) < Real.log (q ^ (2 * K + 2)) := by
    rw [hpow, Real.log_div hε.ne' hG0.ne']
    rw [Real.log_div hG0.ne' hε.ne'] at hNlog
    linarith
  have := (Real.log_lt_log_iff (by positivity) (by positivity)).1 hlt
  rw [div_lt_iff₀ hG0] at this
  linarith

/-- **Chain lower bound for an arbitrary gossip matrix.** -/
theorem lower_bound (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ) (hΔ : 0 < Δ)
    (hε : 0 < ε) (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ)
    (W : Matrix (Fin n) (Fin n) ℝ) (δ : Fin n → ℕ) (D : ℕ) (hD : 1 ≤ D) (hδD : ∀ i, δ i ≤ D)
    (hδ : ∀ i j, W i j ≠ 0 → δ i ≤ δ j + 1 ∧ δ j ≤ δ i + 1)
    (SA SB : Finset (Fin n)) (hSA : ∀ i ∈ SA, δ i = 0) (hSB : ∀ i ∈ SB, δ i = D)
    (hcard : SA.card = SB.card) (hne : SA.Nonempty) (T : ℕ)
    (hT : (T : ℝ) ≤ D * (Real.log (4 * n * μ * Δ / (SA.card * L * ε)) /
        (2 * Real.log ((Real.sqrt (1 + SA.card * L / (n * μ)) + 1) /
          (Real.sqrt (1 + SA.card * L / (n * μ)) - 1))) - 2)) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I W ε := by
  classical
  have hdisj : Disjoint SA SB := by
    rw [Finset.disjoint_left]
    intro i hiA hiB
    have h1 := hSA i hiA
    have h2 := hSB i hiB
    omega
  have hcardpos : 0 < SA.card := Finset.card_pos.2 hne
  have hn : 0 < n := by
    have : SA.card ≤ n := (Finset.card_le_univ SA).trans (by simp)
    omega
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have ha0 : (0 : ℝ) < SA.card := by exact_mod_cast hcardpos
  set α : ℝ := SA.card * (L / 4) / n with hα
  have hα0 : 0 < α := by positivity
  set s : ℝ := Real.sqrt (1 + SA.card * L / (n * μ)) with hs
  have hs2 : s ^ 2 = 1 + 4 * α / μ := by
    rw [hs, Real.sq_sqrt (by positivity), hα]
    field_simp
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hs1 : 1 < s := by
    have h4 : 0 < 4 * α / μ := by positivity
    have : 1 < s ^ 2 := by rw [hs2]; linarith
    nlinarith
  set q : ℝ := (s - 1) / (s + 1) with hq
  have hq0 : 0 < q := div_pos (by linarith) (by linarith)
  have hq1 : q < 1 := by rw [hq, div_lt_one (by linarith)]; linarith
  have hqeq : μ * q = α * (1 - q) ^ 2 := by
    have h1q : 1 - q = 2 / (s + 1) := by rw [hq]; field_simp; ring
    rw [h1q, hq]
    have hsp : 0 < s + 1 := by linarith
    have hα' : α = μ * (s ^ 2 - 1) / 4 := by
      rw [hs2]; field_simp; ring
    rw [hα']
    field_simp
    ring
  set R : ℝ := Real.sqrt (2 * Δ / α) with hR
  have hR2 : R ^ 2 = 2 * Δ / α := Real.sq_sqrt (by positivity)
  have hRΔ : α / 2 * R ^ 2 ≤ Δ := by
    rw [hR2]; field_simp; exact le_refl _
  set d := T + 2 with hd
  have hd1 : 1 ≤ d := by omega
  set I := chainInst n m d SA SB L μ R q with hI
  refine ⟨d, I, chainInst_inClass hm hdisj hcard hd1 hL.le hμ hq0.le hq1.le hqeq hRΔ,
    fun i _ => quad_convex _ (by positivity) _ _ _ (wch_nonneg hq1.le), ?_⟩
  unfold Algorithm.hittingTime
  refine le_iInf₂ fun t ht => ?_
  by_contra hlt
  push_neg at hlt
  have htT : t < T := by exact_mod_cast hlt
  set K := t / D + 1 with hK
  have hKd : K < d := by
    have : t / D ≤ t := Nat.div_le_self _ _
    omega
  -- support of the averaged output
  have hsupp : ∀ ω, supp (A.avgOutput I W ω t) ⊆ Pre d K := by
    intro ω
    refine supp_avgOutput A hA.out_zero_respecting I W ω t K fun i => ?_
    refine (allowed_subset_lev A hA.call_zero_respecting hA.send_zero_respecting I W
      (raise SA SB) rfl (fun i ξ x j hx => chainInst_grad_supp SA SB L μ R q i ξ x j hx)
      (fun v η j hv => chainInst_prox_supp SA SB L μ R q v η j hv) ω _ i).trans ?_
    refine Pre_mono (lev_le W (raise SA SB) δ D hD hδD hδ ?_ t i)
    intro i j
    unfold raise
    by_cases h : (i ∈ SA ∧ j % 2 = 0) ∨ (i ∈ SB ∧ j % 2 = 1)
    · rw [if_pos h]
      right
      refine ⟨rfl, ?_⟩
      rcases h with ⟨hi, hj⟩ | ⟨hi, hj⟩
      · exact Or.inl ⟨hj, hSA i hi⟩
      · exact Or.inr ⟨hj, hSB i hi⟩
    · rw [if_neg h]; exact Or.inl rfl
  -- pathwise gap bound
  have hgap : ∀ ω, ENNReal.ofReal (μ / 2 * (R ^ 2 * q ^ (2 * K + 2))) ≤
      I.gap (A.avgOutput I W ω t) := by
    intro ω
    rw [hI, chainInst_gap hm hdisj hcard hd1 hL.le hμ.le hq1.le hqeq]
    apply ENNReal.ofReal_le_ofReal
    refine φc_gap_ge hd1 hα0.le hμ.le hq1.le hqeq _ K hKd ?_
    simp only [cx, dif_pos hKd]
    exact supp_subset_Pre.1 (hsupp ω) ⟨K, hKd⟩ le_rfl
  have hexp : ENNReal.ofReal (μ / 2 * (R ^ 2 * q ^ (2 * K + 2))) ≤ A.expectedGap I W t := by
    unfold Algorithm.expectedGap
    calc ENNReal.ofReal (μ / 2 * (R ^ 2 * q ^ (2 * K + 2)))
        = ∫⁻ _ω, ENNReal.ofReal (μ / 2 * (R ^ 2 * q ^ (2 * K + 2))) ∂A.P := by
          rw [lintegral_const, measure_univ, mul_one]
      _ ≤ _ := lintegral_mono hgap
  -- the real inequality
  have hG0 : μ / 2 * R ^ 2 = 4 * n * μ * Δ / (SA.card * L) := by
    rw [hR2, hα]; field_simp
  have hkey := key_ineq (G0 := 4 * n * μ * Δ / (SA.card * L)) (by positivity) hε hs1 hD htT
    le_rfl (by
      have : 4 * n * μ * Δ / (SA.card * L) / ε = 4 * n * μ * Δ / (SA.card * L * ε) := by
        field_simp
      rw [this]; exact hT)
  have hfin : ε < μ / 2 * (R ^ 2 * q ^ (2 * K + 2)) := by
    rw [← mul_assoc, hG0]; exact hkey
  have := ht.trans' hexp
  rw [ENNReal.ofReal_le_ofReal_iff hε.le] at this
  linarith

end Chain
end ULB63
