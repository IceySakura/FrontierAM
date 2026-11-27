import Theory.LowerData2

/-!
# The data lower bound
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Blocks

variable {n m c : ℕ}

/-- Pointwise: `q^{2k+2} ≥ q² (1 - k)`. -/
lemma pow_ge_lin {q : ℝ} (hq0 : 0 ≤ q) (k : ℕ) : q ^ 2 * (1 - (k : ℝ)) ≤ q ^ (2 * k + 2) := by
  rcases Nat.eq_zero_or_pos k with hk | hk
  · subst hk; simp
  · have h1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
    have h2 : q ^ 2 * (1 - (k : ℝ)) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by positivity) (by linarith)
    exact h2.trans (by positivity)

/-- **Data lower bound for an arbitrary gossip matrix.** -/
theorem lower_bound_data (hn : 1 ≤ n) (hm : 1 ≤ m) {L μ Δ ρ ε : ℝ} (hL : 0 < L) (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hε : 0 < ε) (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ)
    (W : Matrix (Fin n) (Fin n) ℝ) (T : ℕ)
    (hT : (T : ℝ) ≤ max (m / 5 * (1 - (1 + L / (2 * n * m * μ)) * ε / Δ))
      (m / 10 * Real.log (Δ / ((1 + L / (2 * n * m * μ)) * ε)) /
        Real.log ((Real.sqrt (1 + L / (n * m * μ)) + 1) / (Real.sqrt (1 + L / (n * m * μ)) - 1)))) :
    ∃ d, ∃ I : Instance n m d, I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      (T : ℕ∞) ≤ A.hittingTime I W ε := by
  classical
  have hn' : (0 : ℝ) < n := by exact_mod_cast (by omega : 0 < n)
  have hm' : (0 : ℝ) < m := by exact_mod_cast (by omega : 0 < m)
  set α : ℝ := L / (4 * n * m) with hα
  have hα0 : 0 < α := by positivity
  set s : ℝ := Real.sqrt (1 + L / (n * m * μ)) with hs
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
  have h1q : 1 - q = 2 / (s + 1) := by rw [hq]; field_simp; ring
  have hqeq : μ * q = α * (1 - q) ^ 2 := by
    rw [h1q, hq]
    have hα' : α = μ * (s ^ 2 - 1) / 4 := by
      rw [hs2]; field_simp; ring
    rw [hα']
    field_simp
    ring
  set lam : ℝ := 1 + L / (2 * n * m * μ) with hlam
  have hlam1 : 1 ≤ lam := by have : 0 < L / (2 * n * m * μ) := by positivity
                             linarith
  have hq2 : 1 / lam ≤ (1 - q) ^ 2 := by
    rw [h1q, div_pow, div_le_div_iff₀ (by linarith) (by positivity)]
    have : (s + 1) ^ 2 ≤ 4 * lam := by
      have : 4 * lam = 2 * s ^ 2 + 2 := by
        rw [hs2, hlam, hα]; field_simp; ring
      nlinarith [sq_nonneg (s - 1)]
    linarith
  set R : ℝ := Real.sqrt (2 * Δ / (m * α * q)) with hR
  have hR2 : R ^ 2 = 2 * Δ / (m * α * q) := Real.sq_sqrt (by positivity)
  have hRΔ : (m : ℝ) * (α / 2 * R ^ 2 * q) ≤ Δ := by
    rw [hR2]; field_simp; exact le_refl _
  set b := 5 * T + 1 with hb
  have hb1 : 1 ≤ b := by omega
  let i0 : Fin n := ⟨0, by omega⟩
  set I := blockInst n m b i0 L μ R q with hI
  refine ⟨m * b, I, blockInst_inClass hm hn i0 hb1 hL.le hμ hq0.le hq1.le hqeq hRΔ,
    fun i ξ => bq_convex hL.le _ _ _ (Chain.wch_nonneg hq1.le) ξ, ?_⟩
  unfold Algorithm.hittingTime
  refine le_iInf₂ fun t ht => ?_
  by_contra hlt
  push_neg at hlt
  have htT : t < T := by exact_mod_cast hlt
  have hgrad0 : ∀ ξ (x : E (m * b)) k, supp x ⊆ BP k →
      supp (gradient (I.F i0 ξ) x) ⊆ BP (Function.update k ξ (k ξ + 1)) := by
    intro ξ x k hx
    have hF : I.F i0 ξ = bq L (range (b + 1)) (Chain.wch b q) (Chain.cch R) ξ := by
      simp [hI, blockInst, Si]
    rw [hF, gradient_bq]
    apply supp_Em_of ξ
    rw [Function.update_self, supp_subset_Pre]
    intro p hp
    have hPx := supp_P_of_BP hx ξ
    refine Chain.quadGrad_u_vanish _ _ _ _ (fun k' hk' => by simp [Chain.cch, hk']) (k ξ) (P ξ x)
      ?_ p ?_
    · intro k' hk'
      by_cases hk'b : k' < b
      · simp only [Chain.cx, dif_pos hk'b]
        exact supp_subset_Pre.1 hPx ⟨k', hk'b⟩ hk'
      · exact Chain.cx_of_ge (by omega)
    · split_ifs <;> omega
  have hgradoth : ∀ i, i ≠ i0 → ∀ ξ (x : E (m * b)), gradient (I.F i ξ) x = 0 := by
    intro i hi ξ x
    have hF : I.F i ξ = bq L ∅ (Chain.wch b q) (Chain.cch R) ξ := by
      simp [hI, blockInst, Si, hi]
    rw [hF, gradient_bq]
    ext p
    simp [Chain.quadGrad]
  have hprox : ∀ (v : E (m * b)) η k, supp v ⊆ BP k → supp (I.prox v η) ⊆ BP k := by
    intro v η k hv p hp
    apply hv
    simp only [hI, blockInst, supp, Set.mem_setOf_eq, PiLp.smul_apply, smul_eq_mul] at hp ⊢
    exact right_ne_zero_of_mul hp
  have hcast : ((microSteps * t : ℕ) : ℝ) = 5 * t := by simp [microSteps]
  set X : ℝ := 10 * (t : ℝ) / m with hX
  set G : ℝ := Δ * (1 - q) ^ 2 * max (1 - 5 * (t : ℝ) / m) (q ^ X) with hG
  have hgap : ∀ ω, ENNReal.ofReal G ≤ I.gap (A.avgOutput I W ω t) := by
    intro ω
    obtain ⟨k, hk, hall⟩ := exists_budget A hA.call_zero_respecting hA.send_zero_respecting I W BP
      (fun k k' h => BP_mono h) i0 rfl hgrad0 hgradoth hprox ω (microSteps * t)
    have hsupp := supp_avgOutput_set A hA.out_zero_respecting I W ω t (BP k) hall
    rw [hI, blockInst_gap hm hn i0 hb1 hL.le hμ.le hq1.le hqeq]
    apply ENNReal.ofReal_le_ofReal
    have hkb : ∀ ξ, k ξ < b := by
      intro ξ
      have h1 : k ξ ≤ ∑ ξ', k ξ' := Finset.single_le_sum (fun _ _ => Nat.zero_le _) (mem_univ ξ)
      have h2 : microSteps * t < b := by simp only [microSteps]; omega
      omega
    have hterm : ∀ ξ, μ / 2 * (R ^ 2 * q ^ (2 * k ξ + 2)) ≤
        Chain.φc b α μ R q (P ξ (A.avgOutput I W ω t)) - Chain.φc b α μ R q (Chain.xstar b R q) := by
      intro ξ
      refine Chain.φc_gap_ge hb1 hα0.le hμ.le hq1.le hqeq _ (k ξ) (hkb ξ) ?_
      simp only [Chain.cx, dif_pos (hkb ξ)]
      exact supp_subset_Pre.1 (supp_P_of_BP hsupp ξ) ⟨k ξ, hkb ξ⟩ le_rfl
    have hsum : μ / 2 * R ^ 2 * ∑ ξ, q ^ (2 * k ξ + 2) ≤
        ∑ ξ, (Chain.φc b α μ R q (P ξ (A.avgOutput I W ω t)) -
          Chain.φc b α μ R q (Chain.xstar b R q)) := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun ξ _ => by have := hterm ξ; linarith
    -- the coefficient: `(μ/2) R² q² = Δ (1-q)² / m`
    have hcoef : μ / 2 * R ^ 2 * q ^ 2 = Δ * (1 - q) ^ 2 / m := by
      rw [hR2]
      have : μ * q = α * (1 - q) ^ 2 := hqeq
      field_simp
      nlinarith [this]
    have hk' : ((∑ ξ, k ξ : ℕ) : ℝ) ≤ 5 * t := by
      have : ∑ ξ, k ξ ≤ microSteps * t := hk
      rw [← hcast]; exact_mod_cast this
    -- (a) the counting bound
    have hA' : Δ * (1 - q) ^ 2 * (1 - 5 * (t : ℝ) / m) ≤ μ / 2 * R ^ 2 * ∑ ξ, q ^ (2 * k ξ + 2) := by
      have h1 : q ^ 2 * ((m : ℝ) - 5 * t) ≤ ∑ ξ, q ^ (2 * k ξ + 2) := by
        calc q ^ 2 * ((m : ℝ) - 5 * t) ≤ q ^ 2 * ((m : ℝ) - ((∑ ξ, k ξ : ℕ) : ℝ)) :=
              mul_le_mul_of_nonneg_left (by linarith) (by positivity)
          _ = ∑ ξ, q ^ 2 * (1 - (k ξ : ℝ)) := by
              rw [← Finset.mul_sum, Finset.sum_sub_distrib]; push_cast; simp
          _ ≤ _ := Finset.sum_le_sum fun ξ _ => pow_ge_lin hq0.le (k ξ)
      have h2 : Δ * (1 - q) ^ 2 * (1 - 5 * (t : ℝ) / m) =
          μ / 2 * R ^ 2 * (q ^ 2 * ((m : ℝ) - 5 * t)) := by
        rw [← mul_assoc, hcoef]; field_simp
      rw [h2]
      exact mul_le_mul_of_nonneg_left h1 (by positivity)
    -- (b) the AM–GM bound
    have hB' : Δ * (1 - q) ^ 2 * q ^ X ≤ μ / 2 * R ^ 2 * ∑ ξ, q ^ (2 * k ξ + 2) := by
      have h1 := sum_pow_ge hq0 hq1.le hm k _ hk
      have h2 : q ^ (2 * ((microSteps * t : ℕ) : ℝ) / m + 2) = q ^ X * q ^ 2 := by
        rw [Real.rpow_add hq0, hcast, hX]
        congr 1
        · congr 1; ring
        · norm_cast
      rw [h2] at h1
      have h3 : Δ * (1 - q) ^ 2 * q ^ X = μ / 2 * R ^ 2 * (m * (q ^ X * q ^ 2)) := by
        have : μ / 2 * R ^ 2 * (m * (q ^ X * q ^ 2)) = (μ / 2 * R ^ 2 * q ^ 2) * m * q ^ X := by ring
        rw [this, hcoef]; field_simp
      rw [h3]
      exact mul_le_mul_of_nonneg_left h1 (by positivity)
    rw [hG, mul_max_of_nonneg _ _ (by positivity)]
    exact (max_le hA' hB').trans hsum
  have hexp : ENNReal.ofReal G ≤ A.expectedGap I W t := by
    unfold Algorithm.expectedGap
    calc ENNReal.ofReal G = ∫⁻ _ω, ENNReal.ofReal G ∂A.P := by
          rw [lintegral_const, measure_univ, mul_one]
      _ ≤ _ := lintegral_mono hgap
  have htT' : (t : ℝ) + 1 ≤ T := by exact_mod_cast htT
  have hεG : ε < G := by
    rw [hG]
    rcases le_max_iff.1 hT with hTa | hTb
    · -- the counting branch
      have h1 : 5 * (t : ℝ) / m < 1 - lam * ε / Δ := by
        rw [div_lt_iff₀ hm']
        have : (t : ℝ) < m / 5 * (1 - lam * ε / Δ) := by linarith
        linarith
      have h2 : lam * ε / Δ < 1 - 5 * (t : ℝ) / m := by linarith
      have h3 : ε < Δ / lam * (1 - 5 * (t : ℝ) / m) := by
        have he : Δ / lam * (lam * ε / Δ) = ε := by field_simp
        calc ε = Δ / lam * (lam * ε / Δ) := he.symm
          _ < Δ / lam * (1 - 5 * (t : ℝ) / m) := mul_lt_mul_of_pos_left h2 (by positivity)
      have h4 : Δ / lam ≤ Δ * (1 - q) ^ 2 := by
        rw [div_eq_mul_one_div]; exact mul_le_mul_of_nonneg_left hq2 hΔ.le
      have h5 : 0 ≤ 1 - 5 * (t : ℝ) / m := by
        have : 0 ≤ lam * ε / Δ := by positivity
        linarith
      calc ε < Δ / lam * (1 - 5 * (t : ℝ) / m) := h3
        _ ≤ Δ * (1 - q) ^ 2 * (1 - 5 * (t : ℝ) / m) := mul_le_mul_of_nonneg_right h4 h5
        _ ≤ _ := mul_le_mul_of_nonneg_left (le_max_left _ _) (by positivity)
    · -- the AM–GM branch
      have hX0 : 0 ≤ X := by positivity
      have hs0' : 0 < s - 1 := by linarith
      have hlq : 0 < Real.log ((s + 1) / (s - 1)) := by
        apply Real.log_pos
        rw [one_lt_div hs0']; linarith
      have hXlt : X * Real.log ((s + 1) / (s - 1)) < Real.log (Δ / lam / ε) := by
        have hlogeq : Δ / lam / ε = Δ / (lam * ε) := by field_simp
        rw [hlogeq]
        have h1 : X < Real.log (Δ / (lam * ε)) / Real.log ((s + 1) / (s - 1)) := by
          rw [hX, div_lt_iff₀ hm']
          have h2 : (t : ℝ) < m / 10 * Real.log (Δ / (lam * ε)) / Real.log ((s + 1) / (s - 1)) := by
            linarith
          have h3 : m / 10 * Real.log (Δ / (lam * ε)) / Real.log ((s + 1) / (s - 1)) =
              Real.log (Δ / (lam * ε)) / Real.log ((s + 1) / (s - 1)) * m / 10 := by ring
          rw [h3] at h2
          linarith
        rwa [lt_div_iff₀ hlq] at h1
      have hkey : ε < Δ / lam * q ^ X := by
        have hG : 0 < Δ / lam := by positivity
        have hpow : Real.log (q ^ X) = -(X * Real.log ((s + 1) / (s - 1))) := by
          rw [Real.log_rpow hq0, hq, show (s - 1) / (s + 1) = ((s + 1) / (s - 1))⁻¹ by
            rw [inv_div], Real.log_inv]
          ring
        have hlt' : Real.log (ε / (Δ / lam)) < Real.log (q ^ X) := by
          rw [hpow, Real.log_div hε.ne' hG.ne']
          rw [Real.log_div hG.ne' hε.ne'] at hXlt
          linarith
        have := (Real.log_lt_log_iff (by positivity) (by positivity)).1 hlt'
        rw [div_lt_iff₀ hG] at this
        linarith
      have h4 : Δ / lam ≤ Δ * (1 - q) ^ 2 := by
        rw [div_eq_mul_one_div]; exact mul_le_mul_of_nonneg_left hq2 hΔ.le
      calc ε < Δ / lam * q ^ X := hkey
        _ ≤ Δ * (1 - q) ^ 2 * q ^ X := mul_le_mul_of_nonneg_right h4 (by positivity)
        _ ≤ _ := mul_le_mul_of_nonneg_left (le_max_right _ _) (by positivity)
  have := ht.trans' hexp
  rw [ENNReal.ofReal_le_ofReal_iff hε.le] at this
  linarith

end Blocks
end ULB63
