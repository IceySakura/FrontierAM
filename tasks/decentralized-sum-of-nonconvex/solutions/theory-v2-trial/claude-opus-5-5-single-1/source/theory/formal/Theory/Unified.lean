import Theory.Main

/-!
# The bound in the form of Hannah–Liu–O'Connor–Yin, for all `κ ≥ 2`

A single statement `T_ε ≥ (1/100) (m + (m/(1+(ln(N/κ))₊) + √(m(κ-1)/n)) ln R)` on one instance
with `μ`-strongly convex components, obtained from the chain bound and the counting bound on the
same instance, with `q` chosen according to the regime of `κ`.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Chain

variable {n m : ℕ}

/-- Counting bound for general `q`: `∑ (q^k)² ≥ m - ∑ k`. -/
lemma count_bound {q : ℝ} (k : Fin m → ℕ) (t : ℕ) (hk : ∑ ξ, k ξ ≤ t) :
    (m : ℝ) - t ≤ ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := by
  have hterm : ∀ ξ, (1 : ℝ) - k ξ ≤ (q ^ k ξ) ^ 2 := by
    intro ξ
    rcases Nat.eq_zero_or_pos (k ξ) with h0 | hpos
    · rw [h0]; simp
    · have : (1 : ℝ) ≤ k ξ := by exact_mod_cast hpos
      nlinarith [sq_nonneg (q ^ k ξ)]
  have h1 := Finset.sum_le_sum fun ξ (_ : ξ ∈ Finset.univ) => hterm ξ
  rw [Finset.sum_sub_distrib] at h1
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h1
  have : (∑ ξ : Fin m, (k ξ : ℝ)) ≤ t := by exact_mod_cast hk
  linarith

/-- Either the counting bound or the chain bound beats `ε`: then the expected gap does. -/
theorem expectedGap_gt_of (hn : 0 < n) (hm : 0 < m) {K : ℕ} (hK : 0 < K) {c : ℕ}
    {L μ Δ ρ q ε : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ) (hq : 0 < q) (hq1 : q < 1) (hε : 0 ≤ ε)
    (I : Instance n m (m * K)) (hI : HardProps I hn μ Δ q)
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) (W : Matrix (Fin n) (Fin n) ℝ)
    (t : ℕ) (ht : t < K)
    (h : ε < Δ * (1 - q) ^ 2 * (1 - t / m) ∨ ε < Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)) :
    ENNReal.ofReal ε < A.expectedGap I W t := by
  rcases h with h | h
  · have hlow : ENNReal.ofReal (Δ * (1 - q) ^ 2 * (1 - t / m)) ≤ A.expectedGap I W t := by
      unfold Algorithm.expectedGap
      calc ENNReal.ofReal (Δ * (1 - q) ^ 2 * (1 - t / m))
          = ∫⁻ _ω, ENNReal.ofReal (Δ * (1 - q) ^ 2 * (1 - t / m)) ∂A.P := by
            rw [lintegral_const, measure_univ, mul_one]
        _ ≤ _ := by
          refine lintegral_mono fun ω => ?_
          obtain ⟨k, hk, hgap⟩ := gap_along_run hn hm hK hμ hΔ hq.le hq1 I hI A hA W ω t ht
          refine le_trans (ENNReal.ofReal_le_ofReal ?_) hgap
          have hc := count_bound (q := q) k t hk
          have hm' : (0 : ℝ) < m := by exact_mod_cast hm
          calc Δ * (1 - q) ^ 2 * (1 - t / m) = Δ * (1 - q) ^ 2 / m * (m - t) := by
                field_simp
            _ ≤ Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 :=
                mul_le_mul_of_nonneg_left hc (by positivity)
    exact lt_of_lt_of_le ((ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε).2 h) hlow
  · exact lt_of_lt_of_le ((ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε).2 h)
      (expectedGap_ge_chain hn hm hK hμ hΔ hq hq1 I hI A hA W t ht)

/-- Packaging: a time `T` before which one of the two bounds beats `ε` is a lower bound. -/
lemma finish_SC {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) (T : ℕ)
    (hT : ∀ t : ℕ, t < T →
      ε < Δ * (1 - q) ^ 2 * (1 - t / m) ∨ ε < Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        (T : ℕ∞) ≤ A.hittingTime I W ε := by
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hardSC n m (T + 1) hn μ Δ q,
    hardSC_inClass hn hm hK hμ hΔ hq.le hq1 hqL,
    fun i ξ => hardSC_comp_strongConvex hn hμ hq.le hq1 i ξ, fun A hA W => ?_⟩
  refine le_hittingTime_of A _ W ε T fun t ht => ?_
  exact expectedGap_gt_of hn hm hK hμ hΔ hq hq1 hε.le _ (hardSC_props hn hm hK hμ hΔ hq.le hq1)
    A hA W t (by omega) (hT t ht)

/-- `1 + log y ≤ 2 √y` for `y ≥ 1`. -/
lemma one_add_log_le {y : ℝ} (hy : 1 ≤ y) : 1 + Real.log y ≤ 2 * Real.sqrt y := by
  have hs : 0 < Real.sqrt y := Real.sqrt_pos.2 (by linarith)
  have h1 : Real.log y = 2 * Real.log (Real.sqrt y) := by
    have e : y = Real.sqrt y * Real.sqrt y := (Real.mul_self_sqrt (by linarith)).symm
    conv_lhs => rw [e]
    rw [Real.log_mul hs.ne' hs.ne']; ring
  have h2 := Real.log_le_sub_one_of_pos hs
  linarith

lemma sqrt_split (hm : 0 < m) {a : ℝ} (ha : 0 ≤ a) :
    Real.sqrt ((m : ℝ) * a) = m * Real.sqrt (a / m) := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  rw [show (m : ℝ) * a = m ^ 2 * (a / m) by field_simp, Real.sqrt_mul (by positivity),
    Real.sqrt_sq hm'.le]

lemma log_ge_half {R : ℝ} (hR : 16 ≤ R) {a : ℝ} (ha : 0 < a) (haR : Real.sqrt R ≤ a * R) :
    Real.log R / 2 ≤ Real.log (a * R) := by
  have hR0 : 0 < R := by linarith
  have h1 : Real.log (Real.sqrt R) = Real.log R / 2 := by
    rw [Real.log_sqrt hR0.le]
  rw [← h1]
  exact Real.log_le_log (Real.sqrt_pos.2 hR0) haR

lemma log_le_of_nonneg_le {x : ℝ} (hx : 0 < x) : Real.log x ≤ x - 1 := Real.log_le_sub_one_of_pos hx

/-- Transient part: before `m/2` iterations the counting bound beats `ε`. -/
lemma transient_ok {m t q Δ ε : ℝ} (hm : 0 < m) (ht : t < m / 2) (hq : 1 / 4 ≤ (1 - q) ^ 2)
    (hε : 0 < ε) (hR : 16 * ε ≤ Δ) : ε < Δ * (1 - q) ^ 2 * (1 - t / m) := by
  have htm : t / m < 1 / 2 := by rw [div_lt_iff₀ hm]; linarith
  have h2 : (1 : ℝ) / 2 < 1 - t / m := by linarith
  have hA1 : 4 * ε ≤ Δ * (1 - q) ^ 2 := by nlinarith
  have hA2 : 0 < Δ * (1 - q) ^ 2 := by linarith
  have h3 := mul_lt_mul_of_pos_left h2 hA2
  linarith

/-- Regime `κ - 1 ≤ 8N`. -/
lemma regime1 {m N μ L Δ ε : ℝ} (hm : 0 < m) (hN : 0 < N) (hμ : 0 < μ) (hL2 : 2 * μ ≤ L)
    (h1 : L - μ ≤ 8 * N * μ) (hε : 0 < ε) (hR : 16 * ε ≤ Δ) :
    (m / (1 + max 0 (Real.log (N * μ / L))) + m * Real.sqrt ((L - μ) / (N * μ))) *
        Real.log (Δ / ε) / 98 ≤
      m / 2 * Real.log ((1 - (L - μ) / (16 * N * μ)) ^ 2 * Δ / ε) /
        Real.log (1 / ((L - μ) / (16 * N * μ))) := by
  have hL : 0 < L := by linarith
  have hLμ : 0 < L - μ := by linarith
  have hNμ : 0 < N * μ := by positivity
  have hΔ : 0 < Δ := by linarith
  have hR16 : 16 ≤ Δ / ε := by rw [le_div_iff₀ hε]; linarith
  have hlR0 : 0 ≤ Real.log (Δ / ε) := Real.log_nonneg (by linarith)
  set ℓ := max 0 (Real.log (N * μ / L)) with hℓdef
  have hℓ0 : 0 ≤ ℓ := le_max_left _ _
  set q := (L - μ) / (16 * N * μ) with hqdef
  have hq : 0 < q := by positivity
  have hq2 : q ≤ 1 / 2 := by rw [hqdef, div_le_iff₀ (by positivity)]; linarith
  have h1q : (1 : ℝ) / 4 ≤ (1 - q) ^ 2 := by nlinarith
  have hinv : 1 / q = 16 * N * μ / (L - μ) := by rw [hqdef]; field_simp
  have hlogq : 0 < Real.log (1 / q) := Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  -- (a)
  have ha : Real.log (1 / q) ≤ 3.5 * (1 + ℓ) := by
    rw [hinv]
    have hle : 16 * N * μ / (L - μ) ≤ 32 * (N * μ / L) := by
      rw [div_le_iff₀ hLμ]
      have e : 32 * (N * μ / L) * (L - μ) = 32 * (N * μ) * ((L - μ) / L) := by field_simp
      rw [e]
      have : (1 : ℝ) / 2 ≤ (L - μ) / L := by rw [le_div_iff₀ hL]; linarith
      nlinarith
    have h32 : Real.log 32 ≤ 3.5 := by
      rw [show (32 : ℝ) = 2 ^ 5 by norm_num, Real.log_pow]
      have := Real.log_two_lt_d9
      push_cast; linarith
    calc Real.log (16 * N * μ / (L - μ)) ≤ Real.log (32 * (N * μ / L)) :=
          Real.log_le_log (by positivity) hle
      _ = Real.log 32 + Real.log (N * μ / L) := Real.log_mul (by norm_num) (by positivity)
      _ ≤ 3.5 + ℓ := add_le_add h32 (le_max_right _ _)
      _ ≤ 3.5 * (1 + ℓ) := by nlinarith
  -- (b)
  have hb : Real.log (Δ / ε) / 2 ≤ Real.log ((1 - q) ^ 2 * Δ / ε) := by
    rw [show (1 - q) ^ 2 * Δ / ε = (1 - q) ^ 2 * (Δ / ε) by ring]
    apply log_ge_half hR16 (by positivity)
    have hsR : Real.sqrt (Δ / ε) ≤ (Δ / ε) / 4 := by
      rw [Real.sqrt_le_left (by positivity)]; nlinarith
    nlinarith
  -- (c)
  have hc : m / (1 + ℓ) + m * Real.sqrt ((L - μ) / (N * μ)) ≤ 4 * m / (1 + ℓ) := by
    by_cases hκN : 0 ≤ Real.log (N * μ / L)
    · have hℓe : ℓ = Real.log (N * μ / L) := max_eq_right hκN
      have hy : 1 ≤ N * μ / L := by
        by_contra hy; push Not at hy
        have := Real.log_neg (by positivity) hy
        linarith
      have hlog := one_add_log_le hy
      rw [← hℓe] at hlog
      have hy0 : 0 < Real.sqrt (N * μ / L) := Real.sqrt_pos.2 (by positivity)
      have hsq1 : Real.sqrt ((L - μ) / (N * μ)) ≤ 2 / (1 + ℓ) := by
        have e1 : Real.sqrt (L / (N * μ)) = (Real.sqrt (N * μ / L))⁻¹ := by
          rw [← Real.sqrt_inv, inv_div]
        calc Real.sqrt ((L - μ) / (N * μ)) ≤ Real.sqrt (L / (N * μ)) :=
              Real.sqrt_le_sqrt (div_le_div_of_nonneg_right (by linarith) hNμ.le)
          _ = (Real.sqrt (N * μ / L))⁻¹ := e1
          _ ≤ 2 / (1 + ℓ) := by
            rw [inv_eq_one_div, div_le_div_iff₀ hy0 (by positivity)]
            linarith
      calc m / (1 + ℓ) + m * Real.sqrt ((L - μ) / (N * μ))
          ≤ m / (1 + ℓ) + m * (2 / (1 + ℓ)) := by gcongr
        _ = 3 * m / (1 + ℓ) := by ring
        _ ≤ 4 * m / (1 + ℓ) := by gcongr; norm_num
    · push Not at hκN
      have hℓe : ℓ = 0 := max_eq_left hκN.le
      rw [hℓe]
      have hs8 : Real.sqrt ((L - μ) / (N * μ)) ≤ 3 := by
        rw [Real.sqrt_le_left (by norm_num), div_le_iff₀ hNμ]; nlinarith
      have : m * Real.sqrt ((L - μ) / (N * μ)) ≤ m * 3 := mul_le_mul_of_nonneg_left hs8 hm.le
      simp only [add_zero, div_one]
      linarith
  -- combine
  rw [le_div_iff₀ hlogq]
  have hXl := mul_le_mul_of_nonneg_right hc hlR0
  have e2 : 4 * m / (1 + ℓ) * Real.log (Δ / ε) * Real.log (1 / q) ≤
      4 * m / (1 + ℓ) * Real.log (Δ / ε) * (3.5 * (1 + ℓ)) :=
    mul_le_mul_of_nonneg_left ha (by positivity)
  have e3 : 4 * m / (1 + ℓ) * Real.log (Δ / ε) * (3.5 * (1 + ℓ)) = 14 * m * Real.log (Δ / ε) := by
    field_simp; ring
  have e4 := mul_le_mul_of_nonneg_right hXl hlogq.le
  have e5 : m / 2 * (Real.log (Δ / ε) / 2) ≤ m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) :=
    mul_le_mul_of_nonneg_left hb (by positivity)
  nlinarith

/-- Regime `8N < κ - 1 < 16N`, with `q = 1/3`. -/
lemma regime2 {m N μ L Δ ε : ℝ} (hm : 0 < m) (hN : 0 < N) (hμ : 0 < μ)
    (h1 : 8 * N * μ < L - μ) (h2 : L - μ < 16 * N * μ) (hε : 0 < ε) (hR : 16 * ε ≤ Δ) :
    (m / (1 + max 0 (Real.log (N * μ / L))) + m * Real.sqrt ((L - μ) / (N * μ))) *
        Real.log (Δ / ε) / 98 ≤
      m / 2 * Real.log ((1 - 1 / 3) ^ 2 * Δ / ε) / Real.log (1 / (1 / 3 : ℝ)) := by
  have hNμ : 0 < N * μ := by positivity
  have hL : 0 < L := by nlinarith
  have hΔ : 0 < Δ := by linarith
  have hR16 : 16 ≤ Δ / ε := by rw [le_div_iff₀ hε]; linarith
  have hlR0 : 0 ≤ Real.log (Δ / ε) := Real.log_nonneg (by linarith)
  have hℓe : max 0 (Real.log (N * μ / L)) = 0 := by
    apply max_eq_left
    apply le_of_lt
    apply Real.log_neg (by positivity)
    rw [div_lt_one hL]; nlinarith
  rw [hℓe]
  have hc : m / (1 + 0) + m * Real.sqrt ((L - μ) / (N * μ)) ≤ 5 * m := by
    have hs16 : Real.sqrt ((L - μ) / (N * μ)) ≤ 4 := by
      rw [Real.sqrt_le_left (by norm_num), div_le_iff₀ hNμ]; nlinarith
    have : m * Real.sqrt ((L - μ) / (N * μ)) ≤ m * 4 := mul_le_mul_of_nonneg_left hs16 hm.le
    simp only [add_zero, div_one]
    linarith
  have hlogq : 0 < Real.log (1 / (1 / 3 : ℝ)) := Real.log_pos (by norm_num)
  have ha : Real.log (1 / (1 / 3 : ℝ)) ≤ 2 := by
    have := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 1 / (1 / 3))
    norm_num at this ⊢; linarith
  have hb : Real.log (Δ / ε) / 2 ≤ Real.log ((1 - 1 / 3) ^ 2 * Δ / ε) := by
    rw [show (1 - 1 / 3 : ℝ) ^ 2 * Δ / ε = (4 / 9) * (Δ / ε) by ring]
    apply log_ge_half hR16 (by norm_num)
    have hsR : Real.sqrt (Δ / ε) ≤ (Δ / ε) / 4 := by
      rw [Real.sqrt_le_left (by positivity)]; nlinarith
    nlinarith
  rw [le_div_iff₀ hlogq]
  have hXl := mul_le_mul_of_nonneg_right hc hlR0
  have e4 := mul_le_mul hXl ha hlogq.le (by positivity)
  have e5 : m / 2 * (Real.log (Δ / ε) / 2) ≤ m / 2 * Real.log ((1 - 1 / 3) ^ 2 * Δ / ε) :=
    mul_le_mul_of_nonneg_left hb (by positivity)
  nlinarith

/-- Regime `κ - 1 ≥ 16N`, with `q = 1 - 2√(N/(κ-1))` (here `r = N/(κ-1)`, `s = √r`). -/
lemma regime3 {m N μ L Δ ε : ℝ} (hm : 0 < m) (hN : 0 < N) (hμ : 0 < μ)
    (h2 : 16 * N * μ ≤ L - μ) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (N * μ) ^ 2 * Δ) :
    (m + (m / (1 + max 0 (Real.log (N * μ / L))) + m * Real.sqrt ((L - μ) / (N * μ))) *
        Real.log (Δ / ε)) / 100 ≤
      m / 2 * Real.log ((1 - (1 - 2 * Real.sqrt (N * μ / (L - μ)))) ^ 2 * Δ / ε) /
        Real.log (1 / (1 - 2 * Real.sqrt (N * μ / (L - μ)))) := by
  have hNμ : 0 < N * μ := by positivity
  have hLμ : 0 < L - μ := by nlinarith
  have hL : 0 < L := by nlinarith
  have hΔ : 0 < Δ := by linarith
  have hR16 : 16 ≤ Δ / ε := by rw [le_div_iff₀ hε]; linarith
  have hlR1 : 1 ≤ Real.log (Δ / ε) := by
    have : Real.exp 1 ≤ Δ / ε := by have := Real.exp_one_lt_d9; linarith
    calc (1 : ℝ) = Real.log (Real.exp 1) := (Real.log_exp 1).symm
      _ ≤ Real.log (Δ / ε) := Real.log_le_log (Real.exp_pos 1) this
  set r := N * μ / (L - μ) with hrdef
  have hr : 0 < r := by positivity
  have hr16 : r ≤ 1 / 16 := by rw [hrdef, div_le_iff₀ hLμ]; nlinarith
  set s := Real.sqrt r with hsdef
  have hs : 0 < s := Real.sqrt_pos.2 hr
  have hs2 : s ^ 2 = r := Real.sq_sqrt hr.le
  have hs4 : s ≤ 1 / 4 := by
    rw [hsdef]
    calc Real.sqrt r ≤ Real.sqrt (1 / 16) := Real.sqrt_le_sqrt hr16
      _ = 1 / 4 := by
        rw [show (1 : ℝ) / 16 = (1 / 4) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hq : 0 < 1 - 2 * s := by linarith
  have hℓe : max 0 (Real.log (N * μ / L)) = 0 := by
    apply max_eq_left
    apply le_of_lt
    apply Real.log_neg (by positivity)
    rw [div_lt_one hL]; nlinarith
  have hsqrt : Real.sqrt ((L - μ) / (N * μ)) = s⁻¹ := by
    rw [hsdef, hrdef, ← Real.sqrt_inv, inv_div]
  rw [hℓe, hsqrt]
  have h1q2 : (1 - (1 - 2 * s)) ^ 2 = 4 * r := by nlinarith
  have hlogq : 0 < Real.log (1 / (1 - 2 * s)) :=
    Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have ha : Real.log (1 / (1 - 2 * s)) ≤ 4 * s := by
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < 1 / (1 - 2 * s))
    have h2 : 1 / (1 - 2 * s) - 1 ≤ 4 * s := by
      rw [div_sub_one hq.ne', div_le_iff₀ hq]; nlinarith
    linarith
  have hb : Real.log (Δ / ε) / 2 ≤ Real.log ((1 - (1 - 2 * s)) ^ 2 * Δ / ε) := by
    rw [h1q2, show 4 * r * Δ / ε = (4 * r) * (Δ / ε) by ring]
    apply log_ge_half hR16 (by positivity)
    have hRr : 1 ≤ 16 * r ^ 2 * (Δ / ε) := by
      have e : 16 * r ^ 2 * (Δ / ε) = 16 * (N * μ) ^ 2 * Δ / ((L - μ) ^ 2 * ε) := by
        rw [hrdef]; field_simp
      rw [e, le_div_iff₀ (by positivity)]
      linarith
    rw [Real.sqrt_le_left (by positivity)]
    nlinarith
  have hX : m / (1 + 0) + m * s⁻¹ ≤ 5 / 4 * (m / s) := by
    have : m ≤ (1 / 4) * (m / s) := by
      rw [show (1 / 4) * (m / s) = m / (4 * s) by field_simp, le_div_iff₀ (by positivity)]
      nlinarith
    simp only [add_zero, div_one]
    rw [← div_eq_mul_inv]
    linarith
  have hXm : m ≤ (m / (1 + 0) + m * s⁻¹) * Real.log (Δ / ε) := by
    have : m ≤ m / (1 + 0) + m * s⁻¹ := by
      have : 0 ≤ m * s⁻¹ := by positivity
      simp only [add_zero, div_one]; linarith
    nlinarith
  rw [le_div_iff₀ hlogq]
  have step1 : (m + (m / (1 + 0) + m * s⁻¹) * Real.log (Δ / ε)) / 100 ≤
      5 / 4 * (m / s) * Real.log (Δ / ε) / 50 := by
    have := mul_le_mul_of_nonneg_right hX (by linarith : (0 : ℝ) ≤ Real.log (Δ / ε))
    linarith
  have step3 : 5 / 4 * (m / s) * Real.log (Δ / ε) / 50 * Real.log (1 / (1 - 2 * s)) ≤
      5 / 4 * (m / s) * Real.log (Δ / ε) / 50 * (4 * s) :=
    mul_le_mul_of_nonneg_left ha (by positivity)
  have step4 : 5 / 4 * (m / s) * Real.log (Δ / ε) / 50 * (4 * s) = m * Real.log (Δ / ε) / 10 := by
    field_simp; ring
  have step5 := mul_le_mul_of_nonneg_right step1 hlogq.le
  have e5 : m / 2 * (Real.log (Δ / ε) / 2) ≤ m / 2 * Real.log ((1 - (1 - 2 * s)) ^ 2 * Δ / ε) :=
    mul_le_mul_of_nonneg_left hb (by positivity)
  nlinarith

/-- The regime-dependent choice of `q` behind R9: before the bound, one of the two bounds beats `ε`. -/
lemma hly_choice (hn : 0 < n) (hm : 0 < m) {L μ Δ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ q : ℝ, 0 < q ∧ q < 1 ∧ 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2 ∧
      ∀ t : ℕ, (t : ℝ) < ((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
          Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100 →
        ε < Δ * (1 - q) ^ 2 * (1 - t / m) ∨ ε < Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m) := by
  have hΔ : 0 < Δ := by linarith
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hNR : (0 : ℝ) < (n : ℝ) * m := by positivity
  have hLμ : 0 ≤ L - μ := by linarith
  have hsq : Real.sqrt (m * (L - μ) / (n * μ)) = m * Real.sqrt ((L - μ) / (n * m * μ)) := by
    rw [show (m : ℝ) * (L - μ) / (n * μ) = m * ((L - μ) / (n * μ)) by ring,
      sqrt_split hm (by positivity)]
    congr 2
    field_simp
  rw [hsq]
  have hlR0 : 0 ≤ Real.log (Δ / ε) := Real.log_nonneg (by rw [le_div_iff₀ hε]; linarith)
  -- `t < T` splits into `t < m/2` or `98 t < X lR`
  have hsplit : ∀ t : ℕ, (t : ℝ) < ((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
      m * Real.sqrt ((L - μ) / (n * m * μ))) * Real.log (Δ / ε)) / 100 →
      (t : ℝ) < m / 2 ∨ (t : ℝ) < ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
        m * Real.sqrt ((L - μ) / (n * m * μ))) * Real.log (Δ / ε) / 98 := by
    intro t ht
    by_cases hs : (t : ℝ) < m / 2
    · exact Or.inl hs
    · right; push Not at hs; linarith
  by_cases h1 : L - μ ≤ 8 * ((n : ℝ) * m) * μ
  · -- regime `κ - 1 ≤ 8N`
    have hq : 0 < (L - μ) / (16 * ((n : ℝ) * m) * μ) := by
      apply div_pos (by linarith) (by positivity)
    have hq2 : (L - μ) / (16 * ((n : ℝ) * m) * μ) ≤ 1 / 2 := by
      rw [div_le_iff₀ (by positivity)]; linarith
    have hq1 : (L - μ) / (16 * ((n : ℝ) * m) * μ) < 1 := by linarith
    have h1q : (1 : ℝ) / 4 ≤ (1 - (L - μ) / (16 * ((n : ℝ) * m) * μ)) ^ 2 := by nlinarith
    have hqL : 4 * ((n : ℝ) * m) * μ * ((L - μ) / (16 * ((n : ℝ) * m) * μ)) ≤
        (L - μ) * (1 - (L - μ) / (16 * ((n : ℝ) * m) * μ)) ^ 2 := by
      have e : 4 * ((n : ℝ) * m) * μ * ((L - μ) / (16 * ((n : ℝ) * m) * μ)) = (L - μ) / 4 := by
        field_simp; ring
      rw [e]; nlinarith
    refine ⟨_, hq, hq1, hqL, fun t ht => ?_⟩
    rcases hsplit t ht with hs | hs
    · exact Or.inl (transient_ok hmR hs h1q hε hR)
    · right
      apply chain_threshold hm hΔ hε hq hq1 t
      exact lt_of_lt_of_le hs (regime1 hmR hNR hμ hL2 h1 hε hR)
  · push Not at h1
    by_cases h2 : L - μ < 16 * ((n : ℝ) * m) * μ
    · -- regime `8N < κ - 1 < 16N`
      have hq : (0 : ℝ) < 1 / 3 := by norm_num
      have hq1 : (1 : ℝ) / 3 < 1 := by norm_num
      have h1q : (1 : ℝ) / 4 ≤ (1 - 1 / 3) ^ 2 := by norm_num
      have hqL : 4 * ((n : ℝ) * m) * μ * (1 / 3) ≤ (L - μ) * (1 - 1 / 3) ^ 2 := by nlinarith
      refine ⟨_, hq, hq1, hqL, fun t ht => ?_⟩
      rcases hsplit t ht with hs | hs
      · exact Or.inl (transient_ok hmR hs h1q hε hR)
      · right
        apply chain_threshold hm hΔ hε hq hq1 t
        exact lt_of_lt_of_le hs (regime2 hmR hNR hμ h1 h2 hε hR)
    · -- regime `κ - 1 ≥ 16N`
      push Not at h2
      have hNμ : 0 < (n : ℝ) * m * μ := by positivity
      have hLμ' : 0 < L - μ := by nlinarith
      have hr : 0 < (n : ℝ) * m * μ / (L - μ) := by positivity
      have hr16 : (n : ℝ) * m * μ / (L - μ) ≤ 1 / 16 := by
        rw [div_le_iff₀ hLμ']; nlinarith
      have hs : 0 < Real.sqrt ((n : ℝ) * m * μ / (L - μ)) := Real.sqrt_pos.2 hr
      have hs2 : Real.sqrt ((n : ℝ) * m * μ / (L - μ)) ^ 2 = (n : ℝ) * m * μ / (L - μ) :=
        Real.sq_sqrt hr.le
      have hs4 : Real.sqrt ((n : ℝ) * m * μ / (L - μ)) ≤ 1 / 4 := by
        calc Real.sqrt ((n : ℝ) * m * μ / (L - μ)) ≤ Real.sqrt (1 / 16) := Real.sqrt_le_sqrt hr16
          _ = 1 / 4 := by
            rw [show (1 : ℝ) / 16 = (1 / 4) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
      have hq : 0 < 1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)) := by linarith
      have hq1 : 1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)) < 1 := by linarith
      have hqL : 4 * ((n : ℝ) * m) * μ * (1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ))) ≤
          (L - μ) * (1 - (1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)))) ^ 2 := by
        have e : (L - μ) * (1 - (1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)))) ^ 2 =
            4 * ((n : ℝ) * m * μ) := by
          rw [show (1 - (1 - 2 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)))) ^ 2 =
            4 * Real.sqrt ((n : ℝ) * m * μ / (L - μ)) ^ 2 by ring, hs2]
          field_simp
        rw [e]
        nlinarith
      refine ⟨_, hq, hq1, hqL, fun t ht => ?_⟩
      right
      apply chain_threshold hm hΔ hε hq hq1 t
      exact lt_of_lt_of_le ht (regime3 hmR hNR hμ h2 hε hR hRκ)


/-- **R9 (aux).** The Hannah–Liu–O'Connor–Yin form, for all `κ ≥ 2`. -/
theorem lower_bound_hly_form_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  have hΔ : 0 < Δ := by linarith
  obtain ⟨q, hq, hq1, hqL, hT⟩ := hly_choice hn hm hμ hL2 hε hR hRκ
  exact finish_SC hn hm hμ hΔ hε hq hq1 hqL _ fun t ht => hT t (Nat.lt_ceil.1 ht)

/-- Pathwise packaging (cf. `finish_SC`). -/
lemma finish_SC_pathwise {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) (B : ℝ)
    (hT : ∀ t : ℕ, (t : ℝ) < B →
      ε < Δ * (1 - q) ^ 2 * (1 - t / m) ∨ ε < Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < B → ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) := by
  set T := ⌈B⌉₊
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hardSC n m (T + 1) hn μ Δ q,
    hardSC_inClass hn hm hK hμ hΔ hq.le hq1 hqL,
    fun i ξ => hardSC_comp_strongConvex hn hμ hq.le hq1 i ξ, fun A hA W ω t ht => ?_⟩
  have htK : t < T + 1 := by
    have : t < T := Nat.lt_ceil.2 ht
    omega
  obtain ⟨k, hk, hgap⟩ := gap_along_run hn hm hK hμ hΔ hq.le hq1 _
    (hardSC_props hn hm hK hμ hΔ hq.le hq1) A hA W ω t htK
  refine lt_of_lt_of_le ?_ hgap
  rw [ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε.le]
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hc0 : 0 ≤ Δ * (1 - q) ^ 2 / m := by positivity
  rcases hT t ht with h | h
  · refine lt_of_lt_of_le h ?_
    have hc := count_bound (q := q) k t hk
    calc Δ * (1 - q) ^ 2 * (1 - t / m) = Δ * (1 - q) ^ 2 / m * (m - t) := by field_simp
      _ ≤ Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := mul_le_mul_of_nonneg_left hc hc0
  · refine lt_of_lt_of_le h ?_
    have ha := amgm hm hq hq1 k t hk
    calc Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)
        = Δ * (1 - q) ^ 2 / m * (m * q ^ ((2 * t : ℝ) / m)) := by field_simp
      _ ≤ Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := mul_le_mul_of_nonneg_left ha hc0

/-- **R11 (aux).** Pathwise form of R9. -/
theorem lower_bound_hly_form_pathwise_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL2 : 2 * μ ≤ L) (hε : 0 < ε) (hR : 16 * ε ≤ Δ)
    (hRκ : (L - μ) ^ 2 * ε ≤ 16 * (n * m * μ) ^ 2 * Δ) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < ((m : ℝ) + ((m : ℝ) / (1 + max 0 (Real.log (n * m * μ / L))) +
            Real.sqrt (m * (L - μ) / (n * μ))) * Real.log (Δ / ε)) / 100 →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) := by
  have hΔ : 0 < Δ := by linarith
  obtain ⟨q, hq, hq1, hqL, hT⟩ := hly_choice hn hm hμ hL2 hε hR hRκ
  exact finish_SC_pathwise hn hm hμ hΔ hε hq hq1 hqL _ hT

end Chain
end ULB63
