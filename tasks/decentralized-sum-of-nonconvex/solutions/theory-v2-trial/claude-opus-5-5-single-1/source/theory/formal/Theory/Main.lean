import Theory.StronglyConvex

/-!
# Main lower bounds

All results are for every admissible algorithm, on a single hard instance (chosen independently of
the algorithm), and for every mixing matrix.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Chain

variable {n m : ℕ}

/-- Transient term with `μ`-strongly convex components. -/
theorem lower_bound_transient_SC_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  set T := ⌈(m : ℝ) * (1 - ε / Δ)⌉₊
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hardSC n m (T + 1) hn μ Δ 0,
    hardSC_inClass hn hm hK hμ hΔ le_rfl zero_lt_one (by simp; linarith),
    fun i ξ => hardSC_comp_strongConvex hn hμ le_rfl zero_lt_one i ξ, fun A hA W => ?_⟩
  refine le_hittingTime_of A _ W ε T fun t ht => ?_
  refine lt_of_lt_of_le ?_ (expectedGap_ge_count hn hm hK hμ hΔ _
    (hardSC_props hn hm hK hμ hΔ le_rfl zero_lt_one) A hA W t (by omega))
  have htT : (t : ℝ) < m * (1 - ε / Δ) := Nat.lt_ceil.1 ht
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  rw [ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε.le]
  have h1 : (t : ℝ) / m < 1 - ε / Δ := by rw [div_lt_iff₀ hm']; linarith
  have h2 : ε / Δ < 1 - t / m := by linarith
  rw [div_lt_iff₀ hΔ] at h2
  linarith

/-- Chain term with `μ`-strongly convex components. -/
theorem lower_bound_chain_SC_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  set T := ⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hardSC n m (T + 1) hn μ Δ q,
    hardSC_inClass hn hm hK hμ hΔ hq.le hq1 hqL,
    fun i ξ => hardSC_comp_strongConvex hn hμ hq.le hq1 i ξ, fun A hA W => ?_⟩
  refine le_hittingTime_of A _ W ε T fun t ht => ?_
  refine lt_of_lt_of_le ?_ (expectedGap_ge_chain hn hm hK hμ hΔ hq hq1 _
    (hardSC_props hn hm hK hμ hΔ hq.le hq1) A hA W t (by omega))
  have htT : (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q) :=
    Nat.lt_ceil.1 ht
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1q : 0 < 1 - q := by linarith
  have hlogq : 0 < Real.log (1 / q) := Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  rw [ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε.le]
  have hexp : (2 * t : ℝ) / m * Real.log (1 / q) < Real.log ((1 - q) ^ 2 * Δ / ε) := by
    rw [lt_div_iff₀ hlogq] at htT
    have : (2 * t : ℝ) / m * Real.log (1 / q) = 2 / m * (t * Real.log (1 / q)) := by ring
    rw [this]
    calc 2 / m * (t * Real.log (1 / q)) < 2 / m * (m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε)) :=
          mul_lt_mul_of_pos_left htT (by positivity)
      _ = Real.log ((1 - q) ^ 2 * Δ / ε) := by field_simp
  have hpow : ε / ((1 - q) ^ 2 * Δ) < q ^ ((2 * t : ℝ) / m) := by
    have hpos : 0 < ε / ((1 - q) ^ 2 * Δ) := by positivity
    rw [← Real.log_lt_log_iff hpos (Real.rpow_pos_of_pos hq _), Real.log_rpow hq,
      Real.log_div hε.ne' (by positivity), ← neg_sub, ← Real.log_div (by positivity) hε.ne']
    have : Real.log (1 / q) = - Real.log q := by rw [one_div, Real.log_inv]
    rw [this] at hexp
    linarith
  rw [div_lt_iff₀ (by positivity)] at hpow
  linarith

/-- Regime `0 < κ - 1 ≤ 8N`, with `q = (κ - 1)/(16 N)`. -/
theorem lower_bound_small_kappa_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ < L) (hL8 : L - μ ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / (L - μ))⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε := by
  have hN : (0 : ℝ) < n * m := by
    have : (0 : ℝ) < n := by exact_mod_cast hn
    have : (0 : ℝ) < m := by exact_mod_cast hm
    positivity
  have hLμ : 0 < L - μ := by linarith
  set q := (L - μ) / (16 * n * m * μ) with hqdef
  have hq : 0 < q := by positivity
  have hq2 : q ≤ 1 / 2 := by
    rw [hqdef, div_le_iff₀ (by positivity)]; linarith
  have hq1 : q < 1 := by linarith
  have h1q : (1 : ℝ) / 4 ≤ (1 - q) ^ 2 := by nlinarith
  have hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2 := by
    have e : 4 * ((n : ℝ) * m) * μ * q = (L - μ) / 4 := by
      rw [hqdef]; field_simp; ring
    rw [e]
    nlinarith
  obtain ⟨d, I, hI, hconv, hT⟩ :=
    lower_bound_chain_SC_aux (c := c) (ρ := ρ) hn hm hμ hΔ hε hq hq1 hqL
  refine ⟨d, I, hI, hconv, fun A hA W => le_trans ?_ (hT A hA W)⟩
  apply Nat.cast_le.2
  apply Nat.ceil_mono
  have hinv : 1 / q = 16 * n * m * μ / (L - μ) := by rw [hqdef]; field_simp
  rw [hinv]
  have hlogq : 0 < Real.log (16 * n * m * μ / (L - μ)) := by
    rw [← hinv]; exact Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have hlog : Real.log (Δ / (4 * ε)) ≤ Real.log ((1 - q) ^ 2 * Δ / ε) := by
    apply Real.log_le_log (by positivity)
    rw [div_le_div_iff₀ (by positivity) hε]
    have := mul_le_mul_of_nonneg_right h1q (by positivity : (0 : ℝ) ≤ 4 * Δ * ε)
    nlinarith
  have hm' : (0 : ℝ) ≤ m := by positivity
  apply div_le_div_of_nonneg_right _ hlogq.le
  exact mul_le_mul_of_nonneg_left hlog (by positivity)

/-- Regime `κ - 1 ≥ 16 N`, with `q = 1 - 2 √(N/(κ-1))`. -/
theorem lower_bound_large_kappa_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hL16 : 16 * n * m * μ ≤ L - μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 8 * Real.sqrt ((L - μ) / (n * m * μ)) *
            Real.log (4 * n * m * μ * Δ / ((L - μ) * ε))⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  have hN : (0 : ℝ) < n * m := by
    have : (0 : ℝ) < n := by exact_mod_cast hn
    have : (0 : ℝ) < m := by exact_mod_cast hm
    positivity
  have hNμ : (0 : ℝ) < n * m * μ := by positivity
  have hLμ : 0 < L - μ := by nlinarith
  set r := n * m * μ / (L - μ) with hrdef
  have hr : 0 < r := by positivity
  have hr16 : r ≤ 1 / 16 := by
    rw [hrdef, div_le_iff₀ hLμ]; nlinarith
  set s := Real.sqrt r with hsdef
  have hs : 0 < s := Real.sqrt_pos.2 hr
  have hs2 : s ^ 2 = r := Real.sq_sqrt hr.le
  have hs4 : s ≤ 1 / 4 := by
    rw [hsdef]
    calc Real.sqrt r ≤ Real.sqrt (1 / 16) := Real.sqrt_le_sqrt hr16
      _ = 1 / 4 := by
        rw [show (1 : ℝ) / 16 = (1 / 4) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  set q := 1 - 2 * s with hqdef
  have hq : 0 < q := by linarith
  have hq2 : 1 / 2 ≤ q := by linarith
  have hq1 : q < 1 := by linarith
  have h1q2 : (1 - q) ^ 2 = 4 * r := by rw [hqdef]; nlinarith
  have hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2 := by
    rw [h1q2, hrdef]
    have : (L - μ) * (4 * (n * m * μ / (L - μ))) = 4 * (n * m * μ) := by field_simp
    rw [this]
    nlinarith
  obtain ⟨d, I, hI, hconv, hT⟩ :=
    lower_bound_chain_SC_aux (c := c) (ρ := ρ) hn hm hμ hΔ hε hq hq1 hqL
  refine ⟨d, I, hI, hconv, fun A hA W => le_trans ?_ (hT A hA W)⟩
  apply Nat.cast_le.2
  -- rewrite the target in terms of `s` and `A := log((1-q)²Δ/ε)`
  have hA : Real.log (4 * n * m * μ * Δ / ((L - μ) * ε)) = Real.log ((1 - q) ^ 2 * Δ / ε) := by
    congr 1
    rw [h1q2, hrdef]
    field_simp
  have hsqrt : Real.sqrt ((L - μ) / (n * m * μ)) = s⁻¹ := by
    rw [hsdef, hrdef, ← Real.sqrt_inv, inv_div]
  rw [hA, hsqrt]
  set A := Real.log ((1 - q) ^ 2 * Δ / ε)
  have hlogq : 0 < Real.log (1 / q) := Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have hlogq_le : Real.log (1 / q) ≤ 4 * s := by
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < 1 / q)
    have h2 : 1 / q - 1 ≤ 4 * s := by
      rw [div_sub_one hq.ne', div_le_iff₀ hq]
      rw [hqdef]; nlinarith
    linarith
  have hm' : (0 : ℝ) ≤ m := by positivity
  rcases le_or_gt A 0 with hneg | hpos
  · have : (m : ℝ) / 8 * s⁻¹ * A ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by positivity) hneg
    rw [Nat.ceil_eq_zero.2 this]
    exact Nat.zero_le _
  · apply Nat.ceil_mono
    calc (m : ℝ) / 8 * s⁻¹ * A = (m : ℝ) / 2 * A / (4 * s) := by field_simp; ring
      _ ≤ (m : ℝ) / 2 * A / Real.log (1 / q) :=
        div_le_div_of_nonneg_left (by positivity) hlogq hlogq_le

/-- A lower bound valid on one instance for all admissible algorithms bounds `minimax`. -/
lemma le_minimax_of {c : ℕ} (hn : 0 < n) {L μ Δ ρ ε : ℝ} (hρ : 0 ≤ ρ) (T : ℕ)
    (h : ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        (T : ℕ∞) ≤ A.hittingTime I W ε) :
    (T : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  obtain ⟨d, I, hI, hT⟩ := h
  unfold minimax
  refine le_iInf₂ fun A hA => ?_
  refine le_trans (hT A hA (J n)) ?_
  exact le_iSup_of_le d (le_iSup_of_le I (le_iSup_of_le hI (le_iSup_of_le (J n)
    (le_iSup_of_le (J_commClass hn hρ) le_rfl))))

end Chain
end ULB63

namespace ULB63
namespace Chain

variable {n m : ℕ}

/-- The threshold computation behind the chain bound. -/
lemma chain_threshold (hm : 0 < m) {Δ ε q : ℝ} (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (t : ℕ) (htT : (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)) :
    ε < Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m) := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1q : 0 < 1 - q := by linarith
  have hlogq : 0 < Real.log (1 / q) := Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have hexp : (2 * t : ℝ) / m * Real.log (1 / q) < Real.log ((1 - q) ^ 2 * Δ / ε) := by
    rw [lt_div_iff₀ hlogq] at htT
    have : (2 * t : ℝ) / m * Real.log (1 / q) = 2 / m * (t * Real.log (1 / q)) := by ring
    rw [this]
    calc 2 / m * (t * Real.log (1 / q)) < 2 / m * (m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε)) :=
          mul_lt_mul_of_pos_left htT (by positivity)
      _ = Real.log ((1 - q) ^ 2 * Δ / ε) := by field_simp
  have hpow : ε / ((1 - q) ^ 2 * Δ) < q ^ ((2 * t : ℝ) / m) := by
    have hpos : 0 < ε / ((1 - q) ^ 2 * Δ) := by positivity
    rw [← Real.log_lt_log_iff hpos (Real.rpow_pos_of_pos hq _), Real.log_rpow hq,
      Real.log_div hε.ne' (by positivity), ← neg_sub, ← Real.log_div (by positivity) hε.ne']
    have : Real.log (1 / q) = - Real.log q := by rw [one_div, Real.log_inv]
    rw [this] at hexp
    linarith
  rw [div_lt_iff₀ (by positivity)] at hpow
  linarith

/-- Pathwise chain bound with `μ`-strongly convex components: before the bound, the gap of the
averaged output exceeds `ε` for every seed. -/
theorem lower_bound_chain_pathwise_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧
      (∀ i ξ, StrongConvexOn Set.univ μ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ, ∀ ω : A.Ω,
        ∀ t : ℕ, (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q) →
          ENNReal.ofReal ε < I.gap (A.avgOutput I W ω t) := by
  set T := ⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊
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
  refine lt_of_lt_of_le (chain_threshold hm hΔ hε hq hq1 t ht) ?_
  have h := amgm hm hq hq1 k t hk
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hc : 0 ≤ Δ * (1 - q) ^ 2 / m := by positivity
  calc Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)
      = Δ * (1 - q) ^ 2 / m * (m * q ^ ((2 * t : ℝ) / m)) := by field_simp
    _ ≤ Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := mul_le_mul_of_nonneg_left h hc

end Chain
end ULB63

namespace ULB63
namespace Chain

variable {n m : ℕ}

/-- Regime `1 ≤ κ ≤ 8N` with convex components, `q = κ/(16 N)`. -/
theorem lower_bound_small_kappa_convex_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hL8 : L ≤ 8 * n * m * μ) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log (Δ / (4 * ε)) / Real.log (16 * n * m * μ / L)⌉₊ : ℕ) : ℕ∞)
          ≤ A.hittingTime I W ε := by
  have hN : (0 : ℝ) < n * m := by
    have : (0 : ℝ) < n := by exact_mod_cast hn
    have : (0 : ℝ) < m := by exact_mod_cast hm
    positivity
  have hL : 0 < L := lt_of_lt_of_le hμ hμL
  set q := L / (16 * n * m * μ) with hqdef
  have hq : 0 < q := by positivity
  have hq2 : q ≤ 1 / 2 := by
    rw [hqdef, div_le_iff₀ (by positivity)]; linarith
  have hq1 : q < 1 := by linarith
  have h1q : (1 : ℝ) / 4 ≤ (1 - q) ^ 2 := by nlinarith
  have hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2 := by
    have e : 4 * ((n : ℝ) * m) * μ * q = L / 4 := by
      rw [hqdef]; field_simp; ring
    rw [e]
    nlinarith
  obtain ⟨d, I, hI, hconv, hT⟩ :=
    lower_bound_chain_aux (c := c) (ρ := ρ) hn hm hμ hμL hΔ hε hq hq1 hqL
  refine ⟨d, I, hI, hconv, fun A hA W => le_trans ?_ (hT A hA W)⟩
  apply Nat.cast_le.2
  apply Nat.ceil_mono
  have hinv : 1 / q = 16 * n * m * μ / L := by rw [hqdef]; field_simp
  rw [hinv]
  have hlogq : 0 < Real.log (16 * n * m * μ / L) := by
    rw [← hinv]; exact Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have hlog : Real.log (Δ / (4 * ε)) ≤ Real.log ((1 - q) ^ 2 * Δ / ε) := by
    apply Real.log_le_log (by positivity)
    rw [div_le_div_iff₀ (by positivity) hε]
    have := mul_le_mul_of_nonneg_right h1q (by positivity : (0 : ℝ) ≤ 4 * Δ * ε)
    nlinarith
  apply div_le_div_of_nonneg_right _ hlogq.le
  exact mul_le_mul_of_nonneg_left hlog (by positivity)

end Chain
end ULB63
