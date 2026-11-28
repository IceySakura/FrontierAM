import Theory.RelaxedLower
import Theory.TwoNodeUpper
import Theory.Disconnected

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof (no `sorry`, no new
axioms). Helper lemmas and other submission modules may be used, including the fixed `Theory/Model.lean` and
`Theory/Sanity.lean`; `Theory.Statements` may not be imported.
-/

open MeasureTheory
open scoped ENNReal

namespace ULB63

open BlockGeometry

theorem minimax_data_no_accuracy_prefactor {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hρ : 0 ≤ ρ) (_hρ1 : ρ < 1) (hε : 0 < ε) (_hεΔ : ε < Δ) :
    let q := dataRatio (L / (5 * ((n : ℝ) * m))) μ
    (⌈max ((m : ℝ) * (1 - ε / Δ))
        ((m : ℝ) * Real.log (Δ / ε) / (2 * Real.log (1 / q)))⌉₊ : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  let l := L / (5 * ((n : ℝ) * m))
  have hnR : 0 < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hmR : 0 < (m : ℝ) := by exact_mod_cast (by omega : 0 < m)
  have hpos : 0 < L := hμ.trans_le hL
  have hl : 0 < l := by dsimp [l]; positivity
  have hscaleEq : ((n : ℝ) * m) * (5 * l) = L := by dsimp [l]; field_simp
  have hscale := hscaleEq.le
  obtain ⟨d, I, hI, _, hhard⟩ :=
    block_hard_instance (c := c) (L := L) (ρ := ρ) (by omega : n ≠ 0) (by omega : m ≠ 0)
      hl hμ hscale hΔ hε
  dsimp only
  refine le_iInf fun A => le_iInf fun hA => ?_
  exact (hhard A hA (J n)).trans (le_iSup_of_le d (le_iSup_of_le I (le_iSup_of_le hI
    (le_iSup_of_le (J n) (le_iSup_of_le (J_commClass (by omega) hρ) le_rfl)))))

theorem minimax_data_high_condition {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : 5 * ((n : ℝ) * m) * μ ≤ L) (hΔ : 0 < Δ)
    (hρ : 0 ≤ ρ) (hρ1 : ρ < 1) (hε : 0 < ε) (hεΔ : ε ≤ Δ / 2) :
    (⌈(m : ℝ) / 4 +
        (m : ℝ) * Real.sqrt (L / (μ * ((n : ℝ) * m))) * Real.log (Δ / ε) / 20⌉₊ : ℕ∞) ≤
      minimax n m c L μ Δ ρ ε := by
  let N := (n : ℝ) * m
  let l := L / (5 * N)
  have hnR : 1 ≤ (n : ℝ) := by exact_mod_cast (by omega : 1 ≤ n)
  have hmR : 1 ≤ (m : ℝ) := by exact_mod_cast hm
  have hN : 0 < N := by dsimp [N]; positivity
  have hl : μ ≤ l := (le_div_iff₀ (by positivity)).2 (by simpa [mul_comm] using hL)
  have hN1 : 1 ≤ N := by simpa only [one_mul] using mul_le_mul hnR hmR zero_le_one (by positivity)
  have hN5 : 1 ≤ 5 * N := by linarith
  have hL' : μ ≤ L := by
    calc
      μ = 1 * μ := (one_mul _).symm
      _ ≤ 5 * ((n : ℝ) * m) * μ := mul_le_mul_of_nonneg_right hN5 hμ.le
      _ ≤ L := hL
  have he : 5 * (l/μ) = L / (μ * N) := by dsimp [l]; field_simp
  have hs' := calibrated_high_bound (m := m) hμ hl hΔ hε hεΔ
  simp only [he] at hs'
  exact (ENat.natCast_le_natCast.mpr (Nat.ceil_le_ceil hs')).trans
    (minimax_data_no_accuracy_prefactor hn hm hμ hL' hΔ hρ hρ1 hε (by linarith))

theorem local_average_data_lower {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε θ : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hθ1 : 1 ≤ θ) (hθm : θ ≤ (m : ℝ)) (hε : 0 < ε) (_hεΔ : ε < Δ) :
    let q := dataRatio (θ * L / (5 * ((n : ℝ) * m))) μ
    let B := ⌈max ((m : ℝ) * (1 - ε / Δ))
      ((m : ℝ) * Real.log (Δ / ε) / (2 * Real.log (1 / q)))⌉₊
    ∃ (d : ℕ) (I : Instance n m d), I.InClass (θ * L) μ Δ ∧
      (∀ i, LSmooth L (I.localAverage i)) ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ ρ →
        ∀ W : Matrix (Fin n) (Fin n) ℝ, (B : ℕ∞) ≤ A.hittingTime I W ε :=
  local_smooth_quantitative hn hm hμ hL hΔ hθ1 hθm hε

theorem local_average_only_breaks_hub (c : ℕ) (C : ℝ) (hC : 0 < C) (s : ℕ) :
    ∃ (m : ℕ), 10 ≤ m ∧
      ∃ (d : ℕ) (I : Instance 2 m d), I.InClass ((m : ℝ)^2) 1 1 ∧
        (∀ i, LSmooth (m : ℝ) (I.localAverage i)) ∧
        ∀ (A : Algorithm 2 m c), A.Admissible (m : ℝ) 1 1 0 →
          (⌈C * (m : ℝ)^(5/4 : ℝ) * (1 + Real.log m)^s⌉₊ : ℕ∞) <
            A.hittingTime I (J 2) (1 / (m : ℝ)) := by
  obtain ⟨m, hm10, hsep⟩ := (eventually_separated C hC s).exists
  have hmR : (10 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm10
  have hm0 : 0 < (m : ℝ) := by linarith
  obtain ⟨d, I, hI, hlocal, hhard⟩ := local_smooth_high (n := 2) (m := m) (c := c)
    (ρ := 0) (μ := 1) (Δ := 1) (θ := (m:ℝ)) (L := (m:ℝ)) (ε := 1 / (m:ℝ))
    (by omega) (by omega) one_pos (by linarith) one_pos (by linarith) le_rfl
    (by push_cast; nlinarith [mul_nonneg hm0.le (sub_nonneg.mpr hmR)])
    (by positivity) ((div_le_iff₀ hm0).2 (by linarith))
  refine ⟨m, hm10, d, I, (by simpa [pow_two] using hI), hlocal, fun A hA => ?_⟩
  have hh := hhard A hA (J 2)
  have hratio : (m : ℝ) * m / (1 * ((↑(2 : ℕ) : ℝ) * m)) = (m : ℝ)/2 := by
    push_cast
    field_simp
  rw [hratio, one_div_div] at hh
  simp only [div_one] at hh
  apply lt_of_lt_of_le _ hh
  apply ENat.natCast_lt_natCast.mpr
  apply Nat.lt_ceil.mpr
  have hceil : (⌈C * (m : ℝ)^(5/4 : ℝ) * (1+Real.log m)^s⌉₊ : ℝ) <
      C * (m : ℝ)^(5/4 : ℝ) * (1+Real.log m)^s + 1 :=
    Nat.ceil_lt_add_one (by positivity [Real.log_nonneg (by linarith : (1:ℝ) ≤ (m:ℝ))])
  exact hceil.trans hsep

theorem two_node_local_average_upper {m : ℕ} (hm : 1 ≤ m)
    {L μ Δ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ A : Algorithm 2 m 1, (∀ M ρ, A.Admissible M μ Δ ρ) ∧
      ∀ (d : ℕ) (I : Instance 2 m d), I.InLocalClass L μ Δ →
        ∀ W : Matrix (Fin 2) (Fin 2) ℝ, W 0 1 ≠ 0 → W 1 0 ≠ 0 →
          A.hittingTime I W ε ≤
            (((m+2) * ⌈Real.sqrt (16*(L/μ))⌉₊ * ⌈Real.log (Δ/ε)/Real.log 2⌉₊ : ℕ) : ℕ∞) := by
  have hL0 := hμ.trans_le hL
  refine ⟨TwoNode.algorithm m (Acceleration.epochLength L μ) L,
    fun _ _ => TwoNode.algorithm_admissible hL0, ?_⟩
  intro d I hI W h₁ h₂
  have hW (i : Fin 2) : W i (TwoNode.other i) ≠ 0 := by
    fin_cases i
    · exact h₁
    · exact h₂
  simpa only [TwoNode.cycle, Acceleration.epochLength, epochCount, Nat.mul_assoc] using
    TwoNode.certified_time (by omega : m ≠ 0) hμ hL0 hΔ hε d I hI W hW

theorem two_node_local_average_lower {m c : ℕ} (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε)
    (_hεΔ : ε < Δ) :
    let q := dataRatio (L/10) μ
    ∃ (d : ℕ) (I : Instance 2 m d), I.InLocalClass L μ Δ ∧
      ∀ (A : Algorithm 2 m c), A.Admissible L μ Δ ρ →
        (⌈max ((m : ℝ) * (1 - ε/Δ)) ((m : ℝ) * Real.log (Δ/ε) / (2*Real.log (1/q)))⌉₊ : ℕ∞) ≤
          A.hittingTime I (J 2) ε := by
  have hmR : (1 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
  have hm' : (m : ℝ) ≠ 0 := (zero_lt_one.trans_le hmR).ne'
  obtain ⟨d, I, hI, hloc, hh⟩ := local_smooth_quantitative (n := 2) (m := m) (c := c) (ρ := ρ)
    (by omega) hm hμ hL hΔ hmR le_rfl hε
  have he : (m : ℝ) * L / (5 * (((2 : ℕ) : ℝ) * m)) = L/10 := by push_cast; field_simp; ring
  rw [he] at hh
  exact ⟨d, I, I.inLocalClass_of_local (by omega) hI hloc, fun A hA => hh A hA (J 2)⟩

theorem two_node_standard_upper {m : ℕ} (hm : 1 ≤ m)
    {L μ Δ ρ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hρ : ρ < 1) :
    minimax 2 m 1 L μ Δ ρ ε ≤
      (((m+2) * ⌈Real.sqrt (16*(L/μ))⌉₊ * ⌈Real.log (Δ/ε)/Real.log 2⌉₊ : ℕ) : ℕ∞) := by
  obtain ⟨A, hA, hupper⟩ := two_node_local_average_upper hm hμ hL hΔ hε
  refine (iInf₂_le A (hA L ρ)).trans ?_
  refine iSup_le fun d => iSup_le fun I => iSup_le fun hI => iSup_le fun W => iSup_le fun hW => ?_
  exact hupper d I (I.inLocalClass_of_standard (by omega) hI) W
    (TwoNode.offdiag_ne_zero hW hρ 0) (TwoNode.offdiag_ne_zero hW hρ 1)

theorem minimax_infinite_at_disconnected_endpoint {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ε : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    let l := ((n : ℝ) - 1) * L / (2 * (n : ℝ))
    ε < Δ * (l / (μ + l))^2 → minimax n m c L μ Δ 1 ε = ⊤ := by
  intro l hsmall
  let src : Fin n := ⟨0, by omega⟩
  obtain ⟨I, hI, hhard⟩ := Disconnected.never_hits (c := c) (by omega : n ≠ 0) (by omega : m ≠ 0)
    (hμ.trans_le hL).le hμ hΔ.le hε hsmall src
  apply top_unique
  refine le_iInf fun A => le_iInf fun hA => ?_
  refine le_iSup_of_le 2 (le_iSup_of_le I (le_iSup_of_le hI (le_iSup_of_le 1
    (le_iSup_of_le (identity_commClass (by omega : n ≠ 0)) ?_))))
  rw [hhard A hA]

end ULB63
