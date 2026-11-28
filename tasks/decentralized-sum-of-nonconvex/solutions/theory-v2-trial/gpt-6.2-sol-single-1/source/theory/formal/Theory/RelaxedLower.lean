import Theory.BlockLower

/-! Quantifying the failure of the H-UB order under local-average-only smoothness. -/

open Filter
open scoped ENNReal Topology

noncomputable section

namespace ULB63
open BlockGeometry

theorem local_smooth_quantitative {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε θ : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hθ1 : 1 ≤ θ) (hθm : θ ≤ (m : ℝ)) (hε : 0 < ε) :
    let q := dataRatio (θ * L / (5 * ((n : ℝ) * m))) μ
    let B := ⌈max ((m : ℝ) * (1 - ε / Δ))
      ((m : ℝ) * Real.log (Δ / ε) / (2 * Real.log (1 / q)))⌉₊
    ∃ (d : ℕ) (I : Instance n m d), I.InClass (θ * L) μ Δ ∧
      (∀ i, LSmooth L (I.localAverage i)) ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ ρ →
        ∀ W : Matrix (Fin n) (Fin n) ℝ, (B : ℕ∞) ≤ A.hittingTime I W ε := by
  let l := θ * L / (5 * ((n : ℝ) * m))
  have hnR : 0 < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hmR : 0 < (m : ℝ) := by exact_mod_cast (by omega : 0 < m)
  have hL0 := hμ.trans_le hL
  have hθ0 : 0 < θ := zero_lt_one.trans_le hθ1
  have hl : 0 < l := by dsimp [l]; positivity
  have hscaleEq : ((n : ℝ) * m) * (5 * l) = θ * L := by dsimp [l]; field_simp
  have hscale := hscaleEq.le
  obtain ⟨d, I, hI, hlocal, hhard⟩ :=
    block_hard_instance (c := c) (L := L) (ρ := ρ) (by omega : n ≠ 0) (by omega : m ≠ 0)
      hl hμ hscale hΔ hε
  refine ⟨d, I, hI, fun i => BlockGeometry.LSmooth.weaken (hlocal i) ?_, hhard⟩
  have he : 5 * (n : ℝ) * l = θ * L / m := by dsimp [l]; field_simp
  rw [he]
  exact (div_le_iff₀ hmR).2 (mul_le_mul_of_nonneg_right hθm hL0.le |>.trans_eq (mul_comm _ _))

theorem local_smooth_high {n m c : ℕ} (hn : 2 ≤ n) (hm : 1 ≤ m)
    {L μ Δ ρ ε θ : ℝ} (hμ : 0 < μ) (hL : μ ≤ L) (hΔ : 0 < Δ)
    (hθ1 : 1 ≤ θ) (hθm : θ ≤ (m : ℝ)) (hcond : 5 * ((n : ℝ) * m) * μ ≤ θ * L)
    (hε : 0 < ε) (hεΔ : ε ≤ Δ / 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass (θ * L) μ Δ ∧
      (∀ i, LSmooth L (I.localAverage i)) ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ ρ →
        ∀ W : Matrix (Fin n) (Fin n) ℝ,
          (⌈(m : ℝ)/4 + (m : ℝ) * Real.sqrt (θ * L / (μ * ((n : ℝ) * m))) *
            Real.log (Δ/ε) / 20⌉₊ : ℕ∞) ≤ A.hittingTime I W ε := by
  let N := (n : ℝ)*m
  let l := θ*L/(5*N)
  have hnR : 0 < (n : ℝ) := by exact_mod_cast (by omega : 0 < n)
  have hmR : 0 < (m : ℝ) := by exact_mod_cast (by omega : 0 < m)
  have hN : 0 < N := by dsimp [N]; positivity
  have hl : μ ≤ l := (le_div_iff₀ (by positivity)).2 (by simpa [mul_comm] using hcond)
  obtain ⟨d, I, hI, hlocal, hhard⟩ := local_smooth_quantitative (c := c) (ρ := ρ)
    hn hm hμ hL hΔ hθ1 hθm hε
  refine ⟨d, I, hI, hlocal, fun A hA W => le_trans ?_ (hhard A hA W)⟩
  have he : 5*(l/μ) = θ*L/(μ*N) := by dsimp [l]; field_simp
  have hs := calibrated_high_bound (m := m) hμ hl hΔ hε hεΔ
  simp only [he] at hs
  exact ENat.natCast_le_natCast.mpr (Nat.ceil_le_ceil hs)

/-- The growth comparison for the explicit two-node family; the extra unit handles ceilings. -/
theorem eventually_separated (C : ℝ) (hC : 0 < C) (s : ℕ) :
    ∀ᶠ m : ℕ in atTop, 10 ≤ m ∧
      C * (m : ℝ) ^ (5 / 4 : ℝ) * (1 + Real.log m)^s + 1 <
        (m : ℝ)/4 + (m : ℝ) * Real.sqrt ((m : ℝ) / 2) * Real.log m / 20 := by
  have hsmall := (isLittleO_log_rpow_rpow_atTop (s : ℝ) (by norm_num : 0 < (1/4 : ℝ))).bound
    (show 0 < 1 / (80 * C * 2^s) by positivity)
  have hw : ∀ᶠ x : ℝ in atTop,
      10 ≤ x ∧ 1 ≤ Real.log x ∧
      ‖Real.log x ^ (s : ℝ)‖ ≤ 1 / (80 * C * 2^s) * ‖x ^ (1/4 : ℝ)‖ :=
    (eventually_ge_atTop 10).and ((Real.tendsto_log_atTop.eventually_ge_atTop 1).and hsmall)
  filter_upwards [tendsto_natCast_atTop_atTop.eventually hw] with m hm
  obtain ⟨hm10, hlog, hsmall'⟩ := hm
  have hm0 : 0 < (m : ℝ) := by linarith
  have hm10' : 10 ≤ m := by exact_mod_cast hm10
  refine ⟨hm10', ?_⟩
  have hnonlog := zero_le_one.trans hlog
  have hp : (1 + Real.log (m:ℝ))^s ≤ 2^s * (Real.log (m:ℝ))^s := by
    rw [← mul_pow]
    exact pow_le_pow_left₀ (by positivity) (by linarith) _
  rw [Real.rpow_natCast, Real.norm_of_nonneg (by positivity),
    Real.norm_of_nonneg (by positivity)] at hsmall'
  have haux : C * (1 + Real.log (m:ℝ))^s ≤ (m:ℝ)^(1/4 : ℝ) / 80 := calc
    _ ≤ C * (2^s * (Real.log (m:ℝ))^s) := mul_le_mul_of_nonneg_left hp hC.le
    _ ≤ C * (2^s * (1 / (80 * C * 2^s) * (m:ℝ)^(1/4 : ℝ))) := by gcongr
    _ = _ := by field_simp
  have hk : C * (m : ℝ)^(5/4 : ℝ) * (1+Real.log (m:ℝ))^s ≤
      (m:ℝ) * Real.sqrt (m:ℝ) / 80 := by
    calc
      _ = (m:ℝ)^(5/4 : ℝ) * (C * (1+Real.log (m:ℝ))^s) := by ring
      _ ≤ (m:ℝ)^(5/4 : ℝ) * ((m:ℝ)^(1/4 : ℝ) / 80) := by gcongr
      _ = _ := by
        rw [← mul_div_assoc, ← Real.rpow_add hm0]
        have he : (m:ℝ) * Real.sqrt (m:ℝ) = (m:ℝ)^(1 + 1/2 : ℝ) := by
          rw [Real.sqrt_eq_rpow, Real.rpow_add hm0, Real.rpow_one]
        rw [he]
        norm_num
  have hsqrt : Real.sqrt (m:ℝ) ≤ 2 * Real.sqrt ((m:ℝ)/2) := by
    apply Real.sqrt_le_iff.2
    refine ⟨by positivity, ?_⟩
    rw [mul_pow, Real.sq_sqrt (by positivity)]
    linarith
  have hfinal : (m:ℝ) * Real.sqrt (m:ℝ) / 80 ≤
      (m:ℝ) * Real.sqrt ((m:ℝ)/2) * Real.log m / 20 := by
    have h₁ := mul_le_mul_of_nonneg_left hsqrt hm0.le
    have h₂ := mul_le_mul_of_nonneg_left hlog
      (show 0 ≤ (m:ℝ) * Real.sqrt ((m:ℝ)/2) by positivity)
    nlinarith only [h₁, h₂, show 0 ≤ (m:ℝ) * Real.sqrt ((m:ℝ)/2) by positivity]
  linarith only [hk.trans hfinal, hm10]

end ULB63
