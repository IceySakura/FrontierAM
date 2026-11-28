import Theory.BlockRun
import Theory.LocalSmoothness

/-! The data lower bound, including the accuracy threshold and the initial scan term. -/

open MeasureTheory Finset
open scoped BigOperators ENNReal

noncomputable section

namespace ULB63

/-- The decay factor of a path with coupling `l` and quadratic regularization `μ`. -/
def dataRatio (l μ : ℝ) : ℝ :=
  let r := (Real.sqrt (1 + 4 * (l / μ)) - 1) / 2
  r / (1 + r)

namespace BlockGeometry

theorem ratio_properties {l μ : ℝ} (hl : 0 < l) (hμ : 0 < μ) :
    0 < dataRatio l μ ∧ dataRatio l μ < 1 ∧ μ = curvature l (dataRatio l μ) := by
  let r := (Real.sqrt (1 + 4 * (l / μ)) - 1) / 2
  have hx : 0 < l / μ := div_pos hl hμ
  have hsq := Real.sq_sqrt (by positivity : 0 ≤ 1 + 4 * (l / μ))
  have hr : 0 < r := by
    have hs : 1 < Real.sqrt (1 + 4 * (l / μ)) := by
      rw [← Real.sqrt_one]
      exact Real.sqrt_lt_sqrt (by norm_num) (by linarith)
    unfold r
    linarith
  have hr' : 0 < 1 + r := by positivity
  have he : l = μ * (r * (1 + r)) := by
    have h : r * (1 + r) = l / μ := by
      dsimp [r]
      nlinarith only [hsq]
    rw [h]
    field_simp
  have hq0 : 0 < r / (1 + r) := div_pos hr hr'
  have hq1 : r / (1 + r) < 1 := (div_lt_one hr').2 (by linarith)
  change 0 < r / (1 + r) ∧ r / (1 + r) < 1 ∧ μ = curvature l (r / (1 + r))
  refine ⟨hq0, hq1, ?_⟩
  rw [curvature, he]
  field_simp
  ring

variable {n m c p : ℕ} {l q μ a L Δ ρ ε : ℝ}

/-- Finite sums over the blocks have the uniform startup and exponential lower envelopes. -/
theorem block_expected_envelopes (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 < l) (hq0 : 0 < q) (hq1 : q < 1) (hμ : μ = curvature l q)
    (hΔ : 0 < Δ) (hscale : ((n : ℝ) * m) * (stiffness l q / 2 * a^2) = Δ)
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) (W : Matrix (Fin n) (Fin n) ℝ)
    (t : ℕ) (ht : t ≤ p) :
    ENNReal.ofReal (Δ * (1 - (t : ℝ) / m)) ≤ A.expectedGap (blockInstance l μ q a n m p) W t ∧
    ENNReal.ofReal (Δ * Real.exp (2 * ((t : ℝ) / m) * Real.log q)) ≤
      A.expectedGap (blockInstance l μ q a n m p) W t := by
  classical
  let I := blockInstance l μ q a n m p
  let N : ℝ := (n : ℝ) * m
  have hnR : 0 < (n : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hn
  have hmR : 0 < (m : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hm
  have hN : 0 < N := mul_pos hnR hmR
  have ha : stiffness l q / 2 * a^2 = Δ / N :=
    (eq_div_iff hN.ne').2 (by simpa [N, mul_comm] using hscale)
  have hcard : ((Fintype.card (Blocks n m) : ℕ) : ℝ) = N := by simp [N]
  have hconvert (ω : A.Ω) :
      (∑ b : Blocks n m, (A.visits I W ω b (microSteps*t) : ℝ)) ≤ (n : ℝ) * t := by
    exact_mod_cast A.sum_visits_le I W ω hA t
  have hpoint (ω : A.Ω) :
      (Δ / N) * ∑ b : Blocks n m, q^(2 * A.visits I W ω b (microSteps*t)) ≤
        ∑ b : Blocks n m, energy l q a b (A.avgOutput I W ω t) := by
    rw [mul_sum, ← ha]
    apply sum_le_sum
    intro b _
    let j : Fin (p + 1) := ⟨A.visits I W ω b (microSteps*t),
      Nat.lt_succ_of_le ((A.visits_le_iter I W ω hA b t).trans ht)⟩
    exact energy_of_zero_coord hl.le hq0 hq1 b (A.avgOutput I W ω t) j
      (block_output_support hA b t j le_rfl)
  have hlinear (ω : A.Ω) : Δ * (1 - (t : ℝ) / m) ≤
      (Δ / N) * ∑ b : Blocks n m, q^(2 * A.visits I W ω b (microSteps*t)) := by
    have ht' : (n : ℝ) * t = N * ((t : ℝ) / m) := by
      dsimp [N]
      field_simp
    have hsum : (N - (n : ℝ) * t) ≤
        ∑ b : Blocks n m, q^(2 * A.visits I W ω b (microSteps*t)) := calc
      _ ≤ N - ∑ b : Blocks n m, (A.visits I W ω b (microSteps*t) : ℝ) :=
        sub_le_sub_left (hconvert ω) _
      _ = ∑ b : Blocks n m, (1 - (A.visits I W ω b (microSteps*t) : ℝ)) := by
        rw [sum_sub_distrib, sum_const, nsmul_eq_mul, card_univ, hcard, mul_one]
      _ ≤ _ := by
        apply sum_le_sum
        intro b _
        generalize A.visits I W ω b (microSteps * t) = k
        cases k with
        | zero => simp
        | succ k =>
          have hk : (1 : ℝ) ≤ (k.succ : ℝ) := by exact_mod_cast Nat.succ_pos k
          exact (sub_nonpos.mpr hk).trans (by positivity)
    have he : Δ * (1 - (t : ℝ) / m) = (Δ / N) * (N - (n : ℝ) * t) := by
      rw [ht']
      field_simp
    rw [he]
    exact mul_le_mul_of_nonneg_left hsum (by positivity)
  have hexp (ω : A.Ω) : Δ * Real.exp (2 * ((t : ℝ) / m) * Real.log q) ≤
      (Δ / N) * ∑ b : Blocks n m, q^(2 * A.visits I W ω b (microSteps*t)) := by
    let z : Blocks n m → ℝ := fun b => 2 * (A.visits I W ω b (microSteps*t) : ℝ) * Real.log q
    have hz (b : Blocks n m) : Real.exp (z b) = q^(2 * A.visits I W ω b (microSteps*t)) := by
      dsimp [z]
      rw [show 2 * (A.visits I W ω b (microSteps*t) : ℝ) =
        ((2 * A.visits I W ω b (microSteps*t) : ℕ) : ℝ) by simp, Real.exp_nat_mul, Real.exp_log hq0]
    have hw : ∑ _b : Blocks n m, (1 / N) = 1 := by
      rw [sum_const, nsmul_eq_mul, card_univ, hcard]
      field_simp
    have hj := convexOn_exp.map_sum_le (t := (univ : Finset (Blocks n m)))
      (w := fun _ => 1/N) (p := z) (fun _ _ => by positivity) hw (fun _ _ => Set.mem_univ _)
    simp only [smul_eq_mul, ← mul_sum, hz] at hj
    have hsum : (1/N) * ∑ b : Blocks n m, z b =
        (2 * (1/N) * Real.log q) * ∑ b : Blocks n m, (A.visits I W ω b (microSteps*t) : ℝ) := by
      rw [mul_sum, mul_sum]
      exact sum_congr rfl fun b _ => by dsimp [z]; ring
    have htn : 2 * ((t : ℝ) / m) * Real.log q = (2 * (1/N) * Real.log q) * ((n : ℝ) * t) := by
      dsimp [N]
      field_simp
    have hpre : 2 * ((t : ℝ) / m) * Real.log q ≤ (1/N) * ∑ b : Blocks n m, z b := by
      rw [hsum, htn]
      exact mul_le_mul_of_nonpos_left (hconvert ω)
        (mul_nonpos_of_nonneg_of_nonpos (by positivity) (Real.log_nonpos hq0.le hq1.le))
    calc
      _ ≤ Δ * Real.exp ((1/N) * ∑ b : Blocks n m, z b) :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hpre) hΔ.le
      _ ≤ Δ * ((1/N) * ∑ b : Blocks n m, q^(2 * A.visits I W ω b (microSteps*t))) :=
        mul_le_mul_of_nonneg_left hj hΔ.le
      _ = _ := by ring
  have hintegral (k : ℝ) (hk : ∀ ω : A.Ω, k ≤
      ∑ b : Blocks n m, energy l q a b (A.avgOutput I W ω t)) :
      ENNReal.ofReal k ≤ A.expectedGap I W t := by
    have hmono : (∫⁻ _ : A.Ω, ENNReal.ofReal k ∂A.P) ≤ A.expectedGap I W t :=
      lintegral_mono (fun ω => by
        rw [show I.gap (A.avgOutput I W ω t) = _ from
          blockInstance_gap hn hm hl.le hq0 hq1.le hμ _]
        exact ENNReal.ofReal_le_ofReal (hk ω))
    simpa using hmono
  exact ⟨hintegral _ (fun ω => (hlinear ω).trans (hpoint ω)),
    hintegral _ (fun ω => (hexp ω).trans (hpoint ω))⟩

theorem block_hard_instance {Lind : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 < l) (hμ : 0 < μ)
    (hL : ((n : ℝ) * m) * (5 * l) ≤ Lind) (hΔ : 0 < Δ) (hε : 0 < ε) :
    let q := dataRatio l μ
    let B := ⌈max ((m : ℝ) * (1 - ε/Δ))
      ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)))⌉₊
    ∃ (d : ℕ) (I : Instance n m d), I.InClass Lind μ Δ ∧
      (∀ i, LSmooth (5 * (n : ℝ) * l) (I.localAverage i)) ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ ρ →
        ∀ W : Matrix (Fin n) (Fin n) ℝ, (B : ℕ∞) ≤ A.hittingTime I W ε := by
  let q := dataRatio l μ
  obtain ⟨hq0, hq1, hcurv⟩ := ratio_properties hl hμ
  change 0 < q at hq0
  change q < 1 at hq1
  change μ = curvature l q at hcurv
  let B := ⌈max ((m : ℝ) * (1 - ε/Δ))
    ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)))⌉₊
  have hmR : 0 < (m : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hm
  have hnR : 0 < (n : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hn
  have hst : 0 < stiffness l q := by unfold stiffness; positivity [sub_pos.mpr hq1]
  let a := Real.sqrt (Δ / (((n : ℝ) * m) * (stiffness l q / 2)))
  have hscale : ((n : ℝ) * m) * (stiffness l q / 2 * a^2) = Δ := by
    dsimp [a]
    rw [Real.sq_sqrt (by positivity)]
    field_simp
  refine ⟨Dim n m B, blockInstance l μ q a n m B,
    blockInstance_inClass hn hm hl.le hq0 hq1.le hμ hcurv hL hscale.le,
    blockInstance_local_smooth hm hl.le hq0.le hq1.le, ?_⟩
  intro A hA W
  refine le_iInf fun t => le_iInf fun hsuccess => ?_
  suffices B ≤ t by exact_mod_cast this
  by_contra! ht
  obtain ⟨hlin, hexp⟩ := block_expected_envelopes hn hm hl hq0 hq1 hcurv hΔ hscale A hA W t ht.le
  have hl' := (ENNReal.ofReal_le_ofReal_iff hε.le).mp (hlin.trans hsuccess)
  have he' := (ENNReal.ofReal_le_ofReal_iff hε.le).mp (hexp.trans hsuccess)
  apply (not_le_of_gt ht)
  change ⌈max ((m : ℝ) * (1 - ε/Δ))
    ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)))⌉₊ ≤ t
  rw [Nat.ceil_le, max_le_iff]
  constructor
  · have h₁ := (le_div_iff₀ hΔ).2 (by simpa [mul_comm] using hl')
    have h₂ := (le_div_iff₀ hmR).1 (show 1 - ε/Δ ≤ (t : ℝ)/m by linarith only [h₁])
    simpa only [mul_comm] using h₂
  · have h₁ : Real.exp (2 * ((t : ℝ) / m) * Real.log q) ≤ ε/Δ :=
      (le_div_iff₀ hΔ).2 (by simpa only [mul_comm Δ] using he')
    have h₂ := Real.log_le_log (Real.exp_pos _) h₁
    rw [Real.log_exp, Real.log_div hε.ne' hΔ.ne'] at h₂
    have hqinv : Real.log (1/q) = -Real.log q := by
      rw [Real.log_div one_ne_zero hq0.ne', Real.log_one, zero_sub]
    have hden : 0 < 2 * Real.log (1/q) := by rw [hqinv]; linarith [Real.log_neg hq0 hq1]
    apply (div_le_iff₀ hden).2
    rw [hqinv, Real.log_div hΔ.ne' hε.ne']
    have h₃ := mul_le_mul_of_nonneg_left h₂ hmR.le
    have h₄ : (m : ℝ) * (2 * ((t : ℝ) / m) * Real.log q) =
        2 * (t : ℝ) * Real.log q := by field_simp
    rw [h₄] at h₃
    linarith only [h₃]

theorem high_ratio_bound {l μ : ℝ} (hμ : 0 < μ) (h : μ ≤ l) :
    (Real.sqrt (5 * (l/μ))) / 5 ≤
      (((Real.sqrt (1 + 4 * (l/μ)) - 1) / 2) : ℝ) := by
  let r := (Real.sqrt (1 + 4 * (l / μ)) - 1) / 2
  have hx : 1 ≤ l / μ := (one_le_div hμ).2 h
  have hr0 : 0 ≤ r := by
    dsimp [r]
    have h' : 1 ≤ Real.sqrt (1 + 4 * (l / μ)) := by
      rw [← Real.sqrt_one]
      exact Real.sqrt_le_sqrt (by linarith)
    linarith
  have hsq := Real.sq_sqrt (by positivity : 0 ≤ 1 + 4 * (l / μ))
  have he : r * (1 + r) = l / μ := by dsimp [r]; nlinarith only [hsq]
  have hrh : (1/2 : ℝ) ≤ r := by
    by_contra! h'
    nlinarith only [he, hx, h', mul_nonneg hr0 (show 0 ≤ 1/2-r by linarith)]
  have hk : 5 * (l / μ) ≤ (5 * r)^2 := by
    nlinarith only [he, mul_nonneg hr0 (show 0 ≤ 4*r-1 by linarith)]
  apply (div_le_iff₀ (by norm_num : 0 < (5:ℝ))).2
  rw [mul_comm r]
  exact Real.sqrt_le_iff.2 ⟨by positivity, hk⟩

theorem calibrated_high_bound {m : ℕ} {l μ Δ ε : ℝ} (hμ : 0 < μ) (hl : μ ≤ l)
    (hΔ : 0 < Δ) (hε : 0 < ε) (hεΔ : ε ≤ Δ / 2) :
    let q := dataRatio l μ
    (m : ℝ)/4 + (m : ℝ) * Real.sqrt (5 * (l/μ)) * Real.log (Δ/ε) / 20 ≤
      max ((m : ℝ) * (1 - ε/Δ)) ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q))) := by
  let r := (Real.sqrt (1 + 4 * (l/μ)) - 1) / 2
  let q := dataRatio l μ
  change (m : ℝ)/4 + (m : ℝ) * Real.sqrt (5 * (l/μ)) * Real.log (Δ/ε) / 20 ≤
    max ((m : ℝ) * (1 - ε/Δ)) ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)))
  have hs : Real.sqrt (5 * (l/μ)) / 5 ≤ r := high_ratio_bound hμ hl
  have hl0 := hμ.trans_le hl
  have hr : 0 < r := (div_pos (by positivity) (by norm_num : (0:ℝ) < 5)).trans_le hs
  have hqin : 1 / q = 1 + 1 / r := by
    change 1 / (r / (1 + r)) = 1 + 1 / r
    field_simp
    ring
  have hlogq : 0 < Real.log (1/q) := by
    rw [hqin]
    apply Real.log_pos
    linarith [div_pos (by norm_num : (0:ℝ) < 1) hr]
  have hlogqu : Real.log (1/q) ≤ 1/r := by
    rw [hqin]
    linarith [Real.log_le_sub_one_of_pos (show 0 < 1 + 1 / r by positivity)]
  have hRlog : 0 ≤ Real.log (Δ/ε) :=
    Real.log_nonneg ((one_le_div hε).2 (by linarith))
  have hA : (m : ℝ) / 2 ≤ (m : ℝ) * (1 - ε/Δ) := by
    have hε' : ε/Δ ≤ 1/2 := (div_le_iff₀ hΔ).2 (by linarith)
    linarith only [mul_le_mul_of_nonneg_left hε' (by positivity : 0 ≤ (m:ℝ))]
  have hB : (m : ℝ) * Real.sqrt (5 * (l/μ)) * Real.log (Δ/ε) / 10 ≤
      (m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)) := by
    apply (le_div_iff₀ (by positivity)).2
    have hqr := (le_div_iff₀ hr).1 hlogqu
    have hsr : Real.sqrt (5 * (l/μ)) * Real.log (1/q) ≤ 5 := by
      have hh := mul_le_mul_of_nonneg_right ((div_le_iff₀ (by norm_num : (0:ℝ)<5)).1 hs)
        hlogq.le
      nlinarith only [hh, hqr]
    nlinarith only [mul_le_mul_of_nonneg_left hsr
      (show 0 ≤ (m:ℝ) * Real.log (Δ/ε) by positivity)]
  let M := max ((m : ℝ) * (1 - ε/Δ)) ((m : ℝ) * Real.log (Δ/ε) / (2 * Real.log (1/q)))
  have hAM : (m : ℝ) / 2 ≤ M := hA.trans (le_max_left _ _)
  have hBM : (m : ℝ) * Real.sqrt (5 * (l/μ)) * Real.log (Δ/ε) / 10 ≤ M :=
    hB.trans (le_max_right _ _)
  change _ ≤ M
  linarith only [hAM, hBM]

end BlockGeometry

end ULB63
