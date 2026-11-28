import Theory.LocalClass
import Theory.TwoNodeRun

/-! The restarted algorithm uses only declared tuning information and the local row. -/

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace ULB63
open Acceleration

def epochCount (Δ ε : ℝ) : ℕ := ⌈Real.log (Δ/ε) / Real.log 2⌉₊

theorem epochCount_correct {Δ ε : ℝ} (hΔ : 0 < Δ) (hε : 0 < ε) :
    (1/2 : ℝ)^(epochCount Δ ε) * Δ ≤ ε := by
  let K := epochCount Δ ε
  have hlog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hk : Real.log (Δ/ε) ≤ (K : ℝ)*Real.log 2 := (div_le_iff₀ hlog2).1 (Nat.le_ceil _)
  have he : -(K : ℝ)*Real.log 2 ≤ Real.log (ε/Δ) := by
    rw [Real.log_div hε.ne' hΔ.ne']
    rw [Real.log_div hΔ.ne' hε.ne'] at hk
    linarith only [hk]
  have hh := Real.exp_le_exp.mpr he
  rw [Real.exp_log (by positivity : 0 < ε/Δ)] at hh
  have he' : Real.exp (-(K : ℝ)*Real.log 2) = (1/2 : ℝ)^K := by
    rw [neg_mul, ← mul_neg, Real.exp_nat_mul, Real.exp_neg, Real.exp_log (by norm_num)]
    simp [one_div]
  rw [he'] at hh
  exact (le_div_iff₀ hΔ).1 hh

namespace TwoNode

theorem certified_time {m : ℕ} (hm : m ≠ 0) {L μ Δ ε : ℝ}
    (hμ : 0 < μ) (hL : 0 < L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∀ (d : ℕ) (I : Instance 2 m d), I.InLocalClass L μ Δ →
      ∀ W : Matrix (Fin 2) (Fin 2) ℝ, (∀ i, W i (other i) ≠ 0) →
      (algorithm m (epochLength L μ) L).hittingTime I W ε ≤
        ((cycle m * (epochLength L μ * epochCount Δ ε) : ℕ) : ℕ∞) := by
  intro d I hI W hW
  let K := epochCount Δ ε
  let p := epochLength L μ
  let A := algorithm m p L
  have hp : 0 < p := epochLength_pos hL hμ
  have hsc (i : Fin 2) : LSmooth L (I.collapsed.localAverage i) := by
    rw [I.collapsed_local_eq hm i]
    exact hI.2.smooth i ⟨0, Nat.pos_of_ne_zero hm⟩
  have hf : LSmooth L I.collapsed.f := fSmooth_of_local (by decide) hsc
  obtain ⟨s, hs, hmin⟩ := feasible_minimizer hI.2
  obtain ⟨hx, hgap⟩ := restart_gap hI.2 hμ hL hf hs hmin K
  have hfinal : I.gap (restart I L p K) ≤ ENNReal.ofReal ε := by
    rw [← I.collapsed_restart_eq hm L p K, ← I.collapsed_gap_eq hm, gap_eq_ofReal hI.2 hs hmin hx]
    apply ENNReal.ofReal_le_ofReal
    exact hgap.trans (epochCount_correct hΔ hε)
  have heq (ω : A.Ω) : A.avgOutput I W ω (cycle m*(p*K)) = restart I L p K := by
    rw [averaged_output_matches I W hp hI.1 hW ω, reference_epoch I hp]
  have hE : A.expectedGap I W (cycle m*(p*K)) ≤ ENNReal.ofReal ε := by
    simpa only [Algorithm.expectedGap, heq, lintegral_const, measure_univ, mul_one] using hfinal
  exact iInf₂_le (cycle m*(p*K)) hE

end TwoNode

end ULB63
