import Theory.HardMass
import Mathlib.Topology.Order.IntermediateValue

/-! The inverse of the known radial Gaussian CDF on the annulus `[1,2]`. -/

open MeasureTheory Set
open scoped unitInterval

namespace Cubature

noncomputable section

theorem massCDF_local_upper {d : ℕ} (hd : 0 < d) {a b : ℝ}
    (ha : 1 ≤ a) (hab : a ≤ b) (hb : b ≤ 2) :
    massCDF d b - massCDF d a ≤ radialDensityBound d 2 * (b - a) := by
  unfold massCDF
  rw [radialPrimitive_sub (integrable_radialDensity hd) hab]
  have h := intervalIntegral.integral_mono_on (μ := volume) hab
    ((integrable_radialDensity hd).intervalIntegrable)
    (intervalIntegrable_const (c := radialDensityBound d 2))
    (fun t ht => radialDensity_le d (by norm_num) (ht.2.trans hb))
  simpa only [intervalIntegral.integral_const, smul_eq_mul, mul_comm] using h

theorem massCDF_strictMonoOn {d : ℕ} (hd : 0 < d) :
    StrictMonoOn (massCDF d) (Icc 1 2) := by
  intro a ha b hb hlt
  have hlo := massCDF_local_lower hd ha.1 hlt.le hb.2
  have hpos := mul_pos (radialLower_pos hd) (sub_pos.mpr hlt)
  linarith

theorem massCDF_absdiff {d : ℕ} (hd : 0 < d) {a b : ℝ}
    (ha : a ∈ Icc 1 2) (hb : b ∈ Icc 1 2) :
    |massCDF d b - massCDF d a| ≤ radialDensityBound d 2 * |b-a| := by
  wlog h : a ≤ b generalizing a b
  · have hh := this hb ha (le_of_not_ge h)
    rwa [abs_sub_comm (massCDF d a), abs_sub_comm a] at hh
  rw [abs_of_nonneg (sub_nonneg.mpr ((massCDF_strictMonoOn hd).monotoneOn ha hb h)),
    abs_of_nonneg (sub_nonneg.mpr h)]
  exact massCDF_local_upper hd ha.1 h hb.2

theorem massCDF_continuousOn {d : ℕ} (hd : 0 < d) :
    ContinuousOn (massCDF d) (Icc 1 2) := by
  have hL : LipschitzOnWith (Real.toNNReal (radialDensityBound d 2)) (massCDF d) (Icc 1 2) := by
    apply LipschitzOnWith.of_dist_le_mul
    intro x hx y hy
    have hW : 0 ≤ radialDensityBound d 2 := radialDensityBound_nonneg d (by norm_num)
    rw [Real.dist_eq, Real.dist_eq, Real.coe_toNNReal _ hW]
    exact massCDF_absdiff hd hy hx
  exact hL.continuousOn

theorem measurable_massCDF {d : ℕ} (hd : 0 < d) : Measurable (massCDF d) :=
  measurable_radialPrimitive (integrable_radialDensity hd) (radialDensity_nonneg d)

def cdfSpan (d : ℕ) : ℝ := massCDF d 2 - massCDF d 1

theorem cdfSpan_pos {d : ℕ} (hd : 0 < d) : 0 < cdfSpan d :=
  sub_pos.mpr ((massCDF_strictMonoOn hd) (by norm_num) (by norm_num) (by norm_num))

theorem cdfSpan_le {d : ℕ} (hd : 0 < d) : cdfSpan d ≤ radialDensityBound d 2 := by
  simpa only [cdfSpan, show (2:ℝ) - 1 = 1 by norm_num, mul_one] using
    massCDF_local_upper hd (a := 1) (b := 2) (by norm_num) (by norm_num) (by norm_num)

def toMass (d : ℕ) (r : ℝ) : ℝ := (massCDF d r - massCDF d 1) / cdfSpan d

theorem toMass_mem {d : ℕ} (hd : 0 < d) {r : ℝ} (hr : r ∈ Icc 1 2) :
    toMass d r ∈ I := by
  have hS := cdfSpan_pos hd
  have hM := (massCDF_strictMonoOn hd).monotoneOn
  refine ⟨div_nonneg (sub_nonneg.mpr (hM (by norm_num) hr hr.1)) hS.le, ?_⟩
  exact (div_le_one hS).mpr (sub_le_sub_right (hM hr (by norm_num) hr.2) _)

theorem toMass_identity {d : ℕ} (hd : 0 < d) (r : ℝ) :
    massCDF d 1 + cdfSpan d * toMass d r = massCDF d r := by
  unfold toMass
  have hne := (cdfSpan_pos hd).ne'
  field_simp
  ring

private theorem mass_quantile_exists {d : ℕ} (hd : 0 < d) (q : I) :
    ∃ r ∈ Icc (1:ℝ) 2, massCDF d r = massCDF d 1 + cdfSpan d * q.val := by
  have hq : massCDF d 1 + cdfSpan d * q.val ∈ Icc (massCDF d 1) (massCDF d 2) := by
    have hS := (cdfSpan_pos hd).le
    refine ⟨le_add_of_nonneg_right (mul_nonneg hS q.2.1), ?_⟩
    have hm := mul_le_of_le_one_right hS q.2.2
    unfold cdfSpan at *
    linarith
  exact intermediate_value_Icc (by norm_num) (massCDF_continuousOn hd) hq

def fromMass {d : ℕ} (hd : 0 < d) (q : I) : ℝ :=
  (mass_quantile_exists hd q).choose

theorem fromMass_mem {d : ℕ} (hd : 0 < d) (q : I) : fromMass hd q ∈ Icc 1 2 :=
  (mass_quantile_exists hd q).choose_spec.1

theorem fromMass_cdf {d : ℕ} (hd : 0 < d) (q : I) :
    massCDF d (fromMass hd q) = massCDF d 1 + cdfSpan d * q.val :=
  (mass_quantile_exists hd q).choose_spec.2

theorem fromMass_monotone {d : ℕ} (hd : 0 < d) : Monotone (fromMass hd) := by
  intro u v huv
  apply (massCDF_strictMonoOn hd).le_iff_le (fromMass_mem hd u) (fromMass_mem hd v) |>.mp
  rw [fromMass_cdf hd u, fromMass_cdf hd v]
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left huv (cdfSpan_pos hd).le)

theorem fromMass_lt_iff {d : ℕ} (hd : 0 < d) (q : I) {r : ℝ} (hr : r ∈ Icc 1 2) :
    fromMass hd q < r ↔ q.val < toMass d r := by
  rw [← (massCDF_strictMonoOn hd).lt_iff_lt (fromMass_mem hd q) hr,
    fromMass_cdf hd q, toMass, lt_div_iff₀ (cdfSpan_pos hd)]
  rw [mul_comm (cdfSpan d) q.val]
  constructor <;> intro h <;> linarith

end

end Cubature
