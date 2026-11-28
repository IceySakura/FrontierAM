import Theory.Setup
import Mathlib.Analysis.Calculus.ContDiff.Deriv
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Calculus.Deriv.Abs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! Truncated powers, the increasing phase profile, and its Gaussian fiber integral. -/

open Filter Function MeasureTheory ProbabilityTheory Set
open scoped ContDiff ENNReal NNReal Topology

namespace Cubature

lemma truncatedPower_nonneg (n : ℕ) (t : ℝ) : 0 ≤ truncatedPower n t := by
  unfold truncatedPower
  positivity

lemma truncatedPower_of_nonpos (n : ℕ) {t : ℝ} (h : t ≤ 0) :
    truncatedPower n t = 0 := by
  simp [truncatedPower, max_eq_left h]

lemma continuous_truncatedPower (n : ℕ) : Continuous (truncatedPower n) := by
  unfold truncatedPower
  exact ((continuous_const.max continuous_id).pow _).div_const _

private lemma hasDerivAt_posPart_pow (n : ℕ) (t : ℝ) :
    HasDerivAt (fun x : ℝ => (max 0 x) ^ (n + 2))
      ((↑(n + 2) : ℝ) * (max 0 t) ^ (n + 1)) t := by
  have hpow (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ (n + 2))
      ((↑(n + 2) : ℝ) * x ^ (n + 1)) x := by
    simpa [Pi.pow_def] using (hasDerivAt_id' x).pow (n + 2)
  rcases lt_trichotomy t 0 with h | rfl | h
  · rw [max_eq_left h.le, zero_pow (by omega : n + 1 ≠ 0), mul_zero]
    apply (hasDerivAt_const t (0 : ℝ)).congr_of_eventuallyEq
    filter_upwards [gt_mem_nhds h] with x hx
    simp [max_eq_left hx.le]
  · rw [max_self, zero_pow (by omega : n + 1 ≠ 0), mul_zero]
    have hl : HasDerivWithinAt (fun x : ℝ => (max 0 x) ^ (n + 2)) 0 (Iic 0) 0 := by
      apply (hasDerivWithinAt_const (s := Iic (0 : ℝ)) 0 0).congr
      · intro x hx
        simp [max_eq_left (mem_Iic.mp hx)]
      · simp
    have hr : HasDerivWithinAt (fun x : ℝ => (max 0 x) ^ (n + 2)) 0 (Ici 0) 0 := by
      have hd : HasDerivAt (fun x : ℝ => x ^ (n + 2)) 0 0 := by simpa using hpow 0
      apply hd.hasDerivWithinAt.congr
      · intro x hx
        simp [max_eq_right (mem_Ici.mp hx)]
      · simp
    have hu := hl.union hr
    rw [Iic_union_Ici, hasDerivWithinAt_univ] at hu
    exact hu
  · rw [max_eq_right h.le]
    apply (hpow t).congr_of_eventuallyEq
    filter_upwards [lt_mem_nhds h] with x hx
    simp [max_eq_right hx.le]

lemma hasDerivAt_truncatedPower_succ (n : ℕ) (t : ℝ) :
    HasDerivAt (truncatedPower (n + 1)) (truncatedPower n t) t := by
  have h := (hasDerivAt_posPart_pow n t).div_const ((n + 2).factorial : ℝ)
  have he : (↑(n + 2) : ℝ) * (max 0 t) ^ (n + 1) / ((n + 2).factorial : ℝ) =
      truncatedPower n t := by
    unfold truncatedPower
    rw [show (n + 2).factorial = (n + 2) * (n + 1).factorial from Nat.factorial_succ (n + 1),
      Nat.cast_mul]
    have h₁ : (↑(n + 2) : ℝ) ≠ 0 := by positivity
    have h₂ : (↑(n + 1).factorial : ℝ) ≠ 0 := by positivity
    field_simp
  rwa [he] at h

lemma deriv_truncatedPower_succ (n : ℕ) :
    deriv (truncatedPower (n + 1)) = truncatedPower n :=
  funext fun t => (hasDerivAt_truncatedPower_succ n t).deriv

lemma contDiff_truncatedPower (n : ℕ) : ContDiff ℝ (n : ℕ) (truncatedPower n) := by
  induction n with
  | zero => exact contDiff_zero.mpr (continuous_truncatedPower 0)
  | succ n ih =>
    rw [Nat.cast_succ, contDiff_succ_iff_deriv, deriv_truncatedPower_succ]
    exact ⟨fun t => (hasDerivAt_truncatedPower_succ n t).differentiableAt, by simp, ih⟩

lemma not_contDiff_truncatedPower (n : ℕ) :
    ¬ ContDiff ℝ (↑(n + 1) : ℕ∞ω) (truncatedPower n) := by
  induction n with
  | zero =>
    intro H
    have hd := (H.differentiable (by simp)) 0
    apply not_differentiableAt_abs_zero
    have heq : (fun t : ℝ => |t|) = fun t => 2 * truncatedPower 0 t - t := by
      funext t
      rcases le_total t 0 with ht | ht
      · simp [truncatedPower, abs_of_nonpos ht, max_eq_left ht]
      · simp [truncatedPower, abs_of_nonneg ht, max_eq_right ht]
        ring
    change DifferentiableAt ℝ (fun t : ℝ => |t|) 0
    rw [heq]
    exact (hd.const_mul 2).sub differentiableAt_id
  | succ n ih =>
    intro H
    rw [Nat.cast_succ, contDiff_succ_iff_deriv, deriv_truncatedPower_succ] at H
    exact ih H.2.2

noncomputable def axisProfile (q : ℕ) (t : ℝ) : ℝ := t + truncatedPower q t

lemma continuous_axisProfile (q : ℕ) : Continuous (axisProfile q) :=
  continuous_id.add (continuous_truncatedPower q)

lemma contDiff_axisProfile (q : ℕ) : ContDiff ℝ (q : ℕ) (axisProfile q) :=
  contDiff_id.add (contDiff_truncatedPower q)

lemma not_contDiff_axisProfile (q : ℕ) : ¬ ContDiff ℝ (↑(q + 1) : ℕ∞ω) (axisProfile q) := by
  intro h
  apply not_contDiff_truncatedPower q
  convert h.sub (contDiff_id : ContDiff ℝ (↑(q + 1) : ℕ∞ω) (id : ℝ → ℝ)) using 1
  ext t
  simp [axisProfile]

lemma hasDerivAt_axisProfile {q : ℕ} (hq : 1 ≤ q) (t : ℝ) :
    HasDerivAt (axisProfile q) (1 + truncatedPower (q - 1) t) t := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : q ≠ 0)
  exact (hasDerivAt_id t).add (hasDerivAt_truncatedPower_succ n t)

lemma axisProfile_strictMono (q : ℕ) : StrictMono (axisProfile q) := by
  intro s t h
  unfold axisProfile
  apply add_lt_add_of_lt_of_le h
  unfold truncatedPower
  gcongr

lemma axisProfile_atTop (q : ℕ) : Tendsto (axisProfile q) atTop atTop :=
  tendsto_atTop_mono (fun t => le_add_of_nonneg_right (truncatedPower_nonneg q t)) tendsto_id

lemma axisProfile_atBot (q : ℕ) : Tendsto (axisProfile q) atBot atBot := by
  refine tendsto_id.congr' ?_
  filter_upwards [eventually_le_atBot (0 : ℝ)] with t ht
  simp [axisProfile, truncatedPower_of_nonpos _ ht]

noncomputable def axisOrderIso (q : ℕ) : ℝ ≃o ℝ :=
  (axisProfile_strictMono q).orderIsoOfSurjective _
    ((continuous_axisProfile q).surjective (axisProfile_atTop q) (axisProfile_atBot q))

@[simp] lemma axisOrderIso_apply (q : ℕ) (t : ℝ) : axisOrderIso q t = axisProfile q t := rfl

@[simp] lemma axisProfile_inverse (q : ℕ) (u : ℝ) :
    axisProfile q ((axisOrderIso q).symm u) = u :=
  (axisOrderIso q).apply_symm_apply u

@[simp] lemma inverse_axisProfile (q : ℕ) (t : ℝ) :
    (axisOrderIso q).symm (axisProfile q t) = t :=
  (axisOrderIso q).symm_apply_apply t

lemma contDiff_normalDensity : ContDiff ℝ ∞ (gaussianPDFReal 0 1) := by
  simp only [gaussianPDFReal_def, sub_zero, NNReal.coe_one, mul_one]
  fun_prop

noncomputable def gaussianTail (a : ℝ) : ℝ := ∫ t in Ioi a, gaussianPDFReal 0 1 t

lemma hasDerivAt_gaussianTail (a : ℝ) :
    HasDerivAt gaussianTail (-gaussianPDFReal 0 1 a) a := by
  have hInt : Integrable (gaussianPDFReal 0 1) volume := integrable_gaussianPDFReal 0 1
  have hrepr : gaussianTail = fun x => gaussianTail 0 -
      ∫ t in (0 : ℝ)..x, gaussianPDFReal 0 1 t := by
    funext x
    rw [gaussianTail, gaussianTail, ← intervalIntegral.integral_Ioi_sub_Ioi' (a := 0) (b := x)
      hInt.integrableOn hInt.integrableOn]
    ring
  rw [hrepr]
  exact (intervalIntegral.integral_hasDerivAt_right (hInt.intervalIntegrable (a := 0) (b := a))
      contDiff_normalDensity.continuous.stronglyMeasurable.stronglyMeasurableAtFilter
      contDiff_normalDensity.continuous.continuousAt).const_sub (gaussianTail 0)

lemma contDiff_gaussianTail : ContDiff ℝ ∞ gaussianTail := by
  have he : deriv gaussianTail = fun t => -gaussianPDFReal 0 1 t :=
    funext fun t => (hasDerivAt_gaussianTail t).deriv
  rw [contDiff_infty_iff_deriv, he]
  exact ⟨fun t => (hasDerivAt_gaussianTail t).differentiableAt, contDiff_normalDensity.neg⟩

/-- The preintegrated ridge's profile. -/
noncomputable def integratedProfile (q : ℕ) (u : ℝ) : ℝ :=
  gaussianTail ((axisOrderIso q).symm u)

lemma gaussianPreintegral_sharpPhase {ι : Type*} [Fintype ι] (q : ℕ) (y : ι → ℝ) :
    gaussianPreintegral (sharpPhase q) (fun _ => 1) y = integratedProfile q (∑ i, y i) := by
  let a := (axisOrderIso q).symm (∑ i, y i)
  have hsign (t : ℝ) : 0 < sharpPhase q (fiberPoint t y) ↔ a < t := by
    change 0 < axisProfile q t - ∑ i, y i ↔ a < t
    rw [sub_pos, ← axisProfile_inverse q (∑ i, y i)]
    exact (axisProfile_strictMono q).lt_iff_lt
  simp only [gaussianPreintegral, hsign]
  change (∫ t : ℝ, (if a < t then (1 : ℝ) else 0) ∂gaussianReal 0 1) = gaussianTail a
  have heq : (fun t : ℝ => if a < t then (1 : ℝ) else 0) =
      (Ioi a).indicator (fun _ => (1 : ℝ)) := by ext t; rfl
  rw [heq, integral_indicator measurableSet_Ioi, integral_const, smul_eq_mul, mul_one]
  rw [Measure.real, Measure.restrict_apply_univ,
    gaussianReal_apply_eq_integral 0 (by norm_num : (1 : ℝ≥0) ≠ 0), gaussianTail, ENNReal.toReal_ofReal]
  exact setIntegral_nonneg measurableSet_Ioi (fun x _ => gaussianPDFReal_nonneg 0 1 x)

lemma contDiff_inverseAxis {q : ℕ} (hq : 1 ≤ q) :
    ContDiff ℝ (q : ℕ) ((axisOrderIso q).symm : ℝ → ℝ) := by
  apply (axisOrderIso q).toHomeomorph.contDiff_symm_deriv
    (f' := fun t => 1 + truncatedPower (q - 1) t)
  · intro t
    exact (add_pos_of_pos_of_nonneg zero_lt_one (truncatedPower_nonneg _ _)).ne'
  · intro t
    exact hasDerivAt_axisProfile hq t
  · exact contDiff_axisProfile q

lemma contDiff_integratedProfile {q : ℕ} (hq : 1 ≤ q) :
    ContDiff ℝ (q : ℕ) (integratedProfile q) :=
  (contDiff_gaussianTail.of_le (by simp)).comp (contDiff_inverseAxis hq)

lemma not_contDiff_integratedProfile {q : ℕ} (hq : 1 ≤ q) :
    ¬ ContDiff ℝ (↑(q + 1) : ℕ∞ω) (integratedProfile q) := by
  intro H
  have hd : ContDiff ℝ (q : ℕ) (deriv (integratedProfile q)) := by
    rw [Nat.cast_succ, contDiff_succ_iff_deriv] at H
    exact H.2.2
  have hchain (t : ℝ) :
      deriv (integratedProfile q) (axisProfile q t) * deriv (axisProfile q) t =
        -gaussianPDFReal 0 1 t := by
    have hh := ((H.differentiable (by simp))
      (axisProfile q t)).hasDerivAt
    have ha := (hasDerivAt_axisProfile hq t).differentiableAt.hasDerivAt
    have hc := hh.scomp t ha
    have he : integratedProfile q ∘ axisProfile q = gaussianTail := by
      funext t
      simp [integratedProfile]
    rw [he] at hc
    simpa [smul_eq_mul, mul_comm] using hc.deriv.symm.trans (hasDerivAt_gaussianTail t).deriv
  have hn (t : ℝ) : deriv (integratedProfile q) (axisProfile q t) ≠ 0 := by
    intro he
    have hp := gaussianPDFReal_pos 0 1 t (by norm_num)
    have hc := hchain t
    rw [he, zero_mul] at hc
    linarith
  have he : deriv (axisProfile q) =
      fun t => -gaussianPDFReal 0 1 t / deriv (integratedProfile q) (axisProfile q t) := by
    funext t
    exact (eq_div_iff (hn t)).mpr (by rw [mul_comm]; exact hchain t)
  apply not_contDiff_axisProfile q
  rw [Nat.cast_succ, contDiff_succ_iff_deriv, he]
  refine ⟨fun t => (hasDerivAt_axisProfile hq t).differentiableAt, by simp, ?_⟩
  exact (contDiff_normalDensity.of_le (by simp)).neg.div
    (hd.comp (contDiff_axisProfile q)) hn

end Cubature
