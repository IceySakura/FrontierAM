import Theory.WeakSetup
import Theory.Axis
import Mathlib.Analysis.Calculus.Deriv.Inverse
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-! The bounded-slope fractional splice and the lower bound on its profile's second derivative. -/

open Filter Function MeasureTheory ProbabilityTheory Set
open scoped ContDiff NNReal Topology

namespace Cubature

private lemma fractionalRamp_of_nonpos {t : ℝ} (ht : t ≤ 0) : fractionalRamp t = 0 := by
  simp [fractionalRamp, max_eq_left ht, Real.zero_rpow (by norm_num : (3 / 2 : ℝ) ≠ 0)]

private lemma hasDerivAt_ramp_formula (t : ℝ) :
    HasDerivAt (fun x : ℝ => (2 / 3 : ℝ) * x ^ (3 / 2 : ℝ)) (Real.sqrt t) t := by
  have h := (Real.hasDerivAt_rpow_const (x := t) (p := 3 / 2) (Or.inr (by norm_num))).const_mul
    (2 / 3 : ℝ)
  convert h using 1
  rw [Real.sqrt_eq_rpow, show (3 / 2 : ℝ) - 1 = 1 / 2 by norm_num]
  ring

lemma hasDerivAt_fractionalRamp (t : ℝ) :
    HasDerivAt fractionalRamp (Real.sqrt (max 0 t)) t := by
  rcases lt_trichotomy t 0 with ht | rfl | ht
  · rw [max_eq_left ht.le, Real.sqrt_zero]
    apply (hasDerivAt_const t (0 : ℝ)).congr_of_eventuallyEq
    filter_upwards [gt_mem_nhds ht] with x hx
    exact fractionalRamp_of_nonpos hx.le
  · simp only [max_self, Real.sqrt_zero]
    have hl : HasDerivWithinAt fractionalRamp 0 (Iic 0) 0 :=
      (hasDerivWithinAt_const (s := Iic (0 : ℝ)) 0 0).congr_of_mem
        (fun _ hx => fractionalRamp_of_nonpos hx) (by simp)
    have hr : HasDerivWithinAt fractionalRamp 0 (Ici 0) 0 := by
      have hd := hasDerivAt_ramp_formula 0
      rw [Real.sqrt_zero] at hd
      apply hd.hasDerivWithinAt.congr_of_mem _ (by simp)
      intro x hx
      simp [fractionalRamp, max_eq_right (mem_Ici.mp hx)]
    have hu := hl.union hr
    rwa [Iic_union_Ici, hasDerivWithinAt_univ] at hu
  · rw [max_eq_right ht.le]
    apply (hasDerivAt_ramp_formula t).congr_of_eventuallyEq
    filter_upwards [lt_mem_nhds ht] with x hx
    simp [fractionalRamp, max_eq_right hx.le]

noncomputable def spliceSlope (u : ℝ) : ℝ :=
  1 + Real.sqrt (max 0 u) - Real.sqrt (max 0 (u - 1))

lemma continuous_spliceSlope : Continuous spliceSlope := by
  unfold spliceSlope
  exact (continuous_const.add ((continuous_const.max continuous_id).sqrt)).sub
    ((continuous_const.max (continuous_id.sub continuous_const)).sqrt)

lemma hasDerivAt_inverseSplice (u : ℝ) :
    HasDerivAt inverseSplice (spliceSlope u) u := by
  have hshift : HasDerivAt (fun x => fractionalRamp (x - 1))
      (Real.sqrt (max 0 (u - 1))) u :=
    (hasDerivAt_fractionalRamp (u - 1)).comp_sub_const u 1
  exact ((hasDerivAt_id' u).add (hasDerivAt_fractionalRamp u)).sub hshift

lemma one_le_spliceSlope (u : ℝ) : 1 ≤ spliceSlope u := by
  have h := Real.sqrt_le_sqrt (show max 0 (u - 1) ≤ max 0 u by
    exact max_le_max le_rfl (by linarith))
  dsimp [spliceSlope]
  linarith

lemma spliceSlope_le_two (u : ℝ) : spliceSlope u ≤ 2 := by
  unfold spliceSlope
  by_cases hu : u ≤ 1
  · rw [max_eq_left (show u - 1 ≤ 0 by linarith), Real.sqrt_zero, sub_zero]
    have hs := Real.sqrt_le_sqrt (max_le (by norm_num : (0 : ℝ) ≤ 1) hu)
    rw [Real.sqrt_one] at hs
    linarith
  · have hu' : 1 < u := lt_of_not_ge hu
    rw [max_eq_right (by linarith : 0 ≤ u), max_eq_right (by linarith : 0 ≤ u - 1)]
    have hsq := Real.sq_sqrt (by linarith : 0 ≤ u - 1)
    have hs : Real.sqrt u ≤ 1 + Real.sqrt (u - 1) := by
      apply Real.sqrt_le_iff.mpr
      constructor
      · positivity
      · nlinarith [Real.sqrt_nonneg (u - 1)]
    linarith

lemma inverseSplice_of_nonpos {u : ℝ} (hu : u ≤ 0) : inverseSplice u = u := by
  simp [inverseSplice, fractionalRamp_of_nonpos hu,
    fractionalRamp_of_nonpos (show u - 1 ≤ 0 by linarith)]

lemma le_inverseSplice (u : ℝ) : u ≤ inverseSplice u := by
  have h : Monotone fractionalRamp := monotone_of_hasDerivAt_nonneg
    hasDerivAt_fractionalRamp (fun _ => Real.sqrt_nonneg _)
  unfold inverseSplice
  linarith only [h (by linarith : u - 1 ≤ u)]

lemma inverseSplice_strictMono : StrictMono inverseSplice :=
  strictMono_of_hasDerivAt_pos hasDerivAt_inverseSplice
    (fun u => zero_lt_one.trans_le (one_le_spliceSlope u))

lemma contDiff_inverseSplice : ContDiff ℝ 1 inverseSplice := by
  rw [contDiff_one_iff_deriv]
  have he : deriv inverseSplice = spliceSlope := funext fun u => (hasDerivAt_inverseSplice u).deriv
  exact ⟨fun u => (hasDerivAt_inverseSplice u).differentiableAt, he ▸ continuous_spliceSlope⟩

noncomputable def spliceOrderIso : ℝ ≃o ℝ :=
  inverseSplice_strictMono.orderIsoOfSurjective _ (by
    apply contDiff_inverseSplice.continuous.surjective
    · exact tendsto_atTop_mono le_inverseSplice tendsto_id
    · refine tendsto_id.congr' ?_
      filter_upwards [eventually_le_atBot (0 : ℝ)] with t ht
      exact (inverseSplice_of_nonpos ht).symm)

noncomputable def spliceAxis : ℝ → ℝ := spliceOrderIso.symm

@[simp] lemma spliceAxis_left (u : ℝ) : spliceAxis (inverseSplice u) = u :=
  spliceOrderIso.symm_apply_apply u

@[simp] lemma spliceAxis_right (t : ℝ) : inverseSplice (spliceAxis t) = t :=
  spliceOrderIso.apply_symm_apply t

lemma continuous_spliceAxis : Continuous spliceAxis := spliceOrderIso.symm.continuous

noncomputable def axisSlope (t : ℝ) : ℝ := (spliceSlope (spliceAxis t))⁻¹

lemma continuous_axisSlope : Continuous axisSlope :=
  (continuous_spliceSlope.comp continuous_spliceAxis).inv₀
    (fun t => (zero_lt_one.trans_le (one_le_spliceSlope (spliceAxis t))).ne')

lemma hasDerivAt_spliceAxis (t : ℝ) : HasDerivAt spliceAxis (axisSlope t) t :=
  HasDerivAt.of_local_left_inverse continuous_spliceAxis.continuousAt
    (hasDerivAt_inverseSplice (spliceAxis t))
    (zero_lt_one.trans_le (one_le_spliceSlope (spliceAxis t))).ne'
    (Eventually.of_forall spliceAxis_right)

lemma axisSlope_bounds (t : ℝ) : 1 / 2 ≤ axisSlope t ∧ axisSlope t ≤ 1 := by
  have h := zero_lt_one.trans_le (one_le_spliceSlope (spliceAxis t))
  unfold axisSlope
  refine ⟨?_, ?_⟩
  · rw [← one_div]
    exact one_div_le_one_div_of_le h (spliceSlope_le_two _)
  · exact inv_le_one_of_one_le₀ (one_le_spliceSlope _)

lemma contDiff_spliceAxis : ContDiff ℝ 1 spliceAxis := by
  rw [contDiff_one_iff_deriv]
  have he : deriv spliceAxis = axisSlope := funext fun t => (hasDerivAt_spliceAxis t).deriv
  exact ⟨fun t => (hasDerivAt_spliceAxis t).differentiableAt, he ▸ continuous_axisSlope⟩

lemma abs_spliceAxis_le (t : ℝ) : |spliceAxis t| ≤ |t| := by
  by_cases ht : spliceAxis t ≤ 0
  · have he := inverseSplice_of_nonpos ht
    rw [spliceAxis_right] at he
    rw [← he]
  · have h : 0 ≤ spliceAxis t := le_of_lt (lt_of_not_ge ht)
    have hv := le_inverseSplice (spliceAxis t)
    rw [spliceAxis_right] at hv
    rw [abs_of_nonneg h, abs_of_nonneg (h.trans hv)]
    exact hv

noncomputable def spliceTail (u : ℝ) : ℝ := gaussianTail (inverseSplice u)

noncomputable def spliceTailFirst (u : ℝ) : ℝ :=
  -gaussianPDFReal 0 1 (inverseSplice u) * spliceSlope u

lemma hasDerivAt_spliceTail (u : ℝ) : HasDerivAt spliceTail (spliceTailFirst u) u := by
  unfold spliceTail spliceTailFirst
  simpa [Function.comp_def, smul_eq_mul, mul_comm] using
    (hasDerivAt_gaussianTail (inverseSplice u)).scomp u (hasDerivAt_inverseSplice u)

lemma contDiff_spliceTail : ContDiff ℝ 1 spliceTail :=
  (contDiff_gaussianTail.of_le (by simp)).comp contDiff_inverseSplice

lemma continuous_spliceTailFirst : Continuous spliceTailFirst :=
  ((contDiff_normalDensity.continuous.comp contDiff_inverseSplice.continuous).neg).mul
    continuous_spliceSlope

private lemma hasDerivAt_normalDensity (t : ℝ) :
    HasDerivAt (gaussianPDFReal 0 1) (-t * gaussianPDFReal 0 1 t) t := by
  simp only [gaussianPDFReal_def, sub_zero, NNReal.coe_one, mul_one]
  have h : HasDerivAt (fun x : ℝ => -(x ^ 2) / 2) (-t) t := by
    convert ((hasDerivAt_id' t).pow 2).neg.div_const 2 using 1
    simp
    ring
  convert h.exp.const_mul (Real.sqrt (2 * Real.pi))⁻¹ using 1
  ring

lemma hasDerivAt_spliceSlope {u : ℝ} (hu : 0 < u) (hu' : u < 1) :
    HasDerivAt spliceSlope (1 / (2 * Real.sqrt u)) u := by
  apply ((Real.hasDerivAt_sqrt hu.ne').const_add 1).congr_of_eventuallyEq
  filter_upwards [lt_mem_nhds hu, gt_mem_nhds hu'] with x hx hx'
  simp [spliceSlope, max_eq_right hx.le, max_eq_left (show x - 1 ≤ 0 by linarith)]

noncomputable def spliceTailSecond (u : ℝ) : ℝ :=
  gaussianPDFReal 0 1 (inverseSplice u) *
    (inverseSplice u * spliceSlope u ^ 2 - 1 / (2 * Real.sqrt u))

lemma hasDerivAt_spliceTailFirst {u : ℝ} (hu : 0 < u) (hu' : u < 1) :
    HasDerivAt spliceTailFirst (spliceTailSecond u) u := by
  have hp : HasDerivAt (fun x => gaussianPDFReal 0 1 (inverseSplice x))
      (spliceSlope u * (-inverseSplice u * gaussianPDFReal 0 1 (inverseSplice u))) u := by
    simpa only [Function.comp_def, smul_eq_mul] using
      (hasDerivAt_normalDensity (inverseSplice u)).scomp u (hasDerivAt_inverseSplice u)
  unfold spliceTailFirst
  convert hp.neg.mul (hasDerivAt_spliceSlope hu hu') using 1
  unfold spliceTailSecond
  rw [Pi.neg_apply]
  ring

lemma continuousOn_spliceTailSecond : ContinuousOn spliceTailSecond (Ioo 0 1) := by
  have h := contDiff_inverseSplice.continuous
  unfold spliceTailSecond
  refine ((contDiff_normalDensity.continuous.comp h).continuousOn).mul ?_
  refine (h.mul (continuous_spliceSlope.pow _)).continuousOn.sub ?_
  exact continuousOn_const.div ((continuous_const.mul Real.continuous_sqrt).continuousOn)
    (fun u hu => mul_ne_zero (by norm_num) (Real.sqrt_pos.2 hu.1).ne')

private lemma normalDensity_ge_at_one {t : ℝ} (ht : 0 ≤ t) (ht' : t ≤ 1) :
    gaussianPDFReal 0 1 1 ≤ gaussianPDFReal 0 1 t := by
  simp only [gaussianPDFReal_def, sub_zero, NNReal.coe_one, mul_one, one_pow]
  apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by positivity)
  nlinarith

lemma spliceTailSecond_square_lower {u : ℝ} (hu : 0 < u) (hu' : u < 1 / 16) :
    (gaussianPDFReal 0 1 1 / 4) ^ 2 / u ≤ spliceTailSecond u ^ 2 := by
  have hs : 0 < Real.sqrt u := Real.sqrt_pos.2 hu
  have hs' : Real.sqrt u ≤ 1 / 4 := by
    rw [Real.sqrt_le_iff]
    norm_num
    linarith
  have hsf : Real.sqrt u ^ 2 = u := Real.sq_sqrt hu.le
  have hb : inverseSplice u ≤ 2 * u := by
    rw [inverseSplice, fractionalRamp_of_nonpos (by linarith : u - 1 ≤ 0), sub_zero,
      fractionalRamp, max_eq_right hu.le, show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num,
      Real.rpow_one_add' hu.le (by norm_num), ← Real.sqrt_eq_rpow]
    nlinarith [mul_le_mul_of_nonneg_left hs' hu.le]
  have hbp : 0 ≤ inverseSplice u := hu.le.trans (le_inverseSplice u)
  have hσ : 0 ≤ spliceSlope u := zero_le_one.trans (one_le_spliceSlope u)
  have hmoment : inverseSplice u * spliceSlope u ^ 2 ≤ 8 * u := by
    calc
      _ ≤ (2 * u) * 2 ^ 2 := mul_le_mul hb (pow_le_pow_left₀ hσ (spliceSlope_le_two u) 2)
        (sq_nonneg _) (by positivity)
      _ = 8 * u := by ring
  have hgap : 1 / (4 * Real.sqrt u) ≤
      1 / (2 * Real.sqrt u) - inverseSplice u * spliceSlope u ^ 2 := by
    have he : 8 * u ≤ 1 / (4 * Real.sqrt u) := by
      apply (le_div_iff₀ (by positivity : 0 < 4 * Real.sqrt u)).mpr
      nlinarith [mul_le_mul_of_nonneg_left hs' hu.le]
    have heq : 1 / (2 * Real.sqrt u) = 2 * (1 / (4 * Real.sqrt u)) := by
      field_simp [hs.ne']
      norm_num
    rw [heq]
    linarith
  have hρ := normalDensity_ge_at_one hbp (by linarith)
  have hρ₀ := gaussianPDFReal_pos 0 1 1 (by norm_num)
  have hprod := mul_le_mul hρ hgap (by positivity : 0 ≤ 1 / (4 * Real.sqrt u))
    (gaussianPDFReal_nonneg 0 1 (inverseSplice u))
  have hmain : gaussianPDFReal 0 1 1 / (4 * Real.sqrt u) ≤ -spliceTailSecond u := by
    rw [← mul_one_div]
    refine hprod.trans_eq ?_
    unfold spliceTailSecond
    ring
  calc
    _ = (gaussianPDFReal 0 1 1 / (4 * Real.sqrt u)) ^ 2 := by
      rw [← div_div, div_pow (gaussianPDFReal 0 1 1 / 4), hsf]
    _ ≤ (-spliceTailSecond u) ^ 2 := pow_le_pow_left₀ (by positivity) hmain 2
    _ = spliceTailSecond u ^ 2 := by ring

end Cubature
