import Theory.WeakObstruction
import Theory.SharpPhase

/-! Gaussian weighted hypotheses for the fractional splice. -/

open Filter MeasureTheory ProbabilityTheory Set
open scoped ContDiff NNReal

namespace Cubature

section Phase

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

private lemma polynomialGrowth_spliceAxis : PolynomialGrowth spliceAxis := by
  refine ⟨1, zero_le_one, 1, fun t => ?_⟩
  simpa only [one_mul, pow_one, Real.norm_eq_abs] using
    (abs_spliceAxis_le t).trans (le_add_of_nonneg_left zero_le_one)

private lemma polynomialGrowth_axisSlope : PolynomialGrowth axisSlope := by
  refine ⟨1, zero_le_one, 0, fun t => ?_⟩
  obtain ⟨hl, hu⟩ := axisSlope_bounds t
  rw [abs_of_nonneg (by linarith), pow_zero, mul_one]
  exact hu

private lemma zeroAxisChain (u : ℝ → ℝ) (hc : Continuous u) (w : List (Option ι))
    (hw : w.count none = 0) :
    ContinuousPartialChain w (fun x => u (x none))
      (fun x => if w = [] then u (x none) else 0) := by
  have hc' : Continuous (fun x : Option ι → ℝ => u (x none)) := hc.comp (continuous_apply none)
  cases w with
  | nil => exact ⟨hc', rfl⟩
  | cons i w =>
    cases i with
    | none => simp at hw
    | some i =>
      refine ⟨hc', (fun _ => 0), hasRetainedCoordinatePartial_axis u i, ?_⟩
      simpa using ContinuousPartialChain.const w 0

private lemma oneAxisChain (w : List (Option ι)) (hw : w.count none ≤ 1) :
    ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧
      ContinuousPartialChain w (fun x => spliceAxis (x none)) (fun x => u (x none)) := by
  have hc : Continuous (fun x : Option ι → ℝ => spliceAxis (x none)) :=
    continuous_spliceAxis.comp (continuous_apply none)
  cases w with
  | nil => exact ⟨spliceAxis, continuous_spliceAxis, polynomialGrowth_spliceAxis, hc, rfl⟩
  | cons i w =>
    cases i with
    | some i =>
      refine ⟨fun _ => 0, continuous_const, polynomialGrowth_const 0, hc,
        fun _ => 0, hasRetainedCoordinatePartial_axis spliceAxis i, ?_⟩
      simpa using ContinuousPartialChain.const w 0
    | none =>
      have hz : w.count none = 0 := by simpa using hw
      have h := zeroAxisChain axisSlope continuous_axisSlope w hz
      by_cases he : w = []
      · refine ⟨axisSlope, continuous_axisSlope, polynomialGrowth_axisSlope, hc,
          (fun x => axisSlope (x none)), hasAxisCoordinatePartial _ _ hasDerivAt_spliceAxis, ?_⟩
        simpa [he, ContinuousPartialChain] using h
      · refine ⟨fun _ => 0, continuous_const, polynomialGrowth_const 0, hc,
          (fun x => axisSlope (x none)), hasAxisCoordinatePartial _ _ hasDerivAt_spliceAxis, ?_⟩
        simpa only [ite_eq_right he] using h

private lemma splicePhaseChain (w : List (Option ι)) (hw : w.count none ≤ 1) :
    ∃ g : (Option ι → ℝ) → ℝ, ContinuousPartialChain w (phaseFromAxis spliceAxis) g ∧
      PolynomialGrowth g ∧
      (w ≠ [] → ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧ ∀ x, g x = u (x none)) := by
  obtain ⟨u, hcu, hpu, h⟩ := oneAxisChain w hw
  cases w with
  | nil =>
    have hr : ContinuousPartialChain ([] : List (Option ι)) retainedSum retainedSum :=
      ⟨(contDiff_retainedSum 0).continuous, rfl⟩
    exact ⟨_, h.sub hr, (hpu.comp_coord none).sub polynomialGrowth_retainedSum,
      fun hn => (hn rfl).elim⟩
  | cons i w =>
    obtain ⟨c, hc⟩ := retainedSumChain i w
    exact ⟨_, h.sub hc, (hpu.comp_coord none).sub (polynomialGrowth_const c),
      fun _ => ⟨fun t => u t - c, hcu.sub continuous_const,
        hpu.sub (polynomialGrowth_const c), fun _ => rfl⟩⟩

lemma gaussianMixedInput_splicePhase (r : ι → ℕ) :
    GaussianMixedInput (fun i : Option ι => i.elim 1 r) (phaseFromAxis spliceAxis) := by
  intro w hw
  have he (v : List (Option ι)) :
      @List.count _ instBEqOfDecidableEq none v = v.count none := by
    induction v with
    | nil => rfl
    | cons i v ih => cases i <;> simp [ih]
  obtain ⟨g, hc, hp, _⟩ := splicePhaseChain w (he w ▸ hw none)
  exact ⟨g, hc.partials, hc.continuous_end, hc.weak, hp,
    fun p hf => hp.memLp_gaussianProduct hc.continuous_end p hf⟩

lemma contDiff_splicePhase : ContDiff ℝ 1
    (phaseFromAxis spliceAxis : (Option ι → ℝ) → ℝ) :=
  (contDiff_spliceAxis.comp (contDiff_apply ℝ (fun _ : Option ι => ℝ) none)).sub
    (contDiff_retainedSum 1)

lemma partialsAlong_splicePhase_axis (x : Option ι → ℝ) :
    partialsAlong [none] (phaseFromAxis spliceAxis) x = axisSlope (x none) := by
  have hd := (hasAxisCoordinatePartial (ι := ι) _ _ hasDerivAt_spliceAxis).sub
    (hasCoordinatePartial_retainedSum (none : Option ι))
  exact (hd x).lineDeriv.trans (by simp)

noncomputable def spliceRoot (y : ι → ℝ) : ℝ := inverseSplice (∑ i, y i)

lemma continuous_spliceRoot : Continuous (spliceRoot : (ι → ℝ) → ℝ) :=
  contDiff_inverseSplice.continuous.comp (continuous_finsetSum _ fun i _ => continuous_apply i)

lemma tendsto_splicePhase_fiber (y : ι → ℝ) :
    Tendsto (fun t => phaseFromAxis spliceAxis (fiberPoint t y)) atTop atTop := by
  apply tendsto_atTop.mpr
  intro b
  filter_upwards [eventually_ge_atTop (inverseSplice (b + ∑ i, y i))] with t ht
  have h := spliceOrderIso.symm.monotone ht
  change b ≤ spliceAxis t - ∑ i, y i
  change spliceAxis (inverseSplice (b + ∑ i, y i)) ≤ spliceAxis t at h
  rw [spliceAxis_left] at h
  linarith

lemma spliceRoot_eq_iff (t : ℝ) (y : ι → ℝ) :
    phaseFromAxis spliceAxis (fiberPoint t y) = 0 ↔ t = spliceRoot y := by
  change spliceAxis t - ∑ i, y i = 0 ↔ t = inverseSplice (∑ i, y i)
  rw [sub_eq_zero]
  exact ⟨fun h => by rw [← h, spliceAxis_right], fun h => by rw [h, spliceAxis_left]⟩

private lemma spliceDerivativeProduct (ws : List (List (Option ι)))
    (hw : ∀ w ∈ ws, w ≠ [] ∧ w.count none ≤ 1) :
    ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧
      ∀ x : Option ι → ℝ,
        (ws.map (fun w => partialsAlong w (phaseFromAxis spliceAxis) x)).prod = u (x none) := by
  induction ws with
  | nil => exact ⟨fun _ => 1, continuous_const, polynomialGrowth_const 1, fun _ => rfl⟩
  | cons w ws ih =>
    have hw₁ := hw w (by simp)
    obtain ⟨g, hc, _, hg⟩ := splicePhaseChain w hw₁.2
    obtain ⟨u, hcu, hpu, hu⟩ := hg hw₁.1
    obtain ⟨v, hcv, hpv, hv⟩ := ih (fun w hw' => hw w (List.mem_cons_of_mem _ hw'))
    refine ⟨fun t => u t * v t, hcu.mul hcv, hpu.mul hpv, fun x => ?_⟩
    simp only [List.map_cons, List.prod_cons, hc.partials.eq_partialsAlong, hu, hv]

lemma boundaryFactor_splicePhase (w : List (Option ι))
    (ws : List (List (Option ι))) (hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ 1) (b c : ℕ) :
    Continuous (boundaryFactor (phaseFromAxis spliceAxis) (fun _ => 1) spliceRoot w ws b c) ∧
    ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor (phaseFromAxis spliceAxis) (fun _ => 1)
      spliceRoot w ws b c y| ≤ C := by
  let a : ℝ := if w = [] then 1 else 0
  have ha : partialsAlong w (fun _ => 1) = fun _ => a :=
    (ContinuousPartialChain.const w 1).partials.eq_partialsAlong
  obtain ⟨u, hcu, hpu, hu⟩ := spliceDerivativeProduct ws hw
  obtain ⟨poly, hcp, hpp, hp⟩ := exists_density_polynomial c
  let V : ℝ → ℝ := fun t => (a * u t) * poly t
  let G : ℝ → ℝ := fun t => V t * Real.exp (-(t ^ 2 / 2))
  let D : ℝ → ℝ := fun t => axisSlope t ^ b
  have hD (t : ℝ) : (1 / 2 : ℝ) ^ b ≤ D t :=
    pow_le_pow_left₀ (by norm_num) (axisSlope_bounds t).1 b
  have hpos (t : ℝ) : 0 < D t := (by positivity : 0 < (1 / 2 : ℝ) ^ b).trans_le (hD t)
  have hrepr : boundaryFactor (phaseFromAxis spliceAxis) (fun _ => 1) spliceRoot w ws b c =
      fun y => G (spliceRoot y) / D (spliceRoot y) := by
    funext y
    simp only [boundaryFactor, ha, hu, partialsAlong_splicePhase_axis, fiberPoint,
      Option.elim_none, hp, G, D, V]
    ring
  rw [hrepr]
  constructor
  · have hG : Continuous G := ((continuous_const.mul hcu).mul hcp).mul (by fun_prop)
    have hDc : Continuous D := continuous_axisSlope.pow b
    exact (hG.comp continuous_spliceRoot).div (hDc.comp continuous_spliceRoot)
      (fun y => (hpos (spliceRoot y)).ne')
  · have hVp : PolynomialGrowth V := ((polynomialGrowth_const a).mul hpu).mul hpp
    obtain ⟨B, hB, hGB⟩ := hVp.gaussian_damped_bound
    refine ⟨B / (1 / 2 : ℝ) ^ b, by positivity, fun y => ?_⟩
    rw [abs_div, abs_of_pos (hpos _)]
    exact div_le_div_of_nonneg_right (hGB _) (hpos _).le |>.trans
      (div_le_div_of_nonneg_left hB (by positivity) (hD _))

lemma gaussianPreintegral_splicePhase (y : ι → ℝ) :
    gaussianPreintegral (phaseFromAxis spliceAxis) (fun _ => 1) y =
      spliceTail (∑ i, y i) := by
  let a := spliceRoot y
  have hs (t : ℝ) : 0 < phaseFromAxis spliceAxis (fiberPoint t y) ↔ a < t := by
    change 0 < spliceAxis t - ∑ i, y i ↔ inverseSplice (∑ i, y i) < t
    rw [sub_pos]
    have h := (spliceOrderIso.symm.lt_iff_lt (x := inverseSplice (∑ i, y i)) (y := t))
    change spliceAxis (inverseSplice (∑ i, y i)) < spliceAxis t ↔ _ at h
    simpa only [spliceAxis_left] using h
  simp only [gaussianPreintegral, hs]
  change (∫ t : ℝ, (if a < t then (1 : ℝ) else 0) ∂gaussianReal 0 1) = gaussianTail a
  have heq : (fun t : ℝ => if a < t then (1 : ℝ) else 0) =
      (Ioi a).indicator (fun _ => (1 : ℝ)) := by ext t; rfl
  rw [heq, integral_indicator measurableSet_Ioi, integral_const, smul_eq_mul, mul_one,
    Measure.real, Measure.restrict_apply_univ,
    gaussianReal_apply_eq_integral 0 (by norm_num : (1 : ℝ≥0) ≠ 0), gaussianTail, ENNReal.toReal_ofReal]
  exact setIntegral_nonneg measurableSet_Ioi (fun x _ => gaussianPDFReal_nonneg 0 1 x)

end Phase

lemma pair_gaussianPreintegral_splicePhase :
    (fun y : ℝ × ℝ => gaussianPreintegral (phaseFromAxis spliceAxis) (fun _ => 1) (pairCoordinates y)) =
      spliceTail ∘ ridgeSum := by
  funext y
  rw [gaussianPreintegral_splicePhase, Fin.sum_univ_two]
  simp [pairCoordinates, ridgeSum]

end Cubature
