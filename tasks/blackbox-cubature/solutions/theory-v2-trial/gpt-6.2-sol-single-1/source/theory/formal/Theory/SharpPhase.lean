import Theory.Axis
import Theory.Coordinates
import Theory.Growth

/-! All input hypotheses of the counterexample, including the trace terms retained
when the pure integrated-coordinate regularity is lowered. -/

open Filter MeasureTheory ProbabilityTheory
open scoped ContDiff ENNReal

namespace Cubature

section

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

private noncomputable def linearJet : ℕ → ℝ → ℝ
  | 0 => id
  | 1 => fun _ => 1
  | _ + 2 => fun _ => 0

private lemma continuous_linearJet (k : ℕ) : Continuous (linearJet k) := by
  cases k with
  | zero => exact continuous_id
  | succ k => cases k <;> exact continuous_const

private lemma polynomialGrowth_linearJet (k : ℕ) : PolynomialGrowth (linearJet k) := by
  cases k with
  | zero => exact polynomialGrowth_id
  | succ k => cases k <;> exact polynomialGrowth_const _

private lemma hasDerivAt_linearJet (k : ℕ) (t : ℝ) :
    HasDerivAt (linearJet k) (linearJet (k + 1) t) t := by
  cases k with
  | zero => exact hasDerivAt_id t
  | succ k => cases k <;> exact hasDerivAt_const t _

private noncomputable def axisJet (q k : ℕ) (t : ℝ) : ℝ :=
  linearJet k t + truncatedPower (q - k) t

private lemma continuous_axisJet (q k : ℕ) : Continuous (axisJet q k) :=
  (continuous_linearJet k).add (continuous_truncatedPower (q - k))

private lemma polynomialGrowth_axisJet (q k : ℕ) : PolynomialGrowth (axisJet q k) :=
  (polynomialGrowth_linearJet k).add (polynomialGrowth_truncatedPower (q - k))

private lemma hasDerivAt_axisJet {q k : ℕ} (hk : k < q) (t : ℝ) :
    HasDerivAt (axisJet q k) (axisJet q (k + 1) t) t := by
  have hd : HasDerivAt (truncatedPower (q - k)) (truncatedPower (q - (k + 1)) t) t := by
    rw [show q - k = (q - (k + 1)) + 1 by omega]
    exact hasDerivAt_truncatedPower_succ (q - (k + 1)) t
  exact (hasDerivAt_linearJet k t).add hd

private lemma axisJet_zero (q : ℕ) : axisJet q 0 = axisProfile q := by
  funext t
  simp [axisJet, axisProfile, linearJet]

private lemma axisJet_one (q : ℕ) (t : ℝ) : axisJet q 1 t = 1 + truncatedPower (q - 1) t := rfl

lemma hasAxisCoordinatePartial (u u' : ℝ → ℝ) (h : ∀ t, HasDerivAt u (u' t) t) :
    HasCoordinatePartial (none : Option ι) (fun x => u (x none)) (fun x => u' (x none)) := by
  intro x
  have hd : HasDerivAt u (u' (x none)) (x none + 0) := by simpa using h (x none)
  change HasDerivAt
    (fun a : ℝ => u ((x + a • (Pi.single none 1 : Option ι → ℝ)) none)) (u' (x none)) 0
  simpa using hd.comp_const_add (x none) 0

lemma hasRetainedCoordinatePartial_axis (u : ℝ → ℝ) (i : ι) :
    HasCoordinatePartial (some i) (fun x : Option ι → ℝ => u (x none)) (fun _ => 0) := by
  intro x
  change HasDerivAt
    (fun a : ℝ => u ((x + a • (Pi.single (some i) 1 : Option ι → ℝ)) none)) 0 0
  simpa using hasDerivAt_const 0 (u (x none))

private lemma axisPartialChain (q k : ℕ) (w : List (Option ι))
    (hw : k + w.count none ≤ q) :
    ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧
      ContinuousPartialChain w (fun x => axisJet q k (x none)) (fun x => u (x none)) := by
  have hc (k : ℕ) : Continuous (fun x : Option ι → ℝ => axisJet q k (x none)) :=
    (continuous_axisJet q k).comp (continuous_apply none)
  induction w generalizing k with
  | nil =>
    exact ⟨axisJet q k, continuous_axisJet q k, polynomialGrowth_axisJet q k, hc k, rfl⟩
  | cons i w ih =>
    cases i with
    | some i =>
      refine ⟨fun _ => 0, continuous_const, polynomialGrowth_const 0, hc k, (fun _ => 0),
        hasRetainedCoordinatePartial_axis _ i, ?_⟩
      simpa using (ContinuousPartialChain.const w 0)
    | none =>
      have ht : k + 1 + w.count none ≤ q := by
        simp at hw
        omega
      obtain ⟨u, hu, hp, hc'⟩ := ih (k + 1) ht
      exact ⟨u, hu, hp, hc k, (fun x => axisJet q (k + 1) (x none)),
        hasAxisCoordinatePartial _ _ (hasDerivAt_axisJet (by omega)), hc'⟩

def retainedSum (x : Option ι → ℝ) : ℝ := ∑ i : ι, x (some i)

lemma contDiff_retainedSum (n : ℕ∞ω) :
    ContDiff ℝ n (retainedSum : (Option ι → ℝ) → ℝ) :=
  ContDiff.sum fun i _ => contDiff_apply ℝ (fun _ : Option ι => ℝ) (some i)

lemma polynomialGrowth_retainedSum : PolynomialGrowth (retainedSum : (Option ι → ℝ) → ℝ) :=
  polynomialGrowth_finset_sum Finset.univ (fun (i : ι) (x : Option ι → ℝ) => x (some i))
    (fun i _ => polynomialGrowth_id.comp_coord (some i))

lemma hasCoordinatePartial_retainedSum (i : Option ι) :
    HasCoordinatePartial i retainedSum (fun _ => if i = none then 0 else 1) := by
  intro x
  cases i with
  | none =>
    change HasDerivAt
      (fun a : ℝ => ∑ j : ι, ((x + a • (Pi.single none 1 : Option ι → ℝ)) (some j)))
      0 0
    simpa [retainedSum] using hasDerivAt_const 0 (retainedSum x)
  | some i =>
    change HasDerivAt
      (fun a : ℝ => ∑ j : ι, ((x + a • (Pi.single (some i) 1 : Option ι → ℝ)) (some j))) 1 0
    have h (a : ℝ) :
        (∑ j : ι, ((x + a • (Pi.single (some i) 1 : Option ι → ℝ)) (some j))) =
          retainedSum x + a := by
      unfold retainedSum
      rw [← sum_single_axis i (fun j => x (some j)) a]
      simp only [Pi.add_apply, Pi.smul_apply, Pi.single_apply, Option.some.injEq]
    simp_rw [h]
    exact (hasDerivAt_id' 0).const_add (retainedSum x)

lemma retainedSumChain (i : Option ι) (w : List (Option ι)) :
    ∃ c : ℝ, ContinuousPartialChain (i :: w) retainedSum (fun _ => c) := by
  let c : ℝ := if i = none then 0 else 1
  exact ⟨if w = [] then c else 0, (contDiff_retainedSum 0).continuous, (fun _ => c),
    hasCoordinatePartial_retainedSum i, ContinuousPartialChain.const w c⟩

private lemma phaseChain (q : ℕ) (w : List (Option ι)) (hw : w.count none ≤ q) :
    ∃ g : (Option ι → ℝ) → ℝ, ContinuousPartialChain w (sharpPhase q) g ∧
      PolynomialGrowth g ∧
      (w ≠ [] → ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧ ∀ x, g x = u (x none)) := by
  have h := axisPartialChain q 0 w (by simpa using hw)
  simp only [axisJet_zero] at h
  obtain ⟨u, hcu, hpu, h⟩ := h
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

lemma gaussianMixedInput_sharpPhase (q : ℕ) (r : ι → ℕ) :
    GaussianMixedInput (fun i : Option ι => i.elim q r) (sharpPhase q) := by
  intro w hw
  have he (v : List (Option ι)) :
      @List.count _ instBEqOfDecidableEq none v = v.count none := by
    induction v with
    | nil => rfl
    | cons i v ih => cases i <;> simp [List.count_cons, ih]
  obtain ⟨g, hc, hp, _⟩ := phaseChain q w ((he w) ▸ hw none)
  exact ⟨g, hc.partials, hc.continuous_end, hc.weak, hp,
    fun p hf => hp.memLp_gaussianProduct hc.continuous_end p hf⟩

lemma gaussianMixedInput_one (r : Option ι → ℕ) :
    GaussianMixedInput r (fun _ => 1) := by
  intro w _
  let a : ℝ := if w = [] then 1 else 0
  have h := ContinuousPartialChain.const w 1
  exact ⟨fun _ => a, h.partials, continuous_const, h.weak, polynomialGrowth_const a,
    fun _ _ => memLp_const a⟩

lemma contDiff_sharpPhase (q : ℕ) : ContDiff ℝ (q : ℕ) (sharpPhase q : (Option ι → ℝ) → ℝ) :=
  ((contDiff_axisProfile q).comp (contDiff_apply ℝ (fun _ : Option ι => ℝ) none)).sub
    (contDiff_retainedSum q)

lemma hasDerivAt_sharpPhase_fiber {q : ℕ} (hq : 1 ≤ q) (t : ℝ) (y : ι → ℝ) :
    HasDerivAt (fun u => sharpPhase q (fiberPoint u y))
      (1 + truncatedPower (q - 1) t) t :=
  (hasDerivAt_axisProfile hq t).sub_const (∑ i, y i)

lemma partialsAlong_sharpPhase_axis {q : ℕ} (hq : 1 ≤ q) (x : Option ι → ℝ) :
    partialsAlong [none] (sharpPhase q) x = 1 + truncatedPower (q - 1) (x none) := by
  have hd := (hasAxisCoordinatePartial (ι := ι) _ _ (hasDerivAt_axisProfile hq)).sub
    (hasCoordinatePartial_retainedSum (none : Option ι))
  exact (hd x).lineDeriv.trans (by simp)

noncomputable def sharpRoot (q : ℕ) (y : ι → ℝ) : ℝ :=
  (axisOrderIso q).symm (∑ i, y i)

lemma continuous_sharpRoot (q : ℕ) : Continuous (sharpRoot q : (ι → ℝ) → ℝ) :=
  (axisOrderIso q).symm.continuous.comp (continuous_finsetSum _ fun i _ => continuous_apply i)

lemma tendsto_sharpPhase_fiber (q : ℕ) (y : ι → ℝ) :
    Tendsto (fun t => sharpPhase q (fiberPoint t y)) atTop atTop := by
  apply tendsto_atTop.mpr
  intro b
  filter_upwards [eventually_ge_atTop (b + ∑ i, y i)] with t ht
  change b ≤ t + truncatedPower q t - ∑ i, y i
  linarith only [ht, truncatedPower_nonneg q t]

lemma sharpRoot_eq_iff (q : ℕ) (t : ℝ) (y : ι → ℝ) :
    sharpPhase q (fiberPoint t y) = 0 ↔ t = sharpRoot q y := by
  change axisProfile q t - ∑ i, y i = 0 ↔ t = sharpRoot q y
  rw [sub_eq_zero]
  exact ⟨fun h => by rw [sharpRoot, ← h, inverse_axisProfile],
    fun h => by rw [h, sharpRoot, axisProfile_inverse]⟩

private lemma phaseDerivativeProduct {q : ℕ} (ws : List (List (Option ι)))
    (hw : ∀ w ∈ ws, w ≠ [] ∧ w.count none ≤ q) :
    ∃ u : ℝ → ℝ, Continuous u ∧ PolynomialGrowth u ∧
      ∀ x : Option ι → ℝ, (ws.map (fun w => partialsAlong w (sharpPhase q) x)).prod = u (x none) := by
  induction ws with
  | nil => exact ⟨fun _ => 1, continuous_const, polynomialGrowth_const 1, fun _ => rfl⟩
  | cons w ws ih =>
    have hw₁ := hw w (by simp)
    obtain ⟨g, hc, _, hg⟩ := phaseChain q w hw₁.2
    obtain ⟨u, hcu, hpu, hu⟩ := hg hw₁.1
    obtain ⟨v, hcv, hpv, hv⟩ := ih (fun w hw' => hw w (List.mem_cons_of_mem _ hw'))
    refine ⟨fun t => u t * v t, hcu.mul hcv, hpu.mul hpv, fun x => ?_⟩
    simp only [List.map_cons, List.prod_cons, hc.partials.eq_partialsAlong, hu, hv]

lemma boundaryFactor_sharpPhase {q : ℕ} (hq : 1 ≤ q) (w : List (Option ι))
    (ws : List (List (Option ι))) (hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ q) (b c : ℕ) :
    Continuous (boundaryFactor (sharpPhase q) (fun _ => 1) (sharpRoot q) w ws b c) ∧
    ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor (sharpPhase q) (fun _ => 1)
      (sharpRoot q) w ws b c y| ≤ C := by
  let a : ℝ := if w = [] then 1 else 0
  have ha : partialsAlong w (fun _ => 1) = fun _ => a :=
    (ContinuousPartialChain.const w 1).partials.eq_partialsAlong
  obtain ⟨u, hcu, hpu, hu⟩ := phaseDerivativeProduct ws hw
  obtain ⟨poly, hcp, hpp, hp⟩ := exists_density_polynomial c
  let V : ℝ → ℝ := fun t => (a * u t) * poly t
  have hVp : PolynomialGrowth V := ((polynomialGrowth_const a).mul hpu).mul hpp
  let G : ℝ → ℝ := fun t => V t * Real.exp (-(t ^ 2 / 2))
  let D : ℝ → ℝ := fun t => (1 + truncatedPower (q - 1) t) ^ b
  have hD (t : ℝ) : 1 ≤ D t :=
    one_le_pow₀ (le_add_of_nonneg_right (truncatedPower_nonneg _ _))
  have hrepr : boundaryFactor (sharpPhase q) (fun _ => 1) (sharpRoot q) w ws b c =
      fun y => G (sharpRoot q y) / D (sharpRoot q y) := by
    funext y
    simp only [boundaryFactor, ha, hu, partialsAlong_sharpPhase_axis hq, fiberPoint,
      Option.elim_none, hp, G, D, V]
    ring
  rw [hrepr]
  constructor
  · have hG : Continuous G := by
      exact (((continuous_const.mul hcu).mul hcp).mul (by fun_prop))
    have hDc : Continuous D :=
      (continuous_const.add (continuous_truncatedPower (q - 1))).pow b
    exact (hG.comp (continuous_sharpRoot q)).div (hDc.comp (continuous_sharpRoot q))
      (fun y => (zero_lt_one.trans_le (hD (sharpRoot q y))).ne')
  · obtain ⟨B, hB, hGB⟩ := hVp.gaussian_damped_bound
    refine ⟨B, hB, fun y => ?_⟩
    rw [abs_div, abs_of_nonneg (zero_le_one.trans (hD _))]
    exact (div_le_self (abs_nonneg _) (hD _)).trans (hGB _)

lemma contDiff_preintegrated_sharpPhase {q : ℕ} (hq : 1 ≤ q) :
    ContDiff ℝ (q : ℕ) (gaussianPreintegral (sharpPhase q) (fun _ => 1) : (ι → ℝ) → ℝ) := by
  simp_rw [show gaussianPreintegral (sharpPhase q) (fun _ => 1) =
      (fun y : ι → ℝ => integratedProfile q (∑ i, y i)) from
    funext (gaussianPreintegral_sharpPhase q)]
  exact (contDiff_integratedProfile hq).comp (ContDiff.sum fun i _ =>
    contDiff_apply ℝ (fun _ : ι => ℝ) i)

lemma no_continuous_top_partial_sharpPhase {q : ℕ} (hq : 1 ≤ q) (w : List ι)
    (hw : w.length = q + 1) :
    ¬ ∃ g, HasPartialsAlong w (gaussianPreintegral (sharpPhase q) (fun _ => 1)) g ∧
      Continuous g := by
  rintro ⟨g, hg, hc⟩
  rw [show gaussianPreintegral (sharpPhase q) (fun _ => 1) =
      (fun y : ι → ℝ => integratedProfile q (∑ i, y i)) from
    funext (gaussianPreintegral_sharpPhase q)] at hg
  have hd := contDiff_of_sumRidge_partials w (by omega) _ hg hc
  rw [hw] at hd
  exact not_contDiff_integratedProfile hq hd

lemma not_mixedContinuous_preintegrated_sharpPhase {q : ℕ} (hq : 1 ≤ q) (r : ι → ℕ)
    (hr : q < ∑ i, r i) :
    ¬ MixedContinuous r (gaussianPreintegral (sharpPhase q) (fun _ => 1)) := by
  intro h
  obtain ⟨w, hw, hrw⟩ := exists_word_within_of_le_sum r (q + 1) hr
  exact no_continuous_top_partial_sharpPhase hq w hw (h w hrw)

end

end Cubature
