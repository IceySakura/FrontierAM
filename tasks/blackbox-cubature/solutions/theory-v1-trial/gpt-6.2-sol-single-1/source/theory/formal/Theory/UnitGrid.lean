import Theory.Nonadaptive

/-! An equally spaced radial-mass grid with one shared uniform phase. -/

open MeasureTheory Set
open scoped unitInterval

namespace Cubature

noncomputable section

def gridEdge (B i : ℕ) : ℝ := (i : ℝ) / (B : ℝ)

def gridTest {B : ℕ} (i : Fin B) (v : I) : ℝ := gridEdge B i.val + (1 / (B : ℝ)) * v.val

def gridReadout {B : ℕ} (answers : Fin B → Bool) : ℝ :=
  (1 / (B : ℝ)) * ∑ i, bitvalue (answers i)

def gridFraction (B : ℕ) (q : ℝ) (v : I) : ℝ :=
  gridReadout (fun i : Fin B => decide (gridTest i v < q))

private def step (q t : ℝ) : ℝ := (Iio q).indicator (fun _ => 1) t

private theorem bitvalue_step (t q : ℝ) : bitvalue (decide (t < q)) = step q t := by
  by_cases h : t < q <;> simp [bitvalue, step, indicator, h]

private theorem stepRange (q t : ℝ) : step q t ∈ Icc 0 1 := by
  by_cases h : t < q <;> simp [step, indicator, h]

private theorem stepAntitone (q : ℝ) : Antitone (step q) := by
  intro x y hxy
  by_cases h : y < q
  · simp [step, indicator, h, hxy.trans_lt h]
  · simp [step, indicator, h]
    split <;> norm_num

theorem gridTest_between {B : ℕ} (hB : 0 < B) (i : Fin B) (v : I) :
    gridTest i v ∈ Icc (gridEdge B i.val) (gridEdge B (i.val+1)) := by
  have hδ : 0 ≤ 1 / (B : ℝ) := by positivity
  constructor
  · exact le_add_of_nonneg_right (mul_nonneg hδ v.2.1)
  · have h := mul_le_of_le_one_right hδ v.2.2
    simp only [gridTest, gridEdge, Nat.cast_add, Nat.cast_one, add_div]
    linarith

theorem gridTest_mem {B : ℕ} (hB : 0 < B) (i : Fin B) (v : I) :
    gridTest i v ∈ I := by
  have h := gridTest_between hB i v
  refine ⟨(by unfold gridEdge; positivity : 0 ≤ gridEdge B i.val).trans h.1,
    h.2.trans ?_⟩
  exact (div_le_one (by positivity)).mpr (by exact_mod_cast i.isLt)

theorem gridReadout_mem {B : ℕ} (hB : 0 < B) (answers : Fin B → Bool) :
    gridReadout answers ∈ I := by
  have hvals (i : Fin B) : bitvalue (answers i) ∈ Icc 0 1 := by
    cases answers i <;> norm_num [bitvalue]
  have hpos : 0 < (B : ℝ) := by positivity
  have hs : (∑ i : Fin B, bitvalue (answers i)) ≤ (B : ℝ) := by
    exact (Finset.sum_le_sum (fun i _ => (hvals i).2)).trans_eq (by simp)
  unfold gridReadout
  refine ⟨mul_nonneg (by positivity) (Finset.sum_nonneg fun i _ => (hvals i).1), ?_⟩
  exact (mul_le_mul_of_nonneg_left hs (by positivity)).trans_eq (by field_simp)

/-- Measurability of the grid readout with a measurable threshold and phase. -/
theorem measurable_gridFraction_comp {X : Type*} [MeasurableSpace X] (B : ℕ)
    {q : X → ℝ} {v : X → I} (hq : Measurable q) (hv : Measurable v) :
    Measurable (fun x => gridFraction B (q x) (v x)) := by
  simp_rw [gridFraction, gridReadout, bitvalue_step]
  apply Measurable.const_mul
  refine Finset.measurable_sum _ (fun i _ => ?_)
  have hV : Measurable (fun x => (v x).val) := measurable_subtype_coe.comp hv
  have hG : Measurable (fun x => gridTest i (v x)) := by
    unfold gridTest
    exact measurable_const.add (hV.const_mul _)
  exact Measurable.ite (measurableSet_lt hG hq) measurable_const measurable_const

theorem integrable_gridFraction {B : ℕ} (hB : 0 < B) (q : ℝ) :
    Integrable (fun v : I => gridFraction B q v) := by
  have hM : Measurable (fun v : I => gridFraction B q v) :=
    measurable_gridFraction_comp B (q := fun _ : I => q) (v := id) measurable_const measurable_id
  refine (integrable_const (1:ℝ)).mono' hM.aestronglyMeasurable ?_
  exact Filter.Eventually.of_forall fun v => by
    have hr := gridReadout_mem hB (fun j => decide (gridTest j v < q))
    change ‖gridReadout (fun j : Fin B => decide (gridTest j v < q))‖ ≤ 1
    rw [Real.norm_of_nonneg hr.1]
    exact hr.2

private theorem step_int (q a b : ℝ) :
    IntervalIntegrable (step q) volume a b := by
  rw [intervalIntegrable_iff]
  exact ((intervalIntegrable_const (μ := volume) (a := a) (b := b) (c := (1:ℝ))).def').indicator
    measurableSet_Iio

private theorem step_integral_unit (q : I) : ∫ t in (0:ℝ)..1, step q t = (q : ℝ) := by
  have h01 : (0:ℝ) ≤ 1 := by norm_num
  unfold step
  rw [intervalIntegral.integral_of_le h01, integral_indicator_const (1:ℝ) measurableSet_Iio,
    smul_eq_mul, mul_one, measureReal_def, Measure.restrict_apply measurableSet_Iio]
  have hE : Iio (q : ℝ) ∩ Ioc (0:ℝ) 1 = Ioo 0 (q : ℝ) := by
    ext t
    exact ⟨fun h => ⟨h.2.1,h.1⟩, fun h => ⟨h.2,h.1,h.2.le.trans q.2.2⟩⟩
  rw [hE, Real.volume_Ioo, sub_zero, ENNReal.toReal_ofReal q.2.1]

private theorem gridEdge_end {B : ℕ} (hB : 0 < B) : gridEdge B B = 1 :=
  div_self (by positivity)

private theorem gridEdge_diff {B : ℕ} (i : ℕ) :
    gridEdge B (i+1) - gridEdge B i = 1 / (B : ℝ) := by
  simp only [gridEdge, Nat.cast_add, Nat.cast_one, add_div, add_sub_cancel_left]

private theorem edge_mono {B : ℕ} (i : ℕ) : gridEdge B i ≤ gridEdge B (i+1) := by
  rw [← sub_nonneg, gridEdge_diff]
  positivity

private theorem sum_step_integrals {B : ℕ} (hB : 0 < B) (q : I) :
    ∑ i : Fin B, ∫ t in gridEdge B i.val..gridEdge B (i.val+1), step q t = (q : ℝ) := by
  rw [Fin.sum_univ_eq_sum_range
    (fun i : ℕ => ∫ t in gridEdge B i..gridEdge B (i+1), step q t)]
  rw [intervalIntegral.sum_integral_adjacent_intervals (fun i _ => step_int q _ _)]
  rw [gridEdge_end hB, show gridEdge B 0 = 0 by simp [gridEdge], step_integral_unit]

theorem gridFraction_pathwise {B : ℕ} (hB : 0 < B) (q v : I) :
    |gridFraction B q v - q.val| ≤ 1 / (B : ℝ) := by
  let δ : ℝ := 1 / (B : ℝ)
  have hδ : 0 ≤ δ := by dsimp [δ]; positivity
  let f := step q.val
  let J (i : Fin B) := ∫ t in gridEdge B i.val..gridEdge B (i.val+1), f t
  have hl (i : Fin B) : δ * f (gridEdge B (i.val+1)) ≤ J i := by
    have hl' := intervalIntegral.integral_mono_on (edge_mono (B := B) i.val)
      (intervalIntegrable_const (c := f (gridEdge B (i.val+1)))) (step_int q _ _)
      (fun t ht => stepAntitone q ht.2)
    simpa only [intervalIntegral.integral_const, gridEdge_diff, smul_eq_mul] using hl'
  have hu (i : Fin B) : J i ≤ δ * f (gridEdge B i.val) := by
    have hu' := intervalIntegral.integral_mono_on (edge_mono (B := B) i.val)
      (step_int q _ _) (intervalIntegrable_const (c := f (gridEdge B i.val)))
      (fun t ht => stepAntitone q ht.1)
    simpa only [intervalIntegral.integral_const, gridEdge_diff, smul_eq_mul] using hu'
  have hlast (i : Fin B) : |δ * f (gridTest i v) - J i| ≤
      δ * (f (gridEdge B i.val) - f (gridEdge B (i.val+1))) := by
    have hx := gridTest_between hB i v
    have h₁ := mul_le_mul_of_nonneg_left (stepAntitone q hx.2) hδ
    have h₂ := mul_le_mul_of_nonneg_left (stepAntitone q hx.1) hδ
    rw [abs_le, mul_sub]
    exact ⟨by linarith [hu i], by linarith [hl i]⟩
  unfold gridFraction gridReadout
  simp_rw [bitvalue_step]
  change |δ * (∑ i : Fin B, f (gridTest i v)) - q.val| ≤ δ
  rw [← sum_step_integrals hB q, Finset.mul_sum, ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ i : Fin B, |δ * f (gridTest i v) - J i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i : Fin B, δ * (f (gridEdge B i.val) - f (gridEdge B (i.val+1))) :=
      Finset.sum_le_sum fun i _ => hlast i
    _ = δ * (f (gridEdge B 0) - f (gridEdge B B)) := by
      rw [← Finset.mul_sum, Fin.sum_univ_eq_sum_range (fun i : ℕ =>
        f (gridEdge B i) - f (gridEdge B (i+1))), Finset.sum_range_sub']
    _ ≤ δ := by
      have h₁ := (stepRange (q : ℝ) (gridEdge B 0)).2
      have h₂ := (stepRange (q : ℝ) (gridEdge B B)).1
      exact mul_le_of_le_one_right hδ (by linarith)

private theorem uniform_affine (f : ℝ → ℝ) (a δ : ℝ) :
    (∫ v : I, δ * f (a+δ*v.val)) = ∫ t in a..a+δ, f t := by
  change (∫ v : I, δ * f (a+δ*v.val) ∂((volume : Measure ℝ).comap Subtype.val)) = _
  rw [integral_subtype_comap (measurableSet_Icc (a := (0:ℝ)) (b := 1))
    (fun v => δ*f (a+δ*v)), integral_Icc_eq_integral_Ioc,
    ← intervalIntegral.integral_of_le (by norm_num : (0:ℝ) ≤ 1), intervalIntegral.integral_const_mul]
  simpa only [mul_zero,zero_add,mul_one,add_comm a,smul_eq_mul] using
    (intervalIntegral.smul_integral_comp_mul_add f (a := 0) (b := 1) δ a)

theorem gridFraction_mean {B : ℕ} (hB : 0 < B) (q : I) :
    (∫ v : I, gridFraction B q v) = q.val := by
  let δ : ℝ := 1 / (B : ℝ)
  have hv (i : Fin B) : Integrable (fun v : I => δ * step q (gridTest i v)) := by
    have hg : Measurable (fun v : I => gridTest i v) := by unfold gridTest; fun_prop
    have hM : Measurable (step q) := measurable_const.indicator measurableSet_Iio
    refine (integrable_const δ).mono' ((hM.comp hg).const_mul δ).aestronglyMeasurable ?_
    filter_upwards [] with v
    rw [Real.norm_of_nonneg (mul_nonneg (by dsimp [δ]; positivity) (stepRange _ _).1)]
    exact mul_le_of_le_one_right (by dsimp [δ]; positivity) (stepRange _ _).2
  unfold gridFraction gridReadout
  simp_rw [bitvalue_step, Finset.mul_sum]
  change (∫ v : I, ∑ i : Fin B, δ * step q (gridTest i v)) = _
  rw [integral_finsetSum _ (fun i _ => hv i)]
  conv_rhs => rw [← sum_step_integrals hB q]
  refine Finset.sum_congr rfl fun i _ => ?_
  change (∫ v : I, δ * step q (gridEdge B i.val + δ * v.val)) = _
  have he : gridEdge B i.val + δ = gridEdge B (i.val+1) := by linarith [gridEdge_diff (B := B) i.val]
  rw [uniform_affine, he]

end

end Cubature
