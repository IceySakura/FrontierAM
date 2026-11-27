import Theory.Adaptive
import Theory.SphereCharts
import Mathlib.MeasureTheory.Function.Floor

/-! Tensor-linear interpolation on the cubical charts. -/

open MeasureTheory Set
open scoped unitInterval

namespace Cubature

noncomputable section

def cell (m : ℕ) (t : ℝ) : ℕ := min (m-1) ⌊(m : ℝ) * t⌋₊

private def gridLow {m : ℕ} (hm : 0 < m) (t : ℝ) : Fin (m+1) :=
  ⟨cell m t, lt_of_le_of_lt (min_le_left _ _) (by omega)⟩

private def gridHigh {m : ℕ} (hm : 0 < m) (t : ℝ) : Fin (m+1) :=
  ⟨cell m t + 1, by have := min_le_left (m-1) ⌊(m:ℝ)*t⌋₊; unfold cell; omega⟩

private def frac (m : ℕ) (t : ℝ) : ℝ := (m : ℝ) * t - (cell m t : ℝ)

theorem cell_bounds {m : ℕ} (hm : 0 < m) (t : I) :
    (cell m t : ℝ) ≤ (m : ℝ) * t.val ∧ (m : ℝ) * t.val ≤ (cell m t : ℝ) + 1 := by
  have h₁ : (m : ℝ) * t.val ≤ m := mul_le_of_le_one_right (Nat.cast_nonneg _) t.2.2
  have h₀ : 0 ≤ (m : ℝ) * t.val := mul_nonneg (Nat.cast_nonneg _) t.2.1
  constructor
  · exact (Nat.cast_le.mpr (min_le_right _ _)).trans (Nat.floor_le h₀)
  · by_cases hf : ⌊(m : ℝ) * t.val⌋₊ ≤ m - 1
    · simpa only [cell, min_eq_right hf] using (Nat.lt_floor_add_one ((m:ℝ)*t.val)).le
    · rw [cell, min_eq_left (by omega)]
      have hmc : ((m - 1 : ℕ) : ℝ) + 1 = m := by
        exact_mod_cast (Nat.sub_add_cancel hm : m - 1 + 1 = m)
      exact h₁.trans_eq hmc.symm

private def weight1 {m : ℕ} (hm : 0 < m) (i : Fin (m+1)) (t : ℝ) : ℝ :=
  (if i = gridLow hm t then 1 - frac m t else 0) +
    (if i = gridHigh hm t then frac m t else 0)

private def value1 {m : ℕ} (hm : 0 < m) (i : Fin (m+1)) : I :=
  ⟨(i.val : ℝ) / (m : ℝ), ⟨by positivity, (div_le_one (by positivity)).2
    (Nat.cast_le.mpr (Nat.le_of_lt_succ i.isLt))⟩⟩

theorem measurable_cell (m : ℕ) : Measurable (cell m) :=
  (measurable_from_nat (f := min (m-1))).comp (Nat.measurable_floor.comp (by fun_prop))

private theorem measurable_weight1 {m : ℕ} (hm : 0 < m) (i : Fin (m+1)) :
    Measurable (weight1 hm i) := by
  have hmf : Measurable (frac m) := by
    exact (measurable_id.const_mul _).sub
      ((measurable_from_nat (f := fun n => (n : ℝ))).comp (measurable_cell m))
  have hL : {t : ℝ | i = gridLow hm t} = {t : ℝ | cell m t = i.val} := by
    ext t
    simp only [mem_ofPred_eq, gridLow, Fin.ext_iff, eq_comm]
  have hH : {t : ℝ | i = gridHigh hm t} = {t : ℝ | cell m t + 1 = i.val} := by
    ext t
    simp only [mem_ofPred_eq, gridHigh, Fin.ext_iff, eq_comm]
  have hHL : Measurable (fun t => cell m t + 1) :=
    (measurable_from_nat (f := fun n => n + 1)).comp (measurable_cell m)
  exact (Measurable.ite (hL ▸ (measurable_cell m) (measurableSet_singleton i.val))
    (measurable_const.sub hmf) measurable_const).add
    (Measurable.ite (hH ▸ hHL (measurableSet_singleton i.val)) hmf measurable_const)

private theorem weight1_nonneg {m : ℕ} (hm : 0 < m) (i : Fin (m+1)) (t : I) :
    0 ≤ weight1 hm i t := by
  obtain ⟨h₁, h₂⟩ := cell_bounds hm t
  have h₀ : 0 ≤ frac m t := by unfold frac; linarith
  have h₁' : 0 ≤ 1 - frac m t := by unfold frac; linarith
  unfold weight1
  exact add_nonneg (by split; exacts [h₁', le_rfl]) (by split; exacts [h₀, le_rfl])

private theorem sum_weight1_mul {m : ℕ} (hm : 0 < m) (t : I) (f : Fin (m+1) → ℝ) :
    ∑ i, weight1 hm i t * f i =
      (1 - frac m t) * f (gridLow hm t) + frac m t * f (gridHigh hm t) := by
  simp [weight1, add_mul, Finset.sum_add_distrib, ite_mul]

private theorem sum_weight1 {m : ℕ} (hm : 0 < m) (t : I) :
    ∑ i, weight1 hm i t = 1 := by
  simpa using sum_weight1_mul hm t (fun _ => 1)

private theorem sum_weight1_value {m : ℕ} (hm : 0 < m) (t : I) :
    ∑ i, weight1 hm i t * (value1 hm i : ℝ) = t.val := by
  rw [sum_weight1_mul]
  simp only [value1, gridLow, gridHigh, Nat.cast_add, Nat.cast_one, frac]
  have hM : (m : ℝ) ≠ 0 := by positivity
  field_simp
  ring

private theorem weight1_support {m : ℕ} (hm : 0 < m) (t : I) (i : Fin (m+1))
    (h : weight1 hm i t ≠ 0) :
    |(value1 hm i : ℝ) - t.val| ≤ 1 / (m : ℝ) := by
  have he : i = gridLow hm t ∨ i = gridHigh hm t := by
    by_contra! hn
    exact h (by simp [weight1, hn.1, hn.2])
  obtain ⟨hL, hU⟩ := cell_bounds hm t
  have hM : (0:ℝ) < m := by positivity
  have h₀ : -1 / (m:ℝ) = -(1 / (m:ℝ)) := neg_div _ _
  rw [abs_le, ← h₀]
  rcases he with rfl | rfl
  · change -1/(m:ℝ) ≤ (cell m t : ℝ)/(m:ℝ) - t.val ∧
      (cell m t : ℝ)/(m:ℝ) - t.val ≤ 1/(m:ℝ)
    constructor
    · rw [le_sub_iff_add_le,
        show -1/(m:ℝ) + t.val = (-1 + (m:ℝ)*t.val)/(m:ℝ) by field_simp,
        div_le_div_iff_of_pos_right hM]
      linarith
    · rw [sub_le_iff_le_add]
      rw [show 1/(m:ℝ) + t.val = ((m:ℝ)*t.val + 1)/(m:ℝ) by field_simp; ring]
      exact (div_le_div_iff_of_pos_right hM).mpr (by linarith)
  · simp only [value1, gridHigh, Nat.cast_add, Nat.cast_one]
    constructor
    · rw [le_sub_iff_add_le,
        show -1/(m:ℝ) + t.val = (-1 + (m:ℝ)*t.val)/(m:ℝ) by field_simp,
        div_le_div_iff_of_pos_right hM]
      linarith
    · rw [sub_le_iff_le_add,
        show 1/(m:ℝ) + t.val = ((m:ℝ)*t.val + 1)/(m:ℝ) by field_simp; ring,
        div_le_div_iff_of_pos_right hM]
      linarith

abbrev FaceGrid (p m : ℕ) := Fin p → Fin (m+1)

private def tensorWeight {p m : ℕ} (hm : 0 < m) (v : FaceGrid p m) (x : Cube p) : ℝ :=
  ∏ j, weight1 hm (v j) (x j)

private theorem tensorWeight_nonneg {p m : ℕ} (hm : 0 < m) (v : FaceGrid p m) (x : Cube p) :
    0 ≤ tensorWeight hm v x := Finset.prod_nonneg fun j _ => weight1_nonneg hm (v j) (x j)

private theorem sum_tensorWeight {p m : ℕ} (hm : 0 < m) (x : Cube p) :
    ∑ v : FaceGrid p m, tensorWeight hm v x = 1 := by
  calc
    _ = ∏ j : Fin p, ∑ i, weight1 hm i (x j) :=
      (Fintype.prod_sum (fun j i => weight1 hm i (x j))).symm
    _ = _ := by simp only [sum_weight1, Finset.prod_const_one]

private theorem sum_tensorWeight_coordinate {p m : ℕ} (hm : 0 < m) (x : Cube p) (i : Fin p) :
    ∑ v : FaceGrid p m, tensorWeight hm v x * ((value1 hm (v i) : ℝ) - (x i : ℝ)) = 0 := by
  let c : Fin p → Fin (m+1) → ℝ := fun j a =>
    weight1 hm a (x j) * (if j = i then (value1 hm a : ℝ) - (x i : ℝ) else 1)
  have hc (v : FaceGrid p m) : tensorWeight hm v x * ((value1 hm (v i) : ℝ) - (x i : ℝ)) =
       ∏ j, c j (v j) := by
    unfold c
    rw [Finset.prod_mul_distrib, ← tensorWeight, Fintype.prod_ite_eq' i]
  simp_rw [hc, ← Fintype.prod_sum]
  refine Finset.prod_eq_zero (Finset.mem_univ i) ?_
  simp only [c, ↓reduceIte, mul_sub, Finset.sum_sub_distrib, sum_weight1_value,
    ← Finset.sum_mul, sum_weight1, one_mul, sub_self]

private theorem tensor_error {p m s : ℕ} (hm : 0 < m) {R : Direction (p+1) → ℝ} {A : ℝ}
    (hR : ChartRemainder s A R) (f : Faces p) (x : Cube p) :
    |(∑ v : FaceGrid p m, tensorWeight hm v x * R (faceDirection f (fun i => value1 hm (v i)))) -
      R (faceDirection f x)| ≤ A * (1 / (m : ℝ)) ^ s := by
  obtain ⟨β, hβ⟩ := hR f x
  have hlin : (∑ v : FaceGrid p m, tensorWeight hm v x *
      (∑ j, β j * ((value1 hm (v j) : ℝ) - (x j : ℝ)))) = 0 := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    simp_rw [mul_left_comm _ (β _), ← Finset.mul_sum, sum_tensorWeight_coordinate, mul_zero,
      Finset.sum_const_zero]
  let err : FaceGrid p m → ℝ := fun v =>
    R (faceDirection f (fun i => value1 hm (v i))) - R (faceDirection f x) -
      ∑ j, β j * ((value1 hm (v j) : ℝ) - (x j : ℝ))
  have heq :
      (∑ v : FaceGrid p m, tensorWeight hm v x * R (faceDirection f (fun i => value1 hm (v i)))) -
      R (faceDirection f x) = ∑ v, tensorWeight hm v x * err v := by
    have he (v : FaceGrid p m) : tensorWeight hm v x * err v =
        (tensorWeight hm v x * R (faceDirection f (fun i => value1 hm (v i))) -
          tensorWeight hm v x * R (faceDirection f x)) -
        tensorWeight hm v x * (∑ j, β j * ((value1 hm (v j) : ℝ) - (x j : ℝ))) := by
      unfold err
      rw [mul_sub, mul_sub]
    simp_rw [he, Finset.sum_sub_distrib, hlin, sub_zero]
    rw [← Finset.sum_mul, sum_tensorWeight, one_mul]
  rw [heq]
  calc
    _ ≤ ∑ v : FaceGrid p m, |tensorWeight hm v x * err v| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ v : FaceGrid p m, tensorWeight hm v x * (A * (1 / (m : ℝ)) ^ s) := by
      refine Finset.sum_le_sum fun v _ => ?_
      rw [abs_mul, abs_of_nonneg (tensorWeight_nonneg hm v x)]
      by_cases h : tensorWeight hm v x = 0
      · simp [h]
      · apply mul_le_mul_of_nonneg_left _ (tensorWeight_nonneg hm v x)
        refine hβ _ _ (by positivity) (fun j => weight1_support hm (x j) (v j) ?_)
        intro hj
        exact h (Finset.prod_eq_zero (Finset.mem_univ j) hj)
    _ = _ := by rw [← Finset.sum_mul, sum_tensorWeight, one_mul]

def sphereGridCount (p m : ℕ) : ℕ := (p+1) * 2 * (m+1)^p

private def gridEquiv (p m : ℕ) : Fin (sphereGridCount p m) ≃ Faces p × FaceGrid p m :=
  Fintype.equivOfCardEq (by
    calc
      Fintype.card (Fin (sphereGridCount p m)) = sphereGridCount p m := Fintype.card_fin _
      _ = Fintype.card (Faces p × FaceGrid p m) := by
        simp [sphereGridCount, Fintype.card_fin])

private def sphereWeight {p m : ℕ} (hm : 0 < m) (v : Faces p × FaceGrid p m)
    (x : Direction (p+1)) : ℝ :=
  faceWeight v.1 x * tensorWeight hm v.2 (faceParam v.1 x)

private theorem sum_sphereWeight {p m : ℕ} (hm : 0 < m) (x : Direction (p+1)) :
    ∑ v : Faces p × FaceGrid p m, sphereWeight hm v x = 1 := by
  rw [Fintype.sum_prod_type]
  simp_rw [sphereWeight, ← Finset.mul_sum, sum_tensorWeight, mul_one, sum_faceWeight]

def sphereGrid (p m : ℕ) (hm : 0 < m) : Predictor (Direction (p+1)) where
  count := sphereGridCount p m
  point := fun i =>
    let v := gridEquiv p m i
    faceDirection v.1 (fun j => value1 hm (v.2 j))
  weight := fun i => sphereWeight hm (gridEquiv p m i)
  measurable_weight := by
    intro i
    unfold sphereWeight tensorWeight
    apply (measurable_faceWeight _).mul
    exact Finset.measurable_prod _ fun _ _ =>
      (measurable_weight1 hm _).comp (measurable_faceParam_coord _ _)
  weight_nonneg := fun _ x => mul_nonneg (faceWeight_nonneg _ x) (tensorWeight_nonneg hm _ _)
  sum_weight := fun x => (Fintype.sum_equiv (gridEquiv p m) _ _ (fun _ => rfl)).trans
    (sum_sphereWeight hm x)

theorem sphereGrid_error {p m s : ℕ} (hm : 0 < m) {R : Direction (p+1) → ℝ} {A : ℝ}
    (hR : ChartRemainder s A R) (x : Direction (p+1)) :
    |(sphereGrid p m hm).predict (R ∘ (sphereGrid p m hm).point) x - R x| ≤
      A * (1 / (m : ℝ)) ^ s := by
  let S (f : Faces p) :=
    ∑ v : FaceGrid p m, tensorWeight hm v (faceParam f x) *
      R (faceDirection f (fun j => value1 hm (v j)))
  have hf : (sphereGrid p m hm).predict (R ∘ (sphereGrid p m hm).point) x =
      ∑ f : Faces p, faceWeight f x * S f := by
    change (∑ i : Fin (sphereGridCount p m),
      sphereWeight hm (gridEquiv p m i) x *
        R (faceDirection (gridEquiv p m i).1 (fun j => value1 hm ((gridEquiv p m i).2 j)))) = _
    rw [Fintype.sum_equiv (gridEquiv p m) _ (fun v => sphereWeight hm v x *
      R (faceDirection v.1 (fun j => value1 hm (v.2 j)))) (fun _ => rfl),
      Fintype.sum_prod_type]
    simp_rw [sphereWeight, mul_assoc, ← Finset.mul_sum, S]
  have he : R x = ∑ f : Faces p, faceWeight f x * R x := by
    rw [← Finset.sum_mul, sum_faceWeight, one_mul]
  rw [hf, he, ← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ f : Faces p, |faceWeight f x * (S f - R x)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ f : Faces p, faceWeight f x * (A * (1 / (m : ℝ)) ^ s) := by
      refine Finset.sum_le_sum fun f _ => ?_
      rw [abs_mul, abs_of_nonneg (faceWeight_nonneg f x)]
      by_cases ha : faceActive f x
      · exact mul_le_mul_of_nonneg_left
          (by simpa only [faceDirection_faceParam ha] using tensor_error hm hR f (faceParam f x))
          (faceWeight_nonneg f x)
      · simp [faceWeight_eq_zero ha]
    _ = _ := by rw [← Finset.sum_mul, sum_faceWeight, one_mul]

def dyadicSphereGrid (p k : ℕ) : Predictor (Direction (p+1)) :=
  sphereGrid p (2^k) (by positivity)

theorem dyadicSphereGrid_error {p s : ℕ} {R : Direction (p+1) → ℝ} {A : ℝ}
    (hR : ChartRemainder s A R) (k : ℕ) (x : Direction (p+1)) :
    |(dyadicSphereGrid p k).predict (R ∘ (dyadicSphereGrid p k).point) x - R x| ≤
      precision A s k := by
  simpa only [dyadicSphereGrid, precision, Nat.cast_pow, Nat.cast_ofNat, div_pow, one_pow,
    ← pow_mul, Nat.mul_comm k s, mul_one_div] using
      sphereGrid_error (p := p) (m := 2^k) (by positivity) hR x

end

end Cubature
