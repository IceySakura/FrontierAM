import Theory.LagrangeCell
import Theory.SignedPilot

/-! Higher-order tensor-polynomial reconstruction on the spherical charts. -/

open Set MeasureTheory
open scoped unitInterval

namespace Cubature

noncomputable section

/-- The polynomial in a Taylor model about `x`, with degree `< r` in each coordinate. -/
def chartPolynomial {p r : ℕ} (x : Cube p) (coeff : (Fin p → Fin r) → ℝ) (y : Cube p) : ℝ :=
  ∑ e, coeff e * ∏ j, ((y j : ℝ) - (x j : ℝ)) ^ (e j).val

/-- A uniform order-`r` polynomial remainder on each of the `2d` cubical charts. -/
def ChartPolynomialRemainder {p : ℕ} (r : ℕ) (A : ℝ) (R : Direction (p+1) → ℝ) : Prop :=
  ∀ (f : Faces p) (x : Cube p), ∃ coeff : (Fin p → Fin r) → ℝ,
    ∀ (y : Cube p) (δ : ℝ), 0 ≤ δ →
      (∀ j, |(y j : ℝ) - (x j : ℝ)| ≤ δ) →
      |R (faceDirection f y) - chartPolynomial x coeff y| ≤ A * δ^r

def interpolationNorm (p r : ℕ) : ℝ := ((8 : ℝ)^r)^p

theorem interpolationNorm_pow (p r : ℕ) :
    interpolationNorm p r = (2:ℝ)^(3*r*p) := by
  unfold interpolationNorm
  rw [show (8:ℝ) = (2:ℝ)^3 by norm_num, ← pow_mul, ← pow_mul]
  rw [Nat.mul_assoc]

theorem interpolationNorm_ge_one (p r : ℕ) : 1 ≤ interpolationNorm p r := by
  rw [interpolationNorm_pow]
  exact one_le_pow₀ (by norm_num)

abbrev PolyMesh (p m r : ℕ) := Fin p → PolyAxis m r

private def tensorPolyWeight {p m r : ℕ} (hm : 0 < m) (v : PolyMesh p m r) (x : Cube p) : ℝ :=
  ∏ j, polyWeight hm (v j) (x j)

private def tensorPolyNode {p m r : ℕ} (hm : 0 < m) (hr : 0 < r)
    (v : PolyMesh p m r) : Cube p := fun j => polyNode hm hr (v j)

private theorem tensorPoly_abs_sum {p m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : Cube p) :
    ∑ v : PolyMesh p m r, |tensorPolyWeight hm v x| ≤ interpolationNorm p r := by
  simp only [tensorPolyWeight, Finset.abs_prod]
  calc
    _ = ∏ j : Fin p, ∑ v : PolyAxis m r, |polyWeight hm v (x j)| :=
      (Fintype.prod_sum (fun j v => |polyWeight hm v (x j)|)).symm
    _ ≤ ∏ _ : Fin p, (8:ℝ)^r := Finset.prod_le_prod₀
      (fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _)
      (fun j _ => sum_polyWeight_abs hm hr (x j))
    _ = _ := by simp [interpolationNorm, Fintype.card_fin]

private theorem tensorPoly_moment {p m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : Cube p)
    (e : Fin p → Fin r) :
    (∑ v : PolyMesh p m r, tensorPolyWeight hm v x *
      (∏ j, ((tensorPolyNode hm hr v j : ℝ) - (x j : ℝ)) ^ (e j).val)) =
        ∏ j : Fin p, (0:ℝ) ^ (e j).val := by
  simp only [tensorPolyWeight, tensorPolyNode, ← Finset.prod_mul_distrib]
  calc
    _ = ∏ j : Fin p, ∑ v : PolyAxis m r,
        polyWeight hm v (x j) * ((polyNode hm hr v : ℝ) - (x j : ℝ)) ^ (e j).val :=
      (Fintype.prod_sum (fun j v => polyWeight hm v (x j) *
        ((polyNode hm hr v : ℝ) - (x j : ℝ)) ^ (e j).val)).symm
    _ = _ := Finset.prod_congr rfl (fun j _ => sum_polyWeight_moment hm hr (x j) (e j).isLt)

private theorem tensorPoly_reproduces {p m r : ℕ} (hm : 0 < m) (hr : 0 < r)
    (x : Cube p) (coeff : (Fin p → Fin r) → ℝ) :
    (∑ v : PolyMesh p m r, tensorPolyWeight hm v x * chartPolynomial x coeff (tensorPolyNode hm hr v)) =
      chartPolynomial x coeff x := by
  unfold chartPolynomial
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [mul_left_comm (tensorPolyWeight hm _ x) (coeff _), ← Finset.mul_sum, tensorPoly_moment,
    sub_self]

private theorem tensorPoly_error {p m r : ℕ} (hm : 0 < m) (hr : 0 < r) {A : ℝ} (hA : 0 ≤ A)
    {R : Direction (p+1) → ℝ} (hR : ChartPolynomialRemainder r A R)
    (f : Faces p) (x : Cube p) :
    |(∑ v : PolyMesh p m r, tensorPolyWeight hm v x * R (faceDirection f (tensorPolyNode hm hr v))) -
      R (faceDirection f x)| ≤ interpolationNorm p r * (A * (1/(m:ℝ))^r) := by
  obtain ⟨coeff,hcoeff⟩ := hR f x
  have hex : R (faceDirection f x) = chartPolynomial x coeff x := by
    have ht := hcoeff x 0 le_rfl (fun j => by simp)
    rw [zero_pow hr.ne', mul_zero, abs_nonpos_iff] at ht
    exact sub_eq_zero.mp ht
  rw [hex, ← tensorPoly_reproduces hm hr x coeff, ← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ v : PolyMesh p m r, |tensorPolyWeight hm v x *
        (R (faceDirection f (tensorPolyNode hm hr v)) - chartPolynomial x coeff (tensorPolyNode hm hr v))| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ v : PolyMesh p m r, |tensorPolyWeight hm v x| * (A * (1/(m:ℝ))^r) := by
      refine Finset.sum_le_sum (fun v _ => ?_)
      rw [abs_mul]
      by_cases hv : tensorPolyWeight hm v x = 0
      · simp [hv]
      · refine mul_le_mul_of_nonneg_left
          (hcoeff (tensorPolyNode hm hr v) (1/(m:ℝ)) (by positivity) (fun j => ?_))
          (abs_nonneg _)
        apply polyWeight_support hm hr (x j)
        intro he
        exact hv (Finset.prod_eq_zero (Finset.mem_univ j) he)
    _ ≤ _ := by
      rw [← Finset.sum_mul]
      exact mul_le_mul_of_nonneg_right (tensorPoly_abs_sum hm hr x) (by positivity)

def polynomialGridCount (p m r : ℕ) : ℕ := (p+1)*2*(m*r)^p

private def polyGridEquiv (p m r : ℕ) :
    Fin (polynomialGridCount p m r) ≃ Faces p × PolyMesh p m r :=
  Fintype.equivOfCardEq (by
    calc
      Fintype.card (Fin (polynomialGridCount p m r)) = polynomialGridCount p m r := Fintype.card_fin _
      _ = _ := by simp [polynomialGridCount,Fintype.card_fin])

private def spherePolyWeight {p m r : ℕ} (hm : 0 < m) (v : Faces p × PolyMesh p m r)
    (x : Direction (p+1)) : ℝ :=
  faceWeight v.1 x * tensorPolyWeight hm v.2 (faceParam v.1 x)

private theorem spherePoly_abs_sum {p m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : Direction (p+1)) :
    ∑ v : Faces p × PolyMesh p m r, |spherePolyWeight hm v x| ≤ interpolationNorm p r := by
  rw [Fintype.sum_prod_type]
  simp_rw [spherePolyWeight, abs_mul, abs_of_nonneg (faceWeight_nonneg _ _), ← Finset.mul_sum]
  calc
    _ ≤ ∑ f : Faces p, faceWeight f x * interpolationNorm p r := Finset.sum_le_sum
      (fun f _ => mul_le_mul_of_nonneg_left (tensorPoly_abs_sum hm hr (faceParam f x)) (faceWeight_nonneg _ _))
    _ = _ := by rw [← Finset.sum_mul,sum_faceWeight,one_mul]

def polynomialGrid (p m r : ℕ) (hm : 0 < m) (hr : 0 < r) :
    SignedPredictor (Direction (p+1)) (interpolationNorm p r) where
  count := polynomialGridCount p m r
  point := fun i =>
    let v := polyGridEquiv p m r i
    faceDirection v.1 (tensorPolyNode hm hr v.2)
  weight := fun i => spherePolyWeight hm (polyGridEquiv p m r i)
  measurable_weight := fun i => (measurable_faceWeight _).mul
    (Finset.measurable_prod _ (fun _ _ => (polyWeight_measurable hm _).comp (measurable_faceParam_coord _ _)))
  absWeight_le := fun x => (Fintype.sum_equiv (polyGridEquiv p m r) _ _ (fun _ => rfl)).le.trans
    (spherePoly_abs_sum hm hr x)

theorem polynomialGrid_error {p m r : ℕ} (hm : 0 < m) (hr : 0 < r) {A : ℝ} (hA : 0 ≤ A)
    {R : Direction (p+1) → ℝ} (hR : ChartPolynomialRemainder r A R) (x : Direction (p+1)) :
    |(polynomialGrid p m r hm hr).eval (R ∘ (polynomialGrid p m r hm hr).point) x - R x| ≤
      interpolationNorm p r * (A * (1/(m:ℝ))^r) := by
  let interpolant (f : Faces p) := ∑ v : PolyMesh p m r, tensorPolyWeight hm v (faceParam f x) *
    R (faceDirection f (tensorPolyNode hm hr v))
  have he :
      (polynomialGrid p m r hm hr).eval (R ∘ (polynomialGrid p m r hm hr).point) x =
        ∑ f : Faces p, faceWeight f x * interpolant f := by
    change (∑ i : Fin (polynomialGridCount p m r), spherePolyWeight hm (polyGridEquiv p m r i) x *
      R (faceDirection (polyGridEquiv p m r i).1 (tensorPolyNode hm hr (polyGridEquiv p m r i).2))) = _
    rw [Fintype.sum_equiv (polyGridEquiv p m r) _ (fun v => spherePolyWeight hm v x *
      R (faceDirection v.1 (tensorPolyNode hm hr v.2))) (fun _ => rfl), Fintype.sum_prod_type]
    simp_rw [spherePolyWeight, mul_assoc, ← Finset.mul_sum,interpolant]
  have hf : R x = ∑ f : Faces p, faceWeight f x * R x := by
    rw [← Finset.sum_mul,sum_faceWeight,one_mul]
  rw [he,hf,← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  let E := interpolationNorm p r * (A * (1/(m:ℝ))^r)
  calc
    _ ≤ ∑ f : Faces p, |faceWeight f x * (interpolant f - R x)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ f : Faces p, faceWeight f x * E := by
      refine Finset.sum_le_sum (fun f _ => ?_)
      rw [abs_mul, abs_of_nonneg (faceWeight_nonneg f x)]
      by_cases ha : faceActive f x
      · apply mul_le_mul_of_nonneg_left _ (faceWeight_nonneg f x)
        simpa only [E,interpolant,faceDirection_faceParam ha] using tensorPoly_error hm hr hA hR f (faceParam f x)
      · simp [faceWeight_eq_zero ha]
    _ = _ := by rw [← Finset.sum_mul,sum_faceWeight,one_mul]

def dyadicPolynomialGrid (p r : ℕ) (hr : 0 < r) (j : ℕ) :
    SignedPredictor (Direction (p+1)) (interpolationNorm p r) :=
  polynomialGrid p (2^j) r (by positivity) hr

theorem dyadicPolynomialGrid_error {p r : ℕ} (hr : 0 < r) {A E : ℝ} (hA : 0 ≤ A)
    (hE : interpolationNorm p r * A ≤ E) {R : Direction (p+1) → ℝ}
    (hR : ChartPolynomialRemainder r A R) (j : ℕ) (x : Direction (p+1)) :
    |(dyadicPolynomialGrid p r hr j).eval (R ∘ (dyadicPolynomialGrid p r hr j).point) x - R x| ≤
      precision E r j := by
  have h := polynomialGrid_error (p := p) (m := 2^j) (by positivity) hr hA hR x
  refine h.trans ?_
  simp only [precision, Nat.cast_pow, Nat.cast_ofNat, div_pow, one_pow, ← pow_mul, Nat.mul_comm j r]
  rw [← mul_assoc, mul_one_div]
  exact div_le_div_of_nonneg_right hE (by positivity)

theorem dyadicPolynomial_budget {p r : ℕ} (hp : 0 < p) (hr : 0 < r) (ell k : ℕ) :
    signedPilotBudget (dyadicPolynomialGrid p r hr) r ell k ≤
      2 * ((p+1)*2*(2^k*r)^p) * (r+ell) := by
  let C (k : ℕ) := (p+1)*2*(2^k*r)^p
  have hC (k : ℕ) : 2*C k ≤ C (k+1) := by
    have hB : 2 ≤ (2:ℕ)^p := by
      simpa using pow_le_pow_right' (by norm_num : (1:ℕ) ≤ 2) (Nat.succ_le_of_lt hp)
    calc
      _ ≤ 2^p * C k := Nat.mul_le_mul_right _ hB
      _ = C (k+1) := by
        unfold C
        rw [pow_succ (2:ℕ) k]
        simp only [mul_pow]
        ring
  change signedPilotBudget (dyadicPolynomialGrid p r hr) r ell k ≤ 2*C k*(r+ell)
  induction k with
  | zero => simp [signedPilotBudget]
  | succ k ih =>
    rw [signedPilotBudget]
    change _ + C (k+1)*(r+ell) ≤ _
    calc
      _ ≤ (2*C k)*(r+ell) + C (k+1)*(r+ell) := Nat.add_le_add_right ih _
      _ ≤ _ := by
        rw [← Nat.add_mul]
        exact Nat.mul_le_mul_right _ (by have h := hC k; omega)

end

end Cubature
