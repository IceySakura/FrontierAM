import Theory.SphereGrid
import Mathlib.LinearAlgebra.Lagrange
import Mathlib.Topology.Algebra.Polynomial
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Nat.Choose.Sum

/-! One-coordinate, higher-order interpolation in the selected mesh interval. -/

open Set MeasureTheory Polynomial
open scoped unitInterval Polynomial

namespace Cubature

noncomputable section

private def integerNode {r : ℕ} (i : Fin r) : ℝ := i.val

private def lagrange {r : ℕ} (i : Fin r) : ℝ → ℝ :=
  fun u => (Lagrange.basis Finset.univ integerNode i).eval u

private theorem lagrange_continuous {r : ℕ} (i : Fin r) : Continuous (lagrange i) :=
  (Lagrange.basis Finset.univ integerNode i).continuous

private theorem lagrange_moment {r : ℕ} (u : ℝ) {e : ℕ} (he : e < r) :
    ∑ i : Fin r, lagrange i u * (integerNode i - u)^e = (0:ℝ)^e := by
  have hi : Function.Injective (integerNode (r := r)) := by
    intro i j h
    apply Fin.ext
    unfold integerNode at h
    exact_mod_cast h
  let P : ℝ[X] := (X - C u)^e
  have hd : P.degree < (Finset.univ : Finset (Fin r)).card := by
    have h := degree_pow_le (X - C u) e
    rw [degree_X_sub_C, nsmul_one] at h
    exact h.trans_lt (by exact_mod_cast (show e < (Finset.univ : Finset (Fin r)).card by simpa))
  have h := congrArg (fun p : ℝ[X] => p.eval u)
    (Lagrange.eq_interpolate (s := Finset.univ) (v := integerNode (r := r)) hi.injOn hd)
  simp only [P, eval_pow, eval_sub, eval_X, eval_C, sub_self, Lagrange.interpolate_apply,
    eval_finsetSum, eval_mul] at h
  simpa only [lagrange, mul_comm] using h.symm

private theorem gap_product_range {r i : ℕ} (hi : i < r) :
    (∏ j ∈ (Finset.range r).erase i, |(i:ℝ)-(j:ℝ)|) =
      (i.factorial : ℝ) * ((r-1-i).factorial:ℝ) := by
  induction r generalizing i with
  | zero => omega
  | succ n ih =>
    rw [Finset.range_add_one]
    by_cases h : i = n
    · subst i
      rw [Finset.erase_insert Finset.notMem_range_self]
      simp only [Nat.succ_sub_succ_eq_sub, Nat.sub_zero, Nat.sub_self, Nat.factorial_zero,
        Nat.cast_one, mul_one]
      calc
        _ = ∏ j ∈ Finset.range n, (((n-1-j)+1 : ℕ):ℝ) := by
          apply Finset.prod_congr rfl
          intro j hj
          have hlt := Finset.mem_range.mp hj
          have hE : (n-1-j)+1 = n-j := by omega
          rw [hE, Nat.cast_sub hlt.le]
          exact abs_of_nonneg (sub_nonneg.mpr (Nat.cast_le.mpr hlt.le))
        _ = ∏ j ∈ Finset.range n, ((j+1:ℕ):ℝ) :=
          Finset.prod_range_reflect (fun j => ((j+1:ℕ):ℝ)) n
        _ = _ := by rw [Nat.factorial_eq_prod_range_add_one, Nat.cast_prod]
    · have hlt : i < n := by omega
      have hno : n ∉ (Finset.range n).erase i :=
        fun hh => Finset.notMem_range_self (Finset.mem_of_mem_erase hh)
      rw [Finset.erase_insert_of_ne (Ne.symm h), Finset.prod_insert hno, ih hlt,
        abs_of_nonpos (sub_nonpos.mpr (Nat.cast_le.mpr hlt.le))]
      have hE : (((n-1-i)+1:ℕ):ℝ) = (n:ℝ)-(i:ℝ) := by
        rw [show (n-1-i)+1 = n-i by omega, Nat.cast_sub hlt.le]
      rw [show n.succ-1-i = (n-1-i)+1 by omega, Nat.factorial_succ, Nat.cast_mul, hE]
      ring

private theorem gap_product {r : ℕ} (i : Fin r) :
    (∏ j ∈ Finset.univ.erase i, |integerNode i - integerNode j|) =
      (i.val.factorial : ℝ) * ((r-1-i.val).factorial:ℝ) := by
  let e : Fin r ↪ ℕ := ⟨Fin.val, Fin.val_injective⟩
  have hE : (Finset.univ.erase i).map e = (Finset.range r).erase i.val := by
    ext j
    constructor
    · intro h
      obtain ⟨k,hk,rfl⟩ := Finset.mem_map.mp h
      refine Finset.mem_erase.mpr ⟨?_, Finset.mem_range.mpr k.isLt⟩
      exact fun he => (Finset.mem_erase.mp hk).1 (Fin.ext he)
    · intro h
      obtain ⟨hne,hj⟩ := Finset.mem_erase.mp h
      exact Finset.mem_map.mpr ⟨⟨j, Finset.mem_range.mp hj⟩,
        Finset.mem_erase.mpr ⟨fun he => hne (congrArg Fin.val he),Finset.mem_univ _⟩,rfl⟩
  calc
    _ = ∏ j ∈ (Finset.univ.erase i).map e, |(i.val:ℝ)-(j:ℝ)| := by
      rw [Finset.prod_map]
      rfl
    _ = _ := by rw [hE,gap_product_range i.isLt]

private theorem lagrange_abs_factorial {r : ℕ} (i : Fin r) {u : ℝ} (hu : u ∈ Icc 0 (r:ℝ)) :
    |lagrange i u| ≤ (r:ℝ)^(r-1) / ((i.val.factorial:ℝ) * ((r-1-i.val).factorial:ℝ)) := by
  let F : ℝ := (i.val.factorial : ℝ) * ((r-1-i.val).factorial:ℝ)
  have hF : 0 ≤ F := by dsimp [F]; positivity
  have hL : |lagrange i u| = F⁻¹ * (∏ j ∈ Finset.univ.erase i, |u - integerNode j|) := by
    simp only [lagrange, Lagrange.basis, eval_prod, Lagrange.basisDivisor, eval_mul, eval_sub,
      eval_C, eval_X, Finset.abs_prod, abs_mul, abs_inv, Finset.prod_mul_distrib,
      Finset.prod_inv_distrib, gap_product, F]
  rw [hL]
  calc
    _ ≤ F⁻¹ * (∏ _ ∈ Finset.univ.erase i, (r:ℝ)) := by
      apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr hF)
      refine Finset.prod_le_prod₀ (fun _ _ => abs_nonneg _) (fun j _ => ?_)
      exact abs_le.mpr ⟨by
        have hj' : (integerNode j : ℝ) < r := Nat.cast_lt.mpr j.isLt
        linarith [hu.1], by
        have h : 0 ≤ integerNode j := Nat.cast_nonneg _
        linarith [hu.2]⟩
    _ = _ := by
      simp only [Finset.prod_const, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ,
        Fintype.card_fin, div_eq_mul_inv, F]
      exact mul_comm _ _

private theorem lagrange_sum_abs {r : ℕ} (hr : 0 < r) {u : ℝ} (hu : u ∈ Icc 0 (r:ℝ)) :
    ∑ i : Fin r, |lagrange i u| ≤ (8:ℝ)^r := by
  let F : ℝ := (r-1).factorial
  let A : ℝ := (r:ℝ)^(r-1)
  have hF : F ≠ 0 := by dsimp [F]; positivity
  have hi (i : Fin r) : |lagrange i u| ≤ A/F * ((r-1).choose i.val : ℝ) := by
    refine (lagrange_abs_factorial i hu).trans_eq ?_
    have hj : i.val ≤ r-1 := by have := i.isLt; omega
    have hE : ((r-1).choose i.val : ℝ) * (i.val.factorial:ℝ) * ((r-1-i.val).factorial:ℝ) = F :=
      by unfold F; exact_mod_cast (Nat.choose_mul_factorial_mul_factorial hj)
    have hh₁ : (i.val.factorial:ℝ) ≠ 0 := by positivity
    have hh₂ : ((r-1-i.val).factorial:ℝ) ≠ 0 := by positivity
    field_simp
    linear_combination -A*hE
  have hsum : ∑ i : Fin r, ((r-1).choose i.val : ℝ) = (2:ℝ)^(r-1) := by
    rw [Fin.sum_univ_eq_sum_range (fun j : ℕ => ((r-1).choose j : ℝ))]
    exact_mod_cast (by simpa only [Nat.sub_add_cancel hr] using Nat.sum_range_choose (r-1) :
      ∑ j ∈ Finset.range r, (r-1).choose j = 2^(r-1))
  have hExp : A/F ≤ (3:ℝ)^r := by
    refine (Real.pow_div_factorial_le_exp (r:ℝ) (Nat.cast_nonneg r) (r-1)).trans ?_
    have h := Real.exp_nat_mul 1 r
    simp only [mul_one] at h
    rw [h]
    exact pow_le_pow_left₀ (Real.exp_pos _).le Real.exp_one_lt_three.le _
  have hTwo : (2:ℝ)^(r-1) ≤ (2:ℝ)^r :=
    pow_le_pow_right₀ (by norm_num) (Nat.sub_le _ _)
  calc
    _ ≤ ∑ i : Fin r, A/F * ((r-1).choose i.val : ℝ) := Finset.sum_le_sum fun i _ => hi i
    _ = A/F * (2:ℝ)^(r-1) := by rw [← Finset.mul_sum,hsum]
    _ ≤ (3:ℝ)^r * (2:ℝ)^r := mul_le_mul hExp hTwo (by positivity) (by positivity)
    _ = (6:ℝ)^r := by rw [← mul_pow]; norm_num
    _ ≤ _ := pow_le_pow_left₀ (by norm_num) (by norm_num) _

def meshCell {m : ℕ} (hm : 0 < m) (x : ℝ) : Fin m :=
  ⟨cell m x, lt_of_le_of_lt (min_le_left _ _) (by omega)⟩

abbrev PolyAxis (m r : ℕ) := Fin m × Fin r

def polyNode {m r : ℕ} (hm : 0 < m) (hr : 0 < r) (v : PolyAxis m r) : I :=
  ⟨((v.1.val : ℝ) + (v.2.val : ℝ)/(r:ℝ))/(m:ℝ), by
    refine ⟨by positivity, (div_le_one (by positivity)).mpr ?_⟩
    have hA : (v.2.val : ℝ)/(r:ℝ) ≤ 1 :=
      (div_le_one (by positivity)).mpr (Nat.cast_le.mpr v.2.isLt.le)
    have hJ : (v.1.val : ℝ) + 1 ≤ (m:ℝ) := by exact_mod_cast Nat.succ_le_of_lt v.1.isLt
    linarith⟩

private def localArg (m r : ℕ) (x : ℝ) : ℝ :=
  (r:ℝ) * ((m:ℝ)*x - (cell m x : ℝ))

private theorem localArg_mem {m r : ℕ} (hm : 0 < m) (x : I) :
    localArg m r x ∈ Icc 0 (r:ℝ) := by
  obtain ⟨hL,hU⟩ := cell_bounds hm x
  exact ⟨mul_nonneg (Nat.cast_nonneg r) (by linarith),
    mul_le_of_le_one_right (Nat.cast_nonneg r) (by linarith)⟩

def polyWeight {m r : ℕ} (hm : 0 < m) (v : PolyAxis m r) (x : ℝ) : ℝ :=
  if v.1 = meshCell hm x then lagrange v.2 (localArg m r x) else 0

theorem polyWeight_measurable {m r : ℕ} (hm : 0 < m) (v : PolyAxis m r) :
    Measurable (polyWeight hm v) := by
  have hc := measurable_cell m
  have hv : Measurable (fun x => (cell m x : ℝ)) :=
    (measurable_from_nat (f := fun n => (n : ℝ))).comp hc
  have hArg : Measurable (localArg m r) :=
    ((measurable_id.const_mul _).sub hv).const_mul _
  have hE : {x : ℝ | v.1 = meshCell hm x} = (cell m) ⁻¹' {v.1.val} := by
    ext x
    simp only [mem_ofPred_eq, mem_preimage, mem_singleton_iff, Fin.ext_iff, meshCell, eq_comm]
  have hs : MeasurableSet {x : ℝ | v.1 = meshCell hm x} := hE ▸ hc (measurableSet_singleton _)
  exact Measurable.ite hs ((lagrange_continuous v.2).measurable.comp hArg) measurable_const

private theorem sum_on_cell {m r : ℕ} (j : Fin m) (f : PolyAxis m r → ℝ)
    (hf : ∀ v, v.1 ≠ j → f v = 0) :
    ∑ v, f v = ∑ i : Fin r, f (j,i) := by
  rw [Fintype.sum_prod_type]
  apply Finset.sum_eq_single_of_mem j (Finset.mem_univ _)
  intro v _ hv
  exact Finset.sum_eq_zero fun i _ => hf (v,i) hv

theorem sum_polyWeight_abs {m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : I) :
    ∑ v : PolyAxis m r, |polyWeight hm v x| ≤ (8:ℝ)^r := by
  rw [sum_on_cell (meshCell hm x) (fun v : PolyAxis m r => |polyWeight hm v x|)
    (fun v hv => by simp [polyWeight,hv])]
  simp only [polyWeight, ↓reduceIte]
  exact lagrange_sum_abs hr (localArg_mem hm x)

theorem polyWeight_support {m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : I) (v : PolyAxis m r)
    (hv : polyWeight hm v x ≠ 0) :
    |(polyNode hm hr v : ℝ) - x.val| ≤ 1/(m:ℝ) := by
  have he : v.1 = meshCell hm x := by by_contra hn; exact hv (by simp [polyWeight,hn])
  obtain ⟨hL,hU⟩ := cell_bounds hm x
  have hr' : 0 ≤ (v.2.val : ℝ)/(r:ℝ) ∧ (v.2.val : ℝ)/(r:ℝ) ≤ 1 :=
    ⟨by positivity, (div_le_one (by positivity)).mpr (Nat.cast_le.mpr v.2.isLt.le)⟩
  have hm' : (0:ℝ) < m := by positivity
  have hJ : (v.1.val : ℝ) = (cell m x : ℝ) := by rw [he]; rfl
  change |((v.1.val : ℝ) + (v.2.val : ℝ)/(r:ℝ))/(m:ℝ)-x.val| ≤ _
  rw [hJ, show ((cell m x : ℝ)+(v.2.val : ℝ)/(r:ℝ))/(m:ℝ)-x.val =
    ((cell m x : ℝ)+(v.2.val : ℝ)/(r:ℝ) - (m:ℝ)*x.val)/(m:ℝ) by field_simp]
  rw [abs_div, abs_of_nonneg hm'.le, div_le_div_iff_of_pos_right hm', abs_le]
  exact ⟨by linarith [hr'.1], by linarith [hr'.2]⟩

theorem sum_polyWeight_moment {m r : ℕ} (hm : 0 < m) (hr : 0 < r) (x : I) {e : ℕ} (he : e < r) :
    ∑ v : PolyAxis m r, polyWeight hm v x * ((polyNode hm hr v : ℝ) - (x : ℝ))^e = (0:ℝ)^e := by
  rw [sum_on_cell (meshCell hm x) (fun v : PolyAxis m r =>
    polyWeight hm v x * ((polyNode hm hr v : ℝ) - (x : ℝ))^e)
    (fun v hv => by simp [polyWeight,hv])]
  have H (i : Fin r) : (polyNode hm hr (meshCell hm x,i) : ℝ) - x.val =
      (integerNode i - localArg m r x) / ((r:ℝ)*(m:ℝ)) := by
    unfold polyNode meshCell localArg integerNode
    dsimp only
    have hm' : (m:ℝ) ≠ 0 := by positivity
    have hr' : (r:ℝ) ≠ 0 := by positivity
    field_simp
    ring
  simp only [polyWeight, ↓reduceIte]
  have hv (i : Fin r) :
      lagrange i (localArg m r x) * ((integerNode i - localArg m r x) / ((r:ℝ)*(m:ℝ)))^e =
        (lagrange i (localArg m r x) * (integerNode i - localArg m r x)^e) / ((r:ℝ)*(m:ℝ))^e := by
    rw [div_pow, mul_div_assoc]
  simp_rw [H, hv]
  rw [← Finset.sum_div, lagrange_moment (localArg m r x) he]
  cases e <;> simp

end

end Cubature
