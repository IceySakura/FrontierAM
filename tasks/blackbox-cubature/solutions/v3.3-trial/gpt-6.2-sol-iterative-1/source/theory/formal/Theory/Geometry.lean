import Theory.Gaussian
import Theory.Tree
import Mathlib.Data.Sym.Card

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

theorem normSq_nonneg {d : ℕ} (x : Point d) : 0 ≤ normSq x :=
  Finset.sum_nonneg fun i _ => sq_nonneg (x i)

theorem sq_le_normSq {d : ℕ} (x : Point d) (i : Fin d) : x i ^ 2 ≤ normSq x :=
  Finset.single_le_sum (fun j _ ↦ sq_nonneg (x j)) (Finset.mem_univ i)

theorem normSq_smul {d : ℕ} (a : ℝ) (x : Point d) : normSq (a • x) = a^2 * normSq x := by
  simp only [normSq, Pi.smul_apply, smul_eq_mul, mul_pow, Finset.mul_sum]

theorem quadratic_smul {d : ℕ} (A : Coefficients d) (a : ℝ) (x : Point d) :
    quadratic A (a • x) = a^2 * quadratic A x := by
  simp only [quadratic, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem linear_smul {d : ℕ} (b : Point d) (a : ℝ) (x : Point d) :
    linear b (a • x) = a * linear b x := by
  simp only [linear, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem polynomial_smul {d : ℕ} (A : Coefficients d) (b : Point d) (a : ℝ) (x : Point d) :
    polynomial A b (a • x) = a^2 * quadratic A x + a * linear b x := by
  rw [polynomial, quadratic_smul, linear_smul]

theorem polynomial_contract {d : ℕ} (E : Ellipsoid d) (x : Point d) {t : ℝ}
    (ht : 0 ≤ t) (h1 : t ≤ 1) :
    polynomial E.A E.b (t • x) ≤ t * polynomial E.A E.b x := by
  rw [polynomial_smul, polynomial]
  have hl := (normSq_nonneg x).trans (E.low x)
  have ht2 : t^2 ≤ t := by nlinarith [mul_nonneg ht (sub_nonneg.mpr h1)]
  linarith [mul_le_mul_of_nonneg_right ht2 hl]

theorem inside_normSq {d : ℕ} (E : Ellipsoid d) {x : Point d}
    (hx : polynomial E.A E.b x < 1) : normSq x ≤ 3 := by
  have hl := E.low x
  have hb := (abs_le.mp (E.bias x)).1
  unfold polynomial at hx
  linarith

theorem linear_axis {d : ℕ} (b : Point d) (i : Fin d) : linear b (axis i) = b i := by
  simp [linear, axis, mul_ite]

theorem quadratic_axis {d : ℕ} (A : Coefficients d) (i : Fin d) :
    quadratic A (axis i) = A i i := by
  simp [quadratic, axis, mul_ite, ite_mul]

theorem normSq_axis {d : ℕ} (i : Fin d) : normSq (axis i) = 1 := by
  simp [normSq, axis, apply_ite (fun x : ℝ ↦ x^2)]

theorem normSq_two_axes {d : ℕ} {i j : Fin d} (h : i ≠ j) :
    normSq (axis i + axis j) = 2 := by
  have he (k : Fin d) :
      (axis i k + axis j k)^2 = axis i k + axis j k := by
    by_cases hi : k = i <;> by_cases hj : k = j
    · exact False.elim (h (hi.symm.trans hj))
    · simp [axis, hi, h]
    · simp [axis, hj, Ne.symm h]
    · simp [axis, hi, hj]
  simp only [normSq, Pi.add_apply, he, Finset.sum_add_distrib]
  simp [axis]
  norm_num

theorem quadratic_two_axes {d : ℕ} (E : Ellipsoid d) (i j : Fin d) :
    quadratic E.A (axis i + axis j) = E.A i i + E.A j j + 2*E.A i j := by
  simp only [quadratic, Pi.add_apply, mul_add, add_mul, Finset.sum_add_distrib]
  simp only [axis, mul_ite, ite_mul, mul_one, mul_zero, zero_mul, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true]
  rw [E.symm j i]
  ring

theorem axis_bound {d : ℕ} (i j : Fin d) : |axis i j| ≤ 1 := by
  by_cases h : j = i <;> simp [axis, h]

theorem ray_bound {d : ℕ} (r : RayIndex d) (i : Fin d) : |ray r i| ≤ 1 := by
  rcases r with ⟨⟨j,k⟩, hjk⟩
  by_cases h : j = k
  · simpa [ray, h] using axis_bound j i
  · change |(if j = k then axis j else axis j + axis k) i| ≤ 1
    rw [if_neg h, Pi.add_apply]
    by_cases hi : i = j <;> by_cases hk : i = k
    · exact False.elim (h (hi.symm.trans hk))
    · simp [axis,hi,h]
    · simp [axis,hk,Ne.symm h]
    · simp [axis,hi,hk]

theorem ray_norm {d : ℕ} (r : RayIndex d) : 1 ≤ normSq (ray r) ∧ normSq (ray r) ≤ 2 := by
  rcases r with ⟨⟨i,j⟩, hij⟩
  by_cases h : i = j
  · change 1 ≤ normSq (if i = j then axis i else axis i + axis j) ∧
        normSq (if i = j then axis i else axis i + axis j) ≤ 2
    rw [if_pos h, normSq_axis]
    norm_num
  · change 1 ≤ normSq (if i = j then axis i else axis i + axis j) ∧
        normSq (if i = j then axis i else axis i + axis j) ≤ 2
    rw [if_neg h, normSq_two_axes h]
    norm_num

theorem diagonal_ray {d : ℕ} (i : Fin d) : ray (rayIndex i i) = axis i := by
  simp [ray, rayIndex]

theorem mixed_ray {d : ℕ} {i j : Fin d} (h : i ≠ j) :
    ray (rayIndex i j) = axis i + axis j := by
  rcases le_total i j with hij | hji
  · simp [ray, rayIndex, min_eq_left hij, max_eq_right hij, h]
  · simp [ray, rayIndex, min_eq_right hji, max_eq_left hji, Ne.symm h, add_comm]

theorem card_rayIndex (d : ℕ) : 2 * Fintype.card (RayIndex d) = d*(d+1) := by
  let R : Sym2 (Fin d) → RayIndex d :=
    Sym2.lift ⟨rayIndex, fun i j ↦ by simp [rayIndex, min_comm, max_comm]⟩
  have hR (i j : Fin d) : R (Sym2.mk i j) = rayIndex i j := rfl
  have hsur : Function.Surjective R := by
    rintro ⟨⟨i,j⟩, hij⟩
    exact ⟨Sym2.mk i j, by simp [hR, rayIndex, min_eq_left hij, max_eq_right hij]⟩
  have hinj : Function.Injective R := by
    intro a b
    induction a using Sym2.ind with
    | h i j =>
      induction b using Sym2.ind with
      | h k l =>
        simp only [hR]
        intro hh
        have hp : (min i j, max i j) = (min k l, max k l) := congrArg Subtype.val hh
        have hswap (x y : Fin d) : Sym2.mk (min x y) (max x y) = Sym2.mk x y := by
          rcases le_total x y with h | h
          · simp [min_eq_left h,max_eq_right h]
          · simp [min_eq_right h,max_eq_left h, Sym2.eq_swap]
        have hcong := congrArg (fun z : Fin d × Fin d => Sym2.mk z.1 z.2) hp
        exact (hswap i j).symm.trans (hcong.trans (hswap k l))
  have HC := Fintype.card_congr (Equiv.ofBijective R ⟨hinj,hsur⟩)
  rw [← HC, Sym2.card, Fintype.card_fin, Nat.choose_two_right]
  have hv : (d+1)*(d+1-1) = d*(d+1) := by rw [Nat.add_sub_cancel, Nat.mul_comm]
  rw [hv]
  have heven : 2 ∣ d*(d+1) := by
    rcases Nat.mod_two_eq_zero_or_one d with hd | hd
    · have hdiv : 2 ∣ d := Nat.dvd_of_mod_eq_zero hd
      exact dvd_mul_of_dvd_left hdiv _
    · have hdiv : 2 ∣ d+1 := Nat.dvd_of_mod_eq_zero (by omega)
      exact dvd_mul_of_dvd_right hdiv _
  exact Nat.mul_div_cancel' heven

theorem inverse_radius_exists {d : ℕ} (E : Ellipsoid d) (u : Point d)
    (hu : 1 ≤ normSq u ∧ normSq u ≤ 2) :
    ∃ s ∈ Icc (1/2 : ℝ) 4, s^2 = quadratic E.A u + linear E.b u * s := by
  let α := quadratic E.A u
  let β := linear E.b u
  let g := fun s : ℝ => s^2 - (α+β*s)
  have hα := E.low u
  have hα' := E.high u
  have hb := abs_le.mp (E.bias u)
  change normSq u ≤ α at hα
  change α ≤ 2 * normSq u at hα'
  change -( (1+normSq u)/2 ) ≤ β ∧ β ≤ (1+normSq u)/2 at hb
  have hg1 : g (1/2) ≤ 0 := by dsimp [g]; linarith [hu.1,hb.1]
  have hg2 : 0 ≤ g 4 := by dsimp [g]; linarith [hu.2,hb.2]
  obtain ⟨s, hs, he⟩ := intermediate_value_Icc (by norm_num : (1/2:ℝ) ≤ 4)
    (show ContinuousOn g _ from (by fun_prop : Continuous g).continuousOn) ⟨hg1,hg2⟩
  exact ⟨s,hs, by change s^2 - (α+β*s) = 0 at he; linarith⟩

theorem answerBit_iff {d : ℕ} (S : Set (Point d)) (x : Point d) :
    answerBit S x = true ↔ x ∈ S := by
  by_cases h : x ∈ S <;> simp [answerBit,h]

theorem inverse_radius_answer {d : ℕ} {ρ : ℝ} (hρ : ρ ≠ 0) (E : Ellipsoid d)
    (u : Point d) (hu : 0 < quadratic E.A u) {s : ℝ} (hs : 0 < s)
    (he : s^2 = quadratic E.A u + linear E.b u * s) (t : ℝ) (ht : 0 < t) :
    answerBit (region ρ E) ((ρ/t) • u) = true ↔ s < t := by
  have hr : ρ⁻¹ * (ρ/t) = t⁻¹ := by field_simp
  have hlin : linear E.b u < s := by
    by_contra! h
    have hb := mul_le_mul_of_nonneg_right h hs.le
    nlinarith
  simp only [answerBit_iff, region, mem_ofPred_eq]
  rw [smul_smul, hr, polynomial_smul]
  have h0 : 0 < t ^ 2 := pow_pos ht _
  have heq :
      t^2 * ((t⁻¹)^2 * quadratic E.A u + t⁻¹*linear E.b u) =
      quadratic E.A u + linear E.b u*t := by field_simp
  rw [← mul_lt_mul_iff_right₀ h0]
  have hP :
      t^2 - (quadratic E.A u + linear E.b u*t) =
      (t-s)*(t+s-linear E.b u) := by nlinarith
  rw [heq, mul_one, ← sub_pos, hP,
    mul_pos_iff_of_pos_right (by linarith : 0 < t+s-linear E.b u), sub_pos]

theorem opposite_inverse_radii {a b r s : ℝ} (hr : 0 < r) (hs : 0 < s)
    (h₁ : r^2 = a+b*r) (h₂ : s^2 = a-b*s) : a = r*s ∧ b = r-s := by
  have hz : (r+s)*((r-s)-b) = 0 := by nlinarith
  have hb : (r-s)-b = 0 := (mul_eq_zero.mp hz).resolve_left (by linarith)
  exact ⟨by nlinarith [show b = r-s by linarith], by linarith⟩

end Cubature
