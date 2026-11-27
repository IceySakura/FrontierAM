import Theory.GaussianScaling
import Mathlib.Algebra.Order.Chebyshev

/-! Quadratic-form reconstruction from one- and two-coordinate probes. -/

open MeasureTheory Set Metric WithLp
open scoped Pointwise

namespace Cubature

noncomputable section

abbrev RealMatrix (d : ℕ) := Matrix (Fin d) (Fin d) ℝ

def quadEval {d : ℕ} (Q : RealMatrix d) (z : GaussianSpace d) : ℝ :=
  ∑ i, ∑ j, Q i j * z i * z j

def ellipsoidSet {d : ℕ} (Q : RealMatrix d) : Set (GaussianSpace d) :=
  {z | quadEval Q z < 1}

def ellipsoidOracle {d : ℕ} (Q : RealMatrix d) (z : GaussianSpace d) : Bool :=
  decide (quadEval Q z < 1)

theorem measurableSet_ellipsoid {d : ℕ} (Q : RealMatrix d) : MeasurableSet (ellipsoidSet Q) := by
  have hm : Measurable (quadEval Q) := by
    unfold quadEval
    exact Finset.measurable_sum _ fun i _ => Finset.measurable_sum _ fun j _ => by fun_prop
  exact measurableSet_lt hm measurable_const

theorem quadEval_smul {d : ℕ} (Q : RealMatrix d) (t : ℝ) (z : GaussianSpace d) :
    quadEval Q (t • z) = t^2 * quadEval Q z := by
  unfold quadEval
  simp only [PiLp.smul_apply, smul_eq_mul]
  simp_rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => by ring))

def quadProbe {d : ℕ} (i j : Fin d) : GaussianSpace d :=
  toLp 2 (fun k => if k ∈ ({i,j} : Finset (Fin d)) then 1 else 0)

theorem quadProbe_apply {d : ℕ} (i j k : Fin d) :
    quadProbe i j k = if k ∈ ({i,j} : Finset (Fin d)) then 1 else 0 := rfl

theorem quadProbe_comm {d : ℕ} (i j : Fin d) : quadProbe i j = quadProbe j i := by
  have h : ({i,j}:Finset (Fin d)) = {j,i} := by ext k; simp [or_comm]
  unfold quadProbe
  rw [h]

theorem quadProbe_abs {d : ℕ} (i j k : Fin d) : |quadProbe i j k| ≤ 1 := by
  rw [quadProbe_apply]
  split <;> norm_num

theorem quadProbe_norm {d : ℕ} (i j : Fin d) : 1 ≤ ‖quadProbe i j‖^2 ∧ ‖quadProbe i j‖^2 ≤ 2 := by
  let S : Finset (Fin d) := {i,j}
  have hS : ‖quadProbe i j‖^2 = (S.card : ℝ) := by
    rw [EuclideanSpace.real_norm_sq_eq]
    have h (k : Fin d) : quadProbe i j k ^ 2 = if k ∈ S then 1 else 0 := by
      rw [quadProbe_apply]
      change (if k ∈ S then (1:ℝ) else 0)^2 = _
      split <;> simp
    simp_rw [h]
    exact (Finset.sum_boole (R := ℝ) (fun k => k ∈ S) Finset.univ).trans (by simp)
  have h1 : 1 ≤ S.card := Finset.card_pos.mpr ⟨i,by simp [S]⟩
  have h2 : S.card ≤ 2 := Finset.card_le_two
  rw [hS]
  exact ⟨by exact_mod_cast h1,by exact_mod_cast h2⟩

private theorem quadProbe_sum {d : ℕ} (Q : RealMatrix d) (i j : Fin d) :
    quadEval Q (quadProbe i j) = ∑ a ∈ ({i,j} : Finset (Fin d)), ∑ b ∈ ({i,j} : Finset (Fin d)), Q a b := by
  let S : Finset (Fin d) := {i,j}
  let v := quadProbe i j
  change (∑ a, ∑ b, Q a b*v a*v b) = ∑ a ∈ S, ∑ b ∈ S, Q a b
  have h₁ (a : Fin d) (ha : a ∈ S) : v a = 1 := by simp only [v,quadProbe_apply,S] at ha ⊢; simp [ha]
  have h₀ (a : Fin d) (ha : a ∉ S) : v a = 0 := by simp only [v,quadProbe_apply,S] at ha ⊢; simp [ha]
  calc
    _ = ∑ a ∈ S, ∑ b, Q a b*v a*v b := by
      symm
      apply Finset.sum_subset (Finset.subset_univ _)
      intro a _ ha
      simp [h₀ a ha]
    _ = ∑ a ∈ S, ∑ b ∈ S, Q a b*v a*v b := by
      apply Finset.sum_congr rfl
      intro a _
      symm
      apply Finset.sum_subset (Finset.subset_univ _)
      intro b _ hb
      simp [h₀ b hb]
    _ = _ := Finset.sum_congr rfl (fun a ha => Finset.sum_congr rfl (fun b hb => by simp [h₁ a ha,h₁ b hb]))

theorem quadProbe_diag {d : ℕ} (Q : RealMatrix d) (i : Fin d) :
    quadEval Q (quadProbe i i) = Q i i := by rw [quadProbe_sum]; simp

theorem quadProbe_off {d : ℕ} (Q : RealMatrix d) (hQ : ∀ i j, Q i j = Q j i)
    {i j : Fin d} (hij : i ≠ j) :
    quadEval Q (quadProbe i j) = Q i i + Q j j + 2*Q i j := by
  rw [quadProbe_sum]
  have hnot : i ∉ ({j}:Finset (Fin d)) := by simpa using hij
  simp only [Finset.sum_insert hnot,Finset.sum_singleton]
  rw [← hQ i j]
  ring

def reconstructForm {d : ℕ} (samples : Fin d → Fin d → ℝ) : RealMatrix d :=
  fun i j => if i = j then samples i i else (samples i j - samples i i - samples j j)/2

def formResolution (L U : ℝ) (k : ℕ) : ℝ := (2*U-L)/(2*(2:ℝ)^k)

def relativeFormResolution (d k : ℕ) (L U : ℝ) : ℝ := 2 * formResolution L U k * (d:ℝ)/L

private theorem reconstruct_entry_error {d : ℕ} {Q : RealMatrix d} (hQ : ∀ i j, Q i j = Q j i)
    {samples : Fin d → Fin d → ℝ} {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ i j, |samples i j - quadEval Q (quadProbe i j)| ≤ δ) (i j : Fin d) :
    |reconstructForm samples i j - Q i j| ≤ 2*δ := by
  by_cases he : i = j
  · subst j
    have h' := h i i
    rw [quadProbe_diag] at h'
    simpa only [reconstructForm,↓reduceIte] using h'.trans (by linarith : δ ≤ 2*δ)
  · have h₁ := h i i
    have h₂ := h j j
    have h₃ := h i j
    rw [quadProbe_diag,abs_le] at h₁ h₂
    rw [quadProbe_off Q hQ he,abs_le] at h₃
    simp only [reconstructForm,ite_eq_right he,abs_le]
    exact ⟨by linarith [h₁.2,h₂.2,h₃.1],by linarith [h₁.1,h₂.1,h₃.2]⟩

theorem reconstruct_quad_error {d : ℕ} {Q : RealMatrix d} (hQ : ∀ i j, Q i j = Q j i)
    {samples : Fin d → Fin d → ℝ} {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ i j, |samples i j - quadEval Q (quadProbe i j)| ≤ δ) (z : GaussianSpace d) :
    |quadEval (reconstructForm samples) z - quadEval Q z| ≤ (2*δ*(d:ℝ))*‖z‖^2 := by
  let H := reconstructForm samples
  have hcs : (∑ i : Fin d, |z i|)^2 ≤ (d:ℝ)*‖z‖^2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    simpa only [Finset.card_univ,Fintype.card_fin,sq_abs] using
      sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := fun i : Fin d => |z i|)
  have he (i j : Fin d) : H i j * z i * z j - Q i j * z i * z j =
      (H i j - Q i j) * z i * z j := by ring
  unfold quadEval
  change |(∑ i, ∑ j, H i j * z i * z j) - (∑ i, ∑ j, Q i j * z i * z j)| ≤ _
  simp_rw [← Finset.sum_sub_distrib, he]
  calc
    _ ≤ ∑ i : Fin d, |∑ j, (H i j-Q i j)*z i*z j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i : Fin d, ∑ j : Fin d, |(H i j-Q i j)*z i*z j| := Finset.sum_le_sum (fun _ _ =>
      Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ i : Fin d, ∑ j : Fin d, (2*δ) * |z i| * |z j| := by
      refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
      simp only [abs_mul]
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right (reconstruct_entry_error hQ hδ h i j) (abs_nonneg _)) (abs_nonneg _)
    _ = (2*δ)*(∑ i : Fin d, |z i|)^2 := by
      simp_rw [← Finset.mul_sum, ← Finset.sum_mul, ← Finset.mul_sum]
      ring
    _ ≤ _ := (mul_le_mul_of_nonneg_left hcs (by positivity)).trans_eq (by ring)

theorem ellipsoid_relative_from_form {d : ℕ} (hd : 0 < d) {Q H : RealMatrix d}
    {η : ℝ} (hη : 0 ≤ η) (hsmall : (d:ℝ)*η ≤ 1/2)
    (hpos : ∀ z, 0 ≤ quadEval Q z)
    (herr : ∀ z, |quadEval H z - quadEval Q z| ≤ η*quadEval Q z) :
    |gaussianSetMass (ellipsoidSet H) - gaussianSetMass (ellipsoidSet Q)| ≤
      (2*(d:ℝ)*η)*gaussianSetMass (ellipsoidSet Q) := by
  let t := 1-η
  have hd₁ : (1:ℝ) ≤ d := by exact_mod_cast Nat.succ_le_of_lt hd
  have hle : η ≤ (d:ℝ)*η := le_mul_of_one_le_left hη hd₁
  have ht : 0 < t := by dsimp [t]; linarith
  have ht₁ : t ≤ 1 := by dsimp [t]; linarith
  have ht₂ : t^2 ≤ t := by
    rw [pow_two]
    exact mul_le_of_le_one_right ht.le ht₁
  refine gaussianSetMass_relative hd (measurableSet_ellipsoid Q) hη hsmall ?_ ?_
  · rintro z ⟨x,hx,rfl⟩
    have hxq : quadEval Q x < 1 := hx
    have hu : quadEval H (t • x) ≤ (1+η)*quadEval Q (t • x) := by
      have h := (abs_le.mp (herr (t • x))).2
      linarith
    have hcoef : (1+η)*t^2 ≤ 1 :=
      (mul_le_mul_of_nonneg_left ht₂ (by linarith)).trans (by dsimp [t]; nlinarith [sq_nonneg η])
    change quadEval H (t • x) < 1
    refine hu.trans_lt ?_
    rw [quadEval_smul, ← mul_assoc]
    exact (mul_le_of_le_one_left (hpos x) hcoef).trans_lt hxq
  · intro x hx
    have hxq : quadEval H x < 1 := hx
    have hl : t*quadEval Q x ≤ quadEval H x := by
      have h := (abs_le.mp (herr x)).1
      dsimp [t]
      linarith
    change x ∈ t⁻¹ • ellipsoidSet Q
    rw [mem_smul_set_iff_inv_smul_mem₀ (inv_ne_zero ht.ne'),inv_inv]
    change quadEval Q (t • x) < 1
    rw [quadEval_smul]
    exact (mul_le_mul_of_nonneg_right ht₂ (hpos x)).trans_lt (hl.trans_lt hxq)

end

end Cubature
