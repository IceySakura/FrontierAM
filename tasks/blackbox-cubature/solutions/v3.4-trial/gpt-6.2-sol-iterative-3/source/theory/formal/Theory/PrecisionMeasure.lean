import Theory.PrecisionAlgebra
import Theory.LowerMeasure

open scoped BigOperators Matrix ENNReal
open Matrix Unitary Set MeasureTheory

namespace Cubature

variable {d : ℕ}

theorem measurable_bodyAt (r2 : ℝ) (A : Coefficients d) : MeasurableSet (bodyAt r2 A) :=
  (isClosed_le (continuous_quadratic A) continuous_const).measurableSet

theorem bodyAt_inBox {r2 : ℝ} {A : Coefficients d}
    (hr : r2 ≤ 1/2) (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x)
    {x : Coordinates d} (hx : x ∈ bodyAt r2 A) : inBox 1 x := by
  change quadratic A x ≤ r2 at hx
  intro i
  have h : (x i)^2 ≤ squaredLength x :=
    Finset.single_le_sum (fun j _ => sq_nonneg (x j)) (Finset.mem_univ i)
  rw [← sq_le_sq₀ (abs_nonneg _) (by norm_num : (0:ℝ)≤1),sq_abs]
  linarith [ha x]

theorem bodyAt_finite {r2 : ℝ} {A : Coefficients d}
    (hr : r2 ≤ 1/2) (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x) :
    volume (bodyAt r2 A) ≠ ⊤ ∧ gaussian (bodyAt r2 A) ≠ ⊤ := by
  have hsub : bodyAt r2 A ⊆ Icc (fun _ => (-1:ℝ)) (fun _ => (1:ℝ)) := by
    intro x hx
    have hh := bodyAt_inBox hr ha hx
    exact ⟨fun i => (abs_le.mp (hh i)).1,fun i => (abs_le.mp (hh i)).2⟩
  have hz : volume (Icc (fun _ : Fin d => (-1:ℝ)) (fun _ => (1:ℝ))) ≠ ⊤ := by
    rw [Real.volume_Icc_pi]
    exact ENNReal.prod_ne_top (fun _ _ => ENNReal.ofReal_ne_top)
  exact ⟨ne_top_of_le_ne_top hz (measure_mono hsub),
    gaussian_box_finite 0 1 (measurable_bodyAt _ _)
      (fun x hx => by simpa using bodyAt_inBox hr ha hx)⟩

theorem bodyAt_positive {r2 : ℝ} {A : Coefficients d} (hr : 0 < r2) (hr' : r2 ≤ 1)
    (ha : ∀ x, quadratic A x ≤ (3/2:ℝ)*squaredLength x) :
    0 < gaussian (bodyAt r2 A) := by
  let ρ := r2/(4*((d:ℝ)+1))
  have hρ : 0 < ρ := by dsimp [ρ]; positivity
  have hc : (d:ℝ)*ρ^2 ≤ (2/3)*r2 := by
    dsimp [ρ]
    have h1 : (0:ℝ) ≤ d := Nat.cast_nonneg _
    have hm : 0 < 4*((d:ℝ)+1) := by positivity
    rw [div_pow, ← mul_div_assoc, div_le_iff₀ (pow_pos hm 2)]
    have H : (d:ℝ)*r2 ≤ (2/3)*(4*((d:ℝ)+1))^2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right H hr.le]
  let B : Coefficients d := r2⁻¹ • A
  have hEq : bodyAt r2 A = ellipsoid 0 B 0 := by
    ext x
    change quadratic A x ≤ r2 ↔ quadratic B (x-0)+linear 0 (x-0)≤1
    have h0 : linear (0 : Coordinates d) x = 0 := by simp [linear]
    rw [sub_zero,h0,add_zero]
    change _ ↔ quadratic (r2⁻¹ • A) x ≤ 1
    rw [quadratic_smul_matrix, inv_mul_eq_div, div_le_iff₀ hr,one_mul]
  rw [hEq]
  apply gaussian_ellipsoid_positive _ _ _ hρ
  intro x hx
  have H : squaredLength x ≤ (d:ℝ)*ρ^2 := by
    have hh : ∀ i, (x i)^2 ≤ ρ^2 := by
      intro i
      rw [← sq_abs, sq_le_sq₀ (abs_nonneg _) hρ.le]
      exact hx i
    simpa [squaredLength] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hh i)
  have K := (ha x).trans (mul_le_mul_of_nonneg_left (H.trans hc) (by norm_num))
  have h0 : linear (0 : Coordinates d) x = 0 := by simp [linear]
  rw [h0,add_zero]
  change quadratic (r2⁻¹ • A) x ≤ 1
  rw [quadratic_smul_matrix, inv_mul_eq_div, div_le_iff₀ hr,one_mul]
  linarith

theorem bodyAt_volume_factor (r2 : ℝ) {A : Coefficients d} (hA : A.IsHermitian)
    (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x) :
    ∃ k : ℝ, 0 < k ∧ k^2 = det A ∧
      (volume : Measure (Coordinates d)) (bodyAt r2 (1 : Coefficients d)) =
        ENNReal.ofReal k * (volume : Measure (Coordinates d)) (bodyAt r2 A) := by
  have hs := quadratic_spectrum_range hA ha
  obtain ⟨k,hk,hk2,hvol⟩ := diagonalized_volume r2 (U:=hA.eigenvectorUnitary) hs
  have eqA : A = conjStarAlgAut ℝ _ hA.eigenvectorUnitary (diagonal hA.eigenvalues) := hA.spectral_theorem
  refine ⟨k,hk,?_,?_⟩
  · conv_rhs => rw [eqA,conjugate_diagonal_det]
    exact hk2
  · conv_rhs => rw [eqA]
    exact hvol

theorem gaussian_vol_comparison {r2 : ℝ} (A : Coefficients d)
    (hA : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x) :
    let c := (Real.sqrt (2*Real.pi))⁻¹ ^ d
    let E := bodyAt r2 A
    ENNReal.ofReal (c*(1-r2))*volume E ≤ gaussian E ∧
      gaussian E ≤ ENNReal.ofReal c*volume E := by
  let E := bodyAt r2 A
  let c := (Real.sqrt (2*Real.pi))⁻¹ ^ d
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have me := measurable_bodyAt r2 A
  change ENNReal.ofReal (c*(1-r2))*volume E ≤ gaussian E ∧
      gaussian E ≤ ENNReal.ofReal c*volume E
  constructor
  · rw [gaussian,withDensity_apply _ me,← Measure.restrict_apply_univ,← lintegral_const]
    apply lintegral_mono_ae
    filter_upwards [ae_restrict_mem me] with x hx
    change quadratic A x ≤ r2 at hx
    apply ENNReal.ofReal_le_ofReal
    apply mul_le_mul_of_nonneg_left _ hc
    exact (by linarith [Real.add_one_le_exp (-r2)] : 1-r2 ≤ Real.exp (-r2)).trans
      (Real.exp_le_exp.mpr (by linarith [hA x] : -r2 ≤ -squaredLength x/2))
  · rw [gaussian,withDensity_apply _ me,← Measure.restrict_apply_univ,← lintegral_const]
    apply lintegral_mono
    intro x
    have h : Real.exp (-squaredLength x/2) ≤ 1 := Real.exp_le_one_iff.mpr (by
      linarith [square_sum_nonneg x])
    exact ENNReal.ofReal_le_ofReal ((mul_le_mul_of_nonneg_left h hc).trans_eq (mul_one _))

theorem gaussian_gap_from_determinant {θ r2 : ℝ} (hθ : 0 ≤ θ) (hθ' : θ ≤ 1)
    (hr : 0 < r2) (hr' : r2 ≤ θ^2/100)
    {A C : Coefficients d} (hA : A.IsHermitian) (hC : C.IsHermitian)
    (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x)
    (hc : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic C x ∧ quadratic C x ≤ (3/2:ℝ)*squaredLength x)
    (hb : det A ≤ (1-θ^2/2)*det C) :
    (gaussian (bodyAt r2 C)).toReal ≤ (1-θ^2/5)*(gaussian (bodyAt r2 A)).toReal := by
  let E := bodyAt r2 A
  let F := bodyAt r2 C
  let K := (Real.sqrt (2*Real.pi))⁻¹ ^ d
  have hK : 0 ≤ K := by dsimp [K]; positivity
  have s : θ^2 ≤ 1 := (sq_le_sq₀ hθ zero_le_one).mpr hθ' |>.trans_eq (by norm_num)
  have hs : r2 ≤ 1/2 := by linarith
  obtain ⟨hEA,gEA⟩ := bodyAt_finite hs (fun x => (ha x).1)
  obtain ⟨hFC,gFC⟩ := bodyAt_finite hs (fun x => (hc x).1)
  obtain ⟨p,hp,hp2,hvolA⟩ := bodyAt_volume_factor r2 hA ha
  obtain ⟨q,hq,hq2,hvolC⟩ := bodyAt_volume_factor r2 hC hc
  have hb' : p^2 ≤ (1-θ^2/2)*q^2 := by rwa [hp2,hq2]
  have hstep : p ≤ (1-θ^2/4)*q := by
    have ha0 : 0 ≤ (1-θ^2/4)*q := mul_nonneg (by linarith) hq.le
    rw [← sq_le_sq₀ hp.le ha0]
    have eqn : ((1-θ^2/4)*q)^2 - (1-θ^2/2)*q^2 = (θ^2*q/4)^2 := by ring
    linarith [sq_nonneg (θ^2*q/4)]
  have hv : q*(volume F).toReal = p*(volume E).toReal := by
    have H := congrArg ENNReal.toReal (hvolC.symm.trans hvolA)
    simpa only [ENNReal.toReal_mul, ENNReal.toReal_ofReal hq.le, ENNReal.toReal_ofReal hp.le] using H
  have hvan : (volume F).toReal ≤ (1-θ^2/4)*(volume E).toReal := by
    apply le_of_mul_le_mul_left _ hq
    rw [hv]
    simpa [mul_comm, mul_assoc,mul_left_comm] using
      mul_le_mul_of_nonneg_right hstep (ENNReal.toReal_nonneg (a:=volume E))
  have u := (gaussian_vol_comparison (r2:=r2) C (fun x => (hc x).1)).2
  have l := (gaussian_vol_comparison (r2:=r2) A (fun x => (ha x).1)).1
  have u' := ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top hFC) u
  have l' := ENNReal.toReal_mono gEA l
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal hK] at u'
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (mul_nonneg hK (by linarith))] at l'
  have hκ : 0 ≤ 1-θ^2/5 := by linarith
  have hH : 1-θ^2/4 ≤ (1-θ^2/5)*(1-r2) := by
    have hh := mul_le_mul_of_nonneg_left (sub_le_sub_left hr' 1) hκ
    nlinarith [sq_nonneg (θ^2),sq_nonneg θ]
  change (gaussian F).toReal ≤ (1-θ^2/5)*(gaussian E).toReal
  calc
    (gaussian F).toReal ≤ K*(volume F).toReal := u'
    _ ≤ K*((1-θ^2/4)*(volume E).toReal) := mul_le_mul_of_nonneg_left hvan hK
    _ ≤ (1-θ^2/5)*((K*(1-r2))*(volume E).toReal) := by
      have hh := mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_right hH (ENNReal.toReal_nonneg (a:=volume E))) hK
      simpa [mul_comm,mul_assoc,mul_left_comm] using hh
    _ ≤ (1-θ^2/5)*(gaussian E).toReal := mul_le_mul_of_nonneg_left l' hκ

end Cubature
