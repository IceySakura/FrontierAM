import Theory.LowerAlgebra
import Theory.PrecisionModel

open scoped BigOperators ENNReal Matrix
open Matrix Set MeasureTheory Unitary

namespace Cubature

variable {d : ℕ}

theorem measurable_centeredBody (A : Coefficients d) : MeasurableSet (centeredBody A) :=
  (isClosed_le (continuous_quadratic A) continuous_const).measurableSet

theorem quadratic_conjugated_bounds (U : Matrix.unitaryGroup (Fin d) ℝ) (v : Coordinates d)
    (hv : ∀ i, (1:ℝ)/2 ≤ v i ∧ v i ≤ 3/2) (x : Coordinates d) :
    (1/2:ℝ)*squaredLength x ≤ quadratic (conjStarAlgAut ℝ _ U (diagonal v)) x ∧
      quadratic (conjStarAlgAut ℝ _ U (diagonal v)) x ≤ (3/2:ℝ)*squaredLength x := by
  let y := star (U : Coefficients d) *ᵥ x
  have eqn := unitary_quad U v x
  rw [← unitary_squaredLength U x]
  change (1/2:ℝ)*squaredLength y ≤ _ ∧ _ ≤ (3/2:ℝ)*squaredLength y
  rw [eqn]
  dsimp [squaredLength]
  simp only [Finset.mul_sum]
  exact ⟨Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_right (hv i).1 (sq_nonneg _)),
      Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_right (hv i).2 (sq_nonneg _))⟩

theorem square_sum_nonneg (x : Coordinates d) : 0 ≤ squaredLength x :=
  Finset.sum_nonneg (fun i _ => sq_nonneg (x i))

theorem centeredBody_inside_box {A : Coefficients d}
    (hA : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x)
    {x : Coordinates d} (hx : x ∈ centeredBody A) : inBox 1 x := by
  change quadratic A x ≤ 1/100 at hx
  intro i
  have h : (x i)^2 ≤ squaredLength x :=
    Finset.single_le_sum (fun j _ => sq_nonneg (x j)) (Finset.mem_univ i)
  rw [← sq_le_sq₀ (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 1), sq_abs]
  linarith [hA x]

theorem centeredBody_finite {A : Coefficients d}
    (hA : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x) : gaussian (centeredBody A) ≠ ⊤ :=
  gaussian_box_finite 0 1 (measurable_centeredBody A)
    (fun x hx => by simpa using centeredBody_inside_box hA hx)

theorem centeredBody_identity_positive : 0 < gaussian (centeredBody (1 : Coefficients d)) := by
  let e : ℝ := (10 * (d+1:ℝ))⁻¹
  have hd : (0:ℝ) ≤ d := Nat.cast_nonneg _
  have he : 0 < e := by dsimp [e]; positivity
  let A : Coefficients d := (100:ℝ) • (1 : Coefficients d)
  have h : ellipsoid (0 : Coordinates d) A 0 = centeredBody (1 : Coefficients d) := by
    ext x
    change quadratic A (x-0) + linear 0 (x-0) ≤ 1 ↔ quadratic (1 : Coefficients d) x ≤ 1/100
    rw [sub_zero]
    have h0 : linear (0 : Coordinates d) x = 0 := by simp [linear]
    rw [h0, add_zero]
    change quadratic ((100:ℝ) • (1 : Coefficients d)) x ≤ 1 ↔ _
    rw [quadratic_smul_matrix]
    constructor <;> intro H <;> linarith
  rw [← h]
  apply gaussian_ellipsoid_positive _ _ _ he
  intro x hx
  have hp (i : Fin d) : (x i)^2 ≤ e^2 := by
    rw [← sq_abs (x i), sq_le_sq₀ (abs_nonneg _) he.le]
    exact hx i
  have H := Finset.sum_le_sum (s := Finset.univ) fun i _ => hp i
  have hE : (d:ℝ)*e^2 ≤ 1/100 := by
    dsimp [e]
    have hp : 0 < (10 * ((d:ℝ)+1))^2 := by positivity
    rw [inv_pow, ← div_eq_mul_inv, div_le_iff₀ hp]
    nlinarith
  have h0 : linear (0 : Coordinates d) x = 0 := by simp [linear]
  rw [h0, add_zero, quadratic_smul_matrix, quadratic_one, squaredLength]
  simp at H
  linarith [H.trans hE]

theorem diagonalized_volume (r2 : ℝ) {U : Matrix.unitaryGroup (Fin d) ℝ} {v : Coordinates d}
    (hv : ∀ i, (1:ℝ)/2 ≤ v i ∧ v i ≤ 3/2) :
    ∃ k : ℝ, 0 < k ∧ k^2 = ∏ i, v i ∧
      (volume : Measure (Coordinates d)) (bodyAt r2 (1 : Coefficients d)) =
        ENNReal.ofReal k * (volume : Measure (Coordinates d))
          (bodyAt r2 (conjStarAlgAut ℝ _ U (diagonal v))) := by
  let r : Coordinates d := fun i => Real.sqrt (v i)
  have hr (i) : 0 < r i := Real.sqrt_pos.mpr (by linarith [(hv i).1])
  have hs (i) : (r i)^2 = v i := Real.sq_sqrt (by linarith [(hv i).1])
  let P := diagonal r * star (U : Coefficients d)
  let B := (conjStarAlgAut ℝ _ U (diagonal v))
  have hq (x : Coordinates d) : squaredLength (P *ᵥ x) = quadratic B x := by
    rw [unitary_quad]
    simp only [P, ← Matrix.mulVec_mulVec, mulVec_diagonal, squaredLength, mul_pow]
    simp_rw [hs]
  have hU : (U : Coefficients d) * star (U : Coefficients d) = 1 := Unitary.coe_mul_star_self U
  have hUt : star (U : Coefficients d) * (U : Coefficients d) = 1 := Unitary.coe_star_mul_self U
  have hstar : star (U : Coefficients d) = (U : Coefficients d)ᵀ := by ext i j; rfl
  have hdet : (det (star (U : Coefficients d)))^2 = 1 := by
    have h := congrArg Matrix.det hUt
    simpa [Matrix.det_mul,hstar,Matrix.det_transpose,pow_two] using h
  have hp_det : (det P)^2 = ∏ i, v i := by
    change (det (diagonal r * star (U : Coefficients d)))^2 = _
    rw [Matrix.det_mul, mul_pow, hdet, mul_one, Matrix.det_diagonal]
    rw [← Finset.prod_pow]
    simp_rw [hs]
  have hdP : 0 < |det P| := by
    apply abs_pos.mpr
    intro h
    have hb : 0 < ∏ i, v i := Finset.prod_pos (fun i _ => by linarith [(hv i).1])
    rw [h] at hp_det
    linarith
  have hsurj : Function.Surjective (fun x : Coordinates d => P *ᵥ x) := by
    intro y
    let z := (U : Coefficients d) *ᵥ (diagonal (fun i => (r i)⁻¹) *ᵥ y)
    refine ⟨z,?_⟩
    dsimp [P,z]
    rw [← Matrix.mulVec_mulVec, Matrix.mulVec_mulVec _ (star (U : Coefficients d))
      (U : Coefficients d), hUt, one_mulVec]
    ext i
    simp [mulVec_diagonal, mul_inv_cancel₀ (hr i).ne', ← mul_assoc]
  have himg : (fun x : Coordinates d => P *ᵥ x) '' bodyAt r2 B =
      bodyAt r2 (1 : Coefficients d) := by
    ext y
    constructor
    · rintro ⟨x,hx,rfl⟩
      simpa only [bodyAt, Set.mem_ofPred_eq, quadratic_one,hq] using hx
    · intro hy
      obtain ⟨x,rfl⟩ := hsurj y
      refine ⟨x,?_,rfl⟩
      simpa only [bodyAt, Set.mem_ofPred_eq, quadratic_one,hq] using hy
  refine ⟨|det P|,hdP,by rw [sq_abs,hp_det],?_⟩
  have h := Measure.addHaar_image_linearMap (volume : Measure (Coordinates d)) P.mulVecLin (bodyAt r2 B)
  have hd : LinearMap.det P.mulVecLin = det P := LinearMap.det_toLin' P
  change volume ((fun x => P *ᵥ x) '' bodyAt r2 B) =
    ENNReal.ofReal |LinearMap.det P.mulVecLin| * volume (bodyAt r2 B) at h
  rwa [himg,hd] at h

theorem diagonalized_mass_gap {U : Matrix.unitaryGroup (Fin d) ℝ} {v : Coordinates d}
    (hv : ∀ i, (1:ℝ)/2 ≤ v i ∧ v i ≤ 3/2) (hvdet : (∏ i, v i) ≤ 7/8) :
    (21/20:ℝ)*(gaussian (centeredBody (1 : Coefficients d))).toReal ≤
      (gaussian (centeredBody (conjStarAlgAut ℝ _ U (diagonal v)))).toReal := by
  let A := conjStarAlgAut ℝ _ U (diagonal v)
  let F := centeredBody (1 : Coefficients d)
  let E := centeredBody A
  let c := (Real.sqrt (2*Real.pi))⁻¹ ^ d
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have ha x := (quadratic_conjugated_bounds U v hv x).1
  have hf x : (1/2:ℝ)*squaredLength x ≤ quadratic (1 : Coefficients d) x := by
    rw [quadratic_one]
    linarith [square_sum_nonneg x]
  have hem (x : Coordinates d) (hx : x ∈ E) : squaredLength x ≤ 2/100 := by
    change quadratic A x ≤ 1/100 at hx
    linarith [ha x]
  have h₀ : gaussian F ≤ ENNReal.ofReal c * volume F := by
    rw [gaussian, withDensity_apply _ (measurable_centeredBody _), ← Measure.restrict_apply_univ,
      ← lintegral_const]
    apply lintegral_mono
    intro x
    have h : Real.exp (-squaredLength x/2) ≤ 1 := Real.exp_le_one_iff.mpr (by
      linarith [square_sum_nonneg x])
    exact ENNReal.ofReal_le_ofReal ((mul_le_mul_of_nonneg_left h hc).trans_eq (mul_one _))
  have h₁ : ENNReal.ofReal (c * (99/100)) * volume E ≤ gaussian E := by
    rw [gaussian, withDensity_apply _ (measurable_centeredBody _), ← Measure.restrict_apply_univ,
      ← lintegral_const]
    apply lintegral_mono_ae
    filter_upwards [ae_restrict_mem (measurable_centeredBody A)] with x hx
    apply ENNReal.ofReal_le_ofReal
    apply mul_le_mul_of_nonneg_left _ hc
    calc
      (99/100:ℝ) ≤ Real.exp (-1/100) := by linarith [Real.add_one_le_exp (-1/100)]
      _ ≤ Real.exp (-squaredLength x/2) := Real.exp_le_exp.mpr (by linarith [hem x hx])
  obtain ⟨k,hk,hk₂,hvol⟩ := diagonalized_volume (1/100) (U:=U) hv
  change volume F = ENNReal.ofReal k * volume E at hvol
  have hk' : k ≤ 15/16 := by
    rw [← sq_le_sq₀ hk.le (by norm_num : (0:ℝ)≤15/16),hk₂]
    norm_num
    linarith
  have rel : ENNReal.ofReal (21/20) * gaussian F ≤ gaussian E := by
    calc
      ENNReal.ofReal (21/20) * gaussian F ≤
          ENNReal.ofReal (21/20) * (ENNReal.ofReal c * volume F) := by gcongr
      _ = ENNReal.ofReal ((21/20)*c*k) * volume E := by
        rw [hvol, ← mul_assoc, ← mul_assoc, ← ENNReal.ofReal_mul (by norm_num),
          ← ENNReal.ofReal_mul (by positivity)]
      _ ≤ ENNReal.ofReal (c*(99/100)) * volume E := by
        apply mul_le_mul_of_nonneg_right _ bot_le
        apply ENNReal.ofReal_le_ofReal
        have hh := mul_le_mul_of_nonneg_left hk' (mul_nonneg (by norm_num : (0:ℝ)≤21/20) hc)
        nlinarith
      _ ≤ gaussian E := h₁
  have h := ENNReal.toReal_mono (centeredBody_finite ha) rel
  simpa only [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by norm_num : (0:ℝ) ≤ 21/20)] using h

end Cubature
