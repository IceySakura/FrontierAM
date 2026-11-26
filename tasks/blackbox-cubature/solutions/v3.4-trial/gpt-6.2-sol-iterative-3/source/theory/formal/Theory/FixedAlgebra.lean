import Theory.PrecisionTree

open scoped BigOperators Matrix ENNReal
open Matrix Unitary Set MeasureTheory WithLp

namespace Cubature

variable {d : ℕ}

def InQuarterBand (A : Coefficients d) : Prop :=
  ∀ x, (3/4:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (5/4:ℝ)*squaredLength x

theorem quarter_band_standard {A : Coefficients d} (h : InQuarterBand A) (x : Coordinates d) :
    (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x := by
  constructor <;> linarith [(h x).1,(h x).2,square_sum_nonneg x]

theorem midpoint_quarter {A B : Coefficients d} (ha : InQuarterBand A) (hb : InQuarterBand B) :
    InQuarterBand (midpointMatrix A B) := by
  intro x
  rw [midpoint_quadratic]
  constructor <;> linarith [(ha x).1,(ha x).2,(hb x).1,(hb x).2]

theorem squaredLength_mul_self (P : Coefficients d) (x : Coordinates d) :
    squaredLength (P *ᵥ x) = quadratic (Pᵀ*P) x := by
  simpa [Matrix.mul_one,quadratic_one] using (quadratic_congruence (1 : Coefficients d) P x).symm

theorem hermitian_action_norm {H : Coefficients d} (hH : H.IsHermitian) {L : ℝ} (hL : 0 ≤ L)
    (h : ∀ i, |hH.eigenvalues i| ≤ L) (x : Coordinates d) :
    squaredLength (H *ᵥ x) ≤ L^2*squaredLength x := by
  let U := hH.eigenvectorUnitary
  let v := hH.eigenvalues
  have hd : H=conjStarAlgAut ℝ _ U (diagonal v) := hH.spectral_theorem
  have hu : star (U : Coefficients d) * (U : Coefficients d) = 1 :=
    Unitary.coe_star_mul_self U
  let y := star (U : Coefficients d) *ᵥ x
  have hy : squaredLength y = squaredLength x := unitary_squaredLength U x
  have hx : star (U : Coefficients d) *ᵥ (H *ᵥ x) = diagonal v *ᵥ y := by
    have aux : star (U : Coefficients d) * H = diagonal v * star (U : Coefficients d) := by
      rw [hd,conjStarAlgAut_apply]
      change star (U : Coefficients d) * ((U : Coefficients d) * diagonal v *
        star (U : Coefficients d)) = _
      simp only [← Matrix.mul_assoc,hu,one_mul]
    rw [Matrix.mulVec_mulVec,aux,← Matrix.mulVec_mulVec]
  rw [← unitary_squaredLength U (H *ᵥ x),hx,← hy]
  unfold squaredLength
  simp only [mulVec_diagonal,mul_pow,Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  have he : (v i)^2 ≤ L^2 := by
    rw [← sq_abs (v i), sq_le_sq₀ (abs_nonneg _) hL]
    exact h i
  exact mul_le_mul_of_nonneg_right he (sq_nonneg _)

theorem squaredLength_smul (r : ℝ) (x : Coordinates d) : squaredLength (r • x) =
    r^2*squaredLength x := by
  simp [squaredLength,Pi.smul_apply,smul_eq_mul,mul_pow,Finset.mul_sum]

theorem squaredLength_parallelogram (x y : Coordinates d) :
    squaredLength (x+y) + squaredLength (x-y) = 2*(squaredLength x+squaredLength y) := by
  simp only [squaredLength,Pi.add_apply,Pi.sub_apply,← Finset.sum_add_distrib,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem squaredLength_add_bounded (x y : Coordinates d) {a : ℝ} (ha : 0 ≤ a)
    (hy : squaredLength y ≤ a^2*squaredLength x) :
    squaredLength (x+y) ≤ (1+a)^2*squaredLength x := by
  have h (z : Coordinates d) :
      ‖(toLp 2 z : EuclideanSpace ℝ (Fin d))‖^2 = squaredLength z := by
    rw [EuclideanSpace.real_norm_sq_eq]
    rfl
  have h1 : ‖(toLp 2 y : EuclideanSpace ℝ (Fin d))‖ ≤
      a*‖(toLp 2 x : EuclideanSpace ℝ (Fin d))‖ := by
    rw [← sq_le_sq₀ (norm_nonneg _) (by positivity), mul_pow, h, h]
    exact hy
  have h2 : ‖(toLp 2 (x+y) : EuclideanSpace ℝ (Fin d))‖ ≤
      (1+a)*‖(toLp 2 x : EuclideanSpace ℝ (Fin d))‖ := by
    calc
      ‖(toLp 2 (x+y) : EuclideanSpace ℝ (Fin d))‖ ≤
          ‖(toLp 2 x : EuclideanSpace ℝ (Fin d))‖ + ‖(toLp 2 y : EuclideanSpace ℝ (Fin d))‖ :=
        norm_add_le _ _
      _ ≤ _ := by linarith
  have h3 := (sq_le_sq₀ (norm_nonneg _) (show 0 ≤
      (1+a)*‖(toLp 2 x : EuclideanSpace ℝ (Fin d))‖ by positivity)).mpr h2
  rwa [mul_pow,h,h] at h3

/-- Each normalized endpoint receives a quadratic correction to its first-order transport. -/
noncomputable def spectralStep (u : ℝ) : ℝ := 1-u/2+u^2/4

theorem spectralStep_pos {u : ℝ} (h : |u|≤1/2) : 0 < spectralStep u := by
  unfold spectralStep
  nlinarith [(abs_le.mp h).2,sq_nonneg u]

theorem spectralStep_enclosure {u : ℝ} (h : |u|≤1/2) :
    (1+u)*spectralStep u^2 ≤ 1 := by
  have hu := (abs_le.mp h).2
  have H : 0 ≤ u^2*(1/4-u/2) + u^4*(3/16-u/16) := by
    exact add_nonneg (mul_nonneg (sq_nonneg _) (by linarith))
      (mul_nonneg (by positivity) (by linarith))
  unfold spectralStep
  nlinarith

theorem spectralStep_product (u : ℝ) : 1+u^2/4 ≤ spectralStep u * spectralStep (-u) := by
  unfold spectralStep
  nlinarith [sq_nonneg (u^2)]

noncomputable def stepMatrix (H : Coefficients d) (s : ℝ) : Coefficients d :=
  (1 : Coefficients d) + (-s/2) • H + (1/4:ℝ) • (H*H)

theorem step_spectral_form {H : Coefficients d} (hH : H.IsHermitian) {s : ℝ} (hs : s^2=1) :
    stepMatrix H s = conjStarAlgAut ℝ _ hH.eigenvectorUnitary
        (diagonal fun i => spectralStep (s*hH.eigenvalues i)) := by
  let v := hH.eigenvalues
  have E : diagonal (fun i => spectralStep (s*v i)) =
      ((1 : Coefficients d) + (-s/2) • diagonal v + (1/4:ℝ) • (diagonal v*diagonal v)) := by
    rw [diagonal_mul_diagonal]
    ext i j
    by_cases h : i=j
    · subst j
      simp only [diagonal_apply_eq,Matrix.add_apply,Matrix.one_apply_eq,Matrix.smul_apply,
        smul_eq_mul,Pi.mul_apply,spectralStep,mul_pow,hs,one_mul]
      ring
    · simp [h,diagonal_apply_ne,Matrix.one_apply_ne]
  have H0 : H=conjStarAlgAut ℝ _ hH.eigenvectorUnitary (diagonal v) := hH.spectral_theorem
  rw [E, map_add,map_add,map_one,map_smul,map_smul,map_mul]
  exact congrArg₂ (fun A B : Coefficients d => (1 : Coefficients d) + (-s/2) • A +
      (1/4:ℝ) • (B*B)) H0 H0

theorem shift_spectral_form {H : Coefficients d} (hH : H.IsHermitian) (s : ℝ) :
    (1 : Coefficients d)+s • H = conjStarAlgAut ℝ _ hH.eigenvectorUnitary
      (diagonal fun i => 1+s*hH.eigenvalues i) := by
  have hdiag : diagonal (fun i => 1+s*hH.eigenvalues i) =
      (1 : Coefficients d)+s • diagonal hH.eigenvalues := by
    ext i j
    by_cases h : i=j
    · subst j
      simp
    · simp [h,diagonal_apply_ne,Matrix.one_apply_ne]
  rw [hdiag,map_add,map_one,map_smul]
  congr 2
  exact hH.spectral_theorem

theorem step_preserves_level {H : Coefficients d} (hH : H.IsHermitian)
    (h : ∀ i, |hH.eigenvalues i|≤1/2) {s : ℝ} (hs : s^2=1) (hs' : |s|=1)
    (x : Coordinates d) :
    quadratic ((1 : Coefficients d)+s • H) (stepMatrix H s *ᵥ x) ≤ squaredLength x := by
  let U := hH.eigenvectorUnitary
  let q := fun i => spectralStep (s*hH.eigenvalues i)
  let w := star (U : Coefficients d) *ᵥ x
  have e : star (U : Coefficients d) * stepMatrix H s =
      diagonal q * star (U : Coefficients d) := by
    rw [step_spectral_form hH hs,conjStarAlgAut_apply]
    change star (U : Coefficients d) * ((U : Coefficients d) * diagonal q *
      star (U : Coefficients d)) = _
    simp only [← Matrix.mul_assoc,Unitary.coe_star_mul_self,one_mul]
  rw [shift_spectral_form hH,unitary_quad,Matrix.mulVec_mulVec,e,← Matrix.mulVec_mulVec]
  rw [← unitary_squaredLength U x]
  change ∑ i, (1+s*hH.eigenvalues i)*((diagonal q *ᵥ w) i)^2 ≤ squaredLength w
  unfold squaredLength
  simp only [mulVec_diagonal,mul_pow]
  apply Finset.sum_le_sum
  intro i _
  have H0 : |s*hH.eigenvalues i|≤1/2 := by
    rw [abs_mul,hs',one_mul]
    exact h i
  have hi := spectralStep_enclosure H0
  change (1+s*hH.eigenvalues i)*(q i)^2 ≤ 1 at hi
  have hj := mul_le_mul_of_nonneg_right hi (sq_nonneg (w i))
  linarith [show (1+s*hH.eigenvalues i)*((q i)^2*(w i)^2) =
    ((1+s*hH.eigenvalues i)*(q i)^2)*(w i)^2 by ring]

end Cubature
