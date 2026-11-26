import Theory.PrecisionModel
import Theory.LowerAlgebra

open scoped BigOperators Matrix
open Matrix Set Unitary

namespace Cubature

variable {d : ℕ}

theorem unitary_entry_bound (U : Matrix.unitaryGroup (Fin d) ℝ) (i j : Fin d) :
    |(U : Coefficients d) i j| ≤ 1 := by
  have h := congrArg (fun A : Coefficients d => A j j) (Unitary.coe_star_mul_self U)
  have hs : ∑ k, ((U : Coefficients d) k j)^2 = 1 := by
    simpa [Matrix.mul_apply, pow_two] using h
  have hr : ((U : Coefficients d) i j)^2 ≤ ∑ k, ((U : Coefficients d) k j)^2 :=
    Finset.single_le_sum (fun k _ => sq_nonneg ((U : Coefficients d) k j)) (Finset.mem_univ _)
  rw [← sq_le_sq₀ (abs_nonneg _) (by norm_num : (0:ℝ)≤1), sq_abs, one_pow]
  simpa [hs] using hr

theorem product_entry_bound (A B : Coefficients d) {u v : ℝ}
    (hu : 0 ≤ u) (hv : 0 ≤ v)
    (hA : ∀ i j, |A i j| ≤ u) (hB : ∀ i j, |B i j| ≤ v) (i j : Fin d) :
    |(A*B) i j| ≤ (d:ℝ)*u*v := by
  rw [Matrix.mul_apply]
  calc
    |∑ k, A i k*B k j| ≤ ∑ k, |A i k*B k j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _ : Fin d, u*v := by
      apply Finset.sum_le_sum
      intro k _
      rw [abs_mul]
      exact mul_le_mul (hA i k) (hB k j) (abs_nonneg _) hu
    _ = (d:ℝ)*u*v := by simp; ring

theorem quadratic_entry_bound (A : Coefficients d) (x : Coordinates d) {u : ℝ}
    (hu : 0 ≤ u) (hA : ∀ i j, |A i j| ≤ u) :
    |quadratic A x| ≤ (d:ℝ)*u*squaredLength x := by
  unfold quadratic
  calc
    |∑ i, ∑ j, A i j * x i * x j| ≤ ∑ i, |∑ j, A i j * x i * x j| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ∑ j, u*((x i)^2+(x j)^2)/2 := by
      apply Finset.sum_le_sum
      intro i _
      apply (Finset.abs_sum_le_sum_abs _ _).trans
      apply Finset.sum_le_sum
      intro j _
      rw [abs_mul, abs_mul]
      have hi : |x i| * |x j| ≤ ((x i)^2+(x j)^2)/2 := by
        have h := sq_nonneg (|x i|-|x j|)
        nlinarith [sq_abs (x i), sq_abs (x j)]
      have H : |A i j| * (|x i| * |x j|) ≤ u * (((x i)^2+(x j)^2)/2) :=
        mul_le_mul (hA i j) hi (mul_nonneg (abs_nonneg _) (abs_nonneg _)) hu
      linarith [show |A i j| * |x i| * |x j| = |A i j| * (|x i| * |x j|) by ring]
    _ = (d:ℝ)*u*squaredLength x := by
      simp only [squaredLength, mul_add, add_div, Finset.sum_add_distrib, Finset.sum_const,
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      simp only [div_eq_mul_inv, ← Finset.mul_sum, ← Finset.sum_mul, mul_assoc]
      ring

theorem conjugate_diagonal_det (U : Matrix.unitaryGroup (Fin d) ℝ) (v : Coordinates d) :
    det (conjStarAlgAut ℝ _ U (diagonal v)) = ∏ i, v i := by
  have h := congrArg Matrix.det (Unitary.coe_mul_star_self U)
  simp only [Matrix.det_mul,Matrix.det_one] at h
  change det (U : Coefficients d) * det (star (U : Coefficients d)) = 1 at h
  rw [conjStarAlgAut_apply, Matrix.det_mul, Matrix.det_mul, Matrix.det_diagonal]
  rw [mul_right_comm, h, one_mul]

theorem quadratic_congruence (A P : Coefficients d) (x : Coordinates d) :
    quadratic (Pᵀ*A*P) x = quadratic A (P *ᵥ x) := by
  rw [quadratic_dot, Matrix.mul_assoc, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
    Matrix.dotProduct_transpose_mulVec, dotProduct_comm, ← quadratic_dot]

theorem orthogonal_axis_quadratic (U : Matrix.unitaryGroup (Fin d) ℝ) (v : Coordinates d) (i : Fin d) :
    squaredLength ((U : Coefficients d) *ᵥ coordinate i) = 1 ∧
    quadratic (conjStarAlgAut ℝ _ U (diagonal v)) ((U : Coefficients d) *ᵥ coordinate i) = v i := by
  have hW : star (U : Coefficients d) *ᵥ ((U : Coefficients d) *ᵥ coordinate i) =
      coordinate i := by rw [mulVec_mulVec, Unitary.coe_star_mul_self, one_mulVec]
  have hQ := unitary_squaredLength U ((U : Coefficients d) *ᵥ coordinate i)
  rw [hW] at hQ
  constructor
  · simpa [squaredLength,coordinate] using hQ.symm
  · rw [unitary_quad,hW]
    simp [coordinate,mul_ite]

theorem quadratic_spectrum_range {A : Coefficients d} (hA : A.IsHermitian)
    {l u : ℝ}
    (h : ∀ x, l*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ u*squaredLength x) :
    ∀ i, l ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ u := by
  intro i
  have H := h ((hA.eigenvectorUnitary : Coefficients d) *ᵥ coordinate i)
  have e := orthogonal_axis_quadratic hA.eigenvectorUnitary hA.eigenvalues i
  have eqA : A = conjStarAlgAut ℝ _ hA.eigenvectorUnitary (diagonal hA.eigenvalues) := hA.spectral_theorem
  have eqQ : quadratic A ((hA.eigenvectorUnitary : Coefficients d) *ᵥ coordinate i) =
      hA.eigenvalues i := (congrArg (fun M => quadratic M
        ((hA.eigenvectorUnitary : Coefficients d) *ᵥ coordinate i)) eqA).trans e.2
  rw [e.1, eqQ, mul_one, mul_one] at H
  exact H

theorem bounded_square_factor (A : Coefficients d) (hA : A.IsHermitian)
    (h : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x) :
    ∃ P V : Coefficients d, Pᵀ*P = A ∧ P*V = 1 ∧ V*P = 1 ∧
      (∀ i j, |P i j| ≤ 2) ∧ 0 < det A := by
  let U := hA.eigenvectorUnitary
  let c := hA.eigenvalues
  have hv := quadratic_spectrum_range hA h
  let r := fun i => Real.sqrt (c i)
  let P := diagonal r * star (U : Coefficients d)
  let V := (U : Coefficients d) * diagonal (fun i => (r i)⁻¹)
  have hr i : 0 < r i := Real.sqrt_pos.mpr (by linarith [(hv i).1])
  have hr₂ i : (r i)^2 = c i := Real.sq_sqrt (by linarith [(hv i).1])
  have eqA : A = conjStarAlgAut ℝ _ U (diagonal c) := hA.spectral_theorem
  have hUU : (U : Coefficients d) * star (U : Coefficients d) = 1 := Unitary.coe_mul_star_self U
  have hUUt : star (U : Coefficients d) * (U : Coefficients d) = 1 := Unitary.coe_star_mul_self U
  have hP : Pᵀ = (U : Coefficients d) * diagonal r := by
    dsimp [P]
    rw [transpose_mul, diagonal_transpose]
    rfl
  have hdiag1 : diagonal r * diagonal (fun i => (r i)⁻¹) = (1 : Coefficients d) := by
    rw [diagonal_mul_diagonal]
    convert (Matrix.diagonal_one : diagonal (fun _ : Fin d => (1:ℝ)) = 1) using 2
    ext i
    exact mul_inv_cancel₀ (hr i).ne'
  have hdiag2 : diagonal (fun i => (r i)⁻¹) * diagonal r = (1 : Coefficients d) := by
    rw [diagonal_mul_diagonal]
    convert (Matrix.diagonal_one : diagonal (fun _ : Fin d => (1:ℝ)) = 1) using 2
    ext i
    exact inv_mul_cancel₀ (hr i).ne'
  have hdiag3 : diagonal r * diagonal r = diagonal c := by
    rw [diagonal_mul_diagonal]
    congr 1
    ext i
    simpa [pow_two] using hr₂ i
  refine ⟨P,V,?_,?_,?_,?_,?_⟩
  · calc
      Pᵀ*P = (U : Coefficients d)*(diagonal r*diagonal r)*star (U : Coefficients d) := by
        rw [hP]
        dsimp [P]
        simp only [Matrix.mul_assoc]
      _ = A := by
        rw [hdiag3]
        exact eqA.symm
  · calc
      P*V = diagonal r * (star (U : Coefficients d)*(U : Coefficients d)) *
          diagonal (fun i => (r i)⁻¹) := by
        dsimp [P,V]
        simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hUUt,mul_one,hdiag1]
  · calc
      V*P = (U : Coefficients d) * (diagonal (fun i => (r i)⁻¹) * diagonal r) *
          star (U : Coefficients d) := by
        dsimp [P,V]
        simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hdiag2,mul_one,hUU]
  · intro i j
    have hR : r i ≤ 2 := by nlinarith [hr i,hr₂ i,(hv i).2]
    have huw : |(U : Coefficients d) j i| ≤ 1 := unitary_entry_bound U j i
    change |(diagonal r * star (U : Coefficients d)) i j| ≤ 2
    rw [diagonal_mul,abs_mul,abs_of_pos (hr i)]
    change r i * |(U : Coefficients d) j i| ≤ 2
    exact (mul_le_mul_of_nonneg_left huw (hr i).le).trans_eq (mul_one _) |>.trans hR
  · rw [eqA,conjugate_diagonal_det]
    exact Finset.prod_pos (fun i _ => by linarith [(hv i).1])

theorem hermitian_entry_from_spectrum {H : Coefficients d} (hH : H.IsHermitian) {lim : ℝ}
    (hlim : 0 ≤ lim) (he : ∀ i, |hH.eigenvalues i| ≤ lim) (i j : Fin d) :
    |H i j| ≤ (d:ℝ)*lim := by
  let U := hH.eigenvectorUnitary
  have hf : ∀ i j, |((U : Coefficients d) * diagonal hH.eigenvalues) i j| ≤ lim := by
    intro i j
    rw [mul_diagonal,abs_mul]
    exact (mul_le_mul (unitary_entry_bound U i j) (he j) (abs_nonneg _) zero_le_one).trans_eq (one_mul _)
  have g := product_entry_bound _ (star (U : Coefficients d)) hlim zero_le_one hf
    (fun i j => unitary_entry_bound U j i) i j
  change |(conjStarAlgAut ℝ _ U (diagonal hH.eigenvalues)) i j| ≤ (d:ℝ)*lim*1 at g
  have eqH : H = conjStarAlgAut ℝ _ U (diagonal hH.eigenvalues) := hH.spectral_theorem
  have H := congrArg (fun T : Coefficients d => |T i j|) eqH
  exact H.le.trans (by simpa only [mul_one] using g)

theorem midpoint_quadratic (A B : Coefficients d) (x : Coordinates d) :
    quadratic (midpointMatrix A B) x = (quadratic A x+quadratic B x)/2 := by
  rw [midpointMatrix,quadratic_smul_matrix,quadratic_add]
  ring

theorem midpoint_bounds {A B : Coefficients d}
    (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x)
    (hb : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic B x ∧ quadratic B x ≤ (3/2:ℝ)*squaredLength x) :
    ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic (midpointMatrix A B) x ∧
      quadratic (midpointMatrix A B) x ≤ (3/2:ℝ)*squaredLength x := by
  intro x
  rw [midpoint_quadratic]
  constructor <;> linarith [(ha x).1,(ha x).2,(hb x).1,(hb x).2]

theorem midpoint_hermitian {A B : Coefficients d} (ha : A.IsHermitian) (hb : B.IsHermitian) :
    (midpointMatrix A B).IsHermitian :=
  (ha.add hb).smul (by simp [IsSelfAdjoint])

theorem quadratic_sub (A B : Coefficients d) (x : Coordinates d) :
    quadratic (A-B) x = quadratic A x - quadratic B x := by
  rw [sub_eq_add_neg,quadratic_add]
  simp [quadratic, Matrix.neg_apply, neg_mul, Finset.sum_neg_distrib, sub_eq_add_neg]

theorem determinant_congruence (A V : Coefficients d) :
    det (Vᵀ*A*V) = (det V)^2 * det A := by
  simp only [Matrix.det_mul, Matrix.det_transpose]
  ring

theorem determinant_shift_spectrum {H : Coefficients d} (hH : H.IsHermitian) (s : ℝ) :
    det ((1 : Coefficients d) + s • H) = ∏ i, (1+s*hH.eigenvalues i) := by
  have hdiag : diagonal (fun i => 1+s*hH.eigenvalues i) =
      (1 : Coefficients d)+s • diagonal hH.eigenvalues := by
    ext i j
    by_cases h : i=j
    · subst j
      simp
    · simp [h, diagonal_apply_ne, Matrix.one_apply_ne]
  have HC : (1 : Coefficients d)+s • H =
      conjStarAlgAut ℝ _ hH.eigenvectorUnitary (diagonal fun i => 1+s*hH.eigenvalues i) := by
    rw [hdiag,map_add,map_one,map_smul]
    congr 2
    exact hH.spectral_theorem
  rw [HC,conjugate_diagonal_det]

theorem determinant_midpoint_gap (A B : Coefficients d) (hd : 0 < d)
    (hA : A.IsHermitian) (hB : B.IsHermitian)
    (ha : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x)
    (hb : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic B x ∧ quadratic B x ≤ (3/2:ℝ)*squaredLength x)
    (η : ℝ) (hη : 0 < η) (sep : ∃ i j, η ≤ |A i j-B i j|) :
    let C := midpointMatrix A B
    let θ := η / (8*(d:ℝ)^3)
    0 < θ ∧ θ ≤ 1 ∧
      (det A ≤ (1-θ^2/2)*det C ∨ det B ≤ (1-θ^2/2)*det C) := by
  let C := midpointMatrix A B
  let D : Coefficients d := (1/2:ℝ) • (A-B)
  let θ := η/(8*(d:ℝ)^3)
  have hd' : (0:ℝ)<d := by exact_mod_cast hd
  obtain ⟨P,V,hC,hPV,hVP,hP,hCpos⟩ := bounded_square_factor C (midpoint_hermitian hA hB)
    (midpoint_bounds ha hb)
  let H := Vᵀ*D*V
  have htD : D.IsHermitian := (hA.sub hB).smul (by simp [IsSelfAdjoint])
  have hH : H.IsHermitian := by
    exact Matrix.isHermitian_conjTranspose_mul_mul V htD
  have hT : Vᵀ*C*V = 1 := by
    rw [← hC]
    calc
      Vᵀ * (Pᵀ * P) * V = (P*V)ᵀ*(P*V) := by
        rw [Matrix.transpose_mul]
        simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hPV,transpose_one,mul_one]
  have heA : A = C+D := by
    dsimp [C,D,midpointMatrix]
    module
  have heB : B = C-D := by
    dsimp [C,D,midpointMatrix]
    module
  have h1 : Vᵀ*A*V = 1+H := by
    rw [heA,Matrix.mul_add,Matrix.add_mul,hT]
  have h2 : Vᵀ*B*V = 1-H := by
    rw [heB,Matrix.mul_sub,Matrix.sub_mul,hT]
  have hQ (x : Coordinates d) :
      (-1:ℝ)*squaredLength x ≤ quadratic H x ∧ quadratic H x ≤ squaredLength x := by
    have q1 := (ha (V *ᵥ x)).1
    have q2 := (hb (V *ᵥ x)).1
    rw [← quadratic_congruence A V x,h1,quadratic_add,quadratic_one] at q1
    rw [← quadratic_congruence B V x,h2,quadratic_sub,quadratic_one] at q2
    have h0 : 0 ≤ squaredLength (V *ᵥ x) := Finset.sum_nonneg (fun i _ => sq_nonneg _)
    constructor <;> linarith
  have hlim := quadratic_spectrum_range hH (l:=-1) (u:=1)
    (fun x => ⟨(hQ x).1,by simpa using (hQ x).2⟩)
  let eig := hH.eigenvalues
  have H0 : (Finset.univ : Finset (Fin d)).Nonempty :=
    ⟨⟨0,hd⟩,Finset.mem_univ _⟩
  obtain ⟨p,_,hp⟩ := Finset.exists_max_image Finset.univ (fun i => |eig i|) H0
  let L := |eig p|
  have hL0 : 0 ≤ L := abs_nonneg _
  have hL1 : L ≤ 1 := by
    exact abs_le.mpr (hlim p)
  have hback : D = Pᵀ*H*P := by
    calc
      D = (V*P)ᵀ*D*(V*P) := by rw [hVP,transpose_one,one_mul,mul_one]
      _ = Pᵀ*H*P := by simp only [H,Matrix.transpose_mul,Matrix.mul_assoc]
  have hE : ∀ i j, |H i j| ≤ (d:ℝ)*L :=
    hermitian_entry_from_spectrum hH hL0 (fun i => hp i (Finset.mem_univ _))
  obtain ⟨i,j,hsep⟩ := sep
  have htmp : ∀ i j, |(Pᵀ*H) i j| ≤ (d:ℝ)*2*((d:ℝ)*L) :=
    product_entry_bound Pᵀ H (by norm_num) (by positivity) (fun i j => hP j i) hE
  have hD := product_entry_bound (Pᵀ*H) P (by positivity) (by norm_num) htmp hP i j
  have hθ : θ ≤ L := by
    have hm : |D i j| = |A i j-B i j|/2 := by
      change |(1/2:ℝ)*(A i j-B i j)| = _
      rw [abs_mul,abs_of_pos (by norm_num : (0:ℝ)<1/2)]
      ring
    rw [← hback,hm] at hD
    dsimp [θ]
    rw [div_le_iff₀ (by positivity)]
    linarith
  have hθ0 : 0 < θ := by dsimp [θ]; positivity
  have hθ1 : θ ≤ 1 := hθ.trans hL1
  have hprod : (∏ i, (1+eig i))*(∏ i, (1-eig i)) ≤ 1-θ^2 := by
    rw [← Finset.prod_mul_distrib]
    have hdet i : (1+eig i)*(1-eig i)=1-(eig i)^2 := by ring
    simp_rw [hdet]
    have hsq i : (eig i)^2 ≤ 1 := by
      have he := abs_le.mpr (hlim i)
      calc
        (eig i)^2 = |eig i|^2 := (sq_abs _).symm
        _ ≤ (1:ℝ)^2 := (sq_le_sq₀ (abs_nonneg _) zero_le_one).mpr he
        _ = 1 := by norm_num
    have hpp : θ^2 ≤ (eig p)^2 := by
      calc
        θ^2 ≤ L^2 := (sq_le_sq₀ hθ0.le hL0).mpr hθ
        _ = (eig p)^2 := sq_abs _
    calc
      (∏ i, (1-(eig i)^2)) ≤ ∏ i, if i=p then 1-θ^2 else 1 := by
        apply Finset.prod_le_prod₀
        · intro i _
          exact sub_nonneg.mpr (hsq i)
        · intro i _
          by_cases h : i=p
          · subst i
            simp only [↓reduceIte]
            linarith
          · simp [h, sq_nonneg]
      _ = 1-θ^2 := by simp
  have hdet0 : (det V)^2*det C = 1 := by
    have hh := congrArg Matrix.det hT
    change det (Vᵀ*C*V) = det (1 : Coefficients d) at hh
    rwa [determinant_congruence,Matrix.det_one] at hh
  have hdA : (det V)^2*det A = ∏ i, (1+eig i) := by
    rw [← determinant_congruence,h1]
    simpa using determinant_shift_spectrum hH 1
  have hdB : (det V)^2*det B = ∏ i, (1-eig i) := by
    rw [← determinant_congruence,h2,sub_eq_add_neg,← one_smul ℝ H,← neg_smul]
    simpa [sub_eq_add_neg] using determinant_shift_spectrum hH (-1)
  have hprod2 : det A * det B ≤ (det C)^2*(1-θ^2) := by
    calc
      det A*det B = ((det V)^2*det C)^2*(det A*det B) := by rw [hdet0,one_pow,one_mul]
      _ = (det C)^2 * ((∏ i, (1+eig i))*(∏ i, (1-eig i))) := by rw [← hdA,← hdB]; ring
      _ ≤ (det C)^2*(1-θ^2) := mul_le_mul_of_nonneg_left hprod (sq_nonneg _)
  change 0 < θ ∧ θ ≤ 1 ∧ _
  refine ⟨hθ0,hθ1,?_⟩
  by_contra! H
  have ht : 0 ≤ (1-θ^2/2)*det C := by
    have hh : θ^2≤1 := (sq_le_sq₀ hθ0.le zero_le_one).mpr hθ1 |>.trans_eq (by norm_num)
    exact mul_nonneg (by linarith) hCpos.le
  have hlt : ((1-θ^2/2)*det C)^2 < det A * det B := by
    rw [pow_two]
    calc
      (1-θ^2/2)*det C * ((1-θ^2/2)*det C) ≤ (1-θ^2/2)*det C * det B :=
        mul_le_mul_of_nonneg_left H.2.le ht
      _ < det A * det B := mul_lt_mul_of_pos_right H.1 (lt_of_le_of_lt ht H.2)
  have hle : (det C)^2*(1-θ^2) ≤ ((1-θ^2/2)*det C)^2 := by
    have he : ((1-θ^2/2)*det C)^2 - (det C)^2*(1-θ^2) =
        (θ^2*det C/2)^2 := by ring
    linarith [sq_nonneg (θ^2*det C/2)]
  exact (hprod2.trans hle).not_gt hlt

end Cubature
