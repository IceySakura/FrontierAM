import Theory.FixedAlgebra

open scoped BigOperators Matrix ENNReal
open Matrix Unitary Set MeasureTheory

namespace Cubature

variable {d : ℕ}

structure PairGeometry (A B : Coefficients d) (θ : ℝ) where
  P : Coefficients d
  V : Coefficients d
  H : Coefficients d
  hP : Pᵀ*P = midpointMatrix A B
  hPV : P*V = 1
  hVP : V*P = 1
  hA : Vᵀ*A*V = 1+H
  hB : Vᵀ*B*V = 1-H
  hermitian : H.IsHermitian
  L : ℝ
  hθ : θ ≤ L
  hL₁ : L ≤ 1/2
  spectrum_le : ∀ i, |hermitian.eigenvalues i| ≤ L
  attains : ∃ i, |hermitian.eigenvalues i| = L

theorem midpoint_geometry (A B : Coefficients d) (hd : 0 < d)
    (hA : A.IsHermitian) (hB : B.IsHermitian) (ha : InQuarterBand A) (hb : InQuarterBand B)
    (η : ℝ) (hη : 0 < η) (sep : ∃ i j, η ≤ |A i j-B i j|) :
    Nonempty (PairGeometry A B (η/(8*(d:ℝ)^3))) := by
  let C := midpointMatrix A B
  let D : Coefficients d := (1/2:ℝ) • (A-B)
  let θ := η/(8*(d:ℝ)^3)
  have hd' : (0:ℝ)<d := by exact_mod_cast hd
  have hc : InQuarterBand C := midpoint_quarter ha hb
  obtain ⟨P,V,hC,hPV,hVP,hP,hCpos⟩ := bounded_square_factor C (midpoint_hermitian hA hB)
    (quarter_band_standard hc)
  let H : Coefficients d := Vᵀ*D*V
  have hDs : D.IsHermitian := (hA.sub hB).smul (by simp [IsSelfAdjoint])
  have hHs : H.IsHermitian := Matrix.isHermitian_conjTranspose_mul_mul V hDs
  have hone : Vᵀ*C*V = (1 : Coefficients d) := by
    rw [← hC]
    calc
      Vᵀ * (Pᵀ * P) * V = (P*V)ᵀ*(P*V) := by
        rw [Matrix.transpose_mul]
        simp only [Matrix.mul_assoc]
      _ = 1 := by rw [hPV,transpose_one,mul_one]
  have hpA : Vᵀ*A*V = 1+H := by
    have e : A=C+D := by dsimp [C,D,midpointMatrix]; module
    rw [e,Matrix.mul_add,Matrix.add_mul,hone]
  have hpB : Vᵀ*B*V = 1-H := by
    have e : B=C-D := by dsimp [C,D,midpointMatrix]; module
    rw [e,Matrix.mul_sub,Matrix.sub_mul,hone]
  have hCeq x : quadratic C (V *ᵥ x) = squaredLength x := by
    rw [← quadratic_congruence,hone,quadratic_one]
  have hnorm (x : Coordinates d) :
      (-1/2:ℝ)*squaredLength x ≤ quadratic H x ∧ quadratic H x ≤ (1/2:ℝ)*squaredLength x := by
    have hqd : quadratic H x = (quadratic A (V *ᵥ x)-quadratic B (V *ᵥ x))/2 := by
      change quadratic (Vᵀ*D*V) x = _
      rw [quadratic_congruence]
      change quadratic ((1/2:ℝ) • (A-B)) (V *ᵥ x) = _
      rw [quadratic_smul_matrix,quadratic_sub]
      ring
    have hh := (hc (V *ᵥ x)).1
    rw [hCeq] at hh
    rw [hqd]
    constructor <;> linarith [(ha (V *ᵥ x)).1,(ha (V *ᵥ x)).2,
      (hb (V *ᵥ x)).1,(hb (V *ᵥ x)).2,square_sum_nonneg x]
  have hvals := quadratic_spectrum_range hHs hnorm
  have nonempty : (Finset.univ : Finset (Fin d)).Nonempty := ⟨⟨0,hd⟩,Finset.mem_univ _⟩
  obtain ⟨p,_,hp⟩ := Finset.exists_max_image Finset.univ (fun i => |hHs.eigenvalues i|) nonempty
  let L := |hHs.eigenvalues p|
  have hL0 : 0 ≤ L := abs_nonneg _
  have hL1 : L ≤ 1/2 := abs_le.mpr ⟨by linarith [(hvals p).1],(hvals p).2⟩
  have hr : D=Pᵀ*H*P := by
    calc
      D = (V*P)ᵀ*D*(V*P) := by rw [hVP,transpose_one,one_mul,mul_one]
      _ = Pᵀ*H*P := by simp only [H,Matrix.transpose_mul,Matrix.mul_assoc]
  have hNorm : ∀ i j, |H i j| ≤ (d:ℝ)*L :=
    hermitian_entry_from_spectrum hHs hL0 (fun i => hp i (Finset.mem_univ _))
  have ht : ∀ i j, |(Pᵀ*H) i j| ≤ (d:ℝ)*2*((d:ℝ)*L) :=
    product_entry_bound Pᵀ H (by norm_num) (by positivity) (fun i j => hP j i) hNorm
  obtain ⟨i,j,hsep⟩ := sep
  have hD := product_entry_bound (Pᵀ*H) P (by positivity) (by norm_num) ht hP i j
  have hθ : θ ≤ L := by
    have hm : |D i j| = |A i j-B i j|/2 := by
      change |(1/2:ℝ)*(A i j-B i j)|=_
      rw [abs_mul,abs_of_pos (by norm_num : (0:ℝ)<1/2)]
      ring
    rw [← hr,hm] at hD
    dsimp [θ]
    rw [div_le_iff₀ (by positivity)]
    linarith
  exact ⟨⟨P,V,H,hC,hPV,hVP,hpA,hpB,hHs,L,hθ,hL1,
    fun i => hp i (Finset.mem_univ _),⟨p,rfl⟩⟩⟩

namespace PairGeometry

variable {A B : Coefficients d} {θ : ℝ} (G : PairGeometry A B θ)

noncomputable def mapping (s : ℝ) : Coefficients d := G.V * stepMatrix G.H s * G.P

theorem mapping_level (x : Coordinates d) :
    quadratic A (G.mapping 1 *ᵥ x) ≤ quadratic (midpointMatrix A B) x ∧
    quadratic B (G.mapping (-1) *ᵥ x) ≤ quadratic (midpointMatrix A B) x := by
  have hv y : G.V *ᵥ (G.P *ᵥ y) = y := by rw [mulVec_mulVec,G.hVP,one_mulVec]
  have aux (s : ℝ) : G.mapping s *ᵥ x = G.V *ᵥ (stepMatrix G.H s *ᵥ (G.P *ᵥ x)) := by
    simp [mapping,mulVec_mulVec,Matrix.mul_assoc]
  have hq : squaredLength (G.P *ᵥ x)=quadratic (midpointMatrix A B) x := by
    rw [squaredLength_mul_self,G.hP]
  have hspec := fun i => (G.spectrum_le i).trans G.hL₁
  refine ⟨?_,?_⟩
  · rw [← hq,aux,← quadratic_congruence,G.hA]
    simpa using step_preserves_level G.hermitian hspec (s:=1) (by norm_num) (by norm_num) (G.P *ᵥ x)
  · rw [← hq,aux,← quadratic_congruence,G.hB]
    simpa [sub_eq_add_neg] using
      step_preserves_level G.hermitian hspec (s:=-1) (by norm_num) (by norm_num) (G.P *ᵥ x)

theorem determinant_pair (h0 : 0 ≤ G.L) :
    0 < det (G.mapping 1) ∧ 0 < det (G.mapping (-1)) ∧
      1+G.L^2/4 ≤ det (G.mapping 1)*det (G.mapping (-1)) := by
  let v := G.hermitian.eigenvalues
  let U := G.hermitian.eigenvectorUnitary
  have hdet (s : ℝ) (hs : s^2=1) : det (G.mapping s) = ∏ i, spectralStep (s*v i) := by
    have h := congrArg Matrix.det G.hVP
    simp only [Matrix.det_mul,Matrix.det_one] at h
    rw [mapping,Matrix.det_mul,Matrix.det_mul,mul_right_comm,h,one_mul,
      step_spectral_form G.hermitian hs,conjugate_diagonal_det]
  rw [hdet 1 (by norm_num),hdet (-1) (by norm_num)]
  simp only [one_mul,neg_one_mul,← Finset.prod_mul_distrib]
  have hv (i : Fin d) : |v i|≤1/2 := (G.spectrum_le i).trans G.hL₁
  have hp (i : Fin d) : 0 < spectralStep (v i) := spectralStep_pos (hv i)
  have hm (i : Fin d) : 0 < spectralStep (-v i) :=
    spectralStep_pos (by simpa only [abs_neg] using hv i)
  refine ⟨Finset.prod_pos (fun i _ => hp i),Finset.prod_pos (fun i _ => hm i),?_⟩
  obtain ⟨p,hep⟩ := G.attains
  have hp2 : (v p)^2=G.L^2 := by
    rw [← sq_abs (v p)]
    exact congrArg (fun t : ℝ => t^2) hep
  calc
    1+G.L^2/4 = ∏ i, if i=p then 1+G.L^2/4 else 1 := by simp
    _ ≤ ∏ i, spectralStep (v i)*spectralStep (-v i) := by
      apply Finset.prod_le_prod₀
      · intro i _
        split <;> positivity
      · intro i _
        by_cases h : i=p
        · subst i
          simpa only [hp2,↓reduceIte] using spectralStep_product (v p)
        · simp only [h,↓reduceIte]
          exact (by nlinarith [sq_nonneg (v i)] : (1:ℝ) ≤ 1+(v i)^2/4).trans
            (spectralStep_product _)

theorem paired_norm (h0 : 0 ≤ G.L) (hc : InQuarterBand (midpointMatrix A B)) (x : Coordinates d) :
    squaredLength (G.mapping 1 *ᵥ x)+squaredLength (G.mapping (-1) *ᵥ x) ≤
      2*(1+3*G.L^2)*squaredLength x := by
  let L := G.L
  let y := G.P *ᵥ x
  let t := G.H *ᵥ y
  let u := G.H *ᵥ t
  let a := G.V *ᵥ t
  let b := G.V *ᵥ u
  have hIV z : G.V *ᵥ (G.P *ᵥ z)=z := by rw [mulVec_mulVec,G.hVP,one_mulVec]
  have hIP z : G.P *ᵥ (G.V *ᵥ z)=z := by rw [mulVec_mulVec,G.hPV,one_mulVec]
  have hP (z : Coordinates d) : squaredLength (G.P *ᵥ z) = quadratic (midpointMatrix A B) z := by
    rw [squaredLength_mul_self,G.hP]
  have hV z : squaredLength (G.V *ᵥ z) ≤ (4/3:ℝ)*squaredLength z := by
    have he := (hc (G.V *ᵥ z)).1
    rw [← hP,hIP] at he
    linarith
  have hH := hermitian_action_norm G.hermitian h0 G.spectrum_le
  have ha : squaredLength a ≤ 4*L^2*squaredLength x := by
    have hn := hV t
    have hi := mul_le_mul_of_nonneg_left (hH y) (by norm_num : (0:ℝ)≤4/3)
    have hh := mul_le_mul_of_nonneg_left ((hP x).le.trans (hc x).2) (show 0 ≤ (4/3:ℝ)*L^2 by positivity)
    have h₀ := mul_nonneg (sq_nonneg L) (square_sum_nonneg x)
    dsimp [a,t] at hn ⊢
    change _ ≤ (4/3:ℝ)*squaredLength (G.H *ᵥ y) at hn
    linarith [hn,hi,hh]
  have hb : squaredLength b ≤ 4*(L^2)^2*squaredLength x := by
    have hn := hV u
    have hi := mul_le_mul_of_nonneg_left (hH t) (by norm_num : (0:ℝ)≤4/3)
    have hj := mul_le_mul_of_nonneg_left (hH y) (show 0 ≤ (4/3:ℝ)*L^2 by positivity)
    have hh := mul_le_mul_of_nonneg_left ((hP x).le.trans (hc x).2)
      (show 0 ≤ (4/3:ℝ)*(L^2)^2 by positivity)
    have h₀ := mul_nonneg (sq_nonneg (L^2)) (square_sum_nonneg x)
    dsimp [b,u] at hn ⊢
    linarith [hn,hi,hj,hh]
  have e (s : ℝ) : G.mapping s *ᵥ x = (x+(1/4:ℝ) • b) + (-s/2) • a := by
    have he : G.mapping s *ᵥ x =
        G.V *ᵥ (stepMatrix G.H s *ᵥ (G.P *ᵥ x)) := by
      simp [mapping,mulVec_mulVec,Matrix.mul_assoc]
    rw [he,stepMatrix]
    simp only [add_mulVec,smul_mulVec,one_mulVec,← mulVec_mulVec,mulVec_add,mulVec_smul,hIV]
    change x + (-s/2) • a + (1/4:ℝ) • b = _
    abel
  have H₀ : squaredLength ((1/2:ℝ) • a) ≤ L^2*squaredLength x := by
    rw [squaredLength_smul]
    linarith
  have H₁ : squaredLength ((1/4:ℝ) • b) ≤ (L^2/2)^2*squaredLength x := by
    rw [squaredLength_smul]
    linarith
  have H₂ := squaredLength_add_bounded x ((1/4:ℝ) • b) (show 0 ≤ L^2/2 by positivity) H₁
  have H₃ := squaredLength_parallelogram (x+(1/4:ℝ) • b) ((1/2:ℝ) • a)
  have HH : (L^2)^2 ≤ L^2 := by
    have hL : L^2 ≤ 1 := by
      have h := (sq_le_sq₀ h0 (by norm_num : (0:ℝ)≤1/2)).mpr G.hL₁
      linarith
    nlinarith [mul_le_mul_of_nonneg_left hL (sq_nonneg L)]
  rw [e 1,e (-1)]
  have e1 : (-1/2:ℝ) • a = -((1/2:ℝ) • a) := by
    rw [← neg_smul]
    congr 1
    norm_num
  rw [show (- -(1:ℝ)/2) = 1/2 by norm_num,e1,← sub_eq_add_neg]
  rw [add_comm (squaredLength (_-_)),H₃]
  have hpoly : (1+L^2/2)^2+L^2 ≤ 1+3*L^2 := by nlinarith
  have hk := mul_le_mul_of_nonneg_right hpoly (square_sum_nonneg x)
  linarith

end PairGeometry

end Cubature
