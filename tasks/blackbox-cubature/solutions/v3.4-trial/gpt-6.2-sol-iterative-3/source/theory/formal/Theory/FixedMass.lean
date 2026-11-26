import Theory.FixedTransport

open scoped BigOperators Matrix ENNReal
open Matrix Set MeasureTheory

namespace Cubature

variable {d : ℕ}

theorem gaussian_matrix_image (M : Coefficients d) (hM : 0 < det M)
    {S : Set (Coordinates d)} (hs : MeasurableSet S) :
    gaussian ((fun x => M *ᵥ x) '' S) = ENNReal.ofReal (det M) *
      ∫⁻ x in S, gaussianDensity (M *ᵥ x) := by
  let F : Coordinates d → Coordinates d := fun x => M *ᵥ x
  have hd : LinearMap.det M.mulVecLin ≠ 0 := by
    change LinearMap.det (Matrix.toLin' M) ≠ 0
    rw [LinearMap.det_toLin']
    exact hM.ne'
  let e := (M.mulVecLin.equivOfDetNeZero hd).toContinuousLinearEquiv.toHomeomorph
  have hF : e = F := rfl
  have him : MeasurableSet (F '' S) := by rw [← hF]; exact e.measurableEmbedding.measurableSet_image.mpr hs
  have hpre : F ⁻¹' (F '' S)=S := by rw [← hF, preimage_image_eq _ e.injective]
  have hg : Measurable F := by rw [← hF]; exact e.continuous.measurable
  have map : (volume : Measure (Coordinates d)).map F =
      ENNReal.ofReal (det M)⁻¹ • volume := by
    change Measure.map (Matrix.toLin' M) volume = _
    rw [Real.map_matrix_volume_pi_eq_smul_volume_pi hM.ne',abs_of_pos (inv_pos.mpr hM)]
  have h := setLIntegral_map (μ := (volume : Measure (Coordinates d)))
    him measurable_gaussianDensity hg
  rw [hpre,map,Measure.restrict_smul,lintegral_smul_measure] at h
  change gaussian (F '' S) = ENNReal.ofReal (det M) * ∫⁻ x in S, gaussianDensity (F x)
  rw [gaussian,withDensity_apply _ him,← h,smul_eq_mul,← mul_assoc,← ENNReal.ofReal_mul hM.le,
    mul_inv_cancel₀ hM.ne',ENNReal.ofReal_one,one_mul]

private theorem exponential_pair (x y z k l s : ℝ) (hk : 0 < k) (hl : 0 < l)
    (hs : 0 ≤ s) (hs' : s ≤ 1) (hx : x ≤ 2/100)
    (hdet : 1+s/4≤k*l) (hpair : y+z ≤ 2*(1+3*s)*x) :
    (2*(1+s/16))*Real.exp (-x/2) ≤
      k*Real.exp (-y/2)+l*Real.exp (-z/2) := by
  have hs2 : s^2 ≤ s := by nlinarith [mul_nonneg hs (sub_nonneg.mpr hs')]
  let a := -y/2
  let b := -z/2
  let c := -x/2
  have hC : c-3*s/100 ≤ (a+b)/2 := by
    have fact := mul_le_mul_of_nonneg_left hx (show 0 ≤ (3/2:ℝ)*s by positivity)
    dsimp [a,b,c]
    nlinarith
  let u := k * Real.exp a
  let v := l * Real.exp b
  have hu : 0 ≤ u := by dsimp [u]; positivity
  have hv : 0 ≤ v := by dsimp [v]; positivity
  have am : 2*Real.sqrt (u*v) ≤ u+v := by
    rw [Real.sqrt_mul hu]
    nlinarith [Real.sq_sqrt hu,Real.sq_sqrt hv,sq_nonneg (Real.sqrt u-Real.sqrt v)]
  have hroot : Real.sqrt (u*v) = Real.sqrt (k*l)*Real.exp ((a+b)/2) := by
    have e : u*v = (k*l)*(Real.exp ((a+b)/2))^2 := by
      dsimp [u,v]
      rw [pow_two,← Real.exp_add, show (a+b)/2+(a+b)/2 = a+b by ring, Real.exp_add]
      ring
    rw [e,Real.sqrt_mul (mul_nonneg hk.le hl.le),Real.sqrt_sq (Real.exp_pos _).le]
  have hdet' : 1+s/10 ≤ Real.sqrt (k*l) := by
    rw [← sq_le_sq₀ (by positivity : 0≤1+s/10) (Real.sqrt_nonneg _),
      Real.sq_sqrt (mul_nonneg hk.le hl.le)]
    nlinarith
  have hlin : 1-3*s/100 ≤ Real.exp (-3*s/100) := by
    have H := Real.add_one_le_exp (-3*s/100)
    linarith
  have hr : 1+s/16 ≤ (1+s/10)*(1-3*s/100) := by nlinarith
  have H : (1+s/16)*Real.exp c ≤ Real.sqrt (u*v) := by
    calc
      (1+s/16)*Real.exp c ≤ ((1+s/10)*(1-3*s/100))*Real.exp c :=
        mul_le_mul_of_nonneg_right hr (Real.exp_pos _).le
      _ ≤ Real.sqrt (k*l) * (Real.exp (-3*s/100)*Real.exp c) := by
        have t := mul_le_mul hdet' hlin (by linarith : 0≤1-3*s/100)
          (Real.sqrt_nonneg (k*l))
        simpa only [mul_assoc] using mul_le_mul_of_nonneg_right t (Real.exp_pos _).le
      _ = Real.sqrt (k*l)*Real.exp (c-3*s/100) := by
        rw [← Real.exp_add]
        congr 2
        ring
      _ ≤ Real.sqrt (u*v) := by
        rw [hroot]
        exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hC) (Real.sqrt_nonneg _)
  dsimp [u,v,a,b,c] at am H
  linarith

theorem PairGeometry.gaussian_mass_pair {A B : Coefficients d} {θ : ℝ}
    (G : PairGeometry A B θ) (hθ : 0 < θ) (ha : InQuarterBand A) (hb : InQuarterBand B)
    {r2 : ℝ} (hr : 0 < r2) (hr' : r2 ≤ 1/100) :
    let I := (gaussian (bodyAt r2 (midpointMatrix A B))).toReal
    let I₁ := (gaussian (bodyAt r2 A)).toReal
    let I₂ := (gaussian (bodyAt r2 B)).toReal
    2*(1+θ^2/16)*I ≤ I₁+I₂ := by
  let C := midpointMatrix A B
  let S := bodyAt r2 C
  let M := G.mapping 1
  let N := G.mapping (-1)
  let k := det M
  let l := det N
  let s := G.L^2
  have hL : 0 ≤ G.L := hθ.le.trans G.hθ
  have hs : 0 ≤ s := sq_nonneg _
  have hs' : s ≤ 1 := by
    have h := (sq_le_sq₀ hL (by norm_num : (0:ℝ)≤1/2)).mpr G.hL₁
    dsimp [s]; linarith
  have hsθ : θ^2 ≤ s := (sq_le_sq₀ hθ.le hL).mpr G.hθ
  have hc := midpoint_quarter ha hb
  obtain ⟨hk,hl,hp⟩ := G.determinant_pair hL
  have hmaps (x : Coordinates d) := G.mapping_level x
  have hma : (fun x : Coordinates d => M *ᵥ x) '' S ⊆ bodyAt r2 A := by
    rintro z ⟨x,hx,rfl⟩
    exact (hmaps x).1.trans hx
  have hmb : (fun x : Coordinates d => N *ᵥ x) '' S ⊆ bodyAt r2 B := by
    rintro z ⟨x,hx,rfl⟩
    exact (hmaps x).2.trans hx
  have hm := measurable_bodyAt r2 C
  have hma' := (gaussian_matrix_image M hk hm).symm.trans_le (measure_mono hma)
  have hmb' := (gaussian_matrix_image N hl hm).symm.trans_le (measure_mono hmb)
  have hmass :
      ENNReal.ofReal (2*(1+s/16))*gaussian S ≤
        gaussian (bodyAt r2 A) + gaussian (bodyAt r2 B) := by
    rw [gaussian,withDensity_apply _ hm, ← lintegral_const_mul _ measurable_gaussianDensity]
    calc
      (∫⁻ x in S, ENNReal.ofReal (2*(1+s/16))*gaussianDensity x) ≤
          ∫⁻ x in S, (ENNReal.ofReal k*gaussianDensity (M *ᵥ x) +
                      ENNReal.ofReal l*gaussianDensity (N *ᵥ x)) := by
        apply lintegral_mono_ae
        filter_upwards [ae_restrict_mem hm] with x hx
        have hlo : (1/2:ℝ)*squaredLength x ≤ quadratic C x := (quarter_band_standard hc x).1
        have hi : squaredLength x ≤ 2/100 := by
          change quadratic C x ≤ r2 at hx
          linarith
        have H := exponential_pair (squaredLength x) (squaredLength (M *ᵥ x))
          (squaredLength (N *ᵥ x)) k l s hk hl hs hs' hi hp (G.paired_norm hL hc x)
        let K := (Real.sqrt (2*Real.pi))⁻¹ ^ d
        have hK : 0 ≤ K := by dsimp [K]; positivity
        change ENNReal.ofReal (2*(1+s/16)) * ENNReal.ofReal (K*Real.exp (-squaredLength x/2)) ≤
          ENNReal.ofReal k*ENNReal.ofReal (K*Real.exp (-squaredLength (M *ᵥ x)/2)) +
            ENNReal.ofReal l*ENNReal.ofReal (K*Real.exp (-squaredLength (N *ᵥ x)/2))
        rw [← ENNReal.ofReal_mul (by positivity : 0≤2*(1+s/16)),
          ← ENNReal.ofReal_mul hk.le, ← ENNReal.ofReal_mul hl.le,
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        apply ENNReal.ofReal_le_ofReal
        have hh := mul_le_mul_of_nonneg_left H hK
        linarith
      _ = ENNReal.ofReal k*(∫⁻ x in S, gaussianDensity (M *ᵥ x)) +
          ENNReal.ofReal l*(∫⁻ x in S, gaussianDensity (N *ᵥ x)) := by
        have hM : Measurable (fun x : Coordinates d => gaussianDensity (M *ᵥ x)) :=
          measurable_gaussianDensity.comp (M.mulVecLin.continuous_of_finiteDimensional.measurable)
        have hN : Measurable (fun x : Coordinates d => gaussianDensity (N *ᵥ x)) :=
          measurable_gaussianDensity.comp (N.mulVecLin.continuous_of_finiteDimensional.measurable)
        let f : Coordinates d → ℝ≥0∞ := fun x => ENNReal.ofReal k*gaussianDensity (M *ᵥ x)
        let g : Coordinates d → ℝ≥0∞ := fun x => ENNReal.ofReal l*gaussianDensity (N *ᵥ x)
        have hf : Measurable f := measurable_const.mul hM
        have he := lintegral_add_left (μ := (volume : Measure (Coordinates d)).restrict S) hf g
        change (∫⁻ x in S, f x+g x) = _
        rw [he]
        dsimp [f,g]
        rw [lintegral_const_mul _ hM,lintegral_const_mul _ hN]
      _ ≤ _ := add_le_add hma' hmb'
  have hfA := (bodyAt_finite (show r2 ≤ 1/2 by linarith)
    (fun x => (quarter_band_standard ha x).1)).2
  have hfB := (bodyAt_finite (show r2 ≤ 1/2 by linarith)
    (fun x => (quarter_band_standard hb x).1)).2
  have g := ENNReal.toReal_mono (ENNReal.add_ne_top.mpr ⟨hfA,hfB⟩) hmass
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by positivity),ENNReal.toReal_add hfA hfB] at g
  have h := mul_le_mul_of_nonneg_right (show 2*(1+θ^2/16)≤2*(1+s/16) by linarith)
    (ENNReal.toReal_nonneg (a:=gaussian S))
  exact h.trans g

end Cubature
