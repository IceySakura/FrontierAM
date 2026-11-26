import Theory.FixedMass

open scoped ENNReal
open Matrix MeasureTheory

namespace Cubature

variable {d n m : ℕ}

private theorem same_answer_large_endpoint {θ s t a : ℝ}
    (hθ : 0 < θ) (hθ' : θ ≤ 1) (hs : 0 < s) (hg : (1+θ^2/16)*s ≤ t) :
    θ^2/40*s ≤ |a-s| ∨ θ^2/40*t ≤ |a-t| := by
  have hsq : θ^2 ≤ 1 := (sq_le_sq₀ hθ.le zero_le_one).mpr hθ' |>.trans_eq (by norm_num)
  have hpow : (θ^2)^2 ≤ θ^2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq (sq_nonneg θ)]
  have hgain : 1+θ^2/40 < (1-θ^2/40)*(1+θ^2/16) := by
    nlinarith [sq_pos_of_pos hθ]
  have hf : 0 ≤ 1-θ^2/40 := by linarith
  have hp := mul_lt_mul_of_pos_right hgain hs
  have hq := mul_le_mul_of_nonneg_left hg hf
  by_contra! hh
  have h1 := (abs_lt.mp hh.1).2
  have h2 := (abs_lt.mp hh.2).1
  linarith [show ((1-θ^2/40)*(1+θ^2/16))*s=(1-θ^2/40)*((1+θ^2/16)*s) by ring]

theorem fixed_grid_precision (hd : 0 < d) (hm : 2 ≤ m)
    (P : PointQueryPlan d Bool n) (hN : 2^n < m^(Nat.choose (d+1) 2))
    (r2 : ℝ) (hr : 0 < r2) (hr' : r2 ≤ 1/100) :
    ∃ A : Coefficients d, A.IsHermitian ∧
      (∀ x, (3/4:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (5/4:ℝ)*squaredLength x) ∧
      let truth := (gaussian (bodyAt r2 A)).toReal
      0 < truth ∧ (precisionGap d m)^2/40*truth ≤ |P.run (memberAt r2 A)-truth| := by
  let obs : (UpperIndices d → Fin m) → (Fin n → Bool) :=
    fun g => bitTranscript (memberAt r2 (gridMatrix g)) P
  have hninj : ¬Function.Injective obs := by
    intro H
    have h := Fintype.card_le_of_injective _ H
    have e : m^((d+1).choose 2) ≤ 2^n := by simpa [Fintype.card_fun,upperIndices_card] using h
    exact hN.not_ge e
  obtain ⟨g,h,same,ne⟩ := Function.not_injective_iff.mp hninj
  let θ := precisionGap d m
  let A := gridMatrix g
  let B := gridMatrix h
  let C := midpointMatrix A B
  have ha : InQuarterBand A := gridMatrix_quarter_bounds hd (by omega : 0<m) g
  have hb : InQuarterBand B := gridMatrix_quarter_bounds hd (by omega : 0<m) h
  have hc : InQuarterBand C := midpoint_quarter ha hb
  have hsA : A.IsHermitian := gridMatrix_hermitian g
  have hsB : B.IsHermitian := gridMatrix_hermitian h
  have hsC : C.IsHermitian := midpoint_hermitian hsA hsB
  have hd' : (0:ℝ)<d := by exact_mod_cast hd
  have hm' : (0:ℝ)<m := by exact_mod_cast (by omega : 0<m)
  have hη : 0 < matrixGridPitch d m := by unfold matrixGridPitch; positivity
  have hθ : 0 < θ := by dsimp [θ,precisionGap]; positivity
  obtain ⟨G⟩ : Nonempty (PairGeometry A B θ) := midpoint_geometry A B hd hsA hsB ha hb
    (matrixGridPitch d m) hη (gridMatrix_separation ne hη)
  have ht : θ ≤ 1 := G.hθ.trans (G.hL₁.trans (by norm_num))
  let I (S : Coefficients d) := (gaussian (bodyAt r2 S)).toReal
  have pos (S : Coefficients d) (hS : InQuarterBand S) : 0 < I S :=
    ENNReal.toReal_pos (bodyAt_positive hr (by linarith)
      (fun x => (quarter_band_standard hS x).2)).ne'
      (bodyAt_finite (by linarith) (fun x => (quarter_band_standard hS x).1)).2
  obtain ⟨a1,a2⟩ := identical_bit_transcripts r2 P same
  have ans1 : P.run (memberAt r2 C) = P.run (memberAt r2 A) := a1
  have ans2 : P.run (memberAt r2 C) = P.run (memberAt r2 B) := a1.trans a2
  have mass : 2*(1+θ^2/16)*I C ≤ I A+I B := G.gaussian_mass_pair hθ ha hb hr hr'
  have endpoint : ∃ S : Coefficients d, S.IsHermitian ∧ InQuarterBand S ∧
      (1+θ^2/16)*I C ≤ I S ∧ P.run (memberAt r2 C) = P.run (memberAt r2 S) := by
    by_cases H : (1+θ^2/16)*I C ≤ I A
    · exact ⟨A,hsA,ha,H,ans1⟩
    · exact ⟨B,hsB,hb,by linarith [lt_of_not_ge H],ans2⟩
  obtain ⟨S,hS,hBand,hG,hRun⟩ := endpoint
  rcases same_answer_large_endpoint (a:=P.run (memberAt r2 C)) hθ ht (pos C hc) hG with E | E
  · exact ⟨C,hsC,hc,pos C hc,E⟩
  · refine ⟨S,hS,hBand,pos S hBand,?_⟩
    change θ^2/40*I S ≤ |P.run (memberAt r2 S)-I S|
    rw [← hRun]
    exact E

end Cubature
