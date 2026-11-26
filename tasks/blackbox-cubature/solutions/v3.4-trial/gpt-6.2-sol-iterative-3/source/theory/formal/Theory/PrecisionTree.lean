import Theory.PrecisionMeasure

open scoped BigOperators ENNReal
open Matrix MeasureTheory

namespace Cubature

variable {d n m : ℕ}

private theorem compare_average (a b r : ℝ) (h : decide (a ≤ r) = decide (b ≤ r)) :
    decide ((a+b)/2 ≤ r) = decide (a ≤ r) := by
  by_cases ha : a≤r <;> by_cases hb : b≤r
  · simp [ha, show (a+b)/2≤r by linarith]
  · simp [ha,hb] at h
  · simp [ha,hb] at h
  · simp [ha, show ¬(a+b)/2≤r by linarith [lt_of_not_ge ha,lt_of_not_ge hb]]

theorem identical_bit_transcripts {A B : Coefficients d} (r2 : ℝ) (P : PointQueryPlan d Bool n)
    (hs : bitTranscript (memberAt r2 A) P = bitTranscript (memberAt r2 B) P) :
    P.run (memberAt r2 (midpointMatrix A B)) = P.run (memberAt r2 A) ∧
      P.run (memberAt r2 A) = P.run (memberAt r2 B) := by
  induction P with
  | answer => exact ⟨rfl,rfl⟩
  | @ask k z K ih =>
    have first : memberAt r2 A z = memberAt r2 B z := by
      simpa only [bitTranscript,Fin.cons_zero] using congrFun hs (0 : Fin (k+1))
    have hm : memberAt r2 (midpointMatrix A B) z = memberAt r2 A z := by
      simpa only [memberAt,midpoint_quadratic] using
        compare_average (quadratic A z) (quadratic B z) r2 first
    have tail : bitTranscript (memberAt r2 A) (K (memberAt r2 A z)) =
        bitTranscript (memberAt r2 B) (K (memberAt r2 A z)) := by
      funext i
      simpa only [bitTranscript,Fin.cons_succ,← first] using congrFun hs i.succ
    obtain ⟨h₁,h₂⟩ := ih (memberAt r2 A z) tail
    simp only [PointQueryPlan.run,hm,← first]
    exact ⟨h₁,h₂⟩

noncomputable def gridMatrix (g : UpperIndices d → Fin m) : Coefficients d :=
  1 + symmetricLift d (fun i => matrixGridPitch d m * ((g i : ℕ) : ℝ))

theorem gridMatrix_hermitian (g : UpperIndices d → Fin m) : (gridMatrix g).IsHermitian :=
  Matrix.isHermitian_one.add (symmetricLift_hermitian _)

theorem gridMatrix_quarter_bounds (hd : 0 < d) (hm : 0 < m)
    (g : UpperIndices d → Fin m) (x : Coordinates d) :
    (3/4:ℝ)*squaredLength x ≤ quadratic (gridMatrix g) x ∧
      quadratic (gridMatrix g) x ≤ (5/4:ℝ)*squaredLength x := by
  have hd' : (0:ℝ)<d := by exact_mod_cast hd
  have hm' : (0:ℝ)<m := by exact_mod_cast hm
  let η := matrixGridPitch d m
  let D := symmetricLift d (fun i => η * ((g i : ℕ) : ℝ))
  have hη : 0 ≤ η := by dsimp [η,matrixGridPitch]; positivity
  have v (p : UpperIndices d) : |η*((g p : ℕ):ℝ)| ≤ 1/(4*(d:ℝ)) := by
    have h : ((g p : ℕ):ℝ) ≤ m := by exact_mod_cast (g p).isLt.le
    rw [abs_of_nonneg (by positivity)]
    apply (mul_le_mul_of_nonneg_left h hη).trans_eq
    dsimp [η,matrixGridPitch]
    field_simp
  have base (i j : Fin d) : |D i j| ≤ 1/(4*(d:ℝ)) := by
    change |symmetricEntries d (fun i => η*((g i : ℕ):ℝ)) i j| ≤ _
    unfold symmetricEntries
    split <;> apply v
  have H := quadratic_entry_bound D x (by positivity) base
  have hh : (d:ℝ)*(1/(4*(d:ℝ))) = 1/4 := by field_simp
  change |quadratic D x| ≤ (d:ℝ)*(1/(4*(d:ℝ)))*squaredLength x at H
  rw [hh] at H
  have B := abs_le.mp H
  change _ ≤ quadratic (1+D) x ∧ quadratic (1+D) x ≤ _
  rw [quadratic_add,quadratic_one]
  constructor <;> linarith [B.1,B.2,square_sum_nonneg x]

theorem gridMatrix_bounds (hd : 0 < d) (hm : 0 < m)
    (g : UpperIndices d → Fin m) (x : Coordinates d) :
    (1/2:ℝ)*squaredLength x ≤ quadratic (gridMatrix g) x ∧
      quadratic (gridMatrix g) x ≤ (3/2:ℝ)*squaredLength x := by
  have h := gridMatrix_quarter_bounds hd hm g x
  constructor <;> linarith [h.1,h.2,square_sum_nonneg x]

theorem gridMatrix_separation {g h : UpperIndices d → Fin m} (hne : g ≠ h)
    (hη : 0 < matrixGridPitch d m) :
    ∃ i j, matrixGridPitch d m ≤ |gridMatrix g i j-gridMatrix h i j| := by
  obtain ⟨⟨⟨i,j⟩,hij⟩,hne⟩ := Function.ne_iff.mp hne
  let p : UpperIndices d := ⟨(i,j),hij⟩
  let η := matrixGridPitch d m
  have u : (g p : ℕ) ≠ (h p : ℕ) := fun e => hne (Fin.ext e)
  have q : (1:ℝ) ≤ |((g p : ℕ):ℝ)-((h p : ℕ):ℝ)| := by
    rcases lt_or_gt_of_ne u with H|H
    · have ht : ((g p : ℕ):ℝ)+1≤((h p : ℕ):ℝ) := by exact_mod_cast H
      rw [abs_of_nonpos (by linarith)]
      linarith
    · have ht : ((h p : ℕ):ℝ)+1≤((g p : ℕ):ℝ) := by exact_mod_cast H
      rw [abs_of_nonneg (by linarith)]
      linarith
  refine ⟨i,j,?_⟩
  change η ≤ |(1 : Coefficients d) i j + symmetricEntries d
      (fun k => η*((g k : ℕ):ℝ)) i j -
    ((1 : Coefficients d) i j + symmetricEntries d (fun k => η*((h k : ℕ):ℝ)) i j)|
  have e : i ≤ j := hij
  simp only [symmetricEntries,e,↓reduceDIte]
  rw [show (1 : Coefficients d) i j + η*((g p : ℕ):ℝ) -
      ((1 : Coefficients d) i j + η*((h p : ℕ):ℝ)) =
      η*(((g p : ℕ):ℝ)-((h p : ℕ):ℝ)) by ring,abs_mul,abs_of_pos hη]
  simpa using mul_le_mul_of_nonneg_left q hη.le

private theorem shared_answer_vs_gap {θ t s v : ℝ}
    (hθ : 0 < θ) (ht : 0 < t) (hg : s ≤ (1-θ^2/5)*t) :
    θ^2/20*s ≤ |v-s| ∨ θ^2/20*t ≤ |v-t| := by
  by_contra! H
  have a := (abs_lt.mp H.1).2
  have b := (abs_lt.mp H.2).1
  have hT : (1+θ^2/20)*s ≤ ((1+θ^2/20)*(1-θ^2/5))*t := by
    have hh := mul_le_mul_of_nonneg_left hg (by positivity : 0 ≤ 1+θ^2/20)
    linarith [show (1+θ^2/20)*((1-θ^2/5)*t)=((1+θ^2/20)*(1-θ^2/5))*t by ring]
  have hf : (1+θ^2/20)*(1-θ^2/5) < 1-θ^2/20 := by
    nlinarith [sq_pos_of_pos hθ,sq_nonneg (θ^2)]
  have hp := mul_lt_mul_of_pos_right hf ht
  linarith

theorem grid_precision_lower (hd : 0 < d) (hm : 2 ≤ m)
    (P : PointQueryPlan d Bool n) (hN : 2^n < m^(Nat.choose (d+1) 2))
    (r2 : ℝ) (hr : 0 < r2) (hr' : r2 ≤ (precisionGap d m)^2/100) :
    ∃ A : Coefficients d, A.IsHermitian ∧
      (∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic A x ∧ quadratic A x ≤ (3/2:ℝ)*squaredLength x) ∧
      let truth := (gaussian (bodyAt r2 A)).toReal
      0 < truth ∧ (precisionGap d m)^2/20*truth ≤ |P.run (memberAt r2 A)-truth| := by
  let observe : (UpperIndices d → Fin m) → (Fin n → Bool) :=
    fun g => bitTranscript (memberAt r2 (gridMatrix g)) P
  have hninj : ¬Function.Injective observe := by
    intro H
    have h := Fintype.card_le_of_injective _ H
    have h' : m^((d+1).choose 2) ≤ 2^n := by simpa [Fintype.card_fun,upperIndices_card] using h
    exact hN.not_ge h'
  obtain ⟨g,h,eqout,diff⟩ := Function.not_injective_iff.mp hninj
  let θ := precisionGap d m
  let A := gridMatrix g
  let B := gridMatrix h
  let C := midpointMatrix A B
  have ha := gridMatrix_bounds hd (by omega : 0<m) g
  have hb := gridMatrix_bounds hd (by omega : 0<m) h
  have hA := gridMatrix_hermitian g
  have hB := gridMatrix_hermitian h
  have hc := midpoint_bounds ha hb
  have hC := midpoint_hermitian hA hB
  have hd' : (0:ℝ)<d := by exact_mod_cast hd
  have hm' : (0:ℝ)<m := by exact_mod_cast (by omega : 0<m)
  have hη : 0 < matrixGridPitch d m := by unfold matrixGridPitch; positivity
  have H := determinant_midpoint_gap A B hd hA hB ha hb (matrixGridPitch d m) hη
    (gridMatrix_separation diff hη)
  rcases H with ⟨hθ,hθ',hh⟩
  have hsz : r2 ≤ 1/2 := by
    have hp : θ^2 ≤ 1 := (sq_le_sq₀ hθ.le zero_le_one).mpr hθ' |>.trans_eq (by norm_num)
    linarith
  have pos {M : Coefficients d}
      (hM : ∀ x, (1/2:ℝ)*squaredLength x ≤ quadratic M x ∧ quadratic M x ≤ (3/2:ℝ)*squaredLength x) :
      0 < (gaussian (bodyAt r2 M)).toReal :=
    ENNReal.toReal_pos (bodyAt_positive hr (by linarith) (fun x => (hM x).2)).ne'
      (bodyAt_finite hsz (fun x => (hM x).1)).2
  obtain ⟨o₁,o₂⟩ := identical_bit_transcripts r2 P eqout
  have oc : P.run (memberAt r2 C) = P.run (memberAt r2 A) := o₁
  have ob : P.run (memberAt r2 A) = P.run (memberAt r2 B) := o₂
  rcases hh with HA | HB
  · have gap := gaussian_gap_from_determinant hθ.le hθ' hr hr' hA hC ha hc HA
    rcases shared_answer_vs_gap (v:=P.run (memberAt r2 A)) hθ (pos ha) gap with e|e
    · refine ⟨C,hC,hc,pos hc,?_⟩
      change θ^2/20*(gaussian (bodyAt r2 C)).toReal ≤
        |P.run (memberAt r2 C)-(gaussian (bodyAt r2 C)).toReal|
      rw [oc]
      exact e
    · exact ⟨A,hA,ha,pos ha,e⟩
  · have gap := gaussian_gap_from_determinant hθ.le hθ' hr hr' hB hC hb hc HB
    rcases shared_answer_vs_gap (v:=P.run (memberAt r2 A)) hθ (pos hb) gap with e|e
    · refine ⟨C,hC,hc,pos hc,?_⟩
      change θ^2/20*(gaussian (bodyAt r2 C)).toReal ≤
        |P.run (memberAt r2 C)-(gaussian (bodyAt r2 C)).toReal|
      rw [oc]
      exact e
    · refine ⟨B,hB,hb,pos hb,?_⟩
      change θ^2/20*(gaussian (bodyAt r2 B)).toReal ≤
        |P.run (memberAt r2 B)-(gaussian (bodyAt r2 B)).toReal|
      rw [← ob]
      exact e

end Cubature
