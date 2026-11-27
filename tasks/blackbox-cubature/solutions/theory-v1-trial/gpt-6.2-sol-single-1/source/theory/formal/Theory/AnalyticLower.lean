import Theory.PolynomialGrid
import Theory.RadialQuantile
import Theory.UnitGrid

/-! A fixed-query precision obstruction within the uniform analytic chart class. -/

open Set MeasureTheory
open scoped unitInterval

namespace Cubature

noncomputable section

/-- Constant radii are admitted for every order of the chart remainder. -/
theorem const_polynomial_remainder {p r : ℕ} (hr : 0 < r) {A : ℝ} (hA : 0 ≤ A) (t : ℝ) :
    ChartPolynomialRemainder (p := p) r A (fun _ => t) := by
  classical
  intro f x
  let e : Fin p → Fin r := fun _ => ⟨0, hr⟩
  let coeff (i : Fin p → Fin r) := if i = e then t else 0
  refine ⟨coeff, fun y δ hδ _ => ?_⟩
  have hP : chartPolynomial x coeff y = t := by
    unfold chartPolynomial
    rw [Finset.sum_eq_single_of_mem e (Finset.mem_univ e)]
    · simp [coeff,e]
    · intro i _ hi
      simp [coeff,hi]
  rw [hP,sub_self,abs_zero]
  positivity

private def bitPhase (x : Bool) : I := if x then 1 else 0

private def ballLevel (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (b : Bool) : ℝ :=
  fromMass (Nat.succ_pos p) ⟨gridTest j (bitPhase b), gridTest_mem hJ j (bitPhase b)⟩

private theorem ballLevel_mem (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (b : Bool) :
    ballLevel p hJ j b ∈ Icc 1 2 := fromMass_mem (Nat.succ_pos p) _

private theorem ballLevel_cdf (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (b : Bool) :
    massCDF (p+1) (ballLevel p hJ j b) =
      (massCDF (p+1) 1 + cdfSpan (p+1) * ((j.val:ℝ)/(J:ℝ))) +
        (cdfSpan (p+1)/(J:ℝ)) * bitvalue b := by
  unfold ballLevel
  rw [fromMass_cdf]
  cases b <;> simp [bitPhase,gridTest,gridEdge,bitvalue]
  ring

private theorem ballLevel_order (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (b : Bool) :
    ballLevel p hJ j false ≤ ballLevel p hJ j b ∧
      ballLevel p hJ j b ≤ ballLevel p hJ j true := by
  have h : ballLevel p hJ j false ≤ ballLevel p hJ j true := by
    apply fromMass_monotone (Nat.succ_pos p)
    change gridTest j (bitPhase false) ≤ gridTest j (bitPhase true)
    simp [bitPhase,gridTest]
  cases b
  · exact ⟨le_rfl,h⟩
  · exact ⟨h,le_rfl⟩

private theorem ballLevel_sep (p : ℕ) {J : ℕ} (hJ : 0 < J) {i j : Fin J} (h : i < j) :
    ballLevel p hJ i true ≤ ballLevel p hJ j false := by
  apply fromMass_monotone (Nat.succ_pos p)
  change gridTest i (bitPhase true) ≤ gridTest j (bitPhase false)
  have hh : (i.val : ℝ)+1 ≤ j.val := by exact_mod_cast Nat.succ_le_of_lt h
  simpa [bitPhase,gridTest,gridEdge,add_div] using
    (div_le_div_of_nonneg_right hh (Nat.cast_nonneg J))

private def ballWindow (p : ℕ) {J : ℕ} (hJ : 0 < J) (t : ℝ) (j : Fin J) : Prop :=
  t ∈ Ico (ballLevel p hJ j false) (ballLevel p hJ j true)

private def ballSite (p : ℕ) {J : ℕ} (hJ : 0 < J) (z : GaussianSpace (p+1)) : Option (Fin J × Unit) := by
  classical
  exact if h : ∃ j, ballWindow p hJ ‖z‖ j then some (h.choose, ()) else none

private theorem ballSite_of (p : ℕ) {J : ℕ} (hJ : 0 < J) (z : GaussianSpace (p+1))
    (j : Fin J) (hj : ballWindow p hJ ‖z‖ j) : ballSite p hJ z = some (j,()) := by
  have he : ∃ j, ballWindow p hJ ‖z‖ j := ⟨j,hj⟩
  have hc := he.choose_spec
  have eq : he.choose = j := by
    rcases lt_trichotomy he.choose j with h | h | h
    · exact False.elim (not_lt_of_ge ((ballLevel_sep p hJ h).trans hj.1) hc.2)
    · exact h
    · exact False.elim (not_lt_of_ge ((ballLevel_sep p hJ h).trans hc.1) hj.2)
  rw [ballSite, dite_eq_left he, eq]

private theorem ballFlip (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (i : Unit)
    (bits : Unit → Bool) (z : GaussianSpace (p+1)) (hs : ballSite p hJ z ≠ some (j,i)) :
    starOracle (fun _ => ballLevel p hJ j (flip i bits ())) z =
      starOracle (fun _ => ballLevel p hJ j (bits ())) z := by
  have hNo : ¬ballWindow p hJ ‖z‖ j := by
    intro h
    cases i
    exact hs (ballSite_of p hJ z j h)
  unfold ballWindow at hNo
  have hOld := ballLevel_order p hJ j (bits ())
  have hNew := ballLevel_order p hJ j (flip i bits ())
  unfold starOracle
  apply decide_eq_decide.mpr
  change (‖z‖ < ballLevel p hJ j (flip i bits ())) ↔ (‖z‖ < ballLevel p hJ j (bits ()))
  by_cases h : ‖z‖ < ballLevel p hJ j false
  · exact ⟨fun _ => h.trans_le hOld.1, fun _ => h.trans_le hNew.1⟩
  · have hh : ballLevel p hJ j true ≤ ‖z‖ :=
      le_of_not_gt (fun hn => hNo ⟨not_lt.mp h,hn⟩)
    exact ⟨fun he => (not_lt_of_ge (hNew.2.trans hh) he).elim,
      fun he => (not_lt_of_ge (hOld.2.trans hh) he).elim⟩

private def ballBase (p : ℕ) {J : ℕ} (j : Fin J) : ℝ :=
  massCDF (p+1) 1 + cdfSpan (p+1) * ((j.val:ℝ)/(J:ℝ))

private theorem ballMass (p : ℕ) {J : ℕ} (hJ : 0 < J) (j : Fin J) (bits : Unit → Bool) :
    gaussianStarMass (p := p) (fun _ => ballLevel p hJ j (bits ())) =
      ballBase p j + ∑ i : Unit, (cdfSpan (p+1)/(J:ℝ)) * bitvalue (bits i) := by
  have := angularLaw_isProbability (Nat.succ_pos p)
  rw [gaussianStarMass_eq measurable_const, integral_const]
  simp only [show (angularLaw (p+1)).real univ = 1 by simp, one_smul]
  rw [show radialPrimitive (radialDensity (p+1)) (ballLevel p hJ j (bits ())) =
    massCDF (p+1) (ballLevel p hJ j (bits ())) from rfl]
  simp [ballLevel_cdf,ballBase]

theorem nonadaptive_analytic_prior {p J N : ℕ} (hJ : 0 < J) (hN : 2*N ≤ J)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → NonadaptiveRule (GaussianSpace (p+1)) N)
    (hI : ∀ t ∈ Icc (1:ℝ) 2, Integrable (fun ω =>
      ((alg ω).estimate (starOracle (fun _ => t)) - gaussianStarMass (p := p) (fun _ => t))^2) μ) :
    ∃ t ∈ Icc (1:ℝ) 2, (cdfSpan (p+1)/(J:ℝ))^2/8 ≤
      ∫ ω, ((alg ω).estimate (starOracle (fun _ => t)) - gaussianStarMass (p := p) (fun _ => t))^2 ∂μ := by
  let coeff : Fin J → Unit → ℝ := fun _ _ => cdfSpan (p+1)/(J:ℝ)
  have hcard : 2*N ≤ Fintype.card (Fin J) * Fintype.card Unit := by
    simpa [Fintype.card_fin] using hN
  obtain ⟨j,bits,h⟩ := nonadaptive_prior_lower_bound μ alg (ballSite p hJ)
    (fun j b => starOracle (fun _ => ballLevel p hJ j (b ()))) (ballBase p) coeff
    (div_nonneg (cdfSpan_pos (Nat.succ_pos p)).le (Nat.cast_nonneg _))
    (fun _ _ => le_rfl) (ballFlip p hJ) ⟨⟨0,hJ⟩⟩ ⟨()⟩ hcard (fun j b => by
      change Integrable (fun ω =>
        ((alg ω).estimate (starOracle (fun _ => ballLevel p hJ j (b ()))) -
          (ballBase p j + ∑ i : Unit, (cdfSpan (p+1)/(J:ℝ))*bitvalue (b i)))^2) μ
      simp_rw [← ballMass p hJ]
      exact hI _ (ballLevel_mem p hJ j (b ())))
  refine ⟨ballLevel p hJ j (bits ()),ballLevel_mem p hJ j (bits ()),?_⟩
  rw [ballMass p hJ j bits]
  simpa [coeff] using h

end

end Cubature
