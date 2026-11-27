import Theory.Reconstruction
import Theory.Nonadaptive
import Theory.Adaptive
import Theory.Stratified

open MeasureTheory Set ProbabilityTheory
open scoped BigOperators NNReal

namespace Cubature

theorem ellipsoid_mass_enclosure {d : ℕ}
    (oracle : Vec d → Bool) (anchor : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d)
    (steps : ℕ) (lo hi radius : ℝ)
    (hlo : 0 < lo) (hhi : lo ≤ hi) (hr : 0 ≤ radius)
    (hSym : ∀ i j, A i j = A j i)
    (hPSD : ∀ x, 0 ≤ quadPart A x)
    (hBody : body anchor A b ⊆ shiftedBox anchor radius)
    (hRoots : ∀ i j, positiveInverse A b (sectionDirection i j) ∈ Icc lo hi ∧
      negativeInverse A b (sectionDirection i j) ∈ Icc lo hi)
    (hOracle : ∀ z, (quadratic A b (z-anchor) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-anchor) → oracle z = false))
    (hAllowed : ∀ i, |anchor i| + lo⁻¹ ≤ 40)
    (hsmall : valueAccuracy d steps lo hi radius < 1/2) :
    let history := queryHistory oracle anchor steps lo hi
    let out := massInterval oracle anchor steps lo hi radius
    let truth := (gaussian d).real (body anchor A b)
    let error := valueAccuracy d steps lo hi radius
    let dilation := (1+2*error)/(1-2*error)
    history.length ≤ 2*d*d*steps ∧
    (∀ z ∈ history, ∀ i, |z i| ≤ 40) ∧
    out.1 ≤ truth ∧ truth ≤ out.2 ∧ out.2 ≤ dilationFactor anchor dilation * out.1 := by
  dsimp only
  obtain ⟨hL, hU, hC, hs⟩ := reconstruction_sandwich oracle anchor A b
    steps lo hi radius hlo hhi hr hSym hPSD hBody hRoots hOracle hsmall
  exact ⟨queryHistory_count oracle anchor steps lo hi,
    queryHistory_safe oracle anchor A b steps lo hi hlo hPSD hRoots hOracle hAllowed,
    measureReal_mono hL, measureReal_mono hU,
    (measureReal_mono hC).trans (gaussian_mass_contract_le hs (measurable_proxySet ..))⟩

theorem gaussian_balls_nonadaptive_precision {d : ℕ} (hd : 0 < d)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (m N : ℕ) (hm : 0 < m)
    (queries : Ω → List (Vec d)) (answer : Ω → List Bool → ℝ)
    (hN : ∀ ω, (queries ω).length ≤ N)
    (hMeas : ∀ r ∈ Icc (ballMinRadius d) 1,
      MeasurableSet (ballSuccess r (1 / (8 * (m : ℝ))) queries answer))
    (hSuccess : ∀ r ∈ Icc (ballMinRadius d) 1,
      (3 / 4 : ℝ) ≤ P.real (ballSuccess r (1 / (8 * (m : ℝ))) queries answer)) :
    (3 / 4 : ℝ) * m ≤ (N : ℝ) + 1 := by
  obtain ⟨r, hr, hpack, hmon⟩ := ballMass_packing hd hm
  have hpos (i : Fin m) : 0 ≤ r i :=
    (show 0 ≤ ballMinRadius d from div_nonneg (roundMass_one_pos d).le (by norm_num)).trans (hr i).1
  let eps : ℝ := 1 / (8 * (m : ℝ))
  calc
    (3 / 4 : ℝ) * m = ∑ _ : Fin m, (3 / 4 : ℝ) := by simp; ring
    _ ≤ ∑ i : Fin m, P.real (ballSuccess (r i) eps queries answer) :=
      Finset.sum_le_sum fun i _ => hSuccess (r i) (hr i)
    _ ≤ (N : ℝ) + 1 := nonadaptive_ball_average P r hpos hmon eps
      (packing_is_separated hm r hpack) queries answer hN (fun i => hMeas (r i) (hr i))

theorem gaussian_rare_balls_adaptive {d : ℕ} (hd : 0 < d)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (m N : ℕ) (hm : 0 < m)
    (tree : Ω → MembershipTree d) (hN : ∀ ω, (tree ω).depth ≤ N)
    (hMeas : ∀ (a : Vec d) (r : ℝ), (∀ i, a i ∈ Icc (0 : ℝ) 1) →
      r ∈ Icc (rareInner d m) (rareOuter m) →
      MeasurableSet (displacedBallSuccess tree a r (1/3)))
    (hSuccess : ∀ (a : Vec d) (r : ℝ), (∀ i, a i ∈ Icc (0 : ℝ) 1) →
      r ∈ Icc (rareInner d m) (rareOuter m) →
      (3/4 : ℝ) ≤ P.real (displacedBallSuccess tree a r (1/3))) :
    (m : ℝ)^d ≤ 2*(N : ℝ) ∧
    0 < rareLowerMass d m ∧
    ∀ (a : Vec d) (r : ℝ), (∀ i, a i ∈ Icc (0 : ℝ) 1) →
      r ∈ Icc (rareInner d m) (rareOuter m) →
      rareLowerMass d m ≤ shiftedBallMass a r ∧
      shiftedBallMass a r ≤ (2*rareOuter m)^d := by
  have hin : rareInner d m ∈ Icc (rareInner d m) (rareOuter m) :=
    ⟨le_rfl, rareInner_le_outer hm⟩
  have hout : rareOuter m ∈ Icc (rareInner d m) (rareOuter m) :=
    ⟨rareInner_le_outer hm, le_rfl⟩
  have H := adaptive_rare_average P hm hd tree hN
    (fun j => ⟨hMeas _ _ (rareCenter_mem hm j) hin, hMeas _ _ (rareCenter_mem hm j) hout⟩)
  have Hsmall : ((3/2 : ℝ) * (m : ℝ)^d) ≤ (m : ℝ)^d + N := by
    calc
      _ = ∑ _ : Fin d → Fin m, ((3/4 : ℝ) + 3/4) := by
        simp [Nat.cast_pow]
        ring
      _ ≤ ∑ j : Fin d → Fin m,
        (P.real (displacedBallSuccess tree (rareCenter j) (rareInner d m) (1/3)) +
         P.real (displacedBallSuccess tree (rareCenter j) (rareOuter m) (1/3))) := by
        exact Finset.sum_le_sum fun j _ =>
          add_le_add (hSuccess _ _ (rareCenter_mem hm j) hin) (hSuccess _ _ (rareCenter_mem hm j) hout)
      _ ≤ _ := H
  have hf := pdf_two_pos
  have hi := rareInner_pos (d := d) hm
  refine ⟨by linarith only [Hsmall], by unfold rareLowerMass; positivity, ?_⟩
  intro a r ha hr
  have hp : 0 ≤ r := hi.le.trans hr.1
  constructor
  · refine (pow_le_pow_left₀ (by positivity) ?_ d).trans
      (shiftedBallMass_lower a r ha hp (hr.2.trans (rareOuter_le_one hm)))
    exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hr.1 (by positivity)) (by positivity)
  · exact (shiftedBallMass_upper a r hp).trans (pow_le_pow_left₀ (by positivity)
      (mul_le_mul_of_nonneg_left hr.2 (by positivity)) d)

theorem two_axis_preintegration_stratified {m k : ℕ} (hk : 0 < k)
    (a lam : Vec 2) (hlam : ∀ i, 0 < lam i)
    (g q : Vec m → ℝ) (M : ℝ) (Lg Lq : ℝ≥0)
    (hg : LipschitzWith Lg g) (hq : LipschitzWith Lq q) (hM : ∀ x, |g x| ≤ M)
    (μ : CellIndex m k → Measure (Vec m)) [∀ i, IsProbabilityMeasure (μ i)]
    (centers : CellIndex m k → Vec m)
    (hCell : ∀ i, ∀ᵐ x ∂μ i, dist x (centers i) ≤ 1/(2*(k : ℝ))) :
    let H := preintegrated a lam g q
    let Lip := afterTwoAxesLip lam M Lg Lq
    (∀ x, H x =
      ∫ u : Vec 2, {z : Vec 2 | ∑ i, lam i * (z i-a i)^2 ≤ q x}.indicator
        (fun _ => g x) u ∂gaussian 2) ∧
    (∀ x y, |H x-H y| ≤ Lip * dist x y) ∧
    (∫ sample, stratifiedAverage H sample ∂Measure.pi μ) =
      (∑ i, ∫ x, H x ∂μ i)/(k : ℝ)^m ∧
    variance (stratifiedAverage H) (Measure.pi μ) ≤ Lip^2/(4*(k : ℝ)^2*(k : ℝ)^m) := by
  dsimp only
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  have hL0 : 0 ≤ afterTwoAxesLip lam M Lg Lq := by unfold afterTwoAxesLip; positivity
  have hbound (x : Vec m) : |preintegrated a lam g q x| ≤ M := by
    have hP0 : 0 ≤ ellipseProbability a lam (q x) := measureReal_nonneg
    have hP1 : ellipseProbability a lam (q x) ≤ 1 := measureReal_le_one
    unfold preintegrated
    rw [abs_mul, abs_of_nonneg hP0]
    exact (mul_le_of_le_one_right (abs_nonneg _) hP1).trans (hM x)
  have hLip := preintegrated_dist a lam hlam g q M Lg Lq hg hq hM
  exact ⟨preintegrated_eq_integral a lam g q, hLip,
    stratified_moments hk _ (preintegrated_continuous a lam hlam g q M Lg Lq hg hq hM).measurable
      M _ hbound hL0 hLip μ centers hCell⟩

end Cubature
