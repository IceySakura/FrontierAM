import Theory.Sections
import Theory.BallModel
import Theory.Adaptive
import Theory.Stratified

open MeasureTheory Set ProbabilityTheory
open scoped BigOperators NNReal

namespace Cubature

/-- R1: deterministic Gaussian mass enclosure for an unknown ellipsoid from
bounded-coordinate point queries. The anchor, reciprocal section bracket, and
box radius are inputs. The matrix and linear part are not given to the algorithm. -/
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
  sorry

/-- R2: at precision `1/(8*m)`, an arbitrary randomized nonadaptive algorithm
that succeeds with probability at least `3/4` on every ball of a fixed radius
range must use at least `3*m/4 - 1` membership queries. -/
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
  sorry

/-- R3: locating a rare Gaussian body can cost inverse mass, even for a
randomized adaptive algorithm and even when the body is an ordinary ball. -/
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
  sorry

/-- R4: two positive curvatures allow a nonmonotone Gaussian section to be
preintegrated, with a dimension-reduced stratified variance bound. -/
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
  sorry

end Cubature
