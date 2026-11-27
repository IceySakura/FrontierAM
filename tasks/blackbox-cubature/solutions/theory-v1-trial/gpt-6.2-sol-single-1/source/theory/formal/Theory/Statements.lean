import Theory.GaussianStar
import Theory.HardMass
import Theory.PolynomialGrid
import Theory.RadialQuantile
import Theory.BallGeometry
import Theory.EllipsoidForm

/-! Statement layer for the results in `THEORY.md`. -/

open MeasureTheory Set
open scoped ENNReal

namespace Cubature

/-- R1: adaptive, membership-only integration of a radial Gaussian jump. -/
theorem gaussian_radial_adaptive_cubature
    {p s k n : ℕ} (hp : 0 < p) (hn : 0 < n)
    {a b A : ℝ} (ha : 0 < a) (hab : a ≤ b) (hb : b ≤ 40)
    (hspan : (b - a) / 2 ≤ A) :
    ∃ alg : Samples (Direction (p+1)) n → QueryTree (GaussianSpace (p+1)) ℝ,
      (∀ ω, (alg ω).depth ≤ 2 * ((p+1) * 2 * 2^p * (2^k)^p) * (s+1) + n ∧
        (alg ω).queriesIn (coordinateBox (p+1) 40)) ∧
      ∀ R : Direction (p+1) → ℝ, Measurable R →
        (∀ u, R u ∈ Icc a b) → ChartRemainder s A R →
        let out := fun ω => (alg ω).run (starOracle R)
        let law := samplingLaw (angularLaw (p+1)) n
        Measurable out ∧
        (∫ ω, out ω ∂law) = gaussianStarMass R ∧
        (∫ ω, (out ω - gaussianStarMass R)^2 ∂law) ≤
          (2 * (A / (2 : ℝ)^(s*k)) * radialDensityBound (p+1) b)^2 / (n : ℝ) := by
  sorry

/-- R2: nonadaptive randomized lower bound for the same quadratic chart class. -/
theorem gaussian_radial_nonadaptive_lower
    {p m N : ℕ} (hp : 0 < p) (hm : 0 < m) (hN : 2 * N ≤ m^(p+2))
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → NonadaptiveRule (GaussianSpace (p+1)) N)
    (finiteRisk : ∀ R : Direction (p+1) → ℝ, Measurable R →
      (∀ u, R u ∈ Icc 1 2) → ChartRemainder 2 (hardRemainderBound p) R →
      Integrable (fun ω => ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2) μ) :
    ∃ R : Direction (p+1) → ℝ, Measurable R ∧
      (∀ u, R u ∈ Icc 1 2) ∧ ChartRemainder 2 (hardRemainderBound p) R ∧
      (m : ℝ)^p * (patchMassConstant p / (m : ℝ)^(p+2))^2 / 8 ≤
        ∫ ω, ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2 ∂μ := by
  sorry

/-- R3: matching adaptive randomized lower bound for the quadratic chart class. -/
theorem gaussian_radial_adaptive_lower
    {p m N : ℕ} (hp : 0 < p) (hm : 0 < m) (hN : 2 * N ≤ m^p)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → QueryTree (GaussianSpace (p+1)) ℝ) (budget : ∀ ω, (alg ω).depth ≤ N)
    (finiteRisk : ∀ R : Direction (p+1) → ℝ, Measurable R →
      (∀ u, R u ∈ Icc 1 2) → ChartRemainder 2 (hardRemainderBound p) R →
      Integrable (fun ω => ((alg ω).run (starOracle R) - gaussianStarMass R)^2) μ) :
    ∃ R : Direction (p+1) → ℝ, Measurable R ∧
      (∀ u, R u ∈ Icc 1 2) ∧ ChartRemainder 2 (hardRemainderBound p) R ∧
      (m : ℝ)^p * (patchMassConstant p / (m : ℝ)^(p+2))^2 / 8 ≤
        ∫ ω, ((alg ω).run (starOracle R) - gaussianStarMass R)^2 ∂μ := by
  sorry

/-- R4: nonadaptive unbiased cubature with matching precision exponent for `s = 2`. -/
theorem gaussian_radial_nonadaptive_cubature
    {p s m n : ℕ} (hp : 0 < p) (hm : 0 < m) (hn : 0 < n)
    {A : ℝ} (hA : 0 ≤ A) :
    ∃ alg : Samples (Direction (p+1)) n →
        NonadaptiveRule (GaussianSpace (p+1)) (((p+1)*2*(m+1)^p+n)*m^s),
      (∀ ω i, (alg ω).point i ∈ coordinateBox (p+1) 40) ∧
      ∀ R : Direction (p+1) → ℝ, Measurable R →
        (∀ u, R u ∈ Icc 1 2) → ChartRemainder s A R →
        let out := fun ω => (alg ω).estimate (starOracle R)
        let law := samplingLaw (angularLaw (p+1)) n
        Measurable out ∧
        (∫ ω, out ω ∂law) = gaussianStarMass R ∧
        (∫ ω, (out ω - gaussianStarMass R)^2 ∂law) ≤
          ((A+2) * radialDensityBound (p+1) 2 / (m : ℝ)^s)^2 / (n : ℝ) := by
  sorry

/-- R5: log-free adaptive membership cubature at arbitrary polynomial remainder order. -/
theorem gaussian_radial_polynomial_cubature
    {p r ell k n : ℕ} (hp : 0 < p) (hr : 0 < r) (hn : 0 < n)
    {a b A E : ℝ} (ha : 0 < a) (hab : a ≤ b) (hb : b ≤ 40) (hA : 0 ≤ A)
    (hspan : (b-a)/2 ≤ E) (hE : interpolationNorm p r * A ≤ E)
    (hell : interpolationNorm p r + 1 ≤ (2:ℝ)^ell) :
    ∃ alg : Samples (Direction (p+1)) n → QueryTree (GaussianSpace (p+1)) ℝ,
      (∀ ω, (alg ω).depth ≤ 2*((p+1)*2*(2^k*r)^p)*(r+ell)+n ∧
        (alg ω).queriesIn (coordinateBox (p+1) 40)) ∧
      ∀ R : Direction (p+1) → ℝ, Measurable R →
        (∀ u, R u ∈ Icc a b) → ChartPolynomialRemainder r A R →
        let out := fun ω => (alg ω).run (starOracle R)
        let law := samplingLaw (angularLaw (p+1)) n
        Measurable out ∧
        (∫ ω, out ω ∂law) = gaussianStarMass R ∧
        (∫ ω, (out ω - gaussianStarMass R)^2 ∂law) ≤
          ((2:ℝ)^ell * (E/(2:ℝ)^(r*k)) * radialDensityBound (p+1) b)^2 / (n:ℝ) := by
  sorry

/-- R6: a stretched-exponential error bound for geometrically bounded chart remainders. -/
theorem gaussian_radial_analytic_cubature
    {p r k : ℕ} (hp : 0 < p) (hr : 0 < r)
    {a b A L : ℝ} (ha : 0 < a) (hab : a ≤ b) (hb : b ≤ 40)
    (hspan : (b-a)/2 ≤ A) (hL : 1 ≤ L) (hk : 2*(64:ℝ)^p*L ≤ (2:ℝ)^k) :
    ∃ alg : Samples (Direction (p+1)) 1 → QueryTree (GaussianSpace (p+1)) ℝ,
      (∀ ω, (alg ω).depth ≤ 4*(p+1)*(2^k)^p*(3*p+2)*r^(p+1)+1 ∧
        (alg ω).queriesIn (coordinateBox (p+1) 40)) ∧
      ∀ R : Direction (p+1) → ℝ, Measurable R → (∀ u, R u ∈ Icc a b) →
        (∀ j : ℕ, 0 < j → ChartPolynomialRemainder j (A*L^j) R) →
        let out := fun ω => (alg ω).run (starOracle R)
        let law := samplingLaw (angularLaw (p+1)) 1
        Measurable out ∧
        (∫ ω, out ω ∂law) = gaussianStarMass R ∧
        (∫ ω, (out ω - gaussianStarMass R)^2 ∂law) ≤
          (2*A * radialDensityBound (p+1) b / (2:ℝ)^r)^2 := by
  sorry

/-- R7: a polynomial precision obstruction for nonadaptive queries on the analytic class. -/
theorem gaussian_radial_analytic_nonadaptive_lower
    {p N : ℕ} (hp : 0 < p) (hN : 0 < N) {A L : ℝ} (hA : 0 ≤ A) (hL : 1 ≤ L)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → NonadaptiveRule (GaussianSpace (p+1)) N)
    (finiteRisk : ∀ R : Direction (p+1) → ℝ, Measurable R → (∀ u, R u ∈ Icc 1 2) →
      (∀ j : ℕ, 0 < j → ChartPolynomialRemainder j (A*L^j) R) →
      Integrable (fun ω => ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2) μ) :
    ∃ R : Direction (p+1) → ℝ, Measurable R ∧ (∀ u, R u ∈ Icc 1 2) ∧
      (∀ j : ℕ, 0 < j → ChartPolynomialRemainder j (A*L^j) R) ∧
      (cdfSpan (p+1) / (2*(N:ℝ)))^2/8 ≤
        ∫ ω, ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2 ∂μ := by
  sorry

/-- R8: relative-error search lower bound for analytic balls with unknown centers. -/
theorem gaussian_unlocated_ball_lower
    {d N : ℕ} (hd : 0 < d)
    (hN : ((2*N : ℕ):ℝ≥0∞) * unitBallVolume d < (2:ℝ≥0∞)^d)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → QueryTree (GaussianSpace d) ℝ) (budget : ∀ ω, (alg ω).depth ≤ N)
    (hM : ∀ c ∈ coordinateBox d 1, ∀ r : ℝ, (r = 1/16 ∨ r = 1/2) →
      Measurable (fun ω => (alg ω).run (ballOracle c r))) :
    ∃ c ∈ coordinateBox d 1, ∃ r : ℝ, (r = 1/16 ∨ r = 1/2) ∧
      0 < gaussianBallMass c r ∧
      (1:ℝ)/4 ≤ μ.real {ω | gaussianBallMass c r / 4 <
        |(alg ω).run (ballOracle c r) - gaussianBallMass c r|} := by
  sorry

/-- R9: the same-budget worst-case MSE gap on one fixed analytic class is unbounded. -/
theorem gaussian_analytic_same_budget_gap
    {p k : ℕ} (hp : 0 < p) {A L factor : ℝ}
    (hA : (1:ℝ)/2 ≤ A) (hL : 1 ≤ L) (hk : 2*(64:ℝ)^p*L ≤ (2:ℝ)^k) (hf : 0 < factor)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] :
    ∃ N₀ : ℕ, 1 ≤ N₀ ∧ ∀ N : ℕ, N₀ ≤ N →
      let admissible := fun (R : Direction (p+1) → ℝ) => Measurable R ∧
        (∀ u, R u ∈ Icc 1 2) ∧ (∀ j : ℕ, 0 < j → ChartPolynomialRemainder j (A*L^j) R)
      let lower := (cdfSpan (p+1))^2 / (32*(N:ℝ)^2)
      (∃ alg : Samples (Direction (p+1)) 1 → QueryTree (GaussianSpace (p+1)) ℝ,
        (∀ ω, (alg ω).depth ≤ N ∧ (alg ω).queriesIn (coordinateBox (p+1) 40)) ∧
        ∀ R, admissible R →
          let out := fun ω => (alg ω).run (starOracle R)
          let law := samplingLaw (angularLaw (p+1)) 1
          Measurable out ∧ (∫ ω, out ω ∂law) = gaussianStarMass R ∧
          (∫ ω, (out ω - gaussianStarMass R)^2 ∂law) < lower/factor) ∧
      (∀ alg : Ω → NonadaptiveRule (GaussianSpace (p+1)) N,
        (∀ R, admissible R →
          Integrable (fun ω => ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2) μ) →
        ∃ R, admissible R ∧
          lower ≤ ∫ ω, ((alg ω).estimate (starOracle R) - gaussianStarMass R)^2 ∂μ) := by
  sorry

/-- R10: a dimension-explicit deterministic relative guarantee for centered ellipsoids. -/
theorem gaussian_ellipsoid_relative_cubature
    {d k : ℕ} (hd : 0 < d) {L U : ℝ} (hL : 0 < L) (hLU : L ≤ U) (hcap : 1 ≤ 1600*L)
    (hsmall : (d:ℝ)*relativeFormResolution d k L U ≤ 1/2) :
    ∃ alg : QueryTree (GaussianSpace d) ℝ,
      alg.depth ≤ (d+1).choose 2 * k ∧ alg.queriesIn (coordinateBox d 40) ∧
      ∀ Q : RealMatrix d, (∀ i j, Q i j = Q j i) →
        (∀ z, L*‖z‖^2 ≤ quadEval Q z ∧ quadEval Q z ≤ U*‖z‖^2) →
        0 < gaussianSetMass (ellipsoidSet Q) ∧
        |alg.run (ellipsoidOracle Q) - gaussianSetMass (ellipsoidSet Q)| ≤
          (2*(d:ℝ)*relativeFormResolution d k L U)*gaussianSetMass (ellipsoidSet Q) := by
  sorry

end Cubature
