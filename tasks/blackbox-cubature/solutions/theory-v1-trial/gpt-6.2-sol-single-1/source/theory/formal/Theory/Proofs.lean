import Theory.GaussianStar
import Theory.SphereGrid
import Theory.HardMass
import Theory.TraceLower
import Theory.FixedQueries
import Theory.AnalyticBounds
import Theory.AnalyticLower
import Theory.BallSearch
import Theory.BudgetDominance
import Theory.EllipsoidQueries

/-! Kernel-checked proofs of the results in `THEORY.md`. -/

open MeasureTheory Set
open scoped ENNReal

namespace Cubature

private def gridCap (p k : ℕ) : ℕ := (p+1) * 2 * 2^p * (2^k)^p

private theorem gridCount_le_cap (p k : ℕ) :
    (dyadicSphereGrid p k).count ≤ gridCap p k := by
  change (p+1) * 2 * (2^k+1)^p ≤ (p+1) * 2 * 2^p * (2^k)^p
  have hx : 1 ≤ (2:ℕ)^k := by
    have hh : 0 < (2:ℕ)^k := by positivity
    omega
  calc
    _ ≤ (p+1) * 2 * (2 * 2^k)^p := by
      exact Nat.mul_le_mul_left _ (Nat.pow_le_pow_left (by omega) _)
    _ = _ := by rw [mul_pow, mul_assoc ((p+1)*2)]

private theorem gridCap_double {p : ℕ} (hp : 0 < p) (k : ℕ) :
    2 * gridCap p k ≤ gridCap p (k+1) := by
  have ht : 2 ≤ (2:ℕ)^p := by
    have := pow_le_pow_right' (by norm_num : (1:ℕ) ≤ 2) (Nat.succ_le_of_lt hp)
    simpa using this
  calc
    _ ≤ 2^p * gridCap p k := Nat.mul_le_mul_right _ ht
    _ = _ := by
      unfold gridCap
      rw [pow_succ (2:ℕ) k, mul_pow]
      ac_rfl

private theorem pilotSphere_le {p : ℕ} (hp : 0 < p) (s k : ℕ) :
    pilotBudget (dyadicSphereGrid p) s k ≤ 2 * gridCap p k * (s+1) := by
  induction k with
  | zero => simp [pilotBudget]
  | succ k ih =>
    have hd := gridCap_double hp k
    rw [pilotBudget]
    calc
      _ ≤ 2 * gridCap p k * (s+1) + gridCap p (k+1) * (s+1) :=
        Nat.add_le_add ih (Nat.mul_le_mul_right _ (gridCount_le_cap p (k+1)))
      _ = (2 * gridCap p k + gridCap p (k+1)) * (s+1) := by rw [Nat.add_mul]
      _ ≤ 2 * gridCap p (k+1) * (s+1) := Nat.mul_le_mul_right _ (by omega)

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
  let d := p+1
  let G := dyadicSphereGrid p
  let alg' := adaptiveEstimator (angularLaw d) (radialDensity d) G a b A s k n
  have hA : 0 ≤ A := by linarith
  have h₀ : 0 < d := Nat.succ_pos _
  have hprob := angularLaw_isProbability h₀
  have hleg (ω) := adaptiveEstimator_legal (angularLaw d) (radialDensity d) G hab hA s k n ω
  refine ⟨fun ω => (alg' ω).reindex rayPoint, ?_, ?_⟩
  · intro ω
    constructor
    · rw [QueryTree.depth_reindex]
      exact (adaptiveEstimator_depth (angularLaw d) (radialDensity d) G a b A s k n ω).trans
        (Nat.add_le_add_right (pilotSphere_le hp s k) n)
    · apply QueryTree.queriesIn_reindex
      refine QueryTree.queriesIn_mono ?_ (hleg ω)
      rintro ⟨u, t⟩ (ht : t ∈ Icc a b)
      exact rayPoint_mem_coordinateBox u (ha.le.trans ht.1) (ht.2.trans hb)
  · intro R hRm hR hSmooth
    dsimp only
    have hruneq (ω) : ((alg' ω).reindex rayPoint).run (starOracle R) =
        (alg' ω).run (threshold R) := by
      rw [QueryTree.run_reindex]
      refine QueryTree.run_congr (hleg ω) ?_
      rintro ⟨u, t⟩ (ht : t ∈ Icc a b)
      exact starOracle_rayPoint R u (ha.trans_le ht.1)
    have hmeaseq : (fun ω => ((alg' ω).reindex rayPoint).run (starOracle R)) =
        fun ω => (alg' ω).run (threshold R) := funext hruneq
    have hbound : 0 ≤ b := ha.le.trans hab
    have hs := adaptiveEstimator_stats (angularLaw d) (measurable_radialDensity d)
      (integrable_radialDensity h₀) (radialDensity_nonneg d) G hab hA hspan
      (fun r hr => radialDensity_le d hbound hr.2) (radialDensityBound_nonneg d hbound)
      s k hn hRm hR (dyadicSphereGrid_error hSmooth)
    refine ⟨?_, ?_⟩
    · rw [hmeaseq]
      exact hs.1
    · simp_rw [hruneq, gaussianStarMass_eq hRm]
      exact hs.2

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
  let J := Fin (m^2)
  let B := PatchIndex p m
  have hJ : Nonempty J := ⟨⟨0, by positivity⟩⟩
  have hB : Nonempty B := ⟨fun _ => ⟨0, hm⟩⟩
  have hcJ : Fintype.card J = m^2 := Fintype.card_fin _
  have hcB : Fintype.card (PatchIndex p m) = m^p := by simp [PatchIndex, Fintype.card_fin]
  have hN' : 2 * N ≤ Fintype.card J * Fintype.card B := by
    change 2 * N ≤ Fintype.card J * Fintype.card (PatchIndex p m)
    rw [hcJ, hcB, ← pow_add, Nat.add_comm 2]
    exact hN
  obtain ⟨j, bits, h⟩ := nonadaptive_prior_lower_bound μ alg
    (hardSite (p := p) (m := m)) (fun j b => starOracle (hardRadius j b)) (hardMassBase p)
    hardMassCoeff (div_nonneg (patchMassConstant_pos p).le (by positivity))
    (hardMassCoeff_lower hm) (hardOracle_flip hm) hJ hB hN' (fun j b => by
      simp_rw [← hardMass_identity]
      exact finiteRisk _ (measurable_hardRadius j b) (hardRadius_mem hm j b) (hardRadius_remainder hm j b))
  refine ⟨hardRadius j bits, measurable_hardRadius j bits, hardRadius_mem hm j bits,
    hardRadius_remainder hm j bits, ?_⟩
  rw [hardMass_identity]
  simpa only [hcB, Nat.cast_pow] using h

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
  let j : Fin (m^2) := ⟨0, by positivity⟩
  let site (q : GaussianSpace (p+1)) : Option (PatchIndex p m) :=
    (hardSite (p := p) (m := m) q).bind (fun a => if a.1 = j then some a.2 else none)
  have hcB : Fintype.card (PatchIndex p m) = m^p := by simp [PatchIndex, Fintype.card_fin]
  have hsite (i : PatchIndex p m) (bits : PatchIndex p m → Bool) (q : GaussianSpace (p+1))
      (hs : site q ≠ some i) : starOracle (hardRadius j (flip i bits)) q =
      starOracle (hardRadius j bits) q := by
    apply hardOracle_flip hm
    intro h
    exact hs (by simp [site, h])
  obtain ⟨bits, h⟩ := adaptive_prior_lower_bound μ alg budget site
    (fun b => starOracle (hardRadius j b)) (hardMassBase p j) (hardMassCoeff j)
    (div_nonneg (patchMassConstant_pos p).le (by positivity)) (hardMassCoeff_lower hm j)
    hsite (by rwa [hcB]) (fun b => by
      simp_rw [← hardMass_identity]
      exact finiteRisk _ (measurable_hardRadius j b) (hardRadius_mem hm j b) (hardRadius_remainder hm j b))
  refine ⟨hardRadius j bits, measurable_hardRadius j bits, hardRadius_mem hm j bits,
    hardRadius_remainder hm j bits, ?_⟩
  rw [hardMass_identity]
  simpa only [hcB, Nat.cast_pow] using h

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
  let G := sphereGrid p m hm
  let B := m^s
  have hB : 0 < B := by dsimp [B]; positivity
  refine ⟨fixedRadialRule G hB, fixedRadialRule_legal G hB, ?_⟩
  intro R hRm hR hSmooth
  have hBias (x : Direction (p+1)) :
      |G.predict (R ∘ G.point) x - R x| ≤ A/(B:ℝ) := by
    simpa only [B, Nat.cast_pow, div_pow, one_pow, mul_one_div] using sphereGrid_error hm hSmooth x
  have hs := fixedRadialRule_stats G hB hn hRm hR hBias
  dsimp only at hs ⊢
  have hEq : radialDensityBound (p+1) 2 * (A/(B:ℝ)+2/(B:ℝ)) =
      (A+2) * radialDensityBound (p+1) 2 / (m:ℝ)^s := by
    simp only [B,Nat.cast_pow]
    ring
  rw [hEq] at hs
  exact hs

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
  let d := p+1
  let G := dyadicPolynomialGrid p r hr
  let alg' := signedEstimator (angularLaw d) (radialDensity d) G a b E r ell k n
  have hE₀ : 0 ≤ E := by linarith
  have hd : 0 < d := Nat.succ_pos _
  have := angularLaw_isProbability hd
  have hleg (ω) := signedEstimator_legal (angularLaw d) (radialDensity d) G hab hE₀ r ell k n ω
  refine ⟨fun ω => (alg' ω).reindex rayPoint, ?_, ?_⟩
  · intro ω
    constructor
    · rw [QueryTree.depth_reindex]
      exact (signedEstimator_depth (angularLaw d) (radialDensity d) G a b E r ell k n ω).trans
        (Nat.add_le_add_right (dyadicPolynomial_budget hp hr ell k) n)
    · apply QueryTree.queriesIn_reindex
      refine QueryTree.queriesIn_mono ?_ (hleg ω)
      rintro ⟨u,t⟩ (ht : t ∈ Icc a b)
      exact rayPoint_mem_coordinateBox u (ha.le.trans ht.1) (ht.2.trans hb)
  · intro R hRm hR hSmooth
    dsimp only
    have hruneq (ω) : ((alg' ω).reindex rayPoint).run (starOracle R) =
        (alg' ω).run (threshold R) := by
      rw [QueryTree.run_reindex]
      refine QueryTree.run_congr (hleg ω) ?_
      rintro ⟨u,t⟩ (ht : t ∈ Icc a b)
      exact starOracle_rayPoint R u (ha.trans_le ht.1)
    have hs := signedEstimator_stats (angularLaw d) (measurable_radialDensity d)
      (integrable_radialDensity hd) (radialDensity_nonneg d) G hab hE₀ hspan
      (fun t ht => radialDensity_le d (ha.le.trans hab) ht.2)
      (radialDensityBound_nonneg d (ha.le.trans hab)) r ell k hell hn hRm hR
      (dyadicPolynomialGrid_error hr hA hE hSmooth)
    refine ⟨?_,?_⟩
    · rw [funext hruneq]
      exact hs.1
    · simp_rw [hruneq, gaussianStarMass_eq hRm]
      exact hs.2

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
  let ell := 3*r*p+1
  let A' := A*L^r
  let E := interpolationNorm p r * A'
  have hA : 0 ≤ A := by linarith
  have hA' : 0 ≤ A' := mul_nonneg hA (by positivity)
  have hE : 0 ≤ E := mul_nonneg (by positivity : (0:ℝ) ≤ ((8:ℝ)^r)^p) hA'
  have hs : (b-a)/2 ≤ E := by
    calc
      _ ≤ A := hspan
      _ ≤ A' := le_mul_of_one_le_right hA (one_le_pow₀ hL)
      _ ≤ E := le_mul_of_one_le_left hA' (interpolationNorm_ge_one p r)
  obtain ⟨alg, hq, hstats⟩ := gaussian_radial_polynomial_cubature
    (ell := ell) (k := k) hp hr (by decide : 0 < (1:ℕ)) ha hab hb hA' hs
    (le_refl E) (interpolationNorm_ell p r)
  refine ⟨alg, fun ω => ⟨(hq ω).1.trans (geometricBudget hr), (hq ω).2⟩, ?_⟩
  intro R hRm hR hSmooth
  have h := hstats R hRm hR (hSmooth r hr)
  dsimp only at h ⊢
  refine ⟨h.1,h.2.1,h.2.2.trans ?_⟩
  have hW := radialDensityBound_nonneg (p+1) (ha.le.trans hab)
  have hD := mul_le_mul_of_nonneg_right (geometricRadiusError hA hL hk (r := r)) hW
  have h₀ : 0 ≤ (2:ℝ)^ell * (E/(2:ℝ)^(r*k)) * radialDensityBound (p+1) b := by positivity
  simpa only [Nat.cast_one, div_one, div_mul_eq_mul_div] using pow_le_pow_left₀ h₀ hD 2

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
  have hR (t : ℝ) (j : ℕ) (hj : 0 < j) :
      ChartPolynomialRemainder (p := p) j (A*L^j) (fun _ => t) :=
    const_polynomial_remainder hj (mul_nonneg hA (by positivity)) t
  obtain ⟨t,ht,h⟩ := nonadaptive_analytic_prior (J := 2*N) (by positivity)
    le_rfl μ alg (fun t ht => finiteRisk (fun _ => t) measurable_const (fun _ => ht) (hR t))
  exact ⟨fun _ => t, measurable_const, fun _ => ht, hR t, by
    simpa only [Nat.cast_mul,Nat.cast_ofNat] using h⟩

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
  exact ballSearch_bound hd hN μ alg budget hM

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
  let C := 4*(p+1)*(2^k)^p*(3*p+2)
  let d := p+1
  let W := radialDensityBound d 2
  let S := cdfSpan d
  have hC : 0 < C := by dsimp [C]; positivity
  obtain ⟨N₀,hN₀,hgap⟩ := polynomial_budget_gap_all hC (Nat.succ_pos p)
    (2*A*W) (cdfSpan_pos (Nat.succ_pos p)) hf
  refine ⟨N₀,hN₀,fun N hN => ?_⟩
  obtain ⟨r,hr,hcost,hsep⟩ := hgap N hN
  have hrp : 0 < r := hr
  have hNp : 0 < N := hN₀.trans hN
  constructor
  · obtain ⟨alg,hq,hS⟩ := gaussian_radial_analytic_cubature hp hrp
      (by norm_num : (0:ℝ)<1) (by norm_num : (1:ℝ)≤2) (by norm_num : (2:ℝ)≤40)
      (by linarith : ((2:ℝ)-1)/2 ≤ A) hL hk
    refine ⟨alg,fun ω => ⟨(hq ω).1.trans hcost,(hq ω).2⟩,?_⟩
    intro R hR
    obtain ⟨hm,havg,hvar⟩ := hS R hR.1 hR.2.1 hR.2.2
    refine ⟨hm,havg,hvar.trans_lt ?_⟩
    simpa only [div_div] using hsep
  · intro alg hI
    obtain ⟨R,hm,hR,hSmooth,hvar⟩ := gaussian_radial_analytic_nonadaptive_lower hp hNp
      (by linarith : 0 ≤ A) hL μ alg (fun R h₁ h₂ h₃ => hI R ⟨h₁,h₂,h₃⟩)
    refine ⟨R,⟨hm,hR,hSmooth⟩,?_⟩
    have hEq : (S/(2*(N:ℝ)))^2/8 = S^2/(32*(N:ℝ)^2) := by
      have hne : (N:ℝ) ≠ 0 := by positivity
      field_simp
      <;> ring
    exact hEq.symm.le.trans hvar

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
  exact ⟨ellipsoidRule d k L U,ellipsoidRule_cost d k L U,ellipsoidRule_legal d k hL hLU hcap,
    fun _ => ellipsoidRule_accuracy hd hL hLU hsmall⟩

end Cubature
