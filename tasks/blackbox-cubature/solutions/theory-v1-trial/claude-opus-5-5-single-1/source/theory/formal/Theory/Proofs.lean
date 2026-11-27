import Theory.Defs
import Theory.Sublevel
import Theory.Preint
import Theory.Sharp
import Theory.Stratified
import Theory.Cube
import Theory.Quadric
import Theory.Tube
import Theory.Piecewise
import Theory.Refined
import Theory.Cubic
import Theory.QuadStrat
import Theory.QuadTheta
import Theory.Lower

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof.
The auxiliary lemmas live in `Theory/Sublevel.lean`, `Theory/Preint.lean`, ... (namespace
`Cubature.Aux`).
-/

open MeasureTheory ProbabilityTheory Real Set
open scoped InnerProductSpace

namespace Cubature

/-- **R1 (sharp sublevel-set estimate, van der Corput type, order two).**
If `φ : ℝ → ℝ` is `α`-strongly convex (`α > 0`), then for every `η ≥ 0` the set
`{t | |φ t| ≤ η}` has Lebesgue measure at most `4 √(η / α)`. -/
theorem volume_sublevel_le_of_strongConvexOn {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η)
    (hφ : StrongConvexOn univ α φ) :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(η / α)) := by
  exact Aux.volume_sublevel_le_of_strongConvexOn hα hη hφ

/-- **R2 (½-Hölder regularity of non-monotone preintegration, general weight).**
Jump integrand `f(t, y) = θ(t, y) · ind(φ(t, y))`, preintegrated in `t` against a probability
measure `ν` on `ℝ` whose density is at most `ρ` (`ν S ≤ ρ · Leb(S)`).  Assume that for every `y`
the map `t ↦ φ(t, y)` is `α`-strongly convex or `α`-strongly concave (no monotonicity), that
`φ` and `θ` are Lipschitz in `y` uniformly in `t` (constants `L`, `Lθ`), and `|θ| ≤ B`. Then
`|P(y) - P(y')| ≤ Lθ · d(y, y') + 4 ρ B √(L d(y, y') / α)`. -/
theorem preintJump_holder_half_of_density_le {Y : Type*} [PseudoMetricSpace Y]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y))
    (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump ν φ θ y - preintJump ν φ θ y'| ≤
      Lθ * dist y y' + 4 * ρ * B * √(L * dist y y' / α) := by
  exact Aux.jump_holder hρ hν hα hφ hφL hθm hθB hθL y y'

/-- **R2 (Gaussian case).** The same estimate for preintegration against the standard normal
distribution `N(0,1)` (density bounded by `1/√(2π)`). -/
theorem preintJump_holder_half {Y : Type*} [PseudoMetricSpace Y] {φ θ : ℝ → Y → ℝ}
    {α L Lθ B : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y))
    (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump (gaussianReal 0 1) φ θ y - preintJump (gaussianReal 0 1) φ θ y'| ≤
      Lθ * dist y y' + 4 * B / √(2 * π) * √(L * dist y y' / α) := by
  have := Aux.jump_holder (ρ := (√(2 * π))⁻¹) (by positivity) Aux.gaussianReal_le_volume hα hφ
    hφL hθm hθB hθL y y'
  calc _ ≤ _ := this
    _ = _ := by ring

/-- **R2′ (piecewise non-degenerate sections: the global form of the hypothesis of GKS Theorem 1).**
For every `y` the line is covered by `r` convex pieces `J y i` on each of which `t ↦ φ(t, y)` is
`α`-strongly convex, `α`-strongly concave, or `α`-strongly monotone (`α |t - t'| ≤ |φ(t,y) - φ(t',y)|`);
`φ(·, y)` continuous; the other hypotheses as in R2.  Then
`|P(y) - P(y')| ≤ Lθ d + ρ B r (4 √(L d / α) + 2 L d / α)`, `d = d(y, y')`. -/
theorem preintJump_holder_half_piecewise {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ}
    [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) {r : ℕ}
    (J : Y → Fin r → Set ℝ) (hJ : ∀ y i, Convex ℝ (J y i)) (hJcov : ∀ y, ⋃ i, J y i = univ)
    (hφ : ∀ y i, StrongConvexOn (J y i) α (φ · y) ∨ StrongConcaveOn (J y i) α (φ · y) ∨
        ∀ t ∈ J y i, ∀ t' ∈ J y i, α * |t - t'| ≤ |φ t y - φ t' y|)
    (hφc : ∀ y, Continuous (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump ν φ θ y - preintJump ν φ θ y'| ≤
      Lθ * dist y y' + ρ * B * r * (4 * √(L * dist y y' / α) + 2 * (L * dist y y') / α) := by
  exact Aux.jump_holder_piecewise hρ hν hα J hJ hJcov hφ hφc hφL hθm hθB hθL y y'

/-- **R3 (C^{1,½} regularity of non-monotone preintegrated kinks, general weight).**
Kink integrand `f(t, y) = max(φ(t, y), 0)`.  Under strong convexity/concavity in `t`, a uniform
Lipschitz bound `L` in `y` and a uniform second-difference bound `M` in `y`, the preintegrated
function satisfies the Zygmund-type bound
`|P(y+h) + P(y-h) - 2P(y)| ≤ M ‖h‖² + 4 ρ L ‖h‖ √(L ‖h‖ / α)` (order `‖h‖^{3/2}`). -/
theorem preintKink_second_difference_of_density_le {E : Type*} [SeminormedAddCommGroup E]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ : ℝ → E → ℝ} {α L M : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφint : ∀ y, Integrable (φ · y) ν)
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * ‖y - y'‖)
    (hφM : ∀ t y h, |φ t (y + h) + φ t (y - h) - 2 * φ t y| ≤ M * ‖h‖ ^ 2) (y h : E) :
    |preintKink ν φ (y + h) + preintKink ν φ (y - h) - 2 * preintKink ν φ y| ≤
      M * ‖h‖ ^ 2 + 4 * ρ * L * ‖h‖ * √(L * ‖h‖ / α) := by
  exact Aux.kink_second_difference hρ hν hα hφ hφint hφL hφM y h

/-- **R3 (Gaussian case).** -/
theorem preintKink_second_difference {E : Type*} [SeminormedAddCommGroup E]
    {φ : ℝ → E → ℝ} {α L M : ℝ} (hα : 0 < α)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφint : ∀ y, Integrable (φ · y) (gaussianReal 0 1))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * ‖y - y'‖)
    (hφM : ∀ t y h, |φ t (y + h) + φ t (y - h) - 2 * φ t y| ≤ M * ‖h‖ ^ 2) (y h : E) :
    |preintKink (gaussianReal 0 1) φ (y + h) + preintKink (gaussianReal 0 1) φ (y - h) -
        2 * preintKink (gaussianReal 0 1) φ y| ≤
      M * ‖h‖ ^ 2 + 4 * L / √(2 * π) * ‖h‖ * √(L * ‖h‖ / α) := by
  have := Aux.kink_second_difference (ρ := (√(2 * π))⁻¹) (by positivity)
    Aux.gaussianReal_le_volume hα hφ hφint hφL hφM y h
  calc _ ≤ _ := this
    _ = _ := by ring

/-- **R4a (the constant `4` in R1 is sharp).** For `α > 0`, `η ≥ 0` the function
`t ↦ α/2 · t² - η` is `α`-strongly convex and its band `{|φ| ≤ η}` has measure exactly
`4 √(η / α)`. -/
theorem sublevel_bound_sharp {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η) :
    StrongConvexOn univ α (fun t : ℝ => α / 2 * t ^ 2 - η) ∧
      volume {t : ℝ | |α / 2 * t ^ 2 - η| ≤ η} = ENNReal.ofReal (4 * √(η / α)) := by
  exact Aux.sublevel_sharp hα hη

/-- **R4b (the exponent `1/2` in R2 is sharp).** Gilbert–Kuo–Sloan Example 1:
`φ(t, y) = y - t²` (strongly concave in `t`, `α = 2`, `L = 1`), `θ = 1`, Gaussian weight.
For `0 ≤ y ≤ 1` the preintegrated function satisfies the two-sided square-root bound
`2 e^{-1/2}/√(2π) · √y ≤ P(y) - P(0) ≤ 4/√(2π) · √(y/2)`. -/
theorem gks_parabola_two_sided (y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    2 * rexp (-1 / 2) / √(2 * π) * √y ≤
        preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) y -
          preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) 0 ∧
      preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) y -
          preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) 0 ≤
        4 / √(2 * π) * √(y / 2) := by
  exact ⟨Aux.gks_example_lower y hy0 hy1, Aux.gks_example_upper y hy0⟩

/-- **R5a (stratified sampling, abstract form; classical).** `μ` is partitioned (up to a null
set) into the measurable, pairwise disjoint cells `A i`, `i ∈ ι`, `n = |ι|`.  The points `X i`
are independent and `X i` has law `n · μ|_{A i}` (i.e. `μ` conditioned on `A i`; this forces
`μ (A i) = 1/n`).  If `|F| ≤ B` and the oscillation of `F` on `A i` is at most `c i`, then the
stratified estimator `(1/n) ∑ F (X i)` is unbiased and its mean-square error is at most
`(1/n)² ∑ c i²`. -/
theorem stratified_sampling_mse {Ω Y ι : Type*} [MeasurableSpace Ω] [MeasurableSpace Y]
    [Fintype ι] {P : Measure Ω} [IsProbabilityMeasure P] {μ : Measure Y}
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Function.onFun Disjoint A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (A i))
    {F : Y → ℝ} (hFm : Measurable F) {B : ℝ} (hFB : ∀ x, |F x| ≤ B)
    {c : ι → ℝ} (hFc : ∀ i, ∀ x ∈ A i, ∀ x' ∈ A i, |F x - F x'| ≤ c i) :
    (∫ ω, (1 / Fintype.card ι : ℝ) * ∑ i, F (X i ω) ∂P = ∫ y, F y ∂μ) ∧
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, F (X i ω) - ∫ y, F y ∂μ) ^ 2 ∂P ≤
      (1 / Fintype.card ι : ℝ) ^ 2 * ∑ i, c i ^ 2 := by
  exact Aux.stratified_mse hAm hAd hAcov hXm hXind hXlaw hFm hFB hFc

/-- **R5b (stratified sampling after non-monotone preintegration, general partition).**
Under the hypotheses of R2, apply stratified sampling (R5a) to the preintegrated function
`P = preintJump ν φ θ` on any equal-mass partition of the parameter space.  Cells `i ∈ G` have
diameter at most `D i`; the other cells (e.g. unbounded Gaussian tail cells) are arbitrary.
Then the estimator is unbiased for `∫ P dμ` and its mean-square error is at most
`(1/n)² (∑_{i∈G} (Lθ D i + 4 ρ B √(L D i / α))² + ∑_{i∉G} (2B)²)`. -/
theorem preintJump_stratified_mse {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
    [MeasurableSpace Y] [OpensMeasurableSpace Y]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 ≤ L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    {μ : Measure Y} {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Function.onFun Disjoint A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    (G : Finset ι) {D : ι → ℝ}
    (hAD : ∀ i ∈ G, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ D i)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (A i)) :
    (∫ ω, (1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) ∂P =
        ∫ y, preintJump ν φ θ y ∂μ) ∧
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      (1 / Fintype.card ι : ℝ) ^ 2 * (∑ i ∈ G, (Lθ * D i + 4 * ρ * B * √(L * D i / α)) ^ 2 +
        ∑ i ∈ Gᶜ, (2 * B) ^ 2) := by
  exact Aux.jump_stratified hρ hν hα hL hLθ hφ hφL hθm hθB hθL hAm hAd hAcov G hAD hXm hXind
    hXlaw

/-- **R5c (jittered sampling on the unit cube after non-monotone preintegration: explicit
rate).** `φ, θ` are defined on `ℝ × ℝ^s` and satisfy the hypotheses of R2 there (sup-distance on
`ℝ^s = Fin s → ℝ`); the target is `∫_{[0,1)^s} P` (Lebesgue measure), with `[0,1)^s` split into the
`n = m^s` cells `cubeCell s m k` of side `1/m` and one independent uniform point `X k` per cell.
The jittered estimator of `∫_{[0,1)^s} P` is
unbiased and its mean-square error is at most `(Lθ/m + 4 ρ B √(L/(α m)))² / m^s`, i.e.
`O(n^{-1-1/s})`. -/
theorem preintJump_jittered_cube_mse {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → (Fin s → ℝ) → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 ≤ L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin s → Fin m) → Ω → (Fin s → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ENNReal) ^ s) • volume.restrict (cubeCell s m k)) :
    (∫ ω, (1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) ∂P =
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ∧
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      (Lθ / m + 4 * ρ * B * √(L / (α * m))) ^ 2 / (m : ℝ) ^ s := by
  exact Aux.jump_jittered_cube hm hρ hν hα hL hLθ hφ hφL hθm hθB hθL hXm hXind hXlaw

/-- **R5d (the jittered design of R5c exists).** There are a probability space and independent
random points `X k`, `X k` uniform on `cubeCell s m k`, as required in R5c. -/
theorem jittered_design_exists (s m : ℕ) (hm : 0 < m) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (P : Measure Ω) (_ : IsProbabilityMeasure P)
      (X : (Fin s → Fin m) → Ω → (Fin s → ℝ)),
      (∀ k, Measurable (X k)) ∧ iIndepFun X P ∧
      ∀ k, P.map (X k) = ((m : ENNReal) ^ s) • volume.restrict (cubeCell s m k) := by
  exact Aux.jittered_exists s m hm

/-- **R5e (preintegration preserves the integral — Fubini).** If `φ`, `θ` are jointly measurable
and `θ` is bounded, then `∫ P dμ = ∫ θ · ind(φ) d(ν ⊗ μ)`.  Hence the estimators of R5b/R5c are
unbiased for the original `(s+1)`-dimensional integral `I(f) = E[f(T, Y)]`, `(T, Y) ~ ν ⊗ μ`. -/
theorem preintJump_integral_eq {Y : Type*} [MeasurableSpace Y] {μ : Measure Y}
    [IsFiniteMeasure μ] {ν : Measure ℝ} [IsFiniteMeasure ν] {φ θ : ℝ → Y → ℝ}
    (hφ : Measurable (Function.uncurry φ)) (hθ : Measurable (Function.uncurry θ)) {B : ℝ}
    (hθB : ∀ t y, |θ t y| ≤ B) :
    ∫ y, preintJump ν φ θ y ∂μ =
      ∫ p : ℝ × Y, {q : ℝ × Y | 0 < φ q.1 q.2}.indicator (fun q => θ q.1 q.2) p ∂(ν.prod μ) := by
  exact Aux.preintJump_fubini hφ hθ hθB

/-- **R6a (quadric jump surfaces, arbitrary preintegration direction).** Let
`φ(z) = ⟪z, A z⟫ + ⟪b, z⟫ + c` be a quadric on a real inner-product space `E` (e.g. `ℝ^d`),
`f = ind(φ)`, and preintegrate along an arbitrary direction `u` with `a = ⟪u, A u⟫ ≠ 0`
(generically no direction makes `φ` monotone).  With
`β(x) = ⟪x, A u⟫ + ⟪u, A x⟫ + ⟪b, u⟫` and `D(x) = β(x)² - 4 a φ(x)` (discriminant of
`t ↦ φ(x + t u)`), the preintegrated function `P_u(x) = P(φ(x + Z u) > 0)`, `Z ~ N(0,1)`,
satisfies `|P_u(x) - P_u(x')| ≤ (|β(x) - β(x')| + √|D(x) - D(x')|) / (√(2π) |a|)`. -/
theorem quadric_preint_holder {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu : ⟪u, A u⟫_ℝ ≠ 0)
    (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z) (x x' : E) :
    |preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x -
        preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x'| ≤
      (|β x - β x'| + √|D x - D x'|) / (√(2 * π) * |⟪u, A u⟫_ℝ|) := by
  exact Aux.quadric_preint A b c φ hφ u hu β D hβ hD x x'

/-- **R6b (explicit local ½-Hölder bound for quadrics).** Same setting, `‖u‖ = 1`,
`‖x‖, ‖x'‖ ≤ R`, `δ = ‖x - x'‖`:
`|P_u(x) - P_u(x')| ≤ (2‖A‖δ + √(8‖A‖(2‖A‖R + ‖b‖) δ)) / (√(2π) |⟪u, A u⟫|)`. -/
theorem quadric_preint_holder_local {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) {R : ℝ} (x x' : E) (hx : ‖x‖ ≤ R) (hx' : ‖x'‖ ≤ R) :
    |preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x -
        preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x'| ≤
      (2 * ‖A‖ * ‖x - x'‖ + √(8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * ‖x - x'‖)) /
        (√(2 * π) * |⟪u, A u⟫_ℝ|) := by
  exact Aux.quadric_preint_local A b c φ hφ u hu1 hu x x' hx hx'

/-- **R7a (faster stratified rate under a tube condition, general partition).** Hypotheses of R2,
joint measurability of `φ`, an equal-mass partition of the probability space `(Y, μ)` into cells
of diameter `≤ h`, independent `X i ~ μ(· | A i)`, and the tube (Minkowski-content type) condition
`(ν ⊗ μ){(t, y) : |φ(t, y)| ≤ L h} ≤ K L h`.  Then the stratified estimator of `∫ P dμ` has
mean-square error at most `(2/n) (Lθ² h² + 4 ρ K B² (L h) √(L h / α))`, i.e. `O(h^{3/2} / n)`. -/
theorem preintJump_stratified_mse_tube {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
    [MeasurableSpace Y] [OpensMeasurableSpace Y]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 ≤ L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφm : Measurable (Function.uncurry φ))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {ι : Type*} [Fintype ι] {μ : Measure Y} [IsProbabilityMeasure μ]
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Function.onFun Disjoint A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 ≤ h) (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {K : ℝ} (hK0 : 0 ≤ K)
    (hK : (ν.prod μ) {p : ℝ × Y | |φ p.1 p.2| ≤ L * h} ≤ ENNReal.ofReal (K * (L * h)))
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + 4 * ρ * K * B ^ 2 * (L * h) * √(L * h / α)) := by
  exact Aux.jump_stratified_tube hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hAm hAd hAcov hh hAD
    hK0 hK hXm hXind hXlaw

/-- **R7b (jittered sampling on the unit cube under the tube condition: rate
`O(n^{-1-3/(2s)})`).** Setting of R5c (hypotheses of R2 on `ℝ × ℝ^s`, integral over `[0,1)^s`)
plus joint measurability and the tube condition at scale `L/m`. Mean-square error `≤ (2/m^s) (Lθ²/m² + 4 ρ K B² (L/m) √(L/(α m)))`. -/
theorem preintJump_jittered_cube_mse_tube {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ}
    (hm : 0 < m) {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → (Fin s → ℝ) → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 ≤ L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφm : Measurable (Function.uncurry φ))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {K : ℝ} (hK0 : 0 ≤ K)
    (hK : (ν.prod (volume.restrict (Set.pi univ fun _ => Ico (0 : ℝ) 1)))
      {p : ℝ × (Fin s → ℝ) | |φ p.1 p.2| ≤ L / m} ≤ ENNReal.ofReal (K * (L / m)))
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin s → Fin m) → Ω → (Fin s → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ENNReal) ^ s) • volume.restrict (cubeCell s m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      2 / (m : ℝ) ^ s * (Lθ ^ 2 / m ^ 2 + 4 * ρ * K * B ^ 2 * (L / m) * √(L / (α * m))) := by
  exact Aux.jump_jittered_cube_tube hm hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hK0 hK hXm hXind
    hXlaw

/-- **R7c (the tube condition holds for GKS Example 1).** For `φ(t, y) = y₀ - t²` on
`ℝ × [0,1)^{s+1}` and any probability measure `ν`, `(ν ⊗ U[0,1)^{s+1}){|φ| ≤ η} ≤ 2η`. -/
theorem gks_example_tube_condition (s : ℕ) {ν : Measure ℝ} [IsProbabilityMeasure ν] (η : ℝ)
    (hη : 0 ≤ η) :
    (ν.prod (volume.restrict (Set.pi univ fun _ : Fin (s + 1) => Ico (0 : ℝ) 1)))
      {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η} ≤ ENNReal.ofReal (2 * η) := by
  exact Aux.gks_tube s η hη

/-- **R7d (explicit rate for GKS Example 1 lifted to dimension `d = s + 2`).** Non-monotone jump
`f(t, y) = ind(y₀ - t²)`, `t ~ N(0,1)` preintegrated, `y ~ U[0,1)^{s+1}` jittered with `n = m^{s+1}`
points: mean-square error `≤ 16 / (√(2π) m^{s+2}) · √(1/(2m)) = O(n^{-1-3/(2(s+1))})`. -/
theorem gks_example_jittered_mse {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin (s + 1) → Fin m) → Ω → (Fin (s + 1) → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ENNReal) ^ (s + 1)) • volume.restrict (cubeCell (s + 1) m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ (s + 1)) * ∑ k, preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) (X k ω) -
        ∫ y in Set.pi univ (fun _ : Fin (s + 1) => Ico (0 : ℝ) 1), preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) y) ^ 2 ∂P ≤
      16 / (√(2 * π) * (m : ℝ) ^ (s + 2)) * √(1 / (2 * m)) := by
  exact Aux.gks_example_jittered hm hXm hXind hXlaw

/-- **R8a (refined band estimate away from tangency).** If `ψ : ℝ → ℝ` is `α`-strongly concave
with maximum value `H` (attained), `η > 0` and `2η ≤ H`, then
`Leb{t : |ψ t| ≤ η} ≤ 2 √(24 η² / (α H))` (order `η / √H` instead of `√η`). -/
theorem volume_band_le_of_strongConcaveOn_max {ψ : ℝ → ℝ} {α η H : ℝ} (hα : 0 < α)
    (hψ : StrongConcaveOn univ α ψ) (hle : ∀ t, ψ t ≤ H) {u : ℝ} (hu : ψ u = H) (hη : 0 < η)
    (hH : 2 * η ≤ H) :
    volume {t : ℝ | |ψ t| ≤ η} ≤ ENNReal.ofReal (2 * √(24 * η ^ 2 / (α * H))) := by
  exact Aux.volume_band_refined hα hψ hle hu hη hH

/-- **R8b (near-Lipschitz stratified rate after non-monotone preintegration, general
partition).** Hypotheses of R2 with `t ↦ φ(t, y)` `α`-strongly concave, `φ` jointly measurable,
`H(y) = max_t φ(t, y)` (attained) measurable with `H ≤ Hmax` `μ`-a.e., and the *transversality of
the shadow boundary* `μ ∘ H⁻¹ ≤ K · Leb` (the height `H` has density `≤ K`).  For an equal-mass
partition into cells of diameter `≤ h` with `2 L h ≤ Hmax` and independent `X i ~ μ(· | A i)`, the
stratified estimator has mean-square error at most
`(2/n) (Lθ² h² + B² K (96 ρ² (L h)² / α · log(Hmax / (2 L h)) + 48 ρ² (L h)² / α))`,
i.e. `O(h² log(1/h) / n)` — the rate for Lipschitz integrands up to a logarithm. -/
theorem preintJump_stratified_mse_height {Ω Y : Type*} [MeasurableSpace Ω]
    [PseudoMetricSpace Y] [MeasurableSpace Y] [OpensMeasurableSpace Y]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 < L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConcaveOn univ α (φ · y))
    (hφm : Measurable (Function.uncurry φ))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {H : Y → ℝ} (hHm : Measurable H) (hH1 : ∀ t y, φ t y ≤ H y) (hH2 : ∀ y, ∃ u, φ u y = H y)
    {ι : Type*} [Fintype ι] {μ : Measure Y} [IsProbabilityMeasure μ]
    {Hmax K : ℝ} (hK0 : 0 ≤ K) (hHmax : ∀ᵐ y ∂μ, H y ≤ Hmax)
    (hK : μ.map H ≤ ENNReal.ofReal K • volume)
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Function.onFun Disjoint A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 < h) (h2 : 2 * (L * h) ≤ Hmax)
    (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L * h) ^ 2 / α * Real.log (Hmax / (2 * (L * h))) +
          48 * ρ ^ 2 * (L * h) ^ 2 / α)) := by
  exact Aux.jump_stratified_height hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hHm hH1 hH2 hK0 hHmax hK
    hAm hAd hAcov hh h2 hAD hXm hXind hXlaw

/-- **R8b′ (convex case: outside of closed surfaces).** Same as R8b when `t ↦ φ(t, y)` is
`α`-strongly convex (e.g. `f = ind(φ)` is the indicator of the *outside* of a closed surface, such
as a tail event), with height `H(y) = -min_t φ(t, y)` (attained) in the transversality condition. -/
theorem preintJump_stratified_mse_height_convex {Ω Y : Type*} [MeasurableSpace Ω]
    [PseudoMetricSpace Y] [MeasurableSpace Y] [OpensMeasurableSpace Y]
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 < L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y))
    (hφm : Measurable (Function.uncurry φ))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {H : Y → ℝ} (hHm : Measurable H) (hH1 : ∀ t y, -φ t y ≤ H y) (hH2 : ∀ y, ∃ u, -φ u y = H y)
    {ι : Type*} [Fintype ι] {μ : Measure Y} [IsProbabilityMeasure μ]
    {Hmax K : ℝ} (hK0 : 0 ≤ K) (hHmax : ∀ᵐ y ∂μ, H y ≤ Hmax)
    (hK : μ.map H ≤ ENNReal.ofReal K • volume)
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Function.onFun Disjoint A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 < h) (h2 : 2 * (L * h) ≤ Hmax)
    (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L * h) ^ 2 / α * Real.log (Hmax / (2 * (L * h))) +
          48 * ρ ^ 2 * (L * h) ^ 2 / α)) := by
  exact Aux.jump_stratified_height_convex hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hHm hH1 hH2 hK0
    hHmax hK hAm hAd hAcov hh h2 hAD hXm hXind hXlaw

/-- **R8c (unit cube: `O(n^{-1-2/s} log n)`).** Jittered sampling with `n = m^s` cells of
`[0,1)^s`, under the hypotheses of R8b (with `H ≤ Hmax` on the cube, the density condition for
the uniform measure, and `2 L/m ≤ Hmax`):
MSE `≤ (2/m^s) (Lθ²/m² + B² K (96 ρ² (L/m)²/α · log(Hmax m/(2L)) + 48 ρ² (L/m)²/α))`. -/
theorem preintJump_jittered_cube_mse_height {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ}
    (hm : 0 < m) {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → (Fin s → ℝ) → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 < L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConcaveOn univ α (φ · y))
    (hφm : Measurable (Function.uncurry φ))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {H : (Fin s → ℝ) → ℝ} (hHm : Measurable H) (hH1 : ∀ t y, φ t y ≤ H y)
    (hH2 : ∀ y, ∃ u, φ u y = H y) {Hmax K : ℝ} (hK0 : 0 ≤ K)
    (hHmax : ∀ y ∈ Set.pi univ (fun _ => Ico (0 : ℝ) 1), H y ≤ Hmax)
    (hK : (volume.restrict (Set.pi univ fun _ => Ico (0 : ℝ) 1)).map H ≤
      ENNReal.ofReal K • volume)
    (h2 : 2 * (L / m) ≤ Hmax)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin s → Fin m) → Ω → (Fin s → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ENNReal) ^ s) • volume.restrict (cubeCell s m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      2 / (m : ℝ) ^ s * (Lθ ^ 2 / m ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L / m) ^ 2 / α * Real.log (Hmax / (2 * (L / m))) +
          48 * ρ ^ 2 * (L / m) ^ 2 / α)) := by
  exact Aux.jump_jittered_cube_height hm hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hHm hH1 hH2 hK0
    hHmax hK h2 hXm hXind hXlaw

/-- **R8d (GKS Example 1 lifted to dimension `d = s + 2`: all hypotheses verified).**
`f(t, y) = ind(y₀ - t²)`, `t ~ N(0,1)` preintegrated, `y ~ U[0,1)^{s+1}` jittered with
`n = m^{s+1}` points (`m ≥ 2`): mean-square error `≤ (48 log(m/2) + 24) / (π m^{s+3})`, i.e.
`O(n^{-1-2/(s+1)} log n)`. -/
theorem gks_example_jittered_mse_log {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 2 ≤ m)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin (s + 1) → Fin m) → Ω → (Fin (s + 1) → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ENNReal) ^ (s + 1)) • volume.restrict (cubeCell (s + 1) m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ (s + 1)) * ∑ k, preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) (X k ω) -
        ∫ y in Set.pi univ (fun _ : Fin (s + 1) => Ico (0 : ℝ) 1), preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) y) ^ 2 ∂P ≤
      (48 * Real.log (m / 2) + 24) / (π * (m : ℝ) ^ (s + 3)) := by
  exact Aux.gks_example_jittered_log hm hXm hXind hXlaw

/-- **R9a (sublevel estimate for third-order tangency).** If `φ` is differentiable with
derivative `φ'` and `φ'` is `α`-strongly convex or `α`-strongly concave (so `|φ'''| ≥ α` for `C³`
functions), then `Leb{t : |φ t| ≤ η} ≤ 10 (η / α)^{1/3}`. -/
theorem volume_band_le_of_deriv_strong {φ φ' : ℝ → ℝ} (hd : ∀ t, HasDerivAt φ (φ' t) t)
    {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η)
    (hφ' : StrongConvexOn univ α φ' ∨ StrongConcaveOn univ α φ') :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (10 * (η / α) ^ ((1 : ℝ) / 3)) := by
  exact Aux.volume_band_cubic_opt hd hα hη hφ'

/-- **R9b (⅓-Hölder regularity of preintegration at third-order tangencies).** Hypotheses of R2,
except that instead of strong convexity of `φ(·, y)` we assume that `t ↦ φ(t, y)` has derivative
`φ'(t, y)` which is `α`-strongly convex or concave in `t` (GKS Example 4 type).  Then
`|P(y) - P(y')| ≤ Lθ d + 10 ρ B (L d / α)^{1/3}`. -/
theorem preintJump_holder_third {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ}
    [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ φ' θ : ℝ → Y → ℝ} {α L Lθ B : ℝ} (hα : 0 < α)
    (hd : ∀ t y, HasDerivAt (φ · y) (φ' t y) t)
    (hφ' : ∀ y, StrongConvexOn univ α (φ' · y) ∨ StrongConcaveOn univ α (φ' · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump ν φ θ y - preintJump ν φ θ y'| ≤
      Lθ * dist y y' + 10 * ρ * B * (L * dist y y' / α) ^ ((1 : ℝ) / 3) := by
  exact Aux.jump_holder_third hρ hν hα hd hφ' hφL hθm hθB hθL y y'

/-- **R9c (the exponent `1/3` is sharp: GKS Example 4).** `φ(t, y) = t³ - y`, `θ = 1`, Gaussian
weight: for `0 ≤ y ≤ 1`,
`e^{-1/2}/√(2π) · y^{1/3} ≤ P(0) - P(y) ≤ 10/√(2π) · (y/6)^{1/3}`. -/
theorem gks_example4_two_sided (y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    rexp (-1 / 2) / √(2 * π) * y ^ ((1 : ℝ) / 3) ≤
        preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) 0 -
          preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) y ∧
      preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) 0 -
          preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) y ≤
        10 / √(2 * π) * (y / 6) ^ ((1 : ℝ) / 3) := by
  exact Aux.gks_example4_two_sided y hy0 hy1

/-- **R10 (quadric jump surfaces in an arbitrary direction: stratified rate up to a log).**
`φ(z) = ⟪z, A z⟫ + ⟪b, z⟫ + c` on a real inner-product space `E`, jump `f = ind(φ)`, preintegration
along any unit vector `u` with `a = ⟪u, A u⟫ ≠ 0` against `N(0,1)`; `β`, `D` as in R6 (`D(x)` is the
discriminant of `t ↦ φ(x + t u)`; `{D = 0}` is the shadow boundary).  Let `μ` be a probability
measure on `E` partitioned (up to a null set) into equal-mass cells `C i` of diameter `≤ h` lying in
the ball of radius `R`, with `D ≤ Dmax` on the cells and the law of `D` under `μ` having density
`≤ K` (`μ.map D ≤ K · Leb`).  With `η = 8 ‖A‖ (2 ‖A‖ R + ‖b‖) h` and `2 η ≤ Dmax`, stratified
sampling of the preintegrated function (independent `X i ~ μ(· | C i)`) has mean-square error at
most `(1/n) ((2 ‖A‖ h)² + K (η² log(Dmax / (2η)) + 3 η²)) / (π a²)`, i.e. `O(h² log(1/h) / n)`. -/
theorem quadric_stratified_mse {Ω E : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [MeasurableSpace E] [OpensMeasurableSpace E]
    (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z)
    {ι : Type*} [Fintype ι] {μ : Measure E} [IsProbabilityMeasure μ]
    {C : ι → Set E} (hCm : ∀ i, MeasurableSet (C i))
    (hCd : Pairwise (Function.onFun Disjoint C)) (hCcov : μ (⋃ i, C i)ᶜ = 0)
    {R h Dmax K : ℝ} (hR : 0 < R) (hh : 0 < h)
    (hCR : ∀ i, ∀ x ∈ C i, ‖x‖ ≤ R) (hCD : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, dist x x' ≤ h)
    (hDmax : ∀ i, ∀ x ∈ C i, D x ≤ Dmax) (hK0 : 0 ≤ K)
    (hK : μ.map D ≤ ENNReal.ofReal K • volume)
    (h2 : 2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ≤ Dmax)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → E} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (C i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump (gaussianReal 0 1)
          (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) (X i ω) -
        ∫ x, preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x ∂μ) ^ 2
        ∂P ≤
      1 / Fintype.card ι * ((2 * ‖A‖ * h) ^ 2 +
        K * ((8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2 *
          Real.log (Dmax / (2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h))) +
          3 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2)) / (π * ⟪u, A u⟫_ℝ ^ 2) := by
  exact Aux.quadric_stratified A b c φ hφ u hu1 hu β D hβ hD hCm hCd hCcov hR hh hCR hCD hDmax
    hK0 hK h2 hXm hXind hXlaw

/-- **R10′ (quadric jump surfaces with a Lipschitz amplitude).** Same setting as R10 for the
integrand `f(z) = θ(z) · ind(φ(z))` with `|θ| ≤ B` and `|θ z - θ z'| ≤ Lθ ‖z - z'‖` (a general
one-surface jump function of the class `F_d` is `h₋ + (h₊ - h₋) · ind(φ)`, i.e. a smooth part plus
such a term).  Mean-square error of the stratified estimator:
`(1/n) (2 (Lθ h)² + 2 B² ((2 ‖A‖ h)² + K (η² log(Dmax/(2η)) + 3 η²)) / (π a²))`,
`η = 8 ‖A‖ (2 ‖A‖ R + ‖b‖) h`. -/
theorem quadric_theta_stratified_mse {Ω E : Type*} [MeasurableSpace Ω] [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [MeasurableSpace E] [OpensMeasurableSpace E]
    (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z)
    (θ : E → ℝ) {B Lθ : ℝ} (hθB : ∀ z, |θ z| ≤ B)
    (hθL : ∀ z z', |θ z - θ z'| ≤ Lθ * ‖z - z'‖)
    {ι : Type*} [Fintype ι] {μ : Measure E} [IsProbabilityMeasure μ]
    {C : ι → Set E} (hCm : ∀ i, MeasurableSet (C i))
    (hCd : Pairwise (Function.onFun Disjoint C)) (hCcov : μ (⋃ i, C i)ᶜ = 0)
    {R h Dmax K : ℝ} (hR : 0 < R) (hh : 0 < h)
    (hCR : ∀ i, ∀ x ∈ C i, ‖x‖ ≤ R) (hCD : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, dist x x' ≤ h)
    (hDmax : ∀ i, ∀ x ∈ C i, D x ≤ Dmax) (hK0 : 0 ≤ K)
    (hK : μ.map D ≤ ENNReal.ofReal K • volume)
    (h2 : 2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ≤ Dmax)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → E} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ENNReal) • μ.restrict (C i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump (gaussianReal 0 1)
          (fun t z => φ (z + t • u)) (fun t z => θ (z + t • u)) (X i ω) -
        ∫ x, preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u))
          (fun t z => θ (z + t • u)) x ∂μ) ^ 2 ∂P ≤
      1 / Fintype.card ι * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * ‖A‖ * h) ^ 2 +
        K * ((8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2 *
          Real.log (Dmax / (2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h))) +
          3 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2)) / (π * ⟪u, A u⟫_ℝ ^ 2)) := by
  exact Aux.quadric_theta_stratified A b c φ hφ u hu1 hu β D hβ hD θ hθB hθL hCm hCd hCcov hR hh
    hCR hCD hDmax hK0 hK h2 hXm hXind hXlaw

/-- **R11 (lower bound for deterministic rules on the non-monotone class).** For every `s ≥ 1`,
all nodes `y₁, …, yₙ ∈ ℝ^s` and every `m ≥ 1` with `2n ≤ m^s` there is `φ` in the class of R2
(`t ↦ φ(t, y)` 2-strongly concave, `φ` 1-Lipschitz in `y`, `θ = 1`) such that the jump integrand
`ind(φ(t, y))` vanishes on every line `{(t, yᵢ) : t ∈ ℝ}` through a node, its preintegrated function
vanishes at every node, and `∫_{[0,1)^s} P ≥ √2 e^{-1/2} / (√(2π) 2^{s+2} m)`.  Hence every
deterministic algorithm (any weights, adaptive or not) using `n` values of `P`, or `n` point values
of the `(s+1)`-dimensional integrand, has worst-case error `≥ c_s/m ≍ n^{-1/s}` on this class. -/
theorem deterministic_lower_bound (s n m : ℕ) (hs : 0 < s) (hm : 0 < m) (hnm : 2 * n ≤ m ^ s)
    (y : Fin n → (Fin s → ℝ)) :
    ∃ φ : ℝ → (Fin s → ℝ) → ℝ,
      (∀ z, StrongConcaveOn univ 2 (φ · z)) ∧
      (∀ t z z', |φ t z - φ t z'| ≤ 1 * dist z z') ∧
      (∀ i t, φ t (y i) ≤ 0) ∧
      (∀ i, preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) (y i) = 0) ∧
      √2 * rexp (-1 / 2) / √(2 * π) / (2 ^ (s + 2) * m) ≤
        ∫ z in Set.pi univ (fun _ => Ico (0 : ℝ) 1),
          preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) z := by
  exact Aux.fooling s n m hs hm hnm y

end Cubature
