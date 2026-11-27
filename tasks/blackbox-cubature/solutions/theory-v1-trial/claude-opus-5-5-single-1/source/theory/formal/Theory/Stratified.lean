import Theory.Defs
import Theory.Preint

/-!
# Stratified sampling (jittered / stratified RQMC) for preintegrated integrands
(auxiliary proofs for R5)
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology
open scoped ENNReal

namespace Cubature.Aux

theorem stratified_mse {Ω Y ι : Type*} [MeasurableSpace Ω] [MeasurableSpace Y] [Fintype ι]
    {P : Measure Ω} [IsProbabilityMeasure P] {μ : Measure Y}
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i))
    {F : Y → ℝ} (hFm : Measurable F) {B : ℝ} (hFB : ∀ x, |F x| ≤ B)
    {c : ι → ℝ} (hFc : ∀ i, ∀ x ∈ A i, ∀ x' ∈ A i, |F x - F x'| ≤ c i) :
    (∫ ω, (1 / Fintype.card ι : ℝ) * ∑ i, F (X i ω) ∂P = ∫ y, F y ∂μ) ∧
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, F (X i ω) - ∫ y, F y ∂μ) ^ 2 ∂P ≤
      (1 / Fintype.card ι : ℝ) ^ 2 * ∑ i, c i ^ 2 := by
  set n := Fintype.card ι with hn_def
  -- integrability of the summands
  have hFX : ∀ i, Integrable (fun ω => F (X i ω)) P := fun i =>
    Integrable.of_bound (hFm.comp (hXm i)).aestronglyMeasurable B
      (ae_of_all _ fun ω => by rw [Real.norm_eq_abs]; exact hFB _)
  have hFXL2 : ∀ i, MemLp (fun ω => F (X i ω)) 2 P := fun i =>
    MemLp.of_bound (hFm.comp (hXm i)).aestronglyMeasurable B
      (ae_of_all _ fun ω => by rw [Real.norm_eq_abs]; exact hFB _)
  -- mass of the restricted measures is finite
  -- expectation of each summand
  have hexp : ∀ i, ∫ ω, F (X i ω) ∂P = (n : ℝ) * ∫ y in A i, F y ∂μ := by
    intro i
    rw [← integral_map (hXm i).aemeasurable hFm.aestronglyMeasurable, hXlaw i,
      integral_smul_measure]
    simp
  -- n ≠ 0 since P.map (X i) is a probability measure ... handle n = 0 separately
  rcases Nat.eq_zero_or_pos n with hn | hn
  · have hι : IsEmpty ι := Fintype.card_eq_zero_iff.mp hn
    have : μ = 0 := by
      simpa [Measure.measure_univ_eq_zero] using hAcov
    subst this
    simp
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  -- each A i is integrable-on (F bounded, μ (A i) finite)
  have hμA : ∀ i, μ (A i) < ∞ := by
    intro i
    have h1 : (P.map (X i)) univ = 1 := by
      rw [Measure.map_apply (hXm i) MeasurableSet.univ]; simp
    rw [hXlaw i, Measure.smul_apply, Measure.restrict_apply MeasurableSet.univ, univ_inter]
      at h1
    by_contra h
    rw [not_lt, top_le_iff] at h
    rw [h, smul_eq_mul, ENNReal.mul_top (by exact_mod_cast hn.ne')] at h1
    exact ENNReal.top_ne_one h1
  have hIA : ∀ i, IntegrableOn F (A i) μ := fun i =>
    IntegrableOn.of_bound (hμA i) hFm.aestronglyMeasurable B
      (ae_of_all _ fun x => by rw [Real.norm_eq_abs]; exact hFB x)
  have hsumA : ∑ i, ∫ y in A i, F y ∂μ = ∫ y, F y ∂μ := by
    rw [← integral_iUnion_fintype hAm hAd hIA]
    rw [setIntegral_congr_set (ae_eq_univ.mpr hAcov), setIntegral_univ]
  have hmean : ∫ ω, (1 / n : ℝ) * ∑ i, F (X i ω) ∂P = ∫ y, F y ∂μ := by
    rw [integral_const_mul, integral_finsetSum _ (fun i _ => hFX i)]
    simp_rw [hexp]
    rw [← Finset.mul_sum, hsumA]
    field_simp
  refine ⟨hmean, ?_⟩
  -- the mean-square error is the variance
  set Q : Ω → ℝ := fun ω => (1 / n : ℝ) * ∑ i, F (X i ω) with hQ
  have hQm : AEMeasurable Q P := by
    refine (Measurable.const_mul ?_ _).aemeasurable
    exact Finset.measurable_sum _ fun i _ => hFm.comp (hXm i)
  have hvar : variance Q P = ∫ ω, (Q ω - ∫ y, F y ∂μ) ^ 2 ∂P := by
    rw [variance_eq_integral hQm]
    congr 1
    funext ω
    rw [show (∫ ω, Q ω ∂P) = ∫ y, F y ∂μ from hmean]
  rw [← hvar]
  have hsum_fun : (fun ω => ∑ i, F (X i ω)) = ∑ i, (fun ω => F (X i ω)) := by
    funext ω; simp [Finset.sum_apply]
  have hvar2 : variance Q P = (1 / n : ℝ) ^ 2 * ∑ i, variance (fun ω => F (X i ω)) P := by
    rw [hQ, variance_const_mul, hsum_fun, IndepFun.variance_sum (fun i _ => hFXL2 i)]
    intro i _ j _ hij
    exact (hXind.indepFun hij).comp hFm hFm
  rw [hvar2]
  gcongr with i _
  -- Popoviciu on each cell
  have hae : ∀ᵐ ω ∂P, X i ω ∈ A i := by
    have h0 : (P.map (X i)) (A i)ᶜ = 0 := by
      rw [hXlaw i, Measure.smul_apply, Measure.restrict_apply (hAm i).compl]
      simp
    rw [Measure.map_apply (hXm i) (hAm i).compl] at h0
    exact measure_eq_zero_iff_ae_notMem.mp h0 |>.mono fun ω hω => by simpa using hω
  obtain ⟨ω0, hω0⟩ := hae.exists
  set x0 := X i ω0
  have hbd : ∀ᵐ ω ∂P, F (X i ω) ∈ Icc (F x0 - c i) (F x0 + c i) := by
    filter_upwards [hae] with ω hω
    have := hFc i _ hω x0 hω0
    rw [abs_le] at this
    constructor <;> linarith
  have := variance_le_sq_of_bounded hbd (hFm.comp (hXm i)).aemeasurable
  calc variance (fun ω => F (X i ω)) P ≤ ((F x0 + c i - (F x0 - c i)) / 2) ^ 2 := this
    _ = c i ^ 2 := by ring


lemma abs_preintJump_le {Y : Type*} {ν : Measure ℝ} [IsProbabilityMeasure ν]
    {φ θ : ℝ → Y → ℝ} {B : ℝ} (hθB : ∀ t y, |θ t y| ≤ B) (y : Y) :
    |preintJump ν φ θ y| ≤ B := by
  rw [preintJump, ← Real.norm_eq_abs]
  have h := norm_integral_le_of_norm_le_const (μ := ν) (C := B)
    (f := fun t => {s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t) (ae_of_all _ fun t => by
      rw [Real.norm_eq_abs]
      by_cases h : 0 < φ t y
      · rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y} from h)]; exact hθB t y
      · rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y} from h), abs_zero]
        exact (abs_nonneg _).trans (hθB t y))
  simpa using h

theorem jump_stratified {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
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
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    (G : Finset ι) {D : ι → ℝ}
    (hAD : ∀ i ∈ G, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ D i)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i)) :
    (∫ ω, (1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) ∂P =
        ∫ y, preintJump ν φ θ y ∂μ) ∧
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      (1 / Fintype.card ι : ℝ) ^ 2 * (∑ i ∈ G, (Lθ * D i + 4 * ρ * B * √(L * D i / α)) ^ 2 +
        ∑ i ∈ Gᶜ, (2 * B) ^ 2) := by
  set F := preintJump ν φ θ with hF
  have hhol := jump_holder hρ hν hα hφ hφL hθm hθB hθL
  -- continuity, hence measurability, of the preintegrated function
  have hcont : Continuous F := by
    rw [continuous_iff_continuousAt]
    intro y0
    rw [ContinuousAt, tendsto_iff_dist_tendsto_zero]
    have hg : Continuous fun d : ℝ => Lθ * d + 4 * ρ * B * √(L * d / α) := by fun_prop
    have h1 : Tendsto (fun y => dist y y0) (𝓝 y0) (𝓝 0) := by
      simpa using (tendsto_id.dist (tendsto_const_nhds (x := y0)) :
        Tendsto (fun y => dist y y0) (𝓝 y0) (𝓝 (dist y0 y0)))
    have h2 := (hg.tendsto 0).comp h1
    simp only [Function.comp_def, mul_zero, zero_div, Real.sqrt_zero, add_zero] at h2
    refine squeeze_zero (fun y => dist_nonneg) (fun y => ?_) h2
    rw [Real.dist_eq]
    exact hhol y y0
  have hFm : Measurable F := hcont.measurable
  set c : ι → ℝ := fun i => if i ∈ G then Lθ * D i + 4 * ρ * B * √(L * D i / α) else 2 * B
  have hFc : ∀ i, ∀ x ∈ A i, ∀ x' ∈ A i, |F x - F x'| ≤ c i := by
    intro i x hx x' hx'
    by_cases hi : i ∈ G
    · simp only [c, hi, ite_true]
      have hd := hAD i hi x hx x' hx'
      have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0 x)
      refine (hhol x x').trans ?_
      gcongr
    · simp only [c, hi, ite_false]
      have h1 := abs_preintJump_le (ν := ν) (φ := φ) hθB x
      have h2 := abs_preintJump_le (ν := ν) (φ := φ) hθB x'
      calc |F x - F x'| ≤ |F x| + |F x'| := abs_sub _ _
        _ ≤ 2 * B := by linarith
  obtain ⟨hmean, hmse⟩ := stratified_mse hAm hAd hAcov hXm hXind hXlaw hFm
    (abs_preintJump_le hθB) hFc
  refine ⟨hmean, hmse.trans (le_of_eq ?_)⟩
  congr 1
  rw [← Finset.sum_add_sum_compl G]
  congr 1
  · refine Finset.sum_congr rfl fun i hi => ?_
    simp only [c, hi, ite_true]
  · refine Finset.sum_congr rfl fun i hi => ?_
    rw [Finset.mem_compl] at hi
    simp only [c, hi, ite_false]

end Cubature.Aux

namespace Cubature.Aux

theorem preintJump_fubini {Y : Type*} [MeasurableSpace Y] {μ : Measure Y} [IsFiniteMeasure μ]
    {ν : Measure ℝ} [IsFiniteMeasure ν] {φ θ : ℝ → Y → ℝ}
    (hφ : Measurable (Function.uncurry φ)) (hθ : Measurable (Function.uncurry θ)) {B : ℝ}
    (hθB : ∀ t y, |θ t y| ≤ B) :
    ∫ y, preintJump ν φ θ y ∂μ =
      ∫ p : ℝ × Y, {q : ℝ × Y | 0 < φ q.1 q.2}.indicator (fun q => θ q.1 q.2) p ∂(ν.prod μ) := by
  have hS : MeasurableSet {q : ℝ × Y | 0 < φ q.1 q.2} :=
    measurableSet_lt measurable_const hφ
  have hint : Integrable ({q : ℝ × Y | 0 < φ q.1 q.2}.indicator (fun q => θ q.1 q.2))
      (ν.prod μ) := by
    refine Integrable.of_bound ((hθ.indicator hS).aestronglyMeasurable) B
      (ae_of_all _ fun q => ?_)
    rw [Real.norm_eq_abs]
    by_cases h : q ∈ {q : ℝ × Y | 0 < φ q.1 q.2}
    · rw [indicator_of_mem h]; exact hθB _ _
    · rw [indicator_of_notMem h, abs_zero]; exact (abs_nonneg _).trans (hθB 0 q.2)
  rw [integral_prod_symm _ hint]
  rfl

end Cubature.Aux
