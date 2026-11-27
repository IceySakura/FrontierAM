import Theory.Defs
import Theory.Preint
import Theory.Stratified
import Theory.Cube

/-!
# Faster stratified rate under a tube (Minkowski-content type) condition
(auxiliary proofs for R7)
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology
open scoped ENNReal

namespace Cubature.Aux

/-- Stratified sampling with a pointwise oscillation weight `g`. -/
theorem stratified_mse_weighted {Ω Y ι : Type*} [MeasurableSpace Ω] [MeasurableSpace Y]
    [Fintype ι] {P : Measure Ω} [IsProbabilityMeasure P] {μ : Measure Y}
    {A : ι → Set Y} (hAm : ∀ i, MeasurableSet (A i))
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i))
    {F : Y → ℝ} (hFm : Measurable F) {B : ℝ} (hFB : ∀ x, |F x| ≤ B)
    {g : Y → ℝ} (hgm : Measurable g) {C : ℝ} (hgC : ∀ x, |g x| ≤ C)
    (hFg : ∀ i, ∀ x ∈ A i, ∀ x' ∈ A i, |F x - F x'| ≤ g x) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, F (X i ω) - ∫ y, F y ∂μ) ^ 2 ∂P ≤
      (1 / Fintype.card ι : ℝ) * ∫ y, g y ^ 2 ∂μ := by
  set n := Fintype.card ι with hn_def
  obtain ⟨hmean, -⟩ := stratified_mse (c := fun _ => 2 * B) hAm hAd hAcov hXm hXind hXlaw hFm hFB
    (fun i x _ x' _ => (abs_sub _ _).trans (by linarith [hFB x, hFB x']))
  rcases Nat.eq_zero_or_pos n with hn | hn
  · have hι : IsEmpty ι := Fintype.card_eq_zero_iff.mp hn
    have : μ = 0 := by
      simpa [Measure.measure_univ_eq_zero] using hAcov
    subst this
    simp
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hFXL2 : ∀ i, MemLp (fun ω => F (X i ω)) 2 P := fun i =>
    MemLp.of_bound (hFm.comp (hXm i)).aestronglyMeasurable B
      (ae_of_all _ fun ω => by rw [Real.norm_eq_abs]; exact hFB _)
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
  have hg2int : ∀ i, IntegrableOn (fun y => g y ^ 2) (A i) μ := fun i =>
    IntegrableOn.of_bound (hμA i) (hgm.pow_const 2).aestronglyMeasurable (C ^ 2)
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC x) 2)
  have hsumA : ∑ i, ∫ y in A i, g y ^ 2 ∂μ = ∫ y, g y ^ 2 ∂μ := by
    rw [← integral_iUnion_fintype hAm hAd hg2int]
    rw [setIntegral_congr_set (ae_eq_univ.mpr hAcov), setIntegral_univ]
  -- variance representation (as in `stratified_mse`)
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
  -- each cell
  have hcell : ∀ i, variance (fun ω => F (X i ω)) P ≤ n * ∫ y in A i, g y ^ 2 ∂μ := by
    intro i
    have hae : ∀ᵐ ω ∂P, X i ω ∈ A i := by
      have h0 : (P.map (X i)) (A i)ᶜ = 0 := by
        rw [hXlaw i, Measure.smul_apply, Measure.restrict_apply (hAm i).compl]
        simp
      rw [Measure.map_apply (hXm i) (hAm i).compl] at h0
      exact measure_eq_zero_iff_ae_notMem.mp h0 |>.mono fun ω hω => by simpa using hω
    obtain ⟨ω0, hω0⟩ := hae.exists
    set x0 := X i ω0
    have hZ : AEStronglyMeasurable (fun ω => F (X i ω)) P :=
      (hFm.comp (hXm i)).aestronglyMeasurable
    rw [← variance_sub_const hZ (F x0)]
    refine (variance_le_expectation_sq (hZ.sub aestronglyMeasurable_const)).trans ?_
    have hgX : Integrable (fun ω => g (X i ω) ^ 2) P :=
      Integrable.of_bound ((hgm.pow_const 2).comp (hXm i)).aestronglyMeasurable (C ^ 2)
        (ae_of_all _ fun ω => by
          rw [Real.norm_eq_abs, abs_pow]
          exact pow_le_pow_left₀ (abs_nonneg _) (hgC _) 2)
    calc P[(fun ω => F (X i ω) - F x0) ^ 2] ≤ ∫ ω, g (X i ω) ^ 2 ∂P := by
          refine integral_mono_ae ?_ hgX ?_
          · exact Integrable.of_bound
              (((hFm.comp (hXm i)).sub measurable_const).pow_const 2).aestronglyMeasurable
              ((2 * B) ^ 2) (ae_of_all _ fun ω => by
                simp only [Pi.pow_apply, Real.norm_eq_abs, abs_pow]
                refine pow_le_pow_left₀ (abs_nonneg _) ?_ 2
                exact (abs_sub _ _).trans (by linarith [hFB (X i ω), hFB x0]))
          · filter_upwards [hae] with ω hω
            simp only [Pi.pow_apply]
            have h1 := hFg i _ hω x0 hω0
            have h2 : 0 ≤ g (X i ω) := (abs_nonneg _).trans h1
            calc (F (X i ω) - F x0) ^ 2 = |F (X i ω) - F x0| ^ 2 := (sq_abs _).symm
              _ ≤ g (X i ω) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) h1 2
      _ = n * ∫ y in A i, g y ^ 2 ∂μ := by
          rw [← integral_map (hXm i).aemeasurable (hgm.pow_const 2).aestronglyMeasurable,
            hXlaw i, integral_smul_measure]
          simp
  rw [hvar2]
  calc (1 / n : ℝ) ^ 2 * ∑ i, variance (fun ω => F (X i ω)) P
      ≤ (1 / n : ℝ) ^ 2 * ∑ i, (n * ∫ y in A i, g y ^ 2 ∂μ) := by
        gcongr with i _
        exact hcell i
    _ = (1 / n : ℝ) * ∫ y, g y ^ 2 ∂μ := by
        rw [← Finset.mul_sum, hsumA]
        field_simp

end Cubature.Aux

namespace Cubature.Aux

/-- The pointwise form of R2 before applying the sublevel estimate. -/
lemma jump_diff_le {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ} [IsProbabilityMeasure ν]
    {φ θ : ℝ → Y → ℝ} {L Lθ B : ℝ} (hcont : ∀ y, Continuous (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y') (y y' : Y) :
    |preintJump ν φ θ y - preintJump ν φ θ y'| ≤
      Lθ * dist y y' + B * ν.real {t : ℝ | |φ t y| ≤ L * dist y y'} := by
  set δ := dist y y'
  have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0 y)
  have hmeasA : ∀ z, MeasurableSet {s : ℝ | 0 < φ s z} := fun z =>
    measurableSet_lt measurable_const (hcont z).measurable
  set S := {t : ℝ | |φ t y| ≤ L * δ} with hS
  have hSm : MeasurableSet S :=
    measurableSet_le ((hcont y).measurable.abs) measurable_const
  have hint : ∀ z, Integrable (fun t => {s : ℝ | 0 < φ s z}.indicator (fun s => θ s z) t) ν := by
    intro z
    refine Integrable.of_bound (((hθm z).indicator (hmeasA z)).aestronglyMeasurable) B
      (ae_of_all _ fun t => ?_)
    rw [Real.norm_eq_abs]
    by_cases h : 0 < φ t z
    · rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s z} from h)]; exact hθB t z
    · rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s z} from h), abs_zero]; exact hB
  have hpt : ∀ t, |{s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
      {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t| ≤ Lθ * δ + B * S.indicator 1 t := by
    intro t
    have hθ1 := hθL t y y'
    have hφ1 := hφL t y y'
    by_cases h1 : 0 < φ t y <;> by_cases h2 : 0 < φ t y'
    · rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y'} from h2)]
      have : 0 ≤ B * S.indicator 1 t := mul_nonneg hB (indicator_nonneg (fun _ _ => zero_le_one) t)
      linarith
    · have htS : t ∈ S := by
        show |φ t y| ≤ L * δ
        rw [abs_of_pos h1]; push Not at h2
        have := (abs_le.mp hφ1).2; linarith
      rw [indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y'} from h2), indicator_of_mem htS]
      simp only [sub_zero, Pi.one_apply, mul_one]
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      linarith [hθB t y]
    · have htS : t ∈ S := by
        show |φ t y| ≤ L * δ
        push Not at h1
        rw [abs_of_nonpos h1]
        have := (abs_le.mp hφ1).1; linarith
      rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_mem (show t ∈ {s : ℝ | 0 < φ s y'} from h2), indicator_of_mem htS]
      simp only [zero_sub, abs_neg, Pi.one_apply, mul_one]
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      linarith [hθB t y']
    · rw [indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y} from h1),
        indicator_of_notMem (show t ∉ {s : ℝ | 0 < φ s y'} from h2)]
      have : 0 ≤ B * S.indicator 1 t := mul_nonneg hB (indicator_nonneg (fun _ _ => zero_le_one) t)
      have : 0 ≤ Lθ * δ := (abs_nonneg _).trans hθ1
      simp only [sub_self, abs_zero]
      linarith
  have hI : Integrable (fun t => B * S.indicator (1 : ℝ → ℝ) t) ν :=
    ((integrable_const (1:ℝ)).indicator hSm).const_mul B
  calc |preintJump ν φ θ y - preintJump ν φ θ y'|
      = |∫ t, ({s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
          {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t) ∂ν| := by
        rw [preintJump, preintJump, integral_sub (hint y) (hint y')]
    _ ≤ ∫ t, |{s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t -
          {s : ℝ | 0 < φ s y'}.indicator (fun s => θ s y') t| ∂ν :=
        abs_integral_le_integral_abs
    _ ≤ ∫ t, (Lθ * δ + B * S.indicator 1 t) ∂ν := by
        refine integral_mono ((hint y).sub (hint y')).abs ?_ hpt
        exact (integrable_const _).add hI
    _ = Lθ * δ + B * ν.real S := by
        rw [integral_add (integrable_const _) hI, integral_const, integral_const_mul,
          integral_indicator_one hSm]
        simp

theorem jump_stratified_tube {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
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
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 ≤ h) (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {K : ℝ} (hK0 : 0 ≤ K)
    (hK : (ν.prod μ) {p : ℝ × Y | |φ p.1 p.2| ≤ L * h} ≤ ENNReal.ofReal (K * (L * h)))
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + 4 * ρ * K * B ^ 2 * (L * h) * √(L * h / α)) := by
  set F := preintJump ν φ θ with hF
  have hcontφ : ∀ z, Continuous (φ · z) := fun z => continuous_of_strong hα (hφ z)
  have hhol := jump_holder hρ hν hα hφ hφL hθm hθB hθL
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
  have hB : ∀ y, 0 ≤ B := fun y => (abs_nonneg _).trans (hθB 0 y)
  set S := {p : ℝ × Y | |φ p.1 p.2| ≤ L * h} with hS
  have hSm : MeasurableSet S :=
    measurableSet_le (continuous_abs.measurable.comp hφm) measurable_const
  set m : Y → ℝ := fun y => ν.real {t : ℝ | |φ t y| ≤ L * h} with hm
  have hm_eq : ∀ y, m y = (ν ((fun x => (x, y)) ⁻¹' S)).toReal := fun y => rfl
  have hm_meas : Measurable m := by
    have := (measurable_measure_prodMk_right (μ := ν) hSm).ennreal_toReal
    exact this
  have hm0 : ∀ y, 0 ≤ m y := fun y => measureReal_nonneg
  have hm1 : ∀ y, m y ≤ 1 := fun y => measureReal_le_one
  have hLh : 0 ≤ L * h := mul_nonneg hL hh
  set M0 := ρ * (4 * √(L * h / α)) with hM0
  have hmM : ∀ y, m y ≤ M0 := fun y => measureReal_sublevel_le hρ hν hα hLh (hφ y)
  set g : Y → ℝ := fun y => Lθ * h + B * m y with hg
  have hgm : Measurable g := measurable_const.add (measurable_const.mul hm_meas)
  have hgC : ∀ y, |g y| ≤ Lθ * h + B := by
    intro y
    have := hB y
    rw [abs_of_nonneg (by have := hm0 y; positivity)]
    have : B * m y ≤ B * 1 := mul_le_mul_of_nonneg_left (hm1 y) (hB y)
    simp only [g]; linarith
  have hFg : ∀ i, ∀ x ∈ A i, ∀ x' ∈ A i, |F x - F x'| ≤ g x := by
    intro i x hx x' hx'
    have hd := hAD i x hx x' hx'
    refine (jump_diff_le hcontφ hφL hθm hθB hθL x x').trans ?_
    have h1 : Lθ * dist x x' ≤ Lθ * h := mul_le_mul_of_nonneg_left hd hLθ
    have h2 : ν.real {t : ℝ | |φ t x| ≤ L * dist x x'} ≤ ν.real {t : ℝ | |φ t x| ≤ L * h} := by
      refine measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)
      have ht' : |φ t x| ≤ L * dist x x' := ht
      show |φ t x| ≤ L * h
      exact ht'.trans (mul_le_mul_of_nonneg_left hd hL)
    have h3 := mul_le_mul_of_nonneg_left h2 (hB x)
    simp only [g, m]
    linarith
  have main := stratified_mse_weighted hAm hAd hAcov hXm hXind hXlaw hFm
    (abs_preintJump_le hθB) hgm hgC hFg
  refine main.trans ?_
  -- bound `∫ g²`
  have hint_m : Integrable m μ :=
    Integrable.of_bound hm_meas.aestronglyMeasurable 1
      (ae_of_all _ fun y => by rw [Real.norm_eq_abs, abs_of_nonneg (hm0 y)]; exact hm1 y)
  have hpt : ∀ y, g y ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * m y := by
    intro y
    have h1 : (Lθ * h + B * m y) ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * (B * m y) ^ 2 := by
      nlinarith [sq_nonneg (Lθ * h - B * m y)]
    have h2 : (B * m y) ^ 2 ≤ B ^ 2 * M0 * m y := by
      have : m y * m y ≤ M0 * m y := mul_le_mul_of_nonneg_right (hmM y) (hm0 y)
      have hB2 : 0 ≤ B ^ 2 := sq_nonneg B
      calc (B * m y) ^ 2 = B ^ 2 * (m y * m y) := by ring
        _ ≤ B ^ 2 * (M0 * m y) := mul_le_mul_of_nonneg_left this hB2
        _ = B ^ 2 * M0 * m y := by ring
    simp only [g]; linarith
  have hint_rhs : Integrable (fun y => 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * m y) μ :=
    (integrable_const _).add (hint_m.const_mul _)
  have hint_g2 : Integrable (fun y => g y ^ 2) μ :=
    Integrable.of_bound (hgm.pow_const 2).aestronglyMeasurable ((Lθ * h + B) ^ 2)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC y) 2)
  have hmint : ∫ y, m y ∂μ ≤ K * (L * h) := by
    have e : ∫ y, m y ∂μ = ((ν.prod μ) S).toReal := by
      simp_rw [hm_eq]
      rw [integral_toReal (measurable_measure_prodMk_right (μ := ν) hSm).aemeasurable
        (ae_of_all _ fun y => measure_lt_top _ _), Measure.prod_apply_symm hSm]
    rw [e]
    exact ENNReal.toReal_le_of_le_ofReal (by positivity) hK
  have hg2 : ∫ y, g y ^ 2 ∂μ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * (K * (L * h)) := by
    calc ∫ y, g y ^ 2 ∂μ ≤ ∫ y, (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * m y) ∂μ :=
          integral_mono hint_g2 hint_rhs hpt
      _ = 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * ∫ y, m y ∂μ := by
          rw [integral_add (integrable_const _) (hint_m.const_mul _), integral_const,
            integral_const_mul]
          simp
      _ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * (K * (L * h)) := by
          have : 0 ≤ 2 * B ^ 2 * M0 := by
            have : 0 ≤ M0 := by positivity
            positivity
          gcongr
  have hn0 : 0 ≤ (1 / Fintype.card ι : ℝ) := by positivity
  calc (1 / Fintype.card ι : ℝ) * ∫ y, g y ^ 2 ∂μ
      ≤ (1 / Fintype.card ι : ℝ) * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * M0 * (K * (L * h))) :=
        mul_le_mul_of_nonneg_left hg2 hn0
    _ = 2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + 4 * ρ * K * B ^ 2 * (L * h) * √(L * h / α)) := by
        rw [hM0]; ring

end Cubature.Aux

namespace Cubature.Aux

lemma cube_prob (s : ℕ) :
    IsProbabilityMeasure (volume.restrict (Set.pi univ fun _ : Fin s => Ico (0 : ℝ) 1)) := by
  constructor
  rw [Measure.restrict_apply MeasurableSet.univ, univ_inter, Real.volume_pi_Ico]
  simp

theorem jump_jittered_cube_tube {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
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
    (hXlaw : ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      2 / (m : ℝ) ^ s * (Lθ ^ 2 / m ^ 2 + 4 * ρ * K * B ^ 2 * (L / m) * √(L / (α * m))) := by
  set cube := Set.pi univ (fun _ : Fin s => Ico (0 : ℝ) 1) with hcube
  haveI := cube_prob s
  have hcard : (Fintype.card (Fin s → Fin m) : ℝ) = (m : ℝ) ^ s := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; push_cast; ring
  have hcardE : (Fintype.card (Fin s → Fin m) : ℝ≥0∞) = (m : ℝ≥0∞) ^ s := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; push_cast; ring
  have hlaw' : ∀ k, P.map (X k) =
      (Fintype.card (Fin s → Fin m) : ℝ≥0∞) • (volume.restrict cube).restrict (cubeCell s m k) := by
    intro k
    rw [hXlaw k, hcardE, Measure.restrict_restrict (cubeCell_measurable s m k),
      Set.inter_eq_left.mpr (cubeCell_subset s m k)]
  have hcov : (volume.restrict cube) (⋃ k, cubeCell s m k)ᶜ = 0 := by
    rw [Measure.restrict_apply (MeasurableSet.iUnion fun k => cubeCell_measurable s m k).compl]
    have : (⋃ k, cubeCell s m k)ᶜ ∩ cube = ∅ := by
      ext x; simp only [mem_inter_iff, mem_compl_iff, mem_empty_iff_false, iff_false, not_and]
      intro h hx; exact h (cube_subset_iUnion s m hm hx)
    rw [this, measure_empty]
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hK' : (ν.prod (volume.restrict cube)) {p : ℝ × (Fin s → ℝ) | |φ p.1 p.2| ≤ L * (1 / m)}
      ≤ ENNReal.ofReal (K * (L * (1 / m))) := by
    have e : L * (1 / (m : ℝ)) = L / m := by ring
    rw [e]; exact hK
  have main := jump_stratified_tube hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL
    (μ := volume.restrict cube) (cubeCell_measurable s m) (cubeCell_disjoint s m) hcov
    (h := 1 / m) (by positivity) (fun k => cubeCell_dist s m hm k) hK0 hK' hXm hXind hlaw'
  rw [hcard] at main
  refine main.trans (le_of_eq ?_)
  have e1 : L * (1 / (m : ℝ)) / α = L / (α * m) := by field_simp
  have e2 : L * (1 / (m : ℝ)) = L / m := by ring
  rw [e1, e2]
  ring

end Cubature.Aux

namespace Cubature.Aux

/-- Tube condition for GKS Example 1 lifted to `[0,1)^{s+1}`: `φ(t, y) = y₀ - t²`. -/
theorem gks_tube (s : ℕ) {ν : Measure ℝ} [IsProbabilityMeasure ν] (η : ℝ) (hη : 0 ≤ η) :
    (ν.prod (volume.restrict (Set.pi univ fun _ : Fin (s + 1) => Ico (0 : ℝ) 1)))
      {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η} ≤ ENNReal.ofReal (2 * η) := by
  set cube := Set.pi univ (fun _ : Fin (s + 1) => Ico (0 : ℝ) 1)
  have hcm : MeasurableSet cube := MeasurableSet.univ_pi fun _ => measurableSet_Ico
  have hSm : MeasurableSet {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η} := by
    refine measurableSet_le ?_ measurable_const
    fun_prop
  rw [Measure.prod_apply hSm]
  have hsec : ∀ t : ℝ, (volume.restrict cube) (Prod.mk t ⁻¹'
      {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η}) ≤ ENNReal.ofReal (2 * η) := by
    intro t
    set a : Fin (s + 1) → ℝ := fun j => if j = 0 then t ^ 2 - η else 0
    set b : Fin (s + 1) → ℝ := fun j => if j = 0 then t ^ 2 + η else 1
    have hsub : (Prod.mk t ⁻¹' {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η}) ∩ cube ⊆
        Icc a b := by
      rintro y ⟨hy1, hy2⟩
      have hy1' : |y 0 - t ^ 2| ≤ η := hy1
      have hy2' := Set.mem_univ_pi.mp hy2
      constructor
      · intro j
        by_cases hj : j = 0
        · subst hj; simp only [a, ite_true]; linarith [(abs_le.mp hy1').1]
        · simp only [a, hj, ite_false]; exact (hy2' j).1
      · intro j
        by_cases hj : j = 0
        · subst hj; simp only [b, ite_true]; linarith [(abs_le.mp hy1').2]
        · simp only [b, hj, ite_false]; exact (hy2' j).2.le
    rw [Measure.restrict_apply' hcm]
    refine (measure_mono hsub).trans (le_of_eq ?_)
    rw [Real.volume_Icc_pi, Fin.prod_univ_succ]
    simp only [a, b, ite_true, Fin.succ_ne_zero, if_false, sub_zero]
    rw [show t ^ 2 + η - (t ^ 2 - η) = 2 * η by ring]
    simp
  calc ∫⁻ t, (volume.restrict cube) (Prod.mk t ⁻¹'
        {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ η}) ∂ν
      ≤ ∫⁻ _ : ℝ, ENNReal.ofReal (2 * η) ∂ν := lintegral_mono hsec
    _ = ENNReal.ofReal (2 * η) := by simp

/-- GKS Example 1 lifted to `[0,1)^{s+1}`: all hypotheses of R7 hold, giving an explicit rate. -/
theorem gks_example_jittered {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin (s + 1) → Fin m) → Ω → (Fin (s + 1) → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ (s + 1)) • volume.restrict (cubeCell (s + 1) m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ (s + 1)) * ∑ k, preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) (X k ω) -
        ∫ y in Set.pi univ (fun _ : Fin (s + 1) => Ico (0 : ℝ) 1), preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) y) ^ 2 ∂P ≤
      16 / (√(2 * π) * (m : ℝ) ^ (s + 2)) * √(1 / (2 * m)) := by
  have hconc : ∀ z : Fin (s + 1) → ℝ,
      StrongConcaveOn univ 2 (fun t : ℝ => z 0 - t ^ 2) := by
    intro z
    rw [strongConcaveOn_iff_convex]
    have : (fun t : ℝ => z 0 - t ^ 2 + 2 / 2 * ‖t‖ ^ 2) = fun _ => z 0 := by
      funext t; rw [Real.norm_eq_abs, sq_abs]; ring
    rw [this]
    exact concaveOn_const _ convex_univ
  have hφm : Measurable (Function.uncurry fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) :=
    ((measurable_pi_apply 0).comp measurable_snd).sub (measurable_fst.pow_const 2)
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hK : ((gaussianReal 0 1).prod (volume.restrict (Set.pi univ fun _ => Ico (0 : ℝ) 1)))
      {p : ℝ × (Fin (s + 1) → ℝ) | |p.2 0 - p.1 ^ 2| ≤ 1 / m} ≤ ENNReal.ofReal (2 * (1 / m)) :=
    gks_tube s (1 / m) (by positivity)
  have main := jump_jittered_cube_tube (s := s + 1) (Ω := Ω) hm (ρ := (√(2 * π))⁻¹)
    (by positivity) gaussianReal_le_volume
    (φ := fun t y => y 0 - t ^ 2) (θ := fun _ _ => (1 : ℝ)) (α := 2) (L := 1) (Lθ := 0) (B := 1)
    (by norm_num) (by norm_num) le_rfl (fun z => Or.inr (hconc z)) hφm
    (fun t y y' => by
      have := dist_le_pi_dist y y' 0
      rw [Real.dist_eq] at this
      rw [show y 0 - t ^ 2 - (y' 0 - t ^ 2) = y 0 - y' 0 by ring, one_mul]; exact this)
    (fun _ => measurable_const) (fun _ _ => by simp) (fun _ _ _ => by simp)
    (K := 2) (by norm_num) hK hXm hXind hXlaw
  refine main.trans (le_of_eq ?_)
  have e : (1 : ℝ) / (2 * m) = 1 / (2 * m) := rfl
  field_simp
  ring

end Cubature.Aux
