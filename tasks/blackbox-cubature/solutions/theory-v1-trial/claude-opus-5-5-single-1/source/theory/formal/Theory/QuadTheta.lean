import Theory.Defs
import Theory.Quadric
import Theory.QuadStrat

/-!
# Quadric jump surfaces with a Lipschitz amplitude `θ` (auxiliary proofs for R10′)
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology
open scoped ENNReal InnerProductSpace symmDiff

namespace Cubature.Aux

lemma gauss_symmDiff_Ioo (l r l' r' : ℝ) :
    (gaussianReal 0 1).real (Ioo l r ∆ Ioo l' r') ≤ (√(2 * π))⁻¹ * (|l - l'| + |r - r'|) := by
  set γ := gaussianReal 0 1
  have hsub : Ioo l r ∆ Ioo l' r' ⊆ Icc (min l l') (max l l') ∪ Icc (min r r') (max r r') := by
    intro t ht
    rcases ht with ⟨⟨h1, h2⟩, h3⟩ | ⟨⟨h1, h2⟩, h3⟩
    · simp only [mem_Ioo, not_and_or, not_lt] at h3
      rcases h3 with h3 | h3
      · exact Or.inl ⟨le_trans (min_le_left _ _) h1.le, le_trans h3 (le_max_right _ _)⟩
      · exact Or.inr ⟨le_trans (min_le_right _ _) h3, le_trans h2.le (le_max_left _ _)⟩
    · simp only [mem_Ioo, not_and_or, not_lt] at h3
      rcases h3 with h3 | h3
      · exact Or.inl ⟨le_trans (min_le_right _ _) h1.le, le_trans h3 (le_max_left _ _)⟩
      · exact Or.inr ⟨le_trans (min_le_left _ _) h3, le_trans h2.le (le_max_right _ _)⟩
  have h1 : γ.real (Ioo l r ∆ Ioo l' r') ≤
      γ.real (Icc (min l l') (max l l')) + γ.real (Icc (min r r') (max r r')) :=
    (measureReal_mono hsub).trans (measureReal_union_le _ _)
  have h2 := gauss_Icc_le (min l l') (max l l')
  have h3 := gauss_Icc_le (min r r') (max r r')
  rw [max_sub_min_eq_abs', abs_abs] at h2 h3
  rw [mul_add]
  linarith

/-- The Gaussian mass of the symmetric difference of two positivity sets (concave case). -/
lemma quad_symmDiff_of_neg {a β γ β' γ' : ℝ} (ha : a < 0) :
    (gaussianReal 0 1).real ({t : ℝ | 0 < a * t ^ 2 + β * t + γ} ∆
        {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'}) ≤
      (|β - β'| + |√(max (β ^ 2 - 4 * a * γ) 0) - √(max (β' ^ 2 - 4 * a * γ') 0)|) /
        (√(2 * π) * |a|) := by
  rw [quad_pos_set_of_neg ha, quad_pos_set_of_neg ha]
  refine (gauss_symmDiff_Ioo _ _ _ _).trans ?_
  set D := β ^ 2 - 4 * a * γ
  set D' := β' ^ 2 - 4 * a * γ'
  have habs : 0 < |a| := abs_pos.mpr ha.ne
  have hm : |-β / (2 * a) - -β' / (2 * a)| = |β - β'| / (2 * |a|) := by
    rw [show -β / (2 * a) - -β' / (2 * a) = -(β - β') / (2 * a) by ring, abs_div, abs_neg,
      abs_mul, abs_two]
  have hw : |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| =
      |√(max D 0) - √(max D' 0)| / (2 * |a|) := by
    rw [← sub_div, abs_div, abs_of_pos (by positivity : (0:ℝ) < 2 * |a|)]
  have e1 : |(-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))|
      ≤ |-β / (2 * a) - -β' / (2 * a)| +
        |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| := by
    rw [show (-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))
      = (-β / (2 * a) - -β' / (2 * a)) - (√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|))
      by ring]
    exact abs_sub _ _
  have e2 : |(-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))|
      ≤ |-β / (2 * a) - -β' / (2 * a)| +
        |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| := by
    rw [show (-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))
      = (-β / (2 * a) - -β' / (2 * a)) + (√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|))
      by ring]
    exact abs_add_le _ _
  rw [hm, hw] at e1 e2
  have hsum : |(-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))|
      + |(-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))|
      ≤ (|β - β'| + |√(max D 0) - √(max D' 0)|) / |a| := by
    have : (|β - β'| + |√(max D 0) - √(max D' 0)|) / |a| =
        2 * (|β - β'| / (2 * |a|)) + 2 * (|√(max D 0) - √(max D' 0)| / (2 * |a|)) := by
      field_simp
    rw [this]; linarith
  calc (√(2 * π))⁻¹ * (_ + _) ≤ (√(2 * π))⁻¹ * ((|β - β'| + |√(max D 0) - √(max D' 0)|) / |a|) := by
        gcongr
    _ = (|β - β'| + |√(max D 0) - √(max D' 0)|) / (√(2 * π) * |a|) := by field_simp

/-- Symmetric difference, general sign of the leading coefficient. -/
theorem quad_symmDiff {a β γ β' γ' : ℝ} (ha : a ≠ 0) :
    (gaussianReal 0 1).real ({t : ℝ | 0 < a * t ^ 2 + β * t + γ} ∆
        {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'}) ≤
      (|β - β'| + |√(max (β ^ 2 - 4 * a * γ) 0) - √(max (β' ^ 2 - 4 * a * γ') 0)|) /
        (√(2 * π) * |a|) := by
  rcases lt_or_gt_of_ne ha with hneg | hpos
  · exact quad_symmDiff_of_neg hneg
  · have hn : -a < 0 := by linarith
    have key := quad_symmDiff_of_neg (β := -β) (γ := -γ) (β' := -β') (γ' := -γ') hn
    have e2 : |-β - -β'| = |β - β'| := by rw [show -β - -β' = -(β - β') by ring, abs_neg]
    have d1 : (-β') ^ 2 - 4 * -a * -γ' = β' ^ 2 - 4 * a * γ' := by ring
    have d2 : (-β) ^ 2 - 4 * -a * -γ = β ^ 2 - 4 * a * γ := by ring
    rw [e2, d1, d2, abs_neg] at key
    set S := {t : ℝ | 0 < a * t ^ 2 + β * t + γ}
    set S' := {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'}
    set T := {t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)}
    set T' := {t : ℝ | 0 < (-a) * t ^ 2 + (-β') * t + (-γ')}
    set Z := {t : ℝ | a * t ^ 2 + β * t + γ = 0}
    set Z' := {t : ℝ | a * t ^ 2 + β' * t + γ' = 0}
    have hsub : S ∆ S' ⊆ (T ∆ T') ∪ (Z ∪ Z') := by
      intro t ht
      simp only [S, S', T, T', Z, Z', mem_union, mem_symmDiff, mem_ofPred_eq] at ht ⊢
      rcases ht with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · push Not at h2
        rcases eq_or_lt_of_le h2 with h3 | h3
        · exact Or.inr (Or.inr h3)
        · exact Or.inl (Or.inr ⟨by linarith, by linarith⟩)
      · push Not at h2
        rcases eq_or_lt_of_le h2 with h3 | h3
        · exact Or.inr (Or.inl h3)
        · exact Or.inl (Or.inl ⟨by linarith, by linarith⟩)
    have hZ : (gaussianReal 0 1).real (Z ∪ Z') = 0 := by
      have h0 : (gaussianReal 0 1) (Z ∪ Z') = 0 :=
        measure_union_null (quad_zero_null ha) (quad_zero_null ha)
      rw [measureReal_def, h0, ENNReal.toReal_zero]
    calc (gaussianReal 0 1).real (S ∆ S') ≤ (gaussianReal 0 1).real ((T ∆ T') ∪ (Z ∪ Z')) :=
          measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ (gaussianReal 0 1).real (T ∆ T') + (gaussianReal 0 1).real (Z ∪ Z') :=
          measureReal_union_le _ _
      _ ≤ _ := by rw [hZ, add_zero]; exact key

/-- Indicator integrals with two amplitudes. -/
lemma indicator_integral_diff {S S' : Set ℝ} (hS : MeasurableSet S) (hS' : MeasurableSet S')
    {θ₁ θ₂ : ℝ → ℝ} (h1 : Measurable θ₁) (h2 : Measurable θ₂) {B : ℝ}
    (hB1 : ∀ t, |θ₁ t| ≤ B) (hB2 : ∀ t, |θ₂ t| ≤ B) :
    |∫ t, S.indicator θ₁ t ∂(gaussianReal 0 1) - ∫ t, S'.indicator θ₂ t ∂(gaussianReal 0 1)| ≤
      ∫ t, |θ₁ t - θ₂ t| ∂(gaussianReal 0 1) + B * (gaussianReal 0 1).real (S ∆ S') := by
  set γ := gaussianReal 0 1
  have hB : 0 ≤ B := (abs_nonneg _).trans (hB1 0)
  have hi1 : Integrable (S.indicator θ₁) γ := Integrable.of_bound
    (h1.indicator hS).aestronglyMeasurable B (ae_of_all _ fun t => by
      rw [Real.norm_eq_abs]
      by_cases ht : t ∈ S
      · rw [indicator_of_mem ht]; exact hB1 t
      · rw [indicator_of_notMem ht, abs_zero]; exact hB)
  have hi2 : Integrable (S'.indicator θ₂) γ := Integrable.of_bound
    (h2.indicator hS').aestronglyMeasurable B (ae_of_all _ fun t => by
      rw [Real.norm_eq_abs]
      by_cases ht : t ∈ S'
      · rw [indicator_of_mem ht]; exact hB2 t
      · rw [indicator_of_notMem ht, abs_zero]; exact hB)
  have hsd : MeasurableSet (S ∆ S') := hS.symmDiff hS'
  have hid : Integrable (fun t => |θ₁ t - θ₂ t|) γ := Integrable.of_bound
    (h1.sub h2).abs.aestronglyMeasurable (2 * B) (ae_of_all _ fun t => by
      rw [Real.norm_eq_abs, abs_abs]
      exact (abs_sub _ _).trans (by linarith [hB1 t, hB2 t]))
  have hI : Integrable (fun t => B * (S ∆ S').indicator (1 : ℝ → ℝ) t) γ :=
    ((integrable_const (1 : ℝ)).indicator hsd).const_mul B
  have hpt : ∀ t, |S.indicator θ₁ t - S'.indicator θ₂ t| ≤
      |θ₁ t - θ₂ t| + B * (S ∆ S').indicator 1 t := by
    intro t
    by_cases ht : t ∈ S <;> by_cases ht' : t ∈ S'
    · rw [indicator_of_mem ht, indicator_of_mem ht',
        indicator_of_notMem (show t ∉ S ∆ S' by simp [mem_symmDiff, ht, ht'])]
      simp
    · rw [indicator_of_mem ht, indicator_of_notMem ht',
        indicator_of_mem (show t ∈ S ∆ S' from Or.inl ⟨ht, ht'⟩)]
      simp only [sub_zero, Pi.one_apply, mul_one]
      linarith [hB1 t, abs_nonneg (θ₁ t - θ₂ t)]
    · rw [indicator_of_notMem ht, indicator_of_mem ht',
        indicator_of_mem (show t ∈ S ∆ S' from Or.inr ⟨ht', ht⟩)]
      simp only [zero_sub, abs_neg, Pi.one_apply, mul_one]
      linarith [hB2 t, abs_nonneg (θ₁ t - θ₂ t)]
    · rw [indicator_of_notMem ht, indicator_of_notMem ht',
        indicator_of_notMem (show t ∉ S ∆ S' by simp [mem_symmDiff, ht, ht'])]
      simp
  calc |∫ t, S.indicator θ₁ t ∂γ - ∫ t, S'.indicator θ₂ t ∂γ|
      = |∫ t, (S.indicator θ₁ t - S'.indicator θ₂ t) ∂γ| := by rw [integral_sub hi1 hi2]
    _ ≤ ∫ t, |S.indicator θ₁ t - S'.indicator θ₂ t| ∂γ := abs_integral_le_integral_abs
    _ ≤ ∫ t, (|θ₁ t - θ₂ t| + B * (S ∆ S').indicator 1 t) ∂γ :=
        integral_mono (hi1.sub hi2).abs (hid.add hI) hpt
    _ = ∫ t, |θ₁ t - θ₂ t| ∂γ + B * γ.real (S ∆ S') := by
        rw [integral_add hid hI, integral_const_mul, integral_indicator_one hsd]

end Cubature.Aux

namespace Cubature.Aux

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem quadric_theta_stratified {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace E]
    [OpensMeasurableSpace E] (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z)
    (θ : E → ℝ) {B Lθ : ℝ} (hθB : ∀ z, |θ z| ≤ B)
    (hθL : ∀ z z', |θ z - θ z'| ≤ Lθ * ‖z - z'‖)
    {ι : Type*} [Fintype ι] {μ : Measure E} [IsProbabilityMeasure μ]
    {C : ι → Set E} (hCm : ∀ i, MeasurableSet (C i))
    (hCd : Pairwise (Disjoint on C)) (hCcov : μ (⋃ i, C i)ᶜ = 0)
    {R h Dmax K : ℝ} (hR : 0 < R) (hh : 0 < h)
    (hCR : ∀ i, ∀ x ∈ C i, ‖x‖ ≤ R) (hCD : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, dist x x' ≤ h)
    (hDmax : ∀ i, ∀ x ∈ C i, D x ≤ Dmax) (hK0 : 0 ≤ K)
    (hK : μ.map D ≤ ENNReal.ofReal K • volume)
    (h2 : 2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ≤ Dmax)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → E} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (C i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump (gaussianReal 0 1)
          (fun t z => φ (z + t • u)) (fun t z => θ (z + t • u)) (X i ω) -
        ∫ x, preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u))
          (fun t z => θ (z + t • u)) x ∂μ) ^ 2 ∂P ≤
      1 / Fintype.card ι * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * ‖A‖ * h) ^ 2 +
        K * ((8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2 *
          Real.log (Dmax / (2 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h))) +
          3 * (8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h) ^ 2)) / (π * ⟪u, A u⟫_ℝ ^ 2)) := by
  set a := ⟪u, A u⟫_ℝ with ha
  set γm := gaussianReal 0 1
  set F := preintJump γm (fun t z => φ (z + t • u)) (fun t z => θ (z + t • u)) with hF
  have hapos : 0 < |a| := abs_pos.mpr hu
  have hA : 0 < ‖A‖ := by
    rcases eq_or_lt_of_le (norm_nonneg A) with h0 | h0
    · exfalso; apply hu
      have : A = 0 := norm_eq_zero.mp h0.symm
      simp [ha, this]
    · exact h0
  have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0)
  set η := 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * h with hη_def
  have hη : 0 < η := by positivity
  set S : E → Set ℝ := fun z => {t : ℝ | 0 < a * t ^ 2 + β z * t + φ z} with hSdef
  have hSm : ∀ z, MeasurableSet (S z) := fun z =>
    measurableSet_lt measurable_const (by fun_prop)
  have hθc : Continuous θ := by
    rw [continuous_iff_continuousAt]
    intro z0
    rw [ContinuousAt, tendsto_iff_dist_tendsto_zero]
    have h0 : Tendsto (fun z => Lθ * ‖z - z0‖) (𝓝 z0) (𝓝 0) := by
      have h00 : Tendsto (fun z : E => ‖z - z0‖) (𝓝 z0) (𝓝 0) := by
        have := (continuous_id.sub (continuous_const : Continuous fun _ : E => z0)).norm.tendsto z0
        simpa using this
      simpa using h00.const_mul Lθ
    refine squeeze_zero (fun z => dist_nonneg) (fun z => ?_) h0
    rw [Real.dist_eq]; exact hθL z z0
  have hθt : ∀ z, Measurable fun t : ℝ => θ (z + t • u) := fun z =>
    (hθc.comp (by fun_prop : Continuous fun t : ℝ => z + t • u)).measurable
  -- explicit form
  have hP : ∀ z, F z = ∫ t, (S z).indicator (fun t => θ (z + t • u)) t ∂γm := by
    intro z
    have hset : {t : ℝ | 0 < φ (z + t • u)} = S z := by
      ext t
      simp only [hSdef, mem_ofPred_eq]
      rw [hφ (z + t • u), quadric_line, hβ z, hφ z]
    rw [hF, preintJump, hset]
  -- increment bound
  have hinc : ∀ x x', |F x - F x'| ≤ Lθ * ‖x - x'‖ +
      B * ((|β x - β x'| + |√(max (D x) 0) - √(max (D x') 0)|) / (√(2 * π) * |a|)) := by
    intro x x'
    rw [hP x, hP x']
    refine (indicator_integral_diff (hSm x) (hSm x') (hθt x) (hθt x')
      (fun t => hθB _) (fun t => hθB _)).trans ?_
    have h1 : ∫ t : ℝ, |θ (x + t • u) - θ (x' + t • u)| ∂γm ≤ Lθ * ‖x - x'‖ := by
      have hb : ∀ t : ℝ, |θ (x + t • u) - θ (x' + t • u)| ≤ Lθ * ‖x - x'‖ := by
        intro t
        have := hθL (x + t • u) (x' + t • u)
        rwa [show x + t • u - (x' + t • u) = x - x' by abel] at this
      calc ∫ t : ℝ, |θ (x + t • u) - θ (x' + t • u)| ∂γm ≤ ∫ _t : ℝ, Lθ * ‖x - x'‖ ∂γm :=
            integral_mono_of_nonneg (ae_of_all _ fun t => abs_nonneg _) (integrable_const _)
              (ae_of_all _ hb)
        _ = Lθ * ‖x - x'‖ := by simp
    have h2 : γm.real (S x ∆ S x') ≤
        (|β x - β x'| + |√(max (D x) 0) - √(max (D x') 0)|) / (√(2 * π) * |a|) := by
      have := quad_symmDiff (a := a) (β := β x) (γ := φ x) (β' := β x') (γ' := φ x') hu
      rw [← hD x, ← hD x'] at this
      exact this
    have h3 := mul_le_mul_of_nonneg_left h2 hB
    linarith
  -- continuity
  have hβc : Continuous β := by
    have : β = fun z => ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ := funext hβ
    rw [this]; fun_prop
  have hφc : Continuous φ := by
    have : φ = fun z => ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c := funext hφ
    rw [this]; fun_prop
  have hDc : Continuous D := by
    have : D = fun z => β z ^ 2 - 4 * a * φ z := funext hD
    rw [this]; fun_prop
  have hFc : Continuous F := by
    rw [continuous_iff_continuousAt]
    intro x0
    rw [ContinuousAt, tendsto_iff_dist_tendsto_zero]
    have hbound : ∀ x, dist (F x) (F x0) ≤ Lθ * ‖x - x0‖ +
        B * ((|β x - β x0| + √|D x - D x0|) / (√(2 * π) * |a|)) := by
      intro x
      rw [Real.dist_eq]
      refine (hinc x x0).trans ?_
      have : |√(max (D x) 0) - √(max (D x0) 0)| ≤ √|D x - D x0| :=
        (abs_sqrt_sub_sqrt_le (le_max_right _ _) (le_max_right _ _)).trans
          (Real.sqrt_le_sqrt (abs_max_sub_max_le_abs _ _ _))
      gcongr
    refine squeeze_zero (fun x => dist_nonneg) hbound ?_
    have h0 : Tendsto (fun x => ‖x - x0‖) (𝓝 x0) (𝓝 0) := by
      have := (continuous_id.sub (continuous_const : Continuous fun _ : E => x0)).norm.tendsto x0
      simpa using this
    have h1 : Tendsto (fun x => |β x - β x0|) (𝓝 x0) (𝓝 0) := by
      have := ((hβc.tendsto x0).sub_const (β x0)).abs
      simpa using this
    have h2 : Tendsto (fun x => √|D x - D x0|) (𝓝 x0) (𝓝 0) := by
      have := (((hDc.tendsto x0).sub_const (D x0)).abs).sqrt
      simpa using this
    have := (h0.const_mul Lθ).add (((h1.add h2).div_const (√(2 * π) * |a|)).const_mul B)
    simpa using this
  have hFm : Measurable F := hFc.measurable
  -- weight
  set g : E → ℝ := fun x => Lθ * h + B * ((2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) /
    (√(2 * π) * |a|)) with hg
  have hden : 0 < √(2 * π) * |a| := by positivity
  have hLθ : ∀ (x x' : E), 0 ≤ Lθ * ‖x - x'‖ := fun x x' => (abs_nonneg _).trans (hθL x x')
  have hgm : Measurable g := by
    refine measurable_const.add (measurable_const.mul (Measurable.div_const
      (measurable_const.add ?_) _))
    exact (sqrtProfile_measurable.comp hDc.measurable).sqrt
  have hgC : ∀ x, |g x| ≤ |Lθ| * h + B * ((2 * ‖A‖ * h + √η) / (√(2 * π) * |a|)) := by
    intro x
    have hs : √(sqrtProfile η Dmax (D x)) ≤ √η := Real.sqrt_le_sqrt (sqrtProfile_le hη _)
    have hq : (2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) / (√(2 * π) * |a|) ≤
        (2 * ‖A‖ * h + √η) / (√(2 * π) * |a|) :=
      div_le_div_of_nonneg_right (by linarith) hden.le
    have hq0 : 0 ≤ (2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) / (√(2 * π) * |a|) :=
      div_nonneg (by positivity) hden.le
    calc |g x| ≤ |Lθ * h| + |B * ((2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) /
          (√(2 * π) * |a|))| := abs_add_le _ _
      _ = |Lθ| * h + B * ((2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) / (√(2 * π) * |a|)) := by
          rw [abs_mul, abs_of_pos hh, abs_of_nonneg (mul_nonneg hB hq0)]
      _ ≤ |Lθ| * h + B * ((2 * ‖A‖ * h + √η) / (√(2 * π) * |a|)) := by gcongr
  have hFg : ∀ i, ∀ x ∈ C i, ∀ x' ∈ C i, |F x - F x'| ≤ g x := by
    intro i x hx x' hx'
    obtain ⟨hb1, hb2⟩ := quadric_incr A b c φ hφ u hu1 β D hβ hD x x' (hCR i x hx) (hCR i x' hx')
    have hd : ‖x - x'‖ ≤ h := by rw [← dist_eq_norm]; exact hCD i x hx x' hx'
    have hbe : |β x - β x'| ≤ 2 * ‖A‖ * h := hb1.trans (by gcongr)
    have hDe : |D x - D x'| ≤ η := hb2.trans (by rw [hη_def]; gcongr)
    have hsq := sqrt_incr_sq_le hη hDe (hDmax i x hx)
    have hsq' : |√(max (D x) 0) - √(max (D x') 0)| ≤ √(sqrtProfile η Dmax (D x)) := by
      rw [sq_abs] at hsq; exact Real.abs_le_sqrt hsq
    have hLθ0 : 0 ≤ Lθ := by
      by_contra hneg; push Not at hneg
      by_cases hxx : x = x'
      · subst hxx
        -- use any other point: Lθ ‖z - z'‖ ≥ 0 forces nothing; fall back on h > 0
        have := hθL x (x + (1 : ℝ) • u)
        have hn : ‖x - (x + (1 : ℝ) • u)‖ = 1 := by
          rw [show x - (x + (1 : ℝ) • u) = -u by simp, norm_neg, hu1]
        rw [hn, mul_one] at this
        linarith [abs_nonneg (θ x - θ (x + (1 : ℝ) • u))]
      · have := hθL x (x + (1 : ℝ) • u)
        have hn : ‖x - (x + (1 : ℝ) • u)‖ = 1 := by
          rw [show x - (x + (1 : ℝ) • u) = -u by simp, norm_neg, hu1]
        rw [hn, mul_one] at this
        linarith [abs_nonneg (θ x - θ (x + (1 : ℝ) • u))]
    refine (hinc x x').trans ?_
    simp only [hg]
    have h1 : Lθ * ‖x - x'‖ ≤ Lθ * h := mul_le_mul_of_nonneg_left hd hLθ0
    have h3 : (|β x - β x'| + |√(max (D x) 0) - √(max (D x') 0)|) / (√(2 * π) * |a|) ≤
        (2 * ‖A‖ * h + √(sqrtProfile η Dmax (D x))) / (√(2 * π) * |a|) :=
      div_le_div_of_nonneg_right (by linarith) hden.le
    have h4 := mul_le_mul_of_nonneg_left h3 hB
    linarith
  have main := stratified_mse_weighted hCm hCd hCcov hXm hXind hXlaw hFm
    (abs_preintJump_le (θ := fun t z => θ (z + t • u)) (fun t z => hθB _)) hgm hgC hFg
  refine main.trans ?_
  -- bound `∫ g²`
  set Ψ := sqrtProfile η Dmax with hΨ
  have hΨ0 : ∀ r, 0 ≤ Ψ r := sqrtProfile_nonneg hη
  have hΨi : Integrable Ψ := sqrtProfile_integrable hη
  have hc2 : (√(2 * π) * |a|) ^ 2 = 2 * π * a ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (by positivity), sq_abs]
  set c0 : ℝ := 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * (2 * (2 * ‖A‖ * h) ^ 2 / (2 * π * a ^ 2)) with hc0
  set c1 : ℝ := 2 * B ^ 2 * (2 / (2 * π * a ^ 2)) with hc1
  have hpt : ∀ x, g x ^ 2 ≤ c0 + c1 * Ψ (D x) := by
    intro x
    simp only [hg]
    set q := (2 * ‖A‖ * h + √(Ψ (D x))) / (√(2 * π) * |a|)
    have hq2 : q ^ 2 ≤ (2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) / (2 * π * a ^ 2) := by
      simp only [q]
      rw [div_pow, hc2]
      gcongr
      have h1 := Real.sq_sqrt (hΨ0 (D x))
      nlinarith [sq_nonneg (2 * ‖A‖ * h - √(Ψ (D x)))]
    have : (Lθ * h + B * q) ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * q ^ 2 := by
      nlinarith [sq_nonneg (Lθ * h - B * q)]
    have hB2 : 0 ≤ 2 * B ^ 2 := by positivity
    have e : 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) /
        (2 * π * a ^ 2)) = c0 + c1 * Ψ (D x) := by
      simp only [hc0, hc1]; ring
    calc (Lθ * h + B * q) ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * q ^ 2 := this
      _ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * (2 * ‖A‖ * h) ^ 2 + 2 * Ψ (D x)) /
          (2 * π * a ^ 2)) := by gcongr
      _ = c0 + c1 * Ψ (D x) := e
  have hΨD_int : Integrable (fun x => Ψ (D x)) μ := by
    have : Integrable Ψ (μ.map D) := by
      refine Integrable.mono_measure ?_ hK
      exact hΨi.smul_measure ENNReal.ofReal_ne_top
    exact (integrable_map_measure sqrtProfile_measurable.aestronglyMeasurable
      hDc.measurable.aemeasurable).mp this
  have hint_rhs : Integrable (fun x => c0 + c1 * Ψ (D x)) μ :=
    (integrable_const _).add (hΨD_int.const_mul _)
  have hint_g2 : Integrable (fun x => g x ^ 2) μ :=
    Integrable.of_bound (hgm.pow_const 2).aestronglyMeasurable
      ((|Lθ| * h + B * ((2 * ‖A‖ * h + √η) / (√(2 * π) * |a|))) ^ 2)
      (ae_of_all _ fun x => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC x) 2)
  have hΨint : ∫ x, Ψ (D x) ∂μ ≤ K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2) := by
    calc ∫ x, Ψ (D x) ∂μ = ∫ r, Ψ r ∂(μ.map D) :=
          (integral_map hDc.measurable.aemeasurable sqrtProfile_measurable.aestronglyMeasurable).symm
      _ ≤ ∫ r, Ψ r ∂(ENNReal.ofReal K • volume) :=
          integral_mono_measure hK (ae_of_all _ hΨ0) (hΨi.smul_measure ENNReal.ofReal_ne_top)
      _ = K * ∫ r, Ψ r := by
          rw [integral_smul_measure, ENNReal.toReal_ofReal hK0, smul_eq_mul]
      _ = K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2) := by
          rw [hΨ, sqrtProfile_integral hη h2]
  have hc1n : 0 ≤ c1 := by simp only [hc1]; positivity
  have hg2 : ∫ x, g x ^ 2 ∂μ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * ‖A‖ * h) ^ 2 +
      K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) / (π * a ^ 2) := by
    have hπa : (0 : ℝ) < π * a ^ 2 := by positivity
    calc ∫ x, g x ^ 2 ∂μ ≤ ∫ x, (c0 + c1 * Ψ (D x)) ∂μ := integral_mono hint_g2 hint_rhs hpt
      _ = c0 + c1 * ∫ x, Ψ (D x) ∂μ := by
          rw [integral_add (integrable_const _) (hΨD_int.const_mul _), integral_const,
            integral_const_mul]
          simp
      _ ≤ c0 + c1 * (K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) := by gcongr
      _ = _ := by simp only [hc0, hc1]; field_simp <;> ring
  have hn0 : 0 ≤ (1 / Fintype.card ι : ℝ) := by positivity
  calc (1 / Fintype.card ι : ℝ) * ∫ x, g x ^ 2 ∂μ
      ≤ (1 / Fintype.card ι : ℝ) * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ((2 * ‖A‖ * h) ^ 2 +
          K * (η ^ 2 * Real.log (Dmax / (2 * η)) + 3 * η ^ 2)) / (π * a ^ 2)) :=
        mul_le_mul_of_nonneg_left hg2 hn0
    _ = _ := by rw [hη_def] <;> ring

end Cubature.Aux
