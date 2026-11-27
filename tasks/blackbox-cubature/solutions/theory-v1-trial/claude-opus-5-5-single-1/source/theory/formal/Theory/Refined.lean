import Theory.Defs
import Theory.Preint
import Theory.Tube
import Theory.Cube

/-!
# Refined band estimate away from tangency, and the near-Lipschitz stratified rate
(auxiliary proofs for R8)
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology
open scoped ENNReal

namespace Cubature.Aux

lemma strongConcave_mid {ψ : ℝ → ℝ} {α : ℝ} (hψ : StrongConcaveOn univ α ψ) (x u : ℝ) :
    (ψ x + ψ u) / 2 + α / 8 * (x - u) ^ 2 ≤ ψ ((x + u) / 2) := by
  have := hψ.2 (mem_univ x) (mem_univ u) (show (0 : ℝ) ≤ 1 / 2 by norm_num)
    (show (0 : ℝ) ≤ 1 / 2 by norm_num) (show (1 / 2 : ℝ) + 1 / 2 = 1 by norm_num)
  simp only [smul_eq_mul, Real.norm_eq_abs, sq_abs] at this
  have e : (1 / 2 : ℝ) * x + 1 / 2 * u = (x + u) / 2 := by ring
  rw [e] at this
  linarith

/-- One side of the band, away from the maximum. -/
lemma refined_side {ψ : ℝ → ℝ} {α η H : ℝ} (hα : 0 < α) (hψ : StrongConcaveOn univ α ψ)
    (hle : ∀ t, ψ t ≤ H) {u : ℝ} (hu : ψ u = H) (hη : 0 < η) (hH : 2 * η ≤ H)
    {x y : ℝ} (hx : |ψ x| ≤ η) (hy : |ψ y| ≤ η) (hxy : x ≤ y)
    (hside : y < u ∨ u < x) :
    α * H * (y - x) ^ 2 ≤ 24 * η ^ 2 := by
  have hconc : ConcaveOn ℝ univ ψ := hψ.concaveOn (fun r => by simp only [Pi.zero_apply]; positivity)
  have hHpos : 0 < H := by linarith
  rcases eq_or_lt_of_le hxy with h | hxy'
  · subst h; simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
      mul_zero]; positivity
  rcases hside with hyu | hux
  · -- x < y < u
    have hux : 0 < u - x := by linarith
    -- (1) distance to the maximiser
    have hm := strongConcave_mid hψ x u
    have hmid := hle ((x + u) / 2)
    have hx1 := (abs_le.mp hx).1
    have hd : α * (u - x) ^ 2 ≤ 4 * (H + η) := by
      have : (x - u) ^ 2 = (u - x) ^ 2 := by ring
      rw [this] at hm; nlinarith
    -- (2) concavity between x and u
    set b := (y - x) / (u - x) with hb_def
    set a := (u - y) / (u - x) with ha_def
    have ha : 0 ≤ a := div_nonneg (by linarith) hux.le
    have hb : 0 ≤ b := div_nonneg (by linarith) hux.le
    have hab : a + b = 1 := by rw [ha_def, hb_def, ← add_div]; field_simp; ring
    have hcomb : a • x + b • u = y := by
      simp only [smul_eq_mul]; rw [ha_def, hb_def]; field_simp; ring
    have hc := hconc.2 (mem_univ x) (mem_univ u) ha hb hab
    rw [hcomb] at hc
    simp only [smul_eq_mul] at hc
    have hy2 := (abs_le.mp hy).2
    have hbH : b * H ≤ 2 * η := by
      have e1 : a * (-η) ≤ a * ψ x := mul_le_mul_of_nonneg_left hx1 ha
      have hale : a * η ≤ η := by
        have : a ≤ 1 := by linarith
        nlinarith
      rw [hu] at hc
      linarith
    have hbu : b * (u - x) = y - x := by rw [hb_def]; field_simp
    -- (3) combine
    have h1 : (y - x) ^ 2 = b ^ 2 * (u - x) ^ 2 := by rw [← hbu]; ring
    have h2 : α * H * (y - x) ^ 2 = (b * H) * b * (α * (u - x) ^ 2) := by rw [h1]; ring
    have hb1 : b * H ≤ 2 * η := hbH
    have hbH' : b ≤ 2 * η / H := by rw [le_div_iff₀ hHpos]; linarith
    have h3 : (b * H) * b * (α * (u - x) ^ 2) ≤ (2 * η) * (2 * η / H) * (4 * (H + η)) := by
      have hbHn : 0 ≤ b * H := mul_nonneg hb hHpos.le
      have hαu : 0 ≤ α * (u - x) ^ 2 := by positivity
      calc (b * H) * b * (α * (u - x) ^ 2) ≤ (2 * η) * b * (α * (u - x) ^ 2) := by gcongr
        _ ≤ (2 * η) * (2 * η / H) * (α * (u - x) ^ 2) := by gcongr
        _ ≤ (2 * η) * (2 * η / H) * (4 * (H + η)) := by
            apply mul_le_mul_of_nonneg_left hd; positivity
    have h4 : (2 * η) * (2 * η / H) * (4 * (H + η)) ≤ 24 * η ^ 2 := by
      have : (2 * η) * (2 * η / H) * (4 * (H + η)) = 16 * η ^ 2 * ((H + η) / H) := by
        field_simp; ring
      rw [this]
      have : (H + η) / H ≤ 3 / 2 := by rw [div_le_iff₀ hHpos]; linarith
      nlinarith [sq_nonneg η]
    linarith
  · -- u < x < y
    have hyu : 0 < y - u := by linarith
    have hm := strongConcave_mid hψ y u
    have hmid := hle ((y + u) / 2)
    have hy1 := (abs_le.mp hy).1
    have hd : α * (y - u) ^ 2 ≤ 4 * (H + η) := by nlinarith
    set a := (x - u) / (y - u) with ha_def
    set b := (y - x) / (y - u) with hb_def
    have ha : 0 ≤ a := div_nonneg (by linarith) hyu.le
    have hb : 0 ≤ b := div_nonneg (by linarith) hyu.le
    have hab : a + b = 1 := by rw [ha_def, hb_def, ← add_div]; field_simp; ring
    have hcomb : a • y + b • u = x := by
      simp only [smul_eq_mul]; rw [ha_def, hb_def]; field_simp; ring
    have hc := hconc.2 (mem_univ y) (mem_univ u) ha hb hab
    rw [hcomb] at hc
    simp only [smul_eq_mul] at hc
    have hx2 := (abs_le.mp hx).2
    have hbH : b * H ≤ 2 * η := by
      have e1 : a * (-η) ≤ a * ψ y := mul_le_mul_of_nonneg_left hy1 ha
      have hale : a * η ≤ η := by
        have : a ≤ 1 := by linarith
        nlinarith
      rw [hu] at hc
      linarith
    have hbu : b * (y - u) = y - x := by rw [hb_def]; field_simp
    have h1 : (y - x) ^ 2 = b ^ 2 * (y - u) ^ 2 := by rw [← hbu]; ring
    have h2 : α * H * (y - x) ^ 2 = (b * H) * b * (α * (y - u) ^ 2) := by rw [h1]; ring
    have hbH' : b ≤ 2 * η / H := by rw [le_div_iff₀ hHpos]; linarith
    have h3 : (b * H) * b * (α * (y - u) ^ 2) ≤ (2 * η) * (2 * η / H) * (4 * (H + η)) := by
      have hαu : 0 ≤ α * (y - u) ^ 2 := by positivity
      calc (b * H) * b * (α * (y - u) ^ 2) ≤ (2 * η) * b * (α * (y - u) ^ 2) := by gcongr
        _ ≤ (2 * η) * (2 * η / H) * (α * (y - u) ^ 2) := by gcongr
        _ ≤ (2 * η) * (2 * η / H) * (4 * (H + η)) := by
            apply mul_le_mul_of_nonneg_left hd; positivity
    have h4 : (2 * η) * (2 * η / H) * (4 * (H + η)) ≤ 24 * η ^ 2 := by
      have : (2 * η) * (2 * η / H) * (4 * (H + η)) = 16 * η ^ 2 * ((H + η) / H) := by
        field_simp; ring
      rw [this]
      have : (H + η) / H ≤ 3 / 2 := by rw [div_le_iff₀ hHpos]; linarith
      nlinarith [sq_nonneg η]
    linarith

/-- **Refined band estimate.** -/
theorem volume_band_refined {ψ : ℝ → ℝ} {α η H : ℝ} (hα : 0 < α) (hψ : StrongConcaveOn univ α ψ)
    (hle : ∀ t, ψ t ≤ H) {u : ℝ} (hu : ψ u = H) (hη : 0 < η) (hH : 2 * η ≤ H) :
    volume {t : ℝ | |ψ t| ≤ η} ≤ ENNReal.ofReal (2 * √(24 * η ^ 2 / (α * H))) := by
  set S := {t : ℝ | |ψ t| ≤ η}
  have hHpos : 0 < H := by linarith
  set c := √(24 * η ^ 2 / (α * H)) with hc
  have hc0 : 0 ≤ c := Real.sqrt_nonneg _
  have hc2 : c ^ 2 = 24 * η ^ 2 / (α * H) := Real.sq_sqrt (by positivity)
  have huS : u ∉ S := by
    intro h
    have : |ψ u| ≤ η := h
    rw [hu, abs_of_pos hHpos] at this; linarith
  have hsub : S ⊆ (S ∩ Iio u) ∪ (S ∩ Ioi u) := by
    intro t ht
    rcases lt_trichotomy t u with h | h | h
    · exact Or.inl ⟨ht, h⟩
    · exact absurd (h ▸ ht) huS
    · exact Or.inr ⟨ht, h⟩
  have hdiam : ∀ T : Set ℝ, (∀ x ∈ T, ∀ y ∈ T, x ≤ y → α * H * (y - x) ^ 2 ≤ 24 * η ^ 2) →
      volume T ≤ ENNReal.ofReal c := by
    intro T hT
    refine (Real.volume_le_diam T).trans ?_
    refine Metric.ediam_le fun x hx y hy => ?_
    rw [edist_dist, Real.dist_eq]
    refine ENNReal.ofReal_le_ofReal ?_
    have key : ∀ p q : ℝ, α * H * (q - p) ^ 2 ≤ 24 * η ^ 2 → |q - p| ≤ c := by
      intro p q hpq
      have : (q - p) ^ 2 ≤ c ^ 2 := by
        rw [hc2, le_div_iff₀ (by positivity)]; linarith
      exact abs_le_of_sq_le_sq' this hc0 |> fun h => abs_le.mpr h
    rcases le_total x y with hxy | hxy
    · rw [abs_sub_comm]; exact key x y (hT x hx y hy hxy)
    · exact key y x (hT y hy x hx hxy)
  have hL : volume (S ∩ Iio u) ≤ ENNReal.ofReal c :=
    hdiam _ fun x hx y hy hxy => refined_side hα hψ hle hu hη hH hx.1 hy.1 hxy (Or.inl hy.2)
  have hR : volume (S ∩ Ioi u) ≤ ENNReal.ofReal c :=
    hdiam _ fun x hx y hy hxy => refined_side hα hψ hle hu hη hH hx.1 hy.1 hxy (Or.inr hx.2)
  calc volume S ≤ volume ((S ∩ Iio u) ∪ (S ∩ Ioi u)) := measure_mono hsub
    _ ≤ volume (S ∩ Iio u) + volume (S ∩ Ioi u) := measure_union_le _ _
    _ ≤ ENNReal.ofReal c + ENNReal.ofReal c := add_le_add hL hR
    _ = ENNReal.ofReal (2 * c) := by
      rw [← ENNReal.ofReal_add hc0 hc0]; ring_nf

end Cubature.Aux

namespace Cubature.Aux

/-- The profile `Ψ` bounding `m(y)²` in terms of the height `r = H(y)`. -/
noncomputable def bandProfile (ρ α η Hmax : ℝ) (r : ℝ) : ℝ :=
  (Icc (2 * η) Hmax).indicator (fun r => 96 * ρ ^ 2 * η ^ 2 / (α * r)) r +
    (Ico (-η) (2 * η)).indicator (fun _ => 16 * ρ ^ 2 * η / α) r

lemma bandProfile_nonneg {ρ α η Hmax : ℝ} (hα : 0 < α) (hη : 0 < η) (r : ℝ) :
    0 ≤ bandProfile ρ α η Hmax r := by
  unfold bandProfile
  refine add_nonneg ?_ ?_
  · by_cases h : r ∈ Icc (2 * η) Hmax
    · rw [indicator_of_mem h]
      have : 0 < r := by linarith [h.1]
      positivity
    · rw [indicator_of_notMem h]
  · by_cases h : r ∈ Ico (-η) (2 * η)
    · rw [indicator_of_mem h]; positivity
    · rw [indicator_of_notMem h]

lemma bandProfile_integrable {ρ α η Hmax : ℝ} (hα : 0 < α) (hη : 0 < η) :
    Integrable (bandProfile ρ α η Hmax) := by
  unfold bandProfile
  refine Integrable.add ?_ ?_
  · refine (integrable_indicator_iff measurableSet_Icc).mpr ?_
    refine ContinuousOn.integrableOn_Icc ?_
    refine ContinuousOn.div continuousOn_const (continuousOn_const.mul continuousOn_id) ?_
    intro r hr
    have : 0 < r := by linarith [hr.1]
    positivity
  · exact (integrable_indicator_iff measurableSet_Ico).mpr (integrableOn_const (by simp))

lemma bandProfile_integral {ρ α η Hmax : ℝ} (hα : 0 < α) (hη : 0 < η) (hH : 2 * η ≤ Hmax) :
    ∫ r, bandProfile ρ α η Hmax r = 96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) +
      48 * ρ ^ 2 * η ^ 2 / α := by
  unfold bandProfile
  have h1 : Integrable ((Icc (2 * η) Hmax).indicator (fun r => 96 * ρ ^ 2 * η ^ 2 / (α * r))) := by
    refine (integrable_indicator_iff measurableSet_Icc).mpr ?_
    refine ContinuousOn.integrableOn_Icc ?_
    refine ContinuousOn.div continuousOn_const (continuousOn_const.mul continuousOn_id) ?_
    intro r hr
    have : 0 < r := by linarith [hr.1]
    positivity
  have h2 : Integrable ((Ico (-η) (2 * η)).indicator (fun _ : ℝ => 16 * ρ ^ 2 * η / α)) :=
    (integrable_indicator_iff measurableSet_Ico).mpr (integrableOn_const (by simp))
  rw [integral_add h1 h2, integral_indicator measurableSet_Icc, integral_indicator measurableSet_Ico,
    setIntegral_const, integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le hH]
  have e : ∀ r : ℝ, 96 * ρ ^ 2 * η ^ 2 / (α * r) = (96 * ρ ^ 2 * η ^ 2 / α) * r⁻¹ := by
    intro r; rw [div_mul_eq_div_div, div_eq_mul_inv]
  simp_rw [e]
  rw [intervalIntegral.integral_const_mul, integral_inv_of_pos (by positivity) (by linarith),
    Measure.real, Real.volume_Ico, ENNReal.toReal_ofReal (by linarith), smul_eq_mul]
  ring

end Cubature.Aux

namespace Cubature.Aux

lemma bandProfile_measurable {ρ α η Hmax : ℝ} : Measurable (bandProfile ρ α η Hmax) := by
  unfold bandProfile
  refine Measurable.add ?_ ?_
  · exact (measurable_const.div (measurable_const.mul measurable_id)).indicator measurableSet_Icc
  · exact measurable_const.indicator measurableSet_Ico

/-- Pointwise bound `m(y)² ≤ Ψ(H(y))`. -/
lemma band_sq_le_profile {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S) {ψ : ℝ → ℝ} {α η Hv Hmax : ℝ}
    (hα : 0 < α) (hη : 0 < η) (hψ : StrongConcaveOn univ α ψ) (hle : ∀ t, ψ t ≤ Hv)
    (hatt : ∃ u, ψ u = Hv) (hHv : Hv ≤ Hmax) :
    (ν.real {t : ℝ | |ψ t| ≤ η}) ^ 2 ≤ bandProfile ρ α η Hmax Hv := by
  have hm0 : 0 ≤ ν.real {t : ℝ | |ψ t| ≤ η} := measureReal_nonneg
  unfold bandProfile
  rcases le_or_gt (2 * η) Hv with h1 | h1
  · -- far from tangency
    obtain ⟨u, hu⟩ := hatt
    have hvol := volume_band_refined hα hψ hle hu hη h1
    have hb : ν.real {t : ℝ | |ψ t| ≤ η} ≤ ρ * (2 * √(24 * η ^ 2 / (α * Hv))) := by
      have h3 : ν {t : ℝ | |ψ t| ≤ η} ≤ ENNReal.ofReal (ρ * (2 * √(24 * η ^ 2 / (α * Hv)))) := by
        rw [ENNReal.ofReal_mul hρ]
        exact (hν _).trans (by gcongr)
      exact ENNReal.toReal_le_of_le_ofReal (by positivity) h3
    have hHpos : 0 < Hv := by linarith
    rw [indicator_of_mem (show Hv ∈ Icc (2 * η) Hmax from ⟨h1, hHv⟩),
      indicator_of_notMem (show Hv ∉ Ico (-η) (2 * η) from fun h => by linarith [h.2]), add_zero]
    calc (ν.real {t : ℝ | |ψ t| ≤ η}) ^ 2 ≤ (ρ * (2 * √(24 * η ^ 2 / (α * Hv)))) ^ 2 :=
          pow_le_pow_left₀ hm0 hb 2
      _ = 4 * ρ ^ 2 * (√(24 * η ^ 2 / (α * Hv))) ^ 2 := by ring
      _ = 96 * ρ ^ 2 * η ^ 2 / (α * Hv) := by
          rw [Real.sq_sqrt (by positivity)]; field_simp; ring
  · rcases le_or_gt (-η) Hv with h2 | h2
    · -- near tangency: R1
      have hb := measureReal_sublevel_le hρ hν (φ := ψ) hα hη.le (Or.inr hψ)
      rw [indicator_of_notMem (show Hv ∉ Icc (2 * η) Hmax from fun h => by linarith [h.1]),
        indicator_of_mem (show Hv ∈ Ico (-η) (2 * η) from ⟨h2, h1⟩), zero_add]
      calc (ν.real {t : ℝ | |ψ t| ≤ η}) ^ 2 ≤ (ρ * (4 * √(η / α))) ^ 2 :=
            pow_le_pow_left₀ hm0 hb 2
        _ = 16 * ρ ^ 2 * (√(η / α)) ^ 2 := by ring
        _ = 16 * ρ ^ 2 * η / α := by
            rw [Real.sq_sqrt (by positivity)]; ring
    · -- below the surface everywhere: the band is empty
      have hempty : {t : ℝ | |ψ t| ≤ η} = ∅ := by
        ext t
        simp only [mem_ofPred_eq, mem_empty_iff_false, iff_false, not_le]
        have := hle t
        rw [abs_of_neg (by linarith)]; linarith
      rw [hempty, measureReal_empty]
      simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow]
      exact add_nonneg (indicator_nonneg (fun r hr => by
        have : 0 < r := by linarith [hr.1]
        positivity) _) (indicator_nonneg (fun _ _ => by positivity) _)

end Cubature.Aux

namespace Cubature.Aux

theorem jump_stratified_height {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
    [MeasurableSpace Y] [OpensMeasurableSpace Y]
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
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 < h) (h2 : 2 * (L * h) ≤ Hmax)
    (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L * h) ^ 2 / α * Real.log (Hmax / (2 * (L * h))) +
          48 * ρ ^ 2 * (L * h) ^ 2 / α)) := by
  set F := preintJump ν φ θ with hF
  have hφ' : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y) :=
    fun y => Or.inr (hφ y)
  have hcontφ : ∀ z, Continuous (φ · z) := fun z => continuous_of_strong hα (hφ' z)
  have hhol := jump_holder hρ hν hα hφ' hφL hθm hθB hθL
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
  set η := L * h with hη_def
  have hη : 0 < η := mul_pos hL hh
  set S := {p : ℝ × Y | |φ p.1 p.2| ≤ η} with hS
  have hSm : MeasurableSet S :=
    measurableSet_le (continuous_abs.measurable.comp hφm) measurable_const
  set m : Y → ℝ := fun y => ν.real {t : ℝ | |φ t y| ≤ η} with hm
  have hm_meas : Measurable m := (measurable_measure_prodMk_right (μ := ν) hSm).ennreal_toReal
  have hm0 : ∀ y, 0 ≤ m y := fun y => measureReal_nonneg
  have hm1 : ∀ y, m y ≤ 1 := fun y => measureReal_le_one
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
    have h2 : ν.real {t : ℝ | |φ t x| ≤ L * dist x x'} ≤ ν.real {t : ℝ | |φ t x| ≤ η} := by
      refine measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)
      have ht' : |φ t x| ≤ L * dist x x' := ht
      show |φ t x| ≤ η
      exact ht'.trans (mul_le_mul_of_nonneg_left hd hL.le)
    have h3 := mul_le_mul_of_nonneg_left h2 (hB x)
    simp only [g, m]
    linarith
  have main := stratified_mse_weighted hAm hAd hAcov hXm hXind hXlaw hFm
    (abs_preintJump_le hθB) hgm hgC hFg
  refine main.trans ?_
  -- bound `∫ m²` through the height profile
  set Ψ := bandProfile ρ α η Hmax with hΨ
  have hΨm : Measurable Ψ := bandProfile_measurable
  have hΨ0 : ∀ r, 0 ≤ Ψ r := bandProfile_nonneg hα hη
  have hΨi : Integrable Ψ := bandProfile_integrable hα hη
  have hm2 : ∀ᵐ y ∂μ, m y ^ 2 ≤ Ψ (H y) := by
    filter_upwards [hHmax] with y hy
    exact band_sq_le_profile hρ hν hα hη (hφ y) (fun t => hH1 t y) (hH2 y) hy
  have hint_m2 : Integrable (fun y => m y ^ 2) μ :=
    Integrable.of_bound (hm_meas.pow_const 2).aestronglyMeasurable 1
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow, abs_of_nonneg (hm0 y)]
        have := hm1 y
        nlinarith [hm0 y])
  have hΨH_int : Integrable (fun y => Ψ (H y)) μ := by
    have : Integrable Ψ (μ.map H) := by
      refine Integrable.mono_measure ?_ hK
      exact hΨi.smul_measure ENNReal.ofReal_ne_top
    exact (integrable_map_measure hΨm.aestronglyMeasurable hHm.aemeasurable).mp this
  have hint2 : ∫ y, m y ^ 2 ∂μ ≤ K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) +
      48 * ρ ^ 2 * η ^ 2 / α) := by
    calc ∫ y, m y ^ 2 ∂μ ≤ ∫ y, Ψ (H y) ∂μ := integral_mono_ae hint_m2 hΨH_int hm2
      _ = ∫ r, Ψ r ∂(μ.map H) := (integral_map hHm.aemeasurable hΨm.aestronglyMeasurable).symm
      _ ≤ ∫ r, Ψ r ∂(ENNReal.ofReal K • volume) :=
          integral_mono_measure hK (ae_of_all _ hΨ0) (hΨi.smul_measure ENNReal.ofReal_ne_top)
      _ = K * ∫ r, Ψ r := by
          rw [integral_smul_measure, ENNReal.toReal_ofReal hK0, smul_eq_mul]
      _ = K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) +
            48 * ρ ^ 2 * η ^ 2 / α) := by
          rw [hΨ, bandProfile_integral hα hη h2]
  -- bound `∫ g²`
  have hpt : ∀ y, g y ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2 := by
    intro y
    have h1 : (Lθ * h + B * m y) ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * (B * m y) ^ 2 := by
      nlinarith [sq_nonneg (Lθ * h - B * m y)]
    simp only [g]; nlinarith
  have hint_rhs : Integrable (fun y => 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2) μ :=
    (integrable_const _).add (hint_m2.const_mul _)
  have hint_g2 : Integrable (fun y => g y ^ 2) μ :=
    Integrable.of_bound (hgm.pow_const 2).aestronglyMeasurable ((Lθ * h + B) ^ 2)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC y) 2)
  have hg2 : ∫ y, g y ^ 2 ∂μ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 *
      (K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) + 48 * ρ ^ 2 * η ^ 2 / α)) := by
    calc ∫ y, g y ^ 2 ∂μ ≤ ∫ y, (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2) ∂μ :=
          integral_mono hint_g2 hint_rhs hpt
      _ = 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ∫ y, m y ^ 2 ∂μ := by
          rw [integral_add (integrable_const _) (hint_m2.const_mul _), integral_const,
            integral_const_mul]
          simp
      _ ≤ _ := by gcongr
  have hn0 : 0 ≤ (1 / Fintype.card ι : ℝ) := by positivity
  calc (1 / Fintype.card ι : ℝ) * ∫ y, g y ^ 2 ∂μ
      ≤ (1 / Fintype.card ι : ℝ) * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 *
        (K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) + 48 * ρ ^ 2 * η ^ 2 / α))) :=
        mul_le_mul_of_nonneg_left hg2 hn0
    _ = _ := by ring


theorem jump_stratified_height_convex {Ω Y : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Y]
    [MeasurableSpace Y] [OpensMeasurableSpace Y]
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
    (hAd : Pairwise (Disjoint on A)) (hAcov : μ (⋃ i, A i)ᶜ = 0)
    {h : ℝ} (hh : 0 < h) (h2 : 2 * (L * h) ≤ Hmax)
    (hAD : ∀ i, ∀ y ∈ A i, ∀ y' ∈ A i, dist y y' ≤ h)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : ι → Ω → Y} (hXm : ∀ i, Measurable (X i)) (hXind : iIndepFun X P)
    (hXlaw : ∀ i, P.map (X i) = (Fintype.card ι : ℝ≥0∞) • μ.restrict (A i)) :
    ∫ ω, ((1 / Fintype.card ι : ℝ) * ∑ i, preintJump ν φ θ (X i ω) -
        ∫ y, preintJump ν φ θ y ∂μ) ^ 2 ∂P ≤
      2 / Fintype.card ι * (Lθ ^ 2 * h ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L * h) ^ 2 / α * Real.log (Hmax / (2 * (L * h))) +
          48 * ρ ^ 2 * (L * h) ^ 2 / α)) := by
  set F := preintJump ν φ θ with hF
  have hφ' : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y) :=
    fun y => Or.inl (hφ y)
  have hcontφ : ∀ z, Continuous (φ · z) := fun z => continuous_of_strong hα (hφ' z)
  have hhol := jump_holder hρ hν hα hφ' hφL hθm hθB hθL
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
  set η := L * h with hη_def
  have hη : 0 < η := mul_pos hL hh
  set S := {p : ℝ × Y | |φ p.1 p.2| ≤ η} with hS
  have hSm : MeasurableSet S :=
    measurableSet_le (continuous_abs.measurable.comp hφm) measurable_const
  set m : Y → ℝ := fun y => ν.real {t : ℝ | |φ t y| ≤ η} with hm
  have hm_meas : Measurable m := (measurable_measure_prodMk_right (μ := ν) hSm).ennreal_toReal
  have hm0 : ∀ y, 0 ≤ m y := fun y => measureReal_nonneg
  have hm1 : ∀ y, m y ≤ 1 := fun y => measureReal_le_one
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
    have h2 : ν.real {t : ℝ | |φ t x| ≤ L * dist x x'} ≤ ν.real {t : ℝ | |φ t x| ≤ η} := by
      refine measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)
      have ht' : |φ t x| ≤ L * dist x x' := ht
      show |φ t x| ≤ η
      exact ht'.trans (mul_le_mul_of_nonneg_left hd hL.le)
    have h3 := mul_le_mul_of_nonneg_left h2 (hB x)
    simp only [g, m]
    linarith
  have main := stratified_mse_weighted hAm hAd hAcov hXm hXind hXlaw hFm
    (abs_preintJump_le hθB) hgm hgC hFg
  refine main.trans ?_
  -- bound `∫ m²` through the height profile
  set Ψ := bandProfile ρ α η Hmax with hΨ
  have hΨm : Measurable Ψ := bandProfile_measurable
  have hΨ0 : ∀ r, 0 ≤ Ψ r := bandProfile_nonneg hα hη
  have hΨi : Integrable Ψ := bandProfile_integrable hα hη
  have hm2 : ∀ᵐ y ∂μ, m y ^ 2 ≤ Ψ (H y) := by
    filter_upwards [hHmax] with y hy
    have hconc : StrongConcaveOn univ α (fun t => -φ t y) := UniformConvexOn.neg (hφ y)
    have := band_sq_le_profile hρ hν hα hη hconc (fun t => hH1 t y) (hH2 y) hy
    simpa only [abs_neg] using this
  have hint_m2 : Integrable (fun y => m y ^ 2) μ :=
    Integrable.of_bound (hm_meas.pow_const 2).aestronglyMeasurable 1
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow, abs_of_nonneg (hm0 y)]
        have := hm1 y
        nlinarith [hm0 y])
  have hΨH_int : Integrable (fun y => Ψ (H y)) μ := by
    have : Integrable Ψ (μ.map H) := by
      refine Integrable.mono_measure ?_ hK
      exact hΨi.smul_measure ENNReal.ofReal_ne_top
    exact (integrable_map_measure hΨm.aestronglyMeasurable hHm.aemeasurable).mp this
  have hint2 : ∫ y, m y ^ 2 ∂μ ≤ K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) +
      48 * ρ ^ 2 * η ^ 2 / α) := by
    calc ∫ y, m y ^ 2 ∂μ ≤ ∫ y, Ψ (H y) ∂μ := integral_mono_ae hint_m2 hΨH_int hm2
      _ = ∫ r, Ψ r ∂(μ.map H) := (integral_map hHm.aemeasurable hΨm.aestronglyMeasurable).symm
      _ ≤ ∫ r, Ψ r ∂(ENNReal.ofReal K • volume) :=
          integral_mono_measure hK (ae_of_all _ hΨ0) (hΨi.smul_measure ENNReal.ofReal_ne_top)
      _ = K * ∫ r, Ψ r := by
          rw [integral_smul_measure, ENNReal.toReal_ofReal hK0, smul_eq_mul]
      _ = K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) +
            48 * ρ ^ 2 * η ^ 2 / α) := by
          rw [hΨ, bandProfile_integral hα hη h2]
  -- bound `∫ g²`
  have hpt : ∀ y, g y ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2 := by
    intro y
    have h1 : (Lθ * h + B * m y) ^ 2 ≤ 2 * (Lθ * h) ^ 2 + 2 * (B * m y) ^ 2 := by
      nlinarith [sq_nonneg (Lθ * h - B * m y)]
    simp only [g]; nlinarith
  have hint_rhs : Integrable (fun y => 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2) μ :=
    (integrable_const _).add (hint_m2.const_mul _)
  have hint_g2 : Integrable (fun y => g y ^ 2) μ :=
    Integrable.of_bound (hgm.pow_const 2).aestronglyMeasurable ((Lθ * h + B) ^ 2)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_pow]
        exact pow_le_pow_left₀ (abs_nonneg _) (hgC y) 2)
  have hg2 : ∫ y, g y ^ 2 ∂μ ≤ 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 *
      (K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) + 48 * ρ ^ 2 * η ^ 2 / α)) := by
    calc ∫ y, g y ^ 2 ∂μ ≤ ∫ y, (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * m y ^ 2) ∂μ :=
          integral_mono hint_g2 hint_rhs hpt
      _ = 2 * (Lθ * h) ^ 2 + 2 * B ^ 2 * ∫ y, m y ^ 2 ∂μ := by
          rw [integral_add (integrable_const _) (hint_m2.const_mul _), integral_const,
            integral_const_mul]
          simp
      _ ≤ _ := by gcongr
  have hn0 : 0 ≤ (1 / Fintype.card ι : ℝ) := by positivity
  calc (1 / Fintype.card ι : ℝ) * ∫ y, g y ^ 2 ∂μ
      ≤ (1 / Fintype.card ι : ℝ) * (2 * (Lθ * h) ^ 2 + 2 * B ^ 2 *
        (K * (96 * ρ ^ 2 * η ^ 2 / α * Real.log (Hmax / (2 * η)) + 48 * ρ ^ 2 * η ^ 2 / α))) :=
        mul_le_mul_of_nonneg_left hg2 hn0
    _ = _ := by ring

end Cubature.Aux

namespace Cubature.Aux

theorem jump_jittered_cube_height {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
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
    (hXlaw : ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      2 / (m : ℝ) ^ s * (Lθ ^ 2 / m ^ 2 + B ^ 2 * K *
        (96 * ρ ^ 2 * (L / m) ^ 2 / α * Real.log (Hmax / (2 * (L / m))) +
          48 * ρ ^ 2 * (L / m) ^ 2 / α)) := by
  set cube := Set.pi univ (fun _ : Fin s => Ico (0 : ℝ) 1) with hcube
  have hcm : MeasurableSet cube := MeasurableSet.univ_pi fun _ => measurableSet_Ico
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
  have hHmax' : ∀ᵐ y ∂(volume.restrict cube), H y ≤ Hmax :=
    (ae_restrict_iff' hcm).mpr (ae_of_all _ hHmax)
  have h2' : 2 * (L * (1 / m)) ≤ Hmax := by
    have e : L * (1 / (m : ℝ)) = L / m := by ring
    rw [e]; exact h2
  have main := jump_stratified_height hρ hν hα hL hLθ hφ hφm hφL hθm hθB hθL hHm hH1 hH2
    (μ := volume.restrict cube) hK0 hHmax' hK (cubeCell_measurable s m) (cubeCell_disjoint s m)
    hcov (h := 1 / m) (by positivity) h2' (fun k => cubeCell_dist s m hm k) hXm hXind hlaw'
  rw [hcard] at main
  refine main.trans (le_of_eq ?_)
  have e2 : L * (1 / (m : ℝ)) = L / m := by ring
  rw [e2]
  ring

/-- The height of GKS Example 1 (`y₀`) has a bounded density under the uniform cube measure. -/
lemma cube_eval_zero_le (s : ℕ) :
    (volume.restrict (Set.pi univ fun _ : Fin (s + 1) => Ico (0 : ℝ) 1)).map
      (fun y : Fin (s + 1) → ℝ => y 0) ≤ ENNReal.ofReal 1 • volume := by
  rw [ENNReal.ofReal_one, one_smul]
  refine Measure.le_iff.mpr fun S hS => ?_
  rw [Measure.map_apply (measurable_pi_apply 0) hS,
    Measure.restrict_apply ((measurable_pi_apply 0) hS)]
  have hset : (fun y : Fin (s + 1) → ℝ => y 0) ⁻¹' S ∩ Set.pi univ (fun _ => Ico (0 : ℝ) 1) =
      Set.pi univ (fun j : Fin (s + 1) => if j = 0 then S ∩ Ico 0 1 else Ico (0 : ℝ) 1) := by
    ext y
    simp only [mem_inter_iff, mem_preimage, mem_univ_pi]
    constructor
    · rintro ⟨h1, h2⟩ j
      by_cases hj : j = 0
      · subst hj; simp only [ite_true]; exact ⟨h1, h2 0⟩
      · simp only [hj, ite_false]; exact h2 j
    · intro h
      refine ⟨?_, fun j => ?_⟩
      · have := h 0; simp only [ite_true] at this; exact this.1
      · by_cases hj : j = 0
        · subst hj; have := h 0; simp only [ite_true] at this; exact this.2
        · have := h j; simp only [hj, ite_false] at this; exact this
  rw [hset, volume_pi, Measure.pi_pi, Fin.prod_univ_succ]
  simp only [ite_true, Fin.succ_ne_zero, ite_false, Real.volume_Ico, sub_zero, ENNReal.ofReal_one,
    Finset.prod_const_one, mul_one]
  exact measure_mono inter_subset_left

/-- GKS Example 1 lifted to `[0,1)^{s+1}`: near-Lipschitz rate. -/
theorem gks_example_jittered_log {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 2 ≤ m)
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin (s + 1) → Fin m) → Ω → (Fin (s + 1) → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ (s + 1)) • volume.restrict (cubeCell (s + 1) m k)) :
    ∫ ω, ((1 / (m : ℝ) ^ (s + 1)) * ∑ k, preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) (X k ω) -
        ∫ y in Set.pi univ (fun _ : Fin (s + 1) => Ico (0 : ℝ) 1), preintJump (gaussianReal 0 1)
          (fun (t : ℝ) (y : Fin (s + 1) → ℝ) => y 0 - t ^ 2) (fun _ _ => (1 : ℝ)) y) ^ 2 ∂P ≤
      (48 * Real.log (m / 2) + 24) / (π * (m : ℝ) ^ (s + 3)) := by
  have hm0 : 0 < m := by omega
  have hmR : (2 : ℝ) ≤ m := by exact_mod_cast hm
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
  have main := jump_jittered_cube_height (s := s + 1) (Ω := Ω) hm0 (ρ := (√(2 * π))⁻¹)
    (by positivity) gaussianReal_le_volume
    (φ := fun t y => y 0 - t ^ 2) (θ := fun _ _ => (1 : ℝ)) (α := 2) (L := 1) (Lθ := 0) (B := 1)
    (by norm_num) (by norm_num) le_rfl hconc hφm
    (fun t y y' => by
      have := dist_le_pi_dist y y' 0
      rw [Real.dist_eq] at this
      rw [show y 0 - t ^ 2 - (y' 0 - t ^ 2) = y 0 - y' 0 by ring, one_mul]; exact this)
    (fun _ => measurable_const) (fun _ _ => by simp) (fun _ _ _ => by simp)
    (H := fun y => y 0) (measurable_pi_apply 0) (fun t y => by nlinarith [sq_nonneg t])
    (fun y => ⟨0, by ring⟩) (Hmax := 1) (K := 1) zero_le_one
    (fun y hy => ((Set.mem_univ_pi.mp hy) 0).2.le) (cube_eval_zero_le s)
    (by
      have : (1 : ℝ) / m ≤ 1 / 2 := one_div_le_one_div_of_le (by norm_num) hmR
      linarith) hXm hXind hXlaw
  refine main.trans (le_of_eq ?_)
  have hπ : 0 < π := Real.pi_pos
  have hs2 : (√(2 * π))⁻¹ ^ 2 = 1 / (2 * π) := by
    rw [inv_pow, Real.sq_sqrt (by positivity), one_div]
  have hlog : (1 : ℝ) / (2 * (1 / m)) = m / 2 := by field_simp
  rw [hs2, hlog]
  field_simp
  ring

end Cubature.Aux
