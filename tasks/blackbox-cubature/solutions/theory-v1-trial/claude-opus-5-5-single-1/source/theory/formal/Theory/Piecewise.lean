import Theory.Defs
import Theory.Preint
import Theory.Tube

/-!
# Piecewise non-degenerate sections (auxiliary proofs for R2′)

Sublevel estimates on a convex piece `J ⊆ ℝ` on which `φ` is strongly convex, strongly concave or
strongly monotone, and the resulting ½-Hölder bound for preintegration when every line is
covered by finitely many such pieces.
-/

open MeasureTheory ProbabilityTheory Real Set

namespace Cubature.Aux

lemma strongConvexOn_ineq {J : Set ℝ} {φ : ℝ → ℝ} {α : ℝ} (hφ : StrongConvexOn J α φ) {x y : ℝ}
    (hx : x ∈ J) (hy : y ∈ J) (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    φ (a * x + b * y) ≤ a * φ x + b * φ y - a * b * (α / 2 * (x - y) ^ 2) := by
  have := hφ.2 hx hy ha hb hab
  simpa [smul_eq_mul, Real.norm_eq_abs, sq_abs] using this

lemma band_left_on {J : Set ℝ} {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hφ : StrongConvexOn J α φ)
    {x y u : ℝ} (hxJ : x ∈ J) (huJ : u ∈ J)
    (hx : φ x ≤ η) (hy : -η ≤ φ y) (hu : φ u ≤ -η) (hxy : x ≤ y) (hyu : y < u) :
    α * (y - x) ^ 2 ≤ 4 * η := by
  rcases eq_or_lt_of_le hxy with h | hxy
  · subst h; have : 0 ≤ η := by linarith
    nlinarith
  have hux : 0 < u - x := by linarith
  set a := (u - y) / (u - x) with ha_def
  set b := (y - x) / (u - x) with hb_def
  have ha : 0 < a := div_pos (by linarith) hux
  have hb : 0 < b := div_pos (by linarith) hux
  have hab : a + b = 1 := by rw [ha_def, hb_def, ← add_div]; field_simp; ring
  have hcomb : a * x + b * u = y := by
    rw [ha_def, hb_def]; field_simp; ring
  have key := strongConvexOn_ineq hφ hxJ huJ a b ha.le hb.le hab
  rw [hcomb] at key
  have e1 : a * φ x ≤ a * η := mul_le_mul_of_nonneg_left hx ha.le
  have e2 : b * φ u ≤ b * (-η) := mul_le_mul_of_nonneg_left hu hb.le
  have h1 : a * b * (α / 2 * (x - u) ^ 2) ≤ 2 * a * η := by
    have : η = a * η + b * η := by rw [← add_mul, hab, one_mul]
    nlinarith
  have h2 : b * (α / 2 * (x - u) ^ 2) ≤ 2 * η := by
    have h1' : a * (b * (α / 2 * (x - u) ^ 2)) ≤ a * (2 * η) := by nlinarith
    exact le_of_mul_le_mul_left h1' ha
  have hbu : b * (u - x) = y - x := by rw [hb_def]; field_simp
  have h3 : α / 2 * (y - x) * (u - x) ≤ 2 * η := by
    have : b * (α / 2 * (x - u) ^ 2) = α / 2 * (b * (u - x)) * (u - x) := by ring
    rw [this, hbu] at h2; exact h2
  have h4 : (y - x) ≤ (u - x) := by linarith
  have h5 : α / 2 * (y - x) * (y - x) ≤ α / 2 * (y - x) * (u - x) :=
    mul_le_mul_of_nonneg_left h4 (by positivity)
  nlinarith

lemma band_right_on {J : Set ℝ} {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hφ : StrongConvexOn J α φ)
    {x y u : ℝ} (hyJ : y ∈ J) (huJ : u ∈ J)
    (hx : -η ≤ φ x) (hy : φ y ≤ η) (hu : φ u ≤ -η) (hxy : x ≤ y) (hux : u < x) :
    α * (y - x) ^ 2 ≤ 4 * η := by
  rcases eq_or_lt_of_le hxy with h | hxy
  · subst h; have : 0 ≤ η := by linarith
    nlinarith
  have hyu : 0 < y - u := by linarith
  set a := (y - x) / (y - u) with ha_def
  set b := (x - u) / (y - u) with hb_def
  have ha : 0 < a := div_pos (by linarith) hyu
  have hb : 0 < b := div_pos (by linarith) hyu
  have hab : a + b = 1 := by rw [ha_def, hb_def, ← add_div]; field_simp; ring
  have hcomb : a * u + b * y = x := by
    rw [ha_def, hb_def]; field_simp; ring
  have key := strongConvexOn_ineq hφ huJ hyJ a b ha.le hb.le hab
  rw [hcomb] at key
  have e1 : a * φ u ≤ a * (-η) := mul_le_mul_of_nonneg_left hu ha.le
  have e2 : b * φ y ≤ b * η := mul_le_mul_of_nonneg_left hy hb.le
  have h1 : a * b * (α / 2 * (u - y) ^ 2) ≤ 2 * b * η := by
    have : η = a * η + b * η := by rw [← add_mul, hab, one_mul]
    nlinarith
  have h2 : a * (α / 2 * (u - y) ^ 2) ≤ 2 * η := by
    have h1' : b * (a * (α / 2 * (u - y) ^ 2)) ≤ b * (2 * η) := by nlinarith
    exact le_of_mul_le_mul_left h1' hb
  have hau : a * (y - u) = y - x := by rw [ha_def]; field_simp
  have h3 : α / 2 * (y - x) * (y - u) ≤ 2 * η := by
    have : a * (α / 2 * (u - y) ^ 2) = α / 2 * (a * (y - u)) * (y - u) := by ring
    rw [this, hau] at h2; exact h2
  have h4 : (y - x) ≤ (y - u) := by linarith
  have h5 : α / 2 * (y - x) * (y - x) ≤ α / 2 * (y - x) * (y - u) :=
    mul_le_mul_of_nonneg_left h4 (by positivity)
  nlinarith

lemma abs_le_of_sq' {d c α : ℝ} (hα : 0 < α) (hc : 0 ≤ c) (h : α * d ^ 2 ≤ c ^ 2 * α) :
    |d| ≤ c := by
  have : d ^ 2 ≤ c ^ 2 := by nlinarith
  exact abs_le.mpr ⟨(abs_le_of_sq_le_sq' this hc).1, (abs_le_of_sq_le_sq' this hc).2⟩

/-- R1 on a convex piece. -/
theorem volume_band_le_of_strongConvexOn {J : Set ℝ} (hJ : Convex ℝ J) {φ : ℝ → ℝ}
    {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η) (hφ : StrongConvexOn J α φ) :
    volume {t ∈ J | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(η / α)) := by
  set S := {t ∈ J | |φ t| ≤ η}
  set r := √(η / α) with hr
  have hr0 : 0 ≤ r := Real.sqrt_nonneg _
  have hr2 : r ^ 2 * α = η := by
    rw [hr, Real.sq_sqrt (div_nonneg hη hα.le)]; field_simp
  by_cases hex : ∃ u ∈ J, φ u < -η
  · obtain ⟨u, huJ, hu⟩ := hex
    have hsub : S ⊆ (S ∩ Iio u) ∪ (S ∩ Ioi u) := by
      intro t ht
      rcases lt_trichotomy t u with h | h | h
      · exact Or.inl ⟨ht, h⟩
      · exfalso; subst h; have := (abs_le.mp ht.2).1; linarith
      · exact Or.inr ⟨ht, h⟩
    have hdiam : ∀ T : Set ℝ, (∀ x ∈ T, ∀ y ∈ T, x ≤ y → α * (y - x) ^ 2 ≤ 4 * η) →
        volume T ≤ ENNReal.ofReal (2 * r) := by
      intro T hT
      refine (Real.volume_le_diam T).trans ?_
      refine Metric.ediam_le fun x hx y hy => ?_
      rw [edist_dist, Real.dist_eq]
      refine ENNReal.ofReal_le_ofReal ?_
      have h2r : 0 ≤ 2 * r := by positivity
      rcases le_total x y with hxy | hxy
      · have := hT x hx y hy hxy
        rw [abs_sub_comm]
        exact abs_le_of_sq' hα h2r (by nlinarith)
      · have := hT y hy x hx hxy
        exact abs_le_of_sq' hα h2r (by nlinarith)
    have hL : volume (S ∩ Iio u) ≤ ENNReal.ofReal (2 * r) := by
      refine hdiam _ fun x hx y hy hxy => ?_
      exact band_left_on hα hφ hx.1.1 huJ (abs_le.mp hx.1.2).2 (abs_le.mp hy.1.2).1 hu.le hxy hy.2
    have hR : volume (S ∩ Ioi u) ≤ ENNReal.ofReal (2 * r) := by
      refine hdiam _ fun x hx y hy hxy => ?_
      exact band_right_on hα hφ hy.1.1 huJ (abs_le.mp hx.1.2).1 (abs_le.mp hy.1.2).2 hu.le hxy hx.2
    calc volume S ≤ volume ((S ∩ Iio u) ∪ (S ∩ Ioi u)) := measure_mono hsub
      _ ≤ volume (S ∩ Iio u) + volume (S ∩ Ioi u) := measure_union_le _ _
      _ ≤ ENNReal.ofReal (2 * r) + ENNReal.ofReal (2 * r) := add_le_add hL hR
      _ = ENNReal.ofReal (4 * r) := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]; ring_nf
  · push Not at hex
    refine (Real.volume_le_diam S).trans ?_
    refine Metric.ediam_le fun x hx y hy => ?_
    rw [edist_dist, Real.dist_eq]
    refine ENNReal.ofReal_le_ofReal ?_
    have h4r : 0 ≤ 4 * r := by positivity
    have hmid : (1 / 2 : ℝ) * x + 1 / 2 * y ∈ J :=
      hJ hx.1 hy.1 (by norm_num) (by norm_num) (by norm_num)
    have key := strongConvexOn_ineq hφ hx.1 hy.1 (1 / 2) (1 / 2) (by norm_num) (by norm_num)
      (by norm_num)
    have hlow := hex _ hmid
    have h1 := (abs_le.mp hx.2).2
    have h2 := (abs_le.mp hy.2).2
    have : α * (y - x) ^ 2 ≤ 16 * η := by nlinarith
    rw [abs_sub_comm]
    exact abs_le_of_sq' hα h4r (by nlinarith)

/-- Strongly monotone piece: `α |t - t'| ≤ |φ t - φ t'|` on `J`. -/
theorem volume_band_le_of_monotone {J : Set ℝ} {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α)
    (hη : 0 ≤ η) (hφ : ∀ t ∈ J, ∀ t' ∈ J, α * |t - t'| ≤ |φ t - φ t'|) :
    volume {t ∈ J | |φ t| ≤ η} ≤ ENNReal.ofReal (2 * η / α) := by
  refine (Real.volume_le_diam _).trans ?_
  refine Metric.ediam_le fun x hx y hy => ?_
  rw [edist_dist, Real.dist_eq]
  refine ENNReal.ofReal_le_ofReal ?_
  have h1 := hφ x hx.1 y hy.1
  have h2 : |φ x - φ y| ≤ 2 * η := by
    have := abs_sub (φ x) (φ y)
    linarith [hx.2, hy.2]
  rw [le_div_iff₀ hα, mul_comm]
  linarith

end Cubature.Aux

namespace Cubature.Aux

theorem jump_holder_piecewise {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ}
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
  set η := L * dist y y' with hη_def
  have hη : 0 ≤ η := (abs_nonneg _).trans (hφL 0 y y')
  have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0 y)
  set c := 4 * √(η / α) + 2 * η / α with hc
  have hc0 : 0 ≤ c := by positivity
  have hA : 4 * √(η / α) ≤ c := by
    have : 0 ≤ 2 * η / α := by positivity
    rw [hc]; linarith
  have hM : 2 * η / α ≤ c := by
    have : 0 ≤ 4 * √(η / α) := by positivity
    rw [hc]; linarith
  have hpiece : ∀ i, volume {t ∈ J y i | |φ t y| ≤ η} ≤ ENNReal.ofReal c := by
    intro i
    rcases hφ y i with h | h | h
    · exact (volume_band_le_of_strongConvexOn (hJ y i) hα hη h).trans
        (ENNReal.ofReal_le_ofReal hA)
    · have h' : StrongConvexOn (J y i) α (-(φ · y)) := UniformConcaveOn.neg h
      have := volume_band_le_of_strongConvexOn (hJ y i) hα hη h'
      simp only [Pi.neg_apply, abs_neg] at this
      exact this.trans (ENNReal.ofReal_le_ofReal hA)
    · exact (volume_band_le_of_monotone hα hη h).trans (ENNReal.ofReal_le_ofReal hM)
  have hsub : {t : ℝ | |φ t y| ≤ η} ⊆ ⋃ i, {t ∈ J y i | |φ t y| ≤ η} := by
    intro t ht
    have : t ∈ ⋃ i, J y i := by rw [hJcov y]; exact mem_univ t
    obtain ⟨i, hi⟩ := mem_iUnion.mp this
    exact mem_iUnion.mpr ⟨i, hi, ht⟩
  have hvol : volume {t : ℝ | |φ t y| ≤ η} ≤ ENNReal.ofReal (r * c) := by
    calc volume {t : ℝ | |φ t y| ≤ η} ≤ volume (⋃ i, {t ∈ J y i | |φ t y| ≤ η}) :=
          measure_mono hsub
      _ ≤ ∑ i, volume {t ∈ J y i | |φ t y| ≤ η} := measure_iUnion_fintype_le _ _
      _ ≤ ∑ _i : Fin r, ENNReal.ofReal c := Finset.sum_le_sum fun i _ => hpiece i
      _ = ENNReal.ofReal (r * c) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
            ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_natCast]
  have hνS : ν.real {t : ℝ | |φ t y| ≤ η} ≤ ρ * (r * c) := by
    have h3 : ν {t : ℝ | |φ t y| ≤ η} ≤ ENNReal.ofReal (ρ * (r * c)) := by
      rw [ENNReal.ofReal_mul hρ]
      exact (hν _).trans (by gcongr)
    exact ENNReal.toReal_le_of_le_ofReal (by positivity) h3
  refine (jump_diff_le hφc hφL hθm hθB hθL y y').trans ?_
  have := mul_le_mul_of_nonneg_left hνS hB
  calc Lθ * dist y y' + B * ν.real {t : ℝ | |φ t y| ≤ L * dist y y'}
      ≤ Lθ * dist y y' + B * (ρ * (r * c)) := by linarith
    _ = Lθ * dist y y' + ρ * B * r * (4 * √(L * dist y y' / α) + 2 * (L * dist y y') / α) := by
        rw [hc]; ring

end Cubature.Aux
