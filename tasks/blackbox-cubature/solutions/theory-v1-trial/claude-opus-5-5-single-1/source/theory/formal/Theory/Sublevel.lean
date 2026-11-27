import Mathlib

/-!
# Sublevel-set estimate for strongly convex functions (auxiliary proofs for R1)

If `φ : ℝ → ℝ` is `α`-strongly convex (`α > 0`), then the band `{t | |φ t| ≤ η}` has Lebesgue
measure at most `4 √(η / α)`.  The constant `4` is attained by `φ t = α/2 t² - η`.
-/

open MeasureTheory Real Set

namespace Cubature.Aux

lemma strongConvex_ineq {φ : ℝ → ℝ} {α : ℝ} (hφ : StrongConvexOn univ α φ) (x y a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    φ (a * x + b * y) ≤ a * φ x + b * φ y - a * b * (α / 2 * (x - y) ^ 2) := by
  have := hφ.2 (mem_univ x) (mem_univ y) ha hb hab
  simpa [smul_eq_mul, Real.norm_eq_abs, sq_abs] using this

lemma band_left {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hφ : StrongConvexOn univ α φ)
    {x y u : ℝ} (hx : φ x ≤ η) (hy : -η ≤ φ y) (hu : φ u ≤ -η) (hxy : x ≤ y) (hyu : y < u) :
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
  have key := strongConvex_ineq hφ x u a b ha.le hb.le hab
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

lemma band_right {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hφ : StrongConvexOn univ α φ)
    {x y u : ℝ} (hx : -η ≤ φ x) (hy : φ y ≤ η) (hu : φ u ≤ -η) (hxy : x ≤ y) (hux : u < x) :
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
  have key := strongConvex_ineq hφ u y a b ha.le hb.le hab
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

lemma band_mid {φ : ℝ → ℝ} {α η : ℝ} (_hα : 0 < α) (hφ : StrongConvexOn univ α φ)
    (hlow : ∀ u, -η ≤ φ u) {x y : ℝ} (hx : φ x ≤ η) (hy : φ y ≤ η) :
    α * (y - x) ^ 2 ≤ 16 * η := by
  have key := strongConvex_ineq hφ x y (1/2) (1/2) (by norm_num) (by norm_num) (by norm_num)
  have := hlow (1/2 * x + 1/2 * y)
  nlinarith

lemma abs_le_of_sq {d c α : ℝ} (hα : 0 < α) (hc : 0 ≤ c) (h : α * d ^ 2 ≤ c ^ 2 * α) :
    |d| ≤ c := by
  have : d ^ 2 ≤ c ^ 2 := by nlinarith
  exact abs_le_of_sq_le_sq' this hc |>.2 |> fun h2 => abs_le.mpr ⟨(abs_le_of_sq_le_sq' this hc).1, h2⟩

/-- **Sharp sublevel-set estimate (van der Corput, order two).** -/
theorem volume_sublevel_le_of_strongConvexOn {φ : ℝ → ℝ} {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η)
    (hφ : StrongConvexOn univ α φ) :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(η / α)) := by
  set S := {t : ℝ | |φ t| ≤ η}
  set r := √(η / α) with hr
  have hr0 : 0 ≤ r := Real.sqrt_nonneg _
  have hr2 : r ^ 2 * α = η := by
    rw [hr, Real.sq_sqrt (div_nonneg hη hα.le)]; field_simp
  by_cases hex : ∃ u, φ u < -η
  · obtain ⟨u, hu⟩ := hex
    have hsub : S ⊆ (S ∩ Iio u) ∪ (S ∩ Ioi u) := by
      intro t ht
      rcases lt_trichotomy t u with h | h | h
      · exact Or.inl ⟨ht, h⟩
      · exfalso; subst h; have := (abs_le.mp (show |φ t| ≤ η from ht)).1; linarith
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
        exact abs_le_of_sq hα h2r (by nlinarith)
      · have := hT y hy x hx hxy
        exact abs_le_of_sq hα h2r (by nlinarith)
    have hL : volume (S ∩ Iio u) ≤ ENNReal.ofReal (2 * r) := by
      refine hdiam _ fun x hx y hy hxy => ?_
      exact band_left hα hφ (abs_le.mp hx.1).2 (abs_le.mp hy.1).1 hu.le hxy hy.2
    have hR : volume (S ∩ Ioi u) ≤ ENNReal.ofReal (2 * r) := by
      refine hdiam _ fun x hx y hy hxy => ?_
      exact band_right hα hφ (abs_le.mp hx.1).1 (abs_le.mp hy.1).2 hu.le hxy hx.2
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
    have := band_mid hα hφ hex (abs_le.mp hx).2 (abs_le.mp hy).2
    rw [abs_sub_comm]
    exact abs_le_of_sq hα h4r (by nlinarith)

end Cubature.Aux
