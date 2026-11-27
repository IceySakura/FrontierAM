import Theory.Defs
import Theory.Sublevel
import Theory.Preint
import Theory.Tube

/-!
# Third-order tangencies: cube-root regularity (auxiliary proofs for R9)

If `φ` is differentiable and its derivative `φ'` is `α`-strongly convex (e.g. `φ(t) = t³ - y`,
Gilbert–Kuo–Sloan Example 4), then `Leb{|φ| ≤ η} ≤ 4 √(β/α) + 6 η / β` for every `β > 0`, hence
`≤ 10 (η / α)^{1/3}`.
-/

open MeasureTheory ProbabilityTheory Real Set

namespace Cubature.Aux

lemma mvt_lower {φ φ' : ℝ → ℝ} (hd : ∀ t, HasDerivAt φ (φ' t) t) {β x y : ℝ} (hxy : x ≤ y)
    (h : ∀ z ∈ Icc x y, β ≤ φ' z) : β * (y - x) ≤ φ y - φ x := by
  have hcont : ContinuousOn φ (Icc x y) := fun z _ => (hd z).continuousAt.continuousWithinAt
  have hdiff : DifferentiableOn ℝ φ (interior (Icc x y)) := fun z _ =>
    (hd z).differentiableAt.differentiableWithinAt
  refine (convex_Icc x y).mul_sub_le_image_sub_of_le_deriv hcont hdiff (fun z hz => ?_) x
    (left_mem_Icc.mpr hxy) y (right_mem_Icc.mpr hxy) hxy
  rw [(hd z).deriv]
  exact h z (interior_subset hz)

lemma diam_band_of_slope {φ φ' : ℝ → ℝ} (hd : ∀ t, HasDerivAt φ (φ' t) t) {β η : ℝ} (hβ : 0 < β)
    {T : Set ℝ}
    (hTs : ∀ x ∈ T, ∀ y ∈ T, x ≤ y → (∀ z ∈ Icc x y, β ≤ φ' z) ∨ (∀ z ∈ Icc x y, φ' z ≤ -β))
    (hband : ∀ x ∈ T, |φ x| ≤ η) :
    volume T ≤ ENNReal.ofReal (2 * η / β) := by
  refine (Real.volume_le_diam T).trans ?_
  refine Metric.ediam_le fun x hx y hy => ?_
  rw [edist_dist, Real.dist_eq]
  refine ENNReal.ofReal_le_ofReal ?_
  have key : ∀ p ∈ T, ∀ q ∈ T, p ≤ q → |q - p| ≤ 2 * η / β := by
    intro p hp q hq hpq
    have h1 := hband p hp
    have h2 := hband q hq
    rw [abs_of_nonneg (sub_nonneg.mpr hpq), le_div_iff₀ hβ]
    rcases hTs p hp q hq hpq with h | h
    · have := mvt_lower hd hpq h
      have := (abs_le.mp h1); have := (abs_le.mp h2)
      nlinarith
    · have hneg : ∀ t, HasDerivAt (fun s => -φ s) (-φ' t) t := fun t => (hd t).neg
      have hm : β * (q - p) ≤ -φ q - -φ p :=
        mvt_lower hneg hpq (fun z hz => by have := h z hz; linarith)
      have := (abs_le.mp h1); have := (abs_le.mp h2)
      nlinarith
  rcases le_total x y with hxy | hxy
  · rw [abs_sub_comm]; exact key x hx y hy hxy
  · exact key y hy x hx hxy

/-- Sublevel estimate for third-order tangency, with a free parameter `β`. -/
theorem volume_band_cubic {φ φ' : ℝ → ℝ} (hd : ∀ t, HasDerivAt φ (φ' t) t) {α η β : ℝ}
    (hα : 0 < α) (hη : 0 ≤ η) (hβ : 0 < β) (hφ' : StrongConvexOn univ α φ') :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(β / α) + 6 * η / β) := by
  have hconv : ConvexOn ℝ univ φ' := hφ'.convexOn (fun r => by simp only [Pi.zero_apply]; positivity)
  set I := {t : ℝ | φ' t ≤ β} with hI
  have hIc : Convex ℝ I := by
    have := hconv.convex_le β
    simpa [Set.sep_univ] using this
  have hA4c : Convex ℝ {t : ℝ | φ' t < -β} := by
    have := hconv.convex_lt (-β)
    simpa [Set.sep_univ] using this
  set S := {t : ℝ | |φ t| ≤ η}
  set A1 := {t : ℝ | |φ' t| ≤ β}
  set A2 := {t : ℝ | t ∈ S ∧ β < φ' t ∧ ∀ s ∈ I, t < s}
  set A3 := {t : ℝ | t ∈ S ∧ β < φ' t ∧ ∀ s ∈ I, s < t}
  set A4 := {t : ℝ | t ∈ S ∧ φ' t < -β}
  have hcover : S ⊆ A1 ∪ A2 ∪ A3 ∪ A4 := by
    intro t ht
    by_cases h1 : |φ' t| ≤ β
    · exact Or.inl (Or.inl (Or.inl h1))
    rw [abs_le, not_and_or, not_le, not_le] at h1
    rcases h1 with h1 | h1
    · exact Or.inr ⟨ht, h1⟩
    · by_cases hl : ∀ s ∈ I, t < s
      · exact Or.inl (Or.inl (Or.inr ⟨ht, h1, hl⟩))
      by_cases hr : ∀ s ∈ I, s < t
      · exact Or.inl (Or.inr ⟨ht, h1, hr⟩)
      exfalso
      push Not at hl hr
      obtain ⟨s1, hs1, hs1t⟩ := hl
      obtain ⟨s2, hs2, hs2t⟩ := hr
      have : t ∈ I := hIc.ordConnected.out hs1 hs2 ⟨hs1t, hs2t⟩
      have : φ' t ≤ β := this
      linarith
  have hA1 : volume A1 ≤ ENNReal.ofReal (4 * √(β / α)) :=
    volume_sublevel_le_of_strongConvexOn hα hβ.le hφ'
  have hA2 : volume A2 ≤ ENNReal.ofReal (2 * η / β) := by
    refine diam_band_of_slope hd hβ (T := A2) ?_ (fun x hx => hx.1)
    · intro x hx y hy hxy
      left
      intro z hz
      by_contra hneg
      push Not at hneg
      have hzI : z ∈ I := hneg.le
      have := hy.2.2 z hzI
      linarith [hz.2]
  have hA3 : volume A3 ≤ ENNReal.ofReal (2 * η / β) := by
    refine diam_band_of_slope hd hβ (T := A3) ?_ (fun x hx => hx.1)
    intro x hx y hy hxy
    left
    intro z hz
    by_contra hneg
    push Not at hneg
    have hzI : z ∈ I := hneg.le
    have := hx.2.2 z hzI
    linarith [hz.1]
  have hA4 : volume A4 ≤ ENNReal.ofReal (2 * η / β) := by
    refine diam_band_of_slope hd hβ (T := A4) ?_ (fun x hx => hx.1)
    intro x hx y hy hxy
    right
    intro z hz
    have : z ∈ {t : ℝ | φ' t < -β} := hA4c.ordConnected.out hx.2 hy.2 hz
    exact le_of_lt this
  calc volume S ≤ volume (A1 ∪ A2 ∪ A3 ∪ A4) := measure_mono hcover
    _ ≤ volume A1 + volume A2 + volume A3 + volume A4 := by
        refine (measure_union_le _ _).trans ?_
        gcongr
        refine (measure_union_le _ _).trans ?_
        gcongr
        exact measure_union_le _ _
    _ ≤ ENNReal.ofReal (4 * √(β / α)) + ENNReal.ofReal (2 * η / β) +
          ENNReal.ofReal (2 * η / β) + ENNReal.ofReal (2 * η / β) := by gcongr
    _ = ENNReal.ofReal (4 * √(β / α) + 6 * η / β) := by
        have h0 : 0 ≤ 2 * η / β := by positivity
        rw [← ENNReal.ofReal_add (by positivity) h0, ← ENNReal.ofReal_add (by positivity) h0,
          ← ENNReal.ofReal_add (by positivity) h0]
        congr 1; ring

/-- Optimised cubic sublevel estimate: `Leb{|φ| ≤ η} ≤ 10 (η/α)^{1/3}`. -/
theorem volume_band_cubic_opt {φ φ' : ℝ → ℝ} (hd : ∀ t, HasDerivAt φ (φ' t) t) {α η : ℝ}
    (hα : 0 < α) (hη : 0 ≤ η)
    (hφ' : StrongConvexOn univ α φ' ∨ StrongConcaveOn univ α φ') :
    volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (10 * (η / α) ^ ((1 : ℝ) / 3)) := by
  -- reduce the concave case to the convex one
  have key : ∀ β > 0, volume {t : ℝ | |φ t| ≤ η} ≤ ENNReal.ofReal (4 * √(β / α) + 6 * η / β) := by
    intro β hβ
    rcases hφ' with h | h
    · exact volume_band_cubic hd hα hη hβ h
    · have hd' : ∀ t, HasDerivAt (fun s => -φ s) (-φ' t) t := fun t => (hd t).neg
      have h' : StrongConvexOn univ α (fun s => -φ' s) := UniformConcaveOn.neg h
      have := volume_band_cubic hd' hα hη hβ h'
      simpa only [abs_neg] using this
  rcases eq_or_lt_of_le hη with h0 | hpos
  · -- η = 0: the band is null
    subst h0
    have hz : volume {t : ℝ | |φ t| ≤ 0} = 0 := by
      refine le_antisymm (ENNReal.le_of_forall_pos_le_add fun ε hε _ => ?_) bot_le
      set β : ℝ := α * ((ε : ℝ) / 4) ^ 2 with hβdef
      have hβ : 0 < β := by positivity
      refine (key β hβ).trans ?_
      have : √(β / α) = (ε : ℝ) / 4 := by
        have e : β / α = ((ε : ℝ) / 4) ^ 2 := by rw [hβdef]; field_simp
        rw [e, Real.sqrt_sq (by positivity)]
      rw [this, zero_add, mul_zero, zero_div, add_zero, show 4 * ((ε : ℝ) / 4) = ε by ring,
        ENNReal.ofReal_coe_nnreal]
    rw [hz]; exact bot_le
  · set c := (η / α) ^ ((1 : ℝ) / 3) with hc
    have hcpos : 0 < c := Real.rpow_pos_of_pos (div_pos hpos hα) _
    have hc3 : c ^ 3 = η / α := by
      rw [hc, ← Real.rpow_natCast, ← Real.rpow_mul (div_pos hpos hα).le]
      norm_num
    set β := α * c ^ 2 with hβdef
    have hβ : 0 < β := by positivity
    refine (key β hβ).trans (le_of_eq ?_)
    congr 1
    have e1 : √(β / α) = c := by
      have e : β / α = c ^ 2 := by rw [hβdef]; field_simp
      rw [e, Real.sqrt_sq hcpos.le]
    have e2 : 6 * η / β = 6 * c := by
      rw [hβdef]
      have : η = α * c ^ 3 := by rw [hc3]; field_simp
      rw [this]; field_simp
    rw [e1, e2]; ring

end Cubature.Aux

namespace Cubature.Aux

theorem jump_holder_third {Y : Type*} [PseudoMetricSpace Y] {ν : Measure ℝ}
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
  have hcont : ∀ z, Continuous (φ · z) := fun z =>
    continuous_iff_continuousAt.mpr fun t => (hd t z).continuousAt
  set η := L * dist y y'
  have hη : 0 ≤ η := (abs_nonneg _).trans (hφL 0 y y')
  have hB : 0 ≤ B := (abs_nonneg _).trans (hθB 0 y)
  have hvol := volume_band_cubic_opt (φ := (φ · y)) (φ' := (φ' · y)) (fun t => hd t y) hα hη (hφ' y)
  have hνS : ν.real {t : ℝ | |φ t y| ≤ η} ≤ ρ * (10 * (η / α) ^ ((1 : ℝ) / 3)) := by
    have h3 : ν {t : ℝ | |φ t y| ≤ η} ≤ ENNReal.ofReal (ρ * (10 * (η / α) ^ ((1 : ℝ) / 3))) := by
      rw [ENNReal.ofReal_mul hρ]
      exact (hν _).trans (by gcongr)
    exact ENNReal.toReal_le_of_le_ofReal (by positivity) h3
  refine (jump_diff_le hcont hφL hθm hθB hθL y y').trans ?_
  have := mul_le_mul_of_nonneg_left hνS hB
  calc Lθ * dist y y' + B * ν.real {t : ℝ | |φ t y| ≤ L * dist y y'}
      ≤ Lθ * dist y y' + B * (ρ * (10 * (η / α) ^ ((1 : ℝ) / 3))) := by linarith
    _ = Lθ * dist y y' + 10 * ρ * B * (L * dist y y' / α) ^ ((1 : ℝ) / 3) := by ring

end Cubature.Aux

namespace Cubature.Aux

lemma cube_pos_set {y : ℝ} (hy : 0 ≤ y) :
    {t : ℝ | 0 < t ^ 3 - y} = Ioi (y ^ ((1 : ℝ) / 3)) := by
  set c := y ^ ((1 : ℝ) / 3) with hc
  have hc3 : c ^ 3 = y := by
    rw [hc, ← Real.rpow_natCast, ← Real.rpow_mul hy]; norm_num
  ext t
  simp only [mem_ofPred_eq, mem_Ioi]
  have hmono : StrictMono fun a : ℝ => a ^ 3 := Odd.strictMono_pow (by decide)
  constructor
  · intro h
    have : c ^ 3 < t ^ 3 := by rw [hc3]; linarith
    exact hmono.lt_iff_lt.mp this
  · intro h
    have := hmono h
    simp only at this
    rw [hc3] at this; linarith

theorem gks_example4_two_sided (y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    rexp (-1 / 2) / √(2 * π) * y ^ ((1 : ℝ) / 3) ≤
        preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) 0 -
          preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) y ∧
      preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) 0 -
          preintJump (gaussianReal 0 1) (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) y ≤
        10 / √(2 * π) * (y / 6) ^ ((1 : ℝ) / 3) := by
  set γ := gaussianReal 0 1
  have hP : ∀ z : ℝ, 0 ≤ z → preintJump γ (fun t y => t ^ 3 - y) (fun _ _ => (1 : ℝ)) z =
      γ.real (Ioi (z ^ ((1 : ℝ) / 3))) := by
    intro z hz
    rw [preintJump, cube_pos_set hz, integral_indicator measurableSet_Ioi, setIntegral_const,
      smul_eq_mul, mul_one]
  constructor
  · rw [hP y hy0, hP 0 le_rfl, Real.zero_rpow (by norm_num)]
    set c := y ^ ((1 : ℝ) / 3) with hc
    have hc0 : 0 ≤ c := Real.rpow_nonneg hy0 _
    have hc1 : c ≤ 1 := Real.rpow_le_one hy0 hy1 (by norm_num)
    have hunion : Ioi (0 : ℝ) = Ioc 0 c ∪ Ioi c := (Ioc_union_Ioi_eq_Ioi hc0).symm
    have hdisj : Disjoint (Ioc (0 : ℝ) c) (Ioi c) := by
      rw [Set.disjoint_left]; intro t h1 h2; exact absurd h1.2 (not_le.mpr h2)
    have hsplit : γ.real (Ioi 0) = γ.real (Ioc 0 c) + γ.real (Ioi c) := by
      rw [hunion]; exact measureReal_union hdisj measurableSet_Ioi
    rw [hsplit, add_sub_cancel_right]
    -- lower bound on the Gaussian mass of `(0, c]`
    have hlow : ∀ x ∈ Ioc (0 : ℝ) c, ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) ≤
        gaussianPDF 0 1 x := by
      intro x hx
      rw [gaussianPDF, gaussianPDFReal]
      refine ENNReal.ofReal_le_ofReal ?_
      simp only [NNReal.coe_one, mul_one, sub_zero]
      have hx2 : x ^ 2 ≤ 1 := by nlinarith [hx.1, hx.2]
      have : rexp (-1 / 2) ≤ rexp (-x ^ 2 / 2) := Real.exp_le_exp.mpr (by linarith)
      rw [div_eq_inv_mul]
      exact mul_le_mul_of_nonneg_left this (by positivity)
    have hmeas : ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) * volume (Ioc (0 : ℝ) c) ≤
        γ (Ioc 0 c) := by
      rw [gaussianReal_apply 0 one_ne_zero, ← setLIntegral_const]
      exact setLIntegral_mono (by fun_prop) hlow
    rw [Real.volume_Ioc, ← ENNReal.ofReal_mul (by positivity), sub_zero] at hmeas
    have hfin : γ (Ioc 0 c) ≠ ⊤ := measure_ne_top _ _
    exact (ENNReal.ofReal_le_iff_le_toReal hfin).mp hmeas
  · have h6 : StrongConvexOn univ 6 (fun t : ℝ => 3 * t ^ 2) := by
      rw [strongConvexOn_iff_convex]
      have : (fun t : ℝ => 3 * t ^ 2 - 6 / 2 * ‖t‖ ^ 2) = fun _ => 0 := by
        funext t; rw [Real.norm_eq_abs, sq_abs]; ring
      rw [this]
      exact convexOn_const _ convex_univ
    have hd : ∀ (t : ℝ) (z : ℝ), HasDerivAt (fun s => s ^ 3 - z) (3 * t ^ 2) t := by
      intro t z
      have := (hasDerivAt_pow 3 t).sub_const z
      simpa using this
    have h := jump_holder_third (Y := ℝ) (ρ := (√(2 * π))⁻¹) (by positivity) gaussianReal_le_volume
      (φ := fun t y => t ^ 3 - y) (φ' := fun t _ => 3 * t ^ 2) (θ := fun _ _ => (1 : ℝ))
      (α := 6) (L := 1) (Lθ := 0) (B := 1) (by norm_num) hd (fun _ => Or.inl h6)
      (fun t y y' => by
        rw [Real.dist_eq, show t ^ 3 - y - (t ^ 3 - y') = -(y - y') by ring, abs_neg, one_mul])
      (fun _ => measurable_const) (fun _ _ => by simp) (fun _ _ _ => by simp) 0 y
    rw [Real.dist_eq, zero_sub, abs_neg, abs_of_nonneg hy0] at h
    have := (abs_le.mp h).2
    calc _ ≤ 0 * y + 10 * (√(2 * π))⁻¹ * 1 * (1 * y / 6) ^ ((1 : ℝ) / 3) := this
      _ = 10 / √(2 * π) * (y / 6) ^ ((1 : ℝ) / 3) := by ring_nf

end Cubature.Aux
