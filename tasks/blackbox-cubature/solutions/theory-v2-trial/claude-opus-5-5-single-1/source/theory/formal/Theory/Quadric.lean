import Theory.Defs
import Theory.Tail

/-!
# Preintegration of a jump on a quadric (result R3)

For `a ≠ 0` the Gaussian mass `Q(a, b, c) = γ({t : a t² + b t + c > 0})` is an explicit function of
the two roots `(-b ± √D)/(2a)`, `D = b² - 4ac`, and is `1/2`-Hölder in `(b, D)`:
`|Q(a,b,c) - Q(a,b',c')| ≤ (|b - b'| + √|D - D'|) / (|a| √(2π))`.
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory

namespace GKLS

/-- The Gaussian mass of `{t : a t² + b t + c > 0}`. -/
noncomputable def quadMass (a b c : ℝ) : ℝ :=
  ∫ t, ind (a * t ^ 2 + b * t + c) ∂(gaussianReal 0 1)

lemma rho_le (x : ℝ) : rho x ≤ 1 / √(2 * π) := by
  unfold rho gaussianPDFReal
  have h1 : rexp (-(x - 0) ^ 2 / (2 * ((1 : NNReal) : ℝ))) ≤ 1 := by
    apply Real.exp_le_one_iff.mpr
    have : 0 ≤ (x - 0) ^ 2 := sq_nonneg _
    have h2 : (0 : ℝ) < 2 * ((1 : NNReal) : ℝ) := by norm_num
    exact div_nonpos_of_nonpos_of_nonneg (by linarith) h2.le
  have h3 : (0 : ℝ) ≤ (√(2 * π * ((1 : NNReal) : ℝ)))⁻¹ := by positivity
  calc (√(2 * π * ((1 : NNReal) : ℝ)))⁻¹ * rexp (-(x - 0) ^ 2 / (2 * ((1 : NNReal) : ℝ)))
      ≤ (√(2 * π * ((1 : NNReal) : ℝ)))⁻¹ * 1 := mul_le_mul_of_nonneg_left h1 h3
    _ = 1 / √(2 * π) := by simp

/-- The Gaussian tail is `1/√(2π)`-Lipschitz. -/
lemma abs_gtail_sub_le (u v : ℝ) : |gtail u - gtail v| ≤ |u - v| / √(2 * π) := by
  have hb : ∀ x ∈ (Set.univ : Set ℝ), ‖deriv gtail x‖ ≤ 1 / √(2 * π) := by
    intro x _
    rw [deriv_gtail, Real.norm_eq_abs, abs_neg, abs_of_nonneg (gaussianPDFReal_nonneg 0 1 x)]
    exact rho_le x
  have h := convex_univ.norm_image_sub_le_of_norm_deriv_le
    (fun x _ => (hasDerivAt_gtail x).differentiableAt) hb (mem_univ v) (mem_univ u)
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at h
  calc |gtail u - gtail v| ≤ 1 / √(2 * π) * |u - v| := h
    _ = |u - v| / √(2 * π) := by ring

lemma gtail_sub_eq_measureReal {u v : ℝ} (huv : u ≤ v) :
    gtail u - gtail v = (gaussianReal 0 1).real (Ioo u v) := by
  have h1 : (gaussianReal 0 1).real (Ioo u v) = ∫ x in Ioo u v, rho x := by
    rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero (Ioo u v),
      ENNReal.toReal_ofReal]
    exact setIntegral_nonneg measurableSet_Ioo (fun x _ => gaussianPDFReal_nonneg 0 1 x)
  rw [h1, gtail_eq_sub u, gtail_eq_sub v, ← integral_Ioc_eq_integral_Ioo,
    ← intervalIntegral.integral_of_le huv]
  have := intervalIntegral.integral_add_adjacent_intervals (μ := volume) (f := rho) (a := 0) (b := u)
    (c := v) (continuous_rho.intervalIntegrable _ _) (continuous_rho.intervalIntegrable _ _)
  linarith

/-- The two roots `(-b ∓ √D)/(2a)` (with `√D = 0` if `D < 0`). -/
noncomputable def rootLo (a b c : ℝ) : ℝ := (-b - √(b ^ 2 - 4 * a * c)) / (2 * a)
noncomputable def rootHi (a b c : ℝ) : ℝ := (-b + √(b ^ 2 - 4 * a * c)) / (2 * a)

/-- For `a < 0`, `{t : a t² + b t + c > 0}` is the open interval between the roots. -/
lemma quad_pos_iff_neg {a b c : ℝ} (ha : a < 0) (t : ℝ) :
    0 < a * t ^ 2 + b * t + c ↔ rootHi a b c < t ∧ t < rootLo a b c := by
  set D := b ^ 2 - 4 * a * c with hD
  set s := √D with hs
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have h2a : 2 * a < 0 := by linarith
  have hHi : rootHi a b c = (-b + s) / (2 * a) := rfl
  have hLo : rootLo a b c = (-b - s) / (2 * a) := rfl
  rw [hHi, hLo, div_lt_iff_of_neg h2a, lt_div_iff_of_neg h2a]
  rcases le_or_gt 0 D with hD0 | hD0
  · have hss : s * s = D := Real.mul_self_sqrt hD0
    -- a t² + b t + c = a (t - r₁)(t - r₂), i.e. 4a(a t² + b t + c) = (2at + b)² - s²
    have key : 4 * a * (a * t ^ 2 + b * t + c) = (2 * a * t + b - s) * (2 * a * t + b + s) := by
      nlinarith [hss]
    constructor
    · intro h
      have h4 : 4 * a * (a * t ^ 2 + b * t + c) < 0 := by nlinarith
      rw [key] at h4
      rcases lt_or_gt_of_ne (show 2 * a * t + b - s ≠ 0 by intro h0; rw [h0] at h4; simp at h4) with
        h5 | h5
      · have h6 : 0 < 2 * a * t + b + s := by
          by_contra h6; push Not at h6; nlinarith
        constructor <;> nlinarith
      · have h6 : 2 * a * t + b + s < 0 := by
          by_contra h6; push Not at h6; nlinarith
        exfalso; nlinarith
    · rintro ⟨h1, h2⟩
      have h4 : (2 * a * t + b - s) * (2 * a * t + b + s) < 0 := by
        apply mul_neg_of_neg_of_pos <;> linarith
      rw [← key] at h4
      nlinarith
  · have hs' : s = 0 := Real.sqrt_eq_zero'.mpr hD0.le
    constructor
    · intro h
      exfalso
      have : 4 * a * (a * t ^ 2 + b * t + c) = (2 * a * t + b) ^ 2 - D := by rw [hD]; ring
      nlinarith [sq_nonneg (2 * a * t + b)]
    · rintro ⟨h1, h2⟩
      rw [hs'] at h1 h2
      exfalso; linarith

/-- `|√u - √v| ≤ √|u - v|` (with `√x = 0` for `x < 0`). -/
lemma abs_sqrt_sub_sqrt_le (u v : ℝ) : |√u - √v| ≤ √|u - v| := by
  wlog huv : v ≤ u generalizing u v
  · rw [abs_sub_comm, abs_sub_comm u v]; exact this v u (le_of_not_ge huv)
  have hmono : √v ≤ √u := Real.sqrt_le_sqrt huv
  rw [abs_of_nonneg (sub_nonneg.mpr hmono), abs_of_nonneg (sub_nonneg.mpr huv)]
  rcases le_or_gt 0 v with hv | hv
  · -- `√u ≤ √v + √(u - v)` by squaring
    have h1 : √u ≤ √v + √(u - v) := by
      rw [Real.sqrt_le_left]
      · have e1 := Real.sq_sqrt hv
        have e2 := Real.sq_sqrt (sub_nonneg.mpr huv)
        nlinarith [Real.sqrt_nonneg v, Real.sqrt_nonneg (u - v)]
      · positivity
    linarith
  · have hv0 : √v = 0 := Real.sqrt_eq_zero'.mpr hv.le
    rw [hv0, sub_zero]
    exact Real.sqrt_le_sqrt (by linarith)

/-- The mass of the positivity set for `a < 0`. -/
lemma quadMass_of_neg {a b c : ℝ} (ha : a < 0) :
    quadMass a b c = gtail (rootHi a b c) - gtail (rootLo a b c) := by
  have hle : rootHi a b c ≤ rootLo a b c := by
    unfold rootHi rootLo
    apply div_le_div_of_nonpos_of_le (by linarith)
    linarith [Real.sqrt_nonneg (b ^ 2 - 4 * a * c)]
  have h : (fun t => ind (a * t ^ 2 + b * t + c)) = (Ioo (rootHi a b c) (rootLo a b c)).indicator 1 := by
    funext t
    simp only [ind, Set.indicator, Set.mem_Ioo, Pi.one_apply, quad_pos_iff_neg ha t]
  rw [quadMass, h, integral_indicator_one measurableSet_Ioo, gtail_sub_eq_measureReal hle]

lemma ind_add_ind_neg (u : ℝ) (hu : u ≠ 0) : ind u + ind (-u) = 1 := by
  unfold ind
  rcases lt_or_gt_of_ne hu with h | h
  · simp [h, not_lt.mpr h.le, neg_pos.mpr h]
  · simp [h, not_lt.mpr (neg_nonpos.mpr h.le)]

/-- The zero set of a quadratic with `a ≠ 0` is contained in the two roots. -/
lemma quad_eq_zero_imp {a b c : ℝ} (ha : a ≠ 0) {t : ℝ} (ht : a * t ^ 2 + b * t + c = 0) :
    t = rootLo a b c ∨ t = rootHi a b c := by
  set D := b ^ 2 - 4 * a * c with hD
  have hsq : (2 * a * t + b) ^ 2 = D := by
    have : 4 * a * (a * t ^ 2 + b * t + c) = (2 * a * t + b) ^ 2 - D := by rw [hD]; ring
    rw [ht, mul_zero] at this; linarith
  have hD0 : 0 ≤ D := by rw [← hsq]; positivity
  have hs : √D = |2 * a * t + b| := by rw [← hsq, Real.sqrt_sq_eq_abs]
  have h2a : 2 * a ≠ 0 := by positivity
  unfold rootLo rootHi
  rw [← hD, hs]
  rcases le_or_gt 0 (2 * a * t + b) with h | h
  · rw [abs_of_nonneg h]; right; field_simp; ring
  · rw [abs_of_neg h]; left; field_simp; ring

/-- For `a ≠ 0`, the masses of `{q > 0}` and `{-q > 0}` add up to one. -/
lemma quadMass_add_neg {a b c : ℝ} (ha : a ≠ 0) :
    quadMass a b c + quadMass (-a) (-b) (-c) = 1 := by
  have hnull : ∀ᵐ t ∂(gaussianReal 0 1), a * t ^ 2 + b * t + c ≠ 0 := by
    have hγ : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
    have hsub : {t : ℝ | ¬ (a * t ^ 2 + b * t + c ≠ 0)} ⊆ {rootLo a b c, rootHi a b c} := by
      intro t ht
      simp only [ne_eq, not_not, Set.mem_setOf_eq] at ht
      rcases quad_eq_zero_imp ha ht with h | h <;> simp [h]
    rw [ae_iff]
    exact measure_mono_null hsub (Set.Finite.measure_zero (Set.toFinite _) _)
  have hint : ∀ a b c : ℝ, Integrable (fun t => ind (a * t ^ 2 + b * t + c)) (gaussianReal 0 1) := by
    intro a b c
    refine Integrable.mono' (integrable_const (1 : ℝ)) ?_ (ae_of_all _ fun t => ?_)
    · have hm : Measurable ind := by
        have : ind = (Ioi (0 : ℝ)).indicator 1 := by funext u; simp [ind, Set.indicator]
        rw [this]; exact measurable_one.indicator measurableSet_Ioi
      exact (hm.comp (by fun_prop)).aestronglyMeasurable
    · unfold ind; split_ifs <;> simp
  unfold quadMass
  rw [← integral_add (hint a b c) (hint (-a) (-b) (-c))]
  rw [show (1 : ℝ) = ∫ _t, (1 : ℝ) ∂(gaussianReal 0 1) by simp]
  refine integral_congr_ae ?_
  filter_upwards [hnull] with t ht
  have : -a * t ^ 2 + -b * t + -c = -(a * t ^ 2 + b * t + c) := by ring
  rw [this]
  exact ind_add_ind_neg _ ht

lemma rootHi_neg (a b c : ℝ) : rootHi (-a) (-b) (-c) = rootLo a b c := by
  unfold rootHi rootLo
  rw [show (-b) ^ 2 - 4 * -a * -c = b ^ 2 - 4 * a * c by ring, neg_neg]
  rcases eq_or_ne a 0 with h | h
  · simp [h]
  · field_simp; ring

lemma rootLo_neg (a b c : ℝ) : rootLo (-a) (-b) (-c) = rootHi a b c := by
  unfold rootHi rootLo
  rw [show (-b) ^ 2 - 4 * -a * -c = b ^ 2 - 4 * a * c by ring, neg_neg]
  rcases eq_or_ne a 0 with h | h
  · simp [h]
  · field_simp; ring

lemma abs_root_sub_le (a b c b' c' : ℝ) (sgn : ℝ) (hsgn : |sgn| = 1) :
    |(-b + sgn * √(b ^ 2 - 4 * a * c)) / (2 * a) - (-b' + sgn * √(b' ^ 2 - 4 * a * c')) / (2 * a)|
      ≤ (|b - b'| + √|(b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c')|) / (2 * |a|) := by
  rw [← sub_div, abs_div, abs_mul, abs_two]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have h1 := abs_sqrt_sub_sqrt_le (b ^ 2 - 4 * a * c) (b' ^ 2 - 4 * a * c')
  calc |-b + sgn * √(b ^ 2 - 4 * a * c) - (-b' + sgn * √(b' ^ 2 - 4 * a * c'))|
      = |(b' - b) + sgn * (√(b ^ 2 - 4 * a * c) - √(b' ^ 2 - 4 * a * c'))| := by ring_nf
    _ ≤ |b' - b| + |sgn * (√(b ^ 2 - 4 * a * c) - √(b' ^ 2 - 4 * a * c'))| := abs_add_le _ _
    _ = |b - b'| + |√(b ^ 2 - 4 * a * c) - √(b' ^ 2 - 4 * a * c')| := by
        rw [abs_mul, hsgn, one_mul, abs_sub_comm b' b]
    _ ≤ |b - b'| + √|(b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c')| := by linarith

/-- **Hölder-1/2 bound for `a < 0`.** -/
lemma quadMass_holder_neg {a : ℝ} (ha : a < 0) (b c b' c' : ℝ) :
    |quadMass a b c - quadMass a b' c'| ≤
      (|b - b'| + √|(b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c')|) / (|a| * √(2 * π)) := by
  rw [quadMass_of_neg ha, quadMass_of_neg ha]
  set E := (|b - b'| + √|(b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c')|) with hE
  have hHi : |rootHi a b c - rootHi a b' c'| ≤ E / (2 * |a|) := by
    have := abs_root_sub_le a b c b' c' 1 (by simp)
    simpa [rootHi, one_mul] using this
  have hLo : |rootLo a b c - rootLo a b' c'| ≤ E / (2 * |a|) := by
    have := abs_root_sub_le a b c b' c' (-1) (by simp)
    have e : ∀ b c : ℝ, (-b + -1 * √(b ^ 2 - 4 * a * c)) / (2 * a) = rootLo a b c := by
      intro b c; unfold rootLo; ring
    rw [e, e] at this
    exact this
  have g1 := abs_gtail_sub_le (rootHi a b c) (rootHi a b' c')
  have g2 := abs_gtail_sub_le (rootLo a b c) (rootLo a b' c')
  have hpi : 0 < √(2 * π) := by positivity
  have ha' : 0 < |a| := abs_pos.mpr ha.ne
  calc |gtail (rootHi a b c) - gtail (rootLo a b c) - (gtail (rootHi a b' c') - gtail (rootLo a b' c'))|
      = |(gtail (rootHi a b c) - gtail (rootHi a b' c')) - (gtail (rootLo a b c) - gtail (rootLo a b' c'))| := by
        ring_nf
    _ ≤ |gtail (rootHi a b c) - gtail (rootHi a b' c')| + |gtail (rootLo a b c) - gtail (rootLo a b' c')| :=
        abs_sub _ _
    _ ≤ |rootHi a b c - rootHi a b' c'| / √(2 * π) + |rootLo a b c - rootLo a b' c'| / √(2 * π) :=
        add_le_add g1 g2
    _ ≤ E / (2 * |a|) / √(2 * π) + E / (2 * |a|) / √(2 * π) := by
        gcongr
    _ = E / (|a| * √(2 * π)) := by field_simp; ring

/-- **Hölder-1/2 bound** for the Gaussian mass of `{t : a t² + b t + c > 0}`, `a ≠ 0`:
`|Q(a,b,c) - Q(a,b',c')| ≤ (|b - b'| + √|D - D'|) / (|a| √(2π))`, `D = b² - 4ac`. -/
lemma quadMass_holder {a : ℝ} (ha : a ≠ 0) (b c b' c' : ℝ) :
    |quadMass a b c - quadMass a b' c'| ≤
      (|b - b'| + √|(b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c')|) / (|a| * √(2 * π)) := by
  rcases lt_or_gt_of_ne ha with h | h
  · exact quadMass_holder_neg h b c b' c'
  · have h1 := quadMass_add_neg (a := a) (b := b) (c := c) ha
    have h2 := quadMass_add_neg (a := a) (b := b') (c := c') ha
    have h3 := quadMass_holder_neg (a := -a) (by linarith) (-b) (-c) (-b') (-c')
    have e1 : |-b - -b'| = |b - b'| := by rw [show -b - -b' = -(b - b') by ring, abs_neg]
    have e2 : ((-b) ^ 2 - 4 * -a * -c) - ((-b') ^ 2 - 4 * -a * -c') =
        (b ^ 2 - 4 * a * c) - (b' ^ 2 - 4 * a * c') := by ring
    rw [e1, e2, abs_neg] at h3
    have : quadMass a b c - quadMass a b' c' = -(quadMass (-a) (-b) (-c) - quadMass (-a) (-b') (-c')) := by
      linarith
    rw [this, abs_neg]
    exact h3

lemma rho_neg (x : ℝ) : rho (-x) = rho x := by
  unfold rho gaussianPDFReal; ring_nf

lemma rho_le_of_abs_le {x R : ℝ} (hx : |x| ≤ R) : rho R ≤ rho x := by
  rcases le_or_gt 0 x with h | h
  · rw [abs_of_nonneg h] at hx; exact rho_le_of_le h hx
  · rw [abs_of_neg h] at hx; rw [← rho_neg x]; exact rho_le_of_le (by linarith) hx

/-- **Lower bound** for `a < 0`: if both roots lie in `[-R, R]` then
`Q(a,b,c) ≥ ρ(R) √D / |a|`. -/
lemma quadMass_lower {a b c R : ℝ} (ha : a < 0)
    (hR : |b| + √(b ^ 2 - 4 * a * c) ≤ 2 * |a| * R) :
    rho R * √(b ^ 2 - 4 * a * c) / |a| ≤ quadMass a b c := by
  rw [quadMass_of_neg ha]
  set s := √(b ^ 2 - 4 * a * c) with hs
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have habs : |a| = -a := abs_of_neg ha
  have hHi : rootHi a b c = (b - s) / (2 * |a|) := by
    unfold rootHi; rw [habs, ← hs]; field_simp; ring
  have hLo : rootLo a b c = (b + s) / (2 * |a|) := by
    unfold rootLo; rw [habs, ← hs]; field_simp; ring
  have ha' : 0 < |a| := abs_pos.mpr ha.ne
  have hdiff : rootLo a b c - rootHi a b c = s / |a| := by
    rw [hHi, hLo]; field_simp; ring
  have hbound : ∀ x, rootHi a b c ≤ x → x ≤ rootLo a b c → |x| ≤ R := by
    intro x h1 h2
    rw [hHi] at h1; rw [hLo] at h2
    rw [div_le_iff₀ (by positivity)] at h1
    rw [le_div_iff₀ (by positivity)] at h2
    rw [abs_le]
    constructor
    · have := neg_abs_le b
      nlinarith
    · have := le_abs_self b
      nlinarith
  rcases eq_or_lt_of_le (show rootHi a b c ≤ rootLo a b c by linarith [hdiff, div_nonneg hs0 ha'.le])
    with heq | hlt
  · have : s = 0 := by
      have : s / |a| = 0 := by rw [← hdiff, heq, sub_self]
      rcases div_eq_zero_iff.mp this with h | h
      · exact h
      · exact absurd h ha'.ne'
    rw [heq, this]; simp
  · obtain ⟨ξ, hξ, hξd⟩ := exists_hasDerivAt_eq_slope gtail (fun x => -rho x) hlt
      contDiff_gtail.continuous.continuousOn (fun x _ => hasDerivAt_gtail x)
    have hpos : 0 < rootLo a b c - rootHi a b c := sub_pos.mpr hlt
    have key : gtail (rootHi a b c) - gtail (rootLo a b c) = rho ξ * (rootLo a b c - rootHi a b c) := by
      field_simp at hξd; linarith
    rw [key, hdiff]
    have hρ : rho R ≤ rho ξ := rho_le_of_abs_le (hbound ξ hξ.1.le hξ.2.le)
    have : rho R * s / |a| = rho R * (s / |a|) := by ring
    rw [this]
    exact mul_le_mul_of_nonneg_right hρ (div_nonneg hs0 ha'.le)

/-- Preintegrating the jump across a surface that is quadratic in `x_j`. -/
lemma preintegrate_quadricJump {n : ℕ} (j : Fin (n + 1)) (a : ℝ) (β κ : (Fin n → ℝ) → ℝ)
    (y : Fin n → ℝ) :
    preintegrate j (fun x => ind (a * x j ^ 2 + β (fun k => x (j.succAbove k)) * x j +
      κ (fun k => x (j.succAbove k)))) y = quadMass a (β y) (κ y) := by
  have h : ∀ t : ℝ, (fun k => (Fin.insertNth j t y : Fin (n + 1) → ℝ) (j.succAbove k)) = y := by
    intro t; funext k; simp [Fin.insertNth_apply_succAbove]
  simp only [preintegrate, quadMass, Fin.insertNth_apply_same, h]

lemma quadMass_eq_zero {a b c : ℝ} (ha : a < 0) (hD : b ^ 2 - 4 * a * c ≤ 0) :
    quadMass a b c = 0 := by
  rw [quadMass_of_neg ha]
  have hs : √(b ^ 2 - 4 * a * c) = 0 := Real.sqrt_eq_zero'.mpr hD
  have : rootHi a b c = rootLo a b c := by unfold rootHi rootLo; rw [hs]; ring
  rw [this, sub_self]

/-- Assembly of R3. -/
theorem quadric_assembly (n : ℕ) (j : Fin (n + 1)) (a : ℝ) (ha : a ≠ 0)
    (β κ : (Fin n → ℝ) → ℝ) :
    (∀ y y' : Fin n → ℝ,
      |preintegrate j (quadricJump j a β κ) y - preintegrate j (quadricJump j a β κ) y'| ≤
        (|β y - β y'| + √|(β y ^ 2 - 4 * a * κ y) - (β y' ^ 2 - 4 * a * κ y')|) /
          (|a| * √(2 * π))) ∧
    (a < 0 → ∀ y : Fin n → ℝ, β y ^ 2 - 4 * a * κ y ≤ 0 →
      preintegrate j (quadricJump j a β κ) y = 0) ∧
    (a < 0 → ∀ (R : ℝ) (y : Fin n → ℝ), |β y| + √(β y ^ 2 - 4 * a * κ y) ≤ 2 * |a| * R →
      gaussianPDFReal 0 1 R * √(β y ^ 2 - 4 * a * κ y) / |a| ≤
        preintegrate j (quadricJump j a β κ) y) := by
  have hP : ∀ y, preintegrate j (quadricJump j a β κ) y = quadMass a (β y) (κ y) :=
    fun y => preintegrate_quadricJump j a β κ y
  refine ⟨fun y y' => ?_, fun hneg y hD => ?_, fun hneg R y hR => ?_⟩
  · rw [hP, hP]
    exact quadMass_holder ha (β y) (κ y) (β y') (κ y')
  · rw [hP]
    exact quadMass_eq_zero hneg hD
  · rw [hP]
    exact quadMass_lower hneg hR

end GKLS
