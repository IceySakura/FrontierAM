import Theory.Defs
import Theory.Preint

/-!
# Preintegration of quadric jump surfaces in an arbitrary direction (auxiliary proofs for R6)

For `φ(t) = a t² + β t + γ` with `a ≠ 0`, the Gaussian mass of `{φ > 0}` depends on
`(β, γ)` only through the centre `m = -β/(2a)` and the half-width
`w = √(max(D, 0)) / (2|a|)`, `D = β² - 4aγ`; hence it is `½`-Hölder in `D`.
-/

open MeasureTheory ProbabilityTheory Real Set

namespace Cubature.Aux

lemma abs_sqrt_sub_sqrt_le {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) : |√x - √y| ≤ √|x - y| := by
  wlog h : y ≤ x generalizing x y
  · rw [abs_sub_comm, abs_sub_comm x y]; exact this hy hx (le_of_not_ge h)
  rw [abs_of_nonneg (sub_nonneg.mpr (Real.sqrt_le_sqrt h)), abs_of_nonneg (sub_nonneg.mpr h),
    sub_le_iff_le_add, Real.sqrt_le_left (by positivity)]
  have h1 := Real.sq_sqrt (sub_nonneg.mpr h)
  have h2 := Real.sq_sqrt hy
  nlinarith [Real.sqrt_nonneg (x - y), Real.sqrt_nonneg y, mul_nonneg (Real.sqrt_nonneg (x - y))
    (Real.sqrt_nonneg y)]

/-- Positivity set of a concave quadratic: an open interval centred at `-β/(2a)`. -/
lemma quad_pos_set_of_neg {a β γ : ℝ} (ha : a < 0) :
    {t : ℝ | 0 < a * t ^ 2 + β * t + γ} =
      Ioo (-β / (2 * a) - √(max (β ^ 2 - 4 * a * γ) 0) / (2 * |a|))
          (-β / (2 * a) + √(max (β ^ 2 - 4 * a * γ) 0) / (2 * |a|)) := by
  ext t
  simp only [mem_ofPred_eq, mem_Ioo]
  set D := β ^ 2 - 4 * a * γ with hD
  set m := -β / (2 * a) with hm
  set w := √(max D 0) / (2 * |a|) with hw
  have ha0 : a ≠ 0 := ha.ne
  have habs : |a| = -a := abs_of_neg ha
  have hkey : 4 * a * (a * t ^ 2 + β * t + γ) = 4 * a ^ 2 * (t - m) ^ 2 - D := by
    rw [hm, hD]; field_simp; ring
  have hw0 : 0 ≤ w := by positivity
  have hw2 : 4 * a ^ 2 * w ^ 2 = max D 0 := by
    rw [hw, div_pow, Real.sq_sqrt (le_max_right _ _), mul_pow, sq_abs]
    field_simp
    ring
  have ha2 : 0 < 4 * a ^ 2 := by positivity
  constructor
  · intro h
    -- 4a φ < 0, i.e. 4a²(t-m)² < D ≤ max D 0 = 4a² w²
    have h1 : 4 * a * (a * t ^ 2 + β * t + γ) < 0 := by nlinarith
    have h2 : (t - m) ^ 2 < w ^ 2 := by
      have : 4 * a ^ 2 * (t - m) ^ 2 < 4 * a ^ 2 * w ^ 2 := by
        rw [hw2]; nlinarith [le_max_left D 0]
      exact lt_of_mul_lt_mul_left this ha2.le
    have h3 := abs_lt.mp (abs_lt_of_sq_lt_sq' h2 hw0 |> fun h => abs_lt.mpr h)
    constructor <;> linarith [h3.1, h3.2]
  · rintro ⟨h1, h2⟩
    have h3 : |t - m| < w := abs_lt.mpr ⟨by linarith, by linarith⟩
    have h4 : (t - m) ^ 2 < w ^ 2 := by
      have := pow_lt_pow_left₀ h3 (abs_nonneg _) two_ne_zero
      rwa [sq_abs] at this
    have hwpos : 0 < w := lt_of_le_of_lt (abs_nonneg _) h3
    have hDpos : 0 < D := by
      by_contra hneg
      push Not at hneg
      have : max D 0 = 0 := max_eq_right hneg
      rw [this] at hw2
      have : w ^ 2 = 0 := by nlinarith
      nlinarith
    have h5 : 4 * a ^ 2 * (t - m) ^ 2 < D := by
      have := mul_lt_mul_of_pos_left h4 ha2
      rwa [hw2, max_eq_left hDpos.le] at this
    have h6 : 4 * a * (a * t ^ 2 + β * t + γ) < 0 := by rw [hkey]; linarith
    by_contra hneg
    push Not at hneg
    nlinarith

/-- The zero set of a genuine quadratic is Gaussian-null. -/
lemma quad_zero_null {a β γ : ℝ} (ha : a ≠ 0) :
    gaussianReal 0 1 {t : ℝ | a * t ^ 2 + β * t + γ = 0} = 0 := by
  have hsub : {t : ℝ | a * t ^ 2 + β * t + γ = 0} ⊆
      {(√(β ^ 2 - 4 * a * γ) - β) / (2 * a), (-√(β ^ 2 - 4 * a * γ) - β) / (2 * a)} := by
    intro t ht
    simp only [mem_ofPred_eq] at ht
    have hd : discrim a β γ = (2 * a * t + β) ^ 2 :=
      discrim_eq_sq_of_quadratic_eq_zero (by rw [← ht]; ring)
    rw [discrim] at hd
    have hs : √(β ^ 2 - 4 * a * γ) = |2 * a * t + β| := by rw [hd, Real.sqrt_sq_eq_abs]
    simp only [mem_insert_iff, mem_singleton_iff]
    rcases abs_cases (2 * a * t + β) with ⟨h1, _⟩ | ⟨h1, _⟩
    · left; rw [hs, h1]; field_simp; ring
    · right; rw [hs, h1]; field_simp; ring
  have hfin : ({(√(β ^ 2 - 4 * a * γ) - β) / (2 * a),
      (-√(β ^ 2 - 4 * a * γ) - β) / (2 * a)} : Set ℝ).Finite := by
    exact (Set.finite_singleton _).insert _
  have hvol : volume ({(√(β ^ 2 - 4 * a * γ) - β) / (2 * a),
      (-√(β ^ 2 - 4 * a * γ) - β) / (2 * a)} : Set ℝ) = 0 := hfin.measure_zero _
  exact measure_mono_null hsub
    ((gaussianReal_absolutelyContinuous 0 one_ne_zero) hvol)

lemma gauss_Icc_le (p q : ℝ) :
    (gaussianReal 0 1).real (Icc p q) ≤ (√(2 * π))⁻¹ * |q - p| := by
  have h1 := gaussianReal_le_volume (Icc p q)
  rw [Real.volume_Icc] at h1
  have h2 : gaussianReal 0 1 (Icc p q) ≤ ENNReal.ofReal ((√(2 * π))⁻¹ * |q - p|) := by
    rw [ENNReal.ofReal_mul (p := (√(2 * π))⁻¹) (by positivity)]
    refine h1.trans ?_
    gcongr
    exact le_abs_self _
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) h2

lemma gauss_Ioo_sub_le (l r l' r' : ℝ) :
    (gaussianReal 0 1).real (Ioo l r) - (gaussianReal 0 1).real (Ioo l' r') ≤
      (√(2 * π))⁻¹ * (|l - l'| + |r - r'|) := by
  set γ := gaussianReal 0 1
  have hsub : Ioo l r ⊆ Ioo l' r' ∪ (Icc (min l l') (max l l') ∪ Icc (min r r') (max r r')) := by
    intro t ht
    rcases ht with ⟨h1, h2⟩
    by_cases hl : l' < t
    · by_cases hr : t < r'
      · exact Or.inl ⟨hl, hr⟩
      · push Not at hr
        exact Or.inr (Or.inr ⟨le_trans (min_le_right _ _) hr, le_trans h2.le (le_max_left _ _)⟩)
    · push Not at hl
      exact Or.inr (Or.inl ⟨le_trans (min_le_left _ _) h1.le, le_trans hl (le_max_right _ _)⟩)
  have h1 : γ.real (Ioo l r) ≤ γ.real (Ioo l' r') +
      (γ.real (Icc (min l l') (max l l')) + γ.real (Icc (min r r') (max r r'))) := by
    refine (measureReal_mono hsub).trans ?_
    refine (measureReal_union_le _ _).trans ?_
    gcongr
    exact measureReal_union_le _ _
  have h2 := gauss_Icc_le (min l l') (max l l')
  have h3 := gauss_Icc_le (min r r') (max r r')
  rw [max_sub_min_eq_abs', abs_abs] at h2 h3
  rw [mul_add]
  linarith

/-- Core estimate for a concave quadratic. -/
lemma quad_mass_diff_of_neg {a β γ β' γ' : ℝ} (ha : a < 0) :
    (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β * t + γ} -
        (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'} ≤
      (|β - β'| + √|(β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ')|) / (√(2 * π) * |a|) := by
  rw [quad_pos_set_of_neg ha, quad_pos_set_of_neg ha]
  refine (gauss_Ioo_sub_le _ _ _ _).trans ?_
  set D := β ^ 2 - 4 * a * γ
  set D' := β' ^ 2 - 4 * a * γ'
  have habs : 0 < |a| := abs_pos.mpr ha.ne
  have hm : |-β / (2 * a) - -β' / (2 * a)| = |β - β'| / (2 * |a|) := by
    rw [show -β / (2 * a) - -β' / (2 * a) = -(β - β') / (2 * a) by ring, abs_div, abs_neg,
      abs_mul, abs_two]
  have hw : |√(max D 0) / (2 * |a|) - √(max D' 0) / (2 * |a|)| ≤ √|D - D'| / (2 * |a|) := by
    rw [← sub_div, abs_div, abs_of_pos (by positivity : (0:ℝ) < 2 * |a|)]
    gcongr
    refine (abs_sqrt_sub_sqrt_le (le_max_right _ _) (le_max_right _ _)).trans ?_
    exact Real.sqrt_le_sqrt (abs_max_sub_max_le_abs _ _ _)
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
  rw [hm] at e1 e2
  have hsum : |(-β / (2 * a) - √(max D 0) / (2 * |a|)) - (-β' / (2 * a) - √(max D' 0) / (2 * |a|))|
      + |(-β / (2 * a) + √(max D 0) / (2 * |a|)) - (-β' / (2 * a) + √(max D' 0) / (2 * |a|))|
      ≤ (|β - β'| + √|D - D'|) / |a| := by
    have : (|β - β'| + √|D - D'|) / |a| = 2 * (|β - β'| / (2 * |a|)) + 2 * (√|D - D'| / (2 * |a|))
      := by field_simp
    rw [this]; linarith
  calc (√(2 * π))⁻¹ * (_ + _) ≤ (√(2 * π))⁻¹ * ((|β - β'| + √|D - D'|) / |a|) := by gcongr
    _ = (|β - β'| + √|D - D'|) / (√(2 * π) * |a|) := by field_simp

/-- Complement trick: for `a > 0` the mass of `{φ > 0}` is `1 -` the mass of `{-φ > 0}`. -/
lemma quad_mass_neg {a β γ : ℝ} (ha : a ≠ 0) :
    (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β * t + γ} =
      1 - (gaussianReal 0 1).real {t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)} := by
  set γm := gaussianReal 0 1
  have hcont : Continuous fun t : ℝ => a * t ^ 2 + β * t + γ := by fun_prop
  have hm1 : MeasurableSet {t : ℝ | 0 < a * t ^ 2 + β * t + γ} :=
    measurableSet_lt measurable_const hcont.measurable
  have hm2 : MeasurableSet {t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)} :=
    measurableSet_lt measurable_const (by fun_prop)
  have hm0 : MeasurableSet {t : ℝ | a * t ^ 2 + β * t + γ = 0} :=
    measurableSet_eq_fun hcont.measurable measurable_const
  have hunion : {t : ℝ | 0 < a * t ^ 2 + β * t + γ} ∪
      ({t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)} ∪ {t : ℝ | a * t ^ 2 + β * t + γ = 0})
      = univ := by
    ext t; simp only [mem_union, mem_ofPred_eq, mem_univ, iff_true]
    rcases lt_trichotomy 0 (a * t ^ 2 + β * t + γ) with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inr h.symm)
    · exact Or.inr (Or.inl (by linarith))
  have hd1 : Disjoint {t : ℝ | 0 < a * t ^ 2 + β * t + γ}
      ({t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)} ∪ {t : ℝ | a * t ^ 2 + β * t + γ = 0}) := by
    rw [Set.disjoint_left]
    intro t h1 h2
    simp only [mem_ofPred_eq, mem_union] at h1 h2
    rcases h2 with h2 | h2 <;> linarith
  have hd2 : Disjoint {t : ℝ | 0 < (-a) * t ^ 2 + (-β) * t + (-γ)}
      {t : ℝ | a * t ^ 2 + β * t + γ = 0} := by
    rw [Set.disjoint_left]
    intro t h1 h2
    simp only [mem_ofPred_eq] at h1 h2
    linarith
  have h1 := measureReal_union (μ := γm) hd1 (hm2.union hm0)
  have h2 := measureReal_union (μ := γm) hd2 hm0
  rw [hunion, probReal_univ] at h1
  have h0 : γm.real {t : ℝ | a * t ^ 2 + β * t + γ = 0} = 0 := by
    rw [measureReal_def, quad_zero_null ha, ENNReal.toReal_zero]
  linarith

theorem quad_mass_diff {a β γ β' γ' : ℝ} (ha : a ≠ 0) :
    |(gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β * t + γ} -
        (gaussianReal 0 1).real {t : ℝ | 0 < a * t ^ 2 + β' * t + γ'}| ≤
      (|β - β'| + √|(β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ')|) / (√(2 * π) * |a|) := by
  rcases lt_or_gt_of_ne ha with hneg | hpos
  · rw [abs_le]
    constructor
    · have := quad_mass_diff_of_neg (β := β') (γ := γ') (β' := β) (γ' := γ) hneg
      rw [abs_sub_comm β' β, abs_sub_comm (β' ^ 2 - 4 * a * γ')] at this
      linarith
    · exact quad_mass_diff_of_neg hneg
  · rw [quad_mass_neg ha, quad_mass_neg ha]
    have hn : -a < 0 := by linarith
    have key1 := quad_mass_diff_of_neg (β := -β') (γ := -γ') (β' := -β) (γ' := -γ) hn
    have key2 := quad_mass_diff_of_neg (β := -β) (γ := -γ) (β' := -β') (γ' := -γ') hn
    have e1 : |-β' - -β| = |β - β'| := by rw [show -β' - -β = β - β' by ring]
    have e2 : |-β - -β'| = |β - β'| := by rw [show -β - -β' = -(β - β') by ring, abs_neg]
    have e3 : |((-β') ^ 2 - 4 * -a * -γ') - ((-β) ^ 2 - 4 * -a * -γ)| =
        |(β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ')| := by
      rw [show ((-β') ^ 2 - 4 * -a * -γ') - ((-β) ^ 2 - 4 * -a * -γ) =
        -((β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ')) by ring, abs_neg]
    have e4 : |((-β) ^ 2 - 4 * -a * -γ) - ((-β') ^ 2 - 4 * -a * -γ')| =
        |(β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ')| := by
      rw [show ((-β) ^ 2 - 4 * -a * -γ) - ((-β') ^ 2 - 4 * -a * -γ') =
        (β ^ 2 - 4 * a * γ) - (β' ^ 2 - 4 * a * γ') by ring]
    rw [e1, e3, abs_neg] at key1
    rw [e2, e4, abs_neg] at key2
    rw [abs_le]
    constructor <;> linarith

end Cubature.Aux

namespace Cubature.Aux

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma quadric_line (A : E →L[ℝ] E) (b : E) (c : ℝ) (u x : E) (t : ℝ) :
    ⟪x + t • u, A (x + t • u)⟫_ℝ + ⟪b, x + t • u⟫_ℝ + c =
      ⟪u, A u⟫_ℝ * t ^ 2 + (⟪x, A u⟫_ℝ + ⟪u, A x⟫_ℝ + ⟪b, u⟫_ℝ) * t +
        (⟪x, A x⟫_ℝ + ⟪b, x⟫_ℝ + c) := by
  simp only [map_add, map_smul, inner_add_left, inner_add_right, real_inner_smul_left,
    real_inner_smul_right]
  ring

theorem quadric_preint (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu : ⟪u, A u⟫_ℝ ≠ 0)
    (β D : E → ℝ) (hβ : ∀ z, β z = ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ)
    (hD : ∀ z, D z = β z ^ 2 - 4 * ⟪u, A u⟫_ℝ * φ z) (x x' : E) :
    |preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x -
        preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x'| ≤
      (|β x - β x'| + √|D x - D x'|) / (√(2 * π) * |⟪u, A u⟫_ℝ|) := by
  have hP : ∀ z, preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) z
      = (gaussianReal 0 1).real {t : ℝ | 0 < ⟪u, A u⟫_ℝ * t ^ 2 + β z * t + φ z} := by
    intro z
    have hset : {t : ℝ | 0 < φ (z + t • u)} =
        {t : ℝ | 0 < ⟪u, A u⟫_ℝ * t ^ 2 + β z * t + φ z} := by
      ext t
      simp only [mem_ofPred_eq]
      rw [hφ (z + t • u), quadric_line, hβ z, hφ z]
    have hm : MeasurableSet {t : ℝ | 0 < ⟪u, A u⟫_ℝ * t ^ 2 + β z * t + φ z} :=
      measurableSet_lt measurable_const (by fun_prop)
    rw [preintJump, hset, integral_indicator hm, setIntegral_const, smul_eq_mul, mul_one]
  rw [hP x, hP x', hD x, hD x']
  exact quad_mass_diff hu

end Cubature.Aux

namespace Cubature.Aux

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem quadric_preint_local (A : E →L[ℝ] E) (b : E) (c : ℝ) (φ : E → ℝ)
    (hφ : ∀ z, φ z = ⟪z, A z⟫_ℝ + ⟪b, z⟫_ℝ + c) (u : E) (hu1 : ‖u‖ = 1)
    (hu : ⟪u, A u⟫_ℝ ≠ 0) {R : ℝ} (x x' : E) (hx : ‖x‖ ≤ R) (hx' : ‖x'‖ ≤ R) :
    |preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x -
        preintJump (gaussianReal 0 1) (fun t z => φ (z + t • u)) (fun _ _ => (1 : ℝ)) x'| ≤
      (2 * ‖A‖ * ‖x - x'‖ + √(8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * ‖x - x'‖)) /
        (√(2 * π) * |⟪u, A u⟫_ℝ|) := by
  set β : E → ℝ := fun z => ⟪z, A u⟫_ℝ + ⟪u, A z⟫_ℝ + ⟪b, u⟫_ℝ with hβ
  set a := ⟪u, A u⟫_ℝ with ha
  set D : E → ℝ := fun z => β z ^ 2 - 4 * a * φ z with hD
  have key := quadric_preint A b c φ hφ u hu β D (fun z => rfl) (fun z => rfl) x x'
  refine key.trans ?_
  set δ := ‖x - x'‖
  have hA : 0 ≤ ‖A‖ := norm_nonneg _
  have hδ : 0 ≤ δ := norm_nonneg _
  have hR : 0 ≤ R := (norm_nonneg _).trans hx
  have hAu : ‖A u‖ ≤ ‖A‖ := by
    have := A.le_opNorm u; rwa [hu1, mul_one] at this
  -- bound on the β-difference
  have hβd : |β x - β x'| ≤ 2 * ‖A‖ * δ := by
    have e : β x - β x' = ⟪x - x', A u⟫_ℝ + ⟪u, A (x - x')⟫_ℝ := by
      simp only [hβ, map_sub, inner_sub_left, inner_sub_right]; ring
    rw [e]
    have h1 : |⟪x - x', A u⟫_ℝ| ≤ δ * ‖A‖ :=
      (abs_real_inner_le_norm _ _).trans (by gcongr)
    have h2 : |⟪u, A (x - x')⟫_ℝ| ≤ ‖A‖ * δ := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hu1, one_mul]; exact A.le_opNorm _
    calc |⟪x - x', A u⟫_ℝ + ⟪u, A (x - x')⟫_ℝ| ≤ |⟪x - x', A u⟫_ℝ| + |⟪u, A (x - x')⟫_ℝ| :=
          abs_add_le _ _
      _ ≤ δ * ‖A‖ + ‖A‖ * δ := add_le_add h1 h2
      _ = 2 * ‖A‖ * δ := by ring
  -- bound on β itself
  have hβb : ∀ z : E, ‖z‖ ≤ R → |β z| ≤ 2 * ‖A‖ * R + ‖b‖ := by
    intro z hz
    have h1 : |⟪z, A u⟫_ℝ| ≤ R * ‖A‖ := (abs_real_inner_le_norm _ _).trans (by gcongr)
    have h2 : |⟪u, A z⟫_ℝ| ≤ ‖A‖ * R := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hu1, one_mul]; exact (A.le_opNorm _).trans (by gcongr)
    have h3 : |⟪b, u⟫_ℝ| ≤ ‖b‖ := by
      refine (abs_real_inner_le_norm _ _).trans ?_; rw [hu1, mul_one]
    calc |β z| ≤ |⟪z, A u⟫_ℝ| + |⟪u, A z⟫_ℝ| + |⟪b, u⟫_ℝ| := by
          simp only [hβ]; exact (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ ≤ R * ‖A‖ + ‖A‖ * R + ‖b‖ := by gcongr
      _ = 2 * ‖A‖ * R + ‖b‖ := by ring
  -- bound on the φ-difference
  have hφd : |φ x - φ x'| ≤ (2 * ‖A‖ * R + ‖b‖) * δ := by
    have e : φ x - φ x' = ⟪x - x', A x⟫_ℝ + ⟪x', A (x - x')⟫_ℝ + ⟪b, x - x'⟫_ℝ := by
      rw [hφ x, hφ x']
      simp only [map_sub, inner_sub_left, inner_sub_right]; ring
    rw [e]
    have h1 : |⟪x - x', A x⟫_ℝ| ≤ δ * (‖A‖ * R) :=
      (abs_real_inner_le_norm _ _).trans (by gcongr; exact (A.le_opNorm _).trans (by gcongr))
    have h2 : |⟪x', A (x - x')⟫_ℝ| ≤ R * (‖A‖ * δ) :=
      (abs_real_inner_le_norm _ _).trans (by gcongr; exact A.le_opNorm _)
    have h3 : |⟪b, x - x'⟫_ℝ| ≤ ‖b‖ * δ := abs_real_inner_le_norm _ _
    calc |⟪x - x', A x⟫_ℝ + ⟪x', A (x - x')⟫_ℝ + ⟪b, x - x'⟫_ℝ|
        ≤ |⟪x - x', A x⟫_ℝ| + |⟪x', A (x - x')⟫_ℝ| + |⟪b, x - x'⟫_ℝ| :=
          (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ ≤ δ * (‖A‖ * R) + R * (‖A‖ * δ) + ‖b‖ * δ := by gcongr
      _ = (2 * ‖A‖ * R + ‖b‖) * δ := by ring
  have hab : |a| ≤ ‖A‖ := by
    refine (abs_real_inner_le_norm _ _).trans ?_; rw [hu1, one_mul]; exact hAu
  -- bound on the D-difference
  have hDd : |D x - D x'| ≤ 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * δ := by
    have e : D x - D x' = (β x - β x') * (β x + β x') - 4 * a * (φ x - φ x') := by
      simp only [hD]; ring
    rw [e]
    have hs : |β x + β x'| ≤ 2 * (2 * ‖A‖ * R + ‖b‖) := by
      have := abs_add_le (β x) (β x'); linarith [hβb x hx, hβb x' hx']
    have hK : 0 ≤ 2 * ‖A‖ * R + ‖b‖ := by positivity
    calc |(β x - β x') * (β x + β x') - 4 * a * (φ x - φ x')|
        ≤ |β x - β x'| * |β x + β x'| + 4 * |a| * |φ x - φ x'| := by
          refine (abs_sub _ _).trans ?_
          rw [abs_mul, abs_mul, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 4)]
      _ ≤ (2 * ‖A‖ * δ) * (2 * (2 * ‖A‖ * R + ‖b‖)) +
            4 * ‖A‖ * ((2 * ‖A‖ * R + ‖b‖) * δ) := by gcongr
      _ = 8 * ‖A‖ * (2 * ‖A‖ * R + ‖b‖) * δ := by ring
  have hapos : 0 < √(2 * π) * |a| := by
    have : 0 < |a| := abs_pos.mpr hu
    positivity
  gcongr

end Cubature.Aux
