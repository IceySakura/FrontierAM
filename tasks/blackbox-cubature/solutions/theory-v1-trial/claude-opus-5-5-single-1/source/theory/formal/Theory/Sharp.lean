import Theory.Defs
import Theory.Preint

/-!
# Sharpness (auxiliary proofs for R4)

* the constant `4` in the sublevel-set estimate is attained by `t ↦ α/2 t² - η`;
* for the Gilbert–Kuo–Sloan Example 1 (`φ(t, y) = y - t²`, `θ = 1`) the preintegrated function
  satisfies `c₁ √y ≤ P(y) - P(0) ≤ c₂ √y`, so the exponent `1/2` of R2 cannot be improved.
-/

open MeasureTheory ProbabilityTheory Real Set

namespace Cubature.Aux

lemma strongConvexOn_quadratic {α η : ℝ} :
    StrongConvexOn univ α (fun t : ℝ => α / 2 * t ^ 2 - η) := by
  rw [strongConvexOn_iff_convex]
  have : (fun t : ℝ => α / 2 * t ^ 2 - η - α / 2 * ‖t‖ ^ 2) = fun _ => -η := by
    funext t; rw [Real.norm_eq_abs, sq_abs]; ring
  rw [this]
  exact convexOn_const _ convex_univ

lemma sublevel_quadratic_eq {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η) :
    {t : ℝ | |α / 2 * t ^ 2 - η| ≤ η} = Icc (-(2 * √(η / α))) (2 * √(η / α)) := by
  ext t
  simp only [mem_setOf_eq, mem_Icc]
  have hr : (2 * √(η / α)) ^ 2 = 4 * η / α := by
    rw [mul_pow, Real.sq_sqrt (div_nonneg hη hα.le)]; ring
  have hr0 : 0 ≤ 2 * √(η / α) := by positivity
  constructor
  · intro h
    have h1 := (abs_le.mp h).2
    have : t ^ 2 ≤ (2 * √(η / α)) ^ 2 := by
      rw [hr, le_div_iff₀ hα]; nlinarith
    exact abs_le.mp (abs_le_of_sq_le_sq' this hr0 |> fun h => abs_le.mpr h)
  · intro h
    have h2 : t ^ 2 ≤ (2 * √(η / α)) ^ 2 := by
      have := abs_le.mpr h
      have h3 : |t| ^ 2 ≤ (2 * √(η / α)) ^ 2 := pow_le_pow_left₀ (abs_nonneg t) this 2
      rwa [sq_abs] at h3
    rw [hr, le_div_iff₀ hα] at h2
    rw [abs_le]
    constructor <;> nlinarith [sq_nonneg t]

theorem sublevel_sharp {α η : ℝ} (hα : 0 < α) (hη : 0 ≤ η) :
    StrongConvexOn univ α (fun t : ℝ => α / 2 * t ^ 2 - η) ∧
      volume {t : ℝ | |α / 2 * t ^ 2 - η| ≤ η} = ENNReal.ofReal (4 * √(η / α)) := by
  refine ⟨strongConvexOn_quadratic, ?_⟩
  rw [sublevel_quadratic_eq hα hη, Real.volume_Icc]
  congr 1; ring

/-- Gilbert–Kuo–Sloan Example 1 (`φ(t, y) = y - t²`, `θ = 1`). -/
theorem gks_example_lower (y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    2 * rexp (-1 / 2) / √(2 * π) * √y ≤
      preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) y -
        preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) 0 := by
  have hset : ∀ z : ℝ, 0 ≤ z → {s : ℝ | 0 < z - s ^ 2} = Ioo (-√z) (√z) := by
    intro z hz; ext s
    simp only [mem_setOf_eq, mem_Ioo]
    constructor
    · intro h
      have : s ^ 2 < (√z) ^ 2 := by rw [Real.sq_sqrt hz]; linarith
      exact abs_lt.mp (abs_lt_of_sq_lt_sq' this (Real.sqrt_nonneg z) |> fun h => abs_lt.mpr h)
    · intro h
      have h1 : |s| < √z := abs_lt.mpr h
      have h2 : s ^ 2 < (√z) ^ 2 := by
        rw [← sq_abs]; exact pow_lt_pow_left₀ h1 (abs_nonneg s) two_ne_zero
      rw [Real.sq_sqrt hz] at h2; linarith
  have hP : ∀ z : ℝ, 0 ≤ z → preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2)
      (fun _ _ => (1 : ℝ)) z = (gaussianReal 0 1).real (Ioo (-√z) (√z)) := by
    intro z hz
    rw [preintJump]
    rw [hset z hz]
    rw [integral_indicator measurableSet_Ioo, setIntegral_const, smul_eq_mul, mul_one]
  rw [hP y hy0, hP 0 le_rfl, Real.sqrt_zero, neg_zero, Ioo_self, measureReal_empty, sub_zero]
  -- lower bound for the Gaussian mass of the interval
  have hlow : ∀ x ∈ Ioo (-√y) (√y), ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) ≤
      gaussianPDF 0 1 x := by
    intro x hx
    rw [gaussianPDF, gaussianPDFReal]
    refine ENNReal.ofReal_le_ofReal ?_
    simp only [NNReal.coe_one, mul_one, sub_zero]
    have hx2 : x ^ 2 ≤ 1 := by
      have h1 : |x| < √y := abs_lt.mpr hx
      have h2 : √y ≤ 1 := Real.sqrt_le_one.mpr hy1 |> fun h => by
        calc √y ≤ √1 := Real.sqrt_le_sqrt hy1
          _ = 1 := Real.sqrt_one
      have : |x| ≤ 1 := by linarith
      nlinarith [abs_nonneg x, sq_abs x]
    have : rexp (-1 / 2) ≤ rexp (-x ^ 2 / 2) := Real.exp_le_exp.mpr (by linarith)
    rw [div_eq_inv_mul]
    exact mul_le_mul_of_nonneg_left this (by positivity)
  have hmeas : ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) * volume (Ioo (-√y) (√y)) ≤
      gaussianReal 0 1 (Ioo (-√y) (√y)) := by
    rw [gaussianReal_apply 0 one_ne_zero, ← setLIntegral_const]
    exact setLIntegral_mono (by fun_prop) hlow
  rw [Real.volume_Ioo, ← ENNReal.ofReal_mul (by positivity)] at hmeas
  have hfin : gaussianReal 0 1 (Ioo (-√y) (√y)) ≠ ⊤ := measure_ne_top _ _
  have := (ENNReal.ofReal_le_iff_le_toReal hfin).mp hmeas
  calc 2 * rexp (-1 / 2) / √(2 * π) * √y = rexp (-1 / 2) / √(2 * π) * (√y - -√y) := by ring
    _ ≤ (gaussianReal 0 1).real (Ioo (-√y) (√y)) := this


lemma gks_example_upper (y : ℝ) (hy0 : 0 ≤ y) :
    preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) y -
        preintJump (gaussianReal 0 1) (fun t y => y - t ^ 2) (fun _ _ => (1 : ℝ)) 0 ≤
      4 / √(2 * π) * √(y / 2) := by
  have hconc : ∀ z : ℝ, StrongConcaveOn univ 2 (fun t : ℝ => z - t ^ 2) := by
    intro z
    rw [strongConcaveOn_iff_convex]
    have : (fun t : ℝ => z - t ^ 2 + 2 / 2 * ‖t‖ ^ 2) = fun _ => z := by
      funext t; rw [Real.norm_eq_abs, sq_abs]; ring
    rw [this]
    exact concaveOn_const _ convex_univ
  have h := jump_holder (Y := ℝ) (ρ := (√(2 * π))⁻¹) (by positivity) gaussianReal_le_volume
    (φ := fun t y => y - t ^ 2) (θ := fun _ _ => (1 : ℝ)) (α := 2) (L := 1) (Lθ := 0) (B := 1)
    (by norm_num) (fun z => Or.inr (hconc z))
    (fun t y y' => by rw [Real.dist_eq]; simp [show y - t ^ 2 - (y' - t ^ 2) = y - y' by ring])
    (fun _ => measurable_const) (fun _ _ => by simp) (fun _ _ _ => by simp) y 0
  rw [Real.dist_eq, sub_zero, abs_of_nonneg hy0] at h
  have := (abs_le.mp h).2
  calc _ ≤ 0 * y + 4 * (√(2 * π))⁻¹ * 1 * √(1 * y / 2) := this
    _ = 4 / √(2 * π) * √(y / 2) := by ring_nf

end Cubature.Aux
