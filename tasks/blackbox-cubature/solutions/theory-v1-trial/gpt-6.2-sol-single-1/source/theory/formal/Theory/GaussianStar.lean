import Theory.SphereCharts
import Theory.Slab

/-! Cartesian membership queries and Gaussian masses of open radial bodies. -/

open MeasureTheory ProbabilityTheory Set Metric WithLp
open scoped ENNReal unitInterval

namespace Cubature

noncomputable section

def defaultDirection (p : ℕ) : Direction (p+1) :=
  faceDirection (0, true) (fun _ => 0)

private def normalize {p : ℕ} (z : GaussianSpace (p+1)) : GaussianSpace (p+1) :=
  if z = 0 then (defaultDirection p).val else ‖z‖⁻¹ • z

def directionOf {p : ℕ} (z : GaussianSpace (p+1)) : Direction (p+1) :=
  ⟨normalize z, by
    unfold normalize
    split_ifs with h
    · exact (defaultDirection p).2
    · have hp := norm_pos_iff.mpr h
      rw [mem_sphere_zero_iff_norm, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hp.le),
        inv_mul_cancel₀ hp.ne']⟩

theorem measurable_directionOf (p : ℕ) :
    Measurable (directionOf (p := p)) := by
  apply Measurable.subtype_mk
  unfold normalize
  exact Measurable.ite (measurableSet_singleton _) measurable_const (by fun_prop)

theorem directionOf_val_of_ne_zero {p : ℕ} {z : GaussianSpace (p+1)} (hz : z ≠ 0) :
    (directionOf z).val = ‖z‖⁻¹ • z := by
  change normalize z = _
  rw [normalize, ite_eq_right hz]

def rayPoint {d : ℕ} (q : Direction d × ℝ) : GaussianSpace d :=
  q.2 • q.1.val

theorem norm_rayPoint {d : ℕ} (u : Direction d) {t : ℝ} (ht : 0 ≤ t) :
    ‖rayPoint (u, t)‖ = t := by
  simp [rayPoint, norm_smul, Real.norm_of_nonneg ht, mem_sphere_zero_iff_norm.mp u.2]

theorem directionOf_rayPoint {p : ℕ} (u : Direction (p+1)) {t : ℝ} (ht : 0 < t) :
    directionOf (rayPoint (u, t)) = u := by
  have hz : rayPoint (u, t) ≠ 0 := by
    intro he
    have h := norm_rayPoint u ht.le
    rw [he, norm_zero] at h
    exact ht.ne' h.symm
  apply Subtype.ext
  change normalize (rayPoint (u, t)) = u.val
  rw [normalize, ite_eq_right hz, norm_rayPoint u ht.le]
  exact inv_smul_smul₀ ht.ne' _

def starSet {p : ℕ} (R : Direction (p+1) → ℝ) : Set (GaussianSpace (p+1)) :=
  {z | ‖z‖ < R (directionOf z)}

def starOracle {p : ℕ} (R : Direction (p+1) → ℝ) (z : GaussianSpace (p+1)) : Bool :=
  decide (‖z‖ < R (directionOf z))

theorem starOracle_rayPoint {p : ℕ} (R : Direction (p+1) → ℝ) (u : Direction (p+1))
    {t : ℝ} (ht : 0 < t) : starOracle R (rayPoint (u,t)) = threshold R (u,t) := by
  unfold starOracle threshold
  simp only [norm_rayPoint u ht.le, directionOf_rayPoint u ht]

def gaussianStarMass {p : ℕ} (R : Direction (p+1) → ℝ) : ℝ :=
  (stdGaussian (GaussianSpace (p+1))).real (starSet R)

theorem measurableSet_starSet {p : ℕ} {R : Direction (p+1) → ℝ} (hR : Measurable R) :
    MeasurableSet (starSet R) :=
  measurableSet_lt continuous_norm.measurable (hR.comp (measurable_directionOf p))

theorem gaussianStarMass_eq {p : ℕ} {R : Direction (p+1) → ℝ} (hR : Measurable R) :
    gaussianStarMass R =
      ∫ u, radialPrimitive (radialDensity (p+1)) (R u) ∂angularLaw (p+1) := by
  let d := p + 1
  have hp : 0 < d := Nat.succ_pos p
  have := angularLaw_isProbability hp
  have := radialLaw_isProbability hp
  let f : GaussianSpace d → ℝ := (starSet R).indicator (fun _ => 1)
  have hf : Measurable f := measurable_const.indicator (measurableSet_starSet hR)
  have hfint : Integrable f (stdGaussian (GaussianSpace d)) :=
    (integrable_const 1).indicator (measurableSet_starSet hR)
  have he := stdGaussian_polar hp
  have hInt : Integrable (f ∘ rayPoint) ((angularLaw d).prod (radialLaw d)) :=
    (MeasurePreserving.integrable_comp he hf.aestronglyMeasurable).2 hfint
  have hSlice (u : Direction d) :
      (∫ t : ℝ, f (rayPoint (u,t)) ∂radialLaw d) =
      radialPrimitive (radialDensity d) (R u) := by
    unfold radialPrimitive radialLaw
    rw [integral_withDensity_eq_integral_toReal_smul (measurable_radialDensity d).ennreal_ofReal
      (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)]
    rw [← integral_indicator measurableSet_Iio]
    apply integral_congr_ae
    apply Filter.Eventually.of_forall
    intro t
    dsimp only
    rw [ENNReal.toReal_ofReal (radialDensity_nonneg d t), smul_eq_mul]
    by_cases ht : 0 < t
    · have hb : rayPoint (u,t) ∈ starSet R ↔ t < R u := by
        change (‖rayPoint (u,t)‖ < R (directionOf (rayPoint (u,t)))) ↔ _
        rw [norm_rayPoint u ht.le, directionOf_rayPoint u ht]
      by_cases hr : t < R u <;> simp [f, indicator, hb, hr]
    · simp [radialDensity, ht, indicator]
  unfold gaussianStarMass
  rw [← integral_indicator_one (measurableSet_starSet hR)]
  change (∫ z, f z ∂stdGaussian (GaussianSpace d)) = _
  rw [← he.map_eq, integral_map he.measurable.aemeasurable hf.aestronglyMeasurable]
  change ∫ q, (f ∘ rayPoint) q ∂(angularLaw d).prod (radialLaw d) = _
  rw [integral_prod _ hInt]
  exact integral_congr_ae (Filter.Eventually.of_forall hSlice)

def coordinateBox (d : ℕ) (b : ℝ) : Set (GaussianSpace d) :=
  {z | ∀ i, |z i| ≤ b}

theorem rayPoint_mem_coordinateBox {d : ℕ} (u : Direction d) {t b : ℝ}
    (ht : 0 ≤ t) (hb : t ≤ b) : rayPoint (u,t) ∈ coordinateBox d b := by
  intro i
  rw [← norm_rayPoint u ht] at hb
  exact (Real.norm_eq_abs _ ▸ PiLp.norm_apply_le (rayPoint (u,t)) i).trans hb

end

end Cubature
