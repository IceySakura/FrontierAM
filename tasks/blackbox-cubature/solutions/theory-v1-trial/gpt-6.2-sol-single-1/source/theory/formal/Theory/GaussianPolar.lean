import Mathlib.MeasureTheory.Constructions.HaarToSphere
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Tactic

/-! The sphere law and radial density used by the Gaussian query strategies. -/

open MeasureTheory ProbabilityTheory Set Metric WithLp
open scoped ENNReal

namespace Cubature

noncomputable section

abbrev GaussianSpace (d : ℕ) := EuclideanSpace ℝ (Fin d)
abbrev Direction (d : ℕ) := sphere (0 : GaussianSpace d) 1

def gaussianConstant (d : ℕ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) ^ d

def gaussianDensity (d : ℕ) (r : ℝ) : ℝ :=
  gaussianConstant d * Real.exp (-r ^ 2 / 2)

theorem gaussianDensity_nonneg (d : ℕ) (r : ℝ) : 0 ≤ gaussianDensity d r := by
  unfold gaussianDensity gaussianConstant
  positivity

theorem continuous_gaussianDensity (d : ℕ) : Continuous (gaussianDensity d) := by
  unfold gaussianDensity
  fun_prop

private theorem gaussian_std_density_line (r : ℝ) :
    gaussianPDFReal 0 1 r = gaussianDensity 1 r := by
  simp [gaussianPDFReal, gaussianDensity, gaussianConstant]

private theorem gaussian_pi_density (d : ℕ) :
    Measure.pi (fun _ : Fin d => gaussianReal 0 1) =
      (volume : Measure (Fin d → ℝ)).withDensity
        (fun x => ENNReal.ofReal (∏ i, gaussianPDFReal 0 1 (x i))) := by
  classical
  apply Measure.pi_eq
  intro s hs
  have hsv : MeasurableSet (univ.pi s) := MeasurableSet.pi countable_univ (fun i _ => hs i)
  rw [withDensity_apply _ hsv, volume_pi, Measure.restrict_pi_pi]
  have hi : ∀ i, Integrable (gaussianPDFReal 0 1) (volume.restrict (s i)) :=
    fun _ => (integrable_gaussianPDFReal _ _).restrict
  rw [← ofReal_integral_eq_lintegral_ofReal (Integrable.fintype_prod hi)
    (Filter.Eventually.of_forall (fun x => Finset.prod_nonneg
      (fun i _ => gaussianPDFReal_nonneg _ _ (x i))))]
  rw [integral_fintype_prod_eq_prod, ENNReal.ofReal_prod_of_nonneg]
  · exact Finset.prod_congr rfl (fun i _ =>
      (gaussianReal_apply_eq_integral 0 (by norm_num) (s i)).symm)
  · exact fun i _ => integral_nonneg (fun x => gaussianPDFReal_nonneg _ _ x)

/-- Change the underlying distribution of a nonnegative density by a measure-preserving map. -/
private theorem map_weighted {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    {μ : Measure X} {ν : Measure Y} {e : X → Y}
    (he : MeasurePreserving e μ ν) {w : Y → ℝ≥0∞} (hw : Measurable w) :
    (μ.withDensity (w ∘ e)).map e = ν.withDensity w := by
  ext s hs
  rw [Measure.map_apply he.measurable hs, withDensity_apply _ (he.measurable hs),
    withDensity_apply _ hs, ← lintegral_indicator hs, ← he.map_eq,
    lintegral_map (hw.indicator hs) he.measurable, ← lintegral_indicator (he.measurable hs)]
  congr 1

set_option maxHeartbeats 1000000 in
theorem stdGaussian_density (d : ℕ) :
    stdGaussian (GaussianSpace d) = volume.withDensity
      (fun z => ENNReal.ofReal (gaussianDensity d ‖z‖)) := by
  rw [← map_pi_eq_stdGaussian, gaussian_pi_density]
  have hcoord (x : Fin d → ℝ) :
      ∏ i, gaussianPDFReal 0 1 (x i) = gaussianDensity d ‖toLp 2 x‖ := by
    simp only [gaussian_std_density_line, gaussianDensity, gaussianConstant, pow_one]
    rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin, ← Real.exp_sum,
      EuclideanSpace.real_norm_sq_eq]
    congr 2
    rw [← Finset.sum_div, Finset.sum_neg_distrib]
  simp_rw [hcoord]
  have hm : Measurable (fun z : GaussianSpace d => ENNReal.ofReal (gaussianDensity d ‖z‖)) :=
    ((continuous_gaussianDensity d).comp continuous_norm).measurable.ennreal_ofReal
  exact map_weighted (PiLp.volume_preserving_toLp (Fin d)) hm

def sphereArea (d : ℕ) : ℝ≥0∞ :=
  (volume : Measure (GaussianSpace d)).toSphere univ

/-- Normalized surface measure on the Euclidean unit sphere. -/
def angularLaw (d : ℕ) : Measure (Direction d) :=
  (sphereArea d)⁻¹ • (volume : Measure (GaussianSpace d)).toSphere

theorem sphereArea_ne_top (d : ℕ) : sphereArea d ≠ ∞ := measure_ne_top _ _

theorem sphereArea_ne_zero {d : ℕ} (hd : 0 < d) : sphereArea d ≠ 0 := by
  intro h
  have hz : (volume : Measure (GaussianSpace d)).toSphere = 0 :=
    Measure.measure_univ_eq_zero.mp h
  have hfin : Module.finrank ℝ (GaussianSpace d) = d := by simp [GaussianSpace]
  exact hd.ne' (by simpa [hfin] using
    (Measure.toSphere_eq_zero_iff_finrank (volume : Measure (GaussianSpace d))).mp hz)

theorem angularLaw_isProbability {d : ℕ} (hd : 0 < d) :
    IsProbabilityMeasure (angularLaw d) where
  measure_univ := by
    change (sphereArea d)⁻¹ * sphereArea d = 1
    exact ENNReal.inv_mul_cancel (sphereArea_ne_zero hd) (sphereArea_ne_top d)

/-- Lebesgue density of the Gaussian radius, extended by zero off the positive ray. -/
def radialDensity (d : ℕ) (r : ℝ) : ℝ :=
  if 0 < r then (sphereArea d).toReal * (r ^ (d - 1) * gaussianDensity d r) else 0

theorem radialDensity_nonneg (d : ℕ) (r : ℝ) : 0 ≤ radialDensity d r := by
  unfold radialDensity
  split
  · exact mul_nonneg ENNReal.toReal_nonneg (mul_nonneg (by positivity) (gaussianDensity_nonneg d r))
  · exact le_rfl

theorem measurable_radialDensity (d : ℕ) : Measurable (radialDensity d) := by
  unfold radialDensity
  exact Measurable.ite measurableSet_Ioi (by
    exact measurable_const.mul ((measurable_id.pow_const _).mul
      (continuous_gaussianDensity d).measurable)) measurable_const

def radialLaw (d : ℕ) : Measure ℝ :=
  volume.withDensity (fun r => ENNReal.ofReal (radialDensity d r))

instance (d : ℕ) : SigmaFinite (radialLaw d) := by
  unfold radialLaw
  infer_instance

private theorem polarHaar {d : ℕ} (hd : 0 < d) :
    MeasurePreserving (fun p : Direction d × Ioi (0 : ℝ) => p.2.1 • p.1.1)
      ((volume : Measure (GaussianSpace d)).toSphere.prod (Measure.volumeIoiPow (d-1)))
      (volume : Measure (GaussianSpace d)) := by
  have hd' : Nonempty (Fin d) := ⟨⟨0, hd⟩⟩
  have he : Module.finrank ℝ (GaussianSpace d) = d := by simp [GaussianSpace]
  have hm := (volume : Measure (GaussianSpace d)).measurePreserving_homeomorphUnitSphereProd
  rw [he] at hm
  have hv : MeasurePreserving (Subtype.val : ({0}ᶜ : Set (GaussianSpace d)) → GaussianSpace d)
      ((volume : Measure (GaussianSpace d)).comap Subtype.val) volume := by
    refine ⟨measurable_subtype_coe, ?_⟩
    rw [map_comap_subtype_coe (measurableSet_singleton _).compl, restrict_compl_singleton]
  let e := homeomorphUnitSphereProd (GaussianSpace d)
  have hc := hv.comp (hm.symm e.toMeasurableEquiv)
  have hce : (Subtype.val : ({0}ᶜ : Set (GaussianSpace d)) → _) ∘
      e.toMeasurableEquiv.symm = (fun p : Direction d × Ioi (0:ℝ) => p.2.val • p.1.val) := by
    funext p
    exact homeomorphUnitSphereProd_symm_apply_coe _ _
  rwa [hce] at hc

private theorem stdGaussian_polar_unnorm {d : ℕ} (hd : 0 < d) :
    (((volume : Measure (GaussianSpace d)).toSphere).prod
      (((Measure.volumeIoiPow (d - 1)).withDensity
        (fun r => ENNReal.ofReal (gaussianDensity d r.1))).map Subtype.val)).map
        (fun p : Direction d × ℝ => p.2 • p.1.1) = stdGaussian (GaussianSpace d) := by
  have hd' : Nonempty (Fin d) := ⟨⟨0, hd⟩⟩
  have hw : Measurable (fun z : GaussianSpace d => ENNReal.ofReal (gaussianDensity d ‖z‖)) :=
    ((continuous_gaussianDensity d).comp continuous_norm).measurable.ennreal_ofReal
  rw [stdGaussian_density, ← map_weighted (polarHaar hd) hw]
  have hweight :
      (fun z : GaussianSpace d => ENNReal.ofReal (gaussianDensity d ‖z‖)) ∘
        (fun p : Direction d × Ioi (0 : ℝ) => p.2.1 • p.1.1) =
      (fun (r : Ioi (0 : ℝ)) => ENNReal.ofReal (gaussianDensity d r.1)) ∘ Prod.snd := by
    funext p
    rw [Function.comp_apply, Function.comp_apply, norm_smul,
      Real.norm_of_nonneg p.2.2.le, mem_sphere_zero_iff_norm.mp p.1.2, mul_one]
  let ρ := (Measure.volumeIoiPow (d - 1)).withDensity
      (fun (r : Ioi (0 : ℝ)) => ENNReal.ofReal (gaussianDensity d r.1))
  have hρ : SFinite ρ := by dsimp only [ρ]; infer_instance
  rw [hweight]
  change _ = Measure.map _ ((((volume : Measure (GaussianSpace d)).toSphere).prod
    (Measure.volumeIoiPow (d-1))).withDensity
      (fun p : Direction d × Ioi (0 : ℝ) => ENNReal.ofReal (gaussianDensity d p.2.val)))
  have hwρ : Measurable (fun r : Ioi (0 : ℝ) => ENNReal.ofReal (gaussianDensity d r.val)) :=
    ((continuous_gaussianDensity d).measurable.comp measurable_subtype_coe).ennreal_ofReal
  rw [← prod_withDensity_right hwρ]
  change (((volume : Measure (GaussianSpace d)).toSphere).prod (ρ.map Subtype.val)).map
        (fun p : Direction d × ℝ => p.2 • p.1.1) =
        (((volume : Measure (GaussianSpace d)).toSphere).prod ρ).map
          (fun p : Direction d × Ioi (0 : ℝ) => p.2.1 • p.1.1)
  have hh : (fun (p : Direction d × Ioi (0 : ℝ)) => p.2.1 • p.1.1) =
    (fun (p : Direction d × ℝ) => p.2 • p.1.1) ∘ Prod.map id Subtype.val := rfl
  rw [hh, ← Measure.map_map (by fun_prop) (by fun_prop)]
  have hh' := Measure.map_prod_map ((volume : Measure (GaussianSpace d)).toSphere) ρ
    measurable_id measurable_subtype_coe
  rw [Measure.map_id] at hh'
  rw [hh']

private theorem radialLaw_eq_scaled (d : ℕ) :
    radialLaw d = sphereArea d • (((Measure.volumeIoiPow (d - 1)).withDensity
      (fun r => ENNReal.ofReal (gaussianDensity d r.1))).map Subtype.val) := by
  let w : ℝ → ℝ := fun r => r ^ (d-1) * gaussianDensity d r
  have hw : Measurable w := by
    exact (measurable_id.pow_const _).mul (continuous_gaussianDensity d).measurable
  have he : MeasurePreserving (Subtype.val : Ioi (0 : ℝ) → ℝ)
      ((volume : Measure ℝ).comap Subtype.val) (volume.restrict (Ioi 0)) :=
    measurePreserving_subtype_coe measurableSet_Ioi
  have hvol : (Measure.volumeIoiPow (d - 1)).withDensity
      (fun r => ENNReal.ofReal (gaussianDensity d r.1)) =
      ((volume : Measure ℝ).comap (Subtype.val : Ioi (0:ℝ) → ℝ)).withDensity
        ((fun r => ENNReal.ofReal (w r)) ∘ Subtype.val) := by
    unfold Measure.volumeIoiPow
    let w₁ : Ioi (0 : ℝ) → ℝ≥0∞ := fun r => ENNReal.ofReal (r.val ^ (d-1))
    let w₂ : Ioi (0 : ℝ) → ℝ≥0∞ := fun r => ENNReal.ofReal (gaussianDensity d r.val)
    have h₁ : Measurable w₁ := (measurable_subtype_coe.pow_const _).ennreal_ofReal
    have h₂ : Measurable w₂ :=
      ((continuous_gaussianDensity d).measurable.comp measurable_subtype_coe).ennreal_ofReal
    change (((volume : Measure ℝ).comap Subtype.val).withDensity w₁).withDensity w₂ = _
    calc
      _ = ((volume : Measure ℝ).comap Subtype.val).withDensity (w₁ * w₂) :=
        (withDensity_mul _ h₁ h₂).symm
      _ = _ := by
        congr 1
        funext r
        change ENNReal.ofReal (r.val^(d-1)) * ENNReal.ofReal (gaussianDensity d r.val) =
          ENNReal.ofReal (r.val^(d-1) * gaussianDensity d r.val)
        rw [ENNReal.ofReal_mul (pow_nonneg r.2.le _)]
  rw [hvol, map_weighted he hw.ennreal_ofReal, ← withDensity_indicator measurableSet_Ioi]
  unfold radialLaw
  rw [← withDensity_smul _ (hw.ennreal_ofReal.indicator measurableSet_Ioi)]
  congr 1
  funext r
  change ENNReal.ofReal (radialDensity d r) =
    sphereArea d * ((Ioi (0:ℝ)).indicator (fun x => ENNReal.ofReal (w x)) r)
  by_cases hr : 0 < r
  · simp only [radialDensity, hr, ↓reduceIte, indicator_of_mem (mem_Ioi.2 hr)]
    rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg, ENNReal.ofReal_toReal (sphereArea_ne_top d)]
  · have hnr : r ∉ Ioi (0 : ℝ) := hr
    simp [radialDensity, hr, hnr]

/-- Sampling an independent uniform direction and Gaussian radius recovers the Gaussian law. -/
theorem stdGaussian_polar {d : ℕ} (hd : 0 < d) :
    MeasurePreserving (fun p : Direction d × ℝ => p.2 • p.1.1)
      ((angularLaw d).prod (radialLaw d)) (stdGaussian (GaussianSpace d)) where
  measurable := by fun_prop
  map_eq := by
    rw [angularLaw, radialLaw_eq_scaled, Measure.prod_smul_left, Measure.prod_smul_right,
      smul_smul, ENNReal.inv_mul_cancel (sphereArea_ne_zero hd) (sphereArea_ne_top d),
      one_smul, stdGaussian_polar_unnorm hd]

theorem radialLaw_isProbability {d : ℕ} (hd : 0 < d) :
    IsProbabilityMeasure (radialLaw d) where
  measure_univ := by
    have := angularLaw_isProbability hd
    have he := (stdGaussian_polar hd).map_eq
    have h₁ := congrArg (fun μ : Measure (GaussianSpace d) => μ univ) he
    rw [Measure.map_apply (stdGaussian_polar hd).measurable .univ, preimage_univ] at h₁
    rw [← univ_prod_univ, Measure.prod_prod,
      (angularLaw_isProbability hd).measure_univ, one_mul] at h₁
    exact h₁.trans (measure_univ (μ := stdGaussian (GaussianSpace d)))

theorem integrable_radialDensity {d : ℕ} (hd : 0 < d) : Integrable (radialDensity d) := by
  have := radialLaw_isProbability hd
  have hi : (∫⁻ t, ENNReal.ofReal (radialDensity d t)) ≠ ∞ :=
    isFiniteMeasure_withDensity_iff.mp (inferInstanceAs (IsFiniteMeasure (radialLaw d)))
  simpa only [ENNReal.toReal_ofReal (radialDensity_nonneg d _)] using
    integrable_toReal_of_lintegral_ne_top
      (measurable_radialDensity d).ennreal_ofReal.aemeasurable hi

/-- A bound on the radial density on positive radii at most `b`. -/
def radialDensityBound (d : ℕ) (b : ℝ) : ℝ :=
  (sphereArea d).toReal * (b ^ (d - 1) * gaussianConstant d)

theorem radialDensityBound_nonneg (d : ℕ) {b : ℝ} (hb : 0 ≤ b) :
    0 ≤ radialDensityBound d b := by
  unfold radialDensityBound gaussianConstant
  positivity

theorem radialDensity_le (d : ℕ) {b r : ℝ} (hb : 0 ≤ b) (hr : r ≤ b) :
    radialDensity d r ≤ radialDensityBound d b := by
  by_cases hp : 0 < r
  · simp only [radialDensity, hp, ↓reduceIte, gaussianDensity, radialDensityBound]
    have hE : Real.exp (-r ^ 2 / 2) ≤ 1 := Real.exp_le_one_iff.mpr
      (by linarith [sq_nonneg r])
    have hC : 0 ≤ gaussianConstant d := by unfold gaussianConstant; positivity
    have hM := mul_le_of_le_one_right hC hE
    gcongr
  · simpa [radialDensity, hp] using radialDensityBound_nonneg d hb

end

end Cubature
