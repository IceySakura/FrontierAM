import Theory.GaussianDilation
import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-! Gaussian two-coordinate preintegration with positive quadratic curvature. -/

open MeasureTheory Set ProbabilityTheory
open scoped BigOperators ENNReal NNReal

namespace Cubature

noncomputable section

def diskEnergy (t : ℝ) : Set (Vec 2) := {x | ∑ i, (x i)^2 ≤ t}

def twoGaussian (m : Vec 2) (v : Fin 2 → ℝ≥0) : Measure (Vec 2) :=
  Measure.pi fun i => gaussianReal (m i) (v i)

instance (m : Vec 2) (v : Fin 2 → ℝ≥0) : IsProbabilityMeasure (twoGaussian m v) := by
  unfold twoGaussian
  infer_instance

def diskProbability (m : Vec 2) (v : Fin 2 → ℝ≥0) (t : ℝ) : ℝ :=
  (twoGaussian m v).real (diskEnergy t)

def ellipseProbability (a lam : Vec 2) (t : ℝ) : ℝ :=
  (gaussian 2).real {x | ∑ i, lam i * (x i - a i)^2 ≤ t}

lemma measurable_diskEnergy (t : ℝ) : MeasurableSet (diskEnergy t) := by
  unfold diskEnergy
  exact measurableSet_le (by fun_prop) measurable_const

lemma volume_diskEnergy (t : ℝ) :
    volume (diskEnergy t) = ENNReal.ofReal (Real.pi * max t 0) := by
  by_cases hs : 0 ≤ t
  · have hpre : (WithLp.toLp 2 : Vec 2 → EuclideanSpace ℝ (Fin 2)) ⁻¹'
        Metric.closedBall 0 (Real.sqrt t) = diskEnergy t := by
      ext x
      change dist (WithLp.toLp 2 x : EuclideanSpace ℝ (Fin 2)) 0 ≤ Real.sqrt t ↔ ∑ i, (x i)^2 ≤ t
      rw [dist_zero_right, ← sq_le_sq₀ (norm_nonneg _) (Real.sqrt_nonneg _),
        EuclideanSpace.real_norm_sq_eq, Real.sq_sqrt hs]
    rw [← hpre, (PiLp.volume_preserving_toLp (Fin 2)).measure_preimage
      (Metric.isClosed_closedBall.measurableSet.nullMeasurableSet),
      EuclideanSpace.volume_closedBall_fin_two, max_eq_left hs,
      ← ENNReal.ofReal_pow (Real.sqrt_nonneg _), Real.sq_sqrt hs,
      ← ENNReal.ofReal_mul hs, mul_comm]
  · have he : diskEnergy t = ∅ := by
      apply eq_empty_iff_forall_notMem.mpr
      intro z hz
      have h0 : 0 ≤ ∑ i, (z i)^2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
      exact hs (h0.trans hz)
    simp [he, max_eq_right (le_of_not_ge hs)]

lemma volume_real_diskEnergy (t : ℝ) :
    (volume : Measure (Vec 2)).real (diskEnergy t) = Real.pi*max t 0 := by
  rw [measureReal_def, volume_diskEnergy, ENNReal.toReal_ofReal (by positivity)]

private lemma gaussian_density_upper (m : ℝ) (v : ℝ≥0) (hv : v ≠ 0) :
    gaussianReal m v ≤ ENNReal.ofReal (Real.sqrt (2*Real.pi*(v : ℝ)))⁻¹ • volume := by
  let c := ENNReal.ofReal (Real.sqrt (2*Real.pi*(v : ℝ)))⁻¹
  rw [gaussianReal_of_var_ne_zero _ hv]
  change volume.withDensity (gaussianPDF m v) ≤ c • volume
  calc
    _ ≤ (volume : Measure ℝ).withDensity (c • (1 : ℝ → ℝ≥0∞)) := by
      apply withDensity_mono
      apply ae_of_all
      intro x
      change ENNReal.ofReal (gaussianPDFReal m v x) ≤ c * 1
      rw [mul_one]
      unfold gaussianPDFReal c
      apply ENNReal.ofReal_le_ofReal
      apply mul_le_of_le_one_right (by positivity)
      exact Real.exp_le_one_iff.mpr
        (div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (sq_nonneg _)) (by positivity))
    _ = _ := by rw [withDensity_smul' c 1 ENNReal.ofReal_ne_top, withDensity_one]

lemma twoGaussian_le_volume (m : Vec 2) (v : Fin 2 → ℝ≥0) (hv : ∀ i, v i ≠ 0) :
    twoGaussian m v ≤ ENNReal.ofReal
      ((2*Real.pi * Real.sqrt (v 0 : ℝ) * Real.sqrt (v 1 : ℝ))⁻¹) • volume := by
  let c (i : Fin 2) := (Real.sqrt (2*Real.pi*(v i : ℝ)))⁻¹
  have hc (i : Fin 2) : 0 ≤ c i := by unfold c; positivity
  have h := pi_mono_coord (fun i => gaussian_density_upper (m i) (v i) (hv i))
  change twoGaussian m v ≤ Measure.pi (fun i : Fin 2 => ENNReal.ofReal (c i) • volume) at h
  rw [pi_smul_coord _ (fun i => ENNReal.ofReal (c i)) (fun _ => ENNReal.ofReal_ne_top),
      ← ENNReal.ofReal_prod_of_nonneg (fun i _ => hc i), Fin.prod_univ_two] at h
  have hprod : c 0 * c 1 = (2*Real.pi * Real.sqrt (v 0 : ℝ) * Real.sqrt (v 1 : ℝ))⁻¹ := by
    unfold c
    rw [Real.sqrt_mul (by positivity : 0 ≤ 2*Real.pi), Real.sqrt_mul (by positivity : 0 ≤ 2*Real.pi),
      ← mul_inv]
    congr 1
    rw [show Real.sqrt (2*Real.pi) * Real.sqrt (v 0 : ℝ) *
        (Real.sqrt (2*Real.pi) * Real.sqrt (v 1 : ℝ)) =
      Real.sqrt (2*Real.pi)^2 * Real.sqrt (v 0 : ℝ) * Real.sqrt (v 1 : ℝ) by ring,
      Real.sq_sqrt (by positivity)]
  rw [hprod] at h
  exact h

lemma diskProbability_rise (m : Vec 2) (v : Fin 2 → ℝ≥0) (hv : ∀ i, v i ≠ 0)
    {s t : ℝ} (hst : s ≤ t) :
    0 ≤ diskProbability m v t - diskProbability m v s ∧
    diskProbability m v t - diskProbability m v s ≤
      (t - s)/(2*Real.sqrt (v 0 : ℝ)*Real.sqrt (v 1 : ℝ)) := by
  have hsub : diskEnergy s ⊆ diskEnergy t := by
    intro x hx
    exact (show ∑ i, (x i)^2 ≤ s from hx).trans hst
  have hs1 : 0 < (v 0 : ℝ) := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr (hv 0))
  have hs2 : 0 < (v 1 : ℝ) := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr (hv 1))
  refine ⟨sub_nonneg.mpr (measureReal_mono hsub), ?_⟩
  let C : ℝ := (2*Real.pi * Real.sqrt (v 0 : ℝ) * Real.sqrt (v 1 : ℝ))⁻¹
  have hC : 0 ≤ C := by unfold C; positivity
  have hvT : (volume : Measure (Vec 2)) (diskEnergy t) ≠ ∞ := by
    rw [volume_diskEnergy]
    exact ENNReal.ofReal_ne_top
  have hvo : (volume : Measure (Vec 2)) (diskEnergy t \ diskEnergy s) ≠ ∞ :=
    ne_top_of_le_ne_top hvT (measure_mono diff_subset)
  have h := (twoGaussian_le_volume m v hv) (diskEnergy t \ diskEnergy s)
  rw [Measure.smul_apply, smul_eq_mul] at h
  have h2 := ENNReal.toReal_mono (ENNReal.mul_ne_top ENNReal.ofReal_ne_top hvo) h
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal hC] at h2
  change (twoGaussian m v).real (diskEnergy t \ diskEnergy s) ≤
    C*(volume : Measure (Vec 2)).real (diskEnergy t \ diskEnergy s) at h2
  rw [measureReal_sdiff hsub (measurable_diskEnergy s) (measure_ne_top _ _),
    measureReal_sdiff hsub (measurable_diskEnergy s) hvT,
    volume_real_diskEnergy, volume_real_diskEnergy] at h2
  calc
    _ ≤ C*(Real.pi*max t 0-Real.pi*max s 0) := h2
    _ = (max t 0-max s 0)/(2*Real.sqrt (v 0 : ℝ)*Real.sqrt (v 1 : ℝ)) := by
      unfold C
      field_simp
    _ ≤ _ := by
      apply div_le_div_of_nonneg_right _ (by positivity)
      rcases le_total 0 s with hs | hs
      · rw [max_eq_left hs, max_eq_left (hs.trans hst)]
      · rw [max_eq_right hs]
        rcases le_total 0 t with ht | ht
        · rw [max_eq_left ht]
          linarith only [hs]
        · rw [max_eq_right ht]
          linarith only [hst]

lemma ellipse_eq_disk (a lam : Vec 2) (hlam : ∀ i, 0 < lam i) (t : ℝ) :
    ellipseProbability a lam t =
      diskProbability (fun i => -Real.sqrt (lam i) * a i) (fun i => ⟨lam i, (hlam i).le⟩) t := by
  let T (i : Fin 2) (x : ℝ) := Real.sqrt (lam i) * (x - a i)
  have tm (i : Fin 2) : Measurable (T i) := by unfold T; fun_prop
  have hm (i : Fin 2) : (gaussianReal 0 1).map (T i) =
      gaussianReal (-Real.sqrt (lam i) * a i) ⟨lam i, (hlam i).le⟩ := by
    have eq : T i = (fun x : ℝ => Real.sqrt (lam i) * x) ∘ (fun x : ℝ => x-a i) := rfl
    rw [eq, ← Measure.map_map (by fun_prop) (by fun_prop), gaussianReal_map_sub_const,
      gaussianReal_map_const_mul]
    simp [Real.sq_sqrt (hlam i).le, mul_neg, neg_mul] <;> rfl
  unfold ellipseProbability diskProbability twoGaussian
  have hmap : (gaussian 2).map (fun x i => T i (x i)) =
      Measure.pi (fun i => gaussianReal (-Real.sqrt (lam i) * a i) ⟨lam i, (hlam i).le⟩) := by
    unfold gaussian
    rw [Measure.pi_map_pi (fun i => (tm i).aemeasurable)]
    simp only [hm]
  rw [← hmap]
  have hx : Measurable (fun (x : Vec 2) i => T i (x i)) :=
    Measurable.of_eval (fun i => (tm i).comp (measurable_pi_apply i))
  rw [map_measureReal_apply hx (measurable_diskEnergy t)]
  congr 1
  ext x
  change _ ↔ ∑ i, (Real.sqrt (lam i) * (x i-a i))^2 ≤ t
  simp_rw [mul_pow, Real.sq_sqrt (hlam _).le]
  rfl

lemma ellipse_rise (a lam : Vec 2) (hlam : ∀ i, 0 < lam i) {s t : ℝ} (hst : s ≤ t) :
    0 ≤ ellipseProbability a lam t - ellipseProbability a lam s ∧
    ellipseProbability a lam t - ellipseProbability a lam s ≤
      (t-s)/(2*Real.sqrt (lam 0)*Real.sqrt (lam 1)) := by
  simp only [ellipse_eq_disk a lam hlam]
  exact diskProbability_rise _ _ (fun i => by
    intro h
    exact (hlam i).ne' (congrArg NNReal.toReal h)) hst

lemma ellipse_lipschitz (a lam : Vec 2) (hlam : ∀ i, 0 < lam i) (s t : ℝ) :
    |ellipseProbability a lam s - ellipseProbability a lam t| ≤
      |s-t|/(2*Real.sqrt (lam 0)*Real.sqrt (lam 1)) := by
  rcases le_total t s with h | h
  · obtain ⟨h1,h2⟩ := ellipse_rise a lam hlam h
    rwa [abs_of_nonneg h1, abs_of_nonneg (sub_nonneg.mpr h)]
  · obtain ⟨h1,h2⟩ := ellipse_rise a lam hlam h
    rw [abs_sub_comm (ellipseProbability a lam s), abs_sub_comm s,
      abs_of_nonneg h1, abs_of_nonneg (sub_nonneg.mpr h)]
    exact h2

end

end Cubature
