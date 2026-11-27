import Theory.GaussianStar
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-! Geometry and Gaussian masses for the unknown-center ball problem. -/

open MeasureTheory ProbabilityTheory Set Metric WithLp
open scoped ENNReal

namespace Cubature

noncomputable section

def ballOracle {d : ℕ} (c : GaussianSpace d) (r : ℝ) (z : GaussianSpace d) : Bool :=
  decide (dist z c < r)

def gaussianBallMass {d : ℕ} (c : GaussianSpace d) (r : ℝ) : ℝ :=
  (stdGaussian (GaussianSpace d)).real (ball c r)

def unitBallVolume (d : ℕ) : ℝ≥0∞ := volume (ball (0 : GaussianSpace d) 1)

theorem unitBallVolume_formula {d : ℕ} (hd : 0 < d) :
    unitBallVolume d = ENNReal.ofReal (Real.sqrt Real.pi ^ d / Real.Gamma ((d : ℝ)/2 + 1)) := by
  have : Nonempty (Fin d) := ⟨⟨0,hd⟩⟩
  simp [unitBallVolume, EuclideanSpace.volume_ball, Fintype.card_fin]

private theorem ballVolume {d : ℕ} (hd : 0 < d) (c : GaussianSpace d) {r : ℝ} (hr : 0 ≤ r) :
    volume (ball c r) = ENNReal.ofReal (r^d) * unitBallVolume d := by
  have : Nonempty (Fin d) := ⟨⟨0,hd⟩⟩
  rw [Measure.addHaar_ball volume c hr]
  simp [unitBallVolume, GaussianSpace]

private theorem realBallVolume {d : ℕ} (hd : 0 < d) (c : GaussianSpace d) {r : ℝ} (hr : 0 ≤ r) :
    (volume : Measure (GaussianSpace d)).real (ball c r) = r^d * (unitBallVolume d).toReal := by
  rw [measureReal_def, ballVolume hd c hr, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (pow_nonneg hr _)]

theorem measurableSet_coordinateBox (d : ℕ) (r : ℝ) : MeasurableSet (coordinateBox d r) := by
  have h : coordinateBox d r = ⋂ i : Fin d, {z : GaussianSpace d | |z i| ≤ r} := by
    ext z
    simp [coordinateBox]
  rw [h]
  apply MeasurableSet.iInter
  intro i
  have hE : Measurable (fun z : GaussianSpace d => z i) := by fun_prop
  exact measurableSet_le (continuous_abs.measurable.comp hE) measurable_const

theorem volume_coordinateBox (d : ℕ) : volume (coordinateBox d 1) = (2:ℝ≥0∞)^d := by
  rw [← (PiLp.volume_preserving_toLp (Fin d)).measure_preimage
    (measurableSet_coordinateBox d 1).nullMeasurableSet]
  have he : (toLp 2) ⁻¹' coordinateBox d 1 =
      univ.pi (fun _ : Fin d => Icc (-1:ℝ) 1) := by
    ext z
    rw [mem_preimage,mem_univ_pi]
    change (∀ i, |z i| ≤ 1) ↔ _
    exact forall_congr' fun _ => abs_le
  rw [he, volume_pi, Measure.pi_pi]
  norm_num [Real.volume_Icc, Fintype.card_fin]

/-- Greedy packing of unit-separated centers in the cube. -/
theorem cube_packing {d : ℕ} (hd : 0 < d) (m : ℕ)
    (hm : (m:ℝ≥0∞) * unitBallVolume d < (2:ℝ≥0∞)^d) :
    ∃ c : Fin (m+1) → GaussianSpace d, (∀ i, c i ∈ coordinateBox d 1) ∧
      ∀ i j, i ≠ j → 1 ≤ dist (c i) (c j) := by
  induction m with
  | zero =>
    refine ⟨fun _ => 0, fun _ i => by simp, fun i j h => ?_⟩
    have hi := i.isLt
    have hj := j.isLt
    exact (h (Fin.ext (by omega))).elim
  | succ m ih =>
    have hc : (m:ℝ≥0∞) * unitBallVolume d < (2:ℝ≥0∞)^d := by
      have hcast : (m:ℝ≥0∞) ≤ ((m+1:ℕ):ℝ≥0∞) := Nat.cast_le.mpr (Nat.le_succ _)
      exact (mul_le_mul_of_nonneg_right hcast (by simp)).trans_lt hm
    obtain ⟨c,hc,hdist⟩ := ih hc
    let U := ⋃ i : Fin (m+1), ball (c i) 1
    have hU : (volume : Measure (GaussianSpace d)) U ≤ ((m+1:ℕ):ℝ≥0∞) * unitBallVolume d := by
      refine (measure_iUnion_le (fun i => ball (c i) 1)).trans_eq ?_
      simp only [ballVolume hd _ zero_le_one, one_pow, ENNReal.ofReal_one, one_mul,
        tsum_fintype, Finset.sum_const, Finset.card_univ, Fintype.card_fin,nsmul_eq_mul]
    have hnot : ¬coordinateBox d 1 ⊆ U := by
      intro hh
      have he := (measure_mono (μ := (volume : Measure (GaussianSpace d))) hh).trans hU
      rw [volume_coordinateBox] at he
      exact (not_lt_of_ge he) hm
    obtain ⟨z,hz,hno⟩ := Set.not_subset.mp hnot
    have hsep (i : Fin (m+1)) : 1 ≤ dist z (c i) := by
      by_contra! hh
      exact hno (mem_iUnion.mpr ⟨i,hh⟩)
    refine ⟨Fin.cons z c, fun i => ?_, ?_⟩
    · refine Fin.cases ?_ (fun i => ?_) i
      · simpa using hz
      · simpa using hc i
    · intro i j
      refine Fin.cases ?_ (fun i => ?_) i
      · refine Fin.cases ?_ (fun j _ => ?_) j
        · simp
        · simpa using hsep j
      · refine Fin.cases (fun _ => ?_) (fun j hij => ?_) j
        · simpa [dist_comm] using hsep i
        · simpa using hdist i j (fun he => hij (congrArg Fin.succ he))

private theorem norm_cube_le {d : ℕ} (hd : 0 < d) {c : GaussianSpace d}
    (hc : c ∈ coordinateBox d 1) : ‖c‖ ≤ (d : ℝ) := by
  have hnorm : ‖c‖^2 ≤ (d:ℝ) := by
    rw [EuclideanSpace.real_norm_sq_eq]
    calc
      _ ≤ ∑ _ : Fin d, (1:ℝ) := Finset.sum_le_sum fun i _ => by
        have hi : |c i| ≤ 1 := hc i
        simpa only [sq_abs,one_pow] using pow_le_pow_left₀ (abs_nonneg _) hi 2
      _ = _ := by simp [Fintype.card_fin]
  have h₁ : (1:ℝ) ≤ d := by exact_mod_cast Nat.succ_le_of_lt hd
  have h₂ : (d:ℝ) ≤ (d:ℝ)^2 := by
    rw [pow_two]
    exact le_mul_of_one_le_right (Nat.cast_nonneg _) h₁
  exact (sq_le_sq₀ (norm_nonneg c) (Nat.cast_nonneg _)).mp (hnorm.trans h₂)

private theorem norm_in_ball {d : ℕ} {x c : GaussianSpace d} {r : ℝ} (hx : x ∈ ball c r) :
    ‖c‖-r ≤ ‖x‖ ∧ ‖x‖ ≤ ‖c‖+r := by
  have hdist : ‖x-c‖ < r := by simpa only [dist_eq_norm] using (show dist x c < r from hx)
  have h := abs_le.mp ((abs_norm_sub_norm_le x c).trans hdist.le)
  exact ⟨by linarith [h.1], by linarith [h.2]⟩

theorem gaussianDensity_norm_integrable (d : ℕ) :
    Integrable (fun z : GaussianSpace d => gaussianDensity d ‖z‖) := by
  have hM : Measurable (fun z : GaussianSpace d => gaussianDensity d ‖z‖) :=
    ((continuous_gaussianDensity d).comp continuous_norm).measurable
  have hf : IsFiniteMeasure
      (volume.withDensity (fun z : GaussianSpace d => ENNReal.ofReal (gaussianDensity d ‖z‖))) := by
    rw [← stdGaussian_density]
    infer_instance
  simpa only [ENNReal.toReal_ofReal (gaussianDensity_nonneg d _)] using
    integrable_toReal_of_lintegral_ne_top hM.ennreal_ofReal.aemeasurable
      (isFiniteMeasure_withDensity_iff.mp hf)

private theorem ballMass_integral {d : ℕ} (c : GaussianSpace d) (r : ℝ) :
    gaussianBallMass c r = ∫ x in ball c r, gaussianDensity d ‖x‖ := by
  have hi : IntegrableOn (fun x : GaussianSpace d => gaussianDensity d ‖x‖) (ball c r) :=
    (gaussianDensity_norm_integrable d).integrableOn
  have hn : ∀ᵐ x ∂(volume : Measure (GaussianSpace d)).restrict (ball c r), 0 ≤ gaussianDensity d ‖x‖ :=
    Filter.Eventually.of_forall (fun _ => gaussianDensity_nonneg d _)
  rw [gaussianBallMass, measureReal_def, stdGaussian_density, withDensity_apply _ measurableSet_ball,
    ← ofReal_integral_eq_lintegral_ofReal hi hn, ENNReal.toReal_ofReal (integral_nonneg_of_ae hn)]

private theorem ballMass_lower {d : ℕ} (c : GaussianSpace d) {r : ℝ} :
    gaussianDensity d (‖c‖+r) * (volume : Measure (GaussianSpace d)).real (ball c r) ≤
      gaussianBallMass c r := by
  have hF : IsFiniteMeasure ((volume : Measure (GaussianSpace d)).restrict (ball c r)) :=
    ⟨by simpa only [Measure.restrict_apply_univ] using measure_ball_lt_top (μ := volume)⟩
  rw [ballMass_integral]
  have hh := integral_mono_ae (integrable_const (gaussianDensity d (‖c‖+r)))
    ((gaussianDensity_norm_integrable d).restrict (s := ball c r)) (by
      filter_upwards [ae_restrict_mem (μ := volume) measurableSet_ball] with x hx
      have ht := (norm_in_ball hx).2
      unfold gaussianDensity
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by
        have hsq := pow_le_pow_left₀ (norm_nonneg x) ht 2
        linarith)) (by unfold gaussianConstant; positivity))
  simpa [integral_const, Measure.restrict_apply_univ, measureReal_def, smul_eq_mul, mul_comm] using hh

theorem gaussianBallMass_pos {d : ℕ} (c : GaussianSpace d) {r : ℝ} (hr : 0 < r) :
    0 < gaussianBallMass c r := by
  have hv : 0 < (volume : Measure (GaussianSpace d)).real (ball c r) :=
    ENNReal.toReal_pos (measure_ball_pos _ c hr).ne' measure_ball_lt_top.ne
  exact (mul_pos (by unfold gaussianDensity gaussianConstant; positivity) hv).trans_le (ballMass_lower c)

private theorem ballMass_upper {d : ℕ} (c : GaussianSpace d) {r : ℝ} :
    gaussianBallMass c r ≤
      (gaussianConstant d * Real.exp (-(‖c‖^2-2*r*‖c‖)/2)) *
        (volume : Measure (GaussianSpace d)).real (ball c r) := by
  let V := gaussianConstant d * Real.exp (-(‖c‖^2-2*r*‖c‖)/2)
  have hF : IsFiniteMeasure ((volume : Measure (GaussianSpace d)).restrict (ball c r)) :=
    ⟨by simpa only [Measure.restrict_apply_univ] using measure_ball_lt_top (μ := volume)⟩
  rw [ballMass_integral]
  have hh := integral_mono_ae ((gaussianDensity_norm_integrable d).restrict (s := ball c r))
    (integrable_const V) (by
      filter_upwards [ae_restrict_mem (μ := volume) measurableSet_ball] with x hx
      have ht := mul_le_mul_of_nonneg_right (norm_in_ball hx).1 (norm_nonneg c)
      change gaussianConstant d * Real.exp (-‖x‖^2 / 2) ≤ V
      unfold V
      exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by
        nlinarith [sq_nonneg (‖x‖-‖c‖)])) (by unfold gaussianConstant; positivity))
  simpa [V, integral_const, Measure.restrict_apply_univ, measureReal_def, smul_eq_mul, mul_comm] using hh

/-- The two radii give disjoint relative-error bands, even far from the origin. -/
theorem gaussianBallMass_ratio {d : ℕ} (hd : 0 < d) {c : GaussianSpace d} (hc : c ∈ coordinateBox d 1) :
    2*gaussianBallMass c (1/16) ≤ gaussianBallMass c (1/2) := by
  let a := (‖c‖^2-2*(1/16:ℝ)*‖c‖)/2
  let b := (‖c‖+(1/2:ℝ))^2/2
  let lo := gaussianConstant d * Real.exp (-b)
  let hi := gaussianConstant d * Real.exp (-a)
  have hlo : 0 ≤ lo := by unfold lo gaussianConstant; positivity
  have hd' : 1 ≤ (d:ℝ) := by exact_mod_cast Nat.succ_le_of_lt hd
  have hbar : b-a ≤ (d:ℝ) := by
    unfold b a
    nlinarith [norm_cube_le hd hc]
  have hExp : Real.exp (d:ℝ) ≤ (3:ℝ)^d := by
    rw [show Real.exp (d:ℝ) = (Real.exp 1)^d by simp [← Real.exp_nat_mul]]
    exact pow_le_pow_left₀ (Real.exp_pos _).le Real.exp_one_lt_three.le _
  have hrel : hi ≤ (3:ℝ)^d * lo := by
    calc
      _ = lo * Real.exp (b-a) := by unfold hi lo; rw [mul_assoc,← Real.exp_add]; ring_nf
      _ ≤ lo * (3:ℝ)^d := mul_le_mul_of_nonneg_left ((Real.exp_le_exp.mpr hbar).trans hExp) hlo
      _ = _ := mul_comm _ _
  have hp : 2*(3:ℝ)^d ≤ (8:ℝ)^d := by
    have h₂ : (2:ℝ) ≤ (2:ℝ)^d := by
      simpa using (pow_le_pow_right₀ (by norm_num : (1:ℝ) ≤ 2) (Nat.succ_le_of_lt hd))
    calc
      _ ≤ (2:ℝ)^d*(3:ℝ)^d := mul_le_mul_of_nonneg_right h₂ (by positivity)
      _ = (6:ℝ)^d := by rw [← mul_pow]; norm_num
      _ ≤ _ := pow_le_pow_left₀ (by norm_num) (by norm_num) _
  have hv : (8:ℝ)^d * (volume : Measure (GaussianSpace d)).real (ball c (1/16)) =
      (volume : Measure (GaussianSpace d)).real (ball c (1/2)) := by
    rw [realBallVolume hd c (by norm_num : (0:ℝ) ≤ 1/16),
      realBallVolume hd c (by norm_num : (0:ℝ) ≤ 1/2), ← mul_assoc, ← mul_pow]
    norm_num
  calc
    _ ≤ 2 * (hi * (volume : Measure (GaussianSpace d)).real (ball c (1/16))) :=
      mul_le_mul_of_nonneg_left
        (by simpa only [hi,a,neg_div] using ballMass_upper c (r := 1/16)) (by norm_num)
    _ = (2*hi) * (volume : Measure (GaussianSpace d)).real (ball c (1/16)) := by ring
    _ ≤ ((8:ℝ)^d*lo) * (volume : Measure (GaussianSpace d)).real (ball c (1/16)) := by
      refine mul_le_mul_of_nonneg_right ?_ measureReal_nonneg
      calc
        _ ≤ 2*((3:ℝ)^d*lo) := mul_le_mul_of_nonneg_left hrel (by norm_num)
        _ = (2*(3:ℝ)^d)*lo := by ring
        _ ≤ _ := mul_le_mul_of_nonneg_right hp hlo
    _ = lo * (volume : Measure (GaussianSpace d)).real (ball c (1/2)) := by
      rw [← hv]
      ring
    _ ≤ _ := by simpa only [lo,b,gaussianDensity,neg_div] using ballMass_lower c (r := 1/2)

end

end Cubature
