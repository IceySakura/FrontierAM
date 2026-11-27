import Theory.HardFamily

/-! The additive mass formula and a uniform lower bound on every bit weight. -/

open MeasureTheory Set
open scoped ENNReal

namespace Cubature

noncomputable section

def radialLower (d : ℕ) : ℝ := (sphereArea d).toReal * gaussianDensity d 2

theorem radialLower_pos {d : ℕ} (hd : 0 < d) : 0 < radialLower d := by
  unfold radialLower gaussianDensity gaussianConstant
  exact mul_pos (ENNReal.toReal_pos (sphereArea_ne_zero hd) (sphereArea_ne_top _)) (by positivity)

private theorem radialLower_le (d : ℕ) {t : ℝ} (ht : t ∈ Icc 1 2) :
    radialLower d ≤ radialDensity d t := by
  have h₀ : 0 < t := zero_lt_one.trans_le ht.1
  rw [radialDensity, ite_eq_left h₀, radialLower]
  apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
  have hpos : 0 ≤ gaussianDensity d (2:ℝ) := gaussianDensity_nonneg _ _
  have hG : gaussianDensity d 2 ≤ gaussianDensity d t := by
    unfold gaussianDensity
    apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by unfold gaussianConstant; positivity)
    nlinarith [mul_nonneg (sub_nonneg.mpr ht.2) (show 0 ≤ 2+t by linarith)]
  have hT : (1:ℝ) ≤ t^(d-1) := by
    simpa only [one_pow] using (pow_le_pow_left₀ (show (0:ℝ) ≤ 1 by norm_num) ht.1 (d-1))
  exact (le_mul_of_one_le_left hpos hT).trans (mul_le_mul_of_nonneg_left hG (by positivity))

def massCDF (d : ℕ) : ℝ → ℝ := radialPrimitive (radialDensity d)

theorem massCDF_local_lower {d : ℕ} (hd : 0 < d) {a b : ℝ}
    (ha : 1 ≤ a) (hab : a ≤ b) (hb : b ≤ 2) :
    radialLower d * (b - a) ≤ massCDF d b - massCDF d a := by
  unfold massCDF
  rw [radialPrimitive_sub (integrable_radialDensity hd) hab, intervalIntegral.integral_of_le hab]
  have h := integral_mono_ae (integrable_const (radialLower d))
    ((integrable_radialDensity hd).restrict (s := Ioc a b)) (by
      filter_upwards [ae_restrict_mem (μ := volume) measurableSet_Ioc] with t ht
      exact radialLower_le d ⟨ha.trans ht.1.le, ht.2.trans hb⟩)
  rwa [integral_const, measureReal_def, Measure.restrict_apply_univ, Real.volume_Ioc,
    ENNReal.toReal_ofReal (sub_nonneg.mpr hab), smul_eq_mul, mul_comm] at h

theorem integrable_massCDF {X : Type*} [MeasurableSpace X] (ν : Measure X) [IsFiniteMeasure ν]
    {d : ℕ} (hd : 0 < d) {R : X → ℝ} (hR : Measurable R) :
    Integrable (fun u => massCDF d (R u)) ν := by
  let w := radialDensity d
  refine (integrable_const (∫ t, w t)).mono'
    (((measurable_radialPrimitive (integrable_radialDensity hd) (radialDensity_nonneg d)).comp
      hR).aestronglyMeasurable) ?_
  refine Filter.Eventually.of_forall fun x => ?_
  change ‖radialPrimitive w (R x)‖ ≤ _
  rw [Real.norm_of_nonneg (radialPrimitive_nonneg (radialDensity_nonneg d) _)]
  exact radialPrimitive_le (integrable_radialDensity hd) (radialDensity_nonneg d) _

def hardMassBase (p : ℕ) {m : ℕ} (j : Fin (m^2)) : ℝ := massCDF (p+1) (baseRadius j)

def bumpMassFn {p m : ℕ} (j : Fin (m^2)) (i : PatchIndex p m) (u : Direction (p+1)) : ℝ :=
  massCDF (p+1) (baseRadius j + bumpHeight m * patch i u) - hardMassBase p j

def hardMassCoeff {p m : ℕ} (j : Fin (m^2)) (i : PatchIndex p m) : ℝ :=
  ∫ u, bumpMassFn j i u ∂angularLaw (p+1)

private theorem integrable_bumpMassFn {p m : ℕ} (j : Fin (m^2)) (i : PatchIndex p m) :
    Integrable (bumpMassFn j i) (angularLaw (p+1)) := by
  have := angularLaw_isProbability (Nat.succ_pos p)
  exact (integrable_massCDF _ (Nat.succ_pos p)
    (measurable_const.add ((measurable_patch i).const_mul _))).sub (integrable_const _)

theorem hardMass_identity {p m : ℕ} (j : Fin (m^2)) (b : PatchIndex p m → Bool) :
    gaussianStarMass (hardRadius j b) =
      hardMassBase p j + ∑ i, hardMassCoeff j i * bitvalue (b i) := by
  classical
  have := angularLaw_isProbability (Nat.succ_pos p)
  have heq (u : Direction (p+1)) :
      massCDF (p+1) (hardRadius j b u) =
        hardMassBase p j + ∑ i, bumpMassFn j i u * bitvalue (b i) := by
    by_cases h : ∃ i : PatchIndex p m, patch i u ≠ 0
    · obtain ⟨i, hi⟩ := h
      rw [hardRadius, selectedBumps_single b hi, Finset.sum_eq_single_of_mem i (Finset.mem_univ i)]
      · cases hb : b i <;> simp [bumpMassFn, hardMassBase, bitvalue, hb]
      · intro i' _ hi'
        have hz : patch i' u = 0 := by by_contra hn; exact hi' (patch_support_unique hn hi)
        simp [bumpMassFn, hardMassBase, hz]
    · have hz : ∀ i : PatchIndex p m, patch i u = 0 := by simpa using h
      simp [hardRadius, selectedBumps_zero b hz, bumpMassFn, hardMassBase, hz]
  rw [gaussianStarMass_eq (measurable_hardRadius j b)]
  change ∫ u, massCDF (p+1) (hardRadius j b u) ∂angularLaw (p+1) = _
  simp_rw [heq]
  have hi (i : PatchIndex p m) :
    Integrable (fun u => bumpMassFn j i u * bitvalue (b i)) (angularLaw (p+1)) :=
    (integrable_bumpMassFn j i).mul_const _
  rw [integral_add (integrable_const _) (integrable_finsetSum _ (fun i _ => hi i)),
    integral_const, integral_finsetSum _ (fun i _ => hi i)]
  simp [hardMassCoeff, integral_mul_const]

/-- Constant in the minimal bit weight, independent of radial and angular resolution. -/
def patchMassConstant (p : ℕ) : ℝ :=
  radialLower (p+1) * (2/3 : ℝ)^p * lineDensityLower * (lineDensityLower / 3)^p

theorem patchMassConstant_pos (p : ℕ) : 0 < patchMassConstant p := by
  unfold patchMassConstant
  have := radialLower_pos (Nat.succ_pos p)
  have := lineDensityLower_pos
  positivity

theorem hardMassCoeff_lower {p m : ℕ} (hm : 0 < m) (j : Fin (m^2)) (i : PatchIndex p m) :
    patchMassConstant p / (m : ℝ)^(p+2) ≤ hardMassCoeff j i := by
  have := angularLaw_isProbability (Nat.succ_pos p)
  let c := radialLower (p+1) * bumpHeight m * (2/3 : ℝ)^p
  have hrad := radialLower_pos (Nat.succ_pos p)
  have hB := bumpHeight_pos hm
  have hx (u : Direction (p+1)) : radialLower (p+1) * (bumpHeight m * patch i u) ≤
      bumpMassFn j i u := by
    have he := massCDF_local_lower (Nat.succ_pos p) (baseRadius_ge j)
      (show baseRadius j ≤ baseRadius j + bumpHeight m * patch i u from
        le_add_of_nonneg_right (mul_nonneg hB.le (patch_mem i u).1))
      (show baseRadius j + bumpHeight m * patch i u ≤ 2 from
        (add_le_add (le_refl (baseRadius j)) (mul_le_of_le_one_right hB.le (patch_mem i u).2)).trans
          (baseRadius_add_le hm j))
    simpa only [bumpMassFn, hardMassBase, add_sub_cancel_left] using he
  have hcore (u : Direction (p+1)) :
      (patchCore i).indicator (fun _ => c) u ≤ bumpMassFn j i u := by
    by_cases hu : u ∈ patchCore i
    · rw [indicator_of_mem hu]
      exact (mul_le_mul_of_nonneg_left (patchCore_lower hm i hu) (mul_nonneg hrad.le hB.le)).trans
        (by simpa only [mul_assoc] using hx u)
    · rw [indicator_of_notMem hu]
      exact (mul_nonneg hrad.le (mul_nonneg hB.le (patch_mem i u).1)).trans (hx u)
  have hI := integral_mono ((integrable_const c).indicator (patchCore_measurable i))
    (integrable_bumpMassFn j i) hcore
  rw [integral_indicator_const _ (patchCore_measurable i), smul_eq_mul] at hI
  have hm' : (m : ℝ) ≠ 0 := by positivity
  calc
    _ = (lineDensityLower * (lineDensityLower / (3 * (m : ℝ)))^p) * c := by
      unfold patchMassConstant c bumpHeight
      simp only [pow_add, div_pow, mul_pow]
      have hpow : (m : ℝ)^p ≠ 0 := pow_ne_zero _ hm'
      field_simp
    _ ≤ _ := (mul_le_mul_of_nonneg_right (patchCore_mass hm i) (by positivity)).trans hI

end

end Cubature
