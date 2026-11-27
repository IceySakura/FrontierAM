import Theory.GaussianStar

/-! Lower bounds for the angular probability of a box of slopes. -/

open MeasureTheory ProbabilityTheory Set WithLp
open scoped ENNReal

namespace Cubature

noncomputable section

def chartSlope {p : ℕ} (u : Direction (p+1)) (j : Fin p) : ℝ :=
  u.val j.succ / u.val 0

theorem measurable_chartSlope {p : ℕ} (j : Fin p) : Measurable (fun u => chartSlope u j) := by
  unfold chartSlope
  fun_prop

def slopeBox {p : ℕ} (L H : Fin p → ℝ) : Set (Direction (p+1)) :=
  {u | 0 < u.val 0 ∧ ∀ j, chartSlope u j ∈ Icc (L j) (H j)}

theorem measurableSet_slopeBox {p : ℕ} (L H : Fin p → ℝ) : MeasurableSet (slopeBox L H) := by
  have hm : Measurable (fun u : Direction (p+1) => u.val 0) := by fun_prop
  have he : slopeBox L H = {u : Direction (p+1) | 0 < u.val 0} ∩
      ⋂ j, (fun u => chartSlope u j) ⁻¹' Icc (L j) (H j) := by
    ext u
    simp [slopeBox]
  rw [he]
  exact (measurableSet_lt measurable_const hm).inter
    (MeasurableSet.iInter fun j => measurable_chartSlope j measurableSet_Icc)

private theorem radialLaw_pos {d : ℕ} : ∀ᵐ r ∂radialLaw d, 0 < r := by
  rw [ae_iff]
  rw [show {r : ℝ | ¬ 0 < r} = Iic 0 by ext; simp]
  rw [radialLaw, withDensity_apply _ measurableSet_Iic]
  apply lintegral_eq_zero_of_ae_eq_zero
  filter_upwards [ae_restrict_mem (μ := volume) (measurableSet_Iic (a := (0 : ℝ)))] with r hr
  exact by simp [radialDensity, not_lt_of_ge (show r ≤ 0 from hr)]

theorem directions_preserve {p : ℕ} :
    MeasurePreserving (directionOf (p := p)) (stdGaussian (GaussianSpace (p+1)))
      (angularLaw (p+1)) where
  measurable := measurable_directionOf p
  map_eq := by
    have hp : 0 < p+1 := Nat.succ_pos _
    have := angularLaw_isProbability hp
    have := radialLaw_isProbability hp
    have he := stdGaussian_polar hp
    rw [← he.map_eq, Measure.map_map (measurable_directionOf p) he.measurable]
    have hS := (measurePreserving_snd (μ := angularLaw (p+1)) (ν := radialLaw (p+1)))
    have hr : (fun q => directionOf (rayPoint q)) =ᵐ[(angularLaw (p+1)).prod (radialLaw (p+1))]
        Prod.fst := (hS.quasiMeasurePreserving.ae radialLaw_pos).mono
      (fun q ht => directionOf_rayPoint q.1 ht)
    change ((angularLaw (p+1)).prod (radialLaw (p+1))).map
      (fun q => directionOf (rayPoint q)) = _
    rw [Measure.map_congr hr, Measure.map_fst_prod, measure_univ, one_smul]

/-- A positive lower bound on the standard normal density on `[-2,2]`. -/
def lineDensityLower : ℝ := gaussianDensity 1 2

theorem lineDensityLower_pos : 0 < lineDensityLower := by
  unfold lineDensityLower gaussianDensity gaussianConstant
  positivity

private theorem line_probability_lower {a b : ℝ} (ha : -2 ≤ a) (hb : b ≤ 2) (hab : a ≤ b) :
    ENNReal.ofReal (lineDensityLower * (b-a)) ≤ gaussianReal 0 1 (Icc a b) := by
  rw [gaussianReal_apply_eq_integral 0 (by norm_num)]
  apply ENNReal.ofReal_le_ofReal
  have hle : ∀ᵐ x ∂(volume : Measure ℝ).restrict (Icc a b),
      lineDensityLower ≤ gaussianPDFReal 0 1 x := by
    filter_upwards [ae_restrict_mem (μ := volume) measurableSet_Icc] with x hx
    rw [lineDensityLower, gaussianDensity, gaussianPDFReal, gaussianConstant]
    simp only [pow_one, sub_zero, NNReal.coe_one, mul_one, one_div]
    apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by positivity)
    have h₁ : -2 ≤ x := ha.trans hx.1
    have h₂ : x ≤ 2 := hx.2.trans hb
    nlinarith [mul_nonneg (sub_nonneg.mpr h₂) (show 0 ≤ 2+x by linarith)]
  have h := integral_mono_ae (integrable_const lineDensityLower)
    ((integrable_gaussianPDFReal 0 1).restrict (s := Icc a b)) hle
  rwa [integral_const, MeasureTheory.measureReal_def, Measure.restrict_apply_univ,
    Real.volume_Icc, ENNReal.toReal_ofReal (sub_nonneg.mpr hab), smul_eq_mul, mul_comm] at h

/-- A Gaussian cone with slope intervals in `[-1/2,1/2]` has angular mass at least the
product of their lengths times a positive dimension-dependent constant. -/
theorem slopeBox_mass_lower {p : ℕ} (L H : Fin p → ℝ) {δ : ℝ}
    (hδ : 0 ≤ δ) (hL : ∀ j, -(1/2:ℝ) ≤ L j) (hH : ∀ j, H j ≤ 1/2)
    (hW : ∀ j, δ ≤ H j - L j) :
    lineDensityLower * (lineDensityLower * δ)^p ≤
      (angularLaw (p+1)).real (slopeBox L H) := by
  let g := gaussianReal 0 1
  let other := Measure.pi (fun _ : Fin p => g)
  let τ := (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (p+1) => ℝ) 0).symm
  let point : ℝ × (Fin p → ℝ) → GaussianSpace (p+1) := toLp 2 ∘ τ
  have hp : 0 < p+1 := Nat.succ_pos _
  have := angularLaw_isProbability hp
  have hp₁ : MeasurePreserving point (g.prod other) (stdGaussian (GaussianSpace (p+1))) := by
    have hπ := (measurePreserving_piFinSuccAbove (fun _ : Fin (p+1) => g) 0).symm τ.symm
    have hmap : MeasurePreserving (toLp 2 : (Fin (p+1) → ℝ) → GaussianSpace (p+1))
        (Measure.pi (fun _ : Fin (p+1) => g)) (stdGaussian (GaussianSpace (p+1))) :=
      ⟨by fun_prop, map_pi_eq_stdGaussian⟩
    exact hmap.comp hπ
  let M := g.prod other
  let E : Set (ℝ × (Fin p → ℝ)) :=
    {q | q.1 ∈ Icc (1:ℝ) 2 ∧ ∀ j, q.2 j ∈ Icc (q.1 * L j) (q.1 * H j)}
  have hE : MeasurableSet E := by
    have he : E = Prod.fst ⁻¹' Icc (1:ℝ) 2 ∩
        ⋂ j, {q : ℝ × (Fin p → ℝ) | q.1 * L j ≤ q.2 j ∧ q.2 j ≤ q.1 * H j} := by
      ext q
      simp [E]
    rw [he]
    refine (measurable_fst measurableSet_Icc).inter (MeasurableSet.iInter (fun j => ?_))
    have hm₁ (r : ℝ) : Measurable (fun q : ℝ × (Fin p → ℝ) => q.1 * r) :=
      measurable_fst.mul_const r
    have hm₂ : Measurable (fun q : ℝ × (Fin p → ℝ) => q.2 j) :=
      (measurable_pi_apply j).comp measurable_snd
    exact (measurableSet_le (hm₁ _) hm₂).inter (measurableSet_le hm₂ (hm₁ _))
  have hsub : E ⊆ (directionOf (p := p) ∘ point) ⁻¹' (slopeBox L H) := by
    intro q hq
    rcases (show q.1 ∈ Icc (1:ℝ) 2 ∧ ∀ j, q.2 j ∈ Icc (q.1 * L j) (q.1 * H j) from hq) with
      ⟨ht, hq⟩
    have htpos : 0 < q.1 := zero_lt_one.trans_le ht.1
    have h₀ : point q 0 = q.1 := by simp [point, τ, MeasurableEquiv.piFinSuccAbove_symm_apply]
    have hi (j : Fin p) : point q j.succ = q.2 j := by
      rw [← Fin.zero_succAbove j]
      simp [point, τ, MeasurableEquiv.piFinSuccAbove_symm_apply]
    have hpt : point q ≠ 0 := by
      intro he
      rw [he, PiLp.zero_apply] at h₀
      exact htpos.ne' h₀.symm
    have hnorm := norm_pos_iff.mpr hpt
    change 0 < (directionOf (point q)).val 0 ∧ _
    rw [directionOf_val_of_ne_zero hpt]
    simp only [PiLp.smul_apply, smul_eq_mul, h₀]
    refine ⟨mul_pos (inv_pos.mpr hnorm) htpos, fun j => ?_⟩
    have hn : ‖point q‖⁻¹ ≠ 0 := by positivity
    have hEq : chartSlope (directionOf (point q)) j = q.2 j / q.1 := by
      rw [chartSlope, directionOf_val_of_ne_zero hpt, PiLp.smul_apply, PiLp.smul_apply,
        smul_eq_mul, smul_eq_mul, h₀, hi]
      exact mul_div_mul_left _ _ hn
    change chartSlope (directionOf (point q)) j ∈ _
    rw [hEq]
    exact ⟨(le_div_iff₀ htpos).mpr (by simpa [mul_comm] using (hq j).1),
      (div_le_iff₀ htpos).mpr (by simpa [mul_comm] using (hq j).2)⟩
  have hμ := directions_preserve.comp hp₁
  have hf : M E ≤ angularLaw (p+1) (slopeBox L H) :=
    (measure_mono hsub).trans_eq
      (hμ.measure_preimage (measurableSet_slopeBox L H).nullMeasurableSet)
  have hlower : ENNReal.ofReal (lineDensityLower * (lineDensityLower*δ)^p) ≤ M E := by
    change _ ≤ (g.prod other) E
    rw [Measure.prod_apply hE]
    let c := ENNReal.ofReal (lineDensityLower * δ)
    have hcp (t : ℝ) : (Icc (1:ℝ) 2).indicator (fun _ => c^p) t ≤
        other ((fun s => (t,s)) ⁻¹' E) := by
      by_cases ht : t ∈ Icc (1:ℝ) 2
      · rw [indicator_of_mem ht]
        have hT : (fun s : Fin p → ℝ => (t,s)) ⁻¹' E =
            univ.pi (fun j => Icc (t * L j) (t * H j)) := by
          ext s
          rw [mem_univ_pi]
          exact ⟨fun h => h.2, fun h => ⟨ht,h⟩⟩
        change _ ≤ (Measure.pi (fun _ : Fin p => g)) _
        rw [hT, Measure.pi_pi]
        have hL' (i : Fin p) : -2 ≤ t * L i := by
          have h := mul_le_mul_of_nonneg_left (hL i) (show 0 ≤ t by linarith [ht.1])
          linarith [ht.2]
        have hH' (i : Fin p) : t * H i ≤ 2 := by
          have h := mul_le_mul_of_nonneg_left (hH i) (show 0 ≤ t by linarith [ht.1])
          linarith [ht.2]
        have hδ' (i : Fin p) : δ ≤ t * H i - t * L i := by
          have ht' : 0 ≤ t := by linarith [ht.1]
          rw [← mul_sub]
          exact (le_mul_of_one_le_left hδ ht.1).trans (mul_le_mul_of_nonneg_left (hW i) ht')
        rw [show c^p = ∏ _ : Fin p, c by simp [Fintype.card_fin]]
        apply Finset.prod_le_prod
        intro i _
        have horder : t * L i ≤ t * H i := by linarith [hδ' i]
        exact (ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_left (hδ' i)
          lineDensityLower_pos.le)).trans (line_probability_lower (hL' i) (hH' i) horder)
      · simp [ht]
    refine le_trans ?_ (lintegral_mono hcp)
    rw [lintegral_indicator_const measurableSet_Icc, ENNReal.ofReal_mul lineDensityLower_pos.le,
      ENNReal.ofReal_pow (mul_nonneg lineDensityLower_pos.le hδ)]
    change ENNReal.ofReal lineDensityLower * c^p ≤ c^p * g (Icc 1 2)
    rw [mul_comm (ENNReal.ofReal lineDensityLower)]
    have h01 := line_probability_lower (a := 1) (b := 2) (by norm_num) (by norm_num) (by norm_num)
    gcongr
    simpa only [show (2:ℝ)-1 = 1 by norm_num, mul_one] using h01
  have hh := ENNReal.toReal_mono (measure_ne_top (angularLaw (p+1)) _) (hlower.trans hf)
  rw [ENNReal.toReal_ofReal (mul_nonneg lineDensityLower_pos.le
    (pow_nonneg (mul_nonneg lineDensityLower_pos.le hδ) _))] at hh
  exact hh

end

end Cubature
