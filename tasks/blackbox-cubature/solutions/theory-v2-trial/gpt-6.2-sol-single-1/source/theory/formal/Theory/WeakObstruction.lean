import Theory.Splice
import Mathlib.Analysis.Distribution.AEEqOfIntegralContDiff
import Mathlib.Analysis.Calculus.LineDeriv.IntegrationByParts
import Mathlib.Analysis.SpecialFunctions.Integrability.Basic
import Mathlib.MeasureTheory.Group.Prod

/-! The second mixed weak derivative cannot be locally square integrable. -/

open Filter Function MeasureTheory ProbabilityTheory Set
open scoped ContDiff ENNReal NNReal

namespace Cubature

def ridgeSum (y : ℝ × ℝ) : ℝ := y.1 + y.2

private lemma continuous_ridgeSum : Continuous ridgeSum := continuous_fst.add continuous_snd

private lemma hasLineDerivAt_pairRidge {u : ℝ → ℝ} {a : ℝ} {v y : ℝ × ℝ}
    (hu : HasDerivAt u a (ridgeSum y)) (hv : ridgeSum v = 1) :
    HasLineDerivAt ℝ (u ∘ ridgeSum) a y v := by
  have h (s : ℝ) : ridgeSum (y + s • v) = ridgeSum y + s := by
    change (y.1 + s * v.1) + (y.2 + s * v.2) = (y.1 + y.2) + s
    have he : v.1 + v.2 = 1 := hv
    calc
      _ = (y.1 + y.2) + s * (v.1 + v.2) := by ring
      _ = _ := by rw [he, mul_one]
  have hd : HasDerivAt u a (ridgeSum y + 0) := by simpa only [add_zero] using hu
  change HasDerivAt (fun s : ℝ => u (ridgeSum (y + s • v))) a 0
  simpa only [h] using hd.comp_const_add (ridgeSum y) 0

private lemma test_lineDeriv {v : ℝ × ℝ} {f : ℝ × ℝ → ℝ}
    (hf : ContDiff ℝ ∞ f) (hc : HasCompactSupport f) :
    ContDiff ℝ ∞ (fun x => lineDeriv ℝ f x v) ∧
      HasCompactSupport (fun x => lineDeriv ℝ f x v) ∧
      tsupport (fun x => lineDeriv ℝ f x v) ⊆ tsupport f := by
  simp only [(hf.differentiable (by simp) _).lineDeriv_eq_fderiv]
  exact ⟨(hf.fderiv_right (m := ∞) (by simp)).clm_apply contDiff_const,
    hc.fderiv_apply ℝ _, tsupport_fderiv_apply_subset ℝ _⟩

private lemma integrable_mul_of_tsupport_subset {U : Set (ℝ × ℝ)} {f v : ℝ × ℝ → ℝ}
    (hU : IsOpen U) (hf : ContinuousOn f U) (hv : Continuous v) (hc : HasCompactSupport v)
    (h : tsupport v ⊆ U) : Integrable (fun y => f y * v y) := by
  apply Continuous.integrable_of_hasCompactSupport _ hc.mul_left
  apply continuous_of_tsupport
  intro y hy
  have hyv : y ∈ tsupport v := tsupport_mul_subset_right (f := f) (g := v) hy
  exact ((hf y (h hyv)).continuousAt (hU.mem_nhds (h hyv))).mul hv.continuousAt

private lemma local_ibp {U : Set (ℝ × ℝ)} {f g v : ℝ × ℝ → ℝ} (e : ℝ × ℝ)
    (hU : IsOpen U) (hf : Continuous f) (hg : ContinuousOn g U)
    (hd : ∀ x ∈ U, HasLineDerivAt ℝ f (g x) x e)
    (hv : ContDiff ℝ ∞ v) (hc : HasCompactSupport v) (hvs : tsupport v ⊆ U) :
    (∫ y, g y * v y) = -(∫ y, f y * lineDeriv ℝ v y e) := by
  obtain ⟨hdv, hdvc, _⟩ := test_lineDeriv (v := e) hv hc
  have hgv := integrable_mul_of_tsupport_subset hU hg hv.continuous hc hvs
  have hfv : Integrable (fun y => f y * v y) :=
    (hf.mul hv.continuous).integrable_of_hasCompactSupport hc.mul_left
  have hfdv : Integrable (fun y => f y * lineDeriv ℝ v y e) :=
    (hf.mul hdv.continuous).integrable_of_hasCompactSupport hdvc.mul_left
  have he := integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable
    (B := ContinuousLinearMap.mul ℝ ℝ) hgv hfdv hfv
    (fun x hx => hd x (hvs hx))
    (fun x _ => by
      have hx := (hv.differentiable (by simp) x).hasFDerivAt.hasLineDerivAt e
      exact hx.lineDifferentiableAt.hasLineDerivAt)
  simpa only [ContinuousLinearMap.mul_apply', neg_neg] using congrArg Neg.neg he.symm

private def thinStrip : Set (ℝ × ℝ) :=
  ridgeSum ⁻¹' Ioo 0 (1 / 16) ∩ Prod.snd ⁻¹' Ioo (-1 / 4) (1 / 4)

private lemma isOpen_thinStrip : IsOpen thinStrip :=
  (isOpen_Ioo.preimage continuous_ridgeSum).inter (isOpen_Ioo.preimage continuous_snd)

private def unitBox : Set (ℝ × ℝ) := Icc (-1) 1 ×ˢ Icc (-1) 1

private lemma thinStrip_subset_unitBox : thinStrip ⊆ unitBox := by
  intro y hy
  obtain ⟨⟨h₁, h₂⟩, h₃, h₄⟩ := hy
  change 0 < y.1 + y.2 at h₁
  change y.1 + y.2 < 1 / 16 at h₂
  change -(1 : ℝ) / 4 < y.2 at h₃
  change y.2 < 1 / 4 at h₄
  exact ⟨⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩⟩

private lemma not_integrableOn_strip_inv :
    ¬ IntegrableOn (fun y => (ridgeSum y)⁻¹) thinStrip := by
  intro h
  let e : (ℝ × ℝ) ≃ₜ (ℝ × ℝ) :=
    { toFun := fun z => (z.1 - z.2, z.2)
      invFun := fun z => (z.1 + z.2, z.2)
      left_inv := fun z => by ext <;> simp
      right_inv := fun z => by ext <;> simp
      continuous_toFun := by fun_prop
      continuous_invFun := by fun_prop }
  have hp : MeasurePreserving e volume volume :=
    measurePreserving_sub_prod (volume : Measure ℝ) (volume : Measure ℝ)
  have hpre : e ⁻¹' thinStrip = Ioo 0 (1 / 16) ×ˢ Ioo (-1 / 4) (1 / 4) := by
    ext z
    change (z.1 - z.2 + z.2 ∈ Ioo (0 : ℝ) (1 / 16) ∧
      z.2 ∈ Ioo (-1 / 4) (1 / 4)) ↔ _
    rw [sub_add_cancel]
    rfl
  have hcomp : (fun y => (ridgeSum y)⁻¹) ∘ e = fun z : ℝ × ℝ => z.1⁻¹ := by
    funext z
    change (z.1 - z.2 + z.2)⁻¹ = z.1⁻¹
    rw [sub_add_cancel]
  have h' := (hp.integrableOn_comp_preimage e.measurableEmbedding).mpr h
  rw [hpre, hcomp, IntegrableOn, Measure.volume_eq_prod, ← Measure.prod_restrict] at h'
  have hμ : (volume : Measure ℝ).restrict (Ioo (-1 / 4) (1 / 4)) ≠ 0 := by
    intro he
    apply (Measure.measure_Ioo_pos (volume : Measure ℝ)).mpr
      (by norm_num : (-1 : ℝ) / 4 < 1 / 4) |>.ne'
    rw [← Measure.restrict_apply_univ, he]
    simp
  have h'' : IntegrableOn (fun u : ℝ => u⁻¹) (Ioo 0 (1 / 16)) :=
    h'.of_comp_fst hμ
  have hbad := (intervalIntegral.integrableOn_Ioo_rpow_iff (by norm_num : (0 : ℝ) < 1 / 16)).mp
    (by simpa [Real.rpow_neg_one] using h'' :
      IntegrableOn (fun u : ℝ => u ^ (-1 : ℝ)) (Ioo 0 (1 / 16)))
  exact (lt_irrefl (-1 : ℝ)) hbad

lemma no_square_integrable_weakMixedSecond_splice :
    ¬ ∃ G : ℝ × ℝ → ℝ, LocallyIntegrable G ∧
      IntegrableOn (fun y => G y ^ 2) (Icc (-1) 1 ×ˢ Icc (-1) 1) ∧
      WeakMixedSecond (spliceTail ∘ ridgeSum) G := by
  rintro ⟨G, hl, hsq, hw⟩
  let G₀ : ℝ × ℝ → ℝ := spliceTailSecond ∘ ridgeSum
  have hGc : ContinuousOn G₀ thinStrip := continuousOn_spliceTailSecond.comp
    continuous_ridgeSum.continuousOn (fun _ h => ⟨h.1.1, by linarith [h.1.2]⟩)
  have hG₀ : LocallyIntegrableOn G₀ thinStrip :=
    hGc.locallyIntegrableOn isOpen_thinStrip.measurableSet
  have hae : ∀ᵐ y ∂volume, y ∈ thinStrip → G y - G₀ y = 0 := by
    apply isOpen_thinStrip.ae_eq_zero_of_integral_contDiff_smul_eq_zero
      ((hl.locallyIntegrableOn thinStrip).sub hG₀)
    intro v hv hc hs
    have hfirst : (∫ y, G₀ y * v y) =
        -(∫ y, (spliceTailFirst ∘ ridgeSum) y * lineDeriv ℝ v y (1, 0)) := by
      apply local_ibp (1, 0) isOpen_thinStrip
        (continuous_spliceTailFirst.comp continuous_ridgeSum) hGc _ hv hc hs
      intro y hy
      apply hasLineDerivAt_pairRidge (hasDerivAt_spliceTailFirst hy.1.1 ?_) (by norm_num [ridgeSum])
      exact lt_trans hy.1.2 (by norm_num)
    obtain ⟨hDv, hDvc, _⟩ := test_lineDeriv (v := (1, 0)) hv hc
    have hsecond :
        (∫ y, (spliceTailFirst ∘ ridgeSum) y * lineDeriv ℝ v y (1, 0)) =
          -(∫ y, (spliceTail ∘ ridgeSum) y * lineDeriv ℝ
            (fun z => lineDeriv ℝ v z (1, 0)) y (0, 1)) := by
      apply local_ibp (U := univ) (0, 1) isOpen_univ
        (contDiff_spliceTail.continuous.comp continuous_ridgeSum)
        (continuous_spliceTailFirst.comp continuous_ridgeSum).continuousOn _ hDv hDvc
        (subset_univ _)
      exact fun y _ => hasLineDerivAt_pairRidge (hasDerivAt_spliceTail (ridgeSum y))
        (by norm_num [ridgeSum])
    have hIG : Integrable (fun y => G y * v y) := by
      simpa only [smul_eq_mul, mul_comm] using
        hl.integrable_smul_left_of_hasCompactSupport hv.continuous hc
    have hIG₀ := integrable_mul_of_tsupport_subset isOpen_thinStrip hGc hv.continuous hc hs
    change (∫ y, v y • (G y - G₀ y)) = 0
    simp only [smul_eq_mul, mul_sub]
    rw [integral_sub]
    · rw [sub_eq_zero]
      simpa only [mul_comm (v _), hfirst, hsecond, neg_neg] using hw v hv hc
    · simpa only [mul_comm] using hIG
    · simpa only [mul_comm] using hIG₀
  have h' : IntegrableOn (fun y => G y ^ 2) thinStrip := hsq.mono_set thinStrip_subset_unitBox
  apply not_integrableOn_strip_inv
  let C : ℝ := (gaussianPDFReal 0 1 1 / 4) ^ 2
  have hC : 0 < C := by
    have hρ := gaussianPDFReal_pos 0 1 1 (by norm_num)
    dsimp [C]
    positivity
  apply (h'.const_mul C⁻¹).mono' continuous_ridgeSum.measurable.inv.aestronglyMeasurable
  filter_upwards [hae.filter_mono (ae_mono Measure.restrict_le_self),
    ae_restrict_mem isOpen_thinStrip.measurableSet] with y hye hy
  have he : G y = G₀ y := sub_eq_zero.mp (hye hy)
  have hp : 0 < ridgeSum y := hy.1.1
  have hbound := spliceTailSecond_square_lower hp hy.1.2
  change ‖(ridgeSum y)⁻¹‖ ≤ C⁻¹ * G y ^ 2
  rw [Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hp), inv_mul_eq_div, le_div_iff₀ hC, he]
  simpa [C, G₀, div_eq_mul_inv, mul_comm] using hbound

private lemma density_lower_on_Icc {t : ℝ} (ht : t ∈ Icc (-1) 1) :
    gaussianPDFReal 0 1 1 ≤ gaussianPDFReal 0 1 t := by
  simp only [gaussianPDFReal_def, sub_zero, NNReal.coe_one, mul_one, one_pow]
  apply mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_) (by positivity)
  have h : 0 ≤ (1 - t) * (t + 1) :=
    mul_nonneg (sub_nonneg.mpr ht.2) (by linarith [ht.1])
  nlinarith only [h]

lemma square_integrable_box_of_gaussian {G : ℝ × ℝ → ℝ} (hl : LocallyIntegrable G)
    (hp : MemLp G 2 ((gaussianReal 0 1).prod (gaussianReal 0 1))) :
    IntegrableOn (fun y => G y ^ 2) (Icc (-1) 1 ×ˢ Icc (-1) 1) := by
  have hp' : MemLp G (↑(2 : ℕ) : ℝ≥0∞) ((gaussianReal 0 1).prod (gaussianReal 0 1)) := hp
  have hsq : Integrable (fun y => G y ^ 2) ((gaussianReal 0 1).prod (gaussianReal 0 1)) := by
    simpa only [Real.norm_eq_abs, sq_abs] using hp'.integrable_norm_pow'
  let w : ℝ × ℝ → ℝ≥0∞ := fun y => gaussianPDF 0 1 y.1 * gaussianPDF 0 1 y.2
  have hw : Measurable w :=
    ((measurable_gaussianPDF 0 1).comp measurable_fst).mul
      ((measurable_gaussianPDF 0 1).comp measurable_snd)
  have hwt : ∀ᵐ y ∂(volume : Measure (ℝ × ℝ)), w y < ⊤ :=
    ae_of_all _ fun _ => ENNReal.mul_lt_top gaussianPDF_lt_top gaussianPDF_lt_top
  have hgw : (gaussianReal 0 1).prod (gaussianReal 0 1) =
      (volume : Measure (ℝ × ℝ)).withDensity w := by
    rw [gaussianReal_of_var_ne_zero 0 (by norm_num : (1 : ℝ≥0) ≠ 0),
      prod_withDensity (measurable_gaussianPDF 0 1)
        (measurable_gaussianPDF 0 1)]
    rfl
  rw [hgw, integrable_withDensity_iff_integrable_smul' hw hwt] at hsq
  let c : ℝ := (gaussianPDFReal 0 1 1) ^ 2
  have hρ := gaussianPDFReal_pos 0 1 1 (by norm_num)
  have hc : 0 < c := by dsimp [c]; positivity
  have hbig : IntegrableOn (fun y => c⁻¹ * (w y).toReal * G y ^ 2) unitBox := by
    simpa only [smul_eq_mul, mul_assoc] using (hsq.const_mul c⁻¹).integrableOn
  have hM : AEStronglyMeasurable (fun y => G y ^ 2) (volume : Measure (ℝ × ℝ)) := by
    rw [show (fun y => G y ^ 2) = G * G by funext y; simp [pow_two]]
    exact hl.aestronglyMeasurable.mul hl.aestronglyMeasurable
  apply hbig.mono' hM.restrict
  filter_upwards [ae_restrict_mem (measurableSet_Icc.prod measurableSet_Icc : MeasurableSet unitBox)]
    with y hy
  have hw' : c ≤ (w y).toReal := by
    simp only [c, w, ENNReal.toReal_mul, toReal_gaussianPDF, pow_two]
    exact mul_le_mul (density_lower_on_Icc hy.1) (density_lower_on_Icc hy.2) hρ.le
      (gaussianPDFReal_nonneg _ _ _)
  calc
    ‖G y ^ 2‖ = G y ^ 2 := Real.norm_of_nonneg (sq_nonneg _)
    _ = c⁻¹ * c * G y ^ 2 := by rw [inv_mul_cancel₀ hc.ne', one_mul]
    _ ≤ c⁻¹ * (w y).toReal * G y ^ 2 := by gcongr

end Cubature
