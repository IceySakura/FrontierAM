import Theory.RadialQuantile
import Theory.UnitGrid
import Theory.SphereGrid
import Theory.IIDMean

/-! Nonadaptive cubature using a grid of preplanned tests on every queried ray. -/

open MeasureTheory Set
open scoped unitInterval

namespace Cubature

noncomputable section

private def packIndex {C n B : ℕ} : ((Fin C ⊕ Fin n) × Fin B) ≃ Fin ((C+n)*B) :=
  (Equiv.prodCongr finSumFinEquiv (Equiv.refl (Fin B))).trans finProdFinEquiv

private def fixedPoint {p B n : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (v : Samples (Direction (p+1)) n) :
    (Fin G.count ⊕ Fin n) × Fin B → GaussianSpace (p+1)
  | (.inl i, j) => rayPoint (G.point i, 1 + gridTest j 0)
  | (.inr i, j) => rayPoint ((v i).1,
      fromMass (Nat.succ_pos p) ⟨gridTest j (v i).2, gridTest_mem hB j (v i).2⟩)

/-- Both the interpolation tests and the random tests are fixed from the seed alone. -/
def fixedRadialRule {p B n : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (v : Samples (Direction (p+1)) n) :
    NonadaptiveRule (GaussianSpace (p+1)) ((G.count+n)*B) where
  point := fixedPoint G hB v ∘ packIndex.symm
  postprocess answers :=
    let P := G.predict (fun i => 1 + gridReadout (fun j : Fin B => answers (packIndex (.inl i,j))))
    let F := massCDF (p+1)
    let Z (i : Fin n) := F 1 +
      cdfSpan (p+1) * gridReadout (fun j : Fin B => answers (packIndex (.inr i,j))) -
      F (P (v i).1)
    (∫ u, F (P u) ∂angularLaw (p+1)) + (n : ℝ)⁻¹ * ∑ i, Z i

theorem fixedRadialRule_legal {p B n : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (v : Samples (Direction (p+1)) n) (i : Fin ((G.count+n)*B)) :
    (fixedRadialRule G hB v).point i ∈ coordinateBox (p+1) 40 := by
  change fixedPoint G hB v (packIndex.symm i) ∈ _
  rcases (packIndex.symm i) with ⟨a,j⟩
  cases a with
  | inl i =>
    change rayPoint (G.point i, 1 + gridTest j 0) ∈ _
    have ht := gridTest_mem hB j 0
    exact rayPoint_mem_coordinateBox _ (by linarith [ht.1]) (by linarith [ht.2])
  | inr i =>
    change rayPoint ((v i).1, fromMass (Nat.succ_pos p) ⟨gridTest j (v i).2, _⟩) ∈ _
    have ht := fromMass_mem (Nat.succ_pos p)
      ⟨gridTest j (v i).2, gridTest_mem hB j (v i).2⟩
    exact rayPoint_mem_coordinateBox _ (by linarith [ht.1]) (by linarith [ht.2])

private def fixedPredictor {p B : ℕ} (G : Predictor (Direction (p+1))) (R : Direction (p+1) → ℝ) :
    Direction (p+1) → ℝ :=
  G.predict (fun i => 1 + gridFraction B (R (G.point i) - 1) 0)

private def fixedRayMass {p B : ℕ} (R : Direction (p+1) → ℝ) (v : Direction (p+1) × I) : ℝ :=
  massCDF (p+1) 1 + cdfSpan (p+1) * gridFraction B (toMass (p+1) (R v.1)) v.2

private theorem fixed_eval {p B n : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (v : Samples (Direction (p+1)) n) (R : Direction (p+1) → ℝ) (hR : ∀ u, R u ∈ Icc 1 2) :
    (fixedRadialRule G hB v).estimate (starOracle R) =
      (∫ u, massCDF (p+1) (fixedPredictor (B := B) G R u) ∂angularLaw (p+1)) +
      (n : ℝ)⁻¹ * ∑ i, (fixedRayMass (B := B) R (v i) -
        massCDF (p+1) (fixedPredictor (B := B) G R (v i).1)) := by
  have hleft (i : Fin G.count) (j : Fin B) :
      starOracle R (fixedPoint G hB v (.inl i,j)) =
        decide (gridTest j 0 < R (G.point i) - 1) := by
    change starOracle R (rayPoint (G.point i, 1 + gridTest j 0)) = _
    have hg := (gridTest_mem hB j 0).1
    rw [starOracle_rayPoint R _ (by linarith : 0 < 1 + gridTest j 0)]
    exact decide_eq_decide.mpr (by
      change 1 + gridTest j 0 < R (G.point i) ↔ gridTest j 0 < R (G.point i) - 1
      constructor <;> intro hy <;> linarith)
  have hright (i : Fin n) (j : Fin B) :
      starOracle R (fixedPoint G hB v (.inr i,j)) =
        decide (gridTest j (v i).2 < toMass (p+1) (R (v i).1)) := by
    let q : I := ⟨gridTest j (v i).2, gridTest_mem hB j (v i).2⟩
    have hf := (fromMass_mem (Nat.succ_pos p) q).1
    change starOracle R (rayPoint ((v i).1, fromMass (Nat.succ_pos p) q)) = _
    rw [starOracle_rayPoint R _ (by linarith : 0 < fromMass (Nat.succ_pos p) q)]
    exact decide_eq_decide.mpr (fromMass_lt_iff (Nat.succ_pos p) q (hR (v i).1))
  simp only [NonadaptiveRule.estimate, fixedRadialRule, Function.comp_def, Equiv.symm_apply_apply,
    hleft,hright,fixedPredictor,fixedRayMass,gridFraction]

private theorem fixedPredictor_mem {p B : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (R : Direction (p+1) → ℝ) (u : Direction (p+1)) : fixedPredictor (B := B) G R u ∈ Icc 1 2 := by
  let z (i : Fin G.count) := 1+gridFraction B (R (G.point i)-1) 0
  have hz (i : Fin G.count) : z i ∈ Icc 1 2 := by
    have h := gridReadout_mem hB (fun j => decide (gridTest j 0 < R (G.point i)-1))
    exact ⟨by change 1 ≤ 1+gridReadout _; linarith [h.1],
      by change 1+gridReadout _ ≤ 2; linarith [h.2]⟩
  change (∑ i, G.weight i u * z i) ∈ Icc 1 2
  have coeff (c : ℝ) : ∑ i, G.weight i u * c = c := by rw [← Finset.sum_mul,G.sum_weight,one_mul]
  exact ⟨(coeff 1).symm.le.trans (Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (hz i).1 (G.weight_nonneg i u)),
    (Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hz i).2 (G.weight_nonneg i u)).trans
      (coeff 2).le⟩

private theorem fixedPredictor_error {p B : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    {R : Direction (p+1) → ℝ} {e : ℝ} (hR : ∀ u, R u ∈ Icc 1 2)
    (he : ∀ u, |G.predict (R ∘ G.point) u - R u| ≤ e) (u : Direction (p+1)) :
    |fixedPredictor (B := B) G R u - R u| ≤ e + 1/(B:ℝ) := by
  let z (i : Fin G.count) := 1+gridFraction B (R (G.point i)-1) 0
  have hz (i : Fin G.count) : |z i - R (G.point i)| ≤ 1/(B:ℝ) := by
    have hr := hR (G.point i)
    let q : I := ⟨R (G.point i)-1, ⟨by linarith [hr.1],by linarith [hr.2]⟩⟩
    have ht : z i - R (G.point i) = gridFraction B q 0 - q.val := by unfold z q; ring
    rw [ht]
    exact gridFraction_pathwise hB q 0
  have ht := G.predict_stable (z := R ∘ G.point) hz u
  change |G.predict z u - R u| ≤ _
  rw [show G.predict z u - R u =
    (G.predict z u - G.predict (R ∘ G.point) u) + (G.predict (R ∘ G.point) u - R u) by ring]
  exact (abs_add_le _ _).trans ((add_le_add ht (he u)).trans_eq (add_comm _ _))

private theorem fixedRayMass_error {p B : ℕ} (hB : 0 < B)
    {R : Direction (p+1) → ℝ} (hR : ∀ u, R u ∈ Icc 1 2) (v : Direction (p+1) × I) :
    |fixedRayMass (B := B) R v - massCDF (p+1) (R v.1)| ≤ radialDensityBound (p+1) 2 / (B:ℝ) := by
  let q : I := ⟨toMass (p+1) (R v.1), toMass_mem (Nat.succ_pos p) (hR v.1)⟩
  have hq := gridFraction_pathwise hB q v.2
  have h := cdfSpan_pos (Nat.succ_pos p)
  have ht : fixedRayMass (B := B) R v - massCDF (p+1) (R v.1) =
      cdfSpan (p+1) * (gridFraction B q v.2 - q.val) := by
    unfold fixedRayMass q
    linarith [toMass_identity (Nat.succ_pos p) (R v.1)]
  rw [ht, abs_mul, abs_of_nonneg h.le]
  calc
    _ ≤ cdfSpan (p+1) * (1/(B:ℝ)) := mul_le_mul_of_nonneg_left hq h.le
    _ ≤ _ := by
      simpa only [mul_one_div] using mul_le_mul_of_nonneg_right (cdfSpan_le (Nat.succ_pos p))
        (show (0:ℝ) ≤ 1/(B:ℝ) by positivity)

private def fixedResidual {p B : ℕ} (G : Predictor (Direction (p+1)))
    (R : Direction (p+1) → ℝ) (v : Direction (p+1) × I) : ℝ :=
  fixedRayMass (B := B) R v - massCDF (p+1) (fixedPredictor (B := B) G R v.1)

private theorem fixedResidual_measurable {p B : ℕ} (G : Predictor (Direction (p+1)))
    {R : Direction (p+1) → ℝ} (hRm : Measurable R) :
    Measurable (fixedResidual (B := B) G R) := by
  have hF : Measurable (massCDF (p+1)) := measurable_massCDF (Nat.succ_pos p)
  have hNorm : Measurable (fun u => toMass (p+1) (R u)) := by
    unfold toMass
    exact ((hF.comp hRm).sub_const (massCDF (p+1) 1)).div_const (cdfSpan (p+1))
  have hS : Measurable (Prod.snd : Direction (p+1) × I → I) := measurable_snd
  have hU : Measurable (Prod.fst : Direction (p+1) × I → Direction (p+1)) := measurable_fst
  have hT : Measurable (fun v : Direction (p+1) × I => toMass (p+1) (R v.1)) := hNorm.comp hU
  have hG : Measurable (fun v : Direction (p+1) × I => gridFraction B (toMass (p+1) (R v.1)) v.2) :=
    measurable_gridFraction_comp B hT hS
  have hP : Measurable (fixedPredictor (B := B) G R) := G.measurable_predict _
  unfold fixedResidual fixedRayMass
  exact (measurable_const.add (hG.const_mul (cdfSpan (p+1)))).sub ((hF.comp hP).comp hU)

private theorem fixedResidual_bound {p B : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    {R : Direction (p+1) → ℝ} (hR : ∀ u, R u ∈ Icc 1 2) {e : ℝ}
    (hbias : ∀ u, |G.predict (R ∘ G.point) u - R u| ≤ e) (v : Direction (p+1) × I) :
    |fixedResidual (B := B) G R v| ≤ radialDensityBound (p+1) 2 * (e + 2 / (B:ℝ)) := by
  let P := fixedPredictor (B := B) G R
  let F := massCDF (p+1)
  let W := radialDensityBound (p+1) 2
  have hW : 0 ≤ W := radialDensityBound_nonneg _ (by norm_num)
  have hp : |R v.1 - P v.1| ≤ e + 1/(B:ℝ) := by
    rw [abs_sub_comm]
    exact fixedPredictor_error G hB hR hbias v.1
  have hA : |F (R v.1) - F (P v.1)| ≤ W * (e+1/(B:ℝ)) :=
    (massCDF_absdiff (Nat.succ_pos p) (fixedPredictor_mem G hB R v.1) (hR v.1)).trans
      (mul_le_mul_of_nonneg_left hp hW)
  change |fixedRayMass (B := B) R v - F (P v.1)| ≤ _
  rw [show fixedRayMass (B := B) R v - F (P v.1) = (fixedRayMass (B := B) R v - F (R v.1)) +
    (F (R v.1)-F (P v.1)) by ring]
  refine (abs_add_le _ _).trans ((add_le_add (fixedRayMass_error hB hR v) hA).trans_eq ?_)
  change W/(B:ℝ) + W*(e+1/(B:ℝ)) = W*(e+2/(B:ℝ))
  ring

private theorem fixedRow_mean {p B : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    {R : Direction (p+1) → ℝ} (hR : ∀ u, R u ∈ Icc 1 2) (u : Direction (p+1)) :
    (∫ v : I, fixedResidual (B := B) G R (u,v)) =
      massCDF (p+1) (R u) - massCDF (p+1) (fixedPredictor (B := B) G R u) := by
  let d := p+1
  let q : I := ⟨toMass d (R u), toMass_mem (Nat.succ_pos p) (hR u)⟩
  let C := massCDF d (fixedPredictor (B := B) G R u)
  have hi := integrable_gridFraction hB q.val
  have hAdd : Integrable (fun v : I => massCDF d 1 + cdfSpan d * gridFraction B q.val v) :=
    (integrable_const (massCDF d 1)).add (hi.const_mul (cdfSpan d))
  change (∫ v : I, (massCDF d 1 + cdfSpan d * gridFraction B q.val v - C)) =
    massCDF d (R u) - C
  rw [integral_sub hAdd (integrable_const C),
    integral_add (integrable_const _) (hi.const_mul _), integral_const_mul, gridFraction_mean hB q]
  simp only [integral_const, show (volume : Measure I).real univ = 1 by simp, one_smul]
  rw [show massCDF d 1 + cdfSpan d * q.val = massCDF d (R u) from toMass_identity (Nat.succ_pos p) _]

theorem fixedRadialRule_stats {p B n : ℕ} (G : Predictor (Direction (p+1))) (hB : 0 < B)
    (hn : 0 < n) {R : Direction (p+1) → ℝ} (hRm : Measurable R)
    (hR : ∀ u, R u ∈ Icc 1 2) {e : ℝ}
    (hbias : ∀ u, |G.predict (R ∘ G.point) u - R u| ≤ e) :
    let I0 := gaussianStarMass R
    let out := fun v => (fixedRadialRule G hB v).estimate (starOracle R)
    let law := samplingLaw (angularLaw (p+1)) n
    Measurable out ∧ (∫ v, out v ∂law) = I0 ∧
      (∫ v, (out v - I0)^2 ∂law) ≤ (radialDensityBound (p+1) 2 * (e + 2 / (B:ℝ)))^2 / (n:ℝ) := by
  let d := p+1
  have hd : 0 < d := Nat.succ_pos p
  let ν := angularLaw d
  have : IsProbabilityMeasure ν := angularLaw_isProbability hd
  let μ := ν.prod (volume : Measure I)
  let P := fixedPredictor (B := B) G R
  let C := ∫ x, massCDF d (P x) ∂ν
  let Z := fixedResidual (B := B) G R
  have hZ := fixedResidual_measurable (B := B) G hRm
  have hD := fixedResidual_bound G hB hR hbias
  have hZint : Integrable Z μ :=
    (integrable_const (radialDensityBound d 2 * (e + 2/(B:ℝ)))).mono' hZ.aestronglyMeasurable
      (Filter.Eventually.of_forall fun v => by simpa only [Real.norm_eq_abs] using hD v)
  have hZmean : C + ∫ v, Z v ∂μ = gaussianStarMass R := by
    rw [integral_prod Z hZint]
    have hrow (u : Direction d) : (∫ v : I, Z (u,v)) = massCDF d (R u) - massCDF d (P u) :=
      fixedRow_mean G hB hR u
    simp_rw [hrow]
    have hP : Measurable P := G.measurable_predict _
    rw [integral_sub (integrable_massCDF ν hd hRm) (integrable_massCDF ν hd hP),
      gaussianStarMass_eq hRm]
    change C + ((∫ u, massCDF d (R u) ∂ν) - C) = ∫ u, massCDF d (R u) ∂ν
    ring
  have hs := shifted_iid_mean μ (Z := Z) hZ hD hn C
  dsimp only at hs ⊢
  rw [hZmean] at hs
  have heq (v : Samples (Direction d) n) :
      (fixedRadialRule G hB v).estimate (starOracle R) = C + (n:ℝ)⁻¹ * ∑ i, Z (v i) :=
    fixed_eval G hB v R hR
  refine ⟨?_,?_⟩
  · rw [funext heq]
    exact hs.1
  · simp_rw [heq]
    exact hs.2

end

end Cubature
