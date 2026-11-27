import Theory.GaussianDilation
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic.FunProp

open MeasureTheory ProbabilityTheory Set Filter
open scoped BigOperators ENNReal Topology

namespace Cubature

noncomputable section

def roundBody (d : ℕ) (r : ℝ) : Set (Vec d) :=
  {z | ∑ i, (z i)^2 ≤ r^2}

def roundMass (d : ℕ) (r : ℝ) : ℝ :=
  (gaussian d).real (roundBody d r)

/-- A fixed positive radius bound; it is independent of the requested accuracy. -/
def ballMinRadius (d : ℕ) : ℝ := roundMass d 1 / 8

def sphereTranscript {d : ℕ} (r : ℝ) (points : List (Vec d)) : List Bool := by
  classical
  exact points.map fun z => decide (z ∈ roundBody d r)

def ballSuccess {d : ℕ} {Ω : Type*} (r ε : ℝ)
    (queries : Ω → List (Vec d)) (answer : Ω → List Bool → ℝ) : Set Ω :=
  {ω | |answer ω (sphereTranscript r (queries ω)) - roundMass d r| ≤ ε * roundMass d r}

lemma measurableSet_roundBody (d : ℕ) (r : ℝ) : MeasurableSet (roundBody d r) := by
  unfold roundBody
  exact measurableSet_le (by fun_prop) measurable_const

lemma roundBody_mono {d : ℕ} {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    roundBody d a ⊆ roundBody d b := by
  intro z hz
  exact (show ∑ i, (z i)^2 ≤ a^2 from hz).trans (pow_le_pow_left₀ ha hab 2)

lemma roundMass_mono {d : ℕ} {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    roundMass d a ≤ roundMass d b :=
  measureReal_mono (roundBody_mono ha hab)

lemma standard_pdf_le_one (x : ℝ) : gaussianPDFReal 0 1 x ≤ 1 := by
  have hp : 1 ≤ Real.sqrt (2 * Real.pi) := by
    have hz := Real.pi_gt_three
    exact Real.le_sqrt_of_sq_le (by linarith)
  simp only [gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
  refine mul_le_one₀ (inv_le_one_of_one_le₀ hp) (Real.exp_nonneg _) ?_
  exact Real.exp_le_one_iff.mpr (div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (sq_nonneg _)) (by norm_num))

lemma gaussian_le_volume : gaussianReal 0 1 ≤ (volume : Measure ℝ) := by
  rw [gaussianReal_of_var_ne_zero _ one_ne_zero]
  calc
    volume.withDensity (gaussianPDF 0 1) ≤ (volume : Measure ℝ).withDensity 1 := by
      apply withDensity_mono (μ := (volume : Measure ℝ))
      apply ae_of_all
      intro x
      change ENNReal.ofReal (gaussianPDFReal 0 1 x) ≤ 1
      rw [← ENNReal.ofReal_one]
      exact ENNReal.ofReal_le_ofReal (standard_pdf_le_one x)
    _ = volume := withDensity_one

lemma roundMass_le_two_radius (d : ℕ) (hd : 0 < d) (r : ℝ) (hr : 0 ≤ r) :
    roundMass d r ≤ 2*r := by
  let i : Fin d := ⟨0,hd⟩
  have hsub : roundBody d r ⊆ (fun z : Vec d => z i) ⁻¹' Icc (-r) r := by
    intro z hz
    have hsq : (z i)^2 ≤ r^2 := (Finset.single_le_sum
      (fun j (_ : j ∈ Finset.univ) => sq_nonneg (z j)) (Finset.mem_univ i)).trans hz
    change z i ∈ Icc (-r) r
    exact abs_le.mp ((sq_le_sq₀ (abs_nonneg (z i)) hr).mp (by simpa [sq_abs] using hsq))
  have hmap : (gaussian d).map (fun z : Vec d => z i) = gaussianReal 0 1 := by
    change (gaussian d).map (Function.eval i) = _
    simp [gaussian, Measure.pi_map_eval]
  calc
    roundMass d r ≤ (gaussian d).real ((fun z : Vec d => z i) ⁻¹' Icc (-r) r) :=
      measureReal_mono hsub
    _ = (gaussianReal 0 1).real (Icc (-r) r) := by
      unfold Measure.real
      rw [← Measure.map_apply (measurable_pi_apply i) measurableSet_Icc, hmap]
    _ ≤ 2*r := by
      have hnu := gaussian_le_volume (Icc (-r) r)
      rw [Real.volume_Icc] at hnu
      have h2 := ENNReal.toReal_mono ENNReal.ofReal_ne_top hnu
      rw [ENNReal.toReal_ofReal (by linarith only [hr])] at h2
      unfold Measure.real
      linarith only [h2]

lemma roundMass_one_pos (d : ℕ) : 0 < roundMass d 1 := by
  let w : ℝ := 1/((d : ℝ)+1)
  have hw : 0 < w := by unfold w; positivity
  have hw1 : w ≤ 1 := by unfold w; apply (div_le_iff₀ (by positivity)).2; simp
  have hwd : (d : ℝ)*w ≤ 1 := by
    unfold w
    rw [mul_one_div, div_le_iff₀ (by positivity)]
    simp
  have hw2 : (d : ℝ)*w^2 ≤ 1 := calc
    _ = ((d : ℝ)*w)*w := by ring
    _ ≤ 1*w := mul_le_mul_of_nonneg_right hwd hw.le
    _ ≤ 1 := by simpa using hw1
  have hsub : univ.pi (fun _ : Fin d => Icc (-w) w) ⊆ roundBody d 1 := by
    intro z hz
    simp only [mem_univ_pi, mem_Icc] at hz
    have hp : ∀ i : Fin d, (z i)^2 ≤ w^2 := by
      intro i
      nlinarith only [mul_nonneg (sub_nonneg.mpr (hz i).2) (show 0 ≤ z i + w by
        linarith only [(hz i).1])]
    change ∑ i, (z i)^2 ≤ (1 : ℝ)^2
    calc
      _ ≤ ∑ _ : Fin d, w^2 := Finset.sum_le_sum fun i _ => hp i
      _ ≤ _ := by simpa using hw2
  have hmass : 0 < (gaussianReal 0 1) (Icc (-w) w) := by
    apply pos_iff_ne_zero.mpr
    intro hz
    have hv := gaussianReal_absolutelyContinuous' 0 (v := 1) one_ne_zero hz
    rw [Real.volume_Icc, ENNReal.ofReal_eq_zero] at hv
    linarith only [hv, hw]
  have hprod : 0 < (gaussian d) (univ.pi (fun _ : Fin d => Icc (-w) w)) := by
    unfold gaussian
    rw [Measure.pi_pi]
    exact pos_iff_ne_zero.mpr (Finset.prod_ne_zero_iff.mpr (fun _ _ => hmass.ne'))
  unfold roundMass
  exact ENNReal.toReal_pos (hprod.trans_le (measure_mono hsub)).ne' (measure_ne_top _ _)

private lemma roundMass_dilate (d : ℕ) (r s : ℝ) (hs : 1 ≤ s) :
    roundMass d (s*r) ≤ s^d * roundMass d r := by
  have hs0 : 0 < s := zero_lt_one.trans_le hs
  have hh : contract (0 : Vec d) s ⁻¹' roundBody d r = roundBody d (s*r) := by
    ext z
    simp [contract, roundBody, div_pow, ← Finset.sum_div, div_le_iff₀ (sq_pos_of_pos hs0),
      mul_pow, mul_comm]
  have h := gaussian_mass_contract_le (a := (0 : Vec d)) hs (measurableSet_roundBody d r)
  rw [hh] at h
  simpa [roundMass, dilationFactor] using h

lemma continuousOn_roundMass (d : ℕ) : ContinuousOn (roundMass d) (Ioi 0) := by
  intro r hr
  have hr0 : 0 < r := hr
  let f := roundMass d
  have h1 : Continuous (fun t : ℝ => min 1 ((t/r)^d) * f r) := by
    fun_prop
  have h2 : Continuous (fun t : ℝ => max 1 ((t/r)^d) * f r) := by
    fun_prop
  have L : Tendsto (fun t : ℝ => min 1 ((t/r)^d) * f r) (𝓝[Ioi 0] r) (𝓝 (f r)) := by
    have h := h1.continuousWithinAt (s := Ioi 0) (x := r)
    rw [ContinuousWithinAt] at h
    simpa [hr0.ne'] using h
  have U : Tendsto (fun t : ℝ => max 1 ((t/r)^d) * f r) (𝓝[Ioi 0] r) (𝓝 (f r)) := by
    have h := h2.continuousWithinAt (s := Ioi 0) (x := r)
    rw [ContinuousWithinAt] at h
    simpa [hr0.ne'] using h
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' L U
  · filter_upwards [self_mem_nhdsWithin] with t ht
    have ht0 : 0 < t := ht
    by_cases hrt : r ≤ t
    · have h := pow_le_pow_left₀ (show (0 : ℝ) ≤ 1 by norm_num)
        ((le_div_iff₀ hr0).2 (by simpa using hrt)) d
      rw [one_pow] at h
      rw [min_eq_left h, one_mul]
      exact roundMass_mono hr0.le hrt
    · have hh : (t/r)^d ≤ 1 :=
        (pow_le_pow_left₀ (by positivity) ((div_le_iff₀ hr0).2 (by simpa using le_of_not_ge hrt)) d).trans_eq (one_pow _)
      rw [min_eq_right hh]
      have hs : 1 ≤ r/t := (le_div_iff₀ ht0).2 (by simpa using le_of_not_ge hrt)
      have h := roundMass_dilate d t (r/t) hs
      rw [div_mul_cancel₀ _ ht0.ne'] at h
      have he : (t/r)^d * (r/t)^d = 1 := by rw [← mul_pow]; field_simp; simp
      calc
        (t/r)^d * f r ≤ (t/r)^d * ((r/t)^d * f t) :=
          mul_le_mul_of_nonneg_left h (by positivity)
        _ = f t := by rw [← mul_assoc, he, one_mul]
  · filter_upwards [self_mem_nhdsWithin] with t ht
    have ht0 : 0 < t := ht
    by_cases hrt : r ≤ t
    · have hs : 1 ≤ t/r := (le_div_iff₀ hr0).2 (by simpa using hrt)
      have hh := pow_le_pow_left₀ (show (0 : ℝ) ≤ 1 by norm_num) hs d
      rw [one_pow] at hh
      rw [max_eq_right hh]
      have h := roundMass_dilate d r (t/r) hs
      rwa [div_mul_cancel₀ _ hr0.ne'] at h
    · have htr : t ≤ r := le_of_not_ge hrt
      have hh : (t/r)^d ≤ 1 :=
        (pow_le_pow_left₀ (by positivity) ((div_le_iff₀ hr0).2 (by simpa using htr)) d).trans_eq (one_pow _)
      rw [max_eq_left hh, one_mul]
      exact roundMass_mono ht0.le htr

lemma ballMass_packing {d m : ℕ} (hd : 0 < d) (hm : 0 < m) :
    ∃ r : Fin m → ℝ,
      (∀ i, r i ∈ Icc (ballMinRadius d) 1) ∧
      (∀ i, roundMass d (r i) = roundMass d 1 * (1/2 + (i.val : ℝ)/(2*m))) ∧
      Monotone r := by
  have hpos := roundMass_one_pos d
  have hbound : roundMass d 1 ≤ 1 := measureReal_le_one
  have hmin : 0 < ballMinRadius d := by unfold ballMinRadius; positivity
  have hmin1 : ballMinRadius d ≤ 1 := by unfold ballMinRadius; linarith only [hbound]
  have hbase := roundMass_le_two_radius d hd (ballMinRadius d) hmin.le
  have hcont : ContinuousOn (roundMass d) (Icc (ballMinRadius d) 1) :=
    (continuousOn_roundMass d).mono (fun _ ht => hmin.trans_le ht.1)
  have hR (i : Fin m) : ∃ r ∈ Icc (ballMinRadius d) 1,
      roundMass d r = roundMass d 1 * (1/2 + (i.val : ℝ)/(2*m)) := by
    have hm0 : 0 < (m : ℝ) := by exact_mod_cast hm
    have hi : (i.val : ℝ) ≤ m := by exact_mod_cast i.isLt.le
    have hu : (i.val : ℝ)/(2*m) ≤ 1/2 := (div_le_iff₀ (by positivity)).2 (by linarith only [hi])
    have hl : 0 ≤ (i.val : ℝ)/(2*m) := by positivity
    apply intermediate_value_Icc hmin1 hcont
    constructor
    · unfold ballMinRadius at hbase
      exact hbase.trans (by nlinarith only [hpos, mul_nonneg hpos.le hl])
    · nlinarith only [mul_nonneg hpos.le (sub_nonneg.mpr hu)]
  choose r hr hmass using hR
  refine ⟨r, hr, hmass, ?_⟩
  intro i j hij
  rcases eq_or_lt_of_le hij with rfl | hh
  · exact le_rfl
  · by_contra! hgt
    have hmono := roundMass_mono (d := d) (hmin.le.trans (hr j).1) hgt.le
    rw [hmass i, hmass j] at hmono
    have hx : (i.val : ℝ) < j.val := by exact_mod_cast hh
    have hq : (i.val : ℝ)/(2*m) < (j.val : ℝ)/(2*m) := div_lt_div_of_pos_right hx (by positivity)
    linarith only [hmono, mul_pos hpos (sub_pos.mpr hq)]

end

end Cubature
