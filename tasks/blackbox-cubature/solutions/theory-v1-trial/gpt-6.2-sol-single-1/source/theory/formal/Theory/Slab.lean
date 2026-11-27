import Theory.Query
import Mathlib.MeasureTheory.Constructions.UnitInterval
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Probability.Moments.Variance

/-! An unbiased, one-query correction of the residual mass in a radial bracket. -/

open MeasureTheory ProbabilityTheory Set
open scoped unitInterval BigOperators

namespace Cubature

noncomputable section

def radialPrimitive (w : ℝ → ℝ) (r : ℝ) : ℝ := ∫ t in Iio r, w t

theorem measurable_radialPrimitive {w : ℝ → ℝ} (hw : Integrable w)
    (hn : ∀ r, 0 ≤ w r) : Measurable (radialPrimitive w) := by
  refine Monotone.measurable fun l u hlu => ?_
  exact setIntegral_mono_set hw.integrableOn (Filter.Eventually.of_forall hn)
    (Filter.Eventually.of_forall fun _ ht => lt_of_lt_of_le ht hlu)

theorem radialPrimitive_nonneg {w : ℝ → ℝ} (hn : ∀ r, 0 ≤ w r) (l : ℝ) :
    0 ≤ radialPrimitive w l :=
  integral_nonneg_of_ae (Filter.Eventually.of_forall hn)

theorem radialPrimitive_le {w : ℝ → ℝ} (hw : Integrable w) (hn : ∀ r, 0 ≤ w r) (l : ℝ) :
    radialPrimitive w l ≤ ∫ t, w t := by
  rw [← setIntegral_univ]
  exact setIntegral_mono_set hw.integrableOn (Filter.Eventually.of_forall hn)
    (Filter.Eventually.of_forall fun _ _ => trivial)

theorem radialPrimitive_sub {w : ℝ → ℝ} (hw : Integrable w) {a h : ℝ} (ha : a ≤ h) :
    radialPrimitive w h - radialPrimitive w a = ∫ t in a..h, w t := by
  have hset : Iio h \ Iio a = Ico a h := by
    ext t
    simp only [mem_sdiff, mem_Iio, not_lt, mem_Ico]
    tauto
  unfold radialPrimitive
  rw [← setIntegral_sdiff measurableSet_Iio hw.integrableOn (Iio_subset_Iio ha), hset,
    intervalIntegral.integral_of_le ha, integral_Ico_eq_integral_Ioc]

variable {X : Type*} [MeasurableSpace X]

abbrev Samples (X : Type*) (n : ℕ) := Fin n → X × I

def samplingLaw (ν : Measure X) (n : ℕ) : Measure (Samples X n) :=
  Measure.pi fun _ => ν.prod (volume : Measure I)

instance {ν : Measure X} [IsProbabilityMeasure ν] (n : ℕ) :
    IsProbabilityMeasure (samplingLaw ν n) := by
  unfold samplingLaw
  infer_instance

def slabPoint (l u : X → ℝ) (s : X × I) : X × ℝ :=
  (s.1, l s.1 + (u s.1 - l s.1) * s.2.val)

theorem slabPoint_mem {l u : X → ℝ} (s : X × I) (h : l s.1 ≤ u s.1) :
    (slabPoint l u s).2 ∈ Icc (l s.1) (u s.1) := by
  have h₀ := s.2.2.1
  have h₁ := s.2.2.2
  change l s.1 ≤ _ ∧ _ ≤ u s.1
  constructor
  · simp only [slabPoint, le_add_iff_nonneg_right]
    exact mul_nonneg (sub_nonneg.mpr h) h₀
  · change l s.1 + (u s.1 - l s.1) * s.2.val ≤ u s.1
    linarith [mul_le_of_le_one_right (sub_nonneg.mpr h) h₁]

def slabValue (w : ℝ → ℝ) (l u : X → ℝ) (s : X × I) (answer : Bool) : ℝ :=
  if answer then (u s.1 - l s.1) * w (slabPoint l u s).2 else 0

def slabEstimator (ν : Measure X) (w : ℝ → ℝ)
    (l u : X → ℝ) (n : ℕ) (s : Samples X n) : QueryTree (X × ℝ) ℝ :=
  (QueryTree.batch n (fun i => .ask (slabPoint l u (s i))
    (fun b => .returnValue (slabValue w l u (s i) b)))).map
    (fun v => (∫ x, radialPrimitive w (l x) ∂ν) + (n : ℝ)⁻¹ * ∑ i, v i)

theorem slabEstimator_depth (ν : Measure X) (w : ℝ → ℝ) (l u : X → ℝ)
    (n : ℕ) (s : Samples X n) : (slabEstimator ν w l u n s).depth ≤ n := by
  unfold slabEstimator
  refine (QueryTree.depth_map_le _ _).trans ?_
  simpa using (QueryTree.depth_batch_le n 1 _ (by intro; simp [QueryTree.depth]))

theorem slabEstimator_legal (ν : Measure X) (w : ℝ → ℝ) {l u : X → ℝ} (n : ℕ)
    (s : Samples X n) {a b : ℝ}
    (h₁ : ∀ x, a ≤ l x) (h₂ : ∀ x, l x ≤ u x) (h₃ : ∀ x, u x ≤ b) :
    (slabEstimator ν w l u n s).queriesIn {q | q.2 ∈ Icc a b} := by
  unfold slabEstimator
  refine QueryTree.queriesIn_map _ (QueryTree.queriesIn_batch _ _ (fun i => ?_))
  have hi := slabPoint_mem (s i) (h₂ (s i).1)
  exact ⟨⟨(h₁ _).trans hi.1, hi.2.trans (h₃ _)⟩, fun _ => trivial⟩

private theorem integral_slab_slice (w : ℝ → ℝ) (hw : Integrable w)
    {a r b : ℝ} (ha : a ≤ r) (hb : r ≤ b) :
    (∫ v : I, if a + (b - a) * v.val < r then
          (b - a) * w (a + (b - a) * v.val) else 0)
      = radialPrimitive w r - radialPrimitive w a := by
  let g := (Iio r).indicator w
  have hg (v : ℝ) : (if a + (b - a) * v < r then (b-a) * w (a + (b-a)*v) else 0)
      = (b-a) * g (a + (b-a)*v) := by
    simp only [g, indicator, mem_Iio]
    split_ifs <;> simp
  have heq : (∫ t in a..b, g t) = ∫ t in a..r, w t := by
    rw [intervalIntegral.integral_of_le (ha.trans hb), intervalIntegral.integral_of_le ha,
      integral_indicator measurableSet_Iio, Measure.restrict_restrict measurableSet_Iio]
    have hs : Iio r ∩ Ioc a b = Ioo a r := by
      ext t
      simp only [mem_inter_iff, mem_Iio, mem_Ioc, mem_Ioo]
      exact ⟨fun h => ⟨h.2.1, h.1⟩, fun h => ⟨h.2, h.1, (h.2.le.trans hb)⟩⟩
    rw [hs, integral_Ioc_eq_integral_Ioo]
  simp_rw [hg]
  rw [radialPrimitive_sub hw ha, ← heq]
  change (∫ v : I, (b-a) * g (a + (b-a)*v.val)
    ∂((volume : Measure ℝ).comap Subtype.val)) = _
  rw [integral_subtype_comap (measurableSet_Icc (a := (0 : ℝ)) (b := 1))
    (fun v : ℝ => (b-a) * g (a + (b-a)*v)), integral_Icc_eq_integral_Ioc,
    ← intervalIntegral.integral_of_le (show (0:ℝ) ≤ 1 by norm_num), intervalIntegral.integral_const_mul]
  simpa only [mul_zero, mul_one, zero_add, add_comm a, sub_add_cancel, smul_eq_mul] using
    (intervalIntegral.smul_integral_comp_mul_add g (a := 0) (b := 1) (b - a) a)

theorem slabEstimator_stats
    (ν : Measure X) [IsProbabilityMeasure ν] {w : ℝ → ℝ}
    (hw : Measurable w) (hwi : Integrable w) (hw₀ : ∀ r, 0 ≤ w r)
    {l u h : X → ℝ} (hl : Measurable l) (hu : Measurable u) (hh : Measurable h)
    {a b D W : ℝ} (h₁ : ∀ x, a ≤ l x) (h₂ : ∀ x, l x ≤ h x)
    (h₃ : ∀ x, h x ≤ u x) (h₄ : ∀ x, u x ≤ b) (hD : ∀ x, u x - l x ≤ D)
    (hW : ∀ r ∈ Icc a b, w r ≤ W) (hW₀ : 0 ≤ W) {n : ℕ} (hn : 0 < n) :
    let out := fun s => (slabEstimator ν w l u n s).run (threshold h)
    let target := ∫ x, radialPrimitive w (h x) ∂ν
    Measurable out ∧
      (∫ s, out s ∂samplingLaw ν n) = target ∧
      (∫ s, (out s - target) ^ 2 ∂samplingLaw ν n) ≤ (D * W / 2) ^ 2 / n := by
  let μ := ν.prod (volume : Measure I)
  let C : ℝ := ∫ x, radialPrimitive w (l x) ∂ν
  let Z : X × I → ℝ := fun s => slabValue w l u s (threshold h (slabPoint l u s))
  let S : Samples X n → ℝ := ∑ i, fun s => Z (s i)
  have hzexpr (s : X × I) : Z s =
      if (slabPoint l u s).2 < h s.1 then (u s.1 - l s.1) * w (slabPoint l u s).2 else 0 := by
    by_cases he : l s.1 + (u s.1 - l s.1) * s.2.val < h s.1 <;>
      simp [Z, slabValue, threshold, slabPoint, he]
  have hZ : Measurable Z := by
    rw [funext hzexpr]
    have ht : Measurable (fun s : X × I => (slabPoint l u s).2) := by
      unfold slabPoint
      fun_prop
    exact Measurable.ite (measurableSet_lt ht (hh.comp measurable_fst)) (by fun_prop)
      measurable_const
  have hZb (s : X × I) : Z s ∈ Icc 0 (D * W) := by
    have horder := (h₂ s.1).trans (h₃ s.1)
    have htest := slabPoint_mem s horder
    have hDw : 0 ≤ D := (sub_nonneg.mpr horder).trans (hD s.1)
    rw [hzexpr]
    split
    · exact ⟨mul_nonneg (sub_nonneg.mpr horder) (hw₀ _),
        mul_le_mul (hD _) (hW _ ⟨(h₁ _).trans htest.1, htest.2.trans (h₄ _)⟩)
          (hw₀ _) hDw⟩
    · exact ⟨le_rfl, mul_nonneg hDw hW₀⟩
  have : IsProbabilityMeasure μ := by dsimp [μ]; infer_instance
  have hZlp : MemLp Z 2 μ := memLp_of_bounded (Filter.Eventually.of_forall hZb)
    hZ.aestronglyMeasurable 2
  have hS_eq : S = fun s => ∑ i, Z (s i) := by funext s; simp [S, Finset.sum_apply]
  have hS : Measurable S := by
    rw [hS_eq]
    exact Finset.measurable_sum Finset.univ (fun i _ => hZ.comp (measurable_pi_apply i))
  have hZint : Integrable Z μ := hZlp.integrable (by norm_num)
  have hFint (k : X → ℝ) (hk : Measurable k) :
      Integrable (fun x => radialPrimitive w (k x)) ν :=
    (integrable_const (∫ t, w t)).mono'
      ((measurable_radialPrimitive hwi hw₀).comp hk).aestronglyMeasurable
      (Filter.Eventually.of_forall fun x => by
        rw [Real.norm_of_nonneg (radialPrimitive_nonneg hw₀ _)]
        exact radialPrimitive_le hwi hw₀ _)
  have hmean : ∫ s, Z s ∂μ = (∫ x, radialPrimitive w (h x) ∂ν) - C := by
    rw [integral_prod Z hZint, ← integral_sub (hFint h hh) (hFint l hl)]
    apply integral_congr_ae
    exact Filter.Eventually.of_forall (fun x => by
      simp_rw [hzexpr]
      exact integral_slab_slice w hwi (h₂ x) (h₃ x))
  have hout :
      (fun s => (slabEstimator ν w l u n s).run (threshold h)) =
        fun s => C + (n : ℝ)⁻¹ * S s := by
    funext s
    simp [slabEstimator, QueryTree.run, C, S, Z, Finset.sum_apply]
  have hi (i : Fin n) : Integrable (fun s : Samples X n => Z (s i)) (samplingLaw ν n) :=
    integrable_comp_eval (μ := fun _ : Fin n => μ) (i := i) hZint
  have hsint : Integrable S (samplingLaw ν n) := by
    rw [hS_eq]
    exact integrable_finsetSum _ (fun i _ => hi i)
  have hSn : ∫ s, S s ∂samplingLaw ν n = (n : ℝ) * ∫ s, Z s ∂μ := by
    rw [hS_eq, integral_finsetSum Finset.univ (fun i _ => hi i)]
    have he (i : Fin n) : (∫ s : Samples X n, Z (s i) ∂samplingLaw ν n) = ∫ s, Z s ∂μ :=
      integral_comp_eval (μ := fun _ : Fin n => μ) (i := i) hZ.aestronglyMeasurable
    simp_rw [he]
    simp
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have houtm : Measurable (fun s => C + (n : ℝ)⁻¹ * S s) := by fun_prop
  have hmeanout : (∫ s, (C + (n : ℝ)⁻¹ * S s) ∂samplingLaw ν n) =
      ∫ x, radialPrimitive w (h x) ∂ν := by
    rw [integral_add (integrable_const _) (hsint.const_mul _), integral_const,
      integral_const_mul, hSn, ← mul_assoc, inv_mul_cancel₀ hn', one_mul, hmean]
    simp [sub_eq_add_neg, add_comm C]
  dsimp only
  refine ⟨by rwa [hout], ?_⟩
  simp_rw [congrFun hout]
  refine ⟨hmeanout, ?_⟩
  rw [← hmeanout, ← variance_eq_integral houtm.aemeasurable]
  rw [variance_const_add (hS.const_mul _).aestronglyMeasurable, variance_const_mul]
  change (n : ℝ)⁻¹ ^ 2 * variance S (Measure.pi (fun _ => μ)) ≤ _
  rw [variance_sum_pi (X := fun _ => Z) (fun _ => hZlp)]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hv : variance Z μ ≤ (D * W / 2) ^ 2 := by
    simpa using variance_le_sq_of_bounded (Filter.Eventually.of_forall hZb) hZ.aemeasurable
  calc
    _ ≤ (n : ℝ)⁻¹ ^ 2 * ((n : ℝ) * (D * W / 2) ^ 2) := by gcongr
    _ = _ := by field_simp

end

end Cubature
