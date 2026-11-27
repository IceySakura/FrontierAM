import Theory.Slab

/-! Multilevel adaptive bracket construction, followed by an independent residual estimate. -/

open MeasureTheory Set

namespace Cubature

noncomputable section

/-- A finite interpolation rule with nonnegative weights summing to one. -/
structure Predictor (X : Type*) [MeasurableSpace X] where
  count : ℕ
  point : Fin count → X
  weight : Fin count → X → ℝ
  measurable_weight : ∀ i, Measurable (weight i)
  weight_nonneg : ∀ i x, 0 ≤ weight i x
  sum_weight : ∀ x, ∑ i, weight i x = 1

namespace Predictor

variable {X : Type*} [MeasurableSpace X]

def predict (g : Predictor X) (z : Fin g.count → ℝ) (x : X) : ℝ :=
  ∑ i, g.weight i x * z i

theorem measurable_predict (g : Predictor X) (z : Fin g.count → ℝ) :
    Measurable (g.predict z) :=
  Finset.measurable_sum _ fun i _ => (g.measurable_weight i).mul_const _

theorem predict_stable (g : Predictor X) {v z : Fin g.count → ℝ} {A : ℝ}
    (h : ∀ i, |v i - z i| ≤ A) (x : X) :
    |g.predict v x - g.predict z x| ≤ A := by
  unfold predict
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ i, |g.weight i x * (v i - z i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, g.weight i x * A := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul, abs_of_nonneg (g.weight_nonneg i x)]
      exact mul_le_mul_of_nonneg_left (h i) (g.weight_nonneg i x)
    _ = A := by rw [← Finset.sum_mul, g.sum_weight, one_mul]

end Predictor

variable {X : Type*} [MeasurableSpace X]

def clip (a b t : ℝ) : ℝ := max a (min b t)

theorem clip_mem {a b : ℝ} (hab : a ≤ b) (t : ℝ) : clip a b t ∈ Icc a b :=
  ⟨le_max_left _ _, max_le hab (min_le_left _ _)⟩

theorem clip_mono (a b : ℝ) : Monotone (clip a b) :=
  fun _ _ h => max_le_max le_rfl (min_le_min le_rfl h)

theorem clip_eq {a b t : ℝ} (ht : t ∈ Icc a b) : clip a b t = t := by
  simp [clip, min_eq_right ht.2, max_eq_right ht.1]

theorem clip_width (a b : ℝ) {l u : ℝ} (h : l ≤ u) :
    clip a b u - clip a b l ≤ u - l := by
  unfold clip
  have hm : min b l ≤ min b u := min_le_min le_rfl h
  have he : min b u - min b l ≤ u - l := by
    rcases le_total b l with hl | hl
    · rw [min_eq_left hl, min_eq_left (hl.trans h)]; linarith
    · rw [min_eq_right hl]
      linarith [min_le_right b u]
  rcases le_total a (min b l) with ha | ha
  · rwa [max_eq_right ha, max_eq_right (ha.trans hm)]
  · rw [max_eq_left ha, ← max_sub_sub_right]
    exact max_le (by linarith) (by linarith)

def precision (A : ℝ) (s k : ℕ) : ℝ := A / (2 : ℝ) ^ (s * k)

theorem precision_nonneg {A : ℝ} (hA : 0 ≤ A) (s k : ℕ) : 0 ≤ precision A s k :=
  div_nonneg hA (by positivity)

theorem precision_succ (A : ℝ) (s k : ℕ) :
    precision A s (k+1) = precision A s k / (2 : ℝ)^s := by
  rw [precision, precision, Nat.mul_succ, pow_add, div_div]

def lo (a b R y : ℝ) : ℝ := clip a b (y - R)
def hi (a b R y : ℝ) : ℝ := clip a b (y + R)

theorem bracket_order (a b y : ℝ) {R : ℝ} (hR : 0 ≤ R) :
    lo a b R y ≤ hi a b R y :=
  clip_mono _ _ (by linarith)

theorem bracket_close {a b R y r : ℝ} (hr : r ∈ Icc a b) (he : |y - r| ≤ R) :
    lo a b R y ≤ r ∧ r ≤ hi a b R y := by
  rw [abs_le] at he
  have h₁ := clip_mono a b (show y - R ≤ r by linarith [he.2])
  have h₂ := clip_mono a b (show r ≤ y + R by linarith [he.1])
  rw [clip_eq hr] at h₁ h₂
  exact ⟨h₁, h₂⟩

theorem bracket_width (a b y : ℝ) {R : ℝ} (hR : 0 ≤ R) :
    hi a b R y - lo a b R y ≤ 2 * R :=
  (clip_width a b (show y - R ≤ y + R by linarith)).trans_eq (by ring)

theorem measurable_lo {a b R : ℝ} {f : X → ℝ} (hf : Measurable f) :
    Measurable (fun x => lo a b R (f x)) := by
  unfold lo clip
  fun_prop

theorem measurable_hi {a b R : ℝ} {f : X → ℝ} (hf : Measurable f) :
    Measurable (fun x => hi a b R (f x)) := by
  unfold hi clip
  fun_prop

/-- From a level predictor with radius `2*precision`, learn all radii on the next grid. -/
def learnNext (g g' : Predictor X) (a b A : ℝ) (s k : ℕ)
    (z : Fin g.count → ℝ) : QueryTree (X × ℝ) (Fin g'.count → ℝ) :=
  QueryTree.batch g'.count fun i =>
    let x := g'.point i
    let y := g.predict z x
    let R := 2 * precision A s k
    bisect x (s + 1) (lo a b R y) (hi a b R y)

def pilot (g : ℕ → Predictor X) (a b A : ℝ) (s : ℕ) :
    (k : ℕ) → QueryTree (X × ℝ) (Fin (g k).count → ℝ)
  | 0 => .returnValue (fun _ => midpoint a b)
  | k+1 => (pilot g a b A s k).bind (learnNext (g k) (g (k+1)) a b A s k)

def pilotBudget (g : ℕ → Predictor X) (s : ℕ) : ℕ → ℕ
  | 0 => 0
  | k+1 => pilotBudget g s k + (g (k+1)).count * (s+1)

theorem pilot_depth (g : ℕ → Predictor X) (a b A : ℝ) (s k : ℕ) :
    (pilot g a b A s k).depth ≤ pilotBudget g s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [pilot, pilotBudget]
    refine (QueryTree.depth_bind_le _ _ _ (fun z => ?_)).trans (Nat.add_le_add_right ih _)
    exact QueryTree.depth_batch_le _ _ _ (fun i => le_of_eq (bisect_depth _ _ _ _))

theorem pilot_legal (g : ℕ → Predictor X) {a b A : ℝ} (hab : a ≤ b) (hA : 0 ≤ A)
    (s k : ℕ) : (pilot g a b A s k).queriesIn {q | q.2 ∈ Icc a b} := by
  induction k with
  | zero => trivial
  | succ k ih =>
    refine QueryTree.queriesIn_bind ih (fun z => QueryTree.queriesIn_batch _ _ (fun i => ?_))
    exact bisect_legal _ _ (clip_mem hab _).1
      (bracket_order _ _ _ (mul_nonneg (by norm_num) (precision_nonneg hA _ _)))
      (clip_mem hab _).2

theorem predictor_error (g : Predictor X) (h : X → ℝ) (A : ℝ) (z : Fin g.count → ℝ)
    (hbias : ∀ x, |g.predict (h ∘ g.point) x - h x| ≤ A)
    (hz : ∀ i, |z i - h (g.point i)| ≤ A) (x : X) :
    |g.predict z x - h x| ≤ 2 * A := by
  have hstab := g.predict_stable hz x
  have he : g.predict z x - h x =
      (g.predict z x - g.predict (h ∘ g.point) x) +
        (g.predict (h ∘ g.point) x - h x) := by ring
  rw [he]
  exact (abs_add_le _ _).trans ((add_le_add hstab (hbias x)).trans_eq (by ring))

theorem pilot_error (g : ℕ → Predictor X) {a b A : ℝ} (s : ℕ)
    {h : X → ℝ} (hr : ∀ x, h x ∈ Icc a b) (hspan : (b - a) / 2 ≤ A)
    (hA : 0 ≤ A)
    (hbias : ∀ k x, |(g k).predict (h ∘ (g k).point) x - h x| ≤ precision A s k)
    (k : ℕ) (i : Fin (g k).count) :
    |(pilot g a b A s k).run (threshold h) i - h ((g k).point i)| ≤ precision A s k := by
  induction k with
  | zero =>
    simp only [pilot, QueryTree.run, precision, mul_zero, pow_zero, div_one, abs_le, midpoint]
    have ch := hr ((g 0).point i)
    exact ⟨by linarith [ch.2], by linarith [ch.1]⟩
  | succ k ih =>
    simp only [pilot, QueryTree.run_bind, learnNext, QueryTree.run_batch]
    let x := (g (k+1)).point i
    let p := precision A s k
    let y := (g k).predict ((pilot g a b A s k).run (threshold h)) x
    have hc : |y - h x| ≤ 2 * p := predictor_error _ _ _ _ (hbias k) ih x
    have hb := bracket_close (hr x) hc
    have hw := bracket_width a b y (mul_nonneg (by norm_num : (0:ℝ) ≤ 2)
      (precision_nonneg hA s k))
    refine (bisect_error _ h (threshold_spec h) x (s+1) hb.1 hb.2).trans ?_
    refine (div_le_div_of_nonneg_right hw (by positivity)).trans_eq ?_
    rw [precision_succ, pow_succ]
    have hpow : (2 : ℝ)^s ≠ 0 := by positivity
    field_simp

def adaptiveEstimator (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → Predictor X)
    (a b A : ℝ) (s k n : ℕ) (v : Samples X n) : QueryTree (X × ℝ) ℝ :=
  (pilot g a b A s k).bind fun z =>
    slabEstimator ν w
      (fun x => lo a b (2 * precision A s k) ((g k).predict z x))
      (fun x => hi a b (2 * precision A s k) ((g k).predict z x)) n v

theorem adaptiveEstimator_depth (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → Predictor X)
    (a b A : ℝ) (s k n : ℕ) (v : Samples X n) :
    (adaptiveEstimator ν w g a b A s k n v).depth ≤ pilotBudget g s k + n :=
  (QueryTree.depth_bind_le _ _ _ (fun _ => slabEstimator_depth _ _ _ _ _ _)).trans
    (Nat.add_le_add_right (pilot_depth g a b A s k) n)

theorem adaptiveEstimator_legal (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → Predictor X)
    {a b A : ℝ} (hab : a ≤ b) (hA : 0 ≤ A) (s k n : ℕ) (v : Samples X n) :
    (adaptiveEstimator ν w g a b A s k n v).queriesIn {q | q.2 ∈ Icc a b} := by
  have hR : 0 ≤ 2 * precision A s k := mul_nonneg (by norm_num) (precision_nonneg hA _ _)
  exact QueryTree.queriesIn_bind (pilot_legal g hab hA s k)
    (fun _ => slabEstimator_legal _ _ _ v
      (fun _ => (clip_mem hab _).1) (fun _ => bracket_order _ _ _ hR)
      (fun _ => (clip_mem hab _).2))

theorem adaptiveEstimator_stats (ν : Measure X) [IsProbabilityMeasure ν] {w : ℝ → ℝ}
    (hw : Measurable w) (hwi : Integrable w) (hw₀ : ∀ r, 0 ≤ w r)
    (g : ℕ → Predictor X) {a b A W : ℝ} (hab : a ≤ b) (hA : 0 ≤ A)
    (hspan : (b - a) / 2 ≤ A) (hW : ∀ r ∈ Icc a b, w r ≤ W) (hW₀ : 0 ≤ W)
    (s k : ℕ) {n : ℕ} (hn : 0 < n) {h : X → ℝ} (hh : Measurable h)
    (hr : ∀ x, h x ∈ Icc a b)
    (hbias : ∀ j x, |(g j).predict (h ∘ (g j).point) x - h x| ≤ precision A s j) :
    let out := fun v => (adaptiveEstimator ν w g a b A s k n v).run (threshold h)
    let target := ∫ x, radialPrimitive w (h x) ∂ν
    Measurable out ∧
      (∫ v, out v ∂samplingLaw ν n) = target ∧
      (∫ v, (out v - target) ^ 2 ∂samplingLaw ν n) ≤
        (2 * precision A s k * W) ^ 2 / n := by
  let p := precision A s k
  let z := (pilot g a b A s k).run (threshold h)
  have hR : 0 ≤ 2 * p := mul_nonneg (by norm_num) (precision_nonneg hA _ _)
  have hb (x) := bracket_close (hr x)
    (predictor_error (g k) h p z (hbias k) (pilot_error g s hr hspan hA hbias k) x)
  have hs := slabEstimator_stats ν hw hwi hw₀
    (measurable_lo (g k |>.measurable_predict z)) (measurable_hi (g k |>.measurable_predict z)) hh
    (fun _ => (clip_mem hab _).1) (fun x => (hb x).1) (fun x => (hb x).2)
    (fun _ => (clip_mem hab _).2) (fun x => bracket_width a b ((g k).predict z x) hR)
    hW hW₀ hn
  dsimp only at hs ⊢
  simp only [adaptiveEstimator, QueryTree.run_bind]
  have he : (2 * (2 * p) * W / 2) ^ 2 / (n : ℝ) = (2 * p * W) ^ 2 / n := by ring
  rw [he] at hs
  exact hs

end

end Cubature
