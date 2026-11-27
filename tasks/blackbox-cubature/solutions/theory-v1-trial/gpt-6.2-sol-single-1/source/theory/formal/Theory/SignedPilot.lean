import Theory.Adaptive

/-! The multilevel argument with signed interpolation weights. -/

open MeasureTheory Set

namespace Cubature

noncomputable section

structure SignedPredictor (X : Type*) [MeasurableSpace X] (Λ : ℝ) where
  count : ℕ
  point : Fin count → X
  weight : Fin count → X → ℝ
  measurable_weight : ∀ i, Measurable (weight i)
  absWeight_le : ∀ x, ∑ i, |weight i x| ≤ Λ

variable {X : Type*} [MeasurableSpace X] {Λ : ℝ}

def SignedPredictor.eval (g : SignedPredictor X Λ) (z : Fin g.count → ℝ) (x : X) : ℝ :=
  ∑ i, g.weight i x * z i

theorem SignedPredictor.measurable_eval (g : SignedPredictor X Λ) (z : Fin g.count → ℝ) :
    Measurable (g.eval z) :=
  Finset.measurable_sum _ fun i _ => (g.measurable_weight i).mul_const _

theorem SignedPredictor.stable (g : SignedPredictor X Λ) {z w : Fin g.count → ℝ}
    {δ : ℝ} (hδ : 0 ≤ δ) (h : ∀ i, |z i - w i| ≤ δ) (x : X) :
    |g.eval z x - g.eval w x| ≤ Λ*δ := by
  unfold SignedPredictor.eval
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub]
  calc
    _ ≤ ∑ i, |g.weight i x * (z i - w i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |g.weight i x| * δ := by
      simp_rw [abs_mul]
      exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (h i) (abs_nonneg _)
    _ ≤ _ := by
      rw [← Finset.sum_mul]
      exact mul_le_mul_of_nonneg_right (g.absWeight_le x) hδ

private theorem prediction_bound (g : SignedPredictor X Λ) {h : X → ℝ}
    {δ : ℝ} (hδ : 0 ≤ δ) {ell : ℕ} (hΛ : Λ+1 ≤ (2:ℝ)^ell)
    {z : Fin g.count → ℝ} (hz : ∀ i, |z i - h (g.point i)| ≤ δ)
    (hBias : ∀ x, |g.eval (h ∘ g.point) x - h x| ≤ δ) (x : X) :
    |g.eval z x - h x| ≤ (2:ℝ)^ell * δ := by
  have hS := g.stable (w := h ∘ g.point) hδ hz x
  rw [show g.eval z x - h x =
    (g.eval z x - g.eval (h ∘ g.point) x) + (g.eval (h ∘ g.point) x - h x) by ring]
  refine (abs_add_le _ _).trans ((add_le_add hS (hBias x)).trans ?_)
  rw [show Λ*δ + δ = (Λ+1)*δ by ring]
  exact mul_le_mul_of_nonneg_right hΛ hδ

private def signedStep (g g' : SignedPredictor X Λ) (a b E : ℝ) (s ell j : ℕ)
    (z : Fin g.count → ℝ) : QueryTree (X × ℝ) (Fin g'.count → ℝ) :=
  QueryTree.batch g'.count (fun i =>
    let x := g'.point i
    let y := g.eval z x
    let D := (2:ℝ)^ell * precision E s j
    bisect x (s+ell) (lo a b D y) (hi a b D y))

def signedPilot (g : ℕ → SignedPredictor X Λ) (a b E : ℝ) (s ell : ℕ) :
    (k : ℕ) → QueryTree (X × ℝ) (Fin (g k).count → ℝ)
  | 0 => .returnValue (fun _ => midpoint a b)
  | k+1 => (signedPilot g a b E s ell k).bind (signedStep (g k) (g (k+1)) a b E s ell k)

def signedPilotBudget (g : ℕ → SignedPredictor X Λ) (s ell : ℕ) : ℕ → ℕ
  | 0 => 0
  | k+1 => signedPilotBudget g s ell k + (g (k+1)).count * (s+ell)

theorem signedPilot_depth (g : ℕ → SignedPredictor X Λ) (a b E : ℝ) (s ell k : ℕ) :
    (signedPilot g a b E s ell k).depth ≤ signedPilotBudget g s ell k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [signedPilot, signedPilotBudget]
    refine (QueryTree.depth_bind_le _ _ _ (fun _ => ?_)).trans (Nat.add_le_add_right ih _)
    exact QueryTree.depth_batch_le _ _ _ (fun _ => le_of_eq (bisect_depth _ _ _ _))

theorem signedPilot_legal (g : ℕ → SignedPredictor X Λ) {a b E : ℝ}
    (hab : a ≤ b) (hE : 0 ≤ E) (s ell k : ℕ) :
    (signedPilot g a b E s ell k).queriesIn {q | q.2 ∈ Icc a b} := by
  induction k with
  | zero => trivial
  | succ k ih =>
    refine QueryTree.queriesIn_bind ih (fun _ => QueryTree.queriesIn_batch _ _ (fun _ => ?_))
    exact bisect_legal _ _ (clip_mem hab _).1 (bracket_order _ _ _
      (mul_nonneg (by positivity) (precision_nonneg hE _ _))) (clip_mem hab _).2

theorem signedPilot_accuracy (g : ℕ → SignedPredictor X Λ) (s ell : ℕ) {a b E : ℝ}
    (hΛ : Λ+1 ≤ (2:ℝ)^ell) (hE : 0 ≤ E) (hspan : (b-a)/2 ≤ E)
    {h : X → ℝ} (hh : ∀ x, h x ∈ Icc a b)
    (hBias : ∀ j x, |(g j).eval (h ∘ (g j).point) x - h x| ≤ precision E s j)
    (k : ℕ) (i : Fin (g k).count) :
    |(signedPilot g a b E s ell k).run (threshold h) i - h ((g k).point i)| ≤ precision E s k := by
  induction k with
  | zero =>
    simp only [signedPilot, QueryTree.run, midpoint, precision, mul_zero, pow_zero, div_one, abs_le]
    have hc := hh ((g 0).point i)
    exact ⟨by linarith [hc.2], by linarith [hc.1]⟩
  | succ j ih =>
    simp only [signedPilot, QueryTree.run_bind, signedStep, QueryTree.run_batch]
    let x := (g (j+1)).point i
    let y := (g j).eval ((signedPilot g a b E s ell j).run (threshold h)) x
    let δ := precision E s j
    let D := (2:ℝ)^ell * δ
    have hδ : 0 ≤ δ := precision_nonneg hE _ _
    have hD : 0 ≤ D := mul_nonneg (by positivity) hδ
    have hc := bracket_close (hh x) (prediction_bound (g j) hδ hΛ ih (hBias j) x)
    have hw := bracket_width a b y hD
    refine (bisect_error _ h (threshold_spec h) x (s+ell) hc.1 hc.2).trans
      ((div_le_div_of_nonneg_right hw (by positivity)).trans_eq ?_)
    change (2*D)/(2*(2:ℝ)^(s+ell)) = precision E s (j+1)
    rw [precision_succ, pow_add]
    unfold D
    have h₁ : (2:ℝ)^s ≠ 0 := by positivity
    have h₂ : (2:ℝ)^ell ≠ 0 := by positivity
    field_simp
    rfl

def signedEstimator (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → SignedPredictor X Λ)
    (a b E : ℝ) (s ell k n : ℕ) (v : Samples X n) : QueryTree (X × ℝ) ℝ :=
  (signedPilot g a b E s ell k).bind (fun z =>
    let D := (2:ℝ)^ell * precision E s k
    slabEstimator ν w (fun x => lo a b D ((g k).eval z x))
      (fun x => hi a b D ((g k).eval z x)) n v)

theorem signedEstimator_depth (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → SignedPredictor X Λ)
    (a b E : ℝ) (s ell k n : ℕ) (v : Samples X n) :
    (signedEstimator ν w g a b E s ell k n v).depth ≤ signedPilotBudget g s ell k + n :=
  (QueryTree.depth_bind_le _ _ _ (fun _ => slabEstimator_depth _ _ _ _ _ _)).trans
    (Nat.add_le_add_right (signedPilot_depth g a b E s ell k) n)

theorem signedEstimator_legal (ν : Measure X) (w : ℝ → ℝ) (g : ℕ → SignedPredictor X Λ)
    {a b E : ℝ} (hab : a ≤ b) (hE : 0 ≤ E) (s ell k n : ℕ) (v : Samples X n) :
    (signedEstimator ν w g a b E s ell k n v).queriesIn {q | q.2 ∈ Icc a b} := by
  refine QueryTree.queriesIn_bind (signedPilot_legal g hab hE s ell k) (fun _ =>
    slabEstimator_legal _ _ _ v (fun _ => (clip_mem hab _).1)
      (fun _ => bracket_order _ _ _ (mul_nonneg (by positivity) (precision_nonneg hE _ _)))
      (fun _ => (clip_mem hab _).2))

theorem signedEstimator_stats (ν : Measure X) [IsProbabilityMeasure ν] {w : ℝ → ℝ}
    (hw : Measurable w) (hwi : Integrable w) (hw₀ : ∀ r, 0 ≤ w r)
    (g : ℕ → SignedPredictor X Λ) {a b E W : ℝ} (hab : a ≤ b) (hE : 0 ≤ E)
    (hspan : (b-a)/2 ≤ E) (hW : ∀ r ∈ Icc a b, w r ≤ W) (hW₀ : 0 ≤ W)
    (s ell k : ℕ) (hΛ : Λ+1 ≤ (2:ℝ)^ell) {n : ℕ} (hn : 0 < n) {h : X → ℝ}
    (hh : Measurable h) (hr : ∀ x, h x ∈ Icc a b)
    (hBias : ∀ j x, |(g j).eval (h ∘ (g j).point) x - h x| ≤ precision E s j) :
    let out := fun v => (signedEstimator ν w g a b E s ell k n v).run (threshold h)
    let target := ∫ x, radialPrimitive w (h x) ∂ν
    Measurable out ∧
      (∫ v, out v ∂samplingLaw ν n) = target ∧
      (∫ v, (out v - target)^2 ∂samplingLaw ν n) ≤
        ((2:ℝ)^ell * precision E s k * W)^2 / (n:ℝ) := by
  let p := precision E s k
  let D := (2:ℝ)^ell * p
  let z := (signedPilot g a b E s ell k).run (threshold h)
  have hp : 0 ≤ p := precision_nonneg hE _ _
  have hD : 0 ≤ D := mul_nonneg (by positivity) hp
  have hb (x) := bracket_close (hr x)
    (prediction_bound (g k) hp hΛ (signedPilot_accuracy g s ell hΛ hE hspan hr hBias k) (hBias k) x)
  have hs := slabEstimator_stats ν hw hwi hw₀
    (measurable_lo ((g k).measurable_eval z)) (measurable_hi ((g k).measurable_eval z)) hh
    (fun _ => (clip_mem hab _).1) (fun x => (hb x).1) (fun x => (hb x).2)
    (fun _ => (clip_mem hab _).2) (fun x => bracket_width a b ((g k).eval z x) hD)
    hW hW₀ hn
  dsimp only at hs ⊢
  simp only [signedEstimator, QueryTree.run_bind]
  rw [show 2*D*W/2 = D*W by ring] at hs
  exact hs

end

end Cubature
