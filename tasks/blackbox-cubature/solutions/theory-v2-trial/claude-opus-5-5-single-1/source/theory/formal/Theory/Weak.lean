import Theory.Defs

/-!
# Classical iterated partial derivatives are weak derivatives

If all the iterated partial derivatives of `F` along (the suffixes of) a list `l` exist and are
continuous, then for every smooth compactly supported test function `v`,
`∫ F · D_{reverse l} v = (-1)^{|l|} ∫ (D_l F) · v`, i.e. `D_l F` is the weak derivative.
-/

open MeasureTheory Filter Topology

namespace GKLS

variable {n : ℕ}

lemma update_eq_add_single (x : Fin n → ℝ) (k : Fin n) (t : ℝ) :
    Function.update x k (x k + t) = x + t • Pi.single k (1 : ℝ) := by
  funext i
  by_cases h : i = k
  · subst h; simp
  · simp [Function.update_of_ne h, Pi.single_apply, h]

/-- A partial derivative that exists is a line derivative in the coordinate direction. -/
lemma hasLineDerivAt_of_partial {F : (Fin n → ℝ) → ℝ} {k : Fin n} {x : Fin n → ℝ}
    (h : DifferentiableAt ℝ (fun t => F (Function.update x k t)) (x k)) :
    HasLineDerivAt ℝ F (partialDeriv k F x) x (Pi.single k 1) := by
  have h1 : HasDerivAt (fun t => F (Function.update x k t)) (partialDeriv k F x) (x k) :=
    h.hasDerivAt
  have h2 : HasDerivAt (fun t => F (Function.update x k (x k + t))) (partialDeriv k F x) 0 := by
    have hh : HasDerivAt (fun t : ℝ => x k + t) 1 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_add (x k)
    have h1' : HasDerivAt (fun t => F (Function.update x k t)) (partialDeriv k F x) (x k + 0) := by
      simpa using h1
    have := h1'.comp 0 hh
    simpa [Function.comp_def] using this
  unfold HasLineDerivAt
  simpa [update_eq_add_single] using h2

/-- The partial derivative of a smooth function, as a Fréchet derivative applied to `e_k`. -/
lemma partialDeriv_eq_fderiv {v : (Fin n → ℝ) → ℝ} (hv : Differentiable ℝ v) (k : Fin n) :
    partialDeriv k v = fun x => fderiv ℝ v x (Pi.single k 1) := by
  funext x
  have hc := (hv (Function.update x k (x k))).hasFDerivAt.comp_hasDerivAt (x k)
    (hasDerivAt_update x k (x k))
  simp only [Function.update_eq_self] at hc
  exact hc.deriv

/-- Test functions are stable under partial derivatives. -/
lemma testFun_partialDeriv {v : (Fin n → ℝ) → ℝ} (hv : ContDiff ℝ (⊤ : ℕ∞) v)
    (hvs : HasCompactSupport v) (k : Fin n) :
    ContDiff ℝ (⊤ : ℕ∞) (partialDeriv k v) ∧ HasCompactSupport (partialDeriv k v) := by
  have hd : Differentiable ℝ v := hv.differentiable (by simp)
  rw [partialDeriv_eq_fderiv hd]
  refine ⟨?_, ?_⟩
  · exact (hv.fderiv_right (by simp)).clm_apply contDiff_const
  · exact hvs.fderiv_apply (𝕜 := ℝ) (Pi.single k 1)

lemma testFun_iterPartial {v : (Fin n → ℝ) → ℝ} (hv : ContDiff ℝ (⊤ : ℕ∞) v)
    (hvs : HasCompactSupport v) :
    ∀ l : List (Fin n), ContDiff ℝ (⊤ : ℕ∞) (iterPartial l v) ∧ HasCompactSupport (iterPartial l v) := by
  intro l
  induction l with
  | nil => exact ⟨hv, hvs⟩
  | cons k l ih => exact testFun_partialDeriv ih.1 ih.2 k

lemma iterPartial_append (l₁ l₂ : List (Fin n)) (F : (Fin n → ℝ) → ℝ) :
    iterPartial (l₁ ++ l₂) F = iterPartial l₁ (iterPartial l₂ F) := by
  simp [iterPartial, List.foldr_append]

/-- All the suffix derivatives exist and are continuous. -/
def IterPartialCont : List (Fin n) → ((Fin n → ℝ) → ℝ) → Prop
  | [], F => Continuous F
  | k :: l, F => IterPartialCont l F ∧ Continuous (iterPartial (k :: l) F)

/-- **Classical derivatives are weak derivatives.** -/
theorem integral_iterPartial_eq (F : (Fin n → ℝ) → ℝ) :
    ∀ l : List (Fin n), IterPartialExists l F → IterPartialCont l F →
      ∀ v : (Fin n → ℝ) → ℝ, ContDiff ℝ (⊤ : ℕ∞) v → HasCompactSupport v →
        ∫ x, F x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l F x * v x := by
  intro l
  induction l with
  | nil => intro _ _ v _ _; simp [iterPartial]
  | cons k l ih =>
    intro hex hcont v hv hvs
    obtain ⟨hex', hdiff⟩ := hex
    obtain ⟨hcont', hc⟩ := hcont
    have hFl : Continuous (iterPartial l F) := by
      cases l with
      | nil => exact hcont'
      | cons k' l' => exact hcont'.2
    -- one integration by parts in direction `k`
    have hvd : Differentiable ℝ v := hv.differentiable (by simp)
    obtain ⟨hv', hvs'⟩ := testFun_partialDeriv hv hvs k
    have hvc : Continuous v := hv.continuous
    have hv'c : Continuous (partialDeriv k v) := hv'.continuous
    have ibp : ∫ x, iterPartial l F x * partialDeriv k v x =
        - ∫ x, partialDeriv k (iterPartial l F) x * v x := by
      have := integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable
        (μ := (volume : Measure (Fin n → ℝ)))
        (f := iterPartial l F) (f' := partialDeriv k (iterPartial l F)) (g := v)
        (g' := partialDeriv k v) (v := Pi.single k 1) (B := ContinuousLinearMap.mul ℝ ℝ)
        ((hc.mul hvc).integrable_of_hasCompactSupport hvs.mul_left)
        ((hFl.mul hv'c).integrable_of_hasCompactSupport hvs'.mul_left)
        ((hFl.mul hvc).integrable_of_hasCompactSupport hvs.mul_left)
        (fun x _ => hasLineDerivAt_of_partial (hdiff x))
        (fun x _ => hasLineDerivAt_of_partial ((hvd _).comp _ (hasDerivAt_update x k (x k)).differentiableAt))
      simpa using this
    have ih' := ih hex' hcont' (partialDeriv k v) hv' hvs'
    have hrev : iterPartial (k :: l).reverse v = iterPartial l.reverse (partialDeriv k v) := by
      rw [List.reverse_cons, iterPartial_append]; rfl
    rw [hrev, ih', ibp]
    have : iterPartial (k :: l) F = partialDeriv k (iterPartial l F) := rfl
    rw [this, List.length_cons, pow_succ]
    ring

end GKLS
