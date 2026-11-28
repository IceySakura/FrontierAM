import Theory.Defs
import Theory.Tail

/-!
# Iterated partial derivatives of the two model functions

* ridge functions `y ↦ K (∑ i, y i)` (the preintegrated function will be of this form);
* `x ↦ G (x j) + ∑ i, x i` (the counterexample `φ`).
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory
open scoped ContDiff

namespace GKLS

variable {n : ℕ}

lemma sum_update (y : Fin n → ℝ) (k : Fin n) (t : ℝ) :
    ∑ i, Function.update y k t i = t + (∑ i, y i - y k) := by
  rw [Finset.sum_update_of_mem (Finset.mem_univ k),
    Finset.sum_eq_add_sum_sdiff_singleton_of_mem (Finset.mem_univ k) y]
  ring

lemma IterPartialExists.of_append {l₁ l₂ : List (Fin n)} {F : (Fin n → ℝ) → ℝ}
    (h : IterPartialExists (l₁ ++ l₂) F) : IterPartialExists l₂ F := by
  induction l₁ with
  | nil => simpa using h
  | cons a l ih => exact ih h.1

/-! ### Ridge functions -/

lemma partial_ridge {K : ℝ → ℝ} (hK : Differentiable ℝ K) (k : Fin n) (y : Fin n → ℝ) :
    DifferentiableAt ℝ (fun t => K (∑ i, Function.update y k t i)) (y k) ∧
      partialDeriv k (fun z => K (∑ i, z i)) y = deriv K (∑ i, y i) := by
  have hfun : (fun t => K (∑ i, Function.update y k t i)) =
      fun t => K (t + (∑ i, y i - y k)) := by
    funext t; rw [sum_update]
  have hpt : y k + (∑ i, y i - y k) = ∑ i, y i := by ring
  refine ⟨?_, ?_⟩
  · rw [hfun]
    exact (hK _).comp _ (differentiableAt_id.add_const _)
  · simp only [partialDeriv]
    rw [hfun, deriv_comp_add_const, hpt]

/-- Iterated partial derivatives of a ridge function `y ↦ K (∑ i, y i)` with `K ∈ C^N`, along any
list of at most `N` directions. -/
lemma ridge_iterPartial {N : ℕ} {K : ℝ → ℝ} (hK : ContDiff ℝ N K) :
    ∀ l : List (Fin n), l.length ≤ N →
      IterPartialExists l (fun z => K (∑ i, z i)) ∧
        iterPartial l (fun z => K (∑ i, z i)) = fun z => iteratedDeriv l.length K (∑ i, z i) := by
  intro l
  induction l with
  | nil => intro _; exact ⟨trivial, by funext z; simp [iterPartial]⟩
  | cons k l ih =>
    intro hlen
    simp only [List.length_cons] at hlen
    obtain ⟨hex, heq⟩ := ih (by omega)
    have hdiff : Differentiable ℝ (iteratedDeriv l.length K) :=
      hK.differentiable_iteratedDeriv _ (by exact_mod_cast (show l.length < N by omega))
    refine ⟨⟨hex, fun x => ?_⟩, ?_⟩
    · rw [heq]; exact (partial_ridge hdiff k x).1
    · have : iterPartial (k :: l) (fun z => K (∑ i, z i)) =
          partialDeriv k (iterPartial l (fun z => K (∑ i, z i))) := rfl
      rw [this, heq]
      funext z
      rw [(partial_ridge hdiff k z).2, List.length_cons, iteratedDeriv_succ]

/-- If moreover `K^{(N)}` is not differentiable at `0`, no list of more than `N` directions gives
existing iterated partial derivatives of the ridge function. -/
lemma ridge_not_exists {N : ℕ} {K : ℝ → ℝ} (hK : ContDiff ℝ N K)
    (hbad : ¬ DifferentiableAt ℝ (iteratedDeriv N K) 0) :
    ∀ l : List (Fin n), N < l.length → ¬ IterPartialExists l (fun z => K (∑ i, z i)) := by
  intro l hl hex
  set i := l.length - (N + 1) with hi
  have hsplit : l = l.take i ++ l.drop i := (List.take_append_drop i l).symm
  rw [hsplit] at hex
  have hex2 := hex.of_append
  have hlen : (l.drop i).length = N + 1 := by rw [List.length_drop]; omega
  obtain ⟨k, l', hkl⟩ : ∃ k l', l.drop i = k :: l' := by
    cases h : l.drop i with
    | nil => rw [h] at hlen; simp at hlen
    | cons k l' => exact ⟨k, l', rfl⟩
  rw [hkl] at hex2 hlen
  simp only [List.length_cons, add_left_inj] at hlen
  obtain ⟨_, hall⟩ := hex2
  have h0 := hall 0
  obtain ⟨-, heq⟩ := ridge_iterPartial hK l' (le_of_eq hlen)
  rw [heq, hlen] at h0
  have hfun : (fun t => iteratedDeriv N K (∑ i, Function.update (0 : Fin n → ℝ) k t i)) =
      fun t => iteratedDeriv N K t := by
    funext t; rw [sum_update]; simp
  rw [hfun] at h0
  exact hbad (by simpa using h0)

/-! ### The functions `x ↦ c₁ ∑ x_i + c₂ G_a(x_j) + c₃` -/

lemma partial_state (j k : Fin n) (c₁ c₂ c₃ : ℝ) {Ga : ℝ → ℝ}
    (hG : k = j → c₂ ≠ 0 → Differentiable ℝ Ga) (x : Fin n → ℝ) :
    DifferentiableAt ℝ
        (fun t => c₁ * ∑ i, Function.update x k t i + c₂ * Ga (Function.update x k t j) + c₃) (x k) ∧
      partialDeriv k (fun z => c₁ * ∑ i, z i + c₂ * Ga (z j) + c₃) x =
        c₁ + (if k = j then c₂ else 0) * deriv Ga (x j) := by
  by_cases hkj : k = j
  · subst hkj
    have hfun : (fun t => c₁ * ∑ i, Function.update x k t i + c₂ * Ga (Function.update x k t k) + c₃)
        = fun t => c₁ * (t + (∑ i, x i - x k)) + c₂ * Ga t + c₃ := by
      funext t; rw [sum_update]; simp
    by_cases hc : c₂ = 0
    · subst hc
      have hfun' : (fun t => c₁ * (t + (∑ i, x i - x k)) + 0 * Ga t + c₃) =
          fun t => c₁ * (t + (∑ i, x i - x k)) + c₃ := by funext t; ring
      refine ⟨?_, ?_⟩
      · rw [hfun, hfun']; fun_prop
      · simp only [partialDeriv]
        rw [hfun, hfun']
        have h := ((((hasDerivAt_id' (x k)).add_const (∑ i, x i - x k)).const_mul c₁).add_const c₃)
        refine h.deriv.trans ?_; simp
    · have hGd := hG rfl hc
      refine ⟨?_, ?_⟩
      · rw [hfun]
        exact (((differentiableAt_id.add_const _).const_mul _).add ((hGd _).const_mul _)).add_const _
      · simp only [partialDeriv]
        rw [hfun]
        have h := (((((hasDerivAt_id' (x k)).add_const (∑ i, x i - x k)).const_mul c₁).add
          ((hGd (x k)).hasDerivAt.const_mul c₂)).add_const c₃)
        refine h.deriv.trans ?_; simp
  · have hfun : (fun t => c₁ * ∑ i, Function.update x k t i + c₂ * Ga (Function.update x k t j) + c₃)
        = fun t => c₁ * (t + (∑ i, x i - x k)) + (c₂ * Ga (x j) + c₃) := by
      funext t; rw [sum_update, Function.update_of_ne (Ne.symm hkj)]; ring
    refine ⟨?_, ?_⟩
    · rw [hfun]; fun_prop
    · simp only [partialDeriv]
      rw [hfun]
      have h := ((((hasDerivAt_id' (x k)).add_const (∑ i, x i - x k)).const_mul c₁).add_const
        (c₂ * Ga (x j) + c₃))
      refine h.deriv.trans ?_; simp [hkj]

/-- Iterated partial derivatives of `φ(x) = G(x_j) + ∑_i x_i` for `G ∈ C^m`, along any list with at
most `m` differentiations in the direction `j`. -/
lemma phi_iterPartial {m : ℕ} {G : ℝ → ℝ} (hG : ContDiff ℝ m G) (j : Fin n) :
    ∀ l : List (Fin n), l.count j ≤ m →
      IterPartialExists l (fun x => G (x j) + ∑ i, x i) ∧
        iterPartial l (fun x => G (x j) + ∑ i, x i) = fun x =>
          (if l = [] then 1 else 0) * ∑ i, x i +
            (if ∀ i ∈ l, i = j then 1 else 0) * iteratedDeriv l.length G (x j) +
            (if l.length = 1 then 1 else 0) := by
  intro l
  induction l with
  | nil =>
    intro _
    exact ⟨trivial, by funext x; simp [iterPartial]; ring⟩
  | cons k l ih =>
    intro hcount
    have hcl : l.count j ≤ m := le_trans List.count_le_count_cons hcount
    obtain ⟨hex, heq⟩ := ih hcl
    set c₁ : ℝ := if l = [] then 1 else 0
    set c₂ : ℝ := if ∀ i ∈ l, i = j then 1 else 0 with hc₂
    set c₃ : ℝ := if l.length = 1 then 1 else 0
    have hdiffG : k = j → c₂ ≠ 0 → Differentiable ℝ (iteratedDeriv l.length G) := by
      intro hkj hc
      have hall : ∀ i ∈ l, i = j := by
        by_contra h; exact hc (by simp [hc₂, h])
      have hcnt : l.count j = l.length := List.count_eq_length.mpr (fun i hi => (hall i hi).symm)
      have : (k :: l).count j = l.length + 1 := by
        rw [hkj, List.count_cons_self, hcnt]
      exact hG.differentiable_iteratedDeriv _ (by exact_mod_cast (show l.length < m by omega))
    have hit : iterPartial (k :: l) (fun x => G (x j) + ∑ i, x i) =
        partialDeriv k (iterPartial l (fun x => G (x j) + ∑ i, x i)) := rfl
    refine ⟨⟨hex, fun x => ?_⟩, ?_⟩
    · rw [heq]; exact (partial_state j k c₁ c₂ c₃ hdiffG x).1
    · rw [hit, heq]
      funext x
      rw [(partial_state j k c₁ c₂ c₃ hdiffG x).2, ← iteratedDeriv_succ]
      simp only [List.length_cons, List.mem_cons, forall_eq_or_imp, reduceCtorEq, ↓reduceIte,
        zero_mul, zero_add, add_eq_right, List.length_eq_zero_iff, c₁, c₂]
      by_cases hkj : k = j <;> simp [hkj]; ring

end GKLS
