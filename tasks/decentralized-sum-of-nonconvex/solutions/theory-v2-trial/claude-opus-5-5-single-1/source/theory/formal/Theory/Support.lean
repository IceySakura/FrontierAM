import Theory.Model

/-!
# Support propagation for zero-respecting algorithms

For an instance whose oracles respect a monotone family of coordinate sets `B k`
(`k : Fin m → ℕ` a progress vector), in the sense that a gradient query of node `i₀` to component `ξ`
at a point supported in `B k` returns a vector supported in `B (k + eξ)`, all other oracle answers
stay in `B k`, and `x₀` is supported in `B 0`, every admissible algorithm keeps all its iterates,
messages and outputs inside `B k` with `∑ k ≤ t` after `t` iterations. This holds for every seed,
because node `i₀` makes at most one gradient query per iteration.
-/

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace ULB63
namespace Support

open Algorithm

variable {n m c d : ℕ}

/-- Number of gradient observations in a history. -/
def gradCount (H : List (Obs d c)) : ℕ := H.countP (fun o => o.isGrad)

lemma hist_succ (A : Algorithm n m c) (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (ω : A.Ω) (s : ℕ) (i : Fin n) :
    A.hist I W ω (s + 1) i =
      if s % microSteps = commSlot then
        A.hist I W ω s i ++ [.recv fun k => ∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) k]
      else
        A.hist I W ω s i ++ [answer I i (A.call d (A.info I W ω i) (A.hist I W ω s i))] := rfl

lemma hist_length (A : Algorithm n m c) (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ)
    (ω : A.Ω) (s : ℕ) (i : Fin n) : (A.hist I W ω s i).length = s := by
  induction s generalizing i with
  | zero => rfl
  | succ s ih =>
    rw [hist_succ]
    split_ifs <;> simp [ih]

lemma answer_isGrad (I : Instance n m d) (i : Fin n) (cl : Call m d) :
    (answer (c := c) I i cl).isGrad = cl.isGrad := by
  cases cl <;> rfl

lemma currentIter_append_of_ne {H : List (Obs d c)} {s : ℕ} (hlen : H.length = s)
    (hs : (s + 1) % microSteps ≠ 0) (o : Obs d c) :
    currentIter (H ++ [o]) = currentIter H ++ [o] := by
  unfold currentIter
  simp only [List.length_append, List.length_singleton, hlen]
  have h1 : (s + 1) / microSteps = s / microSteps := by
    simp only [microSteps] at hs ⊢; omega
  rw [h1, List.drop_append_of_le_length]
  rw [hlen]; simp only [microSteps]; omega

lemma currentIter_append_of_eq {H : List (Obs d c)} {s : ℕ} (hlen : H.length = s)
    (hs : (s + 1) % microSteps = 0) (o : Obs d c) :
    currentIter (H ++ [o]) = [] := by
  unfold currentIter
  simp only [List.length_append, List.length_singleton, hlen]
  apply List.drop_eq_nil_of_le
  simp only [List.length_append, List.length_singleton, hlen]
  simp only [microSteps] at hs ⊢; omega

lemma currentIter_mul (H : List (Obs d c)) (t : ℕ) (hlen : H.length = microSteps * t) :
    currentIter H = [] := by
  unfold currentIter
  apply List.drop_eq_nil_of_le
  rw [hlen]
  simp only [microSteps]; omega

/-- At most one gradient query per iteration: after `t` iterations, at most `t` gradient
observations. -/
theorem gradCount_le (A : Algorithm n m c) {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω) (i : Fin n) (t : ℕ) :
    gradCount (A.hist I W ω (microSteps * t) i) ≤ t := by
  have key : ∀ s, gradCount (A.hist I W ω s i) ≤
      s / microSteps + (if (currentIter (A.hist I W ω s i)).any Obs.isGrad then 1 else 0) := by
    intro s
    induction s with
    | zero => simp [gradCount, Algorithm.hist]
    | succ s ih =>
      set H := A.hist I W ω s i with hH
      have hlen : H.length = s := hist_length A I W ω s i
      -- the appended observation and whether it is a gradient
      obtain ⟨o, ho, hog⟩ : ∃ o : Obs d c, A.hist I W ω (s + 1) i = H ++ [o] ∧
          (o.isGrad = true → (currentIter H).any Obs.isGrad = false) := by
        rw [hist_succ]
        split_ifs with hc
        · exact ⟨_, rfl, fun h => by simp [Obs.isGrad] at h⟩
        · refine ⟨_, rfl, fun h => ?_⟩
          rw [answer_isGrad] at h
          by_contra hb
          have := hA.one_grad d (A.info I W ω i) H (by simpa using hb)
          rw [h] at this
          exact absurd this (by simp)
      rw [ho]
      have hcount : gradCount (H ++ [o]) = gradCount H + (if o.isGrad then 1 else 0) := by
        simp [gradCount, List.countP_append]
      rw [hcount]
      by_cases hs : (s + 1) % microSteps = 0
      · rw [currentIter_append_of_eq hlen hs]
        have hdiv : (s + 1) / microSteps = s / microSteps + 1 := by
          simp only [microSteps] at hs ⊢; omega
        rw [hdiv]
        simp only [List.any_nil, Bool.false_eq_true, if_false, add_zero]
        by_cases hg : o.isGrad = true
        · have h0 := hog hg
          rw [h0] at ih
          simp only [Bool.false_eq_true, if_false, add_zero] at ih
          rw [if_pos hg]
          omega
        · rw [if_neg hg]
          split_ifs at ih <;> omega
      · rw [currentIter_append_of_ne hlen hs]
        have hdiv : (s + 1) / microSteps = s / microSteps := by
          simp only [microSteps] at hs ⊢; omega
        rw [hdiv]
        by_cases hg : o.isGrad = true
        · have h0 := hog hg
          rw [h0] at ih
          simp only [Bool.false_eq_true, if_false, add_zero] at ih
          rw [if_pos hg]
          have : (currentIter H ++ [o]).any Obs.isGrad = true := by simp [List.any_append, hg]
          rw [if_pos this]
          omega
        · have hg' : o.isGrad = false := by simpa using hg
          rw [if_neg hg]
          have : (currentIter H ++ [o]).any Obs.isGrad = (currentIter H).any Obs.isGrad := by
            simp [List.any_append, hg']
          rw [this]
          omega
  have := key (microSteps * t)
  rw [currentIter_mul _ t (hist_length A I W ω _ i)] at this
  simpa [microSteps] using this

/-! ## Supports -/

lemma supp_zero : supp (0 : E d) = ∅ := by
  ext k; simp [supp]

lemma supp_sum_smul_subset {ι : Type*} (s : Finset ι) (a : ι → ℝ) (v : ι → E d) {S : Set (Fin d)}
    (hv : ∀ j ∈ s, supp (v j) ⊆ S) : supp (∑ j ∈ s, a j • v j) ⊆ S := by
  intro k hk
  by_contra hkS
  apply hk
  have : ∀ j ∈ s, v j k = 0 := fun j hj => by
    by_contra h
    exact hkS (hv j hj h)
  simp only [WithLp.ofLp_sum, WithLp.ofLp_smul, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  exact Finset.sum_eq_zero fun j hj => by rw [this j hj, mul_zero]

lemma allowed_append {Ω : Type} (info : Info n d Ω) (H : List (Obs d c)) (o : Obs d c) :
    allowed info (H ++ [o]) = allowed info H ∪ o.supp := by
  unfold allowed
  ext k
  simp only [Set.mem_union, Set.mem_iUnion, List.mem_append, List.mem_singleton, exists_prop]
  constructor
  · rintro (h | ⟨o', h1 | h1, h2⟩)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr ⟨o', h1, h2⟩)
    · subst h1; exact Or.inr h2
  · rintro ((h | ⟨o', h1, h2⟩) | h)
    · exact Or.inl h
    · exact Or.inr ⟨o', Or.inl h1, h2⟩
    · exact Or.inr ⟨o, Or.inr rfl, h⟩

lemma allowed_nil {Ω : Type} (info : Info n d Ω) : allowed info ([] : List (Obs d c)) = supp info.x0 := by
  unfold allowed; simp

/-- The progress vector after a call: a gradient query to component `ξ` advances block `ξ`. -/
def bump (k : Fin m → ℕ) : Call m d → (Fin m → ℕ)
  | .grad ξ _ => k + Pi.single ξ 1
  | _ => k

lemma le_bump (k : Fin m → ℕ) (cl : Call m d) : k ≤ bump k cl := by
  cases cl with
  | grad ξ x => intro j; simp [bump]
  | _ => exact le_rfl

lemma sum_bump (k : Fin m → ℕ) (cl : Call m d) :
    ∑ ξ, bump k cl ξ = ∑ ξ, k ξ + (if cl.isGrad then 1 else 0) := by
  cases cl with
  | grad ξ x => simp [bump, Finset.sum_add_distrib, Call.isGrad]
  | idle => simp [bump, Call.isGrad]
  | prox v η => simp [bump, Call.isGrad]

/-- The support invariant. -/
theorem support_invariant (A : Algorithm n m c) {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (i₀ : Fin n)
    (B : (Fin m → ℕ) → Set (Fin d)) (hB : ∀ k k', k ≤ k' → B k ⊆ B k')
    (hx0 : supp I.x0 ⊆ B 0)
    (hgrad₀ : ∀ ξ x k, supp x ⊆ B k → supp (gradient (I.F i₀ ξ) x) ⊆ B (k + Pi.single ξ 1))
    (hgrad : ∀ i, i ≠ i₀ → ∀ ξ x k, supp x ⊆ B k → supp (gradient (I.F i ξ) x) ⊆ B k)
    (hprox : ∀ v η k, supp v ⊆ B k → supp (I.prox v η) ⊆ B k) (ω : A.Ω) :
    ∀ s, ∃ k : Fin m → ℕ, ∑ ξ, k ξ ≤ gradCount (A.hist I W ω s i₀) ∧
      ∀ i, allowed (A.info I W ω i) (A.hist I W ω s i) ⊆ B k := by
  intro s
  induction s with
  | zero =>
    refine ⟨0, by simp, fun i => ?_⟩
    simp only [Algorithm.hist]
    rw [allowed_nil]
    exact hx0
  | succ s ih =>
    obtain ⟨k, hk, hsub⟩ := ih
    by_cases hc : s % microSteps = commSlot
    · -- communication round
      refine ⟨k, ?_, fun i => ?_⟩
      · rw [hist_succ, if_pos hc]
        simpa [gradCount, List.countP_append, Obs.isGrad] using hk
      · rw [hist_succ, if_pos hc, allowed_append]
        refine Set.union_subset (hsub i) ?_
        simp only [Obs.supp]
        refine Set.iUnion_subset fun ch => ?_
        refine supp_sum_smul_subset _ _ _ fun j _ => ?_
        exact (hA.send_zero_respecting d _ _ ch).trans (hsub j)
    · -- oracle slot
      have hcall_supp : ∀ i, (A.call d (A.info I W ω i) (A.hist I W ω s i)).supp ⊆ B k :=
        fun i => (hA.call_zero_respecting d _ _).trans (hsub i)
      refine ⟨bump k (A.call d (A.info I W ω i₀) (A.hist I W ω s i₀)), ?_, fun i => ?_⟩
      · rw [hist_succ, if_neg hc, sum_bump]
        unfold gradCount at hk ⊢
        rw [List.countP_append]
        have e : ∀ cl : Call m d, List.countP (fun o : Obs d c => o.isGrad) [answer I i₀ cl] =
            if cl.isGrad then 1 else 0 := by
          intro cl; cases cl <;> simp [answer, Obs.isGrad, Call.isGrad]
        rw [e]
        exact Nat.add_le_add_right hk _
      · rw [hist_succ, if_neg hc, allowed_append]
        refine Set.union_subset ((hsub i).trans (hB _ _ (le_bump _ _))) ?_
        have hci := hcall_supp i
        by_cases hi : i = i₀
        · subst hi
          generalize A.call d (A.info I W ω i) (A.hist I W ω s i) = cl at hci ⊢
          cases cl with
          | idle => simp [answer, Obs.supp]
          | grad ξ x =>
            simp only [answer, Obs.supp, bump]
            exact hgrad₀ ξ x k hci
          | prox v η =>
            simp only [answer, Obs.supp]
            exact (hprox v η k hci).trans (hB _ _ (le_bump _ _))
        · refine Set.Subset.trans ?_ (hB _ _ (le_bump k _))
          generalize A.call d (A.info I W ω i) (A.hist I W ω s i) = cl at hci ⊢
          cases cl with
          | idle => simp [answer, Obs.supp]
          | grad ξ x =>
            simp only [answer, Obs.supp]
            exact hgrad i hi ξ x k hci
          | prox v η =>
            simp only [answer, Obs.supp]
            exact hprox v η k hci

/-- After `t` iterations the averaged output is supported in `B k` for some `k` with `∑ k ≤ t`. -/
theorem avgOutput_supp (A : Algorithm n m c) {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (i₀ : Fin n)
    (B : (Fin m → ℕ) → Set (Fin d)) (hB : ∀ k k', k ≤ k' → B k ⊆ B k')
    (hx0 : supp I.x0 ⊆ B 0)
    (hgrad₀ : ∀ ξ x k, supp x ⊆ B k → supp (gradient (I.F i₀ ξ) x) ⊆ B (k + Pi.single ξ 1))
    (hgrad : ∀ i, i ≠ i₀ → ∀ ξ x k, supp x ⊆ B k → supp (gradient (I.F i ξ) x) ⊆ B k)
    (hprox : ∀ v η k, supp v ⊆ B k → supp (I.prox v η) ⊆ B k) (ω : A.Ω) (t : ℕ) :
    ∃ k : Fin m → ℕ, ∑ ξ, k ξ ≤ t ∧ supp (A.avgOutput I W ω t) ⊆ B k := by
  obtain ⟨k, hk, hsub⟩ := support_invariant A hA I W i₀ B hB hx0 hgrad₀ hgrad hprox ω
    (microSteps * t)
  refine ⟨k, hk.trans (gradCount_le A hA I W ω i₀ t), ?_⟩
  unfold Algorithm.avgOutput
  refine (supp_smul_subset' _ _).trans ?_
  have := supp_sum_smul_subset (Finset.univ) (fun _ => (1 : ℝ)) (fun i => A.output I W ω t i)
    (S := B k) (fun i _ => (hA.out_zero_respecting d _ _).trans (hsub i))
  simpa using this
where
  supp_smul_subset' (a : ℝ) (x : E d) : supp (a • x) ⊆ supp x := by
    intro k hk h
    apply hk
    simp [h]

end Support
end ULB63
