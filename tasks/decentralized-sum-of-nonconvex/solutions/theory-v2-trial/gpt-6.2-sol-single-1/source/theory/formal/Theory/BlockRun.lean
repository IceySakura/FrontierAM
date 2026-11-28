import Theory.BlockGap

/-! Exact component-query accounting and the block support invariant. -/

open scoped BigOperators ENNReal
open Finset

noncomputable section

namespace ULB63

namespace Algorithm

variable {n m c d : ℕ} (A : Algorithm n m c) (I : Instance n m d)
  (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω)

theorem hist_length (s : ℕ) (i : Fin n) :
    (A.hist I W ω s i).length = s := by
  induction s with
  | zero => rfl
  | succ s ih =>
      rw [hist]
      split <;> simp [ih]

def queried (s : ℕ) (i : Fin n) : Option (Fin m) :=
  if s % microSteps = commSlot then none
  else match A.call d (A.info I W ω i) (A.hist I W ω s i) with
  | .grad ξ _ => some ξ
  | _ => none

def visits (b : Fin n × Fin m) : ℕ → ℕ
  | 0 => 0
  | s + 1 => visits b s + if A.queried I W ω s b.1 = some b.2 then 1 else 0

theorem visits_mono (b : Fin n × Fin m) {s s' : ℕ} (h : s ≤ s') :
    A.visits I W ω b s ≤ A.visits I W ω b s' :=
  (monotone_nat_of_le_succ fun k => by rw [visits]; omega) h

theorem sum_visits_eq (i : Fin n) (s : ℕ) :
    (∑ ξ : Fin m, A.visits I W ω (i, ξ) s) =
      (A.hist I W ω s i).countP Obs.isGrad := by
  classical
  induction s with
  | zero => simp [visits, hist]
  | succ s ih =>
    simp only [visits, sum_add_distrib, ih, hist, queried]
    split
    · simp [Obs.isGrad]
    · cases A.call d (A.info I W ω i) (A.hist I W ω s i) with
      | idle => simp [answer, Obs.isGrad]
      | grad ξ x => simp [answer, Obs.isGrad]
      | prox v η => simp [answer, Obs.isGrad]

private theorem count_budget_step {H : List (Obs d c)} {o : Obs d c}
    (hnew : o.isGrad = true → (currentIter H).any Obs.isGrad = false)
    (hprev : H.countP Obs.isGrad ≤ H.length / microSteps +
      (if (currentIter H).any Obs.isGrad then 1 else 0)) :
    (H ++ [o]).countP Obs.isGrad ≤ (H ++ [o]).length / microSteps +
      (if (currentIter (H ++ [o])).any Obs.isGrad then 1 else 0) := by
  have hb : (H ++ [o]).length = H.length + 1 := by simp
  rw [List.countP_append, List.countP_cons, List.countP_nil, zero_add, hb]
  by_cases h : H.length / microSteps = (H.length + 1) / microSteps
  · have hlen : microSteps * (H.length / microSteps) ≤ H.length := Nat.mul_div_le _ _
    have hs : currentIter (H ++ [o]) = currentIter H ++ [o] := by
      unfold currentIter
      rw [hb, ← h, List.drop_append_of_le_length hlen]
    rw [hs, List.any_append, List.any_cons, List.any_nil, Bool.or_false, ← h]
    cases ho : o.isGrad with
    | false =>
      simpa only [Bool.or_false, Bool.false_eq_true, ite_false, add_zero] using hprev
    | true =>
      have hcur := hnew ho
      simp [hcur] at hprev ⊢
      omega
  · have hquot : (H.length + 1) / microSteps = H.length / microSteps + 1 := by
      unfold microSteps at h ⊢
      omega
    rw [hquot]
    have hincr : (if o.isGrad then 1 else 0) +
        (if (currentIter H).any Obs.isGrad then (1 : ℕ) else 0) ≤ 1 := by
      by_cases ho : o.isGrad = true
      · simp [ho, hnew ho]
      · cases (currentIter H).any Obs.isGrad <;> simp [ho]
    omega

theorem hist_gradient_budget {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ) (s : ℕ) (i : Fin n) :
    (A.hist I W ω s i).countP Obs.isGrad ≤ s / microSteps +
      (if (currentIter (A.hist I W ω s i)).any Obs.isGrad then 1 else 0) := by
  induction s with
  | zero => simp [hist, currentIter]
  | succ s ih =>
    obtain ⟨o, he, hg⟩ : ∃ o : Obs d c, A.hist I W ω (s+1) i = A.hist I W ω s i ++ [o] ∧
        (o.isGrad = true → (currentIter (A.hist I W ω s i)).any Obs.isGrad = false) := by
      rw [hist]
      split
      · exact ⟨_, rfl, by simp [Obs.isGrad]⟩
      · refine ⟨_, rfl, ?_⟩
        intro hnew
        cases hcall : A.call d (A.info I W ω i) (A.hist I W ω s i) with
        | idle => simp [answer, Obs.isGrad, hcall] at hnew
        | prox v η => simp [answer, Obs.isGrad, hcall] at hnew
        | grad ξ x =>
          have h := hA.one_grad d (A.info I W ω i) (A.hist I W ω s i)
          rw [hcall] at h
          cases hb : (currentIter (A.hist I W ω s i)).any Obs.isGrad with
          | false => rfl
          | true =>
            have hf := h hb
            change true = false at hf
            contradiction
    have hb := count_budget_step hg (by rwa [A.hist_length I W ω s i])
    rw [← he, A.hist_length I W ω (s+1) i] at hb
    exact hb

theorem sum_visits_le {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ) (t : ℕ) :
    ∑ b : Fin n × Fin m, A.visits I W ω b (microSteps * t) ≤ n * t := by
  classical
  rw [Fintype.sum_prod_type]
  simp_rw [sum_visits_eq]
  have hdiv : microSteps * t / microSteps = t := by
    unfold microSteps
    omega
  calc
    _ ≤ ∑ _i : Fin n, t := by
      apply sum_le_sum
      intro i _
      have hb := A.hist_gradient_budget I W ω hA (microSteps * t) i
      have hempty : currentIter (A.hist I W ω (microSteps * t) i) = [] := by
        simp [currentIter, A.hist_length, hdiv]
      simpa [hdiv, hempty] using hb
    _ = _ := by simp

theorem visits_le_iter {L μ Δ ρ : ℝ} (hA : A.Admissible L μ Δ ρ) (b : Fin n × Fin m) (t : ℕ) :
    A.visits I W ω b (microSteps * t) ≤ t := by
  classical
  have hb := A.hist_gradient_budget I W ω hA (microSteps * t) b.1
  have hdiv : microSteps * t / microSteps = t := by unfold microSteps; omega
  have hempty : currentIter (A.hist I W ω (microSteps * t) b.1) = [] := by
    simp [currentIter, A.hist_length, hdiv]
  calc
    _ ≤ ∑ ξ : Fin m, A.visits I W ω (b.1, ξ) (microSteps * t) :=
      single_le_sum (f := fun ξ => A.visits I W ω (b.1, ξ) (microSteps * t))
        (fun _ _ => Nat.zero_le _) (mem_univ b.2)
    _ ≤ _ := by rw [sum_visits_eq]; simpa [hdiv, hempty] using hb

end Algorithm

namespace BlockGeometry

variable {n m c p : ℕ} {l μ q a L Δ ρ : ℝ} {A : Algorithm n m c}
  {W : Matrix (Fin n) (Fin n) ℝ} {ω : A.Ω}

private theorem outside_allowed {info : Info n (Dim n m p) A.Ω}
    {H : List (Obs (Dim n m p) c)} {b : Blocks n m} {j : Fin (p + 1)}
    (h0 : coord info.x0 b j = 0) (hH : ∀ o ∈ H, idx b j ∉ o.supp) :
    idx b j ∉ allowed info H := by
  simp only [allowed, Set.mem_union, Set.mem_iUnion, not_or, not_exists]
  exact ⟨fun h => h h0, hH⟩

theorem block_history_support (hA : A.Admissible L μ Δ ρ)
    (b : Blocks n m) (s : ℕ) :
    ∀ i (j : Fin (p + 1)), A.visits (blockInstance l μ q a n m p) W ω b s ≤ j.val →
      ∀ o ∈ A.hist (blockInstance l μ q a n m p) W ω s i, idx b j ∉ o.supp := by
  let I := blockInstance l μ q a n m p
  change ∀ i (j : Fin (p + 1)), A.visits I W ω b s ≤ j.val →
    ∀ o ∈ A.hist I W ω s i, idx b j ∉ o.supp
  induction s with
  | zero => simp [Algorithm.hist]
  | succ s ih =>
    have hallowed (b' : Blocks n m) (j : Fin (p + 1)) (i : Fin n)
        (H : ∀ o ∈ A.hist I W ω s i, idx b' j ∉ o.supp) :
        idx b' j ∉ allowed (A.info I W ω i) (A.hist I W ω s i) :=
      outside_allowed rfl H
    intro i j hj o ho
    have hj' : A.visits I W ω b s ≤ j.val :=
      (A.visits_mono I W ω b (Nat.le_succ s)).trans hj
    rw [Algorithm.hist] at ho
    split at ho
    · simp only [List.mem_append, List.mem_singleton] at ho
      rcases ho with ho | rfl
      · exact ih i j hj' _ ho
      · simp only [Obs.supp, Set.mem_iUnion, not_exists]
        intro k
        change ¬ coord (∑ j', W i j' •
          A.send (Dim n m p) (A.info I W ω j') (A.hist I W ω s j') k) b j ≠ 0
        rw [not_not, coord_sum]
        apply sum_eq_zero
        intro i' _
        have ha := hallowed b j i' (ih i' j hj')
        have hnz : ¬ coord (A.send (Dim n m p) (A.info I W ω i')
            (A.hist I W ω s i') k) b j ≠ 0 :=
          fun h => ha (hA.send_zero_respecting _ _ _ k h)
        simp [not_not.mp hnz]
    · rename_i hcomm
      simp only [List.mem_append, List.mem_singleton] at ho
      rcases ho with ho | rfl
      · exact ih i j hj' _ ho
      · have hquery := hA.call_zero_respecting (Dim n m p) (A.info I W ω i)
            (A.hist I W ω s i)
        have hx (j' : Fin (p + 1)) (hj'' : A.visits I W ω b s ≤ j'.val) :=
          hallowed b j' i (ih i j' hj'')
        cases hcall : A.call (Dim n m p) (A.info I W ω i) (A.hist I W ω s i) with
        | idle => simp [Algorithm.answer, Obs.supp]
        | prox v η =>
          rw [hcall] at hquery
          change ¬ coord ((1 + η * μ)⁻¹ • v) b j ≠ 0
          have hz : coord v b j = 0 := not_not.mp (fun h => hx j hj' (hquery h))
          simp [hz]
        | grad ξ x =>
          rw [hcall] at hquery
          change ¬ coord (gradient (fun y => ((n : ℝ) * m) * block l q a (i, ξ) y) x) b j ≠ 0
          by_cases hbe : (i, ξ) = b
          · have ht : A.queried I W ω s b.1 = some b.2 := by
              rw [← hbe]
              simp [Algorithm.queried, hcomm, hcall]
            have hj'' : A.visits I W ω b s < j.val := by
              rw [Algorithm.visits, ite_eq_left ht] at hj
              omega
            subst b
            exact not_not.mpr (scaled_block_gradient_zero _ _ _ _
              (fun _ j' hj''' => not_not.mp (fun h => hx j' hj''' (hquery h))) (Or.inr hj''))
          · exact not_not.mpr (scaled_block_gradient_zero (k := 0) _ _ _ _
              (fun he => (hbe he).elim) (Or.inl hbe))

theorem block_output_support (hA : A.Admissible L μ Δ ρ)
    (b : Blocks n m) (t : ℕ) (j : Fin (p + 1))
    (hj : A.visits (blockInstance l μ q a n m p) W ω b (microSteps * t) ≤ j.val) :
    coord (A.avgOutput (blockInstance l μ q a n m p) W ω t) b j = 0 := by
  classical
  rw [Algorithm.avgOutput, coord_smul, coord_sum]
  apply mul_eq_zero_of_right
  apply sum_eq_zero
  intro i _
  exact not_not.mp fun h => outside_allowed (A := A) rfl
    (block_history_support hA b _ i j hj) (hA.out_zero_respecting _ _ _ h)

end BlockGeometry

end ULB63
