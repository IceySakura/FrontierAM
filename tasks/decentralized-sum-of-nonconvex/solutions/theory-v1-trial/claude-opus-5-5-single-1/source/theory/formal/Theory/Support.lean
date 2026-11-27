import Theory.Model

/-!
# Support propagation along a run

For an instance whose oracles move a vector supported in the first `j` coordinates to one
supported in the first `j + r i j` coordinates (node `i`), the knowledge of every node of a
zero-respecting algorithm stays inside a prefix whose length `lev s i` is computed by a
deterministic recursion over micro-steps.
-/

open scoped ENNReal

noncomputable section

namespace ULB63

variable {n m c d : ℕ}

/-- The first `j` coordinates. -/
def Pre (d j : ℕ) : Set (Fin d) := {p | (p : ℕ) < j}

lemma Pre_mono {j k : ℕ} (h : j ≤ k) : Pre d j ⊆ Pre d k := fun p hp => by
  simp only [Pre, Set.mem_setOf_eq] at *; omega

lemma supp_subset_Pre {x : E d} {j : ℕ} : supp x ⊆ Pre d j ↔ ∀ p : Fin d, j ≤ p → x p = 0 := by
  constructor
  · intro h p hp
    by_contra hne
    have := h hne
    simp only [Pre, Set.mem_setOf_eq] at this
    omega
  · intro h p hp
    simp only [Pre, Set.mem_setOf_eq]
    by_contra hge
    exact hp (h p (by omega))

lemma allowed_append {Ω : Type} (info : Info n d Ω) (H : List (Obs d c)) (o : Obs d c) :
    allowed info (H ++ [o]) = allowed info H ∪ o.supp := by
  ext p
  simp only [allowed, Set.mem_union, Set.mem_iUnion, List.mem_append, List.mem_singleton,
    exists_prop]
  constructor
  · rintro (h | ⟨o', ho' | rfl, hp⟩)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr ⟨o', ho', hp⟩)
    · exact Or.inr hp
  · rintro ((h | ⟨o', ho', hp⟩) | hp)
    · exact Or.inl h
    · exact Or.inr ⟨o', Or.inl ho', hp⟩
    · exact Or.inr ⟨o, Or.inr rfl, hp⟩

lemma supp_sum_smul (a : Fin n → ℝ) (z : Fin n → E d) (p : Fin d)
    (hp : p ∈ supp (∑ j, a j • z j)) : ∃ j, a j ≠ 0 ∧ p ∈ supp (z j) := by
  simp only [supp, Set.mem_setOf_eq] at hp
  rw [WithLp.ofLp_sum, Finset.sum_apply] at hp
  obtain ⟨j, -, hj⟩ := Finset.exists_ne_zero_of_sum_ne_zero hp
  simp only [PiLp.smul_apply, smul_eq_mul] at hj
  exact ⟨j, left_ne_zero_of_mul hj, right_ne_zero_of_mul hj⟩

/-- Prefix lengths: a communication round takes the maximum over the in-neighbours, an oracle
slot raises node `i`'s prefix from `j` to `j + r i j`. -/
def lev (W : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℕ → ℕ) : ℕ → Fin n → ℕ
  | 0 => fun _ => 0
  | s + 1 => fun i =>
    if s % microSteps = commSlot then
      max (lev W r s i) ((Finset.univ.filter (fun j => W i j ≠ 0)).sup (lev W r s))
    else lev W r s i + r i (lev W r s i)

lemma lev_succ_ge (W : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℕ → ℕ) (s : ℕ) (i : Fin n) :
    lev W r s i ≤ lev W r (s + 1) i := by
  simp only [lev]
  split_ifs
  · exact le_max_left _ _
  · omega

/-- The support invariant. -/
theorem allowed_subset_lev (A : Algorithm n m c)
    (hcall : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)), (A.call d info H).supp ⊆ allowed info H)
    (hsend : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)) k,
      supp (A.send d info H k) ⊆ allowed info H)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℕ → ℕ)
    (hx0 : I.x0 = 0)
    (hgrad : ∀ i ξ (x : E d) j, supp x ⊆ Pre d j → supp (gradient (I.F i ξ) x) ⊆ Pre d (j + r i j))
    (hprox : ∀ (v : E d) η j, supp v ⊆ Pre d j → supp (I.prox v η) ⊆ Pre d j)
    (ω : A.Ω) : ∀ s i, allowed (A.info I W ω i) (A.hist I W ω s i) ⊆ Pre d (lev W r s i) := by
  intro s
  induction s with
  | zero =>
    intro i
    simp only [Algorithm.hist, allowed, Algorithm.info, hx0, List.not_mem_nil, Set.iUnion_of_empty,
      Set.iUnion_empty, Set.union_empty]
    intro p hp
    simp [supp] at hp
  | succ s ih =>
    intro i
    have hmono := Pre_mono (d := d) (lev_succ_ge W r s i)
    by_cases hs : s % microSteps = commSlot
    · have hh : A.hist I W ω (s + 1) i = A.hist I W ω s i ++
          [.recv fun k => ∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) k] := by
        simp only [Algorithm.hist, if_pos hs]
      rw [hh, allowed_append]
      refine Set.union_subset ((ih i).trans hmono) ?_
      intro p hp
      simp only [Obs.supp, Set.mem_iUnion] at hp
      obtain ⟨k, hk⟩ := hp
      obtain ⟨j, hj, hpj⟩ := supp_sum_smul _ _ p hk
      have h1 := (ih j) (hsend d _ _ k hpj)
      have h2 : lev W r s j ≤ lev W r (s + 1) i := by
        simp only [lev, if_pos hs]
        refine le_trans ?_ (le_max_right _ _)
        exact Finset.le_sup (f := lev W r s) (Finset.mem_filter.2 ⟨Finset.mem_univ _, hj⟩)
      exact Pre_mono h2 h1
    · have hh : A.hist I W ω (s + 1) i = A.hist I W ω s i ++
          [Algorithm.answer I i (A.call d (A.info I W ω i) (A.hist I W ω s i))] := by
        simp only [Algorithm.hist, if_neg hs]
      rw [hh, allowed_append]
      refine Set.union_subset ((ih i).trans hmono) ?_
      have hc := hcall d (A.info I W ω i) (A.hist I W ω s i)
      have hlev : lev W r (s + 1) i = lev W r s i + r i (lev W r s i) := by
        simp only [lev, if_neg hs]
      generalize A.call d (A.info I W ω i) (A.hist I W ω s i) = cl at hc ⊢
      cases cl with
      | idle => simp [Algorithm.answer, Obs.supp]
      | grad ξ x =>
        simp only [Algorithm.answer, Obs.supp]
        simp only [Call.supp] at hc
        rw [hlev]
        exact hgrad i ξ x _ (hc.trans (ih i))
      | prox v η =>
        simp only [Algorithm.answer, Obs.supp]
        simp only [Call.supp] at hc
        exact (hprox v η _ (hc.trans (ih i))).trans hmono

/-- The averaged output lies in the prefix of length `K` if every node's prefix does. -/
theorem supp_avgOutput (A : Algorithm n m c)
    (hout : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)), supp (A.out d info H) ⊆ allowed info H)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω) (t K : ℕ)
    (h : ∀ i, allowed (A.info I W ω i) (A.hist I W ω (microSteps * t) i) ⊆ Pre d K) :
    supp (A.avgOutput I W ω t) ⊆ Pre d K := by
  intro p hp
  simp only [Algorithm.avgOutput, supp, Set.mem_setOf_eq, PiLp.smul_apply, smul_eq_mul] at hp
  have hp' := right_ne_zero_of_mul hp
  rw [WithLp.ofLp_sum, Finset.sum_apply] at hp'
  obtain ⟨i, -, hi⟩ := Finset.exists_ne_zero_of_sum_ne_zero hp'
  exact h i (hout d _ _ hi)

/-! ## Level bound from a potential on the communication graph -/

/-- The earliest number of communication rounds after which level `k` can be present at a node
with potential `e` (`e = 0` on the nodes raising even levels, `e = D` on those raising odd ones). -/
def arr (D k e : ℕ) : ℕ := if k = 0 then 0 else (k - 1) * D + (if k % 2 = 1 then e else D - e)

lemma arr_neighbor {D k e e' : ℕ} (he : e ≤ e' + 1) (he' : e' ≤ e + 1) (hD : e ≤ D) (hD' : e' ≤ D) :
    arr D k e ≤ arr D k e' + 1 := by
  unfold arr
  split_ifs <;> omega

lemma arr_raise_even {D k : ℕ} (hk : k % 2 = 0) : arr D (k + 1) 0 = arr D k 0 := by
  unfold arr
  rw [if_neg (by omega), if_pos (by omega)]
  by_cases hk0 : k = 0
  · subst hk0; simp
  · rw [if_neg hk0, if_neg (by omega)]
    obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
    simp [Nat.succ_mul]

lemma arr_raise_odd {D k : ℕ} (hk : k % 2 = 1) : arr D (k + 1) D = arr D k D := by
  unfold arr
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos hk]
  obtain ⟨k', rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
  simp [Nat.succ_mul]

theorem arr_lev_le (W : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℕ → ℕ) (δ : Fin n → ℕ) (D : ℕ)
    (hδD : ∀ i, δ i ≤ D) (hδ : ∀ i j, W i j ≠ 0 → δ i ≤ δ j + 1 ∧ δ j ≤ δ i + 1)
    (hr : ∀ i j, r i j = 0 ∨ (r i j = 1 ∧ ((j % 2 = 0 ∧ δ i = 0) ∨ (j % 2 = 1 ∧ δ i = D)))) :
    ∀ s i, arr D (lev W r s i) (δ i) ≤ (s + 2) / 5 := by
  intro s
  induction s with
  | zero => intro i; simp [lev, arr]
  | succ s ih =>
    intro i
    by_cases hs : s % microSteps = commSlot
    · have hs' : s % 5 = 2 := by simpa [microSteps, commSlot] using hs
      have hstep : (s + 1 + 2) / 5 = (s + 2) / 5 + 1 := by omega
      have hl : lev W r (s + 1) i =
          max (lev W r s i) ((Finset.univ.filter (fun j => W i j ≠ 0)).sup (lev W r s)) := by
        simp only [lev, if_pos hs]
      rw [hl, hstep]
      rcases le_total ((Finset.univ.filter (fun j => W i j ≠ 0)).sup (lev W r s)) (lev W r s i)
        with hle | hle
      · rw [max_eq_left hle]
        have := ih i; omega
      · rw [max_eq_right hle]
        rcases Finset.eq_empty_or_nonempty (Finset.univ.filter (fun j => W i j ≠ 0)) with
          hemp | hne
        · rw [hemp, Finset.sup_empty]
          simp [arr]
        · obtain ⟨j, hj, hjeq⟩ := Finset.exists_mem_eq_sup _ hne (lev W r s)
          rw [hjeq]
          have hWij := (Finset.mem_filter.1 hj).2
          have h1 := arr_neighbor (k := lev W r s j) (hδ i j hWij).1 (hδ i j hWij).2 (hδD i)
            (hδD j)
          have h2 := ih j
          omega
    · have hs' : ¬ s % 5 = 2 := by simpa [microSteps, commSlot] using hs
      have hstep : (s + 1 + 2) / 5 = (s + 2) / 5 := by omega
      have hl : lev W r (s + 1) i = lev W r s i + r i (lev W r s i) := by
        simp only [lev, if_neg hs]
      rw [hl, hstep]
      have hi := ih i
      rcases hr i (lev W r s i) with h0 | ⟨h1, (⟨hev, he⟩ | ⟨hod, he⟩)⟩
      · rw [h0, add_zero]; exact hi
      · rw [h1, he, arr_raise_even hev]
        rw [he] at hi; exact hi
      · rw [h1, he, arr_raise_odd hod]
        rw [he] at hi; exact hi

/-- After `t` iterations every node's prefix has length at most `t / D + 1`. -/
theorem lev_le (W : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℕ → ℕ) (δ : Fin n → ℕ) (D : ℕ)
    (hD : 1 ≤ D) (hδD : ∀ i, δ i ≤ D) (hδ : ∀ i j, W i j ≠ 0 → δ i ≤ δ j + 1 ∧ δ j ≤ δ i + 1)
    (hr : ∀ i j, r i j = 0 ∨ (r i j = 1 ∧ ((j % 2 = 0 ∧ δ i = 0) ∨ (j % 2 = 1 ∧ δ i = D))))
    (t : ℕ) (i : Fin n) : lev W r (microSteps * t) i ≤ t / D + 1 := by
  have h := arr_lev_le W r δ D hδD hδ hr (microSteps * t) i
  have ht : (microSteps * t + 2) / 5 = t := by simp only [microSteps]; omega
  rw [ht] at h
  obtain ⟨k, hk⟩ : ∃ k, lev W r (microSteps * t) i = k := ⟨_, rfl⟩
  rw [hk] at h ⊢
  rcases Nat.eq_zero_or_pos k with hk0 | hk0
  · subst hk0; exact Nat.zero_le _
  · have h1 : (k - 1) * D ≤ t := by
      unfold arr at h
      rw [if_neg (by omega)] at h
      exact le_trans (Nat.le_add_right _ _) h
    have h2 : k - 1 ≤ t / D := by
      rw [Nat.le_div_iff_mul_le (by omega)]
      exact h1
    omega

end ULB63
