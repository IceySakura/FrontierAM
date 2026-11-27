import Theory.Protocol

open MeasureTheory
open scoped ENNReal

noncomputable section

namespace ULB63
namespace Lower

open Protocol

variable {n m c d : ℕ} {L μ Δ ρ ε : ℝ}

theorem allowed_append {Ω : Type} (info : Info n d Ω) (H : List (Obs d c)) (o : Obs d c) :
    allowed info (H ++ [o]) = allowed info H ∪ o.supp := by
  ext k
  simp only [allowed, Set.mem_union, Set.mem_iUnion, List.mem_append, List.mem_singleton]
  constructor
  · rintro (h | ⟨u, hu | rfl, hk⟩)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr ⟨u, hu,hk⟩)
    · exact Or.inr hk
  · rintro ((h | ⟨u, hu, hk⟩) | ho)
    · exact Or.inl h
    · exact Or.inr ⟨u, Or.inl hu, hk⟩
    · exact Or.inr ⟨o, Or.inr rfl, ho⟩

theorem allowed_invariant {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω)
    (S : ℕ → Set (Fin d)) (h0 : supp I.x0 ⊆ S 0)
    (hS : ∀ s, S s ⊆ S (s+1))
    (hg : ∀ s i ξ x, Supported (S s) x → Supported (S (s+1)) (gradient (I.F i ξ) x))
    (hp : ∀ s x η, Supported (S s) x → Supported (S (s+1)) (I.prox x η)) :
    ∀ s i, allowed (A.info I W ω i) (A.hist I W ω s i) ⊆ S s := by
  intro s
  induction s with
  | zero =>
    intro i
    simpa only [Algorithm.hist, allowed, List.not_mem_nil, Set.iUnion_of_empty,
      Set.iUnion_empty, Set.union_empty, Algorithm.info] using h0
  | succ s ih =>
    intro i
    rw [Algorithm.hist]
    split
    · rw [allowed_append]
      apply Set.union_subset ((ih i).trans (hS _))
      apply Set.iUnion_subset
      intro k
      exact (Supported.sum (s := Finset.univ) (fun j _ =>
        Supported.smul (show Supported (S s) (A.send d (A.info I W ω j) (A.hist I W ω s j) k)
          from (hA.send_zero_respecting d _ _ k).trans (ih j)) (W i j))).mono (hS s)
    · rw [allowed_append]
      apply Set.union_subset ((ih i).trans (hS _))
      have hh := (hA.call_zero_respecting d (A.info I W ω i) (A.hist I W ω s i)).trans (ih i)
      cases he : A.call d (A.info I W ω i) (A.hist I W ω s i) with
      | idle => exact Set.empty_subset _
      | grad ξ x =>
        rw [he] at hh
        exact hg s i ξ x hh
      | prox x η =>
        rw [he] at hh
        exact hp s x η hh

theorem avg_from_allowed {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω) (t : ℕ)
    {S : Set (Fin d)}
    (hS : ∀ i, allowed (A.info I W ω i) (A.hist I W ω (microSteps*t) i) ⊆ S) :
    Supported S (A.avgOutput I W ω t) := by
  unfold Algorithm.avgOutput Algorithm.output
  exact (Supported.sum (fun i _ =>
    (hA.out_zero_respecting d _ _).trans (hS i))).smul _

theorem hitting_gt {A : Algorithm n m c} {I : Instance n m d}
    {W : Matrix (Fin n) (Fin n) ℝ} (T : ℕ)
    (h : ∀ t ≤ T, ENNReal.ofReal ε < A.expectedGap I W t) :
    ((T+1 : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  apply le_iInf
  intro t
  apply le_iInf
  intro ht
  exact_mod_cast (show T+1 ≤ t by
    by_contra hh
    exact (not_le_of_gt (h t (by omega))) ht)

theorem minimax_ge {B : ℕ∞}
    (h : ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ →
      ∃ (d : ℕ) (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ),
        I.InClass L μ Δ ∧ CommClass n ρ W ∧ B ≤ A.hittingTime I W ε) :
    B ≤ minimax n m c L μ Δ ρ ε := by
  apply le_iInf
  intro A
  apply le_iInf
  intro hA
  obtain ⟨d, I, W, hI, hW, hT⟩ := h A hA
  apply le_iSup_of_le d
  apply le_iSup_of_le I
  apply le_iSup_of_le hI
  apply le_iSup_of_le W
  apply le_iSup_of_le hW
  exact hT

end Lower
end ULB63
