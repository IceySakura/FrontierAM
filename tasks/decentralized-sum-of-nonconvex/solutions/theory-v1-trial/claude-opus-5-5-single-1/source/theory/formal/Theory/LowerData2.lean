import Theory.LowerData

/-!
# The data lower bound: budget invariant, AM–GM, and the theorem
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Blocks

variable {n m c b d : ℕ}

lemma answer_supp_sub (I : Instance n m d) (i : Fin n) (cl : Call m d) (T0 T1 : Set (Fin d))
    (hT : T0 ⊆ T1) (hcl : cl.supp ⊆ T0)
    (hgr : ∀ ξ x, cl = .grad ξ x → supp (gradient (I.F i ξ) x) ⊆ T1)
    (hpr : ∀ (v : E d) η, supp v ⊆ T0 → supp (I.prox v η) ⊆ T0) :
    (Algorithm.answer (c := c) I i cl).supp ⊆ T1 := by
  cases cl with
  | idle => simp [Algorithm.answer, Obs.supp]
  | grad ξ x => simp only [Algorithm.answer, Obs.supp]; exact hgr ξ x rfl
  | prox v η =>
    simp only [Algorithm.answer, Obs.supp]
    simp only [Call.supp] at hcl
    exact (hpr v η hcl).trans hT

lemma le_update_succ (k : Fin m → ℕ) (ξ : Fin m) : k ≤ Function.update k ξ (k ξ + 1) := by
  intro x
  by_cases hx : x = ξ
  · subst hx; simp
  · simp [Function.update_of_ne hx]

lemma sum_update_succ (k : Fin m → ℕ) (ξ : Fin m) :
    ∑ x, Function.update k ξ (k ξ + 1) x = ∑ x, k x + 1 := by
  classical
  rw [Finset.sum_update_of_mem (mem_univ ξ), Finset.sdiff_singleton_eq_erase,
    ← Finset.add_sum_erase _ k (mem_univ ξ)]
  ring

/-- Budget invariant: the total depth explored in the blocks is at most the number of
micro-steps (only node `i0` can deepen a block, by one per gradient query). -/
theorem exists_budget (A : Algorithm n m c)
    (hcall : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)), (A.call d info H).supp ⊆ allowed info H)
    (hsend : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)) k,
      supp (A.send d info H k) ⊆ allowed info H)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (S : (Fin m → ℕ) → Set (Fin d))
    (hSmono : ∀ k k', k ≤ k' → S k ⊆ S k') (i0 : Fin n) (hx0 : I.x0 = 0)
    (hgrad0 : ∀ ξ (x : E d) k, supp x ⊆ S k →
      supp (gradient (I.F i0 ξ) x) ⊆ S (Function.update k ξ (k ξ + 1)))
    (hgradoth : ∀ i, i ≠ i0 → ∀ ξ (x : E d), gradient (I.F i ξ) x = 0)
    (hprox : ∀ (v : E d) η k, supp v ⊆ S k → supp (I.prox v η) ⊆ S k) (ω : A.Ω) :
    ∀ s, ∃ k : Fin m → ℕ, ∑ ξ, k ξ ≤ s ∧
      ∀ i, allowed (A.info I W ω i) (A.hist I W ω s i) ⊆ S k := by
  classical
  intro s
  induction s with
  | zero =>
    refine ⟨0, by simp, fun i => ?_⟩
    simp only [Algorithm.hist, allowed, Algorithm.info, hx0, List.not_mem_nil, Set.iUnion_of_empty,
      Set.iUnion_empty, Set.union_empty]
    intro p hp
    simp [supp] at hp
  | succ s ih =>
    obtain ⟨k, hk, hall⟩ := ih
    by_cases hs : s % microSteps = commSlot
    · refine ⟨k, by omega, fun i => ?_⟩
      have hh : A.hist I W ω (s + 1) i = A.hist I W ω s i ++
          [.recv fun k' => ∑ j, W i j • A.send d (A.info I W ω j) (A.hist I W ω s j) k'] := by
        simp only [Algorithm.hist, if_pos hs]
      rw [hh, allowed_append]
      refine Set.union_subset (hall i) ?_
      intro p hp
      simp only [Obs.supp, Set.mem_iUnion] at hp
      obtain ⟨k', hk'⟩ := hp
      obtain ⟨j, _, hpj⟩ := supp_sum_smul _ _ p hk'
      exact hall j (hsend d _ _ k' hpj)
    · have hh : ∀ i, A.hist I W ω (s + 1) i = A.hist I W ω s i ++
          [Algorithm.answer I i (A.call d (A.info I W ω i) (A.hist I W ω s i))] := fun i => by
        simp only [Algorithm.hist, if_neg hs]
      have hcl : ∀ i, (A.call d (A.info I W ω i) (A.hist I W ω s i)).supp ⊆ S k :=
        fun i => (hcall d _ _).trans (hall i)
      obtain ⟨k', hkk', hsum', hans⟩ : ∃ k' : Fin m → ℕ, k ≤ k' ∧ ∑ ξ, k' ξ ≤ ∑ ξ, k ξ + 1 ∧
          ∀ i, (Algorithm.answer (c := c) I i
            (A.call d (A.info I W ω i) (A.hist I W ω s i))).supp ⊆ S k' := by
        by_cases hg : ∃ ξ x, A.call d (A.info I W ω i0) (A.hist I W ω s i0) = .grad ξ x
        · obtain ⟨ξ, x, hξx⟩ := hg
          refine ⟨Function.update k ξ (k ξ + 1), le_update_succ k ξ, (sum_update_succ k ξ).le,
            fun i => ?_⟩
          refine answer_supp_sub I i _ (S k) _ (hSmono _ _ (le_update_succ k ξ)) (hcl i) ?_
            (fun v η hv => hprox v η k hv)
          intro ξ' x' hc
          by_cases hi : i = i0
          · subst hi
            rw [hξx] at hc
            cases hc
            have hx : supp x ⊆ S k := by
              have := hcl i; rw [hξx] at this; simpa [Call.supp] using this
            exact hgrad0 ξ x k hx
          · rw [hgradoth i hi]
            intro p hp; simp [supp] at hp
        · push_neg at hg
          refine ⟨k, le_rfl, by omega, fun i => ?_⟩
          refine answer_supp_sub I i _ (S k) _ subset_rfl (hcl i) ?_
            (fun v η hv => hprox v η k hv)
          intro ξ' x' hc
          by_cases hi : i = i0
          · subst hi; exact absurd hc (hg ξ' x')
          · rw [hgradoth i hi]
            intro p hp; simp [supp] at hp
      refine ⟨k', by omega, fun i => ?_⟩
      rw [hh i, allowed_append]
      exact Set.union_subset ((hall i).trans (hSmono _ _ hkk')) (hans i)

theorem supp_avgOutput_set (A : Algorithm n m c)
    (hout : ∀ d (info : Info n d A.Ω) (H : List (Obs d c)), supp (A.out d info H) ⊆ allowed info H)
    (I : Instance n m d) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω) (t : ℕ) (S : Set (Fin d))
    (h : ∀ i, allowed (A.info I W ω i) (A.hist I W ω (microSteps * t) i) ⊆ S) :
    supp (A.avgOutput I W ω t) ⊆ S := by
  intro p hp
  simp only [Algorithm.avgOutput, supp, Set.mem_setOf_eq, PiLp.smul_apply, smul_eq_mul] at hp
  have hp' := right_ne_zero_of_mul hp
  rw [WithLp.ofLp_sum, Finset.sum_apply] at hp'
  obtain ⟨i, -, hi⟩ := Finset.exists_ne_zero_of_sum_ne_zero hp'
  exact h i (hout d _ _ hi)

/-- AM–GM: `∑ q^{2 k_ξ + 2} ≥ m q^{2 S/m + 2}` when `∑ k_ξ ≤ S`. -/
lemma sum_pow_ge {q : ℝ} (hq0 : 0 < q) (hq1 : q ≤ 1) (hm : 1 ≤ m) (k : Fin m → ℕ) (S : ℕ)
    (hS : ∑ ξ, k ξ ≤ S) :
    (m : ℝ) * q ^ (2 * (S : ℝ) / m + 2) ≤ ∑ ξ, q ^ (2 * k ξ + 2) := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast (by omega : 0 < m)
  have amgm := Real.geom_mean_le_arith_mean_weighted (Finset.univ : Finset (Fin m))
    (fun _ => 1 / (m : ℝ)) (fun ξ => q ^ (2 * k ξ + 2)) (fun _ _ => by positivity)
    (by simp; field_simp) (fun _ _ => by positivity)
  have hlhs : ∏ ξ : Fin m, (q ^ (2 * k ξ + 2)) ^ (1 / (m : ℝ)) =
      q ^ (2 * ((∑ ξ, k ξ : ℕ) : ℝ) / m + 2) := by
    have : ∀ ξ : Fin m, (q ^ (2 * k ξ + 2)) ^ (1 / (m : ℝ)) =
        q ^ (((2 * k ξ + 2 : ℕ) : ℝ) * (1 / (m : ℝ))) := by
      intro ξ
      rw [← Real.rpow_natCast, ← Real.rpow_mul hq0.le]
    rw [Finset.prod_congr rfl fun ξ _ => this ξ, ← Real.rpow_sum_of_pos hq0]
    congr 1
    rw [← Finset.sum_mul]
    push_cast
    rw [Finset.sum_add_distrib, ← Finset.mul_sum]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  rw [hlhs, ← Finset.mul_sum] at amgm
  have hmono : q ^ (2 * (S : ℝ) / m + 2) ≤ q ^ (2 * ((∑ ξ, k ξ : ℕ) : ℝ) / m + 2) := by
    apply Real.rpow_le_rpow_of_exponent_ge hq0 hq1
    have : ((∑ ξ, k ξ : ℕ) : ℝ) ≤ S := by exact_mod_cast hS
    have : 2 * ((∑ ξ, k ξ : ℕ) : ℝ) / m ≤ 2 * (S : ℝ) / m := by
      apply div_le_div_of_nonneg_right _ hm'.le; linarith
    linarith
  have h2 : (m : ℝ) * q ^ (2 * ((∑ ξ, k ξ : ℕ) : ℝ) / m + 2) ≤ ∑ ξ, q ^ (2 * k ξ + 2) := by
    have := mul_le_mul_of_nonneg_left amgm hm'.le
    rw [← mul_assoc, mul_one_div_cancel hm'.ne', one_mul] at this
    exact this
  calc (m : ℝ) * q ^ (2 * (S : ℝ) / m + 2) ≤ (m : ℝ) * q ^ (2 * ((∑ ξ, k ξ : ℕ) : ℝ) / m + 2) :=
        mul_le_mul_of_nonneg_left hmono hm'.le
    _ ≤ _ := h2

/-- Real-variable core with a real exponent. -/
lemma key_ineq_rpow {G ε s X : ℝ} (hG : 0 < G) (hε : 0 < ε) (hs1 : 1 < s) (hX : 0 ≤ X)
    (hlt : X * (2 / (s - 1)) < Real.log (G / ε)) : ε < G * ((s - 1) / (s + 1)) ^ X := by
  set q := (s - 1) / (s + 1) with hq
  have hq0 : 0 < q := div_pos (by linarith) (by linarith)
  have hlog1 : Real.log (1 / q) ≤ 2 / (s - 1) := by
    have h := Real.log_le_sub_one_of_pos (show 0 < 1 / q by positivity)
    have : 1 / q - 1 = 2 / (s - 1) := by
      rw [hq]; field_simp; ring
    linarith
  have hpow : Real.log (q ^ X) = -(X * Real.log (1 / q)) := by
    rw [Real.log_rpow hq0, one_div, Real.log_inv]; ring
  have hXl : X * Real.log (1 / q) ≤ X * (2 / (s - 1)) := mul_le_mul_of_nonneg_left hlog1 hX
  have hlt' : Real.log (ε / G) < Real.log (q ^ X) := by
    rw [hpow, Real.log_div hε.ne' hG.ne']
    rw [Real.log_div hG.ne' hε.ne'] at hlt
    linarith
  have := (Real.log_lt_log_iff (by positivity) (by positivity)).1 hlt'
  rw [div_lt_iff₀ hG] at this
  linarith

end Blocks
end ULB63
