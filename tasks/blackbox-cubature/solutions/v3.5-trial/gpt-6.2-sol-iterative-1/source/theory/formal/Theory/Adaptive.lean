import Theory.RareGeometry
import Mathlib.MeasureTheory.Integral.Bochner.Set

open MeasureTheory Set
open scoped BigOperators Classical

namespace Cubature

/-- Once a random seed is fixed, a point-query algorithm for indicators is a
binary decision tree. Each child stores the computation for its response. -/
inductive MembershipTree (d : ℕ)
  | result (estimate : ℝ)
  | probe (point : Vec d) (inside outside : MembershipTree d)

namespace MembershipTree

variable {d : ℕ}

def depth : MembershipTree d → ℕ
  | .result _ => 0
  | .probe _ yes no => 1 + max yes.depth no.depth

def zeroPoints : MembershipTree d → List (Vec d)
  | .result _ => []
  | .probe z _ no => z :: no.zeroPoints

def zeroEstimate : MembershipTree d → ℝ
  | .result n => n
  | .probe _ _ no => no.zeroEstimate

noncomputable def estimate (set : Set (Vec d)) : MembershipTree d → ℝ
  | .result n => n
  | .probe z yes no => if z ∈ set then estimate set yes else estimate set no

lemma zeroPoints_length_le (T : MembershipTree d) : T.zeroPoints.length ≤ T.depth := by
  induction T with
  | result _ => simp [depth, zeroPoints]
  | probe z yes no hy hn =>
    simp only [depth, zeroPoints, List.length_cons]
    have h := hn.trans (le_max_right yes.depth no.depth)
    omega

lemma estimate_eq_zeroEstimate (S : Set (Vec d)) (T : MembershipTree d)
    (h : ∀ z ∈ T.zeroPoints, z ∉ S) : T.estimate S = T.zeroEstimate := by
  induction T with
  | result _ => rfl
  | probe z yes no hy hn =>
    unfold estimate zeroEstimate
    rw [if_neg (h z (by simp [zeroPoints]))]
    exact hn fun w hw => h w (by simp [zeroPoints,hw])

end MembershipTree

noncomputable def displacedBallSuccess {d : ℕ} {Ω : Type*}
    (tree : Ω → MembershipTree d) (a : Vec d) (r ε : ℝ) : Set Ω :=
  {ω | |(tree ω).estimate (shiftedBall a r) - shiftedBallMass a r| ≤ ε * shiftedBallMass a r}

noncomputable section

variable {d : ℕ}

private lemma touched_count {ι : Type*} [Fintype ι] (O : ι → Set (Vec d))
    (hd : Pairwise fun i j => Disjoint (O i) (O j)) (L : List (Vec d)) :
    ((Finset.univ : Finset ι).filter (fun i => ∃ z ∈ L, z ∈ O i)).card ≤ L.length := by
  let S := (Finset.univ : Finset ι).filter (fun i => ∃ z ∈ L, z ∈ O i)
  have hex (i : ι) (hi : i ∈ S) : ∃ k : Fin L.length, L.get k ∈ O i := by
    obtain ⟨z,hz,hi'⟩ := (Finset.mem_filter.mp hi).2
    obtain ⟨k,rfl⟩ := List.mem_iff_get.mp hz
    exact ⟨k,hi'⟩
  let inx (i : ι) : ℕ := if h : i ∈ S then (Classical.choose (hex i h)).val else 0
  have hw (i : ι) (hi : i ∈ S) :
      ∃ k : Fin L.length, inx i = k.val ∧ L.get k ∈ O i := by
    refine ⟨Classical.choose (hex i hi), ?_, Classical.choose_spec (hex i hi)⟩
    simp [inx,hi]
  have inj : Set.InjOn inx (S : Set ι) := by
    intro i hi j hj h
    obtain ⟨k,hk,hi'⟩ := hw i hi
    obtain ⟨l,hl,hj'⟩ := hw j hj
    have heq : k = l := Fin.ext (hk.symm.trans (h.trans hl))
    subst l
    by_contra hne
    exact Set.disjoint_left.mp (hd hne) hi' hj'
  have hmap : Set.MapsTo inx (S : Set ι) (Finset.range L.length : Set ℕ) := by
    intro i hi
    obtain ⟨k,hk,_⟩ := hw i hi
    simpa only [Finset.mem_coe, Finset.mem_range, hk] using k.isLt
  simpa only [Finset.card_range] using Finset.card_le_card_of_injOn inx hmap inj

private lemma integrable_indicator_set {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {S : Set Ω} (hS : MeasurableSet S) :
    Integrable (S.indicator (fun _ : Ω => (1 : ℝ))) P :=
  (integrable_const (1 : ℝ)).indicator hS

lemma adaptive_rare_average {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] {m N : ℕ} (hm : 0 < m) (hd : 0 < d)
    (tree : Ω → MembershipTree d) (hN : ∀ ω, (tree ω).depth ≤ N)
    (hMeas : ∀ j : Fin d → Fin m,
      MeasurableSet (displacedBallSuccess tree (rareCenter j) (rareInner d m) (1/3)) ∧
      MeasurableSet (displacedBallSuccess tree (rareCenter j) (rareOuter m) (1/3))) :
    (∑ j : Fin d → Fin m,
      (P.real (displacedBallSuccess tree (rareCenter j) (rareInner d m) (1/3)) +
       P.real (displacedBallSuccess tree (rareCenter j) (rareOuter m) (1/3)))) ≤ (m : ℝ)^d + N := by
  let J := Fin d → Fin m
  let O (j : J) := shiftedBall (rareCenter j) (rareOuter m)
  let I (j : J) := shiftedBall (rareCenter j) (rareInner d m)
  let inner (j : J) := displacedBallSuccess tree (rareCenter j) (rareInner d m) (1/3)
  let outer (j : J) := displacedBallSuccess tree (rareCenter j) (rareOuter m) (1/3)
  let it (j : J) := (inner j).indicator (fun _ : Ω => (1 : ℝ))
  let ot (j : J) := (outer j).indicator (fun _ : Ω => (1 : ℝ))
  have pointbound (ω : Ω) : (∑ j : J, (it j ω + ot j ω)) ≤ (m : ℝ)^d + N := by
    let T := tree ω
    let S := (Finset.univ : Finset J).filter (fun j => ∃ z ∈ T.zeroPoints, z ∈ O j)
    have hnum : S.card ≤ N :=
      (touched_count O (fun _ _ h => rareBalls_disjoint hm h) T.zeroPoints).trans
        (T.zeroPoints_length_le.trans (hN ω))
    have hij (j : J) : it j ω + ot j ω ≤ 1 + (if j ∈ S then (1 : ℝ) else 0) := by
      by_cases h : j ∈ S
      · simp only [h, ite_true, it, ot, Set.indicator_apply]
        split <;> split <;> norm_num
      · have notouch : ∀ z ∈ T.zeroPoints, z ∉ O j := by
          intro z hz hO
          exact h (Finset.mem_filter.mpr ⟨Finset.mem_univ _,z,hz,hO⟩)
        have hsub : I j ⊆ O j :=
          shiftedBall_mono (rareInner_pos hm).le (rareInner_le_outer hm)
        have h1 := T.estimate_eq_zeroEstimate (I j) (fun z hz hs => notouch z hz (hsub hs))
        have h2 := T.estimate_eq_zeroEstimate (O j) notouch
        have gap := rare_inner_mass_gap hd hm (rareCenter j) (rareCenter_mem hm j)
        have notboth (ha : ω ∈ inner j) (hb : ω ∈ outer j) : False := by
          change |T.estimate (I j) - shiftedBallMass (rareCenter j) (rareInner d m)| ≤ _ at ha
          change |T.estimate (O j) - shiftedBallMass (rareCenter j) (rareOuter m)| ≤ _ at hb
          rw [h1] at ha
          rw [h2] at hb
          linarith only [(abs_le.mp ha).2,(abs_le.mp hb).1,gap.1,gap.2]
        simp only [h, ite_false, add_zero, it, ot, Set.indicator_apply]
        split
        · rename_i ha
          split
          · rename_i hb
            exact (notboth ha hb).elim
          · norm_num
        · split <;> norm_num
    have eq : (∑ j : J, (if j ∈ S then (1 : ℝ) else 0)) = (S.card : ℝ) := by simp
    calc
      _ ≤ ∑ j : J, (1 + if j ∈ S then (1 : ℝ) else 0) := Finset.sum_le_sum fun j _ => hij j
      _ = (m : ℝ)^d + (S.card : ℝ) := by
        rw [Finset.sum_add_distrib, eq]
        simp [J]
      _ ≤ _ := by exact add_le_add le_rfl (by exact_mod_cast hnum)
  have integrates (j : J) : Integrable (fun ω => it j ω + ot j ω) P :=
    (integrable_indicator_set (hMeas j).1).add (integrable_indicator_set (hMeas j).2)
  calc
    (∑ j : J, (P.real (inner j) + P.real (outer j))) =
        ∑ j : J, ∫ ω, (it j ω + ot j ω) ∂P := by
      apply Finset.sum_congr rfl
      intro j _
      rw [integral_add (integrable_indicator_set (hMeas j).1) (integrable_indicator_set (hMeas j).2)]
      exact congrArg₂ (· + ·) (integral_indicator_one (hMeas j).1).symm
        (integral_indicator_one (hMeas j).2).symm
    _ = ∫ ω, ∑ j : J, (it j ω + ot j ω) ∂P :=
      (integral_finsetSum _ (fun j _ => integrates j)).symm
    _ ≤ ∫ _ : Ω, ((m : ℝ)^d + N) ∂P :=
      integral_mono (integrable_finsetSum _ (fun j _ => integrates j)) (integrable_const _) pointbound
    _ = _ := by simp

end

end Cubature
