import Theory.BallGeometry
import Theory.TraceLower

/-! Constant relative-error lower bound when the center is not supplied. -/

open MeasureTheory Set Metric
open scoped ENNReal

namespace Cubature

noncomputable section

private theorem relative_pair_fail {u s t : ℝ} (hs : 0 < s) (ht : 2*s ≤ t) :
    s/4 < |u-s| ∨ t/4 < |u-t| := by
  by_cases h : s/4 < |u-s|
  · exact Or.inl h
  · have hsu := (abs_le.mp (not_lt.mp h)).2
    exact Or.inr ((show t/4 < -(u-t) by linarith).trans_le (neg_le_abs _))

private theorem paired_search_lower {Q J Ω : Type*} [Fintype J] [DecidableEq J] [Nonempty J]
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {N : ℕ} (alg : Ω → QueryTree Q ℝ) (budget : ∀ ω, (alg ω).depth ≤ N)
    (hN : 2*N ≤ Fintype.card J) (site : Q → Option J) (oracle : J → Bool → Q → Bool)
    (mass : J → Bool → ℝ) (hm0 : ∀ j, 0 < mass j false)
    (hm1 : ∀ j, 2*mass j false ≤ mass j true)
    (hloc : ∀ j b q, site q ≠ some j → oracle j b q = false)
    (hm : ∀ j b, Measurable (fun ω => (alg ω).run (oracle j b))) :
    ∃ j b, (1:ℝ)/4 ≤ μ.real {ω | mass j b / 4 < |(alg ω).run (oracle j b) - mass j b|} := by
  classical
  let E (j : J) (b : Bool) : Set Ω :=
    {ω | mass j b / 4 < |(alg ω).run (oracle j b) - mass j b|}
  let L (j : J) (b : Bool) : Ω → ℝ := (E j b).indicator (fun _ => 1)
  let M : ℝ := Fintype.card J
  have hE (j b) : MeasurableSet (E j b) :=
    measurableSet_lt measurable_const (continuous_abs.measurable.comp ((hm j b).sub_const _))
  have hL (j b) : Integrable (L j b) μ := (integrable_const _).indicator (hE j b)
  have hn (j b ω) : 0 ≤ L j b ω := by
    unfold L
    exact indicator_nonneg (fun _ _ => zero_le_one) _
  have hω (ω : Ω) : M/2 ≤ ∑ j, (L j false ω + L j true ω) := by
    let t := alg ω
    let seen := t.revealed site (fun _ => false)
    let S := seenᶜ
    have hC : S.card = Fintype.card J - seen.card := Finset.card_compl _
    have hseen : seen.card ≤ N :=
      (QueryTree.revealed_card site (fun _ => false) t).trans (budget ω)
    have hcs : Fintype.card J ≤ 2*S.card := by omega
    have hcs' : M ≤ 2*(S.card:ℝ) := by dsimp [M]; exact_mod_cast hcs
    have hh (j : J) (hj : j ∈ S) : (1:ℝ) ≤ L j false ω + L j true ω := by
      have hEq (b : Bool) : t.run (oracle j b) = t.run (fun _ => false) :=
        (QueryTree.revealed_flip_invariant site (fun _ => false) (oracle j b) t j
          (hloc j b) (Finset.mem_compl.mp hj)).1
      rcases relative_pair_fail (u := t.run (fun _ => false)) (hm0 j) (hm1 j) with h | h
      · have he : ω ∈ E j false := by
          change mass j false / 4 < |t.run (oracle j false) - mass j false|
          rwa [hEq false]
        exact (by simp only [L,indicator_of_mem he]; linarith [hn j true ω])
      · have he : ω ∈ E j true := by
          change mass j true / 4 < |t.run (oracle j true) - mass j true|
          rwa [hEq true]
        exact (by simp only [L,indicator_of_mem he]; linarith [hn j false ω])
    calc
      _ ≤ (S.card:ℝ) := by linarith
      _ = ∑ _ ∈ S, (1:ℝ) := by simp
      _ ≤ ∑ j ∈ S, (L j false ω + L j true ω) := Finset.sum_le_sum hh
      _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun j _ _ =>
        add_nonneg (hn j false ω) (hn j true ω))
  have hin (j : J) : Integrable (fun ω => L j false ω + L j true ω) μ := (hL _ _).add (hL _ _)
  have hi := integral_mono (integrable_const (M/2))
    (integrable_finsetSum Finset.univ (fun j _ => hin j)) hω
  rw [integral_const, show μ.real univ = 1 by simp, one_smul,
    integral_finsetSum _ (fun j _ => hin j)] at hi
  have hpart (j : J) :
      (∫ ω, (L j false ω + L j true ω) ∂μ) = μ.real (E j false) + μ.real (E j true) := by
    rw [integral_add (hL _ _) (hL _ _)]
    unfold L
    rw [integral_indicator_const _ (hE j false),integral_indicator_const _ (hE j true)]
    simp [smul_eq_mul]
  simp_rw [hpart] at hi
  change ∃ j b, (1:ℝ)/4 ≤ μ.real (E j b)
  by_contra! h
  have hSum := Finset.sum_lt_sum_of_nonempty (Finset.univ_nonempty : (Finset.univ : Finset J).Nonempty)
    (f := fun j => μ.real (E j false) + μ.real (E j true))
    (g := fun _ => (1:ℝ)/2) (fun j _ => by linarith [h j false, h j true])
  simp only [Finset.sum_const, Finset.card_univ,nsmul_eq_mul, mul_one_div] at hSum
  exact (not_lt_of_ge hi) hSum

private def radiusChoice (b : Bool) : ℝ := if b then 1/2 else 1/16

private def searchSite {d J : ℕ} (c : Fin J → GaussianSpace d) (z : GaussianSpace d) :
    Option (Fin J) := by
  classical
  exact if h : ∃ j, z ∈ ball (c j) (1/2) then some h.choose else none

private theorem searchSite_of {d J : ℕ} {c : Fin J → GaussianSpace d}
    (hsep : ∀ i j, i ≠ j → 1 ≤ dist (c i) (c j))
    (z : GaussianSpace d) (j : Fin J) (hj : z ∈ ball (c j) (1/2)) : searchSite c z = some j := by
  have hex : ∃ i, z ∈ ball (c i) (1/2) := ⟨j,hj⟩
  have hc := hex.choose_spec
  have he : hex.choose = j := by
    by_contra h
    have h₁ : dist z (c hex.choose) < 1/2 := hc
    have h₂ : dist z (c j) < 1/2 := hj
    have h₃ := hsep hex.choose j h
    have h₄ := dist_triangle (c hex.choose) z (c j)
    rw [dist_comm (c hex.choose) z] at h₄
    linarith
  rw [searchSite,dite_eq_left hex,he]

theorem ballSearch_bound {d N : ℕ} (hd : 0 < d)
    (hN : ((2*N : ℕ):ℝ≥0∞) * unitBallVolume d < (2:ℝ≥0∞)^d)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (alg : Ω → QueryTree (GaussianSpace d) ℝ) (budget : ∀ ω, (alg ω).depth ≤ N)
    (hM : ∀ c ∈ coordinateBox d 1, ∀ r : ℝ, (r = 1/16 ∨ r = 1/2) →
      Measurable (fun ω => (alg ω).run (ballOracle c r))) :
    ∃ c ∈ coordinateBox d 1, ∃ r : ℝ, (r = 1/16 ∨ r = 1/2) ∧
      0 < gaussianBallMass c r ∧
      (1:ℝ)/4 ≤ μ.real {ω | gaussianBallMass c r / 4 <
        |(alg ω).run (ballOracle c r) - gaussianBallMass c r|} := by
  obtain ⟨c,hc,hsep⟩ := cube_packing hd (2*N) hN
  have hR (b : Bool) : radiusChoice b = 1/16 ∨ radiusChoice b = 1/2 := by
    cases b <;> simp [radiusChoice]
  have hr (b : Bool) : 0 < radiusChoice b := by cases b <;> norm_num [radiusChoice]
  have hsite (i : Fin (2*N+1)) (b : Bool) (q : GaussianSpace d) (h : searchSite c q ≠ some i) :
      ballOracle (c i) (radiusChoice b) q = false := by
    apply decide_eq_false
    intro hh
    have he : radiusChoice b ≤ (1/2:ℝ) := by cases b <;> norm_num [radiusChoice]
    exact h (searchSite_of hsep q i (by exact hh.trans_le he))
  obtain ⟨i,b,hrisk⟩ := paired_search_lower μ alg budget
    (by rw [Fintype.card_fin]; omega : 2*N ≤ Fintype.card (Fin (2*N+1)))
    (searchSite c) (fun i b => ballOracle (c i) (radiusChoice b))
    (fun i b => gaussianBallMass (c i) (radiusChoice b))
    (fun i => gaussianBallMass_pos _ (hr false))
    (fun i => gaussianBallMass_ratio hd (hc i)) hsite
    (fun i b => hM (c i) (hc i) _ (hR b))
  exact ⟨c i,hc i,radiusChoice b,hR b,gaussianBallMass_pos _ (hr b),hrisk⟩

end

end Cubature
