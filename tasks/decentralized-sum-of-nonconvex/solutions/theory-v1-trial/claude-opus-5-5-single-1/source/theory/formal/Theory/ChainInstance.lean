import Theory.ChainValue
import Theory.Support
import Theory.Sanity

/-!
# The chain instance

Nodes in `SA` hold the even links `k = 0, 2, 4, …` of the chain, nodes in `SB` the odd links,
all other nodes the zero function; every component of a node is the same function.
`h = (μ/2)‖·‖²`, `x₀ = 0`.
-/

open scoped RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Chain

variable {n m d : ℕ}

/-- Even link indices. -/
def SAk (d : ℕ) : Finset ℕ := (range (d + 1)).filter (fun k => k % 2 = 0)

/-- Odd link indices. -/
def SBk (d : ℕ) : Finset ℕ := (range (d + 1)).filter (fun k => ¬ k % 2 = 0)

/-- The links held by node `i`. -/
def nodeS (SA SB : Finset (Fin n)) (d : ℕ) (i : Fin n) : Finset ℕ :=
  if i ∈ SA then SAk d else if i ∈ SB then SBk d else ∅

lemma nodeS_subset (SA SB : Finset (Fin n)) (d : ℕ) (i : Fin n) : nodeS SA SB d i ⊆ range (d + 1) := by
  unfold nodeS SAk SBk
  split_ifs
  · exact Finset.filter_subset _ _
  · exact Finset.filter_subset _ _
  · exact Finset.empty_subset _

/-- The hard instance. -/
def chainInst (n m d : ℕ) (SA SB : Finset (Fin n)) (L μ R q : ℝ) : Instance n m d where
  F i _ := quad (u d) (L / 4) (nodeS SA SB d i) (wch d q) (cch R)
  h x := ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1 + η * μ)⁻¹ • v

lemma quad_smul (U : ℕ → E d) (a β : ℝ) (S : Finset ℕ) (w c : ℕ → ℝ) (x : E d) :
    a * quad U β S w c x = quad U (a * β) S w c x := by
  unfold quad; ring

lemma quad_split (U : ℕ → E d) (β : ℝ) (w c : ℕ → ℝ) (x : E d) :
    quad U β (SAk d) w c x + quad U β (SBk d) w c x = quad U β (range (d + 1)) w c x := by
  unfold quad SAk SBk
  rw [← mul_add, Finset.sum_filter_add_sum_filter_not]

lemma quad_empty (U : ℕ → E d) (β : ℝ) (w c : ℕ → ℝ) (x : E d) : quad U β ∅ w c x = 0 := by
  simp [quad]

lemma chainInst_f (hm : 1 ≤ m) {SA SB : Finset (Fin n)} (hdisj : Disjoint SA SB)
    (hcard : SA.card = SB.card) (L μ R q : ℝ) (x : E d) :
    (chainInst n m d SA SB L μ R q).f x =
      quad (u d) (SA.card * (L / 4) / n) (range (d + 1)) (wch d q) (cch R) x := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast (by omega : m ≠ 0)
  simp only [Instance.f, chainInst, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  have h1 : ∀ i : Fin n, 1 / (m : ℝ) * (m * quad (u d) (L / 4) (nodeS SA SB d i) (wch d q) (cch R) x)
      = (if i ∈ SA then quad (u d) (L / 4) (SAk d) (wch d q) (cch R) x else 0) +
        (if i ∈ SB then quad (u d) (L / 4) (SBk d) (wch d q) (cch R) x else 0) := by
    intro i
    rw [← mul_assoc, one_div, inv_mul_cancel₀ hm', one_mul]
    unfold nodeS
    by_cases hA : i ∈ SA
    · have hB : i ∉ SB := Finset.disjoint_left.1 hdisj hA
      simp [hA, hB]
    · by_cases hB : i ∈ SB
      · simp [hA, hB]
      · simp [hA, hB, quad_empty]
  rw [Finset.sum_congr rfl fun i _ => h1 i, Finset.sum_add_distrib,
    Finset.sum_ite_mem, Finset.sum_ite_mem, Finset.univ_inter, Finset.univ_inter,
    Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul, ← hcard, ← mul_add,
    quad_split, ← mul_assoc, quad_smul]
  congr 1
  ring

lemma quad_convex (U : ℕ → E d) {β : ℝ} (hβ : 0 ≤ β) (S : Finset ℕ) (w c : ℕ → ℝ)
    (hw : ∀ k, 0 ≤ w k) : ConvexOn ℝ Set.univ (quad U β S w c) := by
  refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
  unfold quad
  simp only [smul_eq_mul, inner_add_right, inner_smul_right]
  have key : ∀ k ∈ S, w k * (a * ⟪U k, x⟫ + b * ⟪U k, y⟫ + c k) ^ 2 ≤
      a * (w k * (⟪U k, x⟫ + c k) ^ 2) + b * (w k * (⟪U k, y⟫ + c k) ^ 2) := by
    intro k _
    have hb' : b = 1 - a := by linarith
    subst hb'
    nlinarith [mul_nonneg (mul_nonneg ha hb) (mul_nonneg (hw k) (sq_nonneg (⟪U k, x⟫ - ⟪U k, y⟫)))]
  have hsum := Finset.sum_le_sum key
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum] at hsum
  have hβ2 : 0 ≤ β / 2 := by linarith
  calc β / 2 * ∑ k ∈ S, w k * (a * ⟪U k, x⟫ + b * ⟪U k, y⟫ + c k) ^ 2
      ≤ β / 2 * (a * ∑ k ∈ S, w k * (⟪U k, x⟫ + c k) ^ 2 +
          b * ∑ k ∈ S, w k * (⟪U k, y⟫ + c k) ^ 2) := mul_le_mul_of_nonneg_left hsum hβ2
    _ = a * (β / 2 * ∑ k ∈ S, w k * (⟪U k, x⟫ + c k) ^ 2) +
          b * (β / 2 * ∑ k ∈ S, w k * (⟪U k, y⟫ + c k) ^ 2) := by ring

lemma chainInst_φ (hm : 1 ≤ m) {SA SB : Finset (Fin n)} (hdisj : Disjoint SA SB)
    (hcard : SA.card = SB.card) (L μ R q : ℝ) (x : E d) :
    (chainInst n m d SA SB L μ R q).φ x = ((φc d (SA.card * (L / 4) / n) μ R q x : ℝ) : EReal) := by
  simp only [Instance.φ, chainInst_f hm hdisj hcard, φc]
  simp [chainInst, EReal.coe_add]

lemma chainInst_φstar (hm : 1 ≤ m) {SA SB : Finset (Fin n)} (hdisj : Disjoint SA SB)
    (hcard : SA.card = SB.card) {L μ R q : ℝ} (hd : 1 ≤ d) (hL : 0 ≤ L) (hμ : 0 ≤ μ) (hq1 : q ≤ 1)
    (hq : μ * q = (SA.card * (L / 4) / n) * (1 - q) ^ 2) :
    (chainInst n m d SA SB L μ R q).φstar =
      ((φc d (SA.card * (L / 4) / n) μ R q (xstar d R q) : ℝ) : EReal) := by
  have hα : 0 ≤ (SA.card : ℝ) * (L / 4) / n := by positivity
  apply le_antisymm
  · refine (iInf_le _ (xstar d R q)).trans ?_
    rw [chainInst_φ hm hdisj hcard]
  · refine le_iInf fun y => ?_
    rw [chainInst_φ hm hdisj hcard]
    exact EReal.coe_le_coe_iff.2 (φc_xstar_le hd hα hμ hq1 hq y)

lemma chainInst_gap (hm : 1 ≤ m) {SA SB : Finset (Fin n)} (hdisj : Disjoint SA SB)
    (hcard : SA.card = SB.card) {L μ R q : ℝ} (hd : 1 ≤ d) (hL : 0 ≤ L) (hμ : 0 ≤ μ) (hq1 : q ≤ 1)
    (hq : μ * q = (SA.card * (L / 4) / n) * (1 - q) ^ 2) (x : E d) :
    (chainInst n m d SA SB L μ R q).gap x = ENNReal.ofReal
      (φc d (SA.card * (L / 4) / n) μ R q x - φc d (SA.card * (L / 4) / n) μ R q (xstar d R q)) := by
  unfold Instance.gap
  rw [chainInst_φ hm hdisj hcard, chainInst_φstar hm hdisj hcard hd hL hμ hq1 hq, ← EReal.coe_sub,
    EReal.toENNReal_of_ne_top (EReal.coe_ne_top _), EReal.toReal_coe]

theorem chainInst_inClass (hm : 1 ≤ m) {SA SB : Finset (Fin n)} (hdisj : Disjoint SA SB)
    (hcard : SA.card = SB.card) {L μ Δ R q : ℝ} (hd : 1 ≤ d) (hL : 0 ≤ L) (hμ : 0 < μ)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hq : μ * q = (SA.card * (L / 4) / n) * (1 - q) ^ 2)
    (hR : (SA.card * (L / 4) / n) / 2 * R ^ 2 ≤ Δ) :
    (chainInst n m d SA SB L μ R q).InClass L μ Δ := by
  have hα : 0 ≤ (SA.card : ℝ) * (L / 4) / n := by positivity
  have hφfun : (chainInst n m d SA SB L μ R q).φ =
      fun x => ((φc d (SA.card * (L / 4) / n) μ R q x : ℝ) : EReal) :=
    funext fun x => chainInst_φ hm hdisj hcard L μ R q x
  have hfconv : ConvexOn ℝ Set.univ (chainInst n m d SA SB L μ R q).f := by
    have : (chainInst n m d SA SB L μ R q).f =
        quad (u d) (SA.card * (L / 4) / n) (range (d + 1)) (wch d q) (cch R) :=
      funext fun x => chainInst_f hm hdisj hcard L μ R q x
    rw [this]
    exact quad_convex _ hα _ _ _ (wch_nonneg hq1)
  refine
    { smooth := fun i _ => ?_
      f_convex := hfconv
      h_ne_bot := fun _ => EReal.coe_ne_bot _
      h_proper := ⟨0, EReal.coe_ne_top _⟩
      h_closed := (continuous_coe_real_ereal.comp (by fun_prop)).lowerSemicontinuous
      h_convex := ?_
      prox_exact := fun v η hη => isProx_halfSq hμ.le hη v
      φ_strongly_convex := ?_
      minimizer := ⟨xstar d R q, fun y => ?_⟩
      initial_gap := ?_ }
  · exact lsmooth_quad (u d) (by positivity) _ _ _
      (fun k _ => ⟨wch_nonneg hq1 k, wch_le_one hq0 k⟩) (by norm_num)
      (sum_inner_u_sq_le _ (nodeS_subset SA SB d i)) (by linarith)
  · have hs : StrongConvexOn Set.univ μ (fun x : E d => μ / 2 * ‖x‖ ^ 2) := by
      rw [strongConvexOn_iff_convex]; simpa using convexOn_const (0 : ℝ) convex_univ
    have hc : ConvexOn ℝ Set.univ (fun x : E d => μ / 2 * ‖x‖ ^ 2) :=
      UniformConvexOn.convexOn hs fun r => by positivity
    refine (stronglyConvexE_coe_iff 0 _).2 ?_
    rw [strongConvexOn_iff_convex]
    simpa using hc
  · rw [hφfun, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    have : (fun x : E d => φc d (SA.card * (L / 4) / n) μ R q x - μ / 2 * ‖x‖ ^ 2) =
        quad (u d) (SA.card * (L / 4) / n) (range (d + 1)) (wch d q) (cch R) := by
      funext x; simp [φc]
    rw [this]
    exact quad_convex _ hα _ _ _ (wch_nonneg hq1)
  · rw [hφfun]
    exact EReal.coe_le_coe_iff.2 (φc_xstar_le hd hα hμ.le hq1 hq y)
  · rw [chainInst_φstar hm hdisj hcard hd hL hμ.le hq1 hq, chainInst_φ hm hdisj hcard]
    have h0 : (chainInst n m d SA SB L μ R q).x0 = 0 := rfl
    rw [h0, φc_zero hd, ← EReal.coe_sub]
    refine EReal.coe_le_coe_iff.2 ?_
    have : 0 ≤ φc d (SA.card * (L / 4) / n) μ R q (xstar d R q) := by
      unfold φc
      have := quad_nonneg (u d) hα (range (d + 1)) (wch d q) (cch R) (wch_nonneg hq1)
        (xstar d R q)
      positivity
    linarith

end Chain
end ULB63
