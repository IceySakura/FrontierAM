import Theory.Blocks
import Theory.LowerChain

/-!
# The data lower bound: `m` chains held by the components of one node
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63
namespace Blocks

variable {n m c b d : ℕ}

/-- Links held by node `i`: all of them at `i0`, none elsewhere. -/
def Si (i0 i : Fin n) (b : ℕ) : Finset ℕ := if i = i0 then range (b + 1) else ∅

lemma Si_subset (i0 i : Fin n) (b : ℕ) : Si i0 i b ⊆ range (b + 1) := by
  unfold Si; split_ifs
  · exact subset_rfl
  · exact empty_subset _

/-- The data instance: node `i0` holds `m` components, component `ξ` being a chain on block `ξ`. -/
def blockInst (n m b : ℕ) (i0 : Fin n) (L μ R q : ℝ) : Instance n m (m * b) where
  F i ξ := bq L (Si i0 i b) (Chain.wch b q) (Chain.cch R) ξ
  h x := ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1 + η * μ)⁻¹ • v

/-- Knowledge sets: the first `k ξ` coordinates of every block `ξ`. -/
def BP (k : Fin m → ℕ) : Set (Fin (m * b)) :=
  {p | (((code m b).symm p).2 : ℕ) < k ((code m b).symm p).1}

lemma BP_mono {k k' : Fin m → ℕ} (h : k ≤ k') : BP (b := b) k ⊆ BP k' :=
  fun _ hp => lt_of_lt_of_le hp (h _)

lemma supp_P_of_BP {k : Fin m → ℕ} {x : E (m * b)} (hx : supp x ⊆ BP k) (ξ : Fin m) :
    supp (P ξ x) ⊆ Pre b (k ξ) := by
  intro j hj
  have hmem : code m b (ξ, j) ∈ BP k := hx (by simpa [supp] using hj)
  simp only [BP, Set.mem_setOf_eq, Equiv.symm_apply_apply] at hmem
  exact hmem

lemma supp_Em_of {k : Fin m → ℕ} (ξ : Fin m) {y : E b} (hy : supp y ⊆ Pre b (k ξ)) :
    supp (Em ξ y) ⊆ BP k := by
  intro p hp
  simp only [supp, Set.mem_setOf_eq, Em_apply] at hp
  split_ifs at hp with h
  · have := hy hp
    simp only [Pre, Set.mem_setOf_eq] at this
    simp only [BP, Set.mem_setOf_eq, h]
    exact this
  · exact absurd rfl hp

lemma convexOn_finset_sum {ι : Type*} (s : Finset ι) (g : ι → E d → ℝ)
    (h : ∀ i ∈ s, ConvexOn ℝ Set.univ (g i)) : ConvexOn ℝ Set.univ (fun x => ∑ i ∈ s, g i x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using convexOn_const (0 : ℝ) convex_univ
  | insert a s ha ih =>
    simp only [Finset.sum_insert ha]
    exact (h a (mem_insert_self a s)).add (ih fun i hi => h i (mem_insert_of_mem hi))

/-- `x⋆` of the data instance: `x⋆` on every block. -/
def xstarB (m b : ℕ) (R q : ℝ) : E (m * b) :=
  WithLp.toLp 2 (fun p => R * q ^ ((((code m b).symm p).2 : ℕ) + 1))

lemma P_xstarB (ξ : Fin m) (R q : ℝ) : P ξ (xstarB m b R q) = Chain.xstar b R q := by
  ext j; simp only [P_apply, xstarB, Chain.xstar, PiLp.toLp_apply, Equiv.symm_apply_apply]

lemma blockInst_f (hm : 1 ≤ m) (hn : 1 ≤ n) (i0 : Fin n) (L μ R q : ℝ) (x : E (m * b)) :
    (blockInst n m b i0 L μ R q).f x =
      ∑ ξ, Chain.quad (Chain.u b) (L / (4 * n * m)) (range (b + 1)) (Chain.wch b q) (Chain.cch R)
        (P ξ x) := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast (by omega : m ≠ 0)
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
  simp only [Instance.f, blockInst]
  have h1 : ∀ i : Fin n, 1 / (m : ℝ) * ∑ ξ, bq L (Si i0 i b) (Chain.wch b q) (Chain.cch R) ξ x =
      if i = i0 then 1 / (m : ℝ) * ∑ ξ, bq L (range (b + 1)) (Chain.wch b q) (Chain.cch R) ξ x
      else 0 := by
    intro i
    unfold Si
    split_ifs with h
    · rfl
    · simp [bq, Chain.quad_empty]
  rw [Finset.sum_congr rfl fun i _ => h1 i, Finset.sum_ite_eq' Finset.univ i0, if_pos (mem_univ _),
    Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun ξ _ => ?_
  simp only [bq]
  rw [Chain.quad_smul, Chain.quad_smul]
  congr 1
  field_simp

lemma blockInst_φ (hm : 1 ≤ m) (hn : 1 ≤ n) (i0 : Fin n) (L μ R q : ℝ) (x : E (m * b)) :
    (blockInst n m b i0 L μ R q).φ x =
      ((∑ ξ, Chain.φc b (L / (4 * n * m)) μ R q (P ξ x) : ℝ) : EReal) := by
  simp only [Instance.φ, blockInst_f hm hn, Chain.φc, Finset.sum_add_distrib, ← Finset.mul_sum,
    ← norm_sq_eq_sum_P]
  simp [blockInst, EReal.coe_add]

lemma blockInst_φstar (hm : 1 ≤ m) (hn : 1 ≤ n) (i0 : Fin n) {L μ R q : ℝ} (hb : 1 ≤ b)
    (hL : 0 ≤ L) (hμ : 0 ≤ μ) (hq1 : q ≤ 1) (hq : μ * q = L / (4 * n * m) * (1 - q) ^ 2) :
    (blockInst n m b i0 L μ R q).φstar =
      ((∑ _ξ : Fin m, Chain.φc b (L / (4 * n * m)) μ R q (Chain.xstar b R q) : ℝ) : EReal) := by
  have hα : 0 ≤ L / (4 * n * m) := by positivity
  apply le_antisymm
  · refine (iInf_le _ (xstarB m b R q)).trans ?_
    rw [blockInst_φ hm hn]
    simp [P_xstarB]
  · refine le_iInf fun y => ?_
    rw [blockInst_φ hm hn]
    exact EReal.coe_le_coe_iff.2 (Finset.sum_le_sum fun ξ _ =>
      Chain.φc_xstar_le hb hα hμ hq1 hq (P ξ y))

lemma blockInst_gap (hm : 1 ≤ m) (hn : 1 ≤ n) (i0 : Fin n) {L μ R q : ℝ} (hb : 1 ≤ b)
    (hL : 0 ≤ L) (hμ : 0 ≤ μ) (hq1 : q ≤ 1) (hq : μ * q = L / (4 * n * m) * (1 - q) ^ 2)
    (x : E (m * b)) :
    (blockInst n m b i0 L μ R q).gap x = ENNReal.ofReal
      (∑ ξ, (Chain.φc b (L / (4 * n * m)) μ R q (P ξ x) -
        Chain.φc b (L / (4 * n * m)) μ R q (Chain.xstar b R q))) := by
  unfold Instance.gap
  rw [blockInst_φ hm hn, blockInst_φstar hm hn i0 hb hL hμ hq1 hq, ← EReal.coe_sub,
    EReal.toENNReal_of_ne_top (EReal.coe_ne_top _), EReal.toReal_coe, Finset.sum_sub_distrib]

theorem blockInst_inClass (hm : 1 ≤ m) (hn : 1 ≤ n) (i0 : Fin n) {L μ Δ R q : ℝ} (hb : 1 ≤ b)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hq : μ * q = L / (4 * n * m) * (1 - q) ^ 2)
    (hR : m * (L / (4 * n * m) / 2 * R ^ 2 * q) ≤ Δ) :
    (blockInst n m b i0 L μ R q).InClass L μ Δ := by
  have hα : 0 ≤ L / (4 * n * m) := by positivity
  have hφfun : (blockInst n m b i0 L μ R q).φ =
      fun x => ((∑ ξ, Chain.φc b (L / (4 * n * m)) μ R q (P ξ x) : ℝ) : EReal) :=
    funext fun x => blockInst_φ hm hn i0 L μ R q x
  have hfconv : ConvexOn ℝ Set.univ (blockInst n m b i0 L μ R q).f := by
    have : (blockInst n m b i0 L μ R q).f = fun x => ∑ ξ, Chain.quad (Chain.u b) (L / (4 * n * m))
        (range (b + 1)) (Chain.wch b q) (Chain.cch R) (P ξ x) :=
      funext fun x => blockInst_f hm hn i0 L μ R q x
    rw [this]
    refine convexOn_finset_sum _ _ fun ξ _ => ?_
    have h := Chain.quad_convex (Chain.u b) hα (range (b + 1)) (Chain.wch b q) (Chain.cch R)
      (Chain.wch_nonneg hq1)
    have := h.comp_linearMap (Plin ξ)
    rw [Set.preimage_univ] at this
    exact this
  refine
    { smooth := fun i ξ => ?_
      f_convex := hfconv
      h_ne_bot := fun _ => EReal.coe_ne_bot _
      h_proper := ⟨0, EReal.coe_ne_top _⟩
      h_closed := (continuous_coe_real_ereal.comp (by fun_prop)).lowerSemicontinuous
      h_convex := ?_
      prox_exact := fun v η hη => isProx_halfSq hμ.le hη v
      φ_strongly_convex := ?_
      minimizer := ⟨xstarB m b R q, fun y => ?_⟩
      initial_gap := ?_ }
  · exact lsmooth_bq hL _ (Si_subset i0 i b) _ _
      (fun k _ => ⟨Chain.wch_nonneg hq1 k, Chain.wch_le_one hq0 k⟩) ξ
  · have hs : StrongConvexOn Set.univ μ (fun x : E (m * b) => μ / 2 * ‖x‖ ^ 2) := by
      rw [strongConvexOn_iff_convex]; simpa using convexOn_const (0 : ℝ) convex_univ
    have hc : ConvexOn ℝ Set.univ (fun x : E (m * b) => μ / 2 * ‖x‖ ^ 2) :=
      UniformConvexOn.convexOn hs fun r => by positivity
    refine (stronglyConvexE_coe_iff 0 _).2 ?_
    rw [strongConvexOn_iff_convex]
    simpa using hc
  · rw [hφfun, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    have : (fun x : E (m * b) => (∑ ξ, Chain.φc b (L / (4 * n * m)) μ R q (P ξ x)) -
        μ / 2 * ‖x‖ ^ 2) = (blockInst n m b i0 L μ R q).f := by
      funext x
      rw [blockInst_f hm hn]
      simp only [Chain.φc, Finset.sum_add_distrib, ← Finset.mul_sum, ← norm_sq_eq_sum_P]
      ring
    rw [this]
    exact hfconv
  · rw [hφfun]
    refine EReal.coe_le_coe_iff.2 ?_
    simp only [P_xstarB]
    exact Finset.sum_le_sum fun ξ _ => Chain.φc_xstar_le hb hα hμ.le hq1 hq (P ξ y)
  · rw [blockInst_φstar hm hn i0 hb hL hμ.le hq1 hq, blockInst_φ hm hn]
    have h0 : (blockInst n m b i0 L μ R q).x0 = 0 := rfl
    rw [h0, ← EReal.coe_sub]
    refine EReal.coe_le_coe_iff.2 ?_
    simp only [P_zero, Chain.φc_zero hb, Chain.φc_xstar_val hb hq, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    nlinarith

end Blocks
end ULB63
