import Theory.LowerBound

/-!
# Strongly convex components

The same objective `φ = Phi`, split differently: every component carries `(μ/2)‖·‖²` (so every
component is `μ`-strongly convex), `h = 0`, `prox = id`. The smoothness budget becomes
`4 N α + μ ≤ L`.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Chain

variable {n m K : ℕ}

/-- The instance with strongly convex components. -/
def instSC (n m K : ℕ) (hn : 0 < n) (q α β μ : ℝ) : Instance n m (m * K) where
  F i ξ x := (inst n m K hn q α β μ).F i ξ x + μ / 2 * ‖x‖ ^ 2
  h _ := ((0 : ℝ) : EReal)
  x0 := 0
  prox v _ := v

/-- The hard instance with strongly convex components. -/
def hardSC (n m K : ℕ) (hn : 0 < n) (μ Δ q : ℝ) : Instance n m (m * K) :=
  instSC n m K hn q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ

lemma avg_add_const (hn : 0 < n) (hm : 0 < m) (a : Fin n → Fin m → ℝ) (b : ℝ) :
    (1 / n : ℝ) * ∑ i, (1 / m : ℝ) * ∑ ξ, (a i ξ + b) =
      (1 / n : ℝ) * ∑ i, (1 / m : ℝ) * ∑ ξ, a i ξ + b := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_add, Finset.mul_sum]
  field_simp

lemma instSC_f (hn : 0 < n) (hm : 0 < m) (q α β μ : ℝ) (x : E (m * K)) :
    (instSC n m K hn q α β μ).f x = Phi q α β μ x := by
  have h1 : (instSC n m K hn q α β μ).f x = (inst n m K hn q α β μ).f x + μ / 2 * ‖x‖ ^ 2 := by
    unfold Instance.f
    exact avg_add_const hn hm (fun i ξ => (inst n m K hn q α β μ).F i ξ x) _
  rw [h1, inst_f hn hm]
  rfl

lemma instSC_φ (hn : 0 < n) (hm : 0 < m) (q α β μ : ℝ) (x : E (m * K)) :
    (instSC n m K hn q α β μ).φ x = ((Phi q α β μ x : ℝ) : EReal) := by
  unfold Instance.φ
  rw [instSC_f hn hm]
  simp only [instSC, EReal.coe_zero, add_zero]

lemma hardSC_φ (hn : 0 < n) (hm : 0 < m) (μ Δ q : ℝ) (x : E (m * K)) :
    (hardSC n m K hn μ Δ q).φ x =
      ((Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x : ℝ) : EReal) :=
  instSC_φ hn hm _ _ _ _ x

lemma hardSC_φstar (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) : (hardSC n m K hn μ Δ q).φstar = ((-Δ : ℝ) : EReal) := by
  unfold Instance.φstar
  apply le_antisymm
  · refine (iInf_le _ (xstar m K (cst m μ Δ q) q)).trans ?_
    rw [hardSC_φ hn hm, Phi_xstar_eq hm hK hμ hΔ hq1]
  · refine le_iInf fun y => ?_
    rw [hardSC_φ hn hm, EReal.coe_le_coe_iff, ← Phi_xstar_eq (K := K) hm hK hμ hΔ hq1]
    have := Phi_ge (m := m) (c := cst m μ Δ q) hμ.le hK hq hq1 y
    nlinarith [sq_nonneg ‖y - xstar m K (cst m μ Δ q) q‖]

lemma hardSC_gap (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) (x : E (m * K)) :
    (hardSC n m K hn μ Δ q).gap x =
      ENNReal.ofReal (Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x + Δ) := by
  unfold Instance.gap
  rw [hardSC_φ hn hm, hardSC_φstar hn hm hK hμ hΔ hq hq1, ← EReal.coe_sub,
    EReal.real_coe_toENNReal, sub_neg_eq_add]

lemma hasGradientAt_sqn {d : ℕ} (μ : ℝ) (x : E d) :
    HasGradientAt (fun x : E d => μ / 2 * ‖x‖ ^ 2) (μ • x) x := by
  apply hasGradientAt_of_sq_bound (C := |μ| / 2)
  intro v
  have : μ / 2 * ‖x + v‖ ^ 2 - μ / 2 * ‖x‖ ^ 2 - ⟪μ • x, v⟫ = μ / 2 * ‖v‖ ^ 2 := by
    rw [norm_add_sq_real, real_inner_smul_left]; ring
  rw [this, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ ‖v‖ ^ 2), abs_div,
    abs_two]

/-- The gradient of a component of `hardSC`. -/
lemma hardSC_hasGradientAt₀ (hn : 0 < n) (hK : 0 < K) {μ Δ q : ℝ} (hq : 0 ≤ q) (hq1 : q < 1)
    (ξ : Fin m) (x : E (m * K)) :
    HasGradientAt ((hardSC n m K hn μ Δ q).F ⟨0, hn⟩ ξ)
      (gcomp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ x + μ • x) x := by
  have hF : (hardSC n m K hn μ Δ q).F ⟨0, hn⟩ ξ = fun y =>
      comp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) ξ y + μ / 2 * ‖y‖ ^ 2 := by
    funext y; simp [hardSC, instSC, inst]
  rw [hF]
  have h1 := hasGradientAt_comp hq hq1.le ((n : ℝ) * m) (alpha μ q)
    (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ x
  have h2 := hasGradientAt_sqn μ x
  rw [hasGradientAt_iff_hasFDerivAt] at h1 h2 ⊢
  rw [map_add]
  exact h1.add h2

lemma hardSC_hasGradientAt (hn : 0 < n) {μ Δ q : ℝ} (i : Fin n) (hi : i ≠ ⟨0, hn⟩) (ξ : Fin m)
    (x : E (m * K)) : HasGradientAt ((hardSC n m K hn μ Δ q).F i ξ) (μ • x) x := by
  have hF : (hardSC n m K hn μ Δ q).F i ξ = fun y => μ / 2 * ‖y‖ ^ 2 := by
    funext y; simp [hardSC, instSC, inst, hi]
  rw [hF]
  exact hasGradientAt_sqn μ x

lemma supp_add_subset {d : ℕ} {S : Set (Fin d)} {x y : E d} (hx : supp x ⊆ S) (hy : supp y ⊆ S) :
    supp (x + y) ⊆ S := by
  intro c hc
  by_contra hcS
  apply hc
  have h1 : x c = 0 := by by_contra h; exact hcS (hx h)
  have h2 : y c = 0 := by by_contra h; exact hcS (hy h)
  simp [h1, h2]

lemma hardSC_props (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) : HardProps (hardSC n m K hn μ Δ q) hn μ Δ q where
  gap_eq := hardSC_gap hn hm hK hμ hΔ.le hq hq1
  x0_supp := by
    show supp (0 : E (m * K)) ⊆ _
    rw [Support.supp_zero]; exact Set.empty_subset _
  grad₀_supp ξ x k hx := by
    rw [(hardSC_hasGradientAt₀ hn hK hq hq1 ξ x).gradient]
    refine supp_add_subset (gcomp_supp hK hx ξ) ((supp_smul_subset _ _).trans
      (hx.trans (Box_mono ?_)))
    intro j; simp
  grad_supp i hi ξ x k hx := by
    rw [(hardSC_hasGradientAt hn i hi ξ x).gradient]
    exact (supp_smul_subset _ _).trans hx
  prox_supp v η k hv := hv

/-- Every component of `hardSC` is `μ`-strongly convex. -/
lemma hardSC_comp_strongConvex (hn : 0 < n) {μ Δ q : ℝ} (hμ : 0 < μ) (hq : 0 ≤ q) (hq1 : q < 1)
    (i : Fin n) (ξ : Fin m) : StrongConvexOn Set.univ μ ((hardSC n m K hn μ Δ q).F i ξ) := by
  rw [strongConvexOn_iff_convex]
  have : (fun x => (hardSC n m K hn μ Δ q).F i ξ x - μ / 2 * ‖x‖ ^ 2) =
      (hard n m K hn μ Δ q).F i ξ := by
    funext x; simp [hardSC, instSC, hard]
  rw [this]
  exact hard_comp_convex hn hμ hq hq1 i ξ

/-- The hard instance with strongly convex components belongs to `𝓕_{L,μ,Δ,m}`. -/
theorem hardSC_inClass (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {L μ Δ q : ℝ} (hμ : 0 < μ)
    (hΔ : 0 < Δ) (hq : 0 ≤ q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ (L - μ) * (1 - q) ^ 2) :
    (hardSC n m K hn μ Δ q).InClass L μ Δ where
  smooth i ξ := by
    have h1q : 0 < (1 - q) ^ 2 := by
      have : 0 < 1 - q := by linarith
      positivity
    have hNα : 4 * ((n : ℝ) * m) * alpha μ q + μ ≤ L := by
      unfold alpha
      rw [show 4 * ((n : ℝ) * m) * (μ * q / (1 - q) ^ 2) = (4 * ((n : ℝ) * m) * μ * q) / (1 - q) ^ 2
        by ring]
      have := (div_le_iff₀ h1q).2 hqL
      linarith
    have hα : 0 ≤ alpha μ q := by unfold alpha; positivity
    have hN : (0 : ℝ) ≤ n * m := by positivity
    by_cases hi : i = ⟨0, hn⟩
    · subst hi
      refine ⟨fun x => (hardSC_hasGradientAt₀ hn hK hq hq1 ξ x).differentiableAt, fun x y => ?_⟩
      rw [(hardSC_hasGradientAt₀ hn hK hq hq1 ξ x).gradient,
        (hardSC_hasGradientAt₀ hn hK hq hq1 ξ y).gradient]
      have hlip := (comp_smooth (K := K) hq hq1.le hN hα le_rfl hK ξ
        (β := μ * cst m μ Δ q / (1 - q) ^ 2)).2 x y
      rw [gradient_comp hq hq1.le _ _ _ hK, gradient_comp hq hq1.le _ _ _ hK] at hlip
      calc ‖gcomp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ x + μ • x -
            (gcomp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ y + μ • y)‖
          = ‖(gcomp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ x -
              gcomp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) hK ξ y) +
              μ • (x - y)‖ := by congr 1; rw [smul_sub]; abel
        _ ≤ 4 * ((n : ℝ) * m) * alpha μ q * ‖x - y‖ + μ * ‖x - y‖ := by
          refine (norm_add_le _ _).trans (add_le_add hlip ?_)
          rw [norm_smul, Real.norm_eq_abs, abs_of_pos hμ]
        _ ≤ L * ‖x - y‖ := by nlinarith [norm_nonneg (x - y)]
    · refine ⟨fun x => (hardSC_hasGradientAt hn i hi ξ x).differentiableAt, fun x y => ?_⟩
      rw [(hardSC_hasGradientAt hn i hi ξ x).gradient, (hardSC_hasGradientAt hn i hi ξ y).gradient,
        ← smul_sub, norm_smul, Real.norm_eq_abs, abs_of_pos hμ]
      have : μ ≤ L := by nlinarith
      exact mul_le_mul_of_nonneg_right this (norm_nonneg _)
  f_convex := by
    have : (hardSC n m K hn μ Δ q).f = Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ :=
      funext (instSC_f hn hm _ _ _ _)
    rw [this]
    have hs : StrongConvexOn Set.univ μ (Phi (m := m) (K := K) q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ) := by
      rw [strongConvexOn_iff_convex]
      have h2 : (fun x : E (m * K) => Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x -
          μ / 2 * ‖x‖ ^ 2) = fun x => ∑ ξ : Fin m,
            (alpha μ q / 2 * Q q ξ x - (μ * cst m μ Δ q / (1 - q) ^ 2) * xc x ξ 0) := by
        funext x; unfold Phi; ring
      rw [h2]
      exact convex_chainPart hq1.le (by unfold alpha; positivity)
    exact UniformConvexOn.convexOn hs fun r => by positivity
  h_ne_bot _ := EReal.coe_ne_bot _
  h_proper := ⟨0, EReal.coe_ne_top _⟩
  h_closed := lowerSemicontinuous_const
  h_convex := by
    refine (stronglyConvexE_coe_iff 0 (fun _ : E (m * K) => (0 : ℝ))).2 ?_
    rw [strongConvexOn_iff_convex]
    simpa using convexOn_const (0 : ℝ) convex_univ
  prox_exact v η hη := by
    intro u
    show ((0 : ℝ) : EReal) + ((‖v - v‖ ^ 2 / (2 * η) : ℝ) : EReal) ≤
      ((0 : ℝ) : EReal) + ((‖u - v‖ ^ 2 / (2 * η) : ℝ) : EReal)
    rw [← EReal.coe_add, ← EReal.coe_add, EReal.coe_le_coe_iff]
    simp only [sub_self, norm_zero]
    have : 0 ≤ ‖u - v‖ ^ 2 / (2 * η) := by positivity
    simpa using this
  φ_strongly_convex := by
    have : (hardSC n m K hn μ Δ q).φ =
        fun x => ((Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x : ℝ) : EReal) :=
      funext (hardSC_φ hn hm μ Δ q)
    rw [this, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    have h2 : (fun x : E (m * K) => Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x -
        μ / 2 * ‖x‖ ^ 2) = fun x => ∑ ξ : Fin m,
          (alpha μ q / 2 * Q q ξ x - (μ * cst m μ Δ q / (1 - q) ^ 2) * xc x ξ 0) := by
      funext x; unfold Phi; ring
    rw [h2]
    exact convex_chainPart hq1.le (by unfold alpha; positivity)
  minimizer := ⟨xstar m K (cst m μ Δ q) q, fun y => by
    rw [hardSC_φ hn hm, hardSC_φ hn hm, EReal.coe_le_coe_iff]
    have := Phi_ge (m := m) (c := cst m μ Δ q) hμ.le hK hq hq1 y
    nlinarith [sq_nonneg ‖y - xstar m K (cst m μ Δ q) q‖]⟩
  initial_gap := by
    rw [hardSC_φstar hn hm hK hμ hΔ.le hq hq1]
    show (hardSC n m K hn μ Δ q).φ 0 - _ ≤ _
    rw [hardSC_φ hn hm, Phi_zero, ← EReal.coe_sub]
    exact EReal.coe_le_coe_iff.2 (by linarith)

end Chain
end ULB63
