import Theory.Chain
import Theory.Support

/-!
# The separate-chains lower bound

Assembly: the hard instance `hard n m K μ Δ q` is in the class (with convex components), every
admissible algorithm keeps the averaged output in a box `Box k` with `∑ k ≤ t` after `t`
iterations, and on such a box the gap is at least `Δ (1-q)² / m · ∑_ξ q^{2 k ξ}`.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Chain

/-- The scale `c = (1-q) √(2Δ/(mμ))` of the minimiser. -/
def cst (m : ℕ) (μ Δ q : ℝ) : ℝ := (1 - q) * Real.sqrt (2 * Δ / (m * μ))

/-- The hard instance with chains of length `K`. -/
def hard (n m K : ℕ) (hn : 0 < n) (μ Δ q : ℝ) : Instance n m (m * K) :=
  inst n m K hn q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ

variable {n m K : ℕ}

lemma cst_sq (hm : 0 < m) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ) :
    μ * cst m μ Δ q ^ 2 = 2 * Δ * (1 - q) ^ 2 / m := by
  unfold cst
  rw [mul_pow, Real.sq_sqrt (by positivity)]
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  field_simp

lemma Phi_xstar_eq (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (hq1 : q < 1) :
    Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ (xstar m K (cst m μ Δ q) q) = -Δ := by
  rw [Phi_xstar hK hq1, neg_inj]
  have h := cst_sq hm hμ hΔ (q := q)
  have h1q : (1 - q) ≠ 0 := by intro h0; linarith
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have h' : (m : ℝ) * (μ * cst m μ Δ q ^ 2) = 2 * Δ * (1 - q) ^ 2 := by
    rw [h]; field_simp
  field_simp
  linear_combination h'

lemma hard_φ (hn : 0 < n) (hm : 0 < m) (μ Δ q : ℝ) (x : E (m * K)) :
    (hard n m K hn μ Δ q).φ x =
      ((Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x : ℝ) : EReal) :=
  inst_φ hn hm _ _ _ _ x

lemma hard_φstar (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) : (hard n m K hn μ Δ q).φstar = ((-Δ : ℝ) : EReal) := by
  unfold Instance.φstar
  apply le_antisymm
  · refine (iInf_le _ (xstar m K (cst m μ Δ q) q)).trans ?_
    rw [hard_φ hn hm, Phi_xstar_eq hm hK hμ hΔ hq1]
  · refine le_iInf fun y => ?_
    rw [hard_φ hn hm, EReal.coe_le_coe_iff, ← Phi_xstar_eq (K := K) hm hK hμ hΔ hq1]
    have := Phi_ge (m := m) (c := cst m μ Δ q) hμ.le hK hq hq1 y
    nlinarith [sq_nonneg ‖y - xstar m K (cst m μ Δ q) q‖]

lemma hard_gap (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) (x : E (m * K)) :
    (hard n m K hn μ Δ q).gap x =
      ENNReal.ofReal (Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x + Δ) := by
  unfold Instance.gap
  rw [hard_φ hn hm, hard_φstar hn hm hK hμ hΔ hq hq1, ← EReal.coe_sub, EReal.real_coe_toENNReal,
    sub_neg_eq_add]

lemma convex_chainPart' (hn : 0 < n) (hm : 0 < m) {μ Δ q : ℝ} (hμ : 0 < μ) (hq : 0 ≤ q)
    (hq1 : q < 1) : ConvexOn ℝ Set.univ (hard n m K hn μ Δ q).f := by
  have : (hard n m K hn μ Δ q).f = fun x => ∑ ξ : Fin m,
      (alpha μ q / 2 * Q q ξ x - (μ * cst m μ Δ q / (1 - q) ^ 2) * xc x ξ 0) :=
    funext (inst_f hn hm _ _ _ _)
  rw [this]
  exact convex_chainPart hq1.le (by unfold alpha; positivity)

/-- Every component of the hard instance is convex. -/
lemma hard_comp_convex (hn : 0 < n) {μ Δ q : ℝ} (hμ : 0 < μ) (hq : 0 ≤ q) (hq1 : q < 1)
    (i : Fin n) (ξ : Fin m) : ConvexOn ℝ Set.univ ((hard n m K hn μ Δ q).F i ξ) := by
  simp only [hard, inst]
  split_ifs
  · -- `N (α/2 Q_ξ - β x_{ξ,0})` with `N, α ≥ 0`
    refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
    obtain rfl : b = 1 - a := by linarith
    simp only [comp, smul_eq_mul]
    rw [Q_convex_comb, xc_add, xc_smul, xc_smul]
    have hN : (0 : ℝ) ≤ n * m := by positivity
    have hα : 0 ≤ alpha μ q := by unfold alpha; positivity
    have : 0 ≤ (n * m : ℝ) * (alpha μ q / 2 * (a * (1 - a) * Q q ξ (x - y))) :=
      mul_nonneg hN (mul_nonneg (by positivity) (mul_nonneg (mul_nonneg ha hb) (Q_nonneg hq1.le ξ _)))
    nlinarith
  · exact convexOn_const 0 convex_univ

/-- The hard instance belongs to `𝓕_{L,μ,Δ,m}`. -/
theorem hard_inClass (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {L μ Δ q : ℝ} (hμ : 0 < μ)
    (hμL : μ ≤ L) (hΔ : 0 < Δ) (hq : 0 ≤ q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    (hard n m K hn μ Δ q).InClass L μ Δ where
  smooth i ξ := by
    simp only [hard, inst]
    split_ifs with hi
    · apply comp_smooth hq hq1.le (by positivity) (by unfold alpha; positivity) _ hK ξ
      unfold alpha
      have h1q : 0 < (1 - q) ^ 2 := by
        have : 0 < 1 - q := by linarith
        positivity
      rw [show 4 * ((n : ℝ) * m) * (μ * q / (1 - q) ^ 2) = (4 * ((n : ℝ) * m) * μ * q) / (1 - q) ^ 2
        by ring, div_le_iff₀ h1q]
      exact hqL
    · refine ⟨differentiable_const _, fun x y => ?_⟩
      have hL : 0 ≤ L := by linarith
      have h0 : ∀ z, gradient (fun _ : E (m * K) => (0 : ℝ)) z = 0 := fun z => gradient_const z 0
      rw [h0, h0, sub_self, norm_zero]
      exact mul_nonneg hL (norm_nonneg _)
  f_convex := convex_chainPart' hn hm hμ hq hq1
  h_ne_bot _ := EReal.coe_ne_bot _
  h_proper := ⟨0, EReal.coe_ne_top _⟩
  h_closed := (continuous_coe_real_ereal.comp (by fun_prop)).lowerSemicontinuous
  h_convex := by
    have hs : StrongConvexOn Set.univ μ (fun x : E (m * K) => μ / 2 * ‖x‖ ^ 2) := by
      rw [strongConvexOn_iff_convex]; simpa using convexOn_const (0 : ℝ) convex_univ
    have hc : ConvexOn ℝ Set.univ (fun x : E (m * K) => μ / 2 * ‖x‖ ^ 2) :=
      UniformConvexOn.convexOn hs fun r => by positivity
    refine (stronglyConvexE_coe_iff 0 _).2 ?_
    rw [strongConvexOn_iff_convex]
    simpa using hc
  prox_exact v η hη := isProx_halfSq hμ.le hη v
  φ_strongly_convex := by
    have : (hard n m K hn μ Δ q).φ =
        fun x => ((Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x : ℝ) : EReal) :=
      funext (hard_φ hn hm μ Δ q)
    rw [this, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    have h2 : (fun x : E (m * K) => Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x -
        μ / 2 * ‖x‖ ^ 2) = fun x => ∑ ξ : Fin m,
          (alpha μ q / 2 * Q q ξ x - (μ * cst m μ Δ q / (1 - q) ^ 2) * xc x ξ 0) := by
      funext x; unfold Phi; ring
    rw [h2]
    exact convex_chainPart hq1.le (by unfold alpha; positivity)
  minimizer := ⟨xstar m K (cst m μ Δ q) q, fun y => by
    rw [hard_φ hn hm, hard_φ hn hm, EReal.coe_le_coe_iff]
    have := Phi_ge (m := m) (c := cst m μ Δ q) hμ.le hK hq hq1 y
    nlinarith [sq_nonneg ‖y - xstar m K (cst m μ Δ q) q‖]⟩
  initial_gap := by
    rw [hard_φstar hn hm hK hμ hΔ.le hq hq1]
    show (hard n m K hn μ Δ q).φ 0 - _ ≤ _
    rw [hard_φ hn hm, Phi_zero, ← EReal.coe_sub]
    exact EReal.coe_le_coe_iff.2 (by linarith)

/-- Supports of the oracle answers of the hard instance. -/
lemma hard_x0_supp (hn : 0 < n) (μ Δ q : ℝ) :
    supp (hard n m K hn μ Δ q).x0 ⊆ Box 0 := by
  show supp (0 : E (m * K)) ⊆ _
  rw [Support.supp_zero]; exact Set.empty_subset _

lemma hard_grad₀_supp (hn : 0 < n) (hK : 0 < K) {μ Δ q : ℝ} (hq : 0 ≤ q) (hq1 : q < 1)
    (ξ : Fin m) (x : E (m * K)) (k : Fin m → ℕ) (hx : supp x ⊆ Box k) :
    supp (gradient ((hard n m K hn μ Δ q).F ⟨0, hn⟩ ξ) x) ⊆ Box (k + Pi.single ξ 1) := by
  have : (hard n m K hn μ Δ q).F ⟨0, hn⟩ ξ =
      comp q ((n : ℝ) * m) (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) ξ := by
    simp [hard, inst]
  rw [this, gradient_comp hq hq1.le _ _ _ hK]
  exact gcomp_supp hK hx ξ

lemma hard_grad_supp (hn : 0 < n) {μ Δ q : ℝ} (i : Fin n) (hi : i ≠ ⟨0, hn⟩) (ξ : Fin m)
    (x : E (m * K)) (k : Fin m → ℕ) :
    supp (gradient ((hard n m K hn μ Δ q).F i ξ) x) ⊆ Box k := by
  have : (hard n m K hn μ Δ q).F i ξ = fun _ => 0 := by
    simp [hard, inst, hi]
  rw [this]
  have h0 : gradient (fun _ : E (m * K) => (0 : ℝ)) x = 0 := gradient_const x 0
  rw [h0, Support.supp_zero]
  exact Set.empty_subset _

lemma hard_prox_supp (hn : 0 < n) {μ Δ q : ℝ} (v : E (m * K)) (η : ℝ) (k : Fin m → ℕ)
    (hv : supp v ⊆ Box k) : supp ((hard n m K hn μ Δ q).prox v η) ⊆ Box k :=
  (supp_smul_subset _ v).trans hv

/-- What the lower-bound argument needs from an instance: its gap is the chain gap, and its oracles
respect the boxes (only node `0` makes progress, one position per gradient query). -/
structure HardProps {n : ℕ} (I : Instance n m (m * K)) (hn : 0 < n) (μ Δ q : ℝ) : Prop where
  gap_eq : ∀ x, I.gap x =
    ENNReal.ofReal (Phi q (alpha μ q) (μ * cst m μ Δ q / (1 - q) ^ 2) μ x + Δ)
  x0_supp : supp I.x0 ⊆ Box 0
  grad₀_supp : ∀ ξ x k, supp x ⊆ Box k →
    supp (gradient (I.F ⟨0, hn⟩ ξ) x) ⊆ Box (k + Pi.single ξ 1)
  grad_supp : ∀ i, i ≠ ⟨0, hn⟩ → ∀ ξ x k, supp x ⊆ Box k → supp (gradient (I.F i ξ) x) ⊆ Box k
  prox_supp : ∀ v η k, supp v ⊆ Box k → supp (I.prox v η) ⊆ Box k

lemma hard_props (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {μ Δ q : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ)
    (hq : 0 ≤ q) (hq1 : q < 1) : HardProps (hard n m K hn μ Δ q) hn μ Δ q where
  gap_eq := hard_gap hn hm hK hμ hΔ.le hq hq1
  x0_supp := hard_x0_supp hn μ Δ q
  grad₀_supp ξ x k hx := hard_grad₀_supp hn hK hq hq1 ξ x k hx
  grad_supp i hi ξ x k _ := hard_grad_supp hn i hi ξ x k
  prox_supp v η k hv := hard_prox_supp hn v η k hv

/-- Along any run of length `t < K`, the gap of the averaged output is at least
`Δ (1-q)² / m · ∑_ξ q^{2 k ξ}` for a progress vector `k` with `∑ k ≤ t`. -/
theorem gap_along_run (hn : 0 < n) (hm : 0 < m) (hK : 0 < K) {c : ℕ} {L μ Δ ρ q : ℝ}
    (hμ : 0 < μ) (hΔ : 0 < Δ) (hq : 0 ≤ q) (hq1 : q < 1)
    (I : Instance n m (m * K)) (hI : HardProps I hn μ Δ q)
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) (W : Matrix (Fin n) (Fin n) ℝ) (ω : A.Ω)
    (t : ℕ) (ht : t < K) :
    ∃ k : Fin m → ℕ, ∑ ξ, k ξ ≤ t ∧
      ENNReal.ofReal (Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2) ≤
        I.gap (A.avgOutput I W ω t) := by
  obtain ⟨k, hk, hsupp⟩ := Support.avgOutput_supp A hA I W ⟨0, hn⟩
    (Box (K := K)) (fun k k' h => Box_mono h) hI.x0_supp hI.grad₀_supp hI.grad_supp hI.prox_supp
    ω t
  refine ⟨k, hk, ?_⟩
  have hkK : ∀ ξ, k ξ < K := fun ξ =>
    lt_of_le_of_lt ((Finset.single_le_sum (f := k) (fun _ _ => Nat.zero_le _)
      (Finset.mem_univ ξ)).trans hk) ht
  rw [hI.gap_eq]
  apply ENNReal.ofReal_le_ofReal
  have h := Phi_gap_box (c := cst m μ Δ q) hμ.le hK hq hq1 hkK hsupp
  rw [Phi_xstar_eq hm hK hμ hΔ.le hq1, cst_sq hm hμ hΔ.le] at h
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  calc Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2
      = 2 * Δ * (1 - q) ^ 2 / m / 2 * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := by ring
    _ ≤ _ := by linarith

/-- AM–GM: `m q^{2t/m} ≤ ∑_ξ q^{2 k ξ}` when `∑ k ≤ t`. -/
lemma amgm (hm : 0 < m) {q : ℝ} (hq : 0 < q) (hq1 : q < 1) (k : Fin m → ℕ) (t : ℕ)
    (hk : ∑ ξ, k ξ ≤ t) : m * q ^ ((2 * t : ℝ) / m) ≤ ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hw : ∑ _ξ : Fin m, (1 / (m : ℝ)) = 1 := by
    simp [Finset.card_univ]; field_simp
  have hgm := Real.geom_mean_le_arith_mean_weighted (Finset.univ : Finset (Fin m))
    (fun _ => 1 / (m : ℝ)) (fun ξ => (q ^ k ξ) ^ 2) (fun _ _ => by positivity) hw
    (fun _ _ => by positivity)
  have h1 : ∀ ξ, ((q ^ k ξ) ^ 2) ^ (1 / (m : ℝ)) = q ^ ((2 * k ξ : ℝ) / m) := by
    intro ξ
    rw [← pow_mul, ← Real.rpow_natCast, ← Real.rpow_mul hq.le]
    congr 1
    push_cast
    ring
  simp only [h1] at hgm
  rw [← Real.rpow_sum_of_pos hq, ← Finset.mul_sum] at hgm
  have h2 : q ^ ((2 * t : ℝ) / m) ≤ q ^ (∑ ξ : Fin m, (2 * k ξ : ℝ) / m) := by
    apply Real.rpow_le_rpow_of_exponent_ge hq hq1.le
    rw [← Finset.sum_div, ← Finset.mul_sum]
    apply div_le_div_of_nonneg_right _ hm'.le
    have : (∑ ξ : Fin m, (k ξ : ℝ)) ≤ t := by exact_mod_cast hk
    linarith
  have h3 := h2.trans hgm
  calc (m : ℝ) * q ^ ((2 * t : ℝ) / m) ≤ m * (1 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2) :=
        mul_le_mul_of_nonneg_left h3 hm'.le
    _ = ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := by field_simp

end Chain
end ULB63

namespace ULB63
namespace Chain

variable {n m : ℕ}

/-- If the expected gap stays above `ε` before iteration `T`, then `T ≤ T_ε`. -/
lemma le_hittingTime_of {c d : ℕ} (A : Algorithm n m c) (I : Instance n m d)
    (W : Matrix (Fin n) (Fin n) ℝ) (ε : ℝ) (T : ℕ)
    (h : ∀ t, t < T → ENNReal.ofReal ε < A.expectedGap I W t) : (T : ℕ∞) ≤ A.hittingTime I W ε := by
  unfold Algorithm.hittingTime
  refine le_iInf₂ fun t ht => ?_
  by_contra hlt
  push Not at hlt
  have := h t (by exact_mod_cast hlt)
  exact absurd ht (not_le.2 this)

/-- Core bound (chain decay): with chains of length `K > t`, the expected gap after `t`
iterations is at least `Δ (1-q)² q^{2t/m}`. -/
theorem expectedGap_ge_chain (hn : 0 < n) (hm : 0 < m) {K : ℕ} (hK : 0 < K) {c : ℕ}
    {L μ Δ ρ q : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ) (hq : 0 < q) (hq1 : q < 1)
    (I : Instance n m (m * K)) (hI : HardProps I hn μ Δ q)
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) (W : Matrix (Fin n) (Fin n) ℝ)
    (t : ℕ) (ht : t < K) :
    ENNReal.ofReal (Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)) ≤ A.expectedGap I W t := by
  unfold Algorithm.expectedGap
  calc ENNReal.ofReal (Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m))
      = ∫⁻ _ω, ENNReal.ofReal (Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)) ∂A.P := by
        rw [lintegral_const, measure_univ, mul_one]
    _ ≤ _ := by
      refine lintegral_mono fun ω => ?_
      obtain ⟨k, hk, hgap⟩ := gap_along_run hn hm hK hμ hΔ hq.le hq1 I hI A hA W ω t ht
      refine le_trans (ENNReal.ofReal_le_ofReal ?_) hgap
      have h := amgm hm hq hq1 k t hk
      have hm' : (0 : ℝ) < m := by exact_mod_cast hm
      have hc : 0 ≤ Δ * (1 - q) ^ 2 / m := by positivity
      calc Δ * (1 - q) ^ 2 * q ^ ((2 * t : ℝ) / m)
          = Δ * (1 - q) ^ 2 / m * (m * q ^ ((2 * t : ℝ) / m)) := by field_simp
        _ ≤ Δ * (1 - q) ^ 2 / m * ∑ ξ : Fin m, (q ^ k ξ) ^ 2 := mul_le_mul_of_nonneg_left h hc

/-- Core bound (counting): for `q = 0` the components are linear and the expected gap after `t`
iterations is at least `Δ (1 - t/m)`. -/
theorem expectedGap_ge_count (hn : 0 < n) (hm : 0 < m) {K : ℕ} (hK : 0 < K) {c : ℕ}
    {L μ Δ ρ : ℝ} (hμ : 0 < μ) (hΔ : 0 < Δ)
    (I : Instance n m (m * K)) (hI : HardProps I hn μ Δ 0)
    (A : Algorithm n m c) (hA : A.Admissible L μ Δ ρ) (W : Matrix (Fin n) (Fin n) ℝ)
    (t : ℕ) (ht : t < K) :
    ENNReal.ofReal (Δ * (1 - t / m)) ≤ A.expectedGap I W t := by
  unfold Algorithm.expectedGap
  calc ENNReal.ofReal (Δ * (1 - t / m))
      = ∫⁻ _ω, ENNReal.ofReal (Δ * (1 - t / m)) ∂A.P := by
        rw [lintegral_const, measure_univ, mul_one]
    _ ≤ _ := by
      refine lintegral_mono fun ω => ?_
      obtain ⟨k, hk, hgap⟩ := gap_along_run hn hm hK hμ hΔ le_rfl zero_lt_one I hI A hA W ω t ht
      refine le_trans (ENNReal.ofReal_le_ofReal ?_) hgap
      have hm' : (0 : ℝ) < m := by exact_mod_cast hm
      -- `(0 ^ k)² ≥ 1 - k`
      have hterm : ∀ ξ, (1 : ℝ) - k ξ ≤ ((0 : ℝ) ^ k ξ) ^ 2 := by
        intro ξ
        rcases Nat.eq_zero_or_pos (k ξ) with h0 | hpos
        · rw [h0]; simp
        · rw [zero_pow hpos.ne']
          have : (1 : ℝ) ≤ k ξ := by exact_mod_cast hpos
          linarith
      have hsum : (m : ℝ) - t ≤ ∑ ξ : Fin m, ((0 : ℝ) ^ k ξ) ^ 2 := by
        have h1 := Finset.sum_le_sum fun ξ (_ : ξ ∈ Finset.univ) => hterm ξ
        rw [Finset.sum_sub_distrib] at h1
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h1
        have : (∑ ξ : Fin m, (k ξ : ℝ)) ≤ t := by exact_mod_cast hk
        linarith
      have hc : 0 ≤ Δ * (1 - 0) ^ 2 / m := by positivity
      calc Δ * (1 - t / m) = Δ * (1 - 0) ^ 2 / m * (m - t) := by field_simp; ring
        _ ≤ Δ * (1 - 0) ^ 2 / m * ∑ ξ : Fin m, ((0 : ℝ) ^ k ξ) ^ 2 :=
          mul_le_mul_of_nonneg_left hsum hc

/-- **R1 (transient term).** -/
theorem lower_bound_transient_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) * (1 - ε / Δ)⌉₊ : ℕ) : ℕ∞) ≤ A.hittingTime I W ε := by
  set T := ⌈(m : ℝ) * (1 - ε / Δ)⌉₊
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hard n m (T + 1) hn μ Δ 0,
    hard_inClass hn hm hK hμ hμL hΔ le_rfl zero_lt_one (by simp; linarith),
    fun i ξ => hard_comp_convex hn hμ le_rfl zero_lt_one i ξ, fun A hA W => ?_⟩
  refine le_hittingTime_of A _ W ε T fun t ht => ?_
  refine lt_of_lt_of_le ?_ (expectedGap_ge_count hn hm hK hμ hΔ _
    (hard_props hn hm hK hμ hΔ le_rfl zero_lt_one) A hA W t (by omega))
  have htT : (t : ℝ) < m * (1 - ε / Δ) := Nat.lt_ceil.1 ht
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  rw [ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε.le]
  have : ε < Δ * (1 - t / m) := by
    have h1 : (t : ℝ) / m < 1 - ε / Δ := by rw [div_lt_iff₀ hm']; linarith
    have h2 : ε / Δ < 1 - t / m := by linarith
    rw [div_lt_iff₀ hΔ] at h2
    linarith
  exact this

/-- **R2 (chain term).** -/
theorem lower_bound_chain_aux {c : ℕ} (hn : 0 < n) (hm : 0 < m) {L μ Δ ρ ε q : ℝ}
    (hμ : 0 < μ) (hμL : μ ≤ L) (hΔ : 0 < Δ) (hε : 0 < ε) (hq : 0 < q) (hq1 : q < 1)
    (hqL : 4 * ((n : ℝ) * m) * μ * q ≤ L * (1 - q) ^ 2) :
    ∃ (d : ℕ) (I : Instance n m d), I.InClass L μ Δ ∧ (∀ i ξ, ConvexOn ℝ Set.univ (I.F i ξ)) ∧
      ∀ A : Algorithm n m c, A.Admissible L μ Δ ρ → ∀ W : Matrix (Fin n) (Fin n) ℝ,
        ((⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊ : ℕ) : ℕ∞) ≤
          A.hittingTime I W ε := by
  set T := ⌈(m : ℝ) / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q)⌉₊
  have hK : 0 < T + 1 := Nat.succ_pos _
  refine ⟨m * (T + 1), hard n m (T + 1) hn μ Δ q,
    hard_inClass hn hm hK hμ hμL hΔ hq.le hq1 hqL,
    fun i ξ => hard_comp_convex hn hμ hq.le hq1 i ξ, fun A hA W => ?_⟩
  refine le_hittingTime_of A _ W ε T fun t ht => ?_
  refine lt_of_lt_of_le ?_ (expectedGap_ge_chain hn hm hK hμ hΔ hq hq1 _
    (hard_props hn hm hK hμ hΔ hq.le hq1) A hA W t (by omega))
  have htT : (t : ℝ) < m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε) / Real.log (1 / q) :=
    Nat.lt_ceil.1 ht
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1q : 0 < 1 - q := by linarith
  have hlogq : 0 < Real.log (1 / q) := Real.log_pos (by rw [lt_div_iff₀ hq]; linarith)
  have hR : 0 < (1 - q) ^ 2 * Δ / ε := by positivity
  rw [ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε.le]
  -- `(2t/m) log(1/q) < log((1-q)²Δ/ε)`
  have hexp : (2 * t : ℝ) / m * Real.log (1 / q) < Real.log ((1 - q) ^ 2 * Δ / ε) := by
    rw [lt_div_iff₀ hlogq] at htT
    have : (2 * t : ℝ) / m * Real.log (1 / q) = 2 / m * (t * Real.log (1 / q)) := by ring
    rw [this]
    calc 2 / m * (t * Real.log (1 / q)) < 2 / m * (m / 2 * Real.log ((1 - q) ^ 2 * Δ / ε)) :=
          mul_lt_mul_of_pos_left htT (by positivity)
      _ = Real.log ((1 - q) ^ 2 * Δ / ε) := by field_simp
  -- hence `ε / ((1-q)²Δ) < q^{2t/m}`
  have hpow : ε / ((1 - q) ^ 2 * Δ) < q ^ ((2 * t : ℝ) / m) := by
    have hpos : 0 < ε / ((1 - q) ^ 2 * Δ) := by positivity
    rw [← Real.log_lt_log_iff hpos (Real.rpow_pos_of_pos hq _), Real.log_rpow hq,
      Real.log_div hε.ne' (by positivity), ← neg_sub, ← Real.log_div (by positivity) hε.ne']
    have : Real.log (1 / q) = - Real.log q := by rw [one_div, Real.log_inv]
    rw [this] at hexp
    rw [show (1 - q) ^ 2 * Δ / ε = (1 - q) ^ 2 * Δ / ε from rfl]
    linarith
  rw [div_lt_iff₀ (by positivity)] at hpow
  linarith

end Chain
end ULB63

