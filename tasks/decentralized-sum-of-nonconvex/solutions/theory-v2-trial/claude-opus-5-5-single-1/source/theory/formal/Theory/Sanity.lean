import Theory.Model

/-!
# Adequacy checks for the model

Fixed like `Theory/Model.lean`. Proved facts showing that the definitions of `Model.lean` mean
what they are meant to:
the extended-real convexity notions agree with Mathlib's on real-valued functions, the
proximal specification picks the expected point, and none of the classes is empty. The last
one (`minimax_eq_zero_of_le`) shows the metric is not degenerate: when `ε ≥ Δ` the trivial
algorithm wins at time `0`.
-/

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63

theorem stronglyConvexE_coe_iff {d : ℕ} (μ : ℝ) (g : E d → ℝ) :
    StronglyConvexE μ (fun x => (g x : EReal)) ↔ StrongConvexOn Set.univ μ g := by
  constructor
  · intro h
    refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
    obtain rfl : b = 1 - a := by linarith
    have key := h x y a ha (by linarith)
    simp only at key
    norm_cast at key
    simp only [smul_eq_mul]
    nlinarith [key]
  · rintro ⟨-, h⟩ x y t ht₀ ht₁
    have key := h (Set.mem_univ x) (Set.mem_univ y) ht₀ (by linarith : 0 ≤ 1 - t) (by ring)
    simp only [smul_eq_mul] at key
    simp only
    norm_cast
    nlinarith [key]

/-- The proximal map of `(μ/2)‖·‖²` is `v ↦ v / (1 + ημ)`. -/
theorem isProx_halfSq {d : ℕ} {μ η : ℝ} (hμ : 0 ≤ μ) (hη : 0 < η) (v : E d) :
    IsProx (fun x => ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)) η v ((1 + η * μ)⁻¹ • v) := by
  intro u
  set p := (1 + η * μ)⁻¹ • v with hp
  set w := u - p with hw
  have hu : u = p + w := by rw [hw]; abel
  have hv : u - v = (p - v) + w := by rw [hw]; abel
  have ha : 0 < 1 + η * μ := by positivity
  have hcancel : μ * ⟪p, w⟫ + ⟪p - v, w⟫ / η = 0 := by
    rw [inner_sub_left, hp, real_inner_smul_left]
    field_simp
    ring
  have h1 : ‖u‖ ^ 2 = ‖p‖ ^ 2 + 2 * ⟪p, w⟫ + ‖w‖ ^ 2 := by rw [hu]; exact norm_add_sq_real p w
  have h2 : ‖u - v‖ ^ 2 = ‖p - v‖ ^ 2 + 2 * ⟪p - v, w⟫ + ‖w‖ ^ 2 := by
    rw [hv]; exact norm_add_sq_real _ _
  simp only
  norm_cast
  rw [h1, h2]
  have hw2 : 0 ≤ (μ / 2 + 1 / (2 * η)) * ‖w‖ ^ 2 := by positivity
  have expand : μ / 2 * (‖p‖ ^ 2 + 2 * ⟪p, w⟫ + ‖w‖ ^ 2) +
      (‖p - v‖ ^ 2 + 2 * ⟪p - v, w⟫ + ‖w‖ ^ 2) / (2 * η) =
      μ / 2 * ‖p‖ ^ 2 + ‖p - v‖ ^ 2 / (2 * η) + (μ * ⟪p, w⟫ + ⟪p - v, w⟫ / η) +
        (μ / 2 + 1 / (2 * η)) * ‖w‖ ^ 2 := by
    field_simp
    ring
  rw [expand, hcancel]
  linarith

/-- `J ∈ 𝓦_{n,ρ}` for every `ρ ≥ 0`. -/
theorem J_commClass {n : ℕ} (hn : 1 ≤ n) {ρ : ℝ} (hρ : 0 ≤ ρ) : CommClass n ρ (J n) where
  nonneg _ _ := by unfold J; positivity
  symm := by ext i j; rfl
  row_sum _ := by
    have : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
    simp [J, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    field_simp
  contraction v := by
    have : (J n - J n).mulVec v = 0 := by simp
    rw [this]
    have h0 : norm2 (0 : Fin n → ℝ) = 0 := by simp [norm2]
    rw [h0]
    exact mul_nonneg hρ (Real.sqrt_nonneg _)

/-- The instance with all components zero, `h = (μ/2)‖·‖²` and `x₀ = 0`. -/
def zeroInstance (n m d : ℕ) (μ : ℝ) : Instance n m d where
  F _ _ _ := 0
  h x := ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1 + η * μ)⁻¹ • v

theorem zeroInstance_φ {n m d : ℕ} {μ : ℝ} (x : E d) :
    (zeroInstance n m d μ).φ x = ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal) := by
  simp [Instance.φ, Instance.f, zeroInstance]

theorem zeroInstance_φstar {n m d : ℕ} {μ : ℝ} (hμ : 0 ≤ μ) :
    (zeroInstance n m d μ).φstar = 0 := by
  apply le_antisymm
  · refine (iInf_le _ 0).trans ?_
    simp [zeroInstance_φ]
  · refine le_iInf fun x => ?_
    rw [zeroInstance_φ]
    exact_mod_cast (by positivity : (0 : ℝ) ≤ μ / 2 * ‖x‖ ^ 2)

/-- `𝓕_{L,μ,Δ,m}` is nonempty in every dimension. -/
theorem zeroInstance_inClass {n m d : ℕ} {L μ Δ : ℝ} (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) :
    (zeroInstance n m d μ).InClass L μ Δ where
  smooth _ _ := ⟨differentiable_const _, fun x y => by
    simpa [zeroInstance] using mul_nonneg hL (norm_nonneg (x - y))⟩
  f_convex := by
    have : (zeroInstance n m d μ).f = fun _ => 0 := by
      funext x; simp [Instance.f, zeroInstance]
    rw [this]
    exact convexOn_const 0 convex_univ
  h_ne_bot _ := EReal.coe_ne_bot _
  h_proper := ⟨0, EReal.coe_ne_top _⟩
  h_closed := (continuous_coe_real_ereal.comp (by fun_prop)).lowerSemicontinuous
  h_convex := by
    have hs : StrongConvexOn Set.univ μ (fun x : E d => μ / 2 * ‖x‖ ^ 2) := by
      rw [strongConvexOn_iff_convex]; simpa using convexOn_const (0 : ℝ) convex_univ
    have hc : ConvexOn ℝ Set.univ (fun x : E d => μ / 2 * ‖x‖ ^ 2) :=
      UniformConvexOn.convexOn hs fun r => by positivity
    refine (stronglyConvexE_coe_iff 0 _).2 ?_
    rw [strongConvexOn_iff_convex]
    simpa using hc
  prox_exact v η hη := isProx_halfSq hμ.le hη v
  φ_strongly_convex := by
    have : (zeroInstance n m d μ).φ = fun x => ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal) := by
      funext x; exact zeroInstance_φ x
    rw [this, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    simpa using convexOn_const (0 : ℝ) convex_univ
  minimizer := ⟨0, fun y => by
    rw [zeroInstance_φ, zeroInstance_φ]
    exact_mod_cast (by simp; positivity : μ / 2 * ‖(0 : E d)‖ ^ 2 ≤ μ / 2 * ‖y‖ ^ 2)⟩
  initial_gap := by
    rw [zeroInstance_φstar hμ.le, sub_zero]
    show (zeroInstance n m d μ).φ 0 ≤ _
    rw [zeroInstance_φ]
    simpa using (EReal.coe_le_coe_iff.2 hΔ)

/-- The algorithm that never queries, sends zeros and outputs `x₀`. -/
def idleAlgorithm (n m c : ℕ) : Algorithm n m c where
  Ω := Unit
  P := Measure.dirac ()
  call _ _ _ := .idle
  send _ _ _ _ := 0
  out _ info _ := info.x0

instance (n m c : ℕ) : Subsingleton (idleAlgorithm n m c).Ω := inferInstanceAs (Subsingleton Unit)

/-- `𝓐` is nonempty. -/
theorem idleAlgorithm_admissible (n m c : ℕ) (L μ Δ ρ : ℝ) :
    (idleAlgorithm n m c).Admissible L μ Δ ρ where
  call_zero_respecting _ _ _ := by simp [idleAlgorithm, Call.supp]
  send_zero_respecting _ _ _ _ := by simp [idleAlgorithm, supp]
  out_zero_respecting _ info H := Set.subset_union_left
  prox_step_pos _ _ _ _ _ h := by simp [idleAlgorithm] at h
  one_grad _ _ _ _ := rfl
  one_prox _ _ _ _ := rfl
  measurable_call _ _ _ _ _ _ _ _ := Subsingleton.measurable
  measurable_send _ _ _ _ _ _ _ _ := Subsingleton.measurable
  measurable_out _ _ _ _ _ _ _ := Subsingleton.measurable

/-- `Sequential` is satisfiable. -/
theorem idleAlgorithm_sequential (n m c : ℕ) : (idleAlgorithm n m c).Sequential where
  no_late_call _ _ _ _ := rfl
  one_call _ _ _ _ := rfl
  silent_after_call _ _ _ _ := rfl

theorem idleAlgorithm_hittingTime {n m c d : ℕ} (hn : 1 ≤ n) {L μ Δ ε : ℝ} (hΔε : Δ ≤ ε)
    (I : Instance n m d) (hI : I.InClass L μ Δ) (W : Matrix (Fin n) (Fin n) ℝ) :
    (idleAlgorithm n m c).hittingTime I W ε = 0 := by
  have havg : ∀ ω, (idleAlgorithm n m c).avgOutput I W ω 0 = I.x0 := by
    intro ω
    have : (n : ℝ) ≠ 0 := by exact_mod_cast (by omega : n ≠ 0)
    simp only [Algorithm.avgOutput, Algorithm.output, idleAlgorithm, Algorithm.info,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, one_div, inv_mul_cancel₀ this, one_smul]
  have hgap : (idleAlgorithm n m c).expectedGap I W 0 ≤ ENNReal.ofReal ε := by
    simp only [Algorithm.expectedGap, havg, lintegral_const, measure_univ, mul_one]
    calc I.gap I.x0 ≤ (Δ : EReal).toENNReal := EReal.toENNReal_le_toENNReal hI.initial_gap
      _ = ENNReal.ofReal Δ := by simp
      _ ≤ ENNReal.ofReal ε := ENNReal.ofReal_le_ofReal hΔε
  exact le_antisymm ((iInf₂_le 0 hgap).trans (by simp)) (zero_le)

/-- When `ε ≥ Δ` the minimax time is `0`: the metric is not degenerate. -/
theorem minimax_eq_zero_of_le {n m c : ℕ} (hn : 1 ≤ n) {L μ Δ ρ ε : ℝ} (hΔε : Δ ≤ ε) :
    minimax n m c L μ Δ ρ ε = 0 := by
  refine le_antisymm ?_ zero_le
  refine (iInf₂_le (idleAlgorithm n m c) (idleAlgorithm_admissible n m c L μ Δ ρ)).trans ?_
  refine iSup_le fun d => iSup_le fun I => iSup_le fun hI => iSup_le fun W => iSup_le fun _ => ?_
  rw [idleAlgorithm_hittingTime hn hΔε I hI W]

end ULB63
