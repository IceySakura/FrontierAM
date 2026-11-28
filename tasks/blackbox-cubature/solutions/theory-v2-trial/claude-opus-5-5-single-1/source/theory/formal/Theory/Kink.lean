import Theory.Main

/-!
# The kink case `θ = φ` (result R2)

For the same `φ`, the kink `f = φ · ind(φ) = max(φ, 0)` has preintegral `y ↦ K_k(∑ y_i)` with
`K_k' = H_k`, so `P_j f` has total mixed order exactly `r_j + 1`.
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace GKLS

variable {n : ℕ}

lemma mul_ind_eq_max (u : ℝ) : u * ind u = max u 0 := by
  unfold ind
  split_ifs with h
  · rw [mul_one, max_eq_left h.le]
  · rw [mul_zero, max_eq_right (not_lt.mp h)]

lemma measurable_ind : Measurable ind := by
  have : ind = (Ioi (0 : ℝ)).indicator 1 := by
    funext u; simp [ind, Set.indicator]
  rw [this]
  exact measurable_one.indicator measurableSet_Ioi

/-- The kink profile `K_k(s) = ∫ max(g_k(t) + s, 0) dγ(t)`. -/
noncomputable def Kfun (k : ℕ) (s : ℝ) : ℝ := ∫ t, max (gfun k t + s) 0 ∂(gaussianReal 0 1)

lemma abs_gfun_le (k : ℕ) (t : ℝ) : |gfun k t| ≤ 1 / 2 + |t| := by
  have h := epsk_iteratedDeriv_le k 0 (by omega) t
  simp only [iteratedDeriv_zero] at h
  calc |gfun k t| = |epsk k * Pk k t + t| := rfl
    _ ≤ |epsk k * Pk k t| + |t| := abs_add_le _ _
    _ ≤ 1 / 2 + |t| := by linarith

lemma integral_ind_gfun (k : ℕ) (s : ℝ) :
    ∫ t, ind (gfun k t + s) ∂(gaussianReal 0 1) = Hfun k s := by
  have h : (fun t => ind (gfun k t + s)) = (Ioi (psi k (-s))).indicator 1 := by
    funext t
    simp only [ind, Set.indicator, Set.mem_Ioi, Pi.one_apply]
    have : (0 < gfun k t + s) ↔ psi k (-s) < t := by
      rw [psi_lt_iff]; constructor <;> intro h <;> linarith
    simp only [this]
  rw [h, integral_indicator_one measurableSet_Ioi]
  rfl

lemma hasDerivAt_Kfun (k : ℕ) (s₀ : ℝ) : HasDerivAt (Kfun k) (Hfun k s₀) s₀ := by
  have hγ : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
  have hcont : Continuous (gfun k) := continuous_gfun k
  have hid : Integrable (fun t : ℝ => t) (gaussianReal 0 1) :=
    (memLp_id_gaussianReal' 1 (by simp)).integrable le_rfl
  have hF_int : Integrable (fun t => max (gfun k t + s₀) 0) (gaussianReal 0 1) := by
    refine Integrable.mono' ((hid.abs).add (integrable_const (1 / 2 + |s₀|))) ?_ ?_
    · exact ((hcont.add continuous_const).max continuous_const).aestronglyMeasurable
    · refine ae_of_all _ fun t => ?_
      have h1 := abs_gfun_le k t
      show ‖max (gfun k t + s₀) 0‖ ≤ |t| + (1 / 2 + |s₀|)
      rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
      apply max_le
      · have := le_abs_self (gfun k t)
        linarith [abs_nonneg s₀, le_abs_self s₀]
      · positivity
  have hdiff : ∀ᵐ t ∂(gaussianReal 0 1),
      HasDerivAt (fun s => max (gfun k t + s) 0) (ind (gfun k t + s₀)) s₀ := by
    have hnull : ∀ᵐ t ∂(gaussianReal 0 1), t ≠ psi k (-s₀) := by
      rw [ae_iff]
      simp only [ne_eq, not_not, Set.setOf_eq_eq_singleton]
      exact measure_singleton _
    filter_upwards [hnull] with t ht
    have hne : gfun k t + s₀ ≠ 0 := by
      intro h0
      apply ht
      have : gfun k t = -s₀ := by linarith
      rw [← this, psi_gfun]
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have hev : (fun s => max (gfun k t + s) 0) =ᶠ[𝓝 s₀] fun _ => 0 := by
        have : ∀ᶠ s in 𝓝 s₀, gfun k t + s < 0 :=
          (continuous_const.add continuous_id).continuousAt.eventually (gt_mem_nhds hlt)
        filter_upwards [this] with s hs
        exact max_eq_right hs.le
      have hind : ind (gfun k t + s₀) = 0 := by simp [ind, not_lt.mpr hlt.le]
      rw [hind]
      exact (hasDerivAt_const s₀ (0 : ℝ)).congr_of_eventuallyEq hev
    · have hev : (fun s => max (gfun k t + s) 0) =ᶠ[𝓝 s₀] fun s => gfun k t + s := by
        have : ∀ᶠ s in 𝓝 s₀, 0 < gfun k t + s :=
          (continuous_const.add continuous_id).continuousAt.eventually (lt_mem_nhds hgt)
        filter_upwards [this] with s hs
        exact max_eq_left hs.le
      have hind : ind (gfun k t + s₀) = 1 := by simp [ind, hgt]
      rw [hind]
      exact ((hasDerivAt_id' s₀).const_add (gfun k t)).congr_of_eventuallyEq hev
  have hlip : ∀ᵐ t ∂(gaussianReal 0 1),
      LipschitzOnWith (Real.nnabs ((fun _ : ℝ => (1 : ℝ)) t)) (fun s => max (gfun k t + s) 0) univ := by
    refine ae_of_all _ fun t => ?_
    have : Real.nnabs ((fun _ : ℝ => (1 : ℝ)) t) = 1 := by simp
    rw [this]
    refine LipschitzWith.lipschitzOnWith (LipschitzWith.of_dist_le_mul fun a b => ?_)
    simp only [Real.dist_eq, NNReal.coe_one, one_mul]
    calc |max (gfun k t + a) 0 - max (gfun k t + b) 0| ≤ |(gfun k t + a) - (gfun k t + b)| :=
          abs_max_sub_max_le_abs _ _ _
      _ = |a - b| := by ring_nf
  obtain ⟨-, key⟩ := hasDerivAt_integral_of_dominated_loc_of_lip (μ := gaussianReal 0 1)
    (F := fun s t => max (gfun k t + s) 0) (F' := fun t => ind (gfun k t + s₀)) (x₀ := s₀)
    (bound := fun _ => (1 : ℝ)) (s := univ) univ_mem
    (Eventually.of_forall fun s =>
      ((hcont.add continuous_const).max continuous_const).aestronglyMeasurable)
    hF_int ((measurable_ind.comp (hcont.add continuous_const).measurable).aestronglyMeasurable)
    hlip (integrable_const 1) hdiff
  rw [integral_ind_gfun] at key
  exact key

lemma deriv_Kfun (k : ℕ) : deriv (Kfun k) = Hfun k := funext fun s => (hasDerivAt_Kfun k s).deriv

lemma contDiff_Kfun (k : ℕ) : ContDiff ℝ ((k + 2 : ℕ) : WithTop ℕ∞) (Kfun k) := by
  have h : ((k + 2 : ℕ) : WithTop ℕ∞) = ((k + 1 : ℕ) : WithTop ℕ∞) + 1 := by
    rw [show k + 2 = (k + 1) + 1 by ring, Nat.cast_succ]
  rw [h, contDiff_succ_iff_deriv]
  refine ⟨fun s => (hasDerivAt_Kfun k s).differentiableAt, by simp, ?_⟩
  rw [deriv_Kfun]
  exact_mod_cast contDiff_Hfun k

lemma not_differentiableAt_iteratedDeriv_Kfun (k : ℕ) :
    ¬ DifferentiableAt ℝ (iteratedDeriv (k + 2) (Kfun k)) 0 := by
  rw [show k + 2 = (k + 1) + 1 by ring, iteratedDeriv_succ', deriv_Kfun]
  exact not_differentiableAt_iteratedDeriv_Hfun k

/-- The preintegrated kink is the ridge function `y ↦ K_k(∑ y_i)`. -/
lemma preintegrate_kink (k : ℕ) (j : Fin (n + 1)) :
    preintegrate j (fun x => phiCE k j x * ind (phiCE k j x)) = fun y => Kfun k (∑ i, y i) := by
  funext y
  simp only [preintegrate, Kfun, phiCE_insertNth, mul_ind_eq_max]

/-- Assembly of R2. -/
theorem kink_counterexample (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j + 1 < ∑ k : Fin n, r (j.succAbove k)) :
    ∃ φ : (Fin (n + 1) → ℝ) → ℝ,
      ContDiffMixed r φ ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        IterPartialExists l φ ∧ Continuous (iterPartial l φ)) ∧
      (∀ x, 1 / 2 ≤ partialDeriv j φ x) ∧
      (∀ y : Fin n → ℝ, Tendsto (fun t => φ (Fin.insertNth j t y)) atTop atTop) ∧
      (∀ y : Fin n → ℝ, ∃ t, φ (Fin.insertNth j t y) = 0) ∧
      (∀ x, |φ x| ≤ 1 + ∑ i, |x i|) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j → ∀ x, |iterPartial l φ x| ≤ 3 / 2) ∧
      (∀ p : ℝ≥0∞, p ≠ ∞ → ∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        MemLp (iterPartial l φ) p (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      (∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
        ∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j →
          Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))) ∧
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => φ x * ind (φ x))) ∧
      (∀ l : List (Fin n), l.length ≤ r j + 1 →
        IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => φ x * ind (φ x))))) ∧
      (∀ l : List (Fin n), r j + 1 < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x)))) := by
  obtain ⟨k, hk⟩ : ∃ k, r j = k + 1 := ⟨r j - 1, by omega⟩
  have hP := preintegrate_kink (n := n) k j
  have hKC : ContDiff ℝ ((k + 2 : ℕ) : WithTop ℕ∞) (Kfun k) := contDiff_Kfun k
  have hbad := not_differentiableAt_iteratedDeriv_Kfun k
  have hnot : ∀ l : List (Fin n), r j + 1 < l.length →
      ¬ IterPartialExists l (preintegrate j (fun x => phiCE k j x * ind (phiCE k j x))) := by
    intro l hl
    rw [hP]
    exact ridge_not_exists (N := k + 2) hKC hbad l (by omega)
  -- the hypotheses on `φ` are those of R1
  refine ⟨phiCE k j, phiCE_mixed k j r hk, fun l hl => phiCE_smooth k j l (hk ▸ hl), ?_, ?_, ?_,
    phiCE_bound k j, fun l hne hl x => phiCE_bound_nonempty k j l hne (hk ▸ hl) x,
    fun p hp l hl => phiCE_memLp k j p hp l (hk ▸ hl),
    fun p ζ hp hζ l hne hl => phiCE_weighted k j p hp ζ hζ l hne (hk ▸ hl), ?_, ?_, hnot⟩
  · intro x
    rw [deriv_j_phiCE]
    exact half_le_deriv_gfun k (x j)
  · intro y
    have h : ∀ t, t + ∑ i, y i ≤ phiCE k j (Fin.insertNth j t y) := by
      intro t; rw [phiCE_insertNth]; have := le_gfun k t; linarith
    exact tendsto_atTop_mono h (tendsto_atTop_add_const_right _ _ tendsto_id)
  · intro y
    refine ⟨psi k (-∑ i, y i), ?_⟩
    rw [phiCE_insertNth, gfun_psi]; ring
  · intro hmix
    obtain ⟨L, hLc, hLl⟩ := exists_list_count (fun i => r (j.succAbove i))
    exact hnot L (by rw [hLl]; exact hsum) (hmix L (fun i => (hLc i).le)).1
  · intro l hl
    rw [hP]
    obtain ⟨hex, heq⟩ := ridge_iterPartial (N := k + 2) hKC l (by omega)
    refine ⟨hex, ?_⟩
    rw [heq]
    have hc : Continuous (iteratedDeriv l.length (Kfun k)) :=
      hKC.continuous_iteratedDeriv _ (by exact_mod_cast (show l.length ≤ k + 2 by omega))
    fun_prop

end GKLS
