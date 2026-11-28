import Theory.MultiDim

/-!
# Assembly of the counterexample to weakening the condition of GKLS Theorem 3

The counterexample is `φ(x) = ε_k χ(x_j) (max(x_j, 0))^(k+2) + ∑_i x_i` (with `r_j = k + 1`),
`θ ≡ 1`, and `ρ` the standard normal density.
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace GKLS

variable {n : ℕ}

lemma sum_insertNth (j : Fin (n + 1)) (t : ℝ) (y : Fin n → ℝ) :
    ∑ i, (Fin.insertNth j t y : Fin (n + 1) → ℝ) i = t + ∑ i, y i := by
  rw [Fin.sum_univ_succAbove _ j]
  simp [Fin.insertNth_apply_same, Fin.insertNth_apply_succAbove]

/-- The profile `G_k(t) = ε_k P_k(t)` in the direction `x_j`. -/
noncomputable def Gk (k : ℕ) (t : ℝ) : ℝ := epsk k * Pk k t

/-- The counterexample `φ(x) = G_k(x_j) + ∑_i x_i`. -/
noncomputable def phiCE (k : ℕ) (j : Fin (n + 1)) (x : Fin (n + 1) → ℝ) : ℝ :=
  Gk k (x j) + ∑ i, x i

lemma contDiff_Gk (k : ℕ) : ContDiff ℝ (k + 1) (Gk k) := contDiff_const.mul (contDiff_Pk k)

lemma iteratedDeriv_Gk (k a : ℕ) (t : ℝ) :
    iteratedDeriv a (Gk k) t = epsk k * iteratedDeriv a (Pk k) t :=
  iteratedDeriv_const_mul_field (epsk k) (Pk k)

lemma abs_iteratedDeriv_Gk_le (k a : ℕ) (ha : a ≤ k + 1) (t : ℝ) :
    |iteratedDeriv a (Gk k) t| ≤ 1 / 2 := by
  rw [iteratedDeriv_Gk]; exact epsk_iteratedDeriv_le k a ha t

lemma phiCE_insertNth (k : ℕ) (j : Fin (n + 1)) (t : ℝ) (y : Fin n → ℝ) :
    phiCE k j (Fin.insertNth j t y) = gfun k t + ∑ i, y i := by
  simp only [phiCE, Fin.insertNth_apply_same, sum_insertNth, gfun, Gk]; ring

/-- The preintegrated indicator is the ridge function `y ↦ H_k(∑ y_i)`. -/
lemma preintegrate_phiCE (k : ℕ) (j : Fin (n + 1)) :
    preintegrate j (fun x => ind (phiCE k j x)) = fun y => Hfun k (∑ i, y i) := by
  funext y
  have h : (fun t => ind (phiCE k j (Fin.insertNth j t y))) =
      (Ioi (psi k (-∑ i, y i))).indicator 1 := by
    funext t
    rw [phiCE_insertNth]
    simp only [ind, Set.indicator, Set.mem_Ioi, Pi.one_apply]
    have : (0 < gfun k t + ∑ i, y i) ↔ psi k (-∑ i, y i) < t := by
      rw [psi_lt_iff]; constructor <;> intro h <;> linarith
    simp only [this]
  simp only [preintegrate, Hfun, gtail]
  rw [h, integral_indicator_one measurableSet_Ioi]

lemma phiCE_iterPartial (k : ℕ) (j : Fin (n + 1)) (l : List (Fin (n + 1))) (hl : l.count j ≤ k + 1) :
    IterPartialExists l (phiCE k j) ∧
      iterPartial l (phiCE k j) = fun x =>
        (if l = [] then 1 else 0) * ∑ i, x i +
          (if ∀ i ∈ l, i = j then 1 else 0) * iteratedDeriv l.length (Gk k) (x j) +
          (if l.length = 1 then 1 else 0) :=
  phi_iterPartial (m := k + 1) (contDiff_Gk k) j l hl

lemma length_of_all_eq {l : List (Fin (n + 1))} {j : Fin (n + 1)} (hall : ∀ i ∈ l, i = j) :
    l.length = l.count j :=
  (List.count_eq_length.mpr (fun i hi => (hall i hi).symm)).symm

/-- All iterated partial derivatives with at most `k + 1` differentiations in `x_j` (and any
number in the other variables) exist and are continuous. -/
lemma phiCE_smooth (k : ℕ) (j : Fin (n + 1)) (l : List (Fin (n + 1))) (hlj : l.count j ≤ k + 1) :
    IterPartialExists l (phiCE k j) ∧ Continuous (iterPartial l (phiCE k j)) := by
  obtain ⟨hex, heq⟩ := phiCE_iterPartial k j l hlj
  refine ⟨hex, ?_⟩
  rw [heq]
  by_cases hall : ∀ i ∈ l, i = j
  · have hlen : l.length ≤ k + 1 := (length_of_all_eq hall).trans_le hlj
    have hc : Continuous (iteratedDeriv l.length (Gk k)) :=
      (contDiff_Gk k).continuous_iteratedDeriv _ (by exact_mod_cast hlen)
    rw [if_pos hall]
    simp only [one_mul]
    fun_prop
  · simp only [if_neg hall, zero_mul, add_zero]
    fun_prop

lemma phiCE_mixed (k : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ) (hr : r j = k + 1) :
    ContDiffMixed r (phiCE k j) :=
  fun l hl => phiCE_smooth k j l (hr ▸ hl j)

lemma deriv_j_phiCE (k : ℕ) (j : Fin (n + 1)) (x : Fin (n + 1) → ℝ) :
    partialDeriv j (phiCE k j) x = epsk k * deriv (Pk k) (x j) + 1 := by
  obtain ⟨-, heq⟩ := phiCE_iterPartial k j [j] (by simp)
  have : partialDeriv j (phiCE k j) x = iterPartial [j] (phiCE k j) x := rfl
  rw [this, heq]
  simp [iteratedDeriv_one, iteratedDeriv_Gk]

lemma phiCE_bound_nonempty (k : ℕ) (j : Fin (n + 1)) (l : List (Fin (n + 1))) (hne : l ≠ [])
    (hl : l.count j ≤ k + 1) (x : Fin (n + 1) → ℝ) :
    |iterPartial l (phiCE k j) x| ≤ 3 / 2 := by
  obtain ⟨-, heq⟩ := phiCE_iterPartial k j l hl
  rw [heq]
  have hc3 : |(if l.length = 1 then (1 : ℝ) else 0)| ≤ 1 := by split_ifs <;> simp
  simp only [hne, ite_false, zero_mul, zero_add]
  by_cases hall : ∀ i ∈ l, i = j
  · rw [if_pos hall, one_mul]
    have hlen : l.length ≤ k + 1 := (length_of_all_eq hall).trans_le hl
    have := abs_iteratedDeriv_Gk_le k l.length hlen (x j)
    calc _ ≤ |iteratedDeriv l.length (Gk k) (x j)| + |(if l.length = 1 then (1 : ℝ) else 0)| :=
          abs_add_le _ _
      _ ≤ 3 / 2 := by linarith
  · rw [if_neg hall, zero_mul, zero_add]
    linarith

lemma phiCE_bound (k : ℕ) (j : Fin (n + 1)) (x : Fin (n + 1) → ℝ) :
    |phiCE k j x| ≤ 1 + ∑ i, |x i| := by
  have h1 : |Gk k (x j)| ≤ 1 / 2 := by
    have := abs_iteratedDeriv_Gk_le k 0 (by omega) (x j)
    simpa using this
  have h2 : |∑ i, x i| ≤ ∑ i, |x i| := Finset.abs_sum_le_sum_abs _ _
  calc |phiCE k j x| ≤ |Gk k (x j)| + |∑ i, x i| := abs_add_le _ _
    _ ≤ 1 + ∑ i, |x i| := by linarith

/-- The standard Gaussian measure on `ℝ^m`. -/
noncomputable abbrev stdGaussian (m : ℕ) : Measure (Fin m → ℝ) :=
  Measure.pi fun _ : Fin m => gaussianReal 0 1

lemma memLp_coord (m : ℕ) (p : ℝ≥0∞) (hp : p ≠ ∞) (i : Fin m) :
    MemLp (fun x : Fin m → ℝ => x i) p (stdGaussian m) :=
  (memLp_id_gaussianReal' p hp).comp_measurePreserving (measurePreserving_eval _ i)

/-- Weighted `L^p` membership (Gaussian weights) of all the mixed derivatives of `φ`. -/
lemma phiCE_memLp (k : ℕ) (j : Fin (n + 1)) (p : ℝ≥0∞) (hp : p ≠ ∞) (l : List (Fin (n + 1)))
    (hl : l.count j ≤ k + 1) : MemLp (iterPartial l (phiCE k j)) p (stdGaussian (n + 1)) := by
  obtain ⟨-, hc⟩ := phiCE_smooth k j l hl
  by_cases hne : l = []
  · subst hne
    have h1 : MemLp (fun x : Fin (n + 1) → ℝ => Gk k (x j)) p (stdGaussian (n + 1)) := by
      refine MemLp.of_bound (((contDiff_Gk k).continuous.comp (continuous_apply j)).aestronglyMeasurable)
        (1 / 2) (ae_of_all _ fun x => ?_)
      have := abs_iteratedDeriv_Gk_le k 0 (by omega) (x j)
      simpa using this
    have h2 : MemLp (fun x : Fin (n + 1) → ℝ => ∑ i, x i) p (stdGaussian (n + 1)) :=
      memLp_finsetSum _ (fun i _ => memLp_coord (n + 1) p hp i)
    exact h1.add h2
  · exact MemLp.of_bound hc.aestronglyMeasurable (3 / 2)
      (ae_of_all _ fun x => by simpa using phiCE_bound_nonempty k j l hne hl x)

/-- Weighted integrability of the derivatives `D^α φ`, `α ≠ 0`, against any product of integrable
weights `ζ_{α_1}(x_1) ⋯ ζ_{α_d}(x_d)`. -/
lemma phiCE_weighted (k : ℕ) (j : Fin (n + 1)) (p : ℝ) (hp : 0 ≤ p) (ζ : ℕ → ℝ → ℝ)
    (hζ : ∀ i, Integrable (ζ i)) (l : List (Fin (n + 1))) (hne : l ≠ []) (hl : l.count j ≤ k + 1) :
    Integrable (fun x : Fin (n + 1) → ℝ => |iterPartial l (phiCE k j) x| ^ p *
      ∏ i, ζ (l.count i) (x i)) := by
  obtain ⟨-, hc⟩ := phiCE_smooth k j l hl
  have hprod : Integrable (fun x : Fin (n + 1) → ℝ => ∏ i, ζ (l.count i) (x i)) := by
    rw [volume_pi]
    exact Integrable.fintype_prod (fun i => hζ (l.count i))
  refine hprod.bdd_mul (c := (3 / 2) ^ p) ?_ (ae_of_all _ fun x => ?_)
  · exact ((hc.abs).measurable.pow_const p).aestronglyMeasurable
  · rw [Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) p)]
    exact Real.rpow_le_rpow (abs_nonneg _) (phiCE_bound_nonempty k j l hne hl x) hp

/-- A list with prescribed multiplicities. -/
lemma exists_list_count (r : Fin n → ℕ) :
    ∃ l : List (Fin n), (∀ i, l.count i = r i) ∧ l.length = ∑ i, r i := by
  refine ⟨(∑ i, Multiset.replicate (r i) i).toList, fun i => ?_, ?_⟩
  · rw [← Multiset.coe_count, Multiset.coe_toList, Multiset.count_sum']
    simp [Multiset.count_replicate]
  · rw [Multiset.length_toList, Multiset.card_sum]
    simp

/-- Main assembly. -/
theorem main_counterexample (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j < ∑ k : Fin n, r (j.succAbove k)) :
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
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => ind (φ x))) ∧
      (∀ l : List (Fin n), l.length ≤ r j →
        IterPartialExists l (preintegrate j (fun x => ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => ind (φ x))))) ∧
      (∀ l : List (Fin n), r j < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => ind (φ x)))) := by
  obtain ⟨k, hk⟩ : ∃ k, r j = k + 1 := ⟨r j - 1, by omega⟩
  have hP := preintegrate_phiCE (n := n) k j
  have hHC : ContDiff ℝ (k + 1) (Hfun k) := contDiff_Hfun k
  have hbad := not_differentiableAt_iteratedDeriv_Hfun k
  have hnot : ∀ l : List (Fin n), r j < l.length →
      ¬ IterPartialExists l (preintegrate j (fun x => ind (phiCE k j x))) := by
    intro l hl
    rw [hP]
    exact ridge_not_exists (N := k + 1) (by exact_mod_cast hHC) (by exact_mod_cast hbad) l (hk ▸ hl)
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
    obtain ⟨hex, heq⟩ := ridge_iterPartial (N := k + 1) (by exact_mod_cast hHC) l (hk ▸ hl)
    refine ⟨hex, ?_⟩
    rw [heq]
    have hc : Continuous (iteratedDeriv l.length (Hfun k)) :=
      hHC.continuous_iteratedDeriv _ (by exact_mod_cast (hk ▸ hl))
    fun_prop

end GKLS
