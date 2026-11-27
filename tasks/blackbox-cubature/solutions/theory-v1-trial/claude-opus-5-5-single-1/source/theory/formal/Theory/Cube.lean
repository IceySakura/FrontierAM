import Theory.Defs
import Theory.Stratified

/-!
# Jittered sampling on the unit cube for preintegrated non-monotone jump integrands
(auxiliary proofs for R5, cube case)
-/

open MeasureTheory ProbabilityTheory Real Set Function
open scoped ENNReal

namespace Cubature.Aux

lemma cubeCell_measurable (s m : ℕ) (k : Fin s → Fin m) : MeasurableSet (cubeCell s m k) :=
  MeasurableSet.univ_pi fun _ => measurableSet_Ico

lemma cubeCell_disjoint (s m : ℕ) : Pairwise (Disjoint on cubeCell s m) := by
  intro k k' hkk'
  obtain ⟨j, hj⟩ := Function.ne_iff.mp hkk'
  refine Set.disjoint_left.mpr fun x hx hx' => ?_
  have h1 := Set.mem_univ_pi.mp hx j
  have h2 := Set.mem_univ_pi.mp hx' j
  simp only [mem_Ico] at h1 h2
  have hm : (0 : ℝ) < m := by exact_mod_cast Fin.pos (k j)
  have hne : (k j : ℕ) ≠ (k' j : ℕ) := fun h => hj (Fin.ext h)
  rcases lt_or_gt_of_ne hne with hlt | hlt
  · have : ((k j : ℝ) + 1) / m ≤ (k' j : ℝ) / m := by
      gcongr; exact_mod_cast hlt
    linarith [h1.2, h2.1]
  · have : ((k' j : ℝ) + 1) / m ≤ (k j : ℝ) / m := by
      gcongr; exact_mod_cast hlt
    linarith [h1.1, h2.2]

lemma cube_subset_iUnion (s m : ℕ) (hm : 0 < m) :
    (Set.pi univ fun _ : Fin s => Ico (0 : ℝ) 1) ⊆ ⋃ k, cubeCell s m k := by
  intro x hx
  have hx' := Set.mem_univ_pi.mp hx
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hlt : ∀ j, ⌊(m : ℝ) * x j⌋₊ < m := by
    intro j
    have h := hx' j
    simp only [mem_Ico] at h
    have : (m : ℝ) * x j < m := by nlinarith
    exact (Nat.floor_lt (mul_nonneg hmR.le h.1)).mpr (by simpa using this)
  refine Set.mem_iUnion.mpr ⟨fun j => ⟨⌊(m : ℝ) * x j⌋₊, hlt j⟩, ?_⟩
  refine Set.mem_univ_pi.mpr fun j => ?_
  have h := hx' j
  simp only [mem_Ico] at h
  have h0 : 0 ≤ (m : ℝ) * x j := mul_nonneg hmR.le h.1
  simp only [mem_Ico]
  constructor
  · rw [div_le_iff₀ hmR]
    calc (⌊(m : ℝ) * x j⌋₊ : ℝ) ≤ (m : ℝ) * x j := Nat.floor_le h0
      _ = x j * m := mul_comm _ _
  · rw [lt_div_iff₀ hmR]
    calc x j * m = (m : ℝ) * x j := mul_comm _ _
      _ < ⌊(m : ℝ) * x j⌋₊ + 1 := Nat.lt_floor_add_one _

lemma cubeCell_subset (s m : ℕ) (k : Fin s → Fin m) :
    cubeCell s m k ⊆ Set.pi univ fun _ : Fin s => Ico (0 : ℝ) 1 := by
  intro x hx
  refine Set.mem_univ_pi.mpr fun j => ?_
  have h := Set.mem_univ_pi.mp hx j
  simp only [mem_Ico] at h ⊢
  have hm : (0 : ℝ) < m := by exact_mod_cast Fin.pos (k j)
  have hk : ((k j : ℕ) : ℝ) + 1 ≤ m := by exact_mod_cast (k j).isLt
  constructor
  · exact le_trans (by positivity) h.1
  · calc x j < ((k j : ℝ) + 1) / m := h.2
      _ ≤ (m : ℝ) / m := by gcongr
      _ = 1 := div_self hm.ne'

lemma cubeCell_dist (s m : ℕ) (hm : 0 < m) (k : Fin s → Fin m) :
    ∀ y ∈ cubeCell s m k, ∀ y' ∈ cubeCell s m k, dist y y' ≤ 1 / m := by
  intro y hy y' hy'
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  refine (dist_pi_le_iff (by positivity)).mpr fun j => ?_
  have h1 := Set.mem_univ_pi.mp hy j
  have h2 := Set.mem_univ_pi.mp hy' j
  simp only [mem_Ico] at h1 h2
  rw [Real.dist_eq, abs_le]
  have e : ((k j : ℝ) + 1) / m = (k j : ℝ) / m + 1 / m := by ring
  constructor <;> linarith

theorem jump_jittered_cube {Ω : Type*} [MeasurableSpace Ω] {s m : ℕ} (hm : 0 < m)
    {ν : Measure ℝ} [IsProbabilityMeasure ν] {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hν : ∀ S : Set ℝ, ν S ≤ ENNReal.ofReal ρ * volume S)
    {φ θ : ℝ → (Fin s → ℝ) → ℝ} {α L Lθ B : ℝ} (hα : 0 < α) (hL : 0 ≤ L) (hLθ : 0 ≤ Lθ)
    (hφ : ∀ y, StrongConvexOn univ α (φ · y) ∨ StrongConcaveOn univ α (φ · y))
    (hφL : ∀ t y y', |φ t y - φ t y'| ≤ L * dist y y')
    (hθm : ∀ y, Measurable (θ · y)) (hθB : ∀ t y, |θ t y| ≤ B)
    (hθL : ∀ t y y', |θ t y - θ t y'| ≤ Lθ * dist y y')
    {P : Measure Ω} [IsProbabilityMeasure P]
    {X : (Fin s → Fin m) → Ω → (Fin s → ℝ)} (hXm : ∀ k, Measurable (X k))
    (hXind : iIndepFun X P)
    (hXlaw : ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k)) :
    (∫ ω, (1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) ∂P =
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ∧
    ∫ ω, ((1 / (m : ℝ) ^ s) * ∑ k, preintJump ν φ θ (X k ω) -
        ∫ y in Set.pi univ (fun _ => Ico (0 : ℝ) 1), preintJump ν φ θ y) ^ 2 ∂P ≤
      (Lθ / m + 4 * ρ * B * √(L / (α * m))) ^ 2 / (m : ℝ) ^ s := by
  set cube := Set.pi univ (fun _ : Fin s => Ico (0 : ℝ) 1) with hcube
  have hcubem : MeasurableSet cube := MeasurableSet.univ_pi fun _ => measurableSet_Ico
  have hcard : (Fintype.card (Fin s → Fin m) : ℝ) = (m : ℝ) ^ s := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; push_cast; ring
  have hcardE : (Fintype.card (Fin s → Fin m) : ℝ≥0∞) = (m : ℝ≥0∞) ^ s := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; push_cast; ring
  have hlaw' : ∀ k, P.map (X k) =
      (Fintype.card (Fin s → Fin m) : ℝ≥0∞) • (volume.restrict cube).restrict (cubeCell s m k) := by
    intro k
    rw [hXlaw k, hcardE, Measure.restrict_restrict (cubeCell_measurable s m k),
      Set.inter_eq_left.mpr (cubeCell_subset s m k)]
  have hcov : (volume.restrict cube) (⋃ k, cubeCell s m k)ᶜ = 0 := by
    rw [Measure.restrict_apply (MeasurableSet.iUnion fun k => cubeCell_measurable s m k).compl]
    have : (⋃ k, cubeCell s m k)ᶜ ∩ cube = ∅ := by
      ext x; simp only [mem_inter_iff, mem_compl_iff, mem_empty_iff_false, iff_false, not_and]
      intro h hx; exact h (cube_subset_iUnion s m hm hx)
    rw [this, measure_empty]
  obtain ⟨hmean, hmse⟩ := jump_stratified hρ hν hα hL hLθ hφ hφL hθm hθB hθL
    (μ := volume.restrict cube) (cubeCell_measurable s m) (cubeCell_disjoint s m) hcov
    Finset.univ (D := fun _ => 1 / m) (fun k _ => cubeCell_dist s m hm k) hXm hXind hlaw'
  rw [hcard] at hmean hmse
  refine ⟨hmean, hmse.trans (le_of_eq ?_)⟩
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hms : (0 : ℝ) < (m : ℝ) ^ s := by positivity
  have hsum : (∑ _i : Fin s → Fin m, (Lθ * (1 / (m : ℝ)) + 4 * ρ * B * √(L * (1 / m) / α)) ^ 2)
      = (m : ℝ) ^ s * (Lθ * (1 / (m : ℝ)) + 4 * ρ * B * √(L * (1 / m) / α)) ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, hcard]
  have hsum2 : ∑ _i ∈ (Finset.univ : Finset (Fin s → Fin m))ᶜ, (2 * B) ^ 2 = 0 := by simp
  have e1 : Lθ * (1 / (m : ℝ)) = Lθ / m := by ring
  have e2 : L * (1 / (m : ℝ)) / α = L / (α * m) := by field_simp
  rw [hsum, hsum2, add_zero, e1, e2]
  field_simp

end Cubature.Aux

namespace Cubature.Aux

lemma cubeCell_prob (s m : ℕ) (hm : 0 < m) (k : Fin s → Fin m) :
    IsProbabilityMeasure (((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k)) := by
  constructor
  rw [Measure.smul_apply, Measure.restrict_apply MeasurableSet.univ, Set.univ_inter, cubeCell,
    Real.volume_pi_Ico]
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have e : ∀ j, ENNReal.ofReal (((k j : ℝ) + 1) / m - (k j : ℝ) / m) = (m : ℝ≥0∞)⁻¹ := by
    intro j
    rw [show ((k j : ℝ) + 1) / m - (k j : ℝ) / m = (m : ℝ)⁻¹ by field_simp; ring,
      ENNReal.ofReal_inv_of_pos hmR, ENNReal.ofReal_natCast]
  simp only [e, Finset.prod_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
  rw [← mul_pow, ENNReal.mul_inv_cancel (by exact_mod_cast hm.ne') (ENNReal.natCast_ne_top m),
    one_pow]

theorem jittered_exists (s m : ℕ) (hm : 0 < m) :
    ∃ (Ω : Type) (_ : MeasurableSpace Ω) (P : Measure Ω) (_ : IsProbabilityMeasure P)
      (X : (Fin s → Fin m) → Ω → (Fin s → ℝ)),
      (∀ k, Measurable (X k)) ∧ iIndepFun X P ∧
      ∀ k, P.map (X k) = ((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k) := by
  let μ : (Fin s → Fin m) → Measure (Fin s → ℝ) :=
    fun k => ((m : ℝ≥0∞) ^ s) • volume.restrict (cubeCell s m k)
  haveI : ∀ k, IsProbabilityMeasure (μ k) := fun k => cubeCell_prob s m hm k
  refine ⟨(Fin s → Fin m) → (Fin s → ℝ), inferInstance, Measure.pi μ, inferInstance,
    fun k ω => ω k, fun k => measurable_pi_apply k, ?_, fun k => ?_⟩
  · exact iIndepFun_pi (X := fun _ x => x) (fun _ => aemeasurable_id)
  · exact (measurePreserving_eval μ k).map_eq

end Cubature.Aux
