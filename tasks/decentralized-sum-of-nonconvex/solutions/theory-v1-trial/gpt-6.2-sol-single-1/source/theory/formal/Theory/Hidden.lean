import Theory.LowerTools
import Theory.Composite

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

set_option maxHeartbeats 900000

noncomputable section

namespace ULB63
namespace Hidden

open Protocol Lower

variable {n m c : ℕ} {L μ Δ ρ ε : ℝ}

def plant (n m : ℕ) (μ : ℝ) (u : E 1) (r : Fin n) (j : Fin m) : Instance n m 1 :=
  { zeroInstance n m 1 μ with
    F := fun i l x => if i = r ∧ l = j then -(n:ℝ)*(m:ℝ)*μ*⟪u,x⟫ else 0 }

private theorem plant_f (hn : 0 < n) (hm : 0 < m) (u : E 1) (r : Fin n) (j : Fin m) (x : E 1) :
    (plant n m μ u r j).f x = -μ*⟪u,x⟫ := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  let a := -(n:ℝ)*(m:ℝ)*μ*⟪u,x⟫
  change (1/(n:ℝ))*∑ i : Fin n, (1/(m:ℝ))*∑ l : Fin m,
    (if i=r ∧ l=j then a else 0) = _
  have H (i : Fin n) : (∑ l : Fin m, (if i=r ∧ l=j then a else 0)) =
      if i=r then a else 0 := by
    by_cases hi : i=r
    · simp [hi]
    · simp [hi]
  simp only [H, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  unfold a
  field_simp

private theorem plant_phi (hn : 0 < n) (hm : 0 < m) (u : E 1) (r : Fin n) (j : Fin m) (x : E 1) :
    (plant n m μ u r j).φ x = ((μ/2*‖x-u‖^2 - μ/2*‖u‖^2 : ℝ) : EReal) := by
  rw [Instance.φ, plant_f hn hm]
  change ((-μ*⟪u,x⟫ : ℝ):EReal) + ((μ/2*‖x‖^2 : ℝ):EReal) = _
  rw [← EReal.coe_add, EReal.coe_eq_coe_iff, norm_sub_sq_real, real_inner_comm x u]
  ring

private theorem plant_min (hn : 0 < n) (hm : 0 < m) (hμ : 0 ≤ μ) (u : E 1) (r : Fin n)
    (j : Fin m) (x : E 1) : (plant n m μ u r j).φ u ≤ (plant n m μ u r j).φ x := by
  rw [plant_phi hn hm, plant_phi hn hm, EReal.coe_le_coe_iff]
  have hh : 0 ≤ μ/2 * ‖x-u‖^2 := by positivity
  simp
  linarith

private theorem plant_star (hn : 0 < n) (hm : 0 < m) (hμ : 0 ≤ μ) (u : E 1) (r : Fin n)
    (j : Fin m) : (plant n m μ u r j).φstar = (plant n m μ u r j).φ u :=
  Composite.star_eq _ (plant_min hn hm hμ u r j)

def location (μ Δ : ℝ) : E 1 := EuclideanSpace.single 0 (Real.sqrt (2*Δ/μ))

private theorem location_scale (hμ : 0 < μ) (hΔ : 0 ≤ Δ) : μ/2*‖location μ Δ‖^2 = Δ := by
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_one]
  simp only [location, PiLp.single_apply, ite_true, Real.sq_sqrt (show 0 ≤ 2*Δ/μ by positivity)]
  field_simp

private theorem initial_exact (hn : 0 < n) (hm : 0 < m) (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (r : Fin n) (j : Fin m) :
    let I := plant n m μ (location μ Δ) r j
    I.φ 0 - I.φstar = (Δ : EReal) := by
  intro I
  change (plant n m μ (location μ Δ) r j).φ 0 -
    (plant n m μ (location μ Δ) r j).φstar = _
  rw [plant_star hn hm hμ.le, plant_phi hn hm, plant_phi hn hm, ← EReal.coe_sub]
  simp [location_scale hμ hΔ]

theorem inClass (hn : 0 < n) (hm : 0 < m) (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (r : Fin n) (j : Fin m) : (plant n m μ (location μ Δ) r j).InClass L μ Δ := by
  let I := plant n m μ (location μ Δ) r j
  have z := zeroInstance_inClass (n := n) (m := m) (d := 1) hL hμ hΔ
  have fconv : ConvexOn ℝ Set.univ I.f := by
    refine ⟨convex_univ, fun x _ y _ a b ha hb hab => ?_⟩
    simp only [I, plant_f hn hm, inner_add_right, real_inner_smul_right, smul_eq_mul]
    linarith only []
  refine ⟨?_, fconv, z.h_ne_bot, z.h_proper, z.h_closed, z.h_convex, z.prox_exact,
    ?_, ⟨location μ Δ, plant_min hn hm hμ.le _ r j⟩, (initial_exact hn hm hμ hΔ r j).le⟩
  · intro i l
    let a : ℝ := -(n:ℝ)*(m:ℝ)*μ
    let C : E 1 →L[ℝ] ℝ := a • innerSL ℝ (location μ Δ)
    by_cases hh : i = r ∧ l = j
    · have hC : I.F i l = C := by
        funext x
        change (if i = r ∧ l = j then a*⟪location μ Δ,x⟫ else 0) = _
        rw [ite_eq_left hh]
        rfl
      rw [hC]
      refine ⟨C.differentiable, fun x y => ?_⟩
      have hg : gradient C x = gradient C y := by
        apply ext_inner_right ℝ
        intro v
        rw [inner_gradient_left, inner_gradient_left, C.hasFDerivAt.fderiv, C.hasFDerivAt.fderiv]
      simp [hg, mul_nonneg hL (norm_nonneg (x-y))]
    · have hC : I.F i l = fun _ => 0 := by
        funext x
        change (if i = r ∧ l = j then a*⟪location μ Δ,x⟫ else 0) = _
        exact ite_eq_right hh
      rw [hC]
      exact ⟨differentiable_const _, fun x y => by
        simp [mul_nonneg hL (norm_nonneg (x-y))]⟩
  · have hp : I.φ = fun x => ((I.f x + μ/2*‖x‖^2 : ℝ) : EReal) := by
      funext x
      rw [EReal.coe_add]
      rfl
    rw [hp, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    simpa only [add_sub_cancel_right] using fconv

def recorded (A : Algorithm n m c) (r : Fin n) (μ : ℝ) (s : ℕ) (ω : A.Ω) : ℕ :=
  let I := zeroInstance n m 1 μ
  if s % microSteps = commSlot then 0 else
    (A.call 1 (A.info I (J n) ω r) (A.hist I (J n) ω s r)).enc.2.1

def visited (A : Algorithm n m c) (r : Fin n) (μ : ℝ) (T : ℕ) (ω : A.Ω) : Finset ℕ :=
  (Finset.range (microSteps*T)).image (fun s => recorded A r μ s ω)

def missed (A : Algorithm n m c) (r : Fin n) (μ : ℝ) (T : ℕ) (j : Fin m) : Set A.Ω :=
  {ω | j.val ∉ visited A r μ T ω}

private theorem measurable_recorded {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (hn : 1 ≤ n) (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (hρ : 0 ≤ ρ) (r : Fin n) (s : ℕ) :
    Measurable (recorded A r μ s) := by
  by_cases h : s % microSteps = commSlot
  · unfold recorded
    simp only [h, ite_true]
    exact measurable_const
  · unfold recorded
    simp only [h, ite_false]
    exact measurable_fst.comp (measurable_snd.comp (hA.measurable_call 1 _ _
      (zeroInstance_inClass hL hμ hΔ) (J_commClass hn hρ) s r h))

private theorem measurable_missed {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (hn : 1 ≤ n) (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (hρ : 0 ≤ ρ)
    (r : Fin n) (T : ℕ) (j : Fin m) : MeasurableSet (missed A r μ T j) := by
  have H : (missed A r μ T j)ᶜ =
      ⋃ s : Fin (microSteps*T), {ω | recorded A r μ s.val ω = j.val} := by
    ext ω
    simp only [missed, Set.mem_compl_iff, Set.mem_ofPred_eq, not_not, visited, Finset.mem_image,
      Finset.mem_range, Set.mem_iUnion]
    constructor
    · rintro ⟨s, hs, hh⟩
      exact ⟨⟨s, hs⟩, hh⟩
    · rintro ⟨s, hs⟩
      exact ⟨s.val, s.isLt, hs⟩
  apply MeasurableSet.of_compl
  rw [H]
  exact MeasurableSet.iUnion fun s =>
    (measurable_recorded hA hn hL hμ hΔ hρ r s.val) (measurableSet_singleton j.val)

def missingIndices (A : Algorithm n m c) (r : Fin n) (μ : ℝ) (T : ℕ) (ω : A.Ω) : Finset (Fin m) :=
  Finset.univ.filter (fun j => j.val ∉ visited A r μ T ω)

private theorem missed_many (A : Algorithm n m c) (r : Fin n) (μ : ℝ) (T : ℕ) (ω : A.Ω) :
    (missingIndices A r μ T ω).card + microSteps*T ≥ m := by
  classical
  let bad := (Finset.univ : Finset (Fin m)).filter (fun j => j.val ∈ visited A r μ T ω)
  have hb : (bad.image Fin.val).card = bad.card :=
    Finset.card_image_of_injective _ Fin.val_injective
  have hs : bad.image Fin.val ⊆ visited A r μ T ω := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := Finset.mem_image.mp ha
    exact (Finset.mem_filter.mp hj).2
  have hl : (visited A r μ T ω).card ≤ microSteps*T := by
    exact Finset.card_image_le.trans_eq (Finset.card_range _)
  have hh := (Finset.card_le_card hs).trans hl
  rw [hb] at hh
  have h := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin m))) (fun j => j.val ∈ visited A r μ T ω)
  rw [Finset.card_univ, Fintype.card_fin] at h
  have hh' : bad.card + (missingIndices A r μ T ω).card = m := by
    convert h using 1
    congr 1
  omega

private theorem pick_hidden {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (hn : 1 ≤ n) (hm : 1 ≤ m) (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (hρ : 0 ≤ ρ)
    (r : Fin n) (T : ℕ) (hT : 2*(microSteps*T) ≤ m) :
    ∃ j, (1/4 : ℝ≥0∞) ≤ A.P (missed A r μ T j) := by
  classical
  let g (j : Fin m) := (missed A r μ T j).indicator (fun _ => (1 : ℝ≥0∞))
  have hg (j : Fin m) : Measurable (g j) :=
    measurable_const.indicator (measurable_missed hA hn hL hμ hΔ hρ r T j)
  have hgi (j : Fin m) : (∫⁻ ω, g j ω ∂A.P) = A.P (missed A r μ T j) := by
    dsimp only [g]
    rw [lintegral_indicator (measurable_missed hA hn hL hμ hΔ hρ r T j)]
    simp
  have hpoint (ω : A.Ω) : ENNReal.ofReal ((m : ℝ)/2) ≤ ∑ j, g j ω := by
    have hcard := missed_many A r μ T ω
    have hmc : (m : ℝ)/2 ≤ ((missingIndices A r μ T ω).card : ℝ) := by
      have h1 : m ≤ 2*(missingIndices A r μ T ω).card := by omega
      have h2 : (m : ℝ) ≤ 2*((missingIndices A r μ T ω).card : ℝ) := by exact_mod_cast h1
      linarith
    refine (ENNReal.ofReal_le_ofReal hmc).trans ?_
    rw [ENNReal.ofReal_natCast]
    apply le_of_eq
    have H : ∑ j, g j ω = ∑ j ∈ missingIndices A r μ T ω, (1 : ℝ≥0∞) := by
      unfold missingIndices
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro j _
      by_cases hj : j.val ∈ visited A r μ T ω <;> simp [g, missed, hj]
    rw [H, Finset.sum_const, nsmul_one]
  have hs := lintegral_mono hpoint (μ := A.P)
  rw [lintegral_const, measure_univ, mul_one, lintegral_finsetSum] at hs
  · simp only [hgi] at hs
    by_contra! h
    have hh := hs.trans (Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) => (h j).le))
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hh
    have hh' : ENNReal.ofReal ((m : ℝ)/2) ≤ ENNReal.ofReal ((m : ℝ)/4) := by
      have he : (((m : ℕ) : ℝ≥0∞) * (1/4 : ℝ≥0∞)) = ENNReal.ofReal ((m : ℝ)/4) := by
        rw [ENNReal.ofReal_div_of_pos (by norm_num : (0:ℝ) < 4),
          ENNReal.ofReal_natCast, ENNReal.ofReal_ofNat]
        simp [div_eq_mul_inv]
      rwa [he] at hh
    have hm' : (1 : ℝ) ≤ m := by exact_mod_cast hm
    have hbad := (ENNReal.ofReal_le_ofReal_iff (show 0 ≤ (m:ℝ)/4 by positivity)).mp hh'
    linarith
  · intro j _
    exact hg j

private theorem quiet_output {A : Algorithm n m c} (hA : A.Admissible L μ Δ ρ)
    (ω : A.Ω) (t : ℕ) : A.avgOutput (zeroInstance n m 1 μ) (J n) ω t = 0 := by
  let I := zeroInstance n m 1 μ
  have hH := allowed_invariant hA I (J n) ω (fun _ => ∅)
    (by change supp (0 : E 1) ⊆ ∅; simp [supp])
    (fun _ => subset_refl _)
    (fun s i j x hx => by simpa [I, zeroInstance] using (supported_zero (∅ : Set (Fin 1))))
    (fun s v η hv => hv.smul _)
  have hS := avg_from_allowed hA I (J n) ω t (hH _)
  ext k
  exact supported_iff.mp hS k (by simp)

private theorem indistinguishable (A : Algorithm n m c) (r : Fin n) (j : Fin m) (u : E 1)
    (T : ℕ) (ω : A.Ω) (hω : ω ∈ missed A r μ T j) :
    ∀ s ≤ microSteps*T, ∀ i,
      A.hist (plant n m μ u r j) (J n) ω s i = A.hist (zeroInstance n m 1 μ) (J n) ω s i := by
  let I := plant n m μ u r j
  let I₀ := zeroInstance n m 1 μ
  have hinfo (i : Fin n) : A.info I (J n) ω i = A.info I₀ (J n) ω i := rfl
  intro s
  change s ≤ microSteps*T → ∀ i, A.hist I (J n) ω s i = A.hist I₀ (J n) ω s i
  induction s with
  | zero => intros; rfl
  | succ s ih =>
    intro hs i
    have H := ih (by omega : s ≤ microSteps*T)
    simp only [Algorithm.hist, hinfo, H]
    split
    · rfl
    · rename_i ht
      have hnext : Algorithm.answer (c := c) I i
          (A.call 1 (A.info I₀ (J n) ω i) (A.hist I₀ (J n) ω s i)) =
          Algorithm.answer (c := c) I₀ i
          (A.call 1 (A.info I₀ (J n) ω i) (A.hist I₀ (J n) ω s i)) := by
        cases h : A.call 1 (A.info I₀ (J n) ω i) (A.hist I₀ (J n) ω s i) with
        | idle => rfl
        | prox v η => rfl
        | grad l x =>
          have hh : ¬(i = r ∧ l = j) := by
            rintro ⟨rfl,rfl⟩
            apply hω
            refine Finset.mem_image.mpr ⟨s, Finset.mem_range.mpr (by omega), ?_⟩
            simp only [recorded, ht, ite_false, ← show I₀ = zeroInstance n m 1 μ from rfl, h, Call.enc]
          have hf : I.F i l = I₀.F i l := by
            funext v
            simp [I, I₀, plant, hh, zeroInstance]
          simp only [Algorithm.answer, hf]
      rw [hnext]

theorem minimax_hidden (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : μ ≤ L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) (T : ℕ) (hT : 2*(microSteps*T) ≤ m)
    (hε : 0 ≤ ε) (htol : ε < Δ/4) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  apply minimax_ge
  intro A hA
  let r : Fin n := ⟨0,by omega⟩
  let u := location μ Δ
  have hLp : 0 ≤ L := hμ.le.trans hL
  obtain ⟨j,hj⟩ := pick_hidden hA (by omega) hm hLp hμ hΔ.le hρ r T hT
  let I := plant n m μ u r j
  refine ⟨1,I,J n,inClass (by omega) (by omega) hLp hμ hΔ.le r j,
    J_commClass (by omega) hρ,hitting_gt T ?_⟩
  intro t ht
  let B := missed A r μ T j
  have hB := measurable_missed hA (by omega) hLp hμ hΔ.le hρ r T j
  have hpoint (ω : A.Ω) (hω : ω ∈ B) : I.gap (A.avgOutput I (J n) ω t) = ENNReal.ofReal Δ := by
    have he := indistinguishable A r j u T ω hω (microSteps*t) (Nat.mul_le_mul_left _ ht)
    have hv : A.avgOutput I (J n) ω t = 0 := by
      have hinfo (i : Fin n) : A.info I (J n) ω i =
          A.info (zeroInstance n m 1 μ) (J n) ω i := rfl
      unfold Algorithm.avgOutput Algorithm.output
      simp only [I, ← show I = plant n m μ u r j from rfl, he, hinfo]
      exact quiet_output hA ω t
    rw [hv, Instance.gap, initial_exact (by omega : 0 < n) (by omega : 0 < m) hμ hΔ.le r j]
    simp
  have hsum : (∫⁻ ω, B.indicator (fun _ => ENNReal.ofReal Δ) ω ∂A.P) ≤
      A.expectedGap I (J n) t := by
    apply lintegral_mono
    intro ω
    by_cases hh : ω ∈ B
    · rw [Set.indicator_of_mem hh]
      exact le_of_eq (hpoint ω hh).symm
    · rw [Set.indicator_of_notMem hh]
      exact zero_le
  rw [lintegral_indicator hB, lintegral_const, Measure.restrict_apply_univ] at hsum
  have hfour : ENNReal.ofReal (Δ/4) ≤ A.expectedGap I (J n) t := by
    rw [ENNReal.ofReal_div_of_pos (by norm_num : (0:ℝ)<4), ENNReal.ofReal_ofNat, div_eq_mul_inv]
    refine le_trans ?_ hsum
    rw [one_div] at hj
    gcongr
  exact ((ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε).2 htol).trans_le hfour

end Hidden
end ULB63
