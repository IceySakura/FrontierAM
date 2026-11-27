import Theory.Defs
import Theory.Preint
import Theory.Cube

/-!
# A lower bound for deterministic rules after non-monotone preintegration (auxiliary proofs for R11)

For any nodes `y₁, …, yₙ ∈ ℝ^s` and any `m` with `2n ≤ m^s` there is `φ` in the non-monotone class
(2-strongly concave in `t`, 1-Lipschitz in `y`) whose preintegrated jump function vanishes at all
nodes but has integral `≥ c_s / m` over `[0,1)^s`.
-/

open MeasureTheory ProbabilityTheory Real Set Function Filter Topology Metric
open scoped ENNReal

namespace Cubature.Aux

/-- Centre of the cell `k`. -/
noncomputable def cellCenter (s m : ℕ) (k : Fin s → Fin m) : Fin s → ℝ :=
  fun j => ((k j : ℝ) + 1 / 2) / m

lemma ball_subset_cell (s m : ℕ) (hm : 0 < m) (k : Fin s → Fin m) :
    ball (cellCenter s m k) (1 / (2 * m)) ⊆ cubeCell s m k := by
  intro z hz
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  rw [mem_ball, dist_pi_lt_iff (by positivity)] at hz
  refine Set.mem_univ_pi.mpr fun j => ?_
  have h := hz j
  rw [Real.dist_eq, abs_lt] at h
  simp only [cellCenter] at h
  have e1 : ((k j : ℝ) + 1 / 2) / m - 1 / (2 * m) = (k j : ℝ) / m := by field_simp; ring
  have e2 : ((k j : ℝ) + 1 / 2) / m + 1 / (2 * m) = ((k j : ℝ) + 1) / m := by field_simp; ring
  constructor <;> linarith

lemma closedBall_subset_cube (s m : ℕ) (hm : 0 < m) (k : Fin s → Fin m) :
    closedBall (cellCenter s m k) (1 / (4 * m)) ⊆ Set.pi univ fun _ => Ico (0 : ℝ) 1 := by
  intro z hz
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  rw [mem_closedBall, dist_pi_le_iff (by positivity)] at hz
  refine Set.mem_univ_pi.mpr fun j => ?_
  have h := hz j
  rw [Real.dist_eq, abs_le] at h
  simp only [cellCenter] at h
  have hk : ((k j : ℕ) : ℝ) + 1 ≤ m := by exact_mod_cast (k j).isLt
  have hk0 : (0 : ℝ) ≤ (k j : ℕ) := Nat.cast_nonneg _
  constructor
  · have : (0 : ℝ) ≤ ((k j : ℝ) + 1 / 2) / m - 1 / (4 * m) := by
      rw [div_sub_div _ _ hmR.ne' (by positivity)]
      apply div_nonneg _ (by positivity)
      nlinarith
    linarith
  · have : ((k j : ℝ) + 1 / 2) / m + 1 / (4 * m) < 1 := by
      rw [div_add_div _ _ hmR.ne' (by positivity), div_lt_one (by positivity)]
      nlinarith
    linarith

theorem fooling (s n m : ℕ) (hs : 0 < s) (hm : 0 < m) (hnm : 2 * n ≤ m ^ s) (y : Fin n → (Fin s → ℝ)) :
    ∃ φ : ℝ → (Fin s → ℝ) → ℝ,
      (∀ z, StrongConcaveOn univ 2 (φ · z)) ∧
      (∀ t z z', |φ t z - φ t z'| ≤ 1 * dist z z') ∧
      (∀ i t, φ t (y i) ≤ 0) ∧
      (∀ i, preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) (y i) = 0) ∧
      √2 * rexp (-1 / 2) / √(2 * π) / (2 ^ (s + 2) * m) ≤
        ∫ z in Set.pi univ (fun _ => Ico (0 : ℝ) 1),
          preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) z := by
  classical
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  set δ : ℝ := 1 / (2 * m) with hδ
  have hδ0 : 0 < δ := by positivity
  have hδ1 : δ ≤ 1 / 2 := by
    rw [hδ, div_le_div_iff₀ (by positivity) (by norm_num)]
    have : (1 : ℝ) ≤ m := by exact_mod_cast hm
    linarith
  set E : Finset (Fin s → Fin m) := Finset.univ.filter (fun k => ∀ i, y i ∉ cubeCell s m k)
    with hE
  set U : Set (Fin s → ℝ) := ⋃ k ∈ E, ball (cellCenter s m k) δ with hU
  -- `Uᶜ` is nonempty
  have hUc : (Uᶜ).Nonempty := by
    refine ⟨fun _ => -1, ?_⟩
    simp only [hU, mem_compl_iff, mem_iUnion, not_exists]
    intro k _ hk
    have := ball_subset_cell s m hm k (by rw [hδ] at hk; exact hk)
    have h0 := (Set.mem_univ_pi.mp this) ⟨0, hs⟩
    simp only [mem_Ico] at h0
    have : (0 : ℝ) ≤ ((k ⟨0, hs⟩ : ℕ) : ℝ) / m := by positivity
    linarith [h0.1]
  set φ : ℝ → (Fin s → ℝ) → ℝ := fun t z => δ * infDist z Uᶜ - t ^ 2 with hφdef
  have hconc : ∀ z, StrongConcaveOn univ 2 (φ · z) := by
    intro z
    rw [strongConcaveOn_iff_convex]
    have : (fun t : ℝ => φ t z + 2 / 2 * ‖t‖ ^ 2) = fun _ => δ * infDist z Uᶜ := by
      funext t; simp only [hφdef]; rw [Real.norm_eq_abs, sq_abs]; ring
    rw [this]; exact concaveOn_const _ convex_univ
  have hlip : ∀ t z z', |φ t z - φ t z'| ≤ 1 * dist z z' := by
    intro t z z'
    simp only [hφdef]
    have h1 := infDist_le_infDist_add_dist (x := z) (y := z') (s := Uᶜ)
    have h2 := infDist_le_infDist_add_dist (x := z') (y := z) (s := Uᶜ)
    rw [dist_comm] at h2
    have h3 : |infDist z Uᶜ - infDist z' Uᶜ| ≤ dist z z' :=
      abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩
    calc |δ * infDist z Uᶜ - t ^ 2 - (δ * infDist z' Uᶜ - t ^ 2)|
        = δ * |infDist z Uᶜ - infDist z' Uᶜ| := by
          rw [show δ * infDist z Uᶜ - t ^ 2 - (δ * infDist z' Uᶜ - t ^ 2) =
            δ * (infDist z Uᶜ - infDist z' Uᶜ) by ring, abs_mul, abs_of_pos hδ0]
      _ ≤ δ * dist z z' := by gcongr
      _ ≤ 1 * dist z z' := by
          have := dist_nonneg (x := z) (y := z')
          nlinarith
  have hPset : ∀ z, {t : ℝ | 0 < φ t z} =
      Ioo (-√(δ * infDist z Uᶜ)) (√(δ * infDist z Uᶜ)) := by
    intro z; ext t
    simp only [hφdef, mem_ofPred_eq, mem_Ioo]
    have hw : 0 ≤ δ * infDist z Uᶜ := mul_nonneg hδ0.le infDist_nonneg
    constructor
    · intro h
      have : t ^ 2 < (√(δ * infDist z Uᶜ)) ^ 2 := by rw [Real.sq_sqrt hw]; linarith
      exact abs_lt.mp (abs_lt_of_sq_lt_sq' this (Real.sqrt_nonneg _) |> fun h => abs_lt.mpr h)
    · intro h
      have h1 : |t| < √(δ * infDist z Uᶜ) := abs_lt.mpr h
      have h2 : t ^ 2 < (√(δ * infDist z Uᶜ)) ^ 2 := by
        rw [← sq_abs]; exact pow_lt_pow_left₀ h1 (abs_nonneg t) two_ne_zero
      rw [Real.sq_sqrt hw] at h2; linarith
  have hP : ∀ z, preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) z =
      (gaussianReal 0 1).real (Ioo (-√(δ * infDist z Uᶜ)) (√(δ * infDist z Uᶜ))) := by
    intro z
    rw [preintJump, hPset z, integral_indicator measurableSet_Ioo, setIntegral_const,
      smul_eq_mul, mul_one]
  have hyU : ∀ i, y i ∈ Uᶜ := by
    intro i
    simp only [hU, mem_compl_iff, mem_iUnion, not_exists]
    intro k hk hball
    have hk' := (Finset.mem_filter.mp hk).2 i
    exact hk' (ball_subset_cell s m hm k (by rw [hδ] at hball; exact hball))
  refine ⟨φ, hconc, hlip, fun i t => ?_, fun i => ?_, ?_⟩
  · simp only [hφdef, infDist_zero_of_mem (hyU i), mul_zero, zero_sub, neg_nonpos]
    exact sq_nonneg t
  · -- the nodes lie outside `U`
    rw [hP, infDist_zero_of_mem (hyU i), mul_zero, Real.sqrt_zero, neg_zero, Ioo_self,
      measureReal_empty]
  · -- lower bound for the integral
    set F := preintJump (gaussianReal 0 1) φ (fun _ _ => (1 : ℝ)) with hF
    set cube := Set.pi univ (fun _ : Fin s => Ico (0 : ℝ) 1) with hcube
    have hcm : MeasurableSet cube := MeasurableSet.univ_pi fun _ => measurableSet_Ico
    have hcfin : volume cube ≠ ⊤ := by
      rw [hcube, Real.volume_pi_Ico]; simp
    -- continuity (hence measurability) of `F`
    have hhol := jump_holder (Y := Fin s → ℝ) (ρ := (√(2 * π))⁻¹) (by positivity)
      gaussianReal_le_volume (φ := φ) (θ := fun _ _ => (1 : ℝ)) (α := 2) (L := 1) (Lθ := 0)
      (B := 1) (by norm_num) (fun z => Or.inr (hconc z)) hlip (fun _ => measurable_const)
      (fun _ _ => by simp) (fun _ _ _ => by simp)
    have hFc : Continuous F := by
      rw [continuous_iff_continuousAt]
      intro y0
      rw [ContinuousAt, tendsto_iff_dist_tendsto_zero]
      have hg : Continuous fun d : ℝ => 0 * d + 4 * (√(2 * π))⁻¹ * 1 * √(1 * d / 2) := by
        fun_prop
      have h1 : Tendsto (fun z => dist z y0) (𝓝 y0) (𝓝 0) := by
        simpa using (tendsto_id.dist (tendsto_const_nhds (x := y0)) :
          Tendsto (fun z => dist z y0) (𝓝 y0) (𝓝 (dist y0 y0)))
      have h2 := (hg.tendsto 0).comp h1
      simp only [Function.comp_def, mul_zero, zero_div, Real.sqrt_zero, add_zero] at h2
      refine squeeze_zero (fun z => dist_nonneg) (fun z => ?_) h2
      rw [Real.dist_eq]
      exact hhol z y0
    have hF0 : ∀ z, 0 ≤ F z := fun z => by rw [hP]; exact measureReal_nonneg
    have hF1 : ∀ z, F z ≤ 1 := fun z => by rw [hP]; exact measureReal_le_one
    have hFint : IntegrableOn F cube := IntegrableOn.of_bound (lt_top_iff_ne_top.mpr hcfin)
      hFc.measurable.aestronglyMeasurable 1
      (ae_of_all _ fun z => by rw [Real.norm_eq_abs, abs_of_nonneg (hF0 z)]; exact hF1 z)
    -- lower bound of `F` on the inner boxes
    set κ : ℝ := rexp (-1 / 2) / √(2 * π) * (2 * (δ / √2)) with hκ
    have hFlow : ∀ k ∈ E, ∀ z ∈ closedBall (cellCenter s m k) (δ / 2), κ ≤ F z := by
      intro k hk z hz
      have hinf : δ / 2 ≤ infDist z Uᶜ := by
        refine (le_infDist hUc).mpr fun w hw => ?_
        have hw' : w ∉ ball (cellCenter s m k) δ := by
          intro hb
          exact hw (mem_iUnion.mpr ⟨k, mem_iUnion.mpr ⟨hk, hb⟩⟩)
        rw [mem_ball, not_lt] at hw'
        rw [mem_closedBall] at hz
        have := dist_triangle w z (cellCenter s m k)
        rw [dist_comm w z] at this
        linarith
      set c := δ / √2 with hc
      have hc0 : 0 < c := by positivity
      have hc2 : c ^ 2 = δ ^ 2 / 2 := by rw [hc, div_pow, Real.sq_sqrt (by norm_num)]
      have hcw : c ≤ √(δ * infDist z Uᶜ) := by
        rw [Real.le_sqrt hc0.le (mul_nonneg hδ0.le infDist_nonneg), hc2]
        nlinarith
      have hc1 : c ≤ 1 := by
        rw [hc, div_le_one (by positivity)]
        have : (1 : ℝ) ≤ √2 := by rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
        linarith
      rw [hP]
      have hsub : Ioo (-c) c ⊆ Ioo (-√(δ * infDist z Uᶜ)) (√(δ * infDist z Uᶜ)) :=
        Ioo_subset_Ioo (by linarith) hcw
      refine le_trans ?_ (measureReal_mono hsub (measure_ne_top _ _))
      have hlow : ∀ x ∈ Ioo (-c) c, ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) ≤
          gaussianPDF 0 1 x := by
        intro x hx
        rw [gaussianPDF, gaussianPDFReal]
        refine ENNReal.ofReal_le_ofReal ?_
        simp only [NNReal.coe_one, mul_one, sub_zero]
        have hx2 : x ^ 2 ≤ 1 := by
          have : |x| ≤ 1 := by rw [abs_le]; constructor <;> linarith [hx.1, hx.2]
          nlinarith [abs_nonneg x, sq_abs x]
        have : rexp (-1 / 2) ≤ rexp (-x ^ 2 / 2) := Real.exp_le_exp.mpr (by linarith)
        rw [div_eq_inv_mul]
        exact mul_le_mul_of_nonneg_left this (by positivity)
      have hmeas : ENNReal.ofReal (rexp (-1 / 2) / √(2 * π)) * volume (Ioo (-c) c) ≤
          gaussianReal 0 1 (Ioo (-c) c) := by
        rw [gaussianReal_apply 0 one_ne_zero, ← setLIntegral_const]
        exact setLIntegral_mono (by fun_prop) hlow
      rw [Real.volume_Ioo, ← ENNReal.ofReal_mul (by positivity)] at hmeas
      have := (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top _ _)).mp hmeas
      calc κ = rexp (-1 / 2) / √(2 * π) * (c - -c) := by rw [hκ]; ring
        _ ≤ _ := this
    -- the inner boxes are disjoint, measurable, inside the cube
    have hBsub : ∀ k, closedBall (cellCenter s m k) (δ / 2) ⊆ cube := by
      intro k
      have e : δ / 2 = 1 / (4 * m) := by rw [hδ]; field_simp; ring
      rw [e]; exact closedBall_subset_cube s m hm k
    have hBcell : ∀ k, closedBall (cellCenter s m k) (δ / 2) ⊆ cubeCell s m k := by
      intro k z hz
      apply ball_subset_cell s m hm k
      rw [mem_ball]; rw [mem_closedBall] at hz
      have : δ / 2 < 1 / (2 * m) := by rw [← hδ]; linarith
      linarith
    have hBdisj : Set.PairwiseDisjoint (↑E) fun k => closedBall (cellCenter s m k) (δ / 2) := by
      intro k _ k' _ hkk'
      exact (cubeCell_disjoint s m hkk').mono (hBcell k) (hBcell k')
    have hBvol : ∀ k, volume.real (closedBall (cellCenter s m k) (δ / 2)) = δ ^ s := by
      intro k
      rw [Measure.real, Real.volume_pi_closedBall _ (by positivity), Fintype.card_fin,
        ENNReal.toReal_ofReal (by positivity)]
      congr 1; ring
    -- integral over the cube ≥ sum over the inner boxes
    have hU1 : ∫ z in ⋃ k ∈ E, closedBall (cellCenter s m k) (δ / 2), F z ≤ ∫ z in cube, F z := by
      refine setIntegral_mono_set hFint (ae_of_all _ fun z => hF0 z) ?_
      exact Eventually.of_forall (fun z hz => by
        obtain ⟨k, hk⟩ := mem_iUnion.mp hz
        obtain ⟨_, hz'⟩ := mem_iUnion.mp hk
        exact hBsub k hz')
    have hU2 : ∫ z in ⋃ k ∈ E, closedBall (cellCenter s m k) (δ / 2), F z =
        ∑ k ∈ E, ∫ z in closedBall (cellCenter s m k) (δ / 2), F z :=
      integral_biUnion_finset E (fun k _ => isClosed_closedBall.measurableSet) hBdisj
        (fun k _ => hFint.mono_set (hBsub k))
    have hU3 : ∀ k ∈ E, κ * δ ^ s ≤ ∫ z in closedBall (cellCenter s m k) (δ / 2), F z := by
      intro k hk
      have := setIntegral_ge_of_const_le_real (μ := volume) isClosed_closedBall.measurableSet
        (measure_ne_top_of_subset (hBsub k) hcfin) (hFlow k hk) (hFint.mono_set (hBsub k))
      rwa [hBvol k] at this
    -- counting empty cells
    have hcardE : (m : ℝ) ^ s / 2 ≤ (E.card : ℝ) := by
      have hsplit := Finset.card_filter_add_card_filter_not
        (s := (Finset.univ : Finset (Fin s → Fin m))) (fun k => ∀ i, y i ∉ cubeCell s m k)
      have hocc : (Finset.univ.filter (fun k : Fin s → Fin m => ¬ ∀ i, y i ∉ cubeCell s m k)).card
          ≤ n := by
        have hsub : Finset.univ.filter (fun k : Fin s → Fin m => ¬ ∀ i, y i ∉ cubeCell s m k) ⊆
            Finset.univ.biUnion (fun i => Finset.univ.filter (fun k => y i ∈ cubeCell s m k)) := by
          intro k hk
          rw [Finset.mem_filter] at hk
          push Not at hk
          obtain ⟨i, hi⟩ := hk.2
          exact Finset.mem_biUnion.mpr ⟨i, Finset.mem_univ _, Finset.mem_filter.mpr
            ⟨Finset.mem_univ _, hi⟩⟩
        refine (Finset.card_le_card hsub).trans ((Finset.card_biUnion_le).trans ?_)
        calc ∑ i, (Finset.univ.filter (fun k => y i ∈ cubeCell s m k)).card
            ≤ ∑ _i : Fin n, 1 := by
              refine Finset.sum_le_sum fun i _ => Finset.card_le_one.mpr ?_
              intro a ha b hb
              rw [Finset.mem_filter] at ha hb
              by_contra hab
              exact Set.disjoint_left.mp (cubeCell_disjoint s m hab) ha.2 hb.2
          _ = n := by simp
      have hcu : (Finset.univ : Finset (Fin s → Fin m)).card = m ^ s := by simp
      rw [hcu] at hsplit
      have h1 : (m ^ s : ℕ) ≤ E.card + n := by rw [hE]; omega
      have h2 : ((m ^ s : ℕ) : ℝ) ≤ (E.card : ℝ) + n := by exact_mod_cast h1
      have h3 : (2 * n : ℝ) ≤ (m : ℝ) ^ s := by exact_mod_cast hnm
      push_cast at h2
      linarith
    -- assemble
    have hsum : (E.card : ℝ) * (κ * δ ^ s) ≤ ∫ z in cube, F z := by
      calc (E.card : ℝ) * (κ * δ ^ s) = ∑ _k ∈ E, κ * δ ^ s := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ ∑ k ∈ E, ∫ z in closedBall (cellCenter s m k) (δ / 2), F z :=
            Finset.sum_le_sum hU3
        _ = _ := hU2.symm
        _ ≤ _ := hU1
    have hκ0 : 0 ≤ κ * δ ^ s := by positivity
    refine le_trans ?_ hsum
    have hs2 : √2 * √2 = 2 := Real.mul_self_sqrt (by norm_num)
    have hs0 : √2 ≠ 0 := by positivity
    have h2c : 2 * (δ / √2) = √2 * δ := by
      rw [mul_div_assoc', div_eq_iff hs0, mul_comm (√2) δ, mul_assoc, hs2]; ring
    have hκ' : κ = rexp (-1 / 2) / √(2 * π) * (√2 * δ) := by rw [hκ, h2c]
    calc √2 * rexp (-1 / 2) / √(2 * π) / (2 ^ (s + 2) * m)
        = (m : ℝ) ^ s / 2 * (κ * δ ^ s) := by
          rw [hκ', hδ]
          field_simp
          have e1 : (m : ℝ) ^ s * ((m : ℝ))⁻¹ ^ s = 1 := by
            rw [← mul_pow, mul_inv_cancel₀ hmR.ne', one_pow]
          have e2 : ((1 : ℝ) / 2) ^ s * 2 ^ s = 1 := by rw [← mul_pow]; norm_num
          ring_nf
          ring_nf at e1 e2
          linear_combination (-4 * ((m : ℝ) ^ s * ((m : ℝ))⁻¹ ^ s)) * e2 + (-4) * e1
      _ ≤ (E.card : ℝ) * (κ * δ ^ s) := mul_le_mul_of_nonneg_right hcardE hκ0

end Cubature.Aux
