import Theory.BlockGeometry

/-! An exact gap normalization, and a bound from one omitted coordinate of a path. -/

open Finset
open scoped BigOperators ENNReal

noncomputable section

namespace ULB63.BlockGeometry

variable {n m p : ℕ}

def stiffness (l q : ℝ) : ℝ := l * (1 - q) / q

def curvature (l q : ℝ) : ℝ := l * (1 - q)^2 / q

def energy (l q a : ℝ) (b : Blocks n m) (x : E (Dim n m p)) : ℝ :=
  stiffness l q / 2 * (coord x b 0 - a)^2 +
    (l / (2 * q)) * ∑ e : Fin p, (coord x b e.succ - q * coord x b e.castSucc)^2

theorem energy_nonneg {l q a : ℝ} (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q ≤ 1)
    (b : Blocks n m) (x : E (Dim n m p)) :
    0 ≤ energy l q a b x := by
  unfold energy stiffness
  positivity [sub_nonneg.mpr hq1, sum_nonneg (s := (univ : Finset (Fin p)))
    (fun e _ => sq_nonneg (coord x b e.succ - q * coord x b e.castSucc))]

theorem energy_identity {l μ q a : ℝ} (hq : q ≠ 0) (hμ : μ = curvature l q)
    (b : Blocks n m) (x : E (Dim n m p)) :
    block l q a b x + μ / 2 * (∑ j : Fin (p + 1), (coord x b j)^2) +
      stiffness l q / 2 * a^2 = energy l q a b x := by
  have edge (u v : ℝ) :
      l / (2 * q) * (v - q * u)^2 =
        l / 2 * (v - u)^2 + (stiffness l q / 2) * v^2 - (l * (1 - q) / 2) * u^2 := by
    unfold stiffness
    field_simp
    ring
  have hright : (∑ e : Fin p, (coord x b e.succ)^2) =
      (∑ j : Fin (p + 1), (coord x b j)^2) - (coord x b 0)^2 := by
    rw [Fin.sum_univ_succ]
    ring
  have hleft : (∑ e : Fin p, (coord x b e.castSucc)^2) =
      (∑ j : Fin (p + 1), (coord x b j)^2) - (coord x b (Fin.last p))^2 := by
    rw [Fin.sum_univ_castSucc]
    ring
  have hEdgeSum : l / (2 * q) * ∑ e : Fin p, (coord x b e.succ - q * coord x b e.castSucc)^2 =
      l / 2 * (∑ e : Fin p, (coord x b e.succ - coord x b e.castSucc)^2) +
        (stiffness l q / 2) * (∑ e : Fin p, (coord x b e.succ)^2) -
        (l * (1 - q) / 2) * (∑ e : Fin p, (coord x b e.castSucc)^2) := by
    simp only [mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
    exact sum_congr rfl fun e _ => edge _ _
  unfold energy block
  rw [hEdgeSum, hright, hleft, EuclideanSpace.real_norm_sq_eq, hμ]
  simp only [diff_apply, stiffness, curvature]
  field_simp
  ring

/-- One step of the coordinate-error estimate for the residual decomposition. -/
private theorem error_step {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1) (e r : ℝ) :
    (q * e + r)^2 ≤ e^2 + (1 - q)⁻¹ * r^2 := by
  have hpos : 0 < 1 - q := sub_pos.mpr hq1
  apply (mul_le_mul_iff_left₀ hpos).mp
  have hcancel : (1 - q) * (1 - q)⁻¹ = 1 :=
    mul_inv_cancel₀ hpos.ne'
  have h :
      (1 - q) * (e^2 + (1 - q)⁻¹ * r^2) - (1 - q) * (q * e + r)^2 =
        q * ((1 - q) * e - r)^2 + (1 - q)^2 * e^2 := by
    rw [mul_add, ← mul_assoc (1 - q) (1 - q)⁻¹, hcancel]
    ring
  have hh : 0 ≤ (1 - q) * (e^2 + (1 - q)⁻¹ * r^2) - (1 - q) * (q * e + r)^2 := by
    rw [h]
    positivity
  simpa only [mul_comm] using sub_nonneg.mp hh

private def residualSum (q : ℝ) (v : Fin (p + 1) → ℝ) (j : ℕ) : ℝ :=
  ∑ e ∈ univ.filter (fun e : Fin p => e.val < j), (v e.succ - q * v e.castSucc)^2

private theorem residualSum_succ (q : ℝ) (v : Fin (p+1) → ℝ) (j : Fin p) :
    residualSum q v (j.val + 1) = residualSum q v j.val + (v j.succ - q * v j.castSucc)^2 := by
  have hs : univ.filter (fun e : Fin p => e.val < j.val + 1) =
      insert j (univ.filter fun e : Fin p => e.val < j.val) := by
    ext e
    simp only [mem_filter, mem_univ, true_and, mem_insert]
    constructor
    · intro h
      by_cases he : e.val = j.val
      · exact Or.inl (Fin.ext he)
      · right; omega
    · rintro (rfl | h) <;> omega
  rw [residualSum, hs, sum_insert (by simp), residualSum, add_comm]

private theorem error_prefix {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1)
    (v : Fin (p + 1) → ℝ) (a : ℝ) (j : Fin (p + 1)) :
    (v j - a * q^j.val)^2 ≤ (v 0 - a)^2 + (1 - q)⁻¹ * residualSum q v j.val := by
  refine Fin.induction ?_ (fun e ih => ?_) j
  · simp [residualSum]
  · have he : v e.succ - a * q^(e.succ.val) =
        q * (v e.castSucc - a * q^(e.castSucc.val)) + (v e.succ - q * v e.castSucc) := by
      simp only [Fin.val_succ, Fin.val_castSucc, pow_succ]
      ring
    rw [he]
    refine (error_step hq0 hq1 _ _).trans ?_
    simp only [Fin.val_succ, Fin.val_castSucc] at ih ⊢
    rw [residualSum_succ, mul_add, ← add_assoc]
    exact add_le_add_left ih _

theorem energy_of_zero_coord {l q a : ℝ} (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q < 1)
    (b : Blocks n m) (x : E (Dim n m p)) (j : Fin (p + 1)) (hj : coord x b j = 0) :
    stiffness l q / 2 * a^2 * q^(2 * j.val) ≤ energy l q a b x := by
  have ht : 0 < 1 - q := sub_pos.mpr hq1
  have hs : residualSum q (coord x b) j.val ≤
      ∑ e : Fin p, (coord x b e.succ - q * coord x b e.castSucc)^2 := by
    exact sum_le_sum_of_subset_of_nonneg (filter_subset _ _) (fun e _ _ => sq_nonneg _)
  have h := error_prefix hq0.le hq1 (coord x b) a j
  rw [hj, zero_sub, neg_sq, mul_pow] at h
  have h' := (mul_le_mul_of_nonneg_left (h.trans (add_le_add_right
    (mul_le_mul_of_nonneg_left hs (by positivity : 0 ≤ (1 - q)⁻¹)) _))
      (by unfold stiffness; positivity : 0 ≤ stiffness l q / 2))
  have hc : stiffness l q / 2 * (1 - q)⁻¹ = l / (2 * q) := by
    unfold stiffness
    field_simp
  rw [energy, pow_mul, mul_assoc (stiffness l q / 2)]
  calc
    _ ≤ stiffness l q / 2 * ((coord x b 0 - a)^2 + (1 - q)⁻¹ *
        ∑ e : Fin p, (coord x b e.succ - q * coord x b e.castSucc)^2) := by
      simpa only [← pow_mul, Nat.mul_comm j.val 2] using h'
    _ = _ := by rw [mul_add, ← mul_assoc _ (1 - q)⁻¹, hc]

def blockInstance (l μ q a : ℝ) (n m p : ℕ) : Instance n m (Dim n m p) where
  F i j x := ((n : ℝ) * m) * block l q a (i, j) x
  h x := ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1 + η * μ)⁻¹ • v

theorem blockInstance_f {l μ q a : ℝ} (hn : n ≠ 0) (hm : m ≠ 0) (x : E (Dim n m p)) :
    (blockInstance l μ q a n m p).f x = ∑ b : Blocks n m, block l q a b x := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  simp only [Instance.f, blockInstance, Fintype.sum_prod_type, ← mul_sum]
  field_simp

theorem blockInstance_value {l μ q a : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hq : q ≠ 0) (hμ : μ = curvature l q) (x : E (Dim n m p)) :
    (blockInstance l μ q a n m p).φ x =
      (((∑ b : Blocks n m, energy l q a b x) - ((n : ℝ) * m) * (stiffness l q / 2 * a^2) : ℝ) : EReal) := by
  rw [Instance.φ, blockInstance_f hn hm]
  change (↑(∑ b : Blocks n m, block l q a b x) : EReal) + ((μ / 2 * ‖x‖ ^ 2 : ℝ) : EReal) = _
  rw [← EReal.coe_add, EReal.coe_eq_coe_iff]
  rw [norm_sq_blocks, mul_sum, ← sum_add_distrib]
  have hsum := sum_congr (s₁ := (univ : Finset (Blocks n m))) rfl
    (fun b _ => energy_identity (a := a) hq hμ b x)
  rw [sum_add_distrib, sum_const, nsmul_eq_mul, card_univ, Fintype.card_prod,
    Fintype.card_fin, Fintype.card_fin, Nat.cast_mul] at hsum
  linarith only [hsum]

def blockMinimizer (a q : ℝ) : E (Dim n m p) :=
  WithLp.toLp 2 ((fun bj : Blocks n m × Fin (p + 1) => a * q^bj.2.val) ∘
    (Fintype.equivFin (Blocks n m × Fin (p + 1))).symm)

@[simp] theorem coord_blockMinimizer (a q : ℝ) (b : Blocks n m) (j : Fin (p + 1)) :
    coord (blockMinimizer a q) b j = a * q^j.val := by
  simp [coord, blockMinimizer, idx]

theorem energy_at_minimizer (l q a : ℝ) (b : Blocks n m) :
    energy (p := p) l q a b (blockMinimizer a q) = 0 := by
  simp [energy, pow_succ, mul_left_comm q a, mul_comm q]

theorem blockInstance_minimizer {l μ q a : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q ≤ 1) (hμ : μ = curvature l q) (x : E (Dim n m p)) :
    (blockInstance l μ q a n m p).φ (blockMinimizer a q) ≤ (blockInstance l μ q a n m p).φ x := by
  rw [blockInstance_value hn hm hq0.ne' hμ, blockInstance_value hn hm hq0.ne' hμ]
  norm_cast
  simpa [energy_at_minimizer] using
    (sum_nonneg (s := (univ : Finset (Blocks n m))) fun b _ => energy_nonneg hl hq0 hq1 b x)

theorem blockInstance_φstar {l μ q a : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q ≤ 1) (hμ : μ = curvature l q) :
    (blockInstance l μ q a n m p).φstar =
      (((-((n : ℝ) * m) * (stiffness l q / 2 * a^2)) : ℝ) : EReal) := by
  have h : (blockInstance l μ q a n m p).φstar =
      (blockInstance l μ q a n m p).φ (blockMinimizer a q) :=
    le_antisymm (iInf_le _ _) (le_iInf (blockInstance_minimizer hn hm hl hq0 hq1 hμ))
  rw [h, blockInstance_value hn hm hq0.ne' hμ]
  simp [energy_at_minimizer]

theorem blockInstance_gap {l μ q a : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q ≤ 1) (hμ : μ = curvature l q) (x : E (Dim n m p)) :
    (blockInstance l μ q a n m p).gap x = ENNReal.ofReal (∑ b : Blocks n m, energy l q a b x) := by
  rw [Instance.gap, blockInstance_value hn hm hq0.ne' hμ, blockInstance_φstar hn hm hl hq0 hq1 hμ,
    ← EReal.coe_sub]
  rw [EReal.real_coe_toENNReal]
  congr 1
  ring

theorem blockInstance_inClass {l μ q a L Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hl : 0 ≤ l) (hq0 : 0 < q) (hq1 : q ≤ 1) (hμ0 : 0 < μ) (hμ : μ = curvature l q)
    (hL : ((n : ℝ) * m) * (5 * l) ≤ L)
    (hΔ : ((n : ℝ) * m) * (stiffness l q / 2 * a^2) ≤ Δ) :
    (blockInstance l μ q a n m p).InClass L μ Δ := by
  let I := blockInstance l μ q a n m p
  have hz : (zeroInstance n m (Dim n m p) μ).InClass L μ Δ :=
    zeroInstance_inClass ((by positivity : 0 ≤ ((n : ℝ) * m) * (5 * l)).trans hL)
      hμ0 ((by unfold stiffness; positivity [sub_nonneg.mpr hq1] :
        0 ≤ ((n : ℝ) * m) * (stiffness l q / 2 * a^2)).trans hΔ)
  have hf : ConvexOn ℝ Set.univ I.f := by
    have he : I.f = fun x => ∑ b : Blocks n m, block l q a b x :=
      funext (blockInstance_f hn hm)
    rw [he]
    refine ⟨convex_univ, ?_⟩
    intro x _ y _ t u ht hu htu
    simp only [smul_eq_mul, mul_sum, ← sum_add_distrib]
    exact sum_le_sum fun b _ => (convex_block hl hq1 b).2 (Set.mem_univ x) (Set.mem_univ y) ht hu htu
  refine ⟨fun i ξ => scaled_block_smooth (by positivity) hl hq0.le hq1 hL (i, ξ),
    hf, hz.h_ne_bot, hz.h_proper, hz.h_closed, hz.h_convex, hz.prox_exact, ?_, ?_, ?_⟩
  · have he : I.φ = fun x => (((I.f x + μ / 2 * ‖x‖ ^ 2) : ℝ) : EReal) := by
      funext x
      simp [I, Instance.φ, blockInstance]
    rw [he, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    simpa using hf
  · exact ⟨blockMinimizer a q, blockInstance_minimizer hn hm hl hq0 hq1 hμ⟩
  · change (blockInstance l μ q a n m p).φ 0 - _ ≤ _
    rw [Instance.φ, blockInstance_f hn hm, blockInstance_φstar hn hm hl hq0 hq1 hμ]
    change (↑(∑ b : Blocks n m, block l q a b (0 : E (Dim n m p))) : EReal) +
      ((μ / 2 * ‖(0 : E (Dim n m p))‖ ^ 2 : ℝ) : EReal) - _ ≤ _
    simpa [block] using EReal.coe_le_coe_iff.mpr hΔ

end ULB63.BlockGeometry
