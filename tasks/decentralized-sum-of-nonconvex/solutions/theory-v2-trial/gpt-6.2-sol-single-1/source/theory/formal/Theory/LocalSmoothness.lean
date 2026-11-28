import Theory.BlockGap

/-! Orthogonal block supports keep the smoothness of a local average small. -/

open Finset
open scoped BigOperators RealInnerProductSpace

noncomputable section

namespace ULB63

/-- The local average used in the alternative of PROFILE.md §3. -/
def Instance.localAverage {n m d : ℕ} (I : Instance n m d) (i : Fin n) (x : E d) : ℝ :=
  (1 / m : ℝ) * ∑ ξ : Fin m, I.F i ξ x

namespace BlockGeometry

variable {n m p : ℕ}

private def clip (b : Blocks n m) (x : E (Dim n m p)) : E (Dim n m p) :=
  WithLp.toLp 2 (fun k => if ((Fintype.equivFin (Blocks n m × Fin (p+1))).symm k).1 = b
    then x k else 0)

@[simp] private theorem coord_clip (b b' : Blocks n m) (x : E (Dim n m p)) (j : Fin (p+1)) :
    coord (clip b x) b' j = if b' = b then coord x b' j else 0 := by
  simp [clip, coord, idx]

private theorem norm_sq_clip (b : Blocks n m) (x : E (Dim n m p)) :
    ‖clip b x‖ ^ 2 = ∑ j : Fin (p+1), (coord x b j)^2 := by
  rw [norm_sq_blocks]
  rw [sum_eq_single b]
  · simp
  · intro b' _ hne
    simp [hne]
  · simp

private theorem hess_clip (l q : ℝ) (b : Blocks n m) (x : E (Dim n m p)) :
    hess l q b (clip b x) = hess l q b x := by
  have hD : diff b (clip b x) = diff b x := by ext j; simp
  have hP : proj b (Fin.last p) (clip b x) = proj b (Fin.last p) x := by
    change coord (clip b x) b (Fin.last p) = coord x b (Fin.last p)
    simp
  simp only [hess, add_apply, smul_apply, ContinuousLinearMap.comp_apply, hD, hP]

private theorem coord_hess_zero_other (l q : ℝ) {b b' : Blocks n m} (hne : b ≠ b')
    (x : E (Dim n m p)) (j : Fin (p+1)) :
    coord (hess l q b x) b' j = 0 := by
  have h := scaled_block_gradient_zero (k := 0) 1 l q 0 (b := b) (b' := b') (x := x)
    (fun he => (hne he).elim) (j := j) (Or.inl hne)
  rw [(gradient_scaled_block 1 l q 0 b x).gradient] at h
  simpa using h

private theorem norm_hess_clip_le {l q : ℝ} (hl : 0 ≤ l) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (b : Blocks n m) (x : E (Dim n m p)) : ‖hess l q b x‖ ≤ (5*l)*‖clip b x‖ := by
  have h := (scaled_block_smooth (S := 1) (l := l) (a := 0) zero_le_one hl hq0 hq1
    (by rw [one_mul] : 1 * (5*l) ≤ 5*l) b).2 (clip b x) 0
  rw [(gradient_scaled_block 1 l q 0 b (clip b x)).gradient,
    (gradient_scaled_block 1 l q 0 b (0 : E (Dim n m p))).gradient] at h
  simpa [hess_clip] using h

private theorem hess_orthogonal (l q : ℝ) {b b' : Blocks n m} (hne : b ≠ b') (x : E (Dim n m p)) :
    ⟪hess l q b x, hess l q b' x⟫ = 0 := by
  rw [PiLp.inner_apply, ← (Fintype.equivFin (Blocks n m × Fin (p+1))).sum_comp]
  apply sum_eq_zero
  rintro ⟨v, j⟩ _
  change ⟪coord (hess l q b x) v j, coord (hess l q b' x) v j⟫ = 0
  by_cases h : b = v
  · have hz := coord_hess_zero_other l q (fun e => hne (h.trans e.symm)) x j
    simp [hz]
  · simp [coord_hess_zero_other l q h x j]

theorem localAverage_blockInstance {l μ q a : ℝ} (hm : m ≠ 0) (i : Fin n)
    (x : E (Dim n m p)) :
    (blockInstance l μ q a n m p).localAverage i x = ∑ ξ : Fin m, (n : ℝ) * block l q a (i,ξ) x := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  simp only [Instance.localAverage, blockInstance, ← mul_sum]
  field_simp

theorem blockInstance_local_smooth {l μ q a : ℝ} (hm : m ≠ 0)
    (hl : 0 ≤ l) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (i : Fin n) :
    LSmooth (5 * (n : ℝ) * l) ((blockInstance l μ q a n m p).localAverage i) := by
  have hgrad (x : E (Dim n m p)) : HasGradientAt
      ((blockInstance l μ q a n m p).localAverage i)
      (∑ ξ : Fin m, (n : ℝ) • (hess l q (i,ξ) x -
        ((l * (1 - q) / q) * a) • (proj (i,ξ) 0).adjoint 1)) x := by
    rw [show (blockInstance l μ q a n m p).localAverage i = _ from
      funext (localAverage_blockInstance hm i), hasGradientAt_iff_hasFDerivAt, map_sum]
    exact HasFDerivAt.fun_sum (u := univ) fun ξ _ =>
      (gradient_scaled_block (n : ℝ) l q a (i,ξ) x).hasFDerivAt
  refine ⟨fun x => (hgrad x).differentiableAt, ?_⟩
  intro x y
  rw [(hgrad x).gradient, (hgrad y).gradient, ← sum_sub_distrib]
  have hsum :
      (∑ ξ : Fin m, ((n : ℝ) • (hess l q (i,ξ) x - ((l * (1 - q) / q) * a) •
          (proj (i,ξ) 0).adjoint 1) -
        (n : ℝ) • (hess l q (i,ξ) y - ((l * (1 - q) / q) * a) • (proj (i,ξ) 0).adjoint 1))) =
        (n : ℝ) • ∑ ξ : Fin m, hess l q (i,ξ) (x-y) := by
    rw [smul_sum]
    exact sum_congr rfl fun ξ _ => by rw [← smul_sub, sub_sub_sub_cancel_right, map_sub]
  rw [hsum, norm_smul, Real.norm_of_nonneg (by positivity)]
  have hbound (v : E (Dim n m p)) : ‖∑ ξ : Fin m, hess l q (i,ξ) v‖ ≤ (5*l)*‖v‖ := by
    have hpythag : ‖∑ ξ : Fin m, hess l q (i,ξ) v‖ ^ 2 =
        ∑ ξ : Fin m, ‖hess l q (i,ξ) v‖ ^ 2 := by
      rw [← real_inner_self_eq_norm_sq, sum_inner]
      simp_rw [inner_sum]
      apply sum_congr rfl
      intro ξ _
      rw [sum_eq_single ξ]
      · exact real_inner_self_eq_norm_sq _
      · intro ξ' _ hne
        exact hess_orthogonal l q (fun he => hne (congrArg Prod.snd he).symm) v
      · simp
    have hclips : (∑ ξ : Fin m, ‖clip (i,ξ) v‖ ^ 2) ≤ ‖v‖ ^ 2 := by
      simp_rw [norm_sq_clip]
      rw [norm_sq_blocks, Fintype.sum_prod_type]
      exact single_le_sum (f := fun i' : Fin n => ∑ ξ : Fin m, ∑ j : Fin (p+1),
        coord v (i',ξ) j ^ 2) (fun _ _ => sum_nonneg fun _ _ => sum_nonneg fun _ _ => sq_nonneg _)
        (mem_univ i)
    apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
    rw [hpythag, mul_pow]
    calc
      _ ≤ ∑ ξ : Fin m, ((5*l)*‖clip (i,ξ) v‖)^2 :=
        sum_le_sum fun ξ _ => (sq_le_sq₀ (norm_nonneg _) (by positivity)).2
          (norm_hess_clip_le hl hq0 hq1 (i,ξ) v)
      _ = (5*l)^2 * ∑ ξ : Fin m, ‖clip (i,ξ) v‖ ^ 2 := by simp only [mul_pow, mul_sum]
      _ ≤ _ := mul_le_mul_of_nonneg_left hclips (sq_nonneg _)
  calc
    _ ≤ (n : ℝ) * ((5*l)*‖x-y‖) :=
      mul_le_mul_of_nonneg_left (hbound (x-y)) (by positivity)
    _ = _ := by ring

theorem LSmooth.weaken {d : ℕ} {F : E d → ℝ} {L₁ L₂ : ℝ}
    (hF : LSmooth L₁ F) (hL : L₁ ≤ L₂) : LSmooth L₂ F :=
  ⟨hF.1, fun x y => (hF.2 x y).trans (mul_le_mul_of_nonneg_right hL (norm_nonneg _))⟩

end BlockGeometry

end ULB63
