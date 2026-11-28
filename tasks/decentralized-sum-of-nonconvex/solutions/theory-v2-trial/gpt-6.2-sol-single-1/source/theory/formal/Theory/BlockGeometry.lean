import Theory.Sanity

/-! Disjoint quadratic paths, with a free left end and a tuned right end. -/

open scoped BigOperators RealInnerProductSpace
open Finset

noncomputable section

namespace ULB63.BlockGeometry

abbrev Blocks (n m : ℕ) := Fin n × Fin m
abbrev Dim (n m p : ℕ) := Fintype.card (Blocks n m × Fin (p + 1))

variable {n m p : ℕ}

def idx (b : Blocks n m) (j : Fin (p + 1)) : Fin (Dim n m p) :=
  Fintype.equivFin _ (b, j)

def coord (x : E (Dim n m p)) (b : Blocks n m) (j : Fin (p + 1)) : ℝ :=
  x (idx b j)

theorem idx_inj {b b' : Blocks n m} {j j' : Fin (p + 1)} :
    idx b j = idx b' j' ↔ b = b' ∧ j = j' := by
  simp [idx]

@[simp] theorem coord_zero (b : Blocks n m) (j : Fin (p + 1)) :
    coord (0 : E (Dim n m p)) b j = 0 := rfl

@[simp] theorem coord_smul (r : ℝ) (x : E (Dim n m p)) (b : Blocks n m) (j : Fin (p + 1)) :
    coord (r • x) b j = r * coord x b j := rfl

@[simp] theorem coord_add (x y : E (Dim n m p)) (b : Blocks n m) (j : Fin (p + 1)) :
    coord (x + y) b j = coord x b j + coord y b j := rfl

@[simp] theorem coord_sum {α : Type*} (s : Finset α) (x : α → E (Dim n m p))
    (b : Blocks n m) (j : Fin (p + 1)) :
    coord (∑ i ∈ s, x i) b j = ∑ i ∈ s, coord (x i) b j := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert i s hi ih => simp [hi, ih]

theorem norm_sq_blocks (x : E (Dim n m p)) :
    ‖x‖ ^ 2 = ∑ b : Blocks n m, ∑ j : Fin (p + 1), (coord x b j) ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq, ← (Fintype.equivFin (Blocks n m × Fin (p + 1))).sum_comp,
    Fintype.sum_prod_type]
  rfl

/-- The edge differences of a block. -/
def diff (b : Blocks n m) : E (Dim n m p) →L[ℝ] E p :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 (fun j : Fin p => coord x b j.succ - coord x b j.castSucc)
      map_add' := by intros; ext; simp; ring
      map_smul' := by intros; ext; simp; ring }

@[simp] theorem diff_apply (b : Blocks n m) (x : E (Dim n m p)) (j : Fin p) :
    diff b x j = coord x b j.succ - coord x b j.castSucc := rfl

theorem norm_diff_le (b : Blocks n m) (x : E (Dim n m p)) : ‖diff b x‖ ≤ 2 * ‖x‖ := by
  have hb : ∑ j : Fin (p + 1), (coord x b j) ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [norm_sq_blocks]
    exact Finset.single_le_sum (f := fun b' => ∑ j, (coord x b' j) ^ 2)
      (fun _ _ => sum_nonneg fun _ _ => sq_nonneg _) (mem_univ b)
  have hh : ∑ j : Fin p, (coord x b j.succ) ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [Fin.sum_univ_succ] at hb
    linarith [sq_nonneg (coord x b 0)]
  have ht : ∑ j : Fin p, (coord x b j.castSucc) ^ 2 ≤ ‖x‖ ^ 2 := by
    rw [Fin.sum_univ_castSucc] at hb
    linarith [sq_nonneg (coord x b (Fin.last p))]
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  calc
    ‖diff b x‖ ^ 2 = ∑ j : Fin p, (coord x b j.succ - coord x b j.castSucc) ^ 2 := by
      simp [EuclideanSpace.real_norm_sq_eq]
    _ ≤ ∑ j : Fin p, (2 * (coord x b j.succ) ^ 2 + 2 * (coord x b j.castSucc) ^ 2) :=
      sum_le_sum fun j _ => by nlinarith [sq_nonneg (coord x b j.succ + coord x b j.castSucc)]
    _ ≤ (2 * ‖x‖) ^ 2 := by
      rw [sum_add_distrib, ← mul_sum, ← mul_sum]
      nlinarith

theorem opNorm_diff_le (b : Blocks n m) : ‖(diff b : E (Dim n m p) →L[ℝ] E p)‖ ≤ 2 :=
  ContinuousLinearMap.opNorm_le_bound _ (by norm_num) (norm_diff_le b)

def proj (b : Blocks n m) (j : Fin (p + 1)) : E (Dim n m p) →L[ℝ] ℝ :=
  EuclideanSpace.proj (idx b j)

theorem norm_proj_le (b : Blocks n m) (j : Fin (p + 1)) (x : E (Dim n m p)) :
    ‖proj b j x‖ ≤ ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.real_norm_sq_eq]
  simpa [proj, Real.norm_eq_abs, sq_abs] using
    (Finset.single_le_sum (f := fun k : Fin (Dim n m p) => (x k) ^ 2)
      (fun _ _ => sq_nonneg _) (mem_univ (idx b j)))

theorem opNorm_proj_le (b : Blocks n m) (j : Fin (p + 1)) : ‖proj b j‖ ≤ 1 :=
  ContinuousLinearMap.opNorm_le_bound _ zero_le_one fun x => by simpa using norm_proj_le b j x

/-- The component divided by the total number of components. -/
def block (l q a : ℝ) (b : Blocks n m) (x : E (Dim n m p)) : ℝ :=
  l / 2 * ‖diff b x‖ ^ 2 + l * (1 - q) / 2 * (coord x b (Fin.last p)) ^ 2
    - (l * (1 - q) / q) * a * coord x b 0

theorem convex_block {l q a : ℝ} (hl : 0 ≤ l) (hq : q ≤ 1) (b : Blocks n m) :
    ConvexOn ℝ Set.univ (block (p := p) l q a b) := by
  have h : ConvexOn ℝ Set.univ (fun x : E p => ‖x‖ ^ 2) := by
    refine ⟨convex_univ, ?_⟩
    intro x _ y _ t u ht hu htu
    have hnorm := (convexOn_norm (convex_univ : Convex ℝ (Set.univ : Set (E p)))).2
      (Set.mem_univ x) (Set.mem_univ y) ht hu htu
    simp only [smul_eq_mul] at hnorm ⊢
    refine ((sq_le_sq₀ (norm_nonneg _) (by positivity)).2 hnorm).trans ?_
    simpa only [smul_eq_mul] using (convexOn_pow (𝕜 := ℝ) 2).2
      (Set.mem_Ici.mpr (norm_nonneg x)) (Set.mem_Ici.mpr (norm_nonneg y)) ht hu htu
  have hD := h.comp_linearMap ((diff b : E (Dim n m p) →L[ℝ] E p).toLinearMap)
  have hP' : ConvexOn ℝ Set.univ (fun x : E (Dim n m p) => (coord x b (Fin.last p)) ^ 2) := by
    refine ⟨convex_univ, ?_⟩
    intro x _ y _ t u ht hu htu
    simp only [coord_add, coord_smul, smul_eq_mul]
    simpa only [smul_eq_mul] using (Even.convexOn_pow (𝕜 := ℝ) (by decide : Even (2 : ℕ))).2
      (Set.mem_univ (coord x b (Fin.last p))) (Set.mem_univ (coord y b (Fin.last p))) ht hu htu
  have hlin : ConcaveOn ℝ Set.univ (fun x : E (Dim n m p) =>
      (l * (1 - q) / q) * a * coord x b 0) := by
    exact (((l * (1 - q) / q) * a) • (proj (p := p) b 0)).toLinearMap.concaveOn convex_univ
  have hD' : ConvexOn ℝ Set.univ (fun x : E (Dim n m p) => ‖diff b x‖ ^ 2) := by
    simpa [Function.comp_def] using hD
  exact ((ConvexOn.smul (by positivity : 0 ≤ l / 2) hD').add
    (ConvexOn.smul (by positivity [sub_nonneg.mpr hq] : 0 ≤ l * (1 - q) / 2) hP')).sub hlin

theorem hasGradientAt_half_norm_comp {d : ℕ} {V : Type*} [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] [CompleteSpace V] (D : E d →L[ℝ] V) (x : E d) :
    HasGradientAt (fun y => ‖D y‖ ^ 2 / 2) (D.adjoint (D x)) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have hd := (D.hasFDerivAt (x := x)).norm_sq.const_mul (2 : ℝ)⁻¹
  convert hd using 1
  · funext y
    ring
  · ext y
    simp [InnerProductSpace.toDual_apply_apply, ContinuousLinearMap.adjoint_inner_left]

theorem hasGradientAt_half_proj {d : ℕ} (D : E d →L[ℝ] ℝ) (x : E d) :
    HasGradientAt (fun y => (D y) ^ 2 / 2) (D.adjoint (D x)) x := by
  simpa [Real.norm_eq_abs, sq_abs] using hasGradientAt_half_norm_comp D x

theorem hasGradientAt_proj {d : ℕ} (D : E d →L[ℝ] ℝ) (x : E d) :
    HasGradientAt D (D.adjoint 1) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  convert D.hasFDerivAt using 1
  ext y
  simp [InnerProductSpace.toDual_apply_apply, ContinuousLinearMap.adjoint_inner_left]

def hess (l q : ℝ) (b : Blocks n m) : E (Dim n m p) →L[ℝ] E (Dim n m p) :=
  l • ((diff b).adjoint.comp (diff b)) +
    (l * (1 - q)) • ((proj b (Fin.last p)).adjoint.comp (proj b (Fin.last p)))

theorem gradient_scaled_block (S l q a : ℝ) (b : Blocks n m) (x : E (Dim n m p)) :
    HasGradientAt (fun y => S * block l q a b y)
      (S • (hess l q b x - ((l * (1 - q) / q) * a) • (proj b 0).adjoint 1)) x := by
  rw [hasGradientAt_iff_hasFDerivAt, map_smul, map_sub]
  have h₁ := (hasGradientAt_half_norm_comp (diff b) x).hasFDerivAt.const_mul l
  have h₂ := (hasGradientAt_half_proj (proj b (Fin.last p)) x).hasFDerivAt.const_mul (l * (1 - q))
  have h₃ := (hasGradientAt_proj (proj b 0) x).hasFDerivAt.const_mul ((l * (1 - q) / q) * a)
  convert ((h₁.add h₂).sub h₃).const_mul S using 1
  · funext y
    simp only [block, Pi.sub_apply, Pi.add_apply, proj,
      EuclideanSpace.coe_proj, coord]
    ring
  · simp [hess, map_add, map_smul]

theorem scaled_block_smooth {S l q a L : ℝ} (hS : 0 ≤ S) (hl : 0 ≤ l)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hL : S * (5 * l) ≤ L) (b : Blocks n m) :
    LSmooth L (fun x : E (Dim n m p) => S * block l q a b x) := by
  refine ⟨fun x => (gradient_scaled_block S l q a b x).differentiableAt, ?_⟩
  intro x y
  rw [(gradient_scaled_block S l q a b x).gradient,
    (gradient_scaled_block S l q a b y).gradient, ← smul_sub, sub_sub_sub_cancel_right,
    ← map_sub, norm_smul, Real.norm_of_nonneg hS]
  have hD : ‖(diff b : E (Dim n m p) →L[ℝ] E p)‖ ≤ 2 := opNorm_diff_le b
  have hP : ‖proj (p := p) b (Fin.last p)‖ ≤ 1 := opNorm_proj_le b _
  have hH : ‖hess (p := p) l q b‖ ≤ 5 * l := by
    unfold hess
    calc
      _ ≤ ‖l • ((diff b : E (Dim n m p) →L[ℝ] E p).adjoint.comp (diff b))‖ +
          ‖(l * (1 - q)) • ((proj b (Fin.last p)).adjoint.comp (proj b (Fin.last p)))‖ :=
        norm_add_le _ _
      _ ≤ l * (2 * 2) + (l * (1 - q)) * (1 * 1) := by
        rw [norm_smul, norm_smul, Real.norm_of_nonneg hl,
          Real.norm_of_nonneg (mul_nonneg hl (sub_nonneg.mpr hq1))]
        apply add_le_add
        · apply mul_le_mul_of_nonneg_left _ hl
          refine (ContinuousLinearMap.opNorm_comp_le _ _).trans ?_
          simpa using mul_le_mul hD hD (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 2)
        · apply mul_le_mul_of_nonneg_left _ (by positivity [sub_nonneg.mpr hq1])
          refine (ContinuousLinearMap.opNorm_comp_le _ _).trans ?_
          simpa using mul_le_mul hP hP (norm_nonneg _) zero_le_one
      _ ≤ _ := by nlinarith
  calc
    _ ≤ S * (‖hess (p := p) l q b‖ * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left (ContinuousLinearMap.le_opNorm _ _) hS
    _ ≤ L * ‖x - y‖ := by
      rw [← mul_assoc]
      exact mul_le_mul_of_nonneg_right ((mul_le_mul_of_nonneg_left hH hS).trans hL) (norm_nonneg _)

theorem coord_adjoint {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    [CompleteSpace V] (D : E (Dim n m p) →L[ℝ] V) (y : V) (b : Blocks n m)
    (j : Fin (p + 1)) :
    coord (D.adjoint y) b j = ⟪y, D (EuclideanSpace.single (idx b j) 1)⟫ := by
  rw [← D.adjoint_inner_left]
  simp [coord, EuclideanSpace.inner_single_right]

theorem scaled_block_gradient_zero (S l q a : ℝ) {b b' : Blocks n m}
    {x : E (Dim n m p)} {k : ℕ}
    (hx : b = b' → ∀ j : Fin (p + 1), k ≤ j.val → coord x b j = 0)
    {j : Fin (p + 1)} (hj : b ≠ b' ∨ k < j.val) :
    coord (gradient (fun y => S * block l q a b y) x) b' j = 0 := by
  have hhead : proj b 0 (EuclideanSpace.single (idx b' j) 1) = 0 := by
    change (EuclideanSpace.single (idx b' j) (1 : ℝ)) (idx b (0 : Fin (p + 1))) = 0
    apply PiLp.single_eq_of_ne' 2 _ _
    intro he
    have he' := idx_inj.mp he
    rcases hj with h | h
    · exact h he'.1.symm
    · rw [he'.2, Fin.val_zero] at h
      omega
  have hlast : (proj b (Fin.last p) x) *
      (proj b (Fin.last p) (EuclideanSpace.single (idx b' j) 1)) = 0 := by
    by_cases he : idx b' j = idx b (Fin.last p)
    · obtain ⟨hbb, hjj⟩ := idx_inj.mp he
      have hk := hj.resolve_left (not_not_intro hbb.symm)
      change coord x b (Fin.last p) * _ = 0
      rw [hx hbb.symm (Fin.last p) (by simpa [hjj] using hk.le), zero_mul]
    · have hzero : proj b (Fin.last p) (EuclideanSpace.single (idx b' j) 1) = 0 :=
        PiLp.single_eq_of_ne' 2 he _
      rw [hzero, mul_zero]
  have hd : ⟪(diff b : E (Dim n m p) →L[ℝ] E p) x,
      diff b (EuclideanSpace.single (idx b' j) 1)⟫ = 0 := by
    rw [PiLp.inner_apply]
    apply sum_eq_zero
    intro e _
    by_cases h₁ : idx b' j = idx b e.succ
    · obtain ⟨hbb, hjj⟩ := idx_inj.mp h₁
      have hjv := hj.resolve_left (not_not_intro hbb.symm)
      have hk : k ≤ e.val := by
        rw [hjj, Fin.val_succ] at hjv
        omega
      simp [hx hbb.symm e.succ (Nat.le_succ_of_le hk), hx hbb.symm e.castSucc hk]
    · by_cases h₂ : idx b' j = idx b e.castSucc
      · obtain ⟨hbb, hjj⟩ := idx_inj.mp h₂
        have hjv := hj.resolve_left (not_not_intro hbb.symm)
        have hk : k ≤ e.val := by
          rw [hjj, Fin.val_castSucc] at hjv
          omega
        simp [hx hbb.symm e.succ (Nat.le_succ_of_le hk), hx hbb.symm e.castSucc hk]
      · have hz₁ : coord (EuclideanSpace.single (idx b' j) 1) b e.succ = 0 :=
          PiLp.single_eq_of_ne' 2 h₁ _
        have hz₂ : coord (EuclideanSpace.single (idx b' j) 1) b e.castSucc = 0 :=
          PiLp.single_eq_of_ne' 2 h₂ _
        simp [hz₁, hz₂]
  rw [(gradient_scaled_block S l q a b x).gradient]
  simp only [hess, add_apply, smul_apply, ContinuousLinearMap.comp_apply, coord_smul]
  change S * (coord (l • ((diff b).adjoint ((diff b) x)) +
      (l * (1 - q)) • (proj b (Fin.last p)).adjoint (proj b (Fin.last p) x)) b' j -
    coord (((l * (1 - q) / q) * a) • (proj b 0).adjoint 1) b' j) = 0
  simp only [coord_add, coord_smul, coord_adjoint, hd,
    hhead, Real.inner_apply, hlast, mul_zero, add_zero, sub_zero]

end ULB63.BlockGeometry
