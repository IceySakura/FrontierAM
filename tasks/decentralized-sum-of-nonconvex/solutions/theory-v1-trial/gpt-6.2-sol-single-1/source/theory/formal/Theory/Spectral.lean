import Theory.Network

open Matrix Polynomial

noncomputable section

namespace ULB63
namespace Network

variable {n : ℕ} {ρ : ℝ} {W : Matrix (Fin n) (Fin n) ℝ}

private def polynomialStep (A : Matrix (Fin n) (Fin n) ℝ) : ℕ → Matrix (Fin n) (Fin n) ℝ
  | 0 => 1
  | 1 => A
  | k+2 => (2 : ℝ) • (A * polynomialStep A (k+1)) - polynomialStep A k

private def vT (k : ℕ) (x : ℝ) : ℝ := (Chebyshev.T ℝ (k : ℤ)).eval x

@[simp] private theorem vT_zero (x : ℝ) : vT 0 x = 1 := by simp [vT]
@[simp] private theorem vT_one (x : ℝ) : vT 1 x = x := by simp [vT]

private theorem vT_succ (k : ℕ) (x : ℝ) :
    vT (k+2) x = 2*x*vT (k+1) x - vT k x := by
  simp only [vT, Nat.cast_add, Nat.cast_ofNat, Nat.cast_one, Chebyshev.T_add_two,
    Polynomial.eval_sub, Polynomial.eval_mul, Polynomial.eval_ofNat, Polynomial.eval_X]

private theorem polynomialStep_bound {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian)
    (hB : ∀ v, norm2 (A.mulVec v) ≤ norm2 v) (D : ℕ) (v : Fin n → ℝ) :
    norm2 ((polynomialStep A D).mulVec v) ≤ norm2 v := by
  let T := A.toEuclideanLin
  have hT : T.IsSymmetric := Matrix.isSymmetric_toEuclideanLin_iff.2 hA
  have hdim : Module.finrank ℝ (E n) = n := by simp
  let b := hT.eigenvectorBasis hdim
  let r := b.repr
  let lam := hT.eigenvalues hdim
  have hdiag (x : E n) (i : Fin n) : r (T x) i = lam i * r x i := by
    simpa only [r, b, lam, RCLike.ofReal_real_eq_id, id_eq] using
      hT.eigenvectorBasis_apply_self_apply hdim x i
  have hLam (i : Fin n) : |lam i| ≤ 1 := by
    have h := hB (b i).ofLp
    rw [norm2_eq, norm2_eq] at h
    change ‖T (b i)‖ ≤ ‖b i‖ at h
    have he : T (b i) = lam i • b i := hT.apply_eigenvectorBasis hdim i
    rwa [he, norm_smul, Real.norm_eq_abs, b.norm_eq_one, mul_one] at h
  have hstep (k : ℕ) (x : E n) :
      (polynomialStep A (k+2)).toEuclideanLin x =
      (2 : ℝ) • T ((polynomialStep A (k+1)).toEuclideanLin x) -
        (polynomialStep A k).toEuclideanLin x := by
    simp only [polynomialStep, map_sub, map_smul, Matrix.toLpLin_mul_same,
      LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.comp_apply, T]
  have hrep (k : ℕ) (x : E n) (i : Fin n) :
      r ((polynomialStep A k).toEuclideanLin x) i = vT k (lam i) * r x i := by
    induction k using Nat.twoStepInduction with
    | zero => simp [polynomialStep]
    | one =>
      change r (T x) i = vT 1 (lam i) * r x i
      rw [vT_one, hdiag]
    | more k ih₀ ih₁ =>
      rw [hstep, map_sub, map_smul, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
      rw [hdiag]
      rw [ih₁, ih₀, vT_succ]
      ring
  rw [norm2_eq, norm2_eq]
  change ‖(polynomialStep A D).toEuclideanLin (WithLp.toLp 2 v)‖ ≤
    ‖(WithLp.toLp 2 v : E n)‖
  rw [← r.norm_map ((polynomialStep A D).toEuclideanLin (WithLp.toLp 2 v)),
    ← r.norm_map (WithLp.toLp 2 v)]
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  apply Finset.sum_le_sum
  intro i _
  rw [hrep, mul_pow]
  have h := Chebyshev.abs_eval_T_real_le_one (D : ℤ) (hLam i)
  change |vT D (lam i)| ≤ 1 at h
  have hb := (abs_le.mp h)
  have hp : (vT D (lam i))^2 ≤ 1 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hb.2) (by linarith : 0 ≤ 1+vT D (lam i))]
  exact mul_le_of_le_one_left (sq_nonneg _) hp

private theorem proj_step {B P : Matrix (Fin n) (Fin n) ℝ} (h : P*B = 0) (k : ℕ) :
    P * polynomialStep B k = vT k 0 • P := by
  induction k using Nat.twoStepInduction with
  | zero => simp [polynomialStep]
  | one => simpa [polynomialStep] using h
  | more k ih₀ ih₁ =>
    rw [polynomialStep, mul_sub, Matrix.mul_smul, ← mul_assoc, h, zero_mul, smul_zero, ih₀, vT_succ]
    simp [← neg_smul]

private theorem polynomialStep_split {B P : Matrix (Fin n) (Fin n) ℝ} (s : ℝ)
    (hl : P*B = 0) (hr : B*P = 0) (hP : P*P = P) (k : ℕ) :
    polynomialStep (B+s • P) k =
      polynomialStep B k + (vT k s - vT k 0) • P := by
  induction k using Nat.twoStepInduction with
  | zero => simp [polynomialStep]
  | one => simp [polynomialStep]
  | more k ih₀ ih₁ =>
    have hmul : (B+s • P) * (polynomialStep B (k+1) +
        (vT (k+1) s - vT (k+1) 0) • P) =
      B * polynomialStep B (k+1) + (s*vT (k+1) s) • P := by
      simp only [mul_add, add_mul, Matrix.smul_mul, Matrix.mul_smul, hr, hP,
        proj_step hl, smul_smul, smul_zero, zero_add]
      module
    rw [polynomialStep, ih₁, ih₀, hmul, polynomialStep, vT_succ, vT_succ]
    module

private theorem polynomialStep_zero (hW : CommClass n ρ W) (z : ℝ) (k : ℕ) (i j : Fin n)
    (h : (lazy W ^ k) i j = 0) : polynomialStep (z • W) k i j = 0 := by
  induction k using Nat.twoStepInduction generalizing i j with
  | zero => simpa only [pow_zero, polynomialStep] using h
  | one =>
    rw [pow_one, lazy_apply] at h
    have hw : W i j = 0 := by split at h <;> linarith [hW.nonneg i j]
    simp [polynomialStep, hw]
  | more k ih₀ ih₁ =>
    have he : ∑ u, lazy W i u * (lazy W ^ (k+1)) u j = 0 := by
      simpa only [pow_succ', Matrix.mul_apply] using h
    have he' := (Finset.sum_eq_zero_iff_of_nonneg
      (fun u (_ : u ∈ Finset.univ) => mul_nonneg (lazy_nonneg hW i u)
        (pow_nonneg (lazy_nonneg hW) (k+1) u j))).1 he
    have hs : ((z • W) * polynomialStep (z • W) (k+1)) i j = 0 := by
      rw [Matrix.mul_apply]
      apply Finset.sum_eq_zero
      intro u hu
      rcases mul_eq_zero.mp (he' u hu) with hA | hB
      · rw [lazy_apply] at hA
        have hw : W i u = 0 := by split at hA <;> linarith [hW.nonneg i u]
        simp [hw]
      · simp [ih₁ _ _ hB]
    simp only [polynomialStep, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, hs,
      ih₀ i j (zero_of_later_zero hW i j (by omega) h), mul_zero, sub_zero]

theorem positive_of_chebyshev (hn : 0 < n) (hρ : 0 < ρ) (hW : CommClass n ρ W)
    {D : ℕ} (hD : 2*(n:ℝ) < (Polynomial.Chebyshev.T ℝ (D : ℤ)).eval (1/ρ))
    (i j : Fin n) : 0 < (lazy W ^ D) i j := by
  let S : Matrix (Fin n) (Fin n) ℝ := ρ⁻¹ • (W-J n)
  have hS : S.IsHermitian := by
    change (ρ⁻¹ • (W-J n)).conjTranspose = ρ⁻¹ • (W-J n)
    rw [Matrix.conjTranspose_eq_transpose_of_trivial]
    simp [hW.symm, show (J n).transpose = J n from rfl]
  have hB (v : Fin n → ℝ) : norm2 (S.mulVec v) ≤ norm2 v := by
    dsimp only [S]
    rw [Matrix.smul_mulVec, norm2_smul, abs_of_nonneg (inv_nonneg.mpr hρ.le)]
    calc
      _ ≤ ρ⁻¹ * (ρ * norm2 v) :=
        mul_le_mul_of_nonneg_left (hW.contraction v) (by positivity)
      _ = _ := by rw [← mul_assoc, inv_mul_cancel₀ hρ.ne', one_mul]
  have hl : J n*S = 0 := by
    dsimp only [S]
    rw [Matrix.mul_smul, mul_sub, JW hW, JJ hn, sub_self, smul_zero]
  have hr : S*J n = 0 := by
    dsimp only [S]
    rw [Matrix.smul_mul, sub_mul, WJ hW, JJ hn, sub_self, smul_zero]
  have hA : S + ρ⁻¹ • J n = ρ⁻¹ • W := by
    dsimp only [S]
    rw [← smul_add, sub_add_cancel]
  have hsplit := polynomialStep_split (ρ⁻¹) hl hr (JJ hn) D
  rw [hA] at hsplit
  by_contra! h
  have h0 := le_antisymm h (pow_nonneg (lazy_nonneg hW) D i j)
  have hf := polynomialStep_zero hW ρ⁻¹ D i j h0
  rw [hsplit, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul] at hf
  have hco := (abs_coord_le (fun u => polynomialStep S D u j) i).trans (show
      norm2 (fun u => polynomialStep S D u j) ≤ 1 by
    have H := polynomialStep_bound hS hB D (Pi.single j 1)
    rw [Matrix.mulVec_single_one, norm_single] at H
    exact H)
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hn₀ : (0 : ℝ) < n := by linarith
  have hleft := (abs_le.mp hco).1
  have hv := Chebyshev.abs_eval_T_real_le_one (D : ℤ) (by norm_num : |(0 : ℝ)| ≤ 1)
  have hv' : vT D 0 ≤ 1 := (abs_le.mp hv).2
  change 2*(n:ℝ) < vT D (1/ρ) at hD
  rw [one_div] at hD
  change polynomialStep S D i j + (vT D ρ⁻¹ - vT D 0)*(1/(n:ℝ)) = 0 at hf
  have hmul := congrArg (fun t : ℝ => t * (n : ℝ)) hf
  have hrw : (polynomialStep S D i j + (vT D ρ⁻¹ - vT D 0)*(1/(n:ℝ)))*(n:ℝ) =
      polynomialStep S D i j * (n:ℝ) + vT D ρ⁻¹ - vT D 0 := by
    field_simp
    ring
  rw [hrw, zero_mul] at hmul
  have hcontra := mul_le_mul_of_nonneg_right hleft hn₀.le
  linarith

private theorem vT_growth (hρ : 0 < ρ) (hρ₁ : ρ < 1) (D : ℕ) :
    (1+Real.sqrt (1-ρ))^D / 2 ≤ vT D (1/ρ) := by
  let s := Real.sqrt (1-ρ^2)
  let b := (1+s)/ρ
  have hs₂ : s^2 = 1-ρ^2 := by
    apply Real.sq_sqrt
    nlinarith [mul_nonneg hρ.le (sub_nonneg.mpr hρ₁.le)]
  have hs : 0 ≤ s := Real.sqrt_nonneg _
  have hb : 0 < b := by unfold b; positivity
  have hinv : b⁻¹ = (1-s)/ρ := by
    apply inv_eq_of_mul_eq_one_right
    unfold b
    field_simp
    nlinarith
  have hcos : Real.cosh (Real.log b) = 1/ρ := by
    rw [Real.cosh_log hb, hinv]
    unfold b
    ring
  have hβ : 1+Real.sqrt (1-ρ) ≤ b := by
    have hss : Real.sqrt (1-ρ) ≤ s :=
      Real.sqrt_le_sqrt (by nlinarith [mul_nonneg hρ.le (sub_nonneg.mpr hρ₁.le)])
    unfold b
    exact (add_le_add_right hss 1).trans ((le_div_iff₀ hρ).2
      (mul_le_of_le_one_right (by positivity) hρ₁.le))
  have hpw := pow_le_pow_left₀ (show 0 ≤ 1+Real.sqrt (1-ρ) by positivity) hβ D
  rw [← hcos, vT, Chebyshev.T_real_cosh, Int.cast_natCast, Real.cosh_eq, Real.exp_nat_mul,
    Real.exp_log hb]
  linarith [Real.exp_pos (-((D : ℝ)*Real.log b))]

theorem certified_radius (hn : 0 < n) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hW : CommClass n ρ W) {D : ℕ}
    (hD : n-1 ≤ D ∨ 2 * (n : ℝ) * ((1+ρ)/2)^D < 1 ∨
      (D = 1 ∧ (n : ℝ)*ρ < 1) ∨ (0 < ρ ∧ 4*(n : ℝ) < (1+Real.sqrt (1-ρ))^D)) :
    ∀ i j, 0 < (lazy W ^ D) i j := by
  rcases hD with h | h | h | ⟨hp, hh⟩
  · exact positive_of_size hn hρ hρ₁ hW h
  · exact positive_of_power hn hρ hW h
  · obtain ⟨rfl, h⟩ := h
    exact positive_of_dense hn hW h
  · apply positive_of_chebyshev hn hp hW
    change 2*(n:ℝ) < vT D (1/ρ)
    linarith [vT_growth hp hρ₁ D]

end Network
end ULB63
