import Theory.Model

open Matrix

noncomputable section

namespace ULB63
namespace Network

variable {n : ℕ} {ρ : ℝ} {W : Matrix (Fin n) (Fin n) ℝ}

def lazy (W : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  (1/2 : ℝ) • (1+W)

theorem lazy_apply (i j : Fin n) :
    lazy W i j = ((if i = j then 1 else 0) + W i j) / 2 := by
  simp [lazy, Matrix.one_apply, div_eq_mul_inv, mul_comm]

theorem lazy_symm (hW : CommClass n ρ W) : (lazy W).transpose = lazy W := by
  simp [lazy, hW.symm]

theorem lazy_nonneg (hW : CommClass n ρ W) (i j : Fin n) : 0 ≤ lazy W i j := by
  rw [lazy_apply]
  split <;> linarith [hW.nonneg i j]

theorem lazy_diag (hW : CommClass n ρ W) (i : Fin n) : 0 < lazy W i i := by
  rw [lazy_apply, ite_eq_left rfl]
  linarith [hW.nonneg i i]

theorem pow_nonneg {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : ∀ i j, 0 ≤ Q i j)
    (k : ℕ) (i j : Fin n) : 0 ≤ (Q^k) i j := by
  induction k generalizing i with
  | zero => rw [pow_zero, Matrix.one_apply]; split <;> norm_num
  | succ k ih =>
    rw [pow_succ', Matrix.mul_apply]
    exact Finset.sum_nonneg fun a _ => mul_nonneg (hQ i a) (ih a)

theorem norm2_eq (v : Fin n → ℝ) : norm2 v = ‖(WithLp.toLp 2 v : E n)‖ := by
  rw [← Real.sqrt_sq (norm_nonneg (WithLp.toLp 2 v : E n)), EuclideanSpace.real_norm_sq_eq]
  simp [norm2]

private theorem norm2_add (v u : Fin n → ℝ) : norm2 (v+u) ≤ norm2 v + norm2 u := by
  simpa only [norm2_eq, WithLp.toLp_add] using
    norm_add_le (WithLp.toLp 2 v : E n) (WithLp.toLp 2 u : E n)

theorem norm2_sub (v u : Fin n → ℝ) : norm2 (v-u) ≤ norm2 v + norm2 u := by
  simpa only [norm2_eq, WithLp.toLp_sub] using
    norm_sub_le (WithLp.toLp 2 v : E n) (WithLp.toLp 2 u : E n)

theorem norm2_smul (a : ℝ) (v : Fin n → ℝ) : norm2 (a • v) = |a| * norm2 v := by
  simp only [norm2_eq, WithLp.toLp_smul, norm_smul, Real.norm_eq_abs]

theorem abs_coord_le (v : Fin n → ℝ) (i : Fin n) : |v i| ≤ norm2 v := by
  rw [norm2, ← Real.sqrt_sq_eq_abs (v i)]
  exact Real.sqrt_le_sqrt (Finset.single_le_sum (fun j _ => sq_nonneg (v j)) (Finset.mem_univ i))

theorem WJ (hW : CommClass n ρ W) : W * J n = J n := by
  ext i j
  change ∑ k, W i k * (1/n : ℝ) = 1/n
  rw [← Finset.sum_mul, hW.row_sum, one_mul]

theorem JW (hW : CommClass n ρ W) : J n * W = J n := by
  have H := congrArg Matrix.transpose (WJ hW)
  have Hj : (J n).transpose = J n := rfl
  simpa only [Matrix.transpose_mul, hW.symm, Hj] using H

theorem JJ (hn : 0 < n) : J n * J n = J n := by
  have hnr : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  ext i j
  simp [Matrix.mul_apply, J, Finset.sum_const]
  field_simp

private theorem lazyJ (hW : CommClass n ρ W) : lazy W * J n = J n := by
  rw [lazy, Matrix.smul_mul, add_mul, one_mul, WJ hW, ← two_smul ℝ, smul_smul]
  norm_num

private theorem Jlazy (hW : CommClass n ρ W) : J n * lazy W = J n := by
  rw [lazy, Matrix.mul_smul, mul_add, mul_one, JW hW, ← two_smul ℝ, smul_smul]
  norm_num

private theorem Jlazy_pow (hW : CommClass n ρ W) (k : ℕ) : J n * lazy W ^ k = J n := by
  induction k with
  | zero => simp
  | succ k ih => rw [pow_succ, ← mul_assoc, ih, Jlazy hW]

private theorem iterate_sub (hn : 0 < n) (hW : CommClass n ρ W) (k : ℕ) :
    lazy W ^ (k+1) - J n = (lazy W - J n) * (lazy W ^ k - J n) := by
  rw [sub_mul, mul_sub, mul_sub, lazyJ hW, Jlazy_pow hW, JJ hn, sub_self, sub_zero, pow_succ']

private theorem lazy_contract (hW : CommClass n ρ W) {v : Fin n → ℝ}
    (hv : (J n).mulVec v = 0) :
    norm2 ((lazy W - J n).mulVec v) ≤ ((1+ρ)/2) * norm2 v := by
  have hf : W.mulVec v = (W - J n).mulVec v := by rw [Matrix.sub_mulVec, hv, sub_zero]
  rw [Matrix.sub_mulVec, hv, sub_zero, lazy, Matrix.smul_mulVec, Matrix.add_mulVec, Matrix.one_mulVec,
    norm2_smul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 1/2), hf]
  have hb := norm2_add v ((W-J n).mulVec v)
  linarith [hW.contraction v]

theorem lazy_pow_close (hn : 0 < n) (hρ : 0 ≤ ρ) (hW : CommClass n ρ W)
    (k : ℕ) (v : Fin n → ℝ) :
    norm2 ((lazy W ^ k - J n).mulVec v) ≤ ((1+ρ)/2) ^ k * norm2 ((1 - J n).mulVec v) := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [iterate_sub hn hW, ← Matrix.mulVec_mulVec]
    have hv : (J n).mulVec ((lazy W ^ k - J n).mulVec v) = 0 := by
      rw [Matrix.mulVec_mulVec, mul_sub, Jlazy_pow hW, JJ hn, sub_self, Matrix.zero_mulVec]
    rw [pow_succ']
    exact (lazy_contract hW hv).trans (by
      rw [mul_assoc]
      exact mul_le_mul_of_nonneg_left ih (by linarith))

private theorem J_single (j : Fin n) :
    (J n).mulVec (Pi.single j 1) = fun _ => (n : ℝ)⁻¹ := by
  rw [Matrix.mulVec_single_one]
  funext i
  change (1 / n : ℝ) = (n : ℝ)⁻¹
  exact one_div _

theorem norm_single (j : Fin n) : norm2 (Pi.single j (1 : ℝ)) = 1 := by
  simp [norm2, Pi.single_apply, apply_ite (fun a : ℝ => a^2)]

theorem norm_J_single (hn : 0 < n) (j : Fin n) :
    norm2 ((J n).mulVec (Pi.single j 1)) ≤ 1 := by
  have hnr : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hp : (0 : ℝ) < n := by linarith
  rw [J_single, norm2, ← Real.sqrt_one]
  apply Real.sqrt_le_sqrt
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have he : (n : ℝ) * ((n : ℝ)⁻¹)^2 = (n : ℝ)⁻¹ := by field_simp
  rw [he]
  exact inv_le_one_of_one_le₀ hnr

theorem positive_of_power (hn : 0 < n) (hρ : 0 ≤ ρ) (hW : CommClass n ρ W)
    {k : ℕ} (hk : 2 * (n : ℝ) * ((1+ρ)/2)^k < 1) (i j : Fin n) :
    0 < (lazy W ^ k) i j := by
  have H := lazy_pow_close hn hρ hW k (Pi.single j 1)
  rw [Matrix.mulVec_single_one] at H
  have hB : norm2 ((1-J n).mulVec (Pi.single j 1)) ≤ 2 := by
    rw [Matrix.sub_mulVec, Matrix.one_mulVec]
    linarith [norm2_sub (Pi.single j 1) ((J n).mulVec (Pi.single j 1)),
      norm_single j, norm_J_single hn j]
  have hpow : 0 ≤ ((1+ρ)/2)^k := _root_.pow_nonneg (by linarith) _
  have hco := (abs_coord_le ((lazy W ^ k - J n).col j) i).trans
    (H.trans (mul_le_mul_of_nonneg_left hB hpow))
  change |(lazy W^k) i j - (1/n : ℝ)| ≤ _ at hco
  have hnr : (0 : ℝ) < n := by exact_mod_cast hn
  have hsmall : ((1+ρ)/2)^k * 2 < 1 / (n : ℝ) := by
    apply (lt_div_iff₀ hnr).2
    convert hk using 1
    ring
  have hleft := (abs_le.mp hco).1
  linarith

private def reached (W : Matrix (Fin n) (Fin n) ℝ) (j : Fin n) (k : ℕ) : Finset (Fin n) :=
  Finset.univ.filter (fun i => 0 < (lazy W ^ k) i j)

private theorem reached_succ (hW : CommClass n ρ W) {j i : Fin n} {k : ℕ} :
    i ∈ reached W j (k+1) ↔ ∃ u ∈ reached W j k, 0 < lazy W i u := by
  rw [reached, Finset.mem_filter, and_iff_right (Finset.mem_univ _), pow_succ', Matrix.mul_apply,
    Finset.sum_pos_iff_of_nonneg (fun u _ => mul_nonneg (lazy_nonneg hW i u)
      (pow_nonneg (lazy_nonneg hW) k u j))]
  simp only [Finset.mem_univ, true_and, reached, Finset.mem_filter]
  constructor
  · rintro ⟨u, hu⟩
    exact ⟨u, lt_of_le_of_ne (pow_nonneg (lazy_nonneg hW) k u j)
        (fun hz => hu.ne' (by rw [← hz, mul_zero])),
      lt_of_le_of_ne (lazy_nonneg hW i u) (fun hz => hu.ne' (by rw [← hz, zero_mul]))⟩
  · rintro ⟨u, hu, hp⟩
    exact ⟨u, mul_pos hp hu⟩

private theorem reached_mono (hW : CommClass n ρ W) (j : Fin n) (k : ℕ) :
    reached W j k ⊆ reached W j (k+1) :=
  fun i hi => (reached_succ hW).2 ⟨i, hi, lazy_diag hW i⟩

theorem zero_of_later_zero (hW : CommClass n ρ W) (i j : Fin n) {a b : ℕ}
    (hab : a ≤ b) (h : (lazy W ^ b) i j = 0) : (lazy W ^ a) i j = 0 := by
  have hm : reached W j a ⊆ reached W j b :=
    monotone_nat_of_le_succ (reached_mono hW j) hab
  by_contra hn
  have hp := lt_of_le_of_ne (pow_nonneg (lazy_nonneg hW) a i j) (fun hz => hn hz.symm)
  have ha : i ∈ reached W j a := by simp [reached, hp]
  simpa [reached, h] using hm ha

private theorem reached_stays (hW : CommClass n ρ W) {j : Fin n} {k : ℕ}
    (h : reached W j k = reached W j (k+1)) (q : ℕ) :
    reached W j (k+q) = reached W j k := by
  induction q with
  | zero => rfl
  | succ q ih =>
    ext i
    rw [show k + (q+1) = (k+q)+1 by omega, reached_succ hW, ih, ← reached_succ hW, ← h]

theorem positive_of_size (hn : 0 < n) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1) (hW : CommClass n ρ W)
    {k : ℕ} (hk : n-1 ≤ k) (i j : Fin n) :
    0 < (lazy W ^ k) i j := by
  have hev : ∀ᶠ q in Filter.atTop, 2 * (n : ℝ) * ((1+ρ)/2)^q < 1 := by
    have hlim : Filter.Tendsto (fun q : ℕ => 2 * (n : ℝ) * ((1+ρ)/2)^q)
        Filter.atTop (nhds 0) := by
      simpa only [mul_zero] using
        (tendsto_pow_atTop_nhds_zero_of_lt_one (show 0 ≤ (1+ρ)/2 by linarith)
          (show (1+ρ)/2 < 1 by linarith)).const_mul (2 * (n : ℝ))
    exact hlim.eventually_lt_const (by norm_num)
  obtain ⟨K, hK⟩ := Filter.eventually_atTop.mp hev
  have hgrow (q : ℕ) (hq : reached W j q ≠ Finset.univ) :
      (reached W j q).card + 1 ≤ (reached W j (q+1)).card := by
    apply Finset.card_lt_card
    refine Finset.ssubset_iff_subset_ne.mpr ⟨reached_mono hW j q, ?_⟩
    intro hs
    have he : reached W j (q+K) = Finset.univ := by
      ext u
      simp [reached, positive_of_power hn hρ hW (hK (q+K) (by omega)) u j]
    exact hq ((reached_stays hW hs K).symm.trans he)
  have hc (q : ℕ) : min (q+1) n ≤ (reached W j q).card := by
    induction q with
    | zero =>
      refine (min_le_left _ _).trans (Finset.card_pos.mpr ⟨j, ?_⟩)
      simp [reached]
    | succ q ih =>
      by_cases he : reached W j q = Finset.univ
      · have he' : reached W j (q+1) = Finset.univ :=
          Finset.eq_univ_of_forall (fun u => reached_mono hW j q (he ▸ Finset.mem_univ u))
        rw [he', Finset.card_univ, Fintype.card_fin]
        exact min_le_right _ _
      · have hq : (reached W j q).card < n := by
          simpa using (Finset.card_lt_card
            (Finset.ssubset_iff_subset_ne.mpr ⟨Finset.subset_univ _, he⟩))
        have := hgrow q he
        omega
  have hu : reached W j k = Finset.univ := by
    apply Finset.eq_univ_of_card
    rw [Fintype.card_fin]
    have hmin : min (k+1) n = n := min_eq_right (by omega)
    have hh := hc k
    rw [hmin] at hh
    exact le_antisymm (by simpa using Finset.card_le_univ (reached W j k)) hh
  have hi : i ∈ reached W j k := hu ▸ Finset.mem_univ i
  simpa [reached] using hi

theorem positive_of_dense (hn : 0 < n) (hW : CommClass n ρ W)
    (hρ : (n : ℝ)*ρ < 1) (i j : Fin n) :
    0 < (lazy W ^ 1) i j := by
  have H := hW.contraction (Pi.single j 1)
  rw [Matrix.mulVec_single_one, norm_single, mul_one] at H
  have h := (abs_le.mp ((abs_coord_le _ i).trans H)).1
  change -ρ ≤ W i j - (1/n : ℝ) at h
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hρ' := (lt_div_iff₀ hn').2 (by rwa [mul_comm])
  rw [pow_one, lazy_apply]
  split <;> linarith

theorem positive_radius (hn : 0 < n) (hρ : 0 ≤ ρ) (hρ₁ : ρ < 1)
    (hW : CommClass n ρ W) {K : ℕ}
    (hK : n-1 ≤ K ∨ 2 * (n : ℝ) * ((1+ρ)/2)^K < 1 ∨ (K = 1 ∧ (n : ℝ)*ρ < 1)) :
    ∀ i j, 0 < (lazy W ^ K) i j := by
  rcases hK with h | h | ⟨rfl, h⟩
  · exact positive_of_size hn hρ hρ₁ hW h
  · exact positive_of_power hn hρ hW h
  · exact positive_of_dense hn hW h

end Network
end ULB63
