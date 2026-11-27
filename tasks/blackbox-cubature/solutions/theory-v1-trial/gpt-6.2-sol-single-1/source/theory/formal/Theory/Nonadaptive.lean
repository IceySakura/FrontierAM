import Theory.Slab

/-! A finite prior with disjoint query sites for its level and bit indices. -/

open MeasureTheory

namespace Cubature

noncomputable section

/-- The query points are fixed before any answers are available. -/
structure NonadaptiveRule (Q : Type*) (n : ℕ) where
  point : Fin n → Q
  postprocess : (Fin n → Bool) → ℝ

def NonadaptiveRule.estimate {Q : Type*} {n : ℕ} (a : NonadaptiveRule Q n) (o : Q → Bool) : ℝ :=
  a.postprocess (o ∘ a.point)

def bitsign (b : Bool) : ℝ := if b then 1 else -1

def bitvalue (b : Bool) : ℝ := if b then 1 else 0

def flip {M : Type*} [DecidableEq M] (i : M) (b : M → Bool) : M → Bool :=
  Function.update b i (! b i)

@[simp] theorem flip_at {M : Type*} [DecidableEq M] (i : M) (b : M → Bool) :
    flip i b i = ! b i := by simp [flip]

theorem flip_off {M : Type*} [DecidableEq M] {i j : M} (h : j ≠ i) (b : M → Bool) :
    flip i b j = b j := by simp [flip, h]

theorem flip_inv {M : Type*} [DecidableEq M] (i : M) : Function.Involutive (flip i) := by
  intro b
  funext j
  by_cases e : j = i
  · subst j; simp
  · simp [flip_off e]

theorem bitsign_flip (b : Bool) : bitsign (!b) = -bitsign b := by cases b <;> norm_num [bitsign]
theorem bitsign_sq (b : Bool) : (bitsign b)^2 = 1 := by cases b <;> norm_num [bitsign]
private theorem bitvalue_flip (b : Bool) : bitvalue (!b) = bitvalue b - bitsign b := by
  cases b <;> norm_num [bitvalue, bitsign]

section Bits

variable {M : Type*} [Fintype M] [DecidableEq M]

theorem sum_flip (i : M) (f : (M → Bool) → ℝ) :
    ∑ b, f (flip i b) = ∑ b, f b :=
  Fintype.sum_equiv ((flip_inv i).toPerm _) _ _ (fun _ => rfl)

theorem sum_target_flip (c : M → ℝ) (i : M) (b : M → Bool) :
    ∑ j, c j * bitvalue (flip i b j) =
      (∑ j, c j * bitvalue (b j)) - c i * bitsign (b i) := by
  rw [← Finset.sum_erase_add _ _ (Finset.mem_univ i),
      ← Finset.sum_erase_add _ (fun j => c j * bitvalue (b j)) (Finset.mem_univ i),
      flip_at, bitvalue_flip, mul_sub, ← add_sub_assoc]
  congr 2
  apply Finset.sum_congr rfl
  intro j hj
  rw [flip_off (Finset.ne_of_mem_erase hj)]

private theorem bitsign_mult (i j : M) :
    ∑ b : M → Bool, bitsign (b i) * bitsign (b j) =
      if i = j then (Fintype.card (M → Bool) : ℝ) else 0 := by
  by_cases e : i = j
  · subst j
    simp only [← pow_two, bitsign_sq, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_one, ↓reduceIte]
  · rw [ite_eq_right e]
    have he := sum_flip i (fun b => bitsign (b i) * bitsign (b j))
    simp only [flip_at, flip_off (Ne.symm e), bitsign_flip, neg_mul, Finset.sum_neg_distrib] at he
    linarith

private theorem linear_cross (c : M → ℝ) (t : ℝ) (F : (M → Bool) → ℝ) (i : M)
    (hF : ∀ b, F (flip i b) = F b) :
    ∑ b, (F b - (t + ∑ j, c j * bitvalue (b j))) * bitsign (b i) =
      -(Fintype.card (M → Bool) : ℝ) * (c i / 2) := by
  let g (b : M → Bool) := F b - (t + ∑ j, c j * bitvalue (b j))
  have hg (b) : g (flip i b) = g b + c i * bitsign (b i) := by
    simp only [g, hF, sum_target_flip]
    ring
  have he := sum_flip i (fun b => g b * bitsign (b i))
  have h (b : M → Bool) :
      g (flip i b) * bitsign (flip i b i) = -(g b * bitsign (b i)) - c i := by
    rw [hg, flip_at, bitsign_flip]
    linear_combination -c i * bitsign_sq (b i)
  simp only [h, Finset.sum_sub_distrib, Finset.sum_neg_distrib, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul] at he
  change (∑ b, g b * bitsign (b i)) = _
  linarith

/-- Squared error in fitting a weighted sum of fair independent bits; unseen bits give a
Bessel-type lower bound even if the fit is nonlinear in the other bits. -/
theorem unseen_bits_square_error (S : Finset M) (c : M → ℝ) (t : ℝ)
    (F : (M → Bool) → ℝ) (hF : ∀ i ∈ S, ∀ b, F (flip i b) = F b) :
    (Fintype.card (M → Bool) : ℝ) * (∑ i ∈ S, (c i / 2)^2) ≤
      ∑ b, (F b - (t + ∑ j, c j * bitvalue (b j)))^2 := by
  let g (b : M → Bool) := F b - (t + ∑ j, c j * bitvalue (b j))
  let A (i : M) := c i / 2
  let V (b : M → Bool) := ∑ i ∈ S, A i * bitsign (b i)
  let D : ℝ := Fintype.card (M → Bool)
  change D * (∑ i ∈ S, (A i)^2) ≤ ∑ b, (g b)^2
  have hlin :
      ∑ b, g b * V b = -(D * ∑ i ∈ S, (A i)^2) := by
    simp only [V]
    simp_rw [Finset.mul_sum, mul_left_comm (g _) (A _)]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum]
    have he (i : M) (hi : i ∈ S) : (∑ b, g b * bitsign (b i)) = -D * A i :=
      linear_cross c t F i (hF i hi)
    trans ∑ i ∈ S, -(D * (A i)^2)
    · exact Finset.sum_congr rfl fun i hi => by rw [he i hi]; ring
    · simp only [Finset.sum_neg_distrib, Finset.mul_sum]
  have hquad :
      ∑ b, (V b)^2 = D * (∑ i ∈ S, (A i)^2) := by
    calc
      _ = ∑ b : M → Bool, ∑ i ∈ S, ∑ j ∈ S,
        A i * A j * (bitsign (b i) * bitsign (b j)) := by
          simp only [V, pow_two, Finset.sum_mul, Finset.mul_sum]
          refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun i _ =>
            Finset.sum_congr rfl fun j _ => ?_
          ring
      _ = ∑ i ∈ S, ∑ j ∈ S, A i * A j * (if i = j then D else 0) := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl (fun i _ => ?_)
        rw [Finset.sum_comm]
        simp_rw [← Finset.mul_sum, bitsign_mult]
        rfl
      _ = _ := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i hi => ?_
        rw [Finset.sum_eq_single_of_mem i hi (fun j _ h => by simp [Ne.symm h])]
        simp [pow_two, mul_comm D]
  have hp : 0 ≤ ∑ b : M → Bool, (g b + V b)^2 := Finset.sum_nonneg fun b _ => sq_nonneg _
  have hexp (b : M → Bool) : (g b + V b)^2 = (g b)^2 + 2 * (g b * V b) + (V b)^2 := by ring
  simp_rw [hexp, Finset.sum_add_distrib, ← Finset.mul_sum, hlin, hquad] at hp
  linarith

end Bits

section Prior

variable {Q J M Ω : Type*} [Fintype J] [Fintype M] [DecidableEq J] [DecidableEq M]

/-- Disjoint sites for a hidden level and local binary perturbations. There is no geometric
assumption in this lemma; the identifies-at-most-one property is used via `site`. -/
theorem nonadaptive_prior_lower_bound [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (alg : Ω → NonadaptiveRule Q n)
    (site : Q → Option (J × M)) (oracle : J → (M → Bool) → Q → Bool)
    (base : J → ℝ) (coeff : J → M → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hcoeff : ∀ j i, c ≤ coeff j i)
    (hsite : ∀ j i b q, site q ≠ some (j,i) → oracle j (flip i b) q = oracle j b q)
    (hJ : Nonempty J) (hM : Nonempty M)
    (hn : 2 * n ≤ Fintype.card J * Fintype.card M)
    (integrable_risk : ∀ j b, Integrable (fun ω => ((alg ω).estimate (oracle j b) -
      (base j + ∑ i, coeff j i * bitvalue (b i)))^2) μ) :
    ∃ j b, (Fintype.card M : ℝ) * c^2 / 8 ≤
      ∫ ω, ((alg ω).estimate (oracle j b) -
        (base j + ∑ i, coeff j i * bitvalue (b i)))^2 ∂μ := by
  classical
  let D : ℝ := Fintype.card (M → Bool)
  let loss (j : J) (b : M → Bool) (ω : Ω) :=
    ((alg ω).estimate (oracle j b) - (base j + ∑ i, coeff j i * bitvalue (b i)))^2
  let T (ω : Ω) : Finset (Option (J × M)) :=
    Finset.univ.image (fun i : Fin n => site ((alg ω).point i))
  let S (ω : Ω) (j : J) : Finset M :=
    Finset.univ.filter (fun i => some (j,i) ∉ T ω)
  have hsites (ω : Ω) :
      (Fintype.card J : ℝ) * (Fintype.card M : ℝ) / 2 ≤ ∑ j, ((S ω j).card : ℝ) := by
    have hT : (T ω).card ≤ n := Finset.card_image_le.trans_eq (by simp)
    have hsum : ∑ j, (S ω j).card =
        (Finset.univ.filter (fun a : J × M => some a ∉ T ω)).card := by
      have H (j : J) : (S ω j).card = ∑ i : M, if some (j,i) ∉ T ω then 1 else 0 :=
        (Finset.sum_boole (R := ℕ) _ _).symm
      simp_rw [H]
      symm
      have hE : (Finset.univ.filter (fun a : J × M => some a ∉ T ω)).card =
          ∑ a : J × M, if some a ∉ T ω then 1 else 0 := (Finset.sum_boole (R := ℕ) _ _).symm
      rw [hE, Fintype.sum_prod_type]
    let B := Finset.univ.filter (fun a : J × M => some a ∈ T ω)
    have hB : B.card ≤ n := by
      have he : (B.map ⟨some, Option.some_injective _⟩).card = B.card := Finset.card_map _
      rw [← he]
      refine (Finset.card_le_card ?_).trans hT
      intro x hx
      obtain ⟨a, ha, rfl⟩ := Finset.mem_map.mp hx
      exact (Finset.mem_filter.mp ha).2
    have ht : B.card + (Finset.univ.filter (fun a : J × M => some a ∉ T ω)).card =
        Fintype.card J * Fintype.card M := by
      unfold B
      rw [Finset.card_filter_add_card_filter_not, Finset.card_univ, Fintype.card_prod]
    have hN : Fintype.card J * Fintype.card M ≤ 2 * ∑ j, (S ω j).card := by omega
    have hN' : (Fintype.card J : ℝ) * (Fintype.card M : ℝ) ≤ 2 * (∑ j, (S ω j).card : ℕ) := by
      exact_mod_cast hN
    rw [Nat.cast_sum] at hN'
    linarith
  have hsingle (ω : Ω) (j : J) :
      D * (((S ω j).card : ℝ) * (c / 2)^2) ≤ ∑ b, loss j b ω := by
    refine le_trans ?_ (unseen_bits_square_error (S ω j) (coeff j) (base j)
      ((alg ω).estimate ∘ oracle j) (fun i hi b => ?_))
    · apply mul_le_mul_of_nonneg_left _ (Nat.cast_nonneg _)
      rw [← nsmul_eq_mul, ← Finset.sum_const]
      exact Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (by positivity)
        (by linarith [hcoeff j i]) 2
    · apply congrArg (alg ω).postprocess
      funext a
      apply hsite
      intro eq
      exact (Finset.mem_filter.mp hi).2 (Finset.mem_image.mpr ⟨a, Finset.mem_univ _, eq⟩)
  let bound : ℝ := (Fintype.card M : ℝ) * c^2 / 8
  have hω (ω : Ω) :
      ((Fintype.card J : ℝ) * D) * bound ≤ ∑ j, ∑ b, loss j b ω := by
    calc
      _ = D * (((Fintype.card J : ℝ) * (Fintype.card M : ℝ) / 2) * (c/2)^2) := by
        unfold bound
        ring
      _ ≤ D * ((∑ j, ((S ω j).card : ℝ)) * (c/2)^2) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right (hsites ω) (sq_nonneg _))
          (Nat.cast_nonneg _)
      _ = ∑ j, D * (((S ω j).card : ℝ) * (c/2)^2) := by
        rw [Finset.sum_mul, Finset.mul_sum]
      _ ≤ _ := Finset.sum_le_sum fun j _ => hsingle ω j
  have hI : ((Fintype.card J : ℝ) * D) * bound ≤ ∑ j, ∑ b, ∫ ω, loss j b ω ∂μ := by
    have hi (j : J) : Integrable (fun ω => ∑ b, loss j b ω) μ :=
      integrable_finsetSum _ (fun b _ => integrable_risk j b)
    have hm := integral_mono (integrable_const (((Fintype.card J : ℝ) * D) * bound))
      (integrable_finsetSum Finset.univ (fun j _ => hi j)) hω
    rw [integral_const, show μ.real Set.univ = 1 by simp, one_smul,
      integral_finsetSum _ (fun j _ => hi j)] at hm
    have he (j : J) : (∫ ω, ∑ b, loss j b ω ∂μ) = ∑ b, ∫ ω, loss j b ω ∂μ :=
      integral_finsetSum _ (fun b _ => integrable_risk j b)
    simpa only [he] using hm
  change ∃ j b, bound ≤ ∫ ω, loss j b ω ∂μ
  by_contra! hn'
  have ht (j : J) : (∑ b, ∫ ω, loss j b ω ∂μ) < D * bound := by
    have h := Finset.sum_lt_sum_of_nonempty (Finset.univ_nonempty : (Finset.univ : Finset (M → Bool)).Nonempty)
      (fun b _ => hn' j b)
    simpa [D, nsmul_eq_mul] using h
  have hx := Finset.sum_lt_sum_of_nonempty (Finset.univ_nonempty : (Finset.univ : Finset J).Nonempty)
    (fun j _ => ht j)
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← mul_assoc] at hx
  exact (not_lt_of_ge hI) hx

end Prior

end

end Cubature
