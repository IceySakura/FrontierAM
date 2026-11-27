import Theory.Nonadaptive
import Mathlib.Data.Finset.Option

/-! Pathwise unseen-bit accounting for arbitrary adaptive decision trees. -/

open MeasureTheory

namespace Cubature

namespace QueryTree

variable {Q M A : Type*} [DecidableEq M]

def revealed (site : Q → Option M) (o : Q → Bool) : QueryTree Q A → Finset M
  | .returnValue _ => ∅
  | .ask q k => (site q).toFinset ∪ revealed site o (k (o q))

theorem revealed_card (site : Q → Option M) (o : Q → Bool) (t : QueryTree Q A) :
    (t.revealed site o).card ≤ t.depth := by
  induction t with
  | returnValue => simp [revealed,depth]
  | ask q k ih =>
    have ht : (site q).toFinset.card ≤ 1 := by cases site q <;> simp
    have hi := ih (o q)
    have hu := Finset.card_union_le ((site q).toFinset) ((k (o q)).revealed site o)
    have hd : (k (o q)).depth ≤ max (k false).depth (k true).depth := by cases o q <;> simp
    simp only [revealed, depth]
    omega

theorem revealed_flip_invariant (site : Q → Option M) (o o' : Q → Bool)
    (t : QueryTree Q A) (i : M)
    (hdiff : ∀ q, site q ≠ some i → o' q = o q)
    (hi : i ∉ t.revealed site o) :
    t.run o' = t.run o ∧ t.revealed site o' = t.revealed site o := by
  induction t with
  | returnValue => exact ⟨rfl,rfl⟩
  | ask q k ih =>
    simp only [revealed, Finset.mem_union, not_or] at hi
    have hs : site q ≠ some i := by
      intro h
      apply hi.1
      rw [h, Option.toFinset_some, Finset.mem_singleton]
    have he := hdiff q hs
    obtain ⟨hr,hvis⟩ := ih (o q) hi.2
    simp only [run, revealed, he, hr, hvis, and_self]

end QueryTree

section Bits

variable {M : Type*} [Fintype M] [DecidableEq M]

/-- The unobserved set can depend on the bits, provided flipping an unobserved bit
preserves both that set and the output. -/
theorem trace_unseen_square_error (S : (M → Bool) → Finset M) (c : M → ℝ)
    (t : ℝ) (F : (M → Bool) → ℝ)
    (hf : ∀ b i, i ∈ S b → F (flip i b) = F b ∧ S (flip i b) = S b) :
    (∑ b, ∑ i ∈ S b, (c i / 2)^2) ≤
      ∑ b, (F b - (t + ∑ i, c i * bitvalue (b i)))^2 := by
  classical
  let u (i : M) := c i / 2
  let g (b : M → Bool) := F b - (t + ∑ i, c i * bitvalue (b i))
  let V (b : M → Bool) (i : M) : ℝ := if i ∈ S b then u i * bitsign (b i) else 0
  let D (b : M → Bool) (i : M) : ℝ := if i ∈ S b then (u i)^2 else 0
  have hmiss (b : M → Bool) (i : M) (hi : i ∉ S b) : i ∉ S (flip i b) := by
    intro he
    have hh := (hf (flip i b) i he).2
    rw [flip_inv i b] at hh
    exact hi (hh ▸ he)
  have hdiag (b : M → Bool) (i : M) : V b i * V b i = D b i := by
    by_cases hi : i ∈ S b
    · simp only [V,D,hi,↓reduceIte]
      linear_combination (u i)^2 * bitsign_sq (b i)
    · simp [V,D,hi]
  have hcross_point (i : M) (b : M → Bool) :
      g (flip i b) * V (flip i b) i = -(g b * V b i) - 2 * D b i := by
    by_cases hi : i ∈ S b
    · obtain ⟨hF,hS⟩ := hf b i hi
      simp only [g,V,D, hF,hS,hi,↓reduceIte, sum_target_flip,flip_at,bitsign_flip]
      unfold u
      linear_combination -(c i)^2 / 2 * bitsign_sq (b i)
    · have hm := hmiss b i hi
      simp [V,D,hi,hm]
  have hcross (i : M) : (∑ b, g b * V b i) = -(∑ b, D b i) := by
    have H := sum_flip i (fun b => g b * V b i)
    simp only [hcross_point, Finset.sum_sub_distrib, Finset.sum_neg_distrib, ← Finset.mul_sum] at H
    linarith
  have hoff (i j : M) (hij : i ≠ j) : (∑ b, V b i * V b j) = 0 := by
    have hswap (b : M → Bool) : V (flip i b) i * V (flip i b) j = -(V b i * V b j) := by
      by_cases hi : i ∈ S b
      · have hS := (hf b i hi).2
        simp only [V,hS,hi,↓reduceIte,flip_at,bitsign_flip,flip_off (Ne.symm hij),mul_neg,neg_mul]
      · have hm := hmiss b i hi
        simp [V,hi,hm]
    have H := sum_flip i (fun b => V b i * V b j)
    simp only [hswap,Finset.sum_neg_distrib] at H
    linarith
  have hVS (b : M → Bool) :
      (∑ i, D b i) = ∑ i ∈ S b, (u i)^2 := by
    simp only [D, ← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
  let U (b : M → Bool) := ∑ i, V b i
  let K := ∑ b, ∑ i ∈ S b, (u i)^2
  have hlinear : (∑ b, g b * U b) = -K := by
    simp only [U,K,Finset.mul_sum]
    rw [Finset.sum_comm]
    simp only [hcross,Finset.sum_neg_distrib]
    rw [Finset.sum_comm, ← Finset.sum_neg_distrib]
    simp_rw [hVS]
    simp only [Finset.sum_neg_distrib]
  have hquadratic : (∑ b, (U b)^2) = K := by
    calc
      _ = ∑ b : M → Bool, ∑ i : M, ∑ j : M, V b i * V b j := by
        simp only [U,pow_two,Finset.sum_mul,Finset.mul_sum]
        exact Finset.sum_congr rfl (fun b _ => Finset.sum_congr rfl
          (fun i _ => Finset.sum_congr rfl (fun j _ => mul_comm _ _)))
      _ = ∑ i : M, ∑ b, D b i := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl (fun i _ => ?_)
        rw [Finset.sum_comm, Finset.sum_eq_single_of_mem i (Finset.mem_univ i)]
        · simp only [hdiag]
        · exact fun j _ hj => hoff i j (Ne.symm hj)
      _ = _ := by rw [Finset.sum_comm]; simp only [hVS,K]
  have hnonneg : 0 ≤ ∑ b, (g b + U b)^2 := Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  have hp (b : M → Bool) : (g b + U b)^2 = (g b)^2 + 2*(g b * U b) + (U b)^2 := by ring
  simp_rw [hp, Finset.sum_add_distrib, ← Finset.mul_sum, hlinear, hquadratic] at hnonneg
  change K ≤ ∑ b, (g b)^2
  linarith

end Bits

/-- Adaptive lower bound for disjoint bit sites. It charges only the single realized path. -/
theorem adaptive_prior_lower_bound {Q M Ω : Type*} [Fintype M] [DecidableEq M]
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (alg : Ω → QueryTree Q ℝ) (hdepth : ∀ ω, (alg ω).depth ≤ n)
    (site : Q → Option M) (oracle : (M → Bool) → Q → Bool)
    (base : ℝ) (coeff : M → ℝ) {c : ℝ} (hc : 0 ≤ c) (hcoeff : ∀ i, c ≤ coeff i)
    (hsite : ∀ i b q, site q ≠ some i → oracle (flip i b) q = oracle b q)
    (hn : 2 * n ≤ Fintype.card M)
    (integrable_risk : ∀ b, Integrable (fun ω => ((alg ω).run (oracle b) -
      (base + ∑ i, coeff i * bitvalue (b i)))^2) μ) :
    ∃ b, (Fintype.card M : ℝ) * c^2 / 8 ≤
      ∫ ω, ((alg ω).run (oracle b) - (base + ∑ i, coeff i * bitvalue (b i)))^2 ∂μ := by
  classical
  let D : ℝ := Fintype.card (M → Bool)
  let bound : ℝ := (Fintype.card M : ℝ) * c^2 / 8
  let loss (b : M → Bool) (ω : Ω) :=
    ((alg ω).run (oracle b) - (base + ∑ i, coeff i * bitvalue (b i)))^2
  have hω (ω : Ω) : D * bound ≤ ∑ b, loss b ω := by
    let S (b : M → Bool) := ((alg ω).revealed site (oracle b))ᶜ
    have hflip (b : M → Bool) (i : M) (hi : i ∈ S b) :
        (alg ω).run (oracle (flip i b)) = (alg ω).run (oracle b) ∧
        S (flip i b) = S b := by
      have hr := QueryTree.revealed_flip_invariant site _ _ (alg ω) i (hsite i b)
        (Finset.mem_compl.mp hi)
      exact ⟨hr.1, congrArg (fun a : Finset M => aᶜ) hr.2⟩
    refine le_trans ?_ (trace_unseen_square_error S coeff base (fun b => (alg ω).run (oracle b)) hflip)
    have hcS (b : M → Bool) : (Fintype.card M : ℝ) / 2 ≤ ((S b).card : ℝ) := by
      have hd := (QueryTree.revealed_card site (oracle b) (alg ω)).trans (hdepth ω)
      have HH : (S b).card = Fintype.card M - ((alg ω).revealed site (oracle b)).card :=
        Finset.card_compl _
      have hC : Fintype.card M ≤ 2 * (S b).card := by omega
      have hl : (Fintype.card M : ℝ) ≤ 2 * ((S b).card : ℝ) := by exact_mod_cast hC
      linarith
    have hS (b : M → Bool) : bound ≤ ∑ i ∈ S b, (coeff i / 2)^2 := by
      calc
        _ = ((Fintype.card M : ℝ)/2) * (c/2)^2 := by unfold bound; ring
        _ ≤ ((S b).card : ℝ) * (c/2)^2 := mul_le_mul_of_nonneg_right (hcS b) (sq_nonneg _)
        _ = ∑ _ ∈ S b, (c/2)^2 := by simp [nsmul_eq_mul]
        _ ≤ _ := Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (by positivity)
          (by linarith [hcoeff i]) 2
    simpa only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] using
      (Finset.sum_le_sum (s := Finset.univ) (fun b _ => hS b))
  have hi := integral_mono (integrable_const (D * bound))
    (integrable_finsetSum Finset.univ (fun b _ => integrable_risk b)) hω
  rw [integral_const, show μ.real Set.univ = 1 by simp, one_smul,
    integral_finsetSum _ (fun b _ => integrable_risk b)] at hi
  by_contra! h
  have hh := Finset.sum_lt_sum_of_nonempty
    (Finset.univ_nonempty : (Finset.univ : Finset (M → Bool)).Nonempty) (fun b _ => h b)
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at hh
  exact (not_lt_of_ge hi) hh

end Cubature
