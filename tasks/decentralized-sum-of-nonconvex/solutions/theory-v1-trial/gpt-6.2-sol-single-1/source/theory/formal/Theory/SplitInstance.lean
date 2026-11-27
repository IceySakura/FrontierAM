import Theory.Split
import Theory.Path

open scoped ENNReal RealInnerProductSpace

set_option maxHeartbeats 1100000

noncomputable section

namespace ULB63
namespace SplitChain

open Chain

variable {d n m : ℕ} {L μ Δ q : ℝ}

abbrev scale (L : ℝ) : ℝ := L/8/4
abbrev s (L q Δ : ℝ) : ℝ := amplitude (L/8) q Δ
abbrev radius (n : ℕ) : ℕ := n/4
abbrev left (i : Fin n) : Prop := i.val < radius n
abbrev right (i : Fin n) : Prop := n-radius n ≤ i.val
abbrev gain (n : ℕ) : ℝ := (n:ℝ)/(radius n:ℝ)

def driver (d : ℕ) (L q Δ : ℝ) : E (d+1) :=
  (-(scale L)*(1-q^2)*s L q Δ) • EuclideanSpace.single 0 1

def baseLaw (d p : ℕ) (L q Δ : ℝ) : E (d+1) → ℝ :=
  rawQuad (edge (d+1) p) (scale L*q) (if p=0 then driver d L q Δ else 0)

def groupLaw (n d p : ℕ) (L q Δ : ℝ) : E (d+1) → ℝ :=
  rawQuad (edge (d+1) p) (gain n*(scale L*q)) (gain n • (if p=0 then driver d L q Δ else 0))

def nodeLaw (d : ℕ) (L q Δ : ℝ) (i : Fin n) : E (d+1) → ℝ :=
  if left i then groupLaw n d 0 L q Δ
  else if right i then groupLaw n d 1 L q Δ
  else fun _ => 0

def weights (d : ℕ) (L q : ℝ) (i : Fin (d+1)) : ℝ :=
  scale L * (1-q)^2 + scale L*q*(1-q) *
    ((if i=0 then 1 else 0)+(if i=Fin.last d then 1 else 0))

theorem weights_nonneg (hL : 0 ≤ L) (hq : 0 ≤ q) (hq₁ : q ≤ 1) (i : Fin (d+1)) :
    0 ≤ weights d L q i := by
  have hh : 0 ≤ 1-q := sub_nonneg.mpr hq₁
  unfold weights
  positivity

def hardSplit (n m d : ℕ) (L q Δ : ℝ) : Instance n m (d+1) where
  F i _ := nodeLaw d L q Δ i
  h x := (diagValue (weights d L q) x : EReal)
  x0 := 0
  prox v η := diagProx (weights d L q) v η

private theorem scale_upper (hn : 4 ≤ n) :
    0 ≤ gain n ∧ gain n ≤ 7 := by
  have hn' : 0 < (n:ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hr : 0 < (radius n:ℝ) := by exact_mod_cast (show 0 < n/4 by omega)
  have hp : n ≤ 7 * (n/4) := by omega
  exact ⟨by positivity, (div_le_iff₀ hr).2 (by exact_mod_cast hp)⟩

private theorem group_smooth (hn : 4 ≤ n) (hL : 0 ≤ L) (hq : 0 ≤ q) (hq₁ : q ≤ 1)
    (p : ℕ) :
    LSmooth L (groupLaw n d p L q Δ) := by
  obtain ⟨hN,hN₁⟩ := scale_upper hn
  apply rawQuad_smooth _ ((edge _ p).opNorm_le_bound (by norm_num) (edge_norm p)) (by positivity) _
  have h1 : scale L*q ≤ scale L := mul_le_of_le_one_right (by positivity) hq₁
  have h2 := mul_le_mul hN₁ h1 (show 0 ≤ scale L*q by positivity) (by norm_num : (0:ℝ) ≤ 7)
  exact (mul_le_mul_of_nonneg_left h2 (by norm_num)).trans (by dsimp [scale]; linarith)

private theorem group_eq (p : ℕ) (x : E (d+1)) :
    groupLaw n d p L q Δ x = gain n * baseLaw d p L q Δ x := by
  unfold groupLaw baseLaw rawQuad
  rw [real_inner_smul_left]
  ring

private theorem card_left (hn : 4 ≤ n) :
    (Finset.univ.filter (left : Fin n → Prop)).card = radius n := by
  let r : Fin n := ⟨radius n, by change n/4 < n; omega⟩
  have he : Finset.univ.filter (left : Fin n → Prop) = Finset.Iio r := by
    ext i
    simp [r, Fin.lt_def]
  rw [he, Fin.card_Iio]

private theorem card_right (hn : 4 ≤ n) :
    (Finset.univ.filter (right : Fin n → Prop)).card = radius n := by
  let r : Fin n := ⟨n-radius n, by change n-n/4 < n; omega⟩
  have he : Finset.univ.filter (right : Fin n → Prop) = Finset.Ici r := by
    ext i
    simp [r, Fin.le_def]
  rw [he, Fin.card_Ici]
  change n-(n-n/4) = n/4
  omega

theorem split_f (hn : 4 ≤ n) (hm : 0 < m) (x : E (d+1)) :
    (hardSplit n m d L q Δ).f x = baseLaw d 0 L q Δ x + baseLaw d 1 L q Δ x := by
  let a (p : ℕ) := gain n * baseLaw d p L q Δ x
  have hm' : (m:ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  have hn' : (n:ℝ) ≠ 0 := by exact_mod_cast (show n ≠ 0 by omega)
  have hD : (radius n : ℝ) ≠ 0 := by exact_mod_cast (show n/4 ≠ 0 by omega)
  have hlocal (i : Fin n) :
      (1/(m:ℝ))*∑ j : Fin m, (hardSplit n m d L q Δ).F i j x =
        (if left i then a 0 else 0)+(if right i then a 1 else 0) := by
    have H : (1/(m:ℝ))*∑ _ : Fin m, nodeLaw d L q Δ i x = nodeLaw d L q Δ i x := by
      simp [Finset.sum_const, nsmul_eq_mul, hm']
    change (1/(m:ℝ))*∑ _ : Fin m, nodeLaw d L q Δ i x = _
    rw [H, nodeLaw]
    by_cases hl : left i
    · have hr : ¬right i := by dsimp [left,right,radius] at *; omega
      simp [hl, hr, group_eq, a]
    · by_cases hr : right i <;> simp [hl, hr, group_eq, a]
  unfold Instance.f
  simp only [hlocal, Finset.sum_add_distrib, ← Finset.sum_filter, Finset.sum_const,
    card_left hn, card_right hn, nsmul_eq_mul]
  dsimp [a]
  field_simp

private theorem lag_exact (x : E (d+1)) : ‖lag (d+1) x‖^2 = ‖x‖^2 - (x (Fin.last d))^2 := by
  have hp (i : Fin d) : prev i.succ = i.castSucc := Fin.ext (by simp [prev])
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq,
    Fin.sum_univ_succ (fun j => (lag (d+1) x j)^2), Fin.sum_univ_castSucc (fun j => (x j)^2)]
  simp [lag_apply, hp]

private theorem norm_split (hq : 0 ≤ q) (hq₁ : q ≤ 1) (x : E (d+1)) :
    ‖residual (d+1) q x‖^2 =
      q*(‖edge (d+1) 0 x‖^2+‖edge (d+1) 1 x‖^2) +
      (1-q)^2*‖x‖^2+q*(1-q)*((x 0)^2+(x (Fin.last d))^2) := by
  have H (i : Fin (d+1)) :
      (residual (d+1) q x i)^2 =
        q*((edge (d+1) 0 x i)^2+(edge (d+1) 1 x i)^2) +
        (1-q)*(x i)^2-q*(1-q)*(lag (d+1) x i)^2 +
        q*(1-q)*(if i = 0 then (x i)^2 else 0) := by
    by_cases hi : i = 0
    · subst i
      have hw : 0 ≤ 1-q^2 := by nlinarith [mul_nonneg hq (sub_nonneg.mpr hq₁)]
      simp [residual_apply, weight, lag_apply, edge_apply, selected, mul_pow, Real.sq_sqrt hw]
      ring
    · have hi' : i.val ≠ 0 := fun h => hi (Fin.ext h)
      rcases Nat.mod_two_eq_zero_or_one i.val with h | h
      · simp [residual_apply, weight, lag_apply, edge_apply, selected, hi, hi', h]
        ring
      · simp [residual_apply, weight, lag_apply, edge_apply, selected, hi, hi', h]
        ring
  have hsum := congrArg (fun f : Fin (d+1) → ℝ => ∑ i, f i) (funext H)
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true] at hsum
  simp only [← EuclideanSpace.real_norm_sq_eq] at hsum
  rw [hsum, lag_exact]
  ring

private theorem diag_split (x : E (d+1)) :
    diagValue (weights d L q) x = (scale L)/2*((1-q)^2*‖x‖^2+
      q*(1-q)*((x 0)^2+(x (Fin.last d))^2)) := by
  unfold diagValue
  rw [EuclideanSpace.real_norm_sq_eq]
  have H (i : Fin (d+1)) : weights d L q i / 2 * (x i)^2 =
      scale L/2*(1-q)^2*(x i)^2 +
      (if i=0 then scale L/2*q*(1-q)*(x i)^2 else 0) +
      (if i=Fin.last d then scale L/2*q*(1-q)*(x i)^2 else 0) := by
    unfold weights
    split_ifs <;> ring
  simp only [H, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true]
  ring

private theorem cross_eq (hq : 0 ≤ q) (hq₁ : q ≤ 1) (x : E (d+1)) (z : ℝ) :
    ⟪residual (d+1) q x, residual (d+1) q (target (d+1) q z)⟫ = (1-q^2)*z*x 0 := by
  rw [PiLp.inner_apply, Finset.sum_eq_single (0 : Fin (d+1))]
  · have hw : 0 ≤ 1-q^2 := by nlinarith [mul_nonneg hq (sub_nonneg.mpr hq₁)]
    simp only [residual_apply, weight, Fin.val_zero, ite_true, lag_apply, target_apply, pow_zero,
      mul_one, mul_zero, sub_zero, RCLike.inner_apply, RCLike.conj_to_real]
    rw [show (Real.sqrt (1-q^2)*z) * (Real.sqrt (1-q^2)*x 0) =
      (Real.sqrt (1-q^2))^2*z*x 0 by ring, Real.sq_sqrt hw]
  · intro i _ hi
    rw [residual_target i (fun h => hi (Fin.ext h))]
    simp
  · simp

theorem split_phi (hn : 4 ≤ n) (hm : 0 < m) (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1)
    (hΔ : 0 ≤ Δ) (x : E (d+1)) :
    (hardSplit n m d L q Δ).φ x = ((F (d+1) (scale L) q (s L q Δ) x - Δ : ℝ) : EReal) := by
  rw [Instance.φ, split_f hn hm]
  change ((baseLaw d 0 L q Δ x + baseLaw d 1 L q Δ x : ℝ):EReal) +
    ((diagValue (weights d L q) x:ℝ):EReal) = _
  rw [← EReal.coe_add, EReal.coe_eq_coe_iff, diag_split]
  unfold baseLaw rawQuad F
  rw [map_sub, norm_sub_sq_real, cross_eq hq hq₁.le,
    residual_target_zero_norm (Nat.zero_lt_succ _) hq hq₁.le, norm_split hq hq₁.le]
  simp only [ite_true, one_ne_zero, ite_false, inner_zero_left, add_zero, driver, real_inner_smul_left,
    EuclideanSpace.inner_single_left, RCLike.conj_to_real, one_mul]
  have hs := amplitude_eq (L := L/8) (q := q) (Δ := Δ) (by positivity) hq hq₁ hΔ
  change scale L/2 * (1-q^2) * (s L q Δ)^2 = Δ at hs
  linear_combination -hs

theorem split_min (hn : 4 ≤ n) (hm : 0 < m) (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1)
    (hΔ : 0 ≤ Δ) (x : E (d+1)) :
    (hardSplit n m d L q Δ).φ (target (d+1) q (s L q Δ)) ≤ (hardSplit n m d L q Δ).φ x := by
  rw [split_phi hn hm hL hq hq₁ hΔ, split_phi hn hm hL hq hq₁ hΔ, EReal.coe_le_coe_iff]
  have h : 0 ≤ F (d+1) (scale L) q (s L q Δ) x := by unfold F; positivity
  simp [F] at *
  linarith

theorem split_star (hn : 4 ≤ n) (hm : 0 < m) (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1)
    (hΔ : 0 ≤ Δ) : (hardSplit n m d L q Δ).φstar = ((-Δ : ℝ) : EReal) := by
  let u := target (d+1) q (s L q Δ)
  have hp : (hardSplit n m d L q Δ).φ u = ((-Δ : ℝ) : EReal) := by
    rw [split_phi hn hm hL hq hq₁ hΔ]
    simp [F,u]
  rw [← hp]
  exact le_antisymm (iInf_le _ u) (le_iInf (split_min hn hm hL hq hq₁ hΔ))

theorem split_inClass (hn : 4 ≤ n) (hm : 0 < m) (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1)
    (hμq : μ ≤ scale L*(1-q)^2) (hΔ : 0 ≤ Δ) :
    (hardSplit n m d L q Δ).InClass L μ Δ where
  smooth i j := by
    change LSmooth L (nodeLaw d L q Δ i)
    unfold nodeLaw
    split
    · exact group_smooth hn hL.le hq hq₁.le _
    · split
      · exact group_smooth hn hL.le hq hq₁.le _
      · exact ⟨differentiable_const _, fun x y => by simp; positivity⟩
  f_convex := by
    have he := funext (split_f (d := d) (L := L) (q := q) (Δ := Δ) hn hm)
    rw [he]
    exact (rawQuad_convex _ (by positivity) _).add (rawQuad_convex _ (by positivity) _)
  h_ne_bot _ := EReal.coe_ne_bot _
  h_proper := ⟨0,EReal.coe_ne_top _⟩
  h_closed := diag_closed _
  h_convex := diag_convex _ (weights_nonneg hL.le hq hq₁.le)
  prox_exact v η hη := diag_exact _ (weights_nonneg hL.le hq hq₁.le) v hη
  φ_strongly_convex := by
    rw [funext (split_phi (d := d) hn hm hL hq hq₁ hΔ), stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    have hc := (stronglyConvexE_coe_iff (d := d+1) _ _).1
      (strong_F (L := L/8) (by positivity) hq hq₁.le hμq (s L q Δ))
    rw [strongConvexOn_iff_convex] at hc
    have H := hc.add_const (-Δ)
    convert H using 1
    funext x
    simp only [Pi.add_apply]
    ring
  minimizer := ⟨target _ q (s L q Δ), split_min hn hm hL hq hq₁ hΔ⟩
  initial_gap := by
    change (hardSplit n m d L q Δ).φ 0 - (hardSplit n m d L q Δ).φstar ≤ _
    rw [split_phi hn hm hL hq hq₁ hΔ, split_star hn hm hL hq hq₁ hΔ, ← EReal.coe_sub,
      EReal.coe_le_coe_iff, sub_neg_eq_add, sub_add_cancel]
    unfold F
    rw [zero_sub, map_neg, norm_neg, residual_target_zero_norm (Nat.zero_lt_succ _) hq hq₁.le,
      ← mul_assoc, amplitude_eq (L := L/8) (by positivity) hq hq₁ hΔ]

end SplitChain
end ULB63
