import Mathlib.Tactic

/-! Compact quadratic splines and the affine remainders of their tensor products. -/

namespace Cubature

noncomputable section

def positivePart (x : ℝ) : ℝ := max 0 x

def spline (x : ℝ) : ℝ :=
  (2/3) * (positivePart x ^ 2 - 3 * positivePart (x-1) ^ 2 +
    3 * positivePart (x-2) ^ 2 - positivePart (x-3) ^ 2)

def splineSlope (x : ℝ) : ℝ :=
  (4/3) * (positivePart x - 3 * positivePart (x-1) +
    3 * positivePart (x-2) - positivePart (x-3))

theorem continuous_spline : Continuous spline := by
  unfold spline positivePart
  fun_prop

private theorem spline_forms (x : ℝ) :
    (x ≤ 0 → spline x = 0 ∧ splineSlope x = 0) ∧
    (0 ≤ x → x ≤ 1 → spline x = (2/3)*x^2 ∧ splineSlope x = (4/3)*x) ∧
    (1 ≤ x → x ≤ 2 →
      spline x = (2/3)*(1+2*(x-1)*(2-x)) ∧ splineSlope x = (4/3)*(3-2*x)) ∧
    (2 ≤ x → x ≤ 3 → spline x = (2/3)*(x-3)^2 ∧ splineSlope x = (4/3)*(x-3)) ∧
    (3 ≤ x → spline x = 0 ∧ splineSlope x = 0) := by
  have hL (t : ℝ) (h : t ≤ 0) : positivePart t = 0 := max_eq_left h
  have hR (t : ℝ) (h : 0 ≤ t) : positivePart t = t := max_eq_right h
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro h
    simp [spline, splineSlope, hL _ h, hL (x-1) (by linarith), hL (x-2) (by linarith),
      hL (x-3) (by linarith)]
  · intro h₀ h₁
    simp [spline, splineSlope, hR _ h₀, hL (x-1) (by linarith), hL (x-2) (by linarith),
      hL (x-3) (by linarith)]
  · intro h₀ h₁
    simp only [spline, splineSlope, hR x (by linarith), hR (x-1) (by linarith),
      hL (x-2) (by linarith), hL (x-3) (by linarith)]
    constructor <;> ring
  · intro h₀ h₁
    simp only [spline, splineSlope, hR x (by linarith), hR (x-1) (by linarith),
      hR (x-2) (by linarith), hL (x-3) (by linarith)]
    constructor <;> ring
  · intro h
    simp only [spline, splineSlope, hR x (by linarith), hR (x-1) (by linarith),
      hR (x-2) (by linarith), hR (x-3) (by linarith)]
    constructor <;> ring

theorem spline_mem (x : ℝ) : spline x ∈ Set.Icc 0 1 := by
  obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := spline_forms x
  rcases le_or_gt x 0 with h | h
  · rw [(h₁ h).1]; exact ⟨le_rfl, zero_le_one⟩
  rcases le_or_gt x 1 with h' | h'
  · rw [(h₂ h.le h').1]
    exact ⟨by positivity, by nlinarith [mul_nonneg h.le (sub_nonneg.mpr h')]⟩
  rcases le_or_gt x 2 with h'' | h''
  · rw [(h₃ h'.le h'').1]
    exact ⟨by nlinarith [mul_nonneg (sub_nonneg.mpr h'.le) (sub_nonneg.mpr h'')],
      by nlinarith [sq_nonneg (x - 3/2)]⟩
  rcases le_or_gt x 3 with h''' | h'''
  · rw [(h₄ h''.le h''').1]
    exact ⟨by positivity, by
      nlinarith [mul_nonneg (sub_nonneg.mpr h''') (sub_nonneg.mpr h''.le)]⟩
  · rw [(h₅ h'''.le).1]; exact ⟨le_rfl, zero_le_one⟩

theorem spline_core {x : ℝ} (hx : x ∈ Set.Icc 1 2) : 2/3 ≤ spline x := by
  rw [((spline_forms x).2.2.1 hx.1 hx.2).1]
  nlinarith [mul_nonneg (sub_nonneg.mpr hx.1) (sub_nonneg.mpr hx.2)]

theorem spline_support {x : ℝ} (hx : spline x ≠ 0) : x ∈ Set.Ioo 0 3 :=
  ⟨lt_of_not_ge (fun h => hx ((spline_forms x).1 h).1),
    lt_of_not_ge (fun h => hx ((spline_forms x).2.2.2.2 h).1)⟩

theorem splineSlope_zero {x : ℝ} (hx : spline x = 0) : splineSlope x = 0 := by
  obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := spline_forms x
  rcases le_or_gt x 0 with h | h
  · exact (h₁ h).2
  rcases le_or_gt x 1 with h' | h'
  · rw [(h₂ h.le h').1] at hx
    nlinarith [sq_pos_of_pos h]
  rcases le_or_gt x 2 with h'' | h''
  · linarith [spline_core ⟨h'.le,h''⟩]
  rcases le_or_gt x 3 with h''' | h'''
  · have he := h₄ h''.le h'''
    rw [he.1] at hx
    have hz : x - 3 = 0 := by nlinarith [sq_nonneg (x-3)]
    simp [he.2, hz]
  · exact (h₅ h'''.le).2

theorem abs_splineSlope_le (x : ℝ) : |splineSlope x| ≤ 2 := by
  obtain ⟨h₁, h₂, h₃, h₄, h₅⟩ := spline_forms x
  rw [abs_le]
  rcases le_or_gt x 0 with h | h
  · rw [(h₁ h).2]; norm_num
  rcases le_or_gt x 1 with h' | h'
  · rw [(h₂ h.le h').2]; constructor <;> linarith
  rcases le_or_gt x 2 with h'' | h''
  · rw [(h₃ h'.le h'').2]; constructor <;> linarith
  rcases le_or_gt x 3 with h''' | h'''
  · rw [(h₄ h''.le h''').2]; constructor <;> linarith
  · rw [(h₅ h'''.le).2]; norm_num

private theorem positivePart_remainder (x y : ℝ) :
    |positivePart y ^ 2 - positivePart x ^ 2 - 2 * positivePart x * (y-x)| ≤ (y-x)^2 := by
  rw [abs_le]
  unfold positivePart
  rcases le_total x 0 with hx | hx <;> rcases le_total y 0 with hy | hy
  · rw [max_eq_left hx, max_eq_left hy]
    constructor <;> nlinarith [sq_nonneg (y-x)]
  · rw [max_eq_left hx, max_eq_right hy]
    constructor <;> nlinarith [sq_nonneg (y-x), mul_nonneg (neg_nonneg.mpr hx) hy]
  · rw [max_eq_right hx, max_eq_left hy]
    constructor <;> nlinarith [sq_nonneg (y-x), mul_nonneg hx (neg_nonneg.mpr hy), sq_nonneg x,
      sq_nonneg y]
  · rw [max_eq_right hx, max_eq_right hy]
    constructor <;> nlinarith [sq_nonneg (y-x)]

theorem spline_remainder (x y : ℝ) :
    |spline y - spline x - splineSlope x * (y-x)| ≤ 6*(y-x)^2 := by
  have H₀ := abs_le.mp (positivePart_remainder x y)
  have H₁ := abs_le.mp (positivePart_remainder (x-1) (y-1))
  have H₂ := abs_le.mp (positivePart_remainder (x-2) (y-2))
  have H₃ := abs_le.mp (positivePart_remainder (x-3) (y-3))
  rw [abs_le]
  unfold spline splineSlope
  constructor <;> nlinarith [H₀.1, H₀.2, H₁.1, H₁.2, H₂.1, H₂.2, H₃.1, H₃.2, sq_nonneg (y-x)]

theorem spline_lipschitz (x y : ℝ) : |spline y - spline x| ≤ 8*|y-x| := by
  let d := y - x
  have hh : spline y - spline x = (spline y - spline x - splineSlope x * d) +
      splineSlope x * d := by ring
  rcases le_total |d| 1 with h | h
  · rw [hh]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul]
    have hL := mul_le_mul_of_nonneg_right (abs_splineSlope_le x) (abs_nonneg d)
    have hT := spline_remainder x y
    change |spline y - spline x - splineSlope x * d| ≤ 6*d^2 at hT
    have hh' := mul_le_mul_of_nonneg_left h (abs_nonneg d)
    rw [← pow_two, sq_abs] at hh'
    linarith
  · have h₁ := spline_mem x
    have h₂ := spline_mem y
    have h₃ : |spline y - spline x| ≤ 1 :=
      abs_le.mpr ⟨by linarith [h₂.1,h₁.2], by linarith [h₂.2,h₁.1]⟩
    exact h₃.trans (by linarith)

section Tensor

variable {ι : Type*} [DecidableEq ι]

private theorem prod_unit {a : ι → ℝ} (ha : ∀ j, a j ∈ Set.Icc 0 1) (s : Finset ι) :
    (∏ i ∈ s, a i) ∈ Set.Icc 0 1 :=
  ⟨Finset.prod_nonneg fun i _ => (ha i).1,
    (Finset.prod_le_prod₀ (fun i _ => (ha i).1) (fun i _ => (ha i).2)).trans_eq
      Finset.prod_const_one⟩

/-- Linearization of a product of `[0,1]`-valued factors, without dividing by a factor. -/
theorem tensor_remainder (a b c : ι → ℝ)
    {δ L H : ℝ} (hδ : 0 ≤ δ) (hL : 0 ≤ L) (hH : 0 ≤ H)
    (ha : ∀ j, a j ∈ Set.Icc 0 1) (hb : ∀ j, b j ∈ Set.Icc 0 1)
    (hd : ∀ j, |b j - a j| ≤ L * δ)
    (hr : ∀ j, |b j - a j - c j| ≤ H * δ^2) (s : Finset ι) :
    |(∏ i ∈ s, b i) - ∏ i ∈ s, a i| ≤ (s.card : ℝ) * L * δ ∧
    |(∏ i ∈ s, b i) - (∏ i ∈ s, a i) -
      ∑ i ∈ s, c i * (∏ j ∈ s.erase i, a j)| ≤
      ((s.card : ℝ) * H + (s.card : ℝ)^2 * L^2) * δ^2 := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert i s his ih =>
    let P := ∏ j ∈ s, a j
    let Q := ∏ j ∈ s, b j
    let T := ∑ j ∈ s, c j * (∏ k ∈ s.erase j, a k)
    let n : ℝ := s.card
    have hP := prod_unit ha s
    have hQ := prod_unit hb s
    have hP' : |P| ≤ 1 := by rw [abs_of_nonneg hP.1]; exact hP.2
    have hQ' : |Q| ≤ 1 := by rw [abs_of_nonneg hQ.1]; exact hQ.2
    have hA : |a i| ≤ 1 := by rw [abs_of_nonneg (ha i).1]; exact (ha i).2
    have hx : |b i * Q - a i * P| ≤ (n+1) * L * δ := by
      rw [show b i * Q - a i * P = (b i - a i)*Q + a i*(Q-P) by ring]
      refine (abs_add_le _ _).trans ?_
      rw [abs_mul, abs_mul]
      have h₁ := mul_le_mul (hd i) hQ' (abs_nonneg _) (mul_nonneg hL hδ)
      have h₂ := mul_le_mul hA ih.1 (abs_nonneg _) zero_le_one
      change |a i| * |Q-P| ≤ 1 * (n*L*δ) at h₂
      nlinarith only [h₁,h₂]
    have hT :
        ∑ j ∈ insert i s, c j * (∏ k ∈ (insert i s).erase j, a k) = c i * P + a i * T := by
      change _ = c i * P + a i * (∑ j ∈ s, c j * (∏ k ∈ s.erase j, a k))
      rw [Finset.sum_insert his, Finset.erase_insert his, Finset.mul_sum]
      congr 1
      refine Finset.sum_congr rfl (fun j hj => ?_)
      have hn : j ≠ i := fun eq => his (eq ▸ hj)
      rw [Finset.erase_insert_of_ne (Ne.symm hn), Finset.prod_insert (fun h => his (Finset.mem_erase.mp h).2)]
      ring
    have he : b i * Q - a i * P - (c i * P + a i * T) =
        (a i * (Q - P - T) + (b i - a i - c i) * P) + (b i - a i) * (Q-P) := by ring
    rw [Finset.card_insert_of_notMem his, Nat.cast_add, Nat.cast_one,
      Finset.prod_insert his, Finset.prod_insert his, hT]
    refine ⟨hx, ?_⟩
    change |b i * Q - a i * P - (c i * P + a i * T)| ≤
      ((n+1)*H + (n+1)^2*L^2)*δ^2
    rw [he]
    refine (abs_add_le (a i * (Q-P-T) + (b i-a i-c i)*P) ((b i-a i)*(Q-P))).trans ?_
    refine (add_le_add (abs_add_le (a i*(Q-P-T)) ((b i-a i-c i)*P))
      (le_refl |(b i-a i)*(Q-P)|)).trans ?_
    simp only [abs_mul]
    have h₁ := mul_le_mul hA ih.2 (abs_nonneg _) zero_le_one
    have h₂ := mul_le_mul (hr i) hP' (abs_nonneg _) (mul_nonneg hH (sq_nonneg _))
    have h₃ := mul_le_mul (hd i) ih.1 (abs_nonneg _) (mul_nonneg hL hδ)
    change |a i| * |Q-P-T| ≤ 1 * ((n*H+n^2*L^2)*δ^2) at h₁
    change |b i - a i| * |Q-P| ≤ L*δ*(n*L*δ) at h₃
    have hn : 0 ≤ n := Nat.cast_nonneg _
    nlinarith only [h₁,h₂,h₃,mul_nonneg (by linarith : 0 ≤ n+1)
      (mul_nonneg (sq_nonneg L) (sq_nonneg δ))]

end Tensor

end

end Cubature
