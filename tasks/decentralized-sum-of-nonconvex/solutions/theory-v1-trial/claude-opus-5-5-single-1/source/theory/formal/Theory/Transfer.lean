import Mathlib

/-!
# Elementary monotonicity facts used to transfer the chain bound between parameter values
-/

noncomputable section

namespace ULB63
namespace Transfer

lemma sqrt_gt_one {x : ℝ} (hx : 0 < x) : 1 < Real.sqrt (1 + x) := by
  rw [Real.lt_sqrt (by norm_num)]; linarith

lemma ell_pos {s : ℝ} (hs : 1 < s) : 0 < Real.log ((s + 1) / (s - 1)) := by
  apply Real.log_pos
  rw [one_lt_div (by linarith)]; linarith

/-- `s ↦ log ((s+1)/(s-1))` is antitone on `(1, ∞)`. -/
lemma ell_anti {s t : ℝ} (hs : 1 < s) (hst : s ≤ t) :
    Real.log ((t + 1) / (t - 1)) ≤ Real.log ((s + 1) / (s - 1)) := by
  apply Real.log_le_log (div_pos (by linarith) (by linarith))
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- If `T ≤ D₀ (Λ₀/(2ℓ₀) - 2)`, `T ≥ 0`, `D₀ ≤ D`, `Λ₀ ≤ Λ` and `0 < ℓ ≤ ℓ₀`, then
`T ≤ D (Λ/(2ℓ) - 2)`. -/
lemma transfer {Λ₀ Λ ℓ₀ ℓ D₀ D T : ℝ} (hℓ : 0 < ℓ) (hℓℓ : ℓ ≤ ℓ₀) (hΛ : Λ₀ ≤ Λ) (hD₀ : 0 < D₀)
    (hD : D₀ ≤ D) (hT0 : 0 ≤ T) (hT : T ≤ D₀ * (Λ₀ / (2 * ℓ₀) - 2)) :
    T ≤ D * (Λ / (2 * ℓ) - 2) := by
  have hℓ₀ : 0 < ℓ₀ := lt_of_lt_of_le hℓ hℓℓ
  have hX₀ : 2 ≤ Λ₀ / (2 * ℓ₀) := by
    by_contra h
    push_neg at h
    have : D₀ * (Λ₀ / (2 * ℓ₀) - 2) < 0 := mul_neg_of_pos_of_neg hD₀ (by linarith)
    linarith
  have hΛ₀ : 0 ≤ Λ₀ := by
    by_contra h
    push_neg at h
    have : Λ₀ / (2 * ℓ₀) < 0 := div_neg_of_neg_of_pos h (by linarith)
    linarith
  have hX : Λ₀ / (2 * ℓ₀) ≤ Λ / (2 * ℓ) := by
    calc Λ₀ / (2 * ℓ₀) ≤ Λ₀ / (2 * ℓ) :=
          div_le_div_of_nonneg_left hΛ₀ (by linarith) (by linarith)
      _ ≤ Λ / (2 * ℓ) := div_le_div_of_nonneg_right hΛ (by linarith)
  calc T ≤ D₀ * (Λ₀ / (2 * ℓ₀) - 2) := hT
    _ ≤ D * (Λ₀ / (2 * ℓ₀) - 2) := mul_le_mul_of_nonneg_right hD (by linarith)
    _ ≤ D * (Λ / (2 * ℓ) - 2) := mul_le_mul_of_nonneg_left (by linarith) (by linarith)

end Transfer
end ULB63
