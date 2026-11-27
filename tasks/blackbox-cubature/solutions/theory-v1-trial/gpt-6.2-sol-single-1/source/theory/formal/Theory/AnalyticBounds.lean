import Theory.PolynomialGrid

/-! Constants for geometric chart remainders. -/

namespace Cubature

noncomputable section

theorem interpolationNorm_ell (p r : ℕ) :
    interpolationNorm p r + 1 ≤ (2:ℝ)^(3*r*p+1) := by
  rw [pow_succ, ← interpolationNorm_pow p r]
  linarith [interpolationNorm_ge_one p r]

theorem geometricBudget {p r k : ℕ} (hr : 0 < r) :
    2*((p+1)*2*(2^k*r)^p)*(r+(3*r*p+1))+1 ≤
      4*(p+1)*(2^k)^p*(3*p+2)*r^(p+1)+1 := by
  have h : r+(3*r*p+1) ≤ (3*p+2)*r := by
    calc
      _ = (3*p+1)*r+1 := by ring
      _ ≤ (3*p+1)*r+r := Nat.add_le_add_left (Nat.succ_le_of_lt hr) _
      _ = _ := by ring
  calc
    _ ≤ 2*((p+1)*2*(2^k*r)^p)*((3*p+2)*r)+1 := Nat.add_le_add_right (Nat.mul_le_mul_left _ h) _
    _ = _ := by rw [mul_pow, pow_succ r p]; ring

theorem geometricRadiusError {p r k : ℕ} {A L : ℝ} (hA : 0 ≤ A) (hL : 1 ≤ L)
    (hk : 2*(64:ℝ)^p*L ≤ (2:ℝ)^k) :
    (2:ℝ)^(3*r*p+1) * (interpolationNorm p r * (A * L^r) / (2:ℝ)^(r*k)) ≤
      2*A / (2:ℝ)^r := by
  have h₀ : 0 ≤ (64:ℝ)^p * L := mul_nonneg (by positivity) (by linarith)
  have hz : (0:ℝ) < (2:ℝ)^k := by positivity
  have h₁ : (64:ℝ)^p * L / (2:ℝ)^k ≤ 1/2 :=
    (div_le_iff₀ hz).mpr (by linarith [hk])
  have hSquare : interpolationNorm p r * interpolationNorm p r = ((64:ℝ)^p)^r := by
    unfold interpolationNorm
    rw [← mul_pow, ← mul_pow, show (8:ℝ)*8=64 by norm_num, ← pow_mul, ← pow_mul, Nat.mul_comm r p]
  have hpow : ((64:ℝ)^p * L / (2:ℝ)^k)^r ≤ (1/2:ℝ)^r :=
    pow_le_pow_left₀ (div_nonneg h₀ hz.le) h₁ _
  calc
    _ = (2*A) * (interpolationNorm p r * interpolationNorm p r * L^r) / ((2:ℝ)^k)^r := by
      rw [pow_succ, ← interpolationNorm_pow p r, Nat.mul_comm r k, pow_mul]
      ring
    _ = (2*A) * ((64:ℝ)^p * L / (2:ℝ)^k)^r := by
      rw [hSquare, div_pow, mul_pow, mul_div_assoc]
    _ ≤ (2*A) * (1/2:ℝ)^r := mul_le_mul_of_nonneg_left hpow (by positivity)
    _ = _ := by rw [div_pow, one_pow, mul_one_div]

end

end Cubature
