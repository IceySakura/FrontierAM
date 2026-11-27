import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Tactic

/-! Exponential decay dominates the inverse-square scale along polynomial budgets. -/

open Filter
open scoped Topology

namespace Cubature

theorem polynomial_budget_gap (C d : ℕ) (U : ℝ) {S q : ℝ} (hS : 0 < S) (hq : 0 < q) :
    ∃ r₀ : ℕ, 1 ≤ r₀ ∧ ∀ r : ℕ, r₀ ≤ r →
      (U/(2:ℝ)^r)^2 < S^2 / (32*((C*r^d+1:ℕ):ℝ)^2*q) := by
  have hpoly := tendsto_pow_const_div_const_pow_of_one_lt d (r := (2:ℝ)) (by norm_num)
  have hone : Tendsto (fun r : ℕ => 1/(2:ℝ)^r) atTop (𝓝 0) := by
    simpa only [pow_zero] using tendsto_pow_const_div_const_pow_of_one_lt 0
      (r := (2:ℝ)) (by norm_num)
  have hc : Tendsto (fun _ : ℕ => (C:ℝ)) atTop (𝓝 (C:ℝ)) := tendsto_const_nhds
  have hB : Tendsto (fun r : ℕ => ((C*r^d+1:ℕ):ℝ)/(2:ℝ)^r) atTop (𝓝 0) := by
    simpa only [Nat.cast_add,Nat.cast_mul,Nat.cast_pow,Nat.cast_one,add_div,mul_div_assoc,mul_zero,add_zero]
      using (hc.mul hpoly).add hone
  have hU : Tendsto (fun _ : ℕ => U) atTop (𝓝 U) := tendsto_const_nhds
  have hq' : Tendsto (fun _ : ℕ => (32:ℝ)*q) atTop (𝓝 ((32:ℝ)*q)) := tendsto_const_nhds
  have hT : Tendsto (fun r : ℕ => (32:ℝ)*q * (U * (((C*r^d+1:ℕ):ℝ)/(2:ℝ)^r))^2) atTop (𝓝 0) := by
    simpa using hq'.mul ((hU.mul hB).pow 2)
  have he : ∀ᶠ r : ℕ in atTop, (32:ℝ)*q * (U * (((C*r^d+1:ℕ):ℝ)/(2:ℝ)^r))^2 < S^2 :=
    hT.eventually (gt_mem_nhds (sq_pos_of_pos hS))
  obtain ⟨k,hk⟩ := eventually_atTop.mp he
  refine ⟨max 1 k, le_max_left _ _, fun r hr => ?_⟩
  have h := hk r ((le_max_right _ _).trans hr)
  apply (lt_div_iff₀ (by positivity)).mpr
  calc
    _ = (32:ℝ)*q * (U * (((C*r^d+1:ℕ):ℝ)/(2:ℝ)^r))^2 := by ring
    _ < _ := h

theorem polynomial_budget_gap_all {C d : ℕ} (hC : 0 < C) (hd : 0 < d) (U : ℝ)
    {S q : ℝ} (hS : 0 < S) (hq : 0 < q) :
    ∃ N₀ : ℕ, 1 ≤ N₀ ∧ ∀ N : ℕ, N₀ ≤ N →
      ∃ r : ℕ, 1 ≤ r ∧ C*r^d+1 ≤ N ∧
        (U/(2:ℝ)^r)^2 < S^2 / (32*(N:ℝ)^2*q) := by
  classical
  let P (r : ℕ) := C*r^d+1
  have hMono : Monotone P := fun _ _ h =>
    Nat.add_le_add_right (Nat.mul_le_mul_left _ (Nat.pow_le_pow_left h _)) _
  have hRatio (r : ℕ) (hr : 1 ≤ r) : P (r+1) ≤ 2^d*P r := by
    have hOne : 1 ≤ (2:ℕ)^d := Nat.succ_le_of_lt (by positivity)
    unfold P
    calc
      _ ≤ C*(2*r)^d+1 := Nat.add_le_add_right
        (Nat.mul_le_mul_left _ (Nat.pow_le_pow_left (by omega) _)) _
      _ = 2^d*(C*r^d)+1 := by rw [mul_pow]; ring
      _ ≤ 2^d*(C*r^d)+2^d := Nat.add_le_add_left hOne _
      _ = _ := by ring
  have hq' : 0 < q*((2:ℝ)^d)^2 := by positivity
  obtain ⟨r₀,hr₀,hgap⟩ := polynomial_budget_gap C d U hS hq'
  refine ⟨P r₀,by unfold P; omega,fun N hN => ?_⟩
  have hExists : ∃ s : ℕ, N < P s := by
    use N+1
    have hPow : N+1 ≤ (N+1)^d := by
      simpa only [pow_one] using pow_le_pow_right' (show 1 ≤ N+1 by omega) (Nat.succ_le_of_lt hd)
    have hProd : (N+1)^d ≤ C*(N+1)^d := by
      calc
        _ = 1*(N+1)^d := (Nat.one_mul _).symm
        _ ≤ _ := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hC)
    unfold P
    omega
  let s := Nat.find hExists
  have hs : N < P s := Nat.find_spec hExists
  have hs₀ : r₀ < s := by
    by_contra! h
    exact (not_lt_of_ge ((hMono h).trans hN)) hs
  let r := s-1
  have hr : r₀ ≤ r := by dsimp [r]; omega
  have hr' : 1 ≤ r := hr₀.trans hr
  have hs' : r+1 = s := by dsimp [r]; omega
  have hPre : P r ≤ N := le_of_not_gt (Nat.find_min hExists (show r < s by omega))
  have hPost : N ≤ 2^d*P r := (hs.le.trans_eq (congrArg P hs'.symm)).trans (hRatio r hr')
  have hPost' : (N:ℝ) ≤ (2:ℝ)^d*((P r:ℕ):ℝ) := by exact_mod_cast hPost
  have hn : 0 < (N:ℝ) := by
    have hN₁ : 1 ≤ N := by unfold P at hN; omega
    exact_mod_cast (show 0 < N by omega)
  refine ⟨r,hr',hPre,?_⟩
  calc
    _ < S^2/(32*((P r:ℕ):ℝ)^2*(q*((2:ℝ)^d)^2)) := hgap r hr
    _ = S^2/(32*((2:ℝ)^d*((P r:ℕ):ℝ))^2*q) := by congr 1; ring
    _ ≤ _ := by
      apply div_le_div_of_nonneg_left (sq_nonneg _) (by positivity)
      gcongr

end Cubature
