import Theory.ProxFacts

/-! Deterministic restarted accelerated proximal-gradient steps. -/

open scoped ENNReal
noncomputable section

namespace ULB63.Acceleration

variable {n m d : ℕ} {I : Instance n m d} {M L μ Δ : ℝ}

def tau (j : ℕ) : ℝ := (j + 2 : ℝ) / 2

theorem one_le_tau (j : ℕ) : 1 ≤ tau j := by unfold tau; linarith [Nat.cast_nonneg (α := ℝ) j]

theorem tau_pos (j : ℕ) : 0 < tau j := zero_lt_one.trans_le (one_le_tau j)

def extrapolate (j : ℕ) (x z : E d) : E d := (1 - (tau j)⁻¹) • x + (tau j)⁻¹ • z

def momentum (j : ℕ) (old p : E d) : E d := (tau j) • p - (tau j - 1) • old

def step (I : Instance n m d) (L : ℝ) (j : ℕ) (state : E d × E d) : E d × E d :=
  let p := proxGrad I L (extrapolate j state.1 state.2)
  (p, momentum j state.1 p)

theorem step_finite (hI : I.InClass M μ Δ) (hL : 0 < L) (j : ℕ) (state : E d × E d) :
    FiniteAt I (step I L j state).1 := prox_grad_finite hI hL _

theorem estimate_step (hI : I.InClass M μ Δ) (hμ : 0 ≤ μ)
    (hL : 0 < L) (hf : LSmooth L I.f) {s x z : E d} (hs : FiniteAt I s)
    (hx : FiniteAt I x) (j : ℕ) :
    let st := step I L j (x,z)
    (tau j)^2 * realGap I s st.1 + (L/2)*‖st.2-s‖^2 ≤
      ((tau j)^2 - tau j) * realGap I s x + (L/2)*‖z-s‖^2 := by
  let T := tau j
  let t := T⁻¹
  let y := extrapolate j x z
  let p := proxGrad I L y
  let z' := momentum j x p
  let u := t • s + (1-t) • x
  have hT : 0 < T := tau_pos j
  have ht0 : 0 ≤ t := by dsimp [t]; positivity
  have ht1 : t ≤ 1 := by
    dsimp [t, T]
    exact inv_le_one_of_one_le₀ (one_le_tau j)
  obtain ⟨hu, hval⟩ := phi_interp hI hs hx ht0 ht1
  change FiniteAt I u at hu
  have hval' : realValue I u ≤ t * realValue I s + (1-t)*realValue I x := by
    have hc : 0 ≤ μ/2*t*(1-t)*‖s-x‖^2 := by positivity [sub_nonneg.mpr ht1]
    exact (le_add_of_nonneg_right hc).trans hval
  have hstep := prox_grad_three_point hI hL hf y u hu
  change realValue I p ≤ realValue I u + L/2*(‖u-y‖^2 - ‖u-p‖^2) at hstep
  have hA : u-y = t • (s-z) := by dsimp [u, y, extrapolate, t, T]; module
  have hB : u-p = t • (s-z') := by
    dsimp only [u, z', momentum]
    have h1 : t * T = 1 := inv_mul_cancel₀ hT.ne'
    have h2 : t * (T-1) = 1-t := by linarith only [h1]
    change t • s + (1-t) • x - p = t • (s - (T • p - (T-1) • x))
    simp only [smul_sub, smul_smul, h1, h2, one_smul]
    abel
  rw [hA, hB, norm_smul, norm_smul, Real.norm_of_nonneg ht0, mul_pow, mul_pow,
    norm_sub_rev s z, norm_sub_rev s z'] at hstep
  have hlast : realGap I s p ≤ (1-t) * realGap I s x + L/2*t^2*(‖z-s‖^2-‖z'-s‖^2) := by
    dsimp only [realGap]
    linarith only [hval', hstep]
  change T^2 * realGap I s p + (L/2)*‖z'-s‖^2 ≤
    (T^2-T) * realGap I s x + (L/2)*‖z-s‖^2
  have hmul := mul_le_mul_of_nonneg_left hlast (sq_nonneg T)
  have hcancel : T^2 * (1-t) = T^2 - T := by dsimp [t]; field_simp
  have hcancel' : T^2 * (L/2*t^2) = L/2 := by dsimp [t]; field_simp
  rw [mul_add, ← mul_assoc, hcancel, ← mul_assoc, hcancel'] at hmul
  linarith only [hmul]

/-- An epoch started with both state vectors at the previous restart point. -/
def stage (I : Instance n m d) (L : ℝ) (x : E d) : ℕ → E d × E d
  | 0 => (x,x)
  | j+1 => step I L j (stage I L x j)

theorem stage_finite (hI : I.InClass M μ Δ) (hL : 0 < L) {x : E d} (hx : FiniteAt I x) (j : ℕ) :
    FiniteAt I (stage I L x j).1 := by
  cases j with
  | zero => exact hx
  | succ j => exact step_finite hI hL j _

theorem epoch_estimate (hI : I.InClass M μ Δ) (hμ : 0 ≤ μ)
    (hL : 0 < L) (hf : LSmooth L I.f) {s x : E d} (hs : FiniteAt I s)
    (hmin : ∀ w, I.φ s ≤ I.φ w) (hx : FiniteAt I x) (j : ℕ) :
    (tau j)^2 * realGap I s (stage I L x (j+1)).1 + (L/2)*‖(stage I L x (j+1)).2-s‖^2 ≤
      (L/2)*‖x-s‖^2 := by
  induction j with
  | zero =>
    have h := estimate_step hI hμ hL hf hs (z := x) hx 0
    norm_num [tau] at h
    simpa only [stage, tau, Nat.cast_zero, zero_add, div_self (by norm_num : (2:ℝ) ≠ 0),
      one_pow, one_mul] using h
  | succ j ih =>
    have h := estimate_step hI hμ hL hf hs (z := (stage I L x (j+1)).2)
      (stage_finite hI hL hx (j+1)) (j+1)
    have ht : (tau (j+1))^2 - tau (j+1) ≤ (tau j)^2 := by simp [tau]; ring_nf; norm_num
    exact h.trans ((add_le_add_left (mul_le_mul_of_nonneg_right ht
      (realGap_nonneg hI hs hmin (stage_finite hI hL hx (j+1)))) _).trans ih)

def epochLength (L μ : ℝ) : ℕ := ⌈Real.sqrt (16 * (L/μ))⌉₊

theorem epochLength_pos {L μ : ℝ} (hL : 0 < L) (hμ : 0 < μ) : 0 < epochLength L μ := by
  rw [epochLength, Nat.lt_ceil, Nat.cast_zero]
  positivity

theorem epoch_halves (hI : I.InClass M μ Δ) (hμ : 0 < μ) (hL : 0 < L) (hf : LSmooth L I.f)
    {s x : E d} (hs : FiniteAt I s) (hmin : ∀ w, I.φ s ≤ I.φ w) (hx : FiniteAt I x) :
    realGap I s (stage I L x (epochLength L μ)).1 ≤ (1/2)*realGap I s x := by
  obtain ⟨j, hj⟩ := Nat.exists_eq_succ_of_ne_zero (epochLength_pos hL hμ).ne'
  rw [hj]
  have he := epoch_estimate hI hμ.le hL hf hs hmin hx j
  have hdist := strong_gap_distance hI hs hmin hx
  have hlen := Nat.le_ceil (Real.sqrt (16*(L/μ)))
  change _ ≤ (epochLength L μ : ℝ) at hlen
  rw [hj] at hlen
  have hsq := (Real.sqrt_le_iff.mp hlen).2
  have hcoef : 4*(L/μ) ≤ (tau j)^2 := by
    simp only [tau, Nat.cast_succ] at hsq ⊢
    nlinarith only [hsq, Nat.cast_nonneg (α := ℝ) j]
  have hcoef' := (div_le_iff₀ hμ).1 (show (4*L)/μ ≤ (tau j)^2 by
    simpa only [mul_div_assoc] using hcoef)
  have hgap : 0 ≤ realGap I s (stage I L x (j+1)).1 :=
    realGap_nonneg hI hs hmin (stage_finite hI hL hx _)
  have hnon : 0 ≤ (L/2)*‖(stage I L x (j+1)).2-s‖^2 := by positivity
  have hh := mul_le_mul_of_nonneg_right hcoef' hgap
  have hdist' := mul_le_mul_of_nonneg_left hdist (show 0 ≤ 2*L by positivity)
  have he' := mul_le_mul_of_nonneg_left (show
    (tau j)^2 * realGap I s (stage I L x (j+1)).1 ≤ (L/2)*‖x-s‖^2 by linarith only [he, hnon]) hμ.le
  apply (mul_le_mul_iff_right₀ (show 0 < 4*L by positivity)).mp
  nlinarith only [hh, hdist', he']

def restart (I : Instance n m d) (L : ℝ) (p : ℕ) : ℕ → E d
  | 0 => I.x0
  | k+1 => (stage I L (restart I L p k) p).1

theorem restart_gap (hI : I.InClass M μ Δ) (hμ : 0 < μ) (hL : 0 < L) (hf : LSmooth L I.f)
    {s : E d} (hs : FiniteAt I s) (hmin : ∀ w, I.φ s ≤ I.φ w) (k : ℕ) :
    FiniteAt I (restart I L (epochLength L μ) k) ∧
      realGap I s (restart I L (epochLength L μ) k) ≤ (1/2)^k * Δ := by
  induction k with
  | zero => simpa [restart] using initial_feasible_gap hI hs hmin
  | succ k ih =>
    refine ⟨stage_finite hI hL ih.1 _, ?_⟩
    rw [restart, pow_succ]
    linarith only [epoch_halves hI hμ hL hf hs hmin ih.1,
      mul_le_mul_of_nonneg_left ih.2 (by norm_num : 0 ≤ (1/2 : ℝ))]

end ULB63.Acceleration
