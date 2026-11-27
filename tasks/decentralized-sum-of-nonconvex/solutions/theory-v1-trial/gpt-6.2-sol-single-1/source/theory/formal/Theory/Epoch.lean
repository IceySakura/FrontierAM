import Theory.FiniteMean

open scoped RealInnerProductSpace

noncomputable section

namespace ULB63
namespace Epoch

open Composite FiniteMean

variable {n m d : ℕ} {L μ Δ : ℝ} {I : Instance n m d}
variable {σ : Type} [Fintype σ] [Nonempty σ]

def fraction (r : ℕ) : ℝ := (r : ℝ) / ((r : ℝ)+1)

theorem fraction_nonneg (r : ℕ) : 0 ≤ fraction r := by unfold fraction; positivity

theorem fraction_le (r : ℕ) : fraction r ≤ 1 := by
  unfold fraction
  apply (div_le_iff₀ (by positivity)).2
  linarith

structure Trace (d : ℕ) where
  x : E d
  z : E d
  mean : E d

/-- `mean` averages the points just before each proximal step; the final point and
    momentum are retained for the proof. The zero-length mean is the input point. -/
def evaluate (I : Instance n m d) (L a : ℝ) (g : E d → σ → E d) :
    (r : ℕ) → Draws σ r → E d → E d → Trace d
  | 0, _, x, z => ⟨x,z,x⟩
  | r+1, s, x, z =>
    let y := blend a x z
    let p := step I L y (g y (drawHead s))
    let t := evaluate I L a g r (drawTail s) p (momentum a y p z)
    ⟨t.x,t.z,blend (fraction r) x t.mean⟩

omit [Fintype σ] [Nonempty σ] in
theorem evaluate_feasible (hI : I.InClass L μ Δ) (hL : 0 < L)
    (a : ℝ) (g : E d → σ → E d) (r : ℕ) (s : Draws σ r) {x : E d}
    (hx : feasible I x) (z : E d) :
    feasible I (evaluate I L a g r s x z).x ∧ feasible I (evaluate I L a g r s x z).mean := by
  induction r generalizing x z with
  | zero => exact ⟨hx,hx⟩
  | succ r ih =>
    let y := blend a x z
    let p := step I L y (g y (drawHead s))
    have h := ih (drawTail s) (step_feasible hI hL y (g y (drawHead s))) (momentum a y p z)
    exact ⟨h.1, (phi_blend hI hx h.2 (fraction_nonneg _) (fraction_le _)).1⟩

theorem averaged_loss (hI : I.InClass L μ Δ) (hμ : 0 ≤ μ)
    {u x v : E d} (hx : feasible I x) (hv : feasible I v) (r : ℕ) :
    ((r : ℝ)+1) * loss I u (blend (fraction r) x v) ≤ loss I u x + (r : ℝ) * loss I u v := by
  have h := mul_le_mul_of_nonneg_left
    (loss_blend hI hμ hx hv (u := u) (fraction_nonneg r) (fraction_le r))
    (show 0 ≤ (r : ℝ)+1 by positivity)
  have he : ((r : ℝ) + 1) * ((1 - fraction r) * loss I u x + fraction r * loss I u v) =
      loss I u x + (r : ℝ) * loss I u v := by
    unfold fraction
    field_simp
    ring
  rwa [he] at h

private theorem dist_sq_bound (hI : I.InClass L μ Δ) (hμ : 0 < μ) {u b x : E d}
    (hu : ∀ w, I.φ u ≤ I.φ w) (hb : feasible I b) (hx : feasible I x) (z : E d)
    {a : ℝ} (ha : 0 ≤ a) (ha₁ : a ≤ 1) :
    ‖blend a x z - b‖ ^ 2 ≤ (8/μ) * (potential I μ u x z + loss I u b) := by
  let y := blend a x z
  let P := potential I μ u x z
  have hpos : 0 < μ/4 := by positivity
  have hx₂ : ‖x-u‖ ^ 2 ≤ P / (μ/4) :=
    (le_div_iff₀ hpos).mpr (by
      unfold P potential
      linarith [radius_bound hI hu hx, show 0 ≤ μ/4 * ‖z-u‖ ^ 2 by positivity])
  have hz₂ : ‖z-u‖ ^ 2 ≤ P / (μ/4) :=
    (le_div_iff₀ hpos).mpr (by unfold P potential; linarith [loss_nonneg hI hu hx])
  have hb₂ : ‖b-u‖ ^ 2 ≤ loss I u b / (μ/4) :=
    (le_div_iff₀ hpos).mpr (by linarith [radius_bound hI hu hb])
  have hy : ‖y-u‖ ^ 2 ≤ P / (μ/4) := by
    have H := blend_norm_sq ha ha₁ x z u
    change ‖y-u‖ ^ 2 ≤ _ at H
    have hx' := mul_le_mul_of_nonneg_left hx₂ (sub_nonneg.mpr ha₁)
    have hz' := mul_le_mul_of_nonneg_left hz₂ ha
    linarith
  have ht : ‖y-b‖ ≤ ‖y-u‖+‖b-u‖ := by
    simpa only [show (y-u) - (b-u) = y-b by abel] using norm_sub_le (y-u) (b-u)
  have ht' := pow_le_pow_left₀ (norm_nonneg _) ht 2
  change ‖y-b‖ ^ 2 ≤ _
  calc
    _ ≤ 2 * ‖y-u‖ ^ 2 + 2 * ‖b-u‖ ^ 2 := by
      nlinarith [sq_nonneg (‖y-u‖-‖b-u‖)]
    _ ≤ 2 * (P / (μ/4)) + 2 * (loss I u b / (μ/4)) := by linarith
    _ = _ := by ring

section

variable [Nonempty (Fin n)] [Nonempty (Fin m)]

theorem mean_step (hI : I.InClass L μ Δ) (hL : 0 < L) (hμ : 0 < μ)
    {u b x : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hb : feasible I b) (hx : feasible I x)
    (z : E d) {a C : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : 2*L*a^2 = μ/4)
    (hC : 0 ≤ C) (g : E d → σ → E d)
    (hmean : ∀ y, average (fun s => g y s - gradient I.f y) = 0)
    (hvar : ∀ y, average (fun s => ‖g y s - gradient I.f y‖^2) ≤ C * ‖y-b‖^2) :
    let y := blend a x z
    let χ := 2*C/(L*μ)
    average (fun s =>
      let p := step I L y (g y s)
      potential I μ u p (momentum a y p z) + loss I u b)
        + a * loss I u x ≤ (1+χ) * (potential I μ u x z + loss I u b) := by
  intro y χ
  let G := loss I u b
  let P := potential I μ u x z
  let ee : σ → E d := fun s => g y s - gradient I.f y
  let pp : σ → E d := fun s => step I L y (g y s)
  have H (s : σ) :
      (potential I μ u (pp s) (momentum a y (pp s) z) + G) + a * loss I u x ≤
        (P+G) - a * ⟪ee s,z-u⟫ + (4*L)⁻¹ * ‖ee s‖ ^ 2 := by
    have h := potential_step hI hL hμ.le ha ha₁ ha₂ (optimal_feasible hI hu) hx z (ee s)
    change potential I μ u (step I L y (gradient I.f y + ee s))
      (momentum a y (step I L y (gradient I.f y+ee s)) z) + _ ≤ _ at h
    rw [show gradient I.f y + ee s = g y s by dsimp [ee]; abel] at h
    simpa only [pp, G, P, div_eq_inv_mul] using (by linarith :
      (potential I μ u (pp s) (momentum a y (pp s) z) + G) + a * loss I u x ≤
        (P+G) - a * ⟪ee s,z-u⟫ + ‖ee s‖ ^ 2 / (4*L))
  have HH := average_mono H
  have hm : average ee = 0 := hmean y
  simp only [average_add, average_sub, average_mul, average_const, average_inner_left, hm,
    inner_zero_left, mul_zero, sub_zero] at HH
  change average (fun s => potential I μ u (pp s) (momentum a y (pp s) z) + G) +
    a * loss I u x ≤ _
  rw [average_add, average_const]
  refine HH.trans ?_
  have hV := (hvar y).trans (mul_le_mul_of_nonneg_left
    (dist_sq_bound hI hμ hu hb hx z ha.le ha₁) hC)
  have hmul := mul_le_mul_of_nonneg_left hV (show 0 ≤ (4*L)⁻¹ by positivity)
  have hcoef : (4*L)⁻¹ * (C * ((8/μ) * (P+G))) = χ*(P+G) := by
    dsimp [χ]
    field_simp
    ring
  rw [hcoef] at hmul
  change P+G + (4*L)⁻¹ * average (fun s => ‖ee s‖ ^ 2) ≤ (1+χ)*(P+G)
  linarith

theorem mean_energy (hI : I.InClass L μ Δ) (hL : 0 < L) (hμ : 0 < μ)
    {u b : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hb : feasible I b)
    {a C : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : 2*L*a^2 = μ/4)
    (hC : 0 ≤ C) (g : E d → σ → E d)
    (hmean : ∀ y, average (fun s => g y s - gradient I.f y) = 0)
    (hvar : ∀ y, average (fun s => ‖g y s - gradient I.f y‖^2) ≤ C * ‖y-b‖^2)
    (r : ℕ) {x : E d} (hx : feasible I x) (z : E d) :
    average (fun s : Draws σ r =>
      let t := evaluate I L a g r s x z
      potential I μ u t.x t.z + loss I u b + a * (r : ℝ) * loss I u t.mean)
        ≤ (1+2*C/(L*μ))^r * (potential I μ u x z + loss I u b) := by
  let χ := 2*C/(L*μ)
  let G := loss I u b
  have hχ : 0 ≤ χ := by dsimp [χ]; positivity
  induction r generalizing x z with
  | zero => simpa only [evaluate, Nat.cast_zero, mul_zero, zero_mul, add_zero, pow_zero,
      one_mul, average_const] using (le_refl (potential I μ u x z + G))
  | succ r ih =>
    let y := blend a x z
    let pp : σ → E d := fun s => step I L y (g y s)
    let zz : σ → E d := fun s => momentum a y (pp s) z
    have hfac : 1 ≤ (1+χ)^r := one_le_pow₀ (by linarith)
    have hpoint (s : σ) : feasible I (pp s) := step_feasible hI hL _ _
    rw [average_draws_succ]
    calc
      _ ≤ average (fun s : σ =>
          (1+χ)^r * (potential I μ u (pp s) (zz s)+G) + a * loss I u x) := by
        apply average_mono
        intro s
        have hfirst (t : Draws σ r) :=
          averaged_loss hI hμ.le (u := u) hx
            (evaluate_feasible hI hL a g r t (hpoint s) (zz s)).2 r
        have hf := average_mono (fun t : Draws σ r => by
          dsimp only [evaluate, drawHead, drawTail]
          rw [Nat.cast_succ]
          have H := mul_le_mul_of_nonneg_left (hfirst t) ha.le
          linarith : ∀ t : Draws σ r,
            let e := evaluate I L a g (r+1) (show Draws σ (r+1) from (s,t)) x z
            potential I μ u e.x e.z + G + a * ((r+1 : ℕ) : ℝ) * loss I u e.mean ≤
              let e := evaluate I L a g r t (pp s) (zz s)
              (potential I μ u e.x e.z + G + a * (r : ℝ) * loss I u e.mean) + a * loss I u x)
        refine hf.trans ?_
        rw [average_add, average_const]
        exact add_le_add_left (ih (hpoint s) (zz s)) _
      _ ≤ (1+χ)^r * (average (fun s => potential I μ u (pp s) (zz s)+G) + a * loss I u x) := by
        rw [average_add, average_mul, average_const, mul_add]
        have hg : 0 ≤ a * loss I u x := mul_nonneg ha.le (loss_nonneg hI hu hx)
        exact add_le_add_right (le_mul_of_one_le_left hg hfac) _
      _ ≤ (1+χ)^r * ((1+χ) * (potential I μ u x z+G)) :=
        mul_le_mul_of_nonneg_left (mean_step hI hL hμ hu hb hx z ha ha₁ ha₂ hC g hmean hvar)
          (by positivity)
      _ = _ := by rw [pow_succ, mul_assoc]

theorem growth_bound {χ : ℝ} (hχ : 0 ≤ χ) (r : ℕ) (hr : χ * (r : ℝ) ≤ 1/4) :
    (1 + χ)^r ≤ 4/3 := by
  have hc (k : ℕ) : (1+χ)^k * (1-χ*(k:ℝ)) ≤ 1 := by
    induction k with
    | zero => simp
    | succ k ih =>
      rw [pow_succ, mul_assoc]
      refine (mul_le_mul_of_nonneg_left ?_ (by positivity : 0 ≤ (1+χ)^k)).trans ih
      have hh : 0 ≤ ((k : ℝ)+1) * χ^2 := by positivity
      push_cast
      nlinarith
  have h := mul_le_mul_of_nonneg_left (show (3/4 : ℝ) ≤ 1-χ*(r:ℝ) by linarith)
    (show 0 ≤ (1+χ)^r by positivity)
  linarith [h.trans (hc r)]

theorem contract (hI : I.InClass L μ Δ) (hL : 0 < L) (hμ : 0 < μ)
    {u b : E d} (hu : ∀ w, I.φ u ≤ I.φ w) (hb : feasible I b)
    {a C : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1) (ha₂ : 2*L*a^2 = μ/4)
    (hC : 0 ≤ C) (g : E d → σ → E d)
    (hmean : ∀ y, average (fun s => g y s - gradient I.f y) = 0)
    (hvar : ∀ y, average (fun s => ‖g y s - gradient I.f y‖^2) ≤ C * ‖y-b‖^2)
    (r : ℕ) (hr : 6 ≤ a*(r:ℝ)) (hnoise : (2*C/(L*μ)) * (r:ℝ) ≤ 1/4) :
    average (fun s : Draws σ r => loss I u (evaluate I L a g r s b b).mean) ≤
      loss I u b / 2 := by
  let G := loss I u b
  have hG : 0 ≤ G := loss_nonneg hI hu hb
  have hp : potential I μ u b b + G ≤ 3*G := by
    have h := radius_bound hI hu hb
    unfold potential
    linarith
  have he := mean_energy hI hL hμ hu hb ha ha₁ ha₂ hC g hmean hvar r hb b
  have hbound : (1+2*C/(L*μ))^r * (potential I μ u b b + G) ≤ 4*G := by
    calc
      _ ≤ (1+2*C/(L*μ))^r * (3*G) :=
        mul_le_mul_of_nonneg_left hp (by positivity)
      _ ≤ (4/3) * (3*G) :=
        mul_le_mul_of_nonneg_right (growth_bound (by positivity) r hnoise) (by positivity)
      _ = _ := by ring
  have hlow : G ≤ average (fun s : Draws σ r =>
      potential I μ u (evaluate I L a g r s b b).x (evaluate I L a g r s b b).z+G) := by
    calc
      G = average (fun _ : Draws σ r => G) := (average_const _).symm
      _ ≤ _ := average_mono ?_
    intro s
    have hh := loss_nonneg hI hu (evaluate_feasible hI hL a g r s hb b).1
    unfold potential
    have hrad : 0 ≤ μ / 4 * ‖(evaluate I L a g r s b b).z-u‖^2 := by positivity
    linarith
  have h := he.trans hbound
  simp only [average_add, average_mul] at h
  have hsum : a*(r:ℝ) * average (fun s : Draws σ r => loss I u (evaluate I L a g r s b b).mean)
      ≤ 3*G := by
    rw [average_add, average_const] at hlow
    simp only [average_const] at h
    linarith
  by_contra! H
  have hh := mul_lt_mul_of_pos_left H (show 0 < a * (r:ℝ) by linarith)
  have hh' := mul_le_mul_of_nonneg_right hr (show 0 ≤ G/2 by positivity)
  linarith

end

def repeatEval (f : E d → σ → E d) : (k : ℕ) → Draws σ k → E d → E d
  | 0, _, x => x
  | k+1, t, x => repeatEval f k (drawTail t) (f x (drawHead t))

omit [Fintype σ] [Nonempty σ] in
theorem repeat_feasible (f : E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ s, feasible I (f x s))
    (k : ℕ) (s : Draws σ k) {x : E d} (hx : feasible I x) :
    feasible I (repeatEval f k s x) := by
  induction k generalizing x with
  | zero => exact hx
  | succ k ih => exact ih (drawTail s) (hf x hx (drawHead s))

theorem repeat_contract (f : E d → σ → E d)
    (hf : ∀ x, feasible I x → ∀ s, feasible I (f x s))
    (h : ∀ x, feasible I x → average (fun s => loss I u (f x s)) ≤ loss I u x / 2)
    (k : ℕ) {x : E d} (hx : feasible I x) :
    average (fun s : Draws σ k => loss I u (repeatEval f k s x)) ≤ (1/2:ℝ)^k * loss I u x := by
  induction k generalizing x with
  | zero => simp [repeatEval, average_const]
  | succ k ih =>
    rw [average_draws_succ]
    calc
      _ ≤ average (fun s : σ => (1/2:ℝ)^k * loss I u (f x s)) :=
        average_mono (fun s => ih (hf x hx s))
      _ ≤ (1/2:ℝ)^k * (loss I u x / 2) := by
        rw [average_mul]
        exact mul_le_mul_of_nonneg_left (h x hx) (by positivity)
      _ = _ := by rw [pow_succ]; ring

end Epoch
end ULB63
