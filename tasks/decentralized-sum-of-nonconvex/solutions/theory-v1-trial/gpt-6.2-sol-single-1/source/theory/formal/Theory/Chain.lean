import Theory.LowerTools

open MeasureTheory
open scoped ENNReal RealInnerProductSpace

set_option maxHeartbeats 900000

noncomputable section

namespace ULB63
namespace Chain

open Protocol Lower

variable {d n m c : ℕ} {L μ Δ ρ ε q : ℝ}

def prev (i : Fin d) : Fin d := ⟨i.val-1, (Nat.sub_le _ _).trans_lt i.isLt⟩

def lag (d : ℕ) : E d →L[ℝ] E d :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 fun i => if i.val = 0 then 0 else x (prev i)
      map_add' := fun x y => by
        ext i
        by_cases h : i.val = 0 <;> simp [h]
      map_smul' := fun a x => by
        ext i
        by_cases h : i.val = 0 <;> simp [h] }

@[simp] theorem lag_apply (x : E d) (i : Fin d) :
    lag d x i = if i.val = 0 then 0 else x (prev i) := rfl

def weight (q : ℝ) (i : Fin d) : ℝ := if i.val = 0 then Real.sqrt (1-q^2) else 1

def diagonal (d : ℕ) (q : ℝ) : E d →L[ℝ] E d :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => WithLp.toLp 2 fun i => weight q i * x i
      map_add' := fun x y => by ext i; simp [mul_add]
      map_smul' := fun a x => by ext i; simp; ring }

@[simp] theorem diagonal_apply (x : E d) (i : Fin d) :
    diagonal d q x i = weight q i * x i := rfl

def residual (d : ℕ) (q : ℝ) : E d →L[ℝ] E d := diagonal d q - q • lag d

theorem residual_apply (x : E d) (i : Fin d) :
    residual d q x i = weight q i * x i - q * (lag d x) i := rfl

theorem lag_sq (x : E d) : ‖lag d x‖^2 ≤ ‖x‖^2 := by
  classical
  let s := (Finset.univ : Finset (Fin d)).filter (fun i => i.val ≠ 0)
  have hinj : Set.InjOn prev (s : Set (Fin d)) := by
    intro a ha b hb he
    have ha' : a.val ≠ 0 := (Finset.mem_filter.mp ha).2
    have hb' : b.val ≠ 0 := (Finset.mem_filter.mp hb).2
    apply Fin.ext
    have hh := congrArg Fin.val he
    dsimp [prev] at hh
    omega
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  calc
    _ = ∑ i ∈ s, (x (prev i))^2 := by
      unfold s
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro i _
      by_cases h : i.val = 0 <;> simp [h]
    _ = ∑ i ∈ s.image prev, (x i)^2 := by rw [Finset.sum_image hinj]
    _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun i _ _ => sq_nonneg _)

private theorem weight_sqrt_sq (hq : 0 ≤ q) (hq₁ : q ≤ 1) :
    (Real.sqrt (1-q^2))^2 = 1-q^2 := by
  apply Real.sq_sqrt
  nlinarith [mul_nonneg hq (sub_nonneg.mpr hq₁)]

private theorem diag_le (hq : 0 ≤ q) (hq₁ : q ≤ 1) (x : E d) : ‖diagonal d q x‖ ≤ ‖x‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  apply Finset.sum_le_sum
  intro i _
  rw [diagonal_apply, mul_pow]
  apply mul_le_of_le_one_left (sq_nonneg _)
  unfold weight
  split
  · rw [weight_sqrt_sq hq hq₁]
    nlinarith
  · norm_num

theorem residual_le (hq : 0 ≤ q) (hq₁ : q ≤ 1) (x : E d) :
    ‖residual d q x‖ ≤ (1+q) * ‖x‖ := by
  change ‖diagonal d q x - q • lag d x‖ ≤ _
  refine (norm_sub_le _ _).trans ?_
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hq, add_mul, one_mul]
  exact add_le_add (diag_le hq hq₁ x) (mul_le_mul_of_nonneg_left
    ((sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp (lag_sq x)) hq)

theorem residual_ge (hq : 0 ≤ q) (hq₁ : q ≤ 1) (x : E d) :
    (1-q)^2 * ‖x‖^2 ≤ ‖residual d q x‖^2 := by
  have hpoint (i : Fin d) :
      (1-q)*(x i)^2 - q*(1-q)*(lag d x i)^2 ≤ (residual d q x i)^2 := by
    rw [residual_apply, weight, lag_apply]
    split
    · rw [mul_zero, sub_zero, mul_pow, weight_sqrt_sq hq hq₁]
      have H : 0 ≤ q*(1-q)*(x i)^2 := by positivity
      linarith
    · simp only [one_mul]
      have H := mul_nonneg hq (sq_nonneg (x i-x (prev i)))
      nlinarith
  have h := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin d))) (fun i _ => hpoint i)
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    ← EuclideanSpace.real_norm_sq_eq, ← EuclideanSpace.real_norm_sq_eq,
    ← EuclideanSpace.real_norm_sq_eq] at h
  have hl := mul_le_mul_of_nonneg_left (lag_sq x) (mul_nonneg hq (sub_nonneg.mpr hq₁))
  nlinarith

def target (d : ℕ) (q s : ℝ) : E d := WithLp.toLp 2 fun i => s*q^i.val

@[simp] theorem target_apply (q s : ℝ) (i : Fin d) : target d q s i = s*q^i.val := rfl

theorem residual_target (i : Fin d) (h : i.val ≠ 0) (s : ℝ) :
    residual d q (target d q s) i = 0 := by
  rw [residual_apply, weight, lag_apply, ite_eq_right h, ite_eq_right h, one_mul,
    target_apply, target_apply]
  have he : i.val = (prev i).val+1 := by dsimp [prev]; omega
  rw [he, pow_succ]
  ring

def F (d : ℕ) (A q s : ℝ) (x : E d) : ℝ :=
  A/2 * ‖residual d q (x - target d q s)‖^2

private theorem derivative_F (A q s : ℝ) (x : E d) :
    HasFDerivAt (F d A q s)
      ((A/2) • (2 : ℝ) •
        (innerSL ℝ (residual d q (x-target d q s))).comp (residual d q)) x := by
  let B : E d →L[ℝ] E d := residual d q
  let u := target d q s
  have hD := ((B.hasFDerivAt (x := x)).sub_const (B u)).norm_sq.const_mul (A/2)
  change HasFDerivAt (fun z => A/2 * ‖B (z-u)‖^2) ((A/2) • (2 : ℝ) •
    (innerSL ℝ (B (x-u))).comp B) x
  simpa only [map_sub, ← Nat.cast_smul_eq_nsmul ℝ, Nat.cast_ofNat] using hD

theorem gradient_F (A q s : ℝ) (x : E d) :
    gradient (F d A q s) x = A • (residual d q).adjoint (residual d q (x - target d q s)) := by
  let B : E d →L[ℝ] E d := residual d q
  let u := target d q s
  apply ext_inner_right ℝ
  intro y
  rw [inner_gradient_left, (derivative_F A q s x).fderiv, real_inner_smul_left,
    ContinuousLinearMap.adjoint_inner_left]
  change (A/2) * (2 * ⟪B (x-u), B y⟫) = A * ⟪B (x-u), B y⟫
  ring

theorem smooth_F (hL : 0 ≤ L) (hq : 0 ≤ q) (hq₁ : q ≤ 1) (s : ℝ) :
    LSmooth L (F d (L/4) q s) := by
  constructor
  · exact fun x => (derivative_F _ _ _ x).differentiableAt
  · intro x y
    let B : E d →L[ℝ] E d := residual d q
    have hB : ‖B‖ ≤ 1+q := B.opNorm_le_bound (by positivity) (residual_le hq hq₁)
    have he : B (x-target d q s) - B (y-target d q s) = B (x-y) := by
      rw [← map_sub]; congr 1; abel
    rw [gradient_F, gradient_F]
    change ‖(L/4) • B.adjoint (B _) - (L/4) • B.adjoint (B _)‖ ≤ _
    rw [← smul_sub, ← map_sub, he, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (by positivity : 0 ≤ L/4)]
    have hBa : ‖B.adjoint‖ ≤ 2 := by
      rw [ContinuousLinearMap.adjoint.norm_map]
      linarith
    have hb : ‖B (x-y)‖ ≤ 2*‖x-y‖ := (residual_le hq hq₁ (x-y)).trans
      (mul_le_mul_of_nonneg_right (by linarith) (norm_nonneg _))
    calc
      _ ≤ (L/4) * (‖B.adjoint‖ * ‖B (x-y)‖) :=
        mul_le_mul_of_nonneg_left (B.adjoint.le_opNorm _) (by positivity)
      _ ≤ (L/4) * (2*(2*‖x-y‖)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul hBa hb (norm_nonneg _) (by norm_num)) (by positivity)
      _ = _ := by ring

theorem normSq_bary (t : ℝ) (ht : 0 ≤ t) (ht₁ : t ≤ 1) (x y : E d) :
    ‖t • x + (1-t) • y‖^2 + t*(1-t)*‖x-y‖^2 =
      t*‖x‖^2+(1-t)*‖y‖^2 := by
  rw [norm_add_sq_real, norm_sub_sq_real, norm_smul, norm_smul, Real.norm_eq_abs,
    Real.norm_eq_abs, abs_of_nonneg ht, abs_of_nonneg (sub_nonneg.mpr ht₁),
    real_inner_smul_left, real_inner_smul_right]
  ring

theorem strong_F (hL : 0 ≤ L) (hq : 0 ≤ q) (hq₁ : q ≤ 1) (hμ : μ ≤ (L/4)*(1-q)^2) (s : ℝ) :
    StronglyConvexE μ (fun x : E d => ((F d (L/4) q s x : ℝ) : EReal)) := by
  intro x y t ht ht₁
  let B : E d →L[ℝ] E d := residual d q
  let u := target d q s
  let v := B (x-u)
  let w := B (y-u)
  have ht' : B (t • x + (1-t) • y - u) = t • v + (1-t) • w := by
    have he : t • x + (1-t) • y - u = t • (x-u) + (1-t) • (y-u) := by module
    rw [he, map_add, map_smul, map_smul]
  have hw : B (x-y) = v-w := by
    change B (x-y) = B (x-u) - B (y-u)
    rw [← map_sub]; congr 1; abel
  have hrad : μ * ‖x-y‖^2 ≤ (L/4) * ‖B (x-y)‖^2 := by
    calc
      _ ≤ ((L/4)*(1-q)^2) * ‖x-y‖^2 := mul_le_mul_of_nonneg_right hμ (sq_nonneg _)
      _ = (L/4)*((1-q)^2 * ‖x-y‖^2) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (residual_ge hq hq₁ (x-y)) (by positivity)
  rw [hw] at hrad
  have hrad' := mul_le_mul_of_nonneg_left hrad (show 0 ≤ t*(1-t)/2 by positivity)
  have H := normSq_bary t ht ht₁ v w
  have H' := congrArg (fun r : ℝ => (L/4/2)*r) H
  change ((L/4/2 * ‖B (t • x+(1-t) • y-u)‖^2 : ℝ):EReal) + _ ≤
    (t:EReal)*((L/4/2*‖v‖^2:ℝ):EReal) + ((1-t:ℝ):EReal)*((L/4/2*‖w‖^2:ℝ):EReal)
  rw [ht', ← EReal.coe_mul, ← EReal.coe_mul, ← EReal.coe_add, ← EReal.coe_add, EReal.coe_le_coe_iff]
  nlinarith

theorem residual_target_zero_norm (hd : 0 < d) (hq : 0 ≤ q) (hq₁ : q ≤ 1) (s : ℝ) :
    ‖residual d q (target d q s)‖^2 = (1-q^2)*s^2 := by
  let j : Fin d := ⟨0,hd⟩
  rw [EuclideanSpace.real_norm_sq_eq, Finset.sum_eq_single j]
  · rw [residual_apply, weight, lag_apply]
    simp only [show j.val = 0 from rfl, ite_true, target_apply, pow_zero, mul_one, mul_zero, sub_zero,
      mul_pow, weight_sqrt_sq hq hq₁]
  · intro i _ hi
    have h : i.val ≠ 0 := fun hz => hi (Fin.ext hz)
    rw [residual_target i h, zero_pow (by decide : (2:ℕ) ≠ 0)]
  · simp

def amplitude (L q Δ : ℝ) : ℝ := Real.sqrt (2*Δ/((L/4)*(1-q^2)))

theorem amplitude_eq (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1) (hΔ : 0 ≤ Δ) :
    L/4/2 * (1-q^2) * (amplitude L q Δ)^2 = Δ := by
  have hh := mul_pos (sub_pos.mpr hq₁) (show 0 < 1+q by linarith)
  have hw : 0 < 1-q^2 := by nlinarith
  unfold amplitude
  rw [Real.sq_sqrt (by positivity)]
  field_simp

def hard (n m d : ℕ) (L q Δ : ℝ) : Instance n m d where
  F _ _ := F d (L/4) q (amplitude L q Δ)
  h _ := (0 : ℝ)
  x0 := 0
  prox v _ := v

theorem f_eq (hn : 0 < n) (hm : 0 < m) (x : E d) :
    (hard n m d L q Δ).f x = F d (L/4) q (amplitude L q Δ) x := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  simp [Instance.f, hard, Finset.sum_const, nsmul_eq_mul, hn', hm']

theorem phi_eq (hn : 0 < n) (hm : 0 < m) (x : E d) :
    (hard n m d L q Δ).φ x = (F d (L/4) q (amplitude L q Δ) x : EReal) := by
  rw [Instance.φ, f_eq hn hm]
  change _ + ((0 : ℝ) : EReal) = _
  simp

theorem star_zero (hn : 0 < n) (hm : 0 < m) (hL : 0 ≤ L) :
    (hard n m d L q Δ).φstar = 0 := by
  apply le_antisymm
  · apply (iInf_le _ (target d q (amplitude L q Δ))).trans
    simp [phi_eq hn hm, F]
  · apply le_iInf
    intro y
    rw [phi_eq hn hm]
    exact_mod_cast (show 0 ≤ F d (L/4) q (amplitude L q Δ) y by unfold F; positivity)

theorem inClass (hn : 0 < n) (hm : 0 < m) (hd : 0 < d) (hL : 0 < L)
    (hμ : 0 ≤ μ) (hq : 0 ≤ q) (hq₁ : q < 1) (hμq : μ ≤ (L/4)*(1-q)^2) (hΔ : 0 ≤ Δ) :
    (hard n m d L q Δ).InClass L μ Δ where
  smooth _ _ := smooth_F hL.le hq hq₁.le _
  f_convex := by
    have H : (hard n m d L q Δ).f = F d (L/4) q (amplitude L q Δ) := funext (f_eq hn hm)
    rw [H]
    exact UniformConvexOn.convexOn ((stronglyConvexE_coe_iff _ _).1
      (strong_F hL.le hq hq₁.le hμq _)) (fun r => by positivity)
  h_ne_bot _ := EReal.coe_ne_bot _
  h_proper := ⟨0, EReal.coe_ne_top _⟩
  h_closed := continuous_const.lowerSemicontinuous
  h_convex := (stronglyConvexE_coe_iff 0 _).2 ((strongConvexOn_zero).2 (convexOn_const 0 convex_univ))
  prox_exact v η hη := by
    intro y
    have H : (0:ℝ) ≤ ‖y-v‖^2/(2*η) := by positivity
    change ((0:ℝ):EReal) + ((‖v-v‖^2/(2*η):ℝ):EReal) ≤
      ((0:ℝ):EReal) + ((‖y-v‖^2/(2*η):ℝ):EReal)
    simpa only [EReal.coe_zero, sub_self, norm_zero, zero_pow (by decide : (2:ℕ) ≠ 0),
      zero_div, zero_add] using EReal.coe_le_coe_iff.mpr H
  φ_strongly_convex := by
    rw [show (hard n m d L q Δ).φ = fun x =>
      (F d (L/4) q (amplitude L q Δ) x : EReal) from funext (phi_eq hn hm)]
    exact strong_F hL.le hq hq₁.le hμq _
  minimizer := by
    refine ⟨target d q (amplitude L q Δ), fun x => ?_⟩
    rw [phi_eq hn hm, phi_eq hn hm, EReal.coe_le_coe_iff]
    unfold F
    simp
    positivity
  initial_gap := by
    rw [star_zero hn hm hL.le, sub_zero]
    change (hard n m d L q Δ).φ 0 ≤ _
    rw [phi_eq hn hm, EReal.coe_le_coe_iff, F, zero_sub, map_neg, norm_neg,
      residual_target_zero_norm hd hq hq₁.le, ← mul_assoc, amplitude_eq hL hq hq₁ hΔ]

private def front (d s : ℕ) : Set (Fin d) := {j | j.val < s}

theorem gradient_progress (A s : ℝ) (x : E d) (t : ℕ) (hx : Supported (front d t) x) :
    Supported (front d (t+1)) (gradient (F d A q s) x) := by
  rw [supported_iff] at hx ⊢
  intro k hk
  have hkt : t+1 ≤ k.val := Nat.not_lt.mp hk
  let e : E d := EuclideanSpace.single k 1
  let B := residual d q
  let v := B (x-target d q s)
  have hv (j : Fin d) (hj : k.val ≤ j.val) : v j = 0 := by
    have hj₀ : j.val ≠ 0 := by omega
    have hxj : x j = 0 := hx j (by change ¬j.val < t; omega)
    have hxp : x (prev j) = 0 := hx _ (by change ¬j.val-1 < t; omega)
    change residual d q (x-target d q s) j = _
    rw [map_sub, PiLp.sub_apply, residual_target j hj₀, sub_zero, residual_apply,
      hxj, lag_apply, ite_eq_right hj₀, hxp, mul_zero, mul_zero, sub_zero]
  have he (j : Fin d) (hj : j.val < k.val) : B e j = 0 := by
    have hj₀ : j ≠ k := fun h => (ne_of_lt hj) (congrArg Fin.val h)
    have hjp : prev j ≠ k := fun h => by
      have hh := congrArg Fin.val h
      dsimp [prev] at hh
      omega
    change residual d q (EuclideanSpace.single k 1) j = _
    simp [residual_apply, lag_apply, hj₀, hjp]
  have hinner : ⟪v,B e⟫ = 0 := by
    rw [PiLp.inner_apply]
    apply Finset.sum_eq_zero
    intro j _
    rcases lt_or_ge j.val k.val with hh | hh
    · rw [he j hh]
      simp
    · rw [hv j hh]
      simp
  have hco : B.adjoint v k = 0 := by
    have hh : ⟪B.adjoint v,e⟫ = 0 := by rw [ContinuousLinearMap.adjoint_inner_left, hinner]
    simpa [e, EuclideanSpace.inner_single_right, RCLike.conj_to_real] using hh
  rw [gradient_F]
  change A * (B.adjoint v k) = _
  rw [hco, mul_zero]

private theorem energy_at (hq : 0 ≤ q) (hq₁ : q ≤ 1) (z : E d) (k : Fin d) :
    (1-q^2)*(z k)^2 ≤ ‖residual d q z‖^2 := by
  let below (a : ℕ) := (Finset.univ : Finset (Fin d)).filter fun i => i.val ≤ a
  have short (a : ℕ) (ha : a < d) :
      (1-q^2)*(z ⟨a,ha⟩)^2 ≤ ∑ i ∈ below a, (residual d q z i)^2 := by
    induction a with
    | zero =>
      let e : Fin d := ⟨0,ha⟩
      have hb : below 0 = {e} := by
        ext i
        simp [below, Fin.ext_iff, e]
      rw [hb, Finset.sum_singleton, residual_apply, weight, lag_apply]
      simp only [show e.val = 0 from rfl, ite_true, mul_zero, sub_zero, mul_pow, weight_sqrt_sq hq hq₁]
      rfl
    | succ a ih =>
      let e : Fin d := ⟨a+1,ha⟩
      have he : e.val ≠ 0 := by dsimp [e]; omega
      have hpred : (prev e).val = a := by simp [prev,e]
      have hb : below (a+1) = insert e (below a) := by
        ext i
        simp only [below, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
        change i.val ≤ a+1 ↔ i.val = a+1 ∨ i.val ≤ a
        omega
      have hnot : e ∉ below a := by simp [below,e]
      have hl : (1-q^2)*(z (prev e))^2 ≤ ∑ i ∈ below a, (residual d q z i)^2 := by
        have hy : prev e = ⟨a, by omega⟩ := Fin.ext hpred
        rw [hy]
        exact ih (by omega)
      have hstep : (1-q^2)*(z e)^2 ≤
          (1-q^2)*(z (prev e))^2 + (z e-q*z (prev e))^2 := by
        nlinarith [sq_nonneg ((1-q^2)*z (prev e)-q*(z e-q*z (prev e)))]
      rw [hb, Finset.sum_insert hnot, residual_apply, weight, lag_apply, ite_eq_right he,
        ite_eq_right he, one_mul]
      linarith
  refine (short k.val k.isLt).trans ?_
  rw [EuclideanSpace.real_norm_sq_eq]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) (fun i _ _ => sq_nonneg _)

theorem F_at_zero (hL : 0 < L) (hq : 0 ≤ q) (hq₁ : q < 1) (hΔ : 0 ≤ Δ) (x : E d)
    (k : Fin d) (hx : x k = 0) :
    Δ*q^(2*k.val) ≤ F d (L/4) q (amplitude L q Δ) x := by
  have H := mul_le_mul_of_nonneg_left (energy_at hq hq₁.le (x-target d q (amplitude L q Δ)) k)
    (show 0 ≤ L/4/2 by positivity)
  refine le_trans ?_ H
  rw [PiLp.sub_apply, hx, target_apply, zero_sub, neg_sq, mul_pow, Nat.mul_comm 2, pow_mul]
  apply le_of_eq
  linear_combination -(q ^ k.val)^2 * amplitude_eq hL hq hq₁ hΔ

theorem minimax_chain (hn : 2 ≤ n) (hm : 1 ≤ m) (hμ : 0 < μ) (hL : 0 < L)
    (hΔ : 0 < Δ) (hρ : 0 ≤ ρ) {q : ℝ} (hq : 0 ≤ q) (hq₁ : q < 1)
    (hcond : 4*μ ≤ L*(1-q)^2) (T : ℕ) (hε : 0 ≤ ε)
    (htol : ε < Δ * q^(2*(microSteps*T))) :
    ((T+1 : ℕ) : ℕ∞) ≤ minimax n m c L μ Δ ρ ε := by
  let d := microSteps*T+1
  let j : Fin d := ⟨microSteps*T,Nat.lt_succ_self _⟩
  let I := hard n m d L q Δ
  have hI := inClass (d := d) (by omega : 0 < n) (by omega : 0 < m)
    (Nat.zero_lt_succ _) hL hμ.le hq hq₁ (by linarith : μ ≤ L/4*(1-q)^2) hΔ.le
  apply minimax_ge
  intro A hA
  refine ⟨d,I,J n,hI,J_commClass (by omega) hρ, hitting_gt T ?_⟩
  intro t ht
  have hz (ω : A.Ω) : A.avgOutput I (J n) ω t j = 0 := by
    have H := allowed_invariant hA I (J n) ω (front d)
      (by change supp (0 : E d) ⊆ _; simp [supp])
      (fun _ _ hk => Nat.lt_succ_of_lt hk)
      (fun s i ξ x hx => gradient_progress _ _ x s hx)
      (fun s x η hx => hx.mono (fun k hk => Nat.lt_succ_of_lt hk))
    have hS := avg_from_allowed hA I (J n) ω t (H (microSteps*t))
    apply supported_iff.mp hS
    change ¬j.val < microSteps*t
    exact Nat.not_lt.mpr (Nat.mul_le_mul_left _ ht)
  have hpoint (ω : A.Ω) : ENNReal.ofReal (Δ*q^(2*(microSteps*T))) ≤
      I.gap (A.avgOutput I (J n) ω t) := by
    have hh := ENNReal.ofReal_le_ofReal (F_at_zero hL hq hq₁ hΔ.le _ j (hz ω))
    simpa [I, Instance.gap, star_zero (by omega : 0 < n) (by omega : 0 < m) hL.le,
      phi_eq (by omega : 0 < n) (by omega : 0 < m), j] using hh
  have he := lintegral_mono hpoint (μ := A.P)
  simp only [lintegral_const, measure_univ, mul_one] at he
  exact (ENNReal.ofReal_lt_ofReal_iff_of_nonneg hε |>.mpr htol).trans_le he

end Chain
end ULB63
