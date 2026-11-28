import Theory.BlockRun

/-! Even evaluation by an average cannot optimize through a disconnected network. -/

open Finset MeasureTheory
open scoped ENNReal RealInnerProductSpace

noncomputable section

namespace ULB63

theorem identity_commClass {n : ℕ} (hn : n ≠ 0) : CommClass n 1 1 where
  nonneg i j := by simp [Matrix.one_apply]; split <;> norm_num
  symm := by simp
  row_sum i := by simp [Matrix.one_apply]
  contraction v := by
    have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn
    let avg : ℝ := (1/n : ℝ) * ∑ j, v j
    have hJ (i : Fin n) : (J n).mulVec v i = avg := by
      simp [J, avg, Matrix.mulVec, dotProduct, mul_sum]
    have h (i : Fin n) : ((1 - J n).mulVec v) i = v i - avg := by
      rw [Matrix.sub_mulVec, Matrix.one_mulVec, Pi.sub_apply, hJ]
    have ha : (∑ i, v i) = (n : ℝ) * avg := by dsimp [avg]; field_simp
    have hsq : ∑ i : Fin n, (v i - avg)^2 = (∑ i : Fin n, (v i)^2) - (n : ℝ)*avg^2 := by
      simp_rw [sub_sq, sum_add_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum]
      rw [sum_const, nsmul_eq_mul, card_univ, Fintype.card_fin, ha]
      ring
    rw [one_mul, norm2, norm2]
    simp_rw [h]
    rw [hsq]
    exact Real.sqrt_le_sqrt (sub_le_self _ (by positivity))

namespace Disconnected

variable {n m c : ℕ}

/-- Effective coupling when one node supplies the forcing and all other nodes supply the edge. -/
def coupling (n : ℕ) (L : ℝ) : ℝ := ((n : ℝ)-1)*L/(2*n)

def ratio (l μ : ℝ) : ℝ := l/(μ+l)

def stiffness (l μ : ℝ) : ℝ := μ+l-l^2/(μ+l)

def amp (l μ Δ : ℝ) : ℝ := Real.sqrt (2*Δ/stiffness l μ)

private def lin (b : ℝ) (x : E 2) : ℝ := -b*x 0

private def edge (L : ℝ) (x : E 2) : ℝ := L/4*(x 1 - x 0)^2

private def P (j : Fin 2) : E 2 →L[ℝ] ℝ := EuclideanSpace.proj j

private def D : E 2 →L[ℝ] ℝ := P 1 - P 0

private theorem D_apply (x : E 2) : D x = x 1 - x 0 := rfl

private theorem norm_D_le : ‖D‖ ≤ Real.sqrt 2 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro x
  apply (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp
  rw [mul_pow, Real.sq_sqrt (by norm_num), EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two,
    D_apply, Real.norm_eq_abs, sq_abs]
  nlinarith only [sq_nonneg (x 0 + x 1)]

private theorem lin_gradient (b : ℝ) (x : E 2) :
    HasGradientAt (lin b) ((-b) • (P 0).adjoint 1) x := by
  rw [hasGradientAt_iff_hasFDerivAt, map_smul]
  exact (BlockGeometry.hasGradientAt_proj (P 0) x).hasFDerivAt.const_mul (-b)

private theorem lin_gradient_second (b : ℝ) (x : E 2) : gradient (lin b) x 1 = 0 := by
  rw [(lin_gradient b x).gradient]
  have h : ((P 0).adjoint (1 : ℝ)) (1 : Fin 2) = 0 := by
    have e := (P 0).adjoint_inner_left (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) (1 : ℝ)
    rw [EuclideanSpace.inner_single_right] at e
    have hz : P 0 (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 :=
      PiLp.single_eq_of_ne' (β := fun _ : Fin 2 => ℝ) 2 (by decide : (1 : Fin 2) ≠ 0) (1 : ℝ)
    simpa [hz] using e
  simp [h]

private theorem edge_gradient (L : ℝ) (x : E 2) :
    HasGradientAt (edge L) ((L/2) • D.adjoint (D x)) x := by
  rw [hasGradientAt_iff_hasFDerivAt, map_smul]
  convert (BlockGeometry.hasGradientAt_half_proj D x).hasFDerivAt.const_mul (L/2) using 1
  funext y
  simp only [edge, D_apply]
  ring

private theorem edge_gradient_zero (L : ℝ) : gradient (edge L) 0 = 0 := by
  rw [(edge_gradient L 0).gradient]
  simp

private theorem lin_smooth {L : ℝ} (hL : 0 ≤ L) (b : ℝ) : LSmooth L (lin b) := by
  refine ⟨fun x => (lin_gradient b x).differentiableAt, fun x y => ?_⟩
  rw [(lin_gradient b x).gradient, (lin_gradient b y).gradient, sub_self, norm_zero]
  positivity

private theorem edge_smooth {L : ℝ} (hL : 0 ≤ L) : LSmooth L (edge L) := by
  refine ⟨fun x => (edge_gradient L x).differentiableAt, fun x y => ?_⟩
  rw [(edge_gradient L x).gradient, (edge_gradient L y).gradient, ← smul_sub, ← map_sub, ← map_sub,
    norm_smul, Real.norm_of_nonneg (by positivity)]
  have hD (x : E 2) : ‖D x‖ ≤ Real.sqrt 2 * ‖x‖ :=
    (ContinuousLinearMap.le_opNorm D x).trans (mul_le_mul_of_nonneg_right norm_D_le (norm_nonneg _))
  have ha (v : ℝ) : ‖D.adjoint v‖ ≤ Real.sqrt 2 * ‖v‖ := by
    refine (ContinuousLinearMap.le_opNorm _ _).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg _))
    simpa using norm_D_le
  calc
    _ ≤ L/2*(Real.sqrt 2 * ‖D (x-y)‖) := mul_le_mul_of_nonneg_left (ha _) (by positivity)
    _ ≤ L/2*(Real.sqrt 2 * (Real.sqrt 2 * ‖x-y‖)) := by gcongr; exact hD _
    _ = _ := by rw [← mul_assoc (Real.sqrt 2), ← pow_two, Real.sq_sqrt (by norm_num)]; ring

def instanceOf (src : Fin n) (L μ Δ : ℝ) (m : ℕ) : Instance n m 2 where
  F i _ := if i = src then lin ((n : ℝ)*stiffness (coupling n L) μ * amp (coupling n L) μ Δ)
    else edge L
  h x := ((μ/2*‖x‖^2 : ℝ) : EReal)
  x0 := 0
  prox v η := (1+η*μ)⁻¹ • v

private theorem sum_ite_constant (src : Fin n) (u v : ℝ) :
    ∑ i : Fin n, (if i = src then u else v) = u + ((n : ℝ)-1)*v := by
  have he (i : Fin n) : (if i = src then u else v) = v + (if i = src then u-v else 0) := by
    split <;> simp_all
  simp only [he, sum_add_distrib, sum_const, nsmul_eq_mul, card_univ, Fintype.card_fin,
    sum_ite_eq', mem_univ, ite_true]
  ring

private theorem isolated_f {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (src : Fin n) (x : E 2) :
    (instanceOf src L μ Δ m).f x =
      coupling n L/2*(x 1-x 0)^2 - stiffness (coupling n L) μ * amp (coupling n L) μ Δ * x 0 := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hmR : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  have hm' (i : Fin n) :
      (1/m : ℝ) * (∑ ξ, (instanceOf src L μ Δ m).F i ξ x) =
        (if i=src then lin ((n : ℝ)*stiffness (coupling n L) μ * amp (coupling n L) μ Δ) x
          else edge L x) := by
    simp only [instanceOf, sum_const, nsmul_eq_mul, card_univ, Fintype.card_fin]
    field_simp
    split <;> rfl
  unfold Instance.f
  simp only [hm', sum_ite_constant, lin, edge]
  unfold coupling
  field_simp
  ring

private theorem coupling_nonneg {L : ℝ} (hn : 1 ≤ n) (hL : 0 ≤ L) :
    0 ≤ coupling n L := by
  have h : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  unfold coupling
  positivity [sub_nonneg.mpr h]

private theorem stiffness_pos {l μ : ℝ} (hl : 0 ≤ l) (hμ : 0 < μ) :
    0 < stiffness l μ := by
  have hs : 0 < μ+l := by positivity
  unfold stiffness
  apply sub_pos.mpr
  apply (div_lt_iff₀ hs).2
  nlinarith [sq_pos_of_pos hμ, mul_nonneg hμ.le hl]

private theorem initial_energy {l μ Δ : ℝ} (hl : 0 ≤ l) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) :
    stiffness l μ / 2 * (amp l μ Δ)^2 = Δ := by
  have ha := stiffness_pos hl hμ
  unfold amp
  rw [Real.sq_sqrt (by positivity)]
  field_simp

private def energy (l μ Δ : ℝ) (x : E 2) : ℝ :=
  (μ+l)/2 * (x 1-ratio l μ*x 0)^2 + stiffness l μ/2 * (x 0-amp l μ Δ)^2

private theorem isolated_value {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (src : Fin n) (x : E 2) :
    (instanceOf src L μ Δ m).φ x = ((energy (coupling n L) μ Δ x - Δ : ℝ) : EReal) := by
  rw [Instance.φ, isolated_f hn hm]
  change ((_:ℝ):EReal) + ((μ/2*‖x‖^2 : ℝ) : EReal) = _
  rw [← EReal.coe_add, EReal.coe_eq_coe_iff, EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
  have hl := coupling_nonneg (Nat.one_le_iff_ne_zero.mpr hn) hL
  have h₁ := initial_energy hl hμ hΔ
  let l := coupling n L
  change l/2*(x 1-x 0)^2-stiffness l μ * amp l μ Δ * x 0 + μ/2*(x 0 ^ 2+x 1 ^ 2) =
    energy l μ Δ x - Δ
  have hs : 0 < μ+l := by change 0 < μ+coupling n L; positivity
  have hcomp (a : ℝ) :
      l/2*(x 1-x 0)^2 - stiffness l μ*a*x 0 + μ/2*(x 0^2+x 1^2) + stiffness l μ/2*a^2 =
        (μ+l)/2*(x 1-ratio l μ*x 0)^2 + stiffness l μ/2*(x 0-a)^2 := by
    unfold stiffness ratio
    field_simp
    ring
  have hc := hcomp (amp l μ Δ)
  change stiffness l μ/2*(amp l μ Δ)^2 = Δ at h₁
  rw [h₁] at hc
  unfold energy
  linarith only [hc]

private theorem energy_nonneg {l μ Δ : ℝ} (hl : 0 ≤ l) (hμ : 0 < μ) (x : E 2) :
    0 ≤ energy l μ Δ x := by
  unfold energy
  positivity [stiffness_pos hl hμ]

private def minimizer (l μ Δ : ℝ) : E 2 :=
  WithLp.toLp 2 ![amp l μ Δ, ratio l μ * amp l μ Δ]

private theorem energy_minimizer (l μ Δ : ℝ) : energy l μ Δ (minimizer l μ Δ) = 0 := by
  simp [energy, minimizer]

private theorem isolated_min {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (src : Fin n) :
    ∀ x, (instanceOf src L μ Δ m).φ (minimizer (coupling n L) μ Δ) ≤ (instanceOf src L μ Δ m).φ x := by
  intro x
  rw [isolated_value hn hm hL hμ hΔ, isolated_value hn hm hL hμ hΔ, energy_minimizer,
    EReal.coe_le_coe_iff]
  linarith only [energy_nonneg (Δ := Δ) (coupling_nonneg (Nat.one_le_iff_ne_zero.mpr hn) hL) hμ x]

private theorem isolated_star {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (src : Fin n) :
    (instanceOf src L μ Δ m).φstar = ((-Δ : ℝ) : EReal) := by
  have h := isolated_min hn hm hL hμ hΔ src
  have he : (instanceOf src L μ Δ m).φstar = _ :=
    le_antisymm (iInf_le _ _) (le_iInf h)
  rw [he, isolated_value hn hm hL hμ hΔ, energy_minimizer, zero_sub]

private theorem isolated_gap {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (src : Fin n) (x : E 2) :
    (instanceOf src L μ Δ m).gap x = ENNReal.ofReal (energy (coupling n L) μ Δ x) := by
  rw [Instance.gap, isolated_value hn hm hL hμ hΔ, isolated_star hn hm hL hμ hΔ, ← EReal.coe_sub,
    EReal.real_coe_toENNReal]
  congr 1
  ring

private theorem isolated_class {L μ Δ : ℝ} (hn : n ≠ 0) (hm : m ≠ 0)
    (hL : 0 ≤ L) (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (src : Fin n) :
    (instanceOf src L μ Δ m).InClass L μ Δ := by
  let I := instanceOf src L μ Δ m
  let l := coupling n L
  have hh := zeroInstance_inClass (n := n) (m := m) (d := 2) hL hμ hΔ
  have hl := coupling_nonneg (Nat.one_le_iff_ne_zero.mpr hn) hL
  have hf : ConvexOn ℝ Set.univ I.f := by
    have he : I.f = fun x => l/2*(D x)^2 - (stiffness l μ*amp l μ Δ)*(P 0 x) :=
      funext (isolated_f hn hm src)
    rw [he]
    have hp := (Even.convexOn_pow (𝕜 := ℝ) (by decide : Even (2:ℕ))).comp_linearMap D.toLinearMap
    have hp' : ConvexOn ℝ Set.univ (fun x : E 2 => (D x)^2) := by simpa [Function.comp_def] using hp
    have hlin : ConcaveOn ℝ Set.univ (fun x : E 2 => (stiffness l μ*amp l μ Δ)*(P 0 x)) :=
      ((stiffness l μ*amp l μ Δ) • (P 0)).toLinearMap.concaveOn convex_univ
    exact (ConvexOn.smul (by dsimp [l]; positivity) hp').sub hlin
  refine ⟨?_, hf, hh.h_ne_bot, hh.h_proper, hh.h_closed, hh.h_convex, hh.prox_exact, ?_, ?_, ?_⟩
  · intro i ξ
    change LSmooth L (if i = src then _ else _)
    split
    · exact lin_smooth hL _
    · exact edge_smooth hL
  · have he : I.φ = fun x => ((I.f x + μ/2*‖x‖^2 : ℝ) : EReal) := by
      funext x
      rw [EReal.coe_add]
      rfl
    rw [he, stronglyConvexE_coe_iff, strongConvexOn_iff_convex]
    simpa using hf
  · exact ⟨minimizer l μ Δ, isolated_min hn hm hL hμ hΔ src⟩
  · change I.φ 0 - I.φstar ≤ _
    rw [isolated_value hn hm hL hμ hΔ, isolated_star hn hm hL hμ hΔ, ← EReal.coe_sub,
      EReal.coe_le_coe_iff]
    dsimp only [energy]
    simp [initial_energy hl hμ hΔ]

private theorem energy_missing {l μ Δ : ℝ} (hl : 0 ≤ l) (hμ : 0 < μ) (hΔ : 0 ≤ Δ)
    (x : E 2) (hx : x 1 = 0) : Δ * (ratio l μ)^2 ≤ energy l μ Δ x := by
  have hs : 0 < μ+l := by positivity
  have hh (a : ℝ) :
      (μ+l)/2 * (0-ratio l μ*x 0)^2 + stiffness l μ/2 * (x 0-a)^2 =
        (μ+l)/2 * (x 0 - (stiffness l μ/(μ+l))*a)^2 + (stiffness l μ / 2 * a^2) * (ratio l μ)^2 := by
    unfold ratio stiffness
    field_simp
    ring
  have he : energy l μ Δ x =
      (μ+l)/2 * (x 0 - (stiffness l μ/(μ+l))*amp l μ Δ)^2 + Δ * (ratio l μ)^2 := by
    unfold energy
    rw [hx, hh, initial_energy hl hμ hΔ]
  rw [he]
  linarith only [show 0 ≤ (μ+l)/2 * (x 0 - (stiffness l μ/(μ+l))*amp l μ Δ)^2 by positivity]

private def visible (src i : Fin n) : Set (Fin 2) :=
  if i = src then {0} else ∅

private theorem second_not_visible (src i : Fin n) : (1 : Fin 2) ∉ visible src i := by
  unfold visible
  split <;> decide

private theorem visible_of_allowed {Ω : Type} {info : Info n 2 Ω}
    {H : List (Obs 2 c)} {s : Set (Fin 2)} (hx : info.x0 = 0)
    (h : ∀ o ∈ H, o.supp ⊆ s) : allowed info H ⊆ s := by
  intro j hj
  simp only [allowed, Set.mem_union, Set.mem_iUnion, hx, supp, Set.mem_ofPred_eq, PiLp.zero_apply,
    ne_self_iff_false, false_or] at hj
  obtain ⟨o, ho, hz⟩ := hj
  exact h o ho hz

private theorem sum_identity_smul (i : Fin n) (v : Fin n → E 2) :
    ∑ j : Fin n, (1 : Matrix (Fin n) (Fin n) ℝ) i j • v j = v i := by
  classical
  rw [sum_eq_single i]
  · simp
  · intro j _ hj
    simp [Ne.symm hj]
  · simp

private theorem history_visible {L μ Δ : ℝ} (src : Fin n)
    {A : Algorithm n m c} (hA : A.Admissible L μ Δ 1) (ω : A.Ω) (s : ℕ) :
    ∀ i o, o ∈ A.hist (instanceOf src L μ Δ m) 1 ω s i → o.supp ⊆ visible src i := by
  let I := instanceOf src L μ Δ m
  change ∀ i o, o ∈ A.hist I 1 ω s i → o.supp ⊆ visible src i
  induction s with
  | zero => simp [Algorithm.hist]
  | succ s ih =>
    have hall (i : Fin n) : allowed (A.info I 1 ω i) (A.hist I 1 ω s i) ⊆ visible src i :=
      visible_of_allowed rfl (ih i)
    intro i o ho j hj
    rw [Algorithm.hist] at ho
    split at ho
    · simp only [List.mem_append, List.mem_singleton] at ho
      rcases ho with ho | rfl
      · exact ih i o ho hj
      · simp only [Obs.supp, Set.mem_iUnion] at hj
        obtain ⟨k, hk⟩ := hj
        rw [sum_identity_smul] at hk
        exact hall i (hA.send_zero_respecting 2 (A.info I 1 ω i) (A.hist I 1 ω s i) k hk)
    · simp only [List.mem_append, List.mem_singleton] at ho
      rcases ho with ho | rfl
      · exact ih i o ho hj
      · have hQ := hA.call_zero_respecting 2 (A.info I 1 ω i) (A.hist I 1 ω s i)
        have h := hQ.trans (hall i)
        cases he : A.call 2 (A.info I 1 ω i) (A.hist I 1 ω s i) with
        | idle => simp [he, Algorithm.answer, Obs.supp] at hj
        | prox v η =>
          rw [he] at h hj
          apply h
          change v j ≠ 0
          intro hv
          apply hj
          change (1+η*μ)⁻¹ * v j = 0
          rw [hv, mul_zero]
        | grad ξ v =>
          rw [he] at h hj
          change gradient (I.F i ξ) v j ≠ 0 at hj
          by_cases hh : i = src
          · change gradient (if i=src then _ else edge L) v j ≠ 0 at hj
            rw [ite_eq_left hh] at hj
            simp only [visible, hh, ite_true, Set.mem_singleton_iff]
            fin_cases j
            · rfl
            · exact (hj (lin_gradient_second _ _)).elim
          · change gradient (if i=src then _ else edge L) v j ≠ 0 at hj
            rw [ite_eq_right hh] at hj
            have hv : v = 0 := by
              ext k
              by_contra hk
              have hz := h (show k ∈ supp v from hk)
              simp [visible, hh] at hz
            rw [hv, edge_gradient_zero] at hj
            exact (hj rfl).elim

private theorem output_second_zero {L μ Δ : ℝ} (src : Fin n)
    {A : Algorithm n m c} (hA : A.Admissible L μ Δ 1) (ω : A.Ω) (t : ℕ) :
    (A.avgOutput (instanceOf src L μ Δ m) 1 ω t) 1 = 0 := by
  let I := instanceOf src L μ Δ m
  have h (i : Fin n) : A.output I 1 ω t i 1 = 0 := by
    apply not_not.mp
    intro hne
    have hv := hA.out_zero_respecting 2 (A.info I 1 ω i) (A.hist I 1 ω (microSteps*t) i)
      (show 1 ∈ supp (A.output I 1 ω t i) from hne)
    exact second_not_visible src i (visible_of_allowed rfl (history_visible src hA ω _ i) hv)
  change EuclideanSpace.proj (𝕜 := ℝ) (1 : Fin 2) (A.avgOutput I 1 ω t) = 0
  simp [Algorithm.avgOutput, map_sum, h]

theorem never_hits {L μ Δ ε : ℝ} (hn : n ≠ 0) (hm : m ≠ 0) (hL : 0 ≤ L)
    (hμ : 0 < μ) (hΔ : 0 ≤ Δ) (hε : 0 < ε)
    (hsmall : ε < Δ * (ratio (coupling n L) μ)^2) (src : Fin n) :
    ∃ I : Instance n m 2, I.InClass L μ Δ ∧
      ∀ (A : Algorithm n m c), A.Admissible L μ Δ 1 → A.hittingTime I 1 ε = ⊤ := by
  let I := instanceOf src L μ Δ m
  refine ⟨I, isolated_class hn hm hL hμ hΔ src, fun A hA => ?_⟩
  have hbound (t : ℕ) : ENNReal.ofReal (Δ * (ratio (coupling n L) μ)^2) ≤ A.expectedGap I 1 t := by
    have hi : (∫⁻ _ : A.Ω, ENNReal.ofReal (Δ * (ratio (coupling n L) μ)^2) ∂A.P) ≤
        A.expectedGap I 1 t := by
      apply lintegral_mono
      intro ω
      change ENNReal.ofReal (Δ * (ratio (coupling n L) μ)^2) ≤
        (instanceOf src L μ Δ m).gap (A.avgOutput I 1 ω t)
      rw [isolated_gap hn hm hL hμ hΔ]
      exact ENNReal.ofReal_le_ofReal (energy_missing
        (coupling_nonneg (Nat.one_le_iff_ne_zero.mpr hn) hL) hμ hΔ _
        (output_second_zero src hA ω t))
    simpa using hi
  refine le_antisymm le_top (le_iInf fun t => le_iInf fun ht => ?_)
  exact (hsmall.not_ge ((ENNReal.ofReal_le_ofReal_iff hε.le).mp ((hbound t).trans ht))).elim

end Disconnected

end ULB63
