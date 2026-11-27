import Theory.Sanity

open Set Filter
open scoped ENNReal RealInnerProductSpace Topology

set_option maxHeartbeats 1000000

noncomputable section

namespace ULB63
namespace Composite

variable {n m d : ℕ} {L μ Δ : ℝ} {I : Instance n m d}

def average {ι V : Type*} [Fintype ι] [AddCommMonoid V] [Module ℝ V] (v : ι → V) : V :=
  (Fintype.card ι : ℝ)⁻¹ • ∑ i, v i

theorem average_const {ι V : Type*} [Fintype ι] [Nonempty ι] [AddCommMonoid V] [Module ℝ V]
    (v : V) : average (fun _ : ι => v) = v := by
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  simp only [average, Finset.sum_const, Finset.card_univ]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ hn, one_smul]

def feasible (I : Instance n m d) (x : E d) : Prop := I.h x ≠ ⊤

def value (I : Instance n m d) (x : E d) : ℝ := I.f x + (I.h x).toReal

def loss (I : Instance n m d) (u x : E d) : ℝ := value I x - value I u

theorem value_coe (hI : I.InClass L μ Δ) {x : E d} (hx : feasible I x) :
    I.φ x = (value I x : EReal) := by
  rw [Instance.φ, value, EReal.coe_add, EReal.coe_toReal hx (hI.h_ne_bot x)]

theorem optimal_feasible (hI : I.InClass L μ Δ) {u : E d} (hu : ∀ x, I.φ u ≤ I.φ x) :
    feasible I u := by
  obtain ⟨x, hx⟩ := hI.h_proper
  have H := ne_top_of_le_ne_top (EReal.add_ne_top (EReal.coe_ne_top (I.f x)) hx) (hu x)
  intro h
  exact H (by simp [Instance.φ, h])

theorem star_eq (I : Instance n m d) {u : E d} (hu : ∀ x, I.φ u ≤ I.φ x) :
    I.φstar = I.φ u :=
  le_antisymm (iInf_le _ u) (le_iInf hu)

theorem initial_feasible (hI : I.InClass L μ Δ) {u : E d} (hu : ∀ x, I.φ u ≤ I.φ x) :
    feasible I I.x0 := by
  have hh := hI.initial_gap
  rw [star_eq I hu, value_coe hI (optimal_feasible hI hu)] at hh
  intro htop
  simp [Instance.φ, htop] at hh

theorem gap_eq (hI : I.InClass L μ Δ) {u x : E d} (hu : ∀ x, I.φ u ≤ I.φ x)
    (hx : feasible I x) : I.gap x = ENNReal.ofReal (loss I u x) := by
  rw [Instance.gap, star_eq I hu, value_coe hI hx, value_coe hI (optimal_feasible hI hu),
    ← EReal.coe_sub]
  rfl

theorem initial_loss (hI : I.InClass L μ Δ) {u : E d} (hu : ∀ x, I.φ u ≤ I.φ x) :
    loss I u I.x0 ≤ Δ := by
  have hh := hI.initial_gap
  rw [star_eq I hu, value_coe hI (initial_feasible hI hu),
    value_coe hI (optimal_feasible hI hu), ← EReal.coe_sub] at hh
  exact EReal.coe_le_coe_iff.mp hh

theorem loss_nonneg (hI : I.InClass L μ Δ) {u x : E d} (hu : ∀ x, I.φ u ≤ I.φ x)
    (hx : feasible I x) : 0 ≤ loss I u x := by
  have hh := hu x
  rw [value_coe hI (optimal_feasible hI hu), value_coe hI hx] at hh
  exact sub_nonneg.mpr (EReal.coe_le_coe_iff.mp hh)

def blend {d : ℕ} (a : ℝ) (x z : E d) : E d := (1 - a) • x + a • z

theorem phi_blend (hI : I.InClass L μ Δ) {x z : E d} (hx : feasible I x)
    (hz : feasible I z) {a : ℝ} (ha : 0 ≤ a) (ha₁ : a ≤ 1) :
    feasible I (blend a x z) ∧
      value I (blend a x z) + μ / 2 * a * (1 - a) * ‖z - x‖ ^ 2 ≤
        (1 - a) * value I x + a * value I z := by
  have H := hI.φ_strongly_convex z x a ha ha₁
  rw [← add_comm ((1 - a) • x) (a • z)] at H
  change I.φ (blend a x z) + ((μ / 2 * a * (1 - a) * ‖z - x‖ ^ 2 : ℝ) : EReal) ≤ _ at H
  rw [value_coe hI hx, value_coe hI hz] at H
  have HH : I.φ (blend a x z) + ((μ / 2 * a * (1 - a) * ‖z - x‖ ^ 2 : ℝ) : EReal) ≤
      ((a * value I z + (1-a) * value I x : ℝ) : EReal) := by
    push_cast
    exact H
  have hf : feasible I (blend a x z) := by
    intro ht
    have ht' : I.φ (blend a x z) = ⊤ := by simp [Instance.φ, ht]
    rw [ht', EReal.top_add_coe, top_le_iff] at HH
    exact EReal.coe_ne_top _ HH
  rw [value_coe hI hf, ← EReal.coe_add, EReal.coe_le_coe_iff] at HH
  exact ⟨hf, by linarith⟩

theorem loss_blend (hI : I.InClass L μ Δ) (hμ : 0 ≤ μ) {u x z : E d}
    (hx : feasible I x) (hz : feasible I z) {a : ℝ} (ha : 0 ≤ a) (ha₁ : a ≤ 1) :
    loss I u (blend a x z) ≤ (1 - a) * loss I u x + a * loss I u z := by
  have hnon : 0 ≤ μ / 2 * a * (1 - a) * ‖z - x‖ ^ 2 :=
    mul_nonneg (mul_nonneg (mul_nonneg (by positivity) ha) (sub_nonneg.mpr ha₁)) (sq_nonneg _)
  unfold loss
  linarith [(phi_blend hI hx hz ha ha₁).2]

/-- The midpoint estimate suffices for the restarted potential. -/
theorem radius_bound (hI : I.InClass L μ Δ) {u x : E d} (hu : ∀ x, I.φ u ≤ I.φ x)
    (hx : feasible I x) : μ / 4 * ‖x - u‖ ^ 2 ≤ loss I u x := by
  obtain ⟨hv, h⟩ :=
    phi_blend hI (optimal_feasible hI hu) hx (by norm_num : (0 : ℝ) ≤ 1/2)
      (by norm_num : (1/2 : ℝ) ≤ 1)
  have hn := loss_nonneg hI hu hv
  unfold loss at hn ⊢
  nlinarith

theorem prox_feasible (hI : I.InClass L μ Δ) (v : E d) {η : ℝ} (hη : 0 < η) :
    feasible I (I.prox v η) := by
  obtain ⟨x, hx⟩ := hI.h_proper
  have hh := hI.prox_exact v η hη x
  intro hp
  rw [hp, EReal.top_add_coe, top_le_iff] at hh
  exact (EReal.add_ne_top hx (EReal.coe_ne_top _)) hh

def localGrad (I : Instance n m d) (i : Fin n) (x : E d) : E d :=
  average (fun j : Fin m => gradient (I.F i j) x)

private theorem hasFDeriv_f (hI : I.InClass L μ Δ) (x : E d) :
    HasFDerivAt I.f ((1 / (n : ℝ)) •
      ∑ i : Fin n, (1 / (m : ℝ)) • ∑ j : Fin m, fderiv ℝ (I.F i j) x) x := by
  have hder (i : Fin n) (j : Fin m) :=
    ((hI.smooth i j).1 x).hasFDerivAt
  have hinner (i : Fin n) := (HasFDerivAt.fun_sum (fun j (_ : j ∈ Finset.univ) =>
    hder i j)).const_mul (1 / (m : ℝ))
  have hout := (HasFDerivAt.fun_sum (fun i (_ : i ∈ Finset.univ) =>
    hinner i)).const_mul (1 / (n : ℝ))
  exact hout

theorem fullGrad (hI : I.InClass L μ Δ) (x : E d) :
    gradient I.f x = average (fun i : Fin n => localGrad I i x) := by
  have hdf := (hasFDeriv_f hI x).fderiv
  apply ext_inner_right ℝ
  intro z
  simp [hdf, average, localGrad, one_div, sum_inner, real_inner_smul_left]

theorem differentiable_f (hI : I.InClass L μ Δ) : Differentiable ℝ I.f := by
  intro x
  exact (hasFDeriv_f hI x).differentiableAt

theorem average_diff_bound {ι : Type*} [Fintype ι] [Nonempty ι] {u v : ι → E d} {B : ℝ}
    (h : ∀ i, ‖u i - v i‖ ≤ B) : ‖average u - average v‖ ≤ B := by
  have hn : 0 < (Fintype.card ι : ℝ) := by exact_mod_cast Fintype.card_pos
  rw [average, average, ← smul_sub, ← Finset.sum_sub_distrib, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (inv_nonneg.mpr hn.le)]
  exact (mul_le_mul_of_nonneg_left ((norm_sum_le _ _).trans
    (Finset.sum_le_sum fun i _ => h i)) (inv_nonneg.mpr hn.le)).trans (by
    simp [Finset.sum_const, nsmul_eq_mul, hn.ne'])

theorem grad_lip (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)] (x y : E d) :
    ‖gradient I.f x - gradient I.f y‖ ≤ L * ‖x - y‖ := by
  rw [fullGrad hI x, fullGrad hI y]
  refine average_diff_bound (fun i => ?_)
  exact average_diff_bound (fun j => (hI.smooth i j).2 x y)

private theorem seg_deriv {g : E d → ℝ} (hg : Differentiable ℝ g) (x y : E d) (t : ℝ) :
    HasDerivAt (fun t : ℝ => g (x + t • (y - x)))
      ⟪gradient g (x + t • (y - x)), y - x⟫ t := by
  rw [inner_gradient_left]
  exact (hg _).hasFDerivAt.comp_hasDerivAt t
    ((hasDerivAt_id t).smul_const (y - x) |>.const_add x |>.congr_deriv (one_smul _ _))

theorem tangent_lower (hI : I.InClass L μ Δ) (y x : E d) :
    I.f y + ⟪gradient I.f y, x - y⟫ ≤ I.f x := by
  let g : ℝ → ℝ := fun t => I.f (y + t • (x - y))
  have ha (t : ℝ) : y + t • (x - y) = AffineMap.lineMap y x t := by
    rw [AffineMap.lineMap_apply_module]
    module
  have hgc : ConvexOn ℝ Set.univ g := by
    simp only [g, ha]
    simpa [Function.comp_def] using hI.f_convex.comp_affineMap (AffineMap.lineMap y x)
  have h := hgc.le_slope_of_hasDerivAt (Set.mem_univ (0 : ℝ)) (Set.mem_univ (1 : ℝ))
    (show (0 : ℝ) < 1 by norm_num) (seg_deriv (differentiable_f hI) y x 0)
  have hh : ⟪gradient I.f y, x - y⟫ ≤ I.f x - I.f y := by
    simpa [g, slope_def_field] using h
  linarith

/-- A slightly loosened descent lemma, using only the mean-value bound. -/
theorem tangent_upper (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)]
    (hL : 0 ≤ L) (x y : E d) :
    I.f y ≤ I.f x + ⟪gradient I.f x, y - x⟫ + L * ‖y - x‖ ^ 2 := by
  let g : ℝ → ℝ := fun t => I.f (x + t • (y - x)) - t * ⟪gradient I.f x, y - x⟫
  let g' : ℝ → ℝ := fun t => ⟪gradient I.f (x + t • (y - x)) - gradient I.f x, y - x⟫
  have hD (t : ℝ) : HasDerivAt g (g' t) t := by
    dsimp only [g']
    rw [inner_sub_left]
    have hh := (seg_deriv (differentiable_f hI) x y t).sub
      ((hasDerivAt_id t).mul_const (⟪gradient I.f x, y - x⟫))
    rw [one_mul] at hh
    exact hh
  have hb (t : ℝ) (ht : t ∈ Ico (0 : ℝ) 1) : ‖g' t‖ ≤ L * ‖y - x‖ ^ 2 := by
    have hh := grad_lip hI (x + t • (y - x)) x
    rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1] at hh
    calc
      _ ≤ ‖gradient I.f (x + t • (y - x)) - gradient I.f x‖ * ‖y - x‖ :=
        abs_real_inner_le_norm _ _
      _ ≤ (L * (t * ‖y - x‖)) * ‖y - x‖ :=
        mul_le_mul_of_nonneg_right hh (norm_nonneg _)
      _ ≤ _ := by
        have hnon : 0 ≤ L * ‖y - x‖ ^ 2 := by positivity
        nlinarith [mul_le_mul_of_nonneg_right ht.2.le hnon]
  have H := norm_image_sub_le_of_norm_deriv_le_segment_01'
    (fun t (_ : t ∈ Icc (0 : ℝ) 1) => (hD t).hasDerivWithinAt) hb
  rw [Real.norm_eq_abs] at H
  have hh := (abs_le.mp H).2
  simp only [g, one_smul, add_sub_cancel, one_mul, zero_smul, add_zero, zero_mul, sub_zero] at hh
  linarith

private theorem h_coe (hI : I.InClass L μ Δ) {x : E d} (hx : feasible I x) :
    I.h x = ((I.h x).toReal : EReal) := (EReal.coe_toReal hx (hI.h_ne_bot x)).symm

theorem prox_support (hI : I.InClass L μ Δ) (v : E d) {η : ℝ} (hη : 0 < η)
    {u : E d} (hu : feasible I u) :
    let p := I.prox v η
    (I.h p).toReal - (I.h u).toReal ≤ ⟪p - v, u - p⟫ / η := by
  intro p
  have hp : feasible I p := prox_feasible hI v hη
  have limit : Tendsto (fun t : ℝ =>
      (I.h u).toReal - (I.h p).toReal + ⟪p-v,u-p⟫ / η + t * ‖u-p‖ ^ 2 / (2*η))
      (𝓝[>] 0) (𝓝 ((I.h u).toReal - (I.h p).toReal + ⟪p-v,u-p⟫ / η)) := by
    have hcont : Continuous (fun t : ℝ =>
        (I.h u).toReal - (I.h p).toReal + ⟪p-v,u-p⟫ / η + t * ‖u-p‖ ^ 2 / (2*η)) := by
      fun_prop
    have hc := (hcont.continuousAt (x := (0 : ℝ))).continuousWithinAt (s := Ioi (0 : ℝ))
    unfold ContinuousWithinAt at hc
    simpa only [zero_mul, zero_div, add_zero] using hc
  suffices H : 0 ≤ (I.h u).toReal - (I.h p).toReal + ⟪p-v,u-p⟫ / η by linarith
  apply ge_of_tendsto limit
  filter_upwards [Ioo_mem_nhdsGT (by norm_num : (0 : ℝ) < 1)] with t ht
  let q := blend t p u
  have hc := hI.h_convex u p t ht.1.le ht.2.le
  rw [zero_div, zero_mul, zero_mul, zero_mul, EReal.coe_zero, add_zero, h_coe hI hp,
    h_coe hI hu] at hc
  have hq : I.h q ≤ ((t * (I.h u).toReal + (1-t) * (I.h p).toReal : ℝ) : EReal) := by
    rw [EReal.coe_add, EReal.coe_mul, EReal.coe_mul]
    simpa only [q, blend, add_comm] using hc
  have hqf : feasible I q := ne_top_of_le_ne_top (EReal.coe_ne_top _) hq
  rw [h_coe hI hqf, EReal.coe_le_coe_iff] at hq
  have hprox := hI.prox_exact v η hη q
  change I.h p + _ ≤ _ at hprox
  rw [h_coe hI hp, h_coe hI hqf, ← EReal.coe_add, ← EReal.coe_add, EReal.coe_le_coe_iff] at hprox
  have hv : q - v = (p - v) + t • (u-p) := by
    dsimp [q, blend]
    module
  rw [hv, norm_add_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg ht.1.le] at hprox
  apply (mul_nonneg_iff_of_pos_left ht.1).mp
  have H : (I.h p).toReal + ‖p - v‖ ^ 2 / (2 * η) ≤
      t * (I.h u).toReal + (1 - t) * (I.h p).toReal +
        (‖p-v‖ ^ 2 + 2 * (t * ⟪p-v,u-p⟫) + (t * ‖u-p‖)^2) / (2*η) := by linarith
  rw [← sub_nonneg] at H
  convert H using 1
  ring

def step (I : Instance n m d) (L : ℝ) (y g : E d) : E d :=
  I.prox (y - (4*L)⁻¹ • g) (4*L)⁻¹

theorem step_feasible (hI : I.InClass L μ Δ) (hL : 0 < L) (y g : E d) :
    feasible I (step I L y g) := prox_feasible hI _ (by positivity)

theorem three_point (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)]
    (hL : 0 < L) (y e : E d) {w : E d} (hw : feasible I w) :
    let p := step I L y (gradient I.f y + e)
    value I p - value I w ≤
      2*L * (‖w-y‖ ^ 2 - ‖w-p‖ ^ 2) - L * ‖p-y‖ ^ 2 + ⟪e,w-p⟫ := by
  intro p
  let v := y - (4*L)⁻¹ • (gradient I.f y + e)
  have hv : p - v = (p - y) + (4*L)⁻¹ • (gradient I.f y + e) := by
    dsimp [v]
    abel
  have hs := prox_support hI v (by positivity : 0 < (4*L)⁻¹) hw
  change (I.h p).toReal - (I.h w).toReal ≤ ⟪p-v,w-p⟫ / (4*L)⁻¹ at hs
  rw [hv, inner_add_left, real_inner_smul_left, inner_add_left, div_inv_eq_mul] at hs
  have hscale : (⟪p-y,w-p⟫ + (4*L)⁻¹ * (⟪gradient I.f y,w-p⟫ + ⟪e,w-p⟫)) * (4*L) =
      4*L * ⟪p-y,w-p⟫ + ⟪gradient I.f y,w-p⟫ + ⟪e,w-p⟫ := by
    field_simp
    ring
  rw [hscale] at hs
  have hn : ‖w-y‖ ^ 2 = ‖p-y‖ ^ 2 + 2 * ⟪p-y,w-p⟫ + ‖w-p‖ ^ 2 := by
    rw [← norm_add_sq_real, show (p-y)+(w-p) = w-y by abel]
  have hinner : ⟪gradient I.f y, p-y⟫ - ⟪gradient I.f y,w-y⟫ =
      -⟪gradient I.f y,w-p⟫ := by
    rw [← inner_sub_right, ← inner_neg_right]
    congr 1
    abel
  unfold value
  rw [hn]
  linarith [tangent_lower hI y w, tangent_upper hI hL.le y p]

def momentum {d : ℕ} (a : ℝ) (y p z : E d) : E d := z + a⁻¹ • (p-y)

def potential (I : Instance n m d) (μ : ℝ) (u x z : E d) : ℝ :=
  loss I u x + μ / 4 * ‖z-u‖ ^ 2

private theorem error_bound {d : ℕ} {L : ℝ} (hL : 0 < L) (e r : E d) :
    -⟪e,r⟫ - L * ‖r‖ ^ 2 ≤ ‖e‖ ^ 2 / (4*L) := by
  have hinner : -⟪e,r⟫ ≤ ‖e‖ * ‖r‖ :=
    (neg_le_abs _).trans (abs_real_inner_le_norm _ _)
  refine (sub_le_sub_right hinner (L * ‖r‖ ^ 2)).trans ?_
  apply (le_div_iff₀ (by positivity : 0 < 4*L)).2
  nlinarith [sq_nonneg (‖e‖ - 2*L * ‖r‖)]

theorem potential_step (hI : I.InClass L μ Δ) [Nonempty (Fin n)] [Nonempty (Fin m)]
    (hL : 0 < L) (hμ : 0 ≤ μ) {a : ℝ} (ha : 0 < a) (ha₁ : a ≤ 1)
    (ha₂ : 2*L*a^2 = μ/4) {u x : E d} (hu : feasible I u) (hx : feasible I x) (z e : E d) :
    let y := blend a x z
    let p := step I L y (gradient I.f y + e)
    let z' := momentum a y p z
    potential I μ u p z' + a * loss I u x ≤
      potential I μ u x z - a * ⟪e,z-u⟫ + ‖e‖ ^ 2 / (4*L) := by
  intro y p z'
  let w := blend a x u
  have hw := (phi_blend hI hx hu ha.le ha₁).1
  have hwp : w - p = a • (u - z') := by
    have hwy : w - y = a • (u-z) := by dsimp [w,y,blend]; module
    calc
      _ = (w-y) - (p-y) := by abel
      _ = _ := by
        rw [hwy]
        dsimp [z', momentum]
        rw [show u - (z + a⁻¹ • (p-y)) = (u-z) - a⁻¹ • (p-y) by abel,
          smul_sub a (u-z), smul_smul, mul_inv_cancel₀ ha.ne', one_smul]
  have hwy : w-y = a • (u-z) := by dsimp [w,y,blend]; module
  have hin : ⟪e,w-p⟫ = -a * ⟪e,z-u⟫ - ⟪e,p-y⟫ := by
    rw [show w-p = (w-y) - (p-y) by abel, hwy, inner_sub_right, real_inner_smul_right,
      show u-z = -(z-u) by abel, inner_neg_right]
    ring
  have h := three_point hI hL y e hw
  change value I p - value I w ≤ _ at h
  rw [hin, hwp, hwy, norm_smul, norm_smul, Real.norm_eq_abs, abs_of_nonneg ha.le,
    norm_sub_rev u z', norm_sub_rev u z] at h
  have hexp : 2*L * ((a * ‖z-u‖)^2 - (a * ‖z'-u‖)^2) =
      (μ/4) * (‖z-u‖ ^ 2 - ‖z'-u‖ ^ 2) := by
    rw [← ha₂]
    ring
  rw [hexp] at h
  have hc := loss_blend hI hμ hx hu (u := u) ha.le ha₁
  change loss I u w ≤ _ at hc
  unfold potential loss at *
  linarith [error_bound hL e (p-y)]

theorem blend_norm_sq {d : ℕ} {a : ℝ} (ha : 0 ≤ a) (ha₁ : a ≤ 1) (x z u : E d) :
    ‖blend a x z - u‖ ^ 2 ≤ (1-a) * ‖x-u‖ ^ 2 + a * ‖z-u‖ ^ 2 := by
  have he : blend a x z - u = (1-a) • (x-u) + a • (z-u) := by unfold blend; module
  have h := norm_add_le ((1-a) • (x-u)) (a • (z-u))
  rw [← he, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg ha,
    abs_of_nonneg (sub_nonneg.mpr ha₁)] at h
  refine (pow_le_pow_left₀ (norm_nonneg _) h 2).trans ?_
  nlinarith [mul_nonneg (mul_nonneg ha (sub_nonneg.mpr ha₁)) (sq_nonneg (‖x-u‖-‖z-u‖))]

end Composite
end ULB63
