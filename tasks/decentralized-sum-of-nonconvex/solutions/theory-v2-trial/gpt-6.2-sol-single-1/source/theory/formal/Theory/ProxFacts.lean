import Theory.LocalSmoothness

/-! Analytic facts for exact-gradient proximal acceleration in the extended-real model. -/

open Finset Filter Set InnerProductSpace
open scoped ENNReal RealInnerProductSpace Topology

noncomputable section

namespace ULB63

section Gradients

variable {n d : ℕ}

theorem hasGradientAt_fin_average {g : Fin n → E d → ℝ}
    {x : E d} (hg : ∀ i, DifferentiableAt ℝ (g i) x) :
    HasGradientAt (fun y => (1/n : ℝ) * ∑ i, g i y)
      ((1/n : ℝ) • ∑ i, gradient (g i) x) x := by
  rw [hasGradientAt_iff_hasFDerivAt, map_smul, map_sum]
  exact (HasFDerivAt.fun_sum (u := univ) fun i _ => (hg i).hasGradientAt.hasFDerivAt).const_mul _

theorem smooth_fin_average {g : Fin n → E d → ℝ} {L : ℝ} (hn : n ≠ 0)
    (hg : ∀ i, LSmooth L (g i)) :
    LSmooth L (fun y => (1/n : ℝ) * ∑ i, g i y) := by
  have hd (x : E d) := hasGradientAt_fin_average (fun i => (hg i).1 x)
  refine ⟨fun x => (hd x).differentiableAt, fun x y => ?_⟩
  rw [(hd x).gradient, (hd y).gradient, ← smul_sub, ← sum_sub_distrib, norm_smul,
    Real.norm_of_nonneg (by positivity)]
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  calc
    _ ≤ (1/n : ℝ) * ∑ i : Fin n, ‖gradient (g i) x - gradient (g i) y‖ :=
      mul_le_mul_of_nonneg_left (norm_sum_le _ _) (by positivity)
    _ ≤ (1/n : ℝ) * ∑ _i : Fin n, L * ‖x-y‖ :=
      mul_le_mul_of_nonneg_left (sum_le_sum fun i _ => (hg i).2 x y) (by positivity)
    _ = L * ‖x-y‖ := by simp [nsmul_eq_mul]; field_simp

theorem fSmooth_of_local {n m d : ℕ} {I : Instance n m d} {L : ℝ} (hn : n ≠ 0)
    (hi : ∀ i, LSmooth L (I.localAverage i)) : LSmooth L I.f :=
  smooth_fin_average hn hi

theorem localSmooth_of_components {n m d : ℕ} {I : Instance n m d} {L : ℝ} (hm : m ≠ 0)
    (hi : ∀ i ξ, LSmooth L (I.F i ξ)) (i : Fin n) : LSmooth L (I.localAverage i) :=
  smooth_fin_average hm (hi i)

private theorem lineDeriv {g : E d → ℝ} (hg : Differentiable ℝ g) (x v : E d) (t : ℝ) :
    HasDerivAt (fun z : ℝ => g (x + z • v)) ⟪gradient g (x + t • v), v⟫ t := by
  have hline : HasDerivAt (fun z : ℝ => x + z • v) v t := by
    simpa only [id_eq, one_smul] using ((hasDerivAt_id t).smul_const v).const_add x
  simpa only [toDual_apply_apply, Function.comp_def] using
    ((hg (x + t • v)).hasGradientAt.hasFDerivAt.comp_hasDerivAt t hline)

theorem smooth_upper_taylor {g : E d → ℝ} {L : ℝ} (_hL : 0 ≤ L) (hg : LSmooth L g)
    (x y : E d) : g y ≤ g x + ⟪gradient g x, y-x⟫ + (L/2)*‖y-x‖^2 := by
  let v := y-x
  let f := fun t : ℝ => g (x+t • v) - t * ⟪gradient g x, v⟫ - (L*‖v‖^2/2) * t^2
  have hd (t : ℝ) : HasDerivAt f
      (⟪gradient g (x+t • v) - gradient g x, v⟫ - L*‖v‖^2*t) t := by
    have h₁ := lineDeriv hg.1 x v t
    have h₂ := (hasDerivAt_id t).mul_const ⟪gradient g x, v⟫
    have h₃ := ((hasDerivAt_id t).pow 2).const_mul (L*‖v‖^2/2)
    convert (h₁.sub h₂).sub h₃ using 1
    · rfl
    · simp [inner_sub_left]; ring
  have hanti : AntitoneOn f (Icc (0 : ℝ) 1) := by
    apply antitoneOn_of_deriv_nonpos (convex_Icc _ _)
      (fun t _ => (hd t).continuousAt.continuousWithinAt) (fun t _ => (hd t).differentiableAt.differentiableWithinAt)
    intro t ht
    have ht₀ : 0 ≤ t := (interior_subset ht).1
    rw [(hd t).deriv, sub_nonpos]
    have hdiff := hg.2 (x+t • v) x
    rw [add_sub_cancel_left, norm_smul, Real.norm_of_nonneg ht₀] at hdiff
    calc
      _ ≤ ‖gradient g (x+t • v) - gradient g x‖ * ‖v‖ := real_inner_le_norm _ _
      _ ≤ (L * (t * ‖v‖))*‖v‖ := mul_le_mul_of_nonneg_right hdiff (norm_nonneg _)
      _ = _ := by ring
  have he := hanti (left_mem_Icc.2 (by norm_num)) (right_mem_Icc.2 (by norm_num))
    (by norm_num : (0 : ℝ) ≤ 1)
  dsimp only [f] at he
  simp only [one_smul, zero_smul, add_zero, zero_mul, one_mul, one_pow, mul_one, sub_zero] at he
  have hxy : x + v = y := by dsimp [v]; abel
  rw [hxy] at he
  dsimp [v] at he
  linarith only [he]

theorem convex_lower_taylor {g : E d → ℝ} (hc : ConvexOn ℝ univ g)
    (hg : Differentiable ℝ g) (x y : E d) :
    g x + ⟪gradient g x, y-x⟫ ≤ g y := by
  let v := y-x
  let T : ℝ →ᵃ[ℝ] E d := AffineMap.lineMap x y
  have hT (t : ℝ) : T t = x + t • v := by simp [T, AffineMap.lineMap_apply_module', v, add_comm]
  have hv : ConvexOn ℝ univ (fun t : ℝ => g (x+t • v)) := by
    simpa [Function.comp_def, hT] using hc.comp_affineMap T
  have hh := hv.le_slope_of_hasDerivAt (mem_univ (0:ℝ)) (mem_univ (1:ℝ)) zero_lt_one
    (lineDeriv hg x v 0)
  have he : x + v = y := by dsimp [v]; abel
  simp only [slope_def_field, sub_zero, div_one, zero_smul, one_smul, add_zero, he] at hh
  dsimp [v] at hh
  linarith only [hh]

end Gradients

section Finite

variable {n m d : ℕ} {I : Instance n m d} {M L μ Δ : ℝ}

abbrev FiniteAt (I : Instance n m d) (x : E d) : Prop := I.h x ≠ ⊤

def realValue (I : Instance n m d) (x : E d) : ℝ := (I.φ x).toReal

def realGap (I : Instance n m d) (s x : E d) : ℝ := realValue I x - realValue I s

theorem phi_ne_bot (hI : I.InClass M μ Δ) (x : E d) : I.φ x ≠ ⊥ :=
  EReal.add_ne_bot_iff.2 ⟨EReal.coe_ne_bot _, hI.h_ne_bot x⟩

theorem phi_ne_top_iff (x : E d) : I.φ x ≠ ⊤ ↔ FiniteAt I x := by
  simp [Instance.φ]

theorem coe_realValue (hI : I.InClass M μ Δ) {x : E d} (hx : FiniteAt I x) :
    (realValue I x : EReal) = I.φ x :=
  EReal.coe_toReal ((phi_ne_top_iff x).2 hx) (phi_ne_bot hI x)

theorem realValue_eq (hI : I.InClass M μ Δ) {x : E d} (hx : FiniteAt I x) :
    realValue I x = I.f x + (I.h x).toReal := by
  apply EReal.coe_injective
  rw [coe_realValue hI hx, EReal.coe_add, EReal.coe_toReal hx (hI.h_ne_bot x)]
  rfl

theorem feasible_minimizer (hI : I.InClass M μ Δ) :
    ∃ s, FiniteAt I s ∧ ∀ y, I.φ s ≤ I.φ y := by
  obtain ⟨u, hu⟩ := hI.h_proper
  obtain ⟨s, hs⟩ := hI.minimizer
  have hfin := ((hs u).trans_lt (lt_top_iff_ne_top.mpr ((phi_ne_top_iff u).2 hu))).ne
  exact ⟨s, (phi_ne_top_iff s).1 hfin, hs⟩

theorem phiStar_eq_of_min {s : E d} (hs : ∀ y, I.φ s ≤ I.φ y) : I.φstar = I.φ s :=
  le_antisymm (iInf_le _ _) (le_iInf hs)

theorem realGap_nonneg (hI : I.InClass M μ Δ) {s x : E d} (hs : FiniteAt I s)
    (hmin : ∀ y, I.φ s ≤ I.φ y) (hx : FiniteAt I x) : 0 ≤ realGap I s x := by
  rw [realGap, sub_nonneg, ← EReal.coe_le_coe_iff, coe_realValue hI hs, coe_realValue hI hx]
  exact hmin _

theorem initial_feasible_gap (hI : I.InClass M μ Δ) {s : E d} (hs : FiniteAt I s)
    (hmin : ∀ y, I.φ s ≤ I.φ y) : FiniteAt I I.x0 ∧ realGap I s I.x0 ≤ Δ := by
  have hd := hI.initial_gap
  rw [phiStar_eq_of_min hmin, ← coe_realValue hI hs] at hd
  have hx : FiniteAt I I.x0 := (phi_ne_top_iff _).1 fun ht => by
    rw [ht] at hd
    simp at hd
  refine ⟨hx, ?_⟩
  rw [← coe_realValue hI hx, ← EReal.coe_sub] at hd
  exact EReal.coe_le_coe_iff.mp hd

theorem gap_eq_ofReal (hI : I.InClass M μ Δ) {s x : E d} (hs : FiniteAt I s)
    (hmin : ∀ y, I.φ s ≤ I.φ y) (hx : FiniteAt I x) :
    I.gap x = ENNReal.ofReal (realGap I s x) := by
  rw [Instance.gap, phiStar_eq_of_min hmin, ← coe_realValue hI hs, ← coe_realValue hI hx,
    ← EReal.coe_sub, EReal.real_coe_toENNReal]
  rfl

private theorem interp_coe {H : E d → EReal} {σ t : ℝ} (hH : StronglyConvexE σ H)
    (hn : ∀ x, H x ≠ ⊥) {x y : E d} (hx : H x ≠ ⊤) (hy : H y ≠ ⊤)
    (ht₀ : 0 ≤ t) (ht₁ : t ≤ 1) :
    let u := t • x + (1-t) • y
    H u ≠ ⊤ ∧ (H u).toReal + σ/2*t*(1-t)*‖x-y‖^2 ≤
      t * (H x).toReal + (1-t)*(H y).toReal := by
  let u := t • x + (1-t) • y
  have hh := hH x y t ht₀ ht₁
  change H u + ((σ/2*t*(1-t)*‖x-y‖^2 : ℝ) : EReal) ≤ _ at hh
  have hhR : (t : EReal) * H x + ((1-t : ℝ) : EReal) * H y =
      ((t * (H x).toReal + (1-t) * (H y).toReal : ℝ) : EReal) := by
    rw [EReal.coe_add, EReal.coe_mul, EReal.coe_mul,
      EReal.coe_toReal hx (hn x), EReal.coe_toReal hy (hn y)]
  rw [hhR] at hh
  have hu : H u ≠ ⊤ := fun he => by
    rw [he] at hh
    have htop : (⊤ : EReal) + ((σ/2*t*(1-t)*‖x-y‖^2 : ℝ) : EReal) = ⊤ :=
      EReal.add_coe_eq_top_iff.2 rfl
    rw [htop] at hh
    exact EReal.coe_ne_top _ (top_le_iff.mp hh)
  refine ⟨hu, ?_⟩
  rw [← EReal.coe_toReal hu (hn u), ← EReal.coe_add] at hh
  exact EReal.coe_le_coe_iff.mp hh

theorem phi_interp (hI : I.InClass M μ Δ) {x y : E d} (hx : FiniteAt I x) (hy : FiniteAt I y)
    {t : ℝ} (ht₀ : 0 ≤ t) (ht₁ : t ≤ 1) :
    let u := t • x + (1-t) • y
    FiniteAt I u ∧ realValue I u + μ/2*t*(1-t)*‖x-y‖^2 ≤
      t * realValue I x + (1-t)*realValue I y := by
  obtain ⟨hu, h⟩ := interp_coe hI.φ_strongly_convex (phi_ne_bot hI)
    ((phi_ne_top_iff x).2 hx) ((phi_ne_top_iff y).2 hy) ht₀ ht₁
  exact ⟨(phi_ne_top_iff _).1 hu, h⟩

theorem strong_gap_distance (hI : I.InClass M μ Δ) {s x : E d} (hs : FiniteAt I s)
    (hmin : ∀ y, I.φ s ≤ I.φ y) (hx : FiniteAt I x) :
    (μ/4)*‖x-s‖^2 ≤ realGap I s x := by
  obtain ⟨hu, h⟩ := phi_interp hI hx hs (t := 1/2) (by norm_num) (by norm_num)
  have hm := realGap_nonneg hI hs hmin hu
  dsimp only [realGap] at hm ⊢
  linarith only [h, hm]

end Finite

section ThreePoint

variable {n m d : ℕ} {I : Instance n m d} {M L μ Δ : ℝ}

private theorem norm_mix_sq (x y : E d) (t : ℝ) (h₁ : 0 ≤ t) (h₂ : t ≤ 1) :
    ‖t • x + (1-t) • y‖^2 =
      t * ‖x‖^2 + (1-t) * ‖y‖^2 - t*(1-t)*‖x-y‖^2 := by
  rw [norm_add_sq_real, norm_sub_sq_real, real_inner_smul_left, real_inner_smul_right,
    norm_smul, norm_smul, Real.norm_of_nonneg h₁, Real.norm_of_nonneg (sub_nonneg.mpr h₂)]
  ring

def proxGrad (I : Instance n m d) (L : ℝ) (y : E d) : E d :=
  I.prox (y - L⁻¹ • gradient I.f y) L⁻¹

theorem prox_grad_finite (hI : I.InClass M μ Δ) (hL : 0 < L) (y : E d) :
    FiniteAt I (proxGrad I L y) := by
  obtain ⟨u, hu⟩ := hI.h_proper
  have h := hI.prox_exact (y-L⁻¹ • gradient I.f y) L⁻¹ (by positivity) u
  intro ht
  change I.h (proxGrad I L y) + _ ≤ _ at h
  rw [ht] at h
  simp [hu] at h

private theorem prox_real_three (hI : I.InClass M μ Δ) (hL : 0 < L)
    {v p u : E d} (hp : IsProx I.h L⁻¹ v p) (hpf : FiniteAt I p) (huf : FiniteAt I u) :
    (I.h p).toReal + L/2*‖p-v‖^2 + L/2*‖p-u‖^2 ≤ (I.h u).toReal + L/2*‖u-v‖^2 := by
  let Q := fun w : E d => (I.h w).toReal + L/2*‖w-v‖^2
  have hpQ (w : E d) (hw : FiniteAt I w) : Q p ≤ Q w := by
    have hh := hp w
    rw [← EReal.coe_toReal hpf (hI.h_ne_bot p), ← EReal.coe_toReal hw (hI.h_ne_bot w),
      ← EReal.coe_add, ← EReal.coe_add, EReal.coe_le_coe_iff] at hh
    have hcoeff (x : E d) : ‖x-v‖^2 / (2*L⁻¹) = L/2*‖x-v‖^2 := by field_simp
    simpa only [Q, hcoeff] using hh
  have hfor (t : ℝ) (ht : 0 < t) (ht' : t ≤ 1) :
      Q p + L/2*(1-t)*‖p-u‖^2 ≤ Q u := by
    let w := t • u + (1-t) • p
    obtain ⟨hw, hhw⟩ := interp_coe hI.h_convex hI.h_ne_bot huf hpf ht.le ht'
    change FiniteAt I w at hw
    have hnorm := norm_mix_sq (u-v) (p-v) t ht.le ht'
    have he : w-v = t • (u-v) + (1-t) • (p-v) := by dsimp [w]; module
    rw [← he, sub_sub_sub_cancel_right, norm_sub_rev u p] at hnorm
    have hm := hpQ w hw
    have hq : Q w ≤ t * Q u + (1-t) * Q p - L/2*t*(1-t)*‖p-u‖^2 := by
      dsimp only [Q] at ⊢
      rw [hnorm]
      have hh' : (I.h w).toReal ≤ t*(I.h u).toReal+(1-t)*(I.h p).toReal := by simpa using hhw
      linarith only [hh']
    apply (mul_le_mul_iff_right₀ ht).mp
    nlinarith only [hm, hq]
  change Q p + L/2*‖p-u‖^2 ≤ Q u
  have hc : Continuous (fun t : ℝ => Q p + L/2*(1-t)*‖p-u‖^2) := by fun_prop
  have ht : Tendsto (fun t : ℝ => Q p + L/2*(1-t)*‖p-u‖^2) (𝓝[>] 0)
      (𝓝 (Q p + L/2*(1-(0:ℝ))*‖p-u‖^2)) := hc.continuousAt.continuousWithinAt
  have hle : ∀ᶠ t : ℝ in 𝓝[>] 0, Q p + L/2*(1-t)*‖p-u‖^2 ≤ Q u := by
    filter_upwards [Ioo_mem_nhdsGT (by norm_num : (0:ℝ) < 1)] with t ht
    exact hfor t ht.1 ht.2.le
  simpa using le_of_tendsto ht hle

theorem prox_grad_three_point (hI : I.InClass M μ Δ) (hL : 0 < L) (hf : LSmooth L I.f)
    (y u : E d) (hu : FiniteAt I u) :
    let p := proxGrad I L y
    realValue I p ≤ realValue I u + L/2*(‖u-y‖^2 - ‖u-p‖^2) := by
  let p := proxGrad I L y
  let g := gradient I.f y
  let v := y - L⁻¹ • g
  have hp : FiniteAt I p := prox_grad_finite hI hL y
  have hpre : IsProx I.h L⁻¹ v p := hI.prox_exact _ _ (by positivity)
  have ht := prox_real_three hI hL hpre hp hu
  have hexpand (x : E d) : L/2*‖x-v‖^2 =
      L/2*‖x-y‖^2 + ⟪g,x-y⟫ + (L⁻¹/2)*‖g‖^2 := by
    have hvec : x-v = (x-y) + L⁻¹ • g := by dsimp [v]; abel
    rw [hvec, norm_add_sq_real, norm_smul, Real.norm_of_nonneg (by positivity),
      real_inner_smul_right, real_inner_comm (x-y) g]
    field_simp
  rw [hexpand, hexpand, norm_sub_rev p u] at ht
  have hfy := smooth_upper_taylor hL.le hf y p
  have hfu := convex_lower_taylor hI.f_convex hf.1 y u
  change realValue I p ≤ _
  rw [realValue_eq hI hp, realValue_eq hI hu]
  change I.f p ≤ _ at hfy
  linarith only [ht, hfy, hfu]

end ThreePoint

end ULB63
