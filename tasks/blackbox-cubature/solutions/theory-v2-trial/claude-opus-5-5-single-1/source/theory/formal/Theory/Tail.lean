import Theory.OneDim

/-!
# The preintegrated profile `H k s = Ψ(ψ_k(-s))` and its loss of smoothness

* `gtail a = γ((a, ∞))`, the standard Gaussian tail, is `C^∞` with derivative `-ρ`;
* `Hfun k s = gtail (psi k (-s))` is `C^(k+1)`, but its `(k+1)`-st derivative is not
  differentiable at `0`.
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory
open scoped ContDiff

namespace GKLS

/-- The standard normal density. -/
noncomputable abbrev rho : ℝ → ℝ := gaussianPDFReal 0 1

/-- The standard Gaussian tail `Ψ(a) = γ((a, ∞))`. -/
noncomputable def gtail (a : ℝ) : ℝ := (gaussianReal 0 1).real (Ioi a)

lemma continuous_rho : Continuous rho := by
  unfold rho gaussianPDFReal; fun_prop

lemma contDiff_rho : ContDiff ℝ ∞ rho := by
  unfold rho gaussianPDFReal
  apply contDiff_const.mul
  apply Real.contDiff_exp.comp
  fun_prop

lemma integrable_rho : Integrable rho := integrable_gaussianPDFReal 0 1

lemma gtail_eq_integral (a : ℝ) : gtail a = ∫ x in Ioi a, rho x := by
  rw [gtail, measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero (Ioi a),
    ENNReal.toReal_ofReal]
  exact setIntegral_nonneg measurableSet_Ioi (fun x _ => gaussianPDFReal_nonneg 0 1 x)

lemma gtail_eq_sub (a : ℝ) : gtail a = gtail 0 - ∫ x in (0 : ℝ)..a, rho x := by
  rw [gtail_eq_integral, gtail_eq_integral,
    ← intervalIntegral.integral_Ioi_sub_Ioi' integrable_rho.integrableOn integrable_rho.integrableOn]
  ring

lemma hasDerivAt_gtail (a : ℝ) : HasDerivAt gtail (-rho a) a := by
  have h : HasDerivAt (fun u => ∫ x in (0 : ℝ)..u, rho x) (rho a) a :=
    intervalIntegral.integral_hasDerivAt_right (continuous_rho.intervalIntegrable _ _)
      (continuous_rho.stronglyMeasurableAtFilter _ _) continuous_rho.continuousAt
  have h2 := h.const_sub (gtail 0)
  have hfun : gtail = fun u => gtail 0 - ∫ x in (0 : ℝ)..u, rho x := funext gtail_eq_sub
  rw [hfun]
  exact h2

lemma deriv_gtail : deriv gtail = fun a => -rho a := funext fun a => (hasDerivAt_gtail a).deriv

lemma contDiff_gtail : ContDiff ℝ ∞ gtail := by
  rw [contDiff_infty_iff_deriv]
  refine ⟨fun a => (hasDerivAt_gtail a).differentiableAt, ?_⟩
  rw [deriv_gtail]
  exact contDiff_rho.neg

lemma rho_le_of_le {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) : rho b ≤ rho a := by
  unfold rho gaussianPDFReal
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Real.exp_le_exp.mpr
  have : (a - 0) ^ 2 ≤ (b - 0) ^ 2 := by
    simp only [sub_zero]; exact pow_le_pow_left₀ ha hab 2
  have h2 : (0 : ℝ) < 2 * ((1 : NNReal) : ℝ) := by norm_num
  rw [div_le_div_iff_of_pos_right h2]
  linarith

/-- Lower bound for a difference of tails by the mean value theorem. -/
lemma gtail_sub_ge {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    (b - a) * rho b ≤ gtail a - gtail b := by
  rcases eq_or_lt_of_le hab with h | h
  · subst h; simp
  · obtain ⟨c, hc, hcd⟩ := exists_hasDerivAt_eq_slope gtail (fun x => -rho x) h
      contDiff_gtail.continuous.continuousOn (fun x _ => hasDerivAt_gtail x)
    have hba : 0 < b - a := sub_pos.mpr h
    have : gtail a - gtail b = (b - a) * rho c := by
      field_simp at hcd
      linarith
    rw [this]
    exact mul_le_mul_of_nonneg_left (rho_le_of_le (ha.trans hc.1.le) hc.2.le) hba.le

/-- The preintegrated profile `H k s = Ψ(ψ_k(-s))`. -/
noncomputable def Hfun (k : ℕ) (s : ℝ) : ℝ := gtail (psi k (-s))

lemma contDiff_Hfun (k : ℕ) : ContDiff ℝ (k + 1) (Hfun k) := by
  have h1 : ContDiff ℝ (k + 1) gtail := contDiff_gtail.of_le (by exact_mod_cast le_top)
  exact h1.comp ((contDiff_psi k).comp contDiff_neg)

lemma Hfun_of_nonneg (k : ℕ) {s : ℝ} (hs : 0 ≤ s) : Hfun k s = gtail (-s) := by
  rw [Hfun, psi_of_nonpos k (by linarith)]

lemma Hfun_lower (k : ℕ) {s : ℝ} (hs1 : -1 ≤ s) (hs0 : s ≤ 0) :
    rho 1 * epsk k * (-s / 2) ^ (k + 2) ≤ Hfun k s - gtail (-s) := by
  set v := -s with hv
  have hv0 : 0 ≤ v := by linarith
  have hv1 : v ≤ 1 := by linarith
  set w := psi k v with hw
  have hw0 : 0 ≤ w := psi_nonneg k hv0
  have hwv : w ≤ v := psi_le k v
  have heq : w + epsk k * w ^ (k + 2) = v := psi_eq k hv0 hv1
  have hw1 : w ≤ 1 := hwv.trans hv1
  have hpow : w ^ (k + 2) ≤ w := by
    calc w ^ (k + 2) ≤ w ^ 1 := pow_le_pow_of_le_one hw0 hw1 (by omega)
      _ = w := pow_one w
  have heps := epsk_pos k
  have heps2 := epsk_le_half k
  have hepw : epsk k * w ^ (k + 2) ≤ w := by
    have : epsk k * w ^ (k + 2) ≤ 1 / 2 * w := mul_le_mul heps2 hpow (by positivity) (by norm_num)
    linarith
  have hhalf : v / 2 ≤ w := by linarith
  have h1 : (v - w) * rho v ≤ gtail w - gtail v := gtail_sub_ge hw0 hwv
  have h2 : rho 1 ≤ rho v := rho_le_of_le hv0 hv1
  have h3 : (v / 2) ^ (k + 2) ≤ w ^ (k + 2) := pow_le_pow_left₀ (by linarith) hhalf _
  have hrho1 : 0 ≤ rho 1 := gaussianPDFReal_nonneg 0 1 1
  have h4 : v - w = epsk k * w ^ (k + 2) := by linarith
  have : Hfun k s - gtail (-s) = gtail w - gtail v := by simp [Hfun, hw, hv]
  rw [this]
  calc rho 1 * epsk k * (v / 2) ^ (k + 2) ≤ rho 1 * epsk k * w ^ (k + 2) :=
        mul_le_mul_of_nonneg_left h3 (by positivity)
    _ ≤ epsk k * w ^ (k + 2) * rho v := by
        rw [show rho 1 * epsk k * w ^ (k + 2) = epsk k * w ^ (k + 2) * rho 1 by ring]
        exact mul_le_mul_of_nonneg_left h2 (by positivity)
    _ = (v - w) * rho v := by rw [h4]
    _ ≤ gtail w - gtail v := h1

/-! ### A Taylor–Peano type lemma -/

/-- A function vanishing on `[0, ∞)` has zero derivative at every point of `[0, ∞)` where it is
differentiable. -/
lemma deriv_eq_zero_of_vanish {D : ℝ → ℝ} (hD : ∀ s, 0 ≤ s → D s = 0) {s : ℝ} (hs : 0 ≤ s)
    (hdiff : DifferentiableAt ℝ D s) : deriv D s = 0 := by
  have h1 : HasDerivWithinAt D (deriv D s) (Ici 0) s := hdiff.hasDerivAt.hasDerivWithinAt
  have h2 : HasDerivWithinAt D 0 (Ici 0) s :=
    (hasDerivWithinAt_const s (Ici (0 : ℝ)) (0 : ℝ)).congr (fun t ht => hD t ht) (hD s hs)
  exact (uniqueDiffOn_Ici 0 s hs).eq_deriv _ h1 h2

/-- If `D ∈ C^m` vanishes on `[0, ∞)` and `D^{(m)}` is differentiable at `0`, then
`D(s) = o(|s|^{m+1})` as `s → 0`. -/
lemma taylor_vanish (m : ℕ) : ∀ D : ℝ → ℝ, ContDiff ℝ m D → (∀ s, 0 ≤ s → D s = 0) →
    DifferentiableAt ℝ (iteratedDeriv m D) 0 →
    ∀ ε > 0, ∃ δ > 0, ∀ s, |s| < δ → |D s| ≤ ε * |s| ^ (m + 1) := by
  induction m with
  | zero =>
    intro D _ hD hdiff ε hε
    simp only [iteratedDeriv_zero] at hdiff
    have hd0 : deriv D 0 = 0 := deriv_eq_zero_of_vanish hD le_rfl hdiff
    have hlo := hdiff.hasDerivAt
    rw [hd0, hasDerivAt_iff_isLittleO] at hlo
    have := hlo.def hε
    obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.mp this
    refine ⟨δ, hδ, fun s hs => ?_⟩
    have h := hball (show dist s 0 < δ by simpa [Real.dist_eq] using hs)
    simpa [hD 0 le_rfl] using h
  | succ m ih =>
    intro D hC hD hdiff ε hε
    have hC' : ContDiff ℝ (m + 1) D := by exact_mod_cast hC
    rw [contDiff_succ_iff_deriv] at hC'
    obtain ⟨hDdiff, -, hderiv⟩ := hC'
    have hD' : ∀ s, 0 ≤ s → deriv D s = 0 := fun s hs =>
      deriv_eq_zero_of_vanish hD hs (hDdiff s)
    have hdiff' : DifferentiableAt ℝ (iteratedDeriv m (deriv D)) 0 := by
      rwa [← iteratedDeriv_succ']
    obtain ⟨δ, hδ, hb⟩ := ih (deriv D) hderiv hD' hdiff' ε hε
    refine ⟨δ, hδ, fun s hs => ?_⟩
    have hbound : ∀ x ∈ Metric.closedBall (0 : ℝ) |s|, ‖deriv D x‖ ≤ ε * |s| ^ (m + 1) := by
      intro x hx
      rw [Metric.mem_closedBall, Real.dist_eq, sub_zero] at hx
      have h1 := hb x (lt_of_le_of_lt hx hs)
      rw [Real.norm_eq_abs]
      calc |deriv D x| ≤ ε * |x| ^ (m + 1) := h1
        _ ≤ ε * |s| ^ (m + 1) :=
          mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (abs_nonneg x) hx _) hε.le
    have hmvt := (convex_closedBall (0 : ℝ) |s|).norm_image_sub_le_of_norm_deriv_le
      (fun x _ => hDdiff x) hbound (Metric.mem_closedBall_self (abs_nonneg s))
      (show s ∈ Metric.closedBall (0 : ℝ) |s| by simp [Real.dist_eq])
    rw [hD 0 le_rfl, sub_zero, sub_zero, Real.norm_eq_abs, Real.norm_eq_abs] at hmvt
    calc |D s| ≤ ε * |s| ^ (m + 1) * |s| := hmvt
      _ = ε * |s| ^ (m + 1 + 1) := by ring

/-- **Key one-dimensional fact.** The `(k+1)`-st derivative of `H k` is not differentiable at
`0`, so `H k ∉ C^(k+2)`. -/
lemma not_differentiableAt_iteratedDeriv_Hfun (k : ℕ) :
    ¬ DifferentiableAt ℝ (iteratedDeriv (k + 1) (Hfun k)) 0 := by
  intro hdiff
  set D : ℝ → ℝ := fun s => Hfun k s - gtail (-s) with hDdef
  have hG : ContDiff ℝ ∞ (fun s => gtail (-s)) := contDiff_gtail.comp contDiff_neg
  have hG' : ContDiff ℝ (k + 1) (fun s => gtail (-s)) := hG.of_le (by exact_mod_cast le_top)
  have hC : ContDiff ℝ (k + 1) D := (contDiff_Hfun k).sub hG'
  have hvan : ∀ s, 0 ≤ s → D s = 0 := fun s hs => by simp [hDdef, Hfun_of_nonneg k hs]
  have hit : iteratedDeriv (k + 1) D =
      fun s => iteratedDeriv (k + 1) (Hfun k) s - iteratedDeriv (k + 1) (fun s => gtail (-s)) s := by
    funext s
    exact iteratedDeriv_sub (contDiff_Hfun k).contDiffAt hG'.contDiffAt
  have hdiffG : Differentiable ℝ (iteratedDeriv (k + 1) (fun s => gtail (-s))) :=
    hG.differentiable_iteratedDeriv (k + 1) (by exact_mod_cast WithTop.coe_lt_top _)
  have hdiffD : DifferentiableAt ℝ (iteratedDeriv (k + 1) D) 0 := by
    rw [hit]; exact hdiff.sub (hdiffG 0)
  have hc : 0 < rho 1 * epsk k / 2 ^ (k + 2) :=
    div_pos (mul_pos (gaussianPDFReal_pos 0 1 1 one_ne_zero) (epsk_pos k)) (by positivity)
  obtain ⟨δ, hδ, hb⟩ := taylor_vanish (k + 1) D (by exact_mod_cast hC) hvan hdiffD
    (rho 1 * epsk k / 2 ^ (k + 2) / 2) (by positivity)
  set s : ℝ := -min (δ / 2) 1 with hs
  have hspos : 0 < min (δ / 2) 1 := lt_min (by linarith) one_pos
  have habs : |s| = min (δ / 2) 1 := by rw [hs, abs_neg, abs_of_pos hspos]
  have hslt : |s| < δ := by rw [habs]; exact lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have h1 := hb s hslt
  have h2 := Hfun_lower k (s := s) (by rw [hs]; linarith [min_le_right (δ / 2) (1 : ℝ)])
    (by rw [hs]; linarith)
  have h3 : rho 1 * epsk k * (-s / 2) ^ (k + 2) = rho 1 * epsk k / 2 ^ (k + 2) * |s| ^ (k + 2) := by
    rw [habs, hs, neg_neg, div_pow]; ring
  have h4 : D s ≤ |D s| := le_abs_self _
  have hpos : 0 < |s| ^ (k + 2) := by rw [habs]; positivity
  have : rho 1 * epsk k / 2 ^ (k + 2) * |s| ^ (k + 2) ≤
      rho 1 * epsk k / 2 ^ (k + 2) / 2 * |s| ^ (k + 1 + 1) := by
    calc rho 1 * epsk k / 2 ^ (k + 2) * |s| ^ (k + 2) = rho 1 * epsk k * (-s / 2) ^ (k + 2) := h3.symm
      _ ≤ D s := h2
      _ ≤ |D s| := h4
      _ ≤ _ := h1
  have : rho 1 * epsk k / 2 ^ (k + 2) * |s| ^ (k + 2) ≤ 0 := by
    rw [show k + 1 + 1 = k + 2 by ring] at this; nlinarith
  nlinarith [mul_pos hc hpos]

end GKLS
