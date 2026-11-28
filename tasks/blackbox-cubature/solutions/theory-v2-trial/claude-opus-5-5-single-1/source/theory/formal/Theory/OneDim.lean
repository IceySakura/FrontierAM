import Mathlib

/-!
# One-dimensional lemmas for the counterexample

* `pp k t = (max t 0)^k` and its derivatives;
* `Pk k t = (max t 0)^(k+2) χ(t)` with a smooth cutoff `χ`: compactly supported, `C^(k+1)`,
  with bounded derivatives, equal to `t^(k+2)` on `[0, 1]`;
* `gfun k t = ε_k Pk k t + t`, a `C^(k+1)` increasing bijection of `ℝ` with `gfun' ≥ 1/2`, whose
  `(k+1)`-st derivative has a corner at `0`;
* `psi k`, the inverse of `gfun k`, which is again `C^(k+1)`.
-/

open Real Set Filter Topology

namespace GKLS

/-- `pp k t = (max t 0)^k`. -/
noncomputable def pp (k : ℕ) (t : ℝ) : ℝ := (max t 0) ^ k

lemma pp_nonneg (k : ℕ) (t : ℝ) : 0 ≤ pp k t := pow_nonneg (le_max_right _ _) _

lemma pp_of_nonpos {k : ℕ} (hk : k ≠ 0) {t : ℝ} (ht : t ≤ 0) : pp k t = 0 := by
  simp [pp, max_eq_right ht, zero_pow hk]

lemma pp_of_nonneg (k : ℕ) {t : ℝ} (ht : 0 ≤ t) : pp k t = t ^ k := by
  simp [pp, max_eq_left ht]

lemma hasDerivAt_pp (k : ℕ) (x : ℝ) :
    HasDerivAt (pp (k + 2)) (((k : ℝ) + 2) * pp (k + 1) x) x := by
  rcases lt_trichotomy x 0 with hx | hx | hx
  · have h1 : pp (k + 2) =ᶠ[𝓝 x] fun _ => (0 : ℝ) := by
      filter_upwards [Iio_mem_nhds hx] with t ht
      exact pp_of_nonpos (by omega) (le_of_lt ht)
    rw [pp_of_nonpos (by omega) hx.le, mul_zero]
    exact (hasDerivAt_const x (0 : ℝ)).congr_of_eventuallyEq h1
  · subst hx
    rw [pp_of_nonpos (by omega) le_rfl, mul_zero]
    have hl : HasDerivWithinAt (pp (k + 2)) 0 (Iic 0) 0 := by
      refine (hasDerivWithinAt_const (0 : ℝ) (Iic (0 : ℝ)) (0 : ℝ)).congr
        (fun t ht => pp_of_nonpos (by omega) ht) (pp_of_nonpos (by omega) le_rfl)
    have hr : HasDerivWithinAt (pp (k + 2)) 0 (Ici 0) 0 := by
      have h : HasDerivWithinAt (fun t : ℝ => t ^ (k + 2)) 0 (Ici 0) 0 := by
        simpa using (hasDerivAt_pow (k + 2) (0 : ℝ)).hasDerivWithinAt (s := Ici 0)
      exact h.congr (fun t ht => pp_of_nonneg _ ht) (pp_of_nonneg _ le_rfl)
    have h := hl.union hr
    rw [Iic_union_Ici] at h
    exact h.hasDerivAt univ_mem
  · have h1 : pp (k + 2) =ᶠ[𝓝 x] fun t => t ^ (k + 2) := by
      filter_upwards [Ioi_mem_nhds hx] with t ht
      exact pp_of_nonneg _ (le_of_lt ht)
    rw [pp_of_nonneg _ hx.le]
    have h := hasDerivAt_pow (k + 2) x
    simp only [Nat.cast_add, Nat.cast_ofNat] at h
    exact h.congr_of_eventuallyEq h1

lemma deriv_pp (k : ℕ) : deriv (pp (k + 2)) = fun x => ((k : ℝ) + 2) * pp (k + 1) x :=
  funext fun x => (hasDerivAt_pp k x).deriv

lemma continuous_pp (k : ℕ) : Continuous (pp k) :=
  (continuous_id.max continuous_const).pow k

lemma contDiff_pp (k : ℕ) : ContDiff ℝ k (pp (k + 1)) := by
  induction k with
  | zero => simpa using (continuous_pp 1)
  | succ k ih =>
    rw [show ((k + 1 : ℕ) : WithTop ℕ∞) = (k : WithTop ℕ∞) + 1 by push_cast; rfl,
      contDiff_succ_iff_deriv]
    refine ⟨fun x => (hasDerivAt_pp k x).differentiableAt, by simp, ?_⟩
    rw [deriv_pp]
    exact contDiff_const.mul ih

/-- Iterated derivatives of `pp (m+1)` up to order `m`. -/
lemma iteratedDeriv_pp (m a : ℕ) (ha : a ≤ m) :
    iteratedDeriv a (pp (m + 1)) = fun x => ((m + 1).descFactorial a : ℝ) * pp (m + 1 - a) x := by
  induction a with
  | zero => funext x; simp
  | succ a ih =>
    rw [iteratedDeriv_succ, ih (by omega)]
    obtain ⟨k, hk⟩ : ∃ k, m + 1 - a = k + 2 := ⟨m - 1 - a, by omega⟩
    funext x
    rw [hk, deriv_const_mul _ (hasDerivAt_pp k x).differentiableAt, (hasDerivAt_pp k x).deriv,
      Nat.descFactorial_succ, show m + 1 - (a + 1) = k + 1 by omega, hk]
    push_cast
    ring

/-! ### A compactly supported version -/

/-- A smooth cutoff `χ` with `χ = 1` on `[-1, 1]`, `0 ≤ χ ≤ 1`, and support `(-2, 2)`. -/
noncomputable def bump : ContDiffBump (0 : ℝ) := ⟨1, 2, zero_lt_one, one_lt_two⟩

/-- `Pk k t = (max t 0)^(k+2) χ(t)`: compactly supported, `C^(k+1)`, `= t^(k+2)` on `[0,1]`. -/
noncomputable def Pk (k : ℕ) (t : ℝ) : ℝ := pp (k + 2) t * bump t

lemma contDiff_pp_k (k : ℕ) : ContDiff ℝ (k + 1) (pp (k + 2)) := by
  have := contDiff_pp (k + 1); exact_mod_cast this

lemma contDiff_Pk (k : ℕ) : ContDiff ℝ (k + 1) (Pk k) :=
  (contDiff_pp_k k).mul bump.contDiff

lemma Pk_nonneg (k : ℕ) (t : ℝ) : 0 ≤ Pk k t := mul_nonneg (pp_nonneg _ _) bump.nonneg

lemma Pk_of_nonpos (k : ℕ) {t : ℝ} (ht : t ≤ 0) : Pk k t = 0 := by
  simp [Pk, pp_of_nonpos (show k + 2 ≠ 0 by omega) ht]

lemma Pk_of_mem (k : ℕ) {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) : Pk k t = t ^ (k + 2) := by
  have hr : bump.rIn = 1 := rfl
  have hb : bump t = 1 := bump.one_of_mem_closedBall (by
    rw [Metric.mem_closedBall, Real.dist_eq, sub_zero, abs_le, hr]
    exact ⟨by linarith, h1⟩)
  rw [Pk, pp_of_nonneg _ h0, hb, mul_one]

lemma hasCompactSupport_Pk (k : ℕ) : HasCompactSupport (Pk k) := by
  have : Pk k = pp (k + 2) * (bump : ℝ → ℝ) := rfl
  rw [this]
  exact bump.hasCompactSupport.mul_left

lemma iteratedDeriv_Pk_bounded (k a : ℕ) (ha : a ≤ k + 1) :
    ∃ C, ∀ t, |iteratedDeriv a (Pk k) t| ≤ C := by
  have hc : Continuous (iteratedDeriv a (Pk k)) :=
    (contDiff_Pk k).continuous_iteratedDeriv a (by exact_mod_cast ha)
  have hs : HasCompactSupport (iteratedDeriv a (Pk k)) := by
    have h1 := (hasCompactSupport_Pk k).iteratedFDeriv (𝕜 := ℝ) a
    have h2 : iteratedDeriv a (Pk k) = (fun L => L (fun _ => (1 : ℝ))) ∘ iteratedFDeriv ℝ a (Pk k) := by
      funext t; simp [iteratedDeriv_eq_iteratedFDeriv]
    rw [h2]
    exact h1.comp_left (by simp)
  obtain ⟨C, hC⟩ := hc.bounded_above_of_compact_support hs
  exact ⟨C, fun t => by simpa using hC t⟩

/-- A common bound for the derivatives of `Pk k` of orders `0, …, k+1`. -/
lemma Pk_derivs_bounded (k : ℕ) :
    ∃ C, 0 ≤ C ∧ ∀ a ≤ k + 1, ∀ t, |iteratedDeriv a (Pk k) t| ≤ C := by
  choose C hC using fun a : Fin (k + 2) => iteratedDeriv_Pk_bounded k a (by omega)
  refine ⟨∑ a, |C a|, Finset.sum_nonneg (fun a _ => abs_nonneg _), fun a ha t => ?_⟩
  have h1 := hC ⟨a, by omega⟩ t
  have h2 : |C ⟨a, by omega⟩| ≤ ∑ b, |C b| :=
    Finset.single_le_sum (f := fun b => |C b|) (fun b _ => abs_nonneg _) (Finset.mem_univ _)
  exact h1.trans ((le_abs_self _).trans h2)

/-- The common bound `B_k`. -/
noncomputable def Bk (k : ℕ) : ℝ := Classical.choose (Pk_derivs_bounded k)

lemma Bk_nonneg (k : ℕ) : 0 ≤ Bk k := (Classical.choose_spec (Pk_derivs_bounded k)).1

lemma Bk_spec (k a : ℕ) (ha : a ≤ k + 1) (t : ℝ) : |iteratedDeriv a (Pk k) t| ≤ Bk k :=
  (Classical.choose_spec (Pk_derivs_bounded k)).2 a ha t

/-- The scaling `ε_k = 1 / (2 (B_k + 1))`. -/
noncomputable def epsk (k : ℕ) : ℝ := 1 / (2 * (Bk k + 1))

lemma epsk_pos (k : ℕ) : 0 < epsk k := by
  have := Bk_nonneg k; unfold epsk; positivity

lemma epsk_le_half (k : ℕ) : epsk k ≤ 1 / 2 := by
  have := Bk_nonneg k
  unfold epsk
  rw [div_le_div_iff₀ (by positivity) (by norm_num)]
  nlinarith

lemma epsk_mul_Bk (k : ℕ) : epsk k * Bk k ≤ 1 / 2 := by
  have := Bk_nonneg k
  unfold epsk
  rw [div_mul_eq_mul_div, one_mul, div_le_div_iff₀ (by positivity) (by norm_num)]
  nlinarith

/-- `|ε_k P_k^{(a)}| ≤ 1/2` for `a ≤ k + 1`. -/
lemma epsk_iteratedDeriv_le (k a : ℕ) (ha : a ≤ k + 1) (t : ℝ) :
    |epsk k * iteratedDeriv a (Pk k) t| ≤ 1 / 2 := by
  rw [abs_mul, abs_of_pos (epsk_pos k)]
  exact (mul_le_mul_of_nonneg_left (Bk_spec k a ha t) (epsk_pos k).le).trans (epsk_mul_Bk k)

/-! ### The increasing bijection `gfun` -/

/-- `gfun k t = ε_k (max t 0)^(k+2) χ(t) + t`. -/
noncomputable def gfun (k : ℕ) (t : ℝ) : ℝ := epsk k * Pk k t + t

lemma hasDerivAt_gfun (k : ℕ) (x : ℝ) :
    HasDerivAt (gfun k) (epsk k * deriv (Pk k) x + 1) x := by
  have hd : DifferentiableAt ℝ (Pk k) x :=
    ((contDiff_Pk k).differentiable (by simp)) x
  exact (hd.hasDerivAt.const_mul (epsk k)).add (hasDerivAt_id x)

lemma half_le_deriv_gfun (k : ℕ) (x : ℝ) : 1 / 2 ≤ epsk k * deriv (Pk k) x + 1 := by
  have h := epsk_iteratedDeriv_le k 1 (by omega) x
  rw [iteratedDeriv_one] at h
  have := neg_abs_le (epsk k * deriv (Pk k) x)
  linarith

lemma contDiff_gfun (k : ℕ) : ContDiff ℝ (k + 1) (gfun k) :=
  (contDiff_const.mul (contDiff_Pk k)).add contDiff_id

lemma gfun_of_nonpos (k : ℕ) {t : ℝ} (ht : t ≤ 0) : gfun k t = t := by
  simp [gfun, Pk_of_nonpos k ht]

lemma gfun_of_mem (k : ℕ) {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    gfun k t = epsk k * t ^ (k + 2) + t := by
  simp [gfun, Pk_of_mem k h0 h1]

lemma le_gfun (k : ℕ) (t : ℝ) : t ≤ gfun k t := by
  have := mul_nonneg (epsk_pos k).le (Pk_nonneg k t)
  simp only [gfun]; linarith

lemma gfun_strictMono (k : ℕ) : StrictMono (gfun k) :=
  strictMono_of_deriv_pos fun x => by
    rw [(hasDerivAt_gfun k x).deriv]; linarith [half_le_deriv_gfun k x]

lemma continuous_gfun (k : ℕ) : Continuous (gfun k) := (contDiff_gfun k).continuous

lemma gfun_surjective (k : ℕ) : Function.Surjective (gfun k) := by
  intro v
  rcases le_or_gt v 0 with hv | hv
  · exact ⟨v, gfun_of_nonpos k hv⟩
  · have h0 : gfun k 0 ≤ v := by rw [gfun_of_nonpos k le_rfl]; exact hv.le
    have h1 : v ≤ gfun k v := le_gfun k v
    obtain ⟨c, _, hc⟩ := intermediate_value_Icc hv.le (continuous_gfun k).continuousOn ⟨h0, h1⟩
    exact ⟨c, hc⟩

/-- `gfun k` as an order isomorphism of `ℝ`. -/
noncomputable def gIso (k : ℕ) : ℝ ≃o ℝ :=
  StrictMono.orderIsoOfSurjective (gfun k) (gfun_strictMono k) (gfun_surjective k)

/-- The inverse `ψ = gfun⁻¹`. -/
noncomputable def psi (k : ℕ) (v : ℝ) : ℝ := (gIso k).symm v

lemma gfun_psi (k : ℕ) (v : ℝ) : gfun k (psi k v) = v :=
  StrictMono.orderIsoOfSurjective_self_symm_apply _ _ _ v

lemma psi_gfun (k : ℕ) (t : ℝ) : psi k (gfun k t) = t :=
  StrictMono.orderIsoOfSurjective_symm_apply_self _ _ _ t

lemma psi_of_nonpos (k : ℕ) {v : ℝ} (hv : v ≤ 0) : psi k v = v := by
  conv_lhs => rw [← gfun_of_nonpos k hv]
  exact psi_gfun k v

lemma psi_lt_iff (k : ℕ) (v t : ℝ) : psi k v < t ↔ v < gfun k t := by
  conv_rhs => rw [← gfun_psi k v]
  exact (gfun_strictMono k).lt_iff_lt.symm

lemma psi_nonneg (k : ℕ) {v : ℝ} (hv : 0 ≤ v) : 0 ≤ psi k v := by
  by_contra h
  push Not at h
  have := (gfun_strictMono k) h
  rw [gfun_psi, gfun_of_nonpos k le_rfl] at this
  linarith

lemma psi_le (k : ℕ) (v : ℝ) : psi k v ≤ v := by
  have := le_gfun k (psi k v)
  rwa [gfun_psi] at this

lemma psi_eq (k : ℕ) {v : ℝ} (hv : 0 ≤ v) (hv1 : v ≤ 1) :
    psi k v + epsk k * (psi k v) ^ (k + 2) = v := by
  have h := gfun_psi k v
  rw [gfun_of_mem k (psi_nonneg k hv) ((psi_le k v).trans hv1)] at h
  linarith

lemma contDiff_psi (k : ℕ) : ContDiff ℝ (k + 1) (psi k) := by
  have h := Homeomorph.contDiff_symm_deriv (gIso k).toHomeomorph
    (f' := fun x => epsk k * deriv (Pk k) x + 1)
    (fun x => by have := half_le_deriv_gfun k x; positivity)
    (fun x => hasDerivAt_gfun k x) (contDiff_gfun k)
  exact h

end GKLS
