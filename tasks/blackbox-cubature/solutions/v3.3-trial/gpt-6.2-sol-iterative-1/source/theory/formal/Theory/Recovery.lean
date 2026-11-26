import Theory.Geometry

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

noncomputable def signedRay {d : ℕ} (i : Bool × RayIndex d) : Point d :=
  if i.1 then ray i.2 else -ray i.2

theorem signedRay_bound {d : ℕ} (i : Bool × RayIndex d) (j : Fin d) :
    |signedRay i j| ≤ 1 := by
  obtain ⟨b,r⟩ := i
  cases b <;> simpa [signedRay] using ray_bound r j

theorem signedRay_norm {d : ℕ} (i : Bool × RayIndex d) :
    1 ≤ normSq (signedRay i) ∧ normSq (signedRay i) ≤ 2 := by
  have hn : normSq (-ray i.2) = normSq (ray i.2) := by
    simp [normSq]
  obtain ⟨b,r⟩ := i
  cases b <;> simpa [signedRay, hn] using ray_norm r

def recoveredCurvature {d : ℕ} (r : Bool × RayIndex d → ℝ) (i : RayIndex d) : ℝ :=
  r (true,i) * r (false,i)

def recoveredLinear {d : ℕ} (r : Bool × RayIndex d → ℝ) (i : Fin d) : ℝ :=
  r (true,rayIndex i i) - r (false,rayIndex i i)

noncomputable def recoveredMatrix {d : ℕ} (r : Bool × RayIndex d → ℝ) (i j : Fin d) : ℝ :=
  if i=j then recoveredCurvature r (rayIndex i i)
  else (recoveredCurvature r (rayIndex i j) -
        recoveredCurvature r (rayIndex i i) - recoveredCurvature r (rayIndex j j)) / 2

theorem recover_exact {d : ℕ} (E : Ellipsoid d)
    (r : Bool × RayIndex d → ℝ) (hr : ∀ i, 0 < r i)
    (he : ∀ i, (r i)^2 = quadratic E.A (signedRay i) + linear E.b (signedRay i)*r i) :
    recoveredMatrix r = E.A ∧ recoveredLinear r = E.b := by
  have H (i : RayIndex d) :
      recoveredCurvature r i = quadratic E.A (ray i) ∧
      r (true,i) - r (false,i) = linear E.b (ray i) := by
    have h₁ : (r (true,i))^2 = quadratic E.A (ray i) + linear E.b (ray i)*r (true,i) := by
      simpa [signedRay] using he (true,i)
    have h₂ : (r (false,i))^2 = quadratic E.A (ray i) - linear E.b (ray i)*r (false,i) := by
      have hq : quadratic E.A (-ray i) = quadratic E.A (ray i) := by
        simpa [neg_smul] using quadratic_smul E.A (-1) (ray i)
      have hl : linear E.b (-ray i) = -linear E.b (ray i) := by
        simpa [neg_smul] using linear_smul E.b (-1) (ray i)
      simpa [signedRay, hq, hl, sub_eq_add_neg] using he (false,i)
    exact ⟨(opposite_inverse_radii (hr _) (hr _) h₁ h₂).1.symm,
      (opposite_inverse_radii (hr _) (hr _) h₁ h₂).2.symm⟩
  have Hdiag (i : Fin d) : recoveredCurvature r (rayIndex i i) = E.A i i := by
    rw [(H _).1, diagonal_ray, quadratic_axis]
  constructor
  · funext i j
    by_cases h : i = j
    · simpa [recoveredMatrix, h] using Hdiag j
    · rw [recoveredMatrix, if_neg h, Hdiag i, Hdiag j, (H _).1, mixed_ray h,
        quadratic_two_axes]
      ring
  · funext i
    rw [recoveredLinear, (H _).2, diagonal_ray, linear_axis]

theorem recover_stable {d : ℕ} (s r : Bool × RayIndex d → ℝ) {ε : ℝ}
    (he : 0 ≤ ε) (hs : ∀ i, s i ∈ Icc (1/2 : ℝ) 4) (hr : ∀ i, r i ∈ Icc (1/2 : ℝ) 4)
    (hclose : ∀ i, |s i-r i| ≤ ε) :
    (∀ i j, |recoveredMatrix s i j-recoveredMatrix r i j| ≤ 12*ε) ∧
    (∀ i, |recoveredLinear s i-recoveredLinear r i| ≤ 2*ε) := by
  have hk (i : RayIndex d) : |recoveredCurvature s i-recoveredCurvature r i| ≤ 8*ε := by
    have hn : recoveredCurvature s i-recoveredCurvature r i =
        (s (true,i)-r (true,i))*s (false,i) + r (true,i)*(s (false,i)-r (false,i)) := by
      unfold recoveredCurvature
      ring
    rw [hn]
    apply (abs_add_le _ _).trans
    rw [abs_mul, abs_mul, abs_of_nonneg (by linarith [(hs (false,i)).1] : 0 ≤ s (false,i)),
      abs_of_nonneg (by linarith [(hr (true,i)).1] : 0 ≤ r (true,i))]
    have h1 := mul_le_mul (hclose (true,i)) (hs (false,i)).2
      (by linarith [(hs (false,i)).1]) he
    have h2 := mul_le_mul (hr (true,i)).2 (hclose (false,i)) (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 4)
    linarith
  constructor
  · intro i j
    by_cases hij : i=j
    · simp only [recoveredMatrix, if_pos hij]
      exact (hk _).trans (by linarith)
    · simp only [recoveredMatrix, if_neg hij]
      have h₁ := abs_le.mp (hk (rayIndex i j))
      have h₂ := abs_le.mp (hk (rayIndex i i))
      have h₃ := abs_le.mp (hk (rayIndex j j))
      exact abs_le.mpr ⟨by linarith, by linarith⟩
  · intro i
    have h₁ := abs_le.mp (hclose (true,rayIndex i i))
    have h₂ := abs_le.mp (hclose (false,rayIndex i i))
    simp only [recoveredLinear]
    exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem sum_abs_cross {d : ℕ} (x : Point d) :
    (∑ i, ∑ j, |x i| * |x j|) ≤ d * normSq x := by
  calc
    _ ≤ ∑ i : Fin d, ∑ j : Fin d, (x i^2+x j^2)/2 := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      have hh := sq_nonneg (|x i|-|x j|)
      nlinarith [sq_abs (x i),sq_abs (x j)]
    _ = _ := by
      simp_rw [div_eq_mul_inv, ← Finset.sum_mul, Finset.sum_add_distrib, Finset.sum_const,
        Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      rw [← Finset.mul_sum]
      simp only [normSq]
      ring

theorem polynomial_stable {d : ℕ} {A C : Coefficients d} {b c x : Point d} {ε : ℝ}
    (he : 0 ≤ ε) (hx : normSq x ≤ 3)
    (hA : ∀ i j, |C i j-A i j| ≤ 12*ε) (hb : ∀ i, |c i-b i| ≤ 2*ε) :
    |polynomial C c x-polynomial A b x| ≤ 40*d*ε := by
  have hxb (i : Fin d) : |x i| ≤ 2 := by
    have hh := (sq_le_normSq x i).trans hx
    nlinarith [sq_abs (x i), abs_nonneg (x i)]
  have hq : |quadratic C x-quadratic A x| ≤ 36*d*ε := by
    have h : quadratic C x-quadratic A x = ∑ i, ∑ j, (C i j-A i j) * x i * x j := by
      simp only [quadratic, ← Finset.sum_sub_distrib]
      congr 1
      funext i
      congr 1
      funext j
      ring
    rw [h]
    calc
      _ ≤ ∑ i, ∑ j, |(C i j-A i j)*x i*x j| := by
        apply (Finset.abs_sum_le_sum_abs _ _).trans
        exact Finset.sum_le_sum (fun i _ ↦ Finset.abs_sum_le_sum_abs _ _)
      _ ≤ ∑ i, ∑ j, (12*ε)*(|x i| * |x j|) := by
        apply Finset.sum_le_sum
        intro i _
        apply Finset.sum_le_sum
        intro j _
        simp only [abs_mul]
        rw [mul_assoc]
        exact mul_le_mul_of_nonneg_right (hA i j) (mul_nonneg (abs_nonneg _) (abs_nonneg _))
      _ = (12*ε)*(∑ i, ∑ j, |x i| * |x j|) := by simp [Finset.mul_sum]
      _ ≤ _ := by
        have h' := mul_le_mul_of_nonneg_left (sum_abs_cross x) (show 0 ≤ 12*ε by positivity)
        have hx' := mul_le_mul_of_nonneg_left hx (show 0 ≤ 12*ε*d by positivity)
        linarith only [h', hx']
  have hl : |linear c x-linear b x| ≤ 4*d*ε := by
    have h : linear c x-linear b x = ∑ i, (c i-b i)*x i := by
      simp only [linear, ← Finset.sum_sub_distrib, sub_mul]
    rw [h]
    calc
      _ ≤ ∑ i, |(c i-b i)*x i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin d, (4*ε) := by
        refine Finset.sum_le_sum (fun i _ ↦ ?_)
        rw [abs_mul]
        have H := mul_le_mul (hb i) (hxb i) (abs_nonneg (x i)) (by positivity : 0 ≤ 2*ε)
        linarith
      _ = _ := by simp; ring
  have hh : polynomial C c x-polynomial A b x =
      (quadratic C x-quadratic A x)+(linear c x-linear b x) := by unfold polynomial; ring
  rw [hh]
  exact (abs_add_le _ _).trans (by linarith)

noncomputable def clippedRegion {d : ℕ} (ρ : ℝ) (A : Coefficients d) (b : Point d) :
    Set (Point d) :=
  {x | normSq (ρ⁻¹ • x) ≤ 3 ∧ polynomial A b (ρ⁻¹ • x) < 1}

theorem clipped_dilation {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (A : Coefficients d) (b : Point d)
    {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ < 1)
    (h : ∀ x, normSq x ≤ 3 → |polynomial A b x-polynomial E.A E.b x| ≤ δ) :
    (1-δ) • region ρ E ⊆ clippedRegion ρ A b ∧
    clippedRegion ρ A b ⊆ (1+δ) • region ρ E := by
  have h0 : 0 < 1-δ := by linarith
  have h1 : 0 < 1+δ := by linarith
  constructor
  · intro x hx
    obtain ⟨y, hy, rfl⟩ := hx
    have hey : polynomial E.A E.b (ρ⁻¹ • y) < 1 := hy
    have shr : ρ⁻¹ • ((1-δ) • y) = (1-δ) • (ρ⁻¹ • y) := smul_comm _ _ _
    have hn := inside_normSq E hey
    have hnorm : normSq (ρ⁻¹ • ((1-δ) • y)) ≤ 3 := by
      rw [shr, normSq_smul]
      have hq : (1-δ)^2 ≤ 1 := by nlinarith [mul_nonneg hδ0 (sub_nonneg.mpr hδ.le)]
      nlinarith [mul_le_mul_of_nonneg_right hq (normSq_nonneg (ρ⁻¹ • y))]
    have hb := polynomial_contract E (ρ⁻¹ • y) h0.le (by linarith : 1-δ ≤ 1)
    have hf := (abs_le.mp (h _ hnorm)).2
    refine ⟨hnorm, ?_⟩
    rw [shr] at hf ⊢
    have hmul := mul_lt_mul_of_pos_left hey h0
    linarith
  · intro x hx
    have hx' : normSq (ρ⁻¹ • x) ≤ 3 ∧ polynomial A b (ρ⁻¹ • x) < 1 := hx
    have hb : polynomial E.A E.b (ρ⁻¹ • x) < 1+δ := by
      have hs := (abs_le.mp (h _ hx'.1)).1
      linarith [hx'.2]
    have hpos : 0 < (1+δ)⁻¹ := inv_pos.mpr h1
    have hon : (1+δ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (by linarith)
    refine ⟨(1+δ)⁻¹ • x, ?_, ?_⟩
    swap
    · change (1+δ) • ((1+δ)⁻¹ • x) = x
      rw [smul_smul, mul_inv_cancel₀ h1.ne', one_smul]
    change polynomial E.A E.b (ρ⁻¹ • ((1+δ)⁻¹ • x)) < 1
    rw [smul_comm (ρ⁻¹ : ℝ) ((1+δ)⁻¹) x]
    refine (polynomial_contract E _ hpos.le hon).trans_lt ?_
    simpa [inv_mul_cancel₀ h1.ne'] using mul_lt_mul_of_pos_left hb hpos

/-- Run a bisection along both signs of each direction and integrate the surrogate. -/
noncomputable def ellipsoidTree (d : ℕ) (ρ : ℝ) (m : ℕ) : QueryTree (Point d) ℝ :=
  (QueryTree.collect (fun i : Bool × RayIndex d ↦ bisect ρ (signedRay i) m (1/2) 4)
    (Finset.univ : Finset (Bool × RayIndex d)).toList).map
      (fun r ↦ probability (clippedRegion ρ (recoveredMatrix r) (recoveredLinear r)))

theorem ellipsoidTree_depth (d : ℕ) (ρ : ℝ) (m : ℕ) :
    (ellipsoidTree d ρ m).depth ≤ d*(d+1)*m := by
  unfold ellipsoidTree
  refine (QueryTree.depth_map_le _ _).trans ((QueryTree.depth_collect _ _ m ?_).trans ?_)
  · intro i
    rw [bisect_depth]
  · simp [Fintype.card_prod, card_rayIndex]

theorem ellipsoidTree_allowed (d : ℕ) {ρ : ℝ} (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20) (m : ℕ) :
    (ellipsoidTree d ρ m).admissible allowed := by
  unfold ellipsoidTree
  apply QueryTree.admissible_map
  apply QueryTree.admissible_collect
  intro i
  exact bisect_admissible hρ hρ' (signedRay_bound i) m le_rfl (by norm_num)

theorem ellipsoidTree_correct {d : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (m : ℕ)
    (hδ : errorRadius d m < 1) (E : Ellipsoid d) :
    (1-errorRadius d m)^d * probability (region ρ E) ≤
        (ellipsoidTree d ρ m).run (answerBit (region ρ E)) ∧
      (ellipsoidTree d ρ m).run (answerBit (region ρ E)) ≤
        (1+errorRadius d m)^d * probability (region ρ E) := by
  classical
  let δ := errorRadius d m
  let ε := 2 / (2:ℝ)^m
  have hε : 0 ≤ ε := by dsimp [ε]; positivity
  have hδ0 : 0 ≤ δ := by unfold δ errorRadius; positivity
  choose r hr he using fun i : Bool × RayIndex d =>
    inverse_radius_exists E (signedRay i) (signedRay_norm i)
  have hrex := recover_exact E r (fun i ↦ by have H := (hr i).1; linarith) he
  let U := QueryTree.collect (fun i : Bool × RayIndex d ↦ bisect ρ (signedRay i) m (1/2) 4)
    (Finset.univ : Finset (Bool × RayIndex d)).toList
  let s := U.run (answerBit (region ρ E))
  have H (i : Bool × RayIndex d) :
      s i ∈ Icc (1/2:ℝ) 4 ∧ |s i-r i| ≤ ε := by
    have hpoint (t : ℝ) (ht : 0 < t) :
        answerBit (region ρ E) ((ρ/t) • signedRay i) = true ↔ r i < t :=
      inverse_radius_answer hρ.ne' E _ (by linarith [(signedRay_norm i).1, E.low (signedRay i)])
        (by linarith [(hr i).1]) (he i) t ht
    have Hb := bisect_correct ρ (signedRay i) _ (r i) hpoint m (by norm_num) (hr i).1 (hr i).2
    have hs : s i = (bisect ρ (signedRay i) m (1/2) 4).run (answerBit (region ρ E)) :=
      QueryTree.run_collect _ _ _ (by simp)
    rw [hs]
    exact ⟨Hb.1, Hb.2.trans (by
      rw [pow_succ]
      dsimp [ε]
      rw [mul_comm ((2:ℝ)^m) (2:ℝ)]
      rw [← div_div]
      have H : (4-1/2:ℝ)/2 ≤ 2 := by norm_num
      exact div_le_div_of_nonneg_right H (pow_nonneg (by norm_num) _))⟩
  have hrec := recover_stable s r hε (fun i ↦ (H i).1) hr (fun i ↦ (H i).2)
  obtain ⟨hin,hout⟩ := clipped_dilation ρ E (recoveredMatrix s) (recoveredLinear s) hδ0 hδ
    (fun x hx => polynomial_stable hε hx (by rw [hrex.1] at hrec; exact hrec.1)
      (by rw [hrex.2] at hrec; exact hrec.2))
  rw [ellipsoidTree, QueryTree.run_map]
  exact ⟨(probability_dilate_ge (by linarith : 0 < 1-δ) (by linarith : 1-δ ≤ 1)
    (measurable_region ρ E)).trans (probability_mono hin),
    (probability_mono hout).trans (probability_dilate_le (by linarith : 1 ≤ 1+δ)
      (measurable_region ρ E))⟩

end Cubature
