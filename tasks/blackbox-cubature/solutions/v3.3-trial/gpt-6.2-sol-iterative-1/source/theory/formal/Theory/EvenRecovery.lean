import Theory.Antipodal

open Set MeasureTheory
open scoped BigOperators Pointwise

namespace Cubature

abbrev Radii (d : ℕ) := Bool × RayIndex d → ℝ

noncomputable def sortedRadii {d : ℕ} (r : Radii d) (i : Bool × RayIndex d) : ℝ :=
  if i.1 then max (r (true,i.2)) (r (false,i.2))
  else min (r (true,i.2)) (r (false,i.2))

theorem sortedRadii_mem {d : ℕ} (r : Radii d) (i : Bool × RayIndex d)
    (hr : ∀ j, r j ∈ Icc (1/2 : ℝ) 4) : sortedRadii r i ∈ Icc (1/2 : ℝ) 4 := by
  obtain ⟨b,k⟩ := i
  rcases le_total (r (true,k)) (r (false,k)) with H | H
  · cases b
    · simpa [sortedRadii,min_eq_left H] using hr (true,k)
    · simpa [sortedRadii,max_eq_right H] using hr (false,k)
  · cases b
    · simpa [sortedRadii,min_eq_right H] using hr (false,k)
    · simpa [sortedRadii,max_eq_left H] using hr (true,k)

theorem sortedRadii_product {d : ℕ} (r : Radii d) (i : RayIndex d) :
    recoveredCurvature (sortedRadii r) i = recoveredCurvature r i := by
  rcases le_total (r (true,i)) (r (false,i)) with H | H
  · simp [recoveredCurvature,sortedRadii,max_eq_right H,min_eq_left H,mul_comm]
  · simp [recoveredCurvature,sortedRadii,max_eq_left H,min_eq_right H]

def rootWidth {d : ℕ} (r : Radii d) (i : RayIndex d) : ℝ :=
  r (true,i)-r (false,i)

theorem linear_pair {d : ℕ} (b : Point d) (i j : Fin d) :
    linear b (axis i + axis j) = b i+b j := by
  simp [linear, Pi.add_apply, mul_add, Finset.sum_add_distrib, axis, mul_ite]

theorem recover_even_exact {d : ℕ} (E : Ellipsoid d) (r : Radii d)
    (hr : ∀ i, 0 < r i)
    (he : ∀ i, (r i)^2 = quadratic E.A (signedRay i) + linear E.b (signedRay i)*r i) :
    recoveredMatrix (sortedRadii r) = E.A ∧
      ∀ i, rootWidth (sortedRadii r) i = |linear E.b (ray i)| := by
  have H := (recover_exact E r hr he).1
  constructor
  · rw [← H]
    funext i j
    simp only [recoveredMatrix,sortedRadii_product]
  · intro i
    have h₁ : (r (true,i))^2 = quadratic E.A (ray i) + linear E.b (ray i)*r (true,i) := by
      simpa [signedRay] using he (true,i)
    have h₂ : (r (false,i))^2 = quadratic E.A (ray i) - linear E.b (ray i)*r (false,i) := by
      simpa [signedRay,quadratic_neg,linear_neg,sub_eq_add_neg] using he (false,i)
    have hl := (opposite_inverse_radii (hr _) (hr _) h₁ h₂).2
    rw [hl]
    rcases le_total (r (true,i)) (r (false,i)) with H | H
    · simp [rootWidth,sortedRadii,max_eq_right H,min_eq_left H,abs_of_nonpos (sub_nonpos.mpr H)]
    · simp [rootWidth,sortedRadii,max_eq_left H,min_eq_right H,abs_of_nonneg (sub_nonneg.mpr H)]

theorem exists_bestAxis {d : ℕ} (hd : 0 < d) (s : Radii d) :
    ∃ k : Fin d, ∀ i, rootWidth s (rayIndex i i) ≤ rootWidth s (rayIndex k k) := by
  have hne : (Finset.univ : Finset (Fin d)).Nonempty :=
    ⟨⟨0,hd⟩,Finset.mem_univ _⟩
  obtain ⟨j,_,hj⟩ := Finset.exists_max_image (Finset.univ : Finset (Fin d))
    (fun i ↦ rootWidth s (rayIndex i i)) hne
  exact ⟨j,fun i ↦ hj i (Finset.mem_univ i)⟩

noncomputable def bestAxis {d : ℕ} (hd : 0 < d) (s : Radii d) : Fin d :=
  Classical.choose (exists_bestAxis hd s)

noncomputable def pivotLinear {d : ℕ} (hd : 0 < d) (s : Radii d) (i : Fin d) : ℝ :=
  let k := bestAxis hd s
  if i=k then rootWidth s (rayIndex k k)
  else rootWidth s (rayIndex i k)-rootWidth s (rayIndex k k)

theorem pivot_recovery {d : ℕ} (hd : 0 < d) (E : Ellipsoid d) (s r : Radii d) {ε : ℝ}
    (hε : 0 ≤ ε) (hc : ∀ i, |s i-r i| ≤ ε)
    (hex : ∀ i, rootWidth r i = |linear E.b (ray i)|) :
    ∃ E' : Ellipsoid d, (E'=E ∨ E'=reflected E) ∧
      E'.A=E.A ∧ ∀ i, |pivotLinear hd s i-E'.b i| ≤ 12*ε := by
  let k := bestAxis hd s
  let E' := if 0 ≤ E.b k then E else reflected E
  have hcases : E'=E ∨ E'=reflected E := by unfold E'; split <;> simp
  have hA : E'.A=E.A := by
    rcases hcases with H | H <;> simp [H,reflected]
  have hp : 0 ≤ E'.b k := by
    unfold E'
    split
    · assumption
    · simp only [reflected,Pi.neg_apply,neg_nonneg]
      linarith
  have hm (x : Point d) : |linear E'.b x| = |linear E.b x| := by
    rcases hcases with H | H
    · rw [H]
    · simp [H,reflected,linear_neg_coeffs]
  have hspan (i : RayIndex d) : |rootWidth s i - (|linear E'.b (ray i)|)| ≤ 2*ε := by
    rw [hm,← hex]
    have h1 := abs_le.mp (hc (true,i))
    have h2 := abs_le.mp (hc (false,i))
    unfold rootWidth
    exact abs_le.mpr ⟨by linarith,by linarith⟩
  have hk (i : Fin d) : rootWidth s (rayIndex i i) ≤ rootWidth s (rayIndex k k) :=
    Classical.choose_spec (exists_bestAxis hd s) i
  have haa (i : Fin d) : |rootWidth s (rayIndex i i) - (|E'.b i|)| ≤ 2*ε := by
    simpa [diagonal_ray,linear_axis] using hspan (rayIndex i i)
  have hdom (i : Fin d) : |E'.b i| ≤ E'.b k+4*ε := by
    have hi := (abs_le.mp (haa i)).1
    have hk' := (abs_le.mp (haa k)).2
    rw [abs_of_nonneg hp] at hk'
    linarith [hk i]
  refine ⟨E',hcases,hA,?_⟩
  intro i
  change |(if i=k then rootWidth s (rayIndex k k)
    else rootWidth s (rayIndex i k)-rootWidth s (rayIndex k k))-E'.b i| ≤ 12*ε
  by_cases h : i=k
  · rw [ite_eq_left h, h]
    refine (le_of_eq ?_).trans ((haa k).trans (by linarith))
    rw [abs_of_nonneg hp]
  · rw [ite_eq_right h]
    have hab := abs_le.mp (hspan (rayIndex i k))
    rw [mixed_ray h,linear_pair] at hab
    have hka := abs_le.mp (haa k)
    rw [abs_of_nonneg hp] at hka
    have hmin : -4*ε ≤ E'.b i+E'.b k := by linarith [neg_abs_le (E'.b i),hdom i]
    have haux : |E'.b i+E'.b k| ≤ E'.b i+E'.b k+8*ε := by
      by_cases ha : 0 ≤ E'.b i+E'.b k
      · rw [abs_of_nonneg ha]; linarith
      · rw [abs_of_nonpos (le_of_not_ge ha)]; linarith
    exact abs_le.mpr ⟨by linarith [le_abs_self (E'.b i+E'.b k)],by linarith⟩

theorem paired_polynomial_stable {d : ℕ} {A C : Coefficients d} {b c x : Point d} {ε : ℝ}
    (he : 0 ≤ ε) (hx : normSq x ≤ 3)
    (hA : ∀ i j, |C i j-A i j| ≤ 12*ε) (hb : ∀ i, |c i-b i| ≤ 12*ε) :
    |polynomial C c x-polynomial A b x| ≤ 64*d*ε := by
  have hn (i : Fin d) : |x i| ≤ 2 := by
    have hh := (sq_le_normSq x i).trans hx
    nlinarith [sq_abs (x i), abs_nonneg (x i)]
  have hq := polynomial_stable (C := C) (A := A) (c := 0) (b := 0)
    he hx hA (fun _ ↦ by simp [he])
  have hzero : linear (0 : Point d) x = 0 := by simp [linear]
  simp only [polynomial,hzero,add_zero] at hq
  have hr : linear c x-linear b x = ∑ i, (c i-b i)*x i := by
    simp only [linear,← Finset.sum_sub_distrib,sub_mul]
  have Hl : |linear c x-linear b x| ≤ 24*d*ε := by
    rw [hr]
    calc
      _ ≤ ∑ i, |(c i-b i)*x i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin d, (24*ε) := by
        apply Finset.sum_le_sum
        intro i _
        rw [abs_mul]
        have h := mul_le_mul (hb i) (hn i) (abs_nonneg _) (by positivity : 0 ≤ 12*ε)
        linarith
      _ = _ := by simp; ring
  have hw : polynomial C c x-polynomial A b x =
      (quadratic C x-quadratic A x)+(linear c x-linear b x) := by unfold polynomial; ring
  rw [hw]
  exact (abs_add_le _ _).trans (by linarith)

noncomputable def taggedRay {d : ℕ} (ρ : ℝ) (m : ℕ) (i : Bool × RayIndex d) :
    QueryTree (Bool × Point d) ℝ :=
  (bisect ρ (ray i.2) m (1/2) 4).mapPoints (fun x ↦ (i.1,x))

/-- The tree's queries are pairs of real oracle calls, evaluated by `pairedBit`. -/
noncomputable def antipodalTree {d : ℕ} (hd : 0 < d) (ρ : ℝ) (m : ℕ) :
    QueryTree (Bool × Point d) ℝ :=
  (QueryTree.collect (taggedRay ρ m)
    (Finset.univ : Finset (Bool × RayIndex d)).toList).map
      (fun s ↦ probability (clippedRegion ρ (recoveredMatrix s) (pivotLinear hd s)))

theorem antipodalTree_depth {d : ℕ} (hd : 0 < d) (ρ : ℝ) (m : ℕ) :
    (antipodalTree hd ρ m).depth ≤ d*(d+1)*m := by
  unfold antipodalTree
  refine (QueryTree.depth_map_le _ _).trans ((QueryTree.depth_collect _ _ m ?_).trans ?_)
  · intro i
    rw [taggedRay,QueryTree.depth_mapPoints,bisect_depth]
  · simp [Fintype.card_prod,card_rayIndex]

theorem antipodalTree_allowed {d : ℕ} (hd : 0 < d) {ρ : ℝ}
    (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20) (m : ℕ) : (antipodalTree hd ρ m).admissible pairAllowed := by
  unfold antipodalTree
  apply QueryTree.admissible_map
  apply QueryTree.admissible_collect
  intro i
  refine QueryTree.admissible_mapPoints _ _ allowed pairAllowed ?_
    (bisect_admissible hρ hρ' (ray_bound i.2) m le_rfl (by norm_num))
  intro x hx
  exact ⟨hx,fun j ↦ by simpa [Pi.neg_apply,abs_neg] using hx j⟩

theorem paired_threshold {d : ℕ} {ρ : ℝ} (hρ : ρ ≠ 0) (E : Ellipsoid d)
    (h : Point d → ℝ) (ho : ∀ x, h (-x) = -h x) (r : Radii d)
    (hr : ∀ i, 0 < r i)
    (he : ∀ i, (r i)^2 = quadratic E.A (signedRay i) + linear E.b (signedRay i)*r i)
    (i : Bool × RayIndex d) (t : ℝ) (ht : 0 < t) :
    pairedBit (augmentedPayoff ρ E h) (i.1,(ρ/t) • ray i.2) = true ↔ sortedRadii r i < t := by
  obtain ⟨j,k⟩ := i
  have hf (b : Bool) :
      (ρ/t) • signedRay (b,k) ∈ region ρ E ↔ r (b,k) < t :=
    (answerBit_iff _ _).symm.trans (inverse_radius_answer hρ E _
      (by linarith [(signedRay_norm (b,k)).1,E.low (signedRay (b,k))]) (hr _) (he _) _ ht)
  have hp : (ρ/t) • ray k ∈ region ρ E ↔ r (true,k) < t := by
    simpa [signedRay] using hf true
  have hn : -((ρ/t) • ray k) ∈ region ρ E ↔ r (false,k) < t := by
    simpa [signedRay,smul_neg] using hf false
  rw [pairedBit_iff ρ E h ho]
  cases j <;> simp [sortedRadii,hp,hn,min_lt_iff,max_lt_iff]

theorem antipodalTree_correct {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ) (m : ℕ)
    (hδ : oddErrorRadius d m < 1) (E : Ellipsoid d) (h : Point d → ℝ)
    (ho : ∀ x, h (-x) = -h x) :
    (1-oddErrorRadius d m)^d * probability (region ρ E) ≤
        (antipodalTree hd ρ m).run (pairedBit (augmentedPayoff ρ E h)) ∧
      (antipodalTree hd ρ m).run (pairedBit (augmentedPayoff ρ E h)) ≤
        (1+oddErrorRadius d m)^d * probability (region ρ E) := by
  classical
  let ε := 2/(2:ℝ)^m
  let δ := oddErrorRadius d m
  have hε : 0 ≤ ε := by dsimp [ε]; positivity
  have hδ0 : 0 ≤ δ := by unfold δ oddErrorRadius; positivity
  choose r hr he using fun i : Bool × RayIndex d =>
    inverse_radius_exists E (signedRay i) (signedRay_norm i)
  have hr' (i : Bool × RayIndex d) : 0 < r i := by linarith [(hr i).1]
  have H := recover_even_exact E r hr' he
  let f := pairedBit (augmentedPayoff ρ E h)
  let U := QueryTree.collect (taggedRay ρ m) (Finset.univ : Finset (Bool × RayIndex d)).toList
  let s := U.run f
  have hdata (i : Bool × RayIndex d) :
      s i ∈ Icc (1/2:ℝ) 4 ∧ |s i-sortedRadii r i| ≤ ε := by
    have hpoint := paired_threshold hρ.ne' E h ho r hr' he i
    have hmem := sortedRadii_mem r i hr
    have hb := bisect_correct ρ (ray i.2) (fun x ↦ f (i.1,x)) (sortedRadii r i)
      hpoint m (by norm_num) hmem.1 hmem.2
    have hs : s i = (taggedRay ρ m i).run f :=
      QueryTree.run_collect _ _ _ (by simp)
    rw [hs,taggedRay,QueryTree.run_mapPoints]
    exact ⟨hb.1,hb.2.trans (by
      rw [pow_succ,mul_comm ((2:ℝ)^m) (2:ℝ),← div_div]
      have H : (4-1/2:ℝ)/2 ≤ 2 := by norm_num
      exact div_le_div_of_nonneg_right H (pow_nonneg (by norm_num) _))⟩
  have Hclose i := (hdata i).2
  obtain ⟨E',hf,hA,hb⟩ := pivot_recovery hd E s (sortedRadii r) hε Hclose H.2
  have hq := (recover_stable s (sortedRadii r) hε (fun i ↦ (hdata i).1)
    (fun i ↦ sortedRadii_mem r i hr) Hclose).1
  rw [H.1,← hA] at hq
  obtain ⟨hin,hout⟩ := clipped_dilation ρ E' (recoveredMatrix s) (pivotLinear hd s) hδ0 hδ
    (fun _ hx ↦ paired_polynomial_stable hε hx hq hb)
  have hmass : probability (region ρ E) = probability (region ρ E') := by
    rcases hf with H | H
    · rw [H]
    · rw [H,reflected_probability]
  rw [antipodalTree,QueryTree.run_map,hmass]
  exact ⟨(probability_dilate_ge (by linarith : 0 < 1-δ) (by linarith : 1-δ ≤ 1)
    (measurable_region ρ E')).trans (probability_mono hin),
    (probability_mono hout).trans (probability_dilate_le (by linarith : 1 ≤ 1+δ)
      (measurable_region ρ E'))⟩

end Cubature
