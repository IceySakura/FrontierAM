import Theory.LipRay

open MeasureTheory Set
open scoped Pointwise

namespace Cubature

noncomputable def learnGeometry (d : ℕ) {k : ℕ} (hk : 0 < k) (ρ : ℝ) (m : ℕ) :
    ValueTree (Point d) (Bool × RayIndex d → ℝ) :=
  ValueTree.collect (fun i : Bool × RayIndex d ↦ learnRay ρ (signedRay i) hk m)
    (Finset.univ : Finset (Bool × RayIndex d)).toList

theorem learnGeometry_within (d : ℕ) {k : ℕ} (hk : 0 < k) (ρ : ℝ) (m : ℕ) :
    (learnGeometry d hk ρ m).within (d*(d+1)*(k+1+m)) := by
  unfold learnGeometry
  have H := ValueTree.within_collect
    (fun i : Bool × RayIndex d ↦ learnRay ρ (signedRay i) hk m)
    (Finset.univ : Finset (Bool × RayIndex d)).toList (k+1+m)
    (fun _ => learnRay_within _ _ hk m)
  simpa [Fintype.card_prod,card_rayIndex] using H

theorem learnGeometry_allowed (d : ℕ) {k : ℕ} (hk : 0 < k) {ρ : ℝ}
    (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20) (m : ℕ) :
    (learnGeometry d hk ρ m).admissible allowed :=
  ValueTree.admissible_collect _ _ _ (fun i => learnRay_allowed hρ hρ' (signedRay_bound i) hk m)

theorem learnGeometry_stable {d k : ℕ} (hk : 0 < k) {ρ L : ℝ}
    (hρ : 0 < ρ) (hL : 0 ≤ L) (hscale : 64*L*ρ ≤ k)
    (E : Ellipsoid d) (h : Point d → ℝ) (hlip : CoordinateLipschitz L h) (m : ℕ) :
    let s := (learnGeometry d hk ρ m).run (augmentedPayoff ρ E h)
    ∀ x, normSq x ≤ 3 →
      |polynomial (recoveredMatrix s) (recoveredLinear s) x - polynomial E.A E.b x| ≤
        errorRadius d m := by
  classical
  intro s x hx
  choose r hr he using fun i : Bool × RayIndex d =>
    inverse_radius_exists E (signedRay i) (signedRay_norm i)
  have hrex := recover_exact E r (fun i ↦ by linarith [(hr i).1]) he
  have HH (i : Bool × RayIndex d) :
      s i ∈ Icc (1/2:ℝ) 4 ∧ |s i-r i| ≤ 2/(2:ℝ)^m := by
    have hi : s i = (learnRay ρ (signedRay i) hk m).run (augmentedPayoff ρ E h) :=
      ValueTree.run_collect _ _ _ (by simp)
    rw [hi]
    exact learnRay_correct hk hρ hL hscale E _ (signedRay_bound i) (signedRay_norm i)
      h hlip m (hr i) (he i)
  have hh := recover_stable s r (by positivity : 0 ≤ 2/(2:ℝ)^m)
    (fun i => (HH i).1) hr (fun i => (HH i).2)
  rw [hrex.1,hrex.2] at hh
  exact polynomial_stable (by positivity : 0 ≤ 2/(2:ℝ)^m) hx hh.1 hh.2

/-- After geometric training, obtain a background value with at most one
extra call. Queries for a known phase need only the point itself. -/
noncomputable def readBackground {d : ℕ} (ρ : ℝ) (A : Coefficients d) (b : Point d)
    (δ : ℝ) (x : Point d) : ValueTree (Point d) ℝ :=
  .ask x (fun f =>
    let u := ρ⁻¹ • x
    let q := polynomial A b u
    if 3 < normSq u then .ret f
    else if q < 1-δ then .ret (f-1)
    else if 1+δ ≤ q then .ret f
    else .ask ((1+2*δ)⁻¹ • x) (fun g =>
      .ret (if -(1/2 : ℝ) < f-g then f-1 else f)))

theorem readBackground_within {d : ℕ} (ρ : ℝ) (A : Coefficients d) (b : Point d)
    (δ : ℝ) (x : Point d) : (readBackground ρ A b δ x).within 2 := by
  intro f
  simp only [readBackground] at *
  split <;> try trivial
  split <;> try trivial
  split <;> try trivial
  exact fun _ => trivial

theorem readBackground_allowed {d : ℕ} (ρ : ℝ) (A : Coefficients d) (b : Point d)
    {δ : ℝ} (hδ : 0 ≤ δ) {x : Point d} (hx : allowed x) :
    (readBackground ρ A b δ x).admissible allowed := by
  refine ⟨hx,?_⟩
  intro f
  simp only [readBackground] at *
  split <;> try trivial
  split <;> try trivial
  split <;> try trivial
  have ht : 0 ≤ (1+2*δ)⁻¹ := by positivity
  have ht' : (1+2*δ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (by linarith)
  refine ⟨?_,fun _ => trivial⟩
  intro i
  simp only [Pi.smul_apply,smul_eq_mul,abs_mul,abs_of_nonneg ht]
  simpa using mul_le_mul ht' (hx i) (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 1)

theorem payoff_inside_value {d : ℕ} {ρ : ℝ} {E : Ellipsoid d} {x : Point d}
    (hx : x ∈ region ρ E) : payoff ρ E x = 1 := by
  simp [payoff,answerBit,hx]

theorem payoff_outside_value {d : ℕ} {ρ : ℝ} {E : Ellipsoid d} {x : Point d}
    (hx : x ∉ region ρ E) : payoff ρ E x = 0 := by
  simp [payoff,answerBit,hx]

theorem readBackground_correct {d : ℕ} {ρ L : ℝ} (hρ : 0 < ρ) (hL : 0 ≤ L)
    (E : Ellipsoid d) (h : Point d → ℝ) (hlip : CoordinateLipschitz L h)
    (A : Coefficients d) (b : Point d) {δ : ℝ} (hδ : 0 ≤ δ) (htiny : 16*L*ρ*δ ≤ 1)
    (hfit : ∀ u, normSq u ≤ 3 → |polynomial A b u - polynomial E.A E.b u| ≤ δ)
    (x : Point d) :
    (readBackground ρ A b δ x).run (augmentedPayoff ρ E h) = h x := by
  let u := ρ⁻¹ • x
  let p := polynomial A b u
  let f := augmentedPayoff ρ E h
  have hm (y : Point d) : y ∈ region ρ E ↔ polynomial E.A E.b (ρ⁻¹ • y) < 1 := Iff.rfl
  have f_yes (y : Point d) (hy : y ∈ region ρ E) : f y = h y + 1 := by
    simp only [f,augmentedPayoff,payoff_inside_value hy]
  have f_no (y : Point d) (hy : y ∉ region ρ E) : f y = h y := by
    simp only [f,augmentedPayoff,payoff_outside_value hy,add_zero]
  change (if 3 < normSq u then ValueTree.ret (f x)
    else if p < 1-δ then ValueTree.ret (f x-1)
    else if 1+δ ≤ p then ValueTree.ret (f x)
    else ValueTree.ask ((1+2*δ)⁻¹ • x) (fun g =>
      ValueTree.ret (if -(1/2 : ℝ) < f x-g then f x-1 else f x))).run f = h x
  by_cases hnorm : 3<normSq u
  · rw [ite_eq_left hnorm,ValueTree.run]
    exact f_no x (fun hx => not_lt_of_ge (inside_normSq E ((hm x).1 hx)) hnorm)
  rw [ite_eq_right hnorm]
  have hclose := abs_le.mp (hfit u (le_of_not_gt hnorm))
  change -(δ) ≤ p - polynomial E.A E.b u ∧ p - polynomial E.A E.b u ≤ δ at hclose
  by_cases hin : p<1-δ
  · rw [ite_eq_left hin,ValueTree.run,f_yes x ((hm x).2 (by linarith [hclose.1])),add_sub_cancel_right]
  rw [ite_eq_right hin]
  by_cases hout : 1+δ ≤ p
  · rw [ite_eq_left hout,ValueTree.run]
    exact f_no x (fun hx => by have H := (hm x).1 hx; change polynomial E.A E.b u < 1 at H; linarith [hclose.2])
  rw [ite_eq_right hout]
  let t := (1+2*δ)⁻¹
  let y := t • x
  have ht0 : 0 < t := by dsimp [t]; positivity
  have ht1 : t ≤ 1 := inv_le_one_of_one_le₀ (by linarith)
  have heq : t*(1+2*δ) = 1 := inv_mul_cancel₀ (by linarith)
  have hs : 1-t ≤ 2*δ := by
    linarith [mul_le_mul_of_nonneg_left ht1 (show 0 ≤ 2*δ by positivity)]
  have hp : polynomial E.A E.b u < 1+2*δ := by linarith [lt_of_not_ge hout,hclose.1]
  have hy : y ∈ region ρ E := by
    apply (hm y).2
    change polynomial E.A E.b (ρ⁻¹ • (t • x)) < 1
    rw [smul_comm (ρ⁻¹ : ℝ) t x]
    refine (polynomial_contract E u ht0.le ht1).trans_lt ?_
    simpa [heq] using mul_lt_mul_of_pos_left hp ht0
  have hxx (i : Fin d) : |x i| ≤ 2*ρ := by
    have H := (sq_le_normSq u i).trans (le_of_not_gt hnorm)
    have h0 : |u i| ≤ 2 := by nlinarith [sq_abs (u i),abs_nonneg (u i)]
    have hsm : ρ • u = x := by dsimp [u]; rw [smul_smul,mul_inv_cancel₀ hρ.ne',one_smul]
    have hx' := congrArg (fun v : Point d => v i) hsm
    change ρ*u i=x i at hx'
    rw [← hx',abs_mul,abs_of_nonneg hρ.le]
    nlinarith [mul_le_mul_of_nonneg_left h0 hρ.le]
  have hDist (i : Fin d) : |x i-y i| ≤ 4*ρ*δ := by
    change |x i-t*x i| ≤ _
    rw [show x i-t*x i=(1-t)*x i by ring,abs_mul,abs_of_nonneg (sub_nonneg.mpr ht1)]
    nlinarith only [mul_le_mul hs (hxx i) (abs_nonneg _) (show 0 ≤ 2*δ by positivity)]
  have hfd := hlip x y _ (by positivity : 0 ≤ 4*ρ*δ) hDist
  have hdif : |h x-h y| ≤ 1/4 := hfd.trans (by nlinarith only [htiny])
  have hdb := abs_le.mp hdif
  change (if -(1/2 : ℝ) < f x-f y then f x-1 else f x) = h x
  rw [f_yes y hy]
  by_cases hb : x ∈ region ρ E
  · rw [f_yes x hb, ite_eq_left (by linarith [hdb.1]), add_sub_cancel_right]
  · rw [f_no x hb, ite_eq_right (by linarith [hdb.2])]

noncomputable def reductionTree (d : ℕ) {k n : ℕ}
    (hk : 0 < k) (ρ : ℝ) (m : ℕ) (z : Fin n → Point d) :
    ValueTree (Point d) (ℝ × (Fin n → ℝ)) :=
  (learnGeometry d hk ρ m).bind fun s =>
    let A := recoveredMatrix s
    let b := recoveredLinear s
    (ValueTree.collect (fun i ↦ readBackground ρ A b (errorRadius d m) (z i))
      (Finset.univ : Finset (Fin n)).toList).map
        (fun values ↦ (probability (clippedRegion ρ A b), values))

theorem reductionTree_within {d k n : ℕ} (hk : 0 < k) (ρ : ℝ)
    (m : ℕ) (z : Fin n → Point d) :
    (reductionTree d hk ρ m z).within (d*(d+1)*(k+1+m)+2*n) := by
  unfold reductionTree
  apply ValueTree.within_bind _ _ (learnGeometry_within d hk ρ m)
  intro s
  apply ValueTree.within_map
  have H := ValueTree.within_collect
    (fun i ↦ readBackground ρ (recoveredMatrix s) (recoveredLinear s) (errorRadius d m) (z i))
    (Finset.univ : Finset (Fin n)).toList 2 (fun _ => readBackground_within _ _ _ _ _)
  simpa [Nat.mul_comm] using H

theorem reductionTree_allowed {d k n : ℕ} (hk : 0 < k) {ρ : ℝ} (hρ : 0 ≤ ρ) (hρ' : ρ ≤ 20)
    (m : ℕ) (z : Fin n → Point d) (hz : ∀ i, allowed (z i)) :
    (reductionTree d hk ρ m z).admissible allowed := by
  unfold reductionTree
  apply ValueTree.admissible_bind _ _ _ (learnGeometry_allowed d hk hρ hρ' m)
  intro s
  apply ValueTree.admissible_map
  apply ValueTree.admissible_collect
  exact fun i => readBackground_allowed _ _ _ (by unfold errorRadius; positivity) (hz i)

theorem reductionTree_correct {d k n : ℕ} (hk : 0 < k) {ρ L : ℝ}
    (hρ : 0 < ρ) (hL : 0 ≤ L) (hscale : 64*L*ρ ≤ k)
    (m : ℕ) (hδ : errorRadius d m < 1) (htiny : 16*L*ρ*errorRadius d m ≤ 1)
    (z : Fin n → Point d) (E : Ellipsoid d) (h : Point d → ℝ)
    (hlip : CoordinateLipschitz L h) :
    let result := (reductionTree d hk ρ m z).run (augmentedPayoff ρ E h)
    (∀ j, result.2 j = h (z j)) ∧
      (1-errorRadius d m)^d*probability (region ρ E) ≤ result.1 ∧
      result.1 ≤ (1+errorRadius d m)^d*probability (region ρ E) := by
  let δ := errorRadius d m
  have hδ0 : 0 ≤ δ := by unfold δ errorRadius; positivity
  have hi := learnGeometry_stable hk hρ hL hscale E h hlip m
  let s := (learnGeometry d hk ρ m).run (augmentedPayoff ρ E h)
  let A := recoveredMatrix s
  let b := recoveredLinear s
  have hs : ∀ x, normSq x ≤ 3 → |polynomial A b x-polynomial E.A E.b x| ≤ δ := hi
  obtain ⟨hin,hout⟩ := clipped_dilation ρ E A b hδ0 hδ hs
  simp only [reductionTree,ValueTree.run_bind,ValueTree.run_map]
  refine ⟨?_,?_,?_⟩
  · intro j
    rw [ValueTree.run_collect _ _ _ (by simp : j ∈ (Finset.univ : Finset (Fin n)).toList)]
    exact readBackground_correct hρ hL E h hlip A b hδ0 htiny hs (z j)
  · exact (probability_dilate_ge (by linarith : 0 < 1-δ) (by linarith : 1-δ ≤ 1)
      (measurable_region ρ E)).trans (probability_mono hin)
  · exact (probability_mono hout).trans (probability_dilate_le (by linarith : 1 ≤ 1+δ)
      (measurable_region ρ E))

/-- Learn the geometry once, then give an arbitrary background algorithm
its exact value transcript. -/
noncomputable def liftCubature {d k : ℕ} (hk : 0 < k) (ρ : ℝ) (m : ℕ)
    (base : ValueTree (Point d) ℝ) : ValueTree (Point d) (ℝ × ℝ) :=
  (learnGeometry d hk ρ m).bind fun s =>
    let A := recoveredMatrix s
    let b := recoveredLinear s
    (base.replace (readBackground ρ A b (errorRadius d m))).map
      (fun val ↦ (probability (clippedRegion ρ A b),val))

theorem liftCubature_within {d k n : ℕ} (hk : 0 < k) (ρ : ℝ) (m : ℕ)
    (base : ValueTree (Point d) ℝ) (h : base.within n) :
    (liftCubature hk ρ m base).within (d*(d+1)*(k+1+m)+2*n) :=
  ValueTree.within_bind _ _ (learnGeometry_within d hk ρ m) (fun _ =>
    ValueTree.within_map _ _ (ValueTree.within_replace base _ h (fun _ => readBackground_within _ _ _ _ _)))

theorem liftCubature_allowed {d k : ℕ} (hk : 0 < k) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (hρ' : ρ ≤ 20) (m : ℕ) (base : ValueTree (Point d) ℝ) (hb : base.admissible allowed) :
    (liftCubature hk ρ m base).admissible allowed :=
  ValueTree.admissible_bind _ _ _ (learnGeometry_allowed d hk hρ hρ' m) (fun _ =>
    ValueTree.admissible_map _ _ _
      (ValueTree.admissible_replace base _ allowed allowed hb (fun _ h =>
        readBackground_allowed _ _ _ (by unfold errorRadius; positivity) h)))

theorem liftCubature_correct {d k : ℕ} (hk : 0 < k) {ρ L : ℝ}
    (hρ : 0 < ρ) (hL : 0 ≤ L) (hscale : 64*L*ρ ≤ k)
    (m : ℕ) (hδ : errorRadius d m < 1) (htiny : 16*L*ρ*errorRadius d m ≤ 1)
    (base : ValueTree (Point d) ℝ) (E : Ellipsoid d) (h : Point d → ℝ)
    (hlip : CoordinateLipschitz L h) :
    let result := (liftCubature hk ρ m base).run (augmentedPayoff ρ E h)
    result.2 = base.run h ∧
      (1-errorRadius d m)^d*probability (region ρ E) ≤ result.1 ∧
      result.1 ≤ (1+errorRadius d m)^d*probability (region ρ E) := by
  let δ := errorRadius d m
  have hδ0 : 0 ≤ δ := by unfold δ errorRadius; positivity
  let s := (learnGeometry d hk ρ m).run (augmentedPayoff ρ E h)
  let A := recoveredMatrix s
  let b := recoveredLinear s
  have hs : ∀ x, normSq x ≤ 3 → |polynomial A b x-polynomial E.A E.b x| ≤ δ :=
    learnGeometry_stable hk hρ hL hscale E h hlip m
  obtain ⟨hin,hout⟩ := clipped_dilation ρ E A b hδ0 hδ hs
  simp only [liftCubature,ValueTree.run_bind,ValueTree.run_map]
  refine ⟨?_,?_,?_⟩
  · exact ValueTree.run_replace base _ _ _ (fun x =>
      readBackground_correct hρ hL E h hlip A b hδ0 htiny hs x)
  · exact (probability_dilate_ge (by linarith : 0 < 1-δ) (by linarith : 1-δ ≤ 1)
      (measurable_region ρ E)).trans (probability_mono hin)
  · exact (probability_mono hout).trans (probability_dilate_le (by linarith : 1 ≤ 1+δ)
      (measurable_region ρ E))

/-- The points where the decoder needs its second observation. -/
noncomputable def ambiguousRegion {d : ℕ} (ρ : ℝ) (A : Coefficients d) (b : Point d)
    (δ : ℝ) : Set (Point d) := {x | normSq (ρ⁻¹ • x) ≤ 3 ∧
      (1-δ ≤ polynomial A b (ρ⁻¹ • x) ∧ polynomial A b (ρ⁻¹ • x) < 1+δ)}

theorem ambiguous_mass {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (A : Coefficients d) (b : Point d)
    {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : 2*δ < 1)
    (h : ∀ x, normSq x ≤ 3 → |polynomial A b x-polynomial E.A E.b x| ≤ δ) :
    probability (ambiguousRegion ρ A b δ) ≤
      ((1+2*δ)^d-(1-2*δ)^d)*probability (region ρ E) := by
  have h0 : 0 < 1-2*δ := by linarith
  have h1 : 0 < 1+2*δ := by linarith
  let F := region ρ E
  have H (x : Point d) (hx : x ∈ ambiguousRegion ρ A b δ) :
      1-2*δ ≤ polynomial E.A E.b (ρ⁻¹ • x) ∧ polynomial E.A E.b (ρ⁻¹ • x)<1+2*δ := by
    change normSq (ρ⁻¹ • x) ≤ 3 ∧
      (1-δ≤polynomial A b (ρ⁻¹ • x) ∧ polynomial A b (ρ⁻¹ • x)<1+δ) at hx
    have he := abs_le.mp (h _ hx.1)
    exact ⟨by linarith [hx.2.1,he.2],by linarith [hx.2.2,he.1]⟩
  have hs : ambiguousRegion ρ A b δ ⊆ (1+2*δ) • F \ ((1-2*δ) • F) := by
    intro x hx
    refine ⟨?_,?_⟩
    · have hp : 0 < (1+2*δ)⁻¹ := inv_pos.mpr h1
      have hq : (1+2*δ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ (by linarith)
      refine ⟨(1+2*δ)⁻¹ • x, ?_,?_⟩
      · change polynomial E.A E.b (ρ⁻¹ • ((1+2*δ)⁻¹ • x)) < 1
        rw [smul_comm (ρ⁻¹:ℝ) ((1+2*δ)⁻¹) x]
        refine (polynomial_contract E _ hp.le hq).trans_lt ?_
        simpa [inv_mul_cancel₀ h1.ne'] using mul_lt_mul_of_pos_left (H x hx).2 hp
      · change (1+2*δ) • ((1+2*δ)⁻¹ • x) = x
        rw [smul_smul,mul_inv_cancel₀ h1.ne',one_smul]
    · rintro ⟨y,hy,rfl⟩
      have hy' : polynomial E.A E.b (ρ⁻¹ • y)<1 := hy
      have hc := polynomial_contract E (ρ⁻¹ • y) h0.le (by linarith : 1-2*δ≤1)
      have hm := (H _ hx).1
      rw [smul_comm (ρ⁻¹:ℝ) (1-2*δ) y] at hm
      linarith only [hc,hm,mul_lt_mul_of_pos_left hy' h0]
  have hnest : (1-2*δ) • F ⊆ (1+2*δ) • F := by
    rintro x ⟨y,hy,rfl⟩
    let t := (1-2*δ)/(1+2*δ)
    have ht : 0 < t := div_pos h0 h1
    have ht1 : t ≤ 1 := (div_le_one₀ h1).2 (by linarith)
    refine ⟨t • y, ?_,?_⟩
    · change polynomial E.A E.b (ρ⁻¹ • (t • y)) < 1
      rw [smul_comm (ρ⁻¹:ℝ) t y]
      have hy' : polynomial E.A E.b (ρ⁻¹ • y)<1 := hy
      exact (polynomial_contract E _ ht.le ht1).trans_lt
        ((mul_lt_mul_of_pos_left hy' ht).trans_le (by simpa using ht1))
    · change (1+2*δ) • (t • y) = (1-2*δ) • y
      rw [smul_smul]
      have hprod : (1+2*δ)*t=1-2*δ := by dsimp [t]; field_simp
      rw [hprod]
  have hd := probability_mono hs
  have hF : MeasurableSet F := measurable_region _ _
  have hg := probability_dilate_ge h0 (by linarith : 1-2*δ≤1) hF
  have hl := probability_dilate_le (by linarith : 1≤1+2*δ) hF
  unfold probability at hd
  rw [measureReal_sdiff hnest (measurableSet_dilate hF h0.ne')] at hd
  change probability (ambiguousRegion ρ A b δ) ≤
    probability ((1+2*δ) • F)-probability ((1-2*δ) • F) at hd
  change probability (ambiguousRegion ρ A b δ) ≤ ((1+2*δ)^d-(1-2*δ)^d)*probability F
  rw [sub_mul]
  exact hd.trans (sub_le_sub hl hg)

end Cubature
