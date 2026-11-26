import Theory.GaussianDilation

open MeasureTheory Set
open scoped BigOperators

namespace Cubature

variable {d : ℕ}

/-- `bisection` keeps the last bracket and the scalar comparisons actually requested. -/
noncomputable def bisection (p : ℝ → Bool) (l h : ℝ) : ℕ → (ℝ × ℝ) × List ℝ
  | 0 => ((l, h), [])
  | n+1 =>
    let t := (l+h)/2
    let recResult := if p t then bisection p l t n else bisection p t h n
    (recResult.1, t :: recResult.2)

private lemma half_pos_power (n : ℕ) : (0 : ℝ) < (2 : ℝ)^n := by positivity

theorem bisection_spec (p : ℝ → Bool) (r : ℝ) (n : ℕ) {l h : ℝ}
    (hl : 0 < l) (hrl : l ≤ r) (hrh : r ≤ h)
    (hp : ∀ t, 0 < t → (p t = true → r ≤ t) ∧ (p t = false → t ≤ r)) :
    let z := bisection p l h n
    l ≤ z.1.1 ∧ z.1.1 ≤ r ∧ r ≤ z.1.2 ∧ z.1.2 ≤ h ∧
      z.1.2-z.1.1 = (h-l)/(2 : ℝ)^n ∧ z.2.length = n ∧
      ∀ t ∈ z.2, l ≤ t ∧ t ≤ h := by
  induction n generalizing l h with
  | zero => simp [bisection, hrl, hrh]
  | succ n ih =>
    let t := (l+h)/2
    have hlt : l ≤ t := by dsimp [t]; linarith
    have hth : t ≤ h := by dsimp [t]; linarith
    have htp : 0 < t := hl.trans_le hlt
    have hs : (2 : ℝ)^n ≠ 0 := (half_pos_power n).ne'
    rw [bisection]
    change let q := if p t = true then bisection p l t n else bisection p t h n
      l ≤ q.1.1 ∧ q.1.1 ≤ r ∧ r ≤ q.1.2 ∧ q.1.2 ≤ h ∧
        q.1.2-q.1.1 = (h-l)/(2 : ℝ)^(n+1) ∧ (t::q.2).length = n+1 ∧
        ∀ s ∈ t::q.2, l ≤ s ∧ s ≤ h
    by_cases hc : p t = true
    · simp only [ite_eq_left hc]
      have H := ih hl hrl ((hp t htp).1 hc)
      rcases H with ⟨h1,h2,h3,h4,h5,h6,h7⟩
      refine ⟨h1,h2,h3,h4.trans hth,?_,by simp [h6],?_⟩
      · rw [h5, pow_succ]
        dsimp [t]
        field_simp
        ring
      · intro s Hs
        rcases List.mem_cons.mp Hs with rfl | Hs
        · exact ⟨hlt,hth⟩
        · exact ⟨(h7 s Hs).1,(h7 s Hs).2.trans hth⟩
    · simp only [ite_eq_right hc]
      have hc' : p t = false := Bool.eq_false_of_not_eq_true hc
      have H := ih htp ((hp t htp).2 hc') hrh
      rcases H with ⟨h1,h2,h3,h4,h5,h6,h7⟩
      refine ⟨hlt.trans h1,h2,h3,h4,?_,by simp [h6],?_⟩
      · rw [h5, pow_succ]
        dsimp [t]
        field_simp
        ring
      · intro s Hs
        rcases List.mem_cons.mp Hs with rfl | Hs
        · exact ⟨hlt,hth⟩
        · exact ⟨hlt.trans (h7 s Hs).1,(h7 s Hs).2⟩

def axis (i : Fin d) : Point d := Pi.single i (1 : ℝ)

def pairDirection (i j : Fin d) : Point d :=
  if i = j then axis i else axis i + axis j

def signedDirection (i j : Fin d) (σ : Bool) : Point d :=
  if σ then pairDirection i j else -pairDirection i j

def polynomial (A : Fin d → Fin d → ℝ) (b : Point d) (x : Point d) : ℝ :=
  (∑ i, ∑ j, A i j * x i * x j) + ∑ i, b i * x i

def ellipsoid (a : Point d) (A : Fin d → Fin d → ℝ) (b : Point d) : Set (Point d) :=
  {x | polynomial A b (x-a) ≤ 1}

def l1norm (x : Point d) : ℝ := ∑ i, |x i|

noncomputable def reciprocalRadius (A : Fin d → Fin d → ℝ) (b u : Point d) : ℝ :=
  let β := ∑ i, b i * u i
  let q := ∑ i, ∑ j, A i j * u i * u j
  (β + Real.sqrt (β^2 + 4*q))/2

/-- The comparison score is only used for its side of 1/2. On the boundary either answer
is permitted (`oracleCut` has no restriction there). -/
def oracleCut (f : Point d → ℝ) (a : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) : Prop :=
  ∀ x, (polynomial A b (x-a) < 1 → 1/2 < f x) ∧
    (1 < polynomial A b (x-a) → f x ≤ 1/2)

noncomputable def rayPoint (a u : Point d) (t : ℝ) : Point d := a + t⁻¹ • u

noncomputable def raySearch (f : Point d → ℝ) (a u : Point d) (l h : ℝ) (n : ℕ) :
    (ℝ × ℝ) × List ℝ :=
  bisection (fun t ↦ decide (1/2 < f (rayPoint a u t))) l h n

noncomputable def rayApprox (f : Point d → ℝ) (a u : Point d) (l h : ℝ) (n : ℕ) : ℝ :=
  let r := (raySearch f a u l h n).1
  (r.1+r.2)/2

noncomputable def approxDirectionalQuad
    (f : Point d → ℝ) (a : Point d) (l h : ℝ) (n : ℕ) (i j : Fin d) : ℝ :=
  rayApprox f a (signedDirection i j true) l h n *
    rayApprox f a (signedDirection i j false) l h n

noncomputable def approxMatrix
    (f : Point d → ℝ) (a : Point d) (l h : ℝ) (n : ℕ) (i j : Fin d) : ℝ :=
  if i = j then approxDirectionalQuad f a l h n i i
  else (approxDirectionalQuad f a l h n i j - approxDirectionalQuad f a l h n i i -
    approxDirectionalQuad f a l h n j j)/2

noncomputable def approxLinear
    (f : Point d → ℝ) (a : Point d) (l h : ℝ) (n : ℕ) (i : Fin d) : ℝ :=
  rayApprox f a (signedDirection i i true) l h n -
    rayApprox f a (signedDirection i i false) l h n

/-- The algorithm lists all ordered coordinate pairs, including the diagonal, and both signs.
This redundant indexing avoids a data-dependent tie among coordinate pairs. -/
noncomputable def ellipsoidQueries
    (f : Point d → ℝ) (a : Point d) (l h : ℝ) (n : ℕ) : List (Point d) :=
  (List.finRange d).flatMap fun i ↦ (List.finRange d).flatMap fun j ↦
    [true,false].flatMap fun sign ↦
      let u := signedDirection i j sign
      (raySearch f a u l h n).2.map (rayPoint a u)

noncomputable def approxSet
    (f : Point d → ℝ) (a : Point d) (l h R : ℝ) (n : ℕ) : Set (Point d) :=
  {x | l1norm (x-a) ≤ R} ∩
    ellipsoid a (approxMatrix f a l h n) (approxLinear f a l h n)

noncomputable def ellipsoidEstimate
    (f : Point d → ℝ) (a : Point d) (l h R : ℝ) (n : ℕ) : ℝ :=
  (gamma d).real (approxSet f a l h R n)

/-- The symmetric bilinear expression associated with the coefficient array. -/
def bilinear (A : Fin d → Fin d → ℝ) (x y : Point d) : ℝ :=
  ∑ i, ∑ j, A i j * x i * y j

def linear (b x : Point d) : ℝ := ∑ i, b i * x i

lemma polynomial_eq (A : Fin d → Fin d → ℝ) (b x : Point d) :
    polynomial A b x = bilinear A x x + linear b x := rfl

lemma bilinear_neg (A : Fin d → Fin d → ℝ) (x : Point d) :
    bilinear A (-x) (-x) = bilinear A x x := by
  simp [bilinear, Pi.neg_apply]

lemma linear_neg (b x : Point d) : linear b (-x) = -linear b x := by
  simp [linear, Finset.sum_neg_distrib]

lemma bilinear_smul (A : Fin d → Fin d → ℝ) (x : Point d) (s : ℝ) :
    bilinear A (s • x) (s • x) = s^2 * bilinear A x x := by
  unfold bilinear
  simp only [Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  congr 1
  ext i
  congr 1
  ext j
  ring

lemma linear_smul (b x : Point d) (s : ℝ) : linear b (s • x) = s * linear b x := by
  unfold linear
  simp only [Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  congr 1
  ext i
  ring

lemma bilinear_axis (A : Fin d → Fin d → ℝ) (i j : Fin d) :
    bilinear A (axis i) (axis j) = A i j := by
  classical
  simp [bilinear, axis, Pi.single_apply, mul_ite]

lemma linear_axis (b : Point d) (i : Fin d) : linear b (axis i) = b i := by
  classical
  simp [linear, axis, Pi.single_apply, mul_ite]

lemma bilinear_add (A : Fin d → Fin d → ℝ) (x y : Point d) :
    bilinear A (x+y) (x+y) =
      bilinear A x x + bilinear A x y + bilinear A y x + bilinear A y y := by
  simp only [bilinear, Pi.add_apply, mul_add, add_mul, Finset.sum_add_distrib]
  ring

lemma reciprocal_pair (A : Fin d → Fin d → ℝ) (b u : Point d)
    (hA : 0 ≤ bilinear A u u) :
    let p := reciprocalRadius A b u
    let q := reciprocalRadius A b (-u)
    0 ≤ p ∧ 0 ≤ q ∧ p*q = bilinear A u u ∧ p-q = linear b u := by
  let β := linear b u
  let q := bilinear A u u
  let s := Real.sqrt (β^2+4*q)
  have hq : 0 ≤ q := hA
  have hb : β^2 ≤ β^2+4*q := by linarith
  have hp : 0 ≤ β^2+4*q := le_trans (sq_nonneg _) hb
  have hss : s^2=β^2+4*q := Real.sq_sqrt hp
  have habs : |β| ≤ s := Real.abs_le_sqrt hb
  have hx : reciprocalRadius A b u = (β+s)/2 := rfl
  have hy : reciprocalRadius A b (-u) = (-β+s)/2 := by
    unfold reciprocalRadius
    change (linear b (-u) + Real.sqrt ((linear b (-u))^2 + 4*bilinear A (-u) (-u)))/2 = _
    simp only [linear_neg, bilinear_neg, neg_sq]
    rfl
  dsimp only
  rw [hx, hy]
  have H := abs_le.mp habs
  exact ⟨by linarith [H.1], by linarith [H.2], by nlinarith only [hss], by ring⟩

lemma reciprocal_comparison (f : Point d → ℝ) (a u : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (hf : oracleCut f a A b)
    (hA : 0 ≤ bilinear A u u) (t : ℝ) (ht : 0 < t) :
    (decide (1/2 < f (rayPoint a u t)) = true → reciprocalRadius A b u ≤ t) ∧
    (decide (1/2 < f (rayPoint a u t)) = false → t ≤ reciprocalRadius A b u) := by
  let p := reciprocalRadius A b u
  let q := reciprocalRadius A b (-u)
  have H := reciprocal_pair A b u hA
  change 0 ≤ p ∧ 0 ≤ q ∧ p*q = bilinear A u u ∧ p-q = linear b u at H
  have he : rayPoint a u t - a = t⁻¹ • u := by
    unfold rayPoint
    exact add_sub_cancel_left _ _
  have hfct : 1 - polynomial A b (rayPoint a u t-a) = (t-p)*(t+q)/t^2 := by
    rw [he, polynomial_eq, bilinear_smul, linear_smul, ← H.2.2.1, ← H.2.2.2]
    field_simp [ht.ne']
    ring
  constructor
  · intro hh
    by_contra! hlt
    have hneg : (t-p)*(t+q)/t^2 < 0 :=
      div_neg_of_neg_of_pos
        (mul_neg_of_neg_of_pos (sub_neg.mpr hlt) (by linarith [H.2.1])) (sq_pos_of_pos ht)
    have ho := (hf (rayPoint a u t)).2 (by linarith)
    exact (not_lt_of_ge ho) (of_decide_eq_true hh)
  · intro hh
    by_contra! hlt
    have hpos : 0 < (t-p)*(t+q)/t^2 :=
      div_pos (mul_pos (sub_pos.mpr hlt) (by linarith [H.2.1])) (sq_pos_of_pos ht)
    have hi := (hf (rayPoint a u t)).1 (by linarith)
    rw [decide_eq_true hi] at hh
    contradiction

lemma raySearch_spec (f : Point d → ℝ) (a u : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (hf : oracleCut f a A b)
    (hA : 0 ≤ bilinear A u u) {l h : ℝ} (hl : 0 < l) (n : ℕ)
    (hlo : l ≤ reciprocalRadius A b u) (hhi : reciprocalRadius A b u ≤ h) :
    let z := raySearch f a u l h n
    l ≤ z.1.1 ∧ z.1.1 ≤ reciprocalRadius A b u ∧
      reciprocalRadius A b u ≤ z.1.2 ∧ z.1.2 ≤ h ∧
      z.1.2-z.1.1 = (h-l)/(2 : ℝ)^n ∧ z.2.length = n ∧
      ∀ t ∈ z.2, l ≤ t ∧ t ≤ h := by
  exact bisection_spec _ _ _ hl hlo hhi (reciprocal_comparison f a u A b hf hA)

lemma rayApprox_spec (f : Point d → ℝ) (a u : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (hf : oracleCut f a A b)
    (hA : 0 ≤ bilinear A u u) {l h : ℝ} (hl : 0 < l) (n : ℕ)
    (hlo : l ≤ reciprocalRadius A b u) (hhi : reciprocalRadius A b u ≤ h) :
    l ≤ rayApprox f a u l h n ∧ rayApprox f a u l h n ≤ h ∧
      |rayApprox f a u l h n-reciprocalRadius A b u| ≤ (h-l)/(2 : ℝ)^(n+1) := by
  rcases raySearch_spec f a u A b hf hA hl n hlo hhi with ⟨h1,h2,h3,h4,h5,_,_⟩
  unfold rayApprox
  rw [pow_succ, ← div_div, ← h5]
  exact ⟨by linarith, by linarith, abs_le.mpr ⟨by linarith, by linarith⟩⟩

lemma signedDirection_abs (i j k : Fin d) (s : Bool) :
    |signedDirection i j s k| ≤ 1 := by
  classical
  by_cases e : i=j
  · subst j
    cases s <;> by_cases h : k=i <;> simp [signedDirection, pairDirection, axis, h]
  · cases s <;> by_cases hi : k=i <;> by_cases hj : k=j <;>
      simp_all [signedDirection, pairDirection, axis]

lemma recover_query_bound (f : Point d → ℝ) (a : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (hf : oracleCut f a A b)
    (hA : ∀ u, 0 ≤ bilinear A u u) {l h : ℝ} (hl : 0 < l) (n : ℕ)
    (hRad : ∀ i j s, l ≤ reciprocalRadius A b (signedDirection i j s) ∧
      reciprocalRadius A b (signedDirection i j s) ≤ h)
    (hcap : ∀ i, |a i|+l⁻¹ ≤ 40) :
    (ellipsoidQueries f a l h n).length = 2*d*d*n ∧
      ∀ x ∈ ellipsoidQueries f a l h n, ∀ i, |x i| ≤ 40 := by
  have hS (i j : Fin d) (s : Bool) :=
    raySearch_spec f a (signedDirection i j s) A b hf (hA _) hl n (hRad i j s).1 (hRad i j s).2
  have hN (i j : Fin d) (s : Bool) :
      (raySearch f a (signedDirection i j s) l h n).2.length = n :=
    (hS i j s).2.2.2.2.2.1
  constructor
  · unfold ellipsoidQueries
    simp [List.length_flatMap, hN, List.sum_replicate]
    ring
  · intro x hx i
    simp only [ellipsoidQueries, List.mem_flatMap, List.mem_map] at hx
    obtain ⟨j, _, k, _, s, _, t, ht, rfl⟩ := hx
    have hlt : l ≤ t := ((hS j k s).2.2.2.2.2.2 t ht).1
    have ht' : 0 < t := hl.trans_le hlt
    calc
      _ ≤ |a i|+|t⁻¹*signedDirection j k s i| := abs_add_le _ _
      _ ≤ |a i|+l⁻¹ := by
        rw [abs_mul, abs_of_nonneg (inv_nonneg.mpr ht'.le)]
        have h1 := signedDirection_abs j k i s
        have h2 := inv_le_inv₀ ht' hl
        gcongr
        calc
          _ ≤ t⁻¹*1 := mul_le_mul_of_nonneg_left h1 (inv_nonneg.mpr ht'.le)
          _ ≤ l⁻¹ := by simpa using h2.mpr hlt
      _ ≤ 40 := hcap i

end Cubature
