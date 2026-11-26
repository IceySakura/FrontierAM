import Theory.Ellipsoid

open MeasureTheory Set
open scoped BigOperators

namespace Cubature

variable {d : ℕ}

lemma recover_coefficients (f : Point d → ℝ) (a : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (hf : oracleCut f a A b)
    (hA : ∀ u, 0 ≤ bilinear A u u)
    (hsym : ∀ i j, A i j = A j i)
    {l h : ℝ} (hl : 0 < l) (hlh : l ≤ h) (n : ℕ)
    (hRad : ∀ i j s, l ≤ reciprocalRadius A b (signedDirection i j s) ∧
      reciprocalRadius A b (signedDirection i j s) ≤ h) :
    let δ := (h-l)/(2 : ℝ)^(n+1)
    (∀ i j, |approxMatrix f a l h n i j-A i j| ≤ 3*h*δ) ∧
      (∀ i, |approxLinear f a l h n i-b i| ≤ 2*δ) := by
  let δ := (h-l)/(2 : ℝ)^(n+1)
  have hδ : 0 ≤ δ := div_nonneg (sub_nonneg.mpr hlh) (by positivity)
  have hh : 0 ≤ h := (hl.trans_le hlh).le
  have bounds (i j : Fin d) (s : Bool) :=
    rayApprox_spec f a (signedDirection i j s) A b hf (hA _) hl n
      (hRad i j s).1 (hRad i j s).2
  have directional (i j : Fin d) :
      |approxDirectionalQuad f a l h n i j -
        bilinear A (pairDirection i j) (pairDirection i j)| ≤ 2*h*δ := by
    let u := pairDirection i j
    let p := reciprocalRadius A b u
    let q := reciprocalRadius A b (-u)
    let P := rayApprox f a (signedDirection i j true) l h n
    let Q := rayApprox f a (signedDirection i j false) l h n
    have hpq := (reciprocal_pair A b u (hA u)).2.2.1
    change p*q = _ at hpq
    have hp : 0 ≤ p := hl.le.trans (hRad i j true).1
    have hp' : p ≤ h := (hRad i j true).2
    have hQ : 0 ≤ Q := hl.le.trans (bounds i j false).1
    have hQ' : Q ≤ h := (bounds i j false).2.1
    have heP : |P-p| ≤ δ := (bounds i j true).2.2
    have heQ : |Q-q| ≤ δ := (bounds i j false).2.2
    unfold approxDirectionalQuad
    change |P*Q - _| ≤ _
    rw [← hpq, show P*Q-p*q = (P-p)*Q+p*(Q-q) by ring]
    calc
      _ ≤ |(P-p)*Q|+|p*(Q-q)| := abs_add_le _ _
      _ = |P-p| * Q + p * |Q-q| := by rw [abs_mul, abs_mul, abs_of_nonneg hQ, abs_of_nonneg hp]
      _ ≤ δ*h+h*δ := add_le_add
        (mul_le_mul heP hQ' hQ hδ) (mul_le_mul hp' heQ (abs_nonneg _) hh)
      _ = _ := by ring
  have diag (i : Fin d) : |approxDirectionalQuad f a l h n i i - A i i| ≤ 2*h*δ := by
    simpa [pairDirection, bilinear_axis] using directional i i
  constructor
  · intro i j
    by_cases he : i = j
    · subst j
      simp only [approxMatrix, ite_true]
      exact (diag i).trans (by nlinarith only [mul_nonneg hh hδ])
    · simp only [approxMatrix, ite_eq_right he]
      have H := directional i j
      simp only [pairDirection, ite_eq_right he, bilinear_add, bilinear_axis, ← hsym i j] at H
      let F := approxDirectionalQuad f a l h n
      change |F i j - (A i i + A i j + A i j + A j j)| ≤ 2*h*δ at H
      have Hi : |F i i-A i i| ≤ 2*h*δ := diag i
      have Hj : |F j j-A j j| ≤ 2*h*δ := diag j
      change |(F i j-F i i-F j j)/2-A i j| ≤ 3*h*δ
      exact abs_le.mpr ⟨by linarith [(abs_le.mp H).1, (abs_le.mp Hi).2, (abs_le.mp Hj).2],
        by linarith [(abs_le.mp H).2, (abs_le.mp Hi).1, (abs_le.mp Hj).1]⟩
  · intro i
    have Hp := (bounds i i true).2.2
    have Hq := (bounds i i false).2.2
    have he := (reciprocal_pair A b (axis i) (hA _)).2.2.2
    change reciprocalRadius A b (axis i)-reciprocalRadius A b (-axis i) = linear b (axis i) at he
    rw [linear_axis] at he
    simp only [signedDirection, pairDirection, Bool.false_eq_true, ite_false, ite_true] at Hp Hq
    simp only [approxLinear, signedDirection, pairDirection, Bool.false_eq_true, ite_false, ite_true]
    exact abs_le.mpr ⟨by linarith [(abs_le.mp Hp).1, (abs_le.mp Hq).2],
      by linarith [(abs_le.mp Hp).2, (abs_le.mp Hq).1]⟩

lemma l1norm_nonneg (x : Point d) : 0 ≤ l1norm x :=
  Finset.sum_nonneg (fun _ _ ↦ abs_nonneg _)

lemma l1norm_smul (x : Point d) {s : ℝ} (hs : 0 ≤ s) :
    l1norm (s • x) = s*l1norm x := by
  simp [l1norm, Pi.smul_apply, abs_mul, abs_of_nonneg hs, Finset.mul_sum]

lemma polynomial_error (A A' : Fin d → Fin d → ℝ) (b b' : Point d) (x : Point d)
    (eA eB : ℝ) (hA : ∀ i j, |A' i j-A i j| ≤ eA) (hB : ∀ i, |b' i-b i| ≤ eB) :
    |polynomial A' b' x-polynomial A b x| ≤ eA*(l1norm x)^2+eB*l1norm x := by
  have hQA : |bilinear A' x x-bilinear A x x| ≤ eA*(l1norm x)^2 := by
    unfold bilinear
    simp only [← Finset.sum_sub_distrib, ← sub_mul]
    calc
      _ ≤ ∑ i, |∑ j, (A' i j-A i j)*x i*x j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, ∑ j, |(A' i j-A i j)*x i*x j| :=
        Finset.sum_le_sum (fun _ _ ↦ Finset.abs_sum_le_sum_abs _ _)
      _ ≤ ∑ i, ∑ j, eA * |x i| * |x j| := by
        simp_rw [abs_mul]
        exact Finset.sum_le_sum (fun i _ ↦ Finset.sum_le_sum (fun j _ ↦
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (hA i j) (abs_nonneg _))
            (abs_nonneg _)))
      _ = _ := by
        unfold l1norm
        calc
          (∑ i, ∑ j, eA * |x i| * |x j|) =
              ∑ i, eA * |x i| * (∑ j, |x j|) := by
            exact Finset.sum_congr rfl (fun _ _ ↦ (Finset.mul_sum _ _ _).symm)
          _ = _ := by
            rw [← Finset.sum_mul, ← Finset.mul_sum]
            ring
  have hQB : |linear b' x-linear b x| ≤ eB*l1norm x := by
    unfold linear
    simp only [← Finset.sum_sub_distrib, ← sub_mul]
    calc
      _ ≤ ∑ i, |(b' i-b i)*x i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, eB * |x i| := by
        simp_rw [abs_mul]
        exact Finset.sum_le_sum (fun i _ ↦ mul_le_mul_of_nonneg_right (hB i) (abs_nonneg _))
      _ = _ := (Finset.mul_sum _ _ _).symm
  rw [polynomial_eq, polynomial_eq,
    show (bilinear A' x x+linear b' x)-(bilinear A x x+linear b x) =
      (bilinear A' x x-bilinear A x x)+(linear b' x-linear b x) by ring]
  exact (abs_add_le _ _).trans (add_le_add hQA hQB)

lemma measurable_ellipsoid (a : Point d) (A : Fin d → Fin d → ℝ) (b : Point d) :
    MeasurableSet (ellipsoid a A b) := by
  unfold ellipsoid polynomial
  apply measurableSet_le
  all_goals fun_prop

lemma measurable_l1ball (a : Point d) (R : ℝ) : MeasurableSet {x | l1norm (x-a) ≤ R} := by
  unfold l1norm
  apply measurableSet_le
  all_goals fun_prop

lemma two_sided_mass_bound (a : Point d) {t : ℝ} (ht : 1 ≤ t) {E F : Set (Point d)}
    (hE : MeasurableSet E) (hF : MeasurableSet F)
    (hEF : E ⊆ (dilate a t⁻¹) ⁻¹' F) (hFE : F ⊆ (dilate a t⁻¹) ⁻¹' E) :
    |(gamma d).real F - (gamma d).real E| ≤
      (dilationFactor a t-1)*(gamma d).real E := by
  have U := (measureReal_mono (μ := gamma d) hFE).trans (gaussian_preimage_dilation_le a ht hE)
  have L := (measureReal_mono (μ := gamma d) hEF).trans (gaussian_preimage_dilation_le a ht hF)
  have hp : 0 ≤ (gamma d).real E := ENNReal.toReal_nonneg
  have hC : 0 ≤ dilationFactor a t-1 := sub_nonneg.mpr (one_le_dilationFactor a ht)
  apply abs_le.mpr
  refine ⟨?_, by linarith only [U]⟩
  by_cases h : (gamma d).real F ≤ (gamma d).real E
  · have H := mul_le_mul_of_nonneg_left h hC
    linarith only [L, H]
  · have H := mul_nonneg hC hp
    linarith only [h,H]

lemma finite_precision_sandwich (a : Point d) (A A' : Fin d → Fin d → ℝ) (b b' : Point d)
    (hA : ∀ x, 0 ≤ bilinear A x x) {R eA eB : ℝ}
    (hR : 0 ≤ R) (heA : 0 ≤ eA) (heB : 0 ≤ eB)
    (hCA : ∀ i j, |A' i j-A i j| ≤ eA) (hCB : ∀ i, |b' i-b i| ≤ eB)
    (hcover : ∀ x ∈ ellipsoid a A b, l1norm (x-a) ≤ R) :
    let t := 1+(eA*R^2+eB*R)
    let E := ellipsoid a A b
    let F := {x | l1norm (x-a) ≤ R} ∩ ellipsoid a A' b'
    E ⊆ (dilate a t⁻¹) ⁻¹' F ∧ F ⊆ (dilate a t⁻¹) ⁻¹' E := by
  let η := eA*R^2+eB*R
  let t := 1+η
  change ellipsoid a A b ⊆ (dilate a t⁻¹) ⁻¹'
      ({x | l1norm (x-a) ≤ R} ∩ ellipsoid a A' b') ∧
    ({x | l1norm (x-a) ≤ R} ∩ ellipsoid a A' b') ⊆ (dilate a t⁻¹) ⁻¹' ellipsoid a A b
  have hη : 0 ≤ η := add_nonneg (mul_nonneg heA (sq_nonneg _)) (mul_nonneg heB hR)
  have ht : 1 ≤ t := by dsimp [t]; linarith
  let s := t⁻¹
  have hp : 0 < t := by linarith
  have hs0 : 0 ≤ s := inv_nonneg.mpr hp.le
  have hs1 : s ≤ 1 := by dsimp [s]; exact inv_le_one_of_one_le₀ ht
  have hss : s^2 ≤ s := by nlinarith only [mul_nonneg hs0 (sub_nonneg.mpr hs1)]
  have hst : s * (1+η) = 1 := inv_mul_cancel₀ hp.ne'
  have htrans (x : Point d) : dilate a t⁻¹ x-a = s • (x-a) := by
    unfold dilate
    exact add_sub_cancel_left _ _
  have hsmall (x : Point d) :
      polynomial A b (s • x) ≤ s*polynomial A b x := by
    simp only [polynomial_eq, bilinear_smul, linear_smul, mul_add]
    exact add_le_add (mul_le_mul_of_nonneg_right hss (hA x)) le_rfl
  have err (x : Point d) (hx : l1norm x ≤ R) :
      |polynomial A' b' x-polynomial A b x| ≤ η := by
    refine (polynomial_error A A' b b' x eA eB hCA hCB).trans ?_
    have h1 : (l1norm x)^2 ≤ R^2 := by
      exact pow_le_pow_left₀ (l1norm_nonneg x) hx _
    exact add_le_add (mul_le_mul_of_nonneg_left h1 heA) (mul_le_mul_of_nonneg_left hx heB)
  constructor
  · intro x hx
    simp only [Set.mem_preimage, Set.mem_inter_iff, Set.mem_ofPred_eq]
    have hrad := hcover x hx
    have hpoly : polynomial A b (x-a) ≤ 1 := hx
    constructor
    · rw [htrans, l1norm_smul _ hs0]
      exact (mul_le_mul_of_nonneg_left hrad hs0).trans
        (by simpa using mul_le_mul_of_nonneg_right hs1 hR)
    · change polynomial A' b' (dilate a t⁻¹ x-a) ≤ 1
      rw [htrans]
      have H := polynomial_error A A' b b' (s • (x-a)) eA eB hCA hCB
      rw [l1norm_smul _ hs0, mul_pow] at H
      have h1 : (l1norm (x-a))^2 ≤ R^2 :=
        pow_le_pow_left₀ (l1norm_nonneg _) hrad _
      have h2 : eA*(s^2*(l1norm (x-a))^2) ≤ eA*(s*R^2) :=
        mul_le_mul_of_nonneg_left
          (mul_le_mul hss h1 (sq_nonneg _) hs0) heA
      have h3 : eB*(s*l1norm (x-a)) ≤ eB*(s*R) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hrad hs0) heB
      have h4 := (hsmall (x-a)).trans (mul_le_mul_of_nonneg_left hpoly hs0)
      nlinarith only [(abs_le.mp H).2,h2,h3,h4,hst]
  · intro x hx
    change polynomial A b (dilate a t⁻¹ x-a) ≤ 1
    have hpol' : polynomial A' b' (x-a) ≤ 1 := hx.2
    have herr := (abs_le.mp (err (x-a) hx.1)).1
    have hpol : polynomial A b (x-a) ≤ 1+η := by linarith
    rw [htrans, ← hst]
    exact (hsmall _).trans (mul_le_mul_of_nonneg_left hpol hs0)

end Cubature
