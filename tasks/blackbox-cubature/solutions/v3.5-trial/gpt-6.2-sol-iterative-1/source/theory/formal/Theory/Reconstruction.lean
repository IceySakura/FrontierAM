import Theory.Sections
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

open Set MeasureTheory
open scoped BigOperators

namespace Cubature

noncomputable section

variable {d : ℕ}

private lemma abs_difference_le (x y : ℝ) : |x-y| ≤ |x|+|y| := by
  simpa using abs_sub_le x 0 y

private lemma section_factor (a dir : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (u v t : ℝ) (ht : t ≠ 0)
    (hQ : quadPart A dir = u * v) (hL : linearPart b dir = u - v) :
    quadratic A b (sectionPoint a dir t - a) - 1 =
      (u - t) * (v + t) / t ^ 2 := by
  rw [sectionPoint, add_sub_cancel_left, quadratic, quadPart_smul, linearPart_smul, hQ, hL]
  field_simp
  ring

lemma runSection_spec (oracle : Vec d → Bool) (a dir : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (n : ℕ) (lo hi u v : ℝ)
    (hlo : 0 < lo) (hu : u ∈ Icc lo hi) (hv : 0 < v)
    (hQ : quadPart A dir = u * v) (hL : linearPart b dir = u - v)
    (hOracle : ∀ z, (quadratic A b (z-a) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-a) → oracle z = false)) :
    let r := runSection oracle a n lo hi dir
    r.lower ∈ Icc lo hi ∧ |r.lower - u| ≤ sectionAccuracy n lo hi ∧
      ∀ z ∈ r.queries, ∃ t ∈ Icc lo hi, z = sectionPoint a dir t := by
  have hpoint (t : ℝ) (ht : 0 < t) :
      (oracle (sectionPoint a dir t) = true → u ≤ t) ∧
      (oracle (sectionPoint a dir t) = false → t ≤ u) := by
    have fac := section_factor a dir A b u v t ht.ne' hQ hL
    constructor
    · intro h
      by_contra! hlt
      have he : 0 < (u - t) * (v+t) / t ^ 2 :=
        div_pos (mul_pos (sub_pos.mpr hlt) (by positivity)) (by positivity)
      have hb := (hOracle _).2 (by linarith only [he, fac])
      simp [h] at hb
    · intro h
      by_contra! hlt
      have he : (u - t) * (v+t) / t ^ 2 < 0 :=
        div_neg_of_neg_of_pos (mul_neg_of_neg_of_pos (sub_neg.mpr hlt) (by positivity))
          (by positivity)
      have hb := (hOracle _).1 (by linarith only [he, fac])
      simp [h] at hb
  obtain ⟨hll, hlu, huu, huh, hw, htrace⟩ :=
    bisect_spec oracle (sectionPoint a dir) u hpoint n lo hi hlo hu.1 hu.2
  refine ⟨⟨hll, hlu.trans (huu.trans huh)⟩, ?_, htrace⟩
  unfold runSection sectionAccuracy
  rw [abs_of_nonpos (sub_nonpos.mpr hlu), ← hw]
  linarith only [huu]

lemma both_sections_spec (oracle : Vec d → Bool) (a dir : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (n : ℕ) (lo hi : ℝ)
    (hlo : 0 < lo) (hQ : 0 ≤ quadPart A dir)
    (hu : positiveInverse A b dir ∈ Icc lo hi)
    (hv : negativeInverse A b dir ∈ Icc lo hi)
    (hOracle : ∀ z, (quadratic A b (z-a) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-a) → oracle z = false)) :
    let plus := runSection oracle a n lo hi dir
    let minus := runSection oracle a n lo hi (-dir)
    (plus.lower ∈ Icc lo hi ∧ |plus.lower - positiveInverse A b dir| ≤ sectionAccuracy n lo hi ∧
      ∀ z ∈ plus.queries, ∃ t ∈ Icc lo hi, z = sectionPoint a dir t) ∧
    (minus.lower ∈ Icc lo hi ∧ |minus.lower - negativeInverse A b dir| ≤ sectionAccuracy n lo hi ∧
      ∀ z ∈ minus.queries, ∃ t ∈ Icc lo hi, z = sectionPoint a (-dir) t) := by
  have hp := inverse_product A b dir hQ
  have hd := inverse_difference A b dir
  refine ⟨runSection_spec oracle a dir A b n lo hi _ _ hlo hu (hlo.trans_le hv.1)
    hp.symm hd.symm hOracle, ?_⟩
  apply runSection_spec oracle a (-dir) A b n lo hi _ _ hlo hv (hlo.trans_le hu.1) _ _ hOracle
  · rw [← neg_one_smul ℝ dir, quadPart_smul, neg_one_sq, one_mul, mul_comm, hp]
  · rw [← neg_one_smul ℝ dir, linearPart_smul, neg_one_mul, ← hd, neg_sub]

lemma quadratic_coeff_accuracy (oracle : Vec d → Bool) (a : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (n : ℕ) (lo hi : ℝ)
    (hlo : 0 < lo) (hhi : lo ≤ hi) (hSym : ∀ i j, A i j = A j i)
    (hPSD : ∀ x, 0 ≤ quadPart A x)
    (hRoots : ∀ i j, positiveInverse A b (sectionDirection i j) ∈ Icc lo hi ∧
      negativeInverse A b (sectionDirection i j) ∈ Icc lo hi)
    (hOracle : ∀ z, (quadratic A b (z-a) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-a) → oracle z = false)) :
    (∀ i j, |foundA oracle a n lo hi i j - A i j| ≤ 3 * hi * sectionAccuracy n lo hi) ∧
    (∀ i, |foundB oracle a n lo hi i - b i| ≤ 2 * sectionAccuracy n lo hi) := by
  let e := sectionAccuracy n lo hi
  have he : 0 ≤ e := div_nonneg (sub_nonneg.mpr hhi) (by positivity)
  have hh : 0 ≤ hi := hlo.le.trans hhi
  have legs i j := both_sections_spec oracle a (sectionDirection i j) A b n lo hi hlo
    (hPSD _) (hRoots i j).1 (hRoots i j).2 hOracle
  have prodError (i j : Fin d) :
      |foundQ oracle a n lo hi i j - quadPart A (sectionDirection i j)| ≤ 2 * hi * e := by
    let w := sectionDirection i j
    let u := positiveInverse A b w
    let v := negativeInverse A b w
    let uh := (runSection oracle a n lo hi w).lower
    let vh := (runSection oracle a n lo hi (-w)).lower
    change |uh * vh - quadPart A w| ≤ _
    rw [← inverse_product A b w (hPSD _)]
    change |uh * vh - u*v| ≤ _
    have hp := (legs i j).1
    have hm := (legs i j).2
    have h1 : |uh| ≤ hi := (abs_of_nonneg (hlo.le.trans hp.1.1)).le.trans hp.1.2
    have h2 : |v| ≤ hi :=
      (abs_of_nonneg (hlo.le.trans (hRoots i j).2.1)).le.trans (hRoots i j).2.2
    calc
      |uh * vh - u*v| = |uh * (vh-v) + (uh-u)*v| := by congr 1; ring
      _ ≤ |uh * (vh-v)| + |(uh-u)*v| := abs_add_le _ _
      _ = |uh| * |vh-v| + |uh-u| * |v| := by rw [abs_mul, abs_mul]
      _ ≤ hi*e + e*hi := by
        exact add_le_add (mul_le_mul h1 hm.2.1 (abs_nonneg _) hh)
          (mul_le_mul hp.2.1 h2 (abs_nonneg _) he)
      _ = _ := by ring
  have diagError (i : Fin d) :
      |foundQ oracle a n lo hi i i - A i i| ≤ 2 * hi * e := by
    simpa only [section_diag] using prodError i i
  constructor
  · intro i j
    change |foundA oracle a n lo hi i j - A i j| ≤ 3 * hi * e
    by_cases h : i = j
    · subst j
      simpa only [foundA, ite_true] using (diagError i).trans
        (show 2 * hi * e ≤ 3 * hi * e by nlinarith only [mul_nonneg hh he])
    · have hdir : sectionDirection (max i j) (min i j) = sectionDirection i j := by
        rcases le_total i j with hle | hle
        · have hrev : j ≠ i := fun heq => h heq.symm
          simp [max_eq_right hle, min_eq_left hle, sectionDirection, hrev, h, add_comm]
        · rw [max_eq_left hle, min_eq_right hle]
      have hs := prodError (max i j) (min i j)
      rw [hdir, section_pair A i j h (hSym j i)] at hs
      simp only [foundA, if_neg h]
      rw [show (foundQ oracle a n lo hi (max i j) (min i j) -
          foundQ oracle a n lo hi i i - foundQ oracle a n lo hi j j) / 2 - A i j =
          ((foundQ oracle a n lo hi (max i j) (min i j) - (A i i + A j j + 2*A i j)) -
           ((foundQ oracle a n lo hi i i - A i i) + (foundQ oracle a n lo hi j j - A j j))) / 2
           by ring]
      rw [abs_div, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 2)]
      apply (div_le_iff₀ (by positivity : (0 : ℝ) < 2)).2
      refine (abs_difference_le _ _).trans ((add_le_add hs ((abs_add_le _ _).trans
        (add_le_add (diagError i) (diagError j)))).trans ?_)
      ring_nf <;> exact le_rfl
  · intro i
    let u := positiveInverse A b (sectionDirection i i)
    let v := negativeInverse A b (sectionDirection i i)
    let uh := (runSection oracle a n lo hi (sectionDirection i i)).lower
    let vh := (runSection oracle a n lo hi (-sectionDirection i i)).lower
    have heq : u-v = b i := (inverse_difference _ _ _).trans (section_linear_diag _ _)
    change |uh-vh-b i| ≤ 2*e
    rw [← heq, show uh-vh-(u-v) = (uh-u)-(vh-v) by ring]
    exact (abs_difference_le _ _).trans ((add_le_add (legs i i).1.2.1 (legs i i).2.2.1).trans
      (by change e+e ≤ 2*e; linarith))

lemma quadratic_value_accuracy {A Ah : Fin d → Fin d → ℝ} {b bh : Vec d}
    (x : Vec d) (radius e hi : ℝ) (hr : 0 ≤ radius)
    (hx : ∀ i, |x i| ≤ radius)
    (ha : ∀ i j, |Ah i j - A i j| ≤ 3 * hi * e)
    (hb : ∀ i, |bh i - b i| ≤ 2 * e) :
    |quadratic Ah bh x - quadratic A b x| ≤
      (3 * hi * (d : ℝ) ^ 2 * radius ^ 2 + 2 * d * radius) * e := by
  have hQ : |quadPart Ah x - quadPart A x| ≤ (d : ℝ)^2 * (3*hi*e*radius^2) := by
    unfold quadPart quadBilin
    simp only [← Finset.sum_sub_distrib]
    calc
      _ ≤ ∑ i : Fin d, |∑ j : Fin d, (Ah i j * x i * x j - A i j * x i * x j)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i : Fin d, ∑ j : Fin d, 3*hi*e*radius^2 := by
        apply Finset.sum_le_sum
        intro i _
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
        intro j _
        rw [show Ah i j * x i * x j - A i j * x i * x j = (Ah i j - A i j) * (x i * x j)
          by ring, abs_mul, abs_mul, pow_two]
        exact mul_le_mul (ha i j) (mul_le_mul (hx i) (hx j) (abs_nonneg _) hr)
          (mul_nonneg (abs_nonneg _) (abs_nonneg _))
          ((abs_nonneg _).trans (ha i j))
      _ = _ := by simp; ring
  have hL : |linearPart bh x - linearPart b x| ≤ (d : ℝ)*(2*e*radius) := by
    unfold linearPart
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc
      _ ≤ ∑ i : Fin d, 2*e*radius := by
        apply Finset.sum_le_sum
        intro i _
        rw [← sub_mul, abs_mul]
        exact mul_le_mul (hb i) (hx i) (abs_nonneg _) ((abs_nonneg _).trans (hb i))
      _ = _ := by simp
  unfold quadratic
  rw [show quadPart Ah x + linearPart bh x - (quadPart A x + linearPart b x) =
    (quadPart Ah x - quadPart A x) + (linearPart bh x - linearPart b x) by ring]
  exact (abs_add_le _ _).trans ((add_le_add hQ hL).trans (by ring_nf; exact le_rfl))

lemma queryHistory_count (oracle : Vec d → Bool) (a : Vec d) (n : ℕ) (lo hi : ℝ) :
    (queryHistory oracle a n lo hi).length ≤ 2 * d * d * n := by
  have flatMap_bound {α β : Type} (L : List α) (F : α → List β) (k : ℕ)
      (h : ∀ a, (F a).length ≤ k) : (L.flatMap F).length ≤ L.length * k := by
    induction L with
    | nil => simp
    | cons x xs ih =>
      simp only [List.flatMap_cons, List.length_append, List.length_cons, Nat.succ_mul]
      exact (Nat.add_le_add (h x) ih).trans (Nat.add_comm _ _).le
  unfold queryHistory
  refine (flatMap_bound _ _ (n+n) ?_).trans ?_
  · intro ij
    rcases ij with ⟨i,j⟩
    by_cases hj : j ≤ i <;> simp [hj, runSection, bisect_length]
  · simp
    ring_nf
    exact le_rfl

lemma queryHistory_safe (oracle : Vec d → Bool) (a : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (n : ℕ) (lo hi : ℝ)
    (hlo : 0 < lo) (hPSD : ∀ x, 0 ≤ quadPart A x)
    (hRoots : ∀ i j, positiveInverse A b (sectionDirection i j) ∈ Icc lo hi ∧
      negativeInverse A b (sectionDirection i j) ∈ Icc lo hi)
    (hOracle : ∀ z, (quadratic A b (z-a) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-a) → oracle z = false))
    (hAllowed : ∀ i, |a i| + lo⁻¹ ≤ 40) :
    ∀ z ∈ queryHistory oracle a n lo hi, ∀ i, |z i| ≤ 40 := by
  have hDir i j k : |sectionDirection (d := d) i j k| ≤ 1 := by
    by_cases h : i = j
    · simp only [sectionDirection, h, ite_true, axis]
      split <;> simp
    · simp only [sectionDirection, if_neg h, Pi.add_apply, axis]
      by_cases hki : k = i
      · subst k; simp [h]
      · simp [hki]; split <;> simp
  have point_safe (w : Vec d) (hw : ∀ i, |w i| ≤ 1) (t : ℝ) (ht : t ∈ Icc lo hi) :
      ∀ i, |sectionPoint a w t i| ≤ 40 := by
    intro i
    have ht0 := hlo.trans_le ht.1
    simp only [sectionPoint, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    refine (abs_add_le _ _).trans ((add_le_add (le_refl (|a i|)) ?_).trans (hAllowed i))
    rw [abs_mul, abs_of_pos (inv_pos.mpr ht0)]
    refine (mul_le_of_le_one_right (inv_nonneg.mpr ht0.le) (hw i)).trans ?_
    exact inv_anti₀ hlo ht.1
  intro z hz
  unfold queryHistory at hz
  obtain ⟨⟨i,j⟩, _, hz⟩ := List.mem_flatMap.mp hz
  simp only at hz
  split at hz
  · have hh := both_sections_spec oracle a (sectionDirection i j) A b n lo hi hlo (hPSD _)
      (hRoots i j).1 (hRoots i j).2 hOracle
    rcases List.mem_append.mp hz with hz | hz
    · obtain ⟨t, ht, rfl⟩ := hh.1.2.2 z hz
      exact point_safe _ (hDir i j) _ ht
    · obtain ⟨t, ht, rfl⟩ := hh.2.2.2 z hz
      exact point_safe _ (fun i' => by simpa using hDir i j i') _ ht
  · simp at hz

lemma measurable_proxySet (oracle : Vec d → Bool) (a : Vec d) (n : ℕ)
    (lo hi radius level : ℝ) :
    MeasurableSet (proxySet oracle a n lo hi radius level) := by
  unfold proxySet
  apply MeasurableSet.inter
  · have hx : shiftedBox a radius = ⋂ i, {z : Vec d | |z i - a i| ≤ radius} := by
      ext z
      simp [shiftedBox]
    rw [hx]
    exact MeasurableSet.iInter fun i => measurableSet_le (by fun_prop) measurable_const
  · refine measurableSet_le ?_ measurable_const
    unfold quadratic quadPart quadBilin linearPart
    simp only [Pi.sub_apply]
    fun_prop

lemma reconstruction_sandwich (oracle : Vec d → Bool) (a : Vec d)
    (A : Fin d → Fin d → ℝ) (b : Vec d) (n : ℕ) (lo hi radius : ℝ)
    (hlo : 0 < lo) (hhi : lo ≤ hi) (hr : 0 ≤ radius)
    (hSym : ∀ i j, A i j = A j i)
    (hPSD : ∀ x, 0 ≤ quadPart A x)
    (hBody : body a A b ⊆ shiftedBox a radius)
    (hRoots : ∀ i j, positiveInverse A b (sectionDirection i j) ∈ Icc lo hi ∧
      negativeInverse A b (sectionDirection i j) ∈ Icc lo hi)
    (hOracle : ∀ z, (quadratic A b (z-a) < 1 → oracle z = true) ∧
      (1 < quadratic A b (z-a) → oracle z = false))
    (hsmall : valueAccuracy d n lo hi radius < 1/2) :
    let e := valueAccuracy d n lo hi radius
    let L := proxySet oracle a n lo hi radius (1-e)
    let U := proxySet oracle a n lo hi radius (1+e)
    let s := (1+2*e)/(1-2*e)
    L ⊆ body a A b ∧ body a A b ⊆ U ∧ U ⊆ contract a s ⁻¹' L ∧ 1 ≤ s := by
  dsimp only
  let e := valueAccuracy d n lo hi radius
  let L := proxySet oracle a n lo hi radius (1-e)
  let U := proxySet oracle a n lo hi radius (1+e)
  let s : ℝ := (1+2*e)/(1-2*e)
  change L ⊆ body a A b ∧ body a A b ⊆ U ∧ U ⊆ contract a s ⁻¹' L ∧ 1 ≤ s
  have hδ : 0 ≤ sectionAccuracy n lo hi := div_nonneg (sub_nonneg.mpr hhi) (by positivity)
  have hhin : 0 ≤ hi := hlo.le.trans hhi
  have he : 0 ≤ e := by
    unfold e valueAccuracy
    exact mul_nonneg (by positivity) hδ
  have hsden : 0 < 1-2*e := by linarith only [hsmall]
  have hs : 1 ≤ s := (le_div_iff₀ hsden).2 (by linarith only [he])
  have hs0 : 0 < s := zero_lt_one.trans_le hs
  have hsInv : s⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hs
  have hscale : (1+2*e)/s = 1-2*e := by
    have hh : 0 < 1+2*e := by linarith only [he]
    dsimp only [s]
    field_simp
  let Ahat := foundA oracle a n lo hi
  let bhat := foundB oracle a n lo hi
  obtain ⟨ha,hb⟩ := quadratic_coeff_accuracy oracle a A b n lo hi hlo hhi hSym hPSD hRoots hOracle
  have hErr (z : Vec d) (hz : z ∈ shiftedBox a radius) :
      |quadratic Ahat bhat (z-a) - quadratic A b (z-a)| ≤ e :=
    quadratic_value_accuracy (z-a) radius _ hi hr hz ha hb
  refine ⟨?_, ?_, ?_, hs⟩
  · intro z hz
    have h := (abs_le.mp (hErr z hz.1)).1
    have h2 : quadratic Ahat bhat (z-a) ≤ 1-e := hz.2
    change quadratic A b (z-a) ≤ 1
    linarith only [h,h2]
  · intro z hz
    have hbox := hBody hz
    refine ⟨hbox, ?_⟩
    have h := (abs_le.mp (hErr z hbox)).2
    have h2 : quadratic A b (z-a) ≤ 1 := hz
    change quadratic Ahat bhat (z-a) ≤ 1+e
    linarith only [h,h2]
  · intro z hz
    have h := (abs_le.mp (hErr z hz.1)).1
    have h2 : quadratic Ahat bhat (z-a) ≤ 1+e := hz.2
    have h3 : quadratic A b (z-a) ≤ 1+2*e := by linarith only [h,h2]
    have hcoord : contract a s z - a = s⁻¹ • (z-a) := by
      ext i
      simp only [contract, Pi.sub_apply, Pi.smul_apply, smul_eq_mul, div_eq_mul_inv]
      ring
    have hb2 : contract a s z ∈ shiftedBox a radius := by
      intro i
      change |(contract a s z - a) i| ≤ radius
      rw [hcoord, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_pos (inv_pos.mpr hs0)]
      exact (mul_le_of_le_one_left (abs_nonneg _) hsInv).trans (hz.1 i)
    refine ⟨hb2, ?_⟩
    have h4 : quadratic A b (contract a s z - a) ≤ 1-2*e := by
      have hineq : s * quadratic A b (contract a s z-a) ≤ quadratic A b (z-a) := by
        rw [hcoord, quadratic, quadPart_smul, linearPart_smul, mul_add]
        change s * (s⁻¹ ^ 2 * quadPart A (z-a)) + s * (s⁻¹ * linearPart b (z-a)) ≤
          quadPart A (z-a) + linearPart b (z-a)
        rw [← mul_assoc, ← mul_assoc s, mul_inv_cancel₀ hs0.ne', one_mul,
          show s * s⁻¹ ^ 2 = s⁻¹ by field_simp]
        exact add_le_add (mul_le_of_le_one_left (hPSD _) hsInv) le_rfl
      rw [← hscale, le_div_iff₀ hs0]
      rw [mul_comm]
      exact hineq.trans h3
    have hh := (abs_le.mp (hErr (contract a s z) hb2)).2
    change quadratic Ahat bhat (contract a s z-a) ≤ 1-e
    linarith only [h4,hh]

end

end Cubature
