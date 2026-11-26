import Theory.Geometry

open scoped BigOperators
open Set

namespace Cubature

variable {d : ℕ}

private theorem abs_sub_bound (x y : ℝ) : |x-y| ≤ |x| + |y| := by
  simpa [sub_eq_add_neg] using abs_add_le x (-y)

section Reconstruction

variable {f : Coordinates d → ℝ} {J ρ R : ℝ} {a b : Coordinates d}
  {A : Coefficients d} {m : ℕ}

theorem observe_ray (i j : Fin d) (s : Bool)
    (hρ : 0 < ρ) (hR : ρ ≤ R)
    (hinner : ∀ x, inBox ρ x → quadratic A x + linear b x ≤ 1)
    (houter : ∀ x, quadratic A x + linear b x ≤ 1 → inBox R x)
    (hp : 0 < quadratic A (orientedChord i j s))
    (hJ : 0 < J) (hf : JumpOracle f J a A b) :
    ∃ p q : ℝ, 0 < p ∧ 0 < q ∧ p*q = quadratic A (orientedChord i j s) ∧
      p-q = linear b (orientedChord i j s) ∧
      let W := observe f a ρ R m (i,j,s)
      ρ ≤ W.lo ∧ W.lo ≤ p⁻¹ ∧ p⁻¹ ≤ W.hi ∧ W.hi ≤ R ∧
      W.hi-W.lo = bracketWidth ρ R m ∧ W.points.length = m ∧
      ∀ z ∈ W.points, ∃ r ∈ Icc ρ R, z = a+r • orientedChord i j s := by
  obtain ⟨p,q,hp,hq,hc,hb⟩ := positive_quadratic_factors _ (linear b (orientedChord i j s)) hp
  obtain ⟨hl,hu,hbelow,habove⟩ :=
    root_bounds_and_oracle f J a _ A b ρ R p q hρ hR hp hq (chord_abs i j · s)
      ⟨i,chord_at_left i j s⟩ hinner houter hc hb hJ hf
  refine ⟨p,q,hp,hq,hc,hb,?_⟩
  exact raySearch_enclosure f a _ ρ R p⁻¹ m ρ R le_rfl le_rfl hl hu hbelow habove

theorem fit_errors
    (hρ : 0 < ρ) (hR : ρ ≤ R)
    (hinner : ∀ x, inBox ρ x → quadratic A x + linear b x ≤ 1)
    (houter : ∀ x, quadratic A x + linear b x ≤ 1 → inBox R x)
    (hA : ∀ x : Coordinates d, x ≠ 0 → 0 < quadratic A x)
    (hSym : ∀ i j, A i j = A j i)
    (hJ : 0 < J) (hf : JumpOracle f J a A b) :
    let T := observe f a ρ R m
    let e := bracketWidth ρ R m
    (∀ i j, |fittedMatrix T i j - A i j| ≤ 3*e/ρ^3) ∧
    (∀ i, |fittedLinear T i - b i| ≤ 2*e/ρ^2) := by
  let T := observe f a ρ R m
  let e := bracketWidth ρ R m
  have he : 0 ≤ e := by dsimp [e,bracketWidth]; positivity
  have chordError (i j : Fin d) :
      |observedQuadratic T i j - quadratic A (chord i j)| ≤ 2*e/ρ^3 ∧
      |(reciprocal (T (i,j,true)) - reciprocal (T (i,j,false))) -
          linear b (chord i j)| ≤ 2*e/ρ^2 := by
    obtain ⟨p,q,hp,hq,hc,hb⟩ := chord_factors A b i j hA
    have fwd := root_bounds_and_oracle f J a (chord i j) A b ρ R p q hρ hR hp hq
      (by simpa [orientedChord] using (fun k => chord_abs i j k true))
      ⟨i, by simpa [orientedChord] using chord_at_left i j true⟩
      hinner houter hc hb hJ hf
    have back := root_bounds_and_oracle f J a (-chord i j) A b ρ R q p hρ hR hq hp
      (by simpa [orientedChord] using (fun k => chord_abs i j k false))
      ⟨i, by simpa [orientedChord] using chord_at_left i j false⟩ hinner houter
      (by rw [show -chord i j = (-1:ℝ) • chord i j by simp, quadratic_smul]
          nlinarith)
      (by rw [show -chord i j = (-1:ℝ) • chord i j by simp, linear_smul]
          linarith) hJ hf
    obtain ⟨l1,h1,hfb1,hfa1⟩ := fwd
    obtain ⟨l2,h2,hfb2,hfa2⟩ := back
    obtain ⟨s1,s2,s3,_,s5,_,_⟩ :=
      raySearch_enclosure f a (chord i j) ρ R p⁻¹ m ρ R le_rfl le_rfl l1 h1 hfb1 hfa1
    obtain ⟨t1,t2,t3,_,t5,_,_⟩ :=
      raySearch_enclosure f a (-chord i j) ρ R q⁻¹ m ρ R le_rfl le_rfl l2 h2 hfb2 hfa2
    have E1 := reciprocal_error hρ s1 s2 s3 s5
    have E2 := reciprocal_error hρ t1 t2 t3 t5
    rw [inv_inv] at E1 E2
    change 0 < reciprocal (T (i,j,true)) ∧ reciprocal (T (i,j,true)) ≤ ρ⁻¹ ∧
      0 < p ∧ p ≤ ρ⁻¹ ∧ |reciprocal (T (i,j,true))-p| ≤ e/ρ^2 at E1
    change 0 < reciprocal (T (i,j,false)) ∧ reciprocal (T (i,j,false)) ≤ ρ⁻¹ ∧
      0 < q ∧ q ≤ ρ⁻¹ ∧ |reciprocal (T (i,j,false))-q| ≤ e/ρ^2 at E2
    rcases E1 with ⟨u1,u2,_,u4,u5⟩
    rcases E2 with ⟨v1,v2,_,_,v5⟩
    let pp := reciprocal (T (i,j,true))
    let qq := reciprocal (T (i,j,false))
    change |pp*qq-quadratic A (chord i j)| ≤ _ ∧
      |pp-qq-linear b (chord i j)| ≤ _
    refine ⟨?_,?_⟩
    · rw [← hc, show pp*qq-p*q = (pp-p)*qq+p*(qq-q) by ring]
      calc
        |(pp-p)*qq+p*(qq-q)| ≤ |(pp-p)*qq| + |p*(qq-q)| := abs_add_le _ _
        _ = |pp-p| * qq + p * |qq-q| := by rw [abs_mul, abs_mul, abs_of_pos hp, abs_of_pos v1]
        _ ≤ (e/ρ^2)*ρ⁻¹ + ρ⁻¹*(e/ρ^2) := by
          exact add_le_add
            (mul_le_mul u5 v2 v1.le (by positivity))
            (mul_le_mul u4 v5 (abs_nonneg _) (by positivity))
        _ = 2*e/ρ^3 := by field_simp; ring
    · rw [← hb, show pp-qq-(p-q) = (pp-p)-(qq-q) by ring]
      calc
        |(pp-p)-(qq-q)| ≤ |pp-p| + |qq-q| := abs_sub_bound _ _
        _ ≤ e/ρ^2 + e/ρ^2 := add_le_add u5 v5
        _ = 2*e/ρ^2 := by ring
  change (∀ i j, |fittedMatrix T i j - A i j| ≤ 3*e/ρ^3) ∧
    (∀ i, |fittedLinear T i - b i| ≤ 2*e/ρ^2)
  refine ⟨?_,?_⟩
  · intro i j
    have hdiag (i : Fin d) : |observedQuadratic T i i - A i i| ≤ 2*e/ρ^3 := by
      simpa [chord, quadratic_coordinate] using (chordError i i).1
    by_cases hij : i=j
    · subst j
      simpa [fittedMatrix] using (hdiag i).trans (by
        apply div_le_div_of_nonneg_right _ (by positivity)
        linarith)
    · simp only [fittedMatrix, hij, ↓reduceIte]
      have h := (chordError i j).1
      rw [quadratic_chord A hSym hij] at h
      let u := observedQuadratic T i j - (A i i + A j j + 2 * A i j)
      let v := observedQuadratic T i i - A i i
      let w := observedQuadratic T j j - A j j
      rw [show (observedQuadratic T i j-observedQuadratic T i i-observedQuadratic T j j)/2 -
        A i j = (u-v-w)/2 by dsimp [u,v,w]; ring, abs_div, abs_of_pos (by norm_num : (0:ℝ)<2)]
      have hh := (abs_sub_bound (u-v) w).trans (add_le_add (abs_sub_bound u v) (le_refl |w|))
      have hhh := (add_le_add (add_le_add h (hdiag i)) (hdiag j))
      exact (div_le_div_of_nonneg_right (hh.trans hhh) (by norm_num)).trans (by ring_nf; rfl)
  · intro i
    simpa [fittedLinear, chord, linear_coordinate] using (chordError i i).2

theorem evaluation_error (A B : Coefficients d) (b c x : Coordinates d)
    {M N R : ℝ} (hR : 0 ≤ R) (hM : 0 ≤ M) (hN : 0 ≤ N) (hx : inBox R x)
    (hm : ∀ i j, |B i j-A i j| ≤ M) (hn : ∀ i, |c i-b i| ≤ N) :
    |(quadratic B x + linear c x) - (quadratic A x + linear b x)| ≤
      (d:ℝ)^2*M*R^2 + (d:ℝ)*N*R := by
  have identity : (quadratic B x + linear c x) - (quadratic A x + linear b x) =
      (∑ i, ∑ j, (B i j-A i j)*x i*x j) + ∑ i, (c i-b i)*x i := by
    simp only [quadratic, linear, sub_mul, Finset.sum_sub_distrib]
    ring
  rw [identity]
  apply (abs_add_le _ _).trans
  apply add_le_add
  · calc
      |∑ i, ∑ j, (B i j-A i j)*x i*x j| ≤ ∑ i, |∑ j, (B i j-A i j)*x i*x j| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i : Fin d, ∑ j : Fin d, M*R*R := by
        apply Finset.sum_le_sum
        intro i _
        apply (Finset.abs_sum_le_sum_abs _ _).trans
        apply Finset.sum_le_sum
        intro j _
        rw [abs_mul, abs_mul]
        exact mul_le_mul (mul_le_mul (hm i j) (hx i) (abs_nonneg _) hM)
          (hx j) (abs_nonneg _) (mul_nonneg hM hR)
      _ = (d:ℝ)^2*M*R^2 := by simp; ring
  · calc
      |∑ i, (c i-b i)*x i| ≤ ∑ i, |(c i-b i)*x i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ : Fin d, N*R := by
        apply Finset.sum_le_sum
        intro i _
        rw [abs_mul]
        exact mul_le_mul (hn i) (hx i) (abs_nonneg _) hN
      _ = (d:ℝ)*N*R := by simp; ring

end Reconstruction

end Cubature
