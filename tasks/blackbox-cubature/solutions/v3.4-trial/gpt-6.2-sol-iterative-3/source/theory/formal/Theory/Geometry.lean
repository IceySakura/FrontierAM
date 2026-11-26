import Theory.Model

open scoped BigOperators
open Set

namespace Cubature

variable {d : ℕ}

theorem quadratic_smul (A : Coefficients d) (t : ℝ) (x : Coordinates d) :
    quadratic A (t • x) = t^2 * quadratic A x := by
  simp only [quadratic, Finset.mul_sum, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem linear_smul (b : Coordinates d) (t : ℝ) (x : Coordinates d) :
    linear b (t • x) = t * linear b x := by
  simp only [linear, Finset.mul_sum, Pi.smul_apply, smul_eq_mul]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem chord_abs (i j k : Fin d) (s : Bool) :
    |orientedChord i j s k| ≤ 1 := by
  by_cases hij : i = j <;> by_cases hki : k = i <;> by_cases hkj : k = j <;>
    cases s <;> simp_all [orientedChord, chord, coordinate] <;> norm_num

theorem chord_at_left (i j : Fin d) (s : Bool) :
    |orientedChord i j s i| = 1 := by
  by_cases h : i = j <;> cases s <;> simp_all [orientedChord, chord, coordinate]

theorem chord_ne_zero (i j : Fin d) (s : Bool) :
    orientedChord i j s ≠ 0 := by
  intro h
  have hh := chord_at_left i j s
  rw [h] at hh
  norm_num at hh

theorem quadratic_coordinate (A : Coefficients d) (i : Fin d) :
    quadratic A (coordinate i) = A i i := by
  simp [quadratic, coordinate, mul_ite, ite_mul]

theorem linear_coordinate (b : Coordinates d) (i : Fin d) :
    linear b (coordinate i) = b i := by
  simp [linear, coordinate, mul_ite]

theorem quadratic_chord (A : Coefficients d) (hs : ∀ i j, A i j = A j i)
    {i j : Fin d} (h : i ≠ j) :
    quadratic A (chord i j) = A i i + A j j + 2 * A i j := by
  simp [chord, h, quadratic, Pi.add_apply, coordinate, mul_add, add_mul,
    Finset.sum_add_distrib, mul_ite, ite_mul]
  rw [hs j i]
  ring

theorem continuous_quadratic (A : Coefficients d) :
    Continuous (quadratic A) := by
  unfold quadratic
  fun_prop

theorem continuous_linear (b : Coordinates d) :
    Continuous (linear b) := by
  unfold linear
  fun_prop

theorem measurable_ellipsoid (A : Coefficients d) (b a : Coordinates d) :
    MeasurableSet (ellipsoid a A b) := by
  unfold ellipsoid
  have hc : Continuous (fun z : Coordinates d => quadratic A (z-a) + linear b (z-a)) :=
    ((continuous_quadratic A).comp (continuous_id.sub continuous_const)).add
      ((continuous_linear b).comp (continuous_id.sub continuous_const))
  exact isClosed_le hc continuous_const |>.measurableSet

theorem measurable_certifiedSet (a : Coordinates d) (R δ : ℝ) (T : ObservationTable d) :
    MeasurableSet (certifiedSet a R δ T) := by
  have h₁ : MeasurableSet {z : Coordinates d | inBox R (z-a)} := by
    change MeasurableSet {z : Coordinates d | ∀ i, |(z-a) i| ≤ R}
    rw [show {z : Coordinates d | ∀ i, |(z-a) i| ≤ R} =
        ⋂ i, {z : Coordinates d | |(z-a) i| ≤ R} by ext z; simp]
    apply MeasurableSet.iInter
    intro i
    have hc : Continuous (fun z : Coordinates d => |(z-a) i|) := by fun_prop
    exact isClosed_le hc continuous_const |>.measurableSet
  have h₂ : Continuous (fun z : Coordinates d =>
      quadratic (fittedMatrix T) (z-a) + linear (fittedLinear T) (z-a)) :=
    ((continuous_quadratic _).comp (continuous_id.sub continuous_const)).add
      ((continuous_linear _).comp (continuous_id.sub continuous_const))
  exact h₁.inter (isClosed_le h₂ continuous_const |>.measurableSet)

theorem positive_quadratic_factors (c b : ℝ) (hc : 0 < c) :
    ∃ p q : ℝ, 0 < p ∧ 0 < q ∧ p*q = c ∧ p-q = b := by
  have hd : 0 < b^2 + 4*c := by positivity
  have hh := Real.sq_sqrt hd.le
  have hs := Real.sqrt_nonneg (b^2 + 4*c)
  have hpos : |b| < Real.sqrt (b^2 + 4*c) := by
    rw [← sq_lt_sq₀ (abs_nonneg b) hs, sq_abs, hh]
    linarith
  refine ⟨(Real.sqrt (b^2+4*c)+b)/2, (Real.sqrt (b^2+4*c)-b)/2, ?_, ?_, ?_, ?_⟩
  · linarith [neg_abs_le b]
  · linarith [le_abs_self b]
  · nlinarith
  · ring

private theorem below_root {p q s : ℝ} (hp : 0 < p) (hq : 0 < q) (hs : 0 ≤ s) :
    p*q*s^2 + (p-q)*s < 1 ↔ s < p⁻¹ := by
  have hm : 0 < 1+q*s := by positivity
  have hid : 1 - (p*q*s^2 + (p-q)*s) = (1-p*s)*(1+q*s) := by ring
  rw [← sub_pos, hid, mul_pos_iff_of_pos_right hm, sub_pos,
    inv_eq_one_div, lt_div_iff₀ hp, mul_comm]

private theorem above_root {p q s : ℝ} (hp : 0 < p) (hq : 0 < q) (hs : 0 ≤ s) :
    1 < p*q*s^2 + (p-q)*s ↔ p⁻¹ < s := by
  have hm : 0 < 1+q*s := by positivity
  have hid : p*q*s^2 + (p-q)*s - 1 = (p*s-1)*(1+q*s) := by ring
  rw [← sub_pos, hid, mul_pos_iff_of_pos_right hm, sub_pos,
    inv_eq_one_div, div_lt_iff₀ hp, mul_comm]

theorem chord_factors (A : Coefficients d) (b : Coordinates d) (i j : Fin d)
    (hA : ∀ x : Coordinates d, x ≠ 0 → 0 < quadratic A x) :
    ∃ p q : ℝ, 0 < p ∧ 0 < q ∧ p*q = quadratic A (chord i j) ∧
      p-q = linear b (chord i j) := by
  apply positive_quadratic_factors
  apply hA
  simpa [orientedChord] using chord_ne_zero i j true

theorem root_bounds_and_oracle (f : Coordinates d → ℝ) (J : ℝ) (a u : Coordinates d)
    (A : Coefficients d) (b : Coordinates d) (ρ R p q : ℝ)
    (hρ : 0 < ρ) (hR : ρ ≤ R) (hp : 0 < p) (hq : 0 < q)
    (hu : ∀ k, |u k| ≤ 1) (hu₁ : ∃ k, |u k| = 1)
    (hinner : ∀ x, inBox ρ x → quadratic A x + linear b x ≤ 1)
    (houter : ∀ x, quadratic A x + linear b x ≤ 1 → inBox R x)
    (hc : p*q = quadratic A u) (hb : p-q = linear b u)
    (hJ : 0 < J) (hf : JumpOracle f J a A b) :
    ρ ≤ p⁻¹ ∧ p⁻¹ ≤ R ∧
      (∀ s, ρ ≤ s → s < p⁻¹ → 0 < f (a+s • u)) ∧
      (∀ s, ρ ≤ s → p⁻¹ < s → ¬0 < f (a+s • u)) := by
  have he (s : ℝ) : quadratic A (s • u) + linear b (s • u) =
      p*q*s^2 + (p-q)*s := by
    rw [quadratic_smul, linear_smul, ← hc, ← hb]
    ring
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hi : inBox ρ (ρ • u) := by
      intro k
      simp only [Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_pos hρ]
      simpa using mul_le_mul_of_nonneg_left (hu k) hρ.le
    have e := hinner _ hi
    rw [he, ← not_lt, above_root hp hq hρ.le, not_lt] at e
    exact e
  · by_contra! ht
    let s := (R+p⁻¹)/2
    have hs : R < s := by dsimp [s]; linarith
    have hsp : s < p⁻¹ := by dsimp [s]; linarith
    have hsp' : 0 < s := by linarith
    have e : quadratic A (s • u) + linear b (s • u) < 1 := by
      rw [he, below_root hp hq hsp'.le]
      exact hsp
    obtain ⟨k,hk⟩ := hu₁
    have hv := houter _ e.le k
    simp [Pi.smul_apply, abs_mul, abs_of_pos hsp', hk] at hv
    linarith
  · intro s hsr hsp
    have hsn : 0 ≤ s := by linarith
    have e : quadratic A (s • u) + linear b (s • u) < 1 := by
      rw [he, below_root hp hq hsn]
      exact hsp
    simpa [(hf _).1 e] using hJ
  · intro s hsr hsp
    have hsn : 0 ≤ s := by linarith
    have e : 1 < quadratic A (s • u) + linear b (s • u) := by
      rw [he, above_root hp hq hsn]
      exact hsp
    simp [(hf _).2 e]

theorem raySearch_enclosure (f : Coordinates d → ℝ) (a u : Coordinates d)
    (ρ R r : ℝ) (m : ℕ) (lo hi : ℝ)
    (hl : ρ ≤ lo) (hh : hi ≤ R) (hrl : lo ≤ r) (hrh : r ≤ hi)
    (hbelow : ∀ s, ρ ≤ s → s < r → 0 < f (a+s • u))
    (habove : ∀ s, ρ ≤ s → r < s → ¬0 < f (a+s • u)) :
    let v := raySearch f a u m lo hi
    ρ ≤ v.lo ∧ v.lo ≤ r ∧ r ≤ v.hi ∧ v.hi ≤ R ∧
      v.hi - v.lo = (hi-lo)/(2:ℝ)^m ∧
      v.points.length = m ∧
      ∀ z ∈ v.points, ∃ s ∈ Icc ρ R, z = a+s • u := by
  induction m generalizing lo hi with
  | zero => simpa [raySearch] using And.intro hl (And.intro hrl (And.intro hrh hh))
  | succ n ih =>
    let mid := (lo+hi)/2
    have hm : mid ∈ Icc ρ R := ⟨by dsimp [mid]; linarith, by dsimp [mid]; linarith⟩
    change let v := if 0 < f (a+mid • u) then raySearch f a u n mid hi
      else raySearch f a u n lo mid
      ρ ≤ v.lo ∧ v.lo ≤ r ∧ r ≤ v.hi ∧ v.hi ≤ R ∧
        v.hi - v.lo = (hi-lo)/(2:ℝ)^(n+1) ∧
        ((a+mid • u) :: v.points).length = n+1 ∧
        ∀ z ∈ ((a+mid • u) :: v.points), ∃ s ∈ Icc ρ R, z = a+s • u
    by_cases h : 0 < f (a+mid • u)
    · have hmr : mid ≤ r := le_of_not_gt (fun k => habove _ hm.1 k h)
      have q := ih mid hi hm.1 hh hmr hrh
      simp only [h, ↓reduceIte]
      rcases q with ⟨q1,q2,q3,q4,q5,q6,q7⟩
      refine ⟨q1,q2,q3,q4, ?_, by simp [q6], ?_⟩
      · rw [q5, pow_succ]
        dsimp [mid]
        ring
      · intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · exact ⟨mid,hm,rfl⟩
        · exact q7 z hz
    · have hmr : r ≤ mid := le_of_not_gt (fun k => h (hbelow _ hm.1 k))
      have q := ih lo mid hl hm.2 hrl hmr
      simp only [h, ↓reduceIte]
      rcases q with ⟨q1,q2,q3,q4,q5,q6,q7⟩
      refine ⟨q1,q2,q3,q4, ?_, by simp [q6], ?_⟩
      · rw [q5, pow_succ]
        dsimp [mid]
        ring
      · intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · exact ⟨mid,hm,rfl⟩
        · exact q7 z hz

theorem reciprocal_error {v : Bracket d} {r ρ e : ℝ} (hρ : 0 < ρ)
    (hl : ρ ≤ v.lo) (hrl : v.lo ≤ r) (hrh : r ≤ v.hi)
    (he : v.hi-v.lo = e) :
    0 < reciprocal v ∧ reciprocal v ≤ ρ⁻¹ ∧ 0 < r⁻¹ ∧ r⁻¹ ≤ ρ⁻¹ ∧
      |reciprocal v - r⁻¹| ≤ e / ρ^2 := by
  let m := (v.lo+v.hi)/2
  have hm : ρ ≤ m := by dsimp [m]; linarith
  have hr : ρ ≤ r := hl.trans hrl
  have hm' : 0 < m := hρ.trans_le hm
  have hr' : 0 < r := hρ.trans_le hr
  change 0 < m⁻¹ ∧ m⁻¹ ≤ ρ⁻¹ ∧ 0 < r⁻¹ ∧ r⁻¹ ≤ ρ⁻¹ ∧
    |m⁻¹-r⁻¹| ≤ e/ρ^2
  refine ⟨inv_pos.mpr hm', inv_anti₀ hρ hm, inv_pos.mpr hr', inv_anti₀ hρ hr, ?_⟩
  have hid : m⁻¹-r⁻¹ = (r-m)/(m*r) := by field_simp
  have hmr : ρ^2 ≤ m*r := by simpa [pow_two] using mul_le_mul hm hr hρ.le hm'.le
  have he' : 0 ≤ e := by linarith
  rw [hid, abs_div, abs_of_pos (mul_pos hm' hr')]
  exact (div_le_div_of_nonneg_right (abs_le.mpr ⟨by dsimp [m]; linarith,
    by dsimp [m]; linarith⟩) (mul_nonneg hm'.le hr'.le)).trans
      (div_le_div_of_nonneg_left he' (by positivity) hmr)

end Cubature
