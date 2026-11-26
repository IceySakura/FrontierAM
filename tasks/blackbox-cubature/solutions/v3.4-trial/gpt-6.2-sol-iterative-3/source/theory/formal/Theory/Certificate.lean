import Theory.Reconstruct
import Theory.Gaussian

open scoped BigOperators ENNReal
open Set MeasureTheory

namespace Cubature

variable {d : ℕ}

theorem contraction_subset {a b : Coordinates d} {A : Coefficients d}
    (hA : ∀ x : Coordinates d, x ≠ 0 → 0 < quadratic A x)
    (R δ : ℝ) {T : ObservationTable d}
    (hδ : 0 ≤ δ) (hδ' : δ < 1/2)
    (houter : ∀ x, quadratic A x + linear b x ≤ 1 → inBox R x)
    (hfit : ∀ x, inBox R x →
      |(quadratic (fittedMatrix T) x + linear (fittedLinear T) x) -
        (quadratic A x + linear b x)| ≤ δ) :
    contract a (1-2*δ) '' ellipsoid a A b ⊆ certifiedSet a R δ T ∧
      certifiedSet a R δ T ⊆ ellipsoid a A b := by
  refine ⟨?_,?_⟩
  · rintro z ⟨y,hy,rfl⟩
    change quadratic A (y-a) + linear b (y-a) ≤ 1 at hy
    let t := 1-2*δ
    have ht : 0 < t := by dsimp [t]; linarith
    have ht₁ : t ≤ 1 := by dsimp [t]; linarith
    have hz : contract a t y - a = t • (y-a) := by
      ext i
      simp only [contract, Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
      ring
    have hbox := houter _ hy
    have hbx : inBox R (t • (y-a)) := by
      intro i
      simp only [Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_pos ht]
      exact (mul_le_mul_of_nonneg_right ht₁ (abs_nonneg _)).trans (by simpa using hbox i)
    change inBox R (contract a t y-a) ∧ _
    rw [hz]
    refine ⟨hbx,?_⟩
    have hq : 0 ≤ quadratic A (y-a) := by
      by_cases h : y-a=0
      · rw [h]
        simp [quadratic]
      · exact (hA _ h).le
    have hsq : t^2 ≤ t := by nlinarith [mul_nonneg ht.le (sub_nonneg.mpr ht₁)]
    have hbq : quadratic A (t • (y-a)) + linear b (t • (y-a)) ≤ t := by
      rw [quadratic_smul, linear_smul]
      calc
        t^2 * quadratic A (y-a) + t * linear b (y-a) ≤
            t * quadratic A (y-a) + t * linear b (y-a) :=
          add_le_add (mul_le_mul_of_nonneg_right hsq hq) le_rfl
        _ = t * (quadratic A (y-a) + linear b (y-a)) := by ring
        _ ≤ t := by simpa using mul_le_mul_of_nonneg_left hy ht.le
    have h := (abs_le.mp (hfit _ hbx)).2
    dsimp [t] at hbq h ⊢
    linarith
  · intro z hz
    change inBox R (z-a) ∧ _ at hz
    have hx := (abs_le.mp (hfit _ hz.1)).1
    change quadratic A (z-a) + linear b (z-a) ≤ 1
    have hh : quadratic (fittedMatrix T) (z-a) + linear (fittedLinear T) (z-a) ≤ 1-δ := hz.2
    linarith

theorem gaussian_box_finite (a : Coordinates d) (R : ℝ)
    {E : Set (Coordinates d)} (hE : MeasurableSet E)
    (hb : ∀ x ∈ E, inBox R (x-a)) : gaussian E ≠ ⊤ := by
  let l : Coordinates d := fun i => a i - R
  let u : Coordinates d := fun i => a i + R
  have hsu : E ⊆ Icc l u := by
    intro x hx
    have H := hb _ hx
    refine ⟨fun i => ?_,fun i => ?_⟩
    · have := (abs_le.mp (H i)).1
      dsimp [l] at *
      linarith
    · have := (abs_le.mp (H i)).2
      dsimp [u] at *
      linarith
  have vh : (volume : Measure (Coordinates d)) (Icc l u) ≠ ⊤ := by
    rw [Real.volume_Icc_pi]
    exact ENNReal.prod_ne_top (fun _ _ => ENNReal.ofReal_ne_top)
  have hv : (volume : Measure (Coordinates d)) E ≠ ⊤ :=
    ne_top_of_le_ne_top vh (measure_mono hsu)
  let c : ℝ := (Real.sqrt (2 * Real.pi))⁻¹ ^ d
  have hd (x : Coordinates d) : gaussianDensity x ≤ ENNReal.ofReal c := by
    unfold gaussianDensity
    apply ENNReal.ofReal_le_ofReal
    have h₀ : ∑ i : Fin d, (x i)^2 ≥ 0 := Finset.sum_nonneg (fun i _ => sq_nonneg (x i))
    have h₁ : Real.exp (-(∑ i : Fin d, (x i)^2)/2) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hc : 0 ≤ c := by dsimp [c]; positivity
    change c * _ ≤ c
    exact (mul_le_mul_of_nonneg_left h₁ hc).trans_eq (mul_one _)
  rw [gaussian, withDensity_apply _ hE]
  apply ne_top_of_le_ne_top ?_ (lintegral_mono hd)
  rw [lintegral_const, Measure.restrict_apply_univ]
  exact ENNReal.mul_ne_top ENNReal.ofReal_ne_top hv

theorem gaussian_ellipsoid_positive (a b : Coordinates d) (A : Coefficients d)
    {ρ : ℝ} (hρ : 0 < ρ)
    (hinner : ∀ x, inBox ρ x → quadratic A x + linear b x ≤ 1) :
    0 < gaussian (ellipsoid a A b) := by
  let l : Coordinates d := fun i => a i-ρ
  let u : Coordinates d := fun i => a i+ρ
  have hx : Icc l u ⊆ ellipsoid a A b := by
    rintro z ⟨hl,hu⟩
    apply hinner
    intro i
    have h₁ := hl i
    have h₂ := hu i
    dsimp [l,u] at h₁ h₂
    change |z i-a i| ≤ ρ
    exact abs_le.mpr ⟨by linarith,by linarith⟩
  have hv : 0 < (volume : Measure (Coordinates d)) (Icc l u) := by
    rw [Real.volume_Icc_pi]
    apply pos_of_ne_zero
    apply Finset.prod_ne_zero_iff.mpr
    intro i _
    apply ne_of_gt
    rw [ENNReal.ofReal_pos]
    dsimp [l,u]
    linarith
  apply pos_of_ne_zero
  intro h
  change (volume : Measure (Coordinates d)).withDensity gaussianDensity (ellipsoid a A b) = 0 at h
  have h0 := (withDensity_apply_eq_zero measurable_gaussianDensity).mp h
  have heq : {x : Coordinates d | gaussianDensity x ≠ 0} = univ :=
    eq_univ_of_forall (fun x => (gaussianDensity_pos x).ne')
  rw [heq, univ_inter] at h0
  exact hv.not_ge (h0 ▸ measure_mono hx)

end Cubature
