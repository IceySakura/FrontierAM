import Theory.TwoCoordinate
import Mathlib.Probability.Moments.Variance

open Set MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal ProbabilityTheory

namespace Cubature

noncomputable section

abbrev CellIndex (m k : ℕ) := Fin m → Fin k

def preintegrated {m : ℕ} (a lam : Vec 2) (g q : Vec m → ℝ) (x : Vec m) :=
  g x * ellipseProbability a lam (q x)

def afterTwoAxesLip (lam : Vec 2) (amp : ℝ) (Lg Lq : ℝ≥0) : ℝ :=
  (Lg : ℝ) + amp * (Lq : ℝ)/(2*Real.sqrt (lam 0)*Real.sqrt (lam 1))

def stratifiedAverage {m k : ℕ} (h : Vec m → ℝ) (samples : CellIndex m k → Vec m) : ℝ :=
  (∑ i, h (samples i)) / (k : ℝ)^m

lemma preintegrated_eq_integral {m : ℕ} (a lam : Vec 2) (g q : Vec m → ℝ) (x : Vec m) :
    preintegrated a lam g q x =
      ∫ u : Vec 2, {z : Vec 2 | ∑ i, lam i * (z i-a i)^2 ≤ q x}.indicator
        (fun _ => g x) u ∂gaussian 2 := by
  have hs : MeasurableSet {z : Vec 2 | ∑ i, lam i * (z i-a i)^2 ≤ q x} :=
    measurableSet_le (by fun_prop) measurable_const
  rw [integral_indicator_const _ hs]
  simp [preintegrated, ellipseProbability, smul_eq_mul, mul_comm]

lemma preintegrated_dist {m : ℕ} (a lam : Vec 2) (hlam : ∀ i, 0 < lam i)
    (g q : Vec m → ℝ) (M : ℝ) (Lg Lq : ℝ≥0)
    (hg : LipschitzWith Lg g) (hq : LipschitzWith Lq q) (hM : ∀ x, |g x| ≤ M)
    (x y : Vec m) :
    |preintegrated a lam g q x - preintegrated a lam g q y| ≤
      afterTwoAxesLip lam M Lg Lq * dist x y := by
  let p := ellipseProbability a lam
  have hP (r : ℝ) : 0 ≤ p r ∧ p r ≤ 1 := ⟨measureReal_nonneg, measureReal_le_one⟩
  have L₁ := hg.dist_le_mul x y
  have L₂ := hq.dist_le_mul x y
  rw [Real.dist_eq] at L₁ L₂
  have D : 0 < 2*Real.sqrt (lam 0)*Real.sqrt (lam 1) := by
    have h0 := hlam 0
    have h1 := hlam 1
    positivity
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM x)
  unfold preintegrated afterTwoAxesLip
  change |g x*p (q x) - g y*p (q y)| ≤ _
  rw [show g x*p (q x) - g y*p (q y) =
        (g x-g y)*p (q x) + g y*(p (q x)-p (q y)) by ring]
  calc
    _ ≤ |(g x-g y)*p (q x)| + |g y*(p (q x)-p (q y))| := abs_add_le _ _
    _ = |g x-g y| * p (q x) + |g y| * |p (q x)-p (q y)| := by
      rw [abs_mul, abs_mul, abs_of_nonneg (hP _).1]
    _ ≤ (Lg : ℝ)*dist x y + M*((Lq : ℝ)*dist x y /
        (2*Real.sqrt (lam 0)*Real.sqrt (lam 1))) := by
      apply add_le_add
      · exact (mul_le_of_le_one_right (abs_nonneg _) (hP _).2).trans L₁
      · exact mul_le_mul (hM y) ((ellipse_lipschitz a lam hlam _ _).trans
          (div_le_div_of_nonneg_right L₂ D.le)) (abs_nonneg _) hM0
    _ = _ := by ring

lemma preintegrated_continuous {m : ℕ} (a lam : Vec 2) (hlam : ∀ i, 0 < lam i)
    (g q : Vec m → ℝ) (M : ℝ) (Lg Lq : ℝ≥0)
    (hg : LipschitzWith Lg g) (hq : LipschitzWith Lq q) (hM : ∀ x, |g x| ≤ M) :
    Continuous (preintegrated a lam g q) := by
  have h0 := hlam 0
  have h1 := hlam 1
  have hM0 : 0 ≤ M := (abs_nonneg _).trans (hM 0)
  let L : ℝ≥0 := ⟨afterTwoAxesLip lam M Lg Lq, by unfold afterTwoAxesLip; positivity⟩
  have HL : LipschitzWith L (preintegrated a lam g q) := by
    apply LipschitzWith.of_dist_le_mul
    intro x y
    rw [Real.dist_eq]
    exact preintegrated_dist a lam hlam g q M Lg Lq hg hq hM x y
  exact HL.continuous

lemma stratified_moments {m k : ℕ} (hk : 0 < k) (h : Vec m → ℝ)
    (hMeas : Measurable h) (M Lip : ℝ) (hM : ∀ x, |h x| ≤ M) (hLip0 : 0 ≤ Lip)
    (hLip : ∀ x y, |h x-h y| ≤ Lip * dist x y)
    (μ : CellIndex m k → Measure (Vec m)) [∀ i, IsProbabilityMeasure (μ i)]
    (centers : CellIndex m k → Vec m)
    (hCell : ∀ i, ∀ᵐ x ∂μ i, dist x (centers i) ≤ 1/(2*(k : ℝ))) :
    (∫ sample, stratifiedAverage h sample ∂Measure.pi μ) =
      (∑ i, ∫ x, h x ∂μ i)/(k : ℝ)^m ∧
    variance (stratifiedAverage h) (Measure.pi μ) ≤ Lip^2/(4*(k : ℝ)^2*(k : ℝ)^m) := by
  have hk0 : 0 < (k : ℝ) := by exact_mod_cast hk
  let J := CellIndex m k
  let n : ℝ := (k : ℝ)^m
  have hn : 0 < n := pow_pos hk0 _
  let F (i : J) (ω : J → Vec m) := h (ω i)
  have hp (i : J) : MemLp h 2 (μ i) :=
    MemLp.of_bound hMeas.aestronglyMeasurable M (Filter.Eventually.of_forall (by simpa using hM))
  have hh (i : J) : Integrable (F i) (Measure.pi μ) :=
    ((hp i).comp_measurePreserving (measurePreserving_eval μ i)).integrable (by norm_num)
  constructor
  · unfold stratifiedAverage
    simp only [integral_div]
    congr 1
    change (∫ w : J → Vec m, ∑ i : J, F i w ∂Measure.pi μ) =
      ∑ i : J, ∫ x, h x ∂μ i
    rw [integral_finsetSum _ (fun i _ => hh i)]
    exact Finset.sum_congr rfl (fun i _ => integral_comp_eval hMeas.aestronglyMeasurable)
  · have hnume :
        variance (stratifiedAverage h) (Measure.pi μ) =
          n⁻¹ ^ 2 * ∑ i : J, variance h (μ i) := by
      have he : stratifiedAverage h = fun ω : J → Vec m => n⁻¹ * (∑ i : J, F i) ω := by
        funext w
        change (∑ i : J, h (w i))/n = n⁻¹ * (∑ i : J, F i) w
        simp only [Finset.sum_apply, F, div_eq_mul_inv]
        ring
      rw [he, variance_const_mul, variance_sum_pi (X := fun _ => h) hp]
    rw [hnume]
    have hV (i : J) : variance h (μ i) ≤ (Lip/(2*(k : ℝ)))^2 := by
      let z := h (centers i)
      let e := Lip/(2*(k : ℝ))
      have hx : ∀ᵐ x ∂μ i, h x ∈ Icc (z-e) (z+e) := by
        filter_upwards [hCell i] with x hx
        have H : |h x - z| ≤ e := by
          calc
            _ ≤ Lip * dist x (centers i) := hLip x (centers i)
            _ ≤ Lip * (1/(2*(k : ℝ))) := mul_le_mul_of_nonneg_left hx hLip0
            _ = _ := by unfold e; ring
        exact ⟨by linarith only [(abs_le.mp H).1], by linarith only [(abs_le.mp H).2]⟩
      refine (variance_le_sq_of_bounded hx hMeas.aemeasurable).trans_eq ?_
      change (((z+e)-(z-e))/2)^2 = e^2
      ring
    calc
      _ ≤ n⁻¹^2 * ∑ _ : J, (Lip/(2*(k : ℝ)))^2 :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => hV i) (sq_nonneg _)
      _ = Lip^2/(4*(k : ℝ)^2*n) := by
        have hcard : (Fintype.card J : ℝ) = n := by simp [J, CellIndex, n, Nat.cast_pow]
        simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, hcard]
        field_simp
        ring

end

end Cubature
