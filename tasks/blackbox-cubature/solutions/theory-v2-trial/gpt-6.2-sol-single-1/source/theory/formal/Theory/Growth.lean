import Theory.Setup
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.RingTheory.Polynomial.Hermite.Gaussian

/-! Polynomial bounds, Gaussian moments and bounded Gaussian boundary factors. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace Cubature

section PolynomialGrowth

variable {E : Type*} [SeminormedAddGroup E] {f g : E → ℝ}

lemma polynomialGrowth_const (c : ℝ) : PolynomialGrowth (fun _ : E => c) :=
  ⟨|c|, abs_nonneg _, 0, fun _ => by simp⟩

lemma PolynomialGrowth.add (hf : PolynomialGrowth f) (hg : PolynomialGrowth g) :
    PolynomialGrowth (fun x => f x + g x) := by
  obtain ⟨C, hC, n, hn⟩ := hf
  obtain ⟨D, hD, m, hm⟩ := hg
  refine ⟨C + D, add_nonneg hC hD, max n m, fun x => ?_⟩
  calc
    |f x + g x| ≤ |f x| + |g x| := abs_add_le _ _
    _ ≤ C * (1 + ‖x‖) ^ n + D * (1 + ‖x‖) ^ m := add_le_add (hn x) (hm x)
    _ ≤ (C + D) * (1 + ‖x‖) ^ max n m := by
      rw [add_mul]
      have h : 1 ≤ 1 + ‖x‖ := le_add_of_nonneg_right (norm_nonneg x)
      exact add_le_add (mul_le_mul_of_nonneg_left (pow_le_pow_right₀ h (le_max_left _ _)) hC)
        (mul_le_mul_of_nonneg_left (pow_le_pow_right₀ h (le_max_right _ _)) hD)

lemma PolynomialGrowth.neg (h : PolynomialGrowth f) : PolynomialGrowth (fun x => -f x) := by
  obtain ⟨C, hC, n, hn⟩ := h
  exact ⟨C, hC, n, fun x => by simpa using hn x⟩

lemma PolynomialGrowth.sub (hf : PolynomialGrowth f) (hg : PolynomialGrowth g) :
    PolynomialGrowth (fun x => f x - g x) := by
  simpa only [sub_eq_add_neg] using hf.add hg.neg

lemma PolynomialGrowth.mul (hf : PolynomialGrowth f) (hg : PolynomialGrowth g) :
    PolynomialGrowth (fun x => f x * g x) := by
  obtain ⟨C, hC, n, hn⟩ := hf
  obtain ⟨D, hD, m, hm⟩ := hg
  refine ⟨C * D, mul_nonneg hC hD, n + m, fun x => ?_⟩
  calc
    |f x * g x| = |f x| * |g x| := abs_mul _ _
    _ ≤ (C * (1 + ‖x‖) ^ n) * (D * (1 + ‖x‖) ^ m) :=
      mul_le_mul (hn x) (hm x) (abs_nonneg _) (mul_nonneg hC (by positivity))
    _ = C * D * (1 + ‖x‖) ^ (n + m) := by rw [pow_add]; ring

lemma PolynomialGrowth.pow (hf : PolynomialGrowth f) (k : ℕ) :
    PolynomialGrowth (fun x => f x ^ k) := by
  induction k with
  | zero => simpa using (polynomialGrowth_const (E := E) 1)
  | succ k ih => simpa only [pow_succ] using ih.mul hf

lemma polynomialGrowth_finset_sum {ι : Type*} (s : Finset ι) (u : ι → E → ℝ)
    (h : ∀ i ∈ s, PolynomialGrowth (u i)) :
    PolynomialGrowth (fun x => ∑ i ∈ s, u i x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using (polynomialGrowth_const (E := E) 0)
  | insert a s ha ih =>
    simp only [Finset.sum_insert ha]
    exact (h a (Finset.mem_insert_self _ _)).add (ih fun i hi => h i (Finset.mem_insert_of_mem hi))

lemma PolynomialGrowth.comp_coord {ι : Type*} [Fintype ι] {u : ℝ → ℝ}
    (hu : PolynomialGrowth u) (i : ι) : PolynomialGrowth (fun x : ι → ℝ => u (x i)) := by
  obtain ⟨C, hC, k, hk⟩ := hu
  refine ⟨C, hC, k, fun x => (hk (x i)).trans ?_⟩
  have hh : ‖x i‖ ≤ ‖x‖ := norm_le_pi_norm x i
  gcongr

lemma polynomialGrowth_id : PolynomialGrowth (fun x : ℝ => x) := by
  refine ⟨1, zero_le_one, 1, fun x => ?_⟩
  simp [Real.norm_eq_abs]

lemma polynomialGrowth_truncatedPower (n : ℕ) : PolynomialGrowth (truncatedPower n) := by
  have h : PolynomialGrowth (fun t : ℝ => max 0 t) := by
    refine ⟨1, zero_le_one, 1, fun x => ?_⟩
    rw [pow_one, one_mul, abs_of_nonneg (le_max_left _ _), Real.norm_eq_abs]
    exact max_le (by positivity) ((le_abs_self _).trans (le_add_of_nonneg_left zero_le_one))
  change PolynomialGrowth (fun t => (max 0 t) ^ (n + 1) / ((n + 1).factorial : ℝ))
  simpa only [div_eq_mul_inv] using
    (h.pow (n + 1)).mul (polynomialGrowth_const ((↑(n + 1).factorial : ℝ)⁻¹))

end PolynomialGrowth

section Gaussian

private lemma memLp_one_add_pi_norm {ι : Type*} [Fintype ι] (p : ℝ≥0∞) (hp : p ≠ ⊤) :
    MemLp (fun x : ι → ℝ => 1 + ‖x‖) p (gaussianProduct ι) := by
  have he (i : ι) : MemLp (fun x : ι → ℝ => ‖x i‖) p (gaussianProduct ι) := by
    have h := (memLp_id_gaussianReal' (μ := 0) (v := 1) p hp).comp_measurePreserving
      (measurePreserving_eval (fun _ : ι => gaussianReal 0 1) i)
    exact h.norm
  have hs : MemLp (fun x : ι → ℝ => ∑ i, ‖x i‖) p (gaussianProduct ι) :=
    memLp_finsetSum Finset.univ (fun i _ => he i)
  refine (memLp_const (1 : ℝ)).add (hs.mono' (continuous_norm.aestronglyMeasurable) ?_)
  apply ae_of_all
  intro x
  rw [Real.norm_of_nonneg (norm_nonneg x)]
  refine (pi_norm_le_iff_of_nonneg (by positivity)).mpr fun i => ?_
  exact Finset.single_le_sum (fun j _ => norm_nonneg (x j)) (Finset.mem_univ i)

lemma PolynomialGrowth.memLp_gaussianProduct {ι : Type*} [Fintype ι] {f : (ι → ℝ) → ℝ}
    (hg : PolynomialGrowth f) (hf : Continuous f) (p : ℝ≥0∞) (hp : p ≠ ⊤) :
    MemLp f p (gaussianProduct ι) := by
  obtain ⟨C, hC, n, hn⟩ := hg
  have hpow : MemLp (fun x : ι → ℝ => (1 + ‖x‖) ^ n) p (gaussianProduct ι) := by
    by_cases hn0 : n = 0
    · simpa [hn0] using (memLp_const (μ := gaussianProduct ι) (p := p) (1 : ℝ))
    have htop : (↑n : ℝ≥0∞) ≠ ⊤ := by simp
    have hzero : (↑n : ℝ≥0∞) ≠ 0 := by exact_mod_cast hn0
    have hh := (memLp_one_add_pi_norm (ι := ι) (p * ↑n)
      (ENNReal.mul_ne_top hp htop)).norm_rpow_div (↑n)
    rw [ENNReal.mul_div_cancel_right hzero htop] at hh
    convert hh using 1
    funext x
    rw [Real.norm_of_nonneg (show 0 ≤ 1 + ‖x‖ by positivity), ENNReal.toReal_natCast, Real.rpow_natCast]
  refine (hpow.const_mul C).mono' hf.aestronglyMeasurable (ae_of_all _ ?_)
  simpa only [Real.norm_eq_abs] using hn

private lemma polynomialGrowth_hermite (c : ℕ) :
    PolynomialGrowth (fun t : ℝ => Polynomial.aeval t (Polynomial.hermite c)) := by
  simp only [Polynomial.aeval_eq_sum_range]
  refine polynomialGrowth_finset_sum _ _ fun i _ => ?_
  simpa only [zsmul_eq_mul] using
    (polynomialGrowth_const ((Polynomial.hermite c).coeff i : ℝ)).mul (polynomialGrowth_id.pow i)

lemma exists_density_polynomial (c : ℕ) :
    ∃ H : ℝ → ℝ, Continuous H ∧ PolynomialGrowth H ∧
      ∀ t, iteratedDeriv c (gaussianPDFReal 0 1) t = H t * Real.exp (-(t ^ 2 / 2)) := by
  let C := (Real.sqrt (2 * Real.pi))⁻¹
  let H : ℝ → ℝ := fun t => (C * (-1) ^ c) * Polynomial.aeval t (Polynomial.hermite c)
  refine ⟨H, ?_, (polynomialGrowth_const _).mul (polynomialGrowth_hermite c), ?_⟩
  · simp only [H, Polynomial.aeval_eq_sum_range]
    fun_prop
  · intro t
    have he : gaussianPDFReal 0 1 = fun u => C * Real.exp (-(u ^ 2 / 2)) := by
      ext u
      simp [gaussianPDFReal, C, neg_div]
    rw [he, iteratedDeriv_const_mul_field, iteratedDeriv_eq_iterate,
      Polynomial.deriv_gaussian_eq_hermite_mul_gaussian]
    simp only [H]
    ring

lemma PolynomialGrowth.gaussian_damped_bound {f : ℝ → ℝ} (hg : PolynomialGrowth f) :
    ∃ B : ℝ, 0 ≤ B ∧ ∀ t, |f t * Real.exp (-(t ^ 2 / 2))| ≤ B := by
  obtain ⟨C, hC, n, hn⟩ := hg
  refine ⟨C * n.factorial * Real.exp (3 / 2), by positivity, fun t => ?_⟩
  have hp := (div_le_iff₀ (by positivity : (0 : ℝ) < n.factorial)).mp
    (Real.pow_div_factorial_le_exp (1 + |t|) (by positivity) n)
  have he : 1 + |t| + -(t ^ 2 / 2) ≤ (3 : ℝ) / 2 := by
    have hsq := sq_nonneg (|t| - 1)
    nlinarith only [hsq, sq_abs t]
  simp only [abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    |f t| * Real.exp (-(t ^ 2 / 2)) ≤
        (C * (1 + |t|) ^ n) * Real.exp (-(t ^ 2 / 2)) := by
      exact mul_le_mul_of_nonneg_right (by simpa only [Real.norm_eq_abs] using hn t)
        (Real.exp_nonneg _)
    _ ≤ (C * (Real.exp (1 + |t|) * n.factorial)) * Real.exp (-(t ^ 2 / 2)) := by
      gcongr
    _ = C * n.factorial * Real.exp (1 + |t| + -(t ^ 2 / 2)) := by
      rw [Real.exp_add (1 + |t|) (-(t ^ 2 / 2))]
      ring
    _ ≤ C * n.factorial * Real.exp (3 / 2) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr he) (by positivity)

end Gaussian

end Cubature
