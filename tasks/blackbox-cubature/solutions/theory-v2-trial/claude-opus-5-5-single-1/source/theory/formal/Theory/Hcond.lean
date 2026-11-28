import Theory.Main
import Theory.Kink
import Theory.Weak

/-!
# GKLS's conditions (h1)–(h2) for the counterexample

GKLS (Theorems 2–3) require that every function
`h(y) = D^η θ(ψ(y), y) ∏ᵢ D^{γ⁽ⁱ⁾} φ(ψ(y), y) / (D_j φ(ψ(y), y))^b · ρ^{(c)}(ψ(y))`
be `p`-integrable against the product weights. For the counterexample (with `θ ≡ 1`) every such `h`
is continuous and bounded, hence integrable against any integrable product weight.
-/

open Real Set Filter Topology MeasureTheory ProbabilityTheory Polynomial
open scoped ENNReal

namespace GKLS

variable {n : ℕ}

/-! ### Derivatives of the Gaussian density are bounded -/

lemma abs_eval_le (q : ℝ[X]) (x : ℝ) :
    |q.eval x| ≤ (∑ i ∈ Finset.range (q.natDegree + 1), |q.coeff i|) * (1 + x ^ 2) ^ q.natDegree := by
  rw [eval_eq_sum_range, Finset.sum_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i hi => ?_)
  rw [abs_mul, abs_pow]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  have hi' : i ≤ q.natDegree := by simpa [Nat.lt_succ_iff] using hi
  have h1 : |x| ≤ 1 + x ^ 2 := by nlinarith [abs_nonneg x, sq_abs x, sq_nonneg (|x| - 1)]
  have h2 : (1 : ℝ) ≤ 1 + x ^ 2 := by nlinarith [sq_nonneg x]
  calc |x| ^ i ≤ (1 + x ^ 2) ^ i := pow_le_pow_left₀ (abs_nonneg x) h1 i
    _ ≤ (1 + x ^ 2) ^ q.natDegree := pow_le_pow_right₀ h2 hi'

lemma one_add_sq_pow_mul_exp_le (N : ℕ) (x : ℝ) :
    (1 + x ^ 2) ^ N * rexp (-(x ^ 2 / 2)) ≤ 2 ^ N * (1 + 2 ^ N * N.factorial) := by
  set u := x ^ 2 / 2 with hu
  have hu0 : 0 ≤ u := by positivity
  have hexp1 : 1 ≤ rexp u := Real.one_le_exp hu0
  have hfac : u ^ N / N.factorial ≤ rexp u := Real.pow_div_factorial_le_exp u hu0 N
  have hfac' : u ^ N ≤ N.factorial * rexp u := by
    have hpos : (0 : ℝ) < N.factorial := by exact_mod_cast Nat.factorial_pos N
    rw [div_le_iff₀ hpos] at hfac; linarith
  -- `1 + x² ≤ 2 max(1, x²)` and `max(1, x²)^N ≤ 1 + (x²)^N = 1 + 2^N u^N`
  have hmax : (1 + x ^ 2) ^ N ≤ 2 ^ N * (1 + 2 ^ N * u ^ N) := by
    have h1 : 1 + x ^ 2 ≤ 2 * max 1 (x ^ 2) := by
      have := le_max_left 1 (x ^ 2); have := le_max_right 1 (x ^ 2); linarith
    have h2 : (max 1 (x ^ 2)) ^ N ≤ 1 + (x ^ 2) ^ N := by
      rcases le_total 1 (x ^ 2) with h | h
      · rw [max_eq_right h]; linarith [pow_nonneg (sq_nonneg x) N]
      · rw [max_eq_left h, one_pow]; linarith [pow_nonneg (sq_nonneg x) N]
    have h3 : (x ^ 2) ^ N = 2 ^ N * u ^ N := by rw [hu, ← mul_pow]; ring_nf
    calc (1 + x ^ 2) ^ N ≤ (2 * max 1 (x ^ 2)) ^ N := pow_le_pow_left₀ (by positivity) h1 N
      _ = 2 ^ N * (max 1 (x ^ 2)) ^ N := mul_pow _ _ _
      _ ≤ 2 ^ N * (1 + (x ^ 2) ^ N) := mul_le_mul_of_nonneg_left h2 (by positivity)
      _ = 2 ^ N * (1 + 2 ^ N * u ^ N) := by rw [h3]
  have hexpneg : rexp (-(x ^ 2 / 2)) = (rexp u)⁻¹ := by rw [hu, Real.exp_neg]
  rw [hexpneg]
  have hEpos : 0 < rexp u := Real.exp_pos u
  rw [← div_eq_mul_inv, div_le_iff₀ hEpos]
  calc (1 + x ^ 2) ^ N ≤ 2 ^ N * (1 + 2 ^ N * u ^ N) := hmax
    _ ≤ 2 ^ N * (rexp u + 2 ^ N * (N.factorial * rexp u)) := by
        gcongr
    _ = 2 ^ N * (1 + 2 ^ N * N.factorial) * rexp u := by ring

lemma rho_eq_fun : rho = fun x => (√(2 * π))⁻¹ * rexp (-(x ^ 2 / 2)) := by
  funext x
  simp only [rho, gaussianPDFReal, NNReal.coe_one, mul_one, sub_zero]
  congr 2
  ring

/-- Every derivative of the standard normal density is bounded. -/
lemma iteratedDeriv_rho_bounded (c : ℕ) : ∃ C, ∀ x, |iteratedDeriv c rho x| ≤ C := by
  set q : ℝ[X] := (hermite c).map (Int.castRingHom ℝ) with hq
  refine ⟨(√(2 * π))⁻¹ * ((∑ i ∈ Finset.range (q.natDegree + 1), |q.coeff i|) *
    (2 ^ q.natDegree * (1 + 2 ^ q.natDegree * q.natDegree.factorial))), fun x => ?_⟩
  have h1 : iteratedDeriv c rho x = (√(2 * π))⁻¹ * iteratedDeriv c (fun y => rexp (-(y ^ 2 / 2))) x := by
    rw [rho_eq_fun]; exact iteratedDeriv_const_mul_field _ _
  have h2 : iteratedDeriv c (fun y => rexp (-(y ^ 2 / 2))) x =
      (-1 : ℝ) ^ c * q.eval x * rexp (-(x ^ 2 / 2)) := by
    rw [iteratedDeriv_eq_iterate, deriv_gaussian_eq_hermite_mul_gaussian, hq, eval_map,
      aeval_def, algebraMap_int_eq]
  rw [h1, h2, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < (√(2 * π))⁻¹)]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  rw [abs_mul, abs_mul, abs_pow, abs_neg, abs_one, one_pow, one_mul,
    abs_of_pos (Real.exp_pos _)]
  calc |q.eval x| * rexp (-(x ^ 2 / 2))
      ≤ (∑ i ∈ Finset.range (q.natDegree + 1), |q.coeff i|) * (1 + x ^ 2) ^ q.natDegree *
          rexp (-(x ^ 2 / 2)) := mul_le_mul_of_nonneg_right (abs_eval_le q x) (Real.exp_pos _).le
    _ = (∑ i ∈ Finset.range (q.natDegree + 1), |q.coeff i|) *
          ((1 + x ^ 2) ^ q.natDegree * rexp (-(x ^ 2 / 2))) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_left (one_add_sq_pow_mul_exp_le _ x)
          (Finset.sum_nonneg fun i _ => abs_nonneg _)

/-! ### Derivatives of a constant -/

lemma iterPartial_const (c : ℝ) :
    ∀ l : List (Fin n), iterPartial l (fun _ => c) = fun _ => if l = [] then c else 0 := by
  intro l
  induction l with
  | nil => funext x; simp [iterPartial]
  | cons k l ih =>
    have : iterPartial (k :: l) (fun _ => c) = partialDeriv k (iterPartial l (fun _ => c)) := rfl
    rw [this, ih]
    funext x
    simp [partialDeriv]

lemma iterPartialExists_const (c : ℝ) : ∀ l : List (Fin n), IterPartialExists l (fun _ => c) := by
  intro l
  induction l with
  | nil => trivial
  | cons k l ih =>
    refine ⟨ih, fun x => ?_⟩
    rw [iterPartial_const]
    exact differentiableAt_const _

lemma contDiffMixed_const (r : Fin n → ℕ) (c : ℝ) : ContDiffMixed r (fun _ => c) := by
  intro l _
  refine ⟨iterPartialExists_const c l, ?_⟩
  rw [iterPartial_const]
  exact continuous_const

/-! ### A bound for products over lists -/

lemma abs_list_prod_le (l : List ℝ) (B : ℝ) (hB : 0 ≤ B) (h : ∀ x ∈ l, |x| ≤ B) :
    |l.prod| ≤ B ^ l.length := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.prod_cons, abs_mul, List.length_cons, pow_succ']
    exact mul_le_mul (h a (by simp)) (ih fun x hx => h x (by simp [hx])) (abs_nonneg _) hB

/-! ### The conditions (h) for the counterexample -/

/-- The root of `φ(·, y) = 0` is unique: it is `ψ_k(-∑ y_i)`. -/
lemma root_eq (k : ℕ) (j : Fin (n + 1)) {ψ : (Fin n → ℝ) → ℝ}
    (hψ : ∀ y, phiCE k j (Fin.insertNth j (ψ y) y) = 0) : ψ = fun y => psi k (-∑ i, y i) := by
  funext y
  have h := hψ y
  rw [phiCE_insertNth] at h
  have : gfun k (ψ y) = -∑ i, y i := by linarith
  rw [← this, psi_gfun]

/-- GKLS's condition (h2) for the counterexample, for any `θ` whose relevant derivative
`D^η θ(ψ(y), y)` is continuous and bounded. -/
lemma phiCE_hcond_gen (k : ℕ) (j : Fin (n + 1)) (ψ : (Fin n → ℝ) → ℝ)
    (hψ : ∀ y, phiCE k j (Fin.insertNth j (ψ y) y) = 0)
    (θ : (Fin (n + 1) → ℝ) → ℝ) (η : List (Fin (n + 1))) (B : ℝ)
    (hθc : Continuous fun y => iterPartial η θ (Fin.insertNth j (ψ y) y))
    (hθb : ∀ y, |iterPartial η θ (Fin.insertNth j (ψ y) y)| ≤ B)
    (γs : List (List (Fin (n + 1)))) (b c : ℕ)
    (hγ : ∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ k + 1)
    (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ) (hp : 0 ≤ p) (hζ : ∀ i, Integrable (ζ i)) :
    Integrable (fun y => |iterPartial η θ (Fin.insertNth j (ψ y) y) *
        (γs.map fun γ => iterPartial γ (phiCE k j) (Fin.insertNth j (ψ y) y)).prod /
        partialDeriv j (phiCE k j) (Fin.insertNth j (ψ y) y) ^ b *
        iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i)) := by
  have hψeq := root_eq k j hψ
  have hψc : Continuous ψ := by
    rw [hψeq]
    exact (contDiff_psi k).continuous.comp (continuous_finsetSum _ fun i _ => continuous_apply i).neg
  set X : (Fin n → ℝ) → (Fin (n + 1) → ℝ) := fun y => Fin.insertNth j (ψ y) y with hX
  have hXc : Continuous X := Continuous.finInsertNth j hψc continuous_id
  set F1 : (Fin n → ℝ) → ℝ := fun y => iterPartial η θ (X y) with hF1
  set F2 : (Fin n → ℝ) → ℝ := fun y => (γs.map fun γ => iterPartial γ (phiCE k j) (X y)).prod
    with hF2
  set F3 : (Fin n → ℝ) → ℝ := fun y => partialDeriv j (phiCE k j) (X y) with hF3
  set F4 : (Fin n → ℝ) → ℝ := fun y => iteratedDeriv c (gaussianPDFReal 0 1) (ψ y) with hF4
  have hF1c : Continuous F1 := hθc
  have hF1b : ∀ y, |F1 y| ≤ B := hθb
  have hF2c : Continuous F2 := by
    rw [hF2]
    apply continuous_list_prod
    intro γ hγm
    exact (phiCE_smooth k j γ (hγ γ hγm).2).2.comp hXc
  have hF2b : ∀ y, |F2 y| ≤ (3 / 2) ^ γs.length := by
    intro y
    rw [hF2]
    have := abs_list_prod_le (γs.map fun γ => iterPartial γ (phiCE k j) (X y)) (3 / 2) (by norm_num)
      (by
        intro x hx
        obtain ⟨γ, hγm, rfl⟩ := List.mem_map.mp hx
        exact phiCE_bound_nonempty k j γ (hγ γ hγm).1 (hγ γ hγm).2 (X y))
    simpa using this
  have hF3c : Continuous F3 := by
    have : partialDeriv j (phiCE k j) = iterPartial [j] (phiCE k j) := rfl
    rw [hF3, this]
    exact (phiCE_smooth k j [j] (by simp)).2.comp hXc
  have hF3lb : ∀ y, 1 / 2 ≤ F3 y := by
    intro y
    show 1 / 2 ≤ partialDeriv j (phiCE k j) (X y)
    rw [deriv_j_phiCE]; exact half_le_deriv_gfun k _
  have hF4c : Continuous F4 := by
    rw [hF4]
    exact (contDiff_rho.continuous_iteratedDeriv c (by exact_mod_cast le_top)).comp hψc
  obtain ⟨C4, hC4⟩ := iteratedDeriv_rho_bounded c
  have hF4b : ∀ y, |F4 y| ≤ C4 := fun y => hC4 (ψ y)
  set h : (Fin n → ℝ) → ℝ := fun y => F1 y * F2 y / F3 y ^ b * F4 y with hh
  have hF3ne : ∀ y, F3 y ^ b ≠ 0 := fun y => pow_ne_zero _ (by linarith [hF3lb y])
  have hhc : Continuous h := by
    rw [hh]
    exact ((hF1c.mul hF2c).div (hF3c.pow b) hF3ne).mul hF4c
  have hhb : ∀ y, |h y| ≤ B * (3 / 2) ^ γs.length * 2 ^ b * C4 := by
    intro y
    have hB : 0 ≤ B := (abs_nonneg _).trans (hF1b y)
    have hC4' : 0 ≤ C4 := (abs_nonneg _).trans (hF4b y)
    have h3 : |1 / F3 y ^ b| ≤ 2 ^ b := by
      have hpos : 0 < F3 y := by linarith [hF3lb y]
      rw [abs_of_pos (by positivity), one_div, ← inv_pow]
      apply pow_le_pow_left₀ (by positivity)
      rw [inv_le_comm₀ hpos (by norm_num)]
      linarith [hF3lb y]
    calc |h y| = |F1 y| * |F2 y| * |1 / F3 y ^ b| * |F4 y| := by
          rw [hh]; simp only [abs_mul, abs_div, one_div, abs_inv]; ring
      _ ≤ B * (3 / 2) ^ γs.length * 2 ^ b * C4 := by
          gcongr
          · exact hF1b y
          · exact hF2b y
          · exact hF4b y
  have hprod : Integrable (fun y : Fin n → ℝ => ∏ i, ζ (α i) (y i)) := by
    rw [volume_pi]
    exact Integrable.fintype_prod (fun i => hζ (α i))
  refine hprod.bdd_mul (c := (B * (3 / 2) ^ γs.length * 2 ^ b * C4) ^ p) ?_
    (ae_of_all _ fun y => ?_)
  · exact ((hhc.abs).measurable.pow_const p).aestronglyMeasurable
  · rw [Real.norm_eq_abs, abs_of_nonneg (Real.rpow_nonneg (abs_nonneg _) p)]
    exact Real.rpow_le_rpow (abs_nonneg _) (hhb y) hp

/-- (h2) for `θ ≡ 1` (jumps, R1). -/
lemma phiCE_hcond_one (k : ℕ) (j : Fin (n + 1)) (ψ : (Fin n → ℝ) → ℝ)
    (hψ : ∀ y, phiCE k j (Fin.insertNth j (ψ y) y) = 0)
    (η : List (Fin (n + 1))) (γs : List (List (Fin (n + 1)))) (b c : ℕ)
    (hγ : ∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ k + 1)
    (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ) (hp : 0 ≤ p) (hζ : ∀ i, Integrable (ζ i)) :
    Integrable (fun y => |iterPartial η (fun _ => (1 : ℝ)) (Fin.insertNth j (ψ y) y) *
        (γs.map fun γ => iterPartial γ (phiCE k j) (Fin.insertNth j (ψ y) y)).prod /
        partialDeriv j (phiCE k j) (Fin.insertNth j (ψ y) y) ^ b *
        iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i)) := by
  refine phiCE_hcond_gen k j ψ hψ (fun _ => (1 : ℝ)) η 1 ?_ ?_ γs b c hγ p ζ α hp hζ
  · simp only [iterPartial_const]; exact continuous_const
  · intro y; simp only [iterPartial_const]; split_ifs <;> simp

/-- (h2) for `θ = φ` (kinks, R2). -/
lemma phiCE_hcond_kink (k : ℕ) (j : Fin (n + 1)) (ψ : (Fin n → ℝ) → ℝ)
    (hψ : ∀ y, phiCE k j (Fin.insertNth j (ψ y) y) = 0)
    (η : List (Fin (n + 1))) (hη : η.count j ≤ k + 1) (γs : List (List (Fin (n + 1)))) (b c : ℕ)
    (hγ : ∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ k + 1)
    (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ) (hp : 0 ≤ p) (hζ : ∀ i, Integrable (ζ i)) :
    Integrable (fun y => |iterPartial η (phiCE k j) (Fin.insertNth j (ψ y) y) *
        (γs.map fun γ => iterPartial γ (phiCE k j) (Fin.insertNth j (ψ y) y)).prod /
        partialDeriv j (phiCE k j) (Fin.insertNth j (ψ y) y) ^ b *
        iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i)) := by
  have hψc : Continuous ψ := by
    rw [root_eq k j hψ]
    exact (contDiff_psi k).continuous.comp (continuous_finsetSum _ fun i _ => continuous_apply i).neg
  refine phiCE_hcond_gen k j ψ hψ (phiCE k j) η (3 / 2) ?_ ?_ γs b c hγ p ζ α hp hζ
  · exact (phiCE_smooth k j η hη).2.comp (Continuous.finInsertNth j hψc continuous_id)
  · intro y
    by_cases hne : η = []
    · subst hne
      have : iterPartial [] (phiCE k j) (Fin.insertNth j (ψ y) y) = 0 := hψ y
      rw [this]; norm_num
    · exact phiCE_bound_nonempty k j η hne hη _

/-- The counterexample's classical mixed derivatives are its weak derivatives. -/
lemma phiCE_iterPartialCont (k : ℕ) (j : Fin (n + 1)) :
    ∀ l : List (Fin (n + 1)), l.count j ≤ k + 1 → IterPartialCont l (phiCE k j) := by
  intro l
  induction l with
  | nil => intro _; exact (phiCE_smooth k j [] (by simp)).2
  | cons a l ih =>
    intro hl
    exact ⟨ih (le_trans List.count_le_count_cons hl), (phiCE_smooth k j (a :: l) hl).2⟩

lemma phiCE_weak (k : ℕ) (j : Fin (n + 1)) (l : List (Fin (n + 1))) (hl : l.count j ≤ k + 1)
    (v : (Fin (n + 1) → ℝ) → ℝ) (hv : ContDiff ℝ (⊤ : ℕ∞) v) (hvs : HasCompactSupport v) :
    ∫ x, phiCE k j x * iterPartial l.reverse v x =
      (-1) ^ l.length * ∫ x, iterPartial l (phiCE k j) x * v x :=
  integral_iterPartial_eq (phiCE k j) l (phiCE_smooth k j l hl).1 (phiCE_iterPartialCont k j l hl)
    v hv hvs

/-- Assembly of R1 including GKLS's condition (h2). -/
theorem main_counterexample_h (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j < ∑ k : Fin n, r (j.succAbove k)) :
    ∃ φ : (Fin (n + 1) → ℝ) → ℝ,
      ContDiffMixed r (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] →
        iterPartial l (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) = fun _ => 0) ∧
      (∀ p : ℝ≥0∞, MemLp (fun _ : Fin (n + 1) → ℝ => (1 : ℝ)) p
        (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      ContDiffMixed r φ ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        IterPartialExists l φ ∧ Continuous (iterPartial l φ)) ∧
      (∀ x, 1 / 2 ≤ partialDeriv j φ x) ∧
      (∀ y : Fin n → ℝ, Tendsto (fun t => φ (Fin.insertNth j t y)) atTop atTop) ∧
      (∀ y : Fin n → ℝ, ∃ t, φ (Fin.insertNth j t y) = 0) ∧
      (∀ x, |φ x| ≤ 1 + ∑ i, |x i|) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j → ∀ x, |iterPartial l φ x| ≤ 3 / 2) ∧
      (∀ p : ℝ≥0∞, p ≠ ∞ → ∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        MemLp (iterPartial l φ) p (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      (∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
        ∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j →
          Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))) ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        ∀ v : (Fin (n + 1) → ℝ) → ℝ, ContDiff ℝ (⊤ : ℕ∞) v → HasCompactSupport v →
          ∫ x, φ x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l φ x * v x) ∧
      (∀ ψ : (Fin n → ℝ) → ℝ, (∀ y, φ (Fin.insertNth j (ψ y) y) = 0) →
        ∀ (η : List (Fin (n + 1))) (γs : List (List (Fin (n + 1)))) (b c : ℕ),
          (∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ r j) →
          ∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
            Integrable (fun y => |iterPartial η (fun _ => (1 : ℝ)) (Fin.insertNth j (ψ y) y) *
                (γs.map fun γ => iterPartial γ φ (Fin.insertNth j (ψ y) y)).prod /
                partialDeriv j φ (Fin.insertNth j (ψ y) y) ^ b *
                iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i))) ∧
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => ind (φ x))) ∧
      (∀ l : List (Fin n), l.length ≤ r j →
        IterPartialExists l (preintegrate j (fun x => ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => ind (φ x))))) ∧
      (∀ l : List (Fin n), r j < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => ind (φ x)))) := by
  obtain ⟨k, hk⟩ : ∃ k, r j = k + 1 := ⟨r j - 1, by omega⟩
  have hP := preintegrate_phiCE (n := n) k j
  have hHC : ContDiff ℝ (k + 1) (Hfun k) := contDiff_Hfun k
  have hbad := not_differentiableAt_iteratedDeriv_Hfun k
  have hnot : ∀ l : List (Fin n), r j < l.length →
      ¬ IterPartialExists l (preintegrate j (fun x => ind (phiCE k j x))) := by
    intro l hl
    rw [hP]
    exact ridge_not_exists (N := k + 1) (by exact_mod_cast hHC) (by exact_mod_cast hbad) l (hk ▸ hl)
  refine ⟨phiCE k j, contDiffMixed_const r 1,
    fun l hl => by rw [iterPartial_const]; simp [hl], fun p => memLp_const 1,
    phiCE_mixed k j r hk, fun l hl => phiCE_smooth k j l (hk ▸ hl), ?_, ?_, ?_,
    phiCE_bound k j, fun l hne hl x => phiCE_bound_nonempty k j l hne (hk ▸ hl) x,
    fun p hp l hl => phiCE_memLp k j p hp l (hk ▸ hl),
    fun p ζ hp hζ l hne hl => phiCE_weighted k j p hp ζ hζ l hne (hk ▸ hl),
    fun l hl v hv hvs => phiCE_weak k j l (hk ▸ hl) v hv hvs,
    fun ψ hψ η γs b c hγ p ζ α hp hζ => phiCE_hcond_one k j ψ hψ η γs b c
      (fun γ hγm => ⟨(hγ γ hγm).1, hk ▸ (hγ γ hγm).2⟩) p ζ α hp hζ, ?_, ?_, hnot⟩
  · intro x
    rw [deriv_j_phiCE]
    exact half_le_deriv_gfun k (x j)
  · intro y
    have h : ∀ t, t + ∑ i, y i ≤ phiCE k j (Fin.insertNth j t y) := by
      intro t; rw [phiCE_insertNth]; have := le_gfun k t; linarith
    exact tendsto_atTop_mono h (tendsto_atTop_add_const_right _ _ tendsto_id)
  · intro y
    refine ⟨psi k (-∑ i, y i), ?_⟩
    rw [phiCE_insertNth, gfun_psi]; ring
  · intro hmix
    obtain ⟨L, hLc, hLl⟩ := exists_list_count (fun i => r (j.succAbove i))
    exact hnot L (by rw [hLl]; exact hsum) (hmix L (fun i => (hLc i).le)).1
  · intro l hl
    rw [hP]
    obtain ⟨hex, heq⟩ := ridge_iterPartial (N := k + 1) (by exact_mod_cast hHC) l (hk ▸ hl)
    refine ⟨hex, ?_⟩
    rw [heq]
    have hc : Continuous (iteratedDeriv l.length (Hfun k)) :=
      hHC.continuous_iteratedDeriv _ (by exact_mod_cast (hk ▸ hl))
    fun_prop

/-- Assembly of R2 including GKLS's condition (h2) for `θ = φ`. -/
theorem kink_counterexample_h (n : ℕ) (j : Fin (n + 1)) (r : Fin (n + 1) → ℕ)
    (hrj : 1 ≤ r j) (hsum : r j + 1 < ∑ k : Fin n, r (j.succAbove k)) :
    ∃ φ : (Fin (n + 1) → ℝ) → ℝ,
      ContDiffMixed r φ ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        IterPartialExists l φ ∧ Continuous (iterPartial l φ)) ∧
      (∀ x, 1 / 2 ≤ partialDeriv j φ x) ∧
      (∀ y : Fin n → ℝ, Tendsto (fun t => φ (Fin.insertNth j t y)) atTop atTop) ∧
      (∀ y : Fin n → ℝ, ∃ t, φ (Fin.insertNth j t y) = 0) ∧
      (∀ x, |φ x| ≤ 1 + ∑ i, |x i|) ∧
      (∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j → ∀ x, |iterPartial l φ x| ≤ 3 / 2) ∧
      (∀ p : ℝ≥0∞, p ≠ ∞ → ∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        MemLp (iterPartial l φ) p (Measure.pi fun _ : Fin (n + 1) => gaussianReal 0 1)) ∧
      (∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
        ∀ l : List (Fin (n + 1)), l ≠ [] → l.count j ≤ r j →
          Integrable (fun x => |iterPartial l φ x| ^ p * ∏ i, ζ (l.count i) (x i))) ∧
      (∀ l : List (Fin (n + 1)), l.count j ≤ r j →
        ∀ v : (Fin (n + 1) → ℝ) → ℝ, ContDiff ℝ (⊤ : ℕ∞) v → HasCompactSupport v →
          ∫ x, φ x * iterPartial l.reverse v x = (-1) ^ l.length * ∫ x, iterPartial l φ x * v x) ∧
      (∀ ψ : (Fin n → ℝ) → ℝ, (∀ y, φ (Fin.insertNth j (ψ y) y) = 0) →
        ∀ (η : List (Fin (n + 1))) (γs : List (List (Fin (n + 1)))) (b c : ℕ),
          η.count j ≤ r j → (∀ γ ∈ γs, γ ≠ [] ∧ γ.count j ≤ r j) →
          ∀ (p : ℝ) (ζ : ℕ → ℝ → ℝ) (α : Fin n → ℕ), 0 ≤ p → (∀ i, Integrable (ζ i)) →
            Integrable (fun y => |iterPartial η φ (Fin.insertNth j (ψ y) y) *
                (γs.map fun γ => iterPartial γ φ (Fin.insertNth j (ψ y) y)).prod /
                partialDeriv j φ (Fin.insertNth j (ψ y) y) ^ b *
                iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)| ^ p * ∏ i, ζ (α i) (y i))) ∧
      ¬ ContDiffMixed (fun k => r (j.succAbove k)) (preintegrate j (fun x => φ x * ind (φ x))) ∧
      (∀ l : List (Fin n), l.length ≤ r j + 1 →
        IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x))) ∧
        Continuous (iterPartial l (preintegrate j (fun x => φ x * ind (φ x))))) ∧
      (∀ l : List (Fin n), r j + 1 < l.length →
        ¬ IterPartialExists l (preintegrate j (fun x => φ x * ind (φ x)))) := by
  obtain ⟨k, hk⟩ : ∃ k, r j = k + 1 := ⟨r j - 1, by omega⟩
  have hP := preintegrate_kink (n := n) k j
  have hKC : ContDiff ℝ ((k + 2 : ℕ) : WithTop ℕ∞) (Kfun k) := contDiff_Kfun k
  have hbad := not_differentiableAt_iteratedDeriv_Kfun k
  have hnot : ∀ l : List (Fin n), r j + 1 < l.length →
      ¬ IterPartialExists l (preintegrate j (fun x => phiCE k j x * ind (phiCE k j x))) := by
    intro l hl
    rw [hP]
    exact ridge_not_exists (N := k + 2) hKC hbad l (by omega)
  refine ⟨phiCE k j, phiCE_mixed k j r hk, fun l hl => phiCE_smooth k j l (hk ▸ hl), ?_, ?_, ?_,
    phiCE_bound k j, fun l hne hl x => phiCE_bound_nonempty k j l hne (hk ▸ hl) x,
    fun p hp l hl => phiCE_memLp k j p hp l (hk ▸ hl),
    fun p ζ hp hζ l hne hl => phiCE_weighted k j p hp ζ hζ l hne (hk ▸ hl),
    fun l hl v hv hvs => phiCE_weak k j l (hk ▸ hl) v hv hvs,
    fun ψ hψ η γs b c hη hγ p ζ α hp hζ => phiCE_hcond_kink k j ψ hψ η (hk ▸ hη) γs b c
      (fun γ hγm => ⟨(hγ γ hγm).1, hk ▸ (hγ γ hγm).2⟩) p ζ α hp hζ, ?_, ?_, hnot⟩
  · intro x
    rw [deriv_j_phiCE]
    exact half_le_deriv_gfun k (x j)
  · intro y
    have h : ∀ t, t + ∑ i, y i ≤ phiCE k j (Fin.insertNth j t y) := by
      intro t; rw [phiCE_insertNth]; have := le_gfun k t; linarith
    exact tendsto_atTop_mono h (tendsto_atTop_add_const_right _ _ tendsto_id)
  · intro y
    refine ⟨psi k (-∑ i, y i), ?_⟩
    rw [phiCE_insertNth, gfun_psi]; ring
  · intro hmix
    obtain ⟨L, hLc, hLl⟩ := exists_list_count (fun i => r (j.succAbove i))
    exact hnot L (by rw [hLl]; exact hsum) (hmix L (fun i => (hLc i).le)).1
  · intro l hl
    rw [hP]
    obtain ⟨hex, heq⟩ := ridge_iterPartial (N := k + 2) hKC l (by omega)
    refine ⟨hex, ?_⟩
    rw [heq]
    have hc : Continuous (iteratedDeriv l.length (Kfun k)) :=
      hKC.continuous_iteratedDeriv _ (by exact_mod_cast (show l.length ≤ k + 2 by omega))
    fun_prop

end GKLS
