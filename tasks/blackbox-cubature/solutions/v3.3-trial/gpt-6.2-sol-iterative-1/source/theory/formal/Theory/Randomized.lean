import Theory.LowerBounds

open MeasureTheory Set
open scoped BigOperators

namespace Cubature

theorem paired_mass_prior {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ) (n : ℕ) :
    ∃ E₀ E₁ : Fin (2*(n+1)) → Ellipsoid d,
      ∀ (nodes : Fin n → Point d) (out : (Fin n → Bool) → ℝ),
        4*(n+1 : ℝ)*(massGap d ρ/(n+1 : ℝ))^2/128 ≤
          ∑ i, (squaredError ρ (E₀ i) nodes out+squaredError ρ (E₁ i) nodes out) := by
  classical
  let M := 4*(n+1)
  have hM : 0 < M := by dsimp [M]; omega
  obtain ⟨r,hr,hmono,hm⟩ := massGrid hd hρ hM
  let j₀ (i : Fin (2*(n+1))) : Fin (M+1) := ⟨2*i.val, by dsimp [M]; omega⟩
  let j₁ (i : Fin (2*(n+1))) : Fin (M+1) := ⟨2*i.val+1, by dsimp [M]; omega⟩
  let E₀ (i : Fin (2*(n+1))) := ballEllipsoid d hρ (hr (j₀ i))
  let E₁ (i : Fin (2*(n+1))) := ballEllipsoid d hρ (hr (j₁ i))
  let w := massGap d ρ / (M : ℝ)
  have hreg₀ i : region ρ (E₀ i) = arithmeticBall (r (j₀ i)) := region_ballEllipsoid _ hρ _
  have hreg₁ i : region ρ (E₁ i) = arithmeticBall (r (j₁ i)) := region_ballEllipsoid _ hρ _
  have hR (j : Fin (M+1)) : 0 ≤ r j := (by positivity : 0 ≤ 3*ρ/4).trans (hr j).1
  have hR01 (i : Fin (2*(n+1))) : r (j₀ i) ≤ r (j₁ i) :=
    hmono.monotone (by change 2*i.val ≤ 2*i.val+1; omega)
  have hgap i : ballMass d (r (j₁ i))-ballMass d (r (j₀ i)) = w := by
    rw [hm,hm]
    simp only [j₀,j₁, Nat.cast_add, Nat.cast_mul, Nat.cast_one, Nat.cast_ofNat]
    unfold w
    ring
  refine ⟨E₀,E₁,?_⟩
  intro nodes out
  let cuts (i : Fin (2*(n+1))) := Ico (r (j₀ i)^2) (r (j₁ i)^2)
  have hdisj (i k : Fin (2*(n+1))) (h : i < k) :
      ∀ x : ℝ, x ∈ cuts i → x ∈ cuts k → False := by
    have hj : j₁ i ≤ j₀ k := by change 2*i.val+1 ≤ 2*k.val; omega
    have hs := pow_le_pow_left₀ (hR (j₁ i)) (hmono.monotone hj) 2
    intro x hi hk
    exact (not_lt_of_ge (hs.trans hk.1)) hi.2
  let B : Finset (Fin (2*(n+1))) := Finset.univ.filter
    (fun i ↦ ∃ j : Fin n, normSq (nodes j) ∈ cuts i)
  have hb (i : Fin (2*(n+1))) : i ∈ B ↔ ∃ j : Fin n, normSq (nodes j) ∈ cuts i := by
    simp [B]
  have hcount : B.card ≤ n := by
    have H (i : {i // i ∈ B}) : ∃ j : Fin n, normSq (nodes j) ∈ cuts i.val :=
      (hb i.val).1 i.property
    choose F hF using H
    have hinj : Function.Injective F := by
      intro i j hFij
      apply Subtype.ext
      rcases lt_trichotomy i.val j.val with h | h | h
      · exact False.elim (hdisj _ _ h _ (hF i) (hFij ▸ hF j))
      · exact h
      · exact False.elim (hdisj _ _ h _ (hF j) (hFij ▸ hF i))
    simpa using Fintype.card_le_of_injective F hinj
  have hcount' : n+1 ≤ Bᶜ.card := by
    rw [Finset.card_compl, Fintype.card_fin]
    omega
  have hlocal (i : Fin (2*(n+1))) (hi : i ∉ B) :
      w^2/2 ≤ squaredError ρ (E₀ i) nodes out+squaredError ρ (E₁ i) nodes out := by
    have hc (j : Fin n) : normSq (nodes j) ∉ cuts i := fun h ↦ hi ((hb i).2 ⟨j,h⟩)
    have he := same_bits_if_no_cut (hR (j₀ i)) (hR01 i) nodes hc
    unfold squaredError
    rw [hreg₀,hreg₁,← he]
    change w^2/2 ≤ (_-ballMass d (r (j₀ i)))^2+(_-ballMass d (r (j₁ i)))^2
    rw [← hgap i]
    exact (by
      have he' := sq_nonneg (2*out (fun j ↦ answerBit (arithmeticBall (r (j₀ i))) (nodes j))-
        ballMass d (r (j₀ i))-ballMass d (r (j₁ i)))
      nlinarith only [he'])
  have hbound (i : Fin (2*(n+1))) :
      (if i ∈ Bᶜ then w^2/2 else 0) ≤
        squaredError ρ (E₀ i) nodes out+squaredError ρ (E₁ i) nodes out := by
    by_cases h : i ∈ Bᶜ
    · rw [ite_eq_left h]
      exact hlocal i (by simpa using h)
    · rw [ite_eq_right h]
      unfold squaredError
      positivity
  have hsum := Finset.sum_le_sum (s := Finset.univ) (fun i _ ↦ hbound i)
  have hS : (∑ i : Fin (2*(n+1)), if i ∈ Bᶜ then w^2/2 else 0) =
      (Bᶜ.card : ℝ)*(w^2/2) := by
    have hf : (Finset.univ : Finset (Fin (2*(n+1)))).filter (fun i ↦ i ∈ Bᶜ) = Bᶜ := by
      ext i
      simp
    rw [← Finset.sum_filter, hf]
    simp
  rw [hS] at hsum
  have hcast : (n+1 : ℝ) ≤ Bᶜ.card := by exact_mod_cast hcount'
  have hf := mul_le_mul_of_nonneg_right hcast (show 0 ≤ w^2/2 by positivity)
  have he : 4*(n+1 : ℝ)*(massGap d ρ/(n+1 : ℝ))^2/128 = (n+1 : ℝ)*(w^2/2) := by
    dsimp [w, M]
    push_cast
    have hn : (n+1 : ℝ) ≠ 0 := by positivity
    field_simp
    <;> ring
  rw [he]
  exact hf.trans hsum

/-- Any distribution of nonadaptive designs. Integrability is automatic for
    bounded outputs, e.g. after clipping to [0,1]. -/
theorem randomized_nonadaptive_lower {d : ℕ} (hd : 0 < d) {ρ : ℝ} (hρ : 0 < ρ)
    (n : ℕ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (nodes : Ω → Fin n → Point d) (out : Ω → (Fin n → Bool) → ℝ)
    (hI : ∀ E : Ellipsoid d, Integrable (fun ω ↦ squaredError ρ E (nodes ω) (out ω)) μ) :
    ∃ E : Ellipsoid d, (massGap d ρ/(n+1 : ℝ))^2/128 ≤
      ∫ ω, squaredError ρ E (nodes ω) (out ω) ∂μ := by
  classical
  obtain ⟨E₀,E₁,hE⟩ := paired_mass_prior hd hρ n
  let C := (massGap d ρ/(n+1 : ℝ))^2/128
  let F (E : Ellipsoid d) (ω : Ω) := squaredError ρ E (nodes ω) (out ω)
  have hFin : ∀ i, Integrable (fun ω ↦ F (E₀ i) ω+F (E₁ i) ω) μ :=
    fun i ↦ (hI _).add (hI _)
  have H : (4*(n+1 : ℝ))*C ≤ ∑ i, ((∫ ω, F (E₀ i) ω ∂μ)+(∫ ω, F (E₁ i) ω ∂μ)) := by
    have hInt := integral_mono (integrable_const (4*(n+1 : ℝ)*C))
      (integrable_finsetSum _ (fun i _ => hFin i)) (fun ω ↦ by
        dsimp [C,F]
        simpa [mul_div_assoc] using hE (nodes ω) (out ω))
    have hm : μ.real univ = 1 := by simp [Measure.real]
    rw [integral_const, hm, one_smul,
      integral_finsetSum _ (fun i _ => hFin i)] at hInt
    simpa only [integral_add (hI _) (hI _), F] using hInt
  by_contra! hh
  have hLt (i : Fin (2*(n+1))) :
      (∫ ω, F (E₀ i) ω ∂μ)+(∫ ω, F (E₁ i) ω ∂μ) < 2*C := by
    have h0 := hh (E₀ i)
    have h1 := hh (E₁ i)
    dsimp [F,C] at *
    linarith
  have hne : (Finset.univ : Finset (Fin (2*(n+1)))).Nonempty :=
    ⟨⟨0,by omega⟩, Finset.mem_univ _⟩
  have hh' := Finset.sum_lt_sum_of_nonempty hne (fun i _ ↦ hLt i)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_mul, Nat.cast_ofNat, Nat.cast_add, Nat.cast_one] at hh'
  linarith

end Cubature
