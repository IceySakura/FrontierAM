import Theory.EllipsoidBounds
import Theory.KinkBounds
import Theory.HiddenBallsBounds

open MeasureTheory

namespace Cubature

/-- R1: relative Gaussian mass error for an offset ellipsoid from bounded point queries. -/
theorem translated_ellipsoid_cubature (d : ℕ) (f : Point d → ℝ) (a : Point d)
    (A : Fin d → Fin d → ℝ) (b : Point d) (l h R : ℝ) (n : ℕ)
    (hl : 0 < l) (hlh : l ≤ h) (hR : 0 ≤ R)
    (hsym : ∀ i j, A i j = A j i) (hA : ∀ x, 0 ≤ bilinear A x x)
    (hf : oracleCut f a A b)
    (hRad : ∀ i j s, l ≤ reciprocalRadius A b (signedDirection i j s) ∧
      reciprocalRadius A b (signedDirection i j s) ≤ h)
    (hcap : ∀ i, |a i| + l⁻¹ ≤ 40)
    (hcover : ∀ x ∈ ellipsoid a A b, l1norm (x-a) ≤ R) :
    let δ := (h-l)/(2 : ℝ)^(n+1)
    let t := 1+(3*h*δ*R^2 + 2*δ*R)
    let target := (gamma d).real (ellipsoid a A b)
    (ellipsoidQueries f a l h n).length = 2*d*d*n ∧
    (∀ x ∈ ellipsoidQueries f a l h n, ∀ i, |x i| ≤ 40) ∧
    |ellipsoidEstimate f a l h R n-target| ≤ (dilationFactor a t-1)*target := by
  let δ := (h-l)/(2 : ℝ)^(n+1)
  let t := 1+(3*h*δ*R^2 + 2*δ*R)
  have hδ : 0 ≤ δ := div_nonneg (sub_nonneg.mpr hlh) (by positivity)
  have h1 : 0 ≤ 3*h*δ := mul_nonneg (by linarith) hδ
  have h2 : 0 ≤ 2*δ := mul_nonneg (by norm_num) hδ
  have ht : 1 ≤ t := by
    have H := add_nonneg (mul_nonneg h1 (sq_nonneg R)) (mul_nonneg h2 hR)
    dsimp only [t]
    linarith only [H]
  obtain ⟨hq,hbox⟩ := recover_query_bound f a A b hf hA hl n hRad hcap
  refine ⟨hq,hbox,?_⟩
  obtain ⟨hmat,hlin⟩ := recover_coefficients f a A b hf hA hsym hl hlh n hRad
  obtain ⟨U,L⟩ := finite_precision_sandwich a A (approxMatrix f a l h n) b
    (approxLinear f a l h n) hA hR h1 h2 hmat hlin hcover
  have mF : MeasurableSet (approxSet f a l h R n) :=
    (measurable_l1ball a R).inter (measurable_ellipsoid a _ _)
  exact two_sided_mass_bound a ht (measurable_ellipsoid a A b) mF U L

/-- R2: localize a kink using only five value comparisons per round, with noise. -/
theorem noisy_internal_kink (f g noise : ℝ → ℝ) (w J B e : ℝ)
    (hf : ∀ x, f x = g x + J*max 0 (x-w) + noise x)
    (hbg : ∀ x s, 0 ≤ s → |secondDiff g x s| ≤ B*s^2)
    (hne : ∀ x, |noise x| ≤ e) (hJ : J ≠ 0)
    (n : ℕ) (c h : ℝ) (hh : 0 < h) (hw : |w-c| ≤ 3*h/2)
    (hrob : ∀ j < n, let s := h/(2 : ℝ)^j; 8*(B*s^2+4*e) ≤ |J| * s) :
    let z := kinkSearch f c h n
    z.2.length = 5*n ∧ |w-z.1| ≤ 3*(h/(2 : ℝ)^n)/2 ∧
      ∀ x ∈ z.2, c-2*h ≤ x ∧ x ≤ c+2*h := by
  exact kink_search_bound f g noise w J B e hf hbg hne hJ n c h hh hw hrob

/-- R3: even adaptive deterministic queries need exponentially many points
to guarantee relative error below one third for a hidden ball with unknown height. -/
theorem hidden_ball_relative_lower (d n : ℕ) (hn : n < 2^d)
    (query : List ℝ → Point d) (decode : List ℝ → ℝ) :
    ∃ (s : Signs d) (a : ℝ), (a = 1 ∨ a = 2) ∧
      let f := hiddenBallPayoff s a
      let integral := ∫ x, f x ∂(gamma d)
      0 < integral ∧ integral/3 ≤ |decode (transcript query f n)-integral| := by
  obtain ⟨s,hs⟩ := avoid_signBall query n hn
  let p := (gamma d).real (signBall s)
  have ppos : 0 < p := signBall_mass_pos s
  rcases one_third_either (p := p) (decode (List.replicate n 0)) with h1 | h2
  · refine ⟨s,1,Or.inl rfl, ?_⟩
    simp only [hiddenBall_mass, one_mul, transcript_hidden_ball query s 1 n hs]
    exact ⟨ppos,h1⟩
  · refine ⟨s,2,Or.inr rfl, ?_⟩
    simp only [hiddenBall_mass, transcript_hidden_ball query s 2 n hs]
    exact ⟨mul_pos (by norm_num) ppos,h2⟩

/-- R3, finite-seed version: one fixed input is bad for at least
`(2^d-n)/(2*2^d)` of the seeds, when `n ≤ 2^d`. -/
theorem hidden_ball_randomized_relative_lower (d n m : ℕ) (hm : 0 < m)
    (query : Fin m → List ℝ → Point d) (decode : Fin m → List ℝ → ℝ) :
    ∃ input : Signs d × Bool, m*(2^d-n) ≤ (2*2^d)*badSeedCount n query decode input := by
  exact hidden_ball_bad_seed_count d n m hm query decode

end Cubature
