import Theory.Model
import Theory.ValueTree
import Theory.RankCapModel
import Theory.RankAdaptiveModel
import Theory.CylindricalModel

/-! Kernel comparison interface for the results in THEORY.md. -/

open MeasureTheory

namespace Cubature

/-- R1. A deterministic exponential-in-precision adaptive upper bound and an
inverse-query lower bound for arbitrary deterministic nonadaptive designs. -/
theorem ellipsoid_query_separation (d : ℕ) (hd : 0 < d) (ρ : ℝ)
    (hρ : 0 < ρ) (hρ' : ρ ≤ 20) :
    0 < massGap d ρ ∧
    (∀ m, errorRadius d m < 1 →
      ∃ T : QueryTree (Point d) ℝ,
        T.depth ≤ d*(d+1)*m ∧ T.admissible allowed ∧
        ∀ E : Ellipsoid d,
          (1-errorRadius d m)^d * probability (region ρ E) ≤ T.run (answerBit (region ρ E)) ∧
          T.run (answerBit (region ρ E)) ≤ (1+errorRadius d m)^d * probability (region ρ E)) ∧
    (∀ (n : ℕ) (nodes : Fin n → Point d) (out : (Fin n → Bool) → ℝ),
      ∃ E : Ellipsoid d, massGap d ρ / (2*(n+1 : ℝ)) ≤
        |out (fun j ↦ answerBit (region ρ E) (nodes j))-probability (region ρ E)|) := by
  sorry

/-- R1, randomized part. The adversarial input is independent of the seed. -/
theorem ellipsoid_nonadaptive_mean_square (d : ℕ) (hd : 0 < d) (ρ : ℝ) (hρ : 0 < ρ)
    (n : ℕ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (nodes : Ω → Fin n → Point d) (out : Ω → (Fin n → Bool) → ℝ)
    (hI : ∀ E : Ellipsoid d, Integrable (fun ω ↦ squaredError ρ E (nodes ω) (out ω)) μ) :
    ∃ E : Ellipsoid d, (massGap d ρ/(n+1 : ℝ))^2/128 ≤
      ∫ ω, squaredError ρ E (nodes ω) (out ω) ∂μ := by
  sorry

/-- R2. Real-valued oracle with an unknown odd background. Each tree step
costs two calls, and the target is the full integral of the oracle payoff. -/
theorem odd_background_query_separation (d : ℕ) (hd : 0 < d) (ρ : ℝ)
    (hρ : 0 < ρ) (hρ' : ρ ≤ 20) :
    (∀ m, oddErrorRadius d m < 1 →
      ∃ T : QueryTree (Bool × Point d) ℝ,
        2*T.depth ≤ 2*d*(d+1)*m ∧ T.admissible pairAllowed ∧
        ∀ (E : Ellipsoid d) (h : Point d → ℝ), (∀ x, h (-x) = -h x) →
          Integrable h (gaussian d) →
          let I := ∫ x, augmentedPayoff ρ E h x ∂(gaussian d)
          (1-oddErrorRadius d m)^d * I ≤ T.run (pairedBit (augmentedPayoff ρ E h)) ∧
          T.run (pairedBit (augmentedPayoff ρ E h)) ≤ (1+oddErrorRadius d m)^d * I) ∧
    (∀ (n : ℕ) (nodes : Fin n → Point d) (out : (Fin n → ℝ) → ℝ),
      ∃ (E : Ellipsoid d) (h : Point d → ℝ),
        (∀ x, h (-x) = -h x) ∧ Integrable h (gaussian d) ∧
        massGap d ρ / (2*(n+1 : ℝ)) ≤
          |out (fun j ↦ augmentedPayoff ρ E h (nodes j)) -
            (∫ z, augmentedPayoff ρ E h z ∂(gaussian d))|) := by
  sorry

/-- R3. Strip an unknown ellipsoidal jump and run an arbitrary bounded
background cubature tree on its exact transcript. -/
theorem closed_jump_cubature_reduction (d : ℕ) (ρ L : ℝ)
    (hρ : 0 < ρ) (hρ' : ρ ≤ 20) (hL : 0 ≤ L)
    (k m n : ℕ) (hk : 0 < k) (hscan : 64*L*ρ ≤ k)
    (hε : errorRadius d m < 1) (hdecode : 16*L*ρ*errorRadius d m ≤ 1)
    (backgroundRule : ValueTree (Point d) ℝ)
    (hn : backgroundRule.within n) (hlegal : backgroundRule.admissible allowed) :
    ∃ T : ValueTree (Point d) (ℝ × ℝ),
      T.within (d*(d+1)*(k+1+m)+2*n) ∧ T.admissible allowed ∧
      ∀ (E : Ellipsoid d) (h : Point d → ℝ), CoordinateLipschitz L h →
        let result := T.run (augmentedPayoff ρ E h)
        result.2 = backgroundRule.run h ∧
        (1-errorRadius d m)^d*probability (region ρ E) ≤ result.1 ∧
        result.1 ≤ (1+errorRadius d m)^d*probability (region ρ E) := by
  sorry

/-- R3, second-query frequency: under the standard Gaussian, the ambiguous
points are no more frequent than the indicated ellipsoidal shell. -/
theorem decoding_shell_probability (d : ℕ) (ρ : ℝ) (E : Ellipsoid d)
    (A : Coefficients d) (b : Point d) (δ : ℝ) (hδ : 0 ≤ δ) (hδ' : 2*δ < 1)
    (hfit : ∀ x, normSq x ≤ 3 → |polynomial A b x - polynomial E.A E.b x| ≤ δ) :
    probability {x : Point d | normSq (ρ⁻¹ • x) ≤ 3 ∧
        (1-δ ≤ polynomial A b (ρ⁻¹ • x) ∧ polynomial A b (ρ⁻¹ • x) < 1+δ)} ≤
      ((1+2*δ)^d-(1-2*δ)^d)*probability (region ρ E) := by
  sorry

/-- R4. Exact finite query threshold for integrating bounded continuous
quadratic kinks. The lower bound allows adaptive real-valued queries. -/
theorem centered_kink_optimal_queries (d : ℕ) (hd : 0 < d) (ρ : ℝ)
    (hρ : 0 < ρ) (hρ' : ρ ≤ 20) :
    2 * Fintype.card (RayIndex d) = d*(d+1) ∧
    (∃ (z : RayIndex d → Point d) (out : (RayIndex d → ℝ) → ℝ),
      (∀ j, allowed (z j)) ∧
      ∀ E : Ellipsoid d, out (fun j ↦ ellipsoidalKink ρ E (z j)) = kinkIntegral ρ E) ∧
    (∃ ε : ℝ, 0 < ε ∧
      ∀ (n : ℕ), 2*n < d*(d+1) → ∀ T : ValueTree (Point d) ℝ, T.within n →
        ∃ E : Ellipsoid d, ε ≤ |T.run (ellipsoidalKink ρ E)-kinkIntegral ρ E|) := by
  sorry

/-- R5. Exact recovery and integration in linear query count for rank at
most two. The positive lower threshold is independent of dimension. -/
theorem rank_two_kink_linear_queries (ρ : ℝ) (hρ : 0<ρ) (hb : ρ≤20) :
    (∀ d, ∃ T : ValueTree (Point d) (Coefficients d × ℝ),
      T.within (3*d) ∧ T.admissible allowed ∧
      ∀ G : RankTwoCap d, T.run (rankCap ρ G) = (G.matrix,rankCapIntegral ρ G)) ∧
    (0<rankTailTolerance ρ ∧
      ∀ (d n : ℕ), n<d → ∀ T : ValueTree (Point d) ℝ, T.within n →
        ∃ G : RankTwoCap d, rankTailTolerance ρ ≤ |T.run (rankCap ρ G)-rankCapIntegral ρ G|) := by
  sorry

/-- R6. The algorithm is not given a rank bound. The real-valued query path
on a true payoff is bounded in terms of the unknown rank of its Gram matrix. -/
theorem rank_adaptive_exact_cap_cubature (d : ℕ) (ρ : ℝ) (hρ : 0<ρ) (hb : ρ≤20) :
    ∃ T : ValueTree (Point d) (Coefficients d × ℝ),
      T.within (d*(d+1)) ∧ T.admissible allowed ∧
      ∀ (k : ℕ) (G : GramCap d k),
        T.run (gramCapPayoff ρ G) = (gramRows G.rows,gramCapIntegral ρ G) ∧
        (T.trace (gramCapPayoff ρ G)).length ≤
          d*(Module.finrank ℝ (Submodule.span ℝ (Set.range G.rows))+1) := by
  sorry

/-- R7. A stable value-only rule for low-rank centered jumps, given a single
known curved direction but no active subspace or lower bound on the second
eigenvalue. The absolute Gaussian tail term is from surrogate truncation. -/
theorem rank_two_jump_query_bound (d : ℕ) (hd : 0<d) (ρ : ℝ)
    (hρ : 0<ρ) (hρ' : ρ≤16) (w : Point d) (hw : ∀ i, |w i|≤1)
    (m : ℕ) (R : ℝ) (hnoise : cylinderNoise m≤1/8) (hδ : cylinderDilationError d m R<1) :
    ∃ T : QueryTree (Point d) (Coefficients d × ℝ),
      T.depth≤(3*d+1)*m ∧ T.admissible allowed ∧
      ∀ G : RankTwoCap d, 1≤quadratic G.matrix w → quadratic G.matrix w≤2 →
        let p := T.run (answerBit (cylinderRegion ρ G))
        (∀ i j, |p.1 i j-G.matrix i j|≤cylinderMatrixError m) ∧
        (1-cylinderDilationError d m R)^d *
          (probability (cylinderRegion ρ G)-probability (cylinderTail (d := d) ρ R)) ≤ p.2 ∧
        p.2 ≤ (1+cylinderDilationError d m R)^d*probability (cylinderRegion ρ G) := by
  sorry

end Cubature
