import Theory.EvenRecovery
import Theory.Randomized
import Theory.LipReduction
import Theory.KinkLower
import Theory.RankCapLower
import Theory.RankAdaptive
import Theory.CylindricalMass

open MeasureTheory

namespace Cubature

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
  refine ⟨massGap_pos hd hρ, ?_, deterministic_nonadaptive_lower hd hρ⟩
  intro m hm
  exact ⟨ellipsoidTree d ρ m, ellipsoidTree_depth d ρ m,
    ellipsoidTree_allowed d hρ.le hρ' m, ellipsoidTree_correct hρ m hm⟩

theorem ellipsoid_nonadaptive_mean_square (d : ℕ) (hd : 0 < d) (ρ : ℝ) (hρ : 0 < ρ)
    (n : ℕ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (nodes : Ω → Fin n → Point d) (out : Ω → (Fin n → Bool) → ℝ)
    (hI : ∀ E : Ellipsoid d, Integrable (fun ω ↦ squaredError ρ E (nodes ω) (out ω)) μ) :
    ∃ E : Ellipsoid d, (massGap d ρ/(n+1 : ℝ))^2/128 ≤
      ∫ ω, squaredError ρ E (nodes ω) (out ω) ∂μ := by
  exact randomized_nonadaptive_lower hd hρ n μ nodes out hI

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
  constructor
  · intro m hm
    refine ⟨antipodalTree hd ρ m, ?_, antipodalTree_allowed hd hρ.le hρ' m, ?_⟩
    · have H := Nat.mul_le_mul_left 2 (antipodalTree_depth hd ρ m)
      simpa [Nat.mul_assoc] using H
    · intro E h ho hi
      simp only [integral_augmentedPayoff ρ E h ho hi]
      exact antipodalTree_correct hd hρ m hm E h ho
  · intro n nodes out
    let B := fun f : Fin n → Bool => out (fun j => if f j then (1 : ℝ) else 0)
    obtain ⟨E,hE⟩ := deterministic_nonadaptive_lower hd hρ n nodes B
    have ho : ∀ _x : Point d, (0 : ℝ) = -0 := fun _ => neg_zero.symm
    have hi : Integrable (fun _ : Point d => (0 : ℝ)) (gaussian d) := integrable_const 0
    refine ⟨E,fun _ => 0,ho,hi,?_⟩
    rw [integral_augmentedPayoff ρ E (fun _ => 0) ho hi]
    simpa [B,augmentedPayoff,payoff] using hE

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
  exact ⟨liftCubature hk ρ m backgroundRule, liftCubature_within hk ρ m backgroundRule hn,
    liftCubature_allowed hk hρ.le hρ' m backgroundRule hlegal,
    liftCubature_correct hk hρ hL hscan m hε hdecode backgroundRule⟩

theorem decoding_shell_probability (d : ℕ) (ρ : ℝ) (E : Ellipsoid d)
    (A : Coefficients d) (b : Point d) (δ : ℝ) (hδ : 0 ≤ δ) (hδ' : 2*δ < 1)
    (hfit : ∀ x, normSq x ≤ 3 → |polynomial A b x - polynomial E.A E.b x| ≤ δ) :
    probability {x : Point d | normSq (ρ⁻¹ • x) ≤ 3 ∧
        (1-δ ≤ polynomial A b (ρ⁻¹ • x) ∧ polynomial A b (ρ⁻¹ • x) < 1+δ)} ≤
      ((1+2*δ)^d-(1-2*δ)^d)*probability (region ρ E) := by
  exact ambiguous_mass ρ E A b hδ hδ' hfit

theorem centered_kink_optimal_queries (d : ℕ) (hd : 0 < d) (ρ : ℝ)
    (hρ : 0 < ρ) (hρ' : ρ ≤ 20) :
    2 * Fintype.card (RayIndex d) = d*(d+1) ∧
    (∃ (z : RayIndex d → Point d) (out : (RayIndex d → ℝ) → ℝ),
      (∀ j, allowed (z j)) ∧
      ∀ E : Ellipsoid d, out (fun j ↦ ellipsoidalKink ρ E (z j)) = kinkIntegral ρ E) ∧
    (∃ ε : ℝ, 0 < ε ∧
      ∀ (n : ℕ), 2*n < d*(d+1) → ∀ T : ValueTree (Point d) ℝ, T.within n →
        ∃ E : Ellipsoid d, ε ≤ |T.run (ellipsoidalKink ρ E)-kinkIntegral ρ E|) := by
  exact ⟨card_rayIndex d,exact_kink_nonadaptive hρ hρ',uniform_kink_lower d hd ρ hρ⟩

theorem rank_two_kink_linear_queries (ρ : ℝ) (hρ : 0<ρ) (hb : ρ≤20) :
    (∀ d, ∃ T : ValueTree (Point d) (Coefficients d × ℝ),
      T.within (3*d) ∧ T.admissible allowed ∧
      ∀ G : RankTwoCap d, T.run (rankCap ρ G) = (G.matrix,rankCapIntegral ρ G)) ∧
    (0<rankTailTolerance ρ ∧
      ∀ (d n : ℕ), n<d → ∀ T : ValueTree (Point d) ℝ, T.within n →
        ∃ G : RankTwoCap d, rankTailTolerance ρ ≤ |T.run (rankCap ρ G)-rankCapIntegral ρ G|) := by
  exact ⟨fun d => rank_two_cap_upper d hρ hb,rank_two_cap_lower ρ hρ⟩

theorem rank_adaptive_exact_cap_cubature (d : ℕ) (ρ : ℝ) (hρ : 0<ρ) (hb : ρ≤20) :
    ∃ T : ValueTree (Point d) (Coefficients d × ℝ),
      T.within (d*(d+1)) ∧ T.admissible allowed ∧
      ∀ (k : ℕ) (G : GramCap d k),
        T.run (gramCapPayoff ρ G) = (gramRows G.rows,gramCapIntegral ρ G) ∧
        (T.trace (gramCapPayoff ρ G)).length ≤
          d*(Module.finrank ℝ (Submodule.span ℝ (Set.range G.rows))+1) := by
  exact ⟨rankAdaptiveCubature d ρ,rankAdaptive_within d ρ,rankAdaptive_allowed d hρ.le hb,
    fun _ G => rankAdaptive_exact hρ G⟩

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
  exact ⟨cylinderCubature hd ρ w m R,cylinderCubature_budget hd ρ w m R,
    cylinderCubature_allowed hd hρ.le hρ' w hw m R,cylinderCubature_bounds hd hρ w m R hnoise hδ⟩

end Cubature
