import Theory.WeakSetup
import Mathlib.MeasureTheory.Function.LocallyIntegrable

/-! Statements of R1 and R2. `Theory.Proofs` repeats these types and does not import this file. -/

open Filter MeasureTheory ProbabilityTheory Set
open scoped ContDiff

namespace Cubature

/-- R1: lowering the phase order in the integrated coordinate destroys mixed smoothness,
even for a constant-height Gaussian jump and a uniformly transverse root in every fiber. -/
theorem gaussianMixedOrder_sharp (m q : ℕ) (r : Fin m → ℕ)
    (hq : 1 ≤ q) (hr : q < ∑ i, r i) :
    let φ : (Option (Fin m) → ℝ) → ℝ := sharpPhase q
    let θ : (Option (Fin m) → ℝ) → ℝ := fun _ => 1
    let P := gaussianPreintegral φ θ
    ContDiff ℝ (q : ℕ) φ ∧
    (∀ r' : Fin m → ℕ, GaussianMixedInput (fun i : Option (Fin m) => i.elim q r') φ) ∧
    (∀ s : Option (Fin m) → ℕ, GaussianMixedInput s θ) ∧
    (∀ x, 1 ≤ partialsAlong [none] φ x) ∧
    (∀ y, Tendsto (fun t => φ (fiberPoint t y)) atTop atTop) ∧
    (∃ ψ : (Fin m → ℝ) → ℝ, Continuous ψ ∧
      (∀ t y, φ (fiberPoint t y) = 0 ↔ t = ψ y) ∧
      ∀ (w : List (Option (Fin m))) (ws : List (List (Option (Fin m))))
        (_hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ q) (b c : ℕ),
        Continuous (boundaryFactor φ θ ψ w ws b c) ∧
        ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor φ θ ψ w ws b c y| ≤ C) ∧
    ContDiff ℝ (q : ℕ) P ∧
    (∀ w : List (Fin m), w.length = q + 1 →
      ¬ ∃ g, HasPartialsAlong w P g ∧ Continuous g) ∧
    ¬ MixedContinuous r P := by
  sorry

/-- R2: with two first-order retained coordinates, the loss can be a failure
of the Gaussian mixed Sobolev condition itself, not just of classical continuity. -/
theorem gaussianMixedSecond_weakL2_sharp :
    ∃ A : ℝ → ℝ,
      (∀ u, A (inverseSplice u) = u ∧ inverseSplice (A u) = u) ∧
      let φ : (Option (Fin 2) → ℝ) → ℝ := phaseFromAxis A
      let θ : (Option (Fin 2) → ℝ) → ℝ := fun _ => 1
      let P := fun y : ℝ × ℝ => gaussianPreintegral φ θ (pairCoordinates y)
      ContDiff ℝ 1 φ ∧
      (∀ r' : Fin 2 → ℕ, GaussianMixedInput (fun i : Option (Fin 2) => i.elim 1 r') φ) ∧
      (∀ s : Option (Fin 2) → ℕ, GaussianMixedInput s θ) ∧
      (∀ x, 1 / 2 ≤ partialsAlong [none] φ x ∧ partialsAlong [none] φ x ≤ 1) ∧
      (∀ y, Tendsto (fun t => φ (fiberPoint t y)) atTop atTop) ∧
      (∃ ψ : (Fin 2 → ℝ) → ℝ, Continuous ψ ∧
        (∀ t y, φ (fiberPoint t y) = 0 ↔ t = ψ y) ∧
        ∀ (w : List (Option (Fin 2))) (ws : List (List (Option (Fin 2))))
          (_hw : ∀ u ∈ ws, u ≠ [] ∧ u.count none ≤ 1) (b c : ℕ),
          Continuous (boundaryFactor φ θ ψ w ws b c) ∧
          ∃ C : ℝ, 0 ≤ C ∧ ∀ y, |boundaryFactor φ θ ψ w ws b c y| ≤ C) ∧
      ContDiff ℝ 1 P ∧
      ∀ G : ℝ × ℝ → ℝ, LocallyIntegrable G volume → WeakMixedSecond P G →
        (¬ IntegrableOn (fun y => G y ^ 2) (Icc (-1) 1 ×ˢ Icc (-1) 1) volume) ∧
        (¬ MemLp G 2 ((gaussianReal 0 1).prod (gaussianReal 0 1))) := by
  sorry

end Cubature
