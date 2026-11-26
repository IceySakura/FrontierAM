import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral

/-!
# Statement layer

The formal statements of the results claimed in `THEORY.md`, each with the proof `sorry`. Reviewers read this file
(and the submission modules it imports, such as shared definitions) to decide whether each statement says what
`THEORY.md` claims. `Theory/Proofs.lean` repeats every statement verbatim with its proof and must not import this
file. The entry below only shows the layout; replace it.
-/

namespace Cubature

/-- Example: the one-dimensional Gaussian integral. -/
theorem gaussian_integral_example : ∫ x : ℝ, Real.exp (-x ^ 2) = Real.sqrt Real.pi := by
  sorry

end Cubature
