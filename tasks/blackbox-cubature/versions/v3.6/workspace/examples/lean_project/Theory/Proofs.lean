import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral

/-!
# Proofs

Every statement of `Theory/Statements.lean`, with the same name and type, and a complete proof (no `sorry`, no new
axioms). Helper lemmas and other submission modules may be used; `Theory.Statements` may not be imported.
-/

namespace Cubature

theorem gaussian_integral_example : ∫ x : ℝ, Real.exp (-x ^ 2) = Real.sqrt Real.pi := by
  simpa using integral_gaussian 1

end Cubature
