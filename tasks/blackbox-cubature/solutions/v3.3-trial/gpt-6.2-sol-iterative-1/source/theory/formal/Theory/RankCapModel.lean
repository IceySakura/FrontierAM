import Theory.ValueTree

open MeasureTheory Set
open scoped BigOperators

namespace Cubature

/-- A positive semidefinite matrix of rank at most two, with unknown factors.
The upper quadratic-form bound is equivalent to the spectral bound by 2. -/
structure RankTwoCap (d : ℕ) where
  u : Point d
  v : Point d
  upper : ∀ x, (linear u x)^2+(linear v x)^2 ≤ 2*normSq x

def RankTwoCap.matrix {d : ℕ} (G : RankTwoCap d) : Coefficients d :=
  fun i j => G.u i*G.u j+G.v i*G.v j

noncomputable def matrixCap {d : ℕ} (ρ : ℝ) (A : Coefficients d) (x : Point d) : ℝ :=
  max (1-quadratic A (ρ⁻¹ • x)) 0

noncomputable def rankCap {d : ℕ} (ρ : ℝ) (G : RankTwoCap d) : Point d → ℝ :=
  matrixCap ρ G.matrix

noncomputable def rankCapIntegral {d : ℕ} (ρ : ℝ) (G : RankTwoCap d) : ℝ :=
  ∫ x, rankCap ρ G x ∂(gaussian d)

/-- Half the one-sided standard normal tail beyond the known scale. -/
noncomputable def rankTailTolerance (ρ : ℝ) : ℝ :=
  (ProbabilityTheory.gaussianReal 0 1).real (Ioi ρ) / 2

end Cubature
