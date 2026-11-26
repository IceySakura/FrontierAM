import Theory.RankCapModel

open Set MeasureTheory

namespace Cubature

/-- An unknown centered quadratic jump, possibly a cylinder when the rank
is less than the ambient dimension. The boundary takes value zero. -/
noncomputable def cylinderRegion {d : ℕ} (ρ : ℝ) (G : RankTwoCap d) : Set (Point d) :=
  {z | quadratic G.matrix (ρ⁻¹ • z) < 1}

noncomputable def cylinderNoise (m : ℕ) : ℝ := 16/(2:ℝ)^m

noncomputable def cylinderMatrixError (m : ℕ) : ℝ := 8000*cylinderNoise m

noncomputable def cylinderDilationError (d m : ℕ) (T : ℝ) : ℝ :=
  cylinderMatrixError m * d * T^2

noncomputable def cylinderTail {d : ℕ} (ρ T : ℝ) : Set (Point d) :=
  {z | T^2 < normSq (ρ⁻¹ • z)}

end Cubature
