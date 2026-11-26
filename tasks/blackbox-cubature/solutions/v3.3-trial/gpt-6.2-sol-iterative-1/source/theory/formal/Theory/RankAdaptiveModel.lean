import Theory.RankCapModel
import Theory.KinkGeometry
import Mathlib.Analysis.InnerProductSpace.PiL2

open MeasureTheory
open scoped RealInnerProductSpace

namespace Cubature

def gramRows {d : ℕ} {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v : Fin d → E) : Coefficients d := fun i j => ⟪v i,v j⟫

/-- The factor dimension certifies a rank bound but is not given to the
cubature tree. The rows and their Gram matrix are unknown to the tree. -/
structure GramCap (d k : ℕ) where
  rows : Fin d → EuclideanSpace ℝ (Fin k)
  upper : ∀ x, quadratic (gramRows rows) x ≤ 2 * normSq x

noncomputable def gramCapPayoff {d k : ℕ} (ρ : ℝ) (G : GramCap d k) : Point d → ℝ :=
  matrixCap ρ (gramRows G.rows)

noncomputable def gramCapIntegral {d k : ℕ} (ρ : ℝ) (G : GramCap d k) : ℝ :=
  ∫ x, gramCapPayoff ρ G x ∂(gaussian d)

end Cubature
