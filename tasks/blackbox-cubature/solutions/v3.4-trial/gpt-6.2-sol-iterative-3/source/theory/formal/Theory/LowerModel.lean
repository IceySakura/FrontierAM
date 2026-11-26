import Theory.Model
import Mathlib.LinearAlgebra.Matrix.Hermitian

open scoped BigOperators ENNReal
open MeasureTheory Set

namespace Cubature

variable {d : ℕ}

abbrev UpperIndices (d : ℕ) := {p : Fin d × Fin d // p.1 ≤ p.2}

def squaredLength (x : Coordinates d) : ℝ := ∑ i, (x i)^2

/-- Radius one tenth when `A` is the identity. -/
def centeredBody (A : Coefficients d) : Set (Coordinates d) :=
  {x | quadratic A x ≤ (1:ℝ)/100}

noncomputable def membershipOracle (A : Coefficients d) (x : Coordinates d) : Bool := by
  classical
  exact decide (x ∈ centeredBody A)

/-- A fixed-depth deterministic point-query algorithm. An algorithm stopping
earlier can pad any branch with ignored asks. -/
inductive PointQueryPlan (d : ℕ) (V : Type) : ℕ → Type where
  | answer (value : ℝ) : PointQueryPlan d V 0
  | ask {n : ℕ} (point : Coordinates d) (next : V → PointQueryPlan d V n) :
    PointQueryPlan d V (n+1)

def PointQueryPlan.run {n : ℕ} {V : Type} (f : Coordinates d → V) : PointQueryPlan d V n → ℝ
  | .answer v => v
  | .ask p next => (next (f p)).run f

def PointQueryPlan.points {n : ℕ} {V : Type} (f : Coordinates d → V) :
    PointQueryPlan d V n → List (Coordinates d)
  | .answer _ => []
  | .ask p next => p :: (next (f p)).points f

def quadraticProbePoints (d : ℕ) : UpperIndices d → Coordinates d :=
  fun p => chord p.1.1 p.1.2

noncomputable def matrixFromQuadraticValues (values : UpperIndices d → ℝ) : Coefficients d :=
  fun i j => if h : i = j then values ⟨(i,i),le_rfl⟩ else
    (values ⟨(min i j,max i j), (min_le_left _ _).trans (le_max_left _ _)⟩ -
     values ⟨(i,i),le_rfl⟩ - values ⟨(j,j),le_rfl⟩) / 2

end Cubature
