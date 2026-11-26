import Theory.LowerModel

open Set

namespace Cubature

variable {d n : ℕ}

def bodyAt (r2 : ℝ) (A : Coefficients d) : Set (Coordinates d) :=
  {x | quadratic A x ≤ r2}

noncomputable def memberAt (r2 : ℝ) (A : Coefficients d) (x : Coordinates d) : Bool :=
  decide (quadratic A x ≤ r2)

/-- The answer string, indexed by time. The query coordinates themselves
can depend on all previous bits. -/
def bitTranscript {n : ℕ} (f : Coordinates d → Bool) : PointQueryPlan d Bool n → (Fin n → Bool)
  | .answer _ => Fin.elim0
  | .ask z next => Fin.cons (f z) (bitTranscript f (next (f z)))

noncomputable def midpointMatrix (A B : Coefficients d) : Coefficients d := (1/2 : ℝ) • (A+B)

noncomputable def matrixGridPitch (d m : ℕ) : ℝ := 1 / (4*(d:ℝ)*(m:ℝ))

noncomputable def precisionGap (d m : ℕ) : ℝ :=
  matrixGridPitch d m / (8*(d:ℝ)^3)

end Cubature
