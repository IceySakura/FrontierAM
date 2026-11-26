import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Measure.WithDensity
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Tactic

/-!
# Model and observation table for ellipsoid membership cubature

The oracle occurs only in `raySearch` and in the one observation of the
positive height at the interior point. `ObservationTable` is the result of
running the searches once each, not an instruction to re-query upon lookup.
-/

open scoped BigOperators ENNReal
open MeasureTheory Set

namespace Cubature

noncomputable section

abbrev Coordinates (d : ℕ) := Fin d → ℝ
abbrev Coefficients (d : ℕ) := Matrix (Fin d) (Fin d) ℝ

def quadratic {d : ℕ} (A : Coefficients d) (x : Coordinates d) : ℝ :=
  ∑ i, ∑ j, A i j * x i * x j

def linear {d : ℕ} (b x : Coordinates d) : ℝ := ∑ i, b i * x i

def ellipsoid {d : ℕ} (a : Coordinates d) (A : Coefficients d) (b : Coordinates d) :
    Set (Coordinates d) :=
  {z | quadratic A (z - a) + linear b (z - a) ≤ 1}

/-- The two bodies in the comparison use the same exact Gaussian density. -/
noncomputable def gaussianDensity {d : ℕ} (x : Coordinates d) : ℝ≥0∞ :=
  ENNReal.ofReal ((Real.sqrt (2 * Real.pi))⁻¹ ^ d *
    Real.exp (-(∑ i, (x i)^2) / 2))

noncomputable def gaussian {d : ℕ} : Measure (Coordinates d) :=
  (volume : Measure (Coordinates d)).withDensity gaussianDensity

def inBox {d : ℕ} (R : ℝ) (x : Coordinates d) : Prop := ∀ i, |x i| ≤ R

/-- Equality with the boundary is deliberately not constrained. -/
def JumpOracle {d : ℕ} (f : Coordinates d → ℝ) (J : ℝ) (a : Coordinates d)
    (A : Coefficients d) (b : Coordinates d) : Prop :=
  ∀ x, (quadratic A x + linear b x < 1 → f (a+x) = J) ∧
       (1 < quadratic A x + linear b x → f (a+x) = 0)

def coordinate {d : ℕ} (i : Fin d) : Coordinates d := fun j => if j = i then 1 else 0

def chord {d : ℕ} (i j : Fin d) : Coordinates d :=
  if i = j then coordinate i else coordinate i + coordinate j

def orientedChord {d : ℕ} (i j : Fin d) (forward : Bool) : Coordinates d :=
  if forward then chord i j else -chord i j

structure Bracket (d : ℕ) where
  lo : ℝ
  hi : ℝ
  points : List (Coordinates d)

/-- Each real-valued query is made at the midpoint, and contributes one point
to the trace, irrespective of the answer (including an arbitrary boundary value). -/
noncomputable def raySearch {d : ℕ} (f : Coordinates d → ℝ) (a u : Coordinates d) :
    ℕ → ℝ → ℝ → Bracket d
  | 0, lo, hi => ⟨lo, hi, []⟩
  | n+1, lo, hi =>
    let mid := (lo+hi)/2
    let z := a + mid • u
    let tail := if 0 < f z then raySearch f a u n mid hi else raySearch f a u n lo mid
    ⟨tail.lo, tail.hi, z :: tail.points⟩

abbrev ObservationTable (d : ℕ) := (Fin d × Fin d × Bool) → Bracket d

noncomputable def observe {d : ℕ} (f : Coordinates d → ℝ) (a : Coordinates d)
    (ρ R : ℝ) (m : ℕ) : ObservationTable d :=
  fun s => raySearch f a (orientedChord s.1 s.2.1 s.2.2) m ρ R

def queryTrace {d : ℕ} (a : Coordinates d) (table : ObservationTable d) :
    List (Coordinates d) :=
  a :: (Finset.univ.toList : List (Fin d × Fin d × Bool)).flatMap (fun s => (table s).points)

noncomputable def reciprocal {d : ℕ} (r : Bracket d) : ℝ := ((r.lo+r.hi)/2)⁻¹

noncomputable def observedQuadratic {d : ℕ} (table : ObservationTable d) (i j : Fin d) : ℝ :=
  reciprocal (table (i,j,true)) * reciprocal (table (i,j,false))

noncomputable def fittedMatrix {d : ℕ} (table : ObservationTable d) : Coefficients d :=
  fun i j => if i = j then observedQuadratic table i i
    else (observedQuadratic table i j - observedQuadratic table i i -
      observedQuadratic table j j) / 2

noncomputable def fittedLinear {d : ℕ} (table : ObservationTable d) : Coordinates d :=
  fun i => reciprocal (table (i,i,true)) - reciprocal (table (i,i,false))

def certifiedSet {d : ℕ} (a : Coordinates d) (R δ : ℝ) (table : ObservationTable d) :
    Set (Coordinates d) :=
  {z | inBox R (z-a) ∧
    quadratic (fittedMatrix table) (z-a) + linear (fittedLinear table) (z-a) ≤ 1-δ}

def bracketWidth (ρ R : ℝ) (m : ℕ) : ℝ := (R-ρ) / (2:ℝ)^m

def polynomialError (d : ℕ) (ρ R : ℝ) (m : ℕ) : ℝ :=
  bracketWidth ρ R m * (3*(d:ℝ)^2*R^2/ρ^3 + 2*(d:ℝ)*R/ρ^2)

def contract {d : ℕ} (a : Coordinates d) (t : ℝ) (z : Coordinates d) : Coordinates d :=
  t • z + (1-t) • a

noncomputable def massFactor {d : ℕ} (a : Coordinates d) (t : ℝ) : ℝ :=
  t^d * Real.exp (-((1-t) / (2*(1+t)) * ∑ i, (a i)^2))

end

end Cubature
