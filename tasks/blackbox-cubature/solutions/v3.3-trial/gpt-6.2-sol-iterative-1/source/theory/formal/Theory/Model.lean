import Mathlib.MeasureTheory.Integral.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Tactic

/-! The class, the Gaussian target, and the information model used in R1. -/

open MeasureTheory Set
open scoped BigOperators ENNReal

namespace Cubature

abbrev Point (d : ℕ) := Fin d → ℝ
abbrev Coefficients (d : ℕ) := Fin d → Fin d → ℝ

def normSq {d : ℕ} (x : Point d) : ℝ :=
  ∑ i, x i ^ 2

def quadratic {d : ℕ} (A : Coefficients d) (x : Point d) : ℝ :=
  ∑ i, ∑ j, A i j * x i * x j

def linear {d : ℕ} (b x : Point d) : ℝ :=
  ∑ i, b i * x i

def polynomial {d : ℕ} (A : Coefficients d) (b x : Point d) : ℝ :=
  quadratic A x + linear b x

/-- The fixed, normalized model: the spectrum lies in [1,2], and the
linear part satisfies an energy bound (in particular this holds if ‖b‖₂ ≤ 1). -/
structure Ellipsoid (d : ℕ) where
  A : Coefficients d
  b : Point d
  symm : ∀ i j, A i j = A j i
  low : ∀ x, normSq x ≤ quadratic A x
  high : ∀ x, quadratic A x ≤ 2 * normSq x
  bias : ∀ x, |linear b x| ≤ (1 + normSq x) / 2

noncomputable def region {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) : Set (Point d) :=
  {x | polynomial E.A E.b (ρ⁻¹ • x) < 1}

noncomputable def answerBit {d : ℕ} (s : Set (Point d)) (x : Point d) : Bool :=
  @decide (x ∈ s) (Classical.propDecidable _)

/-- Product standard normal density in the supplied coordinates. -/
noncomputable def density {d : ℕ} (x : Point d) : ℝ :=
  ∏ i, ProbabilityTheory.gaussianPDFReal 0 1 (x i)

noncomputable def gaussian (d : ℕ) : Measure (Point d) :=
  (volume : Measure (Point d)).withDensity fun x ↦ ENNReal.ofReal (density x)

noncomputable def probability {d : ℕ} (s : Set (Point d)) : ℝ :=
  (gaussian d).real s

/-- A real-payoff representative of the pure jump class (zero on the boundary). -/
noncomputable def payoff {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (x : Point d) : ℝ :=
  if answerBit (region ρ E) x then 1 else 0

/-- Deterministic, value-only binary decision trees; there is no access to any
parameters of the queried ellipsoid. -/
inductive QueryTree (X : Type*) (Y : Type*) where
  | ret (result : Y)
  | ask (point : X) (outside inside : QueryTree X Y)

namespace QueryTree

def run {X Y : Type*} : QueryTree X Y → (X → Bool) → Y
  | .ret y, _ => y
  | .ask x l r, f => if f x then r.run f else l.run f

def depth {X Y : Type*} : QueryTree X Y → ℕ
  | .ret _ => 0
  | .ask _ l r => 1 + max l.depth r.depth

/-- Every query on every possible answer path is allowed. -/
def admissible {X Y : Type*} (P : X → Prop) : QueryTree X Y → Prop
  | .ret _ => True
  | .ask x l r => P x ∧ l.admissible P ∧ r.admissible P

def bind {X Y Z : Type*} : QueryTree X Y → (Y → QueryTree X Z) → QueryTree X Z
  | .ret y, f => f y
  | .ask x l r, f => .ask x (l.bind f) (r.bind f)

def map {X Y Z : Type*} (t : QueryTree X Y) (f : Y → Z) : QueryTree X Z :=
  t.bind fun y => .ret (f y)

end QueryTree

def allowed {d : ℕ} (x : Point d) : Prop :=
  ∀ i, |x i| ≤ 40

/-- One ray direction per upper-triangular matrix entry. -/
abbrev RayIndex (d : ℕ) := {ij : Fin d × Fin d // ij.1 ≤ ij.2}

def axis {d : ℕ} (i : Fin d) : Point d := fun j ↦ if j = i then 1 else 0

def rayIndex {d : ℕ} (i j : Fin d) : RayIndex d :=
  ⟨(min i j, max i j), min_le_max⟩

def ray {d : ℕ} (r : RayIndex d) : Point d :=
  if r.1.1 = r.1.2 then axis r.1.1 else axis r.1.1 + axis r.1.2

noncomputable def arithmeticBall {d : ℕ} (r : ℝ) : Set (Point d) :=
  {x | normSq x < r ^ 2}

noncomputable def ballMass (d : ℕ) (r : ℝ) : ℝ :=
  probability (arithmeticBall (d := d) r)

noncomputable def massGap (d : ℕ) (ρ : ℝ) : ℝ :=
  ballMass d ρ-ballMass d (3*ρ/4)

noncomputable def errorRadius (d m : ℕ) : ℝ :=
  40 * d * (2 / (2 : ℝ) ^ m)

noncomputable def squaredError {d n : ℕ} (ρ : ℝ) (E : Ellipsoid d)
    (nodes : Fin n → Point d) (output : (Fin n → Bool) → ℝ) : ℝ :=
  (output (fun j ↦ answerBit (region ρ E) (nodes j))-probability (region ρ E))^2

/-- Unit jump with an arbitrary odd, integrable background. -/
noncomputable def augmentedPayoff {d : ℕ} (ρ : ℝ) (E : Ellipsoid d)
    (background : Point d → ℝ) (x : Point d) : ℝ :=
  background x + payoff ρ E x

/-- Each such test costs exactly two real oracle calls, at `x` and `-x`.
The tag asks whether both, or at least one, of the two locations are inside. -/
noncomputable def pairedBit {d : ℕ} (f : Point d → ℝ) (p : Bool × Point d) : Bool :=
  @decide ((if p.1 then (3/2 : ℝ) else 1/2) < f p.2 + f (-p.2))
    (Classical.propDecidable _)

def pairAllowed {d : ℕ} (p : Bool × Point d) : Prop :=
  allowed p.2 ∧ allowed (-p.2)

noncomputable def oddErrorRadius (d m : ℕ) : ℝ :=
  64 * d * (2 / (2 : ℝ)^m)

/-- The centered inward quadratic kink; only the quadratic coefficient of E
is used. -/
noncomputable def ellipsoidalKink {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) (z : Point d) : ℝ :=
  max (1-quadratic E.A (ρ⁻¹ • z)) 0

noncomputable def kinkIntegral {d : ℕ} (ρ : ℝ) (E : Ellipsoid d) : ℝ :=
  ∫ z, ellipsoidalKink ρ E z ∂(gaussian d)

end Cubature
