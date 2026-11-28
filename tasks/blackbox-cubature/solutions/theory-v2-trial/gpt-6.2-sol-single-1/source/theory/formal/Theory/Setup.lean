import Mathlib.Analysis.Calculus.LineDeriv.Basic
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Constructions.Pi

/-! The coordinate-wise definitions used in the statement of the sharpness result. -/

open MeasureTheory ProbabilityTheory
open scoped ContDiff ENNReal

namespace Cubature

/-- `none` denotes the coordinate being integrated; `some i` is retained coordinate `i`. -/
def fiberPoint {ι : Type*} (t : ℝ) (y : ι → ℝ) : Option ι → ℝ :=
  fun i => i.elim t y

/-- The actual standard Gaussian fiber integral of a jump of height `θ` across `φ = 0`. -/
noncomputable def gaussianPreintegral {ι : Type*} (φ θ : (Option ι → ℝ) → ℝ)
    (y : ι → ℝ) : ℝ :=
  ∫ t, (if 0 < φ (fiberPoint t y) then θ (fiberPoint t y) else 0) ∂gaussianReal 0 1

/-- Product standard Gaussian, in the given coordinates. -/
noncomputable def gaussianProduct (ι : Type*) [Fintype ι] : Measure (ι → ℝ) :=
  Measure.pi (fun _ : ι => gaussianReal 0 1)

noncomputable instance {ι : Type*} [Fintype ι] : IsProbabilityMeasure (gaussianProduct ι) := by
  unfold gaussianProduct
  infer_instance

/-- Actual existence of a classical coordinate partial, not just a value of a totalized derivative. -/
def HasCoordinatePartial {ι : Type*} [DecidableEq ι] (i : ι)
    (f g : (ι → ℝ) → ℝ) : Prop :=
  ∀ x, HasLineDerivAt ℝ f (g x) x (Pi.single i 1)

/-- Interpret a list as the order in which coordinate derivatives are taken. -/
def HasPartialsAlong {ι : Type*} [DecidableEq ι] :
    List ι → ((ι → ℝ) → ℝ) → ((ι → ℝ) → ℝ) → Prop
  | [], f, g => f = g
  | i :: w, f, g => ∃ h, HasCoordinatePartial i f h ∧ HasPartialsAlong w h g

/-- Iterated coordinate derivative; its existence is certified separately by `HasPartialsAlong`. -/
noncomputable def partialsAlong {ι : Type*} [DecidableEq ι] :
    List ι → ((ι → ℝ) → ℝ) → ((ι → ℝ) → ℝ)
  | [], f => f
  | i :: w, f => partialsAlong w (fun x => lineDeriv ℝ f x (Pi.single i 1))

/-- The weak-derivative test identity on Lebesgue measure. The test derivatives are
taken in reverse order so each integration by parts transfers the next derivative. -/
def WeakPartialIdentity {ι : Type*} [Fintype ι] [DecidableEq ι] (w : List ι)
    (f g : (ι → ℝ) → ℝ) : Prop :=
  ∀ v : (ι → ℝ) → ℝ, ContDiff ℝ ∞ v → HasCompactSupport v →
    (∫ x, g x * v x) = (-1 : ℝ) ^ w.length * ∫ x, f x * partialsAlong w.reverse v x

/-- The mixed order is a bound on each individual coordinate count. -/
def WithinMixedOrder {ι : Type*} [DecidableEq ι] (r : ι → ℕ) (w : List ι) : Prop :=
  ∀ i, w.count i ≤ r i

/-- Classical mixed smoothness, with no integrability requirement. -/
def MixedContinuous {ι : Type*} [DecidableEq ι] (r : ι → ℕ) (f : (ι → ℝ) → ℝ) : Prop :=
  ∀ w, WithinMixedOrder r w → ∃ g, HasPartialsAlong w f g ∧ Continuous g

def PolynomialGrowth {E : Type*} [Norm E] (f : E → ℝ) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧ ∃ n : ℕ, ∀ x, |f x| ≤ C * (1 + ‖x‖) ^ n

/-- Stronger than the input mixed spaces at any one finite `p`: continuous classical
partials, their weak identities, polynomial growth, and all finite Gaussian moments. -/
def GaussianMixedInput {ι : Type*} [Fintype ι] [DecidableEq ι] (r : ι → ℕ)
    (f : (ι → ℝ) → ℝ) : Prop :=
  ∀ w, WithinMixedOrder r w → ∃ g, HasPartialsAlong w f g ∧ Continuous g ∧
    WeakPartialIdentity w f g ∧ PolynomialGrowth g ∧
    ∀ p : ℝ≥0∞, p ≠ ⊤ → MemLp g p (gaussianProduct ι)

/-- A boundary factor in GKLS Theorems 2–3. `ws` lists its phase derivatives. -/
noncomputable def boundaryFactor {ι : Type*} [DecidableEq ι]
    (φ θ : (Option ι → ℝ) → ℝ) (ψ : (ι → ℝ) → ℝ)
    (w : List (Option ι)) (ws : List (List (Option ι))) (b c : ℕ)
    (y : ι → ℝ) : ℝ :=
  let x := fiberPoint (ψ y) y
  partialsAlong w θ x * (ws.map (fun u => partialsAlong u φ x)).prod /
    (partialsAlong [none] φ x) ^ b * iteratedDeriv c (gaussianPDFReal 0 1) (ψ y)

/-- The factorial-normalized truncated power of degree `n+1`. -/
noncomputable def truncatedPower (n : ℕ) (t : ℝ) : ℝ :=
  (max 0 t) ^ (n + 1) / ((n + 1).factorial : ℝ)

/-- The phase used in the counterexample. -/
noncomputable def sharpPhase {ι : Type*} [Fintype ι] (q : ℕ) (x : Option ι → ℝ) : ℝ :=
  x none + truncatedPower q (x none) - ∑ i : ι, x (some i)

end Cubature
