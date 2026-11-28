import Theory.Setup

open MeasureTheory
open scoped ContDiff

namespace Cubature

/-- A one-sided power with derivative `sqrt (max 0 t)`. -/
noncomputable def fractionalRamp (t : ℝ) : ℝ :=
  (2 / 3 : ℝ) * (max 0 t) ^ (3 / 2 : ℝ)

/-- The desired inverse phase profile. The second ramp bounds the slope at infinity. -/
noncomputable def inverseSplice (u : ℝ) : ℝ :=
  u + fractionalRamp u - fractionalRamp (u - 1)

def phaseFromAxis {ι : Type*} [Fintype ι] (A : ℝ → ℝ) (x : Option ι → ℝ) : ℝ :=
  A (x none) - ∑ i : ι, x (some i)

def pairCoordinates (y : ℝ × ℝ) (i : Fin 2) : ℝ := if i = 0 then y.1 else y.2

/-- The mixed derivative `∂₁∂₂` in the distributional sense, for ordinary Lebesgue measure. -/
def WeakMixedSecond (P G : ℝ × ℝ → ℝ) : Prop :=
  ∀ v : ℝ × ℝ → ℝ, ContDiff ℝ ∞ v → HasCompactSupport v →
    (∫ y, G y * v y) =
      ∫ y, P y * lineDeriv ℝ (fun z => lineDeriv ℝ v z (1, 0)) y (0, 1)

end Cubature
