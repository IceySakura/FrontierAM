import Mathlib

/-!
# Common definitions

Preintegration (Griebel–Kuo–Sloan; Gilbert–Kuo–Sloan, arXiv:2112.11621) of a jump or kink
integrand along one real variable `t`.  The remaining variables are collected in a parameter
`y : Y` (for the problem of `THEORY.md`, `Y = ℝ^{d-1}`).  Following Gilbert–Kuo–Sloan,
`ind(s) = 1` if `s > 0` and `ind(s) = 0` otherwise.
-/

open MeasureTheory

namespace Cubature

variable {Y : Type*}

/-- Preintegrated jump integrand `(P f)(y) = ∫ θ(t, y) · ind(φ(t, y)) dν(t)`, i.e. the integral of
`θ(·, y)` over the set `{t | φ(t, y) > 0}` with respect to the measure `ν` on `ℝ`
(for the problem of `THEORY.md`, `ν` is the standard normal distribution). -/
noncomputable def preintJump (ν : Measure ℝ) (φ θ : ℝ → Y → ℝ) (y : Y) : ℝ :=
  ∫ t, {s : ℝ | 0 < φ s y}.indicator (fun s => θ s y) t ∂ν

/-- Preintegrated kink integrand `(P f)(y) = ∫ max(φ(t, y), 0) dν(t)`. -/
noncomputable def preintKink (ν : Measure ℝ) (φ : ℝ → Y → ℝ) (y : Y) : ℝ :=
  ∫ t, max (φ t y) 0 ∂ν

/-- The `m ^ s` half-open cells of side `1 / m` of the unit cube `[0, 1)^s`, indexed by
`k : Fin s → Fin m`: `cubeCell s m k = ∏ⱼ [k j / m, (k j + 1) / m)`.  Jittered (stratified)
sampling draws one uniform point in each cell, independently. -/
def cubeCell (s m : ℕ) (k : Fin s → Fin m) : Set (Fin s → ℝ) :=
  Set.pi Set.univ fun j => Set.Ico ((k j : ℝ) / m) (((k j : ℝ) + 1) / m)

end Cubature
