import Mathlib

/-!
# Shared definitions

Classical partial derivatives, the classical mixed-smoothness class `C^r_mix`, and the
preintegration operator `P_j` with respect to the standard Gaussian density, in the notation of
Griewank, Kuo, Leövey, Sloan, *High dimensional integration of kinks and jumps — smoothing by
preintegration*, J. Comput. Appl. Math. 344 (2018) 259–274 (arXiv:1712.00920), Section 4.1.

Points of `ℝ^d` are functions `Fin d → ℝ`. For `j : Fin (n + 1)`, the point
`Fin.insertNth j t y` is GKLS's `(x_j, x_{-j}) = (t, y)`: the value `t` is inserted at
position `j` into `y : Fin n → ℝ`, and `y k` is the coordinate `x_{j.succAbove k}`.
-/

open MeasureTheory ProbabilityTheory

namespace GKLS

variable {n : ℕ}

/-- The classical partial derivative `∂F/∂x_k` at `x`: the ordinary derivative of
`t ↦ F (x_1, …, x_{k-1}, t, x_{k+1}, …)` at `t = x_k`. (Lean's `deriv` returns `0` where the
derivative does not exist; existence is tracked separately by `IterPartialExists`.) -/
noncomputable def partialDeriv (k : Fin n) (F : (Fin n → ℝ) → ℝ) (x : Fin n → ℝ) : ℝ :=
  deriv (fun t => F (Function.update x k t)) (x k)

/-- The iterated partial derivative `D_{k₁} D_{k₂} ⋯ D_{kₘ} F` for `l = [k₁, …, kₘ]`
(the last direction of the list is differentiated first). -/
noncomputable def iterPartial (l : List (Fin n)) (F : (Fin n → ℝ) → ℝ) : (Fin n → ℝ) → ℝ :=
  l.foldr partialDeriv F

/-- Every classical partial derivative taken in forming `iterPartial l F` exists at every point
of `ℝ^n`: for `l = k :: l'`, the derivatives along `l'` exist everywhere and the function
`iterPartial l' F` has a partial derivative in direction `k` at every point. -/
def IterPartialExists : List (Fin n) → ((Fin n → ℝ) → ℝ) → Prop
  | [], _ => True
  | k :: l, F => IterPartialExists l F ∧
      ∀ x : Fin n → ℝ, DifferentiableAt ℝ (fun t => iterPartial l F (Function.update x k t)) (x k)

/-- The classical space `C^r_mix(ℝ^n)` of GKLS (Section 4.1): every classical mixed partial
derivative `D^α F` with `α ≤ r` componentwise exists at every point and is continuous. This is
required for every order in which the `|α|` differentiations can be carried out: a list `l` of
directions with `l.count k ≤ r k` for every `k` encodes one such derivative and one order. -/
def ContDiffMixed (r : Fin n → ℕ) (F : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ l : List (Fin n), (∀ k, l.count k ≤ r k) → IterPartialExists l F ∧ Continuous (iterPartial l F)

/-- GKLS's indicator: `ind u = 1` if `u > 0` and `ind u = 0` otherwise. -/
noncomputable def ind (u : ℝ) : ℝ := if 0 < u then 1 else 0

/-- Preintegration along the coordinate `x_j` against the standard Gaussian density `ρ`:
`(P_j f)(y) = ∫_ℝ f(x_j, y) ρ(x_j) dx_j`, where `(x_j, y) = Fin.insertNth j x_j y`. -/
noncomputable def preintegrate (j : Fin (n + 1)) (f : (Fin (n + 1) → ℝ) → ℝ) (y : Fin n → ℝ) : ℝ :=
  ∫ t, f (Fin.insertNth j t y) ∂(gaussianReal 0 1)

/-- The jump `f(x) = ind(a x_j² + β(x_{-j}) x_j + κ(x_{-j}))` across a hypersurface that is quadratic
in the preintegration variable `x_j` (here `x_{-j} = fun k => x (j.succAbove k)`). For a quadric
`q(x) = xᵀAx + bᵀx + c`, `a = A_jj`, `β` is affine and `κ` quadratic in `x_{-j}`. -/
noncomputable def quadricJump (j : Fin (n + 1)) (a : ℝ) (β κ : (Fin n → ℝ) → ℝ)
    (x : Fin (n + 1) → ℝ) : ℝ :=
  ind (a * x j ^ 2 + β (fun k => x (j.succAbove k)) * x j + κ (fun k => x (j.succAbove k)))

end GKLS
