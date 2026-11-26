import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# Shared definitions: query algorithms and two classes of functions on a line

This module is imported by `Theory/Statements.lean` and `Theory/Proofs.lean`.

We study integration of `f : ℝ → ℝ` with respect to the standard Gaussian measure
`ProbabilityTheory.gaussianReal 0 1` from function values only. This is the per-line subproblem of the
cubature problem: along a line the non-smooth surfaces are crossed at unknown points, and the algorithm must
locate the crossings from function values.

* `AdaptiveAlg n`: deterministic adaptive algorithms using `n` function values;
  `AdaptiveAlg.info` is the vector of the `n` values that the algorithm observes.
* `NonadaptiveAlg n`: deterministic nonadaptive algorithms (fixed nodes, arbitrary output map).
* `StepClass` (the class `J`): one crossing, unit step functions (and the constants `0`, `1`).
* `IntervalClass` (the class `P`): two crossings, indicator functions of bounded intervals.

The estimate of an algorithm `A` for `f` is `A.output (A.info f)` and its error is
`|A.output (A.info f) - ∫ t, f t ∂(gaussianReal 0 1)|`; the statements spell this out.
-/

namespace Cubature

/-- A deterministic **adaptive** algorithm that uses `n` values of a function `f : ℝ → ℝ`.

* `node k y` is the point at which the algorithm computes its `(k+1)`-st function value, as a function of the
  `k` values `y 0, …, y (k-1)` observed before. Only `node 0, …, node (n-1)` are ever used.
* `output y` is the estimate, an arbitrary function of the `n` observed values `y`. -/
structure AdaptiveAlg (n : ℕ) where
  /-- The `(k+1)`-st evaluation point, chosen from the `k` values observed so far. -/
  node : (k : ℕ) → (Fin k → ℝ) → ℝ
  /-- The estimate, computed from the `n` observed values. -/
  output : (Fin n → ℝ) → ℝ

namespace AdaptiveAlg

variable {n : ℕ}

/-- `A.observe f k : Fin k → ℝ` is the vector of the first `k` values that `A` observes on `f`, defined by
recursion on `k`: the value with index `k` is `f (A.node k (A.observe f k))`, i.e. `f` at the node that `A`
chooses from the previous `k` values (`Fin.snoc v x` appends the entry `x` to the vector `v`). -/
def observe (A : AdaptiveAlg n) (f : ℝ → ℝ) : (k : ℕ) → Fin k → ℝ
  | 0 => Fin.elim0
  | k + 1 => Fin.snoc (A.observe f k) (f (A.node k (A.observe f k)))

/-- The information that `A` computes on `f`: its `n` observed function values. -/
def info (A : AdaptiveAlg n) (f : ℝ → ℝ) : Fin n → ℝ :=
  A.observe f n

/-- The first `k` observed values do not change when more values are observed. -/
theorem observe_castLE (A : AdaptiveAlg n) (f : ℝ → ℝ) :
    ∀ (d k : ℕ) (j : Fin k), A.observe f (k + d) (Fin.castLE (Nat.le_add_right k d) j) = A.observe f k j
  | 0, _, _ => rfl
  | d + 1, k, j => by
    have h := observe_castLE A f d k j
    have e : (Fin.castLE (Nat.le_add_right k (d + 1)) j : Fin (k + d + 1)) =
        Fin.castSucc (Fin.castLE (Nat.le_add_right k d) j) := Fin.ext rfl
    show A.observe f (k + d + 1) _ = _
    rw [e]
    simp only [observe, Fin.snoc_castSucc]
    exact h

/-- The first `k` observed values do not change when more values are observed (general form). -/
theorem observe_castLE' (A : AdaptiveAlg n) (f : ℝ → ℝ) {k N : ℕ} (h : k ≤ N) (j : Fin k) :
    A.observe f N (Fin.castLE h j) = A.observe f k j := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  exact observe_castLE A f d k j

/-- Sanity check of the definition of `info`: the value with index `i` is `f` evaluated at the node that `A`
chooses from the values with indices `0, …, i-1`. -/
theorem info_apply (A : AdaptiveAlg n) (f : ℝ → ℝ) (i : Fin n) :
    A.info f i = f (A.node i (fun j : Fin i => A.info f (Fin.castLE i.isLt.le j))) := by
  have e : i = Fin.castLE i.isLt (Fin.last i) := Fin.ext rfl
  have h2 : (fun j : Fin i => A.info f (Fin.castLE i.isLt.le j)) = A.observe f i := by
    funext j
    exact observe_castLE' A f _ j
  rw [h2]
  conv_lhs => rw [e]
  unfold info
  rw [observe_castLE' A f i.isLt (Fin.last i)]
  simp only [observe, Fin.snoc_last]

end AdaptiveAlg

/-- A deterministic **nonadaptive** algorithm that uses `n` function values: fixed nodes
`nodes 0, …, nodes (n-1)` and an arbitrary (possibly nonlinear) map `output` from the values to the estimate. -/
structure NonadaptiveAlg (n : ℕ) where
  /-- The fixed evaluation points. -/
  nodes : Fin n → ℝ
  /-- The estimate, computed from the `n` observed values. -/
  output : (Fin n → ℝ) → ℝ

/-- The information of a nonadaptive algorithm: the values of `f` at its nodes. -/
def NonadaptiveAlg.info {n : ℕ} (A : NonadaptiveAlg n) (f : ℝ → ℝ) : Fin n → ℝ :=
  fun i => f (A.nodes i)

/-- The class `J` of **one-crossing** functions on a line (unit steps): either `f` jumps from `0` to `1` at some
point `s` (`f t = 0` for `t < s` and `f t = 1` for `t > s`; the value `f s` on the surface is arbitrary), or `f` is
one of the constants `0`, `1` (the line does not cross the surface). -/
def StepClass : Set (ℝ → ℝ) :=
  {f | f = 0 ∨ f = 1 ∨ ∃ s : ℝ, (∀ t, t < s → f t = 0) ∧ (∀ t, s < t → f t = 1)}

/-- The class `P` of **two-crossing** functions on a line (a closed surface crossed twice): indicator functions of
bounded intervals, `f t = 1` for `a < t < b` and `f t = 0` for `t < a` or `t > b`, where `a ≤ b` are real; the
values `f a`, `f b` on the surface are arbitrary. With `a = b` (and `f a = 0`) this includes `f = 0`, a line that
misses the surface. -/
def IntervalClass : Set (ℝ → ℝ) :=
  {f | ∃ a b : ℝ, a ≤ b ∧ (∀ t, a < t → t < b → f t = 1) ∧ (∀ t, t < a ∨ b < t → f t = 0)}

end Cubature
