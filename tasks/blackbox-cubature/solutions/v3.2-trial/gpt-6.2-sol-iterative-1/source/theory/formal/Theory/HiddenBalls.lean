import Theory.GaussianDilation
import Mathlib.Data.Fintype.Pi

open MeasureTheory Set
open scoped BigOperators

namespace Cubature

abbrev Signs (d : ℕ) := Fin d → Bool

def signCenter {d : ℕ} (s : Signs d) : Point d :=
  fun i ↦ if s i then 1 else -1

/-- Euclidean radius one half, with the boundary assigned value zero. -/
def signBall {d : ℕ} (s : Signs d) : Set (Point d) :=
  {x | (∑ i, (x i-signCenter s i)^2) < (1/4 : ℝ)}

noncomputable def hiddenBallPayoff {d : ℕ} (s : Signs d) (a : ℝ) : Point d → ℝ :=
  (signBall s).indicator fun _ ↦ a

/-- Transcript of a deterministic value-query strategy; earlier query locations are
themselves determined by the earlier values. Variable stopping times can be padded. -/
def transcript {d : ℕ} (query : List ℝ → Point d) (f : Point d → ℝ) : ℕ → List ℝ
  | 0 => []
  | n+1 => let ys := transcript query f n
           ys ++ [f (query ys)]

/-- The second component selects a jump of height one or two. -/
noncomputable def packedPayoff {d : ℕ} (input : Signs d × Bool) : Point d → ℝ :=
  hiddenBallPayoff input.1 (if input.2 then 2 else 1)

def badBallEstimate {d : ℕ} (n : ℕ) (query : List ℝ → Point d) (decode : List ℝ → ℝ)
    (input : Signs d × Bool) : Prop :=
  let f := packedPayoff input
  let p := ∫ x, f x ∂(gamma d)
  p/3 ≤ |decode (transcript query f n)-p|

noncomputable def badSeedCount {d m : ℕ} (n : ℕ) (query : Fin m → List ℝ → Point d)
    (decode : Fin m → List ℝ → ℝ) (input : Signs d × Bool) : ℕ := by
  classical
  exact (Finset.univ.filter (fun j ↦ badBallEstimate n (query j) (decode j) input)).card

end Cubature
