import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Data.List.Basic

namespace Cubature

def secondDiff (f : ℝ → ℝ) (x h : ℝ) : ℝ :=
  f (x+h) - 2*f x + f (x-h)

def kinkNodes (c h : ℝ) : List ℝ := [c-2*h, c-h, c, c+h, c+2*h]

/-- The comparisons in this expression use a cached table of the five nodes. -/
noncomputable def kinkCenter (f : ℝ → ℝ) (c h : ℝ) : ℝ :=
  let L := |secondDiff f (c-h) h|
  let M := |secondDiff f c h|
  let U := |secondDiff f (c+h) h|
  if M ≤ L then (if U ≤ L then c-h else c+h)
  else (if U ≤ M then c else c+h)

/-- The result stores the last center and the full (non-reusing) schedule of point queries. -/
noncomputable def kinkSearch (f : ℝ → ℝ) (c h : ℝ) : ℕ → ℝ × List ℝ
  | 0 => (c, [])
  | n+1 =>
      let ans := kinkSearch f (kinkCenter f c h) (h/2) n
      (ans.1, kinkNodes c h ++ ans.2)

end Cubature
