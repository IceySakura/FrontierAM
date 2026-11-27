import Theory.GaussianDilation
import Mathlib.Tactic.FunProp

/-!
Black-box bisection on inverse distances. The trace includes exactly the
points at which the algorithm reads a membership bit.
-/

open Set
open scoped BigOperators

namespace Cubature

noncomputable section

structure SectionSearch (α : Type) where
  lower : ℝ
  upper : ℝ
  queries : List α

def bisect {α : Type} (oracle : α → Bool) (point : ℝ → α) :
    ℕ → ℝ → ℝ → SectionSearch α
  | 0, lo, hi => ⟨lo, hi, []⟩
  | n + 1, lo, hi =>
    let t := (lo + hi) / 2
    let p := point t
    let recResult := if oracle p then bisect oracle point n lo t
      else bisect oracle point n t hi
    ⟨recResult.lower, recResult.upper, p :: recResult.queries⟩

lemma bisect_length {α : Type} (oracle : α → Bool) (point : ℝ → α)
    (n : ℕ) (lo hi : ℝ) : (bisect oracle point n lo hi).queries.length = n := by
  induction n generalizing lo hi with
  | zero => rfl
  | succ n ih =>
    simp only [bisect, List.length_cons]
    split <;> simp [ih]

lemma bisect_spec {α : Type} (oracle : α → Bool) (point : ℝ → α)
    (target : ℝ) (horacle : ∀ t, 0 < t →
      (oracle (point t) = true → target ≤ t) ∧
      (oracle (point t) = false → t ≤ target))
    (n : ℕ) (lo hi : ℝ) (hlo : 0 < lo) (h1 : lo ≤ target) (h2 : target ≤ hi) :
    let r := bisect oracle point n lo hi
    lo ≤ r.lower ∧ r.lower ≤ target ∧ target ≤ r.upper ∧ r.upper ≤ hi ∧
      r.upper - r.lower = (hi - lo) / 2 ^ n ∧
      ∀ x ∈ r.queries, ∃ t ∈ Icc lo hi, x = point t := by
  induction n generalizing lo hi with
  | zero =>
    simpa [bisect] using And.intro h1 h2
  | succ n ih =>
    let t := (lo + hi) / 2
    have ht : t ∈ Icc lo hi := ⟨by dsimp [t]; linarith, by dsimp [t]; linarith⟩
    have ht0 : 0 < t := hlo.trans_le ht.1
    have hhalf₁ : t - lo = (hi - lo) / 2 := by dsimp [t]; ring
    have hhalf₂ : hi - t = (hi - lo) / 2 := by dsimp [t]; ring
    have hwidth : (hi - lo) / 2 / (2 : ℝ) ^ n = (hi - lo) / 2 ^ (n+1) := by
      simp only [pow_succ, div_eq_mul_inv, mul_inv_rev, mul_assoc]
    have hor := horacle t ht0
    by_cases ha : oracle (point t) = true
    · have ⟨hl, hl', hu', hu, hw, htrace⟩ := ih lo t hlo h1 (hor.1 ha)
      simp only [bisect, ← show t = (lo + hi) / 2 from rfl, ha, if_true]
      refine ⟨hl, hl', hu', hu.trans ht.2, ?_, ?_⟩
      · rw [hw, hhalf₁, hwidth]
      · intro x hx
        rcases List.mem_cons.mp hx with rfl | hx
        · exact ⟨t, ht, rfl⟩
        · obtain ⟨y, hy, rfl⟩ := htrace x hx
          exact ⟨y, ⟨hy.1, hy.2.trans ht.2⟩, rfl⟩
    · have hb : oracle (point t) = false := by
        cases h : oracle (point t) <;> simp_all
      have ⟨hl, hl', hu', hu, hw, htrace⟩ := ih t hi ht0 (hor.2 hb) h2
      simp only [bisect, ← show t = (lo + hi) / 2 from rfl, hb, Bool.false_eq_true, if_false]
      refine ⟨ht.1.trans hl, hl', hu', hu, ?_, ?_⟩
      · rw [hw, hhalf₂, hwidth]
      · intro x hx
        rcases List.mem_cons.mp hx with rfl | hx
        · exact ⟨t, ht, rfl⟩
        · obtain ⟨y, hy, rfl⟩ := htrace x hx
          exact ⟨y, ⟨ht.1.trans hy.1, hy.2⟩, rfl⟩

def axis {d : ℕ} (i : Fin d) : Vec d := fun j => if j = i then 1 else 0

def sectionDirection {d : ℕ} (i j : Fin d) : Vec d :=
  if i = j then axis i else axis i + axis j

def sectionPoint {d : ℕ} (anchor dir : Vec d) (t : ℝ) : Vec d :=
  anchor + t⁻¹ • dir

def quadBilin {d : ℕ} (A : Fin d → Fin d → ℝ) (x y : Vec d) : ℝ :=
  ∑ i, ∑ j, A i j * x i * y j

def quadPart {d : ℕ} (A : Fin d → Fin d → ℝ) (x : Vec d) : ℝ :=
  quadBilin A x x

def linearPart {d : ℕ} (b x : Vec d) : ℝ :=
  ∑ i, b i * x i

def quadratic {d : ℕ} (A : Fin d → Fin d → ℝ) (b x : Vec d) : ℝ :=
  quadPart A x + linearPart b x

def body {d : ℕ} (a : Vec d) (A : Fin d → Fin d → ℝ) (b : Vec d) : Set (Vec d) :=
  {z | quadratic A b (z - a) ≤ 1}

def shiftedBox {d : ℕ} (a : Vec d) (radius : ℝ) : Set (Vec d) :=
  {z | ∀ i, |z i - a i| ≤ radius}

lemma qb_axis {d : ℕ} (A : Fin d → Fin d → ℝ) (i j : Fin d) :
    quadBilin A (axis i) (axis j) = A i j := by
  simp [quadBilin, axis, mul_ite, ite_mul, Finset.sum_ite_eq']

lemma qb_add_left {d : ℕ} (A : Fin d → Fin d → ℝ) (x y z : Vec d) :
    quadBilin A (x+y) z = quadBilin A x z + quadBilin A y z := by
  simp only [quadBilin, Pi.add_apply, mul_add, add_mul, Finset.sum_add_distrib]

lemma qb_add_right {d : ℕ} (A : Fin d → Fin d → ℝ) (x y z : Vec d) :
    quadBilin A x (y+z) = quadBilin A x y + quadBilin A x z := by
  simp only [quadBilin, Pi.add_apply, mul_add, Finset.sum_add_distrib]

lemma quadPart_smul {d : ℕ} (A : Fin d → Fin d → ℝ) (t : ℝ) (x : Vec d) :
    quadPart A (t • x) = t ^ 2 * quadPart A x := by
  simp only [quadPart, quadBilin, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => by ring))

lemma linearPart_smul {d : ℕ} (b : Vec d) (t : ℝ) (x : Vec d) :
    linearPart b (t • x) = t * linearPart b x := by
  simp only [linearPart, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => by ring)

lemma section_diag {d : ℕ} (A : Fin d → Fin d → ℝ) (i : Fin d) :
    quadPart A (sectionDirection i i) = A i i := by
  simp [sectionDirection, quadPart, qb_axis]

lemma section_pair {d : ℕ} (A : Fin d → Fin d → ℝ) (i j : Fin d)
    (hne : i ≠ j) (hsym : A j i = A i j) :
    quadPart A (sectionDirection i j) = A i i + A j j + 2 * A i j := by
  simp only [quadPart, sectionDirection, if_neg hne, qb_add_left, qb_add_right, qb_axis, hsym]
  ring

lemma section_linear_diag {d : ℕ} (b : Vec d) (i : Fin d) :
    linearPart b (sectionDirection i i) = b i := by
  simp [sectionDirection, linearPart, axis, mul_ite, Finset.sum_ite_eq']

def positiveInverse {d : ℕ} (A : Fin d → Fin d → ℝ) (b dir : Vec d) : ℝ :=
  (Real.sqrt (linearPart b dir ^ 2 + 4 * quadPart A dir) + linearPart b dir) / 2

def negativeInverse {d : ℕ} (A : Fin d → Fin d → ℝ) (b dir : Vec d) : ℝ :=
  (Real.sqrt (linearPart b dir ^ 2 + 4 * quadPart A dir) - linearPart b dir) / 2

lemma inverse_product {d : ℕ} (A : Fin d → Fin d → ℝ) (b dir : Vec d)
    (h : 0 ≤ quadPart A dir) :
    positiveInverse A b dir * negativeInverse A b dir = quadPart A dir := by
  unfold positiveInverse negativeInverse
  have he := Real.sq_sqrt (show 0 ≤ linearPart b dir ^ 2 + 4 * quadPart A dir by positivity)
  nlinarith only [he]

lemma inverse_difference {d : ℕ} (A : Fin d → Fin d → ℝ) (b dir : Vec d) :
    positiveInverse A b dir - negativeInverse A b dir = linearPart b dir := by
  unfold positiveInverse negativeInverse
  ring

def runSection {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi : ℝ) (dir : Vec d) : SectionSearch (Vec d) :=
  bisect oracle (sectionPoint a dir) steps lo hi

def foundQ {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi : ℝ) (i j : Fin d) : ℝ :=
  (runSection oracle a steps lo hi (sectionDirection i j)).lower *
    (runSection oracle a steps lo hi (-sectionDirection i j)).lower

def foundA {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi : ℝ) (i j : Fin d) : ℝ :=
  if i = j then foundQ oracle a steps lo hi i i else
    (foundQ oracle a steps lo hi (max i j) (min i j) -
      foundQ oracle a steps lo hi i i - foundQ oracle a steps lo hi j j) / 2

def foundB {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi : ℝ) (i : Fin d) : ℝ :=
  (runSection oracle a steps lo hi (sectionDirection i i)).lower -
    (runSection oracle a steps lo hi (-sectionDirection i i)).lower

def queryHistory {d : ℕ} (oracle : Vec d → Bool) (a : Vec d)
    (steps : ℕ) (lo hi : ℝ) : List (Vec d) :=
  (Finset.univ.toList : List (Fin d × Fin d)).flatMap fun (i,j) =>
    if j ≤ i then
      (runSection oracle a steps lo hi (sectionDirection i j)).queries ++
      (runSection oracle a steps lo hi (-sectionDirection i j)).queries
    else []

def sectionAccuracy (steps : ℕ) (lo hi : ℝ) : ℝ :=
  (hi - lo) / 2 ^ steps

def valueAccuracy (d steps : ℕ) (lo hi radius : ℝ) : ℝ :=
  (3 * hi * (d : ℝ) ^ 2 * radius ^ 2 + 2 * d * radius) * sectionAccuracy steps lo hi

def proxySet {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi radius level : ℝ) : Set (Vec d) :=
  shiftedBox a radius ∩ {z |
    quadratic (foundA oracle a steps lo hi) (foundB oracle a steps lo hi) (z - a) ≤ level}

def massInterval {d : ℕ} (oracle : Vec d → Bool) (a : Vec d) (steps : ℕ)
    (lo hi radius : ℝ) : ℝ × ℝ :=
  let e := valueAccuracy d steps lo hi radius
  ((gaussian d).real (proxySet oracle a steps lo hi radius (1-e)),
   (gaussian d).real (proxySet oracle a steps lo hi radius (1+e)))

end

end Cubature
