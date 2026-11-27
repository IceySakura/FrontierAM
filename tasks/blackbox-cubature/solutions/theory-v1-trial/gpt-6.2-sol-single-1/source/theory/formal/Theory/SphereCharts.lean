import Theory.GaussianPolar
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-! Cubical charts of the sphere, including ties between adjacent faces. -/

open MeasureTheory Set Metric WithLp
open scoped unitInterval

namespace Cubature

noncomputable section

abbrev Faces (p : ℕ) := Fin (p+1) × Bool
abbrev Cube (p : ℕ) := Fin p → I

def faceSign (b : Bool) : ℝ := if b then 1 else -1

@[simp] theorem abs_faceSign (b : Bool) : |faceSign b| = 1 := by cases b <;> norm_num [faceSign]

@[simp] theorem faceSign_sq (b : Bool) : (faceSign b)^2 = 1 := by cases b <;> norm_num [faceSign]

/-- The cube face vector with pivot equal to `+1` or `-1`. -/
def faceVector {p : ℕ} (f : Faces p) (t : Cube p) : GaussianSpace (p+1) :=
  toLp 2 (@Fin.insertNth p (fun _ => ℝ) f.1 (faceSign f.2) (fun j => 2 * (t j).val - 1))

theorem faceVector_ne_zero {p : ℕ} (f : Faces p) (t : Cube p) : faceVector f t ≠ 0 := by
  intro he
  have hh := congrArg (fun z : GaussianSpace (p+1) => z f.1) he
  have hn : faceSign f.2 ≠ 0 := by cases f.2 <;> norm_num [faceSign]
  exact hn (by simpa [faceVector] using hh)

/-- Radial projection of a cubical face to the Euclidean sphere. -/
def faceDirection {p : ℕ} (f : Faces p) (t : Cube p) : Direction (p+1) :=
  ⟨‖faceVector f t‖⁻¹ • faceVector f t, by
    have hp := norm_pos_iff.mpr (faceVector_ne_zero f t)
    rw [mem_sphere_zero_iff_norm, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hp.le),
      inv_mul_cancel₀ hp.ne']⟩

def faceActive {p : ℕ} (f : Faces p) (x : Direction (p+1)) : Prop :=
  0 < faceSign f.2 * x.val f.1 ∧ ∀ j, |x.val j| ≤ faceSign f.2 * x.val f.1

theorem measurableSet_faceActive {p : ℕ} (f : Faces p) :
    MeasurableSet {x : Direction (p+1) | faceActive f x} := by
  have hm (j : Fin (p+1)) : Measurable (fun x : Direction (p+1) => x.val j) := by fun_prop
  have hm' := (hm f.1).const_mul (faceSign f.2)
  have hh : {x : Direction (p+1) | faceActive f x} =
      {x : Direction (p+1) | 0 < faceSign f.2 * x.val f.1} ∩
      ⋂ i, {x : Direction (p+1) | |x.val i| ≤ faceSign f.2 * x.val f.1} := by
    ext x
    simp [faceActive]
  rw [hh]
  exact (measurableSet_lt measurable_const hm').inter
    (MeasurableSet.iInter fun i => measurableSet_le (continuous_abs.measurable.comp (hm i)) hm')

theorem exists_faceActive {p : ℕ} (x : Direction (p+1)) : ∃ f, faceActive f x := by
  obtain ⟨i, _, hi⟩ := Finset.exists_max_image Finset.univ (fun j => |x.val j|)
    (Finset.univ_nonempty : (Finset.univ : Finset (Fin (p+1))).Nonempty)
  have hh (j) : |x.val j| ≤ |x.val i| := hi j (Finset.mem_univ _)
  have hpos : 0 < |x.val i| := by
    by_contra! hn
    have hz : x.val = 0 := by
      ext j
      exact abs_nonpos_iff.mp ((hh j).trans hn)
    simpa [hz] using mem_sphere_zero_iff_norm.mp x.2
  by_cases hs : 0 ≤ x.val i
  · refine ⟨(i, true), ?_⟩
    simpa [faceActive, faceSign, abs_of_nonneg hs] using And.intro hpos hh
  · refine ⟨(i, false), ?_⟩
    simpa [faceActive, faceSign, abs_of_neg (lt_of_not_ge hs)] using And.intro hpos hh

def faceParamRaw {p : ℕ} (f : Faces p) (x : Direction (p+1)) (j : Fin p) : ℝ :=
  (x.val (f.1.succAbove j) / (faceSign f.2 * x.val f.1) + 1) / 2

/- The clipping is only used to give a total coordinate map off the chart. -/
def faceParam {p : ℕ} (f : Faces p) (x : Direction (p+1)) : Cube p :=
  fun j => ⟨max 0 (min 1 (faceParamRaw f x j)), ⟨le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩⟩

theorem measurable_faceParam_coord {p : ℕ} (f : Faces p) (j : Fin p) :
    Measurable (fun x : Direction (p+1) => (faceParam f x j : ℝ)) := by
  change Measurable (fun x : Direction (p+1) => max 0 (min 1 (faceParamRaw f x j)))
  unfold faceParamRaw
  fun_prop

theorem faceParam_coord_of_active {p : ℕ} {f : Faces p} {x : Direction (p+1)}
    (ha : faceActive f x) (j : Fin p) : (faceParam f x j : ℝ) = faceParamRaw f x j := by
  have h := abs_le.mp (ha.2 (f.1.succAbove j))
  have h₁ : -1 ≤ x.val (f.1.succAbove j) / (faceSign f.2 * x.val f.1) :=
    (le_div_iff₀ ha.1).mpr (by simpa using h.1)
  have h₂ : x.val (f.1.succAbove j) / (faceSign f.2 * x.val f.1) ≤ 1 :=
    (div_le_iff₀ ha.1).mpr (by simpa using h.2)
  have h₀ : 0 ≤ faceParamRaw f x j := by unfold faceParamRaw; linarith
  have h₁' : faceParamRaw f x j ≤ 1 := by unfold faceParamRaw; linarith
  change max 0 (min 1 (faceParamRaw f x j)) = _
  rw [min_eq_right h₁', max_eq_right h₀]

theorem faceDirection_faceParam {p : ℕ} {f : Faces p} {x : Direction (p+1)}
    (ha : faceActive f x) : faceDirection f (faceParam f x) = x := by
  let M := faceSign f.2 * x.val f.1
  have hM : 0 < M := ha.1
  have hv : faceVector f (faceParam f x) = M⁻¹ • x.val := by
    apply PiLp.ext
    refine (Fin.forall_iff_succAbove f.1).2 ⟨?_, fun i => ?_⟩
    · simp only [faceVector, PiLp.toLp_apply, Fin.insertNth_apply_same, PiLp.smul_apply,
        smul_eq_mul, M]
      have h₀ : faceSign f.2 * x.val f.1 ≠ 0 := hM.ne'
      rw [eq_inv_mul_iff_mul_eq₀ h₀]
      calc
        (faceSign f.2 * x.val f.1) * faceSign f.2 = faceSign f.2 ^ 2 * x.val f.1 := by ring
        _ = x.val f.1 := by simp
    · simp only [faceVector, PiLp.toLp_apply, Fin.insertNth_apply_succAbove, PiLp.smul_apply,
        smul_eq_mul, faceParam_coord_of_active ha, faceParamRaw, M]
      ring
  apply Subtype.ext
  change ‖faceVector f (faceParam f x)‖⁻¹ • faceVector f (faceParam f x) = x.val
  rw [hv, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hM.le),
    mem_sphere_zero_iff_norm.mp x.2, mul_one, inv_inv, smul_smul, mul_inv_cancel₀ hM.ne', one_smul]

def faceFlag {p : ℕ} (f : Faces p) (x : Direction (p+1)) : ℝ := by
  classical
  exact if faceActive f x then 1 else 0

def faceWeight {p : ℕ} (f : Faces p) (x : Direction (p+1)) : ℝ :=
  faceFlag f x / (∑ g : Faces p, faceFlag g x)

theorem faceFlag_nonneg {p : ℕ} (f : Faces p) (x : Direction (p+1)) : 0 ≤ faceFlag f x := by
  unfold faceFlag
  split <;> norm_num

theorem measurable_faceFlag {p : ℕ} (f : Faces p) : Measurable (faceFlag f) :=
  by
    classical
    exact Measurable.ite (measurableSet_faceActive f) measurable_const measurable_const

theorem measurable_faceWeight {p : ℕ} (f : Faces p) : Measurable (faceWeight f) :=
  (measurable_faceFlag f).div (Finset.measurable_sum _ fun g _ => measurable_faceFlag g)

theorem faceWeight_nonneg {p : ℕ} (f : Faces p) (x : Direction (p+1)) : 0 ≤ faceWeight f x :=
  div_nonneg (faceFlag_nonneg _ _) (Finset.sum_nonneg fun _ _ => faceFlag_nonneg _ _)

theorem faceWeight_eq_zero {p : ℕ} {f : Faces p} {x : Direction (p+1)} (h : ¬faceActive f x) :
    faceWeight f x = 0 := by simp [faceWeight, faceFlag, h]

theorem sum_faceWeight {p : ℕ} (x : Direction (p+1)) : ∑ f : Faces p, faceWeight f x = 1 := by
  obtain ⟨f, hf⟩ := exists_faceActive x
  have h₁ : 1 ≤ ∑ g : Faces p, faceFlag g x := by
    have h := Finset.single_le_sum (f := fun g : Faces p => faceFlag g x)
      (fun g _ => faceFlag_nonneg g x) (Finset.mem_univ f)
    simpa [faceFlag, hf] using h
  simp only [faceWeight, ← Finset.sum_div]
  exact div_self (by linarith)

/-- A uniform order-`s` affine Taylor remainder on the specified cubical charts.
In particular, `s = 2` is satisfied by a chartwise `C^{1,1}` radius, and
`s = 1` is satisfied by a chartwise Lipschitz radius. -/
def ChartRemainder {p : ℕ} (s : ℕ) (A : ℝ) (R : Direction (p+1) → ℝ) : Prop :=
  ∀ (f : Faces p) (x : Cube p), ∃ β : Fin p → ℝ, ∀ (y : Cube p) (δ : ℝ), 0 ≤ δ →
    (∀ j, |(y j : ℝ) - (x j : ℝ)| ≤ δ) →
    |R (faceDirection f y) - R (faceDirection f x) -
      ∑ j, β j * ((y j : ℝ) - (x j : ℝ))| ≤ A * δ ^ s

end

end Cubature
