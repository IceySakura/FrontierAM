import Theory.Splines
import Theory.SlopeMeasure

/-! A grid of disjoint angular bumps with explicit smoothness and mass bounds. -/

open Set MeasureTheory WithLp
open scoped unitInterval

namespace Cubature

noncomputable section

abbrev PatchIndex (p m : ℕ) := Fin p → Fin m

def slopeArg {m : ℕ} (i : Fin m) (t : ℝ) : ℝ :=
  3 * (m : ℝ) * (t + 1/2) - 3 * (i.val : ℝ)

def patch {p m : ℕ} (i : PatchIndex p m) (u : Direction (p+1)) : ℝ :=
  if 0 < u.val 0 then ∏ j, spline (slopeArg (i j) (chartSlope u j)) else 0

theorem measurable_patch {p m : ℕ} (i : PatchIndex p m) : Measurable (patch i) := by
  have hm : Measurable (fun u : Direction (p+1) => u.val 0) := by fun_prop
  unfold patch
  refine Measurable.ite (measurableSet_lt measurable_const hm) ?_ measurable_const
  exact Finset.measurable_prod _ fun j _ =>
    continuous_spline.measurable.comp
      (((measurable_chartSlope j).add_const (1/2)).const_mul _ |>.sub_const _)

theorem patch_mem {p m : ℕ} (i : PatchIndex p m) (u : Direction (p+1)) :
    patch i u ∈ Icc 0 1 := by
  unfold patch
  split
  · refine ⟨Finset.prod_nonneg (fun _ _ => (spline_mem _).1), ?_⟩
    exact (Finset.prod_le_prod₀ (fun _ _ => (spline_mem _).1)
      (fun _ _ => (spline_mem _).2)).trans_eq Finset.prod_const_one
  · exact ⟨le_rfl, zero_le_one⟩

private theorem scaledSpline_support {m : ℕ} (i : Fin m) {t : ℝ}
    (h : spline (slopeArg i t) ≠ 0) :
    (i.val : ℝ) < (m : ℝ) * (t+1/2) ∧ (m : ℝ) * (t+1/2) < (i.val : ℝ) + 1 := by
  have hh := spline_support h
  unfold slopeArg at hh
  exact ⟨by linarith [hh.1], by linarith [hh.2]⟩

theorem patch_support {p m : ℕ} {i : PatchIndex p m} {u : Direction (p+1)}
    (h : patch i u ≠ 0) :
    0 < u.val 0 ∧ ∀ j, (i j).val < (m : ℝ) * (chartSlope u j + 1/2) ∧
      (m : ℝ) * (chartSlope u j + 1/2) < ((i j).val : ℝ) + 1 := by
  unfold patch at h
  split at h
  case isTrue he =>
    refine ⟨he, fun j => scaledSpline_support (i j) ?_⟩
    intro hh
    exact h (Finset.prod_eq_zero (Finset.mem_univ j) hh)
  case isFalse he => exact (h rfl).elim

theorem patch_support_unique {p m : ℕ} {i i' : PatchIndex p m} {u : Direction (p+1)}
    (h : patch i u ≠ 0) (h' : patch i' u ≠ 0) : i = i' := by
  funext j
  apply Fin.ext
  have H := (patch_support h).2 j
  have H' := (patch_support h').2 j
  by_contra hn
  rcases lt_or_gt_of_ne hn with hn | hn
  · have hp : ((i j).val : ℝ) + 1 ≤ (i' j).val := by exact_mod_cast Nat.succ_le_of_lt hn
    linarith [H.2,H'.1]
  · have hp : ((i' j).val : ℝ) + 1 ≤ (i j).val := by exact_mod_cast Nat.succ_le_of_lt hn
    linarith [H'.2,H.1]

def coreLo {m : ℕ} (i : Fin m) : ℝ := ((i.val : ℝ) + 1/3) / (m : ℝ) - 1/2
def coreHi {m : ℕ} (i : Fin m) : ℝ := ((i.val : ℝ) + 2/3) / (m : ℝ) - 1/2

def patchCore {p m : ℕ} (i : PatchIndex p m) : Set (Direction (p+1)) :=
  slopeBox (fun j => coreLo (i j)) (fun j => coreHi (i j))

theorem patchCore_measurable {p m : ℕ} (i : PatchIndex p m) :
    MeasurableSet (patchCore i) := measurableSet_slopeBox _ _

theorem patchCore_lower {p m : ℕ} (hm : 0 < m) (i : PatchIndex p m) {u : Direction (p+1)}
    (hu : u ∈ patchCore i) : (2/3 : ℝ)^p ≤ patch i u := by
  have hu' : 0 < u.val 0 ∧ ∀ j, chartSlope u j ∈ Icc (coreLo (i j)) (coreHi (i j)) := hu
  have hm' : 0 < (m : ℝ) := by positivity
  unfold patch
  rw [ite_eq_left hu'.1, show (2/3 : ℝ)^p = ∏ _ : Fin p, (2/3 : ℝ) by
    simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin]]
  apply Finset.prod_le_prod₀
  · intro _ _; norm_num
  · intro j _
    apply spline_core
    have h := hu'.2 j
    have h₁ : ((i j).val : ℝ) + 1/3 ≤ (chartSlope u j + 1/2) * m := by
      apply (div_le_iff₀ hm').mp
      simpa only [coreLo, sub_le_iff_le_add] using h.1
    have h₂ : (chartSlope u j+1/2)*m ≤ ((i j).val:ℝ)+2/3 := by
      apply (le_div_iff₀ hm').mp
      simpa only [coreHi, le_sub_iff_add_le] using h.2
    exact ⟨by unfold slopeArg; linarith, by unfold slopeArg; linarith⟩

theorem patchCore_mass {p m : ℕ} (hm : 0 < m) (i : PatchIndex p m) :
    lineDensityLower * (lineDensityLower / (3*(m : ℝ)))^p ≤
      (angularLaw (p+1)).real (patchCore i) := by
  have hm' : (0:ℝ) < m := by positivity
  have hL (j : Fin p) : -(1/2:ℝ) ≤ coreLo (i j) := by
    unfold coreLo
    have h : 0 ≤ ((i j).val + (1/3 : ℝ)) / (m : ℝ) := by positivity
    linarith
  have hH (j : Fin p) : coreHi (i j) ≤ 1/2 := by
    have h₁ : ((i j).val : ℝ) + 1 ≤ m := by exact_mod_cast Nat.succ_le_of_lt (i j).isLt
    have h₂ : ((i j).val + (2/3 : ℝ)) / (m : ℝ) ≤ 1 :=
      (div_le_one hm').mpr (by linarith)
    unfold coreHi
    linarith
  have hδ (j : Fin p) : 1/(3*(m:ℝ)) ≤ coreHi (i j) - coreLo (i j) := by
    unfold coreHi coreLo
    apply le_of_eq
    field_simp
    ring
  unfold patchCore
  simpa only [mul_one_div] using slopeBox_mass_lower _ _ (by positivity) hL hH hδ

def primaryFace (p : ℕ) : Faces p := (0,true)

private theorem faceVector_at_pivot {p : ℕ} (f : Faces p) (x : Cube p) :
    faceVector f x f.1 = faceSign f.2 := by simp [faceVector]

private theorem faceVector_bound {p : ℕ} (f : Faces p) (x : Cube p) :
    ∀ i, |faceVector f x i| ≤ 1 := by
  rw [Fin.forall_iff_succAbove f.1]
  constructor
  · rw [faceVector_at_pivot, abs_faceSign]
  · intro j
    simp only [faceVector, PiLp.toLp_apply, Fin.insertNth_apply_succAbove, abs_le]
    exact ⟨by linarith [(x j).2.1], by linarith [(x j).2.2]⟩

private theorem ray_face_slope {p : ℕ} (f : Faces p) (x : Cube p) (j : Fin p) :
    chartSlope (faceDirection f x) j = faceVector f x j.succ / faceVector f x 0 := by
  change (‖faceVector f x‖⁻¹ • faceVector f x) j.succ /
    (‖faceVector f x‖⁻¹ • faceVector f x) 0 = _
  simp only [PiLp.smul_apply, smul_eq_mul]
  exact mul_div_mul_left _ _ (by have := norm_pos_iff.mpr (faceVector_ne_zero f x); positivity)

private theorem patch_on_primary {p m : ℕ} (i : PatchIndex p m) (x : Cube p) :
    patch i (faceDirection (primaryFace p) x) =
      ∏ j, spline (slopeArg (i j) (2*(x j).val-1)) := by
  have h₀ : faceVector (primaryFace p) x 0 = 1 := by
    simpa only [primaryFace, faceSign, ↓reduceIte] using faceVector_at_pivot (primaryFace p) x
  have hPos : 0 < (faceDirection (primaryFace p) x).val 0 := by
    change 0 < (‖faceVector (primaryFace p) x‖⁻¹ • faceVector (primaryFace p) x) 0
    rw [PiLp.smul_apply, smul_eq_mul, h₀, mul_one]
    have := norm_pos_iff.mpr (faceVector_ne_zero (primaryFace p) x)
    positivity
  unfold patch
  rw [ite_eq_left hPos]
  simp_rw [ray_face_slope, h₀, div_one]
  apply Finset.prod_congr rfl
  intro j _
  rw [← Fin.zero_succAbove j]
  simp only [faceVector, primaryFace, PiLp.toLp_apply, Fin.insertNth_apply_succAbove]

private theorem patch_off_primary {p m : ℕ} (hm : 0 < m) (i : PatchIndex p m)
    {f : Faces p} (hf : f ≠ primaryFace p) (x : Cube p) :
    patch i (faceDirection f x) = 0 := by
  by_contra hh
  have h := patch_support hh
  have H : 0 < ‖faceVector f x‖⁻¹ := by
    have := norm_pos_iff.mpr (faceVector_ne_zero f x)
    positivity
  have hp : 0 < faceVector f x 0 := by
    have h₀ := h.1
    change 0 < (‖faceVector f x‖⁻¹ • faceVector f x) 0 at h₀
    rw [PiLp.smul_apply, smul_eq_mul] at h₀
    exact pos_of_mul_pos_right h₀ H.le
  have hne : f.1 ≠ 0 := by
    intro he
    have ht := faceVector_at_pivot f x
    rw [he] at ht
    rw [ht] at hp
    have hg : f.2 = true := by
      cases hb : f.2
      · rw [hb, faceSign] at hp
        norm_num at hp
      · rfl
    exact hf (Prod.ext he hg)
  let j := f.1.pred hne
  have hj : j.succ = f.1 := Fin.succ_pred _ _
  have hx := h.2 j
  rw [ray_face_slope, hj, faceVector_at_pivot] at hx
  have hc : ((i j).val : ℝ) + 1 ≤ m := by exact_mod_cast Nat.succ_le_of_lt (i j).isLt
  have hmp : 0 < (m : ℝ) := by positivity
  have hleft : -1/2 < faceSign f.2 / faceVector f x 0 := by
    have hh' : 0 < ((m : ℝ) * (faceSign f.2 / faceVector f x 0 + 1/2)) :=
      (Nat.cast_nonneg _).trans_lt hx.1
    linarith [pos_of_mul_pos_right hh' hmp.le]
  have hright : faceSign f.2 / faceVector f x 0 < 1/2 := by
    have hc' : (m : ℝ) * (faceSign f.2 / faceVector f x 0 + 1/2) < m := hx.2.trans_le hc
    have := (mul_lt_mul_iff_of_pos_left hmp).mp (hc'.trans_eq (mul_one _).symm)
    linarith
  have hle : faceVector f x 0 ≤ 1 := (le_abs_self _).trans (faceVector_bound f x 0)
  have ha := (lt_div_iff₀ hp).mp hleft
  have hb := (div_lt_iff₀ hp).mp hright
  cases hs : f.2 <;> simp [faceSign, hs] at ha hb <;> linarith

def patchRemainderFactor (p : ℕ) : ℝ := 6 * (p : ℝ) + 64 * (p : ℝ)^2

theorem patch_affine {p m : ℕ} (hm : 0 < m) (i : PatchIndex p m) (f : Faces p) (x : Cube p) :
    ∃ β : Fin p → ℝ,
      (patch i (faceDirection f x) = 0 → ∀ j, β j = 0) ∧
      ∀ (y : Cube p) (δ : ℝ), 0 ≤ δ →
        (∀ j, |(y j : ℝ)-(x j : ℝ)| ≤ δ) →
        |patch i (faceDirection f y) - patch i (faceDirection f x) -
            ∑ j, β j * ((y j : ℝ)-(x j : ℝ))| ≤ patchRemainderFactor p * (6*(m : ℝ)*δ)^2 := by
  classical
  by_cases hf : f = primaryFace p
  · subst f
    let v (z : Cube p) (j : Fin p) := slopeArg (i j) (2*(z j).val-1)
    let a (j : Fin p) := spline (v x j)
    let c (j : Fin p) := (6*(m : ℝ))*splineSlope (v x j)
    refine ⟨fun j => c j * (∏ k ∈ Finset.univ.erase j, a k), ?_, ?_⟩
    · intro he j
      rw [patch_on_primary] at he
      have H : ∏ j, a j = 0 := he
      obtain ⟨k, hk, hkz⟩ := Finset.prod_eq_zero_iff.mp H
      by_cases heq : k = j
      · subst k
        simp [c, splineSlope_zero (show spline (v x j) = 0 from hkz)]
      · dsimp only
        rw [Finset.prod_eq_zero (Finset.mem_erase.mpr ⟨heq,hk⟩) hkz, mul_zero]
    · intro y δ hδ hxy
      let b (j : Fin p) := spline (v y j)
      have hm' : (0:ℝ) ≤ 6*(m:ℝ) := by positivity
      have hF (j : Fin p) : v y j - v x j = (6*(m:ℝ))*((y j : ℝ)-(x j:ℝ)) := by
        simp only [v, slopeArg]
        ring
      have hd (j : Fin p) : |v y j - v x j| ≤ 6*(m:ℝ)*δ := by
        rw [hF, abs_mul, abs_of_nonneg hm']
        exact mul_le_mul_of_nonneg_left (hxy j) hm'
      have hg (j : Fin p) :
          |b j - a j - c j * ((y j : ℝ) - (x j : ℝ))| ≤ 6 * (6*(m : ℝ)*δ)^2 := by
        have hrem := spline_remainder (v x j) (v y j)
        rw [show splineSlope (v x j) * (v y j - v x j) =
            c j * ((y j : ℝ)-(x j : ℝ)) by rw [hF]; unfold c; ring] at hrem
        exact hrem.trans (mul_le_mul_of_nonneg_left (sq_le_sq.mpr (by
          rw [abs_of_nonneg (mul_nonneg hm' hδ)]; exact hd j)) (by norm_num))
      have hs := (tensor_remainder a b (fun j => c j*((y j:ℝ)-(x j:ℝ)))
        (mul_nonneg hm' hδ) (by norm_num : (0:ℝ) ≤ 8) (by norm_num : (0:ℝ) ≤ 6)
        (fun j => spline_mem _) (fun j => spline_mem _)
        (fun j => (spline_lipschitz _ _).trans (mul_le_mul_of_nonneg_left (hd j) (by norm_num)))
        hg Finset.univ).2
      simp only [patch_on_primary, patchRemainderFactor, Finset.card_univ, Fintype.card_fin] at ⊢ hs
      have he (j : Fin p) :
        c j * (↑(y j) - ↑(x j)) * (∏ k ∈ Finset.univ.erase j, a k) =
        c j * (∏ k ∈ Finset.univ.erase j, a k) * (↑(y j) - ↑(x j)) := by ring
      simp_rw [he] at hs
      have hEq : (p:ℝ)*6 + (p:ℝ)^2*8^2 = 6*(p:ℝ)+64*(p:ℝ)^2 := by ring
      rw [hEq] at hs
      exact hs
  · refine ⟨fun _ => 0, fun _ _ => rfl, fun y δ hδ _ => ?_⟩
    rw [patch_off_primary hm i hf x, patch_off_primary hm i hf y]
    unfold patchRemainderFactor
    simp only [zero_mul, Finset.sum_const_zero, sub_zero, abs_zero]
    positivity

end

end Cubature
