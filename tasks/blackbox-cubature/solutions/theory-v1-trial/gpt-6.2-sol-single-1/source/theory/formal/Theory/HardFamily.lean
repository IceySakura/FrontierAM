import Theory.BumpPatch
import Theory.Nonadaptive

/-! Finite hard families of radial bodies. The level is common to all angular patches. -/

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal unitInterval

namespace Cubature

noncomputable section

def bumpHeight (m : ℕ) : ℝ := 1 / (m : ℝ)^2
def baseRadius {m : ℕ} (j : Fin (m^2)) : ℝ := 1 + (j.val : ℝ) * bumpHeight m

theorem bumpHeight_pos {m : ℕ} (hm : 0 < m) : 0 < bumpHeight m := by
  unfold bumpHeight
  positivity

theorem baseRadius_ge {m : ℕ} (j : Fin (m^2)) : 1 ≤ baseRadius j := by
  unfold baseRadius bumpHeight
  exact le_add_of_nonneg_right (by positivity)

theorem baseRadius_add_le {m : ℕ} (hm : 0 < m) (j : Fin (m^2)) :
    baseRadius j + bumpHeight m ≤ 2 := by
  have hj : (j.val : ℝ) + 1 ≤ (m : ℝ)^2 := by exact_mod_cast Nat.succ_le_of_lt j.isLt
  have h := mul_le_mul_of_nonneg_right hj (bumpHeight_pos hm).le
  have he : (m : ℝ)^2 * bumpHeight m = 1 := by
    unfold bumpHeight
    field_simp
  rw [he, add_mul, one_mul] at h
  unfold baseRadius
  linarith

def selectedBumps {p m : ℕ} (b : PatchIndex p m → Bool) (u : Direction (p+1)) : ℝ :=
  ∑ i, bitvalue (b i) * patch i u

def hardRadius {p m : ℕ} (j : Fin (m^2)) (b : PatchIndex p m → Bool)
    (u : Direction (p+1)) : ℝ := baseRadius j + bumpHeight m * selectedBumps b u

theorem measurable_hardRadius {p m : ℕ} (j : Fin (m^2)) (b : PatchIndex p m → Bool) :
    Measurable (hardRadius j b) := by
  unfold hardRadius selectedBumps
  exact measurable_const.add ((Finset.measurable_sum _ (fun i _ =>
    (measurable_patch i).const_mul _)).const_mul _)

theorem selectedBumps_single {p m : ℕ} (b : PatchIndex p m → Bool) {u : Direction (p+1)}
    {i : PatchIndex p m} (h : patch i u ≠ 0) : selectedBumps b u = bitvalue (b i) * patch i u := by
  classical
  apply Finset.sum_eq_single_of_mem i (Finset.mem_univ i)
  intro i' _ hi'
  have h' : patch i' u = 0 := by
    by_contra hn
    exact hi' (patch_support_unique hn h)
  rw [h', mul_zero]

theorem selectedBumps_zero {p m : ℕ} (b : PatchIndex p m → Bool) {u : Direction (p+1)}
    (h : ∀ i : PatchIndex p m, patch i u = 0) : selectedBumps b u = 0 := by
  simp [selectedBumps, h]

private theorem selectedBumps_mem {p m : ℕ} (b : PatchIndex p m → Bool) (u : Direction (p+1)) :
    selectedBumps b u ∈ Icc 0 1 := by
  by_cases h : ∃ i : PatchIndex p m, patch i u ≠ 0
  · obtain ⟨i,hi⟩ := h
    rw [selectedBumps_single b hi]
    cases hb : b i
    · simp [bitvalue, mem_Icc]
    · simpa only [bitvalue, ↓reduceIte, one_mul] using patch_mem i u
  · simp [selectedBumps_zero b (by simpa using h), mem_Icc]

theorem hardRadius_range {p m : ℕ} (hm : 0 < m) (j : Fin (m^2)) (b : PatchIndex p m → Bool)
    (u : Direction (p+1)) : hardRadius j b u ∈ Icc (baseRadius j) (baseRadius j + bumpHeight m) := by
  have hs := selectedBumps_mem b u
  unfold hardRadius
  constructor
  · exact le_add_of_nonneg_right (mul_nonneg (bumpHeight_pos hm).le hs.1)
  · exact add_le_add le_rfl (mul_le_of_le_one_right (bumpHeight_pos hm).le hs.2)

theorem hardRadius_mem {p m : ℕ} (hm : 0 < m) (j : Fin (m^2)) (b : PatchIndex p m → Bool)
    (u : Direction (p+1)) : hardRadius j b u ∈ Icc 1 2 :=
  ⟨(baseRadius_ge j).trans (hardRadius_range hm j b u).1,
    (hardRadius_range hm j b u).2.trans (baseRadius_add_le hm j)⟩

/-- One uniform chart-remainder bound for all grid sizes and all choices of bits. -/
def hardRemainderBound (p : ℕ) : ℝ := 1 + 72 * patchRemainderFactor p

theorem hardRemainderBound_ge (p : ℕ) : 1 ≤ hardRemainderBound p := by
  unfold hardRemainderBound patchRemainderFactor
  apply le_add_of_nonneg_right
  positivity

theorem hardRadius_remainder {p m : ℕ} (hm : 0 < m) (j : Fin (m^2)) (b : PatchIndex p m → Bool) :
    ChartRemainder 2 (hardRemainderBound p) (hardRadius j b) := by
  classical
  intro f x
  choose β hβ using fun i : PatchIndex p m => patch_affine hm i f x
  let H := bumpHeight m
  let w (i : PatchIndex p m) := bitvalue (b i)
  refine ⟨fun k => H * ∑ i, w i * β i k, ?_⟩
  intro y δ hδ hxy
  let sx := faceDirection f x
  let sy := faceDirection f y
  let dx (k : Fin p) : ℝ := y k - (x k : ℝ)
  let rem (i : PatchIndex p m) :=
    patch i sy - patch i sx - ∑ k, β i k * dx k
  let C := patchRemainderFactor p * (6*(m:ℝ)*δ)^2
  have hC : 0 ≤ C := by unfold C patchRemainderFactor; positivity
  have hsumlin : (∑ k, (H * ∑ i, w i * β i k) * dx k) =
      H * (∑ i, w i * (∑ k, β i k * dx k)) := by
    simp only [mul_assoc, ← Finset.mul_sum]
    congr 1
    simp_rw [Finset.sum_mul, mul_assoc]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum]
  have heq : hardRadius j b sy - hardRadius j b sx -
      (∑ k, (H * ∑ i, w i * β i k) * dx k) = H * ∑ i, w i * rem i := by
    rw [hsumlin]
    unfold hardRadius
    change baseRadius j + H * (∑ i, w i * patch i sy) -
      (baseRadius j + H * (∑ i, w i * patch i sx)) -
      H * (∑ i, w i * (∑ k, β i k * dx k)) = _
    simp only [rem, mul_sub, Finset.sum_sub_distrib]
    ring
  change |hardRadius j b sy - hardRadius j b sx -
    (∑ k, (H * ∑ i, w i * β i k) * dx k)| ≤ _
  rw [heq, abs_mul, abs_of_nonneg (bumpHeight_pos hm).le]
  let active (u : Direction (p+1)) := Finset.univ.filter (fun i : PatchIndex p m => patch i u ≠ 0)
  let S := active sx ∪ active sy
  have hc (u) : (active u).card ≤ 1 :=
    Finset.card_le_one.mpr (fun i hi i' hi' => patch_support_unique
      (Finset.mem_filter.mp hi).2 (Finset.mem_filter.mp hi').2)
  have hS : S.card ≤ 2 := (Finset.card_union_le _ _).trans (Nat.add_le_add (hc sx) (hc sy))
  have hrestrict : ∑ i, w i * rem i = ∑ i ∈ S, w i * rem i := by
    symm
    apply Finset.sum_subset (Finset.subset_univ _)
    intro i _ hi
    have hx : patch i sx = 0 := by
      by_contra hh
      exact hi (Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨Finset.mem_univ _,hh⟩))
    have hy : patch i sy = 0 := by
      by_contra hh
      exact hi (Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨Finset.mem_univ _,hh⟩))
    simp only [rem, hx, hy, (hβ i).1 hx, zero_mul, Finset.sum_const_zero, sub_zero, mul_zero]
  have hrem (i : PatchIndex p m) : |rem i| ≤ C := (hβ i).2 y δ hδ hxy
  have hw (i : PatchIndex p m) : |w i| ≤ 1 := by
    cases hb : b i <;> norm_num [w,bitvalue,hb]
  have hbound : |∑ i ∈ S, w i * rem i| ≤ 2 * C := by
    calc
      _ ≤ ∑ i ∈ S, |w i * rem i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _ ∈ S, C := Finset.sum_le_sum fun i _ => by
        rw [abs_mul]
        exact (mul_le_mul (hw i) (hrem i) (abs_nonneg _) zero_le_one).trans_eq (one_mul _)
      _ ≤ _ := by
        rw [Finset.sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast hS) hC
  rw [hrestrict]
  refine (mul_le_mul_of_nonneg_left hbound (bumpHeight_pos hm).le).trans ?_
  have he : H * (2*C) = (72 * patchRemainderFactor p) * δ^2 := by
    unfold H C bumpHeight
    have hm' : (m : ℝ) ≠ 0 := by positivity
    field_simp
    ring
  rw [he, hardRemainderBound, add_mul, one_mul]
  linarith [sq_nonneg δ]

private def locatedAt {p m : ℕ} (q : GaussianSpace (p+1))
    (a : Fin (m^2) × PatchIndex p m) : Prop :=
  ‖q‖ ∈ Ico (baseRadius a.1) (baseRadius a.1 + bumpHeight m) ∧
    patch a.2 (directionOf q) ≠ 0

private theorem locations_unique {p m : ℕ} (hm : 0 < m) {q : GaussianSpace (p+1)}
    {a b : Fin (m^2) × PatchIndex p m} (ha : locatedAt q a) (hb : locatedAt q b) : a = b := by
  refine Prod.ext ?_ (patch_support_unique ha.2 hb.2)
  have hs {i j : Fin (m^2)} (hij : i < j) :
      baseRadius i + bumpHeight m ≤ baseRadius j := by
    have ht : (i.val : ℝ) + 1 ≤ j.val := by exact_mod_cast Nat.succ_le_of_lt hij
    have h := mul_le_mul_of_nonneg_right ht (bumpHeight_pos hm).le
    unfold baseRadius
    linarith
  rcases lt_trichotomy a.1 b.1 with h | h | h
  · exact False.elim ((not_lt_of_ge ((hs h).trans hb.1.1)) ha.1.2)
  · exact h
  · exact False.elim ((not_lt_of_ge ((hs h).trans ha.1.1)) hb.1.2)

def hardSite {p m : ℕ} (q : GaussianSpace (p+1)) : Option (Fin (m^2) × PatchIndex p m) := by
  classical
  exact if h : ∃ a, locatedAt q a then some (Classical.choose h) else none

private theorem hardSite_of {p m : ℕ} (hm : 0 < m) {q : GaussianSpace (p+1)}
    {a : Fin (m^2) × PatchIndex p m} (h : locatedAt q a) : hardSite q = some a := by
  have he : ∃ a, locatedAt q a := ⟨a,h⟩
  unfold hardSite
  rw [dite_eq_left he, locations_unique hm (Classical.choose_spec he) h]

theorem hardOracle_flip {p m : ℕ} (hm : 0 < m) (j : Fin (m^2)) (i : PatchIndex p m)
    (b : PatchIndex p m → Bool) (q : GaussianSpace (p+1))
    (h : hardSite q ≠ some (j,i)) :
    starOracle (hardRadius j (flip i b)) q = starOracle (hardRadius j b) q := by
  by_cases hL : ‖q‖ < baseRadius j
  · have hm₁ := hL.trans_le (hardRadius_range hm j b (directionOf q)).1
    have hm₂ := hL.trans_le (hardRadius_range hm j (flip i b) (directionOf q)).1
    simp [starOracle, hm₁, hm₂]
  by_cases hH : baseRadius j + bumpHeight m ≤ ‖q‖
  · have hm₁ := not_lt_of_ge ((hardRadius_range hm j b (directionOf q)).2.trans hH)
    have hm₂ := not_lt_of_ge ((hardRadius_range hm j (flip i b) (directionOf q)).2.trans hH)
    simp [starOracle, hm₁, hm₂]
  have hz : patch i (directionOf q) = 0 := by
    by_contra hh
    exact h (hardSite_of hm ⟨⟨not_lt.mp hL, not_le.mp hH⟩,hh⟩)
  have he : selectedBumps (flip i b) (directionOf q) = selectedBumps b (directionOf q) := by
    unfold selectedBumps
    apply Finset.sum_congr rfl
    intro i' _
    by_cases hi : i' = i
    · subst i'; simp [hz]
    · rw [flip_off hi]
  unfold starOracle hardRadius
  exact decide_eq_decide.mpr (by rw [he])

end

end Cubature
