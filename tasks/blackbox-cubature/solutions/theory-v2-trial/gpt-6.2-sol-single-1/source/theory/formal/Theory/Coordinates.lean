import Theory.Setup
import Mathlib.Analysis.Calculus.Deriv.Shift
import Mathlib.Analysis.Calculus.LineDeriv.IntegrationByParts
import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Data.Multiset.Replicate

/-! Classical and weak coordinate derivatives; reduction of sum ridges to one dimension. -/

open MeasureTheory
open scoped ContDiff

namespace Cubature

section Coordinates

variable {ι : Type*} [DecidableEq ι]
variable {f g h : (ι → ℝ) → ℝ}

lemma hasCoordinatePartial_const (i : ι) (c : ℝ) :
    HasCoordinatePartial i (fun _ => c) (fun _ => 0) := by
  intro x
  exact hasDerivAt_const 0 c

lemma HasCoordinatePartial.sub {i : ι} {f₁ f₂ g₁ g₂ : (ι → ℝ) → ℝ}
    (hf : HasCoordinatePartial i f₁ g₁) (hg : HasCoordinatePartial i f₂ g₂) :
    HasCoordinatePartial i (fun x => f₁ x - f₂ x) (fun x => g₁ x - g₂ x) := by
  intro x
  exact (hf x : HasDerivAt _ _ _).sub (hg x)

lemma HasCoordinatePartial.eq_lineDeriv {i : ι} (hf : HasCoordinatePartial i f g) :
    (fun x => lineDeriv ℝ f x (Pi.single i 1)) = g :=
  funext fun x => (hf x).lineDeriv

lemma HasPartialsAlong.eq_partialsAlong {w : List ι} (hf : HasPartialsAlong w f g) :
    partialsAlong w f = g := by
  induction w generalizing f g with
  | nil => exact hf
  | cons i w ih =>
    obtain ⟨h, h₁, h₂⟩ := hf
    rw [partialsAlong, h₁.eq_lineDeriv]
    exact ih h₂

lemma partialsAlong_append (u w : List ι) (f : (ι → ℝ) → ℝ) :
    partialsAlong (u ++ w) f = partialsAlong w (partialsAlong u f) := by
  induction u generalizing f with
  | nil => rfl
  | cons i u ih => exact ih _

/-- All intermediate derivatives, including the initial function, are continuous. -/
def ContinuousPartialChain : List ι → ((ι → ℝ) → ℝ) → ((ι → ℝ) → ℝ) → Prop
  | [], f, g => Continuous f ∧ f = g
  | i :: w, f, g => Continuous f ∧
      ∃ h, HasCoordinatePartial i f h ∧ ContinuousPartialChain w h g

lemma ContinuousPartialChain.continuous_start {w : List ι}
    (hf : ContinuousPartialChain w f g) : Continuous f := by
  cases w <;> exact hf.1

lemma ContinuousPartialChain.continuous_end {w : List ι}
    (hf : ContinuousPartialChain w f g) : Continuous g := by
  induction w generalizing f g with
  | nil =>
    obtain ⟨hc, he⟩ := hf
    exact he ▸ hc
  | cons i w ih =>
    obtain ⟨_, _, _, hc⟩ := hf
    exact ih hc

lemma ContinuousPartialChain.partials {w : List ι} (hf : ContinuousPartialChain w f g) :
    HasPartialsAlong w f g := by
  induction w generalizing f g with
  | nil => exact hf.2
  | cons i w ih =>
    obtain ⟨_, h, hh, hc⟩ := hf
    exact ⟨h, hh, ih hc⟩

lemma ContinuousPartialChain.const (w : List ι) (c : ℝ) :
    ContinuousPartialChain w (fun _ => c) (fun _ => if w = [] then c else 0) := by
  induction w generalizing c with
  | nil => exact ⟨continuous_const, rfl⟩
  | cons i w ih =>
    refine ⟨continuous_const, (fun _ => 0), hasCoordinatePartial_const i c, ?_⟩
    simpa using ih 0

lemma ContinuousPartialChain.sub {w : List ι} {f₁ f₂ g₁ g₂ : (ι → ℝ) → ℝ}
    (hf : ContinuousPartialChain w f₁ g₁) (hg : ContinuousPartialChain w f₂ g₂) :
    ContinuousPartialChain w (fun x => f₁ x - f₂ x) (fun x => g₁ x - g₂ x) := by
  induction w generalizing f₁ f₂ g₁ g₂ with
  | nil =>
    obtain ⟨hf, rfl⟩ := hf
    obtain ⟨hg, rfl⟩ := hg
    exact ⟨hf.sub hg, rfl⟩
  | cons i w ih =>
    obtain ⟨hc₁, u, hu, h₁⟩ := hf
    obtain ⟨hc₂, v, hv, h₂⟩ := hg
    exact ⟨hc₁.sub hc₂, fun x => u x - v x, hu.sub hv, ih h₁ h₂⟩

end Coordinates

section Weak

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

private lemma smooth_compact_partials (w : List ι) {v : (ι → ℝ) → ℝ}
    (hv : ContDiff ℝ ∞ v) (hc : HasCompactSupport v) :
    ContDiff ℝ ∞ (partialsAlong w v) ∧ HasCompactSupport (partialsAlong w v) := by
  induction w generalizing v with
  | nil => exact ⟨hv, hc⟩
  | cons i w ih =>
    rw [partialsAlong]
    simp only [(hv.differentiable (by simp) _).lineDeriv_eq_fderiv]
    exact ih ((hv.fderiv_right (m := ∞) (by simp)).clm_apply contDiff_const)
      (hc.fderiv_apply ℝ _)

private lemma coordinate_ibp {i : ι} {f g v : (ι → ℝ) → ℝ} (hf : Continuous f)
    (hg : Continuous g) (hd : HasCoordinatePartial i f g)
    (hv : ContDiff ℝ ∞ v) (hs : HasCompactSupport v) :
    (∫ x, g x * v x) = -(∫ x, f x * lineDeriv ℝ v x (Pi.single i 1)) := by
  have hdv : Continuous (fun x => lineDeriv ℝ v x (Pi.single i 1)) :=
    (smooth_compact_partials [i] hv hs).1.continuous
  have hcv : HasCompactSupport (fun x => lineDeriv ℝ v x (Pi.single i 1)) :=
    (smooth_compact_partials [i] hv hs).2
  have hgvi : Integrable (fun x => g x * v x) :=
    (hg.mul hv.continuous).integrable_of_hasCompactSupport hs.mul_left
  have hfvi : Integrable (fun x => f x * v x) :=
    (hf.mul hv.continuous).integrable_of_hasCompactSupport hs.mul_left
  have hfdvi : Integrable (fun x => f x * lineDeriv ℝ v x (Pi.single i 1)) :=
    (hf.mul hdv).integrable_of_hasCompactSupport hcv.mul_left
  have he := integral_bilinear_hasLineDerivAt_right_eq_neg_left_of_integrable
    (B := ContinuousLinearMap.mul ℝ ℝ) hgvi hfdvi hfvi
    (fun x _ => hd x)
    (fun x _ => ((hv.differentiable (by simp) x).hasFDerivAt.hasLineDerivAt
      (Pi.single i 1)).lineDifferentiableAt.hasLineDerivAt)
  simpa only [ContinuousLinearMap.mul_apply', neg_neg] using congrArg Neg.neg he.symm

lemma ContinuousPartialChain.weak {w : List ι} {f g : (ι → ℝ) → ℝ}
    (hf : ContinuousPartialChain w f g) : WeakPartialIdentity w f g := by
  intro v hv hc
  induction w generalizing f g with
  | nil =>
    obtain ⟨_, rfl⟩ := hf
    simp [partialsAlong]
  | cons i w ih =>
    obtain ⟨hf, h, hd, hs⟩ := hf
    have htest := smooth_compact_partials w.reverse hv hc
    have hg := hs.continuous_start
    rw [ih hs, List.length_cons, List.reverse_cons, partialsAlong_append]
    rw [coordinate_ibp hf hg hd htest.1 htest.2]
    simp [partialsAlong, pow_succ]

end Weak

section SumRidge

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

lemma sum_single_axis (i : ι) (x : ι → ℝ) (t : ℝ) :
    (∑ j, (x + t • (Pi.single i 1 : ι → ℝ)) j) = (∑ j, x j) + t := by
  have heq : t • (Pi.single i 1 : ι → ℝ) = Pi.single i t := by
    ext j
    by_cases h : j = i <;> simp [h]
  rw [heq]
  simp [Finset.sum_add_distrib, Pi.single_apply]

lemma hasCoordinatePartial_sumRidge (u u' : ℝ → ℝ) (i : ι)
    (hu : ∀ x, HasDerivAt u (u' x) x) :
    HasCoordinatePartial i (fun x : ι → ℝ => u (∑ j, x j))
      (fun x => u' (∑ j, x j)) := by
  intro x
  have hd : HasDerivAt u (u' (∑ j, x j)) ((∑ j, x j) + 0) := by simpa using hu (∑ j, x j)
  change HasDerivAt (fun a : ℝ => u (∑ j, (x + a • (Pi.single i 1 : ι → ℝ)) j)) _ 0
  simpa only [sum_single_axis] using hd.comp_const_add (∑ j, x j) 0

private lemma of_hasCoordinatePartial_sumRidge (u : ℝ → ℝ) (i : ι)
    {g : (ι → ℝ) → ℝ}
    (hg : HasCoordinatePartial i (fun x : ι → ℝ => u (∑ j, x j)) g) :
    Differentiable ℝ u ∧ g = fun x => deriv u (∑ j, x j) := by
  have hd (x : ι → ℝ) : HasDerivAt u (g x) (∑ j, x j) := by
    have hh : HasDerivAt (fun a : ℝ => u ((∑ j, x j) + a)) (g x) 0 := by
      have he : HasDerivAt (fun a : ℝ => u (∑ j, (x + a • (Pi.single i 1 : ι → ℝ)) j)) (g x) 0 := hg x
      simpa only [sum_single_axis] using he
    have hh' : HasDerivAt (fun a : ℝ => u ((∑ j, x j) + a)) (g x)
        ((∑ j, x j) + -(∑ j, x j)) := by simpa using hh
    simpa only [add_left_comm (∑ j, x j), add_neg_cancel, add_zero] using
      hh'.comp_add_const (∑ j, x j) (-(∑ j, x j))
  refine ⟨fun t => ?_, funext fun x => (hd x).deriv.symm⟩
  simpa [Pi.single_apply] using (hd (Pi.single i t)).differentiableAt

lemma sumRidge_partials (w : List ι) (u : ℝ → ℝ)
    (hu : ∀ k < w.length, Differentiable ℝ (iteratedDeriv k u)) :
    HasPartialsAlong w (fun x : ι → ℝ => u (∑ j, x j))
      (fun x => iteratedDeriv w.length u (∑ j, x j)) := by
  induction w generalizing u with
  | nil => simp [HasPartialsAlong]
  | cons i w ih =>
    have h₀ : Differentiable ℝ u := by simpa using hu 0 (by simp)
    refine ⟨(fun x => deriv u (∑ j, x j)),
      hasCoordinatePartial_sumRidge u (deriv u) i (fun x => (h₀ x).hasDerivAt), ?_⟩
    rw [List.length_cons, iteratedDeriv_succ']
    exact ih _ fun k hk => by
      rw [← iteratedDeriv_succ']
      exact hu _ (by simpa using Nat.succ_lt_succ hk)

lemma of_sumRidge_partials (w : List ι) (u : ℝ → ℝ) {g : (ι → ℝ) → ℝ}
    (hg : HasPartialsAlong w (fun x : ι → ℝ => u (∑ j, x j)) g) :
    (∀ k < w.length, Differentiable ℝ (iteratedDeriv k u)) ∧
    g = fun x => iteratedDeriv w.length u (∑ j, x j) := by
  induction w generalizing u g with
  | nil => exact ⟨by simp, by simpa using hg.symm⟩
  | cons i w ih =>
    obtain ⟨u', hi, hw⟩ := hg
    obtain ⟨hd, rfl⟩ := of_hasCoordinatePartial_sumRidge u i hi
    obtain ⟨hd', hg'⟩ := ih _ hw
    refine ⟨?_, ?_⟩
    · intro k hk
      cases k with
      | zero => simpa using hd
      | succ k => rw [iteratedDeriv_succ']; exact hd' k (by simpa using hk)
    · simpa only [List.length_cons, iteratedDeriv_succ'] using hg'

lemma contDiff_of_sumRidge_partials (w : List ι) (hw : 0 < w.length) (u : ℝ → ℝ)
    {g : (ι → ℝ) → ℝ} (hg : HasPartialsAlong w (fun x : ι → ℝ => u (∑ j, x j)) g)
    (hc : Continuous g) : ContDiff ℝ (w.length : ℕ) u := by
  obtain ⟨hd, rfl⟩ := of_sumRidge_partials w u hg
  have htop : Continuous (iteratedDeriv w.length u) := by
    have hn : w ≠ [] := by rintro rfl; simp at hw
    obtain ⟨i, w', rfl⟩ := List.exists_cons_of_ne_nil hn
    have hin : Continuous (fun t : ℝ => (Pi.single i t : ι → ℝ)) := by
      apply continuous_pi
      intro j
      by_cases h : j = i
      · subst j
        have he : (fun t : ℝ => (Pi.single i t : ι → ℝ) i) = id := by
          funext t
          simp
        rw [he]
        exact continuous_id
      · simpa [h] using (continuous_const : Continuous (fun _ : ℝ => (0 : ℝ)))
    simpa [Function.comp_def, Pi.single_apply] using hc.comp hin
  exact contDiff_nat_iff_iteratedDeriv.mpr
    ⟨fun k hk => (lt_or_eq_of_le hk).elim (fun h => (hd k h).continuous) (fun h => h ▸ htop),
      hd⟩

lemma exists_word_within_of_le_sum (r : ι → ℕ) (k : ℕ) (hk : k ≤ ∑ i, r i) :
    ∃ w : List ι, w.length = k ∧ WithinMixedOrder r w := by
  let M : Multiset ι := ∑ i, Multiset.replicate (r i) i
  have hM : M.toList.length = ∑ i, r i := by
    simp [M, Multiset.card_sum]
  refine ⟨M.toList.take k, by simp [hM, min_eq_left hk], ?_⟩
  intro i
  have hcount : M.toList.count i = r i := by
    rw [← Multiset.coe_count, Multiset.coe_toList]
    simp [M, Multiset.count_sum', Multiset.count_replicate]
  exact (List.Sublist.count_le i (List.take_sublist k M.toList)).trans hcount.le

end SumRidge

end Cubature
